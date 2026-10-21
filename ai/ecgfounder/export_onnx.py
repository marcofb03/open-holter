"""Phone feasibility: export ECGFounder (1 lead) to ONNX, int8-quantise, compare, time.

    python export_onnx.py                      # export + quantise + parity + CPU benchmark
    python export_onnx.py --random-weights     # same on random weights (size/latency only)

Writes (git-ignored) weights/ecgfounder_1lead_{fp32,int8,int8_static}.onnx and
results/<tag>/phone.json + phone.md with model size, multiply-accumulates per
window, fp32 parity, int8 drift and CPU latency for batch 1 and batch 32 at 1
and all threads. int8 comes in two flavours: dynamic (weights only, activations
quantised at run time) and static QDQ (activations calibrated on 64 MIT-BIH DS1 windows).

The accuracy of the int8 model on the databases is measured separately with
    python eval.py --backend onnx --onnx weights/ecgfounder_1lead_int8_static.onnx --tag int8_static
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infer  # noqa: E402

WINDOWS_PER_DAY = 24 * 3600 // 10  # 8,640 non-overlapping 10 s windows


def count_macs(model) -> int:
    """Multiply-accumulates of one (1, 1, 5000) forward pass (convs + linears)."""
    import torch

    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        macs += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin_hook(m, inp, out):
        nonlocal macs
        macs += out.numel() * m.in_features

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    with torch.inference_mode():
        model(torch.zeros(1, 1, infer.WIN))
    for h in hooks:
        h.remove()
    return macs


class _StaticPads:
    """Export Net1D's pad-same layers in a form onnxruntime optimises correctly.

    net1d.py zero-pads, then max-pools. Once constant folding has resolved the
    pad sizes, onnxruntime fuses that Pad into MaxPool's own `pads`, which pad
    with -inf, not 0. From the first odd length on (625 samples, stage 3) the
    last pooled value then differs whenever it is negative: measured max
    |logit difference| 0.80, |Δp| up to 0.17 against torch. Here the max-pool
    pad is written as a Concat of zeros (not fusable), and the conv pads are
    frozen as constants (the input length is fixed at 5,000 samples). The
    maths is unchanged; the vendored file stays untouched.
    """

    def __enter__(self):
        import torch
        import torch.nn.functional as F
        from net1d import MyConv1dPadSame, MyMaxPool1dPadSame

        def conv_forward(m, x):
            n = int(x.shape[-1])
            out = (n + m.stride - 1) // m.stride
            p = max(0, (out - 1) * m.stride + m.kernel_size - n)
            return m.conv(F.pad(x, (p // 2, p - p // 2), "constant", 0))

        def pool_forward(m, x):
            p = max(0, m.kernel_size - 1)
            z = x[..., :1] * 0
            x = torch.cat([z] * (p // 2) + [x] + [z] * (p - p // 2), dim=-1)
            return m.max_pool(x)

        self.saved = (MyConv1dPadSame.forward, MyMaxPool1dPadSame.forward)
        self.classes = (MyConv1dPadSame, MyMaxPool1dPadSame)
        MyConv1dPadSame.forward, MyMaxPool1dPadSame.forward = conv_forward, pool_forward
        return self

    def __exit__(self, *exc):
        self.classes[0].forward, self.classes[1].forward = self.saved


def export(model, path: Path) -> None:
    import warnings

    import torch

    x = torch.randn(2, 1, infer.WIN)
    kw = dict(input_names=["ecg"], output_names=["logits"], opset_version=17,
              dynamic_axes={"ecg": {0: "batch"}, "logits": {0: "batch"}})
    with _StaticPads(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)  # int(x.shape[-1]) is intended
        try:
            torch.onnx.export(model, (x,), str(path), dynamo=False, **kw)
        except TypeError:  # older torch without the dynamo switch
            torch.onnx.export(model, (x,), str(path), **kw)


def quantise(src: Path, dst: Path, calib: np.ndarray | None = None) -> None:
    """Dynamic int8 (weights int8, activations quantised on the fly; ConvInteger),
    or static int8 QDQ (activations calibrated on `calib`; QLinearConv) if given."""
    import logging

    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    logging.getLogger().setLevel(logging.ERROR)  # one warning per non-float tensor otherwise
    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre))
    if calib is None:
        quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8)
    else:
        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{"ecg": calib[i:i + 1]} for i in range(len(calib))])

            def get_next(self):
                return next(self.it, None)

        quantize_static(str(pre), str(dst), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True)
    pre.unlink()


def bench(fn, x: np.ndarray, reps: int, warmup: int = 2) -> float:
    """Median seconds per call."""
    for _ in range(warmup):
        fn(x)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def sample_windows(n: int) -> tuple[np.ndarray, np.ndarray, str]:
    """(calibration, test) preprocessed windows, n each.

    Real H10-like windows when MIT-BIH is on disk: calibration from the DS1
    (development) records, test from DS2, so int8 calibration never sees the
    held-out data. Otherwise synthetic ECG-like spikes (size/latency only).
    """
    try:
        import wfdb

        import eval as ev

        out = []
        for recs in (sorted(ev.DS1), sorted(set(ev.records("mitdb", ev.REPO / "data" / "mitdb")) - ev.DS1)):
            per = -(-n // len(recs))
            ws = []
            for r in recs:
                rec = wfdb.rdrecord(str(ev.REPO / "data" / "mitdb" / r))
                ch = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
                x = infer.resample_to_500(ev.h10_like(rec.p_signal[:, ch], rec.fs), ev.H10_FS)
                st = infer.window_starts(len(x))
                st = st[np.linspace(0, len(st) - 1, per).astype(int)]  # spread over the record
                ws += [x[s:s + infer.WIN] for s in st]
            out.append(infer.preprocess_windows(np.stack(ws[:n])))
        return out[0], out[1], "MIT-BIH (calibration DS1, test DS2, H10-like)"
    except Exception:
        rng = np.random.default_rng(0)
        t = np.arange(infer.WIN) / infer.FS
        base = np.sin(2 * np.pi * 1.2 * t) ** 63  # sharp periodic spikes ~ 72 bpm
        w = infer.preprocess_windows(base[None] + 0.05 * rng.standard_normal((2 * n, infer.WIN)))
        return w[:n], w[n:], "synthetic"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=infer.DEFAULT_CKPT)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    import onnxruntime as ort
    import torch

    tag = args.tag or ("random_weights" if args.random_weights else "fp32")
    wdir = infer.WEIGHTS_DIR
    wdir.mkdir(exist_ok=True)
    stem = "random" if args.random_weights else "ecgfounder_1lead"
    f32, i8, q8 = (wdir / f"{stem}_{k}.onnx" for k in ("fp32", "int8", "int8_static"))
    if args.random_weights:
        torch.manual_seed(0)
    model = infer.load_torch_model(None if args.random_weights else args.ckpt, allow_random=args.random_weights)

    n_params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model)
    export(model, f32)
    calib, x, src = sample_windows(64)
    quantise(f32, i8)
    quantise(f32, q8, calib)
    with torch.inference_mode():
        ref = model(torch.from_numpy(x)).numpy()

    def ort_session(path: Path, threads: int):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    p_ref = sig(ref)
    focus = sorted({i for idx in infer.GROUP_IDX.values() for i in idx})
    parity = {"windows": src}
    for name, path in (("onnx fp32", f32), ("onnx int8 dynamic", i8), ("onnx int8 static", q8)):
        d = np.abs(sig(ort_session(path, 0).run(None, {"ecg": x})[0]) - p_ref)[:, focus]
        parity[name] = {"max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean())}

    n_cpu = os.cpu_count() or 1
    lat = []
    for threads in sorted({1, n_cpu}):
        torch.set_num_threads(threads)
        sess = {"onnx fp32": ort_session(f32, threads), "onnx int8 dynamic": ort_session(i8, threads),
                "onnx int8 static": ort_session(q8, threads)}
        for batch in (1, 32):
            xb = x[:batch]
            reps = args.reps if batch == 1 else max(3, args.reps // 3)
            with torch.inference_mode():
                t = bench(lambda a: model(torch.from_numpy(a)), xb, reps)
            lat.append({"runtime": "torch fp32", "threads": threads, "batch": batch, "ms_per_window": 1000 * t / batch})
            for name, s in sess.items():
                t = bench(lambda a, s=s: s.run(None, {"ecg": a}), xb, reps)
                lat.append({"runtime": name, "threads": threads, "batch": batch, "ms_per_window": 1000 * t / batch})
    w = np.stack([np.random.default_rng(1).standard_normal(infer.WIN)] * 32)
    pre_ms = 1000 * bench(infer.preprocess_windows, w, 5) / 32

    info = {
        "tag": tag, "params": n_params, "macs_per_window": macs,
        "size_mb": {"pth_checkpoint": (args.ckpt.stat().st_size / 1e6) if args.ckpt.exists() and not args.random_weights else None,
                    "onnx_fp32": f32.stat().st_size / 1e6, "onnx_int8_dynamic": i8.stat().st_size / 1e6,
                    "onnx_int8_static": q8.stat().st_size / 1e6},
        "parity": parity, "latency": lat, "preprocess_ms_per_window": pre_ms,
        "cpu": _cpu(), "threads_available": n_cpu,
        "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__},
    }
    out = HERE / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "phone.json").write_text(json.dumps(info, indent=1))
    md = phone_md(info)
    (out / "phone.md").write_text(md)
    print(md)


def phone_md(info: dict) -> str:
    sz = info["size_mb"]
    lines = [f"Parameters {info['params'] / 1e6:.1f} M, {info['macs_per_window'] / 1e9:.2f} GMAC per 10 s window"
             + (f"; checkpoint file {sz['pth_checkpoint']:.0f} MB" if sz["pth_checkpoint"] else "") + ".", "",
             f"| model | size MB | max abs Δp vs torch fp32 | mean abs Δp |", "|---|---:|---:|---:|"]
    for name, key in (("onnx fp32", "onnx_fp32"), ("onnx int8 dynamic", "onnx_int8_dynamic"),
                      ("onnx int8 static", "onnx_int8_static")):
        d = info["parity"][name]
        lines.append(f"| {name} | {sz[key]:.0f} | {d['max_abs_dp']:.4f} | {d['mean_abs_dp']:.4f} |")
    lines += ["", f"Δp: probability difference on the focus labels, {info['parity']['windows']} windows.", "",
              f"CPU latency ({info['cpu']}; torch {info['versions']['torch']}, onnxruntime {info['versions']['onnxruntime']}):", "",
              "| runtime | threads | batch | ms / window | core-min per day (8,640 windows) |",
              "|---|---:|---:|---:|---:|"]
    for r in info["latency"]:
        lines.append(f"| {r['runtime']} | {r['threads']} | {r['batch']} | {r['ms_per_window']:.1f} | "
                     f"{r['ms_per_window'] * WINDOWS_PER_DAY / 60000 * r['threads']:.1f} |")
    lines += ["", f"Preprocessing (filters + z-score): {info['preprocess_ms_per_window']:.1f} ms / window.",
              "core-min per day = ms/window x 8,640 x threads (an upper bound on CPU time).", ""]
    return "\n".join(lines)


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or platform.machine()


if __name__ == "__main__":
    main()
