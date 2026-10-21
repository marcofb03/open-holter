"""Time one training step of each model on this CPU and the data pipeline per example.

    python measure_step.py                     # all threads, writes results/step_timing.{json,md}
    python measure_step.py --threads 48 --tag euler   # same inside a 48-core Euler job
    python measure_step.py --device cuda --tag gpu    # first 2 minutes on a rented GPU

A step = forward + backward + AdamW update on one batch (ecgfounder / clef:
64 windows of 10 s at 500 Hz; beat: 512 frames of 640 samples), median of
--reps after 2 warm-up steps, fp32 and bfloat16 autocast. The data pipeline
(h10.convert with augmentation + h10.augment + model preprocessing) is timed
per example on one thread, from a synthetic 250 Hz source segment.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import h10  # noqa: E402
import models  # noqa: E402
import train  # noqa: E402


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def step_time(name: str, bf16: bool, reps: int, device: str = "cpu") -> float:
    import torch

    net = models.build(name, pretrained=False).to(device)
    kind = models.KIND[name]
    b = train.BATCH[name]
    L = 5000 if kind == "win" else h10.BEAT
    x = torch.randn(b, 1, L, device=device)
    if kind == "win":
        y = torch.randint(0, 2, (b, models.N_WIN), device=device).float()
        lossf = torch.nn.BCEWithLogitsLoss()
    else:
        y = torch.randint(0, models.N_BEAT, (b,), device=device)
        lossf = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    ts = []
    for k in range(reps + 2):
        t = time.perf_counter()
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=bf16):
            out = net(x).float()
        loss = lossf(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device != "cpu":
            torch.cuda.synchronize()
        if k >= 2:
            ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def pipeline_time(name: str, n: int = 256) -> float:
    import torch

    torch.set_num_threads(1)
    kind = models.KIND[name]
    fs = 250.0
    L = int(round((h10.OUT_LEN[kind] / h10.FS + 2 * h10.MARGIN_S[kind]) * fs))
    rng = np.random.default_rng(0)
    seg = (rng.standard_normal((n, L)) * 300).astype(np.int16)
    y = np.zeros((n, models.N_WIN), np.int8) if kind == "win" else np.zeros(n, np.int8)
    t = time.perf_counter()
    x = h10.convert(seg, fs, kind, rng)
    x, y = h10.augment(x, y, kind, rng, None)
    models.preprocess(name, x)
    return (time.perf_counter() - t) / n


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="local")
    ap.add_argument("--models", nargs="+", default=["beat", "ecgfounder", "clef"])
    ap.add_argument("--device", default="cpu", help="cpu or cuda (rented GPU calibration)")
    args = ap.parse_args()
    res = {"cpu": cpu_name() if args.device == "cpu" else torch.cuda.get_device_name(), "threads": args.threads, "torch": torch.__version__, "models": {}}
    for name in args.models:
        r = {"batch": train.BATCH[name], "pipeline_ms_per_example_1thread": 1000 * pipeline_time(name)}
        torch.set_num_threads(args.threads)
        for bf16 in (False, True):
            s = step_time(name, bf16, args.reps, args.device)
            r["bf16" if bf16 else "fp32"] = {"s_per_step": s, "examples_per_s": r["batch"] / s}
        res["models"][name] = r
        print(name, json.dumps(r), file=sys.stderr, flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"step_timing_{args.tag}.json").write_text(json.dumps(res, indent=1))
    L = [f"CPU: {res['cpu']}, {args.threads} threads, torch {res['torch']}", "",
         "| model | batch | fp32 s/step | fp32 examples/s | bf16 s/step | bf16 examples/s | data pipeline ms/example (1 thread) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in res["models"].items():
        L.append(f"| {name} | {r['batch']} | {r['fp32']['s_per_step']:.2f} | {r['fp32']['examples_per_s']:.1f} | "
                 f"{r['bf16']['s_per_step']:.2f} | {r['bf16']['examples_per_s']:.1f} | {r['pipeline_ms_per_example_1thread']:.2f} |")
    (out / f"step_timing_{args.tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
