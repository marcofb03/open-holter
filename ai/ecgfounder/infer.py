"""Window-level arrhythmia probabilities from the ECGFounder 1-lead model.

ECGFounder (Li et al., NEJM AI 2025; MIT licence, see third_party/) was
trained on 10 s, 500 Hz hospital ECGs (lead I for the 1-lead checkpoint).
This wrapper takes any single-lead signal at any sampling rate, resamples it
to 500 Hz, cuts sliding 10 s windows (5 s step by default), applies
ECGFounder's own preprocessing to each window exactly as in its dataset.py
(50 Hz notch, 0.67-40 Hz Butterworth band-pass, 0.4 s median baseline
removal, z-score), and returns per-window sigmoid probabilities for all 150
labels plus a focused set of label groups (GROUPS) that match what a Holter
screen needs.

The output is a screening aid, not a diagnosis. The model has never seen a
Polar H10 chest-strap signal; see RESULTS.md for how it does on H10-like data.

Usage (from ai/ecgfounder/):
    python infer.py ../../data/recordings/day1.csv -o out.csv         # ECGLogger CSV (130 Hz)
    python infer.py ../../data/mitdb/100 -o out.csv                   # WFDB record (first lead)
    python infer.py signal.npy --fs 130 -o out.csv                    # raw samples
    python infer.py ... --backend onnx --onnx weights/ecgfounder_1lead_int8.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE / "weights"
DEFAULT_CKPT = WEIGHTS_DIR / "1_lead_ECGFounder.pth"

FS = 500          # model input rate (Hz)
WIN_S = 10.0      # model input length (s)
WIN = int(FS * WIN_S)

TASKS = [t.strip() for t in (HERE / "third_party" / "tasks.txt").read_text().splitlines() if t.strip()]
assert len(TASKS) == 150, len(TASKS)

# Focused label groups. A group's score is the max probability of its member
# labels. Names are copied verbatim from tasks.txt (checked at import).
GROUPS: dict[str, list[str]] = {
    "sinus": ["SINUS RHYTHM", "NORMAL SINUS RHYTHM"],
    "pvc": ["PREMATURE VENTRICULAR COMPLEXES",
            "WITH PREMATURE VENTRICULAR OR ABERRANTLY CONDUCTED COMPLEXES",
            "PREMATURE VENTRICULAR AND FUSION COMPLEXES",
            "IN A PATTERN OF BIGEMINY"],
    "bigeminy": ["IN A PATTERN OF BIGEMINY"],
    "pac": ["PREMATURE ATRIAL COMPLEXES",
            "PREMATURE SUPRAVENTRICULAR COMPLEXES",
            "SUPRAVENTRICULAR COMPLEXES"],
    "ectopic_any": ["PREMATURE ECTOPIC COMPLEXES"],
    "vt": ["VENTRICULAR TACHYCARDIA"],
    "wide_qrs_tachy": ["WIDE QRS TACHYCARDIA"],
    "svt": ["SUPRAVENTRICULAR TACHYCARDIA", "MULTIFOCAL ATRIAL TACHYCARDIA"],
    "afib": ["ATRIAL FIBRILLATION"],
    "aflutter": ["ATRIAL FLUTTER"],
    "sinus_brady": ["SINUS BRADYCARDIA", "MARKED SINUS BRADYCARDIA"],
    "sinus_tachy": ["SINUS TACHYCARDIA"],
    "pause": ["WITH SINUS PAUSE",
              "WITH 2ND DEGREE SA BLOCK MOBITZ I",
              "WITH 2ND DEGREE SA BLOCK MOBITZ II"],
    "av_block": ["WITH 1ST DEGREE AV BLOCK",
                 "WITH PROLONGED AV CONDUCTION",
                 "WITH 2ND DEGREE AV BLOCK MOBITZ I",
                 "WITH 2:1 AV CONDUCTION",
                 "WITH VARIABLE AV BLOCK",
                 "BLOCKED",
                 "WITH COMPLETE HEART BLOCK",
                 "WITH AV DISSOCIATION"],
}
# Combined groups used for the evaluation targets.
GROUPS["vt_or_wct"] = GROUPS["vt"] + GROUPS["wide_qrs_tachy"]
GROUPS["af_or_afl"] = GROUPS["afib"] + GROUPS["aflutter"]
GROUPS["high_grade_av_block"] = ["WITH 2ND DEGREE AV BLOCK MOBITZ I", "WITH 2:1 AV CONDUCTION",
                                 "WITH VARIABLE AV BLOCK", "BLOCKED", "WITH COMPLETE HEART BLOCK"]
GROUPS["pause_or_block"] = GROUPS["pause"] + GROUPS["high_grade_av_block"]

_missing = [l for ls in GROUPS.values() for l in ls if l not in TASKS]
assert not _missing, f"labels not in tasks.txt: {_missing}"
GROUP_IDX = {g: [TASKS.index(l) for l in ls] for g, ls in GROUPS.items()}


# --------------------------------------------------------------------------
# Preprocessing (must match ECGFounder exactly)
# --------------------------------------------------------------------------

def filter_bandpass(signal: np.ndarray, fs: float) -> np.ndarray:
    """ECGFounder util.filter_bandpass (MIT, PKUDigitalHealth), vectorised.

    signal: (channels, time). Same steps and parameters as the original:
    iirnotch(50, Q=30) -> 4th-order Butterworth 0.67-40 Hz (both zero-phase
    filtfilt) -> subtract a 0.4 s median filter. scipy.signal.medfilt pads
    with zeros; median_filter(mode="constant", cval=0) is the same operation,
    only faster (checked in tests/test_preprocess.py).
    """
    b, a = iirnotch(50, 30, fs)
    x = filtfilt(b, a, signal, axis=-1)
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    x = filtfilt(b, a, x, axis=-1)
    k = int(0.4 * fs) + 1
    if k % 2 == 0:
        k += 1
    size = (1,) * (x.ndim - 1) + (k,)
    baseline = median_filter(x, size=size, mode="constant", cval=0.0)
    return x - baseline


def z_score(x: np.ndarray) -> np.ndarray:
    """ECGFounder z_score_normalization: one mean/std over the whole sample."""
    axes = tuple(range(1, x.ndim))
    return (x - x.mean(axis=axes, keepdims=True)) / (x.std(axis=axes, keepdims=True) + 1e-8)


def preprocess_windows(windows: np.ndarray) -> np.ndarray:
    """(n, 5000) raw 500 Hz windows -> (n, 1, 5000) float32 model input.

    Each 10 s window is filtered on its own, as every training record was.
    """
    w = np.nan_to_num(np.asarray(windows, dtype=np.float64), nan=0.0)
    w = filter_bandpass(w, FS)
    w = z_score(w)
    return w[:, None, :].astype(np.float32)


def resample_to_500(x: np.ndarray, fs: float) -> np.ndarray:
    """Polyphase resampling (anti-aliased) from fs to 500 Hz."""
    ratio = Fraction(FS / fs).limit_denominator(1000)
    if ratio == 1:
        return np.asarray(x, dtype=np.float64)
    return resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)


def window_starts(n_samples: int, step_s: float = 5.0) -> np.ndarray:
    """Start indices (at 500 Hz) of full 10 s windows; a tail shorter than 10 s is dropped."""
    step = int(round(step_s * FS))
    if n_samples < WIN:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_samples - WIN + 1, step, dtype=np.int64)


# --------------------------------------------------------------------------
# Model backends
# --------------------------------------------------------------------------

def build_net(n_classes: int = 150):
    sys.path.insert(0, str(HERE / "third_party"))
    from net1d import Net1D  # vendored, MIT

    # Same hyper-parameters as ECGFounder finetune_model.ft_1lead_ECGFounder.
    return Net1D(in_channels=1, base_filters=64, ratio=1,
                 filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                 m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
                 kernel_size=16, stride=2, groups_width=16,
                 verbose=False, use_bn=False, use_do=False, n_classes=n_classes)


def load_torch_model(ckpt: Path = DEFAULT_CKPT, allow_random: bool = False):
    import torch

    model = build_net()
    if ckpt is not None and Path(ckpt).exists():
        # weights_only=True: never unpickle arbitrary objects from a download.
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # The original code loads with strict=False; we insist that every
        # model weight was found so a wrong checkpoint cannot pass silently.
        if missing:
            raise RuntimeError(f"checkpoint {ckpt} lacks {len(missing)} weights, e.g. {missing[:3]}")
        if unexpected:
            print(f"note: ignored {len(unexpected)} extra checkpoint keys", file=sys.stderr)
    elif not allow_random:
        raise FileNotFoundError(f"{ckpt} not found - run download_weights.py first")
    else:
        print("WARNING: random weights (benchmark only, outputs are meaningless)", file=sys.stderr)
    model.eval()
    return model


class Predictor:
    """Maps (n, 1, 5000) float32 input to (n, 150) probabilities."""

    def __init__(self, backend: str = "torch", ckpt: Path | None = DEFAULT_CKPT,
                 onnx_path: Path | None = None, threads: int | None = None,
                 batch: int = 32, allow_random: bool = False):
        self.backend = backend
        self.batch = batch
        if backend == "torch":
            import torch

            if threads:
                torch.set_num_threads(threads)
            self.model = load_torch_model(ckpt, allow_random=allow_random)
        elif backend == "onnx":
            import onnxruntime as ort

            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        else:
            raise ValueError(backend)

    def logits(self, x: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(x), self.batch):
            xb = x[i:i + self.batch]
            if self.backend == "torch":
                import torch

                with torch.inference_mode():
                    out.append(self.model(torch.from_numpy(xb)).numpy())
            else:
                out.append(self.sess.run(None, {self.inp: xb})[0])
        return np.concatenate(out) if out else np.zeros((0, len(TASKS)), np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x).astype(np.float64)
        return 1.0 / (1.0 + np.exp(-z))


def group_scores(probs: np.ndarray) -> dict[str, np.ndarray]:
    return {g: probs[:, idx].max(axis=1) for g, idx in GROUP_IDX.items()}


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

@dataclass
class Result:
    t_start: np.ndarray            # window start (s from signal start)
    probs: np.ndarray              # (n_windows, 150)
    groups: dict[str, np.ndarray]  # group -> (n_windows,)
    seconds_model: float
    seconds_preproc: float


def run(signal: np.ndarray, fs: float, predictor: Predictor, step_s: float = 5.0,
        chunk: int = 512) -> Result:
    x = resample_to_500(signal, fs)
    starts = window_starts(len(x), step_s)
    probs, tp, tm = [], 0.0, 0.0
    for i in range(0, len(starts), chunk):
        s = starts[i:i + chunk]
        t0 = time.perf_counter()
        w = preprocess_windows(np.stack([x[k:k + WIN] for k in s]))
        t1 = time.perf_counter()
        probs.append(predictor(w))
        tm += time.perf_counter() - t1
        tp += t1 - t0
    p = np.concatenate(probs) if probs else np.zeros((0, len(TASKS)))
    return Result(starts / FS, p, group_scores(p), tm, tp)


def load_signal(path: str, fs: float | None, lead: str | None) -> tuple[np.ndarray, float]:
    p = Path(path)
    if p.suffix == ".npy":
        if fs is None:
            sys.exit("--fs is required for .npy input")
        return np.load(p).astype(np.float64).ravel(), fs
    if p.suffix.lower() == ".csv":
        sys.path.insert(0, str(HERE.parents[1] / "analysis"))
        from ecgpipe.io import load_ecglogger_csv  # ECGLogger / Polar H10 CSV

        rec = load_ecglogger_csv(p)
        return rec.ecg.astype(np.float64), rec.fs
    import wfdb

    rec = wfdb.rdrecord(str(p.with_suffix("") if p.suffix in (".hea", ".dat") else p))
    ch = rec.sig_name.index(lead) if lead else 0
    return np.nan_to_num(rec.p_signal[:, ch]), float(rec.fs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="ECGLogger .csv, .npy (needs --fs) or WFDB record path")
    ap.add_argument("-o", "--out", help="output CSV (one row per window); default: print summary")
    ap.add_argument("--fs", type=float, help="sampling rate of a .npy input")
    ap.add_argument("--lead", help="WFDB channel name (default: first channel)")
    ap.add_argument("--step", type=float, default=5.0, help="window step in s (default 5)")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--onnx", type=Path, default=WEIGHTS_DIR / "ecgfounder_1lead_fp32.onnx")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--all-labels", action="store_true", help="also write all 150 raw label probabilities")
    args = ap.parse_args()

    sig, fs = load_signal(args.input, args.fs, args.lead)
    pred = Predictor(args.backend, ckpt=args.ckpt, onnx_path=args.onnx, threads=args.threads)
    res = run(sig, fs, pred, step_s=args.step)
    n = len(res.t_start)
    print(f"{n} windows of 10 s from {len(sig) / fs:.0f} s at {fs:g} Hz; "
          f"model {1000 * res.seconds_model / max(n, 1):.1f} ms/window, "
          f"preprocessing {1000 * res.seconds_preproc / max(n, 1):.1f} ms/window", file=sys.stderr)

    if args.out:
        import pandas as pd

        df = pd.DataFrame({"t_start_s": res.t_start, **res.groups})
        if args.all_labels:
            df = pd.concat([df, pd.DataFrame(res.probs, columns=TASKS)], axis=1)
        df.to_csv(args.out, index=False, float_format="%.4f")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        for g, v in res.groups.items():
            if n:
                print(f"{g:20s} max {v.max():.3f}  windows >= 0.5: {(v >= 0.5).sum()}")


if __name__ == "__main__":
    main()
