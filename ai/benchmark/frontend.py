"""Per-record front end of the long-term benchmark (cached, resumable, parallel).

For each record of datasets.records():
    1. one lead -> 130 Hz (Polar H10 rate, polyphase as analysis/tools/eval_physionet.py)
       -> ECGLogger CSV round trip in 1-hour files, as the phone writes them
    2. rules     analysis/ecgpipe analyse_beats (detector, noise gate, N/S/V/Q)
    3. heartKIT  BEAT-3-EFF-SM on the rules' beats (noise gate shared), ARR-2-EFF-SM AF frames
    4. ECGFounder 1-lead, 150 label probabilities per 10 s window
    5. CLEF-Medium 1,024-number embedding per 10 s window (head: ai/clef/head.py)
    6. the uniform reference of datasets.reference()
Windows for 4-5 are non-overlapping 10 s windows from t = 0 on the 130 Hz signal
taken to 500 Hz (as ai/ecgfounder/infer.run does). ECGFounder runs in fp32 as
in the earlier benchmark; CLEF in bfloat16 autocast (AMX CPUs: ~2x faster). On
MIT-BIH 208 bf16 moved ECGFounder probabilities by up to 0.33 (mean 0.005) but
CLEF embeddings only to cosine >= 0.99996 (python frontend.py --check-bf16).

Output: data/work/bench2/<db>/<rec>.pkl (+ .clef.npy, float16), written
atomically, so an interrupted run resumes where it stopped. Raw Icentia11k
segments are deleted once processed (--keep-raw to keep them).

Usage (from ai/benchmark/):
    python frontend.py -j 4                       # everything
    python frontend.py --db ltdb afdb -j 4
    python frontend.py --check-bf16 mitdb/208     # fp32 vs bf16 on one record
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "analysis"), str(REPO / "ai" / "ecgfounder"),
                str(REPO / "ai" / "ecgfounder" / "third_party"), str(REPO / "ai" / "heartkit"),
                str(REPO / "ai" / "clef")]
import datasets as ds  # noqa: E402

FS = 130
STEP_S = 10.0
CACHE = ds.DATA / "work" / "bench2"
CHUNK = 256  # windows per model call (memory bound)

_M: dict = {}


def cache_path(r: ds.Rec) -> Path:
    return CACHE / r.db / f"{r.rec}.pkl"


def _init(threads: int, bf16: str) -> None:
    """bf16: "" (all fp32), "clef" (default) or "all"."""
    import torch

    torch.set_num_threads(threads)
    import clef
    import hk
    import infer

    _M["ecgf"] = infer.Predictor("torch", threads=threads, batch=32, bf16=bf16 == "all")
    _M["clef"] = clef.Embedder(threads=threads, batch=32, bf16=bool(bf16))
    _M["hk_beat"], _M["hk_rhythm"] = hk.Model("beat"), hk.Model("rhythm")


def to_h10(x: np.ndarray, fs: float) -> np.ndarray:
    """Resample to 130 Hz and pass through ECGLogger CSV files of 1 hour."""
    from ecgpipe.io import load_ecglogger_csv, write_ecglogger_csv
    from scipy.signal import resample_poly

    ratio = Fraction(FS, int(round(fs))).limit_denominator(1000)
    y = resample_poly(x, ratio.numerator, ratio.denominator)
    n, parts = 3600 * FS, []
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.csv"
        for i in range(0, len(y), n):
            write_ecglogger_csv(p, 1.7e9 + np.arange(i, min(i + n, len(y))) / FS, y[i:i + n])
            parts.append(load_ecglogger_csv(p).ecg)
    return np.concatenate(parts) if parts else y


def foundation(ecg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """10 s windows -> (t_start, ECGFounder probs (n,150), CLEF embeddings (n,1024))."""
    import clef
    import infer

    x = infer.resample_to_500(ecg, FS)
    starts = infer.window_starts(len(x), STEP_S)
    probs, embs = [], []
    for i in range(0, len(starts), CHUNK):
        raw = np.stack([x[k:k + infer.WIN] for k in starts[i:i + CHUNK]])
        probs.append(_M["ecgf"](infer.preprocess_windows(raw)).astype(np.float16))
        embs.append(_M["clef"](clef.preprocess_windows(raw)).astype(np.float16))
    p = np.concatenate(probs) if probs else np.zeros((0, len(infer.TASKS)), np.float16)
    e = np.concatenate(embs) if embs else np.zeros((0, clef.DIM), np.float16)
    return starts / infer.FS, p, e


def nstdb_signal(r: ds.Rec) -> tuple[np.ndarray, float, str]:
    """NSTDB record: PhysioNet's em mixes as published; 'm' records are our muscle-artefact mixes."""
    import wfdb

    if r.rec[3] == "e":
        rec = wfdb.rdrecord(str(ds.DATA / "nstdb" / r.rec), channels=[0])
        return rec.p_signal[:, 0], float(rec.fs), rec.sig_name[0]
    clean = wfdb.rdrecord(str(ds.DATA / "mitdb" / r.rec[:3]), channels=[0])
    noise = wfdb.rdrecord(str(ds.DATA / "nstdb" / "ma"), channels=[0]).p_signal[:, 0]
    snr = -6 if r.rec[4:] == "_6" else int(r.rec[4:])
    ann = wfdb.rdann(str(ds.DATA / "mitdb" / r.rec[:3]), "atr")
    return ds.nst_mix(clean.p_signal[:, 0], noise, clean.fs, ann.sample, snr), float(clean.fs), "MLII"


def process(r: ds.Rec, keep_raw: bool = False, delete_raw: tuple[str, ...] = ()) -> tuple[str, float]:
    from ecgpipe.beats import analyse_beats

    import hk

    out = cache_path(r)
    if out.exists():
        return r.key, 0.0
    t0 = time.perf_counter()
    timing = {}
    if r.db == "nstdb":
        x, fs, lead = nstdb_signal(r)
    else:
        x, fs, lead = ds.read_signal(r)
    ref = ds.reference(r if r.db != "nstdb" else ds.Rec("mitdb", r.rec[:3], r.rec[:3], "noise"), fs, len(x))
    ecg = to_h10(x, fs)
    timing["csv"] = time.perf_counter() - t0
    t = time.perf_counter()
    _, b = analyse_beats(ecg, FS)
    timing["rules"] = time.perf_counter() - t
    t = time.perf_counter()
    lab, prob = hk.classify_beats(ecg, FS, b.idx, model=_M["hk_beat"])
    lab = np.where(b.noisy | (b.label == "Q"), "Q", lab)  # shared noise gate, as benchmark.py
    af_t, af_p = hk.af_frames(ecg, FS, model=_M["hk_rhythm"])
    timing["heartkit"] = time.perf_counter() - t
    t = time.perf_counter()
    wt, wp, we = foundation(ecg)
    timing["foundation"] = time.perf_counter() - t
    res = {
        "db": r.db, "rec": r.rec, "patient": r.patient, "split": r.split, "subset": ds.subset(r),
        "lead": lead, "fs_in": fs, "dur_s": len(ecg) / FS, "ref": ref,
        "rules": {"idx": b.idx.astype(np.int32), "label": np.asarray(b.label, "<U1"),
                  "irregular": np.asarray(b.irregular, bool), "noisy": np.asarray(b.noisy, bool)},
        "hk": {"label": np.asarray(lab, "<U1"), "prob": prob.astype(np.float16),
               "af_t": af_t.astype(np.float32), "af_p": af_p.astype(np.float16)},
        "win_t": wt.astype(np.float32), "ecgf": wp, "timing": timing,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".clef.tmp.npy"), we)
    os.replace(out.with_suffix(".clef.tmp.npy"), out.with_suffix(".clef.npy"))
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(tmp, out)  # last: its presence means the record is complete
    if r.db == "icentia11k" and not keep_raw:
        for e in (".hea", ".dat", ".atr"):
            (ds.DATA / "icentia11k" / f"{r.rec}{e}").unlink(missing_ok=True)
    elif r.db in delete_raw:
        (ds.DATA / r.db / f"{r.rec}.dat").unlink(missing_ok=True)
    return r.key, time.perf_counter() - t0


def load(r: ds.Rec, emb: bool = False) -> dict:
    d = pickle.loads(cache_path(r).read_bytes())
    if emb:
        d["clef"] = np.load(cache_path(r).with_suffix(".clef.npy"))
    return d


def check_bf16(key: str) -> None:
    """ECGFounder fp32 vs bf16 window probabilities on one record (focus groups)."""
    import infer

    db, rec = key.split("/", 1)
    r = next(x for x in ds.records([db]) if x.rec == rec)
    x, fs, _ = ds.read_signal(r)
    ecg = to_h10(x, fs)
    res = {}
    for bf in ("", "all"):
        _init(os.cpu_count() or 1, bf)
        res[bf] = foundation(ecg)
    p32, p16 = res[""][1].astype(np.float64), res["all"][1].astype(np.float64)
    idx = sorted({i for g in ("pvc", "pac", "vt_or_wct", "af_or_afl", "svt", "pause_or_block") for i in infer.GROUP_IDX[g]})
    d = np.abs(p32[:, idx] - p16[:, idx])
    e32, e16 = res[""][2].astype(np.float64), res["all"][2].astype(np.float64)
    cos = (e32 * e16).sum(1) / np.linalg.norm(e32, axis=1) / np.linalg.norm(e16, axis=1)
    print(f"{key}: {len(p32)} windows; ECGFounder focus labels |dp| max {d.max():.4f} mean {d.mean():.5f}; "
          f"CLEF embedding cosine min {cos.min():.5f} mean {cos.mean():.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", nargs="+", default=None)
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--bf16", choices=["", "clef", "all"], default="clef",
                    help="which models run in bfloat16 autocast (default: CLEF only)")
    ap.add_argument("--keep-raw", action="store_true", help="keep downloaded Icentia11k segments")
    ap.add_argument("--delete-raw", nargs="*", default=[], metavar="DB",
                    help="delete each record's .dat once cached, for these databases (to save disk)")
    ap.add_argument("--max-hours", type=float, default=None, help="stop submitting after this many hours")
    ap.add_argument("--reverse", action="store_true", help="process the list from the end (a second run in parallel)")
    ap.add_argument("--check-bf16", metavar="DB/REC")
    args = ap.parse_args()
    if args.check_bf16:
        return check_bf16(args.check_bf16)

    todo = [r for r in ds.records(args.db) if not cache_path(r).exists()]
    if args.reverse:
        todo = todo[::-1]
    print(f"{len(todo)} records to process", file=sys.stderr, flush=True)
    t_end = time.time() + 3600 * args.max_hours if args.max_hours else None
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(1, args.bf16)) as ex:
        futs = {}
        it = iter(todo)
        for r in it:  # keep the pool busy without queuing everything (so --max-hours can stop)
            futs[ex.submit(process, r, args.keep_raw, tuple(args.delete_raw))] = r
            if len(futs) >= 2 * args.jobs:
                break
        while futs:
            done = next(as_completed(futs))
            r = futs.pop(done)
            try:
                key, sec = done.result()
                print(f"{key} {sec:.0f} s", file=sys.stderr, flush=True)
            except Exception as e:  # report and continue; the record is retried next run
                print(f"FAILED {r.key}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if t_end is None or time.time() < t_end:
                nxt = next((x for x in it if not cache_path(x).exists()), None)  # another run may have done it
                if nxt is not None:
                    futs[ex.submit(process, nxt, args.keep_raw, tuple(args.delete_raw))] = nxt


if __name__ == "__main__":
    main()
