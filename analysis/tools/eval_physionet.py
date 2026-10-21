"""Validate the pipeline against cardiologist-annotated PhysioNet databases.

Every record is resampled to the Polar H10 rate (130 Hz), written in
ECGLogger CSV layout, loaded back and analysed, so the full path
(CSV loader -> beats -> labels -> events) is exercised exactly as for a real
recording. Detected beats are matched to the reference annotations within
150 ms and compared using the AAMI EC57 beat classes:

    N  normal / bundle-branch block / escape   (N L R e j n B)
    S  supraventricular ectopic                (A a J S)
    V  ventricular ectopic                     (V E)
    F  fusion of V and N                       (F)
    Q  paced / unclassifiable                  (/ f Q)

Scoring follows EC57: a reference F beat labelled V is not counted as a
false V; reference Q beats are left out of the class statistics. A
reference beat that falls in a stretch we call unreadable (Q) counts as
missed, so unreadable time is never hidden. Beat-by-beat comparison is
suspended during ventricular flutter / fibrillation ([ ... ] in MIT-BIH).

V / S runs: >= 3 consecutive reference V (S) beats form a reference run.
A run is "found" if a detected run of the same class overlaps it in time.

Usage (from analysis/):
    python tools/eval_physionet.py mitdb --subset DS1
    python tools/eval_physionet.py mitdb --subset DS2 svdb incartdb --markdown ../data/work/results.md

Missing records are downloaded into data/<db>/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

REPO = Path(__file__).resolve().parents[2]
# ECGPIPE_PATH lets you evaluate another version of the code (e.g. a git worktree of an
# older commit) with the same tool: ECGPIPE_PATH=../old/analysis python tools/eval_physionet.py
sys.path.insert(0, os.environ.get("ECGPIPE_PATH", str(REPO / "analysis")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ecgpipe.beats import analyse_beats  # noqa: E402
from ecgpipe.events import find_events  # noqa: E402
from ecgpipe.io import load_ecglogger_csv, write_ecglogger_csv  # noqa: E402
from fetch_physionet import fetch  # noqa: E402

FS_OUT = 130
MATCH_S = 0.15

AAMI = {**{s: "N" for s in "NLRejnB"}, **{s: "S" for s in "AaJS"}, **{s: "V" for s in "VE"},
        "F": "F", **{s: "Q" for s in "/fQ"}}

# de Chazal et al. 2004 inter-patient split of MIT-BIH (paced records excluded).
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
       "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
       "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
PACED = {"102", "104", "107", "217"}

# Which lead to use per database (closest to a chest strap / lead II view).
LEAD = {"mitdb": "MLII", "svdb": "ECG1", "incartdb": "II"}


def records(db: str, subset: str | None) -> list[str]:
    if db == "mitdb":
        return {"DS1": DS1, "DS2": DS2}.get(subset or "", DS1 + DS2)
    folder = fetch(db)
    return sorted(p.stem for p in folder.glob("*.hea"))


def _match(ref_t: np.ndarray, det_t: np.ndarray, tol: float = MATCH_S) -> np.ndarray:
    """Greedy nearest match; returns det index per reference beat (-1 = missed)."""
    j = np.searchsorted(det_t, ref_t)
    out = np.full(len(ref_t), -1)
    used = np.zeros(len(det_t), bool)
    for i, (rt, jj) in enumerate(zip(ref_t, j)):
        cands = [k for k in (jj - 1, jj, jj + 1) if 0 <= k < len(det_t) and not used[k]]
        if not cands:
            continue
        k = min(cands, key=lambda k: abs(det_t[k] - rt))
        if abs(det_t[k] - rt) <= tol:
            out[i] = k
            used[k] = True
    return out


def _ref_runs(cls: np.ndarray, target: str, min_len: int = 3) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, c in enumerate(list(cls) + ["-"]):
        if c == target and start is None:
            start = i
        elif c != target and start is not None:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    return runs


def _overlaps(a0: float, a1: float, spans: list[tuple[float, float]], tol: float = MATCH_S) -> bool:
    return any(a0 - tol <= z and b0 - tol <= a1 for b0, z in spans)


def add_noise(y: np.ndarray, fs: float, beat_t: np.ndarray, kind: str, snr_db: float,
              seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Add synthetic noise in 2-min bursts (on from minute 5, 2 min on / 2 min off),
    following the MIT-BIH Noise Stress Test protocol. Returns (signal, noisy-time mask).

    kind: 'emg'    – muscle noise: white noise band-limited to 20–60 Hz
          'motion' – electrode motion: 0.3–4 Hz wander plus random sharp
                     QRS-like transients (~1 per second), the worst case for
                     false beats.
    SNR as in the NST database: S = (median QRS peak-to-peak)^2 / 8, N = noise power.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if kind == "emg":
        sos = butter(4, [20, min(60, 0.45 * fs)], btype="band", fs=fs, output="sos")
        z = sosfiltfilt(sos, rng.standard_normal(n))
    elif kind == "motion":
        sos = butter(2, [0.3, 4.0], btype="band", fs=fs, output="sos")
        z = sosfiltfilt(sos, rng.standard_normal(n))
        z /= z.std()
        k = np.arange(int(0.4 * fs)) / fs
        for t0 in np.cumsum(rng.exponential(1.0, int(n / fs * 2))):
            i0 = int(t0 * fs)
            if i0 >= n:
                break
            rise, decay = rng.uniform(0.01, 0.03), rng.uniform(0.04, 0.12)
            pulse = np.exp(-k / decay) - np.exp(-k / rise)
            pulse *= rng.choice([-1, 1]) * rng.uniform(1.5, 4.0) / pulse.max()
            m = min(len(pulse), n - i0)
            z[i0: i0 + m] += pulse[:m]
    else:
        raise ValueError(kind)
    half = int(0.06 * fs)
    p2p = [np.ptp(y[max(int(t * fs) - half, 0): int(t * fs) + half + 1]) for t in beat_t[::10]]
    s_pow = np.median(p2p) ** 2 / 8
    z *= np.sqrt(s_pow / 10 ** (snr_db / 10) / np.mean(z ** 2))
    t = np.arange(n) / fs
    on = (t >= 300) & (((t - 300) // 120) % 2 == 0)
    return y + z * on, on


def evaluate(db: str, name: str, noise: str | None = None, snr_db: float = 6.0,
             rules: dict | None = None) -> dict:
    path = str(REPO / "data" / db / name)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    lead = rec.sig_name.index(LEAD[db]) if LEAD[db] in rec.sig_name else 0
    x = np.nan_to_num(rec.p_signal[:, lead])
    ratio = Fraction(FS_OUT, int(round(rec.fs))).limit_denominator(1000)
    y = resample_poly(x, ratio.numerator, ratio.denominator)
    noisy_time = np.zeros(len(y), bool)
    if noise:
        beat_t = np.asarray(ann.sample)[np.isin(ann.symbol, list(AAMI))] / rec.fs
        seed = int.from_bytes(name.encode(), "little") % 2**32
        y, noisy_time = add_noise(y, FS_OUT, beat_t, noise, snr_db, seed)

    # Full CSV round-trip, like a real ECGLogger file.
    t = 1.7e9 + np.arange(len(y)) / FS_OUT
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / f"{db}_{name}.csv"
        write_ecglogger_csv(csv_path, t, y)
        loaded = load_ecglogger_csv(csv_path)
    _, b = analyse_beats(loaded.ecg, loaded.fs, **({"rules": rules} if rules else {}))
    events = find_events(b, loaded.fs)
    det_t = b.idx / loaded.fs
    dur = len(loaded.ecg) / loaded.fs

    # Reference beats; skip ventricular flutter episodes and the first/last second.
    sym = np.asarray(ann.symbol)
    ann_t = np.asarray(ann.sample) / rec.fs
    vf = []
    for k in np.where(sym == "[")[0]:
        ends = np.where((sym == "]") & (np.arange(len(sym)) > k))[0]
        vf.append((ann_t[k], ann_t[ends[0]] if ends.size else dur))
    in_vf = lambda tt: np.array([any(a <= v <= z for a, z in vf) for v in tt], bool)  # noqa: E731
    is_beat = np.isin(sym, list(AAMI))
    ref_t, ref_c = ann_t[is_beat], np.array([AAMI[s] for s in sym[is_beat]])
    keep = (ref_t > 1.0) & (ref_t < dur - 1.0) & ~in_vf(ref_t)
    ref_t, ref_c = ref_t[keep], ref_c[keep]
    det_keep = (det_t > 1.0) & (det_t < dur - 1.0) & ~in_vf(det_t)

    m = _match(ref_t, det_t)
    matched_det = np.zeros(len(det_t), bool)
    matched_det[m[m >= 0]] = True
    ref_of_det = np.full(len(det_t), "-", dtype="<U1")
    ref_of_det[m[m >= 0]] = ref_c[m >= 0]
    det_lab = np.where(m >= 0, b.label[np.clip(m, 0, None)], "-")

    tp = int((m >= 0).sum())
    res = {
        "db": db, "record": name, "ref_beats": len(ref_t),
        "beat_TP": tp, "beat_FN": len(ref_t) - tp,
        "beat_FP": int((det_keep & ~matched_det).sum()),
        "det_beats": int(det_keep.sum()),
        "Q_beats": int((det_keep & (b.label == "Q")).sum()),
        "dur_s": int(dur),
        "noise_s": int(noisy_time.sum() / FS_OUT),
    }
    for c, ignore in (("V", {"F", "Q"}), ("S", {"Q"})):
        ref_pos = ref_c == c
        det_c = det_keep & (b.label == c)
        res[f"{c}_ref"] = int(ref_pos.sum())
        res[f"{c}_TP"] = int((ref_pos & (det_lab == c)).sum())
        res[f"{c}_FP"] = int((det_c & (ref_of_det != c) & ~np.isin(ref_of_det, list(ignore))).sum())
        # Reference class beats lost because their stretch was called unreadable.
        res[f"{c}_inQ"] = int((ref_pos & (det_lab == "Q")).sum())

        # Runs (>= 3 beats).
        rr = [(ref_t[a], ref_t[z]) for a, z in _ref_runs(ref_c, c)]
        dr = [(e.t_start, e.t_end) for e in events if e.kind == f"{c} run (>=3)"
              and 1.0 < e.t_start < dur - 1.0 and not in_vf([e.t_start])[0]]
        res[f"{c}run_ref"] = len(rr)
        res[f"{c}run_found"] = sum(_overlaps(a, z, dr) for a, z in rr)
        res[f"{c}run_det"] = len(dr)
        res[f"{c}run_true"] = sum(_overlaps(a, z, rr) for a, z in dr)
    return res


def _ratio(a: int, b: int) -> float:
    return a / b if b else float("nan")


def metrics(r: dict) -> dict:
    return {
        "beat_Se": _ratio(r["beat_TP"], r["beat_TP"] + r["beat_FN"]),
        "beat_PPV": _ratio(r["beat_TP"], r["beat_TP"] + r["beat_FP"]),
        "Q_pct": 100 * _ratio(r["Q_beats"], r["det_beats"]),
        "V_Se": _ratio(r["V_TP"], r["V_ref"]),
        "V_PPV": _ratio(r["V_TP"], r["V_TP"] + r["V_FP"]),
        "S_Se": _ratio(r["S_TP"], r["S_ref"]),
        "S_PPV": _ratio(r["S_TP"], r["S_TP"] + r["S_FP"]),
    }


def gross(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if isinstance(rows[0][k], int)]
    return {k: sum(r[k] for r in rows) for k in keys}


COLS = ["ref_beats", "beat_Se", "beat_PPV", "Q_pct", "V_ref", "V_Se", "V_PPV",
        "S_ref", "S_Se", "S_PPV", "Vrun_ref", "Vrun_found", "Vrun_det", "Vrun_true"]


def _fmt(v) -> str:
    if isinstance(v, float):
        return "–" if np.isnan(v) else f"{v:.3f}" if v <= 1.0 else f"{v:.1f}"
    return str(v)


def table(title: str, rows: list[dict], per_record: bool) -> str:
    head = "| record | " + " | ".join(COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 1)
    lines = [f"### {title}", "", head, sep]
    if per_record:
        for r in rows:
            v = {**r, **metrics(r)}
            lines.append(f"| {r['record']} | " + " | ".join(_fmt(v[c]) for c in COLS) + " |")
    g = gross(rows)
    v = {**g, **metrics(g)}
    lines.append(f"| **gross** | " + " | ".join(_fmt(v[c]) for c in COLS) + " |")
    false_v = g["Vrun_det"] - g["Vrun_true"]
    s = ("S runs: {Srun_ref} reference, {Srun_found} found; {Srun_det} detected, "
         "{Srun_true} overlap a reference S run. V beats lost in unreadable stretches: "
         "{V_inQ}; S beats: {S_inQ}. ").format(**g)
    s += f"False V runs: {false_v} in {g['dur_s'] / 3600:.1f} h ({false_v / (g['dur_s'] / 3600):.1f} per hour)"
    if g["noise_s"]:
        s += f", of which {g['noise_s'] / 3600:.1f} h with added noise"
    s += "."
    return "\n".join(lines + ["", s, ""])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dbs", nargs="+", help="mitdb, svdb, incartdb")
    ap.add_argument("--subset", choices=["DS1", "DS2", "all"], default="all",
                    help="MIT-BIH split (only applies to mitdb)")
    ap.add_argument("--records", nargs="*", help="only these records (first database)")
    ap.add_argument("--markdown", help="write tables to this file")
    ap.add_argument("--summary", action="store_true", help="print only the gross line per database")
    ap.add_argument("--noise", choices=["emg", "motion"], help="add synthetic noise bursts")
    ap.add_argument("--snr", type=float, default=6.0, help="noise SNR in dB (default 6)")
    ap.add_argument("--rules", type=json.loads, help="override labelling thresholds (JSON), for tuning")
    ap.add_argument("-j", "--jobs", type=int, default=4)
    args = ap.parse_intermixed_args()

    out = []
    for db in args.dbs:
        names = args.records or records(db, args.subset)
        names = [n for n in names if not (db == "mitdb" and n in PACED)]
        fetch(db, names)
        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(evaluate, [db] * len(names), names, [args.noise] * len(names),
                               [args.snr] * len(names), [args.rules] * len(names)))
        title = f"{db}" + (f" {args.subset}" if db == "mitdb" and args.subset != "all" else "")
        if args.noise:
            title += f" + {args.noise} noise at {args.snr:g} dB"
        t = table(f"{title} ({len(rows)} records)", rows, per_record=not args.summary)
        print(t, flush=True)
        out.append(t)
    if args.markdown:
        Path(args.markdown).write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
