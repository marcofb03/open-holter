"""Check the pipeline against cardiologist-annotated MIT-BIH records.

Each record (30 min, 360 Hz) is resampled to the Polar H10 rate (130 Hz),
written in ECGLogger CSV layout, loaded back and analysed, so the full path
(loader -> beats -> labels) is exercised. Detected beats are matched to the
reference annotations within 150 ms.

Usage (from analysis/):  python tools/eval_mitdb.py ../data/mitdb 100 119 208 ...
Missing records are downloaded from PhysioNet into the given folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis"))
from ecgpipe.beats import analyse_beats  # noqa: E402
from ecgpipe.io import load_ecglogger_csv, write_ecglogger_csv  # noqa: E402

FS_OUT = 130
# AAMI beat classes from MIT-BIH symbols.
AAMI = {**{s: "N" for s in "NLRej"}, **{s: "S" for s in "AaJS"}, **{s: "V" for s in "VE"}, "F": "F",
        **{s: "Q" for s in "/fQ"}}


def evaluate(db_dir: Path, rec_name: str, work: Path) -> dict:
    rec = wfdb.rdrecord(str(db_dir / rec_name))
    ann = wfdb.rdann(str(db_dir / rec_name), "atr")
    lead = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    x = rec.p_signal[:, lead]
    fs_in = rec.fs
    y = resample_poly(x, FS_OUT, int(fs_in))  # 360 -> 130 Hz (13/36)

    t = 1.7e9 + np.arange(len(y)) / FS_OUT
    csv_path = work / f"mitdb_{rec_name}.csv"
    write_ecglogger_csv(csv_path, t, y)
    loaded = load_ecglogger_csv(csv_path)

    _, b = analyse_beats(loaded.ecg, loaded.fs)

    ref_t = np.array([s / fs_in for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI])
    ref_c = np.array([AAMI[sym] for sym in ann.symbol if sym in AAMI])
    det_t = b.idx / loaded.fs

    # Greedy nearest matching within 150 ms.
    j = np.searchsorted(det_t, ref_t)
    matched_ref = np.full(len(ref_t), -1)
    used = np.zeros(len(det_t), bool)
    for i, (rt, jj) in enumerate(zip(ref_t, j)):
        cands = [k for k in (jj - 1, jj) if 0 <= k < len(det_t) and not used[k]]
        if not cands:
            continue
        k = min(cands, key=lambda k: abs(det_t[k] - rt))
        if abs(det_t[k] - rt) <= 0.15:
            matched_ref[i] = k
            used[k] = True

    tp = int((matched_ref >= 0).sum())
    res = {
        "record": rec_name,
        "ref_beats": len(ref_t),
        "beat_Se": tp / len(ref_t),
        "beat_PPV": tp / max(len(det_t), 1),
        "unreadable_pct": 100 * (b.label == "Q").mean(),
    }
    det_lab = np.array([b.label[k] if k >= 0 else "-" for k in matched_ref])
    for c in ("V", "S"):
        ref_pos = ref_c == c
        det_pos_all = b.label == c
        tp_c = int((ref_pos & (det_lab == c)).sum())
        res[f"{c}_ref"] = int(ref_pos.sum())
        res[f"{c}_det"] = int(det_pos_all.sum())
        res[f"{c}_Se"] = tp_c / ref_pos.sum() if ref_pos.sum() else float("nan")
        res[f"{c}_PPV"] = tp_c / det_pos_all.sum() if det_pos_all.sum() else float("nan")
    return res


def main() -> None:
    db_dir = Path(sys.argv[1])
    names = sys.argv[2:]
    work = REPO / "data" / "work"
    work.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (db_dir / f"{n}.dat").exists()]
    if missing:
        db_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("mitdb", dl_dir=str(db_dir), records=missing)
    cols = ["record", "ref_beats", "beat_Se", "beat_PPV", "unreadable_pct",
            "V_ref", "V_det", "V_Se", "V_PPV", "S_ref", "S_det", "S_Se", "S_PPV"]
    print(" ".join(f"{c:>9}" for c in cols))
    for n in names:
        r = evaluate(db_dir, n, work)
        print(" ".join(f"{r[c]:>9.3f}" if isinstance(r[c], float) else f"{r[c]:>9}" for c in cols))


if __name__ == "__main__":
    main()
