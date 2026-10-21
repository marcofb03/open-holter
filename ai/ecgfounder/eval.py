"""Evaluate ECGFounder (1 lead) as a window-level arrhythmia flagger on H10-like data.

Databases (PhysioNet, downloaded into --data on first use, git-ignored):
    mitdb     MIT-BIH Arrhythmia, 48 x 30 min, 360 Hz, lead MLII
    svdb      MIT-BIH Supraventricular Arrhythmia, 78 x 30 min, 128 Hz, lead ECG1
    incartdb  St Petersburg INCART 12-lead, 75 x 30 min, 257 Hz, lead I
Paced MIT-BIH records (102, 104, 107, 217) are left out, as in AAMI EC57.

Every record is first resampled to 130 Hz (Polar H10 rate) and then up to
500 Hz, so the model sees what it would see from the strap. It is then cut
into 10 s windows with a 5 s step and passed through infer.py.

Window ground truth (from the cardiologists' beat and rhythm annotations):
    pvc        >= 1 ventricular beat (V, E)
    sve        >= 1 supraventricular ectopic beat (A, a, J, S)
    vrun       >= 3 consecutive V beats inside the window (VT / V run)
    svt        >= 3 consecutive S beats inside, or >= 2 s of an (SVTA episode
    af         >= 50 % of the window in (AFIB or (AFL; 0 % = negative
    sinus      whole window in a sinus rhythm ((N, (SBR, (B, (T; ectopy allowed);
               negative = >= 50 % in AF, flutter, SVT, VT, nodal or idioventricular
    brady      window fully in (N/(SBR, all beats normal, mean rate < 60 bpm
    tachy      same, mean rate > 100 bpm (negatives: same windows, 60-100 bpm and the other side)
    pause      an RR interval > 2.0 s inside the window
Beats within 0.3 s of a window edge, runs only partly inside, and windows with
a partial AF overlap are "ambiguous" and left out of that target (not counted
as negative). Rhythm targets need rhythm annotations, which INCART lacks.

Each target is scored with a label group from infer.GROUPS. A threshold per
target is chosen on MIT-BIH (max F1) and then applied unchanged to SVDB and
INCART, so those two numbers are out-of-sample. Windows overlap and come from
few patients, so 95 % CIs use a bootstrap over records, not windows.

Usage (from ai/ecgfounder/):
    python eval.py                                   # all three databases, torch fp32
    python eval.py --db mitdb --records 100 119 208  # quick check
    python eval.py --backend onnx --onnx weights/ecgfounder_1lead_int8.onnx --tag int8
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
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import infer  # noqa: E402

DBS = {
    "mitdb": {"lead": "MLII", "exclude": {"102", "104", "107", "217"}},
    "svdb": {"lead": "ECG1", "exclude": set()},
    "incartdb": {"lead": "I", "exclude": set()},
}
H10_FS = 130
EDGE_S = 0.3

N_CLASS = set("NLRej")
S_CLASS = set("AaJS")
V_CLASS = set("VE")
BEATS = N_CLASS | S_CLASS | V_CLASS | set("F/fQ")
# Rhythm annotation codes (MIT-BIH style). Bigeminy/trigeminy are sinus with ectopy.
SINUS_RHYTHMS = {"N", "SBR", "B", "T"}
NON_SINUS_RHYTHMS = {"AFIB", "AFL", "SVTA", "VT", "VFL", "NOD", "IVR", "AB"}

# target -> label group from infer.GROUPS used as its score
TARGETS = {
    "pvc": "pvc",
    "sve": "pac",
    "vrun": "vt_or_wct",
    "svt": "svt",
    "af": "af_or_afl",
    "sinus": "sinus",
    "brady": "sinus_brady",
    "tachy": "sinus_tachy",
    "pause": "pause_or_block",
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def ensure_db(db: str, data: Path) -> Path:
    import wfdb

    d = data / db
    if not (d / "RECORDS").exists():
        print(f"downloading {db} from PhysioNet into {d} ...", file=sys.stderr)
        wfdb.dl_database(db, str(d))
        names = sorted(p.stem for p in d.glob("*.hea"))
        (d / "RECORDS").write_text("\n".join(names) + "\n")
    return d


def records(db: str, d: Path) -> list[str]:
    names = [r.strip() for r in (d / "RECORDS").read_text().split() if r.strip()]
    return [r for r in names if r not in DBS[db]["exclude"]]


def h10_like(x: np.ndarray, fs: float) -> np.ndarray:
    """Resample to 130 Hz (as recorded by the H10). infer.run takes it to 500 Hz."""
    from fractions import Fraction

    r = Fraction(H10_FS / fs).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator)


def rhythm_intervals(ann, fs: float, n_samples: int) -> list[tuple[float, float, str]]:
    marks = [(s / fs, a.strip("\x00").strip()) for s, sym, a in zip(ann.sample, ann.symbol, ann.aux_note)
             if sym == "+" and a.strip("\x00").strip().startswith("(")]
    out = []
    for i, (t, r) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else n_samples / fs
        out.append((t, end, r[1:]))
    return out


def runs(mask: np.ndarray, min_len: int = 3) -> list[tuple[int, int]]:
    """[start, end) index ranges of True runs of length >= min_len."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def window_truth(ann, fs: float, n_samples: int, t_start: np.ndarray) -> dict[str, np.ndarray]:
    """Per-window targets: 1 positive, 0 negative, nan ambiguous / not available."""
    sym = np.array(ann.symbol)
    t = np.asarray(ann.sample) / fs
    is_beat = np.isin(sym, list(BEATS))
    bt, bs = t[is_beat], sym[is_beat]
    is_v, is_s = np.isin(bs, list(V_CLASS)), np.isin(bs, list(S_CLASS))
    is_n = np.isin(bs, list(N_CLASS))
    v_runs, s_runs = runs(is_v), runs(is_s)
    rhythm = rhythm_intervals(ann, fs, n_samples)
    has_rhythm = len(rhythm) > 0

    W = infer.WIN_S
    out = {k: np.full(len(t_start), np.nan) for k in TARGETS}
    for i, t0 in enumerate(t_start):
        t1 = t0 + W
        inside = (bt >= t0) & (bt < t1)
        core = (bt >= t0 + EDGE_S) & (bt < t1 - EDGE_S)
        edge = inside & ~core

        for key, m in (("pvc", is_v), ("sve", is_s)):
            if (core & m).any():
                out[key][i] = 1
            elif (edge & m).any():
                out[key][i] = np.nan
            else:
                out[key][i] = 0

        def run_truth(rs):
            best = 0
            touched = False
            for a, b in rs:
                k = int(inside[a:b].sum())
                touched |= k > 0
                best = max(best, k)
            return 1 if best >= 3 else (np.nan if touched else 0)

        out["vrun"][i] = run_truth(v_runs)
        svt = run_truth(s_runs)

        if has_rhythm:
            ov = {}
            for a, b, r in rhythm:
                o = overlap(t0, t1, a, b)
                if o > 0:
                    ov[r] = ov.get(r, 0.0) + o
            if ov.get("SVTA", 0.0) >= 2.0:
                svt = 1
            elif ov.get("SVTA", 0.0) > 0 and svt == 0:
                svt = np.nan
            af = ov.get("AFIB", 0.0) + ov.get("AFL", 0.0)
            out["af"][i] = 1 if af >= W / 2 else (0 if af == 0 else np.nan)

            sinus_s = sum(ov.get(r, 0.0) for r in SINUS_RHYTHMS)
            other_s = sum(ov.get(r, 0.0) for r in NON_SINUS_RHYTHMS)
            out["sinus"][i] = 1 if sinus_s >= W - 1e-6 else (0 if other_s >= W / 2 else np.nan)
            plain_sinus = ov.get("N", 0.0) + ov.get("SBR", 0.0) >= W - 1e-6
            if plain_sinus and inside.sum() >= 2 and is_n[inside].all():
                rr = np.diff(bt[inside])
                hr = 60.0 / rr.mean()
                out["brady"][i] = 1 if hr < 60 else 0
                out["tachy"][i] = 1 if hr > 100 else 0
        out["svt"][i] = svt

        rr_in = np.diff(bt[inside])
        out["pause"][i] = 1 if (rr_in > 2.0).any() else 0
    return out


# --------------------------------------------------------------------------
# Inference with cache
# --------------------------------------------------------------------------

def predict_record(db: str, d: Path, rec_name: str, pred: infer.Predictor, cache: Path,
                   lead_override: str | None) -> dict:
    import wfdb

    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"{db}_{rec_name}.npz"
    rec = wfdb.rdrecord(str(d / rec_name))
    ann = wfdb.rdann(str(d / rec_name), "atr")
    lead = lead_override or DBS[db]["lead"]
    ch = rec.sig_name.index(lead) if lead in rec.sig_name else 0
    x = np.nan_to_num(rec.p_signal[:, ch])
    if cf.exists():
        z = np.load(cf)
        t_start, probs, ms_model, ms_pre = z["t_start"], z["probs"].astype(np.float64), float(z["ms_model"]), float(z["ms_pre"])
    else:
        y = h10_like(x, rec.fs)
        res = infer.run(y, H10_FS, pred)
        n = max(len(res.t_start), 1)
        t_start, probs = res.t_start, res.probs
        ms_model, ms_pre = 1000 * res.seconds_model / n, 1000 * res.seconds_preproc / n
        np.savez_compressed(cf, t_start=t_start, probs=probs.astype(np.float32),
                            ms_model=ms_model, ms_pre=ms_pre)
    truth = window_truth(ann, rec.fs, rec.sig_len, t_start)
    return {"db": db, "record": rec_name, "lead": rec.sig_name[ch], "t_start": t_start, "probs": probs,
            "truth": truth, "ms_model": ms_model, "ms_pre": ms_pre}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    p = s >= thr
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    return {"Se": tp / (tp + fn) if tp + fn else float("nan"),
            "PPV": tp / (tp + fp) if tp + fp else float("nan"),
            "Sp": tn / (tn + fp) if tn + fp else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    if not 0 < y.sum() < len(y):
        return 0.5
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def gather(rows: list[dict], target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = infer.GROUP_IDX[TARGETS[target]]
    ys, ss, rid = [], [], []
    for k, r in enumerate(rows):
        y = r["truth"][target]
        m = ~np.isnan(y)
        ys.append(y[m])
        ss.append(r["probs"][m][:, g].max(axis=1))
        rid.append(np.full(m.sum(), k))
    if not ys:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    return np.concatenate(ys), np.concatenate(ss), np.concatenate(rid)


def bootstrap_auroc(y, s, rid, n=200, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    recs = np.unique(rid)
    by_rec = {r: np.flatnonzero(rid == r) for r in recs}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by_rec[r] for r in rng.choice(recs, len(recs), replace=True)])
        v = auroc(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < n // 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(rows: list[dict], dev_db: str = "mitdb") -> dict:
    dbs = list(dict.fromkeys(r["db"] for r in rows))
    out = {"thresholds": {}, "by_db": {}}
    for target in TARGETS:
        dev = [r for r in rows if r["db"] == dev_db] or rows
        y, s, _ = gather(dev, target)
        out["thresholds"][target] = best_f1_threshold(y, s)
    for db in dbs + ["all"]:
        sub = rows if db == "all" else [r for r in rows if r["db"] == db]
        res = {}
        for target in TARGETS:
            y, s, rid = gather(sub, target)
            n_pos_rec = len(np.unique(rid[y == 1])) if len(y) else 0
            lo, hi = bootstrap_auroc(y, s, rid) if 0 < y.sum() < len(y) else (float("nan"), float("nan"))
            res[target] = {"n": int(len(y)), "pos": int(y.sum()), "pos_records": n_pos_rec,
                           "AUROC": auroc(y, s), "AUROC_lo": lo, "AUROC_hi": hi, "AUPRC": auprc(y, s),
                           "thr": out["thresholds"][target], **at_threshold(y, s, out["thresholds"][target])}
        out["by_db"][db] = res
    return out


def fmt(v: float, nd: int = 3) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def tables_md(metrics: dict, timing: dict, meta: dict) -> str:
    lines = [f"<!-- generated by eval.py: {meta['cmd']} -->",
             f"Backend **{meta['backend']}** ({meta['model']}), {meta['n_windows']} windows from "
             f"{meta['n_records']} records, {meta['date']}.", ""]
    for db, res in metrics["by_db"].items():
        lines += [f"#### {db}", "",
                  "| target (score group) | windows | pos (records) | AUROC [95% CI] | AUPRC | thr | Se | PPV | Sp |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for t, m in res.items():
            if m["n"] == 0:
                continue
            ci = f"{fmt(m['AUROC'])} [{fmt(m['AUROC_lo'], 2)}–{fmt(m['AUROC_hi'], 2)}]"
            lines.append(f"| {t} ({TARGETS[t]}) | {m['n']} | {m['pos']} ({m['pos_records']}) | {ci} | "
                         f"{fmt(m['AUPRC'])} | {fmt(m['thr'], 2)} | {fmt(m['Se'])} | {fmt(m['PPV'])} | {fmt(m['Sp'])} |")
        lines.append("")
    lines += ["#### CPU time per 10 s window", "",
              "| step | ms/window |", "|---|---:|",
              f"| preprocessing (resample excluded) | {timing['ms_pre']:.1f} |",
              f"| model, batch {meta['batch']}, {meta['threads']} threads | {timing['ms_model']:.1f} |", "",
              f"CPU: {meta['cpu']}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", nargs="+", default=list(DBS), choices=list(DBS))
    ap.add_argument("--records", nargs="*", help="limit to these record names")
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--ckpt", type=Path, default=infer.DEFAULT_CKPT)
    ap.add_argument("--onnx", type=Path)
    ap.add_argument("--tag", default="fp32", help="name for cache and results (e.g. fp32, int8)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--incart-lead", default=None, help="INCART lead to use (default I)")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    pred = infer.Predictor(args.backend, ckpt=args.ckpt, onnx_path=args.onnx,
                           threads=args.threads, batch=args.batch)
    cache = args.data / "work" / "ecgfounder" / args.tag
    rows = []
    for db in args.db:
        d = ensure_db(db, args.data)
        for rec in records(db, d):
            if args.records and rec not in args.records:
                continue
            t0 = time.perf_counter()
            r = predict_record(db, d, rec, pred, cache, args.incart_lead if db == "incartdb" else None)
            rows.append(r)
            print(f"{db}/{rec} [{r['lead']}]: {len(r['t_start'])} windows, {time.perf_counter() - t0:.1f} s, "
                  f"V {np.nansum(r['truth']['pvc']):.0f} S {np.nansum(r['truth']['sve']):.0f} "
                  f"AF {np.nansum(r['truth']['af']):.0f}", file=sys.stderr)

    n_win = sum(len(r["t_start"]) for r in rows)
    timing = {"ms_model": sum(r["ms_model"] * len(r["t_start"]) for r in rows) / max(n_win, 1),
              "ms_pre": sum(r["ms_pre"] * len(r["t_start"]) for r in rows) / max(n_win, 1)}
    metrics = evaluate(rows)
    meta = {"cmd": " ".join(["python eval.py"] + sys.argv[1:]), "backend": args.backend,
            "model": str(args.onnx.name if args.backend == "onnx" else args.ckpt.name),
            "n_windows": n_win, "n_records": len(rows), "batch": args.batch,
            "threads": args.threads or os.cpu_count(),
            "cpu": f"{platform.processor() or platform.machine()}, {_cpu_name()}",
            "date": time.strftime("%Y-%m-%d")}
    out = args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "timing": timing, **metrics}, indent=1))
    md = tables_md(metrics, timing, meta)
    (out / "tables.md").write_text(md)
    print(md)


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.platform()


if __name__ == "__main__":
    main()
