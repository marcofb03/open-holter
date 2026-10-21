"""Evaluate ECGFounder (1 lead) as a window-level arrhythmia flagger on H10-like data.

Databases (PhysioNet, downloaded into --data on first use with
analysis/tools/fetch_physionet.py, git-ignored):
    mitdb     MIT-BIH Arrhythmia, 48 x 30 min, 360 Hz, lead MLII
    svdb      MIT-BIH Supraventricular Arrhythmia, 78 x 30 min, 128 Hz, lead ECG1
    incartdb  St Petersburg INCART 12-lead, 75 x 30 min, 257 Hz, lead I
Paced MIT-BIH records (102, 104, 107, 217) are left out, as in AAMI EC57.
MIT-BIH is split as in analysis/ (de Chazal 2004): DS1 is the development
set on which every threshold is chosen; DS2, SVDB and INCART are held out.

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
as negative). Windows touching ventricular flutter/fibrillation are left out
of the beat targets, which have no beat annotations there. Rhythm targets
need rhythm annotations, which INCART lacks.

Each target is scored with a label group from infer.GROUPS. The threshold
per target is the max-F1 point on MIT-BIH DS1. Windows overlap and come from
few patients, so 95 % CIs use a bootstrap over records, not windows.

With --with-rules the rule-based pipeline (analysis/ecgpipe) runs on the
same 130 Hz signal and gives window flags too (>= 1 V / S beat, >= 3 V / S in
a row, RR > 2 s, >= 50 % beats in an irregular rhythm), so rules, model and
their AND / OR combinations are compared on identical windows.

Usage (from ai/ecgfounder/):
    python eval.py --with-rules                      # all three databases, torch fp32
    python eval.py --db mitdb --records 100 119 208  # quick check
    python eval.py --backend onnx --onnx weights/ecgfounder_1lead_int8_static.onnx --tag int8_static
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
# de Chazal et al. 2004 inter-patient split, same as analysis/tools/eval_physionet.py.
DS1 = {"101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
       "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"}
DEV = "mitdb-DS1"
H10_FS = 130
EDGE_S = 0.3

N_CLASS = set("NLRejnB")
S_CLASS = set("AaJS")
V_CLASS = set("VE")
BEATS = N_CLASS | S_CLASS | V_CLASS | set("F/fQ")
# Rhythm annotation codes (MIT-BIH style). Bigeminy/trigeminy are sinus with ectopy.
SINUS_RHYTHMS = {"N", "SBR", "B", "T"}
NON_SINUS_RHYTHMS = {"AFIB", "AFL", "SVTA", "VT", "VFL", "NOD", "IVR", "AB"}
BEAT_TARGETS = ("pvc", "sve", "vrun", "svt", "pause")

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
# Rule-pipeline window score per target, and the score at which the rules flag it.
RULE_FLAG = {"pvc": 1, "sve": 1, "vrun": 3, "svt": 3, "pause": 2.0, "af": 0.5}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def ensure_db(db: str, data: Path) -> Path:
    """Download via analysis/tools/fetch_physionet.py (physionet.org, else its AWS mirror)."""
    d = data / db
    if not (d / "RECORDS").exists():
        sys.path.insert(0, str(REPO / "analysis" / "tools"))
        from fetch_physionet import fetch

        print(f"downloading {db} into {d} ...", file=sys.stderr)
        fetch(db, out=d)
        names = sorted(p.stem for p in d.glob("*.hea"))
        (d / "RECORDS").write_text("\n".join(names) + "\n")
    return d


def records(db: str, d: Path) -> list[str]:
    names = [r.strip() for r in (d / "RECORDS").read_text().split() if r.strip()]
    return [r for r in names if r not in DBS[db]["exclude"]]


def subset(db: str, rec: str) -> str:
    if db == "mitdb":
        return "mitdb-DS1" if rec in DS1 else "mitdb-DS2"
    return db


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


def vf_spans(ann, fs: float, n_samples: int) -> list[tuple[float, float]]:
    """Ventricular flutter/fibrillation: '[' ... ']' markers and (VFL rhythm."""
    sym = list(ann.symbol)
    t = np.asarray(ann.sample) / fs
    out = []
    for k, s in enumerate(sym):
        if s == "[":
            ends = [j for j in range(k + 1, len(sym)) if sym[j] == "]"]
            out.append((t[k], t[ends[0]] if ends else n_samples / fs))
    out += [(a, b) for a, b, r in rhythm_intervals(ann, fs, n_samples) if r == "VFL"]
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


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


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
    vf = vf_spans(ann, fs, n_samples)

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

        if any(overlap(t0, t1, a, b) > 0 for a, b in vf):
            for k in BEAT_TARGETS:
                out[k][i] = np.nan
    return out


def rule_scores(y130: np.ndarray, t_start: np.ndarray) -> dict[str, np.ndarray]:
    """Window scores from the rule-based beat pipeline (analysis/ecgpipe) on the 130 Hz signal."""
    sys.path.insert(0, str(REPO / "analysis"))
    from ecgpipe.beats import analyse_beats

    _, b = analyse_beats(y130, H10_FS)
    bt, lab = b.idx / H10_FS, np.asarray(b.label)
    irregular = np.asarray(b.irregular, bool)
    out = {k: np.zeros(len(t_start)) for k in RULE_FLAG}
    for i, t0 in enumerate(t_start):
        inside = (bt >= t0) & (bt < t0 + infer.WIN_S)
        li = lab[inside]
        out["pvc"][i] = (li == "V").sum()
        out["sve"][i] = (li == "S").sum()
        out["vrun"][i] = longest_run(li == "V")
        out["svt"][i] = longest_run(li == "S")
        rr = np.diff(bt[inside])
        out["pause"][i] = rr.max() if len(rr) else 0.0
        out["af"][i] = irregular[inside].mean() if inside.any() else 0.0
    return out


# --------------------------------------------------------------------------
# Inference with cache
# --------------------------------------------------------------------------

def predict_record(db: str, d: Path, rec_name: str, pred: infer.Predictor, cache: Path,
                   lead_override: str | None, with_rules: bool) -> dict:
    import wfdb

    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"{db}_{rec_name}.npz"
    rec = wfdb.rdrecord(str(d / rec_name))
    ann = wfdb.rdann(str(d / rec_name), "atr")
    lead = lead_override or DBS[db]["lead"]
    ch = rec.sig_name.index(lead) if lead in rec.sig_name else 0
    x = np.nan_to_num(rec.p_signal[:, ch])
    y = h10_like(x, rec.fs)
    if cf.exists():
        z = np.load(cf)
        t_start, probs = z["t_start"], z["probs"].astype(np.float64)
        ms_model, ms_pre = float(z["ms_model"]), float(z["ms_pre"])
    else:
        res = infer.run(y, H10_FS, pred)
        n = max(len(res.t_start), 1)
        t_start, probs = res.t_start, res.probs
        ms_model, ms_pre = 1000 * res.seconds_model / n, 1000 * res.seconds_preproc / n
        np.savez_compressed(cf, t_start=t_start, probs=probs.astype(np.float32),
                            ms_model=ms_model, ms_pre=ms_pre)
    rules = None
    if with_rules:
        rf = cache.parent / "rules" / f"{db}_{rec_name}_{rec.sig_name[ch]}.npz"
        if rf.exists():
            rules = dict(np.load(rf))
        else:
            rules = rule_scores(y, t_start)
            rf.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(rf, **rules)
    truth = window_truth(ann, rec.fs, rec.sig_len, t_start)
    return {"db": db, "set": subset(db, rec_name), "record": rec_name, "lead": rec.sig_name[ch],
            "t_start": t_start, "probs": probs, "rules": rules, "truth": truth,
            "ms_model": ms_model, "ms_pre": ms_pre}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def counts(y: np.ndarray, p: np.ndarray) -> dict:
    p = p.astype(bool)
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


def gather(rows: list[dict], target: str) -> dict[str, np.ndarray]:
    """Valid windows of `target`: truth y, model score s, rule score r (nan if absent), record id."""
    g = infer.GROUP_IDX[TARGETS[target]]
    y, s, r, rid = [], [], [], []
    for k, row in enumerate(rows):
        t = row["truth"][target]
        m = ~np.isnan(t)
        y.append(t[m])
        s.append(row["probs"][m][:, g].max(axis=1))
        rr = row["rules"][target][m] if row["rules"] is not None and target in row["rules"] else np.full(m.sum(), np.nan)
        r.append(rr)
        rid.append(np.full(m.sum(), k))
    cat = lambda a, dt=float: np.concatenate(a) if a else np.zeros(0, dt)  # noqa: E731
    return {"y": cat(y), "s": cat(s), "r": cat(r), "rid": cat(rid, int)}


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


def evaluate(rows: list[dict]) -> dict:
    sets = [s for s in ("mitdb-DS1", "mitdb-DS2", "svdb", "incartdb") if any(r["set"] == s for r in rows)]
    groups = {s: [r for r in rows if r["set"] == s] for s in sets}
    held = [r for r in rows if r["set"] != DEV]
    if held and len(sets) > 1:
        groups["held-out (DS2+SVDB+INCART)"] = held
    out = {"thresholds": {}, "by_set": {}, "rules_vs_model": {}}
    dev = [r for r in rows if r["set"] == DEV] or rows
    for target in TARGETS:
        g = gather(dev, target)
        out["thresholds"][target] = best_f1_threshold(g["y"], g["s"])
    for name, sub in groups.items():
        res, cmp_ = {}, {}
        for target in TARGETS:
            g = gather(sub, target)
            y, s, rid = g["y"], g["s"], g["rid"]
            thr = out["thresholds"][target]
            ok = 0 < y.sum() < len(y)
            lo, hi = bootstrap_auroc(y, s, rid) if ok else (float("nan"), float("nan"))
            res[target] = {"n": int(len(y)), "pos": int(y.sum()),
                           "pos_records": int(len(np.unique(rid[y == 1]))) if len(y) else 0,
                           "AUROC": auroc(y, s), "AUROC_lo": lo, "AUROC_hi": hi, "AUPRC": auprc(y, s),
                           "thr": thr, **counts(y, s >= thr)}
            if target in RULE_FLAG and len(y) and not np.isnan(g["r"]).any():
                pm, pr = s >= thr, g["r"] >= RULE_FLAG[target]
                cmp_[target] = {"n": int(len(y)), "pos": int(y.sum()),
                                "rules_AUROC": auroc(y, g["r"]),
                                "rules": counts(y, pr), "model": counts(y, pm),
                                "AND": counts(y, pr & pm), "OR": counts(y, pr | pm)}
        out["by_set"][name] = res
        if cmp_:
            out["rules_vs_model"][name] = cmp_
    return out


def fmt(v: float, nd: int = 3) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def tables_md(metrics: dict, timing: dict, meta: dict) -> str:
    lines = [f"<!-- generated by eval.py: {meta['cmd']} -->",
             f"Backend **{meta['backend']}** ({meta['model']}), {meta['n_windows']} windows from "
             f"{meta['n_records']} records, {meta['date']}. Thresholds: max F1 on {DEV}.", ""]
    for name, res in metrics["by_set"].items():
        lines += [f"#### {name}", "",
                  "| target (score group) | windows | pos (records) | AUROC [95% CI] | AUPRC | thr | Se | PPV | Sp |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for t, m in res.items():
            if m["n"] == 0 or m["pos"] == 0:
                continue
            ci = f"{fmt(m['AUROC'])} [{fmt(m['AUROC_lo'], 2)}–{fmt(m['AUROC_hi'], 2)}]"
            lines.append(f"| {t} ({TARGETS[t]}) | {m['n']} | {m['pos']} ({m['pos_records']}) | {ci} | "
                         f"{fmt(m['AUPRC'])} | {fmt(m['thr'], 2)} | {fmt(m['Se'])} | {fmt(m['PPV'])} | {fmt(m['Sp'])} |")
        lines.append("")
    for name, cmp_ in metrics["rules_vs_model"].items():
        lines += [f"#### Rules vs model, {name} (Se / PPV per window)", "",
                  "| target | pos / windows | rules AUROC | rules | model | rules AND model | rules OR model |",
                  "|---|---:|---:|---|---|---|---|"]
        for t, c in cmp_.items():
            if c["pos"] == 0:
                continue
            sp = lambda k: f"{fmt(c[k]['Se'], 2)} / {fmt(c[k]['PPV'], 2)}"  # noqa: E731
            lines.append(f"| {t} | {c['pos']} / {c['n']} | {fmt(c['rules_AUROC'])} | {sp('rules')} | {sp('model')} | "
                         f"{sp('AND')} | {sp('OR')} |")
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
    ap.add_argument("--tag", default="fp32", help="name for cache and results (e.g. fp32, int8_static)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--incart-lead", default=None, help="INCART lead to use (default I)")
    ap.add_argument("--with-rules", action="store_true", help="also score the analysis/ rule pipeline")
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
            r = predict_record(db, d, rec, pred, cache, args.incart_lead if db == "incartdb" else None,
                               args.with_rules)
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
