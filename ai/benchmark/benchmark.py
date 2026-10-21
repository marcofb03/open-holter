"""Head-to-head of the open single-lead options on H10-like data.

Systems, all fed the same 130 Hz signal (Polar H10 rate) of the same lead:
    rules       analysis/ecgpipe (merged rule pipeline): detector, noise gate, N/S/V/Q labels
    heartkit    Ambiq heartKIT BEAT-3-EFF-SM on the beats the rules' detector found
                (rules' noise gate kept: unreadable beats stay Q), ARR-2-EFF-SM for AF
    ecgfounder  ECGFounder 1-lead, 10 s windows (5 s step), thresholds from MIT-BIH DS1
    ensembles   combinations declared in ENSEMBLES, chosen on DS1 only

Levels:
    beats    AAMI EC57 scoring exactly as analysis/tools/eval_physionet.py (gross Se/PPV
             of V and S beats, V runs found, false V runs per hour); ECGFounder has
             no beat output and is not in this table
    windows  10 s windows (5 s step) as in ai/ecgfounder/eval.py: >=1 V, V run >=3,
             >=1 S, AF; Se/PPV at each system's operating point
    events   "critical findings" per record, for an indicative comparison with
             DeepRhythmAI (Johnson et al., Nature Medicine 2025): VT/NSVT (V run >= 3),
             sustained SVT (>= 30 s), AF/AFL >= 30 s, pause > 3 s. Per-patient
             sensitivity = share of records with a reference finding in which the
             system found at least one; false findings = detected events that
             overlap no reference finding, scaled to 1,000 patient-days.

Leads: MIT-BIH MLII, SVDB ECG1, INCART II (as analysis/VALIDATION.md). MIT-BIH DS1
is the development set (every threshold and ensemble choice), DS2/SVDB/INCART are
held out. ECGFounder window probabilities come from ai/ecgfounder/eval.py's cache
(`--tag fp32` for MIT-BIH/SVDB, `--tag fp32_leadII --incart-lead II` for INCART).

Usage (from ai/benchmark/):  python benchmark.py   -> results/tables.md, results/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "analysis"), str(REPO / "analysis" / "tools"),
                str(REPO / "ai" / "ecgfounder"), str(REPO / "ai" / "heartkit")]

DATA = REPO / "data"
CACHE = DATA / "work" / "benchmark"
FS = 130
LEAD = {"mitdb": "MLII", "svdb": "ECG1", "incartdb": "II"}
ECGF_TAG = {"mitdb": "fp32", "svdb": "fp32", "incartdb": "fp32_leadII"}
SETS = ("mitdb-DS1", "mitdb-DS2", "svdb", "incartdb")
HELD = "held-out"
CRIT_S = 30.0       # sustained SVT / AF episode length
PAUSE_S = 3.0
TOL = 1.0           # event overlap tolerance (s)


# --------------------------------------------------------------------------
# Per-record front end (cached)
# --------------------------------------------------------------------------

def run_record(job: tuple[str, str]) -> Path:
    import wfdb
    from ecgpipe.beats import analyse_beats
    from ecgpipe.io import load_ecglogger_csv, write_ecglogger_csv
    from scipy.signal import resample_poly

    import hk

    db, rec = job
    out = CACHE / f"{db}_{rec}.pkl"
    if out.exists():
        return out
    r = wfdb.rdrecord(str(DATA / db / rec))
    ann = wfdb.rdann(str(DATA / db / rec), "atr")
    ch = r.sig_name.index(LEAD[db]) if LEAD[db] in r.sig_name else 0
    ratio = Fraction(FS, int(round(r.fs))).limit_denominator(1000)  # as eval_physionet
    y = resample_poly(np.nan_to_num(r.p_signal[:, ch]), ratio.numerator, ratio.denominator)
    with tempfile.TemporaryDirectory() as tmp:  # CSV round trip, like a real recording
        p = Path(tmp) / "x.csv"
        write_ecglogger_csv(p, 1.7e9 + np.arange(len(y)) / FS, y)
        ecg = load_ecglogger_csv(p).ecg
    _, b = analyse_beats(ecg, FS)
    lab, prob = hk.classify_beats(ecg, FS, b.idx)
    lab = np.where(b.noisy | (b.label == "Q"), "Q", lab)  # shared noise gate
    t_af, p_af = hk.af_frames(ecg, FS)
    res = {"db": db, "record": rec, "lead": r.sig_name[ch], "fs_in": r.fs, "dur_s": len(ecg) / FS,
           "ann": {"sample": np.asarray(ann.sample), "symbol": list(ann.symbol), "aux_note": list(ann.aux_note)},
           "b": b, "hk_label": lab, "hk_prob": prob, "af_t": t_af, "af_p": p_af}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(res))
    return out


def load_all(jobs: list[tuple[str, str]], n_jobs: int) -> list[dict]:
    with ProcessPoolExecutor(n_jobs) as ex:
        paths = list(ex.map(run_record, jobs))
    rows = []
    for p in paths:
        r = pickle.loads(p.read_bytes())
        r["set"] = ev_subset(r["db"], r["record"])
        z = DATA / "work" / "ecgfounder" / ECGF_TAG[r["db"]] / f"{r['db']}_{r['record']}.npz"
        if z.exists():
            e = np.load(z)
            r["ef_t"], r["ef_p"] = e["t_start"], e["probs"].astype(np.float64)
        rows.append(r)
    return rows


def ev_subset(db: str, rec: str) -> str:
    import eval as ev

    return ev.subset(db, rec)


class Ann:  # minimal wfdb.Annotation stand-in for ai/ecgfounder/eval.py helpers
    def __init__(self, d: dict):
        self.sample, self.symbol, self.aux_note = d["sample"], d["symbol"], d["aux_note"]


# --------------------------------------------------------------------------
# Beat level: AAMI scoring as analysis/tools/eval_physionet.py
# --------------------------------------------------------------------------

def beat_score(r: dict, labels: np.ndarray) -> dict:
    from ecgpipe.events import find_events
    from eval_physionet import AAMI, _match, _overlaps, _ref_runs

    b = replace(r["b"], label=np.asarray(labels))
    events = find_events(b, FS)
    dur = r["dur_s"]
    det_t = b.idx / FS
    sym = np.asarray(r["ann"]["symbol"])
    ann_t = r["ann"]["sample"] / r["fs_in"]
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
    ref_of_det = np.full(len(det_t), "-", dtype="<U1")
    ref_of_det[m[m >= 0]] = ref_c[m >= 0]
    det_lab = np.where(m >= 0, b.label[np.clip(m, 0, None)], "-")
    res = {"dur_s": int(dur)}
    for c, ignore in (("V", {"F", "Q"}), ("S", {"Q"})):
        ref_pos = ref_c == c
        det_c = det_keep & (b.label == c)
        res[f"{c}_ref"] = int(ref_pos.sum())
        res[f"{c}_TP"] = int((ref_pos & (det_lab == c)).sum())
        res[f"{c}_FP"] = int((det_c & (ref_of_det != c) & ~np.isin(ref_of_det, list(ignore))).sum())
        rr = [(ref_t[a], ref_t[z]) for a, z in _ref_runs(ref_c, c)]
        dr = [(e.t_start, e.t_end) for e in events if e.kind == f"{c} run (>=3)"
              and 1.0 < e.t_start < dur - 1.0 and not in_vf([e.t_start])[0]]
        res[f"{c}run_ref"] = len(rr)
        res[f"{c}run_found"] = sum(_overlaps(a, z, dr) for a, z in rr)
        res[f"{c}run_det"] = len(dr)
        res[f"{c}run_true"] = sum(_overlaps(a, z, rr) for a, z in dr)
    return res


def beat_metrics(rows: list[dict]) -> dict:
    g = {k: sum(r[k] for r in rows) for k in rows[0]}
    f = lambda a, b: a / b if b else float("nan")  # noqa: E731
    h = g["dur_s"] / 3600
    return {"V_ref": g["V_ref"], "V_Se": f(g["V_TP"], g["V_ref"]), "V_PPV": f(g["V_TP"], g["V_TP"] + g["V_FP"]),
            "S_ref": g["S_ref"], "S_Se": f(g["S_TP"], g["S_ref"]), "S_PPV": f(g["S_TP"], g["S_TP"] + g["S_FP"]),
            "Vrun_ref": g["Vrun_ref"], "Vrun_found": g["Vrun_found"],
            "Vrun_false_per_h": f(g["Vrun_det"] - g["Vrun_true"], h), "hours": h}


# --------------------------------------------------------------------------
# Beat labels of each system / ensemble
# --------------------------------------------------------------------------

def combine(a: np.ndarray, b: np.ndarray, how: str) -> np.ndarray:
    """Per-beat label fusion of two N/S/V/Q label arrays."""
    out = np.where((a == "Q") | (b == "Q"), "Q", "N").astype("<U1")
    for c in ("V", "S"):
        hit = (a == c) & (b == c) if how == "and" else (a == c) | (b == c)
        out[hit & (out != "Q") & (out == "N")] = c
    return out


BEAT_SYSTEMS = {
    "rules": lambda r: r["b"].label,
    "heartkit": lambda r: r["hk_label"],
    "rules AND heartkit": lambda r: combine(r["b"].label, r["hk_label"], "and"),
    "rules OR heartkit": lambda r: combine(r["b"].label, r["hk_label"], "or"),
}


# --------------------------------------------------------------------------
# Window level
# --------------------------------------------------------------------------

def longest(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def window_rows(r: dict) -> dict:
    """Truth and per-system window scores on ECGFounder's window grid."""
    import eval as ev

    t0s = r["ef_t"]
    truth = ev.window_truth(Ann(r["ann"]), r["fs_in"], int(r["dur_s"] * r["fs_in"]), t0s)
    bt = r["b"].idx / FS
    sc = {}
    for name, fn in BEAT_SYSTEMS.items():
        lab = np.asarray(fn(r))
        s = {k: np.zeros(len(t0s)) for k in ("pvc", "vrun", "sve")}
        for i, t0 in enumerate(t0s):
            li = lab[(bt >= t0) & (bt < t0 + 10)]
            s["pvc"][i], s["sve"][i], s["vrun"][i] = (li == "V").sum(), (li == "S").sum(), longest(li == "V")
        sc[name] = s
    irr = np.asarray(r["b"].irregular, bool)
    sc["rules"]["af"] = np.array([irr[(bt >= t0) & (bt < t0 + 10)].mean() if ((bt >= t0) & (bt < t0 + 10)).any()
                                  else 0.0 for t0 in t0s])
    inside = lambda t0: (r["af_t"] >= t0) & (r["af_t"] + 5.12 <= t0 + 10 + 1e-9)  # noqa: E731
    sc["heartkit"]["af"] = np.array([r["af_p"][inside(t0)].mean() if inside(t0).any() else 0.0 for t0 in t0s])
    import infer

    g = infer.GROUP_IDX
    sc["ecgfounder"] = {"pvc": r["ef_p"][:, g["pvc"]].max(1), "vrun": r["ef_p"][:, g["vt_or_wct"]].max(1),
                        "sve": r["ef_p"][:, g["pac"]].max(1), "af": r["ef_p"][:, g["af_or_afl"]].max(1)}
    return {"truth": truth, "scores": sc}


# Operating points. Rules/heartKIT: count thresholds; ECGFounder: max-F1 on DS1 (filled at run time).
OP = {"pvc": 1, "sve": 1, "vrun": 3, "af_rules": 0.5, "af_heartkit": 0.75}


def window_eval(rows: list[dict], wins: list[dict]) -> dict:
    import eval as ev

    out = {"thr_ecgfounder": {}}
    dev = [w for r, w in zip(rows, wins) if r["set"] == "mitdb-DS1"]
    for t in ("pvc", "vrun", "sve", "af"):
        y, s = _stack(dev, t, "ecgfounder")
        out["thr_ecgfounder"][t] = ev.best_f1_threshold(y, s)
    res = {}
    for name in SETS + (HELD,):
        sub = [w for r, w in zip(rows, wins) if (r["set"] == name or (name == HELD and r["set"] != "mitdb-DS1"))]
        res[name] = {}
        for t in ("pvc", "vrun", "sve", "af"):
            res[name][t] = {}
            for sysname in list(BEAT_SYSTEMS) + ["ecgfounder"]:
                y, s = _stack(sub, t, sysname)
                if s is None or not len(y) or y.sum() == 0:
                    continue
                if sysname == "ecgfounder":
                    thr = out["thr_ecgfounder"][t]
                elif t == "af":
                    thr = OP.get(f"af_{sysname}")
                    if thr is None:
                        continue
                else:
                    thr = OP[t]
                res[name][t][sysname] = {"n": int(len(y)), "pos": int(y.sum()), "AUROC": ev.auroc(y, s),
                                         **ev.counts(y, s >= thr)}
    out["by_set"] = res
    return out


def _stack(wins: list[dict], target: str, sysname: str):
    ys, ss = [], []
    for w in wins:
        if target not in w["scores"].get(sysname, {}):
            return None, None
        y = w["truth"][target]
        m = ~np.isnan(y)
        ys.append(y[m])
        ss.append(w["scores"][sysname][target][m])
    if not ys:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ys), np.concatenate(ss)


# --------------------------------------------------------------------------
# Event level: critical findings per record
# --------------------------------------------------------------------------

def merge(iv: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def ref_events(r: dict) -> dict:
    """Reference critical findings, plus 'allowed' spans where a detection is not false."""
    import eval as ev
    from eval_physionet import AAMI, _ref_runs

    a = Ann(r["ann"])
    fs, n = r["fs_in"], int(r["dur_s"] * r["fs_in"])
    sym = np.asarray(a.symbol)
    t = np.asarray(a.sample) / fs
    isb = np.isin(sym, list(AAMI))
    bt, bc = t[isb], np.array([AAMI[s] for s in sym[isb]])
    rhythm = ev.rhythm_intervals(a, fs, n)
    vf = ev.vf_spans(a, fs, n)
    has_rhythm = bool(rhythm)
    vruns = [(bt[i], bt[j]) for i, j in _ref_runs(bc, "V")] + [(x, y) for x, y, k in rhythm if k == "VT"]
    sruns = [(bt[i], bt[j]) for i, j in _ref_runs(bc, "S")] + [(x, y) for x, y, k in rhythm if k == "SVTA"]
    af = [(x, y) for x, y, k in rhythm if k in ev.AF_RHYTHMS]
    good = ~np.isin(bc, ["Q"])
    rr = np.diff(bt)
    pauses = [(bt[i], bt[i + 1]) for i in np.where((rr > PAUSE_S) & good[:-1] & good[1:])[0]
              if not any(x <= bt[i] <= y for x, y in vf)]
    return {
        "vt": {"ref": merge(vruns), "ok": merge(vruns + vf), "assessable": True},
        "svt": {"ref": [e for e in merge(sruns) if e[1] - e[0] >= CRIT_S], "ok": merge(sruns + af),
                "assessable": has_rhythm},
        "af": {"ref": [e for e in merge(af) if e[1] - e[0] >= CRIT_S], "ok": merge(af), "assessable": has_rhythm},
        "pause": {"ref": pauses, "ok": merge(pauses + [(x, y) for x, y in zip(bt[:-1], bt[1:]) if y - x > 2.0]),
                  "assessable": True},
    }


def beat_events(r: dict, labels: np.ndarray) -> dict:
    """Critical events from per-beat labels (shared detector)."""
    bt = r["b"].idx / FS
    lab = np.asarray(labels)
    out = {"vt": [], "svt": []}
    for key, c, min_s in (("vt", "V", 0.0), ("svt", "S", CRIT_S)):
        i = 0
        while i < len(lab):
            if lab[i] == c:
                j = i
                while j + 1 < len(lab) and lab[j + 1] == c:
                    j += 1
                if j - i + 1 >= 3 and bt[j] - bt[i] >= min_s:
                    out[key].append((bt[i], bt[j]))
                i = j + 1
            else:
                i += 1
    good = lab != "Q"
    rr = np.diff(bt)
    out["pause"] = [(bt[i], bt[i + 1]) for i in np.where((rr > PAUSE_S) & good[:-1] & good[1:])[0]]
    return out


def episodes(t: np.ndarray, flag: np.ndarray, length: float, min_s: float = 0.0) -> list[tuple[float, float]]:
    ep = merge([(a, a + length) for a, f in zip(t, flag) if f])
    return [e for e in ep if e[1] - e[0] >= min_s]


def system_events(r: dict, thr: dict) -> dict:
    import infer

    g = infer.GROUP_IDX
    bt = r["b"].idx / FS
    irr = np.asarray(r["b"].irregular, bool)
    ev = {}
    ev["rules"] = beat_events(r, r["b"].label)
    ev["rules"]["af"] = [e for e in merge([(bt[i], bt[i + 1]) for i in np.where(irr[:-1])[0]])
                         if e[1] - e[0] >= CRIT_S]
    ev["heartkit"] = beat_events(r, r["hk_label"])
    ev["heartkit"]["af"] = episodes(r["af_t"], r["af_p"] >= OP["af_heartkit"], 5.12, CRIT_S)
    if "ef_p" in r:
        p, t = r["ef_p"], r["ef_t"]
        ev["ecgfounder"] = {
            "vt": episodes(t, p[:, g["vt_or_wct"]].max(1) >= thr["vrun"], 10.0),
            "svt": episodes(t, p[:, g["svt"]].max(1) >= thr["svt"], 10.0, CRIT_S),
            "af": episodes(t, p[:, g["af_or_afl"]].max(1) >= thr["af"], 10.0, CRIT_S),
            "pause": episodes(t, p[:, g["pause_or_block"]].max(1) >= thr["pause"], 10.0),
        }
        # Ensemble: heartKIT + rules beats (OR) for V runs and pauses from the shared
        # detector, each V run confirmed by ECGFounder (pvc or VT/WCT group >= its
        # DS1 confirm threshold in an overlapping window); AF and SVT from ECGFounder.
        both = combine(r["b"].label, r["hk_label"], "or")
        be = beat_events(r, both)
        conf = np.maximum(p[:, g["pvc"]].max(1), p[:, g["vt_or_wct"]].max(1))
        keep = [e for e in be["vt"] if any(conf[(t <= e[1]) & (t + 10 >= e[0])] >= thr["confirm"])]
        ev["ensemble"] = {"vt": keep, "svt": ev["ecgfounder"]["svt"], "af": ev["ecgfounder"]["af"],
                          "pause": be["pause"]}
    return ev


def overlaps(e: tuple[float, float], spans: list[tuple[float, float]]) -> bool:
    return any(e[0] - TOL <= b and a - TOL <= e[1] for a, b in spans)


def event_eval(rows: list[dict], thr: dict) -> dict:
    per = []
    for r in rows:
        ref = ref_events(r)
        sysev = system_events(r, thr)
        for sysname, evs in sysev.items():
            for k, d in evs.items():
                if not ref[k]["assessable"]:
                    continue
                found = [overlaps(e, d) for e in ref[k]["ref"]]
                false = [e for e in d if not overlaps(e, ref[k]["ok"])]
                per.append({"set": r["set"], "record": r["record"], "system": sysname, "type": k,
                            "n_ref": len(ref[k]["ref"]), "n_found": int(sum(found)), "n_false": len(false),
                            "hours": r["dur_s"] / 3600})
    out = {}
    for name in SETS + (HELD,):
        sub = [p for p in per if p["set"] == name or (name == HELD and p["set"] != "mitdb-DS1")]
        out[name] = {}
        for sysname in sorted({p["system"] for p in sub}):
            for k in ("vt", "svt", "af", "pause"):
                q = [p for p in sub if p["system"] == sysname and p["type"] == k]
                if not q:
                    continue
                pts = [p for p in q if p["n_ref"] > 0]
                hours = sum(p["hours"] for p in q)
                nf = sum(p["n_false"] for p in q)
                out[name].setdefault(sysname, {})[k] = {
                    "patients_with": len(pts),
                    "patient_Se": (sum(p["n_found"] > 0 for p in pts) / len(pts)) if pts else float("nan"),
                    "episodes": sum(p["n_ref"] for p in q),
                    "episode_Se": (sum(p["n_found"] for p in q) / max(sum(p["n_ref"] for p in q), 1)),
                    "false": nf, "hours": hours,
                    "false_per_1000_pt_days": nf / hours * 24 * 1000 if hours else float("nan"),
                }
    return out


# --------------------------------------------------------------------------
# Thresholds for ECGFounder events (DS1 only)
# --------------------------------------------------------------------------

def ecgf_thresholds(rows: list[dict], wins: list[dict], wthr: dict) -> dict:
    import eval as ev

    dev = [w for r, w in zip(rows, wins) if r["set"] == "mitdb-DS1"]
    thr = {"vrun": wthr["vrun"], "af": wthr["af"]}
    # svt / pause groups: max-F1 on DS1 windows from ai/ecgfounder/eval.py's targets
    import infer

    for t, grp in (("svt", "svt"), ("pause", "pause_or_block")):
        ys, ss = [], []
        for r, w in zip(rows, wins):
            if r["set"] != "mitdb-DS1":
                continue
            y = w["truth"][t]
            m = ~np.isnan(y)
            ys.append(y[m])
            ss.append(r["ef_p"][m][:, infer.GROUP_IDX[grp]].max(1))
        thr[t] = ev.best_f1_threshold(np.concatenate(ys), np.concatenate(ss))
    # Confirmation threshold for the ensemble: the 5th percentile of the ECGFounder
    # score over DS1 windows that contain a reference V run (keeps ~95 % of them).
    conf = []
    for r, w in zip(rows, wins):
        if r["set"] != "mitdb-DS1":
            continue
        y = w["truth"]["vrun"] == 1
        s = np.maximum(w["scores"]["ecgfounder"]["pvc"], w["scores"]["ecgfounder"]["vrun"])
        conf += list(s[y])
    thr["confirm"] = float(np.percentile(conf, 5)) if conf else 0.0
    del dev
    return thr


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def report(beats: dict, wins: dict, events: dict, thr: dict) -> str:
    L = ["<!-- generated by ai/benchmark/benchmark.py -->", ""]
    L += ["### Beats (AAMI EC57 gross, same detector)", "",
          "| set | system | V Se | V PPV | S Se | S PPV | V runs found / ref | false V runs / h |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, by in beats.items():
        for s, m in by.items():
            L.append(f"| {name} | {s} | {fmt(m['V_Se'])} | {fmt(m['V_PPV'])} | {fmt(m['S_Se'])} | {fmt(m['S_PPV'])} | "
                     f"{m['Vrun_found']} / {m['Vrun_ref']} | {fmt(m['Vrun_false_per_h'], 1)} |")
    L += ["", "### 10 s windows (Se / PPV at the operating point; AUROC of the score)", "",
          "| set | target | " + " | ".join(list(BEAT_SYSTEMS) + ["ecgfounder"]) + " |",
          "|---|---|" + "---|" * (len(BEAT_SYSTEMS) + 1)]
    for name, by in wins["by_set"].items():
        for t, d in by.items():
            if not d:
                continue
            pos = next(iter(d.values()))["pos"]
            cells = []
            for s in list(BEAT_SYSTEMS) + ["ecgfounder"]:
                m = d.get(s)
                cells.append("–" if m is None else f"{fmt(m['Se'], 2)} / {fmt(m['PPV'], 2)} ({fmt(m['AUROC'], 2)})")
            L.append(f"| {name} | {t} ({pos} pos) | " + " | ".join(cells) + " |")
    L += ["", "### Critical findings per record", "",
          "| set | system | finding | patients with | per-patient Se | episodes | episode Se | false | false / 1,000 patient-days |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, by in events.items():
        for s, d in by.items():
            for k, m in d.items():
                L.append(f"| {name} | {s} | {k} | {m['patients_with']} | {fmt(m['patient_Se'])} | {m['episodes']} | "
                         f"{fmt(m['episode_Se'])} | {m['false']} | {fmt(m['false_per_1000_pt_days'], 0)} |")
    L += ["", f"ECGFounder thresholds (DS1): {json.dumps({k: round(v, 3) for k, v in thr.items()})}", ""]
    return "\n".join(L)


def main() -> None:
    import eval as ev

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--db", nargs="+", default=list(LEAD))
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    jobs = [(db, rec) for db in args.db for rec in ev.records(db, ev.ensure_db(db, DATA))]
    rows = load_all(jobs, args.jobs)
    missing = [f"{r['db']}/{r['record']}" for r in rows if "ef_p" not in r]
    if missing:
        sys.exit(f"ECGFounder cache missing for {len(missing)} records, e.g. {missing[:3]}; run ai/ecgfounder/eval.py first")

    beats = {}
    for name in SETS + (HELD,):
        sub = [r for r in rows if r["set"] == name or (name == HELD and r["set"] != "mitdb-DS1")]
        if sub:
            beats[name] = {s: beat_metrics([beat_score(r, fn(r)) for r in sub]) for s, fn in BEAT_SYSTEMS.items()}
    wins = [window_rows(r) for r in rows]
    wres = window_eval(rows, wins)
    thr = ecgf_thresholds(rows, wins, wres["thr_ecgfounder"])
    events = event_eval(rows, thr)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps({"beats": beats, "windows": wres, "events": events,
                                                       "thresholds": thr}, indent=1, default=float))
    md = report(beats, wres, events, thr)
    (args.out / "tables.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
