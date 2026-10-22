"""Sensitivity-first router: per finding, the system / combination / threshold to use.

For every finding of findings.py (VT/NSVT, AF/AFL, sustained SVT, frequent PVC,
frequent PAC, pause, brady, tachy) the router considers:
    atoms     one system at one threshold (ATOMS below): rules (analysis/ecgpipe),
              heartKIT, ECGFounder, CLEF-Medium + head (ai/clef/head.py)
    or        the union of 2-4 systems' detections (each at one of 3 thresholds)
    vote2     detections confirmed by a second system (>= 2 systems agree)
    confirm   system A's detections kept only where system B also fires
It is chosen on the tuning split only (datasets.py), by
    1. per-patient sensitivity, capped at TARGET (98 %): every candidate that
       reaches the target (or the best sensitivity reached, if none does) ties;
    2. then the fewest false findings per day on the tuning records;
    3. then the higher episode sensitivity, then the fewer systems.
The choice is then scored once on the held-out split: per-patient Se with a
95 % Wilson interval, episode Se, PPV, false findings per day on all held-out
records and on the long (>= 10 h) recordings, and a small operating-point table
(the tuning Pareto front of Se vs false/day, with held-out numbers).

Usage (from ai/benchmark/, after frontend.py and ../clef/head.py):
    python router.py          # -> results/router.md, results/router.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "ai" / "ecgfounder")]
import datasets as ds  # noqa: E402
import findings as F  # noqa: E402
import frontend as fe  # noqa: E402

TARGET = 0.98
FS = fe.FS
W = F.WIN_S
CRITICAL = ("vt", "af", "svt", "pause", "brady")  # for the DeepRhythmAI-style summary
ECGF_GROUP = {"vt": "vt_or_wct", "af": "af_or_afl", "svt": "svt", "pvc": "pvc", "pac": "pac",
              "pause": "pause_or_block", "brady": "sinus_brady", "tachy": "sinus_tachy"}
GAP = {"af": W, "svt": W}  # window-based rhythm episodes bridge one missed window
P_GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
NAMES = {"rules": "rules", "heartkit": "heartKIT", "ecgfounder": "ECGFounder", "clef": "CLEF"}


# --------------------------------------------------------------------------
# Per-record inputs
# --------------------------------------------------------------------------

def prepare(r: ds.Rec) -> dict | None:
    import infer

    if not fe.cache_path(r).exists():
        return None
    d = fe.load(r)
    if r.db != "icentia11k":  # references from the local annotations (picks up later fixes, e.g. '~' spans)
        rr_ = r if r.db != "nstdb" else ds.Rec("mitdb", r.rec[:3], r.rec[:3], "noise")
        d["ref"] = ds.reference(rr_, d["fs_in"], int(round(d["dur_s"] * d["fs_in"])))
    hp = fe.CACHE / "clef_head" / r.db / f"{r.rec}.npy"
    ch = np.load(hp).astype(np.float32) if hp.exists() else None
    rr = d["rules"]
    bt = rr["idx"] / FS
    q = (rr["label"] == "Q") | rr["noisy"]
    wt = d["win_t"].astype(np.float64)
    ep = d["ecgf"].astype(np.float32)
    # rules' AF score per window: share of the window's beats in an irregular rhythm
    w = np.floor(bt / W).astype(int)
    n_w = len(wt)
    cnt = np.bincount(w[w < n_w], minlength=n_w)
    irr = np.bincount(w[w < n_w], weights=rr["irregular"][w < n_w].astype(float), minlength=n_w)
    row = {
        "key": f"{r.db}/{r.rec}", "db": r.db, "patient": r.patient, "split": r.split, "subset": ds.subset(r),
        "hours": d["dur_s"] / 3600, "long": r.db in ds.LONG_DBS, "rf": F.ref_findings(d),
        "bt": bt, "lab": rr["label"], "q": q, "hk_lab": d["hk"]["label"],
        "pV": d["hk"]["prob"][:, 2].astype(np.float32), "pS": d["hk"]["prob"][:, 1].astype(np.float32),
        "af_t": d["hk"]["af_t"].astype(np.float64), "af_p": d["hk"]["af_p"].astype(np.float32),
        "wt": wt, "ecgfounder": {f: ep[:, infer.GROUP_IDX[g]].max(1) for f, g in ECGF_GROUP.items()},
        "clef": None if ch is None else {f: ch[:, i] for i, f in enumerate(F.HEAD_TARGETS)},
        "rules_af": np.where(cnt > 0, irr / np.maximum(cnt, 1), 0.0),
        "hr": F.running_hr(bt, ~q),
    }
    return row


# --------------------------------------------------------------------------
# Atoms: one system at one threshold -> detections (n, 2) or burden flag inputs
# --------------------------------------------------------------------------

def merge_np(iv: np.ndarray, gap: float = 0.0) -> np.ndarray:
    if len(iv) == 0:
        return iv.reshape(0, 2)
    iv = iv[np.argsort(iv[:, 0])]
    ends = np.maximum.accumulate(iv[:, 1])
    new = np.r_[True, iv[1:, 0] > ends[:-1] + gap]
    g = np.cumsum(new) - 1
    out = np.zeros((g[-1] + 1, 2))
    out[:, 0] = iv[new, 0]
    np.maximum.at(out[:, 1], g, iv[:, 1])
    return out


def _win_eps(t0: np.ndarray, flag: np.ndarray, length: float, f: str) -> np.ndarray:
    iv = np.stack([t0[flag], t0[flag] + length], 1) if flag.any() else np.zeros((0, 2))
    iv = merge_np(iv, GAP.get(f, 0.0))
    return iv[iv[:, 1] - iv[:, 0] >= F.MIN_S[f]] if len(iv) else iv


def _beat_runs(bt: np.ndarray, mask: np.ndarray, f: str) -> np.ndarray:
    iv = np.asarray([(bt[i], bt[j]) for i, j in F.runs(mask, 3)], np.float64).reshape(-1, 2)
    return iv[iv[:, 1] - iv[:, 0] >= F.MIN_S[f]] if len(iv) else iv


def _rate(row: dict, cond, min_s: float) -> np.ndarray:
    return F.as_array(F.rate_stretches(row["bt"], row["hr"], cond, min_s))


@dataclass(frozen=True)
class Atom:
    fam: str      # system family
    var: str      # variant
    thr: float    # score threshold; pvc/pac: estimated burden at which the patient is flagged

    def label(self) -> str:
        v = f" {self.var}" if self.var else ""
        if self.var.startswith("burden"):
            return f"{NAMES[self.fam]}{v} ≥ {100 * self.thr:g} %"
        return f"{NAMES[self.fam]}{v}" if self.thr == NOTHR else f"{NAMES[self.fam]}{v} ≥ {self.thr:g}"


def atom_dets(row: dict, f: str, a: Atom):
    """Episode findings -> (n, 2) intervals; burden findings -> (k, n) estimate."""
    bt, lab, q = row["bt"], row["lab"], row["q"]
    if f in F.BURDEN:  # (count, total): beats, or 10 s windows with probability >= 0.5
        c = F.BURDEN[f]
        if a.fam in ("rules", "heartkit"):
            if a.fam == "rules":
                m = lab == c
            elif a.var == "burden":
                m = row["hk_lab"] == c
            elif a.var == "burden, AND rules":
                m = (lab == c) & (row["hk_lab"] == c)
            else:  # OR rules
                m = (lab == c) | (row["hk_lab"] == c)
            return float(m.sum()), float((~q).sum())
        s = row[a.fam][f] if row[a.fam] is not None else np.zeros(len(row["wt"]))
        return float((s >= 0.5).sum()), float(len(s))
    if a.fam in ("ecgfounder", "clef"):
        if row[a.fam] is None:
            return np.zeros((0, 2))
        return _win_eps(row["wt"], row[a.fam][f] >= a.thr, W, f)
    if f == "vt":
        if a.fam == "rules" and a.var == "":
            return _beat_runs(bt, lab == "V", f)
        if a.fam == "rules" and a.var == "unreadable":  # VF is often gated as noise
            iv = np.asarray([(bt[max(i - 1, 0)], bt[j]) for i, j in F.runs(q, 2)]).reshape(-1, 2)
            return iv[iv[:, 1] - iv[:, 0] >= a.thr] if len(iv) else iv
        m = (row["pV"] >= a.thr) & ~q
        return _beat_runs(bt, m | (lab == "V") if a.var == "OR rules" else m, f)
    if f == "af":
        if a.fam == "rules":
            return _win_eps(row["wt"], row["rules_af"] >= a.thr, W, f)
        return _win_eps(row["af_t"], row["af_p"] >= a.thr, 5.12, f)
    if f == "svt":
        if a.fam == "rules" and a.var == "rate >":
            return _rate(row, lambda h: h > a.thr, F.MIN_S[f])
        if a.fam == "rules":
            return _beat_runs(bt, lab == "S", f)
        return _beat_runs(bt, (row["pS"] >= a.thr) & ~q, f)
    if f == "pause":
        good = np.ones(len(bt), bool) if a.var == "all beats" else ~q
        return F.as_array(F.pauses(bt, good, a.thr))
    if f == "brady":
        return _rate(row, lambda h: h < a.thr, F.MIN_S[f])
    if f == "tachy":
        return _rate(row, lambda h: h > a.thr, F.MIN_S[f])
    raise ValueError((f, a))


NAN = float("nan")
NOTHR = -1.0  # atom without a threshold (NaN would never compare equal, breaking caches)


def atoms_for(f: str) -> dict[str, list[Atom]]:
    """Per system family, its atoms ordered from most to least sensitive."""
    fm = {"ecgfounder": [Atom("ecgfounder", "", t) for t in P_GRID],
          "clef": [Atom("clef", "", t) for t in P_GRID]}
    if f in F.BURDEN:
        g = BURDEN_GRID
        return {"rules": [Atom("rules", "burden", b) for b in g["beats"]],
                "heartkit": [Atom("heartkit", v, b) for v in ("burden, OR rules", "burden", "burden, AND rules")
                             for b in g["beats"]],
                "ecgfounder": [Atom("ecgfounder", "burden of windows", b) for b in g["windows"]],
                "clef": [Atom("clef", "burden of windows", b) for b in g["windows"]]}
    if f == "vt":
        fm["rules"] = [Atom("rules", "unreadable", 5.0), Atom("rules", "", NOTHR)]
        fm["heartkit"] = ([Atom("heartkit", "OR rules", t) for t in (0.3, 0.5)] +
                          [Atom("heartkit", "", t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)])
    elif f == "af":
        fm["rules"] = [Atom("rules", "", t) for t in (0.3, 0.5, 0.7)]
        fm["heartkit"] = [Atom("heartkit", "", t) for t in (0.3, 0.5, 0.75, 0.9)]
    elif f == "svt":
        fm["rules"] = [Atom("rules", "rate >", 120.0), Atom("rules", "rate >", 140.0), Atom("rules", "", NOTHR)]
        fm["heartkit"] = [Atom("heartkit", "", t) for t in (0.3, 0.5, 0.7)]
    elif f == "pause":
        fm["rules"] = [Atom("rules", "all beats", 2.5), Atom("rules", "", 2.5), Atom("rules", "", 3.0)]
    elif f == "brady":
        fm["rules"] = [Atom("rules", "", t) for t in (50.0, 45.0, 40.0)]
    elif f == "tachy":
        fm["rules"] = [Atom("rules", "", t) for t in (130.0, 140.0, 150.0)]
    return fm


# --------------------------------------------------------------------------
# Candidates and their evaluation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cand:
    mode: str                 # atom / or / vote2 / confirm
    parts: tuple[Atom, ...]

    def label(self) -> str:
        b = ""
        if self.mode == "atom":
            return self.parts[0].label() + b
        if self.mode == "confirm":
            return f"{self.parts[0].label()}, confirmed by {self.parts[1].label()}" + b
        j = " OR " if self.mode == "or" else " + "
        s = j.join(p.label() for p in self.parts)
        return (s if self.mode == "or" else f"≥ 2 of [{s}]") + b

    @property
    def n_systems(self) -> int:
        return len({p.fam for p in self.parts})


def defaults(f: str) -> dict[str, Atom]:
    """Each system at its own default operating point (no tuning): rules as merged, heartKIT's
    configured thresholds (V/S label at 0.5, AF 0.75), ECGFounder / CLEF window probability 0.5
    (burden: >= 10 % of windows), every burden finding flagged at >= 1 %."""
    if f in F.BURDEN:
        return {"rules": Atom("rules", "burden", 0.01), "heartkit": Atom("heartkit", "burden", 0.01),
                "ecgfounder": Atom("ecgfounder", "burden of windows", 0.1),
                "clef": Atom("clef", "burden of windows", 0.1)}
    d = {"ecgfounder": Atom("ecgfounder", "", 0.5), "clef": Atom("clef", "", 0.5)}
    d["rules"] = {"vt": Atom("rules", "", NOTHR), "af": Atom("rules", "", 0.5), "svt": Atom("rules", "", NOTHR),
                  "pause": Atom("rules", "", 3.0), "brady": Atom("rules", "", 40.0),
                  "tachy": Atom("rules", "", 150.0)}[f]
    if f in ("vt", "svt", "af"):
        d["heartkit"] = Atom("heartkit", "", 0.75 if f == "af" else 0.5)
    return d


BURDEN_GRID = {"beats": (0.003, 0.005, 0.01, 0.02), "windows": (0.02, 0.05, 0.1, 0.2)}


def candidates(f: str) -> list[Cand]:
    fams = atoms_for(f)
    out: list[Cand] = []
    out += [Cand("atom", (a,)) for al in fams.values() for a in al]
    if f in F.BURDEN:  # flag the patient when any of the systems flags it
        lo = {fam: [a for a in al if a.thr <= sorted({x.thr for x in al})[1]] for fam, al in fams.items()}
        for k in (2, 3, 4):
            for fs_ in itertools.combinations(lo, k):
                for combo in itertools.product(*(lo[x] if k == 2 else lo[x][:2] for x in fs_)):
                    out.append(Cand("or", combo))
        return list(dict.fromkeys(out))
    pick = {fam: _three(al) for fam, al in fams.items()}
    for k in (2, 3, 4):
        for fs_ in itertools.combinations(pick, k):
            for combo in itertools.product(*(pick[x] for x in fs_)):
                out.append(Cand("or", combo))
    mid = {fam: al[len(al) // 2] for fam, al in fams.items()}
    sens = {fam: al[0] for fam, al in fams.items()}
    for src in (mid, sens):
        for k in (3, 4):
            for fs_ in itertools.combinations(src, k):
                out.append(Cand("vote2", tuple(src[x] for x in fs_)))
    for a_f, b_f in itertools.permutations(fams, 2):
        for a in pick[a_f]:
            for b in fams[b_f][:2]:
                out.append(Cand("confirm", (a, b)))
    return list(dict.fromkeys(out))


def _three(al: list[Atom]) -> list[Atom]:
    if len(al) <= 3:
        return al
    return [al[0], al[len(al) // 2], al[-1]]


class Evaluator:
    """Caches atom detections per record and scores candidates on a set of rows."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.cache: dict = {}

    def dets(self, i: int, f: str, a: Atom):
        k = (i, f, a)
        if k not in self.cache:
            self.cache[k] = atom_dets(self.rows[i], f, a)
        return self.cache[k]

    def record(self, i: int, f: str, c: Cand) -> dict:
        row = self.rows[i]
        ref = row["rf"][f]
        if f in F.BURDEN:
            return {"k": [self.dets(i, f, a) for a in c.parts]}
        parts = [self.dets(i, f, a) for a in c.parts]
        if c.mode in ("atom", "or"):
            d = merge_np(np.concatenate(parts)) if len(parts) > 1 else parts[0]
        elif c.mode == "confirm":
            d = parts[0][F.hits(parts[0], parts[1])]
        else:  # vote2
            keep = [p[F.hits(p, np.concatenate([q for j, q in enumerate(parts) if j != k]))]
                    for k, p in enumerate(parts)]
            d = merge_np(np.concatenate(keep))
        found = F.hits(ref["eps"], d) if len(ref["eps"]) else np.zeros(0, bool)
        false = ~F.hits(d, ref["ok"]) if len(d) else np.zeros(0, bool)
        return {"n_ref": len(ref["eps"]), "found": int(found.sum()), "n_det": len(d), "n_false": int(false.sum())}

    def score(self, f: str, c: Cand, idx: list[int]) -> dict:
        pats: dict = {}
        n_det = n_false = 0
        hours = 0.0
        eps = found_eps = 0
        for i in idx:
            row = self.rows[i]
            if not row["rf"][f]["assess"]:
                continue
            p = pats.setdefault((row["db"], row["patient"]), {"pos": 0, "found": 0, "k": 0.0, "n": 0.0,
                                                              "est": None, "hours": 0.0})
            p["hours"] += row["hours"]
            hours += row["hours"]
            res = self.record(i, f, c)
            if f in F.BURDEN:
                p["k"] += row["rf"][f]["k"]
                p["n"] += row["rf"][f]["n"]
                est = np.array(res["k"], float)
                p["est"] = est if p["est"] is None else p["est"] + est
                continue
            p["pos"] += res["n_ref"]
            p["found"] += res["found"]
            eps += res["n_ref"]
            found_eps += res["found"]
            n_det += res["n_det"]
            n_false += res["n_false"]
        if f in F.BURDEN:
            pos = flag = tp = 0
            for p in pats.values():
                is_pos = p["n"] > 0 and p["k"] / p["n"] >= F.BURDEN_MIN
                hit = any(e[0] / max(e[1], 1) >= a.thr for e, a in zip(p["est"], c.parts))
                pos += is_pos
                flag += hit
                tp += hit and is_pos
            n_det, n_false = flag, flag - tp
            pp, fp_ = pos, tp
        else:
            pp = sum(p["pos"] > 0 for p in pats.values())
            fp_ = sum(p["found"] > 0 for p in pats.values())
        lo, hi = F.wilson(fp_, pp)
        return {"patients": len(pats), "pos": pp, "found": fp_, "Se": fp_ / pp if pp else NAN, "Se_lo": lo, "Se_hi": hi,
                "episodes": eps, "ep_found": found_eps, "ep_Se": found_eps / eps if eps else NAN,
                "det": n_det, "false": n_false, "PPV": (n_det - n_false) / n_det if n_det else NAN,
                "hours": hours, "false_per_day": n_false / hours * 24 if hours else NAN}


def _num(v, default: float) -> float:
    return default if v is None or v != v else v


def rank_key(s: dict, c: Cand, best_se: float, target: float = TARGET) -> tuple:
    se = _num(s["Se"], 0.0)
    cap = min(target, best_se)
    return (-min(se, cap), _num(s["false_per_day"], np.inf), -_num(s["ep_Se"], 0.0), c.n_systems, len(c.parts))


def pareto(scored: list[tuple[Cand, dict]], n: int = 5) -> list[tuple[Cand, dict]]:
    """Tuning front: for each Se level, the fewest false findings; up to n points spread over it."""
    front: list[tuple[Cand, dict]] = []
    for c, s in sorted(scored, key=lambda x: (-_num(x[1]["Se"], 0), _num(x[1]["false_per_day"], np.inf), x[0].n_systems)):
        if not front or _num(s["false_per_day"], np.inf) < _num(front[-1][1]["false_per_day"], np.inf) - 1e-9:
            front.append((c, s))
    if len(front) <= n:
        return front
    keep = sorted({round(k * (len(front) - 1) / (n - 1)) for k in range(n)})
    return [front[k] for k in keep]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def fmt(v, nd=1, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{100 * v:.{nd}f} %" if pct else f"{v:.{nd}f}"


def se_cell(s: dict) -> str:
    if not s["pos"]:
        return "– (0 patients)"
    return f"{fmt(s['Se'], 1, True)} ({s['found']}/{s['pos']}) [{fmt(s['Se_lo'], 0, True)}–{fmt(s['Se_hi'], 0, True)}]"


def run(rows: list[dict]) -> dict:
    ev = Evaluator(rows)
    tune = [i for i, r in enumerate(rows) if r["split"] == "tune"]
    held = [i for i, r in enumerate(rows) if r["split"] == "held"]
    long_ = [i for i in held if rows[i]["long"]]
    noise = [i for i, r in enumerate(rows) if r["split"] == "noise"]
    out: dict = {"findings": {}, "n": {"tune": len(tune), "held": len(held), "long": len(long_), "noise": len(noise)}}
    for f in F.FINDINGS:
        t0 = time.time()
        cands = candidates(f)
        scored = [(c, ev.score(f, c, tune)) for c in cands]
        best_se = max((_num(s["Se"], 0.0) for _, s in scored), default=0.0)
        scored.sort(key=lambda x: rank_key(x[1], x[0], best_se))
        choice, st = scored[0]
        strict, ss = min(scored, key=lambda x: rank_key(x[1], x[0], best_se, 1.0))  # target 100 %
        # best single system per family (same rule), for the "best system" table
        per_sys = {}
        for fam in NAMES:
            fs_ = [(c, s) for c, s in scored if c.mode == "atom" and c.parts[0].fam == fam]
            if fs_:
                b = max((_num(s["Se"], 0.0) for _, s in fs_), default=0.0)
                c, s = min(fs_, key=lambda x: rank_key(x[1], x[0], b))
                per_sys[fam] = {"cand": c.label(), "tune": s, "held": ev.score(f, c, held),
                                "long": ev.score(f, c, long_)}
        ops = []
        for c, s in pareto(scored):
            ops.append({"cand": c.label(), "tune": s, "held": ev.score(f, c, held), "chosen": c == choice})
        if not any(o["chosen"] for o in ops):
            ops.append({"cand": choice.label(), "tune": st, "held": ev.score(f, choice, held), "chosen": True})
        dflt = {fam: {"cand": Cand("atom", (a,)).label(), "held": ev.score(f, Cand("atom", (a,)), held),
                      "long": ev.score(f, Cand("atom", (a,)), long_)} for fam, a in defaults(f).items()}
        res = {"choice": choice.label(), "n_candidates": len(cands), "tune": st, "defaults": dflt,
               "strict": {"choice": strict.label(), "tune": ss, "held": ev.score(f, strict, held),
                          "long": ev.score(f, strict, long_)},
               "held": ev.score(f, choice, held), "long": ev.score(f, choice, long_),
               "noise": {snr: ev.score(f, choice, [i for i in noise if rows[i]["key"].endswith(snr)])
                         for snr in ds.NST_SNR},
               "by_db": {db: ev.score(f, choice, [i for i in held if rows[i]["db"] == db])
                         for db in sorted({rows[i]["db"] for i in held})},
               "systems": per_sys, "operating_points": ops}
        out["findings"][f] = res
        print(f"{f}: {len(cands)} candidates, {time.time() - t0:.0f} s -> {choice.label()}", file=sys.stderr)
    out["critical"] = critical(out)
    return out


def critical(out: dict) -> dict:
    """All critical findings together (DeepRhythmAI-style): per-patient Se over (patient,
    finding) pairs; false findings per day summed over the findings (each on its own
    assessable hours) on the long held-out recordings."""
    res = {}
    for split in ("held", "long"):
        k = sum(out["findings"][f][split]["found"] for f in CRITICAL)
        n = sum(out["findings"][f][split]["pos"] for f in CRITICAL)
        lo, hi = F.wilson(k, n)
        fpd = sum(out["findings"][f][split]["false_per_day"] for f in CRITICAL
                  if out["findings"][f][split]["false_per_day"] == out["findings"][f][split]["false_per_day"])
        res[split] = {"found": k, "pos": n, "Se": k / n if n else NAN, "Se_lo": lo, "Se_hi": hi, "false_per_day": fpd}
    return res


TITLE = {"vt": "VT/NSVT/VF (≥ 3 V)", "af": "AF/AFL ≥ 30 s", "svt": "sustained SVT ≥ 30 s",
         "pvc": "frequent PVCs (≥ 1 %)", "pac": "frequent PACs (≥ 1 %)", "pause": "pause > 3 s",
         "brady": "brady < 40 bpm ≥ 10 s", "tachy": "tachy > 150 bpm ≥ 10 s"}


def report(out: dict) -> str:
    L = ["<!-- generated by ai/benchmark/router.py -->", "",
         f"Records: tuning {out['n']['tune']}, held-out {out['n']['held']} (long ≥ 10 h: {out['n']['long']}), "
         f"noise stress {out['n']['noise']}. Target per-patient Se {100 * TARGET:.0f} %.", "",
         "### Router choice per finding (held-out)", "",
         "| finding | chosen on tuning data | tuning Se | held-out per-patient Se [95 % CI] | episode Se | PPV | false / day (all held-out) | false / day (long recordings) |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for f, r in out["findings"].items():
        t, h, lg = r["tune"], r["held"], r["long"]
        L.append(f"| {TITLE[f]} | {r['choice']} | {fmt(t['Se'], 1, True)} ({t['found']}/{t['pos']}) | {se_cell(h)} | "
                 f"{fmt(h['ep_Se'], 1, True)} | {fmt(h['PPV'], 2)} | {fmt(h['false_per_day'], 2)} | {fmt(lg['false_per_day'], 2)} |")
    L += ["", "### Strict variant: target 100 % on tuning (then fewest false findings)", "",
          "| finding | chosen on tuning data | tuning Se | held-out per-patient Se [95 % CI] | PPV | false / day (long recordings) |",
          "|---|---|---:|---:|---:|---:|"]
    for f, r in out["findings"].items():
        t, h, lg = r["strict"]["tune"], r["strict"]["held"], r["strict"]["long"]
        L.append(f"| {TITLE[f]} | {r['strict']['choice']} | {fmt(t['Se'], 1, True)} ({t['found']}/{t['pos']}) | "
                 f"{se_cell(h)} | {fmt(h['PPV'], 2)} | {fmt(lg['false_per_day'], 2)} |")
    L += ["", "### Best single system per finding (held-out per-patient Se; false / day on long recordings)", "",
          "| finding | " + " | ".join(NAMES.values()) + " |", "|---|" + "---|" * len(NAMES)]
    for f, r in out["findings"].items():
        cells = []
        for fam in NAMES:
            s = r["systems"].get(fam)
            cells.append("–" if s is None else f"{se_cell(s['held'])}; {fmt(s['long']['false_per_day'], 1)}/day<br>"
                         f"<sub>{s['cand']}</sub>")
        L.append(f"| {TITLE[f]} | " + " | ".join(cells) + " |")
    L += ["", "### Each system at its default operating point (no tuning; held-out per-patient Se; "
          "false / day on long recordings)", "",
          "| finding | " + " | ".join(NAMES.values()) + " |", "|---|" + "---|" * len(NAMES)]
    for f, r in out["findings"].items():
        cells = []
        for fam in NAMES:
            s = r["defaults"].get(fam)
            cells.append("–" if s is None else f"{se_cell(s['held'])}; {fmt(s['long']['false_per_day'], 1)}/day")
        L.append(f"| {TITLE[f]} | " + " | ".join(cells) + " |")
    L += ["", "### Operating points (tuning Pareto front; held-out numbers)", ""]
    for f, r in out["findings"].items():
        L += [f"**{TITLE[f]}** ({r['n_candidates']} candidates)", "",
              "| candidate | tuning Se | tuning false/day | held-out Se [95 % CI] | held-out PPV | held-out false/day |",
              "|---|---:|---:|---:|---:|---:|"]
        for o in r["operating_points"]:
            mark = " **(chosen)**" if o["chosen"] else ""
            t, h = o["tune"], o["held"]
            L.append(f"| {o['cand']}{mark} | {fmt(t['Se'], 1, True)} | {fmt(t['false_per_day'], 1)} | {se_cell(h)} | "
                     f"{fmt(h['PPV'], 2)} | {fmt(h['false_per_day'], 1)} |")
        L.append("")
    L += ["### Router per held-out database (per-patient Se; false / day)", "",
          "| finding | " + " | ".join(sorted(next(iter(out["findings"].values()))["by_db"])) + " |",
          "|---|" + "---|" * len(next(iter(out["findings"].values()))["by_db"])]
    for f, r in out["findings"].items():
        L.append(f"| {TITLE[f]} | " + " | ".join(
            "–" if not s["patients"] else f"{fmt(s['Se'], 0, True)} ({s['found']}/{s['pos']}); {fmt(s['false_per_day'], 1)}"
            for s in r["by_db"].values()) + " |")
    L += ["", "### Noise stress (NSTDB 118/119; em = PhysioNet mixes, ma = ours): router false findings per hour", "",
          "| finding | " + " | ".join(f"{s.replace('_', '-')} dB" for s in ds.NST_SNR) + " |",
          "|---|" + "---:|" * len(ds.NST_SNR)]
    for f, r in out["findings"].items():
        L.append(f"| {TITLE[f]} | " + " | ".join(
            fmt(s["false_per_day"] / 24, 1) if s["hours"] else "–" for s in r["noise"].values()) + " |")
    c = out["critical"]
    L += ["", "### All critical findings together (" + ", ".join(CRITICAL) + ")", "",
          "| split | per-patient Se (patient-finding pairs) [95 % CI] | false findings / day | per 1,000 patient-days |",
          "|---|---:|---:|---:|"]
    for k, v in c.items():
        L.append(f"| {k} | {fmt(v['Se'], 1, True)} ({v['found']}/{v['pos']}) [{fmt(v['Se_lo'], 0, True)}–"
                 f"{fmt(v['Se_hi'], 0, True)}] | {fmt(v['false_per_day'], 1)} | {fmt(1000 * v['false_per_day'], 0)} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--db", nargs="+", default=None)
    args = ap.parse_args()
    t0 = time.time()
    rows = [x for x in (prepare(r) for r in ds.records(args.db)) if x is not None]
    print(f"{len(rows)} records loaded in {time.time() - t0:.0f} s", file=sys.stderr)
    out = run(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "router.json").write_text(json.dumps(out, indent=1, default=float))
    md = report(out)
    (args.out / "router.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
