"""Findings of the sensitivity-first benchmark: reference episodes, window targets, scoring.

Findings (reference from the databases' annotations; see datasets.py):
    vt     VT/NSVT/VF: >= 3 consecutive V beats, or a VT / VF / VFL rhythm episode
    af     AF/AFL >= 30 s
    svt    sustained SVT >= 30 s: an S run lasting >= 30 s, or an SVTA / AT / PAT
           (SHDB-AF also NOD) episode >= 30 s
    pvc    frequent PVCs: V beats >= 1 % of all beats of the patient
    pac    frequent PACs: S beats >= 1 % of all beats
    pause  pause / asystole: an RR interval > 3 s between two readable beats, outside
           annotated unreadable spans ('~' marks) and VF
    brady  HR < 40 bpm (8-beat running median) for >= 10 s
    tachy  HR > 150 bpm (8-beat running median) for >= 10 s, any rhythm

A finding is only scored on records whose annotations can confirm or refute it
(ASSESS): e.g. VT needs beat labels (or VT/VF rhythm labels), sustained SVT
needs beat labels or SVT rhythm codes. A detection is a false finding when it
overlaps none of the reference's "allowed" spans (the reference episodes plus
related spans a reader would not call wrong, e.g. any AF span for AF, a V run
or VF for VT; listed in ref_findings).

Scores:
    per-patient Se   share of patients with the finding in whom the system found
                     at least one reference episode (pvc/pac: flagged the patient);
                     95 % Wilson interval
    PPV              share of detections that are not false (pvc/pac: of flagged patients)
    false / day      false detections per 24 h of assessable recording
"""

from __future__ import annotations

import math
import warnings

import numpy as np

FINDINGS = ("vt", "af", "svt", "pvc", "pac", "pause", "brady", "tachy")
BURDEN = {"pvc": "V", "pac": "S"}
BURDEN_MIN = 0.01
MIN_S = {"vt": 0.0, "af": 30.0, "svt": 30.0, "pause": 0.0, "brady": 10.0, "tachy": 10.0}
PAUSE_S, PAUSE_OK_S = 3.0, 2.8
BRADY, BRADY_OK = 40.0, 45.0
TACHY, TACHY_OK = 150.0, 140.0
HR_BEATS = 8
TOL = 1.0  # s, overlap tolerance between a detection and a reference span
SVT_RHYTHM_DBS = {"mitdb", "ltafdb", "shdb-af"}  # rhythm codes that name SVT episodes
VT_RHYTHM_DBS = {"vfdb", "cudb"}                 # VT / VF episodes labelled, beats not
WIN_S = 10.0


# --------------------------------------------------------------------------
# Interval helpers
# --------------------------------------------------------------------------

def merge(iv, gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((float(a), float(b)))
    return out


def as_array(iv) -> np.ndarray:
    return np.asarray(iv, np.float64).reshape(-1, 2)


def hits(dets: np.ndarray, spans: np.ndarray, tol: float = TOL) -> np.ndarray:
    """For each detection (row of dets), does it overlap any span (within tol)?"""
    if len(dets) == 0 or len(spans) == 0:
        return np.zeros(len(dets), bool)
    s = spans[np.argsort(spans[:, 0])]
    run_max_end = np.maximum.accumulate(s[:, 1])
    # last span starting before det end + tol; overlap iff the max end so far reaches det start - tol
    k = np.searchsorted(s[:, 0], dets[:, 1] + tol, side="right") - 1
    ok = k >= 0
    out = np.zeros(len(dets), bool)
    out[ok] = run_max_end[k[ok]] >= dets[ok, 0] - tol
    return out


def runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Inclusive (i, j) index ranges of True runs with at least min_len elements."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(np.int8))
    a, b = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    keep = b - a + 1 >= min_len
    return list(zip(a[keep].tolist(), b[keep].tolist()))


def label_runs(t: np.ndarray, lab: np.ndarray, c: str, min_n: int = 3) -> list[tuple[float, float]]:
    return [(t[i], t[j]) for i, j in runs(np.asarray(lab) == c, min_n)]


def running_hr(t: np.ndarray, good: np.ndarray, w: int = HR_BEATS) -> np.ndarray:
    """8-beat running median heart rate at each beat (nan where too few readable beats)."""
    hr = np.full(len(t), np.nan)
    if len(t) < 2:
        return hr
    hr[1:] = 60.0 / np.diff(t)
    hr[~good] = np.nan
    hr[1:][~good[:-1]] = np.nan
    if len(t) <= w:
        return hr
    win = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.full(w, np.nan), hr]), w + 1)[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=1)


def rate_stretches(t: np.ndarray, hr: np.ndarray, cond, min_s: float) -> list[tuple[float, float]]:
    out = []
    for i, j in runs(cond(np.nan_to_num(hr, nan=np.nanmedian(hr) if np.isfinite(hr).any() else 75.0))):
        a = t[max(i - 1, 0)]
        if t[j] - a >= min_s:
            out.append((a, t[j]))
    return out


def pauses(t: np.ndarray, good: np.ndarray, thr: float) -> list[tuple[float, float]]:
    rr = np.diff(t)
    i = np.where((rr > thr) & good[:-1] & good[1:])[0]
    return [(t[k], t[k + 1]) for k in i]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------------------------
# Reference findings
# --------------------------------------------------------------------------

def _spans(rhythm, codes, min_s: float = 0.0) -> list[tuple[float, float]]:
    return [e for e in merge([(a, b) for a, b, c in rhythm if c in codes]) if e[1] - e[0] >= min_s]


def ref_findings(d: dict) -> dict:
    """{finding: {"assess": bool, "eps": (n,2), "ok": (m,2)}}; pvc/pac: {"assess", "n", "k"}."""
    ref, db = d["ref"], d["db"]
    rh = ref["rhythm"]
    out: dict = {}
    vf = _spans(rh, {"VF"})
    vt_rh = _spans(rh, {"VT"})
    if ref["has_beats"]:
        bt, bc = ref["beats_t"], ref["beats_c"]
        vruns = label_runs(bt, bc, "V")
        sruns = label_runs(bt, bc, "S")
        good = bc != "Q"
        in_vf = np.zeros(len(bt), bool)
        for a, b in vf:
            in_vf |= (bt >= a) & (bt <= b)
        hr = running_hr(bt, good & ~in_vf)
        noise = _spans(rh, {"NOISE"})
        clean = lambda iv: [e for e in iv if not any(a - TOL <= e[1] and e[0] <= b + TOL for a, b in noise + vf)]  # noqa: E731
        p_ref = clean(pauses(bt, good, PAUSE_S))
        p_ok = pauses(bt, np.ones(len(bt), bool), PAUSE_OK_S) + noise  # unreadable: neither true nor false
        brady = clean(rate_stretches(bt, hr, lambda h: h < BRADY, MIN_S["brady"]))
        brady_ok = rate_stretches(bt, hr, lambda h: h < BRADY_OK, 0.0) + p_ok
        tachy = rate_stretches(bt, hr, lambda h: h > TACHY, MIN_S["tachy"])
        tachy_ok = rate_stretches(bt, hr, lambda h: h > TACHY_OK, 0.0) + vf
        nb = int(np.isin(bc, ["N", "S", "V", "F"]).sum())
    else:
        vruns = sruns = []
    af_all = _spans(rh, {"AF"})
    svt_rh = _spans(rh, {"SVT"})

    vt_ok = merge(vruns + vt_rh + vf)
    out["vt"] = {"assess": ref["has_beats"] or db in VT_RHYTHM_DBS, "eps": vt_ok, "ok": vt_ok}
    out["af"] = {"assess": ref["has_rhythm"], "eps": [e for e in af_all if e[1] - e[0] >= MIN_S["af"]],
                 "ok": af_all}
    svt_eps = [e for e in merge(sruns + svt_rh) if e[1] - e[0] >= MIN_S["svt"]]
    out["svt"] = {"assess": ref["has_beats"] or db in SVT_RHYTHM_DBS, "eps": svt_eps,
                  "ok": merge(sruns + svt_rh + af_all)}
    if ref["has_beats"]:
        out["pause"] = {"assess": True, "eps": p_ref, "ok": merge(p_ok)}
        out["brady"] = {"assess": True, "eps": brady, "ok": merge(brady_ok)}
        out["tachy"] = {"assess": True, "eps": tachy, "ok": merge(tachy_ok + vt_ok + svt_eps)}
        for f, c in BURDEN.items():
            out[f] = {"assess": True, "n": nb, "k": int((bc == c).sum())}
    else:
        for f in ("pause", "brady", "tachy"):
            out[f] = {"assess": False, "eps": [], "ok": []}
        for f in BURDEN:
            out[f] = {"assess": False, "n": 0, "k": 0}
    for f in out:
        if "eps" in out[f]:
            out[f]["eps"], out[f]["ok"] = as_array(out[f]["eps"]), as_array(out[f]["ok"])
    return out


# --------------------------------------------------------------------------
# Window targets (for the CLEF head)
# --------------------------------------------------------------------------

HEAD_TARGETS = ("vt", "af", "svt", "pvc", "pac", "pause", "brady", "tachy")
EDGE_S = 0.3


def window_targets(d: dict) -> dict[str, np.ndarray]:
    """Per 10 s window: 1 / 0 / nan (ambiguous or not assessable) for each head target.

    vt   >= 3 consecutive V beats inside, or >= 2 s of VT/VF;  af  >= 50 % AF (0 % = negative)
    svt  >= 3 consecutive S beats or >= 2 s of SVT;  pvc / pac  >= 1 V / S beat (not at an edge)
    pause  an RR > 2 s inside;  brady / tachy  window rate < 50 / > 120 bpm (readable beats)
    """
    ref, t0s = d["ref"], np.asarray(d["win_t"], np.float64)
    n = len(t0s)
    out = {k: np.full(n, np.nan) for k in HEAD_TARGETS}
    rh = ref["rhythm"]

    def cover(codes):
        sp = _spans(rh, codes)
        c = np.zeros(n)
        for a, b in sp:
            i0, i1 = int(max(a // WIN_S, 0)), int(min(b // WIN_S + 1, n))
            for i in range(i0, i1):
                c[i] += max(0.0, min(b, t0s[i] + WIN_S) - max(a, t0s[i]))
        return c

    vtc, svc = cover({"VT", "VF"}), cover({"SVT"})
    if ref["has_rhythm"]:
        afc = cover({"AF"})
        out["af"] = np.where(afc >= WIN_S / 2, 1.0, np.where(afc == 0, 0.0, np.nan))
    if d["db"] in VT_RHYTHM_DBS:
        out["vt"] = np.where(vtc >= 2, 1.0, np.where(vtc == 0, 0.0, np.nan))
    if not ref["has_beats"]:
        if d["db"] in SVT_RHYTHM_DBS:
            out["svt"] = np.where(svc >= 2, 1.0, np.where(svc == 0, 0.0, np.nan))
        return out
    bt, bc = ref["beats_t"], ref["beats_c"]
    w = np.floor(bt / WIN_S).astype(int)
    core = ((bt - w * WIN_S) >= EDGE_S) & ((bt - w * WIN_S) < WIN_S - EDGE_S)
    for key, c in (("pvc", "V"), ("pac", "S")):
        pos = np.zeros(n, bool)
        edge = np.zeros(n, bool)
        m = (bc == c) & (w < n)
        np.logical_or.at(pos, w[m & core], True)
        np.logical_or.at(edge, w[m & ~core], True)
        out[key] = np.where(pos, 1.0, np.where(edge, np.nan, 0.0))
    for key, c, rc in (("vt", "V", vtc), ("svt", "S", svc)):
        full = np.zeros(n, bool)
        part = np.zeros(n, bool)
        for i, j in runs(bc == c, 3):
            wi, wj = w[i], w[j]
            if wi == wj and wi < n:
                full[wi] = True
            else:
                for k in range(wi, min(wj, n - 1) + 1):
                    inside = int(((w[i:j + 1]) == k).sum())
                    (full if inside >= 3 else part)[k] = True
        y = np.where(full | (rc >= 2), 1.0, np.where(part | (rc > 0), np.nan, 0.0))
        out[key] = y
    rr = np.diff(bt)
    pause = np.zeros(n, bool)
    k = w[1:][(rr > 2.0) & (w[1:] == w[:-1]) & (w[1:] < n)]
    pause[k] = True
    out["pause"] = pause.astype(float)
    good = (bc != "Q") & (w < n)
    hr_b = np.full(n, np.nan)
    gw, gt = w[good], bt[good]
    cuts = np.searchsorted(gw, np.arange(n + 1))  # beats are time-ordered, so windows are contiguous
    for i in np.where(np.diff(cuts) >= 3)[0]:
        hr_b[i] = 60.0 / np.median(np.diff(gt[cuts[i]:cuts[i + 1]]))
    out["brady"] = np.where(np.isnan(hr_b), np.nan, (hr_b < 50).astype(float))
    out["tachy"] = np.where(np.isnan(hr_b), np.nan, (hr_b > 120).astype(float))
    return out
