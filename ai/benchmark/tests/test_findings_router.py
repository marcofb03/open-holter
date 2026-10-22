"""Offline checks of the long-term benchmark logic (no data, no models, no torch).

    python -m pytest -q tests        (from ai/benchmark/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import datasets as ds  # noqa: E402
import findings as F  # noqa: E402
import router as R  # noqa: E402


def test_wilson_matches_known_values():
    lo, hi = F.wilson(8, 8)
    assert abs(lo - 0.676) < 0.002 and hi == 1.0
    lo, hi = F.wilson(44, 54)
    assert abs(lo - 0.692) < 0.003 and abs(hi - 0.896) < 0.003
    assert np.isnan(F.wilson(0, 0)[0])


def test_hits_and_merge():
    d = np.array([[0.0, 1.0], [5.0, 6.0], [20.0, 21.0]])
    s = np.array([[6.5, 7.0], [1.5, 2.0], [100.0, 200.0]])
    assert F.hits(d, s, tol=1.0).tolist() == [True, True, False]
    assert F.hits(d, np.zeros((0, 2))).tolist() == [False] * 3
    m = R.merge_np(np.array([[3.0, 4.0], [0.0, 1.0], [0.5, 2.0], [10.0, 11.0]]))
    assert m.tolist() == [[0.0, 2.0], [3.0, 4.0], [10.0, 11.0]]
    assert R.merge_np(np.array([[0.0, 1.0], [2.0, 3.0]]), gap=1.0).tolist() == [[0.0, 3.0]]
    assert F.merge([(0, 1), (0.5, 2)]) == [(0.0, 2.0)]


def test_runs():
    assert F.runs(np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], bool), 3) == [(1, 3), (8, 11)]
    assert F.runs(np.zeros(4, bool)) == []


def _ref(bt, bc, rhythm=(), has_rhythm=True):
    return {"dur_s": float(bt[-1] + 5), "beats_t": np.asarray(bt, float), "beats_c": np.asarray(bc),
            "rhythm": list(rhythm), "has_beats": True, "has_rhythm": has_rhythm}


def test_reference_findings_vt_pause_burden_brady():
    bt = np.arange(0, 600, 1.0)              # 60 bpm for 10 min
    bc = np.array(["N"] * len(bt))
    bc[100:104] = "V"                          # a 4-beat V run
    bt = np.delete(bt, np.arange(300, 303))    # a 4 s pause
    bc = np.delete(bc, np.arange(300, 303))
    rf = F.ref_findings({"db": "mitdb", "ref": _ref(bt, bc)})
    assert len(rf["vt"]["eps"]) == 1 and rf["vt"]["assess"]
    assert len(rf["pause"]["eps"]) == 1
    assert rf["pvc"]["k"] == 4 and rf["pvc"]["n"] == len(bt)
    assert len(rf["brady"]["eps"]) == 0
    slow = np.arange(0, 120, 2.0)               # 30 bpm for 2 min
    rf = F.ref_findings({"db": "mitdb", "ref": _ref(slow, np.array(["N"] * len(slow)))})
    assert len(rf["brady"]["eps"]) == 1


def test_af_needs_30s_and_rhythm_labels():
    bt = np.arange(0, 300, 0.8)
    ref = _ref(bt, np.array(["N"] * len(bt)), [(10.0, 30.0, "AF"), (100.0, 200.0, "AF")])
    rf = F.ref_findings({"db": "mitdb", "ref": ref})
    assert rf["af"]["eps"].tolist() == [[100.0, 200.0]] and len(rf["af"]["ok"]) == 2
    ref["has_rhythm"] = False
    assert not F.ref_findings({"db": "svdb", "ref": ref})["af"]["assess"]


def test_window_targets():
    bt = np.arange(0.5, 60, 1.0)
    bc = np.array(["N"] * len(bt))
    bc[12:15] = "V"                             # 12.5, 13.5, 14.5 s: window 1
    bc[40] = "S"                                # 40.5 s: window 4
    d = {"db": "mitdb", "ref": _ref(bt, bc, [(30.0, 60.0, "AF")]), "win_t": np.arange(0, 51, 10.0)}
    t = F.window_targets(d)
    assert t["vt"][1] == 1 and t["vt"][0] == 0
    assert t["pvc"][1] == 1 and t["pac"][4] == 1 and t["pac"][1] == 0
    assert t["af"].tolist()[:3] == [0, 0, 0] and t["af"][3] == 1
    assert t["brady"][0] == 0 and t["tachy"][0] == 0


def test_split_is_deterministic_and_patient_level():
    assert ds.third("ltafdb", "00") == ds.third("ltafdb", "00")
    assert ds.patient_of("cpsc2021", "Training_set_I/data_12_3") == "12"
    pts = ds.icentia_patients()
    assert len(pts) == ds.ICENTIA_N and all(int(p[1:]) >= ds.HEARTKIT_TRAIN_MAX for p in pts)
    shares = [ds.third("x", str(i)) == "tune" for i in range(3000)]
    assert 0.3 < np.mean(shares) < 0.37


def test_nst_mix_snr():
    rng = np.random.default_rng(0)
    fs = 360.0
    t = np.arange(int(600 * fs)) / fs
    beats = np.arange(int(fs), len(t) - int(fs), int(fs))
    clean = np.zeros(len(t))
    for b in beats:
        clean[b - 5:b + 5] = 1.0
    noise = rng.standard_normal(len(t))
    out = ds.nst_mix(clean, noise, fs, beats, 6.0)
    seg = slice(int(310 * fs), int(400 * fs))   # inside the first noisy 2 min
    assert np.allclose(out[:int(299 * fs)], clean[:int(299 * fs)])
    n = out[seg] - clean[seg]
    s = np.mean([np.ptp(clean[b - 18:b + 18]) ** 2 for b in beats]) / 8
    assert abs(10 * np.log10(s / np.mean(n ** 2)) - 6.0) < 1.0


def test_candidates_and_selection_rule():
    for f in F.FINDINGS:
        c = R.candidates(f)
        assert len(c) == len(set(c)) and len(c) > 20
    a = R.Cand("atom", (R.Atom("rules", "", float("nan")),))
    hi = {"Se": 1.0, "false_per_day": 50.0, "ep_Se": 1.0}
    ok = {"Se": 0.985, "false_per_day": 5.0, "ep_Se": 0.9}
    lo = {"Se": 0.9, "false_per_day": 0.1, "ep_Se": 0.9}
    ranked = sorted([hi, ok, lo], key=lambda s: R.rank_key(s, a, 1.0))
    assert ranked == [ok, hi, lo]  # >= 98 % ties, then fewest false findings
    ranked = sorted([lo, {"Se": 0.95, "false_per_day": 9.0, "ep_Se": 1}], key=lambda s: R.rank_key(s, a, 0.95))
    assert ranked[0]["Se"] == 0.95  # target not reachable: highest Se first
