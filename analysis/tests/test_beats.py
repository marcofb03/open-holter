import numpy as np
import pytest

from ecgpipe.beats import analyse_beats
from ecgpipe.context import Context, rate_events
from ecgpipe.events import find_events

from synth import FS, ecg, rhythm


def _match(truth, idx, tol=0.05):
    """Index into detected beats for every true beat (-1 = missed)."""
    det = idx / FS
    out = []
    for t, _ in truth:
        k = int(np.argmin(np.abs(det - t))) if len(det) else -1
        out.append(k if k >= 0 and abs(det[k] - t) <= tol else -1)
    return np.array(out)


@pytest.fixture(scope="module")
def mixed():
    """3 min of sinus rhythm with PACs, PVCs, one ventricular run and one pause."""
    events = {30: "pac", 60: "pvc", 90: "pac", 120: "pvc", 150: "vrun3", 180: "pause"}
    truth = rhythm(180.0, rr=0.8, events=events)
    x = ecg(truth, 180.0)
    _, b = analyse_beats(x, FS)
    return truth, b


def test_detects_every_beat(mixed):
    truth, b = mixed
    m = _match(truth, b.idx)
    assert (m >= 0).mean() >= 0.995
    assert len(b.idx) <= len(truth) + 1


def test_labels(mixed):
    truth, b = mixed
    m = _match(truth, b.idx)
    kinds = np.array([k for _, k in truth])
    lab = b.label[m[m >= 0]]
    kinds = kinds[m >= 0]
    assert (lab[kinds == "V"] == "V").all(), "every ventricular beat must be V"
    normal_lab = lab[kinds == "N"]
    # The early normal-shaped beats (PACs) should come out as S.
    assert (normal_lab == "S").sum() >= 2
    assert (normal_lab == "N").mean() > 0.97
    assert not (b.label == "Q").any()


def test_events(mixed):
    _, b = mixed
    ev = find_events(b, FS)
    kinds = [e.kind for e in ev]
    assert kinds.count("V run (>=3)") == 1
    run = next(e for e in ev if e.kind == "V run (>=3)")
    assert run.n_beats == 3 and run.rate_bpm == pytest.approx(150, abs=5)
    assert "Pause > 2 s" in kinds
    assert "Possible artefact" not in kinds


def test_fast_rate_event():
    truth = rhythm(60.0, rr=0.8) + [(60.0 + 0.35 * i, "N") for i in range(60)]
    truth += [(81.0 + 0.8 * i, "N") for i in range(40)]
    x = ecg(truth, 113.0)
    _, b = analyse_beats(x, FS)
    ev = find_events(b, FS)
    assert any(e.kind == "Fast rate > 150 bpm" for e in ev)


def test_emg_burst_is_unreadable():
    truth = rhythm(60.0)
    x = ecg(truth, 60.0)
    rng = np.random.default_rng(1)
    a, z = int(20 * FS), int(30 * FS)
    x[a:z] += 1.0 * rng.standard_normal(z - a)        # strong broadband noise
    _, b = analyse_beats(x, FS)
    t = b.idx / FS
    inside = (t > 21) & (t < 29)
    assert (b.label[inside] == "Q").mean() > 0.8
    assert (b.label[(t < 18) | (t > 32)] == "Q").mean() < 0.05


# ---------------------------------------------------------------- marching through

def _spiky(seed=0):
    """Sinus rhythm with sharp artefact spikes between the beats for ~4 s."""
    truth = rhythm(60.0, rr=0.8)
    x = ecg(truth, 60.0)
    rng = np.random.default_rng(seed)
    beats_t = np.array([t for t, _ in truth])
    k = np.arange(int(0.12 * FS)) / FS
    spike = np.exp(-k / 0.02) - np.exp(-k / 0.005)
    spike *= -1.2 / spike.max()
    spikes = []
    for t0 in beats_t[(beats_t > 30) & (beats_t < 34)]:
        ts = t0 + 0.4 + rng.uniform(-0.03, 0.03)       # half way between two beats
        i = int(ts * FS)
        x[i: i + len(spike)] += spike
        spikes.append(ts)
    return truth, x, np.array(spikes)


def test_marching_through_removes_fake_run():
    truth, x, spikes = _spiky()
    _, b = analyse_beats(x, FS)
    ev = find_events(b, FS)
    assert not any(e.kind == "V run (>=3)" for e in ev), "artefact must not be a V run"
    t = b.idx / FS
    # The real sinus beats inside the stretch stay N, the spikes are not V.
    sinus = np.array([tt for tt, _ in truth if 30 < tt < 34.5])
    m = _match([(s, "N") for s in sinus], b.idx)
    assert (b.label[m[m >= 0]] == "N").all()
    near_spike = np.array([np.min(np.abs(spikes - tt)) < 0.06 for tt in t])
    assert not (b.label[near_spike] == "V").any()


def test_marching_through_flags_and_reports():
    truth, x, spikes = _spiky(seed=3)
    _, b = analyse_beats(x, FS)
    t = b.idx / FS
    stretch = (t > 29.5) & (t < 35)
    assert b.artifact[stretch].any()
    assert not b.artifact[~stretch].any()
    # Spikes become unreadable (Q), not ventricular beats.
    near_spike = np.array([np.min(np.abs(spikes - tt)) < 0.06 for tt in t])
    assert (b.label[near_spike] == "Q").all()
    ev = find_events(b, FS)
    assert any(e.kind == "Possible artefact" for e in ev)


def test_marching_through_keeps_real_ventricular_beats():
    # Real ventricular beats replace the sinus QRS (compensatory pause, run):
    # the sinus grid is broken, so the check must not touch them.
    truth = rhythm(90.0, rr=0.8, events={30: "vrun3", 50: "pvc", 52: "pvc", 54: "pvc", 56: "pvc"})
    x = ecg(truth, 90.0)
    _, b = analyse_beats(x, FS)
    assert not b.artifact.any()
    ev = find_events(b, FS)
    assert sum(e.kind == "V run (>=3)" for e in ev) == 1
    m = _match(truth, b.idx)
    kinds = np.array([k for _, k in truth])
    assert (b.label[m[(m >= 0) & (kinds == "V")]] == "V").all()


# ---------------------------------------------------------------- confidence

def test_confidence_from_context(mixed):
    _, b = mixed
    ev = [e for e in find_events(b, FS) if e.kind == "V run (>=3)"]
    run = ev[0]
    rate_events(ev, b, FS, Context())
    assert run.confidence_level == "high" and run.confidence_note == ""

    acc_t = np.arange(0, 180, 0.04)
    movement = np.where(np.abs(acc_t - run.t_start) < 3, 400.0, 20.0)
    ctx = Context(accel_t=acc_t, movement=movement,
                  no_contact=[(run.t_start - 1, run.t_end + 1)],
                  device_rr=[(run.t_end + 0.5, 1500.0), (run.t_end + 1.5, 1500.0)])
    rate_events(ev, b, FS, ctx)
    assert run.confidence_level == "low"
    for why in ("contact", "movement", "disagree"):
        assert why in run.confidence_note


def test_rr_agreement_matches_our_beats(mixed):
    from ecgpipe.context import rr_agreement
    _, b = mixed
    t_end = b.idx[1:] / FS
    rr = np.diff(b.idx) / FS * 1000
    dev = [(te + 1.0, r) for te, r in zip(t_end, rr)][10:40]
    assert rr_agreement(b, FS, dev, 0, 180) == pytest.approx(1.0)
