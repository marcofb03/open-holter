"""Offline checks of the fine-tuning kit (no downloads; torch only where marked).

    python -m pytest -q tests        (from ai/finetune/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import h10  # noqa: E402


def _ecg(fs: float, dur: float = 60.0, hr: float = 70.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(int(dur * fs)) / fs
    beats = np.arange(0.5, dur - 0.5, 60 / hr)
    x = np.zeros(len(t))
    for b in beats:
        x += 1.2 * np.exp(-0.5 * ((t - b) / 0.012) ** 2) + 0.2 * np.exp(-0.5 * ((t - b - 0.25) / 0.05) ** 2)
    return x + 0.3 * np.sin(2 * np.pi * 0.15 * t), beats


def _ref(beats, cls=None):
    return {"dur_s": float(beats[-1] + 1), "beats_t": beats, "beats_c": np.asarray(cls or ["N"] * len(beats)),
            "rhythm": [], "has_beats": True, "has_rhythm": False}


def test_deterministic_conversion_matches_whole_record():
    """A stored source-rate window converted on read equals the same span of the whole-record
    H10-like signal (away from the filters' start-up)."""
    fs = 250.0
    x, beats = _ecg(fs)
    whole = h10.to_h10(x, fs).astype(float)
    ex = h10.examples("mitdb", h10.raw16(x), fs, _ref(beats))
    w = h10.convert(ex["win_x"], fs, "win")
    k = 3  # window 30-40 s
    ref = whole[int(30 * h10.FS):int(40 * h10.FS)]
    a, b = w[k] - w[k].mean(), ref - ref.mean()
    assert w.shape[1] == h10.WIN
    assert np.corrcoef(a, b)[0, 1] > 0.995
    assert abs(a.std() / b.std() - 1) < 0.05


def test_beats_are_centred():
    fs = 360.0
    x, beats = _ecg(fs)
    ex = h10.examples("mitdb", h10.raw16(x), fs, _ref(beats))
    b = h10.convert(ex["beat_x"], fs, "beat")
    assert b.shape[1] == h10.BEAT
    c = h10.BEAT // 2
    peak = np.abs(b[5:-5, c - 40:c + 40]).argmax(1) + c - 40  # neighbours are as tall: look near the centre
    assert np.all(np.abs(peak - c) <= 2)


def test_augmentation_changes_signal_and_keeps_abnormal_labels():
    fs = 250.0
    rng = np.random.default_rng(1)
    x, beats = _ecg(fs)
    cls = ["N"] * len(beats)
    cls[10] = "V"
    ex = h10.examples("mitdb", h10.raw16(x), fs, _ref(beats, cls))

    class LoudNoise:
        def add(self, x, rng, snr):
            return x + 1e5 * rng.standard_normal(x.shape)

    det = h10.convert(ex["beat_x"], fs, "beat")
    aug = h10.convert(ex["beat_x"], fs, "beat", rng)
    assert aug.shape == det.shape and not np.allclose(aug, det)
    _, y = h10.augment(aug, ex["beat_y"].copy(), "beat", np.random.default_rng(3), LoudNoise())
    assert y[10] == 2                      # a V beat is never relabelled as noise
    assert set(np.unique(y)) <= {0, 2, 3}  # normal beats may become noise


def test_window_labels_and_budget():
    fs = 250.0
    x, beats = _ecg(fs, dur=200)
    cls = ["N"] * len(beats)
    for i in (50, 51, 52):
        cls[i] = "V"
    t0s, y = h10.window_labels("mitdb", _ref(beats, cls), len(x) / fs)
    vt = h10.WINDOW_TARGETS.index("vt")
    assert len(t0s) == 20 and y[:, vt].max() == 1 and y.dtype == np.int8
    keep = h10.budget_windows(y, np.random.default_rng(0), n_neg=3, cap_pos=10)
    assert (y[keep] == 1).any(1).sum() == (y == 1).any(1).sum() and len(keep) <= (y == 1).any(1).sum() + 3


def test_shard_roundtrip_and_resumable_stream(tmp_path):
    pytest.importorskip("h5py")
    import shards

    for i in range(2):
        n = 300
        shards.write_shard(tmp_path / f"s{i}.h5", {"beat_x": np.full((n, 8), i, np.int16),
                                                   "beat_y": np.arange(n, dtype=np.int8) % 4}, {"fs": 250.0})
        assert shards.is_complete(tmp_path / f"s{i}.h5")
    files = sorted(tmp_path.glob("*.h5"))
    tf = lambda x, y, fs, rng, **kw: (x + rng.integers(0, 1000) + kw["epoch"], y)  # noqa: E731

    def take(stream, k):
        it = iter(stream)
        return [next(it)[0].sum() for _ in range(k)]

    a = shards.ChunkStream(files, "beat", tf, chunk=100, seed=5, threads=2)
    first = take(a, 4)
    state = a.state()
    rest = take(a, 3)
    b = shards.ChunkStream(files, "beat", tf, chunk=100, seed=5, threads=2)
    b.load_state(state)
    assert take(b, 3) == rest and len(a.index) == 6 and first


def test_models_build_and_forward():
    torch = pytest.importorskip("torch")
    import models

    net = models.build("beat")
    x = models.preprocess("beat", np.random.default_rng(0).standard_normal((2, h10.BEAT)) * 500)
    assert net(torch.from_numpy(x)).shape == (2, models.N_BEAT)
    assert sum(p.numel() for p in net.parameters()) < 600_000
    w = models.build("ecgfounder", pretrained=False)
    xw = models.preprocess("ecgfounder", np.random.default_rng(0).standard_normal((2, h10.WIN)) * 500)
    assert xw.shape == (2, 1, 5000) and w(torch.from_numpy(xw)).shape == (2, models.N_WIN)


def test_frames_cover_every_grid_position_and_frame0_is_deterministic():
    fs = 250.0
    x, beats = _ecg(fs)
    ex = h10.examples("mitdb", h10.raw16(x), fs, _ref(beats))
    q = h10.n_frames(fs)
    assert q == 25 and h10.n_frames(500.0) == 50 and h10.n_frames(128.0) == 64
    seg = ex["beat_x"][5:6].repeat(q, 0)
    det = h10.convert(ex["beat_x"][5:6], fs, "beat")
    fr = h10.convert_frames(seg, fs, "beat", np.arange(q))
    assert np.allclose(fr[0], det[0], atol=1e-3)          # frame 0 = the deterministic conversion
    assert len({np.round(r, 1).tobytes() for r in fr}) == q  # every frame is a different 130 Hz version
    c = h10.BEAT // 2
    peak = np.abs(fr[:, c - 20:c + 20]).argmax(1) + c - 20
    assert np.all(np.abs(peak - c) <= 2)                    # beats stay centred on every frame
    # frame ids: a different frame each epoch for an example, all frames used equally in an epoch
    uid = np.arange(25_000, dtype=np.uint64)
    f0, f1 = h10.frame_ids(uid, 0, fs), h10.frame_ids(uid, 1, fs)
    assert np.all(f1 == (f0 + 1) % q)
    counts = np.bincount(f0, minlength=q)
    assert counts.min() > 0.8 * counts.mean()


def test_plateau_reduces_lr_then_stops_and_resumes():
    import train

    p = train.Plateau(lr_patience=2, stop_patience=5, min_delta=0.01)
    acts = [p.step(s) for s in (0.5, 0.6, 0.605, 0.6, float("nan"), 0.59, 0.6)]
    # 0.605 is not a gain > min_delta; NaN counts as no gain; LR is lowered every 2 misses, stop at 5
    assert acts == ["improved", "improved", "wait", "reduce", "wait", "reduce", "stop"]
    q = train.Plateau(2, 5, 0.01)
    q.load(train.Plateau(2, 5, 0.01).state() | {"best": 0.6, "bad": 4, "since_reduce": 0, "reductions": 2})
    assert q.step(0.7) == "improved" and q.bad == 0  # a real gain resets the patience


def test_purge_deadline_from_oldest_shard(tmp_path):
    import os
    import train

    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    os.utime(a, (1_000_000, 1_000_000))
    assert train.purge_deadline([a, b], 14) == 1_000_000 + 14 * 86400
    assert train.purge_deadline([a, b], 0) == float("inf") and train.purge_deadline([], 14) == float("inf")


def test_prep_repair_only_listed_patients(tmp_path, monkeypatch):
    pytest.importorskip("h5py")
    import json
    import h5py
    import prep
    import sources

    fs = 250.0
    x, beats = _ecg(fs)
    monkeypatch.setattr(sources, "patients", lambda src, policy: [(f"p{i}", "train", None) for i in range(5)])
    monkeypatch.setattr(sources, "read_patient", lambda src, pid, pl, tmp: [("mitdb", h10.raw16(x), fs, _ref(beats))])
    monkeypatch.setattr(sources, "labels_override", lambda ns, ref, y: y)
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(prep, "ProcessPoolExecutor", ThreadPoolExecutor)  # keep the patches (spawn would not)
    prep.run_task("icentia11k", 0, 1, tmp_path, tmp_path, 1, "benchmark", None, only=["p3", "p1"])
    files = list((tmp_path / "icentia11k" / "train").glob("repair-*.h5"))
    assert len(files) == 1
    with h5py.File(files[0]) as f:
        assert sorted(json.loads(f.attrs["patients"])) == ["p1", "p3"]
    with pytest.raises(SystemExit):
        prep.run_task("icentia11k", 0, 1, tmp_path, tmp_path, 1, "benchmark", None, only=["nope"])


def _beat_shards(root, rng):
    import shards

    fs = 250.0
    for split, n in (("train", 4), ("val", 2)):
        parts = []
        for r in range(n):
            x, beats = _ecg(fs, 120.0, hr=60 + 8 * r)
            cls = ["N"] * len(beats)
            for i in range(5, len(beats), 9):
                cls[i] = "V" if i % 2 else "S"
            ex = h10.examples("mitdb", h10.raw16(x + 0.02 * rng.standard_normal(len(x))), fs, _ref(beats, cls))
            ex["win_pid"] = np.zeros(len(ex["win_y"]), np.int32)
            ex["beat_pid"] = np.zeros(len(ex["beat_y"]), np.int32)
            parts.append(ex)
        shards.write_shard(root / "toy" / split / "task-0000-of-0001.h5", shards.concat(parts), {"fs": fs, "patients": ["p"]})


def test_valset_keeps_every_rare_example(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    import valset

    _beat_shards(tmp_path, np.random.default_rng(0))
    out = valset.build("beat", tmp_path, n_rest=10)
    with h5py.File(tmp_path / "toy" / "val" / "task-0000-of-0001.h5") as f:
        y = f["beat_y"][:]
    pos = valset.positives("beat", out)
    assert pos["S"] == int((y == 1).sum()) and pos["V"] == int((y == 2).sum()) and pos["N"] == 10
    vs = valset.ValSet(out, "beat", lambda x, y, fs, rng, **kw: (x, y))
    assert sum(len(vs.read(k)[1]) for k in range(len(vs.index))) == sum(pos.values())


def test_train_switches_to_valset_runs_on_and_rescores(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    import subprocess
    import valset

    _beat_shards(tmp_path / "sh", np.random.default_rng(1))
    run = [sys.executable, str(HERE / "train.py"), "--model", "beat", "--shards", str(tmp_path / "sh"),
           "--run", str(tmp_path / "run"), "--sources", "toy", "--batch", "32", "--val-every", "0.5", "--val-chunks", "2",
           "--warmup-steps", "2", "--data-threads", "1", "--threads", "2", "--purge-days", "0"]
    r = subprocess.run(run + ["--epochs", "1"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    (tmp_path / "run" / "DONE").unlink()
    valset.build("beat", tmp_path / "sh", n_rest=50)
    r = subprocess.run(run + ["--epochs", "3", "--no-early-stop", "--lr-patience", "1"], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "validation set changed (random-chunks" in r.stderr and "smoothed" in r.stderr
    assert "-> stop" not in r.stderr and '"reason": "max_epochs"' in (tmp_path / "run" / "DONE").read_text()
    r = subprocess.run(run + ["--eval", str(tmp_path / "run" / "best.pt")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and (tmp_path / "run" / "rescore.json").exists(), r.stderr[-1500:]
