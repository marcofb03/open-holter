"""Offline checks of the rhythm transformer kit (no downloads; torch only where marked).

    python -m pytest -q tests        (from ai/rhythm/)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import extra_sources as xs  # noqa: E402
import seg  # noqa: E402


def _ecg(fs: float, dur: float, hr: float = 70.0, v_at: tuple = ()):
    t = np.arange(int(dur * fs)) / fs
    beats = np.arange(0.5, dur - 0.5, 60 / hr)
    x = np.zeros(len(t))
    for b in beats:
        x += 1.2 * np.exp(-0.5 * ((t - b) / 0.012) ** 2) + 0.2 * np.exp(-0.5 * ((t - b - 0.25) / 0.05) ** 2)
    cls = np.array(["N"] * len(beats), dtype="<U1")
    for i in v_at:
        cls[i] = "V"
    ref = {"dur_s": dur, "beats_t": beats, "beats_c": cls, "rhythm": [], "has_beats": True, "has_rhythm": True}
    return seg.h10.raw16(x), ref


def test_parsers():
    assert xs.parse_2015_header("a103l 3 250 82500\n#Asystole\n#False alarm\n") == ("Asystole", False)
    assert xs.parse_2015_header("v1 3 250 1\n#Ventricular_Tachycardia\n#True alarm\n") == ("Ventricular_Tachycardia", True)
    q = xs.butqdb_quality("1,198867,2,1,19525,1,1,7047,2,1,7047,2\n7048,17209,1,7048,17209,1,7048,17209,1,7048,17209,3\n", 1000.0)
    assert q == [(0.0, 7.047, 2), (7.047, 17.209, 3)]
    assert xs.sph_codes("50+346;60") == {"50", "60"}
    st = xs.statements(xs.sph_codes("50+346;60;88"), xs.SPH_MAP)
    assert st["af"] == 1 and st["pvc"] == 1 and st["avb"] == 1 and st["pac"] == 0
    hdr = "JS1 12 500 5000\n#Age: 85\n#Dx: 164889003,233917008\n"
    st = xs.statements(xs.chapman_codes(hdr), xs.CHAPMAN_MAP, xs.CHAPMAN_AMBIGUOUS_AVB)
    assert st["af"] == 1 and st["avb"] == -1
    assert xs.vf_onset("30 2 250 22099250 12:00:00\n#vfon: 13:00:00\n") == 3600.0
    assert xs.pick_lead(["II", "V", "PLETH"]) == 0 and xs.pick_lead(["PLETH", "aVR"]) == 1
    assert xs.pick_lead(["PLETH", "ABP"]) is None


def test_segments_beats_and_vt_label():
    fs = 250.0
    x16, ref = _ecg(fs, 130.0, v_at=(76, 77, 78))  # 3 V beats in a row at 65.6-67.4 s (inside one 10 s window)
    ex = seg.examples("mitdb", x16, fs, ref)
    assert ex["seg_x"].shape == (2, int(62 * fs)) and ex["seg_y"].shape == (2, 6, 10)
    t_run = ref["beats_t"][81]
    k = int((t_run - 60) // 10)
    assert ex["seg_y"][1, k, seg.SEG_TARGETS.index("vt")] == 1
    assert ex["seg_y"][0, :, seg.SEG_TARGETS.index("vt")].max() == 0
    assert ex["seg_hb"].tolist() == [1, 1] and np.allclose(ex["seg_valid"], 60)
    nb = (ex["seg_bc"][1] >= 0).sum()
    assert 60 <= nb <= 80 and (ex["seg_bc"][1] == 2).sum() == 3
    assert np.all(ex["seg_bt"][1][: nb] < 60) and ex["seg_bt"][1][0] >= 0


def test_short_statement_record_and_alarm_record():
    fs = 500.0
    x16, _ = _ecg(fs, 10.0)
    ref = xs._empty_ref(10.0) | {"statements": {"af": 1, "avb": 0}}
    ex = seg.examples("sph", x16, fs, ref)
    assert len(ex["seg_y"]) == 1 and ex["seg_valid"][0] == 10
    y = ex["seg_y"][0]
    assert y[0, seg.SEG_TARGETS.index("af")] == 1 and y[0, seg.SEG_TARGETS.index("avb")] == 0
    assert (y[1:] == -1).all()  # the zero-padded 50 s are unknown
    assert np.all(ex["seg_x"][0, int(12 * fs):] == 0)  # zero padding, never edge padding
    x16, _ = _ecg(250.0, 330.0)
    ref = xs._empty_ref(330.0) | {"alarm": {"t": 300.0, "target": "vt", "true": False}}
    ex = seg.examples("vtac", x16, 250.0, ref)
    assert len(ex["seg_y"]) == 1 and ex["seg_t"][0] == 250.0
    assert ex["seg_y"][0, 4, seg.SEG_TARGETS.index("vt")] == 0 and (ex["seg_y"][0, 5] == -1).all()


def test_to_input_masks_padding():
    x = np.ones((2, 7800)) * 300.0
    out = seg.to_input(x, np.array([60.0, 10.0]))
    assert out.shape == (2, 2, 7800)
    assert out[1, 1, :1300].min() == 1 and out[1, 1, 1301:].max() == 0 and out[1, 0, 2000:].max() == 0


def test_model_shapes():
    torch = pytest.importorskip("torch")
    import model as M

    net = M.build("tiny")
    out = net(torch.randn(2, 2, 7800), torch.zeros(2, M.n_tokens(7800), dtype=torch.bool))
    T = M.n_tokens(7800)
    assert T == 488
    assert out["qrs"].shape == (2, T) and out["beat"].shape == (2, T, 4)
    assert out["win"].shape == (2, 6, 10) and out["recon"].shape == (2, T, 16)


def test_pretrain_then_finetune_end_to_end(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    sys.path.insert(0, str(HERE.parent / "finetune"))
    import shards

    rng = np.random.default_rng(0)
    for split, n_rec in (("train", 6), ("val", 2)):
        parts = []
        for r in range(n_rec):
            x16, ref = _ecg(250.0, 130.0, hr=60 + 10 * r, v_at=(40, 41, 42) if r % 2 else ())
            parts.append(seg.examples("mitdb", x16 + rng.integers(-20, 20, len(x16)).astype(np.int16), 250.0, ref))
        arr = shards.concat(parts)
        arr["seg_pid"] = np.zeros(len(arr["seg_y"]), np.int32)
        shards.write_shard(tmp_path / "seg" / "toy" / split / "task-0000-of-0001.h5", arr, {"fs": 250.0, "patients": ["p"]})
    common = ["--shards", str(tmp_path / "seg"), "--size", "tiny", "--batch", "4", "--val-every", "0.5",
              "--val-chunks", "2", "--warmup-steps", "2", "--data-threads", "1", "--threads", "2"]
    run = [sys.executable, str(HERE / "train_rhythm.py")]
    r1 = subprocess.run(run + ["--stage", "pretrain", "--run", str(tmp_path / "pre"), "--epochs", "2"] + common,
                        capture_output=True, text=True, timeout=600)
    assert r1.returncode == 0, r1.stderr[-2000:]
    assert (tmp_path / "pre" / "DONE").exists() and (tmp_path / "pre" / "best.pt").exists()
    r2 = subprocess.run(run + ["--stage", "finetune", "--run", str(tmp_path / "ft"), "--epochs", "2",
                               "--init", str(tmp_path / "pre" / "best.pt")] + common,
                        capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert "se_at_sp95" in r2.stderr and (tmp_path / "ft" / "DONE").exists()
