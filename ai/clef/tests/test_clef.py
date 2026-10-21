"""Checks that need neither the CLEF weights nor any database.

    python -m pytest -q tests        (from ai/clef/)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import clef  # noqa: E402


def original_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Verbatim copy of CLEF clef/data/downstream_dataloader.py bandpass_filter (Clear BSD, Nokia Bell Labs)."""
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = filtfilt(b, a, data[i, :])

    return filtered_data


def original_z_score_normalization(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def test_preprocessing_matches_clef():
    rng = np.random.default_rng(0)
    t = np.arange(clef.WIN) / clef.FS
    x = np.stack([np.sin(2 * np.pi * 1.2 * t) ** 21 + 0.4 * np.sin(2 * np.pi * 0.25 * t)
                  + 0.05 * rng.standard_normal(clef.WIN) for _ in range(3)])
    ref = np.stack([original_z_score_normalization(r) for r in original_bandpass_filter(x, 0.67, 40, fs=500)])
    ours = clef.preprocess_windows(x)
    assert ours.shape == (3, 1, clef.WIN)
    assert np.abs(ours[:, 0] - ref).max() < 1e-5


def test_backbone_is_clef_medium():
    torch = pytest.importorskip("torch")
    net = clef.build_backbone()
    assert abs(sum(p.numel() for p in net.parameters()) / 1e6 - 30.65) < 0.05
    with torch.inference_mode():
        assert net.eval()(torch.zeros(2, 1, clef.WIN)).shape == (2, clef.DIM)
    keys = set(net.state_dict())
    assert "first_conv.conv.weight" in keys and "stage_list.6.block_list.3.se_fc2.bias" in keys
