"""Checks that need neither the models nor data:  python -m pytest tests/  (from ai/heartkit/)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hk  # noqa: E402


def test_double_softmax_is_undone():
    p = np.array([[0.54, 0.26, 0.20], [0.004, 0.218, 0.778], [1.0, 0.0, 0.0]])
    q = hk._softmax(p)                       # what the published TFLite files output
    assert np.allclose(hk._to_probs(q), p, atol=1e-6)


def test_logits_get_softmax():
    z = np.array([[2.0, -1.0, 0.5]])
    assert np.allclose(hk._to_probs(z), hk._softmax(z))


def test_frames_shape_and_norm():
    fs = 130
    x = np.sin(2 * np.pi * 1.2 * np.arange(60 * fs) / fs) ** 31
    f = hk.frames(x, fs, np.array([10, 3000, len(x) - 5]))   # incl. both edges
    assert f.shape == (3, hk.FRAME)
    assert np.allclose(f.mean(axis=1), 0, atol=1e-4)
    assert (f.std(axis=1) < 1.0).all()                       # var / (var + 0.01)
