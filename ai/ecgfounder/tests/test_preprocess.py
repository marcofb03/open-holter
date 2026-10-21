"""Checks that need neither the weights nor the databases.

    python -m pytest tests/        (from ai/ecgfounder/)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import eval as ev  # noqa: E402
import infer  # noqa: E402


def original_filter_bandpass(signal, fs):
    """Verbatim copy of ECGFounder util.filter_bandpass (MIT, PKUDigitalHealth)."""
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    return filtered_signal - baseline


def original_z_score(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def test_preprocessing_matches_ecgfounder():
    rng = np.random.default_rng(0)
    t = np.arange(infer.WIN) / infer.FS
    for _ in range(3):
        x = (np.sin(2 * np.pi * 1.1 * t) ** 31 + 0.3 * np.sin(2 * np.pi * 0.2 * t)
             + 0.05 * np.sin(2 * np.pi * 50 * t) + 0.05 * rng.standard_normal(infer.WIN))
        ref = original_z_score(original_filter_bandpass(x[None, :], infer.FS))
        ours = infer.preprocess_windows(x[None, :])[0]
        assert ours.shape == (1, infer.WIN)
        assert np.abs(ours - ref).max() < 1e-5


def test_resample_and_windows():
    x = np.zeros(130 * 60)  # 60 s at 130 Hz
    y = infer.resample_to_500(x, 130)
    assert len(y) == 500 * 60
    st = infer.window_starts(len(y), 5.0)
    assert len(st) == 11 and st[-1] + infer.WIN == len(y)
    assert len(infer.window_starts(infer.WIN - 1)) == 0


def test_groups_are_valid_labels():
    assert len(infer.TASKS) == 150
    for target, group in ev.TARGETS.items():
        assert group in infer.GROUP_IDX, target


def _ann(beats, rhythm, fs=100):
    """Fake wfdb annotation: beats = [(t, sym)], rhythm = [(t, '(AFIB')]."""
    ev_ = sorted([(t, s, "") for t, s in beats] + [(t, "+", r) for t, r in rhythm])
    return SimpleNamespace(sample=np.array([round(t * fs) for t, _, _ in ev_]),
                           symbol=[s for _, s, _ in ev_], aux_note=[a for _, _, a in ev_])


def test_window_truth():
    fs = 100
    beats = [(0.5 + 0.8 * k, "N") for k in range(75)]          # 60 s, 75 bpm
    beats[5] = (beats[5][0], "V")                               # PVC at 4.5 s
    for k in (40, 41, 42):                                      # V run at 32.5-34.1 s
        beats[k] = (beats[k][0], "V")
    beats[20] = (beats[20][0], "A")                             # PAC at 16.5 s
    ann = _ann(beats, [(0, "(N"), (47, "(AFIB")], fs)
    t0 = np.arange(0, 51, 5.0)  # windows start 0,5,...,50
    tr = ev.window_truth(ann, fs, 60 * fs, t0)
    assert tr["pvc"][0] == 1 and tr["pvc"][1] == 0
    assert tr["sve"][2] == 1 and tr["sve"][3] == 1 and tr["sve"][0] == 0
    assert tr["vrun"][6] == 1 and tr["vrun"][5] == 1 and tr["vrun"][0] == 0
    assert tr["af"][7] == 0 and tr["af"][9] == 1 and tr["af"][10] == 1
    assert np.isnan(tr["af"][8])                               # 40-50 s: only 3 s AF -> ambiguous
    assert tr["sinus"][0] == 1 and tr["sinus"][10] == 0
    assert tr["brady"][1] == 0 and tr["tachy"][1] == 0          # 75 bpm


def test_vf_windows_left_out_of_beat_targets():
    fs = 100
    beats = [(0.5 + 0.8 * k, "N") for k in range(25)]           # 0-20 s
    beats += [(40.5 + 0.8 * k, "N") for k in range(20)]         # 40-56 s
    ann = _ann(beats + [(22.0, "["), (35.0, "]")], [(0, "(N")], fs)
    tr = ev.window_truth(ann, fs, 60 * fs, np.array([0.0, 20.0, 45.0]))
    assert tr["pvc"][0] == 0 and np.isnan(tr["pvc"][1]) and tr["pvc"][2] == 0
    assert np.isnan(tr["vrun"][1]) and np.isnan(tr["pause"][1])


def test_subsets():
    assert ev.subset("mitdb", "101") == "mitdb-DS1"
    assert ev.subset("mitdb", "100") == "mitdb-DS2"
    assert ev.subset("svdb", "800") == "svdb"


def test_fast_median_equals_scipy():
    from scipy.ndimage import median_filter

    x = np.random.default_rng(1).standard_normal((4, 3000))
    for k in (5, 201):
        ref = median_filter(x, size=(1, k), mode="constant", cval=0.0)
        assert np.allclose(infer._median_zero_pad(x, k), ref)
