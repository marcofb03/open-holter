"""Beat detection, signal quality and rule-based beat labelling.

Labels (AAMI-style):
    N  normal (sinus) beat
    S  supraventricular premature beat (PAC-like): early, normal shape
    V  ventricular premature beat (PVC-like): different / wider shape
    Q  unclassifiable (noisy segment)

This is a transparent rule-based first version. It is meant to flag
candidates for a cardiologist to read, not to diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass

import neurokit2 as nk
import numpy as np
from scipy import signal

PRE_S, POST_S = 0.20, 0.40     # beat window around the R peak
QRS_HALF_S = 0.10              # QRS region used for shape comparison


@dataclass
class Beats:
    idx: np.ndarray          # R-peak sample indices
    rr_prev: np.ndarray      # seconds (nan for first beat)
    rr_next: np.ndarray      # seconds (nan for last beat)
    prematurity: np.ndarray  # rr_prev / local median RR (<1 = early)
    corr_qrs: np.ndarray     # correlation of QRS with the normal template
    width_ratio: np.ndarray  # QRS width / template QRS width
    noisy: np.ndarray        # bool
    label: np.ndarray        # 'N' / 'S' / 'V' / 'Q'


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(3, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def detect_r_peaks(clean: np.ndarray, fs: float) -> np.ndarray:
    _, info = nk.ecg_peaks(clean, sampling_rate=int(round(fs)), method="neurokit",
                           correct_artifacts=False)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    # Snap each peak to the largest absolute deflection nearby, then de-duplicate.
    half = int(0.05 * fs)
    snapped = []
    for p in peaks:
        a, b = max(p - half, 0), min(p + half + 1, len(clean))
        snapped.append(a + int(np.argmax(np.abs(clean[a:b]))))
    snapped = np.unique(snapped)
    keep = [snapped[0]] if snapped.size else []
    for p in snapped[1:]:
        if p - keep[-1] >= int(0.2 * fs):
            keep.append(p)
        elif abs(clean[p]) > abs(clean[keep[-1]]):
            keep[-1] = p
    return np.asarray(keep, dtype=int)


def noise_mask(raw: np.ndarray, clean: np.ndarray, fs: float, win_s: float = 2.0) -> np.ndarray:
    """Per-sample boolean: True where the signal is unusable."""
    n = len(raw)
    w = int(win_s * fs)
    mask = np.zeros(n, dtype=bool)
    # Energy above 40 Hz (muscle / friction noise). Baseline wander must not count here.
    if fs > 90:
        sos = signal.butter(3, 40.0, btype="high", fs=fs, output="sos")
        hf = signal.sosfiltfilt(sos, raw)
    else:
        hf = np.zeros(n)
    for a in range(0, n, w):
        b = min(a + w, n)
        seg = clean[a:b]
        if seg.size < w // 2:
            continue
        flat = np.std(raw[a:b]) < 0.02                  # electrode off / no contact
        huge = np.max(np.abs(seg)) > 5.0                 # > 5 mV: motion / saturation
        p_sig = np.mean(seg ** 2) + 1e-12
        p_hf = np.mean(hf[a:b] ** 2)
        emg = p_hf / p_sig > 0.5                         # muscle / friction noise
        if flat or huge or emg:
            mask[a:b] = True
    return mask


def _windows(x: np.ndarray, idx: np.ndarray, pre: int, post: int) -> np.ndarray:
    out = np.full((len(idx), pre + post), np.nan)
    for k, p in enumerate(idx):
        a, b = p - pre, p + post
        if a >= 0 and b <= len(x):
            seg = x[a:b]
            out[k] = seg - np.median(seg[: max(pre // 3, 1)])
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _qrs_width(w: np.ndarray, pre: int, fs: float) -> float:
    """Width (s) of the main deflection: samples above 30% of the peak, contiguous with R."""
    if np.any(np.isnan(w)):
        return np.nan
    amp = np.abs(w)
    thr = 0.3 * amp[pre]
    lo = pre
    while lo > 0 and amp[lo - 1] > thr and pre - lo < 0.15 * fs:
        lo -= 1
    hi = pre
    while hi < len(w) - 1 and amp[hi + 1] > thr and hi - pre < 0.15 * fs:
        hi += 1
    return (hi - lo + 1) / fs


def analyse_beats(raw: np.ndarray, fs: float) -> tuple[np.ndarray, Beats]:
    clean = bandpass(raw, fs)
    idx = detect_r_peaks(clean, fs)
    noisy_samples = noise_mask(raw, clean, fs)
    n = len(idx)

    rr = np.diff(idx) / fs
    rr_prev = np.concatenate(([np.nan], rr))
    rr_next = np.concatenate((rr, [np.nan]))

    # Local reference RR: median of surrounding intervals.
    local = np.full(n, np.nan)
    for i in range(n):
        seg = rr[max(i - 10, 0): i + 10]
        if seg.size >= 4:
            local[i] = np.median(seg)
    prematurity = rr_prev / local

    pre, post = int(PRE_S * fs), int(POST_S * fs)
    q = int(QRS_HALF_S * fs)
    W = _windows(clean, idx, pre, post)

    corr_qrs = np.full(n, np.nan)
    width_ratio = np.full(n, np.nan)
    noisy = noisy_samples[idx] if n else np.zeros(0, bool)
    # Physiologically impossible spacing (> 250 bpm) means noise.
    noisy |= (rr_prev < 0.24) | (rr_next < 0.24)

    # Templates are rebuilt per block of beats to follow posture / electrode changes.
    block = 400
    for s in range(0, n, block):
        e = min(s + block, n)
        Wb = W[s:e]
        ok = ~np.isnan(Wb).any(axis=1) & ~noisy[s:e]
        if ok.sum() < 5:
            continue
        tmpl0 = np.median(Wb[ok], axis=0)
        c0 = np.array([_corr(w[pre - q: pre + q], tmpl0[pre - q: pre + q]) for w in Wb])
        typical = ok & (c0 > 0.9) & (prematurity[s:e] > 0.9) & (prematurity[s:e] < 1.15)
        tmpl = np.median(Wb[typical], axis=0) if typical.sum() >= 5 else tmpl0
        tw = _qrs_width(tmpl, pre, fs)
        for k in range(s, e):
            corr_qrs[k] = _corr(W[k][pre - q: pre + q], tmpl[pre - q: pre + q])
            wk = _qrs_width(W[k], pre, fs)
            width_ratio[k] = wk / tw if tw and not np.isnan(wk) else np.nan

    label = np.full(n, "N", dtype="<U1")
    is_v = (corr_qrs < 0.75) | ((width_ratio > 1.5) & (corr_qrs < 0.9))
    is_s = (prematurity < 0.85) & (corr_qrs >= 0.85) & ~(width_ratio > 1.3)
    label[is_s] = "S"
    label[is_v] = "V"
    label[noisy | np.isnan(corr_qrs)] = "Q"

    return clean, Beats(idx, rr_prev, rr_next, prematurity, corr_qrs, width_ratio, noisy, label)
