"""H10-like signals, labels, training examples and augmentation for the fine-tuning kit.

Storage vs. conversion. Shards keep every example at the SOURCE sampling rate
(250 Hz Icentia11k, 500 Hz PTB-XL, ...) as int16 microvolts, unfiltered, with a
margin on both sides. The conversion to what the phone sees happens when the
example is read:
    val / test (deterministic, identical every time)
        polyphase resampling to 130 Hz -> H10-like front end (first-order 0.5 Hz
        high-pass, as an AC-coupled strap amplifier, then a 2nd-order 40 Hz
        low-pass; both causal, like hardware) -> crop the margin
    train, conversion "frames" (the default; convert_frames): the same deterministic
        conversion, but the 130 Hz grid starts at one of the n_frames(fs) distinct
        sub-sample positions relative to the source samples (25 for 250 Hz, 50 for
        500 Hz, 64 for 128 Hz). The H10 samples the heart at moments unrelated to the
        beat, so every one of these frames is an equally valid H10-like version of the
        same recording, and none is the "right" one. Each example gets a different frame
        every epoch (frame_ids), and every epoch uses all frames equally. Windows also get
        a random crop start (a different 10 s frame of the same real signal). Nothing is
        added to the signal: no gain, filter, noise or rate changes.
    train, conversion "random" (h10.convert with rng + augment; not used by default)
        resampling: source -> r Hz (r uniform 125-135) -> 130 Hz, each stage with
        a random method (polyphase, FFT or linear interpolation) and the second
        with a random sub-sample phase offset; front end with random corners
        (high-pass 0.3-0.8 Hz, order 1-2; low-pass 30-45 Hz, order 1-3); random
        crop shift; amplitude x0.6-1.6; polarity flip (p 0.2); baseline wander
        (0.05-0.5 Hz, up to 0.5 mV); NSTDB electrode-motion / muscle / baseline
        noise at -6 to 24 dB SNR (p 0.5, shards.Noise)
Keeping the source rate costs about 2x the space of 130 Hz shards (Icentia11k
~25 MB per patient, ~280 GB in all) but gives a genuinely different resampling of
the same beats every epoch; the file count is unchanged (a few hundred shards).
The front-end filter is an approximation of the strap, not a measurement of it.

Examples:
    windows  10 s (1,300 samples at 130 Hz) on a 10 s grid, with the 9 window
             targets WINDOW_TARGETS (ai/benchmark/findings.py window_targets plus
             noise: >= half of the window's beats unreadable); -1 = unknown
    beats    640 samples at 130 Hz (4.9 s) centred on each annotated beat, class
             0 N, 1 S, 2 V (AAMI V and F: recall first), 3 noise (unreadable beat)
A per-patient budget keeps every abnormal example (up to a cap) and a random
sample of normal ones.

Noise relabelling (train only): a window whose added noise is below -3 dB SNR
gets noise = 1; a NORMAL beat below -3 dB becomes class 3 (noise). Abnormal
beats keep their class at any SNR, so the noise class can never learn to hide
a PVC (recall first).
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, resample, resample_poly

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "benchmark")]
import findings as F  # noqa: E402

FS = 130
WIN = 1300
BEAT = 640
MARGIN_S = {"win": 1.0, "beat": 0.75}
OUT_LEN = {"win": WIN, "beat": BEAT}
BEAT_CLASSES = ("N", "S", "V", "noise")
WINDOW_TARGETS = F.HEAD_TARGETS + ("noise",)
BEAT_CODE = {"N": 0, "S": 1, "V": 2, "F": 2, "Q": 3}
NOISE_RELABEL_DB = -3.0


# --------------------------------------------------------------------------
# Signal conversion
# --------------------------------------------------------------------------

def raw16(x: np.ndarray) -> np.ndarray:
    """mV -> int16 uV, median removed (no filtering)."""
    x = np.nan_to_num(np.asarray(x, np.float64))
    return np.clip(np.round(1000 * (x - np.median(x))), -32768, 32767).astype(np.int16)


def _poly(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    r = Fraction(fs_out / fs_in).limit_denominator(200)
    return resample_poly(x, r.numerator, r.denominator, axis=-1)


def _resample(x: np.ndarray, fs_in: float, fs_out: float, method: str, phase: float = 0.0) -> np.ndarray:
    n_out = int(round(x.shape[-1] * fs_out / fs_in))
    if method == "poly":
        y = _poly(x, fs_in, fs_out)[..., :n_out]
    elif method == "fft":
        y = resample(x, n_out, axis=-1)
    else:  # linear
        t_in = np.arange(x.shape[-1]) / fs_in
        t_out = (np.arange(n_out) + phase) / fs_out
        return np.stack([np.interp(t_out, t_in, row) for row in x.reshape(-1, x.shape[-1])]).reshape(x.shape[:-1] + (n_out,))
    if phase:  # sub-sample shift by linear interpolation on the output grid
        g = np.arange(y.shape[-1]) + phase
        y = np.stack([np.interp(g, np.arange(y.shape[-1]), row) for row in y.reshape(-1, y.shape[-1])]).reshape(y.shape)
    return y


def front_end(x: np.ndarray, hp: float = 0.5, hp_order: int = 1, lp: float = 40.0, lp_order: int = 2) -> np.ndarray:
    """Causal H10-like front end along the last axis (filter state started at the first sample)."""
    for b, a in (butter(hp_order, hp, "highpass", fs=FS), butter(lp_order, lp, "lowpass", fs=FS)):
        zi = lfilter_zi(b, a) * x[..., :1]  # steady state at the first sample: no start-up step
        x, _ = lfilter(b, a, x, axis=-1, zi=zi)
    return x


def to_h10(x: np.ndarray, fs: float) -> np.ndarray:
    """Whole record in mV at fs -> deterministic H10-like int16 uV at 130 Hz."""
    y = front_end(_poly(np.nan_to_num(np.asarray(x, np.float64)), fs, FS) * 1000.0)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def convert(seg: np.ndarray, fs: float, kind: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """(n, L) int16 uV source-rate segments with margin -> (n, OUT_LEN) float32 uV at 130 Hz.

    rng None: the deterministic val/test path. Otherwise one random draw for the whole
    chunk of the resampling chain and front end (cheap, still new every epoch), per-row
    crop shift.
    """
    x = seg.astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    m = int(round(MARGIN_S[kind] * FS))
    n_out = OUT_LEN[kind]
    if rng is None:
        y = front_end(_poly(x, fs, FS))
        return y[:, m:m + n_out].astype(np.float32)
    r = rng.uniform(125.0, 135.0)
    m1, m2 = rng.choice(["poly", "fft", "linear"], 2)
    y = _resample(_resample(x, fs, r, m1), r, FS, m2, phase=rng.uniform(0, 1))
    y = front_end(y, hp=rng.uniform(0.3, 0.8), hp_order=int(rng.integers(1, 3)),
                  lp=rng.uniform(30.0, 45.0), lp_order=int(rng.integers(1, 4)))
    max_shift = m if kind == "win" else 3  # beats stay (nearly) centred
    s = m + rng.integers(-max_shift, max_shift + 1, len(y))
    s = np.clip(s, 0, y.shape[1] - n_out)
    return np.stack([y[i, a:a + n_out] for i, a in enumerate(s)]).astype(np.float32)


def n_frames(fs: float) -> int:
    """Distinct positions of the 130 Hz grid relative to the samples of an fs-Hz source.

    Starting the conversion j source samples later moves the 130 Hz grid by j * 130 / fs
    output samples. With 130 / fs = p / q in lowest terms the fractional parts of j * p / q
    are exactly the q values 0, 1/q, ..., (q-1)/q, so j = 0 .. q-1 gives every frame once,
    each with the same polyphase filter (no extra interpolation)."""
    return Fraction(FS / fs).limit_denominator(200).denominator


def frame_ids(uid: np.ndarray, epoch: int, fs: float) -> np.ndarray:
    """Frame of each example in this epoch: a fixed per-example start (hash of its uid) plus the
    epoch, mod n_frames: a different frame every epoch for each example, all frames equally used."""
    h = (np.asarray(uid, np.uint64) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(33)
    return ((h + np.uint64(epoch)) % np.uint64(n_frames(fs))).astype(np.int64)


def convert_frames(seg: np.ndarray, fs: float, kind: str, frames: np.ndarray,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Like convert(seg, fs, kind) (deterministic), but row i is converted on frame frames[i]
    (0 .. n_frames(fs)-1; frame 0 is exactly the deterministic conversion). Beats stay centred
    to within half a sample. rng: random crop start for windows only (train)."""
    x = seg.astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    m = int(round(MARGIN_S[kind] * FS))
    n_out = OUT_LEN[kind]
    r = Fraction(FS / fs).limit_denominator(200)
    out = np.empty((len(x), n_out), np.float32)
    frames = np.asarray(frames) % r.denominator
    for j in np.unique(frames):
        rows = np.flatnonzero(frames == j)
        y = front_end(_poly(x[rows, j:], fs, FS))     # grid starts j source samples later
        a = np.full(len(rows), m - int(round(j * r.numerator / r.denominator)))
        if rng is not None and kind == "win":
            a = a + rng.integers(-m, m + 1, len(rows))
        a = np.clip(a, 0, y.shape[1] - n_out)
        out[rows] = np.stack([y[k, s:s + n_out] for k, s in enumerate(a)])
    return out


def augment(x: np.ndarray, y: np.ndarray, kind: str, rng: np.random.Generator, noise=None) -> tuple[np.ndarray, np.ndarray]:
    """H10-domain augmentation of converted 130 Hz examples (train split only)."""
    n, L = x.shape
    x = x * rng.uniform(0.6, 1.6, (n, 1)) * np.where(rng.random((n, 1)) < 0.2, -1.0, 1.0)
    t = np.arange(L) / FS
    f = rng.uniform(0.05, 0.5, (n, 1))
    x = x + rng.uniform(0, 500, (n, 1)) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi, (n, 1))) * (rng.random((n, 1)) < 0.5)
    y = y.copy()
    if noise is not None:
        on = rng.random(n) < 0.5
        snr = rng.uniform(-6.0, 24.0, n)
        if on.any():
            x[on] = noise.add(x[on], rng, snr[on])
        low = on & (snr < NOISE_RELABEL_DB)
        if kind == "win":
            y[low, WINDOW_TARGETS.index("noise")] = 1
        else:
            y[low & (y == 0)] = 3
    return x.astype(np.float32), y


# --------------------------------------------------------------------------
# Labels and examples
# --------------------------------------------------------------------------

def window_labels(db: str, ref: dict, dur_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Window starts (s, 10 s grid) and int8 targets (n, 9): 1 / 0 / -1 unknown."""
    t0s = np.arange(0.0, dur_s - F.WIN_S + 1e-9, F.WIN_S)
    t = F.window_targets({"ref": ref, "win_t": t0s, "db": db})
    y = np.stack([t[k] for k in F.HEAD_TARGETS] + [_noise(ref, t0s)], 1) if len(t0s) else np.zeros((0, 9))
    return t0s, np.where(np.isnan(y), -1, y).astype(np.int8)


def _noise(ref: dict, t0s: np.ndarray) -> np.ndarray:
    if not ref["has_beats"]:
        return np.full(len(t0s), np.nan)
    w = np.floor(ref["beats_t"] / F.WIN_S).astype(int)
    keep = w < len(t0s)
    tot = np.bincount(w[keep], minlength=len(t0s))
    q = np.bincount(w[keep], weights=(ref["beats_c"][keep] == "Q").astype(float), minlength=len(t0s))
    return np.where(tot > 0, (q >= tot / 2).astype(float), 1.0)  # no beat at all in 10 s: unreadable


def beat_labels(ref: dict) -> tuple[np.ndarray, np.ndarray]:
    """Beat times (s) and classes 0-3."""
    if not ref["has_beats"]:
        return np.zeros(0), np.zeros(0, np.int8)
    return np.asarray(ref["beats_t"], np.float64), np.array([BEAT_CODE[c] for c in ref["beats_c"]], np.int8)


def segments(x16: np.ndarray, fs: float, t_start: np.ndarray, kind: str) -> np.ndarray:
    """Source-rate segments [t - margin, t + length + margin) (edge-padded), (n, L) int16."""
    m = MARGIN_S[kind]
    L = int(round((OUT_LEN[kind] / FS + 2 * m) * fs))
    pad = L
    xp = np.pad(x16, (pad, pad), mode="edge")
    s = np.round((np.asarray(t_start, np.float64) - m) * fs).astype(np.int64) + pad
    return xp[s[:, None] + np.arange(L)[None, :]] if len(s) else np.zeros((0, L), np.int16)


def frames_130(x130: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """(n, BEAT) frames of a 130 Hz signal centred on sample indices (edge-padded), for inference."""
    h = BEAT // 2
    xp = np.pad(np.asarray(x130, np.float64), (h, h), mode="edge")
    c = np.asarray(centres, np.int64)
    return xp[c[:, None] + np.arange(BEAT)[None, :]] if len(c) else np.zeros((0, BEAT))


def examples(db: str, x16: np.ndarray, fs: float, ref: dict) -> dict:
    """Every window and beat example of one record (x16: raw16 at the source rate)."""
    dur = len(x16) / fs
    t0s, y = window_labels(db, ref, dur)
    bt, bc = beat_labels(ref)
    ok = (bt >= 0) & (bt < dur)
    bt, bc = bt[ok], bc[ok]
    half = BEAT / FS / 2
    return {"win_x": segments(x16, fs, t0s, "win"), "win_y": y, "win_t": t0s.astype(np.float32),
            "beat_x": segments(x16, fs, bt - half, "beat"), "beat_y": bc, "beat_t": bt.astype(np.float32)}


def budget_windows(y: np.ndarray, rng: np.random.Generator, n_neg: int, cap_pos: int) -> np.ndarray:
    """Indices: every window with a positive target (at most cap_pos) + n_neg others."""
    pos = np.where((y == 1).any(1))[0]
    if len(pos) > cap_pos:
        pos = rng.choice(pos, cap_pos, replace=False)
    rest = np.setdiff1d(np.arange(len(y)), pos)
    neg = rng.choice(rest, min(n_neg, len(rest)), replace=False) if len(rest) else rest
    return np.sort(np.concatenate([pos, neg]))


def budget_beats(cls: np.ndarray, rng: np.random.Generator, caps: dict[int, int]) -> np.ndarray:
    out = []
    for c, cap in caps.items():
        i = np.where(cls == c)[0]
        out.append(rng.choice(i, cap, replace=False) if len(i) > cap else i)
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


def apply_budget(ex: dict, rng: np.random.Generator, budget: dict) -> dict:
    """Keep a patient's budget of examples (ex: all its records concatenated)."""
    wi = budget_windows(ex["win_y"], rng, budget["win_neg"], budget["win_pos"])
    bi = budget_beats(ex["beat_y"], rng, budget["beats"])
    out = {k: v[wi] for k, v in ex.items() if k.startswith("win_")}
    out.update({k: v[bi] for k, v in ex.items() if k.startswith("beat_")})
    return out
