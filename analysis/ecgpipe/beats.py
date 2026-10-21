"""Beat detection, signal quality and rule-based beat labelling.

Labels (AAMI-style):
    N  normal (sinus) beat
    S  supraventricular premature beat (PAC-like): early, normal shape
    V  ventricular premature beat (PVC-like): different / wider shape
    Q  unclassifiable (noisy segment)

This is a transparent rule-based first version. It is meant to flag
candidates for a cardiologist to read, not to diagnose. Thresholds were
chosen on MIT-BIH DS1 (see VALIDATION.md for how, and for the one design
change prompted by a held-out result).

Everything here is plain numpy / scipy so it can be ported to the phone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

PRE_S, POST_S = 0.20, 0.45     # beat window around the R peak
QRS_HALF_S = 0.10              # QRS region used for shape comparison
SHAPE_LP_HZ = 20.0             # shape comparison bandwidth
MAX_LAG_S = 0.05               # allowed shift when aligning a beat to the template
TEMPLATE_BLOCK = 300           # beats per template (follows posture / electrode changes)
REFRACTORY_S = 0.25            # min. spacing of two beats (240 bpm); wide PVCs are not counted twice


@dataclass
class Beats:
    idx: np.ndarray          # R-peak sample indices
    rr_prev: np.ndarray      # seconds (nan for first beat)
    rr_next: np.ndarray      # seconds (nan for last beat)
    rr_ref: np.ndarray       # reference (underlying) RR: median of the preceding intervals
    prematurity: np.ndarray  # rr_prev / rr_ref (<1 = early)
    corr_qrs: np.ndarray     # correlation of the aligned QRS with the normal template
    corr_wide: np.ndarray    # same over QRS + ST-T (PVCs usually have a discordant T wave)
    width_ratio: np.ndarray  # QRS width / template QRS width
    amp_ratio: np.ndarray    # QRS peak-to-peak / template peak-to-peak
    irregular: np.ndarray    # bool: beat sits in a chaotic rhythm (e.g. AF), S not assigned
    noise: np.ndarray        # noise index around the beat (see noise_level)
    noisy: np.ndarray        # bool: unusable signal
    label: np.ndarray        # 'N' / 'S' / 'V' / 'Q'
    artifact: np.ndarray | None = None  # bool: removed by the "sinus marches through" check

    def __post_init__(self) -> None:
        if self.artifact is None:
            self.artifact = np.zeros(len(self.idx), dtype=bool)


# --------------------------------------------------------------------------- filters

def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    """Zero-phase 0.5–40 Hz band-pass used for detection and shape analysis."""
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(3, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def qrs_envelope(clean: np.ndarray, fs: float) -> np.ndarray:
    """Smoothed absolute slope in the QRS band (5–30 Hz): high on QRS complexes."""
    sos = signal.butter(2, [5.0, min(30.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    slope = np.abs(np.gradient(signal.sosfiltfilt(sos, clean)) * fs)
    w = max(int(0.12 * fs), 1)
    return np.convolve(slope, np.ones(w) / w, mode="same")


# --------------------------------------------------------------------------- detection

def _rolling_levels(env: np.ndarray, peaks: np.ndarray, fs: float,
                    win_s: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Typical beat height and background level of the envelope, per sample.

    The beat level is the median of the k largest envelope peaks in the window,
    with k = half the window in seconds (assumes >= 30 bpm). Using a median of
    peaks (not the maximum) keeps huge PVCs or artefacts from hiding normal beats.
    """
    n = len(env)
    hop = max(int(win_s * fs / 2), 1)
    k = max(3, int(0.5 * 2 * hop / fs))
    centers, level, floor = [], [], []
    for c in range(0, n + hop, hop):
        a, b = max(c - hop, 0), min(c + hop, n)
        h = np.sort(env[peaks[(peaks >= a) & (peaks < b)]])[::-1]
        centers.append(min(c, n - 1))
        level.append(np.median(h[:k]) if h.size else 0.0)
        floor.append(np.median(env[a:b]) if b > a else 0.0)
    grid = np.arange(n)
    return np.interp(grid, centers, level), np.interp(grid, centers, floor)


def detect_r_peaks(clean: np.ndarray, fs: float, k: float = 0.3) -> np.ndarray:
    """QRS detection on the slope envelope with an adaptive threshold and search-back.

    1. Envelope peaks at least REFRACTORY_S apart above floor + k * (beat level - floor);
       of two peaks < 0.3 s apart the much weaker one is dropped (T wave).
    2. Search-back: inside a gap > 1.5x the local RR, accept the largest peak above
       half that threshold (finds low-amplitude or wide beats).
    3. The R peak is the largest |deflection| within 80 ms of the envelope peak.
    """
    if len(clean) < fs:
        return np.zeros(0, dtype=int)
    env = qrs_envelope(clean, fs)
    refr = int(REFRACTORY_S * fs)
    peaks, _ = signal.find_peaks(env, distance=refr)
    level, floor = _rolling_levels(env, peaks, fs)
    thr = floor + k * (level - floor)
    beats = []
    for p in peaks[env[peaks] > thr[peaks]]:
        # Two detections < 0.3 s apart, one much weaker: the weaker one is a T wave
        # or the second hump of a wide beat. (Both strong: a very fast rhythm.)
        if beats and p - beats[-1] < 0.3 * fs and min(env[p], env[beats[-1]]) < 0.4 * max(env[p], env[beats[-1]]):
            if env[p] > env[beats[-1]]:
                beats[-1] = p
            continue
        beats.append(p)
    beats = np.asarray(beats, dtype=int)

    found = list(beats)
    if len(beats) > 3:
        gaps = np.diff(beats)
        for i, g in enumerate(gaps):
            if g > 1.5 * np.median(gaps[max(i - 8, 0): i + 8]):
                inside = peaks[(peaks > beats[i] + refr) & (peaks < beats[i + 1] - refr)]
                inside = inside[env[inside] > floor[inside] + 0.5 * k * (level[inside] - floor[inside])]
                if inside.size:
                    found.append(inside[np.argmax(env[inside])])

    half = int(0.08 * fs)
    snapped = []
    for p in np.unique(found):
        a, b = max(p - half, 0), min(p + half + 1, len(clean))
        snapped.append(a + int(np.argmax(np.abs(clean[a:b]))))
    snapped = np.unique(snapped)
    keep = [snapped[0]] if snapped.size else []
    for p in snapped[1:]:
        if p - keep[-1] >= refr:
            keep.append(p)
        elif abs(clean[p]) > abs(clean[keep[-1]]):
            keep[-1] = p
    return np.asarray(keep, dtype=int)


# --------------------------------------------------------------------------- noise

def noise_level(clean: np.ndarray, fs: float, idx: np.ndarray, win_s: float = 2.0) -> np.ndarray:
    """Per-sample noise index: QRS-band activity between the beats / on the beats.

    Clean ECG is quiet in the QRS band between beats (P and T waves are slow),
    so the index is ~0.05–0.15. Muscle noise and motion spikes raise it; at
    ~0.5 the noise is as sharp as the QRS complexes themselves (fast rhythms
    with sharp P waves can reach ~0.4, so the cut-off is kept at 0.5).
    """
    n = len(clean)
    level = np.full(n, np.nan)
    if len(idx) < 3:
        return level
    env = qrs_envelope(clean, fs)
    near = np.zeros(n, dtype=bool)
    h = int(0.15 * fs)
    for p in idx:
        near[max(p - h, 0): p + h + 1] = True
    beat_lvl = np.median(env[idx])
    w = int(win_s * fs)
    for a in range(0, n, w):
        b = min(a + w, n)
        between = env[a:b][~near[a:b]]
        if between.size < 0.4 * fs:      # fast rhythm: too little room between beats
            continue
        ib = idx[(idx >= a) & (idx < b)]
        local = np.median(np.r_[env[ib], beat_lvl]) if ib.size else beat_lvl
        level[a:b] = np.percentile(between, 75) / local
    return level


def noise_mask(raw: np.ndarray, clean: np.ndarray, fs: float, level: np.ndarray | None = None,
               win_s: float = 2.0) -> np.ndarray:
    """Per-sample boolean: True where the signal is unusable.

    Per 2-s window: flat line (no contact), > 5 mV deflections (motion /
    saturation), strong energy above 40 Hz (muscle / friction), or a noise
    index (see noise_level) above 0.5.
    """
    n = len(raw)
    w = int(win_s * fs)
    mask = np.zeros(n, dtype=bool)
    if fs > 90:  # energy above 40 Hz; baseline wander must not count here
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
        emg = np.mean(hf[a:b] ** 2) / p_sig > 0.5        # muscle / friction noise
        busy = level is not None and np.nanmax(np.r_[level[a:b], 0.0]) > 0.5
        if flat or huge or emg or busy:
            mask[a:b] = True
    return mask


# --------------------------------------------------------------------------- shape

def _windows(x: np.ndarray, idx: np.ndarray, pre: int, post: int) -> np.ndarray:
    """Beat windows, baseline-corrected on the first part of the PR segment."""
    out = np.full((len(idx), pre + post), np.nan)
    for k, p in enumerate(idx):
        a, b = p - pre, p + post
        if a >= 0 and b <= len(x):
            seg = x[a:b]
            out[k] = seg - np.median(seg[: max(pre // 4, 1)])
    return out


def _detrend(A: np.ndarray) -> np.ndarray:
    """Remove the best straight line from each row (baseline tilt from motion / breathing)."""
    x = np.linspace(-1.0, 1.0, A.shape[-1])
    A = A - A.mean(axis=-1, keepdims=True)
    slope = (A @ x) / (x @ x)
    return A - np.multiply.outer(slope, x)


def _corr_rows(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Correlation of every row of A with vector t, both with their baseline tilt removed."""
    A = _detrend(np.atleast_2d(A))
    t = _detrend(t)
    d = np.sqrt((A * A).sum(axis=1) * (t * t).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A @ t) / d


def _qrs_width(w: np.ndarray, centre: int, fs: float) -> float:
    """QRS duration (s): time holding the central 80% of the slope 'mass' within ±120 ms."""
    if np.any(np.isnan(w)):
        return np.nan
    h = int(0.12 * fs)
    seg = w[max(centre - h, 0): centre + h + 1]
    s = np.abs(np.diff(seg))
    c = np.cumsum(s)
    if c[-1] <= 0:
        return np.nan
    lo, hi = np.searchsorted(c, [0.1 * c[-1], 0.9 * c[-1]])
    return (hi - lo + 1) / fs


def _template(Wq: np.ndarray, ok: np.ndarray, pre_ok: np.ndarray) -> np.ndarray | None:
    """Dominant QRS shape of a block: start from the most 'central' beat, average its look-alikes."""
    cand = np.where(ok & pre_ok)[0]
    if cand.size < 5:
        cand = np.where(ok)[0]
    if cand.size < 5:
        return None
    sample = cand[np.linspace(0, cand.size - 1, min(cand.size, 60)).astype(int)]
    C = np.corrcoef(Wq[sample])
    medoid = sample[int(np.argmax(np.nanmedian(C, axis=1)))]
    sim = _corr_rows(Wq[cand], Wq[medoid])
    members = cand[sim > 0.9]
    return np.median(Wq[members if members.size >= 5 else cand], axis=0)


def _compare(Wb: np.ndarray, tq: np.ndarray, valid: np.ndarray, ok: np.ndarray,
             pre_ok: np.ndarray, centre: int, q: int, L: int, fs: float) -> dict | None:
    """Align each beat on the QRS template tq (best lag within ±L) and measure its shape."""
    m = len(Wb)
    best = np.full(m, -np.inf)
    lag = np.zeros(m, dtype=int)
    for l in range(-L, L + 1):
        c = _corr_rows(Wb[:, centre - q + l: centre + q + l], tq)
        c = np.where(np.isnan(c), -np.inf, c)
        better = c > best
        best[better], lag[better] = c[better], l
    a, b = centre - q, centre + int(0.35 * fs)          # QRS + ST-T
    aligned = np.full((m, b - a), np.nan)
    for r in np.where(valid)[0]:
        aligned[r] = Wb[r, a + lag[r]: b + lag[r]]
    typical = ok & (best > 0.9) & pre_ok
    if typical.sum() < 5:
        typical = ok & (best > 0.9)
    if typical.sum() < 3:
        return None
    tw = np.median(aligned[typical], axis=0)            # full template (QRS + T)
    t_width, t_amp = _qrs_width(tw, q, fs), np.ptp(tw[: 2 * q])
    width = np.full(m, np.nan)
    amp = np.full(m, np.nan)
    for r in np.where(valid)[0]:
        wk = _qrs_width(aligned[r], q, fs)
        width[r] = wk / t_width if t_width else np.nan
        amp[r] = np.ptp(aligned[r, : 2 * q]) / t_amp if t_amp > 0 else np.nan
    return {"corr": np.where(np.isfinite(best), best, np.nan), "wide": _corr_rows(aligned, tw),
            "width": width, "amp": amp, "lag": lag, "typical": typical}


def shape_features(clean: np.ndarray, idx: np.ndarray, fs: float, usable: np.ndarray,
                   prematurity: np.ndarray, next_ratio: np.ndarray) -> tuple[np.ndarray, ...]:
    """Compare every beat with the normal template(s) of its block, after alignment.

    The main template is the dominant QRS shape. If a group of at least 8 beats
    has another shape but arrives on time, with RR before and after within 10%
    of the main rhythm's RR, that shape is accepted as a second normal template
    (posture change, intermittent bundle-branch block, electrode shift). A
    different shape at a faster rate is never accepted, so runs of fast wide
    beats stay V, and only beats that are themselves on time may use the extra
    template.

    Returns corr_qrs, corr_wide, width_ratio, amp_ratio, aligned R index.
    """
    n = len(idx)
    pre, post = int(PRE_S * fs), int(POST_S * fs)
    L = int(MAX_LAG_S * fs)
    q = int(QRS_HALF_S * fs)
    # Shapes are compared below 20 Hz: at 130 Hz a sharp R spike is only 2–3
    # samples wide and its sampled shape would depend on sub-sample timing.
    sos = signal.butter(2, min(SHAPE_LP_HZ, 0.45 * fs), btype="low", fs=fs, output="sos")
    W = _windows(signal.sosfiltfilt(sos, clean), idx, pre + L, post + L)  # margin to slide
    centre = pre + L
    keys = ("corr", "wide", "width", "amp")
    feats = {k: np.full(n, np.nan) for k in keys}
    lag = np.zeros(n, dtype=int)
    rr_prev = np.diff(idx, prepend=np.nan if n else 0) / fs if n else np.zeros(0)

    valid = ~np.isnan(W).any(axis=1)
    with np.errstate(invalid="ignore"):
        pre_ok = (prematurity > 0.9) & (prematurity < 1.15)
        on_time = (prematurity > 0.93) & (prematurity < 1.1) & (next_ratio > 0.9) & (next_ratio < 1.1)
    for s in range(0, n, TEMPLATE_BLOCK):
        e = min(s + TEMPLATE_BLOCK, n)
        # Merge a short last block into the previous one.
        if e == n and n - s < TEMPLATE_BLOCK // 3 and s > 0:
            s = max(n - TEMPLATE_BLOCK, 0)
        Wb, vb = W[s:e], valid[s:e]
        Wq = Wb[:, centre - q: centre + q]
        ok = vb & usable[s:e]
        tq = _template(Wq, ok, pre_ok[s:e])
        f = _compare(Wb, tq, vb, ok, pre_ok[s:e], centre, q, L, fs) if tq is not None else None
        if f is None:
            continue
        rr_main = np.nanmedian(rr_prev[s:e][f["typical"]])
        rr_b = rr_prev[s:e]
        rr_n = np.append(rr_b[1:], rr_prev[e] if e < n else np.nan)
        with np.errstate(invalid="ignore"):
            regular = (np.abs(rr_b / rr_main - 1) < 0.1) & (np.abs(rr_n / rr_main - 1) < 0.1)
        for _ in range(2):  # up to two extra normal shapes per block
            odd = ok & (f["corr"] < 0.9) & on_time[s:e] & regular
            if odd.sum() < 8:
                break
            alt = _template(Wq, odd, odd)
            g = _compare(Wb, alt, vb, odd, odd, centre, q, L, fs) if alt is not None else None
            if g is None:
                break
            members = odd & (g["corr"] >= 0.9)
            if members.sum() < 8:
                break
            # Only on-time beats in a regular stretch may use the extra template.
            better = regular & (g["corr"] > f["corr"])
            for k in (*keys, "lag"):
                f[k] = np.where(better, g[k], f[k])
        for k in keys:
            feats[k][s:e] = f[k]
        lag[s:e] = f["lag"]
    return feats["corr"], feats["wide"], feats["width"], feats["amp"], idx + lag


# --------------------------------------------------------------------------- rhythm

def reference_rr(rr_prev: np.ndarray, n_back: int = 16) -> np.ndarray:
    """Underlying RR for each beat: median of up to n_back intervals before it.

    Looking back (not around) keeps the sinus rate as reference during the first
    beats of a run, instead of letting the run set its own reference.
    """
    n = len(rr_prev)
    ref = np.full(n, np.nan)
    for i in range(n):
        seg = rr_prev[max(i - n_back, 1): i]
        if seg.size < 4:  # start of the recording: use what follows instead
            seg = rr_prev[i + 1: i + 1 + n_back]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 4:
            ref[i] = np.median(seg)
    return ref


def irregular_rhythm(rr_prev: np.ndarray, half: int = 10, spread: float = 0.12,
                     tol: float = 0.04, min_grouped: float = 0.5) -> np.ndarray:
    """True where the rhythm is chaotic (e.g. atrial fibrillation), per beat.

    Over ±half beats, both must hold:
    1. median |successive RR difference| / median RR > spread (RR jumps a lot), and
    2. fewer than half of the intervals have >= 3 others within ±tol of them.
    Sinus rhythm with frequent extra beats jumps too, but its intervals fall into
    a few tight groups (normal RR, coupling interval, pause); in AF they don't.
    """
    n = len(rr_prev)
    d = np.abs(np.diff(rr_prev, prepend=np.nan))
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        a, b = max(i - half, 0), min(i + half + 1, n)
        di, ri = d[a:b], rr_prev[a:b]
        di, ri = di[np.isfinite(di)], ri[np.isfinite(ri)]
        if di.size < 6 or ri.size < 8 or np.median(di) / np.median(ri) <= spread:
            continue
        close = (np.abs(ri[:, None] - ri[None, :]) <= tol * ri[:, None]).sum(axis=1) - 1
        out[i] = np.mean(close >= 3) < min_grouped
    return out


# --------------------------------------------------------------------------- labels

# Thresholds of the labelling rules (chosen on MIT-BIH DS1 only, see VALIDATION.md).
RULES = {
    "v_corr": 0.7,        # QRS correlation below this: clearly different shape -> V
    "v_corr_soft": 0.96,  # below this, V if a second sign is present:
    "v_wide": 1.3,        #   QRS wider than the template by this factor
    "v_amp_hi": 1.6,      #   or much larger ...
    "v_amp_lo": 0.4,      #   ... or much smaller
    "v_corr_wide": 0.5,   #   or different QRS + ST-T (discordant T wave)
    "early": 0.9,         #   or early (RR < 0.9 x reference) ...
    "pause": 1.1,         #   ... and followed by a pause (> 1.1 x reference)
    "s_prem": 0.85,       # S: RR < 0.85 x reference ...
    "s_abrupt": 0.9,      #    and < 0.9 x the RR before it (sudden, not a gradual speed-up)
    "s_corr": 0.9,        #    with a normal QRS (correlation >= this)
    "s_wide": 1.25,       #    and not wider than this
    "s_run_max_s": 30.0,  # a fast stretch keeps the pre-run reference for at most 30 s
                          # (longer = sustained tachycardia, reported as fast rate)
    "march_corr": 0.85,   # artefact check: min. QRS match of the sinus beats inside a 'run'
}


def label_beats(rr_prev: np.ndarray, rr_next: np.ndarray, rr_ref: np.ndarray,
                corr_qrs: np.ndarray, corr_wide: np.ndarray, width_ratio: np.ndarray,
                amp_ratio: np.ndarray, irregular: np.ndarray, noisy: np.ndarray,
                rules: dict | None = None) -> np.ndarray:
    """The N/S/V/Q rules. Inputs are per-beat arrays; nan means 'unknown'."""
    r = {**RULES, **(rules or {})}
    n = len(rr_prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        prematurity = rr_prev / rr_ref
        next_ratio = rr_next / rr_ref
        sudden = rr_prev < r["s_abrupt"] * np.r_[np.nan, rr_prev[:-1]]
        second_sign = ((width_ratio > r["v_wide"]) | (amp_ratio > r["v_amp_hi"])
                       | (amp_ratio < r["v_amp_lo"]) | (corr_wide < r["v_corr_wide"])
                       | ((prematurity < r["early"]) & (next_ratio > r["pause"])))
        # V: clearly different QRS, or moderately different with a second sign.
        is_v = (corr_qrs < r["v_corr"]) | ((corr_qrs < r["v_corr_soft"]) & second_sign)
        # S candidates: normal-looking QRS in a regular rhythm
        # (in atrial fibrillation every beat is "early" or "late").
        s_ok = (corr_qrs >= r["s_corr"]) & ~(width_ratio > r["s_wide"]) & ~irregular & ~is_v

    # S: a sudden early beat, or the next beat of a run that keeps the fast
    # rate; during a run (up to s_run_max_s seconds) the reference stays the one
    # from before the run, after that the new rate is accepted as the rhythm.
    is_s = np.zeros(n, dtype=bool)
    run_ref, run_s = np.nan, 0.0
    for i in range(n):
        if not s_ok[i]:
            run_ref = np.nan
            continue
        if np.isfinite(run_ref):
            is_s[i] = rr_prev[i] < r["s_prem"] * run_ref and run_s < r["s_run_max_s"]
        else:
            is_s[i] = prematurity[i] < r["s_prem"] and sudden[i]
            run_ref, run_s = (rr_ref[i] if is_s[i] else np.nan), 0.0
        run_s += rr_prev[i] if np.isfinite(rr_prev[i]) else 0.0
        if not is_s[i]:
            run_ref = np.nan

    label = np.full(n, "N", dtype="<U1")
    label[is_s] = "S"
    label[is_v] = "V"
    label[noisy | np.isnan(corr_qrs)] = "Q"
    return label


# --------------------------------------------------------------------------- artefacts

def _stretches(label: np.ndarray, min_count: int = 3) -> list[tuple[int, int]]:
    """Index ranges (first, last) holding >= min_count V/S beats, with at most one
    other beat between two of them (a run, or V/S alternating with normal beats)."""
    ectopic = [i for i, lab in enumerate(label) if lab in ("V", "S")]
    out, group = [], []
    for i in ectopic + [None]:
        if i is not None and group and i - group[-1] <= 2:
            group.append(i)
            continue
        if len(group) >= min_count:
            out.append((group[0], group[-1]))
        group = [i] if i is not None else []
    return out


def marching_through(clean: np.ndarray, fs: float, idx: np.ndarray, label: np.ndarray,
                     corr_qrs: np.ndarray, min_corr: float = 0.85) -> list[tuple[int, int, np.ndarray]]:
    """Find V/S stretches that are artefacts on top of an undisturbed sinus rhythm.

    For every stretch with >= 3 V/S beats (a run, or V/S alternating with
    normal beats, see _stretches), the sinus rhythm before it
    is extended through the stretch (one expected beat per sinus RR, re-anchored
    on every beat found). At each expected time the filtered signal is searched
    (±60 ms) for a QRS that matches the shape of the preceding normal beats. If
    every expected sinus beat (at least 2) is there and the extra beats do not
    look like normal QRS complexes, the 'run' is spikes or motion on top of a
    normal rhythm (or normal beats whose shape was bent by motion). A real ventricular run replaces the normal QRS complexes, so
    it fails this test; a real atrial run is made of normal-looking beats, so it
    is never touched.

    Returns (first, last, on_grid) per artefact stretch; on_grid marks the beats
    of the stretch that sit on the sinus grid.
    """
    sos = signal.butter(2, min(SHAPE_LP_HZ, 0.45 * fs), btype="low", fs=fs, output="sos")
    smooth = signal.sosfiltfilt(sos, clean)
    q = int(QRS_HALF_S * fs)
    tol = int(0.06 * fs)
    out = []
    for a, z in _stretches(label):
        normal = [i for i in range(max(a - 20, 0), a) if label[i] == "N"]
        if len(normal) < 4 or normal[-1] < a - 3:
            continue
        # Sinus RR from pairs of consecutive normal beats just before the stretch.
        pairs = [idx[i] - idx[i - 1] for i in normal if i - 1 in normal]
        if len(pairs) < 3:
            continue
        rr = np.median(pairs)
        W = _windows(smooth, idx[normal], q, q)
        tmpl = np.median(W[~np.isnan(W).any(axis=1)], axis=0) if len(W) else None
        if tmpl is None or np.isnan(tmpl).any():
            continue

        first = normal[-1] + 1                        # beats after the last normal one
        anchor, hits, expected = idx[normal[-1]], [], 0
        while anchor + rr < idx[z] + tol:
            centre = int(anchor + rr)
            cands = np.arange(centre - tol, centre + tol + 1)
            cands = cands[(cands - q >= 0) & (cands + q <= len(smooth))]
            expected += 1
            if cands.size == 0:
                break
            Wc = np.stack([smooth[c - q: c + q] for c in cands])
            Wc = Wc - np.median(Wc[:, : max(q // 2, 1)], axis=1, keepdims=True)
            c = _corr_rows(Wc, tmpl)
            k = int(np.nanargmax(c)) if np.isfinite(c).any() else -1
            if k < 0 or c[k] < min_corr:
                break
            amp = np.ptp(Wc[k]) / max(np.ptp(tmpl), 1e-9)
            if not 0.5 < amp < 2.0:
                break
            hits.append(cands[k])
            anchor = cands[k]
        if expected < 2 or len(hits) < expected:
            continue
        on_grid = np.array([np.min(np.abs(np.asarray(hits) - idx[i])) <= tol
                            for i in range(first, z + 1)])
        extra = ~on_grid
        # Extra beats that look like normal QRS complexes are real (e.g. atrial run).
        if extra.any() and np.nanmedian(corr_qrs[first: z + 1][extra]) >= 0.9:
            continue
        out.append((first, z, on_grid))
    return out


def analyse_beats(raw: np.ndarray, fs: float, rules: dict | None = None) -> tuple[np.ndarray, Beats]:
    """Filter, detect, measure and label every beat. `rules` overrides RULES (for tuning)."""
    clean = bandpass(raw, fs)
    idx = detect_r_peaks(clean, fs)
    n = len(idx)
    level = noise_level(clean, fs, idx)
    noisy_samples = noise_mask(raw, clean, fs, level)

    rr = np.diff(idx) / fs
    rr_prev = np.concatenate(([np.nan], rr))
    rr_next = np.concatenate((rr, [np.nan]))
    rr_ref = reference_rr(rr_prev)
    prematurity = rr_prev / rr_ref

    noisy = noisy_samples[idx] if n else np.zeros(0, bool)

    corr_qrs, corr_wide, width_ratio, amp_ratio, _ = shape_features(
        clean, idx, fs, ~noisy, prematurity, rr_next / rr_ref)
    irregular = irregular_rhythm(rr_prev)
    beat_noise = level[idx] if n else np.zeros(0)
    label = label_beats(rr_prev, rr_next, rr_ref, corr_qrs, corr_wide, width_ratio,
                        amp_ratio, irregular, noisy, rules)
    artifact = np.zeros(n, dtype=bool)
    min_corr = {**RULES, **(rules or {})}["march_corr"]
    for a, z, on_grid in marching_through(clean, fs, idx, label, corr_qrs, min_corr):
        rng = np.arange(a, z + 1)
        label[rng[on_grid]] = "N"
        label[rng[~on_grid]] = "Q"
        artifact[rng] = True
    return clean, Beats(idx, rr_prev, rr_next, rr_ref, prematurity, corr_qrs, corr_wide,
                        width_ratio, amp_ratio, irregular, beat_noise, noisy, label, artifact)
