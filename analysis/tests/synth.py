"""Synthetic single-lead ECG for tests (no real data needed).

Beats are sums of Gaussians (P, Q, R, S, T). A ventricular beat has no P
wave, a wide inverted QRS and a discordant T wave. Values are in mV.
"""

from __future__ import annotations

import numpy as np

FS = 130.0

# (amplitude mV, centre s relative to R, width s)
NORMAL = [(0.15, -0.18, 0.025), (-0.10, -0.025, 0.008), (1.0, 0.0, 0.012),
          (-0.25, 0.025, 0.010), (0.30, 0.25, 0.05)]
VENTRICULAR = [(-1.3, 0.0, 0.035), (0.5, 0.07, 0.03), (0.55, 0.30, 0.07)]


def ecg(beats: list[tuple[float, str]], duration: float, fs: float = FS,
        seed: int = 0, noise_mv: float = 0.01) -> np.ndarray:
    """beats: (time_s, 'N' or 'V'). Returns the signal with a little noise and wander."""
    t = np.arange(int(duration * fs)) / fs
    x = np.zeros_like(t)
    for tb, kind in beats:
        for amp, c, w in (NORMAL if kind == "N" else VENTRICULAR):
            x += amp * np.exp(-0.5 * ((t - tb - c) / w) ** 2)
    rng = np.random.default_rng(seed)
    x += noise_mv * rng.standard_normal(len(t)) + 0.05 * np.sin(2 * np.pi * 0.2 * t)
    return x


def rhythm(duration: float, rr: float = 0.8, events: dict[int, str] | None = None,
           start: float = 0.5) -> list[tuple[float, str]]:
    """Regular sinus rhythm; events maps beat number -> 'pac', 'pvc', 'vrun3', 'pause'.

    pac:   early (60% RR) normal beat, then the rhythm resets from it
    pvc:   early (60% RR) ventricular beat with a full compensatory pause
    vrun3: three ventricular beats at 0.4 s intervals, then a pause
    pause: the next beat comes after 2.5 s
    """
    events = events or {}
    out, t, k = [], start, 0
    while t < duration - 1.0:
        e = events.get(k)
        if e == "pac":
            t_early = out[-1][0] + 0.6 * rr
            out.append((t_early, "N"))
            t = t_early + rr
        elif e == "pvc":
            out.append((out[-1][0] + 0.6 * rr, "V"))
            t = out[-2][0] + 2 * rr
        elif e == "vrun3":
            t0 = out[-1][0]
            out += [(t0 + 0.4 * (i + 1), "V") for i in range(3)]
            t = t0 + 1.2 + 1.2 * rr
        elif e == "pause":
            t = out[-1][0] + 2.5
        out.append((t, "N"))
        t += rr
        k += 1
    return [b for b in out if b[0] < duration - 1.0]
