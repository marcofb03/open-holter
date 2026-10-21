"""Extra evidence from the H10 side files, and a confidence value per event.

Nothing here changes beat labels. It only says how much an event can be
trusted, and why, so the reviewer can look at the doubtful ones critically:

- noisy signal or unreadable beats around the event,
- the "sinus marches through" artefact check fired nearby,
- the H10 reported lost skin contact,
- strong movement on the accelerometer,
- the H10's own RR intervals disagree with ours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .beats import Beats
from .events import Event
from .io import Accel, HeartRate

MOVEMENT_MG = 200.0      # std of acceleration magnitude (milli-g) counted as strong movement
RR_MATCH_MS = 40.0       # an H10 RR interval agrees if one of ours is within this
NOISY_INDEX = 0.3        # noise index (see beats.noise_level) counted as noisy


@dataclass
class Context:
    """Side information for one continuous segment, times in seconds from segment start."""
    accel_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    movement: np.ndarray = field(default_factory=lambda: np.zeros(0))   # milli-g, per accel sample
    no_contact: list[tuple[float, float]] = field(default_factory=list)  # intervals
    device_rr: list[tuple[float, float]] = field(default_factory=list)   # (t, rr_ms)


def movement_level(acc: Accel, win_s: float = 2.0) -> np.ndarray:
    """Moving std of the acceleration magnitude (gravity and slow posture drift removed)."""
    mag = np.linalg.norm(acc.xyz, axis=1)
    dt = np.median(np.diff(acc.t)) if len(acc.t) > 1 else 0.04
    w = max(int(round(win_s / dt)), 1)
    k = np.ones(w) / w
    mean = np.convolve(mag, k, mode="same")
    var = np.convolve(mag ** 2, k, mode="same") - mean ** 2
    return np.sqrt(np.clip(var, 0, None))


def contact_gaps(hr: HeartRate) -> list[tuple[float, float]]:
    """Epoch-second intervals where the sensor reported no skin contact."""
    lost = hr.contact_supported & ~hr.contact
    out, start = [], None
    for k, bad in enumerate(lost):
        if bad and start is None:
            start = hr.t[k]
        elif not bad and start is not None:
            out.append((start, hr.t[k]))
            start = None
    if start is not None:
        out.append((start, hr.t[-1] + 1.0))
    return out


def build_context(t0: float, t1: float, accel: list[Accel] | None = None,
                  hr: list[HeartRate] | None = None,
                  device_rr: list[tuple[float, float]] | None = None) -> Context:
    """Collect side information for the segment [t0, t1] (epoch seconds)."""
    ctx = Context()
    ts, mv = [], []
    for a in accel or []:
        sel = (a.t >= t0 - 5) & (a.t <= t1 + 5)
        if sel.sum() > 1:
            ts.append(a.t[sel] - t0)
            mv.append(movement_level(Accel(a.t[sel], a.xyz[sel])))
    if ts:
        ctx.accel_t, ctx.movement = np.concatenate(ts), np.concatenate(mv)
    rr = list(device_rr or [])
    for h in hr or []:
        ctx.no_contact += [(a - t0, b - t0) for a, b in contact_gaps(h) if b >= t0 and a <= t1]
        rr += h.rr
    ctx.device_rr = sorted((t - t0, v) for t, v in rr if t0 <= t <= t1)
    return ctx


def rr_agreement(b: Beats, fs: float, device_rr: list[tuple[float, float]],
                 a_s: float, z_s: float) -> float | None:
    """Fraction of the H10's RR intervals in [a_s, z_s] that match one of ours (None: no data)."""
    dev = [(t, v) for t, v in device_rr if a_s <= t <= z_s]
    if not dev or len(b.idx) < 2:
        return None
    t_end = b.idx[1:] / fs
    ours = np.diff(b.idx) / fs * 1000
    ok = 0
    for t, v in dev:
        # H10 notifications arrive up to ~1–2 s after the beat that ends the interval.
        near = (t_end >= t - 2.5) & (t_end <= t + 0.5)
        ok += bool(np.any(np.abs(ours[near] - v) <= RR_MATCH_MS))
    return ok / len(dev)


def rate_event(e: Event, b: Beats, fs: float, ctx: Context | None = None) -> tuple[float, list[str]]:
    """Confidence 0–1 that the event is real signal, with the reasons it was lowered."""
    a_s, z_s = e.t_start - 2.0, e.t_end + 2.0
    t = b.idx / fs
    near = (t >= a_s) & (t <= z_s)
    score, why = 1.0, []
    if near.any():
        if np.nanmax(np.r_[b.noise[near], 0.0]) > NOISY_INDEX:
            score -= 0.3
            why.append("noisy signal")
        if (b.label[near] == "Q").any():
            score -= 0.2
            why.append("unreadable beats nearby")
        if b.artifact[near].any() and e.kind != "Possible artefact":
            score -= 0.3
            why.append("artefact check fired nearby")
    if ctx is not None:
        if any(a <= z_s and a_s <= z for a, z in ctx.no_contact):
            score -= 0.4
            why.append("skin contact lost (H10)")
        if ctx.movement.size:
            sel = (ctx.accel_t >= a_s) & (ctx.accel_t <= z_s)
            if sel.any() and ctx.movement[sel].max() > MOVEMENT_MG:
                score -= 0.3
                why.append(f"strong movement ({ctx.movement[sel].max():.0f} mG)")
        agree = rr_agreement(b, fs, ctx.device_rr, a_s, z_s)
        if agree is not None and agree < 0.5:
            score -= 0.2
            why.append(f"H10 RR intervals disagree ({100 * agree:.0f}% match)")
    return max(score, 0.0), why


def rate_events(events: list[Event], b: Beats, fs: float, ctx: Context | None = None) -> None:
    """Fill confidence / confidence_note of every event in place."""
    for e in events:
        e.confidence, reasons = rate_event(e, b, fs, ctx)
        e.confidence_note = ", ".join(reasons)
