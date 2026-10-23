"""60 s segment examples for the rhythm transformer (../TRANSFORMER_PLAN.md, section 3).

A segment is 60 s of one lead at the SOURCE rate (int16 uV, + 1 s margins), like the window
and beat examples of ai/finetune/h10.py, so every 130 Hz frame can be made when it is read
(h10.convert_frames with kind "seg"). Labels per segment:
    seg_y    (6, 10) int8: the 10 s window targets of h10/findings (vt af svt pvc pac pause brady
             tachy noise) + avb, for each 10 s sixth of the segment; 1 / 0 / -1 unknown
    seg_bt   (MAX_BEATS,) float32: beat times in s from the segment start, -1 = none
    seg_bc   (MAX_BEATS,) int8: beat classes 0 N / 1 S / 2 V / 3 unreadable, -1 = none
    seg_hb   int8: 1 if the source has reviewed beats (then "no beat here" is also a label)
    seg_valid float32: seconds of real signal (records shorter than 60 s are zero-padded and
             their missing sixths are unknown; zero padding, never edge padding, so padding
             cannot look like a flat line from an asystole)
Record-level labels: statements (SPH, Chapman) fill every known sixth; an alarm (VTaC,
Challenge 2015) labels the sixth ending at the alarm onset, and that record gives one segment
ending 10 s after the onset (or at the record end).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "finetune"), str(HERE.parent / "benchmark")]
import findings as F  # noqa: E402
import h10  # noqa: E402

SEG_S = 60.0
SUB = int(round(SEG_S / F.WIN_S))  # 6 sixths of 10 s
SEG_TARGETS = tuple(h10.WINDOW_TARGETS) + ("avb",)
MAX_BEATS = 360  # 60 s at 360 bpm
h10.MARGIN_S.setdefault("seg", 1.0)
h10.OUT_LEN.setdefault("seg", int(round(SEG_S * h10.FS)))  # 7,800 samples at 130 Hz


def window_table(db: str, ref: dict, dur: float) -> tuple[np.ndarray, np.ndarray]:
    """Window starts (10 s grid) and (n, 10) int8 targets for a whole record."""
    t0s, y9 = h10.window_labels(db, ref, dur)
    y = np.concatenate([y9, np.full((len(t0s), 1), -1, np.int8)], 1)
    T = SEG_TARGETS
    if "statements" in ref and len(t0s):
        for k, v in ref["statements"].items():
            y[:, T.index(k)] = v
    if "quality" in ref and len(t0s):
        y[:, T.index("noise")] = _quality_noise(ref["quality"], t0s)
    if "alarm" in ref and len(t0s):
        a = ref["alarm"]
        i = int(np.clip((a["t"] - 1e-6) // F.WIN_S, 0, len(t0s) - 1))
        y[i, T.index(a["target"])] = 1 if a["true"] else (0 if a["target"] == "vt" else -1)
    return t0s, y


def _quality_noise(q: list, t0s: np.ndarray) -> np.ndarray:
    """BUT QDB: 1 if >= half of a fully annotated window is unreadable (class 3), 0 if less, -1 if not annotated."""
    out = np.full(len(t0s), -1, np.int8)
    for i, t0 in enumerate(t0s):
        t1 = t0 + F.WIN_S
        cov = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b, _ in q)
        bad = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b, c in q if c == 3)
        if cov >= F.WIN_S - 0.5:
            out[i] = 1 if bad >= F.WIN_S / 2 else 0
    return out


def segment_starts(ref: dict, n_win: int) -> list[int]:
    """First window of every segment: non-overlapping 60 s blocks, or one segment around an alarm."""
    if n_win == 0:
        return [0] if ref["dur_s"] > 0 else []
    if "alarm" in ref:
        i = int(np.clip((ref["alarm"]["t"] - 1e-6) // F.WIN_S, 0, n_win - 1))
        return [max(0, min(i - SUB + 2, n_win - SUB))]
    return list(range(0, max(n_win - SUB + 1, 1), SUB))


def cut(x16: np.ndarray, fs: float, t0: float) -> np.ndarray:
    """[t0 - margin, t0 + 60 s + margin) at the source rate; zero outside the record."""
    m = h10.MARGIN_S["seg"]
    L = int(round((SEG_S + 2 * m) * fs))
    s = int(round((t0 - m) * fs))
    out = np.zeros(L, np.int16)
    a, b = max(s, 0), min(s + L, len(x16))
    if b > a:
        out[a - s:b - s] = x16[a:b]
    return out


def examples(db: str, x16: np.ndarray, fs: float, ref: dict) -> dict:
    """Every 60 s segment example of one record."""
    dur = len(x16) / fs
    t0s, y = window_table(db, ref, dur)
    n_win = len(t0s)
    bt_all, bc_all = h10.beat_labels(ref)
    rows = {k: [] for k in ("seg_x", "seg_y", "seg_bt", "seg_bc", "seg_hb", "seg_valid", "seg_t")}
    for w0 in segment_starts(ref, n_win):
        t0 = w0 * F.WIN_S
        sy = np.full((SUB, len(SEG_TARGETS)), -1, np.int8)
        k = max(0, min(SUB, n_win - w0))
        sy[:k] = y[w0:w0 + k]
        if n_win == 0 and "statements" in ref:  # shorter than one window: statements still hold
            for kk, v in ref["statements"].items():
                sy[0, SEG_TARGETS.index(kk)] = v
        bt = np.full(MAX_BEATS, -1.0, np.float32)
        bc = np.full(MAX_BEATS, -1, np.int8)
        m = (bt_all >= t0) & (bt_all < t0 + SEG_S)
        nb = min(int(m.sum()), MAX_BEATS)
        bt[:nb] = (bt_all[m] - t0)[:nb]
        bc[:nb] = bc_all[m][:nb]
        rows["seg_x"].append(cut(x16, fs, t0))
        rows["seg_y"].append(sy)
        rows["seg_bt"].append(bt)
        rows["seg_bc"].append(bc)
        rows["seg_hb"].append(np.int8(ref["has_beats"]))
        rows["seg_valid"].append(np.float32(min(SEG_S, max(dur - t0, 0.0))))
        rows["seg_t"].append(np.float32(t0))
    if not rows["seg_x"]:
        L = int(round((SEG_S + 2 * h10.MARGIN_S["seg"]) * fs))
        return {"seg_x": np.zeros((0, L), np.int16), "seg_y": np.zeros((0, SUB, len(SEG_TARGETS)), np.int8),
                "seg_bt": np.zeros((0, MAX_BEATS), np.float32), "seg_bc": np.zeros((0, MAX_BEATS), np.int8),
                "seg_hb": np.zeros(0, np.int8), "seg_valid": np.zeros(0, np.float32), "seg_t": np.zeros(0, np.float32)}
    return {k: np.stack(v) for k, v in rows.items()}


def apply_budget(ex: dict, rng: np.random.Generator, cap_pos: int = 60, n_neg: int = 20) -> dict:
    """Keep every segment with a positive target or an S / V beat (at most cap_pos) + n_neg others."""
    n = len(ex["seg_y"])
    if n == 0:
        return ex
    pos = np.where((ex["seg_y"] == 1).any((1, 2)) | np.isin(ex["seg_bc"], (1, 2)).any(1))[0]
    if len(pos) > cap_pos:
        pos = rng.choice(pos, cap_pos, replace=False)
    rest = np.setdiff1d(np.arange(n), pos)
    neg = rng.choice(rest, min(n_neg, len(rest)), replace=False) if len(rest) else rest
    keep = np.sort(np.concatenate([pos, neg]).astype(np.int64))
    return {k: v[keep] for k, v in ex.items()}


def to_input(x130: np.ndarray, valid_s: np.ndarray) -> np.ndarray:
    """(n, 7800) uV at 130 Hz -> (n, 2, 7800) float32: z-scored signal (over the valid part,
    clipped at +-10) and the validity mask (1 real signal, 0 padding)."""
    n, L = x130.shape
    mask = (np.arange(L)[None, :] < np.round(np.asarray(valid_s)[:, None] * h10.FS)).astype(np.float32)
    x = x130 * mask
    cnt = np.maximum(mask.sum(1, keepdims=True), 1)
    mu = x.sum(1, keepdims=True) / cnt
    sd = np.sqrt((((x - mu) * mask) ** 2).sum(1, keepdims=True) / cnt) + 20.0  # + 20 uV: flat lines stay flat
    z = np.clip((x - mu) / sd, -10, 10) * mask
    return np.stack([z, mask], 1).astype(np.float32)
