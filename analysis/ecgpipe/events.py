"""Turn labelled beats into events worth a cardiologist's look."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .beats import Beats

# Higher = more important. Used for sorting the report.
SEVERITY = {
    "V run (>=3)": 5,
    "S run (>=3)": 4,
    "Pause > 2 s": 4,
    "Marker": 4,
    "Fast rate > 150 bpm": 3,
    "Sudden rate jump": 2,
    "V couplet": 3,
    "S couplet": 2,
}


@dataclass
class Event:
    kind: str
    t_start: float      # seconds from recording start
    t_end: float
    n_beats: int = 0
    rate_bpm: float = float("nan")
    note: str = ""

    @property
    def severity(self) -> int:
        return SEVERITY.get(self.kind, 1)


def _runs(labels: np.ndarray, target: str) -> list[tuple[int, int]]:
    out, start = [], None
    for i, l in enumerate(labels):
        if l == target and start is None:
            start = i
        elif l != target and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(labels) - 1))
    return out


def find_events(b: Beats, fs: float, markers_rel: list[float] | None = None) -> list[Event]:
    t = b.idx / fs
    ev: list[Event] = []

    for lab in ("V", "S"):
        for a, z in _runs(b.label, lab):
            n = z - a + 1
            if n < 2:
                continue
            rr = b.rr_prev[a + 1: z + 1]
            rate = 60.0 / np.nanmean(rr) if rr.size and np.isfinite(rr).any() else float("nan")
            kind = f"{lab} run (>=3)" if n >= 3 else f"{lab} couplet"
            ev.append(Event(kind, t[a], t[z], n, rate))

    good = b.label != "Q"
    for i in np.where((b.rr_next > 2.0) & good)[0]:
        if i + 1 < len(t) and good[i + 1]:
            ev.append(Event("Pause > 2 s", t[i], t[i + 1], note=f"{b.rr_next[i]:.1f} s"))

    # Instantaneous rate over 8-beat windows (clean beats only).
    hr = 60.0 / b.rr_prev
    hr[~good] = np.nan
    w = 8
    if len(hr) > w:
        with warnings.catch_warnings():  # windows made only of noisy beats -> nan
            warnings.simplefilter("ignore", RuntimeWarning)
            smooth = np.array([np.nanmedian(hr[max(i - w, 0): i + 1]) for i in range(len(hr))])
        fast = smooth > 150
        for a, z in _runs(np.where(fast, "F", "-"), "F"):
            if t[z] - t[a] >= 10:
                ev.append(Event("Fast rate > 150 bpm", t[a], t[z], z - a + 1,
                                float(np.nanmax(smooth[a: z + 1]))))
        # Jump of >= 30 bpm within ~5 s (abrupt onset, not a gradual climb).
        for i in range(len(t)):
            j = np.searchsorted(t, t[i] - 5.0)
            if j < i and np.isfinite(smooth[i]) and np.isfinite(smooth[j]):
                if smooth[i] - smooth[j] >= 30:
                    if not ev or ev[-1].kind != "Sudden rate jump" or t[i] - ev[-1].t_end > 30:
                        ev.append(Event("Sudden rate jump", t[j], t[i],
                                        rate_bpm=float(smooth[i]),
                                        note=f"{smooth[j]:.0f} -> {smooth[i]:.0f} bpm"))

    for m in markers_rel or []:
        ev.append(Event("Marker", m, m, note="pressed by you"))

    ev.sort(key=lambda e: (-e.severity, e.t_start))
    return ev


def summary(b: Beats, fs: float, duration_s: float) -> dict:
    lab = b.label
    n_total = len(lab)
    n_q = int((lab == "Q").sum())
    n_eval = max(n_total - n_q, 1)
    hr = 60.0 / b.rr_prev[(lab == "N") & np.isfinite(b.rr_prev)]
    return {
        "duration_h": duration_s / 3600,
        "beats": n_total,
        "unreadable_beats_pct": 100 * n_q / max(n_total, 1),
        "V": int((lab == "V").sum()),
        "S": int((lab == "S").sum()),
        "V_pct": 100 * (lab == "V").sum() / n_eval,
        "S_pct": 100 * (lab == "S").sum() / n_eval,
        "hr_min": float(np.percentile(hr, 1)) if hr.size else float("nan"),
        "hr_mean": float(np.mean(hr)) if hr.size else float("nan"),
        "hr_max": float(np.percentile(hr, 99)) if hr.size else float("nan"),
    }
