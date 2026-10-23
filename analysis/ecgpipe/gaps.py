"""Recording gaps and their reason, from the app's events_*.log.

A gap is a stretch without ECG samples (Bluetooth dropout, pause for shower,
app update, strap off, ...). Gaps are never heart pauses: the recording is
split at every gap (io.split_on_gaps) and each piece is analysed on its own,
so no RR interval is ever measured across a gap. This module only says *why*
the ECG is missing, so the reviewer can tell a planned gap from a lost signal.

events_*.log (open-holter Android app), tab-separated with one header line:
    time_local  timestamp_ns  event  detail
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Event that explains a gap -> reason shown in the report. The first match in
# this order wins, so a planned stop (update, pause) beats its side effects.
REASONS = [
    ("update_start", "app update"),
    ("update_installed", "app update"),
    ("pause", "paused by the user (Pause for shower)"),
    ("stop", "recording stopped by the user"),
    ("bluetooth_off", "Bluetooth turned off"),
    ("no_contact", "no skin contact (strap off), ECG stopped to save the H10 battery"),
    ("disconnected", "H10 disconnected (out of range or Bluetooth dropout)"),
    ("service_restart", "recorder restarted (reboot or app restart)"),
    ("no_data", "no data from the H10, streams restarted"),
]

# A cause is logged at most this long before the last sample of the gap (e.g. the
# "pause" event comes ~1 s before the streams stop).
LEAD_S = 15.0


@dataclass
class AppEvent:
    t: float        # epoch seconds
    event: str
    detail: str


@dataclass
class Gap:
    start: float    # epoch seconds of the last sample before the gap
    end: float      # epoch seconds of the first sample after the gap
    reason: str
    events: list[AppEvent]

    @property
    def duration_s(self) -> float:
        return self.end - self.start


def load_events_log(path: str | Path) -> list[AppEvent]:
    out = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            try:
                t = int(row[1]) / 1e9
            except ValueError:  # header
                continue
            out.append(AppEvent(t, row[2].strip(), row[3].strip() if len(row) > 3 else ""))
    return sorted(out, key=lambda e: e.t)


def recording_gaps(t: np.ndarray, events: list[AppEvent], max_gap_s: float = 1.0) -> list[Gap]:
    """Gaps > max_gap_s in the sample times t (epoch s), each with its most likely reason."""
    gaps = []
    for i in np.where(np.diff(t) > max_gap_s)[0]:
        a, b = float(t[i]), float(t[i + 1])
        near = [e for e in events if a - LEAD_S <= e.t <= b]
        names = {e.event for e in near}
        reason = next((r for ev, r in REASONS if ev in names), "unknown (no app event logged)")
        gaps.append(Gap(a, b, reason, near))
    return gaps


def write_gaps_csv(path: str | Path, gaps: list[Gap]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_ns", "end_ns", "duration_s", "reason", "app_events"])
        for g in gaps:
            w.writerow([int(round(g.start * 1e9)), int(round(g.end * 1e9)), f"{g.duration_s:.1f}", g.reason,
                        " | ".join(f"{e.event} {e.detail}".strip() for e in g.events)])
