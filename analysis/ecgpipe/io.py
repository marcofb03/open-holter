"""Loading ECG recordings.

ECGLogger (iOS, Polar H10) CSV format, per ecglogger.com:
    timestamp_ns, ecg_mV, pulse_bpm, rr_ms, marker
Pulse / RR / marker are only written when there is new data, so most rows
carry only timestamp + ECG. Whether a header line exists is not documented,
so it is detected.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NOMINAL_FS = 130.0  # Polar H10 ECG sampling rate (Hz)


@dataclass
class Recording:
    t: np.ndarray            # seconds since epoch, one per sample
    ecg: np.ndarray          # millivolts
    fs: float                # sampling rate (Hz)
    markers: list[float] = field(default_factory=list)   # epoch seconds
    device_rr: list[tuple[float, float]] = field(default_factory=list)  # (t, rr_ms)
    source: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(len(self.ecg) / self.fs)


def _to_float(x: str) -> float | None:
    x = x.strip()
    if not x:
        return None
    try:
        return float(x.replace(",", "."))
    except ValueError:
        return None


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def load_ecglogger_csv(path: str | Path) -> Recording:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        reader = csv.reader(f, delimiter=_sniff_delimiter(sample))

        ts, ecg, markers, rr = [], [], [], []
        for row in reader:
            if not row:
                continue
            t_ns = _to_float(row[0])
            if t_ns is None:  # header or garbage line
                continue
            t = t_ns / 1e9
            v = _to_float(row[1]) if len(row) > 1 else None
            if v is not None:
                ts.append(t)
                ecg.append(v)
            if len(row) > 3:
                r = _to_float(row[3])
                if r is not None:
                    rr.append((t, r))
            if len(row) > 4 and row[4].strip():
                markers.append(t)

    if not ecg:
        raise ValueError(f"No ECG samples found in {path}")

    t_arr = np.asarray(ts, dtype=float)
    dt = np.diff(t_arr)
    good = dt[(dt > 0) & (dt < 0.1)]
    fs = float(1.0 / np.median(good)) if good.size else NOMINAL_FS

    return Recording(
        t=t_arr,
        ecg=np.asarray(ecg, dtype=float),
        fs=fs,
        markers=_dedupe(markers),
        device_rr=rr,
        source=[str(path)],
    )


def _dedupe(times: list[float], min_gap: float = 2.0) -> list[float]:
    """Collapse marker rows that belong to the same button press."""
    out: list[float] = []
    for t in sorted(times):
        if not out or t - out[-1] > min_gap:
            out.append(t)
    return out


def load_many(paths: list[str | Path]) -> list[Recording]:
    """Load several files (e.g. the 1-hour splits) sorted by start time."""
    recs = [load_ecglogger_csv(p) for p in paths]
    return sorted(recs, key=lambda r: r.t[0])


def split_on_gaps(rec: Recording, max_gap_s: float = 1.0) -> list[Recording]:
    """Split a recording where samples are missing (Bluetooth dropouts)."""
    gaps = np.where(np.diff(rec.t) > max_gap_s)[0]
    if gaps.size == 0:
        return [rec]
    bounds = np.concatenate(([0], gaps + 1, [len(rec.t)]))
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < rec.fs * 10:  # ignore fragments shorter than 10 s
            continue
        t0, t1 = rec.t[a], rec.t[b - 1]
        out.append(Recording(
            t=rec.t[a:b], ecg=rec.ecg[a:b], fs=rec.fs,
            markers=[m for m in rec.markers if t0 <= m <= t1],
            device_rr=[x for x in rec.device_rr if t0 <= x[0] <= t1],
            source=rec.source,
        ))
    return out


def write_ecglogger_csv(path: str | Path, t: np.ndarray, ecg_mv: np.ndarray,
                        markers: list[float] | None = None) -> None:
    """Write data in ECGLogger layout (used for tests with public datasets)."""
    marker_idx = set()
    for m in markers or []:
        marker_idx.add(int(np.argmin(np.abs(t - m))))
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        for i, (ti, v) in enumerate(zip(t, ecg_mv)):
            w.writerow([int(round(ti * 1e9)), f"{v:.4f}", "", "", "1" if i in marker_idx else ""])
