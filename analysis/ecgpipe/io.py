"""Loading ECG recordings and optional side files.

ECGLogger (iOS, Polar H10) CSV format, per ecglogger.com:
    timestamp_ns, ecg_mV, pulse_bpm, rr_ms, marker
Pulse / RR / marker are only written when there is new data, so most rows
carry only timestamp + ECG. Whether a header line exists is not documented,
so it is detected.

Optional side files written by the open-holter Android app:
    accelerometer:  timestamp_ns, x_mG, y_mG, z_mG               (25–50 Hz)
    HR / contact:   timestamp_ns, hr_bpm, rr_ms, contact, contact_supported
The loaders only read files; nothing is ever written back to a recording.
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


@dataclass
class Accel:
    t: np.ndarray            # epoch seconds
    xyz: np.ndarray          # (n, 3) milli-g


@dataclass
class HeartRate:
    t: np.ndarray            # epoch seconds, one per notification
    hr: np.ndarray           # bpm (nan if missing)
    contact: np.ndarray      # bool: skin contact reported
    contact_supported: np.ndarray  # bool: the sensor reports contact at all
    rr: list[tuple[float, float]] = field(default_factory=list)  # (t, rr_ms), H10's own RR


def _rows(path: str | Path):
    """Numeric CSV rows (header lines and empty lines skipped)."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        for row in csv.reader(f, delimiter=_sniff_delimiter(sample)):
            if row and _to_float(row[0]) is not None:
                yield row


def _to_bool(x: str) -> bool | None:
    x = x.strip().lower()
    if x in ("1", "true", "yes", "y"):
        return True
    if x in ("0", "false", "no", "n"):
        return False
    return None


def load_accel_csv(path: str | Path) -> Accel:
    t, xyz = [], []
    for row in _rows(path):
        v = [_to_float(c) for c in row[1:4]] if len(row) >= 4 else [None]
        if None in v:
            continue
        t.append(_to_float(row[0]) / 1e9)
        xyz.append(v)
    if not t:
        raise ValueError(f"No accelerometer samples found in {path}")
    order = np.argsort(t, kind="stable")
    return Accel(np.asarray(t)[order], np.asarray(xyz, dtype=float)[order])


def load_hr_csv(path: str | Path) -> HeartRate:
    """HR / contact file. rr_ms may hold several values ('812 798' or '812|798')."""
    t, hr, contact, supported, rr = [], [], [], [], []
    for row in _rows(path):
        ts = _to_float(row[0]) / 1e9
        h = _to_float(row[1]) if len(row) > 1 else None
        c = _to_bool(row[3]) if len(row) > 3 else None
        cs = _to_bool(row[4]) if len(row) > 4 else None
        t.append(ts)
        hr.append(np.nan if h is None else h)
        contact.append(True if c is None else c)
        supported.append(bool(cs) if cs is not None else c is not None)
        if len(row) > 2:
            for part in row[2].replace("|", " ").replace(";", " ").split():
                v = _to_float(part)
                if v is not None and v > 0:
                    rr.append((ts, v))
    if not t:
        raise ValueError(f"No heart-rate rows found in {path}")
    order = np.argsort(t, kind="stable")
    return HeartRate(np.asarray(t)[order], np.asarray(hr, dtype=float)[order],
                     np.asarray(contact, dtype=bool)[order],
                     np.asarray(supported, dtype=bool)[order], sorted(rr))
