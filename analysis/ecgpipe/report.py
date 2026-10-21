"""HTML report with ECG strips for each flagged event."""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .beats import Beats  # noqa: E402
from .events import Event  # noqa: E402

COLORS = {"N": "#2b6cb0", "S": "#dd6b20", "V": "#c53030", "Q": "#a0aec0"}
MAX_STRIPS = 40
FILTER_NOTE = "0.5–40 Hz band-pass"


def _clock(t0_epoch: float | None, t_rel: float) -> str:
    if t0_epoch is None:
        return f"{t_rel / 3600:02.0f}:{(t_rel % 3600) / 60:02.0f}:{t_rel % 60:04.1f} (rel)"
    return dt.datetime.fromtimestamp(t0_epoch + t_rel).strftime("%Y-%m-%d %H:%M:%S")


def _strip(path: Path, raw: np.ndarray, clean: np.ndarray, fs: float, b: Beats, a_s: float,
           z_s: float, title: str, mark: float | None = None) -> None:
    """ECG strip: the raw signal as recorded, with the detection filter as a light overlay.

    The raw trace is drawn unchanged (only the ECGLogger mV values). The filtered
    trace is shifted to the raw trace's median in this strip so the two overlap;
    that shift is for display only.
    """
    a, z = max(int(a_s * fs), 0), min(int(z_s * fs), len(raw))
    if z <= a:
        return
    t = np.arange(a, z) / fs
    fig, ax = plt.subplots(figsize=(14, 2.6), dpi=110)
    offset = float(np.median(raw[a:z]))
    ax.plot(t, clean[a:z] + offset, color="#90cdf4", lw=1.6, alpha=0.8,
            label=f"filtered {FILTER_NOTE} (detection)")
    ax.plot(t, raw[a:z], color="#1a202c", lw=0.8, label="raw, as recorded")
    sel = (b.idx >= a) & (b.idx < z)
    for p, l in zip(b.idx[sel], b.label[sel]):
        ax.annotate(l, (p / fs, raw[p]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color=COLORS[l], weight="bold")
    if mark is not None:
        ax.axvline(mark, color="#805ad5", ls="--", lw=1)
    # ECG-paper style grid: 0.2 s / 0.5 mV major lines.
    ax.set_xticks(np.arange(np.floor(t[0] / 0.2) * 0.2, t[-1], 0.2), minor=True)
    ax.grid(True, which="both", color="#fed7d7", lw=0.5)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("seconds from start")
    ax.set_ylabel("mV")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _overview(path: Path, b: Beats, fs: float) -> None:
    t = b.idx / fs / 3600
    hr = 60.0 / b.rr_prev
    good = (b.label == "N") & np.isfinite(hr)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4.5), dpi=110, sharex=True)
    ax1.plot(t[good], hr[good], ".", ms=1, color="#2b6cb0")
    ax1.set_ylabel("heart rate (bpm)")
    ax1.set_ylim(30, 200)
    ax1.grid(alpha=0.3)
    bins = np.arange(0, (t.max() if t.size else 1) + 5 / 60, 5 / 60)
    for lab in ("V", "S"):
        counts, _ = np.histogram(t[b.label == lab], bins=bins)
        ax2.step(bins[:-1], counts, where="post", color=COLORS[lab], label=f"{lab} per 5 min")
    ax2.set_ylabel("extra beats")
    ax2.set_xlabel("hours from start")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(out_dir: str | Path, raw: np.ndarray, clean: np.ndarray, fs: float, b: Beats,
                 events: list[Event], summ: dict, t0_epoch: float | None,
                 title: str = "ECG analysis") -> Path:
    out = Path(out_dir)
    (out / "strips").mkdir(parents=True, exist_ok=True)
    _overview(out / "overview.png", b, fs)

    rows = []
    for k, e in enumerate(events[:MAX_STRIPS]):
        if e.kind == "Marker":
            a_s, z_s, mark = e.t_start - 20, e.t_start + 5, e.t_start
        else:
            mid = (e.t_start + e.t_end) / 2
            half = max(5.0, (e.t_end - e.t_start) / 2 + 2)
            a_s, z_s, mark = mid - half, mid + half, None
        png = f"strips/event_{k:03d}.png"
        _strip(out / png, raw, clean, fs, b, a_s, z_s,
               f"{e.kind} at {_clock(t0_epoch, e.t_start)} – confidence {e.confidence_level}", mark)
        rows.append(
            f"<tr><td>{k + 1}</td><td>{html.escape(e.kind)}</td>"
            f"<td>{_clock(t0_epoch, e.t_start)}</td><td>{e.n_beats or ''}</td>"
            f"<td>{'' if np.isnan(e.rate_bpm) else f'{e.rate_bpm:.0f}'}</td>"
            f"<td>{html.escape(e.note)}</td>"
            f"<td>{e.confidence_level}{': ' + html.escape(e.confidence_note) if e.confidence_note else ''}</td>"
            f"<td><a href='#ev{k}'>strip</a></td></tr>")
    strips = "\n".join(
        f"<h3 id='ev{k}'>{k + 1}. {html.escape(e.kind)} – {_clock(t0_epoch, e.t_start)}"
        f" (confidence {e.confidence_level})</h3>"
        f"<img src='strips/event_{k:03d}.png' width='100%'>"
        for k, e in enumerate(events[:MAX_STRIPS]))

    s = summ
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;margin:16px;max-width:1200px;color:#1a202c}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border:1px solid #e2e8f0;padding:4px 8px;text-align:left}}
th{{background:#f7fafc}} .warn{{background:#fffaf0;border:1px solid #f6ad55;padding:8px 12px;border-radius:6px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="warn">Automatic screening, not a diagnosis. Labels: <b style="color:{COLORS['N']}">N</b> normal,
<b style="color:{COLORS['S']}">S</b> early beat with normal shape (atrial-like),
<b style="color:{COLORS['V']}">V</b> beat with different shape (ventricular-like),
<b style="color:{COLORS['Q']}">Q</b> unreadable. A cardiologist must review the strips.</p>
<p>Strips show the <b>raw signal as recorded</b> (black, ECGLogger mV values, unchanged).
Beats are detected on a copy filtered with a {FILTER_NOTE} (zero-phase Butterworth),
drawn as a light blue overlay. The recording files are never modified.
Confidence (high / medium / low) is lowered by noise, unreadable beats, the artefact check,
lost skin contact, strong movement, or H10 RR intervals that disagree with ours.</p>
<h2>Summary</h2>
<table>
<tr><th>Recording</th><td>{s['duration_h']:.2f} h, {s['beats']} beats, {s['unreadable_beats_pct']:.1f}% unreadable</td></tr>
<tr><th>Heart rate</th><td>min {s['hr_min']:.0f} · mean {s['hr_mean']:.0f} · max {s['hr_max']:.0f} bpm</td></tr>
<tr><th>V-like beats</th><td>{s['V']} ({s['V_pct']:.2f}%)</td></tr>
<tr><th>S-like beats</th><td>{s['S']} ({s['S_pct']:.2f}%)</td></tr>
<tr><th>Events</th><td>{len(events)} (showing up to {MAX_STRIPS})</td></tr>
</table>
<h2>Overview</h2><img src="overview.png" width="100%">
<h2>Events (most important first)</h2>
<table><tr><th>#</th><th>Type</th><th>Time</th><th>Beats</th><th>Rate</th><th>Note</th><th>Confidence</th><th></th></tr>
{''.join(rows) or '<tr><td colspan=8>No events found.</td></tr>'}
</table>
<h2>ECG strips</h2>
{strips}
</body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
