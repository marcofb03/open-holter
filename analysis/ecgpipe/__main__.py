"""Usage:  python -m ecgpipe <file.csv> [more.csv ...] -o out_dir"""

from __future__ import annotations

import argparse

import numpy as np

from .beats import analyse_beats
from .events import find_events, summary
from .io import Recording, load_many, split_on_gaps
from .report import write_report


def merge(recs: list[Recording]) -> Recording:
    """Join consecutive files; gaps are kept as gaps in time (handled per segment)."""
    t = np.concatenate([r.t for r in recs])
    ecg = np.concatenate([r.ecg for r in recs])
    order = np.argsort(t, kind="stable")
    return Recording(t=t[order], ecg=ecg[order], fs=recs[0].fs,
                     markers=sorted(m for r in recs for m in r.markers),
                     device_rr=[x for r in recs for x in r.device_rr],
                     source=[s for r in recs for s in r.source])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default="report")
    args = ap.parse_args()

    rec = merge(load_many(args.files))
    segments = split_on_gaps(rec)
    print(f"Loaded {len(rec.source)} file(s), {rec.duration_s / 3600:.2f} h at {rec.fs:.1f} Hz, "
          f"{len(segments)} continuous segment(s), {len(rec.markers)} marker(s)")

    # Each continuous stretch (between Bluetooth dropouts) is analysed separately.
    for k, seg in enumerate(segments):
        clean, b = analyse_beats(seg.ecg, seg.fs)
        t0 = float(seg.t[0])
        ev = find_events(b, seg.fs, [m - t0 for m in seg.markers])
        out = args.out if len(segments) == 1 else f"{args.out}/segment_{k:02d}"
        path = write_report(out, clean, seg.fs, b, ev, summary(b, seg.fs, seg.duration_s), t0,
                            title=f"ECG analysis – segment {k + 1}")
        print(f"  segment {k + 1}: {len(b.idx)} beats, {len(ev)} events -> {path}")


if __name__ == "__main__":
    main()
