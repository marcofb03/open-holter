# analysis

Holter-style screening for Polar H10 ECG recordings (ECGLogger CSV, 130 Hz).
It flags candidate events for a cardiologist to read. It does not diagnose.

## Setup

```bash
cd analysis
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python -m ecgpipe ../data/recordings/day1.csv [more.csv ...] -o ../reports/day1
open ../reports/day1/report.html
```

Several 1-hour files from the same day can be passed together.

## What it does

1. Loads ECGLogger CSV (timestamp ns, ECG mV, pulse, RR ms, marker); splits at Bluetooth gaps.
2. Band-pass 0.5–40 Hz, R-peak detection (NeuroKit2), per-window noise check.
3. Per beat: RR before/after, prematurity, QRS shape correlation and width vs the normal template.
4. Rule labels: N normal, S early + normal shape, V different/wider shape, Q unreadable.
5. Events: V/S couplets and runs (>=3), pauses > 2 s, rate > 150 bpm for >= 10 s,
   sudden rate jumps, your markers.
6. HTML report: summary, overview (rate + extra beats per 5 min), ECG strip per event.

## Check against annotated data

```bash
.venv/bin/python tools/eval_mitdb.py ../data/mitdb 100 119 200
```

MIT-BIH records (downloaded automatically) are resampled to 130 Hz and passed
through the full pipeline. First results (3 records): beat detection
Se >= 0.999, PPV >= 0.998; V beats Se 0.93–1.00, PPV 0.99–1.00 on records with
PVCs; S beats are weaker (Se 0.60–0.91, PPV 0.35–1.00), as expected from a
single lead. Three records is a small test; a full evaluation is planned.
