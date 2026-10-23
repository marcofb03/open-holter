# analysis

Holter-style screening for Polar H10 ECG recordings (ECGLogger CSV, 130 Hz).
It flags candidate events for a cardiologist to read. It does not diagnose and
is not a medical device.

## Setup

```bash
cd analysis
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # runtime deps + pytest
```

## Run

```bash
.venv/bin/python -m ecgpipe ../data/recordings/day1.csv [more.csv ...] -o ../reports/day1
open ../reports/day1/report.html
```

Several 1-hour files from the same day can be passed together. Optional side
files from the open-holter Android app add evidence to each event:

```bash
.venv/bin/python -m ecgpipe day1.csv -o ../reports/day1 \
    --acc day1_acc.csv \      # timestamp_ns,x_mG,y_mG,z_mG
    --hr  day1_hr.csv         # timestamp_ns,hr_bpm,rr_ms,contact,contact_supported
```

The app's `events_*.log` next to the ECG files is read too (or pass
`--events`). Every gap in the ECG (dropout, pause, app update, strap off) is
listed with its reason in `gaps.csv` and on the console. Gaps are never
analysed as heart pauses: each piece between two gaps is analysed separately.

Input files are only read, never changed.

## What it does

1. Loads ECGLogger CSV (timestamp ns, ECG mV, pulse, RR ms, marker); splits at Bluetooth gaps.
2. Band-pass 0.5–40 Hz (zero-phase Butterworth) for detection. Report strips show
   the **raw** signal as recorded, with the filtered trace as a light overlay.
3. QRS detection on a slope envelope with an adaptive threshold and search-back
   (plain numpy/scipy, so it can be ported to the phone).
4. Per beat: RR before/after, prematurity against the preceding rhythm, and the QRS
   shape against the normal template of its block of beats (aligned first): correlation,
   QRS + T correlation, width and amplitude ratio. A noise index between the beats.
5. Rule labels: N normal, S sudden early beat with normal shape (not in AF-like chaotic
   rhythm), V clearly different shape (or moderately different plus a second sign),
   Q unreadable. Thresholds are listed in `RULES` in `ecgpipe/beats.py`.
6. Artefact check ("sinus marches through"): if normal QRS complexes keep arriving at
   the expected sinus times through a stretch of V/S beats, the extra beats are
   artefact (Q). Such stretches are listed as "Possible artefact" so nothing is hidden.
7. Events: V/S couplets and runs (>=3), pauses > 2 s, rate > 150 bpm for >= 10 s,
   sudden rate jumps, your markers.
8. Confidence per event (high / medium / low), lowered by noise, unreadable beats,
   the artefact check, lost skin contact or strong movement (side files), or H10 RR
   intervals that disagree with ours.
9. HTML report: summary, overview (rate + extra beats per 5 min), ECG strip per event.

## Validation

See [VALIDATION.md](VALIDATION.md): MIT-BIH Arrhythmia (DS1 for tuning, DS2 held
out), MIT-BIH Supraventricular and INCART, all resampled to 130 Hz and passed
through the CSV loader, plus synthetic noise stress tests.

```bash
.venv/bin/python tools/eval_physionet.py mitdb --subset DS2 svdb incartdb
.venv/bin/python -m pytest -q tests          # synthetic tests, no data needed
```

## Next: AI classifier

The rules are transparent but weak on S beats and on unusual QRS shapes. The plan
is an open beat classifier that can replace or back up `label_beats` (not built yet):

- **Data.** A subset of [Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/)
  (11,000 patients, single-lead chest patch, 250 Hz, beat labels N/S/V/Q), resampled
  to 130 Hz and band-limited like the H10 so training matches what the phone records.
  A few thousand patients, sampled to keep enough S and V beats, is enough to start.
- **Model.** A small 1D CNN on a beat-centred window (0.2 s before to 0.45 s after
  the R peak, ~85 samples at 130 Hz, the same window as the rule templates), with
  RR features (previous / next RR over the local reference, local irregularity)
  concatenated before the dense layer. A few tens of thousands of parameters, so it
  runs comfortably on a phone. Output N/S/V/Q probabilities; low-confidence beats
  stay Q.
- **Evaluation.** Strictly inter-patient: split Icentia11k by patient, and report on
  MIT-BIH DS2, SVDB and INCART with `tools/eval_physionet.py` (same AAMI scoring and
  run counting as the rules), so rules and model are compared on the same footing.
  V-run sensitivity and false V runs per hour stay the main numbers.
- **Deployment.** Export to TensorFlow Lite (or ONNX Runtime Mobile), int8-quantised,
  and run it on the phone next to the ported detector; the rules remain as a fallback
  and as a sanity check on the model.
