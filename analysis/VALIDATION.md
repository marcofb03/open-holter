# Validation of the rule-based pipeline

> **Not a medical device.** These numbers describe how well a home-made,
> rule-based screening tool agrees with cardiologists' annotations on public
> research databases. They do not make it a diagnostic tool. Every flagged event
> still has to be read by a cardiologist on the ECG strip.

## In short

- **Beats are found reliably.** At 130 Hz on a single lead, 99.8–100% of annotated beats
  are detected, and over 99% of detections are real beats, on all three databases.
- **Ventricular (V) beats:** on the held-out data about 79–91% are labelled V
  (sensitivity), and 76–80% of the beats we call V really are V (PPV). Before this
  work, PPV was 43–66%.
- **V runs (≥ 3 ventricular beats in a row), the main question for this project:**
  we find 15/20 (DS2), 42/88 (SVDB) and 99/148 (INCART) of the reference runs. False V
  runs dropped sharply on SVDB (20 → 1.3 per hour) and INCART (25 → 11 per hour), but
  rose on MIT-BIH DS2 (6 → 8 per hour, almost all in one atrial-flutter record).
  **Missed runs and false runs both still happen, so every V-run strip must be read
  by a cardiologist.** A missed run is the more serious error here.
- **Supraventricular (S) beats are the weak point.** On the held-out sets S
  sensitivity is 10–66% and PPV 18–65%. With a single lead and no P-wave analysis,
  an atrial beat that is not clearly early cannot be told apart from a normal one.
- **Noise:** strong muscle noise is mostly marked unreadable (Q) instead of producing
  false V runs. Simulated electrode-motion artefacts still produce many false V
  runs. The artefact check and the new per-event confidence help a reviewer spot
  them, but they do not remove them.

## What was measured

| | |
|---|---|
| Databases | [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) (`mitdb`, 44 records: 48 minus the paced 102, 104, 107, 217 as in AAMI EC57; lead MLII, 360 Hz), [MIT-BIH Supraventricular](https://physionet.org/content/svdb/1.0.0/) (`svdb`, 78 records, first lead `ECG1`, 128 Hz), [St Petersburg INCART](https://physionet.org/content/incartdb/1.0.0/) (`incartdb`, 75 records, lead II, 257 Hz) |
| Signal path | Each record is resampled to 130 Hz (`scipy.signal.resample_poly`), written as an ECGLogger CSV (timestamp ns, mV with 4 decimals), read back with the same loader as real recordings, then analysed by `ecgpipe` exactly as a recording would be |
| Beat classes | AAMI EC57: N = N L R e j n B, S = A a J S, V = V E, F = F, Q = / f Q |
| Matching | A detected beat matches a reference beat within 150 ms (greedy nearest) |
| Scoring | EC57 conventions: a reference F beat called V is not a false V; reference Q beats are left out of class statistics; ventricular flutter episodes (`[ ... ]`) and the first/last second are skipped. **A reference beat that we call unreadable (Q) counts as missed.** |
| Gross statistics | Sums over all records (TP, FP, FN), then Se = TP / (TP + FN), PPV = TP / (TP + FP) |
| Runs | ≥ 3 consecutive reference V (S) beats form a reference run. A run is *found* if a detected run of the same class overlaps it in time. *False V runs* are detected V runs that overlap no reference V run. |
| Unreadable % | Share of detected beats labelled Q |

## How the rules were tuned (and what was held out)

- All thresholds in `RULES` (`ecgpipe/beats.py`) and the detector, template and noise
  settings were chosen on **MIT-BIH DS1 only**, using the inter-patient split of de
  Chazal et al. (2004): 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
  201, 203, 205, 207, 208, 209, 215, 220, 223, 230. Synthetic noise for robustness
  tuning was also added to DS1 only.
- **MIT-BIH DS2, SVDB and INCART are held out.**
- **Disclosure:** the held-out sets were run once part-way through development. That
  run showed three problems, and the fixes are design changes rather than retuning,
  but this knowledge still makes the held-out numbers slightly optimistic:
  1. The first AF guard (which stops S labels in chaotic rhythms) suppressed frequent
     atrial ectopy on SVDB. It was redesigned to require RR intervals that do not
     cluster (sinus + ectopy forms tight RR groups, AF does not), and its parameters
     were chosen against the **rhythm annotations of DS1** (AF beats flagged 91%,
     non-AF beats flagged 6.5% instead of 15%).
  2. A rule inherited from the first version marked any beat < 0.24 s from its
     neighbour as noise. On SVDB, wide PVCs were counted twice, and the rule then hid
     them as Q. That is a safety problem, because it would also hide very fast VT, so
     the rule was removed and replaced by a double-detection check.
  3. The noise index cut-off was raised from 0.4 to 0.5, because fast SVT with sharp
     P waves reached 0.42 and was hidden as Q. Hiding real arrhythmias is worse than
     showing extra noise.
- After that, the rules were frozen and all held-out numbers below come from the final code.

## Results: before and after

*Before* is the first version (commit `8b7eafd`: NeuroKit2 detector, unaligned
template correlation, centred RR median). *After* is this version. Both are run with
the same evaluation tool.

| set | version | beat Se | beat PPV | unreadable % | V Se | V PPV | S Se | S PPV | V runs found | false V runs (per h) | S runs found |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MIT-BIH DS1 (tuning) | before | 0.986 | 0.996 | 0.000 | 0.846 | 0.691 | 0.380 | 0.278 | 12/43 | 109 (9.9) | 3/26 |
| MIT-BIH DS1 (tuning) | after | 0.998 | 0.998 | 0.078 | 0.912 | 0.837 | 0.676 | 0.809 | 35/43 | 61 (5.5) | 17/26 |
| MIT-BIH DS2 (held out) | before | 0.998 | 0.989 | 0.000 | 0.897 | 0.663 | 0.108 | 0.131 | 10/20 | 67 (6.1) | 2/184 |
| MIT-BIH DS2 (held out) | after | 1.000 | 0.998 | 0.070 | 0.907 | 0.762 | 0.096 | 0.178 | 15/20 | 91 (8.2) | 4/184 |
| SVDB (held out) | before | 0.995 | 0.997 | 0.298 | 0.685 | 0.429 | 0.517 | 0.646 | 35/88 | 792 (20.3) | 45/169 |
| SVDB (held out) | after | 0.999 | 0.997 | 0.726 | 0.785 | 0.795 | 0.455 | 0.653 | 42/88 | 51 (1.3) | 44/169 |
| INCART (held out) | before | 0.992 | 0.991 | 0.553 | 0.794 | 0.579 | 0.734 | 0.323 | 87/148 | 923 (24.6) | 2/16 |
| INCART (held out) | after | 0.998 | 0.994 | 1.1 | 0.807 | 0.758 | 0.661 | 0.290 | 99/148 | 421 (11.2) | 5/16 |

Notes:

- MIT-BIH DS2 S beats are dominated by record 232, where 1,382 annotated S beats form
  the underlying rhythm (sinus bradycardia with long pauses) rather than early beats,
  so no RR rule can find them. Without 232, DS2 S sensitivity is 160/455 = 0.35.
  Record 231 (2:1 AV block switching to 1:1 conduction) yields 748 false S, because
  a sudden doubling of the rate looks exactly like an atrial run; a frozen reference
  is held for at most 30 s.
- MIT-BIH DS2 false V runs come almost entirely from record 222 (67 of 91): during
  atrial flutter the flutter waves distort normal QRS complexes. This is a known
  weakness and was **not** tuned away, because 222 is held out.
- SVDB record 865 (long fast SVT) is still 15% unreadable. Most of the 365 S beats lost
  as Q on SVDB come from it.
- INCART: 402 V beats fall into stretches marked unreadable (370 before).

### Synthetic noise stress test (MIT-BIH DS2, held out)

The PhysioNet Noise Stress Test database could not be downloaded from this build
environment. Instead, noise is generated in the same protocol (2-minute bursts,
starting at minute 5, every other 2 minutes, about 4.8 h of noisy signal in 11 h)
with the NST definition of SNR (signal = (median QRS peak-to-peak)² / 8):

- `emg`: white noise band-limited to 20–60 Hz (muscle);
- `motion`: 0.3–4 Hz wander plus random sharp QRS-like transients about once per
  second (electrode motion, the worst case for false beats).

| set | version | beat Se | beat PPV | unreadable % | V Se | V PPV | S Se | S PPV | V runs found | false V runs (per h) | S runs found |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DS2 + motion 12 dB | before | 0.998 | 0.980 | 0.000 | 0.897 | 0.551 | 0.103 | 0.108 | 11/20 | 95 (8.6) | 2/184 |
| DS2 + motion 12 dB | after | 0.999 | 0.997 | 0.082 | 0.905 | 0.659 | 0.089 | 0.194 | 17/20 | 113 (10.2) | 3/184 |
| DS2 + motion 6 dB | before | 0.997 | 0.968 | 0.000 | 0.892 | 0.341 | 0.088 | 0.096 | 11/20 | 428 (38.8) | 2/184 |
| DS2 + motion 6 dB | after | 0.999 | 0.975 | 0.299 | 0.867 | 0.361 | 0.073 | 0.157 | 15/20 | 388 (35.2) | 2/184 |
| DS2 + emg 12 dB | before | 0.994 | 0.968 | 0.078 | 0.872 | 0.515 | 0.102 | 0.090 | 9/20 | 91 (8.2) | 2/184 |
| DS2 + emg 12 dB | after | 0.999 | 0.986 | 0.318 | 0.903 | 0.651 | 0.085 | 0.169 | 14/20 | 112 (10.2) | 3/184 |
| DS2 + emg 6 dB | before | 0.889 | 0.927 | 5.0 | 0.753 | 0.324 | 0.087 | 0.080 | 7/20 | 358 (32.5) | 2/184 |
| DS2 + emg 6 dB | after | 0.994 | 0.883 | 25.2 | 0.681 | 0.390 | 0.059 | 0.169 | 7/20 | 153 (13.9) | 2/184 |

Muscle noise at 6 dB is now mostly marked unreadable (25% Q), which costs V
sensitivity inside the noisy bursts (552 V beats lost as Q), but false V runs fall
from 358 to 153. Motion artefacts at 6 dB still produce about 35 false V runs per
hour; at 12 dB the new version finds more real runs (17/20 vs 11/20) but also flags a
few more false ones. These are synthetic noise models, not real H10 recordings.

## What changed in the pipeline

- **QRS detection:** a portable detector (slope envelope, adaptive threshold from the
  median of the largest peaks, search-back in long gaps, 0.25 s refractory, T-wave and
  double-detection guard) replaces NeuroKit2. On DS1 it lifted beat sensitivity from
  0.986 to 0.998, mostly by finding wide ventricular beats that NeuroKit2 missed.
- **Shape features:** each beat is aligned to the block template before it is compared
  (±50 ms), shapes are compared below 20 Hz (a sharp R spike is only 2–3 samples at
  130 Hz), and a straight baseline tilt is removed. New features: QRS + ST-T
  correlation and amplitude ratio. A second normal template is accepted for on-time
  beats with another shape, which covers posture change and intermittent bundle-branch block.
- **V rule:** correlation < 0.7, or < 0.96 with a second sign (wider × 1.3, amplitude
  × 1.6 or × 0.4, QRS + T correlation < 0.5, or early and followed by a pause).
- **S rule:** RR < 0.85 × the preceding rhythm **and** < 0.9 × the previous RR (sudden,
  not gradual), normal shape, not in an AF-like rhythm. Inside a run, the reference
  from before the run is kept for up to 30 s. On DS1, S PPV went from 0.28 to 0.81.
- **Noise:** a noise index (QRS-band activity between beats relative to the beats) marks
  2-s windows unreadable above 0.5, in addition to flat-line, > 5 mV and > 40 Hz checks.
- **Artefact check ("sinus marches through"):** for stretches with ≥ 3 V/S beats, the
  preceding sinus rhythm is extended through the stretch. If a normal QRS (correlation
  ≥ 0.85 with the preceding beats) is found at every expected time, the extra beats
  are artefact (Q) and are listed as "Possible artefact". Real V runs replace the
  sinus QRS, so they fail this test (see tests). It rarely fires on the databases;
  it is aimed at real-world motion spikes.
- **Per-event confidence:** noise, unreadable beats, artefact check, and, when the
  app's side files are given, lost skin contact, strong movement and disagreement
  with the H10's own RR intervals. This does not change labels; it tells the
  reviewer which strips to look at most critically.

## Strengths and limits (plain language)

**Strengths**

- Beat detection is very reliable at the Polar H10's 130 Hz.
- Typical PVCs are recognised well, and most V runs on the MIT-BIH records are found.
- The whole path is tested end to end, from CSV file to report, and the code is
  plain numpy/scipy that can be ported to the phone.
- Uncertainty is shown, not hidden: unreadable stretches, possible artefacts and a
  confidence level per event.

**Limits**

- **Single lead, 130 Hz.** Fine QRS detail and P waves are hard to see, so atrial
  (S) beats that are not clearly early are missed, and ventricular beats that look
  like normal ones in this lead are missed too.
- **Rule-based.** Fixed thresholds cannot cover every patient. Unusual rhythms
  produce systematic errors: atrial flutter gives false V runs, AV block gives false
  S runs, and a long SVT can be called unreadable.
- **About a third of V runs are missed on SVDB and INCART** (records with
  ventricular beats that look similar to the normal QRS in this lead). A missed V
  run is the most important failure for this project.
- **Motion artefacts** still create false V runs. Real H10 recordings during daily
  life will contain more of them than these databases do.
- The databases are short (30 min) clinical Holter recordings with other electrodes
  and leads. **No real Polar H10 recording with cardiologist annotation was available.**
  Performance on the owner's own recordings may differ.
- **This is not a medical device.** It does not diagnose and must not be used to
  rule out an arrhythmia.

## Reproduce

From `analysis/` (Python 3.12+; numbers above were produced with Python 3.13,
numpy 2.5.4, scipy 1.18.1, wfdb 4.3.1):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

# Download the databases into ../data/<db>/ (git-ignored; physionet.org, or its
# AWS open-data mirror if physionet.org is unreachable)
.venv/bin/python tools/fetch_physionet.py mitdb svdb incartdb

# After (this version)
.venv/bin/python tools/eval_physionet.py mitdb --subset DS1
.venv/bin/python tools/eval_physionet.py mitdb --subset DS2
.venv/bin/python tools/eval_physionet.py svdb
.venv/bin/python tools/eval_physionet.py incartdb
for n in motion emg; do for s in 12 6; do
  .venv/bin/python tools/eval_physionet.py mitdb --subset DS2 --noise $n --snr $s --summary
done; done

# Before (first version), same tool, code taken from commit 8b7eafd
git worktree add /tmp/open-holter-base 8b7eafd
.venv/bin/pip install neurokit2 pandas
ECGPIPE_PATH=/tmp/open-holter-base/analysis .venv/bin/python tools/eval_physionet.py mitdb --subset DS2
# (same for the other commands)

# Unit tests (synthetic signals, no data needed)
.venv/bin/python -m pytest -q tests
```

`--markdown FILE` writes the tables, and `--rules '{"v_corr": 0.75}'` overrides
thresholds for tuning experiments (on DS1). Synthetic noise is seeded per record,
so runs are repeatable.

## Per-record results (after)

Columns as in the summary. Q_pct is the share of detected beats labelled unreadable.

<details><summary>MIT-BIH DS1 (tuning set): per record</summary>

| record | ref_beats | beat_Se | beat_PPV | Q_pct | V_ref | V_Se | V_PPV | S_ref | S_Se | S_PPV | Vrun_ref | Vrun_found | Vrun_det | Vrun_true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 101 | 1863 | 0.999 | 0.998 | 0.000 | 0 | – | 0.000 | 3 | 0.667 | 0.667 | 0 | 0 | 0 | 0 |
| 106 | 2025 | 1.000 | 1.000 | 0.000 | 520 | 0.998 | 0.998 | 0 | – | 0.000 | 1 | 1 | 2 | 1 |
| 108 | 1761 | 0.993 | 0.981 | 1.5 | 17 | 0.824 | 0.072 | 4 | 0.750 | 0.300 | 0 | 0 | 10 | 0 |
| 109 | 2528 | 1.000 | 1.000 | 0.000 | 38 | 0.816 | 0.969 | 0 | – | – | 0 | 0 | 0 | 0 |
| 112 | 2537 | 1.000 | 1.000 | 0.000 | 0 | – | 0.000 | 2 | 0.500 | 1.000 | 0 | 0 | 0 | 0 |
| 114 | 1877 | 1.000 | 0.999 | 0.000 | 43 | 1.000 | 0.977 | 12 | 0.750 | 1.000 | 0 | 0 | 0 | 0 |
| 115 | 1950 | 1.000 | 1.000 | 0.000 | 0 | – | – | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| 116 | 2409 | 0.992 | 1.000 | 0.000 | 109 | 0.991 | 0.931 | 1 | 1.000 | 1.000 | 0 | 0 | 1 | 0 |
| 118 | 2276 | 1.000 | 0.998 | 0.000 | 16 | 0.938 | 0.789 | 96 | 0.896 | 0.977 | 0 | 0 | 0 | 0 |
| 119 | 1985 | 1.000 | 0.996 | 0.000 | 444 | 1.000 | 0.982 | 0 | – | – | 0 | 0 | 0 | 0 |
| 122 | 2473 | 1.000 | 1.000 | 0.000 | 0 | – | – | 0 | – | – | 0 | 0 | 0 | 0 |
| 124 | 1617 | 1.000 | 1.000 | 0.000 | 47 | 0.979 | 1.000 | 31 | 0.032 | 0.500 | 3 | 3 | 3 | 3 |
| 201 | 1961 | 0.996 | 1.000 | 0.000 | 198 | 0.879 | 0.685 | 128 | 0.109 | 0.179 | 0 | 0 | 0 | 0 |
| 203 | 2978 | 0.993 | 0.993 | 0.269 | 444 | 0.941 | 0.633 | 2 | 0.000 | 0.000 | 21 | 19 | 56 | 19 |
| 205 | 2654 | 0.999 | 1.000 | 0.000 | 71 | 0.986 | 0.986 | 3 | 1.000 | 1.000 | 6 | 6 | 6 | 6 |
| 207 | 1857 | 0.998 | 0.996 | 0.107 | 209 | 0.952 | 0.679 | 106 | 0.604 | 0.901 | 3 | 3 | 12 | 4 |
| 208 | 2951 | 0.995 | 0.999 | 0.000 | 991 | 0.938 | 0.984 | 2 | 0.000 | 0.000 | 0 | 0 | 3 | 0 |
| 209 | 3003 | 1.000 | 0.999 | 0.000 | 1 | 1.000 | 0.045 | 383 | 0.880 | 0.985 | 0 | 0 | 0 | 0 |
| 215 | 3359 | 1.000 | 1.000 | 0.119 | 164 | 0.994 | 0.994 | 3 | 1.000 | 1.000 | 2 | 2 | 3 | 2 |
| 220 | 2045 | 1.000 | 1.000 | 0.000 | 0 | – | 0.000 | 94 | 0.723 | 0.971 | 0 | 0 | 0 | 0 |
| 223 | 2603 | 1.000 | 1.000 | 0.000 | 473 | 0.584 | 0.993 | 73 | 0.616 | 0.789 | 7 | 1 | 2 | 2 |
| 230 | 2253 | 1.000 | 1.000 | 0.000 | 1 | 1.000 | 0.500 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| **gross** | 50965 | 0.998 | 0.998 | 0.078 | 3786 | 0.912 | 0.837 | 943 | 0.676 | 0.809 | 43 | 35 | 98 | 37 |

S runs: 26 reference, 17 found; 24 detected, 18 overlap a reference S run. V beats lost in unreadable stretches: 2; S beats: 0. False V runs: 61 in 11.0 h (5.5 per hour).

</details>

<details><summary>MIT-BIH DS2: per record</summary>

| record | ref_beats | beat_Se | beat_PPV | Q_pct | V_ref | V_Se | V_PPV | S_ref | S_Se | S_PPV | Vrun_ref | Vrun_found | Vrun_det | Vrun_true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 2270 | 1.000 | 1.000 | 0.000 | 1 | 1.000 | 0.333 | 33 | 0.939 | 1.000 | 0 | 0 | 0 | 0 |
| 103 | 2082 | 1.000 | 1.000 | 0.000 | 0 | – | – | 2 | 0.500 | 1.000 | 0 | 0 | 0 | 0 |
| 105 | 2570 | 0.998 | 0.987 | 0.808 | 41 | 0.098 | 0.035 | 0 | – | 0.000 | 0 | 0 | 14 | 0 |
| 111 | 2122 | 0.999 | 0.999 | 0.000 | 1 | 1.000 | 0.050 | 0 | – | 0.000 | 0 | 0 | 1 | 0 |
| 113 | 1792 | 1.000 | 1.000 | 0.000 | 0 | – | – | 6 | 1.000 | 0.092 | 0 | 0 | 0 | 0 |
| 117 | 1533 | 1.000 | 1.000 | 0.000 | 0 | – | – | 1 | 1.000 | 0.500 | 0 | 0 | 0 | 0 |
| 121 | 1861 | 1.000 | 1.000 | 0.000 | 1 | 1.000 | 0.200 | 1 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |
| 123 | 1516 | 1.000 | 1.000 | 0.000 | 3 | 1.000 | 1.000 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| 200 | 2598 | 1.000 | 0.998 | 0.000 | 825 | 0.973 | 0.990 | 30 | 0.100 | 0.375 | 6 | 4 | 5 | 4 |
| 202 | 2134 | 0.999 | 1.000 | 0.000 | 19 | 0.947 | 0.818 | 55 | 0.255 | 0.209 | 0 | 0 | 0 | 0 |
| 210 | 2646 | 1.000 | 0.999 | 0.000 | 195 | 0.964 | 0.913 | 22 | 0.409 | 0.060 | 3 | 2 | 2 | 2 |
| 212 | 2745 | 1.000 | 1.000 | 0.000 | 0 | – | 0.000 | 0 | – | – | 0 | 0 | 0 | 0 |
| 213 | 3247 | 1.000 | 1.000 | 0.000 | 220 | 0.273 | 0.952 | 28 | 0.821 | 0.958 | 1 | 1 | 1 | 1 |
| 214 | 2259 | 0.999 | 1.000 | 0.000 | 256 | 0.773 | 0.995 | 0 | – | 0.000 | 2 | 1 | 1 | 1 |
| 219 | 2152 | 1.000 | 1.000 | 0.000 | 64 | 0.906 | 0.967 | 7 | 0.143 | 0.005 | 0 | 0 | 0 | 0 |
| 221 | 2425 | 1.000 | 1.000 | 0.000 | 396 | 1.000 | 1.000 | 0 | – | 0.000 | 2 | 2 | 2 | 2 |
| 222 | 2481 | 1.000 | 0.998 | 0.000 | 0 | – | 0.000 | 209 | 0.258 | 0.692 | 0 | 0 | 67 | 0 |
| 228 | 2051 | 0.996 | 0.988 | 0.677 | 362 | 0.997 | 0.874 | 3 | 0.333 | 0.083 | 0 | 0 | 8 | 0 |
| 231 | 1569 | 1.000 | 1.000 | 0.000 | 2 | 1.000 | 1.000 | 1 | 1.000 | 0.006 | 0 | 0 | 0 | 0 |
| 232 | 1780 | 1.000 | 0.990 | 0.000 | 0 | – | 0.000 | 1382 | 0.012 | 0.773 | 0 | 0 | 0 | 0 |
| 233 | 3075 | 0.999 | 1.000 | 0.000 | 830 | 0.990 | 1.000 | 7 | 1.000 | 0.875 | 6 | 5 | 5 | 5 |
| 234 | 2751 | 1.000 | 1.000 | 0.000 | 3 | 1.000 | 0.750 | 50 | 0.140 | 1.000 | 0 | 0 | 0 | 0 |
| **gross** | 49659 | 1.000 | 0.998 | 0.070 | 3219 | 0.907 | 0.762 | 1837 | 0.096 | 0.178 | 20 | 15 | 106 | 15 |

S runs: 184 reference, 4 found; 42 detected, 4 overlap a reference S run. V beats lost in unreadable stretches: 1; S beats: 0. False V runs: 91 in 11.0 h (8.2 per hour).

</details>

<details><summary>SVDB: per record</summary>

| record | ref_beats | beat_Se | beat_PPV | Q_pct | V_ref | V_Se | V_PPV | S_ref | S_Se | S_PPV | Vrun_ref | Vrun_found | Vrun_det | Vrun_true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 800 | 1882 | 1.000 | 0.996 | 0.000 | 6 | 0.333 | 0.200 | 30 | 0.900 | 0.184 | 0 | 0 | 0 | 0 |
| 801 | 2506 | 1.000 | 1.000 | 0.000 | 267 | 0.903 | 0.956 | 66 | 0.364 | 0.889 | 0 | 0 | 1 | 0 |
| 802 | 1669 | 1.000 | 0.999 | 0.000 | 85 | 0.988 | 0.977 | 0 | – | 0.000 | 2 | 2 | 3 | 3 |
| 803 | 2066 | 1.000 | 1.000 | 0.000 | 143 | 1.000 | 0.993 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| 804 | 2830 | 1.000 | 1.000 | 0.000 | 181 | 0.945 | 0.994 | 0 | – | 0.000 | 3 | 1 | 1 | 1 |
| 805 | 2296 | 0.985 | 0.985 | 0.000 | 207 | 0.831 | 0.835 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| 806 | 3020 | 1.000 | 1.000 | 0.000 | 29 | 0.759 | 1.000 | 40 | 0.725 | 0.967 | 0 | 0 | 0 | 0 |
| 807 | 1939 | 1.000 | 0.999 | 0.000 | 0 | – | – | 29 | 0.724 | 1.000 | 0 | 0 | 0 | 0 |
| 808 | 1773 | 1.000 | 1.000 | 1.1 | 8 | 0.000 | – | 20 | 0.950 | 1.000 | 0 | 0 | 0 | 0 |
| 809 | 2541 | 1.000 | 1.000 | 0.000 | 0 | – | 0.000 | 112 | 0.107 | 1.000 | 0 | 0 | 0 | 0 |
| 810 | 1901 | 0.999 | 0.998 | 0.000 | 56 | 0.982 | 0.932 | 8 | 0.000 | 0.000 | 0 | 0 | 0 | 0 |
| 811 | 1434 | 1.000 | 1.000 | 0.000 | 28 | 1.000 | 1.000 | 1 | 1.000 | 0.025 | 3 | 3 | 3 | 3 |
| 812 | 1840 | 1.000 | 0.994 | 1.2 | 70 | 0.014 | 0.077 | 14 | 0.929 | 0.171 | 0 | 0 | 0 | 0 |
| 820 | 2350 | 1.000 | 1.000 | 0.000 | 3 | 1.000 | 0.750 | 185 | 1.000 | 0.995 | 0 | 0 | 0 | 0 |
| 821 | 3129 | 0.999 | 0.999 | 0.096 | 36 | 0.861 | 0.378 | 614 | 0.754 | 0.994 | 0 | 0 | 0 | 0 |
| 822 | 2332 | 1.000 | 0.998 | 0.000 | 1 | 1.000 | 0.167 | 658 | 0.020 | 0.867 | 0 | 0 | 0 | 0 |
| 823 | 2858 | 1.000 | 1.000 | 0.000 | 0 | – | 0.000 | 382 | 0.785 | 0.958 | 0 | 0 | 0 | 0 |
| 824 | 2417 | 1.000 | 0.999 | 0.000 | 7 | 0.857 | 0.261 | 196 | 0.929 | 0.858 | 0 | 0 | 0 | 0 |
| 825 | 2794 | 1.000 | 1.000 | 0.000 | 0 | – | – | 145 | 1.000 | 0.986 | 0 | 0 | 0 | 0 |
| 826 | 2659 | 1.000 | 1.000 | 0.000 | 0 | – | 0.000 | 80 | 0.800 | 0.258 | 0 | 0 | 0 | 0 |
| 827 | 1860 | 1.000 | 1.000 | 0.000 | 0 | – | – | 18 | 1.000 | 0.900 | 0 | 0 | 0 | 0 |
| 828 | 1911 | 1.000 | 0.997 | 0.104 | 6 | 1.000 | 0.146 | 150 | 0.567 | 0.895 | 0 | 0 | 0 | 0 |
| 829 | 1964 | 1.000 | 0.998 | 0.000 | 0 | – | 0.000 | 80 | 0.912 | 0.839 | 0 | 0 | 0 | 0 |
| 840 | 2383 | 1.000 | 1.000 | 0.000 | 1 | 1.000 | 0.333 | 44 | 0.205 | 0.900 | 0 | 0 | 0 | 0 |
| 841 | 1853 | 1.000 | 1.000 | 0.000 | 254 | 0.024 | 0.857 | 106 | 0.915 | 0.264 | 1 | 0 | 0 | 0 |
| 842 | 2557 | 1.000 | 0.997 | 0.000 | 61 | 0.902 | 0.426 | 104 | 0.173 | 0.462 | 0 | 0 | 3 | 0 |
| 843 | 2704 | 0.998 | 0.992 | 0.331 | 37 | 0.595 | 0.275 | 21 | 0.333 | 0.636 | 2 | 0 | 1 | 0 |
| 844 | 1707 | 1.000 | 1.000 | 0.000 | 0 | – | 0.000 | 59 | 0.797 | 0.783 | 0 | 0 | 0 | 0 |
| 845 | 2885 | 1.000 | 1.000 | 0.139 | 30 | 0.300 | 0.600 | 27 | 0.593 | 0.267 | 0 | 0 | 0 | 0 |
| 846 | 1679 | 0.999 | 0.999 | 0.000 | 0 | – | 0.000 | 20 | 0.600 | 1.000 | 0 | 0 | 0 | 0 |
| 847 | 1760 | 0.999 | 0.988 | 0.393 | 44 | 0.955 | 0.636 | 51 | 0.882 | 0.882 | 0 | 0 | 0 | 0 |
| 848 | 4254 | 0.974 | 1.000 | 3.7 | 1 | 0.000 | 0.000 | 30 | 0.000 | 0.000 | 0 | 0 | 5 | 0 |
| 849 | 2159 | 1.000 | 0.996 | 0.231 | 6 | 0.167 | 0.040 | 74 | 0.270 | 0.741 | 0 | 0 | 0 | 0 |
| 850 | 1834 | 1.000 | 0.999 | 0.000 | 0 | – | 0.000 | 7 | 0.571 | 0.091 | 0 | 0 | 0 | 0 |
| 851 | 2626 | 1.000 | 1.000 | 0.228 | 427 | 0.991 | 0.995 | 32 | 0.281 | 0.529 | 3 | 2 | 2 | 2 |
| 852 | 2661 | 1.000 | 0.999 | 0.075 | 9 | 1.000 | 0.038 | 321 | 0.187 | 0.706 | 0 | 0 | 1 | 0 |
| 853 | 2229 | 0.994 | 1.000 | 0.000 | 64 | 0.859 | 0.917 | 48 | 0.917 | 0.454 | 0 | 0 | 1 | 0 |
| 854 | 2730 | 0.999 | 0.981 | 2.3 | 478 | 0.891 | 0.805 | 460 | 0.150 | 0.885 | 1 | 0 | 6 | 0 |
| 855 | 2536 | 1.000 | 1.000 | 0.000 | 276 | 0.993 | 0.942 | 227 | 0.273 | 0.939 | 1 | 1 | 2 | 1 |
| 856 | 2850 | 1.000 | 0.999 | 0.140 | 28 | 0.964 | 0.675 | 12 | 1.000 | 1.000 | 1 | 1 | 3 | 1 |
| 857 | 2581 | 1.000 | 0.996 | 0.965 | 144 | 0.958 | 0.939 | 37 | 0.622 | 0.719 | 3 | 3 | 4 | 3 |
| 858 | 2178 | 1.000 | 0.999 | 0.000 | 3 | 1.000 | 0.429 | 8 | 1.000 | 0.889 | 1 | 1 | 1 | 1 |
| 859 | 3537 | 0.999 | 0.999 | 2.3 | 464 | 0.864 | 0.980 | 69 | 0.522 | 0.330 | 7 | 2 | 4 | 2 |
| 860 | 2416 | 1.000 | 0.995 | 0.330 | 727 | 0.997 | 0.974 | 17 | 0.059 | 0.250 | 5 | 4 | 7 | 4 |
| 861 | 2470 | 0.999 | 0.996 | 1.3 | 27 | 0.074 | 0.015 | 569 | 0.452 | 0.895 | 1 | 0 | 1 | 0 |
| 862 | 2186 | 0.999 | 1.000 | 0.504 | 23 | 0.739 | 0.895 | 26 | 0.385 | 0.039 | 1 | 1 | 1 | 1 |
| 863 | 3122 | 1.000 | 0.999 | 0.000 | 240 | 0.971 | 0.991 | 321 | 0.807 | 0.989 | 2 | 2 | 2 | 2 |
| 864 | 1893 | 1.000 | 1.000 | 0.000 | 132 | 0.970 | 0.859 | 19 | 0.105 | 0.500 | 13 | 10 | 14 | 10 |
| 865 | 3160 | 0.994 | 0.989 | 14.9 | 234 | 0.722 | 0.849 | 1818 | 0.015 | 0.848 | 3 | 1 | 2 | 1 |
| 866 | 2663 | 1.000 | 0.998 | 0.000 | 446 | 0.271 | 0.680 | 167 | 0.257 | 0.417 | 9 | 1 | 2 | 1 |
| 867 | 2998 | 1.000 | 0.998 | 0.133 | 75 | 0.973 | 0.802 | 136 | 0.316 | 0.156 | 1 | 1 | 1 | 1 |
| 868 | 3345 | 0.999 | 1.000 | 4.6 | 554 | 0.103 | 0.877 | 227 | 0.586 | 0.270 | 10 | 0 | 0 | 0 |
| 869 | 2158 | 1.000 | 1.000 | 0.000 | 22 | 0.955 | 0.750 | 631 | 0.716 | 0.956 | 1 | 1 | 1 | 1 |
| 870 | 2675 | 0.999 | 0.999 | 0.037 | 437 | 0.856 | 0.862 | 454 | 0.458 | 0.889 | 2 | 1 | 8 | 1 |
| 871 | 1794 | 1.000 | 0.998 | 0.000 | 8 | 1.000 | 0.667 | 12 | 0.917 | 0.733 | 0 | 0 | 0 | 0 |
| 872 | 1978 | 1.000 | 1.000 | 0.000 | 74 | 1.000 | 1.000 | 9 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |
| 873 | 1682 | 1.000 | 1.000 | 0.000 | 26 | 0.385 | 1.000 | 15 | 0.733 | 1.000 | 0 | 0 | 0 | 0 |
| 874 | 2238 | 1.000 | 0.999 | 0.000 | 41 | 1.000 | 0.932 | 9 | 0.778 | 0.875 | 1 | 1 | 1 | 1 |
| 875 | 2050 | 1.000 | 1.000 | 0.000 | 97 | 0.474 | 1.000 | 8 | 0.500 | 0.211 | 1 | 0 | 0 | 0 |
| 876 | 2139 | 0.999 | 0.993 | 0.046 | 0 | – | 0.000 | 135 | 0.881 | 0.531 | 0 | 0 | 0 | 0 |
| 877 | 2036 | 1.000 | 0.996 | 0.049 | 1 | 1.000 | 0.020 | 196 | 0.653 | 0.831 | 0 | 0 | 1 | 0 |
| 878 | 1909 | 1.000 | 0.998 | 0.000 | 256 | 1.000 | 0.977 | 64 | 0.484 | 0.443 | 0 | 0 | 0 | 0 |
| 879 | 2060 | 1.000 | 0.999 | 0.000 | 542 | 0.998 | 0.996 | 48 | 0.688 | 0.917 | 0 | 0 | 0 | 0 |
| 880 | 3483 | 1.000 | 1.000 | 1.7 | 141 | 0.043 | 0.750 | 241 | 0.759 | 0.644 | 0 | 0 | 0 | 0 |
| 881 | 2261 | 0.999 | 0.992 | 0.000 | 307 | 0.599 | 0.637 | 628 | 0.253 | 0.841 | 5 | 0 | 2 | 0 |
| 882 | 1933 | 1.000 | 1.000 | 0.103 | 5 | 0.800 | 1.000 | 41 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |
| 883 | 1818 | 1.000 | 1.000 | 0.000 | 7 | 1.000 | 1.000 | 40 | 1.000 | 1.000 | 1 | 1 | 1 | 1 |
| 884 | 2880 | 1.000 | 1.000 | 0.104 | 343 | 0.988 | 0.994 | 53 | 0.792 | 0.840 | 0 | 0 | 0 | 0 |
| 885 | 1958 | 0.999 | 0.978 | 0.200 | 54 | 0.926 | 0.379 | 399 | 0.632 | 0.973 | 0 | 0 | 0 | 0 |
| 886 | 2220 | 0.999 | 0.955 | 5.9 | 27 | 0.741 | 0.250 | 26 | 0.962 | 0.595 | 0 | 0 | 0 | 0 |
| 887 | 2688 | 0.998 | 0.990 | 0.922 | 210 | 0.971 | 0.833 | 125 | 0.816 | 0.962 | 0 | 0 | 0 | 0 |
| 888 | 2310 | 1.000 | 1.000 | 0.000 | 54 | 1.000 | 0.293 | 68 | 0.221 | 0.938 | 0 | 0 | 5 | 0 |
| 889 | 1707 | 1.000 | 0.992 | 0.174 | 91 | 0.692 | 0.818 | 152 | 0.520 | 0.767 | 0 | 0 | 1 | 0 |
| 890 | 2172 | 0.999 | 0.999 | 0.000 | 97 | 0.577 | 0.949 | 74 | 0.986 | 0.676 | 1 | 0 | 0 | 0 |
| 891 | 2603 | 1.000 | 1.000 | 0.000 | 207 | 0.710 | 0.942 | 277 | 0.866 | 0.842 | 0 | 0 | 0 | 0 |
| 892 | 2841 | 1.000 | 1.000 | 0.000 | 577 | 0.931 | 0.927 | 523 | 0.331 | 0.925 | 2 | 1 | 1 | 1 |
| 893 | 2525 | 1.000 | 0.999 | 0.000 | 338 | 1.000 | 0.955 | 0 | – | 0.000 | 1 | 1 | 2 | 1 |
| 894 | 2345 | 1.000 | 1.000 | 0.469 | 53 | 0.623 | 0.892 | 80 | 0.750 | 0.923 | 0 | 0 | 0 | 0 |
| **gross** | 184390 | 0.999 | 0.997 | 0.726 | 9933 | 0.785 | 0.795 | 12193 | 0.455 | 0.653 | 88 | 42 | 94 | 43 |

S runs: 169 reference, 44 found; 123 detected, 44 overlap a reference S run. V beats lost in unreadable stretches: 221; S beats: 365. False V runs: 51 in 39.0 h (1.3 per hour).

</details>

<details><summary>INCART: per record</summary>

| record | ref_beats | beat_Se | beat_PPV | Q_pct | V_ref | V_Se | V_PPV | S_ref | S_Se | S_PPV | Vrun_ref | Vrun_found | Vrun_det | Vrun_true |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I01 | 2755 | 0.985 | 0.934 | 0.654 | 344 | 0.991 | 0.354 | 0 | – | 0.000 | 0 | 0 | 85 | 0 |
| I02 | 2671 | 0.998 | 0.996 | 0.037 | 229 | 0.996 | 0.507 | 0 | – | 0.000 | 0 | 0 | 23 | 0 |
| I03 | 2449 | 0.986 | 0.976 | 5.1 | 125 | 0.920 | 0.714 | 2 | 0.500 | 0.125 | 0 | 0 | 5 | 0 |
| I04 | 2421 | 0.989 | 0.975 | 3.5 | 121 | 0.818 | 0.387 | 16 | 0.438 | 0.778 | 0 | 0 | 10 | 0 |
| I05 | 1774 | 0.989 | 0.994 | 0.227 | 247 | 0.972 | 0.795 | 0 | – | 0.000 | 14 | 13 | 20 | 13 |
| I06 | 2491 | 1.000 | 0.999 | 0.000 | 9 | 1.000 | 0.450 | 48 | 0.854 | 0.854 | 0 | 0 | 0 | 0 |
| I07 | 2703 | 1.000 | 1.000 | 0.000 | 1 | 1.000 | 0.053 | 65 | 0.846 | 0.982 | 0 | 0 | 0 | 0 |
| I08 | 2129 | 0.999 | 0.987 | 0.650 | 350 | 0.991 | 0.783 | 2 | 0.000 | 0.000 | 0 | 0 | 19 | 0 |
| I09 | 2994 | 1.000 | 0.998 | 0.000 | 41 | 1.000 | 0.100 | 0 | – | – | 0 | 0 | 31 | 0 |
| I10 | 3678 | 0.986 | 1.000 | 0.276 | 83 | 0.976 | 0.195 | 0 | – | – | 0 | 0 | 29 | 0 |
| I11 | 2104 | 1.000 | 0.997 | 0.000 | 4 | 1.000 | 0.308 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I12 | 2806 | 1.000 | 0.999 | 0.000 | 6 | 1.000 | 0.462 | 1 | 1.000 | 1.000 | 0 | 0 | 0 | 0 |
| I13 | 2021 | 1.000 | 1.000 | 0.000 | 230 | 1.000 | 1.000 | 0 | – | – | 0 | 0 | 0 | 0 |
| I14 | 1864 | 1.000 | 0.998 | 0.000 | 64 | 1.000 | 0.941 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I15 | 2632 | 1.000 | 0.999 | 0.000 | 3 | 0.333 | 0.250 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I16 | 1520 | 1.000 | 0.992 | 0.261 | 2 | 1.000 | 0.105 | 0 | – | 0.000 | 0 | 0 | 2 | 0 |
| I17 | 1672 | 0.992 | 0.994 | 0.360 | 27 | 1.000 | 0.628 | 0 | – | 0.000 | 0 | 0 | 3 | 0 |
| I18 | 3082 | 1.000 | 0.998 | 0.551 | 366 | 0.948 | 0.842 | 0 | – | 0.000 | 0 | 0 | 3 | 0 |
| I19 | 2060 | 1.000 | 0.997 | 0.000 | 849 | 0.832 | 0.990 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I20 | 2649 | 0.990 | 0.998 | 0.000 | 110 | 0.755 | 0.790 | 179 | 0.933 | 0.954 | 0 | 0 | 0 | 0 |
| I21 | 2182 | 1.000 | 0.999 | 0.000 | 8 | 1.000 | 0.800 | 104 | 0.971 | 0.802 | 0 | 0 | 0 | 0 |
| I22 | 3124 | 0.996 | 0.999 | 0.032 | 185 | 0.914 | 0.960 | 124 | 0.952 | 0.868 | 1 | 1 | 1 | 1 |
| I23 | 2203 | 1.000 | 1.000 | 0.000 | 13 | 1.000 | 1.000 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I24 | 2568 | 1.000 | 0.999 | 0.000 | 6 | 0.833 | 0.556 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I25 | 1710 | 1.000 | 0.974 | 1.3 | 5 | 1.000 | 0.079 | 2 | 1.000 | 0.500 | 1 | 1 | 2 | 1 |
| I26 | 1509 | 0.999 | 0.997 | 0.198 | 4 | 1.000 | 0.190 | 7 | 0.857 | 0.750 | 0 | 0 | 0 | 0 |
| I27 | 2603 | 1.000 | 1.000 | 0.115 | 719 | 1.000 | 0.993 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I28 | 1715 | 1.000 | 0.999 | 0.350 | 4 | 1.000 | 0.571 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I29 | 2618 | 0.996 | 0.991 | 1.2 | 783 | 0.968 | 0.820 | 0 | – | 0.000 | 26 | 21 | 57 | 21 |
| I30 | 2458 | 1.000 | 0.996 | 0.000 | 755 | 0.950 | 0.938 | 0 | – | 0.000 | 17 | 15 | 23 | 16 |
| I31 | 3207 | 0.997 | 0.999 | 0.281 | 1363 | 0.811 | 0.792 | 0 | – | 0.000 | 47 | 15 | 86 | 15 |
| I32 | 1617 | 1.000 | 0.998 | 0.000 | 57 | 1.000 | 0.891 | 0 | – | 0.000 | 0 | 0 | 1 | 0 |
| I33 | 1835 | 0.999 | 0.997 | 0.000 | 1 | 1.000 | 0.071 | 589 | 0.642 | 0.936 | 0 | 0 | 1 | 0 |
| I34 | 1963 | 0.998 | 0.996 | 0.000 | 0 | – | 0.000 | 536 | 0.606 | 0.967 | 0 | 0 | 1 | 0 |
| I35 | 3672 | 1.000 | 1.000 | 0.109 | 457 | 0.904 | 0.990 | 0 | – | – | 0 | 0 | 0 | 0 |
| I36 | 3907 | 1.000 | 1.000 | 9.5 | 450 | 0.589 | 0.895 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I37 | 2459 | 1.000 | 1.000 | 0.000 | 452 | 1.000 | 0.998 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I38 | 2697 | 0.999 | 0.999 | 0.185 | 543 | 0.974 | 0.998 | 0 | – | 0.000 | 2 | 1 | 1 | 1 |
| I39 | 1772 | 1.000 | 0.998 | 0.000 | 313 | 0.987 | 0.987 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I40 | 2664 | 0.997 | 0.984 | 0.593 | 92 | 1.000 | 0.195 | 6 | 0.000 | 0.000 | 0 | 0 | 39 | 0 |
| I41 | 1628 | 1.000 | 0.997 | 0.367 | 1 | 1.000 | 0.048 | 5 | 0.600 | 0.033 | 0 | 0 | 1 | 0 |
| I42 | 3106 | 1.000 | 1.000 | 0.000 | 1553 | 0.854 | 0.940 | 0 | – | 0.000 | 6 | 4 | 14 | 4 |
| I43 | 2206 | 1.000 | 1.000 | 0.000 | 1119 | 0.191 | 0.196 | 0 | – | 0.000 | 2 | 0 | 0 | 0 |
| I44 | 2492 | 1.000 | 1.000 | 0.000 | 683 | 1.000 | 0.999 | 8 | 0.125 | 1.000 | 0 | 0 | 0 | 0 |
| I45 | 1926 | 1.000 | 1.000 | 0.000 | 491 | 1.000 | 0.998 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I46 | 2656 | 1.000 | 1.000 | 0.113 | 422 | 0.995 | 0.940 | 1 | 1.000 | 0.500 | 0 | 0 | 0 | 0 |
| I47 | 1951 | 1.000 | 0.999 | 0.102 | 93 | 0.978 | 0.938 | 1 | 0.000 | 0.000 | 0 | 0 | 0 | 0 |
| I48 | 2355 | 1.000 | 0.999 | 0.000 | 235 | 1.000 | 0.987 | 2 | 0.000 | 0.000 | 0 | 0 | 0 | 0 |
| I49 | 2145 | 1.000 | 1.000 | 0.000 | 27 | 1.000 | 0.931 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I50 | 2995 | 1.000 | 1.000 | 0.000 | 4 | 0.750 | 0.176 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I51 | 2775 | 1.000 | 1.000 | 0.144 | 802 | 0.797 | 0.967 | 3 | 1.000 | 0.018 | 0 | 0 | 0 | 0 |
| I52 | 1745 | 1.000 | 1.000 | 0.000 | 137 | 1.000 | 1.000 | 0 | – | – | 0 | 0 | 0 | 0 |
| I53 | 2258 | 1.000 | 0.994 | 0.176 | 1109 | 0.331 | 0.895 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I54 | 2361 | 0.965 | 0.899 | 18.4 | 22 | 0.818 | 0.130 | 1 | 1.000 | 0.062 | 1 | 1 | 8 | 1 |
| I55 | 2164 | 1.000 | 0.991 | 0.183 | 17 | 1.000 | 0.405 | 1 | 0.000 | 0.000 | 1 | 1 | 1 | 1 |
| I56 | 1703 | 1.000 | 0.999 | 0.000 | 7 | 1.000 | 0.875 | 26 | 1.000 | 0.129 | 0 | 0 | 0 | 0 |
| I57 | 2868 | 0.999 | 0.999 | 0.174 | 21 | 1.000 | 0.488 | 3 | 0.667 | 0.667 | 0 | 0 | 1 | 0 |
| I58 | 2323 | 1.000 | 0.999 | 0.000 | 12 | 1.000 | 0.429 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I59 | 2146 | 1.000 | 0.996 | 0.000 | 81 | 0.963 | 0.703 | 0 | – | 0.000 | 3 | 2 | 8 | 2 |
| I60 | 2472 | 1.000 | 0.996 | 0.000 | 0 | – | 0.000 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| I61 | 1452 | 1.000 | 1.000 | 0.000 | 0 | – | – | 1 | 1.000 | 0.500 | 0 | 0 | 0 | 0 |
| I62 | 2268 | 1.000 | 1.000 | 0.000 | 795 | 0.194 | 0.452 | 9 | 0.444 | 0.052 | 14 | 11 | 20 | 13 |
| I63 | 1992 | 0.999 | 0.996 | 0.000 | 138 | 0.957 | 0.898 | 1 | 1.000 | 0.167 | 7 | 7 | 7 | 7 |
| I64 | 1911 | 0.999 | 0.994 | 0.364 | 23 | 0.957 | 0.489 | 0 | – | 0.000 | 1 | 1 | 2 | 1 |
| I65 | 2662 | 1.000 | 0.998 | 0.150 | 382 | 0.916 | 0.970 | 5 | 0.400 | 0.083 | 3 | 3 | 4 | 3 |
| I66 | 2338 | 1.000 | 0.996 | 0.085 | 200 | 1.000 | 0.926 | 1 | 0.000 | 0.000 | 1 | 1 | 1 | 1 |
| I67 | 2972 | 1.000 | 0.997 | 0.000 | 531 | 1.000 | 0.932 | 5 | 0.800 | 0.118 | 1 | 1 | 1 | 1 |
| I68 | 2642 | 0.996 | 0.979 | 2.3 | 161 | 0.975 | 0.545 | 2 | 0.000 | 0.000 | 0 | 0 | 8 | 0 |
| I69 | 2166 | 0.998 | 0.988 | 0.914 | 167 | 1.000 | 0.874 | 1 | 0.000 | 0.000 | 0 | 0 | 2 | 0 |
| I70 | 1664 | 1.000 | 0.998 | 0.000 | 0 | – | 0.000 | 126 | 0.024 | 0.750 | 0 | 0 | 0 | 0 |
| I71 | 1668 | 1.000 | 0.999 | 0.000 | 0 | – | 0.000 | 35 | 0.086 | 0.188 | 0 | 0 | 0 | 0 |
| I72 | 2267 | 1.000 | 1.000 | 24.7 | 386 | 0.430 | 0.869 | 8 | 0.625 | 0.455 | 0 | 0 | 3 | 0 |
| I73 | 1990 | 1.000 | 1.000 | 0.101 | 70 | 0.971 | 1.000 | 32 | 1.000 | 0.889 | 0 | 0 | 0 | 0 |
| I74 | 2402 | 1.000 | 0.998 | 0.000 | 274 | 0.712 | 0.965 | 0 | – | – | 0 | 0 | 0 | 0 |
| I75 | 2100 | 1.000 | 1.000 | 0.000 | 611 | 0.989 | 1.000 | 0 | – | 0.000 | 0 | 0 | 0 | 0 |
| **gross** | 175727 | 0.998 | 0.994 | 1.1 | 19995 | 0.807 | 0.758 | 1958 | 0.661 | 0.290 | 148 | 99 | 523 | 102 |

S runs: 16 reference, 5 found; 98 detected, 5 overlap a reference S run. V beats lost in unreadable stretches: 402; S beats: 4. False V runs: 421 in 37.5 h (11.2 per hour).

</details>

