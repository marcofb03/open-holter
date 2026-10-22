# Open single-lead arrhythmia options on H10-like data – benchmark

> **Not a medical device.** These numbers compare open research tools on public
> annotated databases. None of the systems is certified, and every flagged strip
> still needs a cardiologist. The owner prefers more false alarms to a missed
> event, so everything below is tuned **sensitivity first**.

This page has two parts:
1. **The long-term benchmark and the sensitivity-first router.** 2,515 records,
   6,000 h; held-out: 344 patients, 4,925 h, 232 of them recorded for ≥ 10 h.
   Machine-generated tables are in [`results/router.md`](results/router.md).
2. **The earlier benchmark** on 30-minute research records (MIT-BIH DS2,
   SVDB, INCART) with its own ranking. It is kept unchanged below.

## 1. Long-term benchmark (held-out: LTAFDB, AFDB, SHDB-AF, LTDB, CPSC2021, VFDB, CUDB, Icentia11k)

### Data, split and systems

Every record was resampled to 130 Hz (the Polar H10 rate) and passed through
ECGLogger CSV files of one hour each, as the phone writes them. All systems
share the rule pipeline's beat detector and noise gate. The split is at
**patient level** and was fixed before any system was run
(`datasets.py`).

| database | what | lead | tuning | held-out |
|---|---|---|---|---:|
| MIT-BIH, SVDB, INCART | 30 min arrhythmia records (the previous benchmark's development and held-out sets) | MLII / ECG1 / II | all 197 | – |
| LTAFDB | 84 x ~24 h, AF + beat labels | ECG 1 | 28 patients (661 h) | 56 patients (1,300 h) |
| CPSC2021 | 105 patients, paroxysmal AF | II | 31 (139 h) | 74 (342 h) |
| VFDB, CUDB | VF / VT episodes | 1 | 8 + 11 | 14 + 24 |
| Icentia11k sample | 60 patients drawn from **10,000–10,999** (heartKIT trained on 0–9,999), first 10 segments (~11.7 h) each | 1 | 12 (140 h) | 48 (559 h) |
| AFDB | 23 x 10 h, AF | ECG1 | – | 23 (234 h) |
| SHDB-AF | 98 x 24 h Japanese Holters, AF + AT/PAT/NOD | ECG1 | – | 98 (2,331 h) |
| LTDB | 7 x ~21 h, beat labels | ECG1 | – | 7 (147 h) |
| NSTDB | 118 / 119 with electrode-motion (PhysioNet mixes) and muscle (ours) noise, 24 to −6 dB | MLII | noise stress only | |

The tuning shares were chosen by `sha256("<db>:<patient>") mod 3 == 0`.
AFDB, SHDB-AF and LTDB have no tuning share at all, so they test transfer to
new databases.

| system | what it is |
|---|---|
| **rules** | `analysis/ecgpipe` (rule-based N/S/V/Q, irregular-rhythm flag, RR intervals) |
| **heartKIT** | Ambiq BEAT-3-EFF-SM on the shared beats + ARR-2-EFF-SM AF frames (BSD-3) |
| **ECGFounder** | 1-lead foundation model, 150 labels per 10 s window (MIT), fp32 |
| **CLEF** | Nokia Bell Labs CLEF-Medium (30.7 M parameters, frozen, bf16) + a small head (1024→256→8) trained **on the tuning patients only**, cross-fitted in 5 patient folds so the router sees out-of-fold scores |

Findings (`findings.py`) and what makes a detection "false":

| finding | reference | assessable where |
|---|---|---|
| VT/NSVT/VF | ≥ 3 consecutive V beats, or a VT / VF / VFL episode | beat labels, or VFDB / CUDB rhythm labels |
| AF/AFL | ≥ 30 s | rhythm labels |
| sustained SVT | S run or SVTA / AT / PAT episode ≥ 30 s | beat labels, MIT-BIH, LTAFDB and SHDB-AF rhythm codes |
| frequent PVC / PAC | ≥ 1 % of beats (per patient) | beat labels |
| pause | RR 3–15 s between readable beats, outside unreadable spans | beat labels |
| brady / tachy | 8-beat median < 40 / > 150 bpm for ≥ 10 s | beat labels |

Per-patient sensitivity counts patients in whom at least one reference episode
was found, with 95 % Wilson intervals. A false finding is a detection that
overlaps no reference episode or related span (`findings.ref_findings`).
False findings per day are reported on all held-out records and on the
**long (≥ 10 h) recordings** (LTAFDB, AFDB, SHDB-AF, LTDB, Icentia11k: 232
patients, 4,571 h).

### Best system per finding (plain summary, held-out)

| finding | use | per-patient Se [95 % CI] | false findings / day (long recordings) | best single system |
|---|---|---:|---:|---|
| **AF/AFL ≥ 30 s** | ECGFounder windows ≥ 0.02, kept where heartKIT AF ≥ 0.5 | **100 %** (201/201) [98–100] | **1.7** | ECGFounder (98.5 %, 14.5/day) |
| **frequent PVCs** | heartKIT AND rules, burden ≥ 0.3 % | **100 %** (23/23) [86–100] | 0.5 | rules or heartKIT (100 %) |
| **frequent PACs** | heartKIT OR rules, burden ≥ 1 % | **100 %** (24/24) [86–100] | 1.0 | rules (100 %, 0.7/day) |
| **brady < 40 bpm** | shared detector < 45 bpm, kept where ECGFounder ≥ 0.02 | **100 %** (12/12) [76–100] | 0.06 | ECGFounder / CLEF (100 %, but 62–89/day) |
| **tachy > 150 bpm** | shared detector > 140 bpm, kept where ECGFounder ≥ 0.02 | 96.8 % (30/31) [84–99] | 0.5 | ECGFounder (100 %, 65/day) |
| **VT/NSVT/VF** | ECGFounder windows ≥ 0.02 containing a V run of heartKIT-OR-rules | 91.4 % (64/70) [83–96] | 38 | ECGFounder ≥ 0.02 alone (98.6 %, 734/day) |
| VT/NSVT/VF, strict | ≥ 2 of ECGFounder, CLEF, heartKIT-OR-rules runs, long unreadable stretches | 95.7 % (67/70) [88–99] | 126 | |
| **sustained SVT** | ECGFounder SVT ≥ 0.1 for ≥ 30 s | 88.9 % (8/9) [56–98] | 17 | ECGFounder / CLEF (88.9 %) |
| **pause > 3 s** | shared detector RR ≥ 2.5 s, kept where ECGFounder ≥ 0.02 | 63.6 % (7/11) [35–85] | 6.1 | ECGFounder ≥ 0.02 (90.9 %, 966/day) |

How to read it:
- **AF, frequent ectopy, brady and tachy reach the target** (≥ 98 % where the
  CI allows) at a handful of false findings per day or fewer.
- **VT falls short of 98 % on held-out data** (91 %) although it reached 98.9 %
  (89/90) on tuning. Of the 6 missed patients, 4 are in LTAFDB, 1 in Icentia11k and
  1 a VFDB VF episode. The strict variant (target 100 % on tuning) finds 3 more
  at 3x the false findings. ECGFounder alone at 0.02 would find 69/70, but
  with 734 false runs per day it is not usable.
- **Pauses are the weakest finding.** The shared detector misses pauses when a
  noise spike in the gap is taken for a beat. ECGFounder or CLEF see them,
  but only at hundreds of false flags per day. This is the first target for
  the fine-tuned beat model with a noise class (`../FINETUNING_PLAN.md`).
- **Sustained SVT** has only 9 held-out patients (6 in SHDB-AF), so its CI
  is wide.

### Sensitivity-first router

`router.py` chooses, **on tuning data only** and per finding, among 111–348
candidates:
- every system at a grid of thresholds;
- OR-unions of 2–4 systems;
- "≥ 2 systems agree" votes;
- "A confirmed by B" pairs.

It ranks them by per-patient sensitivity capped at 98 %, then by fewest false
findings per day, then by episode sensitivity. The table also shows the
strict variant (cap 100 %).

| finding | router choice (tuning Se) | held-out Se [95 % CI] | PPV | false / day all held-out | false / day long | strict 100 % variant: held-out Se; false / day long |
|---|---|---:|---:|---:|---:|---|
| VT/NSVT/VF | ECGFounder ≥ 0.02, confirmed by heartKIT OR rules ≥ 0.5 (98.9 %, 89/90) | 91.4 % (64/70) [83–96] | 0.17 | 37.7 | 37.7 | 95.7 % (67/70); 126 |
| AF/AFL ≥ 30 s | ECGFounder ≥ 0.02, confirmed by heartKIT ≥ 0.5 (100 %, 50/50) | 100 % (201/201) [98–100] | 0.55 | 2.6 | 1.7 | same |
| sustained SVT | ECGFounder ≥ 0.1 (100 %, 5/5) | 88.9 % (8/9) [56–98] | 0.52 | 16.9 | 16.9 | same |
| frequent PVCs | heartKIT AND rules ≥ 0.3 % (99.2 %, 131/132) | 100 % (23/23) [86–100] | 0.37 | 0.47 | 0.47 | 95.7 % (22/23); 0.47 |
| frequent PACs | heartKIT OR rules ≥ 1 % (99.0 %, 95/96) | 100 % (24/24) [86–100] | 0.22 | 0.99 | 0.99 | 100 % (24/24); 0.85 |
| pause > 3 s | ECGFounder ≥ 0.02, confirmed by rules RR ≥ 2.5 s (100 %, 9/9) | 63.6 % (7/11) [35–85] | 0.32 | 6.1 | 6.1 | same |
| brady < 40 bpm | ECGFounder ≥ 0.02, confirmed by rules < 45 bpm (100 %, 13/13) | 100 % (12/12) [76–100] | 0.91 | 0.06 | 0.06 | same |
| tachy > 150 bpm | ECGFounder ≥ 0.02, confirmed by rules > 140 bpm (100 %, 22/22) | 96.8 % (30/31) [84–99] | 0.58 | 0.47 | 0.47 | same |

PPV: the share of detections that are not false (for PVC and PAC, of flagged
patients). The tuning Pareto front of each finding (5 operating points with
held-out Se, PPV and false/day), the per-database breakdown and the
noise-stress table are in [`results/router.md`](results/router.md).

A pattern runs through the choices. **A very sensitive ECGFounder window
(≥ 0.02) confirmed by the beat-level systems** wins for VT, AF, pause, brady
and tachy. ECGFounder decides *where* to look, and the beat detector and
heartKIT decide *whether* something happened. The best choice is a combination
for 7 of the 8 findings (all but sustained SVT).

**Each system at its own default operating point** (no tuning; held-out
per-patient Se; false findings per day on long recordings):

| finding | rules | heartKIT (0.5 / AF 0.75) | ECGFounder (0.5) | CLEF + head (0.5) |
|---|---|---|---|---|
| VT/NSVT/VF | 81.4 %; 407 | 80.0 %; 71 | 51.4 %; 4.9 | 67.1 %; 23 |
| AF/AFL | 96.0 %; 24 | 99.0 %; 19 | 99.0 %; 46 | 95.5 %; 36 |
| sustained SVT | 0 %; 0 | 22 %; 0.3 | 44 %; 0.1 | 56 %; 12 |
| frequent PVCs | 100 %; 0.8 | 95.7 %; 0.4 | 91.3 %; 0.5 | 95.7 %; 0.3 |
| frequent PACs | 91.7 %; 0.5 | 100 %; 1.0 | 70.8 %; 0.0 | 79.2 %; 0.4 |
| pause > 3 s | 54.5 %; 3.8 | – (shared detector) | 0 %; 8.3 | 63.6 %; 52 |

**CLEF-Medium, frozen, with a small head trained on tuning data** never
wins a finding outright. Its cross-validated window AUROC on the tuning
patients (`results/clef_head.json`) is 0.94 for AF and 0.93 for VT,
against ECGFounder's ≈ 0.99 for AF in the earlier benchmark. It appears in the
strict VT vote and the strict PAC OR. It is a fine-tuning candidate, not a
drop-in reader (`../FINETUNING_PLAN.md`).

**Noise stress** (NSTDB, router choices, false findings per hour at 24 / 12 /
0 / −6 dB): VT 0 / 3.5 / 8.5 / 8.5, AF 1.0 / 1.5 / 2.0 / 2.0, every other
finding ≤ 1. Noise mostly costs **false VT runs**: the same weakness as in
the earlier benchmark.

### Open models vs DeepRhythmAI (indicative only)

DeepRhythmAI (Johnson et al., *Nature Medicine* 2025;31:925–931) reported
**98.6 %** per-patient sensitivity for critical arrhythmias and **12 false
findings per 1,000 patient-days** (median, IQR 6–74). That came from 14,606
real ambulatory recordings of about 14 days each, adjudicated by cardiologists.

| | critical findings: per-patient Se (patient–finding pairs) | false findings / day | per 1,000 patient-days |
|---|---:|---:|---:|
| DeepRhythmAI (published) | **98.6 %** | 0.012 | **12** |
| this router, all held-out (VT, AF, SVT, pause, brady) | 96.4 % (292/303) [94–98] | 63 | 63,300 |
| this router, long recordings only | 95.5 % (213/223) [92–98] | 63 | 62,500 |
| earlier ensemble (30-min records, section 2) | 84.6 % | 78 | 77,900 |

**Why this is only indicative:**
- **Data.** Public Holter databases (232 patients ≥ 10 h here, 1–2 days each)
  are not two-week recordings of ordinary patients. They are enriched for AF
  (LTAFDB, AFDB, SHDB-AF, CPSC2021), so AF dominates the patient–finding pairs.
- **Signal.** Hospital Holter leads resampled to 130 Hz, not real H10 strap
  recordings.
- **Definitions.** "Critical" here includes every V run ≥ 3 beats and pauses
  > 3 s. DeepRhythmAI's set and adjudication differ. Most of our false findings
  are short false "V runs" (38/day) and SVT windows (17/day).
- **References.** LTAFDB and Icentia11k beat labels are semi-automatic, and
  LTAFDB contains long unannotated gaps (handled as signal loss, see notes).
  Some "false" findings are probably reference errors.
- **Bottom line.** Combining the open models closes much of the sensitivity
  gap (96 % vs 98.6 %) on public data. It still produces about **5,000 times**
  more false findings per patient-day, almost all of them VT/SVT strips a
  doctor has to dismiss. It is a screening aid that needs every strip read.

### Notes and reference fixes

- **ECGFounder runs in fp32; CLEF in bfloat16.** On MIT-BIH 208, bf16 moved
  ECGFounder probabilities by up to 0.33 (mean 0.005), which is too much to
  keep thresholds comparable. It left CLEF embeddings at cosine ≥ 0.99996.
- **Windows** are non-overlapping 10 s windows (the earlier benchmark used a
  5 s step, at twice the cost). A 3-beat run split by a window edge can
  therefore be missed by the window models.
- **LTDB `~` marks.** Beats are often not annotated in the spans the
  annotators marked unreadable (subtype ≠ 0). Those spans are now neither
  reference pauses nor false findings.
- **Unannotated gaps over 15 s**, e.g. LTAFDB 113 (68–515 s, full of noisy
  detections), are treated the same way: signal loss, not asystole.
- **CPSC2021** has 1,425 records in RECORDS but 11 without files; they are
  skipped.
- **Compute:** 2,515 records, about 12 h of wall-clock on 4 CPU cores
  (`frontend.py -j 4`, ~95 % of it ECGFounder and CLEF). The router itself
  takes 4 minutes, mostly loading.

### Reproduce (long-term part)

```bash
cd analysis && ../ai/.venv/bin/python tools/fetch_physionet.py --aws mitdb svdb incartdb ltafdb afdb ltdb vfdb cudb nstdb cpsc2021 shdb-af
cd ../ai/ecgfounder && ../.venv/bin/python download_weights.py
cd ../heartkit && ../.venv/bin/python download_models.py
cd ../clef && ../.venv/bin/python download_weights.py
cd ../benchmark && ../.venv/bin/python frontend.py -j 4 --delete-raw shdb-af cpsc2021 afdb   # Icentia11k streamed, ~12 h
cd ../clef && ../.venv/bin/python head.py                                                     # CLEF head, tuning patients only
cd ../benchmark && ../.venv/bin/python router.py                                              # -> results/router.md, router.json
```

## 2. Earlier benchmark: 30-minute research records (MIT-BIH DS2, SVDB, INCART)

Systems compared on identical input. Every record was resampled to 130 Hz
(the Polar H10 rate) and passed through an ECGLogger CSV round trip. Leads:
MIT-BIH MLII, SVDB ECG1, INCART II. All systems share the rule pipeline's
beat detector and noise gate.

| system | what it is | size on phone |
|---|---|---:|
| **rules** | merged `analysis/ecgpipe` (rule-based N/S/V/Q, events) | – |
| **heartKIT** | Ambiq BEAT-3-EFF-SM (beats, Icentia11k) + ARR-2-EFF-SM (AF), BSD-3 | 106 kB + 64 kB |
| **ECGFounder** | 1-lead foundation model, 10 s windows, MIT | 32 MB int8 / 123 MB fp32 |
| **ensemble** | rules V runs kept only if heartKIT calls ≥ 1 of their beats V; AF and sustained SVT from ECGFounder; pauses from the shared detector | all of the above |

Every threshold and the choice of ensemble were fixed on **MIT-BIH DS1** (22
records). "Held-out" pools **MIT-BIH DS2 (22) + SVDB (78) + INCART (75)**,
175 records and 87.6 h.

### Open models vs DeepRhythmAI published figures (indicative only)

DeepRhythmAI (Johnson et al., *Nature Medicine* 2025;31:925–931) reported a
**98.6 %** sensitivity for critical arrhythmias and **12 false-positive
findings per 1,000 patient-days** (median, IQR 6–74). That came from 14,606
real ambulatory recordings of about 14 days each, adjudicated by cardiologist
panels. Our held-out numbers:

| | all critical findings: per-patient Se | VT/NSVT (≥ 3 V) | AF/AFL ≥ 30 s | pause > 3 s | false findings / day | false findings / 1,000 patient-days |
|---|---:|---:|---:|---:|---:|---:|
| DeepRhythmAI (published) | **98.6 %** | – | – | – | 0.012 | **12** |
| **ensemble** | **84.6 %** | 81.5 % (44/54) | 100 % (8/8) | 100 % (3/3) | 78 | 77,900 |
| rules | 80.0 % | 83.3 % (45/54) | 50 % (4/8) | 100 % (3/3) | 163 | 163,100 |
| heartKIT | 78.5 % | 74.1 % (40/54) | 100 % (8/8) | 100 % (3/3) | 350 | 350,100 |
| ECGFounder | 61.5 % | 55.6 % (30/54) | 100 % (8/8) | 66.7 % (2/3) | 256 | 256,100 |

"Per-patient Se" here is the share of (patient, finding type) pairs in which
the system flagged at least one episode of that type.

**Why this comparison is only indicative:**
- **Data.** Our records are 30-minute research tapes picked for arrhythmias,
  not two-week recordings of ordinary patients. The false rate is measured
  on 87.6 h and scaled up to a day, which inflates it in arrhythmia-dense
  records. The ratio between systems means more than the absolute number.
- **Leads and sampling.** The data are hospital Holter leads resampled to
  130 Hz, not real H10 strap recordings. DeepRhythmAI used its own patch
  devices.
- **Definitions.** DeepRhythmAI's "critical arrhythmias" set and its
  adjudication differ from ours. We count any V run ≥ 3 as VT/NSVT, which is
  much more sensitive to noise than sustained VT.
- **Small numbers.** Only 8 held-out patients have AF ≥ 30 s and 3 have
  pauses > 3 s. No held-out patient has sustained SVT ≥ 30 s; DS1 has one,
  which every system missed. Sustained SVT is therefore **not assessable**
  here.
- **Bottom line.** Even the best open combination misses about one patient
  in six with a critical finding. It also produces thousands of times more
  false findings per day than DeepRhythmAI, most of them short false "V runs"
  in noisy or bundle-branch-block records. It is a screening aid that needs
  every strip read, not a replacement for a validated service.

### Ranking

| rank | system | strengths (held-out) | weaknesses |
|---:|---|---|---|
| 1 | **ensemble** | best overall per-patient Se (84.6 %) with half the rules' false findings; AF per patient 8/8 | still about 76 false VT events per day; misses 10 of 54 patients' V runs |
| 2 | **rules** | most V runs found (158/256, beat level); best VT per-patient Se (83 %); transparent | AF poor (window Se 0.36, 4 of 8 patients); S beats weak (Se 0.44) |
| 3 | **heartKIT** | tiny; AF model nearly as good as ECGFounder (window AUROC 0.99); best S sensitivity (0.74) | about 2× the rules' false V runs (14.5 vs 6.7 per hour); S PPV 0.27 |
| 4 | **ECGFounder** alone | best AF (window Se 0.84 / PPV 0.97, AUROC 0.99) and best window AUROC for PVC (0.97) and PAC (0.93) | weak on V runs (window Se 0.43 / PPV 0.10, per-patient 56 %); 32–123 MB; no beat-level output |

By task:

| task | best | runner-up | notes |
|---|---|---|---|
| AF / flutter | ECGFounder (window Se 0.84, PPV 0.97; 3 false episodes in 87.6 h) | heartKIT ARR-2 (0.78 / 0.92; 2 false) | rules: 0.36 / 0.56 |
| PVC beats | rules AND heartKIT for high-PPV counts (Se 0.75 / PPV 0.96) | heartKIT (0.85 / 0.78) ≈ rules (0.82 / 0.76) | ECGFounder only sees windows (AUROC 0.97) |
| V runs | rules (158/256 runs, 6.7 false/h) | rules confirmed by heartKIT (fewer false runs) | rules AND heartKIT beat-wise: 92/256, 0.3 false/h |
| PAC / S beats | heartKIT for sensitivity (0.74), rules AND heartKIT for PPV (0.71) | ECGFounder windows (AUROC 0.93) | all weak, as expected from one lead without P-wave analysis |
| pauses | shared detector (3/3 patients, 5 false in 87.6 h) | – | ECGFounder's pause/block labels are not usable (509 false) |

### Recommendation for the phone

1. **Keep the rule pipeline as the base.** It provides the detector, noise
   gate, counts, pauses and rate rules (HR < 40, > 220 and > 150 bpm for ≥ 10 s).
2. **Add heartKIT BEAT-3-EFF-SM as a second beat reader.** It is 106 kB and
   costs about 0.15 ms per beat on one core, about 15 CPU-seconds per day.
   - Report a V run only when heartKIT agrees on at least one of its beats.
     That removes about half the rules' false VT findings for a 2-point drop
     in per-patient VT sensitivity.
   - Report PVC and PAC burden from beats both readers call V (or S), for
     high-PPV counts.
   - Keep a low-priority "possible V run" list with the runs only the rules
     found, so nothing is silently dropped.
3. **For AF, run heartKIT ARR-2-EFF-SM on the phone continuously** (64 kB,
   about 0.1 ms per 5 s frame). It caught all 8 held-out AF patients with 2
   false episodes. **Use ECGFounder (int8, 32 MB) only as a second reader on
   windows that are already flagged:** AF episodes and V runs, a few hundred
   windows a day at about 20–120 ms each. It confirms or names the rhythm
   and gives the best AF precision. Running it on every window would cost
   about 10–35 CPU-minutes a day on a Pixel 8a-class phone (estimate, see
   `ai/ecgfounder/RESULTS.md`). That is feasible, but there's no need to spend it.
4. **Next steps for the false-alarm gap:**
   - The biggest remaining error is false short V runs in noise and bundle
     branch block. Fine-tune a beat model on Icentia11k resampled to 130 Hz
     with a noise class, using the `analysis/` detector and DS1 for tuning.
   - Validate on real H10 recordings with cardiologist-read strips before
     trusting any per-day rate.

### Main tables (held-out)

Beats, AAMI EC57 gross, scored with the same code as `analysis/tools/eval_physionet.py`:

| system | V Se | V PPV | S Se | S PPV | V runs found / ref | false V runs / h |
|---|---:|---:|---:|---:|---:|---:|
| rules | 0.824 | 0.762 | 0.437 | 0.517 | 158 / 256 | 6.7 |
| heartKIT | 0.847 | 0.775 | 0.736 | 0.265 | 134 / 256 | 14.5 |
| rules AND heartKIT | 0.752 | 0.959 | 0.383 | 0.711 | 92 / 256 | 0.3 |
| rules OR heartKIT | 0.919 | 0.660 | 0.710 | 0.267 | 208 / 256 | 21.9 |

10 s windows, Se / PPV at the operating point (AUROC of the score):

| target (positives) | rules | heartKIT | rules AND heartKIT | ECGFounder |
|---|---|---|---|---|
| ≥ 1 V (21,793) | 0.92 / 0.83 (0.92) | 0.94 / 0.89 (0.95) | 0.87 / 0.95 (0.93) | 0.88 / 0.91 (0.97) |
| V run ≥ 3 (530) | 0.59 / 0.25 (0.87) | 0.51 / 0.13 (0.84) | 0.31 / 0.78 (0.79) | 0.43 / 0.10 (0.90) |
| ≥ 1 S (11,014) | 0.71 / 0.66 (0.81) | 0.93 / 0.41 (0.83) | 0.64 / 0.74 (0.80) | 0.82 / 0.66 (0.93) |
| AF (1,966) | 0.36 / 0.56 (0.70) | 0.78 / 0.92 (0.99) | – | 0.84 / 0.97 (0.99) |

The VT ensemble candidates (all declared before the held-out run and picked on
DS1 by highest per-patient Se, then fewest false findings):

| VT candidate | DS1 per-patient Se | DS1 false | held-out per-patient Se | held-out false / day |
|---|---:|---:|---:|---:|
| rules | 7/7 | 66 | 83.3 % | 158 |
| **rules, heartKIT-confirmed** (picked) | 7/7 | 47 | 81.5 % | 76 |
| rules, ECGFounder-confirmed | 7/7 | 65 | 83.3 % | 131 |
| rules AND heartKIT (beat-wise) | 5/7 | 17 | 70.4 % | 7 |

All tables per database, including DS1, DS2, SVDB and INCART separately, are
in [`results/tables.md`](results/tables.md); machine-readable results are in
`results/metrics.json`.

#### Notes on the systems

- **heartKIT.**
  - The published TFLite files apply softmax twice. heartKIT's own demo
    adds a third, which with its 0.5 threshold labels every beat normal.
    We undo the extra softmax (exactly invertible; see
    `ai/heartkit/hk.py`) and then use heartKIT's thresholds.
  - The rhythm model was trained on PTB-XL + LSAD (12-lead), not
    Icentia11k.
  - The S3 "latest" configuration of BEAT-3-EFF-SM is named
    `beat-3-eff-lg`. We used the `beat-3-eff-sm/latest` files as published
    (SHA-256 sums are printed by `download_models.py`).
- **ECGFounder.**
  - It runs on the INCART lead II here, to match the other systems. On
    lead I (its training lead) INCART results are slightly worse: PVC
    AUROC 0.972 vs 0.985, AF Se 0.68 vs 0.99.
  - Window thresholds are max-F1 on DS1. For events, consecutive flagged
    windows are merged.
- **Rules.** Within one or two runs, beat-level numbers match
  `analysis/VALIDATION.md` (DS2: 16/20 V runs, 8.0 false per hour; SVDB:
  44/88, 1.7 per hour). The small differences come from INCART lead
  handling and code version.

### Reproduce

```bash
cd ai/ecgfounder && ../.venv/bin/python eval.py --with-rules           # ECGFounder, MIT-BIH + SVDB (+ INCART lead I)
../.venv/bin/python eval.py --db incartdb --incart-lead II --tag fp32_leadII
cd ../heartkit && ../.venv/bin/python download_models.py
cd ../benchmark && ../.venv/bin/python benchmark.py                     # -> results/tables.md, metrics.json
```

Data are fetched with `analysis/tools/fetch_physionet.py` into `data/`
(git-ignored), and caches go to `data/work/`. No weights or data are committed.
