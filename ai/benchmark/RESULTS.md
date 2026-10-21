# Open single-lead arrhythmia options on H10-like data – benchmark

> **Not a medical device.** These numbers compare open research tools on public
> annotated databases. None of the systems is certified, and every flagged strip
> still needs a cardiologist.

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

## Open models vs DeepRhythmAI published figures (indicative only)

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

## Ranking

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

## Recommendation for the phone

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

## Main tables (held-out)

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

### Notes on the systems

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

## Reproduce

```bash
cd ai/ecgfounder && ../.venv/bin/python eval.py --with-rules           # ECGFounder, MIT-BIH + SVDB (+ INCART lead I)
../.venv/bin/python eval.py --db incartdb --incart-lead II --tag fp32_leadII
cd ../heartkit && ../.venv/bin/python download_models.py
cd ../benchmark && ../.venv/bin/python benchmark.py                     # -> results/tables.md, metrics.json
```

Data are fetched with `analysis/tools/fetch_physionet.py` into `data/`
(git-ignored), and caches go to `data/work/`. No weights or data are committed.
