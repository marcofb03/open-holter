# ECGFounder (1 lead) on H10-like signals – results

> **Not a medical device.** These are research numbers on public databases.

The head-to-head comparison is in [`../benchmark/RESULTS.md`](../benchmark/RESULTS.md):
the rule pipeline vs ECGFounder vs heartKIT vs ensembles, plus the comparison
with DeepRhythmAI's published figures. This page covers ECGFounder alone.

**In short.**
- ECGFounder is excellent at **AF/flutter** on H10-like signals. Held-out
  window AUROC is 0.99; at the DS1 threshold Se is 0.83 and PPV 0.97, and
  every held-out AF patient was found.
- It is good at ranking windows with **PVCs (AUROC 0.97)** and
  **PACs (0.92)**.
- It is weak at **V runs** (AUROC 0.89, but PPV 0.08 at a usable
  sensitivity) and cannot count beats.
- Static int8 ONNX keeps most of the accuracy at a quarter of the size and
  2.5× the speed.

## Setup

- Checkpoint `1_lead_ECGFounder.pth` from Hugging Face (MIT), SHA-256
  `f863a38897fb49a27fec7e44008ea3c7bdbd29c77fa4a02ecbb8c56df4f37603`.
  It is 370 MB: 30.8 M parameters plus optimiser state, stored `val_auroc`
  0.881.
- Signal path:
  - record → 130 Hz (H10 rate) → 500 Hz;
  - 10 s windows with a 5 s step, 70,767 windows from 197 records;
  - ECGFounder's own filtering and z-score (verified identical to upstream);
  - sigmoid.
- Leads:
  - MIT-BIH MLII, SVDB ECG1, INCART lead I (the model's training lead);
  - INCART was also run on lead II, see below.
- Ground truth per window comes from the cardiologists' beat and rhythm
  annotations; definitions are in `eval.py`. Edge beats, partial runs,
  partial AF overlap and VF episodes are left out.
- Thresholds are max-F1 on **MIT-BIH DS1**. **DS2, SVDB and INCART are
  held out.**
- CIs are a 200× bootstrap over records, because windows overlap.

## Accuracy (fp32)

AUROC [95 % CI], then Se / PPV at the DS1 threshold. *pos* is positive
windows, with records in brackets.

| target (label group) | DS1 (dev) | DS2 | SVDB | INCART (lead I) | **held-out pooled** |
|---|---|---|---|---|---|
| ≥ 1 PVC (pvc) | 0.981 [0.96–1.00] 0.86 / 0.93 | 0.987 [0.97–1.00] 0.89 / 0.92 | 0.950 [0.92–0.97] 0.78 / 0.88 | 0.972 [0.96–0.98] 0.88 / 0.93 | **0.965** [0.95–0.97] 0.84 / 0.91 · pos 21,793 (153) |
| ≥ 1 PAC/SVE (pac) | 0.929 [0.89–0.97] 0.73 / 0.79 | 0.869 [0.80–0.95] 0.44 / 0.43 | 0.926 [0.90–0.95] 0.82 / 0.78 | 0.964 [0.93–0.98] 0.92 / 0.32 | **0.922** [0.89–0.95] 0.81 / 0.60 · pos 11,014 (125) |
| V run ≥ 3 (VT or wide-QRS tachy) | 0.949 [0.91–0.98] 0.43 / 0.24 | 0.876 [0.77–0.97] 0.25 / 0.03 | 0.868 [0.81–0.93] 0.30 / 0.06 | 0.897 [0.86–0.94] 0.45 / 0.12 | **0.888** [0.86–0.92] 0.39 / 0.08 · pos 530 (54) |
| SVT: S run ≥ 3 or SVTA (svt) | 0.933 [0.72–0.99] 0.49 / 0.62 | 0.957 [0.92–0.99] 0.03 / 0.25 | 0.922 [0.84–0.96] 0.49 / 0.28 | 0.789 [0.69–0.89] 0.00 / 0.00 | **0.912** [0.86–0.95] 0.28 / 0.13 · pos 807 (57) |
| AF/AFL ≥ 50 % of window (afib, aflutter) | 1.000 0.99 / 0.96 | 0.997 [0.99–1.00] 0.93 / 0.96 | n/a¹ | 0.995 [0.98–1.00] 0.68 / 1.00 | **0.991** [0.97–1.00] 0.83 / 0.97 · pos 1,966 (8) |
| pause: RR > 2 s (pause, SA/AV block) | 0.968 [0.75–0.99] 0.50 / 0.15 | 0.965 [0.92–0.98] 0.84 / 0.22 | – | – | **0.985** [0.97–0.99] 0.84 / 0.08 · pos 141 (2) |
| sinus bradycardia (< 60 bpm) | 1.000 0.99 / 0.99 | 1.000 0.99 / 0.98 | n/a¹ | n/a¹ | **1.000** 0.99 / 0.98 |
| sinus tachycardia (> 100 bpm) | 1.000 1.00 / 1.00 | 1.000 0.98 / 1.00 | n/a¹ | n/a¹ | **1.000** 0.98 / 1.00 |
| sinus rhythm vs non-sinus | 0.819 | 0.774 | n/a¹ | n/a¹ | 0.761² |

¹ SVDB has no rhythm annotations in its `.atr` files. Only 6 INCART
records do (AF in 3, counting WPW-AF).
² At max F1 the sinus label flags every window (Sp 0); only the AUROC is
informative.

**INCART lead I vs lead II** (AUROC, Se / PPV):

| lead | PVC | PAC | V run | AF |
|---|---|---|---|---|
| I | 0.972, 0.88 / 0.93 | 0.964, 0.92 / 0.32 | 0.897, 0.45 / 0.12 | 0.995, 0.68 / 1.00 |
| II | 0.985, 0.95 / 0.93 | 0.985, 0.80 / 0.87 | 0.920, 0.64 / 0.16 | 1.000, 0.99 / 0.99 |

The model, trained on lead I, did better on lead II here. A chest strap gives
yet another lead, so per-lead differences of this size should be expected on
the H10.

**ECGFounder vs the rule pipeline on the same windows** (held-out, Se / PPV;
from `eval.py --with-rules`):

| target | rules | ECGFounder | rules AND model | rules OR model |
|---|---|---|---|---|
| ≥ 1 V | 0.92 / 0.78 | 0.84 / 0.91 | 0.80 / 0.95 | 0.96 / 0.76 |
| V run ≥ 3 | 0.54 / 0.09 | 0.39 / 0.08 | 0.23 / 0.31 | 0.69 / 0.07 |
| ≥ 1 S | 0.69 / 0.65 | 0.81 / 0.60 | 0.62 / 0.82 | 0.88 / 0.54 |
| AF | 0.36 / 0.55 | 0.83 / 0.97 | 0.33 / 0.97 | 0.85 / 0.74 |

For AF the model adds the most value. For V and S, AND with the rules
mainly buys precision. The full benchmark, which adds heartKIT, is in
`../benchmark/RESULTS.md`.

## int8 static ONNX vs fp32

The int8 model is calibrated on 64 DS1 windows, and thresholds are re-chosen
on DS1. AUROC on held-out data, pooled:

| target | fp32 | int8 static | Δ |
|---|---:|---:|---:|
| ≥ 1 PVC | 0.965 | 0.957 | −0.008 |
| ≥ 1 PAC | 0.922 | 0.901 | −0.021 |
| V run ≥ 3 | 0.888 | 0.850 | −0.038 |
| SVT | 0.912 | 0.853 | −0.059 |
| AF | 0.991 | 0.985 | −0.006 |
| pause | 0.985 | 0.950 | −0.035 |

AF and PVC barely move. V-run and SVT lose 0.04–0.06 AUROC. Per window,
the probabilities shift by 0.035 on average and by up to 0.76. Use int8
for AF and PVC screening, or as a second reader on flagged windows. Keep
fp32 if V-run ranking matters.

## Phone feasibility

Measured with `export_onnx.py` on real weights (Intel Xeon 2.1 GHz cloud
vCPU, onnxruntime 1.31). Output: `results/fp32/phone.md`.

| format | size | max abs Δp vs torch | 1 thread, batch 1 | 4 threads, batch 32 |
|---|---:|---:|---:|---:|
| torch fp32 | 370 MB ckpt | – | 62 ms | 11 ms |
| ONNX fp32 | 123 MB | < 1e-4 | 48 ms | 15 ms |
| ONNX int8 dynamic | 31 MB | 0.76 | 159 ms (slower) | 71 ms |
| **ONNX int8 static** | **32 MB** | 0.76 | **19 ms** | 9 ms |

Preprocessing (filters, median baseline, z-score) adds about 11 ms per
window. The model needs 1.14 GMAC per 10 s window.

**Export bug found and fixed.** net1d zero-pads before MaxPool, and
onnxruntime's optimiser folds that Pad into MaxPool's own `pads`. Those pad
with −∞ instead of 0, which changed logits by up to 0.8 (probability by up
to 0.17) as soon as a sequence length became odd. `export_onnx.py` now
writes the pad as a Concat of zeros, and ONNX fp32 matches torch to 1e-5
with all optimisations on.

**Pixel 8a (Tensor G3), estimate, not measured.** Assume a Cortex-A715
core is 2–4× slower than this vCPU. Then static int8 costs about 40–80 ms
plus 20–40 ms of preprocessing per window. That makes **9–17 CPU-minutes a
day** for 8,640 windows, or 17–35 for 17,280 with the 5 s step. Running
every window is **realistic** in batches while charging. Run only on
windows other detectors flagged, a few hundred a day, it costs seconds.
Measure on the phone before relying on this; the onnxruntime XNNPACK
execution provider may be faster.

## Honest limits

- **Domain shift.**
  - The model learned from hospital 12-lead resting ECGs and saw lead I
    only. The H10 is an ambulatory chest-strap lead at 130 Hz.
  - Our simulation reproduces the 130 Hz bandwidth, but not strap noise,
    motion or electrode contact.
  - Results vary by lead (INCART I vs II above).
- **Window labels, not beats.** The model says "this window has PVCs"; it
  cannot count them, place them, or measure burden.
- **One lead.** PAC vs sinus arrhythmia, aberrancy vs PVC, and AV block are
  hard. The pause/AV-block labels are not usable (PPV 0.08).
- **Evaluation data.**
  - The databases are small and arrhythmia-rich: AF is in 8 held-out
    patients, pauses in 2.
  - Windows overlap, so the CIs are bootstrapped over records.
  - Thresholds from DS1 are not tuned for a healthy wearer's base rate.
- **Not certified.** This is not a medical device, and the model has not
  been validated on real H10 recordings against a cardiologist.
- **Licences.**
  - ECGFounder code and weights: MIT (notice in `third_party/`).
  - Icentia11k, for any fine-tuning: CC BY-NC-SA 4.0, so a model trained
    on it is for non-commercial use.

## Recommendation

- **Use ECGFounder as a second reader on flagged windows, not as the
  primary detector.**
  - Its value is AF/flutter confirmation (PPV 0.97), and naming the rhythm
    on windows the rules or heartKIT flagged.
  - The rules plus heartKIT should do the beat counting and V-run
    detection. See `../benchmark/RESULTS.md` for the combined system and
    its numbers.
- **Fine-tuning on Icentia11k** (single chest lead, beat labels) is the next
  step only if window-level V-run detection from ECGFounder is wanted.
  Today the beat-level path (rules + heartKIT) is better for V runs, with
  per-patient Se 81–83 % vs 56 %.

## Reproduce

```bash
cd ai/ecgfounder
python -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
../.venv/bin/python download_weights.py
../.venv/bin/python eval.py --with-rules                                  # results/fp32/
../.venv/bin/python eval.py --db incartdb --incart-lead II --tag fp32_leadII
../.venv/bin/python export_onnx.py                                        # results/fp32/phone.md
../.venv/bin/python eval.py --with-rules --backend onnx --onnx weights/ecgfounder_1lead_int8_static.onnx --tag int8_static
```

## Citation

Li J, Aguirre AD, Moura J, Jin J, Liu C, Zhong L, Sun C, Clifford G,
Westover MB, Hong S. *An Electrocardiogram Foundation Model Built on over 10
Million Recordings.* NEJM AI 2025;2(7):AIoa2401033. Code and weights:
<https://github.com/PKUDigitalHealth/ECGFounder> (MIT).
