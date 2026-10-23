# Rhythm transformer plan: all open data, long context, GPU

> **Not a medical device.** A research screening aid; every flagged strip is read
> by a cardiologist. Sensitivity first, then fewer false alarms.

Status 2026-10-23. Companion to [`FINETUNING_PLAN.md`](FINETUNING_PLAN.md) (the
Euler fine-tuning of the beat model, ECGFounder and CLEF, running now) and to the
dataset survey in `reports/open-ecg-datasets.html` (local).

## 1. Goal

A DeepRhythmAI-style model of our own: a CNN that reads the waveform plus a
transformer that sees 30–60 s of context, trained on every open dataset we can
use, then added to the router as one more specialist. DeepRhythmAI's architecture
is proprietary (Johnson et al., *Nat Med* 2025: CNN + transformer hybrid, a
Vision-Transformer-style model, a dedicated asystole classifier, 7 models, 300 Hz),
so this is the same family, not a copy.

| on the 344 held-out benchmark patients | router now | target |
|---|---:|---:|
| all critical findings, per-patient sensitivity | 96.4 % | ≥ 98.6 % |
| VT / NSVT | 91 % | ≥ 98 % |
| pause > 3 s | 64 % | ≥ 95 % |
| false alarms per day | ~63 | < 5, then < 1 |

## 2. Data

Every source is converted the same way: one lead, H10-like 130 Hz, every 130 Hz
frame (`h10.convert_frames`), nothing added to the signal. The benchmark's
held-out patients are never used for training.

| source | where it comes from | role | labels used |
|---|---|---|---|
| Icentia11k (11,000 patients, single-lead patch) | AWS mirror (fast) | pretraining backbone + beats + AF | N/S/V/Q beats, AF/flutter |
| LTAFDB, CPSC2021, SHDB-AF tuning patients, AFDB | AWS mirror | supervised | beats, AF, SVT codes |
| MIT-BIH, SVDB, INCART | AWS mirror | supervised (beats, V runs) | beats |
| VFDB, CUDB, SDDB | AWS mirror | supervised (VT/VF episodes) | rhythm episodes |
| **Long-Term ST** (86 × 21–24 h) | physionet.org only → VPS download | supervised (beats) + long context | beat by beat, signal quality |
| **European ST-T** (90 × 2 h) | AWS mirror | supervised (beats) | beat by beat by two cardiologists |
| **VTaC** (5,037 ICU VT alarms) | physionet.org only → VPS download | VT positives (1,441) and **hard negatives** (3,596 false VT) | true/false per alarm |
| **Challenge 2015** (750 ICU alarms) | physionet.org only → VPS download | asystole, brady, tachy, VT, VF; true/false | alarm type + true/false |
| **BUT QDB** (18 wearable ≥ 24 h) | AWS mirror | noise class, free-living wearable | signal-quality classes |
| **IRIDIA-AF** (167 × 19–95 h) | Zenodo | AF + long ambulatory negatives | AF episodes |
| **SPH**, **Chapman–Shaoxing–Ningbo** (China, 12-lead 10–60 s) | figshare / AWS mirror | AV block, bigeminy, junctional, SVT | diagnostic statements |
| PTB-XL, CinC 2017, CPSC2020 | AWS / Aliyun | supervised | as now |
| Holtercare-23K (788 Holters, VT 94, asystole 44) | not released | **ask the authors** | 18 beat classes, events |

Splits for the new sources: by patient, 80/10/10, fixed before training (hash of
`source:patient`), so each new source also gives a small held-out test of its own
(e.g. VTaC test patients for true-vs-false VT).

**Downloads.** physionet.org limits each connection to about 9 KB/s (measured from
the Mac, Euler and the VPS, 2026-10-23); the VPS fetches VTaC, Challenge 2015 and
Long-Term ST in the background (resumable, `~/open-holter-data/fetch.sh`). Every
other source is on a fast mirror and is streamed during preprocessing.

## 3. Long-segment examples

The current shards hold 10 s windows and 5 s beats. The transformer needs
**60 s segments** at the source rate (+ 2 s margins), with per-sample masks for
beats (class and position), rhythm episodes, unreadable spans and, for the alarm
datasets, the alarm label and onset. Per Icentia11k patient: every segment that
contains an abnormal beat or rhythm, plus 20 random others (cap 60 per patient).
Rough estimate: up to ~400,000 segments (~6,700 h), ~12–20 GB as int16 at the
source rate plus masks. Built by an
extended `prep.py` (new `--kind seg`), on Euler (free) or on the GPU machine.

## 4. Model

```
60 s at 130 Hz (7,800 samples), every frame, one lead
        ▼
CNN stem: 4 strided conv blocks, ×16 downsampling → ~490 tokens (≈ 0.12 s each)
        ▼
transformer encoder: 8–12 layers, width 384–512, relative position (RoPE)
        ▼
heads
 ├─ per token:   QRS here? · beat class N / S / V / noise (read at detected beats)
 ├─ per token:   unreadable (noise) mask
 └─ per segment: AF · flutter · SVT · VT run · pause/asystole · brady · tachy · AV block
```

About 20–40 M weights. Design choices to settle by measurement on the GPU (first
hour): width/depth, 30 s vs 60 s context, token size.

## 5. Training

1. **Self-supervised pretraining** on all long recordings (Icentia11k, LTST, IRIDIA-AF,
   BUT QDB, …): hide 40 % of the tokens and reconstruct them. Uses every hour,
   labelled or not (DeepRhythmAI's recipe: pretrain on many strips, then fine-tune).
2. **Supervised multitask fine-tuning** on all labelled sources, recall-first
   weights, every frame.
3. **Hard-negative rounds:** run the model and the router on the training patients,
   add every false alarm as a negative, train again (twice).
4. Checkpoints chosen by the same rule as on Euler: train until the validation score
   (sensitivity at a fixed false-alarm rate) stops improving.

## 6. Compute

The owner set no cost limit. One **H100 80 GB** (RunPod, ~USD 2.7/h on 2026-10-22)
for up to 3–4 days ≈ USD 200–260, or an A100 80 GB at about half the speed and price.
Transformers use GPU tensor cores well, unlike ECGFounder's grouped convolutions.
Throughput is **not measured yet**: the first hour on the GPU measures it and sizes
the model before the long runs.

## 7. Evaluation

The same 344 held-out patients and the same metrics as `benchmark/RESULTS.md`, then
router v2 with the transformer as one more specialist, then the per-label report
(`reports/model-scores-by-label.html`) gets its columns. Finally: the owner's own H10
recordings with a cardiologist's reads.

## 8. Order of work

| step | where | status |
|---|---|---|
| Euler fine-tuning baselines (beat model, ECGFounder, CLEF) | Euler | running |
| slow PhysioNet downloads (VTaC, Challenge 2015, LTST) | VPS | running |
| readers + label maps for every new source | code (CPU tests) | next |
| `prep.py --kind seg` (60 s segments) | code, then Euler or GPU | next |
| model + pretraining + fine-tuning code, smoke test on CPU | code | next |
| GPU: measure, pretrain, fine-tune, hard negatives | rented H100 | after the above |
| benchmark, router v2, per-label report | CPU | last |

## 9. Licences

VTaC is CC BY-SA 4.0 and Icentia11k CC BY-SA 4.0: trained weights are shared, if
ever, under CC BY-SA 4.0. The others are CC BY 4.0 or ODC-By (attribution).
CPSC2020 has no licence file (see `finetune/README.md`).
