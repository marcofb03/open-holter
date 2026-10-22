# open-holter

A home-made, open-source long-term ECG monitor ("Holter") built from a
**Polar H10** chest strap, an Android phone, and open analysis code.

> **Not a medical device.** This project is for personal research and
> education. It does not diagnose, treat or prevent any condition, and it is
> not certified (CE / FDA). Its job is to **record** and **flag** moments that a
> qualified cardiologist should look at. Always discuss results with a doctor.
> If you feel unwell, seek medical help immediately.

## Why

Short Holter tests (24 h – 7 days) often miss symptoms that come every few
days or weeks. Commercial long-term monitors are expensive, rarely give you
your raw data, or need a prescription. The Polar H10 (~€90) streams a clean
single-lead ECG at 130 Hz; with a dedicated phone it can record for weeks.

## Project overview

```
 Polar H10 chest strap           Android phone (android/)            Analysis (analysis/)
 1 lead, 130 Hz ECG,     BLE     records 24/7, marker button,  CSV   beats → extra beats → events
 motion, skin contact  ───────►  hourly CSV files on the phone ────► confidence, artefact checks
                                                                      │
                                                                      ▼
                                              HTML report with raw ECG strips ──► cardiologist
```

1. **Record.** The [Android app](android/README.md) streams the H10 to a
   dedicated phone and writes hourly files: ECG (ECGLogger layout), motion and
   the H10's heart rate / skin contact. You press **Mark symptom** when you feel
   something.
2. **Analyse.** The [analysis](analysis/README.md) finds every heartbeat, labels
   it (N normal, S atrial-like early, V ventricular-like, Q unreadable), groups
   them into events (couplets, runs, pauses, fast rate, your markers), and gives
   each event a confidence based on noise, motion and skin contact. It runs on a
   computer today and will move onto the phone.
3. **Review.** The report shows the **raw** ECG strip of every event, so a
   cardiologist can confirm or dismiss it quickly.
4. **AI (in progress).** Open models (ECGFounder, heartKIT, CLEF) are
   benchmarked against the rules on public cardiologist-annotated databases.
   A sensitivity-first router picks, per finding, the system or combination
   that misses the fewest patients. A fine-tuning kit for the ETH Euler
   cluster adapts the models to H10-like signals.

### Repository map

| Path | What it is |
|---|---|
| `android/` | Kotlin app (Polar BLE SDK). `core/` holds file writers and recording logic (unit-tested), `app/` the service and screens. CI builds the APK. |
| `analysis/ecgpipe/` | Python analysis: loading, beat detection and labels, events, confidence, HTML report. |
| `analysis/tests/`, `analysis/tools/` | Synthetic tests (run in CI) and validation on PhysioNet databases. |
| `analysis/VALIDATION.md` | Accuracy on held-out data, including where it is weak. |
| `ai/` | Open AI second readers (ECGFounder, heartKIT, CLEF) and the cross-system benchmark with the sensitivity-first router; results in `ai/benchmark/RESULTS.md`. |
| `ai/finetune/`, `ai/FINETUNING_PLAN.md` | Fine-tuning kit (Icentia11k and seven other sources → H10-like shards, resumable Slurm training) and the plan for Euler CPU or a rented GPU. |
| `.github/workflows/` | Automatic checks: analysis tests, AI tests and Android build. |
| `data/`, `reports/` | Local only, never committed: your recordings, downloaded datasets, generated reports. |

The file formats shared by app and analysis are the contract between the two
parts; they are documented in [android/README.md](android/README.md#files-and-format).

## Status

| Part | State |
|---|---|
| `analysis/` – Python analysis of ECGLogger CSV files | rule-based v2, validated on held-out MIT-BIH DS2, SVDB and INCART, see [analysis/VALIDATION.md](analysis/VALIDATION.md) |
| `android/` – recorder app (Kotlin + Polar BLE SDK) | first version, APK built by CI; not yet tested on a real phone, see [android/README.md](android/README.md) |
| AI second readers – `ai/` (open ECGFounder 1-lead, Ambiq heartKIT, CLEF) | long-term benchmark: 344 held-out patients, 4,925 h at 130 Hz. Router: AF 201/201 patients, frequent PVC/PAC and brady 100 %, VT 91 %, pauses 64 %; about 63 false findings per day. See [ai/benchmark/RESULTS.md](ai/benchmark/RESULTS.md) |
| AI fine-tuning (Icentia11k + 7 sources, beat model with a noise class) | kit ready, not yet run on the cluster; see [ai/FINETUNING_PLAN.md](ai/FINETUNING_PLAN.md) |

## Analysis quick start

```bash
cd analysis
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m ecgpipe /path/to/recording.csv -o ../reports/my_day
```

See [analysis/README.md](analysis/README.md) for details and validation.

## Privacy

Recordings and reports are health data. They are ignored by git
(`data/recordings/`, `reports/`, `*.csv`, `*.edf`). Never commit them.

## Licence

MIT. Third-party components keep their own licences (e.g. Polar BLE SDK).
