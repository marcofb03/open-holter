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

## How it fits together

```
Polar H10 ──Bluetooth──► Android app (records 24/7, marker button)
                              │
                              ▼
                     analysis (beats, extra beats, runs, pauses, rate)
                              │
                              ▼
                   HTML/PDF report with ECG strips ──► cardiologist
```

## Status

| Part | State |
|---|---|
| `analysis/` – Python analysis of ECGLogger CSV files | first version, tested on 3 MIT-BIH records |
| `android/` – recorder app (Kotlin + Polar BLE SDK) | planned |
| AI beat classifier (trained on open data) | planned |

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

MIT. Third-party components keep their own licences (e.g. Polar BLE SDK,
NeuroKit2).
