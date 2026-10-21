# ai/ecgfounder – ECGFounder 1-lead model as a window-level flagger

Wraps the open **ECGFounder** foundation model (1-lead checkpoint, 150 labels)
so it can score Polar H10 recordings in sliding 10 s windows, and measures how
well it does on H10-like versions of three annotated PhysioNet databases.
Results are in [RESULTS.md](RESULTS.md); the comparison with the rule
pipeline and heartKIT is in [../benchmark/RESULTS.md](../benchmark/RESULTS.md).

> Not a medical device. Outputs are screening flags for a cardiologist to
> review, never a diagnosis.

## Setup

```bash
cd ai/ecgfounder
python -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
../.venv/bin/python download_weights.py          # 370 MB -> weights/ (git-ignored)
../.venv/bin/python -m pytest tests/             # preprocessing parity, ground-truth logic
```

## Use

```bash
# ECGLogger CSV (130 Hz) -> one row per 10 s window (5 s step)
../.venv/bin/python infer.py ../../data/recordings/day1.csv -o ../../reports/day1_ai.csv
```

Columns: `t_start_s` and one probability per label group (`pvc`, `pac`, `vt`,
`wide_qrs_tachy`, `svt`, `afib`, `aflutter`, `sinus_brady`, `sinus_tachy`,
`pause`, `av_block`, …; see `GROUPS` in `infer.py`). `--all-labels` adds the
150 raw labels.

## Evaluate and export

```bash
../.venv/bin/python eval.py --with-rules          # MIT-BIH, SVDB, INCART (+ rule pipeline on the same windows)
../.venv/bin/python export_onnx.py                # ONNX fp32 + int8 dynamic/static, parity, CPU latency
../.venv/bin/python eval.py --with-rules --backend onnx --onnx weights/ecgfounder_1lead_int8_static.onnx --tag int8_static
```

Each run writes `results/<tag>/tables.md` and `metrics.json`; per-window
probabilities are cached in `data/work/ecgfounder/<tag>/` (git-ignored).

## Files

| file | what |
|---|---|
| `infer.py` | resample → 10 s windows → ECGFounder preprocessing → probabilities |
| `eval.py` | H10 simulation (→130 Hz→500 Hz), window ground truth, AUROC / Se / PPV |
| `export_onnx.py` | ONNX export (with a fix for an onnxruntime Pad/MaxPool fusion bug), int8 quantisation, size, parity, latency |
| `download_weights.py` | fetches `1_lead_ECGFounder.pth` from Hugging Face |
| `third_party/` | ECGFounder `net1d.py`, `tasks.txt` and licence (MIT) |

## Credit and licence

Model code and weights: ECGFounder, © 2025 PKUDigitalHealth, MIT licence
(`third_party/LICENSE.ECGFounder`), <https://github.com/PKUDigitalHealth/ECGFounder>,
<https://huggingface.co/PKUDigitalHealth/ECGFounder>. Please cite:

> Li J, Aguirre AD, Moura J, Jin J, Liu C, Zhong L, Sun C, Clifford G,
> Westover MB, Hong S. *An Electrocardiogram Foundation Model Built on over
> 10 Million Recordings.* NEJM AI 2025;2(7):AIoa2401033.

The evaluation databases (MIT-BIH Arrhythmia, MIT-BIH Supraventricular
Arrhythmia, St Petersburg INCART) are from PhysioNet (Goldberger et al.,
Circulation 2000) under their own licences; they are downloaded on demand and
never committed.
