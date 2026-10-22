# ai/clef – CLEF-Medium single-lead foundation model

[CLEF](https://github.com/Nokia-Bell-Labs/ecg-foundation-model) (Shu et al.,
arXiv:2512.02180; Nokia Bell Labs) is a single-lead ECG encoder pre-trained
contrastively on MIMIC-IV-ECG. **CLEF-Medium** is the same Net1D backbone as
ECGFounder (30.7 M parameters) without a classification layer. A 10 s, 500 Hz
window maps to a 1,024-number embedding.

> Not a medical device. A research screening aid only.

| file | what |
|---|---|
| `clef.py` | load the checkpoint (weights only), CLEF's preprocessing (0.67–40 Hz 5th-order band-pass, z-score), batched embedder (optional bf16) |
| `head.py` | small head (1024→256→8) on frozen embeddings, trained on the benchmark's **tuning patients only**, cross-fitted in 5 patient folds |
| `download_weights.py` | `clef_medium.ckpt` from Zenodo 17572734, MD5-checked |
| `tests/` | preprocessing parity with CLEF's own code, backbone shape (no weights needed) |

```bash
../.venv/bin/python download_weights.py     # 368 MB -> weights/ (git-ignored)
cd ../benchmark && ../.venv/bin/python frontend.py -j 4    # embeddings for every benchmark window
cd ../clef && ../.venv/bin/python head.py   # -> data/work/bench2/clef_head/, ../benchmark/results/clef_head.json
```

The parity of this loader with CLEF's own `create_net1d_by_size("medium")` was
checked on the real checkpoint (max |difference| 0.0 on random input). Results
on the long-term benchmark are in [`../benchmark/RESULTS.md`](../benchmark/RESULTS.md).
Frozen, the head reaches window AUROC 0.94 for AF and 0.93 for VT, but it
never beats ECGFounder or the beat-level systems as a reader. Fine-tuning it is
part of [`../FINETUNING_PLAN.md`](../FINETUNING_PLAN.md).

**Licences.** Code: Clear BSD (Nokia Bell Labs), `third_party/LICENSE.CLEF`.
The backbone code used here is ECGFounder's `net1d.py` (MIT), which CLEF's
model file adapts. The Zenodo weights state no licence. This repository
downloads them for evaluation and never redistributes them or anything
fine-tuned from them.
