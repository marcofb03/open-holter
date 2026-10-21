# ai/heartkit – Ambiq heartKIT beat and AF models

Runs two pre-trained models from Ambiq's open [heartKIT](https://github.com/AmbiqAI/heartkit)
(BSD-3-Clause) on our H10-like single-lead signal:

| model | task | input | size | trained on |
|---|---|---|---:|---|
| BEAT-3-EFF-SM | one beat: normal / PAC / PVC | 512 samples at 100 Hz, centred on the beat | 106 kB TFLite | Icentia11k (single-lead chest patch) |
| ARR-2-EFF-SM | 5.12 s frame: normal vs AF/AFL | 512 samples at 100 Hz | 64 kB TFLite (int8) | PTB-XL + LSAD (12-lead hospital ECGs) |

Preprocessing follows heartKIT's own code exactly: FFT resampling of a
centred frame, no filter, per-frame layer norm with ε = 0.01. The thresholds
are heartKIT's (0.5 per beat, 0.75 for AF). See the docstring of `hk.py`.

**One correction is needed.** The published `.tflite` files end in a
softmax that has been applied twice. heartKIT's demo then applies a third,
which flattens every output and, with the 0.5 threshold, calls every beat
normal. `hk.py` inverts the extra softmax to recover the model's real
probabilities: rows become valid distributions, e.g. a PVC reads
[0.00 0.22 0.78] instead of [0.23 0.28 0.49].

```bash
cd ai/heartkit
../.venv/bin/pip install ai-edge-litert       # TFLite runtime (or full tensorflow)
../.venv/bin/python download_models.py        # -> weights/ (git-ignored), prints SHA-256
```

```python
import hk
labels, probs = hk.classify_beats(ecg_130hz, 130, r_peak_indices)   # 'N' / 'S' / 'V'
t, p_af = hk.af_frames(ecg_130hz, 130)                              # AF probability per frame
```

The beat model needs R-peak positions; in this project they come from the
rule pipeline's detector (`analysis/ecgpipe`). How the models compare with
the rules and ECGFounder is in [../benchmark](../benchmark/RESULTS.md).

## Licence

heartKIT © 2022 Ambiq, BSD 3-Clause; see `third_party/LICENSE.heartkit`. The
models are downloaded from Ambiq's model zoo and never committed.
