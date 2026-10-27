# ai/rhythm – rhythm transformer kit

A CNN + transformer that reads **60 s** of H10-like ECG and outputs, per ~0.12 s token, QRS
presence and the beat class (N / S / V / unreadable), and per 10 s the ten targets `vt af svt
pvc pac pause brady tachy noise avb`. Why and how: [`../TRANSFORMER_PLAN.md`](../TRANSFORMER_PLAN.md).

> Not a medical device. A research screening aid whose flags a cardiologist reads.

| file | what |
|---|---|
| `extra_sources.py` | readers and label maps for the new sources: Long-Term ST, European ST-T, SDDB, VTaC, Challenge 2015, BUT QDB, IRIDIA-AF, SPH, Chapman–Shaoxing–Ningbo |
| `seg.py` | 60 s segment examples at the source rate (+ 1 s margins), labels per 10 s, beats, validity; `to_input` (z-score + validity mask) |
| `prep_seg.py` | one array task → segment shards per split (all fine-tuning sources + the new ones), `manifest` |
| `fetch_bulk.py` | one-time download + unpack of IRIDIA-AF (Zenodo) and SPH (figshare) |
| `model.py` | `RhythmNet`: 4 strided convolutions (×16) → 488 tokens → pre-norm transformer → qrs / beat / win / recon heads; sizes tiny … large |
| `train_rhythm.py` | `--stage pretrain` (masked reconstruction) and `--stage finetune` (multitask, recall first); every 130 Hz frame; train until the validation score plateaus; resumable; GPU or CPU |
| `tests/` | parsers, segment labels, model shapes, and an end-to-end CPU run (pretrain → finetune) on synthetic shards |

Local copies (physionet.org-only sources and the two archives) live in `$OH_LOCAL_DATA`
(default `data/`): `vtac/`, `challenge-2015/`, `ltstdb/`, `iridia-af/`, `sph/`.

```bash
python fetch_bulk.py iridia-af sph
python prep_seg.py icentia11k --task-id 0 --n-tasks 300 --out $SEG --tmp $TMPDIR   # one array task
python prep_seg.py manifest --out $SEG
python train_rhythm.py --stage pretrain --shards $SEG --run runs/rt_pre --device cuda --bf16
python train_rhythm.py --stage finetune --shards $SEG --run runs/rt_ft --device cuda --bf16 --init runs/rt_pre/best.pt
```

Licences of what comes out: VTaC and Icentia11k are CC BY-SA 4.0, so trained weights are
shared, if ever, under CC BY-SA 4.0; the other sources are CC BY 4.0, CC0 or ODC-By.
