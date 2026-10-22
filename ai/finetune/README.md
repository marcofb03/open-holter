# ai/finetune – fine-tuning kit (ETH Euler CPU, or a rented GPU)

Fine-tunes three models for H10-like single-lead ECG with a recall-first
objective. The why and how long, the costs and the recommended path are in
[`../FINETUNING_PLAN.md`](../FINETUNING_PLAN.md).

| model | what | input |
|---|---|---|
| `ecgfounder` | ECGFounder 1-lead, new 9-target head | 10 s, 130 Hz → 500 Hz |
| `clef` | CLEF-Medium + 9-target head | 10 s, 130 Hz → 500 Hz |
| `beat` | heartKIT-style MBConv beat classifier (0.43 M parameters), classes N / S / V / **noise** | 4.9 s beat-centred, 130 Hz |

The 9 window targets are the benchmark's findings (`vt af svt pvc pac pause
brady tachy`) plus `noise`.

> Not a medical device. Every model trained here is a research screening aid
> whose flags a cardiologist reads.

## Files

| file | what |
|---|---|
| `h10.py` | H10-like conversion (130 Hz, strap-like front end, int16 µV), labels, examples, augmentation |
| `sources.py` | patient lists, patient-level splits, streaming readers for every source |
| `prep.py` | one preprocessing array task → a few large HDF5 shards (idempotent) |
| `shards.py` | shard writer, resumable chunk reader, NSTDB noise |
| `models.py` | the three models and their input preprocessing |
| `train.py` | resumable trainer: checkpoints every 15 min and on SIGUSR1, auto-resume, DONE marker |
| `measure_step.py` | time one training step per model (CPU or GPU) |
| `eval_beat.py` | a trained beat model on the benchmark, scored exactly like heartKIT (augmentation ablation) |
| `slurm/` | `setup_euler.sh`, `submit_prep.sh` + `prep.sbatch`, `submit_chain.sh` + `train.sbatch`, `env.sh` |

## Euler in five commands

```bash
git clone https://github.com/marcofb03/open-holter ~/open-holter
bash ~/open-holter/ai/finetune/slurm/setup_euler.sh          # venv, CPU torch, weights (~10 min)
cd /cluster/scratch/$USER/open-holter
bash ~/open-holter/ai/finetune/slurm/submit_prep.sh           # all preprocessing arrays
bash ~/open-holter/ai/finetune/slurm/submit_chain.sh beat 2   # 2 x 24 h, resumes by itself
```

Ablation of the augmentation (same number of epochs, then compare on the benchmark):

```bash
bash ~/open-holter/ai/finetune/slurm/submit_chain.sh beat 2 24:00:00 beat_aug
bash ~/open-holter/ai/finetune/slurm/submit_chain.sh beat 2 24:00:00 beat_noaug --no-augment
# afterwards, wherever the benchmark caches are (ai/benchmark/frontend.py):
python eval_beat.py runs/beat_aug runs/beat_noaug       # -> results/ablation.md
```

Shards, runs and logs live in `/cluster/scratch/$USER/open-holter/`, which is
purged after 15 days: copy `runs/*/best.pt` and the results off it (see the
plan). Never commit shards, checkpoints or weights.

## Data licences (and what that means for trained weights)

| source | licence | used for |
|---|---|---|
| Icentia11k | CC BY-SA 4.0 | beats + windows (patients 0–9,999 train/val; 10,000–10,999 test) |
| PTB-XL 1.0.3 | CC BY 4.0 | windows (leads I and II) |
| CinC 2017 | ODC-By 1.0 | windows (AF / noise) |
| CPSC2021 | CC BY 4.0 | windows (AF), benchmark split |
| LTAFDB, AFDB, NSTDB | ODC-By 1.0 | windows, beats; NSTDB noise for augmentation |
| SHDB-AF | ODC-By 1.0 | windows (AF, AT/SVT), benchmark split |
| CPSC2020 | challenge data, no licence file (ICBEB) | beats + windows; check terms before redistributing |

Weights trained on Icentia11k are treated as adapted material under
**CC BY-SA 4.0**: if they are ever shared, share them under CC BY-SA 4.0 with
attribution (Tan et al., Icentia11k, PhysioNet). The code stays MIT. The CLEF
checkpoint on Zenodo states no licence, so a fine-tuned CLEF stays private.
ECGFounder is MIT, heartKIT BSD-3.
