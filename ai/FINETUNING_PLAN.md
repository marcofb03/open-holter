# Fine-tuning plan (ETH Euler CPU, optional rented GPU)

> **Not a medical device.** Fine-tuned models are research screening aids;
> every strip they flag is read by a cardiologist. The owner prefers more
> false alarms to a missed event, so every choice below is **recall first**.

What gets trained, with the kit in [`finetune/`](finetune/README.md):

| # | model | starts from | learns | why |
|---|---|---|---|---|
| 1 | **ECGFounder 1-lead** | MIT weights (HEEDB, 10 M ECGs) | 9 window targets on H10-like 130 Hz data | best AF and PVC/PAC window ranking in the benchmark, but never saw a strap or 130 Hz; weak on V runs |
| 2 | **CLEF-Medium** | Zenodo weights (MIMIC-IV-ECG) | same 9 targets | single-lead by design; frozen + small head is weak here (see `benchmark/RESULTS.md`), so it needs real fine-tuning |
| 3 | **heartKIT-style beat model** | scratch | N / S / V / **noise** per beat at 130 Hz | the benchmark's largest error is false V runs in noise; an explicit noise class, learned from real unreadable stretches on every 130 Hz frame, targets exactly that |

The 9 window targets are the benchmark findings (`vt af svt pvc pac pause
brady tachy`, `benchmark/findings.py`) plus `noise`. All three use a
recall-first loss (positives weighted up; V and S beats weighted 2x and 1.5x)
and keep the checkpoint with the best **sensitivity at a fixed false-positive
rate** on the validation patients, not the best accuracy.

## 1. Verified limits this plan is written for

| resource | limit (verified 2026-10-21) | consequence |
|---|---|---|
| Euler account | GUEST on the public share, **no GPU** | CPU-only on Euler |
| job shape | ≤ 48 CPUs; `--cpus-per-task=48 --mem-per-cpu=2G` or `42 x 3G` | `train.sbatch` asks for 48 x 2 GB |
| queue | a 48-core job started 2.5 min after submission; 24 h jobs usually start in 10–30 min | **24 h jobs are the default**; `submit_chain.sh ... 04:00:00` gives the ≤ 4 h variant |
| node | AMD EPYC 9654 (Zen 4, AVX-512 incl. BF16; no AMX), 48 cores allocated | timings below are extrapolated to it |
| scratch | `/cluster/scratch/mfiorani/open-holter/`, ~2.1 TB free of 2.5 TB, **≤ 1,000,000 files**, files **purged after 15 days** | few large shards; copy results off (section 8) |
| Icentia11k | 11,000 patients x 151 files = 1.66 M files, 1.1 TB | never stored raw: streamed patient by patient through node-local `$TMPDIR` |
| internet from jobs | compute nodes reach outside ETH **only via `module load eth_proxy`**; the proxy has limited capacity, and overuse blocks the external site for all Euler users ([docs](https://docs.hpc.ethz.ch/connections/firewall/#eth-proxy), checked 2026-10-22) | `env.sh` loads `eth_proxy`; downloads are throttled to ≤ 32 at a time and the arrays run one after another (section 3) |
| also | Contabo VPS (8 vCPU, 23 GB, ~170 GB free), up to ~USD 50 of rented GPU | the VPS receives the results; the GPU path is section 6 |

## 2. Data: from 1.7 M raw files to ~1,100 shards

`finetune/prep.py <source> --task-id i --n-tasks n` handles one contiguous
block of patients. For each patient it downloads the raw files into `$TMPDIR`
(Slurm `--tmp`), builds the examples, deletes the raw files, and at the end
writes one HDF5 shard per split (`shards/<source>/<split>/task-i-of-n.h5`,
written to `.tmp` and renamed when complete). A re-submitted task whose shards
exist exits in seconds, so a failed array is fixed by submitting it again.

| source | patients | split (patient level) | tasks | shards | size (est.) |
|---|---:|---|---:|---:|---:|
| Icentia11k | 11,000 | 0–9,999 train, every 20th val; **10,000–10,999 test** (heartKIT's test patients; the benchmark's Icentia sample comes from them) | 300 | 900 | ~77 GB (7 MB / patient measured) |
| PTB-XL (leads I, II) | 18,869 | strat_fold 1–8 / 9 / 10 | 20 | 60 | ~0.3 GB |
| CinC 2017 | 8,528 records | 80/10/10 by record hash | 4 | 12 | ~0.2 GB |
| CPSC2020 | 10 | A01–A08 / A09 / A10 | 10 | 30 | ~0.5 GB |
| LTAFDB | 84 | benchmark tuning share → train/val, benchmark held-out → **test** | 12 | 36 | ~1.2 GB |
| CPSC2021 | 105 | same rule | 8 | 24 | ~0.5 GB |
| SHDB-AF | 98 | all benchmark held-out → **test** (use `--policy all` only for a final model) | 16 | 48 | ~1 GB |
| AFDB | 23 | all benchmark held-out → test | 4 | 12 | ~0.1 GB |
| NSTDB noise | – | augmentation only | 1 | 1 | 3 MB |
| **total** | | | **375** | **~1,120 files** | **~80 GB** |

With logs (~400) the kit stays near **1,500 files**, about 0.15 % of the quota,
and under 5 % of the space.

**The split protects the benchmark.** By default (`--policy benchmark`) no
patient of the benchmark's held-out split (`benchmark/datasets.py`) is used for
training or validation, so `benchmark/` remains a fair test of the fine-tuned
models. `--policy all` reshuffles 80/10/10 for a final model trained after
evaluation.

**Examples per patient** (budget, `prep.py BUDGET`): every 10 s window with a
positive target (Icentia11k: up to 1,500) plus 600 random others; beats: up to
1,500 N, 1,000 S, 1,000 V and 500 unreadable (Icentia's Q) per patient.
Smaller databases keep up to 4,000 positive windows and 4,000 / 3,000 / 3,000 /
1,000 beats per patient. That gives about 20 M training windows and 25 M
training beats.

### Conversion to 130 Hz: every frame, nothing added

Each example is stored at its **source rate** (250 Hz Icentia11k, 500 Hz
PTB-XL, 400 Hz CPSC2020, 128 Hz LTAFDB, ...), as unfiltered int16 µV with a
margin of 1 s (windows) or 0.75 s (beats). The 130 Hz H10-like signal is made
when the example is read: polyphase resampling to 130 Hz, then the nominal
H10-like front end (first-order 0.5 Hz high-pass, second-order 40 Hz low-pass,
causal). The front end is an approximation of the strap, not a measurement of it.

**There is no single correct 130 Hz version of a recording.** The H10 samples
the heart at moments unrelated to the beats, so the 130 Hz grid can sit at any
sub-sample position relative to the source samples, and each position (a
"frame") gives a slightly different 130 Hz signal. Starting the conversion *j*
source samples later moves the grid by *j* x 130 / fs output samples. With
130 / fs = p / q in lowest terms, *j* = 0 .. q−1 gives **every distinct frame
exactly once**, with the same polyphase filter and no extra interpolation:

| source rate | frames |
|---|---:|
| 128 Hz (LTAFDB) | 64 |
| 200 Hz (SHDB-AF, CPSC2021) | 20 |
| 250 Hz (Icentia11k, AFDB) | 25 |
| 300 Hz (CinC 2017) | 30 |
| 400 Hz (CPSC2020) | 40 |
| 500 Hz (PTB-XL) | 50 |

Training (`--conversion frames`, the default; `h10.convert_frames`): every
epoch, each example is converted on a different frame (a fixed per-example
start plus the epoch number, mod q), and each epoch uses all frames equally.
Windows also get a random crop start (a different 10 s frame of the same real
signal). Beats stay centred to within half a sample. **Nothing is added to the
signal**: no gain, polarity, baseline, filter or rate changes and no synthetic
noise. The noise class learns from the real unreadable stretches that the
databases mark (Icentia11k `Q` beats, windows with mostly unreadable beats).
Validation and test always use frame 0, which is exactly the deterministic
conversion. Cost: 0.1 ms per beat and 0.4 ms per window on one thread.

Frame 0 is as arbitrary as any other frame, so training on all frames was
adopted by the owner's decision (2026-10-22) without an ablation.
`--conversion fixed` (frame 0 only) remains available for a comparison.
`--conversion random` is the earlier synthetic augmentation (random rates
125–135 Hz and resampling methods, random filter corners, gain, polarity,
baseline wander, NSTDB noise); it is kept in the code but not used.

The source-rate shards give every frame at no extra space: storing q fixed
130 Hz copies would multiply the space by q (25–64).

## 3. Jobs

| job | script | resources | time | count |
|---|---|---|---|---|
| preprocessing | `slurm/submit_prep.sh` → `prep.sbatch` (array) | 8 CPUs x 3 GB, `--tmp=40G` | ≤ 4 h per task (measured: ~16 s of one core per Icentia11k patient, so ~10 min per 37-patient task on 8 cores) | 375 tasks; the arrays run one after another, ≤ 4 tasks at a time (`PAR`), 1 download thread per worker (`DL_THREADS`): ≤ 32 concurrent downloads through the ETH proxy |
| training | `slurm/submit_chain.sh MODEL N [TIME]` → `train.sbatch` | 48 CPUs x 2 GB | 24 h (default) or 4 h | N dependent jobs per model |

Training jobs are **resumable by construction**:

- `train.py` writes `last.pt` (model, optimiser, scheduler, data-stream position,
  step, best score) every **15 minutes** and when Slurm sends `SIGUSR1`, 10 min
  before the limit (`--signal=B:USR1@600`);
- it stops by itself 15 min before the limit;
- it resumes from `last.pt` when started again. The data stream resumes at the
  exact chunk with the same augmentation draws.
- `submit_chain.sh` queues N jobs with `--dependency=afterany`, so a job that
  hit its time limit (the normal case) hands over to the next one;
- a finished run writes `DONE`, and the remaining jobs exit in seconds.

## 4. How long: measured here, extrapolated to 48 Zen 4 cores

Measured in this cloud container: Intel Xeon @ 2.8 GHz (Sapphire Rapids
class, AMX), 4 vCPUs, torch 2.14 CPU. Each figure is one full training step
(forward, backward, AdamW update), median of 3–5 steps after warm-up, from
`finetune/measure_step.py`. Raw numbers are in `finetune/results/step_timing_local*.md`.

| model | batch | 1 thread | 2 threads | 4 threads | scaling 1→4 | bf16 (4 threads) | data pipeline, 1 thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| beat (0.43 M) | 512 beats | 114 /s | 178 /s | 285 /s | 63 % | 327 /s | 0.09 ms / beat |
| ECGFounder (30.8 M) | 64 windows | 4.3 /s | 7.4 /s | 14.5 /s | 84 % | 8.6 /s (slower) | 1.45 ms / window |
| CLEF-Medium (30.7 M) | 64 windows | 4.5 /s | 8.1 /s | 13.0 /s | 72 % | 8.0 /s (slower) | 1.22 ms / window |

The data pipeline (resampling chain, front end, augmentation, model
preprocessing) is < 1.5 ms per example on one thread. At the training rates
below it needs less than half a core, so the GPU or the convolutions are always
the bottleneck, never the shards. bf16 autocast *slowed* the two large models
here (backward pass). On Zen 4, which has AVX-512 BF16 but no AMX, it is
unknown, so `--bf16` is opt-in.

**Extrapolation to one Euler job: 48 cores of an AMD EPYC 9654.**

- Per core, Zen 4 has half the FMA width of this Xeon (512-bit ops
  double-pumped on 256-bit units) but runs at higher all-core clocks. This
  container's vCPUs may also be hyperthreads. I assume a per-thread speed of
  0.6–1.2x the measurement above.
- Parallel efficiency from 4 to 48 threads: 50–70 % for the 30 M models
  (they scaled at 84 % / 72 % to 4 threads) and 25–50 % for the small beat
  model (63 % already at 4 threads).

**Measured on Euler (2026-10-22, AMD EPYC 9654, torch 2.14.1+cpu;
`finetune/results/step_timing_euler*.md`), examples per second:**

| model | 48 threads fp32 | 48 threads bf16 | 24 threads fp32 | 24 threads bf16 | used |
|---|---:|---:|---:|---:|---|
| beat | 346 | 116 | **875** | 955 | 24 CPUs, fp32 |
| ECGFounder | 113 | 183 | 118 | **178** | 24 CPUs, bf16 |
| CLEF-Medium | 116 | **180** | – | – | 24 CPUs, bf16 (24-thread speed assumed like ECGFounder) |

The small beat model is 2.5x slower on 48 threads than on 24, and ECGFounder
gains nothing beyond 24, so every training job asks for 24 CPUs. bf16 helps
the two large models on Zen 4 (unlike the Xeon above). The estimates below
were made before this measurement and are kept for reference.

| model | 48-core estimate (range) | planning figure | per hour |
|---|---:|---:|---:|
| beat | 820–3,300 beats / s | **1,500 / s** | 5.4 M beats |
| ECGFounder | 62–173 windows / s | **100 / s** | 0.36 M windows |
| CLEF-Medium | 65–180 windows / s | **100 / s** | 0.36 M windows |

Replace these estimates with real ones in 10 minutes:
`sbatch $REPO/ai/finetune/slurm/measure_euler.sbatch`. The training logs also
print examples/s every 5 minutes.

**Epoch sizes** come from the measured shards (115 Icentia11k patients: 2,075
windows and 2,450 beats per patient, 8 MB). About **20 M training windows** and
**23.5 M training beats** per epoch.

## 5. Wall-clock on Euler (CPU path, measured speeds)

**Stopping rule: train until the validation score stops improving**, not for a
fixed number of epochs (`train.py`):

| | beat model | ECGFounder, CLEF |
|---|---|---|
| validation | every 0.25 epoch, 51,200 beats of validation patients | every 0.25 epoch, 25,600 windows |
| score (also picks `best.pt`) | mean V and S sensitivity at 1 % false positives among normal beats | mean sensitivity at 95 % specificity over the 9 targets |
| lower the LR (x 0.3) | after 4 validations (1 epoch) without a gain > 0.002 | after 2 validations (0.5 epoch) |
| stop | after 10 validations (2.5 epochs) without a gain: at least two LR reductions first | after 6 validations (1.5 epochs) |
| cap | 25 epochs (~9 days at 770 beats/s) | 6 epochs (~10 days at ~136 windows/s on Zen 4) |
| purge guard | stop 14 days after the oldest shard was written | same |

Measured speeds in the real training loop: beat ~770 beats/s (EPYC 7742, fp32,
24 CPUs); ECGFounder ~137 and CLEF ~135 windows/s (EPYC 9654, bf16, 24 CPUs).
One epoch is ~8.5 h (beat, ~23.5 M beats) and ~41 h (windows, ~20 M windows).

| step | jobs | wall-clock |
|---|---|---|
| preprocessing, all sources | 375 array tasks, 8 cores, 4 at a time, arrays one after another | ~1.5 days (Icentia11k ~29 h, download-bound) |
| beat model | 10 x 24 h, 24 CPUs, any node | until plateau, at most ~9 days |
| ECGFounder | 11 x 24 h, 24 CPUs, Zen 4 only (bf16) | until plateau, at most ~10 days |
| CLEF-Medium | 11 x 24 h, 24 CPUs, Zen 4 only (bf16) | until plateau, at most ~10 days |

The three chains wait for the preprocessing and then run side by side if the
guest share allows. A model that is still improving when it reaches its cap or
the purge deadline continues on a rented GPU (section 6) from its `last.pt`.

The ≤ 4 h variant (`submit_chain.sh MODEL N 04:00:00`) needs 6x as many jobs
and, at the observed ~10 h queue for short jobs, is slower in wall-clock. Use
it only if 24 h jobs stop starting promptly.

## 6. Rented-GPU path (RunPod or Vast.ai)

Copy **only the shards** (~85 GB, `shards/` with `manifest.json`) to the
rented machine. Run the same `train.py` with `--device cuda --bf16
--data-threads 8`. Nothing else moves: no raw data, no Icentia11k download.

```bash
# on the pod (RunPod "PyTorch" template or Vast.ai), once:
git clone https://github.com/marcofb03/open-holter && cd open-holter/ai
pip install numpy scipy pandas wfdb scikit-learn h5py bottleneck
(cd ecgfounder && python download_weights.py) && (cd clef && python download_weights.py)
# from an Euler login node (or via the VPS if outbound SSH to the pod is blocked):
rsync -av --progress /cluster/scratch/mfiorani/open-holter/shards/ root@POD:/workspace/shards/ -e "ssh -p PORT"
# on the pod: calibrate (2 min), then train
python finetune/measure_step.py --device cuda --tag gpu
python finetune/train.py --model ecgfounder --shards /workspace/shards --run /workspace/runs/ecgfounder \
    --device cuda --bf16 --data-threads 8 --epochs 3
python finetune/train.py --model clef --shards /workspace/shards --run /workspace/runs/clef \
    --device cuda --bf16 --data-threads 8 --epochs 3
```

Put the shards and runs on a network volume, so a preempted or restarted pod
resumes from `last.pt` (checkpoints every 15 min) without copying again.

**Throughput is an estimate here: no GPU was available to measure.**

- ECGFounder costs 2.3 GFLOP per window forward, about 6.8 GFLOP per training
  example.
- Its grouped 1-D convolutions use tensor cores poorly, so I assume 5–15 %
  of peak bf16.
- That gives an RTX 4090 1,000–3,000 windows/s. The planning figure is
  **1,500 /s**: 3 epochs (60 M windows) in **~11 h per model**.
- `measure_step.py --device cuda` gives the real figure in the first two
  minutes. If it is lower than planned, the table below still has room under
  USD 50.

| GPU (RunPod Community, 2026-10-22) | USD / h | planning windows / s | ECGFounder + CLEF, 3 epochs each | GPU cost | + setup, copy, calibration (3 h) | storage (150 GB, 3 days) | **total** | at half the planned speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **RTX 4090** | 0.34 | 1,500 | 22 h | 7.5 | 1.0 | 1.0 | **~10** | ~18 |
| L40S | 0.79 | 2,000 | 17 h | 13 | 2.4 | 1.0 | ~17 | ~30 |
| A100 80 GB (PCIe / SXM) | 1.19 / 1.39 | 2,500 | 13 h | 16–19 | 3.6–4.2 | 1.0 | ~21–24 | ~36–42 |
| H100 SXM | 2.69 | 4,500 | 7.5 h | 20 | 8.1 | 1.0 | ~29 | ~49 |

Prices are from runpod.io/pricing on 2026-10-22: Community Cloud, on demand;
network storage USD 0.07/GB-month. Vast.ai lists the RTX 4090 at about
USD 0.30/h on demand and USD 0.15/h interruptible (aggregators,
September 2026); `train.py` resumes after an interruption, so interruptible
instances are fine. The beat model can also run on the GPU: it is
pipeline-bound at ~10,000 beats/s with 8 data threads, 10 epochs in ~7 h,
+USD 2.4 on a 4090. On Euler it is free.

### Recommendation

*Superseded by the Euler measurement (section 4): with 24-CPU jobs and bf16,
3 epochs of each foundation model take ~4 days on Euler, so all three models are
trained on Euler first. The GPU path stays the fallback if the jobs cannot run
side by side.* The original recommendation:

**Use both: Euler for everything CPU-friendly, one rented RTX 4090 for the two
foundation models.**

1. **Euler** (free): preprocessing (~3 h to ~1 day, see section 5). Then the beat model and its
   augmentation ablation (2 x ~2 days in parallel). These are small models,
   and the ablation needs two full runs anyway.
2. **RTX 4090 on RunPod Community or Vast.ai** (~USD 10–20, ~1 day): ECGFounder and CLEF,
   3 epochs each, copying only the shards. On Euler the same work takes about 2
   weeks and collides with the 15-day purge.
3. **Fallback without the GPU budget:** one epoch of each foundation model on
   Euler (~2.5 days each), queued right after the beat model.

Expected total wall-clock: **about 3 days** (hybrid) versus **about 1 week**
(Euler only, 1 epoch per foundation model) or 2+ weeks (Euler only, 3 epochs).

## 7. Local augmentation ablation (small scale)

**Not run.** A small local ablation of the synthetic augmentation was started
in the cloud session that wrote this plan; no result was committed. The
synthetic augmentation is no longer used: training uses every 130 Hz frame of
the real signal instead (section 2).

## 8. The 15-day purge

Everything under `/cluster/scratch/mfiorani/` that has not been modified for 15
days is deleted. Never `touch` files to dodge this; the shards are reproducible.

1. **Copy results off as soon as a run finishes** (from the Mac or the VPS):
   ```bash
   rsync -av --include='*/' --include='best.pt' --include='DONE' --include='*.json' --exclude='*' \
       mfiorani@euler.ethz.ch:/cluster/scratch/mfiorani/open-holter/runs/ ~/open-holter-runs/
   rsync -av mfiorani@euler.ethz.ch:/cluster/scratch/mfiorani/open-holter/shards/manifest.json ~/open-holter-runs/
   rsync -av mfiorani@euler.ethz.ch:/cluster/scratch/mfiorani/open-holter/logs/ ~/open-holter-runs/logs/
   ```
   `best.pt` files are 0.002 GB (beat) and ~0.12 GB (each foundation model,
   weights only; `last.pt` with optimiser state is ~0.37 GB and need not be
   kept once a run is DONE).
2. **Keep the shards reproducible, not stored.** `prep.py` is deterministic
   (per-patient seeds), and `shards/manifest.json` (counts, sizes, SHA-256)
   records what a run was trained on. If the shards are purged, re-running
   `submit_prep.sh` rebuilds them in a few hours.
3. **Mid-run purge risk.** A training chain modifies `last.pt` every 15 min, and
   shards are read, not modified. A chain that starts more than 15 days after
   preprocessing will find shards deleted: re-run `submit_prep.sh` first (it
   only rebuilds missing shards).
4. **Never commit** shards, checkpoints or weights (`.gitignore` covers
   `data/`, `*.h5`, `*.pt`, `ai/**/weights/`). Results tables (`*.md`, small
   `*.json`) are fine.

## 9. Licences of what comes out

Weights trained on Icentia11k (CC BY-SA 4.0) are treated as adapted material:
share them, if ever, under CC BY-SA 4.0 with attribution. The other sources
are CC BY 4.0 or ODC-By 1.0 (attribution). CPSC2020 ships without a licence
file, so check the ICBEB terms before redistributing a model trained on it. The
CLEF checkpoint states no licence, so a fine-tuned CLEF stays private.
ECGFounder (MIT) and the heartKIT architecture (BSD-3) allow sharing with
their notices. The repository code stays MIT.
