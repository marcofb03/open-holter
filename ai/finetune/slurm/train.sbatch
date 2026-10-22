#!/bin/bash
# One training job (resumes from $RUNS/<run>/last.pt). Submitted by submit_chain.sh.
#   MODEL=beat|ecgfounder|clef   RUN=<run name>   TRAIN_ARGS="<extra train.py args>"
#SBATCH --job-name=oh-train
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${REPO:-$HOME/open-holter}/ai/finetune/slurm/env.sh"
# Stop 15 min before the limit (checkpoint, exit 0); USR1 600 s before the limit also checkpoints.
LIMIT_MIN=$(squeue -h -j "$SLURM_JOB_ID" -o %l | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else print $1 }')
export TRAIN_HOURS=$(python -c "print(max(($LIMIT_MIN - 15) / 60, 0.1))")
exec python "$REPO/ai/finetune/train.py" --model "$MODEL" --shards "$SHARDS" --run "$RUNS/$RUN" \
    --threads "$SLURM_CPUS_PER_TASK" ${TRAIN_ARGS:-}
# bf16 is opt-in (TRAIN_ARGS="--bf16"). On Euler's EPYC 9654 it made ECGFounder and CLEF 1.6x faster
# (results/step_timing_euler48.md); use it for them, not for the beat model (fp32, 24 threads).
