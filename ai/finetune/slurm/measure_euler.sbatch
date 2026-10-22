#!/bin/bash
# 10-minute calibration on a real 48-core node: one training step of each model, fp32 and bf16.
#   cd /cluster/scratch/$USER/open-holter && sbatch $REPO/ai/finetune/slurm/measure_euler.sbatch
#SBATCH --job-name=oh-measure
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${REPO:-$HOME/open-holter}/ai/finetune/slurm/env.sh"
python "$REPO/ai/finetune/measure_step.py" --threads 48 --tag euler48
python "$REPO/ai/finetune/measure_step.py" --threads 24 --tag euler24 --models ecgfounder beat
cat "$REPO/ai/finetune/results/step_timing_euler48.md"
