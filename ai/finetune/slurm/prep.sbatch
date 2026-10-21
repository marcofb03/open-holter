#!/bin/bash
# One array task of ai/finetune/prep.py. Submitted by submit_prep.sh with
#   --export=ALL,SOURCE=<source>,NTASKS=<n>  --array=0-<n-1>%<parallel>
#SBATCH --job-name=oh-prep
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "${REPO:-$HOME/open-holter}/ai/finetune/slurm/env.sh"
# Raw files go to node-local $TMPDIR (requested with --tmp) and are deleted patient by patient;
# only the finished shards reach $SCR. A task whose shards exist exits at once (idempotent).
exec python "$REPO/ai/finetune/prep.py" "$SOURCE" --task-id "$SLURM_ARRAY_TASK_ID" --n-tasks "$NTASKS" \
    --out "$SHARDS" --tmp "$TMPDIR" --workers "$SLURM_CPUS_PER_TASK"
