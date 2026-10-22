#!/bin/bash
# One array task of ai/finetune/prep.py. Submitted by submit_prep.sh with
#   --export=ALL,SOURCE=<source>,NTASKS=<n>[,DL_THREADS=<k>][,PREP_OUT=<dir>][,PREP_ARGS="--limit 8"]
#   --array=0-<n-1>%<parallel>
#SBATCH --job-name=oh-prep
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source "${REPO:-$HOME/open-holter}/ai/finetune/slurm/env.sh"
[ -n "${https_proxy:-}" ] || { echo "no https_proxy: eth_proxy is required on compute nodes" >&2; exit 1; }
# Raw files go to node-local $TMPDIR (requested with --tmp) and are deleted patient by patient;
# only the finished shards reach $SCR. A task whose shards exist exits at once (idempotent).
# Concurrent downloads of this task: 8 workers x DL_THREADS (default 1), all through the ETH proxy.
# PREP_OUT/PREP_ARGS are for the pilot (pilot_prep.sh): a --limit run must never write into $SHARDS,
# or its partial shards would count as complete.
exec python "$REPO/ai/finetune/prep.py" "$SOURCE" --task-id "$SLURM_ARRAY_TASK_ID" --n-tasks "$NTASKS" \
    --out "${PREP_OUT:-$SHARDS}" --tmp "$TMPDIR" --workers "$SLURM_CPUS_PER_TASK" \
    --dl-threads "${DL_THREADS:-1}" ${PREP_ARGS:-}
