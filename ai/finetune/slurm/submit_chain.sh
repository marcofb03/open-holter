#!/bin/bash
# Queue N dependent training jobs; each resumes where the previous one stopped.
#   bash submit_chain.sh MODEL N [TIME] [RUN] [extra train.py args...]
# Examples (from /cluster/scratch/$USER/open-holter):
#   bash $REPO/ai/finetune/slurm/submit_chain.sh beat 2                       # 2 x 24 h
#   bash $REPO/ai/finetune/slurm/submit_chain.sh beat 2 24:00:00 beat_noaug --no-augment   # ablation
#   bash $REPO/ai/finetune/slurm/submit_chain.sh ecgfounder 6 04:00:00        # short-job variant
# afterany: the next job starts even if this one hit its time limit (that is the normal case).
# A finished run writes $RUNS/<run>/DONE and the remaining jobs exit in seconds.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=$1; N=$2; TIME=${3:-24:00:00}; RUN=${4:-$MODEL}; shift $(( $# < 4 ? $# : 4 ))
EXTRA="$*"
cd "/cluster/scratch/$USER/open-holter"
prev=""
for i in $(seq 1 "$N"); do
  dep=${prev:+--dependency=afterany:$prev}
  prev=$(sbatch --parsable $dep --time="$TIME" --job-name="oh-$RUN" \
         --export=ALL,MODEL="$MODEL",RUN="$RUN",TRAIN_ARGS="$EXTRA" "$HERE/train.sbatch")
  echo "queued $RUN part $i/$N: job $prev"
done
