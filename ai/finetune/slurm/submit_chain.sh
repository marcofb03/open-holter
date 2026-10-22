#!/bin/bash
# Queue N dependent training jobs; each resumes where the previous one stopped.
#   [CPUS=24] [AFTER=<jobid>] bash submit_chain.sh MODEL N [TIME] [RUN] [extra train.py args...]
# Examples (from /cluster/scratch/$USER/open-holter):
#   bash $REPO/ai/finetune/slurm/submit_chain.sh beat 4                                # 4 x 24 h, all frames
#   bash $REPO/ai/finetune/slurm/submit_chain.sh ecgfounder 5 24:00:00 ecgfounder --bf16
#   AFTER=12345 bash ...   # first job waits until job (array) 12345 finished OK, e.g. the last prep array
# CPUS: 24 by default. Measured on Euler (EPYC 9654, results/step_timing_euler*.md): the beat model
# is 2.5x slower on 48 threads than on 24, and ECGFounder gains nothing beyond 24.
# afterany: the next job starts even if this one hit its time limit (that is the normal case).
# A finished run writes $RUNS/<run>/DONE and the remaining jobs exit in seconds.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=$1; N=$2; TIME=${3:-24:00:00}; RUN=${4:-$MODEL}; shift $(( $# < 4 ? $# : 4 ))
EXTRA="$*"
CPUS=${CPUS:-24}
# bf16 needs AVX-512 BF16 hardware (Zen 4). On Euler's many EPYC 7742 (Zen 2) nodes it is emulated:
# ECGFounder trained at 5 windows/s there vs ~180/s on an EPYC 9654 (measured 2026-10-22). So bf16 jobs
# only go to Zen 4 nodes. The fp32 beat model runs about as fast on any node and is not constrained.
case " $EXTRA " in *" --bf16 "*) CONSTRAINT=${CONSTRAINT:-EPYC_9654|EPYC_9554|EPYC_9354} ;; esac
cd "/cluster/scratch/$USER/open-holter"
prev=""
for i in $(seq 1 "$N"); do
  if [ -n "$prev" ]; then dep="--dependency=afterany:$prev"; else dep=${AFTER:+--dependency=afterok:$AFTER}; fi
  prev=$(sbatch --parsable $dep ${CONSTRAINT:+--constraint=$CONSTRAINT} --time="$TIME" --cpus-per-task="$CPUS" --job-name="oh-$RUN" \
         --export=ALL,MODEL="$MODEL",RUN="$RUN",TRAIN_ARGS="$EXTRA" "$HERE/train.sbatch")
  echo "queued $RUN part $i/$N: job $prev"
done
