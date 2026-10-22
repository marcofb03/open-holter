#!/bin/bash
# Pilot before the full preprocessing: ONE Icentia11k task limited to 8 patients, written to
# $SCR/pilot (never to $SHARDS), to check that downloads work through the ETH proxy and to
# measure the time per patient. Run from /cluster/scratch/$USER/open-holter:
#   bash $REPO/ai/finetune/slurm/pilot_prep.sh
# Then read logs/oh-prep-<job>_0.out: per patient "NNN MB raw" and "download + read N s".
# Delete $SCR/pilot afterwards.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
cd "$SCR"
sbatch --array=0 --time=01:00:00 \
  --export=ALL,SOURCE=icentia11k,NTASKS=300,DL_THREADS=${DL_THREADS:-1},PREP_OUT=$SCR/pilot,PREP_ARGS="--limit 8" \
  "$HERE/prep.sbatch"
