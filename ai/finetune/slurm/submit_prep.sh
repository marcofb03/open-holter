#!/bin/bash
# Submit every preprocessing array (run from /cluster/scratch/$USER/open-holter, where logs/ lives).
# Re-running it after failures is safe: complete tasks exit in seconds.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
cd "$SCR"
python "$REPO/ai/finetune/prep.py" noise --out "$SHARDS"            # NSTDB noise, seconds
# source:tasks:parallel. Icentia11k: 300 tasks x ~37 patients (~1.3 h each), at most 40 at a time.
for spec in icentia11k:300:40 ptb-xl:20:10 cinc2017:4:4 cpsc2020:10:10 ltafdb:12:12 cpsc2021:8:8 shdb-af:16:16 afdb:4:4; do
  IFS=: read -r src n par <<<"$spec"
  sbatch --parsable --array=0-$((n - 1))%$par --export=ALL,SOURCE=$src,NTASKS=$n "$HERE/prep.sbatch"
done
echo "when all are done:  python $REPO/ai/finetune/prep.py manifest --out $SHARDS"
