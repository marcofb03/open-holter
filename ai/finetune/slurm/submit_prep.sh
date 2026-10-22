#!/bin/bash
# Submit every preprocessing array (run from /cluster/scratch/$USER/open-holter, where logs/ lives).
# Re-running it after failures is safe: complete tasks exit in seconds.
#
# All downloads go through the shared ETH proxy, so the arrays run ONE AFTER ANOTHER (afterany
# dependency) and each runs at most PAR tasks at a time. Concurrent downloads = PAR x 8 workers x
# DL_THREADS: 4 x 8 x 1 = 32 by default. Run pilot_prep.sh first and raise PAR only if the proxy
# throughput it measured is the bottleneck and moderate.
#   PAR=4 DL_THREADS=1 bash submit_prep.sh            # defaults
#   bash submit_prep.sh icentia11k                    # one source only
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
cd "$SCR"
PAR=${PAR:-4}; DL_THREADS=${DL_THREADS:-1}
python "$REPO/ai/finetune/prep.py" noise --out "$SHARDS"            # NSTDB noise, seconds
# source:tasks, small sources first. Icentia11k: 300 tasks x ~37 patients (~4 GB of raw downloads each).
SPECS="ptb-xl:20 cinc2017:4 cpsc2020:10 ltafdb:12 cpsc2021:8 shdb-af:16 afdb:4 icentia11k:300"
if [ $# -gt 0 ]; then
  SPECS=$(for s in $SPECS; do for want in "$@"; do if [ "${s%%:*}" = "$want" ]; then echo "$s"; fi; done; done)
fi
prev=""
for spec in $SPECS; do
  IFS=: read -r src n <<<"$spec"
  par=$(( n < PAR ? n : PAR ))
  dep=${prev:+--dependency=afterany:$prev}
  prev=$(sbatch --parsable $dep --array=0-$((n - 1))%$par \
         --export=ALL,SOURCE=$src,NTASKS=$n,DL_THREADS=$DL_THREADS "$HERE/prep.sbatch")
  echo "queued $src: $n tasks, $par at a time: job $prev"
done
echo "when all are done:  python $REPO/ai/finetune/prep.py manifest --out $SHARDS"
