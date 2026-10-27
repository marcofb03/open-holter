#!/bin/bash
# Preprocess the open sources added for the rhythm transformer, in both formats:
#   windows + beats -> $SHARDS/<source>/   (ai/finetune/prep.py, for the beat model, ECGFounder, CLEF)
#   60 s segments   -> $SCR/seg/<source>/  (ai/rhythm/prep_seg.py, for the rhythm transformer)
# Local copies (physionet.org-only sources, the two archives) are read from OH_LOCAL_DATA ($SCR/local).
#   bash submit_extra.sh                       # sources on fast mirrors + the two archives
#   bash submit_extra.sh vtac challenge-2015   # after copying them to $SCR/local
# Like submit_prep.sh: arrays one after another (afterany), <= PAR tasks at a time, downloads through the ETH proxy.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
cd "$SCR"
export OH_LOCAL_DATA=${OH_LOCAL_DATA:-$SCR/local}
mkdir -p "$OH_LOCAL_DATA" "$SCR/seg"
PAR=${PAR:-4}
# source:tasks:workers   (workers: fewer for very long records)
SPECS="edb:2:8 sddb:1:4 butqdb:2:2 chapman:20:8 iridia-af:8:2 sph:10:8 ltstdb:4:4 vtac:4:8 challenge-2015:1:8"
if [ $# -gt 0 ]; then
  SPECS=$(for s in $SPECS; do for want in "$@"; do if [ "${s%%:*}" = "$want" ]; then echo "$s"; fi; done; done)
else
  SPECS=$(for s in $SPECS; do case ${s%%:*} in (ltstdb|vtac|challenge-2015) ;; (*) echo "$s" ;; esac; done)
fi
prev=""
chain=""
if echo "$SPECS" | grep -qE "iridia-af|sph"; then
  prev=$(sbatch --parsable --job-name=oh-fetch-bulk --cpus-per-task=2 --mem-per-cpu=4G --time=12:00:00 \
         --output=logs/%x-%j.out --wrap "source $HERE/env.sh; export OH_LOCAL_DATA=$OH_LOCAL_DATA; \
         python $REPO/ai/rhythm/fetch_bulk.py iridia-af sph")
  echo "bulk download (IRIDIA-AF, SPH): job $prev"
fi
for spec in $SPECS; do
  IFS=: read -r src n w <<<"$spec"
  par=$(( n < PAR ? n : PAR ))
  for fmt in win seg; do
    dep=${chain:+--dependency=afterany:$chain}  # one array at a time: the ETH proxy stays moderate
    case $src in iridia-af|sph) [ -n "$prev" ] && dep="--dependency=afterok:$prev${chain:+,afterany:$chain}" ;; esac
    if [ $fmt = win ]; then py=$REPO/ai/finetune/prep.py; out=$SHARDS; else py=$REPO/ai/rhythm/prep_seg.py; out=$SCR/seg; fi
    j=$(sbatch --parsable $dep --array=0-$((n - 1))%$par --job-name=oh-prep-$fmt \
        --export=ALL,SOURCE=$src,NTASKS=$n,DL_THREADS=1,PREP_PY=$py,PREP_OUT=$out,PREP_WORKERS=$w,OH_LOCAL_DATA=$OH_LOCAL_DATA \
        "$HERE/prep.sbatch")
    chain=$j
    echo "queued $src ($fmt): $n tasks, $par at a time: job $j"
  done
done
