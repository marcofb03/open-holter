#!/bin/bash
# Re-run only the patients whose download failed transiently in the prep arrays (HTTP 5xx / 429,
# timeouts, cut transfers), into repair-<hash>.h5 shards next to the task shards (train.py and the
# manifest pick them up). Expected failures are not retried: a missing annotation file (SHDB-AF has
# rhythm annotations for 98 of its 128 records) or HTTP 404.
#   cd /cluster/scratch/$USER/open-holter && sbatch --dependency=afterany:<last prep array> $REPO/ai/finetune/slurm/repair_prep.sbatch
# Patients that fail again are listed in this job's log (logs/oh-repair-<job>.out).
#SBATCH --job-name=oh-repair
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${REPO:-$HOME/open-holter}/ai/finetune/slurm/env.sh"
[ -n "${https_proxy:-}" ] || { echo "no https_proxy: eth_proxy is required on compute nodes" >&2; exit 1; }
cd "$SCR"
LIST=$SCR/logs/repair-$SLURM_JOB_ID.txt
{ grep -h ": FAILED " logs/oh-prep-*.out | grep -v -e FileNotFoundError -e "HTTP Error 404" \
    | sed -E 's/^ *([^/ ]+)\/([^: ]+): FAILED.*/\1 \2/' | sort -u > "$LIST"; } || true
echo "$(wc -l < "$LIST") patients to repair"
for src in $(cut -d' ' -f1 "$LIST" | sort -u); do
  pts=$(awk -v s="$src" '$1 == s { print $2 }' "$LIST")
  echo "=== $src: $(echo $pts | wc -w) patients"
  python "$REPO/ai/finetune/prep.py" "$src" --patients $pts --out "$SHARDS" --tmp "$TMPDIR" \
      --workers "$SLURM_CPUS_PER_TASK" --dl-threads "${DL_THREADS:-1}"
done
