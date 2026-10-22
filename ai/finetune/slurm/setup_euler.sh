#!/bin/bash
# One-time setup on an Euler login node (about 10 minutes):
#   git clone https://github.com/marcofb03/open-holter $HOME/open-holter
#   bash $HOME/open-holter/ai/finetune/slurm/setup_euler.sh
set -euo pipefail
REPO=${REPO:-$HOME/open-holter}; VENV=${VENV:-$HOME/venvs/open-holter}
module load stack/2024-06 python/3.11.6
python -m venv "$VENV" && source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy pandas wfdb scikit-learn h5py bottleneck
cd "$REPO/ai/ecgfounder" && python download_weights.py     # 370 MB, MIT
cd "$REPO/ai/clef" && python download_weights.py           # 368 MB, Zenodo (no stated licence: private use)
mkdir -p /cluster/scratch/$USER/open-holter/logs
echo "ok: now  cd /cluster/scratch/$USER/open-holter && bash $REPO/ai/finetune/slurm/submit_prep.sh"
