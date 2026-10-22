# Sourced by every job script. Edit the three paths if your layout differs.
export REPO=${REPO:-$HOME/open-holter}                       # git clone of this repository
export VENV=${VENV:-$HOME/venvs/open-holter}                 # python venv (setup_euler.sh)
export SCR=${SCR:-/cluster/scratch/$USER/open-holter}        # 15-day purge: shards, runs, logs only
export SHARDS=$SCR/shards RUNS=$SCR/runs

# Euler software stack (module names as of 2026; `module avail python` if they moved)
module load stack/2024-06 python/3.11.6 >/dev/null 2>&1 || true
# Compute nodes reach the internet only through the ETH proxy (docs.hpc.ethz.ch/connections/firewall/#eth-proxy).
# It has limited capacity and blocks a site for every Euler user if overused: keep downloads throttled
# (submit_prep.sh: PAR x 8 workers x DL_THREADS concurrent downloads).
module load eth_proxy >/dev/null 2>&1 || echo "env.sh: WARNING: module eth_proxy not loaded, downloads will fail" >&2
source "$VENV/bin/activate"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$SCR/logs" "$SHARDS" "$RUNS"
