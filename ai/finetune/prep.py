"""Preprocessing array task: raw PhysioNet / Icentia11k files -> a few large HDF5 shards.

    python prep.py icentia11k --task-id 7 --n-tasks 300 --out $SCRATCH/open-holter/shards --tmp $TMPDIR
    python prep.py ltafdb     --task-id 0 --n-tasks 4   --out ...
    python prep.py noise      --out ...                  # NSTDB bw/em/ma for augmentation (1 small file)
    python prep.py manifest   --out ...                  # counts, sizes and SHA-256 of every shard

Task i of n handles the i-th contiguous block of the source's patients (see
sources.py for the patient lists and splits). For each patient it downloads the
raw files to --tmp, makes H10-like records (h10.py), keeps a budget of window
and beat examples, deletes the raw files, and finally writes one shard per
split: <out>/<source>/<split>/task-<i>-of-<n>.h5 (possibly empty). A task whose
three shards already exist and are complete exits at once, so re-submitting a
whole array after a failure only redoes the failed tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import h10  # noqa: E402
import shards  # noqa: E402
import sources  # noqa: E402

SPLITS = ("train", "val", "test")
VERSION = 1
# Examples kept per patient (windows: all with a positive target up to win_pos, plus win_neg others;
# beats: at most this many per class N / S / V / noise). Short records keep everything they have.
BUDGET = {
    "icentia11k": {"win_pos": 1500, "win_neg": 600, "beats": {0: 1500, 1: 1000, 2: 1000, 3: 500}},
    "default": {"win_pos": 4000, "win_neg": 1500, "beats": {0: 4000, 1: 3000, 2: 3000, 3: 1000}},
}


def shard_path(out: Path, source: str, split: str, i: int, n: int) -> Path:
    return out / source / split / f"task-{i:04d}-of-{n:04d}.h5"


def one_patient(args: tuple) -> tuple[str, str, dict | None, float, float]:
    source, pid, split, payload, tmp, seed = args
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{source}:{pid}".encode()).hexdigest(), 16) % 2**32)
    budget = BUDGET.get(source, BUDGET["default"])
    parts, rates = [], set()
    try:
        recs = sources.read_patient(source, pid, payload, Path(tmp))
        t_read = time.perf_counter() - t0
        for ns, x16, fs, ref in recs:
            rates.add(float(fs))
            ex = h10.examples(ns, x16, fs, ref)
            ex["win_y"] = sources.labels_override(ns, ref, ex["win_y"])
            parts.append(ex)
    except Exception as e:  # noqa: BLE001 - one bad patient must not kill a 4 h task
        print(f"  {source}/{pid}: FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return pid, split, None, time.perf_counter() - t0, 0.0
    if not parts:
        return pid, split, None, time.perf_counter() - t0, t_read
    if len(rates) != 1:
        raise ValueError(f"{source}/{pid}: mixed sampling rates {rates}")
    ex = h10.apply_budget(shards.concat(parts), rng, budget)
    ex["fs"] = rates.pop()
    return pid, split, ex, time.perf_counter() - t0, t_read


def run_task(source: str, i: int, n: int, out: Path, tmp: Path, workers: int, policy: str, limit: int | None) -> None:
    paths = {s: shard_path(out, source, s, i, n) for s in SPLITS}
    if all(shards.is_complete(p) for p in paths.values()):
        print(f"{source} task {i}/{n}: already complete", file=sys.stderr)
        return
    pts = sources.patients(source, policy)
    block = np.array_split(np.arange(len(pts)), n)[i]
    todo = [pts[k] for k in block][:limit]
    print(f"{source} task {i}/{n}: {len(todo)} patients", file=sys.stderr, flush=True)
    acc: dict[str, list] = {s: [] for s in SPLITS}
    pids: dict[str, list] = {s: [] for s in SPLITS}
    jobs = [(source, p, s, pl, str(tmp), VERSION) for p, s, pl in todo]
    t0 = time.perf_counter()
    fs = None
    with ProcessPoolExecutor(workers) as ex:
        for pid, split, e, sec, sec_read in ex.map(one_patient, jobs):
            if e is None:
                continue
            fs_p = e.pop("fs")
            if fs not in (None, fs_p):
                raise ValueError(f"{source}: sampling rate {fs_p} after {fs}")
            fs = fs_p
            k = len(pids[split])
            e["win_pid"] = np.full(len(e["win_y"]), k, np.int32)
            e["beat_pid"] = np.full(len(e["beat_y"]), k, np.int32)
            acc[split].append(e)
            pids[split].append(pid)
            print(f"  {pid} [{split}] {len(e['win_y'])} windows {len(e['beat_y'])} beats {sec:.0f} s "
                  f"(download + read {sec_read:.0f} s)", file=sys.stderr,
                  flush=True)
    L = {k: int(round((h10.OUT_LEN[k] / h10.FS + 2 * h10.MARGIN_S[k]) * (fs or h10.FS))) for k in ("win", "beat")}
    empty = {"win_x": np.zeros((0, L["win"]), np.int16), "win_y": np.zeros((0, len(h10.WINDOW_TARGETS)), np.int8),
             "win_t": np.zeros(0, np.float32), "win_pid": np.zeros(0, np.int32),
             "beat_x": np.zeros((0, L["beat"]), np.int16), "beat_y": np.zeros(0, np.int8),
             "beat_t": np.zeros(0, np.float32), "beat_pid": np.zeros(0, np.int32)}
    for s in SPLITS:
        arrays = shards.concat(acc[s]) if acc[s] else empty
        shards.write_shard(paths[s], arrays, {"source": source, "split": s, "task": i, "n_tasks": n,
                                              "fs": fs or 0.0, "margin_s": h10.MARGIN_S,
                                              "patients": pids[s], "targets": list(h10.WINDOW_TARGETS),
                                              "beat_classes": list(h10.BEAT_CLASSES), "version": VERSION,
                                              "policy": policy})
    print(f"{source} task {i}/{n} done in {time.perf_counter() - t0:.0f} s", file=sys.stderr)


def manifest(out: Path) -> None:
    import h5py

    rows = []
    for p in sorted(out.glob("*/*/*.h5")):
        with h5py.File(p, "r") as f:
            rows.append({"file": str(p.relative_to(out)), "bytes": p.stat().st_size, "windows": len(f["win_y"]),
                         "beats": len(f["beat_y"]), "patients": len(json.loads(f.attrs["patients"])),
                         "complete": int(f.attrs.get("complete", 0))})
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        rows[-1]["sha256"] = h.hexdigest()
    tot = {k: sum(r[k] for r in rows) for k in ("bytes", "windows", "beats", "patients")}
    (out / "manifest.json").write_text(json.dumps({"files": len(rows), **tot, "shards": rows}, indent=1))
    print(json.dumps({"files": len(rows), **tot}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=list(sources.SOURCES) + ["noise", "manifest"])
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--dl-threads", type=int, default=int(os.environ.get("OH_DL_THREADS", 8)),
                    help="parallel downloads per worker (Euler: total = tasks x workers x this, via the ETH proxy)")
    ap.add_argument("--policy", choices=["benchmark", "all"], default="benchmark")
    ap.add_argument("--limit", type=int, default=None, help="patients per task (testing)")
    args = ap.parse_args()
    os.environ["OH_DL_THREADS"] = str(args.dl_threads)  # read by sources.read_patient in the workers
    if args.source == "manifest":
        return manifest(args.out)
    if args.source == "noise":
        p = args.out / "noise" / "nstdb_noise.h5"
        if not shards.is_complete(p):
            shards.write_shard(p, sources.noise_arrays(), {"source": "nstdb", "fs": h10.FS, "unit": "uV"})
        return
    run_task(args.source, args.task_id, args.n_tasks, args.out, args.tmp, args.workers, args.policy, args.limit)


if __name__ == "__main__":
    main()
