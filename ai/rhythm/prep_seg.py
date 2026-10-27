"""Preprocessing array task for the rhythm transformer: raw records -> 60 s segment shards.

    python prep_seg.py icentia11k --task-id 7 --n-tasks 300 --out $SCR/seg --tmp $TMPDIR
    python prep_seg.py vtac       --task-id 0 --n-tasks 4   --out ...     (needs OH_LOCAL_DATA)
    python prep_seg.py manifest   --out ...

Same design as ../finetune/prep.py (one task = a contiguous block of patients, one shard per
split written atomically, re-running a finished task exits at once), with seg.py examples:
<out>/<source>/<split>/task-i-of-n.h5 holding seg_x, seg_y, seg_bt, seg_bc, seg_hb, seg_valid,
seg_t, seg_pid. Sources: the fine-tuning kit's (Icentia11k, LTAFDB, CPSC2021, SHDB-AF, AFDB,
PTB-XL, CinC 2017, CPSC2020; benchmark held-out patients -> test) and the new ones in
extra_sources.py. Raw files are deleted patient by patient.
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
sys.path[:0] = [str(HERE), str(HERE.parent / "finetune")]
import extra_sources as xs  # noqa: E402
import prep  # noqa: E402  (shard paths, manifest)
import seg  # noqa: E402
import shards  # noqa: E402
import sources as fsrc  # noqa: E402

SPLITS = prep.SPLITS
VERSION = 1
BUDGET = {"long": (60, 20), "short": (10_000, 10_000)}  # (cap positive segments, other segments) per patient
SHORT = {"ptb-xl", "cinc2017", "sph", "chapman", "vtac", "challenge-2015"}
ALL = tuple(fsrc.SOURCES) + tuple(xs.SOURCES)


def patients(source: str, policy: str) -> list:
    return xs.patients(source) if source in xs.SOURCES else fsrc.patients(source, policy)


def read(source: str, pid: str, payload, tmp: Path) -> list:
    if source in xs.SOURCES:
        return xs.read_patient(source, pid, payload, tmp)
    recs = fsrc.read_patient(source, pid, payload, tmp)
    return [(ns, x, fs, statements_of_old(ns, ref)) for ns, x, fs, ref in recs]


def statements_of_old(ns: str, ref: dict) -> dict:
    """PTB-XL / CinC 2017 record labels of the fine-tuning kit -> seg statements."""
    if "ptbxl" in ref:
        return ref | {"statements": dict(ref["ptbxl"])}
    if "cinc_label" in ref:
        lab = ref["cinc_label"]
        return ref | {"statements": {"af": 1 if lab == "A" else (0 if lab in "NO" else -1),
                                     "noise": 1 if lab == "~" else 0}}
    return ref


def one_patient(args: tuple):
    source, pid, split, payload, tmp, seed = args
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:seg:{source}:{pid}".encode()).hexdigest(), 16) % 2**32)
    cap, neg = BUDGET["short" if source in SHORT else "long"]
    parts, rates = [], set()
    try:
        for ns, x16, fs, ref in read(source, pid, payload, Path(tmp)):
            rates.add(float(fs))
            parts.append(seg.examples(ns, x16, fs, ref))
    except Exception as e:  # noqa: BLE001 - one bad patient must not kill the task
        print(f"  {source}/{pid}: FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return pid, split, None, time.perf_counter() - t0
    parts = [p for p in parts if len(p["seg_y"])]
    if not parts:
        return pid, split, None, time.perf_counter() - t0
    if len(rates) != 1:
        raise ValueError(f"{source}/{pid}: mixed sampling rates {rates}")
    ex = seg.apply_budget(shards.concat(parts), rng, cap, neg)
    ex["fs"] = rates.pop()
    return pid, split, ex, time.perf_counter() - t0


def run_task(source: str, i: int, n: int, out: Path, tmp: Path, workers: int, policy: str, limit: int | None) -> None:
    paths = {s: prep.shard_path(out, source, s, i, n) for s in SPLITS}
    if all(shards.is_complete(p) for p in paths.values()):
        print(f"{source} task {i}/{n}: already complete", file=sys.stderr)
        return
    pts = patients(source, policy)
    block = np.array_split(np.arange(len(pts)), n)[i]
    todo = [pts[k] for k in block][:limit]
    print(f"{source} task {i}/{n}: {len(todo)} patients", file=sys.stderr, flush=True)
    acc = {s: [] for s in SPLITS}
    pids = {s: [] for s in SPLITS}
    fs = None
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        for pid, split, e, sec in ex.map(one_patient, [(source, p, s, pl, str(tmp), VERSION) for p, s, pl in todo]):
            if e is None:
                continue
            fs_p = e.pop("fs")
            if fs not in (None, fs_p):
                raise ValueError(f"{source}: sampling rate {fs_p} after {fs}")
            fs = fs_p
            e["seg_pid"] = np.full(len(e["seg_y"]), len(pids[split]), np.int32)
            acc[split].append(e)
            pids[split].append(pid)
            print(f"  {pid} [{split}] {len(e['seg_y'])} segments {sec:.0f} s", file=sys.stderr, flush=True)
    for s in SPLITS:
        if acc[s]:
            arrays = shards.concat(acc[s])
        else:
            arrays = seg.examples("none", np.zeros(0, np.int16), fs or 250.0, xs._empty_ref(0.0))
            arrays["seg_pid"] = np.zeros(0, np.int32)
        shards.write_shard(paths[s], arrays, {"source": source, "split": s, "task": i, "n_tasks": n, "fs": fs or 0.0,
                                              "margin_s": seg.h10.MARGIN_S["seg"], "patients": pids[s],
                                              "targets": list(seg.SEG_TARGETS), "version": VERSION, "policy": policy})
    print(f"{source} task {i}/{n} done in {time.perf_counter() - t0:.0f} s", file=sys.stderr)


def manifest(out: Path) -> None:
    import h5py

    rows = []
    for p in sorted(out.glob("*/*/*.h5")):
        with h5py.File(p, "r") as f:
            y = f["seg_y"][:]
            rows.append({"file": str(p.relative_to(out)), "bytes": p.stat().st_size, "segments": len(y),
                         "patients": len(json.loads(f.attrs["patients"])),
                         "positives": {t: int((y[..., k] == 1).any(1).sum()) for k, t in enumerate(seg.SEG_TARGETS)}})
    tot = {"files": len(rows), "segments": sum(r["segments"] for r in rows), "bytes": sum(r["bytes"] for r in rows),
           "positives": {t: sum(r["positives"][t] for r in rows) for t in seg.SEG_TARGETS}}
    (out / "manifest.json").write_text(json.dumps(tot | {"shards": rows}, indent=1))
    print(json.dumps(tot))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=list(ALL) + ["manifest"])
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmp", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--dl-threads", type=int, default=int(os.environ.get("OH_DL_THREADS", 8)))
    ap.add_argument("--policy", choices=["benchmark", "all"], default="benchmark")
    ap.add_argument("--limit", type=int, default=None, help="patients per task (testing)")
    args = ap.parse_args()
    os.environ["OH_DL_THREADS"] = str(args.dl_threads)
    if args.source == "manifest":
        return manifest(args.out)
    run_task(args.source, args.task_id, args.n_tasks, args.out, args.tmp, args.workers, args.policy, args.limit)


if __name__ == "__main__":
    main()
