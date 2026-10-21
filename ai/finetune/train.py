"""Resumable CPU (or GPU) training of the three fine-tuning models on the shards.

    python train.py --model beat       --shards $SCRATCH/open-holter/shards --run $SCRATCH/open-holter/runs/beat
    python train.py --model ecgfounder --shards ... --run .../ecgfounder --hours 23.5
    python train.py --model beat --no-augment --run .../beat_noaug         # augmentation ablation

Resumable by design (cluster jobs are killed at their time limit):
  * a checkpoint (model, optimiser, scheduler, data-stream position, step, best
    score) is written atomically to <run>/last.pt every --ckpt-minutes (default 15)
    and when the job receives SIGUSR1 / SIGTERM (sbatch --signal=B:USR1@600);
  * on start, <run>/last.pt is loaded if present, so resubmitting the same command
    (submit_chain.sh) simply continues; the data stream resumes at the exact chunk;
  * the run stops cleanly after --hours of wall time, or when --epochs are done,
    in which case <run>/DONE is written and later jobs of the chain exit at once.
Validation runs every --val-minutes on the val shards (deterministic H10-like
conversion, no augmentation) and keeps <run>/best.pt by the recall-first score.

Recall-first objective:
  windows  masked binary cross-entropy, positive weight neg/pos per target (capped
           at 50); score = mean over targets of the sensitivity at 95 % specificity
  beats    cross-entropy with class weights sqrt(N / (K n_c)) x {S 1.5, V 2.0};
           score = mean of the V and S sensitivities at 1 % false positives among
           normal beats
Augmentation (h10.convert / h10.augment) is applied to the train split only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import h10  # noqa: E402
import models  # noqa: E402
import shards  # noqa: E402

DEFAULT_SOURCES = {"win": ["icentia11k", "ltafdb", "cpsc2021", "shdb-af", "ptb-xl", "cinc2017", "cpsc2020"],
                   "beat": ["icentia11k", "ltafdb", "cpsc2020"]}
BEAT_BOOST = np.array([1.0, 1.5, 2.0, 1.0])
LR = {"ecgfounder": 1e-4, "clef": 1e-4, "beat": 2e-3}
BATCH = {"ecgfounder": 64, "clef": 64, "beat": 512}


def shard_files(root: Path, sources: list[str], split: str) -> list[Path]:
    return [p for s in sources for p in sorted((root / s / split).glob("*.h5")) if shards.is_complete(p)]


def make_transform(name: str, train: bool, noise: shards.Noise | None):
    kind = models.KIND[name]

    def tf(x16, y, fs, rng):
        if train:
            x = h10.convert(x16, fs, kind, rng)
            x, y = h10.augment(x, y, kind, rng, noise)
        else:
            x = h10.convert(x16, fs, kind)
        return models.preprocess(name, x), y
    return tf


def label_stats(stream: shards.ChunkStream, kind: str, n_chunks: int = 64) -> np.ndarray:
    """Positive weights (windows) or class weights (beats) from the raw labels of a sample of chunks."""
    import h5py

    ys = []
    for i, s in stream.index[:: max(len(stream.index) // n_chunks, 1)]:
        with h5py.File(stream.files[i], "r") as f:
            ys.append(f[f"{kind}_y"][s:s + stream.chunk])
    y = np.concatenate(ys)
    if kind == "win":
        pos, neg = (y == 1).sum(0), (y == 0).sum(0)
        return np.clip(neg / np.maximum(pos, 1), 1.0, 50.0)
    n = np.bincount(y, minlength=models.N_BEAT).astype(float)
    return np.sqrt(n.sum() / (models.N_BEAT * np.maximum(n, 1))) * BEAT_BOOST


def se_at_sp(y: np.ndarray, s: np.ndarray, sp: float) -> float:
    neg, pos = s[y == 0], s[y == 1]
    if not len(neg) or not len(pos):
        return float("nan")
    return float((pos > np.quantile(neg, sp)).mean())


def validate(net, name: str, stream: shards.ChunkStream, max_chunks: int, bf16: bool) -> dict:
    import torch

    kind = models.KIND[name]
    net.eval()
    ys, ss = [], []
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        for k in range(min(max_chunks, len(stream.index))):
            x, y = stream.read(k, 0)  # fixed chunks: comparable across validations
            out = net(torch.from_numpy(x)).float()
            ss.append((torch.sigmoid(out) if kind == "win" else torch.softmax(out, 1)).numpy())
            ys.append(y)
    net.train()
    if not ys:
        return {"score": float("nan")}
    y, s = np.concatenate(ys), np.concatenate(ss)
    if kind == "win":
        per = {t: se_at_sp(y[:, i], s[:, i], 0.95) for i, t in enumerate(h10.WINDOW_TARGETS)}
        vals = [v for v in per.values() if v == v]
        return {"score": float(np.mean(vals)) if vals else float("nan"), "se_at_sp95": per, "n": int(len(y))}
    res = {}
    for c, nm in ((2, "V"), (1, "S")):
        neg = s[y == 0, c]
        res[nm] = float((s[y == c, c] > np.quantile(neg, 0.99)).mean()) if (y == c).any() and len(neg) else float("nan")
    rec = {nm: float((s[y == c].argmax(1) == c).mean()) if (y == c).any() else float("nan")
           for c, nm in enumerate(h10.BEAT_CLASSES)}
    vals = [v for v in res.values() if v == v]
    return {"score": float(np.mean(vals)) if vals else float("nan"), "se_at_fp1": res, "recall": rec, "n": int(len(y))}


def save(path: Path, obj: dict) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(models.KIND), required=True)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--epochs", type=float, default=None, help="default: 3 (window models), 20 (beat)")
    ap.add_argument("--hours", type=float, default=float(os.environ.get("TRAIN_HOURS", 23.5)))
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--val-minutes", type=float, default=60.0)
    ap.add_argument("--val-chunks", type=int, default=40)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    ap.add_argument("--data-threads", type=int, default=None, help="default: threads // 6")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (Zen 4 / AMX CPUs, or GPU)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run / "DONE").exists():
        print(f"{args.run}/DONE exists: nothing to do", file=sys.stderr)
        return
    name, kind = args.model, models.KIND[args.model]
    dthreads = args.data_threads or max(args.threads // 6, 1)
    torch.set_num_threads(max(args.threads - dthreads, 1))
    torch.manual_seed(args.seed)
    srcs = args.sources or DEFAULT_SOURCES[kind]
    noise = None if args.no_augment else shards.Noise(args.shards / "noise" / "nstdb_noise.h5")
    batch = args.batch or BATCH[name]
    train_s = shards.ChunkStream(shard_files(args.shards, srcs, "train"), kind,
                                 make_transform(name, not args.no_augment, noise), chunk=batch,
                                 seed=args.seed, threads=dthreads)
    val_s = shards.ChunkStream(shard_files(args.shards, srcs, "val"), kind, make_transform(name, False, None),
                               chunk=256, seed=12345, threads=dthreads)
    epochs = args.epochs or (3 if kind == "win" else 20)
    steps_per_epoch = len(train_s.index)
    total = max(int(math.ceil(epochs * steps_per_epoch)), 1)

    net = models.build(name).to(args.device)
    lr = args.lr or LR[name]
    head = list((net.dense if hasattr(net, "dense") else net[-1]).parameters())  # new output layer
    body = [p for p in net.parameters() if all(p is not q for q in head)]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": lr * 10 if name != "beat" else lr}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in opt.param_groups], total_steps=total,
                                                pct_start=0.05)
    w = torch.tensor(label_stats(train_s, kind), dtype=torch.float32, device=args.device)
    lossf = (torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="none") if kind == "win"
             else torch.nn.CrossEntropyLoss(weight=w))

    step, best, log = 0, -1.0, []
    last = args.run / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=args.device, weights_only=True)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        train_s.load_state(ck["stream"])
        step, best, log = ck["step"], ck["best"], ck.get("log", [])
        print(f"resumed at step {step}/{total} (epoch {train_s.epoch}, chunk {train_s.pos})", file=sys.stderr)

    stop = {"now": False}

    def on_signal(sig, frame):  # Slurm warns before the time limit: checkpoint and leave
        stop["now"] = True
    for sg in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sg, on_signal)

    def checkpoint():
        save(last, {"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "stream": train_s.state(), "step": step, "best": best, "log": log, "args": vars(args) | {
                        "shards": str(args.shards), "run": str(args.run)}})

    t_ck = t_val = time.time()
    t_log, n_seen = time.time(), 0
    net.train()
    for x, y in train_s:
        if step >= total:
            break
        xb = torch.from_numpy(x).to(args.device)
        with torch.autocast(args.device if args.device != "cpu" else "cpu", dtype=torch.bfloat16, enabled=args.bf16):
            out = net(xb).float()
        if kind == "win":
            yt = torch.from_numpy(y.astype(np.float32)).to(args.device)
            m = (yt >= 0).float()
            loss = (lossf(out, yt.clamp(min=0)) * m).sum() / m.sum().clamp(min=1)
        else:
            loss = lossf(out, torch.from_numpy(y.astype(np.int64)).to(args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        n_seen += len(y)
        now = time.time()
        if now - t_log > 300:
            rate = n_seen / (now - t_log)
            print(f"step {step}/{total} epoch {train_s.epoch} loss {loss.item():.4f} {rate:.0f} examples/s",
                  file=sys.stderr, flush=True)
            log.append({"step": step, "loss": float(loss.item()), "examples_per_s": rate, "t": now})
            t_log, n_seen = now, 0
        if now - t_val > 60 * args.val_minutes:
            v = validate(net, name, val_s, args.val_chunks, args.bf16)
            log.append({"step": step, "val": v})
            print(f"val step {step}: {json.dumps(v)}", file=sys.stderr, flush=True)
            if v["score"] == v["score"] and v["score"] > best:
                best = v["score"]
                save(args.run / "best.pt", {"model": net.state_dict(), "step": step, "val": v, "model_name": name})
            t_val = time.time()
        if stop["now"] or now - t_ck > 60 * args.ckpt_minutes or now - t_start > 3600 * args.hours:
            checkpoint()
            t_ck = time.time()
            if stop["now"] or now - t_start > 3600 * args.hours:
                print(f"stopping at step {step} (time limit); resubmit to continue", file=sys.stderr)
                return
    v = validate(net, name, val_s, args.val_chunks, args.bf16)
    log.append({"step": step, "val": v})
    if v["score"] == v["score"] and v["score"] > best:
        best = v["score"]
        save(args.run / "best.pt", {"model": net.state_dict(), "step": step, "val": v, "model_name": name})
    checkpoint()
    (args.run / "DONE").write_text(json.dumps({"step": step, "best": best, "final_val": v}, indent=1))
    print(f"done: {step} steps, best val score {best:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
