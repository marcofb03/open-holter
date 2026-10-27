"""Resumable CPU (or GPU) training of the three fine-tuning models on the shards.

    python train.py --model beat       --shards $SCRATCH/open-holter/shards --run $SCRATCH/open-holter/runs/beat
    python train.py --model ecgfounder --shards ... --run .../ecgfounder --bf16
    python train.py --model beat --conversion fixed --run .../beat_fixed   # one fixed frame only

Train until the validation score stops improving (not for a fixed number of epochs):
  * validation every --val-every epochs (default 0.25) on a fixed set of val-patient
    examples (--val-chunks x 256; default 200 chunks = 51,200 beats, 100 = 25,600 windows);
  * <run>/best.pt is saved whenever the score improves: the bottom of the valley;
  * the learning rate is multiplied by --lr-factor (0.3) after --lr-patience validations
    without an improvement larger than --min-delta (0.002);
  * training stops after --stop-patience validations without such an improvement
    (beat 10 = 2.5 epochs, windows 6 = 1.5 epochs: at least two LR reductions first),
    or at the cap --epochs (beat 25, windows 6), or before the 15-day scratch purge
    (--purge-days: stop that many days after the oldest shard was written);
  * a short linear warm-up (--warmup-steps) starts every run;
  * validation uses the fixed stratified set <shards>/valset_<kind>.h5 when it exists
    (valset.py: every rare event of the val patients + a fixed random sample), and the
    plateau rules and best.pt follow the mean of the last --val-smooth (3) scores, not one
    noisy score; a resumed run whose validation set changed restarts its best/plateau
    bookkeeping (the learning rate and weights carry on);
  * --no-early-stop: never stop on a plateau, only at the cap or the purge deadline; the
    learning rate still drops on plateaus but not below --min-lr-frac (1e-3) of its start.
DONE records why the run stopped (plateau, max_epochs or purge_deadline).

Resumable by design (cluster jobs are killed at their time limit):
  * a checkpoint (model, optimiser incl. learning rates, data-stream position, step,
    plateau state) is written atomically to <run>/last.pt every --ckpt-minutes (default
    15) and when the job receives SIGUSR1 / SIGTERM (sbatch --signal=B:USR1@600);
  * on start, <run>/last.pt is loaded if present, so resubmitting the same command
    (submit_chain.sh) simply continues; the data stream resumes at the exact chunk;
  * once <run>/DONE exists, later jobs of the chain exit at once. Delete DONE (and pass
    a larger --epochs or --purge-days 0) to continue a run elsewhere, e.g. on a GPU.

Recall-first objective:
  windows  masked binary cross-entropy, positive weight neg/pos per target (capped
           at 50); score = mean over targets of the sensitivity at 95 % specificity
  beats    cross-entropy with class weights sqrt(N / (K n_c)) x {S 1.5, V 2.0};
           score = mean of the V and S sensitivities at 1 % false positives among
           normal beats
The checkpoint is chosen by this score (the clinical goal: high sensitivity at a low
false-alarm rate); the validation loss is logged next to it.

Conversion of the train split (--conversion; val and test always use the fixed frame 0):
  frames  (default) every 130 Hz frame of the real signal, a different one per example
          each epoch (h10.convert_frames); windows also get a random crop start.
          Nothing is added to the signal.
  fixed   frame 0 only (the deterministic conversion)
  random  the earlier synthetic augmentation (random rates, filters, gain, NSTDB noise);
          kept for reference, not used
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
import valset  # noqa: E402

DEFAULT_SOURCES = {"win": ["icentia11k", "ltafdb", "cpsc2021", "shdb-af", "ptb-xl", "cinc2017", "cpsc2020"],
                   "beat": ["icentia11k", "ltafdb", "cpsc2020"]}
BEAT_BOOST = np.array([1.0, 1.5, 2.0, 1.0])
LR = {"ecgfounder": 1e-4, "clef": 1e-4, "beat": 2e-3}
BATCH = {"ecgfounder": 64, "clef": 64, "beat": 512}


def shard_files(root: Path, sources: list[str], split: str) -> list[Path]:
    return [p for s in sources for p in sorted((root / s / split).glob("*.h5")) if shards.is_complete(p)]


def make_transform(name: str, conversion: str, noise: shards.Noise | None = None):
    kind = models.KIND[name]

    def tf(x16, y, fs, rng, uid=None, epoch=0):
        if conversion == "frames":
            x = h10.convert_frames(x16, fs, kind, h10.frame_ids(uid, epoch, fs), rng)
        elif conversion == "random":
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


class Plateau:
    """Patience bookkeeping on the validation score (higher is better; NaN counts as no gain)."""

    def __init__(self, lr_patience: int, stop_patience: int, min_delta: float):
        self.lr_patience, self.stop_patience, self.min_delta = lr_patience, stop_patience, min_delta
        self.best, self.bad, self.since_reduce, self.reductions = -math.inf, 0, 0, 0

    def step(self, score: float) -> str:
        """'improved', 'reduce' (lower the LR), 'stop' or 'wait'."""
        if score == score and score > self.best + self.min_delta:
            self.best, self.bad, self.since_reduce = score, 0, 0
            return "improved"
        self.bad += 1
        self.since_reduce += 1
        if self.bad >= self.stop_patience:
            return "stop"
        if self.since_reduce >= self.lr_patience:
            self.since_reduce = 0
            self.reductions += 1
            return "reduce"
        return "wait"

    def state(self) -> dict:
        return {"best": self.best, "bad": self.bad, "since_reduce": self.since_reduce, "reductions": self.reductions}

    def load(self, st: dict) -> None:
        self.best, self.bad = st["best"], st["bad"]
        self.since_reduce, self.reductions = st["since_reduce"], st["reductions"]


def validate(net, name: str, stream: shards.ChunkStream, max_chunks: int, bf16: bool, lossf=None) -> dict:
    import torch

    kind = models.KIND[name]
    dev = next(net.parameters()).device
    net.eval()
    ys, ss, losses = [], [], []
    with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
        for k in range(min(max_chunks, len(stream.index))):
            x, y = stream.read(k, 0)  # fixed chunks: comparable across validations
            out = net(torch.from_numpy(x).to(dev)).float()
            if lossf is not None:
                losses.append(float(_loss(lossf, kind, out, y, dev)) * len(y))
            ss.append((torch.sigmoid(out) if kind == "win" else torch.softmax(out, 1)).cpu().numpy())
            ys.append(y)
    net.train()
    if not ys:
        return {"score": float("nan")}
    y, s = np.concatenate(ys), np.concatenate(ss)
    loss = sum(losses) / len(y) if losses else float("nan")
    if kind == "win":
        per = {t: se_at_sp(y[:, i], s[:, i], 0.95) for i, t in enumerate(h10.WINDOW_TARGETS)}
        vals = [v for v in per.values() if v == v]
        return {"score": float(np.mean(vals)) if vals else float("nan"), "loss": loss, "se_at_sp95": per,
                "n": int(len(y))}
    res = {}
    for c, nm in ((2, "V"), (1, "S")):
        neg = s[y == 0, c]
        res[nm] = float((s[y == c, c] > np.quantile(neg, 0.99)).mean()) if (y == c).any() and len(neg) else float("nan")
    rec = {nm: float((s[y == c].argmax(1) == c).mean()) if (y == c).any() else float("nan")
           for c, nm in enumerate(h10.BEAT_CLASSES)}
    vals = [v for v in res.values() if v == v]
    return {"score": float(np.mean(vals)) if vals else float("nan"), "loss": loss, "se_at_fp1": res, "recall": rec,
            "n": int(len(y))}


def _loss(lossf, kind: str, out, y: np.ndarray, dev):
    import torch

    if kind == "win":
        yt = torch.from_numpy(y.astype(np.float32)).to(dev)
        m = (yt >= 0).float()
        return (lossf(out, yt.clamp(min=0)) * m).sum() / m.sum().clamp(min=1)
    return lossf(out, torch.from_numpy(y.astype(np.int64)).to(dev))


def _cpu_flags() -> str:
    try:
        return Path("/proc/cpuinfo").read_text()
    except OSError:  # not Linux: no check
        return "avx512_bf16"


def purge_deadline(files: list[Path], days: float) -> float:
    """Unix time `days` after the oldest shard was written (inf if days <= 0 or no files)."""
    if days <= 0 or not files:
        return math.inf
    return min(p.stat().st_mtime for p in files) + 86400 * days


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
    ap.add_argument("--epochs", type=float, default=None, help="cap; default 6 (window models), 25 (beat)")
    ap.add_argument("--hours", type=float, default=float(os.environ.get("TRAIN_HOURS", 23.5)))
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--val-every", type=float, default=0.25, help="validate every this many epochs")
    ap.add_argument("--val-chunks", type=int, default=None, help="x 256 examples; default 100 (windows), 200 (beat)")
    ap.add_argument("--lr-patience", type=int, default=None, help="validations; default 2 (windows), 4 (beat)")
    ap.add_argument("--stop-patience", type=int, default=None, help="validations; default 6 (windows), 10 (beat)")
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--val-set", type=Path, default=None, help="default: <shards>/valset_<kind>.h5 if it exists")
    ap.add_argument("--val-smooth", type=int, default=3, help="plateau and best.pt on the mean of the last N scores")
    ap.add_argument("--no-early-stop", action="store_true", help="run to the cap or the purge deadline")
    ap.add_argument("--min-lr-frac", type=float, default=1e-3)
    ap.add_argument("--eval", type=Path, nargs="+", default=None,
                    help="only score these checkpoints on the validation set (-> <run>/rescore.json)")
    ap.add_argument("--lr-factor", type=float, default=0.3)
    ap.add_argument("--warmup-steps", type=int, default=None, help="default: min(1000, 5 %% of an epoch)")
    ap.add_argument("--purge-days", type=float, default=14.0,
                    help="stop this many days after the oldest shard was written (Euler scratch purges at 15); 0: off")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    ap.add_argument("--data-threads", type=int, default=None, help="default: threads // 6")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (Zen 4 / AMX CPUs, or GPU)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conversion", choices=["frames", "fixed", "random"], default="frames")
    ap.add_argument("--no-augment", action="store_true", help="same as --conversion fixed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    args.run.mkdir(parents=True, exist_ok=True)
    if args.bf16 and args.device == "cpu" and "avx512_bf16" not in _cpu_flags():
        print("WARNING: --bf16 on a CPU without AVX-512 BF16 is emulated and ~35x slower "
              "(submit_chain.sh constrains bf16 jobs to Zen 4 nodes)", file=sys.stderr, flush=True)
    if (args.run / "DONE").exists() and not args.eval:
        print(f"{args.run}/DONE exists: nothing to do", file=sys.stderr)
        return
    name, kind = args.model, models.KIND[args.model]
    win = kind == "win"
    dthreads = args.data_threads or max(args.threads // 6, 1)
    torch.set_num_threads(max(args.threads - dthreads, 1))
    torch.manual_seed(args.seed)
    if args.no_augment:
        args.conversion = "fixed"
    noise = shards.Noise(args.shards / "noise" / "nstdb_noise.h5") if args.conversion == "random" else None
    srcs = args.sources or DEFAULT_SOURCES[kind]
    batch = args.batch or BATCH[name]
    train_files = shard_files(args.shards, srcs, "train")
    train_s = shards.ChunkStream(train_files, kind, make_transform(name, args.conversion, noise), chunk=batch,
                                 seed=args.seed, threads=dthreads)
    vpath = args.val_set or args.shards / f"valset_{kind}.h5"
    if vpath.exists():
        val_s = valset.ValSet(vpath, kind, make_transform(name, "fixed"))
        val_id = val_s.id
    else:
        val_s = shards.ChunkStream(shard_files(args.shards, srcs, "val"), kind, make_transform(name, "fixed"),
                                   chunk=256, seed=12345, threads=dthreads)
        val_id = "random-chunks"
    steps_per_epoch = len(train_s.index)
    total = max(int(math.ceil((args.epochs or (6 if win else 25)) * steps_per_epoch)), 1)
    val_steps = max(int(round(args.val_every * steps_per_epoch)), 1)
    val_chunks = len(val_s.index) if val_id != "random-chunks" else (args.val_chunks or (100 if win else 200))
    warm = args.warmup_steps if args.warmup_steps is not None else min(1000, max(steps_per_epoch // 20, 1))
    stop_pat = 10**9 if args.no_early_stop else (args.stop_patience or (6 if win else 10))
    plateau = Plateau(args.lr_patience or (2 if win else 4), stop_pat, args.min_delta)
    deadline = purge_deadline(train_files, args.purge_days)
    if deadline < math.inf:
        print(f"purge deadline: {time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline))}", file=sys.stderr)

    net = models.build(name).to(args.device)
    lr = args.lr or LR[name]
    head = list((net.dense if hasattr(net, "dense") else net[-1]).parameters())  # new output layer
    body = [p for p in net.parameters() if all(p is not q for q in head)]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": lr * 10 if name != "beat" else lr}],
                            weight_decay=1e-4)
    base_lr = [g["lr"] for g in opt.param_groups]
    w = torch.tensor(label_stats(train_s, kind), dtype=torch.float32, device=args.device)
    lossf = (torch.nn.BCEWithLogitsLoss(pos_weight=w, reduction="none") if win
             else torch.nn.CrossEntropyLoss(weight=w))

    if args.eval:  # re-score saved checkpoints on the (new) validation set, nothing else
        res = {}
        for ckp in args.eval:
            ck = torch.load(ckp, map_location=args.device, weights_only=True)
            net.load_state_dict(ck["model"])
            res[str(ckp)] = validate(net, name, val_s, val_chunks, args.bf16, lossf) | {"step": ck.get("step"), "val_id": val_id}
            print(f"{ckp}: {json.dumps(res[str(ckp)])}", file=sys.stderr, flush=True)
        (args.run / "rescore.json").write_text(json.dumps(res, indent=1))
        return
    step, best, log, next_val, recent = 0, -1.0, [], val_steps, []
    last = args.run / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=args.device, weights_only=True)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])  # includes the current (possibly reduced) learning rates
        train_s.load_state(ck["stream"])
        step, best, log, next_val = ck["step"], ck["best"], ck.get("log", []), ck["next_val"]
        if ck.get("val_id") == val_id:
            plateau.load(ck["plateau"])
            plateau.stop_patience = stop_pat
            recent = ck.get("recent", [])
        else:  # a new yardstick: earlier scores are not comparable
            print(f"validation set changed ({ck.get('val_id', 'random-chunks')} -> {val_id}): best and plateau reset",
                  file=sys.stderr)
            best, plateau.reductions = -1.0, ck["plateau"].get("reductions", 0)
            next_val = step + 1  # score the current weights on the new set right away
        print(f"resumed at step {step} (epoch {train_s.epoch}, chunk {train_s.pos}), lr {opt.param_groups[0]['lr']:.2e}, "
              f"best {best:.4f}, {plateau.bad} validations without gain", file=sys.stderr)
    print(f"{name}: {steps_per_epoch} steps per epoch, cap {total} steps, validation every {val_steps} steps on "
          f"{min(val_chunks, len(val_s.index)) * 256} examples", file=sys.stderr, flush=True)

    stop = {"now": False}

    def on_signal(sig, frame):  # Slurm warns before the time limit: checkpoint and leave
        stop["now"] = True
    for sg in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sg, on_signal)

    def checkpoint():
        save(last, {"model": net.state_dict(), "opt": opt.state_dict(), "stream": train_s.state(), "step": step,
                    "best": best, "log": log, "next_val": next_val, "plateau": plateau.state(),
                    "val_id": val_id, "recent": recent,
                    "args": vars(args) | {"shards": str(args.shards), "run": str(args.run)}})

    def finish(reason: str, v: dict | None) -> None:
        checkpoint()
        (args.run / "DONE").write_text(json.dumps({"reason": reason, "step": step, "epoch": step / steps_per_epoch,
                                                   "best": best, "lr_reductions": plateau.reductions,
                                                   "last_val": v}, indent=1))
        print(f"done ({reason}): {step} steps ({step / steps_per_epoch:.2f} epochs), best val score {best:.4f}",
              file=sys.stderr)

    def run_validation() -> str:
        nonlocal best
        v = validate(net, name, val_s, val_chunks, args.bf16, lossf)
        if v["score"] == v["score"]:
            recent.append(float(v["score"]))
        del recent[:-args.val_smooth]
        v["smoothed"] = float(np.mean(recent)) if recent else float("nan")
        v["val_id"] = val_id
        if v["smoothed"] == v["smoothed"] and v["smoothed"] > best:
            best = v["smoothed"]
            save(args.run / "best.pt", {"model": net.state_dict(), "step": step, "val": v, "model_name": name,
                                        "val_id": val_id})
        act = plateau.step(v["smoothed"])
        if act == "reduce":
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = max(g["lr"] * args.lr_factor, b * args.min_lr_frac)
        log.append({"step": step, "epoch": step / steps_per_epoch, "val": v, "action": act,
                    "lr": opt.param_groups[0]["lr"]})
        print(f"val step {step} (epoch {step / steps_per_epoch:.2f}): score {v['score']:.4f} smoothed {v['smoothed']:.4f} "
              f"loss {v.get('loss', 0):.4f} "
              f"-> {act}, lr {opt.param_groups[0]['lr']:.2e}, {plateau.bad} without gain | {json.dumps(v)}",
              file=sys.stderr, flush=True)
        return act

    t_ck = time.time()
    t_log, n_seen = time.time(), 0
    net.train()
    for x, y in train_s:
        if step < warm:
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = b * (step + 1) / warm
        xb = torch.from_numpy(x).to(args.device)
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.bf16):
            out = net(xb).float()
        loss = _loss(lossf, kind, out, y, args.device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        step += 1
        n_seen += len(y)
        now = time.time()
        if now - t_log > 300:
            rate = n_seen / (now - t_log)
            print(f"step {step} epoch {train_s.epoch} loss {loss.item():.4f} {rate:.0f} examples/s",
                  file=sys.stderr, flush=True)
            log.append({"step": step, "loss": float(loss.item()), "examples_per_s": rate, "t": now})
            t_log, n_seen = now, 0
        if step >= next_val:
            next_val = step + val_steps
            if run_validation() == "stop":
                return finish("plateau", log[-1]["val"])
            if step >= total:
                return finish("max_epochs", log[-1]["val"])
        elif step >= total:
            run_validation()
            return finish("max_epochs", log[-1]["val"])
        if time.time() > deadline:
            run_validation()
            return finish("purge_deadline", log[-1]["val"])
        if stop["now"] or now - t_ck > 60 * args.ckpt_minutes or now - t_start > 3600 * args.hours:
            checkpoint()
            t_ck = time.time()
            if stop["now"] or now - t_start > 3600 * args.hours:
                print(f"stopping at step {step} (time limit); resubmit to continue", file=sys.stderr)
                return


if __name__ == "__main__":
    main()
