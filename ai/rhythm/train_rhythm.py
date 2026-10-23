"""Train the rhythm transformer on 60 s segment shards (prep_seg.py), on a GPU or a CPU.

    python train_rhythm.py --stage pretrain --shards $SEG --run runs/rt_pre  --device cuda --bf16
    python train_rhythm.py --stage finetune --shards $SEG --run runs/rt_ft   --device cuda --bf16 \
        --init runs/rt_pre/best.pt

pretrain  masked reconstruction: 40 % of the tokens (in ~1 s spans) are hidden and their 16
          samples predicted; uses every segment, labelled or not. Score: -validation loss.
finetune  multitask, recall first: 10 s targets (masked BCE, positive weight neg/pos per target,
          cap 50), beat classes at the beat's token (cross-entropy, sqrt-inverse-frequency weights
          x {S 1.5, V 2.0}) and QRS presence per token (sources with reviewed beats only).
          Score: mean of the 10 s targets' sensitivity at 95 % specificity and the V / S beat
          sensitivity at 1 % false positives among normal beats.
Both: every 130 Hz frame of the real signal for training (h10.convert_frames), frame 0 for
validation; train until the score plateaus (the same rules as ../finetune/train.py: LR x0.3
after --lr-patience validations without a gain, stop after --stop-patience), checkpoints every
15 min and on SIGUSR1/SIGTERM, resume from last.pt, DONE with the reason.
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
sys.path[:0] = [str(HERE), str(HERE.parent / "finetune")]
import h10  # noqa: E402
import model as M  # noqa: E402
import seg  # noqa: E402
import shards  # noqa: E402
import train as T  # noqa: E402  (Plateau, purge_deadline, save, se_at_sp, shard_files)

EXTRA = ("bt", "bc", "hb", "valid")
BEAT_BOOST = np.array([1.0, 1.5, 2.0, 1.0])
L130 = h10.OUT_LEN["seg"]
NT = M.n_tokens(L130)


def all_sources(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def make_transform(train: bool):
    def tf(x16, y, fs, rng, uid=None, epoch=0, bt=None, bc=None, hb=None, valid=None):
        if train:
            x = h10.convert_frames(x16, fs, "seg", h10.frame_ids(uid, epoch, fs))
        else:
            x = h10.convert(x16, fs, "seg")
        return seg.to_input(x, valid), {"y": y, "bt": bt, "bc": bc, "hb": hb, "valid": valid}
    return tf


def token_targets(lab: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Beat token indices and classes (flattened), QRS-per-token targets and their mask."""
    bt, bc, hb, valid = lab["bt"], lab["bc"], lab["hb"], lab["valid"]
    n = len(bt)
    ok = bc >= 0
    tok = np.clip((bt * h10.FS / M.PATCH).astype(np.int64), 0, NT - 1)
    rows = np.repeat(np.arange(n)[:, None], bt.shape[1], 1)
    qrs = np.zeros((n, NT), np.float32)
    qrs[rows[ok], tok[ok]] = 1.0
    vt = np.arange(NT)[None, :] * M.PATCH / h10.FS < valid[:, None]
    qmask = (vt & (hb[:, None] == 1)).astype(np.float32)
    return rows[ok], tok[ok], bc[ok].astype(np.int64), qrs, qmask


def span_mask(rng: np.random.Generator, valid: np.ndarray, frac: float = 0.4, span: int = 8) -> np.ndarray:
    """(n, NT) bool: ~frac of each segment's valid tokens hidden, in spans of `span` tokens."""
    n = len(valid)
    out = np.zeros((n, NT), bool)
    vt = np.minimum((valid * h10.FS / M.PATCH).astype(int), NT)
    for i in range(n):
        if vt[i] < span:
            continue
        for s in rng.choice(vt[i] - span + 1, max(int(frac * vt[i] / span), 1), replace=False):
            out[i, s:s + span] = True
    return out


def patches(x: np.ndarray) -> np.ndarray:
    z = x[:, 0]
    pad = NT * M.PATCH - z.shape[1]
    return np.pad(z, ((0, 0), (0, pad))).reshape(len(z), NT, M.PATCH)


def label_stats(stream: shards.ChunkStream, n_chunks: int = 64) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    ys, bcs = [], []
    for i, s in stream.index[:: max(len(stream.index) // n_chunks, 1)]:
        with h5py.File(stream.files[i], "r") as f:
            ys.append(f["seg_y"][s:s + stream.chunk].reshape(-1, len(seg.SEG_TARGETS)))
            bcs.append(f["seg_bc"][s:s + stream.chunk].ravel())
    y, bc = np.concatenate(ys), np.concatenate(bcs)
    pos, neg = (y == 1).sum(0), (y == 0).sum(0)
    pw = np.clip(neg / np.maximum(pos, 1), 1.0, 50.0)
    nb = np.bincount(bc[bc >= 0], minlength=M.N_BEAT).astype(float)
    cw = np.sqrt(nb.sum() / (M.N_BEAT * np.maximum(nb, 1))) * BEAT_BOOST
    return pw, cw


def losses(net, x, lab, stage, lossf, dev, bf16, rng):
    import torch

    xb = torch.from_numpy(x).to(dev)
    if stage == "pretrain":
        m = span_mask(rng, lab["valid"])
        mt = torch.from_numpy(m).to(dev)
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=bf16):
            out = net(xb, mt)
        tgt = torch.from_numpy(patches(x)).to(dev)
        err = ((out["recon"].float() - tgt) ** 2).mean(-1)
        return (err * mt).sum() / mt.sum().clamp(min=1), out
    with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=bf16):
        out = net(xb)
    bce_win, ce_beat = lossf
    y = torch.from_numpy(lab["y"].astype(np.float32)).to(dev)
    mw = (y >= 0).float()
    l_win = (bce_win(out["win"].float(), y.clamp(min=0)) * mw).sum() / mw.sum().clamp(min=1)
    r, t, c, qrs, qmask = token_targets(lab)
    l_beat = ce_beat(out["beat"].float()[torch.from_numpy(r).to(dev), torch.from_numpy(t).to(dev)],
                     torch.from_numpy(c).to(dev)) if len(c) else out["beat"].sum() * 0
    qm = torch.from_numpy(qmask).to(dev)
    l_qrs = (torch.nn.functional.binary_cross_entropy_with_logits(
        out["qrs"].float(), torch.from_numpy(qrs).to(dev), pos_weight=torch.tensor(3.0, device=dev),
        reduction="none") * qm).sum() / qm.sum().clamp(min=1)
    return l_win + l_beat + 0.5 * l_qrs, out


def validate(net, stream, stage, lossf, dev, bf16, max_chunks):
    import torch

    net.eval()
    rng = np.random.default_rng(0)
    tot, n, ys, ws, bcs, bps = 0.0, 0, [], [], [], []
    with torch.inference_mode():
        for k in range(min(max_chunks, len(stream.index))):
            x, lab = stream.read(k, 0)
            loss, out = losses(net, x, lab, stage, lossf, dev, bf16, rng)
            tot += float(loss) * len(x)
            n += len(x)
            if stage == "finetune":
                ys.append(lab["y"].reshape(-1, len(seg.SEG_TARGETS)))
                ws.append(torch.sigmoid(out["win"].float()).cpu().numpy().reshape(-1, len(seg.SEG_TARGETS)))
                r, t, c, _, _ = token_targets(lab)
                bcs.append(c)
                bps.append(torch.softmax(out["beat"].float(), -1).cpu().numpy()[r, t])
    net.train()
    if not n:
        return {"score": float("nan")}
    v = {"loss": tot / n, "n": n}
    if stage == "pretrain":
        return v | {"score": -v["loss"]}
    y, s = np.concatenate(ys), np.concatenate(ws)
    per = {tg: T.se_at_sp(y[:, i], s[:, i], 0.95) for i, tg in enumerate(seg.SEG_TARGETS)}
    bc, bp = np.concatenate(bcs), np.concatenate(bps)
    beat = {}
    for cls, nm in ((2, "V"), (1, "S")):
        neg = bp[bc == 0, cls]
        beat[nm] = float((bp[bc == cls, cls] > np.quantile(neg, 0.99)).mean()) if (bc == cls).any() and len(neg) else float("nan")
    vals = [x for x in list(per.values()) + list(beat.values()) if x == x]
    return v | {"score": float(np.mean(vals)) if vals else float("nan"), "se_at_sp95": per, "beat_se_at_fp1": beat}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--size", choices=list(M.SIZES), default="base")
    ap.add_argument("--init", type=Path, default=None, help="pretrained best.pt to start the fine-tuning from")
    ap.add_argument("--epochs", type=float, default=None, help="cap; default 20 (pretrain), 30 (finetune)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hours", type=float, default=float(os.environ.get("TRAIN_HOURS", 1e6)))
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--val-every", type=float, default=0.25)
    ap.add_argument("--val-chunks", type=int, default=100)
    ap.add_argument("--lr-patience", type=int, default=4)
    ap.add_argument("--stop-patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--lr-factor", type=float, default=0.3)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--purge-days", type=float, default=0.0, help="Euler scratch: stop this many days after the oldest shard")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    ap.add_argument("--data-threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run / "DONE").exists():
        print(f"{args.run}/DONE exists: nothing to do", file=sys.stderr)
        return
    torch.set_num_threads(max(args.threads - (args.data_threads if args.device == "cpu" else 0), 1))
    torch.manual_seed(args.seed)
    dev = args.device
    srcs = args.sources or all_sources(args.shards)
    train_files = T.shard_files(args.shards, srcs, "train")
    train_s = shards.ChunkStream(train_files, "seg", make_transform(True), chunk=args.batch, seed=args.seed,
                                 threads=args.data_threads, extra=EXTRA)
    val_s = shards.ChunkStream(T.shard_files(args.shards, srcs, "val"), "seg", make_transform(False), chunk=64,
                               seed=12345, threads=args.data_threads, extra=EXTRA)
    spe = len(train_s.index)
    total = max(int(math.ceil((args.epochs or (20 if args.stage == "pretrain" else 30)) * spe)), 1)
    val_steps = max(int(round(args.val_every * spe)), 1)
    plateau = T.Plateau(args.lr_patience, args.stop_patience, args.min_delta)
    deadline = T.purge_deadline(train_files, args.purge_days)

    net = M.build(args.size, len(seg.SEG_TARGETS)).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev, weights_only=True)["model"]
        print(f"init from {args.init}: {net.load_state_dict(sd, strict=False)}", file=sys.stderr)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    base_lr = [g["lr"] for g in opt.param_groups]
    lossf = None
    if args.stage == "finetune":
        pw, cw = label_stats(train_s)
        lossf = (torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float32, device=dev), reduction="none"),
                 torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev)))
    rng = np.random.default_rng(args.seed)

    step, best, log, next_val = 0, -math.inf, [], val_steps
    last = args.run / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=True)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        train_s.load_state(ck["stream"])
        plateau.load(ck["plateau"])
        step, best, log, next_val = ck["step"], ck["best"], ck.get("log", []), ck["next_val"]
        print(f"resumed at step {step}, lr {opt.param_groups[0]['lr']:.2e}, best {best:.4f}", file=sys.stderr)
    print(f"{args.stage} {args.size}: {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M parameters, "
          f"{spe} steps per epoch, cap {total}, validation every {val_steps} steps", file=sys.stderr, flush=True)

    stop = {"now": False}
    for sg in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sg, lambda *_: stop.update(now=True))

    def checkpoint():
        T.save(last, {"model": net.state_dict(), "opt": opt.state_dict(), "stream": train_s.state(), "step": step,
                      "best": best, "log": log, "next_val": next_val, "plateau": plateau.state(),
                      "args": {k: str(v) for k, v in vars(args).items()}})

    def finish(reason, v):
        checkpoint()
        (args.run / "DONE").write_text(json.dumps({"reason": reason, "step": step, "epoch": step / spe, "best": best,
                                                   "lr_reductions": plateau.reductions, "last_val": v}, indent=1))
        print(f"done ({reason}): {step} steps ({step / spe:.2f} epochs), best {best:.4f}", file=sys.stderr)

    def run_validation():
        nonlocal best
        v = validate(net, val_s, args.stage, lossf, dev, args.bf16, args.val_chunks)
        if v["score"] == v["score"] and v["score"] > best:
            best = v["score"]
            T.save(args.run / "best.pt", {"model": net.state_dict(), "step": step, "val": v, "size": args.size,
                                          "stage": args.stage, "targets": list(seg.SEG_TARGETS)})
        act = plateau.step(v["score"])
        if act == "reduce":
            for g in opt.param_groups:
                g["lr"] *= args.lr_factor
        log.append({"step": step, "epoch": step / spe, "val": v, "action": act, "lr": opt.param_groups[0]["lr"]})
        print(f"val step {step} (epoch {step / spe:.2f}): score {v['score']:.4f} -> {act} | {json.dumps(v)}",
              file=sys.stderr, flush=True)
        return act, v

    t_ck = t_log = time.time()
    n_seen = 0
    net.train()
    for x, lab in train_s:
        if step < args.warmup_steps:
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = b * (step + 1) / args.warmup_steps
        loss, _ = losses(net, x, lab, args.stage, lossf, dev, args.bf16, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        n_seen += len(x)
        now = time.time()
        if now - t_log > 300:
            rate = n_seen / (now - t_log)
            print(f"step {step} epoch {train_s.epoch} loss {loss.item():.4f} {rate:.1f} segments/s", file=sys.stderr, flush=True)
            log.append({"step": step, "loss": float(loss.item()), "segments_per_s": rate, "t": now})
            t_log, n_seen = now, 0
        if step >= next_val:
            next_val = step + val_steps
            act, v = run_validation()
            if act == "stop":
                return finish("plateau", v)
            if step >= total:
                return finish("max_epochs", v)
        elif step >= total:
            return finish("max_epochs", run_validation()[1])
        if time.time() > deadline:
            return finish("purge_deadline", run_validation()[1])
        if stop["now"] or now - t_ck > 60 * args.ckpt_minutes or now - t_start > 3600 * args.hours:
            checkpoint()
            t_ck = time.time()
            if stop["now"] or now - t_start > 3600 * args.hours:
                print(f"stopping at step {step} (time limit); resubmit to continue", file=sys.stderr)
                return


if __name__ == "__main__":
    main()
