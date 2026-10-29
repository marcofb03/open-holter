"""Per-label TPR / FPR on the fixed validation set, before and after fine-tuning.

    python eval_labels.py --shards $SHARDS --out $SCR/eval_labels --bf16 --threads 8 \
        --system ecgfounder-pretrained clef-probe \
                 ecgfounder=$RUNS/ecgfounder/best.pt clef=$RUNS/clef/best.pt beat=$RUNS/beat/best.pt

Systems
    ecgfounder-pretrained  ECGFounder 1-lead as released (150-label head, not fine-tuned). A target's
                           score is the max probability of its label group, the benchmark router's
                           mapping (ai/benchmark/router.py ECGF_GROUP). It has no noise label.
    clef-probe             frozen CLEF-Medium + a small head (ai/clef/head.py architecture and recipe)
                           trained here on a stratified sample of TRAIN-split windows: CLEF before
                           its backbone is fine-tuned.
    NAME=PATH              a train.py checkpoint of model NAME (ecgfounder, clef, beat); NAME@TAG=PATH
                           names it TAG in the results.

Metrics, per window target (vt ... noise) or per beat class against normal beats (V, S, noise):
n_pos, n_neg, AUROC, TPR at FPR 5 / 1 / 0.1 % and TPR, FPR at the plain threshold 0.5. A window
whose label for a target is unknown (-1) is left out for that target. The thresholds at a given
FPR are set on this same set, which flatters every system a little; the held-out benchmark fixes
them in advance. The set is enriched with rare events and its negatives include other
arrhythmias, so FPR here is not the false-alarm rate of an ordinary day.

Writes <out>/labels.json and <out>/<tag>.npz (scores and labels), so other operating points can
be computed later without running the models again.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "ecgfounder"), str(HERE.parent / "clef")]
import h10  # noqa: E402
import models  # noqa: E402
import shards  # noqa: E402
import valset  # noqa: E402

FPRS = (0.05, 0.01, 0.001)
# ai/benchmark/router.py ECGF_GROUP (copied: importing the router pulls in the whole benchmark)
ECGF_GROUP = {"vt": "vt_or_wct", "af": "af_or_afl", "svt": "svt", "pvc": "pvc", "pac": "pac",
              "pause": "pause_or_block", "brady": "sinus_brady", "tachy": "sinus_tachy"}
BEAT_EVAL = {"V": 2, "S": 1, "noise": 3}  # class index, scored against N (0) as in train.py
PROBE_EPOCHS, PROBE_BATCH, PROBE_LR = 6, 512, 1e-3  # ai/clef/head.py


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata

    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))[: len(pos)]
    return float((r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(pos: np.ndarray, neg: np.ndarray) -> dict:
    out = {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "auroc": auroc(pos, neg)}
    for f in FPRS:
        out[f"tpr_at_fpr{f:g}"] = float((pos > np.quantile(neg, 1 - f)).mean()) if len(pos) and len(neg) else float("nan")
    out["tpr_at_0.5"] = float((pos > 0.5).mean()) if len(pos) else float("nan")
    out["fpr_at_0.5"] = float((neg > 0.5).mean()) if len(neg) else float("nan")
    return out


def score_table(kind: str, y: np.ndarray, s: np.ndarray) -> dict:
    """{target: metrics}; NaN scores (a target the system lacks) give no entry."""
    res = {}
    if kind == "win":
        for i, t in enumerate(h10.WINDOW_TARGETS):
            if np.isnan(s[:, i]).all():
                continue
            res[t] = metrics(s[y[:, i] == 1, i], s[y[:, i] == 0, i])
    else:
        for t, c in BEAT_EVAL.items():
            res[t] = metrics(s[y == c, c], s[y == 0, c])
    return res


def parse_system(spec: str) -> tuple[str, str, Path | None]:
    """-> (tag, model, checkpoint)"""
    if "=" not in spec:
        if spec not in ("ecgfounder-pretrained", "clef-probe"):
            raise SystemExit(f"unknown system {spec}")
        return spec, spec, None
    name, path = spec.split("=", 1)
    model, _, tag = name.partition("@")
    if model not in models.KIND:
        raise SystemExit(f"unknown model {model} in {spec}")
    return tag or f"{model}-finetuned", model, Path(path)


def load_finetuned(model: str, ckpt: Path):
    import torch

    net = models.build(model, pretrained=False)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["model"])
    return net.eval()


def run_nets(nets: dict, x: np.ndarray, bf16: bool, batch: int = 64) -> dict:
    import torch

    out = {k: [] for k in nets}
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch])
            for k, net in nets.items():
                out[k].append(net(xb).float().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def window_scores(systems: list, vpath: Path, bf16: bool, limit: int | None) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """One pass over the window valset: every window system's (n, 9) scores; frozen CLEF embeddings."""
    by_pre = {"ecgfounder": {}, "clef": {}}
    for tag, model, ckpt in systems:
        if tag == "ecgfounder-pretrained":
            import infer

            by_pre["ecgfounder"][tag] = infer.load_torch_model()
        elif tag == "clef-probe":
            import clef

            by_pre["clef"][tag] = clef.load_backbone()
        else:
            by_pre[model][tag] = load_finetuned(model, ckpt)
    vs = valset.ValSet(vpath, "win", lambda x, y, fs, rng, **kw: (h10.convert(x, fs, "win"), y), chunk=512)
    ys, raw = [], {tag: [] for p in by_pre.values() for tag in p}
    n_chunks = len(vs.index) if limit is None else min(limit, len(vs.index))
    t0 = time.time()
    for k in range(n_chunks):
        x130, y = vs.read(k)
        ys.append(y)
        for pre, nets in by_pre.items():
            if nets:
                for tag, o in run_nets(nets, models.preprocess(pre, x130), bf16).items():
                    raw[tag].append(o.astype(np.float32))
        if k % 20 == 0:
            print(f"  windows: chunk {k + 1}/{n_chunks}, {(k + 1) * 512 / (time.time() - t0):.0f} windows/s",
                  file=sys.stderr, flush=True)
    y = np.concatenate(ys)
    scores, emb = {}, None
    for tag, parts in raw.items():
        o = np.concatenate(parts)
        if tag == "ecgfounder-pretrained":
            import infer

            p = 1 / (1 + np.exp(-o.astype(np.float64)))
            s = np.full((len(o), len(h10.WINDOW_TARGETS)), np.nan, np.float32)
            for t, g in ECGF_GROUP.items():
                s[:, h10.WINDOW_TARGETS.index(t)] = p[:, infer.GROUP_IDX[g]].max(1)
            scores[tag] = s
        elif tag == "clef-probe":
            emb = o.astype(np.float16)
        else:
            scores[tag] = (1 / (1 + np.exp(-o.astype(np.float64)))).astype(np.float32)
    return y, scores, emb


def probe_sample(shard_root: Path, sources: list[str], n_rare: int, n_rest: int, seed: int = 7):
    """Stratified TRAIN-split windows: up to n_rare with a rare positive + n_rest others -> (x130, y)."""
    import h5py

    files = [p for s in sources for p in sorted((shard_root / s / "train").glob("*.h5")) if shards.is_complete(p)]
    rare, rest = [], []
    for fi, p in enumerate(files):
        with h5py.File(p, "r") as f:
            if "win_y" not in f or not len(f["win_y"]):
                continue
            r = valset.rare_rows("win", f["win_y"][:])
        rare += [(fi, int(i)) for i in np.where(r)[0]]
        rest += [(fi, int(i)) for i in np.where(~r)[0]]
    rng = np.random.default_rng(seed)
    pick = lambda rows, n: [rows[i] for i in rng.choice(len(rows), min(n, len(rows)), replace=False)] if rows else []
    rows = sorted(pick(rare, n_rare) + pick(rest, n_rest))
    xs, ys = [], []
    for fi in sorted({r[0] for r in rows}):
        idx = np.array([i for f_, i in rows if f_ == fi])
        with h5py.File(files[fi], "r") as f:
            xs.append(h10.convert(f["win_x"][idx], float(f.attrs["fs"]), "win"))
            ys.append(f["win_y"][idx])
    print(f"  probe sample: {len(rows)} train windows ({min(n_rare, len(rare))} rare) from {len(files)} shards",
          file=sys.stderr, flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def train_probe(emb: np.ndarray, y: np.ndarray, seed: int = 0):
    """ai/clef/head.py recipe: LayerNorm -> 1024x256 -> GELU -> dropout -> 256xK, masked BCE, pos weight <= 20."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    K = y.shape[1]
    net = nn.Sequential(nn.LayerNorm(emb.shape[1]), nn.Linear(emb.shape[1], 256), nn.GELU(), nn.Dropout(0.2),
                        nn.Linear(256, K))
    m = y >= 0
    pos = (y == 1).sum(0)
    pw = torch.tensor(np.clip((m.sum(0) - pos) / np.maximum(pos, 1), 1, 20), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    opt = torch.optim.AdamW(net.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    nb = max(len(emb) // PROBE_BATCH, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, PROBE_LR, total_steps=PROBE_EPOCHS * nb)
    xt, yt = torch.from_numpy(emb), torch.from_numpy(np.clip(y, 0, 1).astype(np.float32))
    mt = torch.from_numpy(m.astype(np.float32))
    net.train()
    for _ in range(PROBE_EPOCHS):
        for b in np.array_split(rng.permutation(len(emb)), nb):
            b = torch.from_numpy(b)
            loss = (lossf(net(xt[b].float()), yt[b]) * mt[b]).sum() / mt[b].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return lambda e: torch.sigmoid(net(torch.from_numpy(e).float())).detach().numpy()


def beat_scores(systems: list, vpath: Path, limit: int | None) -> tuple[np.ndarray, dict]:
    import torch

    nets = {tag: load_finetuned(model, ckpt) for tag, model, ckpt in systems}
    vs = valset.ValSet(vpath, "beat", lambda x, y, fs, rng, **kw: (models.preprocess("beat", h10.convert(x, fs, "beat")), y),
                       chunk=4096)
    n_chunks = len(vs.index) if limit is None else min(limit, len(vs.index))
    ys, out = [], {k: [] for k in nets}
    for k in range(n_chunks):
        x, y = vs.read(k)
        ys.append(y)
        for tag, o in run_nets(nets, x, False, batch=4096).items():
            out[tag].append(torch.softmax(torch.from_numpy(o), 1).numpy())
    return np.concatenate(ys), {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--system", nargs="+", required=True)
    ap.add_argument("--probe-sources", nargs="+", default=None, help="default: train.py DEFAULT_SOURCES['win']")
    ap.add_argument("--probe-rare", type=int, default=40_000)
    ap.add_argument("--probe-rest", type=int, default=40_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="valset chunks (testing)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    systems = [parse_system(s) for s in a.system]
    win = [s for s in systems if s[1] != "beat"]
    beat = [s for s in systems if s[1] == "beat"]
    res = {"valset": {}, "systems": {}, "fprs": list(FPRS)}
    if win:
        vpath = a.shards / "valset_win.h5"
        res["valset"]["win"] = valset.ValSet(vpath, "win", None).id
        y, scores, emb = window_scores(win, vpath, a.bf16, a.limit)
        if emb is not None:
            import clef

            from train import DEFAULT_SOURCES

            xp, yp = probe_sample(a.shards, a.probe_sources or DEFAULT_SOURCES["win"], a.probe_rare, a.probe_rest)
            backbone = clef.load_backbone()
            ep = run_nets({"e": backbone}, models.preprocess("clef", xp), a.bf16)["e"].astype(np.float16)
            scores["clef-probe"] = train_probe(ep, yp)(emb).astype(np.float32)
        for tag, s in scores.items():
            np.savez_compressed(a.out / f"{tag}.npz", y=y, s=s.astype(np.float16))
            res["systems"][tag] = {"kind": "win", "labels": score_table("win", y, s)}
    if beat:
        vpath = a.shards / "valset_beat.h5"
        res["valset"]["beat"] = valset.ValSet(vpath, "beat", None).id
        y, scores = beat_scores(beat, vpath, a.limit)
        for tag, s in scores.items():
            np.savez_compressed(a.out / f"{tag}.npz", y=y, s=s.astype(np.float16))
            res["systems"][tag] = {"kind": "beat", "labels": score_table("beat", y, s)}
    for tag, model, ckpt in systems:
        res["systems"][tag]["checkpoint"] = str(ckpt) if ckpt else None
    (a.out / "labels.json").write_text(json.dumps(res, indent=1))
    for tag, r in res["systems"].items():
        print(tag, json.dumps({t: {k: round(v, 3) for k, v in m.items() if k != "n_neg"} for t, m in r["labels"].items()}))


if __name__ == "__main__":
    main()
