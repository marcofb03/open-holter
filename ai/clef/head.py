"""Small arrhythmia head on frozen CLEF-Medium embeddings, trained on tuning data only.

Input: the 1,024-number CLEF embedding of each 10 s window (ai/benchmark/
frontend.py). Output: 8 window probabilities, the benchmark's findings
(ai/benchmark/findings.py HEAD_TARGETS: vt, af, svt, pvc, pac, pause, brady,
tachy). Model: LayerNorm -> 1024x256 -> GELU -> dropout 0.2 -> 256x8, masked
binary cross-entropy (a window with an ambiguous or unavailable target does
not count for that target), positive weight neg/pos capped at 20 (recall
first). AdamW, 6 epochs, batch 512.

Only tuning patients are used (datasets.py split "tune"). To keep the router
from picking CLEF because the head has memorised its own training windows,
the tuning-set scores are cross-fitted: 5 patient-level folds, each fold
scored by a head trained on the other four. Held-out and noise records are
scored by a head trained on all tuning patients. Scores are written to
data/work/bench2/clef_head/<db>/<rec>.npy (float16, windows x 8); the final
head to weights/clef_head.pt (git-ignored), CV AUROC per target to
../benchmark/results/clef_head.json.

    python head.py            # from ai/clef/, after ai/benchmark/frontend.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmark"
sys.path[:0] = [str(BENCH)]
import datasets as ds  # noqa: E402
import findings as F  # noqa: E402
import frontend as fe  # noqa: E402

OUT = fe.CACHE / "clef_head"
K = len(F.HEAD_TARGETS)
FOLDS, EPOCHS, BATCH, LR = 5, 6, 512, 1e-3


def build():
    import torch.nn as nn

    return nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, K))


def fold_of(r: ds.Rec) -> int:
    return int(hashlib.sha256(f"fold:{r.db}:{r.patient}".encode()).hexdigest(), 16) % FOLDS


def load_xy(recs: list[ds.Rec]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, gid = [], [], []
    for i, r in enumerate(recs):
        d = fe.load(r, emb=True)
        t = F.window_targets(d)
        xs.append(d["clef"].astype(np.float16))
        ys.append(np.stack([t[k] for k in F.HEAD_TARGETS], 1).astype(np.float32))
        gid.append(np.full(len(d["clef"]), i, np.int32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gid)


def train(x: np.ndarray, y: np.ndarray, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build()
    m = ~np.isnan(y)
    pos = np.nansum(y, 0)
    neg = m.sum(0) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 20), dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * (len(x) // BATCH + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
    xt = torch.from_numpy(x)  # float16: converted batch by batch (keeps memory at ~1 GB)
    yt = torch.from_numpy(np.nan_to_num(y))
    mt = torch.from_numpy(m.astype(np.float32))
    net.train()
    for _ in range(EPOCHS):
        for b in np.array_split(rng.permutation(len(x)), max(len(x) // BATCH, 1)):
            b = torch.from_numpy(b)
            loss = (lossf(net(xt[b].float()), yt[b]) * mt[b]).sum() / mt[b].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


def predict(net, x: np.ndarray) -> np.ndarray:
    import torch

    out = []
    with torch.inference_mode():
        for i in range(0, len(x), 65536):
            out.append(torch.sigmoid(net(torch.from_numpy(x[i:i + 65536].astype(np.float32)))).numpy())
    return np.concatenate(out) if out else np.zeros((0, K), np.float32)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    m = ~np.isnan(y)
    return float(roc_auc_score(y[m], s[m])) if 0 < y[m].sum() < m.sum() else float("nan")


def save(r: ds.Rec, p: np.ndarray) -> None:
    out = OUT / r.db / f"{r.rec}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, p.astype(np.float16))


def main() -> None:
    import torch

    torch.set_num_threads(4)
    allrecs = [r for r in ds.records() if fe.cache_path(r).exists()]
    tune = [r for r in allrecs if r.split == "tune"]
    print(f"{len(tune)} tuning records of {len(allrecs)} processed", file=sys.stderr)
    x, y, gid = load_xy(tune)
    folds = np.array([fold_of(r) for r in tune])[gid]
    oof = np.zeros_like(y)
    for k in range(FOLDS):
        net = train(x[folds != k], y[folds != k], seed=k)
        oof[folds == k] = predict(net, x[folds == k])
        print(f"fold {k}: {int((folds == k).sum())} windows", file=sys.stderr)
    cv = {t: {"AUROC": auroc(y[:, i], oof[:, i]), "pos": int(np.nansum(y[:, i])), "n": int((~np.isnan(y[:, i])).sum())}
          for i, t in enumerate(F.HEAD_TARGETS)}
    for i, r in enumerate(tune):
        save(r, oof[gid == i])
    net = train(x, y, seed=99)
    (HERE / "weights").mkdir(exist_ok=True)
    torch.save(net.state_dict(), HERE / "weights" / "clef_head.pt")
    del x, y
    for r in allrecs:
        if r.split != "tune":
            save(r, predict(net, fe.load(r, emb=True)["clef"]))
    (BENCH / "results").mkdir(exist_ok=True)
    (BENCH / "results" / "clef_head.json").write_text(json.dumps(
        {"tuning_records": len(tune), "windows": int(len(gid)), "cv_auroc": cv}, indent=1))
    print(json.dumps(cv, indent=1))


if __name__ == "__main__":
    main()
