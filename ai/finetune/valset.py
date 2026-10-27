"""Fixed, stratified validation set: every rare event of the validation patients + a random sample.

    python valset.py win  --shards $SHARDS            # -> $SHARDS/valset_win.h5
    python valset.py beat --shards $SHARDS            # -> $SHARDS/valset_beat.h5
    python valset.py seg  --shards $SEG               # -> $SEG/valset_seg.h5 (rhythm transformer)

Why: a random sample of ~25,000 validation windows contained no VT window at all (2026-10-27,
ECGFounder and CLEF on Euler), so the checkpoint was chosen without looking at VT, and the
score jumped by up to 0.15 between validations. This set is built once from the val split
only (never test), keeps EVERY example with a rare positive and adds a fixed random sample of
the rest, so every validation sees the same examples and every target is measured.

    win   rare: vt, svt, pause, brady, tachy (h10.WINDOW_TARGETS) == 1; + `--n-rest` others
    beat  rare: classes S, V, unreadable; + `--n-rest` normal beats
    seg   rare: any of vt, svt, pause, brady, tachy, avb == 1 in a segment; + `--n-rest` others
Rows are copied with their source rate into one HDF5 file per source rate group ("fs" attr per
row block), written atomically; the set's id (hash of its rows) is stored so a resumed run can
tell that its yardstick changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import h10  # noqa: E402
import shards  # noqa: E402

RARE = {"win": ("vt", "svt", "pause", "brady", "tachy"), "seg": ("vt", "svt", "pause", "brady", "tachy", "avb")}
SEG_TARGETS = tuple(h10.WINDOW_TARGETS) + ("avb",)


def rare_rows(kind: str, y: np.ndarray) -> np.ndarray:
    if kind == "beat":
        return np.isin(y, (1, 2, 3))
    names = h10.WINDOW_TARGETS if kind == "win" else SEG_TARGETS
    idx = [names.index(t) for t in RARE[kind]]
    yy = y[:, idx] if kind == "win" else y[:, :, idx].reshape(len(y), -1)
    return (yy == 1).any(1)


def build(kind: str, root: Path, n_rest: int, seed: int = 20261027) -> Path:
    import h5py

    files = sorted(p for p in root.glob("*/val/*.h5") if shards.is_complete(p))
    extra = ("bt", "bc", "hb", "valid") if kind == "seg" else ()
    rare, rest = [], []
    for fi, p in enumerate(files):
        with h5py.File(p, "r") as f:
            if f"{kind}_y" not in f or not len(f[f"{kind}_y"]):
                continue
            y = f[f"{kind}_y"][:]
        r = rare_rows(kind, y)
        rare += [(fi, int(i)) for i in np.where(r)[0]]
        rest += [(fi, int(i)) for i in np.where(~r)[0]]
    rng = np.random.default_rng(seed)
    pick = [rest[i] for i in rng.choice(len(rest), min(n_rest, len(rest)), replace=False)] if rest else []
    rows = sorted(rare + pick)
    out = root / f"valset_{kind}.h5"
    tmp = out.with_suffix(".tmp")
    by_fs: dict[float, dict[str, list]] = {}
    for fi in sorted({r[0] for r in rows}):
        idx = np.array([i for f_, i in rows if f_ == fi])
        with h5py.File(files[fi], "r") as f:
            fs = float(f.attrs["fs"])
            d = by_fs.setdefault(fs, {k: [] for k in ("x", "y") + extra})
            d["x"].append(f[f"{kind}_x"][idx])
            d["y"].append(f[f"{kind}_y"][idx])
            for e in extra:
                d[e].append(f[f"{kind}_{e}"][idx])
    sid = hashlib.sha256(json.dumps([(files[f].name, f, i) for f, i in rows]).encode()).hexdigest()[:12]
    with h5py.File(tmp, "w") as f:
        for g, (fs, d) in enumerate(sorted(by_fs.items())):
            grp = f.create_group(f"fs{g}")
            grp.attrs["fs"] = fs
            for k, v in d.items():
                grp.create_dataset(f"{kind}_{k}", data=np.concatenate(v), compression="lzf" if k == "x" else None)
        f.attrs.update({"kind": kind, "id": sid, "n_rare": len(rare), "n_rest": len(pick), "complete": 1,
                        "rare_targets": json.dumps(RARE.get(kind, ("S", "V", "noise")))})
    os.replace(tmp, out)
    print(json.dumps({"kind": kind, "file": str(out), "id": sid, "rare": len(rare), "rest": len(pick),
                      "positives": positives(kind, out)}))
    return out


def positives(kind: str, path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        ys = [f[g][f"{kind}_y"][:] for g in f]
    y = np.concatenate(ys)
    if kind == "beat":
        return {c: int((y == i).sum()) for i, c in enumerate(h10.BEAT_CLASSES)}
    names = h10.WINDOW_TARGETS if kind == "win" else SEG_TARGETS
    yy = y if kind == "win" else (y == 1).any(1).astype(int)
    return {t: int((yy[:, i] == 1).sum()) for i, t in enumerate(names)}


class ValSet:
    """Reads a valset file in fixed chunks through the same transform as training (frame 0)."""

    def __init__(self, path: Path, kind: str, transform, chunk: int = 256, extra: tuple = ()):
        import h5py

        self.path, self.kind, self.transform, self.chunk, self.extra = path, kind, transform, chunk, extra
        with h5py.File(path, "r") as f:
            self.id = str(f.attrs["id"])
            self.index = [(g, s) for g in f for s in range(0, len(f[g][f"{kind}_y"]), chunk)]
            self.fs = {g: float(f[g].attrs["fs"]) for g in f}

    def read(self, k: int, epoch: int = 0):
        import h5py

        g, s = self.index[k]
        with h5py.File(self.path, "r") as f:
            x = f[g][f"{self.kind}_x"][s:s + self.chunk]
            y = f[g][f"{self.kind}_y"][s:s + self.chunk]
            more = {e: f[g][f"{self.kind}_{e}"][s:s + self.chunk] for e in self.extra}
        uid = np.arange(len(y), dtype=np.uint64)
        return self.transform(x, y, self.fs[g], np.random.default_rng(0), uid=uid, epoch=0, **more)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["win", "beat", "seg"])
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--n-rest", type=int, default=None, help="default: 40,000 (win), 60,000 (beat), 5,000 (seg)")
    a = ap.parse_args()
    build(a.kind, a.shards, a.n_rest or {"win": 40_000, "beat": 60_000, "seg": 5_000}[a.kind])


if __name__ == "__main__":
    main()
