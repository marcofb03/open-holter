"""Large HDF5 shards (few files, as the cluster scratch file system wants) and a resumable reader.

One shard = one array task x one split, e.g. shards/icentia11k/train/task-0007-of-0300.h5:
    win_x  (n, Lw) int16 uV       win_y (n, 9) int8     win_pid (n,) int32   win_t (n,) float32
    beat_x (m, Lb) int16 uV       beat_y (m,) int8      beat_pid (m,) int32  beat_t (m,) float32
    attrs: complete=1, source, split, task, n_tasks, fs, margin_s, patients (json), version
x is the raw signal at the source rate fs with a margin on both sides (h10.py:
Lw = (10 s + 2 x 1 s) x fs, Lb = (4.92 s + 2 x 0.75 s) x fs); h10.convert makes
the 130 Hz H10-like example when it is read.
A shard is written to <name>.tmp and renamed only when complete, so a task that
dies leaves no half-written shard and re-running it is safe (idempotent).

ChunkStream reads fixed-size chunks in a seeded, per-epoch order. Its state is
(epoch, chunk) -- two integers -- so a resumed job continues exactly where the
checkpoint left off, without re-reading what it had already used.
"""

from __future__ import annotations

import json
import os
import queue
from pathlib import Path

import numpy as np


def write_shard(path: Path, arrays: dict[str, np.ndarray], attrs: dict) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in arrays.items():
            kw = {"chunks": (min(len(v), 256),) + v.shape[1:], "compression": "lzf"} if v.ndim == 2 and len(v) else {}
            f.create_dataset(k, data=v, **kw)
        for k, v in attrs.items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        f.attrs["complete"] = 1
    os.replace(tmp, path)


def is_complete(path: Path) -> bool:
    import h5py

    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return int(f.attrs.get("complete", 0)) == 1
    except OSError:
        return False


def concat(parts: list[dict]) -> dict[str, np.ndarray]:
    keys = parts[0].keys() if parts else []
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


class ChunkStream:
    """Deterministic, resumable stream of converted (x, y) chunks from many shards.

    kind: "win" or "beat". Each epoch visits every (shard, chunk) once in an order
    drawn from seed + epoch; rows inside a chunk are shuffled the same way.
    transform(x_int16, y, fs, rng, uid=, epoch=) -> (x, y) runs in `threads` prefetch threads
    (numpy / scipy release the GIL); rng is seeded from (seed, epoch, chunk), so
    a resumed run sees exactly the augmentation it would have seen.
    """

    def __init__(self, files: list[Path], kind: str, transform, chunk: int = 1024, seed: int = 0,
                 threads: int = 4, extra: tuple[str, ...] = ()):
        import h5py

        self.files, self.kind, self.chunk, self.seed = sorted(files), kind, chunk, seed
        self.extra = extra  # more per-row arrays (f"{kind}_{name}") passed to transform as keywords
        self.transform, self.threads = transform, threads
        self.index, self.fs, self.n_rows = [], {}, 0
        for i, p in enumerate(self.files):
            with h5py.File(p, "r") as f:
                n = len(f[f"{kind}_y"])
                self.fs[i] = float(f.attrs.get("fs", 0.0))
            self.index += [(i, s) for s in range(0, n, chunk)]
            self.n_rows += n
        self.epoch, self.pos = 0, 0

    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos}

    def load_state(self, s: dict) -> None:
        self.epoch, self.pos = int(s["epoch"]), int(s["pos"])

    def _order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng(self.seed + 1000 * epoch).permutation(len(self.index))

    def read(self, k: int, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        import h5py

        i, s = self.index[k]
        with h5py.File(self.files[i], "r") as f:
            x = f[f"{self.kind}_x"][s:s + self.chunk]
            y = f[f"{self.kind}_y"][s:s + self.chunk]
            more = {e: f[f"{self.kind}_{e}"][s:s + self.chunk] for e in self.extra}
        rng = np.random.default_rng([self.seed, epoch, k])
        p = rng.permutation(len(y))
        uid = (np.uint64(i) << np.uint64(32)) + (s + p).astype(np.uint64)  # stable id of each example
        return self.transform(x[p], y[p], self.fs[i], rng, uid=uid, epoch=epoch, **{e: v[p] for e, v in more.items()})

    def __iter__(self):
        """Yields converted chunks; (epoch, pos) advances as each chunk is handed out."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.threads) as ex:
            while True:
                order, epoch = self._order(self.epoch), self.epoch
                ahead: queue.Queue = queue.Queue()
                nxt = self.pos
                for _ in range(2 * self.threads):
                    if nxt < len(order):
                        ahead.put(ex.submit(self.read, int(order[nxt]), epoch))
                        nxt += 1
                while not ahead.empty():
                    item = ahead.get().result()
                    if nxt < len(order):
                        ahead.put(ex.submit(self.read, int(order[nxt]), epoch))
                        nxt += 1
                    self.pos += 1
                    yield item
                self.epoch, self.pos = self.epoch + 1, 0


class Noise:
    """NSTDB noise (bw, em, ma at 130 Hz, uV) for augmentation; see prep.py noise."""

    def __init__(self, path: Path):
        import h5py

        with h5py.File(path, "r") as f:
            self.sig = {k: f[k][:].astype(np.float32) for k in ("bw", "em", "ma")}

    def add(self, x: np.ndarray, rng: np.random.Generator, snr_db: np.ndarray) -> np.ndarray:
        """x (n, L) float uV; noise at per-row SNR (S = mean square of the row after its mean)."""
        out = x.copy()
        n, L = x.shape
        kinds = rng.choice(list(self.sig), n)
        for i in range(n):
            s = self.sig[kinds[i]]
            a = rng.integers(0, len(s) - L)
            nz = s[a:a + L] - s[a:a + L].mean()
            ps = np.mean((x[i] - x[i].mean()) ** 2) + 1e-6
            g = np.sqrt(ps / (np.mean(nz ** 2) + 1e-6) / 10 ** (snr_db[i] / 10))
            out[i] += g * nz
        return out
