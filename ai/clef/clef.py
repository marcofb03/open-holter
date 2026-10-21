"""CLEF-Medium single-lead ECG foundation model as a window embedder.

CLEF (Shu et al., arXiv:2512.02180; Nokia Bell Labs, code under the Clear BSD
licence, see third_party/) is a contrastively pre-trained single-lead encoder.
CLEF-Medium is the same Net1D backbone as ECGFounder (30.7 M parameters, 64
base filters, 7 stages) with the classification layer removed: a 10 s, 500 Hz
window maps to a 1,024-number embedding (mean over time of the last stage).
It has no arrhythmia labels of its own; ai/clef/head.py trains a small head on
the benchmark's tuning data only.

Preprocessing follows the CLEF repository (clef/data/*_dataloader.py,
commit 0a1156e): resample to 500 Hz, 5th-order Butterworth band-pass
0.67-40 Hz with zero-phase filtfilt, then z-score, per 10 s window.

The backbone code is ECGFounder's net1d.py (vendored in ai/ecgfounder/
third_party, MIT); CLEF's own CLEF.py is an adaptation of the same file and
loads the same state dict (checked: identical outputs, tests/test_clef.py
has the offline part of that check).

    python download_weights.py          # weights/clef_medium.ckpt (368 MB, Zenodo 17572734)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
ECGF = HERE.parent / "ecgfounder"
DEFAULT_CKPT = HERE / "weights" / "clef_medium.ckpt"
FS = 500
WIN = 5000
DIM = 1024


def build_backbone():
    """CLEF-Medium = ECGFounder's Net1D without its dense layer."""
    import torch.nn as nn

    sys.path[:0] = [str(ECGF), str(ECGF / "third_party")]
    import infer  # ai/ecgfounder: same hyper-parameters as CLEF create_net1d_by_size("medium")

    net = infer.build_net()
    net.dense = nn.Identity()
    return net


def load_backbone(ckpt: Path = DEFAULT_CKPT):
    import torch

    net = build_backbone()
    if not Path(ckpt).exists():
        raise FileNotFoundError(f"{ckpt} not found - run download_weights.py first")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)  # plain tensors only
    sd = {k.removeprefix("backbone."): v for k, v in sd.get("state_dict", sd).items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return net.eval()


def preprocess_windows(windows: np.ndarray) -> np.ndarray:
    """(n, 5000) raw 500 Hz windows -> (n, 1, 5000) float32 CLEF input."""
    x = np.nan_to_num(np.asarray(windows, np.float64))
    b, a = butter(5, [0.67 / (FS / 2), 40 / (FS / 2)], btype="band")
    x = filtfilt(b, a, x, axis=-1)
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    return x[:, None, :].astype(np.float32)


class Embedder:
    """(n, 1, 5000) preprocessed windows -> (n, 1024) float32 embeddings.

    bf16=True runs the convolutions in bfloat16 (CPU autocast; about 2x faster on
    CPUs with AMX/AVX-512 BF16). The drift is measured in ai/benchmark/RESULTS.md.
    """

    def __init__(self, ckpt: Path = DEFAULT_CKPT, threads: int | None = None, batch: int = 32,
                 bf16: bool = False):
        import torch

        if threads:
            torch.set_num_threads(threads)
        self.net = load_backbone(ckpt)
        self.batch, self.bf16 = batch, bf16

    def __call__(self, x: np.ndarray) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            for i in range(0, len(x), self.batch):
                out.append(self.net(torch.from_numpy(x[i:i + self.batch])).float().numpy())
        return np.concatenate(out) if out else np.zeros((0, DIM), np.float32)
