"""The three models of the fine-tuning kit, with their input preprocessing.

    ecgfounder  ECGFounder 1-lead (30.8 M parameters, MIT), pre-trained 150-label head
                replaced by 9 window targets (h10.WINDOW_TARGETS); input: the 10 s
                130 Hz window taken to 500 Hz, ECGFounder's own filtering and z-score
    clef        CLEF-Medium (30.7 M, same Net1D backbone) + a 9-target linear head;
                input: 500 Hz, CLEF's 0.67-40 Hz band-pass and z-score
    beat        heartKIT-style beat classifier trained from scratch at 130 Hz:
                640-sample (4.9 s) beat-centred frame, layer-normed as heartKIT does,
                small MBConv network (0.43 M parameters, phone-sized), 4 classes
                N / S / V / noise (the added noise class)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AI = HERE.parent
sys.path[:0] = [str(HERE), str(AI / "ecgfounder"), str(AI / "ecgfounder" / "third_party"), str(AI / "clef")]
import h10  # noqa: E402

N_WIN = len(h10.WINDOW_TARGETS)
N_BEAT = len(h10.BEAT_CLASSES)
KIND = {"ecgfounder": "win", "clef": "win", "beat": "beat"}


def build(name: str, pretrained: bool = True):
    import torch.nn as nn

    if name == "ecgfounder":
        import infer

        net = infer.load_torch_model(allow_random=not pretrained) if pretrained else infer.build_net()
        net.dense = nn.Linear(net.dense.in_features, N_WIN)
        return net
    if name == "clef":
        import clef

        net = clef.load_backbone() if pretrained else clef.build_backbone()
        net.dense = nn.Linear(clef.DIM, N_WIN)
        return net
    if name == "beat":
        return BeatNet()
    raise ValueError(name)


def preprocess(name: str, x130: np.ndarray) -> np.ndarray:
    """(n, L) float uV at 130 Hz -> model input (n, 1, L') float32."""
    if name == "beat":
        x = x130 - x130.mean(axis=1, keepdims=True)
        return (x / np.sqrt(x.var(axis=1, keepdims=True) + 0.01 * 1e6))[:, None, :].astype(np.float32)  # LN, eps 0.01 mV^2
    import infer

    x500 = infer.resample_to_500(x130 / 1000.0, h10.FS) if x130.ndim == 1 else np.stack(
        [infer.resample_to_500(r / 1000.0, h10.FS) for r in x130])
    x500 = x500[:, :infer.WIN]
    if name == "ecgfounder":
        return infer.preprocess_windows(x500)
    import clef

    return clef.preprocess_windows(x500)


def _beat_layers():
    import torch
    import torch.nn as nn

    class SE(nn.Module):
        def __init__(self, c: int, r: int = 4):
            super().__init__()
            self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(c, max(c // r, 4), 1), nn.SiLU(),
                                    nn.Conv1d(max(c // r, 4), c, 1), nn.Sigmoid())

        def forward(self, x):
            return x * self.fc(x)

    class MBConv(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int, expand: int = 4, k: int = 5):
            super().__init__()
            mid = cin * expand
            self.block = nn.Sequential(
                nn.Conv1d(cin, mid, 1, bias=False), nn.BatchNorm1d(mid), nn.SiLU(),
                nn.Conv1d(mid, mid, k, stride, k // 2, groups=mid, bias=False), nn.BatchNorm1d(mid), nn.SiLU(),
                SE(mid), nn.Conv1d(mid, cout, 1, bias=False), nn.BatchNorm1d(cout))
            self.skip = stride == 1 and cin == cout

        def forward(self, x):
            y = self.block(x)
            return x + y if self.skip else y

    return torch, nn, MBConv


class BeatNet:  # built lazily so importing this module does not need torch
    def __new__(cls, widths=(24, 32, 48, 64, 96), strides=(2, 2, 2, 2)):
        torch, nn, MBConv = _beat_layers()
        layers = [nn.Conv1d(1, widths[0], 7, 2, 3, bias=False), nn.BatchNorm1d(widths[0]), nn.SiLU()]
        for cin, cout, s in zip(widths[:-1], widths[1:], strides):
            layers += [MBConv(cin, cout, s), MBConv(cout, cout, 1)]
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(widths[-1], N_BEAT)]
        return nn.Sequential(*layers)
