"""Rhythm transformer: CNN stem + transformer encoder over 60 s of H10-like ECG (../TRANSFORMER_PLAN.md).

    input  (B, 2, 7800): z-scored 130 Hz signal and validity mask (seg.to_input)
    stem   4 strided convolutions, x16 -> T = 488 tokens of 16 samples (~0.12 s)
    body   pre-norm transformer encoder, learned positions
    heads  qrs   (B, T)      is there a QRS in this token
           beat  (B, T, 4)   N / S / V / unreadable, read at the beat's token
           win   (B, 6, 10)  the 10 s targets of seg.SEG_TARGETS, one row per sixth
           recon (B, T, 16)  the token's 16 samples (masked pretraining)
"""

from __future__ import annotations

SIZES = {  # width, layers, heads
    "tiny": (64, 2, 2),       # tests
    "small": (256, 6, 4),
    "base": (384, 8, 6),      # ~15 M parameters
    "large": (512, 12, 8),    # ~40 M parameters
}
PATCH = 16
N_BEAT = 4
N_SUB = 6


def build(size: str = "base", n_targets: int = 10, max_tokens: int = 512):
    import torch
    import torch.nn as nn

    d, layers, heads = SIZES[size]

    def block(cin, cout, k):
        return nn.Sequential(nn.Conv1d(cin, cout, k, 2, k // 2), nn.BatchNorm1d(cout), nn.GELU())

    class RhythmNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = max(d // 4, 32)
            self.stem = nn.Sequential(block(2, w, 15), block(w, 2 * w, 9), block(2 * w, 4 * w, 9),
                                      nn.Conv1d(4 * w, d, 5, 2, 2))
            self.pos = nn.Parameter(torch.zeros(1, max_tokens, d))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.mask_tok = nn.Parameter(torch.zeros(d))
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.qrs = nn.Linear(d, 1)
            self.beat = nn.Linear(d, N_BEAT)
            self.win = nn.Linear(d, n_targets)
            self.recon = nn.Linear(d, PATCH)

        def forward(self, x, token_mask=None):
            h = self.stem(x).transpose(1, 2)  # (B, T, d)
            if token_mask is not None:
                h = torch.where(token_mask[..., None], self.mask_tok.to(h.dtype), h)
            h = self.norm(self.enc(h + self.pos[:, :h.shape[1]]))
            sub = nn.functional.adaptive_avg_pool1d(h.transpose(1, 2), N_SUB).transpose(1, 2)
            return {"qrs": self.qrs(h)[..., 0], "beat": self.beat(h), "win": self.win(sub), "recon": self.recon(h)}

    return RhythmNet()


def n_tokens(n_samples: int) -> int:
    t = n_samples
    for _ in range(4):
        t = (t - 1) // 2 + 1  # stride 2, "same" padding
    return t
