"""Ambiq heartKIT models (BSD-3-Clause) on single-lead signals, as heartKIT runs them.

Two pre-trained TFLite models from the heartKIT model zoo:
    beat    BEAT-3-EFF-SM  normal / PAC / PVC for one beat, 512 samples at 100 Hz
                           (5.12 s) centred on the beat; trained on Icentia11k.
    rhythm  ARR-2-EFF-SM   normal vs AF/AFL, 512-sample frames at 100 Hz;
                           trained on PTB-XL + LSAD (12-lead hospital ECGs).

Preprocessing follows heartKIT (heartkit/tasks/beat/demo.py and dataloaders,
heartkit/tasks/rhythm/demo.py, helia_edge LayerNormalization1D,
physiokit resample_signal, helia_edge TfLite interpreter) exactly:
  * cut ceil(fs / 100 * 512) samples at the source rate, beat at the centre;
  * resample with scipy.signal.resample (FFT) to round(n * 100 / fs), keep 512;
  * no filtering; layer norm per frame: (x - mean) / sqrt(var + 0.01);
  * int8 models: x / scale + zero_point, cast to int8 (we clip to the int8
    range first so an extreme frame cannot wrap around); outputs dequantised;
  * class probabilities from the outputs (see _to_probs: the published files
    apply softmax twice, which we undo instead of adding a third); beat:
    argmax, but a non-normal class below probability 0.5 falls back to normal;
    AF when its probability >= 0.75 (heartKIT's configured thresholds).
Deviations: the softmax handling above, and a beat closer than 2.56 s to
either end of the signal is classified on an edge-padded frame (heartKIT's
demo skips it).

Get the models with `python download_models.py` (git-ignored weights/).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
MODELS = {"beat": "beat-3-eff-sm", "rhythm": "arr-2-eff-sm"}
FS = 100
FRAME = 512
LN_EPS = 0.01
BEAT_LABELS = np.array(["N", "S", "V"])  # heartKIT classes QRS / PAC / PVC
BEAT_THRESHOLD = 0.5
AF_THRESHOLD = 0.75  # ARR-2-EFF-SM configuration.json "threshold"


def _interpreter(name: str):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # full TensorFlow instead of LiteRT
        from tensorflow.lite.python.interpreter import Interpreter

    path = WEIGHTS / MODELS[name] / "model.tflite"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run download_models.py first")
    it = Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()  # fixed batch of 1 (the graph does not resize)
    return it


def config(name: str) -> dict:
    return json.loads((WEIGHTS / MODELS[name] / "configuration.json").read_text())


def frames(x: np.ndarray, fs: float, centres: np.ndarray) -> np.ndarray:
    """(n, 512) layer-normed 100 Hz frames, each centred on a sample index of x (at fs)."""
    n_in = int(np.ceil(fs / FS * FRAME))
    n_out = int(np.round(n_in * FS / fs))
    half = int(0.5 * n_in)
    xp = np.pad(np.nan_to_num(np.asarray(x, np.float64)), (n_in, n_in), mode="edge")
    starts = np.asarray(centres, int) - half + n_in
    seg = np.stack([xp[s:s + n_in] for s in starts]) if len(starts) else np.zeros((0, n_in))
    seg = resample(seg, n_out, axis=1)[:, :FRAME].astype(np.float32)
    mean = seg.mean(axis=1, keepdims=True)
    var = seg.var(axis=1, keepdims=True)
    return (seg - mean) / np.sqrt(var + LN_EPS)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _to_probs(out: np.ndarray) -> np.ndarray:
    """Class probabilities from the TFLite output.

    The published zoo files already end in softmax, and their rows are a
    softmax of a probability vector (very flat: a clear PVC reads [0.23 0.28
    0.49]). For q = softmax(p) with sum(p) = 1, p = log q + (1 - sum log q) / K
    recovers p exactly (checked: recovered rows are valid distributions, e.g.
    [0.00 0.22 0.78] for that PVC). heartKIT's demo applies yet another softmax
    and then a 0.5 threshold, which on these files labels every beat normal.
    Raw logits (rows not summing to 1) get a plain softmax.
    """
    if np.allclose(out.sum(axis=1), 1.0, atol=0.02) and (out > 0).all():
        lq = np.log(out)
        p = lq + (1.0 - lq.sum(axis=1, keepdims=True)) / out.shape[1]
        return np.clip(p, 0.0, 1.0)
    return _softmax(out)


class Model:
    def __init__(self, name: str):
        self.name = name
        self.it = _interpreter(name)
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def probs(self, f: np.ndarray) -> np.ndarray:
        """(n, 512) normalised frames -> (n, classes) softmax probabilities."""
        out = np.zeros((len(f), self.out["shape"][-1]), np.float32)
        q_in, q_out = self.inp["quantization"], self.out["quantization"]
        xs = f[..., None]
        if q_in[0]:
            info = np.iinfo(self.inp["dtype"])
            xs = np.clip(xs / q_in[0] + q_in[1], info.min, info.max)
        xs = xs.astype(self.inp["dtype"])
        for i in range(len(xs)):
            self.it.set_tensor(self.inp["index"], xs[i:i + 1])
            self.it.invoke()
            out[i] = self.it.get_tensor(self.out["index"])[0]
        if q_out[0]:
            out = (out - q_out[1]) * q_out[0]
        return _to_probs(np.maximum(out, 1.0 / 256))  # int8 softmax floor: no log(0)


def classify_beats(x: np.ndarray, fs: float, beat_idx: np.ndarray, model: Model | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Labels ('N', 'S', 'V') and (n, 3) probabilities for beats at sample indices beat_idx."""
    model = model or Model("beat")
    p = model.probs(frames(x, fs, beat_idx))
    k = p.argmax(axis=1)
    k[(k > 0) & (p.max(axis=1) < BEAT_THRESHOLD)] = 0
    return BEAT_LABELS[k], p


def af_frames(x: np.ndarray, fs: float, step_s: float = 2.5, model: Model | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """AF/AFL probability per 5.12 s frame (frame start times in s, probabilities)."""
    model = model or Model("rhythm")
    n_in = int(np.ceil(fs / FS * FRAME))
    starts = np.arange(0, len(x) - n_in + 1, int(round(step_s * fs)))
    p = model.probs(frames(x, fs, starts + int(0.5 * n_in)))
    return starts / fs, p[:, 1]
