"""Seeded synthetic classification task and NumPy training of the Couplet models
(DESIGN_M3.md section 5).

The task (``make_dataset``).  ``rng = numpy.random.default_rng(seed)``, ``C``
classes (at most 10), ``d`` features.  Prototypes ``mu_c`` in ``[0, 1]^d``:

  * flat task (``image=None``): ``mu_c = 0.5 + 0.3 u``, ``u`` uniform in
    ``[-1, 1]^d`` (drawn class by class, ``d`` values each);
  * image task (``d = image^2``): a ``4 x 4`` grid ``g_c`` uniform in ``[-1, 1]``
    per class, bilinearly interpolated to ``image x image`` (grid node
    ``(i, j)`` sits at pixel ``(i (image-1)/3, j (image-1)/3)``), and
    ``mu_c = 0.5 + 0.35 * interpolation``.

Samples: the label ``y`` uniform in ``[0, C)``, ``x = clip(mu_y + 0.25 N(0, I), 0, 1)``.
The draws happen in this order: all prototypes, then the labels of all
samples, then the Gaussian noise of all samples (shape ``(count, d)``).
``make_dataset(d, classes, train, test, seed, image)`` returns the training
and the test set (``(x, y)`` each) from one generator; the test set follows
the training set in the stream.

Training (``train_mlp``, ``train_cnn``): mini-batch Adam on the softmax
cross-entropy, ``epochs`` passes over the training set in batches of 64,
learning rate 1e-3, betas (0.9, 0.999), initial weights uniform in
``[-a, a]`` with ``a = min(0.5, 2.8 / sqrt(fan_in))`` and zero biases from
``default_rng(init_seed)``, batch order from the same generator.  Hidden
activations: the tables' real functions (relu, tanh, sigmoid, gelu with the
exact erf derivative ``Phi(x) + x phi(x)``).  The CNN is trained on its
kernel (``K^2 x ch``) and its dense layer; its convolution is then exported
as the public linear map of ``couplet.nn``.  The result is a
``couplet.nn.Model`` with float and integer layers at the wire format.

Command line (from artifact/python):
    python -m couplet.tools.train_synthetic --preset mlp16 --n 16 --f 8 --out mlp16.json
prints the float test accuracy, the accuracy of the fixed-point mirror in
mode exact on the test set, the prediction agreement of the two, and the
largest pre-activation magnitude against the range of the wire format.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Optional, Tuple

import numpy as np

from .. import nn as nnm
from ..nn import Model, PRESETS

__all__ = ["accuracy", "make_dataset", "train_cnn", "train_mlp", "train_preset"]

BATCH = 64
LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# The task
# --------------------------------------------------------------------------- #

def _bilinear(grid: np.ndarray, image: int) -> np.ndarray:
    """Bilinear interpolation of a 4 x 4 grid to image x image."""
    g = grid.shape[0]
    coords = np.linspace(0.0, g - 1, image)
    out = np.empty((image, image))
    for a, ya in enumerate(coords):
        i0 = min(int(math.floor(ya)), g - 2)
        ty = ya - i0
        for b, xb in enumerate(coords):
            j0 = min(int(math.floor(xb)), g - 2)
            tx = xb - j0
            out[a, b] = ((1 - ty) * (1 - tx) * grid[i0, j0] + (1 - ty) * tx * grid[i0, j0 + 1]
                         + ty * (1 - tx) * grid[i0 + 1, j0] + ty * tx * grid[i0 + 1, j0 + 1])
    return out


def make_dataset(d: int, classes: int, train: int, test: int, seed: int, image: Optional[int] = None):
    """The seeded task; returns ``((x_train, y_train), (x_test, y_test))``; see the module docstring."""
    if not 1 <= classes <= 10:
        raise ValueError("at most ten classes")
    if image is not None and image * image != d:
        raise ValueError("an image task needs d = image^2")
    rng = np.random.default_rng(int(seed))
    protos = np.empty((classes, d))
    for c in range(classes):
        if image is None:
            protos[c] = 0.5 + 0.3 * rng.uniform(-1.0, 1.0, d)
        else:
            protos[c] = 0.5 + 0.35 * _bilinear(rng.uniform(-1.0, 1.0, (4, 4)), image).reshape(-1)
    count = train + test
    y = rng.integers(0, classes, count)
    noise = rng.normal(0.0, 1.0, (count, d))
    x = np.clip(protos[y] + 0.25 * noise, 0.0, 1.0)
    return (x[:train], y[:train]), (x[train:], y[train:])


# --------------------------------------------------------------------------- #
# Activations and their derivatives
# --------------------------------------------------------------------------- #

def _act(name: str, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(activation, derivative) at z."""
    if name == "relu":
        return np.maximum(z, 0.0), (z > 0).astype(np.float64)
    if name == "tanh":
        t = np.tanh(z)
        return t, 1.0 - t * t
    if name == "sigmoid":
        s = 0.5 * (1.0 + np.tanh(0.5 * z))
        return s, s * (1.0 - s)
    if name == "gelu":
        erf = np.vectorize(math.erf)
        phi_big = 0.5 * (1.0 + erf(z / _SQRT2))
        phi_small = np.exp(-0.5 * z * z) / _SQRT2PI
        return z * phi_big, phi_big + z * phi_small
    raise ValueError(f"training supports relu, tanh, sigmoid and gelu, not {name!r}")


def _softmax_xent(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(p[np.arange(n), y] + 1e-300).mean()
    g = p
    g[np.arange(n), y] -= 1.0
    return float(loss), g / n


class _Adam:
    def __init__(self, params: list) -> None:
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list, grads: list) -> None:
        self.t += 1
        b1, b2 = BETAS
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            p -= LR * mh / (np.sqrt(vh) + EPS)


def _init(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    a = min(0.5, 2.8 / math.sqrt(cols))
    return rng.uniform(-a, a, (rows, cols))


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((np.argmax(logits, axis=1) == y).mean())


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #

def train_mlp(sizes, act: str, data, epochs: int, init_seed: int, n: int, f: int, verbose: bool = False) -> Model:
    """Train an MLP of the given sizes on ``data = (x, y)``; returns the quantised ``Model``."""
    sizes = [int(v) for v in sizes]
    x, y = data
    rng = np.random.default_rng(int(init_seed))
    Ws = [_init(rng, sizes[l + 1], sizes[l]) for l in range(len(sizes) - 1)]
    bs = [np.zeros(sizes[l + 1]) for l in range(len(sizes) - 1)]
    params = Ws + bs
    opt = _Adam(params)
    count = x.shape[0]
    for epoch in range(int(epochs)):
        order = rng.permutation(count)
        total = 0.0
        for start in range(0, count, BATCH):
            idx = order[start:start + BATCH]
            xb, yb = x[idx], y[idx]
            hs, ds = [xb], []
            h = xb
            for l, (W, b) in enumerate(zip(Ws, bs)):
                z = h @ W.T + b
                if l < len(Ws) - 1:
                    h, dh = _act(act, z)
                    ds.append(dh)
                else:
                    h = z
                hs.append(h)
            loss, g = _softmax_xent(hs[-1], yb)
            total += loss * len(idx)
            gW, gb = [None] * len(Ws), [None] * len(Ws)
            for l in range(len(Ws) - 1, -1, -1):
                gW[l] = g.T @ hs[l]
                gb[l] = g.sum(axis=0)
                if l > 0:
                    g = (g @ Ws[l]) * ds[l - 1]
            opt.step(params, gW + gb)
        if verbose:
            print(f"  epoch {epoch + 1}: loss {total / count:.4f}", flush=True)
    return nnm.mlp_from_float(sizes, act, n, f, Ws, bs)


# --------------------------------------------------------------------------- #
# CNN
# --------------------------------------------------------------------------- #

def _patch_index(image: int, K: int, S: int) -> np.ndarray:
    Ho = (image - K) // S + 1
    idx = np.empty((Ho * Ho, K * K), dtype=np.int64)
    for oy in range(Ho):
        for ox in range(Ho):
            for i in range(K):
                for j in range(K):
                    idx[oy * Ho + ox, i * K + j] = (oy * S + i) * image + (ox * S + j)
    return idx


def train_cnn(image: int, channels: int, kernel: int, stride: int, n_out: int, act: str, data, epochs: int,
              init_seed: int, n: int, f: int, verbose: bool = False) -> Model:
    """Train the CNN (one K x K stride-S convolution with ``channels`` channels, activation, dense)."""
    x, y = data
    K, S, ch = int(kernel), int(stride), int(channels)
    idx = _patch_index(image, K, S)
    P = idx.shape[0]
    rng = np.random.default_rng(int(init_seed))
    Wc = _init(rng, ch, K * K).T                 # (K^2, ch)
    bc = np.zeros(ch)
    Wf = _init(rng, n_out, P * ch)
    bf = np.zeros(n_out)
    params = [Wc, bc, Wf, bf]
    opt = _Adam(params)
    count = x.shape[0]
    for epoch in range(int(epochs)):
        order = rng.permutation(count)
        total = 0.0
        for start in range(0, count, BATCH):
            sel = order[start:start + BATCH]
            xb, yb = x[sel], y[sel]
            patches = xb[:, idx]                                   # (B, P, K^2)
            z = patches @ Wc + bc                                  # (B, P, ch)
            h, dh = _act(act, z)
            flat = h.reshape(len(sel), -1)                         # feature pos * ch + c
            logits = flat @ Wf.T + bf
            loss, g = _softmax_xent(logits, yb)
            total += loss * len(sel)
            gWf = g.T @ flat
            gbf = g.sum(axis=0)
            gflat = (g @ Wf).reshape(len(sel), P, ch) * dh
            gWc = np.einsum("bpk,bpc->kc", patches, gflat)
            gbc = gflat.sum(axis=(0, 1))
            opt.step(params, [gWc, gbc, gWf, gbf])
        if verbose:
            print(f"  epoch {epoch + 1}: loss {total / count:.4f}", flush=True)
    return nnm.cnn_from_float(image, ch, K, S, n_out, act, n, f, Wc, bc, Wf, bf)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

def train_preset(preset: str, n: int, f: int, *, epochs: int = 15, train: int = 8192, test: int = 2048,
                 data_seed: int = 20261013, init_seed: int = 3, verbose: bool = False):
    """Train a preset on its task; returns ``(model, test set, report)``."""
    cfg = PRESETS[preset]
    t = time.perf_counter()
    if cfg["kind"] == "mlp":
        sizes = cfg["sizes"]
        d, classes = sizes[0], sizes[-1]
        tr, te = make_dataset(d, classes, train, test, data_seed)
        model = train_mlp(sizes, cfg["act"], tr, epochs, init_seed, n, f, verbose)
    else:
        img, classes = cfg["image"], cfg["n_out"]
        tr, te = make_dataset(img * img, classes, train, test, data_seed, image=img)
        model = train_cnn(img, cfg["channels"], cfg["kernel"], cfg["stride"], classes, cfg["act"], tr, epochs,
                          init_seed, n, f, verbose)
    report = evaluate(model, te)
    report.update({"preset": preset, "epochs": epochs, "train": train, "test": test, "data_seed": data_seed,
                   "init_seed": init_seed, "train_seconds": round(time.perf_counter() - t, 1)})
    return model, te, report


def evaluate(model: Model, test) -> dict:
    """Float accuracy, exact-mode mirror accuracy, agreement, and the pre-activation range."""
    x, y = test
    logits = model.forward_float(x)
    float_pred = np.argmax(logits, axis=1)
    mirror_pred = []
    max_pre = 0
    overflow = 0
    for row in x:
        res = model.mirror(nnm.features_to_codes(row, model.f), "exact")
        mirror_pred.append(int(np.argmax(res.logits)))
        overflow += res.overflow
        for layer in res.pre:
            max_pre = max(max_pre, max(abs(v) for v in layer))
    mirror_pred = np.array(mirror_pred)
    return {"float_accuracy": accuracy(logits, y), "mirror_exact_accuracy": float((mirror_pred == y).mean()),
            "prediction_agreement": float((mirror_pred == float_pred).mean()),
            "max_pre_activation": max_pre / float(1 << (2 * model.f)),
            "code_range": float(1 << (model.n - 1 - model.f)), "code_overflows": int(overflow)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="train a preset model on the seeded synthetic task")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="mlp16")
    ap.add_argument("--n", type=int, default=16, help="wire bits (plane domain)")
    ap.add_argument("--f", type=int, default=8, help="fraction bits")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train", type=int, default=8192)
    ap.add_argument("--test", type=int, default=2048)
    ap.add_argument("--data-seed", type=int, default=20261013)
    ap.add_argument("--init-seed", type=int, default=3)
    ap.add_argument("--out", default=None, help="write the model JSON (couplet.nn.Model.save)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    model, _, report = train_preset(args.preset, args.n, args.f, epochs=args.epochs, train=args.train,
                                    test=args.test, data_seed=args.data_seed, init_seed=args.init_seed,
                                    verbose=not args.quiet)
    print(json.dumps(report, indent=1))
    if args.out:
        model.save(args.out)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
