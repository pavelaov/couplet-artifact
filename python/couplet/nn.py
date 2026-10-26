"""Models of the Couplet layer (DESIGN_M3.md sections 3 and 5): MLP and CNN as public
linear maps, integer quantisation, the cleartext fixed-point mirror, the model hash and
the session plan (manifest).

A ``Model`` is a list of integer layers ``(W_l, b_l)`` (``W_l`` of shape
``rows x cols``, ``b_l`` of length ``rows``) with ``f`` fraction bits,
``W_l = round(W 2^f)``, ``b_l = round(b 2^f)``, an activation name (a table of
``couplet.tables``) and the wire format ``(n, f)``.  The convolution of the
CNN (one ``K x K`` kernel, stride ``S``, ``ch`` channels, valid padding, on a
square image) is stored as its public linear map: output feature
``pos * ch + c`` (``pos = oy * Wo + ox``) reads pixel ``(oy S + i, ox S + j)``
with weight ``kernel[i K + j, c]`` and bias ``bias[c]``.  The second layer of
the CNN is dense to the logits.  Input features are reals in ``[0, 1]``,
encoded as ``x_q = round(x 2^f)``.

Pipeline (one layer): ``p = W a + b 2^f`` with ``a`` the ``f``-fraction-bit
inputs (sign-exact integers), so ``p`` has ``2f`` fraction bits; then
truncate-and-mask (``couplet.truncmask``) to the ``n``-bit code
``z = (p >> f) mod 2^n`` (plus the rounding bit in mode ``round``) and the
activation table ``T[z]`` (``f`` fraction bits).  The logits are the last
``p`` (``2f`` fraction bits), opened as sign-exact values.

``Model.mirror(x_q, mode, thresholds)`` is the cleartext fixed-point mirror:
with the public thresholds ``v_l`` recorded by the secure run it reproduces
the secure result bit for bit in BOTH modes (mode ``exact`` needs no
thresholds).  ``Model.forward_float`` is the floating-point model (the
float weights, if the model carries them).

Vertical partition of the input (public-model mode): features ``[0, d//2)``
belong to P_0, the rest to P_1 (``Model.partition``); each party computes
its additive share of the first-layer pre-activation from its own features
(``first_layer_share``).

Hashes.  ``model_hash`` = SHA-256 (plain, not framed) of ``"CPLTMDL1" | kind byte
(0 mlp, 1 cnn) | le32 len(act) | act | le32 n | le32 f | le32 L | per layer:
le32 rows, le32 cols, W as int64 LE row-major, b as int64 LE``;
``architecture_hash`` the same under ``"CPLTARC1"`` without the weight and
bias blobs (what the private-model manifest binds, since P_1 does not know
the weights).  ``plan_manifest(model, mode, k, s, private_model, tiling)`` lists the
circuit executions of a session in generation order with the table hashes.

Tiled private-model plans (task R-TILE of design/CHANGELOG_M3.md).  A private
model with more than ``Tiling.limit`` weights (default 4,096, the largest
private model of the untiled plans) gets a tiled plan: every weight matrix is
cut into tiles of at most ``Tiling.rows x Tiling.cols`` (default 16 x 16)
weights, row-major (``layer_tiles``), and every tile has its own matrix triple;
the triples of one tile shape form one manifest entry whose batch is the
number of tiles of that shape (shapes in the order of their first tile,
``tile_shapes``), and the triples of an entry go to the tiles of its shape in
row-major order.  The input masks of a party come in instances of
``rows x cols`` masks (one entry with the batch dimension) and one entry for
the remainder, so that no single circuit and no single execution grows with
the model.  The plan of a model at or below the limit is the untiled one,
byte for byte.

Presets (``PRESETS``, built with random or trained weights by
``build_preset``): ``mlp784`` [784, 128, 128, 10] tanh (the full MLP),
``mlp16`` [16, 8, 8, 4], ``mlp32`` [32, 16, 16, 8], ``mlp64`` [64, 32, 32, 10];
``cnn28`` 28 x 28, 8 channels, 5 x 5 stride 2, ReLU, dense to 10 (the full CNN),
``cnn10`` (10 x 10, 2 channels, 4 classes), ``cnn12`` (12 x 12, 2 channels, 8 classes),
``cnn14`` (14 x 14, 4 channels, 10 classes);
``mlp4`` [4, 2, 2, 2] is the tiny shape of the adversarial tests;
``mlp7`` [7, 3, 2] and ``mlp48`` [48, 64, 64, 10] are the reduced shapes of the
tiled private-model plans (``mlp7`` with tiles of 2 x 2 in the mixed-language
test, ``mlp48``, 7,808 weights, with the default tiles of 16 x 16).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import circuits_ppml as cp
from . import fixedpoint as fx
from . import tables

__all__ = ["DEFAULT_TILING", "MirrorResult", "Model", "PRESETS", "Tiling", "act_float", "build_preset",
           "conv_matrix", "decode_logits", "features_to_codes", "layer_tiles", "plan_manifest", "plan_tiled",
           "quantise", "random_model", "tile_shapes"]

MODEL_FORMAT = "couplet-model/1"
_HASH_MODEL = b"CPLTMDL1"
_HASH_ARCH = b"CPLTARC1"
KINDS = ("mlp", "cnn")

# name -> (kind, parameters, activation)
PRESETS: Dict[str, dict] = {
    "mlp4": {"kind": "mlp", "sizes": [4, 2, 2, 2], "act": "tanh"},
    "mlp16": {"kind": "mlp", "sizes": [16, 8, 8, 4], "act": "tanh"},
    "mlp32": {"kind": "mlp", "sizes": [32, 16, 16, 8], "act": "tanh"},
    "mlp64": {"kind": "mlp", "sizes": [64, 32, 32, 10], "act": "tanh"},
    "mlp784": {"kind": "mlp", "sizes": [784, 128, 128, 10], "act": "tanh"},
    # reduced shapes for tiled private-model plans (task R-TILE): edge tiles in both directions
    "mlp7": {"kind": "mlp", "sizes": [7, 3, 2], "act": "tanh"},
    "mlp48": {"kind": "mlp", "sizes": [48, 64, 64, 10], "act": "tanh"},
    "cnn10": {"kind": "cnn", "image": 10, "channels": 2, "kernel": 5, "stride": 2, "n_out": 4, "act": "relu"},
    "cnn12": {"kind": "cnn", "image": 12, "channels": 2, "kernel": 5, "stride": 2, "n_out": 8, "act": "relu"},
    "cnn14": {"kind": "cnn", "image": 14, "channels": 4, "kernel": 5, "stride": 2, "n_out": 10, "act": "relu"},
    "cnn28": {"kind": "cnn", "image": 28, "channels": 8, "kernel": 5, "stride": 2, "n_out": 10, "act": "relu"},
}


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


# --------------------------------------------------------------------------- #
# Quantisation and the convolution as a linear map
# --------------------------------------------------------------------------- #

def quantise(values, f: int) -> np.ndarray:
    """``round(values 2^f)`` (round half to even) as an int64 array."""
    return np.rint(np.asarray(values, dtype=np.float64) * float(1 << int(f))).astype(np.int64)


def features_to_codes(x, f: int) -> List[int]:
    """Real features (in [0, 1]) -> integers with ``f`` fraction bits."""
    return [int(v) for v in quantise(x, f)]


def decode_logits(logits: Sequence[int], f: int) -> List[float]:
    """Opened logits (``2f`` fraction bits) -> reals."""
    return [int(v) / float(1 << (2 * int(f))) for v in logits]


def conv_matrix(image: int, kernel: int, stride: int, channels: int, weights, bias) -> Tuple[np.ndarray, np.ndarray]:
    """The convolution as a dense linear map ``(Ho Wo ch) x (image^2)`` and its bias vector.

    ``weights`` has shape ``(kernel^2, channels)`` (row ``i K + j``), ``bias`` length ``channels``.
    """
    K, S, ch = int(kernel), int(stride), int(channels)
    img = int(image)
    Ho = (img - K) // S + 1
    weights = np.asarray(weights)
    bias = np.asarray(bias)
    if weights.shape != (K * K, ch) or bias.shape != (ch,):
        raise ValueError("kernel weights must have shape (K*K, channels) and the bias (channels,)")
    if Ho < 1:
        raise ValueError("the kernel does not fit the image")
    W = np.zeros((Ho * Ho * ch, img * img), dtype=weights.dtype)
    b = np.zeros(Ho * Ho * ch, dtype=bias.dtype)
    for oy in range(Ho):
        for ox in range(Ho):
            pos = oy * Ho + ox
            for c in range(ch):
                r = pos * ch + c
                b[r] = bias[c]
                for i in range(K):
                    for j in range(K):
                        W[r, (oy * S + i) * img + (ox * S + j)] = weights[i * K + j, c]
    return W, b


def act_float(name: str, x: np.ndarray) -> np.ndarray:
    """The real activation on an array (float64)."""
    x = np.asarray(x, dtype=np.float64)
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 0.5 * (1.0 + np.tanh(0.5 * x))
    if name == "gelu":
        return np.vectorize(lambda v: tables.reference("gelu", v))(x) if x.size else x
    if name == "sign":
        return np.sign(x)
    if name == "recip":
        return np.where(x > 0, 1.0 / np.where(x > 0, x, 1.0), 0.0)
    if name == "rsqrt":
        return np.where(x > 0, 1.0 / np.sqrt(np.where(x > 0, x, 1.0)), 0.0)
    raise ValueError(f"unknown activation {name!r}")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@dataclass
class MirrorResult:
    """Outcome of the cleartext mirror: logits (``2f`` fraction bits), per hidden layer the
    pre-activations ``p``, the codes ``z`` of the sites and the activations ``a``; ``overflow``
    counts the sites whose floor ``p >> f`` does not fit the ``n``-bit code."""

    logits: list
    pre: list = field(default_factory=list)
    codes: list = field(default_factory=list)
    acts: list = field(default_factory=list)
    overflow: int = 0


class Model:
    """An integer model; see the module docstring.

    ``layers``: list of ``(W, b)`` int64 arrays; ``act``: a table name; ``n``,
    ``f``: the wire format; ``kind``: "mlp" or "cnn"; ``meta``: the public
    shape parameters; ``float_layers``: the real weights, or None;
    ``weights_known``: False for the shape-only copy that P_1 holds in
    private-model mode (``layers`` are then None).
    """

    def __init__(self, kind: str, act: str, n: int, f: int, layers, meta: dict,
                 float_layers=None, shapes=None) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if act not in tables.NAMES:
            raise ValueError(f"act must be one of {tables.NAMES}")
        self.kind = kind
        self.act = act
        self.fmt = fx.Format(int(n), int(f))
        self.meta = dict(meta)
        self.float_layers = None
        if layers is None:
            if shapes is None:
                raise ValueError("a shape-only model needs its layer shapes")
            self.layers = None
            self.shapes = [(int(r), int(c)) for r, c in shapes]
        else:
            self.layers = []
            for W, b in layers:
                W = np.asarray(W, dtype=np.int64)
                b = np.asarray(b, dtype=np.int64)
                if W.ndim != 2 or b.shape != (W.shape[0],):
                    raise ValueError("a layer is (W of shape rows x cols, b of length rows)")
                self.layers.append((W, b))
            self.shapes = [(int(W.shape[0]), int(W.shape[1])) for W, _ in self.layers]
            if float_layers is not None:
                self.float_layers = [(np.asarray(W, dtype=np.float64), np.asarray(b, dtype=np.float64))
                                     for W, b in float_layers]
                if [W.shape for W, _ in self.float_layers] != [W.shape for W, _ in self.layers]:
                    raise ValueError("float layers must have the shapes of the integer layers")
        for (r, c), (r2, c2) in zip(self.shapes, self.shapes[1:]):
            if c2 != r:
                raise ValueError("consecutive layers must chain")

    # -- public shape ---------------------------------------------------------- #

    @property
    def n(self) -> int:
        return self.fmt.n

    @property
    def f(self) -> int:
        return self.fmt.f

    @property
    def weights_known(self) -> bool:
        return self.layers is not None

    @property
    def sizes(self) -> List[int]:
        return [self.shapes[0][1]] + [r for r, _ in self.shapes]

    @property
    def d(self) -> int:
        return self.shapes[0][1]

    @property
    def n_out(self) -> int:
        return self.shapes[-1][0]

    @property
    def hidden_widths(self) -> List[int]:
        return [r for r, _ in self.shapes[:-1]]

    @property
    def n_sites(self) -> int:
        return sum(self.hidden_widths)

    @property
    def n_layers(self) -> int:
        return len(self.shapes)

    @property
    def n_weights(self) -> int:
        """Entries of all weight matrices (public: the architecture fixes it)."""
        return sum(r * c for r, c in self.shapes)

    @property
    def partition(self) -> int:
        """Features ``[0, partition)`` belong to P_0, ``[partition, d)`` to P_1."""
        return self.d // 2

    def feature_slice(self, role: int) -> slice:
        return slice(0, self.partition) if role == 0 else slice(self.partition, self.d)

    def shape_only(self) -> "Model":
        """The copy without weights that P_1 holds in private-model mode."""
        return Model(self.kind, self.act, self.n, self.f, None, self.meta, shapes=self.shapes)

    # -- hashes ---------------------------------------------------------------------- #

    def _hash(self, with_weights: bool) -> bytes:
        parts = [_HASH_MODEL if with_weights else _HASH_ARCH, bytes([KINDS.index(self.kind)]),
                 len(self.act).to_bytes(4, "little"), self.act.encode("utf-8"),
                 self.n.to_bytes(4, "little"), self.f.to_bytes(4, "little"),
                 len(self.shapes).to_bytes(4, "little")]
        for i, (rows, cols) in enumerate(self.shapes):
            parts += [rows.to_bytes(4, "little"), cols.to_bytes(4, "little")]
            if with_weights:
                W, b = self.layers[i]
                parts += [np.ascontiguousarray(W, dtype="<i8").tobytes(), np.ascontiguousarray(b, dtype="<i8").tobytes()]
        return hashlib.sha256(b"".join(parts)).digest()

    def architecture_hash(self) -> bytes:
        return self._hash(False)

    def model_hash(self) -> bytes:
        if not self.weights_known:
            raise ValueError("a shape-only model has no model hash")
        return self._hash(True)

    def table(self) -> List[int]:
        """The activation table of the hidden layers at the wire format."""
        return tables.activation_table(self.act, self.n, self.f)

    def table_hash(self) -> bytes:
        return tables.table_hash(self.act, self.n, self.f)

    # -- bounds ------------------------------------------------------------------- #

    def pre_activation_bounds(self) -> List[int]:
        """Per layer a bound on ``|p_i|`` over all inputs in [0, 1] (weights needed)."""
        if not self.weights_known:
            raise ValueError("bounds need the weights")
        amax = 1 << self.f                       # inputs in [0, 1] at f fraction bits
        tmax = max(abs(t) for t in self.table())
        bounds = []
        for i, (W, b) in enumerate(self.layers):
            if i > 0:
                amax = tmax
            bound = int((np.abs(W.astype(object)).sum(axis=1) * amax + np.abs(b.astype(object)) * (1 << self.f)).max())
            bounds.append(bound)
        return bounds

    def sign_exact_in(self, w: int) -> bool:
        """True iff every pre-activation and every table entry fits a sign-exact value of ``w`` bits."""
        half = 1 << (int(w) - 1)
        if self.weights_known:
            return all(bd < half for bd in self.pre_activation_bounds()) and all(abs(t) < half for t in self.table())
        return all(abs(t) < half for t in self.table())

    def release_fits(self, k: int) -> bool:
        """True iff every logit fits a signed payload of ``k`` bits (``|p| < 2^(k-1)``): the release
        decodes ``p mod 2^k``, the part of an opening that the MAC check authenticates."""
        return self.pre_activation_bounds()[-1] < (1 << (int(k) - 1))

    # -- the cleartext computations --------------------------------------------- #

    def forward_float(self, x) -> np.ndarray:
        """Float64 logits of the real model (``float_layers`` needed); ``x``: (d,) or (batch, d)."""
        if self.float_layers is None:
            raise ValueError("the model carries no float weights")
        h = np.asarray(x, dtype=np.float64)
        single = h.ndim == 1
        if single:
            h = h[None, :]
        for i, (W, b) in enumerate(self.float_layers):
            h = h @ W.T + b
            if i < len(self.float_layers) - 1:
                h = act_float(self.act, h)
        return h[0] if single else h

    def first_layer_share(self, role: int, features) -> List[int]:
        """P_role's additive share of the first-layer pre-activation from its own features
        (public-model mode): ``W_1[:, part] x_part`` plus the bias (P_0 only), as integers."""
        if not self.weights_known:
            raise ValueError("the public-model pipeline needs the weights")
        W, b = self.layers[0]
        sl = self.feature_slice(role)
        x = [int(v) for v in features]
        if len(x) != sl.stop - sl.start:
            raise ValueError(f"P_{role} holds {sl.stop - sl.start} features")
        Wp = W[:, sl].astype(object)
        p = Wp.dot(np.array(x, dtype=object)) if x else np.zeros(W.shape[0], dtype=object)
        if role == 0:
            p = p + b.astype(object) * (1 << self.f)
        return [int(v) for v in p]

    def mirror(self, x_q: Sequence[int], mode: str, thresholds: Optional[Sequence[Sequence[int]]] = None,
               first_pre: Optional[Sequence[int]] = None) -> MirrorResult:
        """The cleartext fixed-point mirror of the secure pipeline (bit for bit).

        ``x_q``: the quantised features of both parties (``d`` integers);
        ``thresholds[l][j]``: the public threshold ``v_l`` of site ``j`` of hidden
        layer ``l`` (mode ``round``; ignored in mode ``exact``); ``first_pre``
        replaces the first-layer pre-activation (tests of input substitution).
        """
        if not self.weights_known:
            raise ValueError("the mirror needs the weights")
        if mode not in cp.MODES:
            raise ValueError("mode must be 'round' or 'exact'")
        x = [int(v) for v in x_q]
        if len(x) != self.d:
            raise ValueError(f"expected {self.d} features")
        a = np.array(x, dtype=object)
        T = self.table()
        res = MirrorResult([])
        n, f = self.n, self.f
        half = 1 << (n - 1)
        L = len(self.layers)
        for l, (W, b) in enumerate(self.layers):
            p = W.astype(object).dot(a) + b.astype(object) * (1 << f)
            if l == 0 and first_pre is not None:
                p = np.array([int(v) for v in first_pre], dtype=object)
            if l == L - 1:
                res.logits = [int(v) for v in p]
                break
            codes = []
            for j, pj in enumerate(p):
                pj = int(pj)
                if not -half <= (pj >> f) < half:
                    res.overflow += 1
                v_l = None
                if mode == "round" and f > 0:
                    if thresholds is None:
                        raise ValueError("mode round needs the recorded thresholds")
                    v_l = int(thresholds[l][j])
                z = fx.truncate(pj, f, n)
                if mode == "round" and f > 0:
                    z = (z + int((pj % (1 << f)) > v_l)) % (1 << n)
                codes.append(z)
            acts = [T[z] for z in codes]
            res.pre.append([int(v) for v in p])
            res.codes.append(codes)
            res.acts.append(acts)
            a = np.array(acts, dtype=object)
        return res

    # -- (de)serialisation ------------------------------------------------------- #

    def to_dict(self, with_float: bool = True) -> dict:
        if not self.weights_known:
            raise ValueError("a shape-only model is not serialised")
        doc = {"format": MODEL_FORMAT, "kind": self.kind, "act": self.act, "n": self.n, "f": self.f,
               "sizes": self.sizes, "partition": self.partition, "meta": self.meta,
               "layers": [{"rows": int(W.shape[0]), "cols": int(W.shape[1]), "W": W.tolist(), "b": b.tolist()}
                          for W, b in self.layers],
               "float_layers": None, "model_hash": self.model_hash().hex(),
               "architecture_hash": self.architecture_hash().hex(),
               "table_hash": self.table_hash().hex()}
        if with_float and self.float_layers is not None:
            doc["float_layers"] = [{"W": W.tolist(), "b": b.tolist()} for W, b in self.float_layers]
        return doc

    @classmethod
    def from_dict(cls, doc: dict) -> "Model":
        if doc.get("format") != MODEL_FORMAT:
            raise ValueError(f"unknown model format {doc.get('format')!r}")
        layers = [(np.array(lay["W"], dtype=np.int64), np.array(lay["b"], dtype=np.int64)) for lay in doc["layers"]]
        floats = None
        if doc.get("float_layers"):
            floats = [(np.array(lay["W"], dtype=np.float64), np.array(lay["b"], dtype=np.float64))
                      for lay in doc["float_layers"]]
        model = cls(doc["kind"], doc["act"], int(doc["n"]), int(doc["f"]), layers, doc.get("meta", {}), floats)
        if "model_hash" in doc and model.model_hash().hex() != doc["model_hash"]:
            raise ValueError("the model hash does not match the layers")
        return model

    def save(self, path: str, with_float: bool = True) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(self.to_dict(with_float), fh)
            fh.write("\n")

    @classmethod
    def load(cls, path: str) -> "Model":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def _float_to_model(kind: str, act: str, n: int, f: int, float_layers, meta: dict) -> Model:
    layers = [(quantise(W, f), quantise(b, f)) for W, b in float_layers]
    return Model(kind, act, n, f, layers, meta, float_layers)


def mlp_from_float(sizes: Sequence[int], act: str, n: int, f: int, weights, biases) -> Model:
    """MLP from real weights: ``weights[l]`` of shape ``(sizes[l+1], sizes[l])``, ``biases[l]`` of length ``sizes[l+1]``."""
    sizes = [int(v) for v in sizes]
    layers = []
    for l in range(len(sizes) - 1):
        W = np.asarray(weights[l], dtype=np.float64)
        b = np.asarray(biases[l], dtype=np.float64)
        if W.shape != (sizes[l + 1], sizes[l]) or b.shape != (sizes[l + 1],):
            raise ValueError("weight shapes do not match the sizes")
        layers.append((W, b))
    return _float_to_model("mlp", act, n, f, layers, {"sizes": sizes})


def cnn_from_float(image: int, channels: int, kernel: int, stride: int, n_out: int, act: str, n: int, f: int,
                   conv_w, conv_b, fc_w, fc_b) -> Model:
    """CNN from real weights: ``conv_w`` (K^2, ch), ``conv_b`` (ch,), ``fc_w`` (n_out, features), ``fc_b`` (n_out,)."""
    Wc, bc = conv_matrix(image, kernel, stride, channels, np.asarray(conv_w, dtype=np.float64),
                         np.asarray(conv_b, dtype=np.float64))
    Wf = np.asarray(fc_w, dtype=np.float64)
    bf = np.asarray(fc_b, dtype=np.float64)
    if Wf.shape != (n_out, Wc.shape[0]) or bf.shape != (n_out,):
        raise ValueError("the dense layer must map the convolution features to n_out logits")
    meta = {"image": int(image), "channels": int(channels), "kernel": int(kernel), "stride": int(stride),
            "n_out": int(n_out), "features": int(Wc.shape[0]),
            "kernel_weights": np.asarray(conv_w, dtype=np.float64).tolist(),
            "kernel_bias": np.asarray(conv_b, dtype=np.float64).tolist()}
    return _float_to_model("cnn", act, n, f, [(Wc, bc), (Wf, bf)], meta)


def random_float_layers(shapes: Sequence[Tuple[int, int]], seed: int, scale: float = 1.0) -> list:
    """Uniform random real layers ``W ~ U[-a, a]`` with ``a = scale * min(0.5, 2.8 / sqrt(cols))``, ``b ~ U[-0.05, 0.05]``."""
    rng = np.random.default_rng(int(seed))
    out = []
    for rows, cols in shapes:
        a = scale * min(0.5, 2.8 / math.sqrt(cols))
        out.append(((rng.random((rows, cols)) * 2 - 1) * a, (rng.random(rows) * 2 - 1) * 0.05 * scale))
    return out


def random_model(preset: str, n: int, f: int, seed: int = 1, scale: float = 1.0) -> Model:
    """A preset shape with random (untrained) real weights, quantised at ``f`` bits."""
    cfg = PRESETS[preset]
    if cfg["kind"] == "mlp":
        sizes = cfg["sizes"]
        shapes = [(sizes[l + 1], sizes[l]) for l in range(len(sizes) - 1)]
        fl = random_float_layers(shapes, seed, scale)
        return mlp_from_float(sizes, cfg["act"], n, f, [W for W, _ in fl], [b for _, b in fl])
    K, ch, img, S, n_out = cfg["kernel"], cfg["channels"], cfg["image"], cfg["stride"], cfg["n_out"]
    Ho = (img - K) // S + 1
    (Wc, bc), (Wf, bf) = random_float_layers([(ch, K * K), (n_out, Ho * Ho * ch)], seed, scale)
    return cnn_from_float(img, ch, K, S, n_out, cfg["act"], n, f, Wc.T, bc, Wf, bf)


def build_preset(preset: str, n: int, f: int, *, seed: int = 1, path: Optional[str] = None) -> Model:
    """A preset model: from the JSON file ``path`` (trained) if given, else random weights."""
    if path is not None:
        return Model.load(path)
    return random_model(preset, n, f, seed)


# --------------------------------------------------------------------------- #
# The session plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Tiling:
    """Tiling of the matrix triples of a private-model plan (see the module docstring).

    ``rows`` x ``cols``: the largest tile (and ``rows * cols`` input masks per
    instance of an input-mask entry); ``limit``: a private model with more than
    ``limit`` weights gets a tiled plan (0: every private model is tiled).
    """

    rows: int = 16
    cols: int = 16
    limit: int = 4096

    def __post_init__(self) -> None:
        for name in ("rows", "cols", "limit"):
            v = getattr(self, name)
            if not _is_int(v) or v < (0 if name == "limit" else 1):
                raise ValueError(f"tiling {name} must be a {'non-negative' if name == 'limit' else 'positive'} integer")

    @property
    def masks_per_instance(self) -> int:
        return self.rows * self.cols

    def applies(self, model: "Model") -> bool:
        """True iff a private-model plan of ``model`` is tiled (more than ``limit`` weights)."""
        return model.n_weights > self.limit

    def to_dict(self) -> dict:
        return {"rows": self.rows, "cols": self.cols, "limit": self.limit}


DEFAULT_TILING = Tiling()


def layer_tiles(rows: int, cols: int, tiling: Tiling) -> List[Tuple[int, int, int, int]]:
    """The tiles ``(r0, c0, h, w)`` of a ``rows x cols`` matrix, row-major over the tile grid
    (tile ``(I, J)`` covers rows ``[I R, min((I+1) R, rows))`` and columns ``[J C, min((J+1) C, cols))``)."""
    R, C = tiling.rows, tiling.cols
    return [(r0, c0, min(R, rows - r0), min(C, cols - c0)) for r0 in range(0, rows, R) for c0 in range(0, cols, C)]


def tile_shapes(rows: int, cols: int, tiling: Tiling) -> List[Tuple[Tuple[int, int], int]]:
    """``[((h, w), count)]``: the tile shapes of the matrix in the order of their first tile
    (row-major), with the number of tiles of each shape (one manifest entry each)."""
    counts: Dict[Tuple[int, int], int] = {}
    for _, _, h, w in layer_tiles(rows, cols, tiling):
        counts[(h, w)] = counts.get((h, w), 0) + 1
    return list(counts.items())


def plan_tiled(model: Model, private_model: bool, tiling: Optional[Tiling] = None) -> bool:
    """True iff the session plan of ``model`` is tiled (private-model mode above the limit)."""
    return bool(private_model) and (DEFAULT_TILING if tiling is None else tiling).applies(model)


def _add_inmasks(man: cp.Manifest, owner: int, count: int, use: str, k: int, s: int, per: int) -> None:
    """Input masks of a tiled plan: instances of ``per`` masks (batch) and one entry for the rest."""
    q, r = divmod(count, per)
    if q:
        man.add(f"{use}, {per} per instance", cp.inmask_descriptor(owner, per, k, s), q)
    if r:
        man.add(f"{use}, the last {r}", cp.inmask_descriptor(owner, r, k, s))


def plan_manifest(model: Model, mode: str, k: int, s: int, private_model: bool = False,
                  tiling: Optional[Tiling] = None) -> cp.Manifest:
    """The circuit executions of one inference session, in generation order (the manifest).

    Public-model mode: per hidden layer ``l`` one entry ``site(n, f, mode)`` with
    batch = width of the layer.  Private-model mode adds, before the sites:
    ``inmask(0, d0 + sum(rows cols + rows))`` for P_0's features, weights and
    biases, ``inmask(1, d1)`` for P_1's features, and per layer
    ``mattriple(rows, cols)``; and after the sites one entry ``outmask(k, s)``
    with batch = the number of logits (one output mask ``[[2^k rho]]`` per
    logit, correction C2 of design/CHANGELOG_M3.md: P_1 cannot bound P_0's
    weights, so the logits are released masked).  The model hash is the full
    hash in public-model mode and the architecture hash in private-model mode.

    A tiled plan (``plan_tiled``; ``tiling`` defaults to ``DEFAULT_TILING``)
    replaces the two input-mask entries by instances of ``rows * cols`` masks
    plus a remainder (``"<use>, <per> per instance"`` with the batch dimension,
    ``"<use>, the last <r>"``) and the matrix triple of layer ``l`` by one entry
    ``mattriple(h, w)`` per tile shape (``tile_shapes``), batch = the number of
    tiles of that shape, use ``"matrix triple tiles of layer <l>: <rows>x<cols>
    in tiles of <R>x<C>"``.  Everything else is unchanged.
    """
    if mode not in cp.MODES:
        raise ValueError("mode must be 'round' or 'exact'")
    tiling = DEFAULT_TILING if tiling is None else tiling
    if not isinstance(tiling, Tiling):
        raise ValueError("tiling must be a couplet.nn.Tiling")
    n, f = model.n, model.f
    model_hash = model.architecture_hash() if private_model else model.model_hash()
    man = cp.Manifest(int(k), int(s), mode, bool(private_model), model_hash,
                      [(model.act, n, f, model.table_hash())], [])
    if private_model:
        d0 = model.partition
        d1 = model.d - d0
        count0 = d0 + sum(r * c + r for r, c in model.shapes)
        if not tiling.applies(model):
            man.add("inputs of P_0: features, weights, biases", cp.inmask_descriptor(0, count0, k, s))
            if d1:
                man.add("inputs of P_1: features", cp.inmask_descriptor(1, d1, k, s))
            for l, (rows, cols) in enumerate(model.shapes):
                man.add(f"matrix triple of layer {l + 1}", cp.mattriple_descriptor(rows, cols, k, s))
        else:
            per = tiling.masks_per_instance
            _add_inmasks(man, 0, count0, "inputs of P_0: features, weights, biases", k, s, per)
            if d1:
                _add_inmasks(man, 1, d1, "inputs of P_1: features", k, s, per)
            for l, (rows, cols) in enumerate(model.shapes):
                use = f"matrix triple tiles of layer {l + 1}: {rows}x{cols} in tiles of {tiling.rows}x{tiling.cols}"
                for (h, w), count in tile_shapes(rows, cols, tiling):
                    man.add(use, cp.mattriple_descriptor(h, w, k, s), count)
    for l, width in enumerate(model.hidden_widths):
        man.add(f"sites of layer {l + 1}", cp.site_descriptor(n, f, k, s, mode), width)
    if private_model:
        man.add("output masks of the logits", cp.outmask_descriptor(k, s), model.n_out)
    return man
