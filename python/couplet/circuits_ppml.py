"""Generation circuits of the Couplet layer (DESIGN_M3.md section 2): activation
sites, Beaver triples, matrix-vector triples, input masks, batches of them, the
plaintext reference of every output, Pi_Gen for them and the manifest.

Every circuit is an instance of the gate-circuit machinery of
``couplet.circuits_gen`` (BOUND session input ``delta`` = the (s+1)-bit MAC key
``Delta = Delta_0 + Delta_1``, ``share_out``, the DPF key sub-circuit) and
follows its conventions: bit vectors little-endian, ``w = k + s``, seeds in
the ``couplet.ascon`` layout, ripple-carry adders by default.

Circuits (``<p>`` is the label prefix: empty for a single item, ``i<j>.`` for
item ``j`` of a batch; ``b`` in {0, 1} is the owner; all share_out inputs are
P_0's and all share_out outputs go to P_1)
--------------------------------------------------------------------------
``site(n, f, k, s, mode)``, m = f + n + s (everything one activation site needs)
    (b, "<p>rp.coin")            m      rho' = coin_0 ^ coin_1, uniform in Z_{2^m}
    (0, "<p>rp.sigma"), (0, "<p>rp.mu_sigma")      m     [[rho']] over Z_{2^m}:
    "<p>rp.share1", "<p>rp.mu_share1"      P_1    m     tag Delta rho' mod 2^m
    "<p>dpf.<key labels>"                         plane A: DPF key pair with
        inputs (b, "<p>dpf.rho0"/"rho1")          payload 1, domain 2^n, at
                                                  rho_h = (rho' >> f) mod 2^n
    mode "exact" and f >= 1 only:
    "<p>pb.<key labels>"                          plane B: DPF key pair with
        inputs (b, "<p>pb.rho0"/"rho1")           payload 1, domain 2^f, at
                                                  rho_l = rho' mod 2^f
    (0, "<p>rh.sigma"), (0, "<p>rh.mu_sigma")      n+s   [[rho_h]] over Z_{2^(n+s)}
    "<p>rh.share1", "<p>rh.mu_share1"      P_1    n+s   tag Delta rho_h mod 2^(n+s)
    (b, "<p>r2.coin")                             n+s   r' = coin_0 ^ coin_1
    (0, "<p>r2.sigma"), (0, "<p>r2.mu_sigma")      n+s   [[r']] over Z_{2^(n+s)}
    "<p>r2.share1", "<p>r2.mu_share1"      P_1    n+s
    and plane A is then at r = r' mod 2^n instead of rho_h.
    f = 0 gives the plain offset gate of the core (``gate_descriptor(n, k, s,
    True, True, [], 0)``) in BOTH modes: the circuit is bit for bit the core's
    ``gate_n<n>_k<k>_s<s>_auth_dpf`` (same digest), and the truncation of
    DESIGN_M3.md section 3 degenerates to the wire-mode Mask of the core.

``triples(count, k, s)``: ``count`` Beaver triples over Z_{2^w}, entry j of a
vector at bits [j w, (j+1) w) of the label
    (b, "<p>a.coin"), (b, "<p>b.coin")    count w   a_j, b_j uniform, c_j = a_j b_j
    (0, "<p>a.sigma"), (0, "<p>a.mu_sigma"), "<p>a.share1", "<p>a.mu_share1"  (P_1)
    the same for "b." and "c."                      tags Delta value mod 2^w

``mattriple(rows, cols, k, s)``: a matrix-vector triple over Z_{2^w},
A uniform (rows x cols, entry (i, j) at index i cols + j), b uniform (cols),
c = A b (rows); labels "a.", "b.", "c." as for ``triples``.

``inmask(owner, count, k, s)``: ``count`` authenticated masks known to P_owner
    (owner, "<p>rho")                     count w   the masks (the owner's coins)
    (0, "<p>m.sigma"), (0, "<p>m.mu_sigma"), "<p>m.share1", "<p>m.mu_share1"
                                                    [[rho_j]], tag Delta rho_j

``outmask(k, s)``: ONE output mask ``[[2^k rho]]`` over Z_{2^w} per instance
(``rho`` uniform in [0, 2^s)); the circuit IS the core's on-demand
output-mask circuit ``gate_n1_k<k>_s<s>_shared_om1`` (``couplet.protocol.
outmask_descriptor``; same gates, same digest, and its name is the name of
the core file), labels "rp." (an unused 1-bit offset of the core gate) and
"om0." (``share_out`` of the high s bits: value ``2^k rho``, tag ``2^k (Delta
rho mod 2^s)``).  Used for the logits in private-model mode (correction C2
of design/CHANGELOG_M3.md); a session runs it with the engine's batch
dimension equal to the number of logits.

``ppml_batch(items)``: items concatenated under the prefixes ``i<j>.``, one
BOUND input ``delta`` (reference only; the native code uses the batch
dimension of the engine for sites of one kind).

Reference: ``expected_outputs(descriptor, inputs)`` recomputes every output
from the plaintext inputs with ``couplet.fss`` (explicit roots).

Pi_Gen (``generate`` / ``generate_batch``): exactly as ``couplet.protocol``:
fresh uniform coins for every input declaration the party owns (and, for P_0,
the ``share_out`` values, which are its shares), one circuit execution on the
session engine, assembly of this party's material: ``SiteMaterial``,
``TripleMaterial``, ``MatTripleMaterial``, ``InMaskMaterial``,
``OutMaskMaterial``.

Manifest (DESIGN_M3.md section 2): an ordered list of circuit executions
(descriptor, batch, use) with the model hash and the hashes of every public
table; ``Manifest.encode`` is a canonical byte encoding, ``digest`` its
SHA-256, and ``compare_manifest`` exchanges the digest before any circuit
runs (a mismatch aborts).  Encoding (all integers little-endian):
    "CPLTMAN1" | u32 k | u32 s | u8 mode (0 round, 1 exact) | u8 private_model |
    32 B model hash | u32 n_tables | per table: u32 len, name, u32 n, u32 f, 32 B hash |
    u32 n_entries | per entry: u32 len, use, u32 len, canonical JSON of the
    descriptor (sorted keys, no spaces), u32 batch

``python -m couplet.circuits_ppml`` writes the AND-gate table
``experiments/figdata/ppml_circuit_stats.json`` (sites for n in {8, 12, 14, 16,
20} and f in {0, 6, 8, 12} in both modes, the triple and matrix-triple
circuits with per-triple and per-weight fits, the input mask, and the core's
comparison bundle at the same n).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import circuits_gen as cg
from . import fss
from .abb import bits_to_ints, circuit_digest, ints_to_bits
from .circuit import BOUND, Builder, Circuit, bits_to_int
from .circuits_gen import DELTA, emit_key, mul_mod, share_out
from .circuits_ascon import SEED_BITS
from .hashes import ProtocolAbort
from .prep.shares import as_rand

__all__ = [
    "InMaskMaterial", "KIND_BATCH", "KIND_INMASK", "KIND_MATTRIPLE", "KIND_OUTMASK", "KIND_SITE", "KIND_TRIPLES",
    "MODES", "Manifest", "MatTripleMaterial", "OutMaskMaterial", "SiteMaterial", "TripleMaterial", "and_count",
    "batch_descriptor", "build", "build_batch", "build_inmask", "build_mattriple", "build_outmask", "build_site",
    "build_triples", "canonical_descriptor", "compare_manifest", "deal_material", "descriptor_from_name",
    "descriptor_name", "is_ppml_descriptor",
    "expected_outputs", "generate", "generate_batch", "generation_circuit", "inmask_descriptor",
    "mattriple_descriptor", "outmask_descriptor", "site_descriptor", "site_plaintext", "stats_table",
    "triples_descriptor",
]

KIND_SITE = "site"
KIND_TRIPLES = "triples"
KIND_MATTRIPLE = "mattriple"
KIND_INMASK = "inmask"
KIND_OUTMASK = "outmask"
KIND_BATCH = "ppml_batch"
ITEM_KINDS = (KIND_SITE, KIND_TRIPLES, KIND_MATTRIPLE, KIND_INMASK, KIND_OUTMASK)
MODES = ("round", "exact")
MAX_PLANE_BITS = 20
DESIGN_VERSION = "M3-1"
PHASE_MANIFEST = "manifest"
_MANIFEST_MAGIC = b"CPLTMAN1"
_CIRCUIT_CACHE: "OrderedDict[str, Circuit]" = OrderedDict()
_CIRCUIT_CACHE_SIZE = 8
_CIRCUIT_LOCK = threading.Lock()
_AND_COUNTS: dict = {}

Inputs = Mapping[Tuple[int, str], object]


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #

def _check_ks(k: int, s: int) -> None:
    if not (_is_int(k) and _is_int(s)) or k < 1 or s < 1 or 2 * (k + s) > 320:
        raise ValueError("need integers k >= 1, s >= 1 with 2(k+s) <= 320")


def _check_adder(adder: str) -> None:
    if adder not in cg.ADDERS:
        raise ValueError(f"adder must be one of {cg.ADDERS}")


def site_descriptor(n: int, f: int, k: int, s: int, mode: str, adder: str = "ripple") -> Dict:
    """Descriptor of ``site(n, f, mode)``: plane domain 2^n, f fraction bits, wire format (n, f)."""
    _check_ks(k, s)
    _check_adder(adder)
    if not (_is_int(n) and _is_int(f)):
        raise ValueError("n and f must be integers")
    if not 1 <= n <= MAX_PLANE_BITS:
        raise ValueError(f"n must be in 1..{MAX_PLANE_BITS}")
    if not 0 <= f <= MAX_PLANE_BITS:
        raise ValueError(f"f must be in 0..{MAX_PLANE_BITS}")
    if n > k:
        raise ValueError("sites need n <= k")
    if f + n > k:
        raise ValueError("sites need f + n <= k: a pre-activation over Z_{2^w} is reduced to Z_{2^(f+n+s)}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return {"kind": KIND_SITE, "n": n, "f": f, "k": k, "s": s, "mode": mode, "adder": adder}


def triples_descriptor(count: int, k: int, s: int, adder: str = "ripple") -> Dict:
    """Descriptor of ``triples(count)``: Beaver triples over Z_{2^w}."""
    _check_ks(k, s)
    _check_adder(adder)
    if not _is_int(count) or count < 1:
        raise ValueError("count must be a positive integer")
    return {"kind": KIND_TRIPLES, "count": count, "k": k, "s": s, "adder": adder}


def mattriple_descriptor(rows: int, cols: int, k: int, s: int, adder: str = "ripple") -> Dict:
    """Descriptor of ``mattriple(rows, cols)``: a matrix-vector triple over Z_{2^w}."""
    _check_ks(k, s)
    _check_adder(adder)
    if not (_is_int(rows) and _is_int(cols)) or rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive integers")
    return {"kind": KIND_MATTRIPLE, "rows": rows, "cols": cols, "k": k, "s": s, "adder": adder}


def inmask_descriptor(owner: int, count: int, k: int, s: int, adder: str = "ripple") -> Dict:
    """Descriptor of ``inmask(owner, count)``: authenticated masks known to P_owner."""
    _check_ks(k, s)
    _check_adder(adder)
    if owner not in (0, 1) or isinstance(owner, bool):
        raise ValueError("owner must be 0 or 1")
    if not _is_int(count) or count < 1:
        raise ValueError("count must be a positive integer")
    return {"kind": KIND_INMASK, "owner": owner, "count": count, "k": k, "s": s, "adder": adder}


def outmask_descriptor(k: int, s: int, adder: str = "ripple") -> Dict:
    """Descriptor of ``outmask(k, s)``: one output mask ``[[2^k rho]]`` per instance (the core's circuit)."""
    _check_ks(k, s)
    _check_adder(adder)
    return {"kind": KIND_OUTMASK, "k": k, "s": s, "adder": adder}


def _core_outmask(d: Mapping) -> Dict:
    """The core gate descriptor whose circuit an ``outmask`` item is (``protocol.outmask_descriptor``)."""
    return cg.gate_descriptor(1, d["k"], d["s"], False, False, [], 1, d.get("adder", "ripple"))


def batch_descriptor(items: Sequence, adder: str = "ripple") -> Dict:
    """Descriptor of a batch: item j keeps its labels under the prefix ``i<j>.``."""
    _check_adder(adder)
    descs = []
    for item in items:
        d = canonical_descriptor(item)
        if d["kind"] not in ITEM_KINDS:
            raise ValueError(f"a batch contains items of the kinds {ITEM_KINDS} only")
        d.pop("adder", None)
        descs.append(d)
    if not descs:
        raise ValueError("empty batch")
    if len({(d["k"], d["s"]) for d in descs}) != 1:
        raise ValueError("all items of a batch must use the same k and s")
    return {"kind": KIND_BATCH, "k": descs[0]["k"], "s": descs[0]["s"], "items": descs, "adder": adder}


def canonical_descriptor(desc) -> Dict:
    """Validated, normalised copy of a descriptor (a dict with a ``kind``, or a circuit of this module)."""
    if isinstance(desc, Circuit):
        desc = desc.meta.get("descriptor")
        if desc is None:
            raise ValueError("circuit carries no descriptor")
    if not isinstance(desc, Mapping) or "kind" not in desc:
        raise ValueError("expected a descriptor (dict with a 'kind')")
    kind = desc["kind"]
    adder = desc.get("adder", "ripple")
    try:
        if kind == KIND_SITE:
            return site_descriptor(desc["n"], desc["f"], desc["k"], desc["s"], desc["mode"], adder)
        if kind == KIND_TRIPLES:
            return triples_descriptor(desc["count"], desc["k"], desc["s"], adder)
        if kind == KIND_MATTRIPLE:
            return mattriple_descriptor(desc["rows"], desc["cols"], desc["k"], desc["s"], adder)
        if kind == KIND_INMASK:
            return inmask_descriptor(desc["owner"], desc["count"], desc["k"], desc["s"], adder)
        if kind == KIND_OUTMASK:
            return outmask_descriptor(desc["k"], desc["s"], adder)
        if kind == KIND_BATCH:
            return batch_descriptor(desc["items"], adder)
    except KeyError as exc:
        raise ValueError(f"incomplete {kind} descriptor: missing {exc}") from None
    raise ValueError(f"unknown descriptor kind {kind!r}")


def descriptor_name(desc) -> str:
    """Short file-name-safe name of a descriptor."""
    d = canonical_descriptor(desc)
    kind = d["kind"]
    tail = "" if d["adder"] == "ripple" else "_" + d["adder"]
    ks = f"k{d['k']}_s{d['s']}"
    if kind == KIND_SITE:
        return f"site_n{d['n']}_f{d['f']}_{ks}_{d['mode']}{tail}"
    if kind == KIND_TRIPLES:
        return f"triples{d['count']}_{ks}{tail}"
    if kind == KIND_MATTRIPLE:
        return f"mattriple_{d['rows']}x{d['cols']}_{ks}{tail}"
    if kind == KIND_INMASK:
        return f"inmask_p{d['owner']}_c{d['count']}_{ks}{tail}"
    if kind == KIND_OUTMASK:
        return cg.descriptor_name(_core_outmask(d))          # the core's file: gate_n1_k<k>_s<s>_shared_om1
    digest = hashlib.sha256(json.dumps(d["items"], sort_keys=True).encode()).hexdigest()[:12]
    return f"pbatch{len(d['items'])}_{digest}{tail}"


_NAME_SITE = re.compile(r"^site_n(\d+)_f(\d+)_k(\d+)_s(\d+)_(round|exact)(_sklansky)?$")
_NAME_TRIPLES = re.compile(r"^triples(\d+)_k(\d+)_s(\d+)(_sklansky)?$")
_NAME_MATTRIPLE = re.compile(r"^mattriple_(\d+)x(\d+)_k(\d+)_s(\d+)(_sklansky)?$")
_NAME_INMASK = re.compile(r"^inmask_p([01])_c(\d+)_k(\d+)_s(\d+)(_sklansky)?$")
_NAME_OUTMASK = re.compile(r"^gate_n1_k(\d+)_s(\d+)_shared_om1(_sklansky)?$")


def descriptor_from_name(name: str) -> Dict:
    """Inverse of ``descriptor_name`` for sites, triples, matrix triples and input masks
    (batches are named by a hash and cannot be parsed); raises ``ValueError`` otherwise."""
    def adder(flag) -> str:
        return "sklansky" if flag else "ripple"

    m = _NAME_SITE.match(name)
    if m:
        n, f, k, s = (int(m.group(i)) for i in range(1, 5))
        return site_descriptor(n, f, k, s, m.group(5), adder(m.group(6)))
    m = _NAME_TRIPLES.match(name)
    if m:
        return triples_descriptor(int(m.group(1)), int(m.group(2)), int(m.group(3)), adder(m.group(4)))
    m = _NAME_MATTRIPLE.match(name)
    if m:
        return mattriple_descriptor(*(int(m.group(i)) for i in range(1, 5)), adder(m.group(5)))
    m = _NAME_INMASK.match(name)
    if m:
        return inmask_descriptor(*(int(m.group(i)) for i in range(1, 5)), adder(m.group(5)))
    m = _NAME_OUTMASK.match(name)
    if m:
        return outmask_descriptor(int(m.group(1)), int(m.group(2)), adder(m.group(3)))
    raise ValueError(f"cannot parse the Couplet descriptor name {name!r}")


def is_ppml_descriptor(desc) -> bool:
    """True iff ``desc`` is a mapping whose kind is one of this module's (else it is a core descriptor)."""
    return isinstance(desc, Mapping) and desc.get("kind") in ITEM_KINDS + (KIND_BATCH,)


# --------------------------------------------------------------------------- #
# Sub-circuits
# --------------------------------------------------------------------------- #

def _chunks(bits: Sequence[int], count: int, w: int) -> List[List[int]]:
    return [list(bits[j * w:(j + 1) * w]) for j in range(count)]


def _coins_vec(B: Builder, label: str, count: int, w: int) -> List[List[int]]:
    """``count`` uniform w-bit values from the XOR of both parties' coins under one label."""
    x = B.xor_bits(B.input(0, label, count * w), B.input(1, label, count * w))
    return _chunks(x, count, w)


def _share_out_vec(B: Builder, q: str, values: Sequence[Sequence[int]],
                   tags: Optional[Sequence[Sequence[int]]], w: int) -> None:
    """share_out of ``len(values)`` independent w-bit values (and tags) under ONE label each."""
    count = len(values)
    sigma = B.input(0, q + "sigma", count * w)
    out: List[int] = []
    for j, v in enumerate(values):
        out += B.sub_mod(B.zero_extend(v, w), sigma[j * w:(j + 1) * w], width=w)[0]
    B.output(q + "share1", out, to=1)
    if tags is not None:
        mu = B.input(0, q + "mu_sigma", count * w)
        out = []
        for j, t in enumerate(tags):
            out += B.sub_mod(B.zero_extend(t, w), mu[j * w:(j + 1) * w], width=w)[0]
        B.output(q + "mu_share1", out, to=1)


def _dot_mod(B: Builder, xs: Sequence[Sequence[int]], ys: Sequence[Sequence[int]], w: int) -> List[int]:
    """``sum_j x_j y_j mod 2^w``: all partial products of all products in ONE Dadda reduction."""
    cols: List[List[int]] = [[] for _ in range(w)]
    const = 0
    for x, y in zip(xs, ys):
        for j, yj in enumerate(y):
            if j >= w or yj == 0:
                continue
            for i, xi in enumerate(x[:w - j]):
                pp = B.and_(xi, yj)
                if pp == 1:
                    const += 1 << (i + j)
                elif pp != 0:
                    cols[i + j].append(pp)
    return B._sum_columns(cols, const, w)


def _emit_site(B: Builder, delta: Sequence[int], p: str, d: Mapping) -> None:
    """``site(n, f, mode)`` under the prefix ``p`` (DESIGN_M3.md section 2)."""
    n, f, k, s, mode = d["n"], d["f"], d["k"], d["s"], d["mode"]
    w = k + s
    m = f + n + s
    one = B.const_bits(1, w)
    delta_w = B.zero_extend(delta, w)
    # coins rho' of m bits and [[rho']] over Z_{2^m}
    rp = B.xor_bits(B.input(0, p + "rp.coin", m), B.input(1, p + "rp.coin", m))
    share_out(B, p + "rp.", rp, mul_mod(B, delta, rp, m), m)
    rho_h = rp[f:f + n]
    if f == 0 or mode == "round":
        emit_key(B, p + "dpf.", "dpf", n, k, s, rho_h, one, delta_w)      # plane A at rho_h
        return
    rho_l = rp[:f]
    emit_key(B, p + "pb.", "dpf", f, k, s, rho_l, one, delta_w)            # plane B at rho_l
    share_out(B, p + "rh.", rho_h, mul_mod(B, delta, rho_h, n + s), n + s)  # [[rho_h]]
    r2 = B.xor_bits(B.input(0, p + "r2.coin", n + s), B.input(1, p + "r2.coin", n + s))
    share_out(B, p + "r2.", r2, mul_mod(B, delta, r2, n + s), n + s)       # [[r']]
    emit_key(B, p + "dpf.", "dpf", n, k, s, r2[:n], one, delta_w)          # plane A at r' mod 2^n


def _emit_triples(B: Builder, delta: Sequence[int], p: str, d: Mapping) -> None:
    count, w = d["count"], d["k"] + d["s"]
    a = _coins_vec(B, p + "a.coin", count, w)
    b = _coins_vec(B, p + "b.coin", count, w)
    c = [B.mul(a[j], b[j], w) for j in range(count)]
    for q, vec in (("a.", a), ("b.", b), ("c.", c)):
        _share_out_vec(B, p + q, vec, [mul_mod(B, delta, x, w) for x in vec], w)


def _emit_mattriple(B: Builder, delta: Sequence[int], p: str, d: Mapping) -> None:
    rows, cols, w = d["rows"], d["cols"], d["k"] + d["s"]
    a = _coins_vec(B, p + "a.coin", rows * cols, w)
    b = _coins_vec(B, p + "b.coin", cols, w)
    c = [_dot_mod(B, a[i * cols:(i + 1) * cols], b, w) for i in range(rows)]
    for q, vec in (("a.", a), ("b.", b), ("c.", c)):
        _share_out_vec(B, p + q, vec, [mul_mod(B, delta, x, w) for x in vec], w)


def _emit_inmask(B: Builder, delta: Sequence[int], p: str, d: Mapping) -> None:
    owner, count, w = d["owner"], d["count"], d["k"] + d["s"]
    rho = _chunks(B.input(owner, p + "rho", count * w), count, w)
    _share_out_vec(B, p + "m.", rho, [mul_mod(B, delta, x, w) for x in rho], w)


def _emit_outmask(B: Builder, delta: Sequence[int], p: str, d: Mapping) -> None:
    """``outmask(k, s)``: exactly the core's output-mask gate (``circuits_gen._emit_gate``)."""
    cg._emit_gate(B, delta, p, _core_outmask(d))


_EMIT = {KIND_SITE: _emit_site, KIND_TRIPLES: _emit_triples, KIND_MATTRIPLE: _emit_mattriple,
         KIND_INMASK: _emit_inmask, KIND_OUTMASK: _emit_outmask}


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def build(descriptor) -> Circuit:
    """Build the circuit of a descriptor (deterministic: same descriptor, same digest)."""
    desc = canonical_descriptor(descriptor)
    B = Builder(adder=desc["adder"])
    delta = B.input(BOUND, DELTA, desc["s"] + 1)
    if desc["kind"] == KIND_BATCH:
        for j, item in enumerate(desc["items"]):
            _EMIT[item["kind"]](B, delta, f"i{j}.", item)
    else:
        _EMIT[desc["kind"]](B, delta, "", desc)
    C = B.build()
    C.meta = {"descriptor": desc, "name": descriptor_name(desc), "design_version": DESIGN_VERSION}
    return C


def build_site(n: int, f: int, k: int, s: int, mode: str, *, adder: str = "ripple") -> Circuit:
    """``site(n, f, mode)`` of DESIGN_M3.md section 2 (``mode`` in {"round", "exact"})."""
    return build(site_descriptor(n, f, k, s, mode, adder))


def build_triples(count: int, k: int, s: int, *, adder: str = "ripple") -> Circuit:
    """``count`` Beaver triples over Z_{2^w} with tags."""
    return build(triples_descriptor(count, k, s, adder))


def build_mattriple(rows: int, cols: int, k: int, s: int, *, adder: str = "ripple") -> Circuit:
    """A matrix-vector triple ``(A, b, A b)`` over Z_{2^w} with tags."""
    return build(mattriple_descriptor(rows, cols, k, s, adder))


def build_inmask(owner: int, count: int, k: int, s: int, *, adder: str = "ripple") -> Circuit:
    """``count`` authenticated masks whose values P_owner knows."""
    return build(inmask_descriptor(owner, count, k, s, adder))


def build_outmask(k: int, s: int, *, adder: str = "ripple") -> Circuit:
    """One output mask ``[[2^k rho]]`` per instance: the core's ``gate_n1_k<k>_s<s>_shared_om1``."""
    return build(outmask_descriptor(k, s, adder))


def build_batch(items: Sequence, *, adder: str = "ripple") -> Circuit:
    """Several items in one circuit execution (labels under ``i<j>.``; one BOUND ``delta``)."""
    return build(batch_descriptor(items, adder))


# --------------------------------------------------------------------------- #
# Plaintext reference
# --------------------------------------------------------------------------- #

def _int(value) -> int:
    return value if isinstance(value, int) else bits_to_int(value)


def _get(inputs: Inputs, party, label: str) -> int:
    return _int(inputs[(party, label)])


def _unpack(value: int, count: int, w: int) -> List[int]:
    mask = (1 << w) - 1
    return [(value >> (j * w)) & mask for j in range(count)]


def _pack(entries: Sequence[int], w: int) -> int:
    mask = (1 << w) - 1
    return sum((int(v) & mask) << (j * w) for j, v in enumerate(entries))


def _plain_key(alpha: int, delta: int, n: int, k: int, s: int, inputs: Inputs, p: str) -> Dict[str, int]:
    s0 = _get(inputs, 0, p + "rho0") ^ _get(inputs, 1, p + "rho0")
    s1 = _get(inputs, 0, p + "rho1") ^ _get(inputs, 1, p + "rho1")
    k0, k1 = fss.dpf_gen(alpha, 1, delta, n, k, s, roots=(fss.seed_from_int(s0), fss.seed_from_int(s1)))
    return cg.pack_key_outputs(k0, k1, p)


def _plain_share_out(inputs: Inputs, q: str, value: int, tag: Optional[int], m: int) -> Dict[str, int]:
    mod = 1 << m
    out = {q + "share1": (value - _get(inputs, 0, q + "sigma")) % mod}
    if tag is not None:
        out[q + "mu_share1"] = (tag - _get(inputs, 0, q + "mu_sigma")) % mod
    return out


def _plain_share_out_vec(inputs: Inputs, delta: int, q: str, values: Sequence[int], w: int) -> Dict[str, int]:
    count = len(values)
    mod = 1 << w
    sigma = _unpack(_get(inputs, 0, q + "sigma"), count, w)
    mu = _unpack(_get(inputs, 0, q + "mu_sigma"), count, w)
    return {q + "share1": _pack([(v - sg) % mod for v, sg in zip(values, sigma)], w),
            q + "mu_share1": _pack([(delta * v - ms) % mod for v, ms in zip(values, mu)], w)}


def site_plaintext(desc: Mapping, inputs: Inputs, prefix: str = "") -> Dict[str, int]:
    """The secrets a site circuit derives from the coins: ``rp`` (rho'), ``rho_l``, ``rho_h``,
    ``r2`` (r', mode exact with f >= 1, else None) and ``index`` (the index of plane A)."""
    d = canonical_descriptor(desc)
    n, f, s = d["n"], d["f"], d["s"]
    m = f + n + s
    rp = _get(inputs, 0, prefix + "rp.coin") ^ _get(inputs, 1, prefix + "rp.coin")
    rho_l, rho_h = rp % (1 << f), (rp >> f) % (1 << n)
    r2 = None
    index = rho_h
    if f >= 1 and d["mode"] == "exact":
        r2 = _get(inputs, 0, prefix + "r2.coin") ^ _get(inputs, 1, prefix + "r2.coin")
        index = r2 % (1 << n)
    return {"rp": rp, "rho_l": rho_l, "rho_h": rho_h, "r2": r2, "index": index, "m": m}


def _expected_site(d: Mapping, inputs: Inputs, p: str) -> Dict[str, int]:
    n, f, k, s = d["n"], d["f"], d["k"], d["s"]
    delta = _get(inputs, BOUND, DELTA)
    pt = site_plaintext(d, inputs, p)
    rp, m = pt["rp"], pt["m"]
    out = _plain_share_out(inputs, p + "rp.", rp, delta * rp, m)
    out.update(_plain_key(pt["index"], delta, n, k, s, inputs, p + "dpf."))
    if pt["r2"] is not None:
        out.update(_plain_key(pt["rho_l"], delta, f, k, s, inputs, p + "pb."))
        out.update(_plain_share_out(inputs, p + "rh.", pt["rho_h"], delta * pt["rho_h"], n + s))
        out.update(_plain_share_out(inputs, p + "r2.", pt["r2"], delta * pt["r2"], n + s))
    return out


def _plain_coins_vec(inputs: Inputs, label: str, count: int, w: int) -> List[int]:
    return _unpack(_get(inputs, 0, label) ^ _get(inputs, 1, label), count, w)


def _expected_triples(d: Mapping, inputs: Inputs, p: str) -> Dict[str, int]:
    count, w = d["count"], d["k"] + d["s"]
    delta = _get(inputs, BOUND, DELTA)
    a = _plain_coins_vec(inputs, p + "a.coin", count, w)
    b = _plain_coins_vec(inputs, p + "b.coin", count, w)
    c = [(x * y) % (1 << w) for x, y in zip(a, b)]
    out: Dict[str, int] = {}
    for q, vec in (("a.", a), ("b.", b), ("c.", c)):
        out.update(_plain_share_out_vec(inputs, delta, p + q, vec, w))
    return out


def _expected_mattriple(d: Mapping, inputs: Inputs, p: str) -> Dict[str, int]:
    rows, cols, w = d["rows"], d["cols"], d["k"] + d["s"]
    delta = _get(inputs, BOUND, DELTA)
    a = _plain_coins_vec(inputs, p + "a.coin", rows * cols, w)
    b = _plain_coins_vec(inputs, p + "b.coin", cols, w)
    c = [sum(a[i * cols + j] * b[j] for j in range(cols)) % (1 << w) for i in range(rows)]
    out: Dict[str, int] = {}
    for q, vec in (("a.", a), ("b.", b), ("c.", c)):
        out.update(_plain_share_out_vec(inputs, delta, p + q, vec, w))
    return out


def _expected_inmask(d: Mapping, inputs: Inputs, p: str) -> Dict[str, int]:
    count, w = d["count"], d["k"] + d["s"]
    delta = _get(inputs, BOUND, DELTA)
    rho = _unpack(_get(inputs, d["owner"], p + "rho"), count, w)
    return _plain_share_out_vec(inputs, delta, p + "m.", rho, w)


def _expected_outmask(d: Mapping, inputs: Inputs, p: str) -> Dict[str, int]:
    c = _core_outmask(d)
    return cg.expected_outputs_gate(1, c["k"], c["s"], False, False, [], 1, inputs, p)


_EXPECTED = {KIND_SITE: _expected_site, KIND_TRIPLES: _expected_triples,
             KIND_MATTRIPLE: _expected_mattriple, KIND_INMASK: _expected_inmask, KIND_OUTMASK: _expected_outmask}


def expected_outputs(descriptor, inputs: Inputs) -> Dict[str, int]:
    """Every output of ``build(descriptor)`` recomputed in plaintext; ``inputs`` as for
    ``circuit.evaluate`` (including ``(BOUND, "delta")``); returns {label: int}."""
    d = canonical_descriptor(descriptor)
    if d["kind"] == KIND_BATCH:
        out: Dict[str, int] = {}
        for j, item in enumerate(d["items"]):
            out.update(_EXPECTED[item["kind"]](item, inputs, f"i{j}."))
        return out
    return _EXPECTED[d["kind"]](d, inputs, "")


# --------------------------------------------------------------------------- #
# Material (this party's output of one generation)
# --------------------------------------------------------------------------- #

@dataclass
class SiteMaterial:
    """One party's output of ``site(n, f, mode)`` for one site.

    ``rp``       share (value, tag) of [[rho']] over Z_{2^m}, m = f + n + s;
    ``plane_a``  this party's DPF key of plane A (domain 2^n, payload 1) at
                 rho_h (mode round, or f = 0) or at r' mod 2^n (mode exact);
    ``plane_b``  DPF key of plane B (domain 2^f) at rho_l, or None;
    ``rh``       share of [[rho_h]] over Z_{2^(n+s)}, or None;
    ``r2``       share of [[r']] over Z_{2^(n+s)}, or None.
    The two parties' shares sum to the secrets modulo the stated power of
    two, tags with ``Delta = Delta_0 + Delta_1``.
    """

    role: int
    descriptor: dict
    rp: tuple
    plane_a: fss.DPFKey
    plane_b: Optional[fss.DPFKey] = None
    rh: Optional[tuple] = None
    r2: Optional[tuple] = None

    @property
    def n(self) -> int:
        return self.descriptor["n"]

    @property
    def f(self) -> int:
        return self.descriptor["f"]

    @property
    def k(self) -> int:
        return self.descriptor["k"]

    @property
    def s(self) -> int:
        return self.descriptor["s"]

    @property
    def width(self) -> int:
        return self.k + self.s

    @property
    def m(self) -> int:
        return self.f + self.n + self.s

    @property
    def mode(self) -> str:
        return self.descriptor["mode"]

    @property
    def two_openings(self) -> bool:
        """True iff the site carries plane B (mode exact with f >= 1): two openings."""
        return self.plane_b is not None


@dataclass
class TripleMaterial:
    """One party's shares of ``count`` Beaver triples: lists of (value, tag) over Z_{2^w}."""

    role: int
    descriptor: dict
    a: list
    b: list
    c: list

    @property
    def width(self) -> int:
        return self.descriptor["k"] + self.descriptor["s"]

    @property
    def count(self) -> int:
        return self.descriptor["count"]


@dataclass
class MatTripleMaterial:
    """One party's shares of a matrix-vector triple: ``a`` (rows x cols, row-major), ``b``, ``c``."""

    role: int
    descriptor: dict
    a: list
    b: list
    c: list

    @property
    def width(self) -> int:
        return self.descriptor["k"] + self.descriptor["s"]

    @property
    def rows(self) -> int:
        return self.descriptor["rows"]

    @property
    def cols(self) -> int:
        return self.descriptor["cols"]


@dataclass
class InMaskMaterial:
    """One party's shares of ``count`` input masks; ``values`` (the masks) at the owner only."""

    role: int
    descriptor: dict
    shares: list
    values: Optional[list] = None

    @property
    def width(self) -> int:
        return self.descriptor["k"] + self.descriptor["s"]

    @property
    def owner(self) -> int:
        return self.descriptor["owner"]

    @property
    def count(self) -> int:
        return self.descriptor["count"]


@dataclass
class OutMaskMaterial:
    """One party's share ``(value, tag)`` of one output mask ``[[2^k rho]]`` over Z_{2^w};
    both components are multiples of ``2^k`` (as ``protocol.GateMaterial.outmask_shares``)."""

    role: int
    descriptor: dict
    share: tuple

    @property
    def width(self) -> int:
        return self.descriptor["k"] + self.descriptor["s"]


# --------------------------------------------------------------------------- #
# Pi_Gen for the Couplet items (as couplet.protocol does it for keys and gates)
# --------------------------------------------------------------------------- #

def generation_circuit(descriptor) -> Circuit:
    """The circuit of a descriptor, built once per process and cached."""
    desc = canonical_descriptor(descriptor)
    key = json.dumps(desc, sort_keys=True)
    with _CIRCUIT_LOCK:
        circuit = _CIRCUIT_CACHE.get(key)
        if circuit is None:
            circuit = build(desc)
            circuit_digest(circuit)
            _CIRCUIT_CACHE[key] = circuit
            while len(_CIRCUIT_CACHE) > _CIRCUIT_CACHE_SIZE:
                _CIRCUIT_CACHE.popitem(last=False)
        else:
            _CIRCUIT_CACHE.move_to_end(key)
    return circuit


def and_count(descriptor) -> int:
    """AND gates of ONE instance of the circuit of ``descriptor``."""
    desc = canonical_descriptor(descriptor)
    key = json.dumps(desc, sort_keys=True)
    count = _AND_COUNTS.get(key)
    if count is None:
        count = int(generation_circuit(desc).stats()["n_and"])
        _AND_COUNTS[key] = count
    return count


def _run(session, circuit: Circuit, fixed: dict, batch: int, rng) -> tuple:
    """Evaluate a generation circuit on the session; returns (own inputs, opened outputs) as ints."""
    rand = as_rand(rng)
    role = session.role
    inputs, own = {}, {}
    for owner, label, wires in circuit.inputs:
        if owner != role:
            continue
        width = len(wires)
        if label in fixed:
            bits = ints_to_bits(fixed[label], width)
        else:
            bits = rand.bits(width * batch).reshape(width, batch)
        inputs[label] = bits
        own[label] = bits
    result = session.evaluate(circuit, inputs, batch=batch)
    outputs = {label: bits_to_ints(bits) for label, bits in result.public.items()}
    outputs.update({label: bits_to_ints(bits) for label, bits in result.private.items()})
    session.last_gen = {"circuit": circuit.meta.get("name"), "digest": circuit_digest(circuit).hex(),
                        "batch": batch, "stats": dict(result.stats)}
    return {label: bits_to_ints(bits) for label, bits in own.items()}, outputs


def _assemble_key(role: int, n: int, k: int, s: int, out: dict, prefix: str, j: int) -> fss.DPFKey:
    width = k + s
    seed_mask = (1 << SEED_BITS) - 1
    cws, cwtl, cwtr = out[prefix + "cws"][j], out[prefix + "cwtl"][j], out[prefix + "cwtr"][j]
    return fss.DPFKey(
        b=role, n=n, k=k, s=s, seed=fss.seed_from_int(out[f"{prefix}root{role}"][j]),
        cw_s=tuple(fss.seed_from_int((cws >> (SEED_BITS * i)) & seed_mask) for i in range(n)),
        cw_tl=tuple((cwtl >> i) & 1 for i in range(n)), cw_tr=tuple((cwtr >> i) & 1 for i in range(n)),
        cw_out_v=out[prefix + "cwout_v"][j] % (1 << width), cw_out_t=out[prefix + "cwout_t"][j] % (1 << width))


def _share_scalar(role: int, own: dict, out: dict, q: str, j: int) -> tuple:
    if role == 0:
        return own[q + "sigma"][j], own[q + "mu_sigma"][j]
    return out[q + "share1"][j], out[q + "mu_share1"][j]


def _share_vec(role: int, own: dict, out: dict, q: str, j: int, count: int, w: int) -> list:
    if role == 0:
        values, tags = own[q + "sigma"][j], own[q + "mu_sigma"][j]
    else:
        values, tags = out[q + "share1"][j], out[q + "mu_share1"][j]
    return list(zip(_unpack(values, count, w), _unpack(tags, count, w)))


def _assemble(role: int, desc: dict, own: dict, out: dict, p: str, j: int):
    kind = desc["kind"]
    k, s = desc["k"], desc["s"]
    w = k + s
    if kind == KIND_SITE:
        n, f = desc["n"], desc["f"]
        rp = _share_scalar(role, own, out, p + "rp.", j)
        plane_a = _assemble_key(role, n, k, s, out, p + "dpf.", j)
        if f >= 1 and desc["mode"] == "exact":
            return SiteMaterial(role, desc, rp, plane_a, _assemble_key(role, f, k, s, out, p + "pb.", j),
                                _share_scalar(role, own, out, p + "rh.", j),
                                _share_scalar(role, own, out, p + "r2.", j))
        return SiteMaterial(role, desc, rp, plane_a)
    if kind == KIND_TRIPLES:
        count = desc["count"]
        return TripleMaterial(role, desc, *[_share_vec(role, own, out, p + q, j, count, w)
                                            for q in ("a.", "b.", "c.")])
    if kind == KIND_MATTRIPLE:
        rows, cols = desc["rows"], desc["cols"]
        return MatTripleMaterial(role, desc, _share_vec(role, own, out, p + "a.", j, rows * cols, w),
                                 _share_vec(role, own, out, p + "b.", j, cols, w),
                                 _share_vec(role, own, out, p + "c.", j, rows, w))
    if kind == KIND_OUTMASK:
        value, tag = _share_scalar(role, own, out, p + "om0.", j)       # s-bit shares of the high parts
        return OutMaskMaterial(role, desc, (value << k, tag << k))     # shares of 2^k rho over Z_{2^w}
    count = desc["count"]
    values = _unpack(own[p + "rho"][j], count, w) if role == desc["owner"] else None
    return InMaskMaterial(role, desc, _share_vec(role, own, out, p + "m.", j, count, w), values)


def _check_batch(batch) -> int:
    if isinstance(batch, bool) or int(batch) != batch or int(batch) < 1:
        raise ValueError("batch must be a positive integer")
    return int(batch)


def _check_session(session, desc: dict) -> None:
    if (desc["k"], desc["s"]) != (session.k, session.s):
        raise ValueError("the descriptor does not use the session's k and s")


def generate(session, descriptor, batch: int = 1, *, rng=None) -> list:
    """This party's material for ``batch`` independent instances of one item (one circuit execution).

    ``descriptor`` is a site, triples, mattriple or inmask descriptor with the
    session's ``k`` and ``s``.  Returns a list of ``batch`` material objects.
    Any ``ProtocolAbort`` moves the session to ABORTED.
    """
    session.require_active()
    batch = _check_batch(batch)
    desc = canonical_descriptor(descriptor)
    if desc["kind"] not in ITEM_KINDS:
        raise ValueError(f"generate takes items of the kinds {ITEM_KINDS}")
    _check_session(session, desc)
    circuit = generation_circuit(desc)
    own, out = _run(session, circuit, {}, batch, rng)
    return [_assemble(session.role, desc, own, out, "", j) for j in range(batch)]


def generate_batch(session, items, *, rng=None, adder: str = "ripple") -> list:
    """Several different items from ONE circuit execution (a batch circuit); one object per item."""
    session.require_active()
    desc = batch_descriptor(items, adder)
    _check_session(session, desc)
    circuit = generation_circuit(desc)
    own, out = _run(session, circuit, {}, 1, rng)
    return [_assemble(session.role, item, own, out, f"i{j}.", 0) for j, item in enumerate(desc["items"])]


# --------------------------------------------------------------------------- #
# Trusted-dealer reference (NOT a protocol; tests and formula checks only)
# --------------------------------------------------------------------------- #

def deal_material(descriptor, delta: int, rng=None) -> tuple:
    """Both parties' material of one item as a trusted dealer would produce it.

    NOT a protocol: the caller knows the full MAC key ``delta = Delta_0 +
    Delta_1`` (an unreduced (s+1)-bit integer) and every secret is sampled
    here with ``fss.dpf_gen`` for the keys.  It is the plaintext counterpart
    of ``generate`` for mass tests (the generation circuits are too large to
    run thousands of times).  ``rng`` offers ``getrandbits`` and ``randbytes``
    (``random.Random``; default ``random.SystemRandom``).  Returns
    ``(material of P_0, material of P_1)``.
    """
    import random

    desc = canonical_descriptor(descriptor)
    if desc["kind"] not in ITEM_KINDS:
        raise ValueError(f"deal_material takes items of the kinds {ITEM_KINDS}")
    rnd = random.SystemRandom() if rng is None else rng
    k, s = desc["k"], desc["s"]
    w = k + s
    delta = int(delta)
    if not 0 <= delta < (1 << (s + 1)):
        raise ValueError("delta must be the unreduced sum of two s-bit shares")

    def split(value: int, bits: int) -> tuple:
        """Shares (value, tag) of both parties modulo 2^bits."""
        mod = 1 << bits
        v0, t0 = rnd.getrandbits(bits), rnd.getrandbits(bits)
        return (v0, t0), ((value - v0) % mod, (delta * value - t0) % mod)

    def split_vec(values, bits: int) -> tuple:
        pairs = [split(v, bits) for v in values]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    kind = desc["kind"]
    if kind == KIND_SITE:
        n, f, mode = desc["n"], desc["f"], desc["mode"]
        m = f + n + s
        rp = rnd.getrandbits(m)
        rho_l, rho_h = rp % (1 << f), (rp >> f) % (1 << n)
        rp_sh = split(rp, m)
        if f >= 1 and mode == "exact":
            r2 = rnd.getrandbits(n + s)
            pa = fss.dpf_gen(r2 % (1 << n), 1, delta, n, k, s, rng=rnd)
            pb = fss.dpf_gen(rho_l, 1, delta, f, k, s, rng=rnd)
            rh_sh, r2_sh = split(rho_h, n + s), split(r2, n + s)
            return tuple(SiteMaterial(b, desc, rp_sh[b], pa[b], pb[b], rh_sh[b], r2_sh[b]) for b in (0, 1))
        pa = fss.dpf_gen(rho_h, 1, delta, n, k, s, rng=rnd)
        return tuple(SiteMaterial(b, desc, rp_sh[b], pa[b]) for b in (0, 1))
    if kind == KIND_TRIPLES:
        count = desc["count"]
        a = [rnd.getrandbits(w) for _ in range(count)]
        b = [rnd.getrandbits(w) for _ in range(count)]
        c = [(x * y) % (1 << w) for x, y in zip(a, b)]
        sa, sb, sc = split_vec(a, w), split_vec(b, w), split_vec(c, w)
        return tuple(TripleMaterial(p, desc, sa[p], sb[p], sc[p]) for p in (0, 1))
    if kind == KIND_MATTRIPLE:
        rows, cols = desc["rows"], desc["cols"]
        a = [rnd.getrandbits(w) for _ in range(rows * cols)]
        b = [rnd.getrandbits(w) for _ in range(cols)]
        c = [sum(a[i * cols + j] * b[j] for j in range(cols)) % (1 << w) for i in range(rows)]
        sa, sb, sc = split_vec(a, w), split_vec(b, w), split_vec(c, w)
        return tuple(MatTripleMaterial(p, desc, sa[p], sb[p], sc[p]) for p in (0, 1))
    if kind == KIND_OUTMASK:
        rho = rnd.getrandbits(s)
        v0, t0 = rnd.getrandbits(s), rnd.getrandbits(s)
        v1, t1 = (rho - v0) % (1 << s), (delta * rho - t0) % (1 << s)
        return tuple(OutMaskMaterial(b, desc, (v << k, t << k)) for b, (v, t) in enumerate(((v0, t0), (v1, t1))))
    count, owner = desc["count"], desc["owner"]
    rho = [rnd.getrandbits(w) for _ in range(count)]
    sh = split_vec(rho, w)
    return tuple(InMaskMaterial(p, desc, sh[p], list(rho) if p == owner else None) for p in (0, 1))


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def _u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little")


def _blob(data: bytes) -> bytes:
    return _u32(len(data)) + bytes(data)


def _desc_json(desc: Mapping) -> bytes:
    return json.dumps(canonical_descriptor(desc), sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class Manifest:
    """The plan of a session: every circuit execution with all public parameters, plus the
    hashes of the model and of every public table (the commitment point, DESIGN_M3.md 2).

    ``entries``  list of ``{"use": str, "descriptor": canonical dict, "batch": int}`` in the
                 order of the executions; ``tables`` list of ``(name, n, f, 32-byte hash)``.
    """

    k: int
    s: int
    mode: str
    private_model: bool
    model_hash: bytes
    tables: list
    entries: list

    def add(self, use: str, descriptor, batch: int = 1) -> dict:
        entry = {"use": str(use), "descriptor": canonical_descriptor(descriptor), "batch": _check_batch(batch)}
        self.entries.append(entry)
        return entry

    def encode(self) -> bytes:
        if self.mode not in MODES:
            raise ValueError("bad mode")
        if len(self.model_hash) != 32:
            raise ValueError("the model hash must be 32 bytes")
        parts = [_MANIFEST_MAGIC, _u32(self.k), _u32(self.s), bytes([MODES.index(self.mode)]),
                 bytes([1 if self.private_model else 0]), bytes(self.model_hash), _u32(len(self.tables))]
        for name, n, f, digest in self.tables:
            if len(digest) != 32:
                raise ValueError("a table hash must be 32 bytes")
            parts += [_blob(str(name).encode("utf-8")), _u32(n), _u32(f), bytes(digest)]
        parts.append(_u32(len(self.entries)))
        for entry in self.entries:
            parts += [_blob(entry["use"].encode("utf-8")), _blob(_desc_json(entry["descriptor"])),
                      _u32(entry["batch"])]
        return b"".join(parts)

    @classmethod
    def decode(cls, data: bytes) -> "Manifest":
        data = bytes(data)
        pos = len(_MANIFEST_MAGIC)
        if data[:pos] != _MANIFEST_MAGIC:
            raise ValueError("not a manifest encoding")

        def u32() -> int:
            nonlocal pos
            if pos + 4 > len(data):
                raise ValueError("truncated manifest")
            value = int.from_bytes(data[pos:pos + 4], "little")
            pos += 4
            return value

        def blob() -> bytes:
            nonlocal pos
            n = u32()
            if pos + n > len(data):
                raise ValueError("truncated manifest")
            value = data[pos:pos + n]
            pos += n
            return value

        def raw(n: int) -> bytes:
            nonlocal pos
            if pos + n > len(data):
                raise ValueError("truncated manifest")
            value = data[pos:pos + n]
            pos += n
            return value

        k, s = u32(), u32()
        mode_code, private = raw(1)[0], raw(1)[0]
        if mode_code >= len(MODES) or private > 1:
            raise ValueError("bad manifest flags")
        model_hash = raw(32)
        tables = []
        for _ in range(u32()):
            name = blob().decode("utf-8")
            n, f = u32(), u32()
            tables.append((name, n, f, raw(32)))
        entries = []
        for _ in range(u32()):
            use = blob().decode("utf-8")
            desc = canonical_descriptor(json.loads(blob().decode("utf-8")))
            entries.append({"use": use, "descriptor": desc, "batch": u32()})
        if pos != len(data):
            raise ValueError("trailing bytes")
        return cls(k, s, MODES[mode_code], bool(private), model_hash, tables, entries)

    def digest(self) -> bytes:
        """SHA-256 of the canonical encoding (32 bytes)."""
        return hashlib.sha256(self.encode()).digest()

    def and_budget(self) -> int:
        """AND triples of every execution of the plan (``and_count`` times batch, summed)."""
        return sum(and_count(e["descriptor"]) * e["batch"] for e in self.entries)

    def site_count(self) -> int:
        return sum(e["batch"] for e in self.entries if e["descriptor"]["kind"] == KIND_SITE)

    def summary(self) -> dict:
        return {"k": self.k, "s": self.s, "mode": self.mode, "private_model": self.private_model,
                "model_hash": self.model_hash.hex(), "digest": self.digest().hex(),
                "tables": [(name, n, f, h.hex()) for name, n, f, h in self.tables],
                "executions": len(self.entries), "sites": self.site_count(),
                "and_gates": self.and_budget(),
                "entries": [{"use": e["use"], "name": descriptor_name(e["descriptor"]), "batch": e["batch"],
                             "n_and": and_count(e["descriptor"])} for e in self.entries]}


def compare_manifest(chan, manifest: Manifest) -> bytes:
    """Exchange the manifest digest (one round, phase ``manifest``); abort on a mismatch.

    Returns the digest.  Run before the session setup, so that nothing but
    the digest has been sent when the plans are found to differ.
    """
    digest = manifest.digest()
    phase = chan.phase
    chan.mark(PHASE_MANIFEST)
    try:
        peer = chan.exchange(digest)
    finally:
        chan.mark(phase)
    if len(peer) != 32 or bytes(peer) != digest:
        raise ProtocolAbort("manifest digests differ")
    return digest


# --------------------------------------------------------------------------- #
# AND-gate table (experiments/figdata/ppml_circuit_stats.json)
# --------------------------------------------------------------------------- #

_STAT_KEYS = ("n_and", "n_xor", "n_not", "n_gates", "and_depth", "n_wires", "n_inputs", "n_outputs")


def _row(desc: Mapping, **extra) -> Dict:
    t = time.perf_counter()
    C = cg.build(desc) if desc.get("kind") in ("gate", "dpf", "dcf") else build(desc)
    st = C.stats()
    return {"name": C.meta["name"], **extra, **{key: st[key] for key in _STAT_KEYS},
            "digest": C.digest(), "build_seconds": round(time.perf_counter() - t, 2),
            "descriptor": C.meta["descriptor"]}


def _solve_affine(rows: Sequence[Dict]) -> Dict:
    """Exact fit AND = pw rows cols + pr rows + pc cols + c0 from four matrix-triple rows
    (shapes (a, a), (a, b), (b, a), (b, b)), checked on every further row."""
    r = {(x["rows"], x["cols"]): x["n_and"] for x in rows}
    (a, b) = sorted({k[0] for k in r})[:2]
    pw = (r[(b, b)] - r[(a, b)] - r[(b, a)] + r[(a, a)]) / ((b - a) * (b - a))
    pr = (r[(b, a)] - r[(a, a)]) / (b - a) - pw * a
    pc = (r[(a, b)] - r[(a, a)]) / (b - a) - pw * a
    c0 = r[(a, a)] - pw * a * a - pr * a - pc * a
    exact = all(abs(pw * x["rows"] * x["cols"] + pr * x["rows"] + pc * x["cols"] + c0 - x["n_and"]) < 0.5
                for x in rows)
    return {"per_weight": pw, "per_row": pr, "per_col": pc, "fixed": c0, "exact_for_all_shapes": exact,
            "formula": "AND = per_weight * rows * cols + per_row * rows + per_col * cols + fixed"}


def stats_table(ns: Sequence[int] = (8, 12, 14, 16, 20), fs: Sequence[int] = (0, 6, 8, 12),
                k: int = 64, s: int = 64, verbose: bool = True,
                mat_shapes: Sequence[Tuple[int, int]] = ((4, 4), (4, 8), (8, 4), (8, 8), (8, 16)),
                triple_counts: Sequence[int] = (1, 2, 4), inmask_counts: Sequence[int] = (1, 2, 4)) -> Dict:
    """All rows of ``experiments/figdata/ppml_circuit_stats.json``."""
    def say(row: Dict) -> None:
        if verbose:
            print(f"  {row['name']:<40} AND={row['n_and']:>10,} depth={row['and_depth']:>5} "
                  f"XOR={row['n_xor']:>11,} ({row['build_seconds']}s)", flush=True)

    doc: Dict = {
        "generated_by": "python -m couplet.circuits_ppml", "design_version": DESIGN_VERSION,
        "note": "All figures are the output of the circuit compiler (dead gates removed), ripple-carry "
                "adders, k = s = 64 unless stated.  A site is site(n, f, mode) of DESIGN_M3.md section 2; "
                "the comparison bundle is the core's gate with payload-1 DCFs at c = floor(2^n/3) and 0, "
                "the wrap bits [[w_c]] and an authenticated offset (gate_preset 'comparison_auth').",
        "k": k, "s": s, "sites": [], "beaver_triples": [], "matrix_triples": [], "input_masks": [],
        "comparison_bundles": [],
    }
    for n in ns:
        for f in fs:
            for mode in MODES:
                if f == 0 and mode == "exact":
                    continue            # f = 0: both modes build the plain offset gate (one row)
                row = _row(site_descriptor(n, f, k, s, mode), kind="site", n=n, f=f, mode=mode)
                doc["sites"].append(row)
                say(row)
        row = _row(cg.gate_preset("comparison_auth", n, k, s), kind="comparison_bundle", n=n)
        doc["comparison_bundles"].append(row)
        say(row)
    tr = [_row(triples_descriptor(c, k, s), kind="triples", count=c) for c in triple_counts]
    for row in tr:
        say(row)
    doc["beaver_triples"] = tr
    doc["beaver_triple_fit"] = {"per_triple": tr[1]["n_and"] - tr[0]["n_and"],
                                "fixed": 2 * tr[0]["n_and"] - tr[1]["n_and"],
                                "exact_for_all_counts": all(
                                    r["n_and"] == tr[0]["n_and"] + (r["count"] - 1) * (tr[1]["n_and"] - tr[0]["n_and"])
                                    for r in tr)}
    mt = [_row(mattriple_descriptor(r, c, k, s), kind="mattriple", rows=r, cols=c) for r, c in mat_shapes]
    for row in mt:
        say(row)
    doc["matrix_triples"] = mt
    doc["matrix_triple_fit"] = _solve_affine(mt)
    im = [_row(inmask_descriptor(0, c, k, s), kind="inmask", count=c) for c in inmask_counts]
    for row in im:
        say(row)
    doc["input_masks"] = im
    doc["input_mask_fit"] = {"per_mask": im[1]["n_and"] - im[0]["n_and"],
                             "fixed": 2 * im[0]["n_and"] - im[1]["n_and"]}
    return doc


def default_stats_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "experiments", "figdata", "ppml_circuit_stats.json"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="write the AND-gate table of the Couplet generation circuits")
    ap.add_argument("--out", default=None, help="JSON path (default: experiments/figdata/ppml_circuit_stats.json)")
    ap.add_argument("--ns", default="8,12,14,16,20")
    ap.add_argument("--fs", default="0,6,8,12")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--s", type=int, default=64)
    args = ap.parse_args(argv)
    t = time.perf_counter()
    doc = stats_table([int(x) for x in args.ns.split(",")], [int(x) for x in args.fs.split(",")], args.k, args.s)
    doc["total_build_seconds"] = round(time.perf_counter() - t, 1)
    out = args.out or default_stats_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
