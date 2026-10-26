"""Plaintext two-plane DPF and DCF (DESIGN.md section 3): the bit-for-bit
reference for the generation circuits and for the native implementation.

Both schemes are the standard tree constructions (Boyle-Gilboa-Ishai DPF,
Boyle et al. DCF) whose payload lives in Z_{2^w} x Z_{2^w}, w = k + s: the
value plane carries beta (zero-extended from k bits) and the tag plane
carries beta_t = Delta * beta mod 2^w, so every local evaluation is a SPDZ2k
authenticated share.  The MAC key is the integer Delta = Delta_0 + Delta_1 in
[0, 2^(s+1)) and is NEVER reduced mod 2^s.  Indices are n-bit integers read
MSB first.  ``neg_if(t, v) = -v if t == 1 else v`` (mod 2^w); the final
correction words use ``neg_if(t1, .)`` where t1 is party 1's control bit
after the last level, the DCF level words use the control before the update.

Seeds are 16-byte strings; PRG maps come from ``couplet.ascon``.  Scalar
Eval follows one root-to-leaf path; EvalAll expands breadth first with the
numpy-vectorised permutation (one permutation call per level, all labels
stacked) and returns ``np.ndarray`` objects of dtype ``object`` holding Python
ints in [0, 2^w), leaves in index order.

Randomness.  ``*_gen(..., rng=None, roots=None)``: with ``roots=(s0, s1)``
(two 16-byte strings) generation is deterministic, which is how the output of
a generation circuit is compared bit for bit; otherwise the two roots are
drawn as ``rng.randbytes(16)`` twice (``random.Random``,
``couplet.rng.Sha256CounterRng``; ``None`` uses ``os.urandom``).

Key encoding (``to_bytes``/``from_bytes``; every integer little-endian;
W = ceil((k+s)/8)):
    u32 L                 byte length of the body that follows
    u8  family            1 = DPF, 2 = DCF
    u8  b, u8 n, u8 k, u8 s
    16B root seed s_b
    per level i = 1..n:
        16B CWs_i
        u8  control byte: bit0 = CWtL_i, bit1 = CWtR_i
        [DCF only] W bytes VCW_i, W bytes TCW_i
    W bytes CWout_v, W bytes CWout_t
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from . import ascon

FAMILY_DPF = 1
FAMILY_DCF = 2
SEED_BYTES = ascon.SEED_BYTES


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DPFKey:
    """Party b's DPF key: root seed plus the public correction words."""

    family: ClassVar[int] = FAMILY_DPF

    b: int
    n: int
    k: int
    s: int
    seed: bytes
    cw_s: tuple[bytes, ...]
    cw_tl: tuple[int, ...]
    cw_tr: tuple[int, ...]
    cw_out_v: int
    cw_out_t: int

    @property
    def width(self) -> int:
        return self.k + self.s

    def to_bytes(self) -> bytes:
        return _encode_key(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "DPFKey":
        key = key_from_bytes(data)
        if not isinstance(key, cls):
            raise ValueError("encoded key is not a DPF key")
        return key

    def size_bytes(self) -> int:
        return len(self.to_bytes())


@dataclass(frozen=True)
class DCFKey:
    """Party b's DCF key: as ``DPFKey`` plus per-level value/tag correction words."""

    family: ClassVar[int] = FAMILY_DCF

    b: int
    n: int
    k: int
    s: int
    seed: bytes
    cw_s: tuple[bytes, ...]
    cw_tl: tuple[int, ...]
    cw_tr: tuple[int, ...]
    cw_v: tuple[int, ...]
    cw_t: tuple[int, ...]
    cw_out_v: int
    cw_out_t: int

    @property
    def width(self) -> int:
        return self.k + self.s

    def to_bytes(self) -> bytes:
        return _encode_key(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "DCFKey":
        key = key_from_bytes(data)
        if not isinstance(key, cls):
            raise ValueError("encoded key is not a DCF key")
        return key

    def size_bytes(self) -> int:
        return len(self.to_bytes())


def _word_bytes(width: int) -> int:
    return (width + 7) // 8


def _encode_key(key) -> bytes:
    w = _word_bytes(key.width)
    dcf = key.family == FAMILY_DCF
    body = bytearray(bytes((key.family, key.b, key.n, key.k, key.s)))
    body += key.seed
    for i in range(key.n):
        body += key.cw_s[i]
        body.append(key.cw_tl[i] | (key.cw_tr[i] << 1))
        if dcf:
            body += key.cw_v[i].to_bytes(w, "little")
            body += key.cw_t[i].to_bytes(w, "little")
    body += key.cw_out_v.to_bytes(w, "little")
    body += key.cw_out_t.to_bytes(w, "little")
    return len(body).to_bytes(4, "little") + bytes(body)


def key_from_bytes(data: bytes):
    """Decode a DPF or DCF key (dispatching on the family byte)."""
    data = bytes(data)
    if len(data) < 4:
        raise ValueError("truncated key")
    length = int.from_bytes(data[:4], "little")
    if len(data) != 4 + length:
        raise ValueError("length prefix does not match the data length")
    body = data[4:]
    if len(body) < 5 + SEED_BYTES:
        raise ValueError("truncated key body")
    family, b, n, k, s = body[:5]
    if family not in (FAMILY_DPF, FAMILY_DCF):
        raise ValueError("unknown key family")
    if b not in (0, 1):
        raise ValueError("party index must be 0 or 1")
    _check_params(n, k, s)
    width = k + s
    w = _word_bytes(width)
    mod = 1 << width
    dcf = family == FAMILY_DCF
    expected = 5 + SEED_BYTES + n * (SEED_BYTES + 1 + (2 * w if dcf else 0)) + 2 * w
    if len(body) != expected:
        raise ValueError("key body has the wrong length for its parameters")
    pos = 5
    seed = body[pos:pos + SEED_BYTES]
    pos += SEED_BYTES
    cw_s, cw_tl, cw_tr, cw_v, cw_t = [], [], [], [], []
    for _ in range(n):
        cw_s.append(body[pos:pos + SEED_BYTES])
        pos += SEED_BYTES
        ctrl = body[pos]
        pos += 1
        if ctrl & ~3:
            raise ValueError("control byte has stray bits")
        cw_tl.append(ctrl & 1)
        cw_tr.append((ctrl >> 1) & 1)
        if dcf:
            v = int.from_bytes(body[pos:pos + w], "little")
            t = int.from_bytes(body[pos + w:pos + 2 * w], "little")
            pos += 2 * w
            if v >= mod or t >= mod:
                raise ValueError("correction word out of range")
            cw_v.append(v)
            cw_t.append(t)
    cw_out_v = int.from_bytes(body[pos:pos + w], "little")
    cw_out_t = int.from_bytes(body[pos + w:pos + 2 * w], "little")
    if cw_out_v >= mod or cw_out_t >= mod:
        raise ValueError("output correction word out of range")
    common = dict(b=b, n=n, k=k, s=s, seed=seed, cw_s=tuple(cw_s), cw_tl=tuple(cw_tl),
                  cw_tr=tuple(cw_tr), cw_out_v=cw_out_v, cw_out_t=cw_out_t)
    if dcf:
        return DCFKey(cw_v=tuple(cw_v), cw_t=tuple(cw_t), **common)
    return DPFKey(**common)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _check_params(n: int, k: int, s: int) -> None:
    if not 1 <= n <= 64:
        raise ValueError("n must be in 1..64")
    if k < 1 or s < 0 or 2 * (k + s) > 8 * ascon.STATE_BYTES:
        raise ValueError("need k >= 1, s >= 0 and 2(k+s) <= 320")


def _check_inputs(alpha: int, beta: int, delta: int, n: int, k: int, s: int) -> None:
    _check_params(n, k, s)
    if not 0 <= alpha < (1 << n):
        raise ValueError("alpha out of range")
    if not 0 <= beta < (1 << k):
        raise ValueError("beta out of range")
    if not 0 <= delta < (1 << (s + 1)):
        raise ValueError("delta out of range: need 0 <= Delta < 2^(s+1)")


def neg_if(t: int, v: int, mod: int) -> int:
    """neg_if(t, v) = -v if t == 1 else v, in Z_mod."""
    return (-v if t else v) % mod


def seed_from_int(value: int) -> bytes:
    """16-byte seed from a 128-bit integer (bit j of the integer = string bit j)."""
    return int(value).to_bytes(SEED_BYTES, "little")


def seed_to_int(seed: bytes) -> int:
    """Inverse of ``seed_from_int``."""
    if len(seed) != SEED_BYTES:
        raise ValueError("seed must be 16 bytes")
    return int.from_bytes(seed, "little")


def _roots(rng, roots) -> tuple[bytes, bytes]:
    """The two root seeds: explicit, or two 16-byte draws (s0 first)."""
    if roots is None:
        root0 = _randbytes(rng, SEED_BYTES)
        root1 = _randbytes(rng, SEED_BYTES)
        return root0, root1
    if rng is not None:
        raise ValueError("pass either rng or roots, not both")
    root0, root1 = (bytes(r) for r in roots)
    if len(root0) != SEED_BYTES or len(root1) != SEED_BYTES:
        raise ValueError("roots must be two 16-byte strings")
    return root0, root1


def _check_eval(b: int, key, x: int | None = None) -> None:
    if b not in (0, 1) or key.b != b:
        raise ValueError("party index does not match the key")
    if x is not None and not 0 <= x < (1 << key.n):
        raise ValueError("x out of range")


def _randbytes(rng, n: int) -> bytes:
    return os.urandom(n) if rng is None else rng.randbytes(n)


def _xor16(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(SEED_BYTES, "little")


def _bit(x: int, n: int, i: int) -> int:
    """i-th bit (1-based, MSB first) of the n-bit integer x."""
    return (x >> (n - i)) & 1


def _interleave(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Children in index order: left child of node j at 2j, right child at 2j+1."""
    out = np.empty(2 * left.size, dtype=left.dtype)
    out[0::2] = left
    out[1::2] = right
    return out


def _expand_level_np(lo, hi, t, cw_s: bytes, cw_tl: int, cw_tr: int, labels):
    """One breadth-first level: apply G (+ extra labels) to all nodes and correct.

    Returns (child_lo, child_hi, child_t, extra_states) with children
    interleaved L/R; ``extra_states`` are the permuted states for labels[1:].
    """
    states = ascon.prg_state_np(lo, hi, labels)
    sl_lo, sl_hi, t_l, sr_lo, sr_hi, t_r = ascon.parse_G_state_np(states[0])
    mask = np.where(t.astype(bool), np.uint64(ascon.MASK64), np.uint64(0))
    cw_lo, cw_hi = ascon.words_from_bytes(cw_s)
    sl_lo ^= mask & np.uint64(cw_lo)
    sl_hi ^= mask & np.uint64(cw_hi)
    sr_lo ^= mask & np.uint64(cw_lo)
    sr_hi ^= mask & np.uint64(cw_hi)
    t_l ^= t & np.uint8(cw_tl)
    t_r ^= t & np.uint8(cw_tr)
    return _interleave(sl_lo, sr_lo), _interleave(sl_hi, sr_hi), _interleave(t_l, t_r), states[1:]


def _root_np(key, b: int):
    lo, hi = ascon.words_from_bytes(key.seed)
    return (np.array([lo], dtype=np.uint64), np.array([hi], dtype=np.uint64),
            np.array([b], dtype=np.uint8))


# --------------------------------------------------------------------------- #
# DPF
# --------------------------------------------------------------------------- #

def dpf_gen(alpha: int, beta: int, delta: int, n: int, k: int, s: int, rng=None,
            roots=None) -> tuple[DPFKey, DPFKey]:
    """DPF.Gen(alpha, beta, Delta; s0_root, s1_root): keys for
    f(x) = (beta, Delta*beta mod 2^w) * [x == alpha], Delta in [0, 2^(s+1))."""
    _check_inputs(alpha, beta, delta, n, k, s)
    mod = 1 << (k + s)
    beta_t = delta * beta % mod
    root0, root1 = _roots(rng, roots)
    s0, s1, t0, t1 = root0, root1, 0, 1
    cw_s, cw_tl, cw_tr = [], [], []
    for i in range(1, n + 1):
        a = _bit(alpha, n, i)
        sl0, tl0, sr0, tr0 = ascon.G(s0)
        sl1, tl1, sr1, tr1 = ascon.G(s1)
        if a == 0:
            keep0, keep1, lose0, lose1 = (sl0, tl0), (sl1, tl1), sr0, sr1
        else:
            keep0, keep1, lose0, lose1 = (sr0, tr0), (sr1, tr1), sl0, sl1
        cws = _xor16(lose0, lose1)
        ctl = tl0 ^ tl1 ^ a ^ 1
        ctr = tr0 ^ tr1 ^ a
        ct_keep = ctr if a else ctl
        cw_s.append(cws)
        cw_tl.append(ctl)
        cw_tr.append(ctr)
        s0 = _xor16(keep0[0], cws) if t0 else keep0[0]
        s1 = _xor16(keep1[0], cws) if t1 else keep1[0]
        t0 = keep0[1] ^ (t0 & ct_keep)
        t1 = keep1[1] ^ (t1 & ct_keep)
    c0v, c0t = ascon.conv(s0, k, s)
    c1v, c1t = ascon.conv(s1, k, s)
    cw_out_v = neg_if(t1, beta - c0v + c1v, mod)
    cw_out_t = neg_if(t1, beta_t - c0t + c1t, mod)
    common = dict(n=n, k=k, s=s, cw_s=tuple(cw_s), cw_tl=tuple(cw_tl), cw_tr=tuple(cw_tr),
                  cw_out_v=cw_out_v, cw_out_t=cw_out_t)
    return DPFKey(b=0, seed=root0, **common), DPFKey(b=1, seed=root1, **common)


def dpf_eval(b: int, key: DPFKey, x: int) -> tuple[int, int]:
    """DPF.Eval(b, k_b, x) -> (y_b, tau_b), both mod 2^(k+s)."""
    _check_eval(b, key, x)
    n, k, s = key.n, key.k, key.s
    mod = 1 << key.width
    seed, t = key.seed, b
    for i in range(1, n + 1):
        sl, tl, sr, tr = ascon.G(seed)
        if _bit(x, n, i) == 0:
            nxt, tn, ct = sl, tl, key.cw_tl[i - 1]
        else:
            nxt, tn, ct = sr, tr, key.cw_tr[i - 1]
        if t:
            nxt = _xor16(nxt, key.cw_s[i - 1])
        seed, t = nxt, tn ^ (t & ct)
    cv, ctag = ascon.conv(seed, k, s)
    sign = -1 if b else 1
    return sign * (cv + t * key.cw_out_v) % mod, sign * (ctag + t * key.cw_out_t) % mod


def dpf_eval_all(b: int, key: DPFKey) -> tuple[np.ndarray, np.ndarray]:
    """DPF.EvalAll(b, k_b) -> (y_b, tau_b): object arrays of length 2^n."""
    _check_eval(b, key)
    mod = 1 << key.width
    lo, hi, t = _root_np(key, b)
    for i in range(key.n):
        lo, hi, t, _ = _expand_level_np(lo, hi, t, key.cw_s[i], key.cw_tl[i], key.cw_tr[i],
                                        (ascon.LABEL_G,))
    cv, ct = ascon.conv_np(lo, hi, key.k, key.s)
    sign = -1 if b else 1
    tt = t.astype(object)
    y = (sign * (cv + tt * key.cw_out_v)) % mod
    tau = (sign * (ct + tt * key.cw_out_t)) % mod
    return y, tau


# --------------------------------------------------------------------------- #
# DCF
# --------------------------------------------------------------------------- #

def dcf_gen(alpha: int, beta: int, delta: int, n: int, k: int, s: int, rng=None,
            roots=None) -> tuple[DCFKey, DCFKey]:
    """DCF.Gen(alpha, beta, Delta; roots): keys for
    f(x) = (beta, Delta*beta mod 2^w) * [x < alpha], Delta in [0, 2^(s+1))."""
    _check_inputs(alpha, beta, delta, n, k, s)
    mod = 1 << (k + s)
    dbeta = delta * beta % mod
    root0, root1 = _roots(rng, roots)
    s0, s1, t0, t1 = root0, root1, 0, 1
    va = ta = 0
    cw_s, cw_tl, cw_tr, cw_v, cw_t = [], [], [], [], []
    for i in range(1, n + 1):
        a = _bit(alpha, n, i)
        sl0, tl0, sr0, tr0 = ascon.G(s0)
        sl1, tl1, sr1, tr1 = ascon.G(s1)
        vl0, wl0 = ascon.convv_L(s0, k, s)
        vr0, wr0 = ascon.convv_R(s0, k, s)
        vl1, wl1 = ascon.convv_L(s1, k, s)
        vr1, wr1 = ascon.convv_R(s1, k, s)
        if a == 0:  # keep = L, lose = R
            keep0, keep1, lose0, lose1 = (sl0, tl0), (sl1, tl1), sr0, sr1
            v_keep0, w_keep0, v_keep1, w_keep1 = vl0, wl0, vl1, wl1
            v_lose0, w_lose0, v_lose1, w_lose1 = vr0, wr0, vr1, wr1
        else:  # keep = R, lose = L
            keep0, keep1, lose0, lose1 = (sr0, tr0), (sr1, tr1), sl0, sl1
            v_keep0, w_keep0, v_keep1, w_keep1 = vr0, wr0, vr1, wr1
            v_lose0, w_lose0, v_lose1, w_lose1 = vl0, wl0, vl1, wl1
        cws = _xor16(lose0, lose1)
        ctl = tl0 ^ tl1 ^ a ^ 1
        ctr = tr0 ^ tr1 ^ a
        ct_keep = ctr if a else ctl
        # t1 is the control BEFORE this level's update
        vcw = neg_if(t1, v_lose1 - v_lose0 - va + (beta if a else 0), mod)
        tcw = neg_if(t1, w_lose1 - w_lose0 - ta + (dbeta if a else 0), mod)
        va = (va - v_keep1 + v_keep0 + neg_if(t1, vcw, mod)) % mod
        ta = (ta - w_keep1 + w_keep0 + neg_if(t1, tcw, mod)) % mod
        cw_s.append(cws)
        cw_tl.append(ctl)
        cw_tr.append(ctr)
        cw_v.append(vcw)
        cw_t.append(tcw)
        s0 = _xor16(keep0[0], cws) if t0 else keep0[0]
        s1 = _xor16(keep1[0], cws) if t1 else keep1[0]
        t0 = keep0[1] ^ (t0 & ct_keep)
        t1 = keep1[1] ^ (t1 & ct_keep)
    c0v, c0t = ascon.conv(s0, k, s)
    c1v, c1t = ascon.conv(s1, k, s)
    cw_out_v = neg_if(t1, c1v - c0v - va, mod)
    cw_out_t = neg_if(t1, c1t - c0t - ta, mod)
    common = dict(n=n, k=k, s=s, cw_s=tuple(cw_s), cw_tl=tuple(cw_tl), cw_tr=tuple(cw_tr),
                  cw_v=tuple(cw_v), cw_t=tuple(cw_t), cw_out_v=cw_out_v, cw_out_t=cw_out_t)
    return DCFKey(b=0, seed=root0, **common), DCFKey(b=1, seed=root1, **common)


def dcf_eval(b: int, key: DCFKey, x: int) -> tuple[int, int]:
    """DCF.Eval(b, k_b, x) -> (V_b, T_b), both mod 2^(k+s).

    At every level ConvV_{x_i} is evaluated on the seed and control bit
    BEFORE that level's update."""
    _check_eval(b, key, x)
    n, k, s = key.n, key.k, key.s
    mod = 1 << key.width
    sign = -1 if b else 1
    seed, t = key.seed, b
    value = tag = 0
    for i in range(1, n + 1):
        sl, tl, sr, tr = ascon.G(seed)
        if _bit(x, n, i) == 0:
            v, w = ascon.convv_L(seed, k, s)
            nxt, tn, ct = sl, tl, key.cw_tl[i - 1]
        else:
            v, w = ascon.convv_R(seed, k, s)
            nxt, tn, ct = sr, tr, key.cw_tr[i - 1]
        value += sign * (v + t * key.cw_v[i - 1])
        tag += sign * (w + t * key.cw_t[i - 1])
        if t:
            nxt = _xor16(nxt, key.cw_s[i - 1])
        seed, t = nxt, tn ^ (t & ct)
    cv, ctag = ascon.conv(seed, k, s)
    value += sign * (cv + t * key.cw_out_v)
    tag += sign * (ctag + t * key.cw_out_t)
    return value % mod, tag % mod


def dcf_eval_all(b: int, key: DCFKey) -> tuple[np.ndarray, np.ndarray]:
    """DCF.EvalAll(b, k_b) -> (y_b, tau_b): object arrays of length 2^n."""
    _check_eval(b, key)
    width = key.width
    mod = 1 << width
    sign = -1 if b else 1
    labels = (ascon.LABEL_G, ascon.LABEL_VL, ascon.LABEL_VR)
    lo, hi, t = _root_np(key, b)
    value = np.zeros(1, dtype=object)
    tag = np.zeros(1, dtype=object)
    for i in range(key.n):
        tt = t.astype(object)
        corr_v = tt * key.cw_v[i]
        corr_t = tt * key.cw_t[i]
        lo, hi, t, (st_l, st_r) = _expand_level_np(lo, hi, t, key.cw_s[i], key.cw_tl[i],
                                                  key.cw_tr[i], labels)
        v_l, w_l = ascon.parse_conv_state_np(st_l, width)
        v_r, w_r = ascon.parse_conv_state_np(st_r, width)
        value = _interleave(value + sign * (v_l + corr_v), value + sign * (v_r + corr_v)) % mod
        tag = _interleave(tag + sign * (w_l + corr_t), tag + sign * (w_r + corr_t)) % mod
    cv, ct = ascon.conv_np(lo, hi, key.k, key.s)
    tt = t.astype(object)
    y = (value + sign * (cv + tt * key.cw_out_v)) % mod
    tau = (tag + sign * (ct + tt * key.cw_out_t)) % mod
    return y, tau
