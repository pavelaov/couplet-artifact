"""Ascon-p12 and the Couplet PRG maps (DESIGN.md section 2.1).

The permutation ``p`` is the 12-round Ascon permutation of NIST SP 800-232
(bit for bit identical to Ascon v1.2): each round adds a round constant to
x2, applies the 5-bit S-box to all 64 bit-slices (bitsliced form, 5 AND gates
per slice), then the linear diffusion layer with rotation pairs
(19,28), (61,39), (1,6), (10,17), (7,41) on x0..x4.

Two implementations are provided and agree bit for bit:
  * ``permute``     scalar reference on five Python ints (64-bit words);
  * ``permute_np``  vectorised over many states (five ``np.uint64`` arrays).

PRG.  For a 16-byte seed ``s`` and a 24-byte label ``IV`` (ASCII, zero-padded)

    F_IV(s) = p(s || IV) XOR (s || IV)            (320 bits, feed-forward)

Bit layout of the 320-bit strings ``s || IV`` and ``Y = F_IV(s)``:

    bit j of the string  =  bit (j mod 64) of word x_{j // 64}   (bit 0 = LSB)
    bytes -> words          little-endian, eight bytes per word

    string bits   words      input s || IV              output Y = F_IV(s)
    -----------   --------   ------------------------   -------------------
      0 .. 127    x0, x1     seed s (16 bytes)          G: s_L
    128 .. 255    x2, x3     IV bytes  0 .. 15          G: s_R
    256           x4 bit 0   IV byte 16, bit 0          G: t_L
    257           x4 bit 1   IV byte 16, bit 1          G: t_R
    258 .. 319    x4         rest of IV bytes 16 .. 23  unused by G

    Conv / ConvV_L / ConvV_R with w = k + s (2w <= 320):
      value word = sum_{j<w} Y[j]     2^j       (string bits 0 .. w-1)
      tag word   = sum_{j<w} Y[w + j] 2^j       (string bits w .. 2w-1)

  * ``G(seed)``              -> (s_L, t_L, s_R, t_R), label ``Couplet-G``;
  * ``conv(seed, k, s)``     -> (c_v, c_t),           label ``Couplet-C``;
  * ``convv_L``/``convv_R``  -> (v, u),               labels ``Couplet-VL``/``-VR``.
``*_np`` variants operate on many seeds at once (seeds as two uint64 arrays).
``prg_state``/``prg_state_np`` return Y = F_IV(s) (five words), i.e. the
feed-forward is already applied; ``prg_perm_state`` returns p(s || IV) alone.

``ascon_hash_v12`` (Ascon-Hash, 2021 v1.2 parameters) and ``ascon_hash256``
(SP 800-232 parameters) exist only to validate the permutation against the
published test vectors; see ``selftest``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

MASK64 = (1 << 64) - 1
ROUNDS = 12
ROUND_CONSTANTS = (0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B)
ROTATIONS = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))

SEED_BYTES = 16
IV_BYTES = 24
STATE_BYTES = 40

LABEL_G = b"Couplet-G"
LABEL_C = b"Couplet-C"
LABEL_VL = b"Couplet-VL"
LABEL_VR = b"Couplet-VR"

# Published vectors used by ``selftest``.
ASCON_HASH_V12_IV = 0x00400C0000000100  # Ascon-Hash, v1.2 (2021)
ASCON_HASH256_IV = 0x0000080100CC0002  # Ascon-Hash256, NIST SP 800-232
ASCON_HASH_V12_EMPTY = "7346bc14f036e87ae03d0997913088f5f68411434b3cf8b54fa796a80d251f91"
ASCON_HASH256_EMPTY = "0b3be5850f2f6b98caf29f8fdea89b64a1fa70aa249b8f839bd53baa304d92b2"


# --------------------------------------------------------------------------- #
# Scalar reference permutation
# --------------------------------------------------------------------------- #

def _ror(x: int, r: int) -> int:
    return ((x >> r) | (x << (64 - r))) & MASK64


def permute(state5: Sequence[int]) -> list[int]:
    """12-round Ascon permutation on five 64-bit words (scalar reference)."""
    if len(state5) != 5:
        raise ValueError("state must have exactly five 64-bit words")
    x0, x1, x2, x3, x4 = (int(w) & MASK64 for w in state5)
    for c in ROUND_CONSTANTS:
        # constant addition
        x2 ^= c
        # substitution layer (bitsliced S-box, 5 AND per slice)
        x0 ^= x4
        x4 ^= x3
        x2 ^= x1
        t0 = (x0 ^ MASK64) & x1
        t1 = (x1 ^ MASK64) & x2
        t2 = (x2 ^ MASK64) & x3
        t3 = (x3 ^ MASK64) & x4
        t4 = (x4 ^ MASK64) & x0
        x0 ^= t1
        x1 ^= t2
        x2 ^= t3
        x3 ^= t4
        x4 ^= t0
        x1 ^= x0
        x0 ^= x4
        x3 ^= x2
        x2 ^= MASK64
        # linear diffusion layer
        x0 ^= _ror(x0, 19) ^ _ror(x0, 28)
        x1 ^= _ror(x1, 61) ^ _ror(x1, 39)
        x2 ^= _ror(x2, 1) ^ _ror(x2, 6)
        x3 ^= _ror(x3, 10) ^ _ror(x3, 17)
        x4 ^= _ror(x4, 7) ^ _ror(x4, 41)
    return [x0, x1, x2, x3, x4]


# --------------------------------------------------------------------------- #
# Vectorised permutation (numpy uint64, many states at once)
# --------------------------------------------------------------------------- #

_U64 = np.uint64
_RC_NP = tuple(_U64(c) for c in ROUND_CONSTANTS)
_ROT_NP = tuple((_U64(a), _U64(64 - a), _U64(b), _U64(64 - b)) for a, b in ROTATIONS)


def _diffuse(x: np.ndarray, a, na, b, nb) -> None:
    """x ^= ror(x, a) ^ ror(x, b), in place."""
    x ^= ((x >> a) | (x << na)) ^ ((x >> b) | (x << nb))


def permute_np(x0, x1, x2, x3, x4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """12-round Ascon permutation applied elementwise to many states.

    Inputs are ``np.uint64`` arrays of one common shape (element j of each
    array is word x_i of state j).  The inputs are not modified; five new
    arrays are returned.
    """
    x0, x1, x2, x3, x4 = (np.array(x, dtype=np.uint64) for x in (x0, x1, x2, x3, x4))
    if not (x0.shape == x1.shape == x2.shape == x3.shape == x4.shape):
        raise ValueError("all five word arrays must have the same shape")
    for c in _RC_NP:
        x2 ^= c
        x0 ^= x4
        x4 ^= x3
        x2 ^= x1
        t0 = ~x0 & x1
        t1 = ~x1 & x2
        t2 = ~x2 & x3
        t3 = ~x3 & x4
        t4 = ~x4 & x0
        x0 ^= t1
        x1 ^= t2
        x2 ^= t3
        x3 ^= t4
        x4 ^= t0
        x1 ^= x0
        x0 ^= x4
        x3 ^= x2
        np.invert(x2, out=x2)
        _diffuse(x0, *_ROT_NP[0])
        _diffuse(x1, *_ROT_NP[1])
        _diffuse(x2, *_ROT_NP[2])
        _diffuse(x3, *_ROT_NP[3])
        _diffuse(x4, *_ROT_NP[4])
    return x0, x1, x2, x3, x4


# --------------------------------------------------------------------------- #
# Byte / word helpers (little-endian per 8-byte chunk)
# --------------------------------------------------------------------------- #

def words_from_bytes(data: bytes) -> list[int]:
    """Split ``data`` (length multiple of 8) into 64-bit words, LE per chunk."""
    if len(data) % 8:
        raise ValueError("length must be a multiple of 8")
    return [int.from_bytes(data[i:i + 8], "little") for i in range(0, len(data), 8)]


def bytes_from_words(words: Sequence[int]) -> bytes:
    """Inverse of ``words_from_bytes``."""
    return b"".join((int(w) & MASK64).to_bytes(8, "little") for w in words)


def label_words(label: bytes) -> tuple[int, int, int]:
    """IV words (x2, x3, x4) for an ASCII label: zero-padded to 24 bytes, LE."""
    if len(label) > IV_BYTES:
        raise ValueError("label longer than 24 bytes")
    w = words_from_bytes(label.ljust(IV_BYTES, b"\x00"))
    return w[0], w[1], w[2]


# --------------------------------------------------------------------------- #
# Scalar PRG maps
# --------------------------------------------------------------------------- #

def prg_perm_state(seed: bytes, label: bytes) -> list[int]:
    """p(seed || IV_label) WITHOUT the feed-forward (five words)."""
    if len(seed) != SEED_BYTES:
        raise ValueError("seed must be 16 bytes")
    s0, s1 = words_from_bytes(seed)
    return permute([s0, s1, *label_words(label)])


def prg_state(seed: bytes, label: bytes) -> list[int]:
    """Y = F_IV(seed) = p(seed || IV) XOR (seed || IV) as five 64-bit words."""
    if len(seed) != SEED_BYTES:
        raise ValueError("seed must be 16 bytes")
    inp = [*words_from_bytes(seed), *label_words(label)]
    return [y ^ x for y, x in zip(permute(inp), inp)]


def G(seed: bytes) -> tuple[bytes, int, bytes, int]:
    """Seed-doubling PRG from Y = F_G(seed): returns (s_L, t_L, s_R, t_R)."""
    x0, x1, x2, x3, x4 = prg_state(seed, LABEL_G)
    return bytes_from_words((x0, x1)), x4 & 1, bytes_from_words((x2, x3)), (x4 >> 1) & 1


def conv_from_state(state: Sequence[int], width: int) -> tuple[int, int]:
    """(value, tag): string bits 0..width-1 and width..2*width-1 of Y."""
    if width < 1 or 2 * width > 8 * STATE_BYTES:
        raise ValueError("need 1 <= width and 2*width <= 320")
    full = int.from_bytes(bytes_from_words(state), "little")
    mask = (1 << width) - 1
    return full & mask, (full >> width) & mask


def conv(seed: bytes, k: int, s: int) -> tuple[int, int]:
    """Leaf convert map: (c_val, c_tag), each mod 2^(k+s)."""
    return conv_from_state(prg_state(seed, LABEL_C), k + s)


def convv_L(seed: bytes, k: int, s: int) -> tuple[int, int]:
    """DCF left value/tag words, each mod 2^(k+s)."""
    return conv_from_state(prg_state(seed, LABEL_VL), k + s)


def convv_R(seed: bytes, k: int, s: int) -> tuple[int, int]:
    """DCF right value/tag words, each mod 2^(k+s)."""
    return conv_from_state(prg_state(seed, LABEL_VR), k + s)


# --------------------------------------------------------------------------- #
# Vectorised PRG maps (seeds as two uint64 arrays: lo = x0 words, hi = x1 words)
# --------------------------------------------------------------------------- #

def prg_state_np(lo, hi, labels: Sequence[bytes]) -> list[tuple[np.ndarray, ...]]:
    """Y = F_IV(seed) for every seed and every label in ONE permutation call.

    ``lo``/``hi`` are ``np.uint64`` arrays of equal length N.  All
    len(labels)*N states are permuted together and the feed-forward (XOR
    with the permutation input) is applied; the result is one 5-tuple of
    length-N arrays per label, in label order.
    """
    lo = np.asarray(lo, dtype=np.uint64).ravel()
    hi = np.asarray(hi, dtype=np.uint64).ravel()
    if lo.shape != hi.shape:
        raise ValueError("lo and hi must have the same length")
    n, m = lo.size, len(labels)
    ivs = np.array([label_words(lab) for lab in labels], dtype=np.uint64).reshape(m, 3)
    x0 = np.tile(lo, m)
    x1 = np.tile(hi, m)
    x2 = np.repeat(ivs[:, 0], n)
    x3 = np.repeat(ivs[:, 1], n)
    x4 = np.repeat(ivs[:, 2], n)
    out = permute_np(x0, x1, x2, x3, x4)      # works on copies of the inputs
    for y, x in zip(out, (x0, x1, x2, x3, x4)):
        y ^= x                                # feed-forward
    return [tuple(w[j * n:(j + 1) * n] for w in out) for j in range(m)]


def parse_G_state_np(state: Sequence[np.ndarray]):
    """(sL_lo, sL_hi, tL, sR_lo, sR_hi, tR) from Y = F_G(s); t's are uint8."""
    x0, x1, x2, x3, x4 = state
    t_l = (x4 & _U64(1)).astype(np.uint8)
    t_r = ((x4 >> _U64(1)) & _U64(1)).astype(np.uint8)
    return x0, x1, t_l, x2, x3, t_r


def parse_conv_state_np(state: Sequence[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
    """(value, tag) object arrays of Python ints mod 2^width from Y = F_IV(s)."""
    if width < 1 or 2 * width > 8 * STATE_BYTES:
        raise ValueError("need 1 <= width and 2*width <= 320")
    full = state[0].astype(object)
    for j in range(1, 5):
        full |= state[j].astype(object) << (64 * j)
    mask = (1 << width) - 1
    return full & mask, (full >> width) & mask


def G_np(lo, hi):
    """Vectorised ``G``: (sL_lo, sL_hi, tL, sR_lo, sR_hi, tR)."""
    (state,) = prg_state_np(lo, hi, (LABEL_G,))
    return parse_G_state_np(state)


def conv_np(lo, hi, k: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``conv``: (value, tag) object arrays."""
    (state,) = prg_state_np(lo, hi, (LABEL_C,))
    return parse_conv_state_np(state, k + s)


def convv_L_np(lo, hi, k: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``convv_L``."""
    (state,) = prg_state_np(lo, hi, (LABEL_VL,))
    return parse_conv_state_np(state, k + s)


def convv_R_np(lo, hi, k: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``convv_R``."""
    (state,) = prg_state_np(lo, hi, (LABEL_VR,))
    return parse_conv_state_np(state, k + s)


# --------------------------------------------------------------------------- #
# Hash functions for validation only
# --------------------------------------------------------------------------- #

def _sponge_hash(msg: bytes, iv: int, byteorder: str, pad_byte: bytes) -> bytes:
    """Ascon hash sponge: rate 8 bytes, p^12 everywhere, 32-byte output."""
    st = permute([iv, 0, 0, 0, 0])
    padded = msg + pad_byte + b"\x00" * ((-len(msg) - 1) % 8)
    for i in range(0, len(padded), 8):
        st[0] ^= int.from_bytes(padded[i:i + 8], byteorder)
        st = permute(st)
    out = []
    for j in range(4):
        if j:
            st = permute(st)
        out.append(st[0].to_bytes(8, byteorder))
    return b"".join(out)


def ascon_hash_v12(msg: bytes) -> bytes:
    """Ascon-Hash, v1.2 (2021): IV 0x00400c0000000100, big-endian words, pad 0x80."""
    return _sponge_hash(msg, ASCON_HASH_V12_IV, "big", b"\x80")


def ascon_hash256(msg: bytes) -> bytes:
    """Ascon-Hash256, SP 800-232: IV 0x0000080100cc0002, little-endian words, pad 0x01."""
    return _sponge_hash(msg, ASCON_HASH256_IV, "little", b"\x01")


def selftest() -> None:
    """Check the permutation against the published hash vectors, the
    scalar/numpy parity of the permutation, and the scalar/numpy parity of
    the PRG maps including the feed-forward."""
    if ascon_hash_v12(b"").hex() != ASCON_HASH_V12_EMPTY:
        raise AssertionError("Ascon-Hash v1.2 empty-message vector mismatch")
    if ascon_hash256(b"").hex() != ASCON_HASH256_EMPTY:
        raise AssertionError("Ascon-Hash256 (SP 800-232) empty-message vector mismatch")
    rng = np.random.default_rng(0x5EED)
    words = rng.integers(0, 2**64, size=(5, 64), dtype=np.uint64)
    vec = permute_np(*words)
    for j in range(words.shape[1]):
        ref = permute([int(words[i, j]) for i in range(5)])
        if [int(vec[i][j]) for i in range(5)] != ref:
            raise AssertionError("permute_np disagrees with permute")
    labels = (LABEL_G, LABEL_C, LABEL_VL, LABEL_VR)
    lo, hi = words[0, :8], words[1, :8]
    states = prg_state_np(lo, hi, labels)
    for j in range(lo.size):
        seed = bytes_from_words((int(lo[j]), int(hi[j])))
        for lab, st in zip(labels, states):
            inp = [int(lo[j]), int(hi[j]), *label_words(lab)]
            ref = [y ^ x for y, x in zip(permute(inp), inp)]
            if prg_state(seed, lab) != ref or [int(w[j]) for w in st] != ref:
                raise AssertionError("PRG feed-forward mismatch (scalar vs numpy)")
        g = G_np(lo[j:j + 1], hi[j:j + 1])
        got = (bytes_from_words((int(g[0][0]), int(g[1][0]))), int(g[2][0]),
               bytes_from_words((int(g[3][0]), int(g[4][0]))), int(g[5][0]))
        if got != G(seed):
            raise AssertionError("G_np disagrees with G")
        for fn, fn_np in ((conv, conv_np), (convv_L, convv_L_np), (convv_R, convv_R_np)):
            for k, s in ((32, 64), (64, 64)):
                v, t = fn_np(lo[j:j + 1], hi[j:j + 1], k, s)
                if (int(v[0]), int(t[0])) != fn(seed, k, s):
                    raise AssertionError("vectorised conv map disagrees with the scalar map")


if __name__ == "__main__":
    selftest()
    print("ascon selftest ok")
