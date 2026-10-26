"""Boolean-circuit Ascon-p12 and the Couplet PRG maps (DESIGN.md section 2.1).

Bit layout (identical to ``couplet.ascon``): a 320-bit string is a flat
little-endian wire list; string bit ``j`` is bit ``j mod 64`` of word
``x_{j // 64}``, i.e. bit ``j`` of word ``x_i`` sits at index ``64*i + j``.
A seed is 128 wires occupying ``x0 || x1`` (indices 0..127); the public IV
(ASCII label zero-padded to 24 bytes, little-endian per 8-byte chunk)
occupies ``x2 || x3 || x4`` as constants.

PRG: ``F_IV(s) = p(s || IV) XOR (s || IV)``.  The feed-forward costs no AND
gate: 128 XOR gates with the seed wires, and the XOR with the constant IV
folds into NOT gates.  The maps read ``Y = F_IV(s)``:

    prg_G        s_L = Y[0..127], s_R = Y[128..255], t_L = Y[256], t_R = Y[257]
    conv/convV   value = Y[0..w-1], tag = Y[w..2w-1],  w = k + s

``ascon_p12`` on 320 fresh wires costs exactly 64*5*12 = 3840 AND gates in
12 AND layers.  Inside the PRG maps the IV words are constants, so the
builder's constant folding removes four of the five first-round S-box ANDs
per slice: t2, t3, t4 have a public operand, and t1 = ~x1 & (x1 ^ const)
folds to 0 or ~x1.  One call on a secret seed then costs
3840 - 256 = 3584 ANDs, still 12 AND layers.

Scalar helpers (``G_ref``, ``conv_ref`` ...) on 128-bit integer seeds are
provided for plaintext checking; they use ``couplet.ascon.permute`` when
importable and a local minimal scalar round function otherwise.
"""

from __future__ import annotations

import warnings
from typing import List, Sequence, Tuple

from .circuit import Builder

ROUND_CONSTANTS = (0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B)
ROTATIONS = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))

WORD = 64
STATE_BITS = 320
SEED_BITS = 128
IV_BITS = 192

LABEL_G = b"Couplet-G"
LABEL_C = b"Couplet-C"
LABEL_VL = b"Couplet-VL"
LABEL_VR = b"Couplet-VR"


def iv_int(label: bytes) -> int:
    """192-bit IV as a little-endian integer (byte 0 -> bits 0..7 of x2)."""
    if len(label) > 24:
        raise ValueError("label longer than 24 bytes")
    return int.from_bytes(label.ljust(24, b"\x00"), "little")


IV_G = iv_int(LABEL_G)
IV_C = iv_int(LABEL_C)
IV_VL = iv_int(LABEL_VL)
IV_VR = iv_int(LABEL_VR)


# --------------------------------------------------------------------------- #
# Circuit
# --------------------------------------------------------------------------- #

def ascon_round(B: Builder, x: Sequence[Sequence[int]], c: int) -> List[List[int]]:
    """One Ascon round on five 64-wire words (reference bitsliced S-box)."""
    x0, x1, x2, x3, x4 = (list(w) for w in x)
    # round constant on x2 (XOR with a public constant: NOT gates after folding)
    for j in range(8):
        if (c >> j) & 1:
            x2[j] = B.not_(x2[j])
    # substitution layer
    x0 = B.xor_bits(x0, x4)
    x4 = B.xor_bits(x4, x3)
    x2 = B.xor_bits(x2, x1)
    t0 = [B.and_(B.not_(p), q) for p, q in zip(x0, x1)]
    t1 = [B.and_(B.not_(p), q) for p, q in zip(x1, x2)]
    t2 = [B.and_(B.not_(p), q) for p, q in zip(x2, x3)]
    t3 = [B.and_(B.not_(p), q) for p, q in zip(x3, x4)]
    t4 = [B.and_(B.not_(p), q) for p, q in zip(x4, x0)]
    x0 = B.xor_bits(x0, t1)
    x1 = B.xor_bits(x1, t2)
    x2 = B.xor_bits(x2, t3)
    x3 = B.xor_bits(x3, t4)
    x4 = B.xor_bits(x4, t0)
    x1 = B.xor_bits(x1, x0)
    x0 = B.xor_bits(x0, x4)
    x3 = B.xor_bits(x3, x2)
    x2 = B.not_bits(x2)
    # linear diffusion layer
    words = [x0, x1, x2, x3, x4]
    out = []
    for w, (r1, r2) in zip(words, ROTATIONS):
        out.append(B.xor_bits(B.xor_bits(w, B.rotr(w, r1)), B.rotr(w, r2)))
    return out


def ascon_p12(B: Builder, state_bits: Sequence[int]) -> List[int]:
    """12-round Ascon permutation on 320 wires (flat layout, see module doc)."""
    if len(state_bits) != STATE_BITS:
        raise ValueError("state must have 320 wires")
    x = [list(state_bits[WORD * i:WORD * (i + 1)]) for i in range(5)]
    for c in ROUND_CONSTANTS:
        x = ascon_round(B, x, c)
    return [w for word in x for w in word]


def prg_F(B: Builder, seed_bits: Sequence[int], iv: int) -> List[int]:
    """Y = F_IV(s) = p(s || IV) XOR (s || IV) on 128 seed wires (320 output wires)."""
    if len(seed_bits) != SEED_BITS:
        raise ValueError("seed must have 128 wires")
    if iv < 0 or iv >> IV_BITS:
        raise ValueError("IV must be a 192-bit integer")
    inp = list(seed_bits) + B.const_bits(iv, IV_BITS)
    return B.xor_bits(ascon_p12(B, inp), inp)       # feed-forward: XOR/NOT gates only


_prg_state = prg_F


def prg_G(B: Builder, seed_bits: Sequence[int]) -> Tuple[List[int], int, List[int], int]:
    """G(s) = (s_L, t_L, s_R, t_R) from Y = F_G(s): s_L = Y[0..127],
    s_R = Y[128..255], t_L = Y[256], t_R = Y[257]."""
    st = _prg_state(B, seed_bits, IV_G)
    return st[0:128], st[256], st[128:256], st[257]


def _conv(B: Builder, seed_bits: Sequence[int], k: int, s: int, iv: int) -> Tuple[List[int], List[int]]:
    w = k + s
    if 2 * w > STATE_BITS:
        raise ValueError("need 2(k+s) <= 320")
    st = _prg_state(B, seed_bits, iv)
    return st[0:w], st[w:2 * w]


def conv(B: Builder, seed_bits: Sequence[int], k: int, s: int) -> Tuple[List[int], List[int]]:
    """Conv(s) = (c_val, c_tag), each k+s wires (little-endian)."""
    return _conv(B, seed_bits, k, s, IV_C)


def convV_L(B: Builder, seed_bits: Sequence[int], k: int, s: int) -> Tuple[List[int], List[int]]:
    """ConvV_L(s) = (value word, tag word) for the DCF left branch."""
    return _conv(B, seed_bits, k, s, IV_VL)


def convV_R(B: Builder, seed_bits: Sequence[int], k: int, s: int) -> Tuple[List[int], List[int]]:
    """ConvV_R(s) = (value word, tag word) for the DCF right branch."""
    return _conv(B, seed_bits, k, s, IV_VR)


# --------------------------------------------------------------------------- #
# Scalar reference (plaintext checking only)
# --------------------------------------------------------------------------- #

_MASK64 = (1 << 64) - 1


def _ror(x: int, r: int) -> int:
    return ((x >> r) | (x << (64 - r))) & _MASK64


def scalar_permute(state5: Sequence[int]) -> List[int]:
    """Minimal scalar p^12 (same round function as the Ascon reference code)."""
    x0, x1, x2, x3, x4 = (int(w) & _MASK64 for w in state5)
    for c in ROUND_CONSTANTS:
        x2 ^= c
        x0 ^= x4
        x4 ^= x3
        x2 ^= x1
        t0 = (x0 ^ _MASK64) & x1
        t1 = (x1 ^ _MASK64) & x2
        t2 = (x2 ^ _MASK64) & x3
        t3 = (x3 ^ _MASK64) & x4
        t4 = (x4 ^ _MASK64) & x0
        x0 ^= t1
        x1 ^= t2
        x2 ^= t3
        x3 ^= t4
        x4 ^= t0
        x1 ^= x0
        x0 ^= x4
        x3 ^= x2
        x2 ^= _MASK64
        x0 ^= _ror(x0, 19) ^ _ror(x0, 28)
        x1 ^= _ror(x1, 61) ^ _ror(x1, 39)
        x2 ^= _ror(x2, 1) ^ _ror(x2, 6)
        x3 ^= _ror(x3, 10) ^ _ror(x3, 17)
        x4 ^= _ror(x4, 7) ^ _ror(x4, 41)
    return [x0, x1, x2, x3, x4]


def _load_permute():
    try:
        from .ascon import permute as ext  # noqa: WPS433
    except Exception:  # pragma: no cover - module not yet present
        return scalar_permute, False
    return ext, True


permute, PERMUTE_IS_EXTERNAL = _load_permute()
if not PERMUTE_IS_EXTERNAL:  # pragma: no cover
    warnings.warn("couplet.ascon not importable; using the local scalar Ascon reference")


def words_to_int(words: Sequence[int]) -> int:
    """Flat 320-bit integer from five words (word i at bits 64i..64i+63)."""
    v = 0
    for i, w in enumerate(words):
        v |= (int(w) & _MASK64) << (64 * i)
    return v


def int_to_words(value: int, n: int = 5) -> List[int]:
    return [(value >> (64 * i)) & _MASK64 for i in range(n)]


def permute_int(state: int) -> int:
    """p^12 on a flat 320-bit integer state."""
    return words_to_int(permute(int_to_words(state)))


def prg_state_ref(seed: int, iv: int) -> int:
    """Y = F_IV(seed) = p(seed || IV) XOR (seed || IV) as a flat 320-bit integer."""
    if seed < 0 or seed >> SEED_BITS:
        raise ValueError("seed must be a 128-bit integer")
    inp = seed | (iv << SEED_BITS)
    return permute_int(inp) ^ inp


def G_ref(seed: int) -> Tuple[int, int, int, int]:
    """Scalar G: (s_L, t_L, s_R, t_R) with seeds as 128-bit ints."""
    st = prg_state_ref(seed, IV_G)
    m = (1 << 128) - 1
    return st & m, (st >> 256) & 1, (st >> 128) & m, (st >> 257) & 1


def _conv_ref(seed: int, k: int, s: int, iv: int) -> Tuple[int, int]:
    w = k + s
    st = prg_state_ref(seed, iv)
    m = (1 << w) - 1
    return st & m, (st >> w) & m


def conv_ref(seed: int, k: int, s: int) -> Tuple[int, int]:
    return _conv_ref(seed, k, s, IV_C)


def convV_L_ref(seed: int, k: int, s: int) -> Tuple[int, int]:
    return _conv_ref(seed, k, s, IV_VL)


def convV_R_ref(seed: int, k: int, s: int) -> Tuple[int, int]:
    return _conv_ref(seed, k, s, IV_VR)


def p12_circuit_stats() -> dict:
    """Stats of a standalone p^12 on 320 fresh input wires (3840 ANDs, depth 12)."""
    B = Builder()
    st = B.input(0, "state", STATE_BITS)
    B.output("out", ascon_p12(B, st))
    return B.build().stats()


def prg_call_stats(label: bytes = LABEL_G) -> dict:
    """Stats of one PRG call F_IV(s) on a secret seed with the IV folded
    (all 320 output bits used): 3584 ANDs, AND-depth 12."""
    B = Builder()
    sd = B.input(0, "seed", SEED_BITS)
    B.output("y", prg_F(B, sd, iv_int(label)))
    return B.build().stats()


if __name__ == "__main__":
    print("p12 on 320 fresh wires:", p12_circuit_stats())
    print("F_G on a secret seed:  ", prg_call_stats())
