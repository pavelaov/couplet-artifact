"""Tests for the backend primitives: gf128, hashes, ed25519.

gf128     published/derivable vectors (x * x^127 = 0x87, (2^128-1)^2), a slow
          bitwise reference multiplication, the field laws, and every vector
          kernel (mul_vec_scalar, select_xor, dot, dot_bits) against that
          reference on sizes that straddle the small-N and chunk boundaries.
hashes    H_ccr and ctr_prg against the scalar Ascon permutation with the
          documented state layout, regression vectors, framing of sha256,
          commitments, and coin tossing over an in-process channel including
          six cheating strategies that must end in ProtocolAbort.
composed  the identities the backend relies on, built from these modules:
          the COT-extension consistency check (DESIGN.md 7.2), BoolCheck
          (2.2) and the leaky-AND transfer (7.4).
ed25519   RFC 8032 section 7.1 public keys (tests 1-3), the encoding of the
          base point, group laws, encoding edge cases, small-order points,
          hash_to_group membership in the prime-order subgroup.

Run as a script from artifact/python:   python tests/test_primitives.py
The script prints the number of checks and exits non-zero on any failure.
The ``test_*`` functions are also collectable by pytest.
"""

from __future__ import annotations

import hashlib
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import ed25519 as ed  # noqa: E402
from couplet import gf128 as gf  # noqa: E402
from couplet import hashes as hs  # noqa: E402
from couplet.ascon import permute  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import run_two_party  # noqa: E402

CHECKS = 0
FAILURES: list[str] = []
MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1


def check(cond, what: str) -> None:
    """Count one check; record (and print) it if it fails."""
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(what)
        print("FAIL:", what)


def raises(exc_type, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception:  # a different exception type is a failure of the check
        return False
    return False


def _finish(start_failures: int) -> None:
    """Make the enclosing test function fail under pytest if it recorded a failure."""
    assert len(FAILURES) == start_failures, FAILURES[start_failures:]


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #

def ref_mul(a: int, b: int) -> int:
    """Slow bitwise shift-and-add multiplication modulo x^128 + x^7 + x^2 + x + 1."""
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 128:
            a ^= (1 << 128) | 0x87
    return r


def ref_dot(chi: list[int], m: list[int]) -> int:
    acc = 0
    for a, b in zip(chi, m):
        acc ^= ref_mul(a, b)
    return acc


def rand_ints(rng: random.Random, n: int) -> list[int]:
    return [rng.getrandbits(128) for _ in range(n)]


# --------------------------------------------------------------------------- #
# gf128
# --------------------------------------------------------------------------- #

def test_gf128_scalar() -> None:
    start = len(FAILURES)
    rng = random.Random(0x6F128)

    check(gf.mul(0x02, 1 << 127) == 0x87, "x * x^127 = 0x87")
    check(gf.mul(MASK128, MASK128) == 0x5555555555555555555555555555402F, "(2^128-1)^2")
    check(ref_mul(0x02, 1 << 127) == 0x87, "reference: x * x^127")
    check(ref_mul(MASK128, MASK128) == 0x5555555555555555555555555555402F, "reference: (2^128-1)^2")

    edge = [0, 1, 2, 3, 0x87, 1 << 63, 1 << 64, 1 << 127, MASK64, MASK128, MASK128 - 1]
    for a in edge:
        for b in edge:
            check(gf.mul(a, b) == ref_mul(a, b), f"mul edge {a:#x} {b:#x}")
    for _ in range(2000):
        a, b = rng.getrandbits(128), rng.getrandbits(128)
        check(gf.mul(a, b) == ref_mul(a, b), "mul random vs reference")
    for _ in range(200):
        a, b = rng.getrandbits(128), rng.getrandbits(128)
        check(gf.reduce(gf.clmul(a, b)) == ref_mul(a, b), "reduce(clmul)")
        check(gf.clmul(a, b) == gf.clmul(b, a), "clmul commutes")

    for _ in range(200):
        a, b, c = rand_ints(rng, 3)
        check(gf.mul(a, b) == gf.mul(b, a), "commutativity")
        check(gf.mul(gf.mul(a, b), c) == gf.mul(a, gf.mul(b, c)), "associativity")
        check(gf.mul(a, b ^ c) == gf.mul(a, b) ^ gf.mul(a, c), "distributivity")
        check(gf.mul(a, 1) == a and gf.mul(a, 0) == 0, "one and zero")
        check(gf.square(a) == gf.mul(a, a), "square")
        check(gf.square(a ^ b) == gf.square(a) ^ gf.square(b), "Frobenius is additive")
        check(gf.xtime(a) == gf.mul(a, 2), "xtime")
    for _ in range(40):
        a = rng.getrandbits(128) | 1
        check(gf.mul(a, gf.inv(a)) == 1, "a * inv(a) = 1")
    for _ in range(10):
        a = rng.getrandbits(128)
        check(gf.power(a, 1 << 128) == a, "a^(2^128) = a")
        check(gf.power(a, 5) == gf.mul(gf.mul(gf.square(gf.square(a)), a), 1), "power(a, 5)")
    check(gf.power(0, 0) == 1 and gf.power(7, 0) == 1, "power(., 0) = 1")
    check(raises(ZeroDivisionError, gf.inv, 0), "inv(0) raises")
    check(raises(ValueError, gf.mul, 1 << 128, 1), "mul rejects 129-bit operand")
    check(raises(ValueError, gf.mul, -1, 1), "mul rejects negative operand")
    check(raises(ValueError, gf.reduce, 1 << 256), "reduce rejects degree >= 256")
    check(0 <= gf.rand_elem() <= MASK128, "rand_elem range")
    _finish(start)


def test_gf128_conversions() -> None:
    start = len(FAILURES)
    rng = random.Random(0xC0)
    vals = rand_ints(rng, 33) + [0, 1, MASK128, 1 << 64, MASK64]
    for a in vals:
        check(gf.bytes_to_int(gf.int_to_bytes(a)) == a, "int <-> bytes")
        check(gf.int_to_bytes(a) == a.to_bytes(16, "little"), "bytes are little-endian")
        arr = gf.int_to_arr(a)
        check(arr.dtype == np.uint64 and arr.shape == (2,), "int_to_arr shape")
        check(int(arr[0]) == a & MASK64 and int(arr[1]) == a >> 64, "low word first")
        check(gf.arr_to_int(arr) == a, "int <-> arr")
        bits = gf.unpack_bits(a)
        check(bits.shape == (128,) and bits.dtype == np.uint8, "unpack_bits shape")
        check(all(int(bits[j]) == (a >> j) & 1 for j in range(128)), "unpack_bits order")
        check(gf.pack_bits(bits) == a, "pack_bits inverts unpack_bits")
    for j in (0, 1, 63, 64, 127):
        unit = np.zeros(128, dtype=np.uint8)
        unit[j] = 1
        check(gf.pack_bits(unit) == 1 << j, f"pack_bits(e_{j}) = x^{j}")
    check(gf.pack_bits(np.ones(128, dtype=bool)) == MASK128, "pack_bits accepts bool")
    check(raises(ValueError, gf.pack_bits, np.zeros(127, dtype=np.uint8)), "pack_bits length")
    check(raises(ValueError, gf.pack_bits, np.full(128, 2, dtype=np.uint8)), "pack_bits values")

    vec = gf.ints_to_vec(vals)
    check(vec.shape == (len(vals), 2) and vec.dtype == np.uint64, "ints_to_vec shape")
    check(gf.vec_to_ints(vec) == vals, "ints <-> vec")
    raw = gf.vec_to_bytes(vec)
    check(raw == b"".join(a.to_bytes(16, "little") for a in vals), "vec_to_bytes layout")
    back = gf.bytes_to_vec(raw)
    check(np.array_equal(back, vec) and back.flags.writeable, "bytes_to_vec round trip")
    check(raises(ValueError, gf.bytes_to_vec, b"\x00" * 17), "bytes_to_vec length")
    bitmat = gf.vec_to_bits(vec)
    check(bitmat.shape == (len(vals), 128), "vec_to_bits shape")
    check(all(gf.pack_bits(bitmat[i]) == vals[i] for i in range(len(vals))), "vec_to_bits rows")
    check(np.array_equal(gf.bits_to_vec(bitmat), vec), "bits_to_vec inverts vec_to_bits")
    for n_words in (0, 1, 3, 513):  # 513 words crosses the internal block size
        cols = np.frombuffer(rng.randbytes(128 * 8 * n_words), dtype="<u8").reshape(128, n_words)
        cols = cols.astype(np.uint64)
        rows = gf.cols_to_rows(cols)
        check(rows.shape == (64 * n_words, 2) and rows.dtype == np.uint64,
              f"cols_to_rows shape, {n_words} words")
        col_bits = np.unpackbits(cols.astype("<u8").view(np.uint8), axis=1, bitorder="little")
        check(np.array_equal(gf.vec_to_bits(rows), col_bits.T),
              f"cols_to_rows is the bit-matrix transpose, {n_words} words")
        if n_words:
            i, j = rng.randrange(64 * n_words), rng.randrange(128)
            check((gf.arr_to_int(rows[i]) >> j) & 1 == (int(cols[j, i // 64]) >> (i % 64)) & 1,
                  "cols_to_rows: row i bit j = column j bit i")
    check(raises(ValueError, gf.cols_to_rows, np.zeros((127, 2), dtype=np.uint64)),
          "cols_to_rows needs 128 columns")
    check(gf.ints_to_vec([]).shape == (0, 2) and gf.zeros(3).shape == (3, 2), "empty and zeros")
    check(gf.rand_vec(5).shape == (5, 2), "rand_vec shape")

    check(raises(ValueError, gf.as_vec, np.zeros((3, 2), dtype=np.int64)), "as_vec dtype")
    check(raises(ValueError, gf.as_vec, np.zeros((3, 3), dtype=np.uint64)), "as_vec shape")
    check(raises(ValueError, gf.as_vec, np.zeros(4, dtype=np.uint64)), "as_vec ndim")
    check(raises(ValueError, gf.ints_to_vec, [1 << 128]), "ints_to_vec range")

    a_vec, b_vec = gf.ints_to_vec(vals[:20]), gf.ints_to_vec(vals[10:30])
    check(gf.vec_to_ints(gf.xor(a_vec, b_vec)) == [x ^ y for x, y in zip(vals[:20], vals[10:30])],
          "xor")
    check(raises(ValueError, gf.xor, a_vec, b_vec[:5]), "xor shape mismatch")
    c = vals[31]
    check(gf.vec_to_ints(gf.xor_scalar(a_vec, c)) == [x ^ c for x in vals[:20]], "xor_scalar")
    acc = 0
    for x in vals[:20]:
        acc ^= x
    check(gf.xor_reduce(a_vec) == acc and gf.xor_reduce(gf.zeros(0)) == 0, "xor_reduce")
    bits = np.array([rng.getrandbits(1) for _ in range(20)], dtype=np.uint8)
    check(gf.vec_to_ints(gf.bit_mul(bits, c)) == [c if b else 0 for b in bits.tolist()], "bit_mul")
    check(gf.lsb(a_vec).tolist() == [x & 1 for x in vals[:20]], "lsb")
    _finish(start)


def test_gf128_vectors() -> None:
    start = len(FAILURES)
    rng = random.Random(0xD07)
    sizes = (0, 1, 2, 7, gf._SMALL, gf._SMALL + 1, 300, gf._CHUNK - 1, gf._CHUNK, gf._CHUNK + 5)
    for n in sizes:
        chi_i, m_i = rand_ints(rng, n), rand_ints(rng, n)
        chi, m = gf.ints_to_vec(chi_i), gf.ints_to_vec(m_i)
        expected = ref_dot(chi_i, m_i)
        check(gf.dot(chi, m) == expected, f"dot vs reference, N={n}")
        check(gf.dot(m, chi) == expected, f"dot is symmetric, N={n}")
        check(gf.dot_ints(chi_i, m_i) == expected, f"dot_ints, N={n}")
        check(gf.horner(gf.select_xor(chi, m)) == expected, f"horner(select_xor), N={n}")

        for c in (rng.getrandbits(128), 0, 1, 2, MASK128):
            got = gf.mul_vec_scalar(m, c)
            check(got.shape == (n, 2) and got.dtype == np.uint64, f"mul_vec_scalar shape, N={n}")
            check(gf.vec_to_ints(got) == [ref_mul(a, c) for a in m_i],
                  f"mul_vec_scalar vs reference, N={n}, c={c:#x}")

        bits = np.array([rng.getrandbits(1) for _ in range(n)], dtype=np.uint8)
        acc = 0
        for a, b in zip(chi_i, bits.tolist()):
            if b:
                acc ^= a
        check(gf.dot_bits(chi, bits) == acc, f"dot_bits vs reference, N={n}")
        check(gf.dot_bits(chi, bits.astype(bool)) == acc, f"dot_bits accepts bool, N={n}")
        check(gf.dot(chi, gf.bit_mul(bits, 1)) == acc, f"dot with 0/1 elements, N={n}")
        check(gf.dot_bits(chi, np.zeros(n, dtype=np.uint8)) == 0, f"dot_bits of zero, N={n}")

    # select_xor against the definition (every selector bit, every row).
    n = 150
    sel_i, m_i = rand_ints(rng, n), rand_ints(rng, n)
    parts = gf.vec_to_ints(gf.select_xor(gf.ints_to_vec(sel_i), gf.ints_to_vec(m_i)))
    check(len(parts) == 128, "select_xor returns 128 partial sums")
    for j in range(128):
        acc = 0
        for s, v in zip(sel_i, m_i):
            if (s >> j) & 1:
                acc ^= v
        check(parts[j] == acc, f"select_xor part {j}")
    # Column sums of a bit matrix: part_j = sum_i T[i][j] * chi_i = dot_bits(chi, column j).
    t_rows, chi = gf.ints_to_vec(sel_i), gf.ints_to_vec(m_i)
    cols = gf.vec_to_bits(t_rows)
    col_sums = gf.select_xor(t_rows, chi)
    check(all(gf.arr_to_int(col_sums[j]) == gf.dot_bits(chi, cols[:, j]) for j in range(128)),
          "select_xor gives the per-column dot_bits")

    # Bilinearity of dot on a size that uses the numpy kernel.
    n = 500
    chi = gf.rand_vec(n)
    m1, m2 = gf.rand_vec(n), gf.rand_vec(n)
    c = gf.rand_elem()
    check(gf.dot(chi, m1 ^ m2) == gf.dot(chi, m1) ^ gf.dot(chi, m2), "dot is additive")
    check(gf.dot(chi, gf.mul_vec_scalar(m1, c)) == gf.mul(gf.dot(chi, m1), c), "dot is linear")
    check(gf.dot(gf.mul_vec_scalar(chi, c), m1) == gf.mul(gf.dot(chi, m1), c),
          "dot is linear in the first argument")

    for j in (0, 1, 64, 127):
        parts = gf.zeros(128)
        parts[j, 0] = 1
        check(gf.horner(parts) == gf.power(2, j), f"horner(e_{j}) = x^{j}")
    check(gf.horner(gf.zeros(0)) == 0, "horner of nothing")
    check(raises(ValueError, gf.horner, gf.zeros(129)), "horner length")
    check(raises(ValueError, gf.dot, gf.zeros(3), gf.zeros(4)), "dot shape mismatch")
    check(raises(ValueError, gf.dot_bits, gf.zeros(3), np.zeros(4, dtype=np.uint8)),
          "dot_bits length mismatch")
    check(raises(ValueError, gf.dot_bits, gf.zeros(3), np.array([0, 2, 1], dtype=np.uint8)),
          "dot_bits rejects non-bits")
    check(raises(ValueError, gf.select_xor, gf.zeros(3), gf.zeros(4)), "select_xor shape mismatch")
    check(raises(ValueError, gf.mul_vec_scalar, gf.zeros(3), 1 << 128), "mul_vec_scalar range")
    _finish(start)


# --------------------------------------------------------------------------- #
# hashes: Ascon-based
# --------------------------------------------------------------------------- #

# Regression vectors produced by this implementation after it was validated
# against the scalar permutation; they pin the state layout for the C++ port.
KAT_X = 0x0F0E0D0C0B0A09080706050403020100
KAT_H_X_1_2 = 0x6DA1D8E5727BA7B47D46E228A7058A58          # H_ccr(KAT_X, tweak 1, dom 2)
KAT_H_0_0_0 = 0x9B9BFB8513B560F778EA7AE5CFEBB108          # H_ccr(0, 0, 0)
KAT_H_0_0_2 = 0x155F8AD25DE350245C5C41BF99BFF3A8          # H_ccr(0, 0, 2)
KAT_PRG = [0x06A0383FEE1E2BA2, 0x8837B4BB06FAE342, 0x556312FD72A8D4EA, 0x25D4CF495D02EFD9,
           0xA2A54CB2406CAB33, 0x5E94674EFCAC4CAB, 0x16FDC9AAACD11E27]  # seed 00..0f, dom 7
KAT_PRG_BITS = "0100010111010100011110000111011111111100000111000000010101100000010000"
KAT_SHA_AB_C = "43ee655579de01ca739b3f95c1c2d3f46d353b2c0df818064ea594506cdb2617"
KAT_EXPAND = ("1ab1137394263ac36190fb6a811db4d023d3875fbd6fb950338c7fa37957ccbb"
              "a77cd108ad9186b4")
KAT_H2G_TAG_MSG = "9269b2fb58d6a540a227c29674f8826ab16af3ccdf15851f7993d78e0aa06bb0"
KAT_H2G_EMPTY = "554d65722bea908cfa6820fff04da728aaa65962491fc4ffe674f732f1e93335"


def _ref_h(x: int, tweak: int, dom: int) -> int:
    y = permute([x & MASK64, x >> 64, tweak, dom, 0])
    return y[0] | (y[1] << 64)


def test_h_ccr() -> None:
    start = len(FAILURES)
    rng = random.Random(0xCC12)
    check(hs.H_ccr_scalar(KAT_X, 1, 2) == KAT_H_X_1_2, "H_ccr regression vector 1")
    check(hs.H_ccr_scalar(0, 0, 0) == KAT_H_0_0_0, "H_ccr regression vector 2")
    check(hs.H_ccr_scalar(0, 0, 2) == KAT_H_0_0_2, "H_ccr regression vector 3")
    full = permute([0, 0, 0, 0, 0])
    check(KAT_H_0_0_0 == full[0] | (full[1] << 64), "H_ccr(0,0,0) = first 128 bits of p(0)")

    for n in (0, 1, 3, 257, hs._CHUNK + 3):
        xs = rand_ints(rng, n)
        tweaks = [rng.getrandbits(64) for _ in range(n)]
        dom = rng.getrandbits(64) | (1 << 63)
        x = gf.ints_to_vec(xs)
        got = hs.H_ccr(x, np.array(tweaks, dtype=np.uint64), dom)
        check(got.shape == (n, 2) and got.dtype == np.uint64, f"H_ccr shape, N={n}")
        check(gf.vec_to_ints(got) == [_ref_h(a, t, dom) for a, t in zip(xs, tweaks)],
              f"H_ccr (tweak array) vs scalar permutation, N={n}")
        tw = rng.getrandbits(64)
        got = hs.H_ccr(x, tw, dom)
        check(gf.vec_to_ints(got) == [_ref_h(a, tw, dom) for a in xs],
              f"H_ccr (tweak int) vs scalar permutation, N={n}")
        check(all(hs.H_ccr_scalar(a, tw, dom) == b
                  for a, b in zip(xs[:16], gf.vec_to_ints(got)[:16])), f"H_ccr_scalar, N={n}")
        check(np.array_equal(hs.H_ccr(x, np.uint64(tw), dom), got), f"numpy scalar tweak, N={n}")

    x = gf.ints_to_vec([KAT_X] * 4)
    base = hs.H_ccr(x, 5, 9)
    check(len({tuple(r) for r in base.tolist()}) == 1, "equal inputs hash equally")
    check(not np.array_equal(hs.H_ccr(x, 6, 9), base), "tweak changes the output")
    check(not np.array_equal(hs.H_ccr(x, 5, 10), base), "dom changes the output")
    check(not np.array_equal(hs.H_ccr(gf.xor_scalar(x, 1), 5, 9), base), "x changes the output")
    before = x.copy()
    hs.H_ccr(x, 5, 9)
    check(np.array_equal(x, before), "H_ccr does not modify its input")

    for dom in sorted(hs.RESERVED_DOMS):
        check(raises(ValueError, hs.H_ccr, x, 0, dom), f"reserved dom {dom:#x} rejected")
    check(hs.RESERVED_DOMS == {0x47, 0x43, 0x4C56, 0x5256}, "reserved dom set")
    check(raises(ValueError, hs.H_ccr, x, 0, 1 << 64), "dom range")
    check(raises(ValueError, hs.H_ccr, x, -1, 3), "tweak range")
    check(raises(ValueError, hs.H_ccr, x, np.zeros(3, dtype=np.uint64), 3), "tweak shape")
    check(raises(ValueError, hs.H_ccr, x, np.zeros(4, dtype=np.int64), 3), "tweak dtype")
    check(raises(ValueError, hs.H_ccr, np.zeros((4, 3), dtype=np.uint64), 0, 3), "x shape")
    check(raises(ValueError, hs.H_ccr_scalar, 1 << 128, 0, 3), "scalar x range")
    _finish(start)


def test_ctr_prg() -> None:
    start = len(FAILURES)
    rng = random.Random(0x9126)
    seed0 = bytes(range(16))
    check(hs.ctr_prg(seed0, 7, 7).tolist() == KAT_PRG, "ctr_prg regression vector")
    check("".join(map(str, hs.prg_bits(seed0, 7, 70).tolist())) == KAT_PRG_BITS,
          "prg_bits regression vector")

    for _ in range(4):
        seed = rng.randbytes(16)
        dom = rng.getrandbits(64) | (1 << 62)
        lo, hi = int.from_bytes(seed[:8], "little"), int.from_bytes(seed[8:], "little")
        words = hs.ctr_prg(seed, dom, 100)
        check(words.shape == (100,) and words.dtype == np.uint64, "ctr_prg shape")
        ref = []
        for i in range(34):
            ref.extend(permute([lo, hi, i, dom, 0])[:3])
        check(words.tolist() == ref[:100], "ctr_prg blocks = (x0,x1,x2) of p(seed, i, dom, 0)")
        for n in (0, 1, 2, 3, 4, 99):
            check(hs.ctr_prg(seed, dom, n).tolist() == ref[:n], f"ctr_prg prefix, n={n}")
        check(hs.ctr_prg(seed, dom, 30, start=11).tolist() == ref[33:63], "ctr_prg start offset")

        bits = hs.prg_bits(seed, dom, 1000)
        check(bits.shape == (1000,) and bits.dtype == np.uint8, "prg_bits shape")
        check(all(int(bits[j]) == (ref[j // 64] >> (j % 64)) & 1 for j in range(1000)),
              "prg_bits order (LSB of word 0 first)")
        check(hs.prg_bits(seed, dom, 0).shape == (0,), "prg_bits of length 0")
        elems = hs.prg_gf128(seed, dom, 20)
        check(elems.shape == (20, 2), "prg_gf128 shape")
        check(gf.vec_to_ints(elems) == [ref[2 * i] | (ref[2 * i + 1] << 64) for i in range(20)],
              "prg_gf128 element i = words (2i, 2i+1)")
        check(not np.array_equal(hs.ctr_prg(seed, dom ^ 1, 100), words), "dom separates streams")

    # A long stream crosses the internal chunk boundary; compare against a
    # second expansion made in two pieces and against sampled scalar blocks.
    seed = rng.randbytes(16)
    n_blocks = hs._CHUNK + 100
    words = hs.ctr_prg(seed, 77, 3 * n_blocks)
    tail = hs.ctr_prg(seed, 77, 300, start=hs._CHUNK)
    check(np.array_equal(words[3 * hs._CHUNK:], tail), "long stream = start-offset stream")
    lo, hi = int.from_bytes(seed[:8], "little"), int.from_bytes(seed[8:], "little")
    for i in (0, 1, hs._CHUNK - 1, hs._CHUNK, n_blocks - 1):
        check(words[3 * i:3 * i + 3].tolist() == permute([lo, hi, i, 77, 0])[:3],
              f"long stream block {i}")
    bits = hs.prg_bits(seed, 77, 1 << 16)
    check(abs(float(bits.mean()) - 0.5) < 0.01, "prg_bits is balanced (2^16 bits)")

    seeds = [rng.randbytes(16) for _ in range(9)]
    many = hs.ctr_prg_many(gf.bytes_to_vec(b"".join(seeds)), 5, 50)
    check(many.shape == (9, 50), "ctr_prg_many shape")
    check(all(np.array_equal(many[i], hs.ctr_prg(seeds[i], 5, 50)) for i in range(9)),
          "ctr_prg_many rows = ctr_prg")
    many = hs.ctr_prg_many(gf.bytes_to_vec(b"".join(seeds)), 5, 3 * 2000 + 1, start=3)
    check(all(np.array_equal(many[i], hs.ctr_prg(seeds[i], 5, 6001, start=3)) for i in range(9)),
          "ctr_prg_many (chunked, start offset)")
    check(hs.ctr_prg_many(gf.zeros(0), 5, 10).shape == (0, 10), "ctr_prg_many with no seeds")

    check(raises(ValueError, hs.ctr_prg, b"\x00" * 15, 5, 3), "seed length")
    check(raises(ValueError, hs.ctr_prg, seed0, 0x47, 3), "reserved dom")
    check(raises(ValueError, hs.ctr_prg, seed0, 5, -1), "negative length")
    check(raises(ValueError, hs.ctr_prg, seed0, 5, 6, start=(1 << 64) - 1), "counter overflow")
    check(hs.ctr_prg(seed0, 5, 3, start=(1 << 64) - 1).shape == (3,), "last counter value")
    _finish(start)


# --------------------------------------------------------------------------- #
# The primitives composed as the backend uses them (DESIGN.md 2.2, 7.2, 7.4)
# --------------------------------------------------------------------------- #

def _packed_columns(rows) -> np.ndarray:
    """pack(column j) for the 128 columns of a 128-row block, as a (128, 2) vector."""
    return gf.bits_to_vec(np.ascontiguousarray(gf.vec_to_bits(rows).T))


def test_backend_identities() -> None:
    start = len(FAILURES)
    rng = random.Random(0x1D)

    # Correlated OT extension with the per-column check of DESIGN.md 7.2.
    ell, pad = 896, 128
    n_words = (ell + pad) // 64
    delta = rng.getrandbits(128) | 1
    delta_bits = gf.unpack_bits(delta)
    seeds0 = gf.bytes_to_vec(rng.randbytes(16 * 128))
    seeds1 = gf.bytes_to_vec(rng.randbytes(16 * 128))
    t0_cols = hs.ctr_prg_many(seeds0, 0x101, n_words)
    t1_cols = hs.ctr_prg_many(seeds1, 0x101, n_words)
    u_words = np.frombuffer(rng.randbytes(8 * n_words), dtype="<u8").astype(np.uint64)
    u_bits = np.unpackbits(u_words.astype("<u8").view(np.uint8), bitorder="little")
    chi = hs.prg_gf128(rng.randbytes(16), 0x102, ell)

    def sender_rows(u_cols):
        """q columns = PRG(k_j^{Delta_j}) XOR Delta_j U_j, transposed to rows."""
        sel = delta_bits.astype(bool)[:, None]
        return gf.cols_to_rows(np.where(sel, t1_cols ^ u_cols, t0_cols))

    def sender_accepts(q_rows, u_tilde, v_tilde) -> bool:
        lhs = gf.select_xor(q_rows[:ell], chi) ^ _packed_columns(q_rows[ell:])
        return np.array_equal(lhs, v_tilde ^ gf.bit_mul(delta_bits, u_tilde))

    u_cols = t0_cols ^ t1_cols ^ u_words  # U_j = t_j^0 XOR t_j^1 XOR u, one row per column j
    t_rows = gf.cols_to_rows(t0_cols)
    q_rows = sender_rows(u_cols)
    check(np.array_equal(q_rows, t_rows ^ gf.bit_mul(u_bits, delta)), "COT: q_i = t_i XOR u_i Delta")
    u_tilde = gf.dot_bits(chi, u_bits[:ell]) ^ gf.pack_bits(u_bits[ell:])
    v_tilde = gf.select_xor(t_rows[:ell], chi) ^ _packed_columns(t_rows[ell:])
    check(sender_accepts(q_rows, u_tilde, v_tilde), "COT: honest receiver passes the check")
    j_one = int(np.flatnonzero(delta_bits == 1)[5])
    j_zero = int(np.flatnonzero(delta_bits == 0)[5])
    bad = u_cols.copy()
    bad[j_one, 3] ^= np.uint64(1 << 17)  # inconsistent u in a column with Delta_j = 1
    check(not sender_accepts(sender_rows(bad), u_tilde, v_tilde),
          "COT: inconsistent column is caught when Delta_j = 1")
    bad = u_cols.copy()
    bad[j_zero, 3] ^= np.uint64(1 << 17)  # the sender never uses U_j when Delta_j = 0
    check(sender_accepts(sender_rows(bad), u_tilde, v_tilde),
          "COT: the same deviation is invisible when Delta_j = 0 (selective failure on Delta)")
    check(not sender_accepts(q_rows, u_tilde ^ 1, v_tilde), "COT: wrong u~ is rejected")

    # BoolCheck (DESIGN.md 2.2): V_b = sum chi_j m_{j,b} XOR z Delta_b, accept iff V_0 = V_1.
    n = 300
    d0, d1 = rng.getrandbits(128) | 1, rng.getrandbits(128) & ~1
    y = np.array([rng.getrandbits(1) for _ in range(n)], dtype=np.uint8)
    m0 = gf.bytes_to_vec(rng.randbytes(16 * n))
    m1 = m0 ^ gf.bit_mul(y, d0 ^ d1)
    chi = hs.prg_gf128(rng.randbytes(16), 0x103, n)

    def v_share(m, opened, key_share) -> int:
        return gf.dot(chi, m) ^ gf.mul(gf.dot_bits(chi, opened), key_share)

    check(v_share(m0, y, d0) == v_share(m1, y, d1), "BoolCheck: honest openings pass")
    forged = y.copy()
    forged[77] ^= 1
    check(v_share(m0, forged, d0) != v_share(m1, forged, d1), "BoolCheck: a flipped bit fails")
    check(v_share(m0, forged, d0) ^ v_share(m1, forged, d1)
          == gf.mul(gf.arr_to_int(chi[77]), d0 ^ d1), "BoolCheck: the gap is chi_j * Delta")

    # Leaky-AND transfer (DESIGN.md 7.4): R_0 = H(M[x_0]) XOR x_0 G_1 = S_1 XOR x_0 m_1.
    n = 200
    dom = 0x104
    tweak = np.arange(n, dtype=np.uint64)
    x0 = np.array([rng.getrandbits(1) for _ in range(n)], dtype=np.uint8)
    key = gf.bytes_to_vec(rng.randbytes(16 * n))  # K[x_0], held by P_1
    mac = key ^ gf.bit_mul(x0, d1)  # M[x_0], held by P_0
    m1_share = gf.bytes_to_vec(rng.randbytes(16 * n))
    s1 = hs.H_ccr(key, tweak, dom)
    g1 = s1 ^ hs.H_ccr(gf.xor_scalar(key, d1), tweak, dom) ^ m1_share
    r0 = hs.H_ccr(mac, tweak, dom) ^ np.where(x0[:, None] == 1, g1, np.uint64(0))
    check(np.array_equal(r0, s1 ^ np.where(x0[:, None] == 1, m1_share, np.uint64(0))),
          "leaky AND transfer: R_0 = S_1 XOR x_0 m_1")
    _finish(start)


# --------------------------------------------------------------------------- #
# hashes: SHA-256 based
# --------------------------------------------------------------------------- #

def _framed(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(len(p).to_bytes(8, "little") + p)
    return h.digest()


def test_sha256_and_commit() -> None:
    start = len(FAILURES)
    check(hs.sha256(b"ab", b"c").hex() == KAT_SHA_AB_C, "sha256 regression vector")
    check(hs.sha256() == hashlib.sha256(b"").digest(), "sha256 of no parts")
    check(hs.sha256(b"ab", b"c") == _framed(b"ab", b"c"), "framing: 8-byte LE length per part")
    check(hs.sha256(b"ab", b"c") != hs.sha256(b"a", b"bc"), "framing is injective (split)")
    check(hs.sha256(b"abc") != hs.sha256(b"abc", b""), "framing is injective (empty part)")
    check(hs.sha256(bytearray(b"ab"), memoryview(b"c")) == _framed(b"ab", b"c"), "bytes-like parts")
    check(raises(TypeError, hs.sha256, "ab"), "sha256 rejects str")
    big = bytes(1 << 20)
    check(hs.sha256(big) == _framed(big), "sha256 of 1 MiB")

    out = hs.sha256_expand(40, b"x", b"yz")
    check(out.hex() == KAT_EXPAND, "sha256_expand regression vector")
    check(out == (_framed(b"x", b"yz", (0).to_bytes(8, "little"))
                  + _framed(b"x", b"yz", (1).to_bytes(8, "little")))[:40], "sha256_expand blocks")
    check(hs.sha256_expand(100, b"x", b"yz")[:40] == out, "sha256_expand prefix property")
    check(hs.sha256_expand(0, b"x") == b"" and len(hs.sha256_expand(65, b"x")) == 65,
          "sha256_expand lengths")
    check(hs.kdf(b"sid", b"\x01") == _framed(b"Couplet-kdf", b"sid", b"\x01")[:16], "kdf layout")
    check(hs.kdf(b"sid", nbytes=32) == _framed(b"Couplet-kdf", b"sid"), "kdf with 32 bytes")
    check(raises(ValueError, hs.kdf, b"sid", nbytes=33), "kdf length limit")

    sid, msg = b"session-1", b"the message"
    com, opening = hs.commit(sid, msg)
    check(len(com) == 32 and len(opening) == 32, "commitment and opening are 32 bytes")
    check(com == _framed(b"Couplet-com", sid, b"", b"", opening, msg), "commitment layout")
    check(hs.verify(sid, com, msg, opening), "correct opening verifies")
    check(not hs.verify(sid, com, msg + b"!", opening), "wrong message")
    check(not hs.verify(sid, com, msg, bytes(32)), "wrong opening")
    check(not hs.verify(sid + b"x", com, msg, opening), "wrong sid")
    check(not hs.verify(sid, com, msg, opening, label=b"l"), "wrong label")
    check(not hs.verify(sid, com, msg, opening, party=0), "wrong party binding")
    check(not hs.verify(sid, com[:31], msg, opening), "short commitment")
    check(not hs.verify(sid, com, msg, opening[:31]), "short opening")
    com2, opening2 = hs.commit(sid, msg)
    check(com2 != com and opening2 != opening, "commitments are randomised")
    for party in (0, 1):
        com, opening = hs.commit(sid, msg, party=party, label=b"L")
        check(com == _framed(b"Couplet-com", sid, b"L", bytes([party]), opening, msg),
              f"party-bound commitment layout, party {party}")
        check(hs.verify(sid, com, msg, opening, party=party, label=b"L"), "party-bound verifies")
        check(not hs.verify(sid, com, msg, opening, party=1 - party, label=b"L"),
              "commitment of one party does not verify for the other")
        check(not hs.verify(sid, com, msg, opening, label=b"L"), "party tag is not optional")
    check(raises(ValueError, hs.commit, sid, msg, party=2), "party must be 0 or 1")
    _finish(start)


def test_coin_toss() -> None:
    start = len(FAILURES)
    sid, label = b"sid-coin", b"chi"

    for nbytes in (0, 1, 16, 32, 100):
        r0, r1, st0, st1 = run_two_party(lambda c: hs.coin_toss(c, sid, label, nbytes),
                                         lambda c: hs.coin_toss(c, sid, label, nbytes), timeout=20)
        check(r0 == r1 and len(r0) == nbytes, f"coin_toss agrees, {nbytes} bytes")
        check(st0["rounds"] == 2 and st1["rounds"] == 2, "coin_toss takes two rounds")
        check(st0["bytes_sent"] == 32 + 64 and st1["bytes_sent"] == 32 + 64,
              "coin_toss sends 32 + 64 bytes per party")
    a = run_two_party(lambda c: hs.coin_toss(c, sid, label, 32),
                      lambda c: hs.coin_toss(c, sid, label, 32), timeout=20)[0]
    b = run_two_party(lambda c: hs.coin_toss(c, sid, label, 32),
                      lambda c: hs.coin_toss(c, sid, label, 32), timeout=20)[0]
    check(a != b, "two coin tosses differ")

    m0, m1, st0, st1 = run_two_party(lambda c: hs.committed_exchange(c, sid, b"v", b"from-0"),
                                     lambda c: hs.committed_exchange(c, sid, b"v", b"from-one"),
                                     timeout=20)
    check(m0 == b"from-one" and m1 == b"from-0", "committed_exchange delivers both messages")
    check(st0["rounds"] == 2 and st1["rounds"] == 2, "committed_exchange takes two rounds")

    def honest(chan):
        return hs.coin_toss(chan, sid, label, 32)

    def expect_abort(adversary, text: str, name: str) -> None:
        for honest_party in (0, 1):
            fns = (honest, adversary) if honest_party == 0 else (adversary, honest)
            try:
                run_two_party(*fns, timeout=20)
            except ProtocolAbort as exc:
                check(text in str(exc), f"{name}: honest party {honest_party} aborts ({exc})")
            else:
                check(False, f"{name}: honest party {honest_party} did not abort")

    coin_label = b"coin|" + label

    def adv_bad_opening(chan):
        mine = bytes(32)
        com, omega = hs.commit(sid, mine, party=chan.party, label=coin_label)
        chan.exchange(com)
        chan.exchange(omega + b"\x01" + mine[1:])  # opens a different string
        return None

    def adv_mirror(chan):
        chan.send(chan.recv())  # replay the honest commitment ...
        chan.send(chan.recv())  # ... and the honest opening
        return None

    def adv_wrong_party(chan):
        mine = bytes(32)
        com, omega = hs.commit(sid, mine, party=1 - chan.party, label=coin_label)
        chan.exchange(com)
        chan.exchange(omega + mine)
        return None

    def adv_short_commitment(chan):
        chan.exchange(b"\x00" * 31)
        chan.exchange(b"\x00" * 64)
        return None

    def adv_short_share(chan):
        mine = bytes(31)  # a valid commitment to a string of the wrong length
        com, omega = hs.commit(sid, mine, party=chan.party, label=coin_label)
        chan.exchange(com)
        chan.exchange(omega + mine)
        return None

    def adv_short_opening(chan):
        com, _ = hs.commit(sid, bytes(32), party=chan.party, label=coin_label)
        chan.exchange(com)
        chan.exchange(b"\x00" * 5)
        return None

    expect_abort(adv_bad_opening, "opening failed", "bad opening")
    expect_abort(adv_mirror, "mirrored", "mirrored commitment")
    expect_abort(adv_wrong_party, "opening failed", "commitment under the wrong party index")
    expect_abort(adv_short_commitment, "malformed commitment", "short commitment")
    expect_abort(adv_short_share, "wrong length", "share of the wrong length")
    expect_abort(adv_short_opening, "malformed opening", "short opening")
    _finish(start)


# --------------------------------------------------------------------------- #
# ed25519
# --------------------------------------------------------------------------- #

# RFC 8032, section 7.1, TEST 1-3: (secret key, public key).
RFC8032_KEYS = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"),
]
BASE_POINT_ENC = "5866666666666666666666666666666666666666666666666666666666666666"


def _clamped_scalar(secret_hex: str) -> int:
    h = bytearray(hashlib.sha512(bytes.fromhex(secret_hex)).digest()[:32])
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    return int.from_bytes(h, "little")


def test_ed25519_vectors() -> None:
    start = len(FAILURES)
    check(ed.P == 2**255 - 19, "field prime")
    check(ed.Q == 2**252 + 27742317777372353535851937790883648493, "group order")
    check(ed.D * 121666 % ed.P == ed.P - 121665, "curve constant d = -121665/121666")
    check(ed.is_on_curve(ed.B) and ed.is_on_curve(ed.IDENTITY), "B and identity are on the curve")
    check(ed.compress(ed.B).hex() == BASE_POINT_ENC, "encoding of the base point")
    check(ed.equal(ed.decompress(bytes.fromhex(BASE_POINT_ENC)), ed.B), "decoding of the base point")
    check(ed.compress(ed.IDENTITY) == (1).to_bytes(32, "little"), "encoding of the identity")
    for secret, public in RFC8032_KEYS:
        a = _clamped_scalar(secret)
        check(ed.compress(ed.base_mult(a)).hex() == public, "RFC 8032 public key (base_mult)")
        check(ed.compress(ed.scalar_mult(a, ed.B)).hex() == public,
              "RFC 8032 public key (scalar_mult)")
    check(ed.is_identity(ed.scalar_mult(ed.Q, ed.B)), "q * B = identity")
    check(ed.is_identity(ed.base_mult(ed.Q)), "base_mult(q) = identity")
    check(ed.equal(ed.scalar_mult(ed.Q - 1, ed.B), ed.neg(ed.B)), "(q - 1) * B = -B")
    check(not ed.is_identity(ed.B) and ed.is_identity(ed.IDENTITY), "is_identity")
    check(ed.in_subgroup(ed.B) and ed.in_subgroup(ed.IDENTITY), "in_subgroup")
    _finish(start)


def test_ed25519_group_laws() -> None:
    start = len(FAILURES)
    rng = random.Random(0xED25519)
    for _ in range(20):
        a, b, c = (rng.randrange(1, ed.Q) for _ in range(3))
        pa, pb, pc = ed.base_mult(a), ed.base_mult(b), ed.base_mult(c)
        check(ed.is_on_curve(pa), "base_mult result is on the curve")
        check(ed.equal(pa, ed.scalar_mult(a, ed.B)), "base_mult = scalar_mult on B")
        check(ed.equal(ed.add(pa, pb), ed.add(pb, pa)), "commutativity")
        check(ed.equal(ed.add(ed.add(pa, pb), pc), ed.add(pa, ed.add(pb, pc))), "associativity")
        check(ed.equal(ed.add(pa, ed.IDENTITY), pa), "identity is neutral")
        check(ed.is_identity(ed.add(pa, ed.neg(pa))), "P + (-P) = identity")
        check(ed.is_identity(ed.sub(pa, pa)), "P - P = identity")
        check(ed.equal(ed.double(pa), ed.add(pa, pa)), "double = add with itself")
        check(ed.equal(ed.add(pa, pb), ed.base_mult((a + b) % ed.Q)), "aB + bB = (a+b)B")
        check(ed.equal(ed.sub(pa, pb), ed.base_mult((a - b) % ed.Q)), "aB - bB = (a-b)B")
        check(ed.equal(ed.scalar_mult(a, pb), ed.base_mult(a * b % ed.Q)), "a(bB) = (ab)B")
        check(ed.equal(ed.scalar_mult(a, pb), ed.scalar_mult(b, pa)), "Diffie-Hellman agreement")
        check(ed.equal(ed.scalar_mult(-a, pb), ed.neg(ed.scalar_mult(a, pb))), "negative scalar")
        check(ed.equal(ed.clear_cofactor(pa), ed.scalar_mult(8, pa)), "clear_cofactor = 8P")
        enc = ed.compress(pa)
        check(len(enc) == 32 and ed.equal(ed.decompress(enc), pa), "compress/decompress round trip")
        check(ed.compress(ed.decompress(enc)) == enc, "encoding is canonical")
        check(ed.compress(ed.neg(pa))[:31] == enc[:31]
              and ed.compress(ed.neg(pa))[31] == enc[31] ^ 0x80, "negation flips the sign bit")
        check(ed.equal(ed.decompress_subgroup(enc), pa), "decompress_subgroup accepts")
    pa = ed.base_mult(12345)
    check(ed.is_identity(ed.scalar_mult(0, pa)), "0 * P = identity")
    check(ed.equal(ed.scalar_mult(1, pa), pa), "1 * P = P")
    for k in (2, 15, 16, 17, 255, 256, 2**64 + 1):
        acc = ed.IDENTITY
        addend = pa
        kk = k
        while kk:  # plain double-and-add as an independent reference
            if kk & 1:
                acc = ed.add(acc, addend)
            addend = ed.double(addend)
            kk >>= 1
        check(ed.equal(ed.scalar_mult(k, pa), acc), f"scalar_mult({k}) vs double-and-add")
    for _ in range(50):
        k = ed.random_scalar()
        check(1 <= k < ed.Q, "random_scalar range")
    _finish(start)


def test_ed25519_encoding_edges() -> None:
    start = len(FAILURES)
    p = ed.P
    check(raises(ValueError, ed.decompress, b"\x00" * 31), "decompress length (31)")
    check(raises(ValueError, ed.decompress, b"\x00" * 33), "decompress length (33)")
    check(raises(ValueError, ed.decompress, p.to_bytes(32, "little")), "non-canonical y = p")
    check(raises(ValueError, ed.decompress, (p + 1).to_bytes(32, "little")), "non-canonical y = p+1")
    check(raises(ValueError, ed.decompress, (1 | (1 << 255)).to_bytes(32, "little")),
          "x = 0 with sign bit set")
    # Smallest y >= 2 for which (y^2 - 1)/(d y^2 + 1) is a non-residue (Euler criterion).
    y = 2
    while pow((y * y - 1) * pow(ed.D * y * y + 1, p - 2, p) % p, (p - 1) // 2, p) != p - 1:
        y += 1
    check(raises(ValueError, ed.decompress, y.to_bytes(32, "little")), f"y = {y} is not on the curve")
    check(raises(ValueError, ed.decompress, (y | (1 << 255)).to_bytes(32, "little")),
          "same y with the sign bit")
    # A y with a residue decodes to both signs.
    y = 2
    while pow((y * y - 1) * pow(ed.D * y * y + 1, p - 2, p) % p, (p - 1) // 2, p) != 1:
        y += 1
    pos = ed.decompress(y.to_bytes(32, "little"))
    negp = ed.decompress((y | (1 << 255)).to_bytes(32, "little"))
    check(ed.is_on_curve(pos) and ed.is_on_curve(negp), "decoded points are on the curve")
    check(pos[0] % 2 == 0 and negp[0] % 2 == 1 and ed.equal(pos, ed.neg(negp)), "sign bit selects x")

    order2 = ed.decompress((p - 1).to_bytes(32, "little"))  # (0, -1)
    order4 = ed.decompress((0).to_bytes(32, "little"))  # (sqrt(-1), 0)
    check(ed.is_on_curve(order2) and ed.is_on_curve(order4), "small-order points are on the curve")
    check(not ed.is_identity(order2) and ed.is_identity(ed.double(order2)), "(0, -1) has order 2")
    check(not ed.is_identity(ed.double(order4)) and ed.is_identity(ed.double(ed.double(order4))),
          "(sqrt(-1), 0) has order 4")
    check(not ed.in_subgroup(order2) and not ed.in_subgroup(order4), "small order: not in subgroup")
    check(ed.is_identity(ed.clear_cofactor(order2)) and ed.is_identity(ed.clear_cofactor(order4)),
          "cofactor clearing kills small-order points")
    check(raises(ValueError, ed.decompress_subgroup, (p - 1).to_bytes(32, "little")),
          "decompress_subgroup rejects order 2")
    check(raises(ValueError, ed.decompress_subgroup, (0).to_bytes(32, "little")),
          "decompress_subgroup rejects order 4")
    mixed = ed.add(ed.B, order4)  # order 4 q
    check(ed.is_on_curve(mixed) and not ed.in_subgroup(mixed), "B + T4 is not in the subgroup")
    check(raises(ValueError, ed.decompress_subgroup, ed.compress(mixed)),
          "decompress_subgroup rejects a mixed-order point")
    check(ed.equal(ed.clear_cofactor(mixed), ed.base_mult(8)), "8 (B + T4) = 8 B")
    check(ed.in_subgroup(ed.clear_cofactor(mixed)), "cofactor clearing lands in the subgroup")
    ident = (1).to_bytes(32, "little")
    check(ed.is_identity(ed.decompress_subgroup(ident)), "identity is accepted by default")
    check(raises(ValueError, ed.decompress_subgroup, ident, allow_identity=False),
          "identity rejected on request")
    _finish(start)


def test_hash_to_group() -> None:
    start = len(FAILURES)
    check(ed.compress(ed.hash_to_group(b"tag", b"msg")).hex() == KAT_H2G_TAG_MSG,
          "hash_to_group regression vector 1")
    check(ed.compress(ed.hash_to_group(b"", b"")).hex() == KAT_H2G_EMPTY,
          "hash_to_group regression vector 2")
    seen = set()
    for i in range(40):
        msg = i.to_bytes(4, "little")
        pt = ed.hash_to_group(b"mr-test", msg)
        check(ed.is_on_curve(pt), "hash_to_group output is on the curve")
        check(not ed.is_identity(pt), "hash_to_group output is not the identity")
        check(ed.in_subgroup(pt), "hash_to_group output has order q")
        check(ed.equal(pt, ed.hash_to_group(b"mr-test", msg)), "hash_to_group is deterministic")
        seen.add(ed.compress(pt))
    check(len(seen) == 40, "hash_to_group outputs are distinct")
    check(not ed.equal(ed.hash_to_group(b"ab", b"c"), ed.hash_to_group(b"a", b"bc")),
          "tag and message are framed separately")
    check(not ed.equal(ed.hash_to_group(b"t1", b"m"), ed.hash_to_group(b"t2", b"m")),
          "tag separates")

    # Reproduce the try-and-increment by hand for one input.
    ctr = 0
    while True:
        digest = hs.sha256(b"Couplet-h2g", b"tag", b"msg", ctr.to_bytes(4, "little"))
        ctr += 1
        try:
            cand = ed.decompress(digest)
        except ValueError:
            continue
        cand = ed.scalar_mult(8, cand)
        if not ed.is_identity(cand):
            break
    check(ed.equal(cand, ed.hash_to_group(b"tag", b"msg")), "hash_to_group construction")

    # One Masny-Rindal instance, both choice bits: the two parties derive the
    # same group element for the chosen index (API integration check).
    sid = b"sid"
    for choice in (0, 1):
        def h(j: int, point) -> tuple:
            return ed.hash_to_group(b"mr" + sid + bytes([j]), ed.compress(point))
        b_sc, t_sc, a_sc = ed.random_scalar(), ed.random_scalar(), ed.random_scalar()
        r = [None, None]
        r[1 - choice] = ed.base_mult(t_sc)
        r[choice] = ed.sub(ed.base_mult(b_sc), h(choice, r[1 - choice]))
        big_a = ed.base_mult(a_sc)
        sender = [ed.scalar_mult(a_sc, ed.add(r[j], h(j, r[1 - j]))) for j in (0, 1)]
        receiver = ed.scalar_mult(b_sc, big_a)
        check(ed.equal(sender[choice], receiver), f"Masny-Rindal agreement, choice {choice}")
        check(not ed.equal(sender[1 - choice], receiver), f"other key differs, choice {choice}")
    _finish(start)


TESTS = [
    test_gf128_scalar, test_gf128_conversions, test_gf128_vectors,
    test_h_ccr, test_ctr_prg, test_backend_identities, test_sha256_and_commit, test_coin_toss,
    test_ed25519_vectors, test_ed25519_group_laws, test_ed25519_encoding_edges,
    test_hash_to_group,
]


def main() -> int:
    for test in TESTS:
        before = CHECKS
        try:
            test()
        except AssertionError:
            pass  # the failed checks are already recorded in FAILURES
        except Exception as exc:  # an unexpected error is a failure as well
            FAILURES.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print("ERROR:", FAILURES[-1])
        print(f"{test.__name__}: {CHECKS - before} checks")
    print(f"test_primitives: {CHECKS} checks, {len(FAILURES)} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
