"""GF(2^128) arithmetic for the Boolean MAC layer (DESIGN.md sections 2.2, 7.2, 7.4).

Field: GF(2)[x] / (x^128 + x^7 + x^2 + x + 1), polynomial basis.

Representations
  * scalar   Python int in [0, 2^128); bit j is the coefficient of x^j.
  * bytes    16 bytes, little-endian encoding of that int.
  * vector   numpy ``uint64`` array of shape (N, 2); row i = (low word, high
             word) of element i, so ``row[0] | row[1] << 64`` is the scalar and
             ``vec.astype('<u8').tobytes()`` is the concatenated byte encoding.

Operations
  * ``mul``, ``xtime``, ``square``, ``power``, ``inv``      scalar field operations;
  * ``mul_vec_scalar(v, c)``    every row of ``v`` times the scalar ``c`` (byte tables);
  * ``select_xor(sel, m)``      the 128 partial sums  part_j = XOR of the rows m_i
                                whose selector row sel_i has bit j set;
  * ``horner(parts)``           sum_j x^j * part_j;
  * ``dot(chi, m)``             sum_i chi_i * m_i = horner(select_xor(chi, m));
  * ``dot_bits(chi, bits)``     sum_i bits_i * chi_i (XOR of the selected rows);
  * ``pack_bits`` / ``unpack_bits`` and the row-wise ``bits_to_vec`` / ``vec_to_bits``;
  * ``cols_to_rows``            128 packed bit columns -> packed rows (bit-matrix transpose);
  * conversion helpers (int / bytes / vector) and XOR helpers.

Addition in the field is XOR, so vectors are added with ``^``.

None of this is constant time: Python integers and numpy boolean selection
have data-dependent running time.  Side channels are out of scope (DESIGN.md 1).
"""

from __future__ import annotations

import secrets
from typing import Iterable, Sequence

import numpy as np

MASK64 = (1 << 64) - 1
MASK128 = (1 << 128) - 1
ELEM_BYTES = 16
POLY_LOW = 0x87  # x^7 + x^2 + x + 1; the modulus is x^128 + POLY_LOW

_U64 = np.uint64
_LE64 = np.dtype("<u8")
_CHUNK = 8192  # rows per pass; keeps the temporaries of the kernels inside the L2 cache
_SMALL = 48  # below this many rows the pure-int paths are faster than the numpy kernels


# --------------------------------------------------------------------------- #
# Scalar field operations (Python ints)
# --------------------------------------------------------------------------- #

def _check_elem(a: int) -> int:
    a = int(a)
    if a < 0 or a > MASK128:
        raise ValueError("field element out of range [0, 2^128)")
    return a


def reduce(p: int) -> int:
    """Reduce a binary polynomial of degree < 256 modulo x^128 + x^7 + x^2 + x + 1."""
    if p < 0 or p >> 256:
        raise ValueError("polynomial out of range [0, 2^256)")
    hi = p >> 128
    p = (p & MASK128) ^ hi ^ (hi << 1) ^ (hi << 2) ^ (hi << 7)
    hi = p >> 128  # at most 7 bits are left above bit 127
    return (p & MASK128) ^ hi ^ (hi << 1) ^ (hi << 2) ^ (hi << 7)


def clmul(a: int, b: int) -> int:
    """Carry-less (polynomial) product of two elements, not reduced (degree <= 254)."""
    a = _check_elem(a)
    b = _check_elem(b)
    tab = [0] * 16  # tab[v] = a * v for the 4-bit polynomial v
    tab[1] = a
    for v in range(2, 16):
        tab[v] = (tab[v >> 1] << 1) ^ (a if v & 1 else 0)
    r = 0
    for shift in range(124, -4, -4):
        r = (r << 4) ^ tab[(b >> shift) & 15]
    return r


def mul(a: int, b: int) -> int:
    """Field product a * b."""
    return reduce(clmul(a, b))


def xtime(a: int) -> int:
    """a * x."""
    a = _check_elem(a) << 1
    if a >> 128:
        a = (a & MASK128) ^ POLY_LOW
    return a


def square(a: int) -> int:
    """a * a."""
    return mul(a, a)


def power(a: int, e: int) -> int:
    """a ** e for an integer exponent e >= 0 (square and multiply)."""
    if e < 0:
        raise ValueError("negative exponent")
    result, base = 1, _check_elem(a)
    while e:
        if e & 1:
            result = mul(result, base)
        base = mul(base, base)
        e >>= 1
    return result


def inv(a: int) -> int:
    """Multiplicative inverse a ** (2^128 - 2); raises ZeroDivisionError for 0."""
    if _check_elem(a) == 0:
        raise ZeroDivisionError("zero has no inverse in GF(2^128)")
    return power(a, (1 << 128) - 2)


def rand_elem() -> int:
    """Uniform field element from the OS CSPRNG."""
    return secrets.randbits(128)


# --------------------------------------------------------------------------- #
# Conversions: int <-> bytes <-> arrays
# --------------------------------------------------------------------------- #

def int_to_bytes(a: int) -> bytes:
    """16-byte little-endian encoding of a field element."""
    return _check_elem(a).to_bytes(ELEM_BYTES, "little")


def bytes_to_int(data: bytes) -> int:
    """Inverse of ``int_to_bytes`` (exactly 16 bytes)."""
    if len(data) != ELEM_BYTES:
        raise ValueError("a field element is 16 bytes")
    return int.from_bytes(data, "little")


def int_to_arr(a: int) -> np.ndarray:
    """Field element -> ``uint64`` array of shape (2,) = (low word, high word)."""
    a = _check_elem(a)
    return np.array([a & MASK64, a >> 64], dtype=np.uint64)


def arr_to_int(row) -> int:
    """``uint64`` array of shape (2,) -> field element."""
    row = np.asarray(row)
    if row.dtype != np.uint64 or row.shape != (2,):
        raise ValueError("expected a uint64 array of shape (2,)")
    return int(row[0]) | (int(row[1]) << 64)


def as_vec(v) -> np.ndarray:
    """Validate a vector of field elements: ``uint64`` array of shape (N, 2)."""
    v = np.asarray(v)
    if v.dtype != np.uint64 or v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("expected a uint64 array of shape (N, 2)")
    return v


def zeros(n: int) -> np.ndarray:
    """Vector of n zero elements."""
    return np.zeros((n, 2), dtype=np.uint64)


def ints_to_vec(values: Iterable[int]) -> np.ndarray:
    """Sequence of field elements -> vector of shape (N, 2)."""
    vals = [_check_elem(a) for a in values]
    out = np.empty((len(vals), 2), dtype=np.uint64)
    if vals:
        out[:, 0] = [a & MASK64 for a in vals]
        out[:, 1] = [a >> 64 for a in vals]
    return out


def vec_to_ints(v) -> list[int]:
    """Vector of shape (N, 2) -> list of field elements."""
    v = as_vec(v)
    return [lo | (hi << 64) for lo, hi in zip(v[:, 0].tolist(), v[:, 1].tolist())]


def vec_to_bytes(v) -> bytes:
    """Vector -> 16 N bytes (elements in order, each little-endian)."""
    return as_vec(v).astype(_LE64, copy=False).tobytes()


def bytes_to_vec(data: bytes) -> np.ndarray:
    """16 N bytes -> writable vector of shape (N, 2)."""
    if len(data) % ELEM_BYTES:
        raise ValueError("length must be a multiple of 16 bytes")
    return np.frombuffer(data, dtype=_LE64).astype(np.uint64).reshape(-1, 2)


def rand_vec(n: int) -> np.ndarray:
    """n uniform field elements from the OS CSPRNG."""
    return bytes_to_vec(secrets.token_bytes(ELEM_BYTES * n))


def _as_bits(bits, n: int | None = None) -> np.ndarray:
    """Validate a 0/1 vector of shape (N,) and return it as a boolean array."""
    b = np.asarray(bits)
    if b.ndim != 1 or (n is not None and b.shape[0] != n):
        raise ValueError("expected a bit vector of shape (N,)" if n is None
                         else f"expected a bit vector of length {n}")
    if b.dtype == np.bool_:
        return b
    if b.dtype.kind not in "ui":
        raise ValueError("bits must have an integer or boolean dtype")
    if b.size and (int(b.max()) > 1 or int(b.min()) < 0):
        raise ValueError("bits must be 0 or 1")
    return b.astype(np.bool_)


def pack_bits(bits128) -> int:
    """128 bits (b_0, ..., b_127) -> the element sum_j b_j x^j."""
    b = _as_bits(bits128, 128)
    return int.from_bytes(np.packbits(b, bitorder="little").tobytes(), "little")


def unpack_bits(a: int) -> np.ndarray:
    """Element -> its 128 coefficients as a ``uint8`` array (bit j at index j)."""
    raw = np.frombuffer(int_to_bytes(a), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")


def bits_to_vec(bits) -> np.ndarray:
    """0/1 array of shape (N, 128) -> vector (N, 2); row i packs ``bits[i]``."""
    b = np.asarray(bits)
    if b.ndim != 2 or b.shape[1] != 128:
        raise ValueError("expected a bit array of shape (N, 128)")
    flat = _as_bits(b.reshape(-1)).reshape(b.shape)
    packed = np.packbits(flat, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view(_LE64).astype(np.uint64).reshape(-1, 2)


def vec_to_bits(v) -> np.ndarray:
    """Vector (N, 2) -> ``uint8`` array of shape (N, 128) of coefficients."""
    v = as_vec(v)
    raw = np.ascontiguousarray(v.astype(_LE64, copy=False)).view(np.uint8).reshape(-1, ELEM_BYTES)
    return np.unpackbits(raw, axis=1, bitorder="little")


def cols_to_rows(cols) -> np.ndarray:
    """Transpose a bit matrix with 128 columns from packed columns to packed rows.

    ``cols`` is a ``uint64`` array of shape (128, W): column j of the matrix,
    64 W bits long, with bit i stored in bit (i mod 64) of ``cols[j, i // 64]``
    (the bit order of ``couplet.hashes.prg_bits``).  Returns a vector of
    shape (64 W, 2) whose row i has coefficient j equal to bit i of column j.
    """
    cols = np.asarray(cols)
    if cols.dtype != np.uint64 or cols.ndim != 2 or cols.shape[0] != 128:
        raise ValueError("expected a uint64 array of shape (128, W)")
    n_words = cols.shape[1]
    out = np.empty((64 * n_words, 2), dtype=np.uint64)
    step = 512  # words per pass = 32768 rows; the unpacked block is 4 MiB
    for a in range(0, n_words, step):
        blk = np.ascontiguousarray(cols[:, a:a + step].astype(_LE64, copy=False))
        bits = np.unpackbits(blk.view(np.uint8), axis=1, bitorder="little")  # (128, 64 w)
        rows = np.packbits(np.ascontiguousarray(bits.T), axis=1, bitorder="little")  # (64 w, 16)
        out[64 * a:64 * a + rows.shape[0]] = rows.view(_LE64)
    return out


# --------------------------------------------------------------------------- #
# XOR helpers
# --------------------------------------------------------------------------- #

def xor(a, b) -> np.ndarray:
    """Element-wise sum of two vectors of the same shape."""
    a = as_vec(a)
    b = as_vec(b)
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    return a ^ b


def xor_scalar(v, c: int) -> np.ndarray:
    """Add the scalar c to every row of v."""
    return as_vec(v) ^ int_to_arr(c)


def xor_reduce(v) -> int:
    """Sum of all rows of v."""
    v = as_vec(v)
    if v.shape[0] == 0:
        return 0
    return arr_to_int(np.bitwise_xor.reduce(v, axis=0))


def bit_mul(bits, c: int) -> np.ndarray:
    """Vector with row i = bits_i * c, for a 0/1 vector ``bits`` of shape (N,)."""
    b = _as_bits(bits)
    out = np.zeros((b.shape[0], 2), dtype=np.uint64)
    out[b] = int_to_arr(c)
    return out


def lsb(v) -> np.ndarray:
    """Least significant bit (coefficient of x^0) of every row, as ``uint8``."""
    return (as_vec(v)[:, 0] & _U64(1)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Vector kernels
# --------------------------------------------------------------------------- #

def _mul_tables(c: int) -> np.ndarray:
    """T[p, b] = c * (b * x^(8p)) for byte position p < 16 and byte value b < 256."""
    basis = []
    t = _check_elem(c)
    for _ in range(128):
        basis.append(t)
        t = xtime(t)
    base = ints_to_vec(basis)  # base[j] = c * x^j
    tab = np.zeros((16, 256, 2), dtype=np.uint64)
    for p in range(16):
        for bit in range(8):
            h = 1 << bit
            tab[p, h:2 * h] = tab[p, :h] ^ base[8 * p + bit]
    return tab


def mul_vec_scalar(v, c: int) -> np.ndarray:
    """Every row of v times the scalar c; returns a new vector of shape (N, 2).

    Multiplication by a constant is GF(2)-linear, so the product is the XOR of
    sixteen table entries, one per byte of the row.
    """
    v = as_vec(v)
    c = _check_elem(c)
    n = v.shape[0]
    if n <= _SMALL:
        return ints_to_vec(mul(a, c) for a in vec_to_ints(v))
    tab = _mul_tables(c)
    out = np.empty((n, 2), dtype=np.uint64)
    for a in range(0, n, _CHUNK):
        chunk = np.ascontiguousarray(v[a:a + _CHUNK].astype(_LE64, copy=False))
        idx = chunk.view(np.uint8).reshape(-1, ELEM_BYTES)
        acc = tab[0][idx[:, 0]]
        for p in range(1, 16):
            acc ^= tab[p][idx[:, p]]
        out[a:a + _CHUNK] = acc
    return out


def select_xor(sel, m) -> np.ndarray:
    """Partial sums by selector bit: returns P of shape (128, 2) with

        P[j] = XOR of the rows m_i over all i such that bit j of sel_i is 1.

    ``sel`` and ``m`` are vectors of the same shape (N, 2).  This is the
    bit-serial kernel of ``dot``; with ``sel`` the rows of a bit matrix and
    ``m`` a coefficient vector it yields all 128 column sums in one pass
    (the per-column values of the OT-extension consistency check).
    """
    sel = as_vec(sel)
    m = as_vec(m)
    if sel.shape != m.shape:
        raise ValueError("sel and m must have the same shape")
    out = np.zeros((128, 2), dtype=np.uint64)
    one = _U64(1)
    reduce_xor = np.bitwise_xor.reduce
    for a in range(0, sel.shape[0], _CHUNK):
        m_lo = np.ascontiguousarray(m[a:a + _CHUNK, 0])
        m_hi = np.ascontiguousarray(m[a:a + _CHUNK, 1])
        for word in range(2):
            s = np.ascontiguousarray(sel[a:a + _CHUNK, word])
            for j in range(64):
                mask = np.negative((s >> _U64(j)) & one)  # all-ones where the bit is set
                row = out[64 * word + j]
                row[0] ^= reduce_xor(m_lo & mask)
                row[1] ^= reduce_xor(m_hi & mask)
    return out


def horner(parts) -> int:
    """sum_j x^j * parts_j for a vector of at most 128 partial sums (Horner pass)."""
    vals = vec_to_ints(parts)
    if len(vals) > 128:
        raise ValueError("at most 128 partial sums")
    acc = 0
    for part in reversed(vals):
        acc = xtime(acc) ^ part
    return acc


def dot(chi, m) -> int:
    """Inner product sum_i chi_i * m_i over GF(2^128).

    Bit-serial: for each of the 128 coefficient positions j of chi, the rows of
    m selected by that bit are XORed together (``select_xor``), and the 128
    partial sums are combined as sum_j x^j * part_j (``horner``).
    """
    chi = as_vec(chi)
    m = as_vec(m)
    if chi.shape != m.shape:
        raise ValueError("chi and m must have the same shape")
    if chi.shape[0] <= _SMALL:
        acc = 0
        for a, b in zip(vec_to_ints(chi), vec_to_ints(m)):
            acc ^= clmul(a, b)
        return reduce(acc)
    return horner(select_xor(chi, m))


def dot_bits(chi, bits) -> int:
    """sum_i bits_i * chi_i: the XOR of the rows of chi selected by a 0/1 vector."""
    chi = as_vec(chi)
    b = _as_bits(bits, chi.shape[0])
    if not b.any():
        return 0
    lo = np.bitwise_xor.reduce(np.ascontiguousarray(chi[:, 0])[b])
    hi = np.bitwise_xor.reduce(np.ascontiguousarray(chi[:, 1])[b])
    return int(lo) | (int(hi) << 64)


def dot_ints(chi: Sequence[int], m: Sequence[int]) -> int:
    """Inner product of two equally long sequences of scalars (reference path)."""
    if len(chi) != len(m):
        raise ValueError("sequences must have the same length")
    acc = 0
    for a, b in zip(chi, m):
        acc ^= clmul(a, b)
    return reduce(acc)
