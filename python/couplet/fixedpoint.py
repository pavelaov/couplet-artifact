"""Two's-complement fixed point of the Couplet layer (DESIGN_M3.md section 3).

A wire format is ``(ell, f)``: a real ``v`` is represented by the integer
``round(v 2^f)`` (round half to even) as an ``ell``-bit two's-complement code
``c = round(v 2^f) mod 2^ell``, the top half of the ring being negative.
Signed integers travel through the stack in two forms:

  * *sign-exact* in ``Z_{2^w}``: ``embed(v, w) = v mod 2^w`` for an integer
    ``v`` with ``-2^(w-1) <= v < 2^(w-1)``; ``unembed`` inverts it;
  * *n-bit code*: ``v mod 2^n``, meaningful modulo ``2^n`` only.

``truncate(p, f, n) = (p >> f) mod 2^n`` is the floor truncation of a
pre-activation ``p`` with ``2f`` fraction bits to the ``n``-bit code with
``f`` fraction bits (Python's ``>>`` floors negative integers).  Every
function is a pure map on Python integers; nothing here is vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Format", "decode", "embed", "encode", "fits_signed", "floor_shift", "round_half_even",
           "to_signed", "to_unsigned", "truncate", "unembed"]


def round_half_even(x: float) -> int:
    """Nearest integer to ``x``, ties to even (Python's ``round``)."""
    return int(round(float(x)))


def encode(x: float, f: int) -> int:
    """Real -> integer with ``f`` fraction bits (round half to even); not reduced."""
    return round_half_even(float(x) * float(1 << int(f)))


def decode(v: int, f: int) -> float:
    """Signed integer with ``f`` fraction bits -> real."""
    return int(v) / float(1 << int(f))


def to_signed(u: int, bits: int) -> int:
    """Code in ``[0, 2^bits)`` -> signed representative in ``[-2^(bits-1), 2^(bits-1))``."""
    u = int(u) % (1 << int(bits))
    return u - (1 << int(bits)) if u >> (int(bits) - 1) else u


def to_unsigned(v: int, bits: int) -> int:
    """Signed integer -> code ``v mod 2^bits``."""
    return int(v) % (1 << int(bits))


def fits_signed(v: int, bits: int) -> bool:
    """True iff ``-2^(bits-1) <= v < 2^(bits-1)``: the embedding in ``Z_{2^bits}`` is sign-exact."""
    half = 1 << (int(bits) - 1)
    return -half <= int(v) < half


def embed(v: int, w: int) -> int:
    """Sign-exact embedding ``v mod 2^w``; raises ``ValueError`` if ``v`` does not fit ``w - 1`` bits."""
    if not fits_signed(v, w):
        raise ValueError(f"{v} does not fit a sign-exact value of {w} bits")
    return int(v) % (1 << int(w))


unembed = to_signed


def floor_shift(p: int, f: int) -> int:
    """``floor(p / 2^f)`` on signed integers (arithmetic shift)."""
    return int(p) >> int(f)


def truncate(p: int, f: int, n: int) -> int:
    """The ``n``-bit code of ``floor(p / 2^f)``: ``(p >> f) mod 2^n``."""
    return (int(p) >> int(f)) % (1 << int(n))


@dataclass(frozen=True)
class Format:
    """Wire format ``(n, f)``: ``n``-bit codes with ``f`` fraction bits (``n`` is the plane domain)."""

    n: int
    f: int

    def __post_init__(self) -> None:
        if not (isinstance(self.n, int) and isinstance(self.f, int)) or self.n < 1 or self.f < 0:
            raise ValueError("need integers n >= 1 and f >= 0")
        if self.f >= self.n:
            raise ValueError("need f < n")

    @property
    def N(self) -> int:
        return 1 << self.n

    @property
    def ulp(self) -> float:
        return 1.0 / float(1 << self.f)

    @property
    def max_real(self) -> float:
        return decode((1 << (self.n - 1)) - 1, self.f)

    @property
    def min_real(self) -> float:
        return decode(-(1 << (self.n - 1)), self.f)

    def encode_code(self, x: float) -> int:
        """Real -> ``n``-bit code (wraps if out of range, like the ring would)."""
        return to_unsigned(encode(x, self.f), self.n)

    def decode_code(self, c: int) -> float:
        return decode(to_signed(c, self.n), self.f)

    def in_range(self, v: int) -> bool:
        """True iff the signed integer ``v`` (``f`` fraction bits) is an ``n``-bit code without wrap."""
        return fits_signed(v, self.n)
