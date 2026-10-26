"""Activation tables of the Couplet layer (DESIGN_M3.md section 3): exact look-up
tables for the table query of a plane, and the plain reference evaluator.

For a wire format ``(n, f)`` the table of a unary real function ``g`` has
``N = 2^n`` entries indexed by the ``n``-bit two's-complement code ``z``:

    T[z] = round_half_even( g(signed(z) / 2^f) * 2^f )

the correctly rounded value of the real function at every domain point, as a
signed integer with ``f`` fraction bits.  Entries are NOT reduced to ``n``
bits: the plane query embeds them sign-exactly in ``Z_{2^w}`` and the next
linear layer consumes them there, so the output range of ``g`` is not limited
to the wire format.  Functions:

    relu     max(x, 0)                     exact in integers
    sign     sgn(x) in {-1, 0, 1}          times 2^f, exact
    sigmoid  1 / (1 + exp(-x))
    tanh
    gelu     x Phi(x) = x (1 + erf(x / sqrt 2)) / 2      (the exact erf definition)
    recip    1 / x for x > 0, defined as 0 for x <= 0     round(2^(2f) / signed(z)), exact rational rounding
    rsqrt    1 / sqrt(x) for x > 0, 0 for x <= 0          round(sqrt(2^(3f) / signed(z))), exact with isqrt

Correct rounding of the transcendental functions: the value is computed in
float64; if ``g(x) 2^f`` lies within a conservative error margin of a
rounding boundary (a half-integer), it is recomputed with the ``decimal``
module at 60 significant digits (``exp`` of ``decimal``; ``erf`` by its Taylor
series, which converges absolutely for the arguments that matter, with
``erf(x) = sign(x)`` for ``|x| >= 9``, where ``erfc`` is below 10^-36).
``table_hash(name, n, f)`` is the SHA-256 of the framed tuple
``("Couplet-table", name, le32(n), le32(f), entries as int64 little-endian)``
and is what the manifest binds.  ``evaluate_plain`` is the reference used by
the fixed-point mirror.

Table file (for the native implementation, which has no correctly rounded
transcendental functions): ``table_<name>_n<n>_f<f>.bin`` (``table_file_name``),
written by ``write_table`` and by ``python -m couplet.tables NAME N F OUT_DIR``:
    "CPLTTAB1" | le32 len(name) | name (UTF-8) | le32 n | le32 f |
    2^n entries T[0], ..., T[2^n - 1] as int64 little-endian (signed)
The table hash is computed from (name, n, f, entries) as above, so a reader
verifies a file against the ``table_hash`` of a model or of a manifest.
"""

from __future__ import annotations

import math
import os
import sys
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from typing import Dict, List, Tuple

from . import hashes as hs
from .fixedpoint import to_signed

__all__ = ["NAMES", "activation_table", "evaluate_plain", "read_table", "reference", "table_bytes", "table_entry",
           "table_file_name", "table_hash", "write_table"]

NAMES = ("relu", "sigmoid", "tanh", "gelu", "sign", "recip", "rsqrt")
_DEC_DIGITS = 60
_ERF_CUTOFF = 9
_TABLE_CACHE: Dict[Tuple[str, int, int], List[int]] = {}
_FILE_MAGIC = b"CPLTTAB1"
_SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# Real reference functions (float64)
# --------------------------------------------------------------------------- #

def reference(name: str, x: float) -> float:
    """The real function ``g`` of a table name at ``x`` (float64)."""
    x = float(x)
    if name == "relu":
        return x if x > 0.0 else 0.0
    if name == "sign":
        return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)
    if name == "sigmoid":
        return 0.5 * (1.0 + math.tanh(0.5 * x))
    if name == "tanh":
        return math.tanh(x)
    if name == "gelu":
        return x * 0.5 * (1.0 + math.erf(x / _SQRT2))
    if name == "recip":
        return 1.0 / x if x > 0.0 else 0.0
    if name == "rsqrt":
        return 1.0 / math.sqrt(x) if x > 0.0 else 0.0
    raise ValueError(f"unknown table name {name!r}; known: {NAMES}")


# --------------------------------------------------------------------------- #
# High-precision fall-back (decimal)
# --------------------------------------------------------------------------- #

def _dec_tanh(x: Decimal) -> Decimal:
    if x > 60:
        return Decimal(1)
    if x < -60:
        return Decimal(-1)
    e = (2 * x).exp()
    return (e - 1) / (e + 1)


def _dec_sigmoid(x: Decimal) -> Decimal:
    if x > 200:
        return Decimal(1)
    if x < -200:
        return Decimal(0)
    return 1 / (1 + (-x).exp())


def _dec_erf(x: Decimal) -> Decimal:
    if x >= _ERF_CUTOFF:
        return Decimal(1)
    if x <= -_ERF_CUTOFF:
        return Decimal(-1)
    # erf(x) = 2/sqrt(pi) * sum_k (-1)^k x^(2k+1) / (k! (2k+1))
    term = x                      # x^(2k+1) / k!
    total = x
    x2 = x * x
    k = 0
    eps = Decimal(10) ** (-(_DEC_DIGITS + 5))
    while True:
        k += 1
        term = -term * x2 / k
        contribution = term / (2 * k + 1)
        total += contribution
        if abs(contribution) < eps:
            break
    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
    return 2 * total / pi.sqrt()


def _dec_gelu(x: Decimal) -> Decimal:
    return x / 2 * (1 + _dec_erf(x / Decimal(2).sqrt()))


_DEC = {"tanh": _dec_tanh, "sigmoid": _dec_sigmoid, "gelu": _dec_gelu}


def _round_decimal(name: str, v: int, f: int) -> int:
    """Correctly rounded ``g(v / 2^f) * 2^f`` with 60 significant digits."""
    with localcontext() as ctx:
        ctx.prec = _DEC_DIGITS
        x = Decimal(int(v)) / Decimal(1 << int(f))
        r = _DEC[name](x) * Decimal(1 << int(f))
        return int(r.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


# --------------------------------------------------------------------------- #
# Correctly rounded entries
# --------------------------------------------------------------------------- #

def _round_ratio(num: int, den: int) -> int:
    """``round(num / den)`` for integers, ``den > 0``, ties to even."""
    q, r = divmod(int(num), int(den))
    if 2 * r > den or (2 * r == den and q & 1):
        q += 1
    return q


def _round_sqrt_ratio(num: int, den: int) -> int:
    """``round(sqrt(num / den))`` for integers ``num >= 0``, ``den > 0``, ties to even."""
    t = math.isqrt((4 * int(num)) // int(den))          # floor(2 sqrt(num/den))
    if t & 1 and t * t * int(den) == 4 * int(num):      # an exact tie: sqrt = (t)/2 with t odd
        q = (t - 1) // 2
        return q + (q & 1)
    return (t + 1) // 2


def table_entry(name: str, z: int, n: int, f: int) -> int:
    """``T[z]`` of the table ``name`` at format ``(n, f)``: the correctly rounded value."""
    if name not in NAMES:
        raise ValueError(f"unknown table name {name!r}; known: {NAMES}")
    v = to_signed(z, n)
    scale = 1 << f
    if name == "relu":
        return v if v > 0 else 0
    if name == "sign":
        return scale if v > 0 else (-scale if v < 0 else 0)
    if name == "recip":
        return _round_ratio(scale * scale, v) if v > 0 else 0
    if name == "rsqrt":
        return _round_sqrt_ratio(scale * scale * scale, v) if v > 0 else 0
    x = v / scale
    r = reference(name, x) * scale
    margin = (abs(x) + 1.0) * scale * 1e-12
    nearest_boundary = math.floor(r) + 0.5
    if abs(r - nearest_boundary) < margin:
        return _round_decimal(name, v, f)
    return int(round(r))


def activation_table(name: str, n: int, f: int) -> List[int]:
    """The full table ``T[0 .. 2^n - 1]`` of ``name`` at ``(n, f)`` (cached; signed integers)."""
    n, f = int(n), int(f)
    if n < 1 or f < 0:
        raise ValueError("need n >= 1 and f >= 0")
    key = (name, n, f)
    tab = _TABLE_CACHE.get(key)
    if tab is None:
        tab = [table_entry(name, z, n, f) for z in range(1 << n)]
        _TABLE_CACHE[key] = tab
    return tab


def evaluate_plain(name: str, n: int, f: int, z: int) -> int:
    """Plain reference: ``T[z]`` for the ``n``-bit code ``z``."""
    if not 0 <= int(z) < (1 << int(n)):
        raise ValueError("z must be an n-bit code")
    return activation_table(name, n, f)[int(z)]


def table_hash(name: str, n: int, f: int) -> bytes:
    """SHA-256 of ``("Couplet-table", name, le32(n), le32(f), entries as int64 LE)`` (32 bytes)."""
    entries = activation_table(name, n, f)
    for t in entries:
        if not -(1 << 63) <= t < (1 << 63):
            raise ValueError("table entries must fit a signed 64-bit integer")
    blob = b"".join(int(t).to_bytes(8, "little", signed=True) for t in entries)
    return hs.sha256(b"Couplet-table", name.encode("utf-8"), int(n).to_bytes(4, "little"),
                     int(f).to_bytes(4, "little"), blob)


# --------------------------------------------------------------------------- #
# Table files
# --------------------------------------------------------------------------- #

def table_file_name(name: str, n: int, f: int) -> str:
    """``table_<name>_n<n>_f<f>.bin``."""
    return f"table_{name}_n{int(n)}_f{int(f)}.bin"


def table_bytes(name: str, n: int, f: int) -> bytes:
    """The table file of ``name`` at ``(n, f)`` (layout in the module docstring)."""
    entries = activation_table(name, n, f)
    raw = name.encode("utf-8")
    return b"".join([_FILE_MAGIC, len(raw).to_bytes(4, "little"), raw, int(n).to_bytes(4, "little"),
                     int(f).to_bytes(4, "little")] + [int(t).to_bytes(8, "little", signed=True) for t in entries])


def write_table(out_dir: str, name: str, n: int, f: int) -> str:
    """Write ``table_file_name(name, n, f)`` into ``out_dir``; returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, table_file_name(name, n, f))
    with open(path, "wb") as fh:
        fh.write(table_bytes(name, n, f))
    return path


def read_table(path: str) -> Tuple[str, int, int, List[int]]:
    """Parse a table file; returns ``(name, n, f, entries)`` (signed integers)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != _FILE_MAGIC or len(data) < 16:
        raise ValueError("not a table file")
    ln = int.from_bytes(data[8:12], "little")
    pos = 12 + ln
    if len(data) < pos + 8:
        raise ValueError("truncated table file")
    name = data[12:pos].decode("utf-8")
    n = int.from_bytes(data[pos:pos + 4], "little")
    f = int.from_bytes(data[pos + 4:pos + 8], "little")
    pos += 8
    if not 1 <= n <= 30 or len(data) != pos + 8 * (1 << n):
        raise ValueError("table file has the wrong length")
    entries = [int.from_bytes(data[pos + 8 * j:pos + 8 * (j + 1)], "little", signed=True) for j in range(1 << n)]
    return name, n, f, entries


def main(argv=None) -> int:
    """``python -m couplet.tables NAME N F OUT_DIR``: write a table file and print its hash."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 4 or argv[0] not in NAMES:
        print(f"usage: python -m couplet.tables NAME N F OUT_DIR   (NAME in {', '.join(NAMES)})")
        return 2
    path = write_table(argv[3], argv[0], int(argv[1]), int(argv[2]))
    print(f"wrote {path}; table_hash {table_hash(argv[0], int(argv[1]), int(argv[2])).hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
