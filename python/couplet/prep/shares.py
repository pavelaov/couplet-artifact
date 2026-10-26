"""Data model of the Boolean preprocessing (DESIGN.md section 2.2).

Everything here is the view of ONE party.

``Shares``   SPDZ-style authenticated bits ``<x>``: this party's XOR shares
             ``bits`` (uint8, shape (n,)) and MAC shares ``macs`` (uint64,
             shape (n, 2), elements of GF(2^128) as in ``couplet.gf128``).
             Invariant across the two parties:
                 bits_0 XOR bits_1 = x ,   macs_0 XOR macs_1 = x * Delta^B ,
             with Delta^B = Delta^B_0 XOR Delta^B_1.  XOR, slicing, selection
             by index and concatenation are local and need no key; injecting
             a public constant needs the party index and its key share:
             the constant c is held as (c, c Delta^B_0) by P_0 and as
             (0, c Delta^B_1) by P_1.
``Bdoz``     BDOZ bits of this party under the peer's key: own bits ``x`` and
             MACs ``M[x]``.
``Keys``     the peer's BDOZ bits seen from the key holder: ``K[x']`` with
             ``M[x'] = K[x'] XOR x' * (own Delta^B share)``.

Conversions (section 2.2): ``spdz_shares`` turns a BDOZ bit of each party into
a random ``<x>``; ``mask_owner`` / ``mask_other`` turn a BDOZ bit of the owner
into an input mask ``<r>`` whose value is known to the owner.

``Rand`` is the source of this party's secret randomness: the OS CSPRNG by
default, or any object with ``randbytes(n)`` for reproducible tests.

``pack_bit_msg`` / ``unpack_bit_msg`` are the wire encoding of a bit vector
(eight bits per byte, least significant bit first); the decoder rejects a
message of the wrong length or with stray padding bits.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np

from .. import gf128 as gf
from ..hashes import ProtocolAbort

_U64 = np.uint64
_LE64 = np.dtype("<u8")


# --------------------------------------------------------------------------- #
# Randomness
# --------------------------------------------------------------------------- #

class Rand:
    """Secret randomness of one party (OS CSPRNG, or ``rng.randbytes`` if given)."""

    def __init__(self, rng=None) -> None:
        self._randbytes = secrets.token_bytes if rng is None else rng.randbytes

    def bytes(self, n: int) -> bytes:
        return bytes(self._randbytes(int(n)))

    def words(self, n: int) -> np.ndarray:
        """n uniform 64-bit words, shape (n,)."""
        return np.frombuffer(self.bytes(8 * n), dtype=_LE64).astype(np.uint64)

    def vec(self, n: int) -> np.ndarray:
        """n uniform elements of GF(2^128), shape (n, 2)."""
        return gf.bytes_to_vec(self.bytes(gf.ELEM_BYTES * n))

    def bits(self, n: int) -> np.ndarray:
        """n uniform bits as a uint8 array."""
        raw = np.frombuffer(self.bytes((n + 7) // 8), dtype=np.uint8)
        return np.unpackbits(raw, bitorder="little")[:n]

    def elem(self) -> int:
        """Uniform element of GF(2^128)."""
        return int.from_bytes(self.bytes(gf.ELEM_BYTES), "little")

    def below(self, m: int) -> int:
        """Integer in [0, m), uniform up to a statistical distance below 2^-128."""
        m = int(m)
        if m < 1:
            raise ValueError("m must be positive")
        return int.from_bytes(self.bytes((m.bit_length() + 7) // 8 + 16), "little") % m


def as_rand(rng) -> Rand:
    """``rng`` itself if it is a ``Rand``; otherwise a ``Rand`` wrapping it (None = OS CSPRNG)."""
    return rng if isinstance(rng, Rand) else Rand(rng)


# --------------------------------------------------------------------------- #
# Validation and wire encoding of bit vectors
# --------------------------------------------------------------------------- #

def as_bits(bits) -> np.ndarray:
    """Validate a 0/1 vector and return it as a uint8 array of shape (n,)."""
    b = np.asarray(bits)
    if b.ndim != 1:
        raise ValueError("expected a bit vector of shape (n,)")
    if b.dtype == np.bool_:
        return b.astype(np.uint8)
    if b.dtype.kind not in "ui":
        raise ValueError("bits must have an integer or boolean dtype")
    if b.size and (int(b.max()) > 1 or int(b.min()) < 0):
        raise ValueError("bits must be 0 or 1")
    return b if b.dtype == np.uint8 else b.astype(np.uint8)


def bit_mask(bits) -> np.ndarray:
    """Column of 64-bit masks, shape (n, 1): all ones where the bit is 1, zero elsewhere."""
    return np.negative(as_bits(bits).astype(np.uint64))[:, None]


def pack_bit_msg(bits) -> bytes:
    """Bit vector -> ceil(n / 8) bytes, bit i in bit (i mod 8) of byte floor(i / 8)."""
    return np.packbits(as_bits(bits), bitorder="little").tobytes()


def unpack_bit_msg(data: bytes, n: int) -> np.ndarray:
    """Inverse of ``pack_bit_msg`` for a received message; ``ProtocolAbort`` if malformed."""
    n = int(n)
    if len(data) != (n + 7) // 8:
        raise ProtocolAbort("bit message has the wrong length")
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    if bits[n:].any():
        raise ProtocolAbort("bit message has stray padding bits")
    return bits[:n].copy()


def exchange_bits(chan, bits) -> np.ndarray:
    """Send own bits and receive the peer's equally long vector (one round); returns the latter."""
    bits = as_bits(bits)
    return unpack_bit_msg(chan.exchange(pack_bit_msg(bits)), bits.shape[0])


def _index(key):
    """Normalise an index: an integer selects a length-1 vector, never a scalar."""
    if isinstance(key, (int, np.integer)):
        return [int(key)]
    return key


# --------------------------------------------------------------------------- #
# SPDZ-style shares
# --------------------------------------------------------------------------- #

@dataclass(eq=False)
class Shares:
    """One party's shares of n authenticated bits ``<x>`` (see the module docstring)."""

    bits: np.ndarray
    macs: np.ndarray

    def __post_init__(self) -> None:
        self.bits = as_bits(self.bits)
        self.macs = gf.as_vec(self.macs)
        if self.bits.shape[0] != self.macs.shape[0]:
            raise ValueError("bits and macs must have the same length")

    def __len__(self) -> int:
        return self.bits.shape[0]

    def __getitem__(self, key) -> "Shares":
        """Slice, integer, integer array or boolean mask -> the selected shares."""
        key = _index(key)
        return Shares(self.bits[key], self.macs[key])

    def take(self, idx) -> "Shares":
        """Shares at the positions ``idx`` (an integer array; repetitions allowed)."""
        idx = np.asarray(idx)
        if idx.dtype.kind not in "ui":
            raise ValueError("take expects an integer index array")
        return Shares(self.bits[idx], self.macs[idx])

    def __xor__(self, other: "Shares") -> "Shares":
        """Shares of the bitwise XOR (local)."""
        if not isinstance(other, Shares):
            return NotImplemented
        if len(other) != len(self):
            raise ValueError("share vectors must have the same length")
        return Shares(self.bits ^ other.bits, self.macs ^ other.macs)

    def copy(self) -> "Shares":
        return Shares(self.bits.copy(), self.macs.copy())

    @classmethod
    def zeros(cls, n: int) -> "Shares":
        """Shares of n zero bits (valid for both parties)."""
        return cls(np.zeros(n, dtype=np.uint8), gf.zeros(n))

    @classmethod
    def concat(cls, parts) -> "Shares":
        """Concatenation of share vectors, in order."""
        parts = list(parts)
        if not parts:
            return cls.zeros(0)
        return cls(np.concatenate([p.bits for p in parts]),
                   np.concatenate([p.macs for p in parts]))

    @classmethod
    def constant(cls, role: int, delta: int, consts) -> "Shares":
        """Shares of public bits c: P_0 holds (c, c Delta^B_0), P_1 holds (0, c Delta^B_1)."""
        _check_role(role)
        consts = as_bits(consts)
        bits = consts.copy() if role == 0 else np.zeros_like(consts)
        return cls(bits, gf.bit_mul(consts, delta))

    def add_const(self, role: int, delta: int, consts) -> "Shares":
        """Shares of ``x XOR c`` for public bits c (one bit per position)."""
        consts = as_bits(consts)
        if consts.shape[0] != len(self):
            raise ValueError("one constant per share is required")
        return self ^ Shares.constant(role, delta, consts)

    def invert(self, role: int, delta: int) -> "Shares":
        """Shares of ``NOT x``."""
        return self.add_const(role, delta, np.ones(len(self), dtype=np.uint8))

    def mul_const(self, consts) -> "Shares":
        """Shares of ``c AND x`` for public bits c (keeps the positions with c = 1)."""
        consts = as_bits(consts)
        if consts.shape[0] != len(self):
            raise ValueError("one constant per share is required")
        return Shares(self.bits & consts, self.macs & bit_mask(consts))

    def to_field(self) -> tuple[int, int]:
        """Shares of the element ``sum_j X^j x_j`` of GF(2^128) (exactly 128 bits).

        Returns (value share, MAC share); across the parties the value shares
        XOR to R = sum_j X^j x_j and the MAC shares to R * Delta^B.
        """
        if len(self) != 128:
            raise ValueError("a field element needs exactly 128 authenticated bits")
        return gf.pack_bits(self.bits), gf.horner(self.macs)


def _check_role(role: int) -> None:
    if role not in (0, 1):
        raise ValueError("role must be 0 or 1")


# --------------------------------------------------------------------------- #
# BDOZ material
# --------------------------------------------------------------------------- #

@dataclass(eq=False)
class Bdoz:
    """Own bits with their MACs under the peer's key: ``M[x] = K[x] XOR x Delta^B_peer``."""

    bits: np.ndarray
    macs: np.ndarray

    def __post_init__(self) -> None:
        self.bits = as_bits(self.bits)
        self.macs = gf.as_vec(self.macs)
        if self.bits.shape[0] != self.macs.shape[0]:
            raise ValueError("bits and macs must have the same length")

    def __len__(self) -> int:
        return self.bits.shape[0]

    def __getitem__(self, key) -> "Bdoz":
        key = _index(key)
        return Bdoz(self.bits[key], self.macs[key])

    @classmethod
    def concat(cls, parts) -> "Bdoz":
        parts = list(parts)
        if not parts:
            return cls(np.zeros(0, dtype=np.uint8), gf.zeros(0))
        return cls(np.concatenate([p.bits for p in parts]),
                   np.concatenate([p.macs for p in parts]))


@dataclass(eq=False)
class Keys:
    """MAC keys ``K[x']`` of this party for the peer's BDOZ bits ``x'``."""

    keys: np.ndarray

    def __post_init__(self) -> None:
        self.keys = gf.as_vec(self.keys)

    def __len__(self) -> int:
        return self.keys.shape[0]

    def __getitem__(self, key) -> "Keys":
        return Keys(self.keys[_index(key)])

    @classmethod
    def concat(cls, parts) -> "Keys":
        parts = list(parts)
        if not parts:
            return cls(gf.zeros(0))
        return cls(np.concatenate([p.keys for p in parts]))


# --------------------------------------------------------------------------- #
# Conversions of DESIGN.md 2.2
# --------------------------------------------------------------------------- #

def spdz_shares(role: int, delta: int, own: Bdoz, peer: Keys) -> Shares:
    """Random ``<x>`` with x = x_0 XOR x_1 from one BDOZ bit of each party.

    Party b holds ``own`` = (x_b, M[x_b]) and ``peer`` = K[x_{1-b}]; its MAC
    share is ``m_b = M[x_b] XOR K[x_{1-b}] XOR x_b Delta^B_b``.  The result
    shares its bit array with ``own`` (no copy).
    """
    _check_role(role)
    if len(own) != len(peer):
        raise ValueError("own bits and peer keys must have the same length")
    return Shares(own.bits, own.macs ^ peer.keys ^ gf.bit_mul(own.bits, delta))


def mask_owner(delta: int, own: Bdoz) -> Shares:
    """Input mask ``<r>`` at its owner: share r, MAC share ``M[r] XOR r Delta^B_owner``.

    The result shares its bit array with ``own`` (no copy).
    """
    return Shares(own.bits, own.macs ^ gf.bit_mul(own.bits, delta))


def mask_other(peer: Keys) -> Shares:
    """Input mask ``<r>`` at the other party: share 0, MAC share ``K[r]``."""
    return Shares(np.zeros(len(peer), dtype=np.uint8), peer.keys.copy())
