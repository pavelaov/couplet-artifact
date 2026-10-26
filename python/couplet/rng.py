"""Deterministic byte source: SHA-256 in counter mode (for reproducible vectors).

Stream block i (i = 0, 1, 2, ...) is SHA-256(seed || i) with ``seed`` a
32-byte string and ``i`` encoded as an 8-byte little-endian integer.
``randbytes(n)`` returns the next n bytes of the concatenated stream;
``randbelow(m)`` reads 8 bytes, interprets them little-endian and reduces
mod m (the bias is irrelevant for test vectors).  This is not a CSPRNG for
production use; it exists so that the Python and C++ artifacts regenerate
identical golden vectors from the same 32-byte seed.
"""

from __future__ import annotations

import hashlib

SEED_BYTES = 32


class Sha256CounterRng:
    """SHA-256 counter-mode stream with the ``randbytes`` interface of ``random.Random``."""

    def __init__(self, seed: bytes) -> None:
        if len(seed) != SEED_BYTES:
            raise ValueError("seed must be 32 bytes")
        self._seed = bytes(seed)
        self._counter = 0
        self._buf = b""

    def randbytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            if not self._buf:
                self._buf = hashlib.sha256(self._seed + self._counter.to_bytes(8, "little")).digest()
                self._counter += 1
            take = min(n - len(out), len(self._buf))
            out += self._buf[:take]
            self._buf = self._buf[take:]
        return bytes(out)

    def randbelow(self, m: int) -> int:
        if m < 1:
            raise ValueError("m must be positive")
        return int.from_bytes(self.randbytes(8), "little") % m
