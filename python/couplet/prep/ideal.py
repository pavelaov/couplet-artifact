"""IDEAL preprocessing: a trusted in-process dealer (DESIGN.md section 7.7).

NOT a protocol.  ``IdealDealer`` is one object shared by both parties of a
test or of an online-only measurement; it samples the MAC key shares and all
correlations itself and hands each party its part.  It exists for functional
tests and to isolate the online phase; results obtained with it must be
labelled "ideal preprocessing" and never be reported as all-in numbers.

``IdealPrep(role, dealer)`` exposes the interface of ``real.RealPrep``:

    setup()                      no-op
    random_bits(n)               -> Shares
    input_masks(owner, n)        -> (Shares, bits) at the owner, (Shares, None) at the peer
    and_triples(n)               -> (a, b, c), each Shares, c = a AND b
    random_bdoz(n)               -> (Bdoz, Keys)   (the ``AbitSource`` interface)
    delta                        this party's Delta^B share (int)

The two parties must issue the same sequence of requests (kind and size); the
dealer pairs request number k of P_0 with request number k of P_1 and raises
``ValueError`` if they differ.  The dealer is thread-safe.  It uses no channel,
so nothing is booked in the byte counters.
"""

from __future__ import annotations

import threading

import numpy as np

from .. import gf128 as gf
from ..hashes import ProtocolAbort
from .abits import sample_delta
from .shares import Bdoz, Keys, Shares, as_rand, mask_other, mask_owner, spdz_shares


class IdealDealer:
    """Trusted source of correlated randomness for two in-process parties (ideal mode)."""

    def __init__(self, rng=None) -> None:
        self._rand = as_rand(rng)
        self.deltas = (sample_delta(0, self._rand), sample_delta(1, self._rand))
        self._lock = threading.Lock()
        self._pending: dict[int, tuple] = {}
        self._next = [0, 0]

    @property
    def delta(self) -> int:
        """Delta^B = Delta^B_0 XOR Delta^B_1 (known to the dealer only; for tests)."""
        return self.deltas[0] ^ self.deltas[1]

    # -- request pairing ------------------------------------------------------ #

    def request(self, role: int, kind: str, params: tuple):
        """Serve request (kind, params) of party ``role``; see the module docstring."""
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        with self._lock:
            seq = self._next[role]
            self._next[role] += 1
            if seq in self._pending:
                other_kind, other_params, parts = self._pending.pop(seq)
                if (other_kind, other_params) != (kind, params):
                    raise ValueError(
                        f"ideal dealer: request {seq} differs between the parties: "
                        f"{(other_kind, other_params)} versus {(kind, params)}")
                return parts[role]
            parts = getattr(self, "_make_" + kind)(*params)
            self._pending[seq] = (kind, params, parts)
            return parts[role]

    # -- samplers (each returns the pair (part of P_0, part of P_1)) ----------- #

    def _make_bdoz(self, n: int):
        d0, d1 = self.deltas
        x0, x1 = self._rand.bits(n), self._rand.bits(n)
        k0, k1 = self._rand.vec(n), self._rand.vec(n)  # k_b: keys of P_b for the peer's bits
        own0 = Bdoz(x0, k1 ^ gf.bit_mul(x0, d1))
        own1 = Bdoz(x1, k0 ^ gf.bit_mul(x1, d0))
        return (own0, Keys(k0)), (own1, Keys(k1))

    def _make_bits(self, n: int):
        (own0, keys0), (own1, keys1) = self._make_bdoz(n)
        return (spdz_shares(0, self.deltas[0], own0, keys0),
                spdz_shares(1, self.deltas[1], own1, keys1))

    def _make_masks(self, owner: int, n: int):
        r = self._rand.bits(n)
        key = self._rand.vec(n)
        own = Bdoz(r, key ^ gf.bit_mul(r, self.deltas[1 - owner]))
        at_owner = mask_owner(self.deltas[owner], own)
        parts = [None, None]
        parts[owner] = (at_owner, at_owner.bits.copy())
        parts[1 - owner] = (mask_other(Keys(key)), None)
        return tuple(parts)

    def _share(self, values: np.ndarray):
        """Fresh ``<v>`` sharing of the given bits."""
        n = values.shape[0]
        bits0 = self._rand.bits(n)
        macs0 = self._rand.vec(n)
        return (Shares(bits0, macs0),
                Shares(bits0 ^ values, macs0 ^ gf.bit_mul(values, self.delta)))

    def _make_triples(self, n: int):
        a = self._rand.bits(n)
        b = self._rand.bits(n)
        a0, a1 = self._share(a)
        b0, b1 = self._share(b)
        c0, c1 = self._share(a & b)
        return (a0, b0, c0), (a1, b1, c1)


class IdealPrep:
    """One party's handle on an ``IdealDealer`` (interface of ``RealPrep``; ideal mode)."""

    ideal = True

    def __init__(self, role: int, dealer: IdealDealer) -> None:
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        self.role = role
        self.dealer = dealer
        self.delta = dealer.deltas[role]
        self.aborted = False
        self.counters = {"random_bits": 0, "input_masks": 0, "and_triples": 0}

    def _ask(self, kind: str, *params):
        if self.aborted:
            raise ProtocolAbort("session aborted")
        params = tuple(int(p) for p in params)
        if params[-1] < 0:
            raise ValueError("the number of items must be non-negative")
        return self.dealer.request(self.role, kind, params)

    def setup(self) -> None:
        """Nothing to do: the dealer holds the keys."""

    def abort(self) -> None:
        """Retire the session: no further correlations are handed out."""
        self.aborted = True
        self.delta = None

    def random_bits(self, n: int) -> Shares:
        out = self._ask("bits", n)
        self.counters["random_bits"] += int(n)
        return out

    def random_bdoz(self, n: int) -> tuple[Bdoz, Keys]:
        return self._ask("bdoz", n)

    def input_masks(self, owner: int, n: int):
        if owner not in (0, 1):
            raise ValueError("owner must be 0 or 1")
        out = self._ask("masks", owner, n)
        self.counters["input_masks"] += int(n)
        return out

    def and_triples(self, n: int):
        out = self._ask("triples", n)
        self.counters["and_triples"] += int(n)
        return out
