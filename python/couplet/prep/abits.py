"""Authenticated bits from correlated OT (DESIGN.md section 7.3).

A COT row of direction d is a random BDOZ bit of P_{1-d} under the key
Delta^B_d of P_d.  Party b is the COT sender of direction b and the COT
receiver of direction 1 - b, so

  * a random ``<x>`` uses one row of each direction: party b obtains its share
    x_b with M[x_b] (as receiver) and the key K[x_{1-b}] (as sender), and
    ``shares.spdz_shares`` turns these into the SPDZ-style MAC share;
  * an input mask of P_b uses one row of direction 1 - b only.

Key constraint: lsb(Delta^B_0) = 1 and lsb(Delta^B_1) = 0, so that
lsb(Delta^B) = 1 (needed by the leaky AND of section 7.4).  ``sample_delta``
samples a key share with the right least significant bit and ``AbitSource``
refuses any other.  A corrupted party that ignores the constraint only
produces AND triples that fail the triple consistency check.

``AbitSource`` is the party-local source used by the rest of the pipeline:

  * ``random_bdoz(n)``    n random bits in BDOZ form, (``Bdoz``, ``Keys``): what
                          the leaky AND needs for x;
  * ``random_shares(n)``  n random ``<x>``; with ``keep_bdoz=True`` the BDOZ
                          components are returned as well;
  * ``input_masks(owner, n)``  n input masks of P_owner.

Requests are served in chunks of at most ``batch_rows`` rows per extension
call (bounded memory and message size).  Accounting: the extension flights
and the column expansion are booked under the channel phase ``"cotext"``, the
local conversion to shares under ``"abit"``.
"""

from __future__ import annotations

from .cotext import MAX_ROWS, CotReceiver, CotSender, extend_both
from .shares import Bdoz, Keys, as_rand, mask_other, mask_owner, spdz_shares

DEFAULT_BATCH_ROWS = 1 << 20


def required_lsb(role: int) -> int:
    """lsb of Delta^B_role demanded by DESIGN.md 7.3: 1 for P_0, 0 for P_1."""
    if role not in (0, 1):
        raise ValueError("role must be 0 or 1")
    return 1 - role


def check_delta(role: int, delta: int) -> int:
    """Validate a MAC key share (range and least significant bit); returns it as int."""
    delta = int(delta)
    if delta < 0 or delta >> 128:
        raise ValueError("Delta^B share must be a 128-bit value")
    if delta & 1 != required_lsb(role):
        raise ValueError("lsb(Delta^B_0) must be 1 and lsb(Delta^B_1) must be 0")
    return delta


def sample_delta(role: int, rng=None) -> int:
    """Uniform Delta^B_role subject to the lsb constraint (127 bits of entropy)."""
    return (as_rand(rng).elem() & ~1) | required_lsb(role)


class AbitSource:
    """Party-local source of authenticated bits backed by the two COT directions."""

    def __init__(self, chan, role: int, delta: int, sender: CotSender, receiver: CotReceiver,
                 batch_rows: int = DEFAULT_BATCH_ROWS) -> None:
        self.chan = chan
        self.role = role
        self.delta = check_delta(role, delta)
        if sender.direction != role or receiver.direction != 1 - role:
            raise ValueError("party b sends in direction b and receives in direction 1 - b")
        if sender.delta != self.delta:
            raise ValueError("the COT sender must use this party's Delta^B share")
        if not 1 <= batch_rows <= MAX_ROWS:
            raise ValueError(f"batch_rows must be in 1..{MAX_ROWS}")
        self._sender = sender
        self._receiver = receiver
        self.batch_rows = int(batch_rows)
        self.rows_both = 0   # rows consumed in both directions (random bits)
        self.rows_mask = [0, 0]  # rows consumed for input masks, by owner

    # -- random bits --------------------------------------------------------- #

    def random_bdoz(self, n: int, cheat=None) -> tuple[Bdoz, Keys]:
        """n random BDOZ bits of each party: (own bits and MACs, keys for the peer's bits).

        Three rounds per chunk of ``batch_rows`` rows.  ``cheat`` is passed to
        ``CotReceiver.start`` of the first chunk (tests only).
        """
        n = _count(n)
        own_parts, key_parts = [], []
        done = 0
        self.chan.mark("cotext")
        while done < n:
            step = min(self.batch_rows, n - done)
            own, keys = extend_both(self.chan, self._sender, self._receiver, step,
                                    cheat if done == 0 else None)
            own_parts.append(own)
            key_parts.append(keys)
            done += step
        self.chan.mark("abit")
        self.rows_both += n
        if len(own_parts) == 1:
            return own_parts[0], key_parts[0]
        return Bdoz.concat(own_parts), Keys.concat(key_parts)

    def random_shares(self, n: int, keep_bdoz: bool = False, cheat=None):
        """n random ``<x>``; with ``keep_bdoz`` returns (Shares, Bdoz, Keys)."""
        own, keys = self.random_bdoz(n, cheat)
        shares = spdz_shares(self.role, self.delta, own, keys)
        return (shares, own, keys) if keep_bdoz else shares

    # -- input masks ---------------------------------------------------------- #

    def input_masks(self, owner: int, n: int):
        """n input masks of P_owner: ``(Shares, bits)`` at the owner, ``(Shares, None)`` else.

        Uses rows of direction 1 - owner only (three flights per chunk).
        """
        if owner not in (0, 1):
            raise ValueError("owner must be 0 or 1")
        n = _count(n)
        parts = []
        done = 0
        self.chan.mark("cotext")
        while done < n:
            step = min(self.batch_rows, n - done)
            if owner == self.role:
                parts.append(self._receiver.extend(step))
            else:
                parts.append(self._sender.extend(step))
            done += step
        self.chan.mark("abit")
        self.rows_mask[owner] += n
        if owner == self.role:
            own = Bdoz.concat(parts) if len(parts) != 1 else parts[0]
            shares = mask_owner(self.delta, own)
            return shares, shares.bits.copy()
        keys = Keys.concat(parts) if len(parts) != 1 else parts[0]
        return mask_other(keys), None


def _count(n: int) -> int:
    n = int(n)
    if n < 0:
        raise ValueError("the number of bits must be non-negative")
    return n
