"""Correlated OT extension with a global key (DESIGN.md section 7.2).

Direction d: P_d is the COT sender with global key Delta^B_d and the seeds
k_j^{Delta_j} from the base OT; P_{1-d} is the COT receiver with the seed
pairs (k_j^0, k_j^1), j < 128.  One ``extend(ell)`` call produces ``ell``
correlations

    receiver: (u_i, t_i)        sender: q_i = t_i XOR u_i * Delta^B_d ,

i.e. random BDOZ bits of the receiver under the sender's key (``Bdoz`` for the
receiver, ``Keys`` for the sender).

Protocol (IKNP with the per-column universal-hash check of Roy, as specified):

  1. Receiver: random u in F_2^{ell'} with ell' = ell + PAD; columns
     t_j^c = PRG(k_j^c); sends U_j = t_j^0 XOR t_j^1 XOR u for all j.
  2. Sender: q_j = PRG(k_j^{Delta_j}) XOR Delta_j U_j.
  3. The sender, AFTER it has received U, sends a fresh 16-byte seed; both
     expand chi_0 .. chi_{ell-1} with ``hashes.prg_gf128``.  The receiver sends
         u~   = sum_{i<ell} u_i chi_i XOR pack(u_ell .. u_{ell'-1})
         V~_j = sum_{i<ell} t^0_{j,i} chi_i XOR pack(t^0_{j,ell} ..)   for all j,
     and the sender aborts unless, for every column j,
         sum_{i<ell} q_{j,i} chi_i XOR pack(q_{j,ell} ..) = V~_j XOR Delta_j u~ .
  4. Rows i < ell are output; the PAD rows are discarded.

Implementation notes
  * The number of rows is rounded up to a multiple of 64 inside one call
    (the protocol is run with ell_r = 64 * ceil(ell / 64) and the surplus rows
    are discarded), so columns are whole 64-bit words; the padding rows are
    then the last two words of every column and ``pack`` of a padding column
    is that pair of words.
  * ``PRG(k)`` is ``hashes.ctr_prg_many`` with domain
    ``prg_dom(USE_COT_PRG, d, call)``: the call counter is part of the
    domain, so repeated ``extend`` calls read disjoint streams.  The chi are
    expanded under ``prg_dom(USE_COT_CHI, d, call)`` from the fresh seed.
  * Bit-matrix transposition is ``gf128.cols_to_rows`` (numpy
    packbits / unpackbits); the 128 column sums are ``gf128.select_xor``.
  * ``sid`` is kept for identification only; nothing here hashes it, because
    the base-OT seeds (which are derived under ``sid``) are already specific
    to the session and the direction.
  * The classes expose the message steps (``start`` / ``answer`` and
    ``absorb`` / ``verify``) so that ``extend_both`` can run the two
    directions in parallel in three exchanges; ``extend`` runs one direction
    in three flights.  Message sizes: 16 bytes per row and 16 * 128 bytes of
    padding for U, 16 bytes for the seed, 16 * 129 bytes for the check.

A corrupted receiver can use an inconsistent u in column j; the check then
fails exactly if Delta_j = 1 (the sender never reads U_j otherwise).  This is
the standard selective-failure leakage of F_COT on the MAC key.
"""

from __future__ import annotations

import numpy as np

from .. import gf128 as gf
from .. import hashes as hs
from ..hashes import ProtocolAbort
from .params import KAPPA, MAX_PRG_COUNTER, PAD, USE_COT_CHI, USE_COT_PRG, prg_dom
from .shares import Bdoz, Keys, as_rand

_LE64 = np.dtype("<u8")
SEED_BYTES = hs.SEED_BYTES
CHECK_BYTES = gf.ELEM_BYTES * (KAPPA + 1)
MAX_ROWS = 1 << 27  # rows of one call; keeps the U message below the 2^32-byte limit
_MAX_CALLS = MAX_PRG_COUNTER + 1


def rounded_rows(ell: int) -> int:
    """Rows actually produced by one call: ``ell`` rounded up to a multiple of 64."""
    return 64 * ((int(ell) + 63) // 64)


def u_message_bytes(ell: int) -> int:
    """Size of the U message of one ``extend(ell)`` call."""
    return KAPPA * (rounded_rows(ell) + PAD) // 8


def _check_ell(ell: int) -> int:
    ell = int(ell)
    if not 1 <= ell <= MAX_ROWS:
        raise ValueError(f"ell must be in 1..{MAX_ROWS}")
    return ell


def _check_seeds(seeds) -> np.ndarray:
    seeds = gf.as_vec(seeds)
    if seeds.shape[0] != KAPPA:
        raise ValueError(f"exactly {KAPPA} seeds are required")
    return np.ascontiguousarray(seeds)


def _pad_columns(cols: np.ndarray) -> np.ndarray:
    """pack(column_j[ell_r ..]) for all j: the last two words of every column, shape (128, 2)."""
    return np.ascontiguousarray(cols[:, -2:])


class CotReceiver:
    """COT receiver of direction ``direction`` (held by party ``1 - direction``)."""

    def __init__(self, chan, sid: bytes, direction: int, seeds0, seeds1, rng=None) -> None:
        if direction not in (0, 1):
            raise ValueError("direction must be 0 or 1")
        self.chan = chan
        self.sid = bytes(sid)
        self.direction = direction
        self._seeds0 = _check_seeds(seeds0)
        self._seeds1 = _check_seeds(seeds1)
        self._rand = as_rand(rng)
        self.calls = 0
        self.rows = 0  # correlations output so far
        self._state = None

    def start(self, ell: int, cheat=None) -> bytes:
        """Step 1: sample u, expand the columns, return the U message.

        ``cheat`` (tests only): ``{"flip_u": [(column, row), ...]}`` flips bits
        of U, i.e. uses an inconsistent u in those columns.
        """
        ell = _check_ell(ell)
        if self._state is not None:
            raise RuntimeError("an extension is already in progress")
        if self.calls >= _MAX_CALLS:
            raise RuntimeError("extension call counter exhausted")
        ell_r = rounded_rows(ell)
        n_words = (ell_r + PAD) // 64
        dom = prg_dom(USE_COT_PRG, self.direction, self.calls)
        t0 = hs.ctr_prg_many(self._seeds0, dom, n_words)
        u_words = self._rand.words(n_words)
        u_cols = hs.ctr_prg_many(self._seeds1, dom, n_words)
        u_cols ^= t0
        u_cols ^= u_words
        if cheat:
            for col, row in cheat.get("flip_u", ()):
                u_cols[col, row // 64] ^= np.uint64(1 << (row % 64))
        self._state = (ell, ell_r, t0, u_words)
        return u_cols.astype(_LE64, copy=False).tobytes()

    def answer(self, seed: bytes) -> tuple[bytes, Bdoz]:
        """Step 3: the check message (u~ || V~_0 .. V~_127) and the output rows."""
        if self._state is None:
            raise RuntimeError("no extension in progress")
        if len(seed) != SEED_BYTES:
            raise ProtocolAbort("COT: challenge seed has the wrong length")
        ell, ell_r, t0, u_words = self._state
        self._state = None
        chi = hs.prg_gf128(bytes(seed), prg_dom(USE_COT_CHI, self.direction, self.calls), ell_r)
        u_bits = np.unpackbits(u_words.astype(_LE64, copy=False).view(np.uint8), bitorder="little")
        u_tilde = gf.dot_bits(chi, u_bits[:ell_r]) ^ gf.pack_bits(u_bits[ell_r:])
        t_rows = gf.cols_to_rows(t0)
        v_tilde = gf.select_xor(t_rows[:ell_r], chi) ^ _pad_columns(t0)
        self.calls += 1
        self.rows += ell
        out = Bdoz(u_bits[:ell].copy(), t_rows[:ell])
        return gf.int_to_bytes(u_tilde) + gf.vec_to_bytes(v_tilde), out

    def extend(self, ell: int, cheat=None) -> Bdoz:
        """One direction on the channel: send U, receive the seed, send the check."""
        self.chan.send(self.start(ell, cheat))
        msg, out = self.answer(self.chan.recv())
        self.chan.send(msg)
        return out


class CotSender:
    """COT sender of direction ``direction`` (held by party ``direction``) with key ``delta``."""

    def __init__(self, chan, sid: bytes, direction: int, delta: int, seeds, rng=None) -> None:
        if direction not in (0, 1):
            raise ValueError("direction must be 0 or 1")
        self.chan = chan
        self.sid = bytes(sid)
        self.direction = direction
        self.delta = int(delta)
        self._delta_bits = gf.unpack_bits(self.delta)
        self._seeds = _check_seeds(seeds)
        self._rand = as_rand(rng)
        self.calls = 0
        self.rows = 0
        self._state = None

    def absorb(self, ell: int, u_msg: bytes) -> bytes:
        """Step 2: read U, form the q columns; returns the fresh challenge seed."""
        ell = _check_ell(ell)
        if self._state is not None:
            raise RuntimeError("an extension is already in progress")
        if self.calls >= _MAX_CALLS:
            raise RuntimeError("extension call counter exhausted")
        ell_r = rounded_rows(ell)
        n_words = (ell_r + PAD) // 64
        if len(u_msg) != KAPPA * n_words * 8:
            raise ProtocolAbort("COT: U message has the wrong length")
        u_cols = np.frombuffer(u_msg, dtype=_LE64).reshape(KAPPA, n_words)
        q = hs.ctr_prg_many(self._seeds, prg_dom(USE_COT_PRG, self.direction, self.calls),
                            n_words)
        sel = self._delta_bits.astype(bool)
        q[sel] ^= u_cols[sel].astype(np.uint64)  # q_j = PRG(k_j^{Delta_j}) XOR Delta_j U_j
        seed = self._rand.bytes(SEED_BYTES)  # sampled only now, after U is fixed
        self._state = (ell, ell_r, q, seed)
        return seed

    def verify(self, check_msg: bytes) -> Keys:
        """Step 3: verify the receiver's check message; returns the keys or aborts."""
        if self._state is None:
            raise RuntimeError("no extension in progress")
        ell, ell_r, q, seed = self._state
        self._state = None
        if len(check_msg) != CHECK_BYTES:
            raise ProtocolAbort("COT: check message has the wrong length")
        u_tilde = gf.bytes_to_int(check_msg[:gf.ELEM_BYTES])
        v_tilde = gf.bytes_to_vec(check_msg[gf.ELEM_BYTES:])
        chi = hs.prg_gf128(seed, prg_dom(USE_COT_CHI, self.direction, self.calls), ell_r)
        q_rows = gf.cols_to_rows(q)
        lhs = gf.select_xor(q_rows[:ell_r], chi) ^ _pad_columns(q)
        rhs = v_tilde ^ gf.bit_mul(self._delta_bits, u_tilde)
        self.calls += 1
        if not np.array_equal(lhs, rhs):
            raise ProtocolAbort("COT consistency check failed")
        self.rows += ell
        return Keys(q_rows[:ell])

    def extend(self, ell: int) -> Keys:
        """One direction on the channel: receive U, send the seed, receive the check."""
        seed = self.absorb(ell, self.chan.recv())
        self.chan.send(seed)
        return self.verify(self.chan.recv())


def extend_both(chan, sender: CotSender, receiver: CotReceiver, ell: int,
                cheat=None) -> tuple[Bdoz, Keys]:
    """Both directions in parallel (three exchanges).

    ``sender`` and ``receiver`` are this party's two roles (it sends in its
    own direction and receives in the other).  Returns (own BDOZ bits, keys
    for the peer's BDOZ bits), ``ell`` each.
    """
    u_peer = chan.exchange(receiver.start(ell, cheat))
    seed_peer = chan.exchange(sender.absorb(ell, u_peer))
    msg, own = receiver.answer(seed_peer)
    keys = sender.verify(chan.exchange(msg))
    return own, keys
