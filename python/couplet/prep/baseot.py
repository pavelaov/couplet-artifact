"""Base oblivious transfer: Masny-Rindal OT over Edwards25519 (DESIGN.md section 7.1).

One instance, with ``G`` the base point of the prime-order subgroup:

  * Receiver, choice c:  b <- Z_q, M = b G;  r_{1-c} = t G for random t;
                         r_c = M - H_c(r_{1-c});  send (r_0, r_1).
  * Sender:              a <- Z_q, A = a G;  for j in {0, 1}:
                         M_j = r_j + H_j(r_{1-j}),  k_j = KDF(sid, d, i, j, a M_j);
                         send A.
  * Receiver:            k_c = KDF(sid, d, i, c, b A).

``H_j(P) = hash_to_group("mr" || d || j || i || sid, enc(P))`` and ``KDF`` is
``couplet.hashes.kdf`` (16-byte output), so every hash is domain-separated by
(sid, direction d, instance i, branch j).

``base_ot`` runs KAPPA = 128 instances in EACH direction in two exchanges.
Direction d is the direction of the later correlated-OT extension in which
P_d is the COT sender: there P_d is the base-OT RECEIVER and its choice bits
are the bits of its MAC key share Delta^B_d, while P_{1-d} is the base-OT
SENDER and obtains both seeds of every instance.  Party ``role`` therefore

  * receives in direction ``role``   -> ``chosen``: the seeds k_j^{c_j};
  * sends in direction ``1 - role``  -> ``pairs``: the seeds (k_j^0, k_j^1).

Every point received from the peer is decoded with
``ed25519.decompress_subgroup`` (curve equation, order-q subgroup, not the
identity); anything else aborts.  Scalar multiplication is variable time; side
channels are out of scope (DESIGN.md 7.1).

Communication: 64 bytes per instance in the first exchange and 32 in the
second, i.e. 128 * 96 = 12,288 bytes per party.
"""

from __future__ import annotations

import numpy as np

from .. import ed25519 as ed
from .. import gf128 as gf
from .. import hashes as hs
from ..hashes import ProtocolAbort
from .params import KAPPA
from .shares import as_bits, as_rand

POINT = ed.POINT_BYTES


def _h_tag(sid: bytes, direction: int, index: int, branch: int) -> bytes:
    """Tag of H_j: fixed-width (direction, branch, index) followed by sid (injective)."""
    return b"mr" + bytes([direction, branch]) + index.to_bytes(4, "little") + sid


def _seed(sid: bytes, direction: int, index: int, branch: int, point: ed.Point) -> bytes:
    return hs.kdf(b"baseot", sid, bytes([direction]), index.to_bytes(4, "little"),
                  bytes([branch]), ed.compress(point))


def _scalar(rand) -> int:
    """Uniform scalar in [1, q - 1]."""
    return rand.below(ed.Q - 1) + 1


def _decode(data: bytes) -> ed.Point:
    """Decode a received point; ``ProtocolAbort`` unless it is a non-identity subgroup element."""
    try:
        return ed.decompress_subgroup(data, allow_identity=False)
    except ValueError as exc:
        raise ProtocolAbort(f"base OT: invalid point ({exc})") from None


class _OtReceiver:
    """Base-OT receiver of one direction (the later COT sender of that direction)."""

    def __init__(self, sid: bytes, direction: int, choices: np.ndarray, rand) -> None:
        self.sid = sid
        self.direction = direction
        self.choices = [int(c) for c in choices]
        self.rand = rand
        self._secrets: list[int] = []

    def flight1(self) -> bytes:
        out = bytearray()
        for i, c in enumerate(self.choices):
            while True:
                b = _scalar(self.rand)
                enc_other = ed.compress(ed.base_mult(_scalar(self.rand)))  # r_{1-c} = t G
                h = ed.hash_to_group(_h_tag(self.sid, self.direction, i, c), enc_other)
                r_c = ed.sub(ed.base_mult(b), h)  # r_c = M - H_c(r_{1-c})
                if not ed.is_identity(r_c):  # the identity is rejected by the sender
                    break
            enc_c = ed.compress(r_c)
            out += (enc_c + enc_other) if c == 0 else (enc_other + enc_c)
            self._secrets.append(b)
        return bytes(out)

    def finish(self, flight2: bytes) -> np.ndarray:
        n = len(self.choices)
        if len(flight2) != POINT * n:
            raise ProtocolAbort("base OT: second flight has the wrong length")
        keys = bytearray()
        for i, c in enumerate(self.choices):
            big_a = _decode(flight2[POINT * i:POINT * (i + 1)])
            keys += _seed(self.sid, self.direction, i, c, ed.scalar_mult(self._secrets[i], big_a))
        self._secrets = []
        return gf.bytes_to_vec(bytes(keys))


class _OtSender:
    """Base-OT sender of one direction (the later COT receiver of that direction)."""

    def __init__(self, sid: bytes, direction: int, count: int, rand) -> None:
        self.sid = sid
        self.direction = direction
        self.count = count
        self.rand = rand
        self.pairs: tuple[np.ndarray, np.ndarray] | None = None

    def flight2(self, flight1: bytes) -> bytes:
        if len(flight1) != 2 * POINT * self.count:
            raise ProtocolAbort("base OT: first flight has the wrong length")
        out = bytearray()
        keys = (bytearray(), bytearray())
        for i in range(self.count):
            enc = (flight1[2 * POINT * i:2 * POINT * i + POINT],
                   flight1[2 * POINT * i + POINT:2 * POINT * (i + 1)])
            r = (_decode(enc[0]), _decode(enc[1]))
            a = _scalar(self.rand)
            out += ed.compress(ed.base_mult(a))
            for j in (0, 1):
                h = ed.hash_to_group(_h_tag(self.sid, self.direction, i, j), enc[1 - j])
                m_j = ed.add(r[j], h)  # M_j = r_j + H_j(r_{1-j})
                keys[j].extend(_seed(self.sid, self.direction, i, j, ed.scalar_mult(a, m_j)))
        self.pairs = (gf.bytes_to_vec(bytes(keys[0])), gf.bytes_to_vec(bytes(keys[1])))
        return bytes(out)


def base_ot(chan, sid: bytes, role: int, choice_bits, *, rng=None, cheat=None):
    """Run the base OTs of both directions; returns ``(chosen, (seeds0, seeds1))``.

    ``chan``         this party's channel endpoint;
    ``sid``          session identifier (bytes);
    ``role``         this party's index b;
    ``choice_bits``  the KAPPA bits of Delta^B_b (bit j = coefficient of x^j);
    ``rng``          optional randomness source (see ``shares.Rand``).

    ``chosen``  uint64 array (KAPPA, 2): seed k_j^{c_j} of direction b
                (P_b is the COT sender of direction b);
    ``seeds0``, ``seeds1``  uint64 arrays (KAPPA, 2): the pairs (k_j^0, k_j^1)
                of direction 1 - b (P_b is the COT receiver of that direction).

    ``cheat`` (tests only; default honest): ``{"flight1": f}`` and/or
    ``{"flight2": f}`` replace an outgoing flight by ``f(flight)``.

    Two rounds.
    """
    if role not in (0, 1):
        raise ValueError("role must be 0 or 1")
    choices = as_bits(choice_bits)
    if choices.shape[0] != KAPPA:
        raise ValueError(f"exactly {KAPPA} choice bits are required")
    sid = bytes(sid)
    rand = as_rand(rng)
    receiver = _OtReceiver(sid, role, choices, rand)
    sender = _OtSender(sid, 1 - role, KAPPA, rand)

    flight1 = receiver.flight1()
    if cheat and "flight1" in cheat:
        flight1 = cheat["flight1"](flight1)
    peer1 = chan.exchange(flight1)

    flight2 = sender.flight2(peer1)
    if cheat and "flight2" in cheat:
        flight2 = cheat["flight2"](flight2)
    peer2 = chan.exchange(flight2)

    chosen = receiver.finish(peer2)
    return chosen, sender.pairs
