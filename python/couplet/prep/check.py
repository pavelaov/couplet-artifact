"""BoolCheck: the batched MAC check of opened authenticated values (DESIGN.md 2.2).

Setting.  A bit ``<y_j>`` has been opened to the public value ``y~_j``; party b
holds the MAC share ``m_{j,b}`` and its key share ``Delta^B_b``.  It computes

    V_{j,b} = m_{j,b} XOR y~_j * Delta^B_b .

Since ``m_{j,0} XOR m_{j,1} = y_j * Delta^B``, the two parties hold the same
``V_j`` exactly when ``y~_j = y_j``; otherwise the values differ by
``Delta^B``, which the cheating party cannot predict.  The same holds for an
opened element W of GF(2^128) with MAC shares M_b: ``V_b = M_b XOR W * Delta^B_b``.

Comparison (a refinement of the linear-combination check of DESIGN.md 2.2,
recorded in design/CHANGELOG.md): each party hashes its whole list of V values
with SHA-256, together with the session identifier and a check counter, and
the two digests are compared by a simultaneous commit-then-open
(``couplet.hashes.committed_exchange``: both commit in one exchange, both
open in the next, each commitment is bound to its sender).  No coin toss is
needed and nothing but the digest is revealed.  A party that opened a wrong
value passes only if it commits to the digest of a string that contains the
honest party's key share, i.e. with probability about q_H / 2^127 for q_H
hash queries.

A mismatch raises ``ProtocolAbort``; the caller must then retire the session
(change G of DESIGN.md).  Two rounds, 96 bytes per party.
"""

from __future__ import annotations

import secrets

import numpy as np

from .. import gf128 as gf
from .. import hashes as hs
from ..hashes import ProtocolAbort
from .shares import as_bits

_TAG = b"Couplet-boolcheck"


def check_values(delta: int, opened_bits, macs) -> np.ndarray:
    """``V_j = m_j XOR y~_j * Delta^B_b`` for every opened bit, shape (n, 2)."""
    opened = as_bits(opened_bits)
    macs = gf.as_vec(macs)
    if opened.shape[0] != macs.shape[0]:
        raise ValueError("one MAC share per opened bit is required")
    return macs ^ gf.bit_mul(opened, delta)


def check_digest(sid: bytes, counter: int, delta: int, opened_bits, macs,
                 extra_field_items=()) -> bytes:
    """SHA-256 digest of this party's V values (bits first, then field elements)."""
    v_bits = check_values(delta, opened_bits, macs)
    v_field = gf.ints_to_vec(int(m) ^ gf.mul(int(w), delta) for w, m in extra_field_items)
    return hs.sha256(_TAG, bytes(sid), int(counter).to_bytes(8, "little"),
                     gf.vec_to_bytes(v_bits), gf.vec_to_bytes(v_field))


def bool_check(chan, sid: bytes, counter: int, role: int, delta: int, opened_bits, macs,
               extra_field_items=()) -> None:
    """Run BoolCheck on everything opened since the last check; ``ProtocolAbort`` on failure.

    ``chan``     this party's channel endpoint (``chan.party`` must equal ``role``);
    ``sid``      session identifier (bytes);
    ``counter``  number of this check within the session; both parties use the
                 same value and never reuse one under the same ``sid``;
    ``role``     this party's index, 0 or 1;
    ``delta``    this party's share Delta^B_role of the Boolean MAC key (int);
    ``opened_bits``  the public opened bits, shape (n,);
    ``macs``     this party's MAC shares of those bits, shape (n, 2);
    ``extra_field_items``  iterable of pairs (W, M): an opened element W of
                 GF(2^128) (int) and this party's MAC share M of it (int).
    """
    if role not in (0, 1):
        raise ValueError("role must be 0 or 1")
    if chan.party != role:
        raise ValueError("role does not match the channel endpoint")
    counter = int(counter)
    digest = check_digest(sid, counter, delta, opened_bits, macs, extra_field_items)
    label = b"boolcheck|" + counter.to_bytes(8, "little")
    peer = hs.committed_exchange(chan, bytes(sid), label, digest)
    if len(peer) != len(digest) or not secrets.compare_digest(bytes(peer), digest):
        raise ProtocolAbort("BoolCheck failed")
