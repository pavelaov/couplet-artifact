"""Authenticated AND triples: leaky AND, bucketing, consistency check (DESIGN.md 7.4, 7.5).

All functions run ONE party (index ``role``, key share ``delta`` = Delta^B_role)
on its channel endpoint.

Leaky AND (``leaky_ands``).  Inputs of party b for every triple: its share x_b
in BDOZ form (M[x_b] under the peer's key, K[x_{1-b}] for the peer's share)
and its SPDZ-style share (y_b, m_b) of y.  One exchange:

    party b sends   G_b = H(K[x_{1-b}]) XOR H(K[x_{1-b}] XOR Delta^B_b) XOR m_b ,
    keeps           S_b = H(K[x_{1-b}]) ,
    and computes    R_b = H(M[x_b]) XOR x_b G_{1-b}          ( = S_{1-b} XOR x_b m_{1-b} ),
                    n_b = x_b m_b XOR R_b XOR S_b ,   z_b = lsb(n_b) .

Then n_0 XOR n_1 = x y Delta^B and, because lsb(Delta^B) = 1, z_0 XOR z_1 = x y:
``(z_b, n_b)`` is ``<z>`` with z = x y.  ``H`` is ``hashes.H_ccr`` with the
global triple index as tweak and the domain
``session_dom(DOM_CCR, "aand/<sender of the transfer>", sid)``.  The transfer
messages of at most ``LAAND_MSG_TRIPLES`` = 2^24 triples travel in one
exchange; a larger batch is split into consecutive exchanges of that many
triples (one further round each), exactly as the native code does, so that a
message stays below the frame limit of both implementations.

Bucketing (``bucket_permutation``, ``bucket_combine``).  After ALL leaky
triples exist, a public permutation is derived from a coin toss (sort by
128-bit keys expanded from the coin; ties have probability below n^2 / 2^129)
and the triples are grouped into buckets of size B.  Within a bucket with head
(x', y', z') every other member (x'', y'', z'') is folded in with the opened
bit d = y' XOR y'':  x = x' XOR x'',  y = y',  z = z' XOR z'' XOR d x''.

Consistency check (``triple_check``), on the RAW leaky triples: with 128 fresh
random ``<r_j>``, ``<R> = sum_j X^j <r_j>``, and chi_i from a coin toss, the
parties open W = sum_i chi_i z_i XOR R and run ``check.bool_check`` on W (MAC
shares sum_i chi_i n_{i,b} XOR mac_b(R)) together with all opened d bits.
The triples are returned only after this check has passed.

``and_triples`` chains the steps for one batch and books the traffic under the
channel phases ``"cotext"``, ``"abit"``, ``"laand"``, ``"bucket"``, ``"check"``.
It runs with the ``n`` and the bucket size it is given; the caller chooses
them with ``params.triple_request`` (``RealPrep.and_triples`` does), which
pads a small request so that the check covers at least
``params.MIN_LEAKY_TRIPLES`` leaky triples.

Test hooks.  Deviations are injected through the ``cheat`` dictionary (default
None = honest); the honest code path only tests ``if cheat``:
  ``"g_error": (idx, E, guess)``  add E_i to the transfer message of triple
        idx_i and assume x_{idx_i} of the peer equals guess_i (Lemma 3);
  ``"flip_d": [positions]``       flip own shares of opened d bits;
  ``"w_error": D``                add D to the own share of W.
"""

from __future__ import annotations

import numpy as np

from .. import gf128 as gf
from .. import hashes as hs
from ..hashes import ProtocolAbort
from .check import bool_check
from .params import (DOM_CCR, LABEL_LAAND, PAD, USE_BUCKET_PERM, USE_TRIPLE_CHI, prg_dom,
                     session_dom)
from .shares import Bdoz, Keys, Shares, bit_mask, exchange_bits, spdz_shares

_XOR = np.bitwise_xor.reduce
COIN_SEED_BYTES = hs.SEED_BYTES

# Triples per transfer message of the leaky AND (the constant of cpp/include/couplet/aand.hpp):
# larger batches are split into several exchanges.
LAAND_MSG_TRIPLES = 1 << 24


def h_dom(sid: bytes, sender: int) -> int:
    """Domain of H for the transfer sent by party ``sender`` in session ``sid``."""
    if sender not in (0, 1):
        raise ValueError("sender must be 0 or 1")
    return session_dom(DOM_CCR, LABEL_LAAND + (b"0", b"1")[sender], sid)


# --------------------------------------------------------------------------- #
# Leaky AND (section 7.4)
# --------------------------------------------------------------------------- #

def leaky_ands(chan, sid: bytes, role: int, delta: int, index0: int, x_own: Bdoz, x_keys: Keys,
               y: Shares, cheat=None) -> Shares:
    """Leaky ``<z>`` with z = x AND y for n triples (16 n bytes per party).

    One round per ``LAAND_MSG_TRIPLES`` triples, i.e. one round up to 2^24
    triples.  ``index0`` is the global index of the first triple; indices must
    never be reused within a session (they are the tweak of H).
    """
    n = len(y)
    if len(x_own) != n or len(x_keys) != n:
        raise ValueError("x and y must have the same length")
    if index0 < 0 or index0 + n > 1 << 64:
        raise ValueError("triple index out of the 64-bit range")
    tweak = np.arange(index0, index0 + n, dtype=np.uint64)
    dom_out = h_dom(sid, role)       # transfer from this party to the peer
    dom_in = h_dom(sid, 1 - role)    # transfer from the peer to this party

    s_own = hs.H_ccr(x_keys.keys, tweak, dom_out)                       # S_b = H(K[x_{1-b}])
    g_own = s_own ^ hs.H_ccr(gf.xor_scalar(x_keys.keys, delta), tweak, dom_out) ^ y.macs
    if cheat and "g_error" in cheat:
        g_own = _cheat_g(g_own, cheat["g_error"])
    parts = []
    for c0 in range(0, max(n, 1), LAAND_MSG_TRIPLES):       # one exchange also for n = 0
        c1 = min(n, c0 + LAAND_MSG_TRIPLES)
        reply = chan.exchange(gf.vec_to_bytes(g_own[c0:c1]))
        if len(reply) != gf.ELEM_BYTES * (c1 - c0):
            raise ProtocolAbort("leaky AND: transfer message has the wrong length")
        parts.append(gf.bytes_to_vec(reply))
    g_peer = parts[0] if len(parts) == 1 else np.concatenate(parts)

    sel = bit_mask(x_own.bits)
    n_own = hs.H_ccr(x_own.macs, tweak, dom_in)                          # H(M[x_b])
    n_own ^= g_peer & sel                                               # R_b
    n_own ^= y.macs & sel                                               # x_b m_b
    n_own ^= s_own                                                      # S_b
    if cheat and "g_error" in cheat:
        _cheat_n(n_own, cheat["g_error"])
    return Shares(gf.lsb(n_own), n_own)


def _cheat_g(g_own: np.ndarray, spec) -> np.ndarray:
    """Test hook: add the errors E to the transfer messages of the attacked triples."""
    idx, err, _ = spec
    g_bad = g_own.copy()
    g_bad[np.asarray(idx)] ^= gf.as_vec(err)
    return g_bad


def _cheat_n(n_own: np.ndarray, spec) -> None:
    """Test hook: adjust the own MAC share for the guessed peer bits (Lemma 3)."""
    idx, err, guess = spec
    n_own[np.asarray(idx)] ^= gf.as_vec(err) & bit_mask(guess)


# --------------------------------------------------------------------------- #
# Bucketing (section 7.5)
# --------------------------------------------------------------------------- #

def bucket_permutation(seed: bytes, n: int) -> np.ndarray:
    """Public permutation of range(n) from a 16-byte coin: argsort of 128-bit keys.

    Key i is element i of ``hashes.prg_gf128(seed, prg_dom(USE_BUCKET_PERM), n)``
    compared as the integer (high word, low word); the sort is stable.
    """
    keys = hs.prg_gf128(bytes(seed), prg_dom(USE_BUCKET_PERM), n)
    return np.lexsort((keys[:, 0], keys[:, 1]))


def bucket_combine(chan, x: Shares, y: Shares, z: Shares, perm: np.ndarray, bucket: int,
                   cheat=None):
    """Combine ``len(perm) / bucket`` buckets; returns ``(a, b, c, d, d_macs)``.

    Bucket k consists of the leaky triples ``perm[k * bucket + i]``, i < bucket,
    the first one being the head.  ``d`` are the opened bits
    (head y XOR member y, row-major over (bucket, member)), ``d_macs`` this
    party's MAC shares of them; both must go into the next BoolCheck.
    One round, ceil(n (B - 1) / 8) bytes per party.
    """
    total = len(x)
    if len(y) != total or len(z) != total or perm.shape != (total,):
        raise ValueError("x, y, z and the permutation must have the same length")
    if bucket < 1 or total % bucket:
        raise ValueError("the number of leaky triples must be a multiple of the bucket size")
    groups = perm.reshape(total // bucket, bucket)
    head, rest = groups[:, 0], groups[:, 1:]

    d_share = (y.bits[head][:, None] ^ y.bits[rest]).reshape(-1)
    d_macs = (y.macs[head][:, None, :] ^ y.macs[rest]).reshape(-1, 2)
    sent = d_share
    if cheat and "flip_d" in cheat:
        sent = d_share.copy()
        sent[np.asarray(cheat["flip_d"])] ^= 1
    d = sent ^ exchange_bits(chan, sent)

    a = Shares(_XOR(x.bits[groups], axis=1), _XOR(x.macs[groups], axis=1))
    b = Shares(y.bits[head], y.macs[head])
    d_rows = d.reshape(rest.shape)
    c_bits = z.bits[head] ^ _XOR(z.bits[rest] ^ (x.bits[rest] & d_rows), axis=1)
    c_macs = z.macs[head] ^ _XOR(
        z.macs[rest] ^ (x.macs[rest] & np.negative(d_rows.astype(np.uint64))[:, :, None]), axis=1)
    return a, b, Shares(c_bits, c_macs), d, d_macs


# --------------------------------------------------------------------------- #
# Triple consistency check (section 7.4)
# --------------------------------------------------------------------------- #

def triple_check(chan, sid: bytes, counter: int, role: int, delta: int, z: Shares, r: Shares,
                 opened_bits=None, opened_macs=None, cheat=None) -> None:
    """Consistency check of the raw leaky triples ``z`` with the mask bits ``r`` (128 shares).

    Tosses the chi, opens W = sum_i chi_i z_i XOR R and runs BoolCheck number
    ``counter`` on W and on the already opened bits ``opened_bits`` (MAC
    shares ``opened_macs``).  Raises ``ProtocolAbort`` on failure.  Five rounds.
    """
    if len(r) != PAD:
        raise ValueError(f"exactly {PAD} mask bits are required")
    label = b"triple-chi|" + int(counter).to_bytes(8, "little")
    seed = hs.coin_toss(chan, bytes(sid), label, COIN_SEED_BYTES)
    chi = hs.prg_gf128(seed, prg_dom(USE_TRIPLE_CHI), len(z))
    r_val, r_mac = r.to_field()
    w_share = gf.dot_bits(chi, z.bits) ^ r_val
    m_share = gf.dot(chi, z.macs) ^ r_mac
    if cheat and "w_error" in cheat:
        w_share ^= int(cheat["w_error"])
    reply = chan.exchange(gf.int_to_bytes(w_share))
    if len(reply) != gf.ELEM_BYTES:
        raise ProtocolAbort("triple check: share of W has the wrong length")
    w = w_share ^ gf.bytes_to_int(reply)
    if opened_bits is None:
        opened_bits = np.zeros(0, dtype=np.uint8)
        opened_macs = gf.zeros(0)
    bool_check(chan, sid, counter, role, delta, opened_bits, opened_macs,
               extra_field_items=[(w, m_share)])


# --------------------------------------------------------------------------- #
# One batch end to end
# --------------------------------------------------------------------------- #

def and_triples(chan, sid: bytes, role: int, delta: int, source, n: int, bucket: int, *,
                batch: int, index0: int, cheat=None):
    """n bucketed and checked triples ``(a, b, c)`` (each ``Shares``, c = a AND b).

    ``source``  object with ``random_bdoz(m) -> (Bdoz, Keys)`` (``AbitSource``
                or the ideal preparation);
    ``bucket``  bucket size B (``params.bucket_size(n, sigma)``);
    ``batch``   number of this batch in the session (labels of the coin tosses
                and of the BoolCheck);
    ``index0``  global index of the first of the n * B leaky triples.
    """
    n = int(n)
    bucket = int(bucket)
    if n < 1 or bucket < 1:
        raise ValueError("n and the bucket size must be positive")
    total = n * bucket
    own, keys = source.random_bdoz(2 * total + PAD)
    chan.mark("abit")
    x_own, x_keys = own[:total], keys[:total]
    x = spdz_shares(role, delta, x_own, x_keys)
    y = spdz_shares(role, delta, own[total:2 * total], keys[total:2 * total])
    r = spdz_shares(role, delta, own[2 * total:], keys[2 * total:])

    chan.mark("laand")
    z = leaky_ands(chan, sid, role, delta, index0, x_own, x_keys, y, cheat)

    chan.mark("bucket")  # the permutation is tossed only now: all leaky triples exist
    label = b"bucket|" + int(batch).to_bytes(8, "little")
    perm = bucket_permutation(hs.coin_toss(chan, bytes(sid), label, COIN_SEED_BYTES), total)
    a, b, c, d, d_macs = bucket_combine(chan, x, y, z, perm, bucket, cheat)

    chan.mark("check")
    triple_check(chan, sid, batch, role, delta, z, r, d, d_macs, cheat)
    return a, b, c
