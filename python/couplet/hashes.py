"""Hashing, expansion, commitments and coin tossing for the backend.

Two symmetric primitives are used, with a fixed division of labour
(recorded in design/CHANGELOG.md as a refinement of DESIGN.md 2.1 / 2.4 / 7.1):

  * Ascon-p12 (``couplet.ascon.permute`` / ``permute_np``) for the
    correlation-robust hash ``H_ccr`` and for counter-mode expansion
    ``ctr_prg`` (many short, fixed-length calls, vectorised with numpy);
  * SHA-256 (``hashlib``) for everything that absorbs variable-length data:
    transcript hashing, commitments, coin tossing, key derivation and
    hash-to-group.  The pure-Python Ascon sponge is too slow for bulk data.

Ascon-based functions.  Both place a 128-bit secret in the state words
(x0, x1), a 64-bit tweak or counter in x2, a 64-bit domain constant ``dom``
in x3 and zero in x4, apply the permutation once and truncate:

  * ``H_ccr(x, tweak, dom)``     = words (x0, x1) of p(x_lo, x_hi, tweak, dom, 0).
    This is a truncated permutation with no feed-forward: 192 of the 320
    output bits are discarded, so in the random-permutation model it is a
    random function of (x, tweak, dom) and inverse queries do not help.
  * ``ctr_prg(seed, dom, n)``    block i = words (x0, x1, x2) of
    p(seed_lo, seed_hi, i, dom, 0): 192 output bits per call, 128 hidden.

Rules for callers:
  * ``dom`` separates uses.  ``H_ccr(x, i, dom)`` is by construction the first
    two words of block i of ``ctr_prg(x, dom, .)``, so one secret must never be
    used both as a hash input and as a PRG seed under the same ``dom``.
  * A (seed, dom) pair determines one stream; expanding it twice repeats the
    stream.  Use the ``start`` block offset or a fresh ``dom`` to continue.
  * The four ``dom`` values equal to the x3 word of the FSS PRG labels
    (``Couplet-G/-C/-VL/-VR``) are rejected, which keeps every call here
    disjoint from the inputs of the FSS PRG of DESIGN.md 2.1.

SHA-256 based functions.  ``sha256(*parts)`` frames every part with its
8-byte little-endian length, so the encoding of a tuple of byte strings is
injective; ``sha256_expand`` is its counter mode and ``kdf`` its truncation
to a 16-byte key.  ``commit`` / ``verify`` implement Com(m; omega) with a 32-byte
omega.  A commitment is bound to (sid, label, committing party): this stops a
party from replaying the other party's commitment and opening, which would
otherwise break every "both commit, both open, compare" step.
``committed_exchange`` and ``coin_toss`` run on any object with the
``couplet.net.Channel`` interface (``party`` and ``exchange``).

``ProtocolAbort`` is the single exception type for "the peer deviated or
disappeared; discard the session".
"""

from __future__ import annotations

import hashlib
import secrets

import numpy as np

from . import gf128
from .ascon import permute, permute_np

MASK64 = (1 << 64) - 1
SEED_BYTES = 16
COM_BYTES = 32
OPENING_BYTES = 32
COIN_BYTES = 32
PRG_BLOCK_WORDS = 3  # 192 bits of output per permutation call, 128 bits hidden

_CHUNK = 8192  # states per permute_np call; larger batches fall out of the L2 cache

# x3 word of the FSS PRG labels (ASCII, zero-padded to 24 bytes, little-endian words).
RESERVED_DOMS = frozenset(
    int.from_bytes(label.ljust(24, b"\x00")[8:16], "little")
    for label in (b"Couplet-G", b"Couplet-C", b"Couplet-VL", b"Couplet-VR")
)


class ProtocolAbort(Exception):
    """A protocol check failed or the peer went away; the session must be discarded."""


# --------------------------------------------------------------------------- #
# Ascon-p12: correlation-robust hash and counter-mode PRG
# --------------------------------------------------------------------------- #

def _u64(value, name: str) -> int:
    v = int(value)
    if v < 0 or v > MASK64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return v


def _check_dom(dom) -> int:
    d = _u64(dom, "dom")
    if d in RESERVED_DOMS:
        raise ValueError("dom collides with an FSS PRG label word")
    return d


def H_ccr(x, tweak, dom: int) -> np.ndarray:
    """Correlation-robust hash of many 128-bit values.

    ``x``      uint64 array of shape (N, 2), rows = (low word, high word);
    ``tweak``  one 64-bit integer for all rows, or a uint64 array of shape (N,);
    ``dom``    64-bit domain constant.
    Returns a uint64 array of shape (N, 2): row i holds the first 128 bits
    (words x0, x1) of p(x_lo, x_hi, tweak_i, dom, 0).  Nothing is XORed in.
    """
    x = gf128.as_vec(x)
    n = x.shape[0]
    d = _check_dom(dom)
    if isinstance(tweak, (int, np.integer)):
        tw_scalar, tw_arr = _u64(tweak, "tweak"), None
    else:
        tw_scalar, tw_arr = 0, np.asarray(tweak)
        if tw_arr.dtype != np.uint64 or tw_arr.shape != (n,):
            raise ValueError("tweak must be an int or a uint64 array of shape (N,)")
    out = np.empty((n, 2), dtype=np.uint64)
    for a in range(0, n, _CHUNK):
        b = min(n, a + _CHUNK)
        cnt = b - a
        x2 = np.full(cnt, tw_scalar, dtype=np.uint64) if tw_arr is None else tw_arr[a:b]
        y = permute_np(x[a:b, 0], x[a:b, 1], x2,
                       np.full(cnt, d, dtype=np.uint64), np.zeros(cnt, dtype=np.uint64))
        out[a:b, 0] = y[0]
        out[a:b, 1] = y[1]
    return out


def H_ccr_scalar(x: int, tweak: int, dom: int) -> int:
    """Scalar ``H_ccr``: 128-bit int in, 128-bit int out (reference path)."""
    x = int(x)
    if x < 0 or x >> 128:
        raise ValueError("x must be a 128-bit unsigned integer")
    y = permute([x & MASK64, x >> 64, _u64(tweak, "tweak"), _check_dom(dom), 0])
    return y[0] | (y[1] << 64)


def ctr_prg_many(seeds, dom: int, n_words: int, start: int = 0) -> np.ndarray:
    """Counter-mode expansion of M seeds at once.

    ``seeds`` is a uint64 array of shape (M, 2) (rows = low word, high word of
    a 16-byte little-endian seed).  Returns a uint64 array of shape
    (M, n_words); row m is the stream of seed m: the concatenation of the
    blocks (x0, x1, x2) of p(seed_lo, seed_hi, start + i, dom, 0) for
    i = 0, 1, ..., truncated to ``n_words`` words.
    """
    seeds = gf128.as_vec(seeds)
    d = _check_dom(dom)
    n_words = int(n_words)
    start = int(start)
    if n_words < 0:
        raise ValueError("n_words must be non-negative")
    n_blocks = -(-n_words // PRG_BLOCK_WORDS)
    if start < 0 or start + n_blocks > MASK64 + 1:
        raise ValueError("block counter out of the 64-bit range")
    m = seeds.shape[0]
    out = np.empty((m, n_blocks, PRG_BLOCK_WORDS), dtype=np.uint64)
    if m and n_blocks:
        lo = np.ascontiguousarray(seeds[:, 0])
        hi = np.ascontiguousarray(seeds[:, 1])
        step = max(1, _CHUNK // m)
        for a in range(0, n_blocks, step):
            b = min(n_blocks, a + step)
            cnt = b - a
            ctr = np.arange(cnt, dtype=np.uint64) + np.uint64(start + a)
            y = permute_np(np.repeat(lo, cnt), np.repeat(hi, cnt), np.tile(ctr, m),
                           np.full(m * cnt, d, dtype=np.uint64),
                           np.zeros(m * cnt, dtype=np.uint64))
            for word in range(PRG_BLOCK_WORDS):
                out[:, a:b, word] = y[word].reshape(m, cnt)
    return np.ascontiguousarray(out.reshape(m, n_blocks * PRG_BLOCK_WORDS)[:, :n_words])


def _seed_vec(seed: bytes) -> np.ndarray:
    if not isinstance(seed, (bytes, bytearray)) or len(seed) != SEED_BYTES:
        raise ValueError("seed must be 16 bytes")
    return gf128.bytes_to_vec(bytes(seed))


def ctr_prg(seed: bytes, dom: int, n_words: int, start: int = 0) -> np.ndarray:
    """Counter-mode expansion of one 16-byte seed into ``n_words`` uint64 words.

    Block i consists of the words (x0, x1, x2) of
    p(seed_lo, seed_hi, start + i, dom, 0); the output is the concatenation of
    the blocks truncated to ``n_words`` words (shape (n_words,)).
    """
    return ctr_prg_many(_seed_vec(seed), dom, n_words, start)[0]


def prg_bits(seed: bytes, dom: int, nbits: int, start: int = 0) -> np.ndarray:
    """First ``nbits`` bits of the ``ctr_prg`` stream as a uint8 array of 0/1.

    Bit j of the stream is bit (j mod 64) of word floor(j / 64), bit 0 being
    the least significant bit of a word (the convention of DESIGN.md 2.1).
    """
    nbits = int(nbits)
    if nbits < 0:
        raise ValueError("nbits must be non-negative")
    words = ctr_prg(seed, dom, -(-nbits // 64), start)
    raw = np.ascontiguousarray(words.astype("<u8", copy=False)).view(np.uint8)
    return np.unpackbits(raw, bitorder="little")[:nbits]


def prg_gf128(seed: bytes, dom: int, n: int, start: int = 0) -> np.ndarray:
    """``n`` elements of GF(2^128) from the ``ctr_prg`` stream, shape (n, 2).

    Element i is (low word, high word) = stream words (2i, 2i + 1).
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    return ctr_prg(seed, dom, 2 * n, start).reshape(n, 2)


# --------------------------------------------------------------------------- #
# SHA-256: framed hashing, expansion, commitments, coin tossing
# --------------------------------------------------------------------------- #

def sha256(*parts: bytes) -> bytes:
    """SHA-256 of the parts, each preceded by its length as 8 bytes little-endian."""
    h = hashlib.sha256()
    for part in parts:
        if not isinstance(part, (bytes, bytearray, memoryview)):
            raise TypeError("sha256 parts must be bytes-like")
        view = memoryview(part)
        h.update(view.nbytes.to_bytes(8, "little"))
        h.update(view)
    return h.digest()


def sha256_expand(nbytes: int, *parts: bytes) -> bytes:
    """``nbytes`` bytes of SHA-256 in counter mode.

    Block i (i = 0, 1, ...) is ``sha256(*parts, i as 8 bytes little-endian)``.
    """
    nbytes = int(nbytes)
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        out += sha256(*parts, ctr.to_bytes(8, "little"))
        ctr += 1
    return bytes(out[:nbytes])


def kdf(*parts: bytes, nbytes: int = 16) -> bytes:
    """Key derivation: ``sha256("Couplet-kdf", *parts)`` truncated to ``nbytes`` <= 32 bytes."""
    if not 1 <= nbytes <= 32:
        raise ValueError("nbytes must be in 1..32")
    return sha256(b"Couplet-kdf", *parts)[:nbytes]


def _party_tag(party) -> bytes:
    if party is None:
        return b""
    if party not in (0, 1):
        raise ValueError("party must be 0, 1 or None")
    return bytes([party])


def _com(sid: bytes, label: bytes, party, omega: bytes, msg: bytes) -> bytes:
    return sha256(b"Couplet-com", sid, label, _party_tag(party), omega, msg)


def commit(sid: bytes, msg: bytes, *, party: int | None = None,
           label: bytes = b"") -> tuple[bytes, bytes]:
    """Commit to ``msg``: returns (commitment, opening), 32 bytes each.

    commitment = SHA-256 over the framed tuple
    ("Couplet-com", sid, label, party tag, omega, msg) with omega = opening =
    32 fresh random bytes and party tag = b"" (None), b"\\x00" or b"\\x01".
    Whenever both parties commit under the same ``sid``, pass the committer's
    index as ``party`` (or put it into ``sid``); otherwise the peer can replay
    the commitment it has just received.
    """
    omega = secrets.token_bytes(OPENING_BYTES)
    return _com(sid, label, party, omega, msg), omega


def verify(sid: bytes, com: bytes, msg: bytes, opening: bytes, *,
           party: int | None = None, label: bytes = b"") -> bool:
    """True iff ``opening`` opens ``com`` to ``msg`` under (sid, label, party)."""
    if len(com) != COM_BYTES or len(opening) != OPENING_BYTES:
        return False
    return secrets.compare_digest(_com(sid, label, party, opening, msg), bytes(com))


def committed_exchange(chan, sid: bytes, label: bytes, msg: bytes) -> bytes:
    """Simultaneous commit-then-open of one message per party; returns the peer's message.

    Round 1: both parties exchange commitments bound to (sid, label, own party
    index).  Round 2: both exchange ``opening || msg``.  Raises
    ``ProtocolAbort`` if the peer's commitment is malformed or equal to ours,
    or if its opening does not verify.  Two rounds.  The caller checks the
    length and content of the returned message.
    """
    me = chan.party
    if me not in (0, 1):
        raise ValueError("channel has no party index")
    com, omega = commit(sid, msg, party=me, label=label)
    peer_com = chan.exchange(com)
    if len(peer_com) != COM_BYTES:
        raise ProtocolAbort("malformed commitment")
    if secrets.compare_digest(bytes(peer_com), com):
        raise ProtocolAbort("mirrored commitment")
    peer_open = chan.exchange(omega + bytes(msg))
    if len(peer_open) < OPENING_BYTES:
        raise ProtocolAbort("malformed opening")
    peer_omega, peer_msg = peer_open[:OPENING_BYTES], peer_open[OPENING_BYTES:]
    if not verify(sid, peer_com, peer_msg, peer_omega, party=1 - me, label=label):
        raise ProtocolAbort("commitment opening failed")
    return peer_msg


def coin_toss(chan, sid: bytes, label: bytes, nbytes: int) -> bytes:
    """Jointly random string of ``nbytes`` bytes (the same value at both parties).

    Each party commits to 32 random bytes (one simultaneous exchange), both
    open (second exchange) and verify; the two strings are XORed into a seed
    and expanded as ``sha256_expand(nbytes, "Couplet-coin", sid, label, seed)``.
    Raises ``ProtocolAbort`` on a bad opening.  Two rounds.
    """
    mine = secrets.token_bytes(COIN_BYTES)
    theirs = committed_exchange(chan, sid, b"coin|" + bytes(label), mine)
    if len(theirs) != COIN_BYTES:
        raise ProtocolAbort("coin share has the wrong length")
    seed = bytes(a ^ b for a, b in zip(mine, theirs))
    return sha256_expand(nbytes, b"Couplet-coin", sid, bytes(label), seed)
