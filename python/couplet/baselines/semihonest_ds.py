"""Baseline: semi-honest dealer-free DPF generation in the Doerner-shelat style.

Paradigm (J. Doerner, a. shelat, "Scaling ORAM for Secure Computation", ACM
CCS 2017): both parties expand the whole tree locally, and for every level a
small secure computation selects, according to the secret-shared index bit,
which local aggregate becomes the public correction word.  No MACs, no check:
a party that deviates is not detected.

This module is the semi-honest counterpart of ``leaky_zhang`` ON THE SAME
TREE, so that the two baselines differ only in what active security adds:

  * the same node layout, root (s || t)_b = Delta_b XOR W, the same hash H0
    (``hashes.H_ccr``) and the same expansion ``leaky_zhang.expand_level``;
  * the per-level selection is one correlated OT per direction with choice bit
    alpha_b^(i): P_b receives M_b = K_{1-b} XOR alpha_b^(i) Delta_{1-b}, and
        CW_b^(i) = XOR_j H0(node_b^(i,j)) XOR Delta_b XOR alpha_b^(i) Delta_b
                   XOR K_b XOR M_b
    is an XOR share of  H0(x_0) XOR H0(x_1) XOR (alpha^(i) XOR 1) Delta
    (x_b the on-path nodes): exactly the correction word of Figure 5 of Zhang
    et al., which in turn is the semi-honest half-tree construction of Guo et
    al. (EUROCRYPT 2023) that their protocol starts from;
  * the COT rows come from our IKNP extension WITHOUT the consistency check
    (``PlainCot``: no padding rows, no challenge, no check message), on base
    OTs from ``prep.baseot`` (Masny-Rindal; the base OT is the same code as in
    the malicious backend and is booked separately as "baseot");
  * W is the XOR of one random string per party, sent in the clear (no
    commitment);
  * the final correction word carries the payload only:
        CW_b^(n) = XOR_j H0(leaf_b^(j)) XOR beta_b ,
        v_b^(j)  = H0(leaf_b^(j)) XOR t_b^(j) CW^(n) .

It is NOT the protocol of Doerner and shelat bit for bit: they use a
length-doubling PRG with two seeds per node and compute the per-level
correction word inside a garbled circuit; the tree and the selection used here
are the cheaper ones of the half-tree line of work.  The label "Doerner-shelat
style" refers to the paradigm (local full-domain expansion, one secure
selection per level, semi-honest security).

Output of party b: ``t`` (uint8, shape (N,)) and ``v`` (uint64, shape (N, 2)),
XOR shares of unit(N, alpha) and of unit(N, alpha) * beta, with
alpha = alpha_0 XOR alpha_1 and beta = beta_0 XOR beta_1 in GF(2^128).

Channel phases: "baseot" (once per session), "sh-cot" (U messages and the W
shares), "sh-levels", "sh-leaf".  Bytes sent per party and key: 16 * 64 *
ceil(n / 64) + 16 for the COT flight, 16 n for the levels, 16 for the leaf;
rounds: n + 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import gf128 as gf
from .. import hashes as hs
from ..hashes import ProtocolAbort
from ..prep.abits import sample_delta
from ..prep.baseot import base_ot
from ..prep.params import KAPPA, prg_dom
from ..prep.shares import as_bits, as_rand, bit_mask
from .leaky_zhang import MAX_N, expand_level, h0_domain, hash0, index_bits

PHASE_BASEOT = "baseot"
PHASE_COT = "sh-cot"
PHASE_LEVELS = "sh-levels"
PHASE_LEAF = "sh-leaf"
PHASES = (PHASE_COT, PHASE_LEVELS, PHASE_LEAF)

USE_PLAIN_COT = 0x30  # ctr_prg use byte of the unchecked extension (prep uses 0x10 - 0x14)
LABEL_H0 = b"baseline-semihonest/H0"
_SID_TAG = b"Couplet-baseline-semihonest"
_LE64 = np.dtype("<u8")
_ELEM = gf.ELEM_BYTES


def cot_message_bytes(ell: int) -> int:
    """Size of the U message of one ``PlainCot`` call with ``ell`` choice bits."""
    return KAPPA * 8 * ((int(ell) + 63) // 64)


class PlainCot:
    """Semi-honest IKNP correlated-OT extension in both directions, no consistency check.

    Party b is the COT sender of direction b (global key Delta_b, seeds
    k_j^{Delta_j} from the base OT) and the receiver of direction 1 - b (seed
    pairs).  One call transfers ``ell`` CHOSEN bits per party:

        receiver:  U_j = PRG(k_j^0) XOR PRG(k_j^1) XOR u   for all 128 columns j,
        sender:    q_j = PRG(k_j^{Delta_j}) XOR Delta_j U_j ,

    so that row i of the receiver's matrix (M_i) and of the sender's matrix
    (K_i) satisfy M_i = K_i XOR u_i Delta.  ``start`` produces this party's U
    message, ``finish`` consumes the peer's.  Secure against semi-honest
    parties only: nothing stops a receiver from using different u per column.
    """

    def __init__(self, role: int, delta: int, chosen, seeds0, seeds1) -> None:
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        self.role = role
        self.delta = int(delta)
        self._sel = gf.unpack_bits(self.delta).astype(bool)
        self._chosen = np.ascontiguousarray(gf.as_vec(chosen))
        self._seeds0 = np.ascontiguousarray(gf.as_vec(seeds0))
        self._seeds1 = np.ascontiguousarray(gf.as_vec(seeds1))
        if not self._chosen.shape[0] == self._seeds0.shape[0] == self._seeds1.shape[0] == KAPPA:
            raise ValueError(f"exactly {KAPPA} seeds per direction are required")
        self.calls = 0
        self._pending = None

    def start(self, choice_bits) -> bytes:
        """Receiver side of direction 1 - role: returns the U message for the choice bits."""
        if self._pending is not None:
            raise RuntimeError("an extension is already in progress")
        bits = as_bits(choice_bits)
        ell = bits.shape[0]
        if ell < 1:
            raise ValueError("at least one choice bit is required")
        n_words = (ell + 63) // 64
        dom = prg_dom(USE_PLAIN_COT, 1 - self.role, self.calls)
        t0 = hs.ctr_prg_many(self._seeds0, dom, n_words)
        u_cols = hs.ctr_prg_many(self._seeds1, dom, n_words)
        padded = np.zeros(64 * n_words, dtype=np.uint8)
        padded[:ell] = bits
        u_words = np.packbits(padded, bitorder="little").view(_LE64).astype(np.uint64)
        u_cols ^= t0
        u_cols ^= u_words
        self._pending = (ell, n_words, t0)
        return u_cols.astype(_LE64, copy=False).tobytes()

    def finish(self, u_peer: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Sender side of direction role; returns (M rows of own bits, K rows for the peer's)."""
        if self._pending is None:
            raise RuntimeError("no extension in progress")
        ell, n_words, t0 = self._pending
        self._pending = None
        if len(u_peer) != KAPPA * n_words * 8:
            raise ProtocolAbort("plain COT: U message has the wrong length")
        u_cols = np.frombuffer(u_peer, dtype=_LE64).reshape(KAPPA, n_words)
        q = hs.ctr_prg_many(self._chosen, prg_dom(USE_PLAIN_COT, self.role, self.calls), n_words)
        q[self._sel] ^= u_cols[self._sel].astype(np.uint64)
        self.calls += 1
        return gf.cols_to_rows(t0)[:ell], gf.cols_to_rows(q)[:ell]


@dataclass(eq=False)
class SemiHonestDpfOutput:
    """XOR shares of one party: ``t`` of unit(N, alpha), ``v`` of unit(N, alpha) * beta."""

    n: int
    t: np.ndarray
    v: np.ndarray
    transcript: dict


def semihonest_dpf(chan, sid: bytes, key_id: int, role: int, cot: PlainCot, n: int,
                   alpha_share: int, beta_share: int, rng=None) -> SemiHonestDpfOutput:
    """One semi-honest key generation for party ``role`` (n + 2 rounds); see the module text."""
    if role not in (0, 1):
        raise ValueError("role must be 0 or 1")
    if chan.party != role or cot.role != role:
        raise ValueError("role does not match the channel endpoint or the COT state")
    n = int(n)
    if not 1 <= n <= MAX_N:
        raise ValueError(f"n must be in 1..{MAX_N}")
    beta_share = int(beta_share)
    if beta_share < 0 or beta_share >> 128:
        raise ValueError("beta_share must be an element of GF(2^128)")
    a_bits = index_bits(alpha_share, n)
    rand = as_rand(rng)
    delta = cot.delta
    ssid = hs.sha256(_SID_TAG, bytes(sid), int(key_id).to_bytes(8, "little"))
    dom = h0_domain(ssid, LABEL_H0)
    phase = chan.phase
    try:
        # Selection material: one COT per level and direction, choice bits alpha_b^(i);
        # the public offset W = W_0 XOR W_1 rides on the same flight.
        chan.mark(PHASE_COT)
        w_own = rand.bytes(_ELEM)
        reply = chan.exchange(cot.start(a_bits) + w_own)
        if len(reply) != cot_message_bytes(n) + _ELEM:
            raise ProtocolAbort("semi-honest DPF: first flight has the wrong length")
        macs, keys = cot.finish(reply[:-_ELEM])
        w = gf.bytes_to_int(w_own) ^ gf.bytes_to_int(reply[-_ELEM:])
        # share of alpha^(i) Delta:
        #   alpha_b^(i) Delta_b XOR K_b[alpha_{1-b}^(i)] XOR M_b[alpha_b^(i)]
        alpha_term = gf.bit_mul(a_bits, delta) ^ keys ^ macs
        nodes = gf.ints_to_vec([delta ^ w])

        chan.mark(PHASE_LEVELS)
        cws = []
        for i in range(n):
            hashed = hash0(nodes, dom)
            cw_own = gf.xor_reduce(hashed) ^ delta ^ gf.arr_to_int(alpha_term[i])
            reply = chan.exchange(gf.int_to_bytes(cw_own))
            if len(reply) != _ELEM:
                raise ProtocolAbort("semi-honest DPF: correction word has the wrong length")
            cw = cw_own ^ gf.bytes_to_int(reply)
            nodes = expand_level(nodes, hashed, cw)
            cws.append(cw)

        chan.mark(PHASE_LEAF)
        hashed = hash0(nodes, dom)
        cw_own = gf.xor_reduce(hashed) ^ beta_share
        reply = chan.exchange(gf.int_to_bytes(cw_own))
        if len(reply) != _ELEM:
            raise ProtocolAbort("semi-honest DPF: final correction word has the wrong length")
        cw_final = cw_own ^ gf.bytes_to_int(reply)
        t = gf.lsb(nodes)
        v = hashed ^ (gf.int_to_arr(cw_final) & bit_mask(t))
        return SemiHonestDpfOutput(n, t, v, {"W": w, "cws": cws, "cw_final": cw_final})
    finally:
        chan.mark(phase)


class SemiHonestSession:
    """One party's session of the semi-honest baseline: base OT once, then any number of keys.

        s = SemiHonestSession(role, chan, sid);  s.setup()
        out = s.dpf_gen(n, alpha_share, beta_share)

    ``delta`` is this party's COT key share (lsb 1 for P_0 and 0 for P_1, so
    that the control bits of on-path nodes differ).  Both parties must issue
    the same sequence of ``dpf_gen`` calls.
    """

    def __init__(self, role: int, chan, sid: bytes, rng=None) -> None:
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        if chan.party != role:
            raise ValueError("role does not match the channel endpoint")
        self.role = role
        self.chan = chan
        self.sid = bytes(sid)
        self._rand = as_rand(rng)
        self.delta = sample_delta(role, self._rand)
        self._cot: PlainCot | None = None
        self.keys_generated = 0

    def setup(self) -> None:
        """Base OT in both directions (two rounds, 12,288 bytes per party)."""
        if self._cot is not None:
            raise ValueError("setup() was already run")
        phase = self.chan.phase
        self.chan.mark(PHASE_BASEOT)
        try:
            chosen, (seeds0, seeds1) = base_ot(self.chan, self.sid, self.role,
                                               gf.unpack_bits(self.delta), rng=self._rand)
        finally:
            self.chan.mark(phase)
        self._cot = PlainCot(self.role, self.delta, chosen, seeds0, seeds1)

    def dpf_gen(self, n: int, alpha_share: int, beta_share: int) -> SemiHonestDpfOutput:
        """Generate one key pair's full-domain shares; see ``semihonest_dpf``."""
        if self._cot is None:
            raise ValueError("setup() has not been run")
        key_id = self.keys_generated
        self.keys_generated += 1
        return semihonest_dpf(self.chan, self.sid, key_id, self.role, self._cot, n,
                              alpha_share, beta_share, self._rand)


def bytes_per_party(n: int) -> dict:
    """Exact bytes sent by ONE party per key and phase (the base OT is not included)."""
    return {PHASE_COT: cot_message_bytes(n) + _ELEM, PHASE_LEVELS: _ELEM * n, PHASE_LEAF: _ELEM}


def rounds(n: int) -> dict:
    """Rounds per key and phase (the base OT adds two rounds once per session)."""
    return {PHASE_COT: 1, PHASE_LEVELS: n, PHASE_LEAF: 1}
