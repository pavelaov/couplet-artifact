"""Baseline: the one-bit-leaky full-domain DPF protocol Pi_DPF of Zhang et al., re-implemented.

Source.  W. Zhang, X. Guo, K. Yang, R. Zhu, Y. Yu, X. Wang, "Efficient Actively
Secure DPF and RAM-based 2PC with One-Bit Leakage", IEEE Symposium on Security and Privacy 2024 (ePrint
version of 2024-03-12): Figure 5 (protocol Pi_DPF), Figure 2 (BatchCheck),
Section 3.3 (authenticated sharings, B2F, Convert), Figure 4 (Input, Rand,
Open of Pi_2PC), Figure 6 (functionality F_DPF) and Section 5.

What the protocol does.  The parties hold BDOZ-authenticated bits of the index
alpha and a SPDZ-style authenticated payload beta in GF(2^128).  Both expand
the WHOLE binary tree locally (2^n leaves), exchange one share of a correction
word per level and one final correction word, and obtain SPDZ-style
authenticated sharings of

    u = unit(N, alpha)      and      v = unit(N, alpha) * beta ,    N = 2^n,

followed by one batched MAC check of a random linear combination of all 2N
outputs.  A party that adds an error to its share of a correction word and
"removes" the error on a guessed path passes the check exactly when the guess
is right: the abort leaks one predicate of alpha (functionality F_DPF, Check).
This is the leakage that the generation protocol of DESIGN.md avoids, and the
reason this module exists: it is the same-machine comparison point of
DESIGN.md section 9, Q1.

What is the same as in the paper: every message, every local computation and
the order of the steps of Figure 5 and Figure 2 (mapping below).

What differs from the original implementation (state this next to any number):

  * H0.  The original uses a fixed-key AES-based CCR hash; here H0 is
    ``couplet.hashes.H_ccr`` (truncated Ascon-p12) with tweak 0 and a domain
    constant derived from the key identifier.  H1(x) = H0(x) || H0(x XOR 1) as
    in the paper.
  * F_aBit.  The original instantiates F_aBit / F_COT with EMP's Ferret
    (silent COT, communication sublinear in the number of bits); here the
    authenticated bits come from OUR preprocessing (``couplet.prep``: IKNP
    extension with the per-column consistency check, 128 bits per row).
    Absolute running times and the cost of the authenticated bits therefore
    differ from the paper; the communication of Pi_DPF itself (correction
    words, opening, commitments, coins) is determined by the protocol.
  * F_coin and F_com are ``hashes.coin_toss`` and ``hashes.committed_exchange``
    (SHA-256; 96 bytes per party and coin, 64 + |m| bytes per party and
    committed message, two rounds each).
  * lsb convention.  The paper asks for lsb(Delta_b) = b; our layer has
    lsb(Delta_0) = 1 and lsb(Delta_1) = 0.  Only lsb(Delta_0 XOR Delta_1) = 1
    is used (the control bit t is the lsb of a node), so nothing changes.
  * Bit order.  Level i of Figure 5 consumes alpha^(i), so alpha^(0) is the
    MOST significant bit of the leaf index, whereas Figure 6 writes
    alpha = sum_i alpha^(i) 2^i.  We follow the protocol figure and the
    convention of DESIGN.md (index bits most significant first): leaf j = alpha.
  * Input (Figure 4).  A chosen input bit is authenticated by derandomising a
    random BDOZ bit (one bit per input bit).  Steps 1 (second half) and 2 of
    ``Input`` of Pi_2PC, which re-randomise the keys by a public string and give
    the non-owner a trivial sharing of zero, exist for the garbled-circuit part
    of that paper; the offsets they add are public and cancel in every value of
    Figure 5, so they are not sent here (this favours the baseline by 2 kappa
    bits per input bit).
  * The F_aBit call of step 4 (the bits of the mask r) is issued together with
    the input authentication in ``run_pi_dpf`` so that one extension call
    serves the whole key; r is independent of all inputs and messages, hence
    the order is immaterial.  ``pi_dpf`` itself takes <r> as an argument.

Mapping of Figure 5 to the code (``pi_dpf``):

  Initialize   Delta_b with lsb(Delta_0 XOR Delta_1) = 1      -> the session key share
                                                                 ``delta`` of the prep object
  inputs       [[alpha^(i)]]_b = (alpha_b^(i), K_b[alpha_{1-b}^(i)],
               M_b[alpha_b^(i)]),  <beta>_b = (beta_b, M_b[beta])
                                                              -> ``BdozBits``, ``FieldShare``
  step 1       W <- F_coin;  (s || t)_b^(0,0) := Delta_b XOR W -> "zhang-root"
  step 2       CW_b^(i) := XOR_j H0(node_b^(i,j)) XOR Delta_b
               XOR alpha_b^(i) Delta_b XOR K_b[alpha_{1-b}^(i)] XOR M_b[alpha_b^(i)];
               exchange; CW^(i) := CW_0^(i) XOR CW_1^(i);
               left  := H0(node) XOR t CW^(i),
               right := H0(node) XOR node XOR t CW^(i)         -> "zhang-levels"
  step 3       CW_b^(n) := XOR_j H1(leaf_b^(j)) XOR (beta_b || M_b[beta]);
               exchange;  <u^(j)>_b := (t, leaf),
               (v_b^(j) || M_b[v^(j)]) := H1(leaf) XOR t CW^(n) -> "zhang-leaf"
  step 4       <r> random (F_aBit, B2F, Convert);  chi <- F_coin;
               <a> := sum_j chi^j <u^(j)> + sum_j chi^(N+j) <v^(j)> + <r>
  step 5       open a (both send their share), BatchCheck(<a>, a~)
                                                              -> "zhang-check"
  step 6       output <u^(j)>, <v^(j)>                         -> ``LeakyDpfOutput``

BatchCheck (Figure 2, ``batch_check``): 1. chi <- F_coin; 2. <z> := sum_i chi^i
<y^(i)>, z~ := sum_i chi^i y~^(i); 3. V_b := M_b[z] XOR z~ Delta_b, commit;
4. open, abort unless V_0 = V_1.  Figure 5 calls it with ONE value, for which
chi^0 = 1 and the coin of step 1 is never used; it is tossed nevertheless (the
literal figure) unless ``skip_unused_coin=True`` (two rounds and 96 bytes per
party less).

Channel phases: "zhang-input" (authentication of the inputs), "zhang-root",
"zhang-levels", "zhang-leaf", "zhang-check"; the authenticated bits themselves
are booked by the prep object ("cotext", "abit").  Bytes sent per party by
``pi_dpf``: 16 n + 416 (16 n + 320 without the unused coin); rounds: n + 10
(n + 8).  ``authenticate_inputs`` adds ceil((n + 128) / 8) bytes and one round.

Test hooks (``cheat``, default None = honest; the honest path only tests
``if cheat``):

  ``"cw_error": (level, E, guess)``   add E to the own share of CW^(level)
        (0 <= level < n).  If ``guess`` is not None the cheater assumes that
        the on-path node of that level has index ``guess`` (i.e. that the
        ``level`` most significant bits of alpha equal ``guess``) and XORs E
        into its two children of that node.  The check then passes iff the
        guess is right; for level = 1 this is "iff the most significant bit of
        alpha equals guess".  For level = 0 the root is always on the path, so
        guess = 0 removes the error for certain and nothing is learned.
  ``"final_cw_error": (E_v, E_m, guess)``  the same for CW^(n); ``guess`` is a
        leaf index, the check passes iff alpha = guess.
  ``"a_error": D``                    add D to the own share of the opened a.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field

import numpy as np

from .. import gf128 as gf
from .. import hashes as hs
from ..hashes import ProtocolAbort
from ..prep.params import DOM_CCR, KAPPA, session_dom
from ..prep.shares import Bdoz, Keys, Shares, as_bits, bit_mask, exchange_bits, spdz_shares

PHASE_INPUT = "zhang-input"
PHASE_ROOT = "zhang-root"
PHASE_LEVELS = "zhang-levels"
PHASE_LEAF = "zhang-leaf"
PHASE_CHECK = "zhang-check"
PHASES = (PHASE_INPUT, PHASE_ROOT, PHASE_LEVELS, PHASE_LEAF, PHASE_CHECK)

MAX_N = 26  # 2^26 leaves need about 6 GiB per party; the benchmark stops at n = 20

LABEL_H0 = b"baseline-zhang/H0"
_SID_TAG = b"Couplet-baseline-zhang"
_ONE = np.array([1, 0], dtype=np.uint64)  # the field element 1 as a row
_ELEM = gf.ELEM_BYTES

COIN_BYTES = 96                 # per party: commitment, opening, 32-byte share
COMMIT_OVERHEAD_BYTES = 64      # per party: commitment and opening


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(eq=False)
class BdozBits:
    """One party's part [[x]]_b of BDOZ-style authenticated bits (Section 3.3 of the paper).

    ``bits``  own shares x_b (uint8, shape (n,));
    ``keys``  K_b[x_{1-b}], keys for the peer's shares (uint64, shape (n, 2));
    ``macs``  M_b[x_b] = K_{1-b}[x_b] XOR x_b Delta_{1-b} (uint64, shape (n, 2)).
    The authenticated bit is x = x_0 XOR x_1.
    """

    bits: np.ndarray
    keys: np.ndarray
    macs: np.ndarray

    def __post_init__(self) -> None:
        self.bits = as_bits(self.bits)
        self.keys = gf.as_vec(self.keys)
        self.macs = gf.as_vec(self.macs)
        if not self.bits.shape[0] == self.keys.shape[0] == self.macs.shape[0]:
            raise ValueError("bits, keys and macs must have the same length")

    def __len__(self) -> int:
        return self.bits.shape[0]

    def __getitem__(self, key) -> "BdozBits":
        if isinstance(key, (int, np.integer)):
            key = [int(key)]
        return BdozBits(self.bits[key], self.keys[key], self.macs[key])


@dataclass(frozen=True)
class FieldShare:
    """One party's part <x>_b = (x_b, M_b[x]) of a SPDZ-style sharing of x in GF(2^128)."""

    value: int
    mac: int


@dataclass(eq=False)
class FieldShares:
    """One party's shares of a vector of field elements: ``values`` and ``macs``, shape (N, 2).

    Across the parties the values XOR to v^(j) and the MAC shares to v^(j) * Delta.
    """

    values: np.ndarray
    macs: np.ndarray

    def __len__(self) -> int:
        return self.values.shape[0]


@dataclass(eq=False)
class LeakyDpfOutput:
    """Output of Pi_DPF at one party (step 6 of Figure 5).

    ``u``  ``Shares`` of the N bits unit(N, alpha) (bits and MAC shares; the
           format of ``couplet.prep.shares``, so they are ordinary
           authenticated bits of our Boolean layer);
    ``v``  ``FieldShares`` of the N field elements unit(N, alpha) * beta;
    ``transcript``  the public values seen by this party: ``W``, ``cws`` (list
           of n ints), ``cw_final`` (pair of ints), ``chi``, ``a``.
    """

    n: int
    u: Shares
    v: FieldShares
    transcript: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Local building blocks
# --------------------------------------------------------------------------- #

def sub_sid(sid: bytes, key_id: int) -> bytes:
    """Identifier of one key generation: labels and hash domain of that key."""
    return hs.sha256(_SID_TAG, bytes(sid), int(key_id).to_bytes(8, "little"))


def h0_domain(ssid: bytes, label: bytes = LABEL_H0) -> int:
    """Ascon domain constant of H0 for the key generation ``ssid`` (family 0x48, CCR hash)."""
    return session_dom(DOM_CCR, label, ssid)


def hash0(nodes: np.ndarray, dom: int) -> np.ndarray:
    """H0 on every row of ``nodes`` (shape (m, 2)): the untweaked CCR hash of the paper."""
    return hs.H_ccr(nodes, 0, dom)


def hash1(nodes: np.ndarray, dom: int) -> tuple[np.ndarray, np.ndarray]:
    """H1(x) = H0(x) || H0(x XOR 1): returns the two halves (value part, MAC part)."""
    return hash0(nodes, dom), hash0(nodes ^ _ONE, dom)


def expand_level(nodes: np.ndarray, hashed: np.ndarray, cw: int) -> np.ndarray:
    """One level of the tree (step 2 of Figure 5), for all nodes at once.

    ``nodes`` are the strings (s || t) of level i (t = least significant bit),
    ``hashed`` = H0(nodes), ``cw`` the public correction word CW^(i).  Child 2j
    is  H0(node_j) XOR t_j CW  and child 2j + 1 is  H0(node_j) XOR node_j XOR t_j CW.
    """
    m = nodes.shape[0]
    t_mask = np.negative(nodes[:, 0] & np.uint64(1))[:, None]  # all ones where t = 1
    left = hashed ^ (gf.int_to_arr(cw) & t_mask)
    out = np.empty((2 * m, 2), dtype=np.uint64)
    out[0::2] = left
    out[1::2] = left ^ nodes
    return out


def index_bits(alpha: int, n: int) -> np.ndarray:
    """The n bits of ``alpha``, most significant first (entry i is used at level i)."""
    alpha = int(alpha)
    if not 0 <= alpha < 1 << n:
        raise ValueError("index share out of range")
    return np.array([(alpha >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def poly_eval(coeffs: np.ndarray, x: int) -> int:
    """sum_j x^j * coeffs[j] over GF(2^128) by pairwise folding (log passes over the vector)."""
    v = gf.as_vec(coeffs)
    if v.shape[0] == 0:
        return 0
    power = int(x)
    while v.shape[0] > 1:
        if v.shape[0] & 1:
            v = np.concatenate([v, gf.zeros(1)])
        v = v[0::2] ^ gf.mul_vec_scalar(v[1::2], power)
        power = gf.mul(power, power)
    return gf.arr_to_int(v[0])


def convert(delta: int, x: BdozBits) -> Shares:
    """Convert of Section 3.3: <x>_b := (x_b, K_b[x_{1-b}] XOR M_b[x_b] XOR x_b Delta_b)."""
    return Shares(x.bits, x.keys ^ x.macs ^ gf.bit_mul(x.bits, delta))


def pack(delta: int, x: BdozBits) -> FieldShare:
    """Pack of Figure 4 (B2F, then Convert) for exactly 128 bits: <sum_i X^i x^(i)>_b."""
    return FieldShare(*convert(delta, x).to_field())


# --------------------------------------------------------------------------- #
# F_aBit with chosen inputs, on top of random BDOZ bits
# --------------------------------------------------------------------------- #

def authenticate_inputs(chan, delta: int, own: Bdoz, keys: Keys, bits) -> BdozBits:
    """F_aBit command (auth, b, x, l) for BOTH parties at once (one round, ceil(l / 8) bytes each).

    ``own`` / ``keys`` are l random BDOZ bits of each party (this party's bits
    rho_b with M_b[rho_b], and its keys K_b[rho_{1-b}] for the peer's bits).
    Party b sends d_b = x_b XOR rho_b; afterwards M_b[x_b] := M_b[rho_b] and
    K_b[x_{1-b}] := K_b[rho_{1-b}] XOR d_{1-b} Delta_b.  The result is this
    party's part of [[x^(i)]] with x^(i) = x_0^(i) XOR x_1^(i).
    """
    bits = as_bits(bits)
    if len(own) != bits.shape[0] or len(keys) != bits.shape[0]:
        raise ValueError("one random BDOZ bit of each party is needed per input bit")
    d_peer = exchange_bits(chan, bits ^ own.bits)
    return BdozBits(bits, keys.keys ^ gf.bit_mul(d_peer, delta), own.macs)


# --------------------------------------------------------------------------- #
# BatchCheck (Figure 2)
# --------------------------------------------------------------------------- #

def batch_check(chan, ssid: bytes, label: bytes, delta: int, opened, mac_shares, *,
                skip_unused_coin: bool = False) -> None:
    """BatchCheck of Figure 2 on the opened values y~^(i) with MAC shares M_b[y^(i)] (ints).

    Raises ``ProtocolAbort`` if V_0 != V_1.  Four rounds (two if the coin is
    skipped, which is allowed only for a single value).
    """
    opened = [int(y) for y in opened]
    mac_shares = [int(m) for m in mac_shares]
    if len(opened) != len(mac_shares) or not opened:
        raise ValueError("one MAC share per opened value is required (at least one value)")
    # 1. Both parties call F_coin to sample chi.
    chi = 0
    if len(opened) > 1 or not skip_unused_coin:
        chi = gf.bytes_to_int(hs.coin_toss(chan, ssid, b"batchcheck|" + bytes(label), _ELEM))
    # 2. <z> := sum_i chi^i <y^(i)>  and  z~ := sum_i chi^i y~^(i)   (i starts at 0).
    z_open, z_mac, power = 0, 0, 1
    for y, m in zip(opened, mac_shares):
        z_open ^= gf.mul(power, y)
        z_mac ^= gf.mul(power, m)
        power = gf.mul(power, chi)
    # 3. V_b := M_b[z] XOR z~ Delta_b, committed with F_com.
    v_own = gf.int_to_bytes(z_mac ^ gf.mul(z_open, delta))
    # 4. Both open; abort unless V_0 = V_1.
    v_peer = hs.committed_exchange(chan, ssid, b"batchcheck-V|" + bytes(label), v_own)
    if len(v_peer) != _ELEM or not secrets.compare_digest(bytes(v_peer), v_own):
        raise ProtocolAbort("BatchCheck failed")


# --------------------------------------------------------------------------- #
# Pi_DPF (Figure 5)
# --------------------------------------------------------------------------- #

def _field_elem(value, what: str) -> int:
    value = int(value)
    if value < 0 or value >> 128:
        raise ValueError(f"{what} must be an element of GF(2^128)")
    return value


def _parse_cheat(cheat, n: int):
    """Validate the test hooks; returns (cw_error, final_cw_error, a_error)."""
    if not cheat:
        return None, None, 0
    unknown = set(cheat) - {"cw_error", "final_cw_error", "a_error"}
    if unknown:
        raise ValueError(f"unknown cheat keys: {sorted(unknown)}")
    cw_error = cheat.get("cw_error")
    if cw_error is not None:
        level, err, guess = cw_error
        if not 0 <= level < n:
            raise ValueError("cw_error: level must be in [0, n)")
        if guess is not None and not 0 <= guess < 1 << level:
            raise ValueError("cw_error: guess must be a node index of that level")
        cw_error = (int(level), _field_elem(err, "cw_error"), guess)
    final = cheat.get("final_cw_error")
    if final is not None:
        err_v, err_m, guess = final
        if guess is not None and not 0 <= guess < 1 << n:
            raise ValueError("final_cw_error: guess must be a leaf index")
        final = (_field_elem(err_v, "final_cw_error"), _field_elem(err_m, "final_cw_error"), guess)
    return cw_error, final, _field_elem(cheat.get("a_error", 0), "a_error")


def pi_dpf(chan, sid: bytes, role: int, delta: int, alpha: BdozBits, beta: FieldShare,
           r: FieldShare, *, key_id: int = 0, cheat=None,
           skip_unused_coin: bool = False) -> LeakyDpfOutput:
    """Protocol Pi_DPF (Figure 5) for one party; see the module docstring for the mapping.

    ``chan``   this party's channel endpoint (``chan.party == role``);
    ``sid``, ``key_id``  session identifier and number of this key (never
               reused under the same ``sid``: they fix all labels and the
               domain of H0);
    ``delta``  this party's share Delta_role of the global key (int);
    ``alpha``  [[alpha^(i)]]_role for i < n, index bits most significant first;
    ``beta``   <beta>_role;
    ``r``      <r>_role, a random authenticated mask (step 4) unknown to both.

    Returns ``LeakyDpfOutput``.  Raises ``ProtocolAbort`` if the batch check
    fails or the peer misbehaves; the caller must then retire the session.
    """
    if role not in (0, 1):
        raise ValueError("role must be 0 or 1")
    if chan.party != role:
        raise ValueError("role does not match the channel endpoint")
    n = len(alpha)
    if not 1 <= n <= MAX_N:
        raise ValueError(f"n must be in 1..{MAX_N}")
    delta = _field_elem(delta, "delta")
    hooks = _parse_cheat(cheat, n)
    phase = chan.phase
    try:
        return _pi_dpf(chan, sub_sid(sid, key_id), delta, alpha, beta, r, hooks,
                       bool(skip_unused_coin))
    finally:
        chan.mark(phase)


def _recv_elems(chan, payload: bytes, count: int) -> list[int]:
    """Exchange ``count`` field elements; returns the peer's elements as ints."""
    reply = chan.exchange(payload)
    if len(reply) != _ELEM * count:
        raise ProtocolAbort("Pi_DPF: message has the wrong length")
    return [gf.bytes_to_int(reply[_ELEM * i:_ELEM * (i + 1)]) for i in range(count)]


def _pi_dpf(chan, ssid: bytes, delta: int, alpha: BdozBits, beta: FieldShare, r: FieldShare,
            hooks, skip_unused_coin: bool) -> LeakyDpfOutput:
    n = len(alpha)
    cw_error, final_error, a_error = hooks
    dom = h0_domain(ssid)

    # ---- Step 1: W <- F_coin;  (s || t)^(0,0)_b := Delta_b XOR W ---------------------------
    chan.mark(PHASE_ROOT)
    w = gf.bytes_to_int(hs.coin_toss(chan, ssid, b"W", _ELEM))
    nodes = gf.ints_to_vec([delta ^ w])

    # ---- Step 2: one correction word per level, whole tree expanded locally ----------------
    chan.mark(PHASE_LEVELS)
    # alpha_b^(i) Delta_b XOR K_b[alpha_{1-b}^(i)] XOR M_b[alpha_b^(i)], for all i at once
    alpha_term = gf.bit_mul(alpha.bits, delta) ^ alpha.keys ^ alpha.macs
    cws = []
    for i in range(n):
        hashed = hash0(nodes, dom)                        # H0(s^(i,j) || t^(i,j)) for all j
        cw_own = gf.xor_reduce(hashed) ^ delta ^ gf.arr_to_int(alpha_term[i])  # CW_b^(i)
        if cw_error and cw_error[0] == i:
            cw_own ^= cw_error[1]                         # test hook: additive error E
        cw = cw_own ^ _recv_elems(chan, gf.int_to_bytes(cw_own), 1)[0]  # CW^(i) = CW_0 XOR CW_1
        nodes = expand_level(nodes, hashed, cw)           # level i + 1
        if cw_error and cw_error[0] == i and cw_error[2] is not None:
            # test hook: remove E on the children of the guessed on-path node
            nodes[2 * cw_error[2]:2 * cw_error[2] + 2] ^= gf.int_to_arr(cw_error[1])
        cws.append(cw)

    # ---- Step 3: final correction word, authenticated outputs ------------------------------
    chan.mark(PHASE_LEAF)
    h_val, h_mac = hash1(nodes, dom)                      # H1 = (value half, MAC half)
    cw_val = gf.xor_reduce(h_val) ^ beta.value            # CW_b^(n), first half:  XOR beta_b
    cw_mac = gf.xor_reduce(h_mac) ^ beta.mac              # second half:           XOR M_b[beta]
    if final_error:
        cw_val ^= final_error[0]
        cw_mac ^= final_error[1]
    peer_val, peer_mac = _recv_elems(chan, gf.int_to_bytes(cw_val) + gf.int_to_bytes(cw_mac), 2)
    cw_val ^= peer_val
    cw_mac ^= peer_mac
    t = gf.lsb(nodes)                                     # u_b^(j) = t_b^(n,j)
    t_mask = bit_mask(t)
    u = Shares(t, nodes)                                  # M_b[u^(j)] = (s || t)_b^(n,j)
    v = FieldShares(h_val ^ (gf.int_to_arr(cw_val) & t_mask),   # v_b^(j)
                    h_mac ^ (gf.int_to_arr(cw_mac) & t_mask))   # M_b[v^(j)]
    if final_error and final_error[2] is not None:
        v.values[final_error[2]] ^= gf.int_to_arr(final_error[0])
        v.macs[final_error[2]] ^= gf.int_to_arr(final_error[1])

    # ---- Step 4: chi <- F_coin;  <a> := sum chi^j <u^(j)> + sum chi^(N+j) <v^(j)> + <r> -----
    chan.mark(PHASE_CHECK)
    chi = gf.bytes_to_int(hs.coin_toss(chan, ssid, b"chi", _ELEM))
    u_field = gf.zeros(1 << n)
    u_field[:, 0] = t                                     # the bit u_b^(j) as a field element
    a_val = poly_eval(np.concatenate([u_field, v.values]), chi) ^ r.value
    a_mac = poly_eval(np.concatenate([u.macs, v.macs]), chi) ^ r.mac

    # ---- Step 5: open a, then BatchCheck(<a>, a~) ------------------------------------------
    a_sent = a_val ^ a_error
    a_open = a_sent ^ _recv_elems(chan, gf.int_to_bytes(a_sent), 1)[0]
    batch_check(chan, ssid, b"a", delta, [a_open], [a_mac], skip_unused_coin=skip_unused_coin)

    # ---- Step 6: outputs ------------------------------------------------------------------
    return LeakyDpfOutput(n, u, v, {"W": w, "cws": cws, "cw_final": (cw_val, cw_mac),
                                    "chi": chi, "a": a_open})


# --------------------------------------------------------------------------- #
# Driver on top of the preprocessing
# --------------------------------------------------------------------------- #

def abit_demand(n: int) -> int:
    """Random BDOZ bits of each party consumed by ``run_pi_dpf``: n + 128 inputs and 128 for r."""
    return int(n) + 2 * KAPPA


def run_pi_dpf(chan, prep, sid: bytes, n: int, alpha_share: int, beta_share: int, *,
               key_id: int = 0, cheat=None, skip_unused_coin: bool = False) -> LeakyDpfOutput:
    """Generate one leaky DPF from XOR shares of the index and of the payload.

    ``prep`` is this party's ``RealPrep`` (after ``setup()``) or ``IdealPrep``;
    it supplies the key share and ``random_bdoz``.  The index is
    alpha = alpha_share_0 XOR alpha_share_1 (n bits), the payload
    beta = beta_share_0 XOR beta_share_1 in GF(2^128).  Steps: (1) one request
    for n + 256 random BDOZ bits per party (F_aBit); (2) authentication of the
    n + 128 input bits of each party ("zhang-input"); Pack of the payload bits
    and of the mask bits; (3) ``pi_dpf``.  On any protocol failure the prep
    object is retired (``prep.abort()``) and ``ProtocolAbort`` propagates.
    """
    role = prep.role
    n = int(n)
    if not 1 <= n <= MAX_N:
        raise ValueError(f"n must be in 1..{MAX_N}")
    a_bits = index_bits(alpha_share, n)
    b_bits = gf.unpack_bits(_field_elem(beta_share, "beta_share"))
    _parse_cheat(cheat, n)  # reject malformed hooks before any message is sent
    phase = chan.phase
    try:
        own, keys = prep.random_bdoz(abit_demand(n))
        delta = prep.delta
        chan.mark(PHASE_INPUT)
        n_in = n + KAPPA
        x = authenticate_inputs(chan, delta, own[:n_in], keys[:n_in],
                                np.concatenate([a_bits, b_bits]))
        beta = pack(delta, x[n:])                                           # <beta>
        r = FieldShare(*spdz_shares(role, delta, own[n_in:], keys[n_in:]).to_field())  # <r>
        return pi_dpf(chan, sid, role, delta, x[:n], beta, r, key_id=key_id, cheat=cheat,
                      skip_unused_coin=skip_unused_coin)
    except BaseException:
        prep.abort()
        raise
    finally:
        chan.mark(phase)


def bytes_per_party(n: int, skip_unused_coin: bool = False) -> dict:
    """Exact bytes sent by ONE party in ``run_pi_dpf``, per phase (excluding the prep phases)."""
    coins_check = 1 if skip_unused_coin else 2
    return {
        PHASE_INPUT: (n + KAPPA + 7) // 8,
        PHASE_ROOT: COIN_BYTES,
        PHASE_LEVELS: _ELEM * n,
        PHASE_LEAF: 2 * _ELEM,
        PHASE_CHECK: coins_check * COIN_BYTES + _ELEM + COMMIT_OVERHEAD_BYTES + _ELEM,
    }


def rounds(n: int, skip_unused_coin: bool = False) -> dict:
    """Rounds of ``run_pi_dpf`` per phase (excluding the prep phases)."""
    return {PHASE_INPUT: 1, PHASE_ROOT: 2, PHASE_LEVELS: n, PHASE_LEAF: 1,
            PHASE_CHECK: (5 if skip_unused_coin else 7)}
