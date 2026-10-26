"""Generation circuits of DESIGN.md section 4: session setup, C_dpf, C_dcf,
C_gate and batches of them, plus plaintext reference functions.

Every builder returns a frozen ``couplet.circuit.Circuit`` whose
``meta["descriptor"]`` is a JSON-serialisable description from which
``build(descriptor)`` rebuilds the identical circuit (same digest).

Conventions
-----------
* Bit vectors are little-endian.  An n-bit index ``alpha`` has ``alpha_i``
  (DESIGN.md: MSB first, i = 1..n) at bit ``n - i``.
* ``w = k + s``.  The MAC key is the (s+1)-bit integer
  ``Delta = Delta_0 + Delta_1`` (never reduced).  It enters every key or gate
  circuit as the BOUND input ``delta`` (s+1 wires): the engine binds it to the
  RETAIN output ``delta`` of the session-setup circuit.
* Seeds are 128 bits in the ``couplet.ascon`` layout.
* ``share_out(v, t)`` over m bits: P_0 inputs uniform ``sigma`` and
  ``mu_sigma`` (m bits each), which are its shares; the circuit outputs
  ``share1 = v - sigma`` and ``mu_share1 = t - mu_sigma`` (mod 2^m) privately
  to P_1.
* Adders are ripple-carry by default (fewest AND gates); ``adder="sklansky"``
  trades about 2 to 40 percent more AND gates for a much smaller AND depth.

Label scheme (``<p>`` is the prefix: empty for a single circuit, ``i<j>.``
for item j of a batch; ``b`` in {0, 1} is the owner)
--------------------------------------------------------------------------
Session setup ``build_session_setup(s)``
    inputs   (b, "delta")            s      MAC key share Delta_b
    output   "delta"       RETAIN    s+1    Delta_0 + Delta_1 (s AND gates)

Every other circuit
    input    (BOUND, "delta")        s+1    shared by all items of a batch

Key circuits ``build_dpf`` / ``build_dcf`` (key labels, prefix ``<p>``)
    inputs   (b, "<p>alpha")         n      index share (XOR or additive)
             (b, "<p>beta")          k      payload share, beta_mode "secret" only
             (b, "<p>rho0")          128    root coins, s0 = rho0_0 ^ rho0_1
             (b, "<p>rho1")          128    root coins, s1 = rho1_0 ^ rho1_1
    outputs  "<p>cws"      PUBLIC    128 n  CWs_i at bits [128 (i-1), 128 i)
             "<p>cwtl"     PUBLIC    n      CWtL_i at bit i-1
             "<p>cwtr"     PUBLIC    n      CWtR_i at bit i-1
             "<p>vcw"      PUBLIC    w n    VCW_i at bits [w (i-1), w i)   (DCF)
             "<p>tcw"      PUBLIC    w n    TCW_i likewise                 (DCF)
             "<p>cwout_v"  PUBLIC    w
             "<p>cwout_t"  PUBLIC    w
             "<p>root0"    P_0       128    s0
             "<p>root1"    P_1       128    s1

Gate circuits ``build_gate`` (DESIGN.md 4.3), m = n + s if auth_input else n
    offset   (b, "<p>rp.coin")       m      r' = coin_0 ^ coin_1, r = r' mod 2^n
             (0, "<p>rp.sigma")      m      P_0's share of r' (auth) or of r
             (0, "<p>rp.mu_sigma")   m      P_0's tag share            (auth only)
             "<p>rp.share1"     P_1  m      r' - sigma  (auth) / r - sigma
             "<p>rp.mu_share1"  P_1  m      Delta r' - mu_sigma        (auth only)
    DPF      "<p>dpf.<key labels>"          key at index r, payload 1 (dpf_one);
                                            inputs (b, "<p>dpf.rho0"/"rho1")
    threshold i = 0, 1, ... with (c, payload):
             "<p>th<i>.<key labels>"        DCF key at alpha_c = (r + c) mod 2^n,
                                            payload 1 ("one") or r ("r");
                                            inputs (b, "<p>th<i>.rho0"/"rho1")
             (0, "<p>th<i>.w.sigma"), (0, "<p>th<i>.w.mu_sigma")          w bits
             "<p>th<i>.w.share1", "<p>th<i>.w.mu_share1"       P_1        w bits
                                            [[w_c]], w_c = [r + c >= 2^n],
                                            tag Delta w_c
             (0, "<p>th<i>.rw.sigma"), (0, "<p>th<i>.rw.mu_sigma")        w bits
             "<p>th<i>.rw.share1", "<p>th<i>.rw.mu_share1"     P_1        w bits
                                            [[r w_c]], payload "r" only,
                                            tag Delta r w_c mod 2^w
    output mask j = 0, 1, ...:
             (b, "<p>om<j>.coin")    s      rho = coin_0 ^ coin_1
             (0, "<p>om<j>.sigma"), (0, "<p>om<j>.mu_sigma")              s bits
             "<p>om<j>.share1", "<p>om<j>.mu_share1"           P_1        s bits
                                            share_out of the HIGH s bits: the
                                            SPDZ2k shares over Z_{2^w} are
                                            2^k * (these values); value
                                            2^k rho, tag 2^k (Delta rho mod 2^s)

Reference functions ``expected_outputs_*`` recompute every output of a
circuit from the plaintext inputs with ``couplet.fss`` (explicit roots), so
that tests can compare bit for bit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import fss
from .circuit import BOUND, ONE, RETAIN, ZERO, Builder, Circuit, bits_to_int
from .circuits_ascon import SEED_BITS, conv, convV_L, convV_R, prg_G

DESIGN_VERSION = "3.1"
INDEX_MODES = ("xor", "add")
BETA_MODES = ("secret", "public_one")
PAYLOADS = ("one", "r")
ADDERS = ("ripple", "sklansky")
DELTA = "delta"
_BETA_ALIASES = {"secret": "secret", "public_one": "public_one", "public": "public_one", "one": "public_one"}

Inputs = Mapping[Tuple[int, str], Union[int, Sequence[int]]]


# --------------------------------------------------------------------------- #
# Parameter checks and descriptors
# --------------------------------------------------------------------------- #

def _check_ks(k: int, s: int) -> None:
    if k < 1 or s < 1 or 2 * (k + s) > 320:
        raise ValueError("need k >= 1, s >= 1 and 2(k+s) <= 320")


def _check_key_params(n: int, k: int, s: int, index_mode: str, beta_mode: str) -> str:
    if not 1 <= n <= 64:
        raise ValueError("n must be in 1..64")
    _check_ks(k, s)
    if index_mode not in INDEX_MODES:
        raise ValueError(f"index_mode must be one of {INDEX_MODES}")
    if beta_mode not in _BETA_ALIASES:
        raise ValueError(f"beta_mode must be one of {BETA_MODES}")
    return _BETA_ALIASES[beta_mode]


def _check_adder(adder: str) -> None:
    if adder not in ADDERS:
        raise ValueError(f"adder must be one of {ADDERS}")


def _norm_thresholds(n: int, thresholds) -> List[Tuple[int, str]]:
    res = []
    for item in thresholds:
        c, payload = item
        c = int(c)
        if not 0 <= c <= (1 << n):
            raise ValueError("threshold constants must lie in [0, 2^n]")
        if payload not in PAYLOADS:
            raise ValueError(f"payload must be one of {PAYLOADS}")
        res.append((c, payload))
    return res


def session_setup_descriptor(s: int, adder: str = "ripple") -> Dict:
    return {"kind": "session_setup", "s": int(s), "adder": adder}


def dpf_descriptor(n: int, k: int, s: int, index_mode: str = "xor", beta_mode: str = "secret",
                   adder: str = "ripple") -> Dict:
    beta_mode = _check_key_params(n, k, s, index_mode, beta_mode)
    return {"kind": "dpf", "n": int(n), "k": int(k), "s": int(s), "index_mode": index_mode,
            "beta_mode": beta_mode, "adder": adder}


def dcf_descriptor(n: int, k: int, s: int, index_mode: str = "xor", beta_mode: str = "secret",
                   adder: str = "ripple") -> Dict:
    d = dpf_descriptor(n, k, s, index_mode, beta_mode, adder)
    d["kind"] = "dcf"
    return d


def gate_descriptor(n: int, k: int, s: int, auth_input: bool, dpf_one: bool, thresholds,
                    n_outmask: int, adder: str = "ripple") -> Dict:
    if not 1 <= n <= 64:
        raise ValueError("n must be in 1..64")
    _check_ks(k, s)
    if n > k:
        raise ValueError("gates need n <= k")
    if int(n_outmask) < 0:
        raise ValueError("n_outmask must be >= 0")
    return {"kind": "gate", "n": int(n), "k": int(k), "s": int(s), "auth_input": bool(auth_input),
            "dpf_one": bool(dpf_one), "thresholds": [[c, p] for c, p in _norm_thresholds(n, thresholds)],
            "n_outmask": int(n_outmask), "adder": adder}


def _as_descriptor(item) -> Dict:
    if isinstance(item, Circuit):
        item = item.meta.get("descriptor")
        if item is None:
            raise ValueError("circuit carries no descriptor")
    if not isinstance(item, Mapping) or "kind" not in item:
        raise ValueError("expected a descriptor (dict with a 'kind') or a circuit built by this module")
    return dict(item)


def batch_descriptor(items: Sequence, adder: str = "ripple") -> Dict:
    """Descriptor of a batch: item j keeps its labels under the prefix ``i<j>.``."""
    descs = []
    for item in items:
        d = _as_descriptor(item)
        if d["kind"] not in ("dpf", "dcf", "gate"):
            raise ValueError("a batch contains dpf, dcf and gate items only")
        d = _canonical(d)
        d.pop("adder", None)              # one builder, one adder for the whole batch
        descs.append(d)
    if not descs:
        raise ValueError("empty batch")
    if len({d["s"] for d in descs}) != 1:
        raise ValueError("all items of a batch must use the same s (they share Delta)")
    return {"kind": "batch", "s": descs[0]["s"], "items": descs, "adder": adder}


def _canonical(desc: Mapping) -> Dict:
    """Validated, normalised copy of a descriptor."""
    kind = desc.get("kind")
    adder = desc.get("adder", "ripple")
    _check_adder(adder)
    if kind == "session_setup":
        _check_ks(1, desc["s"])
        return session_setup_descriptor(desc["s"], adder)
    if kind in ("dpf", "dcf"):
        make = dpf_descriptor if kind == "dpf" else dcf_descriptor
        return make(desc["n"], desc["k"], desc["s"], desc.get("index_mode", "xor"),
                    desc.get("beta_mode", "secret"), adder)
    if kind == "gate":
        return gate_descriptor(desc["n"], desc["k"], desc["s"], desc["auth_input"], desc["dpf_one"],
                               desc["thresholds"], desc["n_outmask"], adder)
    if kind == "batch":
        return batch_descriptor(desc["items"], adder)
    raise ValueError(f"unknown descriptor kind {kind!r}")


canonical_descriptor = _canonical


def descriptor_name(desc: Mapping) -> str:
    """Short file-name-safe name of a descriptor."""
    d = _canonical(desc)
    kind = d["kind"]
    tail = "" if d["adder"] == "ripple" else "_" + d["adder"]
    if kind == "session_setup":
        return f"setup_s{d['s']}{tail}"
    if kind in ("dpf", "dcf"):
        beta = "secret" if d["beta_mode"] == "secret" else "one"
        return f"{kind}_n{d['n']}_k{d['k']}_s{d['s']}_{d['index_mode']}_{beta}{tail}"
    if kind == "gate":
        name = f"gate_n{d['n']}_k{d['k']}_s{d['s']}_{'auth' if d['auth_input'] else 'shared'}"
        if d["dpf_one"]:
            name += "_dpf"
        for c, payload in d["thresholds"]:
            name += f"_t{c}{'r' if payload == 'r' else 'o'}"
        if d["n_outmask"]:
            name += f"_om{d['n_outmask']}"
        return name + tail
    names = [descriptor_name(x) for x in d["items"]]
    if len(set(names)) == 1:
        return f"batch{len(names)}x_{names[0]}{tail}"
    digest = hashlib.sha256(json.dumps(d["items"], sort_keys=True).encode()).hexdigest()[:12]
    return f"batch{len(names)}_{digest}{tail}"


_NAME_KEY = re.compile(r"^(dpf|dcf)_n(\d+)_k(\d+)_s(\d+)_(xor|add)_(secret|one)(_sklansky)?$")
_NAME_SETUP = re.compile(r"^setup_s(\d+)(_sklansky)?$")
_NAME_GATE = re.compile(r"^gate_n(\d+)_k(\d+)_s(\d+)_(auth|shared)(_dpf)?((?:_t\d+[or])*)(?:_om(\d+))?(_sklansky)?$")
_NAME_BATCH = re.compile(r"^batch(\d+)x_(.+?)(_sklansky)?$")


def descriptor_from_name(name: str) -> Dict:
    """Inverse of ``descriptor_name`` (mixed batches, named by a hash, excepted)."""
    m = _NAME_SETUP.match(name)
    if m:
        return session_setup_descriptor(int(m.group(1)), "sklansky" if m.group(2) else "ripple")
    m = _NAME_KEY.match(name)
    if m:
        make = dpf_descriptor if m.group(1) == "dpf" else dcf_descriptor
        return make(int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5),
                    "secret" if m.group(6) == "secret" else "public_one",
                    "sklansky" if m.group(7) else "ripple")
    m = _NAME_GATE.match(name)
    if m:
        ths = [(int(c), "r" if p == "r" else "one") for c, p in re.findall(r"_t(\d+)([or])", m.group(6))]
        return gate_descriptor(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) == "auth",
                               bool(m.group(5)), ths, int(m.group(7) or 0),
                               "sklansky" if m.group(8) else "ripple")
    m = _NAME_BATCH.match(name)
    if m:
        item = descriptor_from_name(m.group(2))
        return batch_descriptor([item] * int(m.group(1)), "sklansky" if m.group(3) else "ripple")
    raise ValueError(f"cannot parse the descriptor name {name!r}")


# --------------------------------------------------------------------------- #
# Sub-circuits
# --------------------------------------------------------------------------- #

def _mul(B: Builder, x: Sequence[int], y: Sequence[int], width: int) -> List[int]:
    """x * y mod 2^width; the shorter operand supplies the rows."""
    if len(y) > len(x):
        x, y = y, x
    return B.mul(list(x), list(y), width)


def _share_out(B: Builder, q: str, value: Sequence[int], tag: Optional[Sequence[int]], m: int) -> None:
    """share_out of DESIGN.md section 4 over m bits under the label prefix ``q``."""
    sigma = B.input(0, q + "sigma", m)
    B.output(q + "share1", B.sub_mod(B.zero_extend(value, m), sigma, width=m)[0], to=1)
    if tag is not None:
        mu_sigma = B.input(0, q + "mu_sigma", m)
        B.output(q + "mu_share1", B.sub_mod(B.zero_extend(tag, m), mu_sigma, width=m)[0], to=1)


def _root_inputs(B: Builder, p: str) -> Tuple[List[int], List[int]]:
    """Declare the root coins of one key and return (s0, s1)."""
    rho = {}
    for b in (0, 1):
        rho[b, 0] = B.input(b, p + "rho0", SEED_BITS)
        rho[b, 1] = B.input(b, p + "rho1", SEED_BITS)
    return B.xor_bits(rho[0, 0], rho[1, 0]), B.xor_bits(rho[0, 1], rho[1, 1])


def _dcf_plane(B: Builder, w: int, a: int, t1: int, L0, L1, R0, R1, acc, payload):
    """One DCF level for one plane (value or tag).  Returns (CW, new accumulator).

        X   = x_lose1 - x_lose0 + (a ? payload : 0)
        CW  = neg_if(t1, X - acc)                       (section 3)
        acc = x_keep0 - x_keep1 + X                     (closed form, change Q)
    """
    dL = B.sub_mod(L1, L0, width=w)[0]
    dR = B.sub_mod(R1, R0, width=w)[0]
    d_lose = B.mux(a, dL, dR)                           # a = 1: keep R, lose L
    d_keep = B.xor_bits(B.xor_bits(dL, dR), d_lose)     # the other one, free
    X = B.add_mod(d_lose, B.and_bits(a, payload), width=w)[0]
    cw = B.neg_if(t1, B.sub_mod(X, acc, width=w)[0])
    return cw, B.sub_mod(X, d_keep, width=w)[0]


def _tree(B: Builder, family: str, n: int, k: int, s: int, alpha: Sequence[int],
          beta: Sequence[int], beta_t: Sequence[int], s0: List[int], s1: List[int]) -> Dict:
    """Levels and leaf of DPF.Gen / DCF.Gen (DESIGN.md 4.1 steps 4-5, 4.2).

    ``beta`` and ``beta_t`` are w-wire vectors.  The state is (s0, s1, t)
    with t = t_0 and t_1 = t XOR 1 (on-path invariant).
    """
    w = k + s
    dcf = family == "dcf"
    t = ZERO
    acc_v = [ZERO] * w
    acc_t = [ZERO] * w
    key: Dict[str, list] = {"cws": [], "cwtl": [], "cwtr": [], "vcw": [], "tcw": []}
    for i in range(1, n + 1):
        a = alpha[n - i]
        sL0, tL0, sR0, tR0 = prg_G(B, s0)
        sL1, tL1, sR1, tR1 = prg_G(B, s1)
        keep0 = B.mux(a, sR0, sL0)                      # keep = R iff alpha_i = 1
        keep1 = B.mux(a, sR1, sL1)
        # CWs_i = s_lose0 ^ s_lose1 = (sL0 ^ sR0 ^ sL1 ^ sR1) ^ s_keep0 ^ s_keep1
        cws = B.xor_bits(B.xor_bits(B.xor_bits(sL0, sR0), B.xor_bits(sL1, sR1)),
                         B.xor_bits(keep0, keep1))
        cwtl = B.xor(B.xor(tL0, tL1), B.not_(a))        # tL0 ^ tL1 ^ alpha_i ^ 1
        cwtr = B.xor(B.xor(tR0, tR1), a)                # tR0 ^ tR1 ^ alpha_i
        if dcf:
            t1 = B.not_(t)                              # control of P_1 BEFORE the update
            vL0, uL0 = convV_L(B, s0, k, s)
            vR0, uR0 = convV_R(B, s0, k, s)
            vL1, uL1 = convV_L(B, s1, k, s)
            vR1, uR1 = convV_R(B, s1, k, s)
            vcw, acc_v = _dcf_plane(B, w, a, t1, vL0, vL1, vR0, vR1, acc_v, beta)
            tcw, acc_t = _dcf_plane(B, w, a, t1, uL0, uL1, uR0, uR1, acc_t, beta_t)
            key["vcw"].append(vcw)
            key["tcw"].append(tcw)
        g = B.and_bits(t, cws)
        s0 = B.xor_bits(keep0, g)                       # s_keep0 ^ (t_0 ? CWs : 0)
        s1 = B.xor_bits(B.xor_bits(keep1, g), cws)      # s_keep1 ^ (t_1 ? CWs : 0)
        t_keep0 = B.mux(a, tR0, tL0)
        cwt_keep = B.mux(a, cwtr, cwtl)
        t = B.xor(t_keep0, B.and_(t, cwt_keep))
        key["cws"].append(cws)
        key["cwtl"].append(cwtl)
        key["cwtr"].append(cwtr)
    c0v, c0t = conv(B, s0, k, s)
    c1v, c1t = conv(B, s1, k, s)
    t1 = B.not_(t)
    if dcf:
        key["cwout_v"] = B.neg_if(t1, B.sum_mod([(c1v, False), (c0v, True), (acc_v, True)], w))
        key["cwout_t"] = B.neg_if(t1, B.sum_mod([(c1t, False), (c0t, True), (acc_t, True)], w))
    else:
        key["cwout_v"] = B.neg_if(t1, B.sum_mod([(beta, False), (c0v, True), (c1v, False)], w))
        key["cwout_t"] = B.neg_if(t1, B.sum_mod([(beta_t, False), (c0t, True), (c1t, False)], w))
    return key


def _emit_key(B: Builder, p: str, family: str, n: int, k: int, s: int, alpha: Sequence[int],
              beta: Sequence[int], beta_t: Sequence[int]) -> None:
    """Root coins, tree and outputs of one key under the label prefix ``p``."""
    s0, s1 = _root_inputs(B, p)
    key = _tree(B, family, n, k, s, alpha, beta, beta_t, s0, s1)
    B.output(p + "cws", [x for cw in key["cws"] for x in cw])
    B.output(p + "cwtl", key["cwtl"])
    B.output(p + "cwtr", key["cwtr"])
    if family == "dcf":
        B.output(p + "vcw", [x for cw in key["vcw"] for x in cw])
        B.output(p + "tcw", [x for cw in key["tcw"] for x in cw])
    B.output(p + "cwout_v", key["cwout_v"])
    B.output(p + "cwout_t", key["cwout_t"])
    B.output(p + "root0", s0, to=0)
    B.output(p + "root1", s1, to=1)


def _emit_keygen(B: Builder, delta: Sequence[int], p: str, d: Mapping) -> None:
    """C_dpf / C_dcf (DESIGN.md 4.1, 4.2) under the label prefix ``p``."""
    n, k, s = d["n"], d["k"], d["s"]
    w = k + s
    a0 = B.input(0, p + "alpha", n)
    a1 = B.input(1, p + "alpha", n)
    alpha = B.xor_bits(a0, a1) if d["index_mode"] == "xor" else B.add_mod(a0, a1, width=n)[0]
    if d["beta_mode"] == "secret":
        b0 = B.input(0, p + "beta", k)
        b1 = B.input(1, p + "beta", k)
        beta_k = B.add_mod(b0, b1, width=k)[0]
        beta = B.zero_extend(beta_k, w)
        beta_t = _mul(B, delta, beta_k, w)              # (s+1) x k multiplier, mod 2^w
    else:
        beta = B.const_bits(1, w)
        beta_t = B.zero_extend(delta, w)                # Delta * 1, no gates
    _emit_key(B, p, d["kind"], n, k, s, alpha, beta, beta_t)


def _emit_gate(B: Builder, delta: Sequence[int], p: str, d: Mapping) -> None:
    """C_gate (DESIGN.md 4.3) under the label prefix ``p``."""
    n, k, s = d["n"], d["k"], d["s"]
    w = k + s
    N = 1 << n
    thresholds = [(c, payload) for c, payload in d["thresholds"]]
    one = B.const_bits(1, w)
    delta_w = B.zero_extend(delta, w)

    # 1. offset r' and r = r' mod 2^n
    m = n + s if d["auth_input"] else n
    rp = B.xor_bits(B.input(0, p + "rp.coin", m), B.input(1, p + "rp.coin", m))
    r = rp[:n]
    # 2. [[r']] over Z_{2^(n+s)}, or additive shares of r mod 2^n
    if d["auth_input"]:
        _share_out(B, p + "rp.", rp, _mul(B, delta, rp, m), m)
    else:
        _share_out(B, p + "rp.", r, None, n)
    # 3. DPF at index r, payload 1
    if d["dpf_one"]:
        _emit_key(B, p + "dpf.", "dpf", n, k, s, r, one, delta_w)
    # 4. thresholds
    r_w = B.zero_extend(r, w)
    r_t = _mul(B, delta, r, w) if any(payload == "r" for _, payload in thresholds) else None
    shifted: Dict[int, Tuple[List[int], int, List[int]]] = {}   # c -> (alpha_c, w_c, w_c * Delta)
    for i, (c, payload) in enumerate(thresholds):
        q = f"{p}th{i}."
        if c not in shifted:
            if c == 0:                                  # alpha_c = r, w_c = 0
                alpha_c, w_c = list(r), ZERO
            elif c == N:                                # alpha_c = r, w_c = 1
                alpha_c, w_c = list(r), ONE
            else:                                       # n-bit adder with carry out
                alpha_c, w_c = B.add_mod(r, B.const_bits(c, n), width=n)
            shifted[c] = (alpha_c, w_c, B.and_bits(w_c, delta_w))
        alpha_c, w_c, w_t = shifted[c]
        if payload == "one":
            _emit_key(B, q, "dcf", n, k, s, alpha_c, one, delta_w)
        else:
            _emit_key(B, q, "dcf", n, k, s, alpha_c, r_w, r_t)
        _share_out(B, q + "w.", [w_c], w_t, w)          # [[w_c]], tag w_c ? Delta : 0
        if payload == "r":                              # [[r w_c]], tag w_c ? Delta r : 0
            _share_out(B, q + "rw.", B.and_bits(w_c, r_w), B.and_bits(w_c, r_t), w)
    # 5. output masks: value 2^k rho, tag 2^k (Delta rho mod 2^s); share the high s bits
    for j in range(d["n_outmask"]):
        q = f"{p}om{j}."
        rho = B.xor_bits(B.input(0, q + "coin", s), B.input(1, q + "coin", s))
        _share_out(B, q, rho, _mul(B, delta[:s], rho, s), s)


# Extension point for the Couplet layer (``couplet.circuits_ppml``): the sub-circuits of a
# gate under their public names.  Nothing else of this module changes.
emit_key = _emit_key          # root coins, tree and outputs of one key under a label prefix
share_out = _share_out        # share_out of DESIGN.md section 4 over m bits
mul_mod = _mul                # x * y mod 2^width (the shorter operand supplies the rows)


# --------------------------------------------------------------------------- #
# Public builders
# --------------------------------------------------------------------------- #

def _finish(B: Builder, desc: Dict) -> Circuit:
    C = B.build()
    C.meta = {"descriptor": desc, "name": descriptor_name(desc), "design_version": DESIGN_VERSION}
    return C


def build_session_setup(s: int, *, adder: str = "ripple") -> Circuit:
    """Session setup (DESIGN.md section 4, change H): P_b inputs Delta_b
    (s bits); the RETAIN output ``delta`` is the (s+1)-bit sum, not reduced."""
    desc = _canonical(session_setup_descriptor(s, adder))
    B = Builder(adder=adder)
    d0 = B.input(0, DELTA, s)
    d1 = B.input(1, DELTA, s)
    total, carry = B.add_mod(d0, d1, width=s)
    B.output(DELTA, total + [carry], to=RETAIN)
    return _finish(B, desc)


def build_dpf(n: int, k: int, s: int, index_mode: str = "xor", beta_mode: str = "secret", *,
              adder: str = "ripple") -> Circuit:
    """C_dpf(n, k, s, index_mode, beta_mode) of DESIGN.md 4.1."""
    return build(dpf_descriptor(n, k, s, index_mode, beta_mode, adder))


def build_dcf(n: int, k: int, s: int, index_mode: str = "xor", beta_mode: str = "secret", *,
              adder: str = "ripple") -> Circuit:
    """C_dcf(n, k, s, index_mode, beta_mode) of DESIGN.md 4.2."""
    return build(dcf_descriptor(n, k, s, index_mode, beta_mode, adder))


def build_gate(n: int, k: int, s: int, auth_input: bool, dpf_one: bool,
               thresholds: Sequence[Tuple[int, str]], n_outmask: int, *, adder: str = "ripple") -> Circuit:
    """C_gate(n, k, s, descriptor) of DESIGN.md 4.3.

    ``thresholds`` is a list of ``(c, payload)`` with ``0 <= c <= 2^n`` and
    ``payload`` in {"one", "r"}; ``n_outmask`` output masks are appended."""
    return build(gate_descriptor(n, k, s, auth_input, dpf_one, thresholds, n_outmask, adder))


def build_batch(items: Sequence, *, adder: str = "ripple") -> Circuit:
    """Several key / gate circuits in one circuit execution.

    ``items`` are descriptors (``dpf_descriptor``, ``dcf_descriptor``,
    ``gate_descriptor``) or circuits built by this module.  Item j keeps its
    labels under the prefix ``i<j>.``; all items share the one BOUND input
    ``delta``."""
    return build(batch_descriptor(items, adder))


def build(descriptor: Mapping) -> Circuit:
    """Build the circuit of a descriptor (deterministic: same descriptor, same digest)."""
    desc = _canonical(descriptor)
    kind = desc["kind"]
    if kind == "session_setup":
        return build_session_setup(desc["s"], adder=desc["adder"])
    B = Builder(adder=desc["adder"])
    delta = B.input(BOUND, DELTA, desc["s"] + 1)
    if kind == "batch":
        for j, item in enumerate(desc["items"]):
            (_emit_gate if item["kind"] == "gate" else _emit_keygen)(B, delta, f"i{j}.", item)
    elif kind == "gate":
        _emit_gate(B, delta, "", desc)
    else:
        _emit_keygen(B, delta, "", desc)
    return _finish(B, desc)


# --------------------------------------------------------------------------- #
# Plaintext reference (expected outputs via couplet.fss)
# --------------------------------------------------------------------------- #

def _int(value) -> int:
    return value if isinstance(value, int) else bits_to_int(value)


def _get(inputs: Inputs, party: int, label: str) -> int:
    return _int(inputs[(party, label)])


def pack_key_outputs(key0, key1, prefix: str = "") -> Dict[str, int]:
    """Circuit output values (label -> int) that correspond to an fss key pair."""
    n, w = key0.n, key0.width
    out = {
        prefix + "cws": sum(int.from_bytes(cw, "little") << (SEED_BITS * i) for i, cw in enumerate(key0.cw_s)),
        prefix + "cwtl": sum(bit << i for i, bit in enumerate(key0.cw_tl)),
        prefix + "cwtr": sum(bit << i for i, bit in enumerate(key0.cw_tr)),
    }
    if key0.family == fss.FAMILY_DCF:
        out[prefix + "vcw"] = sum(v << (w * i) for i, v in enumerate(key0.cw_v))
        out[prefix + "tcw"] = sum(v << (w * i) for i, v in enumerate(key0.cw_t))
    out[prefix + "cwout_v"] = key0.cw_out_v
    out[prefix + "cwout_t"] = key0.cw_out_t
    out[prefix + "root0"] = fss.seed_to_int(key0.seed)
    out[prefix + "root1"] = fss.seed_to_int(key1.seed)
    assert len(key0.cw_s) == n
    return out


def unpack_key_outputs(outputs: Mapping[str, Union[int, Sequence[int]]], family: str, n: int, k: int,
                       s: int, prefix: str = ""):
    """(key_0, key_1) as ``fss.DPFKey``/``fss.DCFKey`` from circuit outputs
    (label -> int or bit list).  Reference helper for tests; the protocol
    layer assembles each party's key from the outputs that party receives."""
    w = k + s
    g = lambda lab: _int(outputs[prefix + lab])  # noqa: E731
    m128, mw = (1 << SEED_BITS) - 1, (1 << w) - 1
    cws, cwtl, cwtr = g("cws"), g("cwtl"), g("cwtr")
    common = dict(n=n, k=k, s=s,
                  cw_s=tuple(fss.seed_from_int((cws >> (SEED_BITS * i)) & m128) for i in range(n)),
                  cw_tl=tuple((cwtl >> i) & 1 for i in range(n)),
                  cw_tr=tuple((cwtr >> i) & 1 for i in range(n)),
                  cw_out_v=g("cwout_v"), cw_out_t=g("cwout_t"))
    if family == "dcf":
        vcw, tcw = g("vcw"), g("tcw")
        common["cw_v"] = tuple((vcw >> (w * i)) & mw for i in range(n))
        common["cw_t"] = tuple((tcw >> (w * i)) & mw for i in range(n))
        cls = fss.DCFKey
    elif family == "dpf":
        cls = fss.DPFKey
    else:
        raise ValueError("family must be 'dpf' or 'dcf'")
    return (cls(b=0, seed=fss.seed_from_int(g("root0")), **common),
            cls(b=1, seed=fss.seed_from_int(g("root1")), **common))


def _plain_roots(inputs: Inputs, p: str) -> Tuple[bytes, bytes]:
    s0 = _get(inputs, 0, p + "rho0") ^ _get(inputs, 1, p + "rho0")
    s1 = _get(inputs, 0, p + "rho1") ^ _get(inputs, 1, p + "rho1")
    return fss.seed_from_int(s0), fss.seed_from_int(s1)


def _plain_key(family: str, alpha: int, beta: int, delta: int, n: int, k: int, s: int,
               inputs: Inputs, p: str) -> Dict[str, int]:
    gen = fss.dpf_gen if family == "dpf" else fss.dcf_gen
    k0, k1 = gen(alpha, beta, delta, n, k, s, roots=_plain_roots(inputs, p))
    return pack_key_outputs(k0, k1, p)


def keygen_plaintext(n: int, k: int, s: int, index_mode: str, beta_mode: str, inputs: Inputs,
                     prefix: str = "") -> Dict[str, int]:
    """alpha and beta that a key circuit derives from the parties' inputs."""
    beta_mode = _check_key_params(n, k, s, index_mode, beta_mode)
    a0, a1 = _get(inputs, 0, prefix + "alpha"), _get(inputs, 1, prefix + "alpha")
    alpha = a0 ^ a1 if index_mode == "xor" else (a0 + a1) % (1 << n)
    beta = 1
    if beta_mode == "secret":
        beta = (_get(inputs, 0, prefix + "beta") + _get(inputs, 1, prefix + "beta")) % (1 << k)
    return {"alpha": alpha, "beta": beta}


def gate_plaintext(n: int, k: int, s: int, auth_input: bool, dpf_one: bool, thresholds,
                   n_outmask: int, inputs: Inputs, prefix: str = "") -> Dict:
    """The secret values a gate circuit derives from the coins: ``rp`` (r'),
    ``r``, and per threshold ``alpha`` = (r + c) mod 2^n and ``w`` = [r + c >= 2^n],
    per output mask ``rho``."""
    N = 1 << n
    rp = _get(inputs, 0, prefix + "rp.coin") ^ _get(inputs, 1, prefix + "rp.coin")
    r = rp % N
    ths = _norm_thresholds(n, thresholds)
    return {"rp": rp, "r": r,
            "alpha": [(r + c) % N for c, _ in ths],
            "w": [int(r + c >= N) for c, _ in ths],
            "rho": [_get(inputs, 0, f"{prefix}om{j}.coin") ^ _get(inputs, 1, f"{prefix}om{j}.coin")
                    for j in range(n_outmask)]}


def expected_outputs_session_setup(s: int, inputs: Inputs) -> Dict[str, int]:
    """{"delta": Delta_0 + Delta_1} (an (s+1)-bit integer, not reduced)."""
    return {DELTA: _get(inputs, 0, DELTA) + _get(inputs, 1, DELTA)}


def _expected_keygen(family: str, n: int, k: int, s: int, index_mode: str, beta_mode: str,
                     inputs: Inputs, prefix: str) -> Dict[str, int]:
    pt = keygen_plaintext(n, k, s, index_mode, beta_mode, inputs, prefix)
    return _plain_key(family, pt["alpha"], pt["beta"], _get(inputs, BOUND, DELTA), n, k, s, inputs, prefix)


def expected_outputs_dpf(n: int, k: int, s: int, index_mode: str, beta_mode: str, inputs: Inputs,
                         prefix: str = "") -> Dict[str, int]:
    """Every output of ``build_dpf`` recomputed with ``fss.dpf_gen``; ``inputs``
    is the mapping given to ``evaluate`` (including ``(BOUND, "delta")``)."""
    return _expected_keygen("dpf", n, k, s, index_mode, beta_mode, inputs, prefix)


def expected_outputs_dcf(n: int, k: int, s: int, index_mode: str, beta_mode: str, inputs: Inputs,
                         prefix: str = "") -> Dict[str, int]:
    """Every output of ``build_dcf`` recomputed with ``fss.dcf_gen``."""
    return _expected_keygen("dcf", n, k, s, index_mode, beta_mode, inputs, prefix)


def expected_outputs_gate(n: int, k: int, s: int, auth_input: bool, dpf_one: bool, thresholds,
                          n_outmask: int, inputs: Inputs, prefix: str = "") -> Dict[str, int]:
    """Every output of ``build_gate`` recomputed in plaintext (keys via ``fss``)."""
    p = prefix
    w = k + s
    mod_w, mod_s = 1 << w, 1 << s
    delta = _get(inputs, BOUND, DELTA)
    ths = _norm_thresholds(n, thresholds)
    pt = gate_plaintext(n, k, s, auth_input, dpf_one, ths, n_outmask, inputs, p)
    rp, r = pt["rp"], pt["r"]
    out: Dict[str, int] = {}
    if auth_input:
        mod_m = 1 << (n + s)
        out[p + "rp.share1"] = (rp - _get(inputs, 0, p + "rp.sigma")) % mod_m
        out[p + "rp.mu_share1"] = (delta * rp - _get(inputs, 0, p + "rp.mu_sigma")) % mod_m
    else:
        out[p + "rp.share1"] = (r - _get(inputs, 0, p + "rp.sigma")) % (1 << n)
    if dpf_one:
        out.update(_plain_key("dpf", r, 1, delta, n, k, s, inputs, p + "dpf."))
    for i, (c, payload) in enumerate(ths):
        q = f"{p}th{i}."
        alpha_c, w_c = pt["alpha"][i], pt["w"][i]
        out.update(_plain_key("dcf", alpha_c, 1 if payload == "one" else r, delta, n, k, s, inputs, q))
        out[q + "w.share1"] = (w_c - _get(inputs, 0, q + "w.sigma")) % mod_w
        out[q + "w.mu_share1"] = (delta * w_c - _get(inputs, 0, q + "w.mu_sigma")) % mod_w
        if payload == "r":
            out[q + "rw.share1"] = (r * w_c - _get(inputs, 0, q + "rw.sigma")) % mod_w
            out[q + "rw.mu_share1"] = (delta * r * w_c - _get(inputs, 0, q + "rw.mu_sigma")) % mod_w
    for j, rho in enumerate(pt["rho"]):
        q = f"{p}om{j}."
        out[q + "share1"] = (rho - _get(inputs, 0, q + "sigma")) % mod_s
        out[q + "mu_share1"] = (delta * rho % mod_s - _get(inputs, 0, q + "mu_sigma")) % mod_s
    return out


def expected_outputs_batch(items: Sequence, inputs: Inputs) -> Dict[str, int]:
    """Every output of ``build_batch(items)`` recomputed in plaintext."""
    out: Dict[str, int] = {}
    for j, item in enumerate(batch_descriptor(items)["items"]):
        out.update(_expected_item(item, inputs, f"i{j}."))
    return out


def _expected_item(d: Mapping, inputs: Inputs, prefix: str) -> Dict[str, int]:
    if d["kind"] == "gate":
        return expected_outputs_gate(d["n"], d["k"], d["s"], d["auth_input"], d["dpf_one"],
                                     d["thresholds"], d["n_outmask"], inputs, prefix)
    return _expected_keygen(d["kind"], d["n"], d["k"], d["s"], d["index_mode"], d["beta_mode"],
                            inputs, prefix)


def expected_outputs(descriptor: Mapping, inputs: Inputs) -> Dict[str, int]:
    """Dispatch on the descriptor kind; returns {output label: int}."""
    d = _canonical(descriptor)
    if d["kind"] == "session_setup":
        return expected_outputs_session_setup(d["s"], inputs)
    if d["kind"] == "batch":
        return expected_outputs_batch(d["items"], inputs)
    return _expected_item(d, inputs, "")


def random_inputs(circuit: Circuit, rng, delta: Optional[int] = None) -> Dict[Tuple[int, str], int]:
    """Uniform plaintext inputs for every declaration of ``circuit``.

    ``rng`` offers ``getrandbits`` (``random.Random``).  The BOUND input
    ``delta`` is set to ``delta`` or, if None, to the unreduced sum of two
    fresh uniform s-bit shares."""
    inputs: Dict[Tuple[int, str], int] = {}
    for owner, label, wires in circuit.inputs:
        if owner == BOUND:
            if label != DELTA:
                raise ValueError(f"unexpected BOUND input {label!r}")
            s = len(wires) - 1
            inputs[(owner, label)] = (rng.getrandbits(s) + rng.getrandbits(s)) if delta is None else delta
        else:
            inputs[(owner, label)] = rng.getrandbits(len(wires))
    return inputs


# --------------------------------------------------------------------------- #
# Gate presets and the statistics table (experiments/figdata/circuit_stats.json)
# --------------------------------------------------------------------------- #

GATE_PRESETS = ("table", "table_auth", "comparison", "comparison_auth", "interval", "interval_auth",
                "relu_large", "offset_shared_only", "offset_auth_only", "outmask_only")


def gate_preset(name: str, n: int, k: int = 64, s: int = 64, adder: str = "ripple") -> Dict:
    """Descriptors of the catalogue gates (DESIGN.md section 6) used in the tables.

    ``table`` (also zero test / equality): one payload-1 DPF, shared input.
    ``comparison``: payload-1 DCFs at a generic c = floor(2^n / 3) and at 0.
    ``interval``: payload-1 DCFs at q+1 = floor(2^n / 3) and p = floor(2^n / 5).
    ``relu_large``: payload-1 and payload-r DCFs at 2^n / 2 and 0, authenticated
    input, one output mask.  ``*_auth`` use an authenticated input wire.
    ``offset_*_only`` / ``outmask_only`` contain no key (cost of the sharings).
    """
    N = 1 << n
    c = N // 3
    table = {
        "table": (False, True, [], 0),
        "table_auth": (True, True, [], 0),
        "comparison": (False, False, [(c, "one"), (0, "one")], 0),
        "comparison_auth": (True, False, [(c, "one"), (0, "one")], 0),
        "interval": (False, False, [(c, "one"), (N // 5, "one")], 0),
        "interval_auth": (True, False, [(c, "one"), (N // 5, "one")], 0),
        "relu_large": (True, False, [(N // 2, "one"), (0, "one"), (N // 2, "r"), (0, "r")], 1),
        "offset_shared_only": (False, False, [], 0),
        "offset_auth_only": (True, False, [], 0),
        "outmask_only": (False, False, [], 1),
    }
    if name not in table:
        raise ValueError(f"unknown gate preset {name!r}; known: {GATE_PRESETS}")
    auth, dpf_one, ths, om = table[name]
    return gate_descriptor(n, k, s, auth, dpf_one, ths, om, adder)


_STAT_KEYS = ("n_and", "n_xor", "n_not", "n_gates", "and_depth", "n_wires", "n_inputs",
              "n_input_bits_p0", "n_input_bits_p1", "n_bound_input_bits", "n_outputs",
              "n_public_output_bits", "n_private_output_bits_p0", "n_private_output_bits_p1",
              "n_retain_output_bits")


def stats_row(descriptor: Mapping, **extra) -> Dict:
    """Build a circuit and return its statistics row (with digest and build time)."""
    t = time.perf_counter()
    C = build(descriptor)
    st = C.stats()
    row = {"name": C.meta["name"], **extra, **{key: st[key] for key in _STAT_KEYS},
           "digest": C.digest(), "build_seconds": round(time.perf_counter() - t, 2),
           "descriptor": C.meta["descriptor"]}
    return row


def _per_level(rows: Sequence[Dict]) -> Dict:
    """Affine fit AND(n) = fixed + per_level * n (and the same for depth) from
    the two smallest n, checked against every other n."""
    rows = sorted(rows, key=lambda r_: r_["n"])
    a, b = rows[0], rows[1]
    dn = b["n"] - a["n"]
    res = {}
    for key in ("n_and", "and_depth"):
        slope, rem = divmod(b[key] - a[key], dn)
        fixed = a[key] - slope * a["n"]
        exact = rem == 0 and all(r_[key] == fixed + slope * r_["n"] for r_ in rows)
        res[key] = {"per_level": slope, "fixed": fixed, "exact_for_all_n": exact}
    return res


def circuit_stats(ns_keys: Sequence[int] = (8, 12, 16, 20, 24, 32),
                  ks: Sequence[Tuple[int, int]] = ((64, 64), (32, 64)),
                  ns_gates: Sequence[int] = (8, 12, 16),
                  adders: Sequence[str] = ("ripple", "sklansky"),
                  verbose: bool = True) -> Dict:
    """All rows of experiments/figdata/circuit_stats.json."""
    from .circuits_ascon import p12_circuit_stats, prg_call_stats

    def say(row: Dict) -> None:
        if verbose:
            print(f"  {row['name']:<58} AND={row['n_and']:>9,} depth={row['and_depth']:>5} "
                  f"XOR={row['n_xor']:>10,} in={row['n_inputs']:>5} out={row['n_outputs']:>6} "
                  f"({row['build_seconds']}s)", flush=True)

    doc: Dict = {
        "generated_by": "python -m couplet.circuits_gen",
        "design_version": DESIGN_VERSION,
        "note": "All figures are the output of the circuit compiler (dead gates removed). "
                "adder = ripple is the default configuration; sklansky rows show the "
                "AND-count / AND-depth trade-off. index_mode = xor everywhere.",
        "primitives": {"p12_on_320_fresh_wires": p12_circuit_stats(),
                       "prg_call_secret_seed_constant_iv": prg_call_stats()},
        "session_setup": [], "keys": [], "per_level": [], "gates": [], "gate_increments": [],
    }
    for adder in adders:
        for s in sorted({s for _, s in ks}):
            row = stats_row(session_setup_descriptor(s, adder), kind="session_setup", s=s, adder=adder)
            doc["session_setup"].append(row)
            say(row)
    for adder in adders:
        for k, s in ks:
            for family in ("dpf", "dcf"):
                for beta_mode in BETA_MODES:
                    group = []
                    for n in ns_keys:
                        make = dpf_descriptor if family == "dpf" else dcf_descriptor
                        row = stats_row(make(n, k, s, "xor", beta_mode, adder), kind=family, n=n, k=k,
                                        s=s, index_mode="xor", beta_mode=beta_mode, adder=adder)
                        doc["keys"].append(row)
                        group.append(row)
                        say(row)
                    if len(group) >= 2:
                        fit = _per_level(group)
                        doc["per_level"].append({"kind": family, "k": k, "s": s, "beta_mode": beta_mode,
                                                 "adder": adder, **fit})
    for adder in adders:
        for k, s in ks:
            for n in ns_gates:
                rows = {}
                for preset in GATE_PRESETS:
                    row = stats_row(gate_preset(preset, n, k, s, adder), kind="gate", gate=preset,
                                    n=n, k=k, s=s, adder=adder)
                    rows[preset] = row
                    doc["gates"].append(row)
                    say(row)
                base = rows["offset_shared_only"]
                for what, preset in (("one_output_mask", "outmask_only"),
                                     ("authenticated_offset_instead_of_shared", "offset_auth_only")):
                    doc["gate_increments"].append({
                        "what": what, "n": n, "k": k, "s": s, "adder": adder,
                        "n_and": rows[preset]["n_and"] - base["n_and"],
                        "n_xor": rows[preset]["n_xor"] - base["n_xor"],
                        "and_depth_of_part": rows[preset]["and_depth"],
                        "computed_as": f"{rows[preset]['name']} minus {base['name']}"})
                doc["gate_increments"].append({
                    "what": "authenticated_offset_alone", "n": n, "k": k, "s": s, "adder": adder,
                    "n_and": rows["offset_auth_only"]["n_and"], "n_xor": rows["offset_auth_only"]["n_xor"],
                    "and_depth_of_part": rows["offset_auth_only"]["and_depth"],
                    "computed_as": rows["offset_auth_only"]["name"]})
                doc["gate_increments"].append({
                    "what": "shared_offset_alone", "n": n, "k": k, "s": s, "adder": adder,
                    "n_and": base["n_and"], "n_xor": base["n_xor"],
                    "and_depth_of_part": base["and_depth"], "computed_as": base["name"]})
    return doc


def default_stats_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "experiments", "figdata", "circuit_stats.json"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="write the circuit statistics table (AND/XOR/depth/inputs/outputs)")
    ap.add_argument("--out", default=None, help="JSON path (default: experiments/figdata/circuit_stats.json)")
    ap.add_argument("--ns", default="8,12,16,20,24,32", help="n values for the key circuits")
    ap.add_argument("--gate-ns", default="8,12,16", help="n values for the gate circuits")
    ap.add_argument("--adders", default="ripple,sklansky")
    args = ap.parse_args(argv)
    t = time.perf_counter()
    doc = circuit_stats(ns_keys=[int(x) for x in args.ns.split(",")],
                        ns_gates=[int(x) for x in args.gate_ns.split(",")],
                        adders=args.adders.split(","))
    doc["total_build_seconds"] = round(time.perf_counter() - t, 1)
    out = args.out or default_stats_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("per-level AND gates (fixed + per_level * n):")
    for row in doc["per_level"]:
        a, dp = row["n_and"], row["and_depth"]
        print(f"  {row['kind']} k={row['k']} s={row['s']} beta={row['beta_mode']:<10} adder={row['adder']:<8} "
              f"AND = {a['fixed']:>6} + {a['per_level']:>6} n  (exact: {a['exact_for_all_n']});  "
              f"depth = {dp['fixed']:>4} + {dp['per_level']} n  (exact: {dp['exact_for_all_n']})")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
