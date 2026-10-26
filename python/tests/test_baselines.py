"""Tests for the evaluation baselines (couplet.baselines; DESIGN.md section 9, Q1 / Q2).

(a) building blocks   polynomial evaluation, tree expansion and H1 against the
                      scalar hash, input authentication, BatchCheck (Figure 2
                      of Zhang et al.), the unchecked COT extension.
(b) leaky protocol    Pi_DPF of Zhang et al. with ideal preprocessing for
                      n in {1, 2, 4, 8, 10}: the outputs reconstruct to
                      unit(N, alpha) and unit(N, alpha) * beta, the MAC
                      invariants hold, bytes and rounds per phase are exact.
(c) real aBits        the same on the real preprocessing (base OT, extension)
                      for n in {4, 8, 10} in one session.
(d) selective failure an additive error on one level's correction-word share,
                      removed on a guessed path: over 200 runs the batch check
                      passes iff the most significant index bit equals the
                      guess (the one-bit leakage of F_DPF); further deviations.
(e) honest runs       never abort (every honest execution of this file is
                      counted, plus 100 random ones).
(f) semi-honest       Doerner-shelat-style baseline for n in {1, 4, 8, 10}:
                      outputs, exact bytes and rounds; a deviation goes
                      unnoticed (there is no check).
(g) cost models       schema of every entry, arithmetic identities, agreement
                      with circuit_stats.json and with the implementations.
(h) key separation    both parties' trees are recomputed from their key shares
                      and the public transcript: the hash domain of every key
                      generation is the one of ITS (sid, key number), in both
                      baselines; the opened value of the leaky protocol is
                      masked by a fresh uniform r (checked in (b), (c), (e));
                      BatchCheck uses the powers of chi (a pair of errors that
                      cancels under repeated coefficients is rejected, (a)).

The checks of (h), the mask checks and the pair check were added after a
mutation run in which four single-point mutants survived the earlier version
of this file.

Two parties are run with ``couplet.net.run_two_party`` (two threads of this
process).  Run as a script from artifact/python:

    python tests/test_baselines.py

Prints the number of checks and exits non-zero on any failure.  The test
inputs (indices, payloads, guesses, injected errors) are drawn from a seeded
generator; ``COUPLET_TEST_SEED=<int>`` selects another sample.
"""

from __future__ import annotations

import os
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import gf128 as gf  # noqa: E402
from couplet import hashes as hs  # noqa: E402
from couplet.baselines import costmodels as cm  # noqa: E402
from couplet.baselines import leaky_zhang as lz  # noqa: E402
from couplet.baselines import semihonest_ds as sh  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.prep import IdealDealer, IdealPrep, RealPrep  # noqa: E402
from couplet.prep.cotext import CHECK_BYTES, SEED_BYTES, u_message_bytes  # noqa: E402
from couplet.prep.shares import Bdoz, Keys, bit_mask  # noqa: E402

CHECKS = 0
FAILURES: list[str] = []
HONEST_RUNS = 0
HONEST_ABORTS = 0
MASKS: list[int] = []          # the masks r of all honest executions (reconstructed, test only)
SID = b"test-baselines"
KAPPA = 128
SEED = int(os.environ.get("COUPLET_TEST_SEED", "20261010"))
RNG = random.Random(SEED)  # test inputs only; all protocol randomness comes from the OS


def check(cond, what: str) -> None:
    """Count one check; record (and print) it if it fails."""
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(what)
        print("FAIL:", what)


def raises(exc_type, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception:
        return False
    return False


def _finish(start_failures: int) -> None:
    """Make the enclosing test function fail under pytest if it recorded a failure."""
    assert len(FAILURES) == start_failures, FAILURES[start_failures:]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def rand_elem() -> int:
    return RNG.getrandbits(128)


def rand_vec(n: int) -> np.ndarray:
    return gf.bytes_to_vec(RNG.randbytes(16 * n))


def unit_vector(n: int, alpha: int) -> np.ndarray:
    unit = np.zeros(1 << n, dtype=np.uint8)
    unit[alpha] = 1
    return unit


def leaky_ok(out0, out1, n: int, alpha: int, beta: int, delta: int) -> tuple[bool, bool, bool, bool]:
    """(u = unit, MACs of u, v = unit * beta, MACs of v) for the two parties' outputs."""
    unit = unit_vector(n, alpha)
    return (np.array_equal(out0.u.bits ^ out1.u.bits, unit),
            np.array_equal(out0.u.macs ^ out1.u.macs, gf.bit_mul(unit, delta)),
            np.array_equal(out0.v.values ^ out1.v.values, gf.bit_mul(unit, beta)),
            np.array_equal(out0.v.macs ^ out1.v.macs, gf.bit_mul(unit, gf.mul(beta, delta))))


def opened_mask(out0, out1) -> int:
    """The mask r of step 4: the opened a XOR the unmasked combination of the outputs.

    a = sum_j chi^j u^(j) + sum_j chi^(N+j) v^(j) + r, so r is recovered from
    the public (chi, a) and the reconstructed outputs (which no party sees).
    If a were not masked it would reveal alpha and beta: a = chi^alpha + chi^(N+alpha) beta.
    """
    u = out0.u.bits ^ out1.u.bits
    u_field = gf.zeros(u.shape[0])
    u_field[:, 0] = u
    combination = lz.poly_eval(np.concatenate([u_field, out0.v.values ^ out1.v.values]),
                               out0.transcript["chi"])
    return out0.transcript["a"] ^ combination


def check_masked(out0, out1, tag: str) -> None:
    mask = opened_mask(out0, out1)
    MASKS.append(mask)
    check(mask != 0, tag + ": the opened value a is masked (a differs from the unmasked combination)")


def recompute_leaves(delta_b: int, transcript: dict, dom: int) -> np.ndarray:
    """The leaf strings (s || t)^(n, j) of one party from its key share and the public transcript."""
    nodes = gf.ints_to_vec([delta_b ^ transcript["W"]])
    for cw in transcript["cws"]:
        nodes = lz.expand_level(nodes, lz.hash0(nodes, dom), cw)
    return nodes


def leaky_run(n: int, alpha: int, beta: int, cheat=(None, None), skip=False, key_id: int = 0):
    """One Pi_DPF execution on ideal preprocessing; returns (out0, out1, stats0, stats1, Delta).

    The index and the payload are XOR-shared at random.  ``ProtocolAbort``
    propagates; honest executions are tallied for test (e).
    """
    global HONEST_RUNS, HONEST_ABORTS
    dealer = IdealDealer()
    a0, b0 = RNG.getrandbits(n), rand_elem()
    shares = ((a0, b0), (alpha ^ a0, beta ^ b0))
    preps = [IdealPrep(0, dealer), IdealPrep(1, dealer)]

    def party(role):
        def fn(chan):
            return lz.run_pi_dpf(chan, preps[role], SID, n, shares[role][0], shares[role][1],
                                 key_id=key_id, cheat=cheat[role], skip_unused_coin=skip)
        return fn

    honest = cheat[0] is None and cheat[1] is None
    HONEST_RUNS += honest
    try:
        out0, out1, st0, st1 = run_two_party(party(0), party(1))
    except ProtocolAbort:
        HONEST_ABORTS += honest
        raise
    return out0, out1, st0, st1, dealer.delta


def attack_run(n: int, alpha: int, beta: int, cheater: int, hook: dict):
    """Run with a cheating party; returns (passed, outputs correct and authenticated, message)."""
    cheat = [None, None]
    cheat[cheater] = hook
    try:
        out0, out1, _, _, delta = leaky_run(n, alpha, beta, tuple(cheat))
    except ProtocolAbort as exc:
        return False, False, str(exc)
    return True, all(leaky_ok(out0, out1, n, alpha, beta, delta)), ""


def phase_bytes(stats: dict, names) -> dict:
    return {p: stats["phases"][p]["bytes_sent"] for p in names}


def phase_rounds(stats: dict, names) -> dict:
    return {p: stats["phases"][p]["rounds"] for p in names}


class TamperChan:
    """Channel wrapper that modifies the payload of one ``exchange`` call (tests only)."""

    def __init__(self, chan, index: int, fn) -> None:
        self._chan = chan
        self._index = index
        self._fn = fn
        self._count = 0

    def exchange(self, data: bytes) -> bytes:
        if self._count == self._index:
            data = self._fn(bytes(data))
        self._count += 1
        return self._chan.exchange(data)

    def __getattr__(self, name):
        return getattr(self._chan, name)


# --------------------------------------------------------------------------- #
# (a) Building blocks
# --------------------------------------------------------------------------- #

def test_building_blocks() -> None:
    start = len(FAILURES)

    # poly_eval against the definition
    for m in (0, 1, 2, 3, 5, 48, 49, 64, 100, 257):
        coeffs = rand_vec(m)
        x = rand_elem()
        acc, power = 0, 1
        for c in gf.vec_to_ints(coeffs):
            acc ^= gf.mul(power, c)
            power = gf.mul(power, x)
        check(lz.poly_eval(coeffs, x) == acc, f"poly_eval, {m} coefficients")
    coeffs = rand_vec(8)
    check(lz.poly_eval(coeffs, 0) == gf.arr_to_int(coeffs[0]), "poly_eval at 0 is the constant term")
    check(lz.poly_eval(coeffs, 1) == gf.xor_reduce(coeffs), "poly_eval at 1 is the sum")

    # index bits: most significant first
    check(lz.index_bits(0b1011, 4).tolist() == [1, 0, 1, 1], "index_bits is MSB first")
    check(lz.index_bits(1, 3).tolist() == [0, 0, 1], "index_bits pads on the left")
    check(raises(ValueError, lz.index_bits, 8, 3), "index_bits rejects an index >= 2^n")
    check(raises(ValueError, lz.index_bits, -1, 3), "index_bits rejects a negative index")

    # H0 / H1 / expansion against the scalar hash
    ssid = lz.sub_sid(SID, 3)
    dom = lz.h0_domain(ssid)
    check(dom >> 56 == 0x48, "H0 domain is in the CCR family (top byte 0x48)")
    check(lz.h0_domain(lz.sub_sid(SID, 4)) != dom, "H0 domain depends on the key identifier")
    check(lz.h0_domain(ssid, sh.LABEL_H0) != dom, "the two baselines use different H0 domains")
    nodes = rand_vec(9)
    ints = gf.vec_to_ints(nodes)
    hashed = lz.hash0(nodes, dom)
    check(gf.vec_to_ints(hashed) == [hs.H_ccr_scalar(x, 0, dom) for x in ints],
          "hash0 is H_ccr with tweak 0")
    h_val, h_mac = lz.hash1(nodes, dom)
    check(gf.vec_to_ints(h_val) == [hs.H_ccr_scalar(x, 0, dom) for x in ints]
          and gf.vec_to_ints(h_mac) == [hs.H_ccr_scalar(x ^ 1, 0, dom) for x in ints],
          "hash1(x) = H0(x) || H0(x XOR 1)")
    cw = rand_elem()
    children = gf.vec_to_ints(lz.expand_level(nodes, hashed, cw))
    expect = []
    for x in ints:
        left = hs.H_ccr_scalar(x, 0, dom) ^ (cw if x & 1 else 0)
        expect += [left, left ^ x]
    check(children == expect, "expand_level: left = H0(x) + t CW, right = H0(x) + x + t CW")

    # the level invariant of Figure 5 on a hand-made node pair
    delta0, delta1 = rand_elem() | 1, rand_elem() & ~1
    delta = delta0 ^ delta1
    for bit in (0, 1):
        x0 = rand_elem()
        x1 = x0 ^ delta
        pair = [gf.ints_to_vec([x0]), gf.ints_to_vec([x1])]
        hh = [lz.hash0(p, dom) for p in pair]
        cw = gf.arr_to_int(hh[0][0]) ^ gf.arr_to_int(hh[1][0]) ^ ((bit ^ 1) * delta)
        kids = [gf.vec_to_ints(lz.expand_level(pair[b], hh[b], cw)) for b in (0, 1)]
        diff = [kids[0][c] ^ kids[1][c] for c in (0, 1)]
        check(diff[bit] == delta and diff[1 - bit] == 0,
              f"level invariant: the child on side alpha^(i) = {bit} differs by Delta, the other by 0")

    # Convert, Pack and the input authentication on ideal BDOZ bits
    dealer = IdealDealer()
    d0, d1 = dealer.deltas
    n_bits = 128 + 11
    bits = [np.frombuffer(RNG.randbytes(n_bits), dtype=np.uint8) & 1 for _ in (0, 1)]

    def auth(role):
        def fn(chan):
            own, keys = IdealPrep(role, dealer).random_bdoz(n_bits)
            return lz.authenticate_inputs(chan, dealer.deltas[role], own, keys, bits[role])
        return fn

    x0, x1, st0, st1 = run_two_party(auth(0), auth(1))
    check(np.array_equal(x0.bits, bits[0]) and np.array_equal(x1.bits, bits[1]),
          "authenticate_inputs keeps the input bits as shares")
    check(np.array_equal(x0.macs, x1.keys ^ gf.bit_mul(x0.bits, d1))
          and np.array_equal(x1.macs, x0.keys ^ gf.bit_mul(x1.bits, d0)),
          "authenticate_inputs: M_b[x_b] = K_{1-b}[x_b] + x_b Delta_{1-b}")
    check(st0["bytes_sent"] == st1["bytes_sent"] == (n_bits + 7) // 8 and st0["rounds"] == 1,
          "authenticate_inputs: one bit per input bit, one round")
    c0, c1 = lz.convert(d0, x0), lz.convert(d1, x1)
    x = bits[0] ^ bits[1]
    check(np.array_equal(c0.bits ^ c1.bits, x)
          and np.array_equal(c0.macs ^ c1.macs, gf.bit_mul(x, d0 ^ d1)),
          "Convert: SPDZ-style shares of x_0 + x_1 with MAC shares of x Delta")
    p0, p1 = lz.pack(d0, x0[:128]), lz.pack(d1, x1[:128])
    packed = gf.pack_bits(x[:128])
    check(p0.value ^ p1.value == packed and p0.mac ^ p1.mac == gf.mul(packed, d0 ^ d1),
          "Pack: <sum_i X^i x_i> with a valid MAC")
    check(len(x0[3]) == 1 and len(x0[5:9]) == 4, "BdozBits slicing")
    check(raises(ValueError, lz.BdozBits, bits[0], x0.keys[:5], x0.macs), "BdozBits length check")

    # BatchCheck (Figure 2)
    def bc_run(count, tamper=None, skip=False):
        values = [rand_elem() for _ in range(count)]
        mac0 = [rand_elem() for _ in range(count)]
        mac1 = [m ^ gf.mul(v, d0 ^ d1) for m, v in zip(mac0, values)]
        opened = [list(values), list(values)]
        macs = [mac0, mac1]
        if tamper == "value":       # both parties hold the same wrong opened value
            opened[0][-1] ^= 1
            opened[1][-1] ^= 1
        elif tamper == "mac":
            macs[1][0] ^= 1 << 77
        elif tamper == "view":      # the parties disagree on the opened value
            opened[1][0] ^= 2
        elif tamper == "pair":      # the same error on values 1 and 2 (cancels unless chi^1 != chi^2)
            for view in opened:
                view[1] ^= 0xE44
                view[2] ^= 0xE44
        elif tamper == "pair-last":  # the same error on the last two values
            for view in opened:
                view[-1] ^= 1 << 100
                view[-2] ^= 1 << 100

        def party(role):
            def fn(chan):
                lz.batch_check(chan, b"bc-sid", b"t", dealer.deltas[role], opened[role],
                               macs[role], skip_unused_coin=skip)
                return True
            return fn
        return run_two_party(party(0), party(1))

    for count in (1, 2, 3, 5):
        r0, r1, st0, st1 = bc_run(count)
        check(r0 and r1, f"BatchCheck passes on {count} honest opening(s)")
        check(st0["bytes_sent"] == 96 + 80 and st0["rounds"] == 4,
              f"BatchCheck on {count} value(s): 176 bytes per party, 4 rounds")
        for tamper in ("value", "mac", "view") + (("pair", "pair-last") if count >= 3 else ()):
            check(raises(ProtocolAbort, bc_run, count, tamper),
                  f"BatchCheck aborts ({count} value(s), tampered {tamper})")
    r0, r1, st0, st1 = bc_run(1, skip=True)
    check(r0 and r1 and st0["bytes_sent"] == 80 and st0["rounds"] == 2,
          "BatchCheck, single value without the unused coin: 80 bytes, 2 rounds")
    check(raises(ProtocolAbort, bc_run, 1, "value", True),
          "BatchCheck without the coin still aborts on a wrong opening")
    r0, r1, st0, st1 = bc_run(3, skip=True)
    check(st0["rounds"] == 4, "the coin is skipped only for a single value")
    check(raises(ValueError, lz.batch_check, None, b"s", b"l", d0, [], []),
          "BatchCheck needs at least one value")

    # Unchecked COT extension (semi-honest baseline) on trusted base-OT output
    def trusted(delta):
        s0, s1 = rand_vec(KAPPA), rand_vec(KAPPA)
        sel = gf.unpack_bits(delta).astype(bool)[:, None]
        return np.where(sel, s1, s0), (s0, s1)

    chosen0, pairs0 = trusted(d0)   # direction 0: P_0 sends, P_1 holds the pairs
    chosen1, pairs1 = trusted(d1)   # direction 1: P_1 sends, P_0 holds the pairs
    cots = [sh.PlainCot(0, d0, chosen0, *pairs1), sh.PlainCot(1, d1, chosen1, *pairs0)]
    previous = None
    for ell in (1, 20, 64, 65, 200):
        u = [np.frombuffer(RNG.randbytes(ell), dtype=np.uint8) & 1 for _ in (0, 1)]
        msgs = [cots[b].start(u[b]) for b in (0, 1)]
        check(len(msgs[0]) == sh.cot_message_bytes(ell) == 1024 * ((ell + 63) // 64),
              f"plain COT, {ell} rows: U message is 16 bytes per 64-row-rounded row")
        (m0, k0), (m1, k1) = cots[0].finish(msgs[1]), cots[1].finish(msgs[0])
        check(np.array_equal(m0, k1 ^ gf.bit_mul(u[0], d1))
              and np.array_equal(m1, k0 ^ gf.bit_mul(u[1], d0)),
              f"plain COT, {ell} rows: M = K + u Delta in both directions")
        if previous is not None:
            check(not np.array_equal(previous, k0[:1]), "plain COT: every call reads a fresh stream")
        previous = k0[:1].copy()
    check(raises(RuntimeError, cots[0].finish, b""), "plain COT: finish without start is refused")
    cots[0].start([1, 0, 1])
    check(raises(ProtocolAbort, cots[0].finish, b"\x00" * 7), "plain COT: malformed U aborts")
    _finish(start)


# --------------------------------------------------------------------------- #
# (b) Leaky protocol on ideal preprocessing
# --------------------------------------------------------------------------- #

def test_leaky_correctness() -> None:
    start = len(FAILURES)
    for n in (1, 2, 4, 8, 10):
        size = 1 << n
        alphas = sorted({0, size - 1, RNG.randrange(size), RNG.randrange(size)})
        for alpha in alphas:
            for beta in (rand_elem(), 0, 1):
                out0, out1, st0, st1, delta = leaky_run(n, alpha, beta)
                ok = leaky_ok(out0, out1, n, alpha, beta, delta)
                tag = f"leaky n = {n}, alpha = {alpha}, beta = {beta:#x}"[:70]
                check(ok[0], tag + ": u_0 + u_1 = unit(N, alpha)")
                check(ok[1], tag + ": M_0[u] + M_1[u] = u Delta")
                check(ok[2], tag + ": v_0 + v_1 = unit(N, alpha) * beta")
                check(ok[3], tag + ": M_0[v] + M_1[v] = v Delta")
                check(out0.transcript == out1.transcript, tag + ": both parties see one transcript")
                check_masked(out0, out1, tag)
        # accounting of the last run
        exp_bytes, exp_rounds = lz.bytes_per_party(n), lz.rounds(n)
        for st in (st0, st1):
            check(phase_bytes(st, lz.PHASES) == exp_bytes,
                  f"leaky n = {n}: bytes per phase are exact (party {st['party']})")
            check(phase_rounds(st, lz.PHASES) == exp_rounds,
                  f"leaky n = {n}: rounds per phase are exact (party {st['party']})")
        check(st0["bytes_sent"] == 16 * n + (n + 135) // 8 + 416 == sum(exp_bytes.values()),
              f"leaky n = {n}: 16 n + ceil((n + 128) / 8) + 416 bytes per party")
        check(st0["rounds"] == n + 11 == st1["rounds"], f"leaky n = {n}: n + 11 rounds")
        check(len(out0.transcript["cws"]) == n and len(out0.u) == size == len(out0.v),
              f"leaky n = {n}: n correction words, N outputs")
        check(out0.u.bits.dtype == np.uint8 and out0.v.values.shape == (size, 2),
              f"leaky n = {n}: output formats")

        # variant without the unused BatchCheck coin
        alpha, beta = RNG.randrange(size), rand_elem()
        out0, out1, st0, st1, delta = leaky_run(n, alpha, beta, skip=True)
        check(all(leaky_ok(out0, out1, n, alpha, beta, delta)),
              f"leaky n = {n}, no unused coin: outputs correct and authenticated")
        check(phase_bytes(st0, lz.PHASES) == lz.bytes_per_party(n, True)
              and st0["bytes_sent"] == 16 * n + (n + 135) // 8 + 320,
              f"leaky n = {n}, no unused coin: 96 bytes per party less")
        check(st0["rounds"] == n + 9, f"leaky n = {n}, no unused coin: n + 9 rounds")

    # shares of u are ordinary authenticated bits of our layer: XOR of two outputs stays valid
    n = 5
    out_a = leaky_run(n, 3, rand_elem())
    check(len(out_a[0].u[2:7]) == 5 and (out_a[0].u ^ out_a[0].u).bits.sum() == 0,
          "u is a couplet.prep.shares.Shares object (slicing, XOR)")

    # two keys under one identifier but different key numbers use different coins and domains
    o1 = leaky_run(4, 5, 7, key_id=0)
    o2 = leaky_run(4, 5, 7, key_id=1)
    check(o1[0].transcript["W"] != o2[0].transcript["W"], "fresh public randomness W per key")

    # argument validation happens before any message
    dealer = IdealDealer()
    prep = IdealPrep(0, dealer)

    class _NoChan:
        party = 0
        phase = "x"

        def mark(self, _phase):
            pass

    check(raises(ValueError, lz.run_pi_dpf, _NoChan(), prep, SID, 0, 0, 0), "n = 0 is rejected")
    check(raises(ValueError, lz.run_pi_dpf, _NoChan(), prep, SID, 3, 8, 0),
          "an index share >= 2^n is rejected")
    check(raises(ValueError, lz.run_pi_dpf, _NoChan(), prep, SID, 3, 1, 1 << 128),
          "a payload share outside GF(2^128) is rejected")
    check(raises(ValueError, lz.run_pi_dpf, _NoChan(), prep, SID, 3, 1, 1,
                 cheat={"cw_error": (3, 1, None)}), "a malformed test hook is rejected")
    check(not prep.aborted, "invalid arguments do not retire the session")
    _finish(start)


# --------------------------------------------------------------------------- #
# (c) Leaky protocol on the real preprocessing
# --------------------------------------------------------------------------- #

def _real_session(body, sid=SID):
    """Run ``body(role, chan, prep)`` for both parties on a fresh RealPrep session."""
    def party(role):
        def fn(chan):
            prep = RealPrep(role, chan, sid)
            prep.setup()
            return body(role, chan, prep), prep
        return fn
    (r0, prep0), (r1, prep1), st0, st1 = run_two_party(party(0), party(1))
    return r0, r1, st0, st1, prep0, prep1


def test_leaky_real_prep() -> None:
    global HONEST_RUNS
    start = len(FAILURES)
    ns = (4, 8, 10)
    cases = [(n, RNG.randrange(1 << n), rand_elem()) for n in ns]
    masks = [(RNG.getrandbits(n), rand_elem()) for n in ns]

    def body(role, chan, prep):
        outs = []
        for key_id, ((n, alpha, beta), (a0, b0)) in enumerate(zip(cases, masks)):
            a, b = (a0, b0) if role == 0 else (alpha ^ a0, beta ^ b0)
            outs.append(lz.run_pi_dpf(chan, prep, SID, n, a, b, key_id=key_id))
        return outs, prep.delta

    t0 = time.perf_counter()
    (outs0, delta0), (outs1, delta1), st0, st1, prep0, prep1 = _real_session(body)
    wall = time.perf_counter() - t0
    HONEST_RUNS += len(ns)
    delta = delta0 ^ delta1
    check(delta & 1 == 1, "real prep: lsb(Delta_0 + Delta_1) = 1")
    for (n, alpha, beta), o0, o1 in zip(cases, outs0, outs1):
        ok = leaky_ok(o0, o1, n, alpha, beta, delta)
        check(ok[0] and ok[2], f"real prep, n = {n}: outputs reconstruct to unit and unit * beta")
        check(ok[1] and ok[3], f"real prep, n = {n}: MAC invariants of u and v")
        check_masked(o0, o1, f"real prep, n = {n}")
    exp = {p: sum(lz.bytes_per_party(n)[p] for n in ns) for p in lz.PHASES}
    exp["baseot"] = 12288
    exp["cotext"] = sum(u_message_bytes(lz.abit_demand(n)) + SEED_BYTES + CHECK_BYTES for n in ns)
    exp["abit"] = 0
    for st in (st0, st1):
        got = phase_bytes(st, exp)
        check(got == exp, f"real prep: bytes per phase are exact (party {st['party']}: {got})")
        check(sum(got.values()) == st["bytes_sent"], "real prep: the phases sum to the total")
    exp_rounds = 2 + sum(3 + sum(lz.rounds(n).values()) for n in ns)
    check(st0["rounds"] == st1["rounds"] == exp_rounds,
          f"real prep: {exp_rounds} rounds (2 base OT, 3 per extension call, n + 11 per key)")
    check(not prep0.aborted and not prep1.aborted, "real prep: the session is still alive")
    print(f"  real preprocessing, keys n = {ns}: {wall:.2f} s for both parties, "
          f"{st0['bytes_sent'] + st1['bytes_sent']} bytes")
    _finish(start)


# --------------------------------------------------------------------------- #
# (d) Selective failure
# --------------------------------------------------------------------------- #

def test_selective_failure() -> None:
    start = len(FAILURES)
    n = 6

    # Main experiment: 200 runs, error on the share of CW^(1), removed on the children of the
    # node the cheater guesses to be on the path, i.e. on a guess of the top bit of alpha.
    runs = 200
    passed = consistent = correct_when_passed = 0
    by_cheater = [0, 0]
    for it in range(runs):
        alpha, beta = RNG.randrange(1 << n), rand_elem()
        cheater, guess = it & 1, RNG.getrandbits(1)
        ok, good, msg = attack_run(n, alpha, beta, cheater,
                                   {"cw_error": (1, rand_elem() | 1, guess)})
        top_bit = alpha >> (n - 1)
        passed += ok
        by_cheater[cheater] += ok
        consistent += (ok == (top_bit == guess))
        correct_when_passed += (ok and good)
        check(ok == (top_bit == guess),
              f"selective failure run {it}: check passes iff msb(alpha) = guess")
        check(ok or msg == "BatchCheck failed", f"selective failure run {it}: abort is the batch check")
    check(consistent == runs, f"selective failure: predicate holds in all {runs} runs")
    check(correct_when_passed == passed,
          "selective failure: whenever the check passes the outputs are correct and authenticated")
    check(70 <= passed <= 130, f"selective failure: about half of the runs pass ({passed} of {runs})")
    print(f"  selective failure (error on CW^(1), guess of the top index bit): {passed} of {runs} "
          f"runs passed ({by_cheater[0]} with P_0 cheating, {by_cheater[1]} with P_1 cheating); "
          f"pass iff msb(alpha) = guess in {consistent} of {runs} runs")

    # Deeper level: the predicate is "the first `level` bits of alpha equal the guess".
    level, deep_runs, deep_pass = 3, 64, 0
    for it in range(deep_runs):
        alpha, beta = RNG.randrange(1 << n), rand_elem()
        prefix = alpha >> (n - level)
        guess = prefix if it % 4 == 0 else RNG.randrange(1 << level)
        ok, good, _ = attack_run(n, alpha, beta, it & 1, {"cw_error": (level, rand_elem(), guess)})
        deep_pass += ok
        check(ok == (guess == prefix) and (good or not ok),
              f"error on CW^({level}), run {it}: passes iff the {level}-bit prefix is guessed")
    print(f"  error on CW^({level}) with a prefix guess: {deep_pass} of {deep_runs} runs passed")

    # Level 0: the root is on the path for certain; removing the error needs no guess, the
    # execution is then an honest one and nothing is learned.
    for it in range(10):
        alpha, beta = RNG.randrange(1 << n), rand_elem()
        ok, good, _ = attack_run(n, alpha, beta, it & 1, {"cw_error": (0, rand_elem(), 0)})
        check(ok and good, f"error on CW^(0) removed at the root, run {it}: passes, outputs correct")

    # An error that is not removed makes the check fail whatever alpha is.
    for it in range(20):
        alpha, beta = RNG.randrange(1 << n), rand_elem()
        level = RNG.randrange(n)
        ok, _, msg = attack_run(n, alpha, beta, it & 1, {"cw_error": (level, rand_elem() | 1, None)})
        check(not ok and msg == "BatchCheck failed",
              f"error on CW^({level}) not removed, run {it}: abort")

    # Final correction word: the guess is the whole index.
    for it in range(12):
        alpha, beta = RNG.randrange(1 << n), rand_elem()
        right = it % 3 == 0
        guess = alpha if right else (alpha + 1 + RNG.randrange((1 << n) - 1)) % (1 << n)
        hook = {"final_cw_error": (rand_elem(), rand_elem(), guess)}
        ok, good, _ = attack_run(n, alpha, beta, it & 1, hook)
        check(ok == right and (good or not ok),
              f"error on CW^(n), run {it}: passes iff alpha itself is guessed")
    for it in range(4):
        hook = {"final_cw_error": (rand_elem() if it < 2 else 0, rand_elem(), None)}
        ok, _, _ = attack_run(n, RNG.randrange(1 << n), rand_elem(), it & 1, hook)
        check(not ok, f"error on CW^(n) not removed, run {it}: abort")

    # A wrong share of the opened value a.
    for it in range(4):
        ok, _, msg = attack_run(n, RNG.randrange(1 << n), rand_elem(), it & 1,
                                {"a_error": 1 << RNG.randrange(128)})
        check(not ok and msg == "BatchCheck failed", f"wrong share of a, run {it}: abort")

    # The same attack end to end on the real preprocessing; an abort retires the session.
    for it, right in enumerate((True, False, False, True)):
        n_real = 5
        alpha, beta = RNG.randrange(1 << n_real), rand_elem()
        top = alpha >> (n_real - 1)
        cheater = it & 1
        guess = top if right else top ^ 1
        a0, b0 = RNG.getrandbits(n_real), rand_elem()
        holder = {}

        def body(role, chan, prep, _a=(a0, alpha ^ a0), _b=(b0, beta ^ b0), _c=cheater, _g=guess):
            holder[role] = prep
            cheat = {"cw_error": (1, 0xABCDEF, _g)} if role == _c else None
            return lz.run_pi_dpf(chan, prep, SID, n_real, _a[role], _b[role], cheat=cheat), prep.delta

        try:
            (o0, d0), (o1, d1), _, _, _, _ = _real_session(body, sid=SID + bytes([it]))
            ok = True
            good = all(leaky_ok(o0, o1, n_real, alpha, beta, d0 ^ d1))
        except ProtocolAbort:
            ok, good = False, False
        check(ok == right, f"real prep, attack run {it}: passes iff the top bit is guessed")
        if right:
            check(good, f"real prep, attack run {it}: outputs correct after a right guess")
        else:
            honest = holder[1 - cheater]
            check(honest.aborted and honest.delta is None,
                  f"real prep, attack run {it}: the honest party retired the session")
            check(raises(ProtocolAbort, honest.random_bits, 1),
                  f"real prep, attack run {it}: the retired session refuses further requests")
    _finish(start)


# --------------------------------------------------------------------------- #
# (e) Honest executions never abort
# --------------------------------------------------------------------------- #

def test_honest_never_aborts() -> None:
    start = len(FAILURES)
    aborts = 0
    runs = 100
    for it in range(runs):
        n = RNG.randrange(1, 11)
        alpha, beta = RNG.randrange(1 << n), rand_elem()
        try:
            out0, out1, _, _, delta = leaky_run(n, alpha, beta, skip=bool(it & 1), key_id=it)
            good = all(leaky_ok(out0, out1, n, alpha, beta, delta))
            MASKS.append(opened_mask(out0, out1))
        except ProtocolAbort:
            aborts += 1
            good = False
        check(good, f"honest run {it} (n = {n}): no abort, outputs correct and authenticated")
    check(aborts == 0, f"{runs} random honest executions: no abort")
    check(HONEST_ABORTS == 0 and HONEST_RUNS >= runs,
          f"no honest execution of this test file aborted ({HONEST_RUNS} so far)")
    # the mask r of the opened value: non-zero, fresh in every execution, bits balanced
    # (mean weight of a uniform 128-bit string is 64 with standard deviation 5.66 / sqrt(#masks))
    weight = sum(bin(m).count("1") for m in MASKS) / max(len(MASKS), 1)
    check(len(MASKS) >= runs and all(MASKS), f"the opened value was masked in all {len(MASKS)} honest executions")
    check(len(set(MASKS)) == len(MASKS), "every execution uses a fresh mask r")
    check(abs(weight - 64) < 4, f"the masks look uniform (mean weight {weight:.2f} of 128 bits)")
    print(f"  honest executions of the leaky protocol so far: {HONEST_RUNS}, aborts: {HONEST_ABORTS}")
    _finish(start)


# --------------------------------------------------------------------------- #
# (f) Semi-honest baseline
# --------------------------------------------------------------------------- #

def _semihonest_session(cases, tamper=None, sid=b"test-semihonest"):
    """Run one session with the keys ``cases`` = [(n, alpha, beta)]; returns outputs and stats."""
    masks = [(RNG.getrandbits(n), rand_elem()) for n, _, _ in cases]

    def party(role):
        def fn(chan):
            if tamper is not None and tamper[0] == role:
                chan = TamperChan(chan, tamper[1], tamper[2])
            session = sh.SemiHonestSession(role, chan, sid)
            session.setup()
            outs = []
            for (n, alpha, beta), (a0, b0) in zip(cases, masks):
                a, b = (a0, b0) if role == 0 else (alpha ^ a0, beta ^ b0)
                outs.append(session.dpf_gen(n, a, b))
            return outs, session.delta
        return fn

    (o0, d0), (o1, d1), st0, st1 = run_two_party(party(0), party(1))
    return o0, o1, st0, st1, d0 ^ d1


def test_semihonest() -> None:
    start = len(FAILURES)
    ns = (1, 4, 8, 10, 10)
    cases = [(n, RNG.randrange(1 << n), rand_elem()) for n in ns]
    cases[3] = (10, 0, 0)                     # zero payload, first leaf
    cases[4] = (10, (1 << 10) - 1, 1)         # last leaf
    t0 = time.perf_counter()
    outs0, outs1, st0, st1, delta = _semihonest_session(cases)
    wall = time.perf_counter() - t0
    check(delta & 1 == 1, "semi-honest: lsb(Delta_0 + Delta_1) = 1")
    for (n, alpha, beta), o0, o1 in zip(cases, outs0, outs1):
        unit = unit_vector(n, alpha)
        check(np.array_equal(o0.t ^ o1.t, unit), f"semi-honest n = {n}: t_0 + t_1 = unit(N, alpha)")
        check(np.array_equal(o0.v ^ o1.v, gf.bit_mul(unit, beta)),
              f"semi-honest n = {n}: v_0 + v_1 = unit(N, alpha) * beta")
        check(o0.transcript == o1.transcript and len(o0.transcript["cws"]) == n,
              f"semi-honest n = {n}: one public transcript with n correction words")
        check(o0.v.shape == (1 << n, 2) and o0.t.dtype == np.uint8,
              f"semi-honest n = {n}: output formats")
    check(outs0[3].transcript["W"] != outs0[4].transcript["W"], "semi-honest: fresh W per key")
    exp = {p: sum(sh.bytes_per_party(n)[p] for n in ns) for p in sh.PHASES}
    exp["baseot"] = 12288
    exp_rounds = {p: sum(sh.rounds(n)[p] for n in ns) for p in sh.PHASES}
    exp_rounds["baseot"] = 2
    for st in (st0, st1):
        check(phase_bytes(st, exp) == exp,
              f"semi-honest: bytes per phase are exact (party {st['party']})")
        check(phase_rounds(st, exp_rounds) == exp_rounds,
              f"semi-honest: rounds per phase are exact (party {st['party']})")
        check(sum(exp.values()) == st["bytes_sent"], "semi-honest: the phases sum to the total")
    for n in (1, 8, 20, 26):
        per_party = sum(sh.bytes_per_party(n).values())
        check(per_party == 16 * n + 1024 + 32 and sum(sh.rounds(n).values()) == n + 2,
              f"semi-honest n = {n}: 16 n + 1056 bytes per party and n + 2 rounds per key")
    print(f"  semi-honest session, keys n = {ns}: {wall:.2f} s for both parties "
          f"(base OT included), {st0['bytes_sent'] + st1['bytes_sent']} bytes")

    # No check: a party that sends a wrong correction word is not detected; the output is wrong.
    n, alpha, beta = 6, RNG.randrange(64), rand_elem()

    def flip(data: bytes) -> bytes:
        return bytes([data[0] ^ 1]) + data[1:]

    # Exchanges of P_1: 2 (base OT), 1 (COT flight), then one per level, then the leaf word.
    # P_0 uses the word it received only at its nodes with control bit 1, so a tampered word
    # of level i changes nothing with probability 2^-(2^i) (no such node; 1/16 for level 2,
    # which made an earlier version of this check fail once in 16 runs).  The last level
    # (32 nodes) and the leaf word (64 leaves) are tampered here: 2^-32 and 2^-64.
    unit = unit_vector(n, alpha)
    for what, index in (("last-level", 2 + 1 + (n - 1)), ("final", 2 + 1 + n)):
        outs0, outs1, _, _, _ = _semihonest_session([(n, alpha, beta)], tamper=(1, index, flip),
                                                    sid=b"test-semihonest-tamper-" + what.encode())
        wrong = (not np.array_equal(outs0[0].t ^ outs1[0].t, unit)
                 or not np.array_equal(outs0[0].v ^ outs1[0].v, gf.bit_mul(unit, beta)))
        check(wrong, f"semi-honest: a tampered {what} correction word yields wrong shares and no abort")

    # argument validation
    cot = sh.PlainCot(0, 1, rand_vec(KAPPA), rand_vec(KAPPA), rand_vec(KAPPA))

    class _Chan:
        party = 0
        phase = "x"

        def mark(self, _phase):
            pass

    check(raises(ValueError, sh.semihonest_dpf, _Chan(), SID, 0, 0, cot, 0, 0, 0), "n = 0 rejected")
    check(raises(ValueError, sh.semihonest_dpf, _Chan(), SID, 0, 0, cot, 3, 9, 0),
          "index share out of range rejected")
    check(raises(ValueError, sh.semihonest_dpf, _Chan(), SID, 0, 1, cot, 3, 1, 0),
          "role mismatch rejected")
    session = sh.SemiHonestSession(0, _Chan(), SID)
    check(raises(ValueError, session.dpf_gen, 3, 1, 1), "dpf_gen before setup is refused")
    check(session.delta & 1 == 1 and sh.SemiHonestSession(1, type("C", (), {"party": 1})(),
                                                         SID).delta & 1 == 0,
          "semi-honest key shares: lsb 1 for P_0, 0 for P_1")
    _finish(start)


# --------------------------------------------------------------------------- #
# (h) Separation of the keys of a session
# --------------------------------------------------------------------------- #

def test_key_separation() -> None:
    """Each key generation hashes under the domain of ITS (sid, key number).

    The tree of each party is a function of its key share, the public
    transcript (W and the correction words) and the hash domain.  Both trees
    are recomputed here with the domain the specification prescribes and
    compared with the protocol outputs; a second domain is used as a negative
    control.  A protocol that ran two keys of a session under one domain
    (same labels, same H0) fails the comparison.
    """
    start = len(FAILURES)
    n = 5
    sid = b"test-baselines-separation"

    # leaky protocol: M_b[u^(j)] is the leaf string itself
    for key_id in (0, 1, 5):
        dealer = IdealDealer()
        alpha, beta = RNG.randrange(1 << n), rand_elem()
        a0, b0 = RNG.getrandbits(n), rand_elem()
        shares = ((a0, b0), (alpha ^ a0, beta ^ b0))

        def party(role, key_id=key_id, dealer=dealer, shares=shares):
            def fn(chan):
                return lz.run_pi_dpf(chan, IdealPrep(role, dealer), sid, n, shares[role][0],
                                     shares[role][1], key_id=key_id)
            return fn

        out0, out1, _, _ = run_two_party(party(0), party(1))
        dom = lz.h0_domain(lz.sub_sid(sid, key_id))
        other = lz.h0_domain(lz.sub_sid(sid, key_id + 1))
        for b, out in enumerate((out0, out1)):
            leaves = recompute_leaves(dealer.deltas[b], out.transcript, dom)
            check(np.array_equal(out.u.macs, leaves) and np.array_equal(out.u.bits, gf.lsb(leaves)),
                  f"leaky, key number {key_id}, P_{b}: the tree is the one of H0 under (sid, {key_id})")
            h_val, h_mac = lz.hash1(leaves, dom)
            cw_val, cw_mac = out.transcript["cw_final"]
            t_mask = bit_mask(gf.lsb(leaves))
            check(np.array_equal(out.v.values, h_val ^ (gf.int_to_arr(cw_val) & t_mask))
                  and np.array_equal(out.v.macs, h_mac ^ (gf.int_to_arr(cw_mac) & t_mask)),
                  f"leaky, key number {key_id}, P_{b}: v and its MAC shares are H1 of the leaves")
            check(not np.array_equal(out.u.macs, recompute_leaves(dealer.deltas[b], out.transcript, other)),
                  f"leaky, key number {key_id}, P_{b}: another key number gives another tree (control)")
        check(all(leaky_ok(out0, out1, n, alpha, beta, dealer.delta)),
              f"leaky, key number {key_id}: outputs correct and authenticated")

    # semi-honest baseline: three keys in one session, key numbers 0, 1, 2
    cases = [(n, RNG.randrange(1 << n), rand_elem()) for _ in range(3)]
    masks = [(RNG.getrandbits(n), rand_elem()) for _ in cases]

    def sh_party(role):
        def fn(chan):
            session = sh.SemiHonestSession(role, chan, sid)
            session.setup()
            outs = []
            for (_, alpha, beta), (a0, b0) in zip(cases, masks):
                a, b = (a0, b0) if role == 0 else (alpha ^ a0, beta ^ b0)
                outs.append(session.dpf_gen(n, a, b))
            return outs, session.delta, session.keys_generated
        return fn

    (o0, d0, c0), (o1, d1, c1), _, _ = run_two_party(sh_party(0), sh_party(1))
    check((c0, c1) == (3, 3), "semi-honest: the session counted its three keys")
    tag_sh = b"Couplet-baseline-semihonest"
    for key_id, (_, alpha, beta) in enumerate(cases):
        dom = lz.h0_domain(hs.sha256(tag_sh, sid, key_id.to_bytes(8, "little")), sh.LABEL_H0)
        other = lz.h0_domain(hs.sha256(tag_sh, sid, (key_id + 1).to_bytes(8, "little")), sh.LABEL_H0)
        for b, (out, delta_b) in enumerate(((o0[key_id], d0), (o1[key_id], d1))):
            leaves = recompute_leaves(delta_b, out.transcript, dom)
            t = gf.lsb(leaves)
            final = gf.int_to_arr(out.transcript["cw_final"])
            check(np.array_equal(out.t, t),
                  f"semi-honest, key number {key_id}, P_{b}: the tree is the one of H0 under (sid, {key_id})")
            check(np.array_equal(out.v, lz.hash0(leaves, dom) ^ (final & bit_mask(t))),
                  f"semi-honest, key number {key_id}, P_{b}: v is H0 of the leaves and the final word")
            wrong = recompute_leaves(delta_b, out.transcript, other)
            check(not np.array_equal(out.v, lz.hash0(wrong, other) ^ (final & bit_mask(gf.lsb(wrong)))),
                  f"semi-honest, key number {key_id}, P_{b}: another key number gives other shares (control)")
        unit = unit_vector(n, alpha)
        check(np.array_equal(o0[key_id].t ^ o1[key_id].t, unit)
              and np.array_equal(o0[key_id].v ^ o1[key_id].v, gf.bit_mul(unit, beta)),
              f"semi-honest, key number {key_id}: outputs reconstruct")
    check(sh._SID_TAG == tag_sh and lz._SID_TAG == b"Couplet-baseline-zhang",
          "the two baselines derive their per-key identifiers under different tags")
    _finish(start)


# --------------------------------------------------------------------------- #
# (g) Cost models
# --------------------------------------------------------------------------- #

def _walk_entries(obj):
    """Yield every cost entry (a dict with 'value', 'unit', 'source', 'kind') inside ``obj``."""
    if isinstance(obj, dict):
        if {"value", "unit", "source", "kind"} <= set(obj):
            yield obj
        for item in obj.values():
            yield from _walk_entries(item)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk_entries(item)


def test_costmodels() -> None:
    start = len(FAILURES)

    # schema of everything the module can return
    everything = [
        cm.comparison_rows(), cm.zhang_reported(), cm.authenticated_garbling_per_and(),
        cm.authenticated_garbling_aes_reported(), cm.ferret_bits_per_cot("reg"),
        cm.ferret_bits_per_cot("uni"), cm.silent_backend_bytes_per_and(3), cm.aes_call_ands(),
        cm.mascot_multiplication_bytes(), cm.backend_bytes_per_and(50_000),
        cm.zhang_instantiated_bytes(8, True), cm.zhang_instantiated_rounds(8, True),
        cm.gen_in_2pc_bytes(8, family="dcf", beta_mode="secret", expander="aes"),
    ]
    entries = list(_walk_entries(everything))
    check(len(entries) > 150, f"cost models: {len(entries)} entries inspected")
    bad = [e for e in entries
           if e["kind"] not in cm.KINDS or not isinstance(e["source"], str) or len(e["source"]) < 20
           or not isinstance(e["unit"], str) or not e["unit"] or e["reported"] != (e["kind"] == "reported")
           or e["value"] is None]
    check(not bad, f"cost models: every entry has value, unit, source, kind, reported ({bad[:1]})")
    kinds = {e["kind"] for e in entries}
    check(kinds == {"reported", "derived"}, "cost models: nothing is labelled measured by the module")
    check(raises(ValueError, cm.entry, 1, "x", "y", "guessed"), "an unknown kind is rejected")

    # every authenticated-garbling number is flagged as reported and carries its formula
    ag = cm.authenticated_garbling_per_and(128, 40, 3)
    check(len(ag) == 8 and all(e["reported"] and e["kind"] == "reported" and e["formula"]
                               for e in ag.values()),
          "authenticated garbling: 8 entries, all reported=True with their formula")
    check(ag["WRK17/function-independent"]["value"] == 3 * (5 * 128 + 3 * 40) == 2280
          and ag["WRK17/function-dependent"]["value"] == 3 * 128 + 4 * 40 == 544,
          "WRK17: (5 kappa + 3 rho) B and 3 kappa + 4 rho")
    check(ag["KRRW18-v1/function-independent"]["value"] == 10 * 128 + 8 * 40
          and ag["KRRW18-v1/function-dependent"]["value"] == 384
          and ag["KRRW18-v2/function-independent"]["value"] == 14 * 128 + 11 * 40
          and ag["KRRW18-v2/function-dependent"]["value"] == 256,
          "KRRW18: (4B - 2) kappa + (3B - 1) rho, 3 kappa; (4B + 2) kappa + (3B + 2) rho, 2 kappa")
    check(ag["CWYY23/one-way"]["value"] == 261 and ag["CWYY23/two-way"]["value"] == 301,
          "CWYY23: 2 kappa + 5 and 2 kappa + rho + 5")
    # the reported formulas reproduce the reported AES table (6,800 AND gates, B = 4)
    ag4 = cm.authenticated_garbling_per_and(128, 40, 4)
    table = {row["protocol"]: row for row in cm.authenticated_garbling_aes_reported()}
    check(all(row["reported"] for row in table.values()) and len(table) == 3,
          "AES table rows are flagged as reported")

    def megabytes(bits):
        return 6800 * bits / 8 / 1e6

    check(abs(megabytes(ag4["KRRW18-v1/function-independent"]["value"])
              - table["KRRW18-v1"]["one_way"]["function_independent_MB"]) < 0.05
          and abs(megabytes(ag4["KRRW18-v1/function-dependent"]["value"])
                  - table["KRRW18-v1"]["one_way"]["function_dependent_MB"]) < 0.01,
          "KRRW18 v1: formulas at B = 4 match the reported AES row (1.9 MB, 0.33 MB)")
    check(abs(megabytes(ag4["KRRW18-v2/function-independent"]["value"])
              - table["KRRW18-v2"]["one_way"]["function_independent_MB"]) < 0.1
          and abs(megabytes(ag4["KRRW18-v2/function-dependent"]["value"])
                  - table["KRRW18-v2"]["one_way"]["function_dependent_MB"]) < 0.01,
          "KRRW18 v2: formulas at B = 4 match the reported AES row (2.5 MB, 0.22 MB) within 0.1 MB")
    est = cm.authenticated_garbling_bytes(1000, "KRRW18-v1", bucket=3)
    check(est["kind"] == "derived" and est["value"] == 1000 * (1600 + 384) / 8
          and all(part["reported"] for part in est["inputs"]),
          "authenticated-garbling estimate: derived from reported inputs")
    check(cm.authenticated_garbling_bytes(8, "CWYY23")["value"] == 301
          and raises(ValueError, cm.authenticated_garbling_bytes, 8, "nobody"),
          "authenticated-garbling estimate: CWYY23 two-way total; unknown protocol rejected")

    # Zhang et al.: hybrid-model counts and our instantiation
    check(cm.zhang_hybrid_bits(20)["value"] == 2 * 23 * 128 and cm.zhang_hybrid_rounds(20)["value"] == 22
          and cm.zhang_abit_demand(20)["value"] == 276 == lz.abit_demand(20),
          "Zhang et al., hybrid model: 2 (n + 3) kappa bits, n + 2 rounds, n + 2 kappa aBits")
    for n in (8, 12, 16, 20):
        check(cm.zhang_instantiated_bytes(n)["per_party"] == 16 * n + (n + 135) // 8 + 416
              and cm.zhang_instantiated_rounds(n)["value"] == n + 11
              and cm.zhang_instantiated_bytes(n, True)["per_party"] == 16 * n + (n + 135) // 8 + 320
              and cm.zhang_instantiated_rounds(n, True)["value"] == n + 9,
              f"Zhang et al., our instantiation, n = {n}: bytes and rounds as implemented")
        check(cm.semihonest_instantiated_bytes(n)["per_party"] == 16 * n + 1056
              and cm.semihonest_instantiated_rounds(n)["value"] == n + 2,
              f"semi-honest baseline, n = {n}: bytes and rounds as implemented")
    for n in (8, 20, 70):
        single = cm.zhang_abit_bytes(n, "iknp_single")
        check(single["per_party"] == u_message_bytes(n + 256) + SEED_BYTES + CHECK_BYTES
              and cm.zhang_abit_bytes(n, "iknp_amortized")["value"] == 32 * (n + 256)
              and abs(cm.zhang_abit_bytes(n, "ferret")["value"] - 2 * (n + 256) * 0.44 / 8) < 1e-9,
              f"Zhang et al., n = {n}: authenticated-bit cost under the three instantiations")
        check(cm.zhang_total_bytes(n)["value"]
              == cm.zhang_instantiated_bytes(n)["value"] + single["value"],
              f"Zhang et al., n = {n}: total = protocol + authenticated bits")
    check(cm.zhang_abit_bytes(20)["value"] == 18496 and raises(ValueError, cm.zhang_abit_bytes, 8, "x"),
          "Zhang et al., n = 20: 18,496 bytes for the authenticated bits on our extension")
    ratio = cm.zhang_vs_bcg20(20)
    check(ratio["kind"] == "derived" and 40 < ratio["value"] < 100,
          f"Boyle et al. / Pi_DPF at n = 20: {ratio['value']:.1f}x, the order of the reported 50x")
    rep = cm.zhang_reported()
    check(rep[0]["value"] == 50 and all(e["reported"] for e in rep),
          "Zhang et al.: the reported statements are flagged as reported")

    # Boyle et al. as characterised by Zhang et al.
    check(cm.mascot_multiplication_bytes()["value"] == 33000
          and cm.mascot_multiplication_bytes()["reported"],
          "MASCOT multiplication: 33,000 bytes, reported")
    bcg = cm.bcg20_dpf_bytes(20)
    check(bcg["value"] == 3 * 33000 + 2 * 21 * 128 / 8 and bcg["reported"]
          and bcg["rest_bits"] / 8 < 0.01 * bcg["value"],
          "Boyle et al.: three multiplications dominate (99,000 bytes + 672)")

    # AND counts
    check(cm.aes_call_ands()["value"] == 5120 and cm.aes_expander_ands_per_call(256)["value"] == 10240
          and cm.aes_expander_ands_per_call(192)["value"] == 10240
          and cm.aes_expander_ands_per_call(128)["value"] == 5120,
          "AES: 5,120 AND gates per call, two calls per G")
    path = cm.circuit_stats_path()
    if os.path.exists(path):
        check(cm.load_circuit_affine() == cm.CIRCUIT_AFFINE,
              "the affine AND-count forms equal those in circuit_stats.json")
    else:
        print("  circuit_stats.json not found: comparison with the compiler output skipped")
    for n in (8, 12, 16, 20):
        check(cm.gen_circuit_ands(n)["value"] == 7555 * n + 7799
              and cm.gen_circuit_ands(n)["and_depth"] == 14 * n + 138
              and cm.gen_circuit_ands(n, beta_mode="secret")["value"] == 7555 * n + 16117,
              f"C_dpf, n = {n}: 7,555 n + 7,799 (public payload) / + 16,117 (secret)")
        check(cm.gen_circuit_ands(n, expander="aes")["value"] == 20867 * n + 21111,
              f"C_dpf with an AES expander, n = {n}: 20,867 n + 21,111")
    check(cm.gen_circuit_ands(16)["value"] == 128679, "C_dpf, n = 16: 128,679 AND gates")
    aes_dcf = cm.gen_circuit_ands(16, "dcf", expander="aes")
    check(aes_dcf["per_level"] == 23736 - 6 * 3584 + 6 * 10240
          and aes_dcf["fixed"] == 7796 - 2 * 3584 + 2 * 10240,
          "C_dcf with an AES expander: six expander calls per level, two at the leaf")
    aes_32 = cm.gen_circuit_ands(8, "dpf", 32, 64, "public_one", "aes")
    check(aes_32["per_level"] == 20867 and aes_32["fixed"] == 7479 - 2 * 3520 + 2 * 10240,
          "C_dpf (k = 32) with an AES expander: Conv calls of 3,520 AND gates replaced")
    check(raises(ValueError, cm.gen_circuit_ands, 8, "dpf", 16, 64)
          and raises(ValueError, cm.gen_circuit_ands, 8, expander="sha"),
          "unknown circuit parameters are rejected")

    # backend bytes per AND gate: the exact accounting behind the measured figures
    b50 = cm.backend_bytes_per_and(50_000)
    b300 = cm.backend_bytes_per_and(300_000)
    check(b50["bucket"] == 4 and abs(b50["value"] - 385.0092) < 1e-3,
          f"backend, ell = 50,000: B = 4, 385.01 bytes per AND ({b50['value']:.4f})")
    check(b300["bucket"] == 3 and abs(b300["value"] - 288.5707) < 1e-3,
          f"backend, ell = 300,000: B = 3, 288.57 bytes per AND ({b300['value']:.4f})")
    check(b50["per_party_phases"]["laand"] == 16 * 50_000 * 4
          and b50["per_party_phases"]["check"] == 208
          and b50["per_party_phases"]["bucket"] == 96 + 18750
          and b50["per_party_phases"]["cotext"] == 16 * (400_128 + 128) + 16 + 2064,
          "backend, ell = 50,000: phases as in CHANGELOG PY-C item 10")
    g = cm.gen_in_2pc_bytes(16)
    check(g["and_gates"] == 128679 and abs(g["value"] - 128679 * (
        cm.backend_bytes_per_and(128679)["value"] + 0.5)) < 1e-6,
          "generation in 2PC: AND gates x (bytes per AND + 4 bits)")
    fake = cm.entry(300.0, "bytes per AND triple (both parties)", "a measured figure for the test",
                    "measured")
    check(cm.gen_in_2pc_bytes(16, bytes_per_and=fake)["value"] == 128679 * 300.5,
          "generation in 2PC: a measured per-AND figure can be supplied")

    # silent-OT variant
    s3 = cm.silent_backend_bytes_per_and(3, "reg")
    check(abs(s3["value"] - (3 * (4 * 0.44 + 256) + 4 + 4) / 8) < 1e-9
          and abs(s3["same_model_with_iknp_rows"] - (3 * 768 + 8) / 8) < 1e-9
          and s3["kind"] == "derived" and s3["inputs"][0]["reported"],
          "silent-OT variant: B (4 c + 2 kappa) + 2 (B - 1) + 4 bits per AND gate")
    check(cm.ferret_bits_per_cot("reg")["value"] == 0.44 and cm.ferret_bits_per_cot("uni")["value"] == 0.73
          and "secondary" in cm.ferret_bits_per_cot()["note"]
          and raises(ValueError, cm.ferret_bits_per_cot, "fast"),
          "Ferret figures: reported, with the verification caveat")

    # comparison table
    rows = cm.comparison_rows((8, 20))
    check([row["n"] for row in rows] == [8, 20]
          and rows[1]["this_work_gen_in_2pc"]["and_gates"]["value"] == 158899
          and rows[1]["naive_aes_gen_in_2pc"]["and_gates"]["value"] == 438451
          and rows[1]["naive_aes_gen_in_2pc"]["bucket"] == 3
          and rows[0]["this_work_gen_in_2pc"]["bucket"] == 4,
          "comparison rows: AND counts and bucket sizes for n = 8 and n = 20")
    rows_m = cm.comparison_rows((8,), measured_bytes_per_and={4: fake})
    check(rows_m[0]["this_work_gen_in_2pc"]["bytes"]["inputs"][1]["kind"] == "measured"
          and rows_m[0]["this_work_gen_in_2pc"]["bytes"]["value"] == (7555 * 8 + 7799) * 300.5,
          "comparison rows: a measured per-AND figure is used where the bucket size matches")
    _finish(start)


TESTS = [
    test_building_blocks,
    test_leaky_correctness,
    test_leaky_real_prep,
    test_selective_failure,
    test_honest_never_aborts,
    test_semihonest,
    test_key_separation,
    test_costmodels,
]


def main() -> int:
    t_all = time.perf_counter()
    for test in TESTS:
        before = CHECKS
        t0 = time.perf_counter()
        try:
            test()
        except AssertionError:
            pass  # the failed checks are already recorded in FAILURES
        except Exception as exc:  # an unexpected error is a failure as well
            FAILURES.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print("ERROR:", FAILURES[-1])
        print(f"{test.__name__}: {CHECKS - before} checks, {time.perf_counter() - t0:.1f} s")
    print(f"test_baselines: {CHECKS} checks, {len(FAILURES)} failures, "
          f"{time.perf_counter() - t_all:.1f} s")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
