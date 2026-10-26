"""End-to-end tests of Pi_AuthEval (couplet.autheval; DESIGN.md sections 5.3, 5.4, 6;
changes D, E, F, G).

Two parties run the party-local driver ``AuthEval`` as two threads:

  * pipelines with IDEAL preprocessing: shared input -> table gate (ReLU
    table) -> wire -> comparison gate -> Open, for every x; Lin with public
    coefficients and constants (released with an output mask); DPF and DCF
    keys queried at every public point and opened; additive index shares;
  * the same pipeline once with the REAL preprocessing (base OT, OT extension,
    leaky AND, bucketing);
  * one-time use: a second ``Mask`` on a gate raises and NOTHING is sent, in
    both modes, also after a failed first attempt; output masks are used once;
  * shared-input ``Mask`` is a commit-then-open exchange: a rushing party that
    chooses its share after it has seen the honest share (to fix ``x_hat``)
    fails the opening of its commitment; mirrored and replayed commitments
    are rejected; the exact bytes on the wire;
  * triple budget: every honest session of this file runs in strict mode
    (the default) on the exact reservation computed from the planned
    commands and makes ONE triple request; a short pool is an error before
    any message and any change of state; refill mode counts its requests;
  * the order of ``Open``: the deferred openings are verified before any
    output mask is generated, so a wrong wire-mode share ends the session in
    the first check of ``Open`` and nothing else is sent;
  * state machine and abort: commands before setup, monotone identifiers,
    argument errors that must not abort, a peer that deviates (the session is
    dead afterwards and sends nothing), a key that is never CERTIFIED when the
    output check fails, restart on persisted counters;
  * the carry-leak regression (change D): in wire mode the opened masked value
    ``x + r'`` is uniform over the whole ring Z_{2^(n+s)} whatever x is
    (chi-square over 20000 openings per input for n = 3 with a small s, where
    the ring has 512 elements, and over projections for s = 64), whereas the
    naive variant that masks with ``r = r' mod 2^n`` instead of ``r'`` is
    distinguishable on the very same samples (advantage about 1 - 1/N) although
    its MAC check passes.

Run from artifact/python:  python tests/test_autheval.py   (``-k name`` selects
tests, ``--no-real`` skips the real preprocessing).  Prints the number of
checks; exits non-zero on any failure.
"""

from __future__ import annotations

import math
import os
import random
import secrets
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import circuits_gen as cg  # noqa: E402
from couplet import gates  # noqa: E402
from couplet import hashes as hs  # noqa: E402
from couplet import protocol as pg  # noqa: E402
from couplet.attacks.adversaries import (  # noqa: E402
    CheatingChecker, Mirror, Replay, Rewrite, Rush, TamperChannel)
from couplet.gates import GateSpec  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.prep import IdealDealer  # noqa: E402
from couplet.spdz2k import AShare, deal  # noqa: E402
from couplet.state import CERTIFIED, GENERATING, USED, GateUsedError, StateError  # noqa: E402
from couplet.autheval import ABORTED, ACTIVE, TripleBudgetError, AuthEval  # noqa: E402

NO_REAL = "--no-real" in sys.argv
S = 64
CHECKS = 0
NOTES: list = []
RNG = random.Random(0x7E51F55)


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

def check(cond, msg: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg or "check failed")


def check_eq(got, exp, msg: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if got != exp:
        raise AssertionError(f"{msg}: got {got!r}, expected {exp!r}")


def raises(exc, fn) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001 - a different exception is a failed expectation
        return False
    return False


def run_drivers(script, k: int = 32, s: int = S, prep: str = "ideal", wrap=(None, None),
                factories=(None, None), delta_shares=(None, None), state_paths=(None, None),
                sid: bytes | None = None, setup: bool = True, timeout: float = 900, reserve=0,
                outmasks: int = 0, allow_refill=None, classes=(None, None)):
    """Run ``script(role, vf)`` at both parties; returns (out0, out1, stats0, stats1, drivers).

    ``out_b`` is ``("ok", value)`` or ``("abort", message)``.  ``wrap[b]`` wraps
    party b's channel endpoint, ``factories[b]`` replaces its arithmetic checker,
    ``classes[b]`` is a deviating subclass of the driver.  ``reserve`` (a number
    of AND triples, or a list of planned calls of ``couplet.protocol``) and the
    triples of ``outmasks`` output masks are drawn together with those of the
    session setup.  The drivers are built WITHOUT the ``allow_refill`` argument
    unless it is given here, so every run of this file exercises the default
    of the driver, which is the strict mode.
    """
    dealer = IdealDealer() if prep == "ideal" else None
    sid = b"test-autheval-" + secrets.token_bytes(8) if sid is None else sid
    drivers = [None, None]
    if not isinstance(reserve, int):
        reserve = pg.and_budget(reserve)
    mode = {} if allow_refill is None else {"allow_refill": allow_refill}

    def party(role):
        def fn(chan):
            if wrap[role] is not None:
                chan = wrap[role](chan)
            vf = (classes[role] or AuthEval)(
                role, chan, sid, k=k, s=s, prep=prep, dealer=dealer, checker_factory=factories[role],
                delta_share=delta_shares[role], state_path=state_paths[role], **mode)
            drivers[role] = vf
            try:
                if setup:
                    vf.setup(reserve, outmasks)
                return "ok", script(role, vf)
            except ProtocolAbort as exc:
                return "abort", str(exc)
        return fn

    out0, out1, st0, st1 = run_two_party(party(0), party(1), timeout=timeout)
    return out0, out1, st0, st1, drivers


def run_ok(script, exact: bool = True, requests: int = 1, **kwargs):
    """An honest run.  In strict mode it must have made exactly ONE triple request (``requests``
    for the scripts that add a population explicitly) and, with ``exact`` (the reservation is
    the budget of the script), used every reserved triple."""
    out0, out1, st0, st1, drivers = run_drivers(script, **kwargs)
    check(out0[0] == "ok" and out1[0] == "ok",
          f"honest run aborted: {out0[1] if out0[0] != 'ok' else out1[1]}")
    if kwargs.get("setup", True) and not kwargs.get("allow_refill"):
        check_eq([vf.triple_requests for vf in drivers], [requests, requests],
                 "an honest strict session makes ONE triple request (one bucketing population)")
        if exact and all(vf.state == ACTIVE for vf in drivers):
            check_eq([vf.session.engine.triples_available() for vf in drivers], [0, 0],
                     "the reservation was exact: no reserved triple is left")
    return out0[1], out1[1], st0, st1, drivers


def deviating_mask(transform):
    """A driver class whose shared-input Mask commits to and opens ``transform(honest message)``.

    The party keeps computing with its honest share: a deviation with a
    consistent commitment (what a malicious party can always do).
    """
    class DeviatingMask(AuthEval):
        def _mask_exchange(self, gid, share, nbytes):
            label = b"mask|" + gid.to_bytes(8, "little")
            msg = transform(share.to_bytes(nbytes, "little"))
            return hs.committed_exchange(self.chan, self._sid_mask, label, msg)
    return DeviatingMask


def split(value: int, modulus: int) -> tuple:
    a = RNG.randrange(modulus)
    return a, (value - a) % modulus


def silent(vf, fn, exc) -> bool:
    """``fn`` raises ``exc`` and the party sends and receives nothing."""
    chan = vf.chan
    before = (chan.msgs_sent, chan.bytes_sent, chan.msgs_recv)
    ok = raises(exc, fn)
    return ok and (chan.msgs_sent, chan.bytes_sent, chan.msgs_recv) == before


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #

def test_pipeline_ideal():
    """shared input -> table gate (ReLU) -> wire -> comparison gate -> Open, every x."""
    for (n, k) in ((4, 32), (3, 64)):
        size = 1 << n
        relu = gates.relu_table(n, k)
        c = RNG.randrange(1, size // 2)
        xs = list(range(size))
        shares = [split(x, size) for x in xs]
        spec_table = GateSpec.table(n, k, S, relu)
        spec_cmp = GateSpec.comparison(n, k, S, c, auth_input=True)

        def script(role, vf):
            res = []
            for j, x in enumerate(xs):
                g1 = vf.gate(2 * j, spec_table)
                g2 = vf.gate(2 * j + 1, spec_cmp)
                rounds0, bytes0 = vf.chan.rounds, vf.chan.bytes_sent
                x_hat = vf.mask_shared(g1, shares[j][role])
                cost_shared = (vf.chan.rounds - rounds0, vf.chan.bytes_sent - bytes0)
                y = vf.apply(g1)                               # [[ReLU(x)]], exact
                rounds0, bytes0 = vf.chan.rounds, vf.chan.bytes_sent
                y_hat = vf.mask_wire(g2, y)
                cost_wire = (vf.chan.rounds - rounds0, vf.chan.bytes_sent - bytes0)
                z = vf.apply(g2)                               # [[ReLU(x) < c]], exact
                rounds0 = vf.chan.rounds
                out = vf.open(z)
                cost_open = vf.chan.rounds - rounds0
                res.append((x_hat, y_hat, out, y.exact and z.exact, cost_shared, cost_wire, cost_open,
                            vf.registry.gate(g2).opened, vf.checker.pending))
            return res, vf.info()

        r0, r1, st0, st1, drivers = run_ok(script, k=k, reserve=[pg.planned_gate(spec_table, size),
                                                                 pg.planned_gate(spec_cmp, size)])
        for j, x in enumerate(xs):
            a, b = r0[0][j], r1[0][j]
            tag = f"n={n} k={k} x={x}"
            check_eq(a[2], int(relu[x] < c), f"{tag}: opened [ReLU(x) < {c}] at P_0")
            check_eq(b[2], a[2], f"{tag}: both parties open the same value")
            check_eq((a[0], a[1]), (b[0], b[1]), f"{tag}: both parties see the same masked inputs")
            check(a[3] and b[3], f"{tag}: table and comparison outputs are exact")
            check_eq(a[4], (2, 64 + (n + 7) // 8),
                     f"{tag}: shared-input Mask is two rounds (commit, open), 64 + ceil(n/8) bytes")
            check_eq(a[5], (1, (n + S + 7) // 8), f"{tag}: wire Mask is one round, ceil((n+s)/8) bytes")
            check_eq(a[6], 9, f"{tag}: Open = deferred check (4) + opening (1) + check (4) rounds")
            check_eq(a[7] % size, a[1], f"{tag}: x_hat is the opening modulo 2^n")
            check_eq(a[8], 0, f"{tag}: nothing is pending after Open")
        info = r0[1]
        check_eq(info["registry"]["gates"], {GENERATING: 0, CERTIFIED: 0, USED: 2 * size}, "every gate is USED")
        check_eq(info["state"], ACTIVE, "session state")
        check_eq((info["triple_requests"], info["triples_unused"], info["allow_refill"]), (1, 0, False),
                 "info(): one bucketing population for the whole session (strict mode)")
        check_eq((info["arith_openings"], info["arith_checks"]), (2 * size, 2 * size), "openings and checks")
        check_eq(st0["bytes_sent"], st1["bytes_recv"], "bytes sent equal bytes received")
        check_eq(st0["rounds"], st1["rounds"], "round counters agree")
        check(len({r[1] for r in r0[0]}) > 1, "the masked wire values vary (fresh offsets)")
    NOTES.append("ideal preprocessing: pipeline shared input -> ReLU table gate -> wire -> comparison -> "
                 "Open correct for every x, (n, k) = (4, 32) and (3, 64)")


def test_lin_and_output_masks():
    """Lin with public coefficients and constants; releases of non-exact sharings use output masks."""
    n, k = 4, 32
    size, mod_k = 1 << n, 1 << k
    table = [RNG.randrange(mod_k) for _ in range(size)]
    x = RNG.randrange(size)
    xs = split(x, size)
    coeffs = [3, -5, mod_k - 1, 0, 1 << 40]
    const = -12345

    def script(role, vf):
        gids = vf.gate_batch(0, GateSpec.table(n, k, S, table), 2)
        g3 = vf.gate(None, GateSpec.comparison(n, k, S, size // 2, auth_input=True))
        a = [vf.apply(g) for g in gids if vf.mask_shared(g, xs[role]) is not None]
        t = a[0]                                               # [[T[x]]] twice (two fresh gates)
        s1 = vf.lin([(coeffs[0], a[0]), (coeffs[1], a[1])], const)
        s2 = vf.lin([(coeffs[2], t)])
        s3 = vf.lin([(coeffs[3], t)], 7)
        s4 = vf.lin([(coeffs[4], t)])
        same = vf.lin([(1, t)]) is t
        pool0 = vf.outmasks_available
        v1 = vf.open(s1)                                       # generates one output mask on demand
        pool1 = vf.outmasks_available
        vf.reserve_outmasks(5)
        pool2 = vf.outmasks_available
        v2, v3, v4 = vf.open_many([s2, s3, s4])
        public = list(vf.checker.last_opened)
        pool3 = vf.outmasks_available
        # a non-exact wire may be masked into the next gate: x' = Val mod 2^n
        y_hat = vf.mask_wire(g3, s1)
        z = vf.open(vf.apply(g3))
        exact_val = vf.open(t)
        return (v1, v2, v3, v4, z, exact_val, same, (pool0, pool1, pool2, pool3),
                [s.exact for s in (s1, s2, s3, s4)], public, y_hat)

    # six output masks are generated (one on demand, five in a batch); four are used
    r0, r1, _, _, _ = run_ok(script, k=k, outmasks=6, reserve=[
        pg.planned_gate(GateSpec.table(n, k, S, table), 2),
        pg.planned_gate(GateSpec.comparison(n, k, S, size // 2, auth_input=True))])
    t = table[x]
    want1 = (coeffs[0] * t + coeffs[1] * t + const) % mod_k
    check_eq(r0[:6], (want1, coeffs[2] * t % mod_k, 7, coeffs[4] * t % mod_k,
                      int(want1 % size < size // 2), t), "opened values of the linear combinations")
    check_eq(r1[:6], r0[:6], "both parties open the same values")
    check(r0[6], "Lin with the single term 1 * v returns v itself (still exact)")
    check_eq(r0[7], (0, 0, 5, 2), "output masks: generated on demand, reserved in a batch, used once each")
    check_eq(r0[8], [False] * 4, "Lin results are not exact")
    # the public openings of the non-exact values carry the mask in the high s bits
    reps = [coeffs[2] * t, 7, coeffs[4] * t]
    masked_high = sum(1 for rep, pub in zip(reps, r0[9]) if pub >> k != (rep % (1 << (k + S))) >> k)
    check(masked_high >= 2, "the high parts of the released representatives are masked")
    check_eq([p % mod_k for p in r0[9]], [v % mod_k for v in reps], "the low k bits are the values")


def test_dpf_dcf_query_open():
    """Keys queried at every public point and opened; index shares XOR and additive; public payload."""
    for (n, k) in ((4, 32), (5, 64)):
        size, mod_k = 1 << n, 1 << k
        alpha = RNG.randrange(size)
        beta = RNG.randrange(1, mod_k)
        a_xor = (RNG.randrange(size),)
        a_xor += (a_xor[0] ^ alpha,)
        a_add = split(alpha, size)
        b_sh = split(beta, mod_k)

        def script(role, vf):
            k_dpf = vf.key(0, "dpf", n, a_xor[role], b_sh[role])
            k_dcf = vf.key(1, "dcf", n, a_add[role], b_sh[role], index_mode="add")
            k_one, k_two = vf.key_batch(2, "dpf", n, [a_xor[role], a_add[role]])   # payload 1; alpha, and a second index
            all_dpf = vf.query_all(k_dpf)
            pts = [vf.query(k_dpf, x) for x in range(size)]
            same = all((p.value, p.mac) == (q.value, q.mac) for p, q in zip(all_dpf, pts))
            out_dpf = vf.open_many(all_dpf)
            out_dcf = vf.open_many(vf.query_all(k_dcf))
            out_one = [vf.open(vf.query(k_one, x)) for x in (alpha, (alpha + 1) % size, 0, size - 1)]
            total = vf.open(vf.lin([(1, p) for p in all_dpf]))          # sum over the domain = beta
            weighted = vf.open(vf.lin([(x, p) for x, p in enumerate(vf.query_all(k_one))]))  # = alpha
            return (out_dpf, out_dcf, out_one, total, weighted, same,
                    all(p.exact for p in all_dpf), vf.info()["registry"]["keys"], vf.info()["counters"]["next"])

        r0, r1, _, _, _ = run_ok(script, k=k, outmasks=2, reserve=[
            pg.planned_key("dpf", n, k, S, "xor", "secret"), pg.planned_key("dcf", n, k, S, "add", "secret"),
            pg.planned_key("dpf", n, k, S, "xor", "public_one", 2)])
        check_eq(r0[0], [beta if x == alpha else 0 for x in range(size)], f"n={n} k={k}: DPF at every point")
        check_eq(r0[1], [beta if x < alpha else 0 for x in range(size)], f"n={n} k={k}: DCF at every point")
        check_eq(r0[2], [1, 0, int(alpha == 0), int(alpha == size - 1)], "payload-1 DPF at chosen points")
        check_eq(r0[3], beta, "sum of all DPF outputs")
        check_eq(r0[4], alpha, "sum of x * [x = alpha] recovers alpha (a secret-index selection)")
        check(r0[5] and r0[6], "Query equals EvalAll; query outputs are exact")
        check_eq(r0[7], {GENERATING: 0, CERTIFIED: 4}, "four certified keys")
        check_eq(r0[8]["kid"], 4, "key identifiers")
        check_eq(r1[:5], r0[:5], "both parties open the same values")
    NOTES.append("ideal preprocessing: DPF and DCF keys (secret and public payload, XOR and additive index "
                 "shares) queried at every point and opened, n in {4, 5}, k in {32, 64}")


def test_pipeline_real():
    """The pipeline once with the complete real preprocessing."""
    if NO_REAL:
        NOTES.append("real preprocessing: SKIPPED (--no-real)")
        return
    n, k = 4, 32
    size = 1 << n
    relu = gates.relu_table(n, k)
    c = 3
    x = 2
    xs = split(x, size)
    x2 = 13
    xs2 = split(x2, size)
    times = {}
    spec_t = GateSpec.table(n, k, S, relu)
    spec_c = GateSpec.comparison(n, k, S, c, auth_input=True)
    # all triples of the session in ONE bucketing population (2^-sigma for the session, section 7.5)
    reserve = 2 * sum(pg.generation_circuit(sp.descriptor()).stats()["n_and"] for sp in (spec_t, spec_c))

    def script(role, vf):
        t0 = time.perf_counter()
        g1, g1b = vf.gate_batch(0, spec_t, 2)
        g2, g2b = vf.gate_batch(2, spec_c, 2)
        t1 = time.perf_counter()
        outs = []
        for ga, gb, sh in ((g1, g2, xs), (g1b, g2b, xs2)):
            vf.mask_shared(ga, sh[role])
            y = vf.apply(ga)
            vf.mask_wire(gb, y)
            outs.append(vf.open(vf.apply(gb)))
            outs.append(vf.open(y))
        times[role] = (t1 - t0, time.perf_counter() - t1)
        return outs, vf.info()

    t = time.perf_counter()
    r0, r1, st0, st1, _ = run_ok(script, k=k, prep="real", reserve=reserve)
    wall = time.perf_counter() - t
    want = [int(relu[x] < c), relu[x], int(relu[x2] < c), relu[x2]]
    check_eq(r0[0], want, "real preprocessing: opened values at P_0")
    check_eq(r1[0], want, "real preprocessing: opened values at P_1")
    check_eq(r0[1]["prep"], "real", "the session used the real preprocessing")
    check(r0[1]["triples_used"] > 100000, "the generation circuits consumed real triples")
    check_eq((r0[1]["triple_requests"], r0[1]["triples_unused"]), (1, 0),
             "all triples of the session come from one bucketing population")
    check_eq(st0["bytes_sent"], st1["bytes_recv"], "bytes sent equal bytes received")
    NOTES.append(
        f"REAL preprocessing (sigma = 40), n = 4, k = 32: two pipelines shared input -> ReLU table -> wire "
        f"-> comparison -> Open correct; {r0[1]['triples_used']} AND triples in "
        f"{r0[1]['triple_requests']} bucketing population (bucket size {r0[1]['last_bucket_size']}), "
        f"{st0['bytes_sent'] + st1['bytes_sent']} bytes in both directions, {st0['rounds']} rounds, "
        f"{wall:.1f} s wall for both parties as two threads (session setup with the whole preprocessing "
        f"{wall - times[0][0] - times[0][1]:.1f} s, the four generation circuits {times[0][0]:.1f} s, "
        f"masks, local evaluation and opens {times[0][1] * 1000:.0f} ms)")


# --------------------------------------------------------------------------- #
# One-time use
# --------------------------------------------------------------------------- #

def test_one_time_use():
    n, k = 4, 32
    size = 1 << n
    x = RNG.randrange(size)
    xs = split(x, size)
    spec_t = GateSpec.table(n, k, S, list(range(size)))
    spec_z = GateSpec.zero_test(n, k, S, auth_input=True)

    def script(role, vf):
        facts = {}
        g_sh, g_sh2 = vf.gate_batch(0, spec_t, 2)
        g_w = vf.gate(None, spec_z)
        # refused before any state change: the gate stays CERTIFIED
        facts["bad_share"] = silent(vf, lambda: vf.mask_shared(g_sh, size), ValueError)
        huge = vf.gate(1 << 64, spec_t)                       # an identifier beyond the 8-byte label
        facts["huge_gid"] = (silent(vf, lambda: vf.mask_shared(huge, 0), ValueError)
                             and vf.registry.gate(huge).state == CERTIFIED)
        facts["bad_share2"] = silent(vf, lambda: vf.mask_shared(g_sh, -1), ValueError)
        facts["unknown_gate"] = silent(vf, lambda: vf.mask_shared(99, 0), StateError)
        facts["apply_before_mask"] = silent(vf, lambda: vf.apply(g_sh), StateError)
        facts["wire_without_auth"] = silent(
            vf, lambda: vf.mask_wire(g_sh, AShare(0, 0, vf.w, True)), ValueError)
        facts["wire_small_modulus"] = silent(
            vf, lambda: vf.mask_wire(g_w, AShare(0, 0, n + S - 1)), ValueError)
        facts["wire_not_a_share"] = silent(vf, lambda: vf.mask_wire(g_w, 5), ValueError)
        facts["still_certified"] = [vf.registry.gate(g).state for g in (g_sh, g_w)]
        # the one Mask
        x_hat = vf.mask_shared(g_sh, xs[role])
        facts["used_after_mask"] = vf.registry.gate(g_sh).state
        # second Mask on the same gate: raises, nothing is sent (several variants)
        facts["second_same"] = silent(vf, lambda: vf.mask_shared(g_sh, xs[role]), GateUsedError)
        facts["second_other_input"] = silent(vf, lambda: vf.mask_shared(g_sh, (xs[role] + 1) % size),
                                             GateUsedError)
        wire = vf.apply(g_sh)
        facts["second_as_wire"] = silent(vf, lambda: vf.mask_wire(g_sh, wire), (GateUsedError, ValueError))
        y_hat = vf.mask_wire(g_w, wire)
        facts["wire_second_same"] = silent(vf, lambda: vf.mask_wire(g_w, wire), GateUsedError)
        facts["wire_second_other"] = silent(
            vf, lambda: vf.mask_wire(g_w, vf.lin([(1, wire)], 1)), GateUsedError)
        facts["wire_second_as_shared"] = silent(vf, lambda: vf.mask_shared(g_w, 0), GateUsedError)
        facts["x_hat_unchanged"] = (vf.registry.gate(g_sh).x_hat, vf.registry.gate(g_w).x_hat) == (x_hat, y_hat)
        # Apply may be repeated (it is local) and gives the same share
        facts["apply_repeatable"] = vf.apply(g_w) == vf.apply(g_w)
        out = vf.open(vf.apply(g_w))
        # after the refused attempts the session still works: the spare gate is fine
        vf.mask_shared(g_sh2, xs[role])
        out2 = vf.open(vf.apply(g_sh2))
        return facts, out, out2, vf.state

    r0, r1, _, _, _ = run_ok(script, k=k, reserve=[pg.planned_gate(spec_t, 3), pg.planned_gate(spec_z)])
    for r in (r0, r1):
        facts = r[0]
        for name in ("bad_share", "bad_share2", "huge_gid", "unknown_gate", "apply_before_mask",
                     "wire_without_auth", "wire_small_modulus", "wire_not_a_share"):
            check(facts[name], f"{name}: refused with the right exception and nothing sent")
        check_eq(facts["still_certified"], [CERTIFIED, CERTIFIED], "refused calls do not consume the gate")
        check_eq(facts["used_after_mask"], USED, "Mask marks the gate USED")
        for name in ("second_same", "second_other_input", "second_as_wire", "wire_second_same",
                     "wire_second_other", "wire_second_as_shared"):
            check(facts[name], f"{name}: a second Mask raises and NOTHING is sent")
        check(facts["x_hat_unchanged"] and facts["apply_repeatable"], "the recorded x_hat is not touched")
        check_eq((r[1], r[2], r[3]), (int(x == 0), x, ACTIVE), "the session is intact after the refusals")


def test_one_time_use_after_failure():
    """A failed or interrupted Mask leaves the gate USED: a retry raises and sends nothing."""
    n, k = 4, 32
    size = 1 << n
    spec_t = GateSpec.table(n, k, S, list(range(size)))
    spec_z = GateSpec.zero_test(n, k, S, auth_input=True)

    # (a) the peer commits to and opens a malformed message, or spoils its opening
    for name, mutate, tamper, want in (
            ("too long", lambda m: m + b"\x00", None, "mask: message has the wrong length"),
            ("empty", lambda m: b"", None, "mask: message has the wrong length"),
            ("out of range", lambda m: bytes([m[0] | 0x80]), None, "mask: share out of range"),
            ("opening changed", None, lambda m: m[:-1] + bytes([m[-1] ^ 1]), "commitment opening failed"),
            ("opening truncated", None, lambda m: m[:31], "malformed opening"),
            ("commitment truncated", None, None, "malformed commitment")):
        cls = None if mutate is None else deviating_mask(mutate)

        def wrap(chan, tamper=tamper):
            if tamper is not None:          # the second message of the phase: opening || share
                return TamperChannel(chan, [Rewrite("mask", 1, tamper)])
            return TamperChannel(chan, [Rewrite("mask", 0, lambda m: m[:31])])

        def script(role, vf):
            g = vf.gate(0, spec_t)
            try:
                vf.mask_shared(g, 3)
            except ProtocolAbort as exc:
                state = vf.registry._gates[g].state
                retry = silent(vf, lambda: vf.mask_shared(g, 3), (ProtocolAbort, GateUsedError))
                retry2 = silent(vf, lambda: vf.mask_shared(g, 5), (ProtocolAbort, GateUsedError))
                return str(exc), state, retry and retry2, vf.state, vf.delta_share
            return "no abort", None, None, vf.state, None

        out0, out1, st0, _, _ = run_drivers(script, k=k, wrap=(None, None if cls else wrap),
                                            classes=(None, cls), reserve=[pg.planned_gate(spec_t)])
        check_eq(out0, ("ok", (want, USED, True, ABORTED, None)), f"malformed mask reply ({name})")
        # the honest party sent one commitment and, unless the commitment round failed, one opening
        check_eq(st0["phases"]["mask"]["msgs_sent"], 1 if name == "commitment truncated" else 2,
                 f"{name}: at most one commitment and one opening left the honest party")

    # (b) the transport fails before the message is sent (an exception the caller catches)
    class Broken:
        def __init__(self, inner):
            self._inner = inner
            self.armed = False

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def exchange(self, data):
            if self.armed and self._inner.phase in ("mask", "arith-open"):
                raise RuntimeError("transport failure before sending")
            return self._inner.exchange(data)

    for mode in ("shared", "wire"):
        def script(role, vf, mode=mode):
            g = vf.gate(0, spec_t if mode == "shared" else spec_z)
            wire = AShare(0, 0, vf.w, True)                   # a share of zero held by both
            if role == 1:
                return None
            vf.chan.armed = True
            first = lambda: vf.mask_shared(g, 3) if mode == "shared" else vf.mask_wire(g, wire)  # noqa: E731
            caught = silent(vf, first, RuntimeError)
            state = vf.registry._gates[g].state
            vf.chan.armed = False
            retry = silent(vf, first, (ProtocolAbort, GateUsedError))
            return caught, state, retry, vf.state

        out0, out1, st0, _, _ = run_drivers(script, k=k, wrap=(Broken, None),
                                            reserve=[pg.planned_gate(spec_t if mode == "shared" else spec_z)])
        check_eq(out0, ("ok", (True, USED, True, ABORTED)),
                 f"{mode} mode: after a caught failure the gate is USED, the retry is refused, nothing is sent")
        check("mask" not in st0["phases"] or st0["phases"]["mask"]["msgs_sent"] == 0,
              f"{mode} mode: no masked opening was sent at all")


# --------------------------------------------------------------------------- #
# Shared-input Mask: commit, then open
# --------------------------------------------------------------------------- #

def test_mask_shared_commitment():
    """The two unauthenticated shares are exchanged by commit-then-open (hashes.committed_exchange)."""
    n, k = 4, 32
    size = 1 << n
    spec = GateSpec.table(n, k, S, list(range(size)))
    x = 11
    xs = split(x, size)
    plan = [pg.planned_gate(spec, 2)]

    # (a) the bytes on the wire of an honest run (recorded by a transparent wrapper at P_1)
    def record(chan):
        return TamperChannel(chan, [])

    def honest(role, vf):
        g0, g1 = vf.gate_batch(5, spec, 2)
        hat = vf.mask_shared(g0, xs[role])
        share = (xs[role] + vf.registry.gate(g0).material.rp_share) % size
        hat1 = vf.mask_shared(g1, 0)
        return hat, share, vf.open(vf.apply(g0)), hat1, vf._sid_mask

    sid = b"test-mask-wire-" + secrets.token_bytes(4)
    r0, r1, st0, st1, drivers = run_ok(honest, k=k, wrap=(record, record), sid=sid, reserve=plan)
    check_eq((r0[0], r0[2]), (r1[0], x), "honest shared-input Mask: same x_hat, the gate evaluates T[x]")
    check_eq(r0[0], (r0[1] + r1[1]) % size, "x_hat is the sum of the two opened shares")
    check_eq((r0[4], r1[4]), (hs.sha256(b"Couplet-mask-sid", sid),) * 2,
             "sub-session identifier of Mask: SHA-256(frame('Couplet-mask-sid', sid))")
    for role, (r, vf) in enumerate(zip((r0, r1), drivers)):
        sent, got = vf.chan.sent, vf.chan.received
        com, opening = sent[("mask", 0)], sent[("mask", 1)]
        label = b"mask|" + (5).to_bytes(8, "little")
        msg = r[1].to_bytes(1, "little")
        check_eq((len(com), len(opening)), (32, 33), f"P{role}: 32-byte commitment, then opening || share")
        check_eq(opening[32:], msg, f"P{role}: the opened message is x_b + r_b mod 2^n, ceil(n/8) bytes")
        check_eq(com, hs.sha256(b"Couplet-com", r[4], label, bytes([role]), opening[:32], msg),
                 f"P{role}: commitment = SHA-256(frame('Couplet-com', sid_mask, 'mask|' || le64(gid), "
                 f"party, omega, share))")
        check(hs.verify(r[4], got[("mask", 0)], got[("mask", 1)][32:], got[("mask", 1)][:32],
                        party=1 - role, label=label), f"P{role}: the peer's opening verifies under the same label")
        other = b"mask|" + (6).to_bytes(8, "little")
        check(hs.verify(r[4], sent[("mask", 2)], sent[("mask", 3)][32:], sent[("mask", 3)][:32], party=role,
                        label=other) and not hs.verify(r[4], sent[("mask", 2)], sent[("mask", 3)][32:],
                                                       sent[("mask", 3)][:32], party=role, label=label),
              f"P{role}: the commitment of the next gate is bound to its own gid")
    check_eq((st0["phases"]["mask"]["rounds"], st0["phases"]["mask"]["bytes_sent"]), (4, 2 * 65),
             "two shared-input Masks: 2 rounds and 64 + ceil(n/8) bytes each")

    # (b) the rushing attack that the commitment prevents: P_1 waits for the honest share and
    #     then picks its own share so that x_hat becomes a target of its choice
    #     (the target differs from the honest x_hat: otherwise, with probability 1/N, the "chosen"
    #     share would be the committed one and the run would be an honest run)
    seen = {}

    def choose(own: bytes, incoming: bytes) -> bytes:
        peer = incoming[32]                                  # the honest share, read first
        target = 6 if (peer + own[32]) % size != 6 else 7
        seen["honest_share"] = peer
        seen["target"] = target
        seen["would_be_x_hat"] = (peer + (target - peer) % size) % size
        return own[:32] + bytes([(target - peer) % size])    # same opening randomness, another share

    def rushing(chan):
        return TamperChannel(chan, [Rush("mask", 1, choose)])

    def victim(role, vf):
        g = vf.gate(0, spec)
        try:
            vf.mask_shared(g, xs[role])
        except ProtocolAbort as exc:
            return (str(exc), vf.registry._gates[g].state, vf.registry._gates[g].x_hat, vf.state,
                    silent(vf, lambda: vf.mask_shared(g, xs[role]), (ProtocolAbort, GateUsedError)))
        return "x_hat accepted", None, vf.registry.gate(g).x_hat, vf.state, None

    out0, out1, st0, _, _ = run_drivers(victim, k=k, wrap=(None, rushing), reserve=[pg.planned_gate(spec)])
    check_eq(out0, ("ok", ("commitment opening failed", USED, None, ABORTED, True)),
             "a share chosen after seeing the honest share does not open the commitment: abort, no x_hat")
    check_eq(seen.get("would_be_x_hat"), seen.get("target", -1),
             "without the commitment the rushing party would have fixed x_hat to its target")
    check_eq(st0["phases"]["mask"]["msgs_sent"], 2, "the honest party sent one commitment and one opening")

    # the same attack with the party's own share kept (only the timing differs) is the honest run:
    # the commitment does not get in the way of a party that opens what it committed to
    def keep(own: bytes, incoming: bytes) -> bytes:
        return own

    out0, out1, _, _, _ = run_drivers(victim, k=k, reserve=[pg.planned_gate(spec)],
                                      wrap=(None, lambda chan: TamperChannel(chan, [Rush("mask", 1, keep)])))
    check(out0[0] == "ok" and out0[1][0] == "x_hat accepted" and out0[1][3] == ACTIVE,
          "a rushing party that opens its committed share is accepted (control)")

    # (c) the rushing party cannot adapt its COMMITMENT either: mirroring the honest commitment
    #     (and then the honest opening) is rejected because a commitment is bound to its sender
    for name, rules, want in (
            ("mirror the commitment and the opening", [Mirror("mask", 0), Mirror("mask", 1)],
             "mirrored commitment"),
            ("mirror the opening only", [Mirror("mask", 1)], "commitment opening failed")):
        out0, _, _, _, _ = run_drivers(victim, k=k, reserve=[pg.planned_gate(spec)],
                                       wrap=(None, lambda chan, rules=rules: TamperChannel(chan, rules)))
        check_eq(out0[1][:4], (want, USED, None, ABORTED), f"{name}: abort, no x_hat")

    # (d) replay of the party's own commitment and opening of an earlier gate: bound to the gid
    def two_gates(role, vf):
        g0, g1 = vf.gate_batch(0, spec, 2)
        first = vf.mask_shared(g0, xs[role])
        try:
            vf.mask_shared(g1, xs[role])
        except ProtocolAbort as exc:
            return first, str(exc), vf.state
        return first, "second x_hat accepted", vf.state

    def replaying(chan):
        return TamperChannel(chan, [Replay("mask", 2, ("mask", 0)), Replay("mask", 3, ("mask", 1))])

    out0, _, _, _, _ = run_drivers(two_gates, k=k, wrap=(None, replaying), reserve=plan)
    check(out0[0] == "ok" and out0[1][1:] == ("commitment opening failed", ABORTED),
          "a replayed commitment and opening of another gate is rejected (the label carries the gid)")
    NOTES.append("shared-input Mask: commit-then-open, 2 rounds, 64 + ceil(n/8) bytes per party; a share "
                 "chosen after the honest share was seen fails the opening (abort, no x_hat)")


# --------------------------------------------------------------------------- #
# Triple budget: one bucketing population per session
# --------------------------------------------------------------------------- #

def test_triple_budget_strict():
    """Strict mode is the default of the driver: reserve once, or get a clear error and no message."""
    n, k = 3, 32
    size = 1 << n
    spec = GateSpec.table(n, k, S, list(range(size)))
    key_and = pg.generation_circuit(cg.dpf_descriptor(n, k, S, "xor", "public_one")).stats()["n_and"]
    gate_and = pg.generation_circuit(spec.descriptor()).stats()["n_and"]
    om_and = pg.generation_circuit(pg.outmask_descriptor(k, S)).stats()["n_and"]

    def commands(role, vf):
        kid = vf.key(None, "dpf", n, 5 if role == 0 else 0)
        g = vf.gate(None, spec)
        vf.mask_shared(g, 3 if role == 0 else 0)
        doubled = vf.lin([(2, vf.apply(g))])                  # not exact: needs an output mask
        return vf.open(doubled), vf.open(vf.query(kid, 5)), vf.info()

    # (a) the budget methods of the driver equal the circuit statistics
    def budgets(role, vf):
        return (vf.key_budget("dpf", n), vf.key_budget("dpf", n, count=3),
                vf.key_budget("dcf", n, secret_payload=True, index_mode="add"),
                vf.gate_budget(spec), vf.gate_budget(spec.descriptor(), 4), vf.outmask_budget(),
                vf.outmask_budget(5), vf.outmask_budget(0), vf.allow_refill, vf.triple_requests)

    r0, r1, st0, _, _ = run_ok(budgets, k=k, setup=False)
    dcf_and = pg.generation_circuit(cg.dcf_descriptor(n, k, S, "add", "secret")).stats()["n_and"]
    check_eq(r0, (key_and, 3 * key_and, dcf_and, gate_and, 4 * gate_and, om_and, 5 * om_and, 0, False, 0),
             "key_budget / gate_budget / outmask_budget: AND gates of the circuit times the count")
    check_eq((r1, st0["bytes_sent"]), (r0, 0), "the budget is public: same numbers at both parties, no message")

    # the defaults themselves: the driver is strict, the session and engine classes below it refill
    class _Chan:
        party = 0
        phase = "x"

    plain = AuthEval(0, _Chan(), b"defaults", prep="ideal", dealer=IdealDealer())
    check_eq((plain.allow_refill, plain.session.allow_refill, plain.session.engine.allow_refill),
             (False, False, False), "AuthEval() without the argument is strict, down to its engine")
    loose = AuthEval(0, _Chan(), b"defaults", prep="ideal", dealer=IdealDealer(), allow_refill=True)
    check_eq((loose.allow_refill, loose.session.engine.allow_refill), (True, True), "allow_refill=True is passed down")
    check(raises(ValueError, lambda: AuthEval(0, _Chan(), b"defaults", prep="ideal", dealer=IdealDealer(),
                                             allow_refill="yes")), "allow_refill must be a bool")
    check_eq(om_and, pg.outmask_and_count(k, S), "one output mask costs the output-mask circuit")

    # (b) exact reservation: ONE request, nothing left, and the helper predicts its size
    r0, r1, _, _, _ = run_ok(commands, k=k, reserve=key_and + gate_and, outmasks=1)
    check_eq((r0[0], r0[1], r1[0], r1[1]), (6, 1, 6, 1), "strict session on the exact reservation")
    info = r0[2]
    check_eq((info["triple_requests"], info["triples_unused"], info["triples_requested"], info["allow_refill"]),
             (1, 0, S + key_and + gate_and + om_and, False), "one request: setup circuit + keys + gates + masks")
    check_eq(info["triples_requested"], pg.session_triples(
        [pg.planned_key("dpf", n, k, S), pg.planned_gate(spec), pg.planned_outmasks(k, S, 1)], S),
        "protocol.session_triples predicts the size of the one request")

    # (c) no reservation in strict mode: every generation is refused before any message and before
    #     any change of state; the session stays ACTIVE and usable
    def refused(role, vf):
        facts = {}
        before = vf.registry.counters()["next"]

        def attempt(fn):
            chan = vf.chan
            sent = (chan.msgs_sent, chan.bytes_sent, chan.msgs_recv)
            try:
                fn()
            except TripleBudgetError as exc:
                quiet = (chan.msgs_sent, chan.bytes_sent, chan.msgs_recv) == sent
                return quiet, (exc.needed, exc.available, isinstance(exc, ValueError), "reserve" in str(exc))
            return False, "ran"

        for name, fn in (("key", lambda: vf.key(None, "dpf", n, 0)),
                         ("key_batch", lambda: vf.key_batch(None, "dpf", n, [0, 1])),
                         ("gate", lambda: vf.gate(None, spec)),
                         ("gate_batch", lambda: vf.gate_batch(None, spec, 2)),
                         ("outmasks", lambda: vf.reserve_outmasks(2))):
            facts[name] = attempt(fn)
        facts["ids"] = vf.registry.counters()["next"] == before
        facts["registry"] = vf.registry.summary()
        facts["bad_id_first"] = silent(vf, lambda: vf.key(-1, "dpf", n, 0), ValueError) and not silent(
            vf, lambda: vf.key(-1, "dpf", n, 0), TripleBudgetError)
        facts["state"] = (vf.state, vf.triple_requests, vf.outmasks_available)
        return facts

    r0, r1, _, _, _ = run_ok(refused, k=k)
    exp = {"key": key_and, "key_batch": 2 * key_and, "gate": gate_and, "gate_batch": 2 * gate_and,
           "outmasks": 2 * om_and}
    for r in (r0, r1):
        for name, need in exp.items():
            check_eq(r[name], (True, (need, 0, True, True)),
                     f"strict, no reservation: {name} raises TripleBudgetError and sends nothing")
        check(r["ids"], "a refused generation consumes no identifier")
        check_eq((r["registry"]["keys"], r["registry"]["gates"]),
                 ({GENERATING: 0, CERTIFIED: 0}, {GENERATING: 0, CERTIFIED: 0, USED: 0}),
                 "a refused generation leaves no record behind")
        check(r["bad_id_first"], "a bad identifier is reported before the budget")
        check_eq(r["state"], (ACTIVE, 1, 0), "the session is ACTIVE with its one population")

    # (d) one triple too few for the output mask of Open: refused, nothing sent, the share can
    #     still be opened after an explicit (counted) reservation
    def short_mask(role, vf):
        kid = vf.key(None, "dpf", n, 5 if role == 0 else 0)
        doubled = vf.lin([(2, vf.query(kid, 5))])
        pending = vf.checker.pending
        try:
            vf.open(doubled)
            first = "opened"
        except TripleBudgetError as exc:
            first = (exc.needed, exc.available)
        quiet = silent(vf, lambda: vf.open(doubled), TripleBudgetError)
        exact_ok = vf.open(vf.query(kid, 5))                  # exact values need no mask
        vf.session.reserve_triples(1)                         # the explicit way to add a population
        value = vf.open(doubled)
        return first, quiet, exact_ok, value, vf.triple_requests, vf.state, pending

    r0, r1, _, _, _ = run_ok(short_mask, k=k, reserve=key_and + om_and - 1, requests=2)
    check_eq((r0, r1), (((om_and, om_and - 1), True, 1, 2, 2, ACTIVE, 0),) * 2,
             "strict: Open of a non-exact value without a reserved mask is refused before any message; "
             "an explicit reserve_triples is a second, counted population")

    # (e) refill mode: no reservation needed, every circuit execution is one more population
    r0, r1, _, _, _ = run_ok(commands, k=k, allow_refill=True)
    check_eq((r0[0], r0[1]), (6, 1), "refill mode: the same commands run without a reservation")
    check_eq((r0[2]["triple_requests"], r1[2]["triple_requests"], r0[2]["allow_refill"]), (4, 4, True),
             "refill mode: setup, key, gate and output mask are four populations, and info() says so")
    NOTES.append(f"triple budget (n = {n}, k = {k}): key {key_and:,}, table gate {gate_and:,}, output mask "
                 f"{om_and:,} AND triples; strict session: 1 request; refill mode: 4 requests")


# --------------------------------------------------------------------------- #
# State machine, arguments, abort
# --------------------------------------------------------------------------- #

def test_open_order():
    """``Open`` step by step: (1) ArithCheck on the deferred openings, (2) output masks
    (generated by Pi_Gen if the pool is short), (3) the opening, (4) its ArithCheck."""
    n, k = 4, 32
    size, mod_k = 1 << n, 1 << k
    table = [RNG.randrange(mod_k) for _ in range(size)]
    x = RNG.randrange(size)
    xs = split(x, size)
    spec_t = GateSpec.table(n, k, S, table)
    spec_c = GateSpec.comparison(n, k, S, size // 2, auth_input=True)
    plan = [pg.planned_gate(spec_t), pg.planned_gate(spec_c)]
    engine_phases = {"input", "online", "outcheck", "output"}

    def prepare(vf, role):
        """Two gates; a non-exact wire is masked into the second one (a deferred opening)."""
        g1 = vf.gate(None, spec_t)
        g2 = vf.gate(None, spec_c)
        vf.mask_shared(g1, xs[role])
        s1 = vf.lin([(3, vf.apply(g1))], 5)                    # not exact
        x_hat = vf.mask_wire(g2, s1)                           # opened, not yet verified
        s2 = vf.lin([(7, vf.apply(g2))], 1)                    # not exact: Open needs an output mask
        return s2, x_hat

    def sent_phases(vf, mark):
        return [phase for phase, _ in list(vf.chan.sent)[mark:]]

    # (a) honest run with an empty mask pool: check, then mask generation, then opening and check
    def honest(role, vf):
        s2, _ = prepare(vf, role)
        before = (vf.checker.pending, vf.outmasks_available, vf.checker.checks,
                  vf.session.engine.evaluations)
        mark = len(vf.chan.sent)
        value = vf.open(s2)
        return (value, before, sent_phases(vf, mark), vf.checker.checks, vf.checker.pending,
                vf.session.engine.evaluations, vf.outmasks_available)

    wrap = (TamperChannel, TamperChannel)
    r0, r1, _, _, _ = run_ok(honest, k=k, reserve=plan, outmasks=1, wrap=wrap)
    inner = (3 * table[x] + 5) % mod_k
    want = (7 * int(inner % size < size // 2) + 1) % mod_k
    check_eq((r0[0], r1[0]), (want, want), "Open of a non-exact value behind a wire-mode Mask")
    check_eq(r0[1], (1, 0, 0, 3), "before Open: one deferred opening, no output mask, no check yet")
    phases = r0[2]
    check_eq(phases[:4], ["arith-check"] * 4, "Open starts with the ArithCheck on the deferred opening")
    check_eq(phases[-5:], ["arith-open"] + ["arith-check"] * 4, "Open ends with the opening and its check")
    middle = phases[4:-5]
    check(len(middle) > 6 and middle[0] == "input" and set(middle) == engine_phases,
          f"the output mask is generated between the two checks ({sorted(set(middle))})")
    check_eq(r0[3:], (2, 0, 4, 0), "two checks, nothing pending, one more circuit evaluation, mask used")
    check_eq(r1[2], phases, "both parties send in the same order")

    # (b) strict mode: without triples for the mask, Open is refused before the first check
    def refused(role, vf):
        s2, _ = prepare(vf, role)
        quiet = silent(vf, lambda: vf.open(s2), TripleBudgetError)
        facts = (quiet, vf.checker.pending, vf.checker.checks, vf.state)
        vf.session.reserve_triples(vf.outmask_budget(1))       # an explicit second population
        return vf.open(s2), facts

    r0, r1, _, _, _ = run_ok(refused, k=k, reserve=plan, requests=2)
    check_eq((r0, r1), ((want, (True, 1, 0, ACTIVE)),) * 2,
             "a mask generation beyond the budget is refused before any message; the opening stays pending")

    # (c) a wrong share in the wire-mode Mask: the first check of Open rejects and nothing else
    #     is sent, in particular no output mask is generated for the faulty session
    def cheated(role, vf):
        s2, x_hat = prepare(vf, role)
        evaluations = vf.session.engine.evaluations
        mark = len(vf.chan.sent)
        try:
            vf.open(s2)
        except ProtocolAbort as exc:
            return (str(exc), x_hat, sent_phases(vf, mark), vf.session.engine.evaluations - evaluations,
                    _dead_facts(vf))
        return "ok", x_hat, None, None, None

    factory = CheatingChecker.factory(value_errors={0: 1})     # opening 0 is the wire-mode Mask
    out0, out1, _, _, _ = run_drivers(cheated, k=k, reserve=plan, outmasks=1, wrap=wrap,
                                      factories=(None, factory))
    check_eq(out0[0], "ok", "the honest party returns from the failed Open")
    msg, x_hat, phases, evaluations, dead = out0[1]
    check_eq(msg, "ArithCheck failed", "a shifted masked input is rejected by the next Open")
    check_eq(out1[1][1], x_hat, "until the check both parties hold the same (shifted) masked input")
    check_eq(phases, ["arith-check"] * 4, "the honest party sends the four messages of the check only")
    check_eq(evaluations, 0, "no output mask is generated after a wrong deferred opening")
    check_eq(dead, (True, True), "the session is dead and silent afterwards")


def test_state_and_arguments():
    n, k = 3, 32
    size = 1 << n

    def script(role, vf):
        facts = {}
        errs = {
            "gate descriptor": lambda: vf.gate(0, {"kind": "gate", "n": 3}),
            "gate descriptor kind": lambda: vf.gate(0, {"kind": "dpf", "n": 3, "k": k, "s": S}),
            "family": lambda: vf.key(0, "xyz", n, 0),
            "n": lambda: vf.key(0, "dpf", 0, 0),
            "alpha range": lambda: vf.key(0, "dpf", n, size),
            "alpha type": lambda: vf.key(0, "dpf", n, 1.5),
            "beta range": lambda: vf.key(0, "dpf", n, 0, 1 << k),
            "index mode": lambda: vf.key(0, "dpf", n, 0, index_mode="mul"),
            "batch sizes": lambda: vf.key_batch(0, "dpf", n, [0, 1], [1]),
            "empty batch": lambda: vf.key_batch(0, "dpf", n, []),
            "gate spec type": lambda: vf.gate(0, {"kind": "gate"}),
            "gate k": lambda: vf.gate(0, GateSpec.zero_test(n, 16, S)),
            "gate count": lambda: vf.gate_batch(0, GateSpec.zero_test(n, k, S), 0),
            "lin term": lambda: vf.lin([(1, 5)]),
            "lin coefficient": lambda: vf.lin([(1.5, AShare(0, 0, vf.w))]),
            "lin const": lambda: vf.lin([], 0.5),
            "lin modulus": lambda: vf.lin([(1, AShare(0, 0, 8))]),
            "open type": lambda: vf.open(5),
            "open modulus": lambda: vf.open(AShare(0, 0, vf.w - 1, True)),
            "outmask count": lambda: vf.reserve_outmasks(-1),
            "key budget family": lambda: vf.key_budget("xyz", n),
            "gate budget count": lambda: vf.gate_budget(GateSpec.zero_test(n, k, S), 0),
            "outmask budget count": lambda: vf.outmask_budget(-1),
        }
        facts["value_errors"] = {name: silent(vf, fn, ValueError) for name, fn in errs.items()}
        facts["setup_twice"] = silent(vf, lambda: vf.setup(), StateError)
        facts["unknown_key"] = silent(vf, lambda: vf.query(0, 0), StateError)
        kid = vf.key(3, "dpf", n, 5 if role == 0 else 0)
        facts["kid"] = kid
        facts["reuse_kid"] = silent(vf, lambda: vf.key(3, "dpf", n, 0), StateError)
        facts["smaller_kid"] = silent(vf, lambda: vf.key(1, "dpf", n, 0), StateError)
        facts["query_range"] = silent(vf, lambda: vf.query(kid, size), ValueError)
        gid = vf.gate(10, GateSpec.zero_test(n, k, S))
        facts["reuse_gid"] = silent(vf, lambda: vf.gate(10, GateSpec.zero_test(n, k, S)), StateError)
        facts["smaller_gid"] = silent(vf, lambda: vf.gate(2, GateSpec.zero_test(n, k, S)), StateError)
        facts["next_gid"] = vf.gate(None, gates.offset_descriptor(n, k, S))   # a plain descriptor: mask only
        facts["offset_only_apply"] = (vf.mask_wire(facts["next_gid"], vf.query(kid, 5)) is not None
                                      and silent(vf, lambda: vf.apply(facts["next_gid"]), StateError))
        facts["state"] = vf.state
        facts["open"] = vf.open(vf.query(kid, 5))            # the session still works
        facts["empty"] = (vf.open_many([]), vf.lin([]).value, vf.lin([]).exact)
        vf.reserve_outmasks(0)
        return facts

    r0, r1, _, _, _ = run_ok(script, k=k, reserve=[
        pg.planned_key("dpf", n, k, S), pg.planned_gate(GateSpec.zero_test(n, k, S)),
        pg.planned_gate(gates.offset_descriptor(n, k, S))])
    for r in (r0, r1):
        for name, ok in r["value_errors"].items():
            check(ok, f"invalid argument ({name}): ValueError, nothing sent")
        for name in ("setup_twice", "unknown_key", "reuse_kid", "smaller_kid", "query_range", "reuse_gid",
                     "smaller_gid", "offset_only_apply"):
            check(r[name], f"{name}: refused, nothing sent")
        check_eq((r["kid"], r["next_gid"], r["state"], r["open"]), (3, 11, ACTIVE, 1),
                 "identifiers are monotone and argument errors do not abort the session")
        check_eq(r["empty"], ([], 0, False), "empty Open and empty Lin")

    # commands before setup
    def before_setup(role, vf):
        bad_setup = [silent(vf, fn, ValueError) for fn in (
            lambda: vf.setup(-1), lambda: vf.setup(0, -1), lambda: vf.setup(True), lambda: vf.setup(1.5))]
        if not all(bad_setup):
            raise AssertionError("an invalid reservation must be a ValueError before any message")
        names = {"key": lambda: vf.key(0, "dpf", n, 0), "gate": lambda: vf.gate(0, GateSpec.zero_test(n, k, S)),
                 "query": lambda: vf.query(0, 0), "mask": lambda: vf.mask_shared(0, 0),
                 "apply": lambda: vf.apply(0), "lin": lambda: vf.lin([]),
                 "open": lambda: vf.open(AShare(0, 0, vf.w, True)),
                 "outmasks": lambda: vf.reserve_outmasks(1)}
        return {name: silent(vf, fn, StateError) for name, fn in names.items()}, vf.state

    r0, r1, st0, _, _ = run_ok(before_setup, k=k, setup=False)
    check(all(r0[0].values()) and all(r1[0].values()), "every command before setup raises StateError")
    check_eq((r0[1], st0["bytes_sent"]), (ACTIVE, 0), "nothing was sent before setup")
    check(raises(ValueError, lambda: AuthEval(0, type("C", (), {"party": 1})(), b"sid")), "role / channel mismatch")


def _dead_facts(vf) -> tuple:
    """(session dead, every later command raises ProtocolAbort and sends nothing)."""
    n, k = 3, vf.k
    cmds = [lambda: vf.setup(), lambda: vf.key(None, "dpf", n, 0),
            lambda: vf.gate(None, GateSpec.zero_test(n, k, vf.s)), lambda: vf.query(0, 0),
            lambda: vf.query_all(0), lambda: vf.mask_shared(0, 0),
            lambda: vf.mask_wire(0, AShare(0, 0, vf.w, True)), lambda: vf.apply(0),
            lambda: vf.lin([]), lambda: vf.open(AShare(0, 0, vf.w, True)),
            lambda: vf.reserve_outmasks(1)]
    quiet = all(silent(vf, cmd, ProtocolAbort) for cmd in cmds)
    ses = vf.session
    dead = (vf.state == ABORTED and vf.registry.state == ABORTED and ses.state == ABORTED
            and ses.delta_share is None and ses.delta_wires is None and ses.engine.dead
            and (vf.checker is None or (vf.checker.dead and vf.checker._delta is None))
            and all(r.key is None for r in vf.registry._keys.values())
            and all(r.material is None for r in vf.registry._gates.values())
            and vf.outmasks_available == 0)
    return dead, quiet


def test_abort_is_absorbing():
    n, k = 3, 32
    size = 1 << n

    # (a) explicit Abort
    def explicit(role, vf):
        kid = vf.key(0, "dpf", n, 3 if role == 0 else 0)
        share = vf.query(kid, 3)
        vf.reserve_outmasks(2)
        vf.abort()
        return _dead_facts(vf), raises(ProtocolAbort, lambda: vf.open(share))

    r0, r1, _, _, _ = run_ok(explicit, k=k, reserve=[pg.planned_key("dpf", n, k, S)], outmasks=2)
    check(r0[0] == (True, True) and r1[0] == (True, True), "explicit Abort: the session is dead and silent")
    check(r0[1], "a sharing handed out before the abort cannot be opened")

    # (b) the peer opens a wrong value: the honest party aborts, returns nothing, and is dead
    def cheated(role, vf):
        kid = vf.key(0, "dpf", n, 3 if role == 0 else 0, 9 if role == 0 else 0)
        share = vf.query(kid, 3)
        try:
            value = vf.open(share)
        except ProtocolAbort as exc:
            return str(exc), None, _dead_facts(vf)
        return "ok", value, None

    factory = CheatingChecker.factory(value_errors={0: 1})
    out0, out1, _, _, _ = run_drivers(cheated, k=k, factories=(None, factory),
                                      reserve=[pg.planned_key("dpf", n, k, S, "xor", "secret")])
    check_eq(out0, ("ok", ("ArithCheck failed", None, (True, True))),
             "wrong opening by the peer: abort, no output, dead and silent session")

    # (c) the output check of Pi_Gen fails: the key is never CERTIFIED (change F)
    def gen_cheated(role, vf):
        if role == 1:
            vf.session.engine.cheat = {"flip_public": ("cwout_v", 3, 0)}
        try:
            vf.key(0, "dpf", n, 3 if role == 0 else 0)
        except ProtocolAbort as exc:
            return str(exc), vf.registry.summary()["keys"], vf.registry._keys[0].state, _dead_facts(vf)
        return "ok", None, None, None

    out0, _, _, _, _ = run_drivers(gen_cheated, k=k, reserve=[pg.planned_key("dpf", n, k, S)])
    check_eq(out0, ("ok", ("BoolCheck failed", {GENERATING: 1, CERTIFIED: 0}, GENERATING, (True, True))),
             "failed output check: the key record stays GENERATING and the session is dead")

    # (d) the same for a gate: no material is certified
    def gate_cheated(role, vf):
        if role == 1:
            vf.session.engine.cheat = {"flip_private": ("dpf.root0", 5, 0)}
        try:
            vf.gate(0, GateSpec.zero_test(n, k, S))
        except ProtocolAbort as exc:
            return str(exc), vf.registry.summary()["gates"], _dead_facts(vf)
        return "ok", None, None

    out0, _, _, _, _ = run_drivers(gate_cheated, k=k, reserve=[pg.planned_gate(GateSpec.zero_test(n, k, S))])
    check_eq(out0, ("ok", ("BoolCheck failed", {GENERATING: 1, CERTIFIED: 0, USED: 0}, (True, True))),
             "failed output check of a gate: nothing is certified")

    # (e) a wrong share in shared-input mode is an input substitution, not a detectable deviation
    x = 5
    shift = 3

    def substituted(role, vf, peer_share=shift):
        g = vf.gate(0, GateSpec.table(n, k, S, list(range(size))))
        vf.mask_shared(g, x if role == 0 else peer_share)
        return vf.open(vf.apply(g))

    table_plan = [pg.planned_gate(GateSpec.table(n, k, S, list(range(size))))]
    out0, out1, _, _, _ = run_drivers(substituted, k=k, reserve=table_plan)
    check_eq((out0, out1), (("ok", (x + shift) % size),) * 2,
             "shared-input mode: a shifted share IS the input x + shift for both parties (by design)")

    # (f) a peer that commits to and opens a shifted share but keeps evaluating at its own x_hat
    #     holds a wrong share of the gate output: the release is rejected (whatever x is)
    shifted = deviating_mask(lambda m: bytes([(m[0] + shift) % size]))
    out0, out1, _, _, _ = run_drivers(lambda role, vf: substituted(role, vf, 0), k=k, classes=(None, shifted),
                                      reserve=table_plan)
    check_eq(out0, ("abort", "ArithCheck failed"),
             "inconsistent views of x_hat: the peer's share is wrong and Open aborts")


def test_restart_on_persisted_counters():
    n, k = 3, 32
    size = 1 << n
    tmp = tempfile.mkdtemp(prefix="couplet-test-driver-")
    paths = tuple(os.path.join(tmp, f"p{b}.json") for b in (0, 1))
    sid = b"restart-" + secrets.token_bytes(4)
    spec = GateSpec.table(n, k, S, list(range(size)))
    try:
        def first(role, vf):
            g = vf.gate_batch(0, spec, 3)
            vf.mask_shared(g[0], 4 if role == 0 else 0)
            return g, vf.registry.counters()

        a0, a1, _, _, _ = run_ok(first, k=k, state_paths=paths, sid=sid, reserve=[pg.planned_gate(spec, 3)])
        check_eq(a0[0], [0, 1, 2], "first process: gates 0, 1, 2")

        def second(role, vf):
            facts = {
                "old used": silent(vf, lambda: vf.mask_shared(0, 1), StateError),
                "old unused": silent(vf, lambda: vf.mask_shared(1, 1), StateError),
                "old id again": silent(vf, lambda: vf.gate(2, spec), StateError),
                "old apply": silent(vf, lambda: vf.apply(0), StateError),
            }
            g = vf.gate(None, spec)
            vf.mask_shared(g, 6 if role == 0 else 0)
            return facts, g, vf.open(vf.apply(g)), vf.registry.counters()

        b0, b1, _, _, _ = run_ok(second, k=k, state_paths=paths, sid=sid, reserve=[pg.planned_gate(spec)])
        for name, ok in b0[0].items():
            check(ok, f"after the restart ({name}): refused, nothing sent")
        check(b0[1] >= a0[1]["reserved"]["gid"] > 2, "the new gate identifier is above everything issued before")
        check_eq((b0[2], b0[3]["starts"], b1[1]), (6, 2, b0[1]), "the restarted session works with fresh material")

        # a persisted ABORTED session cannot be continued
        def third(role, vf):
            vf.abort()
            return True

        run_ok(third, k=k, state_paths=paths, sid=sid)

        def fourth(role, vf):
            return vf.state, raises(ProtocolAbort, lambda: vf.setup())

        c0, c1, st0, _, _ = run_ok(fourth, k=k, state_paths=paths, sid=sid, setup=False)
        check_eq((c0, st0["bytes_sent"]), ((ABORTED, True), 0), "an aborted session stays aborted after a restart")
    finally:
        for name in os.listdir(tmp):
            os.remove(os.path.join(tmp, name))
        os.rmdir(tmp)


# --------------------------------------------------------------------------- #
# Carry-leak regression (change D)
# --------------------------------------------------------------------------- #

def chi2_critical(df: int, z: float = 4.753) -> float:
    """Upper critical value of chi-square with ``df`` degrees of freedom (Wilson-Hilferty; p = 1e-6)."""
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def chi2_uniform(samples: list, bins: int) -> float:
    counts = [0] * bins
    for v in samples:
        counts[v] += 1
    expected = len(samples) / bins
    return sum((c - expected) ** 2 / expected for c in counts)


def wire_openings(n: int, k: int, s: int, x: int, count: int, batch: int, naive_runs: int = 0):
    """``count`` masked openings of a wire holding ``x`` (wire mode), each under a fresh gate.

    Returns (public opened values, the offsets r' reconstructed from both
    parties' material (test only), opened value of the wire, naive openings).
    With ``naive_runs`` the NAIVE variant is also run through the real opening
    code: the wire is masked with a dealer-made authenticated sharing of
    ``r = r' mod 2^n`` zero-extended in Z_{2^(n+s)}, and the MAC check passes.
    """
    size = 1 << n
    alpha = RNG.randrange(size)
    d = (RNG.getrandbits(s), RNG.getrandbits(s))
    delta = d[0] + d[1]
    naive_r = [RNG.randrange(size) for _ in range(naive_runs)]
    naive_shares = [deal(r, delta, n + s, rng=RNG) for r in naive_r]
    spec = GateSpec.offset_only(n, k, s, auth_input=True)

    def script(role, vf):
        kid = vf.key(0, "dpf", n, alpha if role == 0 else 0, x if role == 0 else 0)
        wire = vf.query(kid, alpha)                            # [[x]]
        opened, rps = [], []
        for start in range(0, count, batch):
            gids = vf.gate_batch(None, spec, min(batch, count - start))
            for g in gids:
                vf.mask_wire(g, wire)
                rec = vf.registry.gate(g)
                opened.append(rec.opened)
                rps.append(rec.material.rp_share)
        naive = [vf.checker.open_deferred(wire.reduce(n, s) + sh[role]) for sh in naive_shares]
        pending = vf.checker.pending
        value = vf.open(wire)                                  # ArithCheck over ALL deferred openings
        return opened, rps, value, naive, pending

    r0, r1, _, _, _ = run_ok(script, k=k, s=s, delta_shares=d, reserve=[
        pg.planned_key("dpf", n, k, s, "xor", "secret"), pg.planned_gate(spec, count)])
    check_eq(r0[0], r1[0], "both parties see the same openings")
    check_eq((r0[2], r1[2]), (x, x), "the wire holds x and the deferred checks pass")
    check_eq(r0[4], count + naive_runs, "all masked openings were deferred to the final check")
    rps = [(a + b) % (1 << (n + s)) for a, b in zip(r0[1], r1[1])]
    return r0[0], rps, naive_r, r0[3]


def carry_leak_experiment(n: int, k: int, s: int, count: int, batch: int, full_ring: bool, naive_runs: int):
    size = 1 << n
    ring = 1 << (n + s)
    worlds = {0: 0, 1: size - 1}
    real, naive_same, naive_proto = {}, {}, {}
    for w, x in worlds.items():
        opened, rps, naive_r, naive_opened = wire_openings(n, k, s, x, count, batch, naive_runs)
        check(all((v - rp) % ring == x for v, rp in zip(opened, rps)),
              f"x = {x}: every opening is x + r' in Z_(2^(n+s))")
        check(len(set(rps)) > count * 0.9 or ring < 4 * count, "the offsets are fresh")
        real[w] = opened
        # the naive variant ON THE SAME SAMPLES: mask with r = r' mod N (zero-extended)
        naive_same[w] = [x + (rp % size) for rp in rps]
        naive_proto[w] = naive_opened
        check_eq(naive_opened, [x + r for r in naive_r],
                 "naive variant through the real opening code: opens x + r without reduction, MAC check passes")
        check(all(v >> n == (1 if v >= size else 0) for v in naive_opened), "naive openings lie in [x, x + N)")
    crit = chi2_critical(ring - 1) if full_ring else None
    stats = {}
    for w in worlds:
        if full_ring:
            chi = chi2_uniform(real[w], ring)
            check(chi < crit, f"world {w}: openings uniform over the whole ring (chi2 = {chi:.1f} < {crit:.1f})")
            stats[f"chi2_ring_world{w}"] = chi
        low = chi2_uniform([v % (2 * size) for v in real[w]], 2 * size)      # x_hat and the carry bit
        top = chi2_uniform([v >> (n + s - 4) for v in real[w]], 16)
        check(low < chi2_critical(2 * size - 1), f"world {w}: low n + 1 bits uniform (chi2 = {low:.1f})")
        check(top < chi2_critical(15), f"world {w}: top 4 bits uniform (chi2 = {top:.1f})")
        stats[f"chi2_low_world{w}"] = low
        stats[f"chi2_top_world{w}"] = top
        # the naive values are nowhere near uniform
        chi_naive = chi2_uniform([v % (2 * size) for v in naive_same[w]], 2 * size)
        check(chi_naive > 10 * chi2_critical(2 * size - 1), f"world {w}: naive openings are far from uniform")
    if full_ring:                                             # homogeneity of the two worlds
        c0, c1 = [0] * ring, [0] * ring
        for v in real[0]:
            c0[v] += 1
        for v in real[1]:
            c1[v] += 1
        homog = sum((a - b) ** 2 / (a + b) for a, b in zip(c0, c1) if a + b)
        check(homog < crit, f"the two worlds have the same distribution (chi2 = {homog:.1f} < {crit:.1f})")
        stats["chi2_homogeneity"] = homog

    def advantage(samples: dict) -> float:
        """Distinguisher 'the carry bit (bit n) is set': |Pr[1 | x = N-1] - Pr[1 | x = 0]|."""
        rate = [sum(1 for v in samples[w] if (v >> n) & 1) / len(samples[w]) for w in (0, 1)]
        return abs(rate[1] - rate[0])

    def advantage_ge(samples: dict) -> float:
        """Distinguisher 'the opened value is >= N' (the same event for the naive variant)."""
        rate = [sum(1 for v in samples[w] if v >= size) / len(samples[w]) for w in (0, 1)]
        return abs(rate[1] - rate[0])

    adv_naive, adv_real = advantage(naive_same), advantage(real)
    sd = math.sqrt(0.5 / count)                               # generous bound on the sampling error
    check(abs(adv_naive - (1 - 1 / size)) < 5 * sd, f"naive variant: advantage {adv_naive:.3f} is about 1 - 1/N")
    check(adv_real < 5 * sd, f"real protocol: the same distinguisher has advantage {adv_real:.4f} (noise)")
    check(advantage_ge(real) < 5 * sd and advantage_ge(naive_same) > 0.8, "the >= N distinguisher")
    if naive_runs:
        adv_proto = advantage(naive_proto)
        check(adv_proto > 0.6, f"naive variant run through the opening code: advantage {adv_proto:.3f}")
        stats["naive_protocol_advantage"] = adv_proto
    stats.update(naive_advantage=adv_naive, real_advantage=adv_real, samples_per_world=count)
    return stats


def test_carry_leak_small_ring():
    """n = 3, s = 6: the ring Z_{2^(n+s)} has 512 elements; 20000 openings for x = 0 and for x = N - 1."""
    st = carry_leak_experiment(n=3, k=8, s=6, count=20000, batch=20000, full_ring=True, naive_runs=300)
    NOTES.append(
        f"carry leak, n = 3, s = 6 (ring of 512 elements), {st['samples_per_world']} wire-mode openings per "
        f"input: chi-square over the whole ring {st['chi2_ring_world0']:.1f} (x = 0) and "
        f"{st['chi2_ring_world1']:.1f} (x = N - 1), homogeneity of the two worlds {st['chi2_homogeneity']:.1f} "
        f"(df 511, critical value {chi2_critical(511):.1f} at p = 1e-6); carry-bit distinguisher: advantage "
        f"{st['real_advantage']:.4f} on the protocol's openings, {st['naive_advantage']:.4f} on the naive "
        f"variant (r instead of r') computed from the same offsets, {st['naive_protocol_advantage']:.3f} on "
        f"300 naive openings per world run through the opening code (their MAC checks pass)")


def test_carry_leak_s64():
    """The production parameter s = 64 (ring of 2^67 elements): projections of the openings."""
    st = carry_leak_experiment(n=3, k=64, s=64, count=2000, batch=2000, full_ring=False, naive_runs=100)
    NOTES.append(
        f"carry leak, n = 3, s = 64, {st['samples_per_world']} wire-mode openings per input: chi-square of the "
        f"low n + 1 bits {st['chi2_low_world0']:.1f} / {st['chi2_low_world1']:.1f} (df 15) and of the top 4 "
        f"bits {st['chi2_top_world0']:.1f} / {st['chi2_top_world1']:.1f} (df 15, critical value "
        f"{chi2_critical(15):.1f}); carry-bit distinguisher: advantage {st['real_advantage']:.4f} on the "
        f"protocol's openings, {st['naive_advantage']:.4f} on the naive variant on the same offsets")


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    if "-k" in sys.argv:
        key = sys.argv[sys.argv.index("-k") + 1]
        tests = [(name, fn) for name, fn in tests if key in name]
    failed = []
    start = time.perf_counter()
    print(f"Pi_AuthEval end-to-end tests, {len(tests)} test functions "
          f"(real preprocessing: {'no' if NO_REAL else 'yes'})")
    for name, fn in tests:
        before = CHECKS
        t = time.perf_counter()
        try:
            fn()
        except Exception:  # noqa: BLE001 - report every failure and continue
            failed.append(name)
            print(f"  FAIL {name}")
            traceback.print_exc()
            continue
        print(f"  ok   {name:<40} {CHECKS - before:>7} checks {time.perf_counter() - t:7.1f} s", flush=True)
    elapsed = time.perf_counter() - start
    for note in NOTES:
        print("  note:", note)
    if failed:
        print(f"FAILED: {len(failed)} of {len(tests)} tests ({', '.join(failed)}); {CHECKS} checks run, {elapsed:.1f} s")
        return 1
    print(f"TOTAL CHECKS: {CHECKS}  ({len(tests)} tests, all passed, {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
