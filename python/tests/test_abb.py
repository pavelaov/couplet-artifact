"""Tests for the authenticated Boolean engine (couplet.abb; DESIGN.md 2.2, 5.1, 7.6).

Covered, with the ideal dealer AND with the real preprocessing:
  * evaluation plans: every circuit is evaluated in plaintext through its
    compiled plan and compared with ``circuit.evaluate`` (independent of the
    protocol); invalid circuits are rejected;
  * honest evaluations for batch in {1, 7}: adders, multipliers, multiplexers,
    NOT gates, constant and input wires as outputs, random gate lists, a
    circuit without AND gates and one without opened outputs, compared bit for
    bit with the plaintext evaluator; private outputs reach only their owner
    and the public opening of a private wire is masked;
  * BOUND inputs and RETAIN outputs chained across three executions, including
    a retained block of batch 1 shared by all instances of a later execution;
    MAC invariants of retained wires;
  * exact round and byte accounting of an evaluation; the per-phase counters
    sum to the totals;
  * deviations injected through the engine's test hooks: a flipped ``d`` bit of
    an AND opening, a flipped share of a public output, a flipped share of a
    masked private output (each must raise ``ProtocolAbort`` at the honest
    party, which returns nothing and is dead afterwards); a wrong input-mask
    difference is an input substitution: no abort, both parties compute on the
    substituted input;
  * header mismatches (different circuits, different bound wires), malformed
    messages, argument errors that must NOT abort, the triple pool.

Two parties are run with ``couplet.net.run_two_party`` (two threads of this
process).  Run as a script from artifact/python:  python tests/test_abb.py
Prints the number of checks and exits non-zero on any failure.  ``-k name``
selects test functions; ``--no-real`` skips the runs with real preprocessing.
"""

from __future__ import annotations

import os
import random
import secrets
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import numpy as np  # noqa: E402

from couplet import abb, fss  # noqa: E402
from couplet import circuits_gen as cg  # noqa: E402
from couplet import gf128 as gf  # noqa: E402
from couplet.abb import Engine, TripleBudgetError, WireShares, bits_to_ints, ints_to_bits  # noqa: E402
from couplet.circuit import (  # noqa: E402
    AND, BOUND, ONE, PUBLIC, RETAIN, ZERO, Builder, Circuit, bits_to_int, evaluate,
)
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.prep import IdealDealer, IdealPrep, RealPrep  # noqa: E402

NO_REAL = "--no-real" in sys.argv
MODES = ("ideal",) if NO_REAL else ("ideal", "real")
BATCHES = (1, 7)
CHECKS = 0
NOTES: list[str] = []


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #

def check(cond, msg: str = "") -> None:
    """Count one check; raise on failure."""
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg or "check failed")


def check_eq(got, exp, msg: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if got != exp:
        raise AssertionError(f"{msg}: got {got!r}, expected {exp!r}" if msg else f"got {got!r}, expected {exp!r}")


def check_raises(exc, fn, *args, **kwargs) -> None:
    global CHECKS
    CHECKS += 1
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


def run_parties(mode: str, script, cheats=(None, None)):
    """Run ``script(role, engine)`` at both parties.

    Returns ``(out0, out1, stats0, stats1, box)``; ``out_b`` is ``("ok", value)``
    or ``("abort", message)`` (a ``ProtocolAbort`` raised at party b);
    ``box["eng"]`` holds the two engines, ``box["delta"]`` the Boolean MAC key
    shares, ``box["dealer"]`` the ideal dealer (ideal mode).
    """
    dealer = IdealDealer() if mode == "ideal" else None
    sid = b"test-abb-" + secrets.token_bytes(8)
    box = {"eng": [None, None], "delta": [None, None], "dealer": dealer}

    def party(role):
        def fn(chan):
            if mode == "ideal":
                prep = IdealPrep(role, dealer)
            else:
                prep = RealPrep(role, chan, sid)
                prep.setup()
            box["delta"][role] = prep.delta
            eng = Engine(role, chan, prep, sid, cheat=cheats[role])
            box["eng"][role] = eng
            try:
                return "ok", script(role, eng)
            except ProtocolAbort as exc:
                return "abort", str(exc)
        return fn

    out0, out1, st0, st1 = run_two_party(party(0), party(1), timeout=300)
    return out0, out1, st0, st1, box


def run_honest(mode: str, script):
    out0, out1, st0, st1, box = run_parties(mode, script)
    check(out0[0] == "ok" and out1[0] == "ok", f"honest run aborted: {out0[1] if out0[0] != 'ok' else out1[1]}")
    return out0[1], out1[1], st0, st1, box


# --------------------------------------------------------------------------- #
# Test circuits
# --------------------------------------------------------------------------- #

def c_adder(width=16, adder="ripple") -> Circuit:
    B = Builder(adder=adder)
    x = B.input(0, "x", width)
    y = B.input(1, "y", width)
    s, c = B.add_mod(x, y)
    B.output("sum", s + [c])
    return B.build()


def c_mul() -> Circuit:
    B = Builder()
    x = B.input(0, "x", 8)
    y = B.input(1, "y", 8)
    p = B.mul(x, y, 16)
    B.output("prod", p, to=0)
    par = p[0]
    for w in p[1:]:
        par = B.xor(par, w)
    B.output("parity", par)
    return B.build()


def c_mux() -> Circuit:
    B = Builder()
    sel = B.input(1, "sel", 1)
    a = B.input(0, "a", 12)
    b = B.input(1, "b", 12)
    B.output("mx", B.mux(sel[0], a, b), to=1)
    B.output("diff", B.sub_mod(a, b)[0], to=0)
    return B.build()


def c_mixed() -> Circuit:
    """NOT gates, constants and input wires as outputs, a wire used by two outputs."""
    B = Builder(adder="sklansky")
    x = B.input(0, "x", 10)
    y = B.input(1, "y", 10)
    z = B.input(1, "z", 4)
    s, c = B.add_mod(x, y)
    neg = B.neg_if(z[0], s)
    B.output("pub", B.not_bits(x[:3]) + [ZERO, ONE, y[4], c, neg[0], neg[0]])
    B.output("p0", neg + [ONE, z[1]], to=0)
    B.output("p1", B.mul(x[:5], z, 9) + [ZERO, x[9]], to=1)
    B.output("orr", [B.or_(a, b) for a, b in zip(x, y)])
    return B.build()


def c_xor_only() -> Circuit:
    """No AND gate: the first BoolCheck is empty."""
    B = Builder()
    x = B.input(0, "x", 9)
    y = B.input(1, "y", 9)
    B.output("x_xor_y", B.xor_bits(x, y))
    B.output("nx", B.not_bits(B.xor_bits(x, B.rotr(y, 3))), to=1)
    return B.build()


def c_random(seed: int, n_gates: int = 220) -> Circuit:
    rng = random.Random(seed)
    B = Builder()
    pool = B.input(0, "a", 7) + B.input(1, "b", 7) + B.input(0, "c", 3)
    for _ in range(n_gates):
        kind = rng.choice(("xor", "and", "and", "not", "xor"))
        u, v = rng.choice(pool), rng.choice(pool)
        pool.append(B.not_(u) if kind == "not" else (B.xor(u, v) if kind == "xor" else B.and_(u, v)))
    picks = [rng.choice(pool[10:]) for _ in range(20)]
    B.output("pub", picks[:8])
    B.output("q0", picks[8:13], to=0)
    B.output("q1", picks[13:18], to=1)
    B.output("more", picks[18:] + [pool[-1]])
    return B.build()


def c_chain1() -> Circuit:
    B = Builder()
    x = B.input(0, "x", 8)
    y = B.input(1, "y", 8)
    s, c = B.add_mod(x, y)
    B.output("acc", s + [c], to=RETAIN)                    # 9 bits
    B.output("prod", B.mul(x, y, 8), to=RETAIN)            # 8 bits
    par = x[0]
    for w in x[1:] + y:
        par = B.xor(par, w)
    B.output("par", par)
    return B.build()


def c_chain2() -> Circuit:
    B = Builder()
    acc = B.input(BOUND, "acc", 9)
    prod = B.input(BOUND, "prod", 8)
    z = B.input(0, "z", 8)
    u = B.input(1, "u", 8)
    t = B.add_mod(B.mul(acc, z, 16), prod, width=16)[0]
    B.output("out", t)
    B.output("mx", B.mux(u[0], acc[:8], prod), to=1)
    s, c = B.add_mod(acc, u, width=9)
    B.output("acc2", s + [c], to=RETAIN)                   # 10 bits
    return B.build()


def c_chain3() -> Circuit:
    B = Builder()
    acc2 = B.input(BOUND, "acc2", 10)
    acc = B.input(BOUND, "acc", 9)
    v = B.input(0, "v", 10)
    B.output("fin", B.add_mod(B.xor_bits(acc2, v), acc, width=10)[0])
    B.output("low", B.and_bits(acc[0], acc2), to=0)
    return B.build()


def honest_circuits():
    return [("adder", c_adder()), ("adder-sklansky", c_adder(11, "sklansky")), ("mul", c_mul()),
            ("mux", c_mux()), ("mixed", c_mixed()), ("xor-only", c_xor_only()),
            ("random-1", c_random(1)), ("random-2", c_random(2, 400)), ("random-3", c_random(3, 60))]


def random_assignments(C: Circuit, rng: random.Random, batch: int, bound=None) -> list:
    """``batch`` plaintext assignments; ``bound`` = label -> list of values per instance."""
    out = []
    for j in range(batch):
        a = {}
        for owner, label, wires in C.inputs:
            if owner == BOUND:
                vals = bound[label]
                a[(owner, label)] = vals[j] if len(vals) > 1 else vals[0]
            else:
                a[(owner, label)] = rng.getrandbits(len(wires))
        out.append(a)
    return out


def party_inputs(C: Circuit, assignments, role: int) -> dict:
    return {label: ints_to_bits([a[(role, label)] for a in assignments], len(wires))
            for owner, label, wires in C.inputs if owner == role}


def reconstruct(ws0: WireShares, ws1: WireShares) -> list:
    """Values (one per instance) of a retained block from the two parties' shares."""
    return bits_to_ints(ws0.bits ^ ws1.bits)


def check_mac_invariant(ws0: WireShares, ws1: WireShares, delta: int, tag: str) -> None:
    """m_0 XOR m_1 = x * Delta^B for every wire and instance."""
    bits = (ws0.bits ^ ws1.bits).reshape(-1)
    macs = (ws0.macs ^ ws1.macs).reshape(-1, 2)
    check(np.array_equal(macs, gf.bit_mul(bits, delta)), f"{tag}: MAC invariant of retained wires")


MASK_STATS = {"bits": 0, "differ": 0}


def check_result(C: Circuit, assignments, res0, res1, eng0, eng1, tag: str) -> dict:
    """Compare both parties' results with the plaintext evaluator; returns retained values."""
    owners = {label: to for to, label, _ in C.outputs}
    pub = {l for l, t in owners.items() if isinstance(t, str) and t == PUBLIC}
    ret = {l for l, t in owners.items() if isinstance(t, str) and t == RETAIN}
    priv = [{l for l, t in owners.items() if not isinstance(t, str) and t == b} for b in (0, 1)]
    for res, b in ((res0, 0), (res1, 1)):
        check_eq(set(res.public), pub, f"{tag}: public labels at P{b}")
        check_eq(set(res.private), priv[b], f"{tag}: P{b} receives exactly its private outputs")
        check_eq(set(res.retained), ret, f"{tag}: retained labels at P{b}")
    exp = [evaluate(C, a) for a in assignments]
    batch = len(assignments)
    retained = {}
    for to, label, wires in C.outputs:
        want = [bits_to_int(e[label]) for e in exp]
        if label in pub:
            check_eq(res0.ints(label), want, f"{tag}: public output {label} at P0")
            check_eq(res1.ints(label), want, f"{tag}: public output {label} at P1")
            check_eq(res0.public[label].shape, (len(wires), batch), f"{tag}: shape of {label}")
        elif label in ret:
            ws0, ws1 = res0.retained[label], res1.retained[label]
            check_eq(reconstruct(ws0, ws1), want, f"{tag}: retained output {label}")
            check(np.array_equal(ws0.ids, ws1.ids) and (ws0.width, ws0.batch) == (len(wires), batch),
                  f"{tag}: retained block {label} has equal identifiers and the right shape")
            retained[label] = want
        else:
            owner = to
            res = (res0, res1)[owner]
            check_eq(res.ints(label), want, f"{tag}: private output {label} at its owner P{owner}")
            # the opening both parties see is masked by an input mask of the owner
            m0 = eng0.last_opened["masked"][label]
            m1 = eng1.last_opened["masked"][label]
            check(np.array_equal(m0, m1), f"{tag}: both parties see the same masked opening of {label}")
            true_bits = ints_to_bits(want, len(wires))
            MASK_STATS["bits"] += true_bits.size
            MASK_STATS["differ"] += int((m0 ^ true_bits).sum())
    return retained


def expected_cost(C: Circuit, batch: int, role: int) -> tuple[int, int]:
    """(rounds, bytes sent by ``role``) of one evaluation, without the preprocessing."""
    st = C.stats()
    n_in = st["n_input_bits_p0"] if role == 0 else st["n_input_bits_p1"]
    n_open = st["n_public_output_bits"] + st["n_private_output_bits"]
    rounds = 1 + st["and_depth"]
    sent = 32 + (n_in * batch + 7) // 8
    for s, e in C.layer_ranges():
        sent += (2 * (e - s) * batch + 7) // 8
    if st["n_and"]:
        rounds += 2
        sent += 96
    if n_open:
        rounds += 3
        sent += (n_open * batch + 7) // 8 + 96
    return rounds, sent


# --------------------------------------------------------------------------- #
# Tests: helpers and plans (no protocol)
# --------------------------------------------------------------------------- #

def test_bit_helpers():
    rng = random.Random(7)
    for width in (1, 7, 8, 9, 64, 65, 128, 200):
        vals = [rng.getrandbits(width) for _ in range(5)] + [0, (1 << width) - 1]
        bits = ints_to_bits(vals, width)
        check_eq(bits.shape, (width, len(vals)), "shape (width, batch)")
        check_eq(bits_to_ints(bits), vals, "round trip")
        check_eq([int(bits[i, 0]) for i in range(width)], [(vals[0] >> i) & 1 for i in range(width)],
                 "row i holds bit i")
    check_raises(ValueError, ints_to_bits, [4], 2)
    check_raises(ValueError, ints_to_bits, [-1], 8)
    check_eq(bits_to_ints(np.array([1, 0, 1], dtype=np.uint8)), [5], "a vector is one instance")


def plan_eval(plan, C: Circuit, assignments) -> list:
    """Plaintext evaluation THROUGH the compiled plan (rows x instances), outputs as ints."""
    batch = len(assignments)
    table = np.zeros((plan.n_slots, batch), dtype=np.uint8)
    table[1] = 1
    for owner in (0, 1):
        for label, off, width in plan.in_fields[owner]:
            table[plan.in_slots[owner][off:off + width]] = ints_to_bits(
                [a[(owner, label)] for a in assignments], width)
    for label, slots in plan.bound_fields:
        table[slots] = ints_to_bits([a[(BOUND, label)] for a in assignments], slots.shape[0])
    triples = 0
    for is_and, u, v, ia, ib, t0 in plan.steps:
        if is_and:
            check_eq(t0, triples, "triples are consumed consecutively")
            triples += v - u
            table[u:v] = table[ia] & table[ib]
        else:
            table[u:v] = table[ia] ^ table[ib]
    check_eq(triples, plan.n_and, "every AND gate has a triple")
    out = []
    opened = table[plan.open_slots]
    for j in range(batch):
        row = {}
        for label, off, width in plan.out_public + plan.out_private[0] + plan.out_private[1]:
            row[label] = bits_to_ints(opened[off:off + width, j:j + 1])[0]
        for label, slots in plan.out_retain:
            row[label] = bits_to_ints(table[slots][:, j:j + 1])[0]
        out.append(row)
    return out


def test_plans_match_plaintext():
    rng = random.Random(11)
    circuits = honest_circuits() + [("chain1", c_chain1()), ("chain2", c_chain2()), ("chain3", c_chain3()),
                                    ("setup", cg.build_session_setup(64)),
                                    ("dpf-n3", cg.build_dpf(3, 32, 64, "add", "secret")),
                                    ("dcf-n2", cg.build_dcf(2, 64, 64, "xor", "public_one"))]
    for name, C in circuits:
        plan = abb.compile_plan(C)
        st = C.stats()
        check_eq(plan.digest.hex(), C.digest(), f"{name}: plan digest is the circuit digest")
        check_eq((plan.n_gates, plan.n_and, plan.and_depth), (st["n_gates"], st["n_and"], st["and_depth"]),
                 f"{name}: sizes")
        written = np.zeros(plan.n_slots, dtype=np.int64)
        for is_and, u, v, ia, ib, _ in plan.steps:
            written[u:v] += 1
            check(len(ia) == len(ib) == v - u and int(max(ia.max(), ib.max())) < u,
                  f"{name}: a step reads only rows written before it")
        base = 2 + C.n_input_wires
        check(bool((written[base:] == 1).all()) and not written[:base].any(),
              f"{name}: every gate row is written exactly once")
        check_eq(sum(v - u for is_and, u, v, *_ in plan.steps if is_and), st["n_and"], f"{name}: AND rows")
        check_eq(sum(1 for step in plan.steps if step[0]), st["and_depth"], f"{name}: one step per AND layer")
        bound = {label: [rng.getrandbits(len(w)) for _ in range(5)] for o, label, w in C.inputs if o == BOUND}
        assignments = random_assignments(C, rng, 5, bound)
        got = plan_eval(plan, C, assignments)
        for a, row in zip(assignments, got):
            exp = {label: bits_to_int(bits) for label, bits in evaluate(C, a).items()}
            check_eq(row, exp, f"{name}: plan evaluation equals circuit.evaluate")
        check(abb.get_plan(C) is abb.get_plan(C), f"{name}: plans are cached")
        check(abb.get_plan(Circuit.from_bytes(C.to_bytes())) is abb.get_plan(C),
              f"{name}: the cache key is the circuit digest")
    NOTES.append("plans: " + ", ".join(
        f"{name} {abb.get_plan(C).n_gates} gates in {len(abb.get_plan(C).steps)} steps"
        for name, C in circuits[-3:]))


def test_plan_rejects_invalid_circuits():
    C = c_adder(8)
    plan = abb.compile_plan(C)
    check(plan.n_and == 8, "reference circuit")

    def variant(**kw):
        args = dict(n_wires=C.n_wires, op=bytearray(C.op), a=C.a[:], b=C.b[:], out=C.out[:],
                    inputs=C.inputs, outputs=C.outputs, layers=C.layer_ranges(),
                    n_input_wires=C.n_input_wires)
        args.update(kw)
        return Circuit(**args)

    check(abb.compile_plan(variant()).n_and == 8, "an unchanged copy compiles")
    check_raises(ValueError, abb.compile_plan, variant(layers=[]))                    # ANDs outside layers
    check_raises(ValueError, abb.compile_plan, variant(layers=C.layer_ranges()[:-1]))
    first_and = bytes(C.op).index(AND)
    bad = C.a[:]
    bad[first_and] = C.n_wires - 1                                                    # reads a later wire
    check_raises(ValueError, abb.compile_plan, variant(a=bad))
    bad = C.out[:]
    bad[0], bad[1] = bad[1], bad[0]
    check_raises(ValueError, abb.compile_plan, variant(out=bad))
    ops = bytearray(C.op)
    ops[0] = 7
    check_raises(ValueError, abb.compile_plan, variant(op=ops))
    check_raises(ValueError, abb.compile_plan, variant(n_wires=C.n_wires + 1))
    check_raises(ValueError, abb.get_plan, "not a circuit")
    # a long XOR chain exceeds the vectorised level iteration and uses the sequential fallback
    B = Builder()
    x = B.input(0, "x", 400)
    y = B.input(1, "y", 1)
    acc = y[0]
    chain = []
    for w in x:
        acc = B.xor(acc, w)
        chain.append(acc)
    B.output("chain", chain[::7] + [B.and_(acc, x[0])])
    L = B.build()
    rng = random.Random(5)
    a = random_assignments(L, rng, 3)
    exp = [{k: bits_to_int(v) for k, v in evaluate(L, x_).items()} for x_ in a]
    check_eq(plan_eval(abb.compile_plan(L), L, a), exp, "long XOR chain (sequential levels)")


# --------------------------------------------------------------------------- #
# Tests: honest evaluations
# --------------------------------------------------------------------------- #

def _honest_suite(mode: str) -> None:
    circuits = honest_circuits()
    rng = random.Random(f"honest-{mode}")
    jobs = [(name, C, batch, random_assignments(C, rng, batch)) for name, C in circuits for batch in BATCHES]

    def script(role, eng):
        results = []
        for name, C, batch, assignments in jobs:
            sent, rounds = eng.chan.bytes_sent, eng.chan.rounds
            pool_before = eng.triple_requests
            res = eng.eval_circuit(C, party_inputs(C, assignments, role), None, batch)
            results.append((res, dict(eng.last_opened), eng.chan.bytes_sent - sent,
                            eng.chan.rounds - rounds, eng.triple_requests - pool_before))
        return results

    t = time.perf_counter()
    r0, r1, st0, st1, box = run_honest(mode, script)
    elapsed = time.perf_counter() - t
    eng0, eng1 = box["eng"]
    gates = 0
    for (name, C, batch, assignments), (res0, op0, sent0, rounds0, req0), (res1, op1, sent1, rounds1, _) in zip(jobs, r0, r1):
        tag = f"{mode} {name} batch={batch}"
        # check_result reads the masked openings from the engines: give it this job's snapshot
        eng0.last_opened, eng1.last_opened = op0, op1
        check_result(C, assignments, res0, res1, eng0, eng1, tag)
        st = C.stats()
        gates += st["n_gates"] * batch
        check_eq(req0, 1 if st["n_and"] else 0, f"{tag}: one triple request per evaluation")
        check_eq(res0.stats["triples"], st["n_and"] * batch, f"{tag}: n_and * batch triples")
        if mode == "ideal":       # no preprocessing traffic: the counters are those of the evaluation
            check_eq((rounds0, sent0), expected_cost(C, batch, 0), f"{tag}: rounds and bytes of P0")
            check_eq((rounds1, sent1), expected_cost(C, batch, 1), f"{tag}: rounds and bytes of P1")
            check_eq(res0.stats["rounds"], rounds0, f"{tag}: stats report the rounds")
    for st in (st0, st1):
        for name in ("bytes_sent", "bytes_recv", "rounds", "msgs_sent"):
            check_eq(sum(ph[name] for ph in st["phases"].values()), st[name], f"{mode}: phases sum to total {name}")
    check_eq(st0["bytes_sent"], st1["bytes_recv"], f"{mode}: sent by P0 = received by P1")
    check_eq(st0["rounds"], st1["rounds"], f"{mode}: both parties count the same rounds")
    check(not eng0.dead and not eng1.dead, f"{mode}: engines alive after honest runs")
    NOTES.append(f"honest suite ({mode}): {len(jobs)} evaluations, {gates:,} gate instances, "
                 f"{st0['rounds']} rounds, {st0['bytes_sent']:,} bytes sent by P0, {elapsed:.2f} s")


def test_honest_ideal():
    _honest_suite("ideal")


def test_honest_real():
    if NO_REAL:
        return
    _honest_suite("real")


def test_private_openings_are_masked():
    """The public opening of a private wire is the value XOR a uniform mask of its owner."""
    rng = random.Random("masked")
    before = dict(MASK_STATS)
    jobs = [(C, random_assignments(C, rng, 128)) for C in (c_mux(), c_mul())]

    def script(role, eng):
        out = []
        for C, assignments in jobs:
            res = eng.eval_circuit(C, party_inputs(C, assignments, role), None, 128)
            out.append((res, dict(eng.last_opened)))
        return out

    r0, r1, _, _, box = run_honest("ideal", script)
    eng0, eng1 = box["eng"]
    for (C, assignments), (res0, op0), (res1, op1) in zip(jobs, r0, r1):
        eng0.last_opened, eng1.last_opened = op0, op1
        check_result(C, assignments, res0, res1, eng0, eng1, "masked openings")
    bits = MASK_STATS["bits"] - before["bits"]
    differ = MASK_STATS["differ"] - before["differ"]
    frac = differ / bits
    check_eq(bits, (12 + 12 + 16) * 128, "private output bits of the two circuits")
    check(0.44 < frac < 0.56, f"masked openings differ from the private values in {frac:.3f} of {bits} bits")
    NOTES.append(f"masked private openings: {differ} of {bits} opened bits differ from the true value ({frac:.3f})")


def test_masks_are_fresh():
    """Every input bit and every private output bit of every instance has its own mask."""
    B = Builder()
    x = B.input(0, "x", 64)
    y = B.input(1, "y", 64)
    B.output("x_back", x, to=0)             # private outputs that equal input wires
    B.output("x_again", x, to=0)
    B.output("y_back", y, to=1)
    B.output("pub", B.xor_bits(x, y))
    C = B.build()
    rng = random.Random("fresh-masks")
    for mode in MODES:
        batch = 5
        assignments = random_assignments(C, rng, batch)

        def script(role, eng):
            res = eng.eval_circuit(C, party_inputs(C, assignments, role), None, batch)
            return res, dict(eng.last_opened)

        r0, r1, _, _, box = run_honest(mode, script)
        eng0, eng1 = box["eng"]
        eng0.last_opened, eng1.last_opened = r0[1], r1[1]
        check_result(C, assignments, r0[0], r1[0], eng0, eng1, f"{mode} fresh masks")
        for view in (r0[1], r1[1]):          # both parties see the same public transcript values
            d_x, d_y = view["input_diffs"]
            xs = ints_to_bits([a[(0, "x")] for a in assignments], 64)
            ys = ints_to_bits([a[(1, "y")] for a in assignments], 64)
            masks = {"in x": d_x ^ xs, "in y": d_y ^ ys, "x_back": view["masked"]["x_back"] ^ xs,
                     "x_again": view["masked"]["x_again"] ^ xs, "y_back": view["masked"]["y_back"] ^ ys}
            columns = [bytes(m[:, j]) for m in masks.values() for j in range(batch)]
            check_eq(len(set(columns)), 5 * batch, f"{mode}: all 64-bit masks of all wires and instances differ")
            check(all(any(col) for col in columns), f"{mode}: no mask is zero")
        check(np.array_equal(r0[1]["input_diffs"][0], r1[1]["input_diffs"][0])
              and np.array_equal(r0[1]["input_diffs"][1], r1[1]["input_diffs"][1]),
              f"{mode}: both parties hold the same input differences")


def _chain(mode: str) -> None:
    C1, C2, C3 = c_chain1(), c_chain2(), c_chain3()
    rng = random.Random(f"chain-{mode}")
    for b1, b2 in ((1, 1), (7, 7), (1, 7)):          # (1, 7): a batch-1 block shared by 7 instances
        a1 = random_assignments(C1, rng, b1)
        exp1 = [evaluate(C1, a) for a in a1]
        acc = [bits_to_int(e["acc"]) for e in exp1]
        prod = [bits_to_int(e["prod"]) for e in exp1]
        a2 = random_assignments(C2, rng, b2, {"acc": acc, "prod": prod})
        exp2 = [evaluate(C2, a) for a in a2]
        acc2 = [bits_to_int(e["acc2"]) for e in exp2]
        a3 = random_assignments(C3, rng, b2, {"acc2": acc2, "acc": acc})
        box_res = {}

        def script(role, eng):
            r1 = eng.eval_circuit(C1, party_inputs(C1, a1, role), None, b1)
            op1 = dict(eng.last_opened)
            # a list of single wires is accepted for a BOUND input as well as a block
            bound2 = {"acc": list(r1.retained["acc"]), "prod": r1.retained["prod"]}
            r2 = eng.eval_circuit(C2, party_inputs(C2, a2, role), bound2, b2)
            op2 = dict(eng.last_opened)
            r3 = eng.eval_circuit(C3, party_inputs(C3, a3, role),
                                  {"acc2": r2.retained["acc2"], "acc": r1.retained["acc"]}, b2)
            return r1, op1, r2, op2, r3, dict(eng.last_opened)

        r0, r1, _, _, box = run_honest(mode, script)
        eng0, eng1 = box["eng"]
        delta = box["delta"][0] ^ box["delta"][1]
        tag = f"{mode} chain {b1}->{b2}"
        for C, assignments, i in ((C1, a1, 0), (C2, a2, 2), (C3, a3, 4)):
            eng0.last_opened, eng1.last_opened = r0[i + 1], r1[i + 1]
            got = check_result(C, assignments, r0[i], r1[i], eng0, eng1, f"{tag} circuit {i // 2 + 1}")
            box_res[i] = got
        check_eq(box_res[0]["acc"], acc, f"{tag}: retained acc")
        check_eq(box_res[2]["acc2"], acc2, f"{tag}: retained acc2 computed from bound wires")
        for res_a, res_b, label in ((r0[0], r1[0], "acc"), (r0[0], r1[0], "prod"), (r0[2], r1[2], "acc2")):
            check_mac_invariant(res_a.retained[label], res_b.retained[label], delta, f"{tag} {label}")
        ids = np.concatenate([r0[0].retained["acc"].ids, r0[0].retained["prod"].ids, r0[2].retained["acc2"].ids])
        check_eq(len(set(ids.tolist())), 27, f"{tag}: wire identifiers are unique in the session")
        ws = r0[0].retained["acc"]
        check(ws[2].width == 1 and np.array_equal(ws[2].bits[0], ws.wire(2)[0])
              and np.array_equal(ws[1:4].macs, ws.macs[1:4]) and len(list(ws)) == 9
              and np.array_equal(WireShares.concat(list(ws)).macs, ws.macs)
              and np.array_equal(ws[-1].ids, ws.ids[-1:]), f"{tag}: WireShares indexing")


def test_bound_retain_chain_ideal():
    _chain("ideal")


def test_bound_retain_chain_real():
    if NO_REAL:
        return
    _chain("real")


def test_setup_circuit_and_keygen_throughput():
    """Session-setup circuit (RETAIN only), then a DPF generation circuit on the bound Delta."""
    n, k, s = 8, 64, 64
    Cs = cg.build_session_setup(s)
    C = cg.build_dpf(n, k, s, "xor", "public_one")
    rng = random.Random("throughput")
    top = (1 << s) - 1
    for batch, d0, d1 in ((1, top, top - 5), (7, rng.getrandbits(s - 2), rng.getrandbits(s - 2))):
        a_setup = [{(0, "delta"): d0, (1, "delta"): d1}]
        delta = d0 + d1
        assignments = random_assignments(C, rng, batch, {"delta": [delta]})

        def script(role, eng):
            r = eng.eval_circuit(Cs, party_inputs(Cs, a_setup, role), None, 1)
            op = dict(eng.last_opened)
            res = eng.eval_circuit(C, party_inputs(C, assignments, role), {"delta": r.retained["delta"]}, batch)
            return r, op, res, dict(eng.last_opened)

        r0, r1, st0, _, box = run_honest("ideal", script)
        tag = f"dpf n={n} batch={batch}"
        check_eq(reconstruct(r0[0].retained["delta"], r1[0].retained["delta"]), [delta],
                 f"{tag}: retained Delta = Delta_0 + Delta_1 (s+1 bits, not reduced)")
        check(r0[0].public == {} and r0[0].private == {} and r0[1]["public"] == {} and r0[1]["masked"] == {},
              f"{tag}: the setup circuit opens nothing")
        check_eq(r0[0].stats["rounds"], 1 + 64 + 2, f"{tag}: setup rounds (no output opening)")
        eng0, eng1 = box["eng"]
        eng0.last_opened, eng1.last_opened = r0[3], r1[3]
        check_result(C, assignments, r0[2], r1[2], eng0, eng1, tag)
        for j, a in enumerate(assignments):
            out = {label: r0[2].ints(label)[j] for label in r0[2].public}
            out["root0"], out["root1"] = r0[2].ints("root0")[j], r1[2].ints("root1")[j]
            k0, k1 = cg.unpack_key_outputs(out, "dpf", n, k, s)
            alpha = a[(0, "alpha")] ^ a[(1, "alpha")]
            e0, e1 = fss.dpf_gen(alpha, 1, delta, n, k, s, roots=(k0.seed, k1.seed))
            check(k0 == e0 and k1 == e1, f"{tag}: keys equal fss.dpf_gen with the unreduced Delta")
        stt = r0[2].stats
        check(stt["gates_per_s"] > 0 and stt["n_gates"] == C.stats()["n_gates"], f"{tag}: throughput reported")
        NOTES.append(f"{tag}: {stt['n_gates']:,} gates ({stt['n_and']:,} AND, depth {stt['and_depth']}, "
                     f"{stt['linear_steps']} linear steps), online {stt['online_s'] * 1e3:.1f} ms, "
                     f"{stt['gates_per_s'] / 1e6:.1f} M gate instances/s "
                     f"({stt['gates_per_s_local'] / 1e6:.1f} M/s excluding time blocked in exchanges), "
                     f"{stt['rounds']} rounds, {stt['bytes_sent']:,} bytes sent")


# --------------------------------------------------------------------------- #
# Tests: deviations
# --------------------------------------------------------------------------- #

def _tamper(mode: str, C: Circuit, batch: int, cheater: int, hook: dict, tag: str, rng) -> None:
    """One run with a cheating party: the honest party must abort and return nothing."""
    assignments = random_assignments(C, rng, batch)
    after = {}

    def script(role, eng):
        res = eng.eval_circuit(C, party_inputs(C, assignments, role), None, batch)
        return res

    cheats = [None, None]
    cheats[cheater] = hook
    out0, out1, _, _, box = run_parties(mode, script, tuple(cheats))
    honest = 1 - cheater
    out_h = (out0, out1)[honest]
    eng_h = box["eng"][honest]
    check_eq(out_h[0], "abort", f"{tag}: the honest party P{honest} aborts")
    check_eq(out_h[1], "BoolCheck failed", f"{tag}: the abort comes from the MAC check")
    check(not isinstance(out_h[1], abb.EvalResult), f"{tag}: no output is returned")
    check(eng_h.dead and eng_h.prep.aborted and eng_h.prep.delta is None and eng_h._delta is None,
          f"{tag}: the honest engine is dead and its key share is dropped")
    check_raises(ProtocolAbort, eng_h.eval_circuit, C, party_inputs(C, assignments, honest), None, batch)
    check_raises(ProtocolAbort, eng_h.reserve_triples, 5)
    after["x"] = True


def test_tamper_and_opening():
    rng = random.Random("tamper-and")
    C = c_mixed()
    layers = C.layer_ranges()
    last = len(layers) - 1
    width_last = layers[last][1] - layers[last][0]
    cases = [("ideal", 1, 0, (0, 0)), ("ideal", 1, 1, (0, 1)), ("ideal", 7, 1, (2, 3)),
             ("ideal", 7, 0, (last, 2 * width_last * 7 - 1)),        # the last e bit of the last layer
             ("ideal", 1, 0, (last, 0))]
    if not NO_REAL:
        cases.append(("real", 7, 1, (1, 5)))
    for mode, batch, cheater, target in cases:
        _tamper(mode, C, batch, cheater, {"flip_and": target},
                f"{mode} flipped d/e bit {target} by P{cheater}, batch {batch}", rng)
    # every AND layer of a small circuit, both parties
    C = c_adder(6)
    for layer in range(C.and_depth()):
        _tamper("ideal", C, 1, layer % 2, {"flip_and": (layer, 0)}, f"adder layer {layer}", rng)


def test_tamper_public_output():
    rng = random.Random("tamper-public")
    cases = [("ideal", c_adder(), 1, 0, ("sum", 0, 0)), ("ideal", c_adder(), 7, 1, ("sum", 16, 6)),
             ("ideal", c_mixed(), 7, 0, ("pub", 3, 2)),        # a constant output wire
             ("ideal", c_mixed(), 1, 1, ("orr", 9, 0)),
             ("ideal", c_xor_only(), 7, 1, ("x_xor_y", 4, 3))]  # no AND gate at all
    if not NO_REAL:
        cases.append(("real", c_adder(), 7, 0, ("sum", 5, 4)))
    for mode, C, batch, cheater, target in cases:
        _tamper(mode, C, batch, cheater, {"flip_public": target},
                f"{mode} flipped public output share {target} by P{cheater}, batch {batch}", rng)


def test_tamper_private_output():
    rng = random.Random("tamper-private")
    # (label, owner): the cheater flips its share of the masked opening
    cases = [("ideal", c_mul(), 1, 1, ("prod", 0, 0)),          # private output of the honest P0
             ("ideal", c_mul(), 7, 1, ("prod", 15, 6)),
             ("ideal", c_mux(), 7, 0, ("mx", 5, 1)),            # private output of the honest P1
             ("ideal", c_mux(), 1, 0, ("diff", 2, 0)),          # the cheater's own private output
             ("ideal", c_mixed(), 7, 1, ("p1", 9, 3))]          # the cheater's own, a constant wire
    if not NO_REAL:
        cases.append(("real", c_mul(), 7, 1, ("prod", 7, 2)))
    for mode, C, batch, cheater, target in cases:
        _tamper(mode, C, batch, cheater, {"flip_private": target},
                f"{mode} flipped masked private output share {target} by P{cheater}, batch {batch}", rng)


def test_input_substitution_is_not_detected():
    rng = random.Random("substitution")
    cases = [("ideal", c_adder(), 1, 0, ("x", 3, 0)), ("ideal", c_mixed(), 7, 1, ("z", 0, 4)),
             ("ideal", c_mul(), 7, 0, ("x", 7, 6)), ("ideal", c_mux(), 1, 1, ("sel", 0, 0))]
    if not NO_REAL:
        cases.append(("real", c_mixed(), 7, 1, ("y", 9, 2)))
    for mode, C, batch, cheater, (label, bit, inst) in cases:
        assignments = random_assignments(C, rng, batch)

        def script(role, eng):
            res = eng.eval_circuit(C, party_inputs(C, assignments, role), None, batch)
            return res, dict(eng.last_opened)

        cheats = [None, None]
        cheats[cheater] = {"flip_input": (label, bit, inst)}
        out0, out1, _, _, box = run_parties(mode, script, tuple(cheats))
        tag = f"{mode} wrong input-mask difference for {label}[{bit}] of instance {inst} by P{cheater}"
        check(out0[0] == "ok" and out1[0] == "ok", f"{tag}: not detected (it is an input substitution)")
        effective = [dict(a) for a in assignments]
        effective[inst][(cheater, label)] ^= 1 << bit
        eng0, eng1 = box["eng"]
        eng0.last_opened, eng1.last_opened = out0[1][1], out1[1][1]
        check_result(C, effective, out0[1][0], out1[1][0], eng0, eng1, tag + " (substituted input)")
        differs = any(evaluate(C, a) != evaluate(C, e) for a, e in zip(assignments, effective))
        check(differs, f"{tag}: the substitution changes the outputs")
        check(not eng0.dead and not eng1.dead, f"{tag}: the session stays alive")


def test_header_mismatch():
    def circ(variant: int) -> Circuit:
        B = Builder()
        x = B.input(0, "x", 4)
        y = B.input(1, "y", 4)
        t = [B.and_(a, b) for a, b in zip(x, y)]
        B.output("o", B.xor_bits(t, x if variant == 0 else y))
        return B.build()

    Ca, Cb = circ(0), circ(1)
    check(Ca.stats() == Cb.stats() and Ca.digest() != Cb.digest(), "same shape, different gate lists")
    rng = random.Random("header")
    a = random_assignments(Ca, rng, 1)

    def script(role, eng):
        return eng.eval_circuit(Ca if role == 0 else Cb, party_inputs(Ca, a, role), None, 1)

    out0, out1, _, _, box = run_parties("ideal", script)
    for b, out in ((0, out0), (1, out1)):
        check(out[0] == "abort" and "header mismatch" in out[1], f"different circuits: P{b} aborts in the first exchange")
        check(box["eng"][b].dead, f"different circuits: engine of P{b} is dead")

    out0, out1, _, _, box = run_parties("ideal", lambda role, eng: eng.eval_circuit(Ca, party_inputs(Ca, a, role), None, 1),
                                        ({"wrong_header": True}, None))
    check(out1[0] == "abort" and "header mismatch" in out1[1], "wrong header: the honest party aborts")

    # the parties bind different session wires of the same width
    C1 = c_chain1()
    B = Builder()
    w = B.input(BOUND, "w", 8)
    B.output("o", B.and_bits(w[0], w))
    Cw = B.build()
    a1 = random_assignments(C1, rng, 1)

    def script_bound(role, eng):
        r1 = eng.eval_circuit(C1, party_inputs(C1, a1, role), None, 1)
        pick = r1.retained["prod"] if role == 0 else r1.retained["acc"][:8]
        return eng.eval_circuit(Cw, {}, {"w": pick}, 1)

    out0, out1, _, _, _ = run_parties("ideal", script_bound)
    check(out0[0] == "abort" and out1[0] == "abort" and "header mismatch" in out0[1],
          "different bound wires: both abort in the first exchange")

    def script_same(role, eng):
        r1 = eng.eval_circuit(C1, party_inputs(C1, a1, role), None, 1)
        return eng.eval_circuit(Cw, {}, {"w": r1.retained["acc"][:8]}, 1), r1

    (r0, _), (r1, _), _, _, _ = run_honest("ideal", script_same)
    acc = bits_to_int(evaluate(C1, a1[0])["acc"]) & 0xFF
    check_eq(r0.ints("o"), [acc if acc & 1 else 0], "the same slice of a retained block can be bound")
    check_eq(r1.ints("o"), r0.ints("o"), "both parties agree")


def test_malformed_messages():
    C = c_adder(8)
    rng = random.Random("malformed")
    a = random_assignments(C, rng, 1)
    plan = abb.get_plan(C)
    header_len = 32

    def honest(role, dealer):
        def fn(chan):
            eng = Engine(role, chan, IdealPrep(role, dealer), b"sid-malformed")
            try:
                eng.eval_circuit(C, party_inputs(C, a, role), None, 1)
            except ProtocolAbort as exc:
                return "abort", str(exc), eng.dead
            return "ok", None, eng.dead
        return fn

    def relay_then(garbage_at: int, garbage: bytes):
        """Peer that mirrors the honest header but sends ``garbage`` as message number ``garbage_at``."""
        def fn(chan):
            first = chan.recv()
            if garbage_at == 0:
                chan.send(garbage)
                return None
            chan.send(first[:header_len] + bytes(len(first) - header_len))   # d = 0 for its 8 input bits
            for i in range(1, 200):
                chan.recv()
                chan.send(garbage if i == garbage_at else b"\x00")
                if i == garbage_at:
                    return None
        return fn

    for name, at, garbage in (("short header", 0, b"abc"), ("wrong input length", 0, bytes(40)),
                              ("AND opening too long", 1, b"\x00\x00"),
                              ("AND opening with stray bits", 2, b"\xf0"),
                              ("empty AND opening", 3, b"")):
        dealer = IdealDealer()
        out0, _, _, _ = run_two_party(honest(0, dealer), relay_then(at, garbage), timeout=60)
        check(out0[0] == "abort" and out0[2], f"malformed message ({name}): abort and dead engine, got {out0}")
    check(plan.n_and == 8, "reference circuit")

    # the peer disappears in the middle of the evaluation
    def vanish(chan):
        chan.recv()
        return None

    out0, _, _, _ = run_two_party(honest(0, IdealDealer()), vanish, timeout=60)
    check(out0[0] == "abort" and out0[2], "peer closed: abort and dead engine")


def test_argument_errors_do_not_abort():
    C = c_mixed()
    C2 = c_chain2()
    rng = random.Random("arguments")
    a = random_assignments(C, rng, 2)
    counts = {}

    def script(role, eng):
        good = party_inputs(C, a, role)
        label = "x" if role == 0 else "y"
        n = 0
        for bad_inputs, bound, batch in (
                ({k: v for k, v in good.items() if k != label}, None, 2),            # missing input
                ({**good, "nope": good[label]}, None, 2),                             # unknown label
                ({**good, label: good[label][:5]}, None, 2),                          # wrong width
                ({**good, label: good[label] * 2}, None, 2),                          # not binary
                ({**good, label: good[label].astype(np.float64)}, None, 2),           # wrong dtype
                ({**good, label: np.zeros((10, 3), dtype=np.uint8)}, None, 2),        # wrong batch
                ({**good, (1 - role, label): good[label]}, None, 2),                  # other party's key
                (good, {"delta": None}, 2),                                           # unexpected BOUND
                (good, None, 0), (good, None, -1), (good, None, True), (good, None, 1.5)):
            try:
                eng.eval_circuit(C, bad_inputs, bound, batch)
            except ValueError:
                n += 1
        try:
            eng.eval_circuit("circuit", good, None, 2)
        except ValueError:
            n += 1
        try:
            eng.eval_circuit(C2, {"z" if role == 0 else "u": 1}, None, 1)             # missing BOUND
        except ValueError:
            n += 1
        try:
            eng.reserve_triples(-1)
        except ValueError:
            n += 1
        counts[role] = n
        alive = not eng.dead
        # integers and (width,) vectors are accepted and used for every instance
        mixed = dict(good)
        mixed[label] = a[0][(role, label)]
        res_int = eng.eval_circuit(C, mixed, None, 2)
        res = eng.eval_circuit(C, {(role, k): v for k, v in good.items()}, None, 2)
        return alive, res_int, res, dict(eng.last_opened)

    r0, r1, _, _, box = run_honest("ideal", script)
    check_eq(counts, {0: 15, 1: 15}, "every invalid call raises ValueError")
    check(r0[0] and r1[0], "the engines stay alive after argument errors")
    eng0, eng1 = box["eng"]
    eng0.last_opened, eng1.last_opened = r0[3], r1[3]
    check_result(C, a, r0[2], r1[2], eng0, eng1, "evaluation after argument errors ((role, label) keys)")
    same = [dict(x) for x in a]
    same[1][(0, "x")] = a[0][(0, "x")]
    same[1][(1, "y")] = a[0][(1, "y")]
    check_eq(r0[1].ints("orr"), [bits_to_int(evaluate(C, x)["orr"]) for x in same], "integer inputs are broadcast")

    # wires of another session, wrong width and wrong batch of a BOUND block
    C1 = c_chain1()
    a1 = random_assignments(C1, rng, 3)
    shared = {}

    def script_bound(role, eng):
        r = eng.eval_circuit(C1, party_inputs(C1, a1, role), None, 3)
        other = Engine(role, eng.chan, IdealPrep(role, IdealDealer()), b"other")
        foreign = WireShares(r.retained["acc"].bits, r.retained["acc"].macs, r.retained["acc"].ids, other._token)
        n = 0
        ins = {"z" if role == 0 else "u": 3}
        for bound in ({"acc": foreign, "prod": r.retained["prod"]},
                      {"acc": r.retained["acc"][:8], "prod": r.retained["prod"]},
                      {"acc": r.retained["acc"], "prod": r.retained["prod"]},              # batch 3 into 2
                      {"acc": r.retained["acc"], "prod": r.retained["prod"], "x": r.retained["prod"]},
                      {"acc": [], "prod": r.retained["prod"]}):
            try:
                eng.eval_circuit(C2, ins, bound, 2 if bound.get("x") is None else 3)
            except ValueError:
                n += 1
        shared[role] = n
        return not eng.dead

    r0, r1, _, _, _ = run_honest("ideal", script_bound)
    check_eq(shared, {0: 5, 1: 5}, "invalid BOUND arguments raise ValueError")
    check(r0 and r1, "the engines stay alive")
    check_raises(ValueError, Engine, 2, None, None, b"sid")
    check_raises(ValueError, WireShares.concat, [])


def test_triple_pool():
    C = c_adder(8)            # 8 AND gates
    rng = random.Random("pool")
    a = [random_assignments(C, rng, 2) for _ in range(4)]

    def script(role, eng):
        trace = []
        eng.reserve_triples(0)
        trace.append((eng.triple_requests, eng.triples_available()))
        eng.reserve_triples(40)                          # 16 per evaluation at batch 2
        trace.append((eng.triple_requests, eng.triples_available()))
        results = []
        for i in range(4):
            results.append(eng.eval_circuit(C, party_inputs(C, a[i], role), None, 2))
            trace.append((eng.triple_requests, eng.triples_available(), eng.triples_used))
        return results, trace, dict(eng.prep.counters)

    r0, r1, _, _, box = run_honest("ideal", script)
    eng0, eng1 = box["eng"]
    for i in range(4):
        for res in (r0[0][i], r1[0][i]):
            check_eq(res.ints("sum"), [bits_to_int(evaluate(C, x)["sum"]) for x in a[i]], f"pool: evaluation {i}")
    # 40 reserved; evaluations 0 and 1 use 32; evaluation 2 needs 16, has 8: one request for 8;
    # evaluation 3 needs 16, has 0: one request for 16
    check_eq(r0[1], [(0, 0), (1, 40), (1, 24, 16), (1, 8, 32), (2, 0, 48), (3, 0, 64)], "pool trace of P0")
    check_eq(r1[1], r0[1], "pool trace of P1")
    check_eq(r0[2]["and_triples"], 64, "triples drawn from the preprocessing")
    check_eq((eng0.triples_requested, eng0.allow_refill), (64, True),
             "the low-level engine refills by default and counts what it requested")


def test_triple_pool_strict():
    """Strict mode (``allow_refill=False``): a short pool is an error BEFORE any message.

    The bucketing bound holds per request (Lemma 4), so the engine must never
    start a new population silently; with ``allow_refill=True`` it may, and the
    request counter says so.
    """
    C = c_adder(8)            # 8 AND gates
    rng = random.Random("pool-strict")
    a = [random_assignments(C, rng, 2) for _ in range(4)]

    def counters(eng):
        return eng.triple_requests, eng.triples_requested, eng.triples_available(), eng.triples_used

    def script(role, eng):
        eng.allow_refill = False
        facts = {}
        before = (eng.chan.msgs_sent, eng.chan.bytes_sent, eng.chan.msgs_recv)
        try:
            eng.eval_circuit(C, party_inputs(C, a[0], role), None, 2)       # empty pool
            facts["empty"] = "returned"
        except TripleBudgetError as exc:
            facts["empty"] = (exc.needed, exc.available, isinstance(exc, ValueError),
                              "reserve" in str(exc) and "allow_refill" in str(exc))
        facts["empty_silent"] = (eng.chan.msgs_sent, eng.chan.bytes_sent, eng.chan.msgs_recv) == before
        facts["empty_state"] = (eng.dead, eng.evaluations) + counters(eng)
        eng.reserve_triples(40)                             # ONE request; 16 per evaluation at batch 2
        results = [eng.eval_circuit(C, party_inputs(C, a[i], role), None, 2) for i in range(2)]
        trace = [counters(eng)]
        before = (eng.chan.msgs_sent, eng.chan.bytes_sent, eng.chan.msgs_recv)
        try:
            eng.eval_circuit(C, party_inputs(C, a[2], role), None, 2)       # needs 16, holds 8
            facts["short"] = "returned"
        except TripleBudgetError as exc:
            facts["short"] = (exc.needed, exc.available)
        try:
            eng.require_triples(9)
            facts["require"] = "returned"
        except TripleBudgetError as exc:
            facts["require"] = (exc.needed, exc.available)
        eng.require_triples(8)                              # fits: no error
        facts["short_silent"] = ((eng.chan.msgs_sent, eng.chan.bytes_sent, eng.chan.msgs_recv) == before
                                 and not eng.dead and eng.evaluations == 2)
        trace.append(counters(eng))
        one = eng.eval_circuit(C, party_inputs(C, a[2][:1], role), None, 1)  # needs 8: uses the rest
        trace.append(counters(eng))
        eng.reserve_triples(16)                             # an EXPLICIT further population
        results.append(eng.eval_circuit(C, party_inputs(C, a[2], role), None, 2))
        trace.append(counters(eng))
        eng.allow_refill = True                             # refill mode: allowed, and counted
        results.append(eng.eval_circuit(C, party_inputs(C, a[3], role), None, 2))
        trace.append(counters(eng))
        return facts, results, trace, one, dict(one.stats), dict(results[-1].stats)

    r0, r1, st0, st1, box = run_honest("ideal", script)
    for r in (r0, r1):
        facts = r[0]
        check_eq(facts["empty"], (16, 0, True, True),
                 "strict: an empty pool raises TripleBudgetError (a ValueError that says what to do)")
        check(facts["empty_silent"], "strict: nothing was sent or received")
        check_eq(facts["empty_state"], (False, 0, 0, 0, 0, 0),
                 "strict: the engine is alive, no evaluation was counted, no request was made")
        check_eq(facts["short"], (16, 8), "strict: a pool of 8 does not serve 16 triples")
        check_eq(facts["require"], (9, 8), "require_triples reports the same condition")
        check(facts["short_silent"], "strict: the refused evaluation sent nothing and left the engine alive")
        # (requests, triples requested, available, used)
        check_eq(r[2], [(1, 40, 8, 32), (1, 40, 8, 32), (1, 40, 0, 40), (2, 56, 0, 56), (3, 72, 0, 72)],
                 "strict: one population until the caller adds one explicitly; a refill is counted")
        check_eq((r[4]["triple_requests"], r[4]["triples_requested"], r[4]["triples_available"]), (1, 40, 0),
                 "the evaluation statistics report the populations drawn so far")
        check_eq(r[5]["triple_requests"], 3, "statistics after an explicit request and one refill")
    for i in range(2):
        for res in (r0[1][i], r1[1][i]):
            check_eq(res.ints("sum"), [bits_to_int(evaluate(C, x)["sum"]) for x in a[i]],
                     f"strict pool: evaluation {i}")
    for res in (r0[3], r1[3]):
        check_eq(res.ints("sum"), [bits_to_int(evaluate(C, a[2][0])["sum"])], "strict pool: batch 1 from the rest")
    for j, i in ((2, 2), (3, 3)):
        for res in (r0[1][j], r1[1][j]):
            check_eq(res.ints("sum"), [bits_to_int(evaluate(C, x)["sum"]) for x in a[i]],
                     f"strict pool: evaluation after request {j}")
    check_eq(st0["bytes_sent"], st1["bytes_recv"], "strict pool: the two transcripts match")
    check_raises(ValueError, Engine, 0, type("C", (), {"party": 0})(), IdealPrep(0, IdealDealer()), b"s",
                 allow_refill=1)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    if "-k" in sys.argv:
        key = sys.argv[sys.argv.index("-k") + 1]
        tests = [(name, fn) for name, fn in tests if key in name]
    failed = []
    start = time.perf_counter()
    print(f"Couplet Boolean engine tests (preprocessing modes: {', '.join(MODES)}), {len(tests)} test functions")
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
        print(f"  ok   {name:<46} {CHECKS - before:>7} checks {time.perf_counter() - t:7.2f} s", flush=True)
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
