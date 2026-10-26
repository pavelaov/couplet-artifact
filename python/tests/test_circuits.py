"""Tests for the Boolean circuit layer (DESIGN.md sections 2.1, 4 and 6).

Covered:
  * Ascon circuit against ``ascon.permute``; the PRG maps (feed-forward)
    against ``ascon.G`` / ``conv`` / ``convv_L`` / ``convv_R``;
  * the circuit IR: builder arithmetic (both adders), constant folding,
    layers and schedule, BOUND inputs and RETAIN outputs, binary and text
    serialisation (format version 2), Bristol Fashion export (checked with an
    independent evaluator), batch evaluation;
  * session setup: Delta = Delta_0 + Delta_1 as an (s+1)-bit integer;
  * DPF / DCF generation circuits against ``fss.dpf_gen`` / ``fss.dcf_gen``
    with the same roots: bit-for-bit key equality AND full-domain
    correctness, for n in 1..6, k in {32, 64}, s = 64, both index modes, both
    payload modes, MAC keys on both sides of 2^s;
  * gate circuits for n in 3..6 and every combination of auth_input,
    dpf_one, n_outmask in {0, 2}, with thresholds c in {0, 1, N/2, N-1, N}
    and both payloads: r = r' mod N, all share sums (value and tag, with the
    unreduced Delta), DCF index (r + c) mod N, w_c, r w_c, output masks, the
    table identity, and the comparison identity of section 6 for all x;
  * batches of circuits sharing the one BOUND delta.

Run:  python tests/test_circuits.py [--quick] [-k substring]
The exit status is non-zero on failure; the number of checks is printed.
``--quick`` restricts the gate matrix to n in {3, 4} (smoke runs only).
The ``test_*`` functions are also collectable by pytest (not required).
"""

from __future__ import annotations

import os
import random
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import numpy as np  # noqa: E402

from couplet import ascon, fss  # noqa: E402
from couplet import circuits_ascon as ca  # noqa: E402
from couplet import circuits_gen as cg  # noqa: E402
from couplet.circuit import (  # noqa: E402
    AND, BOUND, NOT, ONE, PUBLIC, RETAIN, XOR, ZERO, Builder, Circuit, bits_to_int, evaluate,
    evaluate_batch, int_to_bits, outputs_as_ints,
)

QUICK = "--quick" in sys.argv
S = 64
SEED_BITS = 128
CHECKS = 0

# Ascon-Hash v1.2 precomputed initial state = p^12(0x00400c0000000100, 0, 0, 0, 0)
# (reference implementation, hash.c).
ASCON_HASH_V12_STATE = (0xEE9398AADB67F03D, 0x8BB21831C60F1002, 0xB48A92DB98D5DA62,
                        0x43189921B8F8E3E8, 0x348FA5C9D525E140)


# --------------------------------------------------------------------------- #
# Small test harness
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


def eval_bristol(text: str, values):
    """Independent Bristol Fashion evaluator (XOR / AND / INV only).

    ``values``: one integer per input value (little-endian wires).  Returns
    one integer per output value.  Checks that every wire is written at most
    once and read only after it was written."""
    lines = text.split("\n")
    n_gates, n_wires = (int(x) for x in lines[0].split())
    head_in = [int(x) for x in lines[1].split()]
    head_out = [int(x) for x in lines[2].split()]
    assert head_in[0] == len(head_in) - 1 and head_out[0] == len(head_out) - 1 and lines[3] == ""
    in_w, out_w = head_in[1:], head_out[1:]
    assert len(values) == len(in_w)
    wires = [None] * n_wires
    pos = 0
    for width, val in zip(in_w, values):
        assert 0 <= val < (1 << width)
        for i in range(width):
            wires[pos] = (val >> i) & 1
            pos += 1
    body = lines[4:4 + n_gates]
    assert len(body) == n_gates and lines[4 + n_gates:] == [""]
    for line in body:
        f = line.split()
        if f[-1] == "INV":
            assert f[0] == "1" and f[1] == "1" and len(f) == 5
            a, o = int(f[2]), int(f[3])
            val = 1 - wires[a]
        else:
            assert f[0] == "2" and f[1] == "1" and len(f) == 6
            a, b, o = int(f[2]), int(f[3]), int(f[4])
            if f[5] == "XOR":
                val = wires[a] ^ wires[b]
            elif f[5] == "AND":
                val = wires[a] & wires[b]
            else:
                raise AssertionError(f"unexpected gate {line!r}")
        assert wires[o] is None, "wire written twice"
        wires[o] = val
    outs = []
    pos = n_wires - sum(out_w)
    for width in out_w:
        outs.append(sum(wires[pos + i] << i for i in range(width)))
        pos += width
    return outs


def bristol_values(C: Circuit, inputs) -> list:
    """Input values of ``C.to_bristol()`` (party 0, party 1, bound) from an input mapping."""
    owner_of = {0: 0, 1: 1, "bound": BOUND}
    vals = []
    for group in C.bristol_layout()["inputs"]:
        v = 0
        for field in group["fields"]:
            v |= inputs[(owner_of[group["owner"]], field["label"])] << field["offset"]
        vals.append(v)
    return vals


# --------------------------------------------------------------------------- #
# Ascon and the PRG maps
# --------------------------------------------------------------------------- #

def test_ascon_reference_known_answer():
    iv_state = [0x00400C0000000100, 0, 0, 0, 0]
    check_eq(tuple(ca.scalar_permute(iv_state)), ASCON_HASH_V12_STATE, "local scalar p12")
    check_eq(tuple(ascon.permute(iv_state)), ASCON_HASH_V12_STATE, "ascon.permute")
    check(ca.PERMUTE_IS_EXTERNAL, "circuits_ascon must use couplet.ascon.permute")
    rng = random.Random(11)
    for _ in range(50):
        st = [rng.getrandbits(64) for _ in range(5)]
        check_eq(list(ascon.permute(st)), ca.scalar_permute(st), "permute vs local scalar")
    ascon.selftest()
    check(True, "ascon.selftest")


def test_ascon_circuit_matches_permute():
    st = ca.p12_circuit_stats()
    check_eq(st["n_and"], 3840, "p12 AND count")
    check_eq(st["and_depth"], 12, "p12 AND depth")
    B = Builder()
    w = B.input(0, "state", 320)
    B.output("out", ca.ascon_p12(B, w))
    C = B.build()
    rng = random.Random(2024)
    states = [rng.getrandbits(320) for _ in range(50)]
    outs = evaluate_batch(C, [{(0, "state"): s} for s in states])
    for o, s in zip(outs, states):
        words = [(s >> (64 * i)) & ascon.MASK64 for i in range(5)]
        exp = sum(wd << (64 * i) for i, wd in enumerate(ascon.permute(words)))
        check_eq(bits_to_int(o["out"]), exp, "Ascon circuit vs ascon.permute")
    o = evaluate(C, {(0, "state"): states[0]})
    check_eq(bits_to_int(o["out"]), ca.permute_int(states[0]), "unbatched evaluation")


def test_prg_maps_match_ascon():
    """Circuit PRG maps (with feed-forward) against couplet.ascon, byte level."""
    rng = random.Random(5)
    seeds = [0, (1 << 128) - 1] + [rng.getrandbits(SEED_BITS) for _ in range(14)]
    for k, s in ((64, 64), (32, 64)):
        B = Builder()
        sd = B.input(0, "seed", SEED_BITS)
        sL, tL, sR, tR = ca.prg_G(B, sd)
        B.output("sL", sL)
        B.output("tL", tL)
        B.output("sR", sR)
        B.output("tR", tR)
        for name, fn in (("C", ca.conv), ("VL", ca.convV_L), ("VR", ca.convV_R)):
            v, t = fn(B, sd, k, s)
            B.output(f"{name}_v", v)
            B.output(f"{name}_t", t)
        B.output("Y", ca.prg_F(B, sd, ca.IV_G))
        C = B.build()
        check_eq(C.stats()["and_depth"], 12, "PRG AND depth")
        for o, seed in zip(evaluate_batch(C, [{(0, "seed"): x} for x in seeds]), seeds):
            v = outputs_as_ints(o)
            sb = seed.to_bytes(16, "little")
            g = ascon.G(sb)
            check_eq((v["sL"].to_bytes(16, "little"), v["tL"], v["sR"].to_bytes(16, "little"), v["tR"]), g, "G")
            check_eq((v["C_v"], v["C_t"]), ascon.conv(sb, k, s), "Conv")
            check_eq((v["VL_v"], v["VL_t"]), ascon.convv_L(sb, k, s), "ConvV_L")
            check_eq((v["VR_v"], v["VR_t"]), ascon.convv_R(sb, k, s), "ConvV_R")
            # the scalar integer references used elsewhere agree as well
            check_eq(ca.G_ref(seed), (v["sL"], v["tL"], v["sR"], v["tR"]), "G_ref")
            check_eq(ca.conv_ref(seed, k, s), (v["C_v"], v["C_t"]), "conv_ref")
            check_eq(ca.convV_L_ref(seed, k, s), (v["VL_v"], v["VL_t"]), "convV_L_ref")
            check_eq(ca.convV_R_ref(seed, k, s), (v["VR_v"], v["VR_t"]), "convV_R_ref")
            # feed-forward, straight from the definition F_IV(s) = p(s||IV) ^ (s||IV)
            inp = [*ascon.words_from_bytes(sb), *ascon.label_words(ascon.LABEL_G)]
            y = sum((a ^ b) << (64 * i) for i, (a, b) in enumerate(zip(ascon.permute(inp), inp)))
            check_eq(v["Y"], y, "F_G feed-forward")
    st = ca.prg_call_stats()
    check_eq((st["n_and"], st["and_depth"]), (3584, 12), "one PRG call on a secret seed")


# --------------------------------------------------------------------------- #
# Builder / Circuit unit tests
# --------------------------------------------------------------------------- #

def test_builder_arithmetic():
    for adder in ("ripple", "sklansky"):
        B = Builder(adder=adder)
        a = B.input(0, "a", 64)
        b = B.input(1, "b", 64)
        sel = B.input(0, "sel", 1)[0]
        c = B.input(1, "c", 40)
        d = B.input(BOUND, "d", 65)
        e = B.input(0, "e", 80)
        s, cout = B.add_mod(a, b)
        df, nb = B.sub_mod(a, b)
        B.output("s", s)
        B.output("cout", cout)
        B.output("d", df)
        B.output("nb", nb)
        B.output("neg", B.neg_mod(a))
        B.output("cneg", B.cond_neg(sel, a))
        B.output("negif", B.neg_if(sel, c))
        B.output("mul", B.mul(a, b, 128))
        B.output("mul_trunc", B.mul(a, c, 70))
        B.output("mul_delta", B.mul(e, d, 80))
        B.output("mul_low", B.mul(d[:64], a, 64))
        B.output("sum3", B.sum_mod([(a, False), (b, True), (c, True)], 64))
        B.output("sum4", B.sum_mod([(a, True), (b, True), (c, False), (a, False)], 80))
        B.output("sum_const", B.sum_mod([(B.const_bits(1, 128), False), (a, True), (b, False)], 128))
        B.output("sum9", B.sum_mod([(a, False), (b, False), (c, False), (d, True), (e, False),
                                    (a, False), (b, True), (c, False), (e, True)], 90))
        B.output("mux", B.mux(sel, a, b))
        B.output("addc", B.add_mod(a, c, cin=sel, width=64)[0])
        B.output("addk", B.add_mod(a[:16], B.const_bits(0x5555, 16), width=16)[0])
        C = B.build()
        C.validate()
        M64 = (1 << 64) - 1
        rng = random.Random(3)
        cases = []
        for _ in range(40):
            cases.append({(0, "a"): rng.getrandbits(64), (1, "b"): rng.getrandbits(64),
                          (0, "sel"): rng.getrandbits(1), (1, "c"): rng.getrandbits(40),
                          (BOUND, "d"): rng.getrandbits(65), (0, "e"): rng.getrandbits(80)})
        cases.append({(0, "a"): M64, (1, "b"): M64, (0, "sel"): 1, (1, "c"): (1 << 40) - 1,
                      (BOUND, "d"): (1 << 65) - 1, (0, "e"): (1 << 80) - 1})
        cases.append({(0, "a"): 0, (1, "b"): 0, (0, "sel"): 1, (1, "c"): 0, (BOUND, "d"): 0, (0, "e"): 0})
        for inp, o in zip(cases, evaluate_batch(C, cases)):
            x, y, sl, z = inp[(0, "a")], inp[(1, "b")], inp[(0, "sel")], inp[(1, "c")]
            dd, ee = inp[(BOUND, "d")], inp[(0, "e")]
            v = outputs_as_ints(o)
            check(v["s"] == (x + y) & M64 and v["cout"] == (x + y) >> 64, "add")
            check(v["d"] == (x - y) & M64 and v["nb"] == int(x >= y), "sub")
            check_eq(v["neg"], (-x) & M64, "neg")
            check_eq(v["cneg"], (-x) & M64 if sl else x, "cond_neg")
            check_eq(v["negif"], (-z) % (1 << 40) if sl else z, "neg_if")
            check_eq(v["mul"], x * y, "mul 64x64")
            check_eq(v["mul_trunc"], (x * z) % (1 << 70), "mul truncated")
            check_eq(v["mul_delta"], (dd * ee) % (1 << 80), "mul (s+1) x (n+s)")
            check_eq(v["mul_low"], (dd % (1 << 64)) * x % (1 << 64), "mul low half")
            check_eq(v["sum3"], (x - y - z) & M64, "sum3")
            check_eq(v["sum4"], (-x - y + z + x) % (1 << 80), "sum4")
            check_eq(v["sum_const"], (1 - x + y) % (1 << 128), "sum with constant")
            check_eq(v["sum9"], (x + y + z - dd + ee + x - y + z - ee) % (1 << 90), "sum9")
            check_eq(v["mux"], x if sl else y, "mux")
            check_eq(v["addc"], (x + z + sl) & M64, "add with carry-in")
            check_eq(v["addk"], ((x & 0xFFFF) + 0x5555) & 0xFFFF, "add constant")
        st = C.stats()
        if adder == "ripple":
            check(st["and_depth"] >= 127, "ripple depth")
        else:
            check(st["and_depth"] <= 24, "sklansky depth")


def test_adder_and_multiplier_counts():
    """Documented costs: ripple adder one AND per bit (last carry dead)."""
    for adder, expect_and, max_depth in (("ripple", 127, 127), ("sklansky", 895, 8)):
        B = Builder(adder=adder)
        a = B.input(0, "a", 128)
        b = B.input(1, "b", 128)
        B.output("s", B.add_mod(a, b)[0])
        st = B.build().stats()
        check_eq(st["n_and"], expect_and, f"{adder} adder ANDs")
        check(st["and_depth"] <= max_depth, f"{adder} adder depth")
    B = Builder()
    a = B.input(0, "a", 128)
    sel = B.input(0, "sel", 1)[0]
    B.output("m", B.mux(sel, a, [ZERO] * 128))
    check_eq(B.build().stats()["n_and"], 128, "mux ANDs")
    B = Builder()
    a = B.input(0, "a", 128)
    sel = B.input(0, "sel", 1)[0]
    B.output("m", B.neg_if(sel, a))
    check_eq(B.build().stats()["n_and"], 127, "neg_if ANDs")
    B = Builder()
    a = B.input(0, "a", 64)
    b = B.input(1, "b", 64)
    B.output("m", B.mul(a, b, 128))
    st = B.build().stats()
    check(4096 + 127 <= st["n_and"] <= 2 * 4096, "64x64 multiplier ANDs about 2P - w")
    B = Builder(adder="sklansky")
    a = B.input(0, "a", 64)
    b = B.input(1, "b", 64)
    B.output("m", B.mul(a, b, 128))
    check(B.build().stats()["and_depth"] <= 24, "Dadda reduction is logarithmic in depth")


def test_constant_folding_and_dead_code():
    B = Builder()
    a = B.input(0, "a", 4)
    check(B.and_(a[0], ZERO) == ZERO and B.and_(a[0], ONE) == a[0], "AND folding")
    check(B.xor(a[0], ZERO) == a[0] and B.xor(a[0], a[0]) == ZERO, "XOR folding")
    na = B.not_(a[0])
    check(B.not_(na) == a[0] and B.xor(a[0], ONE) == na, "NOT folding")
    check(B.and_(a[0], na) == ZERO and B.xor(a[0], na) == ONE, "x op ~x folding")
    dead = B.and_(a[1], a[2])          # never used by an output
    B.and_(dead, a[3])
    B.output("o", [B.and_(a[0], a[1]), na, ONE, a[3]])
    C = B.build()
    st = C.stats()
    check(st["n_and"] == 1 and st["n_not"] == 1 and st["n_gates"] == 2, "dead gates removed")
    check_eq(evaluate(C, {(0, "a"): [1, 1, 0, 1]})["o"], [1, 0, 1, 1], "folded circuit value")


def test_owners_bound_and_retain():
    B = Builder()
    a = B.input(0, "a", 4)
    b = B.input(1, "a", 4)             # same label, different owner
    d = B.input(BOUND, "delta", 3)
    check_raises(ValueError, B.input, 3, "x", 1)
    check_raises(ValueError, B.input, 0, "a", 4)
    check_raises(ValueError, B.input, BOUND, "delta", 3)
    check_raises(ValueError, B.input, 0, "bad label", 1)
    s, c = B.add_mod(a, b)
    B.output("sum", s)
    B.output("carry", c, to=0)
    B.output("keep", B.xor_bits(d, a[:3]) + [c], to=RETAIN)
    B.output("mix", [a[0], ONE, ZERO, s[0], s[0], d[2]], to=1)
    check_raises(ValueError, B.output, "sum", s)
    check_raises(ValueError, B.output, "x", s, to=2)
    check_raises(ValueError, B.output, "x", s, to="bound")
    C = B.build()
    C.validate()
    st = C.stats()
    check_eq((st["n_input_bits_p0"], st["n_input_bits_p1"], st["n_bound_input_bits"]), (4, 4, 3), "input bits")
    check_eq((st["n_public_output_bits"], st["n_private_output_bits_p0"], st["n_private_output_bits_p1"],
              st["n_retain_output_bits"]), (4, 1, 6, 4), "output bits")
    check_eq(st["n_private_output_bits"], 7, "private output bits")
    check_eq(C.output_owner("keep"), RETAIN, "RETAIN owner")
    check_eq(C.output_owner("sum"), PUBLIC, "PUBLIC owner")
    check_eq(C.input_wires(BOUND, "delta"), [10, 11, 12], "BOUND wires")
    check(C.has_input(BOUND, "delta") and not C.has_input(0, "delta"), "has_input")
    for x in range(16):
        for y in (0, 3, 9, 15):
            for dv in (0, 5, 7):
                o = outputs_as_ints(evaluate(C, {(0, "a"): x, (1, "a"): y, (BOUND, "delta"): dv}))
                check_eq(o["sum"], (x + y) & 15, "sum")
                check_eq(o["carry"], (x + y) >> 4, "carry")
                check_eq(o["keep"], (dv ^ (x & 7)) | (((x + y) >> 4) << 3), "retained output")
                check_eq(o["mix"], (x & 1) | 2 | (((x + y) & 1) << 3) | (((x + y) & 1) << 4) | ((dv >> 2) << 5), "mix")
    check_raises(KeyError, evaluate, C, {(0, "a"): 1, (1, "a"): 1})
    check_raises(ValueError, evaluate, C, {(0, "a"): 1, (1, "a"): 1, (BOUND, "delta"): 8})


def _check_schedule(C: Circuit) -> None:
    n_in = C.n_input_wires
    layers = C.layers()
    check_eq(len(layers), C.and_depth(), "layer count")
    produced = bytearray(C.n_wires)
    produced[0] = produced[1] = 1
    for w in range(2, 2 + n_in):
        produced[w] = 1
    depth = [0] * C.n_wires
    layer_of = {}
    for li, (s, e) in enumerate(C.layer_ranges()):
        for g in range(s, e):
            layer_of[g] = li + 1
    ok = True
    for gi, (op, a, b, o) in enumerate(C.gates()):
        ok &= bool(produced[a] and produced[b] and not produced[o])
        produced[o] = 1
        d = max(depth[a], depth[b])
        if op == AND:
            d += 1
            ok &= layer_of.get(gi) == d and depth[a] < d and depth[b] < d
        else:
            ok &= gi not in layer_of
        depth[o] = d
    check(ok, "topological order and AND layers")
    all_and = [gi for gi, op in enumerate(C.op) if op == AND]
    check_eq(sorted(g for lay in layers for g in lay), all_and, "every AND gate in exactly one layer")
    sched = C.schedule()
    check_eq(len(sched), len(layers) + 1, "schedule length")
    ok = True
    prev = 0
    for li, ((ls, le), (as_, ae)) in enumerate(sched):
        ok &= ls == prev and le == as_
        prev = ae
        for g in range(ls, le):
            ok &= C.op[g] != AND and depth[C.out[g]] == li
        for g in range(as_, ae):
            ok &= C.op[g] == AND
    check(ok and prev == C.n_gates, "schedule covers the gate list; linear gates have the layer depth")


def test_layers_are_valid_schedule():
    for C in (cg.build_dpf(2, 32, S), cg.build_dcf(2, 32, S, "add", "public_one"),
              cg.build_gate(3, 32, S, True, True, [(3, "r")], 1), cg.build_session_setup(S),
              cg.build_dpf(2, 32, S, adder="sklansky")):
        C.validate()
        check(True, "validate")
        _check_schedule(C)


def test_serialisation_roundtrip_and_digest():
    C = cg.build_gate(3, 32, S, True, True, [(3, "one"), (8, "r")], 1)
    data = C.to_bytes()
    check_eq(data[:8], b"CPLTCIR2", "magic / format version")
    D = Circuit.from_bytes(data)
    check(D.to_bytes() == data and D.digest() == C.digest(), "binary round trip")
    check(D.inputs == C.inputs and D.outputs == C.outputs and D.layer_ranges() == C.layer_ranges(),
          "declarations and layers survive")
    check(any(p == BOUND for p, _, _ in D.inputs), "BOUND input survives")
    T = Circuit.from_text(C.to_text())
    check(T.to_bytes() == data, "text round trip")
    S0 = cg.build_session_setup(S)
    S1 = Circuit.from_bytes(S0.to_bytes())
    S2 = Circuit.from_text(S0.to_text())
    check(S1.output_owner("delta") == RETAIN and S2.output_owner("delta") == RETAIN, "RETAIN survives")
    check(S1.digest() == S0.digest() == S2.digest(), "setup digests")
    # deterministic compilation (same descriptor -> same gate list / hash)
    check_eq(cg.build(C.meta["descriptor"]).digest(), C.digest(), "rebuild from descriptor")
    check(cg.build_dpf(1, 32, S, "xor", "public_one").digest() == cg.build_dpf(1, 32, S, "xor", "public_one").digest(),
          "deterministic digest")
    digests = {cg.build_dpf(2, 32, S, im, bm, adder=ad).digest()
               for im in ("xor", "add") for bm in ("secret", "public_one") for ad in ("ripple", "sklansky")}
    check_eq(len(digests), 8, "different descriptors, different digests")
    check_eq(cg.build_dpf(1, 32, S, "add").digest(), cg.build_dpf(1, 32, S, "xor").digest(),
             "for n = 1 the additive and the XOR index coincide")
    # the three forms evaluate identically
    rng = random.Random(77)
    inp = cg.random_inputs(C, rng)
    check(evaluate(C, inp) == evaluate(D, inp) == evaluate(T, inp), "same values")
    # text format sanity: one gate per line after 'gates'
    txt = C.to_text().splitlines()
    gi = txt.index("gates")
    check_eq(len(txt) - gi - 1, C.n_gates, "one gate per line")
    check(txt[1] == "version 2" and txt[gi + 1].split()[-1] in ("XOR", "AND", "INV"), "text header")
    check(any(line.startswith("input bound delta 65 ") for line in txt), "text BOUND line")
    check(any(line.startswith("output retain delta 65 ") for line in S0.to_text().splitlines()), "text RETAIN line")
    # rejected encodings
    check_raises(ValueError, Circuit.from_bytes, b"CPLTCIR1" + data[8:])      # old format version
    check_raises(ValueError, Circuit.from_bytes, b"XXXXXXXX" + data[8:])
    check_raises(ValueError, Circuit.from_bytes, data[:-4])
    check_raises(ValueError, Circuit.from_bytes, data + b"\x00\x00\x00\x00")
    check_raises(ValueError, Circuit.from_text, C.to_text().replace("version 2", "version 1"))
    bad = bytearray(data)
    off = 8 + 20 + 16 * (C.layer_ranges()[0][0])          # first AND gate: make it a XOR
    check_eq(int.from_bytes(bad[off:off + 4], "little"), AND, "located the first AND gate")
    bad[off:off + 4] = (XOR).to_bytes(4, "little")
    check_raises(ValueError, Circuit.from_bytes, bytes(bad))                   # linear gate inside an AND layer
    bad = bytearray(data)
    last = 8 + 20 + 16 * (C.n_gates - 1)
    bad[last + 4:last + 8] = (C.n_wires - 1).to_bytes(4, "little")           # gate reads its own output
    check_raises(ValueError, Circuit.from_bytes, bytes(bad))
    check(Circuit.from_bytes(bytes(bad), validate=False) is not None, "validation can be skipped")


def test_bristol_export():
    rng = random.Random(4242)
    circuits = [cg.build_session_setup(S), cg.build_dpf(2, 32, S, "add", "secret"),
                cg.build_dcf(1, 32, S, "xor", "public_one"),
                cg.build_gate(3, 32, S, True, True, [(0, "one"), (8, "r")], 1)]
    B = Builder()                       # outputs that need copies and constants
    a = B.input(0, "a", 4)
    b = B.input(1, "b", 4)
    d = B.input(BOUND, "d", 3)
    s, c = B.add_mod(a, b)
    B.output("s", s)
    B.output("c", c, to=0)
    B.output("keep", B.xor_bits(d, a[:3]), to=RETAIN)
    B.output("mix", [a[0], ONE, ZERO, s[0], s[0], d[2]], to=1)
    circuits.append(B.build())
    for C in circuits:
        text = C.to_bristol()
        lay = C.bristol_layout()
        head = text.split("\n", 3)
        n_gates, n_wires = (int(x) for x in head[0].split())
        st = C.stats()
        check_eq(text.count(" AND\n"), st["n_and"], "Bristol AND count")
        check(n_gates >= C.n_gates and n_gates <= C.n_gates + 2 + st["n_outputs"], "Bristol gate count")
        owners = [g["owner"] for g in lay["inputs"]]
        check_eq(owners, [o for o in (0, 1, "bound") if o in owners], "input values ordered P0, P1, bound")
        check_eq([int(x) for x in head[1].split()], [len(lay["inputs"])] + [g["width"] for g in lay["inputs"]],
                 "input value widths")
        check_eq([int(x) for x in head[2].split()], [len(C.outputs)] + [len(w) for _, _, w in C.outputs],
                 "output value widths")
        check_eq([(o["label"], o["owner"]) for o in lay["outputs"]], [(l, t) for t, l, _ in C.outputs],
                 "output layout")
        for _ in range(4):
            inp = {(p, l): rng.getrandbits(len(w)) for p, l, w in C.inputs}
            exp = outputs_as_ints(evaluate(C, inp))
            got = eval_bristol(text, bristol_values(C, inp))
            check_eq(got, [exp[l] for _, l, _ in C.outputs], "Bristol evaluation")


def test_evaluate_batch_matches_single_and_int_inputs():
    B = Builder()
    a = B.input(0, "a", 8)
    b = B.input(1, "b", 8)
    B.output("p", B.mul(a, b, 16))
    B.output("n", B.not_bits(a))
    C = B.build()
    rng = random.Random(1)
    cases = [{(0, "a"): rng.getrandbits(8), (1, "b"): int_to_bits(rng.getrandbits(8), 8)} for _ in range(70)]
    batch = evaluate_batch(C, cases)
    for inp, o in zip(cases, batch):
        check(evaluate(C, inp) == o, "batch equals single")
        check_eq(bits_to_int(o["p"]), inp[(0, "a")] * bits_to_int(inp[(1, "b")]), "product")
        check_eq(bits_to_int(o["n"]), inp[(0, "a")] ^ 0xFF, "NOT")
    check_raises(KeyError, evaluate, C, {(0, "a"): 1})
    check_raises(ValueError, evaluate, C, {(0, "a"): 256, (1, "b"): 0})


# --------------------------------------------------------------------------- #
# Session setup
# --------------------------------------------------------------------------- #

def test_session_setup():
    for s in (64, 40):
        for adder in ("ripple", "sklansky"):
            C = cg.build_session_setup(s, adder=adder)
            st = C.stats()
            check_eq([(p, l, len(w)) for p, l, w in C.inputs], [(0, "delta", s), (1, "delta", s)], "inputs")
            check_eq([(t, l, len(w)) for t, l, w in C.outputs], [(RETAIN, "delta", s + 1)], "RETAIN output")
            if adder == "ripple":
                check_eq(st["n_and"], s, "s AND gates")
            check_eq(st["n_retain_output_bits"], s + 1, "retained bits")
            rng = random.Random(1000 + s)
            top = (1 << s) - 1
            cases = [(0, 0), (top, top), (top, 1), (1, top), (1 << (s - 1), 1 << (s - 1))]
            cases += [(rng.getrandbits(s), rng.getrandbits(s)) for _ in range(40)]
            ins = [{(0, "delta"): d0, (1, "delta"): d1} for d0, d1 in cases]
            n_big = 0
            for (d0, d1), inp, o in zip(cases, ins, evaluate_batch(C, ins)):
                got = bits_to_int(o["delta"])
                check_eq(got, d0 + d1, "Delta = Delta_0 + Delta_1, not reduced")
                check_eq(cg.expected_outputs_session_setup(s, inp), {"delta": d0 + d1}, "reference")
                check_eq(cg.expected_outputs(C.meta["descriptor"], inp), {"delta": got}, "dispatch")
                n_big += got >> s
            check(5 <= n_big <= len(cases) - 5, "MAC keys on both sides of 2^s")


# --------------------------------------------------------------------------- #
# DPF / DCF generation circuits
# --------------------------------------------------------------------------- #

def _fss_reference(family, n, k, s, index_mode, beta_mode, inp, prefix=""):
    """(alpha, beta, delta, key0, key1) computed in the test from the plaintext inputs."""
    a0, a1 = inp[(0, prefix + "alpha")], inp[(1, prefix + "alpha")]
    alpha = a0 ^ a1 if index_mode == "xor" else (a0 + a1) % (1 << n)
    beta = 1
    if beta_mode == "secret":
        beta = (inp[(0, prefix + "beta")] + inp[(1, prefix + "beta")]) % (1 << k)
    delta = inp[(BOUND, "delta")]
    roots = _roots(inp, prefix)
    gen = fss.dpf_gen if family == "dpf" else fss.dcf_gen
    k0, k1 = gen(alpha, beta, delta, n, k, s, roots=roots)
    return alpha, beta, delta, k0, k1


def _roots(inp, prefix):
    return ((inp[(0, prefix + "rho0")] ^ inp[(1, prefix + "rho0")]).to_bytes(16, "little"),
            (inp[(0, prefix + "rho1")] ^ inp[(1, prefix + "rho1")]).to_bytes(16, "little"))


def _labels_from_keys(k0, k1, prefix=""):
    """Circuit output values that an fss key pair corresponds to (independent packing)."""
    w = k0.width
    out = {"cws": int.from_bytes(b"".join(k0.cw_s), "little"),
           "cwtl": int("".join(str(b) for b in reversed(k0.cw_tl)), 2),
           "cwtr": int("".join(str(b) for b in reversed(k0.cw_tr)), 2),
           "cwout_v": k0.cw_out_v, "cwout_t": k0.cw_out_t,
           "root0": int.from_bytes(k0.seed, "little"), "root1": int.from_bytes(k1.seed, "little")}
    if isinstance(k0, fss.DCFKey):
        out["vcw"] = sum(v << (w * i) for i, v in enumerate(k0.cw_v))
        out["tcw"] = sum(v << (w * i) for i, v in enumerate(k0.cw_t))
    return {prefix + lab: v for lab, v in out.items()}


def _public_part(key):
    d = dict(key.__dict__)
    d.pop("b")
    d.pop("seed")
    return d


def _check_key_on_domain(family, k0, k1, alpha, beta, delta):
    """Full-domain correctness of a key pair on both planes; returns the reconstructed vectors."""
    n, w = k0.n, k0.width
    mod = 1 << w
    ev = fss.dpf_eval_all if family == "dpf" else fss.dcf_eval_all
    y0, t0 = ev(0, k0)
    y1, t1 = ev(1, k1)
    val = [(int(a) + int(b)) % mod for a, b in zip(y0, y1)]
    tag = [(int(a) + int(b)) % mod for a, b in zip(t0, t1)]
    ind = [int(x == alpha) if family == "dpf" else int(x < alpha) for x in range(1 << n)]
    check_eq(val, [beta * i for i in ind], f"{family} value plane on the whole domain")
    check_eq(tag, [delta * beta * i % mod for i in ind], f"{family} tag plane on the whole domain")
    return (y0, t0), (y1, t1)


def _deltas(rng, s, count):
    """MAC keys for ``count`` assignments: the maximum, one >= 2^s, one < 2^s, then random sums."""
    top = (1 << s) - 1
    out = [2 * top, (1 << s) + rng.getrandbits(s - 1), rng.getrandbits(s - 1)]
    while len(out) < count:
        out.append(rng.getrandbits(s) + rng.getrandbits(s))
    return out[:count]


def _run_keygen_family(family: str, trials: int = 5) -> None:
    build = cg.build_dpf if family == "dpf" else cg.build_dcf
    expected = cg.expected_outputs_dpf if family == "dpf" else cg.expected_outputs_dcf
    for n in range(1, 7):
        for k in (32, 64):
            w = k + S
            for index_mode in ("xor", "add"):
                for beta_mode in ("secret", "public_one"):
                    tag = f"{family} n={n} k={k} {index_mode} {beta_mode}"
                    C = build(n, k, S, index_mode, beta_mode)
                    check_eq(C.input_wires(BOUND, "delta"), list(range(2, 2 + S + 1)), f"{tag}: BOUND delta first")
                    check_eq(C.stats()["n_bound_input_bits"], S + 1, f"{tag}: delta is s+1 bits")
                    check(not C.has_input(0, "delta") and not C.has_input(1, "delta"), f"{tag}: no party inputs delta")
                    check_eq(C.has_input(0, "beta"), beta_mode == "secret", f"{tag}: beta input")
                    owners = {l: t for t, l, _ in C.outputs}
                    exp_owner = {"cws": PUBLIC, "cwtl": PUBLIC, "cwtr": PUBLIC, "cwout_v": PUBLIC,
                                 "cwout_t": PUBLIC, "root0": 0, "root1": 1}
                    widths = {"cws": 128 * n, "cwtl": n, "cwtr": n, "cwout_v": w, "cwout_t": w,
                              "root0": 128, "root1": 128}
                    if family == "dcf":
                        exp_owner.update(vcw=PUBLIC, tcw=PUBLIC)
                        widths.update(vcw=w * n, tcw=w * n)
                    check_eq(owners, exp_owner, f"{tag}: output owners")
                    check_eq({l: len(x) for _, l, x in C.outputs}, widths, f"{tag}: output widths")
                    rng = random.Random(f"{tag}-v3.1")
                    cases = [cg.random_inputs(C, rng, delta=d) for d in _deltas(rng, S, trials)]
                    if index_mode == "add":           # force a wrap of the additive index shares
                        cases[0][(0, "alpha")] = cases[0][(1, "alpha")] = (1 << n) - 1
                    if beta_mode == "secret":         # and of the payload shares; also payload 2^k - 1
                        cases[0][(0, "beta")] = (1 << k) - 1
                        cases[0][(1, "beta")] = (1 << k) - 1
                        cases[1][(0, "beta")], cases[1][(1, "beta")] = (1 << k) - 1, 0
                    for inp, o in zip(cases, evaluate_batch(C, cases)):
                        got = outputs_as_ints(o)
                        alpha, beta, delta, k0, k1 = _fss_reference(family, n, k, S, index_mode, beta_mode, inp)
                        # bit for bit: every circuit output equals the plaintext generator's
                        check_eq(got, _labels_from_keys(k0, k1), f"{tag}: outputs vs fss.{family}_gen")
                        check_eq(cg.expected_outputs(C.meta["descriptor"], inp), got, f"{tag}: expected_outputs")
                        check_eq(expected(n, k, S, index_mode, beta_mode, inp), got, f"{tag}: expected_outputs_*")
                        c0, c1 = cg.unpack_key_outputs(o, family, n, k, S)
                        check(c0 == k0 and c1 == k1, f"{tag}: keys from circuit outputs equal fss keys")
                        check(c0.to_bytes() == k0.to_bytes() and c1.to_bytes() == k1.to_bytes(),
                              f"{tag}: key encodings identical")
                        check_eq(_public_part(c0), _public_part(c1), f"{tag}: both keys share the CWs")
                        check_eq(cg.pack_key_outputs(c0, c1), got, f"{tag}: pack/unpack")
                        check_eq(cg.keygen_plaintext(n, k, S, index_mode, beta_mode, inp),
                                 {"alpha": alpha, "beta": beta}, f"{tag}: plaintext index/payload")
                        # full-domain correctness of the circuit-generated keys (unreduced Delta)
                        _check_key_on_domain(family, c0, c1, alpha, beta, delta)


def test_dpf_circuits():
    _run_keygen_family("dpf")


def test_dcf_circuits():
    _run_keygen_family("dcf")


def test_and_counts_per_level():
    """DESIGN.md section 4: DPF costs 2*3584 + 3*128 + 3 = 7,555 ANDs per level."""
    for k in (32, 64):
        for beta_mode in ("public_one", "secret"):
            a = {n: cg.build_dpf(n, k, S, "xor", beta_mode).stats() for n in (1, 2, 3, 4)}
            check_eq(a[3]["n_and"] - a[2]["n_and"], 7555, "DPF per-level ANDs")
            check_eq(a[4]["n_and"] - a[3]["n_and"], 7555, "DPF per-level ANDs")
            check_eq(a[2]["n_and"] - a[1]["n_and"], 7555, "DPF per-level ANDs")
            check_eq(a[4]["and_depth"] - a[3]["and_depth"], 14, "DPF per-level AND depth")
            d = {n: cg.build_dcf(n, k, S, "xor", beta_mode).stats() for n in (2, 3, 4)}
            per = d[3]["n_and"] - d[2]["n_and"]
            check_eq(d[4]["n_and"] - d[3]["n_and"], per, "DCF ANDs are affine in n")
            check(6 * 3584 + 387 < per < 6 * 3584 + 387 + 17 * (k + S), "DCF arithmetic below 17 w per level")
            check_eq(d[4]["and_depth"] - d[3]["and_depth"], 14, "DCF per-level AND depth")
    x = cg.build_dpf(4, 64, S, "xor", "public_one").stats()["n_and"]
    y = cg.build_dpf(4, 64, S, "add", "public_one").stats()["n_and"]
    check_eq(y - x, 3, "additive index costs n - 1 ANDs")


# --------------------------------------------------------------------------- #
# Gate circuits
# --------------------------------------------------------------------------- #

def test_comparison_identity_exhaustive():
    """DESIGN.md section 6, plain integers: for n <= 8, ALL r, x and ALL c in [0, N],
    [x < c] = [x_hat < (r + c) mod N] - [x_hat < r] + w_c with x_hat = (x + r) mod N and
    w_c = [r + c >= N]; and the interval form for n <= 5, all p <= q."""
    for n in range(1, 9):
        N = 1 << n
        r = np.arange(N, dtype=np.int64)[:, None]
        x = np.arange(N, dtype=np.int64)[None, :]
        x_hat = (x + r) % N
        below_r = (x_hat < r).astype(np.int64)
        terms = []
        for c in range(N + 1):
            d_c = (x_hat < (r + c) % N).astype(np.int64)
            w_c = (r + c >= N).astype(np.int64)
            rhs = d_c - below_r + w_c
            check(bool(np.array_equal(rhs, np.broadcast_to((x < c).astype(np.int64), rhs.shape))),
                  f"comparison identity n={n} c={c}")
            terms.append(d_c + w_c)
        if n <= 5:
            for p in range(N):
                for q in range(p, N):
                    inside = np.broadcast_to(((p <= x) & (x <= q)).astype(np.int64), (N, N))
                    check(bool(np.array_equal(terms[q + 1] - terms[p], inside)), f"interval identity n={n} [{p},{q}]")


def _gate_cases(C, rng, n, count=3):
    """Assignments with r = N - 1 (wrap for every c >= 1), r = 0 (wrap only for c = N), random r."""
    N = 1 << n
    targets = [N - 1, 0] + [None] * (count - 2)
    cases = []
    for delta, target in zip(_deltas(rng, S, count), targets):
        inp = cg.random_inputs(C, rng, delta=delta)
        if target is not None:
            c0, c1 = inp[(0, "rp.coin")], inp[(1, "rp.coin")]
            inp[(1, "rp.coin")] = (c1 & ~(N - 1)) | ((c0 ^ target) & (N - 1))
        cases.append(inp)
    return cases


def _check_gate(n, k, auth_input, dpf_one, thresholds, n_outmask, rng):
    s = S
    w = k + s
    N, mod, mod_s = 1 << n, 1 << w, 1 << s
    tag = f"gate n={n} k={k} auth={auth_input} dpf={dpf_one} om={n_outmask}"
    C = cg.build_gate(n, k, s, auth_input, dpf_one, thresholds, n_outmask)
    m = n + s if auth_input else n
    # declarations
    check_eq(C.stats()["n_bound_input_bits"], s + 1, f"{tag}: BOUND delta")
    check_eq(len(C.input_wires(0, "rp.coin")), m, f"{tag}: coin width")
    check_eq(C.has_input(0, "rp.mu_sigma"), auth_input, f"{tag}: tag share only if auth_input")
    check_eq(C.has_output("rp.mu_share1"), auth_input, f"{tag}: tag share only if auth_input")
    check_eq(C.has_output("dpf.cws"), dpf_one, f"{tag}: DPF present")
    check(all(p == 0 for p, l, _ in C.inputs if l.endswith("sigma")), f"{tag}: sigma values come from P_0")
    check(all(t == 1 for t, l, _ in C.outputs if l.endswith("share1")), f"{tag}: complements go to P_1")
    check(all(t == PUBLIC for t, l, _ in C.outputs if ".cw" in l), f"{tag}: correction words public")
    check(all(t == int(l[-1]) for t, l, _ in C.outputs if l.endswith(("root0", "root1"))), f"{tag}: roots private")
    check(not any(t == RETAIN for t, _, _ in C.outputs), f"{tag}: nothing retained")
    cases = _gate_cases(C, rng, n)
    table = [rng.getrandbits(k) for _ in range(N)]
    for inp, o in zip(cases, evaluate_batch(C, cases)):
        got = outputs_as_ints(o)
        delta = inp[(BOUND, "delta")]
        # bit for bit against the plaintext reference
        check_eq(cg.expected_outputs_gate(n, k, s, auth_input, dpf_one, thresholds, n_outmask, inp), got,
                 f"{tag}: expected_outputs_gate")
        # 1. r = r' mod N and its sharing
        rp = inp[(0, "rp.coin")] ^ inp[(1, "rp.coin")]
        r = rp % N
        pt = cg.gate_plaintext(n, k, s, auth_input, dpf_one, thresholds, n_outmask, inp)
        check_eq((pt["rp"], pt["r"]), (rp, r), f"{tag}: r = r' mod N")
        share = inp[(0, "rp.sigma")] + got["rp.share1"]
        if auth_input:
            check_eq(share % (1 << m), rp, f"{tag}: [[r']] value shares over Z_2^(n+s)")
            check_eq((inp[(0, "rp.mu_sigma")] + got["rp.mu_share1"]) % (1 << m), delta * rp % (1 << m),
                     f"{tag}: [[r']] tag shares with the unreduced Delta")
            check_eq(share % N, r, f"{tag}: [[r']] reduces to r mod 2^n")
        else:
            check_eq(share % N, r, f"{tag}: additive shares of r mod 2^n")
        # 2. DPF at index r, payload 1, and the table gate
        if dpf_one:
            k0, k1 = cg.unpack_key_outputs(o, "dpf", n, k, s, "dpf.")
            r0, r1 = fss.dpf_gen(r, 1, delta, n, k, s, roots=_roots(inp, "dpf."))
            check(k0.to_bytes() == r0.to_bytes() and k1.to_bytes() == r1.to_bytes(), f"{tag}: DPF key at index r")
            (y0, t0), (y1, t1) = _check_key_on_domain("dpf", k0, k1, r, 1, delta)
            for x in range(N):
                x_hat = (x + r) % N
                val = sum(table[(x_hat - u) % N] * (int(y0[u]) + int(y1[u])) for u in range(N)) % mod
                tg = sum(table[(x_hat - u) % N] * (int(t0[u]) + int(t1[u])) for u in range(N)) % mod
                check(val == table[x] and tg == delta * table[x] % mod, f"{tag}: table gate at x={x}")
        # 3. thresholds: DCF index, w_c, r w_c
        rec = {}
        for i, (c, payload) in enumerate(thresholds):
            q = f"th{i}."
            alpha_c, w_c = (r + c) % N, int(r + c >= N)
            check_eq((pt["alpha"][i], pt["w"][i]), (alpha_c, w_c), f"{tag}: alpha_c, w_c")
            beta = 1 if payload == "one" else r
            k0, k1 = cg.unpack_key_outputs(o, "dcf", n, k, s, q)
            r0, r1 = fss.dcf_gen(alpha_c, beta, delta, n, k, s, roots=_roots(inp, q))
            check(k0.to_bytes() == r0.to_bytes() and k1.to_bytes() == r1.to_bytes(),
                  f"{tag}: DCF key {i} at index (r + c) mod N")
            shares = _check_key_on_domain("dcf", k0, k1, alpha_c, beta, delta)
            w_sh = ((inp[(0, q + "w.sigma")], inp[(0, q + "w.mu_sigma")]),
                    (got[q + "w.share1"], got[q + "w.mu_share1"]))
            check_eq((w_sh[0][0] + w_sh[1][0]) % mod, w_c, f"{tag}: [[w_c]] value, c={c}")
            check_eq((w_sh[0][1] + w_sh[1][1]) % mod, delta * w_c % mod, f"{tag}: [[w_c]] tag, c={c}")
            rw_sh = None
            if payload == "r":
                rw_sh = ((inp[(0, q + "rw.sigma")], inp[(0, q + "rw.mu_sigma")]),
                         (got[q + "rw.share1"], got[q + "rw.mu_share1"]))
                check_eq((rw_sh[0][0] + rw_sh[1][0]) % mod, r * w_c, f"{tag}: [[r w_c]] value, c={c}")
                check_eq((rw_sh[0][1] + rw_sh[1][1]) % mod, delta * r * w_c % mod, f"{tag}: [[r w_c]] tag, c={c}")
            else:
                check(q + "rw.share1" not in got, f"{tag}: no [[r w_c]] for payload one")
            rec[(c, payload)] = (shares, w_sh, rw_sh)
        # 4. comparison identity of section 6, evaluated locally by each party, all x
        for (c, payload), (shares_c, w_sh, rw_sh) in rec.items():
            if (0, payload) not in rec:
                continue
            shares_0 = rec[(0, payload)][0]
            corr = w_sh if payload == "one" else rw_sh
            scale = 1 if payload == "one" else r
            ok = True
            for x in range(N):
                x_hat = (x + r) % N
                lt = int(x < c)
                # integer identity [x < c] = [x_hat < (r+c) mod N] - [x_hat < r] + w_c
                ok &= lt == int(x_hat < (r + c) % N) - int(x_hat < r) + int(r + c >= N)
                val = tg = 0
                for b in (0, 1):                 # party b's local evaluation
                    val += int(shares_c[b][0][x_hat]) - int(shares_0[b][0][x_hat]) + corr[b][0]
                    tg += int(shares_c[b][1][x_hat]) - int(shares_0[b][1][x_hat]) + corr[b][1]
                ok &= val % mod == scale * lt and tg % mod == delta * scale * lt % mod
                if payload == "r":               # ReLU-style combination x_hat*b - r*b = x*b mod 2^n
                    ok &= (x_hat * lt - val) % N == x * lt
            check(ok, f"{tag}: comparison identity, c={c}, payload {payload}")
        # 4b. interval identity [p <= x <= q] = D_{q+1} - D_p + w_{q+1} - w_p for every pair
        ones = [(c, v) for (c, payload), v in rec.items() if payload == "one"]
        for hi, (sh_hi, w_hi, _) in ones:
            for lo, (sh_lo, w_lo, _) in ones:
                if lo >= hi:
                    continue
                ok = True
                for x in range(N):
                    x_hat = (x + r) % N
                    val = tg = 0
                    for b in (0, 1):
                        val += int(sh_hi[b][0][x_hat]) - int(sh_lo[b][0][x_hat]) + w_hi[b][0] - w_lo[b][0]
                        tg += int(sh_hi[b][1][x_hat]) - int(sh_lo[b][1][x_hat]) + w_hi[b][1] - w_lo[b][1]
                    inside = int(lo <= x < hi)
                    ok &= val % mod == inside and tg % mod == delta * inside % mod
                check(ok, f"{tag}: interval identity [{lo}, {hi})")
        # 5. output masks: value 2^k rho, tag Delta 2^k rho mod 2^w
        for j in range(n_outmask):
            q = f"om{j}."
            rho = inp[(0, q + "coin")] ^ inp[(1, q + "coin")]
            check_eq(pt["rho"][j], rho, f"{tag}: rho")
            v0, v1 = (inp[(0, q + "sigma")] << k), (got[q + "share1"] << k)
            m0, m1 = (inp[(0, q + "mu_sigma")] << k), (got[q + "mu_share1"] << k)
            check(v0 < mod and v1 < mod and m0 < mod and m1 < mod and got[q + "share1"] < mod_s,
                  f"{tag}: mask shares are s-bit values shifted by k")
            check_eq((v0 + v1) % mod, (rho << k) % mod, f"{tag}: output mask value 2^k rho")
            check_eq((m0 + m1) % mod, delta * (rho << k) % mod, f"{tag}: output mask tag Delta 2^k rho")
        check_eq(len(pt["rho"]), n_outmask, f"{tag}: number of masks")
    return C


def test_gate_circuits():
    ns = (3, 4) if QUICK else (3, 4, 5, 6)
    for n in ns:
        k = 64 if n % 2 == 0 else 32
        N = 1 << n
        thresholds = [(c, payload) for c in (0, 1, N // 2, N - 1, N) for payload in ("one", "r")]
        rng = random.Random(f"gate-{n}-v3.1")
        for auth_input in (False, True):
            for dpf_one in (False, True):
                for n_outmask in (0, 2):
                    t = time.perf_counter()
                    C = _check_gate(n, k, auth_input, dpf_one, thresholds, n_outmask, rng)
                    print(f"    gate n={n} k={k} auth={int(auth_input)} dpf={int(dpf_one)} om={n_outmask}: "
                          f"{C.stats()['n_and']:>9,} ANDs, {time.perf_counter() - t:5.1f} s", flush=True)


def test_gate_small_variants():
    """Other k, single thresholds, no thresholds, and the catalogue presets on a small domain."""
    rng = random.Random("gate-variants-v3.1")
    n = 3
    N = 1 << n
    for k in (32, 64):
        _check_gate(n, k, False, True, [], 0, rng)                       # table / zero test
        _check_gate(n, k, True, False, [], 1, rng)                       # [[r']] and one mask only
        _check_gate(n, k, False, False, [(5, "one"), (0, "one")], 0, rng)  # comparison
        _check_gate(n, k, True, False, [(N // 2, "one"), (0, "one"), (N // 2, "r"), (0, "r")], 1, rng)  # ReLU
    for name in cg.GATE_PRESETS:
        d = cg.gate_preset(name, n, 32, S)
        _check_gate(n, 32, d["auth_input"], d["dpf_one"], [tuple(x) for x in d["thresholds"]], d["n_outmask"], rng)
    # c = 0 and c = N need no adder (the DCF index is r itself); a generic c needs one
    base = cg.build_gate(4, 32, S, False, False, [(0, "one")], 0).stats()["n_and"]
    top = cg.build_gate(4, 32, S, False, False, [(16, "one")], 0).stats()["n_and"]
    mid = cg.build_gate(4, 32, S, False, False, [(5, "one")], 0).stats()["n_and"]
    check(abs(top - base) <= 2, f"c = N costs the same as c = 0 up to constant folding ({top} vs {base})")
    check(mid > max(base, top), f"a generic c adds the n-bit adder and the w_c selections ({mid} vs {base})")


def test_gate_rejects_bad_descriptors():
    check_raises(ValueError, cg.build_gate, 3, 32, S, False, False, [(9, "one")], 0)
    check_raises(ValueError, cg.build_gate, 3, 32, S, False, False, [(-1, "one")], 0)
    check_raises(ValueError, cg.build_gate, 3, 32, S, False, False, [(1, "beta")], 0)
    check_raises(ValueError, cg.build_gate, 3, 32, S, False, False, [], -1)
    check_raises(ValueError, cg.build_gate, 40, 32, S, False, True, [], 0)        # n > k
    check_raises(ValueError, cg.build_dpf, 0, 32, S)
    check_raises(ValueError, cg.build_dpf, 3, 128, S)                             # 2(k+s) > 320
    check_raises(ValueError, cg.build_dpf, 3, 32, S, "mul", "secret")
    check_raises(ValueError, cg.build_dcf, 3, 32, S, "xor", "hidden")
    check_raises(ValueError, cg.build_dpf, 3, 32, S, adder="kogge")
    check_raises(ValueError, cg.build_batch, [])
    check_raises(ValueError, cg.build_batch, [cg.session_setup_descriptor(S)])
    check_raises(ValueError, cg.build_batch, [cg.dpf_descriptor(2, 32, 64), cg.dpf_descriptor(2, 32, 40)])
    check_raises(ValueError, cg.build, {"kind": "unknown"})
    check_eq(cg.build_dpf(2, 32, S, "xor", "public").digest(), cg.build_dpf(2, 32, S, "xor", "public_one").digest(),
             "beta_mode alias")


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #

def test_batch_circuits():
    items = [cg.dpf_descriptor(3, 32, S, "xor", "public_one"),
             cg.dcf_descriptor(2, 64, S, "add", "secret"),
             cg.gate_descriptor(3, 64, S, True, True, [(3, "r"), (0, "r"), (8, "one")], 1),
             cg.build_dpf(4, 64, S, "add", "secret"),            # a built circuit is accepted too
             cg.dpf_descriptor(3, 32, S, "xor", "public_one")]   # the same descriptor twice
    C = cg.build_batch(items)
    C.validate()
    singles = [it if isinstance(it, Circuit) else cg.build(it) for it in items]
    bound = [(p, l, len(w)) for p, l, w in C.inputs if p == BOUND]
    check_eq(bound, [(BOUND, "delta", S + 1)], "one shared BOUND delta")
    check_eq(C.stats()["n_and"], sum(x.stats()["n_and"] for x in singles), "ANDs add up")
    check_eq(C.stats()["and_depth"], max(x.stats()["and_depth"] for x in singles), "depth is the maximum")
    for j, single in enumerate(singles):
        pre = f"i{j}."
        check_eq([(p, l, len(w)) for p, l, w in C.inputs if l.startswith(pre)],
                 [(p, pre + l, len(w)) for p, l, w in single.inputs if p != BOUND], f"item {j}: inputs prefixed")
        check_eq([(t, l, len(w)) for t, l, w in C.outputs if l.startswith(pre)],
                 [(t, pre + l, len(w)) for t, l, w in single.outputs], f"item {j}: outputs prefixed")
    check(all(l == "delta" or l[0] == "i" for _, l, _ in C.inputs), "every other label is prefixed")
    rng = random.Random("batch-v3.1")
    cases = [cg.random_inputs(C, rng, delta=d) for d in _deltas(rng, S, 4)]
    for inp, o in zip(cases, evaluate_batch(C, cases)):
        got = outputs_as_ints(o)
        check_eq(cg.expected_outputs_batch(items, inp), got, "expected_outputs_batch")
        check_eq(cg.expected_outputs(C.meta["descriptor"], inp), got, "expected_outputs dispatch")
        for j, single in enumerate(singles):
            pre = f"i{j}."
            sub = {(p, l): inp[(p, l if p == BOUND else pre + l)] for p, l, _ in single.inputs}
            exp = outputs_as_ints(evaluate(single, sub))
            check_eq({l: got[pre + l] for l in exp}, exp, f"item {j}: same outputs as the stand-alone circuit")
        # the two identical items received independent coins, hence different keys
        check(got["i0.cws"] != got["i4.cws"] and got["i0.root0"] != got["i4.root0"], "independent items")
        # keys of the batch are correct with the shared Delta
        delta = inp[(BOUND, "delta")]
        k0, k1 = cg.unpack_key_outputs(o, "dpf", 3, 32, S, "i0.")
        _check_key_on_domain("dpf", k0, k1, inp[(0, "i0.alpha")] ^ inp[(1, "i0.alpha")], 1, delta)
        k0, k1 = cg.unpack_key_outputs(o, "dcf", 2, 64, S, "i1.")
        _check_key_on_domain("dcf", k0, k1, (inp[(0, "i1.alpha")] + inp[(1, "i1.alpha")]) % 4,
                             (inp[(0, "i1.beta")] + inp[(1, "i1.beta")]) % (1 << 64), delta)
    # a batch of identical gates (the typical use), names and digests
    same = cg.build_batch([cg.gate_preset("table", 3, 32, S)] * 3)
    check_eq(same.meta["name"], "batch3x_gate_n3_k32_s64_shared_dpf", "batch name")
    check_eq(cg.build(same.meta["descriptor"]).digest(), same.digest(), "batch rebuild")
    check(C.meta["name"].startswith("batch5_"), "mixed batch name")
    inp = cg.random_inputs(same, rng)
    check_eq(cg.expected_outputs(same.meta["descriptor"], inp), outputs_as_ints(evaluate(same, inp)),
             "batch of gates")


def test_descriptor_names_and_presets():
    names = set()
    descs = [cg.session_setup_descriptor(S)]
    for n in (8, 12):
        for bm in ("secret", "public_one"):
            descs += [cg.dpf_descriptor(n, 64, S, "xor", bm), cg.dcf_descriptor(n, 32, S, "add", bm)]
        descs += [cg.gate_preset(p, n) for p in cg.GATE_PRESETS]
        descs += [cg.gate_preset("table", n, adder="sklansky")]
    for d in descs:
        name = cg.descriptor_name(d)
        check(name not in names and all(ch.isalnum() or ch == "_" for ch in name), f"name {name}")
        names.add(name)
        check_eq(cg.canonical_descriptor(d), d, "descriptors are canonical")
        check_eq(cg.descriptor_from_name(name), d, f"descriptor_from_name({name})")
    batch = cg.batch_descriptor([cg.gate_preset("comparison", 8)] * 16, adder="sklansky")
    check_eq(cg.descriptor_from_name(cg.descriptor_name(batch)), batch, "batch name round trip")
    check_raises(ValueError, cg.descriptor_from_name, "dpf_n8")
    check_eq(cg.descriptor_name(cg.dpf_descriptor(8, 64, 64, "xor", "public_one")), "dpf_n8_k64_s64_xor_one", "name")
    check_eq(cg.descriptor_name(cg.gate_preset("relu_large", 8)), "gate_n8_k64_s64_auth_t128o_t0o_t128r_t0r_om1",
             "ReLU preset")


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
    print(f"Couplet circuit tests ({'quick' if QUICK else 'full'} mode), {len(tests)} test functions")
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
        print(f"  ok   {name:<52} {CHECKS - before:>8} checks {time.perf_counter() - t:7.1f} s", flush=True)
    elapsed = time.perf_counter() - start
    if failed:
        print(f"FAILED: {len(failed)} of {len(tests)} tests ({', '.join(failed)}); {CHECKS} checks run, {elapsed:.1f} s")
        return 1
    print(f"TOTAL CHECKS: {CHECKS}  ({len(tests)} tests, all passed, {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
