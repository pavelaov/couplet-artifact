"""Tests of couplet.circuits_ppml: the generation circuits of the Couplet layer
(DESIGN_M3.md section 2), their plaintext reference, Pi_Gen for them and the manifest.

  * sites: for n in {3, 4, 5, 6}, f in {0, 2, 4}, both modes and k in {32, 64}
    (s = 64) the circuit is evaluated in plaintext on random inputs with the
    UNREDUCED Delta = Delta_0 + Delta_1 (including Delta >= 2^s) and every
    output relation is checked: [[rho']] over Z_{2^m} with its tag, plane A at
    rho_h (or at r' mod 2^n) on the whole domain with its tag plane, and in
    mode exact plane B at rho_l, [[rho_h]] and [[r']] over Z_{2^(n+s)}; the
    reference ``expected_outputs`` agrees; f = 0 builds, in both modes, the
    core's authenticated-offset table gate (same digest);
  * Beaver triples, matrix-vector triples and input masks: the relations
    c = a b, c = A b, the tags, the owner's masks; batches equal the single
    circuits; descriptors, names and their validation;
  * output masks (correction C2): the ``outmask`` item is the core's
    output-mask circuit (same digest and name), its material is
    ``[[2^k rho]]`` with both components multiples of ``2^k``;
  * the manifest: canonical encoding round trip, digest sensitivity, the
    comparison on a channel (equal plans pass, different plans abort);
  * Pi_Gen on a session with ideal preprocessing (``generate``,
    ``generate_batch``): the two parties' material reconstructs the secrets;
    the trusted-dealer reference ``deal_material`` satisfies the same
    relations; the statistics table runs on a tiny configuration.

Run from artifact/python:  python tests/test_ppml_circuits.py  (``-k name``
selects tests).  Prints the number of checks; exits non-zero on any failure.
"""

from __future__ import annotations

import json
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
from couplet import circuits_ppml as cp  # noqa: E402
from couplet import fss  # noqa: E402
from couplet.circuit import BOUND, evaluate, outputs_as_ints  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.prep import IdealDealer  # noqa: E402
from couplet.session import Session  # noqa: E402

S = 64
CHECKS = 0
RNG = random.Random(0xC0A7E7)


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
    except Exception:  # noqa: BLE001
        return False
    return False


def deltas(count: int) -> list:
    """Unreduced MAC keys, the first one at least 2^s."""
    out = [((1 << S) - 1) + ((1 << S) - 1)]
    while len(out) < count:
        out.append(RNG.getrandbits(S) + RNG.getrandbits(S))
    return out[:count]


def key_plane_ok(k0, k1, index: int, delta: int) -> bool:
    """Both planes of a payload-1 DPF pair on the whole domain."""
    n, mod = k0.n, 1 << k0.width
    y0, t0 = fss.dpf_eval_all(0, k0)
    y1, t1 = fss.dpf_eval_all(1, k1)
    for u in range(1 << n):
        want = int(u == index)
        if (int(y0[u]) + int(y1[u])) % mod != want or (int(t0[u]) + int(t1[u])) % mod != delta * want % mod:
            return False
    return True


# --------------------------------------------------------------------------- #
# Sites
# --------------------------------------------------------------------------- #

def check_site(n: int, f: int, k: int, mode: str):
    desc = cp.site_descriptor(n, f, k, S, mode)
    C = cp.build(desc)
    m = f + n + S
    for delta in deltas(2):
        inp = cg.random_inputs(C, RNG, delta=delta)
        got = outputs_as_ints(evaluate(C, inp))
        check_eq(got, cp.expected_outputs(desc, inp), f"site n={n} f={f} k={k} {mode}: reference")
        pt = cp.site_plaintext(desc, inp)
        rp = pt["rp"]
        check_eq((inp[(0, "rp.sigma")] + got["rp.share1"]) % (1 << m), rp, "[[rho']] value")
        check_eq((inp[(0, "rp.mu_sigma")] + got["rp.mu_share1"]) % (1 << m), delta * rp % (1 << m), "[[rho']] tag")
        ka0, ka1 = cg.unpack_key_outputs(got, "dpf", n, k, S, "dpf.")
        check(key_plane_ok(ka0, ka1, pt["index"], delta), "plane A on the whole domain")
        two = f >= 1 and mode == "exact"
        check_eq("pb.cws" in got, two, "plane B present iff mode exact and f >= 1")
        if two:
            check_eq(pt["index"], pt["r2"] % (1 << n), "plane A at r' mod 2^n")
            kb0, kb1 = cg.unpack_key_outputs(got, "dpf", f, k, S, "pb.")
            check(key_plane_ok(kb0, kb1, pt["rho_l"], delta), "plane B at rho_l")
            for q, secret in (("rh.", pt["rho_h"]), ("r2.", pt["r2"])):
                check_eq((inp[(0, q + "sigma")] + got[q + "share1"]) % (1 << (n + S)), secret, f"{q} value")
                check_eq((inp[(0, q + "mu_sigma")] + got[q + "mu_share1"]) % (1 << (n + S)),
                         delta * secret % (1 << (n + S)), f"{q} tag")
        else:
            check_eq(pt["index"], pt["rho_h"], "plane A at rho_h")
    return C


def test_sites():
    for n in (3, 4, 5, 6):
        for f in (0, 2, 4):
            for mode in cp.MODES:
                for k in (32, 64):
                    t = time.perf_counter()
                    C = check_site(n, f, k, mode)
                    print(f"    site n={n} f={f} k={k} {mode:<5}: {C.stats()['n_and']:>8,} ANDs, "
                          f"{time.perf_counter() - t:4.1f} s", flush=True)


def test_f0_is_core_gate():
    for n, k in ((3, 32), (4, 64), (6, 32)):
        core = cg.build_gate(n, k, S, True, True, [], 0)
        for mode in cp.MODES:
            site = cp.build_site(n, 0, k, S, mode)
            check_eq(site.digest(), core.digest(), f"site(n={n}, f=0, {mode}) is the core gate")
            check_eq(site.stats()["n_and"], core.stats()["n_and"], "same AND count")
        check_eq(cp.and_count(cp.site_descriptor(n, 0, k, S, "round")), core.stats()["n_and"], "and_count")


def check_outmask_material(o0, o1, delta: int) -> None:
    k, w = o0.descriptor["k"], o0.width
    mod = 1 << w
    for x in o0.share + o1.share:
        check_eq(x % (1 << k), 0, "output mask shares are multiples of 2^k")
    v = (o0.share[0] + o1.share[0]) % mod
    check_eq((o0.share[1] + o1.share[1]) % mod, delta * v % mod, "output mask tag")


def test_outmask_is_core_circuit():
    from couplet import protocol as pg
    for k in (32, 64):
        d = cp.outmask_descriptor(k, S)
        C = cp.build(d)
        core = cg.build(pg.outmask_descriptor(k, S))
        check_eq(C.digest(), core.digest(), f"outmask k={k} is the core's output-mask circuit")
        check_eq(cp.descriptor_name(d), cg.descriptor_name(pg.outmask_descriptor(k, S)), "and has its name")
        check_eq(cp.and_count(d), pg.outmask_and_count(k, S), "AND gates")
        check_eq(cp.descriptor_from_name(cp.descriptor_name(d)), d, "the name parses back")
        for delta in deltas(2):
            inp = cg.random_inputs(C, RNG, delta=delta)
            got = outputs_as_ints(evaluate(C, inp))
            check_eq(got, cp.expected_outputs(d, inp), "outmask reference")
            rho = inp[(0, "om0.coin")] ^ inp[(1, "om0.coin")]
            check_eq((inp[(0, "om0.sigma")] + got["om0.share1"]) % (1 << S), rho, "high part of the mask value")
            check_eq((inp[(0, "om0.mu_sigma")] + got["om0.mu_share1"]) % (1 << S), delta * rho % (1 << S),
                     "high part of the mask tag")
    check_eq(pg.outmask_and_count(64, S), 4159, "4,159 AND gates per output mask at k = s = 64")


# --------------------------------------------------------------------------- #
# Triples, matrix triples, input masks, batches
# --------------------------------------------------------------------------- #

def unpack(value: int, count: int, w: int) -> list:
    mask = (1 << w) - 1
    return [(value >> (j * w)) & mask for j in range(count)]


def rec_vec(inp, got, q: str, count: int, w: int, delta: int) -> list:
    """Reconstructed (value, tag ok) per entry of a share_out vector."""
    mod = 1 << w
    sig, mu = unpack(inp[(0, q + "sigma")], count, w), unpack(inp[(0, q + "mu_sigma")], count, w)
    sh, msh = unpack(got[q + "share1"], count, w), unpack(got[q + "mu_share1"], count, w)
    out = []
    for j in range(count):
        v = (sig[j] + sh[j]) % mod
        out.append((v, (mu[j] + msh[j]) % mod == delta * v % mod))
    return out


def test_triples():
    for k in (32, 64):
        w = k + S
        counts = []
        for count in (1, 2, 3):
            desc = cp.triples_descriptor(count, k, S)
            C = cp.build(desc)
            counts.append(C.stats()["n_and"])
            for delta in deltas(2):
                inp = cg.random_inputs(C, RNG, delta=delta)
                got = outputs_as_ints(evaluate(C, inp))
                check_eq(got, cp.expected_outputs(desc, inp), "triples reference")
                a = rec_vec(inp, got, "a.", count, w, delta)
                b = rec_vec(inp, got, "b.", count, w, delta)
                c = rec_vec(inp, got, "c.", count, w, delta)
                for j in range(count):
                    check(a[j][1] and b[j][1] and c[j][1], "triple tags")
                    check_eq(c[j][0], a[j][0] * b[j][0] % (1 << w), "c = a b")
                    coins_a = unpack(inp[(0, "a.coin")] ^ inp[(1, "a.coin")], count, w)
                    check_eq(a[j][0], coins_a[j], "a is the XOR of the coins")
        check_eq(counts[2] - counts[1], counts[1] - counts[0], f"k={k}: AND gates are affine in the count")
        print(f"    triples k={k}: {counts[0]:,} ANDs for one, {counts[1] - counts[0]:,} per further triple", flush=True)


def test_mattriple():
    for rows, cols, k in ((1, 1, 32), (2, 3, 64), (3, 2, 32)):
        w = k + S
        desc = cp.mattriple_descriptor(rows, cols, k, S)
        C = cp.build(desc)
        for delta in deltas(2):
            inp = cg.random_inputs(C, RNG, delta=delta)
            got = outputs_as_ints(evaluate(C, inp))
            check_eq(got, cp.expected_outputs(desc, inp), "mattriple reference")
            a = rec_vec(inp, got, "a.", rows * cols, w, delta)
            b = rec_vec(inp, got, "b.", cols, w, delta)
            c = rec_vec(inp, got, "c.", rows, w, delta)
            check(all(t for _, t in a + b + c), "matrix triple tags")
            for i in range(rows):
                check_eq(c[i][0], sum(a[i * cols + j][0] * b[j][0] for j in range(cols)) % (1 << w), "c = A b")
        print(f"    mattriple {rows}x{cols} k={k}: {C.stats()['n_and']:,} ANDs", flush=True)


def test_inmask():
    for owner in (0, 1):
        for count, k in ((1, 64), (3, 32)):
            w = k + S
            desc = cp.inmask_descriptor(owner, count, k, S)
            C = cp.build(desc)
            for delta in deltas(2):
                inp = cg.random_inputs(C, RNG, delta=delta)
                got = outputs_as_ints(evaluate(C, inp))
                check_eq(got, cp.expected_outputs(desc, inp), "inmask reference")
                rho = unpack(inp[(owner, "rho")], count, w)
                rec = rec_vec(inp, got, "m.", count, w, delta)
                check_eq([v for v, _ in rec], rho, "the masks are the owner's input")
                check(all(t for _, t in rec), "mask tags")
            check_eq(C.input_bits(owner), count * w + (2 * count * w if owner == 0 else 0), "input bits")


def test_batch_equals_singles():
    items = [cp.site_descriptor(3, 2, 32, S, "exact"), cp.triples_descriptor(1, 32, S),
             cp.inmask_descriptor(1, 2, 32, S), cp.mattriple_descriptor(2, 2, 32, S)]
    desc = cp.batch_descriptor(items)
    C = cp.build(desc)
    delta = deltas(1)[0]
    inp = cg.random_inputs(C, RNG, delta=delta)
    got = outputs_as_ints(evaluate(C, inp))
    check_eq(got, cp.expected_outputs(desc, inp), "batch reference")
    total = 0
    for j, item in enumerate(items):
        single = cp.build(item)
        sub = {(BOUND, cg.DELTA): delta}
        for owner, label, _ in single.inputs:
            if owner != BOUND:
                sub[(owner, label)] = inp[(owner, f"i{j}.{label}")]
        exp = outputs_as_ints(evaluate(single, sub))
        check_eq({lab[len(f"i{j}."):]: v for lab, v in got.items() if lab.startswith(f"i{j}.")}, exp,
                 f"item {j} equals the single circuit")
        total += single.stats()["n_and"]
    check_eq(C.stats()["n_and"], total, "the batch has the AND gates of its items")
    check_eq(cp.descriptor_name(desc)[:7], "pbatch4", "batch name")


def test_descriptors():
    d = cp.site_descriptor(16, 8, 64, 64, "round")
    check_eq(cp.canonical_descriptor(d), d, "canonical is idempotent")
    check_eq(cp.descriptor_name(d), "site_n16_f8_k64_s64_round", "site name")
    check_eq(cp.descriptor_name(cp.triples_descriptor(3, 64, 64)), "triples3_k64_s64", "triples name")
    check_eq(cp.descriptor_name(cp.mattriple_descriptor(8, 16, 64, 64)), "mattriple_8x16_k64_s64", "mattriple name")
    check_eq(cp.descriptor_name(cp.inmask_descriptor(1, 5, 64, 64)), "inmask_p1_c5_k64_s64", "inmask name")
    bad = [lambda: cp.site_descriptor(0, 8, 64, 64, "round"), lambda: cp.site_descriptor(21, 8, 64, 64, "round"),
           lambda: cp.site_descriptor(16, 8, 64, 64, "nearest"), lambda: cp.site_descriptor(40, 8, 32, 64, "round"),
           lambda: cp.site_descriptor(16, -1, 64, 64, "exact"), lambda: cp.triples_descriptor(0, 64, 64),
           lambda: cp.site_descriptor(20, 16, 32, 64, "round"),
           lambda: cp.mattriple_descriptor(2, 0, 64, 64), lambda: cp.inmask_descriptor(2, 1, 64, 64),
           lambda: cp.inmask_descriptor(0, 1, 64, 200), lambda: cp.batch_descriptor([]),
           lambda: cp.batch_descriptor([cp.triples_descriptor(1, 32, 64), cp.triples_descriptor(1, 64, 64)]),
           lambda: cp.canonical_descriptor({"kind": "gate", "n": 4}), lambda: cp.canonical_descriptor({"kind": "site"}),
           lambda: cp.site_descriptor(4, 2, 64, 64, "round", adder="carry-save")]
    for fn in bad:
        check(raises(ValueError, fn), "bad descriptors are refused")
    C = cp.build_site(3, 1, 32, S, "exact")
    check_eq(cp.canonical_descriptor(C), C.meta["descriptor"], "a circuit stands for its descriptor")
    check_eq(cp.build(C.meta["descriptor"]).digest(), C.digest(), "rebuilding gives the same digest")
    # names parse back (the circuit exporter of python/bench accepts them)
    for desc in (d, cp.site_descriptor(5, 0, 32, 64, "exact", "sklansky"), cp.triples_descriptor(3, 64, 64),
                 cp.mattriple_descriptor(8, 16, 64, 64), cp.inmask_descriptor(1, 5, 32, 64)):
        check_eq(cp.descriptor_from_name(cp.descriptor_name(desc)), desc, "descriptor_from_name inverts the name")
        check(cp.is_ppml_descriptor(desc), "a Couplet descriptor")
    check(not cp.is_ppml_descriptor(cg.gate_preset("table", 4, 32, S)), "a core descriptor is not one")
    for name in ("site_n4_f2_k64_s64_nearest", "gate_n4_k64_s64_auth_dpf", "mattriple_2x_k64_s64",
                 "pbatch2_0123456789ab"):
        check(raises(ValueError, lambda: cp.descriptor_from_name(name)), f"{name} is refused")
    sys.path.insert(0, os.path.normpath(os.path.join(_PY, "bench")))
    import export_circuits  # noqa: E402  (python/bench/export_circuits.py)
    from couplet.circuit import Circuit  # noqa: E402
    with tempfile.TemporaryDirectory(prefix="couplet_ppml_export_") as tmp:
        for name in ("site_n3_f1_k32_s64_exact", "inmask_p0_c2_k32_s64"):
            desc = export_circuits.parse_name(name)
            entry = export_circuits.export_one(desc, tmp)
            with open(os.path.join(tmp, entry["binary_file"]), "rb") as fh:
                back = Circuit.from_bytes(fh.read())
            check_eq((entry["name"], back.digest()), (name, cp.build(desc).digest()), f"{name}: exported and read back")
        core = export_circuits.parse_name("json:" + json.dumps(cg.gate_preset("table", 3, 32, S)))
        check_eq(core["kind"], "gate", "core descriptors still parse")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def make_manifest(mode: str = "round", k: int = 64) -> cp.Manifest:
    man = cp.Manifest(k, S, mode, False, secrets.token_bytes(32), [("tanh", 8, 4, secrets.token_bytes(32))], [])
    man.add("sites of layer 1", cp.site_descriptor(8, 4, k, S, mode), 8)
    man.add("sites of layer 2", cp.site_descriptor(8, 4, k, S, mode), 4)
    man.add("inputs of P_0", cp.inmask_descriptor(0, 3, k, S))
    return man


def test_manifest():
    man = make_manifest()
    enc = man.encode()
    dec = cp.Manifest.decode(enc)
    check_eq(dec.encode(), enc, "encode(decode) round trip")
    check_eq((dec.k, dec.s, dec.mode, dec.private_model, dec.model_hash, dec.tables, len(dec.entries)),
             (man.k, man.s, man.mode, man.private_model, man.model_hash, man.tables, 3), "decoded fields")
    check_eq(dec.digest(), man.digest(), "same digest")
    check_eq(len(man.digest()), 32, "32-byte digest")
    check_eq(man.site_count(), 12, "site count")
    check_eq(man.and_budget(), 12 * cp.and_count(cp.site_descriptor(8, 4, 64, S, "round"))
             + cp.and_count(cp.inmask_descriptor(0, 3, 64, S)), "AND budget")
    variants = [make_manifest("exact"), make_manifest("round", 32)]
    other = make_manifest()
    other.entries[0]["batch"] = 9
    variants.append(other)
    other = make_manifest()
    other.private_model = True
    variants.append(other)
    other = make_manifest()
    other.tables = [("relu", 8, 4, other.tables[0][3])]
    variants.append(other)
    other = make_manifest()
    other.entries[2]["use"] = "inputs of P_1"
    variants.append(other)
    digests = {man.digest()} | {v.digest() for v in variants}
    check_eq(len(digests), 1 + len(variants), "every field changes the digest")
    check(raises(ValueError, lambda: cp.Manifest.decode(enc[:-1])), "truncated encoding")
    check(raises(ValueError, lambda: cp.Manifest.decode(enc + b"\x00")), "trailing bytes")
    summary = man.summary()
    check_eq((summary["executions"], summary["sites"], summary["entries"][0]["batch"]), (3, 12, 8), "summary")
    # comparison on a channel: equal plans pass, different plans abort at both parties
    same = make_manifest()
    same.model_hash, same.tables = man.model_hash, man.tables
    r0, r1, _, _ = run_two_party(lambda ch: cp.compare_manifest(ch, man), lambda ch: cp.compare_manifest(ch, same))
    check_eq((r0, r1), (man.digest(), man.digest()), "equal plans pass")
    diff = make_manifest("exact")

    def party(ch, m):
        try:
            cp.compare_manifest(ch, m)
            return "ok"
        except ProtocolAbort:
            return "abort"

    r0, r1, st0, _ = run_two_party(lambda ch: party(ch, man), lambda ch: party(ch, diff))
    check_eq((r0, r1), ("abort", "abort"), "different plans abort at both parties")
    check_eq((st0["bytes_sent"], st0["rounds"]), (32, 1), "the comparison is one 32-byte exchange")


# --------------------------------------------------------------------------- #
# Pi_Gen and the dealer
# --------------------------------------------------------------------------- #

def generate_both(k: int, calls, prep: str = "ideal"):
    """Run ``calls(session)`` at both parties on one session; returns (out0, out1, delta)."""
    dealer = IdealDealer() if prep == "ideal" else None
    sid = b"ppml-gen-" + secrets.token_bytes(4)
    deltas_ = [None, None]

    def party(role):
        def fn(chan):
            ses = Session(role, chan, sid, k=k, s=S, prep=prep, dealer=dealer, allow_refill=True)
            ses.setup()
            out = calls(ses)
            deltas_[role] = ses.delta_share
            return out
        return fn

    o0, o1, _, _ = run_two_party(party(0), party(1), timeout=600)
    return o0, o1, deltas_[0] + deltas_[1]


def check_site_material(m0: cp.SiteMaterial, m1: cp.SiteMaterial, delta: int) -> None:
    n, f, s, m = m0.n, m0.f, m0.s, m0.m
    rp = (m0.rp[0] + m1.rp[0]) % (1 << m)
    check_eq((m0.rp[1] + m1.rp[1]) % (1 << m), delta * rp % (1 << m), "[[rho']] tag")
    rho_h, rho_l = (rp >> f) % (1 << n), rp % (1 << f)
    if m0.two_openings:
        r2 = (m0.r2[0] + m1.r2[0]) % (1 << (n + s))
        check_eq((m0.r2[1] + m1.r2[1]) % (1 << (n + s)), delta * r2 % (1 << (n + s)), "[[r']] tag")
        check_eq((m0.rh[0] + m1.rh[0]) % (1 << (n + s)), rho_h, "[[rho_h]] value")
        check_eq((m0.rh[1] + m1.rh[1]) % (1 << (n + s)), delta * rho_h % (1 << (n + s)), "[[rho_h]] tag")
        check(key_plane_ok(m0.plane_a, m1.plane_a, r2 % (1 << n), delta), "plane A at r'")
        check(key_plane_ok(m0.plane_b, m1.plane_b, rho_l, delta), "plane B at rho_l")
    else:
        check(key_plane_ok(m0.plane_a, m1.plane_a, rho_h, delta), "plane A at rho_h")


def check_triples_material(t0, t1, delta: int) -> None:
    w, mod = t0.width, 1 << t0.width
    for j in range(t0.count):
        a = (t0.a[j][0] + t1.a[j][0]) % mod
        b = (t0.b[j][0] + t1.b[j][0]) % mod
        c = (t0.c[j][0] + t1.c[j][0]) % mod
        check_eq(c, a * b % mod, "c = a b")
        for (x, y), v in (((t0.a[j], t1.a[j]), a), ((t0.b[j], t1.b[j]), b), ((t0.c[j], t1.c[j]), c)):
            check_eq((x[1] + y[1]) % mod, delta * v % mod, "triple tag")


def check_mattriple_material(t0, t1, delta: int) -> None:
    rows, cols, mod = t0.rows, t0.cols, 1 << t0.width
    a = [(x[0] + y[0]) % mod for x, y in zip(t0.a, t1.a)]
    b = [(x[0] + y[0]) % mod for x, y in zip(t0.b, t1.b)]
    c = [(x[0] + y[0]) % mod for x, y in zip(t0.c, t1.c)]
    for i in range(rows):
        check_eq(c[i], sum(a[i * cols + j] * b[j] for j in range(cols)) % mod, "c = A b")
    for pairs, vals in (((t0.a, t1.a), a), ((t0.b, t1.b), b), ((t0.c, t1.c), c)):
        for (x, y), v in zip(zip(*pairs), vals):
            check_eq((x[1] + y[1]) % mod, delta * v % mod, "matrix triple tag")


def check_inmask_material(i0, i1, delta: int) -> None:
    mod = 1 << i0.width
    owner = (i0, i1)[i0.owner]
    for j in range(i0.count):
        rho = (i0.shares[j][0] + i1.shares[j][0]) % mod
        check_eq(owner.values[j], rho, "the owner knows the mask")
        check_eq((i0.shares[j][1] + i1.shares[j][1]) % mod, delta * rho % mod, "mask tag")
    check((i1 if i0.owner == 0 else i0).values is None, "the other party does not know the masks")


def test_generation_protocol():
    def calls(ses):
        out = {"sites": cp.generate(ses, cp.site_descriptor(3, 2, 32, S, "exact"), 2),
               "sites_r": cp.generate(ses, cp.site_descriptor(4, 1, 32, S, "round"), 1),
               "triples": cp.generate(ses, cp.triples_descriptor(2, 32, S), 1)[0],
               "mat": cp.generate(ses, cp.mattriple_descriptor(2, 2, 32, S), 1)[0],
               "im0": cp.generate(ses, cp.inmask_descriptor(0, 2, 32, S), 1)[0],
               "im1": cp.generate(ses, cp.inmask_descriptor(1, 1, 32, S), 1)[0],
               "om": cp.generate(ses, cp.outmask_descriptor(32, S), 3),
               "batch": cp.generate_batch(ses, [cp.site_descriptor(3, 0, 32, S, "exact"),
                                                cp.triples_descriptor(1, 32, S), cp.inmask_descriptor(1, 1, 32, S)])}
        out["last"] = dict(ses.last_gen)
        return out

    o0, o1, delta = generate_both(32, calls)
    for m0, m1 in zip(o0["sites"] + o0["sites_r"], o1["sites"] + o1["sites_r"]):
        check_site_material(m0, m1, delta)
    check_eq((o0["sites"][0].two_openings, o0["sites_r"][0].two_openings), (True, False), "plane B iff exact")
    check_triples_material(o0["triples"], o1["triples"], delta)
    check_mattriple_material(o0["mat"], o1["mat"], delta)
    check_inmask_material(o0["im0"], o1["im0"], delta)
    check_inmask_material(o0["im1"], o1["im1"], delta)
    check_eq(len(o0["om"]), 3, "three output masks from one execution")
    for m0, m1 in zip(o0["om"], o1["om"]):
        check_outmask_material(m0, m1, delta)
    b0, b1 = o0["batch"], o1["batch"]
    check_site_material(b0[0], b1[0], delta)
    check(not b0[0].two_openings, "f = 0: a plain offset gate")
    check_triples_material(b0[1], b1[1], delta)
    check_inmask_material(b0[2], b1[2], delta)
    check_eq(o0["last"]["circuit"][:7], "pbatch3", "last_gen names the batch circuit")
    check_eq(o0["last"]["digest"], o1["last"]["digest"], "both parties ran the same circuit")
    check(delta >= 0, "delta")


def test_dealer_reference():
    for delta in deltas(2):
        m0, m1 = cp.deal_material(cp.site_descriptor(4, 2, 32, S, "exact"), delta, RNG)
        check_site_material(m0, m1, delta)
        m0, m1 = cp.deal_material(cp.site_descriptor(5, 2, 64, S, "round"), delta, RNG)
        check_site_material(m0, m1, delta)
        t0, t1 = cp.deal_material(cp.triples_descriptor(3, 32, S), delta, RNG)
        check_triples_material(t0, t1, delta)
        t0, t1 = cp.deal_material(cp.mattriple_descriptor(2, 3, 64, S), delta, RNG)
        check_mattriple_material(t0, t1, delta)
        i0, i1 = cp.deal_material(cp.inmask_descriptor(1, 2, 32, S), delta, RNG)
        check_inmask_material(i0, i1, delta)
        o0, o1 = cp.deal_material(cp.outmask_descriptor(32, S), delta, RNG)
        check_outmask_material(o0, o1, delta)
    check(raises(ValueError, lambda: cp.deal_material(cp.site_descriptor(4, 2, 32, S, "exact"), 1 << (S + 1), RNG)),
          "delta out of range")


def test_stats_small():
    doc = cp.stats_table(ns=(3,), fs=(0, 1), k=32, s=S, verbose=False, mat_shapes=((1, 1), (1, 2), (2, 1), (2, 2)),
                         triple_counts=(1, 2), inmask_counts=(1, 2))
    check_eq([r["name"] for r in doc["sites"]], ["site_n3_f0_k32_s64_round", "site_n3_f1_k32_s64_round",
                                                  "site_n3_f1_k32_s64_exact"], "site rows (f = 0 once)")
    check_eq(doc["comparison_bundles"][0]["n"], 3, "comparison bundle row")
    fit = doc["matrix_triple_fit"]
    check(fit["exact_for_all_shapes"], "the matrix-triple fit is exact on the four shapes")
    check(doc["beaver_triple_fit"]["per_triple"] > 0 and doc["input_mask_fit"]["per_mask"] > 0, "positive increments")
    with tempfile.TemporaryDirectory(prefix="couplet_ppml_stats_") as tmp:
        path = os.path.join(tmp, "stats.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        with open(path, encoding="utf-8") as fh:
            check_eq(json.load(fh)["k"], 32, "the table is JSON")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    selected = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "-k" in sys.argv:
        selected = [sys.argv[sys.argv.index("-k") + 1]]
    failed = []
    start = time.perf_counter()
    for name, fn in tests:
        if selected and not any(s in name for s in selected):
            continue
        before = CHECKS
        t = time.perf_counter()
        try:
            fn()
            print(f"  ok   {name:<34} {CHECKS - before:>8} checks {time.perf_counter() - t:7.1f} s", flush=True)
        except Exception:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}\n{traceback.format_exc()}", flush=True)
    elapsed = time.perf_counter() - start
    if failed:
        print(f"FAILED: {len(failed)} of {len(tests)} tests ({', '.join(failed)}); {CHECKS} checks run, {elapsed:.1f} s")
        return 1
    print(f"TOTAL CHECKS: {CHECKS}  ({len(tests)} tests, all passed, {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
