"""Tests of couplet.truncmask: truncate-and-mask in both modes (DESIGN_M3.md section 3)
and the openings of an inference session.

  * the identity for ALL p in [-2^(f+n+1), 2^(f+n+1)) (2^(f+n+2) values around
    zero) for (n, f) in {(3, 1), (4, 2), (5, 3)}, both modes, with the
    pre-activation given as additive shares without MAC (first layer,
    commit-then-open) and as authenticated sharings (wire mode, deferred MAC
    check, verified afterwards), on trusted-dealer material: both parties
    obtain the same x_hat; the masked value z = x_hat - alpha equals the
    plaintext rule; mode exact gives the exact floor (p >> f) mod 2^n; mode
    round gives floor + [p mod 2^f > v_l] with v_l = v mod 2^f; the opened
    v is (p + rho') mod 2^m;
  * random p for (16, 8) (including the whole 2f-bit range), both modes;
  * uniformity of the opened value: chi-square over the whole opening ring
    for a small s (m = 9 bits, 4096 sites) and over the low bits for s = 64;
  * the same on material from the real generation protocol (ideal
    preprocessing), both modes and both input kinds, with a table query of
    plane A at the result;
  * argument checks and the refusal to release a sharing that is not marked
    sign-exact (nothing is sent); the release is the signed k-bit decoding of
    the authenticated payload v mod 2^k (bits above k are ignored).

Run from artifact/python:  python tests/test_truncmask.py  (``-k name``
selects tests).  Prints the number of checks; exits non-zero on any failure.
"""

from __future__ import annotations

import math
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

from couplet import circuits_ppml as cp  # noqa: E402
from couplet import tables  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import loopback_pair, run_two_party  # noqa: E402
from couplet.prep import IdealDealer  # noqa: E402
from couplet.session import Session  # noqa: E402
from couplet.spdz2k import AShare, ArithChecker, deal  # noqa: E402
from couplet.truncmask import Opener, Site, plain_truncate, truncate_and_mask  # noqa: E402

CHECKS = 0
RNG = random.Random(0x7A0C)


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


def chi2_critical(df: int, z: float = 4.753) -> float:
    """Wilson-Hilferty critical value at about 10^-6 (as in the core tests)."""
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def chi2_uniform(samples, bins: int) -> float:
    counts = [0] * bins
    for v in samples:
        counts[v] += 1
    e = len(samples) / bins
    return sum((c - e) ** 2 / e for c in counts)


# --------------------------------------------------------------------------- #
# Running one layer at both parties
# --------------------------------------------------------------------------- #

def secrets_of(m0: cp.SiteMaterial, m1: cp.SiteMaterial) -> dict:
    n, f, s, m = m0.n, m0.f, m0.s, m0.m
    rp = (m0.rp[0] + m1.rp[0]) % (1 << m)
    sec = {"rp": rp, "rho_l": rp % (1 << f), "rho_h": (rp >> f) % (1 << n), "alpha": (rp >> f) % (1 << n)}
    if m0.two_openings:
        r2 = (m0.r2[0] + m1.r2[0]) % (1 << (n + s))
        sec["r2"] = r2
        sec["alpha"] = r2 % (1 << n)
    return sec


def run_layer(mats0, mats1, delta_shares, p_values, first_layer: bool, k: int, s: int, verify: bool = True):
    """Both parties run truncate_and_mask on all sites; returns (results of P_0, results of P_1, stats)."""
    sid = b"tm-" + secrets.token_bytes(4)
    w = k + s
    delta = delta_shares[0] + delta_shares[1]
    if first_layer:
        m = mats0[0].m
        shares = []
        for p in p_values:
            a = RNG.getrandbits(m)
            shares.append((a, (p - a) % (1 << m)))
        p_in = ([a for a, _ in shares], [b for _, b in shares])
    else:
        pairs = [deal(p % (1 << w), delta, w) for p in p_values]
        p_in = ([a for a, _ in pairs], [b for _, b in pairs])

    def party(role):
        def fn(chan):
            checker = ArithChecker(role, chan, sid, delta_shares[role], s)
            op = Opener(role, chan, sid, checker, delta_shares[role])
            sites = [Site(m) for m in (mats0 if role == 0 else mats1)]
            res = truncate_and_mask(op, sites, p_in[role], first_layer=first_layer)
            if verify:
                op.check()
            return res, checker.openings, checker.checks, checker.pending
        return fn

    (r0, o0, c0, pend0), (r1, o1, c1, pend1), st0, st1 = run_two_party(party(0), party(1), timeout=600)
    check_eq((pend0, pend1), (0, 0), "every deferred opening was verified")
    return r0, r1, {"openings": o0, "checks": c0, "rounds": st0["rounds"], "bytes": st0["bytes_sent"]}


def check_results(mats0, mats1, p_values, r0, r1, mode: str) -> None:
    n, f, m = mats0[0].n, mats0[0].f, mats0[0].m
    N = 1 << n
    for p, a, b, x, y in zip(p_values, mats0, mats1, r0, r1):
        sec = secrets_of(a, b)
        check_eq((x.x_hat, x.v, x.v_l), (y.x_hat, y.v, y.v_l), "both parties see the same public values")
        check_eq(x.v, (p + sec["rp"]) % (1 << m), "v = p + rho' mod 2^m")
        check_eq(x.v_l, x.v % (1 << f), "v_l = v mod 2^f")
        z = (x.x_hat - sec["alpha"]) % N
        check_eq(z, plain_truncate(p, f, n, mode, x.v_l), "the plaintext rule")
        floor = (p >> f) % N
        if mode == "exact" or f == 0:
            check_eq(z, floor, "mode exact: the exact floor")
            check(x.exact, "flagged exact")
        else:
            beta = int((p % (1 << f)) > x.v_l)
            check_eq(x.x_hat, (floor + beta + sec["rho_h"]) % N, "x_hat = x + beta + rho_h")
            check(not x.exact, "flagged as rounded")


def dealt_sites(n: int, f: int, k: int, s: int, mode: str, count: int):
    d0, d1 = RNG.getrandbits(s), RNG.getrandbits(s)
    desc = cp.site_descriptor(n, f, k, s, mode)
    pairs = [cp.deal_material(desc, d0 + d1, RNG) for _ in range(count)]
    return [a for a, _ in pairs], [b for _, b in pairs], (d0, d1)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_identity_all_p():
    for n, f, k in ((3, 1, 32), (4, 2, 32), (5, 3, 64)):
        lo = 1 << (f + n + 1)
        p_values = list(range(-lo, lo))
        for mode in cp.MODES:
            for first in (True, False):
                t = time.perf_counter()
                m0, m1, ds = dealt_sites(n, f, k, 64, mode, len(p_values))
                r0, r1, st = run_layer(m0, m1, ds, p_values, first, k, 64)
                check_results(m0, m1, p_values, r0, r1, mode)
                want_open = 0 if first else len(p_values)
                if mode == "exact":
                    want_open += len(p_values)
                check_eq(st["openings"], want_open, "deferred openings per site")
                check_eq(st["rounds"], (2 if first else 1) + (1 if mode == "exact" else 0) + (4 if want_open else 0),
                         "rounds: the openings plus the check")
                print(f"    n={n} f={f} k={k} {mode:<5} {'shared' if first else 'wire  '}: {len(p_values)} values, "
                      f"{time.perf_counter() - t:4.1f} s", flush=True)


def test_random_p_16_8():
    n, f, k = 16, 8, 64
    p_values = [RNG.randrange(-(1 << (2 * f + n - 1)), 1 << (2 * f + n - 1)) for _ in range(24)]
    p_values += [0, -1, 1 << f, -(1 << f), (1 << (f + n)) - 1, -(1 << (f + n)), (1 << 40) + 5, -(1 << 40) - 7]
    for mode in cp.MODES:
        for first in (True, False):
            m0, m1, ds = dealt_sites(n, f, k, 64, mode, len(p_values))
            r0, r1, st = run_layer(m0, m1, ds, p_values, first, k, 64)
            check_results(m0, m1, p_values, r0, r1, mode)
    # a table query of plane A at the result reproduces T[z] (one site per mode)
    T = tables.activation_table("tanh", n, f)
    for mode in cp.MODES:
        m0, m1, ds = dealt_sites(n, f, k, 64, mode, 1)
        p = p_values[0]
        r0, r1, _ = run_layer(m0, m1, ds, [p], True, k, 64)
        s0, s1 = Site(m0[0]), Site(m1[0])
        a, b = s0.plane_a.table(T, r0[0].x_hat), s1.plane_a.table(T, r1[0].x_hat)
        mod = 1 << (k + 64)
        check_eq((a.value + b.value) % mod, T[plain_truncate(p, f, n, mode, r0[0].v_l)] % mod,
                 f"{mode}: table of plane A at x_hat")


def test_threshold_uniform():
    # whole opening ring with a small s: m = f + n + s = 2 + 3 + 4 = 9 bits, 4096 sites, p fixed
    n, f, k, s = 3, 2, 8, 4
    count = 4096
    for mode in cp.MODES:
        m0, m1, ds = dealt_sites(n, f, k, s, mode, count)
        p_values = [RNG.randrange(-(1 << (2 * f + n - 1)), 1 << (2 * f + n - 1))] * count
        r0, r1, _ = run_layer(m0, m1, ds, p_values, True, k, s)
        stat = chi2_uniform([r.v for r in r0], 1 << 9)
        check(stat < chi2_critical((1 << 9) - 1), f"{mode}: v is uniform over Z_(2^m) (chi2 {stat:.1f})")
        stat = chi2_uniform([r.v_l for r in r0], 1 << f)
        check(stat < chi2_critical((1 << f) - 1), f"{mode}: v_l is uniform (chi2 {stat:.1f})")
        if mode == "round":
            ups = sum(int(((p >> f) % (1 << n)) != (r.x_hat - secrets_of(a, b)["rho_h"]) % (1 << n))
                      for p, r, a, b in zip(p_values, r0, m0, m1))
            frac = (p_values[0] % (1 << f)) / (1 << f)
            check(abs(ups / count - frac) < 0.05, f"rounding up with probability p mod 2^f / 2^f ({ups / count:.3f} vs {frac})")
    # s = 64: the low f + n bits of v over 2^(f+n+2) sites, (n, f) = (4, 2)
    n, f, k, s = 4, 2, 32, 64
    count = 1 << (f + n + 2)
    m0, m1, ds = dealt_sites(n, f, k, s, "round", count)
    r0, _, _ = run_layer(m0, m1, ds, [5] * count, False, k, s)
    stat = chi2_uniform([r.v % (1 << (f + n)) for r in r0], 1 << (f + n))
    check(stat < chi2_critical((1 << (f + n)) - 1), f"s = 64: the low bits of v are uniform (chi2 {stat:.1f})")


def protocol_sites(n: int, f: int, k: int, count: int, mode: str):
    dealer = IdealDealer()
    sid = b"tm-gen-" + secrets.token_bytes(4)
    desc = cp.site_descriptor(n, f, k, 64, mode)
    out = [None, None]

    def party(role):
        def fn(chan):
            ses = Session(role, chan, sid, k=k, s=64, prep="ideal", dealer=dealer, allow_refill=False)
            ses.setup(reserve_triples=cp.and_count(desc) * count)
            out[role] = (cp.generate(ses, desc, count), ses.delta_share)
        return fn

    run_two_party(party(0), party(1), timeout=600)
    return out[0][0], out[1][0], (out[0][1], out[1][1])


def test_protocol_material():
    n, f, k = 4, 2, 32
    T = tables.activation_table("relu", n, f)
    for mode in cp.MODES:
        p_values = [RNG.randrange(-(1 << (f + n + 1)), 1 << (f + n + 1)) for _ in range(6)]
        for first in (True, False):
            m0, m1, ds = protocol_sites(n, f, k, len(p_values), mode)
            r0, r1, _ = run_layer(m0, m1, ds, p_values, first, k, 64)
            check_results(m0, m1, p_values, r0, r1, mode)
            mod = 1 << (k + 64)
            for p, a, b, x, y in zip(p_values, m0, m1, r0, r1):
                qa, qb = Site(a).plane_a.table(T, x.x_hat), Site(b).plane_a.table(T, y.x_hat)
                check_eq((qa.value + qb.value) % mod, T[plain_truncate(p, f, n, mode, x.v_l)] % mod, "table at x_hat")
                check_eq((qa.mac + qb.mac) % mod, (ds[0] + ds[1]) * T[plain_truncate(p, f, n, mode, x.v_l)] % mod, "tag")


def test_arguments():
    n, f, k, s = 4, 2, 32, 64
    m0, m1, ds = dealt_sites(n, f, k, s, "round", 2)
    e0, e1, _ = dealt_sites(n, f, k, s, "exact", 1)
    c0, c1 = loopback_pair()
    checker = ArithChecker(0, c0, b"x", ds[0], s)
    op = Opener(0, c0, b"x", checker, ds[0])
    sites = [Site(m) for m in m0]
    before = (c0.msgs_sent, c0.bytes_sent)
    check(raises(ValueError, lambda: truncate_and_mask(op, sites, [1], True)), "one value per site")
    check(raises(ValueError, lambda: truncate_and_mask(op, sites, [1, AShare(0, 0, 96)], True)), "ints for the first layer")
    check(raises(ValueError, lambda: truncate_and_mask(op, sites, [1, 2], False)), "sharings in wire mode")
    check(raises(ValueError, lambda: truncate_and_mask(op, sites + [Site(e0[0])], [1, 2, 3], True)), "mixed modes")
    check(raises(ValueError, lambda: truncate_and_mask(op, sites, [AShare(0, 0, 8), AShare(0, 0, 8)], False)),
          "modulus too small")
    check(raises(ValueError, lambda: op.open_shared([1 << 10], 10)), "share out of range")
    check(raises(ValueError, lambda: op.open_sign_exact([AShare(1, 2, 96)])), "release needs sign_exact")
    check(raises(ValueError, lambda: Site(e0[0]).plane_b if False else Site(m0[0]).plane_b), "no plane B in mode round")
    check(raises(ValueError, lambda: Opener(1, c0, b"x", checker, 0)), "role must match the channel")
    check_eq((c0.msgs_sent, c0.bytes_sent), before, "argument errors send nothing")
    check_eq(truncate_and_mask(op, [], [], True), [], "an empty layer")
    check_eq(plain_truncate(-5, 2, 4, "exact"), (-5 >> 2) % 16, "plain floor")
    check_eq(plain_truncate(7, 2, 4, "round", 2), ((7 >> 2) + 1) % 16, "plain rounding up")
    check_eq(plain_truncate(7, 2, 4, "round", 3), (7 >> 2) % 16, "plain no rounding")
    check(raises(ValueError, lambda: plain_truncate(7, 2, 4, "round")), "round needs v_l")
    # a peer that sends a short opening message aborts
    sid = b"tm-abort"
    sites1 = [Site(m) for m in m1]

    def honest(chan):
        ck = ArithChecker(0, chan, sid, ds[0], s)
        o = Opener(0, chan, sid, ck, ds[0])
        try:
            truncate_and_mask(o, sites, [3, 4], True)
            return "ok"
        except ProtocolAbort as exc:
            return str(exc)

    def short(chan):
        ck = ArithChecker(1, chan, sid, ds[1], s)
        o = Opener(1, chan, sid, ck, ds[1])
        try:
            truncate_and_mask(o, sites1[:1], [3], True)
        except ProtocolAbort:
            return "abort"
        return "ok"

    r0, r1, _, _ = run_two_party(honest, short)
    check(r0 != "ok" and "length" in r0, "a short message aborts the honest party")
    c0.close()
    c1.close()


def test_release_payload():
    """The release decodes the authenticated payload v mod 2^k as a signed k-bit integer."""
    k, s = 32, 64
    w = k + s
    d0, d1 = RNG.getrandbits(s), RNG.getrandbits(s)
    delta = d0 + d1
    values = [5, -7, (1 << (k - 1)) - 1, -(1 << (k - 1)), 0]
    # the sign-exact embeddings, and the same representatives plus a multiple of 2^k
    reps = [v % (1 << w) for v in values] + [(v + (1 << (w - 1)) + (3 << k)) % (1 << w) for v in values]
    shares = [deal(r, delta, w) for r in reps]
    sid = b"tm-release"

    def party(role):
        def fn(chan):
            ck = ArithChecker(role, chan, sid, (d0, d1)[role], s)
            op = Opener(role, chan, sid, ck, (d0, d1)[role])
            return op.open_sign_exact([sh[role].with_sign_exact(True) for sh in shares])
        return fn

    o0, o1, _, _ = run_two_party(party(0), party(1))
    check_eq(o0, o1, "both parties release the same integers")
    check_eq(o0, values + values, "the release is the signed k-bit payload; bits above k are ignored")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    selected = [sys.argv[sys.argv.index("-k") + 1]] if "-k" in sys.argv else []
    failed = []
    start = time.perf_counter()
    for name, fn in tests:
        if selected and not any(s in name for s in selected):
            continue
        before = CHECKS
        t = time.perf_counter()
        try:
            fn()
            print(f"  ok   {name:<30} {CHECKS - before:>8} checks {time.perf_counter() - t:7.1f} s", flush=True)
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
