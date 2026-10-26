"""Exhaustive tests of couplet.planes: the local queries of one authenticated plane
(DESIGN_M3.md section 3, Carve).

For n in {3, ..., 8} the sites (plane A keys at the offsets rho_h) are produced
by the REAL generation protocol (``circuits_ppml.generate`` on a session with
ideal preprocessing; three fresh offsets per n) and, for n in {4, 6}, on the
real preprocessing (base OT, OT extension, leaky AND, bucketing; two offsets).
For EVERY x_hat in [N] the two parties' queries are reconstructed on BOTH
components (value and tag, with the unreduced Delta = Delta_0 + Delta_1) and
compared with the plaintext model z = (x_hat - alpha) mod N:

  * table: three public tables (random sign-exact entries, ReLU, identity);
  * equality [z = c] for every c; comparison [z < c] for every c in [0, N];
    sign, nonneg, wrap bit; prefix and suffix sums for every j in [0, N];
  * interval containment [lo <= z <= hi]: every pair (lo, hi) for every
    x_hat for n <= 6, 24 random pairs per x_hat for n = 7, 8 (the query is the
    locator at [x_hat - hi, x_hat - lo]); the cyclic locator [alpha in [l, u]]:
    every pair (l, u) for every n;
  * the flags (indicators exact and sign-exact, tables sign-exact only), the
    modulus of a reduced query, the argument checks.

Run from artifact/python:  python tests/test_planes.py  (``-k name`` selects
tests, ``--no-real`` skips the real preprocessing).  Prints the number of
checks; exits non-zero on any failure.
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

from couplet import circuits_ppml as cp  # noqa: E402
from couplet import fss  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.planes import Plane  # noqa: E402
from couplet.prep import IdealDealer  # noqa: E402
from couplet.session import Session  # noqa: E402
from couplet.spdz2k import AShare  # noqa: E402

NO_REAL = "--no-real" in sys.argv
S = 64
CHECKS = 0
RNG = random.Random(0x9A7E5)


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


# --------------------------------------------------------------------------- #
# Sites from the generation protocol
# --------------------------------------------------------------------------- #

def protocol_sites(n: int, f: int, k: int, count: int, prep: str, mode: str = "round"):
    """Both parties' site material and the MAC key, from Pi_Gen on one session."""
    dealer = IdealDealer() if prep == "ideal" else None
    sid = b"planes-" + secrets.token_bytes(4)
    desc = cp.site_descriptor(n, f, k, S, mode)
    budget = cp.and_count(desc) * count
    out = [None, None]

    def party(role):
        def fn(chan):
            ses = Session(role, chan, sid, k=k, s=S, prep=prep, dealer=dealer, allow_refill=False)
            ses.setup(reserve_triples=budget)
            mats = cp.generate(ses, desc, count)
            out[role] = (mats, ses.delta_share, ses.triple_requests)
            return None
        return fn

    run_two_party(party(0), party(1), timeout=900)
    (m0, d0, q0), (m1, d1, q1) = out
    check_eq((q0, q1), (1, 1), "one triple request")
    return m0, m1, d0 + d1


def offset_of(m0: cp.SiteMaterial, m1: cp.SiteMaterial) -> int:
    """The index alpha of plane A from both parties' material (mode round: rho_h)."""
    rp = (m0.rp[0] + m1.rp[0]) % (1 << m0.m)
    return (rp >> m0.f) % (1 << m0.n)


# --------------------------------------------------------------------------- #
# The exhaustive check of one plane pair
# --------------------------------------------------------------------------- #

def rec(a: AShare, b: AShare, delta: int, want: int, tag: str, exact: bool) -> None:
    """One check: value, tag and flags of a reconstructed query."""
    global CHECKS
    CHECKS += 1
    mod = 1 << a.bits
    if a.bits != b.bits:
        raise AssertionError(f"{tag}: moduli differ")
    v = (a.value + b.value) % mod
    if v != want % mod:
        raise AssertionError(f"{tag}: value {v}, expected {want % mod}")
    if (a.mac + b.mac) % mod != delta * want % mod:
        raise AssertionError(f"{tag}: tag mismatch")
    if a.sign_exact != b.sign_exact or not a.sign_exact:
        raise AssertionError(f"{tag}: sign_exact flag")
    if a.exact != exact or b.exact != exact:
        raise AssertionError(f"{tag}: exact flag {a.exact}, expected {exact}")


def exhaustive(P0: Plane, P1: Plane, alpha: int, delta: int, k: int, label: str) -> None:
    n, N = P0.n, P0.N
    w = k + S
    half = 1 << (w - 1)
    tables = {"random": [RNG.randrange(-(1 << 40), 1 << 40) for _ in range(N)],
              "relu": [z if z < N // 2 else 0 for z in range(N)],
              "identity": list(range(N))}
    tables["random"][RNG.randrange(N)] = -half                   # the most negative sign-exact entry
    tables["random"][RNG.randrange(N)] = half - 1
    full = n <= 6
    for x_hat in range(N):
        z = (x_hat - alpha) % N
        for name, T in tables.items():
            rec(P0.table(T, x_hat), P1.table(T, x_hat), delta, T[z], f"{label} table {name} x_hat={x_hat}", False)
        for c in range(N):
            rec(P0.equality(x_hat, c), P1.equality(x_hat, c), delta, int(z == c), f"{label} equality", True)
        for c in range(N + 1):
            rec(P0.comparison(x_hat, c), P1.comparison(x_hat, c), delta, int(z < c), f"{label} comparison", True)
        rec(P0.sign(x_hat), P1.sign(x_hat), delta, int(z >= N // 2), f"{label} sign", True)
        rec(P0.nonneg(x_hat), P1.nonneg(x_hat), delta, int(z < N // 2), f"{label} nonneg", True)
        rec(P0.wrap(x_hat), P1.wrap(x_hat), delta, int(alpha > x_hat), f"{label} wrap", True)
        pairs = ([(lo, hi) for lo in range(N) for hi in range(lo, N)] if full
                 else [sorted((RNG.randrange(N), RNG.randrange(N))) for _ in range(24)])
        for lo, hi in pairs:
            rec(P0.z_interval(x_hat, lo, hi), P1.z_interval(x_hat, lo, hi), delta, int(lo <= z <= hi),
                f"{label} interval", True)
    for j in range(N + 1):
        rec(P0.prefix(j), P1.prefix(j), delta, int(alpha < j), f"{label} prefix", True)
        rec(P0.suffix(j), P1.suffix(j), delta, int(alpha >= j), f"{label} suffix", True)
    for l, u in ((l, u) for l in range(N) for u in range(N)):
        inside = (l <= alpha <= u) if l <= u else (alpha >= l or alpha <= u)
        rec(P0.locator(l, u), P1.locator(l, u), delta, int(inside), f"{label} locator", True)
    for u in range(N):
        rec(P0.point(u), P1.point(u), delta, int(u == alpha), f"{label} point", True)
    # reduced modulus
    a, b = P0.suffix(1, bits=n + S), P1.suffix(1, bits=n + S)
    check_eq((a.bits, b.bits), (n + S, n + S), "reduced query modulus")
    check_eq((a.value + b.value) % (1 << (n + S)), int(alpha >= 1), "reduced query value")


def run_n(n: int, prep: str, count: int, f: int = 2) -> None:
    k = 32 if n % 2 else 64
    t = time.perf_counter()
    m0, m1, delta = protocol_sites(n, f, k, count, prep)
    gen = time.perf_counter() - t
    before = CHECKS
    for j, (a, b) in enumerate(zip(m0, m1)):
        alpha = offset_of(a, b)
        P0, P1 = Plane(0, a.plane_a), Plane(1, b.plane_a)
        check_eq((P0.n, P0.N, P0.width), (n, 1 << n, k + S), "plane parameters")
        exhaustive(P0, P1, alpha, delta, k, f"n={n} {prep} site {j}")
    print(f"    n={n} k={k} {prep:<5}: {count} offsets, {CHECKS - before:>7} checks, generation {gen:5.1f} s, "
          f"queries {time.perf_counter() - t - gen:5.1f} s", flush=True)


def test_exhaustive_ideal():
    for n in (3, 4, 5, 6, 7, 8):
        run_n(n, "ideal", 3)


def test_exhaustive_real():
    if NO_REAL:
        print("    skipped (--no-real)")
        return
    for n in (4, 6):
        run_n(n, "real", 2)


def test_flags_and_arguments():
    delta = RNG.getrandbits(S) + RNG.getrandbits(S)
    k0, k1 = fss.dpf_gen(5, 1, delta, 4, 32, S)
    P0, P1 = Plane(0, k0), Plane(1, k1)
    check(raises(ValueError, lambda: Plane(1, k0)), "wrong party")
    check(raises(ValueError, lambda: Plane(0, fss.dcf_gen(1, 1, delta, 4, 32, S)[0])), "a DCF key is not a plane")
    check(raises(ValueError, lambda: P0.table([1] * 15, 0)), "table length")
    check(raises(ValueError, lambda: P0.table([1 << 95] * 16, 0)), "table entry not sign-exact")
    check(raises(ValueError, lambda: P0.table([1] * 16, 16)), "x_hat range")
    check(raises(ValueError, lambda: P0.comparison(0, 17)), "c range")
    check(raises(ValueError, lambda: P0.z_interval(0, 3, 2)), "lo > hi")
    check(raises(ValueError, lambda: P0.locator(16, 0)), "locator range")
    check(raises(ValueError, lambda: P0.suffix(17)), "suffix range")
    t = P0.table([-3] * 16, 0)
    check_eq((t.sign_exact, t.exact, t.bits), (True, False, 96), "table flags")
    e = P1.equality(3, 3)
    check_eq((e.sign_exact, e.exact), (True, True), "indicator flags")
    s = (P0.equality(2, 1) + P1.equality(2, 1))
    check_eq((s.sign_exact, s.exact), (False, False), "linear operations clear the flags")
    check_eq(P0.table([-1] * 16, 7).bits, 96, "default modulus w")


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
            print(f"  ok   {name:<30} {CHECKS - before:>9} checks {time.perf_counter() - t:7.1f} s", flush=True)
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
