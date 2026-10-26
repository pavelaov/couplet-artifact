"""Exhaustive correctness tests for the two-plane DPF and DCF (DESIGN.md section 3).

For n in 1..8, k in {32, 64}, s = 64, EVERY alpha in [2^n], beta in
{0, 1, 2^k - 1, random} and a MAC key Delta = Delta_0 + Delta_1 formed from
two independent uniform s-bit shares WITHOUT reduction (so about half of the
keys use Delta >= 2^s), the test checks that

  * EvalAll (both parties, both planes) satisfies, for ALL x in [2^n],
        y0 + y1   == beta               * [x == alpha]  (DPF) / [x < alpha] (DCF)
        tau0+tau1 == (Delta*beta mod 2^w) * [x == alpha]  (DPF) / [x < alpha] (DCF)
    mod 2^w, w = k + s, and all outputs lie in [0, 2^w);
  * per-point Eval agrees with EvalAll and satisfies the same identities,
    at ALL x when n <= FULL_SCALAR_MAX_N (6 by default) and at a boundary +
    random sample of 16 points otherwise; ``--full`` (or COUPLET_FULL=1)
    raises the bound to 8 so every x is also evaluated point by point;
  * keys survive to_bytes/from_bytes and have the documented size;
  * the release check of DESIGN.md 2.3 passes on honest shares: with
    v = y0 + y1, a coin chi in [0, 2^s), y = chi * v and
    zeta_b = chi * tau_b - Delta_b * y mod 2^w, zeta_0 + zeta_1 == 0
    (single openings and batched openings); a negative control shows that the
    same check FAILS for keys generated with Delta reduced mod 2^s;
  * explicit roots reproduce keys bit for bit, and the Delta range
    [0, 2^(s+1)) is enforced.

Run as a script: ``python tests/test_fss_exhaustive.py [--full] [--quick]``
(``--quick`` restricts the exhaustive part to n <= 5; it is meant for smoke
runs only).  The exit status is non-zero on any failure and the number of
individual checks is printed at the end.  All randomness is seeded.  The
``test_*`` functions are also collectable by pytest, which is not required.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from couplet import fss  # noqa: E402

QUICK = "--quick" in sys.argv
NS = tuple(range(1, 6)) if QUICK else tuple(range(1, 9))
KS = (32, 64)
S = 64
SEED = "Couplet.1-exhaustive-2026-10-10"
FULL = "--full" in sys.argv or os.environ.get("COUPLET_FULL") == "1"
FULL_SCALAR_MAX_N = 8 if FULL else 6
SAMPLE_POINTS = 16

FAMILIES = {
    "DPF": (fss.dpf_gen, fss.dpf_eval, fss.dpf_eval_all, fss.DPFKey),
    "DCF": (fss.dcf_gen, fss.dcf_eval, fss.dcf_eval_all, fss.DCFKey),
}
CASES = [(fam, n, k) for fam in FAMILIES for n in NS for k in KS]


class Counter:
    """Counts individual checks by kind."""

    def __init__(self) -> None:
        self.by_kind: dict[str, int] = {}
        self.keys = 0
        self.keys_big_delta = 0

    def add(self, kind: str, count: int = 1) -> None:
        self.by_kind[kind] = self.by_kind.get(kind, 0) + count

    @property
    def total(self) -> int:
        return sum(self.by_kind.values())

    def merge(self, other: "Counter") -> None:
        for kind, c in other.by_kind.items():
            self.add(kind, c)
        self.keys += other.keys
        self.keys_big_delta += other.keys_big_delta


GLOBAL = Counter()


def draw_delta(rng: random.Random) -> tuple[int, int, int]:
    """(Delta_0, Delta_1, Delta = Delta_0 + Delta_1): independent s-bit shares, no reduction."""
    d0 = rng.randrange(1 << S)
    d1 = rng.randrange(1 << S)
    return d0, d1, d0 + d1


def expected_vectors(family: str, n: int, alpha: int, beta: int, dbeta: int, mod: int):
    xs = np.arange(1 << n)
    ind = (xs == alpha) if family == "DPF" else (xs < alpha)
    ind = ind.astype(object)
    return (ind * beta) % mod, (ind * dbeta) % mod


def expected_size(family: str, n: int, k: int, s: int) -> int:
    w = (k + s + 7) // 8
    per_level = 16 + 1 + (2 * w if family == "DCF" else 0)
    return 4 + 5 + 16 + n * per_level + 2 * w


def sample_points(n: int, alpha: int, rng: random.Random) -> list[int]:
    size = 1 << n
    pts = {0, size - 1, alpha, (alpha - 1) % size, (alpha + 1) % size}
    while len(pts) < min(SAMPLE_POINTS, size):
        pts.add(rng.randrange(size))
    return sorted(pts)


def _assert_all(ok: np.ndarray, label: str) -> None:
    if not bool(np.all(ok)):
        bad = int(np.flatnonzero(np.logical_not(np.asarray(ok, dtype=bool)))[0])
        raise AssertionError(f"{label}: first failure at x={bad}")


def release_check(chis, values, tags0, tags1, d0: int, d1: int, mod: int) -> bool:
    """ArithCheck of DESIGN.md 2.3 on opened values (one modulus).

    y = sum chi_j v_j;  zeta_b = sum chi_j tau_{j,b} - Delta_b y;  accept iff
    zeta_0 + zeta_1 == 0 (mod ``mod``)."""
    y = sum(c * v for c, v in zip(chis, values)) % mod
    zeta0 = (sum(c * t for c, t in zip(chis, tags0)) - d0 * y) % mod
    zeta1 = (sum(c * t for c, t in zip(chis, tags1)) - d1 * y) % mod
    return (zeta0 + zeta1) % mod == 0


def run_case(family: str, n: int, k: int, counter: Counter | None = None) -> Counter:
    """Run every check for one (family, n, k); returns the counter."""
    counter = counter if counter is not None else Counter()
    gen, ev, evall, keycls = FAMILIES[family]
    rng = random.Random(f"{SEED}-{family}-{n}-{k}")
    size = 1 << n
    mod = 1 << (k + S)
    betas = (0, 1, (1 << k) - 1, rng.randrange(1 << k))
    exhaustive_points = n <= FULL_SCALAR_MAX_N
    for alpha in range(size):
        for beta in betas:
            d0, d1, delta = draw_delta(rng)
            dbeta = delta * beta % mod
            k0, k1 = gen(alpha, beta, delta, n, k, S, rng)
            counter.keys += 2
            if delta >> S:
                counter.keys_big_delta += 2
            tag = f"{family} n={n} k={k} alpha={alpha} beta={beta} delta={delta:#x}"

            # EvalAll correctness on both planes, for ALL x.
            y0, t0 = evall(0, k0)
            y1, t1 = evall(1, k1)
            exp_y, exp_t = expected_vectors(family, n, alpha, beta, dbeta, mod)
            for name, arr in (("y0", y0), ("tau0", t0), ("y1", y1), ("tau1", t1)):
                if len(arr) != size or not all(0 <= int(v) < mod for v in arr):
                    raise AssertionError(f"{tag}: {name} has wrong length or out-of-range entries")
                counter.add("evalall_range")
            _assert_all((y0 + y1) % mod == exp_y, f"{tag}: EvalAll value plane")
            _assert_all((t0 + t1) % mod == exp_t, f"{tag}: EvalAll tag plane")
            counter.add("evalall_value", size)
            counter.add("evalall_tag", size)

            # Release check over the whole domain as one batch, fresh coins.
            chis = [rng.randrange(1 << S) for _ in range(size)]
            if not release_check(chis, (y0 + y1) % mod, t0, t1, d0, d1, mod):
                raise AssertionError(f"{tag}: batched release check failed")
            counter.add("release_batch")

            # Per-point Eval: agreement with EvalAll and direct correctness.
            xs = range(size) if exhaustive_points else sample_points(n, alpha, rng)
            for x in xs:
                a0 = ev(0, k0, x)
                a1 = ev(1, k1, x)
                if a0 != (y0[x], t0[x]):
                    raise AssertionError(f"{tag}: Eval(0,x={x}) != EvalAll")
                if a1 != (y1[x], t1[x]):
                    raise AssertionError(f"{tag}: Eval(1,x={x}) != EvalAll")
                if (a0[0] + a1[0]) % mod != exp_y[x] or (a0[1] + a1[1]) % mod != exp_t[x]:
                    raise AssertionError(f"{tag}: Eval sum wrong at x={x}")
                counter.add("eval_vs_evalall", 4)
                counter.add("eval_sum", 2)

            # Serialisation round trip and size.
            for key in (k0, k1):
                blob = key.to_bytes()
                if keycls.from_bytes(blob) != key or fss.key_from_bytes(blob) != key:
                    raise AssertionError(f"{tag}: serialisation round trip failed")
                if key.size_bytes() != len(blob) or len(blob) != expected_size(family, n, k, S):
                    raise AssertionError(f"{tag}: unexpected key size {len(blob)}")
                counter.add("serialisation", 3)
    return counter


def test_exhaustive() -> None:
    """All (family, n, k) cases; about half of the keys must use Delta >= 2^s."""
    for family, n, k in CASES:
        t0 = time.perf_counter()
        local = run_case(family, n, k)
        GLOBAL.merge(local)
        print(f"  {family} n={n} k={k}: {local.total:>9} checks, {local.keys:>4} keys "
              f"({local.keys_big_delta:>4} with Delta >= 2^s), {time.perf_counter() - t0:6.1f} s",
              flush=True)
    frac = GLOBAL.keys_big_delta / GLOBAL.keys
    if not 0.4 < frac < 0.6:
        raise AssertionError(f"fraction of keys with Delta >= 2^s is {frac:.3f}, expected about 1/2")
    GLOBAL.add("delta_carry_fraction")


def test_release_check() -> None:
    """Explicit release check on random keys and points (DESIGN.md 2.3).

    Single openings: zeta_b = chi*tau_b - Delta_b*(chi*v) with v = y_0 + y_1,
    zeta_0 + zeta_1 == 0 mod 2^w.  Negative control: keys generated with
    Delta reduced mod 2^s fail the same check at a point where the opened
    value is odd (so chi*v is non-zero mod 2^k for odd chi)."""
    rng = random.Random(f"{SEED}-release")
    n_pos = n_neg = 0
    for family, (gen, ev, _, _) in FAMILIES.items():
        for k in KS:
            mod = 1 << (k + S)
            for trial in range(200):
                n = rng.randrange(1, 13)
                alpha = rng.randrange(1 << n)
                beta = rng.choice((0, 1, (1 << k) - 1, rng.randrange(1 << k)))
                d0, d1, delta = draw_delta(rng)
                k0, k1 = gen(alpha, beta, delta, n, k, S, rng)
                pts = {alpha, (alpha - 1) % (1 << n), rng.randrange(1 << n), rng.randrange(1 << n)}
                for x in sorted(pts):
                    y0, tau0 = ev(0, k0, x)
                    y1, tau1 = ev(1, k1, x)
                    v = (y0 + y1) % mod
                    chi = rng.randrange(1 << S)
                    y = chi * v % mod
                    zeta0 = (chi * tau0 - d0 * y) % mod
                    zeta1 = (chi * tau1 - d1 * y) % mod
                    if (zeta0 + zeta1) % mod != 0:
                        raise AssertionError(f"release check failed: {family} n={n} k={k} x={x} "
                                             f"alpha={alpha} beta={beta} delta={delta:#x}")
                    if not release_check([chi], [v], [tau0], [tau1], d0, d1, mod):
                        raise AssertionError("release_check helper disagrees with the direct formula")
                    n_pos += 1
            # Negative control: a reduced Delta must be caught.
            for trial in range(20):
                n = rng.randrange(2, 9)
                alpha = rng.randrange(1, 1 << n)        # alpha >= 1 so that DCF has x < alpha
                beta = rng.randrange(1 << k) | 1        # odd payload
                while True:
                    d0, d1, delta = draw_delta(rng)
                    if delta >> S:
                        break
                k0, k1 = gen(alpha, beta, delta & ((1 << S) - 1), n, k, S, rng)
                x = alpha if family == "DPF" else alpha - 1
                y0, tau0 = ev(0, k0, x)
                y1, tau1 = ev(1, k1, x)
                v = (y0 + y1) % mod
                if v != beta:
                    raise AssertionError("negative control: unexpected opened value")
                chi = rng.randrange(1 << S) | 1          # odd coin: chi*v is odd
                if release_check([chi], [v], [tau0], [tau1], d0, d1, mod):
                    raise AssertionError("negative control: reduced Delta passed the release check")
                n_neg += 1
    GLOBAL.add("release_single", n_pos)
    GLOBAL.add("release_negative_control", n_neg)
    print(f"  release check: {n_pos} honest openings accepted, {n_neg} reduced-Delta openings rejected")


def _raises(exc, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


def test_explicit_roots_and_ranges() -> None:
    """roots=(s0, s1) reproduces keys bit for bit; Delta range is [0, 2^(s+1))."""
    rng = random.Random(f"{SEED}-roots")
    checks = 0
    for family, (gen, ev, _, _) in FAMILIES.items():
        for k in KS:
            mod = 1 << (k + S)
            for _ in range(20):
                n = rng.randrange(1, 11)
                alpha, beta = rng.randrange(1 << n), rng.randrange(1 << k)
                _, _, delta = draw_delta(rng)
                k0, k1 = gen(alpha, beta, delta, n, k, S, rng)
                r0, r1 = gen(alpha, beta, delta, n, k, S, roots=(k0.seed, k1.seed))
                if (r0, r1) != (k0, k1) or r0.to_bytes() != k0.to_bytes() or r1.to_bytes() != k1.to_bytes():
                    raise AssertionError(f"{family}: explicit roots do not reproduce the keys")
                if fss.seed_from_int(fss.seed_to_int(k0.seed)) != k0.seed:
                    raise AssertionError("seed int conversion is not a bijection")
                checks += 2
            # extreme MAC keys: 0, 2^s - 1, 2^s, 2^(s+1) - 1
            for delta in (0, (1 << S) - 1, 1 << S, (1 << (S + 1)) - 1):
                n = 5
                alpha, beta = rng.randrange(1, 1 << n), (1 << k) - 1
                k0, k1 = gen(alpha, beta, delta, n, k, S, rng)
                for x in range(1 << n):
                    y0, t0 = ev(0, k0, x)
                    y1, t1 = ev(1, k1, x)
                    ind = (x == alpha) if family == "DPF" else (x < alpha)
                    if (y0 + y1) % mod != beta * ind or (t0 + t1) % mod != delta * beta * ind % mod:
                        raise AssertionError(f"{family}: wrong output for extreme Delta {delta:#x}")
                    checks += 2
            _raises(ValueError, gen, 0, 1, 1 << (S + 1), 3, k, S, rng)
            _raises(ValueError, gen, 0, 1, -1, 3, k, S, rng)
            _raises(ValueError, gen, 0, 1 << k, 1, 3, k, S, rng)
            _raises(ValueError, gen, 8, 1, 1, 3, k, S, rng)
            _raises(ValueError, gen, 0, 1, 1, 3, k, S, roots=(b"\x00" * 15, b"\x00" * 16))
            _raises(ValueError, gen, 0, 1, 1, 3, k, S, rng, roots=(b"\x00" * 16, b"\x00" * 16))
            checks += 6
    GLOBAL.add("roots_and_ranges", checks)
    print(f"  explicit roots / parameter ranges: {checks} checks")


def main() -> int:
    mode = "full (scalar Eval at every x for all n)" if FULL else \
        f"default (scalar Eval at every x for n <= {FULL_SCALAR_MAX_N}, {SAMPLE_POINTS}-point sample above)"
    print(f"Couplet exhaustive DPF/DCF test, n in {NS[0]}..{NS[-1]}, k in {KS}, s={S}, "
          f"Delta = Delta_0 + Delta_1 unreduced; mode: {mode}")
    start = time.perf_counter()
    try:
        test_explicit_roots_and_ranges()
        test_release_check()
        test_exhaustive()
    except AssertionError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"keys generated: {GLOBAL.keys} ({GLOBAL.keys_big_delta} with Delta >= 2^s)")
    for kind in sorted(GLOBAL.by_kind):
        print(f"  {kind:<26} {GLOBAL.by_kind[kind]:>10}")
    print(f"TOTAL CHECKS: {GLOBAL.total}  (all passed, {time.perf_counter() - start:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
