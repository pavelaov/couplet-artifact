"""Tests for couplet.spdz2k: SPDZ2k sharings, ArithCheck and the release discipline
(DESIGN.md sections 2.3 and 5.3; changes C, E, F, G).

  * ``AShare`` algebra on dealer-style sharings: every local operation keeps
    ``v_0 + v_1 = v~`` and ``mu_0 + mu_1 = Delta v~`` with the UNREDUCED key
    ``Delta = Delta_0 + Delta_1`` (including ``Delta >= 2^s``; a reduced key is
    shown to break the MAC relation); exactness flags; argument validation;
  * honest ArithCheck: never aborts, for mixed moduli, for ``Delta_0 + Delta_1``
    above and below ``2^s``; exact message and round counts;
  * ``open_final``: check, open, check; output masks for sharings that are not
    exact; refusals before any message;
  * Lemma 2, empirically, for ``s = 4, k = 4``: the pass rate of a cheating
    opener for six strategies, each compared with the EXACT probability
    obtained by enumerating all keys, coefficients and guesses, and with the
    bound ``2^(-s + log2(s+1))``;
  * ``s = 64``: every wrong opening, wrong MAC share and wrong key share is
    rejected; the checker is dead afterwards;
  * malformed, mirrored and replayed messages; binding of a check to its
    openings, its counter and its session.

Two parties run as two threads (``couplet.net.run_two_party``).  Run from
artifact/python:  python tests/test_spdz2k.py   (``-k name`` selects tests,
``--trials N`` sets the number of Lemma 2 trials per strategy, default 20000).
Prints the number of checks and exits non-zero on any failure.
"""

from __future__ import annotations

import itertools
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

from couplet import hashes as hs  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.spdz2k import (AShare, ArithChecker, check_coefficients, deal, lemma2_bound,  # noqa: E402
                             reconstruct, share_bytes)

CHECKS = 0
NOTES: list = []
RNG = random.Random(0x5BD2)


def _arg(name: str, default: int) -> int:
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


LEMMA2_TRIALS = _arg("--trials", 20000)


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


def check_raises(exc, fn, msg: str = "") -> None:
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{msg}: did not raise {exc.__name__}")


def run2(fn0, fn1):
    """Run both parties; each function returns its result (exceptions propagate)."""
    r0, r1, st0, st1 = run_two_party(fn0, fn1, timeout=120)
    return r0, r1, st0, st1


def keys(s: int, mode: str = "random") -> tuple:
    """Two key shares in [0, 2^s): random, forced carry (sum >= 2^s) or no carry."""
    top = (1 << s) - 1
    if mode == "carry":
        d0 = RNG.randrange(1 << (s - 1), 1 << s)
        return d0, RNG.randrange((1 << s) - d0, 1 << s)
    if mode == "nocarry":
        d0 = RNG.randrange(1 << (s - 1))
        return d0, RNG.randrange(1 << (s - 1))
    if mode == "max":
        return top, top
    return RNG.getrandbits(s), RNG.getrandbits(s)


def mac_ok(a0: AShare, a1: AShare, delta: int) -> bool:
    value, mac = reconstruct(a0, a1)
    return mac == delta * value % (1 << a0.bits)


def outmask(rho: int, delta: int, k: int, s: int) -> tuple:
    """Dealer-style output mask [[2^k rho]]: 2^k times s-bit shares of rho and of Delta rho mod 2^s."""
    m0, t0 = RNG.getrandbits(s), RNG.getrandbits(s)
    return (AShare(m0 << k, t0 << k, k + s),
            AShare(((rho - m0) % (1 << s)) << k, ((delta * rho - t0) % (1 << s)) << k, k + s))


# --------------------------------------------------------------------------- #
# AShare
# --------------------------------------------------------------------------- #

def test_ashare_algebra():
    for (k, s) in ((64, 64), (32, 64), (4, 4), (8, 4), (16, 8)):
        w = k + s
        mod = 1 << w
        for mode in ("carry", "nocarry", "max", "random"):
            d0, d1 = keys(s, mode)
            delta = d0 + d1                                   # never reduced
            if mode in ("carry", "max"):
                check(delta >= 1 << s, "the forced-carry key exceeds 2^s")
            for _ in range(12):
                u, v = RNG.randrange(mod), RNG.randrange(1 << k)
                a = deal(u, delta, w, rng=RNG)
                b = deal(v, delta, w, exact=True, rng=RNG)
                check_eq(reconstruct(*a)[0], u, "deal reconstructs the representative")
                check(mac_ok(*a, delta) and mac_ok(*b, delta), "deal: MAC relation with the unreduced key")
                # a reduced key breaks the relation as soon as the product overflows
                if delta >> s and (u * (1 << s)) % mod:
                    check(not mac_ok(*a, delta % (1 << s)),
                          "Delta mod 2^s does not satisfy the MAC relation")
                c = RNG.choice([0, 1, 2, 3, -1, -7, RNG.getrandbits(k), -RNG.getrandbits(w + 5)])
                cases = {
                    "add": ([x + y for x, y in zip(a, b)], (u + v) % mod),
                    "sub": ([x - y for x, y in zip(a, b)], (u - v) % mod),
                    "neg": ([-x for x in a], -u % mod),
                    "scale": ([x.scale(c) for x in a], u * c % mod),
                    "rmul": ([c * x for x in a], u * c % mod),
                    "add_public": ([x.add_public(c, role, d) for role, (x, d) in
                                    enumerate(zip(a, (d0, d1)))], (u + c) % mod),
                }
                for name, (shares, want) in cases.items():
                    check_eq(reconstruct(*shares)[0], want, f"{name}: value")
                    check(mac_ok(*shares, delta), f"{name}: MAC relation")
                    check(shares[0].bits == w and shares[0].exact == shares[1].exact, f"{name}: metadata")
                # exactness flags (public metadata)
                check(not (a[0] + b[0]).exact and not (b[0] + b[0]).exact, "a sum is not marked exact")
                check(not (-b[0]).exact and not b[0].scale(2).exact, "negation and multiples are not exact")
                check(b[0].scale(1).exact and b[0].scale(0).exact and a[0].scale(0).exact,
                      "c = 1 keeps the flag, c = 0 gives an exact zero")
                check(not a[0].scale(1).exact, "c = 1 does not create exactness")
                check(b[0].add_public(0, 0, d0).exact and not b[0].add_public(1, 0, d0).exact,
                      "adding a non-zero constant clears the flag")
                check(b[0].with_exact(False).exact is False and a[0].with_exact().exact is True, "with_exact")
                # reduce: sharing of the representative mod 2^(n+s), payload v mod 2^n
                for n in sorted({1, 2, max(1, k // 2), k}):
                    red = [x.reduce(n, s) for x in a]
                    m = n + s
                    check_eq(reconstruct(*red)[0], u % (1 << m), f"reduce({n}): value")
                    check(mac_ok(*red, delta), f"reduce({n}): MAC relation modulo 2^(n+s)")
                    check_eq(reconstruct(*red)[0] % (1 << n), u % (1 << n), f"reduce({n}): payload")
                    redb = [x.reduce(n, s) for x in b]
                    check_eq(redb[0].exact, n == k, "reduce keeps the flag only for the full modulus")
                # an exact sharing really has a small representative; a reduced one need not
                check(reconstruct(*b)[0] < 1 << k, "exact means representative < 2^k")
    # the carry of the key is what the tests above exercise: at least one key had Delta >= 2^s
    check(sum(keys(4, "carry")) >= 16 and sum(keys(4, "nocarry")) < 16, "key helper")


def test_ashare_validation():
    a = AShare(5, 7, 12)
    check_raises(ValueError, lambda: AShare(1 << 12, 0, 12), "value out of range")
    check_raises(ValueError, lambda: AShare(0, -1, 12), "negative mac")
    check_raises(ValueError, lambda: AShare(0, 0, 0), "bits = 0")
    check_raises(ValueError, lambda: AShare(True, 0, 12), "bool value")
    check_raises(ValueError, lambda: AShare(0, 0, 12, exact=1), "exact must be a bool")
    check_raises(ValueError, lambda: a + AShare(1, 1, 13), "different moduli")
    check_raises(TypeError, lambda: a + 3, "adding an int is not a share operation")
    check_raises(TypeError, lambda: a.scale(1.5), "float coefficient")
    check_raises(TypeError, lambda: a * 2.0, "float coefficient via *")
    check_raises(ValueError, lambda: a.add_public(1, 2, 3), "role")
    check_raises(ValueError, lambda: a.add_public(1, 0, -3), "negative key share")
    check_raises(ValueError, lambda: a.reduce(9, 4), "reduce to a larger modulus")
    check_raises(ValueError, lambda: a.reduce(0, 4), "n = 0")
    check_raises(Exception, lambda: setattr(a, "value", 1), "shares are immutable")
    check_eq(a.modulus, 1 << 12, "modulus")
    check_eq(share_bytes(68), 9, "share_bytes")
    check_eq(lemma2_bound(4), 5 / 16, "Lemma 2 bound for s = 4")
    check(abs(lemma2_bound(64) - 2.0 ** (-64 + math.log2(65))) < 1e-30, "Lemma 2 bound for s = 64")
    chis = check_coefficients(b"\x01" * 32, 128, 50, 4)
    check(len(chis) == 50 and all(0 <= c < 16 for c in chis) and len(set(chis)) > 4,
          "coefficients are s-bit values")
    check(check_coefficients(b"\x01" * 32, 96, 50, 4) != chis, "coefficients depend on the modulus")
    check(check_coefficients(b"\x02" * 32, 128, 50, 4) != chis, "coefficients depend on the seed")
    check_raises(ValueError, lambda: ArithChecker(0, type("C", (), {"party": 1})(), b"x", 0, 4),
                 "role must match the channel")
    check_raises(ValueError, lambda: ArithChecker(0, type("C", (), {"party": 0})(), b"x", 16, 4),
                 "key share out of range")


# --------------------------------------------------------------------------- #
# Honest runs
# --------------------------------------------------------------------------- #

def _checkers(sid: bytes, s: int, d0: int, d1: int, body, cls=(ArithChecker, ArithChecker)):
    """Run ``body(role, checker)`` at both parties; returns (out0, out1, stats0, stats1)."""
    def party(role):
        def fn(chan):
            return body(role, cls[role](role, chan, sid, (d0, d1)[role], s))
        return fn
    return run2(party(0), party(1))


def test_honest_checks():
    runs = 0
    for (k, s) in ((64, 64), (32, 64), (8, 4), (4, 4)):
        w = k + s
        for mode in ("carry", "nocarry", "max", "random"):
            for rep in range(6):
                d0, d1 = keys(s, mode)
                delta = d0 + d1
                moduli = [w] + [n + s for n in sorted({1, 3, min(k, 5), k // 2}) if n <= k]
                plan = [RNG.choice(moduli) for _ in range(RNG.randrange(1, 14))]
                values = [RNG.randrange(1 << m) for m in plan]
                pairs = [deal(v, delta, m, rng=RNG) for v, m in zip(values, plan)]
                split = RNG.randrange(len(plan) + 1)

                def body(role, ck, pairs=pairs, split=split):
                    out = [ck.open_deferred(p[role]) for p in pairs[:split]]
                    pending_mid = ck.pending
                    ck.check()
                    out += ck.open_deferred_many([p[role] for p in pairs[split:]])
                    ck.check()
                    ck.check()                           # nothing pending: no message
                    return out, pending_mid, ck.pending, ck.openings, ck.checks

                r0, r1, st0, st1 = _checkers(secrets.token_bytes(8), s, d0, d1, body)
                runs += 1
                check_eq(r0[0], values, "P_0 opens the representatives")
                check_eq(r1[0], values, "P_1 opens the representatives")
                check_eq((r0[1], r0[2], r0[3]), (split, 0, len(plan)), "pending / opening counters")
                n_checks = int(split > 0) + int(split < len(plan))
                check_eq(r0[4], n_checks, "a check without openings is skipped")
                n_open_msgs = split + int(split < len(plan))
                check_eq(st0["rounds"], n_open_msgs + 4 * n_checks, "rounds: 1 per opening call, 4 per check")
                check_eq(st0["rounds"], st1["rounds"], "both endpoints count the same rounds")
                # bytes: value shares, plus per check a coin toss (96) and a committed zeta per modulus
                expect = sum(share_bytes(m) for m in plan)
                for part in (plan[:split], plan[split:]):
                    if part:
                        expect += 96 + 64 + sum(share_bytes(m) for m in set(part))
                check_eq(st0["bytes_sent"], expect, "bytes sent by P_0")
                check_eq(st1["bytes_sent"], expect, "bytes sent by P_1")
                check(set(st0["phases"]) <= {"arith-open", "arith-check", "(unmarked)"}, "channel phases")
    NOTES.append(f"honest ArithCheck: {runs} sessions with mixed moduli, none aborted "
                 f"(a quarter each with Delta >= 2^s forced, Delta < 2^s forced, Delta_b = 2^s - 1, random)")


def test_open_final():
    for (k, s) in ((64, 64), (32, 64), (4, 4)):
        w = k + s
        mod = 1 << w
        for mode in ("carry", "nocarry", "random"):
            d0, d1 = keys(s, mode)
            delta = d0 + d1
            v_exact = RNG.randrange(1 << k)
            v_big = RNG.randrange(1 << k, mod)                 # representative >= 2^k
            rho = RNG.getrandbits(s)
            exact = deal(v_exact, delta, w, exact=True, rng=RNG)
            big = deal(v_big, delta, w, rng=RNG)
            mask = outmask(rho, delta, k, s)
            check(mac_ok(*mask, delta) and reconstruct(*mask)[0] == rho << k, "output mask [[2^k rho]]")
            deferred = deal(RNG.randrange(1 << (3 + s)), delta, 3 + s, rng=RNG)

            def body(role, ck):
                chan = ck.chan
                # refusals: nothing is sent, the checker stays alive
                before = chan.msgs_sent
                refused = 0
                for bad in (lambda: ck.open_final(big[role]),
                            lambda: ck.open_final(exact[role], mask[role]),
                            lambda: ck.open_final(big[role], AShare(1, 0, w)),
                            lambda: ck.open_final(big[role], AShare(0, 1 << (k - 1) if k > 1 else 1, w)),
                            lambda: ck.open_final(big[role], AShare(0, 0, w - 1)),
                            lambda: ck.open_final(AShare(0, 0, s, True)),
                            lambda: ck.open_final_many([big[role]], []),
                            lambda: ck.open_deferred(5)):
                    try:
                        bad()
                    except ValueError:
                        refused += 1
                silent = chan.msgs_sent == before and not ck.dead
                x = ck.open_deferred(deferred[role])
                a = ck.open_final(exact[role])               # checks the deferred opening first
                pend = ck.pending
                b = ck.open_final(big[role], mask[role])
                public = ck.last_opened[0]
                return refused, silent, x, a, pend, b, public, ck.checks

            r0, r1, st0, _ = _checkers(secrets.token_bytes(8), s, d0, d1, body)
            for r in (r0, r1):
                check_eq(r[0], 8, "all eight invalid releases are refused with ValueError")
                check(r[1], "a refused release sends nothing and does not abort")
                check_eq(r[2], reconstruct(*deferred)[0], "deferred opening")
                check_eq(r[3], v_exact, "an exact sharing is released as it is")
                check_eq(r[4], 0, "nothing is pending after a release")
                check_eq(r[5], v_big % (1 << k), "a masked release returns v mod 2^k")
                check_eq(r[6], (v_big + (rho << k)) % mod, "the public opening is v + 2^k rho")
                check(r[6] != v_big or rho == 0, "the high bits of the representative are masked")
                check_eq(r[7], 3, "deferred check, check of release 1, check of release 2")
            # rounds: 1 deferred + (4 + 1 + 4) + (1 + 4)
            check_eq(st0["rounds"], 1 + 9 + 5, "rounds of the two releases")
    # several values in one release; one output mask per sharing that is not exact
    k, s = 32, 64
    w = k + s
    d0, d1 = keys(s, "carry")
    delta = d0 + d1
    vals = [RNG.randrange(1 << k) for _ in range(5)] + [RNG.randrange(1 << w) for _ in range(3)]
    flags = [True] * 5 + [False] * 3
    shares = [deal(v, delta, w, exact=f, rng=RNG) for v, f in zip(vals, flags)]
    rhos = [RNG.getrandbits(s) for _ in range(3)]
    masks = [outmask(r, delta, k, s) for r in rhos]

    def body(role, ck):
        return ck.open_final_many([p[role] for p in shares],
                                  [None] * 5 + [m[role] for m in masks]), ck.last_opened

    r0, r1, st0, _ = _checkers(b"many", s, d0, d1, body)
    check_eq(r0[0], [v % (1 << k) for v in vals], "open_final_many returns every value mod 2^k")
    check_eq(r1[0], r0[0], "both parties obtain the same values")
    check_eq(r0[1][5:], [(v + (r << k)) % (1 << w) for v, r in zip(vals[5:], rhos)],
             "masked representatives in the transcript")
    check_eq(st0["rounds"], 5, "one opening exchange and one check for eight values")
    check_eq(st0["bytes_sent"], 8 * 12 + 96 + 64 + 12, "bytes of the batched release")


def test_outmask_hides_high_bits():
    """With a uniform rho the public opening of a non-exact value is uniform in its high s bits."""
    k = s = 4
    w = 8
    runs = 3200
    d0, d1 = keys(s, "carry")
    delta = d0 + d1
    tops = {0: [0] * 16, 1: [0] * 16}
    for world, v in ((0, 0x0A), (1, 0xFA)):                   # same payload, different high parts
        material = [(deal(v, delta, w, rng=RNG), outmask(RNG.getrandbits(s), delta, k, s))
                    for _ in range(runs)]

        def body(role, ck, material=material):
            seen = []
            for sh, mk in material:
                got = ck.open_final(sh[role], mk[role])
                seen.append((got, ck.last_opened[0]))
            return seen

        r0, _, _, _ = _checkers(b"mask%d" % world, s, d0, d1, body)
        check(all(got == 0x0A for got, _ in r0), "every release returns the payload")
        for _, public in r0:
            tops[world][public >> k] += 1
    # critical value of chi-square with 15 degrees of freedom at p = 1e-6 (Wilson-Hilferty)
    critical = 15 * (1 - 2 / 135 + 4.753 * math.sqrt(2 / 135)) ** 3
    stats = []
    for world in (0, 1):
        chi2 = sum((c - runs / 16) ** 2 / (runs / 16) for c in tops[world])
        stats.append(chi2)
        check(chi2 < critical, f"world {world}: high part of the public opening is uniform (chi2 = {chi2:.1f}, df 15)")
    NOTES.append(f"output mask: chi-square of the high s bits of {runs} public openings per world, "
                 f"16 bins: {stats[0]:.1f} and {stats[1]:.1f} (critical value {critical:.1f} at p = 1e-6)")


# --------------------------------------------------------------------------- #
# Lemma 2
# --------------------------------------------------------------------------- #

class Cheater(ArithChecker):
    """Opens ``value + delta_j`` and, optionally, compensates its zeta with a guessed key share."""

    errors: dict = {}
    guess = None          # None = no compensation; otherwise the guessed Delta_h

    def _outgoing_values(self, shares, first_index):
        return [(sh.value + self.errors.get(first_index + j, 0)) % sh.modulus
                for j, sh in enumerate(shares)]

    def _outgoing_zeta(self, bits, zeta, chis, items, counter):
        if self.guess is None:
            return zeta
        e = sum(chi * self.errors.get(index, 0) for chi, (index, _, _) in zip(chis, items))
        return (zeta + (self._delta + self.guess) * e) % (1 << bits)


def exact_pass_probability(s: int, m: int, deltas: list, compensate: bool) -> float:
    """Pass probability of the strategy by enumeration of all keys, coefficients and guesses.

    With compensation the check passes iff ``(g - Delta_h) e = 0 mod 2^m``; without it iff
    ``(Delta_h + Delta_a) e = 0 mod 2^m``; ``e = sum chi_j delta_j``.
    """
    size = 1 << s
    mod = 1 << m
    hits = total = 0
    for chis in itertools.product(range(size), repeat=len(deltas)):
        e = sum(c * d for c, d in zip(chis, deltas)) % mod
        for a in range(size):
            for b in range(size):
                factor = (a - b) if compensate else (a + b)
                hits += (factor * e) % mod == 0
                total += 1
    return hits / total


def lemma2_run(s: int, m: int, deltas: list, compensate: bool, trials: int):
    """``trials`` independent checks with fresh keys.

    Returns (passes, passes that accepted a wrong representative, passes that
    accepted a wrong PAYLOAD, i.e. a value that is wrong modulo ``2^(m-s)``).
    """
    key_pairs = [(RNG.getrandbits(s), RNG.getrandbits(s)) for _ in range(trials)]
    values = [[RNG.randrange(1 << m) for _ in deltas] for _ in range(trials)]
    shares = [[deal(v, d0 + d1, m, rng=RNG) for v in vs] for vs, (d0, d1) in zip(values, key_pairs)]
    sid = secrets.token_bytes(8)
    cheater_role = 1

    def party(role):
        def fn(chan):
            out = []
            for t in range(trials):
                sub = sid + t.to_bytes(4, "little")
                if role == cheater_role:
                    ck = Cheater(role, chan, sub, key_pairs[t][role], s)
                    ck.errors = dict(enumerate(deltas))
                    ck.guess = secrets.randbelow(1 << s) if compensate else None
                else:
                    ck = ArithChecker(role, chan, sub, key_pairs[t][role], s)
                opened = ck.open_deferred_many([sh[role] for sh in shares[t]])
                try:
                    ck.check()
                    out.append((True, opened))
                except ProtocolAbort:
                    out.append((False, opened))
            return out
        return fn

    r0, r1, _, _ = run2(party(0), party(1))
    honest = r0
    passes = sum(1 for ok, _ in honest if ok)
    wrong = sum(1 for (ok, opened), vs in zip(honest, values) if ok and opened != vs)
    low = (1 << (m - s)) - 1
    wrong_payload = sum(1 for (ok, opened), vs in zip(honest, values)
                        if ok and [o & low for o in opened] != [v & low for v in vs])
    return passes, wrong, wrong_payload


def test_lemma2_tiny():
    s = k = 4
    m = k + s
    bound = lemma2_bound(s)
    trials = LEMMA2_TRIALS
    # (name, modulus bits, errors, compensation, inside the scope of Lemma 2)
    strategies = [
        ("delta = 1, no compensation", m, [1], False, True),
        ("delta = 1, guess of Delta_h", m, [1], True, True),
        ("delta = 2^(k-1), guess (best single value)", m, [1 << (k - 1)], True, True),
        ("deltas (1, 2^(k-1)), guess", m, [1, 1 << (k - 1)], True, True),
        ("modulus 2^(n+s), n = 2: delta = 2^(n-1), guess", 2 + s, [1 << 1], True, True),
        ("delta = 2^k (high bits only), guess", m, [1 << k], True, False),
        ("delta = 2^(m-1) (top bit only), no compensation", m, [1 << (m - 1)], False, False),
        ("delta = 2^(m-2), no compensation", m, [1 << (m - 2)], False, False),
        ("delta = 2^(m-1), guess", m, [1 << (m - 1)], True, False),
    ]
    rows = []
    for name, bits, deltas, comp, in_scope in strategies:
        exact = exact_pass_probability(s, bits, deltas, comp)
        t0 = time.perf_counter()
        passes, wrong, wrong_payload = lemma2_run(s, bits, deltas, comp, trials)
        rate = passes / trials
        sd = math.sqrt(exact * (1 - exact) / trials)
        rows.append((name, rate, exact, time.perf_counter() - t0))
        check(abs(rate - exact) < 5 * sd,
              f"{name}: measured pass rate {rate:.4f} against the exact probability {exact:.4f}")
        if in_scope:
            check(exact <= bound, f"{name}: exact probability {exact:.4f} is below the bound {bound:.4f}")
            check(rate <= bound, f"{name}: measured pass rate {rate:.4f} is below the bound {bound:.4f}")
            check_eq(wrong, passes, f"{name}: a passed check accepts the wrong value (the bounded event)")
            check_eq(wrong_payload, passes, f"{name}: the accepted value is wrong modulo 2^(m-s)")
        else:
            # an error that is 0 mod 2^(m-s) is outside Lemma 2 and harmless: the payload is unchanged
            check(all(d % (1 << (bits - s)) == 0 for d in deltas), "errors on the high bits only")
            check_eq((wrong, wrong_payload), (passes, 0),
                     f"{name}: every pass carries the error in the high bits and the payload is never wrong")
            j = bits - deltas[0].bit_length() + 1                 # the error is 2^(m-j)
            check(len(deltas) == 1 and deltas[0] == 1 << (bits - j) and abs(exact - (j + 2) / (1 << (j + 1))) < 1e-12,
                  f"{name}: exact pass probability (j + 2) / 2^(j+1) for the error 2^(m-j), j = {j}")
        print(f"    {name:<50} measured {rate:.4f}  exact {exact:.4f}  bound {bound:.4f}", flush=True)
    best = max(exact for (_, _, exact, _), (*_, in_scope) in zip(rows, strategies) if in_scope)
    # The exact optimum for one wrong value is 2^-s (s + 5) / 4: the coefficient can be 0 mod 2^s.
    check(abs(exact_pass_probability(s, m, [1 << (k - 1)], True) - (s + 5) / 4 / (1 << s)) < 1e-12,
          "closed form 2^-s (s + 5) / 4 for the best single-value strategy")
    check(abs(best - (1 + (s + 1) / 4) / (1 << s)) < 1e-12,
          "the optimum attains 2^-s (1 + (s+1)/4), the constant of the proof of Lemma 2")
    check(best <= bound, "the bound of Lemma 2 holds for every strategy in its scope")
    # errors 2^(m-j) that are 0 mod 2^(m-s): the check passes iff Delta chi = 0 mod 2^j (with a
    # guess g of Delta_h: iff (g - Delta_h) chi = 0 mod 2^j), i.e. with probability (j + 2) / 2^(j+1)
    for j in range(1, s + 1):
        for comp in (False, True):
            check(abs(exact_pass_probability(s, m, [1 << (m - j)], comp) - (j + 2) / (1 << (j + 1))) < 1e-12,
                  f"error 2^(m-{j}), {'guess' if comp else 'no compensation'}: (j + 2) / 2^(j+1) by enumeration")
    NOTES.append(f"Lemma 2 at s = k = 4, {trials} checks per strategy: " + "; ".join(
        f"{name}: {rate:.4f} (exact {exact:.4f})" for name, rate, exact, _ in rows)
        + f"; bound 2^(-s+log2(s+1)) = {bound:.4f}")


def test_lemma2_exact_small_parameters():
    """The bound against the exact optimum for every s <= 5 and every single error (enumeration)."""
    worst = {}
    for s in range(1, 6):
        for k in range(1, 5):
            m = k + s
            best = 0.0
            for delta in range(1, 1 << k):                    # delta != 0 mod 2^(m-s)
                best = max(best, exact_pass_probability(s, m, [delta], True))
            check(best <= lemma2_bound(s) + 1e-12, f"s = {s}, k = {k}: exact optimum {best:.4f} <= bound")
            check(abs(best - (s + 5) / 4 / (1 << s)) < 1e-12 or k == 0,
                  f"s = {s}, k = {k}: the optimum is 2^-s (s + 5) / 4")
            worst[s] = max(worst.get(s, 0.0), best)
    NOTES.append("exact optimum of a single wrong value, by enumeration: " + ", ".join(
        f"s = {s}: {worst[s]:.4f} (bound {lemma2_bound(s):.4f})" for s in sorted(worst)))


# --------------------------------------------------------------------------- #
# s = 64: every deviation is rejected; abort semantics
# --------------------------------------------------------------------------- #

class WrongMac(ArithChecker):
    def _recorded_mac(self, share, index):
        return (share.mac + 1) % share.modulus


class WrongDelta(ArithChecker):
    offset = 1

    def _outgoing_zeta(self, bits, zeta, chis, items, counter):
        y = sum(chi * value for chi, (_, value, _) in zip(chis, items))
        return (zeta - self.offset * y) % (1 << bits)


def _cheat_session(k: int, s: int, cls, setup=None, release: bool = False, values=None):
    """One session with a deviating P_1; returns the honest party's outcome and facts."""
    w = k + s
    d0, d1 = keys(s)
    delta = d0 + d1
    values = [RNG.randrange(1, 1 << k) | 1 for _ in range(3)] if values is None else values
    shares = [deal(v, delta, w, exact=True, rng=RNG) for v in values]
    sid = b"s64" + secrets.token_bytes(4)

    def party(role):
        def fn(chan):
            ck = (ArithChecker if role == 0 else cls)(role, chan, sid, (d0, d1)[role], s)
            if role == 1 and setup is not None:
                setup(ck)
            try:
                if release:
                    out = ck.open_final_many([p[role] for p in shares])
                else:
                    out = ck.open_deferred_many([p[role] for p in shares])
                    ck.check()
                status = "ok"
            except ProtocolAbort as exc:
                out, status = None, str(exc)
            sent = chan.msgs_sent
            later = 0
            for fn2 in (lambda: ck.check(), lambda: ck.open_deferred(shares[0][role]),
                        lambda: ck.open_final(shares[0][role].with_exact(True))):
                try:
                    fn2()
                except ProtocolAbort:
                    later += 1
            return status, out, ck.dead, ck._delta is None, ck.pending, later, chan.msgs_sent == sent
        return fn

    r0, r1, _, _ = run2(party(0), party(1))
    return r0, r1, values


def test_s64_deviations_rejected():
    counts = {}
    for (k, s) in ((64, 64), (32, 64)):
        w = k + s
        cases = [
            ("value + 1", Cheater, lambda ck: setattr(ck, "errors", {0: 1})),
            ("value + 2^(k-1)", Cheater, lambda ck: setattr(ck, "errors", {1: 1 << (k - 1)})),
            ("random error with a key guess", Cheater,
             lambda ck: (setattr(ck, "errors", {2: RNG.randrange(1, 1 << w)}),
                         setattr(ck, "guess", RNG.getrandbits(s)))),
            ("errors on all values with a key guess", Cheater,
             lambda ck: (setattr(ck, "errors", {j: RNG.randrange(1, 1 << k) for j in range(3)}),
                         setattr(ck, "guess", RNG.getrandbits(s)))),
            ("error on the high s bits only", Cheater,
             lambda ck: setattr(ck, "errors", {0: RNG.randrange(1, 1 << s) << k})),
            ("wrong MAC share", WrongMac, None),
            ("wrong key share (+1)", WrongDelta, None),
            ("wrong key share (+2^(s-1))", WrongDelta, lambda ck: setattr(ck, "offset", 1 << (s - 1))),
        ]
        for name, cls, setup in cases:
            for release in (False, True):
                for _ in range(8):
                    r0, r1, _ = _cheat_session(k, s, cls, setup, release)
                    counts[name] = counts.get(name, 0) + 1
                    check_eq(r0[0], "ArithCheck failed", f"k = {k}, {name}: the honest party aborts")
                    check(r0[1] is None, f"{name}: nothing is returned")
                    check(r0[2] and r0[3] and r0[4] == 0, f"{name}: the checker is dead, its key share dropped")
                    check_eq(r0[5], 3, f"{name}: every later call raises ProtocolAbort")
                    check(r0[6], f"{name}: a dead checker sends nothing")
    # controls: the same harness with honest code passes; a zero error passes
    for release in (False, True):
        r0, r1, values = _cheat_session(64, 64, ArithChecker, None, release, values=[5, 6, 7])
        check_eq(r0[0], "ok", "control: an honest peer passes")
        check_eq(r0[1], values, "control: opened values")
        r0, _, _ = _cheat_session(64, 64, Cheater, lambda ck: setattr(ck, "errors", {0: 0}), release,
                                  values=[5, 6, 7])
        check_eq(r0[0], "ok", "control: a zero error passes")
    # a wrong key share goes unnoticed only if every opened value is zero (y = 0): then nothing was forged
    r0, _, _ = _cheat_session(64, 64, WrongDelta, None, False, values=[0, 0, 0])
    check_eq(r0[0], "ok", "wrong key share with all-zero openings: y = 0, the check is satisfied")
    NOTES.append("s = 64: " + ", ".join(f"{name}: {n}/{n} rejected" for name, n in counts.items()))


def test_release_checks_deferred_first():
    """A wrong deferred opening is caught BEFORE the release exchange (the honest share is not sent)."""
    k = s = 64
    w = k + s
    for _ in range(10):
        d0, d1 = keys(s)
        delta = d0 + d1
        masked = deal(RNG.randrange(1 << (4 + s)), delta, 4 + s, rng=RNG)
        secret = deal(RNG.randrange(1 << k), delta, w, exact=True, rng=RNG)

        def party(role):
            def fn(chan):
                cls = ArithChecker if role == 0 else Cheater
                ck = cls(role, chan, b"deferred", (d0, d1)[role], s)
                if role == 1:
                    ck.errors = {0: 1}
                ck.open_deferred(masked[role])
                try:
                    return "ok", ck.open_final(secret[role]), chan.stats()["phases"]
                except ProtocolAbort as exc:
                    return str(exc), None, chan.stats()["phases"]
            return fn

        r0, _, _, _ = run2(party(0), party(1))
        check_eq(r0[0], "ArithCheck failed", "the deferred opening is rejected")
        check_eq(r0[2]["arith-open"]["msgs_sent"], 1, "the honest value share of the release was never sent")
        check_eq(r0[2]["arith-open"]["bytes_sent"], share_bytes(4 + s), "only the masked opening left the party")


# --------------------------------------------------------------------------- #
# Message-level deviations
# --------------------------------------------------------------------------- #

class _Tamper:
    """Minimal channel wrapper for the peer: ``hook(index, message) -> message`` on its exchanges."""

    def __init__(self, inner, hook):
        self._inner = inner
        self._hook = hook
        self.count = 0
        self.log = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def exchange(self, data):
        out = self._hook(self, self.count, bytes(data))
        self.count += 1
        if out is None:                                     # mirror: answer with the peer's own message
            incoming = self._inner.recv()
            self._inner.send(incoming)
            return incoming
        self.log.append(out)
        return self._inner.exchange(out)


def _message_session(hook, bits_list=(128,), s=64, sid1=b"msg", honest_body=None):
    """Honest P_0 against a P_1 whose outgoing messages pass through ``hook``."""
    d0, d1 = keys(s)
    delta = d0 + d1
    shares = [deal(RNG.randrange(1 << m), delta, m, rng=RNG) for m in bits_list]
    shares2 = [deal(RNG.randrange(1 << m), delta, m, rng=RNG) for m in bits_list]

    def default_body(ck, role):
        ck.open_deferred_many([p[role] for p in shares])
        ck.check()
        ck.open_deferred_many([p[role] for p in shares2])
        ck.check()
        return "ok"

    body = honest_body or default_body

    def party(role):
        def fn(chan):
            if role == 1:
                chan = _Tamper(chan, hook)
            ck = ArithChecker(role, chan, sid1, (d0, d1)[role], s)
            try:
                return body(ck, role), ck.dead
            except ProtocolAbort as exc:
                return str(exc), ck.dead
        return fn

    r0, r1, _, _ = run2(party(0), party(1))
    return r0, r1


def test_message_deviations():
    def at(index, fn):
        return lambda tc, i, msg: fn(msg) if i == index else msg

    # exchanges of the default body: 0 values, 1 coin commit, 2 coin open, 3 zeta commit, 4 zeta open,
    # then 5 values, 6..9 the second check
    cases = [
        ("value share one byte short", at(0, lambda m: m[:-1]), "opening: message has the wrong length"),
        ("value share one byte long", at(0, lambda m: m + b"\x00"), "opening: message has the wrong length"),
        ("empty value message", at(0, lambda m: b""), "opening: message has the wrong length"),
        ("coin commitment truncated", at(1, lambda m: m[:-1]), "malformed commitment"),
        ("coin opening changed", at(2, lambda m: m[:-1] + bytes([m[-1] ^ 1])), "commitment opening failed"),
        ("zeta commitment changed", at(3, lambda m: bytes([m[0] ^ 1]) + m[1:]), "commitment opening failed"),
        ("zeta changed after the commitment", at(4, lambda m: m[:-1] + bytes([m[-1] ^ 1])),
         "commitment opening failed"),
        ("zeta opening truncated", at(4, lambda m: m[:20]), "malformed opening"),
        ("mirrored coin commitment", at(1, lambda m: None), "mirrored commitment"),
        ("mirrored zeta commitment", at(3, lambda m: None), "mirrored commitment"),
    ]
    for name, hook, want in cases:
        r0, _ = _message_session(hook)
        check_eq(r0[0], want, f"{name}: the honest party aborts")
        check(r0[1], f"{name}: the checker is dead")

    # value >= 2^m in a modulus that does not fill its bytes (68 bits in 9 bytes)
    r0, _ = _message_session(at(0, lambda m: m[:8] + bytes([m[8] | 0x80])), bits_list=(68,))
    check_eq(r0[0], "opening: value share out of range", "a value share above the modulus is rejected")

    # replay of the four messages of check 0 in check 1 (same session): counter binding
    def replay_check(tc, i, msg):
        return tc.log[i - 5] if 6 <= i <= 9 else msg
    r0, _ = _message_session(replay_check)
    check_eq(r0[0], "commitment opening failed", "replayed check messages are rejected (counter in the label)")

    # replay of only the zeta commitment and opening of check 0 in check 1 (the coin toss of
    # check 1 is run honestly): rejected as a commitment, because its label carries the counter
    r0, _ = _message_session(lambda tc, i, msg: tc.log[i - 5] if i in (8, 9) else msg)
    check_eq(r0[0], "commitment opening failed", "a zeta commitment of an earlier check is rejected as such")

    # replay of an old value share: the opening differs, so the transcripts differ and the
    # coin toss of the check is already rejected (the label contains the list of openings)
    r0, _ = _message_session(lambda tc, i, msg: tc.log[0] if i == 5 else msg)
    check_eq(r0[0], "commitment opening failed", "a replayed value share desynchronises the check label")

    # a value share changed in transit without the sender updating its view: same effect
    r0, _ = _message_session(at(0, lambda m: bytes([m[0] ^ 1]) + m[1:]))
    check_eq(r0[0], "commitment opening failed", "views that differ in an opened value never reach the MAC comparison")

    # messages of another session (other sid): the recorded transcript of P_1 is replayed
    recorded = {}

    def record(tc, i, msg):
        recorded[i] = msg
        return msg
    r0, _ = _message_session(record, sid1=b"session-A")
    check_eq(r0[0], "ok", "recording run")
    r0, _ = _message_session(lambda tc, i, msg: recorded[i] if i >= 1 else msg, sid1=b"session-B")
    check_eq(r0[0], "commitment opening failed", "check messages of another session are rejected")

    # an out-of-range zeta that is properly committed to
    class BigZeta(ArithChecker):
        def _outgoing_zeta(self, bits, zeta, chis, items, counter):
            return zeta | (1 << (bits + 1))

    d0, d1 = keys(64)
    sh = deal(77, d0 + d1, 68, rng=RNG)

    def party(role):
        def fn(chan):
            ck = (ArithChecker, BigZeta)[role](role, chan, b"big", (d0, d1)[role], 64)
            ck.open_deferred(sh[role])
            try:
                ck.check()
                return "ok"
            except ProtocolAbort as exc:
                return str(exc)
        return fn
    r0, _, _, _ = run2(party(0), party(1))
    check_eq(r0, "ArithCheck: value out of range", "a committed zeta above the modulus is rejected")

    # the commitments are bound to the committer: the two parties' commitments of one honest check differ
    com0, _ = hs.commit(b"sid", b"m", party=0, label=b"l")
    com1, _ = hs.commit(b"sid", b"m", party=1, label=b"l")
    check(com0 != com1, "commitments carry the committer's index")


def test_counters_never_reused():
    """Openings and checks are numbered session-wide; labels of different checks differ."""
    s = 64
    d0, d1 = keys(s)
    delta = d0 + d1
    shares = [deal(RNG.randrange(1 << 128), delta, 128, rng=RNG) for _ in range(6)]
    seen = {0: [], 1: []}

    def party(role):
        def fn(chan):
            ck = ArithChecker(role, chan, b"ctr", (d0, d1)[role], s)
            for i in range(0, 6, 2):
                ck.open_deferred(shares[i][role])
                ck.open_deferred(shares[i + 1][role])
                seen[role].append([entry[1] for entry in ck._pending])
                ck.check()
            return ck.openings, ck.checks
        return fn

    r0, r1, _, _ = run2(party(0), party(1))
    check_eq(r0, (6, 3), "six openings, three checks")
    check_eq(seen[0], [[0, 1], [2, 3], [4, 5]], "opening indices are consecutive and never reused")
    check_eq(seen[0], seen[1], "both parties number the openings alike")


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    if "-k" in sys.argv:
        key = sys.argv[sys.argv.index("-k") + 1]
        tests = [(name, fn) for name, fn in tests if key in name]
    failed = []
    start = time.perf_counter()
    print(f"SPDZ2k sharing / ArithCheck tests, {len(tests)} test functions "
          f"(Lemma 2: {LEMMA2_TRIALS} checks per strategy)")
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
