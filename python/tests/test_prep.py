"""Tests for the Boolean preprocessing (couplet.prep; DESIGN.md 2.2, 7.1 - 7.5, 7.7).

(a) params      bucket sizes of Lemma 4: exact thresholds, minimality, a
                brute-force evaluation of the bound, the parameter table;
                ``triple_request``: every run has at least 306 leaky triples
                (known answers shared with cpp/src/test_prep.cpp).
    shares      the data model: XOR, slicing, selection, concatenation,
                constants, wire encoding.
(b) base OT     Masny-Rindal, both directions, 20 runs; invalid points abort.
(c) COT         q_i = t_i XOR u_i Delta for all rows; an inconsistent column
                of U is rejected exactly when the sender reads that column.
(d) abits       MAC invariants of 10^5 random bits, BDOZ invariants, input masks.
(e) leaky AND   z = x AND y and the MAC invariant for 10^5 triples; the
                transfer split into several messages.
(f) Lemma 3     additive error on the transfer message of t triples: the
                consistency check passes iff every guess is right.
    bucketing   the public permutation and the bucket combination.
(g) RealPrep    and_triples end to end, n in {1000, 50000}: correctness, MAC
                invariants, bucket size, exact byte accounting per phase;
                small requests are padded to at least 306 leaky triples and
                the whole padded run is checked.
(h) IdealPrep   the same invariants for the ideal dealer.
(i) BoolCheck   honest openings pass; a tampered opened bit, MAC share or
                field opening aborts; a failed check retires the session.

Two parties are run with ``couplet.net.run_two_party`` (two threads of this
process), so the timings printed at the end are wall-clock times for BOTH
parties together on one machine.

Run as a script from artifact/python:   python tests/test_prep.py
Prints the number of checks and exits non-zero on any failure.  Options:
``--no-large`` (or COUPLET_NO_LARGE=1) skips the n = 300000 timing run.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import ed25519 as ed  # noqa: E402
from couplet import gf128 as gf  # noqa: E402
from couplet import hashes as hs  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import loopback_pair, run_two_party  # noqa: E402
from couplet.prep import aand, params  # noqa: E402
from couplet.prep import IdealDealer, IdealPrep, RealPrep, bool_check  # noqa: E402
from couplet.prep.abits import AbitSource, check_delta, sample_delta  # noqa: E402
from couplet.prep import baseot  # noqa: E402
from couplet.prep.baseot import base_ot  # noqa: E402
from couplet.prep.check import check_digest, check_values  # noqa: E402
from couplet.prep.cotext import (CotReceiver, CotSender, extend_both, rounded_rows,  # noqa: E402
                                  u_message_bytes)
from couplet.prep.shares import (Bdoz, Keys, Rand, Shares, exchange_bits, mask_other,  # noqa: E402
                                  mask_owner, pack_bit_msg, spdz_shares, unpack_bit_msg)

CHECKS = 0
FAILURES: list[str] = []
SID = b"test-prep-session"
KAPPA = params.KAPPA
PAD = params.PAD
COT_CHECK_BYTES = 16 * (KAPPA + 1)
COIN_BYTES = 96        # per party: 32-byte commitment, 32-byte opening, 32-byte share
BOOLCHECK_BYTES = 96   # per party: 32-byte commitment, 32-byte opening, 32-byte digest


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
    except Exception:  # a different exception type is a failure of the check
        return False
    return False


def _finish(start_failures: int) -> None:
    """Make the enclosing test function fail under pytest if it recorded a failure."""
    assert len(FAILURES) == start_failures, FAILURES[start_failures:]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def rand_vec(rng: random.Random, n: int) -> np.ndarray:
    return gf.bytes_to_vec(rng.randbytes(16 * n))


def rand_bits(rng: random.Random, n: int) -> np.ndarray:
    return np.frombuffer(rng.randbytes(n), dtype=np.uint8) & 1


def test_deltas(rng: random.Random) -> tuple[int, int]:
    """Key shares with lsb(Delta_0) = 1 and lsb(Delta_1) = 0."""
    return rng.getrandbits(128) | 1, rng.getrandbits(128) & ~1


test_deltas.__test__ = False  # a helper, not a pytest test


def trusted_seeds(rng: random.Random, delta: int):
    """Base-OT output sampled by the test: (chosen seeds, (seeds0, seeds1)) for key ``delta``."""
    s0, s1 = rand_vec(rng, KAPPA), rand_vec(rng, KAPPA)
    sel = gf.unpack_bits(delta).astype(bool)[:, None]
    return np.where(sel, s1, s0), (s0, s1)


def mac_ok(s0: Shares, s1: Shares, delta: int, value=None) -> bool:
    """m_0 XOR m_1 = x Delta for all positions (and x = value if given)."""
    x = s0.bits ^ s1.bits
    if value is not None and not np.array_equal(x, value):
        return False
    return np.array_equal(s0.macs ^ s1.macs, gf.bit_mul(x, delta))


def bdoz_ok(own: Bdoz, keys: Keys, key_delta: int) -> bool:
    """M[x] = K[x] XOR x Delta for all positions."""
    return np.array_equal(own.macs, keys.keys ^ gf.bit_mul(own.bits, key_delta))


def abit_session(rng: random.Random, body, batch_rows: int = 1 << 20, **kwargs):
    """Run ``body(role, chan, source, delta)`` for both parties on COT states with trusted seeds.

    Returns (result_0, result_1, stats_0, stats_1, (Delta_0, Delta_1)).
    """
    deltas = test_deltas(rng)
    seeds = [trusted_seeds(rng, deltas[d]) for d in (0, 1)]  # direction d: sender P_d

    def party(role: int):
        def run(chan):
            sender = CotSender(chan, SID, role, deltas[role], seeds[role][0])
            receiver = CotReceiver(chan, SID, 1 - role, *seeds[1 - role][1])
            source = AbitSource(chan, role, deltas[role], sender, receiver, batch_rows)
            return body(role, chan, source, deltas[role])
        return run

    r0, r1, st0, st1 = run_two_party(party(0), party(1), **kwargs)
    return r0, r1, st0, st1, deltas


class Tamper:
    """Channel proxy of a test adversary: rewrites selected outgoing messages.

    ``rules`` maps the index of an outgoing message (0 = first message this
    endpoint sends, counting ``send`` and ``exchange``) to a function
    bytes -> bytes.  Everything else is delegated to the real endpoint.
    """

    def __init__(self, chan, rules: dict) -> None:
        self._chan = chan
        self._rules = rules
        self._sent = 0

    def _out(self, data: bytes) -> bytes:
        fn = self._rules.get(self._sent)
        self._sent += 1
        return data if fn is None else fn(bytes(data))

    def send(self, data: bytes) -> None:
        self._chan.send(self._out(data))

    def exchange(self, data: bytes) -> bytes:
        return self._chan.exchange(self._out(data))

    def __getattr__(self, name):
        return getattr(self._chan, name)


def flip_bit(position: int):
    """Message rewriter: flip one bit of the message."""
    def fn(data: bytes) -> bytes:
        out = bytearray(data)
        out[position // 8] ^= 1 << (position % 8)
        return bytes(out)
    return fn


def guarded(fn):
    """Wrap a party function: a ProtocolAbort becomes the return value ("abort", message)."""
    def run(chan):
        try:
            return fn(chan)
        except ProtocolAbort as exc:
            return ("abort", str(exc))
    return run


def aborted(result) -> bool:
    return isinstance(result, tuple) and len(result) == 2 and result[0] == "abort"


# --------------------------------------------------------------------------- #
# (a) Parameters and data model
# --------------------------------------------------------------------------- #

def _bound_brute(ell: int, bucket: int) -> Fraction:
    """Lemma 4 bound evaluated over ALL t in B..ell*B (no early stop)."""
    total = ell * bucket
    return max(Fraction(ell * math.comb(t, bucket), math.comb(total, bucket) << t)
               for t in range(bucket, total + 1))


def test_params() -> None:
    start = len(FAILURES)
    check(params.KAPPA == 128 and params.PAD == 128, "KAPPA = PAD = 128")

    # Values quoted in DESIGN.md 7.5 (sigma = 40).
    check(params.bucket_size(280_000, 40) == 3, "sigma=40: B = 3 at ell = 280,000")
    check(params.bucket_size(3_100, 40) == 4, "sigma=40: B = 4 at ell = 3,100")
    check(params.bucket_size(3_000, 40) == 5, "sigma=40: B = 5 at ell = 3,000")
    t3, t4, t5 = (params.min_ell(b, 40) for b in (3, 4, 5))
    print(f"  exact thresholds, sigma=40: B=3 from ell={t3}, B=4 from ell={t4}, B=5 from ell={t5}")
    print(f"  relative to DESIGN.md: {t3 / 280_000 - 1:+.2%} (280,000), {t4 / 3_100 - 1:+.2%} (3,100)")
    check(t3 == 276_325, "sigma=40: exact threshold of B = 3 is 276,325")
    check(t4 == 3_044, "sigma=40: exact threshold of B = 4 is 3,044")
    check(t3 <= 280_000 and abs(t3 / 280_000 - 1) < 0.02,
          "B = 3 threshold is at most, and within 2% of, the quoted 280,000")
    check(t4 <= 3_100 and abs(t4 / 3_100 - 1) < 0.02,
          "B = 4 threshold is at most, and within 2% of, the quoted 3,100")
    for b, thr in ((3, t3), (4, t4), (5, t5)):
        check(params.bucket_size(thr, 40) == b and params.bucket_size(thr - 1, 40) == b + 1,
              f"sigma=40: bucket size switches to {b} exactly at ell = {thr}")

    # The scan with early stop equals the maximum over all t; closed form of the maximiser.
    for ell in (1, 2, 3, 5, 8, 13, 40):
        for bucket in (1, 2, 3, 4, 5, 7):
            check(params.leak_bound(ell, bucket) == _bound_brute(ell, bucket),
                  f"leak_bound({ell}, {bucket}) equals the brute-force maximum")
    for ell, bucket in ((1000, 3), (50_000, 4), (1 << 20, 3)):
        t = 2 * bucket - 1
        closed = Fraction(ell * math.comb(t, bucket), math.comb(ell * bucket, bucket) << t)
        check(params.leak_bound(ell, bucket) == closed,
              f"leak_bound({ell}, {bucket}) is attained at t = 2B - 1")

    # Minimality and monotonicity.
    for sigma in (40, 64):
        target = Fraction(1, 1 << sigma)
        prev = None
        for ell in (1, 2, 7, 100, 1 << 10, 5000, 1 << 16, 1 << 20, 1 << 24):
            b = params.bucket_size(ell, sigma)
            check(params.leak_bound(ell, b) <= target, f"sigma={sigma}, ell={ell}: bound holds")
            check(b == 2 or params.leak_bound(ell, b - 1) > target,
                  f"sigma={sigma}, ell={ell}: B - 1 is not enough")
            check(prev is None or b <= prev, f"sigma={sigma}: B is non-increasing in ell")
            prev = b
    check(params.bucket_size(1, 40) == 40 and params.bucket_size(1, 64) == 64,
          "a single bucket needs B = sigma")
    check(raises(ValueError, params.bucket_size, 0, 40), "bucket_size rejects ell = 0")
    check(raises(ValueError, params.min_ell, 1, 40), "min_ell rejects B = 1")

    # Parameter table (written to experiments/figdata and read back).
    path = params.write_table()
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)
    check(os.path.basename(path) == "bucket_params.json"
          and os.path.basename(os.path.dirname(path)) == "figdata", "table path")
    check(len(table["rows"]) == 30, "table has 2 sigmas x 15 sizes")
    check(all(row["B"] == params.bucket_size(row["ell"], row["sigma"]) for row in table["rows"]),
          "table rows agree with bucket_size")
    check(all(row["log2_leak_bound"] <= -row["sigma"] for row in table["rows"]),
          "every table row meets its statistical level")
    check({(r["sigma"], r["log2_ell"]) for r in table["rows"]}
          == {(s, e) for s in (40, 64) for e in range(10, 25)}, "table covers 2^10 .. 2^24")
    check(table["min_ell_for_bucket"]["40"]["3"] == t3
          and table["min_ell_for_bucket"]["40"]["4"] == t4, "table thresholds")
    by_key = {(r["sigma"], r["log2_ell"]): r["B"] for r in table["rows"]}
    print("  sigma=40:", " ".join(f"2^{e}:{by_key[(40, e)]}" for e in range(10, 25)))
    print("  sigma=64:", " ".join(f"2^{e}:{by_key[(64, e)]}" for e in range(10, 25)))

    # Domain constants: distinct and disjoint from the FSS label words.
    doms = {params.prg_dom(use, d, ctr)
            for use in (params.USE_COT_PRG, params.USE_COT_CHI, params.USE_BUCKET_PERM,
                        params.USE_TRIPLE_CHI)
            for d in (0, 1) for ctr in (0, 1, (1 << 40) - 1)}
    check(len(doms) == 24, "prg_dom is injective on (use, direction, counter)")
    check(all(d >> 56 == 0x50 and d & ((1 << 56) - 1) >= 0x100 for d in doms),
          "PRG domains: family byte 0x50, identifiers above the reserved range")
    ccr = {aand.h_dom(sid, d) for sid in (b"", b"a", b"b") for d in (0, 1)}
    check(len(ccr) == 6 and all(d >> 56 == 0x48 for d in ccr),
          "hash domains: family byte 0x48, one per (sid, direction)")
    check(not ((doms | ccr) & hs.RESERVED_DOMS), "domains avoid the FSS label words")
    manual = hs.sha256(b"Couplet-dom", b"aand/1", b"a")[:8]
    check(aand.h_dom(b"a", 1) == (0x48 << 56) | (int.from_bytes(manual, "little") & ((1 << 56) - 1)),
          "session_dom = kind || low 56 bits of SHA-256(frame('Couplet-dom', label, sid))")
    check(raises(ValueError, params.prg_dom, 0) and raises(ValueError, params.prg_dom, 1, 2)
          and raises(ValueError, params.prg_dom, 1, 0, 1 << 40)
          and raises(ValueError, params.session_dom, 0, b"x", b"y"),
          "domain helpers validate their fields")
    check(aand.h_dom(b"a", 0) != aand.h_dom(b"a", 1) and aand.h_dom(b"a", 0) != aand.h_dom(b"b", 0),
          "the leaky-AND hash domain depends on the direction and on sid")
    _finish(start)


def test_shares_model() -> None:
    start = len(FAILURES)
    rng = random.Random(0x5A)
    d0, d1 = test_deltas(rng)
    delta = d0 ^ d1
    n = 257

    def fresh(values):
        bits0 = rand_bits(rng, len(values))
        macs0 = rand_vec(rng, len(values))
        return Shares(bits0, macs0), Shares(bits0 ^ values, macs0 ^ gf.bit_mul(values, delta))

    x, y = rand_bits(rng, n), rand_bits(rng, n)
    x0, x1 = fresh(x)
    y0, y1 = fresh(y)
    check(len(x0) == n and mac_ok(x0, x1, delta, x), "fresh sharing satisfies the invariant")
    check(mac_ok(x0 ^ y0, x1 ^ y1, delta, x ^ y), "XOR is local")
    check(mac_ok(x0[10:50], x1[10:50], delta, x[10:50]), "slicing")
    check(len(x0[7]) == 1 and mac_ok(x0[7], x1[7], delta, x[7:8]), "integer index gives length 1")
    check(mac_ok(x0[-1], x1[-1], delta, x[-1:]), "negative integer index")
    idx = np.array([5, 5, 0, 256, 31])
    check(mac_ok(x0.take(idx), x1.take(idx), delta, x[idx]), "take (with repetitions)")
    check(mac_ok(x0[idx], x1[idx], delta, x[idx]), "index-array selection")
    mask = x.astype(bool)
    check(mac_ok(x0[mask], x1[mask], delta, x[mask]), "boolean-mask selection")
    cat0, cat1 = Shares.concat([x0, y0[:3], x0[:0]]), Shares.concat([x1, y1[:3], x1[:0]])
    check(len(cat0) == n + 3 and mac_ok(cat0, cat1, delta, np.concatenate([x, y[:3]])), "concat")
    check(len(Shares.concat([])) == 0 and len(Shares.zeros(4)) == 4, "empty concat and zeros")
    check(mac_ok(Shares.zeros(9), Shares.zeros(9), delta, np.zeros(9, dtype=np.uint8)),
          "zero shares are valid")

    c = rand_bits(rng, n)
    c0, c1 = Shares.constant(0, d0, c), Shares.constant(1, d1, c)
    check(mac_ok(c0, c1, delta, c), "constant: (c, c Delta_0) and (0, c Delta_1)")
    check(np.array_equal(c0.bits, c) and not c1.bits.any(), "constant: P_0 holds the bit, P_1 zero")
    check(mac_ok(x0.add_const(0, d0, c), x1.add_const(1, d1, c), delta, x ^ c), "add_const")
    check(mac_ok(x0.invert(0, d0), x1.invert(1, d1), delta, x ^ 1), "invert")
    check(mac_ok(x0.mul_const(c), x1.mul_const(c), delta, x & c), "mul_const")
    cp = x0.copy()
    cp.bits[0] ^= 1
    check(cp.bits[0] != x0.bits[0], "copy is independent")

    r = rand_bits(rng, 128)
    r0, r1 = fresh(r)
    (v0, m0), (v1, m1) = r0.to_field(), r1.to_field()
    check(v0 ^ v1 == gf.pack_bits(r), "to_field: value shares XOR to sum_j X^j r_j")
    check(m0 ^ m1 == gf.mul(gf.pack_bits(r), delta), "to_field: MAC shares XOR to R Delta")
    check(raises(ValueError, x0.to_field), "to_field needs exactly 128 bits")

    check(raises(ValueError, Shares, np.array([0, 2], dtype=np.uint8), gf.zeros(2)),
          "Shares rejects non-bits")
    check(raises(ValueError, Shares, np.zeros(3, dtype=np.uint8), gf.zeros(2)),
          "Shares rejects mismatched lengths")
    check(raises(ValueError, lambda: x0 ^ y0[:5]), "XOR rejects mismatched lengths")
    check(raises(ValueError, Shares.constant, 2, d0, c), "constant rejects a bad role")
    check(raises(ValueError, x0.add_const, 0, d0, c[:5]), "add_const rejects a bad length")

    # BDOZ containers and conversions (section 2.2).
    xb0, xb1 = rand_bits(rng, n), rand_bits(rng, n)
    k0, k1 = rand_vec(rng, n), rand_vec(rng, n)          # k_b: keys of P_b for the peer's bits
    own0 = Bdoz(xb0, k1 ^ gf.bit_mul(xb0, d1))
    own1 = Bdoz(xb1, k0 ^ gf.bit_mul(xb1, d0))
    s0 = spdz_shares(0, d0, own0, Keys(k0))
    s1 = spdz_shares(1, d1, own1, Keys(k1))
    check(mac_ok(s0, s1, delta, xb0 ^ xb1), "spdz_shares: m_b = M[x_b] ^ K[x_1-b] ^ x_b Delta_b")
    check(mac_ok(mask_owner(d0, own0), mask_other(Keys(k1)), delta, xb0), "input mask of P_0")
    check(mac_ok(mask_other(Keys(k0)), mask_owner(d1, own1), delta, xb1), "input mask of P_1")
    check(len(own0[3:9]) == 6 and len(Keys(k0)[3]) == 1 and len(Bdoz.concat([own0, own1])) == 2 * n
          and len(Keys.concat([Keys(k0), Keys(k1)[:1]])) == n + 1, "Bdoz / Keys slicing and concat")
    check(bdoz_ok(own0[5:60], Keys(k1)[5:60], d1), "BDOZ slices stay consistent")

    # Wire encoding of bit vectors.
    for m in (0, 1, 7, 8, 9, 257):
        bits = rand_bits(rng, m)
        check(np.array_equal(unpack_bit_msg(pack_bit_msg(bits), m), bits), f"bit message, n = {m}")
    check(raises(ProtocolAbort, unpack_bit_msg, b"\x00\x00", 9 + 8), "short bit message rejected")
    check(raises(ProtocolAbort, unpack_bit_msg, b"\xff\x03", 9), "stray padding bits rejected")
    check(np.array_equal(unpack_bit_msg(b"\xff\x01", 9), np.ones(9, dtype=np.uint8)),
          "full bit message accepted")

    # Rand: wrapper around a randbytes source.
    ra, rb = Rand(random.Random(7)), Rand(random.Random(7))
    check(ra.bytes(5) == rb.bytes(5) and np.array_equal(ra.bits(77), rb.bits(77))
          and np.array_equal(ra.vec(3), rb.vec(3)) and ra.elem() == rb.elem()
          and ra.below(ed.Q) == rb.below(ed.Q), "Rand is deterministic for a seeded source")
    check(ra.bits(1001).shape == (1001,) and int(ra.bits(1001).max()) <= 1
          and ra.words(9).dtype == np.uint64 and 0 <= ra.below(10) < 10, "Rand output shapes")
    check(sample_delta(0, ra) & 1 == 1 and sample_delta(1, ra) & 1 == 0, "sample_delta fixes the lsb")
    check(raises(ValueError, check_delta, 0, 2) and raises(ValueError, check_delta, 1, 3)
          and raises(ValueError, check_delta, 0, 1 << 128) and check_delta(1, 6) == 6,
          "check_delta enforces lsb(Delta_0) = 1, lsb(Delta_1) = 0")
    _finish(start)


# --------------------------------------------------------------------------- #
# (b) Base OT
# --------------------------------------------------------------------------- #

def _base_ot_run(deltas, cheat0=None, cheat1=None, rngs=(None, None)):
    def party(role, cheat):
        def run(chan):
            return base_ot(chan, SID, role, gf.unpack_bits(deltas[role]), rng=rngs[role],
                           cheat=cheat)
        return run
    return run_two_party(party(0, cheat0), party(1, cheat1))


def test_base_ot() -> None:
    start = len(FAILURES)
    rng = random.Random(0xB07)
    runs = 20
    t_begin = time.perf_counter()
    for run in range(runs):
        deltas = (rng.getrandbits(128), rng.getrandbits(128))
        r0, r1, st0, st1 = _base_ot_run(deltas)
        res = (r0, r1)
        for d in (0, 1):  # direction d: P_d holds the chosen seeds, P_{1-d} the pairs
            chosen = res[d][0]
            s0, s1 = res[1 - d][1]
            sel = gf.unpack_bits(deltas[d]).astype(bool)[:, None]
            check(chosen.shape == (KAPPA, 2) and s0.shape == (KAPPA, 2) and s1.shape == (KAPPA, 2),
                  f"base OT run {run}, direction {d}: shapes")
            check(np.array_equal(chosen, np.where(sel, s1, s0)),
                  f"base OT run {run}, direction {d}: the receiver obtains the chosen seeds")
            check(not (s0 == s1).all(axis=1).any(),
                  f"base OT run {run}, direction {d}: the two seeds of an instance differ")
            check(not (chosen == np.where(sel, s0, s1)).all(axis=1).any(),
                  f"base OT run {run}, direction {d}: the other seed is not the chosen one")
            check(len({bytes(row) for row in np.concatenate([s0, s1])}) == 2 * KAPPA,
                  f"base OT run {run}, direction {d}: all 256 seeds are distinct")
        check(not np.array_equal(r0[1][0], r1[1][0]), f"base OT run {run}: directions are independent")
        check(st0["rounds"] == 2 and st1["rounds"] == 2, f"base OT run {run}: two rounds")
        check(st0["bytes_sent"] == 96 * KAPPA and st1["bytes_sent"] == 96 * KAPPA,
              f"base OT run {run}: 96 bytes per instance and party")
    elapsed = time.perf_counter() - t_begin
    print(f"  base OT: {runs} runs, {elapsed / runs:.2f} s per run "
          f"(2 x {KAPPA} instances, both parties in one process)")

    # Determinism under a seeded source, and sid separation.
    deltas = (rng.getrandbits(128), rng.getrandbits(128))
    a = _base_ot_run(deltas, rngs=(random.Random(1), random.Random(2)))
    b = _base_ot_run(deltas, rngs=(random.Random(1), random.Random(2)))
    check(np.array_equal(a[0][0], b[0][0]) and np.array_equal(a[1][1][1], b[1][1][1]),
          "base OT is reproducible with seeded randomness")

    # Deviations: every received point must be a non-identity element of the subgroup.
    identity = ed.compress(ed.IDENTITY)
    order2 = ((ed.P - 1) | 0).to_bytes(32, "little")          # (0, -1)
    order4 = (0).to_bytes(32, "little")                        # (sqrt(-1), 0) up to sign
    not_on_curve = next(c for c in ((2 + i).to_bytes(32, "little") for i in range(50))
                        if raises(ValueError, ed.decompress, c))
    mixed = ed.compress(ed.add(ed.base_mult(12345), ed.decompress(order2)))  # order 2q
    check(ed.is_on_curve(ed.decompress(order2)) and ed.is_on_curve(ed.decompress(order4))
          and not ed.in_subgroup(ed.decompress(mixed)), "test points are as intended")

    def put(offset: int, blob: bytes):
        return lambda msg: msg[:offset] + blob + msg[offset + len(blob):]

    bad_points = (("identity", identity), ("order-2 point", order2), ("order-4 point", order4),
                  ("non-point", not_on_curve), ("point outside the subgroup", mixed))
    for name, blob in bad_points:
        check(raises(ProtocolAbort, baseot._decode, blob), f"base OT decoding rejects a {name}")
    check(ed.equal(baseot._decode(ed.compress(ed.base_mult(5))), ed.base_mult(5)),
          "base OT decoding accepts a subgroup element")
    for (name, blob), (flight, offset) in zip(
            (bad_points[0], bad_points[4], bad_points[3], bad_points[2]),
            (("flight1", 64 * 5), ("flight1", 64 * 9 + 32), ("flight2", 32 * 7), ("flight2", 32 * 127))):
        got = raises(ProtocolAbort, _base_ot_run, deltas, {flight: put(offset, blob)})
        check(got, f"base OT aborts on a {name} in {flight}")
    check(raises(ProtocolAbort, _base_ot_run, deltas, {"flight1": lambda m: m[:-1]}),
          "base OT aborts on a short first flight")
    check(raises(ProtocolAbort, _base_ot_run, deltas, None, {"flight2": lambda m: m + bytes(1)}),
          "base OT aborts on a long second flight")

    # A sender that replaces A by another valid point only desynchronises the seeds
    # (no abort here); the OT-extension check catches it (see test_real_prep_aborts).
    other = ed.compress(ed.base_mult(777))
    r0, r1, _, _ = _base_ot_run(deltas, {"flight2": put(32 * 3, other)})
    sel = gf.unpack_bits(deltas[1]).astype(bool)[:, None]      # direction 1: P_0 sends
    agree = (r1[0] == np.where(sel, r0[1][1], r0[1][0])).all(axis=1)
    check(not agree[3] and agree.sum() == KAPPA - 1, "a replaced A corrupts exactly that instance")
    _finish(start)


# --------------------------------------------------------------------------- #
# (c) Correlated OT extension
# --------------------------------------------------------------------------- #

def _cot_one_direction(direction, delta, seeds, ells, cheat=None, sender_tamper=None,
                       receiver_tamper=None):
    """Run ``extend`` for every ell in one direction; returns (keys list, bdoz list, stats)."""
    chosen, (s0, s1) = seeds

    def sender_fn(chan):
        if sender_tamper:
            chan = Tamper(chan, sender_tamper)
        snd = CotSender(chan, SID, direction, delta, chosen)
        return [snd.extend(ell) for ell in ells]

    def receiver_fn(chan):
        if receiver_tamper:
            chan = Tamper(chan, receiver_tamper)
        rcv = CotReceiver(chan, SID, direction, s0, s1)
        return [rcv.extend(ell, cheat if i == 0 else None) for i, ell in enumerate(ells)]

    fns = (guarded(sender_fn), guarded(receiver_fn))
    if direction == 1:
        fns = fns[::-1]
    res = run_two_party(*fns)
    snd_i = 0 if direction == 0 else 1
    return res[snd_i], res[1 - snd_i], res[2 + snd_i], res[3 - snd_i]


def test_cot() -> None:
    start = len(FAILURES)
    rng = random.Random(0xC07)

    # Honest runs, one direction at a time, two calls per state.
    for direction in (0, 1):
        for ell in (1000, 100_000):
            delta = rng.getrandbits(128)
            seeds = trusted_seeds(rng, delta)
            t0 = time.perf_counter()
            keys, own, st_s, st_r = _cot_one_direction(direction, delta, seeds, [ell, 777])
            dt = time.perf_counter() - t0
            tag = f"COT direction {direction}, ell = {ell}"
            check(len(own[0]) == ell and len(keys[0]) == ell and len(own[1]) == 777, f"{tag}: sizes")
            check(bdoz_ok(own[0], keys[0], delta), f"{tag}: q_i = t_i XOR u_i Delta for all rows")
            check(bdoz_ok(own[1], keys[1], delta), f"{tag}: second call correlates as well")
            check(abs(float(own[0].bits.mean()) - 0.5) < 5 * 0.5 / math.sqrt(ell),
                  f"{tag}: the choice bits look uniform")
            check(not np.array_equal(own[0].macs[:777], own[1].macs),
                  f"{tag}: the second call reads a fresh PRG stream")
            check(len({bytes(r) for r in own[0].macs[:2000]}) == min(ell, 2000), f"{tag}: rows differ")
            expect = u_message_bytes(ell) + u_message_bytes(777) + 2 * COT_CHECK_BYTES
            check(st_r["bytes_sent"] == expect and st_s["bytes_sent"] == 32,
                  f"{tag}: bytes = U + check for the receiver, two seeds for the sender")
            check(st_r["rounds"] == st_s["rounds"] == 5, f"{tag}: three flights per call")
            if ell == 100_000:
                print(f"  COT one direction, ell = {ell} (+777): {dt:.2f} s, "
                      f"{st_r['bytes_sent'] / (ell + 777):.2f} bytes per row")
    check(rounded_rows(1) == 64 and rounded_rows(64) == 64 and rounded_rows(65) == 128
          and u_message_bytes(1000) == 16 * (1024 + PAD), "row rounding and U size")

    # Both directions in parallel.
    for ell in (1000, 100_000):
        deltas = (rng.getrandbits(128), rng.getrandbits(128))
        seeds = [trusted_seeds(rng, deltas[d]) for d in (0, 1)]

        def party(role):
            def run(chan):
                snd = CotSender(chan, SID, role, deltas[role], seeds[role][0])
                rcv = CotReceiver(chan, SID, 1 - role, *seeds[1 - role][1])
                return extend_both(chan, snd, rcv, ell)
            return run

        (own0, keys0), (own1, keys1), st0, st1 = run_two_party(party(0), party(1))
        check(bdoz_ok(own0, keys1, deltas[1]) and bdoz_ok(own1, keys0, deltas[0]),
              f"COT both directions, ell = {ell}: both correlations hold")
        check(st0["rounds"] == 3 and st1["rounds"] == 3, f"COT both directions, ell = {ell}: 3 rounds")
        check(st0["bytes_sent"] == u_message_bytes(ell) + 16 + COT_CHECK_BYTES,
              f"COT both directions, ell = {ell}: bytes per party")

    # The padding columns used in the check are pack() of the last 128 rows of each column.
    cols = np.frombuffer(rng.randbytes(8 * KAPPA * 5), dtype="<u8").astype(np.uint64).reshape(KAPPA, 5)
    rows = gf.cols_to_rows(cols)
    packed = gf.ints_to_vec(gf.pack_bits(gf.vec_to_bits(rows[-PAD:])[:, j]) for j in range(KAPPA))
    check(np.array_equal(packed, cols[:, -2:]), "pack of a padding column = its last two words")

    # Malicious receiver: an inconsistent u in ONE column j (one flipped bit of U_j).
    # The sender reads U_j only if Delta_j = 1: the check rejects exactly then.
    ell = 500
    n_rows = rounded_rows(ell) + PAD
    trials, n_abort, n_wrong = 60, 0, 0
    for _ in range(trials):
        delta = rng.getrandbits(128)
        seeds = trusted_seeds(rng, delta)
        col, row = rng.randrange(KAPPA), rng.randrange(n_rows)
        keys, own, _, _ = _cot_one_direction(rng.randrange(2), delta, seeds, [ell],
                                             cheat={"flip_u": [(col, row)]})
        bit = (delta >> col) & 1
        n_abort += aborted(keys)
        if aborted(keys) != bool(bit):
            n_wrong += 1
        elif not aborted(keys) and not bdoz_ok(own[0], keys[0], delta):
            n_wrong += 1
    check(n_wrong == 0, "flipped U bit: abort iff Delta_j = 1; otherwise the output is intact")
    check(10 <= n_abort <= 50, f"flipped U bit: about half of the columns are read ({n_abort}/{trials})")
    print(f"  COT, one inconsistent column: rejected in {n_abort}/{trials} runs "
          f"(exactly the runs with Delta_j = 1)")

    # The same deviation in a column that the sender reads is always rejected.
    rejected = 0
    for _ in range(20):
        delta = rng.getrandbits(128) | 1 << 77
        keys, _, _, _ = _cot_one_direction(rng.randrange(2), delta, trusted_seeds(rng, delta), [ell],
                                           cheat={"flip_u": [(77, rng.randrange(n_rows))]})
        rejected += aborted(keys)
    check(rejected == 20, "flipped U bit in a column with Delta_j = 1: rejected in 20/20 runs")

    # An inconsistent u in 64 columns survives only if Delta is 0 on all of them (2^-64).
    rejected = 0
    for _ in range(10):
        delta = rng.getrandbits(128)
        flips = [(c, rng.randrange(n_rows)) for c in rng.sample(range(KAPPA), 64)]
        keys, _, _, _ = _cot_one_direction(0, delta, trusted_seeds(rng, delta), [ell],
                                           cheat={"flip_u": flips})
        rejected += aborted(keys)
    check(rejected == 10, "inconsistent u in 64 columns: rejected in 10/10 runs")

    # Tampered check message and malformed messages.
    delta = rng.getrandbits(128) | 1
    seeds = trusted_seeds(rng, delta)
    keys, _, _, _ = _cot_one_direction(0, delta, seeds, [ell], receiver_tamper={1: flip_bit(3)})
    check(aborted(keys), "COT: a wrong u~ is rejected")
    keys, _, _, _ = _cot_one_direction(0, delta, seeds, [ell],
                                       receiver_tamper={1: flip_bit(8 * 16 * 40)})
    check(aborted(keys), "COT: a wrong V~_j is rejected")
    keys, _, _, _ = _cot_one_direction(0, delta, seeds, [ell], receiver_tamper={0: lambda m: m[:-8]})
    check(aborted(keys), "COT: a short U message is rejected")
    keys, _, _, _ = _cot_one_direction(0, delta, seeds, [ell], receiver_tamper={1: lambda m: m + b"\0"})
    check(aborted(keys), "COT: a long check message is rejected")
    _, own, _, _ = _cot_one_direction(0, delta, seeds, [ell], sender_tamper={0: lambda m: m[:8]})
    check(aborted(own), "COT: a short challenge seed is rejected by the receiver")
    keys, own, _, _ = _cot_one_direction(0, delta, seeds, [ell])
    check(not aborted(keys) and bdoz_ok(own[0], keys[0], delta), "COT: honest run after the deviations")

    # Argument validation.
    snd = CotSender(None, SID, 0, delta, seeds[0])
    rcv = CotReceiver(None, SID, 0, *seeds[1])
    check(raises(ValueError, rcv.start, 0) and raises(ValueError, snd.absorb, 0, b""),
          "extend rejects ell = 0")
    check(raises(ValueError, CotSender, None, SID, 2, delta, seeds[0])
          and raises(ValueError, CotReceiver, None, SID, 0, seeds[1][0][:5], seeds[1][1]),
          "constructors validate direction and seeds")
    _finish(start)


# --------------------------------------------------------------------------- #
# (d) Authenticated bits
# --------------------------------------------------------------------------- #

def test_abits() -> None:
    start = len(FAILURES)
    rng = random.Random(0xAB17)
    n, n_mask = 100_000, 5_000

    def body(role, chan, source, delta):
        shares, own, keys = source.random_shares(n, keep_bdoz=True)
        m0 = source.input_masks(0, n_mask)
        m1 = source.input_masks(1, n_mask + 3)
        more = source.random_shares(300)
        return shares, own, keys, m0, m1, more

    t0 = time.perf_counter()
    r0, r1, st0, st1, (d0, d1) = abit_session(rng, body)
    dt = time.perf_counter() - t0
    delta = d0 ^ d1
    check(delta & 1 == 1, "lsb(Delta^B) = 1")
    check(len(r0[0]) == n and mac_ok(r0[0], r1[0], delta), f"abits: m_0 ^ m_1 = x Delta for {n} bits")
    check(bdoz_ok(r0[1], r1[2], d1), "abits: M[x_0] = K[x_0] ^ x_0 Delta_1")
    check(bdoz_ok(r1[1], r0[2], d0), "abits: M[x_1] = K[x_1] ^ x_1 Delta_0")
    check(np.array_equal(r0[0].bits, r0[1].bits) and np.array_equal(r1[0].bits, r1[1].bits),
          "abits: the share of x is the party's own BDOZ bit")
    x = r0[0].bits ^ r1[0].bits
    check(abs(float(x.mean()) - 0.5) < 5 * 0.5 / math.sqrt(n), "abits: x looks uniform")
    check(abs(float(r0[0].bits.mean()) - 0.5) < 5 * 0.5 / math.sqrt(n), "abits: x_0 looks uniform")

    (sh00, bits00), (sh01, bits01) = r0[3], r1[3]          # masks of P_0
    check(bits01 is None and bits00 is not None, "input masks of P_0: only P_0 learns the bits")
    check(mac_ok(sh00, sh01, delta, bits00), "input masks of P_0: <r> authenticates r")
    check(np.array_equal(sh00.bits, bits00) and not sh01.bits.any(),
          "input masks of P_0: r_0 = r, r_1 = 0")
    (sh10, bits10), (sh11, bits11) = r0[4], r1[4]          # masks of P_1
    check(bits10 is None and len(bits11) == n_mask + 3, "input masks of P_1: only P_1 learns the bits")
    check(mac_ok(sh10, sh11, delta, bits11) and not sh10.bits.any(), "input masks of P_1")
    check(abs(float(bits00.mean()) - 0.5) < 5 * 0.5 / math.sqrt(n_mask), "mask bits look uniform")
    check(mac_ok(r0[5], r1[5], delta), "abits: later requests stay consistent")

    # Byte accounting: everything is OT extension traffic, nothing in "abit".
    per_party = u_message_bytes(n) + u_message_bytes(300) + 2 * (16 + COT_CHECK_BYTES)
    mask_recv = lambda m: u_message_bytes(m) + COT_CHECK_BYTES    # noqa: E731
    exp0 = per_party + mask_recv(n_mask) + 16          # P_0 receives for its own masks
    exp1 = per_party + 16 + mask_recv(n_mask + 3)
    check(st0["phases"]["cotext"]["bytes_sent"] == exp0 == st0["bytes_sent"]
          and st1["phases"]["cotext"]["bytes_sent"] == exp1 == st1["bytes_sent"],
          "abits: bytes are exactly the U, seed and check messages")
    check(st0["phases"]["abit"]["bytes_sent"] == 0 and st0["phases"]["abit"]["rounds"] == 0,
          "abits: the conversion to shares is local")
    check(st0["rounds"] == st1["rounds"], "abits: both endpoints count the same rounds")
    print(f"  abits: {n} random bits + {2 * n_mask + 3} masks + 300 bits in {dt:.2f} s, "
          f"{(st0['bytes_sent'] + st1['bytes_sent']) / (n + 300 + n_mask + 1.5):.1f} bytes per item")

    # Chunked extension gives the same guarantees.
    def body_chunked(role, chan, source, delta):
        return source.random_shares(10_000), source.input_masks(1, 9_000), source._sender.calls

    r0, r1, _, _, (d0, d1) = abit_session(rng, body_chunked, batch_rows=4096)
    check(mac_ok(r0[0], r1[0], d0 ^ d1) and mac_ok(r0[1][0], r1[1][0], d0 ^ d1, r1[1][1]),
          "abits in chunks of 4096 rows")
    check(r0[2] == 3 + 3 and r1[2] == 3, "abits: one extension call per chunk")

    # n = 0 needs no message; the lsb constraint is enforced.
    r0, r1, st0, _, _ = abit_session(rng, lambda role, chan, src, d: (src.random_shares(0),
                                                                    src.input_masks(0, 0)))
    check(len(r0[0]) == 0 and len(r0[1][0]) == 0 and st0["bytes_sent"] == 0, "abits: empty requests")
    d0, d1 = test_deltas(rng)
    seeds = trusted_seeds(rng, d1)
    snd = CotSender(None, SID, 0, d1, seeds[0])
    rcv = CotReceiver(None, SID, 1, *seeds[1])
    check(raises(ValueError, AbitSource, None, 0, d1, snd, rcv), "AbitSource rejects lsb(Delta_0) = 0")
    snd1 = CotSender(None, SID, 1, d0, seeds[0])
    rcv1 = CotReceiver(None, SID, 0, *seeds[1])
    check(raises(ValueError, AbitSource, None, 1, d0, snd1, rcv1), "AbitSource rejects lsb(Delta_1) = 1")
    snd_ok = CotSender(None, SID, 0, d0, seeds[0])
    check(raises(ValueError, AbitSource, None, 0, d0, snd_ok, rcv1),
          "AbitSource rejects a receiver of the wrong direction")
    check(raises(ValueError, AbitSource, None, 0, d0 ^ 2, snd_ok, rcv),
          "AbitSource rejects a sender with another key")
    _finish(start)


# --------------------------------------------------------------------------- #
# (e) Leaky AND, honest
# --------------------------------------------------------------------------- #

def _leaky_body(n, index0=0, cheat=None, cheater=None, with_check=False):
    def body(role, chan, source, delta):
        own, keys = source.random_bdoz(2 * n + (PAD if with_check else 0))
        x_own, x_keys = own[:n], keys[:n]
        x = spdz_shares(role, delta, x_own, x_keys)
        y = spdz_shares(role, delta, own[n:2 * n], keys[n:2 * n])
        chan.mark("laand")
        z = aand.leaky_ands(chan, SID, role, delta, index0, x_own, x_keys, y,
                            cheat if role == cheater else None)
        passed = None
        if with_check:
            r = spdz_shares(role, delta, own[2 * n:], keys[2 * n:])
            try:
                aand.triple_check(chan, SID, 0, role, delta, z, r)
                passed = True
            except ProtocolAbort:
                passed = False
        return x, y, z, passed
    return body


def test_leaky_and() -> None:
    start = len(FAILURES)
    rng = random.Random(0x1EA)
    n = 100_000
    t0 = time.perf_counter()
    r0, r1, st0, st1, (d0, d1) = abit_session(rng, _leaky_body(n, index0=12345))
    dt = time.perf_counter() - t0
    delta = d0 ^ d1
    x = r0[0].bits ^ r1[0].bits
    y = r0[1].bits ^ r1[1].bits
    z = r0[2].bits ^ r1[2].bits
    check(mac_ok(r0[0], r1[0], delta) and mac_ok(r0[1], r1[1], delta), "leaky AND: inputs are valid")
    check(np.array_equal(z, x & y), f"leaky AND: z = x AND y for {n} triples")
    check(np.array_equal(r0[2].macs ^ r1[2].macs, gf.bit_mul(x & y, delta)),
          "leaky AND: n_0 ^ n_1 = x y Delta")
    check(np.array_equal(r0[2].bits, gf.lsb(r0[2].macs))
          and np.array_equal(r1[2].bits, gf.lsb(r1[2].macs)),
          "leaky AND: z_b = lsb(n_b)")
    check(abs(float(z.mean()) - 0.25) < 5 * math.sqrt(0.1875 / n),
          "leaky AND: z is 1 a quarter of the time")
    check(abs(float(r0[2].bits.mean()) - 0.5) < 5 * 0.5 / math.sqrt(n), "leaky AND: z_0 looks uniform")
    la0 = st0["phases"]["laand"]
    check(la0["bytes_sent"] == 16 * n and la0["bytes_recv"] == 16 * n and la0["rounds"] == 1,
          "leaky AND: one round, 16 bytes per triple and party")
    print(f"  leaky AND: {n} triples (with their 2 x {2 * n} COT rows) in {dt:.2f} s")

    # The triple check accepts honest triples; the hash depends on index, direction and sid.
    r0, r1, _, _, (d0, d1) = abit_session(rng, _leaky_body(500, with_check=True))
    check(r0[3] is True and r1[3] is True, "triple check passes on honest leaky triples")
    keys = rand_vec(rng, 4)
    tweak = np.arange(4, dtype=np.uint64)
    h = hs.H_ccr(keys, tweak, aand.h_dom(SID, 0))
    check(not np.array_equal(h, hs.H_ccr(keys, tweak, aand.h_dom(SID, 1)))
          and not np.array_equal(h, hs.H_ccr(keys, tweak + np.uint64(1), aand.h_dom(SID, 0)))
          and not np.array_equal(h, hs.H_ccr(keys, tweak, aand.h_dom(SID + b"x", 0))),
          "leaky AND hash is separated by direction, triple index and session")

    # A party that ignores the lsb constraint (lsb(Delta) = 0) is caught by the triple check.
    dealer = IdealDealer()
    dealer.deltas = (dealer.deltas[0], dealer.deltas[1] | 1)

    def party(role):
        def run(chan):
            prep = IdealPrep(role, dealer)
            return _leaky_body(64, with_check=True)(role, chan, prep, prep.delta)
        return run

    r0, r1, _, _ = run_two_party(party(0), party(1))
    check(r0[3] is False and r1[3] is False, "lsb(Delta^B) = 0 makes the triple check fail")

    # Malformed transfer message.
    def short_g(role):
        def run(chan):
            prep = IdealPrep(role, dealer2)
            c = Tamper(chan, {0: lambda m: m[:-16]}) if role == 1 else chan
            return _leaky_body(64)(role, c, prep, prep.delta)
        return guarded(run)

    dealer2 = IdealDealer()
    r0, r1, _, _ = run_two_party(short_g(0), short_g(1))
    check(aborted(r0), "leaky AND: a short transfer message is rejected")

    # The transfer is split into messages of LAAND_MSG_TRIPLES triples (2^24, the constant of
    # the native code).  With a small constant the same inputs give the same shares, in
    # ceil(n / constant) rounds; with the default the transfer is one message.
    check(aand.LAAND_MSG_TRIPLES == 1 << 24,
          "LAAND_MSG_TRIPLES = 2^24 (as in cpp/include/couplet/aand.hpp)")
    dealer4 = IdealDealer()
    n4 = 200
    drawn = [IdealPrep(role, dealer4).random_bdoz(2 * n4) for role in (0, 1)]

    def split_party(role):
        def run(chan):
            own, keys = drawn[role]
            delta = dealer4.deltas[role]
            y = spdz_shares(role, delta, own[n4:], keys[n4:])
            chan.mark("laand")
            return aand.leaky_ands(chan, SID, role, delta, 7, own[:n4], keys[:n4], y)
        return run

    outs = {}
    saved = aand.LAAND_MSG_TRIPLES
    try:
        for limit in (saved, 64, 200, 199, 1):
            aand.LAAND_MSG_TRIPLES = limit
            z0, z1, st0, st1 = run_two_party(split_party(0), split_party(1))
            outs[limit] = (z0, z1)
            want_rounds = -(-n4 // limit)
            la = st0["phases"]["laand"]
            check(la["rounds"] == want_rounds == st1["phases"]["laand"]["rounds"]
                  and la["msgs_sent"] == want_rounds
                  and la["bytes_sent"] == 16 * n4 == la["bytes_recv"],
                  f"leaky AND, {n4} triples, {limit} per message: {want_rounds} rounds, "
                  f"16 bytes per triple")
    finally:
        aand.LAAND_MSG_TRIPLES = saved
    x4 = drawn[0][0].bits[:n4] ^ drawn[1][0].bits[:n4]
    y4 = drawn[0][0].bits[n4:] ^ drawn[1][0].bits[n4:]
    z0, z1 = outs[saved]
    check(np.array_equal(z0.bits ^ z1.bits, x4 & y4)
          and np.array_equal(z0.macs ^ z1.macs, gf.bit_mul(x4 & y4, dealer4.delta)),
          "leaky AND in one message: z = x AND y with its MAC")
    check(all(np.array_equal(a.bits, z0.bits) and np.array_equal(a.macs, z0.macs)
              and np.array_equal(b.bits, z1.bits) and np.array_equal(b.macs, z1.macs)
              for a, b in outs.values()),
          "leaky AND: the shares do not depend on how the transfer is split")
    _finish(start)


# --------------------------------------------------------------------------- #
# (f) Lemma 3: additive error on the transfer message
# --------------------------------------------------------------------------- #

def _attack_run(rng: random.Random, t: int, cheater: int, n: int = 24):
    """One run of the Lemma 3 attack; returns (passed_0, passed_1, guesses right, triples ok)."""
    dealer = IdealDealer()
    idx = np.array(sorted(rng.sample(range(n), t)))
    err = rand_vec(rng, t)
    err[:, 1] |= np.uint64(1 << 63)                      # E_i != 0
    guess = rand_bits(rng, t)
    cheat = {"g_error": (idx, err, guess)}

    def party(role):
        def run(chan):
            prep = IdealPrep(role, dealer)
            return _leaky_body(n, cheat=cheat, cheater=cheater, with_check=True)(
                role, chan, prep, prep.delta)
        return run

    r0, r1, _, _ = run_two_party(party(0), party(1))
    honest = (r0, r1)[1 - cheater]
    right = bool(np.array_equal(honest[0].bits[idx], guess))   # the honest party's x shares
    x = r0[0].bits ^ r1[0].bits
    y = r0[1].bits ^ r1[1].bits
    z_ok = np.array_equal(r0[2].bits ^ r1[2].bits, x & y)
    mac_good = np.array_equal(r0[2].macs ^ r1[2].macs, gf.bit_mul(x & y, dealer.delta))
    return r0[3], r1[3], right, (z_ok and mac_good)


def test_leaky_and_attack() -> None:
    start = len(FAILURES)
    rng = random.Random(0xF00D)
    runs = 200
    for t in (1, 2, 3):
        passes = mismatch = bad_pass = good_fail = 0
        for i in range(runs):
            p0, p1, right, triples_ok = _attack_run(rng, t, cheater=i % 2)
            passed = p0 and p1
            passes += passed
            mismatch += (p0 != p1) or (passed != right)
            bad_pass += passed and not triples_ok
            good_fail += (not passed) and triples_ok
        rate, expect = passes / runs, 2.0 ** -t
        sd = math.sqrt(expect * (1 - expect) / runs)
        print(f"  Lemma 3, t = {t}: check passed in {passes}/{runs} runs "
              f"(rate {rate:.3f}, expected {expect:.3f})")
        check(mismatch == 0, f"t = {t}: the check passes iff every guess is right (both parties agree)")
        check(bad_pass == 0, f"t = {t}: whenever the check passes all triples are correct")
        check(good_fail == 0, f"t = {t}: a wrong guess leaves an inconsistent triple")
        check(abs(rate - expect) < 5 * sd, f"t = {t}: pass rate {rate:.3f} is about 2^-{t}")

    # An error on the share of W alone, or on a raw z share, is always caught.
    cases = (("share of W", {"w_error": 1}, None),
             ("share of W (high bit)", {"w_error": 1 << 100}, None),
             ("raw z share", None, 3))
    for name, cheat, flip_z in cases:
        dealer = IdealDealer()

        def party(role, cheat=cheat, flip_z=flip_z, dealer=dealer):
            def run(chan):
                prep = IdealPrep(role, dealer)
                own, keys = prep.random_bdoz(2 * 40 + PAD)
                x_own, x_keys = own[:40], keys[:40]
                y = spdz_shares(role, prep.delta, own[40:80], keys[40:80])
                r = spdz_shares(role, prep.delta, own[80:], keys[80:])
                z = aand.leaky_ands(chan, SID, role, prep.delta, 0, x_own, x_keys, y)
                if role == 0 and flip_z is not None:
                    z.bits[flip_z] ^= 1
                aand.triple_check(chan, SID, 0, role, prep.delta, z, r,
                                  cheat=cheat if role == 0 else None)
                return "ok"
            return guarded(run)

        r0, r1, _, _ = run_two_party(party(0), party(1))
        check(aborted(r0) and aborted(r1), f"triple check: an error on the {name} is rejected")
    _finish(start)


# --------------------------------------------------------------------------- #
# Bucketing
# --------------------------------------------------------------------------- #

def test_bucketing() -> None:
    start = len(FAILURES)
    rng = random.Random(0xB0C)

    # The public permutation.
    seed = rng.randbytes(16)
    perm = aand.bucket_permutation(seed, 10_000)
    check(np.array_equal(np.sort(perm), np.arange(10_000)), "bucket_permutation is a permutation")
    check(np.array_equal(perm, aand.bucket_permutation(seed, 10_000)), "deterministic in the seed")
    check(not np.array_equal(perm, aand.bucket_permutation(rng.randbytes(16), 10_000)),
          "depends on the seed")
    check(int((perm == np.arange(10_000)).sum()) < 20, "few fixed points")
    counts: dict[tuple, int] = {}
    for _ in range(1200):
        p = tuple(int(v) for v in aand.bucket_permutation(rng.randbytes(16), 3))
        counts[p] = counts.get(p, 0) + 1
    check(len(counts) == 6 and all(abs(c - 200) < 5 * math.sqrt(1200 * 5 / 36) for c in counts.values()),
          f"all 6 permutations of 3 elements are about equally likely ({sorted(counts.values())})")

    # Bucket combination on ideal (correct) leaky triples.
    for bucket in (1, 2, 3, 5):
        n_out = 300
        total = n_out * bucket
        dealer = IdealDealer()
        seed = rng.randbytes(16)

        def party(role, dealer=dealer, seed=seed, bucket=bucket, total=total):
            def run(chan):
                prep = IdealPrep(role, dealer)
                x, y, z = prep.and_triples(total)
                perm = aand.bucket_permutation(seed, total)
                a, b, c, d, d_macs = aand.bucket_combine(chan, x, y, z, perm, bucket)
                bool_check(chan, SID, bucket, role, prep.delta, d, d_macs)
                return (x, y, z), (a, b, c), d, perm
            return run

        r0, r1, st0, _ = run_two_party(party(0), party(1))
        delta = dealer.delta
        (a0, b0, c0), (a1, b1, c1) = r0[1], r1[1]
        a, b, c = a0.bits ^ a1.bits, b0.bits ^ b1.bits, c0.bits ^ c1.bits
        tag = f"bucket size {bucket}"
        check(len(a0) == n_out and np.array_equal(c, a & b), f"{tag}: c = a AND b")
        check(mac_ok(a0, a1, delta) and mac_ok(b0, b1, delta) and mac_ok(c0, c1, delta),
              f"{tag}: MAC invariants of the combined triples")
        x = r0[0][0].bits ^ r1[0][0].bits
        y = r0[0][1].bits ^ r1[0][1].bits
        groups = r0[3].reshape(n_out, bucket)
        check(np.array_equal(a, np.bitwise_xor.reduce(x[groups], axis=1)), f"{tag}: a = XOR of the x")
        check(np.array_equal(b, y[groups[:, 0]]), f"{tag}: b = y of the head")
        check(np.array_equal(r0[2], r1[2]) and np.array_equal(
            r0[2].reshape(n_out, bucket - 1), y[groups[:, :1]] ^ y[groups[:, 1:]]),
            f"{tag}: d = y' XOR y''")
        check(st0["bytes_sent"] == (n_out * (bucket - 1) + 7) // 8 + BOOLCHECK_BYTES,
              f"{tag}: (B - 1) bits per triple")
    check(raises(ValueError, aand.bucket_combine, None, Shares.zeros(5), Shares.zeros(5),
                 Shares.zeros(5), np.arange(5), 2), "bucket_combine rejects a ragged population")

    # A leaky triple that is wrong makes its whole bucket wrong (correctness is not
    # repaired by bucketing; it is enforced by the triple check).
    dealer = IdealDealer()
    seed = rng.randbytes(16)

    def party_bad(role):
        def run(chan):
            prep = IdealPrep(role, dealer)
            x, y, z = prep.and_triples(90)
            if role == 0:
                z.bits[17] ^= 1
            perm = aand.bucket_permutation(seed, 90)
            a, b, c, _, _ = aand.bucket_combine(chan, x, y, z, perm, 3)
            return a, b, c, perm
        return run

    r0, r1, _, _ = run_two_party(party_bad(0), party_bad(1))
    wrong = (r0[2].bits ^ r1[2].bits) != ((r0[0].bits ^ r1[0].bits) & (r0[1].bits ^ r1[1].bits))
    where = int(np.flatnonzero(r0[3] == 17)[0]) // 3
    check(wrong.sum() == 1 and wrong[where], "a wrong leaky triple corrupts exactly its bucket")
    _finish(start)


# --------------------------------------------------------------------------- #
# (g) RealPrep end to end
# --------------------------------------------------------------------------- #

def _real_session(body, sid=SID, sigma=40, cheat=(None, None), rngs=(None, None), **kwargs):
    """Run ``body(prep, chan)`` for both parties on a fresh RealPrep session (after setup)."""
    def party(role):
        def run(chan):
            prep = RealPrep(role, chan, sid, sigma, rngs[role], cheat=cheat[role])
            try:
                prep.setup()
                out = body(prep, chan)
            except ProtocolAbort as exc:
                return ("abort", str(exc)), prep
            return out, prep
        return run
    (out0, prep0), (out1, prep1), st0, st1 = run_two_party(party(0), party(1), **kwargs)
    return out0, out1, st0, st1, prep0, prep1


def _triples_ok(t0, t1, delta) -> tuple[bool, bool]:
    a = t0[0].bits ^ t1[0].bits
    b = t0[1].bits ^ t1[1].bits
    c = t0[2].bits ^ t1[2].bits
    return bool(np.array_equal(c, a & b)), all(mac_ok(t0[i], t1[i], delta) for i in range(3))


def _and_bytes_expected(n: int, bucket: int) -> dict:
    """Bytes SENT by one party during and_triples(n), per phase (exact)."""
    total = n * bucket
    return {
        "cotext": u_message_bytes(2 * total + PAD) + 16 + COT_CHECK_BYTES,
        "abit": 0,
        "laand": 16 * total,
        "bucket": COIN_BYTES + (n * (bucket - 1) + 7) // 8,
        "check": COIN_BYTES + 16 + BOOLCHECK_BYTES,
    }


def _measure_triples(n: int):
    """One session: setup, and_triples(n).  Returns a dict of measurements and the results."""
    def body(prep, chan):
        delta = prep.delta
        t_start = time.perf_counter()
        triples = prep.and_triples(n)
        return triples, time.perf_counter() - t_start, delta

    t0 = time.perf_counter()
    out0, out1, st0, st1, prep0, prep1 = _real_session(body)
    wall = time.perf_counter() - t0
    return out0, out1, st0, st1, prep0, prep1, wall


def _report_triples(n, out0, st0, st1, prep0, wall) -> None:
    bucket = prep0.last_bucket_size
    phases = st0["phases"]
    prep_bytes = sum(phases[p]["bytes_sent"] + phases[p]["bytes_recv"]
                     for p in ("cotext", "abit", "laand", "bucket", "check"))
    total_bytes = st0["bytes_sent"] + st0["bytes_recv"]
    print(f"  RealPrep.and_triples(n = {n}), sigma = 40, B = {bucket}:")
    print(f"    wall time (both parties, one process): and_triples {out0[1]:.2f} s, "
          f"session incl. base OT {wall:.2f} s; {out0[1] / n * 1e6:.1f} us per final triple")
    print(f"    bytes per final AND triple (both directions): {prep_bytes / n:.1f} without base OT, "
          f"{total_bytes / n:.1f} with base OT; model 96 B = {96 * bucket}")
    print("    per phase (bytes both directions / rounds / seconds at P_0): " + ", ".join(
        f"{p} {phases[p]['bytes_sent'] + phases[p]['bytes_recv']}/{phases[p]['rounds']}/"
        f"{phases[p]['time_s']:.2f}" for p in ("baseot", "cotext", "abit", "laand", "bucket", "check")))


def test_real_prep() -> None:
    start = len(FAILURES)
    for n in (1000, 50_000):
        out0, out1, st0, st1, prep0, prep1, wall = _measure_triples(n)
        tag = f"RealPrep n = {n}"
        bucket = params.bucket_size(n, 40)
        (t0, _, delta0), (t1, _, delta1) = out0, out1
        delta = delta0 ^ delta1
        check(delta0 & 1 == 1 and delta1 & 1 == 0, f"{tag}: lsb(Delta_0) = 1, lsb(Delta_1) = 0")
        check(all(len(s) == n for s in t0 + t1), f"{tag}: n triples for both parties")
        correct, macs = _triples_ok(t0, t1, delta)
        check(correct, f"{tag}: c = a AND b for all triples")
        check(macs, f"{tag}: MAC invariants of a, b and c")
        a, b = t0[0].bits ^ t1[0].bits, t0[1].bits ^ t1[1].bits
        check(abs(float(a.mean()) - 0.5) < 5 * 0.5 / math.sqrt(n)
              and abs(float(b.mean()) - 0.5) < 5 * 0.5 / math.sqrt(n)
              and abs(float((a & b).mean()) - 0.25) < 5 * math.sqrt(0.1875 / n),
              f"{tag}: a and b look uniform and independent")
        check(prep0.last_bucket_size == bucket == prep1.last_bucket_size,
              f"{tag}: bucket size {bucket} as params.bucket_size says")
        check(prep0.counters["leaky_triples"] == n * bucket and prep0.counters["and_triples"] == n,
              f"{tag}: n * B leaky triples were consumed")
        check(prep0.cot_rows_for_triples(n) == 2 * n * bucket + PAD, f"{tag}: COT rows per direction")
        expected = _and_bytes_expected(n, bucket)
        expected["baseot"] = 96 * KAPPA
        for st in (st0, st1):
            got = {p: st["phases"][p]["bytes_sent"] for p in expected}
            check(got == expected, f"{tag}: bytes sent per phase by P_{st['party']} are exact "
                                   f"({got} vs {expected})")
            check(st["bytes_sent"] == sum(expected.values()) and set(st["phases"]) == set(expected),
                  f"{tag}: the phases sum to the total for P_{st['party']}")
        check(st0["bytes_sent"] == st1["bytes_recv"] and st1["bytes_sent"] == st0["bytes_recv"],
              f"{tag}: sent and received bytes match")
        rounds = {p: st0["phases"][p]["rounds"] for p in st0["phases"]}
        check(rounds == {"baseot": 2, "cotext": 3, "abit": 0, "laand": 1, "bucket": 3, "check": 5}
              and st1["rounds"] == st0["rounds"] == 14, f"{tag}: rounds per phase ({rounds})")
        per_and = 2 * sum(v for p, v in expected.items() if p != "baseot") / n
        check(96 * bucket <= per_and <= 96 * bucket * (1.0 + 0.02) + 12_000 / n,
              f"{tag}: {per_and:.1f} bytes per triple against the model 96 B = {96 * bucket}")
        _report_triples(n, out0, st0, st1, prep0, wall)

    # Several requests in one session; sigma = 64; empty requests; phase restoration.
    def body(prep, chan):
        chan.mark("online")
        first = prep.and_triples(200)
        check_phase = chan.phase
        bits = prep.random_bits(777)
        masks0 = prep.input_masks(0, 100)
        masks1 = prep.input_masks(1, 60)
        second = prep.and_triples(50)
        empty = (prep.and_triples(0), prep.random_bits(0), prep.input_masks(0, 0),
                 prep.input_masks(1, 0))
        return first, bits, masks0, masks1, second, empty, check_phase, prep.delta, dict(prep.counters)

    out0, out1, st0, st1, prep0, prep1 = _real_session(body, sigma=64)
    delta = out0[7] ^ out1[7]
    check(all(_triples_ok(out0[0], out1[0], delta)) and all(_triples_ok(out0[4], out1[4], delta)),
          "RealPrep: two and_triples batches in one session are both correct")
    check(mac_ok(out0[1], out1[1], delta), "RealPrep.random_bits: MAC invariant")
    check(mac_ok(out0[2][0], out1[2][0], delta, out0[2][1]) and out1[2][1] is None,
          "RealPrep.input_masks(0): invariant and ownership")
    check(mac_ok(out0[3][0], out1[3][0], delta, out1[3][1]) and out0[3][1] is None,
          "RealPrep.input_masks(1): invariant and ownership")
    b200, b50 = params.bucket_size(200, 64), params.bucket_size(50, 64)
    check(out0[8] == {"random_bits": 777, "input_masks": 160, "and_triples": 250,
                      "leaky_triples": 200 * b200 + 50 * b50},
          f"RealPrep counters (sigma = 64: B = {b200} for 200, B = {b50} for 50)")
    check(prep0.last_bucket_size == b50 and b50 > params.bucket_size(50, 40),
          "sigma = 64 buckets are larger")
    check(out0[6] == "online", "RealPrep restores the caller's accounting phase")
    check(all(len(s) == 0 for s in out0[5][0]) and len(out0[5][1]) == 0
          and len(out0[5][2][1]) == 0 and out0[5][3][1] is None, "RealPrep: empty requests")
    check(set(st0["phases"]) == {"baseot", "cotext", "abit", "laand", "bucket", "check", "online"}
          and st0["phases"]["online"]["bytes_sent"] == 0, "RealPrep books nothing under foreign phases")
    check(st0["rounds"] == st1["rounds"], "RealPrep: both endpoints count the same rounds")

    # The same session over real sockets.
    def body_tcp(prep, chan):
        return prep.and_triples(300), prep.delta

    out0, out1, st0, st1, _, _ = _real_session(body_tcp, tcp=True)
    check(all(_triples_ok(out0[0], out1[0], out0[1] ^ out1[1])) and st0["transport"] == "tcp",
          "RealPrep over TCP on 127.0.0.1")

    # Seeded randomness fixes the key share; argument validation leaves the session alive.
    def body_args(prep, chan):
        delta = prep.delta
        bad = (raises(ValueError, prep.random_bits, -1), raises(ValueError, prep.input_masks, 2, 5),
               raises(ValueError, prep.and_triples, 1.5), raises(ValueError, prep.setup))
        return bad, prep.aborted, prep.random_bits(10), delta

    seeded = (random.Random(11), random.Random(12))
    out0, out1, _, _, _, _ = _real_session(body_args, rngs=seeded)
    check(all(out0[0]) and out0[1] is False and mac_ok(out0[2], out1[2], out0[3] ^ out1[3]),
          "RealPrep: invalid arguments raise ValueError and do not abort the session")
    check(out0[3] == sample_delta(0, random.Random(11))
          and out1[3] == sample_delta(1, random.Random(12)),
          "RealPrep: the key share is the first draw of a seeded source")
    c0, c1 = loopback_pair()
    check(raises(ValueError, RealPrep, 1, c0, SID) and raises(ValueError, RealPrep, 2, c0, SID),
          "RealPrep validates the role against the endpoint")
    prep = RealPrep(0, c0, SID)
    check(raises(ValueError, prep.random_bits, 5) and not prep.aborted, "RealPrep requires setup()")
    c0.close()
    c1.close()
    _finish(start)


def test_small_requests() -> None:
    """A request below 306 leaky triples is padded; the padded run is a complete protocol run."""
    start = len(FAILURES)
    m = params.MIN_LEAKY_TRIPLES

    def body_for(sizes):
        def body(prep, chan):
            delta = prep.delta
            rows = []
            for n in sizes:
                sent, rounds = chan.bytes_sent, chan.rounds
                triples = prep.and_triples(n)
                rows.append((triples, chan.bytes_sent - sent, chan.rounds - rounds,
                             prep.last_bucket_size, prep.last_run_triples, dict(prep.counters),
                             prep.cot_rows_for_triples(n)))
            return rows, delta, prep._triple_index, prep._batch
        return body

    for sigma, sizes in ((40, (1, 5, 43, 44, 45)), (64, (3, 25, 26))):
        out0, out1, st0, st1, prep0, prep1 = _real_session(body_for(sizes), sigma=sigma)
        delta = out0[1] ^ out1[1]
        leaky = 0
        for i, n in enumerate(sizes):
            run, bucket = params.triple_request(n, sigma)
            leaky += run * bucket
            tag = f"sigma = {sigma}, request of {n}"
            (t0, sent0, rounds0, b0, run0, ctr0, rows0) = out0[0][i]
            (t1, sent1, rounds1, b1, run1, ctr1, rows1) = out1[0][i]
            check(run * bucket >= m and (run > n) == (n * params.bucket_size(n, sigma) < m),
                  f"{tag}: the run has {run} x {bucket} = {run * bucket} >= {m} leaky triples")
            check(all(len(sh) == n for sh in t0 + t1), f"{tag}: exactly {n} triples are returned")
            correct, macs = _triples_ok(t0, t1, delta)
            check(correct and macs, f"{tag}: the triples are correct and authenticated")
            check((b0, run0) == (bucket, run) == (b1, run1),
                  f"{tag}: bucket size and run size as planned")
            check(ctr0["leaky_triples"] == leaky == ctr1["leaky_triples"]
                  and ctr0["and_triples"] == sum(sizes[:i + 1]),
                  f"{tag}: the counters hold the leaky triples of the padded run")
            check(rows0 == 2 * run * bucket + PAD == rows1, f"{tag}: COT rows per direction")
            expected = sum(_and_bytes_expected(run, bucket).values())
            check(sent0 == expected == sent1,
                  f"{tag}: the bytes are those of a request for {run} triples ({sent0} vs {expected})")
            check(rounds0 == rounds1 == 12, f"{tag}: twelve rounds")
        check(out0[2] == leaky == out1[2] and out0[3] == len(sizes),
              f"sigma = {sigma}: the triple index (tweak of H) advances by the leaky triples "
              f"of every run")

    # The check covers the padding: deviations on leaky triples and opened bits that exist
    # only because the request was padded abort the session.
    def body_one(prep, chan):
        return prep.and_triples(1)

    idx = 44 + np.arange(40) * 6                     # leaky triples 44 .. 278 of the 308
    err = gf.rand_vec(40)
    err[:, 0] |= np.uint64(2)
    out0, out1, _, _, prep0, prep1 = _real_session(
        body_one, cheat=(None, {"g_error": (idx, err, np.zeros(40, dtype=np.uint8))}))
    check(aborted(out0) and aborted(out1) and prep0.aborted and prep1.aborted,
          "request of 1: errors on 40 transfer messages of the padded run are caught")
    for cheater in (0, 1):
        cheat = [None, None]
        cheat[cheater] = {"flip_d": [200]}           # one of the 44 x 6 = 264 opened bits
        out0, out1, _, _, prep0, prep1 = _real_session(body_one, cheat=tuple(cheat))
        check(aborted(out0) and aborted(out1) and "BoolCheck" in out0[1] + out1[1],
              f"request of 1: P_{cheater} flips an opened bit of the padded run: BoolCheck aborts")
    _finish(start)


def test_real_prep_aborts() -> None:
    start = len(FAILURES)

    def after_abort(prep) -> bool:
        return (prep.aborted and prep.delta is None
                and raises(ProtocolAbort, prep.random_bits, 4)
                and raises(ProtocolAbort, prep.and_triples, 4)
                and raises(ProtocolAbort, prep.input_masks, 0, 4))

    # (i) at the pipeline level: a tampered opened d bit makes the BoolCheck abort.
    def body(prep, chan):
        return prep.and_triples(400)

    for cheater in (0, 1):
        cheat = [None, None]
        cheat[cheater] = {"flip_d": [123]}
        out0, out1, st0, _, prep0, prep1 = _real_session(body, cheat=tuple(cheat))
        check(aborted(out0) and aborted(out1) and "BoolCheck" in out0[1] + out1[1],
              f"P_{cheater} flips an opened d bit: both parties abort in BoolCheck")
        check(after_abort(prep0) and after_abort(prep1),
              f"P_{cheater} flips an opened d bit: the session is retired on both sides")

    # Lemma 3 deviation through the whole pipeline: errors on 40 transfer messages
    # survive only if all 40 guesses are right (probability 2^-40).
    idx = np.arange(40) * 5
    err = gf.rand_vec(40)
    err[:, 0] |= np.uint64(2)
    out0, out1, _, _, prep0, prep1 = _real_session(
        body, cheat=(None, {"g_error": (idx, err, np.zeros(40, dtype=np.uint8))}))
    check(aborted(out0) and aborted(out1) and after_abort(prep0),
          "errors on 40 transfer messages are caught by the triple check")

    # An error on the share of W.
    out0, out1, _, _, prep0, prep1 = _real_session(body, cheat=({"w_error": 3}, None))
    check(aborted(out0) and aborted(out1) and after_abort(prep1),
          "a wrong share of W aborts the session")

    # An inconsistent U in 40 columns of the extension is caught by the COT check
    # (it survives only if Delta is 0 on all 40 columns).  The cheater notices at its next receive.
    def body_bits(prep, chan):
        return prep.random_bits(1000), prep.random_bits(10)

    flips = [(c, 5 + c) for c in range(0, 120, 3)]
    out0, out1, _, _, prep0, prep1 = _real_session(body_bits, cheat=(None, {"flip_u": flips}))
    check(aborted(out0) and "COT" in out0[1] and after_abort(prep0),
          "an inconsistent extension matrix aborts the honest sender")
    check(aborted(out1) and prep1.aborted, "the cheater's session ends as well (peer closed)")

    # A base-OT sender that replaces every second A: the seeds of those columns disagree
    # and the extension check of the honest COT sender fails.
    blob = ed.compress(ed.base_mult(424242))

    def replace(msg: bytes) -> bytes:
        out = bytearray(msg)
        for i in range(0, KAPPA, 2):
            out[32 * i:32 * (i + 1)] = blob
        return bytes(out)

    out0, out1, _, _, prep0, prep1 = _real_session(body, cheat=({"flight2": replace}, None))
    check(aborted(out1) and "COT" in out1[1] and aborted(out0),
          "corrupted base-OT seeds are caught by the extension check")

    # The peer disappears in the middle of a request.
    def party(role):
        def run(chan):
            prep = RealPrep(role, chan, SID)
            prep.setup()
            if role == 1:
                return "left", prep
            try:
                prep.and_triples(100)
            except ProtocolAbort as exc:
                return ("abort", str(exc)), prep
            return "ok", prep
        return run

    (out0, prep0), (out1, prep1), _, _ = run_two_party(party(0), party(1))
    check(aborted(out0) and after_abort(prep0), "a vanished peer aborts the session")
    _finish(start)


# --------------------------------------------------------------------------- #
# (h) Ideal preprocessing
# --------------------------------------------------------------------------- #

def test_ideal_prep() -> None:
    start = len(FAILURES)
    n = 100_000
    dealer = IdealDealer()
    d0, d1 = dealer.deltas
    delta = d0 ^ d1
    check(d0 & 1 == 1 and d1 & 1 == 0 and dealer.delta == delta,
          "ideal: lsb constraint on the key shares")

    def party(role):
        def run(chan):
            prep = IdealPrep(role, dealer)
            prep.setup()
            return (prep.random_bits(n), prep.input_masks(0, 1000), prep.input_masks(1, 1000),
                    prep.and_triples(n), prep.random_bdoz(1000), prep.delta, prep.ideal,
                    dict(prep.counters))
        return run

    r0, r1, st0, _ = run_two_party(party(0), party(1))
    check(r0[5] == d0 and r1[5] == d1 and r0[6] is True, "ideal: each party gets its key share")
    check(mac_ok(r0[0], r1[0], delta) and len(r0[0]) == n, f"ideal: MAC invariant for {n} random bits")
    x = r0[0].bits ^ r1[0].bits
    check(abs(float(x.mean()) - 0.5) < 5 * 0.5 / math.sqrt(n), "ideal: random bits look uniform")
    check(mac_ok(r0[1][0], r1[1][0], delta, r0[1][1]) and r1[1][1] is None
          and not r1[1][0].bits.any(), "ideal: input masks of P_0")
    check(mac_ok(r0[2][0], r1[2][0], delta, r1[2][1]) and r0[2][1] is None
          and not r0[2][0].bits.any(), "ideal: input masks of P_1")
    correct, macs = _triples_ok(r0[3], r1[3], delta)
    check(correct and macs, f"ideal: {n} triples are correct and authenticated")
    a, b = r0[3][0].bits ^ r1[3][0].bits, r0[3][1].bits ^ r1[3][1].bits
    check(abs(float((a & b).mean()) - 0.25) < 5 * math.sqrt(0.1875 / n), "ideal: triples look uniform")
    check(bdoz_ok(r0[4][0], r1[4][1], d1) and bdoz_ok(r1[4][0], r0[4][1], d0), "ideal: BDOZ bits")
    check(st0["bytes_sent"] == 0, "ideal: no communication")
    check(r0[7] == {"random_bits": n, "input_masks": 2000, "and_triples": n}, "ideal: counters")

    # Either party may ask first; requests must agree; the leaky AND runs on ideal bits.
    p0, p1 = IdealPrep(0, dealer), IdealPrep(1, dealer)
    s1 = p1.random_bits(50)
    s0 = p0.random_bits(50)
    check(mac_ok(s0, s1, delta), "ideal: P_1 may ask first")
    p0.random_bits(5)
    check(raises(ValueError, p1.random_bits, 6), "ideal: mismatching sizes are detected")
    p0.and_triples(5)
    check(raises(ValueError, p1.random_bits, 5), "ideal: mismatching kinds are detected")
    check(raises(ValueError, IdealPrep(0, IdealDealer()).random_bits, -1)
          and raises(ValueError, IdealPrep(0, IdealDealer()).input_masks, 2, 1)
          and raises(ValueError, IdealPrep, 2, dealer), "ideal: argument validation")
    q0 = IdealPrep(0, IdealDealer())
    q0.abort()
    check(q0.aborted and q0.delta is None and raises(ProtocolAbort, q0.random_bits, 1),
          "ideal: an aborted session hands out nothing")
    fixed = IdealDealer(random.Random(5)).deltas
    check(fixed == IdealDealer(random.Random(5)).deltas and fixed != dealer.deltas,
          "ideal: a seeded dealer is reproducible")

    # The real triple pipeline on top of the ideal bit source.
    dealer3 = IdealDealer()

    def party3(role):
        def run(chan):
            prep = IdealPrep(role, dealer3)
            return aand.and_triples(chan, SID, role, prep.delta, prep, 500, 3, batch=0, index0=0)
        return run

    t0, t1, st0, _ = run_two_party(party3(0), party3(1))
    check(all(_triples_ok(t0, t1, dealer3.delta)), "and_triples on ideal authenticated bits")
    _finish(start)


# Known answers of ``params.triple_request``: (ell, sigma) -> (ell_run, B).  The same table is
# checked by cpp/src/test_prep.cpp, so the two languages pad a request in the same way.
TRIPLE_REQUEST_KAT = {
    (1, 40): (44, 7), (2, 40): (44, 7), (18, 40): (44, 7), (33, 40): (44, 7), (43, 40): (44, 7),
    (44, 40): (44, 7), (45, 40): (45, 7), (64, 40): (64, 7), (83, 40): (83, 7), (84, 40): (84, 6),
    (100, 40): (100, 6), (320, 40): (320, 5), (3044, 40): (3044, 4), (276325, 40): (276325, 3),
    (1, 64): (26, 12), (25, 64): (26, 12), (26, 64): (26, 12), (27, 64): (27, 12),
    (30, 64): (30, 11), (50, 64): (50, 10), (200, 64): (200, 8),
    (1, 8): (153, 2), (152, 8): (153, 2), (153, 8): (153, 2), (154, 8): (154, 2),
    (1, 20): (77, 4), (1, 128): (11, 28), (1, 306): (2, 154), (1, 400): (2, 201),
}


def test_triple_request() -> None:
    """Size of the run that serves a request: at least MIN_LEAKY_TRIPLES = 306 leaky triples."""
    start = len(FAILURES)
    m = params.MIN_LEAKY_TRIPLES
    check(m == 306, "MIN_LEAKY_TRIPLES = 306")
    # 306 is the smallest L with (3/4)^L < 2^-127 (exact integers).
    check(3 ** m << 127 < 4 ** m and 3 ** (m - 1) << 127 >= 4 ** (m - 1),
          "306 is the smallest number of leaky triples with (3/4)^L < 2^-127")
    for (ell, sigma), want in TRIPLE_REQUEST_KAT.items():
        check(params.triple_request(ell, sigma) == want,
              f"triple_request({ell}, {sigma}) = {want} (got {params.triple_request(ell, sigma)})")
    check(params.triple_request(5) == params.triple_request(5, 40), "the default sigma is 40")

    for sigma in (40, 64):
        target = Fraction(1, 1 << sigma)
        prev = 0
        padded = []
        ok = True
        for ell in range(1, 401):
            run, bucket = params.triple_request(ell, sigma)
            ok = (run >= max(ell, 2) and run * bucket >= m
                  and bucket == params.bucket_size(run, sigma)
                  and params.leak_bound(run, bucket) <= target)
            # minimality: no admissible run size below the returned one
            ok = ok and all(r * params.bucket_size(r, sigma) < m for r in range(max(ell, 2), run))
            # a request is padded exactly if it is a single triple or too small for the check
            small = ell < 2 or ell * params.bucket_size(ell, sigma) < m
            ok = ok and (run != ell) == small and run >= prev
            prev = run
            if run != ell:
                padded.append(ell)
            if not ok:
                break
        check(ok, f"sigma={sigma}: run size, bucket size and minimality for ell = 1 .. 400")
        check(padded == list(range(1, len(padded) + 1)) and len(padded) == {40: 43, 64: 25}[sigma],
              f"sigma={sigma}: exactly the requests up to {len(padded)} triples are padded")
    # The session-setup circuit alone (64 AND gates for s = 64) is above the minimum.
    check(params.triple_request(64, 40) == (64, 7) and 64 * 7 == 448,
          "a request of 64 triples runs unpadded with B = 7 (448 leaky triples)")
    for ell in (1000, 68_303, 128_743, 1 << 20):
        check(params.triple_request(ell, 40) == (ell, params.bucket_size(ell, 40)),
              f"triple_request({ell}, 40) does not pad")
    check(raises(ValueError, params.triple_request, 0, 40), "triple_request rejects ell = 0")
    _finish(start)


# --------------------------------------------------------------------------- #
# (i) BoolCheck
# --------------------------------------------------------------------------- #

def _bool_check_run(n, tamper=None, extra=0, tamper_extra=None, counter=(7, 7), sids=(SID, SID)):
    """Both parties run bool_check on n opened bits (+ ``extra`` field openings).

    ``tamper(role, opened, macs)`` / ``tamper_extra(role, items)`` may modify
    one party's view before the check.  Returns the two results.
    """
    dealer = IdealDealer()

    def party(role):
        def run(chan):
            prep = IdealPrep(role, dealer)
            shares = prep.random_bits(n)
            opened = shares.bits ^ exchange_bits(chan, shares.bits)
            macs = shares.macs
            items = []
            if extra:
                rs = prep.random_bits(128 * extra)
                for k in range(extra):
                    val, mac = rs[128 * k:128 * (k + 1)].to_field()
                    reply = chan.exchange(gf.int_to_bytes(val))
                    items.append((val ^ gf.bytes_to_int(reply), mac))
            if tamper:
                opened, macs = tamper(role, opened.copy(), macs.copy())
            if tamper_extra:
                items = tamper_extra(role, list(items))
            bool_check(chan, sids[role], counter[role], role, prep.delta, opened, macs, items)
            return "ok"
        return guarded(run)

    r0, r1, st0, st1 = run_two_party(party(0), party(1))
    return r0, r1, st0, st1


def test_bool_check() -> None:
    start = len(FAILURES)
    rng = random.Random(0xB001)

    for n, extra in ((0, 0), (1, 0), (1000, 0), (0, 1), (5000, 3)):
        r0, r1, st0, st1 = _bool_check_run(n, extra=extra)
        check(r0 == "ok" and r1 == "ok",
              f"BoolCheck passes on {n} honest bits and {extra} field openings")
    check(st0["bytes_sent"] - (5000 + 7) // 8 - 3 * 16 == BOOLCHECK_BYTES,
          "BoolCheck costs 96 bytes per party, independent of the number of openings")

    # (i) A tampered opened bit: both parties hold the same wrong public value.
    for pos in (0, 499, 999):
        def flip(role, opened, macs, pos=pos):
            opened[pos] ^= 1
            return opened, macs
        r0, r1, _, _ = _bool_check_run(1000, tamper=flip)
        check(aborted(r0) and aborted(r1) and "BoolCheck failed" in r0[1],
              f"BoolCheck aborts when opened bit {pos} was tampered with")

    # One party's opened value differs from the other's (inconsistent views).
    r0, r1, _, _ = _bool_check_run(
        1000, tamper=lambda role, o, m: (o ^ (np.arange(1000) == 5).astype(np.uint8) * (role == 0), m))
    check(aborted(r0) and aborted(r1), "BoolCheck aborts when the parties' views differ")

    # A wrong MAC share, a wrong field opening, a wrong field MAC share.
    def bad_mac(role, opened, macs):
        if role == 1:
            macs[3, 1] ^= np.uint64(1 << 40)
        return opened, macs
    r0, r1, _, _ = _bool_check_run(1000, tamper=bad_mac)
    check(aborted(r0) and aborted(r1), "BoolCheck aborts on a wrong MAC share")
    r0, r1, _, _ = _bool_check_run(
        10, extra=2, tamper_extra=lambda role, it: [it[0], (it[1][0] ^ 1, it[1][1])])
    check(aborted(r0) and aborted(r1), "BoolCheck aborts on a tampered field opening")
    r0, r1, _, _ = _bool_check_run(
        10, extra=2, tamper_extra=lambda role, it: it if role else [(it[0][0], it[0][1] ^ 2), it[1]])
    check(aborted(r0) and aborted(r1), "BoolCheck aborts on a wrong MAC share of a field opening")
    r0, r1, _, _ = _bool_check_run(10, extra=2, tamper_extra=lambda role, it: it[::-1] if role else it)
    check(aborted(r0) and aborted(r1), "BoolCheck binds the order of the openings")

    # Counter and sid are bound (a digest of another check or session cannot be reused).
    r0, r1, _, _ = _bool_check_run(100, counter=(7, 8))
    check(aborted(r0) and aborted(r1), "BoolCheck: different counters do not match")
    r0, r1, _, _ = _bool_check_run(100, sids=(SID, SID + b"'"))
    check(aborted(r0) and aborted(r1), "BoolCheck: different session identifiers do not match")

    # The digest is what the documentation says.
    d0, d1 = test_deltas(rng)
    opened = rand_bits(rng, 50)
    m0 = rand_vec(rng, 50)
    m1 = m0 ^ gf.bit_mul(opened, d0 ^ d1)
    w, big_m0 = rng.getrandbits(128), rng.getrandbits(128)
    big_m1 = big_m0 ^ gf.mul(w, d0 ^ d1)
    v0 = check_values(d0, opened, m0)
    check(np.array_equal(v0, check_values(d1, opened, m1)), "V_j agrees for correctly opened bits")
    check(np.array_equal(v0, m0 ^ gf.bit_mul(opened, d0)), "V_j = m_j XOR y_j Delta_b")
    dig0 = check_digest(SID, 3, d0, opened, m0, [(w, big_m0)])
    check(dig0 == check_digest(SID, 3, d1, opened, m1, [(w, big_m1)]), "the two digests agree")
    manual = hs.sha256(b"Couplet-boolcheck", SID, (3).to_bytes(8, "little"), gf.vec_to_bytes(v0),
                       gf.int_to_bytes(big_m0 ^ gf.mul(w, d0)))
    check(dig0 == manual, "digest = SHA-256(tag, sid, counter, V bits, V field) with length framing")
    check(dig0 != check_digest(SID, 4, d0, opened, m0, [(w, big_m0)])
          and dig0 != check_digest(SID, 3, d0, opened, m0, [(w ^ 1, big_m0)])
          and dig0 != check_digest(SID, 3, d0, opened, m0), "the digest depends on every input")
    flipped = opened.copy()
    flipped[9] ^= 1
    gap = check_values(d0, flipped, m0) ^ check_values(d1, flipped, m1)
    check(gf.arr_to_int(gap[9]) == d0 ^ d1 and not np.delete(gap, 9, axis=0).any(),
          "a wrongly opened bit shifts V_j by Delta^B")

    # Replaying the honest party's commitment and opening (the "mirror" deviation) fails.
    dealer = IdealDealer()

    def honest(chan):
        prep = IdealPrep(0, dealer)
        s = prep.random_bits(20)
        opened = s.bits ^ exchange_bits(chan, s.bits)   # the adversary lied about bit 4
        bool_check(chan, SID, 0, 0, prep.delta, opened, s.macs)
        return "ok"

    def mirror(chan):
        prep = IdealPrep(1, dealer)
        s = prep.random_bits(20)
        bits = s.bits.copy()
        bits[4] ^= 1
        exchange_bits(chan, bits)
        com = chan.recv()                                # rushing: read the commitment first
        chan.send(com)
        try:
            chan.send(chan.recv())
        except ProtocolAbort:
            return "caught"
        return "sent"

    r0, r1, _, _ = run_two_party(guarded(honest), guarded(mirror))
    check(aborted(r0) and "mirrored" in r0[1], "BoolCheck: a replayed commitment is rejected")

    # Argument validation.
    c0, c1 = loopback_pair()
    check(raises(ValueError, bool_check, c0, SID, 0, 1, d1, opened, m1), "role must match the endpoint")
    check(raises(ValueError, bool_check, c0, SID, 0, 0, d0, opened, m0[:5]), "one MAC per opened bit")
    c0.close()
    c1.close()
    _finish(start)


# --------------------------------------------------------------------------- #
# Timing of the large batch (printed; one correctness check)
# --------------------------------------------------------------------------- #

def test_large_batch() -> None:
    start = len(FAILURES)
    if "--no-large" in sys.argv or os.environ.get("COUPLET_NO_LARGE"):
        print("  n = 300000 run skipped on request")
        return
    n = 300_000
    out0, out1, st0, st1, prep0, prep1, wall = _measure_triples(n)
    (t0, _, delta0), (t1, _, delta1) = out0, out1
    correct, macs = _triples_ok(t0, t1, delta0 ^ delta1)
    check(correct and macs, f"RealPrep n = {n}: triples are correct and authenticated")
    check(prep0.last_bucket_size == 3 == params.bucket_size(n, 40), f"RealPrep n = {n}: B = 3")
    expected = _and_bytes_expected(n, 3)
    # the extension runs in chunks of 2^20 rows: one seed and one check message per chunk
    rows = 2 * 3 * n + PAD
    chunks = [min(1 << 20, rows - a) for a in range(0, rows, 1 << 20)]
    expected["cotext"] = sum(u_message_bytes(c) + 16 + COT_CHECK_BYTES for c in chunks)
    got = {p: st0["phases"][p]["bytes_sent"] for p in expected}
    check(got == expected, f"RealPrep n = {n}: bytes per phase are exact ({got} vs {expected})")
    _report_triples(n, out0, st0, st1, prep0, wall)
    _finish(start)


TESTS = [
    test_params,
    test_triple_request,
    test_shares_model,
    test_base_ot,
    test_cot,
    test_abits,
    test_leaky_and,
    test_leaky_and_attack,
    test_bucketing,
    test_real_prep,
    test_small_requests,
    test_real_prep_aborts,
    test_ideal_prep,
    test_bool_check,
    test_large_batch,
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
    print(f"test_prep: {CHECKS} checks, {len(FAILURES)} failures, "
          f"{time.perf_counter() - t_all:.1f} s")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
