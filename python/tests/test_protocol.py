"""Tests for the session layer and Pi_Gen (couplet.session, couplet.protocol;
DESIGN.md sections 4, 5.2, 5.4, 6; changes C, G, H).

Everything is checked on the TWO PARTIES' ACTUAL OUTPUTS of a protocol run:

  * session setup: Delta_b uniform in [0, 2^s); the persistent wires hold
    Delta_0 + Delta_1 as an (s+1)-bit integer (not reduced); state machine;
  * Pi_Gen for DPF and DCF, n in {4, 8}, k in {32, 64}, both index modes, both
    payload modes, batch in {1, 3}, ideal preprocessing: both keys carry the
    same correction words, equal ``fss.dpf_gen`` / ``dcf_gen`` run on the
    reconstructed (alpha, beta, Delta) with the keys' own roots bit for bit,
    and satisfy full-domain correctness on the value plane and on the tag
    plane with the session ``Delta = Delta_0 + Delta_1`` (one session with
    Delta >= 2^s, one below); the release identity zeta_0 + zeta_1 = 0 of
    section 2.3 holds with the parties' own Delta_b;
  * gate mode: the invariants of tests/test_circuits.py (offset sharing,
    DPF / DCF indices, [[w_c]], [[r w_c]], output masks, table, comparison and
    interval identities for all x) from the two parties' ``GateMaterial``;
  * ``gen_batch``: different keys and gates from one circuit execution;
  * abort semantics (change G) and argument validation;
  * triple budget: the helpers of ``couplet.protocol`` compute the AND budget
    of a planned list of generation calls from the circuit statistics; a
    session that reserves it at ``setup`` makes ONE triple request (one
    bucketing population) and uses every triple; in strict mode
    (``allow_refill=False``) a short pool is an error before any message, in
    refill mode it is one more counted request;
  * real preprocessing: DPF n = 4 and n = 8 with public payload and a DCF with
    n = 4 in one session whose triples form ONE bucketing population; a
    batch of 3; and DPF n = 16 with public payload, for which time, bytes per
    phase and rounds are printed.

Two parties are run with ``couplet.net.run_two_party`` (two threads of this
process), so all times are wall-clock times for BOTH parties on one machine.

Run as a script from artifact/python:  python tests/test_protocol.py
Prints the number of checks and exits non-zero on any failure.  Options:
``-k name`` selects test functions, ``--no-real`` skips the real
preprocessing, ``--no-large`` skips the n = 16 run.
"""

from __future__ import annotations

import json
import os
import pickle
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

from couplet import circuits_gen as cg  # noqa: E402
from couplet import fss  # noqa: E402
from couplet import gf128 as gf  # noqa: E402
from couplet import protocol as pg  # noqa: E402
from couplet.abb import TripleBudgetError, bits_to_ints  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.prep import IdealDealer, bucket_size  # noqa: E402
from couplet.session import ABORTED, ACTIVE, Session  # noqa: E402

NO_REAL = "--no-real" in sys.argv
NO_LARGE = "--no-large" in sys.argv or NO_REAL
S = 64
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


def run_sessions(mode: str, script, k: int = 64, cheats=(None, None), delta_shares=(None, None),
                 reserve: int = 0, setup: bool = True, setup_adder: str = "ripple",
                 outmasks: int = 0, allow_refill: bool = True):
    """Run ``script(role, session)`` at both parties after ``session.setup(reserve, outmasks)``.

    ``allow_refill=False`` is the strict mode: the script must fit into the
    reservation (one bucketing population).

    Returns ``(out0, out1, stats0, stats1, box)``; ``out_b`` is ``("ok", value)``
    or ``("abort", message)``; ``box["ses"]`` are the two sessions,
    ``box["delta"]`` the arithmetic key shares Delta_b captured after setup,
    ``box["dealer"]`` the ideal dealer.
    """
    dealer = IdealDealer() if mode == "ideal" else None
    sid = b"test-protocol-" + secrets.token_bytes(8)
    box = {"ses": [None, None], "delta": [None, None], "dealer": dealer, "setup_s": [None, None]}

    def party(role):
        def fn(chan):
            ses = Session(role, chan, sid, k=k, s=S, prep=mode, dealer=dealer, cheat=cheats[role],
                          delta_share=delta_shares[role], setup_adder=setup_adder,
                          allow_refill=allow_refill)
            box["ses"][role] = ses
            try:
                if setup:
                    t = time.perf_counter()
                    ses.setup(reserve, outmasks)
                    box["setup_s"][role] = time.perf_counter() - t
                    box["delta"][role] = ses.delta_share
                return "ok", script(role, ses)
            except ProtocolAbort as exc:
                return "abort", str(exc)
        return fn

    out0, out1, st0, st1 = run_two_party(party(0), party(1), timeout=900)
    return out0, out1, st0, st1, box


def run_honest(mode: str, script, **kwargs):
    out0, out1, st0, st1, box = run_sessions(mode, script, **kwargs)
    check(out0[0] == "ok" and out1[0] == "ok",
          f"honest run aborted: {out0[1] if out0[0] != 'ok' else out1[1]}")
    return out0[1], out1[1], st0, st1, box


def public_part(key) -> dict:
    d = dict(key.__dict__)
    d.pop("b")
    d.pop("seed")
    return d


def combine(index_mode: str, a0: int, a1: int, n: int) -> int:
    return a0 ^ a1 if index_mode == "xor" else (a0 + a1) % (1 << n)


def check_key_pair(family: str, k0, k1, n: int, k: int, alpha: int, beta: int, d0: int, d1: int,
                   tag: str, rng: random.Random):
    """All properties of one generated key pair; returns the full-domain share vectors."""
    delta = d0 + d1                      # the session MAC key, NOT reduced mod 2^s
    mod = 1 << (k + S)
    N = 1 << n
    cls = fss.DPFKey if family == "dpf" else fss.DCFKey
    check(isinstance(k0, cls) and isinstance(k1, cls), f"{tag}: key types")
    check_eq([(key.b, key.n, key.k, key.s) for key in (k0, k1)], [(0, n, k, S), (1, n, k, S)],
             f"{tag}: key parameters")
    check_eq(public_part(k0), public_part(k1), f"{tag}: both parties hold the same correction words")
    check(k0.seed != k1.seed, f"{tag}: the roots differ")
    gen, ev, point = ((fss.dpf_gen, fss.dpf_eval_all, fss.dpf_eval) if family == "dpf"
                      else (fss.dcf_gen, fss.dcf_eval_all, fss.dcf_eval))
    e0, e1 = gen(alpha, beta, delta, n, k, S, roots=(k0.seed, k1.seed))
    check(e0.to_bytes() == k0.to_bytes() and e1.to_bytes() == k1.to_bytes(),
          f"{tag}: keys equal fss.{family}_gen(alpha, beta, Delta_0 + Delta_1) bit for bit")
    y0, t0 = ev(0, k0)
    y1, t1 = ev(1, k1)
    val = [(int(a) + int(b)) % mod for a, b in zip(y0, y1)]
    tg = [(int(a) + int(b)) % mod for a, b in zip(t0, t1)]
    ind = [int(x == alpha) if family == "dpf" else int(x < alpha) for x in range(N)]
    check_eq(val, [beta * i for i in ind], f"{tag}: value plane on the whole domain")
    check_eq(tg, [delta * beta * i % mod for i in ind], f"{tag}: tag plane on the whole domain (unreduced Delta)")
    reduced = delta % (1 << S)           # only meaningful if Delta >= 2^s and the function is not zero
    check(not (delta >> S and beta and any(ind)) or tg != [reduced * beta * i % mod for i in ind],
          f"{tag}: Delta mod 2^s would give another tag plane")
    # release identity (section 2.3) with each party's own Delta_b: single openings and one batch
    for x in (alpha, 0, N - 1, rng.randrange(N)):
        chi = rng.getrandbits(S)
        y = chi * val[x]
        zeta0 = (chi * int(t0[x]) - d0 * y) % mod
        zeta1 = (chi * int(t1[x]) - d1 * y) % mod
        check((zeta0 + zeta1) % mod == 0, f"{tag}: release identity zeta_0 + zeta_1 = 0 at x = {x}")
        check(point(0, k0, x) == (int(y0[x]), int(t0[x])) and point(1, k1, x) == (int(y1[x]), int(t1[x])),
              f"{tag}: Eval equals EvalAll at x = {x}")
    chis = [rng.getrandbits(S) for _ in range(N)]
    y = sum(c * v for c, v in zip(chis, val))
    zeta0 = (sum(c * int(t) for c, t in zip(chis, t0)) - d0 * y) % mod
    zeta1 = (sum(c * int(t) for c, t in zip(chis, t1)) - d1 * y) % mod
    check((zeta0 + zeta1) % mod == 0, f"{tag}: batched release identity over the whole domain")
    for key in (k0, k1):
        check(fss.key_from_bytes(key.to_bytes()) == key and pickle.loads(pickle.dumps(key)) == key,
              f"{tag}: key serialisation round trip")
    return (y0, t0), (y1, t1)


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #

def test_session_setup_and_state():
    top = (1 << S) - 1
    for d_fixed in ((top, top), (None, None), (0, 0), (1 << (S - 1), 1 << (S - 1))):
        def script(role, ses):
            ids = [ses.next_id("kid"), ses.next_id("kid"), ses.next_id("gid"), ses.next_id("kid")]
            errors = 0
            for call in (lambda: ses.setup(), lambda: ses.next_id("zid"), lambda: ses.reserve_triples(-1)):
                try:
                    call()
                except ValueError:
                    errors += 1
            return ids, errors, ses.delta_wires, ses.info(), dict(ses.setup_stats)

        r0, r1, st0, st1, box = run_honest("ideal", script, delta_shares=d_fixed)
        ses0, ses1 = box["ses"]
        d0, d1 = box["delta"]
        tag = f"setup {'fixed' if d_fixed[0] is not None else 'random'}"
        check(0 <= d0 < (1 << S) and 0 <= d1 < (1 << S), f"{tag}: Delta_b in [0, 2^s)")
        if d_fixed[0] is not None:
            check_eq((d0, d1), d_fixed, f"{tag}: explicit shares are used")
        w0, w1 = r0[2], r1[2]
        check_eq((w0.width, w0.batch), (S + 1, 1), f"{tag}: s + 1 persistent wires")
        check_eq(bits_to_ints(w0.bits ^ w1.bits), [d0 + d1], f"{tag}: the wires hold Delta_0 + Delta_1, not reduced")
        macs = (w0.macs ^ w1.macs).reshape(-1, 2)
        check(np.array_equal(macs, gf.bit_mul((w0.bits ^ w1.bits).reshape(-1), box["dealer"].delta)),
              f"{tag}: the wires are authenticated")
        check_eq((r0[0], r0[1]), ([0, 1, 0, 2], 3), f"{tag}: monotone identifiers; argument errors")
        check_eq((ses0.state, ses1.state, ses0.is_setup), (ACTIVE, ACTIVE, True), f"{tag}: state")
        info = r0[3]
        check_eq((info["evaluations"], info["triples_used"], info["triple_requests"], info["prep"]),
                 (1, S, 1, "ideal"), f"{tag}: one evaluation, s triples, one request")
        check_eq(r0[4]["rounds"], 1 + S + 2, f"{tag}: rounds of the setup evaluation")
        check_eq(sorted(st0["phases"]), ["setup-input", "setup-online", "setup-outcheck", "setup-prep"],
                 f"{tag}: setup phases")
    NOTES.append("session setup: Delta wires verified for Delta in {2^(s+1) - 2, random, 0, 2^s}")

    # parallel-prefix adder for the setup circuit: fewer rounds, the same wires
    sk = cg.build_session_setup(S, adder="sklansky")

    def script_sk(role, ses):
        return (ses.delta_wires, dict(ses.setup_stats), pg.gen_dpf(ses, 2, "xor", "public_one", role),
                ses.info())

    r0, r1, _, _, box = run_honest("ideal", script_sk, k=32, delta_shares=(top, 1), setup_adder="sklansky",
                                   reserve=pg.and_budget([pg.planned_key("dpf", 2, 32, S)]), allow_refill=False)
    check_eq((r0[3]["triple_requests"], r0[3]["triples_unused"], r0[3]["triples_requested"]),
             (1, 0, pg.session_triples([pg.planned_key("dpf", 2, 32, S)], S, "sklansky")),
             "sklansky setup: one triple request for the setup circuit and the key")
    check_eq(bits_to_ints(r0[0].bits ^ r1[0].bits), [top + 1], "sklansky setup: Delta = 2^s exactly")
    check_eq((r0[1]["rounds"], r0[1]["n_and"]), (sk.and_depth() + 3, sk.stats()["n_and"]),
             "sklansky setup: rounds = AND depth + 3")
    check(sk.and_depth() < 16 < S, "sklansky setup: logarithmic depth")
    check_key_pair("dpf", r0[2][0], r1[2][0], 2, 32, 1, 1, top, 1, "key after a sklansky setup",
                   random.Random("sklansky"))
    NOTES.append(f"session setup rounds: {S + 3} with the ripple adder ({S} AND), {sk.and_depth() + 3} with "
                 f"the Sklansky adder ({sk.stats()['n_and']} AND)")

    # samples of Delta_b: uniform s-bit values (top bit and bit 0 both occur)
    seen_top, seen_low = set(), set()
    for _ in range(24):
        _, _, _, _, box = run_honest("ideal", lambda role, ses: None)
        for d in box["delta"]:
            seen_top.add(d >> (S - 1))
            seen_low.add(d & 1)
    check(seen_top == {0, 1} and seen_low == {0, 1}, "sampled Delta_b cover the s-bit range")

    # before setup / constructor arguments
    def no_setup(role, ses):
        n = 0
        for call in (lambda: ses.evaluate(cg.build_session_setup(S), {"delta": 1}),
                     lambda: pg.gen_dpf(ses, 2, "xor", "public_one", 0)):
            try:
                call()
            except ValueError:
                n += 1
        return n, ses.state

    r0, r1, _, _, _ = run_honest("ideal", no_setup, setup=False)
    check_eq((r0, r1), ((2, ACTIVE), (2, ACTIVE)), "generation before setup() is a ValueError, not an abort")

    class _Chan:
        party = 0
        phase = "x"

    dealer = IdealDealer()
    check_raises(ValueError, Session, 2, _Chan(), b"s", prep="ideal", dealer=dealer)
    check_raises(ValueError, Session, 1, _Chan(), b"s", prep="ideal", dealer=dealer)
    check_raises(ValueError, Session, 0, _Chan(), "sid", prep="ideal", dealer=dealer)
    check_raises(ValueError, Session, 0, _Chan(), b"s", k=128, s=64, prep="ideal", dealer=dealer)
    check_raises(ValueError, Session, 0, _Chan(), b"s", prep="ideal")
    check_raises(ValueError, Session, 0, _Chan(), b"s", prep="real", dealer=dealer)
    check_raises(ValueError, Session, 0, _Chan(), b"s", prep="other")
    check_raises(ValueError, Session, 0, _Chan(), b"s", prep="ideal", dealer=dealer, delta_share=1 << S)
    check_raises(ValueError, Session, 0, _Chan(), b"s", prep="ideal", dealer=dealer, sigma=0)
    check_raises(ValueError, Session, 0, _Chan(), b"s", prep="ideal", dealer=dealer, setup_adder="kogge")


# --------------------------------------------------------------------------- #
# Pi_Gen for DPF / DCF
# --------------------------------------------------------------------------- #

def _shares(rng: random.Random, n: int, k: int, index_mode: str, beta_mode: str, batch: int):
    """Shares of (alpha, beta) for ``batch`` instances with boundary cases in the first ones."""
    N, K = 1 << n, 1 << k
    alpha = [[rng.randrange(N) for _ in range(batch)] for _ in range(2)]
    beta = [[rng.randrange(K) for _ in range(batch)] for _ in range(2)]
    if batch >= 3:
        alpha[1][0] = alpha[0][0] if index_mode == "xor" else (N - alpha[0][0]) % N     # alpha = 0
        alpha[0][1] = alpha[1][1] = N - 1                                               # wrap / alpha = 0 (xor)
        beta[0][0] = beta[1][0] = K - 1                                                 # beta = 2^k - 2 (wrap)
        beta[1][1] = (K - beta[0][1]) % K                                               # beta = 0
    if beta_mode != "secret":
        beta = [None, None]
    return alpha, beta


def _keygen_jobs(rng: random.Random, k: int, families, ns, batches):
    jobs = []
    for family in families:
        for n in ns:
            for index_mode in ("xor", "add"):
                for beta_mode in ("secret", "public_one"):
                    for batch in batches:
                        alpha, beta = _shares(rng, n, k, index_mode, beta_mode, batch)
                        jobs.append((family, n, index_mode, beta_mode, batch, alpha, beta))
    return jobs


def _keygen_budget(k: int, jobs) -> int:
    """AND triples of the generation calls of ``jobs`` (the reservation of a strict session)."""
    return pg.and_budget([pg.planned_key(family, n, k, S, index_mode, beta_mode, batch)
                          for family, n, index_mode, beta_mode, batch, _, _ in jobs])


def _run_keygen_jobs(mode: str, k: int, jobs, **kwargs):
    def script(role, ses):
        out = []
        for family, n, index_mode, beta_mode, batch, alpha, beta in jobs:
            gen = pg.gen_dpf if family == "dpf" else pg.gen_dcf
            keys = gen(ses, n, index_mode, beta_mode, alpha[role],
                       None if beta[role] is None else beta[role], batch)
            out.append((keys, dict(ses.last_gen)))
        return out, ses.info()

    return run_honest(mode, script, k=k, **kwargs)


def _published_rows() -> dict:
    """Rows of experiments/figdata/circuit_stats.json for the key circuits (ripple adders)."""
    path = os.path.normpath(os.path.join(_PY, "..", "experiments", "figdata", "circuit_stats.json"))
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except OSError:
        return {}
    return {(r["kind"], r["n"], r["k"], r["s"], r["index_mode"], r["beta_mode"]): r
            for r in doc.get("keys", []) if r.get("adder") == "ripple"}


PUBLISHED = _published_rows()
PUBLISHED_HITS = [0]


def _check_keygen_jobs(mode: str, k: int, jobs, r0, r1, box, rng) -> int:
    d0, d1 = box["delta"]
    n_keys = 0
    seeds = set()
    for (family, n, index_mode, beta_mode, batch, alpha, beta), (keys0, gen0), (keys1, gen1) in zip(jobs, r0[0], r1[0]):
        tag = f"{mode} {family} n={n} k={k} {index_mode} {beta_mode} batch={batch}"
        check_eq((len(keys0), len(keys1)), (batch, batch), f"{tag}: one key per instance")
        st0, st1 = gen0["stats"], gen1["stats"]
        check_eq(gen0["digest"], gen1["digest"], f"{tag}: both parties ran the same circuit")
        check_eq(st0["triples"], st0["n_and"] * batch, f"{tag}: n_and * batch triples")
        make = cg.dpf_descriptor if family == "dpf" else cg.dcf_descriptor
        row = PUBLISHED.get((family, n, k, S, index_mode, beta_mode))
        if row is not None:      # the circuit table exported by the circuit compiler
            check_eq((gen0["digest"], st0["n_and"], st0["and_depth"]),
                     (row["digest"], row["n_and"], row["and_depth"]), f"{tag}: digest and size as published")
            PUBLISHED_HITS[0] += 1
        elif family == "dpf" and n <= 4:      # small circuit: rebuild it independently
            fresh = cg.build(make(n, k, S, index_mode, beta_mode))
            check_eq((gen0["digest"], st0["n_and"], st0["and_depth"]),
                     (fresh.digest(), fresh.stats()["n_and"], fresh.and_depth()), f"{tag}: digest and size")
        if mode == "ideal":
            # rounds: inputs + AND depth + check, open, check; shared by all instances of the batch
            check_eq((st0["rounds"], st1["rounds"]), (st0["and_depth"] + 6,) * 2,
                     f"{tag}: rounds = AND depth + 6, independent of the batch size")
        for j in range(batch):
            a = combine(index_mode, alpha[0][j], alpha[1][j], n)
            b = 1 if beta[0] is None else (beta[0][j] + beta[1][j]) % (1 << k)
            check_key_pair(family, keys0[j], keys1[j], n, k, a, b, d0, d1, f"{tag} instance {j}", rng)
            seeds.update((keys0[j].seed, keys1[j].seed))
            n_keys += 1
    check_eq(len(seeds), 2 * n_keys, f"{mode} k={k}: all roots of all instances are distinct")
    return n_keys


def test_keygen_ideal():
    top = (1 << S) - 1
    half = 1 << (S - 1)
    for k, d_fixed in ((32, (top, top - 2)), (64, (half - 1, half - 9))):
        rng = random.Random(f"keygen-ideal-{k}")
        jobs = _keygen_jobs(rng, k, ("dpf", "dcf"), (4, 8), (1, 3))
        t = time.perf_counter()
        budget = _keygen_budget(k, jobs)
        r0, r1, st0, st1, box = _run_keygen_jobs("ideal", k, jobs, delta_shares=d_fixed, reserve=budget,
                                                 allow_refill=False)
        run_s = time.perf_counter() - t
        delta = box["delta"][0] + box["delta"][1]
        check_eq(delta >> S, 1 if k == 32 else 0, f"k={k}: Delta on the intended side of 2^s")
        n_keys = _check_keygen_jobs("ideal", k, jobs, r0, r1, box, rng)
        for st in (st0, st1):
            check_eq(sum(ph["bytes_sent"] for ph in st["phases"].values()), st["bytes_sent"],
                     f"k={k}: per-phase bytes sum to the total")
        check_eq((box["ses"][0].state, box["ses"][1].state), (ACTIVE, ACTIVE), f"k={k}: sessions stay active")
        info = r0[1]
        check_eq((info["triple_requests"], info["triples_unused"], info["triples_used"]), (1, 0, S + budget),
                 f"k={k}: ONE triple request for the whole session, reserved exactly (strict mode)")
        check_eq(r1[1]["triple_requests"], 1, f"k={k}: one triple request at P1 as well")
        NOTES.append(f"keygen ideal k={k}: {len(jobs)} generations, {n_keys} key pairs, "
                     f"Delta {'>=' if delta >> S else '<'} 2^s, {info['triples_used']:,} triples, "
                     f"{st0['rounds']} rounds, {st0['bytes_sent']:,} bytes sent by P0, {run_s:.1f} s")
    check(not PUBLISHED or PUBLISHED_HITS[0] >= 16, "circuits of the published table were compared")
    NOTES.append(f"keygen ideal: {PUBLISHED_HITS[0]} generations ran a circuit whose digest, AND count and depth "
                 f"equal the row of experiments/figdata/circuit_stats.json")


# --------------------------------------------------------------------------- #
# Gate mode
# --------------------------------------------------------------------------- #

def check_gate_pair(desc: dict, g0, g1, d0: int, d1: int, tag: str, rng: random.Random) -> dict:
    """The invariants of tests/test_circuits.py::_check_gate on the two parties' material."""
    n, k = desc["n"], desc["k"]
    w = k + S
    N, mod = 1 << n, 1 << w
    delta = d0 + d1
    thresholds = [(c, p) for c, p in desc["thresholds"]]
    for b, g in ((0, g0), (1, g1)):
        check(isinstance(g, pg.GateMaterial) and g.role == b and g.descriptor == desc,
              f"{tag}: material of P{b} carries its role and the descriptor")
        check_eq((g.n, g.k, g.s, g.width, g.auth_input, g.thresholds),
                 (n, k, S, w, desc["auth_input"], thresholds), f"{tag}: material properties")
        check_eq((len(g.dcf_keys), len(g.w_shares), len(g.rw_shares), len(g.outmask_shares)),
                 (len(thresholds),) * 3 + (desc["n_outmask"],), f"{tag}: one entry per threshold / mask")
        again = pg.GateMaterial.from_bytes(g.to_bytes())
        check(again == g and again.to_bytes() == g.to_bytes() and pickle.loads(pickle.dumps(g)) == g
              and g.size_bytes() == len(g.to_bytes()), f"{tag}: serialisation round trip of P{b}")
    # 1. offset: r' in Z_{2^(n+s)} (authenticated input) or r in Z_{2^n}
    m = n + S if desc["auth_input"] else n
    check_eq((g0.rp_bits, g1.rp_bits), (m, m), f"{tag}: modulus bits of the offset sharing")
    rp = (g0.rp_share + g1.rp_share) % (1 << m)
    r = rp % N
    if desc["auth_input"]:
        check_eq((g0.rp_tag_share + g1.rp_tag_share) % (1 << m), delta * rp % (1 << m),
                 f"{tag}: [[r']] tag shares with the unreduced Delta")
    else:
        check(g0.rp_tag_share is None and g1.rp_tag_share is None, f"{tag}: no tag without auth_input")
    # 2. DPF at index r with payload 1; table gate for all x
    if desc["dpf_one"]:
        (y0, t0), (y1, t1) = check_key_pair("dpf", g0.dpf_key, g1.dpf_key, n, k, r, 1, d0, d1,
                                            f"{tag} DPF at r", rng)
        table = [rng.getrandbits(k) for _ in range(N)]
        ok = True
        for x in range(N):
            x_hat = (x + r) % N
            val = sum(table[(x_hat - u) % N] * (int(y0[u]) + int(y1[u])) for u in range(N)) % mod
            tg = sum(table[(x_hat - u) % N] * (int(t0[u]) + int(t1[u])) for u in range(N)) % mod
            ok &= val == table[x] and tg == delta * table[x] % mod
        check(ok, f"{tag}: table gate T[x] for all x (value and tag)")
    else:
        check(g0.dpf_key is None and g1.dpf_key is None, f"{tag}: no DPF key")
    # 3. thresholds: DCF index (r + c) mod N, payload, [[w_c]], [[r w_c]]
    rec = {}
    for i, (c, payload) in enumerate(thresholds):
        alpha_c, w_c = (r + c) % N, int(r + c >= N)
        beta = 1 if payload == "one" else r
        shares = check_key_pair("dcf", g0.dcf_keys[i], g1.dcf_keys[i], n, k, alpha_c, beta, d0, d1,
                                f"{tag} DCF {i} (c={c}, payload {payload})", rng)
        w_sh = (g0.w_shares[i], g1.w_shares[i])
        check_eq((w_sh[0][0] + w_sh[1][0]) % mod, w_c, f"{tag}: [[w_c]] value, c={c}")
        check_eq((w_sh[0][1] + w_sh[1][1]) % mod, delta * w_c % mod, f"{tag}: [[w_c]] tag, c={c}")
        rw_sh = None
        if payload == "r":
            rw_sh = (g0.rw_shares[i], g1.rw_shares[i])
            check_eq((rw_sh[0][0] + rw_sh[1][0]) % mod, r * w_c, f"{tag}: [[r w_c]] value, c={c}")
            check_eq((rw_sh[0][1] + rw_sh[1][1]) % mod, delta * r * w_c % mod, f"{tag}: [[r w_c]] tag, c={c}")
        else:
            check(g0.rw_shares[i] is None and g1.rw_shares[i] is None, f"{tag}: no [[r w_c]] for payload one")
        rec[(c, payload)] = (shares, w_sh, rw_sh)
    # 4. comparison identity of section 6, evaluated locally by each party, for all x
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
            val = tg = 0
            for b in (0, 1):
                val += int(shares_c[b][0][x_hat]) - int(shares_0[b][0][x_hat]) + corr[b][0]
                tg += int(shares_c[b][1][x_hat]) - int(shares_0[b][1][x_hat]) + corr[b][1]
            ok &= val % mod == scale * lt and tg % mod == delta * scale * lt % mod
            if payload == "r":                       # ReLU-style: x_hat b - r b = x b mod 2^n
                ok &= (x_hat * lt - val) % N == x * lt
        check(ok, f"{tag}: comparison identity [x < {c}] for all x, payload {payload}")
    # 4b. interval identity [lo <= x < hi] = D_hi - D_lo + w_hi - w_lo
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
    # 5. output masks: value 2^k rho with rho < 2^s, tag Delta 2^k rho mod 2^w
    rhos = []
    for j in range(desc["n_outmask"]):
        (v0, m0), (v1, m1) = g0.outmask_shares[j], g1.outmask_shares[j]
        check(all(x < mod and x % (1 << k) == 0 for x in (v0, m0, v1, m1)),
              f"{tag}: mask shares are multiples of 2^k below 2^w")
        value = (v0 + v1) % mod
        check_eq((m0 + m1) % mod, delta * value % mod, f"{tag}: output mask tag Delta 2^k rho")
        rhos.append(value >> k)
    return {"rp": rp, "r": r, "rhos": rhos}


def test_gate_ideal():
    cases = [
        (32, cg.gate_descriptor(3, 32, S, True, True, [(0, "one"), (3, "one"), (8, "one"), (0, "r"), (4, "r")], 2), 2),
        (64, cg.gate_descriptor(4, 64, S, False, True, [(5, "one"), (0, "one")], 0), 1),      # table + comparison
        (64, cg.gate_preset("relu_large", 3, 64, S), 2),
        (32, cg.gate_preset("table", 4, 32, S), 3),
        (32, cg.gate_preset("interval_auth", 4, 32, S), 1),
        (64, cg.gate_preset("outmask_only", 5, 64, S), 3),
    ]
    rng = random.Random("gate-ideal")
    for k in (32, 64):
        mine = [(desc, batch) for kk, desc, batch in cases if kk == k]

        def script(role, ses):
            out = []
            for desc, batch in mine:
                out.append((pg.gen_gate(ses, desc, batch), dict(ses.last_gen)))
            return out

        t = time.perf_counter()
        r0, r1, st0, _, box = run_honest("ideal", script, k=k, allow_refill=False,
                                         reserve=pg.and_budget([pg.planned_gate(d, b) for d, b in mine]))
        d0, d1 = box["delta"]
        check_eq([(ses.triple_requests, ses.engine.triples_available()) for ses in box["ses"]], [(1, 0)] * 2,
                 f"gate mode k={k}: ONE triple request, reserved exactly")
        offsets, rho_all = [], []
        for (desc, batch), (mats0, gen0), (mats1, _) in zip(mine, r0, r1):
            name = cg.descriptor_name(desc)
            check_eq((len(mats0), len(mats1)), (batch, batch), f"{name}: one material per instance")
            circuit = pg.generation_circuit(desc)
            check_eq(gen0["stats"]["rounds"], circuit.and_depth() + 6, f"{name}: rounds = AND depth + 6")
            for j in range(batch):
                got = check_gate_pair(desc, mats0[j], mats1[j], d0, d1, f"{name} instance {j}", rng)
                if desc["auth_input"]:
                    offsets.append(got["rp"])
                rho_all += got["rhos"]
            sizes = (mats0[0].size_bytes(), mats1[0].size_bytes())
            check_eq(sizes[0], sizes[1], f"{name}: both parties' material has the same size")
            NOTES.append(f"gate {name} x{batch}: {circuit.stats()['n_and']:,} AND, depth {circuit.and_depth()}, "
                         f"material {sizes[0]} bytes per party")
        check_eq(len(set(offsets)), len(offsets), f"k={k}: the (n+s)-bit offsets r' of all instances differ")
        check_eq(len(set(rho_all)), len(rho_all), f"k={k}: all output masks differ")
        NOTES.append(f"gate mode k={k}: {st0['rounds']} rounds, {st0['bytes_sent']:,} bytes sent by P0, "
                     f"{time.perf_counter() - t:.1f} s")

    # encoding errors
    g = r0[0][0][0]
    data = g.to_bytes()
    check_raises(ValueError, pg.GateMaterial.from_bytes, data[:-1])
    check_raises(ValueError, pg.GateMaterial.from_bytes, data + b"\x00")
    check_raises(ValueError, pg.GateMaterial.from_bytes, b"XXXXXXXX" + data[8:])
    check_raises(ValueError, pg.GateMaterial.from_bytes, data[:8] + b"\x02" + data[9:])
    check_raises(ValueError, pg.GateMaterial.from_bytes, b"")


def test_gen_batch_ideal():
    k = 64
    rng = random.Random("gen-batch")
    gate_desc = cg.gate_descriptor(3, k, S, True, True, [(3, "r"), (0, "r"), (8, "one")], 1)
    a = [[rng.randrange(8), rng.randrange(16), rng.randrange(8)] for _ in range(2)]
    beta = [rng.getrandbits(k) for _ in range(2)]

    def script(role, ses):
        items = [pg.key_request("dpf", 3, "xor", "public_one", a[role][0]),
                 pg.key_request("dcf", 4, "add", "secret", a[role][1], beta[role]),
                 pg.gate_request(gate_desc),
                 pg.key_request("dpf", 3, "xor", "public_one", a[role][2]),
                 pg.gate_request(gate_desc)]
        before = ses.engine.evaluations
        out = pg.gen_batch(ses, items)
        return out, ses.engine.evaluations - before, dict(ses.last_gen)

    plan = pg.planned_batch(k, S, [pg.key_request("dpf", 3, "xor", "public_one"),
                                   pg.key_request("dcf", 4, "add", "secret"), pg.gate_request(gate_desc),
                                   pg.key_request("dpf", 3, "xor", "public_one"), pg.gate_request(gate_desc)])
    r0, r1, _, _, box = run_honest("ideal", script, k=k, reserve=pg.and_budget([plan]), allow_refill=False)
    d0, d1 = box["delta"]
    check_eq([(ses.triple_requests, ses.engine.triples_available()) for ses in box["ses"]], [(1, 0)] * 2,
             "gen_batch: ONE triple request, reserved exactly from planned_batch")
    out0, out1 = r0[0], r1[0]
    check_eq((r0[1], r1[1]), (1, 1), "gen_batch: ONE circuit execution")
    check_eq([type(x).__name__ for x in out0], ["DPFKey", "DCFKey", "GateMaterial", "DPFKey", "GateMaterial"],
             "gen_batch: one output per item, in order")
    check_key_pair("dpf", out0[0], out1[0], 3, k, a[0][0] ^ a[1][0], 1, d0, d1, "batch item 0 (DPF)", rng)
    check_key_pair("dcf", out0[1], out1[1], 4, k, (a[0][1] + a[1][1]) % 16, (beta[0] + beta[1]) % (1 << k),
                   d0, d1, "batch item 1 (DCF, secret payload)", rng)
    g2 = check_gate_pair(gate_desc, out0[2], out1[2], d0, d1, "batch item 2 (gate)", rng)
    check_key_pair("dpf", out0[3], out1[3], 3, k, a[0][2] ^ a[1][2], 1, d0, d1, "batch item 3 (DPF)", rng)
    g4 = check_gate_pair(gate_desc, out0[4], out1[4], d0, d1, "batch item 4 (gate)", rng)
    check(g2["rp"] != g4["rp"] and out0[0].seed != out0[3].seed, "gen_batch: items receive independent coins")
    descs = [cg.dpf_descriptor(3, k, S, "xor", "public_one"), cg.dcf_descriptor(4, k, S, "add", "secret"),
             gate_desc, cg.dpf_descriptor(3, k, S, "xor", "public_one"), gate_desc]
    circuit = cg.build_batch(descs)
    check_eq(r0[2]["digest"], circuit.digest(), "gen_batch: the circuit is build_batch of the items")
    check_eq(r0[2]["stats"]["rounds"], circuit.and_depth() + 6, "gen_batch: rounds = depth of the deepest item + 6")
    NOTES.append(f"gen_batch: 5 items, {circuit.stats()['n_and']:,} AND, depth {circuit.and_depth()}, "
                 f"{r0[2]['stats']['rounds']} rounds")


# --------------------------------------------------------------------------- #
# Aborts and argument validation
# --------------------------------------------------------------------------- #

def test_abort_semantics():
    n, k = 2, 32

    depth = pg.generation_circuit(cg.dpf_descriptor(n, k, S, "xor", "public_one")).and_depth()
    hooks = [("flipped AND opening", {"flip_and": (depth // 2, 0)}, "BoolCheck failed"),
             ("flipped public output", {"flip_public": ("cwout_t", 5, 0)}, "BoolCheck failed"),
             ("flipped private output", {"flip_private": ("root0", 100, 0)}, "BoolCheck failed"),
             ("wrong evaluation header", {"wrong_header": True}, "header mismatch")]
    for name, hook, reason in hooks:
        for cheater in (0, 1):
            cheat = {}          # filled after the first generation: only the second one is attacked
            cheats = [None, None]
            cheats[cheater] = cheat

            def script_h(role, ses, cheat=cheat, cheater=cheater, hook=hook):
                first = pg.gen_dpf(ses, n, "xor", "public_one", role)
                if role == cheater:
                    cheat.update(hook)
                try:
                    pg.gen_dpf(ses, n, "xor", "public_one", 1)
                    second = "returned"
                except ProtocolAbort as exc:
                    second = str(exc)
                at_once = (ses.state, ses.delta_share, ses.delta_wires, ses.engine.dead)
                later = 0
                for call in (lambda: pg.gen_dpf(ses, n, "xor", "public_one", 1),
                             lambda: pg.gen_dcf(ses, n, "xor", "public_one", 1),
                             lambda: pg.gen_gate(ses, cg.gate_preset("table", 2, k, S)),
                             lambda: pg.gen_batch(ses, [pg.key_request("dpf", 2)]),
                             lambda: ses.evaluate(cg.build_session_setup(S), {"delta": 0}),
                             lambda: ses.setup(), lambda: ses.reserve_triples(4),
                             lambda: ses.next_id("kid"), lambda: ses.require_active()):
                    try:
                        call()
                    except ProtocolAbort:
                        later += 1
                return first, second, later, at_once

            out0, out1, _, _, box = run_sessions("ideal", script_h, k=k, cheats=tuple(cheats))
            honest = 1 - cheater
            tag = f"{name} by P{cheater}"
            check(out0[0] == "ok" and out1[0] == "ok", f"{tag}: the scripts ran to the end")
            first, second, later, at_once = (out0, out1)[honest][1]
            check_eq(at_once, (ABORTED, None, None, True),
                     f"{tag}: the session is ABORTED and retired as soon as the generation fails")
            check(len(first) == 1 and isinstance(first[0], fss.DPFKey), f"{tag}: the earlier key was produced")
            check(reason in second, f"{tag}: the honest party aborts ({second!r}) and gets no key")
            check_eq(later, 9, f"{tag}: every later call raises ProtocolAbort")
            ses = box["ses"][honest]
            check_eq(ses.state, ABORTED, f"{tag}: the honest session is ABORTED")
            check(ses.delta_share is None and ses.delta_wires is None and ses.engine.dead
                  and ses.prep.aborted and ses.prep.delta is None,
                  f"{tag}: key shares, session wires and preprocessing are retired")
            check_eq(box["ses"][cheater].state, ABORTED, f"{tag}: the other session is ABORTED as well")

    if not NO_REAL:
        # different public parameters: the real preprocessing already disagrees on the sizes
        def script_mismatch(role, ses):
            return pg.gen_dpf(ses, 2 if role == 0 else 3, "xor", "public_one", 0)

        out0, out1, _, _, box = run_sessions("real", script_mismatch, k=32)
        check(out0[0] == "abort" and out1[0] == "abort", "parameter mismatch (real): both parties abort")
        check_eq((box["ses"][0].state, box["ses"][1].state), (ABORTED, ABORTED),
                 "parameter mismatch (real): both sessions are ABORTED")

    # a peer that disappears
    def script_vanish(role, ses):
        if role == 1:
            return None
        return pg.gen_dpf(ses, 2, "xor", "public_one", 0)

    out0, _, _, _, box = run_sessions("ideal", script_vanish, k=32)
    check(out0[0] == "abort" and box["ses"][0].state == ABORTED, "peer closed: the session is ABORTED")


def test_triple_budget():
    """AND budget helpers, strict mode and refill mode of the session (ideal preprocessing)."""
    k = 32
    gate_desc = cg.gate_preset("table", 3, k, S)
    calls = [pg.planned_key("dpf", 3, k, S), pg.planned_key("dcf", 2, k, S, "add", "secret", 2),
             pg.planned_gate(gate_desc, 2), pg.planned_outmasks(k, S, 3)]
    om = pg.outmask_descriptor(k, S)

    # the helpers against the circuit statistics
    def n_and(desc):
        return cg.build(desc).stats()["n_and"]

    want = [n_and(cg.dpf_descriptor(3, k, S, "xor", "public_one")),
            2 * n_and(cg.dcf_descriptor(2, k, S, "add", "secret")), 2 * n_and(gate_desc), 3 * n_and(om)]
    check_eq([c["triples"] for c in calls], want, "planned calls: n_and of the circuit times the batch")
    check_eq([c["call"] for c in calls], ["gen_dpf", "gen_dcf", "gen_gate", "outmasks"], "planned calls: names")
    budget = pg.and_budget(calls)
    check_eq(budget, sum(want), "and_budget is the sum over the planned calls")
    check_eq(pg.and_budget([(gate_desc, 2), (om, 3)]), want[2] + want[3], "and_budget accepts (descriptor, batch)")
    check_eq(pg.and_budget([]), 0, "empty plan")
    check_eq((pg.and_count(om), pg.outmask_and_count(k, S)), (n_and(om),) * 2, "the output-mask circuit")
    check_eq(om, cg.gate_descriptor(1, k, S, False, False, [], 1), "descriptor of the output-mask circuit")
    check_eq(pg.session_triples(calls, S), S + budget, "session_triples adds the setup circuit (ripple)")
    check_eq(pg.session_triples(calls, S, "sklansky"), cg.build_session_setup(S, adder="sklansky").stats()["n_and"]
             + budget, "session_triples adds the setup circuit (sklansky)")
    items = [pg.key_request("dpf", 3), pg.gate_request(gate_desc)]
    check_eq(pg.planned_batch(k, S, items)["triples"],
             cg.build_batch([cg.dpf_descriptor(3, k, S, "xor", "public_one"), gate_desc]).stats()["n_and"],
             "planned_batch: the concatenated circuit")
    check_eq(pg.planned_outmasks(k, S, 0)["triples"], 0, "no output masks, no triples")
    for bad in (lambda: pg.planned_key("dpx", 3, k, S), lambda: pg.planned_key("dpf", 3, k, S, batch=0),
                lambda: pg.planned_gate(cg.dpf_descriptor(3, k, S)), lambda: pg.planned_outmasks(k, S, -1),
                lambda: pg.planned_batch(k, S, ["dpf"]), lambda: pg.planned_gate(gate_desc, True)):
        check_raises(ValueError, bad)

    def generate(role, ses):
        keys = pg.gen_dpf(ses, 3, "xor", "public_one", role)
        keys += pg.gen_dcf(ses, 2, "add", "secret", [1, 2], [role, 5], 2)
        mats = pg.gen_gate(ses, gate_desc, 2)
        masks = pg.gen_gate(ses, om, 2) + pg.gen_gate(ses, om, 1)       # three masks, two executions
        return len(keys), len(mats), len(masks)

    # (a) strict mode with the exact reservation: one population, nothing left
    def script_exact(role, ses):
        return generate(role, ses), ses.info(), ses.triple_requests

    r0, r1, _, _, box = run_honest("ideal", script_exact, k=k, reserve=pg.and_budget(calls[:3]), outmasks=3,
                                   allow_refill=False)
    for r in (r0, r1):
        info = r[1]
        check_eq(r[0], (3, 2, 3), "strict, exact reservation: every generation ran")
        check_eq((info["triple_requests"], r[2], info["triples_unused"], info["triples_used"],
                  info["triples_requested"], info["allow_refill"]),
                 (1, 1, 0, S + budget, pg.session_triples(calls, S), False),
                 "strict, exact reservation: ONE request of setup + budget triples, all used")
        check_eq(info["bucketing_bound_log2"], None, "ideal preprocessing has no bucketing term")

    # (b) strict mode without (or with too small) a reservation: an error before any message
    for reserve, masks in ((0, 0), (budget - 1, 0)):
        def script_short(role, ses, reserve=reserve):
            facts = []
            for call in (lambda: pg.gen_dpf(ses, 3, "xor", "public_one", role),
                         lambda: pg.gen_dcf(ses, 2, "add", "secret", [1, 2], [role, 5], 2),
                         lambda: pg.gen_gate(ses, gate_desc, 2), lambda: pg.gen_gate(ses, om, 3),
                         lambda: pg.gen_batch(ses, items)):
                before = (ses.chan.msgs_sent, ses.chan.msgs_recv, ses.engine.evaluations)
                try:
                    call()
                    facts.append("ran")
                except TripleBudgetError as exc:
                    quiet = (ses.chan.msgs_sent, ses.chan.msgs_recv, ses.engine.evaluations) == before
                    facts.append(("refused", exc.needed, exc.available, quiet))
            return facts, ses.state, ses.engine.dead, ses.triple_requests, ses.engine.triples_available()

        r0, r1, _, _, _ = run_honest("ideal", script_short, k=k, reserve=reserve, outmasks=masks,
                                     allow_refill=False)
        if reserve == 0:
            exp = [("refused", need, 0, True) for need in want[:3]] + [("refused", want[3], 0, True),
                                                                      ("refused", pg.planned_batch(k, S, items)["triples"], 0, True)]
            left = 0
        else:       # the first three generations fit; the three masks find one triple too few
            left = want[3] - 1
            exp = ["ran", "ran", "ran", ("refused", want[3], left, True),
                   ("refused", pg.planned_batch(k, S, items)["triples"], left, True)]
        check_eq(r0[0], exp, f"strict, reserve {reserve}: a short pool is refused before any message")
        check_eq(r1[0], exp, f"strict, reserve {reserve}: the other party refuses at the same point")
        check_eq(r0[1:], (ACTIVE, False, 1, left), f"strict, reserve {reserve}: the session stays ACTIVE with "
                 "one population")

    # (c) after a refusal the caller may add a population explicitly; it is counted
    def script_explicit(role, ses):
        try:
            pg.gen_dpf(ses, 3, "xor", "public_one", role)
            return "ran"
        except TripleBudgetError as exc:
            ses.reserve_triples(exc.needed - exc.available)
        key = pg.gen_dpf(ses, 3, "xor", "public_one", role)
        return key[0], ses.triple_requests, ses.engine.triples_available(), ses.info()["triples_requested"]

    r0, r1, _, _, box = run_honest("ideal", script_explicit, k=k, allow_refill=False)
    check_eq((r0[1:], r1[1:]), ((2, 0, S + want[0]),) * 2, "strict: an explicit reserve_triples is a second request")
    check_key_pair("dpf", r0[0], r1[0], 3, k, 1, 1, box["delta"][0], box["delta"][1],
                   "key generated after an explicit second request", random.Random("budget"))

    # (d) refill mode: the same script runs without a reservation and every circuit is one more request
    r0, r1, _, _, _ = run_honest("ideal", script_exact, k=k, allow_refill=True)
    check_eq((r0[1]["triple_requests"], r0[1]["triples_unused"], r0[1]["triples_requested"],
              r0[1]["allow_refill"]), (1 + 5, 0, S + budget, True),
             "refill mode: one triple request per circuit execution (setup + 5), counted")
    # refill mode with a partial reservation: only the missing triples are requested
    r0, r1, _, _, _ = run_honest("ideal", script_exact, k=k, reserve=want[0] + want[1] + 7, allow_refill=True)
    check_eq((r0[1]["triple_requests"], r0[1]["triples_requested"]), (1 + 3, S + budget),
             "refill mode: a partial reservation serves the first circuits, the rest are refills")

    # (e) arguments of setup
    def script_args(role, ses):
        n = 0
        for call in (lambda: ses.setup(-1), lambda: ses.setup(0, -1), lambda: ses.setup(True),
                     lambda: ses.setup(1.5), lambda: ses.setup(0, 2.5)):
            try:
                call()
            except ValueError:
                n += 1
        return n, ses.state, ses.is_setup, ses.chan.msgs_sent

    r0, r1, _, _, _ = run_honest("ideal", script_args, k=k, setup=False)
    check_eq((r0, r1), ((5, ACTIVE, False, 0),) * 2, "invalid reservations are ValueErrors before any message")

    # the defaults of the session class (the driver autheval.AuthEval is strict by default instead)
    class _Chan:
        party = 0
        phase = "x"

    plain = Session(0, _Chan(), b"defaults", prep="ideal", dealer=IdealDealer())
    strict = Session(0, _Chan(), b"defaults", prep="ideal", dealer=IdealDealer(), allow_refill=False)
    check_eq((plain.allow_refill, plain.engine.allow_refill, plain.triple_requests), (True, True, 0),
             "Session() refills by default (and counts)")
    check_eq((strict.allow_refill, strict.engine.allow_refill), (False, False), "allow_refill=False is passed down")
    check_raises(ValueError, Session, 0, _Chan(), b"defaults", prep="ideal", dealer=IdealDealer(), allow_refill=None)
    NOTES.append(f"triple budget (k = {k}): DPF n=3 {want[0]:,}, 2 x DCF n=2 secret {want[1]:,}, 2 x table gate "
                 f"n=3 {want[2]:,}, 3 output masks {want[3]:,} AND triples; one request of "
                 f"{pg.session_triples(calls, S):,} in strict mode, 6 requests without a reservation")


def test_argument_validation():
    k = 32

    def script(role, ses):
        n = 0
        calls = [
            lambda: pg.gen_dpf(ses, 3, "xor", "public_one", 8),                   # alpha out of range
            lambda: pg.gen_dpf(ses, 3, "xor", "public_one", -1),
            lambda: pg.gen_dpf(ses, 3, "xor", "secret", 1),                       # beta share missing
            lambda: pg.gen_dpf(ses, 3, "xor", "public_one", 1, 5),                # beta share not allowed
            lambda: pg.gen_dpf(ses, 3, "xor", "secret", 1, 1 << k),               # beta out of range
            lambda: pg.gen_dpf(ses, 3, "xor", "public_one", [1, 2], None, 3),     # wrong number of shares
            lambda: pg.gen_dpf(ses, 3, "mul", "public_one", 1),
            lambda: pg.gen_dpf(ses, 3, "xor", "hidden", 1),
            lambda: pg.gen_dpf(ses, 0, "xor", "public_one", 0),
            lambda: pg.gen_dcf(ses, 3, "xor", "public_one", 1, None, 0),          # batch 0
            lambda: pg.gen_dcf(ses, 3, "xor", "public_one", True),
            lambda: pg.gen_gate(ses, cg.gate_preset("table", 3, 64, S)),          # another k
            lambda: pg.gen_gate(ses, cg.dpf_descriptor(3, k, S)),                 # not a gate
            lambda: pg.gen_gate(ses, cg.gate_preset("table", 3, k, S), 0),
            lambda: pg.gen_batch(ses, []),
            lambda: pg.gen_batch(ses, [{"kind": "session_setup"}]),
            lambda: pg.gen_batch(ses, ["dpf"]),
            lambda: pg.key_request("dpx", 3),
        ]
        for call in calls:
            try:
                call()
            except ValueError:
                n += 1
        ok = pg.gen_dpf(ses, 3, "add", "public_one", [3 + role, 7], None, 2)
        return n, len(calls), ses.state, ok, ses.engine.evaluations

    r0, r1, _, _, box = run_honest("ideal", script, k=k)
    check_eq((r0[0], r1[0]), (r0[1], r1[1]), "every invalid call raises ValueError")
    check_eq((r0[2], r1[2], r0[4]), (ACTIVE, ACTIVE, 2), "the sessions stay ACTIVE; no circuit was run for invalid calls")
    rng = random.Random("validation")
    for j, alpha in enumerate(((3 + 4) % 8, (7 + 7) % 8)):
        check_key_pair("dpf", r0[3][j], r1[3][j], 3, k, alpha, 1, box["delta"][0], box["delta"][1],
                       f"generation after argument errors, instance {j}", rng)


# --------------------------------------------------------------------------- #
# Real preprocessing
# --------------------------------------------------------------------------- #

def _phase_line(st: dict) -> str:
    return ", ".join(f"{name} {ph['bytes_sent']:,} B / {ph['rounds']} r / {ph['time_s']:.2f} s"
                     for name, ph in st["phases"].items())


def test_keygen_real():
    if NO_REAL:
        return
    k = 64
    rng = random.Random("keygen-real")
    jobs = []
    for family, n in (("dpf", 4), ("dpf", 8), ("dcf", 4)):
        alpha, beta = _shares(rng, n, k, "xor", "public_one", 1)
        jobs.append((family, n, "xor", "public_one", 1, alpha, beta))
    n_and = [pg.generation_circuit((cg.dpf_descriptor if f == "dpf" else cg.dcf_descriptor)(n, k, S, "xor", "public_one")
                                   ).stats()["n_and"] for f, n, *_ in jobs]
    total = S + sum(n_and)
    t = time.perf_counter()
    check_eq(_keygen_budget(k, jobs), sum(n_and), "real: and_budget equals the sum of the circuits' AND gates")
    r0, r1, st0, st1, box = _run_keygen_jobs("real", k, jobs, reserve=_keygen_budget(k, jobs), allow_refill=False)
    run_s = time.perf_counter() - t
    n_keys = _check_keygen_jobs("real", k, jobs, r0, r1, box, rng)
    info = r0[1]
    bucket = bucket_size(total, 40)
    check_eq((info["triple_requests"], info["triples_used"], info["triples_unused"]), (1, total, 0),
             "real: all triples of the session come from ONE request (one bucketing population)")
    check_eq((info["last_bucket_size"], info["prep_counters"]["leaky_triples"]), (bucket, bucket * total),
             "real: bucket size of Lemma 4 for the whole session")
    check_eq((info["bucketing_bound_log2"], info["allow_refill"]), (-40.0, False),
             "real: one population, i.e. 2^-sigma for the triples of the whole session")
    for st in (st0, st1):
        for name in ("bytes_sent", "bytes_recv", "rounds"):
            check_eq(sum(ph[name] for ph in st["phases"].values()), st[name], f"real: phases sum to total {name}")
    check(all(name in st0["phases"] for name in ("baseot", "cotext", "laand", "bucket", "check", "online")),
          "real: preprocessing and online phases are reported")
    check_eq(st0["phases"]["laand"]["bytes_sent"], 16 * bucket * total, "real: 16 bytes per leaky AND and party")
    NOTES.append(f"keygen real (DPF n=4, DPF n=8, DCF n=4; public payload): {n_keys} key pairs, {total:,} triples, "
                 f"B = {bucket}, {st0['rounds']} rounds, {st0['bytes_sent'] + st1['bytes_sent']:,} bytes (both "
                 f"directions), {run_s:.1f} s wall for both parties")

    # a batch of 3 with real preprocessing in REFILL mode (triples requested by the evaluation
    # itself): two populations, and the session says so
    alpha, beta = _shares(rng, 4, k, "add", "public_one", 3)
    jobs = [("dpf", 4, "add", "public_one", 3, alpha, beta)]
    r0, r1, st0, st1, box = _run_keygen_jobs("real", k, jobs, allow_refill=True)
    _check_keygen_jobs("real", k, jobs, r0, r1, box, rng)
    check_eq(r0[1]["triple_requests"], 2, "real batch: one request for the setup circuit, one for the batch")
    check_eq(r0[1]["bucketing_bound_log2"], -39.0, "real batch, refill mode: two populations, 2 * 2^-sigma")


def test_dpf_n16_real():
    """DPF, n = 16, public payload, real preprocessing: correctness and the measured cost."""
    if NO_LARGE:
        return
    n, k = 16, 64
    rng = random.Random("dpf-n16-real")
    alpha, beta = _shares(rng, n, k, "xor", "public_one", 1)
    jobs = [("dpf", n, "xor", "public_one", 1, alpha, beta)]
    t = time.perf_counter()
    circuit = pg.generation_circuit(cg.dpf_descriptor(n, k, S, "xor", "public_one"))
    build_s = time.perf_counter() - t
    st_c = circuit.stats()
    total = S + st_c["n_and"]
    t = time.perf_counter()
    r0, r1, st0, st1, box = _run_keygen_jobs("real", k, jobs, reserve=_keygen_budget(k, jobs),
                                             allow_refill=False)
    run_s = time.perf_counter() - t
    _check_keygen_jobs("real", k, jobs, r0, r1, box, rng)
    info = r0[1]
    bucket = bucket_size(total, 40)
    check_eq((info["triple_requests"], info["last_bucket_size"]), (1, bucket), "n=16: one population, bucket size")
    for st in (st0, st1):
        check_eq(sum(ph["bytes_sent"] for ph in st["phases"].values()), st["bytes_sent"],
                 "n=16: per-phase bytes sum to the total")
        check_eq(sum(ph["rounds"] for ph in st["phases"].values()), st["rounds"],
                 "n=16: per-phase rounds sum to the total")
    check_eq(st0["rounds"], st1["rounds"], "n=16: both parties count the same rounds")
    gen = r0[0][0][1]["stats"]
    gen_rounds = {name: st0["phases"][name]["rounds"] for name in ("input", "online", "outcheck", "output")}
    check_eq(gen_rounds, {"input": 1, "online": st_c["and_depth"], "outcheck": 4, "output": 1},
             "n=16: rounds of the generation circuit = 1 + AND depth + 2 + 1 + 2")
    key = r0[0][0][0][0]
    both = st0["bytes_sent"] + st1["bytes_sent"]
    NOTES.append(
        f"DPF n=16 k=64 s=64 public payload, REAL preprocessing, sigma=40, batch 1, loopback threads: "
        f"{st_c['n_and']:,} AND (+{S} session setup), AND depth {st_c['and_depth']}, B = {bucket}; "
        f"wall {run_s:.2f} s for both parties (session setup {box['setup_s'][0]:.2f} s incl. base OT and all "
        f"triples; circuit build {build_s:.2f} s not included); rounds {st0['rounds']} in total, "
        f"{sum(gen_rounds.values())} for the generation circuit; bytes {both:,} in both directions "
        f"({st0['bytes_sent']:,} sent by P0), i.e. {both / st_c['n_and']:.1f} bytes per AND gate; "
        f"key {key.size_bytes()} bytes; online evaluation {gen['online_s'] * 1e3:.0f} ms "
        f"({gen['gates_per_s'] / 1e6:.1f} M gates/s)")
    NOTES.append("DPF n=16 phases of P0 (bytes sent / rounds / seconds): " + _phase_line(st0))


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
    print(f"Couplet session / Pi_Gen tests (real preprocessing: {'no' if NO_REAL else 'yes'}, "
          f"n = 16 run: {'no' if NO_LARGE else 'yes'}), {len(tests)} test functions")
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
