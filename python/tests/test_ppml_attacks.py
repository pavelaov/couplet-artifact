"""Adversarial tests of the inference protocol (couplet.infer; DESIGN_M3.md sections 3
and 4, the inference theorem): every detectable deviation aborts before any output, an
input substitution is not detectable and not harmful, and no abort frequency depends on
the honest inputs.

One party runs the unmodified driver, the other is an adversary built from the
test hooks: a deviating arithmetic checker (``couplet.attacks.adversaries.
CheatingChecker``: wrong value shares, replays, the MAC-guess compensation),
a model whose first-layer share is tampered (used consistently: an input
substitution) or a driver that substitutes a plane key locally.  The model is the tiny MLP [4, 2, 2, 2] at the
wire format (4, 2) with random integer weights; the standard setting is
k = 32, s = 64, the tiny one k = 8, s = 4 (where a guess of the MAC key share
has a noticeable chance).

Deviations
  deferred        a wrong value share (+1) in the first deferred opening
                  (the truncation opening of layer 2): the first ArithCheck
                  of the release fails; no logit is released
  deferred-high   an error 2^(m-1) there (0 mod 2^(m-s)): the payload is
                  unchanged and the check passes with probability 3/4; the
                  verdict is predicted from the algebra of the check, and a
                  passing run releases the correct logits (the mirror)
  release         a wrong share (+1) of the logit opening: the second check
                  fails, the honest party outputs nothing
  release-high    an error 2^(w-1) on the logit opening: passes with
                  probability 3/4; the release decodes the authenticated
                  payload mod 2^k, so a passing run releases the correct
                  logits (the mirror)
  first-layer     +e on the own masked first-layer share: an input
                  substitution; no abort, both parties agree, and the result
                  is the mirror on the substituted pre-activation
  wrong-key       the adversary evaluates a plane A key that is not the one
                  of the session (a freshly dealt key): caught at the first
                  check
  replay          the value share of opening 0 is sent again at opening 1:
                  caught at the first check
  tiny-guess      k = 8, s = 4: a wrong share (+1) at opening 0 with the
                  compensation by a uniform guess of the honest key share:
                  passes with the probability the algebra predicts (about
                  2^-s times a small constant); never a wrong value is
                  released unnoticed unless the guess was right

Part 1 (both truncation modes, a few runs each): the verdicts above, the
state of the honest party after an abort (dead, silent), the logits.

Part 3 (``test_private_logit_mask``, correction C2 of design/CHANGELOG_M3.md):
in private-model mode a malicious P_0 inputs the logit-layer weights
``W = 2^k e_j`` (every logit row selects the hidden activation ``a_j``, scaled
by ``2^k``; P_0 skips its own bound check).  Without an output mask the
opened logit ``2^k a_j + b 2^f`` would carry ``a_j`` (a function of P_1's
features) in its high ``s`` bits, which F_infer does not release.  With the
masks: (a) the honest party's output equals the functionality's output, the
low ``k`` bits of the logit read in two's complement (``b 2^f``), in both
modes, and both parties agree; (b) the opened high bits ``h = v >> k`` are
uniform: over ``MASK_RUNS`` runs (two logits each, k = 32, s = 64, mode
round, the honest features uniform per run) three tests are applied to the
samples ``h``: the chi-square statistic of the low nibble ``h mod 16`` and of
the top nibble ``h >> 60`` (16 cells, 15 degrees of freedom) and the
frequency of ones over all 64 bit positions (normal approximation).  Each
test rejects at the false-alarm probability ``ALPHA_MASK = 1e-4`` (chi-square
p-value from the regularised incomplete gamma function, two-sided normal
tail), so an honest run of the whole test raises a false alarm with
probability at most ``3e-4`` (union bound); (c) a control shows the power of
the same statistics: the unmasked high bits (the opened value minus the
mask, reconstructed from both parties' mask shares, which only a test can
do) equal ``a_j + floor(b 2^f / 2^k)`` exactly and the chi-square test of
their low nibble rejects (p-value below ``ALPHA_MASK``).
Part 2 (mode round, ``RUNS`` runs per world, two honest-input worlds: the
honest party's features all 0 or all 1.0, the adversary's features 0.5 in
both worlds, so that only the honest input differs): the honest party's
abort frequencies are compared with a two-proportion z-score, |z| < 3.3 for
every adversary (the deterministic ones have z = 0; the randomised ones are
the real test), and every randomised verdict equals its prediction.  The trials run in worker
processes (``--jobs N``; ``--runs R`` changes the runs per world).

Run from artifact/python:  python tests/test_ppml_attacks.py  (``-k name``
selects tests).  Prints the number of checks; exits non-zero on any failure.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import random
import secrets
import statistics
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import circuits_ppml as cp  # noqa: E402
from couplet import nn as nnm  # noqa: E402
from couplet.attacks.adversaries import CheatingChecker, two_proportion_z  # noqa: E402
from couplet.fixedpoint import to_signed  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.infer import InferParty  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.prep import IdealDealer  # noqa: E402
from couplet.session import ABORTED  # noqa: E402
from couplet.truncmask import Site  # noqa: E402

CHECKS = 0
Z_THRESHOLD = 3.3
RUNS = 100
N, F = 4, 2
ADVERSARIES = ("deferred", "deferred-high", "release", "release-high", "first-layer", "wrong-key", "replay",
               "tiny-guess")
RANDOMISED = {"deferred-high": 0.75, "release-high": 0.75, "tiny-guess": None}
MASK_RUNS = 400                # runs of part 3 (two logits each)
ALPHA_MASK = 1e-4              # false-alarm probability of each of the three uniformity tests
MASK_K, MASK_S, MASK_J = 32, 64, 0


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


def _arg(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


# --------------------------------------------------------------------------- #
# Adversaries
# --------------------------------------------------------------------------- #

class TamperModel(nnm.Model):
    """The adversary's model: its first-layer share of site 0 carries the error ``error``.

    The deviating party uses the tampered share consistently (it commits to it
    and computes with it), which is exactly an input substitution.
    """

    error = 37

    def first_layer_share(self, role: int, features):
        p = super().first_layer_share(role, features)
        p[0] += self.error
        return p


def tamper_model(model: nnm.Model) -> TamperModel:
    return TamperModel(model.kind, model.act, model.n, model.f, model.layers, model.meta, model.float_layers)


class WrongKeyParty(InferParty):
    """Evaluates, for site 0 of layer 1, a plane A key that was not generated in the session."""

    def _layer_sites(self, layer: int):
        sites = list(self.sites[layer])
        if layer == 0:
            desc = sites[0].material.descriptor
            rnd = random.Random(secrets.randbits(64))
            fake = cp.deal_material(desc, rnd.getrandbits(self.s) + rnd.getrandbits(self.s), rnd)[self.role]
            mat = sites[0].material
            fake.rp, fake.rh, fake.r2, fake.plane_b = mat.rp, mat.rh, mat.r2, mat.plane_b
            sites[0] = Site(fake)
        return sites


def model_for(setting: str):
    """The tiny model (random integer weights) and the session parameters of a setting."""
    k, s = (8, 4) if setting == "tiny" else (32, 64)
    model = nnm.random_model("mlp4", N, F, seed=11)
    return model, k, s


ADV_CODE = 1 << (F - 1)        # the adversary's own features: 0.5 in both worlds


def world_codes(world: int, adv: int, model: nnm.Model) -> list:
    """All features: the honest party's slice is 0 (world 0) or 1.0 (world 1), the adversary's 0.5."""
    codes = [ADV_CODE] * model.d
    honest = model.feature_slice(1 - adv)
    for i in range(honest.start, honest.stop):
        codes[i] = world * (1 << F)
    return codes


def trial(name: str, world: int, mode: str, seed: int) -> dict:
    """One run of adversary ``name`` against an honest party; returns the honest verdict."""
    setting = "tiny" if name == "tiny-guess" else "standard"
    model, k, s = model_for(setting)
    rnd = random.Random(seed)
    adv = rnd.randrange(2)
    codes = world_codes(world, adv, model)
    dealer = IdealDealer()
    sid = b"ppml-attack-" + secrets.token_bytes(4)
    widths = model.hidden_widths
    before_release = (sum(widths) + widths[0]) if mode == "exact" else sum(widths[1:])
    w = k + s
    m_bits = F + N + s
    kwargs: dict = {}
    cls = InferParty
    if name == "deferred":
        kwargs["checker_factory"] = CheatingChecker.factory(value_errors={0: 1})
    elif name == "deferred-high":
        kwargs["checker_factory"] = CheatingChecker.factory(value_errors={0: 1 << (m_bits - 1)})
    elif name == "release":
        kwargs["checker_factory"] = CheatingChecker.factory(value_errors={before_release: 1})
    elif name == "release-high":
        kwargs["checker_factory"] = CheatingChecker.factory(value_errors={before_release: 1 << (w - 1)})
    elif name == "first-layer":
        kwargs["model"] = tamper_model(model)
    elif name == "wrong-key":
        cls = WrongKeyParty
    elif name == "replay":
        kwargs["checker_factory"] = CheatingChecker.factory(replay={1: 0})
    elif name == "tiny-guess":
        kwargs["checker_factory"] = CheatingChecker.factory(value_errors={0: 1}, compensate=True)
    else:
        raise ValueError(name)
    parties = [None, None]

    def party(role):
        def fn(chan):
            c = cls if role == adv else InferParty
            own = dict(kwargs) if role == adv else {}
            m = own.pop("model", model)
            p = c(role, chan, sid, m, mode, prep="ideal", dealer=dealer, k=k, s=s, **own)
            parties[role] = p
            res = {"logits": None, "aborted": False, "message": ""}
            try:
                res["logits"] = p.run(codes[model.feature_slice(role)])
            except ProtocolAbort as exc:
                res["aborted"] = True
                res["message"] = str(exc)
            res["delta"] = p.delta_share if p.state != ABORTED else None
            res["state"] = p.state
            res["record"] = dict(p.record)
            return res
        return fn

    r0, r1, st0, st1 = run_two_party(party(0), party(1), timeout=300)
    honest, cheater = (r1, r0) if adv == 0 else (r0, r1)
    hp, ap = parties[1 - adv], parties[adv]
    out = {"name": name, "world": world, "mode": mode, "adv": adv, "aborted": honest["aborted"],
           "message": honest["message"], "honest_logits": honest["logits"], "adv_logits": cheater["logits"],
           "state": honest["state"], "error": None}
    # the honest party after an abort: dead and silent
    if honest["aborted"]:
        chan = hp.chan
        sent = (chan.msgs_sent, chan.bytes_sent)
        refused = 0
        for cmd in (lambda: hp.setup(), lambda: hp.preprocess(), lambda: hp.infer([0, 0])):
            try:
                cmd()
            except ProtocolAbort:
                refused += 1
            except Exception:  # noqa: BLE001
                pass
        out["dead"] = bool(hp.state == ABORTED and hp.session.delta_share is None and hp.checker.dead
                           and hp.sites == [])
        out["silent"] = refused == 3 and (chan.msgs_sent, chan.bytes_sent) == sent
        out["released"] = bool(hp.record.get("logits"))
    # predictions of the randomised verdicts
    ck = ap.checker
    if isinstance(ck, CheatingChecker) and name in RANDOMISED:
        delta_h = hp.session.delta_share if hp.session.delta_share is not None else honest.get("delta")
        if delta_h is None:                     # the honest key share was retired by the abort; recover it
            delta_h = hp._delta_for_prediction if hasattr(hp, "_delta_for_prediction") else None
        out["predicted_pass"] = None if delta_h is None else bool(ck.predict_pass(delta_h))
    if name != "first-layer" and not honest["aborted"] and honest["record"].get("thresholds") is not None:
        out["mirror_logits"] = model.mirror(codes, mode, honest["record"]["thresholds"]).logits
    if name == "first-layer":
        # the honest party's share plus the tampered share of the adversary = the substituted p
        p_h = model.first_layer_share(1 - adv, codes[model.feature_slice(1 - adv)])
        p_a = model.first_layer_share(adv, codes[model.feature_slice(adv)])
        pre = [a + b for a, b in zip(p_h, p_a)]
        pre[0] += TamperModel.error
        thr = honest["record"].get("thresholds")
        out["mirror_logits"] = model.mirror(codes, mode, thr, first_pre=pre).logits if thr is not None else None
    return out


# --------------------------------------------------------------------------- #
# Part 1: the verdicts
# --------------------------------------------------------------------------- #

def record_delta(party_cls):
    """Keep the honest key share for the prediction of randomised verdicts (test hook)."""
    orig = party_cls.setup

    def setup(self):
        orig(self)
        self._delta_for_prediction = self.session.delta_share
    party_cls.setup = setup


record_delta(InferParty)


def test_verdicts():
    for mode in cp.MODES:
        for name in ADVERSARIES:
            for i in range(3):
                res = trial(name, i % 2, mode, 1000 * i + len(name))
                tag = f"{name} {mode} run {i}"
                check(res["error"] is None, f"{tag}: ran")
                if name == "first-layer":
                    check(not res["aborted"], f"{tag}: an input substitution does not abort")
                    check_eq(res["honest_logits"], res["adv_logits"], f"{tag}: both parties agree")
                    check_eq(res["honest_logits"], res["mirror_logits"], f"{tag}: the mirror on the substituted input")
                elif name in RANDOMISED:
                    check(res["predicted_pass"] is not None, f"{tag}: prediction available")
                    check_eq(res["aborted"], not res["predicted_pass"], f"{tag}: the verdict is the predicted one")
                    if res["aborted"]:
                        check(res["honest_logits"] is None and res["dead"] and res["silent"], f"{tag}: dead and silent")
                    else:
                        check(res["honest_logits"] is not None and res["honest_logits"] == res["adv_logits"],
                              f"{tag}: a passed check releases the unchanged payload")
                        if name != "tiny-guess":
                            check_eq(res["honest_logits"], res.get("mirror_logits"),
                                     f"{tag}: the released logits are the correct ones (the mirror)")
                else:
                    check(res["aborted"], f"{tag}: detected")
                    check(res["honest_logits"] is None and not res["released"], f"{tag}: no output was released")
                    check(res["dead"] and res["silent"], f"{tag}: the honest party is dead and silent")
                    check_eq(res["state"], ABORTED, f"{tag}: ABORTED")
            print(f"    {name:<14} {mode:<5}: ok", flush=True)


# --------------------------------------------------------------------------- #
# Part 2: no abort frequency depends on the honest inputs
# --------------------------------------------------------------------------- #

def _job(args):
    name, world, mode, seed = args
    try:
        return trial(name, world, mode, seed)
    except Exception:  # noqa: BLE001
        return {"name": name, "world": world, "aborted": None, "error": traceback.format_exc()[-2000:],
                "predicted_pass": None}


def test_distinguish():
    runs = _arg("--runs", RUNS)
    jobs = _arg("--jobs", max(1, min(8, (os.cpu_count() or 2) // 2)))
    tasks = [(name, world, "round", 7919 * i + 13 * world + len(name))
             for name in ADVERSARIES for world in (0, 1) for i in range(runs)]
    t = time.perf_counter()
    if jobs > 1:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            results = pool.map(_job, tasks, chunksize=4)
    else:
        results = [_job(task) for task in tasks]
    print(f"    {len(tasks)} runs in {time.perf_counter() - t:.0f} s with {jobs} process(es)", flush=True)
    print(f"    {'adversary':<14} {'aborts w0':>9} {'aborts w1':>9} {'runs':>5} {'z':>6}  predicted", flush=True)
    for name in ADVERSARIES:
        rows = [r for r in results if r["name"] == name]
        errors = [r["error"] for r in rows if r.get("error")]
        check(not errors, f"{name}: runs failed: {errors[:1]}")
        aborts = [sum(r["aborted"] for r in rows if r["world"] == w) for w in (0, 1)]
        z = two_proportion_z(aborts[0], runs, aborts[1], runs)
        check(abs(z) < Z_THRESHOLD, f"{name}: abort frequencies differ between the worlds (z = {z:.2f})")
        if name == "first-layer":
            check_eq(aborts, [0, 0], f"{name}: never aborts")
        elif name in RANDOMISED:
            predicted = [r["predicted_pass"] for r in rows]
            check(all(p is not None for p in predicted), f"{name}: predictions available")
            check(all(r["aborted"] == (not r["predicted_pass"]) for r in rows), f"{name}: every verdict as predicted")
            total = aborts[0] + aborts[1]
            check(0 < total < 2 * runs, f"{name}: a randomised verdict ({total} of {2 * runs} aborts)")
            p = RANDOMISED[name]
            if p is not None:
                check(abs(total / (2 * runs) - (1 - p)) < 0.12, f"{name}: abort rate {total / (2 * runs):.2f} near {1 - p}")
                passed = [r for r in rows if not r["aborted"]]
                check(all(r["honest_logits"] is not None and r["honest_logits"] == r.get("mirror_logits")
                          for r in passed), f"{name}: every passing run released the correct logits")
        else:
            check_eq(aborts, [runs, runs], f"{name}: always detected")
        pred = "" if name not in RANDOMISED else "all verdicts as predicted"
        print(f"    {name:<14} {aborts[0]:>9} {aborts[1]:>9} {runs:>5} {z:>6.2f}  {pred}", flush=True)


# --------------------------------------------------------------------------- #
# Part 3: private-model mode, malicious logit-layer weights W = 2^k e_j
# --------------------------------------------------------------------------- #

class EvilWeightsParty(InferParty):
    """P_0 that inputs ``evil`` (the logit layer replaced by ``2^k e_j``) instead of its model.

    The honest model is given to the constructor (so the honest-looking bound
    checks pass and the manifest binds the same architecture); the inputs of
    the online phase are taken from ``evil``.
    """

    def __init__(self, *args, evil=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = evil


def evil_model(model: nnm.Model, k: int, j: int) -> nnm.Model:
    rows, cols = model.shapes[-1]
    W = np.zeros((rows, cols), dtype=np.int64)
    W[:, j] = 1 << k
    layers = list(model.layers[:-1]) + [(W, model.layers[-1][1])]
    return nnm.Model(model.kind, model.act, model.n, model.f, layers, model.meta)


def _regularised_gamma_q(a: float, x: float) -> float:
    """Q(a, x) = Gamma(a, x) / Gamma(a) (series for x < a + 1, continued fraction otherwise)."""
    if x <= 0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(10000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - lg))
    b, c, d = x + 1 - a, 1e300, 1.0 / (x + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - lg) * h


def chi2_sf(stat: float, df: int) -> float:
    """Survival function of the chi-square distribution with ``df`` degrees of freedom."""
    return _regularised_gamma_q(df / 2.0, stat / 2.0)


def chi2_nibble(values: list, shift: int) -> float:
    counts = [0] * 16
    for v in values:
        counts[(v >> shift) & 15] += 1
    exp = len(values) / 16.0
    return sum((c - exp) ** 2 / exp for c in counts)


def mask_trial(args) -> dict:
    """One private-model run with the malicious P_0; returns what the tests need."""
    mode, seed = args
    try:
        k, s, j = MASK_K, MASK_S, MASK_J
        w = k + s
        model = nnm.random_model("mlp4", N, F, seed=11)
        evil = evil_model(model, k, j)
        rnd = random.Random(seed)
        codes = [rnd.randrange(0, (1 << F) + 1) for _ in range(model.d)]
        dealer = IdealDealer()
        sid = b"ppml-mask-" + secrets.token_bytes(4)
        parties = [None, None]

        def party(role):
            def fn(chan):
                if role == 0:
                    p = EvilWeightsParty(0, chan, sid, model, mode, prep="ideal", dealer=dealer, k=k, s=s,
                                         private_model=True, evil=evil)
                else:
                    p = InferParty(1, chan, sid, model.shape_only(), mode, prep="ideal", dealer=dealer, k=k, s=s,
                                   private_model=True)
                parties[role] = p
                return p.run(codes[model.feature_slice(role)])
            return fn

        l0, l1, _, _ = run_two_party(party(0), party(1), timeout=600)
        p0, p1 = parties
        thr = p1.record["thresholds"]
        mir = evil.mirror(codes, mode, thr)
        bias = [int(b) for b in model.layers[-1][1]]
        masks = [(m0.value + m1.value) % (1 << w) for m0, m1 in zip(p0.outmasks, p1.outmasks)]
        return {"error": None, "mode": mode, "logits": (l0, l1),
                "functionality": [to_signed(v % (1 << k), k) for v in mir.logits],
                "released_adv": list(p0.record["released"]), "released_honest": list(p1.record["released"]),
                "masks": masks, "a_j": int(mir.acts[-1][j]), "bias": bias, "f": model.f}
    except Exception:  # noqa: BLE001
        return {"error": traceback.format_exc()[-2000:]}


def test_private_logit_mask():
    runs = _arg("--mask-runs", MASK_RUNS)
    jobs = _arg("--jobs", max(1, min(8, (os.cpu_count() or 2) // 2)))
    k, s = MASK_K, MASK_S
    w = k + s
    tasks = [("exact", 501), ("exact", 502)] + [("round", 9001 + i) for i in range(runs)]
    t = time.perf_counter()
    if jobs > 1:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            results = pool.map(mask_trial, tasks, chunksize=2)
    else:
        results = [mask_trial(task) for task in tasks]
    errors = [r["error"] for r in results if r.get("error")]
    check(not errors, f"runs failed: {errors[:1]}")
    high, control = [], []
    for r in results:
        l0, l1 = r["logits"]
        check_eq(l0, l1, f"{r['mode']}: both parties output the same logits")
        check_eq(l1, r["functionality"], f"{r['mode']}: the honest output is the functionality's (low k bits)")
        check_eq(r["released_adv"], r["released_honest"], "both parties saw the same opening")
        for i, v in enumerate(r["released_adv"]):
            check_eq(to_signed(v % (1 << k), k), r["functionality"][i], "the low k bits of the opening")
            check_eq(to_signed(v % (1 << k), k), r["bias"][i] << r["f"], "the low k bits are b 2^f")
            if r["mode"] != "round":
                continue
            high.append(v >> k)
            u = ((v - r["masks"][i]) % (1 << w)) >> k
            check_eq(to_signed(u, s), r["a_j"] + ((r["bias"][i] << r["f"]) >> k),
                     "control: without the mask the high bits are a_j (+ the borrow of the bias)")
            control.append(u)
    m = len(high)
    check_eq(m, 2 * runs, "two masked logits per run")
    p_low = chi2_sf(chi2_nibble(high, 0), 15)
    p_top = chi2_sf(chi2_nibble(high, s - 4), 15)
    ones = sum(bin(h).count("1") for h in high)
    z = (ones - 32.0 * m) / math.sqrt(16.0 * m)
    z_thr = statistics.NormalDist().inv_cdf(1 - ALPHA_MASK / 2)
    check(p_low > ALPHA_MASK, f"low nibble of the opened high bits uniform (p = {p_low:.3g})")
    check(p_top > ALPHA_MASK, f"top nibble of the opened high bits uniform (p = {p_top:.3g})")
    check(abs(z) < z_thr, f"frequency of ones in the opened high bits (z = {z:.2f})")
    p_ctrl = chi2_sf(chi2_nibble(control, 0), 15)
    check(p_ctrl < ALPHA_MASK, f"control: the same test detects the unmasked high bits (p = {p_ctrl:.3g})")
    check(abs(chi2_sf(24.996, 15) - 0.05) < 1e-4 and abs(chi2_sf(30.578, 15) - 0.01) < 1e-4,
          "chi-square survival function at two tabulated quantiles")
    print(f"    {len(tasks)} runs ({runs} round, 2 exact) in {time.perf_counter() - t:.0f} s with {jobs} process(es); "
          f"{m} masked logits: p(low nibble) = {p_low:.3f}, p(top nibble) = {p_top:.3f}, z(ones) = {z:.2f} "
          f"(|z| < {z_thr:.2f}); control p = {p_ctrl:.2g}", flush=True)


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
            print(f"  ok   {name:<24} {CHECKS - before:>7} checks {time.perf_counter() - t:7.1f} s", flush=True)
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
    multiprocessing.freeze_support()
    sys.exit(main())
