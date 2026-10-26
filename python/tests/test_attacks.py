"""Adversarial regression for Pi_Gen and Pi_AuthEval (couplet.attacks; DESIGN.md sections 2.2,
2.3, 5.3, 5.4, 7.2, 7.4 and the appendix of section 9).

  * the adversary's tools: the tampering channel wrapper (rewrite, replay,
    mirror, rush, stop after receiving), the deviating arithmetic checker, the
    two-proportion z-score, the exact binomial tests;
  * every deviation A1 - A15 of ``couplet.attacks.adversaries.ATTACKS``: the
    honest party runs the unmodified protocol code in its default strict mode
    (one triple request per session), the other party deviates; checked per
    deviation: every effective deviation is detected (the honest party outputs
    abort), no honest output is released, the session is dead afterwards and
    sends nothing; for the deviations that pass with a known probability the
    number of passes is tested against the prediction with an exact binomial
    test at significance 10^-4 (A4: 2^-t and "passes iff every guess is
    right", in every preprocessing mode including the complete real pipeline;
    A5: "aborts iff the attacked MAC-key bit is 1"; A14: (j + 2) / 2^(j+1) for
    an error 2^(m-j) that leaves the payload unchanged; s = 4 rows: at most
    the Lemma 2 bound) and every released output of the s = 64 rows is
    correct; for the deviations of the arithmetic layer (A6, A7, A9, A11, A14)
    the honest verdict of EVERY trial is compared with the verdict that the
    algebra of the check predicts from the adversary's error, its guess and
    the honest key share;
  * the distinguishing experiment for the absence of selective failure: abort
    frequency of each adversary A1 - A7 and A14 in two honest-input worlds,
    two-proportion z-score below 3.3 in absolute value; and the positive
    control (the one-bit-leaky baseline), whose z-score must exceed the
    threshold.  The threshold has a false-alarm probability of about 10^-3 per
    adversary whose abort frequency is not 0 or 1 (A4, A5, A14 and the s = 4
    rows).

By default the counts are reduced (four to eight minutes with one process,
under two minutes with ``--jobs 8``); ``--full`` uses the
counts of ``couplet/attacks/run_all.py`` (200 runs per world).  ``--jobs N``
spreads the trials over N processes, ``--no-real`` skips the two rows on the
complete real preprocessing.  This script does NOT write
``experiments/raw/attacks.json``; ``run_all.py`` does.

Run from artifact/python:  python tests/test_attacks.py   (``-k name`` selects
tests).  Prints the number of checks; exits non-zero on any failure.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import numpy as np  # noqa: E402

from couplet.attacks import adversaries as adv  # noqa: E402
from couplet.attacks import run_all  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import loopback_pair  # noqa: E402
from couplet.spdz2k import ArithChecker, deal, lemma2_bound  # noqa: E402

FULL = "--full" in sys.argv
NO_REAL = "--no-real" in sys.argv
CHECKS = 0
NOTES: list = []


def _arg(name: str, default: int) -> int:
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


JOBS = _arg("--jobs", 1)
SEED = _arg("--seed", 7)
RUNS_PER_WORLD = _arg("--runs", run_all.DEFAULT_RUNS if FULL else 40)


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


def raises_value_error(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def trials_for(name: str) -> int:
    """Trials of one deviation: the catalogue's count with ``--full``, a reduced count otherwise."""
    _, setting, _, _, expect, default = adv.ATTACKS[name]
    if FULL:
        return default
    if setting.prep == "real":
        return 2
    if isinstance(expect, tuple):
        return min(default, 60 if expect[0] == "rate" else 120)
    return min(default, 10)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

def test_adversary_tools():
    # two-proportion z-score
    check_eq(adv.two_proportion_z(200, 200, 200, 200), 0.0, "equal frequencies of 1")
    check_eq(adv.two_proportion_z(0, 200, 0, 200), 0.0, "equal frequencies of 0")
    check(abs(adv.two_proportion_z(100, 200, 100, 200)) < 1e-12, "equal frequencies")
    z = adv.two_proportion_z(100, 200, 120, 200)
    pooled = 220 / 400
    check(abs(z - (0.6 - 0.5) / math.sqrt(pooled * (1 - pooled) * (2 / 200))) < 1e-12, "z formula")
    check(adv.two_proportion_z(0, 200, 200, 200) > 19 and adv.two_proportion_z(200, 200, 0, 200) < -19,
          "totally different frequencies")
    check_eq(adv.two_proportion_z(0, 0, 5, 10), 0.0, "an empty world")
    check_eq(adv.flip_bit(9)(b"\x00\x00"), b"\x00\x02", "flip_bit")

    # the tampering channel: honest endpoint c0, adversary endpoint wrapped
    def pair(rules):
        c0, c1 = loopback_pair(timeout=10)
        return c0, adv.TamperChannel(c1, rules)

    def exchange_both(c0, tc, msgs0, msgs1):
        got = [[], []]

        def honest():
            try:
                for m in msgs0:
                    got[0].append(c0.exchange(m))
            except ProtocolAbort as exc:
                got[0].append(("abort", str(exc)))

        th = threading.Thread(target=honest, daemon=True)
        th.start()
        try:
            for m in msgs1:
                got[1].append(tc.exchange(m))
        except adv.AdversaryStop as exc:
            got[1].append(("stop", str(exc)))
        th.join(10)
        return got

    c0, tc = pair([adv.Rewrite("(unmarked)", 1, lambda m: m.upper())])
    got = exchange_both(c0, tc, [b"a", b"b", b"c"], [b"x", b"y", b"z"])
    check_eq(got, [[b"x", b"Y", b"z"], [b"a", b"b", b"c"]], "Rewrite changes one outgoing message")
    check_eq(tc.sent[("(unmarked)", 1)], b"Y", "the wrapper records what was really sent")
    check_eq(tc.party, 1, "attributes are delegated to the endpoint")

    c0, tc = pair([adv.Replay("(unmarked)", 2, ("(unmarked)", 0))])
    got = exchange_both(c0, tc, [b"a", b"b", b"c"], [b"x", b"y", b"z"])
    check_eq(got[0], [b"x", b"y", b"x"], "Replay re-sends an earlier own message")

    c0, tc = pair([adv.Replay("(unmarked)", 0, b"canned")])
    check_eq(exchange_both(c0, tc, [b"a"], [b"x"])[0], [b"canned"], "Replay of given bytes")

    c0, tc = pair([adv.Mirror("(unmarked)", 1)])
    got = exchange_both(c0, tc, [b"a", b"b", b"c"], [b"x", b"y", b"z"])
    check_eq(got, [[b"x", b"b", b"z"], [b"a", b"b", b"c"]], "Mirror sends the honest message back")

    c0, tc = pair([adv.StopAfterReceive("(unmarked)", 1)])
    got = exchange_both(c0, tc, [b"a", b"b", b"c"], [b"x", b"y", b"z"])
    check_eq(got[0], [b"x", ("abort", "peer closed")], "StopAfterReceive: the honest party sees the peer vanish")
    check_eq((tc.captured, got[1][1][0]), (b"b", "stop"), "the adversary has read the honest message first")

    # rules armed later count from the moment they are added
    c0, tc = pair([])
    exchange_both(c0, tc, [b"a"], [b"x"])
    tc.add(adv.Rewrite("(unmarked)", 0, lambda m: b"late"))
    check_eq(exchange_both(c0, tc, [b"b", b"c"], [b"y", b"z"])[0], [b"late", b"z"], "a rule added later")
    check_eq(tc.position("(unmarked)"), 3, "operation counter per phase")

    # the deviating checker keeps a consistent view and compensates with (Delta_a + g) e
    s, bits = 4, 8
    d0, d1 = 9, 6
    sh = deal(77, d0 + d1, bits)
    c0, c1 = loopback_pair(timeout=10)
    honest = ArithChecker(0, c0, b"t", d0, s)
    cheat = adv.CheatingChecker(1, c1, b"t", d1, s, value_errors={0: 5}, compensate=True)
    out = {}

    def run_honest():
        out["opened"] = honest.open_deferred(sh[0])
        try:
            honest.check()
            out["passed"] = True
        except ProtocolAbort:
            out["passed"] = False

    th = threading.Thread(target=run_honest, daemon=True)
    th.start()
    opened = cheat.open_deferred(sh[1])
    try:
        cheat.check()
        adv_passed = True
    except ProtocolAbort:
        adv_passed = False
    th.join(10)
    check_eq((opened, out["opened"]), (82, 82), "both parties see the wrong opening 77 + 5")
    check_eq(cheat.errors, {0: 5}, "the cheater knows its error")
    check_eq(adv_passed, out["passed"], "the cheater knows the verdict of the honest party")
    check_eq(len(cheat.guesses), 1, "one guess of Delta_h per check")

    # Rush: the adversary reads the honest message of the exchange before it answers
    c0, tc = pair([adv.Rush("(unmarked)", 1, lambda own, seen: own + b"<" + seen)])
    got = exchange_both(c0, tc, [b"a", b"b", b"c"], [b"x", b"y", b"z"])
    check_eq(got, [[b"x", b"y<b", b"z"], [b"a", b"b", b"c"]], "Rush answers after it has seen the honest message")
    check_eq(tc.sent[("(unmarked)", 1)], b"y<b", "Rush: the wrapper records what was really sent")

    # exact binomial tests against rational arithmetic
    def exact_two_sided(k, n, num, den):
        pmf = [math.comb(n, i) * num ** i * (den - num) ** (n - i) for i in range(n + 1)]
        return sum(v for v in pmf if v <= pmf[k]) / den ** n

    def exact_upper(k, n, num, den):
        return sum(math.comb(n, i) * num ** i * (den - num) ** (n - i) for i in range(k, n + 1)) / den ** n

    for k, n, num, den in ((3, 8, 1, 2), (0, 8, 1, 2), (8, 8, 1, 2), (22, 400, 1, 256), (310, 400, 3, 4),
                           (263, 400, 3, 4), (0, 48, 1, 2), (24, 48, 1, 2), (49, 400, 5, 16), (5, 5, 1, 1)):
        got = adv.binomial_two_sided_p(k, n, num / den)
        check(abs(got - exact_two_sided(k, n, num, den)) <= 1e-9 * max(got, 1e-300) + 1e-15,
              f"two-sided binomial p-value, {k} of {n} at p = {num}/{den}")
        up = adv.binomial_upper_p(k, n, num / den)
        check(abs(up - exact_upper(k, n, num, den)) <= 1e-9 * max(up, 1e-300) + 1e-15,
              f"upper binomial tail, {k} of {n} at p = {num}/{den}")
    check_eq((adv.binomial_two_sided_p(0, 0, 0.3), adv.binomial_upper_p(0, 7, 0.3)), (1.0, 1.0), "degenerate cases")
    check(adv.binomial_two_sided_p(48, 48, 0.5) < adv.binomial_two_sided_p(10, 48, 0.5) < adv.SIGNIFICANCE
          < adv.binomial_two_sided_p(11, 48, 0.5),
          "48 real-pipeline trials at p = 1/2: the test accepts 11 to 37 passes and rejects the rest at 10^-4")
    check(adv.binomial_two_sided_p(8, 8, 0.5) > adv.SIGNIFICANCE,
          "8 trials could not reject anything at 10^-4 (the reason for raising the real rows to 48)")
    check(raises_value_error(lambda: adv.binomial_two_sided_p(5, 4, 0.5)), "more successes than trials")
    check_eq([adv.high_error_pass_rate(j) for j in (1, 2, 3, 4, 8)], [3 / 4, 1 / 2, 5 / 16, 6 / 32, 10 / 512],
             "pass probability (j + 2) / 2^(j+1) of an error 2^(m-j)")
    # ... which is Pr[Delta chi = 0 mod 2^j] for uniform Delta, chi (enumeration for small j)
    for j in (1, 2, 3, 4):
        hits = sum(1 for d in range(1 << j) for c in range(1 << j) if (d * c) % (1 << j) == 0)
        check_eq(hits / 4 ** j, adv.high_error_pass_rate(j), f"enumeration of Delta chi = 0 mod 2^{j}")

    # work units
    tasks = adv.plan_tasks({"A1": 45}, {"A2": 30}, seed=3, chunk=20)
    check_eq([(t[0], t[1], t[2]) for t in tasks],
             [("attack", "A1", 20), ("attack", "A1", 20), ("attack", "A1", 5),
              ("distinguish", "A2", 20), ("distinguish", "A2", 10)], "work units")
    check_eq(len({t[3] for t in tasks if t[1] == "A1"}), 3, "every unit has its own seed")
    check(set(adv.DISTINGUISHERS) >= {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A14", "A14-wire"},
          "a distinguishing experiment exists for each of A1 - A7 and for A14")
    prefixes = {name.split("-")[0] for name in adv.ATTACKS}
    check(prefixes >= {f"A{i}" for i in range(1, 16)}, "the catalogue covers A1 - A15")
    real_rows = [name for name, spec in adv.ATTACKS.items() if spec[1].prep == "real"]
    check(sorted(real_rows) == ["A4-real", "A5-real"] and all(adv.ATTACKS[n][5] >= 48 for n in real_rows),
          "the rows on the complete real pipeline have at least 48 trials")

    # the honest x shares of the leaky triples are recorded on the real pipeline as well
    class _Bdoz:
        def __init__(self, bits):
            self.bits = bits

    class _Source:
        def random_bdoz(self, n, cheat=None):
            return _Bdoz(np.arange(n) % 2), None

    rec = adv._RecordingAbits(_Source())
    rec.random_bdoz(6)
    rec.random_bdoz(4)
    check_eq(rec.first_bits.tolist(), [0, 1, 0, 1, 0, 1], "the recorder keeps the bits of the first request only")


# --------------------------------------------------------------------------- #
# Deviations
# --------------------------------------------------------------------------- #

def test_deviations():
    names = [n for n in adv.ATTACKS if not (NO_REAL and adv.ATTACKS[n][1].prep == "real")]
    plan = {name: trials_for(name) for name in names}
    start = time.perf_counter()
    records, _ = adv.run_tasks(adv.plan_tasks(plan, {}, SEED), JOBS)
    elapsed = time.perf_counter() - start
    rows = []
    for name in names:
        recs = records[name]
        summary = adv.summarise(name, recs)
        _, setting, _, _, expect, _ = adv.ATTACKS[name]
        tag = f"{name} ({summary['description']})"
        check_eq(summary["errors"], 0, f"{tag}: no experiment error ({summary['first_error']})")
        check(summary["session_dead_after_every_abort"],
              f"{tag}: after every abort the session is dead and sends nothing")
        check(summary["ok"], f"{tag}: expectation not met: {json.dumps(summary, default=str)[:600]}")
        if expect == "always":
            check_eq(summary["detected"], summary["effective_trials"] + sum(
                1 for r in recs if not r.get("effective", True) and r["aborted"]),
                f"{tag}: every effective deviation detected")
            eff = [r for r in recs if r.get("effective", True)]
            check(all(r["aborted"] and not r["accepted"] for r in eff),
                  f"{tag}: the honest party released no output in any effective trial")
            check_eq(summary["wrong_outputs_accepted"], 0, f"{tag}: no wrong output accepted")
        elif expect == "refused":
            check_eq(summary["refused"], summary["mask_attempts_on_used_offsets"], f"{tag}: every attempt refused")
            check_eq(summary["messages_sent_by_refused_attempts"], 0, f"{tag}: refused attempts send nothing")
        else:
            kind, p = expect
            rate = summary["measured_pass_rate"]
            pvalue = summary["binomial_p_value"]
            if kind == "rate":
                check_eq(pvalue, adv.binomial_two_sided_p(summary["passes"], summary["trials"], p),
                         f"{tag}: the recorded p-value is the exact two-sided binomial test")
                check(pvalue >= adv.SIGNIFICANCE,
                      f"{tag}: {summary['passes']} passes of {summary['trials']} (rate {rate:.3f}) against the "
                      f"predicted {p:.4f}: exact two-sided binomial p-value {pvalue:.3g}")
                check_eq(summary["wrong_outputs_accepted"], 0,
                         f"{tag}: every run that passes yields a correct, authenticated result")
            else:
                check(pvalue >= adv.SIGNIFICANCE,
                      f"{tag}: pass rate {rate:.3f} is not above the bound {p:.4f} (exact one-sided p-value "
                      f"{pvalue:.3g})")
                check_eq(setting.s, 4, f"{tag}: a pass that accepts a wrong value only at the tiny s")
        if "honest_triple_requests_max" in summary:
            check(summary["honest_triple_requests_max"] <= 1,
                  f"{tag}: the honest party drew at most ONE bucketing population (strict mode)")
        rows.append((name, summary))
    by = dict(rows)
    # facts specific to single deviations
    for name in by:
        s = by[name]
        if name.startswith("A4"):
            check_eq(s.get("trials_with_recorded_honest_shares"), s["trials"],
                     f"{name}: the honest x shares were recorded in every trial ({s['setting']})")
            check(s.get("pass_iff_all_guesses_right") is True,
                  f"{name}: the triple check passes iff every guess is right")
        if name.startswith("A14"):
            check(s["payload_never_wrong"],
                  f"{name}: the payload of the attacked opening is unchanged in every trial")
            check_eq(s["wrong_outputs_accepted"], 0, f"{name}: no wrong output is ever accepted")
            check_eq(s["aborts_whose_verdict_depended_on_the_key"], s["detected"],
                     f"{name}: every abort is a statement about the MAC key (the error hit a key-dependent term)")
        if name.startswith("A15"):
            check_eq((s["x_hat_accepted_by_honest_party"], s["arith_openings_sent_by_honest_party"]), (0, 0),
                     f"{name}: the honest party accepts no x_hat and sends nothing afterwards")
        if name.startswith("A4") or name.startswith("A5"):
            check(s.get("aborts_before_any_input_message", True),
                  f"{name}: the abort happens during the session setup, before any key-generation input")
        if name.startswith("A5"):
            check(s["abort_iff_key_bit_is_1"],
                  f"{name}: the extension check fails iff the attacked bit of the honest MAC key is 1")
        if name in ("A3", "A13-bool"):
            check_eq(s["output_opening_sent_by_honest_party"], 0,
                     f"{name}: the first BoolCheck fails before the outputs are opened")
        if name.startswith("A6"):
            check_eq(s["honest_output_share_sent"], 0,
                     f"{name}: the deferred check fails before the honest output share is sent")
        if "verdict_equals_prediction" in s:
            check(s["verdict_equals_prediction"],
                  f"{name}: in every trial the honest verdict is the one the algebra of the check predicts")
        if name == "A12-value":
            check_eq(s["adversary_learned_output"], s["trials"],
                     "A12: the adversary has the output and the honest party outputs abort")
    if "A6-tiny" in by and "A7-tiny" in by:
        for name in ("A6-tiny", "A7-tiny"):
            check(by[name]["predicted_pass_rate"] == lemma2_bound(4)
                  and by[name]["binomial_p_value"] >= adv.SIGNIFICANCE,
                  f"{name}: measured forgery rate within the Lemma 2 bound for s = 4")
    width = max(len(n) for n in by)
    for name, s in rows:
        extra = (f"  pass {s['measured_pass_rate']:.3f} (predicted {s['predicted_pass_rate']:.4f})"
                 if "measured_pass_rate" in s else "")
        print(f"    {name:<{width}} trials {s['trials']:>4}  detected {s['detected']:>4}  "
              f"released {s['honest_outputs_released']:>3}  wrong {s['wrong_outputs_accepted']:>3}{extra}",
              flush=True)
    total = sum(s["trials"] for s in by.values())
    NOTES.append(f"{len(rows)} deviations, {total} trials in {elapsed:.0f} s "
                 f"({'catalogue counts' if FULL else 'reduced counts; --full for the counts of run_all.py'})")


# --------------------------------------------------------------------------- #
# Selective failure: the distinguishing experiment
# --------------------------------------------------------------------------- #

def test_no_selective_failure():
    plan = {name: RUNS_PER_WORLD for name in adv.DISTINGUISHERS}
    start = time.perf_counter()
    _, records = adv.run_tasks(adv.plan_tasks({}, plan, SEED), JOBS)
    elapsed = time.perf_counter() - start
    for name in adv.DISTINGUISHERS:
        d = adv.summarise_distinguisher(name, records[name], run_all.Z_THRESHOLD)
        check_eq(d["errors"], 0, f"{name}: no experiment error ({d['first_error']})")
        check_eq(d["runs_per_world"], [RUNS_PER_WORLD, RUNS_PER_WORLD], f"{name}: runs per world")
        f0, f1 = d["abort_frequency"]
        if d["positive_control"]:
            check(abs(d["z"]) >= run_all.Z_THRESHOLD,
                  f"{name}: the leaky baseline IS distinguished (z = {d['z']:.2f})")
            check_eq((d["aborts"][0], d["aborts"][1]), (0, RUNS_PER_WORLD),
                     f"{name}: the baseline aborts iff the index bit differs from the guess")
        else:
            check(abs(d["z"]) < run_all.Z_THRESHOLD,
                  f"{name}: abort frequencies {f0:.3f} / {f1:.3f} are indistinguishable (z = {d['z']:.2f})")
        if "verdict_equals_prediction" in d:
            check(d["verdict_equals_prediction"],
                  f"{name}: every abort / non-abort is the one the algebra of the check predicts")
        check(d["ok"], f"{name}: verdict")
        print(f"    {name:<14} aborts {d['aborts'][0]:>3} / {d['aborts'][1]:>3} of {RUNS_PER_WORLD} "
              f"per world, z = {d['z']:6.2f}{'  (positive control)' if d['positive_control'] else ''}",
              flush=True)
    NOTES.append(f"distinguishing experiment: {len(plan)} adversaries, {RUNS_PER_WORLD} runs per world, "
                 f"{elapsed:.0f} s")


def test_run_all_report_shape():
    """``run_all.run`` on a tiny plan: the structure of the JSON record and of the printed table."""
    res = run_all.run({"A1": 2, "A7-tiny": 6, "A8-second": 1, "A14-half": 4, "A15-rush": 2},
                      {"A2": 5, "A14": 5, "control-leaky": 5}, SEED, 1, verbose=False)
    text = json.dumps(res, sort_keys=True)                    # everything is JSON-serialisable
    back = json.loads(text)
    check_eq(sorted(back), ["all_ok", "attacks", "distinguishing", "seconds"], "top-level keys")
    for name, a in back["attacks"].items():
        for key in ("trials", "detected", "honest_outputs_released", "wrong_outputs_accepted", "setting",
                    "description", "expected", "ok"):
            check(key in a, f"{name}: field {key}")
    check("measured_pass_rate" in back["attacks"]["A7-tiny"]
          and back["attacks"]["A7-tiny"]["predicted_pass_rate"] == lemma2_bound(4), "pass-rate fields")
    for name in ("A7-tiny", "A14-half"):
        for key in ("passes", "binomial_p_value", "significance", "statistical_test"):
            check(key in back["attacks"][name], f"{name}: field {key}")
        check("five_sigma" not in back["attacks"][name], f"{name}: the normal-approximation band is gone")
    half = back["attacks"]["A14-half"]
    check(half["predicted_pass_rate"] == 0.75 and half["payload_never_wrong"] is True
          and half["wrong_outputs_accepted"] == 0 and "honest_triple_requests_max" in half,
          "A14 row: prediction 3/4, payload never wrong, populations of the honest party")
    check(back["attacks"]["A15-rush"]["x_hat_accepted_by_honest_party"] == 0, "A15 row: no x_hat accepted")
    for name, d in back["distinguishing"].items():
        for key in ("runs_per_world", "aborts", "abort_frequency", "z", "threshold", "ok"):
            check(key in d, f"{name}: field {key}")
    lines = run_all.format_tables(res)
    check(any(line.startswith("A1 ") for line in lines) and any("positive control" in line for line in lines),
          "printed table")
    check(run_all.DEFAULT_OUT.parts[-3:] == ("experiments", "raw", "attacks.json"), "default output path")


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    if "-k" in sys.argv:
        key = sys.argv[sys.argv.index("-k") + 1]
        tests = [(name, fn) for name, fn in tests if key in name]
    failed = []
    start = time.perf_counter()
    print(f"adversarial regression tests, {len(tests)} test functions "
          f"({'full' if FULL else 'reduced'} counts, {RUNS_PER_WORLD} runs per world, {JOBS} process(es))")
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
