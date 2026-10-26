"""Adversarial rows of the inference protocol for the evaluation (DESIGN_M3.md section 6 item 6).

Runs the deviations of ``tests/test_ppml_attacks.py`` (the same trial
functions, imported from that file) with the counts of a results table and
writes ``artifact/experiments/raw/attacks_ppml.json``:

  rows          one per deviation: trials in both truncation modes (RUNS_PER_WORLD
                runs per honest-input world in mode round, EXACT_TRIALS in mode
                exact), aborts of the honest party, outputs released, wrong outputs
                accepted (released logits that differ from the mirror on the
                inputs of the run, on the substituted pre-activation for
                first-layer), the state after every abort (dead and silent), the
                abort frequency in the two worlds with the two-proportion z-score
                (|z| < 3.3 required) and, for the randomised verdicts, whether every
                verdict equals the prediction of the check's algebra
  output_mask   correction C2: a malicious P_0 with logit weights 2^k e_j in
                private-model mode; the uniformity tests of the opened high bits
                (p-values) and the power control (``test_private_logit_mask``)

The deviations, the model (the tiny MLP [4, 2, 2, 2] at (4, 2), k = 32, s = 64;
the tiny setting k = 8, s = 4) and the expectations are those of the test
file's docstring.  Two threads of one process per run, ideal preprocessing
(the deviations target the online phase).  Exit status 0 iff every row meets
its expectation.

Usage (from artifact/python):
    python bench/ppml_attacks.py [--runs 100] [--exact-trials 20] [--mask-runs 400] [--jobs 4] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
_TESTS = os.path.join(_PY, "tests")
for _p in (_PY, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import test_ppml_attacks as tpa  # noqa: E402

DEFAULT_OUT = os.path.normpath(os.path.join(_PY, "..", "experiments", "raw", "attacks_ppml.json"))
DESCRIPTIONS = {
    "deferred": "wrong value share (+1) in the first deferred opening (truncation opening of layer 2)",
    "deferred-high": "error 2^(m-1) in the first deferred opening (0 mod 2^(m-s)): passes with probability 3/4",
    "release": "wrong value share (+1) of the logit opening",
    "release-high": "error 2^(w-1) on the logit opening: passes with probability 3/4, decodes the payload mod 2^k",
    "first-layer": "error on the own first-layer share, used consistently: an input substitution",
    "wrong-key": "evaluates a plane A key that was not generated in the session",
    "replay": "replays the value share of opening 0 at opening 1",
    "tiny-guess": "k = 8, s = 4: wrong share (+1) compensated by a guess of the honest MAC key share",
}
EXPECT = {"deferred": "always", "release": "always", "wrong-key": "always", "replay": "always",
          "first-layer": "never", "deferred-high": "rate", "release-high": "rate", "tiny-guess": "guess"}


def _job(args):
    name, world, mode, seed = args
    try:
        res = tpa.trial(name, world, mode, seed)
    except Exception:  # noqa: BLE001
        return {"name": name, "world": world, "mode": mode, "aborted": None, "error": traceback.format_exc()[-2000:]}
    keep = ("name", "world", "mode", "aborted", "error", "dead", "silent", "released", "predicted_pass")
    out = {k: res.get(k) for k in keep}
    out["released_logits"] = res["honest_logits"] is not None
    out["agree"] = res["honest_logits"] is not None and res["honest_logits"] == res.get("adv_logits")
    mirror = res.get("mirror_logits")
    out["correct"] = None if res["honest_logits"] is None or mirror is None else res["honest_logits"] == mirror
    return out


def _mask_job(args):
    return tpa.mask_trial(args)


def summarise(name: str, recs: list, runs: int) -> dict:
    errors = [r["error"] for r in recs if r.get("error")]
    good = [r for r in recs if not r.get("error")]
    rnd = [r for r in good if r["mode"] == "round"]
    aborts = [sum(1 for r in rnd if r["world"] == w and r["aborted"]) for w in (0, 1)]
    per_world = [sum(1 for r in rnd if r["world"] == w) for w in (0, 1)]
    z = tpa.two_proportion_z(aborts[0], per_world[0], aborts[1], per_world[1])
    detected = sum(1 for r in good if r["aborted"])
    released = sum(1 for r in good if r["released_logits"])
    wrong = sum(1 for r in good if r["released_logits"] and r["correct"] is False)
    dead = all(r.get("dead") and r.get("silent") and not r.get("released") for r in good if r["aborted"])
    row = {"id": name, "description": DESCRIPTIONS[name], "setting": "tiny (k = 8, s = 4)" if name == "tiny-guess" else
           "standard (k = 32, s = 64)", "trials": len(recs),
           "trials_by_mode": {m: sum(1 for r in recs if r["mode"] == m) for m in ("round", "exact")},
           "detected": detected, "honest_outputs_released": released, "wrong_outputs_accepted": wrong,
           "parties_agree_when_released": all(r["agree"] for r in good if r["released_logits"]),
           "dead_and_silent_after_every_abort": dead,
           "runs_per_world_round": per_world, "aborts_per_world_round": aborts,
           "abort_frequency_round": [aborts[w] / per_world[w] if per_world[w] else None for w in (0, 1)],
           "z": z, "z_threshold": tpa.Z_THRESHOLD, "errors": len(errors), "first_error": errors[0] if errors else None,
           "expectation": EXPECT[name]}
    ok = not errors and dead and abs(z) < tpa.Z_THRESHOLD and all(per_world)
    exp = EXPECT[name]
    if exp == "always":
        ok = ok and detected == len(good) and released == 0
        row["expected"] = "every run detected; no output released"
    elif exp == "never":
        ok = ok and detected == 0 and wrong == 0 and row["parties_agree_when_released"]
        row["expected"] = "never aborts; both parties output the mirror on the substituted pre-activation"
    else:
        pred = [r.get("predicted_pass") for r in good]
        row["verdicts_as_predicted"] = all(p is not None for p in pred) and all(r["aborted"] == (not r["predicted_pass"])
                                                                                   for r in good)
        ok = ok and row["verdicts_as_predicted"]
        if exp == "rate":
            row["predicted_pass_rate"] = tpa.RANDOMISED[name]
            row["measured_pass_rate"] = (len(good) - detected) / len(good) if good else None
            ok = ok and wrong == 0
            row["expected"] = ("passes with probability 3/4 (verdict predicted run by run); a passing run releases the "
                               "correct logits")
        else:
            row["predicted_pass_rate"] = None
            row["measured_pass_rate"] = (len(good) - detected) / len(good) if good else None
            row["expected"] = ("passes only when the guess of the MAC key share is right (verdict predicted run by run); "
                               "wrong_outputs_accepted counts the passing runs whose logits differ from the mirror on "
                               "the thresholds the honest party opened")
    row["ok"] = bool(ok)
    return row


def mask_rows(results: list) -> dict:
    """The statistics of test_private_logit_mask over the given runs."""
    k, s = tpa.MASK_K, tpa.MASK_S
    w = k + s
    errors = [r["error"] for r in results if r.get("error")]
    if errors:
        return {"ok": False, "errors": len(errors), "first_error": errors[0]}
    high, control = [], []
    agree = functional = low_ok = control_ok = True
    for r in results:
        l0, l1 = r["logits"]
        agree = agree and l0 == l1 and r["released_adv"] == r["released_honest"]
        functional = functional and l1 == r["functionality"]
        for i, v in enumerate(r["released_adv"]):
            low_ok = low_ok and tpa.to_signed(v % (1 << k), k) == (r["bias"][i] << r["f"])
            if r["mode"] != "round":
                continue
            high.append(v >> k)
            u = ((v - r["masks"][i]) % (1 << w)) >> k
            control_ok = control_ok and tpa.to_signed(u, s) == r["a_j"] + ((r["bias"][i] << r["f"]) >> k)
            control.append(u)
    m = len(high)
    p_low = tpa.chi2_sf(tpa.chi2_nibble(high, 0), 15)
    p_top = tpa.chi2_sf(tpa.chi2_nibble(high, s - 4), 15)
    ones = sum(bin(h).count("1") for h in high)
    z = (ones - 32.0 * m) / math.sqrt(16.0 * m) if m else 0.0
    z_thr = statistics.NormalDist().inv_cdf(1 - tpa.ALPHA_MASK / 2)
    p_ctrl = tpa.chi2_sf(tpa.chi2_nibble(control, 0), 15)
    ok = (agree and functional and low_ok and control_ok and p_low > tpa.ALPHA_MASK and p_top > tpa.ALPHA_MASK
          and abs(z) < z_thr and p_ctrl < tpa.ALPHA_MASK)
    return {"id": "logit-mask", "description": "private-model mode, malicious P_0 inputs the logit weights 2^k e_j "
                                               "(correction C2): the opened high bits must be uniform",
            "runs_round": sum(1 for r in results if r["mode"] == "round"),
            "runs_exact": sum(1 for r in results if r["mode"] == "exact"), "masked_logits": m,
            "both_parties_same_logits_and_opening": agree, "honest_output_is_functionality": functional,
            "low_k_bits_are_b_2f": low_ok, "p_low_nibble": p_low, "p_top_nibble": p_top, "z_ones": z,
            "z_threshold": z_thr, "alpha_per_test": tpa.ALPHA_MASK,
            "control_unmasked_high_bits_equal_a_j": control_ok, "control_p_low_nibble": p_ctrl, "ok": bool(ok)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="adversarial rows of the inference protocol")
    ap.add_argument("--runs", type=int, default=tpa.RUNS, help="runs per honest-input world in mode round")
    ap.add_argument("--exact-trials", type=int, default=20, help="trials per deviation in mode exact")
    ap.add_argument("--mask-runs", type=int, default=tpa.MASK_RUNS, help="runs of the output-mask test (mode round)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    tasks = [(name, world, "round", 7919 * i + 13 * world + len(name) + 104729)
             for name in tpa.ADVERSARIES for world in (0, 1) for i in range(args.runs)]
    tasks += [(name, i % 2, "exact", 6007 * i + len(name) + 7) for name in tpa.ADVERSARIES for i in range(args.exact_trials)]
    mask_tasks = [("exact", 501), ("exact", 502)] + [("round", 19001 + i) for i in range(args.mask_runs)]
    with multiprocessing.get_context("spawn").Pool(args.jobs) as pool:
        recs = pool.map(_job, tasks, chunksize=4)
        mask_results = pool.map(_mask_job, mask_tasks, chunksize=2)
    rows = [summarise(name, [r for r in recs if r["name"] == name], args.runs) for name in tpa.ADVERSARIES]
    mask = mask_rows(mask_results)
    doc = {"schema": "couplet.attacks_ppml/1", "generated_by": "python bench/ppml_attacks.py",
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
           "parameters": {"runs_per_world_round": args.runs, "exact_trials": args.exact_trials,
                          "mask_runs": args.mask_runs, "jobs": args.jobs, "model": "mlp4 [4, 2, 2, 2] at (4, 2), random "
                          "integer weights (seed 11)", "worlds": "honest features all 0 or all 1.0; the adversary's 0.5",
                          "preprocessing": "ideal (trusted in-process dealer); the deviations target the online phase",
                          "parties": "two threads of one process on an in-process channel"},
           "rows": rows, "output_mask": mask, "all_ok": all(r["ok"] for r in rows) and mask["ok"],
           "seconds": round(time.perf_counter() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"{'deviation':<14} {'trials':>6} {'detected':>8} {'released':>8} {'wrong':>5} {'w0':>5} {'w1':>5} {'z':>6}  ok")
    for r in rows:
        print(f"{r['id']:<14} {r['trials']:>6} {r['detected']:>8} {r['honest_outputs_released']:>8} "
              f"{r['wrong_outputs_accepted']:>5} {r['aborts_per_world_round'][0]:>5} {r['aborts_per_world_round'][1]:>5} "
              f"{r['z']:>6.2f}  {'yes' if r['ok'] else 'NO'}")
    if mask.get("masked_logits"):
        print(f"logit-mask: {mask['masked_logits']} masked logits, p(low) = {mask['p_low_nibble']:.3f}, "
              f"p(top) = {mask['p_top_nibble']:.3f}, z(ones) = {mask['z_ones']:.2f}, control p = {mask['control_p_low_nibble']:.2g}"
              f"  {'yes' if mask['ok'] else 'NO'}")
    print(f"all expectations met: {doc['all_ok']}; {doc['seconds']} s; wrote {args.out}")
    return 0 if doc["all_ok"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
