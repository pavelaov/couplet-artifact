"""Run every adversarial experiment and record the results (DESIGN.md section 9, appendix).

Usage (from artifact/python, or from anywhere with the full path):

    python -m couplet.attacks.run_all [--out PATH] [--jobs N] [--runs R] [--quick]
                                        [--only ID ...] [--seed S] [--no-real]
    python couplet/attacks/run_all.py ...                 (the same)

Part 1 runs every deviation of ``couplet.attacks.adversaries.ATTACKS`` and
reports, for each: the number of trials, how many the honest party detected
(it output abort), whether any honest output was released, how many released
outputs were wrong, and - for the deviations that pass with a known
probability (A4: 2^-t, A5: 1/2, A14: (j + 2) / 2^(j+1) for an error 2^(m-j)
that is 0 mod 2^(m-s); the tiny-s rows: at most the Lemma 2 bound) - the
number of passes with the p-value of the exact binomial test against the
prediction (two-sided for a predicted probability, one-sided for a bound;
significance 10^-4).  The honest party runs in strict mode on the exact
triple reservation of its scenario: ``honest_triple_requests_max`` must be 1.

Part 2 is the distinguishing experiment for the absence of selective failure:
for the adversaries A1 - A7 and A14 the honest party's abort indicator is recorded over
``R`` runs (default 200) in each of two honest-input worlds (a key index with
most significant bit 0 or 1; a gate input x = 0 or x = N - 1) and the two abort
frequencies are compared with a two-proportion z-score; ``|z| < 3.3`` is
required for every adversary.  The last row is a POSITIVE CONTROL: the same
experiment against the re-implemented one-bit-leaky baseline, where the z-score
must be large.

The results go to ``artifact/experiments/raw/attacks.json`` (``--out`` changes
the path) and a table is printed.  ``--jobs N`` spreads the work over N
processes; ``--quick`` divides all counts by five; ``--no-real`` skips the two
rows that run the complete real preprocessing (about three seconds per trial,
48 trials each).
The exit status is non-zero if any row fails its expectation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve()
if str(_HERE.parents[2]) not in sys.path:          # .../artifact/python
    sys.path.insert(0, str(_HERE.parents[2]))

import numpy as np                                                  # noqa: E402

from couplet.attacks import adversaries as adv                     # noqa: E402

DEFAULT_OUT = _HERE.parents[3] / "experiments" / "raw" / "attacks.json"
Z_THRESHOLD = 3.3
DEFAULT_RUNS = 200


def default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 4))


def run(attack_trials: dict, distinguisher_runs: dict, seed: int, jobs: int, verbose: bool = True) -> dict:
    """Run the given experiments; returns the result dictionary (without the meta block)."""
    tasks = adv.plan_tasks(attack_trials, distinguisher_runs, seed)
    done = {"n": 0}
    start = time.perf_counter()

    def progress(kind, name, count):
        done["n"] += 1
        if verbose and (done["n"] % 25 == 0 or done["n"] == len(tasks)):
            print(f"  ... {done['n']}/{len(tasks)} work units, {time.perf_counter() - start:.0f} s",
                  flush=True)

    attack_records, dist_records = adv.run_tasks(tasks, jobs, progress)
    attacks = {name: adv.summarise(name, attack_records[name]) for name in attack_trials}
    dist = {name: adv.summarise_distinguisher(name, dist_records[name], Z_THRESHOLD)
            for name in distinguisher_runs}
    return {"attacks": attacks, "distinguishing": dist,
            "all_ok": all(a["ok"] for a in attacks.values()) and all(d["ok"] for d in dist.values()),
            "seconds": round(time.perf_counter() - start, 1)}


def _rate(entry: dict) -> str:
    if "measured_pass_rate" not in entry:
        return ""
    return (f"pass {entry['measured_pass_rate']:.4f} / pred {entry['predicted_pass_rate']:.4f}"
            f" (p = {entry['binomial_p_value']:.2g})")


def format_tables(results: dict) -> list:
    lines = ["", "Part 1: deviations (effective = trials in which the deviation changed something the "
                 "honest party depends on)",
             f"{'id':<13} {'setting':<30} {'trials':>6} {'effective':>9} {'detected':>8} {'released':>8} "
             f"{'wrong':>5}  {'pass rate (measured / predicted, exact binomial p-value)':<46} ok",
             "-" * 140]
    for name, a in results["attacks"].items():
        lines.append(f"{name:<13} {a['setting']:<30} {a['trials']:>6} "
                     f"{a.get('effective_trials', a['trials']):>9} {a['detected']:>8} "
                     f"{a['honest_outputs_released']:>8} {a['wrong_outputs_accepted']:>5}  "
                     f"{_rate(a):<46} {'yes' if a['ok'] else 'NO'}")
        lines.append(f"{'':<13} {a['description']}")
    lines += ["", "Part 2: abort frequency in two honest-input worlds (two-proportion z-score, "
                  f"|z| < {Z_THRESHOLD} required; the control must exceed it)",
              f"{'adversary':<14} {'runs/world':>10} {'aborts w0':>9} {'aborts w1':>9} "
              f"{'freq w0':>8} {'freq w1':>8} {'z':>7}  ok",
              "-" * 80]
    for name, d in results["distinguishing"].items():
        f0, f1 = d["abort_frequency"]
        lines.append(f"{name:<14} {d['runs_per_world'][0]:>10} {d['aborts'][0]:>9} {d['aborts'][1]:>9} "
                     f"{f0:>8.3f} {f1:>8.3f} {d['z']:>7.2f}  {'yes' if d['ok'] else 'NO'}"
                     f"{'  (positive control)' if d['positive_control'] else ''}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--jobs", type=int, default=default_jobs(), help="worker processes")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="runs per world in part 2")
    ap.add_argument("--quick", action="store_true", help="one fifth of all counts")
    ap.add_argument("--only", nargs="*", default=None, help="subset of ids, e.g. A4-t1 A7")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--no-real", action="store_true", help="skip the rows on the complete real pipeline")
    args = ap.parse_args(argv)

    scale = 5 if args.quick else 1
    attack_trials = {name: max(2, spec[5] // scale) for name, spec in adv.ATTACKS.items()}
    dist_runs = {name: max(10, args.runs // scale) for name in adv.DISTINGUISHERS}
    if args.no_real:
        attack_trials = {n: t for n, t in attack_trials.items() if adv.ATTACKS[n][1].prep != "real"}
    if args.only:
        attack_trials = {n: t for n, t in attack_trials.items() if n in args.only}
        dist_runs = {n: t for n, t in dist_runs.items() if n in args.only}

    print(f"adversarial regression: {len(attack_trials)} deviations, {len(dist_runs)} distinguishing "
          f"experiments ({args.runs // scale} runs per world), {args.jobs} process(es)", flush=True)
    results = run(attack_trials, dist_runs, args.seed, args.jobs)
    results["meta"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "seed": args.seed, "jobs": args.jobs, "quick": bool(args.quick),
        "runs_per_world": args.runs // scale, "z_threshold": Z_THRESHOLD,
        "pass_rate_test": {
            "significance": adv.SIGNIFICANCE,
            "rate_rows": "exact two-sided binomial test of the number of passes against the predicted "
                         "probability (sum of the probabilities of all outcomes at most as likely as the "
                         "observed one)",
            "bound_rows": "exact one-sided binomial test: Pr[X >= passes] at the Lemma 2 bound",
        },
        "honest_party": "unmodified protocol code in strict mode (allow_refill = False) on the exact triple "
                        "reservation of its scenario; honest_triple_requests_max is the largest number of "
                        "bucketing populations any honest party drew (0 if it aborted before the first "
                        "request completed)",
        "row_notes": {
            "A13": "commitment mirror in BoolCheck / ArithCheck (unchanged identifier)",
            "A14": "errors that are 0 mod 2^(m-s) on a value share (the rows requested by the audit "
                   "follow-up; filed as A14 because A13 was already the mirror deviation): the payload is "
                   "unchanged in every trial, the check passes with probability (j + 2) / 2^(j+1) for the "
                   "error 2^(m-j), and the verdict depends only on the MAC key, the coefficient and the "
                   "error (verdict_equals_prediction), not on the honest inputs (part 2)",
            "A15": "shared-input Mask against a rushing party: caught by the commit-then-open exchange",
            "A4-real / A5-real": "complete real pipeline, 48 trials each; A4-real records the honest x "
                                 "shares, so pass_iff_all_guesses_right is asserted there as well",
        },
        "parties": "two threads of one process on an in-process channel",
        "settings": {"standard": adv.STANDARD.tag(), "tiny": adv.TINY.tag()},
        "preprocessing_labels": {
            "ideal": "trusted in-process dealer (section 7.7)",
            "hybrid": "authenticated bits from the dealer; leaky AND, bucketing and triple check real",
            "fast-real": "base-OT seeds from a trusted party; OT extension, authenticated bits and "
                         "AND triples real",
            "real": "complete pipeline including the Masny-Rindal base OT",
        },
        "columns": {
            "effective_trials": "trials in which the deviation changed a value the honest party depends on "
                                "(a substituted key that leaves the party's shares unchanged is not a deviation)",
            "detected": "trials in which the honest party output abort",
            "honest_outputs_released": "trials in which the honest party returned an output",
            "wrong_outputs_accepted": "released outputs that differ from the correct value",
        },
    }
    lines = format_tables(results)
    print("\n".join(lines))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"\nall expectations met: {results['all_ok']}; {results['seconds']} s; wrote {args.out}")
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
