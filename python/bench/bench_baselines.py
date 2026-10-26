"""Benchmark of the two Python baselines (DESIGN.md section 9, Q1).

Runs, for every requested domain size n,

  * ``leaky_zhang``    the re-implemented one-bit-leaky protocol Pi_DPF of
                       Zhang et al. on the real preprocessing (base OT, OT
                       extension with consistency check, authenticated bits);
  * ``semihonest_ds``  the semi-honest Doerner-shelat-style generation on the
                       same tree (base OT, unchecked extension, no MACs);

each in a fresh session per repetition, and records bytes, rounds and seconds
per channel phase.  It also measures the bytes per authenticated AND triple of
our backend for the two bucket sizes that occur in the comparison (B = 4 and
B = 3) and evaluates the cost models of ``couplet.baselines.costmodels`` with
these measured figures.  Everything is written to

    artifact/experiments/raw/baselines_python.json

How to read the numbers
  * Both parties are two threads of ONE Python process, connected by TCP over
    127.0.0.1 (``--loopback`` uses in-process queues instead).  Seconds are
    wall-clock times of a phase at one endpoint, including the time it waits
    for the peer; the summary takes the larger of the two endpoints and the
    median over the repetitions.  This is a pure-Python/numpy reference
    implementation: the times are not comparable with the C++ numbers of the
    evaluation or with the numbers of the original papers.
  * Bytes are payload bytes sent (the TCP framing adds 4 bytes per message and
    is reported separately); "both" is the sum over the two parties.
  * Both baselines output shares of the whole truth table (2^n entries) with a
    GF(2^128) payload.

Usage (from artifact/python):

    python bench/bench_baselines.py                 # n = 8, 12, 16, 20; 3 repetitions
    python bench/bench_baselines.py --n 8 12 --reps 5 --loopback --no-triples
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import gf128 as gf  # noqa: E402
from couplet.baselines import costmodels as cm  # noqa: E402
from couplet.baselines import leaky_zhang as lz  # noqa: E402
from couplet.baselines import semihonest_ds as sh  # noqa: E402
from couplet.net import run_two_party  # noqa: E402
from couplet.prep import RealPrep  # noqa: E402

DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "experiments", "raw",
                                            "baselines_python.json"))
DEFAULT_NS = (8, 12, 16, 20)
TRIPLE_BATCHES = (50_000, 300_000)   # bucket sizes 4 and 3 at sigma = 40
PREP_PHASES = ("cotext", "abit", "laand", "bucket", "check")


def _random_case(n: int):
    """Random index and payload with random XOR shares: ((alpha, beta), shares of P_0, of P_1)."""
    alpha, a0 = secrets.randbelow(1 << n), secrets.randbelow(1 << n)
    beta, b0 = secrets.randbits(128), secrets.randbits(128)
    return (alpha, beta), (a0, b0), (alpha ^ a0, beta ^ b0)


def _unit(n: int, alpha: int) -> np.ndarray:
    unit = np.zeros(1 << n, dtype=np.uint8)
    unit[alpha] = 1
    return unit


def _phase_table(st0: dict, st1: dict) -> dict:
    """Per-phase record of one run from the two endpoints' statistics."""
    table = {}
    for name in st0["phases"]:
        p0, p1 = st0["phases"][name], st1["phases"].get(name)
        if p1 is None:
            raise RuntimeError(f"phase {name} is missing at party 1")
        if p0["rounds"] != p1["rounds"]:
            raise RuntimeError(f"round counts differ in phase {name}")
        table[name] = {
            "bytes_p0": p0["bytes_sent"], "bytes_p1": p1["bytes_sent"],
            "bytes_both": p0["bytes_sent"] + p1["bytes_sent"],
            "msgs_both": p0["msgs_sent"] + p1["msgs_sent"],
            "rounds": p0["rounds"],
            "seconds_p0": round(p0["time_s"], 6), "seconds_p1": round(p1["time_s"], 6),
        }
    return table


def _run_record(st0: dict, st1: dict, wall: float, verified: bool) -> dict:
    return {
        "phases": _phase_table(st0, st1),
        "bytes_both": st0["bytes_sent"] + st1["bytes_sent"],
        "rounds": st0["rounds"],
        "frame_overhead_bytes": st0["frame_overhead"] * (st0["msgs_sent"] + st1["msgs_sent"]),
        "seconds_p0": round(st0["time_s"], 6), "seconds_p1": round(st1["time_s"], 6),
        "wall_seconds": round(wall, 6),
        "outputs_verified": bool(verified),
    }


def _summarise(n: int, runs: list[dict], setup_phases=("baseot",)) -> dict:
    """Bytes and rounds (identical in every repetition) and median seconds per phase."""
    first = runs[0]["phases"]
    for run in runs[1:]:
        for name, rec in run["phases"].items():
            if (rec["bytes_both"], rec["rounds"]) != (first[name]["bytes_both"],
                                                     first[name]["rounds"]):
                raise RuntimeError(f"n = {n}: bytes or rounds of phase {name} vary between runs")
    phases = {}
    for name, rec in first.items():
        secs = [max(run["phases"][name]["seconds_p0"], run["phases"][name]["seconds_p1"])
                for run in runs]
        phases[name] = {"bytes_both": rec["bytes_both"], "bytes_p0": rec["bytes_p0"],
                        "bytes_p1": rec["bytes_p1"], "rounds": rec["rounds"],
                        "seconds_median": round(statistics.median(secs), 6),
                        "seconds_min": round(min(secs), 6), "seconds_max": round(max(secs), 6)}
    per_key = [name for name in phases if name not in setup_phases]
    return {
        "n": n,
        "repetitions": len(runs),
        "phases": phases,
        "total": {
            "bytes_both": sum(p["bytes_both"] for p in phases.values()),
            "rounds": sum(p["rounds"] for p in phases.values()),
            "seconds_median": round(sum(p["seconds_median"] for p in phases.values()), 6),
        },
        "per_key_without_base_ot": {
            "bytes_both": sum(phases[name]["bytes_both"] for name in per_key),
            "rounds": sum(phases[name]["rounds"] for name in per_key),
            "seconds_median": round(sum(phases[name]["seconds_median"] for name in per_key), 6),
        },
        "all_outputs_verified": all(run["outputs_verified"] for run in runs),
        "runs": runs,
    }


# --------------------------------------------------------------------------- #
# The two baselines
# --------------------------------------------------------------------------- #

def run_leaky(n: int, tcp: bool, tag: bytes) -> dict:
    """One session of the leaky protocol: setup (base OT) and one key on the real backend."""
    (alpha, beta), share0, share1 = _random_case(n)
    shares = (share0, share1)

    def party(role):
        def fn(chan):
            prep = RealPrep(role, chan, b"bench-leaky|" + tag)
            prep.setup()
            out = lz.run_pi_dpf(chan, prep, b"bench-leaky|" + tag, n, *shares[role])
            return out, prep.delta
        return fn

    t0 = time.perf_counter()
    (o0, d0), (o1, d1), st0, st1 = run_two_party(party(0), party(1), tcp=tcp)
    wall = time.perf_counter() - t0
    unit, delta = _unit(n, alpha), d0 ^ d1
    verified = (np.array_equal(o0.u.bits ^ o1.u.bits, unit)
                and np.array_equal(o0.u.macs ^ o1.u.macs, gf.bit_mul(unit, delta))
                and np.array_equal(o0.v.values ^ o1.v.values, gf.bit_mul(unit, beta))
                and np.array_equal(o0.v.macs ^ o1.v.macs, gf.bit_mul(unit, gf.mul(beta, delta))))
    return _run_record(st0, st1, wall, verified)


def run_semihonest(n: int, tcp: bool, tag: bytes) -> dict:
    """One session of the semi-honest baseline: setup (base OT) and one key."""
    (alpha, beta), share0, share1 = _random_case(n)
    shares = (share0, share1)

    def party(role):
        def fn(chan):
            session = sh.SemiHonestSession(role, chan, b"bench-semihonest|" + tag)
            session.setup()
            return session.dpf_gen(n, *shares[role])
        return fn

    t0 = time.perf_counter()
    o0, o1, st0, st1 = run_two_party(party(0), party(1), tcp=tcp)
    wall = time.perf_counter() - t0
    unit = _unit(n, alpha)
    verified = (np.array_equal(o0.t ^ o1.t, unit)
                and np.array_equal(o0.v ^ o1.v, gf.bit_mul(unit, beta)))
    return _run_record(st0, st1, wall, verified)


# --------------------------------------------------------------------------- #
# Measured bytes per AND triple of the backend
# --------------------------------------------------------------------------- #

def measure_triples(ell: int, tcp: bool) -> dict:
    """Run ``RealPrep.and_triples(ell)`` once; returns a measured cost entry (bytes per triple)."""
    def party(role):
        def fn(chan):
            prep = RealPrep(role, chan, b"bench-triples|" + str(ell).encode())
            prep.setup()
            a, b, c = prep.and_triples(ell)
            return a, b, c, prep.delta, prep.last_bucket_size
        return fn

    t0 = time.perf_counter()
    r0, r1, st0, st1 = run_two_party(party(0), party(1), tcp=tcp)
    wall = time.perf_counter() - t0
    (a0, b0, c0, d0, bucket), (a1, b1, c1, d1, _) = r0, r1
    delta = d0 ^ d1
    a, b, c = a0.bits ^ a1.bits, b0.bits ^ b1.bits, c0.bits ^ c1.bits
    verified = (np.array_equal(c, a & b)
                and np.array_equal(a0.macs ^ a1.macs, gf.bit_mul(a, delta))
                and np.array_equal(b0.macs ^ b1.macs, gf.bit_mul(b, delta))
                and np.array_equal(c0.macs ^ c1.macs, gf.bit_mul(c, delta)))
    table = _phase_table(st0, st1)
    prep_bytes = sum(table[name]["bytes_both"] for name in PREP_PHASES)
    prep_seconds = sum(max(table[name]["seconds_p0"], table[name]["seconds_p1"])
                       for name in PREP_PHASES)
    model = cm.backend_bytes_per_and(ell)
    return cm.entry(prep_bytes / ell, "bytes per AND triple (both parties)",
                    f"this run of bench/bench_baselines.py: RealPrep.and_triples({ell}), phases "
                    "cotext + abit + laand + bucket + check, base OT excluded", "measured",
                    ell=ell, bucket=bucket, bytes_both=prep_bytes,
                    seconds=round(prep_seconds, 4),
                    microseconds_per_triple=round(1e6 * prep_seconds / ell, 3),
                    wall_seconds=round(wall, 4), phases=table, triples_verified=bool(verified),
                    accounting_model=model["value"],
                    equals_accounting_model=abs(model["value"] - prep_bytes / ell) < 1e-9)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _environment(tcp: bool) -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "transport": "tcp 127.0.0.1" if tcp else "in-process queues",
        "parties": "two threads of one Python process",
    }


def _print_summary(name: str, rows: list[dict]) -> None:
    print(f"{name}:")
    for row in rows:
        key = row["per_key_without_base_ot"]
        print(f"  n = {row['n']:2d}: {key['bytes_both']:7d} bytes, {key['rounds']:3d} rounds, "
              f"{key['seconds_median']:.3f} s per key (base OT "
              f"{row['phases']['baseot']['seconds_median']:.2f} s, "
              f"{row['phases']['baseot']['bytes_both']} bytes); "
              f"verified: {row['all_outputs_verified']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_NS),
                        help="domain sizes (bits), default 8 12 16 20")
    parser.add_argument("--reps", type=int, default=3, help="repetitions per size (default 3)")
    parser.add_argument("--loopback", action="store_true",
                        help="in-process queues instead of TCP over 127.0.0.1")
    parser.add_argument("--no-triples", action="store_true",
                        help="skip the measurement of the backend's bytes per AND triple")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output JSON file")
    args = parser.parse_args(argv)
    if args.reps < 1 or any(not 1 <= n <= lz.MAX_N for n in args.n):
        parser.error(f"reps must be positive and n in 1..{lz.MAX_N}")
    tcp = not args.loopback
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t_all = time.perf_counter()

    leaky, semi = [], []
    for n in args.n:
        runs = [run_leaky(n, tcp, f"{n}|{rep}".encode()) for rep in range(args.reps)]
        row = _summarise(n, runs)
        expected = cm.zhang_instantiated_bytes(n)["value"]
        measured = sum(row["phases"][p]["bytes_both"] for p in lz.PHASES)
        row["pi_dpf_bytes_both"] = measured
        row["pi_dpf_bytes_match_model"] = measured == expected
        row["pi_dpf_rounds"] = sum(row["phases"][p]["rounds"] for p in lz.PHASES)
        abit = row["phases"]["cotext"]["bytes_both"] + row["phases"]["abit"]["bytes_both"]
        row["abit_bytes_both"] = abit
        row["abit_bytes_match_model"] = abit == cm.zhang_abit_bytes(n, "iknp_single")["value"]
        leaky.append(row)
        runs = [run_semihonest(n, tcp, f"{n}|{rep}".encode()) for rep in range(args.reps)]
        row = _summarise(n, runs)
        row["bytes_match_model"] = (row["per_key_without_base_ot"]["bytes_both"]
                                    == cm.semihonest_instantiated_bytes(n)["value"])
        semi.append(row)
        print(f"n = {n}: done ({time.perf_counter() - t_all:.1f} s elapsed)")

    measured_per_and, by_bucket = [], {}
    if not args.no_triples:
        for ell in TRIPLE_BATCHES:
            item = measure_triples(ell, tcp)
            measured_per_and.append(item)
            # the per-phase table stays with the measurement; the models get a slim copy
            by_bucket[item["bucket"]] = {k: v for k, v in item.items() if k != "phases"}
            print(f"and_triples({ell}): B = {item['bucket']}, {item['value']:.2f} bytes per triple, "
                  f"{item['seconds']:.2f} s")

    result = {
        "description": "Python baselines for DESIGN.md section 9, Q1: the re-implemented "
                       "one-bit-leaky protocol of Zhang et al. (S&P 2024, Figure 5) and a "
                       "semi-honest Doerner-shelat-style generation on the same tree, both on "
                       "this artifact's Python backend; plus the cost models of "
                       "couplet/baselines/costmodels.py.",
        "generated_by": "python bench/bench_baselines.py " + " ".join(sys.argv[1:] if argv is None
                                                                      else argv),
        "started": started,
        "environment": _environment(tcp),
        "conventions": {
            "bytes": "payload bytes sent; 'both' = party 0 + party 1; TCP framing (4 bytes per "
                     "message) is listed as frame_overhead_bytes and not included",
            "rounds": "as counted by couplet.net.Channel (a simultaneous exchange is one round)",
            "seconds": "wall-clock time of a phase at one endpoint including waiting for the "
                       "peer; summaries: larger endpoint, median over the repetitions; one fresh "
                       "session (with its own base OT) per repetition",
            "phases_leaky": "baseot, cotext, abit = authenticated bits from our preprocessing "
                            "(the original uses Ferret), summed in abit_bytes_both; the phases "
                            "zhang-* are summed in pi_dpf_bytes_both and are what the protocol "
                            "of Figure 5 itself sends; zhang-input = authentication of the "
                            "chosen input bits; zhang-root, zhang-levels, zhang-leaf, zhang-check "
                            "= steps 1, 2, 3 and 4-5 of Figure 5",
            "phases_semihonest": "baseot; sh-cot = unchecked IKNP rows for the n selections and "
                                 "the public offset W; sh-levels; sh-leaf",
            "payload": "GF(2^128) in both baselines; outputs are shares of all 2^n values",
            "caveat": "reference implementation in Python/numpy; times are not comparable with "
                      "native implementations or with the numbers reported in the original "
                      "papers. H0 is truncated Ascon-p12, not fixed-key AES.",
        },
        "leaky_zhang": leaky,
        "semihonest_ds": semi,
        "backend_bytes_per_and_measured": measured_per_and,
        "cost_models": cm.comparison_rows(tuple(args.n), measured_bytes_per_and=by_bucket),
        "reported_by_other_papers": {
            "zhang_et_al": cm.zhang_reported(),
            "authenticated_garbling_per_and": cm.authenticated_garbling_per_and(),
            "authenticated_garbling_aes": cm.authenticated_garbling_aes_reported(),
            "ferret_bits_per_cot": [cm.ferret_bits_per_cot("reg"), cm.ferret_bits_per_cot("uni")],
            "mascot_multiplication": cm.mascot_multiplication_bytes(),
        },
        "total_seconds": round(time.perf_counter() - t_all, 2),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    _print_summary("leaky_zhang (real preprocessing)", leaky)
    _print_summary("semihonest_ds", semi)
    ok = (all(r["all_outputs_verified"] and r["pi_dpf_bytes_match_model"]
              and r["abit_bytes_match_model"] for r in leaky)
          and all(r["all_outputs_verified"] and r["bytes_match_model"] for r in semi)
          and all(m["triples_verified"] for m in measured_per_and))
    print(f"written: {args.out}  ({result['total_seconds']} s; all outputs verified and byte counts "
          f"as modelled: {ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
