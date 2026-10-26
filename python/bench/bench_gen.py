"""Benchmark of Pi_Gen: one configuration per invocation, one JSON record per run.

Runs the dealer-free generation of DPF keys, DCF keys or gate material
(``couplet.protocol``) end to end: session setup (base OT, the persistent
wires of Delta), preprocessing (OT extension, authenticated bits, leaky ANDs,
bucketing), circuit evaluation and the output checks.  Bytes, rounds and
seconds are taken from the channel statistics per accounting phase.

Modes
  in-process (default)   both parties as two threads of this process, joined by
                         in-memory queues, or by loopback TCP sockets with ``--tcp``;
  ``--role 0|1``         ONE party of a two-process run over TCP: role 0 listens
                         on ``--host`` / ``--port``, role 1 connects.  Each process
                         writes its own record;
  ``--spawn``            convenience: starts the two ``--role`` processes itself,
                         checks their outputs against each other and writes one
                         merged record (the two processes hand their outputs to
                         the parent through temporary files for this check).

Examples (from artifact/python)
    python bench/bench_gen.py --family dpf --n 16
    python bench/bench_gen.py --family dcf --n 8 --beta secret --batch 4 --prep ideal
    python bench/bench_gen.py --family gate --gate comparison --n 8 --delay-ms 20 --tcp
    python bench/bench_gen.py --family dpf --n 12 --role 0 --port 47001     (first shell)
    python bench/bench_gen.py --family dpf --n 12 --role 1 --port 47001     (second shell)
    python bench/bench_gen.py --family dpf --n 12 --spawn

Record (``artifact/experiments/raw/gen_<circuit>_b<batch>_<prep>_s<sigma>_d<delay>_<mode>.json``):
parameters, circuit statistics, preprocessing parameters, per party the
channel totals and the per-phase bytes / rounds / seconds, phase groups,
totals, key sizes and the checks that were run.  ``checks.phase_sums_ok``
states that the per-phase bytes (and rounds) sum to the totals; the process
exits non-zero if a check fails.

One bucketing population.  The statistical bound 2^-sigma of the triples
holds per preprocessing request, so the session runs in strict mode
(``allow_refill=False``): the AND budget of the generation
(``protocol.and_budget``) is reserved at the session setup, in ONE request
together with the setup circuit, and the party fails (before writing a
record) unless it made exactly one triple request and used every triple.
``preprocessing.triple_requests`` and ``checks.single_bucketing_population``
record it.

Network emulation is application level inside the channel (``--delay-ms`` per
flight, ``--mbps`` per direction), as described in ``couplet.net``.

Ideal preprocessing (``--prep ideal``) uses a trusted dealer and is NOT a
protocol: such records carry ``"ideal_preprocessing": true`` and must never be
reported as all-in numbers.  In the two-process modes the dealer of each
process is derived from a shared seed (``--dealer-seed``).

All times are wall-clock times of a pure-Python reference implementation; in
the in-process mode both parties share one interpreter (and its lock), so
``wall_s`` is the time for BOTH parties on one machine.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import platform
import secrets
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import numpy as np  # noqa: E402

from couplet import abb  # noqa: E402
from couplet import circuits_gen as cg  # noqa: E402
from couplet import fss  # noqa: E402
from couplet import protocol as pg  # noqa: E402
from couplet.net import TcpChannel, run_two_party  # noqa: E402
from couplet.prep import IdealDealer, bucket_size  # noqa: E402
from couplet.session import Session  # noqa: E402

SCHEMA = "couplet.bench_gen/1"
DEFAULT_OUT = os.path.normpath(os.path.join(_PY, "..", "experiments", "raw"))
COUNTERS = ("bytes_sent", "bytes_recv", "msgs_sent", "msgs_recv", "rounds")

# accounting phase -> group of the phase breakdown (DESIGN.md section 9, Q3)
GROUPS = {
    "baseot": "base_ot", "cotext": "ot_extension", "abit": "authenticated_bits",
    "laand": "leaky_and", "bucket": "bucketing", "check": "triple_check",
    "setup-prep": "session_setup_circuit", "setup-input": "session_setup_circuit",
    "setup-online": "session_setup_circuit", "setup-outcheck": "session_setup_circuit",
    "setup-output": "session_setup_circuit",
    "prep": "preprocessing_local", "input": "input", "online": "online_evaluation",
    "outcheck": "output_mac_checks", "output": "output_opening", "local": "local",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="run one Pi_Gen configuration and write a JSON record")
    ap.add_argument("--family", choices=("dpf", "dcf", "gate"), default="dpf")
    ap.add_argument("--n", type=int, default=8, help="domain bits")
    ap.add_argument("--k", type=int, default=64, help="payload bits")
    ap.add_argument("--s", type=int, default=64, help="MAC bits")
    ap.add_argument("--beta", choices=("public_one", "secret"), default="public_one",
                    help="payload mode of a key")
    ap.add_argument("--index", choices=("xor", "add"), default="xor", help="index sharing of a key")
    ap.add_argument("--gate", choices=cg.GATE_PRESETS, default="comparison",
                    help="gate preset (family gate)")
    ap.add_argument("--batch", type=int, default=1, help="instances generated by one circuit execution")
    ap.add_argument("--prep", choices=("real", "ideal"), default="real")
    ap.add_argument("--sigma", type=int, default=40, help="statistical parameter of the bucketing")
    ap.add_argument("--setup-adder", choices=cg.ADDERS, default="ripple",
                    help="adder of the session-setup circuit (ripple: s rounds; sklansky: about log s)")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="emulated one-way latency per flight")
    ap.add_argument("--mbps", type=float, default=None, help="emulated bandwidth per direction")
    ap.add_argument("--tcp", action="store_true", help="in-process mode over loopback TCP sockets")
    ap.add_argument("--role", type=int, choices=(0, 1), default=None,
                    help="run ONE party of a two-process run (0 listens, 1 connects)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47311)
    ap.add_argument("--spawn", action="store_true", help="start both --role processes and merge their records")
    ap.add_argument("--sid", default=None, help="session identifier (text); default: derived from the configuration")
    ap.add_argument("--dealer-seed", default="couplet-bench-ideal-dealer",
                    help="shared seed of the ideal dealer in the two-process modes")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds one receive may block")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="", help="suffix of the file name")
    ap.add_argument("--no-verify", action="store_true", help="skip the correctness check of the outputs")
    ap.add_argument("--dump", default=None, help=argparse.SUPPRESS)  # used by --spawn (verification data)
    args = ap.parse_args(argv)
    if args.batch < 1:
        ap.error("--batch must be positive")
    if args.spawn and args.role is not None:
        ap.error("--spawn and --role exclude each other")
    return args


def descriptor_of(args) -> dict:
    if args.family == "gate":
        return cg.gate_preset(args.gate, args.n, args.k, args.s)
    make = cg.dpf_descriptor if args.family == "dpf" else cg.dcf_descriptor
    return make(args.n, args.k, args.s, args.index, args.beta)


def session_id(args, name: str) -> bytes:
    if args.sid is not None:
        return args.sid.encode("utf-8")
    return f"bench-gen|{name}|b{args.batch}|{args.prep}|{args.sigma}".encode("utf-8")


def mode_name(args) -> str:
    if args.spawn:
        return "two-process"
    if args.role is not None:
        return f"two-process-p{args.role}"
    return "in-process-tcp" if args.tcp else "in-process"


def record_path(args, name: str) -> str:
    delay = f"{args.delay_ms:g}".replace(".", "p")
    parts = [f"gen_{name}", f"b{args.batch}", args.prep, f"s{args.sigma}", f"d{delay}ms"]
    if args.setup_adder != "ripple":
        parts.append(f"setup-{args.setup_adder}")
    if args.mbps is not None:
        parts.append(f"{args.mbps:g}mbps".replace(".", "p"))
    parts.append({"in-process": "inproc", "in-process-tcp": "inproc-tcp", "two-process": "2proc"}.get(
        mode_name(args), f"proc-p{args.role}"))
    if args.tag:
        parts.append(args.tag)
    return os.path.join(args.out_dir, "_".join(parts) + ".json")


# --------------------------------------------------------------------------- #
# Ideal dealer for two processes (shared seed; ideal mode only)
# --------------------------------------------------------------------------- #

class _ShakeRng:
    """Deterministic byte source: call i returns SHAKE-256(seed || i) (ideal mode only)."""

    def __init__(self, seed: bytes) -> None:
        self._seed = bytes(seed)
        self._calls = 0

    def randbytes(self, n: int) -> bytes:
        out = hashlib.shake_256(self._seed + self._calls.to_bytes(8, "little")).digest(int(n))
        self._calls += 1
        return out


class SharedSeedDealer(IdealDealer):
    """Ideal dealer replicated in both processes from a shared seed.

    Each process serves only its own party; the two replicas produce the same
    correlations because the parties issue the same sequence of requests.  This
    is the two-process form of the ideal mode of DESIGN.md 7.7: NOT a protocol.
    """

    def __init__(self, seed: bytes) -> None:
        super().__init__(_ShakeRng(seed))

    def request(self, role: int, kind: str, params: tuple):
        return getattr(self, "_make_" + kind)(*params)[role]


# --------------------------------------------------------------------------- #
# One party
# --------------------------------------------------------------------------- #

def party_job(role: int, chan, args, desc: dict, circuit, sid: bytes, dealer) -> dict:
    """Run one party: session setup, one generation.  The result holds secrets (verification only)."""
    clock = time.perf_counter
    chan.mark("local")
    t0 = clock()
    ses = Session(role, chan, sid, k=args.k, s=args.s, prep=args.prep, sigma=args.sigma, dealer=dealer,
                  setup_adder=args.setup_adder, allow_refill=False)      # strict: never a silent refill
    budget = pg.and_budget([(desc, args.batch)])       # AND gates of the circuit x instances
    if budget != circuit.stats()["n_and"] * args.batch:
        raise RuntimeError("the AND budget does not match the statistics of the generation circuit")
    ses.setup(reserve_triples=budget)                  # all triples of the session in ONE request
    t1 = clock()
    shares = {"alpha": None, "beta": None}
    if args.family == "gate":
        outputs = pg.gen_gate(ses, desc, args.batch)
    else:
        shares["alpha"] = [secrets.randbelow(1 << args.n) for _ in range(args.batch)]
        if args.beta == "secret":
            shares["beta"] = [secrets.randbelow(1 << args.k) for _ in range(args.batch)]
        gen = pg.gen_dpf if args.family == "dpf" else pg.gen_dcf
        outputs = gen(ses, args.n, args.index, args.beta, shares["alpha"], shares["beta"], args.batch)
    t2 = clock()
    sizes = [len(x.to_bytes()) for x in outputs]
    t3 = clock()
    if ses.triple_requests != 1 or ses.engine.triples_available() != 0:
        raise RuntimeError(f"the session is not ONE fully used bucketing population: "
                           f"{ses.triple_requests} triple requests, "
                           f"{ses.engine.triples_available()} triples left")
    return {
        "role": role, "outputs": outputs, "shares": shares, "delta_share": ses.delta_share,
        "info": ses.info(), "gen_stats": dict(ses.last_gen["stats"]), "setup_stats": dict(ses.setup_stats),
        "digest": ses.last_gen["digest"], "output_bytes": sizes,
        "times": {"setup_s": t1 - t0, "generation_s": t2 - t1, "serialise_s": t3 - t2, "total_s": t3 - t0},
    }


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def phase_sums_ok(stats: dict) -> bool:
    """True iff every counter of the channel equals the sum over its phases."""
    return all(sum(ph[name] for ph in stats["phases"].values()) == stats[name] for name in COUNTERS)


def _key_pair_ok(family: str, k0, k1, alpha: int, beta: int, d0: int, d1: int, full: bool) -> bool:
    """Bit-for-bit comparison with the plaintext generator and correctness on the domain (or a sample)."""
    n, k, s = k0.n, k0.k, k0.s
    delta = d0 + d1
    mod = 1 << (k + s)
    gen = fss.dpf_gen if family == "dpf" else fss.dcf_gen
    e0, e1 = gen(alpha, beta, delta, n, k, s, roots=(k0.seed, k1.seed))
    if e0.to_bytes() != k0.to_bytes() or e1.to_bytes() != k1.to_bytes():
        return False
    indicator = (lambda x: int(x == alpha)) if family == "dpf" else (lambda x: int(x < alpha))
    if full:
        ev = fss.dpf_eval_all if family == "dpf" else fss.dcf_eval_all
        y0, t0 = ev(0, k0)
        y1, t1 = ev(1, k1)
        for x in range(1 << n):
            want = beta * indicator(x)
            if (int(y0[x]) + int(y1[x])) % mod != want or (int(t0[x]) + int(t1[x])) % mod != delta * want % mod:
                return False
        return True
    ev = fss.dpf_eval if family == "dpf" else fss.dcf_eval
    points = {alpha, 0, (1 << n) - 1, (alpha + 1) % (1 << n)} | {secrets.randbelow(1 << n) for _ in range(28)}
    for x in points:
        (y0, t0), (y1, t1) = ev(0, k0, x), ev(1, k1, x)
        want = beta * indicator(x)
        if (y0 + y1) % mod != want or (t0 + t1) % mod != delta * want % mod:
            return False
    return True


def verify_outputs(args, desc: dict, res0: dict, res1: dict) -> dict:
    """Check the two parties' outputs against each other (needs both parties' secrets)."""
    d0, d1 = res0["delta_share"], res1["delta_share"]
    delta = d0 + d1
    n, k, s = args.n, args.k, args.s
    full = n <= 16
    ok = res0["digest"] == res1["digest"]
    count = 0
    for j in range(args.batch):
        o0, o1 = res0["outputs"][j], res1["outputs"][j]
        if args.family != "gate":
            a0, a1 = res0["shares"]["alpha"][j], res1["shares"]["alpha"][j]
            alpha = a0 ^ a1 if args.index == "xor" else (a0 + a1) % (1 << n)
            beta = 1
            if args.beta == "secret":
                beta = (res0["shares"]["beta"][j] + res1["shares"]["beta"][j]) % (1 << k)
            ok &= _key_pair_ok(args.family, o0, o1, alpha, beta, d0, d1, full)
            count += 1
            continue
        mod = 1 << (k + s)
        m = o0.rp_bits
        rp = (o0.rp_share + o1.rp_share) % (1 << m)
        r = rp % (1 << n)
        if desc["auth_input"]:
            ok &= (o0.rp_tag_share + o1.rp_tag_share) % (1 << m) == delta * rp % (1 << m)
        if desc["dpf_one"]:
            ok &= _key_pair_ok("dpf", o0.dpf_key, o1.dpf_key, r, 1, d0, d1, full)
            count += 1
        for i, (c, payload) in enumerate(desc["thresholds"]):
            w_c = int(r + c >= (1 << n))
            ok &= _key_pair_ok("dcf", o0.dcf_keys[i], o1.dcf_keys[i], (r + c) % (1 << n),
                               1 if payload == "one" else r, d0, d1, full)
            count += 1
            ok &= (o0.w_shares[i][0] + o1.w_shares[i][0]) % mod == w_c
            ok &= (o0.w_shares[i][1] + o1.w_shares[i][1]) % mod == delta * w_c % mod
            if payload == "r":
                ok &= (o0.rw_shares[i][0] + o1.rw_shares[i][0]) % mod == r * w_c
                ok &= (o0.rw_shares[i][1] + o1.rw_shares[i][1]) % mod == delta * r * w_c % mod
        for (v0, t0), (v1, t1) in zip(o0.outmask_shares, o1.outmask_shares):
            value = (v0 + v1) % mod
            ok &= value % (1 << k) == 0 and (t0 + t1) % mod == delta * value % mod
    return {"outputs_correct": bool(ok), "key_pairs_checked": count,
            "domain_check": "full domain, both planes" if full else "32 sampled points, both planes",
            "delta_at_least_2_pow_s": bool(delta >> s)}


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #

def environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count()}


def circuit_info(circuit, build_s: float, plan_s: float) -> dict:
    st = circuit.stats()
    plan = abb.get_plan(circuit)
    keys = ("n_and", "n_xor", "n_not", "n_gates", "and_depth", "n_wires", "n_input_bits_p0",
            "n_input_bits_p1", "n_bound_input_bits", "n_public_output_bits",
            "n_private_output_bits_p0", "n_private_output_bits_p1")
    info = {"name": circuit.meta.get("name"), "digest": circuit.digest(),
            "descriptor": circuit.meta.get("descriptor")}
    info.update({key: st[key] for key in keys})
    info.update({"linear_steps": plan.n_linear_steps, "build_s": build_s, "plan_compile_s": plan_s})
    return info


def party_record(res: dict, stats: dict) -> dict:
    """The publishable part of one party's result (no secrets) with its channel statistics."""
    gen = res["gen_stats"]
    return {
        "role": res["role"],
        "channel": {key: stats[key] for key in COUNTERS + ("time_s", "emulated_wait_s", "transport",
                                                           "frame_overhead", "delay_ms", "mbps")},
        "phases": stats["phases"],
        "phase_sums_ok": phase_sums_ok(stats),
        "times": res["times"],
        "session": res["info"],
        "generation": {key: gen[key] for key in (
            "batch", "n_gates", "n_and", "and_depth", "triples", "input_masks", "opened_bits",
            "checked_bits", "rounds", "bytes_sent", "prep_s", "input_s", "online_s",
            "online_exchange_wait_s", "check_s", "output_s", "total_s", "gates_per_s",
            "gates_per_s_local")},
        "session_setup_evaluation": {key: res["setup_stats"][key] for key in (
            "n_and", "and_depth", "rounds", "bytes_sent", "total_s")},
        "output_bytes": res["output_bytes"],
    }


def group_phases(parties: list) -> dict:
    """Phase groups: bytes sent by all recorded parties, rounds and seconds of the first one."""
    groups: dict = {}
    for idx, party in enumerate(parties):
        for name, ph in party["phases"].items():
            g = groups.setdefault(GROUPS.get(name, name), {"bytes": 0, "rounds": 0, "seconds": 0.0, "phases": []})
            g["bytes"] += ph["bytes_sent"]
            if idx == 0:
                g["rounds"] += ph["rounds"]
                g["seconds"] += ph["time_s"]
                g["phases"].append(name)
    return groups


def build_record(args, desc: dict, circuit, build_s: float, plan_s: float, results: list,
                 stats: list, wall_s: float, verification) -> dict:
    name = circuit.meta["name"]
    parties = [party_record(res, st) for res, st in zip(results, stats)]
    n_and = circuit.stats()["n_and"]
    setup_st = setup_circuit(args).stats()
    triples = setup_st["n_and"] + n_and * args.batch
    bucket = bucket_size(triples, args.sigma)
    info = results[0]["info"]
    both = len(parties) == 2
    sent = sum(p["channel"]["bytes_sent"] for p in parties)
    total_bytes = sent if both else parties[0]["channel"]["bytes_sent"] + parties[0]["channel"]["bytes_recv"]
    groups = group_phases(parties)
    if not both:        # one party recorded: its received bytes are the peer's sent bytes
        for g in groups.values():
            g["bytes"] = None
    checks = {"phase_sums_ok": all(p["phase_sums_ok"] for p in parties),
              "single_bucketing_population": all(r["info"]["triple_requests"] == 1 for r in results),
              "strict_mode": not any(r["info"]["allow_refill"] for r in results),
              "all_triples_used": info["triples_unused"] == 0 and info["triples_used"] == triples}
    if both:
        checks["sent_equals_received"] = all(
            parties[b]["channel"]["bytes_sent"] == parties[1 - b]["channel"]["bytes_recv"] for b in (0, 1))
        checks["rounds_agree"] = parties[0]["channel"]["rounds"] == parties[1]["channel"]["rounds"]
        checks["group_bytes_sum_to_total"] = sum(g["bytes"] for g in groups.values()) == total_bytes
    if verification is not None:
        checks["outputs_correct"] = verification["outputs_correct"]
    gen = parties[0]["generation"]
    out_bytes = parties[0]["output_bytes"]
    record = {
        "schema": SCHEMA,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "implementation": "python reference (couplet)",
        "environment": environment(),
        "params": {
            "family": args.family, "n": args.n, "k": args.k, "s": args.s,
            "index_mode": args.index if args.family != "gate" else None,
            "beta_mode": args.beta if args.family != "gate" else None,
            "gate": args.gate if args.family == "gate" else None,
            "batch": args.batch, "prep": args.prep, "sigma": args.sigma,
            "setup_adder": args.setup_adder, "delay_ms": args.delay_ms, "mbps": args.mbps,
            "mode": mode_name(args),
            "transport": parties[0]["channel"]["transport"], "tag": args.tag,
        },
        "ideal_preprocessing": args.prep == "ideal",
        "circuit": circuit_info(circuit, build_s, plan_s),
        "session_setup_circuit": {"name": setup_circuit(args).meta.get("name"), "n_and": setup_st["n_and"],
                                  "and_depth": setup_st["and_depth"], "n_gates": setup_st["n_gates"]},
        "preprocessing": {
            "triples": triples, "triples_generation_circuit": n_and * args.batch,
            "triples_session_setup": setup_st["n_and"], "triple_requests": info["triple_requests"],
            "triples_requested": info["triples_requested"], "allow_refill": info["allow_refill"],
            "bucketing_bound_log2": info["bucketing_bound_log2"],
            "bucket_size": bucket if args.prep == "real" else None,
            "bucket_size_measured": info["last_bucket_size"],
            "leaky_triples": info["prep_counters"].get("leaky_triples"),
            "input_masks": info["prep_counters"].get("input_masks"),
            "model_bytes_per_and": 96 * bucket if args.prep == "real" else None,
        },
        "parties": parties,
        "phase_groups": groups,
        "phase_groups_note": ("bytes: sent by all recorded parties (null if only one party is recorded); "
                              "rounds and seconds: the first recorded party"),
        "totals": {
            "bytes": total_bytes,
            "bytes_note": ("sum of the bytes sent by both parties" if both
                           else "bytes sent plus bytes received by the recorded party"),
            "rounds": parties[0]["channel"]["rounds"],
            "wall_s": wall_s,
            "setup_s": max(p["times"]["setup_s"] for p in parties),
            "generation_s": max(p["times"]["generation_s"] for p in parties),
            "online_s": gen["online_s"],
            "gates_per_s": gen["gates_per_s"],
            "gates_per_s_local": gen["gates_per_s_local"],
            "bytes_per_and_gate": total_bytes / (n_and * args.batch),
            "bytes_per_instance": total_bytes / args.batch,
            "seconds_per_instance": wall_s / args.batch,
        },
        "outputs": {
            "kind": "gate material" if args.family == "gate" else f"{args.family} key",
            "instances": args.batch,
            "bytes_per_instance_and_party": out_bytes[0],
            "bytes_all_instances_per_party": sum(out_bytes),
        },
        "verification": verification,
        "checks": checks,
    }
    if args.prep == "ideal":
        record["note"] = ("IDEAL preprocessing (trusted dealer): online phase only; "
                          "not an all-in measurement of the protocol")
    return record


def write_record(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")


def summary(record: dict, path: str) -> str:
    t, c, p = record["totals"], record["circuit"], record["params"]
    return (f"{c['name']} batch={p['batch']} prep={p['prep']} mode={p['mode']}: "
            f"{c['n_and']:,} AND, depth {c['and_depth']}; {t['bytes']:,} bytes, {t['rounds']} rounds, "
            f"wall {t['wall_s']:.2f} s (setup {t['setup_s']:.2f} s, generation {t['generation_s']:.2f} s, "
            f"online {t['online_s'] * 1e3:.0f} ms); checks {record['checks']}\n  -> {path}")


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #

def setup_circuit(args):
    return cg.build_session_setup(args.s, adder=args.setup_adder)


def prepare(args):
    """Build the circuits and compile their plans before the timed run (public, deterministic work)."""
    desc = descriptor_of(args)
    t = time.perf_counter()
    circuit = pg.generation_circuit(desc)
    build_s = time.perf_counter() - t
    t = time.perf_counter()
    abb.get_plan(circuit)
    abb.get_plan(setup_circuit(args))
    plan_s = time.perf_counter() - t
    return desc, circuit, build_s, plan_s


def run_in_process(args) -> dict:
    desc, circuit, build_s, plan_s = prepare(args)
    sid = session_id(args, circuit.meta["name"])
    dealer = IdealDealer() if args.prep == "ideal" else None

    def party(role):
        return lambda chan: party_job(role, chan, args, desc, circuit, sid, dealer)

    t = time.perf_counter()
    res0, res1, st0, st1 = run_two_party(party(0), party(1), delay_ms=args.delay_ms, mbps=args.mbps,
                                         tcp=args.tcp, timeout=args.timeout)
    wall = time.perf_counter() - t
    verification = None if args.no_verify else verify_outputs(args, desc, res0, res1)
    return build_record(args, desc, circuit, build_s, plan_s, [res0, res1], [st0, st1], wall, verification)


def run_role(args) -> dict:
    desc, circuit, build_s, plan_s = prepare(args)
    sid = session_id(args, circuit.meta["name"])
    dealer = None
    if args.prep == "ideal":
        dealer = SharedSeedDealer(hashlib.sha256(args.dealer_seed.encode("utf-8") + sid).digest())
    kwargs = dict(delay_ms=args.delay_ms, mbps=args.mbps, timeout=args.timeout)
    if args.role == 0:
        chan = TcpChannel.listen(args.port, args.host, party=0, accept_timeout=args.timeout, **kwargs)
    else:
        chan = TcpChannel.connect(args.host, args.port, party=1, connect_timeout=args.timeout, **kwargs)
    t = time.perf_counter()
    try:
        res = party_job(args.role, chan, args, desc, circuit, sid, dealer)
    finally:
        chan.close()
    wall = time.perf_counter() - t
    stats = chan.stats()
    if args.dump:      # verification data for --spawn: contains this party's secrets
        with open(args.dump, "wb") as fh:
            pickle.dump({"res": res, "stats": stats, "wall": wall}, fh)
    return build_record(args, desc, circuit, build_s, plan_s, [res], [stats], wall, None)


def run_spawn(args) -> dict:
    """Start both parties as separate processes, verify their outputs and merge the records."""
    desc, circuit, build_s, plan_s = prepare(args)
    passthrough = ["--family", args.family, "--n", str(args.n), "--k", str(args.k), "--s", str(args.s),
                   "--beta", args.beta, "--index", args.index, "--gate", args.gate,
                   "--batch", str(args.batch), "--prep", args.prep, "--sigma", str(args.sigma),
                   "--setup-adder", args.setup_adder,
                   "--delay-ms", str(args.delay_ms), "--host", args.host, "--port", str(args.port),
                   "--dealer-seed", args.dealer_seed, "--timeout", str(args.timeout), "--no-verify"]
    if args.mbps is not None:
        passthrough += ["--mbps", str(args.mbps)]
    if args.sid is not None:
        passthrough += ["--sid", args.sid]
    if args.tag:
        passthrough += ["--tag", args.tag]
    with tempfile.TemporaryDirectory() as tmp:
        dumps = [os.path.join(tmp, f"p{b}.pickle") for b in (0, 1)]
        t = time.perf_counter()
        # the per-party records of the children stay in the temporary directory: the merged
        # record written by this process contains both of them
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), *passthrough,
                                   "--out-dir", tmp, "--role", str(b), "--dump", dumps[b]],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for b in (0, 1)]
        logs = [p.communicate()[0] for p in procs]
        elapsed = time.perf_counter() - t
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError("a party process failed:\n" + "\n".join(logs))
        data = []
        for path in dumps:
            with open(path, "rb") as fh:
                data.append(pickle.load(fh))
    results = [d["res"] for d in data]
    stats = [d["stats"] for d in data]
    wall = max(d["wall"] for d in data)
    verification = None if args.no_verify else verify_outputs(args, desc, results[0], results[1])
    record = build_record(args, desc, circuit, build_s, plan_s, results, stats, wall, verification)
    record["totals"]["processes_elapsed_s"] = elapsed
    record["totals"]["wall_note"] = ("wall_s is the protocol time of the slower party process; "
                                     "processes_elapsed_s includes interpreter start and circuit compilation")
    return record


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.spawn:
        record = run_spawn(args)
    elif args.role is not None:
        record = run_role(args)
    else:
        record = run_in_process(args)
    path = record_path(args, record["circuit"]["name"])
    write_record(path, record)
    print(summary(record, path))
    return 0 if all(record["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
