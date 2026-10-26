"""Benchmark of the inference protocol (couplet.infer): one configuration per invocation,
one JSON record per run.

Runs the whole session end to end: manifest comparison, session setup (base
OT, the persistent wires of Delta), generation of all correlated randomness
(sites of every hidden layer, in private-model mode also the input masks and
the matrix triples), the online pipeline and the release of the logits.
Bytes, rounds and seconds are taken from the channel statistics per
accounting phase; the AND gates, the sites, the triple requests and the
openings from the driver's record.

Modes
  in-process (default)   both parties as two threads of this process (in-memory
                         queues, or loopback TCP sockets with ``--tcp``);
  ``--role 0|1``         ONE party of a two-process run over TCP (role 0 listens);
  ``--spawn``            starts the two ``--role`` processes, checks their outputs
                         against each other and against the mirror, writes one record.

Examples (from artifact/python)
    python bench/bench_infer.py --model mlp16 --mode round --prep ideal
    python bench/bench_infer.py --model cnn12 --mode exact --prep real --delay-ms 20
    python bench/bench_infer.py --model mlp32 --mode round --private-model --prep ideal
    python bench/bench_infer.py --model mlp16 --mode exact --spawn
    python bench/bench_infer.py --model-file mlp16_model.json --mode round --prep real

The model is a preset (``couplet.nn.PRESETS``) with random weights from
``--weights-seed`` (untrained; stated in the record) or a trained model file
(``--model-file``, written by ``couplet.tools.train_synthetic``).  The input
features are uniform in [0, 1] from ``--input-seed`` (both processes derive the
same model and features).  ``--n`` / ``--f`` set the wire format.

Record (``<out-dir>/infer_<model>_n<n>_f<f>_<mode>[_private]_<prep>_s<sigma>_d<delay>ms[_cap<c>][_<mbps>mbps]_<run>[_<tag>].json``,
``<run>`` one of inproc, inproc-tcp, 2proc; ``--out-dir`` defaults to ``experiments/raw``):
parameters, the model summary (sizes, sites, hashes), the manifest summary
(executions, AND gates), per party the channel totals and the per-phase
bytes / rounds / seconds, phase groups (manifest, session_setup,
preprocessing, online, checks), the counters (triple requests, triples left,
executions, openings, checks), the times of the three steps, the logits and
the correctness flags: ``logits_equal`` (both parties), ``equal_to_mirror``
(the cleartext mirror on the recorded thresholds) and, with float weights,
the argmax agreement with the float model.  ``checks`` must all hold for the
exit status 0.  With ``--triple-cap`` the budget is drawn in several requests
and ``preprocessing.triple_requests`` records how many.

Ideal preprocessing (``--prep ideal``) uses a trusted dealer and is NOT a
protocol: such records carry ``"ideal_preprocessing": true``; in the
two-process modes the dealer is replicated from ``--dealer-seed``.  All times
are wall-clock times of the pure-Python reference implementation; in the
in-process mode both parties share one interpreter.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from bench_gen import SharedSeedDealer, environment, phase_sums_ok  # noqa: E402
from couplet import nn as nnm  # noqa: E402
from couplet.infer import InferParty  # noqa: E402
from couplet.net import TcpChannel, run_two_party  # noqa: E402
from couplet.prep import IdealDealer, bucket_size  # noqa: E402

SCHEMA = "couplet.bench_infer/1"
DEFAULT_OUT = os.path.normpath(os.path.join(_PY, "..", "experiments", "raw"))
COUNTERS = ("bytes_sent", "bytes_recv", "msgs_sent", "msgs_recv", "rounds")
GROUPS = {"manifest": "manifest", "baseot": "session_setup", "cotext": "preprocessing", "abit": "preprocessing",
          "laand": "preprocessing", "bucket": "preprocessing", "check": "preprocessing",
          "setup-prep": "session_setup", "setup-input": "session_setup", "setup-online": "session_setup",
          "setup-outcheck": "session_setup", "setup-output": "session_setup",
          "gen-prep": "preprocessing", "gen-input": "preprocessing", "gen-online": "preprocessing",
          "gen-outcheck": "preprocessing", "gen-output": "preprocessing", "prep": "preprocessing",
          "online-local": "online", "tm-open": "online", "input-open": "online", "arith-open": "online",
          "arith-check": "checks", "local": "local"}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="run one inference configuration and write a JSON record")
    ap.add_argument("--model", choices=sorted(nnm.PRESETS), default="mlp16")
    ap.add_argument("--model-file", default=None, help="trained model JSON (overrides --model and --weights-seed)")
    ap.add_argument("--n", type=int, default=16, help="wire bits (plane domain)")
    ap.add_argument("--f", type=int, default=8, help="fraction bits")
    ap.add_argument("--mode", choices=("round", "exact"), default="round")
    ap.add_argument("--private-model", action="store_true")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--s", type=int, default=64)
    ap.add_argument("--sigma", type=int, default=40)
    ap.add_argument("--prep", choices=("real", "ideal"), default="real")
    ap.add_argument("--triple-cap", type=int, default=None, help="generation triples per request (default: all in one)")
    ap.add_argument("--setup-adder", choices=("ripple", "sklansky"), default="ripple")
    ap.add_argument("--weights-seed", type=int, default=1)
    ap.add_argument("--input-seed", type=int, default=2)
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--mbps", type=float, default=None)
    ap.add_argument("--tcp", action="store_true")
    ap.add_argument("--role", type=int, choices=(0, 1), default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47321)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--sid", default=None)
    ap.add_argument("--dealer-seed", default="couplet-bench-infer-ideal-dealer")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.spawn and args.role is not None:
        ap.error("--spawn and --role exclude each other")
    return args


def load_model(args) -> nnm.Model:
    if args.model_file:
        return nnm.Model.load(args.model_file)
    return nnm.random_model(args.model, args.n, args.f, seed=args.weights_seed)


def features(args, model: nnm.Model) -> list:
    rng = np.random.default_rng(args.input_seed)
    return nnm.features_to_codes(rng.random(model.d), model.f)


def model_name(args, model: nnm.Model) -> str:
    if args.model_file:
        return os.path.splitext(os.path.basename(args.model_file))[0]
    return args.model


def session_id(args, model: nnm.Model) -> bytes:
    if args.sid is not None:
        return args.sid.encode("utf-8")
    return (f"bench-infer|{model_name(args, model)}|{model.n}|{model.f}|{args.mode}|{int(args.private_model)}|"
            f"{args.prep}|{args.sigma}").encode("utf-8")


def mode_name(args) -> str:
    if args.spawn:
        return "two-process"
    if args.role is not None:
        return f"two-process-p{args.role}"
    return "in-process-tcp" if args.tcp else "in-process"


def record_path(args, model: nnm.Model) -> str:
    delay = f"{args.delay_ms:g}".replace(".", "p")
    parts = [f"infer_{model_name(args, model)}", f"n{model.n}", f"f{model.f}", args.mode]
    if args.private_model:
        parts.append("private")
    parts += [args.prep, f"s{args.sigma}", f"d{delay}ms"]
    if args.triple_cap:
        parts.append(f"cap{args.triple_cap}")
    if args.mbps is not None:
        parts.append(f"{args.mbps:g}mbps".replace(".", "p"))
    parts.append({"in-process": "inproc", "in-process-tcp": "inproc-tcp", "two-process": "2proc"}.get(
        mode_name(args), f"proc-p{args.role}"))
    if args.tag:
        parts.append(args.tag)
    return os.path.join(args.out_dir, "_".join(parts) + ".json")


# --------------------------------------------------------------------------- #
# One party
# --------------------------------------------------------------------------- #

def party_job(role: int, chan, args, model: nnm.Model, codes: list, sid: bytes, dealer) -> dict:
    clock = time.perf_counter
    chan.mark("local")
    own = model if (role == 0 or not args.private_model) else model.shape_only()
    t0 = clock()
    p = InferParty(role, chan, sid, own, args.mode, prep=args.prep, dealer=dealer, k=args.k, s=args.s,
                   sigma=args.sigma, private_model=args.private_model, triple_cap=args.triple_cap,
                   setup_adder=args.setup_adder)
    p.setup()
    t1 = clock()
    p.preprocess()
    t2 = clock()
    logits = p.infer(codes[model.feature_slice(role)])
    t3 = clock()
    return {"role": role, "logits": logits, "record": dict(p.record), "info": p.info(),
            "manifest": p.manifest.summary(),
            "times": {"setup_s": t1 - t0, "preprocess_s": t2 - t1, "infer_s": t3 - t2, "total_s": t3 - t0}}


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #

def party_record(res: dict, stats: dict) -> dict:
    rec = {key: value for key, value in res["record"].items() if key not in ("opened", "x_hat")}
    return {"role": res["role"],
            "channel": {key: stats[key] for key in COUNTERS + ("time_s", "emulated_wait_s", "transport",
                                                               "frame_overhead", "delay_ms", "mbps")},
            "phases": stats["phases"], "phase_sums_ok": phase_sums_ok(stats), "times": res["times"],
            "record": rec, "session": {k: v for k, v in res["info"].items() if k not in ("manifest", "record")}}


def group_phases(parties: list) -> dict:
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


def verify(model: nnm.Model, codes: list, results: list) -> dict:
    logits = [r["logits"] for r in results]
    rec = results[0]["record"]
    mirror = model.mirror(codes, rec["mode"], rec["thresholds"]).logits
    out = {"logits_equal": all(l == logits[0] for l in logits), "equal_to_mirror": logits[0] == mirror,
           "logits": logits[0], "logits_real": nnm.decode_logits(logits[0], model.f),
           "thresholds_equal": all(r["record"]["thresholds"] == rec["thresholds"] for r in results)}
    if model.float_layers is not None:
        fl = model.forward_float(np.array(codes) / float(1 << model.f))
        out["float_logits"] = [float(v) for v in fl]
        out["argmax_agrees_with_float"] = int(np.argmax(fl)) == int(np.argmax(logits[0]))
    return out


def build_record(args, model: nnm.Model, codes: list, results: list, stats: list, wall: float) -> dict:
    parties = [party_record(res, st) for res, st in zip(results, stats)]
    both = len(parties) == 2
    sent = sum(p["channel"]["bytes_sent"] for p in parties)
    total_bytes = sent if both else parties[0]["channel"]["bytes_sent"] + parties[0]["channel"]["bytes_recv"]
    groups = group_phases(parties)
    if not both:
        for g in groups.values():
            g["bytes"] = None
    rec0 = results[0]["record"]
    info = results[0]["info"]
    man = results[0]["manifest"]
    triples = int(info["triples_used"])
    checks = {"phase_sums_ok": all(p["phase_sums_ok"] for p in parties),
              "all_triples_used": info["triples_unused"] == 0,
              "strict_mode": not info["allow_refill"],
              "single_bucketing_population": info["triple_requests"] == 1 if not args.triple_cap else True,
              "state_active": all(r["info"]["state"] == "ACTIVE" for r in results)}
    verification = verify(model, codes, results) if both else None
    if both:
        checks["sent_equals_received"] = all(
            parties[b]["channel"]["bytes_sent"] == parties[1 - b]["channel"]["bytes_recv"] for b in (0, 1))
        checks["rounds_agree"] = parties[0]["channel"]["rounds"] == parties[1]["channel"]["rounds"]
        checks["logits_equal"] = verification["logits_equal"]
        checks["equal_to_mirror"] = verification["equal_to_mirror"]
    record = {
        "schema": SCHEMA,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "implementation": "python reference (couplet)",
        "environment": environment(),
        "params": {"model": model_name(args, model), "model_file": args.model_file, "kind": model.kind,
                   "act": model.act, "sizes": model.sizes, "n": model.n, "f": model.f, "mode": args.mode,
                   "private_model": args.private_model, "k": args.k, "s": args.s, "sigma": args.sigma,
                   "prep": args.prep, "triple_cap": args.triple_cap, "setup_adder": args.setup_adder,
                   "weights": "trained (model file)" if args.model_file else f"random, seed {args.weights_seed}",
                   "input_seed": args.input_seed, "delay_ms": args.delay_ms, "mbps": args.mbps,
                   "run_mode": mode_name(args), "transport": parties[0]["channel"]["transport"], "tag": args.tag},
        "ideal_preprocessing": args.prep == "ideal",
        "model": {"model_hash": model.model_hash().hex(), "architecture_hash": model.architecture_hash().hex(),
                  "table_hash": model.table_hash().hex(), "sites": model.n_sites, "layers": model.n_layers,
                  "weights": sum(r * c for r, c in model.shapes), "partition": model.partition},
        "manifest": {k: v for k, v in man.items() if k != "entries"},
        "manifest_entries": man["entries"],
        "preprocessing": {"and_gates": rec0["and_gates"], "sites": rec0["sites"], "executions": rec0["executions"],
                          "triples": triples, "triple_requests": info["triple_requests"],
                          "triples_requested": info["triples_requested"], "allow_refill": info["allow_refill"],
                          "bucketing_bound_log2": info["bucketing_bound_log2"],
                          "bucket_size_measured": info["last_bucket_size"],
                          "bucket_size": bucket_size(triples, args.sigma) if args.prep == "real" and not args.triple_cap else None,
                          "leaky_triples": info["prep_counters"].get("leaky_triples")},
        "online": {"openings": rec0["openings"], "openings_before_release": rec0["openings_before_release"],
                   "checks": rec0["checks"], "shared_openings": rec0["shared_openings"],
                   "thresholds": rec0["thresholds"]},
        "parties": parties,
        "phase_groups": groups,
        "phase_groups_note": ("bytes: sent by all recorded parties (null if only one party is recorded); "
                              "rounds and seconds: the first recorded party"),
        "totals": {"bytes": total_bytes,
                   "bytes_note": ("sum of the bytes sent by both parties" if both
                                  else "bytes sent plus bytes received by the recorded party"),
                   "rounds": parties[0]["channel"]["rounds"], "wall_s": wall,
                   "setup_s": max(p["times"]["setup_s"] for p in parties),
                   "preprocess_s": max(p["times"]["preprocess_s"] for p in parties),
                   "infer_s": max(p["times"]["infer_s"] for p in parties),
                   "online_bytes": groups.get("online", {}).get("bytes"),
                   "online_rounds": groups.get("online", {}).get("rounds", 0) + groups.get("checks", {}).get("rounds", 0),
                   "bytes_per_site": total_bytes / max(1, model.n_sites)},
        "verification": verification,
        "checks": checks,
    }
    if args.prep == "ideal":
        record["note"] = "IDEAL preprocessing (trusted dealer): online phase only; not an all-in measurement"
    return record


def write_record(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(record, fh, indent=1)
        fh.write("\n")


def summary(record: dict, path: str) -> str:
    t, p, pre = record["totals"], record["params"], record["preprocessing"]
    return (f"{p['model']} ({p['n']}, {p['f']}) {p['mode']}{' private' if p['private_model'] else ''} prep={p['prep']} "
            f"{p['run_mode']}: {pre['sites']} sites, {pre['and_gates']:,} AND, {pre['triple_requests']} request(s); "
            f"{t['bytes']:,} bytes, {t['rounds']} rounds, wall {t['wall_s']:.1f} s (setup {t['setup_s']:.1f}, "
            f"preprocess {t['preprocess_s']:.1f}, online {t['infer_s'] * 1e3:.0f} ms, online rounds {t['online_rounds']}); "
            f"checks {record['checks']}\n  -> {path}")


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #

def run_in_process(args) -> dict:
    model = load_model(args)
    codes = features(args, model)
    sid = session_id(args, model)
    dealer = IdealDealer() if args.prep == "ideal" else None

    def party(role):
        return lambda chan: party_job(role, chan, args, model, codes, sid, dealer)

    t = time.perf_counter()
    r0, r1, st0, st1 = run_two_party(party(0), party(1), delay_ms=args.delay_ms, mbps=args.mbps, tcp=args.tcp,
                                     timeout=args.timeout)
    return build_record(args, model, codes, [r0, r1], [st0, st1], time.perf_counter() - t)


def run_role(args) -> dict:
    model = load_model(args)
    codes = features(args, model)
    sid = session_id(args, model)
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
        res = party_job(args.role, chan, args, model, codes, sid, dealer)
    finally:
        chan.close()
    wall = time.perf_counter() - t
    stats = chan.stats()
    if args.dump:
        with open(args.dump, "wb") as fh:
            pickle.dump({"res": res, "stats": stats, "wall": wall}, fh)
    return build_record(args, model, codes, [res], [stats], wall)


def run_spawn(args) -> dict:
    model = load_model(args)
    codes = features(args, model)
    passthrough = ["--model", args.model, "--n", str(args.n), "--f", str(args.f), "--mode", args.mode,
                   "--k", str(args.k), "--s", str(args.s), "--sigma", str(args.sigma), "--prep", args.prep,
                   "--setup-adder", args.setup_adder, "--weights-seed", str(args.weights_seed),
                   "--input-seed", str(args.input_seed), "--delay-ms", str(args.delay_ms), "--host", args.host,
                   "--port", str(args.port), "--dealer-seed", args.dealer_seed, "--timeout", str(args.timeout)]
    if args.model_file:
        passthrough += ["--model-file", args.model_file]
    if args.private_model:
        passthrough.append("--private-model")
    if args.triple_cap:
        passthrough += ["--triple-cap", str(args.triple_cap)]
    if args.mbps is not None:
        passthrough += ["--mbps", str(args.mbps)]
    if args.sid is not None:
        passthrough += ["--sid", args.sid]
    if args.tag:
        passthrough += ["--tag", args.tag]
    with tempfile.TemporaryDirectory(prefix="couplet_bench_infer_") as tmp:
        dumps = [os.path.join(tmp, f"p{b}.pickle") for b in (0, 1)]
        t = time.perf_counter()
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), *passthrough, "--out-dir", tmp,
                                   "--role", str(b), "--dump", dumps[b]],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for b in (0, 1)]
        logs = [p.communicate()[0] for p in procs]
        elapsed = time.perf_counter() - t
        if any(p.returncode != 0 for p in procs):
            raise RuntimeError("a party process failed:\n" + "\n".join(logs))
        data = []
        for path in dumps:
            with open(path, "rb") as fh:
                data.append(pickle.load(fh))
    record = build_record(args, model, codes, [d["res"] for d in data], [d["stats"] for d in data],
                          max(d["wall"] for d in data))
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
    path = record_path(args, load_model(args))
    write_record(path, record)
    print(summary(record, path))
    return 0 if all(record["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
