"""End-to-end benchmark of the C++ implementation: two processes over loopback TCP.

For every configuration the script starts ``build/couplet_party`` twice on
127.0.0.1 (party 0 listens, party 1 connects), waits for both, cross-checks
the two JSON records and stores them in ``artifact/experiments/raw/e2e_cpp/``:

    <name>_p0.json, <name>_p1.json   the record of each party (the run with the
                                     median time if --repeat > 1)
    summary.json                     one row per configuration: bytes in both
                                     directions, rounds, seconds (median, min,
                                     max over the repetitions), bucket size,
                                     per-phase bytes and seconds
    summary.txt                      the same table as text

``<name>`` is ``e2e_<circuit>_b<batch>_<prep>_s<sigma>_d<delay>ms``.

Every run uses ``--verify``: after the measured protocol the two processes
exchange their keys in the clear (a TEST-ONLY step that is not counted) and
check the outputs on the whole domain.  A configuration is accepted only if
both processes exit with status 0, both verified the outputs, the per-phase
counters sum to the totals at both parties, the bytes sent by one party equal
the bytes received by the other, and both count the same rounds.

Bucketing populations (DESIGN.md 7.5; CHANGELOG F1 W2, F2 item 2).
``couplet_party`` reserves all AND triples of the session in ONE preprocessing
request and runs the session in strict mode, so a second population cannot
appear silently.  The runner checks this on both records (``triple_requests``
== 1, the request covers exactly the setup circuit and the generation
circuit, every triple was used, ``strict_mode``) and copies
``triple_requests``, ``triples_requested``, ``bucketing_bound_log2``
(``log2(requests) - sigma``; ``null`` for ideal preprocessing) and
``strict_mode`` into every row of the summary.  ``--build-dir DIR`` takes
``couplet_party`` from ``DIR`` instead of ``build``.

Default configurations (real preprocessing, sigma = 40):
    dpf  n = 8, 12, 16, 20   public payload, batch 1
    dpf  n = 16              batch 16
    dcf  n = 8, 16           batch 1
    gate-table n = 12, 16    (shared input)
    gate-cmp   n = 16        (shared input)
    dpf  n = 16              one-way delay 10 ms and 30 ms

Usage (from any directory; build first with build_msvc.bat or build.sh):

    python artifact/cpp/run_e2e.py                    # all default configurations
    python artifact/cpp/run_e2e.py --repeat 3         # median of three runs each
    python artifact/cpp/run_e2e.py --only dpf_n16     # configurations whose name contains the text
    python artifact/cpp/run_e2e.py --list
    python artifact/cpp/run_e2e.py --config task=dcf,n=12,batch=4,delay_ms=5

The times are wall-clock times with BOTH processes on one machine; the delay
is emulated at application level inside the channel (see couplet_party.cpp).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, "..", "experiments", "raw", "e2e_cpp"))
CIRCUITS = os.path.normpath(os.path.join(_HERE, "..", "circuits"))      # the exported circuits of the artifact
PHASES = ("baseot", "cotext", "abit", "laand", "bucket", "check", "input", "online", "output")


def default_configs() -> list:
    base = {"k": 64, "beta": "public", "auth": False, "batch": 1, "prep": "real", "sigma": 40, "delay_ms": 0.0}
    cfgs = []
    for n in (8, 12, 16, 20):
        cfgs.append(dict(base, task="dpf", n=n))
    cfgs.append(dict(base, task="dpf", n=16, batch=16))
    for n in (8, 16):
        cfgs.append(dict(base, task="dcf", n=n))
    for n in (12, 16):
        cfgs.append(dict(base, task="gate-table", n=n))
    cfgs.append(dict(base, task="gate-cmp", n=16))
    for delay in (10.0, 30.0):
        cfgs.append(dict(base, task="dpf", n=16, delay_ms=delay))
    return cfgs


def parse_config(text: str) -> dict:
    cfg = {"k": 64, "beta": "public", "auth": False, "batch": 1, "prep": "real", "sigma": 40, "delay_ms": 0.0}
    for item in text.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key in ("n", "k", "batch", "sigma"):
            cfg[key] = int(value)
        elif key == "delay_ms":
            cfg[key] = float(value)
        elif key == "auth":
            cfg[key] = value.strip().lower() in ("1", "true", "yes")
        elif key in ("task", "beta", "prep"):
            cfg[key] = value.strip()
        else:
            raise ValueError(f"unknown configuration key {key!r}")
    if "task" not in cfg or "n" not in cfg:
        raise ValueError("a configuration needs at least task= and n=")
    return cfg


def circuit_name(cfg: dict) -> str:
    n, k = cfg["n"], cfg["k"]
    if cfg["task"] in ("dpf", "dcf"):
        return f"{cfg['task']}_n{n}_k{k}_s64_xor_{'secret' if cfg['beta'] == 'secret' else 'one'}"
    mode = "auth" if cfg["auth"] else "shared"
    if cfg["task"] == "gate-table":
        return f"gate_n{n}_k{k}_s64_{mode}_dpf"
    return f"gate_n{n}_k{k}_s64_{mode}_t{(1 << n) // 3}o_t0o"


def config_name(cfg: dict) -> str:
    delay = cfg["delay_ms"]
    delay_text = str(int(delay)) if float(delay).is_integer() else str(delay).replace(".", "p")
    return f"e2e_{circuit_name(cfg)}_b{cfg['batch']}_{cfg['prep']}_s{cfg['sigma']}_d{delay_text}ms"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_binary(build_dir=None) -> str:
    build_dir = build_dir or os.path.join(_HERE, "build")
    for name in ("couplet_party.exe", "couplet_party"):
        path = os.path.join(build_dir, name)
        if os.path.isfile(path):
            return path
    raise SystemExit(f"couplet_party not found in {build_dir}: run build_msvc.bat (or build.sh) first")


def run_once(binary: str, cfg: dict, out_dir: str, tag: str, threads: int, timeout: float) -> dict:
    """One two-process run; returns {'ok', 'records': [p0, p1], 'files', 'error'}."""
    port = free_port()
    files = [os.path.join(out_dir, f"{tag}_p{role}.json") for role in (0, 1)]
    procs = []
    for role in (0, 1):
        cmd = [binary, "--role", str(role), "--host", "127.0.0.1", "--port", str(port), "--task", cfg["task"],
               "--n", str(cfg["n"]), "--k", str(cfg["k"]), "--batch", str(cfg["batch"]), "--prep", cfg["prep"],
               "--sigma", str(cfg["sigma"]), "--delay-ms", repr(float(cfg["delay_ms"])), "--threads", str(threads),
               "--verify", "--quiet", "--out", files[role], "--circuits", CIRCUITS]
        if cfg["task"] in ("dpf", "dcf"):
            cmd += ["--beta", cfg["beta"]]
        elif cfg["auth"]:
            cmd += ["--auth"]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        if role == 0:
            time.sleep(0.2)                       # let party 0 start listening (party 1 retries anyway)
    outs = []
    for p in procs:
        try:
            outs.append(p.communicate(timeout=timeout))
        except subprocess.TimeoutExpired:
            for other in procs:
                other.kill()
            return {"ok": False, "error": "timeout", "files": files}
    codes = [p.returncode for p in procs]
    if any(codes):
        return {"ok": False, "files": files,
                "error": f"exit codes {codes}: " + " | ".join((o[1] or o[0]).strip()[-300:] for o in outs)}
    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    r0, r1 = records
    problems = []
    for r in records:
        if not r["checks"]["phase_sums_ok"]:
            problems.append("phase sums")
        if not (r["verification"]["performed"] and r["verification"]["outputs_correct"]):
            problems.append("verification")
        pre = r["preprocessing"]
        if not r["checks"]["single_bucketing_population"] or pre.get("triple_requests") != 1:
            problems.append("bucketing population (triple_requests != 1)")
        if pre.get("triples_requested") != pre["triples"] or not r["checks"]["all_triples_used"]:
            problems.append("the one triple request is not exactly the triples of the session")
        if not r["checks"].get("strict_mode") or pre.get("allow_refill") is not False:
            problems.append("the session did not run in strict mode")
        expected_bound = None if cfg["prep"] == "ideal" else -float(cfg["sigma"])
        if pre.get("bucketing_bound_log2") != expected_bound:
            problems.append("bucketing bound is not that of one population")
        if sum(r["phases"][p]["bytes_sent"] for p in PHASES) != r["totals"]["bytes_sent"]:
            problems.append("phase bytes do not sum to the total")
        if sum(r["phases"][p]["rounds"] for p in PHASES) != r["totals"]["rounds"]:
            problems.append("phase rounds do not sum to the total")
    if r0["totals"]["bytes_sent"] != r1["totals"]["bytes_recv"] or r0["totals"]["bytes_recv"] != r1["totals"]["bytes_sent"]:
        problems.append("sent != received")
    if r0["totals"]["rounds"] != r1["totals"]["rounds"]:
        problems.append("round counts differ")
    if r0["circuit"]["digest"] != r1["circuit"]["digest"]:
        problems.append("circuit digests differ")
    if problems:
        return {"ok": False, "error": "; ".join(sorted(set(problems))), "files": files, "records": records}
    return {"ok": True, "records": records, "files": files}


def summarise(cfg: dict, name: str, runs: list) -> dict:
    """Row of the summary: totals of the median run and the spread over all runs."""
    secs = [max(r["records"][0]["totals"]["seconds"], r["records"][1]["totals"]["seconds"]) for r in runs]
    order = sorted(range(len(runs)), key=lambda i: secs[i])
    mid = order[len(order) // 2]
    r0, r1 = runs[mid]["records"]
    phases = {}
    for p in PHASES:
        phases[p] = {
            "bytes": r0["phases"][p]["bytes_sent"] + r1["phases"][p]["bytes_sent"],
            "rounds": r0["phases"][p]["rounds"],
            "seconds_p0": r0["phases"][p]["seconds"],
            "seconds_p1": r1["phases"][p]["seconds"],
        }
    return {
        "name": name, "config": cfg, "circuit": r0["circuit"]["name"], "digest": r0["circuit"]["digest"],
        "n_and": r0["circuit"]["stats"]["n_and"], "and_depth": r0["circuit"]["stats"]["and_depth"],
        "n_gates": r0["circuit"]["stats"]["n_gates"],
        "triples": r0["preprocessing"]["triples"], "bucket_size": r0["preprocessing"]["bucket_size"],
        "triple_requests": max(r0["preprocessing"]["triple_requests"], r1["preprocessing"]["triple_requests"]),
        "triples_requested": r0["preprocessing"]["triples_requested"],
        "bucketing_bound_log2": r0["preprocessing"]["bucketing_bound_log2"],
        "strict_mode": bool(r0["checks"]["strict_mode"] and r1["checks"]["strict_mode"]),
        "bytes": r0["totals"]["bytes_sent"] + r1["totals"]["bytes_sent"],
        "bytes_per_instance": (r0["totals"]["bytes_sent"] + r1["totals"]["bytes_sent"]) / cfg["batch"],
        "bytes_per_and_gate": (r0["totals"]["bytes_sent"] + r1["totals"]["bytes_sent"]) / r0["preprocessing"]["triples"],
        "rounds": r0["totals"]["rounds"],
        "seconds": secs[mid], "seconds_min": min(secs), "seconds_max": max(secs), "seconds_all": secs,
        "seconds_per_instance": secs[mid] / cfg["batch"],
        "setup_s": max(r0["totals"]["setup_s"], r1["totals"]["setup_s"]),
        "generation_s": max(r0["totals"]["generation_s"], r1["totals"]["generation_s"]),
        "online_s": max(r0["totals"]["online_s"], r1["totals"]["online_s"]),
        "emulated_wait_s": max(r0["totals"]["emulated_wait_s"], r1["totals"]["emulated_wait_s"]),
        "output_bytes_per_instance_and_party": r0["outputs"]["bytes_per_instance_and_party"],
        "output_kind": r0["outputs"]["kind"],
        "phases": phases,
        "verified": True, "repetitions": len(runs), "median_run": mid,
        "points_checked": r0["verification"]["points_checked"],
    }


def table_text(rows: list) -> str:
    head = (f"{'configuration':<58} {'AND gates':>10} {'B':>2} {'req':>3} {'bytes (both dir.)':>18} {'rounds':>7} "
            f"{'seconds':>9} {'min':>8} {'max':>8} {'key B':>6}")
    lines = [head, "-" * len(head)]
    for r in rows:
        requests = r.get("triple_requests")               # rows written before 2026-10-11 do not record it
        lines.append(f"{r['name'][4:]:<58} {r['triples']:>10,} {r['bucket_size']:>2} "
                     f"{('?' if requests is None else requests):>3} {r['bytes']:>18,} {r['rounds']:>7} "
                     f"{r['seconds']:>9.3f} {r['seconds_min']:>8.3f} {r['seconds_max']:>8.3f} "
                     f"{r['output_bytes_per_instance_and_party']:>6}")
    lines.append("req = preprocessing requests for AND triples (bucketing populations) of the session")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="two-process end-to-end benchmark of the C++ implementation")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: artifact/experiments/raw/e2e_cpp)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per configuration (the median run is kept)")
    ap.add_argument("--only", default="", help="run only the configurations whose name contains this text")
    ap.add_argument("--config", action="append", default=[], help="extra configuration, e.g. task=dcf,n=12,batch=4")
    ap.add_argument("--no-default", action="store_true", help="run only the --config configurations")
    ap.add_argument("--threads", type=int, default=1, help="worker threads of the preprocessing per party (default 1)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per run")
    ap.add_argument("--list", action="store_true", help="print the configuration names and exit")
    ap.add_argument("--build-dir", default=None, help="directory of couplet_party (default: build)")
    args = ap.parse_args(argv)

    cfgs = ([] if args.no_default else default_configs()) + [parse_config(c) for c in args.config]
    cfgs = [c for c in cfgs if args.only in config_name(c)]
    if args.list:
        for c in cfgs:
            print(config_name(c))
        return 0
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    binary = find_binary(args.build_dir)
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "summary.json")
    rows = {}
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            rows = {r["name"]: r for r in json.load(f)["rows"]}
    except (OSError, ValueError, KeyError):
        pass
    failed = []
    for cfg in cfgs:
        name = config_name(cfg)
        runs = []
        for rep in range(args.repeat):
            res = run_once(binary, cfg, args.out, f"{name}_run{rep}", args.threads, args.timeout)
            if not res["ok"]:
                print(f"FAILED {name} (run {rep}): {res['error']}", flush=True)
                failed.append(name)
                break
            runs.append(res)
            t = max(r["totals"]["seconds"] for r in res["records"])
            print(f"  {name} run {rep}: {res['records'][0]['totals']['bytes']:,} bytes, "
                  f"{res['records'][0]['totals']['rounds']} rounds, {t:.3f} s, outputs verified", flush=True)
        if len(runs) != args.repeat:
            continue
        row = summarise(cfg, name, runs)
        for role in (0, 1):                               # keep the median run under the plain name
            final = os.path.join(args.out, f"{name}_p{role}.json")
            os.replace(runs[row["median_run"]]["files"][role], final)
        for rep, res in enumerate(runs):
            for path in res["files"]:
                if os.path.exists(path):
                    os.remove(path)
        rows[name] = row
        ordered = [rows[k] for k in sorted(rows)]
        with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"generated_by": "python artifact/cpp/run_e2e.py",
                       "note": "two processes over loopback TCP on one machine; bytes = payload bytes in both directions; "
                               "seconds = wall-clock time of the slower party from the first protocol message to the "
                               "certified outputs (median over the repetitions); every run was verified on the whole domain "
                               "by a test-only step that is not counted",
                       "rows": ordered}, f, indent=1)
            f.write("\n")
        with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(table_text(ordered))
    print()
    print(table_text([rows[k] for k in sorted(rows)]))
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
