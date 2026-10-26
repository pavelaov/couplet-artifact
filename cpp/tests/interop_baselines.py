"""Cross-language test of the two evaluation baselines (DESIGN.md section 9, Q1).

One party is the C++ program ``build/test_baselines --interop`` (header
``couplet/baselines.hpp``), the other party is this script, which runs the
Python reference (``couplet.baselines.leaky_zhang`` on ``RealPrep``, or
``couplet.baselines.semihonest_ds``) over the same TCP connection.  The two
programs share nothing but the wire: base OT, the OT extension (checked for
the leaky protocol, unchecked for the semi-honest one), the input
authentication, every correction word, the coins, the opening and BatchCheck
must agree message for message, or the batch check aborts or the outputs are
wrong.

After the protocol both sides run the TEST-ONLY verification step of
``baselines::testonly`` (NOT part of either protocol, not counted): per key

    round 1   Delta_b (16 bytes) || alpha_b (8 bytes, little-endian) || beta_b (16 bytes)
    round 2   SHA-256 of the party's output vectors, where party 0 first XORs
              the expected values into entry alpha (see baselines.hpp)

and the digests must be equal.  The script then compares the two programs'
accounting: bytes sent, bytes received and rounds per phase must be the same
numbers on both sides and must equal the formulas of the Python modules.

Usage (from artifact/cpp; build first with build_msvc.bat or build.sh):

    python tests/interop_baselines.py --spawn --all            # the list used for the report
    python tests/interop_baselines.py --spawn --proto zhang --n 4,8 --role 1

or in two shells:

    python tests/interop_baselines.py --proto sh --role 1 --port 47300 --n 8
    build/test_baselines --interop sh 0 47300 8

``--role`` is the role of THIS script; with ``--spawn`` the C++ program takes
the other role.  Exit status 0 iff every run completed, both sides verified
the outputs and the accounting agrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "python"))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import gf128 as gf  # noqa: E402
from couplet.baselines import leaky_zhang as lz  # noqa: E402
from couplet.baselines import semihonest_ds as sh  # noqa: E402
from couplet.net import TcpChannel  # noqa: E402
from couplet.prep import RealPrep  # noqa: E402
from couplet.prep.cotext import CHECK_BYTES, SEED_BYTES, u_message_bytes  # noqa: E402

SID = b"interop-baselines"
PREP_PHASES = ("baseot", "cotext", "abit")


def reveal(chan, delta: int, alpha_share: int, beta_share: int):
    """Round 1 of the TEST-ONLY step: returns (Delta, alpha, beta)."""
    mine = gf.int_to_bytes(delta) + int(alpha_share).to_bytes(8, "little") + gf.int_to_bytes(beta_share)
    theirs = chan.exchange(mine)
    if len(theirs) != 40:
        raise RuntimeError("verification step: malformed message")
    return (delta ^ gf.bytes_to_int(theirs[:16]), alpha_share ^ int.from_bytes(theirs[16:24], "little"),
            beta_share ^ gf.bytes_to_int(theirs[24:40]))


def _patched(vec: np.ndarray, role: int, alpha: int, value: int) -> bytes:
    out = np.array(vec, dtype=np.uint64, copy=True)
    if role == 0:
        out[alpha] ^= gf.int_to_arr(value)
    return gf.vec_to_bytes(out)


def _patched_bits(bits: np.ndarray, role: int, alpha: int) -> bytes:
    out = np.array(bits, dtype=np.uint8, copy=True)
    if role == 0:
        out[alpha] ^= 1
    return out.tobytes()


def leaky_digest(role: int, out, delta: int, alpha: int, beta: int) -> bytes:
    h = hashlib.sha256()
    h.update(_patched_bits(out.u.bits, role, alpha))
    h.update(_patched(out.u.macs, role, alpha, delta))
    h.update(_patched(out.v.values, role, alpha, beta))
    h.update(_patched(out.v.macs, role, alpha, gf.mul(beta, delta)))
    return h.digest()


def semihonest_digest(role: int, out, alpha: int, beta: int) -> bytes:
    h = hashlib.sha256()
    h.update(_patched_bits(out.t, role, alpha))
    h.update(_patched(out.v, role, alpha, beta))
    return h.digest()


def expected_phases(proto: str, ns: list) -> dict:
    """Bytes sent per party and rounds per phase according to the Python modules."""
    if proto == "zhang":
        exp = {"baseot": (12288, 2),
               "cotext": (sum(u_message_bytes(lz.abit_demand(n)) + SEED_BYTES + CHECK_BYTES for n in ns), 3 * len(ns)),
               "abit": (0, 0)}
        for phase in lz.PHASES:
            exp[phase] = (sum(lz.bytes_per_party(n)[phase] for n in ns), sum(lz.rounds(n)[phase] for n in ns))
        return exp
    exp = {"baseot": (12288, 2)}
    for phase in sh.PHASES:
        exp[phase] = (sum(sh.bytes_per_party(n)[phase] for n in ns), sum(sh.rounds(n)[phase] for n in ns))
    return exp


def run_party(proto: str, role: int, host: str, port: int, ns: list) -> dict:
    """Run this script's party; returns its record (phases, totals, verification result)."""
    if role == 0:
        chan = TcpChannel.listen(port, host, party=0, accept_timeout=120)
    else:
        chan = TcpChannel.connect(host, port, party=1, connect_timeout=120)
    try:
        shares = [(secrets.randbits(n), secrets.randbits(128)) for n in ns]
        if proto == "zhang":
            prep = RealPrep(role, chan, SID)
            prep.setup()
            outs = [lz.run_pi_dpf(chan, prep, SID, n, a, b, key_id=k) for k, (n, (a, b)) in enumerate(zip(ns, shares))]
            delta_share = prep.delta
        else:
            session = sh.SemiHonestSession(role, chan, SID)
            session.setup()
            outs = [session.dpf_gen(n, a, b) for n, (a, b) in zip(ns, shares)]
            delta_share = session.delta
        stats = chan.stats()                      # the protocol ends here
        chan.mark("test-only-verification")
        verified = True
        for out, (a, b) in zip(outs, shares):
            delta, alpha, beta = reveal(chan, delta_share, a, b)
            mine = (leaky_digest(role, out, delta, alpha, beta) if proto == "zhang"
                    else semihonest_digest(role, out, alpha, beta))
            verified &= chan.exchange(mine) == mine
        phases = {name: {"bytes_sent": rec["bytes_sent"], "bytes_recv": rec["bytes_recv"], "rounds": rec["rounds"]}
                  for name, rec in stats["phases"].items()}
        return {"peer": "python reference", "proto": proto, "role": role, "n": list(ns),
                "bytes_sent": stats["bytes_sent"], "bytes_recv": stats["bytes_recv"], "rounds": stats["rounds"],
                "phases": phases, "outputs_correct": bool(verified)}
    finally:
        chan.close()


def compare(mine: dict, theirs: dict, proto: str, ns: list) -> list:
    """Problems found when comparing the two programs' records with each other and with the formulas."""
    problems = []
    if not mine["outputs_correct"]:
        problems.append("the Python side did not verify the outputs")
    if not theirs.get("outputs_correct"):
        problems.append("the C++ side did not verify the outputs")
    exp = expected_phases(proto, ns)
    for name, (exp_bytes, exp_rounds) in exp.items():
        for side, rec in (("python", mine), ("c++", theirs)):
            got = rec["phases"].get(name)
            if got is None:
                problems.append(f"{side}: phase {name} is missing")
            elif (got["bytes_sent"], got["bytes_recv"], got["rounds"]) != (exp_bytes, exp_bytes, exp_rounds):
                problems.append(f"{side}: phase {name}: {got} instead of {exp_bytes} bytes each way, {exp_rounds} rounds")
    extra = [name for name in mine["phases"] if name not in exp and name != "(unmarked)"]
    extra += [name for name in theirs["phases"] if name not in exp]
    if extra:
        problems.append(f"unexpected phases: {extra}")
    total = (sum(b for b, _ in exp.values()), sum(r for _, r in exp.values()))
    for side, rec in (("python", mine), ("c++", theirs)):
        if (rec["bytes_sent"], rec["bytes_recv"], rec["rounds"]) != (total[0], total[0], total[1]):
            problems.append(f"{side}: totals {rec['bytes_sent']} / {rec['bytes_recv']} bytes, {rec['rounds']} rounds "
                            f"instead of {total[0]} bytes each way, {total[1]} rounds")
    return problems


def find_binary() -> str:
    for name in ("test_baselines.exe", "test_baselines"):
        path = os.path.join(_HERE, "..", "build", name)
        if os.path.isfile(path):
            return os.path.normpath(path)
    raise SystemExit("build/test_baselines not found: build the C++ code first")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn(proto: str, role: int, host: str, ns: list, binary: str) -> bool:
    """Start the C++ program in the other role, run this role here, compare the records."""
    port = free_port()
    cmd = [binary, "--interop", proto, str(1 - role), str(port), ",".join(str(n) for n in ns)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mine, error = None, ""
    try:
        mine = run_party(proto, role, host, port, ns)
    except Exception as exc:  # noqa: BLE001 - report and fail
        error = f"python side failed: {type(exc).__name__}: {exc}"
    try:
        out, err = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    theirs = None
    for line in (out or "").splitlines():
        if line.startswith("{"):
            theirs = json.loads(line)
    if error or mine is None or theirs is None or proc.returncode != 0:
        print(f"  FAILED: {error} c++ exit {proc.returncode}: {(out or '').strip()} {(err or '').strip()}", flush=True)
        return False
    problems = compare(mine, theirs, proto, ns)
    per_phase = ", ".join(f"{name} {rec['bytes_sent']} B / {rec['rounds']} r" for name, rec in theirs["phases"].items())
    print(f"  python role {role}, c++ role {1 - role}: {mine['bytes_sent']} bytes per party, {mine['rounds']} rounds; "
          f"both verified: {mine['outputs_correct'] and theirs['outputs_correct']}", flush=True)
    print(f"  phases (bytes sent per party / rounds, identical in both programs): {per_phase}", flush=True)
    for problem in problems:
        print("  PROBLEM:", problem, flush=True)
    return not problems


def parse_ns(text: str) -> list:
    ns = [int(part) for part in text.split(",") if part]
    if not ns or any(not 1 <= n <= lz.MAX_N for n in ns):
        raise argparse.ArgumentTypeError(f"n must be a comma-separated list of integers in 1..{lz.MAX_N}")
    return ns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Python peer of build/test_baselines --interop (cross-language test)")
    ap.add_argument("--proto", default="zhang", choices=("zhang", "sh"),
                    help="zhang = leaky protocol on the real preprocessing, sh = semi-honest baseline")
    ap.add_argument("--role", type=int, default=1, choices=(0, 1), help="role of this script (default 1)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47300)
    ap.add_argument("--n", type=parse_ns, default=[8], help="domain sizes of the keys of the session, e.g. 4,8,10")
    ap.add_argument("--spawn", action="store_true", help="start build/test_baselines --interop in the other role")
    ap.add_argument("--all", action="store_true", help="with --spawn: both protocols, both role assignments")
    ap.add_argument("--binary", default=None, help="path of the test_baselines binary (default: ../build)")
    args = ap.parse_args(argv)
    if not args.spawn:
        record = run_party(args.proto, args.role, args.host, args.port, args.n)
        print(json.dumps(record), flush=True)
        return 0 if record["outputs_correct"] else 4
    binary = os.path.abspath(args.binary) if args.binary else find_binary()
    if not args.all:
        return 0 if spawn(args.proto, args.role, args.host, args.n, binary) else 1
    configs = [
        ("zhang", 0, [4, 8, 10]),
        ("zhang", 1, [4, 8, 10]),
        ("zhang", 1, [1, 12, 12]),
        ("sh", 0, [1, 4, 8, 10, 10]),
        ("sh", 1, [1, 4, 8, 10, 10]),
        ("sh", 0, [12, 10, 2]),
    ]
    failed = 0
    for proto, role, ns in configs:
        print(f"{proto}, keys n = {ns}:", flush=True)
        failed += not spawn(proto, role, args.host, ns, binary)
    print(f"{len(configs) - failed} of {len(configs)} mixed-language runs succeeded")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
