"""Cross-language test of the triple request: Python ``RealPrep`` against ``test_prep --interop``.

One party is the C++ program ``build/test_prep --interop PARTY PORT N OUTFILE``
(``RealPrep.setup()`` and ``and_triples(N)`` over TCP on 127.0.0.1); the other
party is the Python reference, run by this script.  The C++ party writes its
key share and its triple shares to OUTFILE.  The script reads them and checks
across the two languages: N triples for both parties, ``c = a AND b``, the
MAC invariant of ``a``, ``b`` and ``c``, the bucket size, and that both
programs counted the same bytes.

A request below 306 raw leaky triples is padded by both implementations with
the same rule (``couplet.prep.params.triple_request``): N = 1, 5 and 43 are
padded to 44 buckets of size 7, N = 44 and N = 200 are not padded.

    python tests/interop_prep.py --spawn --all          # the five sizes, both role assignments
    python tests/interop_prep.py --spawn --n 5 --role 0

``--build-dir DIR`` takes the C++ binary from ``DIR`` instead of ``../build``.
Exit status 0 iff every run completed and passed all checks.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "python"))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import gf128 as gf  # noqa: E402
from couplet.net import TcpChannel  # noqa: E402
from couplet.prep import RealPrep, params  # noqa: E402

SID = b"interop-session"          # the identifier used by test_prep --interop
SIGMA = 40
ALL_SIZES = (1, 5, 43, 44, 200)
_BUILD_DIR = os.path.normpath(os.path.join(_HERE, "..", "build"))   # replaced by --build-dir


def find_binary(stem: str = "test_prep") -> str:
    for name in (stem + ".exe", stem):
        path = os.path.join(_BUILD_DIR, name)
        if os.path.isfile(path):
            return os.path.normpath(path)
    raise SystemExit(f"{stem} not found in {_BUILD_DIR}: build the C++ code first")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def read_view(path: str, n: int):
    """The view written by the C++ party: Delta share, then (bits, MAC shares) of a, b, c."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != 16 + 3 * 17 * n:
        raise ValueError(f"the view has {len(data)} bytes, expected {16 + 3 * 17 * n}")
    delta = int.from_bytes(data[:16], "little")
    parts, pos = [], 16
    for _ in range(3):
        bits = np.frombuffer(data[pos:pos + n], dtype=np.uint8)
        macs = gf.bytes_to_vec(data[pos + n:pos + 17 * n])
        parts.append((bits, macs))
        pos += 17 * n
    return delta, parts


def run_python_party(role: int, port: int, n: int) -> dict:
    if role == 0:
        chan = TcpChannel.listen(port, "127.0.0.1", party=0, accept_timeout=120)
    else:
        chan = TcpChannel.connect("127.0.0.1", port, party=1, connect_timeout=120)
    try:
        prep = RealPrep(role, chan, SID, SIGMA)
        delta = prep.delta
        prep.setup()
        a, b, c = prep.and_triples(n)
        return {"delta": delta, "triples": (a, b, c), "bucket": prep.last_bucket_size,
                "leaky": prep.counters["leaky_triples"], "bytes_sent": chan.bytes_sent,
                "bytes_recv": chan.bytes_recv, "rounds": chan.rounds}
    finally:
        chan.close()


def run_one(n: int, role: int) -> bool:
    """One mixed-language run with the Python reference in ``role``; True iff every check passes."""
    ell, bucket = params.triple_request(n, SIGMA)
    port = free_port()
    failures = []
    with tempfile.TemporaryDirectory(prefix="interop_prep_") as tmp:
        view = os.path.join(tmp, "cpp_view.bin")
        proc = subprocess.Popen([find_binary(), "--interop", str(1 - role), str(port), str(n), view],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        mine = None
        try:
            mine = run_python_party(role, port, n)
        except Exception as exc:  # noqa: BLE001 - report and fail
            failures.append(f"python side failed: {type(exc).__name__}: {exc}")
        try:
            out, err = proc.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        if proc.returncode != 0:
            failures.append(f"c++ side: exit {proc.returncode}: {(out or '').strip()} {(err or '').strip()}")
        if not failures:
            delta_cpp, parts = read_view(view, n)
            delta = mine["delta"] ^ delta_cpp
            d0, d1 = (mine["delta"], delta_cpp) if role == 0 else (delta_cpp, mine["delta"])
            if d0 & 1 != 1 or d1 & 1 != 0:
                failures.append("lsb(Delta_0) = 1, lsb(Delta_1) = 0")
            values = []
            for name, share, (bits, macs) in zip("abc", mine["triples"], parts):
                if len(share) != n or share.macs.shape != (n, 2):
                    failures.append(f"python holds {len(share)} shares of {name}, expected {n}")
                    continue
                x = share.bits ^ bits
                values.append(x)
                if not np.array_equal(share.macs ^ macs, gf.bit_mul(x, delta)):
                    failures.append(f"MAC invariant of {name}")
            if len(values) == 3 and not np.array_equal(values[2], values[0] & values[1]):
                failures.append("c = a AND b")
            m = re.search(r"bucket=(\d+) bytes_sent=(\d+) bytes_recv=(\d+) rounds=(\d+)", out or "")
            if not m:
                failures.append("the c++ side printed no summary line")
            else:
                cpp_bucket, cpp_sent, cpp_recv, cpp_rounds = (int(g) for g in m.groups())
                if not cpp_bucket == mine["bucket"] == bucket:
                    failures.append(f"bucket size: c++ {cpp_bucket}, python {mine['bucket']}, expected {bucket}")
                if (cpp_sent, cpp_recv) != (mine["bytes_recv"], mine["bytes_sent"]):
                    failures.append(f"bytes: c++ sent {cpp_sent} / received {cpp_recv}, python received "
                                    f"{mine['bytes_recv']} / sent {mine['bytes_sent']}")
                if cpp_rounds != mine["rounds"]:
                    failures.append(f"rounds: c++ {cpp_rounds}, python {mine['rounds']}")
            if mine["leaky"] != ell * bucket:
                failures.append(f"leaky triples: {mine['leaky']}, expected {ell * bucket}")
    status = "ok" if not failures else "FAILED: " + "; ".join(failures)
    sent = mine["bytes_sent"] if mine else 0
    print(f"  n = {n:>3}, python role {role}: request of {ell} buckets, B = {bucket} ({ell * bucket} leaky triples), "
          f"{sent} bytes sent by the python party: {status}", flush=True)
    return not failures


def main(argv=None) -> int:
    global _BUILD_DIR
    ap = argparse.ArgumentParser(description="Python RealPrep against test_prep --interop (cross-language test)")
    ap.add_argument("--role", type=int, default=1, choices=(0, 1), help="role of this script (default 1)")
    ap.add_argument("--n", type=int, default=5, help="number of triples of the request (default 5)")
    ap.add_argument("--spawn", action="store_true", help="start build/test_prep --interop in the other role")
    ap.add_argument("--all", action="store_true", help="with --spawn: the five sizes, both role assignments")
    ap.add_argument("--build-dir", default=None, help="directory of the C++ binaries (default: ../build)")
    args = ap.parse_args(argv)
    if args.build_dir:
        _BUILD_DIR = os.path.abspath(args.build_dir)
    if not args.spawn:
        ap.error("this script starts the C++ party itself: pass --spawn")
    if args.n < 1:
        ap.error("--n must be positive")
    runs = [(n, role) for n in ALL_SIZES for role in (0, 1)] if args.all else [(args.n, args.role)]
    good = sum(run_one(n, role) for n, role in runs)
    print(f"{good} of {len(runs)} mixed-language runs succeeded")
    return 0 if good == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
