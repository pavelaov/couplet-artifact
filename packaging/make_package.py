"""Assemble the minimal artifact package from this working tree.

Copies a fixed list of sources (no build outputs, caches, circuit files or raw
measurement records), the README of the package, and the final outputs of the
campaign into ``results/``, then regenerates ``results/tables`` and
``results/figures`` from ``results/summary.json`` with the package's own
generators.

Usage (from any directory):
    python packaging/make_package.py TARGET_DIR [--test-log FILE] [--replace]

TARGET_DIR must not exist, unless --replace is given (then it is deleted
first; use it only on a directory that this script created).  --test-log
copies FILE to results/test_log.txt.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys

TREE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# (source relative to the working tree, destination relative to the package); directories are copied
# recursively without __pycache__ and *.pyc.
FILES = [
    ("LICENSE", "LICENSE"),
    ("requirements.txt", "requirements.txt"),
    ("packaging/README_package.md", "README.md"),
    ("circuits/manifest.json", "circuits/manifest.json"),
    ("cpp/build.sh", "cpp/build.sh"),
    ("cpp/build_msvc.bat", "cpp/build_msvc.bat"),
    ("cpp/include/couplet", "cpp/include/couplet"),
    ("cpp/src", "cpp/src"),
    ("cpp/tests/circuits", "cpp/tests/circuits"),
    ("cpp/tests/vectors_protocol.json", "cpp/tests/vectors_protocol.json"),
    ("cpp/tests/interop_baselines.py", "cpp/tests/interop_baselines.py"),
    ("cpp/tests/interop_peer.py", "cpp/tests/interop_peer.py"),
    ("cpp/tests/interop_prep.py", "cpp/tests/interop_prep.py"),
    ("python/couplet", "python/couplet"),
    ("python/tests", "python/tests"),
    ("python/vectors/ascon_vectors.json", "python/vectors/ascon_vectors.json"),
    ("python/vectors/fss_vectors.json", "python/vectors/fss_vectors.json"),
    ("python/vectors/ppml", "python/vectors/ppml"),
    ("python/bench/run_campaign.py", "python/bench/run_campaign.py"),
    ("python/bench/make_tables.py", "python/bench/make_tables.py"),
    ("python/bench/make_figures.py", "python/bench/make_figures.py"),
    ("python/bench/export_circuits.py", "python/bench/export_circuits.py"),
    ("python/bench/ppml_attacks.py", "python/bench/ppml_attacks.py"),
    ("python/bench/regen_circuits.py", "python/bench/regen_circuits.py"),
    ("scripts", "scripts"),
    # final outputs of the campaign
    ("experiments/raw/campaign/summary.json", "results/summary.json"),
    ("experiments/raw/campaign/summary.txt", "results/summary.txt"),
    ("experiments/raw/machine.json", "results/machine.json"),
    ("experiments/raw/attacks.json", "results/attacks.json"),
    ("experiments/raw/attacks_ppml.json", "results/attacks_ppml.json"),
]
IGNORE = ("__pycache__", "*.pyc", "*.pyo", "*.obj", "*.exe", "*.tmp")


def _ignored(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in IGNORE)


def copy(src: str, dst: str) -> int:
    if os.path.isdir(src):
        n = 0
        os.makedirs(dst, exist_ok=True)
        for entry in sorted(os.listdir(src)):
            if not _ignored(entry):
                n += copy(os.path.join(src, entry), os.path.join(dst, entry))
        return n
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="assemble the minimal artifact package")
    ap.add_argument("target")
    ap.add_argument("--test-log", default=None, help="file to copy to results/test_log.txt")
    ap.add_argument("--replace", action="store_true", help="delete TARGET first (only a directory made by this script)")
    args = ap.parse_args(argv)
    target = os.path.abspath(args.target)
    if os.path.exists(target):
        if not args.replace:
            raise SystemExit(f"{target} exists; use --replace only for a directory that this script created")
        shutil.rmtree(target)
    n = 0
    for src, dst in FILES:
        n += copy(os.path.join(TREE, src), os.path.join(target, dst))
    py = os.path.join(target, "python")
    summary = os.path.join(target, "results", "summary.json")
    for script, out in (("make_tables.py", "tables"), ("make_figures.py", "figures")):
        subprocess.run([sys.executable, "-B", os.path.join("bench", script), "--summary", summary,
                        "--out", os.path.join(target, "results", out)], cwd=py, check=True)
    if args.test_log:
        shutil.copyfile(args.test_log, os.path.join(target, "results", "test_log.txt"))
    for dp, dns, _ in os.walk(target):                          # the generators must not leave caches
        for d in [d for d in dns if d == "__pycache__"]:
            shutil.rmtree(os.path.join(dp, d))
    files = sum(len(fs) for _, _, fs in os.walk(target))
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(target) for f in fs)
    print(f"{target}: {files} files, {size:,} bytes ({n} copied, the rest generated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
