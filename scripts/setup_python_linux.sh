#!/bin/sh
# setup_python_linux.sh -- create the virtual environment .venv (Python 3.10) and install requirements.txt.
# Usage: sh scripts/setup_python_linux.sh      (PYTHON_BASE=python3.x selects another interpreter >= 3.10)
# The other scripts use .venv/bin/python when it exists (or the interpreter named by PYTHON).
# Written for Ubuntu 22.04 (Python 3.10) but not executed there (see README.md).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)

fail() {
    echo "setup_python_linux.sh: $*" >&2
    exit 1
}

BASE=${PYTHON_BASE:-}
if [ -z "$BASE" ]; then
    if command -v python3.10 >/dev/null 2>&1; then
        BASE=python3.10
    elif command -v python3 >/dev/null 2>&1; then
        BASE=python3
    else
        fail "python3.10 was not found (Ubuntu 22.04: sudo apt install python3.10 python3.10-venv)"
    fi
fi
command -v "$BASE" >/dev/null 2>&1 || fail "the interpreter '$BASE' was not found"
"$BASE" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' ||
    fail "$("$BASE" --version 2>&1) is too old: Python 3.10 or newer is needed"
"$BASE" -c 'import venv, ensurepip' 2>/dev/null ||
    fail "the venv module of $BASE is incomplete (Ubuntu 22.04: sudo apt install python3.10-venv)"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "creating $ROOT/.venv with $("$BASE" --version 2>&1)"
    "$BASE" -m venv "$ROOT/.venv" || fail "python -m venv failed (Ubuntu 22.04: sudo apt install python3.10-venv)"
fi
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt" || fail "pip install -r requirements.txt failed"
"$ROOT/.venv/bin/python" -c 'import sys, numpy, matplotlib; print("Python", sys.version.split()[0], "NumPy", numpy.__version__, "matplotlib", matplotlib.__version__)' ||
    fail "numpy or matplotlib cannot be imported"
echo "OK: the scripts use $ROOT/.venv/bin/python"
