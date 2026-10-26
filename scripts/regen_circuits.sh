#!/bin/sh
# regen_circuits.sh -- rebuild the circuit files of circuits/ from circuits/manifest.json and check
# their SHA-256 digests against it (the files are not shipped; the compiler is deterministic).
# Usage: sh scripts/regen_circuits.sh            the 36 circuits that the tests load (about 0.4 GB, < 1 min)
#        sh scripts/regen_circuits.sh --missing  every circuit of the manifest that is missing (3.6 GB)
#        sh scripts/regen_circuits.sh --check    verify the files that are present
#        sh scripts/regen_circuits.sh NAME ...   the named circuits
# The Bristol text versions are deleted after their digest was checked (no program reads them).
# Python: $PYTHON, else .venv/bin/python, else python3.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python3; fi
fi
"$PYTHON" -c "import numpy" >/dev/null 2>&1 ||
    { echo "regen_circuits.sh: '$PYTHON' cannot import numpy: run sh scripts/setup_python_linux.sh first" >&2; exit 2; }

if [ "$#" -eq 0 ]; then
    set -- --tests
elif [ "$1" = "--missing" ]; then
    shift
fi
exec "$PYTHON" "$ROOT/python/bench/regen_circuits.py" --no-bristol "$@"
