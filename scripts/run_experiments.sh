#!/bin/sh
# run_experiments.sh -- the measurement campaign (python/bench/run_campaign.py).
# Usage: sh scripts/run_experiments.sh --quick      smoke run of every driver (minutes), output in
#                                                    experiments/quick/campaign
#        sh scripts/run_experiments.sh [ARGS...]     the whole campaign (about 8 hours on the authors'
#                                                    machine; the largest rows need 36 GiB of memory per
#                                                    party), output in experiments/raw/campaign
# Further ARGS go to run_campaign.py (for example --only u,k or --match mlp16; see --help).
# The whole campaign first rebuilds every circuit of circuits/manifest.json that is missing (3.6 GB).
# Then: sh scripts/make_tables.sh quick   or   sh scripts/make_tables.sh campaign
# Background load and peak memory are read from Windows counters only; on Linux those fields stay
# empty and every run counts as valid for timing (use a quiet machine).
# Python: $PYTHON, else .venv/bin/python, else python3.
# Written for Ubuntu 22.04 but not executed there (see README.md).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python3; fi
fi
"$PYTHON" -c "import numpy" >/dev/null 2>&1 ||
    { echo "run_experiments.sh: '$PYTHON' cannot import numpy: run sh scripts/setup_python_linux.sh first" >&2; exit 2; }
for p in couplet_item couplet_party couplet_infer bench_ppml bench_baselines; do
    [ -x "$ROOT/cpp/build/$p" ] || [ -f "$ROOT/cpp/build/$p.exe" ] ||
        { echo "run_experiments.sh: cpp/build/$p is missing: run sh scripts/build_linux.sh first" >&2; exit 2; }
done

QUICK=""
for a in "$@"; do
    [ "$a" = "--quick" ] && QUICK=1
done
if [ -z "$QUICK" ]; then
    case " $* " in
        *" --list "*|*" --summary-only "*|*" --help "*|*" -h "*) ;;
        *) "$PYTHON" "$ROOT/python/bench/regen_circuits.py" --no-bristol ;;
    esac
fi
cd "$ROOT/python"
exec "$PYTHON" bench/run_campaign.py "$@"
