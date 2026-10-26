#!/bin/sh
# make_tables.sh -- generate the tables (LaTeX bodies, number macros, PROVENANCE.md) and the figures (PDF)
# of the evaluation from a campaign summary, into out/tables and out/figures.
# Usage: sh scripts/make_tables.sh [results|campaign|quick|PATH/summary.json]
#   results   (default) results/summary.json, the shipped outputs; the generated files are then
#             compared with results/tables and results/figures (exit status 1 if any differs)
#   campaign  experiments/raw/campaign/summary.json (sh scripts/run_experiments.sh)
#   quick     experiments/quick/campaign/summary.json (sh scripts/run_experiments.sh --quick)
# Configurations that a summary does not contain are left out and listed in out/tables/PROVENANCE.md.
# Python: $PYTHON, else .venv/bin/python, else python3 (needs numpy and matplotlib).
# Written for Ubuntu 22.04 but not executed there (see README.md).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python3; fi
fi
"$PYTHON" -c "import numpy, matplotlib" >/dev/null 2>&1 ||
    { echo "make_tables.sh: '$PYTHON' cannot import numpy and matplotlib: run sh scripts/setup_python_linux.sh first" >&2; exit 2; }

SRC=${1:-results}
case "$SRC" in
    results) SUMMARY="$ROOT/results/summary.json" ;;
    campaign) SUMMARY="$ROOT/experiments/raw/campaign/summary.json" ;;
    quick) SUMMARY="$ROOT/experiments/quick/campaign/summary.json" ;;
    *) SUMMARY=$SRC ;;
esac
[ -f "$SUMMARY" ] || { echo "make_tables.sh: no summary at $SUMMARY" >&2; exit 2; }
OUT="$ROOT/out"
rm -rf "$OUT/tables" "$OUT/figures"
cd "$ROOT/python"
"$PYTHON" bench/make_tables.py --summary "$SUMMARY" --out "$OUT/tables"
"$PYTHON" bench/make_figures.py --summary "$SUMMARY" --out "$OUT/figures"

if [ "$SRC" = "results" ]; then
    # Tables must be reproduced byte for byte.  The PDF files embed the matplotlib version and its
    # font subsets, so they are equal only with the matplotlib of the recorded run (3.11.2).
    differ=0
    for f in "$ROOT/results/tables"/*; do
        b=$(basename "$f")
        if cmp -s "$f" "$OUT/tables/$b"; then
            echo "same     results/tables/$b"
        else
            echo "DIFFERS  results/tables/$b"
            differ=$((differ + 1))
        fi
    done
    for f in "$ROOT/results/figures"/*; do
        b=$(basename "$f")
        if cmp -s "$f" "$OUT/figures/$b"; then
            echo "same     results/figures/$b"
        else
            echo "differs  results/figures/$b (byte-identical only with matplotlib 3.11.2; compare visually)"
        fi
    done
    if [ "$differ" -ne 0 ]; then
        echo "$differ table files differ from results/tables"
        exit 1
    fi
    echo "every table file of results/tables was reproduced byte for byte in out/tables"
fi
