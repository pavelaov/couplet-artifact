#!/bin/sh
# run_tests.sh -- run every test of the artifact (POSIX shell).
# Usage: sh scripts/run_tests.sh [GROUP ...]       groups: python cpp interop (default: all three)
#   python   the 17 suites python/tests/test_*.py (the last five are the Couplet layer)
#   cpp      the 9 programs selftest, check_vectors and test_* of cpp/build (test_ppml: Couplet layer)
#   interop  the mixed-language runs of cpp/tests (one party in C++, one in Python)
# Before cpp and interop: sh scripts/build_linux.sh and sh scripts/regen_circuits.sh.
# Python: $PYTHON, else .venv/bin/python, else python3.
# Output: one line per program on the console; test_logs/NAME.log (complete output of a program),
# test_logs/summary.txt, and test_logs/test_log.txt (the summary followed by every log).
# Exit status 0 iff every program ended with status 0.  Nothing is repeated automatically;
# python tests/test_net.py asserts wall-clock times and can fail on a busy machine.
# Written for Ubuntu 22.04 but not executed there (see README.md).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON=python3; fi
fi
LOGDIR="$ROOT/test_logs"
TOTAL=0
PASSED=0
FAILED=0
FAILED_NAMES=""

if [ "$#" -eq 0 ]; then
    set -- python cpp interop
fi
for g in "$@"; do
    case "$g" in
        python|cpp|interop) ;;
        *) echo "unknown group $g; the groups are: python cpp interop"; exit 2 ;;
    esac
done

if ! "$PYTHON" -c "import numpy" >/dev/null 2>&1; then
    echo "The interpreter '$PYTHON' was not found or cannot import numpy: run sh scripts/setup_python_linux.sh."
    exit 2
fi
mkdir -p "$LOGDIR"
rm -f "$LOGDIR"/*.log "$LOGDIR/full.tmp"
echo "name | exit status | seconds | result line of the output" > "$LOGDIR/summary.txt"
: > "$LOGDIR/full.tmp"
echo "logs: test_logs/"
echo "python: $("$PYTHON" -c 'import sys, numpy; print(sys.version.split()[0], "NumPy", numpy.__version__)')"

# run NAME DIR COMMAND...: runs COMMAND in DIR (relative to the artifact root).
run() {
    name=$1
    dir=$2
    shift 2
    TOTAL=$((TOTAL + 1))
    log="$LOGDIR/$name.log"
    echo "[$TOTAL] $name   (in $dir: $*)"
    t0=$(date +%s)
    ( cd "$ROOT/$dir" && "$@" ) > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    # The line to show: the last one that names its checks, else the last line of the output.
    last=$(tr -d '\r' < "$log" | grep -i 'checks' | tail -n 1)
    [ -n "$last" ] || last=$(tr -d '\r' < "$log" | grep -v '^[[:space:]]*$' | tail -n 1)
    if [ "$rc" -eq 0 ]; then
        PASSED=$((PASSED + 1))
        echo "      PASS  ($elapsed s)  $last"
    else
        FAILED=$((FAILED + 1))
        FAILED_NAMES="$FAILED_NAMES $name"
        echo "      FAIL  exit status $rc  ($elapsed s)  $last"
    fi
    echo "$name | $rc | $elapsed | $last" >> "$LOGDIR/summary.txt"
    { echo; echo "===== $name (in $dir: $*): exit status $rc, $elapsed s"; tr -d '\r' < "$log"; } >> "$LOGDIR/full.tmp"
}

# The native programs and the circuit files that they load must exist.
NATIVE_CHECKED=""
NATIVE_OK=""
need_native() {
    if [ -z "$NATIVE_CHECKED" ]; then
        NATIVE_CHECKED=1
        if [ ! -x "$ROOT/cpp/build/test_protocol" ] && [ ! -f "$ROOT/cpp/build/test_protocol.exe" ]; then
            missing="cpp/build is not built: run sh scripts/build_linux.sh first"
        elif ! "$PYTHON" "$ROOT/python/bench/regen_circuits.py" --tests --check > "$LOGDIR/circuits_check.txt" 2>&1; then
            missing="the circuit files of the tests are missing or differ: run sh scripts/regen_circuits.sh first"
        else
            missing=""
        fi
        if [ -z "$missing" ]; then
            NATIVE_OK=1
        else
            TOTAL=$((TOTAL + 1))
            FAILED=$((FAILED + 1))
            FAILED_NAMES="$FAILED_NAMES native-prerequisites"
            echo "[$TOTAL] native prerequisites"
            echo "      FAIL  $missing"
            echo "native-prerequisites | 1 | 0 | $missing" >> "$LOGDIR/summary.txt"
        fi
    fi
    [ -n "$NATIVE_OK" ]
}

group_python() {
    for t in test_abb test_attacks test_baselines test_circuits test_fss_exhaustive test_gates \
             test_net test_prep test_primitives test_protocol test_spdz2k test_autheval \
             test_ppml_circuits test_planes test_truncmask test_nn test_ppml_attacks; do
        run "py_$t" python "$PYTHON" "tests/$t.py"
    done
}

group_cpp() {
    need_native || return 0
    for t in selftest check_vectors test_backend_prims test_fss test_prep test_engine test_protocol \
             test_baselines test_ppml; do
        run "cpp_$t" cpp "build/$t"
    done
}

group_interop() {
    need_native || return 0
    run interop_gen cpp "$PYTHON" tests/interop_peer.py --spawn --all
    run interop_online cpp "$PYTHON" tests/interop_peer.py --online --spawn --all
    run interop_baselines cpp "$PYTHON" tests/interop_baselines.py --spawn --all
    run interop_prep cpp "$PYTHON" tests/interop_prep.py --spawn --all
    run interop_infer cpp "$PYTHON" tests/interop_peer.py --infer --spawn --all
}

T_START=$(date +%s)
for g in "$@"; do
    "group_$g"
done
T_ALL=$(($(date +%s) - T_START))

echo
echo "SUMMARY: $TOTAL programs, $PASSED passed, $FAILED failed ($T_ALL s)"
echo "SUMMARY: $TOTAL programs, $PASSED passed, $FAILED failed ($T_ALL s)" >> "$LOGDIR/summary.txt"
STATUS=0
if [ "$FAILED" -ne 0 ]; then
    echo "failed:$FAILED_NAMES"
    echo "failed:$FAILED_NAMES" >> "$LOGDIR/summary.txt"
    STATUS=1
else
    echo "ALL TESTS PASSED"
fi
{
    echo "Test log of scripts/run_tests.sh (groups: $*)"
    echo "python: $("$PYTHON" -c 'import sys, numpy; print(sys.version.split()[0], "NumPy", numpy.__version__)')"
    echo
    cat "$LOGDIR/summary.txt"
    cat "$LOGDIR/full.tmp"
} > "$LOGDIR/test_log.txt"
rm -f "$LOGDIR/full.tmp"
exit $STATUS
