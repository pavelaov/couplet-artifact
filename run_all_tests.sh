#!/bin/sh
# run_all_tests.sh -- run every test program of the artifact (POSIX shell).
#
# Usage: ./run_all_tests.sh [GROUP ...]          (from any directory)
#
# Groups (default: the first three, in this order):
#   python      the 17 suites python/tests/test_*.py of the reference implementation (the last
#               five, test_ppml_circuits to test_ppml_attacks, are the Couplet layer)
#   cpp         the 9 programs selftest, check_vectors and test_* of cpp/build (test_ppml is the
#               Couplet layer)
#   interop     the mixed-language runs (one party per language) of cpp/tests; interop_infer is
#               the inference of the Couplet layer (couplet_infer against the reference)
#   generators  the self-tests of the table and figure generators in python/bench.
#               Not in the default set: they read the measurement records under
#               experiments/raw, written by python/bench/run_campaign.py.
#
# Before the groups cpp and interop: build with cpp/build.sh, and make sure that the circuit
# files exist (python3 python/bench/regen_circuits.py --tests; they are not part of the packaged archive).
# The Python interpreter is "python3"; set the variable PYTHON to use another one.
#
# Every program runs with its default arguments.  Its complete output goes to
# test_logs/NAME.log; the console and test_logs/summary.txt get one line per program
# (name, exit status, seconds, and the result line: the last output line that names its
# checks, else the last line).  The exit status of this script is 0 iff every program
# ended with status 0.  Nothing is repeated automatically: python
# tests/test_net.py asserts wall-clock times and can fail on a busy machine; run it again
# alone in that case.
#
# NOTE: the runs recorded so far used the MSVC binaries (this script finds cpp/build/*.exe
# as well, under a POSIX shell on Windows); the g++ build of cpp/build.sh is still to be run.
set -u
ROOT=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
LOGDIR="$ROOT/test_logs"
TOTAL=0
PASSED=0
FAILED=0
FAILED_NAMES=""
NATIVE_REPORTED=""

if [ "$#" -eq 0 ]; then
    set -- python cpp interop
fi
for g in "$@"; do
    case "$g" in
        python|cpp|interop|generators) ;;
        *) echo "unknown group $g; the groups are: python cpp interop generators"; exit 2 ;;
    esac
done

if ! $PYTHON -c "import numpy" >/dev/null 2>&1; then
    echo "The interpreter \"$PYTHON\" was not found or cannot import numpy; see requirements.txt."
    exit 2
fi
mkdir -p "$LOGDIR"
echo "name | exit status | seconds | result line of the output" > "$LOGDIR/summary.txt"
echo "logs: $LOGDIR"

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
}

# The native programs and the circuit files that they load must exist.
need_native() {
    missing=""
    if [ ! -x "$ROOT/cpp/build/test_protocol" ] && [ ! -f "$ROOT/cpp/build/test_protocol.exe" ]; then
        missing="cpp/build is not built: run cpp/build.sh first"
    elif [ ! -f "$ROOT/circuits/dpf_n8_k64_s64_xor_one.bin" ]; then
        missing="the circuit files are missing: run  $PYTHON python/bench/regen_circuits.py --tests  first"
    fi
    [ -z "$missing" ] && return 0
    [ -n "$NATIVE_REPORTED" ] && return 1
    NATIVE_REPORTED=1
    TOTAL=$((TOTAL + 1))
    FAILED=$((FAILED + 1))
    FAILED_NAMES="$FAILED_NAMES native-prerequisites"
    echo "[$TOTAL] native prerequisites"
    echo "      FAIL  $missing"
    echo "native-prerequisites | 1 | 0 | $missing" >> "$LOGDIR/summary.txt"
    return 1
}

group_python() {
    for t in test_abb test_attacks test_baselines test_circuits test_fss_exhaustive test_gates \
             test_net test_prep test_primitives test_protocol test_spdz2k test_autheval \
             test_ppml_circuits test_planes test_truncmask test_nn test_ppml_attacks; do
        run "py_$t" python $PYTHON "tests/$t.py"
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
    run interop_gen cpp $PYTHON tests/interop_peer.py --spawn --all
    run interop_online cpp $PYTHON tests/interop_peer.py --online --spawn --all
    run interop_baselines cpp $PYTHON tests/interop_baselines.py --spawn --all
    run interop_prep cpp $PYTHON tests/interop_prep.py --spawn --all
    run interop_infer cpp $PYTHON tests/interop_peer.py --infer --spawn --all
}

group_generators() {
    run gen_make_tables python $PYTHON bench/make_tables.py --selftest
    run gen_make_figures python $PYTHON bench/make_figures.py --selftest
}

for g in "$@"; do
    "group_$g"
done

echo
echo "SUMMARY: $TOTAL programs, $PASSED passed, $FAILED failed"
echo "SUMMARY: $TOTAL programs, $PASSED passed, $FAILED failed" >> "$LOGDIR/summary.txt"
if [ "$FAILED" -ne 0 ]; then
    echo "failed:$FAILED_NAMES"
    echo "failed:$FAILED_NAMES" >> "$LOGDIR/summary.txt"
    exit 1
fi
echo "ALL TESTS PASSED"
exit 0
