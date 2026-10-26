#!/bin/sh
# build.sh -- build the couplet C++ tests, benchmarks and drivers with g++ (or clang++).
# Usage: sh build.sh  (from any directory; CXX and CXXFLAGS override the defaults)
# The code needs -maes -mpclmul and uses AVX2 when -mavx2 is given (the CPU must then have it).
# scripts/build_linux.sh checks the tools and the CPU first and calls this script.
# The g++ build was written for Ubuntu 22.04 (g++ 11) but NOT executed on the machine of the
# recorded runs (MSVC only there): every recorded number comes from build_msvc.bat.
set -eu
cd "$(dirname "$0")"
mkdir -p build
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--O3 -std=c++17 -maes -mpclmul -mavx2 -pthread -DNDEBUG -Iinclude}"

TARGETS="selftest bench_prims check_vectors test_backend_prims test_fss bench_fss test_prep bench_prep
test_engine test_protocol couplet_party test_baselines bench_baselines app_e2e bench_gates test_ppml
couplet_infer bench_ppml couplet_item"

for t in $TARGETS; do
    echo "[$CXX] $t"
    rm -f "build/$t"
    # shellcheck disable=SC2086  (CXXFLAGS is a list of options)
    if ! $CXX $CXXFLAGS "src/$t.cpp" -o "build/$t"; then
        echo "BUILD FAILED: $t (compiler: $CXX, flags: $CXXFLAGS)"
        exit 1
    fi
    if [ ! -x "build/$t" ]; then
        echo "BUILD FAILED: build/$t was not produced"
        exit 1
    fi
done
# The flags of this build, for the machine description of bench/run_campaign.py.
echo "$CXXFLAGS" > build/build_flags.txt
echo "BUILD OK"
