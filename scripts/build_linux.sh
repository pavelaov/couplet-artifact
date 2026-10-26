#!/bin/sh
# build_linux.sh -- build the 19 native programs into cpp/build (Ubuntu 22.04, g++ 11).
# Usage: sh scripts/build_linux.sh        (CXX=... selects another compiler)
# Flags: -O3 -std=c++17 -maes -mpclmul -mavx2 -pthread (plus -DNDEBUG -Iinclude); no cmake.
# Written for Ubuntu 22.04 with g++ 11 but not executed there (see README.md).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CXX=${CXX:-g++}
FLAGS="-O3 -std=c++17 -maes -mpclmul -mavx2 -pthread -DNDEBUG -Iinclude"

fail() {
    echo "build_linux.sh: $*" >&2
    exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "the compiler '$CXX' was not found (Ubuntu: sudo apt install g++)"
if "$CXX" -dM -E -x c++ /dev/null 2>/dev/null | grep -q '__clang__'; then
    echo "compiler: $("$CXX" --version | head -n 1) (clang; the tested target is g++ 11)"
else
    ver=$("$CXX" -dumpversion 2>/dev/null || echo 0)
    major=${ver%%.*}
    case "$major" in ''|*[!0-9]*) major=0 ;; esac
    [ "$major" -ge 11 ] || fail "g++ $ver is too old: g++ 11 or newer is needed (Ubuntu 22.04: sudo apt install g++-11)"
    echo "compiler: $("$CXX" --version | head -n 1)"
fi

# The programs use AES-NI/PCLMULQDQ and AVX2 instructions: refuse a CPU without them.
if [ -r /proc/cpuinfo ]; then
    for f in aes pclmulqdq avx2; do
        grep -q -w "$f" /proc/cpuinfo || fail "this CPU lacks '$f' (needed by the -maes -mpclmul -mavx2 build)"
    done
else
    echo "warning: /proc/cpuinfo not readable; the CPU must support AES-NI, PCLMULQDQ and AVX2"
fi

# The compiler must accept the flags and the intrinsics headers.
TMP=$(mktemp -d "${TMPDIR:-/tmp}/couplet_build_check.XXXXXX") || fail "mktemp failed"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/check.cpp" <<'EOF'
#include <immintrin.h>
#include <wmmintrin.h>
#include <filesystem>
#include <thread>
int main() {
    __m128i a = _mm_set_epi64x(3, 5), b = _mm_clmulepi64_si128(a, a, 0);
    __m256i c = _mm256_xor_si256(_mm256_set1_epi64x(1), _mm256_set1_epi64x(2));
    unsigned __int128 w = (unsigned __int128)_mm_cvtsi128_si64(b) * 7u;
    std::thread t([] {});
    t.join();
    return (int)(w & 1) + (int)_mm256_extract_epi64(c, 0) + (int)std::filesystem::path("x").string().size() - 5;
}
EOF
# shellcheck disable=SC2086
$CXX $FLAGS "$TMP/check.cpp" -o "$TMP/check" 2> "$TMP/check.err" ||
    { cat "$TMP/check.err" >&2; fail "'$CXX $FLAGS' cannot compile a test program (see the messages above)"; }

CXX="$CXX" CXXFLAGS="$FLAGS" sh "$ROOT/cpp/build.sh"
