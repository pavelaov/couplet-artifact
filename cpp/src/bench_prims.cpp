// bench_prims.cpp -- micro-benchmarks of the couplet primitives.  Prints
// one JSON object to stdout.  Numbers are single-thread wall-clock
// (std::chrono::steady_clock); each kernel is chained through its own output
// where latency is the natural metric (scalar p12, X25519) and run over
// independent inputs where throughput is (interleaved p12).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "couplet/ascon.hpp"
#include "couplet/gf128.hpp"
#include "couplet/sha256.hpp"
#include "couplet/x25519.hpp"

using namespace couplet;

static double now_ns() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t g_sink = 0;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1Dull; }
};

static double bench_p12_scalar(uint64_t iters) {
    uint64_t s[5] = {1, 2, 3, 4, 5};
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) ascon::p12(s);
    double t1 = now_ns();
    g_sink ^= s[0] ^ s[4];
    return (t1 - t0) / (double)iters;
}

static double bench_p12_x4(uint64_t iters) {
    uint64_t st[4][5];
    for (int j = 0; j < 4; j++) for (int w = 0; w < 5; w++) st[j][w] = (uint64_t)(j * 5 + w + 1);
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) ascon::p12_x4(st);
    double t1 = now_ns();
    for (int j = 0; j < 4; j++) g_sink ^= st[j][0];
    return (t1 - t0) / (double)(iters * 4);
}

static double bench_p12_x8(uint64_t iters) {
    uint64_t st[8][5];
    for (int j = 0; j < 8; j++) for (int w = 0; w < 5; w++) st[j][w] = (uint64_t)(j * 5 + w + 1);
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) ascon::p12_x8(st);
    double t1 = now_ns();
    for (int j = 0; j < 8; j++) g_sink ^= st[j][0];
    return (t1 - t0) / (double)(iters * 8);
}

#if COUPLET_HAVE_AVX2
static double bench_p12_x4_avx2(uint64_t iters) {
    uint64_t st[4][5];
    for (int j = 0; j < 4; j++) for (int w = 0; w < 5; w++) st[j][w] = (uint64_t)(j * 5 + w + 1);
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) ascon::p12_x4_avx2(st);
    double t1 = now_ns();
    for (int j = 0; j < 4; j++) g_sink ^= st[j][0];
    return (t1 - t0) / (double)(iters * 4);
}
#endif

// Independent permutations in a large batch (the EvalAll access pattern):
// 4-way blocks over a buffer of many states.
static double bench_p12_batch(size_t nstates, int reps) {
    std::vector<uint64_t> buf(nstates * 5);
    Rng rng(9);
    for (auto& x : buf) x = rng.next();
    double t0 = now_ns();
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i + 4 <= nstates; i += 4) ascon::p12_x4_fast((uint64_t(*)[5])&buf[i * 5]);
    double t1 = now_ns();
    g_sink ^= buf[0] ^ buf[buf.size() - 1];
    return (t1 - t0) / (double)((nstates / 4) * 4 * (size_t)reps);
}

static double bench_G(uint64_t iters) {
    ascon::seed_t s{};
    s[0] = 1;
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) {
        ascon::GOut g = ascon::G(s);
        s = g.sL;
    }
    double t1 = now_ns();
    g_sink ^= s[0];
    return (t1 - t0) / (double)iters;
}

static double bench_conv(uint64_t iters) {
    ascon::seed_t s{};
    s[0] = 2;
    u128 acc = 0;
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) {
        ascon::ConvOut c = ascon::Conv(s, 64, 64);
        acc = acc + c.val + c.tag;
        ascon::store64le(s.data(), u128_lo(acc));
    }
    double t1 = now_ns();
    g_sink ^= u128_lo(acc);
    return (t1 - t0) / (double)iters;
}

static double bench_gf128(uint64_t iters) {
    gf128 a(0x0123456789abcdefull, 0xfedcba9876543210ull), b(0x9e3779b97f4a7c15ull, 0xf39cc0605cedc834ull);
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) a = gf128_mul(a, b);
    double t1 = now_ns();
    g_sink ^= a.lo;
    return (t1 - t0) / (double)iters;
}

static double bench_gf128_portable(uint64_t iters) {
    gf128 a(0x0123456789abcdefull, 0xfedcba9876543210ull), b(0x9e3779b97f4a7c15ull, 0xf39cc0605cedc834ull);
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) a = gf128_mul_portable(a, b);
    double t1 = now_ns();
    g_sink ^= a.lo;
    return (t1 - t0) / (double)iters;
}

static double bench_gf128_throughput(uint64_t iters) {
    // 8 independent chains
    gf128 a[8], b(0x9e3779b97f4a7c15ull, 0xf39cc0605cedc834ull);
    for (int j = 0; j < 8; j++) a[j] = gf128((uint64_t)j + 1, (uint64_t)j * 7 + 3);
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++)
        for (int j = 0; j < 8; j++) a[j] = gf128_mul(a[j], b);
    double t1 = now_ns();
    for (int j = 0; j < 8; j++) g_sink ^= a[j].lo;
    return (t1 - t0) / (double)(iters * 8);
}

static double bench_x25519(uint64_t iters) {
    x25519::bytes32 k = x25519::basepoint(), u = x25519::basepoint();
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) {
        x25519::bytes32 r = x25519::scalarmult(k, u);
        u = k;
        k = r;
    }
    double t1 = now_ns();
    g_sink ^= k[0];
    return (t1 - t0) / (double)iters;
}

static double bench_sha256_bulk(size_t bytes, int reps, double& ns_total) {
    std::vector<uint8_t> buf(bytes);
    Rng rng(11);
    for (auto& x : buf) x = (uint8_t)rng.next();
    uint8_t out[32];
    double t0 = now_ns();
    for (int r = 0; r < reps; r++) {
        sha256(buf.data(), buf.size(), out);
        buf[0] ^= out[0];
    }
    double t1 = now_ns();
    g_sink ^= out[0];
    ns_total = t1 - t0;
    return (double)bytes * reps / ((t1 - t0) / 1e9) / 1e6;  // MB/s
}

static double bench_sha256_small(uint64_t iters) {
    uint8_t buf[32] = {0};
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) sha256(buf, 32, buf);
    double t1 = now_ns();
    g_sink ^= buf[0];
    return (t1 - t0) / (double)iters;
}

int main() {
    const uint64_t P = 2000000;  // permutations per kernel
    double p12_scalar = bench_p12_scalar(P);
    double p12_x4 = bench_p12_x4(P / 4);
    double p12_x8 = bench_p12_x8(P / 8);
#if COUPLET_HAVE_AVX2
    double p12_avx2 = bench_p12_x4_avx2(P / 4);
#else
    double p12_avx2 = -1;
#endif
    double p12_batch = bench_p12_batch(1 << 16, 8);
    double g_ns = bench_G(1000000);
    double conv_ns = bench_conv(1000000);
    double gf_lat = bench_gf128(5000000);
    double gf_thr = bench_gf128_throughput(1000000);
    double gf_port = bench_gf128_portable(200000);
    double x_ns = bench_x25519(2000);
    double sha_total;
    double sha_mbs = bench_sha256_bulk(1 << 20, 64, sha_total);
    double sha_small = bench_sha256_small(1000000);

    std::printf("{\n");
#if defined(_MSC_VER)
    std::printf("  \"compiler\": \"MSVC %d\",\n", _MSC_VER);
#elif defined(__clang__)
    std::printf("  \"compiler\": \"clang %s\",\n", __clang_version__);
#elif defined(__GNUC__)
    std::printf("  \"compiler\": \"gcc %d.%d\",\n", __GNUC__, __GNUC_MINOR__);
#else
    std::printf("  \"compiler\": \"unknown\",\n");
#endif
    std::printf("  \"avx2\": %s,\n", COUPLET_HAVE_AVX2 ? "true" : "false");
    std::printf("  \"pclmul\": %s,\n", COUPLET_HAVE_PCLMUL ? "true" : "false");
    std::printf("  \"p12_scalar_ns\": %.2f,\n", p12_scalar);
    std::printf("  \"p12_x4_interleaved_ns\": %.2f,\n", p12_x4);
    std::printf("  \"p12_x8_interleaved_ns\": %.2f,\n", p12_x8);
    if (p12_avx2 >= 0) std::printf("  \"p12_x4_avx2_ns\": %.2f,\n", p12_avx2);
    else std::printf("  \"p12_x4_avx2_ns\": null,\n");
    std::printf("  \"p12_batch_65536_ns\": %.2f,\n", p12_batch);
    std::printf("  \"p12_scalar_per_s\": %.0f,\n", 1e9 / p12_scalar);
    std::printf("  \"G_ns\": %.2f,\n", g_ns);
    std::printf("  \"Conv_k64_s64_ns\": %.2f,\n", conv_ns);
    std::printf("  \"gf128_mul_latency_ns\": %.3f,\n", gf_lat);
    std::printf("  \"gf128_mul_throughput_ns\": %.3f,\n", gf_thr);
    std::printf("  \"gf128_mul_portable_ns\": %.2f,\n", gf_port);
    std::printf("  \"x25519_scalarmult_us\": %.2f,\n", x_ns / 1000.0);
    std::printf("  \"sha256_bulk_MBps\": %.1f,\n", sha_mbs);
    std::printf("  \"sha256_32B_ns\": %.1f,\n", sha_small);
    std::printf("  \"sink\": \"%016llx\"\n", (unsigned long long)g_sink);
    std::printf("}\n");
    return 0;
}
