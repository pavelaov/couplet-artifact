// bench_fss.cpp -- local-evaluation benchmarks of the two-plane DPF and DCF
// (couplet/fss.hpp).  Writes one JSON document:
//   eval       ns per dpf_eval / dcf_eval (party 0, random points) for
//              n in {8, 12, 16, 20, 24}
//   gen        plaintext ("dealer-style") Gen time for the same n: the cost
//              of running DPF.Gen / DCF.Gen in the clear, i.e. what a
//              trusted dealer would pay
//   key_bytes  encoded key sizes for the same n
//   eval_all   EvalAll time and throughput (leaves per second) for
//              n in {12, 16, 20, 24}, one thread, output buffers allocated
//              and touched before timing
//   eval_all_threads
//              the same call for n = 24 on several thread counts
// each for (k, s) in {(64, 64), (32, 64)}.
//
// Every figure is wall-clock time (std::chrono::steady_clock): the median
// over the repetitions of a row, with the minimum next to it.  Keys, points
// and roots come from a seeded test generator; its cost inside the Gen loop
// is a few nanoseconds per key pair.
//
// Usage: bench_fss [output.json] [--quick] [--pin CPU]
// Without a path the result goes to <exe dir>/../../experiments/raw/
// bench_fss_cpp.json.  The JSON is also printed to stdout.  --quick shortens
// every row (for smoke runs; do not report those numbers).  --pin CPU binds
// the measuring thread to one logical processor for the single-thread rows
// (useful on hybrid processors, where the scheduler may otherwise move the
// thread between performance and efficiency cores); the binding is removed
// for the multi-thread rows.  The choice is recorded in the output.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "couplet/fss.hpp"
#include "couplet/u128.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define COUPLET_BENCH_CPUID 1
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#define COUPLET_BENCH_CPUID 2
#else
#define COUPLET_BENCH_CPUID 0
#endif

using namespace couplet;

// ------------------------------------------------------------ helpers
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t g_sink = 0;

// Bind the calling thread to one logical processor / restore the previous
// binding.  pin_thread returns false where this is not supported.
#if defined(_WIN32)
static DWORD_PTR g_old_affinity = 0;
static bool pin_thread(int cpu) {
    if (cpu < 0 || cpu >= (int)(8 * sizeof(DWORD_PTR))) return false;
    const DWORD_PTR old = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
    if (!old) return false;
    g_old_affinity = old;
    return true;
}
static void unpin_thread() {
    if (g_old_affinity) SetThreadAffinityMask(GetCurrentThread(), g_old_affinity);
    g_old_affinity = 0;
}
#elif defined(__linux__)
static cpu_set_t g_old_affinity;
static bool g_have_old_affinity = false;
static bool pin_thread(int cpu) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    if (sched_getaffinity(0, sizeof g_old_affinity, &g_old_affinity) != 0) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) return false;
    g_have_old_affinity = true;
    return true;
}
static void unpin_thread() {
    if (g_have_old_affinity) sched_setaffinity(0, sizeof g_old_affinity, &g_old_affinity);
    g_have_old_affinity = false;
}
#else
static bool pin_thread(int) { return false; }
static void unpin_thread() {}
#endif

// xorshift64*: seeded test generator (not a cryptographic generator).
struct TestRng {
    uint64_t s;
    explicit TestRng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    uint64_t bits64(unsigned bits) { return bits >= 64 ? next() : (next() & (((uint64_t)1 << bits) - 1)); }
    fss::seed_t seed() {
        fss::seed_t r;
        const uint64_t a = next(), b = next();
        ascon::store64le(r.data(), a);
        ascon::store64le(r.data() + 8, b);
        return r;
    }
};

struct Stat {
    double median, min;
};
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stat st;
    st.min = v.front();
    st.median = (v.size() % 2) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    return st;
}

static std::string cpu_brand() {
#if COUPLET_BENCH_CPUID
    unsigned regs[12];
    std::memset(regs, 0, sizeof regs);
    for (unsigned leaf = 0; leaf < 3; leaf++) {
#if COUPLET_BENCH_CPUID == 1
        int r[4];
        __cpuid(r, (int)(0x80000002u + leaf));
        for (int i = 0; i < 4; i++) regs[4 * leaf + i] = (unsigned)r[i];
#else
        unsigned a = 0, b = 0, c = 0, d = 0;
        __get_cpuid(0x80000002u + leaf, &a, &b, &c, &d);
        regs[4 * leaf] = a; regs[4 * leaf + 1] = b; regs[4 * leaf + 2] = c; regs[4 * leaf + 3] = d;
#endif
    }
    char buf[49];
    std::memcpy(buf, regs, 48);
    buf[48] = 0;
    std::string s(buf);
    std::string out;
    for (char ch : s)  // printable ASCII only, single spaces
        if (ch >= 32 && ch < 127 && ch != '"' && ch != '\\' && !(ch == ' ' && (out.empty() || out.back() == ' '))) out.push_back(ch);
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out.empty() ? std::string("unknown") : out;
#else
    return "unknown";
#endif
}

static std::string g_json;
static void emit(const char* fmt, ...) {
    char buf[1024];
    std::va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    g_json += buf;
}

// ------------------------------------------------------------ the two families
struct Dpf {
    typedef fss::DpfKey Key;
    static const char* name() { return "DPF"; }
    static std::pair<Key, Key> gen(uint64_t a, uint64_t b, u128 d, unsigned n, unsigned k, unsigned s, const fss::seed_t& r0,
                                   const fss::seed_t& r1) {
        return fss::dpf_gen(a, b, d, n, k, s, r0, r1);
    }
    static fss::Share eval(const Key& key, uint64_t x) { return fss::dpf_eval(key, x); }
    static void eval_all(const Key& key, u128* y, u128* t, unsigned threads) { fss::dpf_eval_all(key, y, t, threads); }
    static bool ind(uint64_t x, uint64_t alpha) { return x == alpha; }
    static size_t key_size(unsigned n, unsigned k, unsigned s) { return fss::dpf_key_size(n, k, s); }
};
struct Dcf {
    typedef fss::DcfKey Key;
    static const char* name() { return "DCF"; }
    static std::pair<Key, Key> gen(uint64_t a, uint64_t b, u128 d, unsigned n, unsigned k, unsigned s, const fss::seed_t& r0,
                                   const fss::seed_t& r1) {
        return fss::dcf_gen(a, b, d, n, k, s, r0, r1);
    }
    static fss::Share eval(const Key& key, uint64_t x) { return fss::dcf_eval(key, x); }
    static void eval_all(const Key& key, u128* y, u128* t, unsigned threads) { fss::dcf_eval_all(key, y, t, threads); }
    static bool ind(uint64_t x, uint64_t alpha) { return x < alpha; }
    static size_t key_size(unsigned n, unsigned k, unsigned s) { return fss::dcf_key_size(n, k, s); }
};

template <class F>
static std::pair<typename F::Key, typename F::Key> random_keys(TestRng& rng, unsigned n, unsigned k, unsigned s,
                                                               uint64_t* alpha_out = nullptr, uint64_t* beta_out = nullptr,
                                                               u128* delta_out = nullptr) {
    const uint64_t alpha = rng.bits64(n), beta = rng.bits64(k);
    const u128 delta = fss::make_delta(rng.bits64(s), rng.bits64(s));  // unreduced sum of two s-bit shares
    const fss::seed_t r0 = rng.seed(), r1 = rng.seed();
    if (alpha_out) *alpha_out = alpha;
    if (beta_out) *beta_out = beta;
    if (delta_out) *delta_out = delta;
    return F::gen(alpha, beta, delta, n, k, s, r0, r1);
}

// ------------------------------------------------------------ kernels
// Point evaluation: back-to-back calls on 16 keys and 1024 random points.
template <class F>
static Stat bench_eval(unsigned n, unsigned k, unsigned s, uint64_t iters, int reps) {
    TestRng rng(0xE7A1ull * (n + 1) + k);
    std::vector<typename F::Key> keys;
    for (int i = 0; i < 16; i++) keys.push_back(random_keys<F>(rng, n, k, s).first);
    std::vector<uint64_t> xs(1024);
    for (auto& x : xs) x = rng.bits64(n);
    std::vector<double> ns;
    for (int r = 0; r < reps + 1; r++) {  // the first repetition is a warm-up
        uint64_t acc = 0;
        const double t0 = now_s();
        for (uint64_t i = 0; i < iters; i++) {
            const fss::Share sh = F::eval(keys[i & 15], xs[i & 1023]);
            acc ^= u128_lo(sh.y) ^ u128_lo(sh.tau);
        }
        const double t1 = now_s();
        g_sink ^= acc;
        if (r) ns.push_back((t1 - t0) * 1e9 / (double)iters);
    }
    return stat_of(ns);
}

// Plaintext Gen with fresh inputs and roots per call.
template <class F>
static Stat bench_gen(unsigned n, unsigned k, unsigned s, uint64_t iters, int reps) {
    TestRng rng(0x6E4ull * (n + 1) + k);
    std::vector<double> ns;
    for (int r = 0; r < reps + 1; r++) {
        uint64_t acc = 0;
        const double t0 = now_s();
        for (uint64_t i = 0; i < iters; i++) {
            const std::pair<typename F::Key, typename F::Key> kp = random_keys<F>(rng, n, k, s);
            acc ^= u128_lo(kp.first.cw_out_v) ^ u128_lo(kp.second.cw_out_t);
        }
        const double t1 = now_s();
        g_sink ^= acc;
        if (r) ns.push_back((t1 - t0) * 1e9 / (double)iters);
    }
    return stat_of(ns);
}

// EvalAll of party 0 into preallocated buffers.  Before timing, both parties
// are evaluated once and the reconstruction is verified on the whole domain,
// so that the timed code is known to be correct on this key.
template <class F>
static Stat bench_eval_all(unsigned n, unsigned k, unsigned s, int reps, unsigned threads, bool& verified) {
    TestRng rng(0xA11ull * (n + 1) + k + 7 * threads);
    uint64_t alpha = 0, beta = 0;
    u128 delta = u128(0);
    const std::pair<typename F::Key, typename F::Key> kp = random_keys<F>(rng, n, k, s, &alpha, &beta, &delta);
    const size_t size = (size_t)1 << n;
    const u128 mask = u128_mask(k + s);
    const u128 beta_t = (delta * u128(beta)) & mask;
    std::vector<u128> y0(size), t0(size);
    {
        std::vector<u128> y1(size), t1(size);
        F::eval_all(kp.first, y0.data(), t0.data(), threads);
        F::eval_all(kp.second, y1.data(), t1.data(), threads);
        verified = true;
        for (size_t x = 0; verified && x < size; x++) {
            const bool ind = F::ind(x, alpha);
            verified = ((y0[x] + y1[x]) & mask) == (ind ? u128(beta) : u128(0)) &&
                       ((t0[x] + t1[x]) & mask) == (ind ? beta_t : u128(0));
        }
    }
    std::vector<double> secs;
    for (int r = 0; r < reps; r++) {
        const double a = now_s();
        F::eval_all(kp.first, y0.data(), t0.data(), threads);
        const double b = now_s();
        g_sink ^= u128_lo(y0[size - 1]) ^ u128_lo(t0[size / 2]);
        secs.push_back(b - a);
    }
    return stat_of(secs);
}

// ------------------------------------------------------------ rows
static bool g_first_row = true;
static void row_sep() {
    emit(g_first_row ? "\n" : ",\n");
    g_first_row = false;
}

template <class F>
static void rows_eval(unsigned k, unsigned s, bool quick) {
    const unsigned ns[] = {8, 12, 16, 20, 24};
    for (unsigned n : ns) {
        const uint64_t iters = (quick ? 200000ull : 4000000ull) / n;
        const int reps = quick ? 3 : 9;
        const Stat st = bench_eval<F>(n, k, s, iters, reps);
        row_sep();
        emit("    {\"family\": \"%s\", \"n\": %u, \"k\": %u, \"s\": %u, \"iters\": %llu, \"reps\": %d, "
             "\"ns_per_eval_median\": %.1f, \"ns_per_eval_min\": %.1f}",
             F::name(), n, k, s, (unsigned long long)iters, reps, st.median, st.min);
        std::fprintf(stderr, "eval      %s n=%2u k=%u: %8.1f ns\n", F::name(), n, k, st.median);
    }
}

template <class F>
static void rows_gen(unsigned k, unsigned s, bool quick) {
    const unsigned ns[] = {8, 12, 16, 20, 24};
    for (unsigned n : ns) {
        const uint64_t iters = (quick ? 100000ull : 1000000ull) / n;
        const int reps = quick ? 3 : 9;
        const Stat st = bench_gen<F>(n, k, s, iters, reps);
        row_sep();
        emit("    {\"family\": \"%s\", \"n\": %u, \"k\": %u, \"s\": %u, \"iters\": %llu, \"reps\": %d, "
             "\"us_per_gen_median\": %.3f, \"us_per_gen_min\": %.3f}",
             F::name(), n, k, s, (unsigned long long)iters, reps, st.median / 1000.0, st.min / 1000.0);
        std::fprintf(stderr, "gen       %s n=%2u k=%u: %8.3f us\n", F::name(), n, k, st.median / 1000.0);
    }
}

template <class F>
static void rows_key_bytes(unsigned k, unsigned s, bool& sizes_ok) {
    const unsigned ns[] = {8, 12, 16, 20, 24};
    TestRng rng(0xB17E5ull + k);
    for (unsigned n : ns) {
        const std::pair<typename F::Key, typename F::Key> kp = random_keys<F>(rng, n, k, s);
        const size_t bytes = kp.first.to_bytes().size();
        sizes_ok = sizes_ok && bytes == kp.second.to_bytes().size() && bytes == F::key_size(n, k, s);
        row_sep();
        emit("    {\"family\": \"%s\", \"n\": %u, \"k\": %u, \"s\": %u, \"bytes\": %zu}", F::name(), n, k, s, bytes);
    }
}

template <class F>
static void row_eval_all(unsigned n, unsigned k, unsigned s, unsigned threads, bool quick, bool& all_verified) {
    int reps = n <= 12 ? 201 : n <= 16 ? 31 : n <= 20 ? 9 : 5;
    if (quick) reps = n <= 16 ? 5 : 2;
    bool verified = false;
    const Stat st = bench_eval_all<F>(n, k, s, reps, threads, verified);
    all_verified = all_verified && verified;
    const double leaves = (double)((uint64_t)1 << n);
    row_sep();
    emit("    {\"family\": \"%s\", \"n\": %u, \"k\": %u, \"s\": %u, \"threads\": %u, \"reps\": %d, "
         "\"seconds_median\": %.6f, \"seconds_min\": %.6f, \"leaves_per_second\": %.0f, \"ns_per_leaf\": %.2f, "
         "\"reconstruction_verified\": %s}",
         F::name(), n, k, s, threads, reps, st.median, st.min, leaves / st.median, st.median * 1e9 / leaves,
         verified ? "true" : "false");
    std::fprintf(stderr, "eval_all  %s n=%2u k=%u threads=%2u: %9.4f ms, %6.2f M leaves/s\n", F::name(), n, k, threads,
                 st.median * 1e3, leaves / st.median / 1e6);
}

int main(int argc, char** argv) {
    bool quick = false;
    int pin_cpu = -1;
    std::string out_path;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        else if (std::strcmp(argv[i], "--pin") == 0 && i + 1 < argc) pin_cpu = std::atoi(argv[++i]);
        else out_path = argv[i];
    }
    if (out_path.empty()) {
        std::string exe = argc > 0 && argv[0] ? argv[0] : "";
        const size_t pos = exe.find_last_of("/\\");
        const std::string dir = pos == std::string::npos ? std::string() : exe.substr(0, pos + 1);
        out_path = dir + "../../experiments/raw/bench_fss_cpp.json";
    }
    // Fail early if the output cannot be written.  An existing file is kept
    // until the run has finished; no empty file is left behind.
    {
        const bool existed = std::ifstream(out_path.c_str(), std::ios::binary).good();
        const bool writable = std::ofstream(out_path.c_str(), std::ios::binary | std::ios::app).good();
        if (!writable) {
            std::fprintf(stderr, "bench_fss: cannot open %s for writing\n", out_path.c_str());
            return 2;
        }
        if (!existed) std::remove(out_path.c_str());
    }

    if (pin_cpu >= 0 && !pin_thread(pin_cpu)) {
        std::fprintf(stderr, "bench_fss: cannot bind the thread to logical processor %d\n", pin_cpu);
        return 2;
    }

    const unsigned params[2][2] = {{64, 64}, {32, 64}};
    const unsigned hw = std::thread::hardware_concurrency();
    const double start = now_s();
    char stamp[32] = "unknown";
    {
        const std::time_t now = std::time(nullptr);
        std::tm tmv;
#if defined(_WIN32)
        if (gmtime_s(&tmv, &now) == 0) std::strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%SZ", &tmv);
#else
        if (gmtime_r(&now, &tmv)) std::strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%SZ", &tmv);
#endif
    }

    emit("{\n");
    emit("  \"benchmark\": \"bench_fss\",\n");
    emit("  \"description\": \"Two-plane DPF/DCF (DESIGN.md section 3): point Eval, plaintext Gen, key sizes, EvalAll\",\n");
    emit("  \"design_version\": \"3.1\",\n");
    emit("  \"timestamp_utc\": \"%s\",\n", stamp);
#if defined(_MSC_VER)
    emit("  \"compiler\": \"MSVC %d\",\n", _MSC_FULL_VER);
#elif defined(__clang__)
    emit("  \"compiler\": \"clang %d.%d.%d\",\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    emit("  \"compiler\": \"gcc %d.%d.%d\",\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    emit("  \"compiler\": \"unknown\",\n");
#endif
    emit("  \"avx2\": %s,\n", COUPLET_HAVE_AVX2 ? "true" : "false");
    emit("  \"cpu\": \"%s\",\n", cpu_brand().c_str());
    emit("  \"hardware_threads\": %u,\n", hw);
    emit("  \"quick_mode\": %s,\n", quick ? "true" : "false");
    if (pin_cpu >= 0) emit("  \"pinned_logical_cpu\": %d,\n", pin_cpu);
    else emit("  \"pinned_logical_cpu\": null,\n");
    emit("  \"method\": {\n");
    emit("    \"clock\": \"std::chrono::steady_clock, wall time\",\n");
    emit("    \"statistic\": \"median over the repetitions of a row; the minimum is given next to it\",\n");
    emit("    \"eval\": \"party 0, back-to-back calls cycling over 16 random keys and 1024 random points, one thread; "
         "one untimed warm-up repetition\",\n");
    emit("    \"gen\": \"plaintext Gen (what a trusted dealer runs) with fresh alpha, beta, Delta and roots per call "
         "from a seeded test generator, including the allocation of both key objects, one thread\",\n");
    emit("    \"eval_all\": \"party 0, output buffers (two arrays of 2^n 128-bit ring elements) allocated and touched "
         "before timing; subtree depth %u\",\n", fss::EVAL_ALL_CHUNK_BITS);
    emit("    \"pinning\": \"the single-thread rows (eval, gen, eval_all) run on the logical processor "
         "pinned_logical_cpu when it is not null; eval_all_threads rows are never pinned\",\n");
    emit("    \"delta\": \"Delta = Delta_0 + Delta_1, two uniform s-bit shares, not reduced\"\n");
    emit("  },\n");
    emit("  \"params\": [{\"k\": 64, \"s\": 64}, {\"k\": 32, \"s\": 64}],\n");

    emit("  \"eval\": [");
    g_first_row = true;
    for (auto& p : params) {
        rows_eval<Dpf>(p[0], p[1], quick);
        rows_eval<Dcf>(p[0], p[1], quick);
    }
    emit("\n  ],\n");

    emit("  \"gen\": [");
    g_first_row = true;
    for (auto& p : params) {
        rows_gen<Dpf>(p[0], p[1], quick);
        rows_gen<Dcf>(p[0], p[1], quick);
    }
    emit("\n  ],\n");

    bool sizes_ok = true;
    emit("  \"key_bytes\": [");
    g_first_row = true;
    for (auto& p : params) {
        rows_key_bytes<Dpf>(p[0], p[1], sizes_ok);
        rows_key_bytes<Dcf>(p[0], p[1], sizes_ok);
    }
    emit("\n  ],\n");

    bool all_verified = true;
    emit("  \"eval_all\": [");
    g_first_row = true;
    const unsigned all_ns[] = {12, 16, 20, 24};
    for (auto& p : params) {
        for (unsigned n : all_ns) row_eval_all<Dpf>(n, p[0], p[1], 1, quick, all_verified);
        for (unsigned n : all_ns) row_eval_all<Dcf>(n, p[0], p[1], 1, quick, all_verified);
    }
    emit("\n  ],\n");

    emit("  \"eval_all_threads\": [");
    g_first_row = true;
    unpin_thread();
    {
        const unsigned counts[] = {2, 4, 8, 16, 32};
        for (unsigned th : counts) {
            if (hw && th > hw) continue;
            row_eval_all<Dpf>(24, 64, 64, th, quick, all_verified);
            row_eval_all<Dcf>(24, 64, 64, th, quick, all_verified);
        }
    }
    emit("\n  ],\n");

    emit("  \"key_sizes_match_formula\": %s,\n", sizes_ok ? "true" : "false");
    emit("  \"all_eval_all_outputs_verified\": %s,\n", all_verified ? "true" : "false");
    emit("  \"total_seconds\": %.1f,\n", now_s() - start);
    emit("  \"sink\": \"%016llx\"\n", (unsigned long long)g_sink);
    emit("}\n");

    {
        std::ofstream f(out_path.c_str(), std::ios::binary | std::ios::trunc);
        f.write(g_json.data(), (std::streamsize)g_json.size());
        f.close();
        if (!f) {
            std::fprintf(stderr, "bench_fss: cannot write %s\n", out_path.c_str());
            return 2;
        }
    }
    std::fputs(g_json.c_str(), stdout);
    std::fprintf(stderr, "bench_fss: wrote %s\n", out_path.c_str());
    return (sizes_ok && all_verified) ? 0 : 1;
}
