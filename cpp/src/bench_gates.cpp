// bench_gates.cpp -- local evaluation (Apply) of the catalogue gates
// (couplet/gates.hpp; DESIGN.md section 6) in a single process.
//
// For each gate the benchmark deals one-time material with gates::deal_gate
// (the trusted-dealer reference of Gate.Gen, NOT a protocol: only the LOCAL
// work of Apply is measured here), fixes a random input x and the masked
// input x_hat = (x + r) mod 2^n, and times party 0's Apply.  A gate is used
// once, so every timed call runs on a FRESH evaluator (no cached key
// evaluation); building the evaluators is not timed.
//
//   table       n in {8, 12, 16, 20}, k = 64: T[x] for a public table of 2^n
//               64-bit records.  seconds = one Apply = expansion of the DPF
//               key on the whole domain (EvalAll, both planes) plus the
//               convolution sum_u T[(x_hat - u) mod 2^n] y(u) on both planes.
//               The two parts are also timed separately (expand_*,
//               convolution_*: the convolution alone on an evaluator whose
//               expansion is cached).
//   zero        n in {8, 12, 16, 20}: [x = 0], one DPF point evaluation.
//   comparison  n in {8, 12, 16, 20}: [x < c], c = floor(2^n / 3): two DCF
//               point evaluations.
//   interval    n in {8, 12, 16, 20}: [p <= x <= q], p = floor(2^n / 5),
//               q = floor(2^n / 3) - 1: two DCF point evaluations.
//   relu        n = k in {8, 16, 32, 64} (relu_large: four DCF point
//               evaluations; correct modulo 2^k).
//
// Before timing, both parties' outputs are reconstructed and compared with
// the plaintext gate on the value plane and the MAC plane (unreduced Delta).
// Each row gives the material size of one party ("bytes"), the median,
// minimum and maximum over the repetitions; a repetition of a fast gate
// averages over `iters` fresh evaluators.  One thread.
//
// Usage: bench_gates [output.json] [--quick] [--pin CPU]
// Without a path the result goes to <exe dir>/../../experiments/raw/
// bench_gates_cpp.json.  The JSON is also printed to stdout.  --pin CPU binds
// the thread to one logical processor (hybrid processors).  Exit code 0 iff
// every gate reconstructed correctly.
//
// Build (a target of build_msvc.bat and build.sh; by hand, from artifact/cpp):
//   cl /nologo /O2 /std:c++17 /EHsc /arch:AVX2 /W4 /DNDEBUG /Iinclude src\bench_gates.cpp
//      /Fobuild\ /Febuild\bench_gates.exe ws2_32.lib bcrypt.lib
//   g++ -O2 -std=c++17 -march=native -Iinclude src/bench_gates.cpp -o build/bench_gates -lpthread
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
#include "couplet/gates.hpp"
#include "couplet/rng.hpp"
#include "couplet/u128.hpp"

#if defined(__linux__)
#include <sched.h>
#endif

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

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint64_t g_sink = 0;

static bool pin_thread(int cpu) {
#if defined(_WIN32)
    if (cpu < 0 || cpu >= (int)(8 * sizeof(DWORD_PTR))) return false;
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu) != 0;
#elif defined(__linux__)
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof set, &set) == 0;
#else
    (void)cpu;
    return false;
#endif
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
    std::string out;
    for (char ch : std::string(buf))  // printable ASCII only, single spaces
        if (ch >= 32 && ch < 127 && ch != '"' && ch != '\\' && !(ch == ' ' && (out.empty() || out.back() == ' '))) out.push_back(ch);
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out.empty() ? std::string("unknown") : out;
#else
    return "unknown";
#endif
}

struct Stat {
    double median = 0, min = 0, max = 0;
};
static Stat stat_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    Stat st;
    st.min = v.front();
    st.max = v.back();
    st.median = (v.size() % 2) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    return st;
}

static std::string g_json;
static void emit(const char* fmt, ...) {
    char buf[2048];
    std::va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    g_json += buf;
}

// One benchmarked gate: material of both parties, the masked input and the plaintext value.
struct Case {
    gates::GateSpec spec;
    std::pair<protocol::GateMaterial, protocol::GateMaterial> mats;
    u128 delta = u128(0);
    uint64_t x = 0, x_hat = 0;
};

static Case make_case(const gates::GateSpec& spec, CtrPrgRng& rng) {
    Case c;
    c.spec = spec;
    const unsigned s = spec.s;
    const uint64_t smask = s >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << s) - 1);
    c.delta = fss::make_delta(rng.u64() & smask, rng.u64() & smask);  // unreduced sum of two s-bit shares
    const protocol::GateDescriptor desc = spec.descriptor();
    c.mats = gates::deal_gate(desc, c.delta, rng);
    const uint64_t dm = gates::domain_mask(spec.n);
    const uint64_t r = u128_lo((c.mats.first.rp_share + c.mats.second.rp_share) & u128_mask(desc.rp_bits())) & dm;
    c.x = rng.u64() & dm;
    c.x_hat = (c.x + r) & dm;
    return c;
}

// Both parties' Apply on fresh evaluators against the plaintext gate, on both planes.
static bool reconstructs(const Case& c) {
    gates::GateEval e0(c.mats.first), e1(c.mats.second);
    e0.set_delta_share(0);  // not needed by the benchmarked gates (no public constant is added)
    e1.set_delta_share(0);
    const std::vector<gates::AShare> a = c.spec.evaluate(e0, c.x_hat), b = c.spec.evaluate(e1, c.x_hat);
    const std::vector<uint64_t> want = c.spec.plain(c.x);
    if (a.size() != want.size() || b.size() != want.size()) return false;
    const unsigned w = c.spec.k + c.spec.s;
    bool ok = true;
    for (size_t i = 0; i < want.size(); i++) {
        const u128 v = (a[i].value + b[i].value) & u128_mask(w), t = (a[i].mac + b[i].mac) & u128_mask(w);
        ok = ok && (c.spec.exact() ? v == u128(want[i]) : (v & u128_mask(c.spec.k)) == u128(want[i])) &&
             t == ((c.delta * v) & u128_mask(w)) && a[i].exact == c.spec.exact();
    }
    return ok;
}

// Seconds of ONE Apply of party 0 on a fresh evaluator; each repetition averages over `iters` evaluators.
static Stat time_apply(const Case& c, size_t iters, int reps) {
    std::vector<double> secs;
    for (int r = 0; r < reps + 1; r++) {  // the first repetition is a warm-up
        std::vector<gates::GateEval> evs;
        evs.reserve(iters);
        for (size_t i = 0; i < iters; i++) evs.emplace_back(c.mats.first);
        uint64_t acc = 0;
        const double t0 = now_s();
        for (size_t i = 0; i < iters; i++) {
            const std::vector<gates::AShare> y = c.spec.evaluate(evs[i], c.x_hat);
            acc ^= u128_lo(y[0].value) ^ u128_lo(y[0].mac);
        }
        const double t1 = now_s();
        g_sink ^= acc;
        if (r) secs.push_back((t1 - t0) / (double)iters);
    }
    return stat_of(secs);
}

// Table gate only: the expansion (EvalAll) and the convolution timed separately.
static void time_table_parts(const Case& c, size_t iters, int reps, Stat* expand, Stat* conv) {
    std::vector<double> se, sc;
    for (int r = 0; r < reps + 1; r++) {
        std::vector<gates::GateEval> evs;
        evs.reserve(iters);
        for (size_t i = 0; i < iters; i++) evs.emplace_back(c.mats.first);
        uint64_t acc = 0;
        const double t0 = now_s();
        for (size_t i = 0; i < iters; i++) acc ^= u128_lo(evs[i].dpf_y()[0]) ^ u128_lo(evs[i].dpf_tau()[0]);
        const double t1 = now_s();
        for (size_t i = 0; i < iters; i++) {
            const gates::AShare y = gates::table(evs[i], c.x_hat, c.spec.table);  // the expansion is cached
            acc ^= u128_lo(y.value) ^ u128_lo(y.mac);
        }
        const double t2 = now_s();
        g_sink ^= acc;
        if (r) {
            se.push_back((t1 - t0) / (double)iters);
            sc.push_back((t2 - t1) / (double)iters);
        }
    }
    *expand = stat_of(se);
    *conv = stat_of(sc);
}

static bool g_first_row = true;
static bool g_all_ok = true;

static void row(const char* gate, const gates::GateSpec& spec, const char* what, CtrPrgRng& rng, bool quick) {
    const Case c = make_case(spec, rng);
    const bool ok = reconstructs(c);
    g_all_ok = g_all_ok && ok;
    const bool is_table = spec.kind == gates::GateSpec::TABLE;
    // Enough evaluators per repetition for a few milliseconds of work; fewer for the table gate.
    size_t iters = is_table ? (spec.n <= 8 ? 400 : spec.n <= 12 ? 40 : spec.n <= 16 ? 4 : 1) : 2000;
    int reps = is_table && spec.n >= 20 ? 9 : 15;
    if (quick) {
        iters = std::max<size_t>(1, iters / 10);
        reps = 3;
    }
    const Stat st = time_apply(c, iters, reps);
    const size_t bytes = c.mats.first.to_bytes().size();
    emit(g_first_row ? "\n" : ",\n");
    g_first_row = false;
    emit("    {\"gate\": \"%s\", \"n\": %u, \"k\": %u, \"s\": %u, \"threads\": 1, \"iters\": %zu, \"reps\": %d, "
         "\"seconds_median\": %.9f, \"seconds_min\": %.9f, \"seconds_max\": %.9f, \"bytes\": %zu, \"auth_input\": %s, "
         "\"dcf_keys\": %zu, \"dpf_keys\": %d, \"exact\": %s, \"reconstruction_verified\": %s",
         gate, spec.n, spec.k, spec.s, iters, reps, st.median, st.min, st.max, bytes, spec.auth_input ? "true" : "false",
         c.mats.first.dcf_keys.size(), c.mats.first.desc.dpf_one ? 1 : 0, spec.exact() ? "true" : "false", ok ? "true" : "false");
    if (is_table) {
        Stat ex, cv;
        time_table_parts(c, iters, reps, &ex, &cv);
        const double leaves = (double)((uint64_t)1 << spec.n);
        emit(", \"expand_seconds_median\": %.9f, \"expand_seconds_min\": %.9f, \"convolution_seconds_median\": %.9f, "
             "\"convolution_seconds_min\": %.9f, \"convolution_ns_per_entry\": %.3f",
             ex.median, ex.min, cv.median, cv.min, cv.median * 1e9 / leaves);
        std::fprintf(stderr, "%-10s n=%2u k=%2u: apply %10.3f us (expand %10.3f us, convolution %10.3f us), %zu bytes\n", gate,
                     spec.n, spec.k, st.median * 1e6, ex.median * 1e6, cv.median * 1e6, bytes);
    } else {
        std::fprintf(stderr, "%-10s n=%2u k=%2u: apply %10.3f us, %zu bytes\n", gate, spec.n, spec.k, st.median * 1e6, bytes);
    }
    emit(", \"what\": \"%s\"}", what);
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
        out_path = (pos == std::string::npos ? std::string() : exe.substr(0, pos + 1)) + "../../experiments/raw/bench_gates_cpp.json";
    }
    if (pin_cpu >= 0 && !pin_thread(pin_cpu)) {
        std::fprintf(stderr, "bench_gates: cannot bind the thread to logical processor %d\n", pin_cpu);
        return 2;
    }
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
    emit("  \"benchmark\": \"bench_gates\",\n");
    emit("  \"description\": \"Local evaluation (Apply) of the catalogue gates of DESIGN.md section 6 on dealer-made one-time "
         "material; single process, no protocol message\",\n");
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
    emit("  \"hardware_threads\": %u,\n", std::thread::hardware_concurrency());
    emit("  \"quick_mode\": %s,\n", quick ? "true" : "false");
    if (pin_cpu >= 0) emit("  \"pinned_logical_cpu\": %d,\n", pin_cpu);
    else emit("  \"pinned_logical_cpu\": null,\n");
    emit("  \"gates_method\": {\n");
    emit("    \"clock\": \"std::chrono::steady_clock, wall time\",\n");
    emit("    \"material\": \"gates::deal_gate (trusted-dealer reference of Gate.Gen, not a protocol); Delta = Delta_0 + Delta_1, "
         "two uniform s-bit shares, not reduced\",\n");
    emit("    \"timed\": \"party 0, one Apply on a FRESH evaluator (a gate is used once; nothing is cached); a repetition "
         "averages over iters evaluators that are built before the clock starts; one untimed warm-up repetition; one thread\",\n");
    emit("    \"statistic\": \"seconds_median / _min / _max over the repetitions of a row, in seconds per Apply\",\n");
    emit("    \"table\": \"seconds = DPF expansion on the whole domain (EvalAll, both planes) plus the convolution with the "
         "public table on both planes; expand_* and convolution_* are the two parts timed separately\",\n");
    emit("    \"bytes\": \"encoded gate material of one party (GateMaterial::to_bytes)\"\n");
    emit("  },\n");
    emit("  \"gates\": [");

    CtrPrgRng rng((uint64_t)0x6a7e5be7c4ull);  // seeded test generator: material, inputs and the random table
    const unsigned s = 64;
    for (unsigned n : {8u, 12u, 16u, 20u}) {
        std::vector<uint64_t> T((size_t)1 << n);
        for (uint64_t& t : T) t = rng.u64();
        row("table", gates::GateSpec::table_gate(n, 64, s, T, false), "T[x], public table of 2^n 64-bit records, shared input", rng,
            quick);
    }
    for (unsigned n : {8u, 12u, 16u, 20u}) row("zero", gates::GateSpec::zero_test(n, 64, s, false), "[x = 0], one DPF point evaluation", rng, quick);
    for (unsigned n : {8u, 12u, 16u, 20u})
        row("comparison", gates::GateSpec::comparison(n, 64, s, u128(((uint64_t)1 << n) / 3), false),
            "[x < c], c = floor(2^n / 3), two DCF point evaluations", rng, quick);
    for (unsigned n : {8u, 12u, 16u, 20u}) {
        const uint64_t N = (uint64_t)1 << n;
        row("interval", gates::GateSpec::interval(n, 64, s, N / 5, N / 3 - 1, false),
            "[p <= x <= q], p = floor(2^n / 5), q = floor(2^n / 3) - 1, two DCF point evaluations", rng, quick);
    }
    for (unsigned k : {8u, 16u, 32u, 64u})
        row("relu", gates::GateSpec::relu_large(k, s), "ReLU on k-bit two's complement, n = k (relu_large), four DCF point evaluations", rng,
            quick);
    emit("\n  ],\n");
    emit("  \"all_gates_verified\": %s,\n", g_all_ok ? "true" : "false");
    emit("  \"gates_total_seconds\": %.1f,\n", now_s() - start);
    emit("  \"gates_sink\": \"%016llx\"\n", (unsigned long long)g_sink);
    emit("}\n");

    {
        std::ofstream f(out_path.c_str(), std::ios::binary | std::ios::trunc);
        f.write(g_json.data(), (std::streamsize)g_json.size());
        f.close();
        if (!f) {
            std::fprintf(stderr, "bench_gates: cannot write %s\n", out_path.c_str());
            return 2;
        }
    }
    std::fputs(g_json.c_str(), stdout);
    std::fprintf(stderr, "bench_gates: wrote %s\n", out_path.c_str());
    return g_all_ok ? 0 : 1;
}
