// bench_ppml.cpp -- local times of the plane queries of the Couplet layer
// (DESIGN_M3.md section 3, Carve; planes.hpp): for n in {8, 12, 14, 16, 20}
// and 1 and 8 threads, the full-domain evaluation of one plane (payload-1 DPF
// key, k = s = 64, values and tags), the prefix tables of one plane, one table
// read (the inner product of a public table of 2^n entries with both
// components of the plane) and one locator read (two prefix differences), and
// the evaluation of a layer of 8 planes (evaluate_planes).  Nothing here sends
// a message; one party's work is measured (the other party's is the same).
//
// Usage: bench_ppml [--out FILE.json] [--quick] [--ns 8,12,14,16,20] [--threads 1,8]
// Default output: artifact/experiments/raw/bench_ppml_local.json.  Every time
// is the median over repetitions (at least 3, or more until 0.3 s of samples
// have been taken); the table and locator reads are timed over batches of
// calls.  Before timing, each configuration checks the reconstruction of a
// table read and a locator read against the other party's plane.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "couplet/circuit.hpp"
#include "couplet/fss.hpp"
#include "couplet/planes.hpp"
#include "couplet/rng.hpp"
#include "couplet/u128.hpp"

using namespace couplet;
typedef std::chrono::steady_clock bclock;

static double now_s() { return std::chrono::duration<double>(bclock::now().time_since_epoch()).count(); }

// Median of the times of fn() (seconds per call of fn), at least `min_reps`
// repetitions and until `budget` seconds of samples were taken.
static double median_time(const std::function<void()>& fn, int min_reps, double budget, int* reps_out) {
    std::vector<double> t;
    double spent = 0;
    while ((int)t.size() < min_reps || (spent < budget && t.size() < 1000)) {
        const double a = now_s();
        fn();
        const double d = now_s() - a;
        t.push_back(d);
        spent += d;
    }
    std::sort(t.begin(), t.end());
    if (reps_out) *reps_out = (int)t.size();
    return t[t.size() / 2];
}

static std::vector<unsigned> parse_list(const std::string& s) {
    std::vector<unsigned> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back((unsigned)std::atoi(item.c_str()));
    return v;
}

static std::string num(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%.9g", x);
    return b;
}

static std::string compiler_name() {
#if defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#else
    return "unknown";
#endif
}

static std::string utc_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm_utc;
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S+00:00", &tm_utc);
    return buf;
}

int main(int argc, char** argv) {
    std::string out = circuit::dirname(argv[0]) + "/../../experiments/raw/bench_ppml_local.json";
    std::vector<unsigned> ns = {8, 12, 14, 16, 20}, ths = {1, 8};
    bool quick = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--quick") quick = true;
        else if (a == "--ns" && i + 1 < argc) ns = parse_list(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) ths = parse_list(argv[++i]);
        else {
            std::printf("usage: bench_ppml [--out FILE.json] [--quick] [--ns 8,12,14,16,20] [--threads 1,8]\n");
            return 2;
        }
    }
    const unsigned k = 64, s = 64, w = k + s;
    const double budget = quick ? 0.05 : 0.3;
    CtrPrgRng rng(0xbe9c4);
    const u128 delta = fss::make_delta(rng.u64(), rng.u64());
    bool all_ok = true;
    std::string rows;
    std::printf("%4s %7s %14s %14s %14s %14s %14s\n", "n", "threads", "plane eval s", "prefix s", "table read s", "locator s",
                "8 planes s");
    for (unsigned n : ns) {
        const uint64_t N = (uint64_t)1 << n;
        const uint64_t alpha = rng.u64() & (N - 1);
        const std::pair<fss::DpfKey, fss::DpfKey> keys = fss::dpf_gen(alpha, 1, delta, n, k, s, rng);
        std::vector<int64_t> T(N);
        for (uint64_t z = 0; z < N; z++) T[z] = (int64_t)(rng.u64() >> 8) - ((int64_t)1 << 55);
        const planes::EmbeddedTable ET(T, w);
        std::vector<fss::DpfKey> layer_keys;
        for (int i = 0; i < 8; i++) layer_keys.push_back(fss::dpf_gen(rng.u64() & (N - 1), 1, delta, n, k, s, rng).first);
        std::vector<const fss::DpfKey*> layer_ptrs;
        for (const fss::DpfKey& kk : layer_keys) layer_ptrs.push_back(&kk);
        for (unsigned th : ths) {
            // correctness spot check
            const planes::Plane p0(0, keys.first, th), p1(1, keys.second, th);
            const uint64_t xh = rng.u64() & (N - 1), l = rng.u64() & (N - 1), u = rng.u64() & (N - 1);
            const spdz2k::AShare t0 = p0.table(ET, xh, 0, th), t1 = p1.table(ET, xh, 0, th);
            const std::pair<u128, u128> rt = spdz2k::reconstruct(t0, t1);
            const std::pair<u128, u128> rl = spdz2k::reconstruct(p0.locator(l, u), p1.locator(l, u));
            const bool in = l <= u ? (alpha >= l && alpha <= u) : (alpha >= l || alpha <= u);
            const bool ok = rt.first == ET.e[(xh + N - alpha) & (N - 1)] && rt.second == delta * ET.e[(xh + N - alpha) & (N - 1)] &&
                            rl.first == u128(in ? 1 : 0);
            all_ok = all_ok && ok;
            // plane evaluation (full domain, values and tags), without prefix tables
            int reps_eval = 0, reps_pre = 0, reps_tab = 0, reps_loc = 0, reps_layer = 0;
            planes::Plane scratch;
            const double t_eval = median_time([&] { scratch.evaluate(0, keys.first, th, false); }, 3, budget, &reps_eval);
            const double t_pre = median_time([&] { scratch.build_prefix(th); }, 3, budget, &reps_pre);
            // one table read (batches of calls for the small domains)
            const int tb = N <= 4096 ? 64 : 1;
            volatile uint64_t sink = 0;
            const double t_tab = median_time(
                                     [&] {
                                         for (int i = 0; i < tb; i++) sink = sink + u128_lo(scratch.table(ET, (xh + (uint64_t)i) & (N - 1), 0, th).value);
                                     },
                                     5, budget, &reps_tab) /
                                 tb;
            const int lb = 100000;
            const double t_loc = median_time(
                                     [&] {
                                         for (int i = 0; i < lb; i++) sink = sink + u128_lo(scratch.locator((l + (uint64_t)i) & (N - 1), u).value);
                                     },
                                     3, budget, &reps_loc) /
                                 lb;
            const double t_layer = median_time([&] { planes::evaluate_planes(0, layer_ptrs, th, false); }, 3, budget, &reps_layer);
            std::printf("%4u %7u %14.6g %14.6g %14.6g %14.6g %14.6g%s\n", n, th, t_eval, t_pre, t_tab, t_loc, t_layer, ok ? "" : "  CHECK FAILED");
            std::fflush(stdout);
            if (!rows.empty()) rows += ",\n";
            rows += "  {\"n\": " + std::to_string(n) + ", \"threads\": " + std::to_string(th) + ", \"plane_eval_s\": " + num(t_eval) +
                    ", \"plane_eval_reps\": " + std::to_string(reps_eval) + ", \"prefix_tables_s\": " + num(t_pre) +
                    ", \"prefix_tables_reps\": " + std::to_string(reps_pre) + ", \"table_read_s\": " + num(t_tab) +
                    ", \"table_read_reps\": " + std::to_string(reps_tab * tb) + ", \"locator_read_s\": " + num(t_loc) +
                    ", \"locator_read_reps\": " + std::to_string(reps_loc * lb) + ", \"layer8_eval_s\": " + num(t_layer) +
                    ", \"layer8_eval_reps\": " + std::to_string(reps_layer) + ", \"plane_bytes\": " +
                    std::to_string((uint64_t)4 * N * sizeof(u128)) + ", \"check_ok\": " + (ok ? "true" : "false") + "}";
            (void)sink;
        }
    }
    std::string doc = "{\n \"schema\": \"couplet.bench_ppml_local/1\",\n \"implementation\": \"C++ (couplet, native)\",\n";
    doc += " \"compiler\": \"" + compiler_name() + "\",\n \"timestamp_utc\": \"" + utc_now() + "\",\n";
    doc += " \"hardware_threads\": " + std::to_string(std::thread::hardware_concurrency()) + ",\n";
    doc += " \"params\": {\"k\": 64, \"s\": 64, \"w\": 128, \"quick\": " + std::string(quick ? "true" : "false") +
           ", \"min_sample_seconds\": " + num(budget) + "},\n";
    doc += " \"note\": \"one party, local work only. plane_eval_s: full-domain evaluation of one payload-1 DPF key (values and "
           "tags, 2^n points each, no prefix tables); prefix_tables_s: both prefix tables of one evaluated plane; table_read_s: "
           "one table query (inner product of a public table of 2^n entries with both components); locator_read_s: one locator "
           "query (prefix differences); layer8_eval_s: evaluate_planes over 8 keys; plane_bytes: memory of one plane with its "
           "prefix tables. Medians over the stated repetitions.\",\n";
    doc += " \"rows\": [\n" + rows + "\n ],\n";
    doc += " \"checks\": {\"reconstruction_ok\": " + std::string(all_ok ? "true" : "false") + "}\n}\n";
    std::ofstream f(out, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", out.c_str());
        return 1;
    }
    f << doc;
    std::printf("wrote %s (reconstruction checks %s)\n", out.c_str(), all_ok ? "ok" : "FAILED");
    return all_ok ? 0 : 1;
}
