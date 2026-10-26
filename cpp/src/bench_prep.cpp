// bench_prep.cpp -- benchmark of the Boolean preprocessing (couplet/prep.hpp;
// DESIGN.md Sections 7.1 - 7.5): authenticated AND triples from base OT,
// correlated OT extension, leaky ANDs, bucketing and the consistency check.
//
// For every requested number ell of FINAL AND triples (default 2^14, 2^17,
// 2^20, 2^22; sigma = 40, bucket size B = bucket_size(ell, 40)) and every
// requested number of worker threads per party, the two parties run one
// fresh session each repetition:  RealPrep::setup()  (base OT) followed by one
// RealPrep::and_triples(ell).  Both parties are threads of this process and
// talk over an in-process LoopbackChannel, so the times are wall-clock times
// with both parties on one machine and no network delay.
//
// Reported per row (JSON):
//   seconds            and_triples wall time (the slower party), every
//                      repetition and their median;
//   triples_per_second ell / median seconds (base OT excluded), and the same
//                      with the base OT of the session included;
//   bytes_per_triple   payload bytes of both directions per final triple,
//                      without and with the base OT; the model 96 B;
//   phases             for baseot, cotext, abit, laand, bucket, check: bytes
//                      sent by P_0 and by P_1, rounds, and the seconds each
//                      party spent in the phase (median repetition).  A
//                      party's phase time includes waiting for the peer.
// Byte and round counts are deterministic; they are checked to be the same in
// every repetition.  After the timed part of every run the output triples of
// both parties are verified (c = a AND b and the MAC invariant on a, b, c).
//
// Usage: bench_prep [output.json] [--quick] [--reps N] [--threads LIST] [--sizes LIST]
//   output.json   default <exe dir>/../../experiments/raw/bench_prep_cpp.json
//   --quick       ell in {2^10, 2^14}, one repetition (smoke run; do not report)
//   --reps N      repetitions per row (default 3)
//   --threads L   comma-separated worker threads per party (default 1,4)
//   --sizes L     comma-separated log2(ell) (default 14,17,20,22)
// The JSON is also printed to stdout.  Exit code 0 iff every run verified.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "couplet/prep.hpp"
#include "couplet/twoparty.hpp"

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
using namespace couplet::prep;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string cpu_brand() {
#if COUPLET_BENCH_CPUID
    unsigned regs[12];
    std::memset(regs, 0, sizeof regs);
    for (unsigned i = 0; i < 3; i++) {
#if COUPLET_BENCH_CPUID == 1
        int r[4];
        __cpuid(r, (int)(0x80000002u + i));
        for (int k = 0; k < 4; k++) regs[4 * i + (unsigned)k] = (unsigned)r[k];
#else
        unsigned a = 0, b = 0, c = 0, d = 0;
        __get_cpuid(0x80000002u + i, &a, &b, &c, &d);
        regs[4 * i] = a;
        regs[4 * i + 1] = b;
        regs[4 * i + 2] = c;
        regs[4 * i + 3] = d;
#endif
    }
    char name[49];
    std::memcpy(name, regs, 48);
    name[48] = 0;
    std::string s(name);
    while (!s.empty() && s.back() == ' ') s.pop_back();
    size_t first = 0;
    while (first < s.size() && s[first] == ' ') first++;
    return s.substr(first);
#else
    return "unknown";
#endif
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

static std::string fmt(const char* f, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, f, v);
    return buf;
}

struct PartyRun {
    Triples t;
    gf128 delta;
    PrepStats st;
    unsigned bucket = 0;
    double and_seconds = 0.0;      // and_triples only
    double session_seconds = 0.0;  // setup + and_triples
};

struct Run {
    bool ok = false;
    std::string error;
    PrepStats st[2];
    unsigned bucket = 0;
    double and_seconds = 0.0, session_seconds = 0.0;
};

static bool verify(const Triples& t0, const Triples& t1, const gf128& delta) {
    const Shares* s0[3] = {&t0.a, &t0.b, &t0.c};
    const Shares* s1[3] = {&t1.a, &t1.b, &t1.c};
    const size_t n = t0.size();
    for (int k = 0; k < 3; k++)
        if (s0[k]->bits.size() != n || s1[k]->bits.size() != n || s0[k]->macs.size() != n || s1[k]->macs.size() != n) return false;
    uint64_t bad = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned v[3];
        for (int k = 0; k < 3; k++) {
            v[k] = (unsigned)(s0[k]->bits[i] ^ s1[k]->bits[i]);
            const gf128 m = s0[k]->macs[i] ^ s1[k]->macs[i] ^ gf_and_bit(delta, v[k]);
            bad |= m.lo | m.hi | (uint64_t)(v[k] >> 1);
        }
        bad |= (uint64_t)(v[2] ^ (v[0] & v[1]));
    }
    return bad == 0;
}

static Run run_once(size_t ell, unsigned threads, int rep) {
    const std::string sid_s = "bench-prep/" + std::to_string(ell) + "/" + std::to_string(threads) + "/" + std::to_string(rep);
    const bytes sid(sid_s.begin(), sid_s.end());
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            PartyRun o;
            RealPrep prep(role, ch, sid, SIGMA_DEFAULT);
            prep.set_threads(threads);
            o.delta = prep.delta();
            const double t0 = now_s();
            prep.setup();
            const double t1 = now_s();
            o.t = prep.and_triples(ell);
            const double t2 = now_s();
            o.and_seconds = t2 - t1;
            o.session_seconds = t2 - t0;
            o.st = prep.stats();
            o.bucket = prep.last_bucket_size();
            return o;
        };
    };
    auto f0 = party(0);
    auto f1 = party(1);
    auto res = run_two_party(f0, f1);
    Run r;
    if (!res.first.ok || !res.second.ok) {
        r.error = "P0: " + res.first.error + " / P1: " + res.second.error;
        return r;
    }
    const PartyRun& a = res.first.value;
    const PartyRun& b = res.second.value;
    r.st[0] = a.st;
    r.st[1] = b.st;
    r.bucket = a.bucket;
    r.and_seconds = std::max(a.and_seconds, b.and_seconds);
    r.session_seconds = std::max(a.session_seconds, b.session_seconds);
    if (a.bucket != b.bucket || a.t.size() != ell) {
        r.error = "inconsistent outputs";
        return r;
    }
    if (!verify(a.t, b.t, a.delta ^ b.delta)) {
        r.error = "output triples failed verification";
        return r;
    }
    r.ok = true;
    return r;
}

static std::vector<unsigned> parse_list(const char* s) {
    std::vector<unsigned> v;
    const char* p = s;
    while (*p) {
        char* end = nullptr;
        const unsigned long x = std::strtoul(p, &end, 10);
        if (end == p) break;
        v.push_back((unsigned)x);
        p = (*end == ',') ? end + 1 : end;
    }
    return v;
}

int main(int argc, char** argv) {
    bool quick = false;
    int reps = 3;
    std::vector<unsigned> thread_list = {1, 4};
    std::vector<unsigned> sizes = {14, 17, 20, 22};
    std::string out_path;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) thread_list = parse_list(argv[++i]);
        else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) sizes = parse_list(argv[++i]);
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "usage: bench_prep [output.json] [--quick] [--reps N] [--threads LIST] [--sizes LIST]\n");
            return 2;
        } else out_path = argv[i];
    }
    if (quick) {
        sizes = {10, 14};
        reps = 1;
    }
    if (reps < 1 || thread_list.empty() || sizes.empty()) {
        std::fprintf(stderr, "bench_prep: nothing to run\n");
        return 2;
    }
    for (unsigned e : sizes) {
        if (e < 1 || e > 26) {
            std::fprintf(stderr, "bench_prep: log2(ell) must be in 1..26\n");
            return 2;
        }
    }
    for (unsigned& t : thread_list)
        if (t < 1) t = 1;
    if (out_path.empty()) {
        const std::string exe = argc > 0 && argv[0] ? argv[0] : "";
        const size_t pos = exe.find_last_of("/\\");
        const std::string dir = pos == std::string::npos ? std::string() : exe.substr(0, pos + 1);
        out_path = dir + "../../experiments/raw/bench_prep_cpp.json";
    }
    {
        // Fail early if the output cannot be written; do not leave an empty file behind.
        const bool existed = std::ifstream(out_path.c_str(), std::ios::binary).good();
        const bool writable = std::ofstream(out_path.c_str(), std::ios::binary | std::ios::app).good();
        if (!writable) {
            std::fprintf(stderr, "bench_prep: cannot open %s for writing\n", out_path.c_str());
            return 2;
        }
        if (!existed) std::remove(out_path.c_str());
    }

    const double t_all = now_s();
    bool all_ok = true;
    std::string rows;
    for (unsigned threads : thread_list) {
        for (unsigned e : sizes) {
            const size_t ell = (size_t)1 << e;
            std::vector<Run> runs;
            bool row_ok = true, counts_equal = true;
            std::string row_error;
            for (int rep = 0; rep < reps; rep++) {
                Run r = run_once(ell, threads, rep);
                std::fprintf(stderr, "threads=%u ell=2^%u rep=%d: %s, and_triples %.3f s, session %.3f s\n", threads, e, rep,
                             r.ok ? "ok" : r.error.c_str(), r.and_seconds, r.session_seconds);
                if (!r.ok) {
                    row_ok = false;
                    row_error = r.error;
                    break;
                }
                if (!runs.empty()) {
                    for (int role = 0; role < 2; role++)
                        for (int p = 0; p < PH_COUNT; p++)
                            counts_equal = counts_equal && r.st[role][p].bytes_sent == runs[0].st[role][p].bytes_sent &&
                                           r.st[role][p].rounds == runs[0].st[role][p].rounds;
                }
                runs.push_back(std::move(r));
            }
            all_ok = all_ok && row_ok && counts_equal;
            if (!rows.empty()) rows += ",\n";
            rows += "    {\"threads_per_party\": " + std::to_string(threads) + ", \"log2_ell\": " + std::to_string(e) +
                    ", \"ell\": " + std::to_string(ell) + ", \"sigma\": " + std::to_string(SIGMA_DEFAULT);
            if (!row_ok) {
                for (char& c : row_error)
                    if (c == '"' || c == '\\' || (unsigned char)c < 0x20) c = ' ';
                rows += ", \"verified\": false, \"error\": \"" + row_error + "\"}";
                continue;
            }
            // the median repetition by and_triples time
            std::vector<size_t> order(runs.size());
            for (size_t i = 0; i < order.size(); i++) order[i] = i;
            std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return runs[x].and_seconds < runs[y].and_seconds; });
            const Run& med = runs[order[(order.size() - 1) / 2]];
            const unsigned bucket = med.bucket;
            uint64_t bytes_prep = 0, bytes_all = 0, rounds_prep = 0, rounds_all = 0;
            for (int p = 0; p < PH_COUNT; p++) {
                const uint64_t b = med.st[0][p].bytes_sent + med.st[1][p].bytes_sent;
                bytes_all += b;
                rounds_all += med.st[0][p].rounds;
                if (p != PH_BASEOT) {
                    bytes_prep += b;
                    rounds_prep += med.st[0][p].rounds;
                }
            }
            double sec_min = runs[0].and_seconds, sec_max = runs[0].and_seconds;
            std::string sec_list, session_list;
            for (size_t i = 0; i < runs.size(); i++) {
                sec_min = std::min(sec_min, runs[i].and_seconds);
                sec_max = std::max(sec_max, runs[i].and_seconds);
                sec_list += (i ? ", " : "") + fmt("%.4f", runs[i].and_seconds);
                session_list += (i ? ", " : "") + fmt("%.4f", runs[i].session_seconds);
            }
            rows += ", \"B\": " + std::to_string(bucket) + ", \"leaky_triples\": " + std::to_string(ell * bucket) +
                    ", \"cot_rows_per_direction\": " + std::to_string(2 * ell * bucket + PAD) +
                    ", \"repetitions\": " + std::to_string(runs.size()) + ", \"verified\": true" +
                    ", \"counts_equal_across_repetitions\": " + (counts_equal ? "true" : "false") + ",\n" +
                    "     \"seconds\": {\"and_triples_median\": " + fmt("%.4f", med.and_seconds) + ", \"and_triples_min\": " +
                    fmt("%.4f", sec_min) + ", \"and_triples_max\": " + fmt("%.4f", sec_max) + ", \"and_triples_all\": [" + sec_list +
                    "], \"session_with_base_ot_median_run\": " + fmt("%.4f", med.session_seconds) +
                    ", \"session_with_base_ot_all\": [" + session_list + "]},\n" +
                    "     \"triples_per_second\": " + fmt("%.0f", (double)ell / med.and_seconds) +
                    ", \"triples_per_second_with_base_ot\": " + fmt("%.0f", (double)ell / med.session_seconds) +
                    ", \"us_per_triple\": " + fmt("%.4f", med.and_seconds / (double)ell * 1e6) + ",\n" +
                    "     \"bytes_per_triple\": " + fmt("%.3f", (double)bytes_prep / (double)ell) +
                    ", \"bytes_per_triple_with_base_ot\": " + fmt("%.3f", (double)bytes_all / (double)ell) +
                    ", \"model_bytes_per_triple_96B\": " + std::to_string(96 * bucket) +
                    ", \"bytes_total\": " + std::to_string(bytes_prep) + ", \"bytes_total_with_base_ot\": " + std::to_string(bytes_all) +
                    ", \"rounds\": " + std::to_string(rounds_prep) + ", \"rounds_with_base_ot\": " + std::to_string(rounds_all) + ",\n" +
                    "     \"phases\": {";
            for (int p = 0; p < PH_COUNT; p++) {
                rows += std::string(p ? ",\n                " : "") + "\"" + phase_name(p) + "\": {\"bytes_sent_p0\": " +
                        std::to_string(med.st[0][p].bytes_sent) + ", \"bytes_sent_p1\": " + std::to_string(med.st[1][p].bytes_sent) +
                        ", \"bytes\": " + std::to_string(med.st[0][p].bytes_sent + med.st[1][p].bytes_sent) +
                        ", \"rounds\": " + std::to_string(med.st[0][p].rounds) + ", \"seconds_p0\": " +
                        fmt("%.4f", med.st[0][p].seconds) + ", \"seconds_p1\": " + fmt("%.4f", med.st[1][p].seconds) + "}";
            }
            rows += "}}";
        }
    }

    char stamp[32];
    {
        const std::time_t now = std::time(nullptr);
        std::tm tmv;
#if defined(_WIN32)
        gmtime_s(&tmv, &now);
#else
        gmtime_r(&now, &tmv);
#endif
        std::strftime(stamp, sizeof stamp, "%Y-%m-%dT%H:%M:%SZ", &tmv);
    }
    prep::detail::FastSha256 probe;
    std::string json = "{\n";
    json += "  \"benchmark\": \"bench_prep\",\n";
    json += "  \"description\": \"Boolean preprocessing (DESIGN.md sections 7.1-7.5): authenticated AND triples, "
            "RealPrep.setup() + one and_triples(ell) per session, sigma = 40\",\n";
    json += "  \"design_version\": \"3.1\",\n";
    json += std::string("  \"timestamp_utc\": \"") + stamp + "\",\n";
    json += "  \"compiler\": \"" + compiler_name() + "\",\n";
    json += std::string("  \"avx2\": ") + (COUPLET_HAVE_AVX2 ? "true" : "false") + ",\n";
    json += std::string("  \"pclmul\": ") + (COUPLET_HAVE_PCLMUL ? "true" : "false") + ",\n";
    json += std::string("  \"sha_extensions\": ") + (probe.accelerated() ? "true" : "false") + ",\n";
    json += "  \"cpu\": \"" + cpu_brand() + "\",\n";
    json += "  \"hardware_threads\": " + std::to_string(std::thread::hardware_concurrency()) + ",\n";
    json += std::string("  \"quick_mode\": ") + (quick ? "true" : "false") + ",\n";
    json += "  \"transport\": \"in-process LoopbackChannel, two party threads in one process, no delay\",\n";
    json += "  \"method\": \"wall clock (steady_clock); seconds = and_triples time of the slower party; median over the "
            "repetitions; phase seconds are per party and include waiting for the peer; bytes are payload bytes "
            "(bytes_sent of P_0 plus bytes_sent of P_1); threads_per_party = worker threads of the bulk kernels of each "
            "party (1 = the party thread only)\",\n";
    json += "  \"extension_chunk_rows\": " + std::to_string(DEFAULT_BATCH_ROWS) + ",\n";
    json += "  \"kappa\": " + std::to_string(KAPPA) + ",\n";
    json += "  \"rows\": [\n" + rows + "\n  ],\n";
    json += std::string("  \"all_runs_verified\": ") + (all_ok ? "true" : "false") + ",\n";
    json += "  \"total_seconds\": " + fmt("%.1f", now_s() - t_all) + "\n";
    json += "}\n";

    std::fputs(json.c_str(), stdout);
    std::ofstream f(out_path.c_str(), std::ios::binary | std::ios::trunc);
    f << json;
    f.close();
    if (!f) {
        std::fprintf(stderr, "bench_prep: writing %s failed\n", out_path.c_str());
        return 2;
    }
    std::fprintf(stderr, "bench_prep: wrote %s\n", out_path.c_str());
    return all_ok ? 0 : 1;
}
