// bench_baselines.cpp -- benchmark of the two evaluation baselines
// (couplet/baselines.hpp; DESIGN.md Section 9, Q1) on this machine:
//
//   leaky_zhang    OUR RE-IMPLEMENTATION of the one-bit-leaky, actively secure
//                  DPF protocol Pi_DPF of Zhang et al. (2024, Figure
//                  5) on the authenticated-bit layer of this artifact;
//   semihonest_ds  semi-honest Doerner-shelat-style generation on the same
//                  tree (no MACs, no check, unchecked COT).
//
// Neither is the original implementation: the hash is truncated Ascon-p12
// (not fixed-key AES), the authenticated bits / COT rows come from our
// IKNP-style extension (not Ferret), kappa = 128; see the header comment of
// baselines.hpp.  Both output shares of the WHOLE truth table (2^n entries)
// with a GF(2^128) payload.
//
// For every requested n (default 8, 12, 16, 20, 24), both baselines and both
// transports, the two parties run one fresh session per repetition: base OT,
// then ONE key.
//
//   loopback_two_threads  both parties are threads of this process and talk
//                         over an in-process LoopbackChannel;
//   tcp_two_processes     this program starts itself twice (--party 0 / 1);
//                         the two processes talk over TCP on 127.0.0.1
//                         (4-byte length framing, TCP_NODELAY).
//
// One thread per party unless --threads is given (worker threads of the local
// tree passes).  Two harness steps surround the key generation; neither is
// part of the protocols, neither is timed or counted: after the base OT the
// parties exchange one byte (a barrier: both start the key generation at the
// same moment, so that no party books the peer's remaining base-OT time under
// its first key phase), and after the measured part of every run they execute
// the TEST-ONLY verification step of baselines::testonly (key shares and
// inputs are revealed, one digest of the outputs is compared).  Every key
// allocates and first-touches its buffers inside the timed region.
//
// Reported per n, baseline and transport (JSON):
//   phases   bytes sent by P_0 and by P_1, rounds, and seconds: for each
//            repetition the larger of the two parties' wall-clock times in the
//            phase (a party's phase time includes waiting for the peer), then
//            median, minimum and maximum over the repetitions;
//   per_key_without_base_ot, protocol_messages (the phases zhang-* / sh-*),
//   authenticated_bits (cotext + abit; leaky baseline only), base_ot:
//            bytes of both parties, rounds, seconds (median, min, max);
//   runs     every repetition: seconds per phase and party, verification.
// Byte and round counts are deterministic; they are checked to be equal in
// every repetition and at both parties and to equal the formulas of
// baselines.hpp (the ones the Python tests assert).
//
// Usage: bench_baselines [output.json] [--quick] [--reps N] [--n LIST]
//                        [--threads T] [--pin A,B] [--no-loopback] [--no-tcp]
//   output.json   default <exe dir>/../../experiments/raw/baselines_cpp.json
//   --quick       n in {8, 12}, two repetitions (smoke run; do not report)
//   --reps N      repetitions per row (default 5)
//   --n LIST      comma-separated domain sizes in bits (default 8,12,16,20,24)
//   --threads T   worker threads per party for the local passes (default 1)
//   --pin A,B     bind the thread of party 0 to logical processor A and that
//                 of party 1 to B (default: no binding)
// The JSON is also printed to stdout.  Exit code 0 iff every run verified and
// every count equals its formula.
//
// Internal: bench_baselines --party ROLE --proto leaky|semihonest --n N
//           --port P --tag TEXT --record FILE [--threads T] [--pin CPU]
// runs ONE party of one session over TCP and writes its record to FILE.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "couplet/baselines.hpp"
#include "couplet/prep.hpp"
#include "couplet/twoparty.hpp"

#if defined(_WIN32)
#include <process.h>
#else
#include <sched.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
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
using namespace couplet::baselines;

// ------------------------------------------------------------ small helpers
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
static std::string sec(double v) { return fmt("%.6f", v); }
static std::string num(uint64_t v) { return std::to_string(v); }

static std::string json_escape(std::string s) {
    for (char& c : s)
        if (c == '"' || c == '\\' || (unsigned char)c < 0x20) c = ' ';
    return s;
}

// Binds the calling thread to one logical processor for a scope (cpu < 0: no
// binding); ok() is false where this is not supported or fails.
class PinScope {
public:
    explicit PinScope(int cpu) : ok_(true), bound_(false) {
        if (cpu < 0) return;
#if defined(_WIN32)
        if (cpu >= (int)(8 * sizeof(DWORD_PTR))) {
            ok_ = false;
            return;
        }
        old_ = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
        ok_ = bound_ = old_ != 0;
#elif defined(__linux__)
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        ok_ = bound_ = sched_getaffinity(0, sizeof old_, &old_) == 0 && sched_setaffinity(0, sizeof set, &set) == 0;
#else
        ok_ = false;
#endif
    }
    ~PinScope() {
        if (!bound_) return;
#if defined(_WIN32)
        SetThreadAffinityMask(GetCurrentThread(), old_);
#elif defined(__linux__)
        sched_setaffinity(0, sizeof old_, &old_);
#endif
    }
    PinScope(const PinScope&) = delete;
    PinScope& operator=(const PinScope&) = delete;
    bool ok() const { return ok_; }

private:
    bool ok_, bound_;
#if defined(_WIN32)
    DWORD_PTR old_ = 0;
#elif defined(__linux__)
    cpu_set_t old_;
#endif
};

// ------------------------------------------------------------ one party of one session
struct PhaseRow {
    std::string name;
    uint64_t bytes_sent = 0, bytes_recv = 0, msgs_sent = 0, rounds = 0;
    double seconds = 0.0;
};

struct PartyRecord {
    bool ok = false;        // the session completed
    bool verified = false;  // the TEST-ONLY verification step succeeded
    std::string error;
    double setup_seconds = 0.0;  // base OT
    double key_seconds = 0.0;    // everything of the key after the base OT
    uint64_t total_bytes_sent = 0, total_bytes_recv = 0, total_rounds = 0;  // channel totals of the protocol
    std::vector<PhaseRow> phases;
};

static PhaseRow row(const char* name, const prep::PhaseStat& s) {
    PhaseRow r;
    r.name = name;
    r.bytes_sent = s.bytes_sent;
    r.bytes_recv = s.bytes_recv;
    r.msgs_sent = s.msgs_sent;
    r.rounds = s.rounds;
    r.seconds = s.seconds;
    return r;
}

// Harness step, NOT part of either protocol and not counted: one exchange of a
// single byte after the base OT.  The two parties finish their base OT at
// different moments (its last step is local); without the barrier the party
// that is done first would book the peer's remaining base-OT time under the
// first phase of its key generation.
struct SyncBarrier {
    uint64_t bytes_sent = 0, bytes_recv = 0, rounds = 0;
    void run(Channel& ch) {
        const ChannelStats before = ch.stats();
        ch.exchange(bytes(1, 0));
        const ChannelStats after = ch.stats();
        bytes_sent = after.bytes_sent - before.bytes_sent;
        bytes_recv = after.bytes_recv - before.bytes_recv;
        rounds = after.rounds - before.rounds;
    }
};

// Base OT and one key for party `role` on the endpoint ch; random input shares.
static PartyRecord run_party_session(const std::string& proto, int role, Channel& ch, unsigned n, const bytes& sid,
                                     unsigned threads) {
    PartyRecord rec;
    SystemRng sys;
    const uint64_t a_share = sys.u64() >> (64 - n);
    const gf128 b_share = sys.gf();
    SyncBarrier barrier;
    if (proto == "leaky") {
        prep::RealPrep prep(role, ch, sid);
        prep.set_threads(threads);
        Meter meter(ch);
        LeakyOptions opt;
        opt.threads = threads;
        const double t0 = now_s();
        prep.setup();
        const double t1 = now_s();
        barrier.run(ch);
        const double t1s = now_s();
        const LeakyDpfOutput out = run_pi_dpf(ch, prep, sid, n, a_share, b_share, opt, &meter);
        const double t2 = now_s();
        const ChannelStats cs = ch.stats();
        const prep::PrepStats ps = prep.stats();
        const PhaseStats st = meter.stats();
        rec.setup_seconds = t1 - t0;
        rec.key_seconds = t2 - t1s;
        rec.total_bytes_sent = cs.bytes_sent - barrier.bytes_sent;
        rec.total_bytes_recv = cs.bytes_recv - barrier.bytes_recv;
        rec.total_rounds = cs.rounds - barrier.rounds;
        rec.phases.push_back(row("baseot", ps[prep::PH_BASEOT]));
        rec.phases.push_back(row("cotext", ps[prep::PH_COTEXT]));
        rec.phases.push_back(row("abit", ps[prep::PH_ABIT]));
        for (int p = BP_ZH_INPUT; p <= BP_ZH_CHECK; p++) rec.phases.push_back(row(phase_name(p), st[p]));
        rec.verified = testonly::verify_leaky(ch, role, out, prep.delta(), a_share, b_share);
    } else {
        SemiHonestSession session(role, ch, sid);
        session.set_threads(threads);
        const double t0 = now_s();
        session.setup();
        const double t1 = now_s();
        barrier.run(ch);
        const double t1s = now_s();
        const SemiHonestDpfOutput out = session.dpf_gen(n, a_share, b_share);
        const double t2 = now_s();
        const ChannelStats cs = ch.stats();
        const PhaseStats st = session.stats();
        rec.setup_seconds = t1 - t0;
        rec.key_seconds = t2 - t1s;
        rec.total_bytes_sent = cs.bytes_sent - barrier.bytes_sent;
        rec.total_bytes_recv = cs.bytes_recv - barrier.bytes_recv;
        rec.total_rounds = cs.rounds - barrier.rounds;
        for (int p = BP_SH_BASEOT; p <= BP_SH_LEAF; p++) rec.phases.push_back(row(phase_name(p), st[p]));
        rec.verified = testonly::verify_semihonest(ch, role, out, session.delta(), a_share, b_share);
    }
    rec.ok = true;
    return rec;
}

// The record as text lines (the child process writes it, the parent reads it).
static std::string record_text(const PartyRecord& r) {
    std::string s = std::string("ok ") + (r.ok ? "1" : "0") + "\nverified " + (r.verified ? "1" : "0") + "\n";
    s += "setup_seconds " + fmt("%.9f", r.setup_seconds) + "\nkey_seconds " + fmt("%.9f", r.key_seconds) + "\n";
    s += "totals " + num(r.total_bytes_sent) + " " + num(r.total_bytes_recv) + " " + num(r.total_rounds) + "\n";
    for (const PhaseRow& p : r.phases)
        s += "phase " + p.name + " " + num(p.bytes_sent) + " " + num(p.bytes_recv) + " " + num(p.msgs_sent) + " " +
             num(p.rounds) + " " + fmt("%.9f", p.seconds) + "\n";
    if (!r.error.empty()) s += "error " + json_escape(r.error) + "\n";
    s += "end\n";
    return s;
}

static PartyRecord parse_record(const std::string& path) {
    PartyRecord r;
    std::ifstream f(path.c_str());
    if (!f) {
        r.error = "no record file";
        return r;
    }
    std::string line;
    bool complete = false;
    while (std::getline(f, line)) {
        std::istringstream in(line);
        std::string key;
        in >> key;
        if (key == "ok") {
            int v = 0;
            in >> v;
            r.ok = v != 0;
        } else if (key == "verified") {
            int v = 0;
            in >> v;
            r.verified = v != 0;
        } else if (key == "setup_seconds") {
            in >> r.setup_seconds;
        } else if (key == "key_seconds") {
            in >> r.key_seconds;
        } else if (key == "totals") {
            in >> r.total_bytes_sent >> r.total_bytes_recv >> r.total_rounds;
        } else if (key == "phase") {
            PhaseRow p;
            in >> p.name >> p.bytes_sent >> p.bytes_recv >> p.msgs_sent >> p.rounds >> p.seconds;
            r.phases.push_back(p);
        } else if (key == "error") {
            std::getline(in, r.error);
        } else if (key == "end") {
            complete = true;
        }
    }
    if (!complete) {
        r.ok = false;
        if (r.error.empty()) r.error = "incomplete record file";
    }
    return r;
}

static bytes make_sid(const std::string& proto, const std::string& tag) {
    const std::string s = "bench-" + proto + "|" + tag;
    return bytes(s.begin(), s.end());
}

// ------------------------------------------------------------ child mode: one party over TCP
static int child_main(const std::string& proto, int role, unsigned n, int port, const std::string& tag,
                      const std::string& record_path, unsigned threads, int pin) {
    PartyRecord rec;
    try {
        const PinScope pinned(pin);
        if (!pinned.ok()) throw std::runtime_error("cannot bind the thread to the requested logical processor");
        std::unique_ptr<TcpChannel> ch = TcpChannel::create(role, "127.0.0.1", (uint16_t)port, 60000);
        rec = run_party_session(proto, role, *ch, n, make_sid(proto, tag), threads);
    } catch (const std::exception& e) {
        rec.ok = false;
        rec.error = e.what();
    }
    std::ofstream f(record_path.c_str(), std::ios::binary | std::ios::trunc);
    f << record_text(rec);
    f.close();
    if (!rec.ok) std::fprintf(stderr, "bench_baselines party %d: %s\n", role, rec.error.c_str());
    return rec.ok && rec.verified ? 0 : 1;
}

// ------------------------------------------------------------ process control
static std::string self_path(const char* argv0) {
#if defined(_WIN32)
    char buf[4096];
    const DWORD len = GetModuleFileNameA(nullptr, buf, (DWORD)sizeof buf);
    if (len > 0 && len < sizeof buf) return std::string(buf, len);
#elif defined(__linux__)
    char buf[4096];
    const ssize_t len = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (len > 0) return std::string(buf, (size_t)len);
#endif
    return argv0 ? argv0 : "";
}

#if defined(_WIN32)
typedef intptr_t child_t;
static const child_t no_child = -1;
static std::string quote_arg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    return "\"" + a + "\"";
}
static child_t start_child(const std::string& exe, const std::vector<std::string>& args) {
    std::vector<std::string> quoted;
    quoted.push_back(quote_arg(exe));
    for (const std::string& a : args) quoted.push_back(quote_arg(a));
    std::vector<const char*> argv;
    for (const std::string& q : quoted) argv.push_back(q.c_str());
    argv.push_back(nullptr);
    return _spawnv(_P_NOWAIT, exe.c_str(), argv.data());
}
// Exit status, or -1 if the child had to be terminated after timeout_ms.
static int wait_child(child_t c, int timeout_ms) {
    const HANDLE h = (HANDLE)c;
    if (WaitForSingleObject(h, (DWORD)timeout_ms) != WAIT_OBJECT_0) {
        TerminateProcess(h, 99);
        WaitForSingleObject(h, 5000);
        CloseHandle(h);
        return -1;
    }
    DWORD code = 1;
    GetExitCodeProcess(h, &code);
    CloseHandle(h);
    return (int)code;
}
#else
typedef pid_t child_t;
static const child_t no_child = -1;
static child_t start_child(const std::string& exe, const std::vector<std::string>& args) {
    std::vector<std::string> all;
    all.push_back(exe);
    all.insert(all.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (std::string& a : all) argv.push_back(&a[0]);
    argv.push_back(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        execv(exe.c_str(), argv.data());
        _exit(127);
    }
    return pid;
}
static int wait_child(child_t c, int timeout_ms) {
    const double deadline = now_s() + timeout_ms / 1000.0;
    for (;;) {
        int status = 0;
        const pid_t r = waitpid(c, &status, WNOHANG);
        if (r == c) return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        if (r < 0) return -1;
        if (now_s() > deadline) {
            kill(c, SIGKILL);
            waitpid(c, &status, 0);
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
#endif

static uint16_t free_port() {
    TcpListener probe("127.0.0.1", 0);
    return probe.port();
}

// ------------------------------------------------------------ one run (both parties)
struct Run {
    PartyRecord p[2];
    bool ok() const { return p[0].ok && p[1].ok && p[0].verified && p[1].verified; }
    std::string error() const {
        if (!p[0].ok || !p[1].ok) return "P0: " + p[0].error + " / P1: " + p[1].error;
        if (!p[0].verified || !p[1].verified) return "the outputs failed the verification step";
        return "";
    }
};

static Run run_loopback(const std::string& proto, unsigned n, int rep, unsigned threads, const int pins[2]) {
    const bytes sid = make_sid(proto, "loopback|" + std::to_string(n) + "|" + std::to_string(rep));
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            const PinScope pinned(pins[role]);
            if (!pinned.ok()) throw std::runtime_error("cannot bind the thread to the requested logical processor");
            return run_party_session(proto, role, ch, n, sid, threads);
        };
    };
    auto f0 = party(0);
    auto f1 = party(1);
    auto res = run_two_party(f0, f1);
    Run r;
    r.p[0] = std::move(res.first.value);
    r.p[1] = std::move(res.second.value);
    if (!res.first.ok) {
        r.p[0].ok = false;
        r.p[0].error = res.first.error;
    }
    if (!res.second.ok) {
        r.p[1].ok = false;
        r.p[1].error = res.second.error;
    }
    return r;
}

static Run run_tcp(const std::string& exe, const std::string& proto, unsigned n, int rep, unsigned threads, const int pins[2],
                   const std::string& record_base) {
    Run r;
    const uint16_t port = free_port();
    const std::string tag = "tcp|" + std::to_string(n) + "|" + std::to_string(rep);
    std::string rec_path[2];
    child_t child[2] = {no_child, no_child};
    for (int role = 0; role < 2; role++) {
        rec_path[role] = record_base + ".p" + std::to_string(role) + ".tmp";
        std::remove(rec_path[role].c_str());
        std::vector<std::string> args = {"--party", std::to_string(role), "--proto", proto, "--n", std::to_string(n),
                                         "--port", std::to_string(port), "--tag", tag, "--record", rec_path[role],
                                         "--threads", std::to_string(threads)};
        if (pins[role] >= 0) {
            args.push_back("--pin");
            args.push_back(std::to_string(pins[role]));
        }
        child[role] = start_child(exe, args);
    }
    int status[2] = {-1, -1};
    for (int role = 0; role < 2; role++) {
        if (child[role] == no_child) {
            r.p[role].error = "the party process could not be started";
            continue;
        }
        status[role] = wait_child(child[role], 600000);
    }
    for (int role = 0; role < 2; role++) {
        if (child[role] != no_child) r.p[role] = parse_record(rec_path[role]);
        if (status[role] != 0) {
            r.p[role].verified = false;
            if (r.p[role].ok && r.p[role].error.empty()) r.p[role].error = "exit status " + std::to_string(status[role]);
            if (status[role] < 0) r.p[role].ok = false;
        }
        std::remove(rec_path[role].c_str());
    }
    return r;
}

// ------------------------------------------------------------ summaries
struct Stat3 {
    double median = 0.0, min = 0.0, max = 0.0;
};
static Stat3 stat3(std::vector<double> v) {
    Stat3 s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    const size_t k = v.size() / 2;
    s.median = (v.size() & 1) ? v[k] : 0.5 * (v[k - 1] + v[k]);
    return s;
}
static std::string stat_json(const Stat3& s) {
    return "\"seconds_median\": " + sec(s.median) + ", \"seconds_min\": " + sec(s.min) + ", \"seconds_max\": " + sec(s.max);
}

// Bytes sent per party and rounds that a phase must show for one session with one key of size n.
static bool expected_counts(const std::string& proto, unsigned n, const std::string& phase, uint64_t& bytes, uint64_t& rounds) {
    if (phase == "baseot") {
        bytes = 12288;
        rounds = 2;
        return true;
    }
    if (proto == "leaky") {
        if (phase == "cotext") {
            bytes = prep::u_message_bytes(abit_demand(n)) + prep::COT_SEED_BYTES + prep::COT_CHECK_BYTES;
            rounds = 3;
            return true;
        }
        if (phase == "abit") {
            bytes = 0;
            rounds = 0;
            return true;
        }
        for (int p = BP_ZH_INPUT; p <= BP_ZH_CHECK; p++) {
            if (phase == phase_name(p)) {
                bytes = leaky_bytes_per_party(n, p);
                rounds = leaky_rounds(n, p);
                return true;
            }
        }
        return false;
    }
    for (int p = BP_SH_COT; p <= BP_SH_LEAF; p++) {
        if (phase == phase_name(p)) {
            bytes = semihonest_bytes_per_party(n, p);
            rounds = semihonest_rounds(n, p);
            return true;
        }
    }
    return false;
}

static bool is_protocol_phase(const std::string& name) { return name.compare(0, 6, "zhang-") == 0 || name.compare(0, 3, "sh-") == 0; }
static bool is_abit_phase(const std::string& name) { return name == "cotext" || name == "abit"; }

struct ModeResult {
    bool ran = false;
    bool verified = true;       // every repetition completed and verified
    bool counts_equal = true;   // the same bytes and rounds in every repetition; rounds equal at both parties
    bool counts_match = true;   // bytes and rounds equal the formulas
    std::string error;
    std::vector<Run> runs;
    double key_median = 0.0;    // for the console table
    std::string json;
};

// Sum of the seconds of the phases selected by `pick`, per party; the larger party.
template <class Pick>
static double group_seconds(const Run& r, Pick pick) {
    double best = 0.0;
    for (int role = 0; role < 2; role++) {
        double s = 0.0;
        for (const PhaseRow& p : r.p[role].phases)
            if (pick(p.name)) s += p.seconds;
        best = std::max(best, s);
    }
    return best;
}

static void summarise(const std::string& proto, unsigned n, ModeResult& m) {
    m.ran = true;
    for (const Run& r : m.runs) {
        if (!r.ok()) {
            m.verified = false;
            if (m.error.empty()) m.error = r.error();
        }
    }
    if (!m.verified || m.runs.empty()) {
        m.counts_equal = m.counts_match = false;
        m.json = "{\"repetitions\": " + num(m.runs.size()) + ", \"all_outputs_verified\": false, \"error\": \"" +
                 json_escape(m.error) + "\"}";
        return;
    }
    const Run& first = m.runs[0];
    const size_t n_phases = first.p[0].phases.size();
    for (const Run& r : m.runs) {
        m.counts_equal = m.counts_equal && r.p[0].phases.size() == n_phases && r.p[1].phases.size() == n_phases;
        for (int role = 0; role < 2; role++) {
            for (size_t k = 0; m.counts_equal && k < n_phases; k++) {
                const PhaseRow& a = r.p[role].phases[k];
                const PhaseRow& b = first.p[role].phases[k];
                m.counts_equal = a.name == b.name && a.bytes_sent == b.bytes_sent && a.rounds == b.rounds &&
                                 a.rounds == r.p[1 - role].phases[k].rounds && a.bytes_sent == r.p[1 - role].phases[k].bytes_recv;
            }
        }
    }
    if (!m.counts_equal) {
        m.counts_match = false;
        m.json = "{\"repetitions\": " + num(m.runs.size()) + ", \"all_outputs_verified\": true, \"error\": \"bytes or rounds "
                 "differ between repetitions or parties\"}";
        return;
    }

    std::string phases;
    uint64_t key_bytes = 0, key_rounds = 0, proto_bytes = 0, proto_rounds = 0, abit_bytes = 0, abit_rounds = 0;
    uint64_t ot_bytes = 0, ot_rounds = 0, sum_bytes[2] = {0, 0}, sum_rounds = 0;
    for (size_t k = 0; k < n_phases; k++) {
        const PhaseRow& a = first.p[0].phases[k];
        const PhaseRow& b = first.p[1].phases[k];
        uint64_t exp_bytes = 0, exp_rounds = 0;
        const bool known = expected_counts(proto, n, a.name, exp_bytes, exp_rounds);
        m.counts_match = m.counts_match && known && a.bytes_sent == exp_bytes && b.bytes_sent == exp_bytes && a.rounds == exp_rounds;
        std::vector<double> secs;
        for (const Run& r : m.runs) secs.push_back(std::max(r.p[0].phases[k].seconds, r.p[1].phases[k].seconds));
        const uint64_t both = a.bytes_sent + b.bytes_sent;
        phases += std::string(k ? ",\n" : "") + "          \"" + a.name + "\": {\"bytes_p0\": " + num(a.bytes_sent) +
                  ", \"bytes_p1\": " + num(b.bytes_sent) + ", \"bytes_both\": " + num(both) + ", \"rounds\": " + num(a.rounds) +
                  ", \"model_bytes_per_party\": " + num(exp_bytes) + ", \"model_rounds\": " + num(exp_rounds) + ", " +
                  stat_json(stat3(secs)) + "}";
        sum_bytes[0] += a.bytes_sent;
        sum_bytes[1] += b.bytes_sent;
        sum_rounds += a.rounds;
        if (a.name == "baseot") {
            ot_bytes += both;
            ot_rounds += a.rounds;
        } else {
            key_bytes += both;
            key_rounds += a.rounds;
        }
        if (is_protocol_phase(a.name)) {
            proto_bytes += both;
            proto_rounds += a.rounds;
        }
        if (is_abit_phase(a.name)) {
            abit_bytes += both;
            abit_rounds += a.rounds;
        }
    }
    // the phases account for all traffic of the protocol
    m.counts_match = m.counts_match && sum_bytes[0] == first.p[0].total_bytes_sent && sum_bytes[1] == first.p[1].total_bytes_sent &&
                     sum_rounds == first.p[0].total_rounds && sum_rounds == first.p[1].total_rounds;

    std::vector<double> key_s, setup_s, proto_s, abit_s;
    std::string runs;
    for (size_t i = 0; i < m.runs.size(); i++) {
        const Run& r = m.runs[i];
        key_s.push_back(std::max(r.p[0].key_seconds, r.p[1].key_seconds));
        setup_s.push_back(std::max(r.p[0].setup_seconds, r.p[1].setup_seconds));
        proto_s.push_back(group_seconds(r, is_protocol_phase));
        abit_s.push_back(group_seconds(r, is_abit_phase));
        std::string per_phase;
        for (size_t k = 0; k < n_phases; k++)
            per_phase += std::string(k ? ", " : "") + "\"" + r.p[0].phases[k].name + "\": [" + sec(r.p[0].phases[k].seconds) + ", " +
                         sec(r.p[1].phases[k].seconds) + "]";
        runs += std::string(i ? ",\n" : "") + "          {\"key_seconds_p0\": " + sec(r.p[0].key_seconds) + ", \"key_seconds_p1\": " +
                sec(r.p[1].key_seconds) + ", \"base_ot_seconds_p0\": " + sec(r.p[0].setup_seconds) + ", \"base_ot_seconds_p1\": " +
                sec(r.p[1].setup_seconds) + ", \"outputs_verified\": true,\n           \"phase_seconds_p0_p1\": {" + per_phase + "}}";
    }
    const Stat3 key_stat = stat3(key_s);
    m.key_median = key_stat.median;
    m.json = "{\"repetitions\": " + num(m.runs.size()) + ", \"all_outputs_verified\": true, \"counts_equal_across_repetitions\": true" +
             ", \"counts_match_formulas\": " + (m.counts_match ? "true" : "false") + ",\n        \"per_key_without_base_ot\": {\"bytes_both\": " +
             num(key_bytes) + ", \"rounds\": " + num(key_rounds) + ", " + stat_json(key_stat) + "},\n        \"protocol_messages\": {\"bytes_both\": " +
             num(proto_bytes) + ", \"rounds\": " + num(proto_rounds) + ", " + stat_json(stat3(proto_s)) + "},\n";
    if (proto == "leaky")
        m.json += "        \"authenticated_bits\": {\"bytes_both\": " + num(abit_bytes) + ", \"rounds\": " + num(abit_rounds) + ", " +
                  stat_json(stat3(abit_s)) + "},\n";
    m.json += "        \"base_ot\": {\"bytes_both\": " + num(ot_bytes) + ", \"rounds\": " + num(ot_rounds) + ", " + stat_json(stat3(setup_s)) +
              "},\n        \"phases\": {\n" + phases + "},\n        \"runs\": [\n" + runs + "]}";
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

static int usage() {
    std::fprintf(stderr, "usage: bench_baselines [output.json] [--quick] [--reps N] [--n LIST] [--threads T] [--pin A,B] "
                         "[--no-loopback] [--no-tcp]\n");
    return 2;
}

int main(int argc, char** argv) {
    bool quick = false, do_loopback = true, do_tcp = true;
    int reps = 5, party = -1, port = 0;
    unsigned threads = 1;
    int pins[2] = {-1, -1};
    std::vector<unsigned> ns = {8, 12, 16, 20, 24};
    std::string out_path, proto, tag, record_path;
    for (int i = 1; i < argc; i++) {
        const bool more = i + 1 < argc;
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
        else if (std::strcmp(argv[i], "--no-loopback") == 0) do_loopback = false;
        else if (std::strcmp(argv[i], "--no-tcp") == 0) do_tcp = false;
        else if (std::strcmp(argv[i], "--reps") == 0 && more) reps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--n") == 0 && more) ns = parse_list(argv[++i]);
        else if (std::strcmp(argv[i], "--threads") == 0 && more) threads = (unsigned)std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--pin") == 0 && more) {
            const std::vector<unsigned> p = parse_list(argv[++i]);
            if (p.empty() || p.size() > 2) return usage();
            pins[0] = (int)p[0];
            pins[1] = p.size() == 2 ? (int)p[1] : -1;
        } else if (std::strcmp(argv[i], "--party") == 0 && more) party = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--proto") == 0 && more) proto = argv[++i];
        else if (std::strcmp(argv[i], "--port") == 0 && more) port = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tag") == 0 && more) tag = argv[++i];
        else if (std::strcmp(argv[i], "--record") == 0 && more) record_path = argv[++i];
        else if (argv[i][0] == '-') return usage();
        else out_path = argv[i];
    }
    if (threads < 1) threads = 1;
    for (unsigned n : ns)
        if (n < 1 || n > MAX_N) {
            std::fprintf(stderr, "bench_baselines: n must be in 1..%u\n", MAX_N);
            return 2;
        }

    if (party >= 0) {  // internal: one party of a two-process run
        if ((party != 0 && party != 1) || (proto != "leaky" && proto != "semihonest") || ns.size() != 1 || port <= 0 || port > 65535 ||
            record_path.empty())
            return usage();
        return child_main(proto, party, ns[0], port, tag, record_path, threads, pins[0]);
    }

    if (quick) {
        ns = {8, 12};
        reps = 2;
    }
    if (reps < 1 || ns.empty() || (!do_loopback && !do_tcp)) {
        std::fprintf(stderr, "bench_baselines: nothing to run\n");
        return 2;
    }
    const std::string exe = self_path(argc > 0 ? argv[0] : nullptr);
    if (out_path.empty()) {
        const size_t pos = exe.find_last_of("/\\");
        const std::string dir = pos == std::string::npos ? std::string() : exe.substr(0, pos + 1);
        out_path = dir + "../../experiments/raw/baselines_cpp.json";
    }
    {
        // Fail early if the output cannot be written; do not leave an empty file behind.
        const bool existed = std::ifstream(out_path.c_str(), std::ios::binary).good();
        const bool writable = std::ofstream(out_path.c_str(), std::ios::binary | std::ios::app).good();
        if (!writable) {
            std::fprintf(stderr, "bench_baselines: cannot open %s for writing\n", out_path.c_str());
            return 2;
        }
        if (!existed) std::remove(out_path.c_str());
    }

    const double t_all = now_s();
    bool all_verified = true, all_match = true;
    const char* protos[2] = {"leaky", "semihonest"};
    const char* proto_keys[2] = {"leaky_zhang", "semihonest_ds"};
    std::string sections[2];
    for (unsigned n : ns) {
        for (int pi = 0; pi < 2; pi++) {
            ModeResult loop, tcp;
            if (do_loopback) {
                for (int rep = 0; rep < reps; rep++) loop.runs.push_back(run_loopback(protos[pi], n, rep, threads, pins));
                summarise(protos[pi], n, loop);
            }
            if (do_tcp) {
                for (int rep = 0; rep < reps; rep++) tcp.runs.push_back(run_tcp(exe, protos[pi], n, rep, threads, pins, out_path));
                summarise(protos[pi], n, tcp);
            }
            for (const ModeResult* m : {&loop, &tcp}) {
                if (!m->ran) continue;
                all_verified = all_verified && m->verified;
                all_match = all_match && m->counts_equal && m->counts_match;
            }
            std::fprintf(stderr, "%-10s n = %2u: per key (median) loopback %s s, tcp %s s%s%s\n", protos[pi], n,
                         loop.ran && loop.verified ? fmt("%.4f", loop.key_median).c_str() : "-",
                         tcp.ran && tcp.verified ? fmt("%.4f", tcp.key_median).c_str() : "-",
                         (loop.ran && !loop.error.empty()) ? (" [loopback: " + loop.error + "]").c_str() : "",
                         (tcp.ran && !tcp.error.empty()) ? (" [tcp: " + tcp.error + "]").c_str() : "");
            uint64_t model_bytes = 0, model_rounds = 0;
            if (pi == 0) {
                model_bytes = leaky_bytes_per_party(n);
                model_rounds = leaky_rounds(n);
            } else {
                model_bytes = semihonest_bytes_per_party(n);
                model_rounds = semihonest_rounds(n);
            }
            std::string& s = sections[pi];
            if (!s.empty()) s += ",\n";
            s += "    {\"n\": " + num(n) + ", \"leaves\": " + num((uint64_t)1 << n) + ", \"model_protocol_bytes_per_party\": " +
                 num(model_bytes) + ", \"model_protocol_rounds\": " + num(model_rounds);
            if (pi == 0)
                s += ", \"model_authenticated_bit_bytes_per_party\": " +
                     num(prep::u_message_bytes(abit_demand(n)) + prep::COT_SEED_BYTES + prep::COT_CHECK_BYTES) +
                     ", \"authenticated_bits_per_party\": " + num(abit_demand(n));
            if (loop.ran) s += ",\n      \"loopback_two_threads\": " + loop.json;
            if (tcp.ran) s += ",\n      \"tcp_two_processes\": " + tcp.json;
            s += "}";
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
    json += "  \"benchmark\": \"bench_baselines\",\n";
    json += "  \"description\": \"C++ baselines for DESIGN.md section 9, Q1, measured on this machine: OUR RE-IMPLEMENTATION of the "
            "one-bit-leaky actively secure DPF protocol Pi_DPF of Zhang et al. (2024, Figure 5) and a semi-honest "
            "Doerner-shelat-style generation on the same tree; one fresh session (base OT) and one key per repetition\",\n";
    json += "  \"design_version\": \"3.1\",\n";
    json += std::string("  \"timestamp_utc\": \"") + stamp + "\",\n";
    json += "  \"compiler\": \"" + compiler_name() + "\",\n";
    json += std::string("  \"avx2\": ") + (COUPLET_HAVE_AVX2 ? "true" : "false") + ",\n";
    json += std::string("  \"pclmul\": ") + (COUPLET_HAVE_PCLMUL ? "true" : "false") + ",\n";
    json += std::string("  \"sha_extensions\": ") + (probe.accelerated() ? "true" : "false") + ",\n";
    json += "  \"cpu\": \"" + cpu_brand() + "\",\n";
    json += "  \"hardware_threads\": " + num(std::thread::hardware_concurrency()) + ",\n";
    json += std::string("  \"quick_mode\": ") + (quick ? "true" : "false") + ",\n";
    json += "  \"repetitions\": " + num((uint64_t)reps) + ",\n";
    json += "  \"threads_per_party\": " + num(threads) + ",\n";
    json += "  \"pinned_logical_cpus\": " +
            (pins[0] < 0 && pins[1] < 0 ? std::string("null") : "[" + std::to_string(pins[0]) + ", " + std::to_string(pins[1]) + "]") + ",\n";
    json += "  \"transports\": {\n"
            "    \"loopback_two_threads\": \"both parties are threads of one process, in-process LoopbackChannel, no delay\",\n"
            "    \"tcp_two_processes\": \"two processes of this program, TCP on 127.0.0.1, 4-byte length framing (not counted), "
            "TCP_NODELAY, no delay\"},\n";
    json += "  \"reimplementation\": {\n"
            "    \"what\": \"not the original implementations: both baselines are written from the protocol descriptions on top of "
            "this artifact's primitives; message flow, bytes and rounds per phase are those of the Python reference modules "
            "couplet/baselines/leaky_zhang.py and semihonest_ds.py (cross-language runs: cpp/tests/interop_baselines.py)\",\n"
            "    \"hash\": \"H0(x) = first 128 bits of Ascon-p12(x, tweak 0, domain, 0), no feed-forward, 4-way AVX2 interleaving; "
            "H1(x) = H0(x) || H0(x XOR 1); the original of Zhang et al. uses a fixed-key AES-based CCR hash\",\n"
            "    \"cot_source\": \"authenticated bits / COT rows from this artifact's preprocessing: 128 Masny-Rindal base OTs per "
            "direction (Edwards25519) and IKNP-style extension, with the consistency check for the leaky baseline (one call of n + "
            "256 rows per key) and without it for the semi-honest baseline; the original of Zhang et al. uses Ferret (silent COT)\",\n"
            "    \"security_parameters\": \"kappa = 128: MAC keys, tree nodes and payload in GF(2^128), 128-bit COT rows; no "
            "statistical parameter enters (no bucketed triples); coins and commitments: SHA-256 with 32-byte openings\",\n"
            "    \"other\": \"chosen inputs are authenticated by derandomising random BDOZ bits (the key re-randomisation of Input "
            "of Pi_2PC is not sent); BatchCheck is Figure 2 literally including its coin that is unused for a single value; index "
            "bits most significant first; see the header comment of cpp/include/couplet/baselines.hpp\",\n"
            "    \"semihonest\": \"Doerner-shelat PARADIGM (local full-domain expansion, one secure selection per level), not their "
            "protocol bit for bit: the tree and the per-level correlated OT are those of the half-tree construction\"},\n";
    json += "  \"conventions\": {\n"
            "    \"bytes\": \"payload bytes sent; bytes_both = party 0 + party 1; the 4 bytes of TCP framing per message are not "
            "included\",\n"
            "    \"rounds\": \"simultaneous exchanges (Channel::exchange) and flights, as counted by couplet::Channel; identical at "
            "both parties\",\n"
            "    \"seconds\": \"wall clock (steady_clock) of a phase at one party, including waiting for the peer; per repetition "
            "the larger of the two parties; median, minimum and maximum over the repetitions; both parties run on this one "
            "machine\",\n"
            "    \"per_key_without_base_ot\": \"everything after the base OT: authenticated bits (leaky) or the COT flight "
            "(semi-honest), input authentication, tree, final word, check; seconds measured around the key generation call\",\n"
            "    \"protocol_messages\": \"the phases zhang-* (leaky: steps of Figure 5 and the input authentication) or sh-* "
            "(semi-honest)\",\n"
            "    \"phases_leaky\": \"baseot; cotext + abit = authenticated bits from our preprocessing; zhang-input = "
            "authentication of the n + 128 chosen input bits; zhang-root, zhang-levels, zhang-leaf, zhang-check = steps 1, 2, 3 "
            "and 4-5 of Figure 5 (the correction of the last level is applied by the leaf pass and timed under zhang-leaf)\",\n"
            "    \"phases_semihonest\": \"baseot; sh-cot = unchecked IKNP rows for the n selections and the public offset W; "
            "sh-levels; sh-leaf\",\n"
            "    \"payload\": \"GF(2^128) in both baselines; outputs are shares of all 2^n values (the leaky baseline also "
            "outputs MAC shares of all of them)\",\n"
            "    \"harness_steps_not_counted\": \"between the base OT and the key generation the parties exchange one byte (a "
            "barrier, so that both start the key generation together); after the measured part of every run the TEST-ONLY step "
            "of baselines::testonly certifies the outputs; neither step is part of the protocols, neither is timed or counted\",\n"
            "    \"memory\": \"every key allocates and first-touches its buffers inside the timed region: 49 bytes of output per "
            "leaf and party for the leaky baseline, 17 for the semi-honest one, plus the tree buffers (32 bytes per leaf summed over "
            "the levels, at most 24 bytes per leaf alive at a time)\"},\n";
    json += "  \"formulas\": {\n"
            "    \"leaky_protocol_bytes_per_party\": \"16 n + ceil((n + 128) / 8) + 416\",\n"
            "    \"leaky_protocol_rounds\": \"n + 11\",\n"
            "    \"leaky_authenticated_bit_bytes_per_party\": \"16 (64 ceil((n + 256) / 64) + 128) + 16 + 2064, 3 rounds\",\n"
            "    \"semihonest_bytes_per_party\": \"16 n + 1024 ceil(n / 64) + 32\",\n"
            "    \"semihonest_rounds\": \"n + 2\",\n"
            "    \"base_ot_bytes_per_party\": \"12288, 2 rounds\"},\n";
    json += "  \"leaky_zhang\": [\n" + sections[0] + "\n  ],\n";
    json += "  \"semihonest_ds\": [\n" + sections[1] + "\n  ],\n";
    json += std::string("  \"all_outputs_verified\": ") + (all_verified ? "true" : "false") + ",\n";
    json += std::string("  \"all_counts_match_formulas\": ") + (all_match ? "true" : "false") + ",\n";
    json += "  \"total_seconds\": " + fmt("%.1f", now_s() - t_all) + "\n";
    json += "}\n";

    std::fputs(json.c_str(), stdout);
    std::ofstream f(out_path.c_str(), std::ios::binary | std::ios::trunc);
    f << json;
    f.close();
    if (!f) {
        std::fprintf(stderr, "bench_baselines: writing %s failed\n", out_path.c_str());
        return 2;
    }
    std::fprintf(stderr, "bench_baselines: wrote %s\n", out_path.c_str());
    return all_verified && all_match ? 0 : 1;
}
