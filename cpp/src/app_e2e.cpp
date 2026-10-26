// app_e2e.cpp -- ONE party of a two-process run of Pi_AuthEval that goes beyond
// a single call of Pi_Gen: the online layer (Query, Mask, Apply, Open) and the
// four applications of the evaluation (DESIGN.md sections 5.3, 6, 9; Q4, Q5),
// plus three generation-only tasks that couplet_party cannot express.  The
// other party is a second process started with the same public arguments and
// the other --role.  Party 0 listens, party 1 connects.  Measurement driver
// only: no protocol code lives here.
//
//   app_e2e --role 0|1 [--host 127.0.0.1] [--port 47111] --app APP --n N
//           [--k 64] [--count C] [--points Q] [--table random|relu]
//           [--family dpf|dcf] [--beta public|secret] [--auth]
//           [--sigma 40] [--delay-ms D] [--bandwidth-mbps R]
//           [--bandwidth-bucket-bytes C] [--threads T] [--eval-threads E]
//           [--verify] [--out FILE.json] [--circuits DIR] [--timeout-ms MS]
//           [--quiet]
//
// Generation only (the measured part is the one of couplet_party: Session,
// strict mode, ONE triple request, protocol::gen_*):
//
//   gen-key       C keys of --family with XOR-shared index (and --beta).
//                 couplet_party runs the same task; this variant exists for
//                 domains whose full-domain verification does not fit into
//                 memory (n > 24): --verify then compares every key pair with
//                 the plaintext Gen on the same inputs and roots (bit for bit)
//                 and evaluates it at alpha, its neighbours, the two ends of
//                 the domain and 2^16 random points on both planes.
//   gen-interval  material of C interval gates [p <= x <= q] with
//                 p = floor(2^n / 5), q = floor(2^n / 3) - 1 (the preset
//                 "interval" of circuits_gen.py; two payload-1 DCFs).
//   gen-relu      material of C ReLU gates for n = k (relu_large: four DCFs,
//                 an authenticated input offset and one output mask); the
//                 session uses k = --k = n.
//
// Online layer and applications (driver couplet::Party, strict mode: all
// triples of the session are reserved in setup, one bucketing population):
//
//   query         Key: one DPF key (XOR-shared index, public payload 1);
//                 Query at Q public points; Open of the Q exact values with
//                 one opening exchange and one ArithCheck.
//   table         Gate: C table gates on a public table of 2^n k-bit records
//                 (--table random: records from a fixed public seed;
//                 --table relu: ReLU on n-bit two's complement); Mask of C
//                 additively shared inputs; Apply; Open of the C values.
//                 C = 1, n = 16: one private table lookup (A1); C = 16: a
//                 batch (A1); --table relu, n = 12, C = 64: an activation
//                 layer (A3).
//   cmp           Gate: C comparison gates [x < c], c = floor(2^n / 3), shared
//                 input; Mask; Apply; Open.  C = 16, n = 16: A4.
//   cmp-wire      a table gate feeding a comparison gate: C table gates
//                 (shared input, records in [0, 2^n)) and C comparison gates
//                 with an authenticated input wire; Mask (shared), Apply,
//                 Mask (wire; deferred opening over Z_{2^(n+s)}), Apply, Open
//                 of [T[x] < c].
//   histogram     Key: C DPF keys over a domain of 2^n (XOR-shared index,
//                 public payload 1) from ONE circuit execution; the EvalAll
//                 vectors are summed locally and the 2^n bins are opened with
//                 one opening exchange and one ArithCheck (A2).  Each key
//                 contributes an exact indicator, so every bin is an integer
//                 in [0, C]: its representative is smaller than 2^k and the
//                 sum is opened as an exact sharing, without output masks
//                 (DESIGN.md 2.3: the caller sets the flag where the identity
//                 holds over the integers).  C < 2^k is checked.
//
// Inputs are sampled by each party from the operating system (its shares);
// the public points of `query` and the public tables are functions of public
// constants.  --verify adds a TEST-ONLY step after the measured run, which is
// not part of the protocol and is not counted: the processes send each other
// their input shares in the clear (for the generation-only tasks also their
// MAC key shares and outputs, as couplet_party does) and each checks every
// output against the plaintext function.  For `query` the step ends with one
// further Query at the now known index and an Open, which must give 1.
//
// The record (JSON, printed and written to --out) has the fields of the
// record of couplet_party, with three more phases ("mask", "arith-open",
// "arith-check"), and in addition
//
//   steps         traffic and time of this party per command group: setup
//                 (base OT, ALL triples of the session, session-setup
//                 circuit), generation (Key / Gate: evaluation of the
//                 generation circuits and their output checks), mask,
//                 local_eval (Query / Apply: no message), open;
//   online_layer  mask + local_eval + open: bytes, rounds, seconds and,
//                 separately, the seconds of the local evaluation.
//
// Per-phase and per-step bytes, messages and rounds must sum to the channel
// totals; exit status 3 if they do not or if the session was not a single,
// fully used bucketing population, 4 if the verification fails, 1 on any
// other error.  The emulated latency (--delay-ms) and the emulated bandwidth
// limit (--bandwidth-mbps, --bandwidth-bucket-bytes; default: no limit) are
// those of couplet_party (DelayTcpChannel, couplet/netem.hpp).
//
// Build (a target of build_msvc.bat and build.sh; by hand, from artifact/cpp):
//   cl /nologo /O2 /std:c++17 /EHsc /arch:AVX2 /W4 /DNDEBUG /Iinclude src\app_e2e.cpp
//      /Fobuild\ /Febuild\app_e2e.exe ws2_32.lib bcrypt.lib
//   g++ -O2 -std=c++17 -march=native -Iinclude src/app_e2e.cpp -o build/app_e2e -lpthread
//
// The channel (DelayTcpChannel of couplet/netem.hpp, which couplet_party.cpp
// includes), the verification helpers of the generation tasks and the record
// helpers are those of couplet_party.cpp, which is included here as a source
// file with its main() renamed, so that both drivers measure through the same
// code.
#define main couplet_party_main
#include "couplet_party.cpp"
#undef main

#include <functional>
#include <map>

#include "couplet/state.hpp"

#if defined(_WIN32)
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

struct AppArgs {
    int role = -1;
    std::string host = "127.0.0.1";
    int port = 47111;
    std::string app;
    unsigned n = 0, k = 64, s = 64;
    size_t count = 1, points = 1;
    std::string table = "random", family = "dpf", beta = "public";
    bool auth = false;
    int sigma = 40;
    double delay_ms = 0.0;
    double bandwidth_mbps = 0.0;  // 0: no bandwidth limit
    double bucket_bytes = DelayTcpChannel::DEFAULT_BUCKET_BYTES;
    unsigned threads = 1, eval_threads = 1;
    bool verify = false, quiet = false;
    std::string out, circuits;
    int timeout_ms = 60000;
};

void app_usage() {
    std::printf("usage: app_e2e --role 0|1 [--host H] [--port P] --app gen-key|gen-interval|gen-relu|query|table|cmp|cmp-wire|histogram\n"
                "               --n N [--k 64] [--count C] [--points Q] [--table random|relu] [--family dpf|dcf]\n"
                "               [--beta public|secret] [--auth] [--sigma 40] [--delay-ms D] [--bandwidth-mbps R]\n"
                "               [--bandwidth-bucket-bytes C] [--threads T] [--eval-threads E]\n"
                "               [--verify] [--out FILE.json] [--circuits DIR] [--timeout-ms MS] [--quiet]\n");
}

bool is_gen_app(const std::string& a) { return a == "gen-key" || a == "gen-interval" || a == "gen-relu"; }

AppArgs app_parse(int argc, char** argv) {
    AppArgs a;
    for (int i = 1; i < argc; i++) {
        const std::string f = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + f);
            return argv[++i];
        };
        if (f == "--role") a.role = std::atoi(val().c_str());
        else if (f == "--host") a.host = val();
        else if (f == "--port") a.port = std::atoi(val().c_str());
        else if (f == "--app") a.app = val();
        else if (f == "--n") a.n = (unsigned)std::atoi(val().c_str());
        else if (f == "--k") a.k = (unsigned)std::atoi(val().c_str());
        else if (f == "--count") a.count = (size_t)std::atoll(val().c_str());
        else if (f == "--points") a.points = (size_t)std::atoll(val().c_str());
        else if (f == "--table") a.table = val();
        else if (f == "--family") a.family = val();
        else if (f == "--beta") a.beta = val();
        else if (f == "--auth") a.auth = true;
        else if (f == "--sigma") a.sigma = std::atoi(val().c_str());
        else if (f == "--delay-ms") a.delay_ms = std::atof(val().c_str());
        else if (f == "--bandwidth-mbps") a.bandwidth_mbps = std::atof(val().c_str());
        else if (f == "--bandwidth-bucket-bytes") a.bucket_bytes = std::atof(val().c_str());
        else if (f == "--threads") a.threads = (unsigned)std::atoi(val().c_str());
        else if (f == "--eval-threads") a.eval_threads = (unsigned)std::atoi(val().c_str());
        else if (f == "--verify") a.verify = true;
        else if (f == "--quiet") a.quiet = true;
        else if (f == "--out") a.out = val();
        else if (f == "--circuits") a.circuits = val();
        else if (f == "--timeout-ms") a.timeout_ms = std::atoi(val().c_str());
        else throw std::invalid_argument("unknown argument " + f);
    }
    if (a.role != 0 && a.role != 1) throw std::invalid_argument("--role must be 0 or 1");
    static const char* apps[] = {"gen-key", "gen-interval", "gen-relu", "query", "table", "cmp", "cmp-wire", "histogram"};
    bool known = false;
    for (const char* x : apps) known = known || a.app == x;
    if (!known) throw std::invalid_argument("--app must be gen-key, gen-interval, gen-relu, query, table, cmp, cmp-wire or histogram");
    if (a.n < 1 || a.n > 40) throw std::invalid_argument("--n must be in 1..40");
    if (a.k < 1 || a.k > 64) throw std::invalid_argument("--k must be in 1..64");
    if (a.count < 1 || a.count > 65536) throw std::invalid_argument("--count must be in 1..65536");
    if (a.points < 1 || a.points > ((size_t)1 << 20)) throw std::invalid_argument("--points must be in 1..2^20");
    if (a.table != "random" && a.table != "relu") throw std::invalid_argument("--table must be random or relu");
    if (a.family != "dpf" && a.family != "dcf") throw std::invalid_argument("--family must be dpf or dcf");
    if (a.beta != "public" && a.beta != "secret") throw std::invalid_argument("--beta must be public or secret");
    if (a.sigma < 1 || a.sigma > 128) throw std::invalid_argument("--sigma must be in 1..128");
    if (a.delay_ms < 0 || a.delay_ms > 10000) throw std::invalid_argument("--delay-ms must be in 0..10000");
    check_bandwidth_args(a.bandwidth_mbps, a.bucket_bytes);
    if (a.port < 1 || a.port > 65535) throw std::invalid_argument("--port must be in 1..65535");
    if (a.threads < 1) a.threads = 1;
    if (a.eval_threads < 1) a.eval_threads = 1;
    if (a.app != "gen-key" && (a.family != "dpf" || a.beta != "public"))
        throw std::invalid_argument("--family and --beta apply to gen-key only");
    if (a.auth && a.app != "gen-interval") throw std::invalid_argument("--auth applies to gen-interval only");
    if (a.app == "gen-relu" && a.n != a.k) throw std::invalid_argument("gen-relu needs --k equal to --n");
    if (a.app != "gen-key" && a.n > a.k) throw std::invalid_argument("gates need n <= k");
    const bool table_like = a.app == "table" || a.app == "cmp-wire" || a.app == "histogram" || a.app == "query";
    if (table_like && a.n > 24) throw std::invalid_argument("this task evaluates a key on the whole domain: --n must be at most 24");
    if (a.app == "query" && a.n < 40 && a.points > ((size_t)1 << a.n)) throw std::invalid_argument("--points must be at most 2^n");
    if (a.app == "histogram" && a.k < 64 && (a.count >> a.k) != 0) throw std::invalid_argument("histogram needs count < 2^k");
    return a;
}

// Peak memory of this process so far (bytes): physical (working set / resident set) and committed.
void peak_memory(uint64_t* working_set, uint64_t* committed) {
    *working_set = 0;
    *committed = 0;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    std::memset(&pmc, 0, sizeof pmc);
    pmc.cb = sizeof pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        *working_set = (uint64_t)pmc.PeakWorkingSetSize;
        *committed = (uint64_t)pmc.PeakPagefileUsage;
    }
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) *working_set = (uint64_t)ru.ru_maxrss * 1024;  // kilobytes on Linux
#endif
}

// Traffic and time of one command group at this party.
struct StepStat {
    uint64_t bytes_sent = 0, bytes_recv = 0, msgs_sent = 0, msgs_recv = 0, rounds = 0;
    double seconds = 0.0;
    std::string json() const {
        return "{\"bytes_sent\":" + std::to_string(bytes_sent) + ",\"bytes_recv\":" + std::to_string(bytes_recv) + ",\"msgs_sent\":" +
               std::to_string(msgs_sent) + ",\"rounds\":" + std::to_string(rounds) + ",\"seconds\":" + num(seconds) + "}";
    }
    StepStat& operator+=(const StepStat& o) {
        bytes_sent += o.bytes_sent;
        bytes_recv += o.bytes_recv;
        msgs_sent += o.msgs_sent;
        msgs_recv += o.msgs_recv;
        rounds += o.rounds;
        seconds += o.seconds;
        return *this;
    }
};

// Runs `body` and adds its traffic and wall-clock time to `acc`.
template <class F>
auto timed(StepStat& acc, Channel& ch, F&& body) -> decltype(body()) {
    struct Scope {
        StepStat& acc;
        Channel& ch;
        ChannelStats before;
        std::chrono::steady_clock::time_point t;
        Scope(StepStat& a, Channel& c) : acc(a), ch(c), before(c.stats()), t(std::chrono::steady_clock::now()) {}
        ~Scope() {
            const ChannelStats& now = ch.stats();
            acc.bytes_sent += now.bytes_sent - before.bytes_sent;
            acc.bytes_recv += now.bytes_recv - before.bytes_recv;
            acc.msgs_sent += now.msgs_sent - before.msgs_sent;
            acc.msgs_recv += now.msgs_recv - before.msgs_recv;
            acc.rounds += now.rounds - before.rounds;
            acc.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        }
    } scope(acc, ch);
    return body();
}

// One generation circuit of the session and the number of instances it is run on.
struct UsedCircuit {
    std::string call, name, digest;
    size_t instances = 0;
    circuit::Stats stats;
};

// The public table of the table tasks: 2^n records of `bits` bits.
std::vector<uint64_t> public_table(const std::string& kind, unsigned n, unsigned k, unsigned bits) {
    if (kind == "relu") return gates::relu_table(n, k);
    CtrPrgRng rng((uint64_t)0x7ab1e7ab1e000000ull + n * 256 + bits);  // a public constant
    std::vector<uint64_t> T((size_t)1 << n);
    for (uint64_t& t : T) t = rng.u64() & gates::domain_mask(bits);
    return T;
}

// The Q distinct public points of the task `query`: an odd multiple modulo 2^n.
std::vector<uint64_t> public_points(unsigned n, size_t q) {
    std::vector<uint64_t> xs(q);
    for (size_t i = 0; i < q; i++) xs[i] = ((uint64_t)i * 0x9E3779B1ull + 12345u) & gates::domain_mask(n);
    return xs;
}

// Key pair against the plaintext Gen (bit for bit) and at chosen points on both planes; for
// domains whose full expansion does not fit into memory.
template <class Key, class GenFn, class EvalFn>
bool pair_ok_sampled(const Key& k0, const Key& k1, uint64_t alpha, uint64_t beta, u128 delta, bool less_than, GenFn gen, EvalFn eval,
                     uint64_t* points) {
    if (k0.b != 0 || k1.b != 1 || k0.n != k1.n || k0.k != k1.k || k0.s != k1.s) return false;
    const unsigned n = k0.n, w = k0.width();
    const std::pair<Key, Key> ref = gen(alpha, beta, delta, n, k0.k, k0.s, k0.seed, k1.seed);
    if (!(ref.first == k0) || !(ref.second == k1)) return false;
    const u128 mask = u128_mask(w), bt = (delta * u128(beta)) & mask;
    const uint64_t dm = gates::domain_mask(n);
    std::vector<uint64_t> xs = {alpha, (alpha + 1) & dm, (alpha - 1) & dm, 0, dm};
    CtrPrgRng rng(alpha * 0x9E3779B97F4A7C15ull + 1);  // test points only
    for (int i = 0; i < 65536; i++) xs.push_back(rng.u64() & dm);
    bool ok = true;
    for (uint64_t x : xs) {
        const fss::Share a = eval(k0, x), b = eval(k1, x);
        const bool hit = less_than ? x < alpha : x == alpha;
        ok = ok && ((a.y + b.y) & mask) == (hit ? u128(beta) : u128(0)) && ((a.tau + b.tau) & mask) == (hit ? bt : u128(0));
    }
    *points += xs.size();
    return ok;
}

int app_run(const AppArgs& a) {
    typedef std::chrono::steady_clock clock;
    const bool gen_only = is_gen_app(a.app);
    const unsigned k = a.k, s = a.s;
    const protocol::Family family = a.family == "dcf" ? protocol::FAMILY_DCF : protocol::FAMILY_DPF;
    const protocol::BetaMode beta_mode = a.beta == "secret" ? protocol::BETA_SECRET : protocol::BETA_PUBLIC_ONE;
    const uint64_t dmask = gates::domain_mask(a.n);
    const u128 cmp_c = u128(((uint64_t)1 << a.n) / 3);  // the public threshold of the comparison tasks

    // ---------------------------------------------------------------- circuits and the plan (not measured)
    const clock::time_point t_load = clock::now();
    circuit::CircuitStore store(a.circuits);
    const std::string setup_name = Session::setup_circuit_name(s);
    const std::shared_ptr<const circuit::Circuit> setup_c = store.get(setup_name);
    const circuit::Stats st_setup = setup_c->stats();

    std::vector<protocol::Planned> plan;
    std::unique_ptr<gates::GateSpec> spec1, spec2;  // the gates of the online tasks
    protocol::GateDescriptor gen_desc;              // gen-interval / gen-relu
    std::vector<uint64_t> T;                        // public table
    if (a.app == "gen-key") {
        plan.push_back(protocol::planned_key(store, family, a.n, k, s, protocol::INDEX_XOR, beta_mode, a.count));
    } else if (a.app == "gen-interval") {
        const uint64_t N = (uint64_t)1 << a.n;
        gen_desc = gates::interval_descriptor(a.n, k, s, N / 5, N / 3 - 1, a.auth);
        plan.push_back(protocol::planned_gate(store, gen_desc, a.count));
    } else if (a.app == "gen-relu") {
        gen_desc = gates::relu_large_descriptor(k, s);
        plan.push_back(protocol::planned_gate(store, gen_desc, a.count));
    } else if (a.app == "query") {
        plan.push_back(protocol::planned_key(store, protocol::FAMILY_DPF, a.n, k, s, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, 1));
    } else if (a.app == "histogram") {
        plan.push_back(
            protocol::planned_key(store, protocol::FAMILY_DPF, a.n, k, s, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, a.count));
    } else if (a.app == "table") {
        T = public_table(a.table, a.n, k, k);
        spec1.reset(new gates::GateSpec(gates::GateSpec::table_gate(a.n, k, s, T, false)));
        plan.push_back(protocol::planned_gate(store, spec1->descriptor(), a.count));
    } else if (a.app == "cmp") {
        spec1.reset(new gates::GateSpec(gates::GateSpec::comparison(a.n, k, s, cmp_c, false)));
        plan.push_back(protocol::planned_gate(store, spec1->descriptor(), a.count));
    } else {  // cmp-wire
        T = public_table("random", a.n, k, a.n);  // records in [0, 2^n): the value that is compared
        spec1.reset(new gates::GateSpec(gates::GateSpec::table_gate(a.n, k, s, T, false)));
        spec2.reset(new gates::GateSpec(gates::GateSpec::comparison(a.n, k, s, cmp_c, true)));
        plan.push_back(protocol::planned_gate(store, spec1->descriptor(), a.count));
        plan.push_back(protocol::planned_gate(store, spec2->descriptor(), a.count));
    }
    std::vector<UsedCircuit> used;
    for (const protocol::Planned& p : plan) {
        const std::shared_ptr<const circuit::Circuit> c = store.get(p.circuit);
        UsedCircuit u;
        u.call = p.call;
        u.name = p.circuit;
        u.digest = c->digest_hex();
        u.instances = p.batch;
        u.stats = c->stats();
        used.push_back(u);
    }
    const size_t reserve = protocol::and_budget(plan);
    const uint64_t n_triples = (uint64_t)protocol::session_triples(store, plan, s);
    const double load_s = secs(t_load, clock::now());

    // ---------------------------------------------------------------- connection and an unmetered hello
    std::string params = "couplet-app/1|app=" + a.app + "|n=" + std::to_string(a.n) + "|k=" + std::to_string(k) + "|s=" +
                         std::to_string(s) + "|count=" + std::to_string(a.count) + "|points=" + std::to_string(a.points) + "|table=" +
                         a.table + "|family=" + a.family + "|beta=" + a.beta + "|auth=" + (a.auth ? "1" : "0") + "|sigma=" +
                         std::to_string(a.sigma) + "|delay_ms=" + num(a.delay_ms) + bandwidth_hello(a.bandwidth_mbps, a.bucket_bytes) +
                         "|verify=" + (a.verify ? "1" : "0");
    for (const UsedCircuit& u : used) params += "|circuit=" + u.digest;
    std::unique_ptr<DelayTcpChannel> chp =
        DelayTcpChannel::open(a.role, a.host, (uint16_t)a.port, a.delay_ms, a.timeout_ms, a.bandwidth_mbps, a.bucket_bytes);
    DelayTcpChannel& ch = *chp;
    SystemRng sys;
    bytes nonce[2];
    nonce[a.role] = sys.get(16);
    {
        bytes hello(params.begin(), params.end());
        hello.push_back(0);
        hello.insert(hello.end(), nonce[a.role].begin(), nonce[a.role].end());
        const bytes theirs = ch.exchange(hello, 4096, "hello: message too long");
        if (theirs.size() != hello.size() || std::memcmp(theirs.data(), hello.data(), params.size() + 1) != 0)
            throw std::runtime_error("the two processes were started with different parameters (peer: " +
                                     std::string(theirs.begin(), theirs.begin() + (ptrdiff_t)std::min(theirs.size(), (size_t)300)) + ")");
        nonce[1 - a.role].assign(theirs.end() - 16, theirs.end());
    }
    const hashes::hash32 sid_hash = hashes::framed_sha256({"Couplet-app-sid", params, nonce[0], nonce[1]});
    const bytes sid(sid_hash.begin(), sid_hash.end());

    // This party's inputs (its shares), sampled before the measured run.
    std::vector<uint64_t> in_shares, beta_shares;
    const size_t n_inputs = a.app == "query" ? 1 : (a.app == "gen-interval" || a.app == "gen-relu") ? 0 : a.count;
    for (size_t j = 0; j < n_inputs; j++) in_shares.push_back(sys.u64() & dmask);
    if (a.app == "gen-key" && beta_mode == protocol::BETA_SECRET)
        for (size_t j = 0; j < a.count; j++) beta_shares.push_back(sys.u64() & gates::domain_mask(k));
    const std::vector<uint64_t> points = a.app == "query" ? public_points(a.n, a.points) : std::vector<uint64_t>();
    ch.reset_stats();

    // ---------------------------------------------------------------- the measured run
    StepStat st_setup_step, st_gen, st_mask, st_local, st_open;
    std::vector<std::string> gen_stats;        // EvalStats of every generation call
    std::vector<fss::DpfKey> dpf_keys;         // gen-key
    std::vector<fss::DcfKey> dcf_keys;         // gen-key
    std::vector<protocol::GateMaterial> mats;  // gen-interval / gen-relu
    std::vector<u128> opened;                  // online tasks: the opened outputs
    std::vector<uint64_t> x_hats;              // online gate tasks: the public masked inputs
    size_t out_bytes_first = 0, out_bytes_all = 0;
    std::string out_kind;
    uint64_t arith_openings = 0, arith_checks = 0;

    const clock::time_point t0 = clock::now();
    Session ses(a.role, ch, sid, store, k, s, a.sigma);
    ses.real_prep()->set_threads(a.threads);
    std::unique_ptr<Party> party;
    uint64_t query_kid = 0;
    if (gen_only) {
        ses.set_allow_refill(false);  // strict mode, as couplet_party
        timed(st_setup_step, ch, [&] {
            ses.setup(reserve);  // all triples of the session in ONE preprocessing request
            return 0;
        });
        timed(st_gen, ch, [&] {
            if (a.app == "gen-key") {
                if (family == protocol::FAMILY_DPF)
                    dpf_keys = protocol::gen_dpf(ses, a.n, protocol::INDEX_XOR, beta_mode, in_shares, beta_shares);
                else
                    dcf_keys = protocol::gen_dcf(ses, a.n, protocol::INDEX_XOR, beta_mode, in_shares, beta_shares);
            } else {
                mats = protocol::gen_gate(ses, gen_desc, a.count);
            }
            return 0;
        });
        gen_stats.push_back(ses.last_gen.stats.json());
    } else {
        party.reset(new Party(ses));  // strict by default
        Party& p = *party;
        p.set_eval_threads(a.eval_threads);
        timed(st_setup_step, ch, [&] {
            p.setup(reserve, 0);  // ONE bucketing population; no task below opens a value that is not exact
            return 0;
        });
        if (a.app == "query") {
            query_kid = timed(st_gen, ch, [&] { return p.key(Party::NEXT, protocol::FAMILY_DPF, a.n, in_shares[0]); });
            gen_stats.push_back(ses.last_gen.stats.json());
            std::vector<Party::AShare> ys;
            timed(st_local, ch, [&] {
                ys.reserve(points.size());
                for (uint64_t x : points) ys.push_back(p.query(query_kid, x));
                return 0;
            });
            opened = timed(st_open, ch, [&] { return p.open_many(ys); });
        } else if (a.app == "histogram") {
            const std::vector<uint64_t> kids =
                timed(st_gen, ch, [&] { return p.key_batch(Party::NEXT, protocol::FAMILY_DPF, a.n, in_shares); });
            gen_stats.push_back(ses.last_gen.stats.json());
            std::vector<Party::AShare> bins;
            timed(st_local, ch, [&] {
                const size_t size = (size_t)1 << a.n;
                bins.assign(size, Party::AShare(u128(0), u128(0), p.w(), true));
                for (uint64_t kid : kids) {
                    const std::vector<Party::AShare> v = p.query_all(kid);
                    for (size_t x = 0; x < size; x++) bins[x] = bins[x] + v[x];
                }
                // Each key is an exact indicator, so bin x is the integer #{i : alpha_i = x} <= count < 2^k.
                for (Party::AShare& b : bins) b = b.with_exact(true);
                return 0;
            });
            opened = timed(st_open, ch, [&] { return p.open_many(bins); });
        } else {
            const std::vector<uint64_t> g1 = timed(st_gen, ch, [&] { return p.gate_batch(Party::NEXT, *spec1, a.count); });
            gen_stats.push_back(ses.last_gen.stats.json());
            std::vector<uint64_t> g2;
            if (spec2) {
                g2 = timed(st_gen, ch, [&] { return p.gate_batch(Party::NEXT, *spec2, a.count); });
                gen_stats.push_back(ses.last_gen.stats.json());
            }
            std::vector<Party::AShare> ys;
            ys.reserve(a.count);
            for (size_t i = 0; i < a.count; i++)
                x_hats.push_back(timed(st_mask, ch, [&] { return p.mask_shared(g1[i], in_shares[i]); }));
            for (size_t i = 0; i < a.count; i++) {
                Party::AShare y = timed(st_local, ch, [&] { return p.apply(g1[i])[0]; });
                if (spec2) {
                    timed(st_mask, ch, [&] { return p.mask_wire(g2[i], y); });  // deferred opening over Z_{2^(n+s)}
                    y = timed(st_local, ch, [&] { return p.apply(g2[i])[0]; });
                }
                ys.push_back(y);
            }
            opened = timed(st_open, ch, [&] { return p.open_many(ys); });
        }
        arith_openings = p.checker()->openings();
        arith_checks = p.checker()->checks();
    }
    const clock::time_point t_end = clock::now();
    // ---------------------------------------------------------------- end of the measured run

    const ChannelStats total = ch.stats();
    const double emulated_wait = ch.emulated_wait_s();
    const double bandwidth_wait = ch.bandwidth_wait_s();
    const double wall = secs(t0, t_end);
    uint64_t peak_ws = 0, peak_commit = 0;
    peak_memory(&peak_ws, &peak_commit);  // before the verification step allocates anything

    Reveal mine;  // what this party reveals in the TEST-ONLY step
    mine.alpha = in_shares;
    mine.beta = beta_shares;
    if (gen_only) {
        for (const fss::DpfKey& key : dpf_keys) mine.outputs.push_back(key.to_bytes());
        for (const fss::DcfKey& key : dcf_keys) mine.outputs.push_back(key.to_bytes());
        for (const protocol::GateMaterial& m : mats) mine.outputs.push_back(m.to_bytes());
        mine.delta = ses.delta_share();
        for (const bytes& o : mine.outputs) out_bytes_all += o.size();
        out_bytes_first = mine.outputs.empty() ? 0 : mine.outputs[0].size();
        out_kind = a.app == "gen-key" ? (family == protocol::FAMILY_DPF ? "dpf key" : "dcf key") : "gate material";
    } else {
        // Key and gate material this party holds for the task (per instance; the keys stay in the registry).
        if (a.app == "query" || a.app == "histogram") {
            out_bytes_first = fss::dpf_key_size(a.n, k, s);
            out_bytes_all = out_bytes_first * (a.app == "query" ? 1 : a.count);
            out_kind = "dpf key";
        } else {
            const state::GateRecord& rec = party->registry().gate(0);
            out_bytes_first = rec.material().to_bytes().size();
            if (spec2) out_bytes_first += party->registry().gate(a.count).material().to_bytes().size();
            out_bytes_all = out_bytes_first * a.count;
            out_kind = "gate material";
        }
    }

    // Phases: the preprocessing books its own; the engine's and the online layer's are in the session book.
    const prep::PrepStats pst = ses.real_prep()->stats();
    abb::PhaseBook& book = ses.book();
    const prep::PhaseStat ph_input = plus(book.get("setup-input"), book.get("input"));
    const prep::PhaseStat ph_online = plus(book.get("setup-online"), book.get("online"));
    const prep::PhaseStat ph_output =
        plus(plus(book.get("setup-outcheck"), book.get("setup-output")), plus(book.get("outcheck"), book.get("output")));
    const prep::PhaseStat ph_mask = book.get("mask"), ph_aopen = book.get("arith-open"), ph_acheck = book.get("arith-check");
    prep::PhaseStat sum = pst.total();
    sum += ph_input;
    sum += ph_online;
    sum += ph_output;
    sum += ph_mask;
    sum += ph_aopen;
    sum += ph_acheck;
    const prep::PhaseStat book_total = book.total();
    const bool sums_ok = sum.bytes_sent == total.bytes_sent && sum.bytes_recv == total.bytes_recv && sum.rounds == total.rounds &&
                         sum.msgs_sent == total.msgs_sent && sum.msgs_recv == total.msgs_recv &&
                         pst.total().bytes_sent + book_total.bytes_sent == total.bytes_sent &&
                         pst.total().rounds + book_total.rounds == total.rounds;
    StepStat step_sum = st_setup_step;
    step_sum += st_gen;
    step_sum += st_mask;
    step_sum += st_local;
    step_sum += st_open;
    StepStat online_layer = st_mask;
    online_layer += st_local;
    online_layer += st_open;
    const bool steps_ok = step_sum.bytes_sent == total.bytes_sent && step_sum.bytes_recv == total.bytes_recv &&
                          step_sum.rounds == total.rounds && step_sum.msgs_sent == total.msgs_sent && st_local.bytes_sent == 0 &&
                          st_local.bytes_recv == 0 && st_local.rounds == 0 &&
                          ph_mask.bytes_sent + ph_aopen.bytes_sent + ph_acheck.bytes_sent == online_layer.bytes_sent &&
                          ph_mask.rounds + ph_aopen.rounds + ph_acheck.rounds == online_layer.rounds;

    const unsigned bucket = ses.real_prep()->last_bucket_size();
    const bool single_population = ses.triple_requests() == 1 && ses.triples_requested() == n_triples;
    const bool all_used = ses.triples_used() == n_triples && ses.triples_unused() == 0;
    double bound_log2 = 0;
    const bool has_bound = ses.bucketing_bound_log2(&bound_log2);

    // ---------------------------------------------------------------- TEST-ONLY verification
    // Not part of the protocol and not counted above.
    bool verified = false, correct = false;
    uint64_t checked = 0, points_checked = 0;
    double verify_s = 0;
    std::string verify_mode = "none";
    if (a.verify) {
        const clock::time_point tv = clock::now();
        const bytes mine_enc = mine.encode();
        const Reveal theirs = Reveal::decode(ch.exchange(mine_enc, 2 * mine_enc.size() + 65536, "verification message too long"));
        const Reveal& r0 = a.role == 0 ? mine : theirs;
        const Reveal& r1 = a.role == 0 ? theirs : mine;
        correct = r0.alpha.size() == n_inputs && r1.alpha.size() == n_inputs;
        const unsigned vt = std::max(4u, a.threads);
        if (correct && gen_only) {
            const u128 delta = fss::make_delta(r0.delta, r1.delta);  // the integer Delta_0 + Delta_1, never reduced
            correct = r0.outputs.size() == a.count && r1.outputs.size() == a.count && r0.beta.size() == r1.beta.size() &&
                      (a.app == "gen-key" && beta_mode == protocol::BETA_SECRET ? r0.beta.size() == a.count : r0.beta.empty());
            const bool full = a.n <= 24;
            verify_mode = a.app != "gen-key" ? "gate material: offset sharing, keys on the whole domain, [[w_c]], output masks"
                          : full ? "plaintext Gen identity and the whole domain on both planes"
                                 : "plaintext Gen identity and 65,541 points (alpha, its neighbours, 0, 2^n - 1, random) on both planes";
            for (size_t j = 0; correct && j < a.count; j++) {
                if (a.app == "gen-key") {
                    const uint64_t alpha = r0.alpha[j] ^ r1.alpha[j];
                    const uint64_t beta =
                        beta_mode == protocol::BETA_SECRET ? ((r0.beta[j] + r1.beta[j]) & gates::domain_mask(k)) : 1;
                    if (family == protocol::FAMILY_DPF) {
                        const fss::DpfKey k0 = fss::DpfKey::from_bytes(r0.outputs[j]), k1 = fss::DpfKey::from_bytes(r1.outputs[j]);
                        correct = full ? dpf_pair_ok(k0, k1, alpha, beta, delta, vt, &points_checked)
                                       : pair_ok_sampled(
                                             k0, k1, alpha, beta, delta, false,
                                             [](uint64_t al, uint64_t be, u128 de, unsigned n, unsigned kk, unsigned ss, const fss::seed_t& s0,
                                                const fss::seed_t& s1) { return fss::dpf_gen(al, be, de, n, kk, ss, s0, s1); },
                                             [](const fss::DpfKey& key, uint64_t x) { return fss::dpf_eval(key, x); }, &points_checked);
                    } else {
                        const fss::DcfKey k0 = fss::DcfKey::from_bytes(r0.outputs[j]), k1 = fss::DcfKey::from_bytes(r1.outputs[j]);
                        correct = full ? dcf_pair_ok(k0, k1, alpha, beta, delta, vt, &points_checked)
                                       : pair_ok_sampled(
                                             k0, k1, alpha, beta, delta, true,
                                             [](uint64_t al, uint64_t be, u128 de, unsigned n, unsigned kk, unsigned ss, const fss::seed_t& s0,
                                                const fss::seed_t& s1) { return fss::dcf_gen(al, be, de, n, kk, ss, s0, s1); },
                                             [](const fss::DcfKey& key, uint64_t x) { return fss::dcf_eval(key, x); }, &points_checked);
                    }
                } else {
                    correct = gate_pair_ok(protocol::GateMaterial::from_bytes(r0.outputs[j]),
                                           protocol::GateMaterial::from_bytes(r1.outputs[j]), gen_desc, delta, vt, &points_checked);
                }
                checked++;
            }
        } else if (correct) {
            verify_mode = "every opened output against the plaintext function of the revealed inputs";
            if (a.app == "query") {
                const uint64_t alpha = r0.alpha[0] ^ r1.alpha[0];
                correct = opened.size() == points.size();
                for (size_t i = 0; correct && i < points.size(); i++, checked++) correct = opened[i] == u128(points[i] == alpha ? 1 : 0);
                // One more Query at the now known index, through the protocol: the value must be 1.
                if (correct) {
                    correct = party->open(party->query(query_kid, alpha)) == u128(1);
                    checked++;
                    verify_mode += "; plus one Query at the revealed index and its Open (value 1)";
                }
            } else if (a.app == "histogram") {
                std::vector<uint64_t> hist((size_t)1 << a.n, 0);
                for (size_t j = 0; j < a.count; j++) hist[(size_t)(r0.alpha[j] ^ r1.alpha[j])]++;
                correct = opened.size() == hist.size();
                for (size_t x = 0; correct && x < hist.size(); x++, checked++) correct = opened[x] == u128(hist[x]);
            } else {
                correct = opened.size() == a.count && x_hats.size() == a.count;
                for (size_t i = 0; correct && i < a.count; i++, checked++) {
                    const uint64_t x = (r0.alpha[i] + r1.alpha[i]) & dmask;
                    uint64_t want = 0;
                    if (a.app == "table") want = T[(size_t)x] & gates::domain_mask(k);
                    else if (a.app == "cmp") want = u128(x) < cmp_c ? 1 : 0;
                    else want = u128(T[(size_t)x] & dmask) < cmp_c ? 1 : 0;
                    correct = opened[i] == u128(want);
                }
            }
        }
        verified = true;
        verify_s = secs(tv, clock::now());
    }

    // ---------------------------------------------------------------- the record
    auto b2s = [](bool b) { return std::string(b ? "true" : "false"); };
    size_t n_and_total = 0, depth_max = 0, gates_total = 0;
    for (const UsedCircuit& u : used) {
        n_and_total += (size_t)u.stats.n_and * u.instances;
        depth_max = std::max(depth_max, (size_t)u.stats.and_depth);
        gates_total += (size_t)u.stats.n_gates * u.instances;
    }
    const size_t items = a.app == "query" ? a.points : a.app == "histogram" ? ((size_t)1 << a.n) : a.count;  // opened values
    std::string rec = "{\n";
    rec += " \"schema\": \"couplet.app_e2e/1\",\n";
    rec += " \"implementation\": \"C++ (couplet, native)\",\n";
    rec += " \"compiler\": " + q(compiler_name()) + ",\n";
    rec += " \"timestamp_utc\": " + q(utc_now()) + ",\n";
    rec += " \"role\": " + std::to_string(a.role) + ",\n";
    rec += " \"params\": {\"app\": " + q(a.app) + ", \"n\": " + std::to_string(a.n) + ", \"k\": " + std::to_string(k) + ", \"s\": " +
           std::to_string(s) + ", \"count\": " + std::to_string(a.count) + ", \"batch\": " + std::to_string(a.count) +
           ", \"points\": " + std::to_string(a.points) + ", \"table\": " + q(a.table) + ", \"family\": " + q(a.family) +
           ", \"index_mode\": \"xor\", \"beta\": " + q(a.beta) + ", \"auth_input\": " + b2s(a.auth) + ", \"prep\": \"real\", \"sigma\": " +
           std::to_string(a.sigma) + ", \"delay_ms\": " + num(a.delay_ms) + netem_params_json(a.bandwidth_mbps, a.bucket_bytes) +
           ", \"threads\": " + std::to_string(a.threads) +
           ", \"eval_threads\": " + std::to_string(a.eval_threads) + ", \"mode\": \"two-process\", \"transport\": \"tcp\", \"host\": " +
           q(a.host) + ", \"port\": " + std::to_string(a.port) + "},\n";
    rec += " \"ideal_preprocessing\": false,\n";
    rec += " \"circuit\": {\"name\": " + q(used[0].name) + ", \"digest\": " + q(used[0].digest) + ", \"stats\": " + used[0].stats.json() +
           "},\n";
    rec += " \"circuits\": [";
    for (size_t i = 0; i < used.size(); i++)
        rec += std::string(i ? ", " : "") + "{\"call\": " + q(used[i].call) + ", \"name\": " + q(used[i].name) + ", \"digest\": " +
               q(used[i].digest) + ", \"instances\": " + std::to_string(used[i].instances) + ", \"n_and\": " +
               std::to_string(used[i].stats.n_and) + ", \"and_depth\": " + std::to_string(used[i].stats.and_depth) + ", \"n_gates\": " +
               std::to_string(used[i].stats.n_gates) + "}";
    rec += "],\n";
    rec += " \"circuits_total\": {\"n_and_all_instances\": " + std::to_string(n_and_total) + ", \"and_depth_max\": " +
           std::to_string(depth_max) + ", \"n_gates_all_instances\": " + std::to_string(gates_total) + "},\n";
    rec += " \"session_setup_circuit\": {\"name\": " + q(setup_name) + ", \"digest\": " + q(setup_c->digest_hex()) + ", \"n_and\": " +
           std::to_string(st_setup.n_and) + ", \"and_depth\": " + std::to_string(st_setup.and_depth) + ", \"n_gates\": " +
           std::to_string(st_setup.n_gates) + "},\n";
    rec += " \"preprocessing\": {\"triples\": " + std::to_string(n_triples) + ", \"triples_generation_circuit\": " +
           std::to_string(reserve) + ", \"triples_session_setup\": " + std::to_string(st_setup.n_and) + ", \"triple_requests\": " +
           std::to_string(ses.triple_requests()) + ", \"triples_requested\": " + std::to_string(ses.triples_requested()) +
           ", \"allow_refill\": " + b2s(ses.allow_refill()) + ", \"bucketing_bound_log2\": " +
           (has_bound ? num(bound_log2) : std::string("null")) + ", \"bucket_size\": " + std::to_string(bucket) + ", \"leaky_triples\": " +
           std::to_string(n_triples * bucket) + ", \"model_bytes_per_and\": " + std::to_string(96 * bucket) + ", \"input_masks\": " +
           std::to_string(ses.prep().counters().input_masks) + "},\n";
    rec += " \"phases\": {";
    for (int p = 0; p < prep::PH_COUNT; p++) rec += q(prep::phase_name(p)) + ": " + phase_json(pst.phase[p]) + ", ";
    rec += "\"input\": " + phase_json(ph_input) + ", \"online\": " + phase_json(ph_online) + ", \"output\": " + phase_json(ph_output) +
           ", \"mask\": " + phase_json(ph_mask) + ", \"arith-open\": " + phase_json(ph_aopen) + ", \"arith-check\": " +
           phase_json(ph_acheck) + "},\n";
    rec += " \"phases_note\": \"baseot .. check: preprocessing; input / online / output: Boolean engine (session-setup circuit and "
           "generation circuits; output = the two BoolChecks of Output and the opening); mask / arith-open / arith-check: the online "
           "layer (Mask of a shared input; arithmetic openings incl. the deferred opening of a wire Mask; ArithCheck); seconds are "
           "wall-clock at this party and include waiting for the peer\",\n";
    rec += " \"phases_detail\": {";
    for (size_t i = 0; i < book.size(); i++) rec += std::string(i ? ", " : "") + q(book.name(i)) + ": " + phase_json(book.stat(i));
    rec += "},\n";
    rec += " \"steps\": {\"setup\": " + st_setup_step.json() + ", \"generation\": " + st_gen.json() + ", \"mask\": " + st_mask.json() +
           ", \"local_eval\": " + st_local.json() + ", \"open\": " + st_open.json() + "},\n";
    rec += " \"steps_note\": \"setup = base OT, all AND triples of the session and the session-setup circuit; generation = Key / Gate "
           "(Pi_Gen on the reserved triples); mask = Mask; local_eval = Query / Apply (no message); open = Open (opening exchange and "
           "ArithChecks)\",\n";
    rec += " \"online_layer\": {\"bytes_sent\": " + std::to_string(online_layer.bytes_sent) + ", \"bytes_recv\": " +
           std::to_string(online_layer.bytes_recv) + ", \"bytes\": " + std::to_string(online_layer.bytes_sent + online_layer.bytes_recv) +
           ", \"rounds\": " + std::to_string(online_layer.rounds) + ", \"seconds\": " + num(online_layer.seconds) + ", \"mask_s\": " +
           num(st_mask.seconds) + ", \"local_eval_s\": " + num(st_local.seconds) + ", \"open_s\": " + num(st_open.seconds) +
           ", \"opened_values\": " + std::to_string(gen_only ? 0 : items) + ", \"arith_openings\": " + std::to_string(arith_openings) +
           ", \"arith_checks\": " + std::to_string(arith_checks) + "},\n";
    rec += " \"totals\": {\"bytes_sent\": " + std::to_string(total.bytes_sent) + ", \"bytes_recv\": " + std::to_string(total.bytes_recv) +
           ", \"bytes\": " + std::to_string(total.bytes_sent + total.bytes_recv) + ", \"msgs_sent\": " + std::to_string(total.msgs_sent) +
           ", \"rounds\": " + std::to_string(total.rounds) + ", \"seconds\": " + num(wall) + ", \"setup_s\": " +
           num(st_setup_step.seconds) + ", \"generation_s\": " + num(st_gen.seconds) + ", \"online_s\": " +
           num(ses.last_gen.stats.online_s) + ", \"online_layer_s\": " + num(online_layer.seconds) + ", \"phase_seconds_sum\": " +
           num(sum.seconds) + ", \"emulated_wait_s\": " + num(emulated_wait) + ", \"bandwidth_wait_s\": " + num(bandwidth_wait) +
           ", \"load_circuits_s\": " + num(load_s) +
           ", \"bytes_per_and_gate\": " + num((double)(total.bytes_sent + total.bytes_recv) / (double)n_triples) +
           ", \"bytes_per_instance\": " + num((double)(total.bytes_sent + total.bytes_recv) / (double)a.count) +
           ", \"seconds_per_instance\": " + num(wall / (double)a.count) + "},\n";
    rec += " \"totals_note\": \"bytes = payload bytes in both directions (sent + received by this party); seconds = this party's "
           "wall-clock time from the first protocol message to the last accepted output (session setup, preprocessing, generation "
           "and online layer), both processes on one machine; online_s = AND layers of the last generation circuit\",\n";
    rec += " \"peak_memory\": {\"working_set_bytes\": " + std::to_string(peak_ws) + ", \"committed_bytes\": " +
           std::to_string(peak_commit) + ", \"note\": \"peak of this process at the end of the measured run, before the test-only "
           "step; includes the loaded circuits\"},\n";
    rec += " \"generations\": [";
    for (size_t i = 0; i < gen_stats.size(); i++) rec += std::string(i ? ", " : "") + gen_stats[i];
    rec += "],\n";
    rec += " \"generation\": " + gen_stats.back() + ",\n";
    rec += " \"session_setup_evaluation\": " + ses.setup_stats().json() + ",\n";
    rec += " \"outputs\": {\"kind\": " + q(out_kind) + ", \"instances\": " + std::to_string(a.app == "query" ? 1 : a.count) +
           ", \"bytes_per_instance_and_party\": " + std::to_string(out_bytes_first) + ", \"bytes_all_instances_per_party\": " +
           std::to_string(out_bytes_all) + "},\n";
    rec += std::string(" \"verification\": {\"performed\": ") + b2s(verified) + ", \"outputs_correct\": " +
           (verified ? b2s(correct) : std::string("null")) + ", \"instances_checked\": " + std::to_string(checked) +
           ", \"points_checked\": " + std::to_string(points_checked) + ", \"mode\": " + q(verify_mode) + ", \"seconds\": " +
           num(verify_s) + ", \"note\": \"TEST ONLY, not part of the protocol and not counted in any total: after the measured run "
           "the two processes exchange their input shares (generation-only tasks: also MAC key shares and outputs) in the clear and "
           "check every output\"},\n";
    rec += " \"checks\": {\"phase_sums_ok\": " + b2s(sums_ok) + ", \"steps_sum_ok\": " + b2s(steps_ok) +
           ", \"single_bucketing_population\": " + b2s(single_population) + ", \"all_triples_used\": " + b2s(all_used) +
           ", \"strict_mode\": " + b2s(!ses.allow_refill()) + ", \"session_active\": " + b2s(ses.active()) + "}\n";
    rec += "}\n";

    if (!a.out.empty()) {
        std::ofstream f(a.out, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + a.out);
        f << rec;
    }
    if (a.quiet) {
        std::printf("party %d: %s n=%u count=%zu: %llu bytes (both directions), %llu rounds, %.3f s (online layer %.6f s, local %.6f s)%s\n",
                    a.role, a.app.c_str(), a.n, a.count, (unsigned long long)(total.bytes_sent + total.bytes_recv),
                    (unsigned long long)total.rounds, wall, online_layer.seconds, st_local.seconds,
                    verified ? (correct ? ", outputs verified" : ", VERIFICATION FAILED") : "");
    } else {
        std::fputs(rec.c_str(), stdout);
    }
    std::fflush(stdout);
    if (!sums_ok || !steps_ok) {
        std::fprintf(stderr, "error: the per-phase or per-step counters do not sum to the channel totals\n");
        return 3;
    }
    if (!single_population || !all_used) {
        std::fprintf(stderr, "error: the session did not consist of a single, fully used bucketing population\n");
        return 3;
    }
    if (verified && !correct) {
        std::fprintf(stderr, "error: the outputs are NOT correct\n");
        return 4;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    AppArgs a;
    try {
        a = app_parse(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        app_usage();
        return 2;
    }
    if (a.circuits.empty()) a.circuits = circuit::dirname(argv[0]) + "/../../circuits";
    try {
        return app_run(a);
    } catch (const ProtocolAbort& e) {
        std::fprintf(stderr, "party %d: PROTOCOL ABORT: %s\n", a.role, e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "party %d: error: %s\n", a.role, e.what());
    }
    return 1;
}
