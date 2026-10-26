// couplet_party.cpp -- ONE party of a two-process run of Pi_Gen over TCP
// (DESIGN.md sections 4, 5.2, 7, 9): session setup (base OT, the persistent
// wires of Delta), preprocessing (OT extension, authenticated bits, leaky
// ANDs, bucketing), evaluation of the generation circuit and the output
// checks.  The other party is a second process started with the same public
// arguments and the other --role.  Party 0 listens, party 1 connects.
//
//   couplet_party --role 0|1 [--host 127.0.0.1] [--port 47110]
//                 --task dpf|dcf|gate-table|gate-cmp --n N [--k 64]
//                 [--beta public|secret] [--auth] [--batch B]
//                 [--prep real|ideal] [--sigma 40] [--delay-ms D]
//                 [--bandwidth-mbps R] [--bandwidth-bucket-bytes C]
//                 [--threads T] [--verify] [--out FILE.json]
//                 [--circuits DIR] [--timeout-ms MS] [--quiet]
//
//   --task        dpf / dcf: one key pair per batch instance (index shares are
//                 XOR shares; --beta secret adds additive payload shares);
//                 gate-table: material of the table / zero-test gate (one
//                 payload-1 DPF at the offset r); gate-cmp: material of the
//                 comparison gate [x < c] with c = floor(2^n / 3) (two DCFs).
//                 --auth generates the offset as [[r']] over Z_{2^(n+s)} for
//                 an authenticated input wire.
//   --batch       number of independent instances of the circuit that share
//                 the rounds (the SIMD dimension of the engine).
//   --prep ideal  NOT a protocol: both processes derive the same trusted
//                 dealer from a fixed public seed.  Such records carry
//                 "ideal_preprocessing": true and must never be reported as
//                 all-in numbers.
//   --delay-ms    one-way latency emulated at application level inside the
//                 channel (DelayTcpChannel, couplet/netem.hpp).
//   --bandwidth-mbps
//                 bandwidth limit per direction in 10^6 bit/s, emulated in
//                 the same channel by a token bucket that throttles the
//                 sender (0, the default: no limit); the delay is added per
//                 message as before.  --bandwidth-bucket-bytes is the depth
//                 of the bucket (default 1,500).  No message changes; the
//                 record gives the parameters and the time the sends were
//                 held back (totals.bandwidth_wait_s).
//   --verify      after the measured run, a TEST-ONLY step that is NOT part of
//                 the protocol and is not counted: the two processes send
//                 each other, in the clear, their MAC key shares, inputs and
//                 outputs, and each checks the generated material on the
//                 whole domain on both planes.  This destroys all secrecy and
//                 exists only to certify the benchmark outputs.
//
// The circuits are loaded from artifact/circuits/ (exported by
// python/bench/export_circuits.py); loading is not part of the measured time.
// All triples of the session (setup circuit and generation circuit) are drawn
// in ONE preprocessing request, so the session is a single bucketing
// population (2^-sigma, DESIGN.md 7.5).  The budget is computed with
// protocol::and_budget and the session runs in strict mode
// (allow_refill = false): a generation that did not fit into the reservation
// would be refused, not served by a second population.  The record carries
// "preprocessing".{"triple_requests", "triples_requested", "allow_refill",
// "bucketing_bound_log2"} and "checks"."strict_mode".
//
// The process prints a JSON record and writes it to --out: parameters,
// circuit statistics, bytes / rounds / seconds per phase (baseot, cotext,
// abit, laand, bucket, check from the preprocessing; input, online, output
// from the engine, where output = the two output MAC checks and the opening),
// a finer breakdown, totals, output sizes and the verification result.  The
// per-phase bytes, messages and rounds must sum to the channel totals; the
// process exits with status 3 if they do not or if the session was not a
// single, fully used bucketing population, with status 4 if the verification
// fails, and with status 1 on any other error.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "couplet/abb.hpp"
#include "couplet/channel.hpp"
#include "couplet/circuit.hpp"
#include "couplet/fss.hpp"
#include "couplet/gates.hpp"
#include "couplet/hashes.hpp"
#include "couplet/netem.hpp"
#include "couplet/prep.hpp"
#include "couplet/protocol.hpp"
#include "couplet/rng.hpp"
#include "couplet/session.hpp"

using namespace couplet;

// ------------------------------------------------------------------ channel with emulated latency and bandwidth
// DelayTcpChannel (couplet/netem.hpp): TCP endpoint with 4-byte length
// framing that emulates a one-way latency (--delay-ms) and a token-bucket
// bandwidth limit per direction (--bandwidth-mbps, --bandwidth-bucket-bytes)
// at application level.  Without either option no stamp is sent and nothing is
// throttled: the wire format is the plain framing of TcpChannel and of the
// Python package (tests/interop_peer.py).

// ------------------------------------------------------------------ ideal preprocessing in two processes
// NOT a protocol.  Each process builds the same prep::IdealDealer from a fixed
// public seed and takes its own half of every correlation; the other half is
// requested as well and thrown away, so the dealer keeps nothing.  Both
// processes issue the same requests in the same order, hence they hold
// consistent material.  Anybody who knows the seed knows every secret.
class SharedSeedPrep : public prep::Prep {
public:
    SharedSeedPrep(int role, uint64_t seed) : rng_(seed), dealer_(&rng_), own_(role, dealer_), other_(1 - role, dealer_) {}

    int role() const override { return own_.role(); }
    bool ideal() const override { return true; }
    bool aborted() const override { return own_.aborted(); }
    gf128 delta() const override { return own_.delta(); }
    const prep::PrepCounters& counters() const override { return own_.counters(); }
    void setup() override {}
    void abort() override {
        own_.abort();
        other_.abort();
    }
    prep::Shares random_bits(size_t n) override {
        prep::Shares s = own_.random_bits(n);
        other_.random_bits(n);
        return s;
    }
    prep::BdozPair random_bdoz(size_t n) override {
        prep::BdozPair p = own_.random_bdoz(n);
        other_.random_bdoz(n);
        return p;
    }
    prep::InputMasks input_masks(int owner, size_t n) override {
        prep::InputMasks m = own_.input_masks(owner, n);
        other_.input_masks(owner, n);
        return m;
    }
    prep::Triples and_triples(size_t n) override {
        prep::Triples t = own_.and_triples(n);
        other_.and_triples(n);
        return t;
    }

private:
    CtrPrgRng rng_;
    prep::IdealDealer dealer_;
    prep::IdealPrep own_, other_;
};

// ------------------------------------------------------------------ arguments
struct Args {
    int role = -1;
    std::string host = "127.0.0.1";
    int port = 47110;
    std::string task;
    unsigned n = 0, k = 64, s = 64;
    std::string beta = "public";
    bool auth = false;
    size_t batch = 1;
    std::string prep = "real";
    int sigma = 40;
    double delay_ms = 0.0;
    double bandwidth_mbps = 0.0;  // 0: no bandwidth limit
    double bucket_bytes = DelayTcpChannel::DEFAULT_BUCKET_BYTES;
    unsigned threads = 1;
    bool verify = false, quiet = false;
    std::string out, circuits;
    int timeout_ms = 60000;
};

static void usage() {
    std::printf("usage: couplet_party --role 0|1 [--host H] [--port P] --task dpf|dcf|gate-table|gate-cmp --n N [--k 64]\n"
                "                     [--beta public|secret] [--auth] [--batch B] [--prep real|ideal] [--sigma 40]\n"
                "                     [--delay-ms D] [--bandwidth-mbps R] [--bandwidth-bucket-bytes C] [--threads T]\n"
                "                     [--verify] [--out FILE.json] [--circuits DIR] [--timeout-ms MS] [--quiet]\n");
}

// The bandwidth options, shared with app_e2e: range checks and the text that the hello carries
// (empty without a limit, so that the hello and the session identifier of a run without a limit
// are those of the earlier versions and of tests/interop_peer.py).
static void check_bandwidth_args(double bandwidth_mbps, double bucket_bytes) {
    if (!(bandwidth_mbps >= 0) || bandwidth_mbps > 100000)
        throw std::invalid_argument("--bandwidth-mbps must be in 0..100000 (0: no limit)");
    if (!(bucket_bytes >= 1) || bucket_bytes > 1073741824.0)
        throw std::invalid_argument("--bandwidth-bucket-bytes must be in 1..2^30");
}
static std::string bandwidth_hello(double bandwidth_mbps, double bucket_bytes);

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        const std::string f = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + f);
            return argv[++i];
        };
        if (f == "--role") a.role = std::atoi(val().c_str());
        else if (f == "--host") a.host = val();
        else if (f == "--port") a.port = std::atoi(val().c_str());
        else if (f == "--task") a.task = val();
        else if (f == "--n") a.n = (unsigned)std::atoi(val().c_str());
        else if (f == "--k") a.k = (unsigned)std::atoi(val().c_str());
        else if (f == "--beta") a.beta = val();
        else if (f == "--auth") a.auth = true;
        else if (f == "--batch") a.batch = (size_t)std::atoll(val().c_str());
        else if (f == "--prep") a.prep = val();
        else if (f == "--sigma") a.sigma = std::atoi(val().c_str());
        else if (f == "--delay-ms") a.delay_ms = std::atof(val().c_str());
        else if (f == "--bandwidth-mbps") a.bandwidth_mbps = std::atof(val().c_str());
        else if (f == "--bandwidth-bucket-bytes") a.bucket_bytes = std::atof(val().c_str());
        else if (f == "--threads") a.threads = (unsigned)std::atoi(val().c_str());
        else if (f == "--verify") a.verify = true;
        else if (f == "--quiet") a.quiet = true;
        else if (f == "--out") a.out = val();
        else if (f == "--circuits") a.circuits = val();
        else if (f == "--timeout-ms") a.timeout_ms = std::atoi(val().c_str());
        else throw std::invalid_argument("unknown argument " + f);
    }
    if (a.role != 0 && a.role != 1) throw std::invalid_argument("--role must be 0 or 1");
    if (a.task != "dpf" && a.task != "dcf" && a.task != "gate-table" && a.task != "gate-cmp")
        throw std::invalid_argument("--task must be dpf, dcf, gate-table or gate-cmp");
    if (a.n < 1 || a.n > 40) throw std::invalid_argument("--n must be in 1..40");
    if (a.k < 1 || a.k > 64) throw std::invalid_argument("--k must be in 1..64");
    if (a.beta != "public" && a.beta != "secret") throw std::invalid_argument("--beta must be public or secret");
    if (a.batch < 1 || a.batch > 65536) throw std::invalid_argument("--batch must be in 1..65536");
    if (a.prep != "real" && a.prep != "ideal") throw std::invalid_argument("--prep must be real or ideal");
    if (a.sigma < 1 || a.sigma > 128) throw std::invalid_argument("--sigma must be in 1..128");
    if (a.delay_ms < 0 || a.delay_ms > 10000) throw std::invalid_argument("--delay-ms must be in 0..10000");
    check_bandwidth_args(a.bandwidth_mbps, a.bucket_bytes);
    if (a.port < 1 || a.port > 65535) throw std::invalid_argument("--port must be in 1..65535");
    if (a.threads < 1) a.threads = 1;
    const bool is_key = a.task == "dpf" || a.task == "dcf";
    if (is_key && a.auth) throw std::invalid_argument("--auth applies to gate tasks only");
    if (!is_key && a.beta != "public") throw std::invalid_argument("--beta applies to dpf and dcf only");
    return a;
}

// ------------------------------------------------------------------ small helpers
static double secs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
static std::string num(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.6f", x);
    return buf;
}
static std::string q(const std::string& s) { return json::quote(s); }

static std::string bandwidth_hello(double bandwidth_mbps, double bucket_bytes) {
    if (bandwidth_mbps <= 0) return "";
    return "|bandwidth_mbps=" + num(bandwidth_mbps) + "|bucket_bytes=" + num(bucket_bytes);
}
// The emulation parameters of the record ("params").
static std::string netem_params_json(double bandwidth_mbps, double bucket_bytes) {
    return ", \"bandwidth_mbps\": " + num(bandwidth_mbps) + ", \"bandwidth_bucket_bytes\": " +
           (bandwidth_mbps > 0 ? num(bucket_bytes) : std::string("null"));
}

static void put_u64(bytes& out, uint64_t v) {
    for (int i = 0; i < 8; i++) out.push_back((uint8_t)(v >> (8 * i)));
}
static void put_blob(bytes& out, const bytes& b) {
    put_u64(out, b.size());
    out.insert(out.end(), b.begin(), b.end());
}
struct Reader {
    const bytes& b;
    size_t pos;
    uint64_t u64() {
        if (b.size() - pos < 8) throw std::runtime_error("verification message is truncated");
        uint64_t v = 0;
        for (int i = 7; i >= 0; i--) v = (v << 8) | b[pos + (size_t)i];
        pos += 8;
        return v;
    }
    bytes blob() {
        const uint64_t n = u64();
        if (b.size() - pos < n) throw std::runtime_error("verification message is truncated");
        bytes r(b.begin() + (ptrdiff_t)pos, b.begin() + (ptrdiff_t)(pos + n));
        pos += (size_t)n;
        return r;
    }
};

// What a party reveals in the TEST-ONLY verification step.
struct Reveal {
    uint64_t delta = 0;
    std::vector<uint64_t> alpha, beta;
    std::vector<bytes> outputs;  // encoded keys or gate material, one per instance

    bytes encode() const {
        bytes out;
        put_u64(out, delta);
        put_u64(out, alpha.size());
        for (uint64_t a : alpha) put_u64(out, a);
        put_u64(out, beta.size());
        for (uint64_t b : beta) put_u64(out, b);
        put_u64(out, outputs.size());
        for (const bytes& o : outputs) put_blob(out, o);
        return out;
    }
    static Reveal decode(const bytes& data) {
        Reader r{data, 0};
        Reveal v;
        v.delta = r.u64();
        const uint64_t na = r.u64();
        if (na > 1000000) throw std::runtime_error("verification message is malformed");
        for (uint64_t i = 0; i < na; i++) v.alpha.push_back(r.u64());
        const uint64_t nb = r.u64();
        if (nb > 1000000) throw std::runtime_error("verification message is malformed");
        for (uint64_t i = 0; i < nb; i++) v.beta.push_back(r.u64());
        const uint64_t no = r.u64();
        if (no > 1000000) throw std::runtime_error("verification message is malformed");
        for (uint64_t i = 0; i < no; i++) v.outputs.push_back(r.blob());
        if (r.pos != data.size()) throw std::runtime_error("verification message has trailing bytes");
        return v;
    }
};

// Full-domain checks of a key pair on both planes with the unreduced Delta; the pair must also
// be the output of the plaintext Gen on the same inputs and roots.
static bool dpf_pair_ok(const fss::DpfKey& k0, const fss::DpfKey& k1, uint64_t alpha, uint64_t beta, u128 delta, unsigned threads,
                        uint64_t* points) {
    if (k0.b != 0 || k1.b != 1 || k0.n != k1.n || k0.k != k1.k || k0.s != k1.s) return false;
    const unsigned n = k0.n, w = k0.width();
    const std::pair<fss::DpfKey, fss::DpfKey> ref = fss::dpf_gen(alpha, beta, delta, n, k0.k, k0.s, k0.seed, k1.seed);
    if (!(ref.first == k0) || !(ref.second == k1)) return false;
    const u128 mask = u128_mask(w), bt = (delta * u128(beta)) & mask;
    std::vector<u128> y0, t0, y1, t1;
    fss::dpf_eval_all(k0, y0, t0, threads);
    fss::dpf_eval_all(k1, y1, t1, threads);
    bool ok = true;
    for (size_t x = 0; x < y0.size(); x++) {
        const bool hit = x == alpha;
        ok = ok && ((y0[x] + y1[x]) & mask) == (hit ? u128(beta) : u128(0)) && ((t0[x] + t1[x]) & mask) == (hit ? bt : u128(0));
    }
    *points += y0.size();
    return ok;
}
static bool dcf_pair_ok(const fss::DcfKey& k0, const fss::DcfKey& k1, uint64_t alpha, uint64_t beta, u128 delta, unsigned threads,
                        uint64_t* points) {
    if (k0.b != 0 || k1.b != 1 || k0.n != k1.n || k0.k != k1.k || k0.s != k1.s) return false;
    const unsigned n = k0.n, w = k0.width();
    const std::pair<fss::DcfKey, fss::DcfKey> ref = fss::dcf_gen(alpha, beta, delta, n, k0.k, k0.s, k0.seed, k1.seed);
    if (!(ref.first == k0) || !(ref.second == k1)) return false;
    const u128 mask = u128_mask(w), bt = (delta * u128(beta)) & mask;
    std::vector<u128> y0, t0, y1, t1;
    fss::dcf_eval_all(k0, y0, t0, threads);
    fss::dcf_eval_all(k1, y1, t1, threads);
    bool ok = true;
    for (size_t x = 0; x < y0.size(); x++) {
        const bool hit = x < alpha;
        ok = ok && ((y0[x] + y1[x]) & mask) == (hit ? u128(beta) : u128(0)) && ((t0[x] + t1[x]) & mask) == (hit ? bt : u128(0));
    }
    *points += y0.size();
    return ok;
}

// Gate material of both parties: offset sharing, keys at the indices that depend on r, [[w_c]].
static bool gate_pair_ok(const protocol::GateMaterial& g0, const protocol::GateMaterial& g1, const protocol::GateDescriptor& desc,
                         u128 delta, unsigned threads, uint64_t* points) {
    if (g0.role != 0 || g1.role != 1 || !(g0.desc == desc) || !(g1.desc == desc)) return false;
    const unsigned n = desc.n, w = desc.width(), m = desc.rp_bits();
    const u128 rp = (g0.rp_share + g1.rp_share) & u128_mask(m);
    const uint64_t r = u128_lo(rp) & gates::domain_mask(n);
    bool ok = true;
    if (desc.auth_input) ok = ok && ((g0.rp_tag_share + g1.rp_tag_share) & u128_mask(m)) == ((delta * rp) & u128_mask(m));
    if (desc.dpf_one) ok = ok && dpf_pair_ok(g0.dpf_key, g1.dpf_key, r, 1, delta, threads, points);
    for (size_t i = 0; i < desc.thresholds.size(); i++) {
        const protocol::Threshold& th = desc.thresholds[i];
        const u128 sum = u128(r) + th.c;
        const bool wc = !(sum < gates::domain_size(n));
        ok = ok && dcf_pair_ok(g0.dcf_keys[i], g1.dcf_keys[i], u128_lo(sum) & gates::domain_mask(n), th.payload_r ? r : 1, delta,
                               threads, points);
        const u128 wv = (g0.w_shares[i].value + g1.w_shares[i].value) & u128_mask(w);
        const u128 wt = (g0.w_shares[i].tag + g1.w_shares[i].tag) & u128_mask(w);
        ok = ok && wv == u128(wc ? 1 : 0) && wt == ((delta * wv) & u128_mask(w));
        if (th.payload_r) {
            const u128 rv = (g0.rw_shares[i].value + g1.rw_shares[i].value) & u128_mask(w);
            const u128 rt = (g0.rw_shares[i].tag + g1.rw_shares[i].tag) & u128_mask(w);
            ok = ok && rv == u128(wc ? r : 0) && rt == ((delta * rv) & u128_mask(w));
        }
    }
    for (size_t i = 0; i < desc.n_outmask; i++) {
        const u128 v = (g0.outmask_shares[i].value + g1.outmask_shares[i].value) & u128_mask(w);
        const u128 t = (g0.outmask_shares[i].tag + g1.outmask_shares[i].tag) & u128_mask(w);
        ok = ok && (v & u128_mask(desc.k)) == u128(0) && t == ((delta * v) & u128_mask(w));
    }
    return ok;
}

static std::string phase_json(const prep::PhaseStat& p) {
    return "{\"bytes_sent\":" + std::to_string(p.bytes_sent) + ",\"bytes_recv\":" + std::to_string(p.bytes_recv) +
           ",\"msgs_sent\":" + std::to_string(p.msgs_sent) + ",\"rounds\":" + std::to_string(p.rounds) + ",\"seconds\":" +
           num(p.seconds) + "}";
}

static prep::PhaseStat plus(prep::PhaseStat a, const prep::PhaseStat& b) {
    a += b;
    return a;
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

static int run(const Args& a) {
    typedef std::chrono::steady_clock clock;
    const bool is_key = a.task == "dpf" || a.task == "dcf";
    const protocol::Family family = a.task == "dcf" ? protocol::FAMILY_DCF : protocol::FAMILY_DPF;
    const protocol::BetaMode beta_mode = a.beta == "secret" ? protocol::BETA_SECRET : protocol::BETA_PUBLIC_ONE;
    protocol::GateDescriptor desc;
    std::string circuit_name;
    if (is_key) {
        circuit_name = protocol::key_circuit_name(family, a.n, a.k, a.s, protocol::INDEX_XOR, beta_mode);
    } else {
        desc = a.task == "gate-table" ? gates::table_descriptor(a.n, a.k, a.s, a.auth)
                                      : gates::comparison_descriptor(a.n, a.k, a.s, u128(((uint64_t)1 << a.n) / 3), a.auth);
        circuit_name = desc.name();
    }

    // Circuits (not part of the measured time).
    const clock::time_point t_load = clock::now();
    circuit::CircuitStore store(a.circuits);
    const std::string setup_name = "setup_s" + std::to_string(a.s);
    const std::shared_ptr<const circuit::Circuit> setup_c = store.get(setup_name), gen_c = store.get(circuit_name);
    const double load_s = secs(t_load, clock::now());
    const circuit::Stats st = gen_c->stats(), st_setup = setup_c->stats();
    // The triple budget of the session: one planned generation call (public parameters only).
    const std::vector<protocol::Planned> plan = {
        is_key ? protocol::planned_key(store, family, a.n, a.k, a.s, protocol::INDEX_XOR, beta_mode, a.batch)
               : protocol::planned_gate(store, desc, a.batch)};
    const size_t reserve = protocol::and_budget(plan);

    // Connection and an unmetered hello: public parameters and a nonce for the session identifier.
    const std::string params = "couplet-e2e/1|task=" + a.task + "|n=" + std::to_string(a.n) + "|k=" + std::to_string(a.k) +
                               "|s=" + std::to_string(a.s) + "|beta=" + a.beta + "|auth=" + (a.auth ? "1" : "0") + "|batch=" +
                               std::to_string(a.batch) + "|prep=" + a.prep + "|sigma=" + std::to_string(a.sigma) + "|delay_ms=" +
                               num(a.delay_ms) + bandwidth_hello(a.bandwidth_mbps, a.bucket_bytes) + "|verify=" +
                               (a.verify ? "1" : "0") + "|circuit=" + gen_c->digest_hex();
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
    const hashes::hash32 sid_hash = hashes::framed_sha256({"Couplet-e2e-sid", params, nonce[0], nonce[1]});
    const bytes sid(sid_hash.begin(), sid_hash.end());
    ch.reset_stats();

    // ---------------------------------------------------------------- the measured run
    const clock::time_point t0 = clock::now();
    std::unique_ptr<SharedSeedPrep> ideal;
    std::unique_ptr<Session> sesp;
    if (a.prep == "real") {
        sesp.reset(new Session(a.role, ch, sid, store, a.k, a.s, a.sigma));
        sesp->real_prep()->set_threads(a.threads);
    } else {
        ideal.reset(new SharedSeedPrep(a.role, 0x1dea1dea1dea1deaull));  // a public constant: NOT a protocol
        sesp.reset(new Session(a.role, ch, sid, store, *ideal, a.k, a.s));
    }
    Session& ses = *sesp;
    ses.set_allow_refill(false);  // strict mode: no second bucketing population behind the caller's back
    ses.setup(reserve);           // all triples of the session in ONE preprocessing request
    const clock::time_point t1 = clock::now();

    Reveal mine;
    std::vector<fss::DpfKey> dpf_keys;
    std::vector<fss::DcfKey> dcf_keys;
    std::vector<protocol::GateMaterial> mats;
    if (is_key) {
        for (size_t j = 0; j < a.batch; j++) {
            mine.alpha.push_back(sys.u64() & gates::domain_mask(a.n));
            if (beta_mode == protocol::BETA_SECRET) mine.beta.push_back(sys.u64() & gates::domain_mask(a.k));
        }
        if (family == protocol::FAMILY_DPF) dpf_keys = protocol::gen_dpf(ses, a.n, protocol::INDEX_XOR, beta_mode, mine.alpha, mine.beta);
        else dcf_keys = protocol::gen_dcf(ses, a.n, protocol::INDEX_XOR, beta_mode, mine.alpha, mine.beta);
    } else {
        mats = protocol::gen_gate(ses, desc, a.batch);
    }
    const clock::time_point t2 = clock::now();
    // ---------------------------------------------------------------- end of the measured run

    const ChannelStats total = ch.stats();
    const double emulated_wait = ch.emulated_wait_s();
    const double bandwidth_wait = ch.bandwidth_wait_s();
    for (const fss::DpfKey& key : dpf_keys) mine.outputs.push_back(key.to_bytes());
    for (const fss::DcfKey& key : dcf_keys) mine.outputs.push_back(key.to_bytes());
    for (const protocol::GateMaterial& m : mats) mine.outputs.push_back(m.to_bytes());
    mine.delta = ses.delta_share();
    size_t out_bytes_all = 0;
    for (const bytes& o : mine.outputs) out_bytes_all += o.size();

    // Phases: the preprocessing books its own; the engine's are in the session book.
    prep::PrepStats pst;
    if (ses.real_prep()) pst = ses.real_prep()->stats();
    abb::PhaseBook& book = ses.book();
    const prep::PhaseStat ph_input = plus(book.get("setup-input"), book.get("input"));
    const prep::PhaseStat ph_online = plus(book.get("setup-online"), book.get("online"));
    const prep::PhaseStat ph_output =
        plus(plus(book.get("setup-outcheck"), book.get("setup-output")), plus(book.get("outcheck"), book.get("output")));
    const prep::PhaseStat ph_dealer = plus(book.get("setup-prep"), book.get("prep"));  // ideal mode only: time, no traffic
    prep::PhaseStat sum = pst.total();
    sum += ph_input;
    sum += ph_online;
    sum += ph_output;
    sum += ph_dealer;
    const prep::PhaseStat book_total = book.total();
    const bool sums_ok = sum.bytes_sent == total.bytes_sent && sum.bytes_recv == total.bytes_recv && sum.rounds == total.rounds &&
                         sum.msgs_sent == total.msgs_sent && sum.msgs_recv == total.msgs_recv &&
                         pst.total().bytes_sent + book_total.bytes_sent == total.bytes_sent &&
                         pst.total().rounds + book_total.rounds == total.rounds && ph_dealer.bytes_sent == 0 && ph_dealer.rounds == 0;

    const unsigned bucket = ses.real_prep() ? ses.real_prep()->last_bucket_size() : 0;
    const uint64_t n_triples = (uint64_t)protocol::session_triples(store, plan, a.s);  // setup circuit + budget
    const bool single_population = ses.triple_requests() == 1 && ses.triples_requested() == n_triples;
    const bool all_used = ses.triples_used() == n_triples && ses.triples_unused() == 0;
    double bound_log2 = 0;
    const bool has_bound = ses.bucketing_bound_log2(&bound_log2);  // real preprocessing only
    const double wall = secs(t0, t2);

    // ---------------------------------------------------------------- TEST-ONLY verification
    // Not part of the protocol and not counted above: both processes reveal their MAC key
    // shares, inputs and outputs to each other in the clear and check the generated material.
    bool verified = false, correct = false, delta_big = false;
    uint64_t points = 0;
    double verify_s = 0;
    if (a.verify) {
        const clock::time_point tv = clock::now();
        const bytes mine_enc = mine.encode();
        // The peer reveals the same kind and number of objects: its message is about as long as this one.
        const Reveal theirs = Reveal::decode(ch.exchange(mine_enc, 2 * mine_enc.size() + 65536, "verification message too long"));
        const Reveal& r0 = a.role == 0 ? mine : theirs;
        const Reveal& r1 = a.role == 0 ? theirs : mine;
        const u128 delta = fss::make_delta(r0.delta, r1.delta);  // the integer Delta_0 + Delta_1, never reduced
        delta_big = (delta >> a.s) != u128(0);
        const unsigned vt = std::max(4u, a.threads);
        correct = r0.outputs.size() == a.batch && r1.outputs.size() == a.batch;
        for (size_t j = 0; correct && j < a.batch; j++) {
            if (is_key) {
                correct = r0.alpha.size() == a.batch && r1.alpha.size() == a.batch && r0.beta.size() == r1.beta.size() &&
                          (beta_mode == protocol::BETA_SECRET ? r0.beta.size() == a.batch : r0.beta.empty());
                if (!correct) break;
                const uint64_t alpha = r0.alpha[j] ^ r1.alpha[j];
                const uint64_t beta = beta_mode == protocol::BETA_SECRET ? ((r0.beta[j] + r1.beta[j]) & gates::domain_mask(a.k)) : 1;
                if (family == protocol::FAMILY_DPF)
                    correct = dpf_pair_ok(fss::DpfKey::from_bytes(r0.outputs[j]), fss::DpfKey::from_bytes(r1.outputs[j]), alpha, beta,
                                          delta, vt, &points);
                else
                    correct = dcf_pair_ok(fss::DcfKey::from_bytes(r0.outputs[j]), fss::DcfKey::from_bytes(r1.outputs[j]), alpha, beta,
                                          delta, vt, &points);
            } else {
                correct = gate_pair_ok(protocol::GateMaterial::from_bytes(r0.outputs[j]),
                                       protocol::GateMaterial::from_bytes(r1.outputs[j]), desc, delta, vt, &points);
            }
        }
        verified = true;
        verify_s = secs(tv, clock::now());
    }

    // ---------------------------------------------------------------- the record
    const char* kind = !is_key ? "gate material" : (family == protocol::FAMILY_DPF ? "dpf key" : "dcf key");
    std::string rec = "{\n";
    rec += " \"schema\": \"couplet.e2e_cpp/1\",\n";
    rec += " \"implementation\": \"C++ (couplet, native)\",\n";
    rec += " \"compiler\": " + q(compiler_name()) + ",\n";
    rec += " \"timestamp_utc\": " + q(utc_now()) + ",\n";
    rec += " \"role\": " + std::to_string(a.role) + ",\n";
    rec += " \"params\": {\"task\": " + q(a.task) + ", \"n\": " + std::to_string(a.n) + ", \"k\": " + std::to_string(a.k) +
           ", \"s\": " + std::to_string(a.s) + ", \"index_mode\": \"xor\", \"beta\": " + q(is_key ? a.beta : "public") +
           ", \"auth_input\": " + (a.auth ? "true" : "false") + ", \"batch\": " + std::to_string(a.batch) + ", \"prep\": " + q(a.prep) +
           ", \"sigma\": " + std::to_string(a.sigma) + ", \"delay_ms\": " + num(a.delay_ms) +
           netem_params_json(a.bandwidth_mbps, a.bucket_bytes) + ", \"threads\": " +
           std::to_string(a.threads) + ", \"mode\": \"two-process\", \"transport\": \"tcp\", \"host\": " + q(a.host) + ", \"port\": " +
           std::to_string(a.port) + "},\n";
    rec += std::string(" \"ideal_preprocessing\": ") + (a.prep == "ideal" ? "true" : "false") + ",\n";
    rec += " \"circuit\": {\"name\": " + q(circuit_name) + ", \"digest\": " + q(gen_c->digest_hex()) + ", \"binary_bytes\": " +
           std::to_string(gen_c->encoded_bytes) + ", \"stats\": " + st.json() + "},\n";
    rec += " \"session_setup_circuit\": {\"name\": " + q(setup_name) + ", \"digest\": " + q(setup_c->digest_hex()) + ", \"n_and\": " +
           std::to_string(st_setup.n_and) + ", \"and_depth\": " + std::to_string(st_setup.and_depth) + ", \"n_gates\": " +
           std::to_string(st_setup.n_gates) + "},\n";
    rec += " \"preprocessing\": {\"triples\": " + std::to_string(n_triples) + ", \"triples_generation_circuit\": " +
           std::to_string(reserve) + ", \"triples_session_setup\": " + std::to_string(st_setup.n_and) + ", \"triple_requests\": " +
           std::to_string(ses.triple_requests()) + ", \"triples_requested\": " + std::to_string(ses.triples_requested()) +
           ", \"allow_refill\": " + (ses.allow_refill() ? "true" : "false") + ", \"bucketing_bound_log2\": " +
           (has_bound ? num(bound_log2) : std::string("null")) + ", \"bucket_size\": " + std::to_string(bucket) + ", \"leaky_triples\": " +
           std::to_string(n_triples * bucket) + ", \"model_bytes_per_and\": " + std::to_string(96 * bucket) + ", \"input_masks\": " +
           std::to_string(ses.prep().counters().input_masks) + "},\n";
    rec += " \"phases\": {";
    for (int p = 0; p < prep::PH_COUNT; p++) rec += q(prep::phase_name(p)) + ": " + phase_json(pst.phase[p]) + ", ";
    rec += "\"input\": " + phase_json(ph_input) + ", \"online\": " + phase_json(ph_online) + ", \"output\": " + phase_json(ph_output);
    if (a.prep == "ideal") rec += ", \"ideal_dealer\": " + phase_json(ph_dealer);
    rec += "},\n";
    rec += " \"phases_note\": \"baseot .. check: preprocessing; input / online / output: engine, session-setup circuit and "
           "generation circuit together; output = the two BoolChecks of Output and the opening; seconds are wall-clock at this party "
           "and include waiting for the peer\",\n";
    rec += " \"phases_detail\": {";
    for (size_t i = 0; i < book.size(); i++) rec += std::string(i ? ", " : "") + q(book.name(i)) + ": " + phase_json(book.stat(i));
    rec += "},\n";
    rec += " \"totals\": {\"bytes_sent\": " + std::to_string(total.bytes_sent) + ", \"bytes_recv\": " + std::to_string(total.bytes_recv) +
           ", \"bytes\": " + std::to_string(total.bytes_sent + total.bytes_recv) + ", \"msgs_sent\": " + std::to_string(total.msgs_sent) +
           ", \"rounds\": " + std::to_string(total.rounds) + ", \"seconds\": " + num(wall) + ", \"setup_s\": " + num(secs(t0, t1)) +
           ", \"generation_s\": " + num(secs(t1, t2)) + ", \"online_s\": " + num(ses.last_gen.stats.online_s) +
           ", \"phase_seconds_sum\": " + num(sum.seconds) + ", \"emulated_wait_s\": " + num(emulated_wait) +
           ", \"bandwidth_wait_s\": " + num(bandwidth_wait) + ", \"load_circuits_s\": " +
           num(load_s) + ", \"bytes_per_and_gate\": " + num((double)(total.bytes_sent + total.bytes_recv) / (double)n_triples) +
           ", \"bytes_per_instance\": " + num((double)(total.bytes_sent + total.bytes_recv) / (double)a.batch) +
           ", \"seconds_per_instance\": " + num(wall / (double)a.batch) + "},\n";
    rec += " \"totals_note\": \"bytes = payload bytes in both directions (sent + received by this party); seconds = this party's "
           "wall-clock time from the first protocol message to the certified outputs, both processes on one machine\",\n";
    rec += " \"generation\": " + ses.last_gen.stats.json() + ",\n";
    rec += " \"session_setup_evaluation\": " + ses.setup_stats().json() + ",\n";
    rec += " \"outputs\": {\"kind\": " + q(kind) + ", \"instances\": " + std::to_string(a.batch) +
           ", \"bytes_per_instance_and_party\": " + std::to_string(mine.outputs.empty() ? 0 : mine.outputs[0].size()) +
           ", \"bytes_all_instances_per_party\": " + std::to_string(out_bytes_all) + "},\n";
    rec += std::string(" \"verification\": {\"performed\": ") + (verified ? "true" : "false") + ", \"outputs_correct\": " +
           (verified ? (correct ? "true" : "false") : "null") + ", \"instances_checked\": " + std::to_string(verified ? a.batch : 0) +
           ", \"points_checked\": " + std::to_string(points) + ", \"delta_at_least_2_pow_s\": " +
           (verified ? (delta_big ? "true" : "false") : "null") + ", \"seconds\": " + num(verify_s) +
           ", \"note\": \"TEST ONLY, not part of the protocol and not counted in any total: after the measured run the two "
           "processes exchange their MAC key shares, inputs and outputs in the clear and check every key pair on the whole domain "
           "on both planes with the unreduced Delta, and against the plaintext Gen with the same roots\"},\n";
    rec += std::string(" \"checks\": {\"phase_sums_ok\": ") + (sums_ok ? "true" : "false") + ", \"single_bucketing_population\": " +
           (single_population ? "true" : "false") + ", \"all_triples_used\": " + (all_used ? "true" : "false") +
           ", \"strict_mode\": " + (ses.allow_refill() ? "false" : "true") + ", \"session_active\": " +
           (ses.active() ? "true" : "false") + "}\n";
    rec += "}\n";

    if (!a.out.empty()) {
        std::ofstream f(a.out, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + a.out);
        f << rec;
    }
    if (a.quiet) {
        std::printf("party %d: %s batch %zu: %llu bytes (both directions), %llu rounds, %.3f s%s\n", a.role, circuit_name.c_str(),
                    a.batch, (unsigned long long)(total.bytes_sent + total.bytes_recv), (unsigned long long)total.rounds, wall,
                    verified ? (correct ? ", outputs verified" : ", VERIFICATION FAILED") : "");
    } else {
        std::fputs(rec.c_str(), stdout);
    }
    std::fflush(stdout);
    if (!sums_ok) {
        std::fprintf(stderr, "error: the per-phase counters do not sum to the channel totals\n");
        return 3;
    }
    if (!single_population || !all_used) {
        std::fprintf(stderr, "error: the session did not consist of a single, fully used bucketing population\n");
        return 3;
    }
    if (verified && !correct) {
        std::fprintf(stderr, "error: the generated outputs are NOT correct\n");
        return 4;
    }
    return 0;
}

int main(int argc, char** argv) {
    Args a;
    try {
        a = parse(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage();
        return 2;
    }
    if (a.circuits.empty()) a.circuits = circuit::dirname(argv[0]) + "/../../circuits";
    try {
        return run(a);
    } catch (const ProtocolAbort& e) {
        std::fprintf(stderr, "party %d: PROTOCOL ABORT: %s\n", a.role, e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "party %d: error: %s\n", a.role, e.what());
    }
    return 1;
}
