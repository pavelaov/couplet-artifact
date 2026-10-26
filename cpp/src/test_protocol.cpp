// test_protocol.cpp -- tests of session.hpp, protocol.hpp, spdz2k.hpp,
// gates.hpp and state.hpp (DESIGN.md sections 2.3, 4, 5, 6): Pi_Gen for DPF
// keys, DCF keys and gate bundles, the SPDZ2k openings with ArithCheck, the
// local gate evaluations and the state machine.
//
// The generation circuits are the files exported by
// python/bench/export_circuits.py into artifact/circuits/ (the C++ code
// compiles nothing).
//
//   vectors     values produced by the Python reference
//               (tests/vectors_protocol.json): gate material encodings parse,
//               re-encode to the same bytes and evaluate correctly; ArithCheck
//               coefficients; sub-session identifiers; evaluation headers of
//               the engine; circuit names and descriptor JSON.
//   spdz2k      AShare arithmetic; honest openings; every deviation of a
//               party (wrong value share, wrong MAC share, wrong zeta) makes
//               the honest party abort; releases of non-exact values need an
//               output mask; Lemma 2 pass rate for a tiny s.
//   gates       local evaluation of every catalogue gate on dealer-made
//               material (gates::deal_gate, NOT a protocol): exhaustive over
//               r, x and c for small n, both planes, with Delta >= 2^s.
//   state       identifiers, one-time use of gates, persistence, abort; the
//               ABORTED state on disk (state file and create-only marker).
//   budget      bucketing populations: the AND-budget helpers against the
//               circuit manifest; refill mode and strict mode of the session;
//               the driver is strict by default and refuses a generation or
//               an on-demand output mask that the reserved pool cannot
//               serve, before any message and any change of state.
//   mask        Mask of a shared input on the wire: the bytes of an honest
//               run recomputed from the specification (commit-then-open
//               under the sub-session identifier of the Mask command), two
//               rounds; a rushing, mirroring or otherwise deviating party
//               makes the honest party abort.
//   gen ideal   Pi_Gen for DPF and DCF, n in {4, 8}, public and secret
//               payload, XOR and additive index shares, k in {32, 64}, batch
//               1 and 3, ideal preprocessing: every key pair is bit-identical
//               to fss::dpf_gen / dcf_gen on the reconstructed inputs with
//               the keys' own roots and correct on the whole domain on both
//               planes with the UNREDUCED session Delta (one session is
//               forced to Delta_0 + Delta_1 >= 2^s).
//   gen real    the same for n in {4, 8} with the real preprocessing in one
//               session with a single bucketing population; exact accounting.
//   gates e2e   table gate and comparison gate end to end (Gate, Mask, Apply,
//               Open) for ALL x at n = 8, with a shared input and with an
//               authenticated wire, ideal preprocessing; a two-gate chain;
//               ReLU for n = k = 8 with an output mask.
//   open order  Open verifies the deferred openings before it generates an
//               output mask: the order of the messages, the refusal in
//               strict mode before any message, and a wrong wire-mode share
//               that ends the session in the first check of Open.
//   abort       a wrong share in a masked wire opening or in a release aborts
//               the whole session, also when the checker is driven directly
//               (abort hook); a tampered generation aborts and certifies
//               nothing; key material is not handed out by a dead session.
//   dpf16 real  one run of DPF n = 16, public payload, real preprocessing.
//
// Both parties run as two threads of this process (run_two_party).
// Usage: test_protocol [--circuits DIR] [--tests DIR] [--quick]
//        test_protocol [--circuits DIR] --interop ROLE PORT
//   --quick    fewer x in the end-to-end gate tests, no ReLU circuit, no n = 16.
//   --interop  run ONE party of a fixed script of the online layer over TCP
//              with the real preprocessing, to be paired with another
//              implementation (tests/interop_peer.py --online runs the Python
//              reference); see run_interop.
// Exit code 0 iff every check passes.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "couplet/abb.hpp"
#include "couplet/channel.hpp"
#include "couplet/circuit.hpp"
#include "couplet/fss.hpp"
#include "couplet/gates.hpp"
#include "couplet/hashes.hpp"
#include "couplet/prep.hpp"
#include "couplet/protocol.hpp"
#include "couplet/rng.hpp"
#include "couplet/session.hpp"
#include "couplet/spdz2k.hpp"
#include "couplet/state.hpp"
#include "couplet/twoparty.hpp"

using namespace couplet;
using circuit::BitMat;
using circuit::Circuit;
using gates::GateSpec;
using protocol::GateDescriptor;
using protocol::GateMaterial;
using spdz2k::AShare;

// ------------------------------------------------------------ harness
static long long g_pass = 0, g_fail = 0;
static bool g_quick = false;
static std::string g_tests;

static void check(bool ok, const std::string& what) {
    if (ok) {
        g_pass++;
    } else {
        g_fail++;
        if (g_fail <= 60) std::printf("  FAIL: %s\n", what.c_str());
    }
}
// `n` comparisons of which `good` held.
static void check_many(long long good, long long n, const std::string& what) {
    g_pass += good;
    if (good != n) {
        g_fail += n - good;
        std::printf("  FAIL: %s (%lld of %lld)\n", what.c_str(), good, n);
    }
}

struct Section {
    const char* name;
    long long pass0, fail0;
    std::chrono::steady_clock::time_point t0;
    explicit Section(const char* n) : name(n), pass0(g_pass), fail0(g_fail), t0(std::chrono::steady_clock::now()) {}
    ~Section() {
        const long long f = g_fail - fail0, p = g_pass - pass0;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("[%s] %s (%lld checks, %.2f s)\n", f ? "FAIL" : "PASS", name, p + f, s);
        std::fflush(stdout);
    }
};

template <class F>
static bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
    }
    return false;
}
template <class E, class F>
static bool throws(F&& f) {
    try {
        f();
    } catch (const E&) {
        return true;
    } catch (...) {
    }
    return false;
}

static std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static u128 parse_dec(const std::string& s) {
    json::Value v;
    v.type = json::Value::NUM;
    v.text = s;
    return v.as_u128();
}

static uint64_t mask_bits(unsigned n) { return gates::domain_mask(n); }

static std::string hex_of(const uint8_t* p, size_t n) { return circuit::hex(bytes(p, p + n)); }
static std::string hex_of(const bytes& b) { return circuit::hex(b); }

// A fresh, uniquely named directory for the files of one test, outside the
// source tree (parallel runs of this program do not share it).  The caller
// removes it.
static std::filesystem::path unique_temp_dir(const char* tag) {
    SystemRng sys;
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / (std::string("couplet_") + tag + "_" + hex_of(sys.get(8)));
    std::filesystem::create_directories(dir);
    return dir;
}

// Channel wrapper that records every message sent and received through it
// (tests of wire formats).
class RecordChannel : public Channel {
public:
    explicit RecordChannel(Channel& inner) : Channel(inner.party()), inner_(&inner) {}
    std::vector<bytes> sent, received;

protected:
    void send_raw(const uint8_t* p, size_t n) override {
        sent.push_back(bytes(p, p + n));
        inner_->send(sent.back());
    }
    bytes recv_raw() override {
        received.push_back(inner_->recv());
        return received.back();
    }
    bytes recv_raw_bounded(size_t max_len, bool from_caller) override {
        bytes b = from_caller ? inner_->recv(max_len) : inner_->recv();
        if (b.size() > max_len) throw MessageTooLong(b.size(), max_len);
        received.push_back(b);
        return b;
    }

private:
    Channel* inner_;
};

// (value, tag) of two shares; checks tag = Delta * value mod 2^bits.
static bool mac_ok(const AShare& a, const AShare& b, u128 delta, u128* value = nullptr) {
    const std::pair<u128, u128> r = spdz2k::reconstruct(a, b);
    if (value) *value = r.first;
    return r.second == ((delta * r.first) & u128_mask(a.bits));
}

// ------------------------------------------------------------ key verification
// Full-domain check of a DPF key pair on both planes against (alpha, beta, Delta);
// the keys must also be the output of the plaintext Gen with their own roots.
static void verify_dpf(const fss::DpfKey& k0, const fss::DpfKey& k1, uint64_t alpha, uint64_t beta, u128 delta, unsigned n,
                       unsigned k, unsigned s, const std::string& what, unsigned threads = 1) {
    const unsigned w = k + s;
    const u128 mask = u128_mask(w), beta_t = (delta * u128(beta)) & mask;
    check(k0.b == 0 && k1.b == 1 && k0.n == n && k1.n == n && k0.k == k && k0.s == s, what + ": key parameters");
    const std::pair<fss::DpfKey, fss::DpfKey> ref = fss::dpf_gen(alpha, beta, delta, n, k, s, k0.seed, k1.seed);
    check(ref.first == k0 && ref.second == k1, what + ": keys equal the plaintext Gen with the same roots");
    check(k0.to_bytes() == ref.first.to_bytes() && k0.size_bytes() == fss::dpf_key_size(n, k, s), what + ": key encoding");
    std::vector<u128> y0, t0, y1, t1;
    fss::dpf_eval_all(k0, y0, t0, threads);
    fss::dpf_eval_all(k1, y1, t1, threads);
    long long good_v = 0, good_t = 0;
    const size_t size = (size_t)1 << n;
    for (size_t x = 0; x < size; x++) {
        const bool hit = x == alpha;
        good_v += ((y0[x] + y1[x]) & mask) == (hit ? u128(beta) : u128(0));
        good_t += ((t0[x] + t1[x]) & mask) == (hit ? beta_t : u128(0));
    }
    check_many(good_v, (long long)size, what + ": value plane on the whole domain");
    check_many(good_t, (long long)size, what + ": tag plane on the whole domain");
    const fss::Share p0 = fss::dpf_eval(k0, alpha), p1 = fss::dpf_eval(k1, alpha);
    check(p0.y == y0[(size_t)alpha] && p1.tau == t1[(size_t)alpha], what + ": point Eval equals EvalAll");
}

static void verify_dcf(const fss::DcfKey& k0, const fss::DcfKey& k1, uint64_t alpha, uint64_t beta, u128 delta, unsigned n,
                       unsigned k, unsigned s, const std::string& what, unsigned threads = 1) {
    const unsigned w = k + s;
    const u128 mask = u128_mask(w), beta_t = (delta * u128(beta)) & mask;
    check(k0.b == 0 && k1.b == 1 && k0.n == n && k1.n == n && k0.k == k && k0.s == s, what + ": key parameters");
    const std::pair<fss::DcfKey, fss::DcfKey> ref = fss::dcf_gen(alpha, beta, delta, n, k, s, k0.seed, k1.seed);
    check(ref.first == k0 && ref.second == k1, what + ": keys equal the plaintext Gen with the same roots");
    check(k1.to_bytes() == ref.second.to_bytes() && k1.size_bytes() == fss::dcf_key_size(n, k, s), what + ": key encoding");
    std::vector<u128> y0, t0, y1, t1;
    fss::dcf_eval_all(k0, y0, t0, threads);
    fss::dcf_eval_all(k1, y1, t1, threads);
    long long good_v = 0, good_t = 0;
    const size_t size = (size_t)1 << n;
    for (size_t x = 0; x < size; x++) {
        const bool hit = x < alpha;
        good_v += ((y0[x] + y1[x]) & mask) == (hit ? u128(beta) : u128(0));
        good_t += ((t0[x] + t1[x]) & mask) == (hit ? beta_t : u128(0));
    }
    check_many(good_v, (long long)size, what + ": value plane on the whole domain");
    check_many(good_t, (long long)size, what + ": tag plane on the whole domain");
    const uint64_t x = alpha ? alpha - 1 : 0;
    const fss::Share p0 = fss::dcf_eval(k0, x), p1 = fss::dcf_eval(k1, x);
    check(p0.y == y0[(size_t)x] && p1.tau == t1[(size_t)x], what + ": point Eval equals EvalAll");
}

// ------------------------------------------------------------ vectors
static void test_vectors(circuit::CircuitStore& store) {
    Section sec("vectors (Python reference)");
    const json::Value doc = json::parse(read_text(g_tests + "/vectors_protocol.json"));

    // Gate material produced by gates.deal_gate.
    for (const json::Value& e : doc.at("gate_materials").items) {
        const std::string name = e.at("name").str();
        const bytes b0 = circuit::unhex(e.at("p0").str()), b1 = circuit::unhex(e.at("p1").str());
        const GateMaterial m0 = GateMaterial::from_bytes(b0), m1 = GateMaterial::from_bytes(b1);
        check(m0.to_bytes() == b0 && m1.to_bytes() == b1, name + ": re-encoding gives the reference bytes");
        check(m0.desc.to_json() == e.at("descriptor_json").str() && m0.desc == m1.desc, name + ": canonical descriptor JSON");
        check(m0.desc.name() == e.at("circuit_name").str(), name + ": circuit name");
        check(GateDescriptor::from_json(e.at("descriptor_json").str()) == m0.desc, name + ": descriptor parses");
        check(m0.role == 0 && m1.role == 1 && m0.size_bytes() == b0.size(), name + ": roles and size");
        const u128 delta = parse_dec(e.at("delta0").str()) + parse_dec(e.at("delta1").str());
        const unsigned n = m0.n(), k = m0.k(), s = m0.s(), w = k + s, m = m0.rp_bits();
        const uint64_t r = u128_lo(parse_dec(e.at("r").str()));
        const u128 rp = (m0.rp_share + m1.rp_share) & u128_mask(m);
        check((u128_lo(rp) & mask_bits(n)) == r, name + ": offset r");
        if (m0.auth_input())
            check(((m0.rp_tag_share + m1.rp_tag_share) & u128_mask(m)) == ((delta * rp) & u128_mask(m)), name + ": tag of r'");
        for (size_t i = 0; i < m0.outmask_shares.size(); i++) {
            const u128 v = (m0.outmask_shares[i].value + m1.outmask_shares[i].value) & u128_mask(w);
            const u128 t = (m0.outmask_shares[i].tag + m1.outmask_shares[i].tag) & u128_mask(w);
            check((v & u128_mask(k)) == u128(0) && t == ((delta * v) & u128_mask(w)), name + ": output mask is [[2^k rho]]");
        }
        // Evaluate at a few masked inputs.
        gates::GateEval g0(m0), g1(m1);
        CtrPrgRng rng(11);
        long long good = 0, total = 0;
        for (int t = 0; t < 24; t++) {
            const uint64_t x = (t == 0 ? 0 : t == 1 ? mask_bits(n) : rng.u64()) & mask_bits(n);
            const uint64_t x_hat = (x + r) & mask_bits(n);
            u128 value;
            if (m0.desc.dpf_one) {
                total += 2;
                good += mac_ok(gates::zero_test(g0, x_hat), gates::zero_test(g1, x_hat), delta, &value) && value == u128(x == 0 ? 1 : 0);
                const uint64_t c = rng.u64() & mask_bits(n);
                good += mac_ok(gates::equality(g0, x_hat, c), gates::equality(g1, x_hat, c), delta, &value) && value == u128(x == c ? 1 : 0);
            }
            for (size_t i = 0; i < m0.desc.thresholds.size(); i++) {
                const protocol::Threshold& th = m0.desc.thresholds[i];
                const u128 sum = u128(r) + th.c;
                const bool wc = !(sum < gates::domain_size(n));
                const uint64_t idx = u128_lo(sum) & mask_bits(n);
                const uint64_t payload = th.payload_r ? r : 1;
                total += 2;
                good += mac_ok(g0.dcf(th.c, th.payload_r, x_hat), g1.dcf(th.c, th.payload_r, x_hat), delta, &value) &&
                        value == u128(x_hat < idx ? payload : 0);
                good += mac_ok(g0.w(th.c, th.payload_r), g1.w(th.c, th.payload_r), delta, &value) && value == u128(wc ? 1 : 0);
                if (th.payload_r) {
                    total++;
                    good += mac_ok(g0.rw(th.c), g1.rw(th.c), delta, &value) && value == u128(wc ? r : 0);
                }
            }
        }
        check_many(good, total, name + ": local evaluations reconstruct with correct tags");
        // Truncations and mutations of the encoding are rejected.
        bool all = true;
        for (size_t cut : {(size_t)0, (size_t)8, (size_t)12, b0.size() / 2, b0.size() - 1})
            all = all && throws_invalid([&] { GateMaterial::from_bytes(bytes(b0.begin(), b0.begin() + (ptrdiff_t)cut)); });
        bytes more = b0;
        more.push_back(0);
        all = all && throws_invalid([&] { GateMaterial::from_bytes(more); });
        bytes role = b0;
        role[8] = 2;
        all = all && throws_invalid([&] { GateMaterial::from_bytes(role); });
        bytes magic = b0;
        magic[7] = '2';
        all = all && throws_invalid([&] { GateMaterial::from_bytes(magic); });
        if (m0.rp_bits() % 8) {  // a share above the modulus
            bytes big = b0;
            const size_t pos = 13 + m0.desc.to_json().size() + (m0.rp_bits() + 7) / 8 - 1;
            big[pos] |= 0x80;
            all = all && throws_invalid([&] { GateMaterial::from_bytes(big); });
        }
        if (!m0.outmask_shares.empty()) {  // an output mask that is not a multiple of 2^k
            bytes odd = b0;
            odd[b0.size() - 2 * ((w + 7) / 8)] |= 1;
            all = all && throws_invalid([&] { GateMaterial::from_bytes(odd); });
        }
        check(all, name + ": malformed encodings are rejected");
    }

    // Descriptor names of the exported circuits.
    check(protocol::key_circuit_name(protocol::FAMILY_DPF, 16, 64, 64, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE) ==
                  "dpf_n16_k64_s64_xor_one" &&
              protocol::key_circuit_name(protocol::FAMILY_DCF, 4, 32, 64, protocol::INDEX_ADD, protocol::BETA_SECRET) ==
                  "dcf_n4_k32_s64_add_secret",
          "key circuit names");
    check(gates::table_descriptor(12, 64, 64).name() == "gate_n12_k64_s64_shared_dpf" &&
              gates::comparison_descriptor(16, 64, 64, u128(21845), true).name() == "gate_n16_k64_s64_auth_t21845o_t0o" &&
              gates::relu_large_descriptor(8, 64).name() == "gate_n8_k8_s64_auth_t128o_t0o_t128r_t0r_om1" &&
              gates::offset_descriptor(1, 64, 64, false, 1).name() == "gate_n1_k64_s64_shared_om1",
          "gate circuit names");
    for (const char* nm : {"setup_s64", "dpf_n8_k64_s64_xor_one", "dcf_n8_k64_s64_xor_secret", "gate_n8_k64_s64_shared_dpf",
                           "gate_n8_k64_s64_auth_dpf", "gate_n8_k64_s64_shared_t85o_t0o", "gate_n8_k64_s64_auth_t85o_t0o",
                           "gate_n1_k64_s64_shared_om1"})
        check(store.has(nm), std::string("exported circuit ") + nm);
    check(throws_invalid([] { GateDescriptor(8, 4, 64, false, true, {}, 0); }) &&       // n > k
              throws_invalid([] { GateDescriptor(8, 64, 64, false, false, {protocol::Threshold(u128(257), false)}, 0); }) &&
              throws_invalid([] { GateDescriptor::from_json("{\"kind\":\"dpf\"}"); }) &&
              !throws_invalid([] { GateDescriptor(8, 64, 64, false, false, {protocol::Threshold(u128(256), false)}, 0); }),
          "descriptor validation");

    // ArithCheck coefficients.
    for (const json::Value& e : doc.at("check_coefficients").items) {
        const std::vector<uint64_t> chi = spdz2k::check_coefficients(circuit::unhex(e.at("seed").str()), (unsigned)e.at("bits").as_u64(),
                                                                     (size_t)e.at("count").as_u64(), (unsigned)e.at("s").as_u64());
        bool ok = chi.size() == e.at("chi").items.size();
        for (size_t i = 0; ok && i < chi.size(); i++) ok = u128(chi[i]) == parse_dec(e.at("chi").items[i].str());
        check(ok, "check_coefficients for s = " + std::to_string(e.at("s").as_u64()));
    }

    // Sub-session identifiers and evaluation headers of the engine: capture the messages.
    const bytes sid = circuit::unhex(doc.at("sub_sessions").at("sid").str());
    {
        const hashes::hash32 a = hashes::framed_sha256({"Couplet-abb-sid", sid});
        const hashes::hash32 b = hashes::framed_sha256({"Couplet-arith-sid", sid});
        check(circuit::hex(a.data(), 32) == doc.at("sub_sessions").at("sid_abb").str() &&
                  circuit::hex(b.data(), 32) == doc.at("sub_sessions").at("sid_arith").str(),
              "sub-session identifiers");
    }
    circuit::CircuitStore tstore(g_tests + "/circuits");
    std::shared_ptr<const Circuit> c5 = tstore.get("rnd05"), ca = tstore.get("chain_a"), cb = tstore.get("chain_b");
    prep::IdealDealer dealer;
    CtrPrgRng rng(12);
    auto rand_in = [&](const Circuit& c, int role, size_t batch) {
        abb::Inputs in;
        for (const circuit::Decl& d : c.inputs) {
            if (d.owner != (uint32_t)role) continue;
            BitMat m(d.wires.size(), batch);
            rng.fill_bits(m.v.data(), m.v.size());
            in[d.label] = m;
        }
        return in;
    };
    const abb::Inputs in5[2] = {rand_in(*c5, 0, 3), rand_in(*c5, 1, 3)}, ina[2] = {rand_in(*ca, 0, 2), rand_in(*ca, 1, 2)},
                      inb[2] = {rand_in(*cb, 0, 2), rand_in(*cb, 1, 2)};
    auto party = [&](int role) {
        return [&, role](Channel& raw) {
            std::vector<bytes> sent;
            std::map<size_t, TamperChannel::Rule> rules;
            for (size_t i = 0; i < 400; i++)
                rules[i] = [&sent](const bytes& m) {
                    sent.push_back(m);
                    return m;
                };
            TamperChannel ch(raw, rules);
            prep::IdealPrep prep(role, dealer);
            abb::Engine eng(role, ch, prep, sid);
            std::vector<std::string> headers;
            size_t at = ch.messages_sent();
            eng.eval_circuit(*c5, in5[role], abb::Bound(), 3);
            headers.push_back(circuit::hex(sent.at(at).data(), 32));
            at = ch.messages_sent();
            const abb::EvalResult ra = eng.eval_circuit(*ca, ina[role], abb::Bound(), 2);
            headers.push_back(circuit::hex(sent.at(at).data(), 32));
            at = ch.messages_sent();
            abb::Bound bound;
            bound["keep"] = ra.retained.at("keep");
            eng.eval_circuit(*cb, inb[role], bound, 2);
            headers.push_back(circuit::hex(sent.at(at).data(), 32));
            return headers;
        };
    };
    const auto pr = run_two_party(party(0), party(1));
    check(pr.first.ok && pr.second.ok, "header capture run (" + pr.first.error + pr.second.error + ")");
    if (pr.first.ok && pr.second.ok) {
        const std::vector<json::Value>& hs = doc.at("headers").items;
        for (size_t i = 0; i < hs.size(); i++)
            check(pr.first.value.at(i) == hs[i].at("header").str() && pr.second.value.at(i) == hs[i].at("header").str(),
                  "evaluation header of " + hs[i].at("circuit").str() + " equals the reference");
    }
}

// ------------------------------------------------------------ spdz2k
// A checker that deviates in one of the three protected steps.
class CheatChecker : public spdz2k::ArithChecker {
public:
    enum Mode { HONEST, VALUE, MAC, ZETA, ZETA_GUESS };
    CheatChecker(int role, Channel& ch, const bytes& sid, uint64_t delta, unsigned s, Mode mode, u128 error, uint64_t target = 0,
                 uint64_t guess = 0)
        : ArithChecker(role, ch, sid, delta, s), mode_(mode), error_(error), target_(target), guess_(guess) {}

protected:
    std::vector<u128> outgoing_values(const std::vector<AShare>& shares, uint64_t first) override {
        std::vector<u128> v = ArithChecker::outgoing_values(shares, first);
        if (mode_ == VALUE || mode_ == ZETA_GUESS)
            for (size_t i = 0; i < v.size(); i++)
                if (first + i == target_) v[i] = v[i] + error_;
        return v;
    }
    u128 recorded_mac(const AShare& share, uint64_t index) override {
        return mode_ == MAC && index == target_ ? share.mac + error_ : share.mac;
    }
    u128 outgoing_zeta(unsigned bits, u128 zeta, const std::vector<uint64_t>& chis, const std::vector<Opening>& items,
                       uint64_t counter) override {
        (void)bits;
        (void)counter;
        if (mode_ == ZETA) return zeta + error_;
        if (mode_ == ZETA_GUESS) {
            // The best strategy of Lemma 2.  The opened value is off by e = error_, so the two
            // honest zeta values sum to -(Delta_own + Delta_honest) chi e.  The cheater adds
            // Delta_own chi e, which it knows, and its guess of Delta_honest chi e.
            for (size_t j = 0; j < items.size(); j++)
                if (items[j].index == target_) {
                    const u128 ce = u128(chis[j]) * error_;
                    return zeta + (u128(delta_share()) + u128(guess_)) * ce;
                }
        }
        return zeta;
    }

private:
    Mode mode_;
    u128 error_;
    uint64_t target_, guess_;
};

struct ArithOut {
    std::vector<u128> opened;
    bool dead = false;
    uint64_t checks = 0, openings = 0;
    ChannelStats stats;
};

static void test_spdz2k() {
    Section sec("spdz2k");
    CtrPrgRng rng(21);
    const unsigned k = 64, s = 64, w = 128;
    const uint64_t d0 = rng.u64() | ((uint64_t)1 << 63), d1 = rng.u64() | ((uint64_t)1 << 63);
    const u128 delta = u128(d0) + u128(d1);  // 65 bits, not reduced

    // AShare arithmetic against plain integers.
    long long good = 0, total = 0;
    for (int t = 0; t < 200; t++) {
        const unsigned bits = t % 3 == 0 ? 128 : (t % 3 == 1 ? 96 : 72);
        const u128 mask = u128_mask(bits);
        const u128 a = spdz2k::detail::random_u128(rng, bits), b = spdz2k::detail::random_u128(rng, bits);
        const u128 c = u128(rng.u64());
        const auto sa = spdz2k::deal(a, delta, bits, false, rng), sb = spdz2k::deal(b, delta, bits, false, rng);
        u128 v;
        total += 6;
        good += mac_ok(sa.first, sa.second, delta, &v) && v == a;
        good += mac_ok(sa.first + sb.first, sa.second + sb.second, delta, &v) && v == ((a + b) & mask);
        good += mac_ok(sa.first - sb.first, sa.second - sb.second, delta, &v) && v == ((a - b) & mask);
        good += mac_ok(-sa.first, -sa.second, delta, &v) && v == ((u128(0) - a) & mask);
        good += mac_ok(sa.first.scale(c), sa.second.scale(c), delta, &v) && v == ((a * c) & mask);
        good += mac_ok(sa.first.add_public(c, 0, d0), sa.second.add_public(c, 1, d1), delta, &v) && v == ((a + c) & mask);
        const unsigned n = 8 + (unsigned)(t % 40);
        if (n + 64 <= bits) {
            total++;
            good += mac_ok(sa.first.reduce(n, 64), sa.second.reduce(n, 64), delta, &v) && v == (a & u128_mask(n + 64));
        }
    }
    check_many(good, total, "AShare arithmetic keeps the MAC relation with an unreduced Delta");
    {
        const AShare e(u128(5), u128(9), 128, true), f(u128(1), u128(2), 128, true);
        check(!(e + f).exact && !(e - f).exact && !(-e).exact && e.scale(u128(1)).exact && !e.scale(u128(2)).exact &&
                  e.scale(u128(0)).exact && e.add_public(u128(0), 0, 1).exact && !e.add_public(u128(1), 0, 1).exact &&
                  e.reduce(64, 64).exact && !e.reduce(8, 64).exact && e.with_exact(false).exact == false,
              "exactness flags of the linear operations");
        check(throws_invalid([&] { e + AShare(u128(0), u128(0), 96); }) && throws_invalid([] { AShare(u128(4), u128(0), 2); }) &&
                  throws_invalid([&] { e.reduce(100, 64); }) && throws_invalid([] { AShare(u128(0), u128(0), 129); }),
              "AShare argument checks");
        check(spdz2k::lemma2_bound(3) == 0.5 && spdz2k::share_bytes(72) == 9 && spdz2k::share_bytes(128) == 16, "helpers");
    }

    // Protocol runs.  Values: v0 exact, v1 not exact (needs a mask), v2 over Z_{2^(n+s)}.
    const bytes sid = {'a', 'r'};
    const u128 v0 = u128(123456789), v1 = u128_from(77, 5), v2 = u128_from(0xdeadbeef, 0x1f), rho = u128(0x1234567);
    const auto s0 = spdz2k::deal(v0, delta, w, true, rng), s1 = spdz2k::deal(v1, delta, w, false, rng);
    const auto s2 = spdz2k::deal(v2, delta, 16 + s, false, rng);
    // An output mask [[2^k rho]]: every share component is a multiple of 2^k (the parties hold
    // s-bit shares of rho and of Delta rho mod 2^s and shift them by k, DESIGN.md 4.3 step 5).
    const auto om_hi = spdz2k::deal(rho, delta, s, false, rng);
    const std::pair<AShare, AShare> om(AShare(om_hi.first.value << k, om_hi.first.mac << k, w),
                                       AShare(om_hi.second.value << k, om_hi.second.mac << k, w));
    check(mac_ok(om.first, om.second, delta) && spdz2k::reconstruct(om.first, om.second).first == (rho << k),
          "an output mask built from s-bit shares is a sharing of 2^k rho");
    const auto bad_om = spdz2k::deal(rho << k, delta, w, false, rng);  // shares that are not multiples of 2^k

    typedef std::function<ArithOut(spdz2k::ArithChecker&, int)> Body;
    uint64_t openings_at_failure[2] = {0, 0};  // openings a party had made when its checker failed
    size_t published_at_failure[2] = {0, 0};   // last_opened().size() + unverified().size() after the failure
    auto run = [&](const Body& body, int cheater = -1, CheatChecker::Mode mode = CheatChecker::HONEST, u128 error = u128(0),
                   uint64_t target = 0, uint64_t guess = 0, unsigned s_run = 64, uint64_t dd0 = 0, uint64_t dd1 = 0) {
        auto party = [&, cheater, mode, error, target, guess, s_run](int role) {
            return [&, role, cheater, mode, error, target, guess, s_run](Channel& ch) {
                const uint64_t dshare = s_run == 64 ? (role == 0 ? d0 : d1) : (role == 0 ? dd0 : dd1);
                CheatChecker chk(role, ch, sid, dshare, s_run, role == cheater ? mode : CheatChecker::HONEST, error, target, guess);
                ArithOut out;
                try {
                    out = body(chk, role);
                } catch (...) {
                    bool again = false;
                    try {
                        chk.check();
                    } catch (const ProtocolAbort&) {
                        again = true;
                    } catch (...) {
                    }
                    if (!chk.dead() || !again || chk.pending() != 0) throw std::logic_error("checker not retired after a failure");
                    openings_at_failure[role] = chk.openings();
                    published_at_failure[role] = chk.last_opened().size() + chk.unverified().size();
                    throw;
                }
                out.dead = chk.dead();
                out.checks = chk.checks();
                out.openings = chk.openings();
                out.stats = ch.stats();
                return out;
            };
        };
        return run_two_party(party(0), party(1));
    };

    // Honest: deferred opening, then a release of an exact and a masked value.
    const Body honest = [&](spdz2k::ArithChecker& chk, int role) {
        ArithOut out;
        const AShare& a2 = role == 0 ? s2.first : s2.second;
        out.opened.push_back(chk.open_deferred(a2));
        if (chk.pending() != 1) throw std::logic_error("pending");
        const std::vector<AShare> shares = {role == 0 ? s0.first : s0.second, role == 0 ? s1.first : s1.second};
        const AShare mask = role == 0 ? om.first : om.second;
        const std::vector<u128> rel = chk.open_final_many(shares, {nullptr, &mask});
        out.opened.insert(out.opened.end(), rel.begin(), rel.end());
        if (chk.pending() != 0) throw std::logic_error("pending after release");
        chk.check();  // nothing to verify: no message
        return out;
    };
    {
        const auto pr = run(honest);
        check(pr.first.ok && pr.second.ok, "honest openings pass (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            for (const auto* o : {&pr.first, &pr.second}) {
                check(o->value.opened.size() == 3 && o->value.opened[0] == v2 && o->value.opened[1] == v0 &&
                          o->value.opened[2] == (v1 & u128_mask(k)),
                      "opened values: representative for a deferred opening, v mod 2^k for a release");
                // Rounds: 1 (deferred) + 4 (check) + 1 (release) + 4 (check); bytes: 10 + (96 + 96 + 10) + 32 + (96 + 96 + 16).
                check(o->value.checks == 2 && o->value.openings == 3 && o->value.stats.rounds == 10, "two checks, ten rounds");
                check(o->value.stats.bytes_sent == 10 + (96 + 64 + 10) + 32 + (96 + 64 + 16), "bytes of the openings and checks");
            }
        }
    }
    // Every deviation makes the honest party abort.
    struct Dev {
        CheatChecker::Mode mode;
        uint64_t target;
        u128 error;
        const char* name;
    };
    const Dev devs[] = {
        {CheatChecker::VALUE, 0, u128(1), "wrong value share in a deferred opening"},
        {CheatChecker::VALUE, 0, u128(1) << 16, "error in the tag-only part of a deferred opening (multiple of 2^n)"},
        {CheatChecker::VALUE, 1, u128(1), "wrong value share in a release (exact value)"},
        {CheatChecker::VALUE, 2, u128(1) << 64, "wrong value share in a release (masked value, high part)"},
        {CheatChecker::VALUE, 2, u128(3), "wrong value share in a release (masked value)"},
        {CheatChecker::MAC, 0, u128(1), "wrong MAC share of a deferred opening"},
        {CheatChecker::MAC, 2, u128(1) << 100, "wrong MAC share of a release"},
        {CheatChecker::ZETA, 0, u128(1), "wrong zeta"},
    };
    for (const Dev& d : devs) {
        for (int cheater = 0; cheater < 2; cheater++) {
            const auto pr = run(honest, cheater, d.mode, d.error, d.target);
            const auto& h = cheater == 0 ? pr.second : pr.first;
            check(h.protocol_abort && h.error == "ArithCheck failed", std::string(d.name) + ": the honest party aborts (" + h.error + ")");
        }
    }
    // The deferred error is caught BEFORE the release is opened: the cheater sees no output.
    {
        openings_at_failure[0] = openings_at_failure[1] = 99;
        const auto pr = run(honest, 1, CheatChecker::VALUE, u128(1), 0);
        check(!pr.second.ok && !pr.first.ok && openings_at_failure[0] == 1 && openings_at_failure[1] == 1,
              "a wrong deferred opening is caught before the release is opened (no party sent a release share)");
    }
    // Opened values are accepted only when the check that covers them has passed (change F):
    // unverified() holds what was opened since the last check, last_opened() what the latest
    // passed check covered.
    {
        const Body body = [&](spdz2k::ArithChecker& chk, int role) {
            ArithOut out;
            if (!chk.last_opened().empty() || !chk.unverified().empty()) throw std::logic_error("a fresh checker has opened nothing");
            const u128 d = chk.open_deferred(role == 0 ? s2.first : s2.second);
            if (!chk.last_opened().empty() || chk.unverified() != std::vector<u128>(1, d))
                throw std::logic_error("a deferred opening is public but not verified");
            chk.check();
            if (chk.last_opened() != std::vector<u128>(1, d) || !chk.unverified().empty())
                throw std::logic_error("the check publishes what it covered");
            const std::vector<AShare> shares = {role == 0 ? s0.first : s0.second, role == 0 ? s1.first : s1.second};
            const AShare mask = role == 0 ? om.first : om.second;
            const std::vector<u128> rel = chk.open_final_many(shares, {nullptr, &mask});
            if (!chk.unverified().empty()) throw std::logic_error("a release leaves nothing unverified");
            out.opened = chk.last_opened();  // the released representatives
            out.opened.insert(out.opened.end(), rel.begin(), rel.end());
            return out;
        };
        const auto pr = run(body);
        check(pr.first.ok && pr.second.ok, "published openings: honest run (" + pr.first.error + pr.second.error + ")");
        for (const auto* o : {&pr.first, &pr.second})
            check(o->ok && o->value.opened.size() == 4 && o->value.opened[0] == v0 && o->value.opened[1] == ((v1 + (rho << k)) & u128_mask(w)) &&
                      o->value.opened[2] == v0 && o->value.opened[3] == (v1 & u128_mask(k)),
                  "last_opened after a release: the verified representatives (the masked one unreduced)");
        // A release whose check fails publishes nothing: what the first check had verified is
        // dropped as well, and nothing stays readable as unverified.
        published_at_failure[0] = published_at_failure[1] = 99;
        const auto bad = run(honest, 1, CheatChecker::VALUE, u128(1), 1);
        check(!bad.first.ok && bad.first.protocol_abort && published_at_failure[0] == 0 && published_at_failure[1] == 0,
              "after a failed check neither last_opened() nor unverified() holds a value");
    }
    // The abort hook (change G: the checker is one component of a session).  It runs once, when
    // the checker dies, whatever the cause; a second abort() does not run it again; a hook that
    // throws does not keep the checker alive.
    {
        int fired[2] = {0, 0};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                CheatChecker chk(role, ch, sid, role == 0 ? d0 : d1, 64, role == 1 ? CheatChecker::VALUE : CheatChecker::HONEST, u128(1), 0);
                chk.set_on_abort([&fired, role] { fired[role]++; });
                std::string verdict = "opened";
                try {
                    chk.open_final(role == 0 ? s0.first : s0.second);
                } catch (const ProtocolAbort& e) {
                    verdict = e.what();
                }
                const int after_failure = fired[role];
                chk.abort();
                chk.abort();
                return verdict + "|" + std::to_string(after_failure) + "|" + std::to_string(fired[role]) + (chk.dead() ? "|dead" : "|alive");
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.first.value == "ArithCheck failed|1|1|dead", "a failed ArithCheck runs the abort hook exactly once (" + (pr.first.ok ? pr.first.value : pr.first.error) + ")");
        const auto chans = LoopbackChannel::make_pair();
        spdz2k::ArithChecker quiet(0, *chans.first, sid, d0, 64), loud(0, *chans.first, sid, d0, 64), plain(0, *chans.first, sid, d0, 64);
        int calls = 0;
        quiet.set_on_abort([&calls] { calls++; });
        loud.set_on_abort([] { throw std::runtime_error("hook failed"); });
        quiet.abort();
        quiet.abort();
        loud.abort();
        plain.abort();  // no hook
        check(calls == 1 && quiet.dead() && loud.dead() && plain.dead() && throws<ProtocolAbort>([&] { loud.check(); }),
              "abort(): the hook runs on the first call only; a throwing hook does not keep the checker alive");
    }
    // Argument errors: refused before any message, the checker stays usable.
    {
        const Body args = [&](spdz2k::ArithChecker& chk, int role) {
            ArithOut out;
            const AShare a0 = role == 0 ? s0.first : s0.second, a1 = role == 0 ? s1.first : s1.second;
            const AShare mask = role == 0 ? om.first : om.second, bad = role == 0 ? bad_om.first : bad_om.second;
            const AShare small(u128(1), u128(1), 64);
            int n = 0;
            n += throws_invalid([&] { chk.open_final(a1); });                 // not exact, no mask
            n += throws_invalid([&] { chk.open_final(a0, &mask); });          // exact with a mask
            n += throws_invalid([&] { chk.open_final(a1, &bad); });           // mask not a multiple of 2^k
            n += throws_invalid([&] { chk.open_final(a1, &small); });         // mask over another modulus
            n += throws_invalid([&] { chk.open_deferred(small); });           // modulus not larger than 2^s
            n += throws_invalid([&] { chk.open_final_many({a0, a1}, {nullptr}); });
            if (n != 6 || chk.dead() || chk.openings() != 0) throw std::logic_error("argument errors");
            out.opened.push_back(chk.open_final(a0));
            return out;
        };
        const auto pr = run(args);
        check(pr.first.ok && pr.second.ok && pr.first.value.opened[0] == v0 && pr.first.value.stats.rounds == 5,
              "argument errors send nothing and leave the checker usable (" + pr.first.error + pr.second.error + ")");
        check(throws_invalid([&] {
                  const auto ch = LoopbackChannel::make_pair();
                  spdz2k::ArithChecker c(0, *ch.first, sid, 8, 3);
              }) &&
                  throws_invalid([&] {
                      const auto ch = LoopbackChannel::make_pair();
                      spdz2k::ArithChecker c(1, *ch.first, sid, 1, 64);
                  }),
              "constructor argument checks");
    }
    // Malformed openings: a value share that does not fit the modulus (68-bit modulus, nine-byte
    // encoding, bit 71 set) and a message of the wrong length are refused.
    {
        const auto sh = spdz2k::deal(u128(9), delta, 4 + 64, true, rng);
        for (int variant = 0; variant < 2; variant++) {
            for (int cheater = 0; cheater < 2; cheater++) {
                auto party = [&](int role) {
                    return [&, role](Channel& raw) {
                        std::map<size_t, TamperChannel::Rule> rules;
                        if (role == cheater) rules[0] = variant == 0 ? flip_bit_rule(71) : truncate_rule(1);
                        TamperChannel ch(raw, rules);
                        spdz2k::ArithChecker chk(role, ch, sid, role == 0 ? d0 : d1, 64);
                        std::string verdict = "opened";
                        try {
                            chk.open_final(role == 0 ? sh.first : sh.second);
                        } catch (const ProtocolAbort& e) {
                            verdict = e.what();
                        } catch (const std::exception& e) {
                            verdict = std::string("error: ") + e.what();
                        }
                        return verdict + (chk.dead() ? "|dead" : "|alive");
                    };
                };
                const auto pr = run_two_party(party(0), party(1));
                const auto& h = cheater == 0 ? pr.second : pr.first;
                const char* want = variant == 0 ? "opening: value share out of range|dead" : "opening: message has the wrong length|dead";
                check(h.ok && h.value == want, std::string("a malformed opening aborts (") + (h.ok ? h.value : h.error) + ")");
            }
        }
    }
    // Lemma 2 with a tiny s: the best strategy (guess Delta_h) passes with probability about
    // 2^-s + 2^-s (chi = 0), below the bound (s+1) / 2^s.
    {
        const unsigned ss = 3, bits = 8 + ss;
        const int runs = g_quick ? 300 : 1200;
        int pass = 0, consistent = 0;
        for (int t = 0; t < runs; t++) {
            const uint64_t e0 = rng.u64() & 7, e1 = rng.u64() & 7, guess = rng.u64() & 7;
            const u128 dl = u128(e0) + u128(e1), val = u128(rng.u64() & 0xFF);
            const auto sh = spdz2k::deal(val, dl, bits, true, rng);
            const Body body = [&](spdz2k::ArithChecker& chk, int role) {
                ArithOut out;
                out.opened.push_back(chk.open_final(role == 0 ? sh.first : sh.second));
                return out;
            };
            const auto pr = run(body, 1, CheatChecker::ZETA_GUESS, u128(1), 0, guess, ss, e0, e1);
            if (pr.first.ok) {
                pass++;
                consistent += pr.first.value.opened[0] == ((val + u128(1)) & u128_mask(8));
            } else {
                consistent += pr.first.protocol_abort && pr.first.error == "ArithCheck failed";
            }
        }
        const double rate = (double)pass / runs;
        check(consistent == runs, "Lemma 2 experiment: abort, or the forged value is accepted");
        check(rate > 0.14 && rate < 0.34 && rate < spdz2k::lemma2_bound(ss),
              "Lemma 2 experiment (s = 3): pass rate " + std::to_string(rate) + " is about 15/64 and below the bound 0.5");
        std::printf("  Lemma 2 experiment, s = 3: %d of %d forged openings passed (expected about %.0f, bound %.0f)\n", pass, runs,
                    runs * 15.0 / 64.0, runs * 0.5);
    }
}

// ------------------------------------------------------------ gates (local formulas on dealer material)
static void test_gates_local() {
    Section sec("gates (local evaluation, dealer material)");
    CtrPrgRng rng(31);
    const unsigned s = 64;
    // Exhaustive comparison identity: n <= 6: all r, all c in [0, N], all x; n = 8: sample of r.
    long long good = 0, total = 0, good_misc = 0, total_misc = 0;
    for (unsigned n : {1u, 2u, 3u, 4u, 5u, 6u, 8u}) {
        const unsigned k = n <= 4 ? 32 : 64, w = k + s;
        const uint64_t N = (uint64_t)1 << n;
        std::vector<protocol::Threshold> ths;
        for (uint64_t c = 0; c <= N; c++) ths.push_back(protocol::Threshold(u128(c), false));
        const GateDescriptor desc(n, k, s, false, true, ths, 0);
        std::vector<uint64_t> rs;
        if (n <= 6 && !(g_quick && n > 4)) {
            for (uint64_t r = 0; r < N; r++) rs.push_back(r);
        } else {
            rs = {0, 1, N / 2, N - 1};
            for (int i = 0; i < (g_quick ? 2 : 12); i++) rs.push_back(rng.u64() & (N - 1));
        }
        std::vector<uint64_t> T(N);
        for (uint64_t& t : T) t = rng.u64() & mask_bits(k);
        for (uint64_t r : rs) {
            const uint64_t e0 = rng.u64() | ((uint64_t)1 << 63), e1 = (r & 1) ? (rng.u64() | ((uint64_t)1 << 63)) : (rng.u64() >> 1);
            const u128 delta = u128(e0) + u128(e1), rp = u128(r);
            const std::pair<GateMaterial, GateMaterial> mats = gates::deal_gate(desc, delta, rng, &rp);
            gates::GateEval g0(mats.first, true), g1(mats.second, true);
            g0.set_delta_share(e0);
            g1.set_delta_share(e1);
            for (uint64_t x = 0; x < N; x++) {
                const uint64_t x_hat = (x + r) & (N - 1);
                u128 v;
                for (uint64_t c = 0; c <= N; c++) {
                    const AShare a = gates::comparison(g0, x_hat, u128(c)), b = gates::comparison(g1, x_hat, u128(c));
                    total++;
                    good += a.exact && mac_ok(a, b, delta, &v) && v == u128(x < c ? 1 : 0);
                }
                // The other gates on the same material.
                total_misc += 6;
                good_misc += mac_ok(gates::zero_test(g0, x_hat), gates::zero_test(g1, x_hat), delta, &v) && v == u128(x == 0 ? 1 : 0);
                const uint64_t c = rng.u64() & (N - 1);
                good_misc += mac_ok(gates::equality(g0, x_hat, c), gates::equality(g1, x_hat, c), delta, &v) && v == u128(x == c ? 1 : 0);
                const AShare ta = gates::table(g0, x_hat, T), tb = gates::table(g1, x_hat, T);
                good_misc += ta.exact && mac_ok(ta, tb, delta, &v) && v == u128(T[x]);
                uint64_t p = rng.u64() & (N - 1), q = rng.u64() & (N - 1);
                if (p > q) std::swap(p, q);
                good_misc += mac_ok(gates::interval(g0, x_hat, p, q), gates::interval(g1, x_hat, p, q), delta, &v) &&
                             v == u128(p <= x && x <= q ? 1 : 0);
                good_misc += mac_ok(gates::nonneg(g0, x_hat), gates::nonneg(g1, x_hat), delta, &v) && v == u128(x < N / 2 ? 1 : 0);
                const AShare ma = gates::msb(g0, x_hat), mb = gates::msb(g1, x_hat);
                good_misc += ma.exact && mac_ok(ma, mb, delta, &v) && v == u128(x >= N / 2 ? 1 : 0);
                if (x % 7 == 0 && N >= 4) {
                    std::vector<u128> ends = {u128(0), u128(N / 4), u128(N / 2 + 1), u128(N)};
                    const std::vector<AShare> ia = gates::multi_interval(g0, x_hat, ends), ib = gates::multi_interval(g1, x_hat, ends);
                    for (size_t j = 0; j < 3; j++) {
                        total_misc++;
                        good_misc += mac_ok(ia[j], ib[j], delta, &v) && v == u128(!(u128(x) < ends[j]) && u128(x) < ends[j + 1] ? 1 : 0);
                    }
                }
            }
            (void)w;
        }
    }
    check_many(good, total, "comparison identity [x < c] = D_c - D_0 + w_c for all r, c, x (exact, both planes)");
    check_many(good_misc, total_misc, "zero test, equality, table, interval, multi-interval, nonneg, msb");

    // ReLU and linear splines, n = k (correct modulo 2^k, not exact), with output masks.
    long long good_r = 0, total_r = 0;
    for (unsigned k : {4u, 8u}) {
        const uint64_t N = (uint64_t)1 << k;
        const unsigned w = k + s;
        const std::vector<u128> ends = {u128(0), u128(N / 4), u128(N / 2), u128(N)};
        const std::vector<gates::Piece> pieces = {gates::Piece(u128(3), u128(1)), gates::Piece(u128(0) - u128(2), u128(7)),
                                                  gates::Piece(u128(0), u128(5))};
        const GateSpec relu = GateSpec::relu_large(k, s), spline = GateSpec::linear_spline(k, s, ends, pieces);
        for (int t = 0; t < (g_quick ? 3 : (k == 4 ? 16 : 24)); t++) {
            const uint64_t e0 = rng.u64(), e1 = rng.u64();
            const u128 delta = u128(e0) + u128(e1);
            const auto mr = gates::deal_gate(relu.descriptor(), delta, rng), ms = gates::deal_gate(spline.descriptor(), delta, rng);
            const uint64_t r_relu = u128_lo(mr.first.rp_share + mr.second.rp_share) & (N - 1);
            const uint64_t r_spl = u128_lo(ms.first.rp_share + ms.second.rp_share) & (N - 1);
            gates::GateEval a0(mr.first, true), a1(mr.second, true), b0(ms.first, true), b1(ms.second, true);
            for (uint64_t x = 0; x < N; x++) {
                u128 v;
                const AShare ra = relu.evaluate(a0, (x + r_relu) & (N - 1))[0], rb = relu.evaluate(a1, (x + r_relu) & (N - 1))[0];
                total_r += 2;
                good_r += !ra.exact && mac_ok(ra, rb, delta, &v) && (u128_lo(v) & (N - 1)) == relu.plain(x)[0] && relu.plain(x)[0] == gates::plain_relu(x, k);
                const AShare sa = spline.evaluate(b0, (x + r_spl) & (N - 1))[0], sb = spline.evaluate(b1, (x + r_spl) & (N - 1))[0];
                good_r += !sa.exact && mac_ok(sa, sb, delta, &v) && (u128_lo(v) & (N - 1)) == spline.plain(x)[0];
                if (x == 3) {
                    // A masked release hides the high part: value + 2^k rho is uniform above bit k.
                    const AShare m0(mr.first.outmask_shares[0].value, mr.first.outmask_shares[0].tag, w);
                    const AShare m1(mr.second.outmask_shares[0].value, mr.second.outmask_shares[0].tag, w);
                    total_r++;
                    good_r += mac_ok(ra + m0, rb + m1, delta, &v) && (u128_lo(v) & (N - 1)) == relu.plain(x)[0];
                }
            }
        }
        check(gates::spline_table(k, k, ends, pieces)[1] == 4 && gates::relu_table(k, k)[N - 1] == 0 && gates::relu_table(k, k)[3] == 3 &&
                  gates::sign_table(k)[N / 2] == 1 && gates::sign_table(k)[N / 2 - 1] == 0,
              "public tables");
    }
    check_many(good_r, total_r, "relu_large and linear_spline are correct modulo 2^k with correct tags");

    // The representative of a non-exact gate really exceeds 2^k for some inputs (so the mask is needed).
    {
        const unsigned k = 8;
        const GateSpec relu = GateSpec::relu_large(k, s);
        bool above = false;
        for (int t = 0; t < 20 && !above; t++) {
            const u128 delta = u128(rng.u64()) + u128(rng.u64());
            const auto m = gates::deal_gate(relu.descriptor(), delta, rng);
            const uint64_t r = u128_lo(m.first.rp_share + m.second.rp_share) & 0xFF;
            gates::GateEval a0(m.first, true), a1(m.second, true);
            for (uint64_t x = 0; x < 256 && !above; x++) {
                const std::pair<u128, u128> rec = spdz2k::reconstruct(relu.evaluate(a0, (x + r) & 0xFF)[0], relu.evaluate(a1, (x + r) & 0xFF)[0]);
                above = (rec.first >> k) != u128(0);
            }
        }
        check(above, "the unreduced representative of ReLU exceeds 2^k for some input (change E is needed)");
    }

    // Argument checks and GateSpec plumbing.
    const GateDescriptor d = gates::comparison_descriptor(5, 64, s, u128(7));
    const u128 delta = u128(rng.u64()) + u128(rng.u64());
    const auto m = gates::deal_gate(d, delta, rng);
    gates::GateEval g(m.first);
    check(throws_invalid([&] { gates::comparison(g, 32, u128(7)); }) && throws_invalid([&] { gates::comparison(g, 3, u128(8)); }) &&
              throws_invalid([&] { gates::comparison(g, 3, u128(33)); }) && throws_invalid([&] { gates::zero_test(g, 3); }) &&
              throws_invalid([&] { gates::msb(g, 3); }) && throws_invalid([&] { gates::relu_large(g, 3); }) &&
              throws_invalid([&] { gates::interval(g, 3, 5, 4); }) && throws_invalid([&] { gates::table(g, 3, std::vector<uint64_t>(32)); }),
          "gate argument checks");
    check(GateSpec::comparison(5, 64, s, u128(7)).descriptor() == d && GateSpec::zero_test(8, 64, s).descriptor().name() == "gate_n8_k64_s64_shared_dpf" &&
              GateSpec::msb(8, 64, s, true).descriptor().name() == "gate_n8_k64_s64_auth_t128o_t0o" &&
              GateSpec::interval(8, 64, s, 3, 9).descriptor().name() == "gate_n8_k64_s64_shared_t10o_t3o" &&
              GateSpec::relu_large(8, s).exact() == false && GateSpec::nonneg(8, 64, s).exact() &&
              GateSpec::multi_interval(8, 64, s, {u128(1), u128(5), u128(9)}).n_outputs() == 2,
          "GateSpec descriptors");
    check(throws_invalid([&] { GateSpec::multi_interval(8, 64, s, {u128(5), u128(5)}); }) &&
              throws_invalid([&] { GateSpec::equality(4, 64, s, 16); }) && throws_invalid([&] { GateSpec::interval(4, 64, s, 9, 3); }) &&
              throws_invalid([&] { GateSpec::offset_only(4, 64, s).evaluate(g, 0); }),
          "GateSpec argument checks");

    // GateEval::dcf / dpf are public: a point outside [0, 2^n) is an argument error in point mode
    // and in full-domain mode (where it would index the 2^n-entry table), before and after the
    // table exists, and the evaluator stays usable.
    {
        const unsigned n = 4;
        const GateDescriptor dd(n, 64, s, false, true, {protocol::Threshold(u128(5), false), protocol::Threshold(u128(0), false)}, 0);
        const auto mm = gates::deal_gate(dd, delta, rng);
        long long ok = 0, cases = 0;
        for (int full = 0; full < 2; full++) {
            gates::GateEval e0(mm.first, full != 0), e1(mm.second, full != 0);
            for (int pass = 0; pass < 2; pass++) {  // pass 1: the tables of full-domain mode are built
                for (uint64_t bad : {(uint64_t)16, (uint64_t)17, (uint64_t)1000, (uint64_t)1 << 32, ~(uint64_t)0}) {
                    cases += 3;
                    ok += throws_invalid([&] { e0.dcf(u128(5), false, bad); });
                    ok += throws_invalid([&] { e0.dcf(u128(0), false, bad); });
                    ok += throws_invalid([&] { e0.dpf(bad); });
                }
                u128 v;
                cases += 2;
                ok += mac_ok(e0.dcf(u128(5), false, 15), e1.dcf(u128(5), false, 15), delta, &v) && (v == u128(0) || v == u128(1));
                ok += mac_ok(e0.dpf(15), e1.dpf(15), delta, &v) && (v == u128(0) || v == u128(1));
            }
        }
        check_many(ok, cases, "GateEval::dcf / dpf refuse a point outside the domain (point and full-domain mode)");
    }
    // The public tables and the plaintext ReLU validate n before the half-domain constant is formed.
    check(throws_invalid([] { gates::relu_table(0, 8); }) && throws_invalid([] { gates::relu_table(31, 8); }) &&
              throws_invalid([] { gates::relu_table(64, 8); }) && throws_invalid([] { gates::sign_table(0); }) &&
              throws_invalid([] { gates::sign_table(31, 1); }) && throws_invalid([] { gates::plain_relu(1, 0); }) &&
              throws_invalid([] { gates::plain_relu(1, 65); }) && throws_invalid([] { gates::domain_size(65); }) &&
              throws_invalid([] { gates::msb_descriptor(0, 64, 64); }) && throws_invalid([] { gates::relu_large_descriptor(0, 64); }) &&
              throws_invalid([] { GateSpec::msb(0, 64, 64); }),
          "relu_table, sign_table, plain_relu and the half-domain descriptors reject n outside their range");
    check(gates::relu_table(1, 8) == std::vector<uint64_t>({0, 0}) && gates::sign_table(1) == std::vector<uint64_t>({0, 1}) &&
              gates::relu_table(3, 8) == std::vector<uint64_t>({0, 1, 2, 3, 0, 0, 0, 0}) && gates::plain_relu(5, 1) == 0 &&
              gates::plain_relu(0, 1) == 0 && gates::plain_relu(((uint64_t)1 << 63) - 1, 64) == ((uint64_t)1 << 63) - 1 &&
              gates::plain_relu((uint64_t)1 << 63, 64) == 0 && gates::domain_size(64) == (u128(1) << 64) && gates::domain_size(0) == u128(1),
          "relu_table, sign_table and plain_relu at the ends of their ranges");
}

// ------------------------------------------------------------ state machine
static void test_state() {
    Section sec("state");
    namespace fs = std::filesystem;
    const bytes sid = {'s', 't'};
    {
        state::Registry reg(sid);
        check(reg.begin_key() == 0 && reg.begin_key(state::Registry::NEXT, 3) == 1 && reg.begin_key(10) == 10 && reg.begin_key() == 11,
              "key identifiers are monotone");
        check(throws<state::StateError>([&] { reg.begin_key(5); }) && throws<state::StateError>([&] { reg.begin_key(11); }),
              "an identifier that was issued is refused");
        check(throws<state::StateError>([&] { reg.key(0); }) && throws<state::StateError>([&] { reg.key(99); }),
              "a key is not usable before it is certified");
        state::KeyRecord filled;
        filled.n = 4;
        reg.certify_key(0, filled);
        check(reg.key(0).state == state::CERTIFIED && reg.key(0).n == 4 && throws<state::StateError>([&] { reg.certify_key(0, filled); }),
              "GENERATING -> CERTIFIED happens once");
        const uint64_t gid = reg.begin_gate();
        check(throws<state::StateError>([&] { reg.use_gate(gid); }) && throws<state::StateError>([&] { reg.masked_gate(gid); }),
              "a gate is not usable before it is certified");
        CtrPrgRng rng(41);
        const auto mats = gates::deal_gate(gates::table_descriptor(4, 64, 64), u128(5), rng);
        reg.certify_gate(gid, std::make_shared<gates::GateEval>(mats.first));
        check(throws<state::StateError>([&] { reg.set_masked(gid, 3, "shared"); }), "no Mask result without use_gate");
        reg.use_gate(gid);
        check(throws<state::GateUsedError>([&] { reg.use_gate(gid); }), "a second Mask is refused (GateUsedError)");
        check(throws<state::StateError>([&] { reg.masked_gate(gid); }), "a gate whose Mask did not complete cannot be applied");
        reg.set_masked(gid, 3, "shared");
        check(reg.masked_gate(gid).x_hat == 3 && throws<state::StateError>([&] { reg.set_masked(gid, 4, "shared"); }) &&
                  throws<state::GateUsedError>([&] { reg.use_gate(gid); }),
              "x_hat is recorded once");
        check(reg.next_qid() == 0 && reg.next_qid() == 1, "query identifiers");
        const state::Registry::Summary sm = reg.summary();
        check(sm.keys_certified == 1 && sm.keys_generating == 5 && sm.gates_used == 1 && sm.active, "summary");
        reg.abort();
        check(!reg.active() && throws<ProtocolAbort>([&] { reg.begin_key(); }) && throws<ProtocolAbort>([&] { reg.gate(gid); }) &&
                  throws<ProtocolAbort>([&] { reg.next_qid(); }) && throws<ProtocolAbort>([&] { reg.key(0); }),
              "after abort every method throws ProtocolAbort");
    }
    // Persistence.
    const fs::path dir = unique_temp_dir("test_protocol_state");
    std::error_code ec;
    const std::string path = (dir / "state.json").string();
    {
        state::Registry a(sid, path);
        check(a.starts() == 1 && a.begin_gate() == 0 && a.begin_gate() == 1 && a.begin_key(7) == 7, "first process issues identifiers");
        const json::Value doc = json::parse(read_text(path));
        check(doc.at("format").str() == "couplet-state-1" && doc.at("sid").str() == circuit::hex(sid) && doc.at("state").str() == "ACTIVE" &&
                  doc.at("reserved").at("gid").as_u64() == 65 && doc.at("reserved").at("kid").as_u64() == 72 && doc.at("starts").as_u64() == 1,
              "the reserved bounds are on disk before the identifiers are used");
    }
    {
        state::Registry b(sid, path);  // a restart: everything below the stored bound is gone
        check(b.starts() == 2 && throws<state::StateError>([&] { b.begin_gate(2); }) && throws<state::StateError>([&] { b.gate(0); }),
              "a restarted process cannot touch identifiers of the earlier one");
        check(b.begin_gate() == 65 && b.begin_key() == 72 && b.counters().floor[state::Registry::GID] == 65, "a restart continues above the stored bounds");
        check(throws_invalid([&] { state::Registry other(bytes{'x'}, path); }), "a state file of another session is refused");
        check(b.abort_marker_path() == path + ".aborted" && !fs::exists(path + ".aborted") && !b.abort_persist_failed(),
              "no abort marker while the session is ACTIVE");
        const bool durable = b.abort();
        const json::Value marker = json::parse(read_text(path + ".aborted"));
        check(durable && !b.abort_persist_failed() && b.abort_persist_error().empty() && marker.at("format").str() == "couplet-aborted-1" &&
                  marker.at("sid").str() == circuit::hex(sid) && json::parse(read_text(path)).at("state").str() == "ABORTED",
              "abort() writes the create-only marker and the state file, and reports success");
        check(b.abort() && !b.abort_persist_failed(), "a second abort() reports the same");
    }
    {
        state::Registry c(sid, path);
        check(!c.active() && throws<ProtocolAbort>([&] { c.begin_gate(); }), "a persisted ABORTED session stays aborted");
        std::ofstream(path, std::ios::binary) << "{\"format\": \"couplet-state-1\", \"sid\": 5}";
        check(throws_invalid([&] { state::Registry d(sid, path); }), "an unreadable state file is refused, not reset");
    }

    // ABORTED on disk when the state file cannot be replaced at abort time (a directory in the
    // place of its temporary file makes every store() fail).  The marker is written through an
    // independent, create-only file, so a later process on the same state file stays ABORTED
    // although the state file itself still says ACTIVE.
    {
        const std::string p2 = (dir / "blocked.json").string();
        {
            state::Registry a(sid, p2);
            a.begin_gate();
            fs::create_directories(p2 + ".tmp");  // from now on the state file cannot be rewritten
            const bool durable = a.abort();
            check(durable && !a.abort_persist_failed() && !a.abort_persist_error().empty() && !a.active() &&
                      a.abort_persist_error().find("state file") != std::string::npos,
                  "state file not replaceable: the abort is durable through the marker, the failed write is reported (" +
                      a.abort_persist_error() + ")");
            check(json::parse(read_text(p2)).at("state").str() == "ACTIVE" && fs::is_regular_file(p2 + ".aborted"),
                  "the state file still says ACTIVE; the marker exists");
        }
        check(throws<std::runtime_error>([&] { state::Registry again(sid, p2); }), "while the state file is blocked a new process cannot start on it");
        fs::remove_all(p2 + ".tmp", ec);
        {
            state::Registry later(sid, p2);  // the failure is gone; the state file says ACTIVE, the marker ABORTED
            check(!later.active() && throws<ProtocolAbort>([&] { later.begin_gate(); }) && throws<ProtocolAbort>([&] { later.next_qid(); }),
                  "a later process honours the marker: the session stays ABORTED");
            check(json::parse(read_text(p2)).at("state").str() == "ABORTED", "and repairs the state file");
        }
    }
    // Neither the state file nor the marker can be written: the in-memory state protects this
    // process and the failure is reported to the caller.
    {
        const std::string p3 = (dir / "both.json").string();
        state::Registry a(sid, p3);
        fs::create_directories(p3 + ".tmp");
        fs::create_directories(p3 + ".aborted");  // a directory in the place of the marker
        const bool durable = a.abort();
        check(!durable && a.abort_persist_failed() && !a.active() && throws<ProtocolAbort>([&] { a.begin_key(); }) &&
                  a.abort_persist_error().find("abort marker") != std::string::npos &&
                  a.abort_persist_error().find("state file") != std::string::npos,
              "no durable abort possible: abort() returns false and says why (" + a.abort_persist_error() + ")");
        check(!a.abort() && a.abort_persist_failed(), "the failure stays reported");
        fs::remove_all(p3 + ".tmp", ec);
        fs::remove_all(p3 + ".aborted", ec);
        state::Registry later(sid, p3);
        check(later.active(), "(control) without marker and with an ACTIVE state file a later process is ACTIVE: this is what the report is for");
    }
    // The marker alone decides, whatever its content, unless it names another session; it is
    // created once and never rewritten.
    {
        const std::string p4 = (dir / "marker.json").string();
        { state::Registry a(sid, p4); }  // ACTIVE state file
        std::ofstream(p4 + ".aborted", std::ios::binary) << "";  // an empty marker
        {
            state::Registry b(sid, p4);
            check(!b.active(), "an empty (unparseable) marker counts: ABORTED");
        }
        fs::remove(p4 + ".aborted", ec);
        fs::remove(p4, ec);
        const std::string own = "{\"format\": \"couplet-aborted-1\", \"sid\": \"" + circuit::hex(sid) + "\", \"note\": \"kept\"}";
        std::ofstream(p4 + ".aborted", std::ios::binary) << own;
        {
            state::Registry c(sid, p4);  // a marker without a state file
            check(!c.active() && json::parse(read_text(p4)).at("state").str() == "ABORTED", "a marker without a state file: ABORTED");
        }
        fs::remove(p4 + ".aborted", ec);
        fs::remove(p4, ec);
        std::ofstream(p4 + ".aborted", std::ios::binary) << "{\"format\": \"couplet-aborted-1\", \"sid\": \"78\"}";  // session "x"
        check(throws_invalid([&] { state::Registry d(sid, p4); }), "a marker of another session is refused");
        fs::remove(p4 + ".aborted", ec);
        {
            state::Registry e(sid, p4);  // ACTIVE, no marker
            std::ofstream(p4 + ".aborted", std::ios::binary) << own;  // as if another process of the session had aborted meanwhile
            check(e.active() && e.abort() && !e.abort_persist_failed() && e.abort_persist_error().empty() && read_text(p4 + ".aborted") == own,
                  "an existing marker is not rewritten (create-only) and counts as written");
        }
        state::Registry none(sid);  // no persistence
        check(none.abort() && !none.abort_persist_failed() && none.abort_marker_path().empty(), "without persistence abort() has nothing to write");
    }
    fs::remove_all(dir, ec);
}

// ------------------------------------------------------------ Pi_Gen: keys
struct KeyCase {
    protocol::Family family;
    unsigned n;
    protocol::IndexMode index_mode;
    protocol::BetaMode beta_mode;
    size_t batch;
    std::vector<uint64_t> alpha[2], beta[2];
};

struct GenOut {
    std::vector<std::vector<fss::DpfKey>> dpf;  // per case
    std::vector<std::vector<fss::DcfKey>> dcf;
    uint64_t delta = 0;
    uint64_t triple_requests = 0, triples_used = 0, evaluations = 0;
    unsigned bucket = 0;
    bool sums_ok = true;
    ChannelStats stats;
    prep::PhaseStat prep_total, book_total;
    std::vector<abb::EvalStats> gen_stats;
    abb::EvalStats setup_stats;
    std::string last_circuit;
};

static std::vector<KeyCase> make_cases(RandomSource& rng, unsigned k, bool with_add, const std::vector<unsigned>& ns) {
    std::vector<KeyCase> cases;
    auto add = [&](protocol::Family f, unsigned n, protocol::IndexMode im, protocol::BetaMode bm, size_t batch) {
        KeyCase c;
        c.family = f;
        c.n = n;
        c.index_mode = im;
        c.beta_mode = bm;
        c.batch = batch;
        for (size_t j = 0; j < batch; j++)
            for (int role = 0; role < 2; role++) {
                c.alpha[role].push_back(rng.u64() & mask_bits(n));
                if (bm == protocol::BETA_SECRET) c.beta[role].push_back(rng.u64() & mask_bits(k));
            }
        cases.push_back(c);
    };
    for (unsigned n : ns)
        for (protocol::Family f : {protocol::FAMILY_DPF, protocol::FAMILY_DCF})
            for (protocol::BetaMode bm : {protocol::BETA_PUBLIC_ONE, protocol::BETA_SECRET}) {
                if (k == 32 && (n != 4 || bm != protocol::BETA_SECRET)) continue;  // exported for k = 32: n = 4, secret payload
                add(f, n, protocol::INDEX_XOR, bm, 1);
                add(f, n, protocol::INDEX_XOR, bm, 3);
            }
    if (with_add) {
        add(protocol::FAMILY_DPF, 4, protocol::INDEX_ADD, protocol::BETA_SECRET, 2);
        add(protocol::FAMILY_DCF, 4, protocol::INDEX_ADD, protocol::BETA_PUBLIC_ONE, 2);
    }
    // Boundary indices: alpha = 0 and alpha = N - 1 (XOR shares), payload 0 and 2^k - 1.
    if (k == 64) {
        for (protocol::Family f : {protocol::FAMILY_DPF, protocol::FAMILY_DCF}) {
            add(f, 4, protocol::INDEX_XOR, protocol::BETA_SECRET, 3);
            KeyCase& c = cases.back();
            c.alpha[1][0] = c.alpha[0][0];                    // alpha = 0
            c.alpha[1][1] = c.alpha[0][1] ^ 15;               // alpha = N - 1
            c.beta[1][0] = (uint64_t)0 - c.beta[0][0];        // beta = 0
            c.beta[1][1] = ~(uint64_t)0 - c.beta[0][1];       // beta = 2^k - 1
        }
    }
    return cases;
}

static GenOut run_cases(Session& ses, const std::vector<KeyCase>& cases, int role) {
    GenOut out;
    out.delta = ses.delta_share();
    out.setup_stats = ses.setup_stats();
    for (const KeyCase& c : cases) {
        if (c.family == protocol::FAMILY_DPF) {
            out.dpf.push_back(protocol::gen_dpf(ses, c.n, c.index_mode, c.beta_mode, c.alpha[role], c.beta[role]));
            out.dcf.emplace_back();
        } else {
            out.dcf.push_back(protocol::gen_dcf(ses, c.n, c.index_mode, c.beta_mode, c.alpha[role], c.beta[role]));
            out.dpf.emplace_back();
        }
        out.gen_stats.push_back(ses.last_gen.stats);
    }
    out.last_circuit = ses.last_gen.circuit;
    out.triple_requests = ses.engine().triple_requests();
    out.triples_used = ses.engine().triples_used();
    out.evaluations = ses.engine().evaluations();
    out.stats = ses.channel().stats();
    out.book_total = ses.book().total();
    if (ses.real_prep()) {
        out.prep_total = ses.real_prep()->stats().total();
        out.bucket = ses.real_prep()->last_bucket_size();
    }
    out.sums_ok = out.prep_total.bytes_sent + out.book_total.bytes_sent == out.stats.bytes_sent &&
                  out.prep_total.bytes_recv + out.book_total.bytes_recv == out.stats.bytes_recv &&
                  out.prep_total.rounds + out.book_total.rounds == out.stats.rounds &&
                  out.prep_total.msgs_sent + out.book_total.msgs_sent == out.stats.msgs_sent;
    return out;
}

static void verify_cases(const std::vector<KeyCase>& cases, const GenOut& o0, const GenOut& o1, unsigned k, unsigned s,
                         const std::string& tag) {
    const u128 delta = fss::make_delta(o0.delta, o1.delta);  // the integer Delta_0 + Delta_1, NOT reduced
    for (size_t i = 0; i < cases.size(); i++) {
        const KeyCase& c = cases[i];
        for (size_t j = 0; j < c.batch; j++) {
            const uint64_t alpha = c.index_mode == protocol::INDEX_XOR ? (c.alpha[0][j] ^ c.alpha[1][j])
                                                                       : ((c.alpha[0][j] + c.alpha[1][j]) & mask_bits(c.n));
            const uint64_t beta = c.beta_mode == protocol::BETA_SECRET ? ((c.beta[0][j] + c.beta[1][j]) & mask_bits(k)) : 1;
            const std::string what = tag + " " + protocol::key_circuit_name(c.family, c.n, k, s, c.index_mode, c.beta_mode) +
                                     " batch " + std::to_string(c.batch) + " #" + std::to_string(j);
            if (c.family == protocol::FAMILY_DPF) {
                check(o0.dpf[i].size() == c.batch && o1.dpf[i].size() == c.batch, what + ": one key per instance");
                verify_dpf(o0.dpf[i][j], o1.dpf[i][j], alpha, beta, delta, c.n, k, s, what);
            } else {
                check(o0.dcf[i].size() == c.batch && o1.dcf[i].size() == c.batch, what + ": one key per instance");
                verify_dcf(o0.dcf[i][j], o1.dcf[i][j], alpha, beta, delta, c.n, k, s, what);
            }
        }
        // Instances of one batch are independent: different roots.
        if (c.batch > 1) {
            if (c.family == protocol::FAMILY_DPF) check(o0.dpf[i][0].seed != o0.dpf[i][1].seed, "batch instances have different roots");
            else check(o1.dcf[i][0].seed != o1.dcf[i][1].seed, "batch instances have different roots");
        }
    }
}

static void test_gen_ideal(circuit::CircuitStore& store) {
    Section sec("Pi_Gen keys, ideal preprocessing");
    CtrPrgRng rng(51);
    for (int variant = 0; variant < 3; variant++) {
        // 0: k = 64, Delta forced >= 2^s;  1: k = 64, Delta forced < 2^s;  2: k = 32, random Delta.
        const unsigned k = variant == 2 ? 32 : 64, s = 64;
        const std::vector<KeyCase> cases = make_cases(rng, k, variant == 0, variant == 1 && g_quick ? std::vector<unsigned>{4} : std::vector<unsigned>{4, 8});
        prep::IdealDealer dealer;
        const bytes sid = {'g', 'i', (uint8_t)('0' + variant)};
        const uint64_t forced[2] = {variant == 0 ? (rng.u64() | ((uint64_t)1 << 63)) : (rng.u64() >> 1),
                                    variant == 0 ? (rng.u64() | ((uint64_t)1 << 63)) : (rng.u64() >> 1)};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                if (variant != 2) ses.set_delta_share(forced[role]);
                ses.setup();
                return run_cases(ses, cases, role);
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        const std::string tag = "ideal v" + std::to_string(variant);
        check(pr.first.ok && pr.second.ok, tag + ": no abort (" + pr.first.error + pr.second.error + ")");
        if (!pr.first.ok || !pr.second.ok) continue;
        const u128 delta = fss::make_delta(pr.first.value.delta, pr.second.value.delta);
        if (variant == 0) check((delta >> 64) == u128(1), tag + ": Delta_0 + Delta_1 >= 2^s in this session");
        if (variant == 1) check((delta >> 64) == u128(0), tag + ": Delta_0 + Delta_1 < 2^s in this session");
        verify_cases(cases, pr.first.value, pr.second.value, k, s, tag);
        check(pr.first.value.evaluations == cases.size() + 1 && pr.first.value.triple_requests == cases.size() + 1, tag + ": one evaluation per case");
        check(pr.first.value.setup_stats.n_and == 64 && pr.first.value.setup_stats.rounds == 67, tag + ": session setup is 64 AND gates and 67 rounds");
        // Rounds of a generation: AND depth + 6.
        bool rounds_ok = true;
        for (const abb::EvalStats& st : pr.first.value.gen_stats) rounds_ok = rounds_ok && st.rounds == st.and_depth + 6;
        check(rounds_ok, tag + ": a generation takes AND depth + 6 rounds, whatever the batch");
        check(pr.first.value.sums_ok && pr.second.value.sums_ok, tag + ": the phases sum to the channel totals");
    }
    // Delta_b is sampled from the whole range [0, 2^s): over 24 sessions the top bit takes both
    // values and the unreduced sum exceeds 2^s in some session and not in another.
    {
        std::vector<uint64_t> d[2];
        bool all_ok = true;
        for (int t = 0; t < 24; t++) {
            prep::IdealDealer dealer;
            const bytes sid = {'g', 'd', (uint8_t)t};
            auto party = [&](int role) {
                return [&, role](Channel& ch) {
                    Session ses(role, ch, sid, store, dealer, 64, 64);
                    ses.setup();
                    return ses.delta_share();
                };
            };
            const auto pr = run_two_party(party(0), party(1));
            all_ok = all_ok && pr.first.ok && pr.second.ok;
            d[0].push_back(pr.first.value);
            d[1].push_back(pr.second.value);
        }
        int top = 0, big = 0, low32 = 0;
        for (size_t t = 0; t < d[0].size(); t++) {
            top += (int)(d[0][t] >> 63) + (int)(d[1][t] >> 63);
            big += (fss::make_delta(d[0][t], d[1][t]) >> 64) != u128(0);
            low32 += (d[0][t] & 0xFFFFFFFFull) != 0 && (d[1][t] >> 32) != 0;
        }
        check(all_ok && top >= 8 && top <= 40 && big >= 2 && big <= 22 && low32 == 24,
              "Delta_b is uniform over [0, 2^s): top bit set in " + std::to_string(top) + " of 48 shares, Delta >= 2^s in " +
                  std::to_string(big) + " of 24 sessions");
    }

    // Argument errors do not abort and send nothing.
    {
        prep::IdealDealer dealer;
        const bytes sid = {'g', 'a'};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, 64, 64);
                int n = 0;
                n += throws_invalid([&] { protocol::gen_dpf(ses, 4, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {1}); });  // no setup
                n += throws_invalid([&] { ses.set_delta_share(5); ses.setup(); ses.set_delta_share(6); });
                const uint64_t sent = ch.stats().msgs_sent;
                n += throws_invalid([&] { protocol::gen_dpf(ses, 4, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {16}); });
                n += throws_invalid([&] { protocol::gen_dpf(ses, 4, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {}); });
                n += throws_invalid([&] { protocol::gen_dpf(ses, 4, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {1}, {2}); });
                n += throws_invalid([&] { protocol::gen_dpf(ses, 4, protocol::INDEX_XOR, protocol::BETA_SECRET, {1, 2}, {2}); });
                n += throws_invalid([&] { protocol::gen_dcf(ses, 0, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {0}); });
                n += throws_invalid([&] { protocol::gen_gate(ses, gates::table_descriptor(8, 32, 64)); });      // another k
                n += throws_invalid([&] { protocol::gen_gate(ses, gates::table_descriptor(8, 64, 64), 0); });   // batch 0
                n += throws_invalid([&] { ses.setup(); });                                                         // twice
                n += throws<std::runtime_error>([&] { protocol::gen_dpf(ses, 5, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {1}); });  // not exported
                const bool quiet = ch.stats().msgs_sent == sent && ses.active();
                const std::vector<fss::DpfKey> keys = protocol::gen_dpf(ses, 4, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {(uint64_t)(role ? 9 : 3)});
                return std::make_pair(n * 10 + (quiet ? 1 : 0), keys[0].to_bytes());
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok && pr.first.value.first == 111 && pr.second.value.first == 111,
              "argument errors of Pi_Gen send nothing and do not abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            const fss::DpfKey k0 = fss::DpfKey::from_bytes(pr.first.value.second), k1 = fss::DpfKey::from_bytes(pr.second.value.second);
            verify_dpf(k0, k1, 3 ^ 9, 1, u128(5) + u128(5), 4, 64, 64, "key after the argument errors");
        }
    }
}

static void test_gen_real(circuit::CircuitStore& store) {
    Section sec("Pi_Gen keys, real preprocessing");
    CtrPrgRng rng(61);
    const unsigned k = 64, s = 64;
    std::vector<KeyCase> cases;
    {
        const std::vector<KeyCase> all = make_cases(rng, k, false, {4, 8});
        for (const KeyCase& c : all) {
            // DPF and DCF for n in {4, 8} with public payload (batch 1), plus two secret-payload cases and one batch of 3.
            const bool pub1 = c.beta_mode == protocol::BETA_PUBLIC_ONE && c.batch == 1;
            const bool secret = c.beta_mode == protocol::BETA_SECRET && c.n == 4 && c.batch == (c.family == protocol::FAMILY_DPF ? 3u : 1u);
            if ((pub1 || secret) && !(g_quick && c.n == 8 && c.family == protocol::FAMILY_DCF)) cases.push_back(c);
            if (cases.size() == (g_quick ? 5u : 6u)) break;
        }
    }
    size_t reserve = 0;
    for (const KeyCase& c : cases)
        reserve += store.get(protocol::key_circuit_name(c.family, c.n, k, s, c.index_mode, c.beta_mode))->n_and() * c.batch;
    const bytes sid = {'g', 'r'};
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            Session ses(role, ch, sid, store, k, s, 40);
            ses.setup(reserve);  // all triples of the session in ONE request
            return run_cases(ses, cases, role);
        };
    };
    const auto pr = run_two_party(party(0), party(1));
    check(pr.first.ok && pr.second.ok, "real: no abort (" + pr.first.error + pr.second.error + ")");
    if (!pr.first.ok || !pr.second.ok) return;
    verify_cases(cases, pr.first.value, pr.second.value, k, s, "real");
    for (const GenOut* o : {&pr.first.value, &pr.second.value}) {
        check(o->triple_requests == 1 && o->triples_used == reserve + 64, "real: a single bucketing population for the whole session");
        check(o->bucket == prep::bucket_size(reserve + 64, 40), "real: bucket size of the population");
        check(o->sums_ok, "real: preprocessing phases plus engine phases equal the channel totals");
    }
    check(pr.first.value.stats.rounds == pr.second.value.stats.rounds && pr.first.value.stats.bytes_sent == pr.second.value.stats.bytes_recv,
          "real: both parties agree on rounds and bytes");
    std::printf("  real preprocessing: %zu generations, %zu triples (B = %u), %.1f MB sent by P_0, %llu rounds, %.2f s\n", cases.size(),
                reserve + 64, pr.first.value.bucket, pr.first.value.stats.bytes_sent / 1e6, (unsigned long long)pr.first.value.stats.rounds,
                pr.first.seconds);
}

// ------------------------------------------------------------ triple budget (bucketing populations)
static void test_budget(circuit::CircuitStore& store) {
    Section sec("triple budget (strict mode, one bucketing population)");
    const unsigned k = 64, s = 64;
    const GateSpec tab = GateSpec::table_gate(8, k, s, std::vector<uint64_t>(256, 7), false);
    const size_t N_KEY = 38019, N_TAB = 68246, N_OM = 4159, N_SETUP = 64;  // AND gates recorded by the exporter

    // The helpers against the circuit manifest (written by the Python exporter from Circuit.stats()).
    {
        const json::Value manifest = json::parse(read_text(store.dir() + "/manifest.json"));
        auto recorded = [&](const std::string& name) -> uint64_t {
            for (const json::Value& e : manifest.at("circuits").items)
                if (e.at("name").str() == name) return e.at("stats").at("n_and").as_u64();
            throw std::runtime_error("not in the manifest: " + name);
        };
        const GateDescriptor om = protocol::outmask_descriptor(k, s);
        check(om.name() == "gate_n1_k64_s64_shared_om1" && om.name() == Session::outmask_circuit_name(k, s) &&
                  om == gates::offset_descriptor(1, k, s, false, 1) && Session::setup_circuit_name(s) == "setup_s64",
              "names of the output-mask circuit and of the setup circuit");
        const protocol::Planned pk =
            protocol::planned_key(store, protocol::FAMILY_DPF, 4, k, s, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, 3);
        const protocol::Planned pg = protocol::planned_gate(store, tab.descriptor(), 2);
        const protocol::Planned po = protocol::planned_outmasks(store, k, s, 5), po0 = protocol::planned_outmasks(store, k, s, 0);
        check(pk.call == "gen_dpf" && pk.circuit == "dpf_n4_k64_s64_xor_one" && pk.batch == 3 && pk.n_and == recorded(pk.circuit) &&
                  pk.n_and == N_KEY && pk.triples == 3 * N_KEY,
              "planned_key: n_and of the manifest times the batch");
        check(pg.call == "gen_gate" && pg.circuit == "gate_n8_k64_s64_shared_dpf" && pg.batch == 2 && pg.n_and == recorded(pg.circuit) &&
                  pg.n_and == N_TAB && pg.triples == 2 * N_TAB,
              "planned_gate");
        check(po.call == "outmasks" && po.circuit == om.name() && po.n_and == recorded(po.circuit) && po.n_and == N_OM &&
                  po.triples == 5 * N_OM && po0.triples == 0 && protocol::outmask_and_count(store, k, s) == N_OM,
              "planned_outmasks: 4,159 AND gates per mask");
        check(protocol::and_budget({pk, pg, po}) == 3 * N_KEY + 2 * N_TAB + 5 * N_OM && protocol::and_budget({}) == 0 &&
                  protocol::session_triples(store, {pk, pg, po}, s) == N_SETUP + 3 * N_KEY + 2 * N_TAB + 5 * N_OM &&
                  protocol::session_triples(store, {}, s) == N_SETUP && protocol::and_count(store, "setup_s64") == recorded("setup_s64"),
              "and_budget and session_triples (the setup circuit has 64 AND gates)");
        check(protocol::planned_key(store, protocol::FAMILY_DCF, 4, k, s, protocol::INDEX_XOR, protocol::BETA_SECRET).triples ==
                      recorded("dcf_n4_k64_s64_xor_secret") &&
                  protocol::planned_key(store, protocol::FAMILY_DPF, 4, k, s, protocol::INDEX_ADD, protocol::BETA_SECRET).circuit ==
                      "dpf_n4_k64_s64_add_secret" &&
                  protocol::and_count(store, GateSpec::comparison(8, k, s, u128(85), true).descriptor()) ==
                      recorded("gate_n8_k64_s64_auth_t85o_t0o"),
              "budgets of other families, index modes and gates");
        check(throws_invalid([&] { protocol::planned_key(store, protocol::FAMILY_DPF, 4, k, s, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, 0); }) &&
                  throws_invalid([&] { protocol::planned_gate(store, tab.descriptor(), 0); }) &&
                  throws_invalid([&] { protocol::planned_key(store, (protocol::Family)7, 4, k, s); }) &&
                  throws<std::runtime_error>([&] { protocol::planned_key(store, protocol::FAMILY_DPF, 5, k, s); }),
              "argument checks of the budget helpers (batch 0, unknown family, circuit that was not exported)");
        check(abb::TripleBudgetError(7, 3).needed == 7 && abb::TripleBudgetError(7, 3).available == 3 &&
                  std::string(abb::TripleBudgetError(7, 3).what()).find("needs 7 AND triples") != std::string::npos &&
                  std::string(abb::TripleBudgetError(7, 3).what()).find("reserve") != std::string::npos,
              "TripleBudgetError carries the two counts and tells the caller to reserve");
    }

    // Session: refill mode by default (every refill is one more population and one more count),
    // strict mode on request (a short pool is refused before any message, nothing changes).
    {
        prep::IdealDealer dealer;
        const bytes sid = {'b', 'u', 's'};
        struct Out {
            std::vector<uint64_t> log;
            std::string info;
            abb::EvalStats last;
        };
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                const std::vector<uint64_t> alpha = {(uint64_t)(role ? 9 : 3)};
                Session ses(role, ch, sid, store, dealer, k, s);
                auto key = [&] { protocol::gen_dpf(ses, 4, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, alpha); };
                Out out;
                std::vector<uint64_t>& log = out.log;
                log.push_back(ses.allow_refill() && ses.triple_requests() == 0 && ses.triples_requested() == 0);  // default: refill mode
                ses.setup(N_KEY, 2);  // the setup circuit, one key and two on-demand masks in ONE request
                log.push_back(ses.triple_requests());
                log.push_back(ses.triples_requested());
                log.push_back(ses.triples_unused());
                ses.set_allow_refill(false);  // strict from here on
                key();                        // fits
                log.push_back(ses.triple_requests());
                log.push_back(ses.triples_unused());
                const uint64_t sent = ch.stats().msgs_sent, evals = ses.engine().evaluations();
                size_t needed = 0, available = 0;
                bool refused = false;
                try {
                    key();  // the pool holds the mask budget only
                } catch (const abb::TripleBudgetError& e) {
                    refused = true;
                    needed = e.needed;
                    available = e.available;
                }
                log.push_back(refused && needed == N_KEY && available == 2 * N_OM && ch.stats().msgs_sent == sent && ses.active() &&
                              !ses.engine().dead() && ses.triples_unused() == 2 * N_OM && ses.engine().evaluations() == evals &&
                              ses.triple_requests() == 1);
                log.push_back(throws_invalid([&] { protocol::gen_gate(ses, tab.descriptor()); }) && ses.active() &&
                              ch.stats().msgs_sent == sent);  // TripleBudgetError is a std::invalid_argument
                ses.reserve_triples(N_KEY - 2 * N_OM);  // explicit: allowed in strict mode, one more population
                log.push_back(ses.triple_requests());
                key();  // uses exactly the pool
                log.push_back(ses.triples_unused());
                ses.set_allow_refill(true);
                key();  // refilled silently: a third population
                log.push_back(ses.triple_requests());
                log.push_back(ses.triples_requested());
                log.push_back(ses.triples_used());
                double bound = 0;
                log.push_back(ses.bucketing_bound_log2(&bound) ? 1 : 0);  // the ideal dealer has no bucketing
                out.info = ses.info_json();
                out.last = ses.last_gen.stats;
                return out;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "session budget: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            const std::vector<uint64_t> want = {1, 1, N_SETUP + N_KEY + 2 * N_OM, N_KEY + 2 * N_OM, 1, 2 * N_OM, 1, 1, 2, 0, 3,
                                                N_SETUP + 3 * N_KEY, N_SETUP + 3 * N_KEY, 0};
            check(pr.first.value.log == want && pr.second.value.log == want,
                  "Session: one request for setup + reservation; strict mode refuses a short pool before any message; an explicit "
                  "reservation and a refill are one population each");
            const json::Value info = json::parse(pr.first.value.info);
            check(info.at("triple_requests").as_u64() == 3 && info.at("triples_requested").as_u64() == N_SETUP + 3 * N_KEY &&
                      info.at("triples_unused").as_u64() == 0 && info.at("allow_refill").as_bool() && info.at("prep").str() == "ideal" &&
                      info.at("state").str() == "ACTIVE" && info.at("evaluations").as_u64() == 4 && info.at("k").as_u64() == 64 &&
                      pr.first.value.info.find("\"bucketing_bound_log2\": null") != std::string::npos &&
                      pr.first.value.info.find("\"last_bucket_size\": null") != std::string::npos,
                  "Session::info_json reports the populations (" + pr.first.value.info + ")");
            const abb::EvalStats& st = pr.first.value.last;
            check(st.triple_requests == 3 && st.triples_requested == N_SETUP + 3 * N_KEY && st.triples_available == 0 &&
                      st.json().find("\"triple_requests\":3,\"triples_requested\":" + std::to_string(N_SETUP + 3 * N_KEY) +
                                     ",\"triples_available\":0") != std::string::npos,
                  "EvalStats carries triple_requests, triples_requested and triples_available");
        }
    }

    // Real preprocessing: the bucketing term of a session is requests * 2^-sigma.
    {
        const bytes sid = {'b', 'u', 'r'};
        struct Out {
            bool before = true;
            double one = 0, two = 0;
            uint64_t requests = 0;
            unsigned bucket = 0;
            std::string info;
        };
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, k, s, 40);
                Out out;
                double b = 0;
                out.before = ses.bucketing_bound_log2(&b);  // no request yet
                ses.setup();                                 // one population: the 64 triples of the setup circuit
                ses.bucketing_bound_log2(&out.one);
                ses.reserve_triples(100);                    // a second population
                ses.bucketing_bound_log2(&out.two);
                out.requests = ses.triple_requests();
                out.bucket = ses.real_prep()->last_bucket_size();
                out.info = ses.info_json();
                return out;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "real budget: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok)
            for (const auto* o : {&pr.first, &pr.second})
                check(!o->value.before && o->value.one == -40.0 && o->value.two == -39.0 && o->value.requests == 2 &&
                          o->value.bucket == prep::bucket_size(100, 40) &&
                          o->value.info.find("\"bucketing_bound_log2\": -39.000000") != std::string::npos &&
                          o->value.info.find("\"prep\": \"real\"") != std::string::npos &&
                          o->value.info.find("\"last_bucket_size\": " + std::to_string(o->value.bucket)) != std::string::npos,
                      "real preprocessing: log2 of the bucketing term is -40 for one population and -39 for two");
    }

    // The driver is strict by default: Key, Gate and on-demand output masks are served from the
    // one reservation of setup; what does not fit is refused before any message and before any
    // change of state, and a bad identifier is still reported first, by the registry.
    {
        prep::IdealDealer dealer;
        const bytes sid = {'b', 'u', 'p'};
        const uint64_t x = 201, xs[2] = {150, (x - 150) & 0xFF};
        struct Out {
            std::vector<uint64_t> log;
            std::vector<u128> opened;
        };
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                const bool was_refill = ses.allow_refill();
                Party p(ses);
                Out out;
                std::vector<uint64_t>& log = out.log;
                log.push_back(was_refill && !ses.allow_refill() && !p.allow_refill());  // the party switched its session to strict mode
                log.push_back(p.key_budget(protocol::FAMILY_DPF, 4));
                log.push_back(p.key_budget(protocol::FAMILY_DPF, 4, false, protocol::INDEX_XOR, 3));
                log.push_back(p.gate_budget(tab) == p.gate_budget(tab.descriptor()) ? p.gate_budget(tab, 2) : 0);
                log.push_back(p.outmask_budget() + p.outmask_budget(0) + p.outmask_budget(3));
                p.setup(N_KEY + N_TAB, 1);  // one key, one table gate, one on-demand output mask
                log.push_back(p.triple_requests());
                const uint64_t alpha = role ? 9 : 3;
                // Two gates do not fit (the pool holds one gate, one key and one mask), one does.
                uint64_t sent = ch.stats().msgs_sent;
                bool refused = throws<abb::TripleBudgetError>([&] { p.gate_batch(Party::NEXT, tab, 2); });
                state::Registry::Summary sm = p.registry().summary();
                log.push_back(refused && p.registry().counters().next[state::Registry::GID] == 0 && sm.gates_generating == 0 &&
                              ch.stats().msgs_sent == sent && p.active() && ses.triples_unused() == N_KEY + N_TAB + N_OM);
                const uint64_t gid = p.gate(Party::NEXT, tab);
                const uint64_t kid = p.key(Party::NEXT, protocol::FAMILY_DPF, 4, alpha);  // fits
                // A second key does not fit: only the mask budget is left.
                sent = ch.stats().msgs_sent;
                refused = throws<abb::TripleBudgetError>([&] { p.key(Party::NEXT, protocol::FAMILY_DPF, 4, alpha); });
                sm = p.registry().summary();
                log.push_back(refused && p.registry().counters().next[state::Registry::KID] == kid + 1 && sm.keys_generating == 0 &&
                              sm.keys_certified == 1 && ch.stats().msgs_sent == sent && p.active());
                // The order of the two refusals: an identifier that was issued is a StateError, not a budget error.
                log.push_back(throws<state::StateError>([&] { p.key(kid, protocol::FAMILY_DPF, 4, alpha); }) &&
                              throws<state::StateError>([&] { p.gate(gid, tab); }) && p.active() && ch.stats().msgs_sent == sent);
                log.push_back(ses.triples_unused());  // the mask budget is left
                p.mask_shared(gid, xs[role]);
                const AShare t = p.apply(gid)[0];
                const AShare y = p.lin({{u128(3), t}}, u128(5)), y2 = p.lin({{u128(5), t}}, u128(1));  // neither is exact
                // Two masks are needed, one was reserved: refused before the opening.
                sent = ch.stats().msgs_sent;
                const uint64_t openings = p.checker()->openings();
                refused = throws<abb::TripleBudgetError>([&] { p.open_many({y, y2}); });
                log.push_back(refused && ch.stats().msgs_sent == sent && p.checker()->openings() == openings && p.outmasks_available() == 0 &&
                              p.active() && ses.triples_unused() == N_OM);
                out.opened.push_back(p.open(y));  // the reserved mask is generated now, from the pool
                log.push_back(p.triple_requests());
                log.push_back(ses.triples_unused() + p.outmasks_available());
                log.push_back(throws<abb::TripleBudgetError>([&] { p.open(y2); }) && p.active());
                ses.reserve_triples(p.outmask_budget());  // explicit further population
                out.opened.push_back(p.open(y2));
                log.push_back(p.triple_requests());
                const json::Value info = json::parse(p.info_json());
                log.push_back(info.at("state").str() == "ACTIVE" && info.at("session").at("triple_requests").as_u64() == 2 &&
                              !info.at("session").at("allow_refill").as_bool() && info.at("registry").at("gates_used").as_u64() == 1 &&
                              info.at("arith_checks").as_u64() == p.checker()->checks() && !info.at("abort_persist_failed").as_bool());
                return out;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "driver budget: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            const std::vector<uint64_t> want = {1, N_KEY, 3 * N_KEY, 2 * N_TAB, 4 * N_OM, 1, 1, 1, 1, N_OM, 1, 1, 0, 1, 2, 1};
            check(pr.first.value.log == want && pr.second.value.log == want,
                  "Party: strict by default; a key, gates or output masks beyond the reservation are refused without any change of state");
            const uint64_t T = 7;
            for (const auto* o : {&pr.first, &pr.second})
                check(o->value.opened.size() == 2 && o->value.opened[0] == u128(3 * T + 5) && o->value.opened[1] == u128(5 * T + 1),
                      "Party: the values opened from the reserved pool are correct");
        }
    }

    // Refill mode of the driver (explicitly allowed): every generation is its own population.
    {
        prep::IdealDealer dealer;
        const bytes sid = {'b', 'u', 'f'};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses, std::string(), true);
                std::vector<uint64_t> log;
                log.push_back(p.allow_refill());
                p.setup();
                log.push_back(p.triple_requests());
                p.key(Party::NEXT, protocol::FAMILY_DPF, 4, role ? 9 : 3);
                log.push_back(p.triple_requests());
                const uint64_t gid = p.gate(Party::NEXT, tab);
                log.push_back(p.triple_requests());
                p.mask_shared(gid, role ? 20 : 30);
                p.open(p.lin({{u128(3), p.apply(gid)[0]}}, u128(5)));  // an on-demand output mask
                log.push_back(p.triple_requests());
                log.push_back(ses.triples_unused());
                return log;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        const std::vector<uint64_t> want = {1, 1, 2, 3, 4, 0};
        check(pr.first.ok && pr.second.ok && pr.first.value == want && pr.second.value == want,
              "Party(session, path, allow_refill = true): setup, key, gate and one on-demand mask are four populations (" +
                  pr.first.error + pr.second.error + ")");
    }
}

// ------------------------------------------------------------ Mask of a shared input (CHANGELOG F1 W1)
// A party that deviates in the exchange of the masked shares.
class CheatMaskParty : public Party {
public:
    enum Mode {
        HONEST,
        RUSH_CHANGE,       // P_1: wait for the honest opening, then open the share that makes x_hat = target
        RUSH_KEEP,         // P_1: wait for the honest opening, then open what was committed (control)
        MIRROR_BOTH,       // P_1: answer the commitment and the opening with copies of them
        MIRROR_OPENING,    // P_1: own commitment, but a copy of the honest opening
        OTHER_GID,         // commitment and opening made for another gate
        SHORT_COMMITMENT,  // 31 bytes
        SHORT_OPENING,     // 31 bytes
        CHANGED_OPENING,   // one bit of omega flipped
        LONG_MESSAGE,      // a committed message of ceil(n/8) + 1 bytes
        EMPTY_MESSAGE,     // a committed empty message
        LARGE_VALUE,       // a committed value >= 2^n (n not a multiple of 8)
        PLAIN_EXCHANGE     // the share in the clear, as before W1
    };
    CheatMaskParty(Session& ses, Mode mode, unsigned n, uint64_t target) : Party(ses), mode_(mode), n_(n), target_(target) {}

protected:
    bytes mask_exchange(uint64_t gid, const bytes& msg) override {
        if (mode_ == HONEST) return Party::mask_exchange(gid, msg);
        Channel& ch = session().channel();
        const int me = ch.party();
        if (mode_ == PLAIN_EXCHANGE) return ch.exchange(msg);
        static const char prefix[] = "mask|";
        bytes label(prefix, prefix + 5);
        const std::array<uint8_t, 8> g = hashes::le64(mode_ == OTHER_GID ? gid + 1 : gid);
        label.insert(label.end(), g.begin(), g.end());
        bytes committed = msg;
        if (mode_ == LONG_MESSAGE) committed.push_back(0);
        if (mode_ == EMPTY_MESSAGE) committed.clear();
        if (mode_ == LARGE_VALUE) committed.back() |= 0x80;
        SystemRng sys;
        hashes::opening32 omega;
        sys.fill(omega.data(), omega.size());
        const hashes::hash32 com = hashes::commit_with_opening(sid_mask(), committed, omega, me, label);
        bytes com_msg(com.begin(), com.end());
        if (mode_ == SHORT_COMMITMENT) com_msg.pop_back();
        auto opening = [&](const bytes& m) {
            bytes o(omega.begin(), omega.end());
            o.insert(o.end(), m.begin(), m.end());
            return o;
        };
        bytes peer_open;
        if (mode_ == MIRROR_BOTH) {
            const bytes peer_com = ch.recv();
            ch.send(peer_com);
            ch.flight();
            peer_open = ch.recv();
            ch.send(peer_open);
            ch.flight();
        } else if (mode_ == RUSH_CHANGE || mode_ == RUSH_KEEP || mode_ == MIRROR_OPENING) {
            if (me != 1) throw std::logic_error("only party 1 sees the other message of an exchange first");
            ch.exchange(com_msg);
            peer_open = ch.recv();  // the honest opening is known before this party opens
            bytes mine = committed;
            if (mode_ == RUSH_CHANGE && peer_open.size() == 32 + msg.size()) {
                uint64_t peer = 0;
                for (size_t i = 0; i < msg.size(); i++) peer |= (uint64_t)peer_open[32 + i] << (8 * i);
                const uint64_t want = (target_ - peer) & gates::domain_mask(n_);
                for (size_t i = 0; i < mine.size(); i++) mine[i] = (uint8_t)(want >> (8 * i));
            }
            ch.send(mode_ == MIRROR_OPENING ? peer_open : opening(mine));
            ch.flight();
        } else {
            ch.exchange(com_msg);
            bytes mine_open = opening(committed);
            if (mode_ == SHORT_OPENING) mine_open.resize(31);
            if (mode_ == CHANGED_OPENING) mine_open[3] ^= 1;
            peer_open = ch.exchange(mine_open);
        }
        if (peer_open.size() < 32) throw std::runtime_error("cheater: the honest opening is too short");
        return bytes(peer_open.begin() + 32, peer_open.end());
    }

private:
    Mode mode_;
    unsigned n_;
    uint64_t target_;
};

static void test_mask_shared(circuit::CircuitStore& store) {
    Section sec("Mask of a shared input (commit-then-open)");
    const unsigned k = 64, s = 64;

    // The known answers of the specification (CHANGELOG F1 W1), spelled out byte by byte.
    {
        auto le64 = [](bytes& out, uint64_t v) {
            for (int i = 0; i < 8; i++) out.push_back((uint8_t)(v >> (8 * i)));
        };
        const std::string tag = "Couplet-mask-sid", sid_text = "interop-online";
        bytes raw;
        le64(raw, 16);
        raw.insert(raw.end(), tag.begin(), tag.end());
        le64(raw, sid_text.size());
        raw.insert(raw.end(), sid_text.begin(), sid_text.end());
        uint8_t digest[32];
        sha256(raw.data(), raw.size(), digest);
        prep::IdealDealer dealer;
        const auto chans = LoopbackChannel::make_pair();
        Session ses(0, *chans.first, bytes(sid_text.begin(), sid_text.end()), store, dealer, k, s);
        Party p(ses);
        check(tag.size() == 16 && hex_of(digest, 32) == "51b14489101769cae73640fa4d0978b46d1dd80f04b860cedec27118aecf8235" &&
                  hex_of(p.sid_mask()) == hex_of(digest, 32),
              "sid_mask = SHA-256(le64(16) || \"Couplet-mask-sid\" || le64(len(sid)) || sid), known answer");
    }

    // Honest runs: the bytes on the wire are recomputed from the specification.
    struct Honest {
        uint64_t x_hat = 0, mine = 0, rounds = 0, bytes_sent = 0, bytes_recv = 0, triple_requests = 0;
        std::vector<bytes> sent, received;
        prep::PhaseStat phase;
        bytes sid_mask;
    };
    struct Case {
        GateSpec spec;
        uint64_t gid;
        const char* name;
    };
    const std::vector<Case> cases = {
        {GateSpec::table_gate(8, k, s, std::vector<uint64_t>(256, 1), false), 7, "n = 8 (1 byte), gate 7"},
        {GateSpec::table_gate(12, k, s, std::vector<uint64_t>(4096, 1), false), 300, "n = 12 (2 bytes), gate 300"},
        {GateSpec::offset_only(1, k, s, false, 1), 0, "n = 1 (1 byte), gate 0"},
    };
    CtrPrgRng rng(101);
    for (const Case& c : cases) {
        if (g_quick && c.spec.n == 12) continue;
        const unsigned n = c.spec.n, nb = (n + 7) / 8;
        const uint64_t x = rng.u64() & mask_bits(n), x0 = rng.u64() & mask_bits(n), xsh[2] = {x0, (x - x0) & mask_bits(n)};
        prep::IdealDealer dealer;
        const bytes sid = {'m', 'k', (uint8_t)n};
        auto party = [&](int role) {
            return [&, role](Channel& raw) {
                RecordChannel ch(raw);
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses);
                p.setup(p.gate_budget(c.spec));
                const uint64_t gid = p.gate(c.gid, c.spec);
                Honest out;
                out.mine = (xsh[role] + u128_lo(p.registry().gate(gid).material().rp_share)) & mask_bits(n);
                const size_t s0 = ch.sent.size(), r0 = ch.received.size();
                const ChannelStats before = ch.stats();
                out.x_hat = p.mask_shared(gid, xsh[role]);
                out.sent.assign(ch.sent.begin() + (ptrdiff_t)s0, ch.sent.end());
                out.received.assign(ch.received.begin() + (ptrdiff_t)r0, ch.received.end());
                out.rounds = ch.stats().rounds - before.rounds;
                out.bytes_sent = ch.stats().bytes_sent - before.bytes_sent;
                out.bytes_recv = ch.stats().bytes_recv - before.bytes_recv;
                out.phase = ses.book().get("mask");
                out.sid_mask = p.sid_mask();
                out.triple_requests = p.triple_requests() + ses.triples_unused();
                return out;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        const std::string what = std::string("honest Mask, ") + c.name;
        check(pr.first.ok && pr.second.ok, what + ": no abort (" + pr.first.error + pr.second.error + ")");
        if (!pr.first.ok || !pr.second.ok) continue;
        const Honest* h[2] = {&pr.first.value, &pr.second.value};
        bytes label = {0x6d, 0x61, 0x73, 0x6b, 0x7c};  // "mask|"
        for (int i = 0; i < 8; i++) label.push_back((uint8_t)(c.gid >> (8 * i)));
        if (c.gid == 7) check(hex_of(label) == "6d61736b7c0700000000000000", "label of gate 7: \"mask|\" || le64(7), known answer");
        for (int b = 0; b < 2; b++) {
            const Honest& o = *h[b];
            const bool shape = o.sent.size() == 2 && o.received.size() == 2 && o.sent[0].size() == 32 && o.sent[1].size() == 32 + nb;
            check(shape, what + ": two messages per party, 32 and 32 + ceil(n/8) bytes");
            if (!shape) continue;
            const bytes omega(o.sent[1].begin(), o.sent[1].begin() + 32), msg(o.sent[1].begin() + 32, o.sent[1].end());
            bytes want_msg(nb);
            for (unsigned i = 0; i < nb; i++) want_msg[i] = (uint8_t)(o.mine >> (8 * i));
            check(msg == want_msg, what + ": the opened message is x_b + r_b mod 2^n, little-endian");
            // com_b = SHA-256(frame("Couplet-com", sid_mask, label, byte(b), omega_b, msg_b)), the frame written out.
            const std::string com_tag = "Couplet-com";
            const bytes party_byte(1, (uint8_t)b);
            bytes frame;
            for (const bytes& part : {bytes(com_tag.begin(), com_tag.end()), o.sid_mask, label, party_byte, omega, msg}) {
                for (int i = 0; i < 8; i++) frame.push_back((uint8_t)((uint64_t)part.size() >> (8 * i)));
                frame.insert(frame.end(), part.begin(), part.end());
            }
            uint8_t com[32];
            sha256(frame.data(), frame.size(), com);
            check(o.sent[0] == bytes(com, com + 32), what + ": the commitment is bound to sid_mask, the gate label and the party index");
            check(o.received[0] == h[1 - b]->sent[0] && o.received[1] == h[1 - b]->sent[1], what + ": each party receives what the other sent");
            check(o.rounds == 2 && o.bytes_sent == 64 + nb && o.bytes_recv == 64 + nb, what + ": 2 rounds and 64 + ceil(n/8) bytes per party");
            check(o.phase.rounds == 2 && o.phase.bytes_sent == 64 + nb && o.phase.msgs_sent == 2, what + ": both messages are booked under the phase mask");
            check(o.triple_requests == 1, what + ": one triple request, no triple left");
        }
        check(h[0]->x_hat == h[1]->x_hat && h[0]->x_hat == ((h[0]->mine + h[1]->mine) & mask_bits(n)), what + ": x_hat = msg_0 + msg_1 mod 2^n at both parties");
        check(h[0]->sent[1] != h[1]->sent[1] && h[0]->sent[0] != h[1]->sent[0], what + ": the two parties' messages differ");
    }

    // Deviations.  The honest party aborts with the stated reason, records no x_hat, sends
    // nothing afterwards (a retry on the same gate included) and has sent no arithmetic opening.
    struct Dev {
        CheatMaskParty::Mode mode;
        int cheater;  // -1: both roles in turn
        const char* want;
        const char* name;
    };
    const Dev devs[] = {
        {CheatMaskParty::RUSH_CHANGE, 1, "committed_exchange: commitment opening failed", "a rushing party opens another share than it committed to"},
        {CheatMaskParty::MIRROR_BOTH, 1, "committed_exchange: mirrored commitment", "the commitment and the opening are mirrored"},
        {CheatMaskParty::MIRROR_OPENING, 1, "committed_exchange: commitment opening failed", "the opening alone is mirrored"},
        {CheatMaskParty::OTHER_GID, -1, "committed_exchange: commitment opening failed", "commitment and opening of another gate"},
        {CheatMaskParty::SHORT_COMMITMENT, -1, "committed_exchange: malformed commitment", "a commitment of 31 bytes"},
        {CheatMaskParty::SHORT_OPENING, -1, "committed_exchange: malformed opening", "an opening of 31 bytes"},
        {CheatMaskParty::CHANGED_OPENING, -1, "committed_exchange: commitment opening failed", "a changed opening"},
        {CheatMaskParty::LONG_MESSAGE, -1, "committed_exchange: opening has the wrong length", "a committed message that is too long"},
        {CheatMaskParty::EMPTY_MESSAGE, -1, "mask: message has the wrong length", "a committed empty message"},
        {CheatMaskParty::LARGE_VALUE, -1, "mask: share out of range", "a committed value >= 2^n"},
        {CheatMaskParty::PLAIN_EXCHANGE, -1, "committed_exchange: malformed commitment", "the share in a plain exchange (the message before W1)"},
    };
    const GateSpec bit = GateSpec::offset_only(1, k, s, false, 1), tab = GateSpec::table_gate(8, k, s, std::vector<uint64_t>(256, 1), false);
    struct Verdict {
        std::string text;
        uint64_t x_hat = 0;
        bool has_x_hat = false;
    };
    auto run_dev = [&](CheatMaskParty::Mode mode, int cheater, const GateSpec& spec, uint64_t target, const uint64_t* shares, const bytes& sid) {
        prep::IdealDealer dealer;
        auto party = [&, mode, cheater, target](int role) {
            return [&, role, mode, cheater, target](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                std::unique_ptr<Party> p(role == cheater ? new CheatMaskParty(ses, mode, spec.n, target) : new Party(ses));
                p->setup(p->gate_budget(spec, 2));
                const std::vector<uint64_t> gids = p->gate_batch(Party::NEXT, spec, 2);
                Verdict v;
                try {
                    v.x_hat = p->mask_shared(gids[0], shares[role]);
                    v.has_x_hat = true;
                    v.text = "x_hat";
                } catch (const ProtocolAbort& e) {
                    v.text = std::string("abort:") + e.what();
                } catch (const std::exception& e) {
                    v.text = std::string("error:") + e.what();
                }
                if (!v.has_x_hat) {
                    const uint64_t sent = ch.stats().msgs_sent;
                    int dead = 0;
                    dead += throws<ProtocolAbort>([&] { p->mask_shared(gids[0], shares[role]); });  // a retry
                    dead += throws<ProtocolAbort>([&] { p->mask_shared(gids[1], shares[role]); });  // another gate
                    dead += throws<ProtocolAbort>([&] { p->apply(gids[0]); });
                    dead += !p->active() && !ses.active() && ses.engine().dead() && ses.prep().aborted() && ch.stats().msgs_sent == sent;
                    dead += p->checker()->openings() == 0 && p->checker()->dead();
                    v.text += "|" + std::to_string(dead);
                }
                return v;
            };
        };
        return run_two_party(party(0), party(1));
    };
    int n_dev = 0;
    for (const Dev& d : devs) {
        for (int cheater = 0; cheater < 2; cheater++) {
            if (d.cheater >= 0 && d.cheater != cheater) continue;
            const GateSpec& spec = d.mode == CheatMaskParty::LARGE_VALUE ? bit : tab;
            const uint64_t shares[2] = {rng.u64() & mask_bits(spec.n), rng.u64() & mask_bits(spec.n)};
            const auto pr = run_dev(d.mode, cheater, spec, 0x5A, shares, bytes{'m', 'd', (uint8_t)d.mode, (uint8_t)cheater});
            const auto& h = cheater == 0 ? pr.second : pr.first;
            check(h.ok && h.value.text == std::string("abort:") + d.want + "|5" && !h.value.has_x_hat,
                  std::string(d.name) + " (party " + std::to_string(cheater) + "): the honest party aborts and the session is retired (" +
                      (h.ok ? h.value.text : h.error) + ")");
            n_dev++;
        }
    }
    // Control: the same rushing party is accepted when it opens what it committed to, and then
    // x_hat is the honest value, not the target.
    {
        const uint64_t shares[2] = {0x31, 0x42};
        const auto pr = run_dev(CheatMaskParty::RUSH_KEEP, 1, tab, 0x5A, shares, bytes{'m', 'c'});
        check(pr.first.ok && pr.second.ok && pr.first.value.has_x_hat && pr.second.value.has_x_hat &&
                  pr.first.value.x_hat == pr.second.value.x_hat,
              "(control) a rushing party that opens what it committed to is accepted (" + pr.first.value.text + pr.first.error + ")");
        const auto honest = run_dev(CheatMaskParty::HONEST, 1, tab, 0, shares, bytes{'m', 'h'});
        check(honest.first.ok && honest.second.ok && honest.first.value.has_x_hat && honest.first.value.x_hat == honest.second.value.x_hat,
              "(control) the cheating class in honest mode runs the protocol");
    }
    std::printf("  shared-input Mask: %d deviations, each aborted by the honest party\n", n_dev);
}

// ------------------------------------------------------------ gates end to end
struct GateRun {
    std::vector<u128> opened;      // released values
    std::vector<uint64_t> x_hat;
    uint64_t delta = 0;
    uint64_t arith_checks = 0, arith_openings = 0;
    uint64_t triple_requests = 0;  // bucketing populations of the session
    size_t triples_left = 0;       // unused triples at the end
    size_t outmasks_left = 0;
    bool state_errors_ok = true;
    double seconds_gen = 0, seconds_online = 0;
};

// Gate, Mask, Apply, Open for every x of `xs` (one gate per x; `chunk` gates per circuit execution).
//   wire = false: the input is additively shared (x0[i] + x1[i] = x mod 2^n);
//   wire = true:  the input is the authenticated wire [[x]] = x * [[1]], where [[1]] is the output
//                 of a DPF key (public payload 1) queried at its index.
static std::pair<PartyOutcome<GateRun>, PartyOutcome<GateRun>> run_gate(circuit::CircuitStore& store, const GateSpec& spec,
                                                                        const std::vector<uint64_t>& xs, bool wire, size_t chunk,
                                                                        const bytes& sid) {
    const unsigned n = spec.n, k = spec.k, s = spec.s;
    CtrPrgRng rng(71);
    std::vector<uint64_t> xsh[2];
    for (uint64_t x : xs) {
        const uint64_t a = rng.u64() & mask_bits(n);
        xsh[0].push_back(a);
        xsh[1].push_back((x - a) & mask_bits(n));
    }
    const uint64_t alpha_sh[2] = {rng.u64() & 15, rng.u64() & 15};
    prep::IdealDealer dealer;
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            typedef std::chrono::steady_clock clock;
            Session ses(role, ch, sid, store, dealer, k, s);
            Party p(ses);  // strict mode: everything below is served from the one reservation of setup
            p.setup((wire ? p.key_budget(protocol::FAMILY_DPF, 4) : 0) + p.gate_budget(spec, xs.size()));
            GateRun out;
            out.delta = ses.delta_share();
            AShare one;
            if (wire) {
                const uint64_t kid = p.key(Party::NEXT, protocol::FAMILY_DPF, 4, alpha_sh[role]);
                one = p.query(kid, alpha_sh[0] ^ alpha_sh[1]);  // exact [[1]]
            }
            for (size_t done = 0; done < xs.size(); done += chunk) {
                const size_t cnt = std::min(chunk, xs.size() - done);
                clock::time_point t0 = clock::now();
                const std::vector<uint64_t> gids = p.gate_batch(Party::NEXT, spec, cnt);
                clock::time_point t1 = clock::now();
                std::vector<AShare> results;
                for (size_t i = 0; i < cnt; i++) {
                    const uint64_t x = xs[done + i];
                    uint64_t x_hat;
                    if (wire) {
                        const AShare xw = p.lin({{u128(x), one}});
                        x_hat = p.mask_wire(gids[i], xw);
                        if (i == 0) {
                            // One offset masks one input: a second Mask is refused and sends nothing.
                            const uint64_t sent = ch.stats().msgs_sent;
                            bool refused = false;
                            try {
                                p.mask_wire(gids[i], xw);
                            } catch (const state::GateUsedError&) {
                                refused = true;
                            }
                            out.state_errors_ok = out.state_errors_ok && refused && ch.stats().msgs_sent == sent && p.active();
                        }
                    } else {
                        x_hat = p.mask_shared(gids[i], xsh[role][done + i]);
                        if (i == 0) {
                            const uint64_t sent = ch.stats().msgs_sent;
                            bool refused = false;
                            try {
                                p.mask_shared(gids[i], xsh[role][done + i]);
                            } catch (const state::GateUsedError&) {
                                refused = true;
                            }
                            out.state_errors_ok = out.state_errors_ok && refused && ch.stats().msgs_sent == sent && p.active();
                        }
                    }
                    out.x_hat.push_back(x_hat);
                    const std::vector<AShare> sh = p.apply(gids[i]);
                    results.insert(results.end(), sh.begin(), sh.end());
                }
                if (wire && p.checker()->pending() != cnt) out.state_errors_ok = false;  // the checks are deferred
                const std::vector<u128> opened = p.open_many(results);
                if (p.checker()->pending() != 0) out.state_errors_ok = false;
                out.opened.insert(out.opened.end(), opened.begin(), opened.end());
                out.seconds_gen += std::chrono::duration<double>(t1 - t0).count();
                out.seconds_online += std::chrono::duration<double>(clock::now() - t1).count();
            }
            out.arith_checks = p.checker()->checks();
            out.arith_openings = p.checker()->openings();
            out.outmasks_left = p.outmasks_available();
            out.triple_requests = p.triple_requests();
            out.triples_left = ses.triples_unused();
            return out;
        };
    };
    return run_two_party(party(0), party(1));
}

static void check_gate_run(const std::pair<PartyOutcome<GateRun>, PartyOutcome<GateRun>>& pr, const GateSpec& spec,
                           const std::vector<uint64_t>& xs, const std::string& what) {
    check(pr.first.ok && pr.second.ok, what + ": no abort (" + pr.first.error + pr.second.error + ")");
    if (!pr.first.ok || !pr.second.ok) return;
    long long good = 0, good_hat = 0;
    for (size_t i = 0; i < xs.size(); i++) {
        const uint64_t want = spec.plain(xs[i])[0];
        good += pr.first.value.opened.at(i) == u128(want) && pr.second.value.opened.at(i) == u128(want);
        good_hat += pr.first.value.x_hat.at(i) == pr.second.value.x_hat.at(i);
    }
    check_many(good, (long long)xs.size(), what + ": both parties open G(x) for every x");
    check_many(good_hat, (long long)xs.size(), what + ": both parties see the same masked input");
    check(pr.first.value.state_errors_ok && pr.second.value.state_errors_ok, what + ": one-time use and deferred checks");
    check(pr.first.value.triple_requests == 1 && pr.second.value.triple_requests == 1 && pr.first.value.triples_left == 0 &&
              pr.second.value.triples_left == 0,
          what + ": the session is one bucketing population, reserved at setup and used up");
    // The masked inputs are x + r for fresh uniform r: over many gates they are spread out.
    if (xs.size() >= 128) {
        std::vector<int> seen((size_t)1 << spec.n, 0);
        size_t distinct = 0;
        for (size_t i = 0; i < xs.size(); i++) distinct += seen[(size_t)((pr.first.value.x_hat[i] - xs[i]) & mask_bits(spec.n))]++ == 0;
        check(distinct > xs.size() / 2, what + ": offsets of different gates differ (" + std::to_string(distinct) + " distinct values of r)");
    }
    std::printf("  %s: %zu gates, generation %.2f s, mask/apply/open %.2f s (P_0)\n", what.c_str(), xs.size(), pr.first.value.seconds_gen,
                pr.first.value.seconds_online);
    std::fflush(stdout);
}

static void test_gates_e2e(circuit::CircuitStore& store) {
    Section sec("gates end to end, ideal preprocessing");
    CtrPrgRng rng(81);
    const unsigned n = 8, k = 64, s = 64;
    std::vector<uint64_t> xs;
    for (uint64_t x = 0; x < 256; x += (g_quick ? 9 : 1)) xs.push_back(x);
    std::vector<uint64_t> T(256);
    for (uint64_t& t : T) t = rng.u64();
    const size_t chunk_table = g_quick ? 16 : 64, chunk_cmp = g_quick ? 8 : 32;

    // Table gate (random table with 64-bit entries), shared input and wire.
    {
        const GateSpec spec = GateSpec::table_gate(n, k, s, T, false);
        check_gate_run(run_gate(store, spec, xs, false, chunk_table, bytes{'t', 's'}), spec, xs, "table gate, shared input");
        const GateSpec spec_w = GateSpec::table_gate(n, k, s, T, true);
        check_gate_run(run_gate(store, spec_w, xs, true, chunk_table, bytes{'t', 'w'}), spec_w, xs, "table gate, authenticated wire");
    }
    // Zero test and equality on the same kind of material (a sample of x).
    {
        std::vector<uint64_t> few = {0, 1, 85, 86, 255};
        const GateSpec zt = GateSpec::zero_test(n, k, s), eq = GateSpec::equality(n, k, s, 85, true);
        check_gate_run(run_gate(store, zt, few, false, 5, bytes{'z', 't'}), zt, few, "zero test, shared input");
        check_gate_run(run_gate(store, eq, few, true, 5, bytes{'e', 'q'}), eq, few, "equality with 85, authenticated wire");
    }
    // Comparison gate [x < 85], shared input and wire.
    {
        const GateSpec spec = GateSpec::comparison(n, k, s, u128(85), false);
        check_gate_run(run_gate(store, spec, xs, false, chunk_cmp, bytes{'c', 's'}), spec, xs, "comparison gate, shared input");
        const GateSpec spec_w = GateSpec::comparison(n, k, s, u128(85), true);
        check_gate_run(run_gate(store, spec_w, xs, true, chunk_cmp, bytes{'c', 'w'}), spec_w, xs, "comparison gate, authenticated wire");
    }

    // A chain of two gates: u (shared) -> table gate -> [[T8[u]]] -> comparison gate (wire) -> open.
    {
        std::vector<uint64_t> T8(256);
        for (uint64_t& t : T8) t = rng.u64() & 0xFF;
        const GateSpec tab = GateSpec::table_gate(n, k, s, T8, false), cmp = GateSpec::comparison(n, k, s, u128(85), true);
        std::vector<uint64_t> us;
        for (int i = 0; i < (g_quick ? 4 : 16); i++) us.push_back(rng.u64() & 0xFF);
        std::vector<uint64_t> ush[2];
        for (uint64_t u : us) {
            const uint64_t a = rng.u64() & 0xFF;
            ush[0].push_back(a);
            ush[1].push_back((u - a) & 0xFF);
        }
        prep::IdealDealer dealer;
        const bytes sid = {'c', 'h'};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses);
                p.setup(p.gate_budget(tab, us.size()) + p.gate_budget(cmp, us.size()));
                const std::vector<uint64_t> g1 = p.gate_batch(Party::NEXT, tab, us.size()), g2 = p.gate_batch(Party::NEXT, cmp, us.size());
                std::vector<AShare> out;
                for (size_t i = 0; i < us.size(); i++) {
                    p.mask_shared(g1[i], ush[role][i]);
                    const AShare mid = p.apply(g1[i])[0];        // exact [[T8[u]]]
                    p.mask_wire(g2[i], mid);
                    out.push_back(p.apply(g2[i])[0]);
                }
                const std::vector<u128> opened = p.open_many(out);
                if (p.triple_requests() != 1 || ses.triples_unused() != 0) throw std::logic_error("the chain is not one bucketing population");
                return opened;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "gate chain: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            long long good = 0;
            for (size_t i = 0; i < us.size(); i++)
                good += pr.first.value[i] == u128(T8[us[i]] < 85 ? 1 : 0) && pr.second.value[i] == pr.first.value[i];
            check_many(good, (long long)us.size(), "gate chain: [T[u] < 85]");
        }
    }

    // Lin and a masked release: 3 * [[T[x]]] + 5 is not exact; Open takes a fresh output mask from Pi_Gen.
    {
        const GateSpec tab = GateSpec::table_gate(n, k, s, T, false);
        prep::IdealDealer dealer;
        const bytes sid = {'l', 'n'};
        const uint64_t x = 77, x0 = rng.u64() & 0xFF, x1 = (x - x0) & 0xFF;
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses);
                p.setup(p.gate_budget(tab), 3);  // one gate, and the triples of the three output masks generated below
                const uint64_t gid = p.gate(Party::NEXT, tab);
                p.mask_shared(gid, role == 0 ? x0 : x1);
                const AShare t = p.apply(gid)[0];
                const AShare y = p.lin({{u128(3), t}}, u128(5));
                std::vector<u128> out;
                if (y.exact || p.outmasks_available() != 0) throw std::logic_error("flags");
                p.reserve_outmasks(3);
                out.push_back(p.open(y));                         // uses one mask
                out.push_back(p.open(p.lin({{u128(1), t}})));     // exact: no mask
                out.push_back(u128((uint64_t)p.outmasks_available()));
                out.push_back(p.open(p.lin({{u128(0) - u128(1), t}, {u128(2), y}})));  // -T + 2 (3 T + 5) = 5 T + 10
                out.push_back(u128((uint64_t)p.outmasks_available()));
                out.push_back(u128(p.triple_requests()));
                out.push_back(u128((uint64_t)ses.triples_unused()));
                return out;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "Lin and masked release: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            const uint64_t t = T[x];
            for (const auto* o : {&pr.first, &pr.second})
                check(o->value[0] == u128(3 * t + 5) && o->value[1] == u128(t) && o->value[2] == u128(2) && o->value[3] == u128(5 * t + 10) &&
                          o->value[4] == u128(1) && o->value[5] == u128(1) && o->value[6] == u128(0),
                      "Lin and masked release: values modulo 2^k, one mask per non-exact opening, one triple request");
        }
    }

    // ReLU for n = k = 8 (two's complement), the whole pipeline with an authenticated wire and an
    // output mask: u (shared) -> table gate (identity) -> [[u]] -> relu_large -> open.
    if (!g_quick) {
        const unsigned k8 = 8;
        std::vector<uint64_t> id(256);
        for (size_t i = 0; i < 256; i++) id[i] = i;
        const GateSpec tab = GateSpec::table_gate(8, k8, s, id, false), relu = GateSpec::relu_large(k8, s);
        std::vector<uint64_t> us = {0, 1, 127, 128, 129, 255, 77, 200};
        std::vector<uint64_t> ush[2];
        for (uint64_t u : us) {
            const uint64_t a = rng.u64() & 0xFF;
            ush[0].push_back(a);
            ush[1].push_back((u - a) & 0xFF);
        }
        prep::IdealDealer dealer;
        const bytes sid = {'r', 'l'};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k8, s);
                Party p(ses);
                p.setup(p.gate_budget(tab, us.size()) + p.gate_budget(relu, us.size()));  // the ReLU gates bring their masks
                const std::vector<uint64_t> g1 = p.gate_batch(Party::NEXT, tab, us.size()), g2 = p.gate_batch(Party::NEXT, relu, us.size());
                if (p.outmasks_available() != us.size()) throw std::logic_error("the ReLU gates bring their output masks");
                std::vector<AShare> out;
                for (size_t i = 0; i < us.size(); i++) {
                    p.mask_shared(g1[i], ush[role][i]);
                    p.mask_wire(g2[i], p.apply(g1[i])[0]);
                    out.push_back(p.apply(g2[i])[0]);
                    if (out.back().exact) throw std::logic_error("ReLU must not be exact");
                }
                std::vector<u128> opened = p.open_many(out);
                opened.push_back(u128((uint64_t)p.outmasks_available()));
                if (p.triple_requests() != 1 || ses.triples_unused() != 0) throw std::logic_error("the pipeline is not one bucketing population");
                return opened;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "ReLU pipeline: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            long long good = 0;
            for (size_t i = 0; i < us.size(); i++) good += pr.first.value[i] == u128(gates::plain_relu(us[i], 8)) && pr.second.value[i] == pr.first.value[i];
            check_many(good, (long long)us.size(), "ReLU pipeline (n = k = 8): ReLU(u) for boundary and random u");
            check(pr.first.value.back() == u128(0), "ReLU pipeline: every output mask was used exactly once");
        }
    }
}

// ------------------------------------------------------------ the order of Open
// Open, step by step (the protocol figure of the paper): (1) ArithCheck on the deferred
// openings, (2) output masks, generated by Pi_Gen if the pool is short, (3) the opening,
// (4) its ArithCheck.  The mirror of test_open_order in python/tests/test_autheval.py.
static void test_open_order(circuit::CircuitStore& store) {
    Section sec("Open: deferred check, then output masks, then the opening and its check");
    const unsigned n = 8, k = 64, s = 64;
    CtrPrgRng rng(131);
    std::vector<uint64_t> T(256);
    for (uint64_t& t : T) t = rng.u64();
    const GateSpec tab = GateSpec::table_gate(n, k, s, T, false), cmp = GateSpec::comparison(n, k, s, u128(85), true);
    const uint64_t x = rng.u64() & 0xFF, x0 = rng.u64() & 0xFF, xsh[2] = {x0, (x - x0) & 0xFF};
    const uint64_t inner = 3 * T[x] + 5;                                 // modulo 2^64
    const u128 want = u128((uint64_t)(7 * ((inner & 0xFF) < 85 ? 1 : 0) + 1));
    // Messages of one ArithCheck over one modulus of `bits` bits: commitment and opening of the
    // coin share, commitment and opening of zeta.
    auto check_sizes = [](unsigned bits) { return std::vector<size_t>{32, 64, 32, 32 + (size_t)spdz2k::share_bytes(bits)}; };

    struct Out {
        std::string verdict;
        u128 value = u128(0);
        uint64_t x_hat = 0, pending = 0, pool = 0, checks = 0, evaluations = 0, requests = 0, unused = 0, pending_after = 0;
        std::vector<size_t> sizes;  // of the messages this party sent during Open
        bool quiet = false, active = false, retired = false;
    };
    // Two gates; a non-exact wire is masked into the second one (a deferred opening), and the
    // value to be opened is not exact either, so Open needs an output mask.
    auto prepare = [&](Party& p, int role, uint64_t& x_hat) {
        const uint64_t g1 = p.gate(Party::NEXT, tab), g2 = p.gate(Party::NEXT, cmp);
        p.mask_shared(g1, xsh[role]);
        const AShare s1 = p.lin({{u128(3), p.apply(g1)[0]}}, u128(5));   // not exact
        x_hat = p.mask_wire(g2, s1);                                     // opened, not yet verified
        return p.lin({{u128(7), p.apply(g2)[0]}}, u128(1));              // not exact
    };

    // (a) honest run with an empty mask pool: check, then mask generation, then opening and check
    {
        prep::IdealDealer dealer;
        const bytes sid = {'o', 'o', 'a'};
        auto party = [&](int role) {
            return [&, role](Channel& raw) {
                RecordChannel ch(raw);
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses);
                p.setup(p.gate_budget(tab) + p.gate_budget(cmp), 1);
                Out o;
                const AShare s2 = prepare(p, role, o.x_hat);
                o.pending = p.checker()->pending();
                o.pool = p.outmasks_available();
                const uint64_t checks0 = p.checker()->checks(), evals0 = ses.engine().evaluations();
                const size_t mark = ch.sent.size();
                o.value = p.open(s2);
                for (size_t i = mark; i < ch.sent.size(); i++) o.sizes.push_back(ch.sent[i].size());
                o.checks = p.checker()->checks() - checks0;
                o.evaluations = ses.engine().evaluations() - evals0;
                o.pending_after = p.checker()->pending();
                o.requests = p.triple_requests();
                o.unused = ses.triples_unused() + p.outmasks_available();
                return o;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "Open order, honest: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            const Out& a = pr.first.value;
            const Out& b = pr.second.value;
            check(a.value == want && b.value == want && a.x_hat == b.x_hat, "Open of a non-exact value behind a wire-mode Mask");
            check(a.pending == 1 && a.pool == 0, "before Open: one deferred opening, no output mask");
            const std::vector<size_t>& z = a.sizes;
            const std::vector<size_t> first = check_sizes(n + s), last = check_sizes(k + s);
            check(z.size() >= 9 + 7 && std::vector<size_t>(z.begin(), z.begin() + 4) == first,
                  "Open starts with the ArithCheck on the deferred opening (32, 64, 32, 41 bytes)");
            check(z.size() >= 9 + 7 && z[z.size() - 5] == spdz2k::share_bytes(k + s) &&
                      std::vector<size_t>(z.end() - 4, z.end()) == last,
                  "Open ends with the opening (16 bytes) and its check (32, 64, 32, 48 bytes)");
            check(a.checks == 2 && a.evaluations == 1 && a.pending_after == 0 && b.checks == 2 && b.evaluations == 1,
                  "two checks and one circuit evaluation (the output mask) between them");
            check(a.requests == 1 && a.unused == 0 && b.requests == 1 && b.unused == 0,
                  "one triple request, every reserved triple and the mask used");
            check(b.sizes.size() == z.size() && std::vector<size_t>(b.sizes.begin(), b.sizes.begin() + 4) == first,
                  "both parties send in the same order");
        }
    }

    // (b) strict mode: without triples for the mask, Open is refused before the first check
    {
        prep::IdealDealer dealer;
        const bytes sid = {'o', 'o', 'b'};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses);
                p.setup(p.gate_budget(tab) + p.gate_budget(cmp));  // nothing for an output mask
                Out o;
                const AShare s2 = prepare(p, role, o.x_hat);
                const ChannelStats before = ch.stats();
                const bool refused = throws<abb::TripleBudgetError>([&] { p.open(s2); });
                o.quiet = refused && ch.stats().msgs_sent == before.msgs_sent && ch.stats().msgs_recv == before.msgs_recv;
                o.pending = p.checker()->pending();
                o.checks = p.checker()->checks();
                o.active = p.active();
                ses.reserve_triples(p.outmask_budget(1));  // an explicit second population
                o.value = p.open(s2);
                o.requests = p.triple_requests();
                return o;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "Open order, budget: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok)
            for (const auto* o : {&pr.first, &pr.second})
                check(o->value.quiet && o->value.pending == 1 && o->value.checks == 0 && o->value.active &&
                          o->value.value == want && o->value.requests == 2,
                      "a mask generation beyond the budget is refused before any message; the opening stays pending");
    }

    // (c) a wrong share in the wire-mode Mask: the first check of Open rejects and nothing else is
    //     sent, in particular no output mask is generated for the faulty session
    for (int cheater = 0; cheater < 2; cheater++) {
        prep::IdealDealer dealer;
        const bytes sid = {'o', 'o', 'c', (uint8_t)cheater};
        auto party = [&](int role) {
            return [&, role](Channel& raw) {
                RecordChannel ch(raw);
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses);
                if (role == cheater)
                    p.set_checker_factory([&](int r, Channel& c, const bytes& sd, uint64_t d, unsigned ss, abb::PhaseBook*) {
                        // Opening 0 is the wire-mode Mask.
                        return std::unique_ptr<spdz2k::ArithChecker>(new CheatChecker(r, c, sd, d, ss, CheatChecker::VALUE, u128(1), 0));
                    });
                p.setup(p.gate_budget(tab) + p.gate_budget(cmp), 1);
                Out o;
                const AShare s2 = prepare(p, role, o.x_hat);
                const uint64_t evals0 = ses.engine().evaluations();
                const size_t mark = ch.sent.size();
                try {
                    p.open(s2);
                    o.verdict = "opened";
                } catch (const ProtocolAbort& e) {
                    o.verdict = std::string("abort:") + e.what();
                } catch (const std::exception& e) {
                    o.verdict = std::string("error:") + e.what();
                }
                for (size_t i = mark; i < ch.sent.size(); i++) o.sizes.push_back(ch.sent[i].size());
                o.evaluations = ses.engine().evaluations() - evals0;
                o.retired = !p.active() && !ses.active() && ses.engine().dead() && !p.registry().active() &&
                            p.outmasks_available() == 0 && throws<ProtocolAbort>([&] { p.open(s2); });
                return o;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        const std::string tag = "Open order, P_" + std::to_string(cheater) + " shifts a masked wire";
        check(pr.first.ok && pr.second.ok, tag + ": the parties return (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            const Out& h = cheater == 0 ? pr.second.value : pr.first.value;
            const Out& c = cheater == 0 ? pr.first.value : pr.second.value;
            check(h.verdict == "abort:ArithCheck failed", tag + ": rejected by the next Open (" + h.verdict + ")");
            check(h.x_hat == c.x_hat, tag + ": until the check both parties hold the same (shifted) masked input");
            check(h.sizes == check_sizes(n + s), tag + ": the honest party sends the four messages of the check only");
            check(h.evaluations == 0, tag + ": no output mask is generated after a wrong deferred opening");
            check(h.retired, tag + ": the session is retired");
        }
    }
}

// ------------------------------------------------------------ aborts of the online layer
static void test_abort(circuit::CircuitStore& store) {
    Section sec("abort");
    const unsigned n = 8, k = 64, s = 64;
    CtrPrgRng rng(91);
    std::vector<uint64_t> T(256);
    for (uint64_t& t : T) t = rng.u64();
    const GateSpec tab_w = GateSpec::table_gate(n, k, s, T, true), tab = GateSpec::table_gate(n, k, s, T, false);
    const uint64_t alpha_sh[2] = {3, 9};

    // mode 0: wrong value share in the masked wire opening (deferred): caught at Open, before the release.
    // mode 1: wrong value share in the release of the result.
    // mode 2: as mode 1, but the release is run on the checker directly, not through the driver: the
    //         abort hook of the checker retires the session and the registry all the same (change G).
    // In every mode the whole session is retired AT ONCE, i.e. before the next command looks at it.
    for (int mode = 0; mode < 3; mode++) {
        for (int cheater = 0; cheater < 2; cheater++) {
            prep::IdealDealer dealer;
            const bytes sid = {'a', 'b', (uint8_t)mode, (uint8_t)cheater};
            auto party = [&](int role) {
                return [&, role](Channel& ch) {
                    Session ses(role, ch, sid, store, dealer, k, s);
                    Party p(ses);
                    if (role == cheater)
                        p.set_checker_factory([&](int r, Channel& c, const bytes& sd, uint64_t d, unsigned ss, abb::PhaseBook*) {
                            // Openings: index 0 is the masked wire, index 1 the release.
                            return std::unique_ptr<spdz2k::ArithChecker>(
                                new CheatChecker(r, c, sd, d, ss, CheatChecker::VALUE, u128(1), mode == 0 ? 0 : 1));
                        });
                    p.setup(p.key_budget(protocol::FAMILY_DPF, 4) + p.gate_budget(tab_w) + p.gate_budget(tab));
                    const uint64_t kid = p.key(Party::NEXT, protocol::FAMILY_DPF, 4, alpha_sh[role]);
                    const AShare one = p.query(kid, 3 ^ 9);
                    const uint64_t gid = p.gate(Party::NEXT, tab_w), spare = p.gate(Party::NEXT, tab);
                    const uint64_t x_hat = p.mask_wire(gid, p.lin({{u128(40), one}}));
                    const AShare res = p.apply(gid)[0];
                    std::string verdict;
                    try {
                        if (mode == 2) p.checker()->open_final(res);  // the same messages as p.open(res)
                        else p.open(res);
                        verdict = "opened";
                    } catch (const ProtocolAbort& e) {
                        verdict = std::string("abort:") + e.what();
                    } catch (const std::exception& e) {
                        verdict = std::string("error:") + e.what();
                    }
                    // Nothing below may be what retires the session: it must be dead already.
                    const bool at_once = !p.active() && !ses.active() && ses.engine().dead() && ses.prep().aborted() &&
                                         !p.registry().active() && !ses.has_delta_share() && p.outmasks_available() == 0;
                    // After an abort every command throws ProtocolAbort and nothing is sent.
                    const uint64_t sent = ch.stats().msgs_sent;
                    int dead = 0;
                    dead += throws<ProtocolAbort>([&] { p.open(res); });
                    dead += throws<ProtocolAbort>([&] { p.mask_shared(spare, 1); });
                    dead += throws<ProtocolAbort>([&] { p.gate(Party::NEXT, tab); });
                    dead += throws<ProtocolAbort>([&] { p.query(kid, 0); });
                    dead += throws<ProtocolAbort>([&] { p.apply(gid); });
                    dead += throws<ProtocolAbort>([&] { p.key(Party::NEXT, protocol::FAMILY_DPF, 4, 1); });
                    dead += throws<ProtocolAbort>([&] { ses.evaluate(*store.get("setup_s64"), abb::Inputs()); });
                    const bool retired = !p.active() && !ses.active() && ses.engine().dead() && !ses.has_delta_share() &&
                                         p.checker()->dead() && !p.registry().active() && ch.stats().msgs_sent == sent;
                    (void)x_hat;
                    return verdict + "|" + std::to_string(dead) + "|" + (retired ? "retired" : "ALIVE") + (at_once ? "|at once" : "|LATE");
                };
            };
            const auto pr = run_two_party(party(0), party(1));
            const auto& h = cheater == 0 ? pr.second : pr.first;
            check(h.ok && h.value == "abort:ArithCheck failed|7|retired|at once",
                  std::string(mode == 0 ? "wrong masked-wire opening" : mode == 1 ? "wrong release" : "wrong release on the checker itself") +
                      ": the honest party aborts and the session is retired (" + (h.ok ? h.value : h.error) + ")");
        }
    }

    // A tampered generation (one d share flipped in an AND layer): nothing is certified, the session is dead.
    for (int cheater = 0; cheater < 2; cheater++) {
        prep::IdealDealer dealer;
        const bytes sid = {'a', 'g', (uint8_t)cheater};
        auto party = [&](int role) {
            return [&, role](Channel& raw) {
                // Messages of this party: 67 for the session setup (input, 64 AND layers, two for the
                // BoolCheck; nothing is opened), then the generation: input, then one per AND layer.
                std::map<size_t, TamperChannel::Rule> rules;
                if (role == cheater) rules[67 + 1 + 20] = flip_bit_rule(0);  // the first d share of AND layer 20
                TamperChannel ch(raw, rules);
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses);
                p.setup(p.key_budget(protocol::FAMILY_DPF, 4));
                std::string verdict;
                try {
                    p.key(Party::NEXT, protocol::FAMILY_DPF, 4, alpha_sh[role]);
                    verdict = "key";
                } catch (const ProtocolAbort& e) {
                    verdict = std::string("abort:") + e.what();
                } catch (const std::exception& e) {
                    verdict = std::string("error:") + e.what();
                }
                const state::Registry::Summary sm = p.registry().summary();
                const bool nothing = sm.keys_certified == 0 && sm.keys_generating == 1 && !p.active() &&
                                     throws<ProtocolAbort>([&] { p.query(0, 0); }) && throws<ProtocolAbort>([&] { p.gate(Party::NEXT, tab); });
                return verdict + (nothing ? "|nothing certified" : "|CERTIFIED");
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        const auto& h = cheater == 0 ? pr.second : pr.first;
        check(h.ok && h.value == "abort:BoolCheck failed|nothing certified",
              "tampered generation: the key is not certified and the session is retired (" + (h.ok ? h.value : h.error) + ")");
    }

    // Session::guard: a ProtocolAbort raised by a layer above the engine retires the session at
    // once; another exception does not, as long as the engine and the preprocessing are alive.
    {
        prep::IdealDealer dealer;
        const bytes sid = {'a', 'q'};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                int ok = 0;
                ok += ses.delta_wires().empty() && throws_invalid([&] { ses.delta_share(); });  // before setup: no key material yet
                ses.setup();
                ok += throws<std::runtime_error>([&] { ses.guard([&]() -> int { throw std::runtime_error("not an abort"); }); }) &&
                      ses.active() && ses.has_delta_share() && ses.delta_wires().width == s + 1;
                ok += throws<ProtocolAbort>([&] { ses.guard([&]() -> int { throw ProtocolAbort("raised above the engine"); }); });
                ok += !ses.active() && !ses.has_delta_share() && ses.engine().dead() && ses.prep().aborted();
                // Key material is not handed out by a session that is not ACTIVE.
                ok += throws<ProtocolAbort>([&] { ses.delta_wires(); }) && throws<ProtocolAbort>([&] { ses.delta_share(); });
                ok += throws<ProtocolAbort>([&] { ses.next_id("kid"); }) && throws<ProtocolAbort>([&] { ses.reserve_triples(1); });
                return ok;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok && pr.first.value == 6 && pr.second.value == 6,
              "Session::guard: a ProtocolAbort retires the session immediately, other exceptions do not; a retired session hands out no key material");
    }

    // delta_share() / delta_wires() check the engine and the preprocessing themselves: if the engine
    // died behind the session's back (it is reachable through session.engine()), the next request
    // for key material finds out, moves the session to ABORTED and gets nothing.
    for (int variant = 0; variant < 2; variant++) {
        prep::IdealDealer dealer;
        const bytes sid = {'a', 'k', (uint8_t)variant};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                ses.setup();
                int ok = 0;
                const uint64_t share = ses.delta_share();
                ok += ses.delta_wires().width == s + 1 && ses.has_delta_share();
                if (variant == 0) ses.engine().abort();  // the engine dies; the session has not looked yet
                else ses.prep().abort();                 // or the preprocessing
                ok += ses.active();
                if (variant == 0) ok += throws<ProtocolAbort>([&] { ses.delta_share(); });
                else ok += throws<ProtocolAbort>([&] { ses.delta_wires(); });
                ok += !ses.active() && !ses.has_delta_share() && ses.engine().dead();
                ok += throws<ProtocolAbort>([&] { ses.delta_wires(); }) && throws<ProtocolAbort>([&] { ses.delta_share(); });
                (void)share;
                return ok;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok && pr.first.value == 5 && pr.second.value == 5,
              std::string("delta_share / delta_wires refuse once the ") + (variant == 0 ? "engine" : "preprocessing") + " is dead (" +
                  pr.first.error + pr.second.error + ")");
    }

    // Change G outside the driver: a Session and an ArithChecker used directly.  bind_abort ties the
    // checker to the session, so a failed ArithCheck retires the Boolean engine, Delta_b, the session
    // wires and the preprocessing too.  bound = 0 is the control: without the hook only the checker dies.
    for (int bound = 0; bound < 2; bound++) {
        for (int cheater = 0; cheater < 2; cheater++) {
            prep::IdealDealer dealer;
            const bytes sid = {'a', 'h', (uint8_t)bound, (uint8_t)cheater};
            auto party = [&](int role) {
                return [&, role](Channel& ch) {
                    Session ses(role, ch, sid, store, dealer, k, s);
                    ses.setup();
                    const fss::DpfKey key = protocol::gen_dpf(ses, 4, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {alpha_sh[role]})[0];
                    const fss::Share sh = fss::dpf_eval(key, 3 ^ 9);
                    const AShare one(sh.y, sh.tau, ses.w(), true);  // exact [[1]] under the session key
                    CheatChecker chk(role, ch, ses.sid(), ses.delta_share(), s, role == cheater ? CheatChecker::VALUE : CheatChecker::HONEST,
                                     u128(1), 0);
                    if (bound) ses.bind_abort(chk);
                    std::string verdict;
                    try {
                        chk.open_final(one);
                        verdict = "opened";
                    } catch (const ProtocolAbort& e) {
                        verdict = std::string("abort:") + e.what();
                    } catch (const std::exception& e) {
                        verdict = std::string("error:") + e.what();
                    }
                    const bool session_dead = !ses.active() && ses.engine().dead() && ses.prep().aborted() && !ses.has_delta_share();
                    const bool refuses = !ses.active() && throws<ProtocolAbort>([&] { ses.delta_share(); }) &&
                                         throws<ProtocolAbort>([&] { ses.delta_wires(); });
                    return verdict + (chk.dead() ? "|checker dead" : "|checker alive") + (session_dead ? "|session dead" : "|session alive") +
                           (refuses ? "|no key material" : "|key material");
                };
            };
            const auto pr = run_two_party(party(0), party(1));
            const auto& h = cheater == 0 ? pr.second : pr.first;
            const char* want = bound ? "abort:ArithCheck failed|checker dead|session dead|no key material"
                                     : "abort:ArithCheck failed|checker dead|session alive|key material";
            check(h.ok && h.value == want,
                  std::string(bound ? "Session::bind_abort: a failed ArithCheck retires the session" : "(control) without the hook only the checker dies") +
                      " (" + (h.ok ? h.value : h.error) + ")");
        }
    }

    // A wrong share in a shared-input Mask is an input substitution: no abort, the gate computes on x + 1.
    {
        prep::IdealDealer dealer;
        const bytes sid = {'a', 's'};
        const uint64_t x = 200, x0 = 77, x1 = (x - x0) & 0xFF;
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                Session ses(role, ch, sid, store, dealer, k, s);
                Party p(ses);
                p.setup(p.gate_budget(tab));
                const uint64_t gid = p.gate(Party::NEXT, tab);
                p.mask_shared(gid, role == 0 ? x0 : ((x1 + 1) & 0xFF));   // P_1 substitutes its share (and commits to it)
                return p.open(p.apply(gid)[0]);
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok && pr.first.value == u128(T[x + 1]) && pr.second.value == u128(T[x + 1]),
              "a substituted shared input is not an abort: both parties obtain T[x + 1]");
    }
}

// ------------------------------------------------------------ DPF n = 16 with real preprocessing
static void test_dpf16_real(circuit::CircuitStore& store) {
    Section sec("DPF n = 16, public payload, real preprocessing");
    const unsigned n = 16, k = 64, s = 64;
    CtrPrgRng rng(99);
    const uint64_t a0 = rng.u64() & 0xFFFF, a1 = rng.u64() & 0xFFFF;
    const std::string name = protocol::key_circuit_name(protocol::FAMILY_DPF, n, k, s, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE);
    const size_t n_and = store.get(name)->n_and();
    const bytes sid = {'d', '1', '6'};
    struct Out {
        fss::DpfKey key;
        uint64_t delta = 0, triple_requests = 0;
        unsigned bucket = 0;
        ChannelStats stats;
        prep::PrepStats prep;
        prep::PhaseStat book;
        abb::EvalStats gen;
        std::string phases;
    };
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            Session ses(role, ch, sid, store, k, s, 40);
            ses.setup(n_and);
            Out out;
            out.key = protocol::gen_dpf(ses, n, protocol::INDEX_XOR, protocol::BETA_PUBLIC_ONE, {role == 0 ? a0 : a1})[0];
            out.delta = ses.delta_share();
            out.triple_requests = ses.engine().triple_requests();
            out.bucket = ses.real_prep()->last_bucket_size();
            out.stats = ch.stats();
            out.prep = ses.real_prep()->stats();
            out.book = ses.book().total();
            out.gen = ses.last_gen.stats;
            out.phases = ses.book().json();
            return out;
        };
    };
    const auto pr = run_two_party(party(0), party(1));
    check(pr.first.ok && pr.second.ok, "no abort (" + pr.first.error + pr.second.error + ")");
    if (!pr.first.ok || !pr.second.ok) return;
    const u128 delta = fss::make_delta(pr.first.value.delta, pr.second.value.delta);
    verify_dpf(pr.first.value.key, pr.second.value.key, a0 ^ a1, 1, delta, n, k, s, "dpf n=16 real", 4);
    check(n_and == 128679 && pr.first.value.gen.n_and == 128679 && pr.first.value.gen.and_depth == 362, "the circuit of CHANGELOG PY-A item 10: 128,679 AND gates, depth 362");
    check(pr.first.value.triple_requests == 1 && pr.first.value.bucket == 4, "one bucketing population of 128,743 triples, B = 4");
    check(pr.first.value.key.size_bytes() == 329, "key size 329 bytes");
    // 461 rounds: base OT 2, OT extension 3 + 6 + 6, leaky AND 1, bucketing 3, triple check 5, setup circuit 67, generation 368.
    check(pr.first.value.stats.rounds == 461 && pr.second.value.stats.rounds == 461, "461 rounds, as the Python reference");
    for (const auto* o : {&pr.first, &pr.second}) {
        const prep::PhaseStat pt = o->value.prep.total();
        check(pt.bytes_sent + o->value.book.bytes_sent == o->value.stats.bytes_sent && pt.rounds + o->value.book.rounds == o->value.stats.rounds,
              "the phases sum to the channel totals");
    }
    // Bytes of the Python reference for this configuration (CHANGELOG PY-D item 12): 49,670,502 in both directions.
    check(pr.first.value.stats.bytes_sent + pr.second.value.stats.bytes_sent == 49670502ull, "49,670,502 bytes in both directions, as the Python reference");
    std::printf("  DPF n=16: %llu bytes (both directions), %llu rounds, %.2f s; online %.3f s\n",
                (unsigned long long)(pr.first.value.stats.bytes_sent + pr.second.value.stats.bytes_sent),
                (unsigned long long)pr.first.value.stats.rounds, pr.first.seconds, pr.first.value.gen.online_s);
}

// ------------------------------------------------------------ cross-language run of the online layer
// One party of a fixed script of Pi_AuthEval over TCP on 127.0.0.1 with the real preprocessing
// (party 0 listens).  The peer is tests/interop_peer.py --online, which runs the same script on
// the Python reference: Setup; a DPF key (public payload) queried at its index, which gives
// [[1]]; a table gate with a shared input (Mask, Apply); a table gate with an authenticated
// wire [[x]] = x [[1]] (Mask with a deferred check, Apply); Open of both results; Open of
// 3 T[x] + 5, which is not exact and takes an output mask from Pi_Gen.  All inputs are fixed
// public constants of the test.  Both programs run their driver in strict mode and draw all
// triples of the session in ONE request (CHANGELOG F1 W2): setup(reserve_triples = N,
// reserve_outmasks = 1) with N = the DPF key plus the two table gates; the script succeeds only
// if the session ends as a single, fully used bucketing population.
static int run_interop(circuit::CircuitStore& store, int role, int port) {
    const unsigned n = 8, k = 64, s = 64;
    std::vector<uint64_t> T(256);
    for (uint64_t x = 0; x < 256; x++) T[(size_t)x] = 0x9E3779B97F4A7C15ull * (x + 1) + 7 * x * x;  // modulo 2^64
    const uint64_t x = 177, x_share = role == 0 ? 77 : 100, alpha_share = role == 0 ? 3 : 9;
    const std::unique_ptr<TcpChannel> ch = TcpChannel::create(role, "127.0.0.1", (uint16_t)port, 120000);
    const std::string sid_text = "interop-online";
    const bytes sid(sid_text.begin(), sid_text.end());
    Session ses(role, *ch, sid, store, k, s, 40);
    Party p(ses);  // strict mode
    const GateSpec tab_shared = GateSpec::table_gate(n, k, s, T, false), tab_wire = GateSpec::table_gate(n, k, s, T, true);
    const size_t reserve = p.key_budget(protocol::FAMILY_DPF, 4) + p.gate_budget(tab_shared) + p.gate_budget(tab_wire);
    p.setup(reserve, 1);  // ONE bucketing population: setup circuit, the key, the two gates, one output mask
    const uint64_t kid = p.key(Party::NEXT, protocol::FAMILY_DPF, 4, alpha_share);
    const AShare one = p.query(kid, 3 ^ 9);
    const uint64_t g0 = p.gate(Party::NEXT, tab_shared);
    const uint64_t hat0 = p.mask_shared(g0, x_share);
    const AShare y0 = p.apply(g0)[0];
    const uint64_t g1 = p.gate(Party::NEXT, tab_wire);
    const uint64_t hat1 = p.mask_wire(g1, p.lin({{u128(x), one}}));
    const AShare y1 = p.apply(g1)[0];
    const std::vector<u128> both = p.open_many({y0, y1});
    const u128 z = p.open(p.lin({{u128(3), y0}}, u128(5)));
    const bool one_population = p.triple_requests() == 1 && ses.triples_unused() == 0 &&
                                ses.triples_requested() == 64 + reserve + p.outmask_budget(1);
    const bool ok = both[0] == u128(T[x]) && both[1] == u128(T[x]) && z == u128(3 * T[x] + 5) && p.active() && one_population;
    const ChannelStats st = ch->stats();
    std::printf("{\"peer\": \"c++\", \"role\": %d, \"x_hat\": [%llu, %llu], \"opened\": [%llu, %llu, %llu], "
                "\"expected\": [%llu, %llu, %llu], \"arith_checks\": %llu, \"bytes_sent\": %llu, \"rounds\": %llu, "
                "\"triple_requests\": %llu, \"triples_requested\": %llu, \"triples_unused\": %llu, \"outputs_correct\": %s}\n",
                role, (unsigned long long)hat0, (unsigned long long)hat1, (unsigned long long)u128_lo(both[0]),
                (unsigned long long)u128_lo(both[1]), (unsigned long long)u128_lo(z), (unsigned long long)T[x], (unsigned long long)T[x],
                (unsigned long long)(3 * T[x] + 5), (unsigned long long)p.checker()->checks(), (unsigned long long)st.bytes_sent,
                (unsigned long long)st.rounds, (unsigned long long)p.triple_requests(), (unsigned long long)ses.triples_requested(),
                (unsigned long long)ses.triples_unused(), ok ? "true" : "false");
    return ok ? 0 : 4;
}

int main(int argc, char** argv) {
    const std::string exe_dir = circuit::dirname(argv[0]);
    std::string circuits = exe_dir + "/../../circuits";
    g_tests = exe_dir + "/../tests";
    int interop_role = -1, interop_port = 0;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--quick") g_quick = true;
        else if (arg == "--circuits" && i + 1 < argc) circuits = argv[++i];
        else if (arg == "--tests" && i + 1 < argc) g_tests = argv[++i];
        else if (arg == "--interop" && i + 2 < argc) {
            interop_role = std::atoi(argv[++i]);
            interop_port = std::atoi(argv[++i]);
        } else {
            std::printf("usage: test_protocol [--circuits DIR] [--tests DIR] [--quick]\n"
                        "       test_protocol [--circuits DIR] --interop ROLE PORT\n");
            return 2;
        }
    }
    if (interop_role == 0 || interop_role == 1) {
        try {
            circuit::CircuitStore store(circuits);
            return run_interop(store, interop_role, interop_port);
        } catch (const std::exception& e) {
            std::printf("interop: FAILED: %s\n", e.what());
            return 1;
        }
    }
    try {
        circuit::CircuitStore store(circuits);
        test_vectors(store);
        test_spdz2k();
        test_gates_local();
        test_state();
        test_gen_ideal(store);
        test_gen_real(store);
        test_budget(store);
        test_mask_shared(store);
        test_gates_e2e(store);
        test_open_order(store);
        test_abort(store);
        if (!g_quick) test_dpf16_real(store);
    } catch (const std::exception& e) {
        std::printf("FATAL: %s\n", e.what());
        return 1;
    }
    std::printf("test_protocol: %lld checks, %lld passed, %lld failed\n", g_pass + g_fail, g_pass, g_fail);
    return g_fail ? 1 : 0;
}
