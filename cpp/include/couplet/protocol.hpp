// couplet/protocol.hpp -- Pi_Gen: dealer-free generation of FSS keys and gate
// material (DESIGN.md sections 4, 5.2); the C++ counterpart of
// python/couplet/protocol.py.
//
// Each function runs ONE party on its Session; the other party calls the
// same function with the same public parameters and its own shares.  The
// protocol is exactly "inputs via Input, evaluate C_gen, Output" (Theorem 1):
//
//   1. the generation circuit of the public parameters is taken from the
//      session's circuit::CircuitStore under the name that
//      python/couplet/circuits_gen.descriptor_name gives it (the circuits
//      are compiled by the Python package and exported with
//      python/bench/export_circuits.py; the engine compares the circuit
//      digest with the peer in its first exchange);
//   2. the party supplies its inputs: its shares of alpha / beta (keys) and
//      fresh uniform coins for every other input declaration it owns (root
//      coins rho0 / rho1, the coins of r', the output-mask coins and, for
//      P_0, the share_out values sigma / mu_sigma, which ARE its shares).
//      Coins come from the operating system unless a deterministic source is
//      passed for tests;
//   3. the circuit is evaluated on the session engine, the session wires of
//      Delta being bound to the circuit's BOUND input;
//   4. the party assembles its outputs: fss::DpfKey / fss::DcfKey objects
//      (own root from the private output, public correction words) and, for
//      gates, a GateMaterial with this party's SPDZ2k shares.
//
// Functions (the batch dimension is the SIMD dimension of the engine:
// independent instances of one circuit that share the rounds):
//
//   gen_dpf(session, n, index_mode, beta_mode, alpha_shares, beta_shares)
//       -> one fss::DpfKey per entry of alpha_shares
//   gen_dcf(...)                         -> one fss::DcfKey per entry
//   gen_gate(session, descriptor, batch) -> `batch` GateMaterial
//
// alpha_shares / beta_shares: XOR or additive shares of the index
// (INDEX_XOR / INDEX_ADD), additive shares mod 2^k of the payload
// (BETA_SECRET; BETA_PUBLIC_ONE fixes the payload to 1 and takes no shares).
// The k and s of the session are used for every key.
//
// Not provided natively: gen_batch of the reference (several DIFFERENT
// circuits concatenated into one); it needs a circuit compiled for the
// particular list of items.
//
// Triple budget (design/CHANGELOG.md F1 item 1 (d), F2 item 2).  A generation
// consumes n_and(circuit) * batch AND triples, a public number.  The helpers
// and_count, outmask_descriptor / outmask_and_count, planned_key,
// planned_gate, planned_outmasks, and_budget and session_triples compute it
// from the circuits of the store, so that a session can draw all its triples
// in ONE preprocessing request (Session::setup(reserve_triples,
// reserve_outmasks)) and stay a single bucketing population.
//
// Label scheme of the circuits (circuits_gen.py; b is the owner):
//   keys    inputs (b, "alpha") n, (b, "beta") k [secret], (b, "rho0"),
//           (b, "rho1") 128; outputs "cws" 128 n (level i at bits
//           [128 (i-1), 128 i)), "cwtl", "cwtr" n (level i at bit i-1),
//           "vcw", "tcw" w n [DCF], "cwout_v", "cwout_t" w, "root0" -> P_0,
//           "root1" -> P_1;
//   gates   "rp.coin" (both), "rp.sigma", "rp.mu_sigma" (P_0), "rp.share1",
//           "rp.mu_share1" (-> P_1); "dpf.<key labels>"; per threshold i:
//           "th<i>.<key labels>", "th<i>.w.*", "th<i>.rw.*"; per output mask
//           j: "om<j>.coin", "om<j>.sigma", "om<j>.mu_sigma", "om<j>.share1",
//           "om<j>.mu_share1" (s-bit shares of the HIGH part; the SPDZ2k
//           shares over Z_{2^w} are 2^k times these values).
//
// GateMaterial encoding (identical to the reference; all integers
// little-endian, W = ceil((k+s)/8), M = ceil(m/8) with m = n + s for an
// authenticated input and n otherwise):
//     8B magic "CPLTGAT1" | u8 role | u32 length of the descriptor |
//     descriptor (canonical JSON of circuits_gen.gate_descriptor) |
//     M bytes share of r' (or of r) | [M bytes tag share, auth_input only] |
//     [DPF key in the fss encoding, dpf_one only] |
//     per threshold: DCF key | W bytes [[w_c]] value share | W bytes tag share |
//                    [W bytes [[r w_c]] value share | W bytes tag share, payload r only] |
//     per output mask: W bytes value share | W bytes tag share
//
// Any ProtocolAbort moves the session to ABORTED and nothing is returned.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "abb.hpp"
#include "circuit.hpp"
#include "fss.hpp"
#include "rng.hpp"
#include "session.hpp"
#include "u128.hpp"

namespace couplet {
namespace protocol {

using circuit::BitMat;

enum Family { FAMILY_DPF = 1, FAMILY_DCF = 2 };
enum IndexMode { INDEX_XOR, INDEX_ADD };
enum BetaMode { BETA_PUBLIC_ONE, BETA_SECRET };

const unsigned SEED_BITS = 128;

inline const char* family_name(Family f) { return f == FAMILY_DPF ? "dpf" : "dcf"; }

// Name of the key-generation circuit (circuits_gen.descriptor_name, ripple adders).
inline std::string key_circuit_name(Family family, unsigned n, unsigned k, unsigned s, IndexMode index_mode,
                                    BetaMode beta_mode) {
    return std::string(family_name(family)) + "_n" + std::to_string(n) + "_k" + std::to_string(k) + "_s" +
           std::to_string(s) + (index_mode == INDEX_XOR ? "_xor" : "_add") +
           (beta_mode == BETA_SECRET ? "_secret" : "_one");
}

// ------------------------------------------------------------------ gate descriptors
// One threshold (c, payload) of DESIGN.md 4.3: a DCF at (r + c) mod 2^n with
// payload 1 (payload_r = false) or payload r.
struct Threshold {
    u128 c = u128(0);  // 0 <= c <= 2^n
    bool payload_r = false;
    Threshold() {}
    Threshold(u128 c_, bool r_) : c(c_), payload_r(r_) {}
    friend bool operator==(const Threshold& a, const Threshold& b) { return a.c == b.c && a.payload_r == b.payload_r; }
};

// The descriptor of C_gate (circuits_gen.gate_descriptor).
struct GateDescriptor {
    unsigned n = 0, k = 0, s = 0;
    bool auth_input = false;
    bool dpf_one = false;
    std::vector<Threshold> thresholds;
    unsigned n_outmask = 0;
    std::string adder = "ripple";

    GateDescriptor() {}
    GateDescriptor(unsigned n_, unsigned k_, unsigned s_, bool auth, bool dpf, std::vector<Threshold> ths,
                   unsigned outmasks)
        : n(n_), k(k_), s(s_), auth_input(auth), dpf_one(dpf), thresholds(std::move(ths)), n_outmask(outmasks) {
        validate();
    }

    unsigned width() const { return k + s; }
    unsigned rp_bits() const { return auth_input ? n + s : n; }

    void validate() const {
        if (n < 1 || n > 64) throw std::invalid_argument("n must be in 1..64");
        if (k < 1 || s < 1 || 2 * ((uint64_t)k + s) > 320) throw std::invalid_argument("need k >= 1, s >= 1 and 2(k+s) <= 320");
        if (k > 64 || s > 64) throw std::invalid_argument("this implementation supports k <= 64 and s <= 64");
        if (n > k) throw std::invalid_argument("gates need n <= k");
        if (adder != "ripple" && adder != "sklansky") throw std::invalid_argument("adder must be ripple or sklansky");
        const u128 big = u128(1) << n;  // n <= 64, so 2^n fits
        for (const Threshold& t : thresholds)
            if (t.c > big) throw std::invalid_argument("threshold constants must lie in [0, 2^n]");
    }

    // circuits_gen.descriptor_name.
    std::string name() const {
        std::string out = "gate_n" + std::to_string(n) + "_k" + std::to_string(k) + "_s" + std::to_string(s) +
                          (auth_input ? "_auth" : "_shared");
        if (dpf_one) out += "_dpf";
        for (const Threshold& t : thresholds) out += "_t" + json::dec(t.c) + (t.payload_r ? "r" : "o");
        if (n_outmask) out += "_om" + std::to_string(n_outmask);
        if (adder != "ripple") out += "_" + adder;
        return out;
    }

    // Canonical JSON: json.dumps(descriptor, sort_keys=True, separators=(",", ":")).
    std::string to_json() const {
        std::string out = "{\"adder\":" + json::quote(adder) + ",\"auth_input\":" + (auth_input ? "true" : "false") +
                          ",\"dpf_one\":" + (dpf_one ? "true" : "false") + ",\"k\":" + std::to_string(k) +
                          ",\"kind\":\"gate\",\"n\":" + std::to_string(n) + ",\"n_outmask\":" +
                          std::to_string(n_outmask) + ",\"s\":" + std::to_string(s) + ",\"thresholds\":[";
        for (size_t i = 0; i < thresholds.size(); i++) {
            if (i) out += ",";
            out += "[" + json::dec(thresholds[i].c) + "," + (thresholds[i].payload_r ? "\"r\"" : "\"one\"") + "]";
        }
        return out + "]}";
    }

    static GateDescriptor from_json(const std::string& text) {
        const json::Value v = json::parse(text);
        if (!v.is_obj() || v.at("kind").str() != "gate") throw std::invalid_argument("the descriptor is not a gate descriptor");
        GateDescriptor d;
        d.n = small(v.at("n"));
        d.k = small(v.at("k"));
        d.s = small(v.at("s"));
        d.auth_input = v.at("auth_input").as_bool();
        d.dpf_one = v.at("dpf_one").as_bool();
        d.n_outmask = small(v.at("n_outmask"));
        if (const json::Value* ad = v.find("adder")) d.adder = ad->str();
        const json::Value& ths = v.at("thresholds");
        if (!ths.is_arr()) throw std::invalid_argument("thresholds must be a list");
        for (const json::Value& t : ths.items) {
            if (!t.is_arr() || t.items.size() != 2) throw std::invalid_argument("a threshold is a pair (c, payload)");
            const std::string& payload = t.items[1].str();
            if (payload != "one" && payload != "r") throw std::invalid_argument("payload must be one or r");
            d.thresholds.push_back(Threshold(t.items[0].as_u128(), payload == "r"));
        }
        d.validate();
        return d;
    }

    friend bool operator==(const GateDescriptor& a, const GateDescriptor& b) {
        return a.n == b.n && a.k == b.k && a.s == b.s && a.auth_input == b.auth_input && a.dpf_one == b.dpf_one &&
               a.thresholds == b.thresholds && a.n_outmask == b.n_outmask && a.adder == b.adder;
    }
    friend bool operator!=(const GateDescriptor& a, const GateDescriptor& b) { return !(a == b); }

private:
    static unsigned small(const json::Value& v) {
        const uint64_t x = v.as_u64();
        if (x > 1000000) throw std::invalid_argument("descriptor field out of range");
        return (unsigned)x;
    }
};

// ------------------------------------------------------------------ gate material
// A (value, tag) pair of additive shares.
struct SharePair {
    u128 value = u128(0), tag = u128(0);
    SharePair() {}
    SharePair(u128 v, u128 t) : value(v), tag(t) {}
    friend bool operator==(const SharePair& a, const SharePair& b) { return a.value == b.value && a.tag == b.tag; }
};

// One party's output of Gate.Gen (DESIGN.md 4.3) for one gate instance.
//   rp_share        share of r' mod 2^(n+s) (auth_input) or of r mod 2^n;
//   rp_tag_share    share of Delta r' mod 2^(n+s); meaningful with auth_input only;
//   dpf_key         DPF key at index r with payload 1 (dpf_one only);
//   dcf_keys        one DCF key per threshold, at index (r + c) mod 2^n with payload 1 or r;
//   w_shares        per threshold the share of [[w_c]] over Z_{2^w}, w_c = [r + c >= 2^n];
//   rw_shares       per threshold the share of [[r w_c]] (payload r; zero otherwise);
//   outmask_shares  per output mask the share of [[2^k rho]] over Z_{2^w}; both
//                   components are multiples of 2^k.
// All sharings are additive: the two parties' values sum to the secret modulo
// the stated power of two, tags with Delta = Delta_0 + Delta_1.
struct GateMaterial {
    int role = 0;
    GateDescriptor desc;
    u128 rp_share = u128(0);
    u128 rp_tag_share = u128(0);
    fss::DpfKey dpf_key;
    std::vector<fss::DcfKey> dcf_keys;
    std::vector<SharePair> w_shares;
    std::vector<SharePair> rw_shares;
    std::vector<SharePair> outmask_shares;

    unsigned n() const { return desc.n; }
    unsigned k() const { return desc.k; }
    unsigned s() const { return desc.s; }
    unsigned width() const { return desc.k + desc.s; }
    unsigned rp_bits() const { return desc.rp_bits(); }
    bool auth_input() const { return desc.auth_input; }

    std::vector<uint8_t> to_bytes() const {
        static const char magic[9] = "CPLTGAT1";
        const std::string d = desc.to_json();
        const unsigned wb = (width() + 7) / 8, mb = (rp_bits() + 7) / 8;
        std::vector<uint8_t> out(magic, magic + 8);
        out.push_back((uint8_t)role);
        for (int i = 0; i < 4; i++) out.push_back((uint8_t)((uint32_t)d.size() >> (8 * i)));
        out.insert(out.end(), d.begin(), d.end());
        put(out, rp_share, mb);
        if (desc.auth_input) put(out, rp_tag_share, mb);
        if (desc.dpf_one) append(out, dpf_key.to_bytes());
        for (size_t i = 0; i < desc.thresholds.size(); i++) {
            append(out, dcf_keys.at(i).to_bytes());
            put(out, w_shares.at(i).value, wb);
            put(out, w_shares.at(i).tag, wb);
            if (desc.thresholds[i].payload_r) {
                put(out, rw_shares.at(i).value, wb);
                put(out, rw_shares.at(i).tag, wb);
            }
        }
        for (const SharePair& m : outmask_shares) {
            put(out, m.value, wb);
            put(out, m.tag, wb);
        }
        return out;
    }
    size_t size_bytes() const { return to_bytes().size(); }

    static GateMaterial from_bytes(const uint8_t* data, size_t len) {
        static const char magic[9] = "CPLTGAT1";
        if (len < 13 || std::memcmp(data, magic, 8) != 0) throw std::invalid_argument("not a gate material encoding");
        GateMaterial g;
        g.role = data[8];
        if (g.role > 1) throw std::invalid_argument("party index must be 0 or 1");
        const size_t dlen = (size_t)data[9] | ((size_t)data[10] << 8) | ((size_t)data[11] << 16) | ((size_t)data[12] << 24);
        size_t pos = 13;
        if (len - pos < dlen) throw std::invalid_argument("truncated gate material");
        g.desc = GateDescriptor::from_json(std::string((const char*)data + pos, dlen));
        pos += dlen;
        const unsigned width = g.width(), wb = (width + 7) / 8, m = g.rp_bits(), mb = (m + 7) / 8;
        g.rp_share = take(data, len, pos, mb, m);
        if (g.desc.auth_input) g.rp_tag_share = take(data, len, pos, mb, m);
        if (g.desc.dpf_one) {
            const std::pair<const uint8_t*, size_t> kb = take_key(data, len, pos);
            g.dpf_key = fss::DpfKey::from_bytes(kb.first, kb.second);
            if (g.dpf_key.b != g.role || g.dpf_key.n != g.desc.n || g.dpf_key.k != g.desc.k || g.dpf_key.s != g.desc.s)
                throw std::invalid_argument("embedded key does not match the descriptor");
        }
        for (const Threshold& t : g.desc.thresholds) {
            const std::pair<const uint8_t*, size_t> kb = take_key(data, len, pos);
            fss::DcfKey key = fss::DcfKey::from_bytes(kb.first, kb.second);
            if (key.b != g.role || key.n != g.desc.n || key.k != g.desc.k || key.s != g.desc.s)
                throw std::invalid_argument("embedded key does not match the descriptor");
            g.dcf_keys.push_back(std::move(key));
            SharePair w;
            w.value = take(data, len, pos, wb, width);
            w.tag = take(data, len, pos, wb, width);
            g.w_shares.push_back(w);
            SharePair rw;
            if (t.payload_r) {
                rw.value = take(data, len, pos, wb, width);
                rw.tag = take(data, len, pos, wb, width);
            }
            g.rw_shares.push_back(rw);
        }
        const u128 low = u128_mask(g.desc.k);
        for (unsigned i = 0; i < g.desc.n_outmask; i++) {
            SharePair mk;
            mk.value = take(data, len, pos, wb, width);
            mk.tag = take(data, len, pos, wb, width);
            if ((mk.value & low) != u128(0) || (mk.tag & low) != u128(0))
                throw std::invalid_argument("output mask shares must be multiples of 2^k");
            g.outmask_shares.push_back(mk);
        }
        if (pos != len) throw std::invalid_argument("trailing bytes");
        return g;
    }
    static GateMaterial from_bytes(const std::vector<uint8_t>& data) { return from_bytes(data.data(), data.size()); }

    friend bool operator==(const GateMaterial& a, const GateMaterial& b) {
        return a.role == b.role && a.desc == b.desc && a.rp_share == b.rp_share &&
               (!a.desc.auth_input || a.rp_tag_share == b.rp_tag_share) && (!a.desc.dpf_one || a.dpf_key == b.dpf_key) &&
               a.dcf_keys == b.dcf_keys && a.w_shares == b.w_shares && a.rw_shares == b.rw_shares &&
               a.outmask_shares == b.outmask_shares;
    }

private:
    static void put(std::vector<uint8_t>& out, u128 v, unsigned nbytes) {
        uint8_t buf[16];
        u128_store_le(buf, v);
        out.insert(out.end(), buf, buf + nbytes);
    }
    static void append(std::vector<uint8_t>& out, const std::vector<uint8_t>& more) {
        out.insert(out.end(), more.begin(), more.end());
    }
    static u128 take(const uint8_t* data, size_t len, size_t& pos, unsigned nbytes, unsigned bits) {
        if (len - pos < nbytes) throw std::invalid_argument("truncated gate material");
        uint8_t buf[16] = {0};
        std::memcpy(buf, data + pos, nbytes);
        pos += nbytes;
        const u128 v = u128_load_le(buf);
        if (bits < 128 && (v >> bits) != u128(0)) throw std::invalid_argument("share out of range");
        return v;
    }
    static std::pair<const uint8_t*, size_t> take_key(const uint8_t* data, size_t len, size_t& pos) {
        if (len - pos < 4) throw std::invalid_argument("truncated gate material");
        const size_t body = (size_t)data[pos] | ((size_t)data[pos + 1] << 8) | ((size_t)data[pos + 2] << 16) |
                            ((size_t)data[pos + 3] << 24);
        if (len - pos - 4 < body) throw std::invalid_argument("truncated gate material");
        const std::pair<const uint8_t*, size_t> r(data + pos, 4 + body);
        pos += 4 + body;
        return r;
    }
};

// ------------------------------------------------------------------ AND budget of planned generation calls
// The counterparts of protocol.and_count, outmask_descriptor,
// outmask_and_count, planned_key, planned_gate, planned_outmasks, and_budget
// and session_triples of the reference.  Everything here is a function of
// public parameters and sends no message; both parties compute the same
// numbers.  The counts are read from the circuits themselves
// (Circuit::n_and() of the validated file whose digest the engine compares
// with the peer), which the store loads once and caches.

// AND gates of ONE instance of the circuit `name`.
inline size_t and_count(circuit::CircuitStore& store, const std::string& name) { return store.get(name)->n_and(); }
// AND gates of ONE instance of the generation circuit of a gate descriptor.
inline size_t and_count(circuit::CircuitStore& store, const GateDescriptor& descriptor) {
    descriptor.validate();
    return and_count(store, descriptor.name());
}

// Descriptor of the on-demand output-mask circuit: one mask [[2^k rho]] per
// instance.  A gate descriptor without keys and with a single output mask (it
// contains an unused 1-bit offset because every gate descriptor has one); the
// circuit that Party::reserve_outmasks and Open run on as many instances as
// masks are needed ("gate_n1_k<k>_s<s>_shared_om1").
inline GateDescriptor outmask_descriptor(unsigned k, unsigned s) {
    return GateDescriptor(1, k, s, false, false, std::vector<Threshold>(), 1);
}
// AND triples consumed by ONE on-demand output mask.
inline size_t outmask_and_count(circuit::CircuitStore& store, unsigned k, unsigned s) {
    return and_count(store, outmask_descriptor(k, s));
}

// One planned generation call and the triples it will consume.
struct Planned {
    std::string call;     // "gen_dpf", "gen_dcf", "gen_gate" or "outmasks"
    std::string circuit;  // name of the generation circuit
    size_t batch = 0;     // instances of the SIMD dimension (keys, gates or masks)
    size_t n_and = 0;     // AND gates of one instance
    size_t triples = 0;   // n_and * batch: every instance uses its own triples
};

namespace detail {
inline Planned planned(circuit::CircuitStore& store, const char* call, const std::string& circuit_name, size_t batch,
                       bool allow_zero) {
    if (batch < 1 && !allow_zero) throw std::invalid_argument("batch must be a positive integer");
    Planned p;
    p.call = call;
    p.circuit = circuit_name;
    p.batch = batch;
    p.n_and = and_count(store, circuit_name);
    if (p.n_and && batch > SIZE_MAX / p.n_and) throw std::invalid_argument("the triple budget does not fit in size_t");
    p.triples = p.n_and * batch;
    return p;
}
}  // namespace detail

// Planned gen_dpf / gen_dcf call for `batch` keys on a session with these k and s.
inline Planned planned_key(circuit::CircuitStore& store, Family family, unsigned n, unsigned k, unsigned s,
                           IndexMode index_mode = INDEX_XOR, BetaMode beta_mode = BETA_PUBLIC_ONE, size_t batch = 1) {
    if (family != FAMILY_DPF && family != FAMILY_DCF) throw std::invalid_argument("family must be dpf or dcf");
    return detail::planned(store, family == FAMILY_DPF ? "gen_dpf" : "gen_dcf",
                           key_circuit_name(family, n, k, s, index_mode, beta_mode), batch, false);
}
// Planned gen_gate call for `batch` gates of one descriptor.
inline Planned planned_gate(circuit::CircuitStore& store, const GateDescriptor& descriptor, size_t batch = 1) {
    descriptor.validate();
    return detail::planned(store, "gen_gate", descriptor.name(), batch, false);
}
// `count` on-demand output masks (in however many executions they are generated).
inline Planned planned_outmasks(circuit::CircuitStore& store, unsigned k, unsigned s, size_t count) {
    return detail::planned(store, "outmasks", outmask_descriptor(k, s).name(), count, true);
}

// AND triples consumed by the planned calls: the value of reserve_triples.
// The session-setup circuit is NOT included: Session::setup adds its own
// triples to the reservation.
inline size_t and_budget(const std::vector<Planned>& calls) {
    size_t total = 0;
    for (const Planned& p : calls) {
        if (p.triples > SIZE_MAX - total) throw std::invalid_argument("the triple budget does not fit in size_t");
        total += p.triples;
    }
    return total;
}

// Size of the ONE triple request of a session that runs `calls`: setup circuit plus budget.
inline size_t session_triples(circuit::CircuitStore& store, const std::vector<Planned>& calls, unsigned s) {
    const size_t setup = and_count(store, Session::setup_circuit_name(s)), budget = and_budget(calls);
    if (budget > SIZE_MAX - setup) throw std::invalid_argument("the triple budget does not fit in size_t");
    return setup + budget;
}

// ------------------------------------------------------------------ running a generation circuit
namespace detail {

struct RunResult {
    abb::Inputs own;       // this party's inputs (fixed and random), by label
    abb::EvalResult eval;  // opened outputs
};

// Evaluate a generation circuit: `fixed` holds the non-random inputs of this
// party; every other input declaration it owns is filled with fresh uniform bits.
inline RunResult run(Session& session, const circuit::Circuit& c, abb::Inputs fixed, size_t batch, RandomSource* rng) {
    SystemRng sys;
    RandomSource& r = rng ? *rng : static_cast<RandomSource&>(sys);
    RunResult out;
    size_t used = 0;
    for (const circuit::Decl& d : c.inputs) {
        if (d.owner != (uint32_t)session.role()) continue;
        abb::Inputs::iterator it = fixed.find(d.label);
        if (it != fixed.end()) {
            if (it->second.width != d.wires.size())
                throw std::invalid_argument("input " + d.label + " has the wrong width for this circuit");
            out.own.emplace(d.label, std::move(it->second));
            used++;
        } else {  // fresh uniform coins of this party
            BitMat m(d.wires.size(), batch);
            if (!m.v.empty()) r.fill_bits(m.v.data(), m.v.size());
            out.own.emplace(d.label, std::move(m));
        }
    }
    if (used != fixed.size()) throw std::invalid_argument("the circuit does not take the given inputs");
    out.eval = session.evaluate(c, out.own, batch);
    session.last_gen.circuit = c.name;
    session.last_gen.digest = c.digest_hex();
    session.last_gen.batch = batch;
    session.last_gen.stats = out.eval.stats;
    return out;
}

inline const BitMat& need(const abb::Outputs& m, const std::string& label, size_t width) {
    const abb::Outputs::const_iterator it = m.find(label);
    if (it == m.end()) throw std::runtime_error("the circuit has no output " + label);
    if (it->second.width != width) throw std::runtime_error("output " + label + " has an unexpected width");
    return it->second;
}

// Party `role`'s key of instance j from the opened outputs under `prefix`.
template <class Key>
inline void assemble_common(Key& key, int role, unsigned n, unsigned k, unsigned s, const abb::EvalResult& ev,
                            const std::string& prefix, size_t j) {
    const unsigned w = k + s;
    key.b = (uint8_t)role;
    key.n = (uint8_t)n;
    key.k = (uint8_t)k;
    key.s = (uint8_t)s;
    const std::vector<uint8_t> root = need(ev.priv, prefix + "root" + std::to_string(role), SEED_BITS).bytes(j);
    std::memcpy(key.seed.data(), root.data(), 16);
    const BitMat& cws = need(ev.pub, prefix + "cws", (size_t)SEED_BITS * n);
    const BitMat& cwtl = need(ev.pub, prefix + "cwtl", n);
    const BitMat& cwtr = need(ev.pub, prefix + "cwtr", n);
    key.cw_s.resize(n);
    key.cw_tl.resize(n);
    key.cw_tr.resize(n);
    for (unsigned i = 0; i < n; i++) {
        const std::vector<uint8_t> seed = cws.bytes(j, (size_t)SEED_BITS * i, SEED_BITS);
        std::memcpy(key.cw_s[i].data(), seed.data(), 16);
        key.cw_tl[i] = cwtl.at(i, j);
        key.cw_tr[i] = cwtr.at(i, j);
    }
    key.cw_out_v = need(ev.pub, prefix + "cwout_v", w).word(j);
    key.cw_out_t = need(ev.pub, prefix + "cwout_t", w).word(j);
}

inline fss::DpfKey assemble_dpf(int role, unsigned n, unsigned k, unsigned s, const abb::EvalResult& ev,
                                const std::string& prefix, size_t j) {
    fss::DpfKey key;
    assemble_common(key, role, n, k, s, ev, prefix, j);
    return key;
}

inline fss::DcfKey assemble_dcf(int role, unsigned n, unsigned k, unsigned s, const abb::EvalResult& ev,
                                const std::string& prefix, size_t j) {
    fss::DcfKey key;
    assemble_common(key, role, n, k, s, ev, prefix, j);
    const unsigned w = k + s;
    const BitMat& vcw = need(ev.pub, prefix + "vcw", (size_t)w * n);
    const BitMat& tcw = need(ev.pub, prefix + "tcw", (size_t)w * n);
    key.cw_v.resize(n);
    key.cw_t.resize(n);
    for (unsigned i = 0; i < n; i++) {
        key.cw_v[i] = vcw.word(j, (size_t)w * i, w);
        key.cw_t[i] = tcw.word(j, (size_t)w * i, w);
    }
    return key;
}

// This party's share of a share_out over `bits` bits: P_0 holds its inputs
// sigma (and mu_sigma), P_1 the private outputs share1 (and mu_share1).
inline SharePair share_out(int role, const RunResult& rr, const std::string& q, unsigned bits, bool tagged, size_t j) {
    SharePair p;
    if (role == 0) {
        p.value = need(rr.own, q + "sigma", bits).word(j);
        if (tagged) p.tag = need(rr.own, q + "mu_sigma", bits).word(j);
    } else {
        p.value = need(rr.eval.priv, q + "share1", bits).word(j);
        if (tagged) p.tag = need(rr.eval.priv, q + "mu_share1", bits).word(j);
    }
    return p;
}

inline GateMaterial assemble_gate(int role, const GateDescriptor& desc, const RunResult& rr, const std::string& prefix,
                                  size_t j) {
    GateMaterial g;
    g.role = role;
    g.desc = desc;
    const unsigned n = desc.n, k = desc.k, s = desc.s, w = k + s;
    const SharePair rp = share_out(role, rr, prefix + "rp.", desc.rp_bits(), desc.auth_input, j);
    g.rp_share = rp.value;
    g.rp_tag_share = rp.tag;
    if (desc.dpf_one) g.dpf_key = assemble_dpf(role, n, k, s, rr.eval, prefix + "dpf.", j);
    for (size_t i = 0; i < desc.thresholds.size(); i++) {
        const std::string q = prefix + "th" + std::to_string(i) + ".";
        g.dcf_keys.push_back(assemble_dcf(role, n, k, s, rr.eval, q, j));
        g.w_shares.push_back(share_out(role, rr, q + "w.", w, true, j));
        g.rw_shares.push_back(desc.thresholds[i].payload_r ? share_out(role, rr, q + "rw.", w, true, j) : SharePair());
    }
    for (unsigned i = 0; i < desc.n_outmask; i++) {
        const SharePair hi = share_out(role, rr, prefix + "om" + std::to_string(i) + ".", s, true, j);  // s-bit high parts
        g.outmask_shares.push_back(SharePair(hi.value << k, hi.tag << k));  // SPDZ2k shares of 2^k rho over Z_{2^w}
    }
    return g;
}

inline abb::Inputs key_inputs(unsigned n, unsigned k, BetaMode beta_mode, const std::vector<uint64_t>& alpha_shares,
                              const std::vector<uint64_t>& beta_shares) {
    const size_t batch = alpha_shares.size();
    if (batch < 1) throw std::invalid_argument("at least one alpha share is required");
    if (n < 1 || n > 64) throw std::invalid_argument("n must be in 1..64");
    for (uint64_t a : alpha_shares)
        if (n < 64 && (a >> n) != 0) throw std::invalid_argument("alpha_share must lie in [0, 2^n)");
    abb::Inputs fixed;
    fixed.emplace("alpha", BitMat::from_u64(alpha_shares, n));
    if (beta_mode == BETA_SECRET) {
        if (beta_shares.size() != batch) throw std::invalid_argument("beta_mode secret needs one beta share per key");
        for (uint64_t b : beta_shares)
            if (k < 64 && (b >> k) != 0) throw std::invalid_argument("beta_share must lie in [0, 2^k)");
        fixed.emplace("beta", BitMat::from_u64(beta_shares, k));
    } else if (!beta_shares.empty()) {
        throw std::invalid_argument("beta_mode public_one takes no beta shares");
    }
    return fixed;
}

}  // namespace detail

// ------------------------------------------------------------------ Pi_Gen
// This party's DPF keys for alpha_shares.size() independent point functions.
// The two parties' keys evaluate to additive shares of beta [x = alpha] and
// of Delta beta [x = alpha] in Z_{2^(k+s)} with k, s and Delta of the session.
inline std::vector<fss::DpfKey> gen_dpf(Session& session, unsigned n, IndexMode index_mode, BetaMode beta_mode,
                                        const std::vector<uint64_t>& alpha_shares,
                                        const std::vector<uint64_t>& beta_shares = std::vector<uint64_t>(),
                                        RandomSource* rng = nullptr) {
    session.require_active();
    abb::Inputs fixed = detail::key_inputs(n, session.k(), beta_mode, alpha_shares, beta_shares);
    const std::shared_ptr<const circuit::Circuit> c =
        session.store().get(key_circuit_name(FAMILY_DPF, n, session.k(), session.s(), index_mode, beta_mode));
    const size_t batch = alpha_shares.size();
    const detail::RunResult rr = detail::run(session, *c, std::move(fixed), batch, rng);
    std::vector<fss::DpfKey> keys;
    for (size_t j = 0; j < batch; j++)
        keys.push_back(detail::assemble_dpf(session.role(), n, session.k(), session.s(), rr.eval, "", j));
    return keys;
}

// This party's DCF keys (comparison beta [x < alpha]); as gen_dpf.
inline std::vector<fss::DcfKey> gen_dcf(Session& session, unsigned n, IndexMode index_mode, BetaMode beta_mode,
                                        const std::vector<uint64_t>& alpha_shares,
                                        const std::vector<uint64_t>& beta_shares = std::vector<uint64_t>(),
                                        RandomSource* rng = nullptr) {
    session.require_active();
    abb::Inputs fixed = detail::key_inputs(n, session.k(), beta_mode, alpha_shares, beta_shares);
    const std::shared_ptr<const circuit::Circuit> c =
        session.store().get(key_circuit_name(FAMILY_DCF, n, session.k(), session.s(), index_mode, beta_mode));
    const size_t batch = alpha_shares.size();
    const detail::RunResult rr = detail::run(session, *c, std::move(fixed), batch, rng);
    std::vector<fss::DcfKey> keys;
    for (size_t j = 0; j < batch; j++)
        keys.push_back(detail::assemble_dcf(session.role(), n, session.k(), session.s(), rr.eval, "", j));
    return keys;
}

// This party's material for `batch` independent gates of one descriptor.
// The gate takes no party input; a corrupted P_0 chooses its own share_out
// values, as in F_Gen.
inline std::vector<GateMaterial> gen_gate(Session& session, const GateDescriptor& descriptor, size_t batch = 1,
                                          RandomSource* rng = nullptr) {
    session.require_active();
    if (batch < 1) throw std::invalid_argument("batch must be a positive integer");
    descriptor.validate();
    if (descriptor.k != session.k() || descriptor.s != session.s())
        throw std::invalid_argument("the gate descriptor does not use the session's k and s");
    const std::shared_ptr<const circuit::Circuit> c = session.store().get(descriptor.name());
    const detail::RunResult rr = detail::run(session, *c, abb::Inputs(), batch, rng);
    std::vector<GateMaterial> out;
    for (size_t j = 0; j < batch; j++) out.push_back(detail::assemble_gate(session.role(), descriptor, rr, "", j));
    return out;
}

}  // namespace protocol
}  // namespace couplet
