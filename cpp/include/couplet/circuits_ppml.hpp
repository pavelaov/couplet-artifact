// couplet/circuits_ppml.hpp -- the generation circuits of the Couplet layer
// as seen by the native code (DESIGN_M3.md section 2): descriptors, this
// party's material, Pi_Gen for the Couplet items, the trusted-dealer
// reference and the manifest; the C++ counterpart of
// python/couplet/circuits_ppml.py (the circuits themselves are compiled by
// the reference and loaded from the store, see protocol.hpp).
//
// Descriptors and names (circuits_ppml.descriptor_name; ripple adders unless
// "_sklansky" is appended):
//     site(n, f, k, s, mode)       site_n<n>_f<f>_k<k>_s<s>_<round|exact>
//     triples(count, k, s)         triples<count>_k<k>_s<s>
//     mattriple(rows, cols, k, s)  mattriple_<rows>x<cols>_k<k>_s<s>
//     inmask(owner, count, k, s)   inmask_p<owner>_c<count>_k<k>_s<s>
//     outmask(k, s)                gate_n1_k<k>_s<s>_shared_om1 (the core's
//                                  output-mask circuit, same file and digest;
//                                  one mask [[2^k rho]] per instance)
// to_json() is the canonical JSON of the reference (sorted keys, no spaces),
// the form the manifest hashes.  Constraints of the reference: 1 <= n <= 20,
// 0 <= f <= 20, n <= k and f + n <= k for sites; plus k, s <= 64 here.
//
// Material of one instance at P_b (all sharings additive, tags with the
// unreduced Delta = Delta_0 + Delta_1):
//     SiteMaterial       rp: [[rho']] over Z_{2^m}, m = f + n + s; plane_a: DPF
//                        key (domain 2^n, payload 1) at rho_h, or at r' mod 2^n
//                        in mode exact with f >= 1, where also plane_b (domain
//                        2^f, at rho_l), rh = [[rho_h]] and r2 = [[r']] over
//                        Z_{2^(n+s)} are present;
//     TripleMaterial     a, b, c (c = a b) over Z_{2^w};
//     MatTripleMaterial  a (rows x cols, row-major), b (cols), c = a b (rows);
//     InMaskMaterial     shares of the masks; values (the masks) at the owner;
//     OutMaskMaterial    share of [[2^k rho]] over Z_{2^w} (both components
//                        multiples of 2^k; labels "om0." of the core gate).
// Labels and the output layout are those of the reference module
// docstring: offset "rp.", plane A "dpf.", plane B "pb.", "rh.", "r2.";
// vectors "a.", "b.", "c.", "m." (entry j at bits [j w, (j+1) w) of one
// label), the owner's masks "rho".  P_0's share_out inputs (sigma, mu_sigma)
// are its shares, P_1 receives share1 / mu_share1.
//
// generate(session, desc, batch) runs ONE circuit execution with `batch`
// instances (the batch dimension of the engine) exactly as the reference:
// fresh uniform coins for every input this party owns, evaluation on the
// session engine (the engine compares the circuit digest with the peer),
// assembly.  The deal_* functions are the trusted-dealer reference (NOT a
// protocol; tests only).
//
// Manifest (the commitment point of the plan): k, s, mode, private-model
// flag, model hash, the tables (name, n, f, hash) and the ordered executions
// (use, descriptor, batch).  Encoding (all integers little-endian):
//     "CPLTMAN1" | u32 k | u32 s | u8 mode (0 round, 1 exact) | u8 private |
//     32 B model hash | u32 n_tables | per table: u32 len, name, u32 n, u32 f,
//     32 B hash | u32 n_entries | per entry: u32 len, use, u32 len, canonical
//     JSON of the descriptor, u32 batch
// compare_manifest exchanges its SHA-256 in one round (phase "manifest")
// before anything else; differing digests abort (ProtocolAbort).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "abb.hpp"
#include "channel.hpp"
#include "circuit.hpp"
#include "fss.hpp"
#include "hashes.hpp"
#include "protocol.hpp"
#include "rng.hpp"
#include "session.hpp"
#include "u128.hpp"

namespace couplet {
namespace ppml {

using protocol::SharePair;

enum Kind { KIND_SITE = 0, KIND_TRIPLES = 1, KIND_MATTRIPLE = 2, KIND_INMASK = 3, KIND_OUTMASK = 4 };
const unsigned MAX_PLANE_BITS = 20;

inline const char* kind_name(Kind k) {
    static const char* names[5] = {"site", "triples", "mattriple", "inmask", "outmask"};
    return names[(int)k];
}

// ------------------------------------------------------------------ descriptors
struct Descriptor {
    Kind kind = KIND_SITE;
    unsigned n = 0, f = 0, k = 0, s = 0;
    std::string mode = "round";
    size_t count = 0, rows = 0, cols = 0;
    int owner = 0;
    std::string adder = "ripple";

    static Descriptor site(unsigned n, unsigned f, unsigned k, unsigned s, const std::string& mode,
                           const std::string& adder = "ripple") {
        Descriptor d;
        d.kind = KIND_SITE;
        d.n = n;
        d.f = f;
        d.k = k;
        d.s = s;
        d.mode = mode;
        d.adder = adder;
        d.validate();
        return d;
    }
    static Descriptor triples(size_t count, unsigned k, unsigned s, const std::string& adder = "ripple") {
        Descriptor d;
        d.kind = KIND_TRIPLES;
        d.count = count;
        d.k = k;
        d.s = s;
        d.adder = adder;
        d.validate();
        return d;
    }
    static Descriptor mattriple(size_t rows, size_t cols, unsigned k, unsigned s, const std::string& adder = "ripple") {
        Descriptor d;
        d.kind = KIND_MATTRIPLE;
        d.rows = rows;
        d.cols = cols;
        d.k = k;
        d.s = s;
        d.adder = adder;
        d.validate();
        return d;
    }
    static Descriptor inmask(int owner, size_t count, unsigned k, unsigned s, const std::string& adder = "ripple") {
        Descriptor d;
        d.kind = KIND_INMASK;
        d.owner = owner;
        d.count = count;
        d.k = k;
        d.s = s;
        d.adder = adder;
        d.validate();
        return d;
    }

    static Descriptor outmask(unsigned k, unsigned s, const std::string& adder = "ripple") {
        Descriptor d;
        d.kind = KIND_OUTMASK;
        d.k = k;
        d.s = s;
        d.adder = adder;
        d.validate();
        return d;
    }

    unsigned width() const { return k + s; }
    unsigned m() const { return f + n + s; }  // sites: modulus exponent of [[rho']]
    bool two_openings() const { return kind == KIND_SITE && f >= 1 && mode == "exact"; }

    void validate() const {
        if (k < 1 || s < 1 || 2 * ((uint64_t)k + s) > 320) throw std::invalid_argument("need integers k >= 1, s >= 1 with 2(k+s) <= 320");
        if (k > 64 || s > 64) throw std::invalid_argument("this implementation supports k <= 64 and s <= 64");
        if (adder != "ripple" && adder != "sklansky") throw std::invalid_argument("adder must be ripple or sklansky");
        switch (kind) {
            case KIND_SITE:
                if (n < 1 || n > MAX_PLANE_BITS) throw std::invalid_argument("n must be in 1..20");
                if (f > MAX_PLANE_BITS) throw std::invalid_argument("f must be in 0..20");
                if (n > k) throw std::invalid_argument("sites need n <= k");
                if (f + n > k) throw std::invalid_argument("sites need f + n <= k");
                if (mode != "round" && mode != "exact") throw std::invalid_argument("mode must be round or exact");
                break;
            case KIND_TRIPLES:
                if (count < 1) throw std::invalid_argument("count must be a positive integer");
                break;
            case KIND_MATTRIPLE:
                if (rows < 1 || cols < 1) throw std::invalid_argument("rows and cols must be positive integers");
                break;
            case KIND_INMASK:
                if (owner != 0 && owner != 1) throw std::invalid_argument("owner must be 0 or 1");
                if (count < 1) throw std::invalid_argument("count must be a positive integer");
                break;
            case KIND_OUTMASK:
                break;
            default:
                throw std::invalid_argument("unknown descriptor kind");
        }
    }

    // circuits_ppml.descriptor_name.
    std::string name() const {
        const std::string tail = adder == "ripple" ? "" : "_" + adder;
        const std::string ks = "k" + std::to_string(k) + "_s" + std::to_string(s);
        switch (kind) {
            case KIND_SITE:
                return "site_n" + std::to_string(n) + "_f" + std::to_string(f) + "_" + ks + "_" + mode + tail;
            case KIND_TRIPLES:
                return "triples" + std::to_string(count) + "_" + ks + tail;
            case KIND_MATTRIPLE:
                return "mattriple_" + std::to_string(rows) + "x" + std::to_string(cols) + "_" + ks + tail;
            case KIND_OUTMASK:
                return "gate_n1_" + ks + "_shared_om1" + tail;
            default:
                return "inmask_p" + std::to_string(owner) + "_c" + std::to_string(count) + "_" + ks + tail;
        }
    }

    // json.dumps(descriptor, sort_keys=True, separators=(",", ":")).
    std::string to_json() const {
        const std::string ad = "{\"adder\":" + json::quote(adder);
        const std::string kk = ",\"k\":" + std::to_string(k), ss = ",\"s\":" + std::to_string(s);
        switch (kind) {
            case KIND_SITE:
                return ad + ",\"f\":" + std::to_string(f) + kk + ",\"kind\":\"site\",\"mode\":" + json::quote(mode) +
                       ",\"n\":" + std::to_string(n) + ss + "}";
            case KIND_TRIPLES:
                return ad + ",\"count\":" + std::to_string(count) + kk + ",\"kind\":\"triples\"" + ss + "}";
            case KIND_MATTRIPLE:
                return ad + ",\"cols\":" + std::to_string(cols) + kk + ",\"kind\":\"mattriple\",\"rows\":" +
                       std::to_string(rows) + ss + "}";
            case KIND_OUTMASK:
                return ad + kk + ",\"kind\":\"outmask\"" + ss + "}";
            default:
                return ad + ",\"count\":" + std::to_string(count) + kk + ",\"kind\":\"inmask\",\"owner\":" +
                       std::to_string(owner) + ss + "}";
        }
    }

    friend bool operator==(const Descriptor& a, const Descriptor& b) { return a.to_json() == b.to_json(); }
    friend bool operator!=(const Descriptor& a, const Descriptor& b) { return !(a == b); }
};

// AND gates of ONE instance (from the circuit file the engine will run).
inline size_t and_count(circuit::CircuitStore& store, const Descriptor& d) {
    d.validate();
    return store.get(d.name())->n_and();
}

// ------------------------------------------------------------------ material
struct SiteMaterial {
    int role = 0;
    Descriptor desc;
    SharePair rp;          // over Z_{2^m}
    fss::DpfKey plane_a;   // domain 2^n
    bool has_b = false;    // mode exact with f >= 1
    fss::DpfKey plane_b;   // domain 2^f
    SharePair rh, r2;      // over Z_{2^(n+s)}

    unsigned n() const { return desc.n; }
    unsigned f() const { return desc.f; }
    unsigned k() const { return desc.k; }
    unsigned s() const { return desc.s; }
    unsigned m() const { return desc.m(); }
    unsigned width() const { return desc.width(); }
    bool two_openings() const { return has_b; }
};

struct TripleMaterial {
    int role = 0;
    Descriptor desc;
    std::vector<SharePair> a, b, c;
};

struct MatTripleMaterial {
    int role = 0;
    Descriptor desc;
    std::vector<SharePair> a, b, c;  // a: rows x cols, row-major
    size_t rows() const { return desc.rows; }
    size_t cols() const { return desc.cols; }
};

struct InMaskMaterial {
    int role = 0;
    Descriptor desc;
    std::vector<SharePair> shares;
    bool has_values = false;   // at the owner only
    std::vector<u128> values;  // the masks rho_j
    int owner() const { return desc.owner; }
    size_t count() const { return desc.count; }
};

struct OutMaskMaterial {
    int role = 0;
    Descriptor desc;
    SharePair share;  // [[2^k rho]] over Z_{2^w}; value and tag are multiples of 2^k
};

// The outputs of one execution, by kind (one object per instance).
struct Generated {
    std::vector<SiteMaterial> sites;
    std::vector<TripleMaterial> triples;
    std::vector<MatTripleMaterial> mattriples;
    std::vector<InMaskMaterial> inmasks;
    std::vector<OutMaskMaterial> outmasks;
};

// ------------------------------------------------------------------ Pi_Gen
namespace detail {

using protocol::detail::need;
using protocol::detail::RunResult;

// This party's shares of `count` w-bit values under one share_out label q.
inline std::vector<SharePair> share_vec(int role, const RunResult& rr, const std::string& q, size_t count, unsigned w,
                                        size_t j) {
    const size_t width = count * w;
    const circuit::BitMat& v = role == 0 ? need(rr.own, q + "sigma", width) : need(rr.eval.priv, q + "share1", width);
    const circuit::BitMat& t = role == 0 ? need(rr.own, q + "mu_sigma", width) : need(rr.eval.priv, q + "mu_share1", width);
    std::vector<SharePair> out(count);
    for (size_t i = 0; i < count; i++) out[i] = SharePair(v.word(j, i * w, w), t.word(j, i * w, w));
    return out;
}

inline void assemble(int role, const Descriptor& d, const RunResult& rr, size_t j, Generated& g) {
    const unsigned k = d.k, s = d.s, w = d.width();
    switch (d.kind) {
        case KIND_SITE: {
            SiteMaterial m;
            m.role = role;
            m.desc = d;
            m.rp = protocol::detail::share_out(role, rr, "rp.", d.m(), true, j);
            m.plane_a = protocol::detail::assemble_dpf(role, d.n, k, s, rr.eval, "dpf.", j);
            if (d.two_openings()) {
                m.has_b = true;
                m.plane_b = protocol::detail::assemble_dpf(role, d.f, k, s, rr.eval, "pb.", j);
                m.rh = protocol::detail::share_out(role, rr, "rh.", d.n + s, true, j);
                m.r2 = protocol::detail::share_out(role, rr, "r2.", d.n + s, true, j);
            }
            g.sites.push_back(std::move(m));
            break;
        }
        case KIND_TRIPLES: {
            TripleMaterial m;
            m.role = role;
            m.desc = d;
            m.a = share_vec(role, rr, "a.", d.count, w, j);
            m.b = share_vec(role, rr, "b.", d.count, w, j);
            m.c = share_vec(role, rr, "c.", d.count, w, j);
            g.triples.push_back(std::move(m));
            break;
        }
        case KIND_MATTRIPLE: {
            MatTripleMaterial m;
            m.role = role;
            m.desc = d;
            m.a = share_vec(role, rr, "a.", d.rows * d.cols, w, j);
            m.b = share_vec(role, rr, "b.", d.cols, w, j);
            m.c = share_vec(role, rr, "c.", d.rows, w, j);
            g.mattriples.push_back(std::move(m));
            break;
        }
        case KIND_OUTMASK: {
            OutMaskMaterial m;
            m.role = role;
            m.desc = d;
            const SharePair hi = protocol::detail::share_out(role, rr, "om0.", s, true, j);  // s-bit high parts
            m.share = SharePair(hi.value << k, hi.tag << k);                                // shares of 2^k rho
            g.outmasks.push_back(m);
            break;
        }
        default: {
            InMaskMaterial m;
            m.role = role;
            m.desc = d;
            m.shares = share_vec(role, rr, "m.", d.count, w, j);
            if (role == d.owner) {
                const circuit::BitMat& rho = need(rr.own, "rho", d.count * w);
                m.has_values = true;
                m.values.resize(d.count);
                for (size_t i = 0; i < d.count; i++) m.values[i] = rho.word(j, i * w, w);
            }
            g.inmasks.push_back(std::move(m));
            break;
        }
    }
}

}  // namespace detail

// This party's material for `batch` independent instances of one item (ONE
// circuit execution on the session engine).  `rng` (tests) replaces the
// operating-system coins.
inline Generated generate(Session& session, const Descriptor& desc, size_t batch = 1, RandomSource* rng = nullptr) {
    session.require_active();
    if (batch < 1) throw std::invalid_argument("batch must be a positive integer");
    desc.validate();
    if (desc.k != session.k() || desc.s != session.s())
        throw std::invalid_argument("the descriptor does not use the session's k and s");
    const std::shared_ptr<const circuit::Circuit> c = session.store().get(desc.name());
    const protocol::detail::RunResult rr = protocol::detail::run(session, *c, abb::Inputs(), batch, rng);
    Generated g;
    for (size_t j = 0; j < batch; j++) detail::assemble(session.role(), desc, rr, j, g);
    return g;
}

// ------------------------------------------------------------------ trusted-dealer reference (tests only)
namespace detail {
inline u128 random_bits(RandomSource& rng, unsigned bits) {
    uint8_t buf[16];
    rng.fill(buf, 16);
    return u128_load_le(buf) & u128_mask(bits);
}
// Shares (value, tag) of both parties modulo 2^bits.
inline std::pair<SharePair, SharePair> split(u128 value, u128 delta, unsigned bits, RandomSource& rng) {
    const u128 mask = u128_mask(bits);
    const u128 v0 = random_bits(rng, bits), t0 = random_bits(rng, bits);
    value = value & mask;
    return std::make_pair(SharePair(v0, t0), SharePair((value - v0) & mask, (delta * value - t0) & mask));
}
}  // namespace detail

// Both parties' material of one site as a trusted dealer would produce it;
// `delta` is the unreduced Delta_0 + Delta_1.  NOT a protocol.
inline std::pair<SiteMaterial, SiteMaterial> deal_site(const Descriptor& d, u128 delta, RandomSource& rng) {
    d.validate();
    if (d.kind != KIND_SITE) throw std::invalid_argument("deal_site takes a site descriptor");
    const unsigned n = d.n, f = d.f, k = d.k, s = d.s, m = d.m();
    const u128 rp = detail::random_bits(rng, m);
    const uint64_t rho_l = u128_lo(rp & u128_mask(f)), rho_h = u128_lo((rp >> f) & u128_mask(n));
    SiteMaterial m0, m1;
    m0.role = 0;
    m1.role = 1;
    m0.desc = m1.desc = d;
    const std::pair<SharePair, SharePair> rps = detail::split(rp, delta, m, rng);
    m0.rp = rps.first;
    m1.rp = rps.second;
    if (d.two_openings()) {
        const u128 r2 = detail::random_bits(rng, n + s);
        const std::pair<fss::DpfKey, fss::DpfKey> pa = fss::dpf_gen(u128_lo(r2 & u128_mask(n)), 1, delta, n, k, s, rng);
        const std::pair<fss::DpfKey, fss::DpfKey> pb = fss::dpf_gen(rho_l, 1, delta, f, k, s, rng);
        const std::pair<SharePair, SharePair> rh = detail::split(u128(rho_h), delta, n + s, rng);
        const std::pair<SharePair, SharePair> r2s = detail::split(r2, delta, n + s, rng);
        m0.plane_a = pa.first;
        m1.plane_a = pa.second;
        m0.has_b = m1.has_b = true;
        m0.plane_b = pb.first;
        m1.plane_b = pb.second;
        m0.rh = rh.first;
        m1.rh = rh.second;
        m0.r2 = r2s.first;
        m1.r2 = r2s.second;
    } else {
        const std::pair<fss::DpfKey, fss::DpfKey> pa = fss::dpf_gen(rho_h, 1, delta, n, k, s, rng);
        m0.plane_a = pa.first;
        m1.plane_a = pa.second;
    }
    return std::make_pair(std::move(m0), std::move(m1));
}

inline std::pair<MatTripleMaterial, MatTripleMaterial> deal_mattriple(const Descriptor& d, u128 delta, RandomSource& rng) {
    d.validate();
    if (d.kind != KIND_MATTRIPLE) throw std::invalid_argument("deal_mattriple takes a mattriple descriptor");
    const unsigned w = d.width();
    const u128 mask = u128_mask(w);
    std::vector<u128> a(d.rows * d.cols), b(d.cols), c(d.rows, u128(0));
    for (u128& x : a) x = detail::random_bits(rng, w);
    for (u128& x : b) x = detail::random_bits(rng, w);
    for (size_t i = 0; i < d.rows; i++)
        for (size_t j = 0; j < d.cols; j++) c[i] = (c[i] + a[i * d.cols + j] * b[j]) & mask;
    MatTripleMaterial m0, m1;
    m0.role = 0;
    m1.role = 1;
    m0.desc = m1.desc = d;
    auto fill = [&](const std::vector<u128>& v, std::vector<SharePair>& s0, std::vector<SharePair>& s1) {
        for (u128 x : v) {
            const std::pair<SharePair, SharePair> p = detail::split(x, delta, w, rng);
            s0.push_back(p.first);
            s1.push_back(p.second);
        }
    };
    fill(a, m0.a, m1.a);
    fill(b, m0.b, m1.b);
    fill(c, m0.c, m1.c);
    return std::make_pair(std::move(m0), std::move(m1));
}

inline std::pair<TripleMaterial, TripleMaterial> deal_triples(const Descriptor& d, u128 delta, RandomSource& rng) {
    d.validate();
    if (d.kind != KIND_TRIPLES) throw std::invalid_argument("deal_triples takes a triples descriptor");
    const unsigned w = d.width();
    TripleMaterial m0, m1;
    m0.role = 0;
    m1.role = 1;
    m0.desc = m1.desc = d;
    for (size_t j = 0; j < d.count; j++) {
        const u128 a = detail::random_bits(rng, w), b = detail::random_bits(rng, w), c = (a * b) & u128_mask(w);
        const std::pair<SharePair, SharePair> pa = detail::split(a, delta, w, rng), pb = detail::split(b, delta, w, rng),
                                              pc = detail::split(c, delta, w, rng);
        m0.a.push_back(pa.first);
        m1.a.push_back(pa.second);
        m0.b.push_back(pb.first);
        m1.b.push_back(pb.second);
        m0.c.push_back(pc.first);
        m1.c.push_back(pc.second);
    }
    return std::make_pair(std::move(m0), std::move(m1));
}

inline std::pair<InMaskMaterial, InMaskMaterial> deal_inmask(const Descriptor& d, u128 delta, RandomSource& rng) {
    d.validate();
    if (d.kind != KIND_INMASK) throw std::invalid_argument("deal_inmask takes an inmask descriptor");
    const unsigned w = d.width();
    InMaskMaterial m[2];
    for (int b = 0; b < 2; b++) {
        m[b].role = b;
        m[b].desc = d;
    }
    m[d.owner].has_values = true;
    for (size_t j = 0; j < d.count; j++) {
        const u128 rho = detail::random_bits(rng, w);
        const std::pair<SharePair, SharePair> p = detail::split(rho, delta, w, rng);
        m[0].shares.push_back(p.first);
        m[1].shares.push_back(p.second);
        m[d.owner].values.push_back(rho);
    }
    return std::make_pair(std::move(m[0]), std::move(m[1]));
}

inline std::pair<OutMaskMaterial, OutMaskMaterial> deal_outmask(const Descriptor& d, u128 delta, RandomSource& rng) {
    d.validate();
    if (d.kind != KIND_OUTMASK) throw std::invalid_argument("deal_outmask takes an outmask descriptor");
    const u128 rho = detail::random_bits(rng, d.s), v0 = detail::random_bits(rng, d.s), t0 = detail::random_bits(rng, d.s);
    const u128 ms = u128_mask(d.s);
    OutMaskMaterial m0, m1;
    m0.role = 0;
    m1.role = 1;
    m0.desc = m1.desc = d;
    m0.share = SharePair(v0 << d.k, t0 << d.k);
    m1.share = SharePair(((rho - v0) & ms) << d.k, ((delta * rho - t0) & ms) << d.k);
    return std::make_pair(m0, m1);
}

// ------------------------------------------------------------------ manifest
struct TableRef {
    std::string name;
    unsigned n = 0, f = 0;
    hashes::hash32 hash{};
};

struct ManifestEntry {
    std::string use;
    Descriptor desc;
    size_t batch = 1;
};

const char* const PHASE_MANIFEST = "manifest";

struct Manifest {
    unsigned k = 64, s = 64;
    std::string mode = "round";
    bool private_model = false;
    hashes::hash32 model_hash{};
    std::vector<TableRef> tables;
    std::vector<ManifestEntry> entries;

    void add(const std::string& use, const Descriptor& desc, size_t batch = 1) {
        desc.validate();
        if (batch < 1) throw std::invalid_argument("batch must be a positive integer");
        ManifestEntry e;
        e.use = use;
        e.desc = desc;
        e.batch = batch;
        entries.push_back(std::move(e));
    }

    bytes encode() const {
        if (mode != "round" && mode != "exact") throw std::invalid_argument("bad mode");
        bytes out = {'C', 'P', 'L', 'T', 'M', 'A', 'N', '1'};
        u32(out, k);
        u32(out, s);
        out.push_back(mode == "round" ? 0 : 1);
        out.push_back(private_model ? 1 : 0);
        out.insert(out.end(), model_hash.begin(), model_hash.end());
        u32(out, (uint32_t)tables.size());
        for (const TableRef& t : tables) {
            blob(out, t.name);
            u32(out, t.n);
            u32(out, t.f);
            out.insert(out.end(), t.hash.begin(), t.hash.end());
        }
        u32(out, (uint32_t)entries.size());
        for (const ManifestEntry& e : entries) {
            blob(out, e.use);
            blob(out, e.desc.to_json());
            u32(out, (uint32_t)e.batch);
        }
        return out;
    }
    // SHA-256 of the encoding (plain, not framed).
    hashes::hash32 digest() const {
        const bytes enc = encode();
        Sha256 h;
        h.update(enc.data(), enc.size());
        return h.finish();
    }
    // AND triples of every execution of the plan.
    size_t and_budget(circuit::CircuitStore& store) const {
        size_t total = 0;
        for (const ManifestEntry& e : entries) {
            const size_t per = and_count(store, e.desc);
            if (per && e.batch > (SIZE_MAX - total) / per) throw std::invalid_argument("the triple budget does not fit in size_t");
            total += per * e.batch;
        }
        return total;
    }
    size_t site_count() const {
        size_t n = 0;
        for (const ManifestEntry& e : entries)
            if (e.desc.kind == KIND_SITE) n += e.batch;
        return n;
    }

private:
    static void u32(bytes& out, uint32_t v) {
        for (int i = 0; i < 4; i++) out.push_back((uint8_t)(v >> (8 * i)));
    }
    static void blob(bytes& out, const std::string& v) {
        u32(out, (uint32_t)v.size());
        out.insert(out.end(), v.begin(), v.end());
    }
};

// Exchange the manifest digest (one round, phase "manifest"); abort on a
// mismatch.  Returns the digest.  Run before the session setup.
inline hashes::hash32 compare_manifest(Channel& ch, const Manifest& manifest, abb::PhaseBook* book = nullptr) {
    const hashes::hash32 d = manifest.digest();
    const int prev = book ? book->current() : abb::PhaseBook::NONE;
    if (book) book->mark(PHASE_MANIFEST);
    bytes peer;
    try {
        peer = ch.exchange(bytes(d.begin(), d.end()), 32, "manifest digests differ");
    } catch (...) {
        if (book) book->mark(prev);
        throw;
    }
    if (book) book->mark(prev);
    if (peer.size() != 32 || std::memcmp(peer.data(), d.data(), 32) != 0) throw ProtocolAbort("manifest digests differ");
    return d;
}

inline std::string hex(const uint8_t* p, size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s.push_back(digits[p[i] >> 4]);
        s.push_back(digits[p[i] & 15]);
    }
    return s;
}
inline std::string hex(const hashes::hash32& h) { return hex(h.data(), h.size()); }
inline std::string hex(const bytes& b) { return hex(b.data(), b.size()); }

}  // namespace ppml
}  // namespace couplet
