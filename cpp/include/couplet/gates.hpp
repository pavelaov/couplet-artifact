// couplet/gates.hpp -- local evaluation of the catalogue gates (DESIGN.md
// section 6, Theorem 4); the C++ counterpart of python/couplet/gates.py.
//
// A gate is used in three steps (section 5.3): Gate generates one-time
// material with protocol::gen_gate (an offset r' uniform in Z_{2^(n+s)},
// r = r' mod 2^n, and FSS keys at indices that depend on r); Mask opens
// x_hat = (x + r) mod 2^n; Apply is the LOCAL computation of this header:
// from its GateMaterial and the public x_hat each party derives its share of
// [[G(x)]] over Z_{2^w}, w = k + s.  No function here sends a message.
//
// Notation: N = 2^n; D_c is the payload-1 DCF at index (r + c) mod N
// evaluated at x_hat ([x_hat < (r + c) mod N]), E_c the payload-r DCF,
// w_c = [r + c >= N].  Every function returns this party's spdz2k::AShare
// with the exactness flag of the table of section 6:
//
//   function                      keys used                value                           exact
//   zero_test(g, x_hat)           DPF at r, payload 1      [x = 0]                         yes
//   equality(g, x_hat, c)         same DPF                 [x = c] = DPF(x_hat - c)        yes
//   table(g, x_hat, T)            same DPF                 T[x] = sum_u T[(x_hat-u) mod N] y(u)  yes
//   comparison(g, x_hat, c)       DCFs at c and 0          [x < c] = D_c - D_0 + w_c       yes
//   interval(g, x_hat, p, q)      DCFs at q+1 and p        [p <= x <= q]                   yes
//   multi_interval(g, x_hat, E)   one DCF per endpoint     [e_j <= x < e_(j+1)], a list    yes
//   nonneg(g, x_hat)              DCFs at N/2 and 0        [x < N/2]                       yes
//   msb(g, x_hat) (= sign)        same                     [x >= N/2] = 1 - [x < N/2]      yes
//   relu_large(g, x_hat)          payload-1 and payload-r  x [x < N/2] mod 2^k, n = k      no
//                                 DCFs at N/2 and 0
//   linear_spline(g, x_hat, E, P) payload-1 and payload-r  a_j x + b_j mod 2^k on          no
//                                 DCFs at every endpoint   [e_j, e_(j+1)), n = k
//
// The comparison identity [x < c] = [x_hat < (r+c) mod N] - [x_hat < r] + w_c
// holds over the integers for every c in [0, N], so indicator outputs are
// exact; relu_large and linear_spline use x = x_hat - r modulo 2^k and are
// correct modulo 2^k only (they must be opened with an output mask, change
// E).  msb adds the public constant 1 and therefore needs this party's
// Delta_b.  ReLU, sign and splines on a small domain are table gates:
// relu_table, sign_table, spline_table and function_table build the public
// tables.
//
// GateEval binds one GateMaterial and caches key evaluations;
// GateEval(material, true) evaluates every key once on the whole domain
// (fss::dpf_eval_all / dcf_eval_all) and then answers every x_hat by a
// look-up; the default evaluates the DCFs at the single point x_hat.  The
// table gate always expands the DPF on the whole domain and is Theta(N) work:
// one inner product per plane.
//
// Descriptors.  <gate>_descriptor(...) returns the protocol::GateDescriptor
// that protocol::gen_gate needs for the gate.  GateSpec bundles a gate kind
// with its public parameters: descriptor() for generation, evaluate(g, x_hat)
// for Apply and plain(x) for the plaintext value (the reference of the
// tests).
//
// deal_gate is a trusted-dealer reference of Gate.Gen (NOT a protocol): it
// exists for formula tests of gates whose generation circuit is too large to
// run in a test and for negative controls.
//
// Not provided (section 6): products of two FSS outputs, payloads r^j with
// j >= 2, vector payloads.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fss.hpp"
#include "protocol.hpp"
#include "rng.hpp"
#include "spdz2k.hpp"
#include "u128.hpp"

namespace couplet {
namespace gates {

using protocol::GateDescriptor;
using protocol::GateMaterial;
using protocol::Threshold;
using spdz2k::AShare;

const bool PAYLOAD_ONE = false;
const bool PAYLOAD_R = true;

// 2^n for n <= 64; any other n is an argument error (callers write
// domain_size(n - 1) for the half domain, so n = 0 arrives here as 2^32 - 1
// and must not reach the shift).
inline u128 domain_size(unsigned n) {
    if (n > 64) throw std::invalid_argument("the domain exponent must be in 0..64");
    return u128(1) << n;
}
inline uint64_t domain_mask(unsigned n) { return n >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1); }

// ------------------------------------------------------------------ evaluator
// Party-local evaluator bound to one GateMaterial.  The Delta share is needed
// only by gates that add a public constant (msb).  full_domain = true
// evaluates each key once on the whole domain and serves every later request
// from the table.
class GateEval {
public:
    explicit GateEval(GateMaterial material, bool full_domain = false, unsigned threads = 1)
        : mat_(std::move(material)), full_(full_domain), threads_(threads ? threads : 1) {
        mat_.desc.validate();
        if (mat_.dcf_keys.size() != mat_.desc.thresholds.size() || mat_.w_shares.size() != mat_.desc.thresholds.size() ||
            mat_.rw_shares.size() != mat_.desc.thresholds.size() || mat_.outmask_shares.size() != mat_.desc.n_outmask)
            throw std::invalid_argument("inconsistent gate material");
        if (full_ && mat_.desc.n > fss::EVAL_ALL_MAX_N) throw std::invalid_argument("domain too large for full evaluation");
    }

    const GateMaterial& material() const { return mat_; }
    int role() const { return mat_.role; }
    unsigned n() const { return mat_.desc.n; }
    unsigned k() const { return mat_.desc.k; }
    unsigned s() const { return mat_.desc.s; }
    unsigned width() const { return mat_.desc.k + mat_.desc.s; }
    bool has_delta_share() const { return has_delta_; }
    uint64_t delta_share() const { return delta_; }
    void set_delta_share(uint64_t d) {
        delta_ = d;
        has_delta_ = true;
    }

    uint64_t check_x_hat(uint64_t x_hat) const {
        if ((x_hat & ~domain_mask(n())) != 0) throw std::invalid_argument("x_hat must be an integer in [0, 2^n)");
        return x_hat;
    }

    // Index of the threshold (c, payload) in the material.
    size_t threshold(u128 c, bool payload_r) const {
        for (size_t i = 0; i < mat_.desc.thresholds.size(); i++)
            if (mat_.desc.thresholds[i].c == c && mat_.desc.thresholds[i].payload_r == payload_r) return i;
        throw std::invalid_argument("the gate material has no threshold (" + json::dec(c) + ", " +
                                    (payload_r ? "r" : "one") + ")");
    }

    // [[w_c]] of the threshold (c, payload); w_c = [r + c >= N].
    AShare w(u128 c, bool payload_r = PAYLOAD_ONE) const {
        const protocol::SharePair& p = mat_.w_shares[threshold(c, payload_r)];
        return AShare(p.value, p.tag, width(), true);
    }
    // [[r w_c]] of the payload-r threshold at c.
    AShare rw(u128 c) const {
        const protocol::SharePair& p = mat_.rw_shares[threshold(c, PAYLOAD_R)];
        return AShare(p.value, p.tag, width(), true);
    }

    // Share of the DCF of threshold (c, payload) at x_hat (exact).  x_hat
    // outside [0, 2^n) is an argument error in both modes (the full-domain
    // table has exactly 2^n entries).
    AShare dcf(u128 c, bool payload_r, uint64_t x_hat) {
        check_x_hat(x_hat);
        const size_t i = threshold(c, payload_r);
        if (full_) {
            std::map<size_t, Table>::iterator it = dcf_tables_.find(i);
            if (it == dcf_tables_.end()) {
                Table t;
                fss::dcf_eval_all(mat_.dcf_keys[i], t.y, t.tau, threads_);
                it = dcf_tables_.emplace(i, std::move(t)).first;
            }
            return AShare(it->second.y.at((size_t)x_hat), it->second.tau.at((size_t)x_hat), width(), true);
        }
        const std::pair<size_t, uint64_t> key(i, x_hat);
        std::map<std::pair<size_t, uint64_t>, fss::Share>::iterator it = dcf_points_.find(key);
        if (it == dcf_points_.end()) it = dcf_points_.emplace(key, fss::dcf_eval(mat_.dcf_keys[i], x_hat)).first;
        return AShare(it->second.y, it->second.tau, width(), true);
    }

    // Full-domain shares (y, tau) of the payload-1 DPF at r.
    const std::vector<u128>& dpf_y() {
        expand_dpf();
        return dpf_table_.y;
    }
    const std::vector<u128>& dpf_tau() {
        expand_dpf();
        return dpf_table_.tau;
    }

    // Share of the DPF at the point u (exact); u outside [0, 2^n) is an
    // argument error in both modes.
    AShare dpf(uint64_t u) {
        need_dpf();
        if ((u & ~domain_mask(n())) != 0) throw std::invalid_argument("the evaluation point must lie in [0, 2^n)");
        if (full_ || dpf_ready_) {
            expand_dpf();
            return AShare(dpf_table_.y.at((size_t)u), dpf_table_.tau.at((size_t)u), width(), true);
        }
        const fss::Share sh = fss::dpf_eval(mat_.dpf_key, u);
        return AShare(sh.y, sh.tau, width(), true);
    }

    // D_c + w_c (payload 1) or E_c + r w_c (payload r): the two terms of a threshold.
    AShare step(u128 c, bool payload_r, uint64_t x_hat) {
        if (!payload_r) return dcf(c, PAYLOAD_ONE, x_hat) + w(c, PAYLOAD_ONE);
        return dcf(c, PAYLOAD_R, x_hat) + rw(c);
    }

private:
    struct Table {
        std::vector<u128> y, tau;
    };
    void need_dpf() const {
        if (!mat_.desc.dpf_one) throw std::invalid_argument("the gate material has no DPF key");
    }
    void expand_dpf() {
        need_dpf();
        if (dpf_ready_) return;
        if (mat_.desc.n > fss::EVAL_ALL_MAX_N) throw std::invalid_argument("domain too large for full evaluation");
        fss::dpf_eval_all(mat_.dpf_key, dpf_table_.y, dpf_table_.tau, threads_);
        dpf_ready_ = true;
    }

    GateMaterial mat_;
    bool full_;
    unsigned threads_;
    bool has_delta_ = false;
    uint64_t delta_ = 0;
    std::map<size_t, Table> dcf_tables_;
    std::map<std::pair<size_t, uint64_t>, fss::Share> dcf_points_;
    Table dpf_table_;
    bool dpf_ready_ = false;
};

// ------------------------------------------------------------------ gates on the payload-1 DPF at index r
// [[ [x = c] ]] = the DPF at (x_hat - c) mod N; exact.
inline AShare equality(GateEval& g, uint64_t x_hat, uint64_t c = 0) {
    g.check_x_hat(x_hat);
    if ((c & ~domain_mask(g.n())) != 0) throw std::invalid_argument("c must lie in [0, 2^n)");
    return g.dpf((x_hat - c) & domain_mask(g.n()));
}

// [[ [x = 0] ]]; exact.
inline AShare zero_test(GateEval& g, uint64_t x_hat) { return equality(g, x_hat, 0); }

// [[ T[x] ]] for a public table T: [N] -> Z_{2^k}; exact; Theta(N) work.
// sum_u T[(x_hat - u) mod N] y(u) on both planes: the DPF shares y are an
// indicator of u = r, so the sum selects T[(x_hat - r) mod N] = T[x].
inline AShare table(GateEval& g, uint64_t x_hat, const std::vector<uint64_t>& T) {
    g.check_x_hat(x_hat);
    if (g.n() > 40 || T.size() != ((size_t)1 << g.n())) throw std::invalid_argument("the table must have 2^n entries");
    const uint64_t mask_k = domain_mask(g.k());
    const std::vector<u128>& y = g.dpf_y();
    const std::vector<u128>& tau = g.dpf_tau();
    const size_t size = T.size(), mask_n = size - 1;
    u128 acc_y = u128(0), acc_t = u128(0);
    for (size_t u = 0; u < size; u++) {
        const u128 t = u128(T[((size_t)x_hat - u) & mask_n] & mask_k);
        acc_y = acc_y + t * y[u];
        acc_t = acc_t + t * tau[u];
    }
    const u128 mask = u128_mask(g.width());
    return AShare(acc_y & mask, acc_t & mask, g.width(), true);
}

// ------------------------------------------------------------------ gates on DCFs
namespace detail {
inline u128 threshold_const(const GateEval& g, u128 c) {
    if (c > domain_size(g.n())) throw std::invalid_argument("a threshold must lie in [0, 2^n]");
    return c;
}
inline std::vector<u128> endpoints(const GateEval& g, const std::vector<u128>& pts) {
    if (pts.size() < 2) throw std::invalid_argument("endpoints must be at least two strictly increasing values in [0, 2^n]");
    for (size_t i = 0; i < pts.size(); i++) {
        threshold_const(g, pts[i]);
        if (i && !(pts[i - 1] < pts[i]))
            throw std::invalid_argument("endpoints must be at least two strictly increasing values in [0, 2^n]");
    }
    return pts;
}
inline void need_n_eq_k(const GateEval& g, const char* name) {
    if (g.n() != g.k()) throw std::invalid_argument(std::string(name) + " needs n = k (use a table gate for a small domain)");
}
}  // namespace detail

// [[ [x < c] ]] = D_c - D_0 + w_c for a public c in [0, N]; exact.
inline AShare comparison(GateEval& g, uint64_t x_hat, u128 c) {
    g.check_x_hat(x_hat);
    detail::threshold_const(g, c);
    return (g.step(c, PAYLOAD_ONE, x_hat) - g.dcf(u128(0), PAYLOAD_ONE, x_hat)).with_exact(true);
}

// [[ [p <= x <= q] ]] = D_(q+1) - D_p + w_(q+1) - w_p, 0 <= p <= q < N; exact.
inline AShare interval(GateEval& g, uint64_t x_hat, uint64_t p, uint64_t q) {
    g.check_x_hat(x_hat);
    if (p > q || (q & ~domain_mask(g.n())) != 0) throw std::invalid_argument("need 0 <= p <= q < 2^n");
    return (g.step(u128(q) + u128(1), PAYLOAD_ONE, x_hat) - g.step(u128(p), PAYLOAD_ONE, x_hat)).with_exact(true);
}

// [[ [e_j <= x < e_(j+1)] ]] for consecutive endpoints: m + 1 DCFs, m exact shares.
inline std::vector<AShare> multi_interval(GateEval& g, uint64_t x_hat, const std::vector<u128>& endpoints) {
    g.check_x_hat(x_hat);
    const std::vector<u128> pts = detail::endpoints(g, endpoints);
    std::vector<AShare> steps;
    for (const u128& e : pts) steps.push_back(g.step(e, PAYLOAD_ONE, x_hat));
    std::vector<AShare> out;
    for (size_t j = 0; j + 1 < steps.size(); j++) out.push_back((steps[j + 1] - steps[j]).with_exact(true));
    return out;
}

// [[ [x < N/2] ]]: 1 iff the two's-complement value of x is non-negative; exact.
inline AShare nonneg(GateEval& g, uint64_t x_hat) { return comparison(g, x_hat, domain_size(g.n() - 1)); }

// [[ msb(x) ]] = 1 - [x < N/2], the sign bit; exact.  Needs this party's Delta_b.
inline AShare msb(GateEval& g, uint64_t x_hat) {
    if (!g.has_delta_share()) throw std::invalid_argument("msb adds a public constant and needs this party's Delta share");
    const AShare below = comparison(g, x_hat, domain_size(g.n() - 1));
    return (-below).add_public(u128(1), g.role(), g.delta_share()).with_exact(true);
}
inline AShare sign(GateEval& g, uint64_t x_hat) { return msb(g, x_hat); }

// [[ ReLU(x) ]] for n = k, two's complement: x_hat b - (E_c - E_0 + [[r w_c]])
// with c = N/2 and b = D_c - D_0 + w_c = [x < N/2].  Correct modulo 2^k; the
// representative is not reduced, so the share is not exact.
inline AShare relu_large(GateEval& g, uint64_t x_hat) {
    detail::need_n_eq_k(g, "relu_large");
    g.check_x_hat(x_hat);
    const u128 c = domain_size(g.n() - 1);
    const AShare b = g.step(c, PAYLOAD_ONE, x_hat) - g.dcf(u128(0), PAYLOAD_ONE, x_hat);  // [[ [x < N/2] ]]
    const AShare rb = g.step(c, PAYLOAD_R, x_hat) - g.dcf(u128(0), PAYLOAD_R, x_hat);     // [[ r [x < N/2] ]]
    return b.scale(u128(x_hat)) - rb;
}

// One piece a x + b of a spline; a and b are elements of Z_{2^k} (negative
// coefficients in two's complement).
struct Piece {
    u128 a = u128(0), b = u128(0);
    Piece() {}
    Piece(u128 a_, u128 b_) : a(a_), b(b_) {}
};

// [[ a_j x + b_j ]] for x in [e_j, e_(j+1)) (0 elsewhere), n = k; not exact.
// sum_j (a_j x_hat + b_j) ind_j - a_j (r ind_j) with the indicators ind_j from
// the payload-1 DCFs and r ind_j from the payload-r DCFs at the endpoints.
inline AShare linear_spline(GateEval& g, uint64_t x_hat, const std::vector<u128>& endpoints,
                            const std::vector<Piece>& pieces) {
    detail::need_n_eq_k(g, "linear_spline");
    g.check_x_hat(x_hat);
    const std::vector<u128> pts = detail::endpoints(g, endpoints);
    if (pieces.size() != pts.size() - 1)
        throw std::invalid_argument("one piece (a_j, b_j) per interval [e_j, e_(j+1)) is required");
    std::vector<AShare> ones, rs;
    for (const u128& e : pts) {
        ones.push_back(g.step(e, PAYLOAD_ONE, x_hat));
        rs.push_back(g.step(e, PAYLOAD_R, x_hat));
    }
    AShare acc(u128(0), u128(0), g.width());
    for (size_t j = 0; j < pieces.size(); j++)
        acc = acc + (ones[j + 1] - ones[j]).scale(pieces[j].a * u128(x_hat) + pieces[j].b) -
              (rs[j + 1] - rs[j]).scale(pieces[j].a);
    return acc.with_exact(false);
}

// ------------------------------------------------------------------ public tables and plaintext references
// T[x] = f(x) mod 2^k for x in [0, 2^n).
inline std::vector<uint64_t> function_table(unsigned n, unsigned k, const std::function<uint64_t(uint64_t)>& f) {
    if (n > 30) throw std::invalid_argument("table too large");
    std::vector<uint64_t> T((size_t)1 << n);
    for (size_t x = 0; x < T.size(); x++) T[x] = f(x) & domain_mask(k);
    return T;
}
// ReLU on n-bit two's complement: T[x] = x for x < N/2 and 0 otherwise.
// 1 <= n <= 30 (checked before the half-domain constant is formed).
inline std::vector<uint64_t> relu_table(unsigned n, unsigned k) {
    if (n < 1 || n > 30) throw std::invalid_argument("relu_table needs 1 <= n <= 30");
    const uint64_t half = (uint64_t)1 << (n - 1);
    return function_table(n, k, [half](uint64_t x) { return x < half ? x : 0; });
}
// Sign bit of n-bit two's complement: T[x] = [x >= N/2].  1 <= n <= 30.
inline std::vector<uint64_t> sign_table(unsigned n, unsigned k = 1) {
    if (n < 1 || n > 30) throw std::invalid_argument("sign_table needs 1 <= n <= 30");
    const uint64_t half = (uint64_t)1 << (n - 1);
    return function_table(n, k ? k : 1, [half](uint64_t x) { return (uint64_t)(x >= half); });
}
// Plaintext spline: a_j x + b_j mod 2^k on [e_j, e_(j+1)), 0 outside.
inline uint64_t plain_linear_spline(uint64_t x, unsigned k, const std::vector<u128>& endpoints,
                                    const std::vector<Piece>& pieces) {
    for (size_t j = 0; j < pieces.size() && j + 1 < endpoints.size(); j++)
        if (!(u128(x) < endpoints[j]) && u128(x) < endpoints[j + 1])
            return u128_lo(pieces[j].a * u128(x) + pieces[j].b) & domain_mask(k);
    return 0;
}
inline std::vector<uint64_t> spline_table(unsigned n, unsigned k, const std::vector<u128>& endpoints,
                                          const std::vector<Piece>& pieces) {
    return function_table(n, k, [&](uint64_t x) { return plain_linear_spline(x, k, endpoints, pieces); });
}
// ReLU on n-bit two's complement, as an element of [0, 2^n).  1 <= n <= 64.
inline uint64_t plain_relu(uint64_t x, unsigned n) {
    if (n < 1 || n > 64) throw std::invalid_argument("plain_relu needs 1 <= n <= 64");
    return x < ((uint64_t)1 << (n - 1)) ? x : 0;
}

// ------------------------------------------------------------------ descriptors
// One payload-1 DPF at index r: zero test, equality and any table.
inline GateDescriptor table_descriptor(unsigned n, unsigned k, unsigned s, bool auth_input = false, unsigned n_outmask = 0) {
    return GateDescriptor(n, k, s, auth_input, true, std::vector<Threshold>(), n_outmask);
}
// One payload-1 DCF per endpoint (in the given order; duplicates removed).
inline GateDescriptor multi_interval_descriptor(unsigned n, unsigned k, unsigned s, const std::vector<u128>& endpoints,
                                                bool auth_input = false, unsigned n_outmask = 0) {
    std::vector<Threshold> ths;
    for (const u128& e : endpoints) {
        bool seen = false;
        for (const Threshold& t : ths) seen = seen || t.c == e;
        if (!seen) ths.push_back(Threshold(e, PAYLOAD_ONE));
    }
    return GateDescriptor(n, k, s, auth_input, false, ths, n_outmask);
}
// Payload-1 DCFs at c and at 0.
inline GateDescriptor comparison_descriptor(unsigned n, unsigned k, unsigned s, u128 c, bool auth_input = false,
                                            unsigned n_outmask = 0) {
    return multi_interval_descriptor(n, k, s, std::vector<u128>{c, u128(0)}, auth_input, n_outmask);
}
// Payload-1 DCFs at q + 1 and at p.
inline GateDescriptor interval_descriptor(unsigned n, unsigned k, unsigned s, uint64_t p, uint64_t q,
                                          bool auth_input = false, unsigned n_outmask = 0) {
    if (p > q || (q & ~domain_mask(n)) != 0) throw std::invalid_argument("need 0 <= p <= q < 2^n");
    return multi_interval_descriptor(n, k, s, std::vector<u128>{u128(q) + u128(1), u128(p)}, auth_input, n_outmask);
}
// Sign / MSB: the comparison with c = N/2.
inline GateDescriptor msb_descriptor(unsigned n, unsigned k, unsigned s, bool auth_input = false, unsigned n_outmask = 0) {
    return comparison_descriptor(n, k, s, domain_size(n - 1), auth_input, n_outmask);
}
// Payload-1 and payload-r DCFs at every endpoint, n = k; one output mask by default.
inline GateDescriptor linear_spline_descriptor(unsigned k, unsigned s, const std::vector<u128>& endpoints,
                                               bool auth_input = true, unsigned n_outmask = 1) {
    std::vector<u128> pts;
    for (const u128& e : endpoints) {
        bool seen = false;
        for (const u128& p : pts) seen = seen || p == e;
        if (!seen) pts.push_back(e);
    }
    std::vector<Threshold> ths;
    for (const u128& e : pts) ths.push_back(Threshold(e, PAYLOAD_ONE));
    for (const u128& e : pts) ths.push_back(Threshold(e, PAYLOAD_R));
    return GateDescriptor(k, k, s, auth_input, false, ths, n_outmask);
}
// Payload-1 and payload-r DCFs at N/2 and 0, n = k; one output mask by default.
inline GateDescriptor relu_large_descriptor(unsigned k, unsigned s, bool auth_input = true, unsigned n_outmask = 1) {
    return linear_spline_descriptor(k, s, std::vector<u128>{domain_size(k - 1), u128(0)}, auth_input, n_outmask);
}
// Material without keys: an offset (and output masks) only.
inline GateDescriptor offset_descriptor(unsigned n, unsigned k, unsigned s, bool auth_input = true, unsigned n_outmask = 0) {
    return GateDescriptor(n, k, s, auth_input, false, std::vector<Threshold>(), n_outmask);
}

// ------------------------------------------------------------------ trusted-dealer reference of Gate.Gen
// Both parties' GateMaterial as a trusted dealer would produce it (section
// 4.3).  NOT a protocol: the caller knows the full MAC key delta = Delta_0 +
// Delta_1 and the function samples r', the roots and all shares itself, with
// fss::dpf_gen / dcf_gen for the keys.  It is the plaintext counterpart of
// protocol::gen_gate.  `rp` (optional) fixes r'.
inline std::pair<GateMaterial, GateMaterial> deal_gate(const GateDescriptor& desc, u128 delta, RandomSource& rng,
                                                       const u128* rp_fixed = nullptr) {
    desc.validate();
    const unsigned n = desc.n, k = desc.k, s = desc.s, width = k + s, m = desc.rp_bits();
    if (s + 1 < 128 && (delta >> (s + 1)) != u128(0))
        throw std::invalid_argument("delta must be the unreduced sum of two s-bit shares");
    const u128 rp = (rp_fixed ? *rp_fixed : spdz2k::detail::random_u128(rng, m)) & u128_mask(m);
    const uint64_t r = u128_lo(rp) & domain_mask(n);
    std::pair<GateMaterial, GateMaterial> out;
    GateMaterial* g[2] = {&out.first, &out.second};
    for (int b = 0; b < 2; b++) {
        g[b]->role = b;
        g[b]->desc = desc;
    }
    // Additive shares (value, tag) of both parties modulo 2^bits.
    auto split = [&](u128 value, u128 tag, unsigned bits, bool tagged, protocol::SharePair sh[2]) {
        const u128 mask = u128_mask(bits);
        const u128 v0 = spdz2k::detail::random_u128(rng, bits), t0 = spdz2k::detail::random_u128(rng, bits);
        sh[0] = protocol::SharePair(v0, tagged ? t0 : u128(0));
        sh[1] = protocol::SharePair((value - v0) & mask, tagged ? ((tag - t0) & mask) : u128(0));
    };
    protocol::SharePair sh[2];
    split(desc.auth_input ? rp : u128(r), (delta * rp) & u128_mask(m), m, desc.auth_input, sh);
    for (int b = 0; b < 2; b++) {
        g[b]->rp_share = sh[b].value;
        g[b]->rp_tag_share = sh[b].tag;
    }
    if (desc.dpf_one) {
        std::pair<fss::DpfKey, fss::DpfKey> keys = fss::dpf_gen(r, 1, delta, n, k, s, rng);
        out.first.dpf_key = std::move(keys.first);
        out.second.dpf_key = std::move(keys.second);
    }
    const u128 wmask = u128_mask(width);
    for (const Threshold& t : desc.thresholds) {
        const u128 sum = u128(r) + t.c;
        const bool w_c = !(sum < domain_size(n));
        const uint64_t alpha = u128_lo(sum) & domain_mask(n);
        std::pair<fss::DcfKey, fss::DcfKey> keys = fss::dcf_gen(alpha, t.payload_r ? r : 1, delta, n, k, s, rng);
        out.first.dcf_keys.push_back(std::move(keys.first));
        out.second.dcf_keys.push_back(std::move(keys.second));
        split(u128(w_c ? 1 : 0), w_c ? (delta & wmask) : u128(0), width, true, sh);
        for (int b = 0; b < 2; b++) g[b]->w_shares.push_back(sh[b]);
        if (t.payload_r) {
            const u128 rw = w_c ? u128(r) : u128(0);
            split(rw, (delta * rw) & wmask, width, true, sh);
        } else {
            sh[0] = sh[1] = protocol::SharePair();
        }
        for (int b = 0; b < 2; b++) g[b]->rw_shares.push_back(sh[b]);
    }
    for (unsigned i = 0; i < desc.n_outmask; i++) {
        const u128 rho = spdz2k::detail::random_u128(rng, s);
        split(rho, (delta * rho) & u128_mask(s), s, true, sh);  // shares of the high s bits
        for (int b = 0; b < 2; b++) g[b]->outmask_shares.push_back(protocol::SharePair(sh[b].value << k, sh[b].tag << k));
    }
    return out;
}

// ------------------------------------------------------------------ GateSpec
// A catalogue gate G with its public parameters (the G of Gate(gid, G)).
struct GateSpec {
    enum Kind {
        ZERO_TEST, EQUALITY, TABLE, COMPARISON, INTERVAL, MULTI_INTERVAL, NONNEG, MSB, RELU_LARGE, LINEAR_SPLINE,
        OFFSET_ONLY
    };
    Kind kind = OFFSET_ONLY;
    unsigned n = 0, k = 0, s = 0;
    bool auth_input = false;
    unsigned n_outmask = 0;
    u128 c = u128(0);                 // EQUALITY, COMPARISON
    uint64_t p = 0, q = 0;            // INTERVAL
    std::vector<u128> endpoints;      // MULTI_INTERVAL, LINEAR_SPLINE
    std::vector<Piece> pieces;        // LINEAR_SPLINE
    std::vector<uint64_t> table;      // TABLE

    static GateSpec zero_test(unsigned n, unsigned k, unsigned s, bool auth = false) {
        return make(ZERO_TEST, n, k, s, auth, 0);
    }
    static GateSpec equality(unsigned n, unsigned k, unsigned s, uint64_t c, bool auth = false) {
        if ((c & ~domain_mask(n)) != 0) throw std::invalid_argument("c must lie in [0, 2^n)");
        GateSpec g = make(EQUALITY, n, k, s, auth, 0);
        g.c = u128(c);
        return g;
    }
    static GateSpec table_gate(unsigned n, unsigned k, unsigned s, std::vector<uint64_t> T, bool auth = false) {
        if (n > 40 || T.size() != ((size_t)1 << n)) throw std::invalid_argument("the table must have 2^n entries");
        for (uint64_t& v : T) v &= domain_mask(k);
        GateSpec g = make(TABLE, n, k, s, auth, 0);
        g.table = std::move(T);
        return g;
    }
    static GateSpec comparison(unsigned n, unsigned k, unsigned s, u128 c, bool auth = false) {
        GateSpec g = make(COMPARISON, n, k, s, auth, 0);
        g.c = c;
        g.descriptor();
        return g;
    }
    static GateSpec interval(unsigned n, unsigned k, unsigned s, uint64_t p, uint64_t q, bool auth = false) {
        GateSpec g = make(INTERVAL, n, k, s, auth, 0);
        g.p = p;
        g.q = q;
        g.descriptor();
        return g;
    }
    static GateSpec multi_interval(unsigned n, unsigned k, unsigned s, std::vector<u128> endpoints, bool auth = false) {
        GateSpec g = make(MULTI_INTERVAL, n, k, s, auth, 0);
        g.endpoints = std::move(endpoints);
        g.descriptor();
        return g;
    }
    static GateSpec nonneg(unsigned n, unsigned k, unsigned s, bool auth = false) { return make(NONNEG, n, k, s, auth, 0); }
    static GateSpec msb(unsigned n, unsigned k, unsigned s, bool auth = false) { return make(MSB, n, k, s, auth, 0); }
    static GateSpec relu_large(unsigned k, unsigned s, bool auth = true, unsigned n_outmask = 1) {
        return make(RELU_LARGE, k, k, s, auth, n_outmask);
    }
    static GateSpec linear_spline(unsigned k, unsigned s, std::vector<u128> endpoints, std::vector<Piece> pieces,
                                  bool auth = true, unsigned n_outmask = 1) {
        GateSpec g = make(LINEAR_SPLINE, k, k, s, auth, n_outmask);
        g.endpoints = std::move(endpoints);
        g.pieces = std::move(pieces);
        g.descriptor();
        return g;
    }
    static GateSpec offset_only(unsigned n, unsigned k, unsigned s, bool auth = true, unsigned n_outmask = 0) {
        return make(OFFSET_ONLY, n, k, s, auth, n_outmask);
    }

    // The flag of the output sharing.
    bool exact() const { return kind != RELU_LARGE && kind != LINEAR_SPLINE; }
    size_t n_outputs() const {
        if (kind == MULTI_INTERVAL) return endpoints.size() - 1;
        return kind == OFFSET_ONLY ? 0 : 1;
    }

    // The gate descriptor of the material this gate needs.
    GateDescriptor descriptor() const {
        switch (kind) {
            case ZERO_TEST:
            case EQUALITY:
            case TABLE:
                return table_descriptor(n, k, s, auth_input, n_outmask);
            case COMPARISON:
                return comparison_descriptor(n, k, s, c, auth_input, n_outmask);
            case INTERVAL:
                return interval_descriptor(n, k, s, p, q, auth_input, n_outmask);
            case MULTI_INTERVAL:
                check_increasing();
                return multi_interval_descriptor(n, k, s, endpoints, auth_input, n_outmask);
            case NONNEG:
            case MSB:
                return msb_descriptor(n, k, s, auth_input, n_outmask);
            case RELU_LARGE:
                return relu_large_descriptor(k, s, auth_input, n_outmask);
            case LINEAR_SPLINE:
                check_increasing();
                if (pieces.size() != endpoints.size() - 1)
                    throw std::invalid_argument("a spline needs increasing endpoints and one piece per interval");
                return linear_spline_descriptor(k, s, endpoints, auth_input, n_outmask);
            default:
                return offset_descriptor(n, k, s, auth_input, n_outmask);
        }
    }

    // Apply: this party's share(s) of [[G(x)]] (one entry, several for MULTI_INTERVAL).
    std::vector<AShare> evaluate(GateEval& g, uint64_t x_hat) const {
        if (g.n() != n || g.k() != k || g.s() != s)
            throw std::invalid_argument("the gate material does not match the gate parameters");
        switch (kind) {
            case ZERO_TEST: return one(gates::zero_test(g, x_hat));
            case EQUALITY: return one(gates::equality(g, x_hat, u128_lo(c)));
            case TABLE: return one(gates::table(g, x_hat, table));
            case COMPARISON: return one(gates::comparison(g, x_hat, c));
            case INTERVAL: return one(gates::interval(g, x_hat, p, q));
            case MULTI_INTERVAL: return gates::multi_interval(g, x_hat, endpoints);
            case NONNEG: return one(gates::nonneg(g, x_hat));
            case MSB: return one(gates::msb(g, x_hat));
            case RELU_LARGE: return one(gates::relu_large(g, x_hat));
            case LINEAR_SPLINE: return one(gates::linear_spline(g, x_hat, endpoints, pieces));
            default: throw std::invalid_argument("an offset-only gate has no function to apply");
        }
    }

    // Plaintext G(x) as elements of [0, 2^k) (one entry, several for MULTI_INTERVAL).
    std::vector<uint64_t> plain(uint64_t x) const {
        if ((x & ~domain_mask(n)) != 0) throw std::invalid_argument("x out of range");
        const u128 half = domain_size(n - 1), xx = u128(x);
        switch (kind) {
            case ZERO_TEST: return {(uint64_t)(x == 0)};
            case EQUALITY: return {(uint64_t)(xx == c)};
            case TABLE: return {table[(size_t)x]};
            case COMPARISON: return {(uint64_t)(xx < c)};
            case INTERVAL: return {(uint64_t)(p <= x && x <= q)};
            case MULTI_INTERVAL: {
                std::vector<uint64_t> out;
                for (size_t j = 0; j + 1 < endpoints.size(); j++)
                    out.push_back((uint64_t)(!(xx < endpoints[j]) && xx < endpoints[j + 1]));
                return out;
            }
            case NONNEG: return {(uint64_t)(xx < half)};
            case MSB: return {(uint64_t)!(xx < half)};
            case RELU_LARGE: return {xx < half ? x : 0};
            case LINEAR_SPLINE: return {plain_linear_spline(x, k, endpoints, pieces)};
            default: throw std::invalid_argument("an offset-only gate has no function");
        }
    }

private:
    static GateSpec make(Kind kind, unsigned n, unsigned k, unsigned s, bool auth, unsigned n_outmask) {
        GateSpec g;
        g.kind = kind;
        g.n = n;
        g.k = k;
        g.s = s;
        g.auth_input = auth;
        g.n_outmask = n_outmask;
        offset_descriptor(n, k, s, auth, n_outmask);  // validates n, k, s
        return g;
    }
    static std::vector<AShare> one(const AShare& a) { return std::vector<AShare>(1, a); }
    void check_increasing() const {
        if (endpoints.size() < 2) throw std::invalid_argument("endpoints must be strictly increasing");
        for (size_t i = 1; i < endpoints.size(); i++)
            if (!(endpoints[i - 1] < endpoints[i])) throw std::invalid_argument("endpoints must be strictly increasing");
    }
};

}  // namespace gates
}  // namespace couplet
