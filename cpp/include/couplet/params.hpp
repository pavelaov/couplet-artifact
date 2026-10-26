// couplet/params.hpp -- parameters of the Boolean preprocessing
// (DESIGN.md Sections 1, 7.2, 7.5).  The definitions are those of the Python
// module couplet/prep/params.py.
//
// Constants
//   KAPPA = 128          computational parameter: base OTs per direction, bit
//                        length of the Boolean MAC keys and of the COT rows.
//   PAD   = 128          random padding rows of every OT-extension call.
//   SIGMA_DEFAULT = 40   statistical parameter of the triple bucketing.
//   prg_dom / session_dom  layout of the 64-bit domain constants handed to
//                        hashes::ctr_prg and hashes::H_ccr by the
//                        preprocessing.  The top byte names the family (0x50
//                        counter-mode expansion, 0x48 correlation-robust
//                        hash), so a hash call never coincides with a PRG
//                        block and no constant equals an FSS PRG label word.
//
// Bucket size (Lemma 4 of DESIGN.md 7.5).  ell * B leaky triples are permuted
// and grouped into ell buckets of size B.  An adversary that attacks t leaky
// triples survives the consistency check with probability 2^-t, and a fixed
// bucket then consists of attacked triples only with probability
// C(t, B) / C(ell B, B); a union bound over the buckets gives
//
//     leak_bound(ell, B) = max_{B <= t <= ell B} 2^-t * ell * C(t, B) / C(ell B, B).
//
// The ratio of consecutive terms is (t + 1) / (2 (t + 1 - B)), which is at
// least 1 exactly for t <= 2B - 1, so the maximum is attained at
// t* = min(2B - 1, ell B).  bucket_size(ell, sigma) is the smallest B >= 2
// with leak_bound(ell, B) <= 2^-sigma.  The comparison is exact: both sides
// are integers held in a small multi-precision type (no floating point), so
// the thresholds agree with the rational arithmetic of the Python module.
//
// Size of a triple request (triple_request; Lemma 3 of DESIGN.md 7.4 with the
// constants of design/CHANGELOG.md, "Paper pass 1 corrections", C8 and I1;
// the leaky-AND lemma and the backend theorem of the paper).  The triple
// consistency check also has to catch a party whose key share ignores the
// convention lsb(Delta^B_0) = 1, lsb(Delta^B_1) = 0; that case passes with
// probability at most (3/4)^(ell B) plus a negligible term, which is below
// 2^-127 only from ell B = 306 leaky triples on (MIN_LEAKY_TRIPLES).  A
// request for fewer triples is therefore PADDED: the protocol is run for
// ell' >= max(ell, 2) triples, the smallest such number with
// ell' * bucket_size(ell', sigma) >= 306, and the ell' - ell surplus triples
// are discarded after the check.  For sigma = 40 this concerns requests of at
// most 43 triples (they run with ell' = 44, B = 7, 308 leaky triples), for
// sigma = 64 requests of at most 25 (ell' = 26, B = 12, 312 leaky triples).
// A run always has ell' >= 2 buckets, which is the range in which the closed
// form of Lemma 4 is stated.  Both parties compute the same pair from the
// public (ell, sigma); the Python module has the same function.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "hashes.hpp"

namespace couplet {
namespace prep {

const size_t KAPPA = 128;
const size_t PAD = 128;
const int SIGMA_DEFAULT = 40;

// Families (top byte of the Ascon domain constant).
const unsigned DOM_CCR = 0x48;  // every H_ccr call
const unsigned DOM_PRG = 0x50;  // every ctr_prg expansion

// Uses of ctr_prg inside the preprocessing (second byte of the domain constant).
const unsigned USE_COT_PRG = 0x10;      // column expansion of the OT extension; counter = extend call
const unsigned USE_COT_CHI = 0x11;      // coefficients of the OT-extension check; counter = call
const unsigned USE_BUCKET_PERM = 0x13;  // sort keys of the public bucketing permutation
const unsigned USE_TRIPLE_CHI = 0x14;   // coefficients of the triple consistency check

const uint64_t MAX_PRG_COUNTER = (1ull << 40) - 1;

// Domain of a counter-mode expansion:
//   0x50 << 56 | use << 48 | direction << 40 | counter.
// `use` is in 1..255, so the low 56 bits are at least 2^48 (identifiers below
// 0x100 belong to the primitive layer) and the value never equals a reserved
// FSS label word (all below 2^24).
inline uint64_t prg_dom(unsigned use, int direction = 0, uint64_t counter = 0) {
    if (use < 1 || use > 0xff) throw std::invalid_argument("prg_dom: use must be in 1..255");
    if (direction != 0 && direction != 1) throw std::invalid_argument("prg_dom: direction must be 0 or 1");
    if (counter > MAX_PRG_COUNTER) throw std::invalid_argument("prg_dom: counter must fit in 40 bits");
    return ((uint64_t)DOM_PRG << 56) | ((uint64_t)use << 48) | ((uint64_t)direction << 40) | counter;
}

// Session-specific domain: `kind` in the top byte, below it the low 56 bits of
// the little-endian integer formed by the first 8 bytes of
// SHA-256(frame("Couplet-dom", label, sid)).
inline uint64_t session_dom(unsigned kind, const hashes::Part& label, const hashes::Part& sid) {
    if (kind < 1 || kind > 0xff) throw std::invalid_argument("session_dom: kind must be in 1..255");
    return hashes::session_dom((uint8_t)kind, label, sid);
}

// ------------------------------------------------------------------ Lemma 4
namespace detail {

// Minimal unsigned multi-precision integer (32-bit limbs, little-endian):
// just what the exact evaluation of the Lemma 4 bound needs.
class BigUInt {
public:
    BigUInt() {}
    explicit BigUInt(uint64_t v) {
        if (v) {
            limb_.push_back((uint32_t)v);
            if (v >> 32) limb_.push_back((uint32_t)(v >> 32));
        }
    }

    void mul_u32(uint32_t m) {
        uint64_t carry = 0;
        for (size_t i = 0; i < limb_.size(); i++) {
            const uint64_t t = (uint64_t)limb_[i] * m + carry;
            limb_[i] = (uint32_t)t;
            carry = t >> 32;
        }
        if (carry) limb_.push_back((uint32_t)carry);
        trim();
    }

    void mul_u64(uint64_t m) {
        BigUInt high = *this;
        high.mul_u32((uint32_t)(m >> 32));
        high.shift_left(32);
        mul_u32((uint32_t)m);
        add(high);
    }

    // Division by a 32-bit value; returns the remainder.
    uint32_t div_u32(uint32_t d) {
        if (d == 0) throw std::invalid_argument("BigUInt: division by zero");
        uint64_t rem = 0;
        for (size_t i = limb_.size(); i-- > 0;) {
            const uint64_t cur = (rem << 32) | limb_[i];
            limb_[i] = (uint32_t)(cur / d);
            rem = cur % d;
        }
        trim();
        return (uint32_t)rem;
    }

    void shift_left(uint64_t bits) {
        if (limb_.empty() || bits == 0) return;
        const size_t words = (size_t)(bits / 32);
        const unsigned rest = (unsigned)(bits % 32);
        if (rest) {
            uint32_t carry = 0;
            for (size_t i = 0; i < limb_.size(); i++) {
                const uint32_t v = limb_[i];
                limb_[i] = (v << rest) | carry;
                carry = v >> (32 - rest);
            }
            if (carry) limb_.push_back(carry);
        }
        if (words) limb_.insert(limb_.begin(), words, 0u);
    }

    void add(const BigUInt& o) {
        if (limb_.size() < o.limb_.size()) limb_.resize(o.limb_.size(), 0u);
        uint64_t carry = 0;
        for (size_t i = 0; i < limb_.size(); i++) {
            const uint64_t t = (uint64_t)limb_[i] + (i < o.limb_.size() ? o.limb_[i] : 0u) + carry;
            limb_[i] = (uint32_t)t;
            carry = t >> 32;
        }
        if (carry) limb_.push_back((uint32_t)carry);
    }

    // -1, 0, +1 for *this <, ==, > o.
    int compare(const BigUInt& o) const {
        if (limb_.size() != o.limb_.size()) return limb_.size() < o.limb_.size() ? -1 : 1;
        for (size_t i = limb_.size(); i-- > 0;) {
            if (limb_[i] != o.limb_[i]) return limb_[i] < o.limb_[i] ? -1 : 1;
        }
        return 0;
    }

    bool is_zero() const { return limb_.empty(); }

    // Value as a double (rounded; for tests and logarithms of moderate sizes).
    double to_double() const {
        double v = 0.0;
        for (size_t i = limb_.size(); i-- > 0;) v = v * 4294967296.0 + (double)limb_[i];
        return v;
    }

private:
    void trim() {
        while (!limb_.empty() && limb_.back() == 0) limb_.pop_back();
    }
    std::vector<uint32_t> limb_;
};

// C(n, k), exact.  k must be below 2^32.
inline BigUInt binomial(uint64_t n, uint64_t k) {
    if (k > n) return BigUInt();
    if (k > 0xffffffffull) throw std::invalid_argument("binomial: k too large");
    BigUInt r(1);
    for (uint64_t i = 1; i <= k; i++) {
        r.mul_u64(n - k + i);
        if (r.div_u32((uint32_t)i) != 0) throw std::logic_error("binomial: inexact division");
    }
    return r;
}

inline void check_bucket_args(uint64_t ell, uint64_t bucket) {
    if (ell < 1 || bucket < 1) throw std::invalid_argument("ell and the bucket size must be positive");
    if (bucket > 0xffffffffull || ell > UINT64_MAX / bucket)
        throw std::invalid_argument("ell * bucket does not fit in 64 bits");
}

// t* = min(2B - 1, ell B): where the Lemma 4 maximum is attained.
inline uint64_t leak_argmax(uint64_t ell, uint64_t bucket) {
    const uint64_t total = ell * bucket;
    return (2 * bucket - 1 < total) ? 2 * bucket - 1 : total;
}

}  // namespace detail

// Exact test  leak_bound(ell, B) <= 2^-sigma,  i.e.
//   ell * C(t*, B) * 2^sigma <= C(ell B, B) * 2^t*.
inline bool leak_bound_at_most(uint64_t ell, uint64_t bucket, int sigma) {
    detail::check_bucket_args(ell, bucket);
    if (sigma < 0) throw std::invalid_argument("sigma must be non-negative");
    const uint64_t t = detail::leak_argmax(ell, bucket);
    detail::BigUInt lhs = detail::binomial(t, bucket);
    lhs.mul_u64(ell);
    lhs.shift_left((uint64_t)sigma);
    detail::BigUInt rhs = detail::binomial(ell * bucket, bucket);
    rhs.shift_left(t);
    return lhs.compare(rhs) <= 0;
}

// log2 of leak_bound(ell, B) in floating point (for reports; not used for decisions).
inline double log2_leak_bound(uint64_t ell, uint64_t bucket) {
    detail::check_bucket_args(ell, bucket);
    const uint64_t t = detail::leak_argmax(ell, bucket);
    auto log2_binom = [](uint64_t n, uint64_t k) {
        return (std::lgamma((double)n + 1.0) - std::lgamma((double)k + 1.0) - std::lgamma((double)(n - k) + 1.0)) /
               std::log(2.0);
    };
    return std::log2((double)ell) + log2_binom(t, bucket) - (double)t - log2_binom(ell * bucket, bucket);
}

// Smallest B >= 2 with leak_bound(ell, B) <= 2^-sigma.  B = sigma always
// suffices (for ell = 1 the bound is 2^-B; for ell >= 2, 2^-t C(t, B) <= 1/2
// and ell / C(ell B, B) <= 2^(1-B)).
inline unsigned bucket_size(uint64_t ell, int sigma = SIGMA_DEFAULT) {
    if (ell < 1) throw std::invalid_argument("ell must be positive");
    if (sigma < 1) throw std::invalid_argument("sigma must be positive");
    const unsigned top = sigma > 2 ? (unsigned)sigma : 2u;
    for (unsigned bucket = 2; bucket <= top; bucket++) {
        if (leak_bound_at_most(ell, bucket, sigma)) return bucket;
    }
    throw std::logic_error("bucket_size: no bucket size found");
}

// Smallest number of leaky triples of one triple request: (3/4)^306 < 2^-127
// (see the header comment and triple_request).
const uint64_t MIN_LEAKY_TRIPLES = 306;

// The protocol run that serves a request for AND triples.
struct TripleRequest {
    uint64_t ell;     // triples the run produces: at least the request, and at least 2
    unsigned bucket;  // bucket size B = bucket_size(ell, sigma)
    uint64_t leaky() const { return ell * bucket; }  // leaky triples the run generates and checks
};

// The run that serves a request for `ell` AND triples: run.ell is the
// smallest integer >= max(ell, 2) with run.ell * bucket_size(run.ell, sigma)
// >= MIN_LEAKY_TRIPLES.  The caller keeps `ell` of the run.ell combined
// triples and discards the rest.  run.ell == ell whenever ell >= 2 and
// ell * bucket_size(ell, sigma) >= MIN_LEAKY_TRIPLES (for sigma = 40: from
// ell = 44 on).  The search ends at run.ell = 153 at the latest, because
// B >= 2.
inline TripleRequest triple_request(uint64_t ell, int sigma = SIGMA_DEFAULT) {
    if (ell < 1) throw std::invalid_argument("ell must be positive");
    for (uint64_t run = ell < 2 ? 2 : ell;; run++) {
        const unsigned bucket = bucket_size(run, sigma);
        if (run >= MIN_LEAKY_TRIPLES || run * bucket >= MIN_LEAKY_TRIPLES) return TripleRequest{run, bucket};
    }
}

// Smallest ell for which bucket size `bucket` is sufficient at level sigma
// (leak_bound decreases in ell for B >= 2: doubling, then bisection).
inline uint64_t min_ell(unsigned bucket, int sigma = SIGMA_DEFAULT) {
    if (bucket < 2) throw std::invalid_argument("the bucket size must be at least 2");
    if (sigma < 1) throw std::invalid_argument("sigma must be positive");
    if (leak_bound_at_most(1, bucket, sigma)) return 1;
    uint64_t lo = 1, hi = 2;  // invariant: lo fails, hi is a candidate
    while (!leak_bound_at_most(hi, bucket, sigma)) {
        lo = hi;
        if (hi > (UINT64_MAX / bucket) / 2) throw std::overflow_error("min_ell: threshold exceeds 64 bits");
        hi *= 2;
    }
    while (hi - lo > 1) {
        const uint64_t mid = lo + (hi - lo) / 2;
        if (leak_bound_at_most(mid, bucket, sigma)) hi = mid;
        else lo = mid;
    }
    return hi;
}

}  // namespace prep
}  // namespace couplet
