// couplet/fss.hpp -- plaintext two-plane DPF and DCF (DESIGN.md section 3),
// bit for bit identical to the Python reference couplet/fss.py.
//
// Both schemes are the standard tree constructions whose payload lives in
// Z_{2^w} x Z_{2^w}, w = k + s: the value plane carries beta (zero-extended
// from k bits) and the tag plane carries beta_t = Delta * beta mod 2^w, so
// every local evaluation is a SPDZ2k authenticated share.  The MAC key is
// the integer Delta = Delta_0 + Delta_1 in [0, 2^(s+1)); it is passed as a
// u128 and is NEVER reduced mod 2^s.  Indices are n-bit integers read MSB
// first.  neg_if(t, v) = -v if t == 1 else v (mod 2^w); the final correction
// words use neg_if(t1, .) with t1 the control bit of party 1 after the last
// level, the DCF level words use the control bit before the update.
//
// Parameter range of this implementation: 1 <= n <= 64, 1 <= k <= 64,
// s >= 0, k + s <= 128 (ring elements are held in a u128; the reference
// allows k + s <= 160).  The golden vectors cover (k, s) = (32, 64) and
// (64, 64).
//
// Functions (namespace couplet::fss):
//   dpf_gen / dcf_gen       explicit roots (deterministic), or any byte
//                           source with fill(uint8_t*, size_t) (root 0 is
//                           drawn first, then root 1)
//   dpf_eval / dcf_eval     one point, Share{y, tau}
//   dpf_eval_all / dcf_eval_all
//                           all 2^n points in index order.  The tree is
//                           expanded breadth first with the 4-way permutation
//                           (AVX2 when available): first the top n - 10
//                           levels, then each remaining subtree of 2^10
//                           leaves in cache-sized buffers; subtrees can be
//                           spread over threads.  Results do not depend on
//                           the split or on the thread count.
//   DpfKey / DcfKey         to_bytes / from_bytes, the canonical encoding of
//                           the reference (all integers little-endian,
//                           W = ceil((k+s)/8)):
//       u32 L                 byte length of the body that follows
//       u8  family            1 = DPF, 2 = DCF
//       u8  b, u8 n, u8 k, u8 s
//       16B root seed s_b
//       per level i = 1..n:
//           16B CWs_i
//           u8  control byte: bit0 = CWtL_i, bit1 = CWtR_i
//           [DCF only] W bytes VCW_i, W bytes TCW_i
//       W bytes CWout_v, W bytes CWout_t
//
// Every argument error throws std::invalid_argument.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "ascon.hpp"
#include "u128.hpp"

namespace couplet {
namespace fss {

typedef ascon::seed_t seed_t;

const uint8_t FAMILY_DPF = 1;
const uint8_t FAMILY_DCF = 2;
const unsigned SEED_BYTES = 16;

// Depth of the subtrees that EvalAll expands in private buffers (a tuning
// knob: any value >= 1 gives the same outputs).
#ifndef COUPLET_EVAL_ALL_CHUNK_BITS
#define COUPLET_EVAL_ALL_CHUNK_BITS 10
#endif
const unsigned EVAL_ALL_CHUNK_BITS = COUPLET_EVAL_ALL_CHUNK_BITS;
// Largest domain EvalAll accepts (2^n ring elements per plane are returned).
const unsigned EVAL_ALL_MAX_N = 40;

// One party's output at a point: a share of the value and of its MAC tag.
struct Share {
    u128 y;    // value plane, in [0, 2^w)
    u128 tau;  // tag plane,   in [0, 2^w)
};

// ---------------------------------------------------------------- helpers
namespace detail {

inline unsigned word_bytes(unsigned width) { return (width + 7) / 8; }

// Parameters this implementation can hold (see the header comment).
inline void check_params(unsigned n, unsigned k, unsigned s) {
    if (n < 1 || n > 64) throw std::invalid_argument("fss: n must be in 1..64");
    if (k < 1 || k > 160 || s > 160 || 2 * (k + s) > 320)
        throw std::invalid_argument("fss: need k >= 1, s >= 0 and 2(k+s) <= 320");
    if (k > 64 || k + s > 128)
        throw std::invalid_argument("fss: this implementation supports k <= 64 and k + s <= 128");
}

inline void check_inputs(uint64_t alpha, uint64_t beta, u128 delta, unsigned n, unsigned k, unsigned s) {
    check_params(n, k, s);
    if (n < 64 && (alpha >> n) != 0) throw std::invalid_argument("fss: alpha out of range");
    if (k < 64 && (beta >> k) != 0) throw std::invalid_argument("fss: beta out of range");
    if (s + 1 < 128 && (delta >> (s + 1)) != u128(0))
        throw std::invalid_argument("fss: delta out of range: need 0 <= Delta < 2^(s+1)");
}

// i-th bit (1-based, MSB first) of the n-bit integer x.
inline unsigned bit_msb(uint64_t x, unsigned n, unsigned i) { return (unsigned)((x >> (n - i)) & 1); }

inline u128 neg_if(unsigned t, u128 v) { return t ? (u128(0) - v) : v; }

inline seed_t xor16(const seed_t& a, const seed_t& b) {
    seed_t r;
    for (unsigned i = 0; i < SEED_BYTES; i++) r[i] = (uint8_t)(a[i] ^ b[i]);
    return r;
}

// Little-endian 64-bit load (one seed word).
inline uint64_t ld64(const uint8_t* p) {
#if defined(_M_X64) || defined(_M_IX86) || defined(_M_ARM64) || \
    (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
#else
    return ascon::load64le(p);
#endif
}

// c if m is all ones, 0 if m is zero (m = 0 - t for a control bit t).
inline u128 sel(u128 c, uint64_t m) { return u128_from(u128_lo(c) & m, u128_hi(c) & m); }

// (value, tag) words of a PRG output Y: bits [0,w) and [w,2w), w <= 128.
inline void conv_words(const uint64_t y[5], unsigned w, u128& val, u128& tag) {
    if (w == 128) {
        val = u128_from(y[0], y[1]);
        tag = u128_from(y[2], y[3]);
    } else if (w == 96) {
        val = u128_from(y[0], y[1] & 0xFFFFFFFFull);
        tag = u128_from((y[1] >> 32) | (y[2] << 32), y[2] >> 32);
    } else {
        val = ascon::extract_bits(y, 0, w);
        tag = ascon::extract_bits(y, w, w);
    }
}

inline void put_word(std::vector<uint8_t>& out, u128 v, unsigned nbytes) {
    uint8_t buf[16];
    u128_store_le(buf, v);
    out.insert(out.end(), buf, buf + nbytes);
}

inline u128 get_word(const uint8_t* p, unsigned nbytes) {
    uint8_t buf[16];
    std::memset(buf, 0, sizeof buf);
    std::memcpy(buf, p, nbytes);
    return u128_load_le(buf);
}

inline size_t key_size(bool dcf, unsigned n, unsigned k, unsigned s) {
    const size_t wb = word_bytes(k + s);
    return 4 + 5 + SEED_BYTES + (size_t)n * (SEED_BYTES + 1 + (dcf ? 2 * wb : 0)) + 2 * wb;
}

}  // namespace detail

// Encoded key length in bytes (including the 4-byte length prefix).
inline size_t dpf_key_size(unsigned n, unsigned k, unsigned s) { return detail::key_size(false, n, k, s); }
inline size_t dcf_key_size(unsigned n, unsigned k, unsigned s) { return detail::key_size(true, n, k, s); }

// Delta = Delta_0 + Delta_1 as an integer (65 bits for s = 64), unreduced.
inline u128 make_delta(uint64_t delta0, uint64_t delta1) { return u128(delta0) + u128(delta1); }

// ---------------------------------------------------------------- keys
struct DcfKey;

// Party b's DPF key: root seed plus the public correction words.
struct DpfKey {
    uint8_t b = 0, n = 0, k = 0, s = 0;
    seed_t seed{};
    std::vector<seed_t> cw_s;       // CWs_i,  i = 1..n at index i-1
    std::vector<uint8_t> cw_tl;     // CWtL_i
    std::vector<uint8_t> cw_tr;     // CWtR_i
    u128 cw_out_v = u128(0);        // CWout_v
    u128 cw_out_t = u128(0);        // CWout_t

    unsigned width() const { return (unsigned)k + s; }
    size_t size_bytes() const { return dpf_key_size(n, k, s); }
    std::vector<uint8_t> to_bytes() const;
    static DpfKey from_bytes(const uint8_t* data, size_t len);
    static DpfKey from_bytes(const std::vector<uint8_t>& data) { return from_bytes(data.data(), data.size()); }

    friend bool operator==(const DpfKey& a, const DpfKey& c) {
        return a.b == c.b && a.n == c.n && a.k == c.k && a.s == c.s && a.seed == c.seed && a.cw_s == c.cw_s &&
               a.cw_tl == c.cw_tl && a.cw_tr == c.cw_tr && a.cw_out_v == c.cw_out_v && a.cw_out_t == c.cw_out_t;
    }
    friend bool operator!=(const DpfKey& a, const DpfKey& c) { return !(a == c); }
};

// Party b's DCF key: as DpfKey plus per-level value / tag correction words.
struct DcfKey {
    uint8_t b = 0, n = 0, k = 0, s = 0;
    seed_t seed{};
    std::vector<seed_t> cw_s;
    std::vector<uint8_t> cw_tl;
    std::vector<uint8_t> cw_tr;
    std::vector<u128> cw_v;         // VCW_i
    std::vector<u128> cw_t;         // TCW_i
    u128 cw_out_v = u128(0);
    u128 cw_out_t = u128(0);

    unsigned width() const { return (unsigned)k + s; }
    size_t size_bytes() const { return dcf_key_size(n, k, s); }
    std::vector<uint8_t> to_bytes() const;
    static DcfKey from_bytes(const uint8_t* data, size_t len);
    static DcfKey from_bytes(const std::vector<uint8_t>& data) { return from_bytes(data.data(), data.size()); }

    friend bool operator==(const DcfKey& a, const DcfKey& c) {
        return a.b == c.b && a.n == c.n && a.k == c.k && a.s == c.s && a.seed == c.seed && a.cw_s == c.cw_s &&
               a.cw_tl == c.cw_tl && a.cw_tr == c.cw_tr && a.cw_v == c.cw_v && a.cw_t == c.cw_t &&
               a.cw_out_v == c.cw_out_v && a.cw_out_t == c.cw_out_t;
    }
    friend bool operator!=(const DcfKey& a, const DcfKey& c) { return !(a == c); }
};

namespace detail {

// Structural validity of a key object (sizes, ranges) before it is used.
template <class Key>
inline void check_key(const Key& key, bool dcf, const std::vector<u128>* cw_v, const std::vector<u128>* cw_t) {
    check_params(key.n, key.k, key.s);
    if (key.b > 1) throw std::invalid_argument("fss: party index must be 0 or 1");
    const size_t n = key.n;
    if (key.cw_s.size() != n || key.cw_tl.size() != n || key.cw_tr.size() != n)
        throw std::invalid_argument("fss: key has the wrong number of correction words");
    if (dcf && (cw_v->size() != n || cw_t->size() != n))
        throw std::invalid_argument("fss: key has the wrong number of value correction words");
    const u128 mask = u128_mask(key.width());
    for (size_t i = 0; i < n; i++) {
        if (key.cw_tl[i] > 1 || key.cw_tr[i] > 1) throw std::invalid_argument("fss: control correction is not a bit");
        if (dcf && (((*cw_v)[i] & mask) != (*cw_v)[i] || ((*cw_t)[i] & mask) != (*cw_t)[i]))
            throw std::invalid_argument("fss: correction word out of range");
    }
    if ((key.cw_out_v & mask) != key.cw_out_v || (key.cw_out_t & mask) != key.cw_out_t)
        throw std::invalid_argument("fss: output correction word out of range");
}

template <class Key>
inline std::vector<uint8_t> encode_key(const Key& key, uint8_t family, const std::vector<u128>* cw_v,
                                       const std::vector<u128>* cw_t) {
    const bool dcf = family == FAMILY_DCF;
    check_key(key, dcf, cw_v, cw_t);
    const unsigned wb = word_bytes(key.width());
    const size_t total = key_size(dcf, key.n, key.k, key.s);
    std::vector<uint8_t> out;
    out.reserve(total);
    const uint32_t body = (uint32_t)(total - 4);
    for (int i = 0; i < 4; i++) out.push_back((uint8_t)(body >> (8 * i)));
    out.push_back(family);
    out.push_back(key.b);
    out.push_back(key.n);
    out.push_back(key.k);
    out.push_back(key.s);
    out.insert(out.end(), key.seed.begin(), key.seed.end());
    for (size_t i = 0; i < key.n; i++) {
        out.insert(out.end(), key.cw_s[i].begin(), key.cw_s[i].end());
        out.push_back((uint8_t)(key.cw_tl[i] | (key.cw_tr[i] << 1)));
        if (dcf) {
            put_word(out, (*cw_v)[i], wb);
            put_word(out, (*cw_t)[i], wb);
        }
    }
    put_word(out, key.cw_out_v, wb);
    put_word(out, key.cw_out_t, wb);
    if (out.size() != total) throw std::logic_error("fss: internal error in key encoding");
    return out;
}

// Decode into `key`; cw_v / cw_t are the DCF level words (null for a DPF).
// Mirrors fss.key_from_bytes of the reference check for check.
template <class Key>
inline void decode_key(const uint8_t* data, size_t len, uint8_t want_family, Key& key, std::vector<u128>* cw_v,
                       std::vector<u128>* cw_t) {
    if (data == nullptr || len < 4) throw std::invalid_argument("fss: truncated key");
    const size_t length = (size_t)data[0] | ((size_t)data[1] << 8) | ((size_t)data[2] << 16) | ((size_t)data[3] << 24);
    if (len - 4 != length) throw std::invalid_argument("fss: length prefix does not match the data length");
    const uint8_t* body = data + 4;
    if (length < 5 + SEED_BYTES) throw std::invalid_argument("fss: truncated key body");
    const uint8_t family = body[0], b = body[1], n = body[2], k = body[3], s = body[4];
    if (family != FAMILY_DPF && family != FAMILY_DCF) throw std::invalid_argument("fss: unknown key family");
    if (family != want_family)
        throw std::invalid_argument(want_family == FAMILY_DPF ? "fss: encoded key is not a DPF key"
                                                              : "fss: encoded key is not a DCF key");
    if (b > 1) throw std::invalid_argument("fss: party index must be 0 or 1");
    check_params(n, k, s);
    const bool dcf = family == FAMILY_DCF;
    const unsigned width = (unsigned)k + s;
    const unsigned wb = word_bytes(width);
    const u128 mask = u128_mask(width);
    if (len != key_size(dcf, n, k, s)) throw std::invalid_argument("fss: key body has the wrong length for its parameters");
    key.b = b; key.n = n; key.k = k; key.s = s;
    size_t pos = 5;
    std::memcpy(key.seed.data(), body + pos, SEED_BYTES);
    pos += SEED_BYTES;
    key.cw_s.resize(n);
    key.cw_tl.resize(n);
    key.cw_tr.resize(n);
    if (dcf) {
        cw_v->resize(n);
        cw_t->resize(n);
    }
    for (size_t i = 0; i < n; i++) {
        std::memcpy(key.cw_s[i].data(), body + pos, SEED_BYTES);
        pos += SEED_BYTES;
        const uint8_t ctrl = body[pos++];
        if (ctrl & ~3u) throw std::invalid_argument("fss: control byte has stray bits");
        key.cw_tl[i] = (uint8_t)(ctrl & 1);
        key.cw_tr[i] = (uint8_t)((ctrl >> 1) & 1);
        if (dcf) {
            const u128 v = get_word(body + pos, wb), t = get_word(body + pos + wb, wb);
            pos += 2 * (size_t)wb;
            if ((v & mask) != v || (t & mask) != t) throw std::invalid_argument("fss: correction word out of range");
            (*cw_v)[i] = v;
            (*cw_t)[i] = t;
        }
    }
    key.cw_out_v = get_word(body + pos, wb);
    key.cw_out_t = get_word(body + pos + wb, wb);
    if ((key.cw_out_v & mask) != key.cw_out_v || (key.cw_out_t & mask) != key.cw_out_t)
        throw std::invalid_argument("fss: output correction word out of range");
}

}  // namespace detail

inline std::vector<uint8_t> DpfKey::to_bytes() const { return detail::encode_key(*this, FAMILY_DPF, nullptr, nullptr); }
inline std::vector<uint8_t> DcfKey::to_bytes() const { return detail::encode_key(*this, FAMILY_DCF, &cw_v, &cw_t); }

inline DpfKey DpfKey::from_bytes(const uint8_t* data, size_t len) {
    DpfKey key;
    detail::decode_key(data, len, FAMILY_DPF, key, nullptr, nullptr);
    return key;
}
inline DcfKey DcfKey::from_bytes(const uint8_t* data, size_t len) {
    DcfKey key;
    detail::decode_key(data, len, FAMILY_DCF, key, &key.cw_v, &key.cw_t);
    return key;
}

// Family byte of an encoded key (1 = DPF, 2 = DCF) for dispatching.
inline uint8_t encoded_key_family(const uint8_t* data, size_t len) {
    if (data == nullptr || len < 5) throw std::invalid_argument("fss: truncated key");
    if (data[4] != FAMILY_DPF && data[4] != FAMILY_DCF) throw std::invalid_argument("fss: unknown key family");
    return data[4];
}
inline uint8_t encoded_key_family(const std::vector<uint8_t>& data) { return encoded_key_family(data.data(), data.size()); }

// ---------------------------------------------------------------- DPF
// DPF.Gen(alpha, beta, Delta; s0_root, s1_root): keys for
// f(x) = (beta, Delta*beta mod 2^w) * [x == alpha], Delta in [0, 2^(s+1)).
inline std::pair<DpfKey, DpfKey> dpf_gen(uint64_t alpha, uint64_t beta, u128 delta, unsigned n, unsigned k, unsigned s,
                                         const seed_t& root0, const seed_t& root1) {
    detail::check_inputs(alpha, beta, delta, n, k, s);
    const unsigned w = k + s;
    const u128 mask = u128_mask(w);
    const u128 beta_v = u128(beta);
    const u128 beta_t = (delta * beta_v) & mask;
    seed_t s0 = root0, s1 = root1;
    unsigned t0 = 0, t1 = 1;
    std::vector<seed_t> cw_s(n);
    std::vector<uint8_t> cw_tl(n), cw_tr(n);
    for (unsigned i = 1; i <= n; i++) {
        const unsigned a = detail::bit_msb(alpha, n, i);
        const ascon::PrgCall calls[2] = {{s0.data(), &ascon::IV_G()}, {s1.data(), &ascon::IV_G()}};
        uint64_t y[2][5];
        ascon::prg_state_batch(calls, 2, y);
        const ascon::GOut g0 = ascon::G_from_state(y[0]), g1 = ascon::G_from_state(y[1]);
        const seed_t& keep0 = a ? g0.sR : g0.sL;
        const seed_t& keep1 = a ? g1.sR : g1.sL;
        const seed_t& lose0 = a ? g0.sL : g0.sR;
        const seed_t& lose1 = a ? g1.sL : g1.sR;
        const unsigned tk0 = a ? g0.tR : g0.tL, tk1 = a ? g1.tR : g1.tL;
        const seed_t cws = detail::xor16(lose0, lose1);
        const unsigned ctl = (unsigned)(g0.tL ^ g1.tL) ^ a ^ 1u;
        const unsigned ctr = (unsigned)(g0.tR ^ g1.tR) ^ a;
        const unsigned ct_keep = a ? ctr : ctl;
        cw_s[i - 1] = cws;
        cw_tl[i - 1] = (uint8_t)ctl;
        cw_tr[i - 1] = (uint8_t)ctr;
        s0 = t0 ? detail::xor16(keep0, cws) : keep0;
        s1 = t1 ? detail::xor16(keep1, cws) : keep1;
        t0 = tk0 ^ (t0 & ct_keep);
        t1 = tk1 ^ (t1 & ct_keep);
    }
    const ascon::ConvOut c0 = ascon::Conv(s0, k, s), c1 = ascon::Conv(s1, k, s);
    DpfKey k0;
    k0.n = (uint8_t)n; k0.k = (uint8_t)k; k0.s = (uint8_t)s;
    k0.cw_s = std::move(cw_s);
    k0.cw_tl = std::move(cw_tl);
    k0.cw_tr = std::move(cw_tr);
    k0.cw_out_v = detail::neg_if(t1, beta_v - c0.val + c1.val) & mask;
    k0.cw_out_t = detail::neg_if(t1, beta_t - c0.tag + c1.tag) & mask;
    DpfKey k1 = k0;
    k0.b = 0; k0.seed = root0;
    k1.b = 1; k1.seed = root1;
    return std::make_pair(std::move(k0), std::move(k1));
}

// As above with fresh roots: rng.fill(uint8_t*, size_t) is called for the
// 16 bytes of root 0, then for the 16 bytes of root 1.
template <class Rng>
inline std::pair<DpfKey, DpfKey> dpf_gen(uint64_t alpha, uint64_t beta, u128 delta, unsigned n, unsigned k, unsigned s,
                                         Rng& rng) {
    detail::check_inputs(alpha, beta, delta, n, k, s);
    seed_t root0, root1;
    rng.fill(root0.data(), SEED_BYTES);
    rng.fill(root1.data(), SEED_BYTES);
    return dpf_gen(alpha, beta, delta, n, k, s, root0, root1);
}

// DPF.Eval(b, k_b, x) -> (y_b, tau_b), both mod 2^(k+s); b is key.b.
inline Share dpf_eval(const DpfKey& key, uint64_t x) {
    detail::check_key(key, false, nullptr, nullptr);
    const unsigned n = key.n;
    if (n < 64 && (x >> n) != 0) throw std::invalid_argument("fss: x out of range");
    const unsigned w = key.width();
    const u128 mask = u128_mask(w);
    const ascon::IV& iv_g = ascon::IV_G();
    // The path is followed on seed words; selections by x_i and by the
    // control bit are done with masks (no data-dependent branches).
    uint64_t lo = detail::ld64(key.seed.data()), hi = detail::ld64(key.seed.data() + 8);
    uint64_t t = key.b;
    uint64_t y[5];
    for (unsigned i = 1; i <= n; i++) {
        ascon::prg_state_words(y, lo, hi, iv_g);  // Y = F_G(s): sL = y0,y1; sR = y2,y3; tL, tR = bits 0, 1 of y4
        const uint64_t side = detail::bit_msb(x, n, i);
        const uint64_t sm = (uint64_t)0 - side, tm = (uint64_t)0 - t;
        const uint64_t cl = key.cw_tl[i - 1], cr = key.cw_tr[i - 1];
        const uint64_t tn = (y[4] >> side) & 1;
        const uint64_t ct = cl ^ ((cl ^ cr) & side);
        lo = (y[0] ^ ((y[0] ^ y[2]) & sm)) ^ (detail::ld64(key.cw_s[i - 1].data()) & tm);
        hi = (y[1] ^ ((y[1] ^ y[3]) & sm)) ^ (detail::ld64(key.cw_s[i - 1].data() + 8) & tm);
        t = tn ^ (t & ct);
    }
    ascon::prg_state_words(y, lo, hi, ascon::IV_C());
    u128 cv, ct;
    detail::conv_words(y, w, cv, ct);
    const uint64_t tm = (uint64_t)0 - t;
    cv = cv + detail::sel(key.cw_out_v, tm);
    ct = ct + detail::sel(key.cw_out_t, tm);
    Share out;
    out.y = detail::neg_if(key.b, cv) & mask;
    out.tau = detail::neg_if(key.b, ct) & mask;
    return out;
}

// Same with the party index stated explicitly; it must match the key.
inline Share dpf_eval(unsigned b, const DpfKey& key, uint64_t x) {
    if (b > 1 || key.b != b) throw std::invalid_argument("fss: party index does not match the key");
    return dpf_eval(key, x);
}

// ---------------------------------------------------------------- DCF
// DCF.Gen(alpha, beta, Delta; roots): keys for
// f(x) = (beta, Delta*beta mod 2^w) * [x < alpha], Delta in [0, 2^(s+1)).
inline std::pair<DcfKey, DcfKey> dcf_gen(uint64_t alpha, uint64_t beta, u128 delta, unsigned n, unsigned k, unsigned s,
                                         const seed_t& root0, const seed_t& root1) {
    detail::check_inputs(alpha, beta, delta, n, k, s);
    const unsigned w = k + s;
    const u128 mask = u128_mask(w);
    const u128 beta_v = u128(beta);
    const u128 beta_t = (delta * beta_v) & mask;
    seed_t s0 = root0, s1 = root1;
    unsigned t0 = 0, t1 = 1;
    u128 va = u128(0), ta = u128(0);
    std::vector<seed_t> cw_s(n);
    std::vector<uint8_t> cw_tl(n), cw_tr(n);
    std::vector<u128> cw_v(n), cw_t(n);
    for (unsigned i = 1; i <= n; i++) {
        const unsigned a = detail::bit_msb(alpha, n, i);
        // G, ConvV_L, ConvV_R on both seeds: six permutations in lock step.
        const ascon::PrgCall calls[6] = {{s0.data(), &ascon::IV_G()},  {s1.data(), &ascon::IV_G()},
                                         {s0.data(), &ascon::IV_VL()}, {s1.data(), &ascon::IV_VL()},
                                         {s0.data(), &ascon::IV_VR()}, {s1.data(), &ascon::IV_VR()}};
        uint64_t y[6][5];
        ascon::prg_state_batch(calls, 6, y);
        const ascon::GOut g0 = ascon::G_from_state(y[0]), g1 = ascon::G_from_state(y[1]);
        const ascon::ConvOut l0 = ascon::conv_from_state(y[2], w), l1 = ascon::conv_from_state(y[3], w);
        const ascon::ConvOut r0 = ascon::conv_from_state(y[4], w), r1 = ascon::conv_from_state(y[5], w);
        const seed_t& keep0 = a ? g0.sR : g0.sL;
        const seed_t& keep1 = a ? g1.sR : g1.sL;
        const seed_t& lose0 = a ? g0.sL : g0.sR;
        const seed_t& lose1 = a ? g1.sL : g1.sR;
        const unsigned tk0 = a ? g0.tR : g0.tL, tk1 = a ? g1.tR : g1.tL;
        const ascon::ConvOut& vkeep0 = a ? r0 : l0;
        const ascon::ConvOut& vkeep1 = a ? r1 : l1;
        const ascon::ConvOut& vlose0 = a ? l0 : r0;
        const ascon::ConvOut& vlose1 = a ? l1 : r1;
        const seed_t cws = detail::xor16(lose0, lose1);
        const unsigned ctl = (unsigned)(g0.tL ^ g1.tL) ^ a ^ 1u;
        const unsigned ctr = (unsigned)(g0.tR ^ g1.tR) ^ a;
        const unsigned ct_keep = a ? ctr : ctl;
        // t1 is the control bit BEFORE this level's update.
        const u128 vcw = detail::neg_if(t1, vlose1.val - vlose0.val - va + (a ? beta_v : u128(0))) & mask;
        const u128 tcw = detail::neg_if(t1, vlose1.tag - vlose0.tag - ta + (a ? beta_t : u128(0))) & mask;
        va = (va - vkeep1.val + vkeep0.val + detail::neg_if(t1, vcw)) & mask;
        ta = (ta - vkeep1.tag + vkeep0.tag + detail::neg_if(t1, tcw)) & mask;
        cw_s[i - 1] = cws;
        cw_tl[i - 1] = (uint8_t)ctl;
        cw_tr[i - 1] = (uint8_t)ctr;
        cw_v[i - 1] = vcw;
        cw_t[i - 1] = tcw;
        s0 = t0 ? detail::xor16(keep0, cws) : keep0;
        s1 = t1 ? detail::xor16(keep1, cws) : keep1;
        t0 = tk0 ^ (t0 & ct_keep);
        t1 = tk1 ^ (t1 & ct_keep);
    }
    const ascon::ConvOut c0 = ascon::Conv(s0, k, s), c1 = ascon::Conv(s1, k, s);
    DcfKey k0;
    k0.n = (uint8_t)n; k0.k = (uint8_t)k; k0.s = (uint8_t)s;
    k0.cw_s = std::move(cw_s);
    k0.cw_tl = std::move(cw_tl);
    k0.cw_tr = std::move(cw_tr);
    k0.cw_v = std::move(cw_v);
    k0.cw_t = std::move(cw_t);
    k0.cw_out_v = detail::neg_if(t1, c1.val - c0.val - va) & mask;
    k0.cw_out_t = detail::neg_if(t1, c1.tag - c0.tag - ta) & mask;
    DcfKey k1 = k0;
    k0.b = 0; k0.seed = root0;
    k1.b = 1; k1.seed = root1;
    return std::make_pair(std::move(k0), std::move(k1));
}

template <class Rng>
inline std::pair<DcfKey, DcfKey> dcf_gen(uint64_t alpha, uint64_t beta, u128 delta, unsigned n, unsigned k, unsigned s,
                                         Rng& rng) {
    detail::check_inputs(alpha, beta, delta, n, k, s);
    seed_t root0, root1;
    rng.fill(root0.data(), SEED_BYTES);
    rng.fill(root1.data(), SEED_BYTES);
    return dcf_gen(alpha, beta, delta, n, k, s, root0, root1);
}

// DCF.Eval(b, k_b, x) -> (V_b, T_b), both mod 2^(k+s); b is key.b.  At every
// level ConvV_{x_i} is evaluated on the seed and control bit BEFORE that
// level's update.
inline Share dcf_eval(const DcfKey& key, uint64_t x) {
    detail::check_key(key, true, &key.cw_v, &key.cw_t);
    const unsigned n = key.n, w = key.width();
    if (n < 64 && (x >> n) != 0) throw std::invalid_argument("fss: x out of range");
    const u128 mask = u128_mask(w);
    const ascon::IV& iv_g = ascon::IV_G();
    const ascon::IV* const iv_side[2] = {&ascon::IV_VL(), &ascon::IV_VR()};
    uint64_t lo = detail::ld64(key.seed.data()), hi = detail::ld64(key.seed.data() + 8);
    uint64_t t = key.b;
    u128 value = u128(0), tag = u128(0);  // accumulated before the sign (-1)^b
    u128 cv, cu;
    for (unsigned i = 1; i <= n; i++) {
        const uint64_t side = detail::bit_msb(x, n, i);
        // G(s) and ConvV_{x_i}(s) on the same pre-update seed, in lock step.
        uint64_t y[2][5];
        ascon::prg_state_words_x2(y, lo, hi, iv_g, lo, hi, *iv_side[side]);
        const uint64_t sm = (uint64_t)0 - side, tm = (uint64_t)0 - t;
        detail::conv_words(y[1], w, cv, cu);
        value = value + cv + detail::sel(key.cw_v[i - 1], tm);
        tag = tag + cu + detail::sel(key.cw_t[i - 1], tm);
        const uint64_t cl = key.cw_tl[i - 1], cr = key.cw_tr[i - 1];
        const uint64_t tn = (y[0][4] >> side) & 1;
        const uint64_t ct = cl ^ ((cl ^ cr) & side);
        lo = (y[0][0] ^ ((y[0][0] ^ y[0][2]) & sm)) ^ (detail::ld64(key.cw_s[i - 1].data()) & tm);
        hi = (y[0][1] ^ ((y[0][1] ^ y[0][3]) & sm)) ^ (detail::ld64(key.cw_s[i - 1].data() + 8) & tm);
        t = tn ^ (t & ct);
    }
    uint64_t yc[5];
    ascon::prg_state_words(yc, lo, hi, ascon::IV_C());
    detail::conv_words(yc, w, cv, cu);
    const uint64_t tm = (uint64_t)0 - t;
    value = value + cv + detail::sel(key.cw_out_v, tm);
    tag = tag + cu + detail::sel(key.cw_out_t, tm);
    Share out;
    out.y = detail::neg_if(key.b, value) & mask;
    out.tau = detail::neg_if(key.b, tag) & mask;
    return out;
}

inline Share dcf_eval(unsigned b, const DcfKey& key, uint64_t x) {
    if (b > 1 || key.b != b) throw std::invalid_argument("fss: party index does not match the key");
    return dcf_eval(key, x);
}

// ---------------------------------------------------------------- EvalAll
namespace detail {

// One tree level in structure-of-arrays form.  Buffers hold at least eight
// nodes so that a level of one or two nodes can be processed as a full
// group of four (the surplus lanes are computed and never read).
struct Nodes {
    std::vector<uint64_t> lo, hi;  // seed words x0, x1
    std::vector<uint64_t> tm;      // control bit as a mask: 0 or all ones
    std::vector<u128> v, u;        // DCF: partial sums of both planes, already signed by (-1)^b
    void resize(size_t m, bool dcf) {
        const size_t cap = m < 8 ? 8 : m;
        lo.resize(cap);
        hi.resize(cap);
        tm.resize(cap);
        if (dcf) {
            v.resize(cap);
            u.resize(cap);
        }
    }
};

// Correction words of one level.
struct LevelCW {
    uint64_t s_lo, s_hi;
    uint8_t tl, tr;
    u128 v, u;  // DCF only
};

// Everything EvalAll needs from a key.
struct TreeKey {
    unsigned n, w;
    bool dcf, neg;  // neg: party 1 negates
    uint64_t root_lo, root_hi;
    std::vector<LevelCW> levels;
    u128 out_v, out_t;
};

#if COUPLET_HAVE_AVX2
// dst[0..7] = l0 r0 l1 r1 l2 r2 l3 r3: children in index order.
inline void store_interleaved(uint64_t* dst, __m256i l, __m256i r) {
    const __m256i a = _mm256_unpacklo_epi64(l, r);  // l0 r0 l2 r2
    const __m256i b = _mm256_unpackhi_epi64(l, r);  // l1 r1 l3 r3
    _mm256_storeu_si256((__m256i*)dst, _mm256_permute2x128_si256(a, b, 0x20));
    _mm256_storeu_si256((__m256i*)(dst + 4), _mm256_permute2x128_si256(a, b, 0x31));
}
#endif

// Children of the m nodes of `in`, in index order (left child of node j at
// 2j, right child at 2j+1), with the level's correction applied.  m is a
// power of two; nodes are processed in groups of four.
inline void expand_level(const TreeKey& key, const LevelCW& cw, const Nodes& in, size_t m, Nodes& out) {
    const ascon::IV& iv_g = ascon::IV_G();
    const ascon::IV& iv_l = ascon::IV_VL();
    const ascon::IV& iv_r = ascon::IV_VR();
    const uint64_t tl_m = (uint64_t)0 - (uint64_t)cw.tl, tr_m = (uint64_t)0 - (uint64_t)cw.tr;
    const uint64_t* in_lo = in.lo.data();
    const uint64_t* in_hi = in.hi.data();
    const uint64_t* in_tm = in.tm.data();
    uint64_t* out_lo = out.lo.data();
    uint64_t* out_hi = out.hi.data();
    uint64_t* out_tm = out.tm.data();
#if COUPLET_HAVE_AVX2
    const __m256i cw_lo = _mm256_set1_epi64x((long long)cw.s_lo), cw_hi = _mm256_set1_epi64x((long long)cw.s_hi);
    const __m256i cw_tl = _mm256_set1_epi64x((long long)tl_m), cw_tr = _mm256_set1_epi64x((long long)tr_m);
    const __m256i one = _mm256_set1_epi64x(1), zero = _mm256_setzero_si256();
#endif
    for (size_t j = 0; j < m; j += 4) {
        const size_t cnt = (m - j < 4) ? (m - j) : 4;
        // G: s' = s_side XOR (t ? CWs : 0), t' = t_side XOR (t AND CWt_side).
#if COUPLET_HAVE_AVX2
        const __m256i s_lo = _mm256_loadu_si256((const __m256i*)(in_lo + j));
        const __m256i s_hi = _mm256_loadu_si256((const __m256i*)(in_hi + j));
        const __m256i tm = _mm256_loadu_si256((const __m256i*)(in_tm + j));
        __m256i y0, y1, y2, y3, y4;
        ascon::prg_state_x4_regs(s_lo, s_hi, iv_g, y0, y1, y2, y3, y4);
        const __m256i c_lo = _mm256_and_si256(tm, cw_lo), c_hi = _mm256_and_si256(tm, cw_hi);
        store_interleaved(out_lo + 2 * j, _mm256_xor_si256(y0, c_lo), _mm256_xor_si256(y2, c_lo));
        store_interleaved(out_hi + 2 * j, _mm256_xor_si256(y1, c_hi), _mm256_xor_si256(y3, c_hi));
        const __m256i tl = _mm256_sub_epi64(zero, _mm256_and_si256(y4, one));                        // mask of t_L
        const __m256i tr = _mm256_sub_epi64(zero, _mm256_and_si256(_mm256_srli_epi64(y4, 1), one));  // mask of t_R
        store_interleaved(out_tm + 2 * j, _mm256_xor_si256(tl, _mm256_and_si256(tm, cw_tl)),
                          _mm256_xor_si256(tr, _mm256_and_si256(tm, cw_tr)));
#else
        uint64_t y[5][4];
        ascon::prg_state_x4_soa(in_lo + j, in_hi + j, iv_g, y);
        for (size_t q = 0; q < 4; q++) {
            const size_t p = j + q;
            const uint64_t tm = in_tm[p];
            const uint64_t c_lo = tm & cw.s_lo, c_hi = tm & cw.s_hi;
            out_lo[2 * p] = y[0][q] ^ c_lo;
            out_hi[2 * p] = y[1][q] ^ c_hi;
            out_lo[2 * p + 1] = y[2][q] ^ c_lo;
            out_hi[2 * p + 1] = y[3][q] ^ c_hi;
            out_tm[2 * p] = ((uint64_t)0 - (y[4][q] & 1)) ^ (tm & tl_m);
            out_tm[2 * p + 1] = ((uint64_t)0 - ((y[4][q] >> 1) & 1)) ^ (tm & tr_m);
        }
#endif
        if (key.dcf) {
            // ConvV_L / ConvV_R on the seed and control bit BEFORE the update:
            // child sum = parent sum + (-1)^b (v_side + t VCW), same for the tags.
            // Sums are kept mod 2^128 and reduced mod 2^w at the leaves.
            uint64_t yl[5][4], yr[5][4];
            ascon::prg_state_x4_soa(in_lo + j, in_hi + j, iv_l, yl);
            ascon::prg_state_x4_soa(in_lo + j, in_hi + j, iv_r, yr);
            for (size_t q = 0; q < cnt; q++) {
                const size_t p = j + q;
                const uint64_t sl[5] = {yl[0][q], yl[1][q], yl[2][q], yl[3][q], yl[4][q]};
                const uint64_t sr[5] = {yr[0][q], yr[1][q], yr[2][q], yr[3][q], yr[4][q]};
                u128 vl, ul, vr, ur;
                conv_words(sl, key.w, vl, ul);
                conv_words(sr, key.w, vr, ur);
                const u128 cv = sel(cw.v, in_tm[p]), cu = sel(cw.u, in_tm[p]);
                if (key.neg) {
                    const u128 bv = in.v[p] - cv, bu = in.u[p] - cu;
                    out.v[2 * p] = bv - vl;
                    out.v[2 * p + 1] = bv - vr;
                    out.u[2 * p] = bu - ul;
                    out.u[2 * p + 1] = bu - ur;
                } else {
                    const u128 bv = in.v[p] + cv, bu = in.u[p] + cu;
                    out.v[2 * p] = bv + vl;
                    out.v[2 * p + 1] = bv + vr;
                    out.u[2 * p] = bu + ul;
                    out.u[2 * p + 1] = bu + ur;
                }
            }
        }
    }
}

// Leaf outputs of the m nodes of `in`:
// (-1)^b (Conv(s) + t CWout), plus the accumulated sums for a DCF.
inline void leaf_level(const TreeKey& key, const Nodes& in, size_t m, u128* y_out, u128* tau_out) {
    const ascon::IV& iv_c = ascon::IV_C();
    const u128 mask = u128_mask(key.w);
    const uint64_t* in_lo = in.lo.data();
    const uint64_t* in_hi = in.hi.data();
    const uint64_t* in_tm = in.tm.data();
    uint64_t y[5][4];
    for (size_t j = 0; j < m; j += 4) {
        const size_t cnt = (m - j < 4) ? (m - j) : 4;
        ascon::prg_state_x4_soa(in_lo + j, in_hi + j, iv_c, y);
        for (size_t q = 0; q < cnt; q++) {
            const size_t p = j + q;
            const uint64_t st[5] = {y[0][q], y[1][q], y[2][q], y[3][q], y[4][q]};
            u128 cv, ct;
            conv_words(st, key.w, cv, ct);
            cv = cv + sel(key.out_v, in_tm[p]);
            ct = ct + sel(key.out_t, in_tm[p]);
            if (key.neg) {
                cv = u128(0) - cv;
                ct = u128(0) - ct;
            }
            if (key.dcf) {
                cv = cv + in.v[p];
                ct = ct + in.u[p];
            }
            y_out[p] = cv & mask;
            tau_out[p] = ct & mask;
        }
    }
}

// Run work(r0, r1) over the index range [0, n), cut into `threads` contiguous
// parts, one std::thread per part (the calling thread only waits).  With one
// part the work runs in the calling thread.  An exception inside a worker
// (e.g. std::bad_alloc from its buffers) must not leave the thread function,
// which would end the process: every worker keeps its exception, and the
// first one (in the order of the parts) is rethrown in the calling thread
// after ALL workers have been joined.
template <class Work>
inline void run_parts(size_t n, size_t threads, Work&& work) {
    size_t parts = threads < 1 ? 1 : threads;
    if (parts > n) parts = n;
    if (parts <= 1) {
        work((size_t)0, n);
        return;
    }
    std::vector<std::exception_ptr> errors(parts);
    auto guarded = [&work, &errors](size_t part, size_t r0, size_t r1) {
        try {
            work(r0, r1);
        } catch (...) {
            errors[part] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(parts);
    try {
        for (size_t i = 0; i < parts; i++) pool.emplace_back(guarded, i, n * i / parts, n * (i + 1) / parts);
    } catch (...) {  // a thread could not be started: wait for the others, then report
        for (std::thread& th : pool) th.join();
        throw;
    }
    for (std::thread& th : pool) th.join();
    for (const std::exception_ptr& e : errors)
        if (e) std::rethrow_exception(e);
}

inline void eval_all_tree(const TreeKey& key, u128* y, u128* tau, unsigned threads) {
    const unsigned n = key.n;
    if (n > EVAL_ALL_MAX_N) throw std::invalid_argument("fss: domain too large for EvalAll");
    if (y == nullptr || tau == nullptr) throw std::invalid_argument("fss: EvalAll needs two output buffers");
    const unsigned bot = n < EVAL_ALL_CHUNK_BITS ? n : EVAL_ALL_CHUNK_BITS;  // levels inside a subtree
    const unsigned top = n - bot;                                            // levels above the subtrees
    const size_t n_sub = (size_t)1 << top, sub_size = (size_t)1 << bot;

    // Phase 1: the top levels, breadth first from the root.
    Nodes roots, spare;
    roots.resize(n_sub, key.dcf);
    spare.resize(n_sub, key.dcf);
    roots.lo[0] = key.root_lo;
    roots.hi[0] = key.root_hi;
    roots.tm[0] = key.neg ? ~(uint64_t)0 : (uint64_t)0;  // t = b at the root
    if (key.dcf) {
        roots.v[0] = u128(0);
        roots.u[0] = u128(0);
    }
    size_t m = 1;
    for (unsigned i = 0; i < top; i++, m *= 2) {
        expand_level(key, key.levels[i], roots, m, spare);
        std::swap(roots, spare);
    }

    // Phase 2: every subtree, breadth first, in buffers of 2^bot nodes.
    auto work = [&key, &roots, y, tau, top, bot, sub_size](size_t r0, size_t r1) {
        Nodes cur, nxt;
        cur.resize(sub_size, key.dcf);
        nxt.resize(sub_size, key.dcf);
        for (size_t r = r0; r < r1; r++) {
            cur.lo[0] = roots.lo[r];
            cur.hi[0] = roots.hi[r];
            cur.tm[0] = roots.tm[r];
            if (key.dcf) {
                cur.v[0] = roots.v[r];
                cur.u[0] = roots.u[r];
            }
            size_t mm = 1;
            for (unsigned i = 0; i < bot; i++, mm *= 2) {
                expand_level(key, key.levels[top + i], cur, mm, nxt);
                std::swap(cur, nxt);
            }
            leaf_level(key, cur, mm, y + r * sub_size, tau + r * sub_size);
        }
    };
    run_parts(n_sub, threads, work);  // exceptions of the workers are rethrown here
}

template <class Key>
inline TreeKey tree_key(const Key& key, bool dcf, const std::vector<u128>* cw_v, const std::vector<u128>* cw_t) {
    check_key(key, dcf, cw_v, cw_t);
    TreeKey tk;
    tk.n = key.n;
    tk.w = key.width();
    tk.dcf = dcf;
    tk.neg = key.b != 0;
    tk.root_lo = ld64(key.seed.data());
    tk.root_hi = ld64(key.seed.data() + 8);
    tk.levels.resize(key.n);
    for (size_t i = 0; i < key.n; i++) {
        LevelCW& cw = tk.levels[i];
        cw.s_lo = ld64(key.cw_s[i].data());
        cw.s_hi = ld64(key.cw_s[i].data() + 8);
        cw.tl = key.cw_tl[i];
        cw.tr = key.cw_tr[i];
        cw.v = dcf ? (*cw_v)[i] : u128(0);
        cw.u = dcf ? (*cw_t)[i] : u128(0);
    }
    tk.out_v = key.cw_out_v;
    tk.out_t = key.cw_out_t;
    return tk;
}

}  // namespace detail

// DPF.EvalAll(b, k_b): y[x], tau[x] for x = 0 .. 2^n - 1, each mod 2^(k+s).
// The buffers must hold 2^n elements each.
inline void dpf_eval_all(const DpfKey& key, u128* y, u128* tau, unsigned threads = 1) {
    detail::eval_all_tree(detail::tree_key(key, false, nullptr, nullptr), y, tau, threads);
}
inline void dpf_eval_all(const DpfKey& key, std::vector<u128>& y, std::vector<u128>& tau, unsigned threads = 1) {
    const detail::TreeKey tk = detail::tree_key(key, false, nullptr, nullptr);
    if (tk.n > EVAL_ALL_MAX_N) throw std::invalid_argument("fss: domain too large for EvalAll");
    y.resize((size_t)1 << tk.n);
    tau.resize((size_t)1 << tk.n);
    detail::eval_all_tree(tk, y.data(), tau.data(), threads);
}

// DCF.EvalAll(b, k_b), same conventions.
inline void dcf_eval_all(const DcfKey& key, u128* y, u128* tau, unsigned threads = 1) {
    detail::eval_all_tree(detail::tree_key(key, true, &key.cw_v, &key.cw_t), y, tau, threads);
}
inline void dcf_eval_all(const DcfKey& key, std::vector<u128>& y, std::vector<u128>& tau, unsigned threads = 1) {
    const detail::TreeKey tk = detail::tree_key(key, true, &key.cw_v, &key.cw_t);
    if (tk.n > EVAL_ALL_MAX_N) throw std::invalid_argument("fss: domain too large for EvalAll");
    y.resize((size_t)1 << tk.n);
    tau.resize((size_t)1 << tk.n);
    detail::eval_all_tree(tk, y.data(), tau.data(), threads);
}

// Byte encoding of an EvalAll vector as used by the golden vectors: the
// values in index order, each W = ceil(width/8) bytes little-endian.
inline std::vector<uint8_t> encode_vector(const u128* values, size_t count, unsigned width) {
    const unsigned wb = detail::word_bytes(width);
    std::vector<uint8_t> out(count * wb);
    uint8_t buf[16];
    for (size_t i = 0; i < count; i++) {
        u128_store_le(buf, values[i]);
        std::memcpy(out.data() + i * wb, buf, wb);
    }
    return out;
}
inline std::vector<uint8_t> encode_vector(const std::vector<u128>& values, unsigned width) {
    return encode_vector(values.data(), values.size(), width);
}

}  // namespace fss
}  // namespace couplet
