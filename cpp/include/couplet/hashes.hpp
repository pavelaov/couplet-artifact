// couplet/hashes.hpp -- symmetric helpers of the TinyOT-style backend
// (DESIGN.md Sections 2.1, 2.4, 7.1, 7.2, 7.4): the correlation-robust hash
// H_ccr, the counter-mode PRG ctr_prg, length-framed SHA-256 with its
// counter mode and key derivation, commitments, the simultaneous
// commit-then-open exchange, coin tossing, and the ProtocolAbort exception.
//
// The definitions are the ones of the Python module couplet/hashes.py, bit
// for bit (the test suite checks vectors produced by that module); they are
// stated completely here.  Conventions: a 16-byte string maps to two 64-bit
// words little-endian (bytes 0..7 -> lo, bytes 8..15 -> hi), which is also
// the gf128 layout; p12 is the 12-round Ascon permutation on the words
// (x0, x1, x2, x3, x4); le64(v) is v as 8 little-endian bytes.
//
// (a) H_ccr(x, tweak, dom) -> 128 bits
//         (y0, y1, ., ., .) = p12(x_lo, x_hi, tweak, dom, 0);  output (y0, y1).
//     A truncated permutation without feed-forward: 192 of the 320 output
//     bits are discarded, so in the random permutation model it is a random
//     function of (x, tweak, dom).  It is the hash H(tweak, X) of DESIGN.md
//     2.1 with a 64-bit tweak in x2, a 64-bit domain constant in x3 and
//     x4 = 0.  H_ccr_batch evaluates n inputs with the 4-way permutation.
//
// (b) ctr_prg(seed, dom): an unbounded stream of 64-bit words.
//         block i (i = 0, 1, ...) = (y0, y1, y2) of p12(seed_lo, seed_hi, i, dom, 0)
//     stream word j = word (j mod 3) of block floor(j / 3)  (rate 192 bits,
//     capacity 128 bits; this is the PRG of DESIGN.md 7.2).  Views:
//         bytes : each word as 8 little-endian bytes, concatenated;
//         bits  : bit i = bit (i mod 64) of word floor(i / 64), bit 0 = lsb;
//         gf128 : element j = (lo = word 2j, hi = word 2j + 1).
//
//     Rules for callers of (a) and (b).
//       * H_ccr(x, i, dom) is by construction the first two words of block i
//         of ctr_prg(x, dom): one secret must never be used as a hash input
//         and as a PRG seed under the same dom.  In this code base domains
//         for H_ccr carry 0x48 ('H') in the top byte and domains for ctr_prg
//         carry 0x50 ('P') (namespace dom).
//       * A (seed, dom) pair determines one stream.
//       * The four dom values equal to the x3 word of the FSS PRG labels
//         Couplet-G / -C / -VL / -VR (0x47, 0x43, 0x4c56, 0x5256)
//         are rejected with std::invalid_argument, which keeps every call
//         disjoint from the inputs s || IV of the FSS PRG of DESIGN.md 2.1.
//
// (c) Framed SHA-256.  For byte strings part_1, ..., part_m
//         frame(part_1, ..., part_m) = le64(len(part_1)) || part_1 || ... ||
//                                      le64(len(part_m)) || part_m
//         framed_sha256(part_1, ..., part_m) = SHA-256(frame(part_1, ..., part_m)).
//     The framing is injective on tuples of byte strings.
//         sha256_expand(n, parts...) = first n bytes of B_0 || B_1 || ...,
//                                      B_i = framed_sha256(parts..., le64(i));
//         kdf(parts...)              = first 16 bytes of
//                                      framed_sha256("Couplet-kdf", parts...).
//
//     Commitment (DESIGN.md 2.4) to msg with a 32-byte uniformly random
//     opening omega, bound to a session string sid, a label and optionally
//     to the committing party:
//         com = framed_sha256("Couplet-com", sid, label, party_tag, omega, msg)
//     with party_tag = empty (no party), the byte 0x00 (party 0) or the byte
//     0x01 (party 1).  Whenever both parties commit under the same sid and
//     label, the party index MUST be bound: otherwise the party that speaks
//     second can replay the commitment it has just received and later replay
//     its opening (this would pass a "both commit, both open, compare" check
//     without knowing the committed value, and force a coin to zero).
//
// (d) committed_exchange(channel, sid, label, msg) -> the peer's message.
//       round 1: both parties exchange com (32 bytes) bound to their own
//                party index; a malformed commitment, or one equal to the
//                local commitment, aborts;
//       round 2: both exchange omega || msg; the peer's commitment is
//                verified under the peer's party index.
//     Both receives are bounded (channel.hpp): the commitment by 32 bytes,
//     the opening by 32 + max_peer_msg bytes, where max_peer_msg is the
//     longest message the caller accepts from the peer (the protocol code
//     passes the exact length; without the argument the endpoint-wide limit
//     of the channel applies).
//     coin_toss(channel, sid, label, n): each party draws 32 random bytes
//     r_b, the two run committed_exchange under the label "coin|" || label,
//     seed = r_0 XOR r_1, output = sha256_expand(n, "Couplet-coin", sid,
//     label, seed).  Two rounds, 96 payload bytes in each direction.
//     (sid, label) must be unique per toss.
//     coin_toss_gf128 derives n elements of GF(2^128) from one toss: the 16
//     coin bytes are the seed of ctr_prg under the domain dom::PRG_COIN.
//
// (e) struct ProtocolAbort : std::runtime_error -- thrown when a check on
//     data received from the other party fails.  DESIGN.md change G: the
//     caller must move the whole session to ABORTED.  The struct is defined
//     in channel.hpp (a channel refuses over-long messages with it), which
//     this header includes.
//
// Hash instantiation.  DESIGN.md 2.4 and 7.1 name Ascon-Hash256 and
// Ascon-XOF128 for commitments, coins, key derivation and hash-to-group; the
// backend uses SHA-256 for these in both languages (design/CHANGELOG.md).
// ascon_hash256 / ascon_xof128 (NIST SP 800-232) are implemented and tested
// at the end of this file; they are not used by the backend.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "ascon.hpp"
#include "channel.hpp"
#include "gf128.hpp"
#include "sha256.hpp"

namespace couplet {

// struct ProtocolAbort (a check on the other party's messages failed, or a
// message was malformed) is defined in channel.hpp, included above.

namespace hashes {

typedef std::array<uint8_t, 32> hash32;
typedef std::array<uint8_t, 16> seed16;  // same type as ascon::seed_t

// ------------------------------------------------------------------ domains
namespace dom {
const uint64_t CCR_BASE = 0x4800000000000000ull;  // 'H': domains of H_ccr
const uint64_t PRG_BASE = 0x5000000000000000ull;  // 'P': domains of ctr_prg
const uint64_t LOW56 = 0x00ffffffffffffffull;

inline uint64_t ccr(uint64_t id) { return CCR_BASE | (id & LOW56); }
inline uint64_t prg(uint64_t id) { return PRG_BASE | (id & LOW56); }

// Identifiers below 0x100 are reserved for this header and rng.hpp.
const uint64_t PRG_RNG = PRG_BASE | 0x01;   // CtrPrgRng byte stream (rng.hpp)
const uint64_t PRG_COIN = PRG_BASE | 0x02;  // coin_toss_gf128 expansion
const uint64_t CCR_TEST = CCR_BASE | 0x01;  // tests and benchmarks
const uint64_t PRG_TEST = PRG_BASE | 0x03;  // tests and benchmarks

// The x3 words of the FSS PRG labels Couplet-G, -C, -VL, -VR.
inline bool is_reserved(uint64_t d) {
    return d == 0x47ull || d == 0x43ull || d == 0x4c56ull || d == 0x5256ull;
}
inline void check(uint64_t d) {
    if (is_reserved(d)) throw std::invalid_argument("dom collides with an FSS PRG label word");
}
}  // namespace dom

// ------------------------------------------------------------------ (a) H_ccr
inline void H_ccr(uint64_t x_lo, uint64_t x_hi, uint64_t tweak, uint64_t domain, uint64_t out[2]) {
    dom::check(domain);
    uint64_t s[5] = {x_lo, x_hi, tweak, domain, 0};
    ascon::p12(s);
    out[0] = s[0];
    out[1] = s[1];
}

inline gf128 H_ccr(const gf128& x, uint64_t tweak, uint64_t domain) {
    dom::check(domain);
    uint64_t s[5] = {x.lo, x.hi, tweak, domain, 0};
    ascon::p12(s);
    return gf128(s[0], s[1]);
}

namespace detail {
// out[i] = H_ccr(x[i], tw(i), domain) with the 4-way permutation.
template <class TweakFn>
inline void H_ccr_batch_impl(const gf128* x, size_t n, TweakFn tw, uint64_t domain, gf128* out) {
    dom::check(domain);
    size_t i = 0;
    uint64_t st[4][5];
    for (; i + 4 <= n; i += 4) {
        for (size_t j = 0; j < 4; j++) {
            st[j][0] = x[i + j].lo;
            st[j][1] = x[i + j].hi;
            st[j][2] = tw(i + j);
            st[j][3] = domain;
            st[j][4] = 0;
        }
        ascon::p12_x4_fast(st);
        for (size_t j = 0; j < 4; j++) out[i + j] = gf128(st[j][0], st[j][1]);
    }
    for (; i < n; i++) {
        uint64_t s[5] = {x[i].lo, x[i].hi, tw(i), domain, 0};
        ascon::p12(s);
        out[i] = gf128(s[0], s[1]);
    }
}
}  // namespace detail

// out[i] = H_ccr(x[i], tweak0 + i, dom) for i < n.  out may alias x.
inline void H_ccr_batch(const gf128* x, size_t n, uint64_t tweak0, uint64_t domain, gf128* out) {
    detail::H_ccr_batch_impl(x, n, [tweak0](size_t i) { return tweak0 + (uint64_t)i; }, domain, out);
}

// out[i] = H_ccr(x[i], tweak, dom) for i < n (one tweak for all).  out may alias x.
inline void H_ccr_batch_same_tweak(const gf128* x, size_t n, uint64_t tweak, uint64_t domain, gf128* out) {
    detail::H_ccr_batch_impl(x, n, [tweak](size_t) { return tweak; }, domain, out);
}

// out[i] = H_ccr(x[i], tweak[i], dom) for i < n.  out may alias x.
inline void H_ccr_batch_tweaks(const gf128* x, const uint64_t* tweak, size_t n, uint64_t domain, gf128* out) {
    detail::H_ccr_batch_impl(x, n, [tweak](size_t i) { return tweak[i]; }, domain, out);
}

// ------------------------------------------------------------------ (b) ctr_prg
namespace detail {
// Stream words [3 first_block, 3 first_block + n_words) for the seed words (k0, k1).
inline void ctr_prg_core(uint64_t k0, uint64_t k1, uint64_t domain, uint64_t* out, size_t n_words,
                         uint64_t first_block) {
    const size_t full = n_words / 3;
    size_t b = 0;
    uint64_t st[4][5];
    for (; b + 4 <= full; b += 4) {
        for (size_t j = 0; j < 4; j++) {
            st[j][0] = k0;
            st[j][1] = k1;
            st[j][2] = first_block + (uint64_t)(b + j);
            st[j][3] = domain;
            st[j][4] = 0;
        }
        ascon::p12_x4_fast(st);
        for (size_t j = 0; j < 4; j++) {
            out[3 * (b + j)] = st[j][0];
            out[3 * (b + j) + 1] = st[j][1];
            out[3 * (b + j) + 2] = st[j][2];
        }
    }
    for (; b < full; b++) {
        uint64_t s[5] = {k0, k1, first_block + (uint64_t)b, domain, 0};
        ascon::p12(s);
        out[3 * b] = s[0];
        out[3 * b + 1] = s[1];
        out[3 * b + 2] = s[2];
    }
    const size_t rem = n_words - 3 * full;
    if (rem) {
        uint64_t s[5] = {k0, k1, first_block + (uint64_t)full, domain, 0};
        ascon::p12(s);
        for (size_t r = 0; r < rem; r++) out[3 * full + r] = s[r];
    }
}

const size_t CHUNK_BLOCKS = 256;               // blocks per internal chunk
const size_t CHUNK_WORDS = 3 * CHUNK_BLOCKS;   // 768 words = 6144 bytes
}  // namespace detail

// The first n_words stream words; with first_block != 0 the n_words stream
// words starting at word 3 * first_block.
inline void ctr_prg(const uint8_t seed[16], uint64_t domain, uint64_t* out, size_t n_words,
                    uint64_t first_block = 0) {
    dom::check(domain);
    detail::ctr_prg_core(ascon::load64le(seed), ascon::load64le(seed + 8), domain, out, n_words, first_block);
}
inline void ctr_prg(const seed16& seed, uint64_t domain, uint64_t* out, size_t n_words,
                    uint64_t first_block = 0) {
    ctr_prg(seed.data(), domain, out, n_words, first_block);
}
inline std::vector<uint64_t> ctr_prg_words(const seed16& seed, uint64_t domain, size_t n_words,
                                           uint64_t first_block = 0) {
    std::vector<uint64_t> w(n_words);
    ctr_prg(seed.data(), domain, w.data(), n_words, first_block);
    return w;
}

// The first n_bytes bytes of the byte view of the stream.
inline void ctr_prg_bytes(const uint8_t seed[16], uint64_t domain, uint8_t* out, size_t n_bytes) {
    dom::check(domain);
    const uint64_t k0 = ascon::load64le(seed), k1 = ascon::load64le(seed + 8);
    uint64_t buf[detail::CHUNK_WORDS];
    uint64_t blk = 0;
    size_t done = 0;
    while (done < n_bytes) {
        const size_t take = (n_bytes - done < 8 * detail::CHUNK_WORDS) ? (n_bytes - done) : 8 * detail::CHUNK_WORDS;
        const size_t words = (take + 7) / 8;
        detail::ctr_prg_core(k0, k1, domain, buf, words, blk);
        for (size_t i = 0; i < take; i++) out[done + i] = (uint8_t)(buf[i / 8] >> (8 * (i % 8)));
        done += take;
        blk += detail::CHUNK_BLOCKS;
    }
}
inline void ctr_prg_bytes(const seed16& seed, uint64_t domain, uint8_t* out, size_t n_bytes) {
    ctr_prg_bytes(seed.data(), domain, out, n_bytes);
}
inline bytes ctr_prg_bytes(const seed16& seed, uint64_t domain, size_t n_bytes) {
    bytes out(n_bytes);
    ctr_prg_bytes(seed.data(), domain, out.data(), n_bytes);
    return out;
}

// The first n_bits stream bits, one bit per output byte (values 0 / 1).
inline void ctr_prg_bits(const uint8_t seed[16], uint64_t domain, uint8_t* bits, size_t n_bits) {
    dom::check(domain);
    const uint64_t k0 = ascon::load64le(seed), k1 = ascon::load64le(seed + 8);
    uint64_t buf[detail::CHUNK_WORDS];
    uint64_t blk = 0;
    size_t done = 0;
    while (done < n_bits) {
        const size_t take = (n_bits - done < 64 * detail::CHUNK_WORDS) ? (n_bits - done) : 64 * detail::CHUNK_WORDS;
        const size_t words = (take + 63) / 64;
        detail::ctr_prg_core(k0, k1, domain, buf, words, blk);
        for (size_t i = 0; i < take; i++) bits[done + i] = (uint8_t)((buf[i / 64] >> (i % 64)) & 1);
        done += take;
        blk += detail::CHUNK_BLOCKS;
    }
}
inline void ctr_prg_bits(const seed16& seed, uint64_t domain, uint8_t* bits, size_t n_bits) {
    ctr_prg_bits(seed.data(), domain, bits, n_bits);
}

// The first n stream elements of GF(2^128).
inline void ctr_prg_gf128(const uint8_t seed[16], uint64_t domain, gf128* out, size_t n) {
    dom::check(domain);
    const uint64_t k0 = ascon::load64le(seed), k1 = ascon::load64le(seed + 8);
    uint64_t buf[detail::CHUNK_WORDS];
    uint64_t blk = 0;
    size_t done = 0;
    while (done < n) {
        const size_t take = (n - done < detail::CHUNK_WORDS / 2) ? (n - done) : detail::CHUNK_WORDS / 2;
        detail::ctr_prg_core(k0, k1, domain, buf, 2 * take, blk);
        for (size_t i = 0; i < take; i++) out[done + i] = gf128(buf[2 * i], buf[2 * i + 1]);
        done += take;
        blk += detail::CHUNK_BLOCKS;
    }
}
inline void ctr_prg_gf128(const seed16& seed, uint64_t domain, gf128* out, size_t n) {
    ctr_prg_gf128(seed.data(), domain, out, n);
}
inline std::vector<gf128> ctr_prg_gf128(const seed16& seed, uint64_t domain, size_t n) {
    std::vector<gf128> out(n);
    ctr_prg_gf128(seed.data(), domain, out.data(), n);
    return out;
}

// Stateful reader of the byte view: successive fill() calls return
// successive bytes of the stream, independently of how the calls are split.
class CtrPrgStream {
public:
    CtrPrgStream(const uint8_t seed[16], uint64_t domain)
        : k0_(ascon::load64le(seed)), k1_(ascon::load64le(seed + 8)), dom_(domain), block_(0), pos_(BUF_BYTES) {
        dom::check(domain);
    }
    CtrPrgStream(const seed16& seed, uint64_t domain) : CtrPrgStream(seed.data(), domain) {}

    void fill(uint8_t* out, size_t n) {
        while (n) {
            if (pos_ == BUF_BYTES) refill();
            size_t take = BUF_BYTES - pos_;
            if (take > n) take = n;
            std::memcpy(out, buf_ + pos_, take);
            out += take;
            n -= take;
            pos_ += take;
        }
    }

    // Number of stream bytes consumed so far.
    uint64_t consumed() const { return block_ * 24 - (uint64_t)(BUF_BYTES - pos_); }

private:
    enum : size_t { BUF_BLOCKS = 64, BUF_BYTES = 24 * BUF_BLOCKS };

    void refill() {
        uint64_t w[3 * BUF_BLOCKS];
        detail::ctr_prg_core(k0_, k1_, dom_, w, 3 * BUF_BLOCKS, block_);
        block_ += BUF_BLOCKS;
        for (size_t i = 0; i < 3 * BUF_BLOCKS; i++) ascon::store64le(buf_ + 8 * i, w[i]);
        pos_ = 0;
    }

    uint64_t k0_, k1_, dom_, block_;
    size_t pos_;
    uint8_t buf_[BUF_BYTES];
};

// ------------------------------------------------------------------ (c) framed SHA-256
// A non-owning view of one part of a framed hash input.  The viewed bytes
// must stay alive for the duration of the call the Part is passed to.
struct Part {
    const uint8_t* data;
    size_t size;
    Part() : data(nullptr), size(0) {}  // the empty string
    Part(const uint8_t* p, size_t n) : data(p), size(n) {}
    Part(const std::vector<uint8_t>& v) : data(v.data()), size(v.size()) {}
    Part(const std::string& s) : data((const uint8_t*)s.data()), size(s.size()) {}
    Part(const char* s) : data((const uint8_t*)s), size(std::strlen(s)) {}  // ASCII, no NUL
    template <size_t N>
    Part(const std::array<uint8_t, N>& a) : data(a.data()), size(N) {}
};

inline std::array<uint8_t, 8> le64(uint64_t v) {
    std::array<uint8_t, 8> a;
    ascon::store64le(a.data(), v);
    return a;
}

namespace detail {
inline void absorb_part(Sha256& h, const Part& p) {
    const std::array<uint8_t, 8> len = le64((uint64_t)p.size);
    h.update(len.data(), 8);
    if (p.size) h.update(p.data, p.size);
}
template <class It>
inline bytes frame_range(It first, It last) {
    size_t total = 0;
    for (It it = first; it != last; ++it) total += 8 + it->size;
    bytes out;
    out.reserve(total);
    for (It it = first; it != last; ++it) {
        const std::array<uint8_t, 8> len = le64((uint64_t)it->size);
        out.insert(out.end(), len.begin(), len.end());
        if (it->size) out.insert(out.end(), it->data, it->data + it->size);
    }
    return out;
}
template <class It>
inline hash32 framed_sha256_range(It first, It last) {
    Sha256 h;
    for (It it = first; it != last; ++it) absorb_part(h, *it);
    return h.finish();
}
}  // namespace detail

// le64(len(part)) || part for every part, concatenated.
inline bytes frame(std::initializer_list<Part> parts) { return detail::frame_range(parts.begin(), parts.end()); }
inline bytes frame(const std::vector<Part>& parts) { return detail::frame_range(parts.begin(), parts.end()); }

// SHA-256(frame(parts)).
inline hash32 framed_sha256(std::initializer_list<Part> parts) {
    return detail::framed_sha256_range(parts.begin(), parts.end());
}
inline hash32 framed_sha256(const std::vector<Part>& parts) {
    return detail::framed_sha256_range(parts.begin(), parts.end());
}

// SHA-256 in counter mode: the first n_bytes of B_0 || B_1 || ... with
// B_i = framed_sha256(parts..., le64(i)).
inline bytes sha256_expand(size_t n_bytes, std::initializer_list<Part> parts) {
    Sha256 prefix;
    for (const Part& p : parts) detail::absorb_part(prefix, p);
    bytes out;
    out.reserve(n_bytes + 32);
    for (uint64_t i = 0; out.size() < n_bytes; i++) {
        Sha256 h = prefix;
        detail::absorb_part(h, le64(i));
        const hash32 b = h.finish();
        out.insert(out.end(), b.begin(), b.end());
    }
    out.resize(n_bytes);
    return out;
}

// First 16 bytes of a 32-byte digest.
inline seed16 truncate16(const hash32& h) {
    seed16 s;
    std::memcpy(s.data(), h.data(), 16);
    return s;
}

// Key derivation (DESIGN.md 7.1): framed_sha256("Couplet-kdf", parts...),
// truncated to 16 bytes by kdf().
inline hash32 kdf32(std::initializer_list<Part> parts) {
    Sha256 h;
    detail::absorb_part(h, Part("Couplet-kdf"));
    for (const Part& p : parts) detail::absorb_part(h, p);
    return h.finish();
}
inline seed16 kdf(std::initializer_list<Part> parts) { return truncate16(kdf32(parts)); }

// Session-dependent 64-bit domain for H_ccr / ctr_prg (a helper of the C++
// backend): `kind` (0x48 or 0x50) in the top byte, below it the low 56 bits
// of the little-endian integer formed by the first 8 bytes of
// framed_sha256("Couplet-dom", label, sid).
inline uint64_t session_dom(uint8_t kind, const Part& label, const Part& sid) {
    const hash32 h = framed_sha256({"Couplet-dom", label, sid});
    return ((uint64_t)kind << 56) | (ascon::load64le(h.data()) & dom::LOW56);
}
inline uint64_t session_dom_ccr(const Part& label, const Part& sid) { return session_dom(0x48, label, sid); }
inline uint64_t session_dom_prg(const Part& label, const Part& sid) { return session_dom(0x50, label, sid); }

// Equality of two byte strings without data-dependent branches.
inline bool ct_equal(const uint8_t* a, const uint8_t* b, size_t n) {
    unsigned acc = 0;
    for (size_t i = 0; i < n; i++) acc |= (unsigned)(a[i] ^ b[i]);
    return acc == 0;
}

// ------------------------------------------------------------------ commitments
typedef std::array<uint8_t, 32> opening32;

const int NO_PARTY = -1;  // commitment not bound to a party index

struct Commitment {
    hash32 com;         // sent first
    opening32 opening;  // omega: sent, together with the message, to open
};

namespace detail {
inline Part party_tag(int party) {
    static const uint8_t tags[2] = {0x00, 0x01};
    if (party == NO_PARTY) return Part();
    if (party != 0 && party != 1) throw std::invalid_argument("party must be 0, 1 or NO_PARTY");
    return Part(&tags[party], 1);
}
}  // namespace detail

// com = framed_sha256("Couplet-com", sid, label, party_tag, omega, msg).
inline hash32 commit_with_opening(const Part& sid, const Part& msg, const opening32& omega,
                                  int party = NO_PARTY, const Part& label = Part()) {
    return framed_sha256({"Couplet-com", sid, label, detail::party_tag(party), omega, msg});
}

// Commit to msg with fresh randomness from `rng` (any object with
// fill(uint8_t*, size_t), e.g. the sources of rng.hpp).  Pass the
// committer's index as `party` whenever both parties commit under the same
// (sid, label); see the header comment.
template <class ByteSource>
inline Commitment commit(const Part& sid, const Part& msg, ByteSource& rng, int party = NO_PARTY,
                         const Part& label = Part()) {
    Commitment c;
    rng.fill(c.opening.data(), c.opening.size());
    c.com = commit_with_opening(sid, msg, c.opening, party, label);
    return c;
}

// True iff (opening, msg) opens com under (sid, label, party).
inline bool verify_commit(const hash32& com, const Part& sid, const Part& msg, const opening32& opening,
                          int party = NO_PARTY, const Part& label = Part()) {
    const hash32 expect = commit_with_opening(sid, msg, opening, party, label);
    return ct_equal(expect.data(), com.data(), 32);
}
inline bool verify(const hash32& com, const Part& sid, const Part& msg, const opening32& opening,
                   int party = NO_PARTY, const Part& label = Part()) {
    return verify_commit(com, sid, msg, opening, party, label);
}

// ------------------------------------------------------------------ (d) committed exchange, coins
// committed_exchange without a limit of its own for the peer's message: the
// endpoint-wide limit of the channel applies.
const size_t CHANNEL_LIMIT = ~(size_t)0;

// Simultaneous commit-then-open of one message per party; returns the
// peer's message (the caller checks its length and content).  Two rounds;
// 32 + 32 + len(msg) payload bytes in each direction.  Throws ProtocolAbort
// if the peer's commitment is malformed or equal to the local one, if its
// opening does not verify, or if its opening is longer than 32 +
// max_peer_msg bytes (refused by the channel before it is read).
template <class ByteSource>
inline bytes committed_exchange(Channel& ch, const Part& sid, const Part& label, const Part& msg,
                                ByteSource& rng, size_t max_peer_msg = CHANNEL_LIMIT) {
    const int me = ch.party();
    const Commitment c = commit(sid, msg, rng, me, label);

    const bytes peer_com = ch.exchange(bytes(c.com.begin(), c.com.end()), 32, "committed_exchange: malformed commitment");
    if (peer_com.size() != 32) throw ProtocolAbort("committed_exchange: malformed commitment");
    if (ct_equal(peer_com.data(), c.com.data(), 32)) throw ProtocolAbort("committed_exchange: mirrored commitment");

    bytes my_open(c.opening.begin(), c.opening.end());
    if (msg.size) my_open.insert(my_open.end(), msg.data, msg.data + msg.size);
    const bytes peer_open =
        max_peer_msg >= Channel::MAX_MESSAGE - 32
            ? ch.exchange(my_open)
            : ch.exchange(my_open, 32 + max_peer_msg, "committed_exchange: opening has the wrong length");
    if (peer_open.size() < 32) throw ProtocolAbort("committed_exchange: malformed opening");

    hash32 pc;
    opening32 peer_omega;
    std::memcpy(pc.data(), peer_com.data(), 32);
    std::memcpy(peer_omega.data(), peer_open.data(), 32);
    const Part peer_msg(peer_open.data() + 32, peer_open.size() - 32);
    if (!verify_commit(pc, sid, peer_msg, peer_omega, 1 - me, label))
        throw ProtocolAbort("committed_exchange: commitment opening failed");
    return bytes(peer_open.begin() + 32, peer_open.end());
}

// The label under which the coin shares are committed: "coin|" || label.
inline bytes coin_label(const Part& label) {
    static const char prefix[] = "coin|";
    bytes l(prefix, prefix + 5);
    if (label.size) l.insert(l.end(), label.data, label.data + label.size);
    return l;
}

// sha256_expand(n_bytes, "Couplet-coin", sid, label, seed).
inline bytes coin_expand(const Part& sid, const Part& label, const std::array<uint8_t, 32>& seed, size_t n_bytes) {
    return sha256_expand(n_bytes, {"Couplet-coin", sid, label, seed});
}

// The joint 32-byte seed r_0 XOR r_1 of a coin toss (two rounds).
template <class ByteSource>
inline std::array<uint8_t, 32> coin_toss_seed(Channel& ch, const Part& sid, const Part& label, ByteSource& rng) {
    std::array<uint8_t, 32> mine;
    rng.fill(mine.data(), mine.size());
    const bytes theirs = committed_exchange(ch, sid, coin_label(label), mine, rng, mine.size());
    if (theirs.size() != 32) throw ProtocolAbort("coin_toss: coin share has the wrong length");
    std::array<uint8_t, 32> seed;
    for (size_t i = 0; i < 32; i++) seed[i] = (uint8_t)(mine[i] ^ theirs[i]);
    return seed;
}

// n_bytes jointly random public bytes (the same value at both parties).
template <class ByteSource>
inline bytes coin_toss(Channel& ch, const Part& sid, const Part& label, size_t n_bytes, ByteSource& rng) {
    return coin_expand(sid, label, coin_toss_seed(ch, sid, label, rng), n_bytes);
}

// A jointly random 16-byte seed: coin_toss(..., 16).
template <class ByteSource>
inline seed16 coin_toss_seed16(Channel& ch, const Part& sid, const Part& label, ByteSource& rng) {
    const bytes b = coin_toss(ch, sid, label, 16, rng);
    seed16 s;
    std::memcpy(s.data(), b.data(), 16);
    return s;
}

// n jointly random public elements of GF(2^128): the gf128 view of
// ctr_prg(coin_toss(..., 16), dom::PRG_COIN).
inline std::vector<gf128> coin_gf128_from_seed(const seed16& seed, size_t n) {
    return ctr_prg_gf128(seed, dom::PRG_COIN, n);
}
template <class ByteSource>
inline std::vector<gf128> coin_toss_gf128(Channel& ch, const Part& sid, const Part& label, size_t n, ByteSource& rng) {
    return coin_gf128_from_seed(coin_toss_seed16(ch, sid, label, rng), n);
}

// ------------------------------------------------------------------ Ascon-Hash256 / Ascon-XOF128
// NIST SP 800-232: rate 64 bits, little-endian words, padding byte 0x01,
// p12 for initialisation, absorption and squeezing.  Not used by the
// backend (see the header comment).
namespace detail {
inline void ascon_sponge(uint64_t iv, const uint8_t* in, size_t len, uint8_t* out, size_t outlen) {
    uint64_t s[5] = {iv, 0, 0, 0, 0};
    ascon::p12(s);
    while (len >= 8) {
        s[0] ^= ascon::load64le(in);
        ascon::p12(s);
        in += 8;
        len -= 8;
    }
    uint64_t last = (uint64_t)0x01 << (8 * len);
    for (size_t i = 0; i < len; i++) last |= (uint64_t)in[i] << (8 * i);
    s[0] ^= last;
    ascon::p12(s);
    while (outlen > 8) {
        ascon::store64le(out, s[0]);
        ascon::p12(s);
        out += 8;
        outlen -= 8;
    }
    uint8_t tail[8];
    ascon::store64le(tail, s[0]);
    if (outlen) std::memcpy(out, tail, outlen);
}
}  // namespace detail

inline void ascon_hash256(const uint8_t* in, size_t len, uint8_t out[32]) {
    detail::ascon_sponge(0x0000080100cc0002ull, in, len, out, 32);
}
inline hash32 ascon_hash256(const bytes& in) {
    hash32 h;
    ascon_hash256(in.data(), in.size(), h.data());
    return h;
}
inline void ascon_xof128(const uint8_t* in, size_t len, uint8_t* out, size_t outlen) {
    detail::ascon_sponge(0x0000080000cc0003ull, in, len, out, outlen);
}
inline bytes ascon_xof128(const bytes& in, size_t outlen) {
    bytes out(outlen);
    ascon_xof128(in.data(), in.size(), out.data(), outlen);
    return out;
}

// A 32-byte hash functor for ed25519::hash_to_group_with.
struct AsconXof128Hash32 {
    void operator()(const uint8_t* data, size_t len, uint8_t out[32]) const { ascon_xof128(data, len, out, 32); }
};

}  // namespace hashes
}  // namespace couplet
