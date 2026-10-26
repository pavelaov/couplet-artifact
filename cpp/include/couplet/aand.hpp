// couplet/aand.hpp -- authenticated AND triples: leaky AND, bucketing and
// the triple consistency check (DESIGN.md Sections 7.4, 7.5); the C++
// counterpart of couplet/prep/aand.py (same messages, labels and domains).
//
// All functions run ONE party (index role, key share delta = Delta^B_role)
// on its channel endpoint.
//
// Leaky AND (leaky_ands).  Inputs of party b for every triple: its share x_b
// in BDOZ form (M[x_b] under the peer's key, K[x_{1-b}] for the peer's share)
// and its SPDZ-style MAC share m_b of y.  One exchange:
//
//     party b sends   G_b = H(K[x_{1-b}]) XOR H(K[x_{1-b}] XOR Delta^B_b) XOR m_b ,
//     keeps           S_b = H(K[x_{1-b}]) ,
//     and computes    R_b = H(M[x_b]) XOR x_b G_{1-b}        ( = S_{1-b} XOR x_b m_{1-b} ),
//                     n_b = x_b m_b XOR R_b XOR S_b ,   z_b = lsb(n_b) .
//
// Then n_0 XOR n_1 = x y Delta^B and, because lsb(Delta^B) = 1,
// z_0 XOR z_1 = x y: (z_b, n_b) is <z> with z = x y.  H is hashes::H_ccr with
// the global triple index as tweak and the domain
// session_dom(0x48, "aand/<sender of the transfer>", sid), evaluated four at
// a time with the interleaved permutation.
//
// Bucketing (bucket_permutation, bucket_combine).  After ALL leaky triples
// exist, a public permutation is derived from a coin toss: the argsort of the
// 128-bit keys ctr_prg_gf128(coin, prg_dom(USE_BUCKET_PERM)) compared as
// integers, ties by index (probability below n^2 / 2^129).  The triples are
// grouped into buckets of size B.  Within a bucket with head (x', y', z')
// every other member (x'', y'', z'') is folded in with the opened bit
// d = y' XOR y'':  x = x' XOR x'',  y = y',  z = z' XOR z'' XOR d x''.
//
// Consistency check (triple_check), on the RAW leaky triples: with 128 fresh
// random <r_j>, <R> = sum_j X^j <r_j>, and chi_i from a coin toss, the parties
// open W = sum_i chi_i z_i XOR R and run bool_check on W (MAC shares
// sum_i chi_i n_{i,b} XOR mac_b(R)) together with all opened d bits.  The
// triples are returned only after this check has passed.
//
// and_triples chains the steps for one batch and books traffic and time under
// the phases cotext, abit, laand, bucket, check.
//
// The bulk loops split their work over the worker threads of the calling
// party (shares.hpp, WorkerScope); the results do not depend on the number
// of workers.
//
// Test hooks (struct Cheat, default nullptr = honest):
//   g_error   add g_err[i] to the transfer message of triple g_idx[i] and
//             assume the peer's x share of that triple equals g_guess[i]
//             (the deviation of Lemma 3);
//   flip_d    flip own shares of opened d bits;
//   w_error   add an error to the own share of W.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "channel.hpp"
#include "gf128.hpp"
#include "hashes.hpp"
#include "params.hpp"
#include "rng.hpp"
#include "shares.hpp"

namespace couplet {
namespace prep {

// Triples per transfer message of the leaky AND (larger batches are split
// into several exchanges so that a message stays below the frame limit).
const size_t LAAND_MSG_TRIPLES = (size_t)1 << 24;

// Leaky triples of one and_triples batch (the permutation uses 32-bit indices).
const uint64_t MAX_LEAKY_PER_BATCH = 0xffffffffull;

// Domain of H for the transfer sent by party `sender` in session `sid`.
inline uint64_t h_dom(const bytes& sid, int sender) {
    check_role(sender);
    return session_dom(DOM_CCR, sender == 0 ? "aand/0" : "aand/1", sid);
}

// label || le64(counter).
inline bytes counter_label(const char* prefix, uint64_t counter) {
    bytes label(prefix, prefix + std::strlen(prefix));
    const std::array<uint8_t, 8> c = hashes::le64(counter);
    label.insert(label.end(), c.begin(), c.end());
    return label;
}

// ------------------------------------------------------------------ leaky AND (Section 7.4)
// Leaky <z> with z = x AND y for n triples (16 n bytes per party; one round
// per LAAND_MSG_TRIPLES triples).
//   index0   global index of the first triple; indices must never be reused
//            within a session (they are the tweak of H);
//   x_bits, x_macs   own share x_b with M[x_b];   x_keys   K[x_{1-b}];
//   y_macs   SPDZ-style MAC share m_b of y.
// The result has z.bits[i] = lsb(z.macs[i]).
inline void leaky_ands(Channel& ch, const bytes& sid, int role, const gf128& delta, uint64_t index0, size_t n,
                       const uint8_t* x_bits, const gf128* x_macs, const gf128* x_keys, const gf128* y_macs,
                       Shares& z, const Cheat* cheat = nullptr) {
    check_role(role);
    if (index0 + (uint64_t)n < index0) throw std::invalid_argument("triple index out of the 64-bit range");
    const uint64_t dom_out = h_dom(sid, role);      // transfer from this party to the peer
    const uint64_t dom_in = h_dom(sid, 1 - role);   // transfer from the peer to this party
    const bool g_error = cheat && cheat->has_g_error;
    if (g_error) {
        if (cheat->g_err.size() != cheat->g_idx.size() || cheat->g_guess.size() != cheat->g_idx.size())
            throw std::invalid_argument("g_error: idx, err and guess must have the same length");
        for (size_t idx : cheat->g_idx)
            if (idx >= n) throw std::invalid_argument("g_error: index out of range");
    }
    z.bits.assign(n, 0);
    z.macs.resize(n);
    gf128* nz = z.macs.data();
    uint8_t* zb = z.bits.data();

    const size_t BLK = 512;
    size_t c0 = 0;
    do {
        const size_t c1 = (n - c0 < LAAND_MSG_TRIPLES) ? n : c0 + LAAND_MSG_TRIPLES;
        bytes msg(16 * (c1 - c0));
        detail::parallel_chunks(c1 - c0, BLK, 16 * BLK, [&](size_t r0, size_t r1, unsigned) {
            GfVec scratch(BLK);
            gf128* buf = scratch.data();
            for (size_t i = c0 + r0; i < c0 + r1; i += BLK) {
                const size_t m = (c0 + r1 - i < BLK) ? (c0 + r1 - i) : BLK;
                detail::ccr_hash(x_keys + i, m, index0 + i, dom_out, nz + i);          // S_b = H(K[x_{1-b}])
                detail::ccr_hash(x_keys + i, m, index0 + i, dom_out, buf, delta);      // H(K[x_{1-b}] XOR Delta^B_b)
                for (size_t j = 0; j < m; j++) buf[j] ^= nz[i + j] ^ y_macs[i + j];    // G_b
                gf_store(msg.data() + 16 * (i - c0), buf, m);
            }
        });
        if (g_error) {  // test hook: add E to the transfer messages of the attacked triples
            for (size_t t = 0; t < cheat->g_idx.size(); t++) {
                const size_t idx = cheat->g_idx[t];
                if (idx < c0 || idx >= c1) continue;
                uint8_t* p = msg.data() + 16 * (idx - c0);
                (gf128::from_bytes(p) ^ cheat->g_err[t]).to_bytes(p);
            }
        }
        const bytes reply = ch.exchange(msg, msg.size(), "leaky AND: transfer message has the wrong length");
        if (reply.size() != msg.size()) throw ProtocolAbort("leaky AND: transfer message has the wrong length");
        detail::parallel_chunks(c1 - c0, BLK, 16 * BLK, [&](size_t r0, size_t r1, unsigned) {
            GfVec scratch(2 * BLK);
            gf128* buf = scratch.data();
            gf128* peer = scratch.data() + BLK;
            for (size_t i = c0 + r0; i < c0 + r1; i += BLK) {
                const size_t m = (c0 + r1 - i < BLK) ? (c0 + r1 - i) : BLK;
                detail::ccr_hash(x_macs + i, m, index0 + i, dom_in, buf);  // H(M[x_b])
                gf_load(peer, reply.data() + 16 * (i - c0), m);           // G_{1-b}
                for (size_t j = 0; j < m; j++) {
                    const uint64_t sel = 0 - (uint64_t)(x_bits[i + j] & 1);
                    // n_b = H(M[x_b]) XOR x_b (G_{1-b} XOR m_b) XOR S_b
                    const uint64_t lo = nz[i + j].lo ^ buf[j].lo ^ ((peer[j].lo ^ y_macs[i + j].lo) & sel);
                    const uint64_t hi = nz[i + j].hi ^ buf[j].hi ^ ((peer[j].hi ^ y_macs[i + j].hi) & sel);
                    nz[i + j] = gf128(lo, hi);
                    zb[i + j] = (uint8_t)(lo & 1);
                }
            }
        });
        c0 = c1;
    } while (c0 < n);
    if (g_error) {  // test hook: adjust the own MAC share for the guessed peer bits (Lemma 3)
        for (size_t t = 0; t < cheat->g_idx.size(); t++) {
            const size_t idx = cheat->g_idx[t];
            nz[idx] ^= gf_and_bit(cheat->g_err[t], cheat->g_guess[t]);
            zb[idx] = (uint8_t)(nz[idx].lo & 1);
        }
    }
}

inline Shares leaky_ands(Channel& ch, const bytes& sid, int role, const gf128& delta, uint64_t index0,
                         const Bdoz& x_own, const Keys& x_keys, const Shares& y, const Cheat* cheat = nullptr) {
    const size_t n = y.size();
    if (x_own.size() != n || x_keys.size() != n) throw std::invalid_argument("x and y must have the same length");
    Shares z;
    leaky_ands(ch, sid, role, delta, index0, n, x_own.bits.data(), x_own.macs.data(), x_keys.keys.data(),
               y.macs.data(), z, cheat);
    return z;
}

// ------------------------------------------------------------------ bucketing (Section 7.5)
namespace detail {

// Element i of the GF(2^128) view of ctr_prg(seed, dom): stream words (2i, 2i + 1).
inline gf128 prg_gf128_at(const hashes::seed16& seed, uint64_t dom, uint64_t i) {
    uint64_t w[6];
    hashes::ctr_prg(seed.data(), dom, w, 6, (2 * i) / 3);
    const size_t o = (size_t)((2 * i) % 3);
    return gf128(w[o], w[o + 1]);
}

// One leaky triple as the bucketing reads it: one cache line.
struct alignas(64) LeakyRec {
    uint64_t xm[2], ym[2], zm[2];  // MAC shares of x, y, z
    uint64_t bits;                 // bit 0: x share, bit 8: y share, bit 16: z share
    uint64_t pad;
};
static_assert(sizeof(LeakyRec) == 64, "LeakyRec must be one cache line");

}  // namespace detail

// Public permutation of {0 .. n-1} from a 16-byte coin: the stable argsort of
// the keys ctr_prg_gf128(seed, prg_dom(USE_BUCKET_PERM)) as 128-bit integers.
// Implementation: the entries (top 32 key bits, index) are distributed by the
// top key byte, every part is radix-sorted on the next 24 bits (three stable
// byte passes inside the cache), and runs with equal top 32 bits are then
// ordered by the full key and the index.
inline std::vector<uint32_t> bucket_permutation(const hashes::seed16& seed, size_t n) {
    if (n > 0xffffffffull) throw std::invalid_argument("bucket_permutation: at most 2^32 - 1 elements");
    const uint64_t dom = prg_dom(USE_BUCKET_PERM);
    const size_t STEP = 1536;  // elements per piece: 3072 stream words = 1024 PRG blocks
    std::vector<uint64_t> a(n), b(n);
    size_t start[257];
    {
        // keys and, per part of the range, the histogram of the top byte
        const unsigned workers = detail::worker_count();
        std::vector<size_t> count((size_t)workers * 256, 0);
        detail::parallel_ranges(n, STEP, 64 * STEP, [&](size_t r0, size_t r1, unsigned part) {
            std::vector<uint64_t> w(2 * STEP);
            size_t* cnt = count.data() + (size_t)part * 256;
            for (size_t done = r0; done < r1; done += STEP) {
                const size_t take = (r1 - done < STEP) ? (r1 - done) : STEP;
                detail::prg_words(seed, dom, w.data(), 2 * take, (uint64_t)(2 * done / 3));
                for (size_t i = 0; i < take; i++) {
                    const uint64_t v = (w[2 * i + 1] & 0xffffffff00000000ull) | (uint64_t)(done + i);
                    a[done + i] = v;
                    cnt[v >> 56]++;
                }
            }
        });
        // first output position of (top byte v, part p): stable across the parts
        size_t pos = 0;
        for (unsigned v = 0; v < 256; v++) {
            start[v] = pos;
            for (unsigned p = 0; p < workers; p++) {
                const size_t c = count[(size_t)p * 256 + v];
                count[(size_t)p * 256 + v] = pos;
                pos += c;
            }
        }
        start[256] = pos;
        // distribute by the top byte: a -> b (the same split of the range as above)
        detail::parallel_ranges(n, STEP, 64 * STEP, [&](size_t r0, size_t r1, unsigned part) {
            size_t* next = count.data() + (size_t)part * 256;
            const uint64_t* src = a.data();
            uint64_t* dst = b.data();
            for (size_t i = r0; i < r1; i++) dst[next[src[i] >> 56]++] = src[i];
        });
    }
    // every part: three stable byte passes on bits 32 .. 55 (b -> a -> b -> a)
    detail::parallel_chunks(256, 1, n >= ((size_t)1 << 18) ? 4 : 257, [&](size_t p0, size_t p1, unsigned) {
        std::vector<size_t> cnt(3 * 256);
        for (size_t part = p0; part < p1; part++) {
            const size_t lo = start[part], len = start[part + 1] - start[part];
            if (len == 0) continue;
            uint64_t* from = b.data() + lo;
            uint64_t* to = a.data() + lo;
            std::fill(cnt.begin(), cnt.end(), (size_t)0);
            for (size_t i = 0; i < len; i++) {
                const uint64_t v = from[i];
                cnt[(v >> 32) & 0xff]++;
                cnt[256 + ((v >> 40) & 0xff)]++;
                cnt[512 + ((v >> 48) & 0xff)]++;
            }
            for (unsigned d = 0; d < 3; d++) {
                size_t* c = cnt.data() + 256 * d;
                size_t pos = 0;
                for (unsigned v = 0; v < 256; v++) {
                    const size_t k = c[v];
                    c[v] = pos;
                    pos += k;
                }
                const unsigned shift = 32 + 8 * d;
                for (size_t i = 0; i < len; i++) to[c[(from[i] >> shift) & 0xff]++] = from[i];
                std::swap(from, to);
            }
        }
    });
    // a is sorted by the top 32 key bits, stable in the index.  Resolve runs.
    struct Full {
        uint64_t hi, lo;
        uint32_t idx;
    };
    std::vector<Full> run;
    for (size_t i = 0; i < n;) {
        size_t j = i + 1;
        while (j < n && (a[j] >> 32) == (a[i] >> 32)) j++;
        if (j - i > 1) {
            run.clear();
            for (size_t t = i; t < j; t++) {
                const uint32_t idx = (uint32_t)a[t];
                const gf128 key = detail::prg_gf128_at(seed, dom, idx);
                run.push_back(Full{key.hi, key.lo, idx});
            }
            std::sort(run.begin(), run.end(), [](const Full& p, const Full& q) {
                if (p.hi != q.hi) return p.hi < q.hi;
                if (p.lo != q.lo) return p.lo < q.lo;
                return p.idx < q.idx;
            });
            for (size_t t = i; t < j; t++) a[t] = (a[t] & 0xffffffff00000000ull) | run[t - i].idx;
        }
        i = j;
    }
    std::vector<uint32_t> perm(n);
    for (size_t i = 0; i < n; i++) perm[i] = (uint32_t)a[i];
    return perm;
}

// Reference for bucket_permutation (tests): sort of the full keys.
inline std::vector<uint32_t> bucket_permutation_ref(const hashes::seed16& seed, size_t n) {
    const GfVec keys = hashes::ctr_prg_gf128(seed, prg_dom(USE_BUCKET_PERM), n);
    std::vector<uint32_t> perm(n);
    for (size_t i = 0; i < n; i++) perm[i] = (uint32_t)i;
    std::stable_sort(perm.begin(), perm.end(), [&keys](uint32_t p, uint32_t q) {
        if (keys[p].hi != keys[q].hi) return keys[p].hi < keys[q].hi;
        return keys[p].lo < keys[q].lo;
    });
    return perm;
}

struct BucketOut {
    Shares a, b, c;  // the combined triples
    BitVec d;        // opened bits (head y XOR member y), row-major over (bucket, member)
    GfVec d_macs;    // this party's MAC shares of the d bits
};

// Combine len(perm) / bucket buckets.  Bucket k consists of the leaky triples
// perm[k * bucket + i], i < bucket, the first one being the head.  The opened
// bits d and their MAC shares must go into the next BoolCheck.
// One round, ceil(n (B - 1) / 8) bytes per party.
//
// The permutation scatters the members of a bucket over the whole input, so
// the leaky triples are first packed into one cache line each (sequential
// pass) and the buckets then read one line per member.
inline BucketOut bucket_combine(Channel& ch, const ShareView& x, const ShareView& y, const ShareView& z,
                                const std::vector<uint32_t>& perm, size_t bucket, const Cheat* cheat = nullptr) {
    const size_t total = x.n;
    if (y.n != total || z.n != total || perm.size() != total)
        throw std::invalid_argument("x, y, z and the permutation must have the same length");
    if (bucket < 1 || total % bucket) throw std::invalid_argument("the number of leaky triples must be a multiple of the bucket size");
    for (uint32_t p : perm)
        if (p >= total) throw std::invalid_argument("permutation entry out of range");
    const size_t n = total / bucket;
    const size_t m = bucket - 1;  // members besides the head

    BucketOut out;
    out.a = Shares(n);
    out.b = Shares(n);
    out.c = Shares(n);
    out.d_macs.resize(n * m);
    BitVec d_share(n * m);
    BitVec xs_bits(n * m);  // x'' of the members, in bucket order
    GfVec xs_macs(n * m);

    // Pack: one record per leaky triple.
    std::unique_ptr<detail::LeakyRec[]> rec(new detail::LeakyRec[total]);
    detail::parallel_chunks(total, 1, (size_t)1 << 16, [&](size_t i0, size_t i1, unsigned) {
        detail::LeakyRec* r = rec.get();
        for (size_t i = i0; i < i1; i++) {
            const uint64_t bits = (uint64_t)(x.bits[i] & 1) | ((uint64_t)(y.bits[i] & 1) << 8) | ((uint64_t)(z.bits[i] & 1) << 16);
#if COUPLET_HAVE_AVX2
            const __m256i v0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(x.macs + i))),
                                                       _mm_loadu_si128((const __m128i*)(y.macs + i)), 1);
            const __m256i v1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(z.macs + i))),
                                                       _mm_cvtsi64_si128((long long)bits), 1);
            _mm256_stream_si256((__m256i*)(r + i), v0);  // the records are read back in random order: bypass the cache
            _mm256_stream_si256((__m256i*)((uint8_t*)(r + i) + 32), v1);
#else
            r[i].xm[0] = x.macs[i].lo;
            r[i].xm[1] = x.macs[i].hi;
            r[i].ym[0] = y.macs[i].lo;
            r[i].ym[1] = y.macs[i].hi;
            r[i].zm[0] = z.macs[i].lo;
            r[i].zm[1] = z.macs[i].hi;
            r[i].bits = bits;
            r[i].pad = 0;
#endif
        }
#if COUPLET_HAVE_AVX2
        _mm_sfence();
#endif
    });

    // Pass 1: everything that does not depend on the opened d bits.
    detail::parallel_chunks(n, 1, (size_t)1 << 13, [&](size_t k0, size_t k1, unsigned) {
        const detail::LeakyRec* r = rec.get();
        const size_t AHEAD = 6;  // buckets of read-ahead
        for (size_t k = k0; k < k1; k++) {
            const uint32_t* g = perm.data() + k * bucket;
#if COUPLET_PREP_X86
            if (k + AHEAD < k1) {
                const uint32_t* f = g + AHEAD * bucket;
                for (size_t i = 0; i < bucket; i++) _mm_prefetch((const char*)(r + f[i]), _MM_HINT_T0);
            }
#endif
            const detail::LeakyRec& h = r[g[0]];
            uint64_t a_lo = h.xm[0], a_hi = h.xm[1], c_lo = h.zm[0], c_hi = h.zm[1];
            uint64_t ac_bits = h.bits;  // bit 0 accumulates x, bit 16 accumulates z
            for (size_t i = 1; i < bucket; i++) {
                const detail::LeakyRec& t = r[g[i]];
                const size_t pos = k * m + (i - 1);
                a_lo ^= t.xm[0];
                a_hi ^= t.xm[1];
                c_lo ^= t.zm[0];
                c_hi ^= t.zm[1];
                ac_bits ^= t.bits;
                d_share[pos] = (uint8_t)(((h.bits ^ t.bits) >> 8) & 1);
                out.d_macs[pos] = gf128(h.ym[0] ^ t.ym[0], h.ym[1] ^ t.ym[1]);
                xs_bits[pos] = (uint8_t)(t.bits & 1);
                xs_macs[pos] = gf128(t.xm[0], t.xm[1]);
            }
            out.a.bits[k] = (uint8_t)(ac_bits & 1);
            out.a.macs[k] = gf128(a_lo, a_hi);
            out.b.bits[k] = (uint8_t)((h.bits >> 8) & 1);
            out.b.macs[k] = gf128(h.ym[0], h.ym[1]);
            out.c.bits[k] = (uint8_t)((ac_bits >> 16) & 1);
            out.c.macs[k] = gf128(c_lo, c_hi);
        }
    });
    rec.reset();

    // Open d = y' XOR y''.
    if (cheat) {
        for (size_t pos : cheat->flip_d) {
            if (pos >= d_share.size()) throw std::invalid_argument("flip_d out of range");
            d_share[pos] ^= 1;  // the cheater sends (and uses) a wrong share
        }
    }
    out.d = exchange_bits(ch, d_share);
    for (size_t i = 0; i < out.d.size(); i++) out.d[i] ^= d_share[i];

    // Pass 2: z = z' XOR z'' XOR d x''.
    detail::parallel_chunks(n, 1, (size_t)1 << 16, [&](size_t k0, size_t k1, unsigned) {
        for (size_t k = k0; k < k1; k++) {
            uint8_t c_bit = out.c.bits[k];
            gf128 c_mac = out.c.macs[k];
            for (size_t i = 0; i < m; i++) {
                const size_t pos = k * m + i;
                const uint8_t d = out.d[pos];
                c_bit ^= (uint8_t)(d & xs_bits[pos]);
                c_mac ^= gf_and_bit(xs_macs[pos], d);
            }
            out.c.bits[k] = c_bit;
            out.c.macs[k] = c_mac;
        }
    });
    return out;
}

// ------------------------------------------------------------------ triple consistency check (Section 7.4)
// Consistency check of the raw leaky triples z with the mask bits r (exactly
// PAD = 128 shares).  Tosses the chi, opens W = sum_i chi_i z_i XOR R and runs
// BoolCheck number `counter` on W and on the already opened bits.  Throws
// ProtocolAbort on failure.  Five rounds.
template <class ByteSource>
inline void triple_check(Channel& ch, const bytes& sid, uint64_t counter, int role, const gf128& delta,
                         const ShareView& z, const ShareView& r, const uint8_t* opened_bits,
                         const gf128* opened_macs, size_t n_opened, ByteSource& rng, const Cheat* cheat = nullptr) {
    if (r.n != PAD) throw std::invalid_argument("exactly PAD mask bits are required");
    const bytes label = counter_label("triple-chi|", counter);
    const hashes::seed16 seed = hashes::coin_toss_seed16(ch, sid, label, rng);
    const uint64_t dom = prg_dom(USE_TRIPLE_CHI);

    const size_t STEP = 1536;  // elements per piece: 3072 stream words = 1024 PRG blocks
    const unsigned workers = detail::worker_count();
    GfVec w_parts(workers), m_parts(workers);
    detail::parallel_chunks(z.n, STEP, 32 * STEP, [&](size_t r0, size_t r1, unsigned part) {
        std::vector<uint64_t> w(2 * STEP);
        GfVec chi(STEP);
        gf128 w_acc, m_acc;
        for (size_t done = r0; done < r1; done += STEP) {
            const size_t take = (r1 - done < STEP) ? (r1 - done) : STEP;
            detail::prg_words(seed, dom, w.data(), 2 * take, (uint64_t)(2 * done / 3));
            for (size_t i = 0; i < take; i++) chi[i] = gf128(w[2 * i], w[2 * i + 1]);
            w_acc ^= gf_dot_bits(chi.data(), z.bits + done, take);
            m_acc ^= gf_dot(chi.data(), z.macs + done, take);
        }
        w_parts[part] ^= w_acc;
        m_parts[part] ^= m_acc;
    });
    gf128 w_share, m_share;
    for (unsigned p = 0; p < workers; p++) {
        w_share ^= w_parts[p];
        m_share ^= m_parts[p];
    }
    const std::pair<gf128, gf128> rf = to_field(r);
    w_share ^= rf.first;
    m_share ^= rf.second;
    if (cheat && cheat->has_w_error) w_share ^= cheat->w_error;

    const bytes reply = ch.exchange(gf_to_bytes(w_share), 16, "triple check: share of W has the wrong length");
    if (reply.size() != 16) throw ProtocolAbort("triple check: share of W has the wrong length");
    const gf128 big_w = w_share ^ gf128::from_bytes(reply.data());
    bool_check(ch, sid, counter, role, delta, opened_bits, opened_macs, n_opened,
               std::vector<FieldOpening>{FieldOpening(big_w, m_share)}, rng);
}

// ------------------------------------------------------------------ one batch end to end
// n bucketed and checked triples (a, b, c), c = a AND b.
//   source  random BDOZ bits (AbitSource or the ideal preparation);
//   n, bucket  size of the run and bucket size B.  The caller chooses them
//           with triple_request (RealPrep::and_triples does), which pads a
//           small request so that the check covers at least MIN_LEAKY_TRIPLES
//           leaky triples;
//   batch   number of this batch in the session (labels of the coin tosses
//           and of the BoolCheck);
//   index0  global index of the first of the n * B leaky triples;
//   meter   optional per-phase accounting.
template <class ByteSource>
inline Triples and_triples(Channel& ch, const bytes& sid, int role, const gf128& delta, BdozSource& source, size_t n,
                           size_t bucket, uint64_t batch, uint64_t index0, ByteSource& rng,
                           PhaseMeter* meter = nullptr, const Cheat* cheat = nullptr) {
    check_role(role);
    if (n < 1 || bucket < 1) throw std::invalid_argument("n and the bucket size must be positive");
    if ((uint64_t)n > MAX_LEAKY_PER_BATCH / (uint64_t)bucket)
        throw std::invalid_argument("too many leaky triples for one batch");
    const size_t total = n * bucket;

    BdozPair p = source.random_bdoz(2 * total + PAD);
    if (p.own.size() != 2 * total + PAD || p.keys.size() != 2 * total + PAD)
        throw std::logic_error("the bit source returned the wrong number of bits");
    mark_phase(meter, PH_ABIT);
    uint8_t* bits = p.own.bits.data();
    gf128* macs = p.own.macs.data();
    const gf128* keys = p.keys.keys.data();
    // y (rows total .. 2 total) and r (the last PAD rows): SPDZ-style MAC shares, in place.
    spdz_macs_inplace(delta, bits + total, macs + total, keys + total, total + PAD);

    mark_phase(meter, PH_LAAND);
    Shares z;
    leaky_ands(ch, sid, role, delta, index0, total, bits, macs, keys, macs + total, z, cheat);

    mark_phase(meter, PH_ABIT);
    spdz_macs_inplace(delta, bits, macs, keys, total);  // x, once H(M[x]) has been taken
    GfVec().swap(p.keys.keys);
    const ShareView x(bits, macs, total), y(bits + total, macs + total, total), r(bits + 2 * total, macs + 2 * total, PAD);

    mark_phase(meter, PH_BUCKET);  // the permutation is tossed only now: all leaky triples exist
    const hashes::seed16 coin = hashes::coin_toss_seed16(ch, sid, counter_label("bucket|", batch), rng);
    BucketOut bo;
    {
        const std::vector<uint32_t> perm = bucket_permutation(coin, total);
        bo = bucket_combine(ch, x, y, z, perm, bucket, cheat);
    }

    mark_phase(meter, PH_CHECK);
    triple_check(ch, sid, batch, role, delta, z, r, bo.d.data(), bo.d_macs.data(), bo.d.size(), rng, cheat);

    Triples out;
    out.a = std::move(bo.a);
    out.b = std::move(bo.b);
    out.c = std::move(bo.c);
    return out;
}

}  // namespace prep
}  // namespace couplet
