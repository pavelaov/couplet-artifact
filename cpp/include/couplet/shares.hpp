// couplet/shares.hpp -- data model of the Boolean preprocessing and the
// batched MAC check (DESIGN.md Section 2.2).  Everything here is the view of
// ONE party; the definitions mirror the Python modules couplet/prep/shares.py
// and couplet/prep/check.py.
//
// Layout.  Bits are one byte each (values 0 / 1) in a BitVec; MACs and keys
// are elements of GF(2^128) (gf128 = two 64-bit words) in a GfVec.  The
// 16-byte wire form of an element is its little-endian integer.
//
//   Shares   SPDZ-style authenticated bits <x>: this party's XOR shares `bits`
//            and MAC shares `macs`.  Across the two parties
//                bits_0 XOR bits_1 = x ,   macs_0 XOR macs_1 = x * Delta^B ,
//            Delta^B = Delta^B_0 XOR Delta^B_1.  XOR, slicing, selection and
//            concatenation are local; a public constant c is held as
//            (c, c Delta^B_0) by P_0 and as (0, c Delta^B_1) by P_1.
//   Bdoz     BDOZ bits of this party under the peer's key: own bits x and
//            MACs M[x] = K[x] XOR x Delta^B_peer.
//   Keys     the peer's BDOZ bits seen from the key holder: K[x'].
//   ShareView  non-owning (bits, macs, n) view used by the bulk routines.
//
// Conversions of Section 2.2: spdz_shares (one BDOZ bit of each party -> a
// random <x>), mask_owner / mask_other (a BDOZ bit of the owner -> an input
// mask <r>).
//
// BoolCheck (bool_check).  A bit <y_j> has been opened to the public value
// y~_j; party b computes V_{j,b} = m_{j,b} XOR y~_j Delta^B_b, and for an
// opened field element W with MAC share M_b, V_b = M_b XOR W Delta^B_b.  The
// two parties hold equal V values exactly when every opening was correct;
// otherwise they differ by Delta^B.  Each party hashes its whole list
//     SHA-256(frame("Couplet-boolcheck", sid, le64(counter),
//                   V of the bits, V of the field elements))
// and the digests are compared by a simultaneous commit-then-open
// (hashes::committed_exchange, label "boolcheck|" || le64(counter)).  This is
// the refinement of the linear-combination check of DESIGN.md 2.2 recorded in
// design/CHANGELOG.md (PY-C item 1): two rounds, 96 bytes per party, no coin
// toss.  A mismatch throws ProtocolAbort; the caller retires the session
// (change G).
//
// Also here: the wire encoding of bit vectors, GF(2^128) vector helpers
// (inner products with lazy reduction, Horner evaluation), the per-phase
// accounting (Phase, PhaseStat, PrepStats, PhaseMeter), the test hooks
// (Cheat) and the interface BdozSource implemented by the real and the ideal
// sources of authenticated bits.
//
// Bulk kernels shared by the protocol headers (namespace detail):
//   prg_words / ccr_hash   the functions hashes::ctr_prg and hashes::H_ccr
//       with four Ascon permutations per set of AVX2 registers (the portable
//       routines of hashes.hpp are used without AVX2 and for the tails);
//   parallel_ranges / parallel_chunks, WorkerScope   optional worker threads
//       inside one party (default: none);
//   FastSha256   SHA-256 with the SHA extensions when the processor has them
//       (detected at run time); the digest is the standard one either way.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "channel.hpp"
#include "gf128.hpp"
#include "hashes.hpp"
#include "params.hpp"
#include "rng.hpp"
#include "sha256.hpp"

#if (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))) || \
    (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__)))
#define COUPLET_PREP_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define COUPLET_PREP_X86 0
#endif

#if defined(_MSC_VER) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define COUPLET_PREP_LE 1
#else
#define COUPLET_PREP_LE 0
#endif

#if defined(__GNUC__) && COUPLET_PREP_X86
#define COUPLET_PREP_TARGET(x) __attribute__((target(x)))
#else
#define COUPLET_PREP_TARGET(x)
#endif

namespace couplet {
namespace prep {

typedef std::vector<uint8_t> BitVec;  // one bit per byte, values 0 / 1
typedef std::vector<gf128> GfVec;     // elements of GF(2^128)

static_assert(sizeof(gf128) == 16, "gf128 must be two 64-bit words");

inline void check_role(int role) {
    if (role != 0 && role != 1) throw std::invalid_argument("role must be 0 or 1");
}

// ------------------------------------------------------------------ words and bytes
// n 64-bit words <-> 8 n little-endian bytes.
inline void store_words_le(uint8_t* out, const uint64_t* w, size_t n) {
#if COUPLET_PREP_LE
    if (n) std::memcpy(out, w, 8 * n);
#else
    for (size_t i = 0; i < n; i++) ascon::store64le(out + 8 * i, w[i]);
#endif
}
inline void load_words_le(uint64_t* w, const uint8_t* in, size_t n) {
#if COUPLET_PREP_LE
    if (n) std::memcpy(w, in, 8 * n);
#else
    for (size_t i = 0; i < n; i++) w[i] = ascon::load64le(in + 8 * i);
#endif
}

// n field elements <-> 16 n bytes (each element little-endian: lo, then hi).
inline void gf_store(uint8_t* out, const gf128* v, size_t n) {
#if COUPLET_PREP_LE
    if (n) std::memcpy(out, v, 16 * n);
#else
    for (size_t i = 0; i < n; i++) v[i].to_bytes(out + 16 * i);
#endif
}
inline void gf_load(gf128* v, const uint8_t* in, size_t n) {
#if COUPLET_PREP_LE
    if (n) std::memcpy(v, in, 16 * n);
#else
    for (size_t i = 0; i < n; i++) v[i] = gf128::from_bytes(in + 16 * i);
#endif
}
inline bytes gf_to_bytes(const gf128* v, size_t n) {
    bytes out(16 * n);
    gf_store(out.data(), v, n);
    return out;
}
inline bytes gf_to_bytes(const GfVec& v) { return gf_to_bytes(v.data(), v.size()); }
inline bytes gf_to_bytes(const gf128& a) { return gf_to_bytes(&a, 1); }

// ------------------------------------------------------------------ worker threads
// The bulk kernels of the preprocessing (column expansion, row pass, leaky-AND
// hashing, bucketing) can split their work over several threads of ONE party.
// The number of workers is a per-thread setting (default 1 = everything runs
// in the calling thread); it is set for a scope by WorkerScope and by
// RealPrep::set_threads.  The results do not depend on it.
namespace detail {

inline unsigned& worker_count() {
    static thread_local unsigned n = 1;
    return n;
}

// Splits [0, n) into at most worker_count() contiguous ranges and calls
// fn(begin, end, part) for each; all boundaries except n are multiples of
// `unit`, and no range is shorter than `min_per_part` (so small jobs stay in
// the calling thread).  Part 0 runs in the calling thread, the others in
// std::threads; an exception thrown by a part is rethrown in the caller after
// all parts have finished.  Returns the number of parts used.
template <class F>
inline unsigned parallel_ranges(size_t n, size_t unit, size_t min_per_part, F&& fn) {
    if (unit == 0) unit = 1;
    const size_t units = (n + unit - 1) / unit;
    size_t min_units = (min_per_part + unit - 1) / unit;
    if (min_units == 0) min_units = 1;
    size_t parts = worker_count();
    if (parts > units / min_units) parts = units / min_units;
    if (parts <= 1) {
        fn((size_t)0, n, 0u);
        return 1;
    }
    const size_t per = ((units + parts - 1) / parts) * unit;
    std::vector<std::exception_ptr> errors(parts);
    auto run = [&](unsigned part) {
        const size_t begin = per * part;
        const size_t end = (n - begin < per) ? n : begin + per;
        try {
            fn(begin, end, part);
        } catch (...) {
            errors[part] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(parts - 1);
    try {
        for (unsigned p = 1; p < (unsigned)parts; p++) {
            if (per * p >= n) break;
            pool.emplace_back(run, p);
        }
    } catch (...) {  // a thread could not be started: wait for the others, then report
        for (std::thread& t : pool) t.join();
        throw;
    }
    run(0);
    for (std::thread& t : pool) t.join();
    for (const std::exception_ptr& e : errors)
        if (e) std::rethrow_exception(e);
    return (unsigned)parts;
}

// Like parallel_ranges, but the range is cut into chunks (about eight per
// worker, none shorter than `min_chunk`, boundaries on multiples of `unit`)
// that the workers take one after another, so a slow core does not hold the
// others back.  fn(begin, end, worker) may be called several times with the
// same worker index (below worker_count()), never concurrently for one index.
template <class F>
inline unsigned parallel_chunks(size_t n, size_t unit, size_t min_chunk, F&& fn) {
    if (unit == 0) unit = 1;
    const size_t units = (n + unit - 1) / unit;
    size_t min_units = (min_chunk + unit - 1) / unit;
    if (min_units == 0) min_units = 1;
    size_t workers = worker_count();
    if (workers > units / min_units) workers = units / min_units;
    if (workers <= 1) {
        fn((size_t)0, n, 0u);
        return 1;
    }
    size_t chunk_units = units / (workers * 8);
    if (chunk_units < min_units) chunk_units = min_units;
    const size_t chunk = chunk_units * unit;
    const size_t n_chunks = (n + chunk - 1) / chunk;
    std::atomic<size_t> next(0);
    std::vector<std::exception_ptr> errors(workers);
    auto run = [&](unsigned worker) {
        try {
            for (;;) {
                const size_t c = next.fetch_add(1);
                if (c >= n_chunks) break;
                const size_t begin = c * chunk;
                fn(begin, (n - begin < chunk) ? n : begin + chunk, worker);
            }
        } catch (...) {
            errors[worker] = std::current_exception();
            next.store(n_chunks);  // the other workers stop after their current chunk
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    try {
        for (unsigned w = 1; w < (unsigned)workers; w++) pool.emplace_back(run, w);
    } catch (...) {
        next.store(n_chunks);
        for (std::thread& t : pool) t.join();
        throw;
    }
    run(0);
    for (std::thread& t : pool) t.join();
    for (const std::exception_ptr& e : errors)
        if (e) std::rethrow_exception(e);
    return (unsigned)workers;
}

}  // namespace detail

// Sets the number of worker threads of the calling thread for a scope.
class WorkerScope {
public:
    explicit WorkerScope(unsigned workers) : prev_(detail::worker_count()) { detail::worker_count() = workers ? workers : 1; }
    ~WorkerScope() { detail::worker_count() = prev_; }
    WorkerScope(const WorkerScope&) = delete;
    WorkerScope& operator=(const WorkerScope&) = delete;

private:
    unsigned prev_;
};

// ------------------------------------------------------------------ bulk Ascon kernels
// The same functions as hashes::ctr_prg and hashes::H_ccr_batch (the tests
// compare them).  With AVX2 four permutations run in one set of registers,
// without the detour through an array of states.
namespace detail {

#if COUPLET_HAVE_AVX2
inline __m256i ror64x4(__m256i x, int n) { return _mm256_or_si256(_mm256_srli_epi64(x, n), _mm256_slli_epi64(x, 64 - n)); }

// Ascon-p12 on four states: lane j of register w holds word w of state j.
// The complement of x2 at the end of a round is folded into the constant
// addition of the next round (the linear layer of x2 commutes with it).
inline void p12x4(__m256i& r0, __m256i& r1, __m256i& r2, __m256i& r3, __m256i& r4) {
    static const uint64_t RC[12] = {0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b};
    __m256i x0 = r0, x1 = r1, x2 = r2, x3 = r3, x4 = r4;
    for (int r = 0; r < 12; r++) {
        x2 = _mm256_xor_si256(x2, _mm256_set1_epi64x((long long)(r == 0 ? RC[0] : ~RC[r])));
        x0 = _mm256_xor_si256(x0, x4);
        x4 = _mm256_xor_si256(x4, x3);
        x2 = _mm256_xor_si256(x2, x1);
        const __m256i t0 = _mm256_andnot_si256(x0, x1);  // ~x0 & x1
        const __m256i t1 = _mm256_andnot_si256(x1, x2);
        const __m256i t2 = _mm256_andnot_si256(x2, x3);
        const __m256i t3 = _mm256_andnot_si256(x3, x4);
        const __m256i t4 = _mm256_andnot_si256(x4, x0);
        x0 = _mm256_xor_si256(x0, t1);
        x1 = _mm256_xor_si256(x1, t2);
        x2 = _mm256_xor_si256(x2, t3);
        x3 = _mm256_xor_si256(x3, t4);
        x4 = _mm256_xor_si256(x4, t0);
        x1 = _mm256_xor_si256(x1, x0);
        x0 = _mm256_xor_si256(x0, x4);
        x3 = _mm256_xor_si256(x3, x2);  // (x2 is complemented later, see above)
        x0 = _mm256_xor_si256(x0, _mm256_xor_si256(ror64x4(x0, 19), ror64x4(x0, 28)));
        x1 = _mm256_xor_si256(x1, _mm256_xor_si256(ror64x4(x1, 61), ror64x4(x1, 39)));
        x2 = _mm256_xor_si256(x2, _mm256_xor_si256(ror64x4(x2, 1), ror64x4(x2, 6)));
        x3 = _mm256_xor_si256(x3, _mm256_xor_si256(ror64x4(x3, 10), ror64x4(x3, 17)));
        x4 = _mm256_xor_si256(x4, _mm256_xor_si256(ror64x4(x4, 7), ror64x4(x4, 41)));
    }
    r0 = x0;
    r1 = x1;
    r2 = _mm256_xor_si256(x2, _mm256_set1_epi64x(-1));
    r3 = x3;
    r4 = x4;
}
#endif

// Stream words [3 first_block, 3 first_block + n_words) of
// hashes::ctr_prg(seed, dom), seed = (k0, k1) as two little-endian words.
inline void prg_words(uint64_t k0, uint64_t k1, uint64_t dom, uint64_t* out, size_t n_words, uint64_t first_block = 0) {
    hashes::dom::check(dom);
    size_t b = 0;
#if COUPLET_HAVE_AVX2
    const size_t full = n_words / 3;
    const __m256i a0 = _mm256_set1_epi64x((long long)k0), a1 = _mm256_set1_epi64x((long long)k1);
    const __m256i a3 = _mm256_set1_epi64x((long long)dom), four = _mm256_set1_epi64x(4);
    __m256i ctr = _mm256_add_epi64(_mm256_set1_epi64x((long long)first_block), _mm256_set_epi64x(3, 2, 1, 0));
    alignas(32) uint64_t t[12];
    for (; b + 4 <= full; b += 4) {
        __m256i x0 = a0, x1 = a1, x2 = ctr, x3 = a3, x4 = _mm256_setzero_si256();
        p12x4(x0, x1, x2, x3, x4);
        ctr = _mm256_add_epi64(ctr, four);
        _mm256_store_si256((__m256i*)t, x0);
        _mm256_store_si256((__m256i*)(t + 4), x1);
        _mm256_store_si256((__m256i*)(t + 8), x2);
        uint64_t* o = out + 3 * b;
        o[0] = t[0]; o[1] = t[4]; o[2] = t[8];
        o[3] = t[1]; o[4] = t[5]; o[5] = t[9];
        o[6] = t[2]; o[7] = t[6]; o[8] = t[10];
        o[9] = t[3]; o[10] = t[7]; o[11] = t[11];
    }
#endif
    if (3 * b < n_words) {  // the remaining blocks through the reference routine
        uint8_t seed[16];
        ascon::store64le(seed, k0);
        ascon::store64le(seed + 8, k1);
        hashes::ctr_prg(seed, dom, out + 3 * b, n_words - 3 * b, first_block + (uint64_t)b);
    }
}
inline void prg_words(const gf128& seed, uint64_t dom, uint64_t* out, size_t n_words, uint64_t first_block = 0) {
    prg_words(seed.lo, seed.hi, dom, out, n_words, first_block);
}
inline void prg_words(const hashes::seed16& seed, uint64_t dom, uint64_t* out, size_t n_words, uint64_t first_block = 0) {
    prg_words(ascon::load64le(seed.data()), ascon::load64le(seed.data() + 8), dom, out, n_words, first_block);
}

// out[i] = H_ccr(x[i] XOR mask, tweak0 + i, dom) for i < n.  out may alias x.
inline void ccr_hash(const gf128* x, size_t n, uint64_t tweak0, uint64_t dom, gf128* out, const gf128& mask = gf128()) {
    hashes::dom::check(dom);
    size_t i = 0;
#if COUPLET_HAVE_AVX2
    const __m256i m_lo = _mm256_set1_epi64x((long long)mask.lo), m_hi = _mm256_set1_epi64x((long long)mask.hi);
    const __m256i d = _mm256_set1_epi64x((long long)dom), four = _mm256_set1_epi64x(4);
    // the lanes hold the elements in the order 0, 2, 1, 3 (what unpacklo / unpackhi produce)
    __m256i tw = _mm256_add_epi64(_mm256_set1_epi64x((long long)tweak0), _mm256_set_epi64x(3, 1, 2, 0));
    for (; i + 4 <= n; i += 4) {
        const __m256i p = _mm256_loadu_si256((const __m256i*)(x + i));      // lo0 hi0 lo1 hi1
        const __m256i q = _mm256_loadu_si256((const __m256i*)(x + i + 2));  // lo2 hi2 lo3 hi3
        __m256i x0 = _mm256_xor_si256(_mm256_unpacklo_epi64(p, q), m_lo);   // lo0 lo2 lo1 lo3
        __m256i x1 = _mm256_xor_si256(_mm256_unpackhi_epi64(p, q), m_hi);   // hi0 hi2 hi1 hi3
        __m256i x2 = tw, x3 = d, x4 = _mm256_setzero_si256();
        p12x4(x0, x1, x2, x3, x4);
        tw = _mm256_add_epi64(tw, four);
        _mm256_storeu_si256((__m256i*)(out + i), _mm256_unpacklo_epi64(x0, x1));      // elements 0, 1
        _mm256_storeu_si256((__m256i*)(out + i + 2), _mm256_unpackhi_epi64(x0, x1));  // elements 2, 3
    }
#endif
    for (; i < n; i++) out[i] = hashes::H_ccr(x[i] ^ mask, tweak0 + (uint64_t)i, dom);
}

}  // namespace detail

// ------------------------------------------------------------------ GF(2^128) helpers
// a if bit = 1, zero otherwise (no branch).
inline gf128 gf_and_bit(const gf128& a, unsigned bit) {
    const uint64_t m = 0 - (uint64_t)(bit & 1);
    return gf128(a.lo & m, a.hi & m);
}

// pack(b_0 .. b_127) = sum_j b_j x^j.
inline gf128 gf_pack_bits(const uint8_t* bits128) {
    uint64_t lo = 0, hi = 0;
    for (unsigned j = 0; j < 64; j++) {
        lo |= (uint64_t)(bits128[j] & 1) << j;
        hi |= (uint64_t)(bits128[64 + j] & 1) << j;
    }
    return gf128(lo, hi);
}

// The 128 coefficients of a (bit j at index j).
inline void gf_unpack_bits(const gf128& a, uint8_t* bits128) {
    for (unsigned j = 0; j < 64; j++) {
        bits128[j] = (uint8_t)((a.lo >> j) & 1);
        bits128[64 + j] = (uint8_t)((a.hi >> j) & 1);
    }
}
inline BitVec gf_unpack_bits(const gf128& a) {
    BitVec b(128);
    gf_unpack_bits(a, b.data());
    return b;
}

// sum_j x^j * parts_j for at most 128 parts (Horner pass).
inline gf128 gf_horner(const gf128* parts, size_t n) {
    if (n > 128) throw std::invalid_argument("gf_horner: at most 128 parts");
    gf128 acc;
    for (size_t j = n; j-- > 0;) acc = gf128_mulx(acc) ^ parts[j];
    return acc;
}

// Inner product sum_i a_i * b_i over GF(2^128).  With PCLMUL the 256-bit
// carry-less products are accumulated and reduced once at the end.
inline gf128 gf_dot(const gf128* a, const gf128* b, size_t n) {
#if COUPLET_HAVE_PCLMUL
    __m128i lo = _mm_setzero_si128(), mid = _mm_setzero_si128(), hi = _mm_setzero_si128();
    for (size_t i = 0; i < n; i++) {
        const __m128i x = _mm_set_epi64x((long long)a[i].hi, (long long)a[i].lo);
        const __m128i y = _mm_set_epi64x((long long)b[i].hi, (long long)b[i].lo);
        lo = _mm_xor_si128(lo, _mm_clmulepi64_si128(x, y, 0x00));
        hi = _mm_xor_si128(hi, _mm_clmulepi64_si128(x, y, 0x11));
        mid = _mm_xor_si128(mid, _mm_xor_si128(_mm_clmulepi64_si128(x, y, 0x10), _mm_clmulepi64_si128(x, y, 0x01)));
    }
    lo = _mm_xor_si128(lo, _mm_slli_si128(mid, 8));
    hi = _mm_xor_si128(hi, _mm_srli_si128(mid, 8));
    return couplet::detail::gf128_from_m128(couplet::detail::gf128_reduce(lo, hi));
#else
    gf128 acc;
    for (size_t i = 0; i < n; i++) acc ^= gf128_mul(a[i], b[i]);
    return acc;
#endif
}

// sum_i bits_i * chi_i: the XOR of the elements selected by a 0/1 vector.
inline gf128 gf_dot_bits(const gf128* chi, const uint8_t* bits, size_t n) {
    uint64_t lo = 0, hi = 0;
    for (size_t i = 0; i < n; i++) {
        const uint64_t m = 0 - (uint64_t)(bits[i] & 1);
        lo ^= chi[i].lo & m;
        hi ^= chi[i].hi & m;
    }
    return gf128(lo, hi);
}

// ------------------------------------------------------------------ bit vectors
inline void check_bits(const uint8_t* bits, size_t n) {
    uint8_t acc = 0;
    for (size_t i = 0; i < n; i++) acc |= bits[i];
    if (acc > 1) throw std::invalid_argument("bits must be 0 or 1");
}
inline void check_bits(const BitVec& bits) { check_bits(bits.data(), bits.size()); }

// Bit vector -> ceil(n / 8) bytes, bit i in bit (i mod 8) of byte floor(i / 8).
inline bytes pack_bit_msg(const uint8_t* bits, size_t n) {
    bytes out((n + 7) / 8, 0);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        unsigned v = 0;
        for (unsigned k = 0; k < 8; k++) v |= (unsigned)(bits[i + k] & 1) << k;
        out[i / 8] = (uint8_t)v;
    }
    for (; i < n; i++) out[i / 8] = (uint8_t)(out[i / 8] | ((bits[i] & 1) << (i % 8)));
    return out;
}
inline bytes pack_bit_msg(const BitVec& bits) { return pack_bit_msg(bits.data(), bits.size()); }

// Inverse of pack_bit_msg for a received message; ProtocolAbort if the length
// is wrong or a padding bit is set.
inline BitVec unpack_bit_msg(const bytes& data, size_t n) {
    if (data.size() != (n + 7) / 8) throw ProtocolAbort("bit message has the wrong length");
    BitVec bits(n);
    for (size_t i = 0; i < n; i++) bits[i] = (uint8_t)((data[i / 8] >> (i % 8)) & 1);
    if (n % 8 && (data[n / 8] >> (n % 8)) != 0) throw ProtocolAbort("bit message has stray padding bits");
    return bits;
}

// Send own bits and receive the peer's equally long vector (one round).
inline BitVec exchange_bits(Channel& ch, const uint8_t* bits, size_t n) {
    return unpack_bit_msg(ch.exchange(pack_bit_msg(bits, n), (n + 7) / 8, "bit message has the wrong length"), n);
}
inline BitVec exchange_bits(Channel& ch, const BitVec& bits) { return exchange_bits(ch, bits.data(), bits.size()); }

// ------------------------------------------------------------------ SPDZ-style shares
struct Shares;

// Non-owning view of n authenticated bits.
struct ShareView {
    const uint8_t* bits;
    const gf128* macs;
    size_t n;
    ShareView() : bits(nullptr), macs(nullptr), n(0) {}
    ShareView(const uint8_t* b, const gf128* m, size_t count) : bits(b), macs(m), n(count) {}
    ShareView(const Shares& s);  // implicit
    size_t size() const { return n; }
};

struct Shares {
    BitVec bits;
    GfVec macs;

    Shares() {}
    explicit Shares(size_t n) : bits(n, 0), macs(n) {}
    Shares(BitVec b, GfVec m) : bits(std::move(b)), macs(std::move(m)) { validate(); }

    size_t size() const { return bits.size(); }
    bool empty() const { return bits.empty(); }

    void validate() const {
        if (bits.size() != macs.size()) throw std::invalid_argument("bits and macs must have the same length");
        check_bits(bits);
    }

    static Shares zeros(size_t n) { return Shares(n); }

    // Shares at positions [a, b).
    Shares slice(size_t a, size_t b) const {
        if (a > b || b > size()) throw std::out_of_range("Shares::slice");
        Shares r;
        r.bits.assign(bits.begin() + (ptrdiff_t)a, bits.begin() + (ptrdiff_t)b);
        r.macs.assign(macs.begin() + (ptrdiff_t)a, macs.begin() + (ptrdiff_t)b);
        return r;
    }

    // Shares at the positions idx (repetitions allowed).
    Shares take(const std::vector<size_t>& idx) const {
        Shares r(idx.size());
        for (size_t i = 0; i < idx.size(); i++) {
            if (idx[i] >= size()) throw std::out_of_range("Shares::take");
            r.bits[i] = bits[idx[i]];
            r.macs[i] = macs[idx[i]];
        }
        return r;
    }

    Shares& operator^=(const Shares& o) {
        if (o.size() != size()) throw std::invalid_argument("share vectors must have the same length");
        for (size_t i = 0; i < bits.size(); i++) {
            bits[i] ^= o.bits[i];
            macs[i] ^= o.macs[i];
        }
        return *this;
    }
    friend Shares operator^(Shares a, const Shares& b) {
        a ^= b;
        return a;
    }

    void append(const ShareView& o) {
        bits.insert(bits.end(), o.bits, o.bits + o.n);
        macs.insert(macs.end(), o.macs, o.macs + o.n);
    }
    static Shares concat(const std::vector<ShareView>& parts) {
        Shares r;
        for (const ShareView& p : parts) r.append(p);
        return r;
    }

    // Shares of public bits c: P_0 holds (c, c Delta^B_0), P_1 holds (0, c Delta^B_1).
    static Shares constant(int role, const gf128& delta, const BitVec& consts) {
        check_role(role);
        check_bits(consts);
        Shares r(consts.size());
        for (size_t i = 0; i < consts.size(); i++) {
            r.bits[i] = role == 0 ? consts[i] : (uint8_t)0;
            r.macs[i] = gf_and_bit(delta, consts[i]);
        }
        return r;
    }

    // Shares of x XOR c for public bits c (one per position).
    Shares add_const(int role, const gf128& delta, const BitVec& consts) const {
        if (consts.size() != size()) throw std::invalid_argument("one constant per share is required");
        Shares r = constant(role, delta, consts);
        r ^= *this;
        return r;
    }

    // Shares of NOT x.
    Shares invert(int role, const gf128& delta) const { return add_const(role, delta, BitVec(size(), 1)); }

    // Shares of c AND x for public bits c.
    Shares mul_const(const BitVec& consts) const {
        if (consts.size() != size()) throw std::invalid_argument("one constant per share is required");
        check_bits(consts);
        Shares r(size());
        for (size_t i = 0; i < size(); i++) {
            r.bits[i] = (uint8_t)(bits[i] & consts[i]);
            r.macs[i] = gf_and_bit(macs[i], consts[i]);
        }
        return r;
    }

    // Shares of the element sum_j x^j x_j of GF(2^128) (exactly 128 bits):
    // (value share, MAC share).  Across the parties the value shares XOR to
    // R = sum_j x^j x_j and the MAC shares to R * Delta^B.
    std::pair<gf128, gf128> to_field() const {
        if (size() != 128) throw std::invalid_argument("a field element needs exactly 128 authenticated bits");
        return std::make_pair(gf_pack_bits(bits.data()), gf_horner(macs.data(), 128));
    }
};

inline ShareView::ShareView(const Shares& s) : bits(s.bits.data()), macs(s.macs.data()), n(s.size()) {}

// (value share, MAC share) of sum_j x^j <r_j> for a view of exactly 128 bits.
inline std::pair<gf128, gf128> to_field(const ShareView& r) {
    if (r.n != 128) throw std::invalid_argument("a field element needs exactly 128 authenticated bits");
    return std::make_pair(gf_pack_bits(r.bits), gf_horner(r.macs, 128));
}

// ------------------------------------------------------------------ BDOZ material
// Own bits with their MACs under the peer's key: M[x] = K[x] XOR x Delta^B_peer.
struct Bdoz {
    BitVec bits;
    GfVec macs;

    Bdoz() {}
    explicit Bdoz(size_t n) : bits(n, 0), macs(n) {}
    Bdoz(BitVec b, GfVec m) : bits(std::move(b)), macs(std::move(m)) {
        if (bits.size() != macs.size()) throw std::invalid_argument("bits and macs must have the same length");
        check_bits(bits);
    }
    size_t size() const { return bits.size(); }
    Bdoz slice(size_t a, size_t b) const {
        if (a > b || b > size()) throw std::out_of_range("Bdoz::slice");
        Bdoz r;
        r.bits.assign(bits.begin() + (ptrdiff_t)a, bits.begin() + (ptrdiff_t)b);
        r.macs.assign(macs.begin() + (ptrdiff_t)a, macs.begin() + (ptrdiff_t)b);
        return r;
    }
    void append(const Bdoz& o) {
        bits.insert(bits.end(), o.bits.begin(), o.bits.end());
        macs.insert(macs.end(), o.macs.begin(), o.macs.end());
    }
};

// MAC keys K[x'] of this party for the peer's BDOZ bits x'.
struct Keys {
    GfVec keys;

    Keys() {}
    explicit Keys(size_t n) : keys(n) {}
    explicit Keys(GfVec k) : keys(std::move(k)) {}
    size_t size() const { return keys.size(); }
    Keys slice(size_t a, size_t b) const {
        if (a > b || b > size()) throw std::out_of_range("Keys::slice");
        return Keys(GfVec(keys.begin() + (ptrdiff_t)a, keys.begin() + (ptrdiff_t)b));
    }
    void append(const Keys& o) { keys.insert(keys.end(), o.keys.begin(), o.keys.end()); }
};

struct BdozPair {
    Bdoz own;   // own bits and their MACs under the peer's key
    Keys keys;  // keys for the peer's bits
};

// ------------------------------------------------------------------ conversions (Section 2.2)
// In place: macs[i] <- macs[i] XOR keys[i] XOR bits[i] * delta, i.e. the
// SPDZ-style MAC share m_b = M[x_b] XOR K[x_{1-b}] XOR x_b Delta^B_b.
inline void spdz_macs_inplace(const gf128& delta, const uint8_t* bits, gf128* macs, const gf128* keys, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const uint64_t m = 0 - (uint64_t)(bits[i] & 1);
        macs[i].lo ^= keys[i].lo ^ (delta.lo & m);
        macs[i].hi ^= keys[i].hi ^ (delta.hi & m);
    }
}

// Random <x> with x = x_0 XOR x_1 from one BDOZ bit of each party: party b
// holds own = (x_b, M[x_b]) and peer = K[x_{1-b}].
inline Shares spdz_shares(int role, const gf128& delta, const Bdoz& own, const Keys& peer) {
    check_role(role);
    if (own.size() != peer.size()) throw std::invalid_argument("own bits and peer keys must have the same length");
    Shares s;
    s.bits = own.bits;
    s.macs = own.macs;
    spdz_macs_inplace(delta, s.bits.data(), s.macs.data(), peer.keys.data(), s.size());
    return s;
}

// Input mask <r> at its owner: share r, MAC share M[r] XOR r Delta^B_owner.
inline Shares mask_owner(const gf128& delta, const Bdoz& own) {
    Shares s;
    s.bits = own.bits;
    s.macs = own.macs;
    for (size_t i = 0; i < s.size(); i++) s.macs[i] ^= gf_and_bit(delta, s.bits[i]);
    return s;
}

// Input mask <r> at the other party: share 0, MAC share K[r].
inline Shares mask_other(const Keys& peer) {
    Shares s;
    s.bits.assign(peer.size(), 0);
    s.macs = peer.keys;
    return s;
}

// ------------------------------------------------------------------ results of the preprocessing
struct Triples {
    Shares a, b, c;  // c = a AND b
    size_t size() const { return a.size(); }
};

struct InputMasks {
    Shares shares;          // <r>
    BitVec bits;            // r, at the owner only
    bool has_bits = false;  // true at the owner
};

// Source of random BDOZ bits (the real COT pipeline or the ideal dealer).
class BdozSource {
public:
    virtual ~BdozSource() {}
    // n random BDOZ bits of each party: own bits with MACs, and keys for the peer's bits.
    virtual BdozPair random_bdoz(size_t n) = 0;
};

// ------------------------------------------------------------------ test hooks
// Deviations injected by the tests (default: none).  The honest code path
// only tests the pointer and the flags.
struct Cheat {
    typedef std::function<bytes(const bytes&)> Rewrite;
    Rewrite flight1, flight2;                           // base OT: replace an outgoing flight
    std::vector<std::pair<unsigned, size_t>> flip_u;    // OT extension: flip bit (column, row) of U
    bool has_g_error = false;                           // leaky AND (Lemma 3): add g_err[i] to the transfer
    std::vector<size_t> g_idx;                          //   message of triple g_idx[i], assuming the peer's
    GfVec g_err;                                        //   share x of that triple equals g_guess[i]
    BitVec g_guess;
    std::vector<size_t> flip_d;                         // bucketing: flip own shares of opened d bits
    bool has_w_error = false;                           // triple check: add w_error to the own share of W
    gf128 w_error;
};

// ------------------------------------------------------------------ accounting
enum Phase { PH_NONE = -1, PH_BASEOT = 0, PH_COTEXT, PH_ABIT, PH_LAAND, PH_BUCKET, PH_CHECK, PH_COUNT };

inline const char* phase_name(int phase) {
    static const char* names[PH_COUNT] = {"baseot", "cotext", "abit", "laand", "bucket", "check"};
    return (phase >= 0 && phase < PH_COUNT) ? names[phase] : "(none)";
}

struct PhaseStat {
    uint64_t bytes_sent = 0;
    uint64_t bytes_recv = 0;
    uint64_t msgs_sent = 0;
    uint64_t msgs_recv = 0;
    uint64_t rounds = 0;
    double seconds = 0.0;  // wall-clock time of this party in the phase (includes waiting for the peer)

    PhaseStat& operator+=(const PhaseStat& o) {
        bytes_sent += o.bytes_sent;
        bytes_recv += o.bytes_recv;
        msgs_sent += o.msgs_sent;
        msgs_recv += o.msgs_recv;
        rounds += o.rounds;
        seconds += o.seconds;
        return *this;
    }
    std::string json() const {
        char sec[64];
        std::snprintf(sec, sizeof sec, "%.6f", seconds);
        return "{\"bytes_sent\":" + std::to_string(bytes_sent) + ",\"bytes_recv\":" + std::to_string(bytes_recv) +
               ",\"msgs_sent\":" + std::to_string(msgs_sent) + ",\"msgs_recv\":" + std::to_string(msgs_recv) +
               ",\"rounds\":" + std::to_string(rounds) + ",\"seconds\":" + sec + "}";
    }
};

// Bytes, rounds and seconds of one party, per phase of the preprocessing.
struct PrepStats {
    PhaseStat phase[PH_COUNT];

    const PhaseStat& operator[](int p) const { return phase[p]; }
    PhaseStat total() const {
        PhaseStat t;
        for (int p = 0; p < PH_COUNT; p++) t += phase[p];
        return t;
    }
    // Sum over the phases other than the base OT.
    PhaseStat total_without_baseot() const {
        PhaseStat t;
        for (int p = 0; p < PH_COUNT; p++)
            if (p != PH_BASEOT) t += phase[p];
        return t;
    }
    std::string json() const {
        std::string s = "{";
        for (int p = 0; p < PH_COUNT; p++) {
            if (p) s += ",";
            s += std::string("\"") + phase_name(p) + "\":" + phase[p].json();
        }
        return s + "}";
    }
};

// Attributes the traffic of a channel endpoint and the elapsed time to the
// phase that is current.  mark(p) closes the running phase and starts p;
// PH_NONE means "not ours" (nothing is booked).
class PhaseMeter {
public:
    explicit PhaseMeter(Channel& ch) : ch_(&ch), cur_(PH_NONE) { snap(); }

    int phase() const { return cur_; }

    // Start phase p; returns the phase that was current.
    int mark(int p) {
        close();
        const int prev = cur_;
        cur_ = p;
        return prev;
    }

    // Statistics so far (the running phase is included up to now).
    const PrepStats& stats() {
        close();
        return stats_;
    }

private:
    void snap() {
        snap_ = ch_->stats();
        t_ = std::chrono::steady_clock::now();
    }
    void close() {
        const ChannelStats now = ch_->stats();
        const std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
        if (cur_ >= 0 && cur_ < PH_COUNT) {
            PhaseStat& s = stats_.phase[cur_];
            s.bytes_sent += now.bytes_sent - snap_.bytes_sent;
            s.bytes_recv += now.bytes_recv - snap_.bytes_recv;
            s.msgs_sent += now.msgs_sent - snap_.msgs_sent;
            s.msgs_recv += now.msgs_recv - snap_.msgs_recv;
            s.rounds += now.rounds - snap_.rounds;
            s.seconds += std::chrono::duration<double>(t - t_).count();
        }
        snap_ = now;
        t_ = t;
    }

    Channel* ch_;
    PrepStats stats_;
    int cur_;
    ChannelStats snap_;
    std::chrono::steady_clock::time_point t_;
};

// mark() on an optional meter.
inline void mark_phase(PhaseMeter* meter, int phase) {
    if (meter) meter->mark(phase);
}

// ------------------------------------------------------------------ SHA-256 for bulk data
namespace detail {

#if COUPLET_PREP_X86
inline bool cpu_has_sha_ni() {
    static const bool has = [] {
#if defined(_MSC_VER)
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) return false;
        __cpuid(r, 1);
        const bool ssse3 = (r[2] & (1 << 9)) != 0, sse41 = (r[2] & (1 << 19)) != 0;
        __cpuidex(r, 7, 0);
        return ssse3 && sse41 && (r[1] & (1 << 29)) != 0;
#else
        unsigned a = 0, b = 0, c = 0, d = 0;
        if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
        const bool ssse3 = (c & (1u << 9)) != 0, sse41 = (c & (1u << 19)) != 0;
        if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
        return ssse3 && sse41 && (b & (1u << 29)) != 0;
#endif
    }();
    return has;
}

// SHA-256 compression of n 64-byte blocks with the SHA extensions.
// state = (a, b, c, d, e, f, g, h) as in FIPS 180-4.
COUPLET_PREP_TARGET("sha,sse4.1,ssse3")
inline void sha256_blocks_ni(uint32_t state[8], const uint8_t* data, size_t n) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    const __m128i bswap = _mm_set_epi64x(0x0c0d0e0f08090a0bll, 0x0405060700010203ll);

    __m128i tmp = _mm_loadu_si128((const __m128i*)&state[0]);   // d c b a (high .. low)
    __m128i st1 = _mm_loadu_si128((const __m128i*)&state[4]);   // h g f e
    tmp = _mm_shuffle_epi32(tmp, 0xb1);                         // c d a b
    st1 = _mm_shuffle_epi32(st1, 0x1b);                         // e f g h
    __m128i st0 = _mm_alignr_epi8(tmp, st1, 8);                 // a b e f
    st1 = _mm_blend_epi16(st1, tmp, 0xf0);                      // c d g h

    while (n--) {
        const __m128i save0 = st0, save1 = st1;
        __m128i w[4];
        for (int g = 0; g < 16; g++) {
            __m128i m;
            if (g < 4) {
                m = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data + 16 * g)), bswap);
                w[g] = m;
            } else {
                m = _mm_sha256msg1_epu32(w[0], w[1]);
                m = _mm_add_epi32(m, _mm_alignr_epi8(w[3], w[2], 4));
                m = _mm_sha256msg2_epu32(m, w[3]);
                w[0] = w[1];
                w[1] = w[2];
                w[2] = w[3];
                w[3] = m;
            }
            __m128i k = _mm_add_epi32(m, _mm_loadu_si128((const __m128i*)&K[4 * g]));
            st1 = _mm_sha256rnds2_epu32(st1, st0, k);
            k = _mm_shuffle_epi32(k, 0x0e);
            st0 = _mm_sha256rnds2_epu32(st0, st1, k);
        }
        st0 = _mm_add_epi32(st0, save0);
        st1 = _mm_add_epi32(st1, save1);
        data += 64;
    }

    tmp = _mm_shuffle_epi32(st0, 0x1b);        // f e b a
    st1 = _mm_shuffle_epi32(st1, 0xb1);        // d c h g
    st0 = _mm_blend_epi16(tmp, st1, 0xf0);     // d c b a
    st1 = _mm_alignr_epi8(st1, tmp, 8);        // h g f e
    _mm_storeu_si128((__m128i*)&state[0], st0);
    _mm_storeu_si128((__m128i*)&state[4], st1);
}
#endif  // COUPLET_PREP_X86

// Streaming SHA-256 (FIPS 180-4): the SHA extensions when available,
// otherwise the portable Sha256 class.  Same digest in both cases.
class FastSha256 {
public:
    // force_portable = true selects the portable implementation (tests).
    explicit FastSha256(bool force_portable = false) : ni_(false), buflen_(0), total_(0) {
#if COUPLET_PREP_X86
        ni_ = !force_portable && cpu_has_sha_ni();
#else
        (void)force_portable;
#endif
        static const uint32_t H0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(h_, H0, sizeof h_);
    }

    bool accelerated() const { return ni_; }

    void update(const uint8_t* data, size_t len) {
        if (len == 0) return;  // nothing to absorb; `data` may be a null pointer
        if (!ni_) {
            slow_.update(data, len);
            return;
        }
#if COUPLET_PREP_X86
        total_ += len;
        if (buflen_) {
            size_t take = 64 - buflen_;
            if (take > len) take = len;
            std::memcpy(buf_ + buflen_, data, take);
            buflen_ += take;
            data += take;
            len -= take;
            if (buflen_ == 64) {
                sha256_blocks_ni(h_, buf_, 1);
                buflen_ = 0;
            }
        }
        if (len >= 64) {
            const size_t blocks = len / 64;
            sha256_blocks_ni(h_, data, blocks);
            data += 64 * blocks;
            len -= 64 * blocks;
        }
        if (len) {
            std::memcpy(buf_, data, len);
            buflen_ = len;
        }
#endif
    }

    // One part of a framed hash: le64(len) || data (hashes.hpp, (c)).
    void update_framed(const uint8_t* data, size_t len) {
        const std::array<uint8_t, 8> l = hashes::le64((uint64_t)len);
        update(l.data(), 8);
        update(data, len);
    }
    // Only the length prefix of a part whose bytes follow through update().
    void update_frame_length(uint64_t len) {
        const std::array<uint8_t, 8> l = hashes::le64(len);
        update(l.data(), 8);
    }

    hashes::hash32 finish() {
        hashes::hash32 out;
        if (!ni_) {
            slow_.finish(out.data());
            return out;
        }
#if COUPLET_PREP_X86
        const uint64_t bit_len = total_ * 8;
        uint8_t pad[72];
        size_t pad_len = (buflen_ < 56) ? (56 - buflen_) : (120 - buflen_);
        std::memset(pad, 0, sizeof pad);
        pad[0] = 0x80;
        for (int i = 0; i < 8; i++) pad[pad_len + (size_t)i] = (uint8_t)(bit_len >> (56 - 8 * i));
        update(pad, pad_len + 8);
        for (int i = 0; i < 8; i++) {
            out[4 * (size_t)i + 0] = (uint8_t)(h_[i] >> 24);
            out[4 * (size_t)i + 1] = (uint8_t)(h_[i] >> 16);
            out[4 * (size_t)i + 2] = (uint8_t)(h_[i] >> 8);
            out[4 * (size_t)i + 3] = (uint8_t)(h_[i]);
        }
#endif
        return out;
    }

private:
    bool ni_;
    Sha256 slow_;
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t buflen_;
    uint64_t total_;
};

}  // namespace detail

// ------------------------------------------------------------------ BoolCheck
// An opened element W of GF(2^128) with this party's MAC share M of it.
struct FieldOpening {
    gf128 value;
    gf128 mac;
    FieldOpening() {}
    FieldOpening(const gf128& w, const gf128& m) : value(w), mac(m) {}
};

// V_j = m_j XOR y~_j * Delta^B_b for every opened bit.
inline GfVec check_values(const gf128& delta, const uint8_t* opened, const gf128* macs, size_t n) {
    GfVec v(n);
    for (size_t i = 0; i < n; i++) v[i] = macs[i] ^ gf_and_bit(delta, opened[i]);
    return v;
}
inline GfVec check_values(const gf128& delta, const BitVec& opened, const GfVec& macs) {
    if (opened.size() != macs.size()) throw std::invalid_argument("one MAC share per opened bit is required");
    return check_values(delta, opened.data(), macs.data(), macs.size());
}

// SHA-256 digest of this party's V values (bits first, then field elements):
//   SHA-256(frame("Couplet-boolcheck", sid, le64(counter), V bits, V field)).
inline hashes::hash32 check_digest(const bytes& sid, uint64_t counter, const gf128& delta, const uint8_t* opened,
                                   const gf128* macs, size_t n, const std::vector<FieldOpening>& extra) {
    static const char tag[] = "Couplet-boolcheck";
    check_bits(opened, n);
    detail::FastSha256 h;
    h.update_framed((const uint8_t*)tag, sizeof tag - 1);
    h.update_framed(sid.data(), sid.size());
    const std::array<uint8_t, 8> ctr = hashes::le64(counter);
    h.update_framed(ctr.data(), 8);

    h.update_frame_length(16 * (uint64_t)n);
    const size_t STEP = 2048;
    gf128 v[STEP];
    uint8_t buf[16 * STEP];
    for (size_t done = 0; done < n;) {
        const size_t take = (n - done < STEP) ? (n - done) : STEP;
        for (size_t i = 0; i < take; i++) {
            const uint64_t m = 0 - (uint64_t)(opened[done + i] & 1);
            v[i] = gf128(macs[done + i].lo ^ (delta.lo & m), macs[done + i].hi ^ (delta.hi & m));
        }
        gf_store(buf, v, take);
        h.update(buf, 16 * take);
        done += take;
    }

    h.update_frame_length(16 * (uint64_t)extra.size());
    for (const FieldOpening& e : extra) {
        const gf128 vf = e.mac ^ gf128_mul(e.value, delta);
        uint8_t b[16];
        vf.to_bytes(b);
        h.update(b, 16);
    }
    return h.finish();
}
inline hashes::hash32 check_digest(const bytes& sid, uint64_t counter, const gf128& delta, const BitVec& opened,
                                   const GfVec& macs, const std::vector<FieldOpening>& extra = {}) {
    if (opened.size() != macs.size()) throw std::invalid_argument("one MAC share per opened bit is required");
    return check_digest(sid, counter, delta, opened.data(), macs.data(), macs.size(), extra);
}

// The commitment label of BoolCheck number `counter`: "boolcheck|" || le64(counter).
inline bytes bool_check_label(uint64_t counter) {
    static const char prefix[] = "boolcheck|";
    bytes label(prefix, prefix + sizeof prefix - 1);
    const std::array<uint8_t, 8> c = hashes::le64(counter);
    label.insert(label.end(), c.begin(), c.end());
    return label;
}

// Run BoolCheck on everything opened since the last check; ProtocolAbort on
// failure.
//   ch       this party's channel endpoint (ch.party() must equal role);
//   sid      session identifier;
//   counter  number of this check within the session; both parties use the
//            same value and never reuse one under the same sid;
//   role     this party's index;
//   delta    this party's share Delta^B_role of the Boolean MAC key;
//   opened   the n public opened bits, macs this party's MAC shares of them;
//   extra    opened field elements with this party's MAC shares;
//   rng      randomness for the commitment (any object with fill()).
// Two rounds, 96 bytes per party.
template <class ByteSource>
inline void bool_check(Channel& ch, const bytes& sid, uint64_t counter, int role, const gf128& delta,
                       const uint8_t* opened, const gf128* macs, size_t n, const std::vector<FieldOpening>& extra,
                       ByteSource& rng) {
    check_role(role);
    if (ch.party() != role) throw std::invalid_argument("role does not match the channel endpoint");
    const hashes::hash32 digest = check_digest(sid, counter, delta, opened, macs, n, extra);
    const bytes label = bool_check_label(counter);
    const bytes peer = hashes::committed_exchange(ch, sid, label, digest, rng, digest.size());
    if (peer.size() != digest.size() || !hashes::ct_equal(peer.data(), digest.data(), digest.size()))
        throw ProtocolAbort("BoolCheck failed");
}
template <class ByteSource>
inline void bool_check(Channel& ch, const bytes& sid, uint64_t counter, int role, const gf128& delta,
                       const BitVec& opened, const GfVec& macs, const std::vector<FieldOpening>& extra,
                       ByteSource& rng) {
    if (opened.size() != macs.size()) throw std::invalid_argument("one MAC share per opened bit is required");
    bool_check(ch, sid, counter, role, delta, opened.data(), macs.data(), macs.size(), extra, rng);
}

}  // namespace prep
}  // namespace couplet
