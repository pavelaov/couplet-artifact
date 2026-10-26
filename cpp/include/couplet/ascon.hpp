// couplet/ascon.hpp -- Ascon permutation p^12 (NIST SP 800-232) and the
// Couplet PRG maps G / Conv / ConvV_L / ConvV_R (DESIGN.md Section 2.1).
//
// State layout.  The 320-bit state is five 64-bit words x0..x4.  For the PRG
// the 16-byte seed is loaded little-endian into x0,x1 and the 24-byte
// zero-padded ASCII IV little-endian into x2,x3,x4.  The PRG output is the
// feed-forward construction (DESIGN.md v3.1, change A)
//
//     Y = F_IV(s) = p12(s || IV) XOR (s || IV)          (320 bits)
//
// and every map reads Y as the 40-byte little-endian concatenation
// LE(x0)||LE(x1)||...||LE(x4), i.e. bit j of Y is bit (j mod 64) of word
// x_{j/64}:
//   G:    sL = bytes [0,16) (x0x1), sR = bytes [16,32) (x2x3),
//         tL = bit 0 of x4, tR = bit 1 of x4.
//   Conv / ConvV_L / ConvV_R with w = k+s:
//         value = bits [0,w) of the 320-bit little-endian integer,
//         tag   = bits [w, 2w).
// prg_state() returns Y; prg_permute() returns p12(s || IV) alone (the
// "perm_state" field of the golden vectors; no PRG map reads it).
//
// Round function: add constant to x2; bitsliced 5-bit S-box with five
// ANDNOT operations; linear layer with rotations (19,28),(61,39),(1,6),
// (10,17),(7,41).  Round constants 0xf0,0xe1,...,0x4b.
//
// Besides the scalar p12 there are interleaved versions (4-way/8-way scalar
// template, and a 4-way AVX2 version when compiled with /arch:AVX2 or
// -mavx2) that run independent permutations in lock step for throughput.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "u128.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace couplet {
namespace ascon {

typedef std::array<uint8_t, 16> seed_t;

// ---------------------------------------------------------------- helpers
inline uint64_t load64le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
inline void store64le(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
inline uint64_t load64be(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | p[i];
    return v;
}
inline void store64be(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (56 - 8 * i));
}
inline uint64_t ror64(uint64_t x, unsigned n) { return (x >> n) | (x << (64 - n)); }  // 1 <= n <= 63

// ---------------------------------------------------------------- scalar p12
static const uint64_t RC[12] = {0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5,
                                0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b};

inline void round_fn(uint64_t s[5], uint64_t c) {
    uint64_t x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3], x4 = s[4];
    // constant addition
    x2 ^= c;
    // substitution layer (bitsliced 5-bit S-box, 5 ANDNOT ops)
    x0 ^= x4; x4 ^= x3; x2 ^= x1;
    uint64_t t0 = (~x0) & x1;
    uint64_t t1 = (~x1) & x2;
    uint64_t t2 = (~x2) & x3;
    uint64_t t3 = (~x3) & x4;
    uint64_t t4 = (~x4) & x0;
    x0 ^= t1; x1 ^= t2; x2 ^= t3; x3 ^= t4; x4 ^= t0;
    x1 ^= x0; x0 ^= x4; x3 ^= x2; x2 = ~x2;
    // linear diffusion layer
    x0 ^= ror64(x0, 19) ^ ror64(x0, 28);
    x1 ^= ror64(x1, 61) ^ ror64(x1, 39);
    x2 ^= ror64(x2, 1)  ^ ror64(x2, 6);
    x3 ^= ror64(x3, 10) ^ ror64(x3, 17);
    x4 ^= ror64(x4, 7)  ^ ror64(x4, 41);
    s[0] = x0; s[1] = x1; s[2] = x2; s[3] = x3; s[4] = x4;
}

inline void p12(uint64_t s[5]) {
    for (int r = 0; r < 12; r++) round_fn(s, RC[r]);
}

// ---------------------------------------------------------------- N-way scalar interleaving
// st[i][0..4] is the i-th state.  All N permutations advance in lock step so
// that the out-of-order core sees N independent dependency chains.
template <int N>
inline void p12_interleaved(uint64_t (*st)[5]) {
    uint64_t x0[N], x1[N], x2[N], x3[N], x4[N];
    for (int i = 0; i < N; i++) {
        x0[i] = st[i][0]; x1[i] = st[i][1]; x2[i] = st[i][2]; x3[i] = st[i][3]; x4[i] = st[i][4];
    }
    for (int r = 0; r < 12; r++) {
        const uint64_t c = RC[r];
        for (int i = 0; i < N; i++) x2[i] ^= c;
        for (int i = 0; i < N; i++) { x0[i] ^= x4[i]; x4[i] ^= x3[i]; x2[i] ^= x1[i]; }
        for (int i = 0; i < N; i++) {
            uint64_t t0 = (~x0[i]) & x1[i];
            uint64_t t1 = (~x1[i]) & x2[i];
            uint64_t t2 = (~x2[i]) & x3[i];
            uint64_t t3 = (~x3[i]) & x4[i];
            uint64_t t4 = (~x4[i]) & x0[i];
            x0[i] ^= t1; x1[i] ^= t2; x2[i] ^= t3; x3[i] ^= t4; x4[i] ^= t0;
        }
        for (int i = 0; i < N; i++) { x1[i] ^= x0[i]; x0[i] ^= x4[i]; x3[i] ^= x2[i]; x2[i] = ~x2[i]; }
        for (int i = 0; i < N; i++) {
            x0[i] ^= ror64(x0[i], 19) ^ ror64(x0[i], 28);
            x1[i] ^= ror64(x1[i], 61) ^ ror64(x1[i], 39);
            x2[i] ^= ror64(x2[i], 1)  ^ ror64(x2[i], 6);
            x3[i] ^= ror64(x3[i], 10) ^ ror64(x3[i], 17);
            x4[i] ^= ror64(x4[i], 7)  ^ ror64(x4[i], 41);
        }
    }
    for (int i = 0; i < N; i++) {
        st[i][0] = x0[i]; st[i][1] = x1[i]; st[i][2] = x2[i]; st[i][3] = x3[i]; st[i][4] = x4[i];
    }
}

inline void p12_x4(uint64_t st[4][5]) { p12_interleaved<4>(st); }
inline void p12_x8(uint64_t st[8][5]) { p12_interleaved<8>(st); }

// ---------------------------------------------------------------- 4-way AVX2
#if defined(__AVX2__)
#define COUPLET_HAVE_AVX2 1
namespace detail {
inline __m256i ror256(__m256i x, int n) {
    return _mm256_or_si256(_mm256_srli_epi64(x, n), _mm256_slli_epi64(x, 64 - n));
}
}  // namespace detail

// Each 64-bit lane j of register w holds word w of state j.
inline void p12_x4_avx2(uint64_t st[4][5]) {
    __m256i x0 = _mm256_set_epi64x((long long)st[3][0], (long long)st[2][0], (long long)st[1][0], (long long)st[0][0]);
    __m256i x1 = _mm256_set_epi64x((long long)st[3][1], (long long)st[2][1], (long long)st[1][1], (long long)st[0][1]);
    __m256i x2 = _mm256_set_epi64x((long long)st[3][2], (long long)st[2][2], (long long)st[1][2], (long long)st[0][2]);
    __m256i x3 = _mm256_set_epi64x((long long)st[3][3], (long long)st[2][3], (long long)st[1][3], (long long)st[0][3]);
    __m256i x4 = _mm256_set_epi64x((long long)st[3][4], (long long)st[2][4], (long long)st[1][4], (long long)st[0][4]);
    const __m256i ones = _mm256_set1_epi64x(-1);
    for (int r = 0; r < 12; r++) {
        x2 = _mm256_xor_si256(x2, _mm256_set1_epi64x((long long)RC[r]));
        x0 = _mm256_xor_si256(x0, x4); x4 = _mm256_xor_si256(x4, x3); x2 = _mm256_xor_si256(x2, x1);
        __m256i t0 = _mm256_andnot_si256(x0, x1);  // ~x0 & x1
        __m256i t1 = _mm256_andnot_si256(x1, x2);
        __m256i t2 = _mm256_andnot_si256(x2, x3);
        __m256i t3 = _mm256_andnot_si256(x3, x4);
        __m256i t4 = _mm256_andnot_si256(x4, x0);
        x0 = _mm256_xor_si256(x0, t1); x1 = _mm256_xor_si256(x1, t2); x2 = _mm256_xor_si256(x2, t3);
        x3 = _mm256_xor_si256(x3, t4); x4 = _mm256_xor_si256(x4, t0);
        x1 = _mm256_xor_si256(x1, x0); x0 = _mm256_xor_si256(x0, x4); x3 = _mm256_xor_si256(x3, x2);
        x2 = _mm256_xor_si256(x2, ones);
        x0 = _mm256_xor_si256(x0, _mm256_xor_si256(detail::ror256(x0, 19), detail::ror256(x0, 28)));
        x1 = _mm256_xor_si256(x1, _mm256_xor_si256(detail::ror256(x1, 61), detail::ror256(x1, 39)));
        x2 = _mm256_xor_si256(x2, _mm256_xor_si256(detail::ror256(x2, 1),  detail::ror256(x2, 6)));
        x3 = _mm256_xor_si256(x3, _mm256_xor_si256(detail::ror256(x3, 10), detail::ror256(x3, 17)));
        x4 = _mm256_xor_si256(x4, _mm256_xor_si256(detail::ror256(x4, 7),  detail::ror256(x4, 41)));
    }
    alignas(32) uint64_t tmp[5][4];
    _mm256_store_si256((__m256i*)tmp[0], x0);
    _mm256_store_si256((__m256i*)tmp[1], x1);
    _mm256_store_si256((__m256i*)tmp[2], x2);
    _mm256_store_si256((__m256i*)tmp[3], x3);
    _mm256_store_si256((__m256i*)tmp[4], x4);
    for (int j = 0; j < 4; j++)
        for (int w = 0; w < 5; w++) st[j][w] = tmp[w][j];
}
#else
#define COUPLET_HAVE_AVX2 0
#endif

// Best available 4-way permutation.
inline void p12_x4_fast(uint64_t st[4][5]) {
#if COUPLET_HAVE_AVX2
    p12_x4_avx2(st);
#else
    p12_x4(st);
#endif
}

// ---------------------------------------------------------------- PRG (DESIGN.md 2.1)
struct IV {
    uint64_t w[3];
};

// ASCII string zero-padded to 24 bytes, loaded little-endian into x2,x3,x4.
inline IV make_iv(const char* ascii) {
    uint8_t buf[24];
    std::memset(buf, 0, sizeof buf);
    size_t n = std::strlen(ascii);
    if (n > 24) throw std::invalid_argument("IV string longer than 24 bytes");
    std::memcpy(buf, ascii, n);
    IV iv;
    for (int i = 0; i < 3; i++) iv.w[i] = load64le(buf + 8 * i);
    return iv;
}

inline const IV& IV_G()  { static const IV iv = make_iv("Couplet-G");  return iv; }
inline const IV& IV_C()  { static const IV iv = make_iv("Couplet-C");  return iv; }
inline const IV& IV_VL() { static const IV iv = make_iv("Couplet-VL"); return iv; }
inline const IV& IV_VR() { static const IV iv = make_iv("Couplet-VR"); return iv; }

// Initial state (s || IV): seed little-endian into x0,x1; IV into x2,x3,x4.
inline void load_prg_state(uint64_t s[5], const uint8_t seed[16], const IV& iv) {
    s[0] = load64le(seed);
    s[1] = load64le(seed + 8);
    s[2] = iv.w[0];
    s[3] = iv.w[1];
    s[4] = iv.w[2];
}

// p12(seed || IV) WITHOUT the feed-forward.  Not a PRG output: it exists for
// the "perm_state" field of the golden vectors and for tests.
inline void prg_permute(uint64_t s[5], const uint8_t seed[16], const IV& iv) {
    load_prg_state(s, seed, iv);
    p12(s);
}

// Y = F_IV(seed) = p12(seed || IV) XOR (seed || IV): the 320-bit PRG output
// that G, Conv, ConvV_L and ConvV_R read.  The seed is given as its two
// words (lo, hi) = (x0, x1).
inline void prg_state_words(uint64_t y[5], uint64_t lo, uint64_t hi, const IV& iv) {
    y[0] = lo; y[1] = hi; y[2] = iv.w[0]; y[3] = iv.w[1]; y[4] = iv.w[2];
    p12(y);
    y[0] ^= lo; y[1] ^= hi; y[2] ^= iv.w[0]; y[3] ^= iv.w[1]; y[4] ^= iv.w[2];
}

// The same for a 16-byte seed.
inline void prg_state(uint64_t y[5], const uint8_t seed[16], const IV& iv) {
    prg_state_words(y, load64le(seed), load64le(seed + 8), iv);
}

// Two evaluations in lock step (2-way scalar interleaving):
// y[0] = F_iv_a(lo_a, hi_a), y[1] = F_iv_b(lo_b, hi_b).
inline void prg_state_words_x2(uint64_t y[2][5], uint64_t lo_a, uint64_t hi_a, const IV& iv_a, uint64_t lo_b,
                               uint64_t hi_b, const IV& iv_b) {
    y[0][0] = lo_a; y[0][1] = hi_a; y[0][2] = iv_a.w[0]; y[0][3] = iv_a.w[1]; y[0][4] = iv_a.w[2];
    y[1][0] = lo_b; y[1][1] = hi_b; y[1][2] = iv_b.w[0]; y[1][3] = iv_b.w[1]; y[1][4] = iv_b.w[2];
    p12_interleaved<2>(y);
    y[0][0] ^= lo_a; y[0][1] ^= hi_a; y[0][2] ^= iv_a.w[0]; y[0][3] ^= iv_a.w[1]; y[0][4] ^= iv_a.w[2];
    y[1][0] ^= lo_b; y[1][1] ^= hi_b; y[1][2] ^= iv_b.w[0]; y[1][3] ^= iv_b.w[1]; y[1][4] ^= iv_b.w[2];
}

// One PRG evaluation of a batch: Y = F_iv(seed).
struct PrgCall {
    const uint8_t* seed;  // 16 bytes
    const IV* iv;
};

// y[i] = F_{calls[i].iv}(calls[i].seed) for i < count.  Independent
// permutations run in lock step: groups of four through the best 4-way
// permutation, a remaining pair through the 2-way scalar interleaving.
inline void prg_state_batch(const PrgCall* calls, size_t count, uint64_t (*y)[5]) {
    uint64_t in[4][5];
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        for (int j = 0; j < 4; j++) {
            load_prg_state(in[j], calls[i + j].seed, *calls[i + j].iv);
            std::memcpy(y[i + j], in[j], sizeof in[j]);
        }
        p12_x4_fast(y + i);
        for (int j = 0; j < 4; j++)
            for (int w = 0; w < 5; w++) y[i + j][w] ^= in[j][w];
    }
    if (i + 2 <= count) {
        for (int j = 0; j < 2; j++) {
            load_prg_state(in[j], calls[i + j].seed, *calls[i + j].iv);
            std::memcpy(y[i + j], in[j], sizeof in[j]);
        }
        p12_interleaved<2>(y + i);
        for (int j = 0; j < 2; j++)
            for (int w = 0; w < 5; w++) y[i + j][w] ^= in[j][w];
        i += 2;
    }
    if (i < count) prg_state(y[i], calls[i].seed, *calls[i].iv);
}

// Four evaluations with one IV, array-of-states form: y[j] = F_iv(in[j]).
inline void prg_state_x4(uint64_t y[4][5], const seed_t in[4], const IV& iv) {
    const PrgCall calls[4] = {{in[0].data(), &iv}, {in[1].data(), &iv}, {in[2].data(), &iv}, {in[3].data(), &iv}};
    prg_state_batch(calls, 4, y);
}

#if COUPLET_HAVE_AVX2
namespace detail {
// The twelve rounds on five registers (lane j of register w = word w of
// state j).  Same round function as p12_x4_avx2, without the conversion from
// and to the array-of-states layout; used by the bulk PRG below.
inline void p12_regs_avx2(__m256i& r0, __m256i& r1, __m256i& r2, __m256i& r3, __m256i& r4) {
    __m256i x0 = r0, x1 = r1, x2 = r2, x3 = r3, x4 = r4;
    const __m256i ones = _mm256_set1_epi64x(-1);
    for (int r = 0; r < 12; r++) {
        x2 = _mm256_xor_si256(x2, _mm256_set1_epi64x((long long)RC[r]));
        x0 = _mm256_xor_si256(x0, x4); x4 = _mm256_xor_si256(x4, x3); x2 = _mm256_xor_si256(x2, x1);
        __m256i t0 = _mm256_andnot_si256(x0, x1);  // ~x0 & x1
        __m256i t1 = _mm256_andnot_si256(x1, x2);
        __m256i t2 = _mm256_andnot_si256(x2, x3);
        __m256i t3 = _mm256_andnot_si256(x3, x4);
        __m256i t4 = _mm256_andnot_si256(x4, x0);
        x0 = _mm256_xor_si256(x0, t1); x1 = _mm256_xor_si256(x1, t2); x2 = _mm256_xor_si256(x2, t3);
        x3 = _mm256_xor_si256(x3, t4); x4 = _mm256_xor_si256(x4, t0);
        x1 = _mm256_xor_si256(x1, x0); x0 = _mm256_xor_si256(x0, x4); x3 = _mm256_xor_si256(x3, x2);
        x2 = _mm256_xor_si256(x2, ones);
        x0 = _mm256_xor_si256(x0, _mm256_xor_si256(ror256(x0, 19), ror256(x0, 28)));
        x1 = _mm256_xor_si256(x1, _mm256_xor_si256(ror256(x1, 61), ror256(x1, 39)));
        x2 = _mm256_xor_si256(x2, _mm256_xor_si256(ror256(x2, 1),  ror256(x2, 6)));
        x3 = _mm256_xor_si256(x3, _mm256_xor_si256(ror256(x3, 10), ror256(x3, 17)));
        x4 = _mm256_xor_si256(x4, _mm256_xor_si256(ror256(x4, 7),  ror256(x4, 41)));
    }
    r0 = x0; r1 = x1; r2 = x2; r3 = x3; r4 = x4;
}
}  // namespace detail

// Four evaluations with one IV on registers: lane j of (s_lo, s_hi) is seed
// j, lane j of y_w receives word w of Y_j = F_iv(seed_j).
inline void prg_state_x4_regs(__m256i s_lo, __m256i s_hi, const IV& iv, __m256i& y0, __m256i& y1, __m256i& y2,
                              __m256i& y3, __m256i& y4) {
    const __m256i i2 = _mm256_set1_epi64x((long long)iv.w[0]);
    const __m256i i3 = _mm256_set1_epi64x((long long)iv.w[1]);
    const __m256i i4 = _mm256_set1_epi64x((long long)iv.w[2]);
    __m256i x0 = s_lo, x1 = s_hi, x2 = i2, x3 = i3, x4 = i4;
    detail::p12_regs_avx2(x0, x1, x2, x3, x4);
    y0 = _mm256_xor_si256(x0, s_lo);
    y1 = _mm256_xor_si256(x1, s_hi);
    y2 = _mm256_xor_si256(x2, i2);
    y3 = _mm256_xor_si256(x3, i3);
    y4 = _mm256_xor_si256(x4, i4);
}
#endif

// Four evaluations with one IV, structure-of-arrays form (the layout of a
// tree level in EvalAll): seed j is the word pair (lo[j], hi[j]) = (x0, x1)
// and y[w][j] receives word w of Y_j = F_iv(seed_j).
inline void prg_state_x4_soa(const uint64_t lo[4], const uint64_t hi[4], const IV& iv, uint64_t y[5][4]) {
#if COUPLET_HAVE_AVX2
    __m256i y0, y1, y2, y3, y4;
    prg_state_x4_regs(_mm256_loadu_si256((const __m256i*)lo), _mm256_loadu_si256((const __m256i*)hi), iv, y0, y1,
                      y2, y3, y4);
    _mm256_storeu_si256((__m256i*)y[0], y0);
    _mm256_storeu_si256((__m256i*)y[1], y1);
    _mm256_storeu_si256((__m256i*)y[2], y2);
    _mm256_storeu_si256((__m256i*)y[3], y3);
    _mm256_storeu_si256((__m256i*)y[4], y4);
#else
    uint64_t st[4][5];
    for (int j = 0; j < 4; j++) {
        st[j][0] = lo[j]; st[j][1] = hi[j]; st[j][2] = iv.w[0]; st[j][3] = iv.w[1]; st[j][4] = iv.w[2];
    }
    p12_x4(st);
    for (int j = 0; j < 4; j++) {
        y[0][j] = st[j][0] ^ lo[j];
        y[1][j] = st[j][1] ^ hi[j];
        y[2][j] = st[j][2] ^ iv.w[0];
        y[3][j] = st[j][3] ^ iv.w[1];
        y[4][j] = st[j][4] ^ iv.w[2];
    }
#endif
}

// Bits [off, off+w) of the 320-bit little-endian integer formed by the state
// (w <= 128, off + w <= 320).
inline u128 extract_bits(const uint64_t st[5], unsigned off, unsigned w) {
    if (w > 128 || off + w > 320) throw std::invalid_argument("extract_bits: range out of bounds");
    unsigned wi = off / 64, bo = off % 64;
    uint64_t a = st[wi];
    uint64_t b = (wi + 1 < 5) ? st[wi + 1] : 0;
    uint64_t c = (wi + 2 < 5) ? st[wi + 2] : 0;
    uint64_t lo, hi;
    if (bo == 0) { lo = a; hi = b; }
    else {
        lo = (a >> bo) | (b << (64 - bo));
        hi = (b >> bo) | (c << (64 - bo));
    }
    u128 r = u128_from(lo, hi);
    if (w < 128) r = r & u128_mask(w);
    return r;
}

struct GOut {
    seed_t sL;
    seed_t sR;
    uint8_t tL;
    uint8_t tR;
};

inline GOut G_from_state(const uint64_t s[5]) {
    GOut o;
    store64le(o.sL.data(), s[0]);
    store64le(o.sL.data() + 8, s[1]);
    store64le(o.sR.data(), s[2]);
    store64le(o.sR.data() + 8, s[3]);
    o.tL = (uint8_t)(s[4] & 1);
    o.tR = (uint8_t)((s[4] >> 1) & 1);
    return o;
}

// G(seed) -> (sL, tL, sR, tR), read from Y = F_G(seed).
inline GOut G(const seed_t& seed) {
    uint64_t y[5];
    prg_state(y, seed.data(), IV_G());
    return G_from_state(y);
}
inline void G(const uint8_t seed[16], uint8_t sL[16], uint8_t& tL, uint8_t sR[16], uint8_t& tR) {
    uint64_t y[5];
    prg_state(y, seed, IV_G());
    store64le(sL, y[0]); store64le(sL + 8, y[1]);
    store64le(sR, y[2]); store64le(sR + 8, y[3]);
    tL = (uint8_t)(y[4] & 1);
    tR = (uint8_t)((y[4] >> 1) & 1);
}

// Four G evaluations in lock step (uses the best 4-way permutation).
inline void G_x4(const seed_t in[4], GOut out[4]) {
    uint64_t y[4][5];
    prg_state_x4(y, in, IV_G());
    for (int j = 0; j < 4; j++) out[j] = G_from_state(y[j]);
}

struct ConvOut {
    u128 val;  // in Z_{2^{k+s}}
    u128 tag;  // in Z_{2^{k+s}}
};

// (value, tag) = bits [0,ks) and [ks,2ks) of a PRG output Y.  ks = k+s must
// be in [1,128] here (the u128 container); the specification allows up to
// 160, which is not needed for k <= 64, s = 64.
inline ConvOut conv_from_state(const uint64_t y[5], unsigned ks) {
    if (ks == 0 || ks > 128) throw std::invalid_argument("conv: k+s must be in [1,128]");
    ConvOut o;
    o.val = extract_bits(y, 0, ks);
    o.tag = extract_bits(y, ks, ks);
    return o;
}

// Generic convert map: value / tag words of Y = F_iv(seed).
inline ConvOut conv_with_iv(const seed_t& seed, const IV& iv, unsigned ks) {
    if (ks == 0 || ks > 128) throw std::invalid_argument("conv: k+s must be in [1,128]");
    uint64_t y[5];
    prg_state(y, seed.data(), iv);
    return conv_from_state(y, ks);
}

inline ConvOut Conv(const seed_t& seed, unsigned k, unsigned s)    { return conv_with_iv(seed, IV_C(),  k + s); }
inline ConvOut ConvV_L(const seed_t& seed, unsigned k, unsigned s) { return conv_with_iv(seed, IV_VL(), k + s); }
inline ConvOut ConvV_R(const seed_t& seed, unsigned k, unsigned s) { return conv_with_iv(seed, IV_VR(), k + s); }

// Four Conv-type evaluations in lock step.
inline void conv_x4(const seed_t in[4], const IV& iv, unsigned ks, ConvOut out[4]) {
    if (ks == 0 || ks > 128) throw std::invalid_argument("conv: k+s must be in [1,128]");
    uint64_t y[4][5];
    prg_state_x4(y, in, iv);
    for (int j = 0; j < 4; j++) out[j] = conv_from_state(y[j], ks);
}

}  // namespace ascon
}  // namespace couplet
