// couplet/gf128.hpp -- GF(2^128) with the polynomial x^128 + x^7 + x^2 + x + 1.
//
// Representation: bit i (0 <= i < 128) is the coefficient of x^i; `lo` holds
// bits 0..63 and `hi` bits 64..127 (little-endian polynomial order, as used
// by KOS/TinyOT-style implementations).  The 16-byte serialisation is the
// little-endian integer lo | hi<<64.
//
// Multiplication uses PCLMULQDQ (_mm_clmulepi64_si128) when available and a
// portable shift-and-add fallback otherwise (define COUPLET_NO_PCLMUL to
// force the fallback).
#pragma once
#include <cstdint>
#include <cstring>
#include <string>

#if !defined(COUPLET_NO_PCLMUL) && \
    (defined(__PCLMUL__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64))))
#define COUPLET_HAVE_PCLMUL 1
#include <emmintrin.h>
#include <wmmintrin.h>
#else
#define COUPLET_HAVE_PCLMUL 0
#endif

namespace couplet {

struct gf128 {
    uint64_t lo, hi;
    gf128() : lo(0), hi(0) {}
    gf128(uint64_t l, uint64_t h) : lo(l), hi(h) {}
    static gf128 zero() { return gf128(0, 0); }
    static gf128 one() { return gf128(1, 0); }
    static gf128 x() { return gf128(2, 0); }  // the generator x

    bool is_zero() const { return (lo | hi) == 0; }
    friend bool operator==(const gf128& a, const gf128& b) { return a.lo == b.lo && a.hi == b.hi; }
    friend bool operator!=(const gf128& a, const gf128& b) { return !(a == b); }
    friend gf128 operator^(const gf128& a, const gf128& b) { return gf128(a.lo ^ b.lo, a.hi ^ b.hi); }
    friend gf128 operator+(const gf128& a, const gf128& b) { return a ^ b; }
    gf128& operator^=(const gf128& b) { lo ^= b.lo; hi ^= b.hi; return *this; }
    gf128& operator+=(const gf128& b) { return *this ^= b; }

    // bit access (coefficient of x^i)
    unsigned bit(unsigned i) const { return i < 64 ? (unsigned)((lo >> i) & 1) : (unsigned)((hi >> (i - 64)) & 1); }

    void to_bytes(uint8_t out[16]) const {
        for (int i = 0; i < 8; i++) { out[i] = (uint8_t)(lo >> (8 * i)); out[8 + i] = (uint8_t)(hi >> (8 * i)); }
    }
    static gf128 from_bytes(const uint8_t in[16]) {
        uint64_t l = 0, h = 0;
        for (int i = 7; i >= 0; i--) { l = (l << 8) | in[i]; h = (h << 8) | in[8 + i]; }
        return gf128(l, h);
    }
    std::string hex() const {  // 32 hex digits, most significant (x^127) first
        static const char* d = "0123456789abcdef";
        std::string s(32, '0');
        for (int i = 0; i < 16; i++) { s[15 - i] = d[(hi >> (4 * i)) & 15]; s[31 - i] = d[(lo >> (4 * i)) & 15]; }
        return s;
    }
};

// Multiply by x (shift left by one and reduce).
inline gf128 gf128_mulx(gf128 a) {
    uint64_t carry = a.hi >> 63;
    uint64_t hi = (a.hi << 1) | (a.lo >> 63);
    uint64_t lo = (a.lo << 1) ^ (0x87ull & (0 - carry));
    return gf128(lo, hi);
}

// Portable constant-time shift-and-add multiplication.
inline gf128 gf128_mul_portable(gf128 a, gf128 b) {
    uint64_t rlo = 0, rhi = 0;
    uint64_t alo = a.lo, ahi = a.hi;
    for (int i = 0; i < 128; i++) {
        uint64_t bit = (i < 64) ? ((b.lo >> i) & 1) : ((b.hi >> (i - 64)) & 1);
        uint64_t m = 0 - bit;
        rlo ^= alo & m;
        rhi ^= ahi & m;
        uint64_t carry = ahi >> 63;
        ahi = (ahi << 1) | (alo >> 63);
        alo = (alo << 1) ^ (0x87ull & (0 - carry));
    }
    return gf128(rlo, rhi);
}

#if COUPLET_HAVE_PCLMUL
namespace detail {
inline __m128i gf128_to_m128(gf128 a) { return _mm_set_epi64x((long long)a.hi, (long long)a.lo); }
inline gf128 gf128_from_m128(__m128i v) {
    alignas(16) uint64_t t[2];
    _mm_store_si128((__m128i*)t, v);
    return gf128(t[0], t[1]);
}

// Reduce the 256-bit carry-less product (hi:lo) modulo x^128 + x^7 + x^2 + x + 1.
// x^128 == x^7 + x^2 + x + 1 == 0x87, so hi*x^128 == hi*0x87, folded twice.
inline __m128i gf128_reduce(__m128i lo, __m128i hi) {
    const __m128i poly = _mm_set_epi64x(0, 0x87);
    __m128i t3 = _mm_clmulepi64_si128(hi, poly, 0x00);  // hi.lo * 0x87     -> bits [0,71)
    __m128i t1 = _mm_clmulepi64_si128(hi, poly, 0x01);  // hi.hi * 0x87     -> to be placed at bit 64
    __m128i t1s = _mm_slli_si128(t1, 8);                // low 64 bits of t1 -> bits [64,128)
    __m128i t1h = _mm_srli_si128(t1, 8);                // bits [64,71) of t1 = overflow above x^128
    __m128i t2 = _mm_clmulepi64_si128(t1h, poly, 0x00); // overflow * 0x87  -> bits [0,15)
    return _mm_xor_si128(_mm_xor_si128(lo, t1s), _mm_xor_si128(t2, t3));
}

inline __m128i gf128_mul_m128(__m128i a, __m128i b) {
    __m128i t3 = _mm_clmulepi64_si128(a, b, 0x00);  // a.lo * b.lo
    __m128i t4 = _mm_clmulepi64_si128(a, b, 0x10);  // a.lo * b.hi
    __m128i t5 = _mm_clmulepi64_si128(a, b, 0x01);  // a.hi * b.lo
    __m128i t6 = _mm_clmulepi64_si128(a, b, 0x11);  // a.hi * b.hi
    t4 = _mm_xor_si128(t4, t5);
    t5 = _mm_slli_si128(t4, 8);
    t4 = _mm_srli_si128(t4, 8);
    __m128i lo = _mm_xor_si128(t3, t5);
    __m128i hi = _mm_xor_si128(t6, t4);
    return gf128_reduce(lo, hi);
}
}  // namespace detail

inline gf128 gf128_mul_clmul(gf128 a, gf128 b) {
    return detail::gf128_from_m128(detail::gf128_mul_m128(detail::gf128_to_m128(a), detail::gf128_to_m128(b)));
}
#endif

inline gf128 gf128_mul(gf128 a, gf128 b) {
#if COUPLET_HAVE_PCLMUL
    return gf128_mul_clmul(a, b);
#else
    return gf128_mul_portable(a, b);
#endif
}

inline gf128 operator*(const gf128& a, const gf128& b) { return gf128_mul(a, b); }

inline gf128 gf128_sqr(gf128 a) { return gf128_mul(a, a); }

// a^(2^128 - 2) = a^{-1} for a != 0 (returns 0 for a = 0).
inline gf128 gf128_inv(gf128 a) {
    // 2^128 - 2 = sum_{i=1}^{127} 2^i
    gf128 result = gf128::one();
    gf128 sq = a;
    for (int i = 1; i <= 127; i++) {
        sq = gf128_sqr(sq);
        result = gf128_mul(result, sq);
    }
    return result;
}

// Inner product sum_i a[i]*b[i] (used for MAC-check linear combinations).
inline gf128 gf128_dot(const gf128* a, const gf128* b, size_t n) {
    gf128 acc;
    for (size_t i = 0; i < n; i++) acc ^= gf128_mul(a[i], b[i]);
    return acc;
}

// Multiply by a single bit (constant-time select).
inline gf128 gf128_mul_bit(gf128 a, unsigned bit) {
    uint64_t m = 0 - (uint64_t)(bit & 1);
    return gf128(a.lo & m, a.hi & m);
}

}  // namespace couplet
