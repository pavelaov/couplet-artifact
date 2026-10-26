// couplet/u128.hpp -- 128-bit unsigned integer helpers.
//
// On compilers with native 128-bit support (GCC/Clang) `u128` is
// `unsigned __int128`.  On MSVC a tiny struct with the arithmetic needed by
// the SPDZ2k layer (add/sub/neg/mul/shift/mask) is provided, built on
// _umul128 / _addcarry_u64 / _subborrow_u64.  Generic code must go through
// the helper functions u128_from / u128_lo / u128_hi / u128_mask so that it
// compiles identically on both back ends.
#pragma once
#include <cstdint>
#include <string>

#if !defined(__SIZEOF_INT128__) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace couplet {

#if defined(__SIZEOF_INT128__)

typedef unsigned __int128 u128;

inline u128 u128_from(uint64_t lo, uint64_t hi) { return ((u128)hi << 64) | (u128)lo; }
inline uint64_t u128_lo(u128 x) { return (uint64_t)x; }
inline uint64_t u128_hi(u128 x) { return (uint64_t)(x >> 64); }

#else  // MSVC (or any compiler without __int128)

struct u128 {
    uint64_t lo, hi;
    u128() : lo(0), hi(0) {}
    u128(uint64_t v) : lo(v), hi(0) {}  // implicit, mirrors integer promotion
    u128(uint64_t l, uint64_t h) : lo(l), hi(h) {}
    explicit operator uint64_t() const { return lo; }
    explicit operator bool() const { return (lo | hi) != 0; }

    friend u128 operator+(u128 a, u128 b) {
        u128 r;
        unsigned char c = _addcarry_u64(0, a.lo, b.lo, &r.lo);
        _addcarry_u64(c, a.hi, b.hi, &r.hi);
        return r;
    }
    friend u128 operator-(u128 a, u128 b) {
        u128 r;
        unsigned char c = _subborrow_u64(0, a.lo, b.lo, &r.lo);
        _subborrow_u64(c, a.hi, b.hi, &r.hi);
        return r;
    }
    u128 operator-() const { return u128(0) - *this; }
    friend u128 operator*(u128 a, u128 b) {  // low 128 bits of the product
        u128 r;
        r.lo = _umul128(a.lo, b.lo, &r.hi);
        r.hi += a.lo * b.hi + a.hi * b.lo;
        return r;
    }
    friend u128 operator&(u128 a, u128 b) { return u128(a.lo & b.lo, a.hi & b.hi); }
    friend u128 operator|(u128 a, u128 b) { return u128(a.lo | b.lo, a.hi | b.hi); }
    friend u128 operator^(u128 a, u128 b) { return u128(a.lo ^ b.lo, a.hi ^ b.hi); }
    u128 operator~() const { return u128(~lo, ~hi); }
    friend u128 operator<<(u128 a, unsigned n) {
        if (n == 0) return a;
        if (n >= 128) return u128(0, 0);
        if (n >= 64) return u128(0, a.lo << (n - 64));
        return u128(a.lo << n, (a.hi << n) | (a.lo >> (64 - n)));
    }
    friend u128 operator>>(u128 a, unsigned n) {
        if (n == 0) return a;
        if (n >= 128) return u128(0, 0);
        if (n >= 64) return u128(a.hi >> (n - 64), 0);
        return u128((a.lo >> n) | (a.hi << (64 - n)), a.hi >> n);
    }
    friend bool operator==(u128 a, u128 b) { return a.lo == b.lo && a.hi == b.hi; }
    friend bool operator!=(u128 a, u128 b) { return !(a == b); }
    friend bool operator<(u128 a, u128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
    friend bool operator>(u128 a, u128 b) { return b < a; }
    friend bool operator<=(u128 a, u128 b) { return !(b < a); }
    friend bool operator>=(u128 a, u128 b) { return !(a < b); }
    u128& operator+=(u128 b) { return *this = *this + b; }
    u128& operator-=(u128 b) { return *this = *this - b; }
    u128& operator*=(u128 b) { return *this = *this * b; }
    u128& operator&=(u128 b) { return *this = *this & b; }
    u128& operator|=(u128 b) { return *this = *this | b; }
    u128& operator^=(u128 b) { return *this = *this ^ b; }
    u128& operator<<=(unsigned n) { return *this = *this << n; }
    u128& operator>>=(unsigned n) { return *this = *this >> n; }
};

inline u128 u128_from(uint64_t lo, uint64_t hi) { return u128(lo, hi); }
inline uint64_t u128_lo(u128 x) { return x.lo; }
inline uint64_t u128_hi(u128 x) { return x.hi; }

#endif

// 2^bits - 1 for bits in [0,128]; bits >= 128 gives all ones.
inline u128 u128_mask(unsigned bits) {
    if (bits >= 128) return ~u128(0);
    return (u128(1) << bits) - u128(1);
}

// Reduce modulo 2^bits (bits in [1,128]).
inline u128 u128_mod2k(u128 x, unsigned bits) { return x & u128_mask(bits); }

inline std::string u128_to_hex(u128 x) {  // 32 lowercase hex digits, most significant first
    static const char* d = "0123456789abcdef";
    std::string s(32, '0');
    uint64_t hi = u128_hi(x), lo = u128_lo(x);
    for (int i = 0; i < 16; i++) {
        s[15 - i] = d[(hi >> (4 * i)) & 15];
        s[31 - i] = d[(lo >> (4 * i)) & 15];
    }
    return s;
}

inline u128 u128_from_hex(const std::string& s) {  // accepts up to 32 hex digits
    u128 r = 0;
    for (char c : s) {
        unsigned v;
        if (c >= '0' && c <= '9') v = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v = (unsigned)(c - 'A' + 10);
        else continue;
        r = (r << 4) | u128(v);
    }
    return r;
}

// Little-endian 16-byte serialisation.
inline void u128_store_le(uint8_t out[16], u128 x) {
    uint64_t lo = u128_lo(x), hi = u128_hi(x);
    for (int i = 0; i < 8; i++) { out[i] = (uint8_t)(lo >> (8 * i)); out[8 + i] = (uint8_t)(hi >> (8 * i)); }
}
inline u128 u128_load_le(const uint8_t in[16]) {
    uint64_t lo = 0, hi = 0;
    for (int i = 7; i >= 0; i--) { lo = (lo << 8) | in[i]; hi = (hi << 8) | in[8 + i]; }
    return u128_from(lo, hi);
}

}  // namespace couplet
