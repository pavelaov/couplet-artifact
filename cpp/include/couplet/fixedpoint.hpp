// couplet/fixedpoint.hpp -- two's-complement fixed point of the Couplet layer
// (DESIGN_M3.md section 3); the C++ counterpart of python/couplet/fixedpoint.py.
//
// A wire format (n, f) represents a real v by the integer round(v 2^f) (half
// to even) as an n-bit two's-complement code.  Signed integers of up to 128
// bits are held in a u128 in two's complement ("i128 view"): addition,
// subtraction and multiplication modulo 2^128 are the integer operations as
// long as the true result fits 127 bits, which the bound checks of nn.hpp
// guarantee for every value of an inference.  Helpers:
//
//   from_i64(v)            the i128 view of a signed 64-bit integer
//   is_negative(x)         top bit of the i128 view
//   asr(x, sh)             floor(x / 2^sh) (arithmetic shift)
//   to_signed(code, bits)  code mod 2^bits read in two's complement, as int64
//                          (bits <= 64)
//   fits_signed(x, bits)   -2^(bits-1) <= x < 2^(bits-1) for an i128 view
//   embed(v, w)            v mod 2^w (sign-exact embedding), v an int64
//   truncate(p, f, n)      (p >> f) mod 2^n of an i128 view
//   round_half_even(x)     nearest integer, ties to even (Python's round)
//   Format                 the wire format (n, f)
#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "u128.hpp"

namespace couplet {
namespace fixedpoint {

inline u128 from_i64(int64_t v) {
    return v < 0 ? (u128(0) - u128((uint64_t)(-(v + 1)) + 1)) : u128((uint64_t)v);
}
inline bool is_negative(u128 x) { return (u128_hi(x) >> 63) != 0; }

// Arithmetic shift right of the i128 view (floor division by 2^sh).
inline u128 asr(u128 x, unsigned sh) {
    if (sh == 0) return x;
    if (sh >= 128) return is_negative(x) ? ~u128(0) : u128(0);
    const u128 r = x >> sh;
    return is_negative(x) ? (r | ~(~u128(0) >> sh)) : r;
}

// |x| of the i128 view (x != -2^127).
inline u128 abs128(u128 x) { return is_negative(x) ? u128(0) - x : x; }

// code mod 2^bits in two's complement (1 <= bits <= 64).
inline int64_t to_signed(u128 code, unsigned bits) {
    if (bits < 1 || bits > 64) throw std::invalid_argument("to_signed: bits must be in 1..64");
    const uint64_t u = u128_lo(code) & (bits == 64 ? ~(uint64_t)0 : (((uint64_t)1 << bits) - 1));
    if (bits < 64 && ((u >> (bits - 1)) & 1)) return (int64_t)(u | ~(((uint64_t)1 << bits) - 1));
    return (int64_t)u;
}

// -2^(bits-1) <= x < 2^(bits-1) for the i128 view x (1 <= bits <= 128).
inline bool fits_signed(u128 x, unsigned bits) {
    if (bits >= 128) return true;
    const u128 top = asr(x, bits - 1);
    return top == u128(0) || top == ~u128(0);
}
inline bool fits_signed64(int64_t v, unsigned bits) { return fits_signed(from_i64(v), bits); }

// Sign-exact embedding v mod 2^w; throws unless v fits w - 1 bits.
inline u128 embed(int64_t v, unsigned w) {
    if (!fits_signed64(v, w)) throw std::invalid_argument("value does not fit a sign-exact value of w bits");
    return from_i64(v) & u128_mask(w);
}

// The n-bit code of floor(p / 2^f): (p >> f) mod 2^n.
inline uint64_t truncate(u128 p, unsigned f, unsigned n) { return u128_lo(asr(p, f) & u128_mask(n)); }

// Nearest integer, ties to even.
inline int64_t round_half_even(double x) {
    const double r = std::nearbyint(x);  // default rounding mode: to nearest, ties to even
    return (int64_t)r;
}
inline int64_t encode(double x, unsigned f) { return round_half_even(x * std::ldexp(1.0, (int)f)); }
inline double decode(int64_t v, unsigned f) { return (double)v / std::ldexp(1.0, (int)f); }

// Wire format (n, f): n-bit codes with f fraction bits (n is the plane domain).
struct Format {
    unsigned n = 0, f = 0;
    Format() {}
    Format(unsigned n_, unsigned f_) : n(n_), f(f_) {
        if (n < 1 || n > 64) throw std::invalid_argument("need 1 <= n <= 64");
        if (f >= n) throw std::invalid_argument("need f < n");
    }
    uint64_t N() const { return n >= 64 ? 0 : (uint64_t)1 << n; }
    bool in_range(int64_t v) const { return fits_signed64(v, n); }
};

}  // namespace fixedpoint
}  // namespace couplet
