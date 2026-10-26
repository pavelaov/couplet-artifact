// couplet/x25519.hpp -- X25519 (RFC 7748) scalar multiplication.
//
// A faithful transcription of TweetNaCl's crypto_scalarmult (public domain):
// radix-2^16 limbs in int64 (gf = int64[16]), car25519 carry propagation,
// sel25519 constant-time swap, pack25519/unpack25519, A/Z/M/S field ops,
// inv25519 by exponentiation, and the Montgomery ladder with the standard
// clamping (z[0] &= 248; z[31] = (z[31] & 127) | 64).
//
// Also provides hash_to_x(bytes32) = SHA-256(bytes32) with the top bit of
// byte 31 cleared (the H2G map of the Masny-Rindal base OT, DESIGN.md 7.1).
#pragma once
#include <array>
#include <cstdint>

#include "sha256.hpp"

namespace couplet {
namespace x25519 {

typedef int64_t i64;
typedef i64 gf[16];
typedef std::array<uint8_t, 32> bytes32;

namespace detail {

static const gf _121665 = {0xDB41, 1};
static const uint8_t _9[32] = {9};

inline void car25519(gf o) {
    int i;
    i64 c;
    for (i = 0; i < 16; i++) {
        o[i] += (1LL << 16);
        c = o[i] >> 16;
        o[(i + 1) * (i < 15)] += c - 1 + 37 * (c - 1) * (i == 15);
        o[i] -= c * 65536;  // c may be negative: a multiplication, not a left shift (undefined before C++20)
    }
}

inline void sel25519(gf p, gf q, int b) {
    i64 t, i, c = ~(i64)(b - 1);
    for (i = 0; i < 16; i++) {
        t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

inline void pack25519(uint8_t* o, const gf n) {
    int i, j, b;
    gf m, t;
    for (i = 0; i < 16; i++) t[i] = n[i];
    car25519(t);
    car25519(t);
    car25519(t);
    for (j = 0; j < 2; j++) {
        m[0] = t[0] - 0xffed;
        for (i = 1; i < 15; i++) {
            m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
            m[i - 1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        b = (int)((m[15] >> 16) & 1);
        m[14] &= 0xffff;
        sel25519(t, m, 1 - b);
    }
    for (i = 0; i < 16; i++) {
        o[2 * i] = (uint8_t)(t[i] & 0xff);
        o[2 * i + 1] = (uint8_t)(t[i] >> 8);
    }
}

inline void unpack25519(gf o, const uint8_t* n) {
    int i;
    for (i = 0; i < 16; i++) o[i] = n[2 * i] + ((i64)n[2 * i + 1] << 8);
    o[15] &= 0x7fff;
}

inline void A(gf o, const gf a, const gf b) {
    for (int i = 0; i < 16; i++) o[i] = a[i] + b[i];
}

inline void Z(gf o, const gf a, const gf b) {
    for (int i = 0; i < 16; i++) o[i] = a[i] - b[i];
}

inline void M(gf o, const gf a, const gf b) {
    i64 i, j, t[31];
    for (i = 0; i < 31; i++) t[i] = 0;
    for (i = 0; i < 16; i++)
        for (j = 0; j < 16; j++) t[i + j] += a[i] * b[j];
    for (i = 0; i < 15; i++) t[i] += 38 * t[i + 16];
    for (i = 0; i < 16; i++) o[i] = t[i];
    car25519(o);
    car25519(o);
}

inline void S(gf o, const gf a) { M(o, a, a); }

inline void inv25519(gf o, const gf i) {
    gf c;
    int a;
    for (a = 0; a < 16; a++) c[a] = i[a];
    for (a = 253; a >= 0; a--) {
        S(c, c);
        if (a != 2 && a != 4) M(c, c, i);
    }
    for (a = 0; a < 16; a++) o[a] = c[a];
}

}  // namespace detail

// q = n * p (x-only Montgomery ladder).  Returns 0 like TweetNaCl.
inline int scalarmult(uint8_t* q, const uint8_t* n, const uint8_t* p) {
    using namespace detail;
    uint8_t z[32];
    i64 x[80], r, i;
    gf a, b, c, d, e, f;
    for (i = 0; i < 31; i++) z[i] = n[i];
    z[31] = (n[31] & 127) | 64;
    z[0] &= 248;
    unpack25519(x, p);
    for (i = 0; i < 16; i++) {
        b[i] = x[i];
        d[i] = a[i] = c[i] = 0;
    }
    a[0] = d[0] = 1;
    for (i = 254; i >= 0; --i) {
        r = (z[i >> 3] >> (i & 7)) & 1;
        sel25519(a, b, (int)r);
        sel25519(c, d, (int)r);
        A(e, a, c);
        Z(a, a, c);
        A(c, b, d);
        Z(b, b, d);
        S(d, e);
        S(f, a);
        M(a, c, a);
        M(c, b, e);
        A(e, a, c);
        Z(a, a, c);
        S(b, a);
        Z(c, d, f);
        M(a, c, _121665);
        A(a, a, d);
        M(c, c, a);
        M(a, d, f);
        M(d, b, x);
        S(b, e);
        sel25519(a, b, (int)r);
        sel25519(c, d, (int)r);
    }
    for (i = 0; i < 16; i++) {
        x[i + 16] = a[i];
        x[i + 32] = c[i];
        x[i + 48] = b[i];
        x[i + 64] = d[i];
    }
    inv25519(x + 32, x + 32);
    M(x + 16, x + 16, x + 32);
    pack25519(q, x + 16);
    return 0;
}

inline int scalarmult_base(uint8_t* q, const uint8_t* n) { return scalarmult(q, n, detail::_9); }

inline bytes32 scalarmult(const bytes32& n, const bytes32& p) {
    bytes32 q;
    scalarmult(q.data(), n.data(), p.data());
    return q;
}
inline bytes32 scalarmult_base(const bytes32& n) {
    bytes32 q;
    scalarmult_base(q.data(), n.data());
    return q;
}

inline const bytes32& basepoint() {
    static const bytes32 bp = [] { bytes32 b{}; b[0] = 9; return b; }();
    return bp;
}

// H2G: SHA-256 of the input, high bit cleared, used as an x-coordinate.
inline bytes32 hash_to_x(const bytes32& in) {
    bytes32 out;
    sha256(in.data(), in.size(), out.data());
    out[31] &= 0x7f;
    return out;
}

}  // namespace x25519
}  // namespace couplet
