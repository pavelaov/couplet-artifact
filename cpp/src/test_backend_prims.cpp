// test_backend_prims.cpp -- tests and micro-benchmarks of the backend
// primitives:
//   ed25519.hpp  Edwards25519 group, strict point encoding, hash to group;
//   hashes.hpp   H_ccr, ctr_prg, framed SHA-256, commitments, coin tossing,
//                Ascon-Hash256 / Ascon-XOF128, ProtocolAbort;
//   rng.hpp      SystemRng and the deterministic CtrPrgRng.
//
// Known answers come from RFC 8032 (key generation, which needs SHA-512: a
// small implementation is included here for that purpose only), from NIST
// SP 800-232 (Ascon), and from the Python implementation of the same
// definitions (couplet/hashes.py, couplet/ed25519.py: big-integer curve
// arithmetic, hashlib SHA-256, the scalar Ascon permutation), including the
// regression constants of python/tests/test_primitives.py, so that the two
// languages are checked to agree bit for bit.
//
// Usage: test_backend_prims [--no-bench | --bench-only]
//        test_backend_prims --interop PARTY PORT
// Exit code 0 iff every check passes.  The benchmark results are printed as
// one JSON object after the line "BENCH_JSON" (single thread, wall clock).
// The --interop mode runs committed_exchange and coin_toss over TCP on
// 127.0.0.1 (party 0 listens, party 1 connects) and prints the values
// obtained, so that the wire format can be compared with another
// implementation of the same definitions (the Python package).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "couplet/ascon.hpp"
#include "couplet/channel.hpp"
#include "couplet/ed25519.hpp"
#include "couplet/gf128.hpp"
#include "couplet/hashes.hpp"
#include "couplet/rng.hpp"
#include "couplet/sha256.hpp"

using namespace couplet;

// ------------------------------------------------------------ harness
static int g_pass = 0, g_fail = 0;

static void check(bool ok, const std::string& what) {
    if (ok) {
        g_pass++;
    } else {
        g_fail++;
        std::printf("  FAIL: %s\n", what.c_str());
    }
}

struct Section {
    const char* name;
    int pass0, fail0;
    explicit Section(const char* n) : name(n), pass0(g_pass), fail0(g_fail) {}
    ~Section() {
        const int f = g_fail - fail0, p = g_pass - pass0;
        std::printf("[%s] %s (%d checks)\n", f ? "FAIL" : "PASS", name, p + f);
        std::fflush(stdout);
    }
};

static std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(2 * n);
    for (size_t i = 0; i < n; i++) {
        s.push_back(d[p[i] >> 4]);
        s.push_back(d[p[i] & 15]);
    }
    return s;
}
template <size_t N>
static std::string hex(const std::array<uint8_t, N>& a) { return hex(a.data(), N); }
static std::string hex(const bytes& b) { return hex(b.data(), b.size()); }

static bytes unhex(const std::string& s) {
    bytes v;
    for (size_t i = 0; i + 1 < s.size(); i += 2)
        v.push_back((uint8_t)std::strtoul(s.substr(i, 2).c_str(), nullptr, 16));
    return v;
}
template <size_t N>
static std::array<uint8_t, N> unhex_arr(const std::string& s) {
    std::array<uint8_t, N> a{};
    const bytes v = unhex(s);
    for (size_t i = 0; i < N && i < v.size(); i++) a[i] = v[i];
    return a;
}
static bytes str_bytes(const std::string& s) { return bytes(s.begin(), s.end()); }

static double now_ns() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// ------------------------------------------------------------ SHA-512 (test only)
namespace sha512 {
static uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

static void compress(uint64_t h[8], const uint8_t* p) {
    static const uint64_t K[80] = {
        0x428a2f98d728ae22ull, 0x7137449123ef65cdull, 0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull,
        0x3956c25bf348b538ull, 0x59f111f1b605d019ull, 0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull,
        0xd807aa98a3030242ull, 0x12835b0145706fbeull, 0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,
        0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull, 0x9bdc06a725c71235ull, 0xc19bf174cf692694ull,
        0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull, 0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
        0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull, 0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull,
        0x983e5152ee66dfabull, 0xa831c66d2db43210ull, 0xb00327c898fb213full, 0xbf597fc7beef0ee4ull,
        0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull, 0x06ca6351e003826full, 0x142929670a0e6e70ull,
        0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull, 0x4d2c6dfc5ac42aedull, 0x53380d139d95b3dfull,
        0x650a73548baf63deull, 0x766a0abb3c77b2a8ull, 0x81c2c92e47edaee6ull, 0x92722c851482353bull,
        0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull, 0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull,
        0xd192e819d6ef5218ull, 0xd69906245565a910ull, 0xf40e35855771202aull, 0x106aa07032bbd1b8ull,
        0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull, 0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull,
        0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull, 0x5b9cca4f7763e373ull, 0x682e6ff3d6b2b8a3ull,
        0x748f82ee5defb2fcull, 0x78a5636f43172f60ull, 0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
        0x90befffa23631e28ull, 0xa4506cebde82bde9ull, 0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull,
        0xca273eceea26619cull, 0xd186b8c721c0c207ull, 0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull,
        0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull, 0x113f9804bef90daeull, 0x1b710b35131c471bull,
        0x28db77f523047d84ull, 0x32caab7b40c72493ull, 0x3c9ebe0a15c9bebcull, 0x431d67c49c100d4cull,
        0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull, 0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull};
    uint64_t w[80];
    for (int i = 0; i < 16; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++) v = (v << 8) | p[8 * i + j];
        w[i] = v;
    }
    for (int i = 16; i < 80; i++) {
        const uint64_t s0 = rotr(w[i - 15], 1) ^ rotr(w[i - 15], 8) ^ (w[i - 15] >> 7);
        const uint64_t s1 = rotr(w[i - 2], 19) ^ rotr(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint64_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 80; i++) {
        const uint64_t S1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
        const uint64_t ch = (e & f) ^ (~e & g);
        const uint64_t t1 = hh + S1 + ch + K[i] + w[i];
        const uint64_t S0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
        const uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint64_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

static std::array<uint8_t, 64> hash(const uint8_t* data, size_t len) {
    uint64_t h[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
                     0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                     0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
    size_t off = 0;
    for (; off + 128 <= len; off += 128) compress(h, data + off);
    uint8_t last[256];
    std::memset(last, 0, sizeof last);
    const size_t rem = len - off;
    if (rem) std::memcpy(last, data + off, rem);
    last[rem] = 0x80;
    const size_t total = (rem + 1 + 16 <= 128) ? 128 : 256;
    const uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) last[total - 1 - (size_t)i] = (uint8_t)(bits >> (8 * i));
    compress(h, last);
    if (total == 256) compress(h, last + 128);
    std::array<uint8_t, 64> out;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) out[(size_t)(8 * i + j)] = (uint8_t)(h[i] >> (56 - 8 * j));
    return out;
}
}  // namespace sha512

// ------------------------------------------------------------ scalar helpers
typedef ed25519::scalar32 scalar32;
using ed25519::Point;

static scalar32 sc_from_u64(uint64_t v) {
    scalar32 s{};
    for (int i = 0; i < 8; i++) s[(size_t)i] = (uint8_t)(v >> (8 * i));
    return s;
}
// Integer addition of two little-endian scalars (the caller keeps the sum below 2^256).
static scalar32 sc_add(const scalar32& a, const scalar32& b) {
    scalar32 r;
    unsigned carry = 0;
    for (size_t i = 0; i < 32; i++) {
        const unsigned t = (unsigned)a[i] + (unsigned)b[i] + carry;
        r[i] = (uint8_t)t;
        carry = t >> 8;
    }
    return r;
}
static scalar32 rand_scalar256(RandomSource& r) { return r.array<32>(); }
static scalar32 rand_scalar255(RandomSource& r) {
    scalar32 s = r.array<32>();
    s[31] &= 0x7f;
    return s;
}
static std::string point_hex(const Point& p) { return hex(ed25519::compress(p)); }

// A point of the full curve group (generally outside the prime-order
// subgroup): the first valid decoding among SHA-256("curve-point" || seed || ctr).
static Point full_curve_point(uint64_t seed) {
    for (uint64_t ctr = 0;; ctr++) {
        const hashes::hash32 h = hashes::framed_sha256({"curve-point", hashes::le64(seed), hashes::le64(ctr)});
        Point p;
        if (ed25519::decompress(p, h.data())) return p;
    }
}

// ------------------------------------------------------------ SHA-512 tests
static void test_sha512() {
    Section sec("SHA-512 (test helper) known answers");
    auto H = [](const bytes& m) { return hex(sha512::hash(m.data(), m.size())); };
    check(H(bytes()) ==
              "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
          "SHA-512(\"\")");
    check(H(str_bytes("abc")) ==
              "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
          "SHA-512(\"abc\")");
    bytes m200(200);
    for (size_t i = 0; i < 200; i++) m200[i] = (uint8_t)i;
    check(H(m200) ==
              "986058e9895e2c2ab8f9e8cbdf801db12a44842a56a91d5a4e87b1fc98b293722c4664142e42c3c551ff898646268cd92b84ed230b8c94bed7798d4f27cd7465",
          "SHA-512(00..c7)");
    check(H(bytes(111, (uint8_t)'a')) ==
              "fa9121c7b32b9e01733d034cfc78cbf67f926c7ed83e82200ef86818196921760b4beff48404df811b953828274461673c68d04e297b0eb7b2b4d60fc6b566a2",
          "SHA-512(a^111)");
    check(H(bytes(112, (uint8_t)'a')) ==
              "c01d080efd492776a1c43bd23dd99d0a2e626d481e16782e75d54c2503b5dc32bd05f0f1ba33e568b88fd2d970929b719ecbb152f58f130a407c8830604b70ca",
          "SHA-512(a^112)");
}

// ------------------------------------------------------------ Edwards25519 tests
static void test_ed_constants() {
    Section sec("ed25519 constants");
    using namespace ed25519::detail;
    x25519::gf t;
    const x25519::gf c121666 = {0xDB42, 1}, c121665 = {0xDB41, 1}, five = {5}, four = {4};
    M(t, D, c121666);
    A(t, t, c121665);
    check(is_zero25519(t), "d * 121666 + 121665 = 0");
    A(t, D, D);
    check(!neq25519(t, D2), "D2 = 2 d");
    S(t, SQRTM1);
    A(t, t, gf1);
    check(is_zero25519(t), "sqrt(-1)^2 = -1");
    M(t, BY, five);
    check(!neq25519(t, four), "5 * By = 4");
    check(par25519(BX) == 0, "Bx is even");
    const Point B = ed25519::base();
    check(ed25519::on_curve(B), "base point on curve");
    check(!ed25519::is_identity(B), "base point is not the identity");
    check(ed25519::is_identity(ed25519::identity()), "identity is the identity");
    check(ed25519::on_curve(ed25519::identity()), "identity on curve");
    check(point_hex(B) == "5866666666666666666666666666666666666666666666666666666666666666", "enc(B)");
    check(point_hex(ed25519::identity()) == "0100000000000000000000000000000000000000000000000000000000000000",
          "enc(identity)");
    check(hex(ed25519::order()) == "edd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010", "q bytes");
    check(!ed25519::scalar_is_reduced(ed25519::order()), "q is not < q");
    scalar32 qm1 = ed25519::order();
    qm1[0] -= 1;
    check(ed25519::scalar_is_reduced(qm1), "q - 1 < q");
    check(ed25519::scalar_is_reduced(sc_from_u64(0)), "0 < q");
}

// The multiplication kernel of ed25519.hpp must reproduce TweetNaCl's M limb for limb.
static void test_ed_field_mul() {
    Section sec("ed25519 field kernel = TweetNaCl M / car25519 / inv25519 (limb for limb)");
    typedef x25519::i64 i64;
    CtrPrgRng rng(0xf1e1d);
    auto rand_fe = [&rng](x25519::gf r, int mode) {
        for (int i = 0; i < 16; i++) {
            const uint64_t u = rng.u64();
            if (mode == 0) r[i] = (i64)(u & 0xffff);                       // carried limbs
            else if (mode == 1) r[i] = (i64)(u & 0x7ffff) - 0x40000;       // signed, |limb| <= 2^18
            else if (mode == 2) r[i] = (u & 1) ? 0x3ffff : -0x3ffff;       // extreme signs
            else r[i] = (i64)(u & 0x1ffffff) - 0x1000000;                  // signed, |limb| <= 2^24
        }
    };
    bool mul_ok = true, alias_ok = true, sq_ok = true;
    for (int it = 0; it < 4000; it++) {
        x25519::gf a, b, o1, o2;
        rand_fe(a, it % 4);
        rand_fe(b, (it / 4) % 4);
        ed25519::detail::M(o1, a, b);
        x25519::detail::M(o2, a, b);
        mul_ok = mul_ok && std::memcmp(o1, o2, sizeof o1) == 0;
        ed25519::detail::S(o1, a);
        x25519::detail::S(o2, a);
        sq_ok = sq_ok && std::memcmp(o1, o2, sizeof o1) == 0;
        // output aliasing an input
        x25519::gf a1, a2, b1, b2;
        std::memcpy(a1, a, sizeof a);
        std::memcpy(a2, a, sizeof a);
        std::memcpy(b1, b, sizeof b);
        std::memcpy(b2, b, sizeof b);
        ed25519::detail::M(a1, a1, b);
        x25519::detail::M(a2, a2, b);
        ed25519::detail::M(b1, a, b1);
        x25519::detail::M(b2, a, b2);
        alias_ok = alias_ok && std::memcmp(a1, a2, sizeof a1) == 0 && std::memcmp(b1, b2, sizeof b1) == 0;
        ed25519::detail::M(a1, a1, a1);
        x25519::detail::M(a2, a2, a2);
        alias_ok = alias_ok && std::memcmp(a1, a2, sizeof a1) == 0;
    }
    check(mul_ok, "M: 4000 random operand pairs (carried, signed, extreme)");
    check(sq_ok, "S: 4000 random operands");
    check(alias_ok, "M with the output aliasing an operand");
    {
        // all-maximal and all-minimal limbs
        x25519::gf hi, lo, o1, o2;
        for (int i = 0; i < 16; i++) {
            hi[i] = 0xffffff;
            lo[i] = -0xffffff;
        }
        bool ok = true;
        const i64* ops[2] = {hi, lo};
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++) {
                ed25519::detail::M(o1, ops[x], ops[y]);
                x25519::detail::M(o2, ops[x], ops[y]);
                ok = ok && std::memcmp(o1, o2, sizeof o1) == 0;
            }
        check(ok, "M on all-maximal / all-minimal limbs");
    }
    bool carry_ok = true;
    for (int it = 0; it < 4000; it++) {
        x25519::gf c1, c2;
        for (int i = 0; i < 16; i++) {
            const uint64_t u = rng.u64();
            const int shift = (it % 3 == 0) ? 12 : ((it % 3 == 1) ? 30 : 46);
            c1[i] = (i64)(u >> shift) - (i64)((uint64_t)1 << (63 - shift));  // |limb| <= 2^(63 - shift)
            if (it % 7 == 0) c1[i] = (u & 1) ? 0xffff : 0;
            if (it % 11 == 0) c1[i] = -(i64)(u & 3);
            c2[i] = c1[i];
        }
        ed25519::detail::carry_twice(c1);
        x25519::detail::car25519(c2);
        x25519::detail::car25519(c2);
        carry_ok = carry_ok && std::memcmp(c1, c2, sizeof c1) == 0;
    }
    check(carry_ok, "carry_twice = car25519 twice (4000 inputs, negative and large limbs)");
    // car25519 itself against its definition written without shifts of signed values: limb i
    // becomes (limb + 2^16) mod 2^16 and floor((limb + 2^16) / 2^16) - 1 is carried (times 38 from
    // limb 15 into limb 0).  The TweetNaCl original subtracts "c << 16", a left shift of a
    // possibly negative value (undefined before C++20); the header multiplies by 65536 instead.
    {
        bool same = true;
        int negative_carries = 0;
        for (int it = 0; it < 4000; it++) {
            x25519::gf a, ref;
            for (int i = 0; i < 16; i++) {
                const uint64_t u = rng.u64();
                const int shift = (it % 3 == 0) ? 12 : ((it % 3 == 1) ? 30 : 46);
                a[i] = (i64)(u >> shift) - (i64)((uint64_t)1 << (63 - shift));
                if (it % 5 == 0) a[i] = -(i64)(u & 0xfffff);
                if (it % 13 == 0) a[i] = (u & 1) ? -1 : -65536;
                ref[i] = a[i];
            }
            for (int i = 0; i < 16; i++) {
                const i64 v = ref[i] + 65536;
                const i64 low = (i64)((uint64_t)v & 0xffff);  // v mod 2^16, in [0, 65535]
                const i64 c = (v - low) / 65536;              // floor(v / 2^16): an exact division, no shift
                negative_carries += (c - 1) < 0;
                ref[i] = low;
                ref[i < 15 ? i + 1 : 0] += (c - 1) + (i == 15 ? 37 * (c - 1) : 0);
            }
            x25519::detail::car25519(a);
            same = same && std::memcmp(a, ref, sizeof a) == 0;
        }
        check(same && negative_carries > 1000, "car25519 = its definition without signed shifts (4000 inputs, " +
                                                   std::to_string(negative_carries) + " negative carries)");
    }
    bool inv_ok = true, pow_ok = true;
    for (int it = 0; it < 20; it++) {
        x25519::gf a, o1, o2, one;
        rand_fe(a, it % 2);
        ed25519::detail::inv25519(o1, a);
        x25519::detail::inv25519(o2, a);
        inv_ok = inv_ok && std::memcmp(o1, o2, sizeof o1) == 0;
        ed25519::detail::M(one, o1, a);
        inv_ok = inv_ok && !ed25519::detail::neq25519(one, ed25519::detail::gf1);
        // pow2523: (a^((p-5)/8))^8 * a^5 = a^(p-5) * a^5 = a^p = a
        x25519::gf w, w8, a5;
        ed25519::detail::pow2523(w, a);
        ed25519::detail::S(w8, w);
        ed25519::detail::S(w8, w8);
        ed25519::detail::S(w8, w8);
        ed25519::detail::S(a5, a);
        ed25519::detail::S(a5, a5);
        ed25519::detail::M(a5, a5, a);
        ed25519::detail::M(w8, w8, a5);
        pow_ok = pow_ok && !ed25519::detail::neq25519(w8, a);
    }
    check(inv_ok, "inv25519 = TweetNaCl inv25519, and a * a^-1 = 1");
    check(pow_ok, "pow2523: (a^((p-5)/8))^8 * a^5 = a");
    std::printf("  (AVX2 multiplication kernel: %s)\n", COUPLET_ED25519_AVX2 ? "enabled" : "disabled, portable M");
}

static void test_ed_rfc8032() {
    Section sec("ed25519 RFC 8032 public keys (SHA-512, clamp, [a]B)");
    struct V { const char* sk; const char* pk; };
    const V vs[] = {
        {"9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
         "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"},
        {"4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
         "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"},
        {"c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
         "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"}};
    for (const V& v : vs) {
        const bytes sk = unhex(v.sk);
        const std::array<uint8_t, 64> h = sha512::hash(sk.data(), sk.size());
        scalar32 a;
        std::memcpy(a.data(), h.data(), 32);
        a[0] &= 248;
        a[31] &= 127;
        a[31] |= 64;
        check(point_hex(ed25519::mul_base(a)) == v.pk, std::string("mul_base: pk of ") + v.sk);
        check(point_hex(ed25519::mul(ed25519::base(), a)) == v.pk, std::string("mul: pk of ") + v.sk);
        check(point_hex(ed25519::mul_ladder(ed25519::base(), a)) == v.pk, std::string("mul_ladder: pk of ") + v.sk);
    }
}

static void test_ed_order() {
    Section sec("ed25519 group order");
    const Point B = ed25519::base();
    const scalar32& q = ed25519::order();
    check(ed25519::is_identity(ed25519::mul(B, q)), "q * B = identity (mul)");
    check(ed25519::is_identity(ed25519::mul_ladder(B, q)), "q * B = identity (mul_ladder)");
    check(ed25519::is_identity(ed25519::mul_base(q)), "q * B = identity (mul_base)");
    check(ed25519::is_identity(ed25519::mul_order(B)), "mul_order(B) = identity");
    check(ed25519::in_prime_subgroup(B), "B in the prime-order subgroup");
    check(ed25519::in_prime_subgroup(ed25519::identity()), "identity in the prime-order subgroup");
    scalar32 qm1 = q;
    qm1[0] -= 1;
    check(ed25519::equal(ed25519::mul_base(qm1), ed25519::neg(B)), "(q - 1) * B = -B");
    scalar32 qp1 = q;
    qp1[0] += 1;
    check(ed25519::equal(ed25519::mul(B, qp1), B), "(q + 1) * B = B (scalar used unreduced)");
    CtrPrgRng rng(0x0de5);
    for (int i = 0; i < 8; i++) {
        const Point P = ed25519::mul_base(rand_scalar256(rng));
        check(ed25519::is_identity(ed25519::mul(P, q)), "q * P = identity for P = s * B");
        check(!ed25519::is_identity(P), "s * B is not the identity");
    }
}

static void test_ed_python_vectors() {
    Section sec("ed25519 vectors shared with the Python implementation");
    struct V { const char* s; const char* out; };
    const V sb[] = {
        {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
         "ca4a448c3fc4d04945da9fdf920976c05e9bbe3d8cebb1858ea44d587c5e63c3"},
        {"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
         "db27fe4b7a4beb8c1b8c38a21e943a852304c9bb3035a5f36626b51162a68f9c"},
        {"bcaff64aa174de78063e7bcad46ee91ee56264a14bbdc882c4f5247fa146f3bf",
         "394309772243498578fced581b2e08d1001c1532a4854c14e20f763ed68e28f9"},
        {"5e443f6e206855cd66ca3722dd825d1e40dfc09eb86f89c9c71d4430b5b0b11f",
         "8fd8d66d7c113c9e26edcc0bca51a9a34e59326ef26b35d8e5fa11592d33109a"},
        {"ecd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010",
         "58666666666666666666666666666666666666666666666666666666666666e6"},
        {"0100000000000000000000000000000000000000000000000000000000000000",
         "5866666666666666666666666666666666666666666666666666666666666666"},
        {"0800000000000000000000000000000000000000000000000000000000000000",
         "b4b937fca95b2f1e93e41e62fc3c78818ff38a66096fad6e7973e5c90006d321"}};
    const Point B = ed25519::base();
    for (const V& v : sb) {
        const scalar32 s = unhex_arr<32>(v.s);
        check(point_hex(ed25519::mul_base(s)) == v.out, std::string("mul_base ") + v.s);
        check(point_hex(ed25519::mul(B, s)) == v.out, std::string("mul(B) ") + v.s);
        check(point_hex(ed25519::mul_ladder(B, s)) == v.out, std::string("mul_ladder(B) ") + v.s);
    }
    uint32_t tries = 99;
    const Point P = ed25519::hash_to_group(std::string("test-tag"), str_bytes("point-P"), &tries);
    check(point_hex(P) == "b718efb68a785f054595613156e39bd006676542e78484a38bab683540b91be5", "P = hash_to_group");
    check(tries == 0, "P accepted at counter 0");
    const V sp[] = {
        {"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
         "0d35dd76ea56d1d6c329e6becf071177ee83af9ca7cffe27fba5f93c5d09293d"},
        {"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
         "61c03fb01ac4497119ff2865a2178c082bdc94ed5fe638c4cf911aa9d97c6760"},
        {"bcaff64aa174de78063e7bcad46ee91ee56264a14bbdc882c4f5247fa146f3bf",
         "f51254262e1dcfa2103c46356485c07bf2b27517f7584cd98484e123b0815342"},
        {"5e443f6e206855cd66ca3722dd825d1e40dfc09eb86f89c9c71d4430b5b0b11f",
         "ce27874dd88148cb69a7e4d147ff224067d6811bce9d2b8787ffde3ad0f1faa2"}};
    for (const V& v : sp) {
        const scalar32 s = unhex_arr<32>(v.s);
        check(point_hex(ed25519::mul(P, s)) == v.out, std::string("mul(P) ") + v.s);
        check(point_hex(ed25519::mul_ladder(P, s)) == v.out, std::string("mul_ladder(P) ") + v.s);
    }
    check(point_hex(ed25519::add(P, B)) == "7a82d96730d9b5fc65647ee63d01505ed9fa816eedac0b4a62172155fd685adb", "P + B");
    check(point_hex(ed25519::sub(P, B)) == "3e2afa81b1935c00c5660559d07e9937ff6568e453c73115b96e0b3c408f0c01", "P - B");
    check(point_hex(ed25519::dbl(P)) == "3bc6a2d4c2141477da27479bf2d621ef675966dcc1ba7380b653056e013c9d14", "2 P (dbl)");
    check(point_hex(ed25519::add(P, P)) == "3bc6a2d4c2141477da27479bf2d621ef675966dcc1ba7380b653056e013c9d14", "2 P (add)");
}

static void test_ed_group_laws() {
    Section sec("ed25519 group laws and scalar multiplication consistency");
    CtrPrgRng rng(0xed25519);
    const Point B = ed25519::base();
    const Point O = ed25519::identity();
    for (int i = 0; i < 16; i++) {
        const scalar32 a = rand_scalar255(rng), b = rand_scalar255(rng);
        const Point aB = ed25519::mul_base(a), bB = ed25519::mul_base(b);
        check(ed25519::equal(ed25519::mul_base(sc_add(a, b)), ed25519::add(aB, bB)), "(a + b) B = a B + b B");
        check(ed25519::equal(ed25519::mul(bB, a), ed25519::mul(aB, b)), "a (b B) = b (a B)");
        check(ed25519::on_curve(aB) && ed25519::on_curve(ed25519::add(aB, bB)), "results on curve");
        check(ed25519::equal(ed25519::add(aB, bB), ed25519::add(bB, aB)), "commutativity");
        const Point cB = ed25519::mul(B, sc_from_u64(0x123456789abcdefull + (uint64_t)i));
        check(ed25519::equal(ed25519::add(ed25519::add(aB, bB), cB), ed25519::add(aB, ed25519::add(bB, cB))),
              "associativity");
        check(ed25519::equal(ed25519::dbl(aB), ed25519::add(aB, aB)), "dbl(P) = P + P");
        check(ed25519::is_identity(ed25519::add(aB, ed25519::neg(aB))), "P + (-P) = identity");
        check(ed25519::equal(ed25519::add(aB, O), aB) && ed25519::equal(ed25519::add(O, aB), aB), "P + 0 = P");
        check(ed25519::equal(ed25519::sub(ed25519::add(aB, bB), bB), aB), "(P + Q) - Q = P");
        check(ed25519::equal(ed25519::mul_cofactor(aB), ed25519::mul(aB, sc_from_u64(8))), "mul_cofactor = 8 P");
    }
    // mul and mul_base against the reference ladder, full 256-bit scalars.
    for (int i = 0; i < 12; i++) {
        const scalar32 s = rand_scalar256(rng);
        const Point P = ed25519::mul_base(rand_scalar256(rng));
        const Point ref = ed25519::mul_ladder(P, s);
        check(ed25519::equal(ed25519::mul(P, s), ref), "mul = mul_ladder");
        check(ed25519::on_curve(ref), "ladder result on curve");
        check(ed25519::equal(ed25519::mul_base(s), ed25519::mul_ladder(B, s)), "mul_base = mul_ladder(B)");
    }
    // Edge scalars.
    scalar32 top{};
    top[31] = 0x80;
    scalar32 ones;
    ones.fill(0xff);
    const scalar32 edge[] = {sc_from_u64(0), sc_from_u64(1), sc_from_u64(2), sc_from_u64(15), sc_from_u64(16),
                             sc_from_u64(17), sc_from_u64(0xffffffffffffffffull), top, ones};
    const Point P = ed25519::hash_to_group(std::string("edge"), bytes());
    for (const scalar32& s : edge) {
        const Point ref = ed25519::mul_ladder(P, s);
        check(ed25519::equal(ed25519::mul(P, s), ref), "mul = mul_ladder on edge scalar " + hex(s));
        check(ed25519::equal(ed25519::mul_base(s), ed25519::mul_ladder(B, s)), "mul_base = ladder on edge scalar " + hex(s));
    }
    check(ed25519::is_identity(ed25519::mul(P, sc_from_u64(0))), "0 * P = identity");
    check(ed25519::equal(ed25519::mul(P, sc_from_u64(1)), P), "1 * P = P");
    check(ed25519::equal(ed25519::mul(P, sc_from_u64(2)), ed25519::dbl(P)), "2 * P = dbl(P)");
    check(ed25519::is_identity(ed25519::mul(O, ones)), "s * identity = identity");
    check(ed25519::is_identity(ed25519::mul_base(sc_from_u64(0))), "0 * B = identity");
    // The shape of one Masny-Rindal instance (DESIGN.md 7.1).
    for (int c = 0; c < 2; c++) {
        const scalar32 a = ed25519::random_scalar(rng), b = ed25519::random_scalar(rng), t = ed25519::random_scalar(rng);
        const Point M = ed25519::mul_base(b);
        const Point r_other = ed25519::mul_base(t);
        const Point Hc = ed25519::hash_to_group(std::string("mr-test"), ed25519::compress(r_other));
        const Point r_c = ed25519::sub(M, Hc);
        const Point A = ed25519::mul_base(a);
        const Point M_c = ed25519::add(r_c, Hc);
        check(ed25519::equal(M_c, M), "r_c + H(r_other) = M");
        check(point_hex(ed25519::mul(M_c, a)) == point_hex(ed25519::mul(A, b)), "a M = b A (same encoding)");
    }
}

static void test_ed_encoding() {
    Section sec("ed25519 compress / decompress");
    CtrPrgRng rng(0xc0de);
    int odd = 0, even = 0;
    for (int i = 0; i < 40; i++) {
        Point P = ed25519::mul_base(rand_scalar256(rng));
        if (i & 1) P = ed25519::neg(P);
        const ed25519::bytes32 e = ed25519::compress(P);
        if (e[31] & 0x80) odd++; else even++;
        Point Q;
        const bool ok = ed25519::decompress(Q, e);
        check(ok, "decompress(compress(P)) succeeds");
        check(ed25519::equal(P, Q), "decompress(compress(P)) = P");
        check(ed25519::on_curve(Q), "decoded point on curve");
        check(ed25519::compress(Q) == e, "compress(decompress(e)) = e");
        ed25519::bytes32 e2;
        ed25519::compress(e2, P);
        uint8_t e3[32];
        ed25519::compress(e3, P);
        check(e2 == e && std::memcmp(e3, e.data(), 32) == 0, "the three compress interfaces agree");
        // -P has the same y and the other sign bit.
        ed25519::bytes32 en = e;
        en[31] ^= 0x80;
        check(ed25519::compress(ed25519::neg(P)) == en, "enc(-P) flips the sign bit");
        Point R;
        check(ed25519::decompress(R, en) && ed25519::equal(R, ed25519::neg(P)), "decompress returns the point itself, not its negation");
    }
    check(odd > 5 && even > 5, "both sign bits occur");

    // Encodings that must be rejected (non-canonical y, x = 0 with sign, non-square).
    const char* bad[] = {"edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
                         "eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
                         "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
                         "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                         "0100000000000000000000000000000000000000000000000000000000000080",
                         "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                         "0200000000000000000000000000000000000000000000000000000000000000"};
    for (const char* b : bad) {
        Point Q = ed25519::base();
        const bool ok = ed25519::decompress(Q, unhex_arr<32>(b));
        check(!ok, std::string("rejects ") + b);
        check(ed25519::equal(Q, ed25519::base()), "output untouched on failure");
    }
    // The eight points of small order: valid encodings, expected orders.
    struct SO { const char* enc; int order; };
    const SO small[] = {{"0100000000000000000000000000000000000000000000000000000000000000", 1},
                        {"ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f", 2},
                        {"0000000000000000000000000000000000000000000000000000000000000000", 4},
                        {"0000000000000000000000000000000000000000000000000000000000000080", 4},
                        {"26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05", 8},
                        {"26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc85", 8},
                        {"c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a", 8},
                        {"c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac03fa", 8}};
    for (const SO& so : small) {
        Point T;
        const bool ok = ed25519::decompress(T, unhex_arr<32>(so.enc));
        check(ok, std::string("decodes small-order point ") + so.enc);
        if (!ok) continue;
        check(ed25519::on_curve(T), "small-order point on curve");
        int ord = 1;
        Point acc = T;
        while (!ed25519::is_identity(acc) && ord < 16) {
            acc = ed25519::add(acc, T);
            ord++;
        }
        check(ord == so.order, std::string("order of ") + so.enc);
        check(ed25519::is_identity(ed25519::mul_cofactor(T)), "8 T = identity");
        check(ed25519::equal(ed25519::dbl(T), ed25519::add(T, T)), "dbl = add on a small-order point");
        check(ed25519::in_prime_subgroup(T) == (so.order == 1), "subgroup test on a small-order point");
        check(hex(ed25519::compress(T)) == so.enc, "re-encoding of a small-order point");
        // B + T is on the curve but outside the subgroup (unless T is the identity).
        const Point BT = ed25519::add(ed25519::base(), T);
        check(ed25519::on_curve(BT), "B + T on curve");
        check(ed25519::in_prime_subgroup(BT) == (so.order == 1), "B + T in subgroup iff T = identity");
        Point D;
        check(ed25519::decompress_in_subgroup(D, ed25519::compress(BT)) == (so.order == 1),
              "decompress_in_subgroup(B + T)");
        check(ed25519::in_prime_subgroup(ed25519::mul_cofactor(BT)), "8 (B + T) in subgroup");
    }
    // Points of the full curve: q P has order dividing 8, 8 P lies in the subgroup.
    int outside = 0;
    for (uint64_t i = 0; i < 12; i++) {
        const Point P = full_curve_point(i);
        check(ed25519::on_curve(P), "decoded full-curve point on curve");
        const Point T = ed25519::mul_order(P);
        check(ed25519::is_identity(ed25519::mul_cofactor(T)), "8 (q P) = identity");
        check(ed25519::in_prime_subgroup(ed25519::mul_cofactor(P)), "8 P in subgroup");
        if (!ed25519::in_prime_subgroup(P)) outside++;
    }
    check(outside >= 6, "most decoded points lie outside the prime-order subgroup");
}

struct ScriptedSource {  // returns pre-set 32-byte strings (for the rejection-sampling test)
    std::vector<scalar32> items;
    size_t next = 0;
    void fill(uint8_t* out, size_t n) {
        std::memcpy(out, items.at(next).data(), n);
        next++;
    }
};

static void test_ed_hash_to_group() {
    Section sec("ed25519 hash_to_group and random_scalar");
    // Vectors computed by couplet/ed25519.py; the first two are the regression
    // constants KAT_H2G_EMPTY and KAT_H2G_TAG_MSG of python/tests/test_primitives.py.
    struct V { const char* tag; const char* msg; uint32_t ctr; const char* out; };
    const V vs[] = {
        {"", "", 5, "554d65722bea908cfa6820fff04da728aaa65962491fc4ffe674f732f1e93335"},
        {"746167", "6d7367", 0, "9269b2fb58d6a540a227c29674f8826ab16af3ccdf15851f7993d78e0aa06bb0"},
        {"6d72", "616263", 1, "c814c599e2ef4e105dd78b472228edc48a5f986475c2470f7954f90225e5f669"},
        {"436f75706c65742d683267", "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", 0,
         "325520046e30c450bfa3d769271588a66e638b121cfa04bc86878b04c7be941d"},
        {"61", "6263", 0, "f1440331a6e4884e0899be4b6eee857b8bde464ef39671626f3bab724b8cc4d1"},
        {"6162", "63", 1, "0556a765bfc754c60b6877b060dccb3c4814dbd31847b34ba7a4f368b890414b"},
        {"6d722d74657374", "5866666666666666666666666666666666666666666666666666666666666666", 1,
         "708cf6f34fd17ca5c543904c52ba3080a9bcf66fb6e4b4ec8dec9e475f72b446"}};
    for (const V& v : vs) {
        uint32_t tries = 77;
        const Point Q = ed25519::hash_to_group(unhex(v.tag), unhex(v.msg), &tries);
        check(point_hex(Q) == v.out, std::string("hash_to_group(") + v.tag + ", " + v.msg + ")");
        check(tries == v.ctr, "accepted counter");
    }
    {
        // the overloads taking a compressed point as the message (base OT usage)
        const ed25519::bytes32 encB = ed25519::compress(ed25519::base());
        const char* expect = "708cf6f34fd17ca5c543904c52ba3080a9bcf66fb6e4b4ec8dec9e475f72b446";
        check(point_hex(ed25519::hash_to_group(std::string("mr-test"), encB)) == expect, "hash_to_group(string, bytes32)");
        check(point_hex(ed25519::hash_to_group(str_bytes("mr-test"), encB)) == expect, "hash_to_group(bytes, bytes32)");
    }
    // The documented input framing.
    {
        const bytes tag = str_bytes("tg"), msg = str_bytes("message");
        const bytes in = ed25519::hash_to_group_input(tag.data(), tag.size(), msg.data(), msg.size(), 0x01020304);
        const bytes expect = unhex("0b00000000000000" "436f75706c65742d683267" "0200000000000000" "7467"
                                   "0700000000000000" "6d657373616765" "0400000000000000" "04030201");
        check(in == expect, "hash_to_group_input framing");
        check(in == hashes::frame({"Couplet-h2g", "tg", "message", unhex("04030201")}), "framing equals hashes::frame");
        // First principles: counter 0 of ("tag", "msg3") is rejected, counter 1 decodes; 8 * point is the output.
        const bytes t3 = str_bytes("tag"), m3 = str_bytes("msg3");
        const hashes::hash32 h0 = hashes::framed_sha256({"Couplet-h2g", t3, m3, unhex("00000000")});
        const hashes::hash32 h1 = hashes::framed_sha256({"Couplet-h2g", t3, m3, unhex("01000000")});
        Point c0, c1;
        check(!ed25519::decompress(c0, h0), "counter 0 does not decode");
        const bool ok1 = ed25519::decompress(c1, h1);
        check(ok1, "counter 1 decodes");
        check(ok1 && ed25519::equal(ed25519::mul_cofactor(c1), ed25519::hash_to_group(t3, m3)), "output = 8 * decoded point");
    }
    uint64_t total_tries = 0;
    for (uint64_t i = 0; i < 200; i++) {
        const std::array<uint8_t, 8> le = hashes::le64(i);
        const bytes msg(le.begin(), le.end());
        uint32_t tries = 0;
        const Point Q = ed25519::hash_to_group(std::string("order-test"), msg, &tries);
        total_tries += tries;
        if (i < 24) {
            check(ed25519::on_curve(Q), "hash_to_group output on curve");
            check(!ed25519::is_identity(Q), "hash_to_group output is not the identity");
            check(ed25519::is_identity(ed25519::mul(Q, ed25519::order())), "q * hash_to_group(..) = identity");
            check(ed25519::equal(Q, ed25519::hash_to_group(std::string("order-test"), msg)), "deterministic");
            check(!ed25519::equal(Q, ed25519::hash_to_group(std::string("order-tesu"), msg)), "depends on the tag");
        }
    }
    // Each attempt succeeds with probability about 1/2; the Python implementation rejects
    // exactly 189 candidates on these 200 inputs.
    check(total_tries == 189, "rejected attempts over 200 calls = " + std::to_string(total_tries) + " (Python: 189)");
    {
        const bytes msg = str_bytes("abc");
        const Point Q = ed25519::hash_to_group_with(hashes::AsconXof128Hash32(), (const uint8_t*)"mr", 2, msg.data(), msg.size());
        check(ed25519::in_prime_subgroup(Q) && !ed25519::is_identity(Q), "hash_to_group_with(Ascon-XOF128) in the subgroup");
        check(!ed25519::equal(Q, ed25519::hash_to_group(std::string("mr"), msg)), "different hash, different point");
    }

    // random_scalar: uniform in [1, q - 1] by rejection.
    CtrPrgRng rng(7);
    bool all_reduced = true, all_nonzero = true, all_distinct = true;
    scalar32 prev{};
    for (int i = 0; i < 200; i++) {
        const scalar32 s = ed25519::random_scalar(rng);
        all_reduced = all_reduced && ed25519::scalar_is_reduced(s);
        all_nonzero = all_nonzero && !ed25519::scalar_is_zero(s);
        all_distinct = all_distinct && s != prev;
        prev = s;
    }
    check(all_reduced, "200 random scalars are below q");
    check(all_nonzero, "200 random scalars are nonzero");
    check(all_distinct, "consecutive random scalars differ");
    ScriptedSource src;
    scalar32 ones;
    ones.fill(0xff);
    scalar32 q_garbage = ed25519::order();
    q_garbage[31] |= 0xe0;
    scalar32 zero_garbage{};
    zero_garbage[31] = 0xe0;
    scalar32 qm1_garbage = ed25519::order();
    qm1_garbage[0] -= 1;
    const scalar32 qm1 = qm1_garbage;
    qm1_garbage[31] |= 0xa0;
    src.items = {ones, q_garbage, zero_garbage, qm1_garbage};
    const scalar32 got = ed25519::random_scalar(src);
    check(got == qm1, "rejection sampling: 2^253 - 1, q and 0 rejected, q - 1 accepted");
    check(src.next == 4, "four draws consumed");
    // decompress_in_subgroup and the identity.
    {
        Point D;
        const ed25519::bytes32 id = ed25519::compress(ed25519::identity());
        check(ed25519::decompress_in_subgroup(D, id) && ed25519::is_identity(D), "identity accepted by default");
        Point E = ed25519::base();
        check(!ed25519::decompress_in_subgroup(E, id, false) && ed25519::equal(E, ed25519::base()), "identity rejected with allow_identity = false");
        check(ed25519::decompress_in_subgroup(E, ed25519::compress(ed25519::dbl(ed25519::base())), false), "2B accepted with allow_identity = false");
    }
}

// ------------------------------------------------------------ hashes tests
static void test_h_ccr() {
    Section sec("H_ccr (scalar, batched)");
    struct V { uint64_t x_lo, x_hi, tweak, dom, y_lo, y_hi; };
    const V vs[] = {
        {0, 0, 0, 0, 0x78ea7ae5cfebb108ull, 0x9b9bfb8513b560f7ull},
        {1, 2, 3, 4, 0xccd73f3c06ab3a9dull, 0x4deaa0afb44de908ull},
        {0x0123456789abcdefull, 0xfedcba9876543210ull, 0xffffffffffffffffull, 0x4800000000000001ull,
         0x8b2b6d2b0a7d6f29ull, 0x535b635ee1bd9f8eull},
        {0xdeadbeefcafef00dull, 0x0badc0de0d15ea5eull, 77, 0x48000000000000ffull, 0xd79d0ffb1eeedd6eull,
         0xa87dadfc448b0761ull}};
    for (const V& v : vs) {
        const gf128 y = hashes::H_ccr(gf128(v.x_lo, v.x_hi), v.tweak, v.dom);
        check(y.lo == v.y_lo && y.hi == v.y_hi, "H_ccr known answer");
        uint64_t out[2];
        hashes::H_ccr(v.x_lo, v.x_hi, v.tweak, v.dom, out);
        check(out[0] == v.y_lo && out[1] == v.y_hi, "H_ccr (word interface) known answer");
        uint64_t s[5] = {v.x_lo, v.x_hi, v.tweak, v.dom, 0};
        ascon::p12(s);
        check(s[0] == v.y_lo && s[1] == v.y_hi, "definition: words x0, x1 of p12(x_lo, x_hi, tweak, dom, 0)");
    }
    check(hashes::dom::CCR_TEST == 0x4800000000000001ull && hashes::dom::ccr(0xff) == 0x48000000000000ffull,
          "CCR domain constants");
    // Regression constants of python/tests/test_primitives.py (KAT_X, KAT_H_X_1_2, KAT_H_0_0_0, KAT_H_0_0_2).
    {
        const gf128 X(0x0706050403020100ull, 0x0f0e0d0c0b0a0908ull);
        check(hashes::H_ccr(X, 1, 2) == gf128(0x7d46e228a7058a58ull, 0x6da1d8e5727ba7b4ull), "Python regression vector H_ccr(X, 1, 2)");
        check(hashes::H_ccr(gf128(0, 0), 0, 0) == gf128(0x78ea7ae5cfebb108ull, 0x9b9bfb8513b560f7ull), "Python regression vector H_ccr(0, 0, 0)");
        check(hashes::H_ccr(gf128(0, 0), 0, 2) == gf128(0x5c5c41bf99bff3a8ull, 0x155f8ad25de35024ull), "Python regression vector H_ccr(0, 0, 2)");
    }
    // The four reserved domains (x3 words of the FSS PRG labels) are rejected.
    {
        const uint64_t reserved[4] = {ascon::IV_G().w[1], ascon::IV_C().w[1], ascon::IV_VL().w[1], ascon::IV_VR().w[1]};
        check(reserved[0] == 0x47ull && reserved[1] == 0x43ull && reserved[2] == 0x4c56ull && reserved[3] == 0x5256ull,
              "reserved domains = x3 words of Couplet-G / -C / -VL / -VR");
        const hashes::seed16 sd{};
        for (uint64_t d : reserved) {
            int thrown = 0;
            gf128 g(1, 2), o;
            uint64_t w[3], o2[2];
            try { (void)hashes::H_ccr(g, 0, d); } catch (const std::invalid_argument&) { thrown++; }
            try { hashes::H_ccr(1, 2, 0, d, o2); } catch (const std::invalid_argument&) { thrown++; }
            try { hashes::H_ccr_batch(&g, 1, 0, d, &o); } catch (const std::invalid_argument&) { thrown++; }
            try { hashes::ctr_prg(sd, d, w, 3); } catch (const std::invalid_argument&) { thrown++; }
            try { (void)hashes::ctr_prg_bytes(sd, d, 8); } catch (const std::invalid_argument&) { thrown++; }
            try { (void)hashes::ctr_prg_gf128(sd, d, 1); } catch (const std::invalid_argument&) { thrown++; }
            try { hashes::CtrPrgStream st(sd, d); } catch (const std::invalid_argument&) { thrown++; }
            check(thrown == 7 && hashes::dom::is_reserved(d), "reserved domain rejected by H_ccr and ctr_prg");
        }
        check(!hashes::dom::is_reserved(0) && !hashes::dom::is_reserved(hashes::dom::CCR_TEST) && !hashes::dom::is_reserved(hashes::dom::PRG_RNG),
              "ordinary domains accepted");
    }
    CtrPrgRng rng(0xcc5);
    const size_t sizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 13, 64, 1001};
    for (size_t n : sizes) {
        std::vector<gf128> x(n), out(n), ref(n), out2(n);
        std::vector<uint64_t> tw(n);
        rng.fill_gf(x.data(), n);
        for (size_t i = 0; i < n; i++) tw[i] = rng.u64();
        const uint64_t tweak0 = rng.u64();
        const uint64_t d = hashes::dom::ccr(0x77);
        for (size_t i = 0; i < n; i++) ref[i] = hashes::H_ccr(x[i], tweak0 + i, d);
        hashes::H_ccr_batch(x.data(), n, tweak0, d, out.data());
        check(out == ref, "H_ccr_batch = scalar, n = " + std::to_string(n));
        for (size_t i = 0; i < n; i++) ref[i] = hashes::H_ccr(x[i], tw[i], d);
        hashes::H_ccr_batch_tweaks(x.data(), tw.data(), n, d, out2.data());
        check(out2 == ref, "H_ccr_batch_tweaks = scalar, n = " + std::to_string(n));
        std::vector<gf128> inplace = x;
        hashes::H_ccr_batch(inplace.data(), n, tweak0, d, inplace.data());
        check(inplace == out, "H_ccr_batch in place, n = " + std::to_string(n));
        for (size_t i = 0; i < n; i++) ref[i] = hashes::H_ccr(x[i], tweak0, d);
        hashes::H_ccr_batch_same_tweak(x.data(), n, tweak0, d, out2.data());
        check(out2 == ref, "H_ccr_batch_same_tweak = scalar, n = " + std::to_string(n));
    }
    // Sensitivity to each argument.
    const gf128 x0(5, 6);
    const gf128 y0 = hashes::H_ccr(x0, 1, hashes::dom::CCR_TEST);
    check(y0 != hashes::H_ccr(gf128(4, 6), 1, hashes::dom::CCR_TEST), "depends on x_lo");
    check(y0 != hashes::H_ccr(gf128(5, 7), 1, hashes::dom::CCR_TEST), "depends on x_hi");
    check(y0 != hashes::H_ccr(x0, 2, hashes::dom::CCR_TEST), "depends on the tweak");
    check(y0 != hashes::H_ccr(x0, 1, hashes::dom::ccr(2)), "depends on the domain");
}

static void test_ctr_prg() {
    Section sec("ctr_prg (words, bytes, bits, gf128, stream)");
    {
        const hashes::seed16 zero{};
        const std::vector<uint64_t> w = hashes::ctr_prg_words(zero, 0, 3003);
        const uint64_t first[8] = {0x78ea7ae5cfebb108ull, 0x9b9bfb8513b560f7ull, 0x6937f83e03d11a50ull,
                                   0x5e325059c7a7dfc4ull, 0x613d174403673ddaull, 0x0153a9fa2012484dull,
                                   0xdc65bc2cb23daa8cull, 0xd8103c397b73e119ull};
        bool ok = true;
        for (int i = 0; i < 8; i++) ok = ok && w[(size_t)i] == first[i];
        check(ok, "known answer: seed 0, domain 0, words 0..7");
        check(w[3000] == 0x357dbcd042749a7cull && w[3001] == 0xcafaac1d65e6662cull && w[3002] == 0x65ff8458cb0aa254ull,
              "known answer: seed 0, domain 0, words 3000..3002");
    }
    hashes::seed16 seed;
    for (size_t i = 0; i < 16; i++) seed[i] = (uint8_t)i;
    const uint64_t d = 0x5000000000000001ull;
    check(hashes::dom::PRG_RNG == d && hashes::dom::prg(1) == d, "PRG domain constants");
    const size_t N = 3 * 2000 + 2;
    const std::vector<uint64_t> w = hashes::ctr_prg_words(seed, d, N);
    {
        const uint64_t first[8] = {0xbdfeb910399328eeull, 0x91f22d723dead728ull, 0x7fbd0d7db2178cd9ull,
                                   0x00467a5d6991c70aull, 0x77d0bfd594dbaf19ull, 0x7a8850f24ac992b3ull,
                                   0x3c09bc7b500c3414ull, 0x2a3017ed2679dd95ull};
        bool ok = true;
        for (int i = 0; i < 8; i++) ok = ok && w[(size_t)i] == first[i];
        check(ok, "known answer: seed 00..0f, words 0..7");
        check(w[3000] == 0x09b72af45052c343ull && w[3001] == 0x1624d1c82519b186ull && w[3002] == 0xbc1998c882b6819eull,
              "known answer: seed 00..0f, words 3000..3002");
    }
    // Definition: block i = words x0, x1, x2 of p12(seed_lo, seed_hi, i, dom, 0).
    {
        bool ok = true;
        for (uint64_t i = 0; i < 2000; i++) {
            uint64_t s[5] = {ascon::load64le(seed.data()), ascon::load64le(seed.data() + 8), i, d, 0};
            ascon::p12(s);
            ok = ok && w[3 * i] == s[0] && w[3 * i + 1] == s[1] && w[3 * i + 2] == s[2];
        }
        check(ok, "definition of the blocks (2000 blocks)");
    }
    // Every prefix length gives a prefix (exercises the 4-way path, the scalar tail, partial blocks).
    {
        bool ok = true;
        for (size_t n = 0; n <= 64; n++) {
            std::vector<uint64_t> p(n + 1, 0xa5a5a5a5a5a5a5a5ull);
            hashes::ctr_prg(seed, d, p.data(), n);
            for (size_t i = 0; i < n; i++) ok = ok && p[i] == w[i];
            ok = ok && p[n] == 0xa5a5a5a5a5a5a5a5ull;  // no write past the end
        }
        check(ok, "prefix consistency for n_words = 0..64, no overrun");
    }
    // first_block offset.
    {
        std::vector<uint64_t> p(100);
        hashes::ctr_prg(seed.data(), d, p.data(), 100, 500);
        bool ok = true;
        for (size_t i = 0; i < 100; i++) ok = ok && p[i] == w[1500 + i];
        check(ok, "first_block offset");
        check(p[0] == 0xd6d5b7ec5b0c0493ull && p[1] == 0xafee997ed232d41bull && p[2] == 0x8922ea193e20d842ull,
              "known answer: seed 00..0f, start block 500");
        const hashes::seed16 zero{};
        const std::vector<uint64_t> z = hashes::ctr_prg_words(zero, 0, 3, 500);
        check(z[0] == 0xa37198bb43698501ull && z[1] == 0xe2aac8bb12a8e0f6ull && z[2] == 0x550646943b88f94full,
              "known answer: seed 0, domain 0, start block 500");
    }
    // Regression constants of python/tests/test_primitives.py (KAT_PRG, KAT_PRG_BITS: seed 00..0f, dom 7).
    {
        const uint64_t kat[7] = {0x06a0383fee1e2ba2ull, 0x8837b4bb06fae342ull, 0x556312fd72a8d4eaull, 0x25d4cf495d02efd9ull,
                                 0xa2a54cb2406cab33ull, 0x5e94674efcac4cabull, 0x16fdc9aaacd11e27ull};
        const std::vector<uint64_t> k = hashes::ctr_prg_words(seed, 7, 7);
        bool ok = true;
        for (size_t i = 0; i < 7; i++) ok = ok && k[i] == kat[i];
        check(ok, "Python regression vector ctr_prg(00..0f, 7, 7)");
        uint8_t bits[70];
        hashes::ctr_prg_bits(seed, 7, bits, 70);
        std::string bs;
        for (uint8_t b : bits) bs.push_back(b ? '1' : '0');
        check(bs == "0100010111010100011110000111011111111100000111000000010101100000010000", "Python regression vector prg_bits(00..0f, 7, 70)");
        hashes::ctr_prg_bits(seed, d, bits, 70);
        bs.clear();
        for (uint8_t b : bits) bs.push_back(b ? '1' : '0');
        check(bs == "0111011100010100110010011001110000001000100111010111111110111101000101", "bit view known answer (seed 00..0f)");
    }
    // H_ccr(x, i, dom) = first two words of block i of ctr_prg(x, dom): same permutation
    // input layout, separated by the domain constants only (documented in hashes.hpp).
    {
        const gf128 x = gf128::from_bytes(seed.data());
        const gf128 y = hashes::H_ccr(x, 5, d);
        check(y.lo == w[15] && y.hi == w[16], "documented relation between H_ccr and ctr_prg under equal domains");
        check((hashes::dom::CCR_BASE >> 56) != (hashes::dom::PRG_BASE >> 56), "H_ccr and ctr_prg domain families differ");
    }
    // Byte view.
    const size_t byte_sizes[] = {0, 1, 7, 8, 9, 23, 24, 25, 6143, 6144, 6145, 12289, 20000};
    for (size_t n : byte_sizes) {
        bytes b(n + 1, 0x5a);
        hashes::ctr_prg_bytes(seed.data(), d, b.data(), n);
        bool ok = b[n] == 0x5a;
        for (size_t i = 0; i < n; i++) ok = ok && b[i] == (uint8_t)(w[i / 8] >> (8 * (i % 8)));
        check(ok, "byte view, n = " + std::to_string(n));
        b.resize(n);
        check(hashes::ctr_prg_bytes(seed, d, n) == b, "byte view (vector), n = " + std::to_string(n));
    }
    // Bit view.
    const size_t bit_sizes[] = {0, 1, 63, 64, 65, 191, 192, 193, 49151, 49152, 49153, 100001};
    for (size_t n : bit_sizes) {
        bytes b(n + 1, 7);
        hashes::ctr_prg_bits(seed, d, b.data(), n);
        bool ok = b[n] == 7;
        for (size_t i = 0; i < n; i++) ok = ok && b[i] == (uint8_t)((w[i / 64] >> (i % 64)) & 1);
        check(ok, "bit view, n = " + std::to_string(n));
    }
    // GF(2^128) view.
    const size_t gf_sizes[] = {0, 1, 2, 3, 383, 384, 385, 769, 3000};
    for (size_t n : gf_sizes) {
        const std::vector<gf128> g = hashes::ctr_prg_gf128(seed, d, n);
        bool ok = g.size() == n;
        for (size_t i = 0; i < n; i++) ok = ok && g[i].lo == w[2 * i] && g[i].hi == w[2 * i + 1];
        check(ok, "gf128 view, n = " + std::to_string(n));
    }
    // Stream: independent of how the reads are split.
    {
        const bytes ref = hashes::ctr_prg_bytes(seed, d, 10000);
        hashes::CtrPrgStream st(seed, d);
        bytes got(10000);
        size_t off = 0, step = 1;
        while (off < got.size()) {
            const size_t take = (step < got.size() - off) ? step : got.size() - off;
            st.fill(got.data() + off, take);
            off += take;
            step = step * 3 + 1;
            if (step > 4000) step = 5;
        }
        check(got == ref, "CtrPrgStream = byte view under irregular reads");
        check(st.consumed() == 10000, "CtrPrgStream::consumed");
        hashes::CtrPrgStream st2(seed.data(), d);
        bytes one(10000);
        st2.fill(one.data(), one.size());
        check(one == ref, "CtrPrgStream = byte view in one read");
    }
    // Different seeds / domains give different streams.
    {
        hashes::seed16 seed2 = seed;
        seed2[15] ^= 0x80;
        check(hashes::ctr_prg_words(seed2, d, 3) != hashes::ctr_prg_words(seed, d, 3), "depends on the seed");
        check(hashes::ctr_prg_words(seed, d + 1, 3) != hashes::ctr_prg_words(seed, d, 3), "depends on the domain");
    }
}

static void test_framing_commit() {
    Section sec("framed SHA-256, expansion, kdf, commitments, session domains");
    using hashes::framed_sha256;
    bytes seq(32);
    for (size_t i = 0; i < 32; i++) seq[i] = (uint8_t)i;
    check(hex(framed_sha256({})) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "no parts = SHA-256(\"\")");
    check(hex(framed_sha256({""})) == "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", "one empty part");
    check(hex(framed_sha256({"abc"})) == "ce91dc5eec0139adf091900d225971d6ad246a845bad791b5693a9d0d55dd391", "('abc')");
    check(hex(framed_sha256({"ab", "c"})) == "43ee655579de01ca739b3f95c1c2d3f46d353b2c0df818064ea594506cdb2617",
          "('ab','c') (Python KAT_SHA_AB_C)");
    check(hex(framed_sha256({"a", "bc"})) == "9a8acca1b6c6c0befd3fbc756aed625da998c998f7252e738c4ef061906b9b21", "('a','bc')");
    check(hex(framed_sha256({"Couplet", seq, bytes()})) == "8bec8096c0a1e5af4f3d22296be4a9ddc626245cb0bec2349f73da0b0903b5fe",
          "('Couplet', 00..1f, '')");
    check(framed_sha256({"abc"}) != framed_sha256({"abc", ""}), "an empty part changes the hash");
    check(hex(hashes::frame({"ab", "c"})) == "0200000000000000" "6162" "0100000000000000" "63", "frame('ab','c')");
    check(hashes::frame({}).empty(), "frame() is empty");
    {
        const bytes f = hashes::frame({"Couplet", seq, bytes()});
        check(hashes::hash32(sha256(f)) == framed_sha256({"Couplet", seq, bytes()}), "framed_sha256 = SHA-256(frame)");
        std::vector<hashes::Part> parts;
        parts.push_back(hashes::Part("Couplet"));
        parts.push_back(hashes::Part(seq));
        parts.push_back(hashes::Part());
        check(framed_sha256(parts) == framed_sha256({"Couplet", seq, bytes()}), "vector<Part> overload");
        check(hashes::frame(parts) == f, "frame vector<Part> overload");
        const std::string s = "Couplet";
        const std::array<uint8_t, 32> arr = unhex_arr<32>(hex(seq));
        check(framed_sha256({s, arr, ""}) == framed_sha256({"Couplet", seq, bytes()}), "string / array parts");
        check(hex(hashes::le64(0x0102030405060708ull)) == "0807060504030201", "le64");
        check(hex(hashes::truncate16(framed_sha256({"abc"}))) == "ce91dc5eec0139adf091900d225971d6", "truncate16");
    }
    // Counter mode and key derivation.
    {
        check(hex(hashes::sha256_expand(40, {"x", "yz"})) ==
                  "1ab1137394263ac36190fb6a811db4d023d3875fbd6fb950338c7fa37957ccbba77cd108ad9186b4",
              "sha256_expand(40, 'x', 'yz') (Python KAT_EXPAND)");
        const bytes e80 = hashes::sha256_expand(80, {"a", "bc"});
        check(hex(e80) ==
                  "d292061f3c5ca8d5359db92cce111bfabe1b4fdae23d7e02c16854dad9f0417679a230cbea69733c36098ab58dcac96fce2e02b08730c10fbed4535e2bb1b7e1cfc70ae0ae504639f3c65570db184cf8",
              "sha256_expand(80, 'a', 'bc')");
        bool ok = true;
        for (size_t n = 0; n <= 80; n++) {
            const bytes e = hashes::sha256_expand(n, {"a", "bc"});
            ok = ok && e.size() == n && std::equal(e.begin(), e.end(), e80.begin());
        }
        check(ok, "sha256_expand lengths 0..80 are prefixes");
        const hashes::hash32 b1 = framed_sha256({"a", "bc", hashes::le64(1)});
        check(std::equal(b1.begin(), b1.end(), e80.begin() + 32), "block i = framed_sha256(parts..., le64(i))");
        check(hex(hashes::kdf({"sid", hashes::le64(5), seq})) == "c0c1b76eef7158b064a166d061f25d1a", "kdf known answer");
        check(hex(hashes::kdf32({"sid", hashes::le64(5), seq})) == "c0c1b76eef7158b064a166d061f25d1a701e95aace2821b52fe60a5a212f8660",
              "kdf32 known answer");
        check(hashes::kdf32({"sid", hashes::le64(5), seq}) == framed_sha256({"Couplet-kdf", "sid", hashes::le64(5), seq}),
              "kdf = framed_sha256('Couplet-kdf', parts...)");
    }
    // Commitments: vectors from couplet.hashes._com(sid, label, party, omega, msg).
    const bytes sid = str_bytes("session-1"), msg = str_bytes("hello commitment");
    hashes::opening32 omega;
    for (size_t i = 0; i < 32; i++) omega[i] = (uint8_t)i;
    struct CV { int party; const char* label; const char* com; };
    const CV cvs[] = {{hashes::NO_PARTY, "", "b3d7df5090157b045045cd21c6c540a2c78407769546fadd6d1e9a00a5f69777"},
                      {hashes::NO_PARTY, "v", "692fcb74c3e93047d7858688d92e5be257275048f7c752f26f146ec934ae6f32"},
                      {0, "", "cc74e4373931240da08606bb8992292607e28b5f2dce0b649bd859d86617b731"},
                      {0, "v", "3f16a44a20b772acc3efc70bc4e3a864fab828bed9ab7cbe246ce7f5f4c0e9b7"},
                      {1, "", "a5a829986a773bb68254c3ba0680f0433565040d6978853be78a6c0223b1d004"},
                      {1, "v", "dafafce7ce589089bdd4f444e868ad46be4f355a80c5ec21d1b5b95a387a3964"}};
    for (const CV& cv : cvs) {
        const hashes::hash32 com = hashes::commit_with_opening(sid, msg, omega, cv.party, cv.label);
        check(hex(com) == cv.com, std::string("commitment known answer, party ") + std::to_string(cv.party) + ", label '" + cv.label + "'");
        check(hashes::verify_commit(com, sid, msg, omega, cv.party, cv.label), "verify accepts it");
        // every other (party, label) combination is rejected
        int accepted = 0;
        for (const CV& other : cvs)
            if (hashes::verify_commit(com, sid, msg, omega, other.party, other.label)) accepted++;
        check(accepted == 1, "bound to (party, label)");
    }
    check(hashes::commit_with_opening(sid, msg, omega) == framed_sha256({"Couplet-com", sid, "", "", omega, msg}),
          "com = framed_sha256('Couplet-com', sid, label, party tag, omega, msg), no party");
    check(hashes::commit_with_opening(sid, msg, omega, 1, "v") == framed_sha256({"Couplet-com", sid, "v", unhex("01"), omega, msg}),
          "party tag of party 1 is the byte 01");
    {
        bool thrown = false;
        try { (void)hashes::commit_with_opening(sid, msg, omega, 2); } catch (const std::invalid_argument&) { thrown = true; }
        check(thrown, "party index other than 0, 1, NO_PARTY is refused");
    }
    CtrPrgRng rng(0xc0111);
    const hashes::Commitment c = hashes::commit(sid, msg, rng);
    check(c.com == hashes::commit_with_opening(sid, msg, c.opening), "commit = commit_with_opening(fresh omega)");
    check(hashes::verify_commit(c.com, sid, msg, c.opening), "verify accepts the honest opening");
    check(hashes::verify(c.com, sid, msg, c.opening), "verify alias");
    bytes msg2 = msg;
    msg2[0] ^= 1;
    check(!hashes::verify_commit(c.com, sid, msg2, c.opening), "verify rejects another message");
    bytes msg3 = msg;
    msg3.push_back(0);
    check(!hashes::verify_commit(c.com, sid, msg3, c.opening), "verify rejects an extended message");
    check(!hashes::verify_commit(c.com, str_bytes("session-2"), msg, c.opening), "verify rejects another sid");
    check(!hashes::verify_commit(c.com, sid, msg, c.opening, 0), "verify rejects another party binding");
    check(!hashes::verify_commit(c.com, sid, msg, c.opening, hashes::NO_PARTY, "x"), "verify rejects another label");
    hashes::opening32 om2 = c.opening;
    om2[31] ^= 0x80;
    check(!hashes::verify_commit(c.com, sid, msg, om2), "verify rejects another opening");
    hashes::hash32 com2 = c.com;
    com2[0] ^= 1;
    check(!hashes::verify_commit(com2, sid, msg, c.opening), "verify rejects another commitment");
    const hashes::Commitment c2 = hashes::commit(sid, msg, rng);
    check(c2.com != c.com && c2.opening != c.opening, "fresh randomness per commitment");
    {
        const hashes::Commitment ce = hashes::commit(bytes(), bytes(), rng, 1, "l");
        check(hashes::verify_commit(ce.com, bytes(), bytes(), ce.opening, 1, "l"), "empty sid and message");
        check(!hashes::verify_commit(ce.com, bytes(), bytes(), ce.opening, 0, "l"), "empty sid and message, other party");
    }
    {
        RandomSource& base = rng;  // through the abstract interface
        const hashes::Commitment cb = hashes::commit(sid, msg, base, 0, "label");
        check(hashes::verify_commit(cb.com, sid, msg, cb.opening, 0, "label"), "commit through RandomSource&");
    }
    check(hashes::ct_equal(msg.data(), msg.data(), msg.size()) && !hashes::ct_equal(msg.data(), msg2.data(), msg.size()), "ct_equal");
    // An empty update is a no-op, also with a null pointer and a partly filled block buffer
    // (formerly memcpy(dst, nullptr, 0)); an empty part of a framed hash is its length prefix only.
    {
        const bytes part = {1, 2, 3, 4, 5};
        Sha256 a, b;
        a.update(part.data(), part.size());
        b.update(part.data(), 2);
        b.update(nullptr, 0);
        b.update(part.data() + 2, 0);
        b.update(bytes());
        b.update(std::string());
        b.update(part.data() + 2, 3);
        Sha256 e;
        e.update(nullptr, 0);
        check(a.finish() == b.finish() && hex(e.finish()) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "Sha256::update with length 0 (null pointer included) changes nothing");
        const hashes::Part empty_part;  // {nullptr, 0}
        check(hashes::framed_sha256({"a", empty_part, "b"}) == hashes::framed_sha256({"a", bytes(), "b"}) &&
                  hashes::framed_sha256({"a", empty_part, "b"}) != hashes::framed_sha256({"a", "b"}),
              "an empty part of a framed hash is absorbed as its length prefix only");
    }
    // Session-dependent domains (helper of the C++ backend).
    check(hashes::session_dom_ccr("aand", sid) == 0x48052a3d3c886374ull, "session_dom_ccr known answer");
    check(hashes::session_dom_prg("cot", sid) == 0x509e6d6c57ee4206ull, "session_dom_prg known answer");
    check(hashes::session_dom_ccr("aand", sid) != hashes::session_dom_ccr("aand", str_bytes("session-2")), "session_dom depends on sid");
    check(hashes::session_dom_ccr("aand/0", sid) != hashes::session_dom_ccr("aand/1", sid), "session_dom depends on the label");
}

static void test_ascon_hash() {
    Section sec("Ascon-Hash256 / Ascon-XOF128 (SP 800-232)");
    struct V { size_t len; const char* out; };
    const V hv[] = {{0, "0b3be5850f2f6b98caf29f8fdea89b64a1fa70aa249b8f839bd53baa304d92b2"},
                    {1, "0728621035af3ed2bca03bf6fde900f9456f5330e4b5ee23e7f6a1e70291bc80"},
                    {7, "3e4d273ba69b3b9c53216107e88b75cdbeedbcbf8faf0219c3928ab62b116577"},
                    {8, "b88e497ae8e6fb641b87ef622eb8f2fca0ed95383f7ffebe167acf1099ba764f"},
                    {9, "94269c30e0296e1ec86655041841823efa1927f520fd58c8e9bce6197878c1a6"},
                    {15, "6421330df99c05eb715415ee17b455f2674f862ae3cc5badffe43a4a3ed273e1"},
                    {16, "3158c1940a2fbadbd68ab661777859b94a689e4efc375911467addd641835c38"},
                    {17, "f149e99dd0f429599bb89b8079bf3f4dca3f298efefcf9b1ea16fe84f9b8b6e2"},
                    {31, "b900cd3f06f1618b68c16665807206dbe273df40135361f449847d573903fabd"},
                    {32, "bd9d3d60a66b53868eab2a5c74539a518a1f60f01eb176c60e43dee81680b33e"},
                    {33, "a58665a2cb9530c502096a7957a76e428af4ad044b4da5c471f9da6f7b3e5868"},
                    {100, "a4bc453c84f824f10092e8e9031799957e984a29bbae5e84345e82f48dd71192"}};
    for (const V& v : hv) {
        bytes m(v.len);
        for (size_t i = 0; i < v.len; i++) m[i] = (uint8_t)i;
        check(hex(hashes::ascon_hash256(m)) == v.out, "Ascon-Hash256, message 00.. of length " + std::to_string(v.len));
    }
    const V xv[] = {
        {0, "473d5e6164f58b39dfd84aacdb8ae42ec2d91fed33388ee0d960d9b3993295c6ad77855a5d3b13fe6ad9e6098988373af7d0956d05a8f1665d2c67d1a3ad10ff"},
        {1, "51430e0438ecdf642b393630d977625f5f337656ba58ab1e960784ac32a16e0d446405551f5469384f8ea283cf12e64fa72c426bfebaea3aa1529e2c4ab23a2f"},
        {8, "8d1886f5d3ec4af8d15b44bc62b74da6ea91bc28fb82f9c34079b5ed6e38b6c951803d7dfb3c5e512a0ef5e4060062a6fd067f9c73ef9bee527411bda67fc896"},
        {33, "fef74b7ebd183ba1d87bf414000b29258d6a2233a2a03ed519c646b351bc008464cb725c2922e77a5e2b71f2d48e8d1ab34b45c3da91f5d46c9c3d9ae9057e02"}};
    for (const V& v : xv) {
        bytes m(v.len);
        for (size_t i = 0; i < v.len; i++) m[i] = (uint8_t)i;
        const bytes full = hashes::ascon_xof128(m, 64);
        check(hex(full) == v.out, "Ascon-XOF128 (64 bytes), message length " + std::to_string(v.len));
        bool ok = true;
        for (size_t n = 0; n <= 64; n++) {
            const bytes part = hashes::ascon_xof128(m, n);
            ok = ok && part.size() == n && std::equal(part.begin(), part.end(), full.begin());
        }
        check(ok, "Ascon-XOF128 output lengths 0..64 are prefixes");
        uint8_t h32[32];
        hashes::AsconXof128Hash32()(m.data(), m.size(), h32);
        check(std::memcmp(h32, full.data(), 32) == 0, "AsconXof128Hash32 functor");
    }
    {
        // pointer interface = vector interface
        const bytes m = str_bytes("pointer interface");
        hashes::hash32 h;
        hashes::ascon_hash256(m.data(), m.size(), h.data());
        check(h == hashes::ascon_hash256(m), "Ascon-Hash256 pointer interface");
    }
}

// One coin toss between two threads running the real protocol code.
struct TossResult {
    bytes out;
    ChannelStats stats;
    bool aborted = false;
    std::string error;
};

template <class R0, class R1>
static void run_coin_toss(const bytes& sid0, const std::string& label0, const bytes& sid1, const std::string& label1,
                          size_t n, R0& rng0, R1& rng1, TossResult& res0, TossResult& res1) {
    auto pair = LoopbackChannel::make_pair();
    LoopbackChannel* c0 = pair.first.get();
    LoopbackChannel* c1 = pair.second.get();
    std::thread t1([&] {
        try {
            res1.out = hashes::coin_toss(*c1, sid1, label1, n, rng1);
        } catch (const ProtocolAbort& e) {
            res1.aborted = true;
            res1.error = e.what();
        } catch (const std::exception& e) {
            res1.error = e.what();
        }
        res1.stats = c1->stats();
    });
    try {
        res0.out = hashes::coin_toss(*c0, sid0, label0, n, rng0);
    } catch (const ProtocolAbort& e) {
        res0.aborted = true;
        res0.error = e.what();
    } catch (const std::exception& e) {
        res0.error = e.what();
    }
    res0.stats = c0->stats();
    t1.join();
}

// The honest party `honest_party` runs coin_toss(sid, "chi", 32) against a
// scripted peer that drives the other end of the channel message by message.
// `unread` (optional) receives the number of messages the honest party sent
// that the script did not read.
static TossResult honest_vs_script(int honest_party, const bytes& sid, const std::function<void(Channel&)>& script,
                                   size_t* unread = nullptr) {
    auto pair = LoopbackChannel::make_pair();
    Channel* hc = honest_party == 0 ? static_cast<Channel*>(pair.first.get()) : static_cast<Channel*>(pair.second.get());
    Channel* ac = honest_party == 0 ? static_cast<Channel*>(pair.second.get()) : static_cast<Channel*>(pair.first.get());
    TossResult res;
    std::thread honest([&] {
        CtrPrgRng rng(0x40e57 + (uint64_t)honest_party);
        try {
            res.out = hashes::coin_toss(*hc, sid, "chi", 32, rng);
        } catch (const ProtocolAbort& e) {
            res.aborted = true;
            res.error = e.what();
        } catch (const std::exception& e) {
            res.error = e.what();
        }
    });
    script(*ac);
    honest.join();
    if (unread) *unread = (honest_party == 0 ? pair.second : pair.first)->pending();
    return res;
}

static void test_coin_toss() {
    Section sec("committed_exchange and coin_toss over a channel");
    const bytes sid = str_bytes("session-1");
    const std::string label = "chi";
    // Known answers of the building blocks (computed by couplet/hashes.py).
    hashes::opening32 omega;
    std::array<uint8_t, 32> r;
    for (size_t i = 0; i < 32; i++) {
        omega[i] = (uint8_t)i;
        r[i] = (uint8_t)(100 + i);
    }
    check(hex(hashes::coin_label(label)) == hex(str_bytes("coin|chi")), "coin_label = 'coin|' || label");
    check(hex(hashes::commit_with_opening(sid, r, omega, 0, hashes::coin_label(label))) ==
              "2f34751e19f34b248f78b994584a3043517b99f47f649e07cc93950ed6cd2d9f",
          "coin share commitment of party 0, known answer");
    check(hex(hashes::commit_with_opening(sid, r, omega, 1, hashes::coin_label(label))) ==
              "ce97db9c941551ab6d4e8c67db50061a7e19c8027b083ed982843698224f0294",
          "coin share commitment of party 1, known answer");
    check(hex(hashes::coin_expand(sid, label, omega, 80)) ==
              "7da6b3ef668ed02ad0cbcb218572bf430eab7c7103d89b64d183273c94a50f38e3291f0314534477d2d60c313f16fceefbe220f5b4bfc0f452b22ffc4d710e15f16ebeef6af5c59ce40dfe1f89699a29",
          "coin_expand known answer (80 bytes)");
    {
        const bytes e100 = hashes::coin_expand(sid, label, omega, 100);
        bool ok = true;
        for (size_t n = 0; n <= 100; n++) {
            const bytes e = hashes::coin_expand(sid, label, omega, n);
            ok = ok && e.size() == n && std::equal(e.begin(), e.end(), e100.begin());
        }
        check(ok, "coin_expand lengths 0..100 are prefixes");
        hashes::seed16 s16;
        std::memcpy(s16.data(), e100.data(), 16);
        const std::vector<gf128> g = hashes::coin_gf128_from_seed(s16, 2);
        check(g[0] == gf128(0x3c873bd3f59d95f9ull, 0x3e4e8342ab149f1eull) && g[1] == gf128(0x8b3f3e10688c39e1ull, 0x57da10859a8b9d68ull),
              "coin GF(2^128) expansion known answer");
    }

    // committed_exchange delivers both messages (of different lengths) in two rounds.
    {
        auto pair = LoopbackChannel::make_pair();
        LoopbackChannel* c0 = pair.first.get();
        LoopbackChannel* c1 = pair.second.get();
        CtrPrgRng rng0(21), rng1(22);
        const bytes m0 = str_bytes("from-0"), m1 = str_bytes("from-one");
        bytes got0, got1;
        std::thread t1([&] { got1 = hashes::committed_exchange(*c1, sid, "v", m1, rng1); });
        got0 = hashes::committed_exchange(*c0, sid, "v", m0, rng0);
        t1.join();
        check(got0 == m1 && got1 == m0, "committed_exchange delivers both messages");
        check(c0->stats().rounds == 2 && c1->stats().rounds == 2, "committed_exchange: two rounds");
        check(c0->stats().bytes_sent == 32 + 32 + m0.size() && c1->stats().bytes_sent == 32 + 32 + m1.size(),
              "committed_exchange: 32 + 32 + len(msg) payload bytes");
    }
    {
        // empty messages
        auto pair = LoopbackChannel::make_pair();
        LoopbackChannel* c0 = pair.first.get();
        LoopbackChannel* c1 = pair.second.get();
        CtrPrgRng rng0(23), rng1(24);
        bytes got0(1), got1(1);
        std::thread t1([&] { got1 = hashes::committed_exchange(*c1, sid, "e", bytes(), rng1); });
        got0 = hashes::committed_exchange(*c0, sid, "e", bytes(), rng0);
        t1.join();
        check(got0.empty() && got1.empty(), "committed_exchange with empty messages");
    }

    // Honest runs with deterministic randomness: the output is predictable from the definition.
    const size_t sizes[] = {0, 1, 16, 32, 33, 100};
    for (size_t n : sizes) {
        CtrPrgRng rng0(1000 + n), rng1(2000 + n);
        TossResult a, b;
        run_coin_toss(sid, label, sid, label, n, rng0, rng1, a, b);
        check(!a.aborted && !b.aborted && a.error.empty() && b.error.empty(), "honest run completes, n = " + std::to_string(n));
        check(a.out == b.out && a.out.size() == n, "both parties get the same n bytes, n = " + std::to_string(n));
        CtrPrgRng e0(1000 + n), e1(2000 + n);
        const std::array<uint8_t, 32> r0 = e0.array<32>(), r1 = e1.array<32>();
        std::array<uint8_t, 32> seed;
        for (size_t i = 0; i < 32; i++) seed[i] = (uint8_t)(r0[i] ^ r1[i]);
        check(a.out == hashes::coin_expand(sid, label, seed, n), "output = sha256_expand(n, 'Couplet-coin', sid, label, r_0 XOR r_1)");
        check(a.stats.rounds == 2 && b.stats.rounds == 2, "two rounds");
        check(a.stats.bytes_sent == 96 && a.stats.bytes_recv == 96 && b.stats.bytes_sent == 96 && b.stats.bytes_recv == 96,
              "96 payload bytes in each direction");
        check(a.stats.msgs_sent == 2 && b.stats.msgs_sent == 2, "two messages per party");
    }
    // Fresh system randomness: outputs agree and change between runs.
    {
        SystemRng s0, s1;
        TossResult a, b, c, d;
        run_coin_toss(sid, label, sid, label, 32, s0, s1, a, b);
        run_coin_toss(sid, label, sid, label, 32, s0, s1, c, d);
        check(a.out == b.out && c.out == d.out && a.out.size() == 32, "system randomness: parties agree");
        check(a.out != c.out, "system randomness: two tosses differ");
    }
    // Parties that disagree on sid or on the label abort (the commitments are bound to both).
    {
        CtrPrgRng rng0(1), rng1(2);
        TossResult a, b;
        run_coin_toss(sid, label, str_bytes("session-2"), label, 32, rng0, rng1, a, b);
        check(a.aborted && b.aborted, "different sid on the two sides: both abort");
        CtrPrgRng rng2(1), rng3(2);
        TossResult c, d;
        run_coin_toss(sid, label, sid, "psi", 32, rng2, rng3, c, d);
        check(c.aborted && d.aborted, "different label on the two sides: both abort");
    }
    // Same randomness, different sid or label: different coins.
    {
        CtrPrgRng a0(1), a1(2), b0(1), b1(2), c0(1), c1(2);
        TossResult a, b, c, d, e, f;
        run_coin_toss(sid, label, sid, label, 32, a0, a1, a, b);
        const bytes sid2 = str_bytes("session-2");
        run_coin_toss(sid2, label, sid2, label, 32, b0, b1, c, d);
        run_coin_toss(sid, "psi", sid, "psi", 32, c0, c1, e, f);
        check(!a.aborted && !c.aborted && a.out != c.out, "output depends on sid");
        check(!e.aborted && a.out != e.out, "output depends on the label");
    }

    const bytes clabel = hashes::coin_label(label);
    auto opening_msg = [](const hashes::opening32& om, const bytes& m) {
        bytes o(om.begin(), om.end());
        o.insert(o.end(), m.begin(), m.end());
        return o;
    };
    const bytes rb(r.begin(), r.end());

    // A scripted party that follows the protocol is accepted (the scripted driver itself is sound).
    {
        bytes peer_open;
        const TossResult h = honest_vs_script(1, sid, [&](Channel& ch) {
            const hashes::hash32 com = hashes::commit_with_opening(sid, rb, omega, 0, clabel);
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            ch.send(opening_msg(omega, rb));
            peer_open = ch.recv();
        });
        std::array<uint8_t, 32> seed{};
        if (peer_open.size() == 64)
            for (size_t i = 0; i < 32; i++) seed[i] = (uint8_t)(r[i] ^ peer_open[32 + i]);
        check(!h.aborted && h.error.empty() && peer_open.size() == 64 && h.out == hashes::coin_expand(sid, label, seed, 32),
              "scripted honest party 0 is accepted; wire format = commitment, then omega || share");
    }
    // Cheat 1: party 1 mirrors the commitment it has just received (would force the seed to zero).
    {
        const TossResult h = honest_vs_script(0, sid, [&](Channel& ch) {
            const bytes com = ch.recv();
            ch.send(com);
        });
        check(h.aborted && h.out.empty(), "mirrored commitment: ProtocolAbort (" + h.error + ")");
    }
    // Cheat 2: party 1 commits correctly, but under party 0's index, and opens that commitment.
    {
        const TossResult h = honest_vs_script(0, sid, [&](Channel& ch) {
            (void)ch.recv();
            const hashes::hash32 com = hashes::commit_with_opening(sid, rb, omega, 0, clabel);
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            ch.send(opening_msg(omega, rb));
        });
        check(h.aborted, "commitment bound to the wrong party index: ProtocolAbort (" + h.error + ")");
    }
    // Cheat 3: a commitment without any party binding.
    {
        const TossResult h = honest_vs_script(0, sid, [&](Channel& ch) {
            (void)ch.recv();
            const hashes::hash32 com = hashes::commit_with_opening(sid, rb, omega, hashes::NO_PARTY, clabel);
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            ch.send(opening_msg(omega, rb));
        });
        check(h.aborted, "commitment without party binding: ProtocolAbort");
    }
    // Cheat 4: party 0 opens a different value than it committed to.
    {
        const TossResult h = honest_vs_script(1, sid, [&](Channel& ch) {
            const hashes::hash32 com = hashes::commit_with_opening(sid, rb, omega, 0, clabel);
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            bytes other = rb;
            other[0] ^= 1;
            ch.send(opening_msg(omega, other));
            (void)ch.recv();
        });
        check(h.aborted, "opening a different share: ProtocolAbort (" + h.error + ")");
    }
    // Cheat 5: a different opening string omega.
    {
        const TossResult h = honest_vs_script(1, sid, [&](Channel& ch) {
            const hashes::hash32 com = hashes::commit_with_opening(sid, rb, omega, 0, clabel);
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            hashes::opening32 om = omega;
            om[5] ^= 0x10;
            ch.send(opening_msg(om, rb));
            (void)ch.recv();
        });
        check(h.aborted, "wrong opening string: ProtocolAbort");
    }
    // Cheat 6: a commitment made under another label (e.g. taken from another toss).
    {
        const TossResult h = honest_vs_script(1, sid, [&](Channel& ch) {
            const hashes::hash32 com = hashes::commit_with_opening(sid, rb, omega, 0, hashes::coin_label("psi"));
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            ch.send(opening_msg(omega, rb));
            (void)ch.recv();
        });
        check(h.aborted, "commitment from another label: ProtocolAbort");
    }
    // Cheat 7: a correctly committed and opened share of the wrong length.  The opening of a coin
    // share is bounded to 32 + 32 bytes (channel.hpp): a longer one is refused when it arrives, so
    // an honest party 1 aborts WITHOUT sending its own opening (nothing is left for the script to
    // read); an honest party 0 has opened already.  A share that is too short is opened and then
    // refused by the length check of coin_toss.
    {
        size_t unread = 99;
        const TossResult h = honest_vs_script(1, sid, [&](Channel& ch) {
            const bytes longshare(33, 0x11);
            const hashes::hash32 com = hashes::commit_with_opening(sid, longshare, omega, 0, clabel);
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            ch.send(opening_msg(omega, longshare));
        }, &unread);
        check(h.aborted && h.error.find("wrong length") != std::string::npos, "coin share of the wrong length: ProtocolAbort (" + h.error + ")");
        check(unread == 0, "an over-long opening is refused before the honest party 1 opens its own commitment");
        const TossResult h0 = honest_vs_script(0, sid, [&](Channel& ch) {
            const bytes longshare(33, 0x11);
            const hashes::hash32 com = hashes::commit_with_opening(sid, longshare, omega, 1, clabel);
            (void)ch.recv();
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            ch.send(opening_msg(omega, longshare));
        });
        check(h0.aborted && h0.error.find("wrong length") != std::string::npos, "the same against an honest party 0 (" + h0.error + ")");
        const TossResult hs = honest_vs_script(1, sid, [&](Channel& ch) {
            const bytes shortshare(31, 0x11);
            const hashes::hash32 com = hashes::commit_with_opening(sid, shortshare, omega, 0, clabel);
            ch.send(bytes(com.begin(), com.end()));
            (void)ch.recv();
            ch.send(opening_msg(omega, shortshare));
            (void)ch.recv();
        });
        check(hs.aborted && hs.error.find("coin share has the wrong length") != std::string::npos,
              "a coin share that is too short: ProtocolAbort (" + hs.error + ")");
        const TossResult hc = honest_vs_script(0, sid, [&](Channel& ch) {
            (void)ch.recv();
            ch.send(bytes(33, 0));
        });
        check(hc.aborted && hc.error.find("malformed commitment") != std::string::npos, "an over-long commitment: ProtocolAbort (" + hc.error + ")");
    }
    // committed_exchange with an explicit limit for the peer's message, and without one.
    {
        const bytes m0 = str_bytes("from-0"), m1 = str_bytes("from-one");
        for (int limit_case = 0; limit_case < 3; limit_case++) {
            // 0: P_0 accepts up to 8 bytes (enough); 1: up to 7 bytes (one too few); 2: no limit of its own.
            auto pair = LoopbackChannel::make_pair();
            LoopbackChannel* c0 = pair.first.get();
            CtrPrgRng rng0(31), rng1(32);
            bytes got0, got1;
            std::string err0;
            std::thread t1([&] {
                std::unique_ptr<LoopbackChannel> c1 = std::move(pair.second);  // closed when this party is done
                try {
                    got1 = hashes::committed_exchange(*c1, sid, "v", m1, rng1);
                } catch (const std::exception&) {
                }
            });
            try {
                if (limit_case == 2) got0 = hashes::committed_exchange(*c0, sid, "v", m0, rng0);
                else got0 = hashes::committed_exchange(*c0, sid, "v", m0, rng0, limit_case == 0 ? 8 : 7);
            } catch (const ProtocolAbort& e) {
                err0 = e.what();
            }
            pair.first.reset();
            t1.join();
            if (limit_case == 1) check(got0.empty() && err0.find("opening has the wrong length") != std::string::npos,
                                       "committed_exchange refuses a message above the caller's limit (" + err0 + ")");
            else check(got0 == m1 && got1 == m0 && err0.empty(), "committed_exchange within the limit, case " + std::to_string(limit_case));
        }
    }
    // Malformed messages.
    {
        const TossResult h = honest_vs_script(0, sid, [&](Channel& ch) {
            (void)ch.recv();
            ch.send(bytes(31, 0));
        });
        check(h.aborted, "short commitment: ProtocolAbort");
        const TossResult h2 = honest_vs_script(1, sid, [&](Channel& ch) {
            ch.send(bytes(32, 0x77));
            (void)ch.recv();
            ch.send(bytes(31, 0));
            (void)ch.recv();
        });
        check(h2.aborted, "short opening: ProtocolAbort");
    }
    // ProtocolAbort is a std::runtime_error.
    {
        bool caught = false;
        try {
            throw ProtocolAbort("x");
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "x";
        }
        check(caught, "ProtocolAbort derives from std::runtime_error");
    }
    // GF(2^128) coins.
    {
        auto pair = LoopbackChannel::make_pair();
        LoopbackChannel* c0 = pair.first.get();
        LoopbackChannel* c1 = pair.second.get();
        CtrPrgRng rng0(11), rng1(12);
        std::vector<gf128> g0, g1;
        std::thread t1([&] { g1 = hashes::coin_toss_gf128(*c1, sid, label, 1000, rng1); });
        g0 = hashes::coin_toss_gf128(*c0, sid, label, 1000, rng0);
        t1.join();
        check(g0 == g1 && g0.size() == 1000, "coin_toss_gf128: parties agree");
        CtrPrgRng e0(11), e1(12);
        const std::array<uint8_t, 32> r0 = e0.array<32>(), r1 = e1.array<32>();
        std::array<uint8_t, 32> seed;
        for (size_t i = 0; i < 32; i++) seed[i] = (uint8_t)(r0[i] ^ r1[i]);
        const bytes b0 = hashes::coin_expand(sid, label, seed, 16);
        hashes::seed16 s16;
        std::memcpy(s16.data(), b0.data(), 16);
        check(g0 == hashes::ctr_prg_gf128(s16, hashes::dom::PRG_COIN, 1000), "coin_toss_gf128 = ctr_prg_gf128(16 coin bytes, PRG_COIN)");
    }
}

// ------------------------------------------------------------ rng tests
static void test_rng() {
    Section sec("rng (SystemRng, CtrPrgRng)");
    SystemRng sys;
    sys.fill(nullptr, 0);
    const bytes a = sys.get(32), b = sys.get(32);
    check(a.size() == 32 && b.size() == 32 && a != b, "SystemRng: two 32-byte draws differ");
    {
        const size_t n = 100000;
        bytes big(n + 8, 0xee);
        sys.fill(big.data(), n);
        bool tail_ok = true;
        for (size_t i = n; i < n + 8; i++) tail_ok = tail_ok && big[i] == 0xee;
        check(tail_ok, "SystemRng: no write past the end");
        size_t ones = 0;
        for (size_t i = 0; i < n; i++)
            for (int k = 0; k < 8; k++) ones += (big[i] >> k) & 1;
        // 800000 fair bits: mean 400000, standard deviation 447; the window is about 9 sigma.
        check(ones > 396000 && ones < 404000, "SystemRng: bit balance over 100000 bytes (ones = " + std::to_string(ones) + ")");
        int hist[256] = {0};
        for (size_t i = 0; i < n; i++) hist[big[i]]++;
        int mn = hist[0], mx = hist[0];
        for (int v : hist) {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        // Each byte value is expected 390.6 times (standard deviation 19.7).
        check(mn > 250 && mx < 540, "SystemRng: every byte value occurs a plausible number of times");
    }
    for (size_t n : {(size_t)1, (size_t)3, (size_t)255, (size_t)256, (size_t)257}) {
        const bytes x = sys.get(n);
        check(x.size() == n, "SystemRng: fill of " + std::to_string(n) + " bytes");
    }
    // Deterministic generator.
    hashes::seed16 seed;
    for (size_t i = 0; i < 16; i++) seed[i] = (uint8_t)(0xf0 + i);
    CtrPrgRng r1(seed), r2(seed);
    const bytes ref = hashes::ctr_prg_bytes(seed, hashes::dom::PRG_RNG, 5000);
    check(r1.get(5000) == ref, "CtrPrgRng = byte view of ctr_prg(seed, PRG_RNG)");
    bytes pieces;
    while (pieces.size() < 5000) {
        const size_t take = (pieces.size() % 97) + 1;
        const bytes p = r2.get(take);
        pieces.insert(pieces.end(), p.begin(), p.end());
    }
    pieces.resize(5000);
    check(pieces == ref, "CtrPrgRng: independent of how reads are split");
    CtrPrgRng r3(seed, hashes::dom::PRG_TEST);
    check(r3.get(64) != bytes(ref.begin(), ref.begin() + 64), "CtrPrgRng: depends on the domain");
    CtrPrgRng u1(42), u2(42), u3(43);
    check(u1.u64() == u2.u64(), "CtrPrgRng(uint64): reproducible");
    check(u1.get(16) != u3.get(16), "CtrPrgRng(uint64): depends on the seed");
    check(hex(CtrPrgRng::seed_from_u64(0x0102030405060708ull)) == "08070605040302010000000000000000", "seed_from_u64");
    // Typed helpers.
    {
        CtrPrgRng h1(seed), h2(seed);
        const bytes raw = h1.get(8 + 16 + 16 + 3);
        check(h2.u64() == ascon::load64le(raw.data()), "u64() is little-endian");
        check(h2.gf() == gf128::from_bytes(raw.data() + 8), "gf() = gf128::from_bytes");
        const hashes::seed16 sd = h2.seed();
        check(std::memcmp(sd.data(), raw.data() + 24, 16) == 0, "seed()");
        uint8_t bits[20];
        h2.fill_bits(bits, 20);
        bool ok = true;
        for (size_t i = 0; i < 20; i++) ok = ok && bits[i] == ((raw[40 + i / 8] >> (i % 8)) & 1);
        check(ok, "fill_bits layout");
        CtrPrgRng h3(seed), h4(seed);
        std::vector<uint8_t> many(5001);
        h3.fill_bits(many.data(), many.size());
        const bytes rawbits = h4.get(626);
        ok = true;
        for (size_t i = 0; i < many.size(); i++) ok = ok && many[i] == ((rawbits[i / 8] >> (i % 8)) & 1);
        check(ok, "fill_bits across internal chunks");
        std::vector<gf128> g(5);
        CtrPrgRng h5(seed), h6(seed);
        h5.fill_gf(g.data(), 5);
        const bytes rawg = h6.get(80);
        ok = true;
        for (size_t i = 0; i < 5; i++) ok = ok && g[i] == gf128::from_bytes(rawg.data() + 16 * i);
        check(ok, "fill_gf layout");
    }
    CtrPrgRng f1 = CtrPrgRng::from_system(), f2 = CtrPrgRng::from_system();
    check(f1.get(32) != f2.get(32), "CtrPrgRng::from_system: independent instances differ");
    // random_scalar accepts every source type.
    const scalar32 s1 = ed25519::random_scalar(sys);
    RandomSource& base = f1;
    const scalar32 s2 = ed25519::random_scalar(base);
    check(ed25519::scalar_is_reduced(s1) && ed25519::scalar_is_reduced(s2) && s1 != s2, "random_scalar from SystemRng / RandomSource&");
}

// ------------------------------------------------------------ benchmarks
static volatile uint64_t g_sink = 0;

static void run_benchmarks(double table_build_ms) {
    CtrPrgRng rng(0xbe9c);

    // H_ccr
    const size_t NH = (size_t)1 << 16;
    std::vector<gf128> hx(NH), hy(NH);
    rng.fill_gf(hx.data(), NH);
    double t0 = now_ns();
    const int hreps = 200;
    for (int rep = 0; rep < hreps; rep++) hashes::H_ccr_batch(hx.data(), NH, (uint64_t)rep * NH, hashes::dom::CCR_TEST, hy.data());
    double t1 = now_ns();
    const double h_batch_ns = (t1 - t0) / ((double)hreps * (double)NH);
    g_sink = g_sink ^ hy[NH - 1].lo;
    t0 = now_ns();
    const int sreps = 40;
    for (int rep = 0; rep < sreps; rep++)
        for (size_t i = 0; i < NH; i++) hy[i] = hashes::H_ccr(hx[i], (uint64_t)i, hashes::dom::CCR_TEST);
    t1 = now_ns();
    const double h_scalar_ns = (t1 - t0) / ((double)sreps * (double)NH);
    g_sink = g_sink ^ hy[NH - 1].hi;

    // ctr_prg
    const size_t NW = (size_t)1 << 20;  // 8 MiB per call
    std::vector<uint64_t> words(NW);
    hashes::seed16 seed = rng.seed();
    const int preps = 40;
    t0 = now_ns();
    for (int rep = 0; rep < preps; rep++) {
        seed[0] = (uint8_t)rep;
        hashes::ctr_prg(seed, hashes::dom::PRG_TEST, words.data(), NW);
    }
    t1 = now_ns();
    const double prg_mbps = ((double)preps * (double)NW * 8.0 / 1e6) / ((t1 - t0) / 1e9);
    g_sink = g_sink ^ words[NW - 1];
    std::vector<gf128> gfbuf(NW / 2);
    t0 = now_ns();
    for (int rep = 0; rep < preps; rep++) {
        seed[0] = (uint8_t)rep;
        hashes::ctr_prg_gf128(seed, hashes::dom::PRG_TEST, gfbuf.data(), gfbuf.size());
    }
    t1 = now_ns();
    const double prg_gf_mbps = ((double)preps * (double)NW * 8.0 / 1e6) / ((t1 - t0) / 1e9);
    g_sink = g_sink ^ gfbuf.back().lo;
    bytes bitbuf((size_t)1 << 22);
    t0 = now_ns();
    for (int rep = 0; rep < 10; rep++) {
        seed[0] = (uint8_t)rep;
        hashes::ctr_prg_bits(seed, hashes::dom::PRG_TEST, bitbuf.data(), bitbuf.size());
    }
    t1 = now_ns();
    const double prg_bits_ns = (t1 - t0) / (10.0 * (double)bitbuf.size());
    g_sink = g_sink ^ bitbuf.back();
    bytes rbuf((size_t)1 << 23);
    CtrPrgRng fast(seed);
    t0 = now_ns();
    for (int rep = 0; rep < 20; rep++) fast.fill(rbuf.data(), rbuf.size());
    t1 = now_ns();
    const double rng_mbps = (20.0 * (double)rbuf.size() / 1e6) / ((t1 - t0) / 1e9);
    g_sink = g_sink ^ rbuf.back();
    SystemRng sys;
    uint8_t small[32];
    t0 = now_ns();
    for (int i = 0; i < 20000; i++) sys.fill(small, 32);
    t1 = now_ns();
    const double sys32_ns = (t1 - t0) / 20000.0;
    t0 = now_ns();
    for (int rep = 0; rep < 4; rep++) sys.fill(rbuf.data(), rbuf.size());
    t1 = now_ns();
    const double sys_mbps = (4.0 * (double)rbuf.size() / 1e6) / ((t1 - t0) / 1e9);
    g_sink = g_sink ^ small[0] ^ rbuf[1];

    // Edwards25519
    const int NP = 64;
    std::vector<scalar32> sc((size_t)NP);
    std::vector<Point> pts((size_t)NP);
    for (int i = 0; i < NP; i++) {
        sc[(size_t)i] = ed25519::random_scalar(rng);
        pts[(size_t)i] = ed25519::mul_base(ed25519::random_scalar(rng));
    }
    Point acc;
    const int mreps = 4;
    t0 = now_ns();
    for (int rep = 0; rep < mreps; rep++)
        for (int i = 0; i < NP; i++) acc = ed25519::add(acc, ed25519::mul(pts[(size_t)i], sc[(size_t)((i + rep) % NP)]));
    t1 = now_ns();
    const double mul_us = (t1 - t0) / 1e3 / (double)(mreps * NP);
    t0 = now_ns();
    for (int i = 0; i < NP; i++) acc = ed25519::add(acc, ed25519::mul_ladder(pts[(size_t)i], sc[(size_t)i]));
    t1 = now_ns();
    const double ladder_us = (t1 - t0) / 1e3 / (double)NP;
    const int breps = 16;
    t0 = now_ns();
    for (int rep = 0; rep < breps; rep++)
        for (int i = 0; i < NP; i++) acc = ed25519::add(acc, ed25519::mul_base(sc[(size_t)i]));
    t1 = now_ns();
    const double base_us = (t1 - t0) / 1e3 / (double)(breps * NP);
    // hash_to_group on 32-byte messages (compressed points, as in the base OT).
    std::vector<ed25519::bytes32> encs((size_t)NP);
    for (int i = 0; i < NP; i++) encs[(size_t)i] = ed25519::compress(pts[(size_t)i]);
    const int greps = 8;
    uint64_t tries_total = 0;
    t0 = now_ns();
    for (int rep = 0; rep < greps; rep++)
        for (int i = 0; i < NP; i++) {
            uint32_t tries = 0;
            const std::string tag = "bench-h2g-" + std::to_string(rep);
            acc = ed25519::add(acc, ed25519::hash_to_group(tag, encs[(size_t)i], &tries));
            tries_total += tries + 1;
        }
    t1 = now_ns();
    const double h2g_us = (t1 - t0) / 1e3 / (double)(greps * NP);
    const double h2g_attempts = (double)tries_total / (double)(greps * NP);
    t0 = now_ns();
    for (int rep = 0; rep < 8; rep++)
        for (int i = 0; i < NP; i++) {
            Point q;
            ed25519::decompress(q, encs[(size_t)i]);
            acc = ed25519::add(acc, q);
        }
    t1 = now_ns();
    const double dec_us = (t1 - t0) / 1e3 / (double)(8 * NP);
    uint8_t encsink = 0;
    t0 = now_ns();
    for (int rep = 0; rep < 8; rep++)
        for (int i = 0; i < NP; i++) encsink ^= ed25519::compress(pts[(size_t)i])[0];
    t1 = now_ns();
    const double enc_us = (t1 - t0) / 1e3 / (double)(8 * NP);
    int members = 0;
    t0 = now_ns();
    for (int i = 0; i < NP; i++) members += ed25519::in_prime_subgroup(pts[(size_t)i]) ? 1 : 0;
    t1 = now_ns();
    const double subgroup_us = (t1 - t0) / 1e3 / (double)NP;
    t0 = now_ns();
    for (int rep = 0; rep < 200; rep++)
        for (int i = 0; i < NP; i++) ed25519::add_into(acc, pts[(size_t)i]);
    t1 = now_ns();
    const double add_ns = (t1 - t0) / (double)(200 * NP);
    t0 = now_ns();
    for (int rep = 0; rep < 200 * NP; rep++) ed25519::dbl_into(acc);
    t1 = now_ns();
    const double dbl_ns = (t1 - t0) / (double)(200 * NP);
    g_sink = g_sink ^ ed25519::compress(acc)[0] ^ encsink ^ (uint64_t)members;

    double fmul_ns = 0;
    {
        x25519::gf fa, fb;
        for (int i = 0; i < 16; i++) {
            fa[i] = (x25519::i64)(rng.u64() & 0xffff);
            fb[i] = (x25519::i64)(rng.u64() & 0xffff);
        }
        const int freps = 2000000;
        t0 = now_ns();
        for (int i = 0; i < freps; i++) ed25519::detail::M(fa, fa, fb);
        t1 = now_ns();
        fmul_ns = (t1 - t0) / (double)freps;
        g_sink = g_sink ^ (uint64_t)fa[0];
    }

    // Commitments and coin tossing.
    const bytes sid = str_bytes("bench-session");
    const bytes msg16(16, 0x42);
    hashes::Commitment cm;
    t0 = now_ns();
    for (int i = 0; i < 200000; i++) cm = hashes::commit(sid, msg16, fast, i & 1, "bench");
    t1 = now_ns();
    const double commit_ns = (t1 - t0) / 200000.0;
    g_sink = g_sink ^ cm.com[0];
    std::array<uint8_t, 32> cseed = rng.array<32>();
    t0 = now_ns();
    bytes ce;
    for (int i = 0; i < 20; i++) {
        cseed[0] = (uint8_t)i;
        ce = hashes::coin_expand(sid, "bench", cseed, (size_t)1 << 20);
    }
    t1 = now_ns();
    const double expand_mbps = (20.0 * (double)((size_t)1 << 20) / 1e6) / ((t1 - t0) / 1e9);
    g_sink = g_sink ^ ce.back();
    double toss_us = 0;
    {
        auto pair = LoopbackChannel::make_pair();
        LoopbackChannel* c0 = pair.first.get();
        LoopbackChannel* c1 = pair.second.get();
        const int treps = 2000;
        CtrPrgRng ra(1), rb(2);
        t0 = now_ns();
        uint64_t sink_a = 0, sink_b = 0;
        std::thread th([&] {
            for (int i = 0; i < treps; i++) sink_b ^= hashes::coin_toss(*c1, sid, "bench", 16, rb)[0];
        });
        for (int i = 0; i < treps; i++) sink_a ^= hashes::coin_toss(*c0, sid, "bench", 16, ra)[1];
        th.join();
        t1 = now_ns();
        g_sink = g_sink ^ sink_a ^ sink_b;
        toss_us = (t1 - t0) / 1e3 / (double)treps;
    }

    std::printf("BENCH_JSON\n{\n");
#if defined(_MSC_VER)
    std::printf("  \"compiler\": \"MSVC %d\",\n", _MSC_VER);
#elif defined(__clang__)
    std::printf("  \"compiler\": \"clang %s\",\n", __clang_version__);
#elif defined(__GNUC__)
    std::printf("  \"compiler\": \"gcc %d.%d\",\n", __GNUC__, __GNUC_MINOR__);
#else
    std::printf("  \"compiler\": \"unknown\",\n");
#endif
    std::printf("  \"avx2\": %s,\n", COUPLET_HAVE_AVX2 ? "true" : "false");
    std::printf("  \"ed25519_avx2_field_mul\": %s,\n", COUPLET_ED25519_AVX2 ? "true" : "false");
    std::printf("  \"ed25519_field_mul_ns\": %.1f,\n", fmul_ns);
    std::printf("  \"H_ccr_batched_ns_per_call\": %.2f,\n", h_batch_ns);
    std::printf("  \"H_ccr_scalar_ns_per_call\": %.2f,\n", h_scalar_ns);
    std::printf("  \"ctr_prg_words_MBps\": %.1f,\n", prg_mbps);
    std::printf("  \"ctr_prg_gf128_MBps\": %.1f,\n", prg_gf_mbps);
    std::printf("  \"ctr_prg_bits_ns_per_bit\": %.3f,\n", prg_bits_ns);
    std::printf("  \"CtrPrgRng_fill_MBps\": %.1f,\n", rng_mbps);
    std::printf("  \"SystemRng_32B_ns\": %.1f,\n", sys32_ns);
    std::printf("  \"SystemRng_bulk_MBps\": %.1f,\n", sys_mbps);
    std::printf("  \"ed25519_mul_us\": %.2f,\n", mul_us);
    std::printf("  \"ed25519_mul_ladder_us\": %.2f,\n", ladder_us);
    std::printf("  \"ed25519_mul_base_us\": %.2f,\n", base_us);
    std::printf("  \"ed25519_mul_base_table_build_ms\": %.3f,\n", table_build_ms);
    std::printf("  \"ed25519_hash_to_group_us\": %.2f,\n", h2g_us);
    std::printf("  \"ed25519_hash_to_group_attempts_mean\": %.3f,\n", h2g_attempts);
    std::printf("  \"ed25519_decompress_us\": %.2f,\n", dec_us);
    std::printf("  \"ed25519_compress_us\": %.2f,\n", enc_us);
    std::printf("  \"ed25519_in_prime_subgroup_us\": %.2f,\n", subgroup_us);
    std::printf("  \"ed25519_add_ns\": %.1f,\n", add_ns);
    std::printf("  \"ed25519_dbl_ns\": %.1f,\n", dbl_ns);
    std::printf("  \"commit_16B_ns\": %.1f,\n", commit_ns);
    std::printf("  \"sha256_expand_MBps\": %.1f,\n", expand_mbps);
    std::printf("  \"coin_toss_loopback_us\": %.2f,\n", toss_us);
    std::printf("  \"sink\": \"%016llx\"\n", (unsigned long long)g_sink);
    std::printf("}\n");
}

// committed_exchange and coin_toss against a peer on 127.0.0.1:port.
static int run_interop(int party, uint16_t port) {
    try {
        std::unique_ptr<TcpChannel> ch = TcpChannel::create(party, "127.0.0.1", port, 20000);
        SystemRng rng;
        const bytes sid = str_bytes("interop-session");
        const bytes mine = str_bytes(party == 0 ? "message-from-party-0" : "message-from-party-1");
        const bytes peer = hashes::committed_exchange(*ch, sid, "v", mine, rng);
        std::printf("PEERMSG %s\n", hex(peer).c_str());
        const bytes coin = hashes::coin_toss(*ch, sid, "chi", 48, rng);
        std::printf("COIN %s\n", hex(coin).c_str());
        const std::vector<gf128> g = hashes::coin_toss_gf128(*ch, sid, "chi-gf", 2, rng);
        std::printf("COINGF %016llx %016llx %016llx %016llx\n", (unsigned long long)g[0].lo, (unsigned long long)g[0].hi,
                    (unsigned long long)g[1].lo, (unsigned long long)g[1].hi);
        const ChannelStats& st = ch->stats();
        std::printf("STATS rounds=%llu bytes_sent=%llu bytes_recv=%llu\n", (unsigned long long)st.rounds,
                    (unsigned long long)st.bytes_sent, (unsigned long long)st.bytes_recv);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "interop failed: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    if (argc == 4 && !std::strcmp(argv[1], "--interop")) {
        const int party = std::atoi(argv[2]);
        const int port = std::atoi(argv[3]);
        if ((party != 0 && party != 1) || port <= 0 || port > 65535) {
            std::fprintf(stderr, "usage: %s --interop PARTY(0|1) PORT\n", argv[0]);
            return 2;
        }
        return run_interop(party, (uint16_t)port);
    }
    bool do_tests = true, do_bench = true;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--no-bench")) do_bench = false;
        else if (!std::strcmp(argv[i], "--bench-only")) do_tests = false;
        else {
            std::fprintf(stderr, "usage: %s [--no-bench | --bench-only] | --interop PARTY PORT\n", argv[0]);
            return 2;
        }
    }
    // The first fixed-base multiplication builds the table: time it once.
    const double tb0 = now_ns();
    const Point warm = ed25519::mul_base(sc_from_u64(1));
    const double table_build_ms = (now_ns() - tb0) / 1e6;
    if (!ed25519::equal(warm, ed25519::base())) {
        std::printf("FAIL: 1 * B != B\n");
        return 1;
    }

    if (do_tests) {
        test_sha512();
        test_ed_constants();
        test_ed_field_mul();
        test_ed_rfc8032();
        test_ed_order();
        test_ed_python_vectors();
        test_ed_group_laws();
        test_ed_encoding();
        test_ed_hash_to_group();
        test_h_ccr();
        test_ctr_prg();
        test_framing_commit();
        test_ascon_hash();
        test_coin_toss();
        test_rng();
        std::printf("RESULT: %d checks passed, %d failed\n", g_pass, g_fail);
        std::fflush(stdout);
        if (g_fail) return 1;
    }
    if (do_bench) run_benchmarks(table_build_ms);
    return 0;
}
