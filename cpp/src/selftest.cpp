// selftest.cpp -- known-answer and consistency tests for the couplet C++
// primitives (ascon, gf128, x25519/sha256, channel) and of the network
// emulation of the drivers (netem.hpp: token bucket, delay and bandwidth limit
// of DelayTcpChannel; these checks assert wall-clock times, with a slack of
// NETEM_SLACK_S above the emulated time and none below it).  Exit code 0 iff
// all checks pass.  Lines starting with VECTOR print reference outputs for
// cross-checking against the Python implementation.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "couplet/ascon.hpp"
#include "couplet/channel.hpp"
#include "couplet/netem.hpp"
#if defined(_WIN32)
#include <psapi.h>
#endif
#include "couplet/gf128.hpp"
#include "couplet/sha256.hpp"
#include "couplet/u128.hpp"
#include "couplet/x25519.hpp"

using namespace couplet;

// ------------------------------------------------------------ harness
static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) g_pass++; else g_fail++;
}

static std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(2 * n);
    for (size_t i = 0; i < n; i++) { s.push_back(d[p[i] >> 4]); s.push_back(d[p[i] & 15]); }
    return s;
}
template <size_t N>
static std::string hex(const std::array<uint8_t, N>& a) { return hex(a.data(), N); }

static std::vector<uint8_t> unhex(const std::string& s) {
    std::vector<uint8_t> v;
    for (size_t i = 0; i + 1 < s.size(); i += 2) v.push_back((uint8_t)std::strtoul(s.substr(i, 2).c_str(), nullptr, 16));
    return v;
}
template <size_t N>
static std::array<uint8_t, N> unhex_arr(const std::string& s) {
    std::array<uint8_t, N> a{};
    std::vector<uint8_t> v = unhex(s);
    for (size_t i = 0; i < N && i < v.size(); i++) a[i] = v[i];
    return a;
}

struct Rng {  // xorshift64*, deterministic
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
};

// ------------------------------------------------------------ Ascon
// Ascon-Hash (v1.2 / SP 800-232 Ascon-Hash256): IV 0x00400c0000000100, rate
// 64, 12 rounds, big-endian word loading as in the reference implementation.
static void ascon_hash256(const uint8_t* m, size_t len, uint8_t out[32]) {
    uint64_t s[5] = {0x00400c0000000100ull, 0, 0, 0, 0};
    ascon::p12(s);
    while (len >= 8) {
        s[0] ^= ascon::load64be(m);
        ascon::p12(s);
        m += 8;
        len -= 8;
    }
    uint8_t last[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::memcpy(last, m, len);
    last[len] = 0x80;
    s[0] ^= ascon::load64be(last);
    ascon::p12(s);
    for (int i = 0; i < 4; i++) {
        ascon::store64be(out + 8 * i, s[0]);
        if (i < 3) ascon::p12(s);
    }
}

// Bit extraction reference over the 40-byte little-endian state image.
static u128 bits_ref(const uint64_t st[5], unsigned off, unsigned w) {
    uint8_t img[40];
    for (int i = 0; i < 5; i++) ascon::store64le(img + 8 * i, st[i]);
    u128 r = 0;
    for (unsigned i = 0; i < w; i++) {
        unsigned b = off + i;
        unsigned bit = (img[b / 8] >> (b % 8)) & 1;
        r = r | (u128(bit) << i);
    }
    return r;
}

static void test_ascon() {
    uint8_t out[32];
    ascon_hash256(nullptr, 0, out);
    check(hex(out, 32) == "7346bc14f036e87ae03d0997913088f5f68411434b3cf8b54fa796a80d251f91",
          "Ascon-Hash256(\"\") KAT (validates p12)");

    // Precomputed initial state of Ascon-Hash (p12 applied to the IV), as in
    // the reference implementation's constants.
    {
        uint64_t s[5] = {0x00400c0000000100ull, 0, 0, 0, 0};
        ascon::p12(s);
        check(s[0] == 0xee9398aadb67f03dull && s[1] == 0x8bb21831c60f1002ull && s[2] == 0xb48a92db98d5da62ull &&
                  s[3] == 0x43189921b8f8e3e8ull && s[4] == 0x348fa5c9d525e140ull,
              "p12(Ascon-Hash IV) matches reference precomputed state");
    }

    // Interleaved versions agree with the scalar permutation.
    {
        Rng rng(1);
        bool ok4 = true, ok8 = true, okavx = true;
        for (int trial = 0; trial < 50; trial++) {
            uint64_t ref[8][5], a4[4][5], a8[8][5], av[4][5];
            for (int j = 0; j < 8; j++)
                for (int w = 0; w < 5; w++) ref[j][w] = rng.next();
            std::memcpy(a4, ref, sizeof a4);
            std::memcpy(a8, ref, sizeof a8);
            std::memcpy(av, ref, sizeof av);
            for (int j = 0; j < 8; j++) ascon::p12(ref[j]);
            ascon::p12_x4(a4);
            ascon::p12_x8(a8);
            for (int j = 0; j < 4; j++) ok4 = ok4 && std::memcmp(a4[j], ref[j], 40) == 0;
            for (int j = 0; j < 8; j++) ok8 = ok8 && std::memcmp(a8[j], ref[j], 40) == 0;
#if COUPLET_HAVE_AVX2
            ascon::p12_x4_avx2(av);
            for (int j = 0; j < 4; j++) okavx = okavx && std::memcmp(av[j], ref[j], 40) == 0;
#endif
        }
        check(ok4, "p12_x4 (scalar interleaved) == p12");
        check(ok8, "p12_x8 (scalar interleaved) == p12");
#if COUPLET_HAVE_AVX2
        check(okavx, "p12_x4_avx2 == p12");
#else
        std::printf("[SKIP] p12_x4_avx2 (not compiled with AVX2)\n");
#endif
    }

    // IV words (ASCII, zero padded, little-endian).
    check(ascon::IV_G().w[0] == 0x2d74656c70756f43ull && ascon::IV_G().w[1] == 0x47ull && ascon::IV_G().w[2] == 0,
          "IV_G = LE(\"Couplet-G\" || 0^15)");
    check(ascon::IV_C().w[0] == 0x2d74656c70756f43ull && ascon::IV_C().w[1] == 0x43ull && ascon::IV_C().w[2] == 0,
          "IV_C = LE(\"Couplet-C\" || 0^15)");
    check(ascon::IV_VL().w[1] == 0x4c56ull && ascon::IV_VR().w[1] == 0x5256ull && ascon::IV_VL().w[2] == 0,
          "IV_VL / IV_VR words");

    // Feed-forward (DESIGN.md v3.1, change A): prg_state = p12(s||IV) XOR (s||IV).
    ascon::seed_t seed;
    for (int i = 0; i < 16; i++) seed[i] = (uint8_t)i;
    {
        const ascon::IV* ivs[4] = {&ascon::IV_G(), &ascon::IV_C(), &ascon::IV_VL(), &ascon::IV_VR()};
        bool ok = true, differs = true;
        for (int l = 0; l < 4; l++) {
            uint64_t in[5] = {ascon::load64le(seed.data()), ascon::load64le(seed.data() + 8), ivs[l]->w[0],
                              ivs[l]->w[1], ivs[l]->w[2]};
            uint64_t p[5], q[5], y[5];
            std::memcpy(p, in, sizeof p);
            ascon::p12(p);
            ascon::prg_permute(q, seed.data(), *ivs[l]);
            ascon::prg_state(y, seed.data(), *ivs[l]);
            ok = ok && std::memcmp(p, q, 40) == 0;
            for (int i = 0; i < 5; i++) ok = ok && y[i] == (p[i] ^ in[i]);
            differs = differs && std::memcmp(p, y, 40) != 0;
        }
        check(ok, "prg_state = p12(s||IV) XOR (s||IV), prg_permute = p12(s||IV) (all four labels)");
        check(differs, "prg_state differs from the bare permutation output");
    }

    // G layout against a hand-built state.
    {
        uint64_t s[5] = {ascon::load64le(seed.data()), ascon::load64le(seed.data() + 8), ascon::IV_G().w[0],
                         ascon::IV_G().w[1], ascon::IV_G().w[2]};
        uint64_t in[5];
        std::memcpy(in, s, sizeof in);
        ascon::p12(s);
        for (int i = 0; i < 5; i++) s[i] ^= in[i];  // feed-forward
        ascon::GOut g = ascon::G(seed);
        uint8_t exp[40];
        for (int i = 0; i < 5; i++) ascon::store64le(exp + 8 * i, s[i]);
        bool ok = std::memcmp(g.sL.data(), exp, 16) == 0 && std::memcmp(g.sR.data(), exp + 16, 16) == 0 &&
                  g.tL == (s[4] & 1) && g.tR == ((s[4] >> 1) & 1);
        check(ok, "G(seed) layout: sL=x0x1, sR=x2x3, tL=x4 bit0, tR=x4 bit1 of Y (little-endian)");
        // array overload
        uint8_t sL[16], sR[16], tL, tR;
        ascon::G(seed.data(), sL, tL, sR, tR);
        check(std::memcmp(sL, g.sL.data(), 16) == 0 && std::memcmp(sR, g.sR.data(), 16) == 0 && tL == g.tL && tR == g.tR,
              "G pointer overload == G array overload");
        // G_x4
        ascon::seed_t in4[4];
        ascon::GOut out4[4];
        Rng rng(7);
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 16; i++) in4[j][i] = (uint8_t)rng.next();
        ascon::G_x4(in4, out4);
        bool ok4 = true;
        for (int j = 0; j < 4; j++) {
            ascon::GOut r = ascon::G(in4[j]);
            ok4 = ok4 && r.sL == out4[j].sL && r.sR == out4[j].sR && r.tL == out4[j].tL && r.tR == out4[j].tR;
        }
        check(ok4, "G_x4 == 4 x G");
    }

    // Batched PRG evaluations (array-of-states and structure-of-arrays forms,
    // mixed labels, every batch size 1..9) against the scalar prg_state.
    {
        Rng rng(11);
        const ascon::IV* ivs[4] = {&ascon::IV_G(), &ascon::IV_C(), &ascon::IV_VL(), &ascon::IV_VR()};
        bool ok_batch = true, ok_x4 = true, ok_soa = true;
        for (int trial = 0; trial < 200; trial++) {
            ascon::seed_t sd[9];
            ascon::PrgCall calls[9];
            for (int j = 0; j < 9; j++) {
                for (int i = 0; i < 16; i++) sd[j][i] = (uint8_t)rng.next();
                calls[j].seed = sd[j].data();
                calls[j].iv = ivs[rng.next() & 3];
            }
            const size_t count = 1 + (size_t)(trial % 9);
            uint64_t y[9][5], ref[5];
            ascon::prg_state_batch(calls, count, y);
            for (size_t j = 0; j < count; j++) {
                ascon::prg_state(ref, calls[j].seed, *calls[j].iv);
                ok_batch = ok_batch && std::memcmp(ref, y[j], 40) == 0;
            }
            const ascon::IV& iv = *ivs[trial & 3];
            uint64_t y4[4][5], ysoa[5][4], lo[4], hi[4];
            ascon::prg_state_x4(y4, sd, iv);
            for (int j = 0; j < 4; j++) {
                lo[j] = ascon::load64le(sd[j].data());
                hi[j] = ascon::load64le(sd[j].data() + 8);
            }
            ascon::prg_state_x4_soa(lo, hi, iv, ysoa);
            for (int j = 0; j < 4; j++) {
                ascon::prg_state(ref, sd[j].data(), iv);
                ok_x4 = ok_x4 && std::memcmp(ref, y4[j], 40) == 0;
                for (int w = 0; w < 5; w++) ok_soa = ok_soa && ysoa[w][j] == ref[w];
            }
        }
        check(ok_batch, "prg_state_batch == scalar prg_state (batch sizes 1..9, mixed labels)");
        check(ok_x4, "prg_state_x4 == 4 x prg_state");
        check(ok_soa, "prg_state_x4_soa == 4 x prg_state (structure-of-arrays layout)");
    }

    // Known answers of the PRG maps, produced by the Python reference
    // (couplet/ascon.py, feed-forward PRG) for the seeds 00*16 and 00 01 .. 0f.
    {
        struct Kat {
            const char* seed;
            const char* sL; int tL; const char* sR; int tR;
            const char* conv64_v; const char* conv64_t; const char* conv32_v; const char* conv32_t;
            const char* vl64_v; const char* vl64_t; const char* vl32_v; const char* vl32_t;
            const char* vr64_v; const char* vr64_t; const char* vr32_v; const char* vr32_t;
        };
        const Kat kats[2] = {
            {"00000000000000000000000000000000",
             "33b3e32e5dc99c71022842feb484b931", 0, "0d2aeab3177a516eec82b61f34c6e3fa", 1,
             "1375a120735f3144bf62cd87488c81c0", "ad04883383cc7190ee5c38d52abcfb78",
             "735f3144bf62cd87488c81c0", "ee5c38d52abcfb781375a120",
             "ea31428c7891679cfef88e5b68e607a1", "d569ee6edc8e73939169f984bec15f24",
             "7891679cfef88e5b68e607a1", "9169f984bec15f24ea31428c",
             "650962348e8e29418ceee679ad1f0bb0", "a757cde7e9aa75a7af06183db18da88b",
             "8e8e29418ceee679ad1f0bb0", "af06183db18da88b65096234"},
            {"000102030405060708090a0b0c0d0e0f",
             "8074cca913e87c9cd766f9f2993fb0bc", 1, "af956bc2da0691ba2b43611ea30caa13", 0,
             "e00562e0c370791af8dfc702afe058e8", "38f7fc9aeea3b13ba148aeeaac2ac9d6",
             "c370791af8dfc702afe058e8", "a148aeeaac2ac9d6e00562e0",
             "bc9ff75f4fa6d1a7b69154d78b2d83ef", "0f8a64fbe371ca8b63c03e42fa661da1",
             "4fa6d1a7b69154d78b2d83ef", "63c03e42fa661da1bc9ff75f",
             "84a19a7dd4bb180a11b5d4b5d6bc2326", "fcb0bc38471c693e47856a48eb37df62",
             "d4bb180a11b5d4b5d6bc2326", "47856a48eb37df6284a19a7d"},
        };
        for (const Kat& kt : kats) {
            const ascon::seed_t sd = unhex_arr<16>(kt.seed);
            const std::string tag = std::string(" KAT, seed ") + kt.seed;
            ascon::GOut g = ascon::G(sd);
            check(hex(g.sL) == kt.sL && g.tL == kt.tL && hex(g.sR) == kt.sR && g.tR == kt.tR, "G" + tag);
            ascon::ConvOut c64 = ascon::Conv(sd, 64, 64), c32 = ascon::Conv(sd, 32, 64);
            check(c64.val == u128_from_hex(kt.conv64_v) && c64.tag == u128_from_hex(kt.conv64_t) &&
                      c32.val == u128_from_hex(kt.conv32_v) && c32.tag == u128_from_hex(kt.conv32_t),
                  "Conv (k=64 and k=32, s=64)" + tag);
            ascon::ConvOut l64 = ascon::ConvV_L(sd, 64, 64), l32 = ascon::ConvV_L(sd, 32, 64);
            check(l64.val == u128_from_hex(kt.vl64_v) && l64.tag == u128_from_hex(kt.vl64_t) &&
                      l32.val == u128_from_hex(kt.vl32_v) && l32.tag == u128_from_hex(kt.vl32_t),
                  "ConvV_L (k=64 and k=32, s=64)" + tag);
            ascon::ConvOut r64 = ascon::ConvV_R(sd, 64, 64), r32 = ascon::ConvV_R(sd, 32, 64);
            check(r64.val == u128_from_hex(kt.vr64_v) && r64.tag == u128_from_hex(kt.vr64_t) &&
                      r32.val == u128_from_hex(kt.vr32_v) && r32.tag == u128_from_hex(kt.vr32_t),
                  "ConvV_R (k=64 and k=32, s=64)" + tag);
        }
    }

    // Conv layouts.
    {
        uint64_t s[5];
        ascon::prg_state(s, seed.data(), ascon::IV_C());
        ascon::ConvOut c = ascon::Conv(seed, 64, 64);
        check(c.val == u128_from(s[0], s[1]) && c.tag == u128_from(s[2], s[3]),
              "Conv k=64,s=64: val = x0||x1, tag = x2||x3");
        ascon::ConvOut c32 = ascon::Conv(seed, 32, 64);
        check(c32.val == bits_ref(s, 0, 96) && c32.tag == bits_ref(s, 96, 96),
              "Conv k=32,s=64: val = bits[0,96), tag = bits[96,192)");
        uint64_t sl[5], sr[5];
        ascon::prg_state(sl, seed.data(), ascon::IV_VL());
        ascon::prg_state(sr, seed.data(), ascon::IV_VR());
        ascon::ConvOut vl = ascon::ConvV_L(seed, 64, 64), vr = ascon::ConvV_R(seed, 64, 64);
        check(vl.val == u128_from(sl[0], sl[1]) && vl.tag == u128_from(sl[2], sl[3]) &&
                  vr.val == u128_from(sr[0], sr[1]) && vr.tag == u128_from(sr[2], sr[3]),
              "ConvV_L / ConvV_R use IV_VL / IV_VR");
        check(!(vl.val == vr.val) && !(c.val == vl.val), "distinct IVs give distinct outputs");

        // extract_bits vs. byte-level reference for many (off, w).
        bool ok = true;
        Rng rng(3);
        for (int t = 0; t < 2000 && ok; t++) {
            uint64_t st[5];
            for (int i = 0; i < 5; i++) st[i] = rng.next();
            unsigned w = 1 + (unsigned)(rng.next() % 128);
            unsigned off = (unsigned)(rng.next() % (320 - w + 1));
            ok = ascon::extract_bits(st, off, w) == bits_ref(st, off, w);
        }
        check(ok, "extract_bits == bit-level reference (2000 random ranges)");

        // conv_x4
        ascon::seed_t in4[4];
        ascon::ConvOut out4[4];
        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 16; i++) in4[j][i] = (uint8_t)rng.next();
        ascon::conv_x4(in4, ascon::IV_C(), 128, out4);
        bool ok4 = true;
        for (int j = 0; j < 4; j++) {
            ascon::ConvOut r = ascon::Conv(in4[j], 64, 64);
            ok4 = ok4 && r.val == out4[j].val && r.tag == out4[j].tag;
        }
        check(ok4, "conv_x4 == 4 x Conv");
    }

    // Reference vectors for cross-checking with Python.
    ascon::seed_t zero{};
    const ascon::seed_t seeds[2] = {zero, seed};
    const char* names[2] = {"00*16", "000102..0f"};
    for (int i = 0; i < 2; i++) {
        ascon::GOut g = ascon::G(seeds[i]);
        ascon::ConvOut c = ascon::Conv(seeds[i], 64, 64);
        ascon::ConvOut l = ascon::ConvV_L(seeds[i], 64, 64);
        ascon::ConvOut r = ascon::ConvV_R(seeds[i], 64, 64);
        std::printf("VECTOR G(%s): sL=%s tL=%d sR=%s tR=%d\n", names[i], hex(g.sL).c_str(), g.tL, hex(g.sR).c_str(), g.tR);
        std::printf("VECTOR Conv(%s,k=64,s=64): val=0x%s tag=0x%s\n", names[i], u128_to_hex(c.val).c_str(),
                    u128_to_hex(c.tag).c_str());
        std::printf("VECTOR ConvV_L(%s,k=64,s=64): val=0x%s tag=0x%s\n", names[i], u128_to_hex(l.val).c_str(),
                    u128_to_hex(l.tag).c_str());
        std::printf("VECTOR ConvV_R(%s,k=64,s=64): val=0x%s tag=0x%s\n", names[i], u128_to_hex(r.val).c_str(),
                    u128_to_hex(r.tag).c_str());
    }
}

// ------------------------------------------------------------ u128
static void test_u128() {
    u128 a = u128_from(0xffffffffffffffffull, 0x0123456789abcdefull);
    u128 b = u128_from(1, 0);
    check(u128_hi(a + b) == 0x0123456789abcdf0ull && u128_lo(a + b) == 0, "u128 add carries");
    check(u128_lo((a + b) - b) == 0xffffffffffffffffull && u128_hi((a + b) - b) == 0x0123456789abcdefull, "u128 sub borrows");
    check(u128_lo(-b) == 0xffffffffffffffffull && u128_hi(-b) == 0xffffffffffffffffull, "u128 negate");
    u128 m = u128_from(0xffffffffffffffffull, 0) * u128_from(0xffffffffffffffffull, 0);
    check(u128_lo(m) == 1 && u128_hi(m) == 0xfffffffffffffffeull, "u128 mul (2^64-1)^2");
    u128 m2 = u128_from(3, 5) * u128_from(7, 11);  // (5*2^64+3)(11*2^64+7) mod 2^128 = 21 + (35+33) 2^64
    check(u128_lo(m2) == 21 && u128_hi(m2) == 68, "u128 mul cross terms");
    check(u128_lo(u128_mask(96)) == 0xffffffffffffffffull && u128_hi(u128_mask(96)) == 0xffffffffull, "u128_mask(96)");
    check(u128_mask(128) == ~u128(0) && u128_mask(0) == u128(0), "u128_mask(128)/(0)");
    check(u128_to_hex(a) == "0123456789abcdefffffffffffffffff", "u128_to_hex");
    check(u128_from_hex("0123456789abcdefffffffffffffffff") == a, "u128_from_hex");
    uint8_t buf[16];
    u128_store_le(buf, a);
    check(u128_load_le(buf) == a && buf[0] == 0xff && buf[15] == 0x01, "u128 LE serialisation");
    check(u128_lo((a << 68)) == 0 && u128_hi(a << 68) == 0xfffffffffffffff0ull, "u128 shl");
    check(u128_lo(a >> 68) == 0x00123456789abcdeull && u128_hi(a >> 68) == 0, "u128 shr");
    check(u128_lo(u128_mod2k(a, 100)) == 0xffffffffffffffffull && u128_hi(u128_mod2k(a, 100)) == 0x789abcdefull,
          "u128_mod2k(100)");
}

// ------------------------------------------------------------ GF(2^128)
// Slow reference: schoolbook polynomial multiplication over coefficient
// arrays followed by long division by x^128 + x^7 + x^2 + x + 1.
static gf128 gf128_slow(gf128 a, gf128 b) {
    uint8_t prod[256];
    std::memset(prod, 0, sizeof prod);
    for (unsigned i = 0; i < 128; i++)
        if (a.bit(i))
            for (unsigned j = 0; j < 128; j++) prod[i + j] ^= (uint8_t)b.bit(j);
    for (int i = 254; i >= 128; i--) {
        if (prod[i]) {
            prod[i] = 0;
            prod[i - 128 + 7] ^= 1;
            prod[i - 128 + 2] ^= 1;
            prod[i - 128 + 1] ^= 1;
            prod[i - 128 + 0] ^= 1;
        }
    }
    gf128 r;
    for (unsigned i = 0; i < 64; i++) { r.lo |= (uint64_t)prod[i] << i; r.hi |= (uint64_t)prod[64 + i] << i; }
    return r;
}

static void test_gf128() {
    std::printf("gf128 backend: %s\n", COUPLET_HAVE_PCLMUL ? "PCLMULQDQ" : "portable");

    // Analytic known answers (checks the reduction polynomial and bit order).
    struct { gf128 a, b, expect; const char* name; } kats[] = {
        {gf128(0, 1), gf128(0, 1), gf128(0x87, 0), "x^64 * x^64 = x^128 = x^7+x^2+x+1"},
        {gf128(0, 1ull << 63), gf128(2, 0), gf128(0x87, 0), "x^127 * x = 0x87"},
        {gf128(0, 1ull << 63), gf128(4, 0), gf128(0x10e, 0), "x^127 * x^2 = 0x10e"},
        {gf128(0, 1ull << 63), gf128(0, 1ull << 63), gf128(0x1067, 0xC000000000000000ull), "x^127 * x^127"},
        {gf128(1, 0), gf128(0x0123456789abcdefull, 0xfedcba9876543210ull), gf128(0x0123456789abcdefull, 0xfedcba9876543210ull), "1 * a = a"},
    };
    for (auto& k : kats) {
        bool ok = gf128_mul(k.a, k.b) == k.expect && gf128_mul_portable(k.a, k.b) == k.expect && gf128_slow(k.a, k.b) == k.expect;
        check(ok, std::string("gf128 analytic KAT: ") + k.name);
    }

    // Five fixed known-answer pairs computed by the slow reference.
    const char* pairs[5][2] = {
        {"00000000000000000000000000000002", "80000000000000000000000000000000"},
        {"0123456789abcdef0123456789abcdef", "fedcba9876543210fedcba9876543210"},
        {"deadbeefdeadbeefdeadbeefdeadbeef", "0000000000000000cafebabecafebabe"},
        {"ffffffffffffffffffffffffffffffff", "ffffffffffffffffffffffffffffffff"},
        {"9e3779b97f4a7c15f39cc0605cedc834", "1082276bf3a27251f86c6a11d0c18e95"},
    };
    bool allok = true;
    for (int i = 0; i < 5; i++) {
        u128 ua = u128_from_hex(pairs[i][0]), ub = u128_from_hex(pairs[i][1]);
        gf128 a(u128_lo(ua), u128_hi(ua)), b(u128_lo(ub), u128_hi(ub));
        gf128 ref = gf128_slow(a, b);
        gf128 fast = gf128_mul(a, b), port = gf128_mul_portable(a, b);
        bool ok = fast == ref && port == ref;
        allok = allok && ok;
        std::printf("VECTOR gf128 %s * %s = %s\n", a.hex().c_str(), b.hex().c_str(), ref.hex().c_str());
    }
    check(allok, "gf128 5 known-answer pairs (fast & portable == slow reference)");

    // Random agreement and field identities.
    Rng rng(42);
    bool agree = true, comm = true, assoc = true, dist = true, inv = true, mulx = true;
    for (int t = 0; t < 1000; t++) {
        gf128 a(rng.next(), rng.next()), b(rng.next(), rng.next()), c(rng.next(), rng.next());
        gf128 ab = gf128_mul(a, b);
        if (t < 200) agree = agree && ab == gf128_slow(a, b);
        agree = agree && ab == gf128_mul_portable(a, b);
        comm = comm && ab == gf128_mul(b, a);
        assoc = assoc && gf128_mul(ab, c) == gf128_mul(a, gf128_mul(b, c));
        dist = dist && gf128_mul(a, b ^ c) == (ab ^ gf128_mul(a, c));
        mulx = mulx && gf128_mulx(a) == gf128_mul(a, gf128::x());
        if (t < 50 && !a.is_zero()) inv = inv && gf128_mul(a, gf128_inv(a)) == gf128::one();
    }
    check(agree, "gf128 fast == portable (1000) == slow (200) on random inputs");
    check(comm, "gf128 commutative");
    check(assoc, "gf128 associative");
    check(dist, "gf128 distributive");
    check(mulx, "gf128_mulx == mul by x");
    check(inv, "gf128 a * inv(a) == 1 (50 random a)");
    uint8_t buf[16];
    gf128 a(0x0123456789abcdefull, 0xfedcba9876543210ull);
    a.to_bytes(buf);
    check(buf[0] == 0xef && buf[15] == 0xfe && gf128::from_bytes(buf) == a, "gf128 little-endian serialisation");
}

// ------------------------------------------------------------ SHA-256 / X25519
static void test_sha256() {
    check(hex(sha256(std::string(""))) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256(\"\")");
    check(hex(sha256(std::string("abc"))) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256(\"abc\")");
    check(hex(sha256(std::string("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "SHA-256(56-byte two-block message)");
    {
        Sha256 h;
        std::string chunk(1000, 'a');
        for (int i = 0; i < 1000; i++) h.update(chunk);
        check(hex(h.finish()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "SHA-256(10^6 x 'a') incremental");
    }
    {
        // incremental == one-shot on odd chunk sizes
        std::vector<uint8_t> msg(1000);
        Rng rng(5);
        for (auto& b : msg) b = (uint8_t)rng.next();
        Sha256 h;
        size_t pos = 0, step = 1;
        while (pos < msg.size()) {
            size_t n = step;
            if (pos + n > msg.size()) n = msg.size() - pos;
            h.update(msg.data() + pos, n);
            pos += n;
            step = (step * 7 + 3) % 97 + 1;
        }
        check(h.finish() == sha256(msg), "SHA-256 incremental == one-shot");
    }
}

static void test_x25519() {
    using namespace x25519;
    {
        bytes32 k = unhex_arr<32>("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
        bytes32 u = unhex_arr<32>("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
        check(hex(scalarmult(k, u)) == "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552", "X25519 RFC 7748 vector 1");
    }
    {
        bytes32 k = unhex_arr<32>("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d");
        bytes32 u = unhex_arr<32>("e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493");
        check(hex(scalarmult(k, u)) == "95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957", "X25519 RFC 7748 vector 2");
    }
    {
        bytes32 k = basepoint(), u = basepoint();
        bytes32 r = scalarmult(k, u);
        check(hex(r) == "422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079", "X25519 iterated, 1 iteration");
        u = k;
        k = r;
        for (int i = 1; i < 1000; i++) {
            bytes32 t = scalarmult(k, u);
            u = k;
            k = t;
        }
        check(hex(k) == "684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51", "X25519 iterated, 1000 iterations");
    }
    {
        bytes32 a = unhex_arr<32>("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        bytes32 b = unhex_arr<32>("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
        bytes32 A = scalarmult_base(a), B = scalarmult_base(b);
        check(hex(A) == "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a", "X25519 Alice public key");
        check(hex(B) == "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f", "X25519 Bob public key");
        bytes32 s1 = scalarmult(a, B), s2 = scalarmult(b, A);
        check(s1 == s2 && hex(s1) == "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742", "X25519 shared secret");
    }
    {
        bytes32 in = unhex_arr<32>("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
        bytes32 h = hash_to_x(in);
        uint8_t raw[32];
        sha256(in.data(), 32, raw);
        bool ok = (h[31] & 0x80) == 0 && std::memcmp(h.data(), raw, 31) == 0 && h[31] == (raw[31] & 0x7f);
        check(ok, "hash_to_x = SHA-256 with top bit cleared");
        std::printf("VECTOR hash_to_x(%s) = %s\n", hex(in).c_str(), hex(h).c_str());
        // the result is usable as an x-coordinate
        bytes32 k = unhex_arr<32>("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        bytes32 r = scalarmult(k, h);
        std::printf("VECTOR X25519(alice_sk, hash_to_x(alice_pk)) = %s\n", hex(r).c_str());
    }
}

// ------------------------------------------------------------ channels
static bytes make_msg(Rng& rng, size_t n) {
    bytes b(n);
    for (auto& x : b) x = (uint8_t)rng.next();
    return b;
}

static void run_channel_protocol(Channel& ch, bool& ok) {
    // Deterministic script executed by both parties (same seed per party).
    Rng rng(100 + ch.party());
    Rng peer_rng(100 + (1 - ch.party()));
    ok = true;
    // 1. exchange: small, empty, large
    const size_t sizes[3] = {17, 0, 8u << 20};
    for (size_t sz : sizes) {
        bytes mine = make_msg(rng, sz), expect = make_msg(peer_rng, sz);
        bytes peer = ch.exchange(mine);
        ok = ok && peer == expect;
    }
    // 2. three sends in one flight from party 0, then one reply
    if (ch.party() == 0) {
        for (int i = 0; i < 3; i++) ch.send(make_msg(rng, 100 + i));
        bytes r = ch.recv();
        ok = ok && r == make_msg(peer_rng, 50);
    } else {
        for (int i = 0; i < 3; i++) {
            bytes r = ch.recv();
            ok = ok && r == make_msg(peer_rng, 100 + i);
        }
        ch.send(make_msg(rng, 50));
    }
    ch.flight();
    // 3. many small exchanges
    for (int i = 0; i < 200; i++) {
        bytes mine = make_msg(rng, 1 + i % 7), expect = make_msg(peer_rng, 1 + i % 7);
        ok = ok && ch.exchange(mine) == expect;
    }
}

static void check_stats(const Channel& c0, const Channel& c1, const char* name) {
    const ChannelStats &s0 = c0.stats(), &s1 = c1.stats();
    uint64_t exp_bytes0 = 17 + 0 + (8u << 20) + 100 + 101 + 102;
    for (int i = 0; i < 200; i++) exp_bytes0 += 1 + i % 7;
    uint64_t exp_bytes1 = 17 + 0 + (8u << 20) + 50;
    for (int i = 0; i < 200; i++) exp_bytes1 += 1 + i % 7;
    bool ok = s0.bytes_sent == exp_bytes0 && s1.bytes_recv == exp_bytes0 && s1.bytes_sent == exp_bytes1 &&
              s0.bytes_recv == exp_bytes1 && s0.msgs_sent == 3 + 3 + 200 && s1.msgs_recv == 206 &&
              s1.msgs_sent == 3 + 1 + 200 && s0.msgs_recv == 204 && s0.rounds == 3 + 1 + 200 && s1.rounds == 204 &&
              s0.send_flights == 3 + 1 + 200 && s1.send_flights == 3 + 1 + 200;
    check(ok, std::string(name) + " byte/message/round accounting");
    std::printf("  %s stats P0=%s\n  %s stats P1=%s\n", name, s0.json().c_str(), name, s1.json().c_str());
}

// Peak commit charge of this process in bytes (0 if the platform does not tell).
static size_t peak_commit_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    std::memset(&pmc, 0, sizeof pmc);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, (DWORD)sizeof pmc)) return 0;
    return (size_t)pmc.PeakPagefileUsage;
#else
    return 0;
#endif
}

// A raw TCP connection to 127.0.0.1:port (the adversarial peer of the framing tests).
static net::socket_t raw_connect(uint16_t port) {
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(0x7f000001u);
    const net::socket_t s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == net::invalid_socket) throw std::runtime_error("socket failed");
    if (::connect(s, (const struct sockaddr*)&addr, (int)sizeof addr) != 0) {
        net::close_socket(s);
        throw std::runtime_error(net::err_string("raw connect failed"));
    }
    return s;
}
static void raw_header(net::socket_t s, uint32_t n) {
    uint8_t hdr[4];
    for (int i = 0; i < 4; i++) hdr[i] = (uint8_t)(n >> (8 * i));
    net::send_all(s, hdr, 4);
}

// What a receiving call did: "ok:<n>" (a message of n bytes), "long:<length>/<limit>"
// (MessageTooLong), "abort:<text>" (another ProtocolAbort) or "error:<text>".
template <class F>
static std::string outcome(F&& f) {
    try {
        const bytes b = f();
        return "ok:" + std::to_string(b.size());
    } catch (const MessageTooLong& e) {
        return "long:" + std::to_string(e.length) + "/" + std::to_string(e.limit);
    } catch (const ProtocolAbort& e) {
        return std::string("abort:") + e.what();
    } catch (const std::exception& e) {
        return std::string("error:") + e.what();
    }
}

// Bounded receive (channel.hpp): limits named by the caller and the endpoint-wide limit.
static void test_bounded_receive() {
    // In-process channel: the message exists already; its size is compared.
    {
        auto pr = LoopbackChannel::make_pair();
        Channel& a = *pr.first;
        Channel& b = *pr.second;
        check(a.max_message() == Channel::DEFAULT_MAX_MESSAGE && Channel::DEFAULT_MAX_MESSAGE == ((size_t)1 << 30) &&
                  Channel::MAX_MESSAGE == 0x7fffffff,
              "default limit of a receive without a caller's limit: 2^30 bytes");
        a.send(bytes(10, 1));
        check(outcome([&] { return b.recv(10); }) == "ok:10", "recv(max): a message of exactly the limit is delivered");
        a.send(bytes(11, 1));
        check(outcome([&] { return b.recv(10); }) == "long:11/10", "recv(max): a longer message raises MessageTooLong");
        a.send(bytes(11, 1));
        check(outcome([&] { return b.recv(10, "share has the wrong length"); }) == "abort:share has the wrong length",
              "recv(max, error): ProtocolAbort with the caller's text");
        check(b.stats().msgs_recv == 1 && b.stats().bytes_recv == 10, "a refused message is not counted as received");
        b.set_max_message(4);
        a.send(bytes(5, 1));
        check(outcome([&] { return b.recv(); }) == "long:5/4", "recv(): the endpoint-wide limit applies");
        a.send(bytes(5, 1));
        check(outcome([&] { return b.recv(5); }) == "ok:5", "recv(max): the caller's limit is not capped by the endpoint-wide one");
        a.send(bytes(3, 1));
        check(outcome([&] { return b.recv_exact(3); }) == "ok:3", "recv_exact: exact length");
        a.send(bytes(4, 1));
        check(outcome([&] { return b.recv_exact(3); }) == "long:4/3", "recv_exact: a longer message is refused");
        a.send(bytes(2, 1));
        check(outcome([&] { return b.recv_exact(3); }) == "error:recv_exact: unexpected message length", "recv_exact: a shorter message is an error");
        bool bad = false;
        try {
            b.set_max_message((size_t)Channel::MAX_MESSAGE + 1);
        } catch (const std::invalid_argument&) {
            bad = true;
        }
        check(bad, "set_max_message above the frame limit is refused");
        // exchange with a limit: party 1 receives first and refuses before it sends anything.
        a.send(bytes(9, 2));
        const uint64_t sent = b.stats().msgs_sent;
        check(outcome([&] { return b.exchange(bytes(1, 0), 8, "reply too long"); }) == "abort:reply too long" && b.stats().msgs_sent == sent,
              "exchange(mine, max_reply, error): party 1 refuses an over-long message without answering");
        a.send(bytes(8, 2));
        check(outcome([&] { return b.exchange(bytes(1, 0), 8); }) == "ok:8" && b.stats().rounds == 1, "exchange within the limit");
    }
    // TCP: the 4-byte length field is checked before the payload is read and before a buffer is allocated.
    try {
        TcpListener listener("127.0.0.1", 0);
        const uint16_t port = listener.port();
        // (a) a header that announces the largest frame; nothing else is ever sent (the peer then
        //     closes, so that a receiver that went on to read the payload would end with "peer
        //     closed" instead of waiting forever).
        {
            const net::socket_t peer = raw_connect(port);
            std::unique_ptr<TcpChannel> c0 = listener.accept(0);
            raw_header(peer, 0x7fffffffu);
            net::close_socket(peer);
            const size_t before = peak_commit_bytes();
            check(outcome([&] { return c0->recv(64, "message too long"); }) == "abort:message too long",
                  "TcpChannel: a header above the caller's limit is refused at once (no payload was sent)");
            const size_t grown = peak_commit_bytes() - before;
            check(grown < ((size_t)1 << 20), "TcpChannel: refusing the frame allocated nothing (" + std::to_string(grown) + " bytes)");
            check(outcome([&] { return c0->recv(); }).find("error:TcpChannel: the stream is unusable") == 0,
                  "TcpChannel: after a refused frame the endpoint refuses to receive");
            bool send_refused = false;
            try {
                c0->send(bytes(1, 0));
            } catch (const std::exception&) {
                send_refused = true;
            }
            check(send_refused && c0->stats().msgs_recv == 0, "TcpChannel: after a refused frame the endpoint refuses to send");
        }
        // (b) the endpoint-wide limit.
        {
            const net::socket_t peer = raw_connect(port);
            std::unique_ptr<TcpChannel> c0 = listener.accept(0);
            raw_header(peer, ((uint32_t)1 << 30) + 1);
            net::close_socket(peer);
            const size_t before = peak_commit_bytes();
            check(outcome([&] { return c0->recv(); }) == "long:1073741825/1073741824",
                  "TcpChannel: a header above the endpoint-wide limit (2^30) is refused at once");
            check(peak_commit_bytes() - before < ((size_t)1 << 20), "TcpChannel: refusing it allocated nothing");
        }
        // (c) a header below the endpoint-wide limit that announces 2^30 - 1 bytes, followed by 100
        //     bytes and a closed connection: the buffer grows as the bytes arrive, so the receiver
        //     never commits the announced gigabyte.
        {
            const net::socket_t peer = raw_connect(port);
            std::unique_ptr<TcpChannel> c0 = listener.accept(0);
            raw_header(peer, ((uint32_t)1 << 30) - 1);
            const bytes some(100, 7);
            net::send_all(peer, some.data(), some.size());
            net::close_socket(peer);
            const size_t before = peak_commit_bytes();
            const std::string what = outcome([&] { return c0->recv(); });
            const size_t grown = peak_commit_bytes() - before;
            check(what.find("error:recv") == 0, "TcpChannel: a frame whose announced payload never arrives ends with an error (" + what + ")");
            check(grown <= 4 * net::RECV_GROW_STEP,
                  "TcpChannel: an announced gigabyte that is not sent commits at most a few growth steps (" + std::to_string(grown >> 20) + " MiB)");
        }
        // (d) frames within the limits pass, with and without a caller's limit; a large frame grows in steps.
        {
            const net::socket_t peer = raw_connect(port);
            std::unique_ptr<TcpChannel> c0 = listener.accept(0);
            const bytes five = {'h', 'e', 'l', 'l', 'o'};
            raw_header(peer, 5);
            net::send_all(peer, five.data(), 5);
            raw_header(peer, 5);
            net::send_all(peer, five.data(), 5);
            Rng rng(77);
            const bytes big = make_msg(rng, 2 * net::RECV_GROW_STEP + 12345);  // three growth steps
            raw_header(peer, (uint32_t)big.size());
            std::thread feeder([&] { net::send_all(peer, big.data(), big.size()); });
            const bytes m1 = c0->recv(5), m2 = c0->recv(), m3 = c0->recv();
            feeder.join();
            check(m1 == five && m2 == five && m3 == big && c0->stats().msgs_recv == 3 && c0->stats().bytes_recv == 10 + big.size(),
                  "TcpChannel: frames within the limits are delivered (5 bytes with a limit, 5 bytes and 32 MiB without)");
            net::close_socket(peer);
        }
    } catch (const std::exception& e) {
        check(false, std::string("bounded receive test raised: ") + e.what());
    }
}

static void test_channels() {
    // Loopback, two threads.
    {
        auto pr = LoopbackChannel::make_pair();
        bool ok0 = false, ok1 = false;
        std::thread t1([&] { run_channel_protocol(*pr.second, ok1); });
        run_channel_protocol(*pr.first, ok0);
        t1.join();
        check(ok0 && ok1, "LoopbackChannel exchange/send/recv contents (two threads)");
        check_stats(*pr.first, *pr.second, "LoopbackChannel");
    }
    // Loopback, lockstep single thread.
    {
        auto pr = LoopbackChannel::make_pair();
        pr.first->send(bytes{1, 2, 3});
        pr.second->send(bytes{});
        check(pr.second->pending() == 1 && pr.first->pending() == 1, "LoopbackChannel pending()");
        bytes a = pr.second->recv(), b = pr.first->recv();
        check(a == bytes({1, 2, 3}) && b.empty(), "LoopbackChannel lockstep single-thread use");
        pr.first.reset();
        bool threw = false;
        try { pr.second->recv(); } catch (const std::exception&) { threw = true; }
        check(threw, "LoopbackChannel recv after peer closed throws");
    }
    // TCP over 127.0.0.1 on an ephemeral port.
    try {
        TcpListener listener("127.0.0.1", 0);
        uint16_t port = listener.port();
        std::unique_ptr<TcpChannel> c1;
        bool ok1 = false;
        std::thread t1([&] {
            c1 = TcpChannel::connect(1, "127.0.0.1", port);
            run_channel_protocol(*c1, ok1);
        });
        std::unique_ptr<TcpChannel> c0 = listener.accept(0);
        bool ok0 = false;
        run_channel_protocol(*c0, ok0);
        t1.join();
        check(ok0 && ok1, "TcpChannel exchange/send/recv contents (127.0.0.1:" + std::to_string(port) + ")");
        check_stats(*c0, *c1, "TcpChannel");
        c1.reset();
        bool threw = false;
        try { c0->recv(); } catch (const std::exception&) { threw = true; }
        check(threw, "TcpChannel recv after peer closed throws");
    } catch (const std::exception& e) {
        check(false, std::string("TcpChannel test raised: ") + e.what());
    }
}

// ------------------------------------------------------------ network emulation (netem.hpp)
// A measured time may exceed the emulated one by at most this much (a busy machine delays threads);
// it may never be below it (the emulation waits on the clock until the emulated time has passed).
static const double NETEM_SLACK_S = 0.15;

static void check_emulated_time(double got, double emulated, const std::string& what) {
    char buf[64];
    std::snprintf(buf, sizeof buf, ": %.1f ms, emulated %.1f ms", got * 1e3, emulated * 1e3);
    check(got >= emulated - 1e-6 && got <= emulated + NETEM_SLACK_S, what + buf);
}

// Two connected sockets on 127.0.0.1 (an ephemeral port): {listening side, connecting side}.
static std::pair<net::socket_t, net::socket_t> tcp_socket_pair() {
    net::ensure_init();
    const net::socket_t ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls == net::invalid_socket) throw std::runtime_error("socket failed");
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = 0;
    addr.sin_addr.s_addr = htonl(0x7f000001u);
#if defined(_WIN32)
    int len = (int)sizeof addr;
#else
    socklen_t len = (socklen_t)sizeof addr;
#endif
    if (::bind(ls, (const struct sockaddr*)&addr, (int)sizeof addr) != 0 || ::listen(ls, 1) != 0 ||
        ::getsockname(ls, (struct sockaddr*)&addr, &len) != 0) {
        net::close_socket(ls);
        throw std::runtime_error(net::err_string("listen on 127.0.0.1 failed"));
    }
    const net::socket_t c = raw_connect(ntohs(addr.sin_port));  // completes against the backlog
    const net::socket_t a = ::accept(ls, nullptr, nullptr);
    net::close_socket(ls);
    if (a == net::invalid_socket) {
        net::close_socket(c);
        throw std::runtime_error(net::err_string("accept failed"));
    }
    return std::make_pair(a, c);
}

typedef std::unique_ptr<DelayTcpChannel> EmuPtr;
static std::pair<EmuPtr, EmuPtr> emulated_pair(double delay_ms, double mbps) {
    const std::pair<net::socket_t, net::socket_t> s = tcp_socket_pair();
    EmuPtr c0(new DelayTcpChannel(0, s.first, delay_ms, mbps));
    EmuPtr c1(new DelayTcpChannel(1, s.second, delay_ms, mbps));
    return std::make_pair(std::move(c0), std::move(c1));
}

static long long ns_between(netem::clock::time_point a, netem::clock::time_point b) {
    return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

static void test_token_bucket() {
    typedef netem::clock clk;
    const clk::time_point t0 = clk::now();
    const std::chrono::microseconds us(1);
    auto close_to = [](clk::time_point a, clk::time_point b) { return std::llabs(ns_between(a, b)) <= 1; };
    check(netem::mbps_to_bytes_per_second(100) == 12.5e6 && netem::mbps_to_bytes_per_second(1000) == 125e6,
          "netem: 100 Mbit/s = 12.5e6 bytes/s and 1 Gbit/s = 125e6 bytes/s");
    netem::TokenBucket b(8.0, 1000.0, t0);  // 10^6 bytes per second, depth 1,000 bytes, full
    check(b.active() && b.rate_bytes_per_second() == 1e6 && b.depth_bytes() == 1000.0, "TokenBucket: rate and depth");
    const netem::TokenBucket::Admission a1 = b.admit(t0, 600);
    check(a1.done == t0 && a1.tokens == 1000.0, "TokenBucket: a frame within the full bucket is admitted at once");
    const netem::TokenBucket::Admission a2 = b.admit(t0, 600);
    check(a2.start == t0 && a2.tokens == 400.0 && close_to(a2.done, t0 + 200 * us),
          "TokenBucket: 600 bytes with 400 tokens left are admitted after 200 us");
    check(b.admitted_at(a2, 400) == t0 && close_to(b.admitted_at(a2, 500), t0 + 100 * us) && b.admitted_at(a2, 600) == a2.done,
          "TokenBucket: admission times of prefixes (fluid model)");
    const netem::TokenBucket::Admission a3 = b.admit(t0, 100);  // ready before the link is free
    check(a3.start == a2.done && a3.tokens == 0.0 && close_to(a3.done, t0 + 300 * us),
          "TokenBucket: a frame waits for the backlog of the bucket");
    const netem::TokenBucket::Admission a4 = b.admit(t0 + std::chrono::seconds(10), 2500);
    check(a4.tokens == 1000.0 && close_to(a4.done, t0 + std::chrono::seconds(10) + 1500 * us),
          "TokenBucket: tokens never exceed the depth; a frame larger than the depth is admitted at the rate");
    netem::TokenBucket off;
    check(!off.active() && off.admit(t0, 1e9).done == t0 && off.admitted_at(off.admit(t0, 5), 3) == t0,
          "TokenBucket: an inactive bucket admits everything at once");
    int refused = 0;
    try {
        netem::TokenBucket bad(0.0, 1500.0, t0);
    } catch (const std::invalid_argument&) {
        refused++;
    }
    try {
        netem::TokenBucket bad(100.0, 0.5, t0);
    } catch (const std::invalid_argument&) {
        refused++;
    }
    try {
        const std::pair<net::socket_t, net::socket_t> s = tcp_socket_pair();
        net::close_socket(s.second);
        DelayTcpChannel bad(0, s.first, 0.0, -1.0);
    } catch (const std::invalid_argument&) {
        refused++;
    }
    check(refused == 3, "TokenBucket and DelayTcpChannel refuse a rate of 0 or below and a depth below one byte");
}

static void test_network_emulation() {
    typedef netem::clock clk;
    const double R = netem::mbps_to_bytes_per_second(40.0);  // 5 * 10^6 bytes per second
    const double C = DelayTcpChannel::DEFAULT_BUCKET_BYTES;
    const double F = DelayTcpChannel::FRAME_CHARGE;
    try {
        // (1) A flight of 4 messages of 250,000 bytes at 40 Mbit/s, no delay.  The receiver's last
        //     message and the sender's last send end (4 * (250,000 + 4) - C) / R after the first send.
        {
            std::pair<EmuPtr, EmuPtr> pr = emulated_pair(0.0, 40.0);
            Rng rng(501);
            std::vector<bytes> msgs;
            for (int i = 0; i < 4; i++) msgs.push_back(make_msg(rng, 250000));
            clk::time_point start, sender_end, receiver_end;
            bool same = true;
            std::thread rx([&] {
                pr.second->exchange(bytes(1, 2));
                for (int i = 0; i < 4; i++) same = pr.second->recv() == msgs[(size_t)i] && same;
                receiver_end = clk::now();
            });
            pr.first->exchange(bytes(1, 1));  // synchronize
            start = clk::now();
            for (int i = 0; i < 4; i++) pr.first->send(msgs[(size_t)i]);
            sender_end = clk::now();
            rx.join();
            const double emulated = (4 * (250000 + F) - C) / R;
            check_emulated_time(netem::to_seconds(receiver_end - start), emulated, "netem: 4 x 250,000 bytes at 40 Mbit/s arrive");
            check_emulated_time(netem::to_seconds(sender_end - start), emulated, "netem: the sender of 4 x 250,000 bytes at 40 Mbit/s is throttled");
            check(same && pr.second->stats().bytes_recv == 1000000 + 1 && pr.second->stats().msgs_recv == 5 &&
                      pr.first->stats().bytes_sent == 1000000 + 1,
                  "netem: the messages arrive unchanged; the counters count payload bytes only");
            check(pr.first->bandwidth_wait_s() >= 0.5 * emulated && pr.second->bandwidth_wait_s() < 0.01 && pr.first->stamped(),
                  "netem: the sender's sends were held back, the receiver's were not");
        }
        // (2) One message of 500,000 bytes at 40 Mbit/s with a one-way delay of 30 ms: the receiver
        //     gets it (500,004 - C) / R + 30 ms after the send started; the sender does not wait for
        //     the delay.
        {
            std::pair<EmuPtr, EmuPtr> pr = emulated_pair(30.0, 40.0);
            Rng rng(502);
            const bytes msg = make_msg(rng, 500000);
            clk::time_point start, sender_end, receiver_end;
            bool same = false;
            std::thread rx([&] {
                pr.second->exchange(bytes(1, 2));
                same = pr.second->recv() == msg;
                receiver_end = clk::now();
            });
            pr.first->exchange(bytes(1, 1));
            start = clk::now();
            pr.first->send(msg);
            sender_end = clk::now();
            rx.join();
            const double transfer = (500000 + F - C) / R;
            check_emulated_time(netem::to_seconds(receiver_end - start), transfer + 0.030,
                                "netem: 500,000 bytes at 40 Mbit/s with 30 ms delay arrive after transfer + delay");
            check_emulated_time(netem::to_seconds(sender_end - start), transfer,
                                "netem: the sender is held back for the transfer, not for the delay");
            check(same, "netem: the delayed message arrives unchanged");
        }
        // (3) Exchanges at 40 Mbit/s with a delay of 20 ms; party 0 sends m0 bytes, party 1 sends m1.
        //     Both halves are admitted from the time their sender was ready (party 1: when it entered
        //     the receive), so the two directions transfer at the same time.  Because party 1
        //     receives before it sends (Channel::exchange), party 1's half can reach party 0 only
        //     after party 1 has received: with transfer times T0, T1 (from equal starts), party 1 is
        //     done after max(T0 + D, T1) and party 0 after max(T0, T1) + D.  A full-duplex link would
        //     give party 0 max(T0, T1 + D), i.e. up to D less when T0 > T1; transfers that took
        //     turns would give T0 + T1 + D.
        for (int big0 = 1; big0 >= 0; big0--) {
            const size_t n0 = big0 ? 1000000 : 500000, n1 = big0 ? 500000 : 1000000;
            std::pair<EmuPtr, EmuPtr> pr = emulated_pair(20.0, 40.0);
            Rng rng(503 + (uint64_t)big0);
            const bytes m0 = make_msg(rng, n0), m1 = make_msg(rng, n1);
            clk::time_point start0, start1, end0, end1;
            bool same0 = false, same1 = false;
            std::thread p1([&] {
                pr.second->exchange(bytes(1, 2));
                start1 = clk::now();
                same1 = pr.second->exchange(m1) == m0;
                end1 = clk::now();
            });
            pr.first->exchange(bytes(1, 1));
            start0 = clk::now();
            same0 = pr.first->exchange(m0) == m1;
            end0 = clk::now();
            p1.join();
            const double D = 0.020, T0 = ((double)n0 + F - C) / R, T1 = ((double)n1 + F - C) / R;
            const double lag1 = netem::to_seconds(start1 - start0);  // party 1 starts this much later (or earlier)
            const std::string tag = "netem: exchange of " + std::to_string(n0) + " (party 0) and " + std::to_string(n1) +
                                    " bytes (party 1) at 40 Mbit/s, 20 ms: ";
            check_emulated_time(netem::to_seconds(end1 - start0), std::max(T0 + D, lag1 + T1), tag + "party 1 is done after max(T0 + D, T1)");
            check_emulated_time(netem::to_seconds(end0 - start0), std::max(T0, lag1 + T1) + D, tag + "party 0 is done after max(T0, T1) + D");
            const clk::time_point first = start0 < start1 ? start0 : start1;
            const clk::time_point last = end0 > end1 ? end0 : end1;
            check(netem::to_seconds(last - first) < T0 + T1 + D - 0.010, tag + "the two directions do not take turns");
            check(same0 && same1 && pr.first->stats().rounds == 2 && pr.second->stats().rounds == 2,
                  tag + "the messages arrive unchanged; two rounds at both parties");
        }
        // (4) 200 messages of 4,996 bytes (5,000 bytes each with the length) at 40 Mbit/s: the rate
        //     also holds for a flight of small messages.
        {
            std::pair<EmuPtr, EmuPtr> pr = emulated_pair(0.0, 40.0);
            clk::time_point start, receiver_end;
            uint64_t got = 0;
            std::thread rx([&] {
                pr.second->exchange(bytes(1, 2));
                for (int i = 0; i < 200; i++) got += pr.second->recv().size();
                receiver_end = clk::now();
            });
            pr.first->exchange(bytes(1, 1));
            start = clk::now();
            for (int i = 0; i < 200; i++) pr.first->send(bytes(4996, (uint8_t)i));
            rx.join();
            check_emulated_time(netem::to_seconds(receiver_end - start), (200 * 5000.0 - C) / R,
                                "netem: 200 messages of 4,996 bytes at 40 Mbit/s");
            check(got == 200 * 4996, "netem: all 200 messages arrived");
        }
        // (5) Neither delay nor limit: the endpoint speaks the plain framing of TcpChannel (the
        //     wire format of the mixed-language tests) and is not slowed down.
        {
            const std::pair<net::socket_t, net::socket_t> s = tcp_socket_pair();
            EmuPtr c0(new DelayTcpChannel(0, s.first, 0.0, 0.0));
            TcpChannel c1(1, s.second);
            bool ok0 = false, ok1 = false;
            const clk::time_point start = clk::now();
            std::thread t1([&] { run_channel_protocol(c1, ok1); });
            run_channel_protocol(*c0, ok0);
            t1.join();
            check(!c0->stamped() && ok0 && ok1, "netem off: DelayTcpChannel and TcpChannel understand each other (plain framing)");
            check_stats(*c0, c1, "DelayTcpChannel (no emulation) against TcpChannel");
            check(c0->emulated_wait_s() == 0.0 && c0->bandwidth_wait_s() == 0.0 && netem::to_seconds(clk::now() - start) < 2.0,
                  "netem off: no emulated waiting");
        }
        // (6) Delay 1 ms and 1 Gbit/s on both endpoints: the channel script delivers the same
        //     contents and the same counters as without emulation.
        {
            std::pair<EmuPtr, EmuPtr> pr = emulated_pair(1.0, 1000.0);
            bool ok0 = false, ok1 = false;
            std::thread t1([&] { run_channel_protocol(*pr.second, ok1); });
            run_channel_protocol(*pr.first, ok0);
            t1.join();
            check(ok0 && ok1 && pr.first->stamped(), "netem on (1 ms, 1 Gbit/s): the channel script's messages arrive unchanged");
            check_stats(*pr.first, *pr.second, "DelayTcpChannel (1 ms, 1 Gbit/s)");
        }
    } catch (const std::exception& e) {
        check(false, std::string("network emulation test raised: ") + e.what());
    }
}

int main() {
    std::printf("couplet C++ selftest\n");
#if defined(_MSC_VER)
    std::printf("compiler: MSVC %d\n", _MSC_VER);
#elif defined(__clang__)
    std::printf("compiler: clang %s\n", __clang_version__);
#elif defined(__GNUC__)
    std::printf("compiler: gcc %d.%d\n", __GNUC__, __GNUC_MINOR__);
#endif
    std::printf("avx2=%d pclmul=%d native_int128=%d\n", (int)COUPLET_HAVE_AVX2, (int)COUPLET_HAVE_PCLMUL,
#if defined(__SIZEOF_INT128__)
                1
#else
                0
#endif
    );
    test_ascon();
    test_u128();
    test_gf128();
    test_sha256();
    test_x25519();
    test_channels();
    test_bounded_receive();
    test_token_bucket();
    test_network_emulation();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
