// couplet/rng.hpp -- random byte sources.
//
//   RandomSource  abstract interface: fill(out, n) plus small typed helpers
//                 (bytes, 64-bit words, GF(2^128) elements, seeds, bits).
//   SystemRng     the operating system's CSPRNG: BCryptGenRandom with the
//                 system-preferred generator on Windows (links bcrypt.lib),
//                 getrandom(2) on Linux with /dev/urandom as fallback,
//                 /dev/urandom elsewhere.  Fails closed: any error throws,
//                 there is no fallback to a weaker generator.
//   CtrPrgRng     the byte view of hashes::ctr_prg(seed, dom::PRG_RNG): a
//                 deterministic, seedable stream.  Constructed from a fixed
//                 seed it gives reproducible tests; CtrPrgRng::from_system()
//                 keys it with 16 bytes of SystemRng output, which gives a
//                 fast generator that is secure in the random permutation
//                 model as long as the seed stays secret (no forward
//                 secrecy, no reseeding).
//
// Sources are not thread-safe; use one per thread.  The templates in
// hashes.hpp and ed25519.hpp accept any of these (they only call fill()).
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gf128.hpp"
#include "hashes.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#if defined(_MSC_VER)
#pragma comment(lib, "bcrypt.lib")
#endif
#else
#include <cerrno>
#include <cstdio>
#if defined(__linux__)
#include <sys/random.h>
#endif
#endif

namespace couplet {

class RandomSource {
public:
    virtual ~RandomSource() {}

    // Write n random bytes to out.
    virtual void fill(uint8_t* out, size_t n) = 0;

    bytes get(size_t n) {
        bytes b(n);
        if (n) fill(b.data(), n);
        return b;
    }
    template <size_t N>
    std::array<uint8_t, N> array() {
        std::array<uint8_t, N> a;
        fill(a.data(), N);
        return a;
    }
    hashes::seed16 seed() { return array<16>(); }
    uint64_t u64() {  // 8 bytes, little-endian
        uint8_t b[8];
        fill(b, 8);
        return ascon::load64le(b);
    }
    gf128 gf() {  // 16 bytes, little-endian (gf128::from_bytes)
        uint8_t b[16];
        fill(b, 16);
        return gf128::from_bytes(b);
    }
    void fill_gf(gf128* out, size_t n) {
        for (size_t i = 0; i < n; i++) out[i] = gf();
    }
    // n bits, one per output byte: bit i is bit (i mod 8) of byte floor(i / 8)
    // of the next ceil(n / 8) bytes.
    void fill_bits(uint8_t* bits, size_t n) {
        uint8_t buf[256];
        size_t done = 0;
        while (done < n) {
            const size_t take = (n - done < 8 * sizeof buf) ? (n - done) : 8 * sizeof buf;
            fill(buf, (take + 7) / 8);
            for (size_t i = 0; i < take; i++) bits[done + i] = (uint8_t)((buf[i / 8] >> (i % 8)) & 1);
            done += take;
        }
    }
};

// The operating system's cryptographic generator.
class SystemRng : public RandomSource {
public:
    void fill(uint8_t* out, size_t n) override {
#if defined(_WIN32)
        while (n) {
            const ULONG chunk = (n > 0x40000000u) ? (ULONG)0x40000000u : (ULONG)n;
            const NTSTATUS st = BCryptGenRandom(nullptr, (PUCHAR)out, chunk, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
            if (st < 0) throw std::runtime_error("SystemRng: BCryptGenRandom failed");
            out += chunk;
            n -= chunk;
        }
#else
#if defined(__linux__)
        while (n) {
            const ssize_t r = ::getrandom(out, n, 0);
            if (r < 0) {
                if (errno == EINTR) continue;
                if (errno == ENOSYS) break;  // kernel without getrandom: use /dev/urandom
                throw std::runtime_error("SystemRng: getrandom failed");
            }
            out += (size_t)r;
            n -= (size_t)r;
        }
        if (n == 0) return;
#endif
        std::FILE* f = std::fopen("/dev/urandom", "rb");
        if (!f) throw std::runtime_error("SystemRng: cannot open /dev/urandom");
        std::setvbuf(f, nullptr, _IONBF, 0);
        while (n) {
            const size_t r = std::fread(out, 1, n, f);
            if (r == 0) {
                std::fclose(f);
                throw std::runtime_error("SystemRng: cannot read /dev/urandom");
            }
            out += r;
            n -= r;
        }
        std::fclose(f);
#endif
    }
};

// Deterministic stream: the bytes of hashes::ctr_prg(seed, domain).
class CtrPrgRng : public RandomSource {
public:
    explicit CtrPrgRng(const hashes::seed16& seed, uint64_t domain = hashes::dom::PRG_RNG)
        : stream_(seed, domain) {}
    // Convenience for tests: the seed is le64(seed) followed by eight zero bytes.
    explicit CtrPrgRng(uint64_t seed, uint64_t domain = hashes::dom::PRG_RNG)
        : stream_(seed_from_u64(seed), domain) {}

    // A fast generator keyed with fresh operating-system randomness.
    static CtrPrgRng from_system() {
        SystemRng sys;
        return CtrPrgRng(sys.seed());
    }

    void fill(uint8_t* out, size_t n) override { stream_.fill(out, n); }

    static hashes::seed16 seed_from_u64(uint64_t seed) {
        hashes::seed16 s{};
        ascon::store64le(s.data(), seed);
        return s;
    }

private:
    hashes::CtrPrgStream stream_;
};

}  // namespace couplet
