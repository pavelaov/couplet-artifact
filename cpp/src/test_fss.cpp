// test_fss.cpp -- correctness tests of the two-plane DPF and DCF
// (couplet/fss.hpp, DESIGN.md section 3).  Counterpart of
// python/tests/test_fss_exhaustive.py.
//
// Exhaustive part: for n in 1..8, k in {32, 64}, s = 64, EVERY alpha in
// [2^n], beta in {0, 1, 2^k - 1, random} and a MAC key Delta = Delta_0 +
// Delta_1 formed from two independent s-bit shares WITHOUT reduction (about
// half of the keys use Delta >= 2^s):
//   * EvalAll of both parties and both planes reconstructs, at ALL x,
//       y0 + y1     == beta                 * [x == alpha] (DPF) / [x < alpha] (DCF)
//       tau0 + tau1 == (Delta*beta mod 2^w) * [x == alpha] (DPF) / [x < alpha] (DCF)
//     mod 2^w, w = k + s, with all outputs in [0, 2^w);
//   * point Eval equals EvalAll at ALL x, both parties, both planes, and
//     reconstructs the same values;
//   * keys survive to_bytes / from_bytes and have the documented size;
//   * the release check of DESIGN.md 2.3 passes on the honest shares of the
//     whole domain (zeta_0 + zeta_1 == 0 with zeta_b = sum chi_j tau_{j,b} -
//     Delta_b * sum chi_j v_j).
// Further parts: single-opening release checks and a negative control (keys
// generated with Delta reduced mod 2^s are rejected by the same check);
// explicit roots, the draw order of the generator overload, extreme MAC keys
// and argument validation; rejection of malformed encodings; other (k, s)
// pairs; domains larger than one EvalAll subtree (n up to 20) with several
// thread counts; point evaluation for n up to 64.
//
// All randomness is seeded.  The exit status is non-zero on any failure and
// the number of individual checks is printed at the end.
// Usage: test_fss [--quick]     (--quick limits the exhaustive part to n <= 6)
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "couplet/fss.hpp"
#include "couplet/u128.hpp"

using namespace couplet;

// ------------------------------------------------------------ harness
struct Counters {
    uint64_t evalall_range = 0, evalall_value = 0, evalall_tag = 0, eval_vs_evalall = 0, eval_sum = 0,
             serialisation = 0, release_batch = 0, release_single = 0, release_negative = 0, roots_ranges = 0,
             decode_rejects = 0, other_params = 0, large_domain = 0, wide_index = 0, misc = 0, workers = 0;
    uint64_t keys = 0, keys_big_delta = 0;
    uint64_t total() const {
        return evalall_range + evalall_value + evalall_tag + eval_vs_evalall + eval_sum + serialisation +
               release_batch + release_single + release_negative + roots_ranges + decode_rejects + other_params +
               large_domain + wide_index + misc + workers;
    }
};
static Counters C;
static uint64_t g_fail = 0;

static void report_failure(const char* fmt, ...) {
    g_fail++;
    if (g_fail > 20) return;
    std::va_list ap;
    va_start(ap, fmt);
    std::printf("[FAIL] ");
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
}
// One counted check; the message is only formatted on failure.
#define CHECK(cond, counter, ...)            \
    do {                                     \
        (counter)++;                         \
        if (!(cond)) report_failure(__VA_ARGS__); \
    } while (0)

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// xorshift64*: deterministic test generator (not a cryptographic generator).
struct TestRng {
    uint64_t s;
    explicit TestRng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    void fill(uint8_t* out, size_t n) {
        while (n) {
            const uint64_t v = next();
            const size_t take = n < 8 ? n : 8;
            for (size_t i = 0; i < take; i++) out[i] = (uint8_t)(v >> (8 * i));
            out += take;
            n -= take;
        }
    }
    uint64_t bits64(unsigned bits) { return bits >= 64 ? next() : (next() & (((uint64_t)1 << bits) - 1)); }
    u128 bits128(unsigned bits) {  // uniform in [0, 2^bits), bits <= 128
        const uint64_t lo = next(), hi = next();
        return u128_from(lo, hi) & u128_mask(bits);
    }
    fss::seed_t seed() {
        fss::seed_t r;
        fill(r.data(), 16);
        return r;
    }
};

template <class Fn>
static bool throws_invalid(Fn fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// ------------------------------------------------------------ the two families
struct Dpf {
    typedef fss::DpfKey Key;
    static const char* name() { return "DPF"; }
    static bool is_dcf() { return false; }
    static std::pair<Key, Key> gen(uint64_t a, uint64_t b, u128 d, unsigned n, unsigned k, unsigned s, TestRng& rng) {
        return fss::dpf_gen(a, b, d, n, k, s, rng);
    }
    static std::pair<Key, Key> gen_roots(uint64_t a, uint64_t b, u128 d, unsigned n, unsigned k, unsigned s,
                                         const fss::seed_t& r0, const fss::seed_t& r1) {
        return fss::dpf_gen(a, b, d, n, k, s, r0, r1);
    }
    static fss::Share eval(unsigned b, const Key& key, uint64_t x) { return fss::dpf_eval(b, key, x); }
    static void eval_all(const Key& key, std::vector<u128>& y, std::vector<u128>& t, unsigned threads = 1) {
        fss::dpf_eval_all(key, y, t, threads);
    }
    static void eval_all_ptr(const Key& key, u128* y, u128* t, unsigned threads) { fss::dpf_eval_all(key, y, t, threads); }
    static bool ind(uint64_t x, uint64_t alpha) { return x == alpha; }
    static size_t key_size(unsigned n, unsigned k, unsigned s) { return fss::dpf_key_size(n, k, s); }
};
struct Dcf {
    typedef fss::DcfKey Key;
    static const char* name() { return "DCF"; }
    static bool is_dcf() { return true; }
    static std::pair<Key, Key> gen(uint64_t a, uint64_t b, u128 d, unsigned n, unsigned k, unsigned s, TestRng& rng) {
        return fss::dcf_gen(a, b, d, n, k, s, rng);
    }
    static std::pair<Key, Key> gen_roots(uint64_t a, uint64_t b, u128 d, unsigned n, unsigned k, unsigned s,
                                         const fss::seed_t& r0, const fss::seed_t& r1) {
        return fss::dcf_gen(a, b, d, n, k, s, r0, r1);
    }
    static fss::Share eval(unsigned b, const Key& key, uint64_t x) { return fss::dcf_eval(b, key, x); }
    static void eval_all(const Key& key, std::vector<u128>& y, std::vector<u128>& t, unsigned threads = 1) {
        fss::dcf_eval_all(key, y, t, threads);
    }
    static void eval_all_ptr(const Key& key, u128* y, u128* t, unsigned threads) { fss::dcf_eval_all(key, y, t, threads); }
    static bool ind(uint64_t x, uint64_t alpha) { return x < alpha; }
    static size_t key_size(unsigned n, unsigned k, unsigned s) { return fss::dcf_key_size(n, k, s); }
};

// Encoded size from the format description, independent of fss.hpp.
static size_t expected_size(bool dcf, unsigned n, unsigned k, unsigned s) {
    const size_t wb = (k + s + 7) / 8;
    return 4 + 5 + 16 + (size_t)n * (16 + 1 + (dcf ? 2 * wb : 0)) + 2 * wb;
}

static uint64_t max_beta(unsigned k) { return k >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << k) - 1); }

// ArithCheck of DESIGN.md 2.3 on opened values (one modulus 2^w):
// y = sum chi_j v_j; zeta_b = sum chi_j tau_{j,b} - Delta_b y; accept iff
// zeta_0 + zeta_1 == 0.
static bool release_check(const std::vector<u128>& chis, const std::vector<u128>& values,
                          const std::vector<u128>& tags0, const std::vector<u128>& tags1, u128 d0, u128 d1, u128 mask) {
    u128 y = u128(0), a0 = u128(0), a1 = u128(0);
    for (size_t j = 0; j < chis.size(); j++) {
        y = y + chis[j] * values[j];
        a0 = a0 + chis[j] * tags0[j];
        a1 = a1 + chis[j] * tags1[j];
    }
    y = y & mask;
    const u128 zeta0 = (a0 - d0 * y) & mask, zeta1 = (a1 - d1 * y) & mask;
    return ((zeta0 + zeta1) & mask) == u128(0);
}

// Serialisation round trip and size of one key (3 checks, as in the reference).
template <class F>
static void check_serialisation(const typename F::Key& key, unsigned n, unsigned k, unsigned s, const char* what) {
    const std::vector<uint8_t> blob = key.to_bytes();
    const typename F::Key back = F::Key::from_bytes(blob);
    CHECK(back == key, C.serialisation, "%s: from_bytes(to_bytes(key)) != key", what);
    CHECK(back.to_bytes() == blob && F::Key::from_bytes(blob.data(), blob.size()) == key, C.serialisation,
          "%s: re-encoding differs", what);
    CHECK(blob.size() == expected_size(F::is_dcf(), n, k, s) && key.size_bytes() == blob.size() &&
              F::key_size(n, k, s) == blob.size() &&
              fss::encoded_key_family(blob) == (F::is_dcf() ? fss::FAMILY_DCF : fss::FAMILY_DPF),
          C.serialisation, "%s: unexpected key size %zu", what, blob.size());
}

// ------------------------------------------------------------ exhaustive part
template <class F>
static void exhaustive_case(unsigned n, unsigned k) {
    const unsigned s = 64, w = k + s;
    const u128 mask = u128_mask(w);
    const size_t size = (size_t)1 << n;
    TestRng rng(0x5EED0000ull + 1000 * (uint64_t)n + 10 * (uint64_t)k + (F::is_dcf() ? 1 : 0));
    const uint64_t betas[4] = {0, 1, max_beta(k), rng.bits64(k)};
    const Counters before = C;
    const double t_start = now_s();
    std::vector<u128> y0, t0, y1, t1, chis(size), vals(size);
    char what[160];
    for (uint64_t alpha = 0; alpha < size; alpha++) {
        for (uint64_t beta : betas) {
            const uint64_t d0 = rng.next(), d1 = rng.next();
            const u128 delta = fss::make_delta(d0, d1);  // 65-bit integer, never reduced
            const bool big = (delta >> s) != u128(0);
            const u128 beta_t = (delta * u128(beta)) & mask;
            const std::pair<typename F::Key, typename F::Key> keys = F::gen(alpha, beta, delta, n, k, s, rng);
            const typename F::Key& k0 = keys.first;
            const typename F::Key& k1 = keys.second;
            C.keys += 2;
            if (big) C.keys_big_delta += 2;
            std::snprintf(what, sizeof what, "%s n=%u k=%u alpha=%llu beta=%llx delta=%s", F::name(), n, k,
                          (unsigned long long)alpha, (unsigned long long)beta, u128_to_hex(delta).c_str());

            // EvalAll on both planes, for ALL x.
            F::eval_all(k0, y0, t0);
            F::eval_all(k1, y1, t1);
            const std::vector<u128>* arrs[4] = {&y0, &t0, &y1, &t1};
            for (const std::vector<u128>* a : arrs) {
                bool ok = a->size() == size;
                for (size_t x = 0; ok && x < size; x++) ok = ((*a)[x] & mask) == (*a)[x];
                CHECK(ok, C.evalall_range, "%s: EvalAll vector has wrong length or out-of-range entries", what);
            }
            if (y0.size() != size || t0.size() != size || y1.size() != size || t1.size() != size) continue;
            for (size_t x = 0; x < size; x++) {
                const bool ind = F::ind(x, alpha);
                vals[x] = (y0[x] + y1[x]) & mask;
                CHECK(vals[x] == (ind ? u128(beta) : u128(0)), C.evalall_value, "%s: EvalAll value plane at x=%zu", what, x);
                CHECK(((t0[x] + t1[x]) & mask) == (ind ? beta_t : u128(0)), C.evalall_tag,
                      "%s: EvalAll tag plane at x=%zu", what, x);
            }

            // Release check over the whole domain as one batch, fresh coins.
            for (size_t x = 0; x < size; x++) chis[x] = u128(rng.next());
            CHECK(release_check(chis, vals, t0, t1, u128(d0), u128(d1), mask), C.release_batch,
                  "%s: batched release check failed", what);

            // Point Eval at ALL x: agreement with EvalAll and reconstruction.
            for (size_t x = 0; x < size; x++) {
                const fss::Share a0 = F::eval(0, k0, x), a1 = F::eval(1, k1, x);
                CHECK(a0.y == y0[x], C.eval_vs_evalall, "%s: Eval(0, x=%zu).y != EvalAll", what, x);
                CHECK(a0.tau == t0[x], C.eval_vs_evalall, "%s: Eval(0, x=%zu).tau != EvalAll", what, x);
                CHECK(a1.y == y1[x], C.eval_vs_evalall, "%s: Eval(1, x=%zu).y != EvalAll", what, x);
                CHECK(a1.tau == t1[x], C.eval_vs_evalall, "%s: Eval(1, x=%zu).tau != EvalAll", what, x);
                const bool ind = F::ind(x, alpha);
                CHECK(((a0.y + a1.y) & mask) == (ind ? u128(beta) : u128(0)), C.eval_sum, "%s: Eval value sum at x=%zu", what, x);
                CHECK(((a0.tau + a1.tau) & mask) == (ind ? beta_t : u128(0)), C.eval_sum, "%s: Eval tag sum at x=%zu", what, x);
            }

            // Serialisation round trip and size.
            check_serialisation<F>(k0, n, k, s, what);
            check_serialisation<F>(k1, n, k, s, what);
        }
    }
    std::printf("  %s n=%u k=%u: %9llu checks, %4llu keys (%4llu with Delta >= 2^s), %6.2f s\n", F::name(), n, k,
                (unsigned long long)(C.total() - before.total()), (unsigned long long)(C.keys - before.keys),
                (unsigned long long)(C.keys_big_delta - before.keys_big_delta), now_s() - t_start);
    std::fflush(stdout);
}

// ------------------------------------------------------------ release check
template <class F>
static void release_tests(unsigned k, TestRng& rng) {
    const unsigned s = 64, w = k + s;
    const u128 mask = u128_mask(w);
    // Honest single openings.
    for (int trial = 0; trial < 200; trial++) {
        const unsigned n = 1 + (unsigned)(rng.next() % 12);
        const uint64_t alpha = rng.bits64(n);
        const uint64_t choices[4] = {0, 1, max_beta(k), rng.bits64(k)};
        const uint64_t beta = choices[rng.next() & 3];
        const uint64_t d0 = rng.next(), d1 = rng.next();
        const u128 delta = fss::make_delta(d0, d1);
        const auto keys = F::gen(alpha, beta, delta, n, k, s, rng);
        const uint64_t nmask = ((uint64_t)1 << n) - 1;
        const uint64_t pts[4] = {alpha, (alpha - 1) & nmask, rng.bits64(n), rng.bits64(n)};
        for (uint64_t x : pts) {
            const fss::Share a0 = F::eval(0, keys.first, x), a1 = F::eval(1, keys.second, x);
            const u128 v = (a0.y + a1.y) & mask;
            const u128 chi = u128(rng.next());
            const u128 y = (chi * v) & mask;
            const u128 zeta0 = (chi * a0.tau - u128(d0) * y) & mask, zeta1 = (chi * a1.tau - u128(d1) * y) & mask;
            CHECK(((zeta0 + zeta1) & mask) == u128(0), C.release_single,
                  "release check failed: %s n=%u k=%u x=%llu alpha=%llu", F::name(), n, k, (unsigned long long)x,
                  (unsigned long long)alpha);
            CHECK(release_check({chi}, {v}, {a0.tau}, {a1.tau}, u128(d0), u128(d1), mask), C.release_single,
                  "release_check helper disagrees with the direct formula");
        }
    }
    // Negative control: a MAC key reduced mod 2^s must be caught.  The opened
    // value is odd and the coin is odd, so chi * v is non-zero mod 2^k.
    for (int trial = 0; trial < 20; trial++) {
        const unsigned n = 2 + (unsigned)(rng.next() % 7);
        uint64_t alpha = rng.bits64(n);
        if (alpha == 0) alpha = 1;  // so that the DCF has a point x < alpha
        const uint64_t beta = rng.bits64(k) | 1;
        uint64_t d0, d1;
        u128 delta;
        do {
            d0 = rng.next();
            d1 = rng.next();
            delta = fss::make_delta(d0, d1);
        } while ((delta >> s) == u128(0));
        const u128 reduced = delta & u128_mask(s);
        const auto keys = F::gen(alpha, beta, reduced, n, k, s, rng);
        const uint64_t x = F::is_dcf() ? alpha - 1 : alpha;
        const fss::Share a0 = F::eval(0, keys.first, x), a1 = F::eval(1, keys.second, x);
        const u128 v = (a0.y + a1.y) & mask;
        const u128 chi = u128(rng.next() | 1);
        CHECK(v == u128(beta), C.release_negative, "negative control: unexpected opened value (%s k=%u)", F::name(), k);
        CHECK(!release_check({chi}, {v}, {a0.tau}, {a1.tau}, u128(d0), u128(d1), mask), C.release_negative,
              "negative control: reduced Delta passed the release check (%s k=%u)", F::name(), k);
    }
}

// ------------------------------------------------------------ roots, ranges
template <class F>
static void roots_and_ranges(unsigned k, TestRng& rng) {
    const unsigned s = 64, w = k + s;
    const u128 mask = u128_mask(w);
    for (int trial = 0; trial < 20; trial++) {
        const unsigned n = 1 + (unsigned)(rng.next() % 10);
        const uint64_t alpha = rng.bits64(n), beta = rng.bits64(k);
        const u128 delta = fss::make_delta(rng.next(), rng.next());
        // The generator overload draws root 0, then root 1, and nothing else.
        TestRng copy = rng;
        const fss::seed_t e0 = copy.seed(), e1 = copy.seed();
        const auto keys = F::gen(alpha, beta, delta, n, k, s, rng);
        CHECK(keys.first.seed == e0 && keys.second.seed == e1 && rng.s == copy.s, C.roots_ranges,
              "%s: generator overload does not draw root0 then root1", F::name());
        const auto again = F::gen_roots(alpha, beta, delta, n, k, s, keys.first.seed, keys.second.seed);
        CHECK(again.first == keys.first && again.second == keys.second &&
                  again.first.to_bytes() == keys.first.to_bytes() && again.second.to_bytes() == keys.second.to_bytes(),
              C.roots_ranges, "%s: explicit roots do not reproduce the keys", F::name());
        // The two keys share every public correction word.
        typename F::Key swapped = keys.second;
        swapped.b = 0;
        swapped.seed = keys.first.seed;
        CHECK(swapped == keys.first && keys.first.b == 0 && keys.second.b == 1, C.roots_ranges,
              "%s: the two keys differ in more than (b, root)", F::name());
    }
    // Extreme MAC keys: 0, 2^s - 1, 2^s, 2^(s+1) - 1.
    const u128 extremes[4] = {u128(0), u128_mask(s), u128(1) << s, u128_mask(s + 1)};
    for (const u128& delta : extremes) {
        const unsigned n = 5;
        uint64_t alpha = rng.bits64(n);
        if (alpha == 0) alpha = 1;
        const uint64_t beta = max_beta(k);
        const u128 beta_t = (delta * u128(beta)) & mask;
        const auto keys = F::gen(alpha, beta, delta, n, k, s, rng);
        for (uint64_t x = 0; x < ((uint64_t)1 << n); x++) {
            const fss::Share a0 = F::eval(0, keys.first, x), a1 = F::eval(1, keys.second, x);
            const bool ind = F::ind(x, alpha);
            CHECK(((a0.y + a1.y) & mask) == (ind ? u128(beta) : u128(0)), C.roots_ranges,
                  "%s: wrong value for extreme Delta %s", F::name(), u128_to_hex(delta).c_str());
            CHECK(((a0.tau + a1.tau) & mask) == (ind ? beta_t : u128(0)), C.roots_ranges,
                  "%s: wrong tag for extreme Delta %s", F::name(), u128_to_hex(delta).c_str());
        }
    }
    // Argument validation.
    const fss::seed_t z{};
    CHECK(throws_invalid([&] { F::gen(0, 1, u128(1) << (s + 1), 3, k, s, rng); }), C.roots_ranges, "%s: Delta = 2^(s+1) accepted", F::name());
    CHECK(throws_invalid([&] { F::gen(0, 1, ~u128(0), 3, k, s, rng); }), C.roots_ranges, "%s: Delta = 2^128 - 1 accepted", F::name());
    CHECK(throws_invalid([&] { F::gen(8, 1, u128(1), 3, k, s, rng); }), C.roots_ranges, "%s: alpha = 2^n accepted", F::name());
    if (k < 64)
        CHECK(throws_invalid([&] { F::gen(0, (uint64_t)1 << k, u128(1), 3, k, s, rng); }), C.roots_ranges, "%s: beta = 2^k accepted", F::name());
    CHECK(throws_invalid([&] { F::gen(0, 1, u128(1), 0, k, s, rng); }), C.roots_ranges, "%s: n = 0 accepted", F::name());
    CHECK(throws_invalid([&] { F::gen(0, 1, u128(1), 65, k, s, rng); }), C.roots_ranges, "%s: n = 65 accepted", F::name());
    CHECK(throws_invalid([&] { F::gen(0, 0, u128(1), 3, 0, s, rng); }), C.roots_ranges, "%s: k = 0 accepted", F::name());
    CHECK(throws_invalid([&] { F::gen(0, 1, u128(1), 3, 65, 63, rng); }), C.roots_ranges, "%s: k = 65 accepted", F::name());
    CHECK(throws_invalid([&] { F::gen(0, 1, u128(1), 3, 64, 65, rng); }), C.roots_ranges, "%s: k + s = 129 accepted", F::name());
    CHECK(throws_invalid([&] { F::gen(0, 1, u128(1), 3, 1, 0xFFFFFFFFu, rng); }), C.roots_ranges, "%s: huge s accepted", F::name());
    CHECK(throws_invalid([&] { F::gen_roots(8, 1, u128(1), 3, k, s, z, z); }), C.roots_ranges, "%s: alpha = 2^n accepted (roots)", F::name());
    {
        const TestRng before = rng;
        const bool threw = throws_invalid([&] { F::gen(8, 1, u128(1), 3, k, s, rng); });
        CHECK(threw && rng.s == before.s, C.roots_ranges, "%s: a rejected call consumed randomness", F::name());
    }
    const auto keys = F::gen(3, 1, u128(1), 3, k, s, rng);
    CHECK(throws_invalid([&] { F::eval(0, keys.first, 8); }), C.roots_ranges, "%s: x = 2^n accepted", F::name());
    CHECK(throws_invalid([&] { F::eval(1, keys.first, 0); }), C.roots_ranges, "%s: wrong party index accepted", F::name());
    CHECK(throws_invalid([&] { F::eval(2, keys.first, 0); }), C.roots_ranges, "%s: party index 2 accepted", F::name());
    {
        typename F::Key bad = keys.first;
        bad.cw_s.pop_back();
        std::vector<u128> y, t;
        CHECK(throws_invalid([&] { F::eval(0, bad, 0); }) && throws_invalid([&] { F::eval_all(bad, y, t); }) &&
                  throws_invalid([&] { bad.to_bytes(); }),
              C.roots_ranges, "%s: key with a missing correction word accepted", F::name());
        typename F::Key bad2 = keys.first;
        bad2.cw_tl[0] = 2;
        CHECK(throws_invalid([&] { F::eval(0, bad2, 0); }) && throws_invalid([&] { bad2.to_bytes(); }), C.roots_ranges,
              "%s: key with a non-bit control correction accepted", F::name());
        CHECK(throws_invalid([&] { F::eval_all_ptr(keys.first, nullptr, nullptr, 1); }), C.roots_ranges,
              "%s: EvalAll without output buffers accepted", F::name());
    }
    {
        const auto wide = F::gen(1, 1, u128(1), fss::EVAL_ALL_MAX_N + 1, k, s, rng);
        std::vector<u128> y, t;
        CHECK(throws_invalid([&] { F::eval_all(wide.first, y, t); }) && y.empty() && t.empty(), C.roots_ranges,
              "%s: EvalAll on a domain of 2^%u points accepted", F::name(), fss::EVAL_ALL_MAX_N + 1);
    }
}

// ------------------------------------------------------------ malformed encodings
template <class F>
static void decode_rejects(TestRng& rng) {
    const unsigned n = 5, k = 32, s = 64;
    const auto keys = F::gen(rng.bits64(n), rng.bits64(k), fss::make_delta(rng.next(), rng.next()), n, k, s, rng);
    const std::vector<uint8_t> good = keys.second.to_bytes();
    auto rejects = [](const std::vector<uint8_t>& blob) {
        return throws_invalid([&] { F::Key::from_bytes(blob); });
    };
    CHECK(F::Key::from_bytes(good) == keys.second, C.decode_rejects, "%s: valid key rejected", F::name());
    // Every proper prefix and every extension by one byte.
    bool prefixes = true;
    for (size_t len = 0; len < good.size(); len++)
        prefixes = prefixes && rejects(std::vector<uint8_t>(good.begin(), good.begin() + (std::ptrdiff_t)len));
    CHECK(prefixes, C.decode_rejects, "%s: a truncated key was accepted", F::name());
    std::vector<uint8_t> longer = good;
    longer.push_back(0);
    CHECK(rejects(longer), C.decode_rejects, "%s: trailing byte accepted", F::name());
    longer[0]++;  // consistent length prefix, wrong body length for the parameters
    CHECK(rejects(longer), C.decode_rejects, "%s: body longer than its parameters imply accepted", F::name());
    CHECK(throws_invalid([&] { F::Key::from_bytes(nullptr, 0); }), C.decode_rejects, "%s: null buffer accepted", F::name());
    // Header fields: offsets 4 family, 5 b, 6 n, 7 k, 8 s.
    struct { size_t off; uint8_t val; const char* what; } header[] = {
        {4, 0, "family 0"}, {4, 3, "family 3"}, {4, (uint8_t)(F::is_dcf() ? 1 : 2), "the other family"},
        {5, 2, "b = 2"}, {5, 255, "b = 255"}, {6, 0, "n = 0"}, {6, 65, "n = 65"}, {6, 4, "n too small for the length"},
        {6, 6, "n too large for the length"}, {7, 0, "k = 0"}, {7, 65, "k = 65"}, {7, 24, "k changing the word size"},
        {8, 97, "k + s = 129"}, {8, 56, "s changing the word size"}};
    for (auto& h : header) {
        std::vector<uint8_t> bad = good;
        bad[h.off] = h.val;
        CHECK(rejects(bad), C.decode_rejects, "%s: %s accepted", F::name(), h.what);
    }
    // Stray bits in every control byte.
    const size_t wb = (k + s + 7) / 8;
    const size_t per_level = 16 + 1 + (F::is_dcf() ? 2 * wb : 0);
    bool stray = true;
    for (unsigned i = 0; i < n; i++)
        for (unsigned bit = 2; bit < 8; bit++) {
            std::vector<uint8_t> bad = good;
            bad[4 + 5 + 16 + i * per_level + 16] |= (uint8_t)(1u << bit);
            stray = stray && rejects(bad);
        }
    CHECK(stray, C.decode_rejects, "%s: stray control bits accepted", F::name());
    // Any other single-bit change of the body is a different valid key.
    bool flips = true;
    for (int trial = 0; trial < 200; trial++) {
        std::vector<uint8_t> other = good;
        const size_t pos = 9 + (size_t)(rng.next() % (good.size() - 9));
        const unsigned bit = (unsigned)(rng.next() & 7);
        const size_t rel = pos - 25;  // offset inside the level records
        const bool is_ctrl = pos >= 25 && rel < n * per_level && rel % per_level == 16;
        if (is_ctrl && bit >= 2) continue;
        other[pos] ^= (uint8_t)(1u << bit);
        const typename F::Key parsed = F::Key::from_bytes(other);
        flips = flips && parsed != keys.second && parsed.to_bytes() == other;
    }
    CHECK(flips, C.decode_rejects, "%s: a bit flip did not give a different key that re-encodes", F::name());
    // Words wider than the ring: (k, s) = (31, 64) has w = 95 and 12-byte
    // words, so bit 95 of every word must be zero.
    {
        const unsigned k2 = 31, w2 = k2 + s;
        const auto kk = F::gen(rng.bits64(n), rng.bits64(k2), fss::make_delta(rng.next(), rng.next()), n, k2, s, rng);
        const std::vector<uint8_t> g2 = kk.first.to_bytes();
        const size_t wb2 = (w2 + 7) / 8, per2 = 16 + 1 + (F::is_dcf() ? 2 * wb2 : 0);
        CHECK(F::Key::from_bytes(g2) == kk.first && g2.size() == expected_size(F::is_dcf(), n, k2, s), C.decode_rejects,
              "%s: (k, s) = (31, 64) round trip", F::name());
        bool out_of_range = true;
        std::vector<size_t> word_ends;  // index of the last byte of every ring element
        for (unsigned i = 0; i < n && F::is_dcf(); i++) {
            const size_t base = 25 + i * per2 + 17;
            word_ends.push_back(base + wb2 - 1);
            word_ends.push_back(base + 2 * wb2 - 1);
        }
        word_ends.push_back(g2.size() - wb2 - 1);
        word_ends.push_back(g2.size() - 1);
        for (size_t pos : word_ends) {
            std::vector<uint8_t> bad = g2;
            bad[pos] |= 0x80;
            out_of_range = out_of_range && rejects(bad);
        }
        CHECK(out_of_range, C.decode_rejects, "%s: correction word >= 2^w accepted", F::name());
    }
}

// ------------------------------------------------------------ other (k, s)
template <class F>
static void other_params(TestRng& rng) {
    const unsigned params[][2] = {{1, 0},  {1, 1},   {8, 8},   {16, 16}, {17, 40}, {31, 64}, {32, 32}, {33, 95},
                                  {64, 0}, {64, 63}, {63, 64}, {1, 127}, {64, 1},  {7, 57},  {40, 24}, {64, 64}};
    for (auto& p : params) {
        const unsigned k = p[0], s = p[1], w = k + s;
        const u128 mask = u128_mask(w);
        for (unsigned n = 1; n <= 6; n++) {
            const size_t size = (size_t)1 << n;
            const uint64_t alpha = rng.bits64(n);
            const uint64_t beta = (n & 1) ? max_beta(k) : rng.bits64(k);
            const u128 d0 = rng.bits128(s), d1 = rng.bits128(s);
            const u128 delta = d0 + d1;  // < 2^(s+1) <= 2^128, no wrap-around
            const u128 beta_t = (delta * u128(beta)) & mask;
            const auto keys = F::gen(alpha, beta, delta, n, k, s, rng);
            std::vector<u128> y0, t0, y1, t1, vals(size), chis(size);
            F::eval_all(keys.first, y0, t0);
            F::eval_all(keys.second, y1, t1);
            bool ok = y0.size() == size && t0.size() == size && y1.size() == size && t1.size() == size;
            for (size_t x = 0; ok && x < size; x++) {
                const bool ind = F::ind(x, alpha);
                const fss::Share a0 = F::eval(0, keys.first, x), a1 = F::eval(1, keys.second, x);
                vals[x] = (y0[x] + y1[x]) & mask;
                chis[x] = rng.bits128(s);
                ok = vals[x] == (ind ? u128(beta) : u128(0)) && ((t0[x] + t1[x]) & mask) == (ind ? beta_t : u128(0)) &&
                     a0.y == y0[x] && a0.tau == t0[x] && a1.y == y1[x] && a1.tau == t1[x] && (y0[x] & mask) == y0[x] &&
                     (t0[x] & mask) == t0[x] && (y1[x] & mask) == y1[x] && (t1[x] & mask) == t1[x];
            }
            CHECK(ok, C.other_params, "%s (k,s)=(%u,%u) n=%u: EvalAll / Eval wrong", F::name(), k, s, n);
            CHECK(ok && release_check(chis, vals, t0, t1, d0, d1, mask), C.other_params,
                  "%s (k,s)=(%u,%u) n=%u: release check failed", F::name(), k, s, n);
            const std::vector<uint8_t> blob = keys.first.to_bytes();
            CHECK(F::Key::from_bytes(blob) == keys.first && blob.size() == expected_size(F::is_dcf(), n, k, s) &&
                      fss::encode_vector(y0, w).size() == size * ((w + 7) / 8),
                  C.other_params, "%s (k,s)=(%u,%u) n=%u: serialisation", F::name(), k, s, n);
        }
    }
}

// ------------------------------------------------------------ large domains
template <class F>
static void large_domain(unsigned n, unsigned k, TestRng& rng, int variant) {
    const unsigned s = 64, w = k + s;
    const u128 mask = u128_mask(w);
    const size_t size = (size_t)1 << n;
    const uint64_t alpha = variant == 0 ? rng.bits64(n) : variant == 1 ? 0 : variant == 2 ? size - 1 : (size >> 1);
    const uint64_t beta = variant == 0 ? rng.bits64(k) : max_beta(k);
    const uint64_t d0 = rng.next(), d1 = rng.next();
    const u128 delta = fss::make_delta(d0, d1);
    const u128 beta_t = (delta * u128(beta)) & mask;
    const auto keys = F::gen(alpha, beta, delta, n, k, s, rng);
    std::vector<u128> y0, t0, y1, t1;
    F::eval_all(keys.first, y0, t0);
    F::eval_all(keys.second, y1, t1);
    bool ok = y0.size() == size && t0.size() == size && y1.size() == size && t1.size() == size;
    size_t bad = 0;
    for (size_t x = 0; ok && x < size; x++) {
        const bool ind = F::ind(x, alpha);
        ok = ((y0[x] + y1[x]) & mask) == (ind ? u128(beta) : u128(0)) &&
             ((t0[x] + t1[x]) & mask) == (ind ? beta_t : u128(0)) && (y0[x] & mask) == y0[x] && (t0[x] & mask) == t0[x] &&
             (y1[x] & mask) == y1[x] && (t1[x] & mask) == t1[x];
        if (!ok) bad = x;
    }
    CHECK(ok, C.large_domain, "%s n=%u k=%u alpha=%llu: EvalAll reconstruction fails at x=%zu", F::name(), n, k,
          (unsigned long long)alpha, bad);
    if (!ok) return;
    // Point Eval on the boundary of every subtree near alpha and at random points.
    std::vector<uint64_t> pts = {0, size - 1, alpha, (alpha + size - 1) % size, (alpha + 1) % size, 1023 % size, 1024 % size};
    while (pts.size() < 64) pts.push_back(rng.bits64(n));
    bool pts_ok = true;
    for (uint64_t x : pts) {
        const fss::Share a0 = F::eval(0, keys.first, x), a1 = F::eval(1, keys.second, x);
        pts_ok = pts_ok && a0.y == y0[x] && a0.tau == t0[x] && a1.y == y1[x] && a1.tau == t1[x];
    }
    CHECK(pts_ok, C.large_domain, "%s n=%u k=%u: Eval != EvalAll at a sampled point", F::name(), n, k);
    // Thread counts (including more threads than subtrees) and the pointer overload.
    const unsigned thread_counts[] = {0, 2, 3, 7, 64, 200};
    bool th_ok = true;
    std::vector<u128> a(size), b(size);
    for (unsigned th : thread_counts) {
        std::fill(a.begin(), a.end(), ~u128(0));
        std::fill(b.begin(), b.end(), ~u128(0));
        F::eval_all_ptr(keys.second, a.data(), b.data(), th);
        th_ok = th_ok && a == y1 && b == t1;
    }
    std::vector<u128> c, d;
    F::eval_all(keys.first, c, d, 4);
    th_ok = th_ok && c == y0 && d == t0;
    CHECK(th_ok, C.large_domain, "%s n=%u k=%u: EvalAll depends on the thread count", F::name(), n, k);
}

// ------------------------------------------------------------ wide indices
template <class F>
static void wide_index(unsigned n, unsigned k, TestRng& rng) {
    const unsigned s = 64, w = k + s;
    const u128 mask = u128_mask(w);
    const uint64_t nmask = n >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1);
    const uint64_t alphas[5] = {0, nmask, rng.bits64(n), (uint64_t)1 << (n - 1), 1};
    for (uint64_t alpha : alphas) {
        const uint64_t beta = rng.bits64(k) | 1;
        const uint64_t d0 = rng.next(), d1 = rng.next();
        const u128 delta = fss::make_delta(d0, d1);
        const u128 beta_t = (delta * u128(beta)) & mask;
        const auto keys = F::gen(alpha, beta, delta, n, k, s, rng);
        const uint64_t pts[] = {alpha, (alpha - 1) & nmask, (alpha + 1) & nmask, 0, nmask, nmask >> 1, (nmask >> 1) + 1,
                                rng.bits64(n), rng.bits64(n), alpha ^ 1, alpha ^ ((uint64_t)1 << (n - 1))};
        bool ok = true;
        for (uint64_t x : pts) {
            const fss::Share a0 = F::eval(0, keys.first, x), a1 = F::eval(1, keys.second, x);
            const bool ind = F::ind(x, alpha);
            ok = ok && ((a0.y + a1.y) & mask) == (ind ? u128(beta) : u128(0)) &&
                 ((a0.tau + a1.tau) & mask) == (ind ? beta_t : u128(0));
        }
        CHECK(ok, C.wide_index, "%s n=%u k=%u alpha=%llu: Eval reconstruction wrong", F::name(), n, k, (unsigned long long)alpha);
        const std::vector<uint8_t> blob = keys.second.to_bytes();
        CHECK(F::Key::from_bytes(blob) == keys.second && blob.size() == expected_size(F::is_dcf(), n, k, s), C.wide_index,
              "%s n=%u k=%u: serialisation", F::name(), n, k);
    }
}

// ------------------------------------------------------------ worker threads of EvalAll
// fss::detail::run_parts runs the subtrees of EvalAll on worker threads.  An
// exception inside a worker (in the library: std::bad_alloc from the buffers
// of a subtree) must reach the caller as an exception, after every worker was
// joined; before the fix it left the thread function and ended the process.
static void worker_threads() {
    const size_t part_counts[] = {1, 2, 3, 4, 7, 16, 64};
    for (size_t n : {(size_t)1, (size_t)5, (size_t)64, (size_t)1000}) {
        for (size_t threads : part_counts) {
            // Every index of [0, n) is visited exactly once, whatever the number of parts.
            std::vector<std::atomic<int>> seen(n);
            for (auto& s : seen) s = 0;
            std::atomic<int> calls(0);
            fss::detail::run_parts(n, threads, [&](size_t r0, size_t r1) {
                calls++;
                for (size_t r = r0; r < r1; r++) seen[r]++;
            });
            bool once = true;
            for (auto& s : seen) once = once && s == 1;
            const size_t parts = threads < n ? threads : n;
            CHECK(once && (size_t)calls == parts, C.workers, "run_parts(%zu, %zu): every index once, one call per part", n, threads);

            // One worker throws: the caller gets that exception, and all other parts ran to the end.
            for (size_t bad = 0; bad < parts; bad += (parts > 8 ? 5 : 1)) {
                std::atomic<int> finished(0);
                std::string caught;
                try {
                    fss::detail::run_parts(n, threads, [&](size_t r0, size_t) {
                        if (r0 == n * bad / parts) throw std::runtime_error("worker " + std::to_string(bad));
                        finished++;
                    });
                } catch (const std::runtime_error& e) {
                    caught = e.what();
                }
                CHECK(caught == "worker " + std::to_string(bad) && (size_t)finished == parts - 1, C.workers,
                      "run_parts(%zu, %zu): the exception of part %zu reaches the caller after all workers were joined", n, threads, bad);
            }
            // Several workers throw: the first part's exception is reported; std::bad_alloc keeps its type.
            if (parts >= 2) {
                bool bad_alloc = false;
                try {
                    fss::detail::run_parts(n, threads, [&](size_t r0, size_t) {
                        if (r0 == 0) throw std::bad_alloc();
                        throw std::runtime_error("another part");
                    });
                } catch (const std::bad_alloc&) {
                    bad_alloc = true;
                } catch (...) {
                }
                CHECK(bad_alloc, C.workers, "run_parts(%zu, %zu): the exception of the first failing part is rethrown with its type", n, threads);
            }
        }
    }
    // Through the public interface: an invalid call still throws in the calling thread and a valid
    // multi-threaded call is unaffected (the bulk of the thread tests is in large_domain).
    TestRng rng(77);
    const auto keys = fss::dpf_gen(5, 1, fss::make_delta(rng.next(), rng.next()), 12, 64, 64, rng);
    std::vector<u128> y1, t1, y8, t8;
    fss::dpf_eval_all(keys.first, y1, t1, 1);
    fss::dpf_eval_all(keys.first, y8, t8, 8);
    CHECK(y1 == y8 && t1 == t8 && y1.size() == 4096, C.workers, "EvalAll with 8 worker threads equals the single-threaded result");
    CHECK(throws_invalid([&] { fss::dpf_eval_all(keys.first, nullptr, nullptr, 8); }), C.workers, "EvalAll without buffers is refused");
}

int main(int argc, char** argv) {
    bool quick = false;
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    const unsigned max_n = quick ? 6 : 8;
    const unsigned ks[2] = {32, 64};
    std::printf("Couplet C++ DPF/DCF test: exhaustive for n in 1..%u, k in {32, 64}, s = 64, Delta = Delta_0 + Delta_1 "
                "unreduced; Eval at every x; avx2=%d\n", max_n, (int)COUPLET_HAVE_AVX2);
    const double start = now_s();
    try {
        TestRng rng(0x20261010ull);
        for (unsigned k : ks) {
            roots_and_ranges<Dpf>(k, rng);
            roots_and_ranges<Dcf>(k, rng);
        }
        std::printf("  explicit roots / ranges: %llu checks\n", (unsigned long long)C.roots_ranges);
        for (unsigned k : ks) {
            release_tests<Dpf>(k, rng);
            release_tests<Dcf>(k, rng);
        }
        std::printf("  release check: %llu honest openings accepted, %llu negative-control checks\n",
                    (unsigned long long)C.release_single, (unsigned long long)C.release_negative);
        decode_rejects<Dpf>(rng);
        decode_rejects<Dcf>(rng);
        std::printf("  malformed encodings: %llu checks\n", (unsigned long long)C.decode_rejects);
        other_params<Dpf>(rng);
        other_params<Dcf>(rng);
        std::printf("  other (k, s) pairs: %llu checks\n", (unsigned long long)C.other_params);
        const unsigned big_ns[] = {9, 10, 11, 12, 13, 16, 20};
        for (unsigned n : big_ns) {
            if (quick && n > 13) continue;
            for (unsigned k : ks)
                for (int variant = 0; variant < (n <= 13 ? 4 : 1); variant++) {
                    large_domain<Dpf>(n, k, rng, variant);
                    large_domain<Dcf>(n, k, rng, variant);
                }
        }
        std::printf("  large domains (n up to %u, 1..200 threads): %llu checks\n", quick ? 13u : 20u,
                    (unsigned long long)C.large_domain);
        worker_threads();
        std::printf("  worker threads (exceptions are rethrown in the caller): %llu checks\n", (unsigned long long)C.workers);
        const unsigned wide_ns[] = {24, 32, 33, 48, 63, 64};
        for (unsigned n : wide_ns)
            for (unsigned k : ks) {
                wide_index<Dpf>(n, k, rng);
                wide_index<Dcf>(n, k, rng);
            }
        std::printf("  wide indices (n up to 64): %llu checks\n", (unsigned long long)C.wide_index);
        std::fflush(stdout);

        for (unsigned n = 1; n <= max_n; n++)
            for (unsigned k : ks) exhaustive_case<Dpf>(n, k);
        for (unsigned n = 1; n <= max_n; n++)
            for (unsigned k : ks) exhaustive_case<Dcf>(n, k);
        const double frac = C.keys ? (double)C.keys_big_delta / (double)C.keys : 0.0;
        CHECK(frac > 0.4 && frac < 0.6, C.misc, "fraction of keys with Delta >= 2^s is %.3f, expected about 1/2", frac);
    } catch (const std::exception& e) {
        report_failure("unexpected exception: %s", e.what());
    }

    std::printf("keys generated in the exhaustive part: %llu (%llu with Delta >= 2^s)\n", (unsigned long long)C.keys,
                (unsigned long long)C.keys_big_delta);
    const struct { const char* name; uint64_t v; } rows[] = {
        {"evalall_range", C.evalall_range}, {"evalall_value", C.evalall_value}, {"evalall_tag", C.evalall_tag},
        {"eval_vs_evalall", C.eval_vs_evalall}, {"eval_sum", C.eval_sum}, {"serialisation", C.serialisation},
        {"release_batch", C.release_batch}, {"release_single", C.release_single},
        {"release_negative_control", C.release_negative}, {"roots_and_ranges", C.roots_ranges},
        {"malformed_encodings", C.decode_rejects}, {"other_k_s", C.other_params}, {"large_domain", C.large_domain},
        {"wide_index", C.wide_index}, {"delta_carry_fraction", C.misc}, {"worker_threads", C.workers}};
    for (auto& r : rows) std::printf("  %-26s %10llu\n", r.name, (unsigned long long)r.v);
    if (g_fail) {
        std::printf("TOTAL CHECKS: %llu, FAILED: %llu  (%.1f s)\n", (unsigned long long)C.total(), (unsigned long long)g_fail,
                    now_s() - start);
        return 1;
    }
    std::printf("TOTAL CHECKS: %llu  (all passed, %.1f s)\n", (unsigned long long)C.total(), now_s() - start);
    return 0;
}
