// test_prep.cpp -- tests of the Boolean preprocessing (DESIGN.md Sections
// 2.2, 7.1 - 7.5, 7.7): params.hpp, shares.hpp, baseot.hpp, cotext.hpp,
// abits.hpp, aand.hpp, prep.hpp, twoparty.hpp.  The tests mirror
// python/tests/test_prep.py.
//
//   params      bucket sizes of Lemma 4: exact thresholds (compared with the
//               table produced by the Python module), minimality, the
//               position of the maximum; domain constants; triple_request:
//               every run has at least 306 leaky triples (known answers
//               shared with python/tests/test_prep.py).
//   kernels     SHA-256 with and without the SHA extensions, GF(2^128) inner
//               products, the bulk Ascon kernels, 64 x 64 bit transposition,
//               column sums and the bucketing permutation against
//               straightforward references, with one and several workers.
//   shares      the data model: XOR, slicing, selection, constants, wire
//               encoding, BDOZ conversions.
//   vectors     values produced by the Python implementation (hash domains,
//               base-OT derivations, an OT-extension transcript, a leaky-AND
//               transfer message, the triple-check combination, BoolCheck
//               digests, the bucketing permutation).
//   base OT     Masny-Rindal, both directions, 20 runs; invalid points abort.
//   COT         q_i = t_i XOR u_i Delta for all rows; an inconsistent column
//               of U is rejected exactly when the sender reads that column.
//   abits       MAC invariants of 10^5 random bits, BDOZ invariants, masks.
//   leaky AND   z = x AND y and the MAC invariant for 10^5 triples.
//   Lemma 3     additive error on the transfer message of t triples, 200 runs
//               for each t in {1, 2, 3}: the check passes iff every guess is
//               right, and then all triples are correct.
//   bucketing   the public permutation and the bucket combination.
//   RealPrep    and_triples end to end for n in {1000, 300000, 1000000}:
//               correctness, MAC invariants, bucket size, exact bytes and
//               rounds per phase; several requests per session; TCP.
//   small       requests below 306 leaky triples are padded; the whole
//               padded run is checked (deviations in the padding abort).
//   aborts      a tampered opened d bit, errors on transfer messages, a wrong
//               share of W, an inconsistent extension matrix, corrupted
//               base-OT seeds and a vanished peer all retire the session.
//   IdealPrep   the same invariants for the ideal dealer.
//   BoolCheck   honest openings pass; tampered bits, MAC shares, field
//               openings, counters and session identifiers abort.
//
// Both parties run as two threads of this process (run_two_party), so the
// times printed are wall-clock times with both parties on one machine.
//
// Usage: test_prep [--no-large]     (--no-large skips n = 300000 and 1000000)
//        test_prep --interop PARTY PORT N OUTFILE
// Exit code 0 iff every check passes.  The --interop mode runs
// RealPrep.setup() and and_triples(N) over TCP on 127.0.0.1 (party 0 listens,
// party 1 connects) and writes this party's key share and triple shares to
// OUTFILE, so that the run can be paired with another implementation of the
// same protocol (the Python package).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "couplet/aand.hpp"
#include "couplet/abits.hpp"
#include "couplet/baseot.hpp"
#include "couplet/channel.hpp"
#include "couplet/cotext.hpp"
#include "couplet/ed25519.hpp"
#include "couplet/gf128.hpp"
#include "couplet/hashes.hpp"
#include "couplet/params.hpp"
#include "couplet/prep.hpp"
#include "couplet/rng.hpp"
#include "couplet/sha256.hpp"
#include "couplet/shares.hpp"
#include "couplet/twoparty.hpp"

using namespace couplet;
using namespace couplet::prep;

// ------------------------------------------------------------ harness
static int g_pass = 0, g_fail = 0;
static bool g_large = true;

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
    std::chrono::steady_clock::time_point t0;
    explicit Section(const char* n) : name(n), pass0(g_pass), fail0(g_fail), t0(std::chrono::steady_clock::now()) {}
    ~Section() {
        const int f = g_fail - fail0, p = g_pass - pass0;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("[%s] %s (%d checks, %.2f s)\n", f ? "FAIL" : "PASS", name, p + f, s);
        std::fflush(stdout);
    }
};

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

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

static std::string sha_hex(const bytes& b) {
    uint8_t h[32];
    sha256(b.data(), b.size(), h);
    return hex(h, 32);
}

template <class Ex, class F>
static bool throws(F&& f) {
    try {
        f();
    } catch (const Ex&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

static const bytes TSID = {'t', 'e', 's', 't', '-', 'p', 'r', 'e', 'p', '-', 's', 'e', 's', 's', 'i', 'o', 'n'};
static const size_t COIN_BYTES = 96;       // per party: commitment, opening, share (32 bytes each)
static const size_t BOOLCHECK_BYTES = 96;  // per party: commitment, opening, digest

// ------------------------------------------------------------ helpers
// Deterministic test data: views of ctr_prg(seed = 16 bytes `tag`, dom::PRG_TEST),
// the same streams the Python script that produced the vectors used.
static hashes::seed16 tag_seed(uint8_t tag) {
    hashes::seed16 s;
    s.fill(tag);
    return s;
}
static GfVec tvec(uint8_t tag, size_t n) { return hashes::ctr_prg_gf128(tag_seed(tag), hashes::dom::PRG_TEST, n); }
static BitVec tbits(uint8_t tag, size_t n) {
    BitVec b(n);
    hashes::ctr_prg_bits(tag_seed(tag), hashes::dom::PRG_TEST, b.data(), n);
    return b;
}

static CtrPrgRng g_rng((uint64_t)0x7e57);  // values chosen by the tests (reproducible)

static BitVec rand_bits(size_t n) {
    BitVec b(n);
    g_rng.fill_bits(b.data(), n);
    return b;
}
static GfVec rand_vec(size_t n) {
    GfVec v(n);
    bytes raw(16 * n);
    if (n) g_rng.fill(raw.data(), raw.size());
    gf_load(v.data(), raw.data(), n);
    return v;
}
static size_t rand_below(size_t m) { return (size_t)(g_rng.u64() % m); }

// Key shares with lsb(Delta_0) = 1 and lsb(Delta_1) = 0.
static void test_deltas(gf128 d[2]) {
    d[0] = g_rng.gf();
    d[0].lo |= 1;
    d[1] = g_rng.gf();
    d[1].lo &= ~1ull;
}

struct TrustedSeeds {
    GfVec chosen, s0, s1;
};
// Base-OT output sampled by the test for the COT sender key `delta`.
static TrustedSeeds trusted_seeds(const gf128& delta) {
    TrustedSeeds t;
    t.s0 = rand_vec(KAPPA);
    t.s1 = rand_vec(KAPPA);
    t.chosen.resize(KAPPA);
    for (unsigned j = 0; j < KAPPA; j++) t.chosen[j] = delta.bit(j) ? t.s1[j] : t.s0[j];
    return t;
}

static BitVec xor_bits(const BitVec& a, const BitVec& b) {
    BitVec r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = (uint8_t)(a[i] ^ b[i]);
    return r;
}
static BitVec and_bits(const BitVec& a, const BitVec& b) {
    BitVec r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = (uint8_t)(a[i] & b[i]);
    return r;
}
static double mean(const BitVec& b) {
    uint64_t s = 0;
    for (uint8_t v : b) s += v;
    return b.empty() ? 0.0 : (double)s / (double)b.size();
}

// m_0 XOR m_1 = x Delta for all positions (and x = value if given).
static bool mac_ok(const Shares& s0, const Shares& s1, const gf128& delta, const BitVec* value = nullptr) {
    if (s0.size() != s1.size() || s0.bits.size() != s0.macs.size() || s1.bits.size() != s1.macs.size()) return false;
    if (value && value->size() != s0.size()) return false;
    for (size_t i = 0; i < s0.size(); i++) {
        const unsigned x = (unsigned)(s0.bits[i] ^ s1.bits[i]);
        if (x > 1) return false;
        if (value && (*value)[i] != x) return false;
        if ((s0.macs[i] ^ s1.macs[i]) != gf_and_bit(delta, x)) return false;
    }
    return true;
}

// M[x] = K[x] XOR x Delta for all positions.
static bool bdoz_ok(const Bdoz& own, const Keys& keys, const gf128& key_delta) {
    if (own.size() != keys.size()) return false;
    for (size_t i = 0; i < own.size(); i++)
        if (own.macs[i] != (keys.keys[i] ^ gf_and_bit(key_delta, own.bits[i]))) return false;
    return true;
}

// c = a AND b for all triples; MAC invariants of a, b, c.
static bool triples_correct(const Triples& t0, const Triples& t1) {
    if (t0.size() != t1.size()) return false;
    for (size_t i = 0; i < t0.size(); i++) {
        const unsigned a = t0.a.bits[i] ^ t1.a.bits[i], b = t0.b.bits[i] ^ t1.b.bits[i], c = t0.c.bits[i] ^ t1.c.bits[i];
        if (c != (a & b)) return false;
    }
    return true;
}
static bool triples_macs(const Triples& t0, const Triples& t1, const gf128& delta) {
    return mac_ok(t0.a, t1.a, delta) && mac_ok(t0.b, t1.b, delta) && mac_ok(t0.c, t1.c, delta);
}

// Fresh <v> sharing of the given bits under Delta = d0 XOR d1.
static void fresh(const BitVec& values, const gf128& delta, Shares& s0, Shares& s1) {
    s0.bits = rand_bits(values.size());
    s0.macs = rand_vec(values.size());
    s1 = Shares(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        s1.bits[i] = (uint8_t)(s0.bits[i] ^ values[i]);
        s1.macs[i] = s0.macs[i] ^ gf_and_bit(delta, values[i]);
    }
}

// Bytes sent by one party in the extension for `rows` rows per direction, both
// directions in parallel, in chunks of `batch` rows; and the number of chunks.
static uint64_t cot_bytes(size_t rows, size_t batch = DEFAULT_BATCH_ROWS) {
    uint64_t total = 0;
    for (size_t done = 0; done < rows;) {
        const size_t c = std::min(batch, rows - done);
        total += u_message_bytes(c) + 16 + COT_CHECK_BYTES;
        done += c;
    }
    return total;
}
static size_t cot_chunks(size_t rows, size_t batch = DEFAULT_BATCH_ROWS) { return (rows + batch - 1) / batch; }

// A session of two parties holding COT states with trusted seeds:
// body(role, channel, source, delta, meter) -> R.
template <class R, class Body>
static std::pair<PartyOutcome<R>, PartyOutcome<R>> abit_session(const gf128 deltas[2], Body body,
                                                                size_t batch_rows = DEFAULT_BATCH_ROWS) {
    TrustedSeeds seeds[2] = {trusted_seeds(deltas[0]), trusted_seeds(deltas[1])};  // direction d: sender P_d
    auto party = [&](int role) {
        return [&, role](Channel& ch) -> R {
            SystemRng rng;
            PhaseMeter meter(ch);
            CotSender snd(role, deltas[role], seeds[role].chosen, rng);
            CotReceiver rcv(1 - role, seeds[1 - role].s0, seeds[1 - role].s1, rng);
            AbitSource src(ch, role, deltas[role], snd, rcv, batch_rows, &meter);
            return body(role, ch, src, deltas[role], meter);
        };
    };
    auto f0 = party(0);
    auto f1 = party(1);
    return run_two_party(f0, f1);
}

// ------------------------------------------------------------ (a) parameters
static void test_params() {
    Section sec("params: Lemma 4 bucket sizes and domain constants");
    check(KAPPA == 128 && PAD == 128 && SIGMA_DEFAULT == 40, "KAPPA = PAD = 128, sigma = 40");

    // Multi-precision helper.
    check(prep::detail::binomial(10, 3).to_double() == 120.0 && prep::detail::binomial(5, 0).to_double() == 1.0 &&
              prep::detail::binomial(3, 5).is_zero(),
          "binomial: small values");
    check(prep::detail::binomial(64, 32).compare(prep::detail::BigUInt(1832624140942590534ull)) == 0, "binomial(64, 32)");
    check(std::fabs(prep::detail::binomial(100, 50).to_double() / 1.0089134454556419e29 - 1.0) < 1e-12, "binomial(100, 50)");
    {
        prep::detail::BigUInt a(0xffffffffffffffffull), b(1);
        a.mul_u64(0xffffffffffffffffull);  // (2^64 - 1)^2 = 2^128 - 2^65 + 1
        b.shift_left(128);
        prep::detail::BigUInt c(1);
        c.shift_left(65);
        a.add(c);
        b.add(prep::detail::BigUInt(1));
        check(a.compare(b) == 0, "BigUInt: (2^64 - 1)^2 + 2^65 = 2^128 + 1");
        prep::detail::BigUInt q(1);
        q.shift_left(100);
        check(q.div_u32(1u << 20) == 0 && q.to_double() == std::ldexp(1.0, 80), "BigUInt: exact division");
    }

    // Values quoted in DESIGN.md 7.5 (sigma = 40).
    check(bucket_size(280000, 40) == 3, "sigma=40: B = 3 at ell = 280,000");
    check(bucket_size(3100, 40) == 4, "sigma=40: B = 4 at ell = 3,100");
    check(bucket_size(3000, 40) == 5, "sigma=40: B = 5 at ell = 3,000");
    const uint64_t t3 = min_ell(3, 40), t4 = min_ell(4, 40), t5 = min_ell(5, 40);
    std::printf("  exact thresholds, sigma=40: B=3 from ell=%llu, B=4 from ell=%llu, B=5 from ell=%llu\n",
                (unsigned long long)t3, (unsigned long long)t4, (unsigned long long)t5);
    check(t3 == 276325, "sigma=40: exact threshold of B = 3 is 276,325");
    check(t4 == 3044, "sigma=40: exact threshold of B = 4 is 3,044");
    check(t3 <= 280000 && std::fabs((double)t3 / 280000.0 - 1.0) < 0.02, "B = 3 threshold within 2% of the quoted 280,000");
    check(t4 <= 3100 && std::fabs((double)t4 / 3100.0 - 1.0) < 0.02, "B = 4 threshold within 2% of the quoted 3,100");

    // Thresholds of the table written by the Python module (exact rational arithmetic there).
    static const uint64_t THR40[7] = {206158430209ull, 276325ull, 3044ull, 320ull, 84ull, 34ull, 19ull};
    static const uint64_t THR64[7] = {3458764513820540929ull, 1131823262ull, 779083ull, 20434ull, 2300ull, 537ull, 190ull};
    for (unsigned b = 2; b <= 8; b++) {
        check(min_ell(b, 40) == THR40[b - 2], "sigma=40: threshold of B = " + std::to_string(b) + " equals the Python table");
        check(min_ell(b, 64) == THR64[b - 2], "sigma=64: threshold of B = " + std::to_string(b) + " equals the Python table");
        const uint64_t thr = THR40[b - 2];
        check(bucket_size(thr, 40) == b && bucket_size(thr - 1, 40) == b + 1,
              "sigma=40: bucket size switches to " + std::to_string(b) + " exactly at its threshold");
    }
    // Rows of the table: B and log2 of the bound for ell = 2^10 .. 2^24.
    static const unsigned B40[15] = {5, 5, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3};
    static const unsigned B64[15] = {7, 7, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4};
    static const double L40[15] = {-46.723, -50.724, -41.285, -44.285, -47.286, -50.286, -53.286, -56.286,
                                   -59.286, -41.848, -43.848, -45.848, -47.848, -49.848, -51.848};
    static const double L64[15] = {-69.603, -75.605, -68.165, -73.166, -78.166, -66.725, -70.725, -74.725,
                                   -78.725, -82.725, -65.286, -68.286, -71.286, -74.286, -77.286};
    bool rows_b = true, rows_l = true;
    std::string line40 = "  sigma=40:", line64 = "  sigma=64:";
    for (unsigned e = 10; e <= 24; e++) {
        const uint64_t ell = 1ull << e;
        const unsigned b40 = bucket_size(ell, 40), b64 = bucket_size(ell, 64);
        rows_b = rows_b && b40 == B40[e - 10] && b64 == B64[e - 10];
        rows_l = rows_l && std::fabs(log2_leak_bound(ell, b40) - L40[e - 10]) < 2e-3 &&
                 std::fabs(log2_leak_bound(ell, b64) - L64[e - 10]) < 2e-3;
        line40 += " 2^" + std::to_string(e) + ":" + std::to_string(b40);
        line64 += " 2^" + std::to_string(e) + ":" + std::to_string(b64);
    }
    std::printf("%s\n%s\n", line40.c_str(), line64.c_str());
    check(rows_b, "bucket sizes for ell = 2^10 .. 2^24 equal the Python table (sigma = 40 and 64)");
    check(rows_l, "log2 of the bound for ell = 2^10 .. 2^24 equals the Python table to 0.002");

    // The maximum over t is attained at t* = min(2B - 1, ell B): no other term is larger.
    {
        const uint64_t ells[7] = {1, 2, 3, 5, 8, 13, 40};
        const uint64_t buckets[6] = {1, 2, 3, 4, 5, 7};
        bool argmax = true;
        for (uint64_t ell : ells) {
            for (uint64_t b : buckets) {
                const uint64_t ts = prep::detail::leak_argmax(ell, b);
                for (uint64_t t = b; t <= ell * b; t++) {  // C(t, B) 2^t* <= C(t*, B) 2^t
                    prep::detail::BigUInt l = prep::detail::binomial(t, b), r = prep::detail::binomial(ts, b);
                    l.shift_left(ts);
                    r.shift_left(t);
                    if (l.compare(r) > 0) argmax = false;
                }
            }
        }
        check(argmax, "the Lemma 4 term is maximal at t = min(2B - 1, ell B) (brute force over all t)");
    }

    // Minimality and monotonicity.
    for (int sigma : {40, 64}) {
        const uint64_t ells[9] = {1, 2, 7, 100, 1 << 10, 5000, 1 << 16, 1 << 20, 1 << 24};
        unsigned prev = 0;
        bool holds = true, minimal = true, mono = true;
        for (uint64_t ell : ells) {
            const unsigned b = bucket_size(ell, sigma);
            holds = holds && leak_bound_at_most(ell, b, sigma);
            minimal = minimal && (b == 2 || !leak_bound_at_most(ell, b - 1, sigma));
            mono = mono && (prev == 0 || b <= prev);
            prev = b;
        }
        check(holds, "sigma=" + std::to_string(sigma) + ": the bound holds for the chosen B");
        check(minimal, "sigma=" + std::to_string(sigma) + ": B - 1 is not enough");
        check(mono, "sigma=" + std::to_string(sigma) + ": B is non-increasing in ell");
    }
    check(bucket_size(1, 40) == 40 && bucket_size(1, 64) == 64, "a single bucket needs B = sigma");
    check(throws<std::invalid_argument>([] { bucket_size(0, 40); }), "bucket_size rejects ell = 0");
    check(throws<std::invalid_argument>([] { min_ell(1, 40); }), "min_ell rejects B = 1");

    // Domain constants: distinct and disjoint from the FSS label words.
    std::set<uint64_t> doms;
    for (unsigned use : {USE_COT_PRG, USE_COT_CHI, USE_BUCKET_PERM, USE_TRIPLE_CHI})
        for (int d = 0; d < 2; d++)
            for (uint64_t ctr : {(uint64_t)0, (uint64_t)1, MAX_PRG_COUNTER}) doms.insert(prg_dom(use, d, ctr));
    check(doms.size() == 24, "prg_dom is injective on (use, direction, counter)");
    bool fam = true, reserved = false;
    for (uint64_t d : doms) {
        fam = fam && (d >> 56) == 0x50 && (d & hashes::dom::LOW56) >= 0x100;
        reserved = reserved || hashes::dom::is_reserved(d);
    }
    check(fam, "PRG domains: family byte 0x50, identifiers above the reserved range");
    std::set<uint64_t> ccr;
    for (const char* s : {"", "a", "b"})
        for (int d = 0; d < 2; d++) ccr.insert(h_dom(bytes(s, s + std::strlen(s)), d));
    bool ccr_fam = true;
    for (uint64_t d : ccr) {
        ccr_fam = ccr_fam && (d >> 56) == 0x48;
        reserved = reserved || hashes::dom::is_reserved(d);
    }
    check(ccr.size() == 6 && ccr_fam, "hash domains: family byte 0x48, one per (sid, direction)");
    check(!reserved, "domains avoid the FSS label words");
    {
        const bytes a = {'a'};
        const hashes::hash32 manual = hashes::framed_sha256({"Couplet-dom", "aand/1", a});
        check(h_dom(a, 1) == ((0x48ull << 56) | (ascon::load64le(manual.data()) & hashes::dom::LOW56)),
              "session_dom = kind || low 56 bits of SHA-256(frame('Couplet-dom', label, sid))");
    }
    check(throws<std::invalid_argument>([] { prg_dom(0); }) && throws<std::invalid_argument>([] { prg_dom(1, 2); }) &&
              throws<std::invalid_argument>([] { prg_dom(1, 0, 1ull << 40); }) &&
              throws<std::invalid_argument>([] { session_dom(0, "x", "y"); }),
          "domain helpers validate their fields");
}

// ------------------------------------------------------------ size of a triple request
// Known answers of triple_request: (ell, sigma) -> (run size, B).  The same
// table is checked by python/tests/test_prep.py, so the two languages pad a
// request in the same way.
static void test_triple_request() {
    Section sec("params: triple_request (at least 306 leaky triples per run)");
    check(MIN_LEAKY_TRIPLES == 306, "MIN_LEAKY_TRIPLES = 306");
    {
        // 306 is the smallest L with (3/4)^L < 2^-127, i.e. 3^L * 2^127 < 4^L (exact integers).
        auto holds = [](unsigned len) {
            prep::detail::BigUInt lhs(1), rhs(1);
            for (unsigned i = 0; i < len; i++) lhs.mul_u32(3);
            lhs.shift_left(127);
            rhs.shift_left(2 * (uint64_t)len);
            return lhs.compare(rhs) < 0;
        };
        check(holds(306) && !holds(305), "306 is the smallest number of leaky triples with (3/4)^L < 2^-127");
    }
    struct Kat {
        uint64_t ell;
        int sigma;
        uint64_t run;
        unsigned bucket;
    };
    static const Kat KAT[] = {
        {1, 40, 44, 7},       {2, 40, 44, 7},     {18, 40, 44, 7},     {33, 40, 44, 7},         {43, 40, 44, 7},
        {44, 40, 44, 7},      {45, 40, 45, 7},    {64, 40, 64, 7},     {83, 40, 83, 7},         {84, 40, 84, 6},
        {100, 40, 100, 6},    {320, 40, 320, 5},  {3044, 40, 3044, 4}, {276325, 40, 276325, 3}, {1, 64, 26, 12},
        {25, 64, 26, 12},     {26, 64, 26, 12},   {27, 64, 27, 12},    {30, 64, 30, 11},        {50, 64, 50, 10},
        {200, 64, 200, 8},    {1, 8, 153, 2},     {152, 8, 153, 2},    {153, 8, 153, 2},        {154, 8, 154, 2},
        {1, 20, 77, 4},       {1, 128, 11, 28},   {1, 306, 2, 154},    {1, 400, 2, 201},
    };
    for (const Kat& k : KAT) {
        const TripleRequest r = triple_request(k.ell, k.sigma);
        check(r.ell == k.run && r.bucket == k.bucket && r.leaky() == k.run * k.bucket,
              "triple_request(" + std::to_string(k.ell) + ", " + std::to_string(k.sigma) + ") = (" + std::to_string(k.run) +
                  ", " + std::to_string(k.bucket) + "), got (" + std::to_string(r.ell) + ", " + std::to_string(r.bucket) + ")");
    }
    check(triple_request(5).ell == triple_request(5, 40).ell && triple_request(5).bucket == triple_request(5, 40).bucket,
          "the default sigma is 40");
    for (int sigma : {40, 64}) {
        bool ok = true;
        uint64_t prev = 0, padded = 0, last_padded = 0;
        for (uint64_t ell = 1; ell <= 400 && ok; ell++) {
            const TripleRequest r = triple_request(ell, sigma);
            const uint64_t lo = ell < 2 ? 2 : ell;
            ok = r.ell >= lo && r.leaky() >= MIN_LEAKY_TRIPLES && r.bucket == bucket_size(r.ell, sigma) &&
                 leak_bound_at_most(r.ell, r.bucket, sigma);
            // minimality: no admissible run size below the returned one
            for (uint64_t t = lo; t < r.ell && ok; t++) ok = t * bucket_size(t, sigma) < MIN_LEAKY_TRIPLES;
            // a request is padded exactly if it is a single triple or too small for the check
            const bool small = ell < 2 || ell * bucket_size(ell, sigma) < MIN_LEAKY_TRIPLES;
            ok = ok && (r.ell != ell) == small && r.ell >= prev;
            prev = r.ell;
            if (r.ell != ell) {
                padded++;
                last_padded = ell;
            }
        }
        check(ok, "sigma=" + std::to_string(sigma) + ": run size, bucket size and minimality for ell = 1 .. 400");
        check(padded == last_padded && padded == (sigma == 40 ? 43u : 25u),
              "sigma=" + std::to_string(sigma) + ": exactly the requests up to " + std::to_string(padded) + " triples are padded");
    }
    // The session-setup circuit alone (64 AND gates for s = 64) is above the minimum.
    check(triple_request(64, 40).ell == 64 && triple_request(64, 40).bucket == 7 && triple_request(64, 40).leaky() == 448,
          "a request of 64 triples runs unpadded with B = 7 (448 leaky triples)");
    for (uint64_t ell : {(uint64_t)1000, (uint64_t)68303, (uint64_t)128743, (uint64_t)1 << 20})
        check(triple_request(ell, 40).ell == ell && triple_request(ell, 40).bucket == bucket_size(ell, 40),
              "triple_request(" + std::to_string(ell) + ", 40) does not pad");
    check(throws<std::invalid_argument>([] { triple_request(0, 40); }), "triple_request rejects ell = 0");
}

// ------------------------------------------------------------ kernels
static void test_kernels() {
    Section sec("kernels: SHA-256, GF(2^128) sums, transposition, column sums, permutation");

    // FastSha256 (SHA extensions if present, and the portable path) against Sha256.
    {
        bytes data(100000);
        g_rng.fill(data.data(), data.size());
        bool ok_fast = true, ok_slow = true, ok_split = true;
        for (size_t len = 0; len <= 300; len++) {
            uint8_t ref[32];
            sha256(data.data(), len, ref);
            prep::detail::FastSha256 f, s(true);
            f.update(data.data(), len);
            s.update(data.data(), len);
            ok_fast = ok_fast && std::memcmp(f.finish().data(), ref, 32) == 0;
            ok_slow = ok_slow && std::memcmp(s.finish().data(), ref, 32) == 0;
        }
        for (int trial = 0; trial < 50; trial++) {
            const size_t len = rand_below(data.size());
            uint8_t ref[32];
            sha256(data.data(), len, ref);
            prep::detail::FastSha256 f;
            for (size_t off = 0; off < len;) {
                const size_t take = std::min(len - off, 1 + rand_below(1000));
                f.update(data.data() + off, take);
                off += take;
            }
            ok_split = ok_split && std::memcmp(f.finish().data(), ref, 32) == 0;
        }
        check(ok_fast, "FastSha256 equals SHA-256 for every length 0 .. 300");
        check(ok_slow, "FastSha256 (portable path) equals SHA-256 for every length 0 .. 300");
        check(ok_split, "FastSha256 is independent of how the input is split");
        prep::detail::FastSha256 probe;
        std::printf("  SHA-256 for bulk data: %s\n", probe.accelerated() ? "SHA extensions" : "portable");
        // framed parts
        prep::detail::FastSha256 fr;
        const bytes p1 = {1, 2, 3}, p2;
        fr.update_framed(p1.data(), p1.size());
        fr.update_framed(p2.data(), 0);
        fr.update_frame_length(5);
        fr.update(data.data(), 5);
        const bytes p3(data.begin(), data.begin() + 5);
        check(fr.finish() == hashes::framed_sha256({p1, p2, p3}), "FastSha256 framing equals hashes::framed_sha256");
        // An empty update is a no-op on both code paths, also with a null pointer while the block
        // buffer is partly filled (formerly memcpy(dst, nullptr, 0)).
        bool empty_ok = true;
        for (int portable = 0; portable < 2; portable++) {
            uint8_t ref[32], ref0[32];
            sha256(data.data(), 100, ref);
            sha256(data.data(), 0, ref0);
            prep::detail::FastSha256 b(portable != 0), e(portable != 0), f(portable != 0);
            b.update(data.data(), 37);  // the block buffer is partly filled from here on
            b.update(nullptr, 0);
            b.update(data.data() + 37, 0);
            b.update(data.data() + 37, 63);
            b.update(nullptr, 0);
            e.update(nullptr, 0);
            empty_ok = empty_ok && std::memcmp(b.finish().data(), ref, 32) == 0 && std::memcmp(e.finish().data(), ref0, 32) == 0;
            f.update_framed(nullptr, 0);
            f.update_framed(data.data(), 3);
            const bytes none, d3(data.begin(), data.begin() + 3);
            empty_ok = empty_ok && f.finish() == hashes::framed_sha256({none, d3});
        }
        check(empty_ok, "FastSha256: an update of length 0 (null pointer included) changes nothing, on both code paths");
    }

    // Inner products.
    {
        const size_t n = 1000;
        const GfVec a = rand_vec(n), b = rand_vec(n);
        const BitVec s = rand_bits(n);
        gf128 ref, ref_bits;
        for (size_t i = 0; i < n; i++) {
            ref ^= gf128_mul_portable(a[i], b[i]);
            if (s[i]) ref_bits ^= a[i];
        }
        check(gf_dot(a.data(), b.data(), n) == ref, "gf_dot equals the sum of portable products");
        check(gf_dot(a.data(), b.data(), 0).is_zero() && gf_dot(a.data(), b.data(), 1) == gf128_mul_portable(a[0], b[0]),
              "gf_dot: n = 0 and n = 1");
        check(gf_dot_bits(a.data(), s.data(), n) == ref_bits, "gf_dot_bits selects by bit");
        gf128 hor, xp = gf128::one();
        for (size_t j = 0; j < 128; j++) {
            hor ^= gf128_mul_portable(xp, a[j]);
            xp = gf128_mulx(xp);
        }
        check(gf_horner(a.data(), 128) == hor, "gf_horner = sum_j x^j part_j");
        const BitVec bits = gf_unpack_bits(a[5]);
        check(gf_pack_bits(bits.data()) == a[5] && bits[0] == (a[5].lo & 1) && bits[127] == (a[5].hi >> 63),
              "pack / unpack of the 128 coefficients");
        check(gf_and_bit(a[1], 1) == a[1] && gf_and_bit(a[1], 0).is_zero(), "gf_and_bit");
        const bytes ser = gf_to_bytes(a);
        GfVec back(n);
        gf_load(back.data(), ser.data(), n);
        uint8_t one[16];
        a[7].to_bytes(one);
        check(back == a && std::memcmp(ser.data() + 16 * 7, one, 16) == 0, "16-byte little-endian wire form");
    }

    // 64 x 64 transposition.
    {
        bool ok = true, invol = true;
        for (int trial = 0; trial < 200; trial++) {
            uint64_t a[64], ref[64], b[64];
            for (int i = 0; i < 64; i++) a[i] = (trial == 0) ? (i == 3 ? 1ull << 63 : 0) : g_rng.u64();
            prep::detail::transpose64_ref(a, ref);
            std::memcpy(b, a, sizeof a);
            prep::detail::transpose64(b);
            ok = ok && std::memcmp(b, ref, sizeof b) == 0;
            prep::detail::transpose64(b);
            invol = invol && std::memcmp(b, a, sizeof b) == 0;
        }
        check(ok, "transpose64 equals the bit-by-bit reference (200 matrices)");
        check(invol, "transpose64 is an involution");
    }

    // The bulk Ascon kernels against the reference routines of hashes.hpp.
    {
        const hashes::seed16 seed = g_rng.seed();
        const uint64_t dom = prg_dom(USE_COT_PRG, 1, 7);
        bool prg_ok = true;
        for (size_t n_words : {(size_t)0, (size_t)1, (size_t)2, (size_t)3, (size_t)11, (size_t)12, (size_t)13, (size_t)100, (size_t)3001}) {
            for (uint64_t first : {(uint64_t)0, (uint64_t)5, (uint64_t)0xfffffffffffffff0ull}) {
                std::vector<uint64_t> got(n_words + 1, 0x5a5a5a5a5a5a5a5aull), ref(n_words + 1, 0x5a5a5a5a5a5a5a5aull);
                prep::detail::prg_words(seed, dom, got.data(), n_words, first);
                hashes::ctr_prg(seed.data(), dom, ref.data(), n_words, first);
                prg_ok = prg_ok && got == ref;
            }
        }
        check(prg_ok, "prg_words equals hashes::ctr_prg (lengths, offsets, counter wrap, no overrun)");
        bool ccr_ok = true;
        const gf128 mask = g_rng.gf();
        for (size_t n : {(size_t)0, (size_t)1, (size_t)3, (size_t)4, (size_t)5, (size_t)8, (size_t)1023}) {
            const GfVec x = rand_vec(n);
            GfVec got(n), ref(n), got_m(n), ref_m(n), xm(n), inplace = x;
            const uint64_t tw = g_rng.u64(), cdom = h_dom(TSID, 1);
            for (size_t i = 0; i < n; i++) xm[i] = x[i] ^ mask;
            prep::detail::ccr_hash(x.data(), n, tw, cdom, got.data());
            hashes::H_ccr_batch(x.data(), n, tw, cdom, ref.data());
            prep::detail::ccr_hash(x.data(), n, tw, cdom, got_m.data(), mask);
            hashes::H_ccr_batch(xm.data(), n, tw, cdom, ref_m.data());
            prep::detail::ccr_hash(inplace.data(), n, tw, cdom, inplace.data());
            ccr_ok = ccr_ok && got == ref && got_m == ref_m && inplace == ref;
        }
        check(ccr_ok, "ccr_hash equals hashes::H_ccr_batch (with and without the input mask, in place)");
        check(throws<std::invalid_argument>([&] { uint64_t w[3]; prep::detail::prg_words(seed, 0x47, w, 3); }) &&
                  throws<std::invalid_argument>([&] { gf128 o; prep::detail::ccr_hash(&mask, 1, 0, 0x43, &o); }),
              "the kernels reject the reserved FSS domain words");
    }

    // Column sums and the row pass of the extension (with one and with several workers).
    {
        const size_t nw = 1000, blocks = 997, rows = 64 * blocks;  // 997 blocks: exercises the group tail
        std::vector<uint64_t> cols(128 * nw), u(nw);
        for (uint64_t& w : cols) w = g_rng.u64();
        for (uint64_t& w : u) w = g_rng.u64();
        const gf128 seed = g_rng.gf();
        uint8_t seed_bytes[16];
        seed.to_bytes(seed_bytes);
        const uint64_t dom = prg_dom(USE_COT_CHI, 1, 3);
        GfVec chi(rows);
        hashes::ctr_prg_gf128(seed_bytes, dom, chi.data(), rows);
        GfVec ref_rows(rows);
        for (size_t i = 0; i < rows; i++) {
            uint64_t lo = 0, hi = 0;
            for (size_t j = 0; j < 64; j++) {
                lo |= ((cols[j * nw + i / 64] >> (i % 64)) & 1) << j;
                hi |= ((cols[(64 + j) * nw + i / 64] >> (i % 64)) & 1) << j;
            }
            ref_rows[i] = gf128(lo, hi);
        }
        gf128 ref_sums[128], ref_u;
        prep::detail::column_sums_ref(ref_rows.data(), chi.data(), rows, ref_sums);
        for (size_t i = 0; i < rows; i++)
            if ((u[i / 64] >> (i % 64)) & 1) ref_u ^= chi[i];
        for (unsigned workers : {1u, 3u, 8u}) {
            const WorkerScope scope(workers);
            prep::detail::RowPass pass;
            GfVec out(rows - 10);
            gf128 got[128], got_u;
            pass.run(cols.data(), nw, blocks, seed, dom, u.data(), out.data(), out.size(), got, &got_u);
            bool rows_ok = true, sums_ok = true;
            for (size_t i = 0; i < out.size(); i++) rows_ok = rows_ok && out[i] == ref_rows[i];
            for (size_t j = 0; j < 128; j++) sums_ok = sums_ok && got[j] == ref_sums[j];
            const std::string tag = "row pass (" + std::to_string(workers) + " worker" + (workers > 1 ? "s" : "") + ")";
            check(rows_ok, tag + ": rows are the transposed columns");
            check(sums_ok, tag + ": the 128 column sums equal the bit-by-bit reference");
            check(got_u == ref_u, tag + ": u~ = sum_i u_i chi_i");
        }
        // the padding of a column is its last two words: pack of rows 64 (nw - 2) .. 64 nw - 1
        BitVec padbits(128);
        for (size_t i = 0; i < 128; i++) padbits[i] = (uint8_t)((cols[5 * nw + (nw - 2) + i / 64] >> (i % 64)) & 1);
        check(gf_pack_bits(padbits.data()) == gf128(cols[5 * nw + nw - 2], cols[5 * nw + nw - 1]),
              "pack of a padding column = its last two words");
        // parallel_ranges: the parts tile the range, on unit boundaries
        for (unsigned workers : {1u, 2u, 5u}) {
            const WorkerScope scope(workers);
            std::vector<uint8_t> hit(1000, 0);
            std::vector<size_t> begins(workers, 1);
            const unsigned parts = prep::detail::parallel_ranges(1000, 7, 50, [&](size_t b, size_t e, unsigned part) {
                begins[part] = b;
                for (size_t i = b; i < e; i++) hit[i]++;
            });
            bool tiles = parts >= 1 && parts <= workers;
            for (uint8_t h : hit) tiles = tiles && h == 1;
            for (size_t b : begins) tiles = tiles && (b % 7 == 0 || b == 1);
            check(tiles, "parallel_ranges tiles the range (" + std::to_string(workers) + " workers)");
            // parallel_chunks: every index once, chunk boundaries on unit multiples, worker ids in range
            std::vector<std::atomic<int>> hits(1000);
            for (auto& h : hits) h.store(0);
            std::atomic<int> bad(0);
            const unsigned used = prep::detail::parallel_chunks(1000, 7, 21, [&](size_t b, size_t e, unsigned w) {
                if (b % 7 != 0 || w >= workers || e > 1000 || b >= e) bad++;
                for (size_t i = b; i < e; i++) hits[i]++;
            });
            bool once = used >= 1 && used <= workers && bad.load() == 0;
            for (auto& h : hits) once = once && h.load() == 1;
            check(once, "parallel_chunks covers the range exactly once (" + std::to_string(workers) + " workers)");
        }
        {
            const WorkerScope scope(4);
            check(throws<std::runtime_error>([] {
                      prep::detail::parallel_ranges(1000, 1, 1, [](size_t b, size_t, unsigned) {
                          if (b > 0) throw std::runtime_error("part failed");
                      });
                  }),
                  "parallel_ranges rethrows an exception of a worker");
            check(throws<std::runtime_error>([] {
                      prep::detail::parallel_chunks(1000, 1, 1, [](size_t b, size_t, unsigned) {
                          if (b >= 500) throw std::runtime_error("chunk failed");
                      });
                  }),
                  "parallel_chunks rethrows an exception of a worker");
        }
        check(prep::detail::worker_count() == 1, "WorkerScope restores the previous setting");
    }

    // The bucketing permutation against a sort of the full keys.
    {
        const hashes::seed16 seed = g_rng.seed();
        for (size_t n : {(size_t)0, (size_t)1, (size_t)2, (size_t)1000, (size_t)1537, (size_t)1 << 20}) {
            if (n > 100000 && !g_large) continue;
            check(bucket_permutation(seed, n) == bucket_permutation_ref(seed, n),
                  "bucket_permutation equals the argsort of the full keys, n = " + std::to_string(n));
        }
        // ties in the top 32 key bits do occur at n = 2^20 (about n^2 / 2^33 = 128 pairs)
        if (g_large) {
            const GfVec keys = hashes::ctr_prg_gf128(seed, prg_dom(USE_BUCKET_PERM), (size_t)1 << 20);
            std::vector<uint32_t> top(keys.size());
            for (size_t i = 0; i < keys.size(); i++) top[i] = (uint32_t)(keys[i].hi >> 32);
            std::sort(top.begin(), top.end());
            size_t ties = 0;
            for (size_t i = 1; i < top.size(); i++) ties += top[i] == top[i - 1];
            check(ties > 0, "the n = 2^20 comparison exercised ties in the top 32 key bits (" + std::to_string(ties) + ")");
        }
        {
            const WorkerScope scope(4);
            check(bucket_permutation(seed, 300000) == bucket_permutation_ref(seed, 300000),
                  "bucket_permutation with 4 workers equals the reference, n = 300000");
        }
        const gf128 k5 = prep::detail::prg_gf128_at(seed, prg_dom(USE_BUCKET_PERM), 5);
        check(k5 == hashes::ctr_prg_gf128(seed, prg_dom(USE_BUCKET_PERM), 6)[5], "random access to one sort key");
    }
}

// ------------------------------------------------------------ shares
static void test_shares_model() {
    Section sec("shares: data model, conversions, wire encoding");
    gf128 d[2];
    test_deltas(d);
    const gf128 delta = d[0] ^ d[1];
    const size_t n = 257;

    const BitVec x = rand_bits(n), y = rand_bits(n);
    Shares x0, x1, y0, y1;
    fresh(x, delta, x0, x1);
    fresh(y, delta, y0, y1);
    check(x0.size() == n && mac_ok(x0, x1, delta, &x), "fresh sharing satisfies the invariant");
    const BitVec xy = xor_bits(x, y);
    check(mac_ok(x0 ^ y0, x1 ^ y1, delta, &xy), "XOR is local");
    const BitVec xs(x.begin() + 10, x.begin() + 50);
    check(mac_ok(x0.slice(10, 50), x1.slice(10, 50), delta, &xs), "slicing");
    const std::vector<size_t> idx = {5, 5, 0, 256, 31};
    BitVec xi;
    for (size_t i : idx) xi.push_back(x[i]);
    check(mac_ok(x0.take(idx), x1.take(idx), delta, &xi), "take (with repetitions)");
    Shares cat0 = Shares::concat({x0, ShareView(y0.bits.data(), y0.macs.data(), 3)});
    Shares cat1 = Shares::concat({x1, ShareView(y1.bits.data(), y1.macs.data(), 3)});
    BitVec xc = x;
    xc.insert(xc.end(), y.begin(), y.begin() + 3);
    check(cat0.size() == n + 3 && mac_ok(cat0, cat1, delta, &xc), "concat");
    check(Shares::concat({}).size() == 0 && Shares::zeros(4).size() == 4, "empty concat and zeros");
    const BitVec z9(9, 0);
    check(mac_ok(Shares::zeros(9), Shares::zeros(9), delta, &z9), "zero shares are valid");

    const BitVec c = rand_bits(n);
    const Shares c0 = Shares::constant(0, d[0], c), c1 = Shares::constant(1, d[1], c);
    check(mac_ok(c0, c1, delta, &c), "constant: (c, c Delta_0) and (0, c Delta_1)");
    check(c0.bits == c && mean(c1.bits) == 0.0, "constant: P_0 holds the bit, P_1 zero");
    const BitVec xc2 = xor_bits(x, c);
    check(mac_ok(x0.add_const(0, d[0], c), x1.add_const(1, d[1], c), delta, &xc2), "add_const");
    const BitVec xn = xor_bits(x, BitVec(n, 1));
    check(mac_ok(x0.invert(0, d[0]), x1.invert(1, d[1]), delta, &xn), "invert");
    const BitVec xa = and_bits(x, c);
    check(mac_ok(x0.mul_const(c), x1.mul_const(c), delta, &xa), "mul_const");

    const BitVec r = rand_bits(128);
    Shares r0, r1;
    fresh(r, delta, r0, r1);
    const std::pair<gf128, gf128> f0 = r0.to_field(), f1 = r1.to_field();
    check((f0.first ^ f1.first) == gf_pack_bits(r.data()), "to_field: value shares XOR to sum_j X^j r_j");
    check((f0.second ^ f1.second) == gf128_mul(gf_pack_bits(r.data()), delta), "to_field: MAC shares XOR to R Delta");
    check(to_field(ShareView(r0)) == f0, "to_field of a view");
    check(throws<std::invalid_argument>([&] { x0.to_field(); }), "to_field needs exactly 128 bits");

    check(throws<std::invalid_argument>([] { Shares(BitVec{0, 2}, GfVec(2)); }), "Shares rejects non-bits");
    check(throws<std::invalid_argument>([] { Shares(BitVec(3, 0), GfVec(2)); }), "Shares rejects mismatched lengths");
    check(throws<std::invalid_argument>([&] { Shares t = x0; t ^= y0.slice(0, 5); }), "XOR rejects mismatched lengths");
    check(throws<std::invalid_argument>([&] { Shares::constant(2, d[0], c); }), "constant rejects a bad role");
    check(throws<std::invalid_argument>([&] { x0.add_const(0, d[0], BitVec(5, 0)); }), "add_const rejects a bad length");
    check(throws<std::out_of_range>([&] { x0.slice(5, n + 1); }) && throws<std::out_of_range>([&] { x0.take({n}); }),
          "slice and take check their ranges");

    // BDOZ containers and conversions (Section 2.2).
    const BitVec xb0 = rand_bits(n), xb1 = rand_bits(n);
    const GfVec k0 = rand_vec(n), k1 = rand_vec(n);  // k_b: keys of P_b for the peer's bits
    GfVec m0(n), m1(n);
    for (size_t i = 0; i < n; i++) {
        m0[i] = k1[i] ^ gf_and_bit(d[1], xb0[i]);
        m1[i] = k0[i] ^ gf_and_bit(d[0], xb1[i]);
    }
    const Bdoz own0(xb0, m0), own1(xb1, m1);
    const Shares s0 = spdz_shares(0, d[0], own0, Keys(k0)), s1 = spdz_shares(1, d[1], own1, Keys(k1));
    const BitVec xb = xor_bits(xb0, xb1);
    check(mac_ok(s0, s1, delta, &xb), "spdz_shares: m_b = M[x_b] ^ K[x_1-b] ^ x_b Delta_b");
    GfVec inplace = m0;
    spdz_macs_inplace(d[0], xb0.data(), inplace.data(), k0.data(), n);
    check(inplace == s0.macs, "spdz_macs_inplace equals spdz_shares");
    check(mac_ok(mask_owner(d[0], own0), mask_other(Keys(k1)), delta, &xb0), "input mask of P_0");
    check(mac_ok(mask_other(Keys(k0)), mask_owner(d[1], own1), delta, &xb1), "input mask of P_1");
    Bdoz both = own0;
    both.append(own1);
    Keys kk(k0);
    kk.append(Keys(k1).slice(0, 1));
    check(own0.slice(3, 9).size() == 6 && both.size() == 2 * n && kk.size() == n + 1, "Bdoz / Keys slicing and concat");
    check(bdoz_ok(own0.slice(5, 60), Keys(k1).slice(5, 60), d[1]), "BDOZ slices stay consistent");

    // Wire encoding of bit vectors.
    for (size_t m : {(size_t)0, (size_t)1, (size_t)7, (size_t)8, (size_t)9, (size_t)257}) {
        const BitVec bits = rand_bits(m);
        check(unpack_bit_msg(pack_bit_msg(bits), m) == bits && pack_bit_msg(bits).size() == (m + 7) / 8,
              "bit message, n = " + std::to_string(m));
    }
    check(throws<ProtocolAbort>([] { unpack_bit_msg(bytes{0, 0}, 9 + 8); }), "short bit message rejected");
    check(throws<ProtocolAbort>([] { unpack_bit_msg(bytes{0xff, 0x03}, 9); }), "stray padding bits rejected");
    check(unpack_bit_msg(bytes{0xff, 0x01}, 9) == BitVec(9, 1), "full bit message accepted");
    check(pack_bit_msg(BitVec{1, 0, 0, 0, 0, 0, 0, 0, 1}) == bytes({0x01, 0x01}), "bit i goes to bit (i mod 8) of byte i / 8");

    // Key shares.
    CtrPrgRng ra((uint64_t)7), rb((uint64_t)7);
    gf128 first = rb.gf();
    first.lo |= 1;
    check(sample_delta(0, ra) == first, "sample_delta is the first 16 bytes of the source with the lsb fixed");
    CtrPrgRng rc((uint64_t)9);
    check((sample_delta(0, rc).lo & 1) == 1 && (sample_delta(1, rc).lo & 1) == 0, "sample_delta fixes the lsb");
    check(throws<std::invalid_argument>([] { check_delta(0, gf128(2, 0)); }) &&
              throws<std::invalid_argument>([] { check_delta(1, gf128(3, 0)); }) && check_delta(1, gf128(6, 0)) == gf128(6, 0),
          "check_delta enforces lsb(Delta_0) = 1, lsb(Delta_1) = 0");
    check(required_lsb(0) == 1 && required_lsb(1) == 0, "required_lsb");
}

// ------------------------------------------------------------ vectors from the Python implementation
static void test_vectors() {
    Section sec("vectors: values produced by the Python implementation");
    check(h_dom(TSID, 0) == 0x4806c47f0365365bull && h_dom(TSID, 1) == 0x481d56f683ac347full, "leaky-AND hash domains");
    check(hex(RealPrep::prep_sid(TSID)) == "b52107b39efcb112cf4a933a6bca461c5225849b66ae0d3ba7bf5bafe7792f03",
          "identifier of the preprocessing sub-session");
    check(prg_dom(USE_COT_PRG, 1, 5) == 0x5010010000000005ull && prg_dom(USE_TRIPLE_CHI) == 0x5014000000000000ull,
          "prg_dom layout");
    check(hex(counter_label("bucket|", 0x0102)) == "6275636b65747c0201000000000000" &&
              hex(bool_check_label(3)) == "626f6f6c636865636b7c0300000000000000",
          "labels: prefix || le64(counter)");

    hashes::seed16 seed;
    for (size_t i = 0; i < 16; i++) seed[i] = (uint8_t)i;
    const std::vector<uint32_t> expect20 = {11, 7, 17, 19, 1, 10, 6, 15, 8, 9, 16, 2, 12, 13, 0, 18, 14, 5, 4, 3};
    check(bucket_permutation(seed, 20) == expect20, "bucket_permutation(seed, 20)");
    {
        const std::vector<uint32_t> perm = bucket_permutation(seed, 100000);
        bytes raw(4 * perm.size());
        for (size_t i = 0; i < perm.size(); i++)
            for (int k = 0; k < 4; k++) raw[4 * i + (size_t)k] = (uint8_t)(perm[i] >> (8 * k));
        check(sha_hex(raw) == "bfbbc92829f5526cebd53c2c65e602d58743597eecf19d9829c6f9eaab2fa041",
              "bucket_permutation(seed, 100000)");
    }

    // BoolCheck digests.
    {
        gf128 d0 = tvec(1, 1)[0];
        d0.lo |= 1;
        const BitVec opened = tbits(2, 50);
        const GfVec m0 = tvec(3, 50);
        const gf128 w = tvec(4, 1)[0], big_m = tvec(5, 1)[0];
        check(d0.hex() == "27c632aaea0dfd7308c988bf6bcc5b57", "test data streams agree with the Python script");
        check(hex(check_digest(TSID, 3, d0, opened, m0, {FieldOpening(w, big_m)})) ==
                  "a20254669c7c8bda88a32d99ac4982d1abad1fd9a14b055d78a8bda115e5b5ea",
              "BoolCheck digest with a field opening");
        check(hex(check_digest(TSID, 7, d0, opened, m0)) == "9d20091b99c40701c5df7b7443d6667c38b937f8818c1b93fd04a487f6d9334e",
              "BoolCheck digest of bits only");
    }

    // Base-OT derivations.
    {
        ed25519::scalar32 five{};
        five[0] = 5;
        const ed25519::Point p5 = ed25519::mul_base(five);
        const ed25519::bytes32 enc5 = ed25519::compress(p5);
        check(hex(prep::detail::ot_h_tag(TSID, 1, 7, 0)) == "6d72010007000000746573742d707265702d73657373696f6e", "base OT: tag of H_j");
        uint8_t k[16];
        prep::detail::ot_seed(TSID, 1, 7, 0, p5).to_bytes(k);
        check(hex(k, 16) == "5bddc556fb6abd4a95e49b574b6888f7", "base OT: seed derivation");
        check(hex(ed25519::compress(prep::detail::ot_hash_point(TSID, 1, 7, 0, enc5.data()))) ==
                  "442c29f8b4c6b352ddd0ef1f3e02db47b225373213b6de23431a8e62b0d16211",
              "base OT: H_j(P)");
    }

    // An OT-extension transcript with fixed seeds and fixed randomness (two calls).
    {
        const GfVec s0 = tvec(10, 128), s1 = tvec(11, 128);
        const gf128 delta = tvec(12, 1)[0];
        GfVec chosen(128);
        for (unsigned j = 0; j < 128; j++) chosen[j] = delta.bit(j) ? s1[j] : s0[j];
        CtrPrgRng r_rcv(tag_seed(20), hashes::dom::PRG_TEST), r_snd(tag_seed(21), hashes::dom::PRG_TEST);
        CotReceiver rcv(1, s0, s1, r_rcv);
        CotSender snd(1, delta, chosen, r_snd);
        const size_t ell = 1000;
        const bytes u_msg = rcv.start(ell);
        const bytes sd = snd.absorb(ell, u_msg);
        Bdoz own(ell);
        Keys keys(ell);
        const bytes msg = rcv.answer(sd, own.bits.data(), own.macs.data());
        snd.verify(msg, keys.keys.data());
        check(delta.hex() == "0989fe6f6b43bedb19762f27ed78d9c9", "COT vector: key");
        check(sha_hex(u_msg) == "09944709bad517459385554ef1f022d46849dd7afe5e682c9f70ade18d7e156b", "COT vector: U message");
        check(hex(sd) == "b1077173fefe5cbadf3ea6e82310f68d", "COT vector: challenge seed");
        check(sha_hex(msg) == "43d2055fe3027da6b58299de27d487ae44c427c885e5826e196402850b4b1afd", "COT vector: check message");
        check(sha_hex(own.bits) == "ee36b6053ef9b82f8e3971cea70e992a4dbac44c5dfd29cc3a28602d9bca64a1", "COT vector: u bits");
        check(sha_hex(gf_to_bytes(own.macs)) == "935d956de091233c171ec2d96d701858a86d1e212a82581d6b48259bc73897c4", "COT vector: rows t_i");
        check(sha_hex(gf_to_bytes(keys.keys)) == "9dbc019c642d45eca0c1f8c40335a7361c4f65f80750c0f07ff0fb6117616ce5", "COT vector: rows q_i");
        check(bdoz_ok(own, keys, delta), "COT vector: q_i = t_i XOR u_i Delta");
        const bytes u2 = rcv.start(130);
        const bytes sd2 = snd.absorb(130, u2);
        Bdoz own2(130);
        Keys keys2(130);
        const bytes msg2 = rcv.answer(sd2, own2.bits.data(), own2.macs.data());
        snd.verify(msg2, keys2.keys.data());
        check(sha_hex(u2) == "33d288881cfa9ca50b64e6415267817e1b0d1bc5a46f6fc73488e0436af60a45" &&
                  sha_hex(msg2) == "0734ba951105ff753c048ad1b6915d746a945f36da568876e0471c5857cc46c4",
              "COT vector: second call (fresh domain)");
        check(rcv.calls() == 2 && snd.calls() == 2 && rcv.rows() == 1130 && snd.rows() == 1130, "COT counters");
    }

    // The transfer message of the leaky AND for fixed inputs.
    {
        const size_t n = 37;
        const GfVec xk = tvec(30, n), ym = tvec(31, n), xm = tvec(32, n);
        const BitVec xb = tbits(33, n);
        gf128 dl = tvec(34, 1)[0];
        dl.lo |= 1;
        auto f0 = [&](Channel& ch) {
            Shares z;
            leaky_ands(ch, TSID, 0, dl, 1000, n, xb.data(), xm.data(), xk.data(), ym.data(), z);
            return z;
        };
        auto f1 = [&](Channel& ch) {
            const bytes msg = ch.recv();
            ch.send(bytes(16 * n, 0));
            return msg;
        };
        const auto res = run_two_party(f0, f1);
        check(res.first.ok && res.second.ok &&
                  sha_hex(res.second.value) == "ff90e21c114d2e328009bd1eeef878cc5130d231e6e9700a98cbd62320f0983e",
              "leaky AND vector: transfer message G_0");
    }

    // The linear combinations of the triple check.
    {
        const size_t n = 37;
        const GfVec chi = hashes::ctr_prg_gf128(seed, prg_dom(USE_TRIPLE_CHI), n);
        const BitVec zb = tbits(35, n);
        const GfVec zm = tvec(36, n);
        check(gf_dot_bits(chi.data(), zb.data(), n).hex() == "609d3af303990d096841755403c6b9d5", "triple check vector: sum chi_i z_i");
        check(gf_dot(chi.data(), zm.data(), n).hex() == "d9819f08442f17dc5f953581286866da", "triple check vector: sum chi_i n_i");
        const std::pair<gf128, gf128> rf = Shares(tbits(37, 128), tvec(38, 128)).to_field();
        check(rf.first.hex() == "7a2b4f2f9c9eca2873a1acadfd245ca3" && rf.second.hex() == "eb3a0cff03ee0e6889d0e8632ba1f2bf",
              "triple check vector: <R> = sum_j X^j <r_j>");
    }
}

// ------------------------------------------------------------ (b) base OT
typedef std::pair<PartyOutcome<BaseOtResult>, PartyOutcome<BaseOtResult>> OtRun;

static OtRun base_ot_run(const gf128 deltas[2], const Cheat* cheat0 = nullptr, const Cheat* cheat1 = nullptr,
                         RandomSource* rng0 = nullptr, RandomSource* rng1 = nullptr) {
    auto party = [&](int role, const Cheat* cheat, RandomSource* rng) {
        return [deltas, role, cheat, rng](Channel& ch) {
            SystemRng sys;
            RandomSource& use = rng ? *rng : static_cast<RandomSource&>(sys);
            return base_ot(ch, TSID, role, gf_unpack_bits(deltas[role]), use, cheat);
        };
    };
    auto f0 = party(0, cheat0, rng0);
    auto f1 = party(1, cheat1, rng1);
    return run_two_party(f0, f1);
}

static Cheat::Rewrite put_blob(size_t offset, const bytes& blob) {
    return [offset, blob](const bytes& msg) {
        bytes out = msg;
        std::copy(blob.begin(), blob.end(), out.begin() + (ptrdiff_t)offset);
        return out;
    };
}

static void test_base_ot() {
    Section sec("base OT: Masny-Rindal over Edwards25519, both directions");
    const int runs = 20;
    const double t_begin = now_s();
    gf128 deltas[2];
    for (int run = 0; run < runs; run++) {
        deltas[0] = g_rng.gf();
        deltas[1] = g_rng.gf();
        const OtRun res = base_ot_run(deltas);
        const std::string tag = "base OT run " + std::to_string(run);
        check(res.first.ok && res.second.ok, tag + ": both parties finish");
        if (!res.first.ok || !res.second.ok) continue;
        const BaseOtResult* r[2] = {&res.first.value, &res.second.value};
        for (int d = 0; d < 2; d++) {  // direction d: P_d holds the chosen seeds, P_{1-d} the pairs
            const GfVec& chosen = r[d]->chosen;
            const GfVec& s0 = r[1 - d]->seeds0;
            const GfVec& s1 = r[1 - d]->seeds1;
            const std::string t = tag + ", direction " + std::to_string(d);
            check(chosen.size() == KAPPA && s0.size() == KAPPA && s1.size() == KAPPA, t + ": sizes");
            bool got = true, differ = true, other = true;
            std::set<std::pair<uint64_t, uint64_t>> all;
            for (unsigned j = 0; j < KAPPA; j++) {
                const unsigned c = deltas[d].bit(j);
                got = got && chosen[j] == (c ? s1[j] : s0[j]);
                differ = differ && s0[j] != s1[j];
                other = other && chosen[j] != (c ? s0[j] : s1[j]);
                all.insert(std::make_pair(s0[j].lo, s0[j].hi));
                all.insert(std::make_pair(s1[j].lo, s1[j].hi));
            }
            check(got, t + ": the receiver obtains the chosen seeds");
            check(differ, t + ": the two seeds of an instance differ");
            check(other, t + ": the other seed is not the chosen one");
            check(all.size() == 2 * KAPPA, t + ": all 256 seeds are distinct");
        }
        check(r[0]->seeds0 != r[1]->seeds0, tag + ": directions are independent");
        check(res.first.stats.rounds == 2 && res.second.stats.rounds == 2, tag + ": two rounds");
        check(res.first.stats.bytes_sent == 96 * KAPPA && res.second.stats.bytes_sent == 96 * KAPPA,
              tag + ": 96 bytes per instance and party");
    }
    std::printf("  base OT: %d runs, %.3f s per run (2 x %d instances, both parties in one process)\n", runs,
                (now_s() - t_begin) / runs, (int)KAPPA);

    // Determinism under seeded randomness.
    deltas[0] = g_rng.gf();
    deltas[1] = g_rng.gf();
    {
        CtrPrgRng a0((uint64_t)1), a1((uint64_t)2), b0((uint64_t)1), b1((uint64_t)2);
        const OtRun a = base_ot_run(deltas, nullptr, nullptr, &a0, &a1);
        const OtRun b = base_ot_run(deltas, nullptr, nullptr, &b0, &b1);
        check(a.first.ok && b.first.ok && a.first.value.chosen == b.first.value.chosen &&
                  a.second.value.seeds1 == b.second.value.seeds1,
              "base OT is reproducible with seeded randomness");
    }

    // Deviations: every received point must be a non-identity element of the subgroup.
    const ed25519::bytes32 id_enc = ed25519::compress(ed25519::identity());
    const bytes identity(id_enc.begin(), id_enc.end());
    bytes order2(32, 0xff);  // y = p - 1, x = 0: the point (0, -1)
    order2[0] = 0xec;
    order2[31] = 0x7f;
    const bytes order4(32, 0);  // y = 0: a point of order 4
    bytes not_on_curve;
    for (unsigned i = 0; i < 50 && not_on_curve.empty(); i++) {
        bytes cand(32, 0);
        cand[0] = (uint8_t)(2 + i);
        ed25519::Point tmp;
        if (!ed25519::decompress(tmp, cand.data())) not_on_curve = cand;
    }
    ed25519::Point p2, p4;
    const bool dec_ok = ed25519::decompress(p2, order2.data()) && ed25519::decompress(p4, order4.data());
    ed25519::scalar32 sc{};
    sc[0] = 0x39;
    sc[1] = 0x30;  // 12345
    const ed25519::bytes32 mixed_enc = ed25519::compress(ed25519::add(ed25519::mul_base(sc), p2));  // order 2q
    const bytes mixed(mixed_enc.begin(), mixed_enc.end());
    ed25519::Point pm;
    check(dec_ok && !not_on_curve.empty() && ed25519::is_identity(ed25519::dbl(p2)) && !ed25519::is_identity(p2) &&
              ed25519::is_identity(ed25519::dbl(ed25519::dbl(p4))) && !ed25519::is_identity(ed25519::dbl(p4)) &&
              ed25519::decompress(pm, mixed.data()) && !ed25519::in_prime_subgroup(pm),
          "test points are as intended");
    const std::pair<const char*, const bytes*> bad_points[5] = {{"identity", &identity},
                                                               {"order-2 point", &order2},
                                                               {"order-4 point", &order4},
                                                               {"non-point", &not_on_curve},
                                                               {"point outside the subgroup", &mixed}};
    for (const auto& bp : bad_points) {
        const bytes& blob = *bp.second;
        check(throws<ProtocolAbort>([&] { prep::detail::ot_decode(blob.data()); }),
              std::string("base OT decoding rejects a ") + bp.first);
    }
    {
        ed25519::scalar32 five{};
        five[0] = 5;
        const ed25519::bytes32 e5 = ed25519::compress(ed25519::mul_base(five));
        check(ed25519::equal(prep::detail::ot_decode(e5.data()), ed25519::mul_base(five)),
              "base OT decoding accepts a subgroup element");
    }
    {
        // a bad point in the flight of P_0: the honest P_1 must abort
        struct Case {
            int point;
            bool second;
            size_t offset;
        };
        const Case cases[4] = {{0, false, 64 * 5}, {4, false, 64 * 9 + 32}, {3, true, 32 * 7}, {2, true, 32 * 127}};
        for (const Case& c : cases) {
            Cheat cheat;
            (c.second ? cheat.flight2 : cheat.flight1) = put_blob(c.offset, *bad_points[c.point].second);
            const OtRun res = base_ot_run(deltas, &cheat);
            // (after a bad second flight P_0 itself may finish: it has received everything it needs)
            check(res.second.protocol_abort && res.second.error.find("invalid point") != std::string::npos &&
                      (c.second || !res.first.ok),
                  std::string("base OT aborts on a ") + bad_points[c.point].first + (c.second ? " in flight 2" : " in flight 1"));
        }
        Cheat shorter;
        shorter.flight1 = [](const bytes& m) { return bytes(m.begin(), m.end() - 1); };
        check(base_ot_run(deltas, &shorter).second.protocol_abort, "base OT aborts on a short first flight");
        Cheat longer;
        longer.flight2 = [](const bytes& m) {
            bytes o = m;
            o.push_back(0);
            return o;
        };
        check(base_ot_run(deltas, nullptr, &longer).first.protocol_abort, "base OT aborts on a long second flight");
    }
    {
        // A sender that replaces A by another valid point only desynchronises the seeds
        // (no abort here); the OT-extension check catches it (see the abort tests).
        ed25519::scalar32 s777{};
        s777[0] = 0x09;
        s777[1] = 0x03;
        const ed25519::bytes32 other = ed25519::compress(ed25519::mul_base(s777));
        Cheat cheat;
        cheat.flight2 = put_blob(32 * 3, bytes(other.begin(), other.end()));
        const OtRun res = base_ot_run(deltas, &cheat);
        size_t agree = 0;
        bool third = true;
        if (res.first.ok && res.second.ok) {
            for (unsigned j = 0; j < KAPPA; j++) {  // direction 1: P_0 sends
                const bool same = res.second.value.chosen[j] ==
                                  (deltas[1].bit(j) ? res.first.value.seeds1[j] : res.first.value.seeds0[j]);
                agree += same;
                if (j == 3) third = same;
            }
        }
        check(res.first.ok && res.second.ok && !third && agree == KAPPA - 1, "a replaced A corrupts exactly that instance");
    }
    check(throws<std::invalid_argument>([&] {
              SystemRng sys;
              auto pr = LoopbackChannel::make_pair();
              base_ot(*pr.first, TSID, 0, BitVec(5, 0), sys);
          }),
          "base_ot requires KAPPA choice bits");
}

// ------------------------------------------------------------ (c) correlated OT extension
struct CotDirOut {
    std::vector<Keys> keys;  // filled by the sender
    std::vector<Bdoz> own;   // filled by the receiver
    ChannelStats st;         // statistics of the endpoint the party used (the proxy)
};
typedef std::map<size_t, TamperChannel::Rule> Rules;

// Run extend() for every ell in one direction; returns (sender outcome, receiver outcome).
static std::pair<PartyOutcome<CotDirOut>, PartyOutcome<CotDirOut>> cot_one_direction(
    int direction, const gf128& delta, const TrustedSeeds& seeds, const std::vector<size_t>& ells,
    const Cheat* cheat = nullptr, const Rules& sender_tamper = Rules(), const Rules& receiver_tamper = Rules()) {
    std::function<CotDirOut(Channel&)> sender_fn = [&](Channel& raw) {
        TamperChannel ch(raw, sender_tamper);
        SystemRng rng;
        CotSender snd(direction, delta, seeds.chosen, rng);
        CotDirOut out;
        for (size_t ell : ells) out.keys.push_back(snd.extend(ch, ell));
        out.st = ch.stats();
        return out;
    };
    std::function<CotDirOut(Channel&)> receiver_fn = [&](Channel& raw) {
        TamperChannel ch(raw, receiver_tamper);
        SystemRng rng;
        CotReceiver rcv(direction, seeds.s0, seeds.s1, rng);
        CotDirOut out;
        for (size_t i = 0; i < ells.size(); i++) out.own.push_back(rcv.extend(ch, ells[i], i == 0 ? cheat : nullptr));
        out.st = ch.stats();
        return out;
    };
    if (direction == 0) return run_two_party(sender_fn, receiver_fn);
    auto res = run_two_party(receiver_fn, sender_fn);
    return std::make_pair(std::move(res.second), std::move(res.first));
}

static void test_cot() {
    Section sec("COT extension: correlation, consistency check, deviations");

    // Honest runs, one direction at a time, two calls per state.
    for (int direction = 0; direction < 2; direction++) {
        for (size_t ell : {(size_t)1000, (size_t)100000}) {
            const gf128 delta = g_rng.gf();
            const TrustedSeeds seeds = trusted_seeds(delta);
            const double t0 = now_s();
            const auto res = cot_one_direction(direction, delta, seeds, {ell, 777});
            const double dt = now_s() - t0;
            const std::string tag = "COT direction " + std::to_string(direction) + ", ell = " + std::to_string(ell);
            check(res.first.ok && res.second.ok, tag + ": both parties finish");
            if (!res.first.ok || !res.second.ok) continue;
            const std::vector<Keys>& keys = res.first.value.keys;
            const std::vector<Bdoz>& own = res.second.value.own;
            check(own[0].size() == ell && keys[0].size() == ell && own[1].size() == 777, tag + ": sizes");
            check(bdoz_ok(own[0], keys[0], delta), tag + ": q_i = t_i XOR u_i Delta for all rows");
            check(bdoz_ok(own[1], keys[1], delta), tag + ": second call correlates as well");
            check(std::fabs(mean(own[0].bits) - 0.5) < 5 * 0.5 / std::sqrt((double)ell), tag + ": the choice bits look uniform");
            check(!std::equal(own[1].macs.begin(), own[1].macs.end(), own[0].macs.begin()),
                  tag + ": the second call reads a fresh PRG stream");
            std::set<std::pair<uint64_t, uint64_t>> distinct;
            const size_t lim = std::min(ell, (size_t)2000);
            for (size_t i = 0; i < lim; i++) distinct.insert(std::make_pair(own[0].macs[i].lo, own[0].macs[i].hi));
            check(distinct.size() == lim, tag + ": rows differ");
            const uint64_t expect = u_message_bytes(ell) + u_message_bytes(777) + 2 * COT_CHECK_BYTES;
            check(res.second.stats.bytes_sent == expect && res.first.stats.bytes_sent == 32,
                  tag + ": bytes = U + check for the receiver, two seeds for the sender");
            check(res.first.value.st.rounds == 6 && res.second.value.st.rounds == 6 &&
                      res.first.value.st.bytes_sent == 32 && res.second.value.st.bytes_recv == 32,
                  tag + ": three flights per call");
            if (ell == 100000)
                std::printf("  COT one direction, ell = %zu (+777): %.3f s, %.2f bytes per row\n", ell, dt,
                            (double)res.second.stats.bytes_sent / (double)(ell + 777));
        }
    }
    check(rounded_rows(1) == 64 && rounded_rows(64) == 64 && rounded_rows(65) == 128 &&
              u_message_bytes(1000) == 16 * (1024 + PAD),
          "row rounding and U size");

    // Both directions in parallel.
    for (size_t ell : {(size_t)1000, (size_t)100000}) {
        gf128 deltas[2] = {g_rng.gf(), g_rng.gf()};
        const TrustedSeeds seeds[2] = {trusted_seeds(deltas[0]), trusted_seeds(deltas[1])};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                SystemRng rng;
                CotSender snd(role, deltas[role], seeds[role].chosen, rng);
                CotReceiver rcv(1 - role, seeds[1 - role].s0, seeds[1 - role].s1, rng);
                return extend_both(ch, snd, rcv, ell);
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        const auto res = run_two_party(f0, f1);
        const std::string tag = "COT both directions, ell = " + std::to_string(ell);
        check(res.first.ok && res.second.ok && bdoz_ok(res.first.value.own, res.second.value.keys, deltas[1]) &&
                  bdoz_ok(res.second.value.own, res.first.value.keys, deltas[0]),
              tag + ": both correlations hold");
        check(res.first.stats.rounds == 3 && res.second.stats.rounds == 3, tag + ": 3 rounds");
        check(res.first.stats.bytes_sent == u_message_bytes(ell) + 16 + COT_CHECK_BYTES, tag + ": bytes per party");
    }

    // Malicious receiver: an inconsistent u in ONE column j (one flipped bit of U_j).
    // The sender reads U_j only if Delta_j = 1: the check rejects exactly then.
    const size_t ell = 500;
    const size_t n_rows = rounded_rows(ell) + PAD;
    {
        const int trials = 60;
        int n_abort = 0, n_wrong = 0;
        for (int t = 0; t < trials; t++) {
            const gf128 delta = g_rng.gf();
            const TrustedSeeds seeds = trusted_seeds(delta);
            const unsigned col = (unsigned)rand_below(KAPPA);
            Cheat cheat;
            cheat.flip_u.push_back(std::make_pair(col, rand_below(n_rows)));
            const auto res = cot_one_direction((int)rand_below(2), delta, seeds, {ell}, &cheat);
            const bool aborted = res.first.protocol_abort;
            n_abort += aborted;
            if (aborted != (delta.bit(col) == 1)) n_wrong++;
            else if (!aborted && !(res.first.ok && res.second.ok && bdoz_ok(res.second.value.own[0], res.first.value.keys[0], delta)))
                n_wrong++;
        }
        check(n_wrong == 0, "flipped U bit: abort iff Delta_j = 1; otherwise the output is intact");
        check(n_abort >= 10 && n_abort <= 50, "flipped U bit: about half of the columns are read (" + std::to_string(n_abort) + "/60)");
        std::printf("  COT, one inconsistent column: rejected in %d/%d runs (exactly the runs with Delta_j = 1)\n", n_abort, trials);
    }
    {
        int rejected = 0;
        for (int t = 0; t < 20; t++) {
            gf128 delta = g_rng.gf();
            delta.hi |= 1ull << (77 - 64);
            Cheat cheat;
            cheat.flip_u.push_back(std::make_pair(77u, rand_below(n_rows)));
            rejected += cot_one_direction((int)rand_below(2), delta, trusted_seeds(delta), {ell}, &cheat).first.protocol_abort;
        }
        check(rejected == 20, "flipped U bit in a column with Delta_j = 1: rejected in 20/20 runs");
        rejected = 0;
        for (int t = 0; t < 10; t++) {  // survives only if Delta is 0 on all 64 columns (2^-64)
            const gf128 delta = g_rng.gf();
            Cheat cheat;
            std::vector<unsigned> cols(KAPPA);
            for (unsigned j = 0; j < KAPPA; j++) cols[j] = j;
            for (unsigned j = 0; j < 64; j++) {
                std::swap(cols[j], cols[j + rand_below(KAPPA - j)]);
                cheat.flip_u.push_back(std::make_pair(cols[j], rand_below(n_rows)));
            }
            rejected += cot_one_direction(0, delta, trusted_seeds(delta), {ell}, &cheat).first.protocol_abort;
        }
        check(rejected == 10, "inconsistent u in 64 columns: rejected in 10/10 runs");
    }

    // Tampered check message and malformed messages.
    {
        gf128 delta = g_rng.gf();
        delta.lo |= 1;
        const TrustedSeeds seeds = trusted_seeds(delta);
        auto sender_aborts = [&](const Rules& receiver_rules, const char* text) {
            const auto res = cot_one_direction(0, delta, seeds, {ell}, nullptr, Rules(), receiver_rules);
            return res.first.protocol_abort && res.first.error.find(text) != std::string::npos;
        };
        check(sender_aborts(Rules{{1, flip_bit_rule(3)}}, "consistency check failed"), "COT: a wrong u~ is rejected");
        check(sender_aborts(Rules{{1, flip_bit_rule(8 * 16 * 40)}}, "consistency check failed"), "COT: a wrong V~_j is rejected");
        check(sender_aborts(Rules{{0, truncate_rule(8)}}, "U message"), "COT: a short U message is rejected");
        check(sender_aborts(Rules{{1, append_rule(1)}}, "check message"), "COT: a long check message is rejected");
        const auto res = cot_one_direction(0, delta, seeds, {ell}, nullptr, Rules{{0, truncate_rule(8)}});
        check(res.second.protocol_abort && res.second.error.find("challenge seed") != std::string::npos,
              "COT: a short challenge seed is rejected by the receiver");
        const auto ok = cot_one_direction(0, delta, seeds, {ell});
        check(ok.first.ok && ok.second.ok && bdoz_ok(ok.second.value.own[0], ok.first.value.keys[0], delta),
              "COT: honest run after the deviations");

        // Argument validation.
        SystemRng rng;
        CotSender snd(0, delta, seeds.chosen, rng);
        CotReceiver rcv(0, seeds.s0, seeds.s1, rng);
        check(throws<std::invalid_argument>([&] { rcv.start(0); }) && throws<std::invalid_argument>([&] { snd.absorb(0, bytes()); }),
              "extend rejects ell = 0");
        check(throws<std::invalid_argument>([&] { CotSender bad(2, delta, seeds.chosen, rng); }) &&
                  throws<std::invalid_argument>([&] { CotReceiver bad(0, GfVec(5), seeds.s1, rng); }),
              "constructors validate direction and seeds");
        check(throws<std::logic_error>([&] { rcv.answer(bytes(16), nullptr, nullptr); }) &&
                  throws<std::logic_error>([&] { snd.verify(bytes(COT_CHECK_BYTES), nullptr); }),
              "answer / verify need a running extension");
    }
}

// ------------------------------------------------------------ (d) authenticated bits
struct AbitOut {
    Shares shares, more;
    BdozPair keep;
    InputMasks m0, m1;
    PrepStats st;
    uint64_t sender_calls = 0;
};

static void test_abits() {
    Section sec("abits: random authenticated bits and input masks");
    const size_t n = 100000, n_mask = 5000;
    gf128 d[2];
    test_deltas(d);
    const gf128 delta = d[0] ^ d[1];
    const double t0 = now_s();
    TrustedSeeds seeds[2] = {trusted_seeds(d[0]), trusted_seeds(d[1])};
    auto session = [&](size_t batch_rows, std::function<void(AbitSource&, AbitOut&)> body) {
        auto party = [&, batch_rows](int role) {
            return [&, role, batch_rows](Channel& ch) {
                SystemRng rng;
                PhaseMeter meter(ch);
                CotSender snd(role, d[role], seeds[role].chosen, rng);
                CotReceiver rcv(1 - role, seeds[1 - role].s0, seeds[1 - role].s1, rng);
                AbitSource src(ch, role, d[role], snd, rcv, batch_rows, &meter);
                AbitOut out;
                body(src, out);
                meter.mark(PH_NONE);
                out.st = meter.stats();
                out.sender_calls = snd.calls();
                return out;
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        return run_two_party(f0, f1);
    };

    {
        const auto res = session(DEFAULT_BATCH_ROWS, [&](AbitSource& src, AbitOut& out) {
            out.shares = src.random_shares(n, &out.keep);
            out.m0 = src.input_masks(0, n_mask);
            out.m1 = src.input_masks(1, n_mask + 3);
            out.more = src.random_shares(300);
        });
        const double dt = now_s() - t0;
        check(res.first.ok && res.second.ok, "abits: both parties finish");
        if (res.first.ok && res.second.ok) {
            const AbitOut& r0 = res.first.value;
            const AbitOut& r1 = res.second.value;
            check((delta.lo & 1) == 1, "lsb(Delta^B) = 1");
            check(r0.shares.size() == n && mac_ok(r0.shares, r1.shares, delta), "abits: m_0 ^ m_1 = x Delta for 100000 bits");
            check(bdoz_ok(r0.keep.own, r1.keep.keys, d[1]), "abits: M[x_0] = K[x_0] ^ x_0 Delta_1");
            check(bdoz_ok(r1.keep.own, r0.keep.keys, d[0]), "abits: M[x_1] = K[x_1] ^ x_1 Delta_0");
            check(r0.shares.bits == r0.keep.own.bits && r1.shares.bits == r1.keep.own.bits,
                  "abits: the share of x is the party's own BDOZ bit");
            const BitVec x = xor_bits(r0.shares.bits, r1.shares.bits);
            check(std::fabs(mean(x) - 0.5) < 5 * 0.5 / std::sqrt((double)n), "abits: x looks uniform");
            check(std::fabs(mean(r0.shares.bits) - 0.5) < 5 * 0.5 / std::sqrt((double)n), "abits: x_0 looks uniform");

            check(r0.m0.has_bits && !r1.m0.has_bits && r1.m0.bits.empty(), "input masks of P_0: only P_0 learns the bits");
            check(mac_ok(r0.m0.shares, r1.m0.shares, delta, &r0.m0.bits), "input masks of P_0: <r> authenticates r");
            check(r0.m0.shares.bits == r0.m0.bits && mean(r1.m0.shares.bits) == 0.0, "input masks of P_0: r_0 = r, r_1 = 0");
            check(!r0.m1.has_bits && r1.m1.has_bits && r1.m1.bits.size() == n_mask + 3,
                  "input masks of P_1: only P_1 learns the bits");
            check(mac_ok(r0.m1.shares, r1.m1.shares, delta, &r1.m1.bits) && mean(r0.m1.shares.bits) == 0.0, "input masks of P_1");
            check(std::fabs(mean(r0.m0.bits) - 0.5) < 5 * 0.5 / std::sqrt((double)n_mask), "mask bits look uniform");
            check(mac_ok(r0.more, r1.more, delta), "abits: later requests stay consistent");

            // Byte accounting: everything is OT extension traffic, nothing in "abit".
            const uint64_t per_party = u_message_bytes(n) + u_message_bytes(300) + 2 * (16 + COT_CHECK_BYTES);
            const uint64_t exp0 = per_party + (u_message_bytes(n_mask) + COT_CHECK_BYTES) + 16;  // P_0 receives for its masks
            const uint64_t exp1 = per_party + 16 + (u_message_bytes(n_mask + 3) + COT_CHECK_BYTES);
            check(r0.st[PH_COTEXT].bytes_sent == exp0 && res.first.stats.bytes_sent == exp0 &&
                      r1.st[PH_COTEXT].bytes_sent == exp1 && res.second.stats.bytes_sent == exp1,
                  "abits: bytes are exactly the U, seed and check messages");
            check(r0.st[PH_ABIT].bytes_sent == 0 && r0.st[PH_ABIT].rounds == 0, "abits: the conversion to shares is local");
            check(res.first.stats.rounds == res.second.stats.rounds && res.first.stats.rounds == 3 + 3 + 3 + 3,
                  "abits: both endpoints count the same rounds (three per call)");
            std::printf("  abits: %zu random bits + %zu masks + 300 bits in %.3f s, %.1f bytes per item\n", n, 2 * n_mask + 3, dt,
                        (double)(res.first.stats.bytes_sent + res.second.stats.bytes_sent) / ((double)n + 300 + (double)n_mask + 1.5));
        }
    }
    {
        // Chunked extension gives the same guarantees.
        const auto res = session(4096, [&](AbitSource& src, AbitOut& out) {
            out.shares = src.random_shares(10000);
            out.m1 = src.input_masks(1, 9000);
        });
        check(res.first.ok && res.second.ok && mac_ok(res.first.value.shares, res.second.value.shares, delta) &&
                  mac_ok(res.first.value.m1.shares, res.second.value.m1.shares, delta, &res.second.value.m1.bits),
              "abits in chunks of 4096 rows");
        check(res.first.value.sender_calls == 3 + 3 && res.second.value.sender_calls == 3, "abits: one extension call per chunk");
        check(res.first.stats.rounds == 9 + 7 && res.second.stats.rounds == 9 + 7,
              "abits: 3 rounds per chunk for random bits, 3 + 2 (chunks - 1) flights for masks");
        const uint64_t rnd = 2 * (u_message_bytes(4096) + 16 + COT_CHECK_BYTES) + u_message_bytes(10000 - 8192) + 16 + COT_CHECK_BYTES;
        check(res.first.stats.bytes_sent == rnd + 3 * 16 &&
                  res.second.stats.bytes_sent == rnd + 2 * u_message_bytes(4096) + u_message_bytes(9000 - 8192) + 3 * COT_CHECK_BYTES,
              "abits in chunks: exact bytes");
    }
    {
        // n = 0 needs no message.
        const auto res = session(DEFAULT_BATCH_ROWS, [&](AbitSource& src, AbitOut& out) {
            out.shares = src.random_shares(0);
            out.m0 = src.input_masks(0, 0);
        });
        check(res.first.ok && res.first.value.shares.size() == 0 && res.first.value.m0.shares.size() == 0 &&
                  res.first.stats.bytes_sent == 0 && res.first.stats.rounds == 0,
              "abits: empty requests");
    }
    {
        // The lsb constraint is enforced.
        SystemRng rng;
        auto pr = LoopbackChannel::make_pair();
        const TrustedSeeds s = trusted_seeds(d[1]);
        CotSender snd(0, d[1], s.chosen, rng);
        CotReceiver rcv(1, s.s0, s.s1, rng);
        check(throws<std::invalid_argument>([&] { AbitSource a(*pr.first, 0, d[1], snd, rcv); }), "AbitSource rejects lsb(Delta_0) = 0");
        CotSender snd1(1, d[0], s.chosen, rng);
        CotReceiver rcv1(0, s.s0, s.s1, rng);
        check(throws<std::invalid_argument>([&] { AbitSource a(*pr.second, 1, d[0], snd1, rcv1); }), "AbitSource rejects lsb(Delta_1) = 1");
        CotSender snd_ok(0, d[0], s.chosen, rng);
        check(throws<std::invalid_argument>([&] { AbitSource a(*pr.first, 0, d[0], snd_ok, rcv1); }),
              "AbitSource rejects a receiver of the wrong direction");
        gf128 other = d[0];
        other.lo ^= 2;
        check(throws<std::invalid_argument>([&] { AbitSource a(*pr.first, 0, other, snd_ok, rcv); }),
              "AbitSource rejects a sender with another key");
    }
}

// ------------------------------------------------------------ (e) leaky AND, honest
struct LeakyOut {
    Shares x, y, z;
    int passed = -1;  // triple check: 1 passed, 0 aborted, -1 not run
    PrepStats st;
};

// One party: 2n (+ PAD) random BDOZ bits from `source`, the leaky AND, optionally the triple check.
static LeakyOut leaky_body(int role, Channel& ch, BdozSource& source, const gf128& delta, size_t n, uint64_t index0,
                           const Cheat* cheat, bool with_check, PhaseMeter* meter = nullptr) {
    SystemRng rng;
    LeakyOut out;
    const BdozPair p = source.random_bdoz(2 * n + (with_check ? PAD : 0));
    const Bdoz x_own = p.own.slice(0, n);
    const Keys x_keys = p.keys.slice(0, n);
    out.x = spdz_shares(role, delta, x_own, x_keys);
    out.y = spdz_shares(role, delta, p.own.slice(n, 2 * n), p.keys.slice(n, 2 * n));
    mark_phase(meter, PH_LAAND);
    out.z = leaky_ands(ch, TSID, role, delta, index0, x_own, x_keys, out.y, cheat);
    mark_phase(meter, PH_NONE);
    if (with_check) {
        const Shares r = spdz_shares(role, delta, p.own.slice(2 * n, 2 * n + PAD), p.keys.slice(2 * n, 2 * n + PAD));
        try {
            triple_check(ch, TSID, 0, role, delta, out.z, r, nullptr, nullptr, 0, rng);
            out.passed = 1;
        } catch (const ProtocolAbort&) {
            out.passed = 0;
        }
    }
    if (meter) out.st = meter->stats();
    return out;
}

static void test_leaky_and() {
    Section sec("leaky AND: correctness and MAC invariant");
    const size_t n = 100000;
    gf128 d[2];
    test_deltas(d);
    const gf128 delta = d[0] ^ d[1];
    {
        const double t0 = now_s();
        const auto res = abit_session<LeakyOut>(d, [&](int role, Channel& ch, AbitSource& src, const gf128& dl, PhaseMeter& meter) {
            return leaky_body(role, ch, src, dl, n, 12345, nullptr, false, &meter);
        });
        const double dt = now_s() - t0;
        check(res.first.ok && res.second.ok, "leaky AND: both parties finish");
        if (res.first.ok && res.second.ok) {
            const LeakyOut& r0 = res.first.value;
            const LeakyOut& r1 = res.second.value;
            const BitVec x = xor_bits(r0.x.bits, r1.x.bits), y = xor_bits(r0.y.bits, r1.y.bits), z = xor_bits(r0.z.bits, r1.z.bits);
            const BitVec xy = and_bits(x, y);
            check(mac_ok(r0.x, r1.x, delta) && mac_ok(r0.y, r1.y, delta), "leaky AND: inputs are valid");
            check(z == xy, "leaky AND: z = x AND y for 100000 triples");
            check(mac_ok(r0.z, r1.z, delta, &xy), "leaky AND: n_0 ^ n_1 = x y Delta");
            bool lsb = true;
            for (size_t i = 0; i < n; i++) lsb = lsb && r0.z.bits[i] == (r0.z.macs[i].lo & 1) && r1.z.bits[i] == (r1.z.macs[i].lo & 1);
            check(lsb, "leaky AND: z_b = lsb(n_b)");
            check(std::fabs(mean(z) - 0.25) < 5 * std::sqrt(0.1875 / (double)n), "leaky AND: z is 1 a quarter of the time");
            check(std::fabs(mean(r0.z.bits) - 0.5) < 5 * 0.5 / std::sqrt((double)n), "leaky AND: z_0 looks uniform");
            const PhaseStat& la = r0.st[PH_LAAND];
            check(la.bytes_sent == 16 * n && la.bytes_recv == 16 * n && la.rounds == 1,
                  "leaky AND: one round, 16 bytes per triple and party");
            std::printf("  leaky AND: %zu triples (with their 2 x %zu COT rows) in %.3f s\n", n, 2 * n, dt);
        }
    }
    {
        // The triple check accepts honest triples.
        const auto res = abit_session<LeakyOut>(d, [&](int role, Channel& ch, AbitSource& src, const gf128& dl, PhaseMeter&) {
            return leaky_body(role, ch, src, dl, 500, 0, nullptr, true);
        });
        check(res.first.ok && res.second.ok && res.first.value.passed == 1 && res.second.value.passed == 1,
              "triple check passes on honest leaky triples");
    }
    {
        // The hash depends on index, direction and sid.
        const GfVec keys = rand_vec(4);
        GfVec h(4), h_dir(4), h_idx(4), h_sid(4);
        bytes sid2 = TSID;
        sid2.push_back('x');
        hashes::H_ccr_batch(keys.data(), 4, 0, h_dom(TSID, 0), h.data());
        hashes::H_ccr_batch(keys.data(), 4, 0, h_dom(TSID, 1), h_dir.data());
        hashes::H_ccr_batch(keys.data(), 4, 1, h_dom(TSID, 0), h_idx.data());
        hashes::H_ccr_batch(keys.data(), 4, 0, h_dom(sid2, 0), h_sid.data());
        check(h != h_dir && h != h_idx && h != h_sid, "leaky AND hash is separated by direction, triple index and session");
    }
    {
        // A party that ignores the lsb constraint (lsb(Delta) = 0) is caught by the triple check.
        IdealDealer dealer;
        gf128 bad = dealer.delta_share(1);
        bad.lo |= 1;
        dealer.set_delta_share(1, bad);
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                IdealPrep prep(role, dealer);
                return leaky_body(role, ch, prep, prep.delta(), 64, 0, nullptr, true);
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        const auto res = run_two_party(f0, f1);
        check(res.first.ok && res.second.ok && res.first.value.passed == 0 && res.second.value.passed == 0,
              "lsb(Delta^B) = 0 makes the triple check fail");
    }
    {
        // Malformed transfer message.
        IdealDealer dealer;
        auto party = [&](int role) {
            return [&, role](Channel& raw) {
                IdealPrep prep(role, dealer);
                TamperChannel ch(raw, role == 1 ? Rules{{0, truncate_rule(16)}} : Rules());
                return leaky_body(role, ch, prep, prep.delta(), 64, 0, nullptr, false);
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        const auto res = run_two_party(f0, f1);
        check(res.first.protocol_abort && res.first.error.find("transfer message") != std::string::npos,
              "leaky AND: a short transfer message is rejected");
    }
}

// ------------------------------------------------------------ (f) Lemma 3: additive error on the transfer message
struct AttackResult {
    bool p0 = false, p1 = false;  // the triple check passed at P_0 / P_1
    bool right = false;           // every guess of the cheater was right
    bool triples_ok = false;      // all triples correct and MAC-consistent
    bool ran = false;
};

static AttackResult attack_run(size_t t, int cheater, size_t n = 24) {
    IdealDealer dealer;
    Cheat cheat;
    cheat.has_g_error = true;
    std::vector<size_t> pool(n);
    for (size_t i = 0; i < n; i++) pool[i] = i;
    for (size_t i = 0; i < t; i++) std::swap(pool[i], pool[i + rand_below(n - i)]);
    cheat.g_idx.assign(pool.begin(), pool.begin() + (ptrdiff_t)t);
    std::sort(cheat.g_idx.begin(), cheat.g_idx.end());
    cheat.g_err = rand_vec(t);
    for (gf128& e : cheat.g_err) e.hi |= 1ull << 63;  // E_i != 0
    cheat.g_guess = rand_bits(t);

    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            IdealPrep prep(role, dealer);
            return leaky_body(role, ch, prep, prep.delta(), n, 0, role == cheater ? &cheat : nullptr, true);
        };
    };
    auto f0 = party(0);
    auto f1 = party(1);
    const auto res = run_two_party(f0, f1);
    AttackResult a;
    if (!res.first.ok || !res.second.ok) return a;
    a.ran = true;
    const LeakyOut& r0 = res.first.value;
    const LeakyOut& r1 = res.second.value;
    a.p0 = r0.passed == 1;
    a.p1 = r1.passed == 1;
    const LeakyOut& honest = cheater == 0 ? r1 : r0;
    a.right = true;
    for (size_t i = 0; i < t; i++) a.right = a.right && honest.x.bits[cheat.g_idx[i]] == cheat.g_guess[i];
    const BitVec xy = and_bits(xor_bits(r0.x.bits, r1.x.bits), xor_bits(r0.y.bits, r1.y.bits));
    a.triples_ok = mac_ok(r0.z, r1.z, dealer.delta(), &xy);
    return a;
}

// Runs per t of the Lemma 3 experiment (define COUPLET_TEST_ATTACK_RUNS for a longer experiment).
#ifndef COUPLET_TEST_ATTACK_RUNS
#define COUPLET_TEST_ATTACK_RUNS 200
#endif

static void test_leaky_and_attack() {
    Section sec("Lemma 3: guess attack on the leaky AND (200 runs per t by default)");
    const int runs = COUPLET_TEST_ATTACK_RUNS;
    for (size_t t = 1; t <= 3; t++) {
        int passes = 0, mismatch = 0, bad_pass = 0, good_fail = 0, failed_runs = 0;
        for (int i = 0; i < runs; i++) {
            const AttackResult a = attack_run(t, i % 2);
            if (!a.ran) {
                failed_runs++;
                continue;
            }
            const bool passed = a.p0 && a.p1;
            passes += passed;
            mismatch += (a.p0 != a.p1) || (passed != a.right);
            bad_pass += passed && !a.triples_ok;
            good_fail += !passed && a.triples_ok;
        }
        const double rate = (double)passes / runs, expect = std::ldexp(1.0, -(int)t);
        const double sd = std::sqrt(expect * (1 - expect) / runs);
        std::printf("  Lemma 3, t = %zu: check passed in %d/%d runs (rate %.3f, expected %.3f)\n", t, passes, runs, rate, expect);
        const std::string tag = "t = " + std::to_string(t);
        check(failed_runs == 0, tag + ": all runs complete");
        check(mismatch == 0, tag + ": the check passes iff every guess is right (both parties agree)");
        check(bad_pass == 0, tag + ": whenever the check passes all triples are correct");
        check(good_fail == 0, tag + ": a wrong guess leaves an inconsistent triple");
        check(std::fabs(rate - expect) < 5 * sd, tag + ": the pass rate is about 2^-t");
    }

    // An error on the share of W alone, or on a raw z share, is always caught.
    struct Case {
        const char* name;
        bool w_error;
        gf128 err;
        int flip_z;
    };
    const Case cases[3] = {{"share of W", true, gf128(1, 0), -1},
                           {"share of W (high bit)", true, gf128(0, 1ull << 36), -1},
                           {"raw z share", false, gf128(), 3}};
    for (const Case& c : cases) {
        IdealDealer dealer;
        Cheat cheat;
        cheat.has_w_error = c.w_error;
        cheat.w_error = c.err;
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                SystemRng rng;
                IdealPrep prep(role, dealer);
                const gf128 dl = prep.delta();
                const BdozPair p = prep.random_bdoz(2 * 40 + PAD);
                const Shares y = spdz_shares(role, dl, p.own.slice(40, 80), p.keys.slice(40, 80));
                const Shares r = spdz_shares(role, dl, p.own.slice(80, 80 + PAD), p.keys.slice(80, 80 + PAD));
                Shares z = leaky_ands(ch, TSID, role, dl, 0, p.own.slice(0, 40), p.keys.slice(0, 40), y);
                if (role == 0 && c.flip_z >= 0) z.bits[(size_t)c.flip_z] ^= 1;
                triple_check(ch, TSID, 0, role, dl, z, r, nullptr, nullptr, 0, rng, role == 0 ? &cheat : nullptr);
                return 1;
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        const auto res = run_two_party(f0, f1);
        check(res.first.protocol_abort && res.second.protocol_abort,
              std::string("triple check: an error on the ") + c.name + " is rejected");
    }
}

// ------------------------------------------------------------ bucketing
struct BucketRun {
    Triples leaky;  // (x, y, z) before bucketing
    BucketOut out;
    uint64_t sent = 0;
};

static void test_bucketing() {
    Section sec("bucketing: public permutation and bucket combination");

    // The public permutation.
    const hashes::seed16 seed = g_rng.seed();
    const std::vector<uint32_t> perm = bucket_permutation(seed, 10000);
    {
        std::vector<uint32_t> sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        bool is_perm = true;
        size_t fixed = 0;
        for (size_t i = 0; i < sorted.size(); i++) {
            is_perm = is_perm && sorted[i] == i;
            fixed += perm[i] == i;
        }
        check(is_perm, "bucket_permutation is a permutation");
        check(perm == bucket_permutation(seed, 10000), "deterministic in the seed");
        check(perm != bucket_permutation(g_rng.seed(), 10000), "depends on the seed");
        check(fixed < 20, "few fixed points");
        std::map<std::vector<uint32_t>, int> counts;
        for (int i = 0; i < 1200; i++) counts[bucket_permutation(g_rng.seed(), 3)]++;
        bool even = counts.size() == 6;
        for (const auto& kv : counts) even = even && std::fabs(kv.second - 200.0) < 5 * std::sqrt(1200.0 * 5 / 36);
        check(even, "all 6 permutations of 3 elements are about equally likely");
    }

    // Bucket combination on ideal (correct) leaky triples.
    for (size_t bucket : {(size_t)1, (size_t)2, (size_t)3, (size_t)5}) {
        const size_t n_out = 300, total = n_out * bucket;
        IdealDealer dealer;
        const hashes::seed16 coin = g_rng.seed();
        const std::vector<uint32_t> pm = bucket_permutation(coin, total);
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                SystemRng rng;
                IdealPrep prep(role, dealer);
                BucketRun r;
                r.leaky = prep.and_triples(total);
                r.out = bucket_combine(ch, r.leaky.a, r.leaky.b, r.leaky.c, pm, bucket);
                bool_check(ch, TSID, bucket, role, prep.delta(), r.out.d, r.out.d_macs, {}, rng);
                r.sent = ch.stats().bytes_sent;
                return r;
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        const auto res = run_two_party(f0, f1);
        const std::string tag = "bucket size " + std::to_string(bucket);
        check(res.first.ok && res.second.ok, tag + ": both parties finish (the d bits pass BoolCheck)");
        if (!res.first.ok || !res.second.ok) continue;
        const BucketRun& r0 = res.first.value;
        const BucketRun& r1 = res.second.value;
        const gf128 delta = dealer.delta();
        Triples t0, t1;
        t0.a = r0.out.a;
        t0.b = r0.out.b;
        t0.c = r0.out.c;
        t1.a = r1.out.a;
        t1.b = r1.out.b;
        t1.c = r1.out.c;
        check(t0.size() == n_out && triples_correct(t0, t1), tag + ": c = a AND b");
        check(triples_macs(t0, t1, delta), tag + ": MAC invariants of the combined triples");
        const BitVec x = xor_bits(r0.leaky.a.bits, r1.leaky.a.bits), y = xor_bits(r0.leaky.b.bits, r1.leaky.b.bits);
        const BitVec a = xor_bits(t0.a.bits, t1.a.bits), b = xor_bits(t0.b.bits, t1.b.bits);
        bool a_ok = true, b_ok = true, d_ok = r0.out.d == r1.out.d && r0.out.d.size() == n_out * (bucket - 1);
        for (size_t k = 0; k < n_out; k++) {
            unsigned ax = 0;
            for (size_t i = 0; i < bucket; i++) ax ^= x[pm[k * bucket + i]];
            a_ok = a_ok && a[k] == ax;
            b_ok = b_ok && b[k] == y[pm[k * bucket]];
            for (size_t i = 1; i < bucket && d_ok; i++)
                d_ok = r0.out.d[k * (bucket - 1) + i - 1] == (y[pm[k * bucket]] ^ y[pm[k * bucket + i]]);
        }
        check(a_ok, tag + ": a = XOR of the x");
        check(b_ok, tag + ": b = y of the head");
        check(d_ok, tag + ": d = y' XOR y''");
        check(r0.sent == (n_out * (bucket - 1) + 7) / 8 + BOOLCHECK_BYTES, tag + ": (B - 1) bits per triple");
    }
    {
        auto pr = LoopbackChannel::make_pair();
        const Shares five(5);
        const std::vector<uint32_t> p5 = {0, 1, 2, 3, 4};
        check(throws<std::invalid_argument>([&] { bucket_combine(*pr.first, five, five, five, p5, 2); }),
              "bucket_combine rejects a ragged population");
        const std::vector<uint32_t> bad = {0, 1, 2, 3, 9, 5};
        const Shares six(6);
        check(throws<std::invalid_argument>([&] { bucket_combine(*pr.first, six, six, six, bad, 2); }),
              "bucket_combine rejects an index out of range");
    }
    {
        // A leaky triple that is wrong makes its whole bucket wrong (correctness is not
        // repaired by bucketing; it is enforced by the triple check).
        IdealDealer dealer;
        const std::vector<uint32_t> pm = bucket_permutation(g_rng.seed(), 90);
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                IdealPrep prep(role, dealer);
                Triples lk = prep.and_triples(90);
                if (role == 0) lk.c.bits[17] ^= 1;
                return bucket_combine(ch, lk.a, lk.b, lk.c, pm, 3);
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        const auto res = run_two_party(f0, f1);
        size_t wrong = 0, where = 0, at = 0;
        if (res.first.ok && res.second.ok) {
            for (size_t k = 0; k < 30; k++) {
                const unsigned a = res.first.value.a.bits[k] ^ res.second.value.a.bits[k];
                const unsigned b = res.first.value.b.bits[k] ^ res.second.value.b.bits[k];
                const unsigned c = res.first.value.c.bits[k] ^ res.second.value.c.bits[k];
                if (c != (a & b)) {
                    wrong++;
                    at = k;
                }
            }
            for (size_t i = 0; i < 90; i++)
                if (pm[i] == 17) where = i / 3;
        }
        check(res.first.ok && res.second.ok && wrong == 1 && at == where, "a wrong leaky triple corrupts exactly its bucket");
    }
}

// ------------------------------------------------------------ (g) RealPrep end to end
struct RealOut {
    bool finished = false;       // the body ran to completion
    bool proto = false;          // it ended with ProtocolAbort
    std::string err;
    bool aborted = false;        // prep.aborted() afterwards
    bool after_abort = false;    // the retired session refuses every request
    Triples t, t2;
    Shares bits;
    InputMasks m0, m1;
    bool empty_ok = false, args_ok = false;
    gf128 delta;
    PrepStats st;
    unsigned bucket = 0;
    PrepCounters ctr;
    size_t cot_rows = 0;
    double secs = 0.0;
};
typedef std::function<void(RealPrep&, Channel&, RealOut&)> RealBody;

static unsigned g_threads[2] = {1, 1};  // worker threads of P_0 / P_1 in real_party

static RealOut real_party(int role, Channel& ch, const RealBody& body, const bytes& sid, int sigma, const Cheat* cheat,
                          RandomSource* rng, bool do_setup = true) {
    RealOut o;
    RealPrep prep(role, ch, sid, sigma, rng, DEFAULT_BATCH_ROWS, cheat);
    prep.set_threads(g_threads[role]);
    o.delta = prep.delta();
    try {
        if (do_setup) prep.setup();
        body(prep, ch, o);
        o.finished = true;
    } catch (const ProtocolAbort& e) {
        o.proto = true;
        o.err = e.what();
    } catch (const std::exception& e) {
        o.err = e.what();
    }
    o.aborted = prep.aborted();
    o.after_abort = prep.aborted() && prep.delta().is_zero() && throws<ProtocolAbort>([&] { prep.random_bits(4); }) &&
                    throws<ProtocolAbort>([&] { prep.and_triples(4); }) && throws<ProtocolAbort>([&] { prep.input_masks(0, 4); }) &&
                    throws<ProtocolAbort>([&] { prep.setup(); });
    o.st = prep.stats();
    o.bucket = prep.last_bucket_size();
    o.ctr = prep.counters();
    return o;
}

static std::pair<PartyOutcome<RealOut>, PartyOutcome<RealOut>> real_session(const RealBody& body, const bytes& sid = TSID,
                                                                            int sigma = 40, const Cheat* cheat0 = nullptr,
                                                                            const Cheat* cheat1 = nullptr,
                                                                            RandomSource* rng0 = nullptr,
                                                                            RandomSource* rng1 = nullptr) {
    auto f0 = [&](Channel& ch) { return real_party(0, ch, body, sid, sigma, cheat0, rng0); };
    auto f1 = [&](Channel& ch) { return real_party(1, ch, body, sid, sigma, cheat1, rng1); };
    return run_two_party(f0, f1);
}

// Bytes SENT by one party during and_triples(n), per phase (exact).
static void and_bytes_expected(size_t n, size_t bucket, uint64_t out[PH_COUNT]) {
    const size_t total = n * bucket;
    out[PH_BASEOT] = 96 * KAPPA;
    out[PH_COTEXT] = cot_bytes(2 * total + PAD);
    out[PH_ABIT] = 0;
    out[PH_LAAND] = 16 * (uint64_t)total;
    out[PH_BUCKET] = COIN_BYTES + (n * (bucket - 1) + 7) / 8;
    out[PH_CHECK] = COIN_BYTES + 16 + BOOLCHECK_BYTES;
}

static void report_triples(size_t n, const RealOut& o, double wall) {
    const PhaseStat prep = o.st.total_without_baseot(), all = o.st.total();
    std::printf("  RealPrep.and_triples(n = %zu), sigma = 40, B = %u:\n", n, o.bucket);
    std::printf("    wall time (both parties, one process): and_triples %.3f s, session incl. base OT %.3f s; "
                "%.3f us per final triple, %.0f triples/s\n",
                o.secs, wall, o.secs / (double)n * 1e6, (double)n / o.secs);
    std::printf("    bytes per final AND triple (both directions): %.1f without base OT, %.1f with base OT; model 96 B = %u\n",
                (double)(prep.bytes_sent + prep.bytes_recv) / (double)n, (double)(all.bytes_sent + all.bytes_recv) / (double)n,
                96 * o.bucket);
    std::printf("    per phase (bytes both directions / rounds / seconds at P_0):");
    for (int p = 0; p < PH_COUNT; p++)
        std::printf(" %s %llu/%llu/%.3f%s", phase_name(p), (unsigned long long)(o.st[p].bytes_sent + o.st[p].bytes_recv),
                    (unsigned long long)o.st[p].rounds, o.st[p].seconds, p + 1 < PH_COUNT ? "," : "\n");
}

static void test_real_prep() {
    Section sec("RealPrep: and_triples end to end (n = 1000, 300000, 1000000)");
    for (size_t n : {(size_t)1000, (size_t)300000, (size_t)1000000}) {
        if (n > 1000 && !g_large) {
            std::printf("  n = %zu skipped on request\n", n);
            continue;
        }
        const double t0 = now_s();
        const auto res = real_session([n](RealPrep& prep, Channel&, RealOut& o) {
            const double s = now_s();
            o.t = prep.and_triples(n);
            o.secs = now_s() - s;
            o.cot_rows = prep.cot_rows_for_triples(n);
        });
        const double wall = now_s() - t0;
        const std::string tag = "RealPrep n = " + std::to_string(n);
        check(res.first.ok && res.second.ok && res.first.value.finished && res.second.value.finished,
              tag + ": both parties finish (" + res.first.value.err + res.second.value.err + ")");
        if (!(res.first.ok && res.second.ok && res.first.value.finished && res.second.value.finished)) continue;
        const RealOut& o0 = res.first.value;
        const RealOut& o1 = res.second.value;
        const unsigned bucket = bucket_size(n, 40);
        const gf128 delta = o0.delta ^ o1.delta;
        check((o0.delta.lo & 1) == 1 && (o1.delta.lo & 1) == 0, tag + ": lsb(Delta_0) = 1, lsb(Delta_1) = 0");
        check(o0.t.a.size() == n && o0.t.b.size() == n && o0.t.c.size() == n && o1.t.size() == n, tag + ": n triples for both parties");
        check(triples_correct(o0.t, o1.t), tag + ": c = a AND b for all triples");
        check(triples_macs(o0.t, o1.t, delta), tag + ": MAC invariants of a, b and c");
        const BitVec a = xor_bits(o0.t.a.bits, o1.t.a.bits), b = xor_bits(o0.t.b.bits, o1.t.b.bits);
        check(std::fabs(mean(a) - 0.5) < 5 * 0.5 / std::sqrt((double)n) && std::fabs(mean(b) - 0.5) < 5 * 0.5 / std::sqrt((double)n) &&
                  std::fabs(mean(and_bits(a, b)) - 0.25) < 5 * std::sqrt(0.1875 / (double)n),
              tag + ": a and b look uniform and independent");
        check(std::fabs(mean(o0.t.a.bits) - 0.5) < 5 * 0.5 / std::sqrt((double)n) &&
                  std::fabs(mean(o0.t.c.bits) - 0.5) < 5 * 0.5 / std::sqrt((double)n),
              tag + ": the shares of P_0 look uniform");
        check(o0.bucket == bucket && o1.bucket == bucket, tag + ": bucket size as bucket_size says");
        check(o0.ctr.leaky_triples == n * bucket && o0.ctr.and_triples == n, tag + ": n * B leaky triples were consumed");
        check(o0.cot_rows == 2 * n * bucket + PAD, tag + ": COT rows per direction");
        uint64_t expected[PH_COUNT];
        and_bytes_expected(n, bucket, expected);
        uint64_t sum = 0;
        for (int p = 0; p < PH_COUNT; p++) sum += expected[p];
        const PartyOutcome<RealOut>* outs[2] = {&res.first, &res.second};
        for (int role = 0; role < 2; role++) {
            bool exact = true;
            for (int p = 0; p < PH_COUNT; p++) exact = exact && outs[role]->value.st[p].bytes_sent == expected[p];
            check(exact, tag + ": bytes sent per phase by P_" + std::to_string(role) + " are exact");
            check(outs[role]->stats.bytes_sent == sum && outs[role]->value.st.total().bytes_sent == sum,
                  tag + ": the phases sum to the channel total for P_" + std::to_string(role));
        }
        check(res.first.stats.bytes_sent == res.second.stats.bytes_recv && res.second.stats.bytes_sent == res.first.stats.bytes_recv,
              tag + ": sent and received bytes match");
        const uint64_t chunks = cot_chunks(2 * n * bucket + PAD);
        const uint64_t rounds[PH_COUNT] = {2, 3 * chunks, 0, 1, 3, 5};
        bool rounds_ok = true;
        uint64_t rsum = 0;
        for (int p = 0; p < PH_COUNT; p++) {
            rounds_ok = rounds_ok && o0.st[p].rounds == rounds[p] && o1.st[p].rounds == rounds[p];
            rsum += rounds[p];
        }
        check(rounds_ok && res.first.stats.rounds == rsum && res.second.stats.rounds == rsum,
              tag + ": rounds per phase (2, 3 per extension chunk, 0, 1, 3, 5)");
        const double per_and = 2.0 * (double)(sum - expected[PH_BASEOT]) / (double)n;
        check(per_and >= 96.0 * bucket && per_and <= 96.0 * bucket * 1.02 + 12000.0 / (double)n,
              tag + ": bytes per triple against the model 96 B");
        double tsum = 0;
        for (int p = 0; p < PH_COUNT; p++) tsum += o0.st[p].seconds;
        check(tsum > 0 && tsum <= res.first.seconds + 1e-3, tag + ": the phase times are within the session time");
        report_triples(n, o0, wall);
    }

    // Worker threads inside a party change neither the messages nor the results:
    // a party with several workers runs against a party with one, and both with several.
    for (int variant = 0; variant < 2 && g_large; variant++) {
        g_threads[0] = variant == 0 ? 4 : 3;
        g_threads[1] = variant == 0 ? 1 : 2;
        const size_t n = 300000;
        const auto res = real_session([n](RealPrep& prep, Channel&, RealOut& o) {
            const double s = now_s();
            o.t = prep.and_triples(n);
            o.secs = now_s() - s;
            o.bits = prep.random_bits(70000);
            o.m0 = prep.input_masks(0, 1000);
        });
        const std::string tag = "RealPrep n = 300000 with " + std::to_string(g_threads[0]) + " / " + std::to_string(g_threads[1]) + " worker threads";
        g_threads[0] = g_threads[1] = 1;
        check(res.first.ok && res.second.ok && res.first.value.finished && res.second.value.finished, tag + ": both parties finish");
        if (!(res.first.ok && res.second.ok && res.first.value.finished && res.second.value.finished)) continue;
        const RealOut& o0 = res.first.value;
        const RealOut& o1 = res.second.value;
        const gf128 delta = o0.delta ^ o1.delta;
        check(triples_correct(o0.t, o1.t) && triples_macs(o0.t, o1.t, delta), tag + ": triples are correct and authenticated");
        check(mac_ok(o0.bits, o1.bits, delta) && mac_ok(o0.m0.shares, o1.m0.shares, delta, &o0.m0.bits),
              tag + ": random bits and input masks are authenticated");
        uint64_t expected[PH_COUNT];
        and_bytes_expected(n, 3, expected);
        check(o0.st[PH_LAAND].bytes_sent == expected[PH_LAAND] && o0.st[PH_BUCKET].bytes_sent == expected[PH_BUCKET] &&
                  o0.st[PH_CHECK].bytes_sent == expected[PH_CHECK] && o1.st[PH_BUCKET].bytes_sent == expected[PH_BUCKET],
              tag + ": the same bytes as with one thread");
        std::printf("  %s: and_triples %.3f s\n", tag.c_str(), o0.secs);
    }

    // Several requests in one session; sigma = 64; empty requests.
    {
        const auto res = real_session(
            [](RealPrep& prep, Channel&, RealOut& o) {
                o.t = prep.and_triples(200);
                o.bits = prep.random_bits(777);
                o.m0 = prep.input_masks(0, 100);
                o.m1 = prep.input_masks(1, 60);
                o.t2 = prep.and_triples(50);
                const Triples e = prep.and_triples(0);
                const InputMasks e0 = prep.input_masks(0, 0), e1 = prep.input_masks(1, 0);
                o.empty_ok = e.size() == 0 && prep.random_bits(0).size() == 0 && e0.shares.size() == 0 &&
                             e1.shares.size() == 0 && e0.has_bits == (prep.role() == 0) && e1.has_bits == (prep.role() == 1) &&
                             prep.random_bdoz(0).own.size() == 0;
            },
            TSID, 64);
        check(res.first.ok && res.second.ok && res.first.value.finished && res.second.value.finished, "RealPrep: mixed session finishes");
        if (res.first.ok && res.second.ok && res.first.value.finished && res.second.value.finished) {
            const RealOut& o0 = res.first.value;
            const RealOut& o1 = res.second.value;
            const gf128 delta = o0.delta ^ o1.delta;
            check(triples_correct(o0.t, o1.t) && triples_macs(o0.t, o1.t, delta) && triples_correct(o0.t2, o1.t2) &&
                      triples_macs(o0.t2, o1.t2, delta),
                  "RealPrep: two and_triples batches in one session are both correct");
            check(o0.bits.size() == 777 && mac_ok(o0.bits, o1.bits, delta), "RealPrep.random_bits: MAC invariant");
            check(mac_ok(o0.m0.shares, o1.m0.shares, delta, &o0.m0.bits) && o0.m0.has_bits && !o1.m0.has_bits,
                  "RealPrep.input_masks(0): invariant and ownership");
            check(mac_ok(o0.m1.shares, o1.m1.shares, delta, &o1.m1.bits) && o1.m1.has_bits && !o0.m1.has_bits,
                  "RealPrep.input_masks(1): invariant and ownership");
            const unsigned b200 = bucket_size(200, 64), b50 = bucket_size(50, 64);
            check(o0.ctr.random_bits == 777 && o0.ctr.input_masks == 160 && o0.ctr.and_triples == 250 &&
                      o0.ctr.leaky_triples == 200 * b200 + 50 * b50,
                  "RealPrep counters (sigma = 64)");
            check(o0.bucket == b50 && b50 > bucket_size(50, 40), "sigma = 64 buckets are larger");
            check(o0.empty_ok && o1.empty_ok, "RealPrep: empty requests");
            check(res.first.stats.rounds == res.second.stats.rounds && o0.st.total().rounds == res.first.stats.rounds &&
                      o0.st.total().bytes_sent == res.first.stats.bytes_sent,
                  "RealPrep: both endpoints count the same rounds; every byte is booked in a phase");
        }
    }

    // The same session over real sockets.
    {
        TcpListener listener("127.0.0.1", 0);
        const uint16_t port = listener.port();
        RealOut o[2];
        std::string errs[2];
        const RealBody body = [](RealPrep& prep, Channel&, RealOut& out) { out.t = prep.and_triples(300); };
        std::thread t1([&] {
            try {
                std::unique_ptr<TcpChannel> ch = TcpChannel::connect(1, "127.0.0.1", port);
                o[1] = real_party(1, *ch, body, TSID, 40, nullptr, nullptr);
            } catch (const std::exception& e) {
                errs[1] = e.what();
            }
        });
        try {
            std::unique_ptr<TcpChannel> ch = listener.accept(0);
            o[0] = real_party(0, *ch, body, TSID, 40, nullptr, nullptr);
        } catch (const std::exception& e) {
            errs[0] = e.what();
        }
        t1.join();
        check(errs[0].empty() && errs[1].empty() && o[0].finished && o[1].finished && triples_correct(o[0].t, o[1].t) &&
                  triples_macs(o[0].t, o[1].t, o[0].delta ^ o[1].delta),
              "RealPrep over TCP on 127.0.0.1 (" + errs[0] + errs[1] + ")");
    }

    // Seeded randomness fixes the key share; argument validation leaves the session alive.
    {
        CtrPrgRng r0((uint64_t)11), r1((uint64_t)12), e0((uint64_t)11), e1((uint64_t)12);
        const auto res = real_session(
            [](RealPrep& prep, Channel&, RealOut& o) {
                o.args_ok = throws<std::invalid_argument>([&] { prep.input_masks(2, 5); }) &&
                            throws<std::invalid_argument>([&] { prep.setup(); }) &&
                            throws<std::invalid_argument>([&] { prep.and_triples((size_t)1 << 40); }) && !prep.aborted();
                o.bits = prep.random_bits(10);
            },
            TSID, 40, nullptr, nullptr, &r0, &r1);
        check(res.first.ok && res.second.ok && res.first.value.args_ok && res.second.value.args_ok && !res.first.value.aborted &&
                  mac_ok(res.first.value.bits, res.second.value.bits, res.first.value.delta ^ res.second.value.delta),
              "RealPrep: invalid arguments throw std::invalid_argument and do not abort the session");
        check(res.first.value.delta == sample_delta(0, e0) && res.second.value.delta == sample_delta(1, e1),
              "RealPrep: the key share is the first draw of a seeded source");
        auto pr = LoopbackChannel::make_pair();
        check(throws<std::invalid_argument>([&] { RealPrep p(1, *pr.first, TSID); }) &&
                  throws<std::invalid_argument>([&] { RealPrep p(2, *pr.first, TSID); }) &&
                  throws<std::invalid_argument>([&] { RealPrep p(0, *pr.first, TSID, 0); }),
              "RealPrep validates the role against the endpoint");
        RealPrep p(0, *pr.first, TSID);
        check(throws<std::invalid_argument>([&] { p.random_bits(5); }) && !p.aborted() && !p.is_set_up() && !p.ideal(),
              "RealPrep requires setup()");
    }
}

// ------------------------------------------------------------ small requests
// A request below 306 leaky triples is padded; the padded run is a complete protocol run.
static void test_small_requests() {
    Section sec("RealPrep: small requests are padded to at least 306 leaky triples");
    struct Row {
        Triples t;
        uint64_t sent = 0, rounds = 0;
        unsigned bucket = 0;
        size_t run = 0, cot_rows = 0;
        PrepCounters ctr;
    };
    struct Case {
        int sigma;
        std::vector<size_t> sizes;
    };
    const Case cases[] = {{40, {1, 5, 43, 44, 45}}, {64, {3, 25, 26}}};
    for (const Case& cs : cases) {
        std::vector<Row> rows[2];
        const std::vector<size_t>& sizes = cs.sizes;
        const auto res = real_session(
            [&rows, &sizes](RealPrep& prep, Channel& ch, RealOut&) {
                std::vector<Row>& out = rows[prep.role()];
                for (size_t n : sizes) {
                    Row r;
                    const uint64_t sent = ch.stats().bytes_sent, rounds = ch.stats().rounds;
                    r.t = prep.and_triples(n);
                    r.sent = ch.stats().bytes_sent - sent;
                    r.rounds = ch.stats().rounds - rounds;
                    r.bucket = prep.last_bucket_size();
                    r.run = prep.last_run_triples();
                    r.cot_rows = prep.cot_rows_for_triples(n);
                    r.ctr = prep.counters();
                    out.push_back(std::move(r));
                }
            },
            TSID, cs.sigma);
        const std::string stag = "sigma = " + std::to_string(cs.sigma);
        const bool fin = res.first.ok && res.second.ok && res.first.value.finished && res.second.value.finished &&
                         rows[0].size() == sizes.size() && rows[1].size() == sizes.size();
        check(fin, stag + ": the session with small requests finishes (" + res.first.value.err + res.second.value.err + ")");
        if (!fin) continue;
        const gf128 delta = res.first.value.delta ^ res.second.value.delta;
        uint64_t leaky = 0, asked = 0;
        for (size_t i = 0; i < sizes.size(); i++) {
            const size_t n = sizes[i];
            const TripleRequest run = triple_request(n, cs.sigma);
            leaky += run.leaky();
            asked += n;
            const std::string tag = stag + ", request of " + std::to_string(n);
            const Row& r0 = rows[0][i];
            const Row& r1 = rows[1][i];
            check(run.leaky() >= MIN_LEAKY_TRIPLES && (run.ell > n) == (n * bucket_size(n, cs.sigma) < MIN_LEAKY_TRIPLES),
                  tag + ": the run has " + std::to_string(run.ell) + " x " + std::to_string(run.bucket) + " >= 306 leaky triples");
            check(r0.t.a.size() == n && r0.t.b.size() == n && r0.t.c.size() == n && r1.t.a.size() == n && r1.t.b.size() == n &&
                      r1.t.c.size() == n,
                  tag + ": exactly the requested triples are returned");
            check(triples_correct(r0.t, r1.t) && triples_macs(r0.t, r1.t, delta), tag + ": the triples are correct and authenticated");
            check(r0.bucket == run.bucket && r1.bucket == run.bucket && r0.run == run.ell && r1.run == run.ell,
                  tag + ": bucket size and run size as planned");
            check(r0.ctr.leaky_triples == leaky && r1.ctr.leaky_triples == leaky && r0.ctr.and_triples == asked,
                  tag + ": the counters hold the leaky triples of the padded run");
            check(r0.cot_rows == 2 * run.leaky() + PAD && r1.cot_rows == r0.cot_rows, tag + ": COT rows per direction");
            uint64_t expected[PH_COUNT];
            and_bytes_expected((size_t)run.ell, run.bucket, expected);
            uint64_t sum = 0;
            for (int p = 0; p < PH_COUNT; p++)
                if (p != PH_BASEOT) sum += expected[p];
            check(r0.sent == sum && r1.sent == sum,
                  tag + ": the bytes are those of a request for " + std::to_string(run.ell) + " triples (" +
                      std::to_string(r0.sent) + " vs " + std::to_string(sum) + ")");
            check(r0.rounds == 12 && r1.rounds == 12, tag + ": twelve rounds");
        }
    }

    // The check covers the padding: deviations on leaky triples and opened bits that
    // exist only because the request was padded abort the session.
    const RealBody one = [](RealPrep& prep, Channel&, RealOut& o) { o.t = prep.and_triples(1); };
    {
        Cheat cheat;
        cheat.has_g_error = true;
        for (size_t i = 0; i < 40; i++) cheat.g_idx.push_back(44 + 6 * i);  // leaky triples 44 .. 278 of the 308
        cheat.g_err = rand_vec(40);
        for (gf128& e : cheat.g_err) e.lo |= 2;
        cheat.g_guess.assign(40, 0);
        const auto res = real_session(one, TSID, 40, nullptr, &cheat);
        check(res.first.value.proto && res.second.value.proto && res.first.value.after_abort && res.first.value.t.size() == 0,
              "request of 1: errors on 40 transfer messages of the padded run are caught");
    }
    for (int cheater = 0; cheater < 2; cheater++) {
        Cheat cheat;
        cheat.flip_d.push_back(200);  // one of the 44 x 6 = 264 opened bits
        const auto res = real_session(one, TSID, 40, cheater == 0 ? &cheat : nullptr, cheater == 1 ? &cheat : nullptr);
        check(res.first.ok && res.second.ok && res.first.value.proto && res.second.value.proto &&
                  res.first.value.err == "BoolCheck failed" && res.second.value.err == "BoolCheck failed",
              "request of 1: P_" + std::to_string(cheater) + " flips an opened bit of the padded run: BoolCheck aborts");
    }
}

// ------------------------------------------------------------ aborts
static void test_real_prep_aborts() {
    Section sec("RealPrep: deviations retire the session");
    const RealBody triples400 = [](RealPrep& prep, Channel&, RealOut& o) { o.t = prep.and_triples(400); };

    // A tampered opened d bit makes the BoolCheck abort.
    for (int cheater = 0; cheater < 2; cheater++) {
        Cheat cheat;
        cheat.flip_d.push_back(123);
        const auto res = real_session(triples400, TSID, 40, cheater == 0 ? &cheat : nullptr, cheater == 1 ? &cheat : nullptr);
        const std::string tag = "P_" + std::to_string(cheater) + " flips an opened d bit";
        check(res.first.ok && res.second.ok && res.first.value.proto && res.second.value.proto &&
                  res.first.value.err == "BoolCheck failed" && res.second.value.err == "BoolCheck failed",
              tag + ": both parties abort in BoolCheck");
        check(res.first.value.after_abort && res.second.value.after_abort, tag + ": the session is retired on both sides");
        check(res.first.value.t.size() == 0 && res.second.value.t.size() == 0, tag + ": no triple is returned");
    }

    // Lemma 3 deviation through the whole pipeline: errors on 40 transfer messages
    // survive only if all 40 guesses are right (probability 2^-40).
    {
        Cheat cheat;
        cheat.has_g_error = true;
        for (size_t i = 0; i < 40; i++) cheat.g_idx.push_back(5 * i);
        cheat.g_err = rand_vec(40);
        for (gf128& e : cheat.g_err) e.lo |= 2;
        cheat.g_guess.assign(40, 0);
        const auto res = real_session(triples400, TSID, 40, nullptr, &cheat);
        check(res.first.value.proto && res.second.value.proto && res.first.value.after_abort,
              "errors on 40 transfer messages are caught by the triple check");
    }
    {
        Cheat cheat;
        cheat.has_w_error = true;
        cheat.w_error = gf128(3, 0);
        const auto res = real_session(triples400, TSID, 40, &cheat, nullptr);
        check(res.first.value.proto && res.second.value.proto && res.second.value.after_abort,
              "a wrong share of W aborts the session");
    }

    // An inconsistent U in 40 columns of the extension is caught by the COT check
    // (it survives only if Delta is 0 on all 40 columns).
    {
        Cheat cheat;
        for (unsigned c = 0; c < 120; c += 3) cheat.flip_u.push_back(std::make_pair(c, (size_t)(5 + c)));
        const auto res = real_session(
            [](RealPrep& prep, Channel&, RealOut& o) {
                o.bits = prep.random_bits(1000);
                o.bits = prep.random_bits(10);
            },
            TSID, 40, nullptr, &cheat);
        check(res.first.value.proto && res.first.value.err.find("COT") != std::string::npos && res.first.value.after_abort,
              "an inconsistent extension matrix aborts the honest sender");
        check(!res.second.value.finished && res.second.value.aborted, "the cheater's session ends as well (peer closed)");
    }

    // A base-OT sender that replaces every second A: the seeds of those columns
    // disagree and the extension check of the honest COT sender fails.
    {
        ed25519::scalar32 s{};
        s[0] = 0x32;
        s[1] = 0x79;
        s[2] = 0x06;  // 424242
        const ed25519::bytes32 blob = ed25519::compress(ed25519::mul_base(s));
        Cheat cheat;
        cheat.flight2 = [blob](const bytes& msg) {
            bytes out = msg;
            for (size_t i = 0; i < KAPPA; i += 2) std::copy(blob.begin(), blob.end(), out.begin() + (ptrdiff_t)(32 * i));
            return out;
        };
        const auto res = real_session(triples400, TSID, 40, &cheat, nullptr);
        check(res.second.value.proto && res.second.value.err.find("COT") != std::string::npos && !res.first.value.finished,
              "corrupted base-OT seeds are caught by the extension check");
    }

    // The peer disappears in the middle of a request.
    {
        auto f0 = [&](Channel& ch) { return real_party(0, ch, [](RealPrep& prep, Channel&, RealOut& o) { o.t = prep.and_triples(100); }, TSID, 40, nullptr, nullptr); };
        auto f1 = [&](Channel& ch) { return real_party(1, ch, [](RealPrep&, Channel&, RealOut&) {}, TSID, 40, nullptr, nullptr); };
        const auto res = run_two_party(f0, f1);
        check(res.first.ok && !res.first.value.finished && res.first.value.after_abort && res.second.value.finished,
              "a vanished peer aborts the session");
    }
}

// ------------------------------------------------------------ (h) ideal preprocessing
struct IdealOut {
    Shares bits;
    InputMasks m0, m1;
    Triples t;
    BdozPair bd;
    gf128 delta;
    bool ideal = false;
    PrepCounters ctr;
};

static void test_ideal_prep() {
    Section sec("IdealPrep: the ideal dealer");
    const size_t n = 100000;
    IdealDealer dealer;
    const gf128 d0 = dealer.delta_share(0), d1 = dealer.delta_share(1), delta = d0 ^ d1;
    check((d0.lo & 1) == 1 && (d1.lo & 1) == 0 && dealer.delta() == delta, "ideal: lsb constraint on the key shares");
    {
        auto party = [&](int role) {
            return [&, role](Channel&) {
                IdealPrep prep(role, dealer);
                prep.setup();
                IdealOut o;
                o.bits = prep.random_bits(n);
                o.m0 = prep.input_masks(0, 1000);
                o.m1 = prep.input_masks(1, 1000);
                o.t = prep.and_triples(n);
                o.bd = prep.random_bdoz(1000);
                o.delta = prep.delta();
                o.ideal = prep.ideal();
                o.ctr = prep.counters();
                return o;
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        const auto res = run_two_party(f0, f1);
        check(res.first.ok && res.second.ok, "ideal: both parties finish");
        if (res.first.ok && res.second.ok) {
            const IdealOut& r0 = res.first.value;
            const IdealOut& r1 = res.second.value;
            check(r0.delta == d0 && r1.delta == d1 && r0.ideal, "ideal: each party gets its key share");
            check(r0.bits.size() == n && mac_ok(r0.bits, r1.bits, delta), "ideal: MAC invariant for 100000 random bits");
            check(std::fabs(mean(xor_bits(r0.bits.bits, r1.bits.bits)) - 0.5) < 5 * 0.5 / std::sqrt((double)n),
                  "ideal: random bits look uniform");
            check(mac_ok(r0.m0.shares, r1.m0.shares, delta, &r0.m0.bits) && r0.m0.has_bits && !r1.m0.has_bits &&
                      mean(r1.m0.shares.bits) == 0.0,
                  "ideal: input masks of P_0");
            check(mac_ok(r0.m1.shares, r1.m1.shares, delta, &r1.m1.bits) && r1.m1.has_bits && !r0.m1.has_bits &&
                      mean(r0.m1.shares.bits) == 0.0,
                  "ideal: input masks of P_1");
            check(triples_correct(r0.t, r1.t) && triples_macs(r0.t, r1.t, delta), "ideal: 100000 triples are correct and authenticated");
            const BitVec ab = and_bits(xor_bits(r0.t.a.bits, r1.t.a.bits), xor_bits(r0.t.b.bits, r1.t.b.bits));
            check(std::fabs(mean(ab) - 0.25) < 5 * std::sqrt(0.1875 / (double)n), "ideal: triples look uniform");
            check(bdoz_ok(r0.bd.own, r1.bd.keys, d1) && bdoz_ok(r1.bd.own, r0.bd.keys, d0), "ideal: BDOZ bits");
            check(res.first.stats.bytes_sent == 0, "ideal: no communication");
            check(r0.ctr.random_bits == n && r0.ctr.input_masks == 2000 && r0.ctr.and_triples == n, "ideal: counters");
        }
    }
    {
        // Either party may ask first; requests must agree.
        IdealPrep p0(0, dealer), p1(1, dealer);
        const Shares s1 = p1.random_bits(50);
        const Shares s0 = p0.random_bits(50);
        check(mac_ok(s0, s1, delta), "ideal: P_1 may ask first");
        p0.random_bits(5);
        check(throws<std::invalid_argument>([&] { p1.random_bits(6); }), "ideal: mismatching sizes are detected");
        p0.and_triples(5);
        check(throws<std::invalid_argument>([&] { p1.random_bits(5); }), "ideal: mismatching kinds are detected");
        IdealDealer other;
        IdealPrep q(0, other);
        check(throws<std::invalid_argument>([&] { q.input_masks(2, 1); }) && throws<std::invalid_argument>([&] { IdealPrep bad(2, dealer); }),
              "ideal: argument validation");
        q.abort();
        check(q.aborted() && q.delta().is_zero() && throws<ProtocolAbort>([&] { q.random_bits(1); }),
              "ideal: an aborted session hands out nothing");
        CtrPrgRng ra((uint64_t)5), rb((uint64_t)5);
        IdealDealer da(&ra), db(&rb);
        check(da.delta_share(0) == db.delta_share(0) && da.delta_share(1) == db.delta_share(1) && da.delta_share(0) != d0,
              "ideal: a seeded dealer is reproducible");
    }
    {
        // The real triple pipeline on top of the ideal bit source.
        IdealDealer dealer3;
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                SystemRng rng;
                IdealPrep prep(role, dealer3);
                return and_triples(ch, TSID, role, prep.delta(), prep, 500, 3, 0, 0, rng);
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        const auto res = run_two_party(f0, f1);
        check(res.first.ok && res.second.ok && res.first.value.size() == 500 && triples_correct(res.first.value, res.second.value) &&
                  triples_macs(res.first.value, res.second.value, dealer3.delta()),
              "and_triples on ideal authenticated bits");
    }
}

// ------------------------------------------------------------ (i) BoolCheck
typedef std::function<void(int, BitVec&, GfVec&)> BitTamper;
typedef std::function<void(int, std::vector<FieldOpening>&)> FieldTamper;

struct BcOut {
    uint64_t sent = 0;
};

// Both parties run bool_check on n opened bits (+ `extra` field openings).
static std::pair<PartyOutcome<BcOut>, PartyOutcome<BcOut>> bool_check_run(size_t n, BitTamper tamper = nullptr, size_t extra = 0,
                                                                          FieldTamper tamper_extra = nullptr,
                                                                          uint64_t counter0 = 7, uint64_t counter1 = 7,
                                                                          const bytes& sid1 = TSID) {
    IdealDealer dealer;
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            SystemRng rng;
            IdealPrep prep(role, dealer);
            const Shares shares = prep.random_bits(n);
            BitVec opened = xor_bits(shares.bits, exchange_bits(ch, shares.bits));
            GfVec macs = shares.macs;
            std::vector<FieldOpening> items;
            if (extra) {
                const Shares rs = prep.random_bits(128 * extra);
                for (size_t k = 0; k < extra; k++) {
                    const std::pair<gf128, gf128> f = rs.slice(128 * k, 128 * (k + 1)).to_field();
                    const bytes reply = ch.exchange(gf_to_bytes(f.first));
                    items.push_back(FieldOpening(f.first ^ gf128::from_bytes(reply.data()), f.second));
                }
            }
            if (tamper) tamper(role, opened, macs);
            if (tamper_extra) tamper_extra(role, items);
            bool_check(ch, role == 0 ? TSID : sid1, role == 0 ? counter0 : counter1, role, prep.delta(), opened, macs, items, rng);
            BcOut o;
            o.sent = ch.stats().bytes_sent;
            return o;
        };
    };
    auto f0 = party(0);
    auto f1 = party(1);
    return run_two_party(f0, f1);
}

static bool both_fail(const std::pair<PartyOutcome<BcOut>, PartyOutcome<BcOut>>& r) {
    return r.first.protocol_abort && r.second.protocol_abort && r.first.error == "BoolCheck failed" &&
           r.second.error == "BoolCheck failed";
}

static void test_bool_check() {
    Section sec("BoolCheck: batched MAC check of opened values");
    const std::pair<size_t, size_t> shapes[5] = {{0, 0}, {1, 0}, {1000, 0}, {0, 1}, {5000, 3}};
    uint64_t sent_last = 0;
    for (const auto& sh : shapes) {
        const auto res = bool_check_run(sh.first, nullptr, sh.second);
        check(res.first.ok && res.second.ok,
              "BoolCheck passes on " + std::to_string(sh.first) + " honest bits and " + std::to_string(sh.second) + " field openings");
        sent_last = res.first.ok ? res.first.value.sent : 0;
    }
    check(sent_last - (5000 + 7) / 8 - 3 * 16 == BOOLCHECK_BYTES,
          "BoolCheck costs 96 bytes per party, independent of the number of openings");

    // A tampered opened bit: both parties hold the same wrong public value.
    for (size_t pos : {(size_t)0, (size_t)499, (size_t)999}) {
        const auto res = bool_check_run(1000, [pos](int, BitVec& o, GfVec&) { o[pos] ^= 1; });
        check(both_fail(res), "BoolCheck aborts when opened bit " + std::to_string(pos) + " was tampered with");
    }
    // One party's opened value differs from the other's (inconsistent views).
    check(both_fail(bool_check_run(1000, [](int role, BitVec& o, GfVec&) {
              if (role == 0) o[5] ^= 1;
          })),
          "BoolCheck aborts when the parties' views differ");
    // A wrong MAC share, a wrong field opening, a wrong field MAC share.
    check(both_fail(bool_check_run(1000, [](int role, BitVec&, GfVec& m) {
              if (role == 1) m[3].hi ^= 1ull << 40;
          })),
          "BoolCheck aborts on a wrong MAC share");
    check(both_fail(bool_check_run(10, nullptr, 2, [](int, std::vector<FieldOpening>& it) { it[1].value.lo ^= 1; })),
          "BoolCheck aborts on a tampered field opening");
    check(both_fail(bool_check_run(10, nullptr, 2, [](int role, std::vector<FieldOpening>& it) {
              if (role == 0) it[0].mac.lo ^= 2;
          })),
          "BoolCheck aborts on a wrong MAC share of a field opening");
    check(both_fail(bool_check_run(10, nullptr, 2, [](int role, std::vector<FieldOpening>& it) {
              if (role == 1) std::swap(it[0], it[1]);
          })),
          "BoolCheck binds the order of the openings");
    // Counter and sid are bound (a digest of another check or session cannot be reused).
    {
        // (the commitment label contains the counter, so the opening already fails to verify)
        const auto res = bool_check_run(100, nullptr, 0, nullptr, 7, 8);
        check(res.first.protocol_abort && res.second.protocol_abort, "BoolCheck: different counters do not match");
    }
    {
        bytes sid2 = TSID;
        sid2.push_back('\'');
        const auto res = bool_check_run(100, nullptr, 0, nullptr, 7, 7, sid2);
        check(res.first.protocol_abort && res.second.protocol_abort, "BoolCheck: different session identifiers do not match");
    }

    // The digest is what the documentation says.
    {
        gf128 d[2];
        test_deltas(d);
        const gf128 delta = d[0] ^ d[1];
        const BitVec opened = rand_bits(50);
        const GfVec m0 = rand_vec(50);
        GfVec m1(50);
        for (size_t i = 0; i < 50; i++) m1[i] = m0[i] ^ gf_and_bit(delta, opened[i]);
        const gf128 w = g_rng.gf(), big_m0 = g_rng.gf();
        const gf128 big_m1 = big_m0 ^ gf128_mul(w, delta);
        const GfVec v0 = check_values(d[0], opened, m0);
        check(v0 == check_values(d[1], opened, m1), "V_j agrees for correctly opened bits");
        bool vdef = true;
        for (size_t i = 0; i < 50; i++) vdef = vdef && v0[i] == (m0[i] ^ gf_and_bit(d[0], opened[i]));
        check(vdef, "V_j = m_j XOR y_j Delta_b");
        const hashes::hash32 dig0 = check_digest(TSID, 3, d[0], opened, m0, {FieldOpening(w, big_m0)});
        check(dig0 == check_digest(TSID, 3, d[1], opened, m1, {FieldOpening(w, big_m1)}), "the two digests agree");
        const bytes vb = gf_to_bytes(v0), vf = gf_to_bytes(big_m0 ^ gf128_mul(w, d[0]));
        check(dig0 == hashes::framed_sha256({"Couplet-boolcheck", TSID, hashes::le64(3), vb, vf}),
              "digest = SHA-256(tag, sid, counter, V bits, V field) with length framing");
        check(dig0 != check_digest(TSID, 4, d[0], opened, m0, {FieldOpening(w, big_m0)}) &&
                  dig0 != check_digest(TSID, 3, d[0], opened, m0, {FieldOpening(w ^ gf128::one(), big_m0)}) &&
                  dig0 != check_digest(TSID, 3, d[0], opened, m0),
              "the digest depends on every input");
        // many bits: the buffered path of check_digest
        const BitVec big_o = rand_bits(5000);
        const GfVec big_m = rand_vec(5000);
        const bytes big_v = gf_to_bytes(check_values(d[0], big_o, big_m));
        check(check_digest(TSID, 9, d[0], big_o, big_m) ==
                  hashes::framed_sha256({"Couplet-boolcheck", TSID, hashes::le64(9), big_v, bytes()}),
              "digest of 5000 bits equals the one-shot framed hash");
        BitVec flipped = opened;
        flipped[9] ^= 1;
        const GfVec g0 = check_values(d[0], flipped, m0), g1 = check_values(d[1], flipped, m1);
        bool gap = (g0[9] ^ g1[9]) == delta;
        for (size_t i = 0; i < 50; i++)
            if (i != 9) gap = gap && g0[i] == g1[i];
        check(gap, "a wrongly opened bit shifts V_j by Delta^B");
        check(throws<std::invalid_argument>([&] { check_digest(TSID, 3, d[0], BitVec{0, 3}, GfVec(2)); }),
              "check_digest rejects values that are not bits");
    }

    // Replaying the honest party's commitment and opening (the "mirror" deviation) fails.
    {
        IdealDealer dealer;
        auto honest = [&](Channel& ch) {
            SystemRng rng;
            IdealPrep prep(0, dealer);
            const Shares s = prep.random_bits(20);
            const BitVec opened = xor_bits(s.bits, exchange_bits(ch, s.bits));  // the adversary lied about bit 4
            bool_check(ch, TSID, 0, 0, prep.delta(), opened, s.macs, {}, rng);
            return 1;
        };
        auto mirror = [&](Channel& ch) {
            IdealPrep prep(1, dealer);
            const Shares s = prep.random_bits(20);
            BitVec bits = s.bits;
            bits[4] ^= 1;
            exchange_bits(ch, bits);
            const bytes com = ch.recv();  // rushing: read the commitment first
            ch.send(com);
            ch.send(ch.recv());
            return 1;
        };
        const auto res = run_two_party(honest, mirror);
        check(res.first.protocol_abort && res.first.error.find("mirrored") != std::string::npos,
              "BoolCheck: a replayed commitment is rejected");
    }

    // Argument validation.
    {
        SystemRng rng;
        auto pr = LoopbackChannel::make_pair();
        const BitVec opened(4, 0);
        const GfVec macs(4), short_macs(3);
        check(throws<std::invalid_argument>([&] { bool_check(*pr.first, TSID, 0, 1, gf128(), opened, macs, {}, rng); }),
              "role must match the endpoint");
        check(throws<std::invalid_argument>([&] { bool_check(*pr.first, TSID, 0, 0, gf128(), opened, short_macs, {}, rng); }),
              "one MAC per opened bit");
    }
}

// ------------------------------------------------------------ interoperability mode
// Runs setup() and and_triples(n) over TCP and writes this party's view:
//   16 bytes Delta share, then for a, b, c: n bytes of bit shares, 16 n bytes of MAC shares.
static int interop(int party, uint16_t port, size_t n, const char* path) {
    const bytes sid = {'i', 'n', 't', 'e', 'r', 'o', 'p', '-', 's', 'e', 's', 's', 'i', 'o', 'n'};
    try {
        std::unique_ptr<TcpChannel> ch = TcpChannel::create(party, "127.0.0.1", port);
        RealPrep prep(party, *ch, sid);
        const gf128 delta = prep.delta();
        prep.setup();
        const Triples t = prep.and_triples(n);
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open the output file");
        const bytes d = gf_to_bytes(delta);
        f.write((const char*)d.data(), (std::streamsize)d.size());
        for (const Shares* s : {&t.a, &t.b, &t.c}) {
            f.write((const char*)s->bits.data(), (std::streamsize)s->bits.size());
            const bytes m = gf_to_bytes(s->macs);
            f.write((const char*)m.data(), (std::streamsize)m.size());
        }
        f.close();
        if (!f) throw std::runtime_error("cannot write the output file");
        const PrepStats& st = prep.stats();
        std::printf("interop party=%d n=%zu bucket=%u bytes_sent=%llu bytes_recv=%llu rounds=%llu\n", party, n,
                    prep.last_bucket_size(), (unsigned long long)ch->stats().bytes_sent,
                    (unsigned long long)ch->stats().bytes_recv, (unsigned long long)ch->stats().rounds);
        std::printf("interop stats %s\n", st.json().c_str());
        return 0;
    } catch (const std::exception& e) {
        std::printf("interop FAILED: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--interop") == 0) {
        if (argc != 6) {
            std::printf("usage: test_prep --interop PARTY PORT N OUTFILE\n");
            return 2;
        }
        return interop(std::atoi(argv[2]), (uint16_t)std::atoi(argv[3]), (size_t)std::atoll(argv[4]), argv[5]);
    }
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], "--no-large") == 0) g_large = false;
    const double t0 = now_s();
    void (*const tests[])() = {test_params,    test_triple_request, test_kernels,          test_shares_model,
                               test_vectors,   test_base_ot,        test_cot,              test_abits,
                               test_leaky_and, test_leaky_and_attack, test_bucketing,      test_real_prep,
                               test_small_requests, test_real_prep_aborts, test_ideal_prep, test_bool_check};
    for (void (*test)() : tests) {
        try {
            test();
        } catch (const std::exception& e) {  // an exception that a test did not expect is a failure
            check(false, std::string("unexpected exception: ") + e.what());
        } catch (...) {
            check(false, "unexpected exception of unknown type");
        }
    }
    std::printf("RESULT: %d checks passed, %d failed (%.1f s)\n", g_pass, g_fail, now_s() - t0);
    return g_fail ? 1 : 0;
}
