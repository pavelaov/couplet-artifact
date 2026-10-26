// test_baselines.cpp -- tests of the two evaluation baselines
// (couplet/baselines.hpp; DESIGN.md Section 9, Q1).  The tests mirror
// python/tests/test_baselines.py, sections (a) to (f).
//
//   blocks       polynomial evaluation, tree expansion and H1 against the
//                scalar hash, the fused tree kernels against the plain
//                level-by-level form (one and several workers), input
//                authentication, Convert / Pack, BatchCheck (Figure 2 of
//                Zhang et al.), the unchecked COT extension.
//   vectors      values produced by the Python reference (identifiers, hash
//                domains, H0 / H1, one tree, the polynomial, a PlainCot
//                transcript).
//   leaky        Pi_DPF of Zhang et al. on ideal preprocessing for
//                n in {1, 2, 4, 8, 10, 12}: the outputs reconstruct to
//                unit(N, alpha) and unit(N, alpha) * beta, the MAC invariants
//                hold, bytes and rounds per phase equal the formulas.
//   real aBits   the same on the real preprocessing (base OT, extension) for
//                n in {4, 8, 10, 12} in one session.
//   selective    an additive error on one level's correction-word share,
//   failure      removed on a guessed path: over 200 runs the batch check
//                passes iff the most significant index bit equals the guess
//                (the one-bit leakage of F_DPF); further deviations; the same
//                attack on the real preprocessing retires the session.
//   honest       honest executions never abort (every honest execution of
//                this program is counted, plus 100 random ones).
//   semi-honest  Doerner-shelat-style baseline for n in {1, 4, 8, 10, 12}:
//                outputs, exact bytes and rounds; a deviation goes unnoticed.
//   separation   both parties' trees are recomputed from their key shares and
//                the public transcript: the hash domain of every key
//                generation is the one of ITS (sid, key number), in both
//                baselines.  (The mask r of the opened value is checked in
//                the leaky tests: non-zero, fresh, balanced; BatchCheck is
//                checked to use the powers of chi with a cancelling pair.)
//   threads      outputs and messages do not depend on the worker threads.
//   TCP          both protocols over a TCP connection on 127.0.0.1, with the
//                TEST-ONLY verification step of baselines::testonly.
//
// Both parties run as two threads of this process (run_two_party).  The test
// inputs (indices, payloads, guesses, injected errors) come from a seeded
// generator (--seed selects another sample); all protocol randomness comes
// from the operating system.
//
// Usage: test_baselines [--seed INTEGER]
//        test_baselines --interop zhang|sh ROLE PORT N1[,N2,...]
// Exit code 0 iff every check passes.  The --interop mode runs ONE party over
// TCP on 127.0.0.1 (party 0 listens, party 1 connects): a session with one key
// per listed n, followed by the TEST-ONLY verification step, and prints one
// JSON line with the bytes and rounds per phase, so that the run can be paired
// with the Python implementation of the same protocol
// (tests/interop_baselines.py).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "couplet/baselines.hpp"
#include "couplet/channel.hpp"
#include "couplet/gf128.hpp"
#include "couplet/hashes.hpp"
#include "couplet/prep.hpp"
#include "couplet/rng.hpp"
#include "couplet/sha256.hpp"
#include "couplet/shares.hpp"
#include "couplet/twoparty.hpp"

using namespace couplet;
using namespace couplet::baselines;
using prep::IdealDealer;
using prep::IdealPrep;
using prep::RealPrep;

// ------------------------------------------------------------ harness
static int g_pass = 0, g_fail = 0;
static int g_honest_runs = 0, g_honest_aborts = 0;
static std::vector<gf128> g_masks;  // the masks r of honest executions (reconstructed, test only)

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
static std::string hex(const bytes& b) { return hex(b.data(), b.size()); }
static std::string sha_hex(const bytes& b) {
    uint8_t h[32];
    sha256(b.data(), b.size(), h);
    return hex(h, 32);
}
static std::string str(uint64_t v) { return std::to_string(v); }

static const bytes TEST_SID = {'t', 'e', 's', 't', '-', 'b', 'a', 's', 'e', 'l', 'i', 'n', 'e', 's'};

// Test inputs only; all protocol randomness comes from the operating system.
static std::unique_ptr<CtrPrgRng> g_rng;
static gf128 rand_elem() { return g_rng->gf(); }
static uint64_t rand_below(uint64_t m) { return g_rng->u64() % m; }  // the small bias does not matter for test inputs
static uint64_t rand_bits(unsigned n) { return n == 0 ? 0 : (g_rng->u64() >> (64 - n)); }
static GfVec rand_vec(size_t n) {
    GfVec v(n);
    g_rng->fill_gf(v.data(), n);
    return v;
}
static BitVec rand_bitvec(size_t n) {
    BitVec b(n);
    g_rng->fill_bits(b.data(), n);
    return b;
}

static gf128 xor_all(const GfVec& v) {
    gf128 a;
    for (const gf128& e : v) a ^= e;
    return a;
}

// The first correction word of either baseline from the public offset W and the two key shares:
// CW^(0) = H0(Delta_0 + W) + H0(Delta_1 + W) + (alpha^(0) + 1) Delta  (the roots are Delta_b + W).
static bool level0_ok(const gf128& w, const gf128& cw0, const gf128& d0, const gf128& d1, unsigned top_bit, uint64_t dom) {
    return cw0 == (hashes::H_ccr(d0 ^ w, 0, dom) ^ hashes::H_ccr(d1 ^ w, 0, dom) ^ prep::gf_and_bit(d0 ^ d1, top_bit ^ 1));
}

// ------------------------------------------------------------ helpers of the leaky tests
struct OkFlags {
    bool u = false, u_mac = false, v = false, v_mac = false;
    bool all() const { return u && u_mac && v && v_mac; }
};

// (u = unit, MACs of u, v = unit * beta, MACs of v) for the two parties' outputs.
static OkFlags leaky_ok(const LeakyDpfOutput& o0, const LeakyDpfOutput& o1, unsigned n, uint64_t alpha, const gf128& beta,
                        const gf128& delta) {
    const size_t big_n = (size_t)1 << n;
    OkFlags f;
    if (o0.u.size() != big_n || o1.u.size() != big_n || o0.u.macs.size() != big_n || o1.u.macs.size() != big_n ||
        o0.v.values.size() != big_n || o1.v.values.size() != big_n || o0.v.macs.size() != big_n || o1.v.macs.size() != big_n)
        return f;
    const gf128 beta_delta = gf128_mul(beta, delta);
    f.u = f.u_mac = f.v = f.v_mac = true;
    for (size_t j = 0; j < big_n; j++) {
        const unsigned unit = (j == alpha) ? 1u : 0u;
        f.u = f.u && (unsigned)(o0.u.bits[j] ^ o1.u.bits[j]) == unit;
        f.u_mac = f.u_mac && (o0.u.macs[j] ^ o1.u.macs[j]) == prep::gf_and_bit(delta, unit);
        f.v = f.v && (o0.v.values[j] ^ o1.v.values[j]) == prep::gf_and_bit(beta, unit);
        f.v_mac = f.v_mac && (o0.v.macs[j] ^ o1.v.macs[j]) == prep::gf_and_bit(beta_delta, unit);
    }
    return f;
}

struct LeakyParty {
    LeakyDpfOutput out;
    PhaseStats st;
};

struct LeakyRun {
    bool passed = false;         // both parties completed
    bool protocol_abort = false; // a party threw ProtocolAbort
    std::string msg[2];          // exception texts
    LeakyDpfOutput out[2];
    PhaseStats st[2];
    ChannelStats ch[2];
    gf128 delta;                 // Delta_0 XOR Delta_1 (from the dealer)
    gf128 delta_share[2];
};

// One Pi_DPF execution on ideal preprocessing.  The index and the payload are
// XOR-shared at random.  Honest executions are tallied for the "honest" test.
static LeakyRun leaky_run(unsigned n, uint64_t alpha, const gf128& beta, const LeakyCheat* cheat0 = nullptr,
                          const LeakyCheat* cheat1 = nullptr, bool skip = false, uint64_t key_id = 0, unsigned threads0 = 1,
                          unsigned threads1 = 1, const bytes& sid = TEST_SID) {
    IdealDealer dealer;
    const uint64_t a0 = rand_bits(n);
    const gf128 b0 = rand_elem();
    const uint64_t a_share[2] = {a0, alpha ^ a0};
    const gf128 b_share[2] = {b0, beta ^ b0};
    IdealPrep prep0(0, dealer), prep1(1, dealer);
    IdealPrep* preps[2] = {&prep0, &prep1};
    const LeakyCheat* cheats[2] = {cheat0, cheat1};
    const unsigned threads[2] = {threads0, threads1};
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            LeakyParty p;
            Meter meter(ch);
            LeakyOptions opt;
            opt.key_id = key_id;
            opt.cheat = cheats[role];
            opt.skip_unused_coin = skip;
            opt.threads = threads[role];
            p.out = run_pi_dpf(ch, *preps[role], sid, n, a_share[role], b_share[role], opt, &meter);
            p.st = meter.stats();
            return p;
        };
    };
    auto f0 = party(0);
    auto f1 = party(1);
    auto res = run_two_party(f0, f1);
    const bool honest = !cheat0 && !cheat1;
    LeakyRun r;
    r.passed = res.first.ok && res.second.ok;
    r.protocol_abort = res.first.protocol_abort || res.second.protocol_abort;
    r.msg[0] = res.first.error;
    r.msg[1] = res.second.error;
    r.ch[0] = res.first.stats;
    r.ch[1] = res.second.stats;
    r.delta = dealer.delta();
    r.delta_share[0] = dealer.delta_share(0);
    r.delta_share[1] = dealer.delta_share(1);
    if (r.passed) {
        r.out[0] = std::move(res.first.value.out);
        r.out[1] = std::move(res.second.value.out);
        r.st[0] = res.first.value.st;
        r.st[1] = res.second.value.st;
    }
    if (honest) {
        g_honest_runs++;
        if (!r.passed) g_honest_aborts++;
    }
    return r;
}

struct AttackResult {
    bool passed = false;  // the check passed at both parties
    bool good = false;    // outputs correct and authenticated
    bool batch_check = false;  // the honest party aborted with "BatchCheck failed"
};

// Run with a cheating party.
static AttackResult attack_run(unsigned n, uint64_t alpha, const gf128& beta, int cheater, const LeakyCheat& hook) {
    const LeakyRun r = leaky_run(n, alpha, beta, cheater == 0 ? &hook : nullptr, cheater == 1 ? &hook : nullptr);
    AttackResult a;
    a.passed = r.passed;
    if (r.passed) a.good = leaky_ok(r.out[0], r.out[1], n, alpha, beta, r.delta).all();
    a.batch_check = r.msg[1 - cheater] == "BatchCheck failed";
    return a;
}

static bool leaky_phase_counts_match(const PhaseStats& st, unsigned n, bool skip) {
    bool ok = true;
    for (int p = BP_ZH_INPUT; p <= BP_ZH_CHECK; p++) {
        ok = ok && st[p].bytes_sent == leaky_bytes_per_party(n, p, skip) && st[p].bytes_recv == st[p].bytes_sent &&
             st[p].rounds == leaky_rounds(n, p, skip);
    }
    for (int p = BP_SH_BASEOT; p < BP_COUNT; p++) ok = ok && st[p].bytes_sent == 0 && st[p].rounds == 0;
    return ok;
}

// ------------------------------------------------------------ (a) building blocks
// Elements / bits of the deterministic stream that the Python vector script uses.
static const uint64_t DOM_TEST = hashes::dom::PRG_TEST;
static hashes::seed16 kat_seed() {
    hashes::seed16 s;
    for (size_t i = 0; i < 16; i++) s[i] = (uint8_t)i;
    return s;
}
static GfVec kat_gf(size_t n, uint64_t start_block) {
    const std::vector<uint64_t> w = hashes::ctr_prg_words(kat_seed(), DOM_TEST, 2 * n, start_block);
    GfVec v(n);
    for (size_t i = 0; i < n; i++) v[i] = gf128(w[2 * i], w[2 * i + 1]);
    return v;
}
static BitVec kat_bits(size_t n, uint64_t start_block) {
    const std::vector<uint64_t> w = hashes::ctr_prg_words(kat_seed(), DOM_TEST, (n + 63) / 64, start_block);
    BitVec b(n);
    for (size_t i = 0; i < n; i++) b[i] = (uint8_t)((w[i / 64] >> (i % 64)) & 1);
    return b;
}

static gf128 poly_ref(const GfVec& c, const gf128& x) {
    gf128 acc, power = gf128::one();
    for (const gf128& e : c) {
        acc ^= gf128_mul(power, e);
        power = gf128_mul(power, x);
    }
    return acc;
}

// The mask r of step 4: the opened a XOR the unmasked combination of the
// reconstructed outputs (which no party sees).  a = sum_j chi^j u^(j) +
// sum_j chi^(N+j) v^(j) + r; if a were not masked it would reveal alpha and
// beta: a = chi^alpha + chi^(N+alpha) beta.
static gf128 opened_mask(const LeakyDpfOutput& o0, const LeakyDpfOutput& o1) {
    const size_t big_n = o0.u.size();
    GfVec coeffs(2 * big_n);
    for (size_t j = 0; j < big_n; j++) {
        coeffs[j] = gf128((uint64_t)(o0.u.bits[j] ^ o1.u.bits[j]), 0);
        coeffs[big_n + j] = o0.v.values[j] ^ o1.v.values[j];
    }
    return o0.transcript.a ^ poly_ref(coeffs, o0.transcript.chi);
}

static void check_masked(const LeakyDpfOutput& o0, const LeakyDpfOutput& o1, const std::string& tag) {
    const gf128 mask = opened_mask(o0, o1);
    g_masks.push_back(mask);
    check(!mask.is_zero(), tag + ": the opened value a is masked (a differs from the unmasked combination)");
}

// The leaf strings (s || t)^(n, j) of one party from its key share and the
// public transcript, in the plain level-by-level form.
static GfVec recompute_leaves(const gf128& delta_b, const gf128& w, const GfVec& cws, uint64_t dom) {
    GfVec nodes(1, delta_b ^ w);
    for (const gf128& cw : cws) nodes = expand_level(nodes, hash0(nodes, dom), cw);
    return nodes;
}

static unsigned weight(const gf128& x) {
    unsigned w = 0;
    for (unsigned i = 0; i < 128; i++) w += x.bit(i);
    return w;
}

// The tree of one party in the plain form of the Python reference, compared
// with the fused kernels of baselines.hpp.
static void check_tree_kernels(unsigned n, unsigned workers) {
    const prep::WorkerScope scope(workers);
    const uint64_t dom = h0_domain(leaky_sub_sid(TEST_SID, 77));
    const gf128 root = rand_elem();
    const GfVec cws = rand_vec(n);
    const std::string tag = "tree kernels, n = " + str(n) + ", " + str(workers) + " worker(s)";

    // reference
    GfVec level(1, root);
    GfVec ref_acc;
    for (unsigned i = 0; i < n; i++) {
        const GfVec hashed = hash0(level, dom);
        ref_acc.push_back(xor_all(hashed));
        level = expand_level(level, hashed, cws[i]);
    }
    GfVec ref_hv, ref_hm;
    hash1(level, dom, ref_hv, ref_hm);

    // kernels
    gf128 acc;
    baselines::detail::WordBuffer raw = baselines::detail::root_pair(root, dom, acc);
    size_t m = 1;
    bool acc_ok = acc == ref_acc[0];
    for (unsigned i = 1; i < n; i++) {
        baselines::detail::WordBuffer next(8 * m);
        acc = baselines::detail::level_pass(raw.data(), m, cws[i - 1], dom, next.data());
        raw.swap(next);
        m *= 2;
        acc_ok = acc_ok && acc == ref_acc[i];
    }
    check(acc_ok, tag + ": XOR of H0 over every level");
    const size_t big_n = (size_t)1 << n;
    GfVec leaves(big_n), hv(big_n), hm(big_n);
    BitVec tb(big_n);
    gf128 av, am;
    baselines::detail::leaky_leaf_pass(raw.data(), m, cws[n - 1], dom, leaves.data(), tb.data(), hv.data(), hm.data(), av, am);
    bool t_ok = true;
    for (size_t j = 0; j < big_n; j++) t_ok = t_ok && tb[j] == (uint8_t)(level[j].lo & 1);
    check(leaves == level && t_ok, tag + ": leaves and control bits");
    check(hv == ref_hv && hm == ref_hm && av == xor_all(ref_hv) && am == xor_all(ref_hm), tag + ": H1 of the leaves and its sums");
    GfVec hv2(big_n);
    BitVec tb2(big_n);
    gf128 av2;
    baselines::detail::semihonest_leaf_pass(raw.data(), m, cws[n - 1], dom, tb2.data(), hv2.data(), av2);
    check(hv2 == ref_hv && tb2 == tb && av2 == av, tag + ": semi-honest leaf pass");
    const gf128 c = rand_elem();
    GfVec sel = hv;
    baselines::detail::add_selected(sel.data(), tb.data(), big_n, c);
    bool sel_ok = true;
    for (size_t j = 0; j < big_n; j++) sel_ok = sel_ok && sel[j] == (hv[j] ^ prep::gf_and_bit(c, tb[j]));
    check(sel_ok, tag + ": selected addition of the final word");
}

struct BcResult {
    bool pass = false, abort = false;
    ChannelStats st;
};
// BatchCheck between two threads; tamper: 0 none, 1 both hold the same wrong
// opened value, 2 a wrong MAC share, 3 the parties disagree on the opened value.
static BcResult bc_run(IdealDealer& dealer, size_t count, int tamper, bool skip) {
    const gf128 delta = dealer.delta();
    GfVec values = rand_vec(count), mac0 = rand_vec(count), mac1(count);
    for (size_t i = 0; i < count; i++) mac1[i] = mac0[i] ^ gf128_mul(values[i], delta);
    GfVec opened[2] = {values, values};
    GfVec macs[2] = {mac0, mac1};
    if (tamper == 1) {
        opened[0].back().lo ^= 1;
        opened[1].back().lo ^= 1;
    } else if (tamper == 2) {
        macs[1][0].hi ^= (uint64_t)1 << 13;
    } else if (tamper == 3) {
        opened[1][0].lo ^= 2;
    } else if (tamper == 4) {  // the same error on values 1 and 2 (cancels unless chi^1 != chi^2)
        for (GfVec& view : opened) {
            view[1].lo ^= 0xE44;
            view[2].lo ^= 0xE44;
        }
    } else if (tamper == 5) {  // the same error on the last two values
        for (GfVec& view : opened) {
            view[count - 1].hi ^= (uint64_t)1 << 36;
            view[count - 2].hi ^= (uint64_t)1 << 36;
        }
    }
    const bytes sid = {'b', 'c', '-', 's', 'i', 'd'};
    const bytes label = {'t'};
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            SystemRng sys;
            batch_check(ch, sid, label, dealer.delta_share(role), opened[role], macs[role], sys, skip);
            return 1;
        };
    };
    auto f0 = party(0);
    auto f1 = party(1);
    auto res = run_two_party(f0, f1);
    BcResult r;
    r.pass = res.first.ok && res.second.ok;
    r.abort = res.first.protocol_abort && res.second.protocol_abort && res.first.error == "BatchCheck failed" &&
              res.second.error == "BatchCheck failed";
    r.st = res.first.stats;
    return r;
}

// Seeds of both COT directions as a trusted base OT would give them.
struct TrustedSeeds {
    GfVec s0, s1;  // the pairs
    GfVec chosen(const gf128& delta) const {
        GfVec c(KAPPA);
        for (size_t j = 0; j < KAPPA; j++) c[j] = delta.bit((unsigned)j) ? s1[j] : s0[j];
        return c;
    }
};

static void test_building_blocks() {
    Section sec("building blocks");

    // poly_eval against the definition (the sizes cross the block length of the power table)
    for (size_t m : {(size_t)0, (size_t)1, (size_t)2, (size_t)3, (size_t)5, (size_t)48, (size_t)49, (size_t)64, (size_t)100,
                     (size_t)255, (size_t)256, (size_t)257, (size_t)512, (size_t)513, (size_t)1000, (size_t)40000}) {
        const GfVec coeffs = rand_vec(m);
        const gf128 x = rand_elem();
        const gf128 expect = poly_ref(coeffs, x);
        check(poly_eval(coeffs, x) == expect, "poly_eval, " + str(m) + " coefficients");
        {
            const prep::WorkerScope scope(3);
            check(poly_eval(coeffs, x) == expect, "poly_eval, " + str(m) + " coefficients, 3 workers");
        }
        const BitVec bits = rand_bitvec(m);
        GfVec as_field(m);
        for (size_t i = 0; i < m; i++) as_field[i] = gf128((uint64_t)bits[i], 0);
        check(poly_eval_bits(bits.data(), m, x) == poly_ref(as_field, x), "poly_eval_bits, " + str(m) + " coefficients");
        {
            const prep::WorkerScope scope(4);
            check(poly_eval_bits(bits.data(), m, x) == poly_ref(as_field, x), "poly_eval_bits, " + str(m) + " coefficients, 4 workers");
        }
    }
    {
        const GfVec coeffs = rand_vec(8);
        check(poly_eval(coeffs, gf128()) == coeffs[0], "poly_eval at 0 is the constant term");
        check(poly_eval(coeffs, gf128::one()) == xor_all(coeffs), "poly_eval at 1 is the sum");
        const gf128 x = rand_elem();
        gf128 p = gf128::one();
        for (int i = 0; i < 77; i++) p = gf128_mul(p, x);
        check(gf_pow(x, 77) == p && gf_pow(x, 0) == gf128::one() && gf_pow(x, 1) == x, "gf_pow against repeated multiplication");
    }

    // index bits: most significant first
    check(index_bits(0b1011, 4) == BitVec({1, 0, 1, 1}), "index_bits is MSB first");
    check(index_bits(1, 3) == BitVec({0, 0, 1}), "index_bits pads on the left");
    check(throws<std::invalid_argument>([] { index_bits(8, 3); }), "index_bits rejects an index >= 2^n");

    // H0 / H1 / expansion against the scalar hash
    const bytes ssid = leaky_sub_sid(TEST_SID, 3);
    const uint64_t dom = h0_domain(ssid);
    check((dom >> 56) == 0x48, "H0 domain is in the CCR family (top byte 0x48)");
    check(h0_domain(leaky_sub_sid(TEST_SID, 4)) != dom, "H0 domain depends on the key identifier");
    check(h0_domain(ssid, label_h0_semihonest()) != dom, "the two baselines use different H0 domains");
    {
        const GfVec nodes = rand_vec(9);
        const GfVec hashed = hash0(nodes, dom);
        bool ok = true;
        for (size_t i = 0; i < nodes.size(); i++) ok = ok && hashed[i] == hashes::H_ccr(nodes[i], 0, dom);
        check(ok, "hash0 is H_ccr with tweak 0");
        GfVec hv, hm;
        hash1(nodes, dom, hv, hm);
        ok = true;
        for (size_t i = 0; i < nodes.size(); i++)
            ok = ok && hv[i] == hashes::H_ccr(nodes[i], 0, dom) && hm[i] == hashes::H_ccr(gf128(nodes[i].lo ^ 1, nodes[i].hi), 0, dom);
        check(ok, "hash1(x) = H0(x) || H0(x XOR 1)");
        const gf128 cw = rand_elem();
        const GfVec kids = expand_level(nodes, hashed, cw);
        ok = kids.size() == 2 * nodes.size();
        for (size_t i = 0; ok && i < nodes.size(); i++) {
            const gf128 left = hashes::H_ccr(nodes[i], 0, dom) ^ ((nodes[i].lo & 1) ? cw : gf128());
            ok = kids[2 * i] == left && kids[2 * i + 1] == (left ^ nodes[i]);
        }
        check(ok, "expand_level: left = H0(x) + t CW, right = H0(x) + x + t CW");
    }

    // the level invariant of Figure 5 on a hand-made node pair
    {
        gf128 delta0 = rand_elem(), delta1 = rand_elem();
        delta0.lo |= 1;
        delta1.lo &= ~1ull;
        const gf128 delta = delta0 ^ delta1;
        for (unsigned bit = 0; bit < 2; bit++) {
            const gf128 x0 = rand_elem(), x1 = x0 ^ delta;
            const GfVec pair0(1, x0), pair1(1, x1);
            const GfVec h0 = hash0(pair0, dom), h1 = hash0(pair1, dom);
            const gf128 cw = h0[0] ^ h1[0] ^ prep::gf_and_bit(delta, bit ^ 1);
            const GfVec k0 = expand_level(pair0, h0, cw), k1 = expand_level(pair1, h1, cw);
            check((k0[bit] ^ k1[bit]) == delta && (k0[1 - bit] ^ k1[1 - bit]).is_zero(),
                  "level invariant: the child on side alpha^(i) = " + str(bit) + " differs by Delta, the other by 0");
        }
    }

    // the fused tree kernels against the plain form
    for (unsigned n = 1; n <= 11; n++) check_tree_kernels(n, 1);
    check_tree_kernels(14, 1);
    check_tree_kernels(14, 3);
    check_tree_kernels(15, 4);

    // Convert, Pack and the input authentication on ideal BDOZ bits
    {
        IdealDealer dealer;
        const gf128 d0 = dealer.delta_share(0), d1 = dealer.delta_share(1);
        const size_t n_bits = 128 + 11;
        const BitVec bits[2] = {rand_bitvec(n_bits), rand_bitvec(n_bits)};
        IdealPrep p0(0, dealer), p1(1, dealer);
        IdealPrep* preps[2] = {&p0, &p1};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                const prep::BdozPair pair = preps[role]->random_bdoz(n_bits);
                return authenticate_inputs(ch, dealer.delta_share(role), pair.own, pair.keys, bits[role]);
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        auto res = run_two_party(f0, f1);
        check(res.first.ok && res.second.ok, "authenticate_inputs completes");
        const BdozBits& x0 = res.first.value;
        const BdozBits& x1 = res.second.value;
        check(x0.bits == bits[0] && x1.bits == bits[1], "authenticate_inputs keeps the input bits as shares");
        bool ok = x0.size() == n_bits && x1.size() == n_bits;
        for (size_t i = 0; ok && i < n_bits; i++)
            ok = x0.macs[i] == (x1.keys[i] ^ prep::gf_and_bit(d1, x0.bits[i])) && x1.macs[i] == (x0.keys[i] ^ prep::gf_and_bit(d0, x1.bits[i]));
        check(ok, "authenticate_inputs: M_b[x_b] = K_{1-b}[x_b] + x_b Delta_{1-b}");
        check(res.first.stats.bytes_sent == (n_bits + 7) / 8 && res.second.stats.bytes_sent == (n_bits + 7) / 8 &&
                  res.first.stats.rounds == 1,
              "authenticate_inputs: one bit per input bit, one round");
        const prep::Shares c0 = convert(d0, x0), c1 = convert(d1, x1);
        ok = true;
        for (size_t i = 0; i < n_bits; i++) {
            const unsigned x = (unsigned)(bits[0][i] ^ bits[1][i]);
            ok = ok && (unsigned)(c0.bits[i] ^ c1.bits[i]) == x && (c0.macs[i] ^ c1.macs[i]) == prep::gf_and_bit(d0 ^ d1, x);
        }
        check(ok, "Convert: SPDZ-style shares of x_0 + x_1 with MAC shares of x Delta");
        const FieldShare f_0 = pack(d0, x0.slice(0, 128)), f_1 = pack(d1, x1.slice(0, 128));
        BitVec x(128);
        for (size_t i = 0; i < 128; i++) x[i] = (uint8_t)(bits[0][i] ^ bits[1][i]);
        const gf128 packed = prep::gf_pack_bits(x.data());
        check((f_0.value ^ f_1.value) == packed && (f_0.mac ^ f_1.mac) == gf128_mul(packed, d0 ^ d1), "Pack: <sum_i X^i x_i> with a valid MAC");
        check(x0.slice(3, 4).size() == 1 && x0.slice(5, 9).size() == 4 && x0.slice(5, 9).keys[0] == x0.keys[5], "BdozBits slicing");
        check(throws<std::invalid_argument>([&] { BdozBits bad(bits[0], GfVec(5), x0.macs); }), "BdozBits length check");
        check(throws<std::invalid_argument>([&] { pack(d0, x0.slice(0, 127)); }), "Pack needs exactly 128 bits");
    }

    // BatchCheck (Figure 2)
    {
        IdealDealer dealer;
        for (size_t count : {(size_t)1, (size_t)2, (size_t)3, (size_t)5}) {
            const BcResult r = bc_run(dealer, count, 0, false);
            check(r.pass, "BatchCheck passes on " + str(count) + " honest opening(s)");
            check(r.st.bytes_sent == 96 + 80 && r.st.rounds == 4, "BatchCheck on " + str(count) + " value(s): 176 bytes per party, 4 rounds");
            for (int tamper = 1; tamper <= (count >= 3 ? 5 : 3); tamper++)
                check(bc_run(dealer, count, tamper, false).abort, "BatchCheck aborts (" + str(count) + " value(s), tampering " + str((uint64_t)tamper) + ")");
        }
        const BcResult r1 = bc_run(dealer, 1, 0, true);
        check(r1.pass && r1.st.bytes_sent == 80 && r1.st.rounds == 2, "BatchCheck, single value without the unused coin: 80 bytes, 2 rounds");
        check(bc_run(dealer, 1, 1, true).abort, "BatchCheck without the coin still aborts on a wrong opening");
        const BcResult r3 = bc_run(dealer, 3, 0, true);
        check(r3.pass && r3.st.rounds == 4, "the coin is skipped only for a single value");
        auto pair = LoopbackChannel::make_pair();
        SystemRng sys;
        check(throws<std::invalid_argument>([&] { batch_check(*pair.first, TEST_SID, TEST_SID, dealer.delta_share(0), GfVec(), GfVec(), sys); }) &&
                  pair.first->stats().msgs_sent == 0,
              "BatchCheck needs at least one value");
    }

    // Unchecked COT extension (semi-honest baseline) on trusted base-OT output
    {
        gf128 d0 = rand_elem(), d1 = rand_elem();
        d0.lo |= 1;
        d1.lo &= ~1ull;
        TrustedSeeds dir0{rand_vec(KAPPA), rand_vec(KAPPA)};  // direction 0: P_0 sends, P_1 holds the pairs
        TrustedSeeds dir1{rand_vec(KAPPA), rand_vec(KAPPA)};  // direction 1: P_1 sends, P_0 holds the pairs
        PlainCot cot0(0, d0, dir0.chosen(d0), dir1.s0, dir1.s1);
        PlainCot cot1(1, d1, dir1.chosen(d1), dir0.s0, dir0.s1);
        gf128 previous;
        bool have_previous = false;
        for (size_t ell : {(size_t)1, (size_t)20, (size_t)64, (size_t)65, (size_t)200}) {
            const BitVec u0 = rand_bitvec(ell), u1 = rand_bitvec(ell);
            const bytes m0 = cot0.start(u0), m1 = cot1.start(u1);
            check(m0.size() == cot_message_bytes(ell) && m0.size() == 1024 * ((ell + 63) / 64) && m1.size() == m0.size(),
                  "plain COT, " + str(ell) + " rows: U message is 16 bytes per 64-row-rounded row");
            GfVec mac0, key0, mac1, key1;
            cot0.finish(m1, mac0, key0);
            cot1.finish(m0, mac1, key1);
            bool ok = mac0.size() == ell && key0.size() == ell && mac1.size() == ell && key1.size() == ell;
            for (size_t i = 0; ok && i < ell; i++)
                ok = mac0[i] == (key1[i] ^ prep::gf_and_bit(d1, u0[i])) && mac1[i] == (key0[i] ^ prep::gf_and_bit(d0, u1[i]));
            check(ok, "plain COT, " + str(ell) + " rows: M = K + u Delta in both directions");
            if (have_previous) check(previous != key0[0], "plain COT: every call reads a fresh stream");
            previous = key0[0];
            have_previous = true;
        }
        check(cot0.calls() == 5 && cot1.calls() == 5, "plain COT: one call counter step per extension");
        GfVec a, b;
        check(throws<std::logic_error>([&] { cot0.finish(bytes(), a, b); }), "plain COT: finish without start is refused");
        cot0.start(BitVec({1, 0, 1}));
        check(throws<ProtocolAbort>([&] { cot0.finish(bytes(7, 0), a, b); }), "plain COT: malformed U aborts");
        check(throws<std::invalid_argument>([&] { PlainCot bad(0, d0, GfVec(5), dir1.s0, dir1.s1); }), "plain COT: exactly 128 seeds per direction");
    }

    // formulas
    for (unsigned n : {1u, 8u, 12u, 20u, 24u, 26u}) {
        check(leaky_bytes_per_party(n) == 16 * (uint64_t)n + (n + 135) / 8 + 416 && leaky_rounds(n) == (uint64_t)n + 11 &&
                  leaky_bytes_per_party(n, true) == 16 * (uint64_t)n + (n + 135) / 8 + 320 && leaky_rounds(n, true) == (uint64_t)n + 9,
              "leaky n = " + str(n) + ": 16 n + ceil((n + 128) / 8) + 416 bytes per party and n + 11 rounds per key (320 / n + 9 without the coin)");
        check(semihonest_bytes_per_party(n) == 16 * (uint64_t)n + 1024 + 32 && semihonest_rounds(n) == (uint64_t)n + 2 &&
                  semihonest_rounds(n, BP_SH_COT) + semihonest_rounds(n, BP_SH_LEVELS) + semihonest_rounds(n, BP_SH_LEAF) == (uint64_t)n + 2,
              "semi-honest n = " + str(n) + ": 16 n + 1056 bytes per party and n + 2 rounds per key");
    }
    check(abit_demand(20) == 276, "n + 2 kappa authenticated bits per party and key");
}

// ------------------------------------------------------------ (a') vectors of the Python reference
static void test_python_vectors() {
    Section sec("vectors of the Python reference");
    const bytes ssid = leaky_sub_sid(TEST_SID, 3);
    check(hex(ssid) == "9ebf43319e59f12c2478f163f1cac91fb8007020c6adf14b6bc9031f8b200635", "identifier of leaky key 3");
    const uint64_t dom = h0_domain(ssid);
    check(dom == 0x4803bd85dc684e47ull, "H0 domain of the leaky baseline");
    const bytes ssid2 = semihonest_sub_sid(TEST_SID, 7);
    check(hex(ssid2) == "9efafe7c8dec2139c4684aa89b1066912ffbd0260cb3a3d3d1692dc5540de5b1", "identifier of semi-honest key 7");
    check(h0_domain(ssid2, label_h0_semihonest()) == 0x48b0caba73d21dd5ull, "H0 domain of the semi-honest baseline");

    const GfVec nodes = kat_gf(9, 0);
    const GfVec hashed = hash0(nodes, dom);
    check(sha_hex(prep::gf_to_bytes(hashed)) == "0d81e76f22721b7d14f8e14c7afcdf691ea1bedf823a539c26e06c8ca30b0f38", "H0 of nine elements");
    GfVec hv, hm;
    hash1(nodes, dom, hv, hm);
    bytes both = prep::gf_to_bytes(hv);
    const bytes hm_b = prep::gf_to_bytes(hm);
    both.insert(both.end(), hm_b.begin(), hm_b.end());
    check(sha_hex(both) == "f33cb86110f13f00c0d3fe51d87a573cbc63006156701045cdcea6a713b1d209", "H1 of nine elements");
    const gf128 cw = kat_gf(1, 100)[0];
    check(cw.hex() == "2d3d00826567f7d35faf83e6511176da", "the test stream agrees with the Python stream");
    check(sha_hex(prep::gf_to_bytes(expand_level(nodes, hashed, cw))) == "8d7f9781a521e73aaa61cb613ac9c1d2a013d518d5aaad8ecc18c9eb76ec6bb1",
          "expand_level of nine nodes");

    const GfVec coeffs = kat_gf(1000, 200);
    const gf128 x = kat_gf(1, 1000)[0];
    check(x.hex() == "36db3261be05cbb09378b0a8c89a153c" && poly_eval(coeffs, x).hex() == "ee2dc0d3907566d17df5b5ce1f09b484",
          "poly_eval of 1000 coefficients");

    // one tree with fixed root and correction words (n = 5), through the fused kernels
    {
        const unsigned n = 5;
        const gf128 root = kat_gf(1, 2000)[0];
        const GfVec cws = kat_gf(n, 2100);
        check(root.hex() == "e0572bef723ad27d66b1f1e6641d653c", "root of the vector tree");
        gf128 acc;
        baselines::detail::WordBuffer raw = baselines::detail::root_pair(root, dom, acc);
        GfVec accs(1, acc);
        size_t m = 1;
        for (unsigned i = 1; i < n; i++) {
            baselines::detail::WordBuffer next(8 * m);
            accs.push_back(baselines::detail::level_pass(raw.data(), m, cws[i - 1], dom, next.data()));
            raw.swap(next);
            m *= 2;
        }
        check(sha_hex(prep::gf_to_bytes(accs)) == "e8db22eff7a006042f64ef40a2faa24ac415a16cae4764e47a21371d1a9a73e3",
              "tree: XOR of H0 over each of the five levels");
        const size_t big_n = (size_t)1 << n;
        GfVec leaves(big_n), lv(big_n), lm(big_n);
        BitVec tb(big_n);
        gf128 av, am;
        baselines::detail::leaky_leaf_pass(raw.data(), m, cws[n - 1], dom, leaves.data(), tb.data(), lv.data(), lm.data(), av, am);
        bytes all = prep::gf_to_bytes(leaves);
        const bytes b1 = prep::gf_to_bytes(lv), b2 = prep::gf_to_bytes(lm);
        all.insert(all.end(), b1.begin(), b1.end());
        all.insert(all.end(), b2.begin(), b2.end());
        check(sha_hex(all) == "1b5be35d7d2c6442019f4115091d605c9de59348e7a376dd1a8737f048538ec3", "tree: leaves and H1 of the leaves");
    }

    // PlainCot with fixed seeds: both U messages and all rows of two calls
    {
        TrustedSeeds dir0{kat_gf(128, 3000), kat_gf(128, 3200)};
        TrustedSeeds dir1{kat_gf(128, 3400), kat_gf(128, 3600)};
        gf128 d0 = kat_gf(1, 3800)[0], d1 = kat_gf(1, 3900)[0];
        d0.lo |= 1;
        d1.lo &= ~1ull;
        check(d0.hex() == "6c51a874c80e0bb4131e032d6d0e69bd" && d1.hex() == "8348035c849f0c16ab22a758f13f1df6", "key shares of the PlainCot vector");
        PlainCot cot0(0, d0, dir0.chosen(d0), dir1.s0, dir1.s1);
        PlainCot cot1(1, d1, dir1.chosen(d1), dir0.s0, dir0.s1);
        bytes all;
        auto add = [&all](const bytes& b) { all.insert(all.end(), b.begin(), b.end()); };
        for (size_t ell : {(size_t)5, (size_t)70}) {
            const BitVec bits = kat_bits(2 * ell, 4000 + ell);
            const BitVec u0(bits.begin(), bits.begin() + (ptrdiff_t)ell), u1(bits.begin() + (ptrdiff_t)ell, bits.end());
            const bytes m0 = cot0.start(u0), m1 = cot1.start(u1);
            add(m0);
            add(m1);
            GfVec mac0, key0, mac1, key1;
            cot0.finish(m1, mac0, key0);
            cot1.finish(m0, mac1, key1);
            add(prep::gf_to_bytes(mac0));
            add(prep::gf_to_bytes(key0));
            add(prep::gf_to_bytes(mac1));
            add(prep::gf_to_bytes(key1));
        }
        check(sha_hex(all) == "11e33d458aab50802be4a573ff6086c0121065983a7668ca0725a4496f5b75f9", "PlainCot: U messages and rows of two calls");
    }
}

// ------------------------------------------------------------ (b) leaky protocol on ideal preprocessing
static void test_leaky_correctness() {
    Section sec("leaky protocol, ideal preprocessing, n in {1, 2, 4, 8, 10, 12}");
    for (unsigned n : {1u, 2u, 4u, 8u, 10u, 12u}) {
        const uint64_t size = (uint64_t)1 << n;
        std::vector<uint64_t> alphas = {0, size - 1, rand_below(size), rand_below(size)};
        std::sort(alphas.begin(), alphas.end());
        alphas.erase(std::unique(alphas.begin(), alphas.end()), alphas.end());
        LeakyRun last;
        for (uint64_t alpha : alphas) {
            const gf128 betas[3] = {rand_elem(), gf128(), gf128::one()};
            for (int bi = 0; bi < 3; bi++) {
                const gf128& beta = betas[bi];
                LeakyRun r = leaky_run(n, alpha, beta);
                const std::string tag = "leaky n = " + str(n) + ", alpha = " + str(alpha) + ", beta " + (bi == 0 ? "random" : bi == 1 ? "0" : "1");
                check(r.passed, tag + ": completes (" + r.msg[0] + " / " + r.msg[1] + ")");
                const OkFlags ok = leaky_ok(r.out[0], r.out[1], n, alpha, beta, r.delta);
                check(ok.u, tag + ": u_0 + u_1 = unit(N, alpha)");
                check(ok.u_mac, tag + ": M_0[u] + M_1[u] = u Delta");
                check(ok.v, tag + ": v_0 + v_1 = unit(N, alpha) * beta");
                check(ok.v_mac, tag + ": M_0[v] + M_1[v] = v Delta");
                check(r.passed && r.out[0].transcript == r.out[1].transcript, tag + ": both parties see one transcript");
                if (r.passed) check_masked(r.out[0], r.out[1], tag);
                check(r.passed && level0_ok(r.out[0].transcript.W, r.out[0].transcript.cws[0], r.delta_share[0], r.delta_share[1],
                                            (unsigned)(alpha >> (n - 1)), h0_domain(leaky_sub_sid(TEST_SID, 0))),
                      tag + ": CW^(0) = H0(Delta_0 + W) + H0(Delta_1 + W) + (alpha^(0) + 1) Delta");
                last = std::move(r);
            }
        }
        // accounting of the last run
        for (int role = 0; role < 2; role++) {
            check(leaky_phase_counts_match(last.st[role], n, false), "leaky n = " + str(n) + ": bytes and rounds per phase are exact (party " + str((uint64_t)role) + ")");
        }
        check(last.ch[0].bytes_sent == 16 * (uint64_t)n + (n + 135) / 8 + 416 && last.ch[0].bytes_sent == leaky_bytes_per_party(n) &&
                  last.ch[1].bytes_sent == last.ch[0].bytes_sent && last.ch[0].bytes_recv == last.ch[1].bytes_sent,
              "leaky n = " + str(n) + ": 16 n + ceil((n + 128) / 8) + 416 bytes per party");
        check(last.ch[0].rounds == (uint64_t)n + 11 && last.ch[1].rounds == (uint64_t)n + 11, "leaky n = " + str(n) + ": n + 11 rounds");
        check(last.out[0].transcript.cws.size() == n && last.out[0].u.size() == size && last.out[0].v.size() == size && last.out[0].n == n,
              "leaky n = " + str(n) + ": n correction words, N outputs");

        // variant without the unused BatchCheck coin
        const uint64_t alpha = rand_below(size);
        const gf128 beta = rand_elem();
        const LeakyRun r = leaky_run(n, alpha, beta, nullptr, nullptr, true);
        check(r.passed && leaky_ok(r.out[0], r.out[1], n, alpha, beta, r.delta).all(),
              "leaky n = " + str(n) + ", no unused coin: outputs correct and authenticated");
        check(leaky_phase_counts_match(r.st[0], n, true) && leaky_phase_counts_match(r.st[1], n, true) &&
                  r.ch[0].bytes_sent == 16 * (uint64_t)n + (n + 135) / 8 + 320,
              "leaky n = " + str(n) + ", no unused coin: 96 bytes per party less");
        check(r.ch[0].rounds == (uint64_t)n + 9, "leaky n = " + str(n) + ", no unused coin: n + 9 rounds");
    }

    // shares of u are ordinary authenticated bits of our layer: slicing and XOR stay valid
    {
        const unsigned n = 5;
        const LeakyRun r = leaky_run(n, 3, rand_elem());
        bool ok = r.passed;
        if (ok) {
            const prep::Shares s0 = r.out[0].u.slice(2, 7) ^ r.out[0].u.slice(1, 6), s1 = r.out[1].u.slice(2, 7) ^ r.out[1].u.slice(1, 6);
            ok = s0.size() == 5;
            for (size_t i = 0; ok && i < 5; i++) {
                const unsigned bit = (unsigned)(s0.bits[i] ^ s1.bits[i]);
                ok = bit == ((2 + i == 3) != (1 + i == 3) ? 1u : 0u) && (s0.macs[i] ^ s1.macs[i]) == prep::gf_and_bit(r.delta, bit);
            }
        }
        check(ok, "u is a prep::Shares object (slicing, XOR keep the MAC invariant)");
    }

    // two keys under one identifier but different key numbers use different coins and domains
    {
        const LeakyRun o1 = leaky_run(4, 5, gf128(7, 0), nullptr, nullptr, false, 0);
        const LeakyRun o2 = leaky_run(4, 5, gf128(7, 0), nullptr, nullptr, false, 1);
        check(o1.passed && o2.passed && o1.out[0].transcript.W != o2.out[0].transcript.W, "fresh public randomness W per key");
    }

    // argument validation happens before any message
    {
        IdealDealer dealer;
        IdealPrep prep0(0, dealer);
        auto pair = LoopbackChannel::make_pair();
        pair.second.reset();  // no peer: a call that got past the validation fails instead of blocking
        Channel& ch = *pair.first;
        check(throws<std::invalid_argument>([&] { run_pi_dpf(ch, prep0, TEST_SID, 0, 0, gf128()); }), "n = 0 is rejected");
        check(throws<std::invalid_argument>([&] { run_pi_dpf(ch, prep0, TEST_SID, MAX_N + 1, 0, gf128()); }), "n above MAX_N is rejected");
        check(throws<std::invalid_argument>([&] { run_pi_dpf(ch, prep0, TEST_SID, 3, 8, gf128()); }), "an index share >= 2^n is rejected");
        const LeakyCheat bad_level = LeakyCheat::cw(3, gf128::one());
        LeakyOptions opt;
        opt.cheat = &bad_level;
        check(throws<std::invalid_argument>([&] { run_pi_dpf(ch, prep0, TEST_SID, 3, 1, gf128::one(), opt); }), "a malformed test hook is rejected");
        const LeakyCheat bad_guess = LeakyCheat::cw(1, gf128::one(), 2);
        opt.cheat = &bad_guess;
        check(throws<std::invalid_argument>([&] { run_pi_dpf(ch, prep0, TEST_SID, 3, 1, gf128::one(), opt); }), "a guess outside the level is rejected");
        check(!prep0.aborted() && ch.stats().msgs_sent == 0, "invalid arguments send nothing and do not retire the session");
        IdealPrep prep1(1, dealer);
        check(throws<std::invalid_argument>([&] {
                  const prep::BdozPair p = prep1.random_bdoz(3);
                  BdozBits a(p.own.bits, p.keys.keys, p.own.macs);
                  pi_dpf(ch, TEST_SID, 1, dealer.delta_share(1), a, FieldShare(), FieldShare());
              }) && ch.stats().msgs_sent == 0,
              "pi_dpf: a role that does not match the endpoint is rejected");
    }
}

// ------------------------------------------------------------ (c) leaky protocol on the real preprocessing
struct RealParty {
    std::vector<LeakyDpfOutput> outs;
    gf128 delta;
    PhaseStats st;
    prep::PrepStats prep_st;
    bool alive = false;
};

static void test_leaky_real_prep() {
    Section sec("leaky protocol, real preprocessing, n in {4, 8, 10, 12} in one session");
    const std::vector<unsigned> ns = {4, 8, 10, 12};
    std::vector<uint64_t> alpha, a0;
    std::vector<gf128> beta, b0;
    for (unsigned n : ns) {
        alpha.push_back(rand_below((uint64_t)1 << n));
        beta.push_back(rand_elem());
        a0.push_back(rand_bits(n));
        b0.push_back(rand_elem());
    }
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            RealParty p;
            RealPrep prep(role, ch, TEST_SID);
            Meter meter(ch);
            prep.setup();
            for (size_t k = 0; k < ns.size(); k++) {
                LeakyOptions opt;
                opt.key_id = k;
                p.outs.push_back(run_pi_dpf(ch, prep, TEST_SID, ns[k], role == 0 ? a0[k] : (alpha[k] ^ a0[k]),
                                            role == 0 ? b0[k] : (beta[k] ^ b0[k]), opt, &meter));
            }
            p.delta = prep.delta();
            p.st = meter.stats();
            p.prep_st = prep.stats();
            p.alive = !prep.aborted();
            return p;
        };
    };
    const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto f0 = party(0);
    auto f1 = party(1);
    auto res = run_two_party(f0, f1);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
    check(res.first.ok && res.second.ok, "real prep: the session completes (" + res.first.error + " / " + res.second.error + ")");
    if (!res.first.ok || !res.second.ok) return;
    g_honest_runs += (int)ns.size();
    const RealParty& p0 = res.first.value;
    const RealParty& p1 = res.second.value;
    const gf128 delta = p0.delta ^ p1.delta;
    check((delta.lo & 1) == 1, "real prep: lsb(Delta_0 + Delta_1) = 1");
    for (size_t k = 0; k < ns.size(); k++) {
        const OkFlags ok = leaky_ok(p0.outs[k], p1.outs[k], ns[k], alpha[k], beta[k], delta);
        check(ok.u && ok.v, "real prep, n = " + str(ns[k]) + ": outputs reconstruct to unit and unit * beta");
        check(ok.u_mac && ok.v_mac, "real prep, n = " + str(ns[k]) + ": MAC invariants of u and v");
        check_masked(p0.outs[k], p1.outs[k], "real prep, n = " + str(ns[k]));
        check(level0_ok(p0.outs[k].transcript.W, p0.outs[k].transcript.cws[0], p0.delta, p1.delta, (unsigned)(alpha[k] >> (ns[k] - 1)),
                        h0_domain(leaky_sub_sid(TEST_SID, k))),
              "real prep, n = " + str(ns[k]) + ": roots Delta_b + W and the H0 domain of key " + str(k));
    }
    uint64_t exp_cot = 0, exp_rounds = 2;
    uint64_t exp_phase[BP_COUNT] = {0}, exp_phase_rounds[BP_COUNT] = {0};
    for (unsigned n : ns) {
        exp_cot += prep::u_message_bytes(abit_demand(n)) + prep::COT_SEED_BYTES + prep::COT_CHECK_BYTES;
        exp_rounds += 3 + leaky_rounds(n);
        for (int p = BP_ZH_INPUT; p <= BP_ZH_CHECK; p++) {
            exp_phase[p] += leaky_bytes_per_party(n, p);
            exp_phase_rounds[p] += leaky_rounds(n, p);
        }
    }
    const RealParty* parties[2] = {&p0, &p1};
    const ChannelStats* chs[2] = {&res.first.stats, &res.second.stats};
    for (int role = 0; role < 2; role++) {
        const RealParty& p = *parties[role];
        bool ok = p.prep_st[prep::PH_BASEOT].bytes_sent == 12288 && p.prep_st[prep::PH_COTEXT].bytes_sent == exp_cot &&
                  p.prep_st[prep::PH_ABIT].bytes_sent == 0 && p.prep_st[prep::PH_BASEOT].rounds == 2 &&
                  p.prep_st[prep::PH_COTEXT].rounds == 3 * ns.size();
        uint64_t sum = p.prep_st.total().bytes_sent, sum_rounds = p.prep_st.total().rounds;
        for (int ph = 0; ph < BP_COUNT; ph++) {
            ok = ok && p.st[ph].bytes_sent == exp_phase[ph] && p.st[ph].rounds == exp_phase_rounds[ph];
            sum += p.st[ph].bytes_sent;
            sum_rounds += p.st[ph].rounds;
        }
        check(ok, "real prep: bytes and rounds per phase are exact (party " + str((uint64_t)role) + ")");
        check(sum == chs[role]->bytes_sent && sum_rounds == chs[role]->rounds, "real prep: the phases sum to the channel totals (party " + str((uint64_t)role) + ")");
    }
    check(res.first.stats.rounds == exp_rounds && res.second.stats.rounds == exp_rounds,
          "real prep: " + str(exp_rounds) + " rounds (2 base OT, 3 per extension call, n + 11 per key)");
    check(p0.alive && p1.alive, "real prep: the session is still alive");
    std::printf("  real preprocessing, keys n = 4, 8, 10, 12: %.2f s for both parties, %llu bytes\n", wall,
                (unsigned long long)(res.first.stats.bytes_sent + res.second.stats.bytes_sent));
}

// ------------------------------------------------------------ (d) selective failure
struct RealAttackParty {
    bool completed = false;
    bool aborted = false;       // ProtocolAbort
    std::string msg;
    LeakyDpfOutput out;
    gf128 delta;
    bool prep_retired = false;  // prep.aborted() and a zero key share after the abort
    bool refuses = false;       // a later request throws ProtocolAbort
};

static void test_selective_failure() {
    Section sec("selective failure of the leaky protocol");
    const unsigned n = 6;

    // Main experiment: 200 runs, error on the share of CW^(1), removed on the children of the
    // node the cheater guesses to be on the path, i.e. on a guess of the top bit of alpha.
    const int runs = 200;
    int passed = 0, consistent = 0, correct_when_passed = 0, by_cheater[2] = {0, 0}, batch_aborts = 0;
    for (int it = 0; it < runs; it++) {
        const uint64_t alpha = rand_below((uint64_t)1 << n);
        const gf128 beta = rand_elem();
        const int cheater = it & 1;
        const uint64_t guess = rand_bits(1);
        gf128 err = rand_elem();
        err.lo |= 1;
        const AttackResult a = attack_run(n, alpha, beta, cheater, LeakyCheat::cw(1, err, guess));
        const uint64_t top_bit = alpha >> (n - 1);
        passed += a.passed;
        by_cheater[cheater] += a.passed;
        consistent += (a.passed == (top_bit == guess));
        correct_when_passed += (a.passed && a.good);
        batch_aborts += (!a.passed && a.batch_check);
        check(a.passed == (top_bit == guess), "selective failure run " + str((uint64_t)it) + ": check passes iff msb(alpha) = guess");
        check(a.passed || a.batch_check, "selective failure run " + str((uint64_t)it) + ": the abort is the batch check");
    }
    check(consistent == runs, "selective failure: the predicate holds in all " + str((uint64_t)runs) + " runs");
    check(correct_when_passed == passed, "selective failure: whenever the check passes the outputs are correct and authenticated");
    check(batch_aborts == runs - passed, "selective failure: every abort is \"BatchCheck failed\" at the honest party");
    check(passed >= 70 && passed <= 130, "selective failure: about half of the runs pass (" + str((uint64_t)passed) + " of " + str((uint64_t)runs) + ")");
    std::printf("  selective failure (error on CW^(1), guess of the top index bit): %d of %d runs passed (%d with P_0 cheating, "
                "%d with P_1 cheating); pass iff msb(alpha) = guess in %d of %d runs\n",
                passed, runs, by_cheater[0], by_cheater[1], consistent, runs);

    // Deeper level: the predicate is "the first `level` bits of alpha equal the guess".
    {
        const unsigned level = 3;
        const int deep_runs = 64;
        int deep_pass = 0;
        for (int it = 0; it < deep_runs; it++) {
            const uint64_t alpha = rand_below((uint64_t)1 << n);
            const uint64_t prefix = alpha >> (n - level);
            const uint64_t guess = (it % 4 == 0) ? prefix : rand_below((uint64_t)1 << level);
            const AttackResult a = attack_run(n, alpha, rand_elem(), it & 1, LeakyCheat::cw(level, rand_elem(), guess));
            deep_pass += a.passed;
            check(a.passed == (guess == prefix) && (a.good || !a.passed),
                  "error on CW^(3), run " + str((uint64_t)it) + ": passes iff the 3-bit prefix is guessed");
        }
        std::printf("  error on CW^(3) with a prefix guess: %d of %d runs passed\n", deep_pass, deep_runs);
    }

    // Level 0: the root is on the path for certain; removing the error needs no guess, the
    // execution is then an honest one and nothing is learned.
    for (int it = 0; it < 10; it++) {
        const AttackResult a = attack_run(n, rand_below((uint64_t)1 << n), rand_elem(), it & 1, LeakyCheat::cw(0, rand_elem(), 0));
        check(a.passed && a.good, "error on CW^(0) removed at the root, run " + str((uint64_t)it) + ": passes, outputs correct");
    }

    // An error that is not removed makes the check fail whatever alpha is.
    for (int it = 0; it < 20; it++) {
        const unsigned level = (unsigned)rand_below(n);
        gf128 err = rand_elem();
        err.lo |= 1;
        const AttackResult a = attack_run(n, rand_below((uint64_t)1 << n), rand_elem(), it & 1, LeakyCheat::cw(level, err));
        check(!a.passed && a.batch_check, "error on CW^(" + str(level) + ") not removed, run " + str((uint64_t)it) + ": abort");
    }

    // Final correction word: the guess is the whole index.
    for (int it = 0; it < 12; it++) {
        const uint64_t alpha = rand_below((uint64_t)1 << n);
        const bool right = it % 3 == 0;
        const uint64_t guess = right ? alpha : (alpha + 1 + rand_below(((uint64_t)1 << n) - 1)) % ((uint64_t)1 << n);
        const AttackResult a = attack_run(n, alpha, rand_elem(), it & 1, LeakyCheat::final_cw(rand_elem(), rand_elem(), guess));
        check(a.passed == right && (a.good || !a.passed), "error on CW^(n), run " + str((uint64_t)it) + ": passes iff alpha itself is guessed");
    }
    for (int it = 0; it < 4; it++) {
        const AttackResult a = attack_run(n, rand_below((uint64_t)1 << n), rand_elem(), it & 1,
                                          LeakyCheat::final_cw(it < 2 ? rand_elem() : gf128(), rand_elem()));
        check(!a.passed, "error on CW^(n) not removed, run " + str((uint64_t)it) + ": abort");
    }

    // A wrong share of the opened value a.
    for (int it = 0; it < 4; it++) {
        const unsigned bit = (unsigned)rand_below(128);
        const gf128 err = bit < 64 ? gf128((uint64_t)1 << bit, 0) : gf128(0, (uint64_t)1 << (bit - 64));
        const AttackResult a = attack_run(n, rand_below((uint64_t)1 << n), rand_elem(), it & 1, LeakyCheat::a(err));
        check(!a.passed && a.batch_check, "wrong share of a, run " + str((uint64_t)it) + ": abort");
    }

    // The same attack end to end on the real preprocessing; an abort retires the session.
    const bool rights[4] = {true, false, false, true};
    for (int it = 0; it < 4; it++) {
        const unsigned n_real = 5;
        const bool right = rights[it];
        const uint64_t alpha = rand_below((uint64_t)1 << n_real);
        const gf128 beta = rand_elem();
        const uint64_t top = alpha >> (n_real - 1);
        const int cheater = it & 1;
        const uint64_t guess = right ? top : (top ^ 1);
        const uint64_t a0 = rand_bits(n_real);
        const gf128 b0 = rand_elem();
        const LeakyCheat hook = LeakyCheat::cw(1, gf128(0xABCDEF, 0), guess);
        bytes sid = TEST_SID;
        sid.push_back((uint8_t)it);
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                RealAttackParty p;
                RealPrep prep(role, ch, sid);
                prep.setup();
                LeakyOptions opt;
                opt.cheat = role == cheater ? &hook : nullptr;
                try {
                    p.out = run_pi_dpf(ch, prep, sid, n_real, role == 0 ? a0 : (alpha ^ a0), role == 0 ? b0 : (beta ^ b0), opt);
                    p.completed = true;
                    p.delta = prep.delta();
                } catch (const ProtocolAbort& e) {
                    p.aborted = true;
                    p.msg = e.what();
                    p.prep_retired = prep.aborted() && prep.delta().is_zero();
                    p.refuses = throws<ProtocolAbort>([&] { prep.random_bits(1); });
                }
                return p;
            };
        };
        auto f0 = party(0);
        auto f1 = party(1);
        auto res = run_two_party(f0, f1);
        const RealAttackParty& q0 = res.first.value;
        const RealAttackParty& q1 = res.second.value;
        const bool ok = res.first.ok && res.second.ok && q0.completed && q1.completed;
        check(res.first.ok && res.second.ok && ok == right, "real prep, attack run " + str((uint64_t)it) + ": passes iff the top bit is guessed");
        if (right) {
            check(ok && leaky_ok(q0.out, q1.out, n_real, alpha, beta, q0.delta ^ q1.delta).all(),
                  "real prep, attack run " + str((uint64_t)it) + ": outputs correct after a right guess");
        } else {
            const RealAttackParty& honest = cheater == 0 ? q1 : q0;
            check(honest.aborted && honest.msg == "BatchCheck failed" && honest.prep_retired,
                  "real prep, attack run " + str((uint64_t)it) + ": the honest party retired the session");
            check(honest.refuses, "real prep, attack run " + str((uint64_t)it) + ": the retired session refuses further requests");
        }
    }
}

// ------------------------------------------------------------ (e) honest executions never abort
static void test_honest_never_aborts() {
    Section sec("honest executions never abort");
    const int runs = 100;
    int aborts = 0;
    for (int it = 0; it < runs; it++) {
        const unsigned n = 1 + (unsigned)rand_below(10);
        const uint64_t alpha = rand_below((uint64_t)1 << n);
        const gf128 beta = rand_elem();
        const LeakyRun r = leaky_run(n, alpha, beta, nullptr, nullptr, (it & 1) != 0, (uint64_t)it);
        if (!r.passed) aborts++;
        else g_masks.push_back(opened_mask(r.out[0], r.out[1]));
        check(r.passed && leaky_ok(r.out[0], r.out[1], n, alpha, beta, r.delta).all(),
              "honest run " + str((uint64_t)it) + " (n = " + str(n) + "): no abort, outputs correct and authenticated");
    }
    check(aborts == 0, str((uint64_t)runs) + " random honest executions: no abort");
    check(g_honest_aborts == 0 && g_honest_runs >= runs, "no honest execution of this program aborted (" + str((uint64_t)g_honest_runs) + " so far)");
    // the mask r of the opened value: non-zero, fresh in every execution, bits balanced
    // (the mean weight of a uniform 128-bit string is 64 with standard deviation 5.66 / sqrt(number of masks))
    bool all_masked = g_masks.size() >= (size_t)runs, all_fresh = true;
    double mean_weight = 0.0;
    for (size_t i = 0; i < g_masks.size(); i++) {
        all_masked = all_masked && !g_masks[i].is_zero();
        mean_weight += weight(g_masks[i]);
        for (size_t j = 0; j < i; j++) all_fresh = all_fresh && g_masks[i] != g_masks[j];
    }
    mean_weight /= g_masks.empty() ? 1.0 : (double)g_masks.size();
    check(all_masked, "the opened value was masked in all " + str(g_masks.size()) + " honest executions inspected");
    check(all_fresh, "every execution uses a fresh mask r");
    check(mean_weight > 60.0 && mean_weight < 68.0, "the masks look uniform (mean weight " + std::to_string(mean_weight) + " of 128 bits)");
    std::printf("  honest executions of the leaky protocol so far: %d, aborts: %d\n", g_honest_runs, g_honest_aborts);
}

// ------------------------------------------------------------ (f) semi-honest baseline
struct ShParty {
    std::vector<SemiHonestDpfOutput> outs;
    gf128 delta;
    PhaseStats st;
    uint64_t keys_generated = 0;
};
struct ShCase {
    unsigned n;
    uint64_t alpha;
    gf128 beta;
};
struct ShRun {
    bool ok = false;
    std::string msg;
    ShParty p[2];
    ChannelStats ch[2];
};

// One session with the keys `cases`; `tamper` (if >= 0) is the index of the message of P_1 whose first bit is flipped.
static ShRun semihonest_session(const std::vector<ShCase>& cases, const bytes& sid, int tamper = -1, unsigned threads0 = 1,
                                unsigned threads1 = 1) {
    std::vector<uint64_t> a0;
    std::vector<gf128> b0;
    for (const ShCase& c : cases) {
        a0.push_back(rand_bits(c.n));
        b0.push_back(rand_elem());
    }
    const unsigned threads[2] = {threads0, threads1};
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            std::map<size_t, TamperChannel::Rule> rules;
            if (tamper >= 0 && role == 1) rules[(size_t)tamper] = flip_bit_rule(0);
            TamperChannel proxy(ch, rules);
            Channel& use = rules.empty() ? ch : static_cast<Channel&>(proxy);
            ShParty p;
            SemiHonestSession session(role, use, sid);
            session.set_threads(threads[role]);
            session.setup();
            for (size_t k = 0; k < cases.size(); k++)
                p.outs.push_back(session.dpf_gen(cases[k].n, role == 0 ? a0[k] : (cases[k].alpha ^ a0[k]),
                                                 role == 0 ? b0[k] : (cases[k].beta ^ b0[k])));
            p.delta = session.delta();
            p.keys_generated = session.keys_generated();
            p.st = session.stats();
            return p;
        };
    };
    auto f0 = party(0);
    auto f1 = party(1);
    auto res = run_two_party(f0, f1);
    ShRun r;
    r.ok = res.first.ok && res.second.ok;
    r.msg = res.first.error + " / " + res.second.error;
    r.ch[0] = res.first.stats;
    r.ch[1] = res.second.stats;
    if (r.ok) {
        r.p[0] = std::move(res.first.value);
        r.p[1] = std::move(res.second.value);
    }
    return r;
}

static bool semihonest_ok(const SemiHonestDpfOutput& o0, const SemiHonestDpfOutput& o1, unsigned n, uint64_t alpha, const gf128& beta,
                          bool* t_ok = nullptr, bool* v_ok = nullptr) {
    const size_t big_n = (size_t)1 << n;
    bool t = o0.t.size() == big_n && o1.t.size() == big_n, v = o0.v.size() == big_n && o1.v.size() == big_n;
    for (size_t j = 0; t && v && j < big_n; j++) {
        const unsigned unit = (j == alpha) ? 1u : 0u;
        t = (unsigned)(o0.t[j] ^ o1.t[j]) == unit;
        v = (o0.v[j] ^ o1.v[j]) == prep::gf_and_bit(beta, unit);
        if (!t || !v) break;
    }
    if (t_ok) *t_ok = t;
    if (v_ok) *v_ok = v;
    return t && v;
}

static void test_semihonest() {
    Section sec("semi-honest baseline, n in {1, 4, 8, 10, 12}");
    const std::vector<unsigned> ns = {1, 4, 8, 10, 10, 12};
    std::vector<ShCase> cases;
    for (unsigned n : ns) cases.push_back(ShCase{n, rand_below((uint64_t)1 << n), rand_elem()});
    cases[3] = ShCase{10, 0, gf128()};                  // zero payload, first leaf
    cases[4] = ShCase{10, (1u << 10) - 1, gf128::one()};  // last leaf
    const bytes sid = {'t', 'e', 's', 't', '-', 's', 'e', 'm', 'i', 'h', 'o', 'n', 'e', 's', 't'};
    const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const ShRun r = semihonest_session(cases, sid);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - t0;
    check(r.ok, "semi-honest: the session completes (" + r.msg + ")");
    if (!r.ok) return;
    check(((r.p[0].delta ^ r.p[1].delta).lo & 1) == 1 && (r.p[0].delta.lo & 1) == 1 && (r.p[1].delta.lo & 1) == 0,
          "semi-honest key shares: lsb 1 for P_0, 0 for P_1");
    for (size_t k = 0; k < cases.size(); k++) {
        const unsigned n = cases[k].n;
        const SemiHonestDpfOutput& o0 = r.p[0].outs[k];
        const SemiHonestDpfOutput& o1 = r.p[1].outs[k];
        bool t_ok = false, v_ok = false;
        semihonest_ok(o0, o1, n, cases[k].alpha, cases[k].beta, &t_ok, &v_ok);
        check(t_ok, "semi-honest n = " + str(n) + ": t_0 + t_1 = unit(N, alpha)");
        check(v_ok, "semi-honest n = " + str(n) + ": v_0 + v_1 = unit(N, alpha) * beta");
        check(o0.transcript == o1.transcript && o0.transcript.cws.size() == n && o0.n == n,
              "semi-honest n = " + str(n) + ": one public transcript with n correction words");
        check(level0_ok(o0.transcript.W, o0.transcript.cws[0], r.p[0].delta, r.p[1].delta, (unsigned)(cases[k].alpha >> (n - 1)),
                        h0_domain(semihonest_sub_sid(sid, k), label_h0_semihonest())),
              "semi-honest n = " + str(n) + ": roots Delta_b + W and the H0 domain of key " + str(k));
    }
    check(r.p[0].outs[3].transcript.W != r.p[0].outs[4].transcript.W, "semi-honest: fresh W per key");
    uint64_t exp_bytes[BP_COUNT] = {0}, exp_rounds[BP_COUNT] = {0};
    exp_bytes[BP_SH_BASEOT] = 12288;
    exp_rounds[BP_SH_BASEOT] = 2;
    for (unsigned n : ns) {
        for (int p = BP_SH_COT; p <= BP_SH_LEAF; p++) {
            exp_bytes[p] += semihonest_bytes_per_party(n, p);
            exp_rounds[p] += semihonest_rounds(n, p);
        }
    }
    for (int role = 0; role < 2; role++) {
        bool ok = true;
        uint64_t sum = 0, sum_rounds = 0;
        for (int p = 0; p < BP_COUNT; p++) {
            ok = ok && r.p[role].st[p].bytes_sent == exp_bytes[p] && r.p[role].st[p].rounds == exp_rounds[p];
            sum += r.p[role].st[p].bytes_sent;
            sum_rounds += r.p[role].st[p].rounds;
        }
        check(ok, "semi-honest: bytes and rounds per phase are exact (party " + str((uint64_t)role) + ")");
        check(sum == r.ch[role].bytes_sent && sum_rounds == r.ch[role].rounds, "semi-honest: the phases sum to the channel totals (party " + str((uint64_t)role) + ")");
    }
    std::printf("  semi-honest session, keys n = 1, 4, 8, 10, 10, 12: %.2f s for both parties (base OT included), %llu bytes\n", wall,
                (unsigned long long)(r.ch[0].bytes_sent + r.ch[1].bytes_sent));

    // No check: a party that sends a wrong correction word is not detected; the output is wrong.
    // Messages of P_1: 2 (base OT), 1 (COT flight), then one per level, then the leaf word.  P_0
    // uses the word it received only at its nodes with control bit 1, so a tampered word of level i
    // changes nothing with probability 2^-(2^i) (no such node; 1/16 for level 2).  The last level
    // (32 nodes) and the leaf word (64 leaves) are tampered here: 2^-32 and 2^-64.
    {
        const unsigned n = 6;
        const ShCase c{n, rand_below(64), rand_elem()};
        const bytes sid_level = {'t', 'a', 'm', 'p', 'e', 'r', '-', 'l', 'e', 'v', 'e', 'l'};
        const ShRun t1 = semihonest_session({c}, sid_level, 2 + 1 + (int)(n - 1));
        check(t1.ok && !semihonest_ok(t1.p[0].outs[0], t1.p[1].outs[0], n, c.alpha, c.beta),
              "semi-honest: a tampered last-level correction word yields wrong shares and no abort");
        const bytes sid_final = {'t', 'a', 'm', 'p', 'e', 'r', '-', 'f', 'i', 'n', 'a', 'l'};
        const ShRun t2 = semihonest_session({c}, sid_final, 2 + 1 + (int)n);
        check(t2.ok && !semihonest_ok(t2.p[0].outs[0], t2.p[1].outs[0], n, c.alpha, c.beta),
              "semi-honest: a tampered final correction word yields wrong shares and no abort");
    }

    // argument validation
    {
        auto pair = LoopbackChannel::make_pair();
        pair.second.reset();  // no peer: a call that got past the validation fails instead of blocking
        Channel& ch = *pair.first;
        PlainCot cot(0, gf128::one(), rand_vec(KAPPA), rand_vec(KAPPA), rand_vec(KAPPA));
        SystemRng sys;
        check(throws<std::invalid_argument>([&] { semihonest_dpf(ch, TEST_SID, 0, 0, cot, 0, 0, gf128(), sys); }), "n = 0 rejected");
        check(throws<std::invalid_argument>([&] { semihonest_dpf(ch, TEST_SID, 0, 0, cot, 3, 9, gf128(), sys); }), "index share out of range rejected");
        check(throws<std::invalid_argument>([&] { semihonest_dpf(ch, TEST_SID, 0, 1, cot, 3, 1, gf128(), sys); }), "role mismatch rejected");
        SemiHonestSession session(0, ch, TEST_SID);
        check(throws<std::invalid_argument>([&] { session.dpf_gen(3, 1, gf128::one()); }), "dpf_gen before setup is refused");
        check(ch.stats().msgs_sent == 0, "invalid arguments send nothing");
        check(throws<std::invalid_argument>([&] { SemiHonestSession bad(1, ch, TEST_SID); }), "a session on the wrong endpoint is rejected");
    }
}

// ------------------------------------------------------------ (h) separation of the keys of a session
// Each key generation hashes under the domain of ITS (sid, key number).  The
// tree of each party is a function of its key share, the public transcript (W
// and the correction words) and the hash domain.  Both trees are recomputed
// here in the plain form with the domain the specification prescribes and
// compared with the protocol outputs; the domain of the next key number is the
// negative control.  A protocol that ran two keys of a session under one
// domain fails the comparison.
static void test_key_separation() {
    Section sec("separation of the keys of a session");
    const unsigned n = 5;
    const size_t big_n = (size_t)1 << n;
    const bytes sid = {'t', 'e', 's', 't', '-', 'b', 'a', 's', 'e', 'l', 'i', 'n', 'e', 's', '-', 's', 'e', 'p'};

    // leaky protocol: M_b[u^(j)] is the leaf string itself
    for (uint64_t key_id : {(uint64_t)0, (uint64_t)1, (uint64_t)5}) {
        const uint64_t alpha = rand_below(big_n);
        const gf128 beta = rand_elem();
        const LeakyRun r = leaky_run(n, alpha, beta, nullptr, nullptr, false, key_id, 1, 1, sid);
        check(r.passed && leaky_ok(r.out[0], r.out[1], n, alpha, beta, r.delta).all(),
              "leaky, key number " + str(key_id) + ": outputs correct and authenticated");
        if (!r.passed) continue;
        const uint64_t dom = h0_domain(leaky_sub_sid(sid, key_id)), other = h0_domain(leaky_sub_sid(sid, key_id + 1));
        for (int b = 0; b < 2; b++) {
            const LeakyDpfOutput& out = r.out[b];
            const std::string tag = "leaky, key number " + str(key_id) + ", P_" + str((uint64_t)b);
            const GfVec leaves = recompute_leaves(r.delta_share[b], out.transcript.W, out.transcript.cws, dom);
            bool tree = out.u.macs == leaves, shares = true;
            GfVec hv, hm;
            hash1(leaves, dom, hv, hm);
            for (size_t j = 0; j < big_n; j++) {
                const unsigned t = (unsigned)(leaves[j].lo & 1);
                tree = tree && out.u.bits[j] == t;
                shares = shares && out.v.values[j] == (hv[j] ^ prep::gf_and_bit(out.transcript.cw_final_val, t)) &&
                         out.v.macs[j] == (hm[j] ^ prep::gf_and_bit(out.transcript.cw_final_mac, t));
            }
            check(tree, tag + ": the tree is the one of H0 under (sid, " + str(key_id) + ")");
            check(shares, tag + ": v and its MAC shares are H1 of the leaves and the final word");
            check(out.u.macs != recompute_leaves(r.delta_share[b], out.transcript.W, out.transcript.cws, other),
                  tag + ": another key number gives another tree (control)");
        }
    }

    // semi-honest baseline: three keys in one session, key numbers 0, 1, 2
    std::vector<ShCase> cases;
    for (int k = 0; k < 3; k++) cases.push_back(ShCase{n, rand_below(big_n), rand_elem()});
    const ShRun r = semihonest_session(cases, sid);
    check(r.ok, "semi-honest: the session completes (" + r.msg + ")");
    if (r.ok) {
        check(r.p[0].keys_generated == 3 && r.p[1].keys_generated == 3, "semi-honest: the session counted its three keys");
        for (uint64_t key_id = 0; key_id < 3; key_id++) {
            const uint64_t dom = h0_domain(semihonest_sub_sid(sid, key_id), label_h0_semihonest());
            const uint64_t other = h0_domain(semihonest_sub_sid(sid, key_id + 1), label_h0_semihonest());
            for (int b = 0; b < 2; b++) {
                const SemiHonestDpfOutput& out = r.p[b].outs[(size_t)key_id];
                const std::string tag = "semi-honest, key number " + str(key_id) + ", P_" + str((uint64_t)b);
                const GfVec leaves = recompute_leaves(r.p[b].delta, out.transcript.W, out.transcript.cws, dom);
                const GfVec hv = hash0(leaves, dom);
                const GfVec wrong = recompute_leaves(r.p[b].delta, out.transcript.W, out.transcript.cws, other);
                const GfVec hw = hash0(wrong, other);
                bool tree = true, shares = true, same_as_other = true;
                for (size_t j = 0; j < big_n; j++) {
                    const unsigned t = (unsigned)(leaves[j].lo & 1);
                    tree = tree && out.t[j] == t;
                    shares = shares && out.v[j] == (hv[j] ^ prep::gf_and_bit(out.transcript.cw_final, t));
                    same_as_other = same_as_other && out.v[j] == (hw[j] ^ prep::gf_and_bit(out.transcript.cw_final, (unsigned)(wrong[j].lo & 1)));
                }
                check(tree, tag + ": the tree is the one of H0 under (sid, " + str(key_id) + ")");
                check(shares, tag + ": v is H0 of the leaves and the final word");
                check(!same_as_other, tag + ": another key number gives other shares (control)");
            }
            check(semihonest_ok(r.p[0].outs[(size_t)key_id], r.p[1].outs[(size_t)key_id], n, cases[(size_t)key_id].alpha, cases[(size_t)key_id].beta),
                  "semi-honest, key number " + str(key_id) + ": outputs reconstruct");
        }
    }
    check(leaky_sub_sid(sid, 2) != semihonest_sub_sid(sid, 2) && leaky_sub_sid(sid, 2) != leaky_sub_sid(sid, 3) &&
              h0_domain(leaky_sub_sid(sid, 2)) != h0_domain(leaky_sub_sid(sid, 2), label_h0_semihonest()),
          "the two baselines derive their per-key identifiers and domains under different tags and labels");
}

// ------------------------------------------------------------ worker threads
static void test_threads() {
    Section sec("worker threads");
    for (unsigned n : {13u, 14u}) {
        const uint64_t alpha = rand_below((uint64_t)1 << n);
        const gf128 beta = rand_elem();
        const LeakyRun r = leaky_run(n, alpha, beta, nullptr, nullptr, false, 0, 3, 1);
        check(r.passed && leaky_ok(r.out[0], r.out[1], n, alpha, beta, r.delta).all() && leaky_phase_counts_match(r.st[0], n, false) &&
                  leaky_phase_counts_match(r.st[1], n, false),
              "leaky n = " + str(n) + ": three workers at P_0, one at P_1: outputs and counts unchanged");
        const LeakyRun r2 = leaky_run(n, alpha, beta, nullptr, nullptr, false, 0, 2, 4);
        check(r2.passed && leaky_ok(r2.out[0], r2.out[1], n, alpha, beta, r2.delta).all(), "leaky n = " + str(n) + ": two and four workers");
    }
    // an attack is still detected (and a right guess still passes) with workers
    {
        const unsigned n = 13;
        for (int it = 0; it < 4; it++) {
            const uint64_t alpha = rand_below((uint64_t)1 << n);
            const uint64_t guess = (it & 1) ? (alpha >> (n - 1)) : ((alpha >> (n - 1)) ^ 1);
            const LeakyCheat hook = LeakyCheat::cw(1, rand_elem(), guess);
            const LeakyRun r = leaky_run(n, alpha, rand_elem(), &hook, nullptr, false, 0, 4, 3);
            check(r.passed == ((it & 1) != 0), "leaky n = 13 with workers, attack run " + str((uint64_t)it) + ": passes iff the top bit is guessed");
        }
    }
    const bytes sid = {'t', 'h', 'r', 'e', 'a', 'd', 's'};
    const ShCase c{14, rand_below((uint64_t)1 << 14), rand_elem()};
    const ShRun r = semihonest_session({c}, sid, -1, 4, 1);
    check(r.ok && semihonest_ok(r.p[0].outs[0], r.p[1].outs[0], 14, c.alpha, c.beta), "semi-honest n = 14: four workers at P_0, one at P_1");
}

// ------------------------------------------------------------ TCP and the TEST-ONLY verification step
struct TcpParty {
    bool verified = false;
    ChannelStats before_verify;
    PhaseStats st;
    prep::PrepStats prep_st;
};

template <class F>
static bool run_tcp_pair(F body, TcpParty out[2], std::string& error) {
    TcpListener listener("127.0.0.1", 0);
    const uint16_t port = listener.port();
    std::string err1;
    std::thread t1([&] {
        try {
            std::unique_ptr<TcpChannel> ch = TcpChannel::connect(1, "127.0.0.1", port, 10000);
            out[1] = body(1, *ch);
        } catch (const std::exception& e) {
            err1 = e.what();
        }
    });
    try {
        std::unique_ptr<TcpChannel> ch = listener.accept(0);
        out[0] = body(0, *ch);
    } catch (const std::exception& e) {
        error = e.what();
    }
    t1.join();
    if (!err1.empty()) error += " / " + err1;
    return error.empty();
}

static void test_tcp_and_verification() {
    Section sec("TCP on 127.0.0.1 and the TEST-ONLY verification step");
    const bytes sid = {'t', 'c', 'p'};

    // the digests of the verification step, in process
    {
        const unsigned n = 7;
        const uint64_t alpha = rand_below((uint64_t)1 << n);
        const gf128 beta = rand_elem();
        LeakyRun r = leaky_run(n, alpha, beta);
        const testonly::Reveal rv{r.delta, alpha, beta};
        check(r.passed && testonly::leaky_digest(0, r.out[0], rv) == testonly::leaky_digest(1, r.out[1], rv),
              "verification digests agree on correct outputs");
        bool all_detected = r.passed;
        for (int which = 0; r.passed && which < 4; which++) {
            LeakyDpfOutput bad = r.out[1];
            const size_t j = (size_t)rand_below((uint64_t)1 << n);
            if (which == 0) bad.u.bits[j] ^= 1;
            else if (which == 1) bad.u.macs[j].hi ^= 4;
            else if (which == 2) bad.v.values[j].lo ^= 1;
            else bad.v.macs[j].lo ^= (uint64_t)1 << 63;
            all_detected = all_detected && testonly::leaky_digest(0, r.out[0], rv) != testonly::leaky_digest(1, bad, rv);
        }
        check(all_detected, "verification digests differ when one output entry is wrong (bits, MACs, values, value MACs)");
        testonly::Reveal wrong = rv;
        wrong.alpha ^= 1;
        check(r.passed && testonly::leaky_digest(0, r.out[0], wrong) != testonly::leaky_digest(1, r.out[1], wrong),
              "verification digests differ for a wrong index");
    }

    // leaky protocol with the real preprocessing over TCP
    {
        const unsigned n = 9;
        const uint64_t a_share[2] = {rand_bits(n), rand_bits(n)};
        const gf128 b_share[2] = {rand_elem(), rand_elem()};
        TcpParty out[2];
        std::string error;
        const bool ok = run_tcp_pair(
            [&](int role, Channel& ch) {
                TcpParty p;
                RealPrep prep(role, ch, sid);
                Meter meter(ch);
                prep.setup();
                const LeakyDpfOutput o = run_pi_dpf(ch, prep, sid, n, a_share[role], b_share[role], LeakyOptions(), &meter);
                p.st = meter.stats();
                p.prep_st = prep.stats();
                p.before_verify = ch.stats();
                p.verified = testonly::verify_leaky(ch, role, o, prep.delta(), a_share[role], b_share[role]);
                return p;
            },
            out, error);
        check(ok && out[0].verified && out[1].verified, "leaky n = 9 over TCP: completes and both parties verify the outputs (" + error + ")");
        g_honest_runs++;
        if (!ok) g_honest_aborts++;
        bool counts = ok;
        for (int role = 0; ok && role < 2; role++) {
            counts = counts && leaky_phase_counts_match(out[role].st, n, false) &&
                     out[role].before_verify.bytes_sent == 12288 + prep::u_message_bytes(abit_demand(n)) + 16 + prep::COT_CHECK_BYTES + leaky_bytes_per_party(n) &&
                     out[role].before_verify.rounds == 2 + 3 + leaky_rounds(n);
        }
        check(counts, "leaky n = 9 over TCP: bytes and rounds as over the in-process channel");
    }

    // semi-honest baseline over TCP
    {
        const unsigned n = 9;
        const uint64_t a_share[2] = {rand_bits(n), rand_bits(n)};
        const gf128 b_share[2] = {rand_elem(), rand_elem()};
        TcpParty out[2];
        std::string error;
        const bool ok = run_tcp_pair(
            [&](int role, Channel& ch) {
                TcpParty p;
                SemiHonestSession session(role, ch, sid);
                session.setup();
                const SemiHonestDpfOutput o = session.dpf_gen(n, a_share[role], b_share[role]);
                p.st = session.stats();
                p.before_verify = ch.stats();
                p.verified = testonly::verify_semihonest(ch, role, o, session.delta(), a_share[role], b_share[role]);
                return p;
            },
            out, error);
        check(ok && out[0].verified && out[1].verified, "semi-honest n = 9 over TCP: completes and both parties verify the outputs (" + error + ")");
        check(ok && out[0].before_verify.bytes_sent == 12288 + semihonest_bytes_per_party(n) && out[1].before_verify.bytes_sent == out[0].before_verify.bytes_sent &&
                  out[0].before_verify.rounds == 2 + semihonest_rounds(n),
              "semi-honest n = 9 over TCP: bytes and rounds as over the in-process channel");
    }
}

// ------------------------------------------------------------ --interop: one party over TCP
static std::string phase_json(const char* name, const PhaseStat& s) {
    return std::string("\"") + name + "\": {\"bytes_sent\": " + str(s.bytes_sent) + ", \"bytes_recv\": " + str(s.bytes_recv) +
           ", \"rounds\": " + str(s.rounds) + "}";
}

static int run_interop(const std::string& proto, int role, int port, const std::vector<unsigned>& ns) {
    const bytes sid = {'i', 'n', 't', 'e', 'r', 'o', 'p', '-', 'b', 'a', 's', 'e', 'l', 'i', 'n', 'e', 's'};
    try {
        std::unique_ptr<TcpChannel> chp = TcpChannel::create(role, "127.0.0.1", (uint16_t)port, 120000);
        Channel& ch = *chp;
        SystemRng sys;
        std::vector<uint64_t> a_share;
        std::vector<gf128> b_share;
        for (unsigned n : ns) {
            a_share.push_back(sys.u64() >> (64 - n));
            b_share.push_back(sys.gf());
        }
        bool verified = true;
        std::string phases;
        ChannelStats protocol_stats;
        if (proto == "zhang") {
            RealPrep prep(role, ch, sid);
            Meter meter(ch);
            prep.setup();
            std::vector<LeakyDpfOutput> outs;
            for (size_t k = 0; k < ns.size(); k++) {
                LeakyOptions opt;
                opt.key_id = k;
                outs.push_back(run_pi_dpf(ch, prep, sid, ns[k], a_share[k], b_share[k], opt, &meter));
            }
            protocol_stats = ch.stats();
            const prep::PrepStats ps = prep.stats();
            const PhaseStats st = meter.stats();
            phases = phase_json("baseot", ps[prep::PH_BASEOT]) + ", " + phase_json("cotext", ps[prep::PH_COTEXT]) + ", " +
                     phase_json("abit", ps[prep::PH_ABIT]);
            for (int p = BP_ZH_INPUT; p <= BP_ZH_CHECK; p++) phases += ", " + phase_json(phase_name(p), st[p]);
            for (size_t k = 0; k < ns.size(); k++)
                verified = testonly::verify_leaky(ch, role, outs[k], prep.delta(), a_share[k], b_share[k]) && verified;
        } else {
            SemiHonestSession session(role, ch, sid);
            session.setup();
            std::vector<SemiHonestDpfOutput> outs;
            for (size_t k = 0; k < ns.size(); k++) outs.push_back(session.dpf_gen(ns[k], a_share[k], b_share[k]));
            protocol_stats = ch.stats();
            const PhaseStats st = session.stats();
            for (int p = BP_SH_BASEOT; p <= BP_SH_LEAF; p++) phases += (p == BP_SH_BASEOT ? "" : ", ") + phase_json(phase_name(p), st[p]);
            for (size_t k = 0; k < ns.size(); k++)
                verified = testonly::verify_semihonest(ch, role, outs[k], session.delta(), a_share[k], b_share[k]) && verified;
        }
        std::string n_list;
        for (size_t k = 0; k < ns.size(); k++) n_list += (k ? ", " : "") + str(ns[k]);
        std::printf("{\"peer\": \"c++\", \"proto\": \"%s\", \"role\": %d, \"n\": [%s], \"bytes_sent\": %llu, \"bytes_recv\": %llu, "
                    "\"rounds\": %llu, \"phases\": {%s}, \"outputs_correct\": %s}\n",
                    proto.c_str(), role, n_list.c_str(), (unsigned long long)protocol_stats.bytes_sent,
                    (unsigned long long)protocol_stats.bytes_recv, (unsigned long long)protocol_stats.rounds, phases.c_str(),
                    verified ? "true" : "false");
        std::fflush(stdout);
        return verified ? 0 : 4;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_baselines --interop: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--interop") == 0) {
        if (argc != 6) {
            std::fprintf(stderr, "usage: test_baselines --interop zhang|sh ROLE PORT N1[,N2,...]\n");
            return 2;
        }
        const std::string proto = argv[2];
        const int role = std::atoi(argv[3]), port = std::atoi(argv[4]);
        std::vector<unsigned> ns;
        for (const char* p = argv[5]; *p;) {
            char* end = nullptr;
            const unsigned long v = std::strtoul(p, &end, 10);
            if (end == p) break;
            ns.push_back((unsigned)v);
            p = (*end == ',') ? end + 1 : end;
        }
        bool ok = (proto == "zhang" || proto == "sh") && (role == 0 || role == 1) && port > 0 && port < 65536 && !ns.empty();
        for (unsigned n : ns) ok = ok && n >= 1 && n <= MAX_N;
        if (!ok) {
            std::fprintf(stderr, "test_baselines --interop: bad arguments\n");
            return 2;
        }
        return run_interop(proto, role, port, ns);
    }
    uint64_t seed = 20261010;
    if (argc == 3 && std::strcmp(argv[1], "--seed") == 0) {
        seed = std::strtoull(argv[2], nullptr, 10);
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: test_baselines [--seed INTEGER] | test_baselines --interop zhang|sh ROLE PORT N1[,N2,...]\n");
        return 2;
    }
    g_rng.reset(new CtrPrgRng(seed, hashes::dom::PRG_TEST));
    std::printf("test_baselines: test seed %llu, AVX2 %s, PCLMUL %s\n", (unsigned long long)seed, COUPLET_HAVE_AVX2 ? "yes" : "no",
                COUPLET_HAVE_PCLMUL ? "yes" : "no");
    const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    try {
        test_building_blocks();
        test_python_vectors();
        test_leaky_correctness();
        test_leaky_real_prep();
        test_selective_failure();
        test_honest_never_aborts();
        test_semihonest();
        test_key_separation();
        test_threads();
        test_tcp_and_verification();
    } catch (const std::exception& e) {
        g_fail++;
        std::printf("  FAIL: unexpected exception: %s\n", e.what());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("test_baselines: %d checks, %d failures, %.1f s (honest executions of the leaky protocol: %d, aborts: %d)\n",
                g_pass + g_fail, g_fail, s, g_honest_runs, g_honest_aborts);
    return g_fail ? 1 : 0;
}
