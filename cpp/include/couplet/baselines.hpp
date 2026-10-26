// couplet/baselines.hpp -- the two comparison protocols of the evaluation
// (DESIGN.md Section 9, Q1), ported from the Python reference modules
// couplet/baselines/leaky_zhang.py and couplet/baselines/semihonest_ds.py.
// Same messages, same bytes and same rounds per phase as those modules: a
// party of this header and a party of the Python package complete either
// protocol together (tests/interop_baselines.py).
//
// The baselines are comparison points.  Nothing else in the code base depends
// on them, and the leaky protocol must never be used to generate keys for
// Couplet: it leaks a predicate of the index by design.
//
// ---------------------------------------------------------------------------
// (1) Leaky baseline: OUR RE-IMPLEMENTATION of protocol Pi_DPF of
//     W. Zhang, X. Guo, K. Yang, R. Zhu, Y. Yu, X. Wang, "Efficient Actively
//     Secure DPF and RAM-based 2PC with One-Bit Leakage", IEEE Symposium on Security and Privacy 2024
//     (Figure 5 Pi_DPF, Figure 2 BatchCheck, Section 3.3 authenticated
//     sharings / B2F / Convert, Figure 4 Input of Pi_2PC).
//
// The parties hold BDOZ-authenticated bits of the index alpha and a
// SPDZ-style authenticated payload beta in GF(2^128).  Both expand the WHOLE
// tree locally (N = 2^n leaves), exchange one share of a correction word per
// level and one final correction word, and obtain SPDZ-style authenticated
// sharings of  u = unit(N, alpha)  and  v = unit(N, alpha) * beta, followed
// by one batched MAC check of a random linear combination of all 2N outputs.
// A party that adds an error to its share of a correction word and removes
// it on a guessed path passes the check exactly when the guess is right: the
// abort leaks one predicate of alpha.
//
// What is the same as in the paper: every message, every local computation
// and the order of the steps of Figure 5 and Figure 2 (as reconstructed by the
// Python reference; see design/CHANGELOG.md, PY-F, for the steps that could
// only be confirmed by tests).  What DIFFERS from the original implementation
// (to be stated next to every number obtained with this code):
//
//   * Hash function.  The original uses a fixed-key AES-based circular
//     correlation-robust hash.  Here H0(x) = hashes::H_ccr(x, 0, dom): the
//     first 128 bits of Ascon-p12 applied to (x, tweak 0, dom, 0), a truncated
//     permutation without feed-forward and without AES-NI, with the domain
//     constant  dom = session_dom(0x48, "baseline-zhang/H0", ssid),
//     ssid = SHA-256(frame("Couplet-baseline-zhang", sid, le64(key number))).
//     H1(x) = H0(x) || H0(x XOR 1) as in the paper.  The whole tree is hashed
//     four nodes at a time with the interleaved (AVX2) permutation.
//   * COT source.  The original obtains F_COT / F_aBit from Ferret (silent
//     correlated OT of the EMP toolkit: LPN-based, communication sublinear in
//     the number of correlations).  Here the authenticated bits come from OUR
//     preprocessing (prep.hpp): 128 Masny-Rindal base OTs per direction over
//     Edwards25519 and IKNP-style correlated OT extension with the per-column
//     consistency check of DESIGN.md 7.2: per call and party 16 bytes per row
//     (rows rounded up to a multiple of 64) plus 2,048 bytes of padding rows,
//     a 16-byte challenge seed and a 2,064-byte check message.  One call of
//     n + 256 rows serves one key (9,248 bytes per party for n <= 64).  The
//     cost of the authenticated bits and all absolute running times therefore
//     differ from the paper; the messages of Pi_DPF itself are determined by
//     the protocol.
//   * Security parameters.  kappa = 128 everywhere: MAC keys, tree nodes and
//     the payload live in GF(2^128) (x^128 + x^7 + x^2 + x + 1), COT rows have
//     128 bits, there are 128 base OTs per direction.  No statistical
//     parameter enters: the baseline consumes authenticated bits only (no
//     bucketed triples), and its two checks (the extension check of prep.hpp
//     and BatchCheck) are over GF(2^128).  The LPN parameters of Ferret have
//     no counterpart here.  F_coin and F_com are hashes::coin_toss and
//     hashes::committed_exchange (SHA-256, 32-byte openings and coin shares:
//     96 bytes per party and coin, 64 + |m| bytes per party and committed
//     message, two rounds each).
//   * lsb convention.  The paper asks for lsb(Delta_b) = b; our layer has
//     lsb(Delta_0) = 1 and lsb(Delta_1) = 0.  Only lsb(Delta_0 XOR Delta_1) = 1
//     is used (the control bit is the lsb of a node).
//   * Bit order.  Level i consumes alpha^(i), so alpha^(0) is the MOST
//     significant bit of the leaf index (the convention of DESIGN.md), whereas
//     Figure 6 of the paper writes alpha = sum_i alpha^(i) 2^i.
//   * Input (Figure 4).  A chosen input bit is authenticated by derandomising
//     a random BDOZ bit (one bit per input bit).  The re-randomisation of the
//     keys by a public string and the trivial sharing of zero for the
//     non-owner, which exist for the garbled circuits of that paper, add only
//     public offsets that cancel in every value of Figure 5 and are not sent
//     (this favours the baseline by 2 kappa bits per input bit).
//   * The F_aBit call of step 4 (the bits of the mask r) is issued together
//     with the input authentication (one extension call per key); r is
//     independent of all inputs and messages.
//   * BatchCheck is Figure 2 literally, including the coin of its step 1,
//     which is unused when the figure is called with one value;
//     skip_unused_coin omits it (two rounds and 96 bytes per party less).
//
// Mapping of Figure 5 to pi_dpf():
//   step 1  W <- F_coin;  (s || t)_b^(0,0) := Delta_b XOR W          "zhang-root"
//   step 2  CW_b^(i) := XOR_j H0(node_b^(i,j)) XOR Delta_b
//           XOR alpha_b^(i) Delta_b XOR K_b[alpha_{1-b}^(i)] XOR M_b[alpha_b^(i)];
//           exchange; CW^(i) := CW_0^(i) XOR CW_1^(i);
//           left := H0(node) XOR t CW^(i), right := left XOR node    "zhang-levels"
//   step 3  CW_b^(n) := XOR_j H1(leaf_b^(j)) XOR (beta_b || M_b[beta]);
//           exchange; <u^(j)>_b := (t, leaf),
//           (v_b^(j) || M_b[v^(j)]) := H1(leaf) XOR t CW^(n)         "zhang-leaf"
//   step 4  chi <- F_coin;
//           <a> := sum_j chi^j <u^(j)> + sum_j chi^(N+j) <v^(j)> + <r>
//   step 5  open a (both send their share), BatchCheck(<a>, a~)      "zhang-check"
//   step 6  output <u^(j)>, <v^(j)>
// The authentication of the n + 128 input bits is booked as "zhang-input";
// the authenticated bits themselves are booked by the prep object ("baseot",
// "cotext", "abit").
//
// Bytes sent by ONE party in run_pi_dpf, per phase (asserted by the tests):
//   zhang-input ceil((n + 128) / 8), zhang-root 96, zhang-levels 16 n,
//   zhang-leaf 32, zhang-check 288 (192 without the unused coin);
//   total 16 n + ceil((n + 128) / 8) + 416.  Rounds: 1, 2, n, 1, 7 (5), i.e.
//   n + 11 (n + 9).  Authenticated bits: u_message_bytes(n + 256) + 16 + 2064
//   bytes per party and three rounds per key; base OT 12,288 bytes per party
//   and two rounds per session.
//
// ---------------------------------------------------------------------------
// (2) Semi-honest baseline: dealer-free DPF generation in the Doerner-shelat
//     STYLE (J. Doerner, a. shelat, "Scaling ORAM for Secure Computation",
//     ACM CCS 2017): local full-domain expansion and one secure selection per
//     level, no MACs, no check.  It is NOT their protocol bit for bit (they
//     use a length-doubling PRG with two seeds per node and a garbled circuit
//     per level).  It is the semi-honest counterpart of (1) ON THE SAME TREE:
//     the same root Delta_b XOR W, the same H0 (domain label
//     "baseline-semihonest/H0", ssid tag "Couplet-baseline-semihonest") and
//     the same expansion, so that the two baselines differ only in what active
//     security adds.  Per level one correlated OT per direction with choice
//     bit alpha_b^(i) gives the share of alpha^(i) Delta; the COT rows come
//     from PlainCot, an IKNP extension WITHOUT padding rows, challenge or
//     check message (chosen choice bits, PRG domain prg_dom(0x30, direction,
//     call)), on the same base OT as (1).  W is the XOR of two strings sent in
//     the clear.  The final correction word carries the payload only.
//   Bytes per party and key: sh-cot 1024 ceil(n / 64) + 16, sh-levels 16 n,
//   sh-leaf 16; rounds 1, n, 1.  Base OT ("baseot"): 12,288 bytes, two rounds.
//
// ---------------------------------------------------------------------------
// Implementation notes (this header; none of them changes a message).
//   * Tree kernel.  After the hash pass of level i the party keeps, for every
//     node, the pair (H0(node), node); once CW^(i) has arrived, ONE pass
//     derives the two children of every node, hashes them and writes the pairs
//     of level i + 1.  The correction of the last level is applied by the pass
//     that hashes the leaves, so that part of the local work of step 2 is
//     timed under "zhang-leaf" / "sh-leaf".  Bytes and rounds per phase are
//     not affected.
//   * The linear combination of step 4 is evaluated with a table of powers of
//     chi and lazily reduced inner products (PCLMULQDQ when available); the
//     result is the polynomial value of the Python reference.
//   * Worker threads.  LeakyOptions::threads / SemiHonestSession::set_threads
//     let the tree passes and the step-4 sums of ONE party use several
//     threads (default 1); messages and outputs do not depend on it.
//   * Randomness.  Commitment openings and coin shares of the leaky protocol
//     always come from the operating system's generator; the key share, the
//     base-OT scalars and the W share of a semi-honest session come from its
//     RandomSource (default: the operating system's generator).
//   * Memory.  Outputs: 49 bytes per leaf and party (leaky), 17 (semi-honest);
//     tree buffers: 32 bytes per leaf summed over the levels, at most 24 bytes
//     per leaf alive at a time.  Nothing is reused between keys.
//   * Test hooks (LeakyCheat, default none; the honest path only tests the
//     pointer): an additive error on the own share of CW^(level) which is
//     removed again on the two children of a guessed node, the same for the
//     final correction word with a guessed leaf, and an error on the own share
//     of the opened value a.
//   * namespace testonly: a verification step for benchmarks and
//     interoperability runs that is NOT part of either protocol (the parties
//     reveal their key shares and inputs to each other).
#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "abits.hpp"
#include "baseot.hpp"
#include "channel.hpp"
#include "gf128.hpp"
#include "hashes.hpp"
#include "params.hpp"
#include "prep.hpp"
#include "rng.hpp"
#include "shares.hpp"

namespace couplet {
namespace baselines {

using prep::BitVec;
using prep::GfVec;
using prep::KAPPA;
using prep::PhaseStat;

const unsigned MAX_N = 26;                   // 2^26 leaves: about 3.1 GiB of outputs per party in the leaky protocol
const size_t ELEM_BYTES = 16;                // one element of GF(2^128) on the wire
const size_t COIN_BYTES = 96;                // per party and coin: commitment, opening, 32-byte share
const size_t COMMIT_OVERHEAD_BYTES = 64;     // per party and committed message: commitment and opening
const unsigned USE_PLAIN_COT = 0x30;         // ctr_prg use byte of the unchecked extension (prep uses 0x10 - 0x14)

inline const char* label_h0_leaky() { return "baseline-zhang/H0"; }
inline const char* label_h0_semihonest() { return "baseline-semihonest/H0"; }

// ------------------------------------------------------------------ accounting
enum Phase {
    BP_NONE = -1,
    BP_ZH_INPUT = 0,  // "zhang-input"
    BP_ZH_ROOT,       // "zhang-root"
    BP_ZH_LEVELS,     // "zhang-levels"
    BP_ZH_LEAF,       // "zhang-leaf"
    BP_ZH_CHECK,      // "zhang-check"
    BP_SH_BASEOT,     // "baseot" (semi-honest session; the leaky protocol's base OT is booked by the prep object)
    BP_SH_COT,        // "sh-cot"
    BP_SH_LEVELS,     // "sh-levels"
    BP_SH_LEAF,       // "sh-leaf"
    BP_COUNT
};

inline const char* phase_name(int phase) {
    static const char* names[BP_COUNT] = {"zhang-input", "zhang-root", "zhang-levels", "zhang-leaf", "zhang-check",
                                          "baseot",      "sh-cot",     "sh-levels",    "sh-leaf"};
    return (phase >= 0 && phase < BP_COUNT) ? names[phase] : "(none)";
}

// Bytes, messages, rounds and seconds of one party per baseline phase.
struct PhaseStats {
    PhaseStat phase[BP_COUNT];
    const PhaseStat& operator[](int p) const { return phase[p]; }
    PhaseStat sum(int first, int last) const {  // phases first .. last inclusive
        PhaseStat t;
        for (int p = first; p <= last; p++) t += phase[p];
        return t;
    }
};

// Attributes the traffic of a channel endpoint and the elapsed time to the
// baseline phase that is current (the counterpart of Channel.mark of the
// Python package).  BP_NONE means "not ours": nothing is booked, so the meter
// of a RealPrep object on the same endpoint can account for the preprocessing.
class Meter {
public:
    explicit Meter(Channel& ch) : ch_(&ch), cur_(BP_NONE) { snap(); }

    int phase() const { return cur_; }

    // Start phase p; returns the phase that was current.
    int mark(int p) {
        close();
        const int prev = cur_;
        cur_ = p;
        return prev;
    }

    // Statistics so far (the running phase is included up to now).
    const PhaseStats& stats() {
        close();
        return stats_;
    }

private:
    void snap() {
        snap_ = ch_->stats();
        t_ = std::chrono::steady_clock::now();
    }
    void close() {
        const ChannelStats now = ch_->stats();
        const std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
        if (cur_ >= 0 && cur_ < BP_COUNT) {
            PhaseStat& s = stats_.phase[cur_];
            s.bytes_sent += now.bytes_sent - snap_.bytes_sent;
            s.bytes_recv += now.bytes_recv - snap_.bytes_recv;
            s.msgs_sent += now.msgs_sent - snap_.msgs_sent;
            s.msgs_recv += now.msgs_recv - snap_.msgs_recv;
            s.rounds += now.rounds - snap_.rounds;
            s.seconds += std::chrono::duration<double>(t - t_).count();
        }
        snap_ = now;
        t_ = t;
    }

    Channel* ch_;
    PhaseStats stats_;
    int cur_;
    ChannelStats snap_;
    std::chrono::steady_clock::time_point t_;
};

namespace detail {
inline int mark(Meter* meter, int phase) { return meter ? meter->mark(phase) : (int)BP_NONE; }

// Restores the phase that was current when the scope was entered.
class MeterScope {
public:
    explicit MeterScope(Meter* meter) : meter_(meter), prev_(meter ? meter->phase() : (int)BP_NONE) {}
    ~MeterScope() {
        if (meter_) meter_->mark(prev_);
    }
    MeterScope(const MeterScope&) = delete;
    MeterScope& operator=(const MeterScope&) = delete;

private:
    Meter* meter_;
    int prev_;
};
}  // namespace detail

// ------------------------------------------------------------------ formulas (asserted by the tests)
// Random BDOZ bits of each party consumed by run_pi_dpf: n + 128 inputs and 128 for the mask r.
inline size_t abit_demand(unsigned n) { return (size_t)n + 2 * KAPPA; }

// Size of the U message of one PlainCot call with ell choice bits.
inline size_t cot_message_bytes(size_t ell) { return KAPPA * 8 * ((ell + 63) / 64); }

// Exact bytes sent by ONE party in run_pi_dpf, for a leaky phase (excluding the prep phases).
inline uint64_t leaky_bytes_per_party(unsigned n, int phase, bool skip_unused_coin = false) {
    switch (phase) {
        case BP_ZH_INPUT: return ((uint64_t)n + KAPPA + 7) / 8;
        case BP_ZH_ROOT: return COIN_BYTES;
        case BP_ZH_LEVELS: return ELEM_BYTES * (uint64_t)n;
        case BP_ZH_LEAF: return 2 * ELEM_BYTES;
        case BP_ZH_CHECK: return (skip_unused_coin ? 1 : 2) * COIN_BYTES + ELEM_BYTES + COMMIT_OVERHEAD_BYTES + ELEM_BYTES;
        default: return 0;
    }
}
inline uint64_t leaky_bytes_per_party(unsigned n, bool skip_unused_coin = false) {
    uint64_t t = 0;
    for (int p = BP_ZH_INPUT; p <= BP_ZH_CHECK; p++) t += leaky_bytes_per_party(n, p, skip_unused_coin);
    return t;
}

// Rounds of run_pi_dpf, for a leaky phase (excluding the prep phases).
inline uint64_t leaky_rounds(unsigned n, int phase, bool skip_unused_coin = false) {
    switch (phase) {
        case BP_ZH_INPUT: return 1;
        case BP_ZH_ROOT: return 2;
        case BP_ZH_LEVELS: return n;
        case BP_ZH_LEAF: return 1;
        case BP_ZH_CHECK: return skip_unused_coin ? 5 : 7;
        default: return 0;
    }
}
inline uint64_t leaky_rounds(unsigned n, bool skip_unused_coin = false) {
    uint64_t t = 0;
    for (int p = BP_ZH_INPUT; p <= BP_ZH_CHECK; p++) t += leaky_rounds(n, p, skip_unused_coin);
    return t;
}

// Exact bytes sent by ONE party per semi-honest key and phase (the base OT is not included).
inline uint64_t semihonest_bytes_per_party(unsigned n, int phase) {
    switch (phase) {
        case BP_SH_COT: return cot_message_bytes(n) + ELEM_BYTES;
        case BP_SH_LEVELS: return ELEM_BYTES * (uint64_t)n;
        case BP_SH_LEAF: return ELEM_BYTES;
        default: return 0;
    }
}
inline uint64_t semihonest_bytes_per_party(unsigned n) {
    return semihonest_bytes_per_party(n, BP_SH_COT) + semihonest_bytes_per_party(n, BP_SH_LEVELS) +
           semihonest_bytes_per_party(n, BP_SH_LEAF);
}

// Rounds per semi-honest key and phase (the base OT adds two rounds once per session).
inline uint64_t semihonest_rounds(unsigned n, int phase) {
    switch (phase) {
        case BP_SH_COT: return 1;
        case BP_SH_LEVELS: return n;
        case BP_SH_LEAF: return 1;
        default: return 0;
    }
}
inline uint64_t semihonest_rounds(unsigned n) { return (uint64_t)n + 2; }

// ------------------------------------------------------------------ data model
// One party's part [[x]]_b of BDOZ-style authenticated bits (Section 3.3 of the paper):
//   bits  own shares x_b;
//   keys  K_b[x_{1-b}], keys for the peer's shares;
//   macs  M_b[x_b] = K_{1-b}[x_b] XOR x_b Delta_{1-b}.
// The authenticated bit is x = x_0 XOR x_1.
struct BdozBits {
    BitVec bits;
    GfVec keys;
    GfVec macs;

    BdozBits() {}
    BdozBits(BitVec b, GfVec k, GfVec m) : bits(std::move(b)), keys(std::move(k)), macs(std::move(m)) { validate(); }

    size_t size() const { return bits.size(); }
    void validate() const {
        if (bits.size() != keys.size() || bits.size() != macs.size())
            throw std::invalid_argument("bits, keys and macs must have the same length");
        prep::check_bits(bits);
    }
    // Bits at positions [a, b).
    BdozBits slice(size_t a, size_t b) const {
        if (a > b || b > size()) throw std::out_of_range("BdozBits::slice");
        BdozBits r;
        r.bits.assign(bits.begin() + (ptrdiff_t)a, bits.begin() + (ptrdiff_t)b);
        r.keys.assign(keys.begin() + (ptrdiff_t)a, keys.begin() + (ptrdiff_t)b);
        r.macs.assign(macs.begin() + (ptrdiff_t)a, macs.begin() + (ptrdiff_t)b);
        return r;
    }
};

// One party's part <x>_b = (x_b, M_b[x]) of a SPDZ-style sharing of x in GF(2^128).
struct FieldShare {
    gf128 value;
    gf128 mac;
    FieldShare() {}
    FieldShare(const gf128& v, const gf128& m) : value(v), mac(m) {}
};

// One party's shares of a vector of field elements.  Across the parties the
// values XOR to v^(j) and the MAC shares to v^(j) * Delta.
struct FieldShares {
    GfVec values;
    GfVec macs;
    size_t size() const { return values.size(); }
};

// The public values seen by a party of Pi_DPF.
struct LeakyTranscript {
    gf128 W;
    GfVec cws;                     // CW^(0) .. CW^(n-1)
    gf128 cw_final_val, cw_final_mac;  // the two halves of CW^(n)
    gf128 chi;
    gf128 a;
    friend bool operator==(const LeakyTranscript& x, const LeakyTranscript& y) {
        return x.W == y.W && x.cws == y.cws && x.cw_final_val == y.cw_final_val && x.cw_final_mac == y.cw_final_mac &&
               x.chi == y.chi && x.a == y.a;
    }
};

// Output of Pi_DPF at one party (step 6 of Figure 5).
//   u  authenticated bits of unit(N, alpha) in the format of the Boolean
//      layer (prep::Shares: bits and MAC shares);
//   v  shares of the N field elements unit(N, alpha) * beta.
struct LeakyDpfOutput {
    unsigned n = 0;
    prep::Shares u;
    FieldShares v;
    LeakyTranscript transcript;
};

// Test hooks of the leaky protocol (see the header comment).
struct LeakyCheat {
    bool has_cw_error = false;  // add cw_error to the own share of CW^(cw_level), 0 <= cw_level < n
    unsigned cw_level = 0;
    gf128 cw_error;
    bool cw_has_guess = false;  // ... and XOR it into the two children of node cw_guess of that level
    uint64_t cw_guess = 0;      //     (the guess "the cw_level most significant bits of alpha equal cw_guess")
    bool has_final_error = false;  // add (final_error_v, final_error_m) to the own share of CW^(n)
    gf128 final_error_v, final_error_m;
    bool final_has_guess = false;  // ... and remove it at leaf final_guess
    uint64_t final_guess = 0;
    gf128 a_error;                 // added to the own share of the opened a

    static LeakyCheat cw(unsigned level, const gf128& err) {
        LeakyCheat c;
        c.has_cw_error = true;
        c.cw_level = level;
        c.cw_error = err;
        return c;
    }
    static LeakyCheat cw(unsigned level, const gf128& err, uint64_t guess) {
        LeakyCheat c = cw(level, err);
        c.cw_has_guess = true;
        c.cw_guess = guess;
        return c;
    }
    static LeakyCheat final_cw(const gf128& err_v, const gf128& err_m) {
        LeakyCheat c;
        c.has_final_error = true;
        c.final_error_v = err_v;
        c.final_error_m = err_m;
        return c;
    }
    static LeakyCheat final_cw(const gf128& err_v, const gf128& err_m, uint64_t guess) {
        LeakyCheat c = final_cw(err_v, err_m);
        c.final_has_guess = true;
        c.final_guess = guess;
        return c;
    }
    static LeakyCheat a(const gf128& err) {
        LeakyCheat c;
        c.a_error = err;
        return c;
    }
};

struct LeakyOptions {
    uint64_t key_id = 0;              // number of this key under the session identifier (never reused)
    const LeakyCheat* cheat = nullptr;
    bool skip_unused_coin = false;    // omit the unused coin of BatchCheck (Figure 2, step 1)
    unsigned threads = 1;             // worker threads of this party for the local passes
};

// XOR shares of one party in the semi-honest baseline.
struct SemiHonestTranscript {
    gf128 W;
    GfVec cws;
    gf128 cw_final;
    friend bool operator==(const SemiHonestTranscript& x, const SemiHonestTranscript& y) {
        return x.W == y.W && x.cws == y.cws && x.cw_final == y.cw_final;
    }
};
struct SemiHonestDpfOutput {
    unsigned n = 0;
    BitVec t;  // shares of unit(N, alpha)
    GfVec v;   // shares of unit(N, alpha) * beta
    SemiHonestTranscript transcript;
};

// ------------------------------------------------------------------ local building blocks
inline void check_n(size_t n) {
    if (n < 1 || n > MAX_N) throw std::invalid_argument("n must be in 1.." + std::to_string(MAX_N));
}

// Identifier of one key generation: SHA-256(frame(tag, sid, le64(key_id))).
inline bytes sub_sid(const char* tag, const bytes& sid, uint64_t key_id) {
    const hashes::hash32 h = hashes::framed_sha256({tag, sid, hashes::le64(key_id)});
    return bytes(h.begin(), h.end());
}
inline bytes leaky_sub_sid(const bytes& sid, uint64_t key_id) { return sub_sid("Couplet-baseline-zhang", sid, key_id); }
inline bytes semihonest_sub_sid(const bytes& sid, uint64_t key_id) {
    return sub_sid("Couplet-baseline-semihonest", sid, key_id);
}

// Ascon domain constant of H0 for the key generation ssid (family 0x48, CCR hash).
inline uint64_t h0_domain(const bytes& ssid, const char* label = label_h0_leaky()) {
    return prep::session_dom(prep::DOM_CCR, label, ssid);
}

// H0 on every element: the untweaked CCR hash of the paper.  out may alias x.
inline void hash0(const gf128* x, size_t n, uint64_t dom, gf128* out) {
    hashes::H_ccr_batch_same_tweak(x, n, 0, dom, out);
}
inline GfVec hash0(const GfVec& x, uint64_t dom) {
    GfVec out(x.size());
    hash0(x.data(), x.size(), dom, out.data());
    return out;
}

// H1(x) = H0(x) || H0(x XOR 1): the two halves (value part, MAC part).
inline void hash1(const GfVec& x, uint64_t dom, GfVec& h_val, GfVec& h_mac) {
    h_val = hash0(x, dom);
    GfVec flipped(x);
    for (gf128& e : flipped) e.lo ^= 1;
    h_mac = hash0(flipped, dom);
}

// One level of the tree (step 2 of Figure 5) in the plain form of the Python
// reference: `nodes` are the strings (s || t) of level i (t = least
// significant bit), hashed = H0(nodes), cw the public correction word.  Child
// 2j is H0(node_j) XOR t_j CW and child 2j + 1 is child 2j XOR node_j.
inline GfVec expand_level(const GfVec& nodes, const GfVec& hashed, const gf128& cw) {
    if (nodes.size() != hashed.size()) throw std::invalid_argument("one hash per node is required");
    GfVec out(2 * nodes.size());
    for (size_t j = 0; j < nodes.size(); j++) {
        const gf128 left = hashed[j] ^ prep::gf_and_bit(cw, (unsigned)(nodes[j].lo & 1));
        out[2 * j] = left;
        out[2 * j + 1] = left ^ nodes[j];
    }
    return out;
}

// The n bits of alpha, most significant first (entry i is used at level i).
inline BitVec index_bits(uint64_t alpha, unsigned n) {
    if (n > 63 || (alpha >> n) != 0) throw std::invalid_argument("index share out of range");
    BitVec bits(n);
    for (unsigned i = 0; i < n; i++) bits[i] = (uint8_t)((alpha >> (n - 1 - i)) & 1);
    return bits;
}

// x^e by square and multiply.
inline gf128 gf_pow(gf128 x, uint64_t e) {
    gf128 r = gf128::one();
    while (e) {
        if (e & 1) r = gf128_mul(r, x);
        x = gf128_sqr(x);
        e >>= 1;
    }
    return r;
}

namespace detail {

// 1, x, ..., x^(STEP-1) and x^STEP.
struct PowerTable {
    enum : size_t { STEP = 256 };
    gf128 pw[STEP];
    gf128 x_step;
    explicit PowerTable(const gf128& x) {
        pw[0] = gf128::one();
        for (size_t i = 1; i < STEP; i++) pw[i] = gf128_mul(pw[i - 1], x);
        x_step = gf128_mul(pw[STEP - 1], x);
    }
};

// sum_{j < n} x^j c_j for one contiguous range (Horner over blocks of STEP coefficients).
inline gf128 poly_range(const PowerTable& t, const gf128* c, size_t n) {
    const size_t full = n / PowerTable::STEP, rem = n % PowerTable::STEP;
    gf128 acc;
    if (rem) acc = prep::gf_dot(c + full * PowerTable::STEP, t.pw, rem);
    for (size_t b = full; b-- > 0;) acc = gf128_mul(acc, t.x_step) ^ prep::gf_dot(c + b * PowerTable::STEP, t.pw, PowerTable::STEP);
    return acc;
}
// The same for 0/1 coefficients (one byte each).
inline gf128 poly_range_bits(const PowerTable& t, const uint8_t* c, size_t n) {
    const size_t full = n / PowerTable::STEP, rem = n % PowerTable::STEP;
    gf128 acc;
    if (rem) acc = prep::gf_dot_bits(t.pw, c + full * PowerTable::STEP, rem);
    for (size_t b = full; b-- > 0;) acc = gf128_mul(acc, t.x_step) ^ prep::gf_dot_bits(t.pw, c + b * PowerTable::STEP, PowerTable::STEP);
    return acc;
}

const size_t PAR_MIN_COEFFS = (size_t)1 << 14;  // smaller sums stay in the calling thread
const size_t PAR_MIN_NODES = (size_t)1 << 12;   // smaller levels stay in the calling thread

// sum_{j < n} x^j c_j, split over the worker threads of the caller: part p
// evaluates its range relative to its first coefficient and is shifted by
// x^begin afterwards.
template <class Coeff, class RangeFn>
inline gf128 poly_parallel(const PowerTable& t, const gf128& x, const Coeff* c, size_t n, RangeFn range) {
    const unsigned workers = prep::detail::worker_count();
    std::vector<gf128> part_sum(workers);
    std::vector<size_t> part_begin(workers, 0);
    const unsigned parts = prep::detail::parallel_ranges(n, PowerTable::STEP, PAR_MIN_COEFFS, [&](size_t b, size_t e, unsigned part) {
        part_begin[part] = b;
        part_sum[part] = range(t, c + b, e - b);
    });
    gf128 acc;
    for (unsigned p = 0; p < parts && p < workers; p++)
        acc ^= part_begin[p] ? gf128_mul(part_sum[p], gf_pow(x, (uint64_t)part_begin[p])) : part_sum[p];
    return acc;
}
inline gf128 poly_eval(const PowerTable& t, const gf128& x, const gf128* c, size_t n) {
    return poly_parallel(t, x, c, n, [](const PowerTable& tt, const gf128* cc, size_t nn) { return poly_range(tt, cc, nn); });
}
inline gf128 poly_eval_bits(const PowerTable& t, const gf128& x, const uint8_t* c, size_t n) {
    return poly_parallel(t, x, c, n, [](const PowerTable& tt, const uint8_t* cc, size_t nn) { return poly_range_bits(tt, cc, nn); });
}

}  // namespace detail

// sum_j x^j * coeffs[j] over GF(2^128).
inline gf128 poly_eval(const gf128* coeffs, size_t n, const gf128& x) {
    if (n == 0) return gf128();
    const detail::PowerTable t(x);
    return detail::poly_eval(t, x, coeffs, n);
}
inline gf128 poly_eval(const GfVec& coeffs, const gf128& x) { return poly_eval(coeffs.data(), coeffs.size(), x); }
// The same for 0/1 coefficients.
inline gf128 poly_eval_bits(const uint8_t* bits, size_t n, const gf128& x) {
    if (n == 0) return gf128();
    const detail::PowerTable t(x);
    return detail::poly_eval_bits(t, x, bits, n);
}

// Convert of Section 3.3: <x>_b := (x_b, K_b[x_{1-b}] XOR M_b[x_b] XOR x_b Delta_b).
inline prep::Shares convert(const gf128& delta, const BdozBits& x) {
    x.validate();
    prep::Shares s;
    s.bits = x.bits;
    s.macs = x.macs;
    prep::spdz_macs_inplace(delta, s.bits.data(), s.macs.data(), x.keys.data(), s.size());
    return s;
}

// Pack of Figure 4 (B2F, then Convert) for exactly 128 bits: <sum_i X^i x^(i)>_b.
inline FieldShare pack(const gf128& delta, const BdozBits& x) {
    const std::pair<gf128, gf128> f = convert(delta, x).to_field();
    return FieldShare(f.first, f.second);
}

// ------------------------------------------------------------------ F_aBit with chosen inputs
// F_aBit command (auth, b, x, l) for BOTH parties at once (one round,
// ceil(l / 8) bytes each).  own / keys are l random BDOZ bits of each party
// (this party's bits rho_b with M_b[rho_b], and its keys K_b[rho_{1-b}] for
// the peer's bits).  Party b sends d_b = x_b XOR rho_b; afterwards
// M_b[x_b] := M_b[rho_b] and K_b[x_{1-b}] := K_b[rho_{1-b}] XOR d_{1-b} Delta_b.
inline BdozBits authenticate_inputs(Channel& ch, const gf128& delta, const prep::Bdoz& own, const prep::Keys& keys,
                                    const BitVec& bits) {
    prep::check_bits(bits);
    if (own.size() != bits.size() || keys.size() != bits.size())
        throw std::invalid_argument("one random BDOZ bit of each party is needed per input bit");
    BitVec d(bits.size());
    for (size_t i = 0; i < bits.size(); i++) d[i] = (uint8_t)(bits[i] ^ own.bits[i]);
    const BitVec d_peer = prep::exchange_bits(ch, d);
    BdozBits x;
    x.bits = bits;
    x.macs = own.macs;
    x.keys = keys.keys;
    for (size_t i = 0; i < bits.size(); i++) x.keys[i] ^= prep::gf_and_bit(delta, d_peer[i]);
    return x;
}

// ------------------------------------------------------------------ small wire helpers
namespace detail {

inline bytes concat_label(const char* prefix, const bytes& label) {
    bytes l(prefix, prefix + std::strlen(prefix));
    l.insert(l.end(), label.begin(), label.end());
    return l;
}
inline bytes ascii(const char* s) { return bytes(s, s + std::strlen(s)); }

// A jointly random element of GF(2^128): the 16 coin bytes as a little-endian integer.
template <class ByteSource>
inline gf128 coin_elem(Channel& ch, const bytes& ssid, const bytes& label, ByteSource& rng) {
    const bytes coin = hashes::coin_toss(ch, ssid, label, ELEM_BYTES, rng);
    return gf128::from_bytes(coin.data());
}

// Exchange `count` field elements; returns the peer's elements.
inline GfVec exchange_elems(Channel& ch, const gf128* mine, size_t count, const char* what) {
    const bytes reply = ch.exchange(prep::gf_to_bytes(mine, count));
    if (reply.size() != ELEM_BYTES * count) throw ProtocolAbort(std::string(what) + ": message has the wrong length");
    GfVec out(count);
    prep::gf_load(out.data(), reply.data(), count);
    return out;
}
inline gf128 exchange_elem(Channel& ch, const gf128& mine, const char* what) { return exchange_elems(ch, &mine, 1, what)[0]; }

}  // namespace detail

// ------------------------------------------------------------------ BatchCheck (Figure 2)
// BatchCheck on the opened values y~^(i) with this party's MAC shares
// M_b[y^(i)]:  1. chi <- F_coin;  2. <z> := sum_i chi^i <y^(i)>,
// z~ := sum_i chi^i y~^(i);  3. V_b := M_b[z] XOR z~ Delta_b, committed;
// 4. both open, abort unless V_0 = V_1.  Four rounds (two if the coin is
// skipped, which is allowed only for a single value).  Throws
// ProtocolAbort("BatchCheck failed") if V_0 != V_1.
template <class ByteSource>
inline void batch_check(Channel& ch, const bytes& ssid, const bytes& label, const gf128& delta, const GfVec& opened,
                        const GfVec& mac_shares, ByteSource& rng, bool skip_unused_coin = false) {
    if (opened.size() != mac_shares.size() || opened.empty())
        throw std::invalid_argument("one MAC share per opened value is required (at least one value)");
    gf128 chi;
    if (opened.size() > 1 || !skip_unused_coin) chi = detail::coin_elem(ch, ssid, detail::concat_label("batchcheck|", label), rng);
    gf128 z_open, z_mac, power = gf128::one();
    for (size_t i = 0; i < opened.size(); i++) {
        z_open ^= gf128_mul(power, opened[i]);
        z_mac ^= gf128_mul(power, mac_shares[i]);
        power = gf128_mul(power, chi);
    }
    const bytes v_own = prep::gf_to_bytes(z_mac ^ gf128_mul(z_open, delta));
    const bytes v_peer = hashes::committed_exchange(ch, ssid, detail::concat_label("batchcheck-V|", label), v_own, rng);
    if (v_peer.size() != ELEM_BYTES || !hashes::ct_equal(v_peer.data(), v_own.data(), ELEM_BYTES))
        throw ProtocolAbort("BatchCheck failed");
}

// ------------------------------------------------------------------ tree kernels
namespace detail {

// A buffer of 64-bit words without initialisation (level buffers of the tree).
class WordBuffer {
public:
    WordBuffer() : n_(0) {}
    explicit WordBuffer(size_t words) : p_(new uint64_t[words]), n_(words) {}
    uint64_t* data() { return p_.get(); }
    const uint64_t* data() const { return p_.get(); }
    size_t size() const { return n_; }
    void swap(WordBuffer& o) {
        p_.swap(o.p_);
        std::swap(n_, o.n_);
    }
    void release() {
        p_.reset();
        n_ = 0;
    }

private:
    std::unique_ptr<uint64_t[]> p_;
    size_t n_;
};

#if COUPLET_HAVE_AVX2
inline gf128 fold256(__m256i acc) {
    alignas(32) uint64_t t[4];
    _mm256_store_si256((__m256i*)t, acc);
    return gf128(t[0] ^ t[2], t[1] ^ t[3]);
}

// For a register v = [H0(node) | node]: the two children [left | right],
// left = H0(node) XOR t CW, right = left XOR node, t = lsb(node).
// cw2 = [CW | CW]; lsb2 selects the least significant bit of lane 2.
inline __m256i children256(__m256i v, __m256i cw2, __m256i lsb2) {
    const __m256i t = _mm256_sub_epi64(_mm256_setzero_si256(), _mm256_and_si256(v, lsb2));  // lane 2: all ones iff t = 1
    const __m256i m = _mm256_permute4x64_epi64(t, 0xAA);                                     // that lane in all four
    return _mm256_xor_si256(_mm256_xor_si256(v, _mm256_and_si256(m, cw2)), _mm256_permute2x128_si256(v, v, 0x08));
}

// H0 of the four elements held as a = [e0 | e1], b = [e2 | e3], optionally
// with `flip` XORed into the low word of every element: ha = [H0(e0) | H0(e1)],
// hb = [H0(e2) | H0(e3)].
inline void hash4(__m256i a, __m256i b, __m256i dom, __m256i flip, __m256i& ha, __m256i& hb) {
    __m256i x0 = _mm256_xor_si256(_mm256_unpacklo_epi64(a, b), flip);  // lo0 lo2 lo1 lo3
    __m256i x1 = _mm256_unpackhi_epi64(a, b);                          // hi0 hi2 hi1 hi3
    __m256i x2 = _mm256_setzero_si256(), x3 = dom, x4 = _mm256_setzero_si256();
    prep::detail::p12x4(x0, x1, x2, x3, x4);
    ha = _mm256_unpacklo_epi64(x0, x1);  // elements 0, 1
    hb = _mm256_unpackhi_epi64(x0, x1);  // elements 2, 3
}
#endif

// Scalar form of the child derivation: p = (h_lo, h_hi, x_lo, x_hi) of one node.
inline void children_scalar(const uint64_t* p, const gf128& cw, gf128& left, gf128& right) {
    const uint64_t m = 0 - (p[2] & 1);
    left = gf128(p[0] ^ (cw.lo & m), p[1] ^ (cw.hi & m));
    right = gf128(left.lo ^ p[2], left.hi ^ p[3]);
}

// Level pass for the nodes [j0, j1) of a level with pairs `in` (four words
// per node: H0(node), node) and correction word cw: writes the pairs of their
// children to `out` (child c at words 4c .. 4c + 3) and returns the XOR of the
// children's hashes.  j0 must be even.
inline gf128 level_pass_range(const uint64_t* in, size_t j0, size_t j1, const gf128& cw, uint64_t dom, uint64_t* out) {
    gf128 acc;
    size_t j = j0;
#if COUPLET_HAVE_AVX2
    const __m256i cw2 = _mm256_set_epi64x((long long)cw.hi, (long long)cw.lo, (long long)cw.hi, (long long)cw.lo);
    const __m256i lsb2 = _mm256_set_epi64x(0, 1, 0, 0), d = _mm256_set1_epi64x((long long)dom), zero = _mm256_setzero_si256();
    __m256i acc2 = zero;
    for (; j + 2 <= j1; j += 2) {
        const __m256i ca = children256(_mm256_loadu_si256((const __m256i*)(in + 4 * j)), cw2, lsb2);
        const __m256i cb = children256(_mm256_loadu_si256((const __m256i*)(in + 4 * j + 4)), cw2, lsb2);
        __m256i ha, hb;
        hash4(ca, cb, d, zero, ha, hb);
        uint64_t* o = out + 8 * j;
        _mm256_storeu_si256((__m256i*)o, _mm256_permute2x128_si256(ha, ca, 0x20));
        _mm256_storeu_si256((__m256i*)(o + 4), _mm256_permute2x128_si256(ha, ca, 0x31));
        _mm256_storeu_si256((__m256i*)(o + 8), _mm256_permute2x128_si256(hb, cb, 0x20));
        _mm256_storeu_si256((__m256i*)(o + 12), _mm256_permute2x128_si256(hb, cb, 0x31));
        acc2 = _mm256_xor_si256(acc2, _mm256_xor_si256(ha, hb));
    }
    acc = fold256(acc2);
#endif
    for (; j < j1; j++) {
        gf128 c[2];
        children_scalar(in + 4 * j, cw, c[0], c[1]);
        uint64_t* o = out + 8 * j;
        for (int k = 0; k < 2; k++) {
            const gf128 h = hashes::H_ccr(c[k], 0, dom);
            o[4 * k] = h.lo;
            o[4 * k + 1] = h.hi;
            o[4 * k + 2] = c[k].lo;
            o[4 * k + 3] = c[k].hi;
            acc ^= h;
        }
    }
    return acc;
}

// Leaf pass of the leaky protocol for the nodes [j0, j1) of the last inner
// level: leaves 2j, 2j + 1, their control bits, H1 = (hv, hm) of every leaf;
// the XORs of the two halves are added to acc_v / acc_m.
inline void leaky_leaf_range(const uint64_t* in, size_t j0, size_t j1, const gf128& cw, uint64_t dom, gf128* leaves,
                             uint8_t* tbits, gf128* hv, gf128* hm, gf128& acc_v, gf128& acc_m) {
    size_t j = j0;
#if COUPLET_HAVE_AVX2
    const __m256i cw2 = _mm256_set_epi64x((long long)cw.hi, (long long)cw.lo, (long long)cw.hi, (long long)cw.lo);
    const __m256i lsb2 = _mm256_set_epi64x(0, 1, 0, 0), d = _mm256_set1_epi64x((long long)dom), zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi64x(1);
    __m256i av = zero, am = zero;
    for (; j + 2 <= j1; j += 2) {
        const __m256i ca = children256(_mm256_loadu_si256((const __m256i*)(in + 4 * j)), cw2, lsb2);
        const __m256i cb = children256(_mm256_loadu_si256((const __m256i*)(in + 4 * j + 4)), cw2, lsb2);
        _mm256_storeu_si256((__m256i*)(leaves + 2 * j), ca);
        _mm256_storeu_si256((__m256i*)(leaves + 2 * j + 2), cb);
        __m256i ha, hb;
        hash4(ca, cb, d, zero, ha, hb);
        _mm256_storeu_si256((__m256i*)(hv + 2 * j), ha);
        _mm256_storeu_si256((__m256i*)(hv + 2 * j + 2), hb);
        av = _mm256_xor_si256(av, _mm256_xor_si256(ha, hb));
        hash4(ca, cb, d, one, ha, hb);
        _mm256_storeu_si256((__m256i*)(hm + 2 * j), ha);
        _mm256_storeu_si256((__m256i*)(hm + 2 * j + 2), hb);
        am = _mm256_xor_si256(am, _mm256_xor_si256(ha, hb));
        for (size_t k = 0; k < 4; k++) tbits[2 * j + k] = (uint8_t)(leaves[2 * j + k].lo & 1);
    }
    acc_v ^= fold256(av);
    acc_m ^= fold256(am);
#endif
    for (; j < j1; j++) {
        gf128 c[2];
        children_scalar(in + 4 * j, cw, c[0], c[1]);
        for (size_t k = 0; k < 2; k++) {
            const size_t leaf = 2 * j + k;
            leaves[leaf] = c[k];
            tbits[leaf] = (uint8_t)(c[k].lo & 1);
            hv[leaf] = hashes::H_ccr(c[k], 0, dom);
            hm[leaf] = hashes::H_ccr(gf128(c[k].lo ^ 1, c[k].hi), 0, dom);
            acc_v ^= hv[leaf];
            acc_m ^= hm[leaf];
        }
    }
}

// Leaf pass of the semi-honest baseline: control bits and H0 of every leaf.
inline void semihonest_leaf_range(const uint64_t* in, size_t j0, size_t j1, const gf128& cw, uint64_t dom, uint8_t* tbits,
                                  gf128* hv, gf128& acc) {
    size_t j = j0;
#if COUPLET_HAVE_AVX2
    const __m256i cw2 = _mm256_set_epi64x((long long)cw.hi, (long long)cw.lo, (long long)cw.hi, (long long)cw.lo);
    const __m256i lsb2 = _mm256_set_epi64x(0, 1, 0, 0), d = _mm256_set1_epi64x((long long)dom), zero = _mm256_setzero_si256();
    __m256i av = zero;
    alignas(32) uint64_t c[8];
    for (; j + 2 <= j1; j += 2) {
        const __m256i ca = children256(_mm256_loadu_si256((const __m256i*)(in + 4 * j)), cw2, lsb2);
        const __m256i cb = children256(_mm256_loadu_si256((const __m256i*)(in + 4 * j + 4)), cw2, lsb2);
        _mm256_store_si256((__m256i*)c, ca);
        _mm256_store_si256((__m256i*)(c + 4), cb);
        __m256i ha, hb;
        hash4(ca, cb, d, zero, ha, hb);
        _mm256_storeu_si256((__m256i*)(hv + 2 * j), ha);
        _mm256_storeu_si256((__m256i*)(hv + 2 * j + 2), hb);
        av = _mm256_xor_si256(av, _mm256_xor_si256(ha, hb));
        for (size_t k = 0; k < 4; k++) tbits[2 * j + k] = (uint8_t)(c[2 * k] & 1);
    }
    acc ^= fold256(av);
#endif
    for (; j < j1; j++) {
        gf128 ch[2];
        children_scalar(in + 4 * j, cw, ch[0], ch[1]);
        for (size_t k = 0; k < 2; k++) {
            const size_t leaf = 2 * j + k;
            tbits[leaf] = (uint8_t)(ch[k].lo & 1);
            hv[leaf] = hashes::H_ccr(ch[k], 0, dom);
            acc ^= hv[leaf];
        }
    }
}

// The passes over a whole level, split over the worker threads of the caller.
inline gf128 level_pass(const uint64_t* in, size_t m, const gf128& cw, uint64_t dom, uint64_t* out) {
    std::vector<gf128> acc(prep::detail::worker_count());
    prep::detail::parallel_ranges(m, 2, PAR_MIN_NODES, [&](size_t b, size_t e, unsigned part) {
        acc[part] = level_pass_range(in, b, e, cw, dom, out);
    });
    gf128 total;
    for (const gf128& a : acc) total ^= a;
    return total;
}
inline void leaky_leaf_pass(const uint64_t* in, size_t m, const gf128& cw, uint64_t dom, gf128* leaves, uint8_t* tbits,
                            gf128* hv, gf128* hm, gf128& acc_v, gf128& acc_m) {
    const unsigned workers = prep::detail::worker_count();
    std::vector<gf128> av(workers), am(workers);
    prep::detail::parallel_ranges(m, 2, PAR_MIN_NODES, [&](size_t b, size_t e, unsigned part) {
        leaky_leaf_range(in, b, e, cw, dom, leaves, tbits, hv, hm, av[part], am[part]);
    });
    for (unsigned p = 0; p < workers; p++) {
        acc_v ^= av[p];
        acc_m ^= am[p];
    }
}
inline void semihonest_leaf_pass(const uint64_t* in, size_t m, const gf128& cw, uint64_t dom, uint8_t* tbits, gf128* hv,
                                 gf128& acc) {
    std::vector<gf128> av(prep::detail::worker_count());
    prep::detail::parallel_ranges(m, 2, PAR_MIN_NODES, [&](size_t b, size_t e, unsigned part) {
        semihonest_leaf_range(in, b, e, cw, dom, tbits, hv, av[part]);
    });
    for (const gf128& a : av) acc ^= a;
}

// v[j] ^= t_j * cw for all j < n.
inline void add_selected(gf128* v, const uint8_t* tbits, size_t n, const gf128& cw) {
    prep::detail::parallel_ranges(n, 4, 4 * PAR_MIN_NODES, [&](size_t b, size_t e, unsigned) {
        for (size_t j = b; j < e; j++) {
            const uint64_t m = 0 - (uint64_t)(tbits[j] & 1);
            v[j].lo ^= cw.lo & m;
            v[j].hi ^= cw.hi & m;
        }
    });
}

// The pair (H0(root), root) of level 0.
inline WordBuffer root_pair(const gf128& root, uint64_t dom, gf128& acc) {
    WordBuffer raw(4);
    const gf128 h = hashes::H_ccr(root, 0, dom);
    raw.data()[0] = h.lo;
    raw.data()[1] = h.hi;
    raw.data()[2] = root.lo;
    raw.data()[3] = root.hi;
    acc = h;
    return raw;
}

inline void validate_cheat(const LeakyCheat* cheat, size_t n) {
    if (!cheat) return;
    if (cheat->has_cw_error) {
        if (cheat->cw_level >= n) throw std::invalid_argument("cw_error: level must be in [0, n)");
        if (cheat->cw_has_guess && (cheat->cw_guess >> cheat->cw_level) != 0)
            throw std::invalid_argument("cw_error: guess must be a node index of that level");
    } else if (cheat->cw_has_guess) {
        throw std::invalid_argument("cw_error: a guess needs an error");
    }
    if (cheat->has_final_error) {
        if (cheat->final_has_guess && (cheat->final_guess >> n) != 0)
            throw std::invalid_argument("final_cw_error: guess must be a leaf index");
    } else if (cheat->final_has_guess) {
        throw std::invalid_argument("final_cw_error: a guess needs an error");
    }
}

}  // namespace detail

// ------------------------------------------------------------------ Pi_DPF (Figure 5)
// Protocol Pi_DPF for one party; see the header comment for the mapping.
//   ch      this party's channel endpoint (ch.party() == role);
//   sid, opt.key_id   session identifier and number of this key (never reused
//           under the same sid: they fix all labels and the domain of H0);
//   delta   this party's share Delta_role of the global key;
//   alpha   [[alpha^(i)]]_role for i < n, index bits most significant first;
//   beta    <beta>_role;
//   r       <r>_role, a random authenticated mask (step 4) unknown to both;
//   meter   optional phase accounting ("zhang-root" .. "zhang-check").
// Throws ProtocolAbort if the batch check fails or the peer misbehaves; the
// caller must then retire the session.  Invalid arguments throw
// std::invalid_argument before any message is sent.
inline LeakyDpfOutput pi_dpf(Channel& ch, const bytes& sid, int role, const gf128& delta, const BdozBits& alpha,
                             const FieldShare& beta, const FieldShare& r, const LeakyOptions& opt = LeakyOptions(),
                             Meter* meter = nullptr) {
    prep::check_role(role);
    if (ch.party() != role) throw std::invalid_argument("role does not match the channel endpoint");
    alpha.validate();
    const size_t n = alpha.size();
    check_n(n);
    detail::validate_cheat(opt.cheat, n);
    const LeakyCheat* cheat = opt.cheat;
    const detail::MeterScope scope(meter);
    const prep::WorkerScope workers(opt.threads);
    SystemRng sys;  // commitment openings and coin shares
    const bytes ssid = leaky_sub_sid(sid, opt.key_id);
    const uint64_t dom = h0_domain(ssid);
    const size_t big_n = (size_t)1 << n;

    LeakyDpfOutput out;
    out.n = (unsigned)n;
    LeakyTranscript& tr = out.transcript;

    // ---- Step 1: W <- F_coin;  (s || t)^(0,0)_b := Delta_b XOR W
    detail::mark(meter, BP_ZH_ROOT);
    tr.W = detail::coin_elem(ch, ssid, detail::ascii("W"), sys);

    // ---- Step 2: one correction word per level, whole tree expanded locally
    detail::mark(meter, BP_ZH_LEVELS);
    gf128 acc;  // XOR_j H0(node^(i,j)) of the current level
    detail::WordBuffer raw = detail::root_pair(delta ^ tr.W, dom, acc);
    size_t m = 1;  // nodes of the current level
    tr.cws.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (i > 0) {  // level i from the pairs of level i - 1 and CW^(i-1)
            detail::WordBuffer next(8 * m);
            acc = detail::level_pass(raw.data(), m, tr.cws[i - 1], dom, next.data());
            raw.swap(next);
            m *= 2;
        }
        // CW_b^(i) = XOR_j H0(node) XOR Delta_b XOR alpha_b^(i) Delta_b XOR K_b[alpha_{1-b}^(i)] XOR M_b[alpha_b^(i)]
        gf128 cw_own = acc ^ delta ^ prep::gf_and_bit(delta, alpha.bits[i]) ^ alpha.keys[i] ^ alpha.macs[i];
        const bool attacked = cheat && cheat->has_cw_error && cheat->cw_level == i;
        if (attacked) cw_own ^= cheat->cw_error;  // test hook: additive error E
        tr.cws[i] = cw_own ^ detail::exchange_elem(ch, cw_own, "Pi_DPF");  // CW^(i) = CW_0^(i) XOR CW_1^(i)
        if (attacked && cheat->cw_has_guess) {
            // test hook: remove E on the two children of the guessed on-path node (both contain H0(node))
            raw.data()[4 * cheat->cw_guess] ^= cheat->cw_error.lo;
            raw.data()[4 * cheat->cw_guess + 1] ^= cheat->cw_error.hi;
        }
    }

    // ---- Step 3: final correction word, authenticated outputs
    detail::mark(meter, BP_ZH_LEAF);
    out.u.bits.resize(big_n);
    out.u.macs.resize(big_n);  // M_b[u^(j)] = (s || t)_b^(n,j)
    out.v.values.resize(big_n);
    out.v.macs.resize(big_n);
    gf128 cw_fin[2];  // XOR_j H1(leaf): value half, MAC half
    detail::leaky_leaf_pass(raw.data(), m, tr.cws[n - 1], dom, out.u.macs.data(), out.u.bits.data(), out.v.values.data(),
                            out.v.macs.data(), cw_fin[0], cw_fin[1]);
    raw.release();
    cw_fin[0] ^= beta.value;  // CW_b^(n), first half:  XOR beta_b
    cw_fin[1] ^= beta.mac;    // second half:           XOR M_b[beta]
    if (cheat && cheat->has_final_error) {
        cw_fin[0] ^= cheat->final_error_v;
        cw_fin[1] ^= cheat->final_error_m;
    }
    const GfVec peer_fin = detail::exchange_elems(ch, cw_fin, 2, "Pi_DPF");
    tr.cw_final_val = cw_fin[0] ^ peer_fin[0];
    tr.cw_final_mac = cw_fin[1] ^ peer_fin[1];
    detail::add_selected(out.v.values.data(), out.u.bits.data(), big_n, tr.cw_final_val);  // v_b^(j)
    detail::add_selected(out.v.macs.data(), out.u.bits.data(), big_n, tr.cw_final_mac);    // M_b[v^(j)]
    if (cheat && cheat->has_final_error && cheat->final_has_guess) {
        out.v.values[cheat->final_guess] ^= cheat->final_error_v;
        out.v.macs[cheat->final_guess] ^= cheat->final_error_m;
    }

    // ---- Step 4: chi <- F_coin;  <a> := sum_j chi^j <u^(j)> + sum_j chi^(N+j) <v^(j)> + <r>
    detail::mark(meter, BP_ZH_CHECK);
    tr.chi = detail::coin_elem(ch, ssid, detail::ascii("chi"), sys);
    const detail::PowerTable table(tr.chi);
    gf128 chi_n = tr.chi;  // chi^N
    for (size_t i = 0; i < n; i++) chi_n = gf128_sqr(chi_n);
    const gf128 a_val = detail::poly_eval_bits(table, tr.chi, out.u.bits.data(), big_n) ^
                        gf128_mul(chi_n, detail::poly_eval(table, tr.chi, out.v.values.data(), big_n)) ^ r.value;
    const gf128 a_mac = detail::poly_eval(table, tr.chi, out.u.macs.data(), big_n) ^
                        gf128_mul(chi_n, detail::poly_eval(table, tr.chi, out.v.macs.data(), big_n)) ^ r.mac;

    // ---- Step 5: open a, then BatchCheck(<a>, a~)
    const gf128 a_sent = cheat ? (a_val ^ cheat->a_error) : a_val;
    tr.a = a_sent ^ detail::exchange_elem(ch, a_sent, "Pi_DPF");
    batch_check(ch, ssid, detail::ascii("a"), delta, GfVec(1, tr.a), GfVec(1, a_mac), sys, opt.skip_unused_coin);

    // ---- Step 6: outputs
    return out;
}

// Generate one leaky DPF from XOR shares of the index and of the payload.
// `prep` is this party's RealPrep (after setup()) or IdealPrep; it supplies
// the key share and random_bdoz.  The index is alpha = alpha_share_0 XOR
// alpha_share_1 (n bits), the payload beta = beta_share_0 XOR beta_share_1 in
// GF(2^128).  Steps: (1) one request for n + 256 random BDOZ bits per party
// (F_aBit); (2) authentication of the n + 128 input bits of each party
// ("zhang-input"), Pack of the payload bits and of the mask bits; (3) pi_dpf.
// On any protocol failure the prep object is retired (prep.abort()) and the
// exception propagates.  Invalid arguments throw std::invalid_argument before
// any message is sent and do not retire the session.
inline LeakyDpfOutput run_pi_dpf(Channel& ch, prep::Prep& prep, const bytes& sid, unsigned n, uint64_t alpha_share,
                                 const gf128& beta_share, const LeakyOptions& opt = LeakyOptions(),
                                 Meter* meter = nullptr) {
    const int role = prep.role();
    check_n(n);
    const BitVec a_bits = index_bits(alpha_share, n);
    const BitVec b_bits = prep::gf_unpack_bits(beta_share);
    detail::validate_cheat(opt.cheat, n);  // reject malformed hooks before any message is sent
    const detail::MeterScope scope(meter);
    detail::mark(meter, BP_NONE);  // the authenticated bits are booked by the prep object
    try {
        const prep::BdozPair pair = prep.random_bdoz(abit_demand(n));
        const gf128 delta = prep.delta();
        detail::mark(meter, BP_ZH_INPUT);
        const size_t n_in = (size_t)n + KAPPA;
        BitVec inputs(a_bits);
        inputs.insert(inputs.end(), b_bits.begin(), b_bits.end());
        const BdozBits x = authenticate_inputs(ch, delta, pair.own.slice(0, n_in), pair.keys.slice(0, n_in), inputs);
        const FieldShare beta = pack(delta, x.slice(n, n_in));                                               // <beta>
        const std::pair<gf128, gf128> rf =
            prep::spdz_shares(role, delta, pair.own.slice(n_in, n_in + KAPPA), pair.keys.slice(n_in, n_in + KAPPA)).to_field();
        return pi_dpf(ch, sid, role, delta, x.slice(0, n), beta, FieldShare(rf.first, rf.second), opt, meter);  // <r>
    } catch (...) {
        prep.abort();
        throw;
    }
}

// ------------------------------------------------------------------ unchecked COT extension
// Semi-honest IKNP correlated-OT extension in both directions, no consistency
// check.  Party b is the COT sender of direction b (global key Delta_b, seeds
// k_j^{Delta_j} from the base OT) and the receiver of direction 1 - b (seed
// pairs).  One call transfers ell CHOSEN bits per party:
//     receiver:  U_j = PRG(k_j^0) XOR PRG(k_j^1) XOR u   for all 128 columns j,
//     sender:    q_j = PRG(k_j^{Delta_j}) XOR Delta_j U_j ,
// so that row i of the receiver's matrix (M_i) and of the sender's matrix
// (K_i) satisfy M_i = K_i XOR u_i Delta.  start() produces this party's U
// message (128 columns of 64 ceil(ell / 64) bits, column after column, words
// little-endian), finish() consumes the peer's.  PRG = hashes::ctr_prg under
// prg_dom(USE_PLAIN_COT, direction, call).  Secure against semi-honest
// parties only: nothing stops a receiver from using a different u per column.
class PlainCot {
public:
    PlainCot(int role, const gf128& delta, const GfVec& chosen, const GfVec& seeds0, const GfVec& seeds1)
        : role_(role), delta_(delta), chosen_(chosen), seeds0_(seeds0), seeds1_(seeds1) {
        prep::check_role(role);
        if (chosen_.size() != KAPPA || seeds0_.size() != KAPPA || seeds1_.size() != KAPPA)
            throw std::invalid_argument("exactly KAPPA seeds per direction are required");
    }

    int role() const { return role_; }
    const gf128& delta() const { return delta_; }
    uint64_t calls() const { return calls_; }

    // Receiver side of direction 1 - role: the U message for the choice bits.
    bytes start(const BitVec& choice_bits) {
        if (pending_) throw std::logic_error("an extension is already in progress");
        prep::check_bits(choice_bits);
        const size_t ell = choice_bits.size();
        if (ell < 1) throw std::invalid_argument("at least one choice bit is required");
        const size_t nw = (ell + 63) / 64;
        const uint64_t dom = prep::prg_dom(USE_PLAIN_COT, 1 - role_, calls_);
        std::vector<uint64_t> u(nw, 0), tmp(nw);
        for (size_t i = 0; i < ell; i++) u[i / 64] |= (uint64_t)(choice_bits[i] & 1) << (i % 64);
        t0_.assign(KAPPA * nw, 0);
        bytes msg(KAPPA * nw * 8);
        for (size_t j = 0; j < KAPPA; j++) {
            uint64_t* t0 = t0_.data() + j * nw;
            prep::detail::prg_words(seeds0_[j], dom, t0, nw);
            prep::detail::prg_words(seeds1_[j], dom, tmp.data(), nw);
            for (size_t w = 0; w < nw; w++) tmp[w] ^= t0[w] ^ u[w];
            prep::store_words_le(msg.data() + 8 * j * nw, tmp.data(), nw);
        }
        ell_ = ell;
        nw_ = nw;
        pending_ = true;
        return msg;
    }

    // Sender side of direction role: consumes the peer's U message; writes
    // the M rows of the own choice bits and the K rows for the peer's.
    void finish(const bytes& u_peer, GfVec& macs, GfVec& keys) {
        if (!pending_) throw std::logic_error("no extension in progress");
        pending_ = false;
        const size_t nw = nw_;
        if (u_peer.size() != KAPPA * nw * 8) throw ProtocolAbort("plain COT: U message has the wrong length");
        std::vector<uint64_t> q(KAPPA * nw), tmp(nw);
        const uint64_t dom = prep::prg_dom(USE_PLAIN_COT, role_, calls_);
        for (size_t j = 0; j < KAPPA; j++) {
            uint64_t* qj = q.data() + j * nw;
            prep::detail::prg_words(chosen_[j], dom, qj, nw);
            if (delta_.bit((unsigned)j)) {
                prep::load_words_le(tmp.data(), u_peer.data() + 8 * j * nw, nw);
                for (size_t w = 0; w < nw; w++) qj[w] ^= tmp[w];
            }
        }
        calls_++;
        macs = rows(t0_, nw, ell_);
        keys = rows(q, nw, ell_);
        std::vector<uint64_t>().swap(t0_);
    }

private:
    // The first ell rows of a column-major bit matrix (128 columns of nw words).
    static GfVec rows(const std::vector<uint64_t>& cols, size_t nw, size_t ell) {
        GfVec out(ell);
        uint64_t lo[64], hi[64];
        for (size_t k = 0; k < nw && 64 * k < ell; k++) {
            for (size_t j = 0; j < 64; j++) {
                lo[j] = cols[j * nw + k];
                hi[j] = cols[(64 + j) * nw + k];
            }
            prep::detail::transpose64(lo);
            prep::detail::transpose64(hi);
            const size_t take = (ell - 64 * k < 64) ? (ell - 64 * k) : 64;
            for (size_t i = 0; i < take; i++) out[64 * k + i] = gf128(lo[i], hi[i]);
        }
        return out;
    }

    int role_;
    gf128 delta_;
    GfVec chosen_, seeds0_, seeds1_;
    uint64_t calls_ = 0;
    bool pending_ = false;
    size_t ell_ = 0, nw_ = 0;
    std::vector<uint64_t> t0_;
};

// ------------------------------------------------------------------ semi-honest generation
// One semi-honest key generation for party `role` (n + 2 rounds):
//   "sh-cot"     one flight per party: the U message of the n selections and
//                the share W_b of the public offset; root Delta_b XOR W;
//   "sh-levels"  CW_b^(i) = XOR_j H0(node) XOR Delta_b XOR alpha_b^(i) Delta_b
//                XOR K_b XOR M_b, exchanged; expansion as in the leaky protocol;
//   "sh-leaf"    CW_b^(n) = XOR_j H0(leaf) XOR beta_b, exchanged;
//                v_b^(j) = H0(leaf) XOR t CW^(n).
// Output: XOR shares t of unit(N, alpha) and v of unit(N, alpha) * beta with
// alpha = alpha_0 XOR alpha_1 and beta = beta_0 XOR beta_1.
inline SemiHonestDpfOutput semihonest_dpf(Channel& ch, const bytes& sid, uint64_t key_id, int role, PlainCot& cot,
                                          unsigned n, uint64_t alpha_share, const gf128& beta_share, RandomSource& rng,
                                          Meter* meter = nullptr, unsigned threads = 1) {
    prep::check_role(role);
    if (ch.party() != role || cot.role() != role)
        throw std::invalid_argument("role does not match the channel endpoint or the COT state");
    check_n(n);
    const BitVec a_bits = index_bits(alpha_share, n);
    const gf128 delta = cot.delta();
    const bytes ssid = semihonest_sub_sid(sid, key_id);
    const uint64_t dom = h0_domain(ssid, label_h0_semihonest());
    const detail::MeterScope scope(meter);
    const prep::WorkerScope workers(threads);
    const size_t big_n = (size_t)1 << n;

    SemiHonestDpfOutput out;
    out.n = n;
    SemiHonestTranscript& tr = out.transcript;

    // Selection material: one COT per level and direction, choice bits alpha_b^(i);
    // the public offset W = W_0 XOR W_1 rides on the same flight.
    detail::mark(meter, BP_SH_COT);
    const gf128 w_own = rng.gf();
    bytes flight = cot.start(a_bits);
    {
        const bytes w_bytes = prep::gf_to_bytes(w_own);
        flight.insert(flight.end(), w_bytes.begin(), w_bytes.end());
    }
    bytes reply = ch.exchange(flight);
    if (reply.size() != cot_message_bytes(n) + ELEM_BYTES) throw ProtocolAbort("semi-honest DPF: first flight has the wrong length");
    tr.W = w_own ^ gf128::from_bytes(reply.data() + reply.size() - ELEM_BYTES);
    reply.resize(reply.size() - ELEM_BYTES);
    GfVec macs, keys;
    cot.finish(reply, macs, keys);

    detail::mark(meter, BP_SH_LEVELS);
    gf128 acc;
    detail::WordBuffer raw = detail::root_pair(delta ^ tr.W, dom, acc);
    size_t m = 1;
    tr.cws.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (i > 0) {
            detail::WordBuffer next(8 * m);
            acc = detail::level_pass(raw.data(), m, tr.cws[i - 1], dom, next.data());
            raw.swap(next);
            m *= 2;
        }
        // share of alpha^(i) Delta: alpha_b^(i) Delta_b XOR K_b[alpha_{1-b}^(i)] XOR M_b[alpha_b^(i)]
        const gf128 cw_own = acc ^ delta ^ prep::gf_and_bit(delta, a_bits[i]) ^ keys[i] ^ macs[i];
        const bytes peer = ch.exchange(prep::gf_to_bytes(cw_own));
        if (peer.size() != ELEM_BYTES) throw ProtocolAbort("semi-honest DPF: correction word has the wrong length");
        tr.cws[i] = cw_own ^ gf128::from_bytes(peer.data());
    }

    detail::mark(meter, BP_SH_LEAF);
    out.t.resize(big_n);
    out.v.resize(big_n);
    gf128 cw_own;
    detail::semihonest_leaf_pass(raw.data(), m, tr.cws[n - 1], dom, out.t.data(), out.v.data(), cw_own);
    raw.release();
    cw_own ^= beta_share;
    const bytes peer = ch.exchange(prep::gf_to_bytes(cw_own));
    if (peer.size() != ELEM_BYTES) throw ProtocolAbort("semi-honest DPF: final correction word has the wrong length");
    tr.cw_final = cw_own ^ gf128::from_bytes(peer.data());
    detail::add_selected(out.v.data(), out.t.data(), big_n, tr.cw_final);
    return out;
}

// One party's session of the semi-honest baseline: base OT once, then any
// number of keys.
//     SemiHonestSession s(role, ch, sid);  s.setup();
//     SemiHonestDpfOutput out = s.dpf_gen(n, alpha_share, beta_share);
// delta() is this party's COT key share (lsb 1 for P_0 and 0 for P_1, so that
// the control bits of on-path nodes differ).  Both parties must issue the same
// sequence of dpf_gen calls.  stats() gives bytes, rounds and seconds of this
// party per phase ("baseot", "sh-cot", "sh-levels", "sh-leaf").
class SemiHonestSession {
public:
    SemiHonestSession(int role, Channel& ch, const bytes& sid, RandomSource* rng = nullptr)
        : role_(role), ch_(&ch), sid_(sid), rng_(rng ? rng : &sys_), meter_(ch) {
        prep::check_role(role);
        if (ch.party() != role) throw std::invalid_argument("role does not match the channel endpoint");
        delta_ = prep::sample_delta(role, *rng_);
    }
    SemiHonestSession(const SemiHonestSession&) = delete;
    SemiHonestSession& operator=(const SemiHonestSession&) = delete;

    int role() const { return role_; }
    const gf128& delta() const { return delta_; }
    uint64_t keys_generated() const { return keys_generated_; }
    bool is_set_up() const { return (bool)cot_; }
    void set_threads(unsigned threads) { threads_ = threads ? threads : 1; }
    const PhaseStats& stats() { return meter_.stats(); }

    // Base OT in both directions (two rounds, 12,288 bytes per party).
    void setup() {
        if (cot_) throw std::invalid_argument("setup() was already run");
        const detail::MeterScope scope(&meter_);
        meter_.mark(BP_SH_BASEOT);
        prep::BaseOtResult ot = prep::base_ot(*ch_, sid_, role_, prep::gf_unpack_bits(delta_), *rng_);
        cot_.reset(new PlainCot(role_, delta_, ot.chosen, ot.seeds0, ot.seeds1));
    }

    // Generate one key pair's full-domain shares; see semihonest_dpf.
    SemiHonestDpfOutput dpf_gen(unsigned n, uint64_t alpha_share, const gf128& beta_share) {
        if (!cot_) throw std::invalid_argument("setup() has not been run");
        check_n(n);
        index_bits(alpha_share, n);  // validates the share before a key number is consumed
        const uint64_t key_id = keys_generated_++;
        return semihonest_dpf(*ch_, sid_, key_id, role_, *cot_, n, alpha_share, beta_share, *rng_, &meter_, threads_);
    }

private:
    int role_;
    Channel* ch_;
    bytes sid_;
    SystemRng sys_;
    RandomSource* rng_;
    Meter meter_;
    gf128 delta_;
    std::unique_ptr<PlainCot> cot_;
    uint64_t keys_generated_ = 0;
    unsigned threads_ = 1;
};

// ------------------------------------------------------------------ TEST-ONLY verification
// NOT part of either protocol and never counted: after a measured or
// cross-language run the two parties send each other, in the clear, their key
// shares and input shares, and compare one digest of their outputs.  This
// destroys all secrecy and exists only to certify benchmark and
// interoperability runs without shipping 2^n shares.
//
//   round 1  each party sends  Delta_b (16 bytes) || le64(alpha_b) || beta_b (16 bytes);
//            alpha, beta, Delta are the XORs of the two messages' fields;
//   round 2  each party sends a SHA-256 digest of its output vectors, where
//            PARTY 0 first XORs the expected value into entry alpha:
//              leaky:        u bits (one byte each) with bit alpha XOR 1,
//                            u MAC shares with entry alpha XOR Delta,
//                            v values with entry alpha XOR beta,
//                            v MAC shares with entry alpha XOR beta * Delta;
//              semi-honest:  t (one byte each) with bit alpha XOR 1,
//                            v with entry alpha XOR beta
//            (vectors concatenated in this order, field elements as 16 bytes
//            little-endian).  The digests are equal iff the outputs
//            reconstruct to unit(N, alpha), unit * beta and their MACs (up to
//            a SHA-256 collision).
namespace testonly {

struct Reveal {
    gf128 delta;     // Delta_0 XOR Delta_1
    uint64_t alpha;  // alpha_0 XOR alpha_1
    gf128 beta;      // beta_0 XOR beta_1
};

inline Reveal exchange_reveal(Channel& ch, const gf128& delta_share, uint64_t alpha_share, const gf128& beta_share) {
    bytes mine(40);
    delta_share.to_bytes(mine.data());
    ascon::store64le(mine.data() + 16, alpha_share);
    beta_share.to_bytes(mine.data() + 24);
    const bytes theirs = ch.exchange(mine);
    if (theirs.size() != 40) throw std::runtime_error("verification step: malformed message");
    Reveal r;
    r.delta = delta_share ^ gf128::from_bytes(theirs.data());
    r.alpha = alpha_share ^ ascon::load64le(theirs.data() + 16);
    r.beta = beta_share ^ gf128::from_bytes(theirs.data() + 24);
    return r;
}

namespace detail {
inline void absorb_gf(prep::detail::FastSha256& h, const gf128* v, size_t n) {
    const size_t step = 4096;
    uint8_t buf[16 * 4096];
    for (size_t done = 0; done < n;) {
        const size_t take = (n - done < step) ? (n - done) : step;
        prep::gf_store(buf, v + done, take);
        h.update(buf, 16 * take);
        done += take;
    }
}
// The vector with `patch` XORed into entry idx (if patched).
inline void absorb_gf_patched(prep::detail::FastSha256& h, const GfVec& v, bool patched, size_t idx, const gf128& patch) {
    if (!patched) {
        absorb_gf(h, v.data(), v.size());
        return;
    }
    if (idx >= v.size()) throw std::runtime_error("verification step: index out of range");
    absorb_gf(h, v.data(), idx);
    const gf128 e = v[idx] ^ patch;
    absorb_gf(h, &e, 1);
    absorb_gf(h, v.data() + idx + 1, v.size() - idx - 1);
}
inline void absorb_bits_patched(prep::detail::FastSha256& h, const BitVec& v, bool patched, size_t idx) {
    if (!patched) {
        h.update(v.data(), v.size());
        return;
    }
    if (idx >= v.size()) throw std::runtime_error("verification step: index out of range");
    h.update(v.data(), idx);
    const uint8_t e = (uint8_t)(v[idx] ^ 1);
    h.update(&e, 1);
    h.update(v.data() + idx + 1, v.size() - idx - 1);
}
inline bool exchange_digests(Channel& ch, const hashes::hash32& mine) {
    const bytes theirs = ch.exchange(bytes(mine.begin(), mine.end()));
    return theirs.size() == mine.size() && std::memcmp(theirs.data(), mine.data(), mine.size()) == 0;
}
}  // namespace detail

inline hashes::hash32 leaky_digest(int role, const LeakyDpfOutput& out, const Reveal& rv) {
    const bool patch = (role == 0);
    prep::detail::FastSha256 h;
    detail::absorb_bits_patched(h, out.u.bits, patch, (size_t)rv.alpha);
    detail::absorb_gf_patched(h, out.u.macs, patch, (size_t)rv.alpha, rv.delta);
    detail::absorb_gf_patched(h, out.v.values, patch, (size_t)rv.alpha, rv.beta);
    detail::absorb_gf_patched(h, out.v.macs, patch, (size_t)rv.alpha, gf128_mul(rv.beta, rv.delta));
    return h.finish();
}
inline hashes::hash32 semihonest_digest(int role, const SemiHonestDpfOutput& out, const Reveal& rv) {
    const bool patch = (role == 0);
    prep::detail::FastSha256 h;
    detail::absorb_bits_patched(h, out.t, patch, (size_t)rv.alpha);
    detail::absorb_gf_patched(h, out.v, patch, (size_t)rv.alpha, rv.beta);
    return h.finish();
}

// True iff the two parties' outputs are correct (and, for the leaky protocol, correctly authenticated).
inline bool verify_leaky(Channel& ch, int role, const LeakyDpfOutput& out, const gf128& delta_share, uint64_t alpha_share,
                         const gf128& beta_share) {
    const Reveal rv = exchange_reveal(ch, delta_share, alpha_share, beta_share);
    return detail::exchange_digests(ch, leaky_digest(role, out, rv));
}
inline bool verify_semihonest(Channel& ch, int role, const SemiHonestDpfOutput& out, const gf128& delta_share,
                              uint64_t alpha_share, const gf128& beta_share) {
    const Reveal rv = exchange_reveal(ch, delta_share, alpha_share, beta_share);
    return detail::exchange_digests(ch, semihonest_digest(role, out, rv));
}

}  // namespace testonly

}  // namespace baselines
}  // namespace couplet
