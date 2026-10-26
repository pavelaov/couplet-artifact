// couplet/cotext.hpp -- correlated OT extension with a global key
// (DESIGN.md Section 7.2): IKNP with the per-column universal-hash
// consistency check.  Same messages and PRG streams as the Python module
// couplet/prep/cotext.py.
//
// Direction d: P_d is the COT sender with global key Delta^B_d and the seeds
// k_j^{Delta_j} from the base OT; P_{1-d} is the COT receiver with the seed
// pairs (k_j^0, k_j^1), j < 128.  One extend(ell) call produces ell
// correlations
//     receiver: (u_i, t_i)        sender: q_i = t_i XOR u_i * Delta^B_d ,
// i.e. random BDOZ bits of the receiver under the sender's key.
//
// Protocol.
//   1. Receiver: random u in F_2^{ell'} with ell' = ell + PAD; columns
//      t_j^c = PRG(k_j^c); sends U_j = t_j^0 XOR t_j^1 XOR u for all j.
//   2. Sender: q_j = PRG(k_j^{Delta_j}) XOR Delta_j U_j.
//   3. The sender, AFTER it has received U, sends a fresh 16-byte seed; both
//      expand chi_0 .. chi_{ell-1} in GF(2^128).  The receiver sends
//          u~   = sum_{i<ell} u_i chi_i XOR pack(u_ell .. u_{ell'-1})
//          V~_j = sum_{i<ell} t^0_{j,i} chi_i XOR pack(t^0_{j,ell} ..)   for all j,
//      and the sender aborts unless, for every column j,
//          sum_{i<ell} q_{j,i} chi_i XOR pack(q_{j,ell} ..) = V~_j XOR Delta_j u~ .
//   4. Rows i < ell are output; the PAD rows are discarded.
//
// Implementation.
//   * One call is run with ell_r = 64 ceil(ell / 64) rows (surplus rows are
//     discarded), so columns are whole 64-bit words; the padding rows are the
//     last two words of every column and pack() of a padding column is that
//     pair of words.
//   * PRG(k) is hashes::ctr_prg under the domain prg_dom(USE_COT_PRG, d, call):
//     the call counter is part of the domain, so repeated calls read disjoint
//     streams.  The chi are the GF(2^128) view of ctr_prg under
//     prg_dom(USE_COT_CHI, d, call) from the fresh seed.
//   * The bit matrix is held column-major (128 columns of 64-bit words) and
//     transposed 64 rows at a time: two 64 x 64 transpositions (masked swaps,
//     four rows per AVX2 register) give the low and high halves of 64 rows.
//   * The 128 column sums sum_i m_{j,i} chi_i are accumulated with sixteen
//     256-entry tables indexed by the bytes of a row (one table update per
//     row byte) and resolved per bit at the end.
//   * Message sizes: 16 bytes per row and 16 * 128 bytes of padding for U,
//     16 bytes for the seed, 16 * 129 bytes for the check message.
//
// A corrupted receiver can use an inconsistent u in column j; the check then
// fails exactly if Delta_j = 1 (the sender never reads U_j otherwise).  This
// is the standard selective-failure leakage of F_COT on the MAC key.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "channel.hpp"
#include "gf128.hpp"
#include "hashes.hpp"
#include "params.hpp"
#include "rng.hpp"
#include "shares.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace couplet {
namespace prep {

const size_t COT_SEED_BYTES = 16;
const size_t COT_CHECK_BYTES = 16 * (KAPPA + 1);
const size_t COT_MAX_ROWS = (size_t)1 << 26;  // rows of one call: U stays below the 2^31 - 1 byte frame limit

// Rows actually produced by one call: ell rounded up to a multiple of 64.
inline size_t rounded_rows(size_t ell) { return 64 * ((ell + 63) / 64); }

// Size of the U message of one extend(ell) call.
inline size_t u_message_bytes(size_t ell) { return KAPPA * (rounded_rows(ell) + PAD) / 8; }

namespace detail {

inline void check_cot_ell(size_t ell) {
    if (ell < 1 || ell > COT_MAX_ROWS) throw std::invalid_argument("ell must be in 1..2^26");
}
inline void check_cot_seeds(const GfVec& seeds) {
    if (seeds.size() != KAPPA) throw std::invalid_argument("exactly KAPPA seeds are required");
}

// In-place transposition of a 64 x 64 bit matrix: afterwards bit j of a[i]
// is the former bit i of a[j].  Six rounds of masked swaps, one per bit of
// the index (round s exchanges bit s of the row index with bit s of the
// column index).
inline void transpose64(uint64_t a[64]) {
    static const uint64_t MASK[6] = {0x5555555555555555ull, 0x3333333333333333ull, 0x0f0f0f0f0f0f0f0full,
                                     0x00ff00ff00ff00ffull, 0x0000ffff0000ffffull, 0x00000000ffffffffull};
#if defined(__AVX2__)
    for (int lg = 5; lg >= 2; lg--) {
        const int s = 1 << lg;
        const __m256i m = _mm256_set1_epi64x((long long)MASK[lg]);
        const __m128i cnt = _mm_cvtsi32_si128(s);
        for (int base = 0; base < 64; base += 2 * s) {
            for (int i = 0; i < s; i += 4) {
                __m256i x = _mm256_loadu_si256((const __m256i*)(a + base + i));
                __m256i y = _mm256_loadu_si256((const __m256i*)(a + base + s + i));
                const __m256i t = _mm256_and_si256(_mm256_xor_si256(_mm256_srl_epi64(x, cnt), y), m);
                x = _mm256_xor_si256(x, _mm256_sll_epi64(t, cnt));
                y = _mm256_xor_si256(y, t);
                _mm256_storeu_si256((__m256i*)(a + base + i), x);
                _mm256_storeu_si256((__m256i*)(a + base + s + i), y);
            }
        }
    }
    const __m256i m2 = _mm256_set1_epi64x((long long)MASK[1]);
    const __m256i m1 = _mm256_set1_epi64x((long long)MASK[0]);
    for (int base = 0; base < 64; base += 8) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i*)(a + base));      // r0 r1 r2 r3
        const __m256i v1 = _mm256_loadu_si256((const __m256i*)(a + base + 4));  // r4 r5 r6 r7
        // s = 2: pairs (r0, r2), (r1, r3), (r4, r6), (r5, r7)
        __m256i x = _mm256_permute2x128_si256(v0, v1, 0x20);  // r0 r1 r4 r5
        __m256i y = _mm256_permute2x128_si256(v0, v1, 0x31);  // r2 r3 r6 r7
        __m256i t = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi64(x, 2), y), m2);
        x = _mm256_xor_si256(x, _mm256_slli_epi64(t, 2));
        y = _mm256_xor_si256(y, t);
        // s = 1: pairs (r0, r1), (r2, r3), (r4, r5), (r6, r7)
        __m256i p = _mm256_unpacklo_epi64(x, y);  // r0 r2 r4 r6
        __m256i q = _mm256_unpackhi_epi64(x, y);  // r1 r3 r5 r7
        t = _mm256_and_si256(_mm256_xor_si256(_mm256_srli_epi64(p, 1), q), m1);
        p = _mm256_xor_si256(p, _mm256_slli_epi64(t, 1));
        q = _mm256_xor_si256(q, t);
        const __m256i lo = _mm256_unpacklo_epi64(p, q);  // r0 r1 r4 r5
        const __m256i hi = _mm256_unpackhi_epi64(p, q);  // r2 r3 r6 r7
        _mm256_storeu_si256((__m256i*)(a + base), _mm256_permute2x128_si256(lo, hi, 0x20));
        _mm256_storeu_si256((__m256i*)(a + base + 4), _mm256_permute2x128_si256(lo, hi, 0x31));
    }
#else
    for (int lg = 5; lg >= 0; lg--) {
        const int s = 1 << lg;
        const uint64_t m = MASK[lg];
        for (int base = 0; base < 64; base += 2 * s) {
            for (int i = 0; i < s; i++) {
                const uint64_t t = ((a[base + i] >> s) ^ a[base + s + i]) & m;
                a[base + i] ^= t << s;
                a[base + s + i] ^= t;
            }
        }
    }
#endif
}

// Reference transposition (tests): out[i] bit j = in[j] bit i.
inline void transpose64_ref(const uint64_t in[64], uint64_t out[64]) {
    for (int i = 0; i < 64; i++) {
        uint64_t r = 0;
        for (int j = 0; j < 64; j++) r |= ((in[j] >> i) & 1) << j;
        out[i] = r;
    }
}

// Accumulator of the 128 partial sums  P[j] = XOR of chi_i over the rows i
// whose bit j is set.  add() updates sixteen byte-indexed tables; finish()
// resolves them per bit.
class ColumnSums {
public:
    ColumnSums() : tab_(16 * 256) {}

    void reset() { std::fill(tab_.begin(), tab_.end(), gf128()); }

    // rows as (lo, hi) word pairs; chi as consecutive (lo, hi) word pairs.
    void add(const uint64_t* row_lo, const uint64_t* row_hi, const uint64_t* chi, size_t n) {
        gf128* tab = tab_.data();
#if COUPLET_HAVE_PCLMUL  // SSE2 is available whenever PCLMUL is
        for (size_t i = 0; i < n; i++) {
            const __m128i c = _mm_loadu_si128((const __m128i*)(chi + 2 * i));
            const uint64_t l = row_lo[i], h = row_hi[i];
            for (unsigned p = 0; p < 8; p++) {
                __m128i* e = (__m128i*)(tab + ((size_t)p << 8) + ((l >> (8 * p)) & 0xff));
                _mm_storeu_si128(e, _mm_xor_si128(_mm_loadu_si128(e), c));
                __m128i* f = (__m128i*)(tab + ((size_t)(8 + p) << 8) + ((h >> (8 * p)) & 0xff));
                _mm_storeu_si128(f, _mm_xor_si128(_mm_loadu_si128(f), c));
            }
        }
#else
        for (size_t i = 0; i < n; i++) {
            const gf128 c(chi[2 * i], chi[2 * i + 1]);
            const uint64_t l = row_lo[i], h = row_hi[i];
            for (unsigned p = 0; p < 8; p++) {
                tab[((size_t)p << 8) + ((l >> (8 * p)) & 0xff)] ^= c;
                tab[((size_t)(8 + p) << 8) + ((h >> (8 * p)) & 0xff)] ^= c;
            }
        }
#endif
    }

    void finish(gf128 out[128]) const {
        for (unsigned p = 0; p < 16; p++) {
            const gf128* t = tab_.data() + ((size_t)p << 8);
            for (unsigned b = 0; b < 8; b++) {
                gf128 acc;
                for (unsigned v = 0; v < 256; v++)
                    if ((v >> b) & 1) acc ^= t[v];
                out[8 * p + b] = acc;
            }
        }
    }

private:
    GfVec tab_;
};

// Reference for ColumnSums (tests): P[j] = XOR of chi_i over rows with bit j set.
inline void column_sums_ref(const gf128* rows, const gf128* chi, size_t n, gf128 out[128]) {
    for (unsigned j = 0; j < 128; j++) {
        gf128 acc;
        for (size_t i = 0; i < n; i++)
            if (rows[i].bit(j)) acc ^= chi[i];
        out[j] = acc;
    }
}

// One pass over a column-major bit matrix (128 columns of nw words; column j
// at cols + j * nw).  For the first ell_r = 64 * blocks rows it
//   * transposes them to rows,
//   * accumulates the 128 column sums against chi_i = element i of the
//     GF(2^128) view of ctr_prg(seed, chi_dom) and writes them to sums_out,
//   * if u_words is given, computes u~ = sum_i u_i chi_i,
//   * copies the first n_out rows to out_rows.
// The block range is split over the worker threads of the caller (each with
// its own tables; the partial sums are XORed at the end).
class RowPass {
public:
    enum : size_t { GROUP = 3 };  // 3 blocks = 192 rows = 384 chi words = 128 PRG blocks

    void run(const uint64_t* cols, size_t nw, size_t blocks, const gf128& seed, uint64_t chi_dom,
             const uint64_t* u_words, gf128* out_rows, size_t n_out, gf128 sums_out[128], gf128* u_tilde) {
        const unsigned workers = worker_count();
        if (sums_.size() < workers) sums_.resize(workers);
        std::vector<gf128> ut(workers);
        for (unsigned p = 0; p < workers; p++) sums_[p].reset();
        parallel_chunks(blocks, GROUP, 96 * GROUP, [&](size_t b0, size_t b1, unsigned part) {
            ColumnSums& sums = sums_[part];
            uint64_t chi[2 * 64 * GROUP];
            uint64_t lo[64], hi[64];
            uint64_t ut_lo = 0, ut_hi = 0;
            for (size_t k0 = b0; k0 < b1; k0 += GROUP) {
                const size_t kn = (b1 - k0 < GROUP) ? (b1 - k0) : GROUP;
                prg_words(seed, chi_dom, chi, 2 * 64 * kn, (uint64_t)(128 * (k0 / GROUP)));
                for (size_t kk = 0; kk < kn; kk++) {
                    const size_t k = k0 + kk;
#if COUPLET_PREP_X86
                    if ((k & 7) == 0 && k + 8 < nw) {
                        for (size_t j = 0; j < 128; j++) _mm_prefetch((const char*)(cols + j * nw + k + 8), _MM_HINT_T0);
                    }
#endif
                    for (size_t j = 0; j < 64; j++) {
                        lo[j] = cols[j * nw + k];
                        hi[j] = cols[(64 + j) * nw + k];
                    }
                    transpose64(lo);
                    transpose64(hi);
                    const uint64_t* c = chi + 2 * 64 * kk;
                    sums.add(lo, hi, c, 64);
                    if (u_words) {
                        const uint64_t uw = u_words[k];
                        for (unsigned i = 0; i < 64; i++) {
                            const uint64_t m = 0 - ((uw >> i) & 1);
                            ut_lo ^= c[2 * i] & m;
                            ut_hi ^= c[2 * i + 1] & m;
                        }
                    }
                    const size_t row0 = 64 * k;
                    if (row0 < n_out) {
                        const size_t take = (n_out - row0 < 64) ? (n_out - row0) : 64;
                        for (size_t i = 0; i < take; i++) out_rows[row0 + i] = gf128(lo[i], hi[i]);
                    }
                }
            }
            ut[part] ^= gf128(ut_lo, ut_hi);
        });
        gf128 part_sums[128];
        gf128 u_acc;
        for (size_t j = 0; j < 128; j++) sums_out[j] = gf128();
        for (unsigned p = 0; p < workers; p++) {
            sums_[p].finish(part_sums);
            for (size_t j = 0; j < 128; j++) sums_out[j] ^= part_sums[j];
            u_acc ^= ut[p];
        }
        if (u_tilde) *u_tilde = u_acc;
    }

private:
    std::vector<ColumnSums> sums_;
};

// The column expansion of one extension call: fn(j, tmp) is called for every
// column j < 128, on the worker threads of the caller when the matrix is
// large; tmp is a scratch column of nw words owned by the calling part.
template <class F>
inline void for_each_column(size_t nw, F&& fn) {
    const size_t min_cols = (nw >= 2048) ? 4 : KAPPA + 1;  // small matrices stay in the calling thread
    parallel_chunks(KAPPA, 1, min_cols, [&](size_t j0, size_t j1, unsigned) {
        std::vector<uint64_t> tmp(nw);
        for (size_t j = j0; j < j1; j++) fn(j, tmp.data());
    });
}

}  // namespace detail

// COT receiver of direction `direction` (held by party 1 - direction).
class CotReceiver {
public:
    // seeds0, seeds1: the base-OT pairs (k_j^0, k_j^1) of this direction.
    CotReceiver(int direction, const GfVec& seeds0, const GfVec& seeds1, RandomSource& rng)
        : direction_(direction), seeds0_(seeds0), seeds1_(seeds1), rng_(&rng), calls_(0), rows_(0), active_(false),
          ell_(0), ell_r_(0), nw_(0) {
        if (direction != 0 && direction != 1) throw std::invalid_argument("direction must be 0 or 1");
        detail::check_cot_seeds(seeds0_);
        detail::check_cot_seeds(seeds1_);
    }

    int direction() const { return direction_; }
    uint64_t calls() const { return calls_; }  // extension calls so far
    uint64_t rows() const { return rows_; }    // correlations output so far

    // Step 1: sample u, expand the columns, return the U message (a buffer
    // of this object, valid until the next start()).  The test hook
    // cheat->flip_u flips bits (column, row) of U, i.e. uses an inconsistent
    // u in those columns.
    const bytes& start(size_t ell, const Cheat* cheat = nullptr) {
        detail::check_cot_ell(ell);
        if (active_) throw std::logic_error("an extension is already in progress");
        if (calls_ > MAX_PRG_COUNTER) throw std::runtime_error("extension call counter exhausted");
        ell_ = ell;
        ell_r_ = rounded_rows(ell);
        nw_ = (ell_r_ + PAD) / 64;
        const uint64_t dom = prg_dom(USE_COT_PRG, direction_, calls_);
        t0_.resize(KAPPA * nw_);
        u_.resize(nw_);
        {
            bytes raw(8 * nw_);
            rng_->fill(raw.data(), raw.size());
            load_words_le(u_.data(), raw.data(), nw_);
        }
        u_msg_.resize(KAPPA * nw_ * 8);  // reused across calls: no fresh pages per chunk
        bytes& msg = u_msg_;
        const size_t nw = nw_;
        const uint64_t* u = u_.data();
        detail::for_each_column(nw, [&](size_t j, uint64_t* tmp) {
            uint64_t* t0 = t0_.data() + j * nw;
            detail::prg_words(seeds0_[j], dom, t0, nw);
            detail::prg_words(seeds1_[j], dom, tmp, nw);
            for (size_t w = 0; w < nw; w++) tmp[w] ^= t0[w] ^ u[w];
            store_words_le(msg.data() + 8 * j * nw, tmp, nw);
        });
        if (cheat) {
            for (const std::pair<unsigned, size_t>& f : cheat->flip_u) {
                if (f.first >= KAPPA || f.second >= 64 * nw_) throw std::invalid_argument("flip_u out of range");
                msg[8 * ((size_t)f.first * nw_ + f.second / 64) + (f.second % 64) / 8] ^= (uint8_t)(1u << (f.second % 8));
            }
        }
        active_ = true;
        return msg;
    }

    // Step 3: the check message u~ || V~_0 .. V~_127; writes the ell output
    // rows (u_i to bits_out, t_i to macs_out).
    bytes answer(const bytes& seed, uint8_t* bits_out, gf128* macs_out) {
        if (!active_) throw std::logic_error("no extension in progress");
        active_ = false;
        if (seed.size() != COT_SEED_BYTES) throw ProtocolAbort("COT: challenge seed has the wrong length");
        gf128 v[KAPPA];
        gf128 u_tilde;
        pass_.run(t0_.data(), nw_, ell_r_ / 64, gf128::from_bytes(seed.data()), prg_dom(USE_COT_CHI, direction_, calls_),
                  u_.data(), macs_out, ell_, v, &u_tilde);
        u_tilde ^= gf128(u_[nw_ - 2], u_[nw_ - 1]);
        for (size_t j = 0; j < KAPPA; j++) v[j] ^= gf128(t0_[j * nw_ + nw_ - 2], t0_[j * nw_ + nw_ - 1]);
        for (size_t i = 0; i < ell_; i++) bits_out[i] = (uint8_t)((u_[i / 64] >> (i % 64)) & 1);
        bytes msg(COT_CHECK_BYTES);
        u_tilde.to_bytes(msg.data());
        gf_store(msg.data() + 16, v, KAPPA);
        calls_++;
        rows_ += ell_;
        return msg;
    }

    // One direction on the channel (three flights): send U, receive the seed,
    // send the check.
    void extend(Channel& ch, size_t ell, uint8_t* bits_out, gf128* macs_out, const Cheat* cheat = nullptr) {
        ch.send(start(ell, cheat));
        ch.flight();
        const bytes seed = ch.recv(COT_SEED_BYTES, "COT: challenge seed has the wrong length");
        ch.flight();
        ch.send(answer(seed, bits_out, macs_out));
        ch.flight();
    }
    Bdoz extend(Channel& ch, size_t ell, const Cheat* cheat = nullptr) {
        detail::check_cot_ell(ell);
        Bdoz out(ell);
        extend(ch, ell, out.bits.data(), out.macs.data(), cheat);
        return out;
    }

    // Forget the seeds and all state (session abort).
    void wipe() {
        std::fill(seeds0_.begin(), seeds0_.end(), gf128());
        std::fill(seeds1_.begin(), seeds1_.end(), gf128());
        std::vector<uint64_t>().swap(t0_);
        std::vector<uint64_t>().swap(u_);
        bytes().swap(u_msg_);
        active_ = false;
    }

private:
    int direction_;
    GfVec seeds0_, seeds1_;
    RandomSource* rng_;
    uint64_t calls_, rows_;
    bool active_;
    size_t ell_, ell_r_, nw_;
    std::vector<uint64_t> t0_, u_;
    bytes u_msg_;
    detail::RowPass pass_;
};

// COT sender of direction `direction` (held by party `direction`) with key delta.
class CotSender {
public:
    // seeds: the base-OT seeds k_j^{Delta_j} of this direction.
    CotSender(int direction, const gf128& delta, const GfVec& seeds, RandomSource& rng)
        : direction_(direction), delta_(delta), seeds_(seeds), rng_(&rng), calls_(0), rows_(0), active_(false),
          ell_(0), ell_r_(0), nw_(0) {
        if (direction != 0 && direction != 1) throw std::invalid_argument("direction must be 0 or 1");
        detail::check_cot_seeds(seeds_);
    }

    int direction() const { return direction_; }
    const gf128& delta() const { return delta_; }
    uint64_t calls() const { return calls_; }
    uint64_t rows() const { return rows_; }

    // Step 2: read U, form the q columns; returns the fresh challenge seed
    // (sampled only now, after U is fixed).
    bytes absorb(size_t ell, const bytes& u_msg) {
        detail::check_cot_ell(ell);
        if (active_) throw std::logic_error("an extension is already in progress");
        if (calls_ > MAX_PRG_COUNTER) throw std::runtime_error("extension call counter exhausted");
        const size_t ell_r = rounded_rows(ell);
        const size_t nw = (ell_r + PAD) / 64;
        if (u_msg.size() != KAPPA * nw * 8) throw ProtocolAbort("COT: U message has the wrong length");
        ell_ = ell;
        ell_r_ = ell_r;
        nw_ = nw;
        const uint64_t dom = prg_dom(USE_COT_PRG, direction_, calls_);
        q_.resize(KAPPA * nw_);
        detail::for_each_column(nw, [&](size_t j, uint64_t* tmp) {
            uint64_t* q = q_.data() + j * nw;
            detail::prg_words(seeds_[j], dom, q, nw);
            if (delta_.bit((unsigned)j)) {  // q_j = PRG(k_j^{Delta_j}) XOR Delta_j U_j
                load_words_le(tmp, u_msg.data() + 8 * j * nw, nw);
                for (size_t w = 0; w < nw; w++) q[w] ^= tmp[w];
            }
        });
        seed_ = rng_->get(COT_SEED_BYTES);
        active_ = true;
        return seed_;
    }

    // Step 3: verify the receiver's check message and write the ell keys
    // K_i = q_i to keys_out; ProtocolAbort if the check fails (keys_out is
    // then cleared).
    void verify(const bytes& check_msg, gf128* keys_out) {
        if (!active_) throw std::logic_error("no extension in progress");
        active_ = false;
        if (check_msg.size() != COT_CHECK_BYTES) throw ProtocolAbort("COT: check message has the wrong length");
        const gf128 u_tilde = gf128::from_bytes(check_msg.data());
        gf128 v[KAPPA], lhs[KAPPA];
        gf_load(v, check_msg.data() + 16, KAPPA);
        pass_.run(q_.data(), nw_, ell_r_ / 64, gf128::from_bytes(seed_.data()), prg_dom(USE_COT_CHI, direction_, calls_),
                  nullptr, keys_out, ell_, lhs, nullptr);
        uint64_t diff = 0;
        for (size_t j = 0; j < KAPPA; j++) {
            const gf128 l = lhs[j] ^ gf128(q_[j * nw_ + nw_ - 2], q_[j * nw_ + nw_ - 1]);
            const gf128 r = v[j] ^ gf_and_bit(u_tilde, delta_.bit((unsigned)j));
            diff |= (l.lo ^ r.lo) | (l.hi ^ r.hi);
        }
        calls_++;
        if (diff != 0) {
            std::fill(keys_out, keys_out + ell_, gf128());
            throw ProtocolAbort("COT consistency check failed");
        }
        rows_ += ell_;
    }

    // One direction on the channel (three flights): receive U, send the seed,
    // receive the check.
    void extend(Channel& ch, size_t ell, gf128* keys_out) {
        detail::check_cot_ell(ell);  // before the receive: the length of U follows from ell
        const bytes u_msg = ch.recv(u_message_bytes(ell), "COT: U message has the wrong length");
        ch.flight();
        ch.send(absorb(ell, u_msg));
        ch.flight();
        const bytes check = ch.recv(COT_CHECK_BYTES, "COT: check message has the wrong length");
        ch.flight();
        verify(check, keys_out);
    }
    Keys extend(Channel& ch, size_t ell) {
        detail::check_cot_ell(ell);
        Keys out(ell);
        extend(ch, ell, out.keys.data());
        return out;
    }

    // Forget the key, the seeds and all state (session abort).
    void wipe() {
        delta_ = gf128();
        std::fill(seeds_.begin(), seeds_.end(), gf128());
        std::vector<uint64_t>().swap(q_);
        seed_.clear();
        active_ = false;
    }

private:
    int direction_;
    gf128 delta_;
    GfVec seeds_;
    RandomSource* rng_;
    uint64_t calls_, rows_;
    bool active_;
    size_t ell_, ell_r_, nw_;
    std::vector<uint64_t> q_;
    bytes seed_;
    detail::RowPass pass_;
};

// Both directions in parallel (three exchanges).  `sender` and `receiver` are
// this party's two roles (it sends in its own direction and receives in the
// other).  Writes ell own BDOZ bits (own_bits, own_macs) and ell keys for the
// peer's BDOZ bits.
inline void extend_both(Channel& ch, CotSender& sender, CotReceiver& receiver, size_t ell, uint8_t* own_bits,
                        gf128* own_macs, gf128* keys, const Cheat* cheat = nullptr) {
    // Each party receives what it sends: U, a challenge seed, a check message.
    const bytes u_peer = ch.exchange(receiver.start(ell, cheat), u_message_bytes(ell), "COT: U message has the wrong length");
    const bytes seed_peer =
        ch.exchange(sender.absorb(ell, u_peer), COT_SEED_BYTES, "COT: challenge seed has the wrong length");
    const bytes check_peer = ch.exchange(receiver.answer(seed_peer, own_bits, own_macs), COT_CHECK_BYTES,
                                         "COT: check message has the wrong length");
    sender.verify(check_peer, keys);
}
inline BdozPair extend_both(Channel& ch, CotSender& sender, CotReceiver& receiver, size_t ell,
                            const Cheat* cheat = nullptr) {
    detail::check_cot_ell(ell);
    BdozPair out;
    out.own = Bdoz(ell);
    out.keys = Keys(ell);
    extend_both(ch, sender, receiver, ell, out.own.bits.data(), out.own.macs.data(), out.keys.keys.data(), cheat);
    return out;
}

}  // namespace prep
}  // namespace couplet
