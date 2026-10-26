// couplet/prep.hpp -- the Boolean preprocessing behind one object
// (DESIGN.md Sections 7.1 - 7.5, 7.7); the C++ counterpart of
// couplet/prep/real.py and couplet/prep/ideal.py.
//
//   Prep        the interface both implementations expose:
//                 setup()                 prepare the session;
//                 random_bits(n)          n random <x>                 -> Shares
//                 input_masks(owner, n)   n input masks of P_owner     -> InputMasks
//                 and_triples(n)          n authenticated AND triples  -> Triples
//                 random_bdoz(n)          n random BDOZ bits           -> BdozPair
//                 delta()                 this party's share Delta^B_role
//                                         (lsb 1 for P_0, 0 for P_1).
//
//   RealPrep    the real (dealer-free) pipeline of one party over a channel
//               endpoint: base OT in both directions, correlated OT
//               extension, authenticated bits, leaky AND triples, bucketing
//               and the consistency check (run size and bucket size from
//               triple_request(n, sigma)).  Both parties must call the same
//               methods with the same arguments in the same order.
//
//               Statistical error: every and_triples call is one bucketing
//               population and contributes 2^-sigma (Lemma 4); request all
//               triples of a session in one call if the session is to stay
//               at 2^-sigma overall.
//
//               Minimum size of a request: the triple check of a request
//               must cover at least MIN_LEAKY_TRIPLES = 306 leaky triples
//               (the bound against a key share that ignores the lsb
//               convention, Lemma 3).  and_triples(n) therefore runs the
//               protocol for triple_request(n, sigma): a request whose
//               n * bucket_size(n) would be smaller is padded to more
//               triples, and the surplus triples are discarded after the
//               check (last_run_triples() is the size of the run;
//               counters().leaky_triples counts the leaky triples that were
//               generated).  Both parties pad in the same way, so the
//               transcript is that of a request for the padded number.
//
//               Accounting: stats() gives bytes, messages, rounds and
//               seconds of this party per phase (baseot, cotext, abit, laand,
//               bucket, check).  Only traffic caused by RealPrep methods is
//               booked.
//
//               Abort (change G of DESIGN.md): if a check fails or the peer
//               misbehaves, the method throws (ProtocolAbort for a failed
//               check or a malformed message), the object is marked aborted,
//               the key share and all extension state are wiped, and every
//               later call throws ProtocolAbort.  Invalid arguments throw
//               std::invalid_argument before any message is sent and do not
//               abort the session.
//
//               Threads: set_threads(t) lets the bulk kernels of this party
//               (column expansion, row pass, leaky-AND hashing, bucketing)
//               use t worker threads; the default is 1.  Messages and
//               results do not depend on it.
//
//               Randomness: `rng` (default: the operating system's
//               generator) provides the key share, base-OT scalars,
//               extension choice bits and check seeds; commitment randomness
//               and coin-toss shares always come from the operating system.
//
//   IdealDealer / IdealPrep
//               IDEAL preprocessing (Section 7.7): NOT a protocol.  One
//               IdealDealer object is shared by both parties of a test or of
//               an online-only measurement; it samples the MAC key shares
//               and all correlations itself and hands each party its part.
//               The dealer pairs request number k of P_0 with request number
//               k of P_1 and throws std::invalid_argument if kind or size
//               differ.  It is thread-safe and uses no channel.  Results
//               obtained with it must be labelled "ideal preprocessing" and
//               never be reported as all-in numbers.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "aand.hpp"
#include "abits.hpp"
#include "baseot.hpp"
#include "channel.hpp"
#include "cotext.hpp"
#include "gf128.hpp"
#include "hashes.hpp"
#include "params.hpp"
#include "rng.hpp"
#include "shares.hpp"

namespace couplet {
namespace prep {

struct PrepCounters {
    uint64_t random_bits = 0;
    uint64_t input_masks = 0;
    uint64_t and_triples = 0;
    uint64_t leaky_triples = 0;
};

// Interface of the real and of the ideal preprocessing (one party).
class Prep : public BdozSource {
public:
    virtual ~Prep() {}
    virtual int role() const = 0;
    virtual bool ideal() const = 0;
    virtual bool aborted() const = 0;
    virtual gf128 delta() const = 0;  // zero after an abort
    virtual void setup() = 0;
    virtual void abort() = 0;
    virtual Shares random_bits(size_t n) = 0;
    virtual InputMasks input_masks(int owner, size_t n) = 0;
    virtual Triples and_triples(size_t n) = 0;
    virtual const PrepCounters& counters() const = 0;
};

// ------------------------------------------------------------------ real preprocessing
class RealPrep : public Prep {
public:
    // The identifier of the preprocessing sub-session:
    // SHA-256(frame("Couplet-prep-sid", sid)).  It carries the labels of the
    // coin tosses and BoolChecks of the preprocessing and the leaky-AND hash
    // domain, so they cannot collide with labels of the engine under sid.
    static bytes prep_sid(const bytes& sid) {
        const hashes::hash32 h = hashes::framed_sha256({"Couplet-prep-sid", sid});
        return bytes(h.begin(), h.end());
    }

    RealPrep(int role, Channel& ch, const bytes& sid, int sigma = SIGMA_DEFAULT, RandomSource* rng = nullptr,
             size_t batch_rows = DEFAULT_BATCH_ROWS, const Cheat* cheat = nullptr)
        : role_(role), ch_(&ch), sid_(sid), sigma_(sigma), batch_rows_(batch_rows), cheat_(cheat),
          rng_(rng ? rng : &sys_), meter_(ch) {
        check_role(role);
        if (ch.party() != role) throw std::invalid_argument("role does not match the channel endpoint");
        if (sigma < 1) throw std::invalid_argument("sigma must be positive");
        if (batch_rows < 1 || batch_rows > COT_MAX_ROWS) throw std::invalid_argument("batch_rows out of range");
        delta_ = sample_delta(role, *rng_);
        sid_prep_ = prep_sid(sid_);
    }
    RealPrep(const RealPrep&) = delete;
    RealPrep& operator=(const RealPrep&) = delete;

    int role() const override { return role_; }
    bool ideal() const override { return false; }
    bool aborted() const override { return aborted_; }
    gf128 delta() const override { return delta_; }
    const PrepCounters& counters() const override { return counters_; }
    int sigma() const { return sigma_; }
    unsigned last_bucket_size() const { return last_bucket_; }  // 0 before the first and_triples
    // Triples of the latest protocol run: the request, or more if it was padded (0 before the first).
    size_t last_run_triples() const { return last_run_; }
    bool is_set_up() const { return (bool)abits_; }

    // Worker threads used by the bulk kernels of this party (1 = none besides the caller).
    void set_threads(unsigned threads) { threads_ = threads ? threads : 1; }
    unsigned threads() const { return threads_; }

    // Per-phase accounting of this party (see the header comment).
    const PrepStats& stats() { return meter_.stats(); }

    // COT rows per direction consumed by and_triples(n) (excluding extension padding).
    size_t cot_rows_for_triples(size_t n) const { return 2 * (size_t)triple_request(n, sigma_).leaky() + PAD; }

    // Retire the session: wipe the key share and all extension state.
    void abort() override {
        aborted_ = true;
        delta_ = gf128();
        abits_.reset();
        if (sender_) sender_->wipe();
        if (receiver_) receiver_->wipe();
        sender_.reset();
        receiver_.reset();
    }

    // Base OT (both directions) and initialisation of the two COT extensions.  Two rounds.
    void setup() override {
        enter();
        if (abits_) throw std::invalid_argument("setup() was already run");
        guarded([&] {
            meter_.mark(PH_BASEOT);
            BaseOtResult ot = base_ot(*ch_, sid_, role_, gf_unpack_bits(delta_), *rng_, cheat_);
            meter_.mark(PH_COTEXT);
            sender_.reset(new CotSender(role_, delta_, ot.chosen, *rng_));
            receiver_.reset(new CotReceiver(1 - role_, ot.seeds0, ot.seeds1, *rng_));
            abits_.reset(new AbitSource(*ch_, role_, delta_, *sender_, *receiver_, batch_rows_, &meter_));
            return 0;
        });
    }

    // n random authenticated bits <x> (three rounds per extension chunk).
    Shares random_bits(size_t n) override {
        AbitSource& src = source();
        if (n == 0) return Shares();
        return guarded([&] {
            src.set_cheat(cheat_);
            Shares out = src.random_shares(n);
            src.set_cheat(nullptr);
            counters_.random_bits += n;
            return out;
        });
    }

    // n random bits in BDOZ form (the BdozSource interface).
    BdozPair random_bdoz(size_t n) override {
        AbitSource& src = source();
        if (n == 0) return BdozPair();
        return guarded([&] {
            src.set_cheat(cheat_);
            BdozPair out = src.random_bdoz(n);
            src.set_cheat(nullptr);
            return out;
        });
    }

    // n input masks of P_owner: shares and bits at the owner, shares only at the peer.
    InputMasks input_masks(int owner, size_t n) override {
        enter();
        if (owner != 0 && owner != 1) throw std::invalid_argument("owner must be 0 or 1");
        AbitSource& src = source();
        if (n == 0) {
            InputMasks empty;
            empty.has_bits = (owner == role_);
            return empty;
        }
        return guarded([&] {
            InputMasks out = src.input_masks(owner, n);
            counters_.input_masks += n;
            return out;
        });
    }

    // n AND triples (a, b, c): bucketed (Lemma 4) and checked before they are
    // returned.  The run has at least MIN_LEAKY_TRIPLES leaky triples: a
    // smaller request is padded and the surplus triples are discarded (see
    // the header comment).
    Triples and_triples(size_t n) override {
        AbitSource& src = source();
        if (n == 0) return Triples();
        const TripleRequest run = triple_request(n, sigma_);
        const size_t n_run = (size_t)run.ell;
        const unsigned bucket = run.bucket;
        if ((uint64_t)n_run > MAX_LEAKY_PER_BATCH / bucket || 2 * (uint64_t)n_run * bucket + PAD > (uint64_t)SIZE_MAX / 32)
            throw std::invalid_argument("too many triples for one batch");
        return guarded([&] {
            const uint64_t batch = batch_, index0 = triple_index_;
            batch_ += 1;
            triple_index_ += (uint64_t)n_run * bucket;
            Triples out = prep::and_triples(*ch_, sid_prep_, role_, delta_, src, n_run, bucket, batch, index0, sys_,
                                            &meter_, cheat_);
            last_bucket_ = bucket;
            last_run_ = n_run;
            counters_.and_triples += n;
            counters_.leaky_triples += (uint64_t)n_run * bucket;
            if (n_run != n) {  // padded request: the surplus triples are never used
                out.a = out.a.slice(0, n);
                out.b = out.b.slice(0, n);
                out.c = out.c.slice(0, n);
            }
            return out;
        });
    }

private:
    void enter() const {
        if (aborted_) throw ProtocolAbort("session aborted");
    }
    AbitSource& source() {
        enter();
        if (!abits_) throw std::invalid_argument("setup() has not been run");
        return *abits_;
    }
    // Run a protocol step: abort the session on any failure; stop the accounting afterwards.
    template <class F>
    auto guarded(F&& body) -> decltype(body()) {
        struct Stop {
            PhaseMeter& m;
            ~Stop() { m.mark(PH_NONE); }
        } stop{meter_};
        const WorkerScope workers(threads_);
        try {
            return body();
        } catch (...) {
            abort();
            throw;
        }
    }

    int role_;
    Channel* ch_;
    bytes sid_, sid_prep_;
    int sigma_;
    size_t batch_rows_;
    const Cheat* cheat_;
    SystemRng sys_;
    RandomSource* rng_;
    PhaseMeter meter_;
    gf128 delta_;
    bool aborted_ = false;
    std::unique_ptr<CotSender> sender_;
    std::unique_ptr<CotReceiver> receiver_;
    std::unique_ptr<AbitSource> abits_;
    uint64_t batch_ = 0;         // and_triples batches so far
    uint64_t triple_index_ = 0;  // leaky triples so far (tweak of H)
    unsigned last_bucket_ = 0;
    size_t last_run_ = 0;        // triples of the latest protocol run (>= the request)
    unsigned threads_ = 1;
    PrepCounters counters_;
};

// ------------------------------------------------------------------ ideal preprocessing
// Trusted source of correlated randomness for two in-process parties (ideal mode).
class IdealDealer {
public:
    enum Kind { BDOZ, BITS, MASKS, TRIPLES };

    // One party's part of a request (only the fields of its kind are filled).
    struct Part {
        BdozPair bdoz;
        Shares bits;
        InputMasks masks;
        Triples triples;
    };

    explicit IdealDealer(RandomSource* rng = nullptr) : rng_(rng ? rng : &sys_) {
        deltas_[0] = sample_delta(0, *rng_);
        deltas_[1] = sample_delta(1, *rng_);
        next_[0] = next_[1] = 0;
    }
    IdealDealer(const IdealDealer&) = delete;
    IdealDealer& operator=(const IdealDealer&) = delete;

    gf128 delta_share(int role) const {
        check_role(role);
        return deltas_[role];
    }
    // Delta^B = Delta^B_0 XOR Delta^B_1 (known to the dealer only; for tests).
    gf128 delta() const { return deltas_[0] ^ deltas_[1]; }

    // Replace a key share (tests of the lsb constraint).  Not thread-safe;
    // call before any request.
    void set_delta_share(int role, const gf128& d) {
        check_role(role);
        deltas_[role] = d;
    }

    // Serve request (kind, p0, p1) of party `role`.  The two parties must
    // issue the same sequence of requests.
    Part request(int role, Kind kind, uint64_t p0, uint64_t p1 = 0) {
        check_role(role);
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t seq = next_[role]++;
        std::map<uint64_t, Pending>::iterator it = pending_.find(seq);
        if (it != pending_.end()) {
            Pending pend = std::move(it->second);
            pending_.erase(it);
            if (pend.kind != kind || pend.p0 != p0 || pend.p1 != p1)
                throw std::invalid_argument("ideal dealer: request " + std::to_string(seq) +
                                            " differs between the parties");
            return std::move(pend.part);
        }
        Part parts[2];
        switch (kind) {
            case BDOZ:
                make_bdoz((size_t)p0, parts[0].bdoz, parts[1].bdoz);
                break;
            case BITS: {
                BdozPair b[2];
                make_bdoz((size_t)p0, b[0], b[1]);
                for (int r = 0; r < 2; r++) parts[r].bits = spdz_shares(r, deltas_[r], b[r].own, b[r].keys);
                break;
            }
            case MASKS:
                make_masks((int)p0, (size_t)p1, parts);
                break;
            case TRIPLES:
                make_triples((size_t)p0, parts);
                break;
            default:
                throw std::invalid_argument("ideal dealer: unknown request kind");
        }
        Pending pend;
        pend.kind = kind;
        pend.p0 = p0;
        pend.p1 = p1;
        pend.part = std::move(parts[1 - role]);
        pending_.emplace(seq, std::move(pend));
        return std::move(parts[role]);
    }

private:
    struct Pending {
        Kind kind = BDOZ;
        uint64_t p0 = 0, p1 = 0;
        Part part;  // the part of the party that has not asked yet
    };

    BitVec rand_bits(size_t n) {
        BitVec b(n);
        bytes raw((n + 7) / 8);
        if (!raw.empty()) rng_->fill(raw.data(), raw.size());
        for (size_t i = 0; i < n; i++) b[i] = (uint8_t)((raw[i / 8] >> (i % 8)) & 1);
        return b;
    }
    GfVec rand_vec(size_t n) {
        GfVec v(n);
        bytes raw(16 * n);
        if (n) rng_->fill(raw.data(), raw.size());
        gf_load(v.data(), raw.data(), n);
        return v;
    }

    // Random BDOZ bits of both parties: k_b are the keys of P_b for the peer's bits.
    void make_bdoz(size_t n, BdozPair& p0, BdozPair& p1) {
        BitVec x0 = rand_bits(n), x1 = rand_bits(n);
        GfVec k0 = rand_vec(n), k1 = rand_vec(n);
        GfVec m0(n), m1(n);
        for (size_t i = 0; i < n; i++) {
            m0[i] = k1[i] ^ gf_and_bit(deltas_[1], x0[i]);
            m1[i] = k0[i] ^ gf_and_bit(deltas_[0], x1[i]);
        }
        p0.own = Bdoz(std::move(x0), std::move(m0));
        p0.keys = Keys(std::move(k0));
        p1.own = Bdoz(std::move(x1), std::move(m1));
        p1.keys = Keys(std::move(k1));
    }

    void make_masks(int owner, size_t n, Part parts[2]) {
        check_role(owner);
        BitVec r = rand_bits(n);
        GfVec key = rand_vec(n);
        GfVec mac(n);
        for (size_t i = 0; i < n; i++) mac[i] = key[i] ^ gf_and_bit(deltas_[1 - owner], r[i]);
        const Bdoz own(r, std::move(mac));
        parts[owner].masks.shares = mask_owner(deltas_[owner], own);
        parts[owner].masks.bits = std::move(r);
        parts[owner].masks.has_bits = true;
        parts[1 - owner].masks.shares = mask_other(Keys(std::move(key)));
        parts[1 - owner].masks.has_bits = false;
    }

    // Fresh <v> sharing of the given bits.
    void share(const BitVec& values, Shares& s0, Shares& s1) {
        const size_t n = values.size();
        const gf128 d = delta();
        s0.bits = rand_bits(n);
        s0.macs = rand_vec(n);
        s1 = Shares(n);
        for (size_t i = 0; i < n; i++) {
            s1.bits[i] = (uint8_t)(s0.bits[i] ^ values[i]);
            s1.macs[i] = s0.macs[i] ^ gf_and_bit(d, values[i]);
        }
    }

    void make_triples(size_t n, Part parts[2]) {
        const BitVec a = rand_bits(n), b = rand_bits(n);
        BitVec c(n);
        for (size_t i = 0; i < n; i++) c[i] = (uint8_t)(a[i] & b[i]);
        share(a, parts[0].triples.a, parts[1].triples.a);
        share(b, parts[0].triples.b, parts[1].triples.b);
        share(c, parts[0].triples.c, parts[1].triples.c);
    }

    SystemRng sys_;
    RandomSource* rng_;
    gf128 deltas_[2];
    std::mutex mutex_;
    std::map<uint64_t, Pending> pending_;
    uint64_t next_[2];
};

// One party's handle on an IdealDealer (interface of RealPrep; ideal mode).
class IdealPrep : public Prep {
public:
    IdealPrep(int role, IdealDealer& dealer) : role_(role), dealer_(&dealer) {
        check_role(role);
        delta_ = dealer.delta_share(role);
    }

    int role() const override { return role_; }
    bool ideal() const override { return true; }
    bool aborted() const override { return aborted_; }
    gf128 delta() const override { return delta_; }
    const PrepCounters& counters() const override { return counters_; }

    void setup() override {}  // nothing to do: the dealer holds the keys

    // Retire the session: no further correlations are handed out.
    void abort() override {
        aborted_ = true;
        delta_ = gf128();
    }

    Shares random_bits(size_t n) override {
        enter();
        Shares out = dealer_->request(role_, IdealDealer::BITS, n).bits;
        counters_.random_bits += n;
        return out;
    }
    BdozPair random_bdoz(size_t n) override {
        enter();
        return dealer_->request(role_, IdealDealer::BDOZ, n).bdoz;
    }
    InputMasks input_masks(int owner, size_t n) override {
        enter();
        if (owner != 0 && owner != 1) throw std::invalid_argument("owner must be 0 or 1");
        InputMasks out = dealer_->request(role_, IdealDealer::MASKS, (uint64_t)owner, n).masks;
        counters_.input_masks += n;
        return out;
    }
    Triples and_triples(size_t n) override {
        enter();
        Triples out = dealer_->request(role_, IdealDealer::TRIPLES, n).triples;
        counters_.and_triples += n;
        return out;
    }

private:
    void enter() const {
        if (aborted_) throw ProtocolAbort("session aborted");
    }

    int role_;
    IdealDealer* dealer_;
    gf128 delta_;
    bool aborted_ = false;
    PrepCounters counters_;
};

}  // namespace prep
}  // namespace couplet
