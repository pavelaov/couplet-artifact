// couplet/session.hpp -- session of one party: preprocessing, Boolean engine
// and the session MAC key (DESIGN.md sections 1, 4, 5.1, 5.4; changes G, H);
// the C++ counterpart of python/couplet/session.py.
//
// A Session owns, for one party and one session identifier sid:
//
//   * the Boolean preprocessing (prep::RealPrep, or prep::IdealPrep on a
//     shared in-process dealer; an external prep::Prep is accepted as well);
//   * the Boolean engine (abb::Engine) and the accounting book it writes to;
//   * this party's arithmetic MAC key share Delta_b, uniform in [0, 2^s), and
//     the persistent authenticated wires of Delta = Delta_0 + Delta_1
//     (s + 1 bits, never reduced, never opened).
//
// setup(reserve_triples, reserve_outmasks) runs prep.setup(), samples Delta_b
// and evaluates the session-setup circuit "setup_s<s>" once: each party
// inputs its Delta_b and the sum stays in the session as RETAIN wires
// (change H).
//
// Bucketing populations (DESIGN.md 1 and 7.5, Lemma 4; design/CHANGELOG.md F1
// W2 and item 1, F2 item 2).  The statistical bound 2^-sigma holds PER
// preprocessing request for AND triples.  setup draws, in ONE request,
//
//     n_and(setup circuit) + reserve_triples + reserve_outmasks * n_and(C_om)
//
// triples, C_om being the on-demand output-mask circuit
// "gate_n1_k<k>_s<s>_shared_om1" (4,159 AND gates for s = 64); a session
// whose later circuits fit into this reservation consists of a single
// population and is at 2^-sigma as a whole.  protocol::and_budget computes
// reserve_triples for a planned list of generation calls; reserve_outmasks is
// a triple budget only (the masks are generated when they are needed).
// triple_requests() counts the requests made so far (also in info_json(),
// with the resulting bound).  What happens when the pool is too small for an
// evaluation is decided by allow_refill(): with true (the default of this
// class, as of abb::Engine) the engine requests the missing triples, which is
// one more population and one more count; with false (strict mode: the
// default of the driver couplet::Party, which switches its session to it,
// and of the benchmark programs) evaluate throws abb::TripleBudgetError
// before any message, the session stays ACTIVE, and the caller has to
// reserve.  reserve_triples(n) after setup is an explicit further request in
// both modes.
//
// evaluate(circuit, my_inputs, batch, bound) runs one circuit on the engine;
// a BOUND input named "delta" is bound to the session wires unless the caller
// binds it explicitly.  The engine compares the circuit digest with the peer
// in the first exchange of the evaluation.
//
// Circuits are not compiled here: they are loaded from a
// circuit::CircuitStore (files exported by python/bench/export_circuits.py).
//
// State (section 5.4): ACTIVE or ABORTED (absorbing).  A ProtocolAbort inside
// setup, evaluate or a guard() block moves the session to ABORTED: the key
// shares, the session wires and the preprocessing are dropped and every later
// call throws ProtocolAbort (change G).  Any other exception aborts the
// session if the engine or the preprocessing died.  Material handed out
// earlier (keys, shares) must not be used once the session is ABORTED.
// delta_share() and delta_wires() hand out key material and therefore check
// the state themselves: they throw ProtocolAbort unless the session is
// ACTIVE and its engine and preprocessing are alive.  A component that runs
// next to the session with a copy of Delta_b (spdz2k::ArithChecker) is tied
// to it with bind_abort(component): when the component aborts, the session
// aborts.
//
// Parameter range of the native code: 1 <= k <= 64, 1 <= s <= 64 (the key
// share is one machine word and ring elements are 128-bit integers; the
// reference allows 2(k + s) <= 320).  Both profiles of DESIGN.md,
// (k, s) = (32, 64) and (64, 64), are inside the range.
//
// Accounting: the setup evaluation is booked under "setup-input",
// "setup-online", "setup-outcheck", "setup-output"; later evaluations under
// the engine's plain names; RealPrep books its own phases (stats()).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "abb.hpp"
#include "channel.hpp"
#include "circuit.hpp"
#include "hashes.hpp"
#include "prep.hpp"
#include "rng.hpp"

namespace couplet {

enum class SessionState { ACTIVE, ABORTED };

class Session {
public:
    static const char* delta_label() { return "delta"; }
    // Names of the two circuits the session itself needs from the store.
    static std::string setup_circuit_name(unsigned s) { return "setup_s" + std::to_string(s); }
    static std::string outmask_circuit_name(unsigned k, unsigned s) {
        return "gate_n1_k" + std::to_string(k) + "_s" + std::to_string(s) + "_shared_om1";
    }

    // Real preprocessing (base OT, OT extension, bucketed AND triples).
    // `rng` (tests) makes this party's secret choices reproducible.
    Session(int role, Channel& ch, const bytes& sid, circuit::CircuitStore& store, unsigned k = 64, unsigned s = 64,
            int sigma = prep::SIGMA_DEFAULT, RandomSource* rng = nullptr)
        : role_(role), ch_(&ch), sid_(sid), store_(&store), k_(k), s_(s), sigma_(sigma), rng_(rng), book_(ch) {
        check_params(role, ch, k, s);
        if (sigma < 1) throw std::invalid_argument("sigma must be positive");
        real_ = new prep::RealPrep(role, ch, sid_, sigma, rng);
        owned_.reset(real_);
        prep_ = real_;
        engine_.reset(new abb::Engine(role, ch, *prep_, sid_, &book_));
    }

    // Ideal preprocessing on a shared in-process dealer (DESIGN.md 7.7; NOT a protocol).
    Session(int role, Channel& ch, const bytes& sid, circuit::CircuitStore& store, prep::IdealDealer& dealer,
            unsigned k = 64, unsigned s = 64, RandomSource* rng = nullptr)
        : role_(role), ch_(&ch), sid_(sid), store_(&store), k_(k), s_(s), sigma_(0), rng_(rng), book_(ch) {
        check_params(role, ch, k, s);
        owned_.reset(new prep::IdealPrep(role, dealer));
        prep_ = owned_.get();
        engine_.reset(new abb::Engine(role, ch, *prep_, sid_, &book_));
    }

    // An existing preprocessing object (it must outlive the session).
    Session(int role, Channel& ch, const bytes& sid, circuit::CircuitStore& store, prep::Prep& prep, unsigned k = 64,
            unsigned s = 64, RandomSource* rng = nullptr)
        : role_(role), ch_(&ch), sid_(sid), store_(&store), k_(k), s_(s), sigma_(0), rng_(rng), book_(ch) {
        check_params(role, ch, k, s);
        prep_ = &prep;
        real_ = dynamic_cast<prep::RealPrep*>(&prep);
        if (real_) sigma_ = real_->sigma();
        engine_.reset(new abb::Engine(role, ch, *prep_, sid_, &book_));
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    int role() const { return role_; }
    Channel& channel() { return *ch_; }
    const bytes& sid() const { return sid_; }
    unsigned k() const { return k_; }
    unsigned s() const { return s_; }
    unsigned w() const { return k_ + s_; }
    int sigma() const { return sigma_; }
    circuit::CircuitStore& store() { return *store_; }
    abb::Engine& engine() { return *engine_; }
    abb::PhaseBook& book() { return book_; }
    prep::Prep& prep() { return *prep_; }
    prep::RealPrep* real_prep() { return real_; }  // nullptr with ideal preprocessing
    bool ideal() const { return prep_->ideal(); }
    SessionState state() const { return state_; }
    bool active() const { return state_ == SessionState::ACTIVE; }
    bool is_setup() const { return is_setup_; }
    const abb::EvalStats& setup_stats() const { return setup_stats_; }

    // This party's Delta_b and the session wires of Delta.  Both are key
    // material: they are handed out only while the session is ACTIVE and its
    // engine and preprocessing are alive (ProtocolAbort otherwise; a dead
    // engine found here moves the session to ABORTED).  Before setup(),
    // delta_share() is available only if the test hook set_delta_share() was
    // used, and delta_wires() is empty.
    bool has_delta_share() const { return has_delta_; }
    uint64_t delta_share() {
        require_active();
        if (!has_delta_) throw std::invalid_argument("the session has no MAC key share");
        return delta_share_;
    }
    const abb::WireShares& delta_wires() {
        require_active();
        return delta_wires_;
    }

    // -- bucketing populations (see the header comment)
    uint64_t triple_requests() const { return engine_->triple_requests(); }      // preprocessing requests for AND triples
    uint64_t triples_requested() const { return engine_->triples_requested(); }  // triples drawn by them
    uint64_t triples_used() const { return engine_->triples_used(); }
    size_t triples_unused() const { return engine_->triples_available(); }
    bool allow_refill() const { return engine_->allow_refill(); }
    // false = strict mode: a short pool is abb::TripleBudgetError, not a new population.
    void set_allow_refill(bool allow) { engine_->set_allow_refill(allow); }
    // AND gates of the on-demand output-mask circuit of this session (one mask per instance).
    size_t outmask_and_count() { return store_->get(outmask_circuit_name(k_, s_))->n_and(); }
    // log2 of the bucketing term of the session, log2(triple_requests) - sigma.
    // Defined for the real preprocessing once a request was made; returns
    // false otherwise (the ideal dealer has no bucketing).
    bool bucketing_bound_log2(double* out) const {
        const uint64_t requests = engine_->triple_requests();
        if (!real_ || requests == 0) return false;
        if (out) *out = std::log2((double)requests) - (double)sigma_;
        return true;
    }
    // Public parameters and counters of the session (no secrets), as JSON.
    std::string info_json() const {
        double bound = 0;
        char buf[64] = "null";
        if (bucketing_bound_log2(&bound)) std::snprintf(buf, sizeof buf, "%.6f", bound);
        return std::string("{\"role\": ") + std::to_string(role_) + ", \"k\": " + std::to_string(k_) + ", \"s\": " +
               std::to_string(s_) + ", \"sigma\": " + std::to_string(sigma_) + ", \"prep\": \"" +
               (prep_->ideal() ? "ideal" : "real") + "\", \"state\": \"" +
               (state_ == SessionState::ACTIVE ? "ACTIVE" : "ABORTED") + "\", \"evaluations\": " +
               std::to_string(engine_->evaluations()) + ", \"triples_used\": " + std::to_string(engine_->triples_used()) +
               ", \"triples_unused\": " + std::to_string(engine_->triples_available()) + ", \"triple_requests\": " +
               std::to_string(engine_->triple_requests()) + ", \"triples_requested\": " +
               std::to_string(engine_->triples_requested()) + ", \"allow_refill\": " +
               (engine_->allow_refill() ? "true" : "false") + ", \"bucketing_bound_log2\": " + buf +
               ", \"last_bucket_size\": " + (real_ ? std::to_string(real_->last_bucket_size()) : std::string("null")) + "}";
    }

    // Tie a component that holds a copy of this session's key material (e.g.
    // a spdz2k::ArithChecker built from delta_share()) to the session: when
    // the component aborts, the session aborts (change G).  The session must
    // outlive the component.
    template <class Component>
    void bind_abort(Component& component) {
        component.set_on_abort([this] { abort(); });
    }

    // Test hook: fix Delta_b before setup() (e.g. to force Delta_0 + Delta_1 >= 2^s).
    void set_delta_share(uint64_t d) {
        if (is_setup_) throw std::invalid_argument("setup() was already run");
        if (s_ < 64 && (d >> s_) != 0) throw std::invalid_argument("delta_share must lie in [0, 2^s)");
        delta_share_ = d;
        has_delta_ = true;
    }

    // Description of the latest generation run on this session (filled by protocol.hpp).
    struct LastGen {
        std::string circuit, digest;
        size_t batch = 0;
        abb::EvalStats stats;
    };
    LastGen last_gen;

    // Throws ProtocolAbort unless the session is ACTIVE.
    void require_active() {
        if (state_ != SessionState::ACTIVE) throw ProtocolAbort("session aborted");
        if (engine_->dead() || prep_->aborted()) {
            abort();
            throw ProtocolAbort("session aborted");
        }
    }

    // Move to ABORTED: retire the key shares, the session wires and the preprocessing.
    void abort() {
        state_ = SessionState::ABORTED;
        delta_share_ = 0;
        has_delta_ = false;
        delta_wires_ = abb::WireShares();
        engine_->abort();
    }

    // Run a protocol step: a ProtocolAbort inside aborts the session.
    template <class F>
    auto guard(F&& body) -> decltype(body()) {
        require_active();
        try {
            return body();
        } catch (const ProtocolAbort&) {
            abort();
            throw;
        } catch (...) {
            if (engine_->dead() || prep_->aborted()) abort();
            throw;
        }
    }

    // Next identifier of `kind` in ("kid", "gid", "qid"); monotone, never reused.
    uint64_t next_id(const std::string& kind) {
        int idx = kind == "kid" ? 0 : kind == "gid" ? 1 : kind == "qid" ? 2 : -1;
        if (idx < 0) throw std::invalid_argument("kind must be one of kid, gid, qid");
        require_active();
        return ids_[idx]++;
    }

    // Preprocessing setup and the persistent wires of Delta (once per
    // session).  The triples of the setup circuit, reserve_triples further
    // ones and those of reserve_outmasks on-demand output masks are drawn in
    // ONE request (one bucketing population).
    void setup(size_t reserve_triples = 0, size_t reserve_outmasks = 0) {
        require_active();
        if (is_setup_) throw std::invalid_argument("setup() was already run");
        const std::shared_ptr<const circuit::Circuit> c = store_->get(setup_circuit_name(s_));
        const circuit::Decl* in = c->find_input((uint32_t)role_, delta_label());
        const circuit::Decl* out = c->find_output(delta_label());
        if (!in || in->wires.size() != s_ || !out || out->owner != circuit::OUT_RETAIN || out->wires.size() != s_ + 1)
            throw std::invalid_argument("the session-setup circuit has an unexpected interface");
        if (reserve_outmasks) {
            const size_t per_mask = outmask_and_count();
            if (per_mask && reserve_outmasks > (SIZE_MAX - reserve_triples) / per_mask)
                throw std::invalid_argument("the triple reservation is too large");
            reserve_triples += reserve_outmasks * per_mask;
        }
        if (reserve_triples > SIZE_MAX - c->n_and()) throw std::invalid_argument("the triple reservation is too large");
        guard([&] {
            prep_->setup();
            if (!has_delta_) {
                SystemRng sys;
                RandomSource& r = rng_ ? *rng_ : static_cast<RandomSource&>(sys);
                uint64_t d = r.u64();
                if (s_ < 64) d &= ((uint64_t)1 << s_) - 1;
                delta_share_ = d;
                has_delta_ = true;
            }
            abb::Inputs inputs;
            inputs.emplace(delta_label(), circuit::BitMat::from_u64({delta_share_}, s_));
            engine_->set_phase_tag("setup-");
            struct Untag {
                abb::Engine& e;
                ~Untag() { e.set_phase_tag(""); }
            } untag{*engine_};
            engine_->reserve_triples(c->n_and() + reserve_triples);
            abb::EvalResult result = engine_->eval_circuit(*c, inputs, abb::Bound(), 1);
            delta_wires_ = std::move(result.retained.at(delta_label()));
            setup_stats_ = result.stats;
            is_setup_ = true;
            return 0;
        });
    }

    // Draw n further AND triples in one preprocessing request (one more
    // population).  Explicit in both modes: triple_requests() grows by one if n > 0.
    void reserve_triples(size_t n) {
        guard([&] {
            engine_->reserve_triples(n);
            return 0;
        });
    }

    // Evaluate `c` on the session engine (see abb::Engine::eval_circuit).  A
    // BOUND input "delta" is bound to the session wires of Delta unless
    // `bound` already contains it.
    abb::EvalResult evaluate(const circuit::Circuit& c, const abb::Inputs& my_inputs, size_t batch = 1,
                             const abb::Bound& bound = abb::Bound()) {
        require_active();
        if (!is_setup_) throw std::invalid_argument("setup() has not been run");
        const circuit::Decl* d = c.find_input(circuit::IN_BOUND, delta_label());
        if (d && !bound.count(delta_label())) {
            if (d->wires.size() != s_ + 1) throw std::invalid_argument("the circuit was built for another s");
            abb::Bound with = bound;
            with.emplace(delta_label(), delta_wires_);
            return guard([&] { return engine_->eval_circuit(c, my_inputs, with, batch); });
        }
        return guard([&] { return engine_->eval_circuit(c, my_inputs, bound, batch); });
    }

private:
    static void check_params(int role, const Channel& ch, unsigned k, unsigned s) {
        prep::check_role(role);
        if (ch.party() != role) throw std::invalid_argument("role does not match the channel endpoint");
        if (k < 1 || s < 1) throw std::invalid_argument("need k >= 1 and s >= 1");
        if (k > 64 || s > 64) throw std::invalid_argument("this implementation supports k <= 64 and s <= 64");
    }

    int role_;
    Channel* ch_;
    bytes sid_;
    circuit::CircuitStore* store_;
    unsigned k_, s_;
    int sigma_;
    RandomSource* rng_;
    abb::PhaseBook book_;
    std::unique_ptr<prep::Prep> owned_;
    prep::Prep* prep_ = nullptr;
    prep::RealPrep* real_ = nullptr;
    std::unique_ptr<abb::Engine> engine_;
    SessionState state_ = SessionState::ACTIVE;
    bool is_setup_ = false;
    bool has_delta_ = false;
    uint64_t delta_share_ = 0;
    abb::WireShares delta_wires_;
    abb::EvalStats setup_stats_;
    uint64_t ids_[3] = {0, 0, 0};
};

}  // namespace couplet
