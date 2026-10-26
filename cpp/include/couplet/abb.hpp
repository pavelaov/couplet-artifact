// couplet/abb.hpp -- party-local authenticated Boolean engine (DESIGN.md
// sections 2.2, 5.1, 7.6); the C++ counterpart of python/couplet/abb.py,
// with the same messages, labels and derivations.
//
// Engine(role, channel, prep, sid) is the view of ONE party of the reactive
// functionality F_ABB.  It evaluates a circuit::Circuit on SPDZ-style
// authenticated bits <x> (XOR shares of the bit, XOR shares of the MAC
// x * Delta^B in GF(2^128)).  Preprocessed material (AND triples and input
// masks) comes from a prep::Prep (RealPrep or IdealPrep).
//
// Batch dimension.  Every wire carries `batch` independent instances, so one
// circuit execution produces `batch` results and all instances share the
// rounds.  Wire tables are row-major: entry [w * batch + j] belongs to wire w
// of instance j.  With the constant-one row (Delta^B_role, [role == 0]) on
// wire 1, a NOT gate is an XOR with wire 1.
//
// eval_circuit(circuit, my_inputs, bound, batch) runs, in this order:
//
//   1. preprocessing draws: n_and * batch AND triples (one request, or the
//      pool filled by reserve_triples), then the input masks of P_0 and of
//      P_1: one mask per input bit and per private output bit of the party;
//   2. Input of all input wires of both parties in ONE simultaneous exchange:
//      each party sends the 32-byte header
//          SHA-256(frame("Couplet-abb-eval", sid', le64(evaluation counter),
//                        circuit digest, le64(batch), H(bound)))
//      followed by d = x XOR r for its own inputs, bit-packed.  A header
//      mismatch aborts: the parties run different circuits, batch sizes or
//      bound wires.  H(bound) hashes, for every BOUND input, its label, the
//      session-wide identifiers of the wires bound to it and their batch
//      size.  sid' = SHA-256(frame("Couplet-abb-sid", sid));
//   3. XOR / NOT gates locally; one simultaneous exchange per AND layer with
//      the share bits of d = x XOR a for all gates and instances (gate-major)
//      followed by those of e = y XOR b;  <z> = <c> XOR d<b> XOR e<a> XOR d e;
//   4. Output exactly as in section 2.2 (change F):
//        (1) BoolCheck over every d, e opened during the evaluation;
//        (2) ONE exchange opening the public outputs and, for every private
//            output of P_b, the masked wire <y> XOR <r> with r a FRESH input
//            mask of P_b (order: public, private of P_0, private of P_1);
//        (3) a second BoolCheck over exactly these openings;
//        (4) only then is anything returned.
//      RETAIN outputs are not opened and come back as WireShares.
//
// BoolCheck is prep::bool_check (the hash-based refinement recorded in
// design/CHANGELOG.md, PY-C item 1) under sid' with a counter that is never
// reused.  A check over zero openings is skipped; both parties know this from
// the circuit.
//
// Rounds of one evaluation: 1 (inputs) + AND depth + 2 + 1 + 2, plus the
// rounds of the preprocessing requests.
//
// Abort (change G).  Any failure after the argument checks, and any failure
// inside the preprocessing, marks the engine dead: the MAC key share and the
// triple pool are dropped, prep.abort() is called and every later call throws
// ProtocolAbort.  Invalid arguments throw std::invalid_argument before any
// message and leave the engine alive.
//
// Accounting.  PhaseBook attributes channel traffic and wall-clock time to
// named phases (the C++ channel has none).  The engine books "input",
// "online" (AND layers), "outcheck" (the two BoolChecks) and "output" (the
// opening), prefixed with phase_tag.  While a preprocessing method runs the
// book is switched off, because RealPrep books that traffic itself; with
// ideal preprocessing the time spent in the dealer is booked under "prep".
//
// Statistical accounting (Lemma 4; design/CHANGELOG.md F1 item 1, F2 item 2).
// The bound 2^-sigma of the bucketing holds PER prep.and_triples request:
// every request is one bucketing population, and a session that makes
// triple_requests() requests is at triple_requests() * 2^-sigma for the
// triples.  The engine makes this explicit:
//
//   * triple_requests() counts the requests, triples_requested() the triples
//     drawn by them; both are reported in EvalStats, with the pool size;
//   * reserve_triples(n) draws the triples of several evaluations in ONE
//     request (Session::setup reserves together with the setup circuit);
//   * allow_refill() decides what happens when the pool is too small for an
//     evaluation.  true (the default of this low-level class): the missing
//     triples are requested, which is one more population and one more
//     count.  false (strict mode; the default of the driver couplet::Party
//     and of the benchmark programs): eval_circuit throws TripleBudgetError
//     BEFORE any message, the engine stays alive and nothing was consumed;
//     the caller has to reserve.  Both parties hold the same pool, so both
//     throw at the same point.
//
// Memory: about 17 * (n_wires + 5 n_and) * batch bytes per evaluation.
//
// Received messages are length-bounded (channel.hpp): every exchange of the
// engine names the exact length the circuit prescribes for the reply.
//
// The engine has no test hooks: the adversarial tests rewrite messages with
// TamperChannel (twoparty.hpp).
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "channel.hpp"
#include "circuit.hpp"
#include "gf128.hpp"
#include "hashes.hpp"
#include "prep.hpp"
#include "rng.hpp"
#include "shares.hpp"

namespace couplet {
namespace abb {

using circuit::BitMat;
using circuit::Circuit;
using prep::BitVec;
using prep::GfVec;

// The triple pool is too small for an evaluation and refills are not allowed
// (strict mode).  Thrown before any protocol message: the engine and the
// session stay usable and nothing was consumed.  It is a
// std::invalid_argument because, like an invalid argument, it is detected
// locally, at the same point by both parties, and does not abort the session.
struct TripleBudgetError : std::invalid_argument {
    size_t needed, available;
    TripleBudgetError(size_t needed_, size_t available_)
        : std::invalid_argument(
              "the evaluation needs " + std::to_string(needed_) + " AND triples but the reserved pool holds " +
              std::to_string(available_) +
              ": reserve the triples of the whole session when it is set up (Session::setup / Party::setup with "
              "reserve_triples and reserve_outmasks; protocol::and_budget computes the number), so that they form ONE "
              "bucketing population, or allow refills explicitly (set_allow_refill(true)) to accept one more "
              "population (2^-sigma each) per refill"),
          needed(needed_), available(available_) {}
};

// ------------------------------------------------------------------ accounting
// Named accounting phases over one channel endpoint.  mark(p) closes the
// running phase and starts p; NONE means "booked elsewhere" (nothing is
// recorded).  Phases keep the order in which they were first named.
class PhaseBook {
public:
    static const int NONE = -1;

    explicit PhaseBook(Channel& ch) : ch_(&ch), cur_(NONE) { snap(); }

    // Identifier of the phase `name` (registered on first use).
    int id(const std::string& name) {
        for (size_t i = 0; i < names_.size(); i++)
            if (names_[i] == name) return (int)i;
        names_.push_back(name);
        stats_.push_back(prep::PhaseStat());
        return (int)names_.size() - 1;
    }
    int current() const { return cur_; }
    int mark(int p) {
        close();
        const int prev = cur_;
        cur_ = (p >= 0 && p < (int)names_.size()) ? p : NONE;
        return prev;
    }
    int mark(const std::string& name) { return mark(id(name)); }

    size_t size() const { return names_.size(); }
    const std::string& name(size_t i) const { return names_[i]; }
    // Statistics so far (the running phase is included up to now).
    const prep::PhaseStat& stat(size_t i) {
        close();
        return stats_[i];
    }
    prep::PhaseStat get(const std::string& name) {
        close();
        for (size_t i = 0; i < names_.size(); i++)
            if (names_[i] == name) return stats_[i];
        return prep::PhaseStat();
    }
    prep::PhaseStat total() {
        close();
        prep::PhaseStat t;
        for (const prep::PhaseStat& s : stats_) t += s;
        return t;
    }
    std::string json() {
        close();
        std::string s = "{";
        for (size_t i = 0; i < names_.size(); i++) {
            if (i) s += ",";
            s += circuit_json_quote(names_[i]) + ":" + stats_[i].json();
        }
        return s + "}";
    }

private:
    static std::string circuit_json_quote(const std::string& s) { return json::quote(s); }
    void snap() {
        snap_ = ch_->stats();
        t_ = std::chrono::steady_clock::now();
    }
    void close() {
        const ChannelStats now = ch_->stats();
        const std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
        if (cur_ >= 0 && cur_ < (int)stats_.size()) {
            prep::PhaseStat& s = stats_[(size_t)cur_];
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
    std::vector<std::string> names_;
    std::vector<prep::PhaseStat> stats_;
    int cur_;
    ChannelStats snap_;
    std::chrono::steady_clock::time_point t_;
};

// Restores the phase that was current when the scope was entered.
class PhaseRestore {
public:
    explicit PhaseRestore(PhaseBook& book) : book_(&book), prev_(book.current()) {}
    ~PhaseRestore() { book_->mark(prev_); }
    PhaseRestore(const PhaseRestore&) = delete;
    PhaseRestore& operator=(const PhaseRestore&) = delete;

private:
    PhaseBook* book_;
    int prev_;
};

// ------------------------------------------------------------------ wire shares
// One party's shares of `width` authenticated wires over `batch` instances:
// bits[i * batch + j] and macs[i * batch + j] are the XOR share of wire i in
// instance j and the share of its MAC; ids[i] is the session-wide identifier
// of wire i (equal at both parties; part of the evaluation header when the
// wires are bound to a later circuit).  The object belongs to the engine
// that produced it (token) and cannot be bound in another session.
struct WireShares {
    size_t width = 0, batch = 0;
    BitVec bits;
    GfVec macs;
    std::vector<uint64_t> ids;
    uint64_t token = 0;

    bool empty() const { return width == 0; }

    void validate() const {
        if (bits.size() != width * batch || macs.size() != width * batch || ids.size() != width)
            throw std::invalid_argument("inconsistent wire share arrays");
    }

    // Wires [a, b).
    WireShares slice(size_t a, size_t b) const {
        if (a > b || b > width) throw std::out_of_range("WireShares::slice");
        WireShares r;
        r.width = b - a;
        r.batch = batch;
        r.token = token;
        r.bits.assign(bits.begin() + (ptrdiff_t)(a * batch), bits.begin() + (ptrdiff_t)(b * batch));
        r.macs.assign(macs.begin() + (ptrdiff_t)(a * batch), macs.begin() + (ptrdiff_t)(b * batch));
        r.ids.assign(ids.begin() + (ptrdiff_t)a, ids.begin() + (ptrdiff_t)b);
        return r;
    }

    // Concatenation of wire blocks of one engine with the same batch size.
    static WireShares concat(const std::vector<WireShares>& parts) {
        if (parts.empty()) throw std::invalid_argument("expected a non-empty sequence of WireShares");
        WireShares r;
        r.batch = parts[0].batch;
        r.token = parts[0].token;
        for (const WireShares& p : parts) {
            if (p.token != r.token || p.batch != r.batch)
                throw std::invalid_argument("wire blocks must come from one engine and have one batch size");
            r.width += p.width;
            r.bits.insert(r.bits.end(), p.bits.begin(), p.bits.end());
            r.macs.insert(r.macs.end(), p.macs.begin(), p.macs.end());
            r.ids.insert(r.ids.end(), p.ids.begin(), p.ids.end());
        }
        return r;
    }
};

typedef std::map<std::string, BitMat> Inputs;     // label -> bits (width, 1 or batch) of this party's inputs
typedef std::map<std::string, WireShares> Bound;  // label -> session wires for the BOUND inputs
typedef std::map<std::string, BitMat> Outputs;    // label -> opened bits (width, batch)

struct EvalStats {
    uint64_t evaluation = 0, batch = 0, n_gates = 0, n_and = 0, and_depth = 0;
    uint64_t triples = 0, input_masks[2] = {0, 0}, opened_bits = 0, checked_bits = 0;
    uint64_t rounds = 0, bytes_sent = 0;
    // Bucketing populations of the engine after this evaluation: requests for
    // AND triples so far, triples drawn by them, triples left in the pool.
    uint64_t triple_requests = 0, triples_requested = 0, triples_available = 0;
    double prep_s = 0, input_s = 0, online_s = 0, online_exchange_wait_s = 0, check_s = 0, output_s = 0, total_s = 0;

    // Gate instances per second of the online phase (0 if it took no measurable time).
    double gates_per_s() const { return online_s > 0 ? (double)(n_gates * batch) / online_s : 0.0; }
    double gates_per_s_local() const {
        const double t = online_s - online_exchange_wait_s;
        return t > 0 ? (double)(n_gates * batch) / t : 0.0;
    }
    std::string json() const {
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "\"prep_s\":%.6f,\"input_s\":%.6f,\"online_s\":%.6f,\"online_exchange_wait_s\":%.6f,"
                      "\"check_s\":%.6f,\"output_s\":%.6f,\"total_s\":%.6f,\"gates_per_s\":%.1f,"
                      "\"gates_per_s_local\":%.1f",
                      prep_s, input_s, online_s, online_exchange_wait_s, check_s, output_s, total_s, gates_per_s(),
                      gates_per_s_local());
        return "{\"evaluation\":" + std::to_string(evaluation) + ",\"batch\":" + std::to_string(batch) +
               ",\"n_gates\":" + std::to_string(n_gates) + ",\"n_and\":" + std::to_string(n_and) +
               ",\"and_depth\":" + std::to_string(and_depth) + ",\"triples\":" + std::to_string(triples) +
               ",\"input_masks\":[" + std::to_string(input_masks[0]) + "," + std::to_string(input_masks[1]) +
               "],\"opened_bits\":" + std::to_string(opened_bits) + ",\"checked_bits\":" +
               std::to_string(checked_bits) + ",\"rounds\":" + std::to_string(rounds) + ",\"bytes_sent\":" +
               std::to_string(bytes_sent) + ",\"triple_requests\":" + std::to_string(triple_requests) +
               ",\"triples_requested\":" + std::to_string(triples_requested) + ",\"triples_available\":" +
               std::to_string(triples_available) + "," + buf + "}";
    }
};

// Outcome of one circuit evaluation for this party.
struct EvalResult {
    Outputs pub;                                 // every PUBLIC output
    Outputs priv;                                // the private outputs owned by this party
    std::map<std::string, WireShares> retained;  // every RETAIN output
    EvalStats stats;

    // Opened output `label` (public or own private).
    const BitMat& bits(const std::string& label) const {
        Outputs::const_iterator it = pub.find(label);
        if (it != pub.end()) return it->second;
        it = priv.find(label);
        if (it == priv.end()) throw std::invalid_argument("no opened output " + label);
        return it->second;
    }
};

// Public transcript values of the latest evaluation (for tests): the public
// openings y XOR r of the private outputs and the values d = x XOR r sent for
// the inputs of P_0 and P_1 (input wires x batch, row-major).
struct Transcript {
    Outputs masked;
    BitVec input_diffs[2];
};

// ------------------------------------------------------------------ engine
class Engine {
public:
    // `book` is the accounting book to use (default: an own one).
    Engine(int role, Channel& ch, prep::Prep& prep, const bytes& sid, PhaseBook* book = nullptr)
        : role_(role), ch_(&ch), prep_(&prep), sid_(sid), own_book_(ch), book_(book ? book : &own_book_) {
        prep::check_role(role);
        if (ch.party() != role) throw std::invalid_argument("role does not match the channel endpoint");
        if (prep.role() != role) throw std::invalid_argument("role does not match the preprocessing object");
        if (prep.aborted()) throw std::invalid_argument("the preprocessing object has no MAC key share (aborted)");
        const hashes::hash32 h = hashes::framed_sha256({"Couplet-abb-sid", sid_});
        sid_abb_.assign(h.begin(), h.end());
        delta_ = prep.delta();
        static std::atomic<uint64_t> next_token(1);
        token_ = next_token.fetch_add(1);
    }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    int role() const { return role_; }
    Channel& channel() { return *ch_; }
    prep::Prep& prep() { return *prep_; }
    PhaseBook& book() { return *book_; }
    const bytes& sid() const { return sid_; }
    bool dead() const { return dead_; }
    uint64_t token() const { return token_; }
    uint64_t evaluations() const { return evaluations_; }
    uint64_t triple_requests() const { return triple_requests_; }    // bucketing populations drawn so far
    uint64_t triples_requested() const { return triples_requested_; }  // triples drawn by these requests
    uint64_t triples_used() const { return triples_used_; }

    // false = strict mode: an evaluation whose triples are not in the pool
    // throws TripleBudgetError instead of requesting a new population.
    bool allow_refill() const { return allow_refill_; }
    void set_allow_refill(bool allow) { allow_refill_ = allow; }

    // Strict mode: throw TripleBudgetError unless the pool holds n unused
    // triples.  With allow_refill() this never throws (a short pool is
    // refilled by the evaluation).  No message is sent.
    void require_triples(size_t n) const {
        if (!allow_refill_ && n > triples_available()) throw TripleBudgetError(n, triples_available());
    }
    uint64_t checks() const { return checks_; }
    const EvalStats& last_stats() const { return last_stats_; }
    const Transcript& last_transcript() const { return transcript_; }

    // Prefix of the phase names booked by the engine (e.g. "setup-").
    void set_phase_tag(const std::string& tag) { phase_tag_ = tag; }
    const std::string& phase_tag() const { return phase_tag_; }

    // Retire the engine (change G): drop the key share, the pool and the preprocessing.
    void abort() {
        dead_ = true;
        delta_ = gf128();
        pool_ = prep::Triples();
        pool_pos_ = 0;
        try {
            prep_->abort();
        } catch (...) {
        }
    }

    // Unused triples in the pool.
    size_t triples_available() const { return pool_.size() - pool_pos_; }

    // Draw n further AND triples into the pool with ONE preprocessing request.
    // Every call with n > 0 is one bucketing population (triple_requests()
    // grows by one); it is allowed in both modes because it is explicit.
    void reserve_triples(size_t n) {
        require_alive();
        if (n == 0) return;
        const PhaseRestore restore(*book_);
        try {
            book_->mark(prep_phase());
            fill_pool(n);
        } catch (...) {
            abort();
            throw;
        }
    }

    // Evaluate `c` on `batch` independent instances; see the header comment.
    //   my_inputs  label -> bits (width, batch) or (width, 1) (used for every
    //              instance) for every input declaration owned by this party;
    //   bound      label -> WireShares for every BOUND input, batch 1 or `batch`;
    //   batch      number of instances.
    EvalResult eval_circuit(const Circuit& c, const Inputs& my_inputs, const Bound& bound = Bound(), size_t batch = 1) {
        require_alive();
        if (batch < 1) throw std::invalid_argument("batch must be a positive integer");
        if ((uint64_t)c.n_wires > (uint64_t)(SIZE_MAX / 64) / batch)
            throw std::invalid_argument("circuit times batch is too large");
        const Layout lay = layout(c);
        const BitVec x_own = own_inputs(lay, my_inputs, batch);
        const hashes::hash32 bound_desc = check_bound(lay, bound, batch);
        require_triples(c.n_and() * batch);  // strict mode: refused before any message, the engine stays alive
        const PhaseRestore restore(*book_);
        try {
            return run(c, lay, x_own, bound, bound_desc, batch);
        } catch (...) {
            abort();
            throw;
        }
    }

private:
    struct Field {
        std::string label;
        size_t offset;  // into the owner's wire list / the opening vector
        size_t width;
        const circuit::Decl* decl;
    };
    struct Layout {
        std::vector<uint32_t> in_wires[2];   // input wires per owner, declaration order
        std::vector<Field> in_fields[2];
        std::vector<Field> bound_fields;
        std::vector<Field> out_public, out_private[2], out_retain;
        std::vector<uint32_t> open_wires;    // public, private of P_0, private of P_1
        size_t n_public = 0, n_private[2] = {0, 0};
        size_t n_in(int owner) const { return in_wires[owner].size(); }
        size_t n_masks(int owner) const { return in_wires[owner].size() + n_private[owner]; }
    };

    void require_alive() {
        if (dead_ || prep_->aborted()) {
            if (!dead_) abort();
            throw ProtocolAbort("session aborted");
        }
    }

    // Phase under which preprocessing calls are booked: RealPrep books its own
    // traffic, so the book is switched off; the ideal dealer's time goes to "prep".
    int prep_phase() { return prep_->ideal() ? book_->id(phase_tag_ + "prep") : PhaseBook::NONE; }

    void fill_pool(size_t n) {
        prep::Triples fresh = prep_->and_triples(n);
        if (fresh.a.size() != n || fresh.b.size() != n || fresh.c.size() != n || fresh.a.macs.size() != n ||
            fresh.b.macs.size() != n || fresh.c.macs.size() != n)
            throw std::runtime_error("the preprocessing returned the wrong number of triples");
        triple_requests_++;
        triples_requested_ += n;
        if (triples_available() == 0) {
            pool_ = std::move(fresh);
        } else {
            prep::Triples merged;
            merged.a = pool_.a.slice(pool_pos_, pool_.size());
            merged.b = pool_.b.slice(pool_pos_, pool_.size());
            merged.c = pool_.c.slice(pool_pos_, pool_.size());
            merged.a.append(fresh.a);
            merged.b.append(fresh.b);
            merged.c.append(fresh.c);
            pool_ = std::move(merged);
        }
        pool_pos_ = 0;
    }

    // Reserve n triples of the pool for the caller; returns the index of the first one.
    size_t take_triples(size_t n) {
        if (n > triples_available()) {
            // eval_circuit has checked this before any message; kept as a second line of defence.
            if (!allow_refill_) throw TripleBudgetError(n, triples_available());
            fill_pool(n - triples_available());
        }
        const size_t first = pool_pos_;
        pool_pos_ += n;
        triples_used_ += n;
        return first;
    }

    static Layout layout(const Circuit& c) {
        Layout lay;
        for (const circuit::Decl& d : c.inputs) {
            if (d.owner == circuit::IN_BOUND) {
                lay.bound_fields.push_back(Field{d.label, 0, d.wires.size(), &d});
            } else {
                const int owner = (int)d.owner;
                lay.in_fields[owner].push_back(Field{d.label, lay.in_wires[owner].size(), d.wires.size(), &d});
                lay.in_wires[owner].insert(lay.in_wires[owner].end(), d.wires.begin(), d.wires.end());
            }
        }
        std::vector<uint32_t> open[3];  // public, P_0, P_1
        for (const circuit::Decl& d : c.outputs) {
            if (d.owner == circuit::OUT_RETAIN) {
                lay.out_retain.push_back(Field{d.label, 0, d.wires.size(), &d});
            } else if (d.owner == circuit::OUT_PUBLIC) {
                lay.out_public.push_back(Field{d.label, open[0].size(), d.wires.size(), &d});
                open[0].insert(open[0].end(), d.wires.begin(), d.wires.end());
            } else {
                const int owner = (int)d.owner;
                lay.out_private[owner].push_back(Field{d.label, open[1 + owner].size(), d.wires.size(), &d});
                open[1 + owner].insert(open[1 + owner].end(), d.wires.begin(), d.wires.end());
            }
        }
        lay.n_public = open[0].size();
        lay.n_private[0] = open[1].size();
        lay.n_private[1] = open[2].size();
        for (Field& f : lay.out_private[0]) f.offset += lay.n_public;
        for (Field& f : lay.out_private[1]) f.offset += lay.n_public + lay.n_private[0];
        for (int i = 0; i < 3; i++) lay.open_wires.insert(lay.open_wires.end(), open[i].begin(), open[i].end());
        return lay;
    }

    // This party's input bits, (own input wires x batch) row-major.
    BitVec own_inputs(const Layout& lay, const Inputs& given, size_t batch) const {
        const std::vector<Field>& fields = lay.in_fields[role_];
        BitVec out(lay.n_in(role_) * batch);
        size_t used = 0;
        for (const Field& f : fields) {
            const Inputs::const_iterator it = given.find(f.label);
            if (it == given.end())
                throw std::invalid_argument("missing input " + f.label + " of party " + std::to_string(role_));
            const BitMat& m = it->second;
            if (m.width != f.width || (m.batch != 1 && m.batch != batch))
                throw std::invalid_argument("input " + f.label + ": wrong shape");
            m.validate();
            for (size_t q = 0; q < f.width; q++)
                for (size_t j = 0; j < batch; j++)
                    out[(f.offset + q) * batch + j] = m.v[q * m.batch + (m.batch == 1 ? 0 : j)];
            used++;
        }
        if (used != given.size()) throw std::invalid_argument("unknown input labels for party " + std::to_string(role_));
        return out;
    }

    // Validate the BOUND inputs and hash their description for the header.
    hashes::hash32 check_bound(const Layout& lay, const Bound& bound, size_t batch) const {
        std::vector<bytes> store;
        store.reserve(3 * lay.bound_fields.size());
        for (const Field& f : lay.bound_fields) {
            const Bound::const_iterator it = bound.find(f.label);
            if (it == bound.end()) throw std::invalid_argument("missing BOUND input " + f.label);
            const WireShares& ws = it->second;
            ws.validate();
            if (ws.token != token_)
                throw std::invalid_argument("BOUND input " + f.label + ": the wires belong to another session");
            if (ws.width != f.width) throw std::invalid_argument("BOUND input " + f.label + ": wrong number of wires");
            if (ws.batch != 1 && ws.batch != batch)
                throw std::invalid_argument("BOUND input " + f.label + ": batch must be 1 or the evaluation batch");
            store.push_back(bytes(f.label.begin(), f.label.end()));
            bytes ids(8 * ws.width);
            for (size_t i = 0; i < ws.width; i++) ascon::store64le(ids.data() + 8 * i, ws.ids[i]);
            store.push_back(std::move(ids));
            const std::array<uint8_t, 8> b = hashes::le64((uint64_t)ws.batch);
            store.push_back(bytes(b.begin(), b.end()));
        }
        if (bound.size() != lay.bound_fields.size()) throw std::invalid_argument("unknown BOUND labels");
        std::vector<hashes::Part> parts;
        for (const bytes& b : store) parts.push_back(hashes::Part(b));
        return hashes::framed_sha256(parts);
    }

    void check(const uint8_t* opened, const gf128* macs, size_t n) {
        const uint64_t counter = checks_++;
        prep::bool_check(*ch_, sid_abb_, counter, role_, delta_, opened, macs, n, std::vector<prep::FieldOpening>(),
                         sys_);
    }

    static double seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    }

    EvalResult run(const Circuit& c, const Layout& lay, const BitVec& x_own, const Bound& bound,
                   const hashes::hash32& bound_desc, size_t batch) {
        typedef std::chrono::steady_clock clock;
        const int role = role_, peer = 1 - role_;
        const size_t B = batch;
        const clock::time_point t_begin = clock::now();
        const ChannelStats st0 = ch_->stats();
        const uint64_t eval_index = evaluations_++;
        const gf128 delta = delta_;
        const uint8_t kbit = role == 0 ? 1 : 0;

        // 1. Preprocessed material: triples, then the input masks of P_0 and of P_1.
        book_->mark(prep_phase());
        const size_t n_and = c.n_and();
        size_t t_base = 0;
        if (n_and) t_base = take_triples(n_and * B);
        prep::InputMasks masks[2];
        for (int owner = 0; owner < 2; owner++) {
            const size_t count = lay.n_masks(owner) * B;
            if (!count) continue;
            masks[owner] = prep_->input_masks(owner, count);
            if (masks[owner].shares.bits.size() != count || masks[owner].shares.macs.size() != count ||
                (owner == role && (!masks[owner].has_bits || masks[owner].bits.size() != count)))
                throw std::runtime_error("the preprocessing returned the wrong number of input masks");
        }
        const size_t n_wires = c.n_wires;
        GfVec wm(n_wires * B);
        BitVec wb(n_wires * B, 0);
        for (size_t j = 0; j < B; j++) {
            wm[circuit::WIRE_ONE * B + j] = delta;
            wb[circuit::WIRE_ONE * B + j] = kbit;
        }
        const clock::time_point t_prep = clock::now();

        // 2. Input: one simultaneous exchange (header || d = x XOR r for the own inputs).
        book_->mark(phase_tag_ + "input");
        const std::array<uint8_t, 8> le_eval = hashes::le64(eval_index), le_batch = hashes::le64((uint64_t)B);
        const hashes::hash32 header =
            hashes::framed_sha256({"Couplet-abb-eval", sid_abb_, le_eval, c.digest, le_batch, bound_desc});
        const size_t n_in[2] = {lay.n_in(0), lay.n_in(1)};
        BitVec d_in[2];
        d_in[role].resize(n_in[role] * B);
        for (size_t i = 0; i < d_in[role].size(); i++) d_in[role][i] = (uint8_t)(x_own[i] ^ masks[role].bits[i]);
        {
            bytes msg(header.begin(), header.end());
            const bytes packed = prep::pack_bit_msg(d_in[role]);
            msg.insert(msg.end(), packed.begin(), packed.end());
            // A reply longer than this circuit prescribes can only belong to another circuit or batch.
            const bytes reply = ch_->exchange(msg, header.size() + (n_in[peer] * B + 7) / 8,
                                              "evaluation header mismatch (circuit digest, batch or bound wires): "
                                              "the input message is too long");
            if (reply.size() < header.size() || !hashes::ct_equal(reply.data(), header.data(), header.size()))
                throw ProtocolAbort("evaluation header mismatch (circuit digest, batch or bound wires)");
            d_in[peer] = prep::unpack_bit_msg(bytes(reply.begin() + (ptrdiff_t)header.size(), reply.end()), n_in[peer] * B);
        }
        for (int owner = 0; owner < 2; owner++) {
            const std::vector<uint32_t>& wires = lay.in_wires[owner];
            const prep::Shares& r = masks[owner].shares;
            for (size_t q = 0; q < wires.size(); q++) {
                const size_t row = (size_t)wires[q] * B;
                for (size_t j = 0; j < B; j++) {
                    const size_t i = q * B + j;
                    const uint8_t d = d_in[owner][i];
                    const uint64_t m = 0 - (uint64_t)d;
                    wm[row + j] = gf128(r.macs[i].lo ^ (delta.lo & m), r.macs[i].hi ^ (delta.hi & m));
                    wb[row + j] = (uint8_t)(r.bits[i] ^ (d & kbit));
                }
            }
        }
        for (const Field& f : lay.bound_fields) {
            const WireShares& ws = bound.find(f.label)->second;
            for (size_t q = 0; q < f.width; q++) {
                const size_t row = (size_t)f.decl->wires[q] * B;
                for (size_t j = 0; j < B; j++) {
                    const size_t src = q * ws.batch + (ws.batch == 1 ? 0 : j);
                    wm[row + j] = ws.macs[src];
                    wb[row + j] = ws.bits[src];
                }
            }
        }
        const clock::time_point t_input = clock::now();

        // 3. Gates: linear gates locally, one exchange per AND layer.
        book_->mark(phase_tag_ + "online");
        const size_t n_checked = 2 * n_and * B;
        BitVec chk_bits(n_checked);
        GfVec chk_macs(n_checked);
        const uint8_t* ta_b = pool_.a.bits.data();
        const uint8_t* tb_b = pool_.b.bits.data();
        const uint8_t* tc_b = pool_.c.bits.data();
        const gf128* ta_m = pool_.a.macs.data();
        const gf128* tb_m = pool_.b.macs.data();
        const gf128* tc_m = pool_.c.macs.data();
        const size_t base = c.first_gate_wire();
        const uint8_t* op = c.op.data();
        const uint32_t* ga = c.a.data();
        const uint32_t* gb = c.b.data();
        gf128* wmp = wm.data();
        uint8_t* wbp = wb.data();
        size_t pos = 0, t0 = 0;
        double wait = 0.0;
        for (const circuit::Step& step : c.schedule()) {
            if (B == 1) {
                for (size_t g = step.lin_begin; g < step.lin_end; g++) {
                    const size_t x = ga[g], y = op[g] == circuit::OP_NOT ? circuit::WIRE_ONE : gb[g], o = base + g;
                    wmp[o].lo = wmp[x].lo ^ wmp[y].lo;
                    wmp[o].hi = wmp[x].hi ^ wmp[y].hi;
                    wbp[o] = (uint8_t)(wbp[x] ^ wbp[y]);
                }
            } else {
                for (size_t g = step.lin_begin; g < step.lin_end; g++) {
                    const size_t x = (size_t)ga[g] * B, o = (base + g) * B;
                    const size_t y = (size_t)(op[g] == circuit::OP_NOT ? circuit::WIRE_ONE : gb[g]) * B;
                    for (size_t j = 0; j < B; j++) {
                        wmp[o + j].lo = wmp[x + j].lo ^ wmp[y + j].lo;
                        wmp[o + j].hi = wmp[x + j].hi ^ wmp[y + j].hi;
                    }
                    for (size_t j = 0; j < B; j++) wbp[o + j] = (uint8_t)(wbp[x + j] ^ wbp[y + j]);
                }
            }
            const size_t m = step.and_end - step.and_begin;
            if (!m) continue;
            const size_t mb = m * B;
            uint8_t* cb = chk_bits.data() + pos;
            gf128* cm = chk_macs.data() + pos;
            for (size_t g = 0; g < m; g++) {
                const size_t gi = step.and_begin + g;
                const size_t x = (size_t)ga[gi] * B, y = (size_t)gb[gi] * B, t = t_base + (t0 + g) * B;
                for (size_t j = 0; j < B; j++) {  // <d> = <x> ^ <a>, <e> = <y> ^ <b>
                    cb[g * B + j] = (uint8_t)(wbp[x + j] ^ ta_b[t + j]);
                    cm[g * B + j] = gf128(wmp[x + j].lo ^ ta_m[t + j].lo, wmp[x + j].hi ^ ta_m[t + j].hi);
                    cb[mb + g * B + j] = (uint8_t)(wbp[y + j] ^ tb_b[t + j]);
                    cm[mb + g * B + j] = gf128(wmp[y + j].lo ^ tb_m[t + j].lo, wmp[y + j].hi ^ tb_m[t + j].hi);
                }
            }
            const clock::time_point t_x = clock::now();
            const bytes reply = ch_->exchange(prep::pack_bit_msg(cb, 2 * mb), (2 * mb + 7) / 8, "bit message has the wrong length");
            wait += seconds(t_x, clock::now());
            const BitVec theirs = prep::unpack_bit_msg(reply, 2 * mb);
            for (size_t i = 0; i < 2 * mb; i++) cb[i] ^= theirs[i];
            for (size_t g = 0; g < m; g++) {
                const size_t o = (base + step.and_begin + g) * B, t = t_base + (t0 + g) * B;
                for (size_t j = 0; j < B; j++) {
                    const uint8_t d = cb[g * B + j], e = cb[mb + g * B + j];
                    const uint64_t dm = 0 - (uint64_t)d, em = 0 - (uint64_t)e, dem = dm & em;
                    wmp[o + j].lo = tc_m[t + j].lo ^ (tb_m[t + j].lo & dm) ^ (ta_m[t + j].lo & em) ^ (delta.lo & dem);
                    wmp[o + j].hi = tc_m[t + j].hi ^ (tb_m[t + j].hi & dm) ^ (ta_m[t + j].hi & em) ^ (delta.hi & dem);
                    wbp[o + j] = (uint8_t)(tc_b[t + j] ^ (tb_b[t + j] & d) ^ (ta_b[t + j] & e) ^ (kbit & d & e));
                }
            }
            pos += 2 * mb;
            t0 += m;
        }
        // Every triple reserved for this evaluation was consumed by exactly one gate instance.
        if (t0 != n_and || pos != n_checked) throw std::logic_error("engine: triple accounting is inconsistent");
        if (triples_available() == 0) {  // the pool is used up: release it
            pool_ = prep::Triples();
            pool_pos_ = 0;
        }
        const clock::time_point t_online = clock::now();

        // 4. Output (section 2.2, change F).
        book_->mark(phase_tag_ + "outcheck");
        if (n_checked) check(chk_bits.data(), chk_macs.data(), n_checked);  // (1) every d, e of the evaluation
        chk_bits = BitVec();
        chk_macs = GfVec();
        const clock::time_point t_check1 = clock::now();
        EvalResult res;
        Transcript tr;
        const size_t n_open = lay.open_wires.size();
        if (n_open) {
            book_->mark(phase_tag_ + "output");
            BitVec ob(n_open * B);
            GfVec om(n_open * B);
            for (size_t q = 0; q < n_open; q++) {
                const size_t row = (size_t)lay.open_wires[q] * B;
                for (size_t j = 0; j < B; j++) {
                    ob[q * B + j] = wb[row + j];
                    om[q * B + j] = wm[row + j];
                }
            }
            size_t off = lay.n_public;
            for (int owner = 0; owner < 2; owner++) {  // private wires: <y> XOR <r>
                const prep::Shares& r = masks[owner].shares;
                for (size_t q = 0; q < lay.n_private[owner]; q++) {
                    for (size_t j = 0; j < B; j++) {
                        const size_t i = (n_in[owner] + q) * B + j;
                        ob[(off + q) * B + j] ^= r.bits[i];
                        om[(off + q) * B + j] ^= r.macs[i];
                    }
                }
                off += lay.n_private[owner];
            }
            const BitVec theirs = prep::exchange_bits(*ch_, ob);  // (2) open
            for (size_t i = 0; i < ob.size(); i++) ob[i] ^= theirs[i];
            book_->mark(phase_tag_ + "outcheck");
            check(ob.data(), om.data(), ob.size());  // (3) exactly these openings
            // (4) accepted
            for (const Field& f : lay.out_public) res.pub.emplace(f.label, take_rows(ob, f.offset, f.width, B));
            for (int owner = 0; owner < 2; owner++)
                for (const Field& f : lay.out_private[owner]) tr.masked.emplace(f.label, take_rows(ob, f.offset, f.width, B));
            const size_t start = lay.n_public + (role == 1 ? lay.n_private[0] : 0);
            for (const Field& f : lay.out_private[role]) {
                BitMat m = take_rows(ob, f.offset, f.width, B);
                for (size_t q = 0; q < f.width; q++)
                    for (size_t j = 0; j < B; j++)
                        m.v[q * B + j] ^= masks[role].bits[(n_in[role] + f.offset - start + q) * B + j];
                res.priv.emplace(f.label, std::move(m));
            }
        }
        for (const Field& f : lay.out_retain) {
            WireShares ws;
            ws.width = f.width;
            ws.batch = B;
            ws.token = token_;
            ws.bits.resize(f.width * B);
            ws.macs.resize(f.width * B);
            ws.ids.resize(f.width);
            for (size_t q = 0; q < f.width; q++) {
                const size_t row = (size_t)f.decl->wires[q] * B;
                ws.ids[q] = next_wire_id_++;
                for (size_t j = 0; j < B; j++) {
                    ws.bits[q * B + j] = wb[row + j];
                    ws.macs[q * B + j] = wm[row + j];
                }
            }
            res.retained.emplace(f.label, std::move(ws));
        }
        const clock::time_point t_end = clock::now();

        EvalStats& s = res.stats;
        s.evaluation = eval_index;
        s.batch = B;
        s.n_gates = c.n_gates();
        s.n_and = n_and;
        s.and_depth = c.and_depth();
        s.triples = n_and * B;
        s.input_masks[0] = lay.n_masks(0) * B;
        s.input_masks[1] = lay.n_masks(1) * B;
        s.opened_bits = n_open * B;
        s.checked_bits = n_checked + n_open * B;
        const ChannelStats st1 = ch_->stats();
        s.rounds = st1.rounds - st0.rounds;
        s.bytes_sent = st1.bytes_sent - st0.bytes_sent;
        s.triple_requests = triple_requests_;
        s.triples_requested = triples_requested_;
        s.triples_available = triples_available();
        s.prep_s = seconds(t_begin, t_prep);
        s.input_s = seconds(t_prep, t_input);
        s.online_s = seconds(t_input, t_online);
        s.online_exchange_wait_s = wait;
        s.check_s = seconds(t_online, t_check1);
        s.output_s = seconds(t_check1, t_end);
        s.total_s = seconds(t_begin, t_end);
        last_stats_ = s;
        tr.input_diffs[0] = std::move(d_in[0]);
        tr.input_diffs[1] = std::move(d_in[1]);
        transcript_ = std::move(tr);
        return res;
    }

    static BitMat take_rows(const BitVec& all, size_t offset, size_t width, size_t batch) {
        BitMat m(width, batch);
        std::memcpy(m.v.data(), all.data() + offset * batch, width * batch);
        return m;
    }

    int role_;
    Channel* ch_;
    prep::Prep* prep_;
    bytes sid_, sid_abb_;
    PhaseBook own_book_;
    PhaseBook* book_;
    SystemRng sys_;
    std::string phase_tag_;
    gf128 delta_;
    bool dead_ = false;
    uint64_t token_ = 0;
    uint64_t evaluations_ = 0, triple_requests_ = 0, triples_requested_ = 0, triples_used_ = 0, checks_ = 0,
             next_wire_id_ = 0;
    bool allow_refill_ = true;
    prep::Triples pool_;
    size_t pool_pos_ = 0;
    EvalStats last_stats_;
    Transcript transcript_;
};

}  // namespace abb
}  // namespace couplet
