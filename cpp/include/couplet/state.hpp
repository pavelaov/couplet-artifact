// couplet/state.hpp -- the state machine of DESIGN.md section 5.4 and the
// party driver of Pi_AuthEval built on it (sections 5.3, 6; changes D - G);
// the C++ counterpart of python/couplet/state.py and python/couplet/autheval.py.
//
// ---------------------------------------------------------------- Registry
// Registry(sid, path) is the bookkeeping of ONE party for one session:
//
//     session   ACTIVE -> ABORTED                 (absorbing, change G)
//     key       GENERATING -> CERTIFIED           (CERTIFIED only after the
//                                                  output MAC check of Pi_Gen)
//     gate      GENERATING -> CERTIFIED -> USED   (one Mask per gate)
//
// Rules enforced here, all BEFORE any message is sent (a refused call throws
// StateError and leaves the transcript untouched):
//
//   * identifiers kid, gid, qid are monotone: an identifier is accepted only
//     if it is not smaller than the next free one of its kind, in this
//     process or (with persistence) in an earlier one;
//   * a key is evaluated only when it is CERTIFIED;
//   * use_gate moves a gate from CERTIFIED to USED exactly once.  The caller
//     invokes it before the first byte of the masked opening is sent, so one
//     offset masks one input even if the opening fails, the peer disappears,
//     the session aborts or the operation is retried.  A second call throws
//     GateUsedError;
//   * after abort() every method throws ProtocolAbort and the key and gate
//     material is dropped.
//
// Persistence (optional).  With `path` the counters are kept in a small JSON
// file {"format", "reserved": {"gid", "kid", "qid"}, "sid", "starts",
// "state"} in the format of the Python module.  The file stores, per kind, a
// RESERVED bound: an identifier is issued only below the stored bound, and
// the bound is raised (in blocks) and written to disk before the identifiers
// of the block are used.  The file is replaced atomically (temporary file,
// flush, sync, rename).  A process that starts on an existing file continues
// ABOVE the stored bounds, so it can never issue an identifier again; every
// gate of the earlier process, used or not, is below the new floor and can no
// longer be masked (a reconnect never reuses an offset).  A file that records
// ABORTED keeps the session aborted; a file of another session or an
// unreadable file is refused instead of being overwritten.
//
// ABORTED on disk (design/CHANGELOG.md F2 item 4).  abort() makes the state
// durable in two independent ways: (1) the marker file `path + ".aborted"`,
// created with create-only semantics (O_CREAT | O_EXCL: it is never replaced,
// renamed or removed by this class) and containing
// {"format": "couplet-aborted-1", "sid": "<hex>"}; (2) the state file itself,
// rewritten with "state": "ABORTED" as before.  load() treats the session as
// ABORTED if EITHER says so (a marker that exists but cannot be parsed counts
// as a marker; a marker of another session is refused like a state file of
// another session).  So a state file that could not be replaced at abort
// time no longer lets a later process start ACTIVE.  If neither write
// succeeds, the in-memory state still protects this process, abort() returns
// false and abort_persist_failed() / abort_persist_error() report it.
// On Windows the replacement of the state file is retried for about half a
// second while it fails because another process holds the file (sharing
// violation, access denied, lock violation): without the retry a write of
// the counters failed now and then in a directory under the user's temporary
// directory (20 of 10,000 test rounds of seven replacements each; presumably
// a scanner or indexer that opens a file right after it was written), with
// it in none of 20,000 rounds (design/CHANGELOG.md F2 item 4).
//
// ---------------------------------------------------------------- Party
// Party(session) runs ONE party of Pi_AuthEval on a Session; the other party
// issues the same commands with the same public arguments and its own shares.
//
//     setup(reserve_triples, reserve_outmasks)
//                                  Session setup and the ArithChecker; all
//                                  triples of the session in ONE request.
//     key / key_batch              Key: Pi_Gen for DPF / DCF keys; the record
//                                  becomes CERTIFIED when Pi_Gen returns.
//     query / query_all            Query at public points: local Eval; exact.
//     gate / gate_batch            Gate: Pi_Gen for one-time gate material.
//     mask_shared(gid, x_share)    Mask of a shared input x = x_0 + x_1 mod
//                                  2^n: the parties exchange x_b + r_b mod 2^n
//                                  by commit-then-open (two rounds); no MAC (a
//                                  wrong share is an input substitution).
//     mask_wire(gid, share)        Mask of a wire: [[x]] is reduced to
//                                  Z_{2^(n+s)}, the offset [[r']] (uniform over
//                                  that whole ring, change D) is added and the
//                                  sum is opened with open_deferred; the MAC
//                                  check is deferred to the next open.
//     apply(gid)                   Apply: local evaluation of the gate on x_hat.
//     lin(terms, const)            Lin: public affine combination.
//     open / open_many             Open, in the order of the protocol figure of
//                                  the paper: (1) ArithCheck on every deferred
//                                  opening; (2) the output mask [[2^k rho]] of a
//                                  sharing that is not exact is taken from the
//                                  pool (change E), and if the pool is short the
//                                  missing masks are generated by Pi_Gen at this
//                                  point, i.e. only AFTER the deferred openings
//                                  have been verified; (3) the opening of the
//                                  value; (4) ArithCheck on that opening;
//                                  returns v mod 2^k.  In strict mode a mask
//                                  generation that the reserved pool cannot
//                                  serve is refused before step (1), without
//                                  any message.
//     abort()                      Abort: the session is retired.
//
// One-time use (section 5.4).  mask_shared / mask_wire validate their
// arguments, then mark the gate USED in the registry and only then send the
// masked opening.  Output masks are consumed the same way: each is taken
// from a pool exactly once.  Masks come from gates generated with
// n_outmask > 0 and otherwise from Pi_Gen on demand: reserve_outmasks(count)
// runs the circuit "gate_n1_k<k>_s<s>_shared_om1" with batch = count (the
// reference does the same since CHANGELOG F1).
//
// Shared-input Mask on the wire (DESIGN.md v3.2 V5; CHANGELOG F1 W1), byte
// for byte as the reference.  With b the sender's party index and gid the
// gate identifier:
//
//     sid_mask = SHA-256(frame("Couplet-mask-sid", sid))
//     label    = "mask|" || le64(gid)                       (13 bytes)
//     msg_b    = (x_b + r_b mod 2^n) as ceil(n/8) bytes little-endian
//     com_b    = SHA-256(frame("Couplet-com", sid_mask, label, byte(b), omega_b, msg_b))
//     round 1: com_b (32 bytes);   round 2: omega_b || msg_b (32 + ceil(n/8) bytes)
//
// i.e. hashes::committed_exchange under the sub-session identifier of the
// Mask command.  A rushing party can therefore not choose its share, and
// with it x_hat, after it has seen the other share.  The receiver aborts on a
// malformed, mirrored or wrongly opened commitment, on a message that is not
// exactly ceil(n/8) bytes and on a value >= 2^n.  Both messages are booked
// under the phase "mask".  The honest code of the exchange is the protected
// virtual mask_exchange; a deviating party of the tests overrides it.
//
// Triple budget (strict mode; DESIGN.md v3.2 V6; CHANGELOG F1 item 1, F2 item
// 2).  The bucketing bound 2^-sigma holds per preprocessing request for AND
// triples (Lemma 4), so the driver is strict by default: Party(session)
// switches the session to allow_refill = false; all triples of the session
// are requested once, in setup, and a Key / Gate / output-mask generation
// that does not fit into the reserved pool throws abb::TripleBudgetError (a
// std::invalid_argument) before any message and before any change of state:
// no identifier is consumed, no record is created, the session stays ACTIVE.
// An invalid or non-monotone identifier is still reported first, by the
// registry.  The budget is public and both parties compute the same number:
//
//     reserve_triples  = sum of key_budget(...) and gate_budget(...) over the
//                        planned commands (protocol::and_budget for lists);
//     reserve_outmasks = number of non-exact values that will be opened and
//                        whose mask does not come from a gate generated with
//                        n_outmask > 0.
//
// triple_requests() is the number of populations drawn; a session that
// reserved exactly has triple_requests() == 1 and no unused triple at its
// end.  Party(session, path, true) keeps the refill mode: a short pool is
// refilled silently and the count increases.
//
// Abort (change G).  Any ProtocolAbort and any other failure inside a
// protocol step moves the driver, the session, the checker and the registry
// to ABORTED; the MAC key shares and all key and gate material are dropped
// and every later command throws ProtocolAbort without sending anything.
// Argument errors (std::invalid_argument) and state errors (StateError) are
// raised before any message and do not abort.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(_WIN32)
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif

#include "abb.hpp"
#include "channel.hpp"
#include "circuit.hpp"
#include "fss.hpp"
#include "gates.hpp"
#include "hashes.hpp"
#include "protocol.hpp"
#include "session.hpp"
#include "spdz2k.hpp"

namespace couplet {
namespace state {

// A command is not allowed in the current state; nothing was sent.
struct StateError : std::runtime_error {
    explicit StateError(const std::string& what) : std::runtime_error(what) {}
};
// A second Mask on a gate whose offset has already been used.
struct GateUsedError : StateError {
    explicit GateUsedError(const std::string& what) : StateError(what) {}
};

enum RecordState { GENERATING, CERTIFIED, USED };

inline const char* record_state_name(RecordState s) {
    return s == GENERATING ? "GENERATING" : s == CERTIFIED ? "CERTIFIED" : "USED";
}

// One key: GENERATING or CERTIFIED; the key is set when it is certified.
struct KeyRecord {
    uint64_t kid = 0;
    RecordState state = GENERATING;
    protocol::Family family = protocol::FAMILY_DPF;
    unsigned n = 0;
    protocol::BetaMode beta_mode = protocol::BETA_PUBLIC_ONE;
    bool has_key = false;
    fss::DpfKey dpf;
    fss::DcfKey dcf;
};

// One gate: GENERATING, CERTIFIED or USED.  The evaluator (which holds the
// material) and the specification are set when the gate is certified; x_hat
// (public), mode ("shared" or "wire") and opened (the public opened
// representative in wire mode) once Mask has completed.  A gate that is USED
// without x_hat had a failed Mask and can be neither masked again nor applied.
struct GateRecord {
    uint64_t gid = 0;
    RecordState state = GENERATING;
    std::shared_ptr<const gates::GateSpec> spec;
    std::shared_ptr<gates::GateEval> evaluator;
    bool has_x_hat = false;
    uint64_t x_hat = 0;
    std::string mode;
    u128 opened = u128(0);

    bool has_material() const { return (bool)evaluator; }
    const protocol::GateMaterial& material() const { return evaluator->material(); }
};

class Registry {
public:
    static const uint64_t NEXT = ~(uint64_t)0;  // "the next free identifier"
    enum Kind { KID = 0, GID = 1, QID = 2 };

    struct Counters {
        uint64_t next[3], floor[3], reserved[3];
        uint64_t starts;
        bool active;
    };
    struct Summary {
        bool active;
        size_t keys_generating = 0, keys_certified = 0;
        size_t gates_generating = 0, gates_certified = 0, gates_used = 0;
    };

    explicit Registry(const bytes& sid, const std::string& path = std::string()) : sid_(sid), path_(path) {
        for (int i = 0; i < 3; i++) next_[i] = floor_[i] = reserved_[i] = 0;
        if (!path_.empty()) load();
    }
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    bool active() const { return active_; }
    uint64_t starts() const { return starts_; }  // processes that have used the state file

    void require_active() const {
        if (!active_) throw ProtocolAbort("session aborted");
    }

    // ACTIVE -> ABORTED (absorbing); the key and gate material is dropped.
    // The state is then made durable (marker file and state file, see the
    // header comment).  Returns false iff NEITHER could be written: this
    // process is protected by the in-memory state, but a later process on the
    // same file would start ACTIVE; abort_persist_error() has the reason.
    // Never throws.
    bool abort() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        const bool already = !active_;
        active_ = false;
        for (std::pair<const uint64_t, KeyRecord>& r : keys_) {
            r.second.has_key = false;
            r.second.dpf = fss::DpfKey();
            r.second.dcf = fss::DcfKey();
        }
        for (std::pair<const uint64_t, GateRecord>& r : gates_) r.second.evaluator.reset();
        if (!already) persist_abort();
        return !persist_failed_;
    }
    // True iff an abort could not be written to disk at all (see abort()).
    bool abort_persist_failed() const { return persist_failed_; }
    // What went wrong while writing the ABORTED state (empty if nothing did;
    // non-empty with abort_persist_failed() == false if one of the two writes
    // failed and the other succeeded).
    const std::string& abort_persist_error() const { return persist_error_; }
    // The create-only marker file of an aborted session (empty without persistence).
    std::string abort_marker_path() const { return path_.empty() ? std::string() : path_ + ".aborted"; }

    // A fresh query identifier.
    uint64_t next_qid() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        return issue(QID, NEXT, 1);
    }

    Counters counters() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        Counters c;
        for (int i = 0; i < 3; i++) {
            c.next[i] = next_[i];
            c.floor[i] = floor_[i];
            c.reserved[i] = reserved_[i];
        }
        c.starts = starts_;
        c.active = active_;
        return c;
    }

    // -- keys
    // Create `count` key records in state GENERATING; returns the first identifier.
    uint64_t begin_key(uint64_t kid = NEXT, size_t count = 1) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        const uint64_t first = issue(KID, kid, count);
        for (uint64_t i = first; i < first + count; i++) keys_[i].kid = i;
        return first;
    }
    // GENERATING -> CERTIFIED: called when Pi_Gen has returned (its output check passed).
    KeyRecord& certify_key(uint64_t kid, const KeyRecord& filled) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        require_active();
        const std::map<uint64_t, KeyRecord>::iterator it = keys_.find(kid);
        if (it == keys_.end() || it->second.state != GENERATING)
            throw StateError("key " + std::to_string(kid) + " is not being generated");
        it->second = filled;
        it->second.kid = kid;
        it->second.has_key = true;
        it->second.state = CERTIFIED;
        return it->second;
    }
    // The record of a CERTIFIED key.
    const KeyRecord& key(uint64_t kid) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        require_active();
        const std::map<uint64_t, KeyRecord>::const_iterator it = keys_.find(kid);
        if (it == keys_.end()) throw StateError("unknown key " + std::to_string(kid));
        if (it->second.state != CERTIFIED) throw StateError("key " + std::to_string(kid) + " is not certified");
        return it->second;
    }

    // -- gates
    uint64_t begin_gate(uint64_t gid = NEXT, size_t count = 1) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        const uint64_t first = issue(GID, gid, count);
        for (uint64_t i = first; i < first + count; i++) gates_[i].gid = i;
        return first;
    }
    // GENERATING -> CERTIFIED.
    GateRecord& certify_gate(uint64_t gid, std::shared_ptr<gates::GateEval> evaluator,
                             std::shared_ptr<const gates::GateSpec> spec = std::shared_ptr<const gates::GateSpec>()) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        require_active();
        const std::map<uint64_t, GateRecord>::iterator it = gates_.find(gid);
        if (it == gates_.end() || it->second.state != GENERATING)
            throw StateError("gate " + std::to_string(gid) + " is not being generated");
        if (!evaluator) throw std::invalid_argument("a certified gate needs its material");
        it->second.evaluator = std::move(evaluator);
        it->second.spec = std::move(spec);
        it->second.state = CERTIFIED;
        return it->second;
    }
    // The record of a gate in any state (no transition).
    GateRecord& gate(uint64_t gid) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        require_active();
        const std::map<uint64_t, GateRecord>::iterator it = gates_.find(gid);
        if (it == gates_.end()) throw StateError("unknown gate " + std::to_string(gid));
        return it->second;
    }
    // CERTIFIED -> USED, exactly once; call it before the masked opening is sent.
    GateRecord& use_gate(uint64_t gid) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        GateRecord& rec = gate(gid);
        if (rec.state == USED) throw GateUsedError("gate " + std::to_string(gid) + ": the offset has already masked an input");
        if (rec.state != CERTIFIED) throw StateError("gate " + std::to_string(gid) + " is not certified");
        rec.state = USED;
        return rec;
    }
    // Record the public result of a completed Mask.
    GateRecord& set_masked(uint64_t gid, uint64_t x_hat, const std::string& mode, u128 opened = u128(0)) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        GateRecord& rec = gate(gid);
        if (rec.state != USED || rec.has_x_hat) throw StateError("gate " + std::to_string(gid) + ": no Mask in progress");
        rec.x_hat = x_hat;
        rec.has_x_hat = true;
        rec.mode = mode;
        rec.opened = opened;
        return rec;
    }
    // The record of a gate whose Mask has completed (for Apply).
    GateRecord& masked_gate(uint64_t gid) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        GateRecord& rec = gate(gid);
        if (rec.state != USED || !rec.has_x_hat) throw StateError("gate " + std::to_string(gid) + " has not been masked");
        return rec;
    }

    Summary summary() {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        Summary s;
        s.active = active_;
        for (const std::pair<const uint64_t, KeyRecord>& r : keys_)
            (r.second.state == GENERATING ? s.keys_generating : s.keys_certified)++;
        for (const std::pair<const uint64_t, GateRecord>& r : gates_)
            (r.second.state == GENERATING ? s.gates_generating : r.second.state == CERTIFIED ? s.gates_certified : s.gates_used)++;
        return s;
    }

private:
    static const char* kind_name(int kind) { return kind == KID ? "kid" : kind == GID ? "gid" : "qid"; }
    static uint64_t block(int kind) { return kind == QID ? 4096 : 64; }  // identifiers reserved per write of the file

    // Reserve `count` consecutive identifiers; returns the first one.
    uint64_t issue(int kind, uint64_t requested, size_t count) {
        if (count < 1) throw std::invalid_argument("count must be positive");
        require_active();
        const uint64_t nxt = next_[kind];
        uint64_t first = nxt;
        if (requested != NEXT) {
            first = requested;
            if (first < nxt)
                throw StateError(std::string(kind_name(kind)) + " " + std::to_string(first) + " is not monotone (" +
                                 (first < floor_[kind] ? "issued by an earlier process" : "already issued") +
                                 "); next is " + std::to_string(nxt));
        }
        if (first > (~(uint64_t)0 >> 1) || count > ((uint64_t)1 << 32)) throw std::invalid_argument("identifier out of range");
        const uint64_t end = first + count;
        if (end > reserved_[kind]) {
            const uint64_t before = reserved_[kind];
            reserved_[kind] = end + block(kind);
            try {
                store();  // on disk BEFORE any identifier of the block is used
            } catch (...) {
                reserved_[kind] = before;
                throw;
            }
        }
        next_[kind] = end;
        return first;
    }

    static bool is_regular_file(const std::string& p) {
#if defined(_WIN32)
        struct _stat64 st;
        return _stat64(p.c_str(), &st) == 0 && (st.st_mode & _S_IFMT) == _S_IFREG;
#else
        struct stat st;
        return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
    }

    // True iff the abort marker exists.  Its existence is the signal: a marker
    // that cannot be parsed counts (conservative); one that names another
    // session is refused, like a state file of another session.
    bool marker_present() const {
        const std::string mp = abort_marker_path();
        if (mp.empty() || !is_regular_file(mp)) return false;
        bool other = false;
        try {
            std::ifstream f(mp, std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            const json::Value doc = json::parse(text);
            other = doc.at("format").str() == "couplet-aborted-1" && circuit::unhex(doc.at("sid").str()) != sid_;
        } catch (...) {
        }
        if (other) throw std::invalid_argument("abort marker " + mp + " belongs to another session");
        return true;
    }

    // Create the abort marker.  Create-only: an existing marker is left as it
    // is and counts as written.  Throws std::runtime_error on failure.
    void write_marker() {
        const std::string mp = abort_marker_path();
        const std::string text = "{\"format\": \"couplet-aborted-1\", \"sid\": \"" + circuit::hex(sid_) + "\"}";
        int fd = -1;
#if defined(_WIN32)
        if (_sopen_s(&fd, mp.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE) != 0) fd = -1;
#else
        fd = ::open(mp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
#endif
        if (fd < 0) {
            if (is_regular_file(mp)) return;  // marked by an earlier abort
            throw std::runtime_error("cannot create the abort marker " + mp);
        }
        // From here on the marker exists, which is what load() looks at; the
        // content is informative, so a short write is reported but not undone.
#if defined(_WIN32)
        bool ok = _write(fd, text.data(), (unsigned)text.size()) == (int)text.size();
        ok = _commit(fd) == 0 && ok;
        ok = _close(fd) == 0 && ok;
#else
        bool ok = ::write(fd, text.data(), text.size()) == (ssize_t)text.size();
        ok = fsync(fd) == 0 && ok;
        ok = ::close(fd) == 0 && ok;
#endif
        if (!ok) throw std::runtime_error("the abort marker " + mp + " was created but could not be written completely");
    }

    // Make ABORTED durable: the create-only marker AND the state file.  The
    // abort is on disk if either write succeeds (load() honours both).
    void persist_abort() {
        if (path_.empty()) return;  // no persistence was asked for
        bool marker_ok = false, file_ok = false;
        std::string errors;
        try {
            write_marker();
            marker_ok = true;
        } catch (const std::exception& e) {
            errors = e.what();
            marker_ok = is_regular_file(abort_marker_path());  // created but not written completely: still a marker
        } catch (...) {
            errors = "cannot create the abort marker";
        }
        try {
            store();
            file_ok = true;
        } catch (const std::exception& e) {
            errors += (errors.empty() ? "" : "; ") + std::string(e.what());
        } catch (...) {
            errors += (errors.empty() ? "" : "; ") + std::string("cannot write the state file");
        }
        persist_error_ = errors;
        persist_failed_ = !marker_ok && !file_ok;
    }

    void load() {
        const bool marked = marker_present();  // an earlier process aborted, whatever the state file says
        std::ifstream f(path_, std::ios::binary);
        if (!f) {
            if (marked) active_ = false;
            store();
            return;
        }
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        f.close();
        bytes sid;
        uint64_t reserved[3] = {0, 0, 0}, starts = 0;
        bool active = true;
        try {
            const json::Value doc = json::parse(text);
            if (doc.at("format").str() != "couplet-state-1") throw std::invalid_argument("unknown format");
            sid = circuit::unhex(doc.at("sid").str());
            const json::Value& res = doc.at("reserved");
            for (int i = 0; i < 3; i++) reserved[i] = res.at(kind_name(i)).as_u64();
            const std::string& st = doc.at("state").str();
            if (st != "ACTIVE" && st != "ABORTED") throw std::invalid_argument("invalid content");
            active = st == "ACTIVE";
            starts = doc.at("starts").as_u64();
            if (starts < 1) throw std::invalid_argument("invalid content");
        } catch (const std::exception& e) {
            throw std::invalid_argument("state file " + path_ + " is unreadable; refusing to reset the counters (" +
                                        e.what() + ")");
        }
        if (sid != sid_) throw std::invalid_argument("state file " + path_ + " belongs to another session");
        for (int i = 0; i < 3; i++) next_[i] = floor_[i] = reserved_[i] = reserved[i];
        starts_ = starts + 1;
        active_ = active && !marked;
        store();
    }

    void store() {
        if (path_.empty()) return;
        const std::string text = std::string("{\"format\": \"couplet-state-1\", \"reserved\": {\"gid\": ") +
                                 std::to_string(reserved_[GID]) + ", \"kid\": " + std::to_string(reserved_[KID]) +
                                 ", \"qid\": " + std::to_string(reserved_[QID]) + "}, \"sid\": \"" + circuit::hex(sid_) +
                                 "\", \"starts\": " + std::to_string(starts_) + ", \"state\": \"" +
                                 (active_ ? "ACTIVE" : "ABORTED") + "\"}";
        const std::string tmp = path_ + ".tmp";
        std::FILE* fh = nullptr;
#if defined(_MSC_VER)
        if (!retry_while_shared([&] { return fopen_s(&fh, tmp.c_str(), "wb") == 0; })) fh = nullptr;
#else
        fh = std::fopen(tmp.c_str(), "wb");
#endif
        if (!fh) throw std::runtime_error("cannot write the state file " + tmp);
        bool ok = std::fwrite(text.data(), 1, text.size(), fh) == text.size();
        ok = std::fflush(fh) == 0 && ok;
#if defined(_WIN32)
        ok = _commit(_fileno(fh)) == 0 && ok;
#else
        ok = fsync(fileno(fh)) == 0 && ok;
#endif
        ok = std::fclose(fh) == 0 && ok;
        if (!ok) {
            std::remove(tmp.c_str());
            throw std::runtime_error("cannot write the state file " + tmp);
        }
#if defined(_WIN32)
        if (!retry_while_shared([&] {
                return MoveFileExA(tmp.c_str(), path_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
            })) {
#else
        if (std::rename(tmp.c_str(), path_.c_str()) != 0) {
#endif
            std::remove(tmp.c_str());
            throw std::runtime_error("cannot replace the state file " + path_);
        }
    }

#if defined(_WIN32)
    // Run `op` (a file operation that returns true on success) and repeat it
    // for a short while as long as it fails because another process holds the
    // file: 2, 4, ..., 64 ms between the attempts, about half a second in all.
    // Any other failure, and a file that stays in use, is reported at once /
    // after the last attempt.
    template <class Op>
    static bool retry_while_shared(Op&& op) {
        for (int attempt = 0;; attempt++) {
            if (op()) return true;
            const DWORD e = GetLastError();
            if ((e != ERROR_SHARING_VIOLATION && e != ERROR_ACCESS_DENIED && e != ERROR_LOCK_VIOLATION) || attempt >= 11) return false;
            Sleep(attempt < 5 ? (DWORD)(2u << attempt) : 64);
        }
    }
#endif

    bytes sid_;
    std::string path_;
    bool active_ = true;
    bool persist_failed_ = false;
    std::string persist_error_;
    uint64_t starts_ = 1;
    uint64_t next_[3], floor_[3], reserved_[3];
    std::map<uint64_t, KeyRecord> keys_;
    std::map<uint64_t, GateRecord> gates_;
    std::recursive_mutex mutex_;
};

}  // namespace state

// ------------------------------------------------------------------ the party driver of Pi_AuthEval
class Party {
public:
    typedef spdz2k::AShare AShare;
    typedef std::function<std::unique_ptr<spdz2k::ArithChecker>(int, Channel&, const bytes&, uint64_t, unsigned,
                                                                abb::PhaseBook*)>
        CheckerFactory;
    static const uint64_t NEXT = state::Registry::NEXT;

    // `session` must outlive the party.  `state_path` (optional) is the JSON
    // file of the registry counters.  `allow_refill` = false (the default) is
    // strict mode: the party switches the session to it, and a short triple
    // pool is abb::TripleBudgetError instead of a new bucketing population.
    explicit Party(Session& session, const std::string& state_path = std::string(), bool allow_refill = false)
        : session_(&session), registry_(session.sid(), state_path) {
        session_->set_allow_refill(allow_refill);
        const hashes::hash32 h = hashes::framed_sha256({"Couplet-mask-sid", session.sid()});
        sid_mask_.assign(h.begin(), h.end());  // sub-session of the Mask command
        if (!registry_.active()) abort();  // a persisted ABORTED session stays dead
    }
    virtual ~Party() {}
    Party(const Party&) = delete;
    Party& operator=(const Party&) = delete;

    Session& session() { return *session_; }
    state::Registry& registry() { return registry_; }
    spdz2k::ArithChecker* checker() { return checker_.get(); }
    int role() const { return session_->role(); }
    unsigned k() const { return session_->k(); }
    unsigned s() const { return session_->s(); }
    unsigned w() const { return session_->w(); }
    bool active() const { return registry_.active() && session_->active(); }
    size_t outmasks_available() const { return outmasks_.size(); }
    // The sub-session identifier of the Mask command: SHA-256(frame("Couplet-mask-sid", sid)).
    const bytes& sid_mask() const { return sid_mask_; }

    // False in strict mode (the default): a short triple pool is an error, not a refill.
    bool allow_refill() const { return session_->allow_refill(); }
    // Preprocessing requests for AND triples so far (bucketing populations; 1 if reserved).
    uint64_t triple_requests() const { return session_->triple_requests(); }

    // -- triple budget (public parameters only; no message)
    // AND triples of key / key_batch for `count` keys (secret_payload: a beta share is given).
    size_t key_budget(protocol::Family family, unsigned n, bool secret_payload = false,
                      protocol::IndexMode index_mode = protocol::INDEX_XOR, size_t count = 1) {
        return protocol::planned_key(session_->store(), family, n, k(), s(), index_mode,
                                     secret_payload ? protocol::BETA_SECRET : protocol::BETA_PUBLIC_ONE, count)
            .triples;
    }
    // AND triples of gate / gate_batch for `count` gates of `spec` (or of a descriptor).
    size_t gate_budget(const gates::GateSpec& spec, size_t count = 1) { return gate_budget(spec.descriptor(), count); }
    size_t gate_budget(const protocol::GateDescriptor& descriptor, size_t count = 1) {
        return protocol::planned_gate(session_->store(), descriptor, count).triples;
    }
    // AND triples of `count` on-demand output masks (what reserve_outmasks = count adds at setup).
    size_t outmask_budget(size_t count = 1) {
        return protocol::planned_outmasks(session_->store(), k(), s(), count).triples;
    }

    // Test hook: substitute the checker (e.g. a deviating subclass); before setup().
    void set_checker_factory(CheckerFactory f) { factory_ = std::move(f); }
    // Test hook: deterministic coins for Pi_Gen.
    void set_rng(RandomSource* rng) { rng_ = rng; }
    // Worker threads for full-domain key evaluations (table gates, query_all).
    void set_eval_threads(unsigned t) { eval_threads_ = t ? t : 1; }

    // Abort: retire the session (change G).  Returns false iff the ABORTED
    // state could not be written to the state file or its marker (see
    // state::Registry::abort); the party is retired in either case.
    bool abort() {
        outmasks_.clear();
        if (checker_) checker_->abort();  // its hook retires the session and the registry
        session_->abort();
        return registry_.abort();
    }
    // True iff an abort of this party could not be made durable (persistence only).
    bool abort_persist_failed() const { return registry_.abort_persist_failed(); }
    const std::string& abort_persist_error() const { return registry_.abort_persist_error(); }

    // Setup: preprocessing setup, Delta_b and the session wires of Delta.
    // reserve_triples AND triples for the planned Key / Gate commands and the
    // triples of reserve_outmasks on-demand output masks are drawn together
    // with those of the setup circuit, in ONE bucketing population.  The
    // masks themselves are generated when they are needed.
    void setup(size_t reserve_triples = 0, size_t reserve_outmasks = 0) {
        require_active();
        if (checker_) throw state::StateError("setup() was already run");
        // Circuits the setup needs: a missing one is reported before any message.
        session_->store().get(Session::setup_circuit_name(s()));
        if (reserve_outmasks) session_->outmask_and_count();
        step([&] {
            session_->setup(reserve_triples, reserve_outmasks);
            if (factory_)
                checker_ = factory_(role(), session_->channel(), session_->sid(), session_->delta_share(), s(),
                                    &session_->book());
            else
                checker_.reset(new spdz2k::ArithChecker(role(), session_->channel(), session_->sid(),
                                                        session_->delta_share(), s(), &session_->book()));
            // Change G: a failed ArithCheck (or any other death of the checker) retires
            // the whole session, also when the checker is driven from outside this class.
            checker_->set_on_abort([this] {
                outmasks_.clear();
                session_->abort();
                registry_.abort();
            });
            return 0;
        });
    }

    // -- Key and Query
    // Key(kid, family, n; alpha_b, beta_b); beta_share == nullptr fixes the payload to 1.
    uint64_t key(uint64_t kid, protocol::Family family, unsigned n, uint64_t alpha_share,
                 const uint64_t* beta_share = nullptr, protocol::IndexMode index_mode = protocol::INDEX_XOR) {
        return key_batch(kid, family, n, std::vector<uint64_t>(1, alpha_share),
                         beta_share ? std::vector<uint64_t>(1, *beta_share) : std::vector<uint64_t>(),
                         beta_share != nullptr, index_mode)[0];
    }

    // Several keys of one family from ONE circuit execution; returns their identifiers.
    std::vector<uint64_t> key_batch(uint64_t kid, protocol::Family family, unsigned n,
                                    const std::vector<uint64_t>& alpha_shares,
                                    const std::vector<uint64_t>& beta_shares = std::vector<uint64_t>(),
                                    bool secret_beta = false, protocol::IndexMode index_mode = protocol::INDEX_XOR) {
        require_setup();
        if (family != protocol::FAMILY_DPF && family != protocol::FAMILY_DCF)
            throw std::invalid_argument("family must be dpf or dcf");
        const size_t count = alpha_shares.size();
        const protocol::BetaMode beta_mode = secret_beta ? protocol::BETA_SECRET : protocol::BETA_PUBLIC_ONE;
        protocol::detail::key_inputs(n, k(), beta_mode, alpha_shares, beta_shares);  // argument checks only
        const size_t n_and =
            protocol::and_count(session_->store(), protocol::key_circuit_name(family, n, k(), s(), index_mode, beta_mode));
        if (!id_refused(state::Registry::KID, kid)) require_triples(n_and, count);  // strict mode: before any change of state
        const uint64_t first = registry_.begin_key(kid, count);  // GENERATING
        std::vector<state::KeyRecord> recs(count);
        step([&] {
            if (family == protocol::FAMILY_DPF) {
                std::vector<fss::DpfKey> keys =
                    protocol::gen_dpf(*session_, n, index_mode, beta_mode, alpha_shares, beta_shares, rng_);
                for (size_t j = 0; j < count; j++) recs[j].dpf = std::move(keys[j]);
            } else {
                std::vector<fss::DcfKey> keys =
                    protocol::gen_dcf(*session_, n, index_mode, beta_mode, alpha_shares, beta_shares, rng_);
                for (size_t j = 0; j < count; j++) recs[j].dcf = std::move(keys[j]);
            }
            return 0;
        });
        std::vector<uint64_t> ids;
        for (size_t j = 0; j < count; j++) {  // the output check has passed
            recs[j].family = family;
            recs[j].n = n;
            recs[j].beta_mode = beta_mode;
            registry_.certify_key(first + j, recs[j]);
            ids.push_back(first + j);
        }
        return ids;
    }

    // Query(kid, x) at a public point: this party's exact share of [[f(x)]].
    AShare query(uint64_t kid, uint64_t x) {
        require_setup();
        const state::KeyRecord& rec = registry_.key(kid);
        if (rec.n < 64 && (x >> rec.n) != 0) throw std::invalid_argument("x must lie in [0, 2^n)");
        registry_.next_qid();
        const fss::Share sh = rec.family == protocol::FAMILY_DPF ? fss::dpf_eval(rec.dpf, x) : fss::dcf_eval(rec.dcf, x);
        return AShare(sh.y, sh.tau, w(), true);
    }

    // Query at every point of the domain (EvalAll): 2^n exact shares.
    std::vector<AShare> query_all(uint64_t kid) {
        require_setup();
        const state::KeyRecord& rec = registry_.key(kid);
        registry_.next_qid();
        std::vector<u128> y, tau;
        if (rec.family == protocol::FAMILY_DPF) fss::dpf_eval_all(rec.dpf, y, tau, eval_threads_);
        else fss::dcf_eval_all(rec.dcf, y, tau, eval_threads_);
        std::vector<AShare> out;
        out.reserve(y.size());
        for (size_t i = 0; i < y.size(); i++) out.push_back(AShare(y[i], tau[i], w(), true));
        return out;
    }

    // -- Gate
    // Gate(gid, G): generate the one-time material of `spec`; returns gid.
    uint64_t gate(uint64_t gid, const gates::GateSpec& spec) { return gate_batch(gid, spec, 1)[0]; }

    // `count` independent gates of one kind from ONE circuit execution.
    std::vector<uint64_t> gate_batch(uint64_t gid, const gates::GateSpec& spec, size_t count) {
        require_setup();
        if (spec.k != k() || spec.s != s()) throw std::invalid_argument("the gate does not use the session's k and s");
        if (count < 1) throw std::invalid_argument("count must be a positive integer");
        const protocol::GateDescriptor desc = spec.descriptor();
        const size_t n_and = protocol::and_count(session_->store(), desc);
        const std::shared_ptr<const gates::GateSpec> shared_spec(new gates::GateSpec(spec));
        if (!id_refused(state::Registry::GID, gid)) require_triples(n_and, count);  // strict mode: before any change of state
        const uint64_t first = registry_.begin_gate(gid, count);  // GENERATING
        std::vector<protocol::GateMaterial> mats =
            step([&] { return protocol::gen_gate(*session_, desc, count, rng_); });
        std::vector<uint64_t> ids;
        for (size_t j = 0; j < count; j++) {
            for (const protocol::SharePair& m : mats[j].outmask_shares)  // one-time output masks of the gate
                outmasks_.push_back(AShare(m.value, m.tag, w()));
            std::shared_ptr<gates::GateEval> ev(new gates::GateEval(std::move(mats[j]), false, eval_threads_));
            ev->set_delta_share(session_->delta_share());
            registry_.certify_gate(first + j, std::move(ev), shared_spec);
            ids.push_back(first + j);
        }
        return ids;
    }

    // -- Mask
    // Mask(gid, shared input): opens x + r mod 2^n (commit-then-open, no MAC); returns x_hat.
    uint64_t mask_shared(uint64_t gid, uint64_t x_share) {
        require_setup();
        {
            const state::GateRecord& rec = registry_.gate(gid);
            if (!rec.has_material()) throw state::StateError("gate " + std::to_string(gid) + " is not certified");
            if ((x_share & ~gates::domain_mask(rec.material().n())) != 0)
                throw std::invalid_argument("x_share must lie in [0, 2^n)");
        }
        state::GateRecord& rec = registry_.use_gate(gid);  // USED from here on, whatever happens
        const unsigned n = rec.material().n(), nb = (n + 7) / 8;
        const uint64_t mask = gates::domain_mask(n);
        const uint64_t mine = (x_share + u128_lo(rec.material().rp_share)) & mask;
        const uint64_t x_hat = step([&] {
            bytes msg(nb);
            for (unsigned i = 0; i < nb; i++) msg[i] = (uint8_t)(mine >> (8 * i));
            bytes reply;
            {
                const abb::PhaseRestore restore(session_->book());
                session_->book().mark("mask");
                reply = mask_exchange(gid, msg);
            }
            if (reply.size() != nb) throw ProtocolAbort("mask: message has the wrong length");
            uint64_t peer = 0;
            for (unsigned i = 0; i < nb; i++) peer |= (uint64_t)reply[i] << (8 * i);
            if ((peer & ~mask) != 0) throw ProtocolAbort("mask: share out of range");
            return (mine + peer) & mask;
        });
        registry_.set_masked(gid, x_hat, "shared");
        return x_hat;
    }

    // Mask(gid, wire): opens x + r' in Z_{2^(n+s)} (check deferred); returns x_hat.
    uint64_t mask_wire(uint64_t gid, const AShare& share) {
        require_setup();
        {
            const state::GateRecord& rec = registry_.gate(gid);
            if (!rec.has_material()) throw state::StateError("gate " + std::to_string(gid) + " is not certified");
            if (!rec.material().auth_input())
                throw std::invalid_argument("the gate was generated without an authenticated input offset");
            if (share.bits < rec.material().n() + s())
                throw std::invalid_argument("expected a sharing over a modulus of at least 2^(n+s)");
        }
        state::GateRecord& rec = registry_.use_gate(gid);  // USED from here on, whatever happens
        const unsigned n = rec.material().n();
        const AShare offset(rec.material().rp_share, rec.material().rp_tag_share, n + s());  // [[r']], uniform in the ring
        const AShare masked = share.reduce(n, s()) + offset;
        const u128 opened = step([&] { return checker_->open_deferred(masked); });
        const uint64_t x_hat = u128_lo(opened) & gates::domain_mask(n);
        registry_.set_masked(gid, x_hat, "wire", opened);
        return x_hat;
    }

    // -- Apply and Lin
    // Apply(gid): this party's share(s) of [[G(x)]] (local).
    std::vector<AShare> apply(uint64_t gid) {
        require_setup();
        state::GateRecord& rec = registry_.masked_gate(gid);
        if (!rec.spec || rec.spec->kind == gates::GateSpec::OFFSET_ONLY || !rec.evaluator)
            throw state::StateError("gate " + std::to_string(gid) + " has no function");
        return rec.spec->evaluate(*rec.evaluator, rec.x_hat);
    }

    // Lin: sum c_i [[v_i]] + constant for public c_i (elements of Z_{2^w}).
    AShare lin(const std::vector<std::pair<u128, AShare>>& terms, u128 constant = u128(0)) {
        require_setup();
        for (const std::pair<u128, AShare>& t : terms)
            if (t.second.bits != w()) throw std::invalid_argument("terms are pairs (public integer, sharing over Z_{2^w})");
        if (terms.size() == 1 && terms[0].first == u128(1) && constant == u128(0)) return terms[0].second;
        AShare acc(u128(0), u128(0), w());
        for (const std::pair<u128, AShare>& t : terms) acc = acc + t.second.scale(t.first);
        if (constant != u128(0)) acc = acc.add_public(constant, role(), session_->delta_share());
        return acc.with_exact(false);
    }

    // -- Open
    // Generate `count` output masks [[2^k rho]] with one circuit execution.
    void reserve_outmasks(size_t count) {
        require_setup();
        if (count == 0) return;
        // One circuit with a single mask, run on `count` instances.
        const protocol::GateDescriptor desc = protocol::outmask_descriptor(k(), s());
        require_triples(protocol::and_count(session_->store(), desc), count);  // strict mode: before any message
        const std::vector<protocol::GateMaterial> mats =
            step([&] { return protocol::gen_gate(*session_, desc, count, rng_); });
        for (const protocol::GateMaterial& m : mats)
            for (const protocol::SharePair& p : m.outmask_shares) outmasks_.push_back(AShare(p.value, p.tag, w()));
    }

    // Open: check, open (masked unless exact), check; returns v mod 2^k.
    u128 open(const AShare& share) { return open_many(std::vector<AShare>(1, share))[0]; }

    // Open of several sharings with one opening exchange and the two checks.
    // Order: ArithCheck on the deferred openings, then (if the pool is short)
    // the generation of the missing output masks, then the opening and its
    // ArithCheck.
    std::vector<u128> open_many(const std::vector<AShare>& shares) {
        require_setup();
        size_t need = 0;
        for (const AShare& sh : shares) {
            if (sh.bits != w()) throw std::invalid_argument("expected sharings over Z_{2^w}");
            need += sh.exact ? 0 : 1;
        }
        if (shares.empty()) return std::vector<u128>();
        const size_t missing = need > outmasks_.size() ? need - outmasks_.size() : 0;
        if (missing)  // strict mode: refused before any message (nothing has been sent)
            require_triples(protocol::and_count(session_->store(), protocol::outmask_descriptor(k(), s())), missing);
        step([&] {  // step 1: every deferred opening is verified first
            checker_->check();
            return 0;
        });
        if (missing) reserve_outmasks(missing);  // step 2: masks from Pi_Gen, after the check
        std::vector<AShare> used;  // each mask leaves the pool before the opening starts
        used.reserve(need);
        std::vector<const AShare*> masks(shares.size(), nullptr);
        for (size_t j = 0; j < shares.size(); j++) {
            if (shares[j].exact) continue;
            used.push_back(outmasks_.front());
            outmasks_.pop_front();
            masks[j] = &used.back();
        }
        return step([&] { return checker_->open_final_many(shares, masks); });  // steps 3 and 4
    }

    // -- reporting
    // Public parameters and counters (no secrets), as JSON.
    std::string info_json() {
        const state::Registry::Summary sm = registry_.summary();
        return std::string("{\"state\": \"") + (active() ? "ACTIVE" : "ABORTED") + "\", \"session\": " +
               session_->info_json() + ", \"registry\": {\"keys_generating\": " + std::to_string(sm.keys_generating) +
               ", \"keys_certified\": " + std::to_string(sm.keys_certified) + ", \"gates_generating\": " +
               std::to_string(sm.gates_generating) + ", \"gates_certified\": " + std::to_string(sm.gates_certified) +
               ", \"gates_used\": " + std::to_string(sm.gates_used) + "}, \"outmasks_available\": " +
               std::to_string(outmasks_.size()) + ", \"arith_openings\": " +
               std::to_string(checker_ ? checker_->openings() : 0) + ", \"arith_checks\": " +
               std::to_string(checker_ ? checker_->checks() : 0) + ", \"arith_pending\": " +
               std::to_string(checker_ ? checker_->pending() : 0) + ", \"abort_persist_failed\": " +
               (registry_.abort_persist_failed() ? "true" : "false") + "}";
    }

protected:
    // Commit-then-open exchange of the masked shares of gate `gid`; returns
    // the peer's message.  `msg` is this party's x_b + r_b mod 2^n as
    // ceil(n/8) bytes little-endian.  Both parties first exchange commitments
    // bound to (sid_mask, "mask|" || le64(gid), own party index) and only then
    // the openings, so that a rushing party cannot fix x_hat by choosing its
    // share after it has seen the other one.  The honest code of this step is
    // the whole method; a deviating party (tests) overrides it.
    virtual bytes mask_exchange(uint64_t gid, const bytes& msg) {
        static const char prefix[] = "mask|";
        bytes label(prefix, prefix + sizeof prefix - 1);
        const std::array<uint8_t, 8> g = hashes::le64(gid);
        label.insert(label.end(), g.begin(), g.end());
        return hashes::committed_exchange(session_->channel(), sid_mask_, label, msg, sys_, msg.size());
    }

private:
    // Strict mode: refuse a generation that the reserved pool cannot serve
    // (TripleBudgetError; nothing is sent and nothing changes).
    void require_triples(size_t n_and, size_t count) {
        if (n_and && count > SIZE_MAX / n_and) throw std::invalid_argument("the triple budget does not fit in size_t");
        session_->engine().require_triples(n_and * count);
    }
    // True if the registry will refuse `requested` as the next identifier of
    // `kind`.  Used to order the two refusals that precede a generation: a bad
    // identifier is reported by the registry before the triple budget is
    // looked at.
    bool id_refused(state::Registry::Kind kind, uint64_t requested) {
        if (requested == NEXT) return false;
        return requested < registry_.counters().next[kind] || requested > (~(uint64_t)0 >> 1);
    }
    bool broken() const {
        return !registry_.active() || !session_->active() || session_->engine().dead() || session_->prep().aborted() ||
               (checker_ && checker_->dead());
    }
    void require_active() {
        if (broken()) {
            abort();
            throw ProtocolAbort("session aborted");
        }
    }
    void require_setup() {
        require_active();
        if (!checker_) throw state::StateError("setup() has not been run");
    }

    // A protocol step: any failure after the first message, and any abort, retires the session.
    template <class F>
    auto step(F&& body) -> decltype(body()) {
        require_active();
        const ChannelStats before = session_->channel().stats();
        try {
            return body();
        } catch (const std::invalid_argument&) {
            const ChannelStats& now = session_->channel().stats();
            if (now.msgs_sent != before.msgs_sent || now.msgs_recv != before.msgs_recv || broken()) abort();
            throw;
        } catch (const state::StateError&) {
            const ChannelStats& now = session_->channel().stats();
            if (now.msgs_sent != before.msgs_sent || now.msgs_recv != before.msgs_recv || broken()) abort();
            throw;
        } catch (...) {
            abort();
            throw;
        }
    }

    Session* session_;
    state::Registry registry_;
    bytes sid_mask_;
    std::unique_ptr<spdz2k::ArithChecker> checker_;
    CheckerFactory factory_;
    std::deque<AShare> outmasks_;
    RandomSource* rng_ = nullptr;
    SystemRng sys_;  // commitment randomness of the Mask exchange
    unsigned eval_threads_ = 1;
};

}  // namespace couplet
