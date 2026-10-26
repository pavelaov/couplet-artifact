// couplet/spdz2k.hpp -- SPDZ2k authenticated sharings and the batched MAC
// check (DESIGN.md sections 2.3 and 5.3); the C++ counterpart of
// python/couplet/spdz2k.py, with the same messages, labels and derivations.
// Everything here is the view of ONE party.
//
// AShare{value, mac, bits, exact} is this party's share of [[v]] over
// Z_{2^bits}: the two parties' `value` fields sum to a representative v~ of
// the secret and their `mac` fields sum to Delta * v~ modulo 2^bits, where
// Delta = Delta_0 + Delta_1 is the session MAC key as an (s+1)-bit integer
// that is NEVER reduced (change C).  `bits` is the modulus exponent m: w = k +
// s for wires, n + s after reduce().  The payload is v~ mod 2^(bits - s).
// `exact` states that the representative itself is smaller than 2^(bits - s)
// (section 2.3): outputs of DPF / DCF evaluations, indicators and table
// look-ups are exact; sums and multiples are in general not.  The flag is
// public: both parties derive the same value from the public history of a
// sharing.  Linear operations are local:
//
//     a + b, a - b, -a      same bits; the result is marked not exact
//     a.scale(c)            public c (an element of Z_{2^bits}; pass a
//                           negative coefficient in two's complement)
//     a.add_public(c, role, delta_share)
//                           P_0 adds c to its value share, both parties add
//                           Delta_b * c to their MAC share
//     a.reduce(n, s)        the sharing of v~ mod 2^(n+s) over Z_{2^(n+s)}
//     a.with_exact(flag)    set the flag (callers that know an identity over
//                           the integers, e.g. the gates of section 6)
//     a.with_sign_exact(flag)
//                           set the sign_exact flag (Couplet layer)
//
// `sign_exact` is the value class of the Couplet layer (DESIGN_M3.md section
// 3): the representative is the two's-complement embedding of the intended
// signed integer, so an opening needs no output mask.  Like `exact` it is
// public metadata that every linear operation clears and that a caller
// re-sets when its public bound analysis guarantees that the true integer
// fits in bits - 1 bits; the core never sets it (python/couplet/spdz2k.py).
//
// ArithChecker(role, channel, sid, delta_share, s) opens sharings and runs
// ArithCheck of section 2.3:
//
//     open_deferred(share)
//         one exchange of the value shares; the opened representative is
//         returned and recorded together with this party's MAC share.
//         Nothing is verified yet: the caller may only use the value in ways
//         that are safe for an unverified opening (a masked gate input).
//     check()
//         ArithCheck over everything opened since the last check, one check
//         per modulus m: a coin toss fixes chi_j in [0, 2^s) AFTER the
//         openings; y = sum chi_j v~_j; zeta_b = sum chi_j mu_{j,b} - Delta_b y
//         mod 2^m; the zeta_b are exchanged by a simultaneous
//         commit-then-open and the check passes iff zeta_0 + zeta_1 = 0 mod
//         2^m for every modulus.  A failure throws ProtocolAbort (the session
//         must be retired, change G).
//     open_final(share, outmask)
//         the release of section 5.3 (changes E and F): check() on all
//         deferred openings, then the opening of the value, to which the
//         output mask [[2^k rho]] is added unless the sharing is exact, then
//         check() on exactly this opening; only then is v mod 2^k returned
//         (k = bits - s).  A sharing that is not exact is never opened
//         without a mask: the call is refused before any message.
//
// Binding.  The checker works under the sub-session identifier
// SHA-256(frame("Couplet-arith-sid", sid)).  Every opening has a
// session-wide index and every check a counter; neither is ever reused.  The
// coin toss of a check is labelled "arithcheck|" || le64(counter) ||
// SHA-256(frame("Couplet-arith-transcript", sid', le64(counter),
// {le32(m) || le64(index) || value}...)), and the commitments carry the label
// "arithcheck-zeta|" || le64(counter) and the committer's index, so a message
// of another check, of another session or of the other party is rejected.
// chi_j of modulus m is the j-th block of ceil(s / 8) bytes of
// sha256_expand("Couplet-arith-chi", seed, le32(m)), truncated to s bits.
//
// Messages.  A value share over Z_{2^m} is ceil(m / 8) bytes little-endian; a
// message of the wrong length or with a value >= 2^m aborts.  Rounds: one per
// opening call; four per check (coin toss 2, committed exchange 2).  A check
// with nothing to verify sends nothing.  Accounting phases: "arith-open" and
// "arith-check" in the PhaseBook, if one is given.
//
// Abort.  Any failure inside a protocol step retires the checker: the key
// share, the recorded MAC shares and the opened values are dropped and every
// later call throws ProtocolAbort.  Invalid arguments throw
// std::invalid_argument before any message and leave the checker usable.
// A checker is one component of a session (change G: an abort retires the
// WHOLE session).  set_on_abort(hook) registers what else has to be retired:
// the hook runs once, when the checker dies, whatever the cause.  The driver
// couplet::Party installs it; code that uses a Session and an ArithChecker
// directly calls session.bind_abort(checker) (session.hpp).
//
// Opened values.  An opening is public the moment the shares are exchanged,
// but it is only ACCEPTED when the ArithCheck that covers it has passed
// (change F).  The checker keeps the two apart:
//     last_opened()   the representatives covered by the latest check that
//                     PASSED, in opening order (after open_final: the
//                     released values); empty before the first check and
//                     after an abort;
//     unverified()    the representatives opened since the last check; they
//                     are NOT verified and may only be used as a masked gate
//                     input.
//
// Messages are length-bounded (channel.hpp): every receive names the exact
// length the opening or the check prescribes.
//
// Deviations are not built into this class.  A malicious party is modelled by
// a subclass that overrides the three protected steps outgoing_values,
// recorded_mac and outgoing_zeta (the tests do this).
//
// Parameter range: 1 <= s <= 64 and bits <= 128.
//
// deal / reconstruct are dealer-style helpers for tests and formula checks;
// they are not part of any protocol.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "abb.hpp"
#include "channel.hpp"
#include "hashes.hpp"
#include "rng.hpp"
#include "u128.hpp"

namespace couplet {
namespace spdz2k {

// Encoded size of a value in Z_{2^bits}.
inline unsigned share_bytes(unsigned bits) { return (bits + 7) / 8; }

// The bound 2^(-s + log2(s+1)) = (s+1) / 2^s of Lemma 2 (without the commitment term).
inline double lemma2_bound(unsigned s) {
    double b = (double)s + 1.0;
    for (unsigned i = 0; i < s; i++) b *= 0.5;
    return b;
}

// ------------------------------------------------------------------ sharings
struct AShare {
    u128 value = u128(0);
    u128 mac = u128(0);
    unsigned bits = 1;
    bool exact = false;
    bool sign_exact = false;  // Couplet layer only (see the header comment)

    AShare() {}
    AShare(u128 v, u128 m, unsigned b, bool ex = false, bool sx = false)
        : value(v), mac(m), bits(b), exact(ex), sign_exact(sx) {
        if (bits < 1 || bits > 128) throw std::invalid_argument("bits must be in 1..128");
        if (value != (value & u128_mask(bits)) || mac != (mac & u128_mask(bits)))
            throw std::invalid_argument("share components must lie in [0, 2^bits)");
    }

    friend AShare operator+(const AShare& a, const AShare& b) {
        same(a, b);
        const u128 mask = u128_mask(a.bits);
        return AShare((a.value + b.value) & mask, (a.mac + b.mac) & mask, a.bits);
    }
    friend AShare operator-(const AShare& a, const AShare& b) {
        same(a, b);
        const u128 mask = u128_mask(a.bits);
        return AShare((a.value - b.value) & mask, (a.mac - b.mac) & mask, a.bits);
    }
    AShare operator-() const {
        const u128 mask = u128_mask(bits);
        return AShare((u128(0) - value) & mask, (u128(0) - mac) & mask, bits);
    }

    // [[c v]] for a public c (exact only for c = 0, or c = 1 on an exact sharing).
    AShare scale(u128 c) const {
        const u128 mask = u128_mask(bits);
        const u128 cm = c & mask;
        const bool ex = c == u128(0) || (c == u128(1) && exact);
        return AShare((value * cm) & mask, (mac * cm) & mask, bits, ex);
    }

    // [[v + c]] for a public c: needs the party index and its Delta_b.
    AShare add_public(u128 c, int role, uint64_t delta_share) const {
        if (role != 0 && role != 1) throw std::invalid_argument("role must be 0 or 1");
        const u128 mask = u128_mask(bits);
        const u128 v = role == 0 ? ((value + c) & mask) : value;
        return AShare(v, (mac + u128(delta_share) * c) & mask, bits, c == u128(0) && exact);
    }

    // The sharing modulo 2^(n+s) (payload v mod 2^n); local (section 2.3).
    AShare reduce(unsigned n, unsigned s) const {
        if (n < 1) throw std::invalid_argument("need n >= 1 and s >= 0");
        const unsigned nb = n + s;
        if (nb > bits) throw std::invalid_argument("cannot reduce to a larger modulus");
        const u128 mask = u128_mask(nb);
        return AShare(value & mask, mac & mask, nb, exact && nb == bits);
    }

    // The same share with the exactness flag set by a caller that knows the identity.
    AShare with_exact(bool flag = true) const {
        AShare r = *this;
        r.exact = flag;
        return r;
    }
    // The same share with the sign_exact flag set by a caller that knows the bound.
    AShare with_sign_exact(bool flag = true) const {
        AShare r = *this;
        r.sign_exact = flag;
        return r;
    }

    friend bool operator==(const AShare& a, const AShare& b) {
        return a.value == b.value && a.mac == b.mac && a.bits == b.bits && a.exact == b.exact &&
               a.sign_exact == b.sign_exact;
    }

private:
    static void same(const AShare& a, const AShare& b) {
        if (a.bits != b.bits) throw std::invalid_argument("sharings over different moduli cannot be combined (use reduce)");
    }
};

namespace detail {
inline u128 random_u128(RandomSource& rng, unsigned bits) {
    uint8_t buf[16];
    rng.fill(buf, 16);
    return u128_load_le(buf) & u128_mask(bits);
}
}  // namespace detail

// Dealer-style sharing of the representative `value` (tests and formula
// checks only).  `delta` is the full key Delta_0 + Delta_1.  Returns the
// shares of P_0 and P_1.
inline std::pair<AShare, AShare> deal(u128 value, u128 delta, unsigned bits, bool exact, RandomSource& rng) {
    const u128 mask = u128_mask(bits);
    const u128 v0 = detail::random_u128(rng, bits), m0 = detail::random_u128(rng, bits);
    value = value & mask;
    return std::make_pair(AShare(v0, m0, bits, exact),
                          AShare((value - v0) & mask, (delta * value - m0) & mask, bits, exact));
}

// (representative, sum of the MAC shares) of two shares (tests only).
inline std::pair<u128, u128> reconstruct(const AShare& a, const AShare& b) {
    if (a.bits != b.bits) throw std::invalid_argument("shares over different moduli");
    const u128 mask = u128_mask(a.bits);
    return std::make_pair((a.value + b.value) & mask, (a.mac + b.mac) & mask);
}

// The coefficients chi_j in [0, 2^s) of one modulus from the tossed seed.
inline std::vector<uint64_t> check_coefficients(const bytes& seed, unsigned bits, size_t count, unsigned s) {
    if (s < 1 || s > 64) throw std::invalid_argument("s must be in 1..64");
    const unsigned nb = share_bytes(s);
    const uint8_t le_bits[4] = {(uint8_t)bits, (uint8_t)(bits >> 8), (uint8_t)(bits >> 16), (uint8_t)(bits >> 24)};
    const bytes raw = hashes::sha256_expand((size_t)nb * count, {"Couplet-arith-chi", seed, hashes::Part(le_bits, 4)});
    const uint64_t mask = s == 64 ? ~(uint64_t)0 : (((uint64_t)1 << s) - 1);
    std::vector<uint64_t> chi(count);
    for (size_t j = 0; j < count; j++) {
        uint64_t v = 0;
        for (unsigned i = 0; i < nb; i++) v |= (uint64_t)raw[j * nb + i] << (8 * i);
        chi[j] = v & mask;
    }
    return chi;
}

// ------------------------------------------------------------------ ArithCheck
class ArithChecker {
public:
    // One recorded opening: modulus bits, session-wide index, opened
    // representative, this party's MAC share.
    struct Opening {
        unsigned bits;
        uint64_t index;
        u128 value;
        u128 mac;
    };

    ArithChecker(int role, Channel& ch, const bytes& sid, uint64_t delta_share, unsigned s,
                 abb::PhaseBook* book = nullptr)
        : role_(role), ch_(&ch), sid_(sid), s_(s), delta_(delta_share), book_(book) {
        if (role != 0 && role != 1) throw std::invalid_argument("role must be 0 or 1");
        if (ch.party() != role) throw std::invalid_argument("role does not match the channel endpoint");
        if (s < 1 || s > 64) throw std::invalid_argument("s must be in 1..64");
        if (s < 64 && (delta_share >> s) != 0) throw std::invalid_argument("delta_share must lie in [0, 2^s)");
        const hashes::hash32 h = hashes::framed_sha256({"Couplet-arith-sid", sid_});
        sid_arith_.assign(h.begin(), h.end());
    }
    virtual ~ArithChecker() {}
    ArithChecker(const ArithChecker&) = delete;
    ArithChecker& operator=(const ArithChecker&) = delete;

    int role() const { return role_; }
    unsigned s() const { return s_; }
    bool dead() const { return dead_; }
    uint64_t openings() const { return openings_; }  // openings so far
    uint64_t checks() const { return checks_; }      // checks run so far
    size_t pending() const { return pending_.size(); }  // openings not yet verified

    // The representatives covered by the latest check that passed (opening
    // order); empty before the first check and after an abort.
    const std::vector<u128>& last_opened() const { return last_opened_; }
    // The representatives opened since the last check: public, NOT verified.
    std::vector<u128> unverified() const {
        std::vector<u128> v;
        v.reserve(pending_.size());
        for (const Opening& o : pending_) v.push_back(o.value);
        return v;
    }

    // What else to retire when this checker dies (change G): called once, on
    // the transition to dead, from abort().  Typically Session::abort.
    void set_on_abort(std::function<void()> hook) { on_abort_ = std::move(hook); }

    // Retire the checker (change G): drop the key share, the recorded MAC
    // shares and every opened value, then run the abort hook (first call only).
    void abort() {
        const bool first = !dead_;
        dead_ = true;
        delta_ = 0;
        pending_.clear();
        last_opened_.clear();
        if (first && on_abort_) {
            try {
                on_abort_();
            } catch (...) {  // the checker is retired whatever the hook does
            }
        }
    }

    // Open `share` now, verify at the next check(); returns the representative.
    u128 open_deferred(const AShare& share) { return open_deferred_many(std::vector<AShare>(1, share))[0]; }

    // As open_deferred for several sharings in ONE exchange.
    std::vector<u128> open_deferred_many(const std::vector<AShare>& shares) {
        check_shares(shares);
        require_alive();
        if (shares.empty()) return std::vector<u128>();
        return step([&] { return exchange(shares); });
    }

    // ArithCheck over every opening since the last check (section 2.3).
    void check() {
        require_alive();
        if (pending_.empty()) return;
        step([&] {
            run_check();
            return std::vector<u128>();
        });
    }

    // Release `share` (section 5.3): check, open, check; returns v mod 2^(bits - s).
    u128 open_final(const AShare& share, const AShare* outmask = nullptr) {
        return open_final_many(std::vector<AShare>(1, share), std::vector<const AShare*>(1, outmask))[0];
    }

    // Release several sharings with one opening exchange and the two checks.
    // outmasks[j] is the output mask [[2^k rho]] of sharing j (a sharing over
    // the same modulus whose components are multiples of 2^k, never used
    // before) and must be given exactly for the sharings that are not exact.
    std::vector<u128> open_final_many(const std::vector<AShare>& shares,
                                      const std::vector<const AShare*>& outmasks = std::vector<const AShare*>()) {
        check_shares(shares);
        std::vector<const AShare*> masks = outmasks;
        if (masks.empty()) masks.assign(shares.size(), nullptr);
        if (masks.size() != shares.size()) throw std::invalid_argument("one output mask entry per sharing is required");
        std::vector<AShare> release;
        for (size_t j = 0; j < shares.size(); j++) {
            const AShare& share = shares[j];
            const unsigned k = share.bits - s_;
            if (share.exact) {
                if (masks[j]) throw std::invalid_argument("an exact sharing is opened without an output mask");
                release.push_back(share);
                continue;
            }
            if (!masks[j]) throw std::invalid_argument("a sharing that is not exact must be opened with an output mask");
            if (masks[j]->bits != share.bits)
                throw std::invalid_argument("the output mask must be a sharing over the same modulus");
            if ((masks[j]->value & u128_mask(k)) != u128(0) || (masks[j]->mac & u128_mask(k)) != u128(0))
                throw std::invalid_argument("output mask shares must be multiples of 2^k");
            release.push_back(share + *masks[j]);
        }
        require_alive();
        if (shares.empty()) return std::vector<u128>();
        std::vector<u128> opened = step([&] {
            run_check();                                  // every deferred opening first
            std::vector<u128> out = exchange(release);    // the release itself
            run_check();                                  // exactly this opening
            return out;
        });
        for (size_t j = 0; j < opened.size(); j++) opened[j] = opened[j] & u128_mask(shares[j].bits - s_);
        return opened;
    }

protected:
    // -- the three steps a deviating party would change (tests override them)
    // The value shares sent in an opening (honest: this party's shares).
    virtual std::vector<u128> outgoing_values(const std::vector<AShare>& shares, uint64_t first_index) {
        (void)first_index;
        std::vector<u128> v(shares.size());
        for (size_t i = 0; i < shares.size(); i++) v[i] = shares[i].value;
        return v;
    }
    // The MAC share kept for the check of opening `index` (honest: this party's share).
    virtual u128 recorded_mac(const AShare& share, uint64_t index) {
        (void)index;
        return share.mac;
    }
    // The value committed to in the check of modulus 2^bits (honest: zeta).
    virtual u128 outgoing_zeta(unsigned bits, u128 zeta, const std::vector<uint64_t>& chis,
                               const std::vector<Opening>& items, uint64_t counter) {
        (void)bits;
        (void)chis;
        (void)items;
        (void)counter;
        return zeta;
    }

    uint64_t delta_share() const { return delta_; }

private:
    void require_alive() const {
        if (dead_) throw ProtocolAbort("session aborted");
    }
    void check_shares(const std::vector<AShare>& shares) const {
        for (const AShare& sh : shares)
            if (sh.bits <= s_ || sh.bits > 128) throw std::invalid_argument("the modulus must be larger than 2^s");
    }
    void mark(const char* name) {
        if (book_) book_->mark(name);
    }

    // Run one protocol step: any failure retires the checker; the phase is restored.
    template <class F>
    std::vector<u128> step(F&& fn) {
        require_alive();
        const int prev = book_ ? book_->current() : abb::PhaseBook::NONE;
        try {
            std::vector<u128> r = fn();
            if (book_) book_->mark(prev);
            return r;
        } catch (...) {
            abort();
            if (book_) book_->mark(prev);
            throw;
        }
    }

    static void put(bytes& out, u128 v, unsigned nbytes) {
        uint8_t buf[16];
        u128_store_le(buf, v);
        out.insert(out.end(), buf, buf + nbytes);
    }
    static u128 get(const uint8_t* p, unsigned nbytes) {
        uint8_t buf[16] = {0};
        std::memcpy(buf, p, nbytes);
        return u128_load_le(buf);
    }

    // One exchange of the value shares of `shares`; records the openings.
    std::vector<u128> exchange(const std::vector<AShare>& shares) {
        mark("arith-open");
        const uint64_t first = openings_;
        openings_ += shares.size();
        std::vector<u128> sent = outgoing_values(shares, first);
        if (sent.size() != shares.size()) throw std::logic_error("outgoing_values returned the wrong number of values");
        bytes msg;
        for (size_t j = 0; j < shares.size(); j++) {
            sent[j] = sent[j] & u128_mask(shares[j].bits);
            put(msg, sent[j], share_bytes(shares[j].bits));
        }
        const bytes reply = ch_->exchange(msg, msg.size(), "opening: message has the wrong length");
        if (reply.size() != msg.size()) throw ProtocolAbort("opening: message has the wrong length");
        std::vector<u128> opened;
        size_t pos = 0;
        for (size_t j = 0; j < shares.size(); j++) {
            const unsigned bits = shares[j].bits, nb = share_bytes(bits);
            const u128 theirs = get(reply.data() + pos, nb);
            pos += nb;
            if (theirs != (theirs & u128_mask(bits))) throw ProtocolAbort("opening: value share out of range");
            const u128 value = (sent[j] + theirs) & u128_mask(bits);
            opened.push_back(value);
            pending_.push_back(Opening{bits, first + j, value, recorded_mac(shares[j], first + j) & u128_mask(bits)});
        }
        return opened;  // public, not yet verified: see unverified() / last_opened()
    }

    void run_check() {
        if (pending_.empty()) return;  // nothing was opened: no message
        mark("arith-check");
        const uint64_t counter = checks_++;
        const std::array<uint8_t, 8> cbytes = hashes::le64(counter);
        std::map<unsigned, std::vector<Opening>> groups;  // ascending modulus
        std::vector<bytes> store;
        store.reserve(pending_.size());
        for (const Opening& o : pending_) {
            groups[o.bits].push_back(o);
            bytes part;
            for (int i = 0; i < 4; i++) part.push_back((uint8_t)(o.bits >> (8 * i)));
            const std::array<uint8_t, 8> idx = hashes::le64(o.index);
            part.insert(part.end(), idx.begin(), idx.end());
            put(part, o.value, share_bytes(o.bits));
            store.push_back(std::move(part));
        }
        std::vector<hashes::Part> parts;
        parts.push_back(hashes::Part("Couplet-arith-transcript"));
        parts.push_back(hashes::Part(sid_arith_));
        parts.push_back(hashes::Part(cbytes));
        for (const bytes& b : store) parts.push_back(hashes::Part(b));
        const hashes::hash32 transcript = hashes::framed_sha256(parts);
        // The coefficients are tossed only now: every opened value is fixed and both
        // parties agree on the list (it is part of the label of the coin toss).
        static const char coin_prefix[] = "arithcheck|";
        bytes coin_label(coin_prefix, coin_prefix + sizeof coin_prefix - 1);
        coin_label.insert(coin_label.end(), cbytes.begin(), cbytes.end());
        coin_label.insert(coin_label.end(), transcript.begin(), transcript.end());
        const bytes seed = hashes::coin_toss(*ch_, sid_arith_, coin_label, 32, sys_);
        struct Own {
            unsigned bits;
            u128 sent;  // the value committed to (an honest party sends its zeta)
        };
        std::vector<Own> own;
        bytes msg;
        for (const std::pair<const unsigned, std::vector<Opening>>& grp : groups) {
            const unsigned bits = grp.first;
            const std::vector<Opening>& items = grp.second;
            const u128 mask = u128_mask(bits);
            const std::vector<uint64_t> chis = check_coefficients(seed, bits, items.size(), s_);
            u128 y = u128(0), acc = u128(0);
            for (size_t j = 0; j < items.size(); j++) {
                y = y + u128(chis[j]) * items[j].value;
                acc = acc + u128(chis[j]) * items[j].mac;
            }
            y = y & mask;
            const u128 zeta = (acc - u128(delta_) * y) & mask;
            const u128 sent = outgoing_zeta(bits, zeta, chis, items, counter) & mask;
            own.push_back(Own{bits, sent});
            put(msg, sent, share_bytes(bits));
        }
        static const char zeta_prefix[] = "arithcheck-zeta|";
        bytes zeta_label(zeta_prefix, zeta_prefix + sizeof zeta_prefix - 1);
        zeta_label.insert(zeta_label.end(), cbytes.begin(), cbytes.end());
        const bytes peer = hashes::committed_exchange(*ch_, sid_arith_, zeta_label, msg, sys_, msg.size());
        if (peer.size() != msg.size()) throw ProtocolAbort("ArithCheck: message has the wrong length");
        size_t pos = 0;
        bool ok = true;
        for (const Own& o : own) {
            const unsigned nb = share_bytes(o.bits);
            const u128 other = get(peer.data() + pos, nb);
            pos += nb;
            if (other != (other & u128_mask(o.bits))) throw ProtocolAbort("ArithCheck: value out of range");
            ok = ok && ((o.sent + other) & u128_mask(o.bits)) == u128(0);
        }
        if (!ok) throw ProtocolAbort("ArithCheck failed");
        // The check passed: only now are the covered openings accepted.
        last_opened_.clear();
        last_opened_.reserve(pending_.size());
        for (const Opening& o : pending_) last_opened_.push_back(o.value);
        pending_.clear();
    }

    int role_;
    Channel* ch_;
    bytes sid_, sid_arith_;
    unsigned s_;
    uint64_t delta_;
    abb::PhaseBook* book_;
    SystemRng sys_;
    bool dead_ = false;
    uint64_t openings_ = 0, checks_ = 0;
    std::vector<Opening> pending_;
    std::vector<u128> last_opened_;  // covered by the latest passed check
    std::function<void()> on_abort_;
};

}  // namespace spdz2k
}  // namespace couplet
