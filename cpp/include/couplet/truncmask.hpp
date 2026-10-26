// couplet/truncmask.hpp -- truncate-and-mask (DESIGN_M3.md section 3), both
// modes, and the openings of an inference session; the C++ counterpart of
// python/couplet/truncmask.py, with the same messages.
//
// Opener(role, channel, sid, checker, delta_share, book) bundles what one
// party needs to open values during inference:
//
//   open_deferred(shares)   one exchange of value shares, MAC check deferred
//                           (spdz2k::ArithChecker::open_deferred_many);
//   check()                 ArithCheck over every deferred opening;
//   open_sign_exact(shares) the release of public-model mode: check, open,
//                           check; refused before any message unless every
//                           sharing is marked sign_exact; returns
//                           to_signed(v mod 2^k, k), k = bits - s (the part of
//                           an opening the check authenticates);
//   open_masked(shares, masks)
//                           the release of private-model mode (correction C2
//                           of design/CHANGELOG_M3.md): check, add the output
//                           mask [[2^k rho]], open, check (open_final_many of
//                           the checker); returns to_signed(v mod 2^k, k);
//   open_shared(values, bits)
//                           commit-then-open of ADDITIVE shares without MAC
//                           (the first-layer pre-activations): one
//                           hashes::committed_exchange for the whole list under
//                           sid_infer = SHA-256(frame("Couplet-infer-sid", sid))
//                           and the label "tm|" || le64(counter), each value
//                           ceil(bits/8) bytes little-endian; a value >= 2^bits
//                           aborts.  Phase "tm-open".
//   public_share(c, bits)   [[c]] for a public constant.
//
// Site holds one site's material (ppml::SiteMaterial) and its planes; the
// planes are evaluated by the caller (or by truncate_and_mask for plane B).
//
// truncate_and_mask(opener, sites, p, threads) runs one layer on sharings
// p[j] over Z_{2^w}; truncate_and_mask_shared(opener, sites, p, threads) on
// additive integer shares (the first layer, reduced modulo 2^m):
//
//   mode round (one opening): open v = (p + rho') mod 2^m, m = f + n + s;
//     v_lo = v mod 2^(f+n), x_hat = v_lo >> f, v_l = v_lo mod 2^f; x_hat
//     masks x + [p mod 2^f > v_l] for plane A;
//   mode exact (two openings, f >= 1): open v as above; [[beta]] = suffix of
//     plane B above v_l (= [v_l < rho_l]) over Z_{2^(n+s)};
//     [[x]] = (v_lo >> f) - [[rho_h]] - [[beta]]; open x + r' over Z_{2^(n+s)};
//     x_hat = (x + r') mod 2^n masks the exact floor for plane A at r' mod 2^n.
//
// All sites of a layer share one exchange per opening.  plain_truncate(p, f,
// n, exact, v_l) is the value z that plane A masks (the mirror uses it).
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "abb.hpp"
#include "channel.hpp"
#include "circuits_ppml.hpp"
#include "fixedpoint.hpp"
#include "hashes.hpp"
#include "planes.hpp"
#include "rng.hpp"
#include "spdz2k.hpp"
#include "u128.hpp"

namespace couplet {
namespace truncmask {

using spdz2k::AShare;

const char* const PHASE_SHARED = "tm-open";
const char* const PHASE_PLANES = "plane-eval";

// ------------------------------------------------------------------ openings of one party
class Opener {
public:
    Opener(int role, Channel& ch, const bytes& sid, spdz2k::ArithChecker& checker, uint64_t delta_share,
           abb::PhaseBook* book = nullptr)
        : role_(role), ch_(&ch), sid_(sid), checker_(&checker), delta_(delta_share), book_(book), s_(checker.s()) {
        if ((role != 0 && role != 1) || ch.party() != role) throw std::invalid_argument("role must be 0 or 1 and match the channel endpoint");
        if (checker.role() != role) throw std::invalid_argument("expected this party's ArithChecker");
        const hashes::hash32 h = hashes::framed_sha256({"Couplet-infer-sid", sid_});
        sid_infer_.assign(h.begin(), h.end());
    }
    virtual ~Opener() {}
    Opener(const Opener&) = delete;
    Opener& operator=(const Opener&) = delete;

    int role() const { return role_; }
    unsigned s() const { return s_; }
    uint64_t delta_share() const { return delta_; }
    uint64_t shared_openings() const { return shared_openings_; }
    spdz2k::ArithChecker& checker() { return *checker_; }
    Channel& channel() { return *ch_; }
    abb::PhaseBook* book() { return book_; }

    // [[c]] for a public constant c over Z_{2^bits}.
    AShare public_share(u128 c, unsigned bits) const {
        return AShare(u128(0), u128(0), bits).add_public(c & u128_mask(bits), role_, delta_);
    }

    // Commit-then-open of additive shares without MAC; returns the sums modulo 2^bits.
    std::vector<u128> open_shared(const std::vector<u128>& values, unsigned bits) {
        if (bits < 1 || bits > 128) throw std::invalid_argument("shares must lie in [0, 2^bits)");
        for (const u128& v : values)
            if (v != (v & u128_mask(bits))) throw std::invalid_argument("shares must lie in [0, 2^bits)");
        if (values.empty()) return std::vector<u128>();
        const unsigned nb = (bits + 7) / 8;
        bytes label = {'t', 'm', '|'};
        const std::array<uint8_t, 8> ctr = hashes::le64(shared_openings_);
        label.insert(label.end(), ctr.begin(), ctr.end());
        shared_openings_++;
        bytes msg;
        msg.reserve(values.size() * nb);
        for (const u128& v : values) put(msg, v, nb);
        bytes reply;
        {
            const int prev = book_ ? book_->current() : abb::PhaseBook::NONE;
            if (book_) book_->mark(PHASE_SHARED);
            try {
                reply = shared_exchange(label, msg);
            } catch (...) {
                if (book_) book_->mark(prev);
                throw;
            }
            if (book_) book_->mark(prev);
        }
        if (reply.size() != msg.size()) throw ProtocolAbort("truncate-and-mask: message has the wrong length");
        std::vector<u128> out(values.size());
        const u128 mask = u128_mask(bits);
        for (size_t j = 0; j < values.size(); j++) {
            const u128 peer = get(reply.data() + j * nb, nb);
            if (peer != (peer & mask)) throw ProtocolAbort("truncate-and-mask: share out of range");
            out[j] = (values[j] + peer) & mask;
        }
        return out;
    }

    // One exchange of value shares; the representatives, verified at the next check().
    std::vector<u128> open_deferred(const std::vector<AShare>& shares) { return checker_->open_deferred_many(shares); }
    void check() { checker_->check(); }

    // Release sign-exact values: check, open, check; returns to_signed(v mod 2^k, k).
    std::vector<int64_t> open_sign_exact(const std::vector<AShare>& shares) {
        for (const AShare& sh : shares) {
            if (!sh.sign_exact) throw std::invalid_argument("only sharings marked sign_exact are released without an output mask");
            if (sh.bits <= s_ || sh.bits - s_ > 64) throw std::invalid_argument("a released sharing needs s < bits <= s + 64");
        }
        if (shares.empty()) return std::vector<int64_t>();
        checker_->check();
        const std::vector<u128> opened = checker_->open_deferred_many(shares);
        checker_->check();
        std::vector<int64_t> out(shares.size());
        for (size_t j = 0; j < shares.size(); j++) out[j] = fixedpoint::to_signed(opened[j], shares[j].bits - s_);
        return out;
    }

    // Release with output masks: check, add mask, open, check; returns to_signed(v mod 2^k, k).
    std::vector<int64_t> open_masked(const std::vector<AShare>& shares, const std::vector<AShare>& masks) {
        if (masks.size() != shares.size()) throw std::invalid_argument("one output mask per released sharing is required");
        std::vector<AShare> plain;
        std::vector<const AShare*> mp;
        for (size_t j = 0; j < shares.size(); j++) {
            const AShare& sh = shares[j];
            const AShare& mk = masks[j];
            if (sh.bits <= s_ || sh.bits - s_ > 64 || mk.bits != sh.bits)
                throw std::invalid_argument("a released sharing needs s < bits <= s + 64 and a mask over the same modulus");
            const u128 low = u128_mask(sh.bits - s_);
            if ((mk.value & low) != u128(0) || (mk.mac & low) != u128(0))
                throw std::invalid_argument("output mask shares must be multiples of 2^k");
            plain.push_back(sh.with_exact(false));  // every sharing is opened through its mask
        }
        for (const AShare& mk : masks) mp.push_back(&mk);
        if (shares.empty()) return std::vector<int64_t>();
        const std::vector<u128> payloads = checker_->open_final_many(plain, mp);
        std::vector<int64_t> out(shares.size());
        for (size_t j = 0; j < shares.size(); j++) out[j] = fixedpoint::to_signed(payloads[j], shares[j].bits - s_);
        return out;
    }

protected:
    // The commit-then-open exchange of one first-layer opening (a deviating party overrides it).
    virtual bytes shared_exchange(const bytes& label, const bytes& msg) {
        return hashes::committed_exchange(*ch_, sid_infer_, label, msg, sys_, msg.size());
    }
    const bytes& sid_infer() const { return sid_infer_; }

private:
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

    int role_;
    Channel* ch_;
    bytes sid_, sid_infer_;
    spdz2k::ArithChecker* checker_;
    uint64_t delta_;
    abb::PhaseBook* book_;
    unsigned s_;
    uint64_t shared_openings_ = 0;
    SystemRng sys_;
};

// ------------------------------------------------------------------ sites
struct Site {
    ppml::SiteMaterial mat;
    planes::Plane plane_a, plane_b;

    Site() {}
    explicit Site(ppml::SiteMaterial m) : mat(std::move(m)) {}

    unsigned n() const { return mat.n(); }
    unsigned f() const { return mat.f(); }
    unsigned s() const { return mat.s(); }
    unsigned m() const { return mat.m(); }
    bool two_openings() const { return mat.two_openings(); }
    AShare rp_share() const { return AShare(mat.rp.value, mat.rp.tag, mat.m()); }
    AShare rh_share() const { return AShare(mat.rh.value, mat.rh.tag, mat.n() + mat.s()); }
    AShare r2_share() const { return AShare(mat.r2.value, mat.r2.tag, mat.n() + mat.s()); }
    void release_planes() {
        plane_a.release();
        plane_b.release();
    }
};

// The public outcome of one site.
struct TruncResult {
    uint64_t x_hat = 0;   // masked input of plane A
    u128 v = u128(0);     // opened p + rho' mod 2^m
    uint64_t v_l = 0;     // threshold (mode round; recorded in both modes)
    bool exact = false;   // x_hat masks the exact floor
    bool has2 = false;    // second opening (mode exact)
    u128 opened2 = u128(0);
};

// The value z that plane A masks after truncate-and-mask of the integer p
// (an i128 view): (p >> f) mod 2^n, plus [p mod 2^f > v_l] in mode round.
inline uint64_t plain_truncate(u128 p, unsigned f, unsigned n, bool exact, uint64_t v_l = 0) {
    const uint64_t x = fixedpoint::truncate(p, f, n);
    if (exact || f == 0) return x;
    const uint64_t frac = u128_lo(p & u128_mask(f));
    const uint64_t beta = frac > v_l ? 1 : 0;
    return (x + beta) & ((n >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << n) - 1));
}

// Evaluate plane A (or plane B) of the given sites on `threads` threads.
inline void evaluate_planes(int role, std::vector<Site*>& sites, bool plane_b, unsigned threads, bool with_prefix) {
    std::vector<const fss::DpfKey*> keys;
    for (Site* s : sites) keys.push_back(plane_b ? &s->mat.plane_b : &s->mat.plane_a);
    std::vector<planes::Plane> out = planes::evaluate_planes(role, keys, threads, with_prefix);
    for (size_t i = 0; i < sites.size(); i++) (plane_b ? sites[i]->plane_b : sites[i]->plane_a) = std::move(out[i]);
}

namespace detail {
inline void check_layer(const std::vector<Site*>& sites) {
    if (sites.empty()) return;
    const Site& a = *sites[0];
    for (const Site* s : sites)
        if (s->n() != a.n() || s->f() != a.f() || s->s() != a.s() || s->two_openings() != a.two_openings())
            throw std::invalid_argument("all sites of a layer must share n, f, s and the mode");
}

inline std::vector<TruncResult> finish(Opener& op, std::vector<Site*>& sites, const std::vector<u128>& v_all,
                                       unsigned threads) {
    const Site& first = *sites[0];
    const unsigned n = first.n(), f = first.f(), s = first.s();
    const u128 mask_lo = u128_mask(f + n);
    const uint64_t mask_f = f ? u128_lo(u128_mask(f)) : 0;
    std::vector<TruncResult> res(sites.size());
    if (!first.two_openings()) {
        for (size_t j = 0; j < sites.size(); j++) {
            const u128 v_lo = v_all[j] & mask_lo;
            res[j].v = v_all[j];
            res[j].x_hat = u128_lo(v_lo >> f);
            res[j].v_l = u128_lo(v_lo) & mask_f;
            res[j].exact = f == 0;
        }
        return res;
    }
    // mode exact with f >= 1: plane B of every site (with prefix tables), [[x]] and the second opening
    {
        abb::PhaseBook* book = op.book();
        const int prev = book ? book->current() : abb::PhaseBook::NONE;
        if (book) book->mark(PHASE_PLANES);
        evaluate_planes(op.role(), sites, true, threads, true);
        if (book) book->mark(prev);
    }
    std::vector<AShare> x_shares;
    for (size_t j = 0; j < sites.size(); j++) {
        Site& site = *sites[j];
        const u128 v_lo = v_all[j] & mask_lo;
        const uint64_t v_l = u128_lo(v_lo) & mask_f;
        const AShare beta = site.plane_b.suffix(v_l + 1, n + s);  // [v_l < rho_l]
        const AShare x = op.public_share(v_lo >> f, n + s) - site.rh_share() - beta;
        x_shares.push_back(x + site.r2_share());
        res[j].v = v_all[j];
        res[j].v_l = v_l;
        site.plane_b.release();
    }
    const std::vector<u128> opened2 = op.open_deferred(x_shares);
    for (size_t j = 0; j < sites.size(); j++) {
        res[j].x_hat = u128_lo(opened2[j] & u128_mask(n));
        res[j].exact = true;
        res[j].has2 = true;
        res[j].opened2 = opened2[j];
    }
    return res;
}
}  // namespace detail

// Truncate-and-mask of one layer on authenticated pre-activations over Z_{2^w}.
inline std::vector<TruncResult> truncate_and_mask(Opener& op, std::vector<Site*>& sites, const std::vector<AShare>& p,
                                                  unsigned threads = 1) {
    if (sites.size() != p.size()) throw std::invalid_argument("one pre-activation per site is required");
    if (sites.empty()) return std::vector<TruncResult>();
    detail::check_layer(sites);
    const unsigned n = sites[0]->n(), f = sites[0]->f(), s = sites[0]->s(), m = sites[0]->m();
    std::vector<AShare> masked;
    for (size_t j = 0; j < p.size(); j++) {
        if (p[j].bits < m) throw std::invalid_argument("pre-activations must be sharings over at least 2^m");
        masked.push_back(p[j].reduce(f + n, s) + sites[j]->rp_share());
    }
    const std::vector<u128> v_all = op.open_deferred(masked);
    return detail::finish(op, sites, v_all, threads);
}

// Truncate-and-mask of the first layer: additive shares of p (i128 views), no MAC.
inline std::vector<TruncResult> truncate_and_mask_shared(Opener& op, std::vector<Site*>& sites, const std::vector<u128>& p,
                                                         unsigned threads = 1) {
    if (sites.size() != p.size()) throw std::invalid_argument("one pre-activation per site is required");
    if (sites.empty()) return std::vector<TruncResult>();
    detail::check_layer(sites);
    const unsigned m = sites[0]->m();
    std::vector<u128> masked(p.size());
    for (size_t j = 0; j < p.size(); j++) masked[j] = (p[j] + sites[j]->mat.rp.value) & u128_mask(m);
    const std::vector<u128> v_all = op.open_shared(masked, m);
    return detail::finish(op, sites, v_all, threads);
}

}  // namespace truncmask
}  // namespace couplet
