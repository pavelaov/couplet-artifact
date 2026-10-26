// couplet/abits.hpp -- authenticated bits from correlated OT (DESIGN.md
// Section 7.3); the C++ counterpart of couplet/prep/abits.py.
//
// A COT row of direction d is a random BDOZ bit of P_{1-d} under the key
// Delta^B_d of P_d.  Party b is the COT sender of direction b and the COT
// receiver of direction 1 - b, so
//   * a random <x> uses one row of each direction: party b obtains its share
//     x_b with M[x_b] (as receiver) and the key K[x_{1-b}] (as sender), and
//     spdz_shares turns these into the SPDZ-style MAC share;
//   * an input mask of P_b uses one row of direction 1 - b only.
//
// Key constraint: lsb(Delta^B_0) = 1 and lsb(Delta^B_1) = 0, so that
// lsb(Delta^B) = 1 (needed by the leaky AND of Section 7.4).  sample_delta
// samples a key share with the right least significant bit and AbitSource
// refuses any other.  A corrupted party that ignores the constraint only
// produces AND triples that fail the triple consistency check.
//
// AbitSource is the party-local source used by the rest of the pipeline:
//   random_bdoz(n)      n random bits in BDOZ form (what the leaky AND needs);
//   random_shares(n)    n random <x>;
//   input_masks(o, n)   n input masks of P_o.
// Requests are served in chunks of at most batch_rows rows per extension call
// (bounded memory and message size).  Accounting: the extension flights and
// the column expansion are booked under the phase "cotext", the local
// conversion to shares under "abit".
//
// Rounds: three per chunk for random bits (both directions in parallel).
// Input masks run one direction; consecutive chunks share a flight (the
// check of chunk c and the U message of chunk c + 1 travel together), i.e.
// 3 + 2 (chunks - 1) flights.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "channel.hpp"
#include "cotext.hpp"
#include "gf128.hpp"
#include "params.hpp"
#include "rng.hpp"
#include "shares.hpp"

namespace couplet {
namespace prep {

const size_t DEFAULT_BATCH_ROWS = (size_t)1 << 20;

// lsb of Delta^B_role demanded by DESIGN.md 7.3: 1 for P_0, 0 for P_1.
inline unsigned required_lsb(int role) {
    check_role(role);
    return (unsigned)(1 - role);
}

// Validate a MAC key share (least significant bit); returns it.
inline gf128 check_delta(int role, const gf128& delta) {
    if ((delta.lo & 1) != required_lsb(role))
        throw std::invalid_argument("lsb(Delta^B_0) must be 1 and lsb(Delta^B_1) must be 0");
    return delta;
}

// Uniform Delta^B_role subject to the lsb constraint (127 bits of entropy).
inline gf128 sample_delta(int role, RandomSource& rng) {
    const unsigned lsb = required_lsb(role);
    gf128 d = rng.gf();
    d.lo = (d.lo & ~1ull) | lsb;
    return d;
}

// Party-local source of authenticated bits backed by the two COT directions.
class AbitSource : public BdozSource {
public:
    AbitSource(Channel& ch, int role, const gf128& delta, CotSender& sender, CotReceiver& receiver,
               size_t batch_rows = DEFAULT_BATCH_ROWS, PhaseMeter* meter = nullptr)
        : ch_(&ch), role_(role), delta_(check_delta(role, delta)), sender_(&sender), receiver_(&receiver),
          batch_rows_(batch_rows), meter_(meter), rows_both_(0) {
        if (sender.direction() != role || receiver.direction() != 1 - role)
            throw std::invalid_argument("party b sends in direction b and receives in direction 1 - b");
        if (sender.delta() != delta_) throw std::invalid_argument("the COT sender must use this party's Delta^B share");
        if (batch_rows < 1 || batch_rows > COT_MAX_ROWS) throw std::invalid_argument("batch_rows out of range");
        rows_mask_[0] = rows_mask_[1] = 0;
    }

    int role() const { return role_; }
    const gf128& delta() const { return delta_; }
    size_t batch_rows() const { return batch_rows_; }
    uint64_t rows_both() const { return rows_both_; }            // rows consumed in both directions
    uint64_t rows_mask(int owner) const { return rows_mask_[owner & 1]; }  // rows consumed for masks, by owner

    // The test hook passed to CotReceiver::start of the first chunk of the
    // next random_bdoz / random_shares call (nullptr = honest).
    void set_cheat(const Cheat* cheat) { cheat_ = cheat; }

    // n random BDOZ bits of each party (three rounds per chunk).
    BdozPair random_bdoz(size_t n) override {
        BdozPair out;
        out.own = Bdoz(n);
        out.keys = Keys(n);
        mark_phase(meter_, PH_COTEXT);
        for (size_t done = 0; done < n;) {
            const size_t step = (n - done < batch_rows_) ? (n - done) : batch_rows_;
            extend_both(*ch_, *sender_, *receiver_, step, out.own.bits.data() + done, out.own.macs.data() + done,
                        out.keys.keys.data() + done, done == 0 ? cheat_ : nullptr);
            done += step;
        }
        mark_phase(meter_, PH_ABIT);
        rows_both_ += n;
        return out;
    }

    // n random <x>.  With keep != nullptr the BDOZ components are stored there.
    Shares random_shares(size_t n, BdozPair* keep = nullptr) {
        BdozPair p = random_bdoz(n);
        Shares s;
        if (keep) {
            s = spdz_shares(role_, delta_, p.own, p.keys);
            *keep = std::move(p);
        } else {
            spdz_macs_inplace(delta_, p.own.bits.data(), p.own.macs.data(), p.keys.keys.data(), n);
            s.bits = std::move(p.own.bits);
            s.macs = std::move(p.own.macs);
        }
        return s;
    }

    // n input masks of P_owner: shares and the bits at the owner, shares only
    // at the other party.  Uses rows of direction 1 - owner.
    InputMasks input_masks(int owner, size_t n) {
        check_role(owner);
        InputMasks out;
        out.has_bits = (owner == role_);
        mark_phase(meter_, PH_COTEXT);
        if (owner == role_) {
            Bdoz own(n);
            for (size_t done = 0; done < n;) {
                const size_t step = (n - done < batch_rows_) ? (n - done) : batch_rows_;
                ch_->send(receiver_->start(step));
                ch_->flight();
                const bytes seed = ch_->recv(COT_SEED_BYTES, "COT: challenge seed has the wrong length");
                ch_->flight();
                ch_->send(receiver_->answer(seed, own.bits.data() + done, own.macs.data() + done));
                done += step;
                if (done == n) ch_->flight();  // otherwise the next U joins this flight
            }
            mark_phase(meter_, PH_ABIT);
            out.shares = mask_owner(delta_, own);
            out.bits = out.shares.bits;
        } else {
            Keys keys(n);
            bool pending = false;  // a check message of the previous chunk is outstanding
            bytes check;
            size_t prev = 0;
            for (size_t done = 0; done < n;) {
                const size_t step = (n - done < batch_rows_) ? (n - done) : batch_rows_;
                if (pending) check = ch_->recv(COT_CHECK_BYTES, "COT: check message has the wrong length");
                const bytes u_msg = ch_->recv(u_message_bytes(step), "COT: U message has the wrong length");
                ch_->flight();
                if (pending) {
                    // verify the previous chunk before answering the next one
                    sender_->verify(check, keys.keys.data() + prev);
                    pending = false;
                }
                ch_->send(sender_->absorb(step, u_msg));
                ch_->flight();
                prev = done;
                pending = true;
                done += step;
            }
            if (pending) {
                check = ch_->recv(COT_CHECK_BYTES, "COT: check message has the wrong length");
                ch_->flight();
                sender_->verify(check, keys.keys.data() + prev);
            }
            mark_phase(meter_, PH_ABIT);
            out.shares = mask_other(keys);
        }
        rows_mask_[owner] += n;
        return out;
    }

private:
    Channel* ch_;
    int role_;
    gf128 delta_;
    CotSender* sender_;
    CotReceiver* receiver_;
    size_t batch_rows_;
    PhaseMeter* meter_;
    const Cheat* cheat_ = nullptr;
    uint64_t rows_both_;
    uint64_t rows_mask_[2];
};

}  // namespace prep
}  // namespace couplet
