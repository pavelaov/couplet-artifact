// couplet/baseot.hpp -- base oblivious transfer: Masny-Rindal OT over the
// prime-order subgroup of Edwards25519 (DESIGN.md Section 7.1).  Same message
// flow and hash inputs as the Python module couplet/prep/baseot.py.
//
// One instance, with G the base point:
//   Receiver, choice c:  b <- Z_q, M = b G;  r_{1-c} = t G for random t;
//                        r_c = M - H_c(r_{1-c});  send (r_0, r_1).
//   Sender:              a <- Z_q, A = a G;  for j in {0, 1}:
//                        M_j = r_j + H_j(r_{1-j}),  k_j = KDF(sid, d, i, j, a M_j);
//                        send A.
//   Receiver:            k_c = KDF(sid, d, i, c, b A).
//
//   H_j(P) = hash_to_group("mr" || d || j || le32(i) || sid, enc(P)),
//   KDF    = hashes::kdf("baseot", sid, d, le32(i), j, enc(point))   (16 bytes),
// so every hash is separated by (sid, direction d, instance i, branch j).  A
// fresh scalar a is used for every instance.
//
// base_ot() runs KAPPA = 128 instances in EACH direction in two exchanges.
// Direction d is the direction of the later correlated-OT extension in which
// P_d is the COT sender: there P_d is the base-OT RECEIVER and its choice
// bits are the bits of its MAC key share Delta^B_d, while P_{1-d} is the
// base-OT SENDER and obtains both seeds of every instance.  Party `role`
//   * receives in direction role      -> chosen: the seeds k_j^{c_j};
//   * sends in direction 1 - role     -> seeds0, seeds1: the pairs (k_j^0, k_j^1).
//
// Every point received from the peer is decoded with
// ed25519::decompress_in_subgroup (strict encoding, curve equation, order-q
// subgroup, not the identity); anything else throws ProtocolAbort.  Scalar
// multiplication is variable time; side channels are out of scope (7.1).
//
// Communication: 64 bytes per instance in the first exchange and 32 in the
// second, i.e. 128 * 96 = 12,288 bytes per party; two rounds.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "channel.hpp"
#include "ed25519.hpp"
#include "gf128.hpp"
#include "hashes.hpp"
#include "params.hpp"
#include "rng.hpp"
#include "shares.hpp"

namespace couplet {
namespace prep {

const size_t POINT_BYTES = 32;

namespace detail {

// Tag of H_j: "mr" || direction || branch || le32(index) || sid (injective:
// the variable-length sid comes last).
inline bytes ot_h_tag(const bytes& sid, int direction, uint32_t index, int branch) {
    bytes tag;
    tag.reserve(8 + sid.size());
    tag.push_back((uint8_t)'m');
    tag.push_back((uint8_t)'r');
    tag.push_back((uint8_t)direction);
    tag.push_back((uint8_t)branch);
    for (int i = 0; i < 4; i++) tag.push_back((uint8_t)(index >> (8 * i)));
    tag.insert(tag.end(), sid.begin(), sid.end());
    return tag;
}

inline ed25519::Point ot_hash_point(const bytes& sid, int direction, uint32_t index, int branch,
                                    const uint8_t enc[32]) {
    const bytes tag = ot_h_tag(sid, direction, index, branch);
    return ed25519::hash_to_group(tag.data(), tag.size(), enc, POINT_BYTES);
}

// k = kdf("baseot", sid, direction, le32(index), branch, enc(point)).
inline gf128 ot_seed(const bytes& sid, int direction, uint32_t index, int branch, const ed25519::Point& point) {
    const uint8_t d = (uint8_t)direction, j = (uint8_t)branch;
    std::array<uint8_t, 4> idx;
    for (int i = 0; i < 4; i++) idx[(size_t)i] = (uint8_t)(index >> (8 * i));
    const ed25519::bytes32 enc = ed25519::compress(point);
    const hashes::seed16 k =
        hashes::kdf({"baseot", sid, hashes::Part(&d, 1), idx, hashes::Part(&j, 1), enc});
    return gf128::from_bytes(k.data());
}

// Decode a received point: ProtocolAbort unless it is a non-identity element
// of the subgroup of order q in canonical encoding.
inline ed25519::Point ot_decode(const uint8_t enc[32]) {
    ed25519::Point p;
    if (!ed25519::decompress_in_subgroup(p, enc, false)) throw ProtocolAbort("base OT: invalid point");
    return p;
}

}  // namespace detail

// Base-OT receiver of one direction (the later COT sender of that direction).
class OtReceiver {
public:
    // choices: one bit per instance.
    OtReceiver(const bytes& sid, int direction, const BitVec& choices, RandomSource& rng)
        : sid_(sid), direction_(direction), choices_(choices), rng_(&rng) {
        if (direction != 0 && direction != 1) throw std::invalid_argument("direction must be 0 or 1");
        check_bits(choices_);
    }

    // The first flight: r_0 || r_1 for every instance.
    bytes flight1() {
        bytes out;
        out.reserve(2 * POINT_BYTES * choices_.size());
        secrets_.clear();
        for (size_t i = 0; i < choices_.size(); i++) {
            const int c = choices_[i];
            ed25519::scalar32 b;
            ed25519::bytes32 enc_other, enc_c;
            for (;;) {
                b = ed25519::random_scalar(*rng_);
                const ed25519::scalar32 t = ed25519::random_scalar(*rng_);
                enc_other = ed25519::compress(ed25519::mul_base(t));  // r_{1-c} = t G
                const ed25519::Point h = detail::ot_hash_point(sid_, direction_, (uint32_t)i, c, enc_other.data());
                const ed25519::Point r_c = ed25519::sub(ed25519::mul_base(b), h);  // r_c = M - H_c(r_{1-c})
                if (!ed25519::is_identity(r_c)) {  // the sender rejects the identity
                    enc_c = ed25519::compress(r_c);
                    break;
                }
            }
            const ed25519::bytes32& first = (c == 0) ? enc_c : enc_other;
            const ed25519::bytes32& second = (c == 0) ? enc_other : enc_c;
            out.insert(out.end(), first.begin(), first.end());
            out.insert(out.end(), second.begin(), second.end());
            secrets_.push_back(b);
        }
        return out;
    }

    // Process the sender's flight (A per instance); returns the seeds k_i^{c_i}.
    GfVec finish(const bytes& flight2) {
        const size_t n = choices_.size();
        if (secrets_.size() != n) throw std::logic_error("base OT: flight1 has not been produced");
        if (flight2.size() != POINT_BYTES * n) throw ProtocolAbort("base OT: second flight has the wrong length");
        GfVec keys(n);
        for (size_t i = 0; i < n; i++) {
            const ed25519::Point big_a = detail::ot_decode(flight2.data() + POINT_BYTES * i);
            keys[i] = detail::ot_seed(sid_, direction_, (uint32_t)i, choices_[i], ed25519::mul(big_a, secrets_[i]));
        }
        for (ed25519::scalar32& s : secrets_) s.fill(0);
        secrets_.clear();
        return keys;
    }

private:
    bytes sid_;
    int direction_;
    BitVec choices_;
    RandomSource* rng_;
    std::vector<ed25519::scalar32> secrets_;
};

// Base-OT sender of one direction (the later COT receiver of that direction).
class OtSender {
public:
    OtSender(const bytes& sid, int direction, size_t count, RandomSource& rng)
        : sid_(sid), direction_(direction), count_(count), rng_(&rng) {
        if (direction != 0 && direction != 1) throw std::invalid_argument("direction must be 0 or 1");
    }

    // Process the receiver's flight; returns the second flight and fills
    // seeds0 / seeds1 with the pairs (k_i^0, k_i^1).
    bytes flight2(const bytes& flight1) {
        if (flight1.size() != 2 * POINT_BYTES * count_) throw ProtocolAbort("base OT: first flight has the wrong length");
        bytes out;
        out.reserve(POINT_BYTES * count_);
        seeds0.assign(count_, gf128());
        seeds1.assign(count_, gf128());
        for (size_t i = 0; i < count_; i++) {
            const uint8_t* enc0 = flight1.data() + 2 * POINT_BYTES * i;
            const uint8_t* enc1 = enc0 + POINT_BYTES;
            const ed25519::Point r0 = detail::ot_decode(enc0);
            const ed25519::Point r1 = detail::ot_decode(enc1);
            const ed25519::scalar32 a = ed25519::random_scalar(*rng_);
            const ed25519::bytes32 enc_a = ed25519::compress(ed25519::mul_base(a));
            out.insert(out.end(), enc_a.begin(), enc_a.end());
            // M_j = r_j + H_j(r_{1-j})
            const ed25519::Point m0 = ed25519::add(r0, detail::ot_hash_point(sid_, direction_, (uint32_t)i, 0, enc1));
            const ed25519::Point m1 = ed25519::add(r1, detail::ot_hash_point(sid_, direction_, (uint32_t)i, 1, enc0));
            seeds0[i] = detail::ot_seed(sid_, direction_, (uint32_t)i, 0, ed25519::mul(m0, a));
            seeds1[i] = detail::ot_seed(sid_, direction_, (uint32_t)i, 1, ed25519::mul(m1, a));
        }
        return out;
    }

    GfVec seeds0, seeds1;

private:
    bytes sid_;
    int direction_;
    size_t count_;
    RandomSource* rng_;
};

struct BaseOtResult {
    GfVec chosen;          // direction role: the seed k_j^{c_j} of every instance
    GfVec seeds0, seeds1;  // direction 1 - role: the pairs (k_j^0, k_j^1)
};

// Run the base OTs of both directions (two rounds).
//   ch           this party's channel endpoint;
//   sid          session identifier;
//   role         this party's index b;
//   choice_bits  the KAPPA bits of Delta^B_b (bit j = coefficient of x^j);
//   rng          this party's randomness;
//   cheat        test hooks flight1 / flight2 (default honest).
inline BaseOtResult base_ot(Channel& ch, const bytes& sid, int role, const BitVec& choice_bits, RandomSource& rng,
                            const Cheat* cheat = nullptr) {
    check_role(role);
    if (choice_bits.size() != KAPPA) throw std::invalid_argument("exactly KAPPA choice bits are required");
    OtReceiver receiver(sid, role, choice_bits, rng);
    OtSender sender(sid, 1 - role, KAPPA, rng);

    bytes flight1 = receiver.flight1();
    if (cheat && cheat->flight1) flight1 = cheat->flight1(flight1);
    const bytes peer1 = ch.exchange(flight1, 2 * POINT_BYTES * KAPPA, "base OT: first flight has the wrong length");

    bytes flight2 = sender.flight2(peer1);
    if (cheat && cheat->flight2) flight2 = cheat->flight2(flight2);
    const bytes peer2 = ch.exchange(flight2, POINT_BYTES * KAPPA, "base OT: second flight has the wrong length");

    BaseOtResult res;
    res.chosen = receiver.finish(peer2);
    res.seeds0 = std::move(sender.seeds0);
    res.seeds1 = std::move(sender.seeds1);
    return res;
}

}  // namespace prep
}  // namespace couplet
