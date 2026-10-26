// couplet/beaver.hpp -- products with Beaver triples and authenticated input
// (private-model mode; DESIGN_M3.md sections 2 and 3); the C++ counterpart of
// python/couplet/beaver.py, with the same messages.
//
// All sharings are spdz2k::AShare over Z_{2^w}; every opening goes through a
// truncmask::Opener with the MAC check deferred.
//
//   products(op, xs, ys, triples)   [[x_j y_j]] = [[c]] + d [[b]] + e [[a]] + d e,
//                                   d = x - a and e = y - b opened in ONE exchange
//                                   (all d first, then all e)
//   matvec(op, W, x, mt)            [[W x]] with a matrix-vector triple (A, b, c = A b):
//                                   D = W - A (row-major) and e = x - b opened in one
//                                   exchange; [[W x]]_i = [[c_i]] + sum_j D_ij [[b_j]]
//                                   + sum_j e_j [[A_ij]] + sum_j D_ij e_j
//   matvec_tiled(op, W, rows, cols, x, tiles)
//                                   [[W x]] as the sum of tile products (tiled private-model
//                                   plans, task R-TILE): `tiles` in the plan's row-major tile
//                                   order, each with its own triple; per tile D_t = W[tile] - A_t
//                                   (row-major) then e_t = x[c0 .. c0 + w_t) - b_t (its own
//                                   vector mask), all opened in ONE exchange in tile order; the
//                                   tiles must partition W
//   merge_inmasks(parts)            one InMaskMaterial from the instances of a tiled plan
//   exchange_inputs(op, mine, values, peer)
//                                   authenticated input of both parties' values in ONE
//                                   exchange (phase "input-open"): the owner of the masks
//                                   [[rho_j]] sends d_j = x_j - rho_j mod 2^w,
//                                   ceil(w/8) bytes little-endian each, and both set
//                                   [[x_j]] = [[rho_j]] + d_j.  Nothing is sent when
//                                   neither party inputs anything.
//
// Results carry no exactness flags.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "circuits_ppml.hpp"
#include "spdz2k.hpp"
#include "truncmask.hpp"
#include "u128.hpp"

namespace couplet {
namespace beaver {

using spdz2k::AShare;

const char* const PHASE_INPUT = "input-open";

struct Triple {
    AShare a, b, c;
};

struct MatTriple {
    size_t rows = 0, cols = 0;
    std::vector<AShare> A;  // rows x cols, row-major
    std::vector<AShare> b, c;
};

inline std::vector<Triple> triple_shares(const ppml::TripleMaterial& m) {
    const unsigned w = m.desc.width();
    std::vector<Triple> out;
    for (size_t j = 0; j < m.a.size(); j++)
        out.push_back(Triple{AShare(m.a[j].value, m.a[j].tag, w), AShare(m.b[j].value, m.b[j].tag, w),
                             AShare(m.c[j].value, m.c[j].tag, w)});
    return out;
}

inline MatTriple mattriple_shares(const ppml::MatTripleMaterial& m) {
    const unsigned w = m.desc.width();
    MatTriple t;
    t.rows = m.rows();
    t.cols = m.cols();
    for (const protocol::SharePair& p : m.a) t.A.push_back(AShare(p.value, p.tag, w));
    for (const protocol::SharePair& p : m.b) t.b.push_back(AShare(p.value, p.tag, w));
    for (const protocol::SharePair& p : m.c) t.c.push_back(AShare(p.value, p.tag, w));
    if (t.A.size() != t.rows * t.cols || t.b.size() != t.cols || t.c.size() != t.rows)
        throw std::invalid_argument("malformed matrix triple");
    return t;
}

// One tile of a tiled matrix-vector product: rows [r0, r0 + t.rows), columns [c0, c0 + t.cols).
struct TileTriple {
    size_t r0 = 0, c0 = 0;
    MatTriple t;
};

namespace detail {
inline unsigned same_width(const std::vector<AShare>& v, unsigned w = 0) {
    for (const AShare& sh : v) {
        if (w == 0) w = sh.bits;
        else if (sh.bits != w) throw std::invalid_argument("all sharings must be over the same modulus");
    }
    return w;
}
}  // namespace detail

// [[x_j y_j]] for all j with one opening exchange.
inline std::vector<AShare> products(truncmask::Opener& op, const std::vector<AShare>& xs, const std::vector<AShare>& ys,
                                    const std::vector<Triple>& triples) {
    if (xs.size() != ys.size() || xs.size() != triples.size()) throw std::invalid_argument("one triple per product is required");
    if (xs.empty()) throw std::invalid_argument("expected at least one sharing");
    const unsigned w = detail::same_width(xs);
    detail::same_width(ys, w);
    std::vector<AShare> open;
    for (size_t j = 0; j < xs.size(); j++) {
        if (triples[j].a.bits != w || triples[j].b.bits != w || triples[j].c.bits != w)
            throw std::invalid_argument("all sharings must be over the same modulus");
        open.push_back(xs[j] - triples[j].a);
    }
    for (size_t j = 0; j < ys.size(); j++) open.push_back(ys[j] - triples[j].b);
    const std::vector<u128> opened = op.open_deferred(open);
    const size_t count = xs.size();
    std::vector<AShare> out;
    for (size_t j = 0; j < count; j++) {
        const u128 d = opened[j], e = opened[count + j];
        out.push_back((triples[j].c + triples[j].b.scale(d) + triples[j].a.scale(e)).add_public(d * e, op.role(), op.delta_share()));
    }
    return out;
}

// [[W x]] for a shared matrix W (rows x cols, row-major) and a shared vector x.
inline std::vector<AShare> matvec(truncmask::Opener& op, const std::vector<AShare>& W, const std::vector<AShare>& x,
                                  const MatTriple& mt) {
    const size_t rows = mt.rows, cols = mt.cols;
    if (W.size() != rows * cols || x.size() != cols)
        throw std::invalid_argument("the matrix, the vector and the triple have incompatible shapes");
    const unsigned w = detail::same_width(x);
    detail::same_width(W, w);
    std::vector<AShare> open;
    open.reserve(rows * cols + cols);
    for (size_t i = 0; i < rows * cols; i++) open.push_back(W[i] - mt.A[i]);
    for (size_t j = 0; j < cols; j++) open.push_back(x[j] - mt.b[j]);
    const std::vector<u128> opened = op.open_deferred(open);
    const u128 mask = u128_mask(w);
    std::vector<AShare> out;
    for (size_t i = 0; i < rows; i++) {
        u128 value = mt.c[i].value, mac = mt.c[i].mac, cst = u128(0);
        for (size_t j = 0; j < cols; j++) {
            const u128 d = opened[i * cols + j], e = opened[rows * cols + j];
            value = value + mt.b[j].value * d + mt.A[i * cols + j].value * e;
            mac = mac + mt.b[j].mac * d + mt.A[i * cols + j].mac * e;
            cst = cst + d * e;
        }
        out.push_back(AShare(value & mask, mac & mask, w).add_public(cst & mask, op.role(), op.delta_share()));
    }
    return out;
}

// [[W x]] for a shared rows x cols matrix W (row-major) as the sum of tile products.
inline std::vector<AShare> matvec_tiled(truncmask::Opener& op, const std::vector<AShare>& W, size_t rows, size_t cols,
                                        const std::vector<AShare>& x, const std::vector<TileTriple>& tiles) {
    if (rows < 1 || W.size() != rows * cols || x.size() != cols)
        throw std::invalid_argument("the matrix and the vector have incompatible shapes");
    const unsigned w = detail::same_width(x);
    detail::same_width(W, w);
    std::vector<uint8_t> covered(rows * cols, 0);
    size_t n_open = 0;
    for (const TileTriple& tt : tiles) {
        const MatTriple& t = tt.t;
        if (t.rows < 1 || t.cols < 1 || tt.r0 + t.rows > rows || tt.c0 + t.cols > cols || t.A.size() != t.rows * t.cols ||
            t.b.size() != t.cols || t.c.size() != t.rows)
            throw std::invalid_argument("a tile does not fit the matrix");
        for (size_t i = 0; i < t.rows; i++)
            for (size_t j = 0; j < t.cols; j++) {
                uint8_t& c = covered[(tt.r0 + i) * cols + tt.c0 + j];
                if (c) throw std::invalid_argument("the tiles overlap");
                c = 1;
            }
        detail::same_width(t.A, w);
        detail::same_width(t.b, w);
        detail::same_width(t.c, w);
        n_open += t.rows * t.cols + t.cols;
    }
    for (uint8_t c : covered)
        if (!c) throw std::invalid_argument("the tiles do not cover the matrix");
    std::vector<AShare> open;
    open.reserve(n_open);
    for (const TileTriple& tt : tiles) {
        const MatTriple& t = tt.t;
        for (size_t i = 0; i < t.rows; i++)
            for (size_t j = 0; j < t.cols; j++) open.push_back(W[(tt.r0 + i) * cols + tt.c0 + j] - t.A[i * t.cols + j]);
        for (size_t j = 0; j < t.cols; j++) open.push_back(x[tt.c0 + j] - t.b[j]);
    }
    const std::vector<u128> opened = op.open_deferred(open);
    const u128 mask = u128_mask(w);
    std::vector<u128> value(rows, u128(0)), mac(rows, u128(0)), cst(rows, u128(0));
    size_t pos = 0;
    for (const TileTriple& tt : tiles) {
        const MatTriple& t = tt.t;
        const size_t base_e = pos + t.rows * t.cols;
        for (size_t i = 0; i < t.rows; i++) {
            const size_t r = tt.r0 + i;
            u128 v = t.c[i].value, m = t.c[i].mac, c = u128(0);
            for (size_t j = 0; j < t.cols; j++) {
                const u128 d = opened[pos + i * t.cols + j], e = opened[base_e + j];
                v = v + t.b[j].value * d + t.A[i * t.cols + j].value * e;
                m = m + t.b[j].mac * d + t.A[i * t.cols + j].mac * e;
                c = c + d * e;
            }
            value[r] = value[r] + v;
            mac[r] = mac[r] + m;
            cst[r] = cst[r] + c;
        }
        pos = base_e + t.cols;
    }
    std::vector<AShare> out;
    out.reserve(rows);
    for (size_t r = 0; r < rows; r++)
        out.push_back(AShare(value[r] & mask, mac[r] & mask, w).add_public(cst[r] & mask, op.role(), op.delta_share()));
    return out;
}

// One InMaskMaterial from the instances of one owner (tiled plans), in generation
// order; its descriptor names the total count (it is not a circuit that ran).
inline ppml::InMaskMaterial merge_inmasks(std::vector<ppml::InMaskMaterial>& parts) {
    if (parts.empty()) throw std::invalid_argument("no input masks to merge");
    if (parts.size() == 1) return std::move(parts[0]);
    ppml::InMaskMaterial out;
    const ppml::InMaskMaterial& first = parts[0];
    size_t total = 0;
    for (const ppml::InMaskMaterial& p : parts) {
        if (p.owner() != first.owner() || p.role != first.role || p.desc.width() != first.desc.width() || p.has_values != first.has_values)
            throw std::invalid_argument("input masks of different owners, roles or widths");
        total += p.count();
    }
    out.role = first.role;
    out.desc = ppml::Descriptor::inmask(first.owner(), total, first.desc.k, first.desc.s, first.desc.adder);
    out.has_values = first.has_values;
    out.shares.reserve(total);
    if (out.has_values) out.values.reserve(total);
    for (ppml::InMaskMaterial& p : parts) {
        out.shares.insert(out.shares.end(), p.shares.begin(), p.shares.end());
        if (out.has_values) out.values.insert(out.values.end(), p.values.begin(), p.values.end());
    }
    return out;
}

// Authenticated input of this party's values (mine, with `values`) and of the
// peer's (peer) in ONE exchange.  Returns (my shares, the peer's shares).
inline std::pair<std::vector<AShare>, std::vector<AShare>> exchange_inputs(truncmask::Opener& op,
                                                                           const ppml::InMaskMaterial* mine,
                                                                           const std::vector<u128>& values,
                                                                           const ppml::InMaskMaterial* peer) {
    const int role = op.role();
    unsigned w = 0;
    bytes msg;
    if (mine) {
        if (mine->owner() != role || !mine->has_values) throw std::invalid_argument("mine must be this party's own InMaskMaterial");
        w = mine->desc.width();
        if (values.size() != mine->count()) throw std::invalid_argument("one value per input mask is required");
    }
    if (peer) {
        if (peer->owner() != 1 - role) throw std::invalid_argument("peer must be the peer's InMaskMaterial");
        if (w && peer->desc.width() != w) throw std::invalid_argument("the two mask sets must use the same w");
        w = peer->desc.width();
    }
    if (!w) return std::make_pair(std::vector<AShare>(), std::vector<AShare>());
    const unsigned nb = (w + 7) / 8;
    const u128 mask = u128_mask(w);
    std::vector<u128> diffs;
    if (mine) {
        for (size_t j = 0; j < values.size(); j++) {
            const u128 dj = (values[j] - mine->values[j]) & mask;
            diffs.push_back(dj);
            uint8_t buf[16];
            u128_store_le(buf, dj);
            msg.insert(msg.end(), buf, buf + nb);
        }
    }
    const size_t expected = peer ? peer->count() * nb : 0;
    bytes reply;
    {
        abb::PhaseBook* book = op.book();
        const int prev = book ? book->current() : abb::PhaseBook::NONE;
        if (book) book->mark(PHASE_INPUT);
        try {
            reply = op.channel().exchange(msg, expected, "input: message has the wrong length");
        } catch (...) {
            if (book) book->mark(prev);
            throw;
        }
        if (book) book->mark(prev);
    }
    if (reply.size() != expected) throw ProtocolAbort("input: message has the wrong length");
    std::vector<AShare> mine_sh, peer_sh;
    if (mine)
        for (size_t j = 0; j < diffs.size(); j++)
            mine_sh.push_back(AShare(mine->shares[j].value, mine->shares[j].tag, w).add_public(diffs[j], role, op.delta_share()));
    if (peer) {
        for (size_t j = 0; j < peer->count(); j++) {
            uint8_t buf[16] = {0};
            std::memcpy(buf, reply.data() + j * nb, nb);
            const u128 d = u128_load_le(buf);
            if (d != (d & mask)) throw ProtocolAbort("input: value out of range");
            peer_sh.push_back(AShare(peer->shares[j].value, peer->shares[j].tag, w).add_public(d, role, op.delta_share()));
        }
    }
    return std::make_pair(mine_sh, peer_sh);
}

}  // namespace beaver
}  // namespace couplet
