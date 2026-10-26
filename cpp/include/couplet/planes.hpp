// couplet/planes.hpp -- one authenticated one-hot plane and its local queries
// (DESIGN_M3.md section 3, Carve); the C++ counterpart of
// python/couplet/planes.py.
//
// A Plane is this party's full-domain evaluation of a payload-1 DPF key:
// value shares y_b(u) and tag shares tau_b(u) for every u in [N], N = 2^n,
// with y_0 + y_1 = e_alpha and tau_0 + tau_1 = Delta e_alpha in Z_{2^w}, and
// the prefix tables P_b[j] = sum_{u < j} y_b(u) (same on the tags).  Every
// query is a public linear functional of the plane computed on BOTH
// components and returns this party's spdz2k::AShare over Z_{2^w} (`bits`
// reduces it to Z_{2^bits}; 0 keeps w).  x_hat is the public masked input
// and z = (x_hat - alpha) mod N the value the plane masks.
//
//     point(u)                  [[ [u = alpha] ]]                          exact, sign-exact
//     table(T, x_hat)           [[ T[z] ]] = sum_u T[(x_hat - u) mod N] y(u)   sign-exact
//     locator(l, u)             [[ [alpha in [l, u]] ]], cyclic (l > u wraps)   exact, sign-exact
//     z_interval(x_hat, lo, hi) [[ [lo <= z <= hi] ]] (locator [x_hat-hi, x_hat-lo])
//     equality(x_hat, c)        [[ [z = c] ]]
//     comparison(x_hat, c)      [[ [z < c] ]], c in [0, N] (0 for c = 0)
//     sign(x_hat)               [[ [z >= N/2] ]];  nonneg(x_hat): [[ [z < N/2] ]]
//     wrap(x_hat)               [[ [alpha > x_hat] ]] = suffix(x_hat + 1)
//     prefix(j) / suffix(j)     [[ [alpha < j] ]] / [[ [alpha >= j] ]], j in [0, N]
//
// Indicator outputs are marked exact and sign_exact, table outputs
// sign_exact only (entries may be negative); a reduced query keeps neither
// flag (spdz2k AShare::reduce, as in the reference).  EmbeddedTable holds a
// public table embedded sign-exactly in Z_{2^w} (entries must fit w - 1 bits).
//
// Threads.  Plane(role, key, threads) evaluates the tree with
// fss::dpf_eval_all on `threads` threads and builds the prefix tables in
// parallel blocks; evaluate_planes(role, keys, threads) evaluates the planes
// of a layer, spreading the planes over the threads (and, when there are
// fewer planes than threads, each plane over several); table(..., threads)
// splits the inner product.  Threads are started per call, so a single plane
// uses them only from PAR_EVAL_MIN points (evaluation) and PAR_SCAN_MIN
// points (prefix tables, table read) on, where they paid off in
// bench_ppml.cpp.  Results do not depend on the thread count.
// Nothing here sends a message.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "fss.hpp"
#include "spdz2k.hpp"
#include "u128.hpp"

namespace couplet {
namespace planes {

using spdz2k::AShare;

// Smallest domains at which one plane is split over threads.
const uint64_t PAR_EVAL_MIN = (uint64_t)1 << 14;
const uint64_t PAR_SCAN_MIN = (uint64_t)1 << 18;

// Run fn(i) for i in [0, count) on `workers` threads (the calling thread
// takes part); the first exception is rethrown after all threads joined.
template <class Fn>
inline void parallel_tasks(size_t count, unsigned workers, Fn&& fn) {
    if (workers <= 1 || count <= 1) {
        for (size_t i = 0; i < count; i++) fn(i);
        return;
    }
    workers = (unsigned)std::min<size_t>(workers, count);
    std::atomic<size_t> next(0);
    std::exception_ptr err;
    std::mutex mu;
    auto loop = [&] {
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= count) return;
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mu);
                if (!err) err = std::current_exception();
                next.store(count);
                return;
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < workers; t++) pool.emplace_back(loop);
    loop();
    for (std::thread& t : pool) t.join();
    if (err) std::rethrow_exception(err);
}

// fn(part, lo, hi) over `parts` contiguous parts of [0, n).
template <class Fn>
inline void parallel_ranges(size_t n, unsigned parts, Fn&& fn) {
    if (parts < 1) parts = 1;
    if ((size_t)parts > n) parts = (unsigned)(n ? n : 1);
    parallel_tasks(parts, parts, [&](size_t p) { fn(p, n * p / parts, n * (p + 1) / parts); });
}

// A public table embedded sign-exactly in Z_{2^width}.
struct EmbeddedTable {
    unsigned width = 0;
    std::vector<u128> e;

    EmbeddedTable() {}
    EmbeddedTable(const std::vector<int64_t>& entries, unsigned w) : width(w) {
        if (w < 1 || w > 128) throw std::invalid_argument("width must be in 1..128");
        e.resize(entries.size());
        for (size_t i = 0; i < entries.size(); i++) {
            const int64_t t = entries[i];
            const u128 v = t < 0 ? (u128(0) - u128((uint64_t)(-(t + 1)) + 1)) : u128((uint64_t)t);
            if (w < 64) {
                const int64_t lim = (int64_t)1 << (w - 1);
                if (t < -lim || t >= lim) throw std::invalid_argument("a table entry does not fit a sign-exact value of w bits");
            }
            e[i] = v & u128_mask(w);
        }
    }
    size_t size() const { return e.size(); }
};

class Plane {
public:
    Plane() {}
    Plane(int role, const fss::DpfKey& key, unsigned threads = 1, bool with_prefix = true) {
        evaluate(role, key, threads, with_prefix);
    }

    // Full-domain evaluation of `key` (and the prefix tables).
    void evaluate(int role, const fss::DpfKey& key, unsigned threads = 1, bool with_prefix = true) {
        if ((role != 0 && role != 1) || key.b != role) throw std::invalid_argument("the key does not belong to this party");
        if (key.n < 1 || key.n > 30) throw std::invalid_argument("plane domains are limited to 2^30 points");
        role_ = role;
        n_ = key.n;
        N_ = (uint64_t)1 << key.n;
        width_ = key.width();
        s_ = key.s;
        mask_ = u128_mask(width_);
        if (threads < 1) threads = 1;
        y_.assign(N_, u128(0));
        tau_.assign(N_, u128(0));
        fss::dpf_eval_all(key, y_.data(), tau_.data(), N_ >= PAR_EVAL_MIN ? threads : 1);
        py_.clear();
        pt_.clear();
        if (with_prefix) build_prefix(threads);
        ready_ = true;
    }

    // The prefix tables (if evaluate was called without them).
    void build_prefix(unsigned threads = 1) {
        py_.assign(N_ + 1, u128(0));
        pt_.assign(N_ + 1, u128(0));
        const unsigned parts = threads > 1 && N_ >= PAR_SCAN_MIN ? (unsigned)std::min<uint64_t>(threads, N_ / 65536) : 1;
        std::vector<u128> by(parts, u128(0)), bt(parts, u128(0));
        // pass 1: block sums; pass 2: running sums with the block offsets
        parallel_ranges((size_t)N_, parts, [&](size_t p, size_t lo, size_t hi) {
            u128 a = u128(0), b = u128(0);
            for (size_t u = lo; u < hi; u++) {
                a += y_[u];
                b += tau_[u];
            }
            by[p] = a;
            bt[p] = b;
        });
        std::vector<u128> oy(parts, u128(0)), ot(parts, u128(0));
        for (unsigned p = 1; p < parts; p++) {
            oy[p] = oy[p - 1] + by[p - 1];
            ot[p] = ot[p - 1] + bt[p - 1];
        }
        parallel_ranges((size_t)N_, parts, [&](size_t p, size_t lo, size_t hi) {
            u128 a = oy[p], b = ot[p];
            for (size_t u = lo; u < hi; u++) {
                a += y_[u];
                b += tau_[u];
                py_[u + 1] = a;
                pt_[u + 1] = b;
            }
        });
    }

    bool ready() const { return ready_; }
    bool has_prefix() const { return !py_.empty(); }
    int role() const { return role_; }
    unsigned n() const { return n_; }
    uint64_t N() const { return N_; }
    unsigned width() const { return width_; }
    unsigned s() const { return s_; }
    const std::vector<u128>& y() const { return y_; }
    const std::vector<u128>& tau() const { return tau_; }
    const std::vector<u128>& prefix_y() const { return py_; }
    const std::vector<u128>& prefix_tau() const { return pt_; }
    // Memory of the evaluated plane in bytes.
    size_t memory_bytes() const { return (y_.size() + tau_.size() + py_.size() + pt_.size()) * sizeof(u128); }
    void release() {
        std::vector<u128>().swap(y_);
        std::vector<u128>().swap(tau_);
        std::vector<u128>().swap(py_);
        std::vector<u128>().swap(pt_);
        ready_ = false;
    }

    // -- queries
    AShare point(uint64_t u, unsigned bits = 0) const {
        need();
        if (u >= N_) throw std::invalid_argument("u must lie in [0, 2^n)");
        return share(y_[u], tau_[u], true, bits);
    }

    AShare table(const EmbeddedTable& T, uint64_t x_hat, unsigned bits = 0, unsigned threads = 1) const {
        need();
        check_x_hat(x_hat);
        if (T.size() != N_) throw std::invalid_argument("the table must have 2^n entries");
        if (T.width != width_) throw std::invalid_argument("the table is embedded for another w");
        const unsigned parts = threads > 1 && N_ >= PAR_SCAN_MIN ? (unsigned)std::min<uint64_t>(threads, N_ / 32768) : 1;
        std::vector<u128> sv(parts, u128(0)), st(parts, u128(0));
        const u128* t = T.e.data();
        parallel_ranges((size_t)N_, parts, [&](size_t p, size_t lo, size_t hi) {
            u128 a = u128(0), b = u128(0);
            // u <= x_hat: T[x_hat - u]; u > x_hat: T[N + x_hat - u]
            const size_t split = (size_t)x_hat + 1;
            for (size_t u = lo; u < std::min(hi, split); u++) {
                const u128 c = t[x_hat - u];
                a += c * y_[u];
                b += c * tau_[u];
            }
            for (size_t u = std::max(lo, split); u < hi; u++) {
                const u128 c = t[(size_t)N_ + (size_t)x_hat - u];
                a += c * y_[u];
                b += c * tau_[u];
            }
            sv[p] = a;
            st[p] = b;
        });
        u128 a = u128(0), b = u128(0);
        for (unsigned p = 0; p < parts; p++) {
            a += sv[p];
            b += st[p];
        }
        return share(a, b, false, bits);
    }

    AShare prefix(uint64_t j, unsigned bits = 0) const {
        need_prefix();
        if (j > N_) throw std::invalid_argument("j must lie in [0, 2^n]");
        return share(py_[j], pt_[j], true, bits);
    }
    AShare suffix(uint64_t j, unsigned bits = 0) const {
        need_prefix();
        if (j > N_) throw std::invalid_argument("j must lie in [0, 2^n]");
        return share(py_[N_] - py_[j], pt_[N_] - pt_[j], true, bits);
    }
    AShare locator(uint64_t l, uint64_t u, unsigned bits = 0) const {
        need_prefix();
        if (l >= N_ || u >= N_) throw std::invalid_argument("l and u must lie in [0, 2^n)");
        if (l <= u) return share(py_[u + 1] - py_[l], pt_[u + 1] - pt_[l], true, bits);
        return share(py_[N_] - py_[l] + py_[u + 1], pt_[N_] - pt_[l] + pt_[u + 1], true, bits);
    }
    AShare z_interval(uint64_t x_hat, uint64_t lo, uint64_t hi, unsigned bits = 0) const {
        check_x_hat(x_hat);
        if (lo >= N_ || hi >= N_) throw std::invalid_argument("lo and hi must lie in [0, 2^n)");
        if (lo > hi) throw std::invalid_argument("need lo <= hi");
        return locator((x_hat + N_ - hi) & (N_ - 1), (x_hat + N_ - lo) & (N_ - 1), bits);
    }
    AShare equality(uint64_t x_hat, uint64_t c, unsigned bits = 0) const {
        check_x_hat(x_hat);
        if (c >= N_) throw std::invalid_argument("c must lie in [0, 2^n)");
        return point((x_hat + N_ - c) & (N_ - 1), bits);
    }
    AShare comparison(uint64_t x_hat, uint64_t c, unsigned bits = 0) const {
        check_x_hat(x_hat);
        if (c > N_) throw std::invalid_argument("c must lie in [0, 2^n]");
        if (c == 0) {
            need();
            return share(u128(0), u128(0), true, bits);
        }
        return z_interval(x_hat, 0, c - 1, bits);
    }
    AShare sign(uint64_t x_hat, unsigned bits = 0) const { return z_interval(x_hat, N_ / 2, N_ - 1, bits); }
    AShare nonneg(uint64_t x_hat, unsigned bits = 0) const { return comparison(x_hat, N_ / 2, bits); }
    AShare wrap(uint64_t x_hat, unsigned bits = 0) const {
        check_x_hat(x_hat);
        return suffix(x_hat + 1, bits);
    }

private:
    void need() const {
        if (!ready_) throw std::invalid_argument("the plane is not evaluated");
    }
    void need_prefix() const {
        need();
        if (py_.empty()) throw std::invalid_argument("the plane has no prefix tables");
    }
    void check_x_hat(uint64_t x_hat) const {
        need();
        if (x_hat >= N_) throw std::invalid_argument("x_hat must be an integer in [0, 2^n)");
    }
    AShare share(u128 value, u128 tag, bool exact, unsigned bits) const {
        AShare sh(value & mask_, tag & mask_, width_, exact, true);
        if (bits != 0 && bits != width_) {
            if (bits <= s_) throw std::invalid_argument("bits must exceed s");
            sh = sh.reduce(bits - s_, s_);
        }
        return sh;
    }

    bool ready_ = false;
    int role_ = 0;
    unsigned n_ = 0, width_ = 0, s_ = 0;
    uint64_t N_ = 0;
    u128 mask_ = u128(0);
    std::vector<u128> y_, tau_, py_, pt_;
};

// Evaluate the planes of `keys` (one layer) on `threads` threads: planes are
// spread over the threads, and with fewer planes than threads each plane
// gets several.  Returns one Plane per key, in order.
inline std::vector<Plane> evaluate_planes(int role, const std::vector<const fss::DpfKey*>& keys, unsigned threads,
                                          bool with_prefix = true) {
    if (threads < 1) threads = 1;
    std::vector<Plane> out(keys.size());
    if (keys.empty()) return out;
    const unsigned workers = (unsigned)std::min<size_t>(threads, keys.size());
    const unsigned inner = std::max(1u, threads / workers);
    parallel_tasks(keys.size(), workers, [&](size_t i) { out[i].evaluate(role, *keys[i], inner, with_prefix); });
    return out;
}

}  // namespace planes
}  // namespace couplet
