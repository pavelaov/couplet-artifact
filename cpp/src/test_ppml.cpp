// test_ppml.cpp -- tests of the Couplet layer in C++ (DESIGN_M3.md sections 2
// and 3): fixedpoint.hpp, tables.hpp, planes.hpp, circuits_ppml.hpp,
// truncmask.hpp, beaver.hpp, nn.hpp and infer.hpp, against the reference data
// that python/couplet/tools/export_model.py and python/bench/export_circuits.py
// wrote (python/vectors/ppml, circuits/).
//
//   tables      fixed-point helpers; the exact tables against their integer
//               definitions; the table files against the table hashes of the
//               model files; EmbeddedTable.
//   planes      every query of a plane exhaustively for n = 3..8 against the
//               plaintext model, for several offsets, with an unreduced
//               Delta >= 2^s and a random one: point, table (random signed
//               table), prefix and suffix for every j, the cyclic locator for
//               every pair (l, u) (n <= 6; random pairs above), interval
//               containment for every pair (lo, hi) and x_hat (n <= 6; 24
//               random pairs per x_hat above), equality, comparison for every
//               c in [0, N], sign, nonneg, wrap, the reduction to n + s bits,
//               the flags; the result does not depend on the thread count.
//   circuits    descriptors, names and canonical JSON (the output mask is the
//               core's circuit file); the trusted-dealer reference.
//   manifest    for every plan file of python/vectors/ppml the manifest built
//               here from the model file has the reference encoding and digest
//               (public and private model, both modes).
//   mirror      the C++ mirror against the exported logits (mode exact) and
//               the exported round-mode expectations (their thresholds); the
//               float model.
//   truncmask   the identity of truncate-and-mask between two parties on
//               dealt material, for every p of a 2^(f+n+2) range at (3, 1),
//               (4, 2), (5, 3), both modes, authenticated and first-layer
//               (shared) inputs; the ArithCheck over all openings passes.
//   beaver      products and matrix-vector products with dealt triples and
//               authenticated inputs (exchange_inputs), checked by an opening.
//   models      the reduced models end to end with ideal preprocessing (both
//               parties in this process): mlp16, mlp32, cnn10 and mlp64 at
//               (16, 8) and mlp4 at (8, 4), both modes, k = s = 64; mode exact
//               bit-exact with the exported logits, mode round bit-exact with
//               the mirror on the recorded thresholds; counters, rounds and
//               bytes of the online phase; the private-model mode of mlp4 in
//               both modes (output masks of correction C2); a triple cap.
//   real        one run with the real preprocessing (mlp4, mode round): bit
//               exact, one bucketing population, the per-phase accounting
//               sums to the channel totals.
//   tamper      a mismatching plan, a flipped bit in the first-layer opening,
//               in the input exchange and in the second opening of mode
//               exact, a wrong value share in a deferred opening and in the
//               release, a substituted plane key: the honest party throws
//               ProtocolAbort, outputs nothing and is retired; a malicious
//               P_0 with logit weights 2^k e_j in private-model mode obtains
//               only the low k bits as the honest output.
//   arguments   invalid arguments throw std::invalid_argument before any message.
//   tiling      tiled private-model plans (task R-TILE): the tile layout; the
//               tiled plan of mlp48 (default 16 x 16 tiles above 4,096
//               weights); matvec_tiled on dealt material for matrices whose
//               sizes are and are not multiples of the tile (37x21, 64x48,
//               10x128 in 16x16, 5x7 in 2x3, ...) equals the plaintext product
//               in one exchange; a wrong value share in a tile's D or e opening
//               aborts at the check; mlp7 with 2 x 2 tiles end to end in both
//               modes (bit-exact, ten tile triples); a plan whose tile entries
//               do not match the tiles aborts the generation; a wrong value
//               share in a tile opening of the pipeline aborts the honest party.
//               The manifest section also checks the tiled plan files of mlp7
//               and mlp48 and, if present, those of the full MLP in
//               experiments/models.
//
// Usage: test_ppml [--circuits DIR] [--vectors DIR] [--quick]
// Exit code 0 iff every check passes.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "couplet/beaver.hpp"
#include "couplet/channel.hpp"
#include "couplet/circuit.hpp"
#include "couplet/circuits_ppml.hpp"
#include "couplet/fixedpoint.hpp"
#include "couplet/fss.hpp"
#include "couplet/infer.hpp"
#include "couplet/nn.hpp"
#include "couplet/planes.hpp"
#include "couplet/prep.hpp"
#include "couplet/rng.hpp"
#include "couplet/session.hpp"
#include "couplet/spdz2k.hpp"
#include "couplet/tables.hpp"
#include "couplet/truncmask.hpp"
#include "couplet/twoparty.hpp"

using namespace couplet;
using spdz2k::AShare;

// ------------------------------------------------------------ harness
static long long g_pass = 0, g_fail = 0;
static bool g_quick = false;
static std::string g_circuits, g_vectors;

static void check(bool ok, const std::string& what) {
    if (ok) {
        g_pass++;
    } else {
        g_fail++;
        if (g_fail <= 80) std::printf("  FAIL: %s\n", what.c_str());
    }
}

struct Section {
    const char* name;
    long long pass0, fail0;
    std::chrono::steady_clock::time_point t0;
    explicit Section(const char* n) : name(n), pass0(g_pass), fail0(g_fail), t0(std::chrono::steady_clock::now()) {}
    ~Section() {
        const long long f = g_fail - fail0, p = g_pass - pass0;
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("[%s] %s (%lld checks, %.2f s)\n", f ? "FAIL" : "PASS", name, p + f, s);
        std::fflush(stdout);
    }
};

template <class F>
static bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
    }
    return false;
}

static std::string hexs(const hashes::hash32& h) { return ppml::hex(h); }
static bytes to_bytes(const std::string& s) { return bytes(s.begin(), s.end()); }

static u128 rand_bits(RandomSource& rng, unsigned bits) {
    uint8_t b[16];
    rng.fill(b, 16);
    return u128_load_le(b) & u128_mask(bits);
}

// Delta values: the unreduced maximum (>= 2^s) and a random one.
static std::vector<std::pair<uint64_t, uint64_t>> delta_pairs(RandomSource& rng, unsigned s) {
    const uint64_t top = s == 64 ? ~(uint64_t)0 : (((uint64_t)1 << s) - 1);
    std::vector<std::pair<uint64_t, uint64_t>> out;
    out.push_back(std::make_pair(top, top));
    out.push_back(std::make_pair(rng.u64() & top, rng.u64() & top));
    return out;
}

static std::pair<u128, u128> rec(const AShare& a, const AShare& b) { return spdz2k::reconstruct(a, b); }

// ------------------------------------------------------------ tables and fixed point
static void test_tables() {
    Section sec("tables and fixed point");
    using namespace fixedpoint;
    check(to_signed(u128(255), 8) == -1 && to_signed(u128(127), 8) == 127 && to_signed(u128(128), 8) == -128, "to_signed");
    check(to_signed(~u128(0), 64) == -1, "to_signed 64");
    check(truncate(from_i64(-1), 2, 8) == 255 && truncate(from_i64(-5), 2, 8) == 254 && truncate(from_i64(9), 2, 8) == 2,
          "truncate floors negative values");
    check(asr(from_i64(-9), 3) == from_i64(-2) && asr(from_i64(9), 3) == from_i64(1), "arithmetic shift");
    check(fits_signed(from_i64(127), 8) && !fits_signed(from_i64(128), 8) && fits_signed(from_i64(-128), 8) &&
              !fits_signed(from_i64(-129), 8),
          "fits_signed");
    check(embed(-3, 10) == u128(1021) && throws_invalid([] { embed(600, 10); }), "embed");
    check(round_half_even(2.5) == 2 && round_half_even(3.5) == 4 && round_half_even(-2.5) == -2, "ties to even");
    // exact tables
    const tables::Table relu = tables::exact_table("relu", 8, 4), sign = tables::exact_table("sign", 8, 4);
    const tables::Table recip = tables::exact_table("recip", 8, 4), rsq = tables::exact_table("rsqrt", 8, 4);
    bool ok = true;
    for (uint64_t z = 0; z < 256; z++) {
        const int64_t v = fixedpoint::to_signed(u128(z), 8);
        ok = ok && relu.entries[z] == (v > 0 ? v : 0) && sign.entries[z] == (v > 0 ? 16 : (v < 0 ? -16 : 0));
        if (v > 0) {
            // |2 T v - 2^(2f+1)| <= v  and  (2T-1)^2 v <= 2^(3f+2) <= (2T+1)^2 v
            const int64_t T = recip.entries[z];
            ok = ok && std::llabs(2 * T * v - (1 << 9)) <= v;
            const int64_t R = rsq.entries[z];
            ok = ok && (2 * R - 1) * (2 * R - 1) * v <= (1 << 14) && (1 << 14) <= (2 * R + 1) * (2 * R + 1) * v;
        } else {
            ok = ok && recip.entries[z] == 0 && rsq.entries[z] == 0;
        }
    }
    check(ok, "exact tables at (8, 4) against their integer definitions");
    // table files and model hashes
    const nn::Model mlp16 = nn::Model::load(g_vectors + "/mlp16_model.json");
    const tables::Table tanh16 = tables::read_table(g_vectors + "/table_tanh_n16_f8.bin");
    check(tanh16.n == 16 && tanh16.f == 8 && tanh16.entries.size() == 65536, "the tanh table file");
    check(hexs(tables::table_hash(tanh16)) == hexs(mlp16.table_hash()), "the tanh table file has the model's table hash");
    const tables::Table relu_file = tables::read_table(g_vectors + "/table_relu_n16_f8.bin");
    check(relu_file.entries == tables::exact_table("relu", 16, 8).entries, "the relu table file equals the computed table");
    const nn::Model cnn10 = nn::Model::load(g_vectors + "/cnn10_model.json");
    check(cnn10.act == "relu" && hexs(cnn10.table_hash()) == hexs(tables::table_hash(relu_file)), "relu table hash of cnn10");
    check(tables::table_bytes(tanh16) == [] {
        std::ifstream f(g_vectors + "/table_tanh_n16_f8.bin", std::ios::binary);
        return bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }(), "the table file re-encodes to the same bytes");
    ok = true;
    for (size_t z = 0; z < tanh16.entries.size(); z++) {
        const int64_t t = tanh16.entries[z];
        const int64_t v = fixedpoint::to_signed(u128((uint64_t)z), 16);
        // tanh is odd, monotone and bounded by 1 (in units of 2^-8)
        ok = ok && t >= -256 && t <= 256 && (v >= 0 ? t >= 0 : t <= 0);
        if (v > -32768 && v < 32767 && v != 0) ok = ok && t == -tanh16.entries[(size_t)((65536 - z) & 65535)];
    }
    check(ok, "the tanh table is odd and bounded");
    planes::EmbeddedTable et(std::vector<int64_t>{-1, 2, -3}, 10);
    check(et.e[0] == u128(1023) && et.e[1] == u128(2) && et.e[2] == u128(1021), "EmbeddedTable is sign-exact");
    check(throws_invalid([] { planes::EmbeddedTable(std::vector<int64_t>{600}, 10); }), "an entry that does not fit is refused");
}

// ------------------------------------------------------------ planes
// Counts every individual comparison as one check.
struct Tally {
    long long ok = 0, all = 0;
    void add(bool b) {
        all++;
        if (b) ok++;
    }
};
static void report(const Tally& t, const std::string& what) {
    g_pass += t.ok;
    g_fail += t.all - t.ok;
    if (t.ok != t.all && g_fail <= 80) std::printf("  FAIL: %s (%lld of %lld)\n", what.c_str(), t.all - t.ok, t.all);
}

static void check_plane_queries(unsigned n, uint64_t alpha, u128 delta, const fss::DpfKey& k0, const fss::DpfKey& k1,
                                RandomSource& rng, bool exhaustive_pairs) {
    const unsigned w = k0.width(), s = k0.s;
    const uint64_t N = (uint64_t)1 << n;
    const u128 mask = u128_mask(w);
    const planes::Plane p0(0, k0, 1), p1(1, k1, 3);
    const std::string tag = "n=" + std::to_string(n) + " alpha=" + std::to_string(alpha);
    // the plane itself
    Tally t_point;
    for (uint64_t u = 0; u < N; u++) {
        const std::pair<u128, u128> r = rec(p0.point(u), p1.point(u));
        t_point.add(r.first == u128(u == alpha ? 1 : 0) && r.second == ((u == alpha ? delta : u128(0)) & mask));
    }
    report(t_point, tag + ": point(u) on the whole domain, both components");
    // thread independence of the evaluation
    {
        const planes::Plane q(0, k0, 4);
        check(q.y() == p0.y() && q.tau() == p0.tau() && q.prefix_y() == p0.prefix_y(), tag + ": thread count");
    }
    auto ind = [&](const AShare& a, const AShare& b, bool want) {
        const std::pair<u128, u128> r = rec(a, b);
        const u128 m = u128_mask(a.bits);
        return r.first == u128(want ? 1 : 0) && r.second == ((want ? delta : u128(0)) & m) && a.exact && a.sign_exact;
    };
    Tally t_pre;
    for (uint64_t j = 0; j <= N; j++) {
        t_pre.add(ind(p0.prefix(j), p1.prefix(j), alpha < j));
        t_pre.add(ind(p0.suffix(j), p1.suffix(j), alpha >= j));
    }
    report(t_pre, tag + ": prefix(j) and suffix(j) for every j");
    // locator: every pair (l, u) or random pairs
    Tally t_loc;
    auto in_cyclic = [&](uint64_t l, uint64_t u) { return l <= u ? (alpha >= l && alpha <= u) : (alpha >= l || alpha <= u); };
    if (exhaustive_pairs) {
        for (uint64_t l = 0; l < N; l++)
            for (uint64_t u = 0; u < N; u++) t_loc.add(ind(p0.locator(l, u), p1.locator(l, u), in_cyclic(l, u)));
    } else {
        for (int i = 0; i < 4096; i++) {
            const uint64_t l = rng.u64() & (N - 1), u = rng.u64() & (N - 1);
            t_loc.add(ind(p0.locator(l, u), p1.locator(l, u), in_cyclic(l, u)));
        }
    }
    report(t_loc, tag + ": cyclic locator");
    // per x_hat: table, equality, comparison, interval, sign, nonneg, wrap
    std::vector<int64_t> T(N);
    for (uint64_t z = 0; z < N; z++) T[z] = (int64_t)(rng.u64() >> 1) * ((rng.u64() & 1) ? -1 : 1) / 4;
    const planes::EmbeddedTable ET(T, w);
    Tally t_tab, t_eq, t_cmp, t_int, t_misc, t_bits;
    for (uint64_t xh = 0; xh < N; xh++) {
        const uint64_t z = (xh + N - alpha) & (N - 1);
        const AShare t0 = p0.table(ET, xh), t1 = p1.table(ET, xh, 0, 2);
        const std::pair<u128, u128> rt = rec(t0, t1);
        const u128 want = ET.e[z];
        t_tab.add(rt.first == want && rt.second == ((delta * want) & mask) && t0.sign_exact && !t0.exact &&
                  fixedpoint::to_signed(rt.first & u128_mask(64), 64) == T[z]);
        for (uint64_t c = 0; c < N; c++) t_eq.add(ind(p0.equality(xh, c), p1.equality(xh, c), z == c));
        for (uint64_t c = 0; c <= N; c++) t_cmp.add(ind(p0.comparison(xh, c), p1.comparison(xh, c), z < c));
        if (exhaustive_pairs) {
            for (uint64_t lo = 0; lo < N; lo++)
                for (uint64_t hi = lo; hi < N; hi++)
                    t_int.add(ind(p0.z_interval(xh, lo, hi), p1.z_interval(xh, lo, hi), z >= lo && z <= hi));
        } else {
            for (int i = 0; i < 24; i++) {
                uint64_t lo = rng.u64() & (N - 1), hi = rng.u64() & (N - 1);
                if (lo > hi) std::swap(lo, hi);
                t_int.add(ind(p0.z_interval(xh, lo, hi), p1.z_interval(xh, lo, hi), z >= lo && z <= hi));
            }
        }
        t_misc.add(ind(p0.sign(xh), p1.sign(xh), z >= N / 2));
        t_misc.add(ind(p0.nonneg(xh), p1.nonneg(xh), z < N / 2));
        t_misc.add(ind(p0.wrap(xh), p1.wrap(xh), alpha > xh));
        // reduction to Z_{2^(n+s)}: payload [z < c] mod 2^n, tag Delta [..] mod 2^(n+s), no flags
        const uint64_t c = rng.u64() % (N + 1);
        const AShare r0 = p0.comparison(xh, c, n + s), r1 = p1.comparison(xh, c, n + s);
        const std::pair<u128, u128> rr = rec(r0, r1);
        t_bits.add(r0.bits == n + s && rr.first == u128(z < c ? 1 : 0) && rr.second == ((z < c ? delta : u128(0)) & u128_mask(n + s)) &&
                   !r0.exact && !r0.sign_exact);
    }
    report(t_tab, tag + ": table(T, x_hat) for every x_hat (both components, sign-exact)");
    report(t_eq, tag + ": equality for every (x_hat, c)");
    report(t_cmp, tag + ": comparison for every (x_hat, c), c in [0, N]");
    report(t_int, tag + ": interval containment");
    report(t_misc, tag + ": sign, nonneg, wrap");
    report(t_bits, tag + ": queries reduced to n + s bits");
}

static void test_planes() {
    Section sec("planes: every query exhaustively for n = 3..8");
    CtrPrgRng rng(0x91a7e5ull);
    for (unsigned n = 3; n <= 8; n++) {
        const unsigned k = n <= 5 ? 32 : 64, s = 64;
        const std::vector<std::pair<uint64_t, uint64_t>> ds = delta_pairs(rng, s);
        const uint64_t N = (uint64_t)1 << n;
        const uint64_t alphas[3] = {0, N - 1, rng.u64() & (N - 1)};
        for (int i = 0; i < 3; i++) {
            const u128 delta = fss::make_delta(ds[i % 2].first, ds[i % 2].second);
            const std::pair<fss::DpfKey, fss::DpfKey> keys = fss::dpf_gen(alphas[i], 1, delta, n, k, s, rng);
            check_plane_queries(n, alphas[i], delta, keys.first, keys.second, rng, n <= 6);
        }
    }
    // evaluate_planes of a layer equals single evaluations
    CtrPrgRng r2(7);
    std::vector<fss::DpfKey> ks;
    for (int i = 0; i < 5; i++) ks.push_back(fss::dpf_gen(r2.u64() & 1023, 1, u128(12345), 10, 64, 64, r2).first);
    std::vector<const fss::DpfKey*> ptrs;
    for (const fss::DpfKey& kk : ks) ptrs.push_back(&kk);
    const std::vector<planes::Plane> layer = planes::evaluate_planes(0, ptrs, 8);
    bool ok = layer.size() == 5;
    for (size_t i = 0; ok && i < 5; i++) {
        const planes::Plane single(0, ks[i], 1);
        ok = layer[i].y() == single.y() && layer[i].tau() == single.tau() && layer[i].prefix_tau() == single.prefix_tau();
    }
    check(ok, "evaluate_planes (8 threads) equals the single evaluations");
    check(throws_invalid([&] { planes::Plane(1, ks[0], 1); }), "a key of the other party is refused");
    const planes::Plane noprefix(0, ks[0], 2, false);
    check(throws_invalid([&] { noprefix.prefix(3); }) && !noprefix.has_prefix(), "a plane without prefix tables refuses prefix queries");
    check(throws_invalid([&] { noprefix.table(planes::EmbeddedTable(std::vector<int64_t>(8, 1), 128), 0); }), "a table of the wrong size");
}

// ------------------------------------------------------------ descriptors and dealer
static void test_circuits(circuit::CircuitStore& store) {
    Section sec("circuits: descriptors, names, dealer");
    const ppml::Descriptor site = ppml::Descriptor::site(16, 8, 64, 64, "round");
    check(site.name() == "site_n16_f8_k64_s64_round", "site name");
    check(site.to_json() == "{\"adder\":\"ripple\",\"f\":8,\"k\":64,\"kind\":\"site\",\"mode\":\"round\",\"n\":16,\"s\":64}", "site JSON");
    const ppml::Descriptor om = ppml::Descriptor::outmask(64, 64);
    check(om.name() == "gate_n1_k64_s64_shared_om1", "the output mask is the core's circuit file");
    check(om.to_json() == "{\"adder\":\"ripple\",\"k\":64,\"kind\":\"outmask\",\"s\":64}", "outmask JSON");
    check(ppml::and_count(store, om) == 4159 && store.get(om.name())->n_and() == store.get(Session::outmask_circuit_name(64, 64))->n_and(),
          "4,159 AND gates per output mask (k = s = 64)");
    check(ppml::Descriptor::inmask(1, 5, 64, 64).name() == "inmask_p1_c5_k64_s64" &&
              ppml::Descriptor::mattriple(8, 16, 64, 64).name() == "mattriple_8x16_k64_s64" &&
              ppml::Descriptor::triples(3, 64, 64).name() == "triples3_k64_s64",
          "names");
    check(ppml::Descriptor::mattriple(2, 4, 64, 64).to_json() ==
              "{\"adder\":\"ripple\",\"cols\":4,\"k\":64,\"kind\":\"mattriple\",\"rows\":2,\"s\":64}",
          "mattriple JSON");
    check(ppml::Descriptor::inmask(0, 24, 64, 64).to_json() ==
              "{\"adder\":\"ripple\",\"count\":24,\"k\":64,\"kind\":\"inmask\",\"owner\":0,\"s\":64}",
          "inmask JSON");
    check(throws_invalid([] { ppml::Descriptor::site(0, 8, 64, 64, "round"); }) &&
              throws_invalid([] { ppml::Descriptor::site(21, 8, 64, 64, "round"); }) &&
              throws_invalid([] { ppml::Descriptor::site(16, 8, 64, 64, "nearest"); }) &&
              throws_invalid([] { ppml::Descriptor::site(40, 30, 64, 64, "round"); }) &&
              throws_invalid([] { ppml::Descriptor::inmask(2, 1, 64, 64); }) &&
              throws_invalid([] { ppml::Descriptor::mattriple(0, 1, 64, 64); }),
          "bad descriptors are refused");
    // the dealer reference satisfies the relations
    CtrPrgRng rng(99);
    const u128 delta = fss::make_delta(~(uint64_t)0, ~(uint64_t)0);
    for (const char* mode : {"round", "exact"}) {
        const ppml::Descriptor d = ppml::Descriptor::site(5, 2, 64, 64, mode);
        const std::pair<ppml::SiteMaterial, ppml::SiteMaterial> m = ppml::deal_site(d, delta, rng);
        const unsigned mb = d.m(), ns = d.n + d.s;
        const u128 rp = (m.first.rp.value + m.second.rp.value) & u128_mask(mb);
        check(((m.first.rp.tag + m.second.rp.tag) & u128_mask(mb)) == ((delta * rp) & u128_mask(mb)), "dealt [[rho']] tag");
        const uint64_t rho_h = u128_lo(rp >> 2) & 31, rho_l = u128_lo(rp) & 3;
        uint64_t idx = rho_h;
        if (std::string(mode) == "exact") {
            const u128 r2 = (m.first.r2.value + m.second.r2.value) & u128_mask(ns);
            idx = u128_lo(r2) & 31;
            check(((m.first.rh.value + m.second.rh.value) & u128_mask(ns)) == u128(rho_h), "dealt [[rho_h]]");
            const planes::Plane b0(0, m.first.plane_b), b1(1, m.second.plane_b);
            bool okb = true;
            for (uint64_t u = 0; u < 4; u++) okb = okb && rec(b0.point(u), b1.point(u)).first == u128(u == rho_l ? 1 : 0);
            check(okb, "dealt plane B at rho_l");
        }
        const planes::Plane a0(0, m.first.plane_a), a1(1, m.second.plane_a);
        bool oka = true;
        for (uint64_t u = 0; u < 32; u++) oka = oka && rec(a0.point(u), a1.point(u)).first == u128(u == idx ? 1 : 0);
        check(oka, std::string("dealt plane A (") + mode + ")");
    }
    const std::pair<ppml::OutMaskMaterial, ppml::OutMaskMaterial> o = ppml::deal_outmask(om, delta, rng);
    const u128 v = o.first.share.value + o.second.share.value, t = o.first.share.tag + o.second.share.tag;
    check((u128_lo(o.first.share.value) | u128_lo(o.second.share.value) | u128_lo(o.first.share.tag)) == 0 && t == delta * v,
          "dealt output mask: multiples of 2^k, tag Delta value mod 2^w");
}

// ------------------------------------------------------------ manifests of the plan files
static void test_manifest() {
    Section sec("manifest: the plan files of the reference");
    const char* plans[] = {"mlp4_plan_round_k64_s64.json",          "mlp4_plan_exact_k64_s64.json",
                           "mlp4_plan_round_private_k64_s64.json",  "mlp4_plan_exact_private_k64_s64.json",
                           "mlp16_plan_round_k64_s64.json",         "mlp16_plan_exact_k64_s64.json",
                           "mlp32_plan_round_k64_s64.json",         "mlp32_plan_exact_k64_s64.json",
                           "cnn10_plan_round_k64_s64.json",         "cnn10_plan_exact_k64_s64.json",
                           "mlp64_plan_round_k64_s64.json",         "mlp64_plan_exact_k64_s64.json"};
    std::vector<std::string> files(plans, plans + sizeof(plans) / sizeof(plans[0]));
    // tiled plans (task R-TILE): mlp7 with 2 x 2 tiles, mlp48 with the default tiling; the full MLP if exported
    for (const char* nm : {"mlp7_plan_round_private_tile2x2_lim0_k64_s64.json", "mlp7_plan_exact_private_tile2x2_lim0_k64_s64.json",
                           "mlp48_plan_round_private_k64_s64.json", "mlp48_plan_exact_private_k64_s64.json"})
        files.push_back(nm);
    const std::string full_dir = g_vectors + "/../../../experiments/models/";
    for (const char* nm : {"mlp784_16_8_plan_round_private_k64_s64.json", "mlp784_16_8_plan_exact_private_k64_s64.json"})
        if (std::ifstream(full_dir + nm).good()) files.push_back("../../../experiments/models/" + std::string(nm));
        else std::printf("    (absent: experiments/models/%s)\n", nm);
    size_t tiled_files = 0;
    for (const std::string& name_s : files) {
        const char* name = name_s.c_str();
        const std::string dir = g_vectors + "/" + name_s.substr(0, name_s.find_last_of('/') == std::string::npos ? 0 : name_s.find_last_of('/') + 1);
        const json::Value doc = json::parse(nn::detail::read_file(g_vectors + "/" + name));
        const nn::Model model = nn::Model::load(dir + doc.at("model_file").str());
        const bool priv = doc.at("private_model").as_bool();
        nn::Tiling tiling;
        const json::Value* tj = doc.find("tiling");
        if (tj) tiling = nn::Tiling((size_t)tj->at("rows").as_u64(), (size_t)tj->at("cols").as_u64(), (size_t)tj->at("limit").as_u64());
        if (nn::plan_tiled(model, priv, tiling)) {
            tiled_files++;
            size_t tiles = 0;
            for (const nn::Shape& sh : model.shapes) tiles += nn::layer_tiles(sh.first, sh.second, tiling).size();
            check(tj && tj->at("tiled").as_bool() && tj->at("tiles").as_u64() == tiles, std::string(name) + ": the tile count of the file");
        }
        const nn::Model& own = model;
        const ppml::Manifest man = nn::plan_manifest(own, doc.at("mode").str(), 64, 64, priv, tiling);
        check(ppml::hex(man.encode()) == doc.at("manifest_hex").str(), std::string(name) + ": the same CPLTMAN1 encoding");
        check(ppml::hex(man.digest()) == doc.at("digest").str(), std::string(name) + ": the same digest");
        check(man.site_count() == (size_t)doc.at("sites").as_u64(), std::string(name) + ": sites");
        if (priv) {
            const ppml::Manifest p1 = nn::plan_manifest(model.shape_only(), doc.at("mode").str(), 64, 64, true, tiling);
            check(p1.encode() == man.encode(), std::string(name) + ": P_1 (shape only) builds the same plan");
            check(man.entries.back().desc.kind == ppml::KIND_OUTMASK && man.entries.back().batch == model.n_out(),
                  std::string(name) + ": one output mask per logit (correction C2)");
        }
    }
    check(tiled_files >= 4, "the tiled plan files were checked");
}

// ------------------------------------------------------------ mirror against the exported vectors
static void test_mirror() {
    Section sec("mirror: the exported expectations");
    const char* names[] = {"mlp4", "mlp16", "mlp32", "cnn10", "mlp64"};
    for (const char* nm : names) {
        const nn::Model model = nn::Model::load(g_vectors + "/" + nm + "_model.json");
        const nn::Vectors vec = nn::Vectors::load(g_vectors + "/" + nm + "_vectors.json");
        check(vec.model_hash == hexs(model.model_hash()) && vec.architecture_hash == hexs(model.architecture_hash()) &&
                  vec.table_hash == hexs(model.table_hash()),
              std::string(nm) + ": hashes of the vectors file");
        // The exported float logits are those of the real inputs; the quantised inputs differ by at
        // most 2^-(f+1) per feature, and every layer is ||W||_inf-Lipschitz in the max norm (tanh and
        // ReLU are 1-Lipschitz), so the float logits of the quantised inputs are within
        // B = 2^-(f+1) prod_l ||W_l||_inf of the exported ones.
        double lip = std::ldexp(1.0, -(int)model.f - 1);
        for (size_t l = 0; l < model.float_layers.size(); l++) {
            double best = 0;
            for (size_t r = 0; r < model.shapes[l].first; r++) {
                double row = 0;
                for (size_t c = 0; c < model.shapes[l].second; c++) row += std::fabs(model.float_layers[l].W[r * model.shapes[l].second + c]);
                best = std::max(best, row);
            }
            lip *= best;
        }
        bool ok_e = true, ok_r = true, ok_f = !vec.float_logits.empty(), ok_share = true;
        for (size_t i = 0; i < vec.inputs.size(); i++) {
            ok_e = ok_e && nn::to_i64(model.mirror(vec.inputs[i], true).logits) == vec.exact_logits[i];
            ok_r = ok_r && nn::to_i64(model.mirror(vec.inputs[i], false, &vec.round[i].thresholds).logits) == vec.round[i].logits;
            if (!vec.float_logits.empty()) {
                std::vector<double> x;
                for (int64_t v : vec.inputs[i]) x.push_back((double)v / (double)(1 << model.f));
                const std::vector<double> fl = model.forward_float(x);
                for (size_t j = 0; j < fl.size(); j++) ok_f = ok_f && std::fabs(fl[j] - vec.float_logits[i][j]) <= lip + 1e-12;
            }
            const std::vector<int64_t> x0(vec.inputs[i].begin(), vec.inputs[i].begin() + (ptrdiff_t)model.partition());
            const std::vector<int64_t> x1(vec.inputs[i].begin() + (ptrdiff_t)model.partition(), vec.inputs[i].end());
            const std::vector<u128> s0 = model.first_layer_share(0, x0), s1 = model.first_layer_share(1, x1);
            const nn::MirrorResult mr = model.mirror(vec.inputs[i], true);
            for (size_t j = 0; j < s0.size(); j++) ok_share = ok_share && s0[j] + s1[j] == mr.pre[0][j];
        }
        check(ok_e, std::string(nm) + ": exact logits of every input");
        check(ok_r, std::string(nm) + ": round logits with the exported thresholds");
        check(ok_f, std::string(nm) + ": float logits");
        check(ok_share, std::string(nm) + ": the first-layer shares sum to the pre-activation");
        check(model.sign_exact_in(128) && model.release_fits(64), std::string(nm) + ": bounds");
    }
}

// ------------------------------------------------------------ truncate-and-mask identity
struct TmOut {
    std::vector<truncmask::TruncResult> res;
    bool checked = false;
};

static void test_truncmask() {
    Section sec("truncate-and-mask: the identity in both modes");
    CtrPrgRng rng(0x7c11);
    const unsigned k = 32, s = 64, w = k + s;
    const int fmts[3][2] = {{3, 1}, {4, 2}, {5, 3}};
    for (const auto& fm : fmts) {
        const unsigned n = (unsigned)fm[0], f = (unsigned)fm[1];
        for (const char* mode : {"round", "exact"}) {
            for (int shared = 0; shared < 2; shared++) {
                const ppml::Descriptor d = ppml::Descriptor::site(n, f, k, s, mode);
                const uint64_t delta0 = rng.u64(), delta1 = rng.u64();
                const u128 delta = fss::make_delta(delta0, delta1);
                // every p in [-2^(f+n+1), 2^(f+n+1))
                const int64_t half = (int64_t)1 << (f + n + 1);
                std::vector<int64_t> ps;
                for (int64_t p = -half; p < half; p++) ps.push_back(p);
                std::vector<ppml::SiteMaterial> m0, m1;
                std::vector<AShare> a0, a1;
                std::vector<u128> sh0, sh1;
                for (int64_t p : ps) {
                    std::pair<ppml::SiteMaterial, ppml::SiteMaterial> m = ppml::deal_site(d, delta, rng);
                    m0.push_back(m.first);
                    m1.push_back(m.second);
                    const std::pair<AShare, AShare> sp = spdz2k::deal(fixedpoint::from_i64(p) & u128_mask(w), delta, w, false, rng);
                    a0.push_back(sp.first);
                    a1.push_back(sp.second);
                    const u128 r = rand_bits(rng, 128);
                    sh0.push_back(r);
                    sh1.push_back(fixedpoint::from_i64(p) - r);
                }
                const bytes sid = to_bytes("tm-test");
                auto party = [&](int role) {
                    return [&, role](Channel& ch) {
                        spdz2k::ArithChecker ck(role, ch, sid, role == 0 ? delta0 : delta1, s);
                        truncmask::Opener op(role, ch, sid, ck, role == 0 ? delta0 : delta1);
                        std::vector<truncmask::Site> sites;
                        for (const ppml::SiteMaterial& m : (role == 0 ? m0 : m1)) sites.push_back(truncmask::Site(m));
                        std::vector<truncmask::Site*> ptr;
                        for (truncmask::Site& st : sites) ptr.push_back(&st);
                        TmOut out;
                        out.res = shared ? truncmask::truncate_and_mask_shared(op, ptr, role == 0 ? sh0 : sh1, 2)
                                         : truncmask::truncate_and_mask(op, ptr, role == 0 ? a0 : a1, 2);
                        op.check();
                        out.checked = true;
                        return out;
                    };
                };
                const auto r = run_two_party(party(0), party(1));
                const std::string tag = "(" + std::to_string(n) + ", " + std::to_string(f) + ") " + mode + (shared ? " shared" : " auth");
                check(r.first.ok && r.second.ok && r.first.value.checked && r.second.value.checked, tag + ": ran, ArithCheck passed");
                if (!(r.first.ok && r.second.ok)) {
                    std::printf("    %s | %s\n", r.first.error.c_str(), r.second.error.c_str());
                    continue;
                }
                Tally t_same, t_v, t_z, t_flag;
                const uint64_t N = (uint64_t)1 << n;
                const bool ex = std::string(mode) == "exact";
                for (size_t j = 0; j < ps.size(); j++) {
                    const truncmask::TruncResult& x0 = r.first.value.res[j];
                    const truncmask::TruncResult& x1 = r.second.value.res[j];
                    t_same.add(x0.x_hat == x1.x_hat && x0.v_l == x1.v_l && x0.v == x1.v && x0.has2 == x1.has2 && x0.opened2 == x1.opened2);
                    // the opened v = p + rho' mod 2^m, and the index of plane A
                    const u128 rp = (m0[j].rp.value + m1[j].rp.value) & u128_mask(d.m());
                    t_v.add(x0.v == ((fixedpoint::from_i64(ps[j]) + rp) & u128_mask(d.m())));
                    uint64_t alpha = u128_lo(rp >> f) & (N - 1);
                    if (d.two_openings()) alpha = u128_lo(m0[j].r2.value + m1[j].r2.value) & (N - 1);
                    const uint64_t z = (x0.x_hat + N - alpha) & (N - 1);
                    t_z.add(z == truncmask::plain_truncate(fixedpoint::from_i64(ps[j]), f, n, ex, x0.v_l));
                    t_flag.add(x0.exact == ex && x0.has2 == ex);
                }
                report(t_same, tag + ": both parties see the same public values");
                report(t_v, tag + ": the first opening is p + rho' mod 2^m");
                report(t_z, tag + ": x_hat masks the truncation of p for every p");
                report(t_flag, tag + ": the exact flag and the second opening");
            }
        }
    }
}

// ------------------------------------------------------------ Beaver products and inputs
static void test_beaver() {
    Section sec("beaver: products, matrix products, authenticated inputs");
    CtrPrgRng rng(0xbea7);
    const unsigned k = 64, s = 64, w = 128;
    const uint64_t d0 = rng.u64(), d1 = rng.u64();
    const u128 delta = fss::make_delta(d0, d1);
    const ppml::Descriptor md = ppml::Descriptor::mattriple(3, 4, k, s), td = ppml::Descriptor::triples(2, k, s);
    const ppml::Descriptor im0 = ppml::Descriptor::inmask(0, 3 * 4 + 2, k, s), im1 = ppml::Descriptor::inmask(1, 4, k, s);
    const std::pair<ppml::MatTripleMaterial, ppml::MatTripleMaterial> mt = ppml::deal_mattriple(md, delta, rng);
    const std::pair<ppml::TripleMaterial, ppml::TripleMaterial> tt = ppml::deal_triples(td, delta, rng);
    const std::pair<ppml::InMaskMaterial, ppml::InMaskMaterial> i0 = ppml::deal_inmask(im0, delta, rng);
    const std::pair<ppml::InMaskMaterial, ppml::InMaskMaterial> i1 = ppml::deal_inmask(im1, delta, rng);
    std::vector<int64_t> W(12), x(4), y(2);
    for (int64_t& v : W) v = (int64_t)(rng.u64() % 2001) - 1000;
    for (int64_t& v : x) v = (int64_t)(rng.u64() % 2001) - 1000;
    y[0] = -77;
    y[1] = 5;
    const bytes sid = to_bytes("beaver-test");
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            const uint64_t dl = role == 0 ? d0 : d1;
            spdz2k::ArithChecker ck(role, ch, sid, dl, s);
            truncmask::Opener op(role, ch, sid, ck, dl);
            std::vector<u128> mine;
            if (role == 0) {
                for (int64_t v : W) mine.push_back(fixedpoint::from_i64(v));
                for (int64_t v : y) mine.push_back(fixedpoint::from_i64(v));
            } else {
                for (int64_t v : x) mine.push_back(fixedpoint::from_i64(v));
            }
            const ppml::InMaskMaterial& my = role == 0 ? i0.first : i1.second;
            const ppml::InMaskMaterial& pe = role == 0 ? i1.first : i0.second;
            const std::pair<std::vector<AShare>, std::vector<AShare>> in = beaver::exchange_inputs(op, &my, mine, &pe);
            const std::vector<AShare>& p0 = role == 0 ? in.first : in.second;
            const std::vector<AShare>& p1 = role == 0 ? in.second : in.first;
            std::vector<AShare> Wsh(p0.begin(), p0.begin() + 12), ysh(p0.begin() + 12, p0.end());
            std::vector<AShare> prod = beaver::matvec(op, Wsh, p1, beaver::mattriple_shares(role == 0 ? mt.first : mt.second));
            std::vector<AShare> pr = beaver::products(op, ysh, std::vector<AShare>(p1.begin(), p1.begin() + 2),
                                                      beaver::triple_shares(role == 0 ? tt.first : tt.second));
            prod.insert(prod.end(), pr.begin(), pr.end());
            for (AShare& a : prod) a = a.with_sign_exact(true);
            return op.open_sign_exact(prod);
        };
    };
    const auto r = run_two_party(party(0), party(1));
    check(r.first.ok && r.second.ok, "products ran (" + r.first.error + r.second.error + ")");
    if (r.first.ok && r.second.ok) {
        std::vector<int64_t> want;
        for (int i = 0; i < 3; i++) {
            int64_t acc = 0;
            for (int j = 0; j < 4; j++) acc += W[i * 4 + j] * x[j];
            want.push_back(acc);
        }
        want.push_back(y[0] * x[0]);
        want.push_back(y[1] * x[1]);
        check(r.first.value == want && r.second.value == want, "W x and y * x opened correctly at both parties");
        check(r.first.stats.rounds == r.second.stats.rounds, "rounds agree");
    }
    (void)w;
}

// ------------------------------------------------------------ inference runs
struct PartyRun {
    std::vector<int64_t> logits;
    infer::Record rec;
    ppml::Manifest manifest;
    std::map<std::string, prep::PhaseStat> phases;
    prep::PhaseStat prep_total, book_total;
    uint64_t triple_requests = 0, triples_unused = 0, triples_requested = 0;
    unsigned bucket = 0;
    bool active = false;
    bool refused_after = false;
};

typedef std::function<std::unique_ptr<infer::InferParty>(int, Session&, const nn::Model&, const infer::Config&)> PartyFactory;

static std::unique_ptr<infer::InferParty> honest_factory(int role, Session& ses, const nn::Model& m, const infer::Config& c) {
    return std::unique_ptr<infer::InferParty>(new infer::InferParty(role, ses, m, c));
}

struct RunSpec {
    std::string mode = "round";
    bool private_model = false;
    bool real = false;
    size_t cap = 0;
    unsigned threads = 2;
    std::string mode1;  // P_1's mode if different (mismatch tests)
    PartyFactory make[2] = {honest_factory, honest_factory};
    nn::Tiling tiling;  // task R-TILE
    std::map<size_t, TamperChannel::Rule> tamper[2];  // rules by message index (counted from the first message)
};

static std::pair<PartyOutcome<PartyRun>, PartyOutcome<PartyRun>> run_infer(circuit::CircuitStore& store, const nn::Model& model,
                                                                         const std::vector<int64_t>& x, const RunSpec& spec) {
    prep::IdealDealer dealer;
    const bytes sid = to_bytes("ppml-test-" + spec.mode + (spec.private_model ? "-private" : ""));
    auto party = [&](int role) {
        return [&, role](Channel& raw) {
            std::unique_ptr<TamperChannel> tc;
            Channel* ch = &raw;
            if (!spec.tamper[role].empty()) {
                tc.reset(new TamperChannel(raw, spec.tamper[role]));
                ch = tc.get();
            }
            std::unique_ptr<Session> ses;
            if (spec.real) ses.reset(new Session(role, *ch, sid, store, 64, 64, 40));
            else ses.reset(new Session(role, *ch, sid, store, dealer, 64, 64));
            infer::Config cfg;
            cfg.mode = (role == 1 && !spec.mode1.empty()) ? spec.mode1 : spec.mode;
            cfg.private_model = spec.private_model;
            cfg.triple_cap = spec.cap;
            cfg.threads = spec.threads;
            cfg.tiling = spec.tiling;
            const nn::Model shape = model.shape_only();
            const nn::Model& own = (spec.private_model && role == 1) ? shape : model;
            std::unique_ptr<infer::InferParty> p = spec.make[role](role, *ses, own, cfg);
            const std::pair<size_t, size_t> r = model.feature_range(role);
            PartyRun out;
            out.manifest = p->manifest();
            try {
                out.logits = p->run(std::vector<int64_t>(x.begin() + (ptrdiff_t)r.first, x.begin() + (ptrdiff_t)r.second));
            } catch (const ProtocolAbort&) {
                // the honest party after an abort: retired and silent
                const ChannelStats before = ch->stats();
                bool refused = false;
                try {
                    p->infer(std::vector<int64_t>(r.second - r.first, 0));
                } catch (const ProtocolAbort&) {
                    refused = true;
                } catch (...) {
                }
                const ChannelStats after = ch->stats();
                if (!(refused && !p->active() && !ses->active() && after.msgs_sent == before.msgs_sent && p->record().logits.empty()))
                    throw std::runtime_error("the aborted party is not retired and silent");
                throw;
            }
            out.rec = p->record();
            abb::PhaseBook& book = ses->book();
            for (size_t i = 0; i < book.size(); i++) out.phases[book.name(i)] = book.stat(i);
            out.book_total = book.total();
            if (ses->real_prep()) {
                out.prep_total = ses->real_prep()->stats().total();
                out.bucket = ses->real_prep()->last_bucket_size();
            }
            out.triple_requests = ses->triple_requests();
            out.triples_unused = ses->triples_unused();
            out.triples_requested = ses->triples_requested();
            out.active = p->active();
            return out;
        };
    };
    return run_two_party(party(0), party(1));
}

static prep::PhaseStat ph(const PartyRun& r, const std::string& name) {
    const std::map<std::string, prep::PhaseStat>::const_iterator it = r.phases.find(name);
    return it == r.phases.end() ? prep::PhaseStat() : it->second;
}

// Bit-exactness and the counters of one honest run (the counterpart of test_nn.check_run).
static void check_run(const nn::Model& model, const std::string& mode, const std::vector<int64_t>& x, const std::vector<int64_t>* exact_want,
                      const std::pair<PartyOutcome<PartyRun>, PartyOutcome<PartyRun>>& r, const std::string& tag, bool priv,
                      const nn::Tiling* tiling = nullptr) {
    check(r.first.ok && r.second.ok, tag + ": ran (" + r.first.error + " | " + r.second.error + ")");
    if (!(r.first.ok && r.second.ok)) return;
    const PartyRun &a = r.first.value, &b = r.second.value;
    check(a.logits == b.logits, tag + ": both parties hold the same logits");
    check(a.rec.thresholds == b.rec.thresholds, tag + ": the same public thresholds");
    const nn::MirrorResult mir = model.mirror(x, mode == "exact", &a.rec.thresholds);
    check(a.logits == nn::to_i64(mir.logits), tag + ": bit-exact agreement with the mirror");
    if (exact_want) check(a.logits == *exact_want, tag + ": bit-exact with the exported logits");
    check(a.triple_requests == 1 && a.triples_unused == 0 && b.triple_requests == 1 && b.triples_unused == 0,
          tag + ": one triple request, nothing left");
    check(a.rec.executions == a.manifest.entries.size(), tag + ": one execution per plan entry");
    const std::vector<size_t> widths = model.hidden_widths();
    size_t all = 0, later = 0;
    for (size_t i = 0; i < widths.size(); i++) {
        all += widths[i];
        if (i) later += widths[i];
    }
    check(a.rec.sites == all, tag + ": sites");
    uint64_t want;
    if (priv) {
        want = all * (mode == "exact" ? 2 : 1);
        check(a.rec.tiled == (tiling != nullptr), tag + ": tiled as expected");
        for (const nn::Shape& sh : model.shapes) {
            if (!tiling) {
                want += sh.first * sh.second + sh.second;
                continue;
            }
            for (const nn::Tile& t : nn::layer_tiles(sh.first, sh.second, *tiling)) want += t.rows * t.cols + t.cols;  // D_t, e_t
        }
    } else {
        want = mode == "exact" ? later + all : later;
    }
    check(a.rec.openings_before_release == want, tag + ": deferred openings before the release");
    check(a.rec.openings == want + model.n_out(), tag + ": the release opens the logits");
    check(a.rec.checks == (want ? 2u : 1u), tag + ": a check per non-empty batch of openings");
    if (!priv) check(ph(a, "tm-open").rounds == 2, tag + ": the first layer is one commit-then-open");
    check(ph(a, "arith-check").rounds == (want ? 8u : 4u), tag + ": four rounds per check");
    check(ph(a, "manifest").bytes_sent == 32 && ph(a, "manifest").rounds == 1, tag + ": the manifest digest");
    check(a.rec.released.size() == model.n_out(), tag + ": the released representatives");
    bool low_ok = true;
    for (size_t i = 0; i < a.rec.released.size(); i++)
        low_ok = low_ok && fixedpoint::to_signed(a.rec.released[i], 64) == a.logits[i];
    check(low_ok, tag + ": the logits are the low k bits of the release");
    if (!priv) {
        // public model: the release is unmasked and sign-exact (the high s bits are the sign extension)
        bool se = true;
        for (size_t i = 0; i < a.rec.released.size(); i++)
            se = se && a.rec.released[i] == (fixedpoint::from_i64(a.logits[i]) & u128_mask(128));
        check(se, tag + ": the public release is the sign-exact logit");
    }
}

static void test_models(circuit::CircuitStore& store) {
    Section sec("models end to end with ideal preprocessing (k = s = 64)");
    const char* names[] = {"mlp4", "mlp16", "mlp32", "cnn10", "mlp64"};
    for (const char* nm : names) {
        const nn::Model model = nn::Model::load(g_vectors + "/" + nm + "_model.json");
        const nn::Vectors vec = nn::Vectors::load(g_vectors + "/" + nm + "_vectors.json");
        for (const char* mode : {"round", "exact"}) {
            if (g_quick && std::string(nm) == "mlp64" && std::string(mode) == "exact") continue;
            RunSpec spec;
            spec.mode = mode;
            spec.threads = 4;
            const size_t idx = std::string(mode) == "round" ? 1 : 0;
            const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            const auto r = run_infer(store, model, vec.inputs[idx], spec);
            const double sec_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            check_run(model, mode, vec.inputs[idx], std::string(mode) == "exact" ? &vec.exact_logits[idx] : nullptr, r,
                      std::string(nm) + " " + mode, false);
            if (r.first.ok)
                std::printf("    %-6s %-5s %3zu sites, %10zu AND, plane eval %.3f s, online %.3f s, total %.2f s\n", nm, mode,
                            r.first.value.rec.sites, r.first.value.rec.and_gates, r.first.value.rec.plane_eval_s,
                            r.first.value.rec.infer_s, sec_);
        }
    }
    // private-model mode (output masks)
    const nn::Model mlp4 = nn::Model::load(g_vectors + "/mlp4_model.json");
    const nn::Vectors v4 = nn::Vectors::load(g_vectors + "/mlp4_vectors.json");
    for (const char* mode : {"round", "exact"}) {
        RunSpec spec;
        spec.mode = mode;
        spec.private_model = true;
        const auto r = run_infer(store, mlp4, v4.inputs[2], spec);
        check_run(mlp4, mode, v4.inputs[2], std::string(mode) == "exact" ? &v4.exact_logits[2] : nullptr, r,
                  std::string("mlp4 private ") + mode, true);
        if (r.first.ok) {
            const PartyRun& a = r.first.value;
            check(a.manifest.private_model && a.manifest.entries.back().desc.kind == ppml::KIND_OUTMASK,
                  std::string("mlp4 private ") + mode + ": the plan ends with the output masks");
            check(ph(a, "input-open").rounds == 1 && ph(a, "tm-open").rounds == 0,
                  std::string("mlp4 private ") + mode + ": one input exchange, no commit-then-open");
            // the high s bits of the release are masked: they are not the sign extension of the logit
            bool masked = false;
            for (size_t i = 0; i < a.rec.released.size(); i++)
                masked = masked || (a.rec.released[i] >> 64) != (u128(fixedpoint::is_negative(fixedpoint::from_i64(a.logits[i])) ? ~(uint64_t)0 : 0));
            check(masked, std::string("mlp4 private ") + mode + ": the release carries the output masks");
        }
    }
    // a triple cap: mlp16 round, 16 sites in chunks of 3 -> six requests
    {
        const nn::Model mlp16 = nn::Model::load(g_vectors + "/mlp16_model.json");
        const nn::Vectors v16 = nn::Vectors::load(g_vectors + "/mlp16_vectors.json");
        RunSpec spec;
        spec.cap = 3 * ppml::and_count(store, ppml::Descriptor::site(16, 8, 64, 64, "round"));
        const auto r = run_infer(store, mlp16, v16.inputs[3], spec);
        check(r.first.ok && r.second.ok, "triple cap: ran");
        if (r.first.ok && r.second.ok) {
            const PartyRun& a = r.first.value;
            check(a.logits == r.second.value.logits && a.logits == nn::to_i64(mlp16.mirror(v16.inputs[3], false, &a.rec.thresholds).logits),
                  "triple cap: bit-exact");
            check(a.triple_requests == 6 && a.triples_unused == 0 && a.rec.executions == 6 && r.second.value.triple_requests == 6,
                  "triple cap: 16 sites in chunks of 3, six requests and executions, nothing left");
        }
    }
}

static void test_real(circuit::CircuitStore& store) {
    Section sec("real preprocessing: mlp4 (8, 4), mode round");
    const nn::Model mlp4 = nn::Model::load(g_vectors + "/mlp4_model.json");
    const nn::Vectors v4 = nn::Vectors::load(g_vectors + "/mlp4_vectors.json");
    RunSpec spec;
    spec.real = true;
    const auto r = run_infer(store, mlp4, v4.inputs[4], spec);
    check_run(mlp4, "round", v4.inputs[4], nullptr, r, "mlp4 real round", false);
    if (r.first.ok && r.second.ok) {
        for (int b = 0; b < 2; b++) {
            const PartyRun& p = b ? r.second.value : r.first.value;
            const ChannelStats& st = b ? r.second.stats : r.first.stats;
            check(p.prep_total.bytes_sent + p.book_total.bytes_sent == st.bytes_sent &&
                      p.prep_total.bytes_recv + p.book_total.bytes_recv == st.bytes_recv &&
                      p.prep_total.rounds + p.book_total.rounds == st.rounds,
                  "real: the phases sum to the channel totals at P_" + std::to_string(b));
            check(p.bucket >= 3, "real: bucketed triples");
        }
        check(r.first.stats.bytes_sent == r.second.stats.bytes_recv && r.first.stats.rounds == r.second.stats.rounds,
              "real: sent equals received, rounds agree");
        std::printf("    %llu bytes sent by P_0, %llu rounds, bucket size %u, %.1f s\n", (unsigned long long)r.first.stats.bytes_sent,
                    (unsigned long long)r.first.stats.rounds, r.first.value.bucket, r.first.seconds);
    }
}

// ------------------------------------------------------------ adversaries
class CheatingChecker : public spdz2k::ArithChecker {
public:
    CheatingChecker(int role, Channel& ch, const bytes& sid, uint64_t delta, unsigned s, abb::PhaseBook* book, uint64_t index,
                    u128 error)
        : spdz2k::ArithChecker(role, ch, sid, delta, s, book), index_(index), error_(error) {}

protected:
    std::vector<u128> outgoing_values(const std::vector<AShare>& shares, uint64_t first) override {
        std::vector<u128> v = spdz2k::ArithChecker::outgoing_values(shares, first);
        if (index_ >= first && index_ < first + shares.size()) v[(size_t)(index_ - first)] = v[(size_t)(index_ - first)] + error_;
        return v;
    }

private:
    uint64_t index_;
    u128 error_;
};

class CheatingParty : public infer::InferParty {
public:
    CheatingParty(int role, Session& ses, const nn::Model& m, const infer::Config& c, uint64_t index, u128 error)
        : infer::InferParty(role, ses, m, c), index_(index), error_(error) {}

protected:
    std::unique_ptr<spdz2k::ArithChecker> make_checker(int role, Channel& ch, const bytes& sid, uint64_t delta, unsigned s,
                                                       abb::PhaseBook* book) override {
        return std::unique_ptr<spdz2k::ArithChecker>(new CheatingChecker(role, ch, sid, delta, s, book, index_, error_));
    }

private:
    uint64_t index_;
    u128 error_;
};

// Evaluates, for site 0 of layer 1, a plane A key that was not generated in the session.
class WrongKeyParty : public infer::InferParty {
public:
    using infer::InferParty::InferParty;

protected:
    std::vector<truncmask::Site*> layer_sites(size_t layer) override {
        std::vector<truncmask::Site*> s = infer::InferParty::layer_sites(layer);
        if (layer == 0 && !s.empty()) {
            CtrPrgRng rng(4242);
            fake_ = *s[0];
            const ppml::Descriptor& d = fake_.mat.desc;
            fake_.mat.plane_a = fss::dpf_gen(rng.u64() & ((1ull << d.n) - 1), 1, u128(rng.u64()), d.n, d.k, d.s, rng).first;
            fake_.mat.plane_a.b = (uint8_t)role_;
            s[0] = &fake_;
        }
        return s;
    }

private:
    truncmask::Site fake_;
};

// P_0 that inputs the logit-layer weights W = 2^k e_j instead of its model (private-model mode).
class EvilWeightsParty : public infer::InferParty {
public:
    EvilWeightsParty(int role, Session& ses, const nn::Model& honest, const infer::Config& c, const nn::Model& evil)
        : infer::InferParty(role, ses, honest, c) {
        model_ = &evil;
    }
};

static void test_tamper(circuit::CircuitStore& store) {
    Section sec("tamper cases abort");
    const nn::Model mlp4 = nn::Model::load(g_vectors + "/mlp4_model.json");
    const nn::Vectors v4 = nn::Vectors::load(g_vectors + "/mlp4_vectors.json");
    const std::vector<int64_t>& x = v4.inputs[5];
    // message counts of an honest run (ideal preprocessing is deterministic in the number of messages)
    RunSpec base;
    const auto honest = run_infer(store, mlp4, x, base);
    check(honest.first.ok, "the honest reference run");
    RunSpec base_priv;
    base_priv.private_model = true;
    const auto honest_priv = run_infer(store, mlp4, x, base_priv);
    check(honest_priv.first.ok, "the honest reference run (private model)");
    if (!(honest.first.ok && honest_priv.first.ok)) return;
    const uint64_t online0 = honest.first.stats.msgs_sent - honest.first.value.phases.at("tm-open").msgs_sent -
                             ph(honest.first.value, "arith-open").msgs_sent - ph(honest.first.value, "arith-check").msgs_sent;
    // online0 = messages of P_0 before the online phase; the online phase starts with the commitment
    struct Case {
        std::string name;
        RunSpec spec;
    };
    std::vector<Case> cases;
    {
        Case c;
        c.name = "a mismatching plan (P_1 in mode exact)";
        c.spec.mode1 = "exact";
        cases.push_back(c);
    }
    {
        Case c;
        c.name = "a flipped bit in the opening of the first-layer commitment";
        c.spec.tamper[0][online0 + 1] = flip_bit_rule(32 * 8 + 3);  // after omega: the first masked share
        cases.push_back(c);
    }
    {
        Case c;
        c.name = "a wrong value share (+1) in the first deferred opening";
        c.spec.make[1] = [](int role, Session& ses, const nn::Model& m, const infer::Config& cfg) {
            return std::unique_ptr<infer::InferParty>(new CheatingParty(role, ses, m, cfg, 0, u128(1)));
        };
        cases.push_back(c);
    }
    {
        Case c;
        c.name = "a wrong value share (+1) in the release";
        const uint64_t idx = honest.first.value.rec.openings_before_release;
        c.spec.make[0] = [idx](int role, Session& ses, const nn::Model& m, const infer::Config& cfg) {
            return std::unique_ptr<infer::InferParty>(new CheatingParty(role, ses, m, cfg, idx, u128(1)));
        };
        cases.push_back(c);
    }
    {
        Case c;
        c.name = "a substituted plane key";
        c.spec.make[1] = [](int role, Session& ses, const nn::Model& m, const infer::Config& cfg) {
            return std::unique_ptr<infer::InferParty>(new WrongKeyParty(role, ses, m, cfg));
        };
        cases.push_back(c);
    }
    {
        Case c;
        c.name = "mode exact: a wrong value share (+1) in the second opening of layer 1";
        c.spec.mode = "exact";
        // the first deferred opening of mode exact is the opening of x + r' of layer 1
        c.spec.make[0] = [](int role, Session& ses, const nn::Model& m, const infer::Config& cfg) {
            return std::unique_ptr<infer::InferParty>(new CheatingParty(role, ses, m, cfg, 0, u128(1) << 3));
        };
        cases.push_back(c);
    }
    {
        Case c;
        c.name = "private model: a flipped bit in the input exchange";
        c.spec.private_model = true;
        const uint64_t before = honest_priv.first.stats.msgs_sent - ph(honest_priv.first.value, "input-open").msgs_sent -
                                ph(honest_priv.first.value, "arith-open").msgs_sent - ph(honest_priv.first.value, "arith-check").msgs_sent;
        c.spec.tamper[0][before] = flip_bit_rule(5);
        cases.push_back(c);
    }
    for (const Case& c : cases) {
        const auto r = run_infer(store, mlp4, x, c.spec);
        // the honest party: the one without a deviation (both for the plan mismatch)
        const bool dev0 = !c.spec.tamper[0].empty() || c.name.find("release") != std::string::npos ||
                          c.name.find("second opening") != std::string::npos;
        const PartyOutcome<PartyRun>& h = dev0 ? r.second : r.first;
        check(!h.ok && h.protocol_abort, c.name + ": the honest party throws ProtocolAbort (" + h.error + ")");
        check(h.error.find("not retired") == std::string::npos, c.name + ": and is retired and silent afterwards");
        if (c.name.find("mismatching") != std::string::npos) {
            check(r.first.protocol_abort && r.second.protocol_abort && r.first.stats.bytes_sent == 32 && r.second.stats.bytes_sent == 32,
                  c.name + ": both abort after the 32-byte digest");
        }
    }
    // private model, a malicious P_0 with large logit weights: the honest output is the functionality's
    // (the low k bits of the logit) and the opened high bits are masked.  With k = 64 the weight 2^k
    // does not fit the int64 model file, so the weights are 2^62 e_j (the statistical test of the
    // high bits with W = 2^k e_j at k = 32 is python/tests/test_ppml_attacks.py).
    {
        nn::Model evil = mlp4;
        nn::Layer& last = evil.layers.back();
        const unsigned j = 0;
        for (size_t i = 0; i < last.rows; i++)
            for (size_t c = 0; c < last.cols; c++) last.W[i * last.cols + c] = c == j ? ((int64_t)1 << 62) : 0;
        RunSpec spec;
        spec.private_model = true;
        spec.make[0] = [&evil](int role, Session& ses, const nn::Model& m, const infer::Config& cfg) {
            return std::unique_ptr<infer::InferParty>(new EvilWeightsParty(role, ses, m, cfg, evil));
        };
        const auto r = run_infer(store, mlp4, x, spec);
        check(r.first.ok && r.second.ok, "malicious logit weights: the run completes (" + r.first.error + r.second.error + ")");
        if (r.first.ok && r.second.ok) {
            const nn::MirrorResult mir = evil.mirror(x, false, &r.second.value.rec.thresholds);
            std::vector<int64_t> want;
            for (const u128& v : mir.logits) want.push_back(fixedpoint::to_signed(v, 64));
            check(r.second.value.logits == want && r.first.value.logits == want,
                  "malicious logit weights: the honest output is the functionality's (low k bits)");
            // the high s bits of the release are masked: not the high bits of the unmasked logit
            bool differs = true;
            for (size_t i = 0; i < mir.logits.size(); i++) differs = differs && (r.second.value.rec.released[i] >> 64) != (mir.logits[i] >> 64);
            check(differs, "malicious logit weights: the opened high bits are not those of 2^62 a_j + b 2^f");
        }
    }
}

// ------------------------------------------------------------ tiled private-model plans (task R-TILE)
// Drops the tile entries of one shape from the plan (both parties do it, so the digests agree).
class TileDropParty : public infer::InferParty {
public:
    TileDropParty(int role, Session& ses, const nn::Model& m, const infer::Config& c, size_t rows, size_t cols)
        : infer::InferParty(role, ses, m, c) {
        std::vector<ppml::ManifestEntry> kept;
        for (const ppml::ManifestEntry& e : manifest_.entries)
            if (!(e.desc.kind == ppml::KIND_MATTRIPLE && e.desc.rows == rows && e.desc.cols == cols)) kept.push_back(e);
        manifest_.entries = kept;
    }
};

static void test_tiling(circuit::CircuitStore& store) {
    Section sec("tiling: tiled private-model plans");
    // the layout
    struct Lay {
        size_t rows, cols, tr, tc;
    };
    for (const Lay& c : {Lay{37, 21, 16, 16}, Lay{128, 784, 16, 16}, Lay{10, 128, 16, 16}, Lay{5, 7, 2, 3}, Lay{1, 1, 4, 4},
                         Lay{64, 48, 16, 16}, Lay{3, 7, 2, 2}}) {
        const nn::Tiling t(c.tr, c.tc, 0);
        const std::vector<nn::Tile> tiles = nn::layer_tiles(c.rows, c.cols, t);
        std::vector<int> cover(c.rows * c.cols, 0);
        bool sizes = true, order = true;
        for (size_t i = 0; i < tiles.size(); i++) {
            for (size_t a = 0; a < tiles[i].rows; a++)
                for (size_t b = 0; b < tiles[i].cols; b++) cover[(tiles[i].r0 + a) * c.cols + tiles[i].c0 + b]++;
            sizes = sizes && tiles[i].rows >= 1 && tiles[i].rows <= c.tr && tiles[i].cols >= 1 && tiles[i].cols <= c.tc;
            if (i) order = order && (tiles[i - 1].r0 < tiles[i].r0 || (tiles[i - 1].r0 == tiles[i].r0 && tiles[i - 1].c0 < tiles[i].c0));
        }
        const std::string tag = std::to_string(c.rows) + "x" + std::to_string(c.cols);
        check(std::count(cover.begin(), cover.end(), 1) == (ptrdiff_t)cover.size(), tag + ": the tiles partition the matrix");
        check(sizes && order, tag + ": tiles of at most the tile, row-major");
        size_t n = 0;
        for (const auto& e : nn::tile_shapes(c.rows, c.cols, t)) n += e.second;
        check(n == tiles.size(), tag + ": shape counts");
    }
    const std::vector<std::pair<nn::Shape, size_t>> sh = nn::tile_shapes(37, 21, nn::Tiling());
    check(sh.size() == 4 && sh[0] == std::make_pair(nn::Shape(16, 16), (size_t)2) && sh[1] == std::make_pair(nn::Shape(16, 5), (size_t)2) &&
              sh[2] == std::make_pair(nn::Shape(5, 16), (size_t)1) && sh[3] == std::make_pair(nn::Shape(5, 5), (size_t)1),
          "37x21: the shapes in the order of their first tile");
    // the plan of mlp48 [48, 64, 64, 10] (7,808 weights > 4,096): default tiling
    const nn::Model mlp48 = nn::Model::load(g_vectors + "/mlp48_model.json");
    check(mlp48.n_weights() == 7808 && nn::plan_tiled(mlp48, true) && !nn::plan_tiled(mlp48, false), "mlp48: tiled in private-model mode");
    {
        const ppml::Manifest man = nn::plan_manifest(mlp48.shape_only(), "round", 64, 64, true);
        std::string got;
        for (const ppml::ManifestEntry& e : man.entries) got += e.desc.name() + "*" + std::to_string(e.batch) + " ";
        check(got == "inmask_p0_c256_k64_s64*31 inmask_p0_c34_k64_s64*1 inmask_p1_c24_k64_s64*1 mattriple_16x16_k64_s64*12 "
                     "mattriple_16x16_k64_s64*16 mattriple_10x16_k64_s64*4 site_n16_f8_k64_s64_round*64 site_n16_f8_k64_s64_round*64 "
                     "gate_n1_k64_s64_shared_om1*10 ",
              "mlp48: the tiled plan (" + got + ")");
        check(man.entries[3].use == "matrix triple tiles of layer 1: 64x48 in tiles of 16x16" &&
                  man.entries[0].use == "inputs of P_0: features, weights, biases, 256 per instance" &&
                  man.entries[1].use == "inputs of P_0: features, weights, biases, the last 34",
              "mlp48: the use strings");
        check(31 * 256 + 34 == 24 + 7808 + 64 + 64 + 10, "mlp48: one mask per input of P_0");
    }
    // matvec_tiled on dealt material
    CtrPrgRng rng(0x711e);
    const unsigned k = 32, s = 64, w = 96;
    struct Case {
        size_t rows, cols, tr, tc;
        long long tamper;  // index of a wrong value share of P_0 in the tile opening, or -1
    };
    const std::vector<nn::Tile> t3721 = nn::layer_tiles(37, 21, nn::Tiling(16, 16, 0));
    size_t d3 = 0, e_last = 0;  // an entry of D of the third tile, the last e entry
    for (size_t i = 0; i < t3721.size(); i++) {
        if (i < 2) d3 += t3721[i].rows * t3721[i].cols + t3721[i].cols;
        e_last += t3721[i].rows * t3721[i].cols + t3721[i].cols;
    }
    for (const Case& c : {Case{37, 21, 16, 16, -1}, Case{64, 48, 16, 16, -1}, Case{10, 128, 16, 16, -1}, Case{5, 7, 2, 3, -1},
                          Case{3, 7, 2, 2, -1}, Case{16, 16, 16, 16, -1}, Case{1, 9, 4, 4, -1}, Case{9, 1, 4, 4, -1},
                          Case{37, 21, 16, 16, (long long)d3 + 5}, Case{37, 21, 16, 16, (long long)e_last - 1}}) {
        const uint64_t d0 = rng.u64(), d1 = rng.u64();
        const u128 delta = fss::make_delta(d0, d1);
        const nn::Tiling tl(c.tr, c.tc, 0);
        const std::vector<nn::Tile> tiles = nn::layer_tiles(c.rows, c.cols, tl);
        std::vector<std::pair<ppml::MatTripleMaterial, ppml::MatTripleMaterial>> mats;
        for (const nn::Tile& t : tiles) mats.push_back(ppml::deal_mattriple(ppml::Descriptor::mattriple(t.rows, t.cols, k, s), delta, rng));
        std::vector<int64_t> W(c.rows * c.cols), x(c.cols);
        for (int64_t& v : W) v = (int64_t)(rng.u64() % 2049) - 1024;
        for (int64_t& v : x) v = (int64_t)(rng.u64() % 2049) - 1024;
        std::vector<std::pair<AShare, AShare>> sW, sx;
        for (int64_t v : W) {
            const auto sp = ppml::detail::split(fixedpoint::from_i64(v) & u128_mask(w), delta, w, rng);
            sW.push_back(std::make_pair(AShare(sp.first.value, sp.first.tag, w), AShare(sp.second.value, sp.second.tag, w)));
        }
        for (int64_t v : x) {
            const auto sp = ppml::detail::split(fixedpoint::from_i64(v) & u128_mask(w), delta, w, rng);
            sx.push_back(std::make_pair(AShare(sp.first.value, sp.first.tag, w), AShare(sp.second.value, sp.second.tag, w)));
        }
        const bytes sid = to_bytes("tiled-test");
        struct Out {
            std::vector<int64_t> y;
            uint64_t openings = 0, rounds = 0;
        };
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                const uint64_t dl = role == 0 ? d0 : d1;
                std::unique_ptr<spdz2k::ArithChecker> ck;
                if (c.tamper >= 0 && role == 0) ck.reset(new CheatingChecker(role, ch, sid, dl, s, nullptr, (uint64_t)c.tamper, u128(1)));
                else ck.reset(new spdz2k::ArithChecker(role, ch, sid, dl, s));
                truncmask::Opener op(role, ch, sid, *ck, dl);
                std::vector<AShare> Ws, xs;
                for (const auto& p2 : sW) Ws.push_back(role == 0 ? p2.first : p2.second);
                for (const auto& p2 : sx) xs.push_back(role == 0 ? p2.first : p2.second);
                std::vector<beaver::TileTriple> tt;
                for (size_t i = 0; i < tiles.size(); i++) {
                    beaver::TileTriple x2;
                    x2.r0 = tiles[i].r0;
                    x2.c0 = tiles[i].c0;
                    x2.t = beaver::mattriple_shares(role == 0 ? mats[i].first : mats[i].second);
                    tt.push_back(x2);
                }
                const uint64_t r0 = ch.stats().rounds;
                std::vector<AShare> y = beaver::matvec_tiled(op, Ws, c.rows, c.cols, xs, tt);
                Out o;
                o.rounds = ch.stats().rounds - r0;
                for (AShare& a : y) a = a.with_sign_exact(true);
                o.y = op.open_sign_exact(y);
                o.openings = ck->openings();
                return o;
            };
        };
        const auto r = run_two_party(party(0), party(1));
        const std::string tag = std::to_string(c.rows) + "x" + std::to_string(c.cols) + " in tiles of " + std::to_string(c.tr) + "x" +
                                std::to_string(c.tc) + (c.tamper >= 0 ? " (tampered entry " + std::to_string(c.tamper) + ")" : "");
        if (c.tamper >= 0) {
            check(!r.second.ok && r.second.protocol_abort, tag + ": the honest party aborts at the check (" + r.second.error + ")");
            continue;
        }
        check(r.first.ok && r.second.ok, tag + ": ran (" + r.first.error + r.second.error + ")");
        if (!(r.first.ok && r.second.ok)) continue;
        std::vector<int64_t> want(c.rows, 0);
        for (size_t i = 0; i < c.rows; i++)
            for (size_t j = 0; j < c.cols; j++) want[i] += W[i * c.cols + j] * x[j];
        check(r.first.value.y == want && r.second.value.y == want, tag + ": the tiled product equals the plaintext product");
        size_t n_open = 0;
        for (const nn::Tile& t : tiles) n_open += t.rows * t.cols + t.cols;
        check(r.first.value.openings == n_open + c.rows, tag + ": openings D_t, e_t of every tile and the release");
        check(r.first.value.rounds == 1 && r.second.value.rounds == 1, tag + ": one exchange for all tiles");
    }
    // malformed tile lists are refused before any message
    {
        std::pair<std::unique_ptr<LoopbackChannel>, std::unique_ptr<LoopbackChannel>> chs = LoopbackChannel::make_pair();
        spdz2k::ArithChecker ck(0, *chs.first, to_bytes("x"), 1, 64);
        truncmask::Opener op(0, *chs.first, to_bytes("x"), ck, 1);
        const u128 dl = fss::make_delta(1, 2);
        const beaver::MatTriple m22 = beaver::mattriple_shares(ppml::deal_mattriple(ppml::Descriptor::mattriple(2, 2, k, s), dl, rng).first);
        const beaver::MatTriple m21 = beaver::mattriple_shares(ppml::deal_mattriple(ppml::Descriptor::mattriple(2, 1, k, s), dl, rng).first);
        const std::vector<AShare> W6(6, AShare(u128(0), u128(0), w)), x3(3, AShare(u128(0), u128(0), w));
        auto tile = [](size_t r0, size_t c0, const beaver::MatTriple& t) {
            beaver::TileTriple x2;
            x2.r0 = r0;
            x2.c0 = c0;
            x2.t = t;
            return x2;
        };
        check(throws_invalid([&] { beaver::matvec_tiled(op, W6, 2, 3, x3, {tile(0, 0, m22)}); }), "a gap");
        check(throws_invalid([&] { beaver::matvec_tiled(op, W6, 2, 3, x3, {tile(0, 0, m22), tile(0, 1, m22)}); }), "an overlap");
        check(throws_invalid([&] { beaver::matvec_tiled(op, W6, 2, 3, x3, {tile(0, 0, m22), tile(0, 2, m22)}); }), "outside");
        check(throws_invalid([&] { beaver::matvec_tiled(op, W6, 2, 3, x3, {tile(0, 0, m22), tile(0, 2, m21), tile(0, 2, m21)}); }),
              "a tile twice");
        check(chs.first->stats().msgs_sent == 0, "nothing was sent");
        // merged input masks
        std::vector<ppml::InMaskMaterial> parts0, parts1;
        std::vector<protocol::SharePair> all0;
        std::vector<u128> vals0;
        for (size_t c : {4, 4, 3}) {
            auto im = ppml::deal_inmask(ppml::Descriptor::inmask(0, c, k, s), dl, rng);
            all0.insert(all0.end(), im.first.shares.begin(), im.first.shares.end());
            vals0.insert(vals0.end(), im.first.values.begin(), im.first.values.end());
            parts0.push_back(im.first);
            parts1.push_back(im.second);
        }
        const ppml::InMaskMaterial m0 = beaver::merge_inmasks(parts0), m1 = beaver::merge_inmasks(parts1);
        bool same = m0.shares.size() == all0.size();
        for (size_t i = 0; same && i < all0.size(); i++) same = m0.shares[i].value == all0[i].value && m0.shares[i].tag == all0[i].tag;
        check(same && m0.values == vals0 && m0.count() == 11 && m0.has_values && !m1.has_values && m1.count() == 11 && m1.owner() == 0,
              "merged input masks: the instances in order, the total count");
    }
    // mlp7 [7, 3, 2] at (8, 4) with 2 x 2 tiles, both modes, ideal preprocessing
    const nn::Model mlp7 = nn::Model::load(g_vectors + "/mlp7_model.json");
    const nn::Vectors v7 = nn::Vectors::load(g_vectors + "/mlp7_vectors.json");
    const nn::Tiling t22(2, 2, 0);
    for (const char* mode : {"round", "exact"}) {
        RunSpec spec;
        spec.mode = mode;
        spec.private_model = true;
        spec.tiling = t22;
        const size_t idx = std::string(mode) == "round" ? 3 : 4;
        const auto r = run_infer(store, mlp7, v7.inputs[idx], spec);
        check_run(mlp7, mode, v7.inputs[idx], std::string(mode) == "exact" ? &v7.exact_logits[idx] : nullptr, r,
                  std::string("mlp7 tiled 2x2 ") + mode, true, &t22);
        if (r.first.ok && r.second.ok) {
            const infer::Record& a = r.first.value.rec;
            check(a.tiled && a.tiles == 10 && a.tiles_per_layer == std::vector<size_t>({8, 2}) && r.second.value.rec.tiles == 10,
                  std::string("mlp7 tiled ") + mode + ": ten tile triples (8 + 2)");
            check(ph(r.first.value, "input-open").rounds == 1, std::string("mlp7 tiled ") + mode + ": one input exchange");
        }
    }
    // a plan without the 1 x 1 tile entry (both parties agree on it): the generation aborts
    {
        RunSpec spec;
        spec.private_model = true;
        spec.tiling = t22;
        for (int b = 0; b < 2; b++)
            spec.make[b] = [](int role, Session& ses, const nn::Model& m, const infer::Config& cfg) {
                return std::unique_ptr<infer::InferParty>(new TileDropParty(role, ses, m, cfg, 1, 1));
            };
        const auto r = run_infer(store, mlp7, v7.inputs[0], spec);
        check(!r.first.ok && r.first.protocol_abort && !r.second.ok && r.second.protocol_abort &&
                  r.first.error.find("do not match the tiles") != std::string::npos,
              "a missing tile entry aborts the generation at both parties (" + r.first.error + ")");
    }
    // a wrong value share in a tile opening of the pipeline (P_1 deviates): the honest P_0 aborts
    {
        const std::vector<nn::Tile> tl1 = nn::layer_tiles(3, 7, t22);
        const uint64_t e_first = tl1[0].rows * tl1[0].cols;                       // e of the first tile
        const uint64_t d_third = 2 * (tl1[0].rows * tl1[0].cols + tl1[0].cols) + 1;  // D of the third tile
        for (uint64_t idx : {e_first, d_third}) {
            RunSpec spec;
            spec.private_model = true;
            spec.tiling = t22;
            spec.make[1] = [idx](int role, Session& ses, const nn::Model& m, const infer::Config& cfg) {
                return std::unique_ptr<infer::InferParty>(new CheatingParty(role, ses, m, cfg, idx, u128(1)));
            };
            const auto r = run_infer(store, mlp7, v7.inputs[1], spec);
            check(!r.first.ok && r.first.protocol_abort && r.first.error.find("not retired") == std::string::npos,
                  "a wrong value share in entry " + std::to_string(idx) + " of the tile opening: the honest party aborts (" +
                      r.first.error + ")");
        }
    }
}

static void test_arguments(circuit::CircuitStore& store) {
    Section sec("arguments");
    const nn::Model mlp4 = nn::Model::load(g_vectors + "/mlp4_model.json");
    std::pair<std::unique_ptr<LoopbackChannel>, std::unique_ptr<LoopbackChannel>> chs = LoopbackChannel::make_pair();
    prep::IdealDealer dealer;
    Session ses(0, *chs.first, to_bytes("args"), store, dealer, 64, 64);
    infer::Config cfg;
    cfg.mode = "nearest";
    check(throws_invalid([&] { infer::InferParty p(0, ses, mlp4, cfg); }), "mode");
    cfg.mode = "round";
    check(throws_invalid([&] { infer::InferParty p(1, ses, mlp4, cfg); }), "role");
    cfg.private_model = true;
    const nn::Model shape = mlp4.shape_only();
    check(throws_invalid([&] { infer::InferParty p(0, ses, shape, cfg); }), "P_0 needs the weights");
    cfg.private_model = false;
    check(throws_invalid([&] { infer::InferParty p(0, ses, shape, cfg); }), "the public pipeline needs the weights");
    nn::Model loud = mlp4;
    for (int64_t& v : loud.layers.back().W) v = v * ((int64_t)1 << 59);
    check(!loud.release_fits(64) && throws_invalid([&] { infer::InferParty p(0, ses, loud, cfg); }),
          "logits that do not fit a signed payload of k bits are refused");
    infer::InferParty p(0, ses, mlp4, cfg);
    check(throws_invalid([&] { p.preprocess(); }) && throws_invalid([&] { p.infer(std::vector<int64_t>(2, 0)); }), "order of the steps");
    check(chs.first->stats().msgs_sent == 0 && p.active(), "no message was sent and the party is usable");
    check(throws_invalid([&] { mlp4.first_layer_share(0, std::vector<int64_t>(3, 0)); }), "wrong slice length");
    check(throws_invalid([&] { mlp4.mirror(std::vector<int64_t>(4, 0), false); }), "mode round needs thresholds");
}

int main(int argc, char** argv) {
    g_circuits = circuit::dirname(argv[0]) + "/../../circuits";
    g_vectors = circuit::dirname(argv[0]) + "/../../python/vectors/ppml";
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--quick") g_quick = true;
        else if (arg == "--circuits" && i + 1 < argc) g_circuits = argv[++i];
        else if (arg == "--vectors" && i + 1 < argc) g_vectors = argv[++i];
        else {
            std::printf("usage: test_ppml [--circuits DIR] [--vectors DIR] [--quick]\n");
            return 2;
        }
    }
    try {
        circuit::CircuitStore store(g_circuits);
        test_tables();
        test_planes();
        test_circuits(store);
        test_manifest();
        test_mirror();
        test_truncmask();
        test_beaver();
        test_models(store);
        test_real(store);
        test_tamper(store);
        test_arguments(store);
        test_tiling(store);
    } catch (const std::exception& e) {
        std::printf("FATAL: %s\n", e.what());
        return 1;
    }
    std::printf("test_ppml: %lld checks, %lld passed, %lld failed\n", g_pass + g_fail, g_pass, g_fail);
    return g_fail ? 1 : 0;
}
