// couplet/infer.hpp -- party-local inference driver of the Couplet layer
// (DESIGN_M3.md sections 2 and 3, F_infer); the C++ counterpart of
// python/couplet/infer.py, message for message.
//
// InferParty(role, session, model, config) runs ONE party on an existing
// Session (real preprocessing, an ideal dealer or another prep::Prep; the
// session is switched to strict mode).  Three steps:
//
//   setup()       the manifest of the session (nn::plan_manifest) is hashed
//                 and the digests are exchanged BEFORE any circuit runs (phase
//                 "manifest"; a mismatch aborts); then the session is set up
//                 with the first triple reservation (the whole AND budget in
//                 ONE request, or min(budget, triple_cap)); the arithmetic
//                 checker (tied to the session with bind_abort) and the
//                 Opener are created.
//   preprocess()  every manifest entry in order, one batched execution per
//                 entry (engine phases "gen-*").  With triple_cap > 0 an entry
//                 is generated in chunks of max(1, min(batch, cap / n_and))
//                 instances and further requests are made as the pool runs
//                 short, exactly as the reference (record: executions,
//                 triple_requests).
//   infer(x)      the online pipeline on this party's quantised features.
//                 Public model: first layer by commit-then-open of the
//                 additive shares, later layers public linear maps on the
//                 authenticated activations; truncate-and-mask per hidden
//                 layer (one opening in mode round, two in mode exact); plane A
//                 of every site evaluated on the full domain and read with
//                 the activation table; release of the sign-exact logits by
//                 check, open, check.  Private model: authenticated input of
//                 both parties' features and P_0's weights and biases (one
//                 exchange), matrix triples for every layer, release with the
//                 output masks of the manifest (check, add mask, open, check;
//                 correction C2).  Returns to_signed(v mod 2^k, k) per logit
//                 (2f fraction bits).
//
// Tiled private-model plans (task R-TILE; config.tiling, default 16 x 16 tiles
// above 4,096 weights): preprocess() concatenates the instances of each party's
// input masks in generation order (beaver::merge_inmasks; the input exchange is
// unchanged) and gives the triples of each tile-shape entry to the tiles of
// that shape in row-major order, each triple to exactly one tile (a count that
// does not match the plan aborts); infer() evaluates every linear layer with
// beaver::matvec_tiled (one opening exchange per layer).  The record gives
// tiled, tiles and tiles_per_layer.
//
// Threads: config.threads is used for the full-domain evaluation of the
// planes of a layer (and their prefix tables where a query needs them) and
// for the table reads; results do not depend on it.  Planes are evaluated in
// blocks of sites and released after their table reads, so the memory is
// bounded by about 64 * 2^n bytes per site of a block.
//
// Accounting (the session's PhaseBook): "manifest"; "setup-*" and "gen-*"
// (engine); "online-local" (local linear algebra and table reads);
// "plane-eval" (full-domain evaluation and prefix tables); "tm-open",
// "input-open", "arith-open" (openings) and "arith-check" (checks).  The real
// preprocessing books its own phases (prep::RealPrep::stats()), so the two
// sum to the channel totals.
//
// Abort: any ProtocolAbort, and any failure after the first message of a
// step, retires the session, the checker and the material; every later call
// throws ProtocolAbort.  Argument errors are std::invalid_argument before any
// message.  Test hooks (virtual): make_checker, make_opener, layer_sites.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "abb.hpp"
#include "beaver.hpp"
#include "channel.hpp"
#include "circuits_ppml.hpp"
#include "fixedpoint.hpp"
#include "nn.hpp"
#include "planes.hpp"
#include "rng.hpp"
#include "session.hpp"
#include "spdz2k.hpp"
#include "truncmask.hpp"
#include "u128.hpp"

namespace couplet {
namespace infer {

using spdz2k::AShare;

const char* const PHASE_ONLINE = "online-local";

struct Config {
    std::string mode = "round";
    bool private_model = false;
    size_t triple_cap = 0;        // generation triples per request; 0 = the whole budget in one request
    unsigned threads = 1;         // plane evaluation and table reads
    RandomSource* rng = nullptr;  // coins of the generation (tests); nullptr = operating system
    nn::Tiling tiling;            // tiled private-model plans (task R-TILE)
};

struct Execution {
    std::string name;
    size_t batch = 0, n_and = 0;
    double seconds = 0.0;
};

// The public outcome and counters of one party.
struct Record {
    std::string manifest_digest;
    size_t and_gates = 0, sites = 0, executions = 0;
    uint64_t triple_requests = 0, triples_unused = 0;
    std::vector<Execution> execution_list;
    std::vector<std::vector<uint64_t>> thresholds, x_hat;
    std::vector<std::vector<u128>> opened;
    std::vector<int64_t> logits;
    std::vector<u128> released;  // the opened representatives of the release
    bool tiled = false;          // the plan is tiled (task R-TILE)
    size_t tiles = 0;            // tile triples of the plan (0 if untiled)
    std::vector<size_t> tiles_per_layer;
    uint64_t openings = 0, checks = 0, shared_openings = 0, openings_before_release = 0;
    double setup_s = 0, preprocess_s = 0, infer_s = 0, plane_eval_s = 0, table_s = 0;
};

class InferParty {
public:
    enum Stage { NEW, SETUP, READY, DONE };

    InferParty(int role, Session& session, const nn::Model& model, const Config& cfg)
        : role_(role), ses_(&session), model_(&model), cfg_(cfg), k_(session.k()), s_(session.s()), w_(session.k() + session.s()) {
        if ((role != 0 && role != 1) || session.role() != role) throw std::invalid_argument("role must be 0 or 1 and match the session");
        if (cfg.mode != "round" && cfg.mode != "exact") throw std::invalid_argument("mode must be round or exact");
        if (cfg.private_model && role == 0 && !model.weights_known) throw std::invalid_argument("P_0 holds the weights in private-model mode");
        if (!cfg.private_model && !model.weights_known) throw std::invalid_argument("the public-model pipeline needs the weights");
        if (!model.sign_exact_in(w_))
            throw std::invalid_argument("a pre-activation or a table entry of the model does not fit a sign-exact value of w = k + s bits");
        if (model.weights_known && !model.release_fits(k_))
            throw std::invalid_argument("a logit of the model does not fit a signed payload of k bits");
        if (session.is_setup()) throw std::invalid_argument("the session must not be set up yet");
        manifest_ = nn::plan_manifest(model, cfg.mode, k_, s_, cfg.private_model, cfg.tiling);
        tiled_ = nn::plan_tiled(model, cfg.private_model, cfg.tiling);
        if (tiled_) {
            for (size_t l = 0; l < model.shapes.size(); l++) {
                layout_.push_back(nn::layer_tiles(model.shapes[l].first, model.shapes[l].second, cfg.tiling));
                shapes_.push_back(nn::tile_shapes(model.shapes[l].first, model.shapes[l].second, cfg.tiling));
                rec_.tiles_per_layer.push_back(layout_.back().size());
                rec_.tiles += layout_.back().size();
            }
        }
        rec_.tiled = tiled_;
        budget_ = manifest_.and_budget(session.store());
        session.set_allow_refill(false);  // strict mode: a short pool is an error, not a second population
        table_ = planes::EmbeddedTable(model.table.entries, w_);
    }
    virtual ~InferParty() {}
    InferParty(const InferParty&) = delete;
    InferParty& operator=(const InferParty&) = delete;

    int role() const { return role_; }
    Stage stage() const { return stage_; }
    const ppml::Manifest& manifest() const { return manifest_; }
    size_t and_budget() const { return budget_; }
    bool tiled() const { return tiled_; }
    const Record& record() const { return rec_; }
    Session& session() { return *ses_; }
    spdz2k::ArithChecker* checker() { return checker_.get(); }
    truncmask::Opener* opener() { return opener_.get(); }
    bool active() const { return ses_->active() && !(checker_ && checker_->dead()); }

    // Retire the session, the checker and all material.
    void abort() {
        if (checker_) checker_->abort();
        ses_->abort();
        sites_.clear();
        inmask_[0].reset();
        inmask_[1].reset();
        inmask_parts_[0].clear();
        inmask_parts_[1].clear();
        mattriples_.clear();
        staged_.clear();
        tile_dest_ = nullptr;
        tile_triples_.clear();
        outmasks_.clear();
    }

    // -- 1. setup
    void setup() {
        require_active();
        if (stage_ != NEW) throw std::invalid_argument("setup() was already run");
        const clock::time_point t0 = clock::now();
        step([&] {
            const hashes::hash32 d = ppml::compare_manifest(ses_->channel(), manifest_, &ses_->book());
            rec_.manifest_digest = ppml::hex(d);
            const size_t first = cfg_.triple_cap ? std::min(budget_, cfg_.triple_cap) : budget_;
            ses_->setup(first);
            checker_ = make_checker(role_, ses_->channel(), ses_->sid(), ses_->delta_share(), s_, &ses_->book());
            ses_->bind_abort(*checker_);
            opener_ = make_opener(role_, ses_->channel(), ses_->sid(), *checker_, ses_->delta_share(), &ses_->book());
        });
        stage_ = SETUP;
        rec_.setup_s = secs(t0);
    }

    // -- 2. preprocessing
    void preprocess() {
        require_active();
        if (stage_ != SETUP) throw std::invalid_argument("preprocess() needs setup() first (and runs once)");
        const clock::time_point t0 = clock::now();
        long long remaining = (long long)budget_;
        abb::Engine& engine = ses_->engine();
        step([&] {
            engine.set_phase_tag("gen-");
            struct Untag {
                abb::Engine& e;
                ~Untag() { e.set_phase_tag(""); }
            } untag{engine};
            size_t next_tile_entry = 0;  // tiled plan: the next (layer, shape) of the tile entries
            if (tiled_) {
                staged_.assign(shapes_.size(), std::vector<std::vector<ppml::MatTripleMaterial>>());
                for (size_t l = 0; l < shapes_.size(); l++) staged_[l].resize(shapes_[l].size());
            }
            for (const ppml::ManifestEntry& e : manifest_.entries) {
                const size_t per = ppml::and_count(ses_->store(), e.desc);
                if (tiled_ && e.desc.kind == ppml::KIND_MATTRIPLE) {
                    // the tile entries come layer by layer, shapes in the order of nn::tile_shapes
                    size_t idx = next_tile_entry++, l = 0;
                    while (l < shapes_.size() && idx >= shapes_[l].size()) idx -= shapes_[l++].size();
                    if (l == shapes_.size()) throw ProtocolAbort("the manifest has more matrix triples than the tiles");
                    const std::pair<nn::Shape, size_t>& want = shapes_[l][idx];
                    if (e.desc.rows != want.first.first || e.desc.cols != want.first.second || e.batch != want.second)
                        throw ProtocolAbort("the matrix triples do not match the tiles of the plan");
                    tile_dest_ = &staged_[l][idx];
                }
                generate_entry(e, per, remaining);
                tile_dest_ = nullptr;
                remaining -= (long long)(per * e.batch);
            }
            if (tiled_) assign_tiles(next_tile_entry);
        });
        if (sites_.size() != model_->hidden_widths().size()) {
            abort();
            throw ProtocolAbort("the manifest did not provide the sites of every hidden layer");
        }
        if (cfg_.private_model && outmasks_.size() != model_->n_out()) {
            abort();
            throw ProtocolAbort("the manifest did not provide one output mask per logit");
        }
        rec_.and_gates = budget_;
        rec_.sites = manifest_.site_count();
        rec_.triple_requests = ses_->triple_requests();
        rec_.triples_unused = ses_->triples_unused();
        stage_ = READY;
        rec_.preprocess_s = secs(t0);
    }

    // -- 3. the online pipeline
    std::vector<int64_t> infer(const std::vector<int64_t>& features) {
        require_active();
        if (stage_ != READY) throw std::invalid_argument("infer() needs preprocess() first (and runs once per session)");
        const std::pair<size_t, size_t> sl = model_->feature_range(role_);
        if (features.size() != sl.second - sl.first) throw std::invalid_argument("wrong number of features for this party");
        const clock::time_point t0 = clock::now();
        std::vector<int64_t> logits;
        abb::PhaseBook& book = ses_->book();
        step([&] {
            abb::PhaseRestore restore(book);
            book.mark(PHASE_ONLINE);
            const nn::Model& model = *model_;
            const unsigned f = model.f;
            std::vector<AShare> a;
            std::vector<std::vector<AShare>> W_sh, b_sh;
            if (cfg_.private_model) private_inputs(features, a, W_sh, b_sh);
            const size_t L = model.n_layers();
            for (size_t l = 0; l < L; l++) {
                std::vector<AShare> p;
                std::vector<u128> p_shared;
                const bool first_shared = l == 0 && !cfg_.private_model;
                if (cfg_.private_model) {
                    if (tiled_)
                        p = beaver::matvec_tiled(*opener_, W_sh[l], model.shapes[l].first, model.shapes[l].second, a, tile_triples_.at(l));
                    else
                        p = beaver::matvec(*opener_, W_sh[l], a, beaver::mattriple_shares(mattriples_[l]));
                    for (size_t i = 0; i < p.size(); i++) p[i] = p[i] + b_sh[l][i].scale(u128(1) << f);
                } else if (first_shared) {
                    p_shared = model.first_layer_share(role_, features);
                } else {
                    p = linear_public(model.layers[l], a);
                }
                if (l == L - 1) {
                    rec_.openings_before_release = checker_->openings();
                    logits = cfg_.private_model ? opener_->open_masked(p, outmasks_) : opener_->open_sign_exact(p);
                    break;
                }
                std::vector<truncmask::Site*> sites = layer_sites(l);
                const std::vector<truncmask::TruncResult> res =
                    first_shared ? truncmask::truncate_and_mask_shared(*opener_, sites, p_shared, cfg_.threads)
                                 : truncmask::truncate_and_mask(*opener_, sites, p, cfg_.threads);
                std::vector<uint64_t> thr, xh;
                std::vector<u128> op;
                for (const truncmask::TruncResult& r : res) {
                    thr.push_back(r.v_l);
                    xh.push_back(r.x_hat);
                    op.push_back(r.v);
                }
                rec_.thresholds.push_back(thr);
                rec_.x_hat.push_back(xh);
                rec_.opened.push_back(op);
                a = read_tables(sites, xh);
            }
        });
        rec_.logits = logits;
        rec_.released = checker_->last_opened();
        rec_.openings = checker_->openings();
        rec_.checks = checker_->checks();
        rec_.shared_openings = opener_->shared_openings();
        rec_.plane_eval_s = book.get(truncmask::PHASE_PLANES).seconds;
        rec_.infer_s = secs(t0);
        stage_ = DONE;
        return logits;
    }

    std::vector<int64_t> run(const std::vector<int64_t>& features) {
        setup();
        preprocess();
        return infer(features);
    }

protected:
    typedef std::chrono::steady_clock clock;

    // -- test hooks
    virtual std::unique_ptr<spdz2k::ArithChecker> make_checker(int role, Channel& ch, const bytes& sid, uint64_t delta,
                                                               unsigned s, abb::PhaseBook* book) {
        return std::unique_ptr<spdz2k::ArithChecker>(new spdz2k::ArithChecker(role, ch, sid, delta, s, book));
    }
    virtual std::unique_ptr<truncmask::Opener> make_opener(int role, Channel& ch, const bytes& sid, spdz2k::ArithChecker& checker,
                                                           uint64_t delta, abb::PhaseBook* book) {
        return std::unique_ptr<truncmask::Opener>(new truncmask::Opener(role, ch, sid, checker, delta, book));
    }
    // The sites of hidden layer `layer` (a deviating party may substitute some).
    virtual std::vector<truncmask::Site*> layer_sites(size_t layer) {
        std::vector<truncmask::Site*> out;
        for (truncmask::Site& s : sites_.at(layer)) out.push_back(&s);
        return out;
    }

    void require_active() {
        if (!active()) {
            abort();
            throw ProtocolAbort("session aborted");
        }
    }

    // A protocol step: an argument error before any message leaves the session
    // usable; every other failure retires it.
    template <class F>
    void step(F&& body) {
        require_active();
        const ChannelStats before = ses_->channel().stats();
        try {
            body();
        } catch (const ProtocolAbort&) {
            abort();
            throw;
        } catch (const std::invalid_argument&) {
            const ChannelStats now = ses_->channel().stats();
            if (now.msgs_sent != before.msgs_sent || now.msgs_recv != before.msgs_recv || !active()) abort();
            throw;
        } catch (...) {
            abort();
            throw;
        }
    }

    static double secs(clock::time_point t0) { return std::chrono::duration<double>(clock::now() - t0).count(); }

    // One manifest entry: one or more executions, reserving triples as the reference does.
    void generate_entry(const ppml::ManifestEntry& e, size_t per, long long remaining) {
        const size_t cap = cfg_.triple_cap;
        const size_t batch = e.batch;
        const size_t chunk = cap ? std::max<size_t>(1, std::min(batch, per ? cap / per : batch)) : batch;
        std::vector<truncmask::Site> layer;
        for (size_t start = 0; start < batch; start += chunk) {
            const size_t count = std::min(chunk, batch - start);
            const long long need = (long long)(per * count);
            const long long avail = (long long)ses_->triples_unused();
            if (avail < need) {
                long long want = need - avail;
                if (cap) want = std::max(want, std::min((long long)cap, remaining) - avail);
                ses_->reserve_triples((size_t)want);
            }
            const clock::time_point t0 = clock::now();
            ppml::Generated g = ppml::generate(*ses_, e.desc, count, cfg_.rng);
            Execution ex;
            ex.name = e.desc.name();
            ex.batch = count;
            ex.n_and = per;
            ex.seconds = secs(t0);
            rec_.execution_list.push_back(ex);
            rec_.executions++;
            remaining -= need;
            for (ppml::SiteMaterial& m : g.sites) layer.push_back(truncmask::Site(std::move(m)));
            for (ppml::InMaskMaterial& m : g.inmasks) {
                if (tiled_) inmask_parts_[m.owner()].push_back(std::move(m));
                else inmask_[m.owner()].reset(new ppml::InMaskMaterial(std::move(m)));
            }
            for (ppml::MatTripleMaterial& m : g.mattriples) {
                if (tile_dest_) tile_dest_->push_back(std::move(m));
                else mattriples_.push_back(std::move(m));
            }
            for (const ppml::OutMaskMaterial& m : g.outmasks) outmasks_.push_back(AShare(m.share.value, m.share.tag, w_));
        }
        if (e.desc.kind == ppml::KIND_SITE) sites_.push_back(std::move(layer));
    }

    // Tiled plan: the triples of each shape entry go to the tiles of that shape, row-major,
    // each to exactly one tile; then the input-mask instances are merged per party.
    void assign_tiles(size_t tile_entries_seen) {
        size_t planned = 0;
        for (const auto& sh : shapes_) planned += sh.size();
        if (tile_entries_seen != planned) throw ProtocolAbort("the manifest has fewer matrix triples than the tiles");
        tile_triples_.assign(layout_.size(), std::vector<beaver::TileTriple>());
        for (size_t l = 0; l < layout_.size(); l++) {
            std::vector<size_t> cursor(shapes_[l].size(), 0);
            for (const nn::Tile& t : layout_[l]) {
                size_t idx = 0;
                while (idx < shapes_[l].size() && shapes_[l][idx].first != nn::Shape(t.rows, t.cols)) idx++;
                if (idx == shapes_[l].size() || cursor[idx] >= staged_[l][idx].size()) throw ProtocolAbort("a tile has no matrix triple");
                beaver::TileTriple tt;
                tt.r0 = t.r0;
                tt.c0 = t.c0;
                tt.t = beaver::mattriple_shares(staged_[l][idx][cursor[idx]++]);
                tile_triples_[l].push_back(std::move(tt));
            }
            for (size_t idx = 0; idx < cursor.size(); idx++)
                if (cursor[idx] != staged_[l][idx].size()) throw ProtocolAbort("a matrix triple of the plan has no tile");
        }
        staged_.clear();
        for (int owner = 0; owner < 2; owner++)
            if (!inmask_parts_[owner].empty()) {
                inmask_[owner].reset(new ppml::InMaskMaterial(beaver::merge_inmasks(inmask_parts_[owner])));
                inmask_parts_[owner].clear();
            }
    }

    // [[W a + b 2^f]] for public integer W, b (sign-exact by the bound check of the constructor).
    std::vector<AShare> linear_public(const nn::Layer& Ly, const std::vector<AShare>& a) const {
        if (a.size() != Ly.cols) throw std::invalid_argument("the layer does not match its input");
        const u128 mask = u128_mask(w_);
        const uint64_t delta = opener_->delta_share();
        std::vector<AShare> out;
        for (size_t i = 0; i < Ly.rows; i++) {
            u128 v = u128(0), m = u128(0);
            for (size_t j = 0; j < Ly.cols; j++) {
                const u128 c = fixedpoint::from_i64(Ly.W[i * Ly.cols + j]);
                v = v + c * a[j].value;
                m = m + c * a[j].mac;
            }
            const u128 bias = (fixedpoint::from_i64(Ly.b[i]) << model_->f) & mask;
            out.push_back(AShare(v & mask, m & mask, w_).add_public(bias, role_, delta).with_sign_exact(true));
        }
        return out;
    }

    // Plane A of every site (full domain, in blocks) and the table read at x_hat.
    std::vector<AShare> read_tables(std::vector<truncmask::Site*>& sites, const std::vector<uint64_t>& x_hat) {
        abb::PhaseBook& book = ses_->book();
        std::vector<AShare> out(sites.size());
        const unsigned T = std::max(1u, cfg_.threads);
        const size_t block = std::max<size_t>(2 * (size_t)T, 1);
        for (size_t lo = 0; lo < sites.size(); lo += block) {
            const size_t hi = std::min(sites.size(), lo + block);
            std::vector<truncmask::Site*> part(sites.begin() + (ptrdiff_t)lo, sites.begin() + (ptrdiff_t)hi);
            book.mark(truncmask::PHASE_PLANES);
            truncmask::evaluate_planes(role_, part, false, T, false);  // a table read needs no prefix tables
            book.mark(PHASE_ONLINE);
            const clock::time_point t0 = clock::now();
            const unsigned inner = part.size() >= T ? 1u : std::max(1u, T / (unsigned)part.size());
            planes::parallel_tasks(part.size(), std::min<unsigned>(T, (unsigned)part.size()), [&](size_t i) {
                out[lo + i] = part[i]->plane_a.table(table_, x_hat[lo + i], 0, inner);
            });
            rec_.table_s += secs(t0);
            for (truncmask::Site* s : part) s->release_planes();
        }
        return out;
    }

    // Authenticated input of the features (both parties) and of P_0's weights and biases.
    void private_inputs(const std::vector<int64_t>& features, std::vector<AShare>& x, std::vector<std::vector<AShare>>& W_sh,
                        std::vector<std::vector<AShare>>& b_sh) {
        const nn::Model& model = *model_;
        const size_t d0 = model.partition();
        std::vector<u128> mine_vals;
        for (int64_t v : features) mine_vals.push_back(fixedpoint::from_i64(v));
        if (role_ == 0)
            for (const nn::Layer& Ly : model.layers) {
                for (int64_t v : Ly.W) mine_vals.push_back(fixedpoint::from_i64(v));
                for (int64_t v : Ly.b) mine_vals.push_back(fixedpoint::from_i64(v));
            }
        const ppml::InMaskMaterial* mine = inmask_[role_].get();
        const ppml::InMaskMaterial* peer = inmask_[1 - role_].get();
        const std::pair<std::vector<AShare>, std::vector<AShare>> sh =
            beaver::exchange_inputs(*opener_, mine, mine ? mine_vals : std::vector<u128>(), peer);
        const std::vector<AShare>& p0 = role_ == 0 ? sh.first : sh.second;
        const std::vector<AShare>& p1 = role_ == 0 ? sh.second : sh.first;
        if (p0.size() < d0) throw ProtocolAbort("private inputs: too few values of P_0");
        x.assign(p0.begin(), p0.begin() + (ptrdiff_t)d0);
        x.insert(x.end(), p1.begin(), p1.end());
        size_t pos = d0;
        for (const nn::Shape& shp : model.shapes) {
            const size_t rows = shp.first, cols = shp.second;
            if (p0.size() < pos + rows * cols + rows) throw ProtocolAbort("private inputs: too few values of P_0");
            W_sh.push_back(std::vector<AShare>(p0.begin() + (ptrdiff_t)pos, p0.begin() + (ptrdiff_t)(pos + rows * cols)));
            pos += rows * cols;
            b_sh.push_back(std::vector<AShare>(p0.begin() + (ptrdiff_t)pos, p0.begin() + (ptrdiff_t)(pos + rows)));
            pos += rows;
        }
    }

    int role_;
    Session* ses_;
    const nn::Model* model_;  // a deviating party may point it elsewhere
    Config cfg_;
    unsigned k_, s_, w_;
    ppml::Manifest manifest_;
    size_t budget_ = 0;
    planes::EmbeddedTable table_;
    Stage stage_ = NEW;
    Record rec_;
    std::unique_ptr<spdz2k::ArithChecker> checker_;
    std::unique_ptr<truncmask::Opener> opener_;
    std::vector<std::vector<truncmask::Site>> sites_;  // per hidden layer
    std::unique_ptr<ppml::InMaskMaterial> inmask_[2];
    std::vector<ppml::MatTripleMaterial> mattriples_;
    // tiled plans (task R-TILE)
    bool tiled_ = false;
    std::vector<std::vector<nn::Tile>> layout_;                         // per layer, row-major
    std::vector<std::vector<std::pair<nn::Shape, size_t>>> shapes_;     // per layer, nn::tile_shapes
    std::vector<std::vector<std::vector<ppml::MatTripleMaterial>>> staged_;  // per layer and shape
    std::vector<ppml::MatTripleMaterial>* tile_dest_ = nullptr;
    std::vector<ppml::InMaskMaterial> inmask_parts_[2];
    std::vector<std::vector<beaver::TileTriple>> tile_triples_;
    std::vector<AShare> outmasks_;
};

}  // namespace infer
}  // namespace couplet
