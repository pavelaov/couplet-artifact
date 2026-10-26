// couplet/nn.hpp -- models of the Couplet layer (DESIGN_M3.md sections 3 and
// 5): the integer model of a model file, its hashes, bounds, the cleartext
// fixed-point mirror, the first-layer shares and the session plan; the C++
// counterpart of python/couplet/nn.py, reading the files written by
// python/couplet/tools/export_model.py.
//
// Model file "couplet-model/1" (JSON): kind ("mlp" or "cnn"), act (a table
// name), n, f, layers [{rows, cols, W (rows lists of cols integers), b}],
// float_layers (or null), model_hash, architecture_hash, table_hash; the CNN
// convolution is already expanded to its linear map.  Model::load verifies
// model_hash against the layers and reads the activation table: relu, sign,
// recip and rsqrt are computed (tables::exact_table), the others are read
// from table_<act>_n<n>_f<f>.bin in the directory of the model file; the
// table hash must equal the file's table_hash.
//
// Hashes (plain SHA-256, little-endian integers):
//   model_hash        "CPLTMDL1" | kind byte (0 mlp, 1 cnn) | le32 len(act) |
//                     act | le32 n | le32 f | le32 L | per layer le32 rows,
//                     le32 cols, W as int64 row-major, b as int64
//   architecture_hash the same under "CPLTARC1" without W and b
//
// Signed integers of the pipeline are held as u128 in two's complement (the
// "i128 view" of fixedpoint.hpp).  Bounds use saturating arithmetic.
//
// Vertical partition: features [0, d/2) belong to P_0, the rest to P_1.
// mirror(x, exact, thresholds) reproduces the secure pipeline bit for bit:
// p = W a + b 2^f; z = (p >> f) mod 2^n, plus [p mod 2^f > v_l] in mode
// round; a = T[z]; logits = the last p.  plan_manifest lists the circuit
// executions of a session in generation order (nn.plan_manifest), including
// the output masks of the logits in private-model mode (correction C2).
//
// Tiled private-model plans (task R-TILE, nn.plan_manifest): a private model
// with more than Tiling::limit weights (default 4,096) has every weight matrix
// cut into tiles of at most Tiling::rows x Tiling::cols (default 16 x 16),
// row-major (layer_tiles); one manifest entry mattriple(h, w) per tile shape in
// the order of its first tile (tile_shapes) with batch = its number of tiles,
// and the input masks of each party in instances of rows * cols masks plus one
// entry for the remainder.  Plans at or below the limit are the untiled ones.
//
// Vectors file "couplet-vectors/1": inputs, exact_logits, round
// [{thresholds, logits}], float_logits (or null), and the three hashes.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "circuit.hpp"
#include "circuits_ppml.hpp"
#include "fixedpoint.hpp"
#include "hashes.hpp"
#include "sha256.hpp"
#include "tables.hpp"
#include "u128.hpp"

namespace couplet {
namespace nn {

typedef std::pair<size_t, size_t> Shape;

struct Layer {
    size_t rows = 0, cols = 0;
    std::vector<int64_t> W;  // row-major
    std::vector<int64_t> b;
};

struct FloatLayer {
    std::vector<double> W, b;
};

struct MirrorResult {
    std::vector<u128> logits;                 // i128 views
    std::vector<std::vector<u128>> pre;       // per hidden layer
    std::vector<std::vector<uint64_t>> codes;
    std::vector<std::vector<int64_t>> acts;
    size_t overflow = 0;
};

namespace detail {
inline int64_t as_i64(const json::Value& v) {
    if (!v.is_num() || v.text.empty()) throw std::invalid_argument("json: expected an integer");
    const bool neg = v.text[0] == '-';
    json::Value mag;
    mag.type = json::Value::NUM;
    mag.text = neg ? v.text.substr(1) : v.text;
    const uint64_t m = mag.as_u64();
    if ((!neg && m > (uint64_t)INT64_MAX) || (neg && m > (uint64_t)INT64_MAX + 1)) throw std::invalid_argument("json: integer out of range");
    return neg ? (int64_t)(0 - m) : (int64_t)m;
}
inline std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
inline void le32(bytes& out, uint32_t v) {
    for (int i = 0; i < 4; i++) out.push_back((uint8_t)(v >> (8 * i)));
}
inline void le64(bytes& out, uint64_t v) {
    for (int i = 0; i < 8; i++) out.push_back((uint8_t)(v >> (8 * i)));
}
// Saturating arithmetic: results at or above 2^127 are 2^127 (bounds only).
inline u128 sat_add(u128 a, u128 b) {
    const u128 r = a + b;
    return (u128_hi(a) >> 63) || (u128_hi(b) >> 63) || (u128_hi(r) >> 63) ? (u128(1) << 127) : r;
}
inline u128 sat_mul(u128 a, uint64_t b) {
    const u128 cap = u128(1) << 127;
    if (b == 0 || a == u128(0)) return u128(0);
    if (u128_hi(a) >> 63) return cap;
    // a b = lo(a) b + hi(a) b 2^64 with hi(a) < 2^63
    const u128 prod_lo = u128(u128_lo(a)) * u128(b);
    const u128 prod_hi = u128(u128_hi(a)) * u128(b);
    if (u128_hi(prod_hi) != 0 || (u128_lo(prod_hi) >> 63)) return cap;
    const u128 r = prod_lo + (u128(u128_lo(prod_hi)) << 64);
    if ((u128_hi(r) >> 63) || r < prod_lo) return cap;
    return r;
}
inline uint64_t abs64(int64_t v) { return v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; }
}  // namespace detail

class Model {
public:
    std::string kind = "mlp", act = "relu";
    unsigned n = 0, f = 0;
    std::vector<Shape> shapes;
    std::vector<Layer> layers;          // empty for the shape-only copy
    std::vector<FloatLayer> float_layers;
    tables::Table table;
    bool weights_known = false;

    // -- loading
    static Model from_json(const json::Value& doc, const std::string& table_dir) {
        if (doc.at("format").str() != "couplet-model/1") throw std::invalid_argument("unknown model format");
        Model m;
        m.kind = doc.at("kind").str();
        m.act = doc.at("act").str();
        if (m.kind != "mlp" && m.kind != "cnn") throw std::invalid_argument("kind must be mlp or cnn");
        if (!tables::known_name(m.act)) throw std::invalid_argument("unknown activation " + m.act);
        m.n = (unsigned)doc.at("n").as_u64();
        m.f = (unsigned)doc.at("f").as_u64();
        fixedpoint::Format(m.n, m.f);  // validates
        if (m.n > 26) throw std::invalid_argument("the wire format is limited to n <= 26 here");
        for (const json::Value& lay : doc.at("layers").items) {
            Layer L;
            L.rows = (size_t)lay.at("rows").as_u64();
            L.cols = (size_t)lay.at("cols").as_u64();
            const json::Value& W = lay.at("W");
            if (W.items.size() != L.rows) throw std::invalid_argument("model: W has the wrong number of rows");
            for (const json::Value& row : W.items) {
                if (row.items.size() != L.cols) throw std::invalid_argument("model: W has the wrong number of columns");
                for (const json::Value& x : row.items) L.W.push_back(detail::as_i64(x));
            }
            for (const json::Value& x : lay.at("b").items) L.b.push_back(detail::as_i64(x));
            if (L.b.size() != L.rows) throw std::invalid_argument("model: b has the wrong length");
            m.shapes.push_back(Shape(L.rows, L.cols));
            m.layers.push_back(std::move(L));
        }
        if (m.layers.size() < 2) throw std::invalid_argument("a model has at least two layers");
        for (size_t l = 1; l < m.shapes.size(); l++)
            if (m.shapes[l].second != m.shapes[l - 1].first) throw std::invalid_argument("consecutive layers must chain");
        m.weights_known = true;
        const json::Value* fl = doc.find("float_layers");
        if (fl && fl->is_arr()) {
            for (size_t l = 0; l < fl->items.size(); l++) {
                FloatLayer F;
                for (const json::Value& row : fl->items[l].at("W").items)
                    for (const json::Value& x : row.items) F.W.push_back(x.as_double());
                for (const json::Value& x : fl->items[l].at("b").items) F.b.push_back(x.as_double());
                if (l >= m.shapes.size() || F.W.size() != m.shapes[l].first * m.shapes[l].second || F.b.size() != m.shapes[l].first)
                    throw std::invalid_argument("float layers must have the shapes of the integer layers");
                m.float_layers.push_back(std::move(F));
            }
        }
        const json::Value* mh = doc.find("model_hash");
        if (mh && ppml::hex(m.model_hash()) != mh->str()) throw std::invalid_argument("the model hash does not match the layers");
        m.load_table(table_dir);
        const json::Value* th = doc.find("table_hash");
        if (th && ppml::hex(m.table_hash()) != th->str()) throw std::invalid_argument("the table does not match the model's table hash");
        return m;
    }
    static Model load(const std::string& path) {
        return from_json(json::parse(detail::read_file(path)), circuit::dirname(path));
    }

    // The activation table: computed for the exact names, else read from the table file.
    void load_table(const std::string& dir) {
        if (tables::exact_name(act)) {
            table = tables::exact_table(act, n, f);
            return;
        }
        table = tables::read_table(dir + "/" + tables::table_file_name(act, n, f));
        if (table.name != act || table.n != n || table.f != f) throw std::invalid_argument("the table file is for another table");
    }

    // The copy without weights that P_1 holds in private-model mode.
    Model shape_only() const {
        Model m;
        m.kind = kind;
        m.act = act;
        m.n = n;
        m.f = f;
        m.shapes = shapes;
        m.table = table;
        m.weights_known = false;
        return m;
    }

    // -- public shape
    size_t d() const { return shapes.at(0).second; }
    size_t n_out() const { return shapes.back().first; }
    size_t n_layers() const { return shapes.size(); }
    std::vector<size_t> hidden_widths() const {
        std::vector<size_t> w;
        for (size_t l = 0; l + 1 < shapes.size(); l++) w.push_back(shapes[l].first);
        return w;
    }
    size_t n_sites() const {
        size_t t = 0;
        for (size_t w : hidden_widths()) t += w;
        return t;
    }
    size_t n_weights() const {
        size_t t = 0;
        for (const Shape& s : shapes) t += s.first * s.second;
        return t;
    }
    size_t partition() const { return d() / 2; }
    // Features [lo, hi) of P_role.
    std::pair<size_t, size_t> feature_range(int role) const {
        return role == 0 ? std::make_pair((size_t)0, partition()) : std::make_pair(partition(), d());
    }

    // -- hashes
    hashes::hash32 architecture_hash() const { return hash(false); }
    hashes::hash32 model_hash() const {
        if (!weights_known) throw std::invalid_argument("a shape-only model has no model hash");
        return hash(true);
    }
    hashes::hash32 table_hash() const { return tables::table_hash(table); }

    // -- bounds
    std::vector<u128> pre_activation_bounds() const {
        if (!weights_known) throw std::invalid_argument("bounds need the weights");
        uint64_t tmax = 0;
        for (int64_t t : table.entries) tmax = std::max(tmax, detail::abs64(t));
        uint64_t amax = (uint64_t)1 << f;
        std::vector<u128> out;
        for (size_t l = 0; l < layers.size(); l++) {
            if (l > 0) amax = tmax;
            const Layer& L = layers[l];
            u128 best = u128(0);
            for (size_t i = 0; i < L.rows; i++) {
                u128 sum = u128(0);
                for (size_t j = 0; j < L.cols; j++) sum = detail::sat_add(sum, detail::sat_mul(u128(detail::abs64(L.W[i * L.cols + j])), amax));
                sum = detail::sat_add(sum, detail::sat_mul(u128(detail::abs64(L.b[i])), (uint64_t)1 << f));
                if (best < sum) best = sum;
            }
            out.push_back(best);
        }
        return out;
    }
    // Every pre-activation and every table entry fits a sign-exact value of w bits.
    bool sign_exact_in(unsigned w) const {
        const u128 half = u128(1) << (w - 1);
        for (int64_t t : table.entries)
            if (!(u128(detail::abs64(t)) < half)) return false;
        if (!weights_known) return true;
        for (const u128& b : pre_activation_bounds())
            if (!(b < half)) return false;
        return true;
    }
    // Every logit fits a signed payload of k bits.
    bool release_fits(unsigned k) const { return pre_activation_bounds().back() < (u128(1) << (k - 1)); }

    // -- cleartext computations
    // P_role's additive share of the first-layer pre-activation (i128 views).
    std::vector<u128> first_layer_share(int role, const std::vector<int64_t>& features) const {
        if (!weights_known) throw std::invalid_argument("the public-model pipeline needs the weights");
        const std::pair<size_t, size_t> r = feature_range(role);
        if (features.size() != r.second - r.first) throw std::invalid_argument("wrong number of features for this party");
        const Layer& L = layers[0];
        std::vector<u128> p(L.rows, u128(0));
        for (size_t i = 0; i < L.rows; i++) {
            u128 acc = u128(0);
            for (size_t j = r.first; j < r.second; j++)
                acc = acc + fixedpoint::from_i64(L.W[i * L.cols + j]) * fixedpoint::from_i64(features[j - r.first]);
            if (role == 0) acc = acc + (fixedpoint::from_i64(L.b[i]) << f);
            p[i] = acc;
        }
        return p;
    }

    // The cleartext mirror; thresholds[l][j] are needed in mode round only.
    MirrorResult mirror(const std::vector<int64_t>& x, bool exact,
                        const std::vector<std::vector<uint64_t>>* thresholds = nullptr) const {
        if (!weights_known) throw std::invalid_argument("the mirror needs the weights");
        if (x.size() != d()) throw std::invalid_argument("wrong number of features");
        std::vector<u128> a;
        for (int64_t v : x) a.push_back(fixedpoint::from_i64(v));
        MirrorResult res;
        const size_t L = layers.size();
        for (size_t l = 0; l < L; l++) {
            const Layer& Ly = layers[l];
            std::vector<u128> p(Ly.rows);
            for (size_t i = 0; i < Ly.rows; i++) {
                u128 acc = fixedpoint::from_i64(Ly.b[i]) << f;
                for (size_t j = 0; j < Ly.cols; j++) acc = acc + fixedpoint::from_i64(Ly.W[i * Ly.cols + j]) * a[j];
                p[i] = acc;
            }
            if (l == L - 1) {
                res.logits = p;
                break;
            }
            std::vector<uint64_t> codes;
            std::vector<int64_t> acts;
            for (size_t j = 0; j < p.size(); j++) {
                if (!fixedpoint::fits_signed(fixedpoint::asr(p[j], f), n)) res.overflow++;
                uint64_t v_l = 0;
                if (!exact && f > 0) {
                    if (!thresholds || thresholds->size() <= l || (*thresholds)[l].size() <= j)
                        throw std::invalid_argument("mode round needs the recorded thresholds");
                    v_l = (*thresholds)[l][j];
                }
                const uint64_t z = fixedpoint::truncate(p[j], f, n) + ((!exact && f > 0 && u128_lo(p[j] & u128_mask(f)) > v_l) ? 1 : 0);
                codes.push_back(z & (((uint64_t)1 << n) - 1));
                acts.push_back(table.entries[codes.back()]);
            }
            res.pre.push_back(p);
            res.codes.push_back(codes);
            res.acts.push_back(acts);
            a.clear();
            for (int64_t v : acts) a.push_back(fixedpoint::from_i64(v));
        }
        return res;
    }

    // Float logits of the real model (float_layers needed); x: d reals.
    std::vector<double> forward_float(const std::vector<double>& x) const {
        if (float_layers.empty()) throw std::invalid_argument("the model carries no float weights");
        std::vector<double> h = x;
        for (size_t l = 0; l < float_layers.size(); l++) {
            const size_t rows = shapes[l].first, cols = shapes[l].second;
            std::vector<double> o(rows);
            for (size_t i = 0; i < rows; i++) {
                double acc = float_layers[l].b[i];
                for (size_t j = 0; j < cols; j++) acc += float_layers[l].W[i * cols + j] * h[j];
                o[i] = acc;
            }
            if (l + 1 < float_layers.size())
                for (double& v : o) v = act_float(v);
            h = o;
        }
        return h;
    }

private:
    double act_float(double v) const {
        if (act == "relu") return v > 0 ? v : 0.0;
        if (act == "tanh") return std::tanh(v);
        if (act == "sigmoid") return 0.5 * (1.0 + std::tanh(0.5 * v));
        if (act == "gelu") return v * 0.5 * (1.0 + std::erf(v / std::sqrt(2.0)));
        if (act == "sign") return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0);
        if (act == "recip") return v > 0 ? 1.0 / v : 0.0;
        if (act == "rsqrt") return v > 0 ? 1.0 / std::sqrt(v) : 0.0;
        throw std::invalid_argument("unknown activation");
    }
    hashes::hash32 hash(bool with_weights) const {
        bytes buf;
        const char* magic = with_weights ? "CPLTMDL1" : "CPLTARC1";
        buf.insert(buf.end(), magic, magic + 8);
        buf.push_back(kind == "mlp" ? 0 : 1);
        detail::le32(buf, (uint32_t)act.size());
        buf.insert(buf.end(), act.begin(), act.end());
        detail::le32(buf, n);
        detail::le32(buf, f);
        detail::le32(buf, (uint32_t)shapes.size());
        for (size_t l = 0; l < shapes.size(); l++) {
            detail::le32(buf, (uint32_t)shapes[l].first);
            detail::le32(buf, (uint32_t)shapes[l].second);
            if (with_weights) {
                for (int64_t v : layers[l].W) detail::le64(buf, (uint64_t)v);
                for (int64_t v : layers[l].b) detail::le64(buf, (uint64_t)v);
            }
        }
        Sha256 h;
        h.update(buf.data(), buf.size());
        return h.finish();
    }
};

// ------------------------------------------------------------------ tiled private-model plans (task R-TILE)
// rows x cols: the largest tile and the input masks per instance of an
// input-mask entry; limit: a private model with more than `limit` weights
// gets a tiled plan (0: every private model).
struct Tiling {
    size_t rows = 16, cols = 16, limit = 4096;
    Tiling() {}
    Tiling(size_t r, size_t c, size_t lim) : rows(r), cols(c), limit(lim) { validate(); }
    void validate() const {
        if (rows < 1 || cols < 1) throw std::invalid_argument("tiling: rows and cols must be positive");
    }
    size_t masks_per_instance() const { return rows * cols; }
    bool applies(const Model& m) const { return m.n_weights() > limit; }
    bool is_default() const { return rows == 16 && cols == 16 && limit == 4096; }
    friend bool operator==(const Tiling& a, const Tiling& b) { return a.rows == b.rows && a.cols == b.cols && a.limit == b.limit; }
};

// The tile covering rows [r0, r0 + rows) and columns [c0, c0 + cols).
struct Tile {
    size_t r0 = 0, c0 = 0, rows = 0, cols = 0;
};

// The tiles of a rows x cols matrix, row-major over the tile grid.
inline std::vector<Tile> layer_tiles(size_t rows, size_t cols, const Tiling& t) {
    t.validate();
    std::vector<Tile> out;
    for (size_t r0 = 0; r0 < rows; r0 += t.rows)
        for (size_t c0 = 0; c0 < cols; c0 += t.cols) {
            Tile x;
            x.r0 = r0;
            x.c0 = c0;
            x.rows = std::min(t.rows, rows - r0);
            x.cols = std::min(t.cols, cols - c0);
            out.push_back(x);
        }
    return out;
}

// The tile shapes in the order of their first tile, with their counts (one manifest entry each).
inline std::vector<std::pair<Shape, size_t>> tile_shapes(size_t rows, size_t cols, const Tiling& t) {
    std::vector<std::pair<Shape, size_t>> out;
    for (const Tile& x : layer_tiles(rows, cols, t)) {
        const Shape sh(x.rows, x.cols);
        size_t i = 0;
        while (i < out.size() && out[i].first != sh) i++;
        if (i == out.size()) out.push_back(std::make_pair(sh, (size_t)0));
        out[i].second++;
    }
    return out;
}

inline bool plan_tiled(const Model& model, bool private_model, const Tiling& t = Tiling()) {
    return private_model && t.applies(model);
}

namespace detail {
// Input masks of a tiled plan: instances of `per` masks (batch) and one entry for the rest.
inline void add_inmasks(ppml::Manifest& man, int owner, size_t count, const std::string& use, unsigned k, unsigned s, size_t per) {
    const size_t q = count / per, r = count % per;
    if (q) man.add(use + ", " + std::to_string(per) + " per instance", ppml::Descriptor::inmask(owner, per, k, s), q);
    if (r) man.add(use + ", the last " + std::to_string(r), ppml::Descriptor::inmask(owner, r, k, s));
}
}  // namespace detail

// The circuit executions of one inference session, in generation order (nn.plan_manifest).
inline ppml::Manifest plan_manifest(const Model& model, const std::string& mode, unsigned k, unsigned s, bool private_model,
                                    const Tiling& tiling = Tiling()) {
    tiling.validate();
    if (mode != "round" && mode != "exact") throw std::invalid_argument("mode must be round or exact");
    ppml::Manifest man;
    man.k = k;
    man.s = s;
    man.mode = mode;
    man.private_model = private_model;
    man.model_hash = private_model ? model.architecture_hash() : model.model_hash();
    ppml::TableRef t;
    t.name = model.act;
    t.n = model.n;
    t.f = model.f;
    t.hash = model.table_hash();
    man.tables.push_back(t);
    if (private_model) {
        const size_t d0 = model.partition(), d1 = model.d() - d0;
        size_t count0 = d0;
        for (const Shape& sh : model.shapes) count0 += sh.first * sh.second + sh.first;
        if (!tiling.applies(model)) {
            man.add("inputs of P_0: features, weights, biases", ppml::Descriptor::inmask(0, count0, k, s));
            if (d1) man.add("inputs of P_1: features", ppml::Descriptor::inmask(1, d1, k, s));
            for (size_t l = 0; l < model.shapes.size(); l++)
                man.add("matrix triple of layer " + std::to_string(l + 1),
                        ppml::Descriptor::mattriple(model.shapes[l].first, model.shapes[l].second, k, s));
        } else {
            const size_t per = tiling.masks_per_instance();
            detail::add_inmasks(man, 0, count0, "inputs of P_0: features, weights, biases", k, s, per);
            if (d1) detail::add_inmasks(man, 1, d1, "inputs of P_1: features", k, s, per);
            for (size_t l = 0; l < model.shapes.size(); l++) {
                const size_t rows = model.shapes[l].first, cols = model.shapes[l].second;
                const std::string use = "matrix triple tiles of layer " + std::to_string(l + 1) + ": " + std::to_string(rows) + "x" +
                                        std::to_string(cols) + " in tiles of " + std::to_string(tiling.rows) + "x" +
                                        std::to_string(tiling.cols);
                for (const std::pair<Shape, size_t>& e : tile_shapes(rows, cols, tiling))
                    man.add(use, ppml::Descriptor::mattriple(e.first.first, e.first.second, k, s), e.second);
            }
        }
    }
    const std::vector<size_t> widths = model.hidden_widths();
    for (size_t l = 0; l < widths.size(); l++)
        man.add("sites of layer " + std::to_string(l + 1), ppml::Descriptor::site(model.n, model.f, k, s, mode), widths[l]);
    if (private_model) man.add("output masks of the logits", ppml::Descriptor::outmask(k, s), model.n_out());
    return man;
}

// ------------------------------------------------------------------ vectors
struct RoundVector {
    std::vector<std::vector<uint64_t>> thresholds;
    std::vector<int64_t> logits;
};

struct Vectors {
    std::string model_hash, architecture_hash, table_hash;
    unsigned n = 0, f = 0;
    size_t partition = 0;
    std::vector<std::vector<int64_t>> inputs;
    std::vector<std::vector<int64_t>> exact_logits;
    std::vector<RoundVector> round;
    std::vector<std::vector<double>> float_logits;  // empty if null

    static Vectors load(const std::string& path) {
        const json::Value doc = json::parse(detail::read_file(path));
        if (doc.at("format").str() != "couplet-vectors/1") throw std::invalid_argument("unknown vectors format");
        Vectors v;
        v.model_hash = doc.at("model_hash").str();
        v.architecture_hash = doc.at("architecture_hash").str();
        v.table_hash = doc.at("table_hash").str();
        v.n = (unsigned)doc.at("n").as_u64();
        v.f = (unsigned)doc.at("f").as_u64();
        v.partition = (size_t)doc.at("partition").as_u64();
        for (const json::Value& row : doc.at("inputs").items) {
            std::vector<int64_t> r;
            for (const json::Value& x : row.items) r.push_back(detail::as_i64(x));
            v.inputs.push_back(r);
        }
        for (const json::Value& row : doc.at("exact_logits").items) {
            std::vector<int64_t> r;
            for (const json::Value& x : row.items) r.push_back(detail::as_i64(x));
            v.exact_logits.push_back(r);
        }
        for (const json::Value& e : doc.at("round").items) {
            RoundVector rv;
            for (const json::Value& layer : e.at("thresholds").items) {
                std::vector<uint64_t> t;
                for (const json::Value& x : layer.items) t.push_back(x.as_u64());
                rv.thresholds.push_back(t);
            }
            for (const json::Value& x : e.at("logits").items) rv.logits.push_back(detail::as_i64(x));
            v.round.push_back(rv);
        }
        const json::Value* fl = doc.find("float_logits");
        if (fl && fl->is_arr())
            for (const json::Value& row : fl->items) {
                std::vector<double> r;
                for (const json::Value& x : row.items) r.push_back(x.as_double());
                v.float_logits.push_back(r);
            }
        if (v.exact_logits.size() != v.inputs.size() || v.round.size() != v.inputs.size())
            throw std::invalid_argument("vectors: one expectation per input is required");
        return v;
    }
};

// Signed logits (i128 views that fit 64 bits) as int64.
inline std::vector<int64_t> to_i64(const std::vector<u128>& v) {
    std::vector<int64_t> out;
    for (const u128& x : v) {
        if (!fixedpoint::fits_signed(x, 64)) throw std::invalid_argument("a logit does not fit 64 bits");
        out.push_back((int64_t)u128_lo(x));
    }
    return out;
}

}  // namespace nn
}  // namespace couplet
