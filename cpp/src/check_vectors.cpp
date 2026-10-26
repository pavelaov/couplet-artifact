// check_vectors.cpp -- verify the C++ implementation against the golden
// vectors produced by the Python reference (python/vectors/make_vectors.py):
//
//   ascon_vectors.json  Ascon permutation, hash vectors, IV words and the PRG
//                       maps F_IV(s) = p12(s||IV) XOR (s||IV): raw permutation
//                       output, feed-forward output, G, Conv, ConvV_L, ConvV_R.
//   fss_vectors.json    two-plane DPF / DCF (DESIGN.md section 3).  For every
//                       vector the inputs are regenerated from the recorded
//                       seed with the SHA-256 counter-mode generator of the
//                       reference, the keys are generated and compared byte
//                       for byte with the recorded keys, the recorded keys
//                       are parsed and re-encoded, the SHA-256 digests of the
//                       EvalAll vectors of both parties and both planes are
//                       compared (2^n values, each ceil((k+s)/8) bytes
//                       little-endian), and the sample Eval outputs are
//                       compared.
//
// Usage: check_vectors [ascon_vectors.json [fss_vectors.json]]
// Without arguments the files are looked up relative to the executable
// (<exe dir>/../../python/vectors/) and then in the usual locations relative
// to the working directory (vectors/, ../vectors/, python/vectors/, ...).
// With one argument the FSS file is expected next to the Ascon file.
// Exit code 0 iff every vector matches.
//
// Contains a minimal JSON reader (objects, arrays, strings, integers,
// true/false/null) sufficient for these files.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "couplet/ascon.hpp"
#include "couplet/fss.hpp"
#include "couplet/sha256.hpp"
#include "couplet/u128.hpp"

using namespace couplet;

// ------------------------------------------------------------ minimal JSON
struct Json {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } type = NUL;
    bool b = false;
    long long num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json& operator[](const char* key) const {
        if (type != OBJ) throw std::runtime_error(std::string("JSON: not an object when looking up '") + key + "'");
        for (auto& kv : obj)
            if (kv.first == key) return kv.second;
        throw std::runtime_error(std::string("JSON: missing key '") + key + "'");
    }
    const Json& operator[](size_t i) const {
        if (type != ARR) throw std::runtime_error("JSON: not an array");
        if (i >= arr.size()) throw std::runtime_error("JSON: index out of range");
        return arr[i];
    }
    const Json& operator[](int i) const { return (*this)[(size_t)i]; }  // avoids literal-0 ambiguity
    bool has(const char* key) const {
        if (type != OBJ) return false;
        for (auto& kv : obj)
            if (kv.first == key) return true;
        return false;
    }
    size_t size() const { return type == ARR ? arr.size() : type == OBJ ? obj.size() : 0; }
    const std::string& s() const {
        if (type != STR) throw std::runtime_error("JSON: not a string");
        return str;
    }
    long long i() const {
        if (type != NUM) throw std::runtime_error("JSON: not a number");
        return num;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : t_(text), p_(0) {}
    Json parse() {
        Json v = value();
        ws();
        if (p_ != t_.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string& t_;
    size_t p_;

    [[noreturn]] void fail(const char* msg) {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(p_) + ": " + msg);
    }
    void ws() {
        while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r')) p_++;
    }
    bool match(const char* lit) {
        size_t n = std::strlen(lit);
        if (t_.compare(p_, n, lit) == 0) { p_ += n; return true; }
        return false;
    }
    Json value() {
        ws();
        if (p_ >= t_.size()) fail("unexpected end");
        char c = t_[p_];
        Json v;
        if (c == '{') {
            v.type = Json::OBJ;
            p_++;
            ws();
            if (p_ < t_.size() && t_[p_] == '}') { p_++; return v; }
            for (;;) {
                ws();
                if (p_ >= t_.size() || t_[p_] != '"') fail("expected key");
                std::string k = string();
                ws();
                if (p_ >= t_.size() || t_[p_] != ':') fail("expected ':'");
                p_++;
                Json child = value();
                v.obj.emplace_back(std::move(k), std::move(child));
                ws();
                if (p_ < t_.size() && t_[p_] == ',') { p_++; continue; }
                if (p_ < t_.size() && t_[p_] == '}') { p_++; return v; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.type = Json::ARR;
            p_++;
            ws();
            if (p_ < t_.size() && t_[p_] == ']') { p_++; return v; }
            for (;;) {
                v.arr.push_back(value());
                ws();
                if (p_ < t_.size() && t_[p_] == ',') { p_++; continue; }
                if (p_ < t_.size() && t_[p_] == ']') { p_++; return v; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.type = Json::STR; v.str = string(); return v; }
        if (match("true")) { v.type = Json::BOOL; v.b = true; return v; }
        if (match("false")) { v.type = Json::BOOL; v.b = false; return v; }
        if (match("null")) { v.type = Json::NUL; return v; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            size_t start = p_;
            if (t_[p_] == '-') p_++;
            while (p_ < t_.size() && ((t_[p_] >= '0' && t_[p_] <= '9') || t_[p_] == '.' || t_[p_] == 'e' ||
                                      t_[p_] == 'E' || t_[p_] == '+' || t_[p_] == '-'))
                p_++;
            v.type = Json::NUM;
            v.num = std::strtoll(t_.substr(start, p_ - start).c_str(), nullptr, 10);
            return v;
        }
        fail("unexpected character");
    }
    std::string string() {
        if (t_[p_] != '"') fail("expected string");
        p_++;
        std::string out;
        while (p_ < t_.size()) {
            char c = t_[p_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (p_ >= t_.size()) fail("bad escape");
                char e = t_[p_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (p_ + 4 > t_.size()) fail("bad \\u escape");
                        unsigned cp = (unsigned)std::strtoul(t_.substr(p_, 4).c_str(), nullptr, 16);
                        p_ += 4;
                        if (cp < 0x80) out.push_back((char)cp);
                        else if (cp < 0x800) { out.push_back((char)(0xC0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 0x3F))); }
                        else { out.push_back((char)(0xE0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }
};

// ------------------------------------------------------------ helpers
static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& name) {
    if (!ok || g_fail) std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) g_pass++; else g_fail++;
}

static uint64_t hex_u64(const std::string& s) {  // accepts optional 0x prefix
    size_t p = (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ? 2 : 0;
    return std::strtoull(s.c_str() + p, nullptr, 16);
}
static std::vector<uint8_t> unhex(const std::string& s) {
    std::vector<uint8_t> v;
    for (size_t i = 0; i + 1 < s.size(); i += 2) v.push_back((uint8_t)std::strtoul(s.substr(i, 2).c_str(), nullptr, 16));
    return v;
}
static std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) { s.push_back(d[p[i] >> 4]); s.push_back(d[p[i] & 15]); }
    return s;
}
static void read_state(const Json& arr, uint64_t s[5]) {
    if (arr.size() != 5) throw std::runtime_error("state must have 5 words");
    for (int i = 0; i < 5; i++) s[i] = hex_u64(arr[(size_t)i].s());
}
static bool state_eq(const uint64_t a[5], const uint64_t b[5]) { return std::memcmp(a, b, 40) == 0; }

// Generic Ascon hash with 64-bit rate and 12 rounds everywhere (covers
// Ascon-Hash v1.2: big-endian words, pad 0x80; and SP 800-232 Ascon-Hash256:
// little-endian words, pad 0x01).
static std::string ascon_hash_generic(uint64_t iv, bool big_endian, uint8_t pad, const std::vector<uint8_t>& m) {
    uint64_t s[5] = {iv, 0, 0, 0, 0};
    ascon::p12(s);
    size_t off = 0;
    while (m.size() - off >= 8) {
        s[0] ^= big_endian ? ascon::load64be(&m[off]) : ascon::load64le(&m[off]);
        ascon::p12(s);
        off += 8;
    }
    uint8_t last[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t rem = m.size() - off;
    if (rem) std::memcpy(last, &m[off], rem);
    last[rem] = pad;
    s[0] ^= big_endian ? ascon::load64be(last) : ascon::load64le(last);
    ascon::p12(s);
    uint8_t out[32];
    for (int i = 0; i < 4; i++) {
        if (big_endian) ascon::store64be(out + 8 * i, s[0]);
        else ascon::store64le(out + 8 * i, s[0]);
        if (i < 3) ascon::p12(s);
    }
    return hex(out, 32);
}

// The raw permutation output ("perm_state"), the feed-forward output
// ("state" = Y = perm_state XOR (seed || IV)) and the value / tag words.
static void check_prg_states(const Json& blk, const ascon::seed_t& seed, const ascon::IV& iv, const char* name, size_t idx) {
    uint64_t st[5], exp[5], in[5];
    ascon::load_prg_state(in, seed.data(), iv);
    ascon::prg_state(st, seed.data(), iv);
    read_state(blk["state"], exp);
    check(state_eq(st, exp), std::string(name) + " state (feed-forward), vector " + std::to_string(idx));
    if (blk.has("perm_state")) {
        uint64_t perm[5], exp_perm[5];
        ascon::prg_permute(perm, seed.data(), iv);
        read_state(blk["perm_state"], exp_perm);
        bool ok = state_eq(perm, exp_perm);
        for (int i = 0; i < 5; i++) ok = ok && (exp_perm[i] ^ in[i]) == exp[i];
        check(ok, std::string(name) + " perm_state and state = perm_state XOR input, vector " + std::to_string(idx));
    }
    // The structure-of-arrays 4-way form used by EvalAll, seed in every lane.
    uint64_t lo[4], hi[4], y[5][4];
    for (int j = 0; j < 4; j++) { lo[j] = in[0]; hi[j] = in[1]; }
    ascon::prg_state_x4_soa(lo, hi, iv, y);
    bool ok_soa = true;
    for (int j = 0; j < 4; j++)
        for (int w = 0; w < 5; w++) ok_soa = ok_soa && y[w][j] == exp[w];
    check(ok_soa, std::string(name) + " state via prg_state_x4_soa, vector " + std::to_string(idx));
}

static void check_conv_block(const Json& blk, const ascon::seed_t& seed, const ascon::IV& iv, const char* name, size_t idx) {
    check_prg_states(blk, seed, iv, name, idx);
    struct { const char* key; unsigned ks; } profiles[2] = {{"k32_s64", 96}, {"k64_s64", 128}};
    for (auto& pr : profiles) {
        if (!blk.has(pr.key)) continue;
        ascon::ConvOut c = ascon::conv_with_iv(seed, iv, pr.ks);
        u128 ev = u128_from_hex(blk[pr.key]["value"].s()), et = u128_from_hex(blk[pr.key]["tag"].s());
        check(c.val == ev && c.tag == et, std::string(name) + " " + pr.key + " value/tag, vector " + std::to_string(idx));
    }
}

// ------------------------------------------------------------ FSS vectors
// SHA-256 in counter mode, the byte source of the reference
// (couplet/rng.py): stream block i = SHA-256(seed || u64le(i)).
class Sha256CounterRng {
public:
    explicit Sha256CounterRng(const std::vector<uint8_t>& seed) : counter_(0), pos_(32) {
        if (seed.size() != 32) throw std::runtime_error("Sha256CounterRng: seed must be 32 bytes");
        std::memcpy(seed_, seed.data(), 32);
    }
    void fill(uint8_t* out, size_t n) {
        while (n) {
            if (pos_ == 32) {
                uint8_t msg[40];
                std::memcpy(msg, seed_, 32);
                ascon::store64le(msg + 32, counter_++);
                sha256(msg, 40, buf_);
                pos_ = 0;
            }
            size_t take = 32 - pos_;
            if (take > n) take = n;
            std::memcpy(out, buf_ + pos_, take);
            pos_ += take;
            out += take;
            n -= take;
        }
    }
    // randbelow(2^bits) = u64le(randbytes(8)) mod 2^bits, bits <= 64.
    uint64_t randbelow_pow2(unsigned bits) {
        uint8_t b[8];
        fill(b, 8);
        uint64_t v = ascon::load64le(b);
        return bits >= 64 ? v : (v & (((uint64_t)1 << bits) - 1));
    }

private:
    uint8_t seed_[32];
    uint64_t counter_;
    uint8_t buf_[32];
    size_t pos_;
};

static std::string sha256_hex(const std::vector<uint8_t>& v) {
    std::array<uint8_t, 32> d = sha256(v);
    return hex(d.data(), 32);
}
static ascon::seed_t seed_from_hex(const std::string& s) {
    std::vector<uint8_t> v = unhex(s);
    if (v.size() != 16) throw std::runtime_error("seed must be 16 bytes");
    ascon::seed_t a;
    std::memcpy(a.data(), v.data(), 16);
    return a;
}
// A non-negative integer written either as a JSON number or as a hex string.
static uint64_t json_u64(const Json& j) {
    if (j.type == Json::STR) return hex_u64(j.s());
    return (uint64_t)j.i();
}

// Family-independent view of the two schemes for the vector loop.
struct KeyPairOut {
    std::vector<uint8_t> blob0, blob1;       // generated keys, encoded
    bool parse_ok;                           // recorded keys parse, equal the generated ones and re-encode
    std::vector<u128> y0, t0, y1, t1;        // EvalAll, both parties, both planes
    bool threads_ok;                         // EvalAll with 3 threads gives the same vectors
    bool party_check_ok;                     // Eval rejects the wrong party index
};

template <class Key, class GenFn, class EvalAllFn, class EvalFn>
static KeyPairOut run_family(GenFn gen, EvalAllFn eval_all, EvalFn eval, uint64_t alpha, uint64_t beta, u128 delta,
                             unsigned n, unsigned k, unsigned s, const ascon::seed_t& r0, const ascon::seed_t& r1,
                             const std::vector<uint8_t>& rec0, const std::vector<uint8_t>& rec1,
                             std::vector<fss::Share>* samples0, std::vector<fss::Share>* samples1,
                             const std::vector<uint64_t>& sample_x) {
    KeyPairOut out;
    std::pair<Key, Key> keys = gen(alpha, beta, delta, n, k, s, r0, r1);
    out.blob0 = keys.first.to_bytes();
    out.blob1 = keys.second.to_bytes();
    Key p0 = Key::from_bytes(rec0), p1 = Key::from_bytes(rec1);
    out.parse_ok = p0 == keys.first && p1 == keys.second && p0.to_bytes() == rec0 && p1.to_bytes() == rec1 &&
                   p0.b == 0 && p1.b == 1 && p0.size_bytes() == rec0.size() && p1.size_bytes() == rec1.size();
    // Evaluate the PARSED keys, so that the digests also cover the decoder.
    eval_all(p0, out.y0, out.t0, 1u);
    eval_all(p1, out.y1, out.t1, 1u);
    std::vector<u128> a, b;
    eval_all(p0, a, b, 3u);
    out.threads_ok = a == out.y0 && b == out.t0;
    eval_all(p1, a, b, 3u);
    out.threads_ok = out.threads_ok && a == out.y1 && b == out.t1;
    for (uint64_t x : sample_x) {
        samples0->push_back(eval(0u, p0, x));
        samples1->push_back(eval(1u, p1, x));
    }
    out.party_check_ok = false;
    try { eval(1u, p0, 0); } catch (const std::invalid_argument&) { out.party_check_ok = true; }
    return out;
}

static void check_fss_vectors(const Json& root) {
    check(root["design_version"].s() == "3.1", "fss vectors: design_version 3.1");
    const Json& vecs = root["vectors"];
    size_t n_big = 0, n_dpf = 0, n_dcf = 0;
    for (size_t vi = 0; vi < vecs.size(); vi++) {
        const Json& v = vecs[vi];
        const std::string family = v["family"].s();
        const unsigned n = (unsigned)v["n"].i(), k = (unsigned)v["k"].i(), s = (unsigned)v["s"].i();
        const unsigned w = k + s;
        const std::string tag = "fss vector " + std::to_string(vi) + " (" + family + " n=" + std::to_string(n) +
                                " k=" + std::to_string(k) + " s=" + std::to_string(s) + "): ";
        const bool is_dpf = family == "DPF";
        if (!is_dpf && family != "DCF") throw std::runtime_error("unknown family " + family);
        (is_dpf ? n_dpf : n_dcf)++;

        // Recorded inputs.
        const uint64_t alpha = json_u64(v["alpha"]);
        const uint64_t beta = hex_u64(v["beta"].s());
        const uint64_t delta0 = hex_u64(v["delta0"].s()), delta1 = hex_u64(v["delta1"].s());
        const u128 delta = u128_from_hex(v["delta"].s());
        const ascon::seed_t root0 = seed_from_hex(v["root_seed0"].s()), root1 = seed_from_hex(v["root_seed1"].s());

        // Regenerate them from the seed (draw order of make_vectors.py).
        const std::string label = "Couplet-fss-vector/" + family + "/" + std::to_string(n) + "/" + std::to_string(k);
        std::array<uint8_t, 32> seed_digest = sha256(label);
        check(hex(seed_digest.data(), 32) == v["rng_seed"].s(), tag + "rng_seed = SHA-256(label)");
        Sha256CounterRng rng(unhex(v["rng_seed"].s()));
        const uint64_t g_alpha = rng.randbelow_pow2(n), g_beta = rng.randbelow_pow2(k);
        const uint64_t g_d0 = rng.randbelow_pow2(s), g_d1 = rng.randbelow_pow2(s);
        ascon::seed_t g_r0, g_r1;
        rng.fill(g_r0.data(), 16);
        rng.fill(g_r1.data(), 16);
        check(g_alpha == alpha && g_beta == beta && g_d0 == delta0 && g_d1 == delta1 && g_r0 == root0 && g_r1 == root1,
              tag + "inputs regenerated from the seed");

        // MAC key: unreduced sum of the two shares.
        const u128 g_delta = fss::make_delta(g_d0, g_d1);
        const bool big = s < 128 && (g_delta >> s) != u128(0);
        if (big) n_big++;
        const u128 mask = u128_mask(w);
        const u128 beta_t = (g_delta * u128(g_beta)) & mask;
        check(g_delta == delta && big == v["delta_ge_2s"].b && beta_t == u128_from_hex(v["beta_t"].s()),
              tag + "delta = delta0 + delta1 unreduced, beta_t = delta * beta mod 2^w");

        // Keys, EvalAll and samples.
        const std::vector<uint8_t> rec0 = unhex(v["key0"].s()), rec1 = unhex(v["key1"].s());
        const Json& samples = v["samples"];
        std::vector<uint64_t> sample_x;
        for (size_t i = 0; i < samples.size(); i++) sample_x.push_back(json_u64(samples[i]["x"]));
        std::vector<fss::Share> s0, s1;
        KeyPairOut out;
        if (is_dpf) {
            out = run_family<fss::DpfKey>(
                [](uint64_t a, uint64_t b, u128 d, unsigned nn, unsigned kk, unsigned ss, const ascon::seed_t& x0,
                   const ascon::seed_t& x1) { return fss::dpf_gen(a, b, d, nn, kk, ss, x0, x1); },
                [](const fss::DpfKey& key, std::vector<u128>& y, std::vector<u128>& t, unsigned th) {
                    fss::dpf_eval_all(key, y, t, th);
                },
                [](unsigned b, const fss::DpfKey& key, uint64_t x) { return fss::dpf_eval(b, key, x); }, g_alpha,
                g_beta, g_delta, n, k, s, g_r0, g_r1, rec0, rec1, &s0, &s1, sample_x);
        } else {
            out = run_family<fss::DcfKey>(
                [](uint64_t a, uint64_t b, u128 d, unsigned nn, unsigned kk, unsigned ss, const ascon::seed_t& x0,
                   const ascon::seed_t& x1) { return fss::dcf_gen(a, b, d, nn, kk, ss, x0, x1); },
                [](const fss::DcfKey& key, std::vector<u128>& y, std::vector<u128>& t, unsigned th) {
                    fss::dcf_eval_all(key, y, t, th);
                },
                [](unsigned b, const fss::DcfKey& key, uint64_t x) { return fss::dcf_eval(b, key, x); }, g_alpha,
                g_beta, g_delta, n, k, s, g_r0, g_r1, rec0, rec1, &s0, &s1, sample_x);
        }
        check(out.blob0 == rec0, tag + "key0 bytes");
        check(out.blob1 == rec1, tag + "key1 bytes");
        const size_t formula = is_dpf ? fss::dpf_key_size(n, k, s) : fss::dcf_key_size(n, k, s);
        check(rec0.size() == (size_t)v["key_size_bytes"].i() && rec1.size() == rec0.size() && formula == rec0.size() &&
                  fss::encoded_key_family(rec0) == (is_dpf ? fss::FAMILY_DPF : fss::FAMILY_DCF),
              tag + "key size and family byte");
        check(out.parse_ok, tag + "recorded keys parse to the generated keys and re-encode");

        const Json& dg = v["evalall_sha256"];
        check(sha256_hex(fss::encode_vector(out.y0, w)) == dg["y0"].s(), tag + "EvalAll digest y0");
        check(sha256_hex(fss::encode_vector(out.t0, w)) == dg["tau0"].s(), tag + "EvalAll digest tau0");
        check(sha256_hex(fss::encode_vector(out.y1, w)) == dg["y1"].s(), tag + "EvalAll digest y1");
        check(sha256_hex(fss::encode_vector(out.t1, w)) == dg["tau1"].s(), tag + "EvalAll digest tau1");
        check(out.threads_ok, tag + "EvalAll with 3 threads equals EvalAll with 1 thread");

        // Reconstruction on the whole domain.
        bool sum_ok = out.y0.size() == ((size_t)1 << n);
        for (size_t x = 0; sum_ok && x < out.y0.size(); x++) {
            const bool ind = is_dpf ? (x == alpha) : (x < alpha);
            sum_ok = ((out.y0[x] + out.y1[x]) & mask) == (ind ? u128(beta) : u128(0)) &&
                     ((out.t0[x] + out.t1[x]) & mask) == (ind ? beta_t : u128(0));
        }
        check(sum_ok, tag + "y0 + y1 and tau0 + tau1 reconstruct the payload on the whole domain");

        // Sample points: recorded Eval outputs, Eval == EvalAll.
        for (size_t i = 0; i < samples.size(); i++) {
            const Json& sm = samples[i];
            const uint64_t x = sample_x[i];
            const bool ok = s0[i].y == u128_from_hex(sm["y0"].s()) && s0[i].tau == u128_from_hex(sm["tau0"].s()) &&
                            s1[i].y == u128_from_hex(sm["y1"].s()) && s1[i].tau == u128_from_hex(sm["tau1"].s()) &&
                            s0[i].y == out.y0[x] && s0[i].tau == out.t0[x] && s1[i].y == out.y1[x] &&
                            s1[i].tau == out.t1[x];
            check(ok, tag + "Eval at sample x=" + std::to_string(x));
        }
        check(out.party_check_ok, tag + "Eval rejects a mismatching party index");
    }
    if (root.has("n_vectors_with_delta_ge_2s"))
        check((size_t)root["n_vectors_with_delta_ge_2s"].i() == n_big, "fss vectors: number of vectors with delta >= 2^s");
    std::printf("fss vectors: %zu checked (%zu DPF, %zu DCF; %zu with delta >= 2^s)\n", vecs.size(), n_dpf, n_dcf, n_big);
}

// ------------------------------------------------------------ file lookup
static bool read_file(const std::string& path, std::string& text) {
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    text = ss.str();
    return !text.empty();
}
static std::string dir_of(const std::string& path) {  // with the trailing separator; "" if none
    size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string() : path.substr(0, pos + 1);
}

int main(int argc, char** argv) {
    std::vector<std::string> dirs;  // candidate vector directories, each with its trailing separator
    std::string ascon_path, fss_path;
    if (argc > 1) ascon_path = argv[1];
    if (argc > 2) fss_path = argv[2];
    if (ascon_path.empty()) {
        const std::string exe = dir_of(argc > 0 && argv[0] ? argv[0] : "");
        dirs.push_back(exe + "../../python/vectors/");
        const char* defaults[] = {"vectors/", "../vectors/", "python/vectors/", "../python/vectors/", "../../vectors/",
                                  "../../python/vectors/"};
        for (const char* d : defaults) dirs.push_back(d);
    }
    std::string text, path;
    if (!ascon_path.empty()) {
        if (read_file(ascon_path, text)) path = ascon_path;
    } else {
        for (auto& d : dirs)
            if (read_file(d + "ascon_vectors.json", text)) { path = d + "ascon_vectors.json"; break; }
    }
    if (path.empty()) {
        std::fprintf(stderr, "check_vectors: could not open ascon_vectors.json (pass the path as argument)\n");
        return 2;
    }
    if (fss_path.empty()) fss_path = dir_of(path) + "fss_vectors.json";
    std::printf("check_vectors: %s (%zu bytes)\n", path.c_str(), text.size());

    try {
        Json root = JsonParser(text).parse();
        if (root.has("design_version")) check(root["design_version"].s() == "3.1", "ascon vectors: design_version 3.1");

        // Permutation description.
        const Json& perm = root["permutation"];
        check(perm["rounds"].i() == 12, "12 rounds");
        bool rc_ok = perm["round_constants"].size() == 12;
        for (size_t i = 0; rc_ok && i < 12; i++) rc_ok = hex_u64(perm["round_constants"][i].s()) == ascon::RC[i];
        check(rc_ok, "round constants");
        const int rot[5][2] = {{19, 28}, {61, 39}, {1, 6}, {10, 17}, {7, 41}};
        bool rot_ok = perm["rotations"].size() == 5;
        for (size_t i = 0; rot_ok && i < 5; i++)
            rot_ok = perm["rotations"][i][0].i() == rot[i][0] && perm["rotations"][i][1].i() == rot[i][1];
        check(rot_ok, "linear-layer rotations");

        // Permutation vectors.
        const Json& pv = perm["vectors"];
        for (size_t i = 0; i < pv.size(); i++) {
            uint64_t in[5], out[5];
            read_state(pv[i]["input"], in);
            read_state(pv[i]["output"], out);
            uint64_t s[5];
            std::memcpy(s, in, 40);
            ascon::p12(s);
            bool ok = state_eq(s, out);
            // interleaved paths must agree too
            uint64_t st4[4][5];
            for (int j = 0; j < 4; j++) std::memcpy(st4[j], in, 40);
            ascon::p12_x4(st4);
            ok = ok && state_eq(st4[0], out) && state_eq(st4[3], out);
#if COUPLET_HAVE_AVX2
            for (int j = 0; j < 4; j++) std::memcpy(st4[j], in, 40);
            ascon::p12_x4_avx2(st4);
            ok = ok && state_eq(st4[0], out) && state_eq(st4[3], out);
#endif
            check(ok, "p12 vector " + std::to_string(i));
        }
        std::printf("permutation vectors: %zu checked\n", pv.size());

        // Hash vectors (empty message).
        if (root.has("hash_vectors")) {
            const Json& hv = root["hash_vectors"];
            for (auto& kv : hv.obj) {
                const Json& h = kv.second;
                bool big = h["byteorder"].s() == "big";
                std::string got = ascon_hash_generic(hex_u64(h["iv"].s()), big, (uint8_t)hex_u64(h["pad"].s()), {});
                check(got == h["empty_message"].s(), kv.first + " empty-message digest");
            }
        }

        // Labels / IVs.
        const Json& labels = root["labels"];
        struct { const char* key; const char* ascii; const ascon::IV& iv; } lab[4] = {
            {"G", "Couplet-G", ascon::IV_G()}, {"C", "Couplet-C", ascon::IV_C()},
            {"VL", "Couplet-VL", ascon::IV_VL()}, {"VR", "Couplet-VR", ascon::IV_VR()}};
        for (auto& l : lab) {
            const Json& j = labels[l.key];
            bool ok = j["ascii"].s() == l.ascii;
            for (int i = 0; i < 3; i++) ok = ok && hex_u64(j["iv_words"][(size_t)i].s()) == l.iv.w[i];
            check(ok, std::string("label ") + l.key + " IV words");
        }

        // PRG vectors.
        const Json& prg = root["prg_vectors"];
        for (size_t i = 0; i < prg.size(); i++) {
            const Json& v = prg[i];
            std::vector<uint8_t> sb = unhex(v["seed"].s());
            if (sb.size() != 16) throw std::runtime_error("seed must be 16 bytes");
            ascon::seed_t seed;
            std::memcpy(seed.data(), sb.data(), 16);

            const Json& g = v["G"];
            check_prg_states(g, seed, ascon::IV_G(), "G", i);
            ascon::GOut go = ascon::G(seed);
            bool ok = hex(go.sL.data(), 16) == g["sL"].s() && hex(go.sR.data(), 16) == g["sR"].s() &&
                      go.tL == g["tL"].i() && go.tR == g["tR"].i();
            check(ok, "G sL/tL/sR/tR, vector " + std::to_string(i));

            check_conv_block(v["Conv"], seed, ascon::IV_C(), "Conv", i);
            check_conv_block(v["ConvV_L"], seed, ascon::IV_VL(), "ConvV_L", i);
            check_conv_block(v["ConvV_R"], seed, ascon::IV_VR(), "ConvV_R", i);
        }
        std::printf("prg vectors: %zu seeds checked (G, Conv, ConvV_L, ConvV_R; k32_s64 and k64_s64)\n", prg.size());
    } catch (const std::exception& e) {
        std::printf("[FAIL] exception: %s\n", e.what());
        g_fail++;
    }
    const int ascon_checks = g_pass + g_fail;
    std::printf("ascon vectors: %d checks\n", ascon_checks);

    // FSS vectors.
    std::string fss_text;
    if (!read_file(fss_path, fss_text)) {
        std::fprintf(stderr, "check_vectors: could not open %s (pass the path as second argument)\n", fss_path.c_str());
        return 2;
    }
    std::printf("check_vectors: %s (%zu bytes)\n", fss_path.c_str(), fss_text.size());
    try {
        Json root = JsonParser(fss_text).parse();
        check_fss_vectors(root);
    } catch (const std::exception& e) {
        std::printf("[FAIL] exception: %s\n", e.what());
        g_fail++;
    }
    std::printf("fss vectors: %d checks\n", g_pass + g_fail - ascon_checks);

    std::printf("%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
