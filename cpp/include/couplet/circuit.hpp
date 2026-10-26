// couplet/circuit.hpp -- Boolean circuits in the gate-list form of the Python
// compiler (DESIGN.md section 4): loader for the canonical binary format,
// plaintext evaluator, AND-layer schedule, and the store that hands circuits
// to the protocol code.  Nothing here compiles a circuit: the generation
// circuits are built by python/couplet/circuits_gen.py and exported by
// python/bench/export_circuits.py into artifact/circuits/.
//
// Binary format, version 2 (python/couplet/circuit.py, Circuit.to_bytes; all
// integers u32 little-endian):
//
//     magic "CPLTCIR2" (8 bytes)
//     n_wires | n_gates | n_input_decls | n_output_decls | n_layers
//     gates:   n_gates x (op, a, b, out)     op: 0 XOR, 1 AND, 2 NOT (b = a)
//     layers:  n_layers x (start, end)       gate-index ranges of the AND layers
//     inputs:  (owner, label_len, label utf-8, width, wires[width])
//                                            owner: 0, 1, 2 = BOUND
//     outputs: (owner, label_len, label utf-8, width, wires[width])
//                                            owner: 0, 1, 2 = PUBLIC, 3 = RETAIN
//
// Wire 0 is the constant 0 and wire 1 the constant 1; the input wires follow
// in declaration order; gate i drives wire 2 + n_inputs + i.  Gates are
// ordered [linear][AND layer 1][linear][AND layer 2]... so that all AND gates
// of a layer can be evaluated in one round.  Bit vectors are little-endian.
//
// Circuit::from_bytes validates this canonical form completely (the checks
// of Circuit.validate in the Python module, plus bounds), so the engine can
// rely on it.  Because every field is validated and nothing is optional, the
// file is the unique encoding of the circuit it describes; the circuit
// DIGEST is SHA-256 of the file and is what the two parties compare before
// they evaluate anything (abb.hpp puts it into the evaluation header).
//
//   BitMat          bits of `width` wires for `batch` instances, one byte per
//                   bit, row-major: v[i * batch + j] is bit i of instance j.
//   Circuit         the loaded circuit, its statistics, schedule() and
//                   to_bytes() (re-encoding, for tests).
//   evaluate        plaintext evaluation, 64 instances per machine word.
//   CircuitStore    directory of exported circuits with its manifest.json:
//                   get(name) loads <dir>/<name>.bin once (thread-safe) and
//                   refuses a file whose digest is not the one recorded in
//                   the manifest.
//   json::Value     a small JSON reader / writer helper (manifest, gate
//                   descriptors, state files, benchmark records).
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sha256.hpp"
#include "u128.hpp"

namespace couplet {

// ------------------------------------------------------------------ JSON
namespace json {

// Division of a u128 by a small constant (the MSVC helper type has no operator/).
inline u128 div_small(u128 x, uint32_t d, uint32_t* rem = nullptr) {
    uint64_t parts[4] = {u128_hi(x) >> 32, u128_hi(x) & 0xFFFFFFFFull, u128_lo(x) >> 32, u128_lo(x) & 0xFFFFFFFFull};
    uint64_t r = 0;
    for (int i = 0; i < 4; i++) {
        const uint64_t cur = (r << 32) | parts[i];
        parts[i] = cur / d;
        r = cur % d;
    }
    if (rem) *rem = (uint32_t)r;
    return u128_from((parts[2] << 32) | parts[3], (parts[0] << 32) | parts[1]);
}

// A parsed JSON value.  Numbers keep their source text so that integers up
// to 128 bits survive; objects keep the order of their members.
struct Value {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };
    Type type = NUL;
    bool boolean = false;
    std::string text;  // NUM: the literal; STR: the decoded string
    std::vector<Value> items;
    std::vector<std::pair<std::string, Value>> members;

    bool is_null() const { return type == NUL; }
    bool is_bool() const { return type == BOOL; }
    bool is_num() const { return type == NUM; }
    bool is_str() const { return type == STR; }
    bool is_arr() const { return type == ARR; }
    bool is_obj() const { return type == OBJ; }

    const Value* find(const std::string& key) const {
        if (type != OBJ) return nullptr;
        for (const std::pair<std::string, Value>& m : members)
            if (m.first == key) return &m.second;
        return nullptr;
    }
    const Value& at(const std::string& key) const {
        const Value* v = find(key);
        if (!v) throw std::invalid_argument("json: missing member \"" + key + "\"");
        return *v;
    }
    const std::string& str() const {
        if (type != STR) throw std::invalid_argument("json: expected a string");
        return text;
    }
    bool as_bool() const {
        if (type != BOOL) throw std::invalid_argument("json: expected a boolean");
        return boolean;
    }
    // Non-negative integer literal below 2^128.
    u128 as_u128() const {
        if (type != NUM || text.empty()) throw std::invalid_argument("json: expected a non-negative integer");
        u128 r = u128(0);
        for (char c : text) {
            if (c < '0' || c > '9') throw std::invalid_argument("json: expected a non-negative integer");
            const u128 digit = u128((uint64_t)(c - '0'));
            if (r > div_small(~u128(0) - digit, 10)) throw std::invalid_argument("json: integer too large");
            r = r * u128(10) + digit;
        }
        return r;
    }
    uint64_t as_u64() const {
        const u128 v = as_u128();
        if (u128_hi(v) != 0) throw std::invalid_argument("json: integer too large");
        return u128_lo(v);
    }
    double as_double() const {
        if (type != NUM) throw std::invalid_argument("json: expected a number");
        return std::strtod(text.c_str(), nullptr);
    }
};

namespace detail {

struct Parser {
    const char* p;
    const char* end;
    int depth;

    [[noreturn]] void fail(const char* what) const { throw std::invalid_argument(std::string("json: ") + what); }
    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
    bool eat(char c) {
        ws();
        if (p < end && *p == c) {
            p++;
            return true;
        }
        return false;
    }
    void expect(char c) {
        if (!eat(c)) fail("unexpected character");
    }
    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back((char)cp);
        } else if (cp < 0x800) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }
    uint32_t hex4() {
        if (end - p < 4) fail("truncated escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            const char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else fail("bad escape");
        }
        return v;
    }
    std::string string() {
        ws();
        if (p >= end || *p != '"') fail("expected a string");
        p++;
        std::string out;
        for (;;) {
            if (p >= end) fail("unterminated string");
            const char c = *p++;
            if (c == '"') break;
            if ((unsigned char)c < 0x20) fail("control character in a string");
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (p >= end) fail("unterminated string");
            const char e = *p++;
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
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        if (end - p < 2 || p[0] != '\\' || p[1] != 'u') fail("lone surrogate");
                        p += 2;
                        const uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo > 0xDFFF) fail("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("lone surrogate");
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return out;
    }
    Value value() {
        if (++depth > 64) fail("nesting too deep");
        ws();
        if (p >= end) fail("unexpected end");
        Value v;
        const char c = *p;
        if (c == '{') {
            p++;
            v.type = Value::OBJ;
            if (!eat('}')) {
                for (;;) {
                    std::string key = string();
                    expect(':');
                    v.members.emplace_back(std::move(key), value());
                    if (eat(',')) continue;
                    expect('}');
                    break;
                }
            }
        } else if (c == '[') {
            p++;
            v.type = Value::ARR;
            if (!eat(']')) {
                for (;;) {
                    v.items.push_back(value());
                    if (eat(',')) continue;
                    expect(']');
                    break;
                }
            }
        } else if (c == '"') {
            v.type = Value::STR;
            v.text = string();
        } else if (c == 't' && end - p >= 4 && std::memcmp(p, "true", 4) == 0) {
            p += 4;
            v.type = Value::BOOL;
            v.boolean = true;
        } else if (c == 'f' && end - p >= 5 && std::memcmp(p, "false", 5) == 0) {
            p += 5;
            v.type = Value::BOOL;
        } else if (c == 'n' && end - p >= 4 && std::memcmp(p, "null", 4) == 0) {
            p += 4;
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            const char* start = p;
            if (*p == '-') p++;
            while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) p++;
            if (p == start || (p == start + 1 && *start == '-')) fail("bad number");
            v.type = Value::NUM;
            v.text.assign(start, p);
        } else {
            fail("unexpected character");
        }
        depth--;
        return v;
    }
};

}  // namespace detail

inline Value parse(const char* data, size_t len) {
    detail::Parser ps{data, data + len, 0};
    Value v = ps.value();
    ps.ws();
    if (ps.p != ps.end) ps.fail("trailing characters");
    return v;
}
inline Value parse(const std::string& s) { return parse(s.data(), s.size()); }

// "..." with the escapes JSON requires (the input is taken as UTF-8).
inline std::string quote(const std::string& s) {
    static const char* hexd = "0123456789abcdef";
    std::string out = "\"";
    for (char ch : s) {
        const unsigned char c = (unsigned char)ch;
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            out += "\\u00";
            out.push_back(hexd[c >> 4]);
            out.push_back(hexd[c & 15]);
        } else out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

// Decimal text of an unsigned 128-bit integer.
inline std::string dec(u128 x) {
    if (x == u128(0)) return "0";
    std::string s;
    while (x != u128(0)) {
        uint32_t r = 0;
        x = div_small(x, 10, &r);
        s.push_back((char)('0' + r));
    }
    return std::string(s.rbegin(), s.rend());
}

}  // namespace json

namespace circuit {

typedef std::array<uint8_t, 32> digest_t;

const uint8_t OP_XOR = 0;
const uint8_t OP_AND = 1;
const uint8_t OP_NOT = 2;

const uint32_t WIRE_ZERO = 0;
const uint32_t WIRE_ONE = 1;

// Owner codes of the binary format.
const uint32_t OWNER_P0 = 0;
const uint32_t OWNER_P1 = 1;
const uint32_t IN_BOUND = 2;    // input: existing authenticated wires of the session
const uint32_t OUT_PUBLIC = 2;  // output: opened to both parties
const uint32_t OUT_RETAIN = 3;  // output: kept as authenticated wires, not opened

const size_t MAX_LABEL_BYTES = 4096;

inline std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(2 * n);
    for (size_t i = 0; i < n; i++) {
        s.push_back(d[p[i] >> 4]);
        s.push_back(d[p[i] & 15]);
    }
    return s;
}
inline std::string hex(const digest_t& d) { return hex(d.data(), d.size()); }
inline std::string hex(const std::vector<uint8_t>& v) { return hex(v.data(), v.size()); }

inline std::vector<uint8_t> unhex(const std::string& s) {
    if (s.size() % 2) throw std::invalid_argument("hex string of odd length");
    std::vector<uint8_t> out(s.size() / 2);
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        unsigned v;
        if (c >= '0' && c <= '9') v = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v = (unsigned)(c - 'A' + 10);
        else throw std::invalid_argument("bad hex digit");
        out[i / 2] = (uint8_t)((out[i / 2] << 4) | v);
    }
    return out;
}

// ------------------------------------------------------------------ bit matrices
// Bits of `width` wires for `batch` instances: v[i * batch + j] is bit i
// (little-endian position) of instance j, one byte per bit, values 0 / 1.
struct BitMat {
    size_t width = 0, batch = 0;
    std::vector<uint8_t> v;

    BitMat() {}
    BitMat(size_t w, size_t b) : width(w), batch(b), v(w * b, 0) {}

    uint8_t& at(size_t i, size_t j) { return v[i * batch + j]; }
    uint8_t at(size_t i, size_t j) const { return v[i * batch + j]; }

    void validate() const {
        if (v.size() != width * batch) throw std::invalid_argument("bit matrix has the wrong size");
        uint8_t acc = 0;
        for (uint8_t b : v) acc |= b;
        if (acc > 1) throw std::invalid_argument("bits must be 0 or 1");
    }

    // One value per instance; every value must fit into `width` bits.
    static BitMat from_u128(const std::vector<u128>& values, size_t width) {
        BitMat m(width, values.size());
        for (size_t j = 0; j < values.size(); j++) {
            if (width < 128 && (values[j] >> (unsigned)width) != u128(0))
                throw std::invalid_argument("value does not fit into the given width");
            for (size_t i = 0; i < width && i < 128; i++)
                m.v[i * m.batch + j] = (uint8_t)(u128_lo(values[j] >> (unsigned)i) & 1);
        }
        return m;
    }
    static BitMat from_u64(const std::vector<uint64_t>& values, size_t width) {
        std::vector<u128> wide(values.size());
        for (size_t j = 0; j < values.size(); j++) wide[j] = u128(values[j]);
        return from_u128(wide, width);
    }
    // The same value in every instance.
    static BitMat repeat_u128(u128 value, size_t width, size_t batch) {
        return from_u128(std::vector<u128>(batch, value), width);
    }
    // Instance j from the little-endian byte string values[j] (ceil(width / 8) bytes each).
    static BitMat from_bytes(const std::vector<std::vector<uint8_t>>& values, size_t width) {
        BitMat m(width, values.size());
        for (size_t j = 0; j < values.size(); j++) {
            if (values[j].size() != (width + 7) / 8) throw std::invalid_argument("value has the wrong length");
            for (size_t i = 0; i < width; i++) m.v[i * m.batch + j] = (uint8_t)((values[j][i / 8] >> (i % 8)) & 1);
            if (width % 8 && (values[j][width / 8] >> (width % 8)) != 0)
                throw std::invalid_argument("value does not fit into the given width");
        }
        return m;
    }

    // Bits [lo, lo + n) of instance j as an integer (n <= 128).
    u128 word(size_t j, size_t lo, size_t n) const {
        if (n > 128 || lo + n > width || j >= batch) throw std::out_of_range("BitMat::word");
        uint64_t w0 = 0, w1 = 0;
        for (size_t i = 0; i < n; i++) {
            const uint64_t bit = v[(lo + i) * batch + j] & 1;
            if (i < 64) w0 |= bit << i;
            else w1 |= bit << (i - 64);
        }
        return u128_from(w0, w1);
    }
    u128 word(size_t j) const { return word(j, 0, width); }
    uint64_t u64(size_t j, size_t lo, size_t n) const {
        if (n > 64) throw std::out_of_range("BitMat::u64");
        return u128_lo(word(j, lo, n));
    }
    uint64_t u64(size_t j) const { return u64(j, 0, width); }
    // Bits [lo, lo + n) of instance j as ceil(n / 8) little-endian bytes.
    std::vector<uint8_t> bytes(size_t j, size_t lo, size_t n) const {
        if (lo + n > width || j >= batch) throw std::out_of_range("BitMat::bytes");
        std::vector<uint8_t> out((n + 7) / 8, 0);
        for (size_t i = 0; i < n; i++) out[i / 8] = (uint8_t)(out[i / 8] | ((v[(lo + i) * batch + j] & 1) << (i % 8)));
        return out;
    }
    std::vector<uint8_t> bytes(size_t j) const { return bytes(j, 0, width); }

    friend bool operator==(const BitMat& a, const BitMat& b) {
        return a.width == b.width && a.batch == b.batch && a.v == b.v;
    }
    friend bool operator!=(const BitMat& a, const BitMat& b) { return !(a == b); }
};

// ------------------------------------------------------------------ circuits
// An input or output declaration.  Input wires are consecutive (validated).
struct Decl {
    uint32_t owner = 0;
    std::string label;
    std::vector<uint32_t> wires;
    size_t width() const { return wires.size(); }
};

struct Stats {
    uint64_t n_and = 0, n_xor = 0, n_not = 0, n_gates = 0, and_depth = 0, n_wires = 0, n_inputs = 0;
    uint64_t n_input_bits_p0 = 0, n_input_bits_p1 = 0, n_bound_input_bits = 0;
    uint64_t n_outputs = 0, n_public_output_bits = 0, n_private_output_bits_p0 = 0, n_private_output_bits_p1 = 0;
    uint64_t n_retain_output_bits = 0;

    std::string json() const {
        return "{\"n_and\":" + std::to_string(n_and) + ",\"n_xor\":" + std::to_string(n_xor) +
               ",\"n_not\":" + std::to_string(n_not) + ",\"n_gates\":" + std::to_string(n_gates) +
               ",\"and_depth\":" + std::to_string(and_depth) + ",\"n_wires\":" + std::to_string(n_wires) +
               ",\"n_inputs\":" + std::to_string(n_inputs) +
               ",\"n_input_bits_p0\":" + std::to_string(n_input_bits_p0) +
               ",\"n_input_bits_p1\":" + std::to_string(n_input_bits_p1) +
               ",\"n_bound_input_bits\":" + std::to_string(n_bound_input_bits) +
               ",\"n_outputs\":" + std::to_string(n_outputs) +
               ",\"n_public_output_bits\":" + std::to_string(n_public_output_bits) +
               ",\"n_private_output_bits_p0\":" + std::to_string(n_private_output_bits_p0) +
               ",\"n_private_output_bits_p1\":" + std::to_string(n_private_output_bits_p1) +
               ",\"n_retain_output_bits\":" + std::to_string(n_retain_output_bits) + "}";
    }
};

// One step of the schedule: the linear gates [lin_begin, lin_end) are local,
// then the AND gates [and_begin, and_end) form one round.  The last step has
// an empty AND range.
struct Step {
    uint32_t lin_begin, lin_end, and_begin, and_end;
};

class Circuit {
public:
    std::string name;       // file stem if loaded from a file, else empty
    uint32_t n_wires = 2;
    uint32_t n_input_wires = 0;
    std::vector<uint8_t> op;   // per gate: OP_XOR, OP_AND, OP_NOT
    std::vector<uint32_t> a;   // first operand wire
    std::vector<uint32_t> b;   // second operand wire (equal to a for OP_NOT)
    std::vector<std::pair<uint32_t, uint32_t>> layers;  // gate-index ranges of the AND layers
    std::vector<Decl> inputs;
    std::vector<Decl> outputs;
    digest_t digest{};      // SHA-256 of the canonical encoding
    size_t encoded_bytes = 0;

    size_t n_gates() const { return op.size(); }
    uint32_t first_gate_wire() const { return 2 + n_input_wires; }
    size_t and_depth() const { return layers.size(); }
    std::string digest_hex() const { return hex(digest); }

    const Decl* find_input(uint32_t owner, const std::string& label) const {
        for (const Decl& d : inputs)
            if (d.owner == owner && d.label == label) return &d;
        return nullptr;
    }
    const Decl* find_output(const std::string& label) const {
        for (const Decl& d : outputs)
            if (d.label == label) return &d;
        return nullptr;
    }
    bool has_input(uint32_t owner, const std::string& label) const { return find_input(owner, label) != nullptr; }
    bool has_output(const std::string& label) const { return find_output(label) != nullptr; }

    size_t input_bits(uint32_t owner) const {
        size_t n = 0;
        for (const Decl& d : inputs)
            if (d.owner == owner) n += d.wires.size();
        return n;
    }
    size_t output_bits(uint32_t owner) const {
        size_t n = 0;
        for (const Decl& d : outputs)
            if (d.owner == owner) n += d.wires.size();
        return n;
    }
    size_t n_and() const {
        size_t n = 0;
        for (const std::pair<uint32_t, uint32_t>& l : layers) n += l.second - l.first;
        return n;
    }

    Stats stats() const {
        Stats s;
        for (uint8_t code : op) {
            if (code == OP_AND) s.n_and++;
            else if (code == OP_XOR) s.n_xor++;
            else s.n_not++;
        }
        s.n_gates = op.size();
        s.and_depth = layers.size();
        s.n_wires = n_wires;
        s.n_inputs = n_input_wires;
        s.n_input_bits_p0 = input_bits(OWNER_P0);
        s.n_input_bits_p1 = input_bits(OWNER_P1);
        s.n_bound_input_bits = input_bits(IN_BOUND);
        for (const Decl& d : outputs) s.n_outputs += d.wires.size();
        s.n_public_output_bits = output_bits(OUT_PUBLIC);
        s.n_private_output_bits_p0 = output_bits(OWNER_P0);
        s.n_private_output_bits_p1 = output_bits(OWNER_P1);
        s.n_retain_output_bits = output_bits(OUT_RETAIN);
        return s;
    }

    // [(linear range, AND range)] per layer plus a final (linear, empty) step.
    std::vector<Step> schedule() const {
        std::vector<Step> steps;
        uint32_t prev = 0;
        for (const std::pair<uint32_t, uint32_t>& l : layers) {
            steps.push_back(Step{prev, l.first, l.first, l.second});
            prev = l.second;
        }
        const uint32_t ng = (uint32_t)op.size();
        steps.push_back(Step{prev, ng, ng, ng});
        return steps;
    }

    // Checks of the canonical form; throws std::invalid_argument on the first violation.
    void validate() const {
        const size_t ng = op.size();
        if (a.size() != ng || b.size() != ng) fail("gate arrays have different lengths");
        uint64_t next = 2;
        for (size_t i = 0; i < inputs.size(); i++) {
            const Decl& d = inputs[i];
            if (d.owner > IN_BOUND) fail("input " + d.label + ": bad owner");
            check_label(d.label);
            if (d.wires.empty()) fail("input " + d.label + ": empty");
            for (size_t j = 0; j < i; j++)
                if (inputs[j].owner == d.owner && inputs[j].label == d.label) fail("input " + d.label + ": duplicate");
            for (size_t j = 0; j < d.wires.size(); j++)
                if (d.wires[j] != next + j) fail("input " + d.label + ": wires are not numbered consecutively");
            next += d.wires.size();
        }
        if (next - 2 != n_input_wires) fail("n_input_wires does not match the input declarations");
        if ((uint64_t)n_wires != next + ng) fail("n_wires does not match inputs and gates");
        const uint64_t base = next;
        uint64_t prev_end = 0;
        for (const std::pair<uint32_t, uint32_t>& l : layers) {
            if (!(prev_end <= l.first && l.first < l.second && l.second <= ng))
                fail("AND layers must be non-empty, increasing and disjoint");
            prev_end = l.second;
        }
        size_t li = 0;
        for (size_t i = 0; i < ng; i++) {
            while (li < layers.size() && i >= layers[li].second) li++;
            const bool in_layer = li < layers.size() && layers[li].first <= i;
            const uint64_t limit = base + (in_layer ? layers[li].first : i);
            const uint8_t code = op[i];
            if (code == OP_AND) {
                if (!in_layer) fail("AND gate outside every AND layer");
            } else if (code == OP_XOR || code == OP_NOT) {
                if (in_layer) fail("linear gate inside an AND layer");
                if (code == OP_NOT && a[i] != b[i]) fail("NOT gate must repeat its operand");
            } else {
                fail("unknown op code");
            }
            if (a[i] >= limit || b[i] >= limit) fail("gate operand not available");
        }
        for (size_t i = 0; i < outputs.size(); i++) {
            const Decl& d = outputs[i];
            if (d.owner > OUT_RETAIN) fail("output " + d.label + ": bad owner");
            check_label(d.label);
            if (d.wires.empty()) fail("output " + d.label + ": empty");
            for (size_t j = 0; j < i; j++)
                if (outputs[j].label == d.label) fail("output " + d.label + ": duplicate");
            for (uint32_t w : d.wires)
                if (w >= n_wires) fail("output " + d.label + ": unknown wire");
        }
    }

    // Decode and validate the canonical encoding.
    static Circuit from_bytes(const uint8_t* data, size_t len) {
        static const char magic[9] = "CPLTCIR2";
        if (len >= 8 && std::memcmp(data, magic, 7) == 0 && data[7] != (uint8_t)magic[7])
            fail("unsupported circuit format version");
        if (len < 28 || std::memcmp(data, magic, 8) != 0) fail("bad magic");
        size_t pos = 8;
        const uint32_t n_wires = rd32(data + pos), ng = rd32(data + pos + 4), ni = rd32(data + pos + 8),
                       no = rd32(data + pos + 12), nl = rd32(data + pos + 16);
        pos += 20;
        if ((len - pos) / 16 < ng) fail("truncated data");
        Circuit c;
        c.n_wires = n_wires;
        c.op.resize(ng);
        c.a.resize(ng);
        c.b.resize(ng);
        const uint64_t base_expected = (uint64_t)n_wires - ng;  // checked again by validate()
        for (size_t i = 0; i < ng; i++) {
            const uint8_t* g = data + pos + 16 * i;
            const uint32_t code = rd32(g);
            if (code > OP_NOT) fail("unknown op code");
            c.op[i] = (uint8_t)code;
            c.a[i] = rd32(g + 4);
            c.b[i] = rd32(g + 8);
            if ((uint64_t)rd32(g + 12) != base_expected + i) fail("gate output wire is not 2 + n_inputs + index");
        }
        pos += 16 * (size_t)ng;
        if ((len - pos) / 8 < nl) fail("truncated data");
        c.layers.resize(nl);
        for (size_t i = 0; i < nl; i++) {
            c.layers[i].first = rd32(data + pos + 8 * i);
            c.layers[i].second = rd32(data + pos + 8 * i + 4);
        }
        pos += 8 * (size_t)nl;
        uint64_t n_in = 0;
        for (uint32_t i = 0; i < ni; i++) {
            c.inputs.push_back(read_decl(data, len, pos));
            if (c.inputs.back().owner > IN_BOUND) fail("bad input owner code");
            n_in += c.inputs.back().wires.size();
        }
        for (uint32_t i = 0; i < no; i++) {
            c.outputs.push_back(read_decl(data, len, pos));
            if (c.outputs.back().owner > OUT_RETAIN) fail("bad output owner code");
        }
        if (pos != len) fail("trailing bytes");
        if (n_in > 0xFFFFFFFFull) fail("too many input wires");
        c.n_input_wires = (uint32_t)n_in;
        c.validate();
        sha256(data, len, c.digest.data());
        c.encoded_bytes = len;
        return c;
    }
    static Circuit from_bytes(const std::vector<uint8_t>& data) { return from_bytes(data.data(), data.size()); }

    // The canonical encoding (byte-identical to the file a circuit was loaded from).
    std::vector<uint8_t> to_bytes() const {
        std::vector<uint8_t> out;
        out.reserve(28 + 16 * op.size() + 8 * layers.size());
        static const char magic[9] = "CPLTCIR2";
        out.insert(out.end(), magic, magic + 8);
        wr32(out, n_wires);
        wr32(out, (uint32_t)op.size());
        wr32(out, (uint32_t)inputs.size());
        wr32(out, (uint32_t)outputs.size());
        wr32(out, (uint32_t)layers.size());
        const uint32_t base = first_gate_wire();
        for (size_t i = 0; i < op.size(); i++) {
            wr32(out, op[i]);
            wr32(out, a[i]);
            wr32(out, b[i]);
            wr32(out, base + (uint32_t)i);
        }
        for (const std::pair<uint32_t, uint32_t>& l : layers) {
            wr32(out, l.first);
            wr32(out, l.second);
        }
        for (int pass = 0; pass < 2; pass++) {
            for (const Decl& d : (pass == 0 ? inputs : outputs)) {
                wr32(out, d.owner);
                wr32(out, (uint32_t)d.label.size());
                out.insert(out.end(), d.label.begin(), d.label.end());
                wr32(out, (uint32_t)d.wires.size());
                for (uint32_t w : d.wires) wr32(out, w);
            }
        }
        return out;
    }

    // Read <path> (a .bin file written by export_circuits.py).
    static Circuit load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open circuit file " + path);
        f.seekg(0, std::ios::end);
        const std::streamoff size = f.tellg();
        if (size < 0) throw std::runtime_error("cannot read circuit file " + path);
        f.seekg(0, std::ios::beg);
        std::vector<uint8_t> data((size_t)size);
        if (size > 0) f.read((char*)data.data(), size);
        if (!f) throw std::runtime_error("cannot read circuit file " + path);
        Circuit c = from_bytes(data);
        size_t slash = path.find_last_of("/\\");
        std::string stem = slash == std::string::npos ? path : path.substr(slash + 1);
        if (stem.size() > 4 && stem.compare(stem.size() - 4, 4, ".bin") == 0) stem.resize(stem.size() - 4);
        c.name = stem;
        return c;
    }

private:
    [[noreturn]] static void fail(const std::string& what) { throw std::invalid_argument("circuit: " + what); }
    static uint32_t rd32(const uint8_t* p) {
        return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    static void wr32(std::vector<uint8_t>& out, uint32_t v) {
        for (int i = 0; i < 4; i++) out.push_back((uint8_t)(v >> (8 * i)));
    }
    // Labels are non-empty printable ASCII without spaces (the Python module
    // also accepts other non-whitespace characters; the exported circuits use
    // ASCII only, and anything else is refused here).
    static void check_label(const std::string& label) {
        if (label.empty() || label.size() > MAX_LABEL_BYTES) fail("bad label");
        for (char ch : label) {
            const unsigned char c = (unsigned char)ch;
            if (c < 0x21 || c > 0x7E) fail("bad label (only printable ASCII without spaces is accepted)");
        }
    }
    static Decl read_decl(const uint8_t* data, size_t len, size_t& pos) {
        if (len - pos < 8) fail("truncated data");
        Decl d;
        d.owner = rd32(data + pos);
        const uint32_t ln = rd32(data + pos + 4);
        pos += 8;
        if (ln > MAX_LABEL_BYTES || len - pos < ln) fail("truncated data");
        d.label.assign((const char*)data + pos, ln);
        pos += ln;
        if (len - pos < 4) fail("truncated data");
        const uint32_t width = rd32(data + pos);
        pos += 4;
        if ((len - pos) / 4 < width) fail("truncated data");
        d.wires.resize(width);
        for (uint32_t i = 0; i < width; i++) d.wires[i] = rd32(data + pos + 4 * (size_t)i);
        pos += 4 * (size_t)width;
        return d;
    }
};

// ------------------------------------------------------------------ plaintext evaluation
typedef std::map<std::pair<uint32_t, std::string>, BitMat> PlainInputs;  // (owner, label) -> bits
typedef std::map<std::string, BitMat> PlainOutputs;                      // label -> bits

// Evaluate `c` on `batch` assignments.  `inputs` holds one BitMat per input
// declaration (BOUND inputs under the owner IN_BOUND), of the declared width
// and with batch 1 (used for every instance) or `batch`.  Every output,
// RETAIN outputs included, is returned.
inline PlainOutputs evaluate(const Circuit& c, const PlainInputs& inputs, size_t batch = 1) {
    if (batch < 1) throw std::invalid_argument("batch must be positive");
    if (inputs.size() != c.inputs.size()) throw std::invalid_argument("evaluate: wrong number of inputs");
    std::vector<const BitMat*> in(c.inputs.size());
    for (size_t i = 0; i < c.inputs.size(); i++) {
        const Decl& d = c.inputs[i];
        const PlainInputs::const_iterator it = inputs.find(std::make_pair(d.owner, d.label));
        if (it == inputs.end()) throw std::invalid_argument("evaluate: missing input " + d.label);
        const BitMat& m = it->second;
        if (m.width != d.wires.size() || (m.batch != 1 && m.batch != batch))
            throw std::invalid_argument("evaluate: input " + d.label + " has the wrong shape");
        m.validate();
        in[i] = &m;
    }
    PlainOutputs out;
    for (const Decl& d : c.outputs) out.emplace(d.label, BitMat(d.wires.size(), batch));
    std::vector<uint64_t> val(c.n_wires);
    const size_t ng = c.n_gates();
    const uint32_t base = c.first_gate_wire();
    for (size_t j0 = 0; j0 < batch; j0 += 64) {
        const size_t cnt = batch - j0 < 64 ? batch - j0 : 64;
        val[WIRE_ZERO] = 0;
        val[WIRE_ONE] = ~(uint64_t)0;
        for (size_t i = 0; i < c.inputs.size(); i++) {
            const Decl& d = c.inputs[i];
            const BitMat& m = *in[i];
            for (size_t q = 0; q < d.wires.size(); q++) {
                uint64_t w = 0;
                if (m.batch == 1) {
                    w = m.v[q] ? ~(uint64_t)0 : 0;
                } else {
                    for (size_t t = 0; t < cnt; t++) w |= (uint64_t)(m.v[q * m.batch + j0 + t] & 1) << t;
                }
                val[d.wires[q]] = w;
            }
        }
        for (size_t g = 0; g < ng; g++) {
            const uint64_t x = val[c.a[g]];
            const uint8_t code = c.op[g];
            val[base + g] = code == OP_NOT ? ~x : (code == OP_AND ? (x & val[c.b[g]]) : (x ^ val[c.b[g]]));
        }
        for (const Decl& d : c.outputs) {
            BitMat& m = out[d.label];
            for (size_t q = 0; q < d.wires.size(); q++) {
                const uint64_t w = val[d.wires[q]];
                for (size_t t = 0; t < cnt; t++) m.v[q * batch + j0 + t] = (uint8_t)((w >> t) & 1);
            }
        }
    }
    return out;
}

// The value of EVERY wire for `batch` assignments (row w of the result is wire
// w; inputs as for evaluate).  Meant for tests of small circuits: the result
// has n_wires * batch entries.
inline BitMat evaluate_wires(const Circuit& c, const PlainInputs& inputs, size_t batch = 1) {
    if (batch < 1) throw std::invalid_argument("batch must be positive");
    BitMat val(c.n_wires, batch);
    for (size_t j = 0; j < batch; j++) val.v[(size_t)WIRE_ONE * batch + j] = 1;
    size_t seen = 0;
    for (const Decl& d : c.inputs) {
        const PlainInputs::const_iterator it = inputs.find(std::make_pair(d.owner, d.label));
        if (it == inputs.end()) throw std::invalid_argument("evaluate_wires: missing input " + d.label);
        const BitMat& m = it->second;
        if (m.width != d.wires.size() || (m.batch != 1 && m.batch != batch))
            throw std::invalid_argument("evaluate_wires: input " + d.label + " has the wrong shape");
        m.validate();
        for (size_t q = 0; q < d.wires.size(); q++)
            for (size_t j = 0; j < batch; j++) val.v[(size_t)d.wires[q] * batch + j] = m.v[q * m.batch + (m.batch == 1 ? 0 : j)];
        seen++;
    }
    if (seen != inputs.size()) throw std::invalid_argument("evaluate_wires: wrong number of inputs");
    const uint32_t base = c.first_gate_wire();
    for (size_t g = 0; g < c.n_gates(); g++) {
        const size_t o = ((size_t)base + g) * batch, x = (size_t)c.a[g] * batch, y = (size_t)c.b[g] * batch;
        const uint8_t code = c.op[g];
        for (size_t j = 0; j < batch; j++)
            val.v[o + j] = code == OP_NOT ? (uint8_t)(val.v[x + j] ^ 1)
                                          : (code == OP_AND ? (uint8_t)(val.v[x + j] & val.v[y + j]) : (uint8_t)(val.v[x + j] ^ val.v[y + j]));
    }
    return val;
}

// ------------------------------------------------------------------ circuit store
// A directory of exported circuits and its manifest.json.  get(name) returns
// the circuit <dir>/<name>.bin; a circuit that is not listed in the manifest,
// or whose file does not hash to the recorded digest, is refused.
class CircuitStore {
public:
    explicit CircuitStore(const std::string& dir) : dir_(dir) {
        while (dir_.size() > 1 && (dir_.back() == '/' || dir_.back() == '\\')) dir_.pop_back();
        const std::string path = dir_ + "/manifest.json";
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const json::Value doc = json::parse(text);
        const json::Value& list = doc.at("circuits");
        if (!list.is_arr()) throw std::runtime_error(path + ": \"circuits\" is not a list");
        for (const json::Value& e : list.items) {
            Entry en;
            en.file = e.at("binary_file").str();
            const std::vector<uint8_t> d = unhex(e.at("digest").str());
            if (d.size() != 32) throw std::runtime_error(path + ": bad digest");
            std::memcpy(en.digest.data(), d.data(), 32);
            if (en.file.find('/') != std::string::npos || en.file.find('\\') != std::string::npos ||
                en.file.find("..") != std::string::npos)
                throw std::runtime_error(path + ": bad file name");
            entries_[e.at("name").str()] = en;
        }
    }
    CircuitStore(const CircuitStore&) = delete;
    CircuitStore& operator=(const CircuitStore&) = delete;

    const std::string& dir() const { return dir_; }
    bool has(const std::string& name) const { return entries_.count(name) != 0; }
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const std::pair<const std::string, Entry>& e : entries_) out.push_back(e.first);
        return out;
    }
    // The digest recorded in the manifest.
    digest_t manifest_digest(const std::string& name) const {
        const std::map<std::string, Entry>::const_iterator it = entries_.find(name);
        if (it == entries_.end()) throw std::runtime_error("circuit " + name + " is not in the manifest of " + dir_);
        return it->second.digest;
    }

    // The circuit `name`, loaded and validated on first use (thread-safe).
    std::shared_ptr<const Circuit> get(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::map<std::string, std::shared_ptr<const Circuit>>::const_iterator hit = loaded_.find(name);
        if (hit != loaded_.end()) return hit->second;
        const std::map<std::string, Entry>::const_iterator it = entries_.find(name);
        if (it == entries_.end())
            throw std::runtime_error("circuit " + name + " is not in the manifest of " + dir_ +
                                     " (export it with python/bench/export_circuits.py)");
        std::shared_ptr<Circuit> c(new Circuit(Circuit::load(dir_ + "/" + it->second.file)));
        if (c->digest != it->second.digest)
            throw std::runtime_error("circuit " + name + ": the file does not match the digest in the manifest");
        c->name = name;
        loaded_[name] = c;
        return c;
    }

    // Forget a loaded circuit (frees its memory once no session uses it).
    void drop(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        loaded_.erase(name);
    }

private:
    struct Entry {
        std::string file;
        digest_t digest{};
    };
    std::string dir_;
    std::map<std::string, Entry> entries_;
    std::map<std::string, std::shared_ptr<const Circuit>> loaded_;
    std::mutex mutex_;
};

// Directory part of a path ("." if there is none).
inline std::string dirname(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

}  // namespace circuit
}  // namespace couplet
