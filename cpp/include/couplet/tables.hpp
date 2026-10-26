// couplet/tables.hpp -- activation tables of the Couplet layer (DESIGN_M3.md
// section 3); the C++ counterpart of python/couplet/tables.py.
//
// The table of a unary function g at the wire format (n, f) has N = 2^n
// signed entries T[z] = round_half_even(g(signed(z) / 2^f) 2^f), indexed by
// the n-bit two's-complement code z, NOT reduced to n bits.  The native code
// has no correctly rounded transcendental functions, so sigmoid, tanh and
// GELU are read from the table files the reference writes; relu, sign,
// recip and rsqrt are exact integer maps and are also computed here
// (exact_entry), which the tests use to check the files.
//
// Table file table_<name>_n<n>_f<f>.bin (python/couplet/tables.py):
//     "CPLTTAB1" | le32 len(name) | name (UTF-8) | le32 n | le32 f |
//     2^n entries T[0], ..., T[2^n - 1] as int64 little-endian (signed)
// table_hash(name, n, f, entries) = SHA-256(frame("Couplet-table", name,
// le32 n, le32 f, entries as int64 LE)), the value a model file and the
// manifest bind; read_table verifies nothing but the layout, the callers
// compare the hash.
#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "hashes.hpp"
#include "u128.hpp"

namespace couplet {
namespace tables {

struct Table {
    std::string name;
    unsigned n = 0, f = 0;
    std::vector<int64_t> entries;  // 2^n signed entries
};

inline bool known_name(const std::string& name) {
    return name == "relu" || name == "sigmoid" || name == "tanh" || name == "gelu" || name == "sign" ||
           name == "recip" || name == "rsqrt";
}
inline bool exact_name(const std::string& name) {
    return name == "relu" || name == "sign" || name == "recip" || name == "rsqrt";
}

inline std::string table_file_name(const std::string& name, unsigned n, unsigned f) {
    return "table_" + name + "_n" + std::to_string(n) + "_f" + std::to_string(f) + ".bin";
}

namespace detail {
inline void le32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; i++) out.push_back((uint8_t)(v >> (8 * i)));
}
inline uint32_t rd32(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

// floor(sqrt(x)) for a 64-bit x.
inline uint64_t isqrt64(uint64_t x) {
    uint64_t r = 0;
    for (int b = 31; b >= 0; b--) {
        const uint64_t c = r | ((uint64_t)1 << b);
        if (c <= 0xFFFFFFFFull && c * c <= x) r = c;
    }
    return r;
}
// round(num / den), ties to even (num >= 0, den > 0).
inline uint64_t round_ratio(uint64_t num, uint64_t den) {
    uint64_t q = num / den;
    const uint64_t r = num % den;
    if (2 * r > den || (2 * r == den && (q & 1))) q++;
    return q;
}
// round(sqrt(num / den)), ties to even (python/couplet/tables._round_sqrt_ratio).
inline uint64_t round_sqrt_ratio(uint64_t num, uint64_t den) {
    const uint64_t t = isqrt64((4 * num) / den);  // floor(2 sqrt(num / den))
    if ((t & 1) && u128(t) * u128(t) * u128(den) == u128(4) * u128(num)) {
        const uint64_t q = (t - 1) / 2;
        return q + (q & 1);
    }
    return (t + 1) / 2;
}
}  // namespace detail

// T[z] of the exact tables (relu, sign, recip, rsqrt); f <= 20.
inline int64_t exact_entry(const std::string& name, uint64_t z, unsigned n, unsigned f) {
    if (n < 1 || n > 62 || f > 20) throw std::invalid_argument("exact_entry: format out of range");
    const uint64_t mask = ((uint64_t)1 << n) - 1;
    z &= mask;
    const int64_t v = ((z >> (n - 1)) & 1) ? (int64_t)(z | ~mask) : (int64_t)z;
    const int64_t scale = (int64_t)1 << f;
    if (name == "relu") return v > 0 ? v : 0;
    if (name == "sign") return v > 0 ? scale : (v < 0 ? -scale : 0);
    if (name == "recip") return v > 0 ? (int64_t)detail::round_ratio((uint64_t)scale * (uint64_t)scale, (uint64_t)v) : 0;
    if (name == "rsqrt")
        return v > 0 ? (int64_t)detail::round_sqrt_ratio((uint64_t)scale * (uint64_t)scale * (uint64_t)scale, (uint64_t)v) : 0;
    throw std::invalid_argument("exact_entry: " + name + " is not an exact table");
}

inline Table exact_table(const std::string& name, unsigned n, unsigned f) {
    if (!exact_name(name)) throw std::invalid_argument("exact_table: " + name + " is read from a table file");
    if (n > 26) throw std::invalid_argument("exact_table: n too large");
    Table t;
    t.name = name;
    t.n = n;
    t.f = f;
    t.entries.resize((size_t)1 << n);
    for (uint64_t z = 0; z < t.entries.size(); z++) t.entries[z] = exact_entry(name, z, n, f);
    return t;
}

// SHA-256(frame("Couplet-table", name, le32 n, le32 f, entries as int64 LE)).
inline hashes::hash32 table_hash(const std::string& name, unsigned n, unsigned f, const std::vector<int64_t>& entries) {
    std::vector<uint8_t> ln, lf, blob;
    detail::le32(ln, n);
    detail::le32(lf, f);
    blob.resize(entries.size() * 8);
    for (size_t j = 0; j < entries.size(); j++)
        for (int i = 0; i < 8; i++) blob[8 * j + (size_t)i] = (uint8_t)((uint64_t)entries[j] >> (8 * i));
    return hashes::framed_sha256({"Couplet-table", name, ln, lf, blob});
}
inline hashes::hash32 table_hash(const Table& t) { return table_hash(t.name, t.n, t.f, t.entries); }

// The table file of `t` (layout in the header comment).
inline std::vector<uint8_t> table_bytes(const Table& t) {
    std::vector<uint8_t> out = {'C', 'P', 'L', 'T', 'T', 'A', 'B', '1'};
    detail::le32(out, (uint32_t)t.name.size());
    out.insert(out.end(), t.name.begin(), t.name.end());
    detail::le32(out, t.n);
    detail::le32(out, t.f);
    for (int64_t e : t.entries)
        for (int i = 0; i < 8; i++) out.push_back((uint8_t)((uint64_t)e >> (8 * i)));
    return out;
}

inline Table parse_table(const uint8_t* data, size_t len) {
    if (len < 16 || std::memcmp(data, "CPLTTAB1", 8) != 0) throw std::invalid_argument("not a table file");
    const size_t ln = detail::rd32(data + 8);
    if (ln > 256 || len < 12 + ln + 8) throw std::invalid_argument("truncated table file");
    Table t;
    t.name.assign((const char*)data + 12, ln);
    size_t pos = 12 + ln;
    t.n = detail::rd32(data + pos);
    t.f = detail::rd32(data + pos + 4);
    pos += 8;
    if (t.n < 1 || t.n > 30 || len != pos + 8 * ((size_t)1 << t.n)) throw std::invalid_argument("table file has the wrong length");
    t.entries.resize((size_t)1 << t.n);
    for (size_t j = 0; j < t.entries.size(); j++) {
        uint64_t v = 0;
        for (int i = 7; i >= 0; i--) v = (v << 8) | data[pos + 8 * j + (size_t)i];
        t.entries[j] = (int64_t)v;
    }
    return t;
}

inline Table read_table(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open table file " + path);
    const std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse_table(data.data(), data.size());
}

}  // namespace tables
}  // namespace couplet
