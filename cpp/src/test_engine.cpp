// test_engine.cpp -- tests of circuit.hpp and abb.hpp (DESIGN.md sections
// 2.2, 5.1, 7.6): the circuit loader, the plaintext evaluator and the
// authenticated Boolean engine.
//
// The circuits are small files exported by tests/gen_test_circuits.py (the
// C++ code compiles nothing); tests/circuits/cases.json holds input
// assignments with the outputs of the Python plaintext evaluator.
//
//   loader      every file loads, re-encodes to the same bytes and hashes to
//               the digest of the manifest; truncated, extended and mutated
//               encodings are rejected; the store refuses a file that does
//               not match its manifest.
//   plain       the C++ evaluator against the Python reference on every case,
//               single and batched (more than 64 instances), and against
//               integer arithmetic for the adder and the multiplier.
//   honest      every circuit on the engine with ideal preprocessing, batch 1
//               and 7: public outputs, private outputs at their owner only,
//               RETAIN wires with correct MACs, exact rounds and bytes.
//   chain       RETAIN outputs bound to BOUND inputs over three evaluations,
//               also with a batch-1 block shared by all instances.
//   pool        reserve_triples and the triple pool across evaluations.
//   strict      strict mode (allow_refill = false): an evaluation whose
//               triples are not in the pool is refused with TripleBudgetError
//               before any message and leaves the engine alive; the request
//               counters; a circuit without AND gates needs no pool.  Message
//               limits: a message longer than the circuit prescribes is
//               refused with the error of the step it belongs to.
//   real        the same with the real preprocessing (base OT, OT extension,
//               bucketed triples) and the accounting identity "phases sum to
//               the channel totals".
//   assignment  every AND gate instance consumes its own triple and every
//               input bit and private output bit its own mask: the public
//               transcript is compared, position by position, with the
//               plaintext wire values and the material a recording
//               preprocessing object handed out.
//   tamper      a party flips one bit of a message (TamperChannel): every
//               opened d / e bit of every AND layer, every bit of the output
//               opening (public shares and masked private shares): the honest
//               party throws ProtocolAbort("BoolCheck failed"), returns
//               nothing and its engine is dead.  An input substitution itself
//               is an honest run on another input and cannot be detected;
//               what is tested is the inconsistent variant (a wrong input
//               difference is SENT while the sender keeps computing with the
//               right one): the honest party aborts, or the wire influences
//               nothing that is opened and the outputs are unchanged; it
//               never accepts a wrong output.
//   mismatch    different circuits, batch sizes or bound wires at the two
//               parties, wrong headers and malformed messages abort.
//   arguments   invalid arguments throw std::invalid_argument before any
//               message and leave the engine alive.
//
// Both parties run as two threads of this process (run_two_party).
// Usage: test_engine [--circuits DIR] [--quick]
// Exit code 0 iff every check passes.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "couplet/abb.hpp"
#include "couplet/channel.hpp"
#include "couplet/circuit.hpp"
#include "couplet/gf128.hpp"
#include "couplet/hashes.hpp"
#include "couplet/prep.hpp"
#include "couplet/rng.hpp"
#include "couplet/shares.hpp"
#include "couplet/twoparty.hpp"

using namespace couplet;
using circuit::BitMat;
using circuit::Circuit;

// ------------------------------------------------------------ harness
static long long g_pass = 0, g_fail = 0;
static bool g_quick = false;

static void check(bool ok, const std::string& what) {
    if (ok) {
        g_pass++;
    } else {
        g_fail++;
        if (g_fail <= 60) std::printf("  FAIL: %s\n", what.c_str());
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

static bytes read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static BitMat random_bits(RandomSource& rng, size_t width, size_t batch) {
    BitMat m(width, batch);
    if (!m.v.empty()) rng.fill_bits(m.v.data(), m.v.size());
    return m;
}

// Random inputs of every declaration of a circuit (BOUND ones included).
static circuit::PlainInputs random_inputs(const Circuit& c, RandomSource& rng, size_t batch) {
    circuit::PlainInputs in;
    for (const circuit::Decl& d : c.inputs) in[std::make_pair(d.owner, d.label)] = random_bits(rng, d.wires.size(), batch);
    return in;
}

static abb::Inputs party_inputs(const Circuit& c, const circuit::PlainInputs& all, int role) {
    abb::Inputs in;
    for (const circuit::Decl& d : c.inputs)
        if (d.owner == (uint32_t)role) in[d.label] = all.at(std::make_pair(d.owner, d.label));
    return in;
}

static std::string g_dir;

// ------------------------------------------------------------ loader
static void test_loader(circuit::CircuitStore& store) {
    Section sec("loader");
    const std::vector<std::string> names = store.names();
    check(names.size() == 25, "25 test circuits in the manifest");
    for (const std::string& name : names) {
        const bytes raw = read_file(g_dir + "/" + name + ".bin");
        std::shared_ptr<const Circuit> c = store.get(name);
        check(c->digest == store.manifest_digest(name), name + ": digest equals the manifest");
        check(c->to_bytes() == raw, name + ": re-encoding gives the file");
        check(c->encoded_bytes == raw.size() && c->name == name, name + ": name and size");
        std::array<uint8_t, 32> h;
        sha256(raw.data(), raw.size(), h.data());
        check(h == c->digest, name + ": digest is SHA-256 of the file");
        const circuit::Stats st = c->stats();
        check(st.n_and == c->n_and() && st.n_gates == c->n_gates() && st.and_depth == c->and_depth() &&
                  st.n_and + st.n_xor + st.n_not == st.n_gates && st.n_wires == 2 + st.n_inputs + st.n_gates,
              name + ": statistics are consistent");
        const std::vector<circuit::Step> steps = c->schedule();
        bool sched_ok = steps.size() == c->and_depth() + 1;
        size_t next = 0;
        for (const circuit::Step& s : steps) {
            sched_ok = sched_ok && s.lin_begin == next && s.lin_end == s.and_begin && s.and_begin <= s.and_end;
            for (size_t g = s.lin_begin; g < s.lin_end; g++) sched_ok = sched_ok && c->op[g] != circuit::OP_AND;
            for (size_t g = s.and_begin; g < s.and_end; g++) sched_ok = sched_ok && c->op[g] == circuit::OP_AND;
            next = s.and_end;
        }
        check(sched_ok && next == c->n_gates(), name + ": schedule covers the gates in order");
    }

    // Malformed encodings of one circuit.
    const bytes good = read_file(g_dir + "/rnd05.bin");
    auto rejected = [](const bytes& data) { return throws_invalid([&] { Circuit::from_bytes(data); }); };
    check(!rejected(good), "the unmodified encoding is accepted");
    const size_t cuts[] = {0, 1, 7, 8, 27, 28, 29, 100, good.size() / 2, good.size() - 5, good.size() - 4, good.size() - 1};
    for (size_t cut : cuts) check(rejected(bytes(good.begin(), good.begin() + (ptrdiff_t)cut)), "truncated at " + std::to_string(cut));
    {
        bytes m = good;
        m.push_back(0);
        check(rejected(m), "trailing byte");
        m = good;
        m[0] ^= 1;
        check(rejected(m), "bad magic");
        m = good;
        m[7] = '1';
        check(rejected(m), "format version 1");
        m = good;
        m[7] = '3';
        check(rejected(m), "format version 3");
    }
    const Circuit base = Circuit::from_bytes(good);
    auto mutated = [&](const std::function<void(Circuit&)>& f) {
        Circuit c = base;
        f(c);
        return rejected(c.to_bytes());
    };
    check(!mutated([](Circuit&) {}), "re-encoded circuit is accepted");
    const size_t first_and = base.layers[0].first;
    size_t some_lin = 0, last_lin = base.n_gates() - 1;
    while (some_lin < base.n_gates() && base.op[some_lin] == circuit::OP_AND) some_lin++;
    while (last_lin > 0 && base.op[last_lin] == circuit::OP_AND) last_lin--;
    check(some_lin < base.n_gates() && base.op[last_lin] != circuit::OP_AND && base.op[first_and] == circuit::OP_AND,
          "test circuit has the expected shape");
    check(mutated([&](Circuit& c) { c.op[first_and] = circuit::OP_XOR; }), "linear gate inside an AND layer");
    check(mutated([&](Circuit& c) { c.op[some_lin] = circuit::OP_AND; }), "AND gate outside every layer");
    check(mutated([&](Circuit& c) { c.op[some_lin] = 3; }), "unknown op code");
    check(mutated([&](Circuit& c) { c.a[first_and] = c.first_gate_wire() + (uint32_t)first_and; }),
          "AND operand produced inside its own layer");
    check(mutated([&](Circuit& c) { c.b[last_lin] = c.first_gate_wire() + (uint32_t)last_lin; }), "gate reads its own output");
    check(mutated([&](Circuit& c) { c.a[some_lin] = c.n_wires + 5; }), "operand out of range");
    check(mutated([&](Circuit& c) { c.n_wires += 1; }), "wrong wire count");
    check(mutated([&](Circuit& c) { c.layers[0].second += 1; }), "layer range covers a linear gate");
    check(mutated([&](Circuit& c) { c.layers.push_back(c.layers[0]); }), "layers not increasing");
    check(mutated([&](Circuit& c) { c.layers.pop_back(); }), "AND gates outside the declared layers");
    check(mutated([&](Circuit& c) { c.inputs[0].wires[0] += 1; }), "input wires not consecutive");
    check(mutated([&](Circuit& c) { c.inputs[0].owner = 3; }), "bad input owner");
    check(mutated([&](Circuit& c) { c.inputs[1].label = c.inputs[0].label; c.inputs[1].owner = c.inputs[0].owner; }),
          "duplicate input");
    check(mutated([&](Circuit& c) { c.outputs[0].owner = 4; }), "bad output owner");
    check(mutated([&](Circuit& c) { c.outputs[1].label = c.outputs[0].label; }), "duplicate output label");
    check(mutated([&](Circuit& c) { c.outputs[0].wires[0] = c.n_wires; }), "output wire out of range");
    check(mutated([&](Circuit& c) { c.outputs[0].label = "a b"; }), "label with a space");
    check(mutated([&](Circuit& c) { c.outputs[0].label = ""; }), "empty label");
    check(mutated([&](Circuit& c) { c.outputs[0].wires.clear(); }), "empty output");
    {
        // A NOT gate whose second operand differs from the first.
        Circuit c = base;
        size_t g = 0;
        while (g < c.n_gates() && c.op[g] != circuit::OP_NOT) g++;
        check(g < c.n_gates(), "test circuit contains a NOT gate");
        if (g < c.n_gates()) {
            c.b[g] = circuit::WIRE_ZERO;
            check(rejected(c.to_bytes()), "NOT gate must repeat its operand");
        }
        // The output wire field of a gate is checked although it is implied.
        bytes m = good;
        m[28 + 12] ^= 1;
        check(rejected(m), "gate output wire is not 2 + n_inputs + index");
    }

    // The store refuses a file that does not match the manifest, and a name that is not listed.
    // (a uniquely named directory outside the source tree: parallel runs do not share it)
    namespace fs = std::filesystem;
    SystemRng sys_rng;
    const fs::path tmp = fs::temp_directory_path() / ("couplet_test_engine_store_" + circuit::hex(sys_rng.get(8)));
    std::error_code ec;
    fs::create_directories(tmp);
    {
        bytes other = read_file(g_dir + "/rnd04.bin");
        std::ofstream((tmp / "rnd05.bin").string(), std::ios::binary).write((const char*)other.data(), (std::streamsize)other.size());
        std::ofstream((tmp / "ok.bin").string(), std::ios::binary).write((const char*)good.data(), (std::streamsize)good.size());
        std::ofstream man((tmp / "manifest.json").string(), std::ios::binary);
        man << "{\"circuits\": [{\"name\": \"rnd05\", \"binary_file\": \"rnd05.bin\", \"digest\": \"" << base.digest_hex()
            << "\"}, {\"name\": \"ok\", \"binary_file\": \"ok.bin\", \"digest\": \"" << base.digest_hex() << "\"}]}";
    }
    {
        circuit::CircuitStore bad(tmp.string());
        bool refused = false;
        try {
            bad.get("rnd05");
        } catch (const std::runtime_error&) {
            refused = true;
        }
        check(refused, "store refuses a file whose digest differs from the manifest");
        check(bad.get("ok")->digest == base.digest, "store accepts the matching file");
        refused = false;
        try {
            bad.get("nope");
        } catch (const std::runtime_error&) {
            refused = true;
        }
        check(refused, "store refuses a circuit that is not in the manifest");
    }
    fs::remove_all(tmp, ec);

    // JSON helper.
    const json::Value v = json::parse("{\"a\": [1, 2.5, -3, \"x\\n\\u00e9\\ud83d\\ude00\"], \"b\": {\"t\": true, \"n\": null}, "
                                      "\"big\": 340282366920938463463374607431768211455}");
    check(v.at("a").items.size() == 4 && v.at("a").items[0].as_u64() == 1 && v.at("a").items[1].as_double() == 2.5,
          "json numbers");
    check(v.at("a").items[3].str() == std::string("x\n\xc3\xa9\xf0\x9f\x98\x80"), "json string escapes");
    check(v.at("b").at("t").as_bool() && v.at("b").at("n").is_null(), "json literals");
    check(v.at("big").as_u128() == ~u128(0) && json::dec(~u128(0)) == "340282366920938463463374607431768211455",
          "json 128-bit integers");
    check(throws_invalid([] { json::parse("{\"a\": 1,}"); }) && throws_invalid([] { json::parse("[1] x"); }) &&
              throws_invalid([] { json::parse("{\"a\": 340282366920938463463374607431768211456}").at("a").as_u128(); }) &&
              throws_invalid([] { json::parse("\"abc"); }),
          "malformed json is rejected");
    check(json::quote("a\"b\\c\n") == "\"a\\\"b\\\\c\\n\"", "json quoting");
}

// ------------------------------------------------------------ plaintext evaluator
static void test_plain(circuit::CircuitStore& store) {
    Section sec("plain");
    const std::string text = [&] {
        const bytes raw = read_file(g_dir + "/cases.json");
        return std::string(raw.begin(), raw.end());
    }();
    const json::Value doc = json::parse(text);
    CtrPrgRng rng(101);
    size_t n_cases = 0;
    for (const json::Value& e : doc.at("circuits").items) {
        const std::string name = e.at("name").str();
        std::shared_ptr<const Circuit> c = store.get(name);
        check(c->digest_hex() == e.at("digest").str(), name + ": digest of the case file");
        check(c->n_and() == e.at("n_and").as_u64() && c->and_depth() == e.at("and_depth").as_u64() &&
                  c->n_gates() == e.at("n_gates").as_u64(),
              name + ": statistics of the Python build");
        const std::vector<json::Value>& cases = e.at("cases").items;
        std::vector<std::map<std::string, bytes>> ins, outs;
        for (const json::Value& cs : cases) {
            circuit::PlainInputs in;
            std::map<std::string, bytes> flat_in, flat_out;
            for (const circuit::Decl& d : c->inputs) {
                const std::string key = (d.owner == circuit::IN_BOUND ? std::string("bound") : std::to_string(d.owner)) + ":" + d.label;
                const bytes val = circuit::unhex(cs.at("inputs").at(key).str());
                in[std::make_pair(d.owner, d.label)] = BitMat::from_bytes({val}, d.wires.size());
                flat_in[key] = val;
            }
            const circuit::PlainOutputs out = circuit::evaluate(*c, in, 1);
            bool ok = out.size() == c->outputs.size();
            for (const circuit::Decl& d : c->outputs) {
                const bytes want = circuit::unhex(cs.at("outputs").at(d.label).str());
                ok = ok && out.at(d.label).bytes(0) == want;
                flat_out[d.label] = want;
            }
            check(ok, name + ": plaintext evaluation equals the Python reference");
            ins.push_back(flat_in);
            outs.push_back(flat_out);
            n_cases++;
        }
        // All cases at once, repeated to more than 64 instances.
        const size_t batch = 70;
        circuit::PlainInputs in;
        for (const circuit::Decl& d : c->inputs) {
            const std::string key = (d.owner == circuit::IN_BOUND ? std::string("bound") : std::to_string(d.owner)) + ":" + d.label;
            std::vector<bytes> vals;
            for (size_t j = 0; j < batch; j++) vals.push_back(ins[j % ins.size()].at(key));
            in[std::make_pair(d.owner, d.label)] = BitMat::from_bytes(vals, d.wires.size());
        }
        const circuit::PlainOutputs out = circuit::evaluate(*c, in, batch);
        bool ok = true;
        for (const circuit::Decl& d : c->outputs)
            for (size_t j = 0; j < batch; j++) ok = ok && out.at(d.label).bytes(j) == outs[j % outs.size()].at(d.label);
        check(ok, name + ": batched plaintext evaluation (70 instances)");
    }
    check(n_cases == 125, "125 reference cases");

    // Arithmetic meaning of two circuits.
    std::shared_ptr<const Circuit> add = store.get("add32"), mul = store.get("mul16");
    bool ok_add = true, ok_mul = true;
    for (int t = 0; t < 200; t++) {
        const uint64_t a = rng.u64() & 0xFFFFFFFFull, b = t == 0 ? 0xFFFFFFFFull : (rng.u64() & 0xFFFFFFFFull);
        circuit::PlainInputs in;
        in[std::make_pair(0u, std::string("a"))] = BitMat::from_u64({a}, 32);
        in[std::make_pair(1u, std::string("b"))] = BitMat::from_u64({b}, 32);
        const circuit::PlainOutputs out = circuit::evaluate(*add, in);
        ok_add = ok_add && out.at("sum").u64(0) == ((a + b) & 0xFFFFFFFFull) && out.at("carry").u64(0) == ((a + b) >> 32);
        circuit::PlainInputs im;
        im[std::make_pair(0u, std::string("a"))] = BitMat::from_u64({a & 0xFFFF}, 16);
        im[std::make_pair(1u, std::string("b"))] = BitMat::from_u64({b & 0xFFFF}, 16);
        ok_mul = ok_mul && circuit::evaluate(*mul, im).at("prod").u64(0) == (a & 0xFFFF) * (b & 0xFFFF);
    }
    check(ok_add, "add32 adds");
    check(ok_mul, "mul16 multiplies");

    // Argument checks of the evaluator and of BitMat.
    circuit::PlainInputs in = random_inputs(*add, rng, 1);
    check(throws_invalid([&] { circuit::evaluate(*add, in, 0); }), "batch 0 is rejected");
    in.erase(std::make_pair(0u, std::string("a")));
    check(throws_invalid([&] { circuit::evaluate(*add, in, 1); }), "missing input is rejected");
    in[std::make_pair(0u, std::string("a"))] = BitMat(31, 1);
    check(throws_invalid([&] { circuit::evaluate(*add, in, 1); }), "input of the wrong width is rejected");
    in[std::make_pair(0u, std::string("a"))] = BitMat(32, 1);
    in[std::make_pair(0u, std::string("a"))].v[3] = 2;
    check(throws_invalid([&] { circuit::evaluate(*add, in, 1); }), "non-binary input is rejected");
    check(throws_invalid([] { BitMat::from_u64({8}, 3); }) && !throws_invalid([] { BitMat::from_u64({7}, 3); }),
          "BitMat::from_u64 checks the width");
    check(throws_invalid([] { BitMat::from_bytes({bytes{0x10}}, 4); }) && throws_invalid([] { BitMat::from_bytes({bytes{1, 2}}, 4); }),
          "BitMat::from_bytes checks the value");
    const BitMat wide = BitMat::from_u128({u128_from(0x0123456789abcdefull, 0xfedcba9876543210ull)}, 128);
    check(wide.word(0) == u128_from(0x0123456789abcdefull, 0xfedcba9876543210ull) && wide.u64(0, 64, 64) == 0xfedcba9876543210ull &&
              wide.u64(0, 4, 8) == 0xde && wide.bytes(0, 8, 16) == bytes({0xcd, 0xab}),
          "BitMat word extraction");
}

// ------------------------------------------------------------ engine runs
struct Run {
    abb::EvalResult res;
    abb::Transcript tr;
    bool dead = false;
    ChannelStats eval_stats;
};

// One evaluation by both parties with ideal preprocessing.  tamper_party /
// rules: message rewriting at one party (messages are counted from 0).
struct Pair {
    PartyOutcome<Run> p[2];
};

static Pair run_ideal(const Circuit& c, const circuit::PlainInputs& all, size_t batch, int tamper_party = -1,
                      std::map<size_t, TamperChannel::Rule> rules = std::map<size_t, TamperChannel::Rule>(),
                      prep::IdealDealer* dealer_out = nullptr) {
    prep::IdealDealer local;
    prep::IdealDealer& dealer = dealer_out ? *dealer_out : local;
    const bytes sid = {'t', 'e', 's', 't'};
    auto party = [&](int role) {
        return [&, role](Channel& raw) {
            TamperChannel tc(raw, tamper_party == role ? rules : std::map<size_t, TamperChannel::Rule>());
            Channel& ch = tamper_party == role ? static_cast<Channel&>(tc) : raw;
            prep::IdealPrep prep(role, dealer);
            abb::Engine eng(role, ch, prep, sid);
            Run r;
            try {
                r.res = eng.eval_circuit(c, party_inputs(c, all, role), abb::Bound(), batch);
            } catch (...) {
                // The engine must be dead and stay dead.
                bool again = false;
                try {
                    eng.eval_circuit(c, party_inputs(c, all, role), abb::Bound(), batch);
                } catch (const ProtocolAbort&) {
                    again = true;
                } catch (...) {
                }
                if (!eng.dead() || !again || !prep.aborted() || !eng.prep().delta().is_zero())
                    throw std::logic_error("engine not retired after a failure");
                throw;
            }
            r.tr = eng.last_transcript();
            r.dead = eng.dead();
            r.eval_stats = ch.stats();
            return r;
        };
    };
    Pair out;
    std::pair<PartyOutcome<Run>, PartyOutcome<Run>> pr = run_two_party(party(0), party(1));
    out.p[0] = std::move(pr.first);
    out.p[1] = std::move(pr.second);
    return out;
}

// Compare the outcome of an honest run with the plaintext evaluation.
static bool outputs_match(const Circuit& c, const Pair& pr, const circuit::PlainOutputs& want, std::string* why = nullptr) {
    auto fail = [&](const std::string& s) {
        if (why) *why = s;
        return false;
    };
    for (int role = 0; role < 2; role++)
        if (!pr.p[role].ok) return fail("party " + std::to_string(role) + " failed: " + pr.p[role].error);
    for (const circuit::Decl& d : c.outputs) {
        for (int role = 0; role < 2; role++) {
            const abb::EvalResult& r = pr.p[role].value.res;
            if (d.owner == circuit::OUT_PUBLIC) {
                if (!r.pub.count(d.label) || r.pub.at(d.label) != want.at(d.label)) return fail("public output " + d.label);
                if (r.priv.count(d.label) || r.retained.count(d.label)) return fail("public output listed twice");
            } else if (d.owner == circuit::OUT_RETAIN) {
                if (!r.retained.count(d.label) || r.pub.count(d.label) || r.priv.count(d.label)) return fail("retain " + d.label);
            } else {
                const bool mine = d.owner == (uint32_t)role;
                if (r.pub.count(d.label) || r.priv.count(d.label) != (mine ? 1u : 0u)) return fail("private output placement " + d.label);
                if (mine && r.priv.at(d.label) != want.at(d.label)) return fail("private output " + d.label);
            }
        }
    }
    return true;
}

static uint64_t expected_rounds(const Circuit& c) {
    const bool opens = c.output_bits(circuit::OUT_PUBLIC) + c.output_bits(0) + c.output_bits(1) > 0;
    return 1 + c.and_depth() + (c.n_and() ? 2 : 0) + (opens ? 3 : 0);
}
static uint64_t expected_bytes(const Circuit& c, int role, size_t batch) {
    uint64_t total = 32 + (c.input_bits((uint32_t)role) * batch + 7) / 8;
    for (const std::pair<uint32_t, uint32_t>& l : c.layers) total += (2 * (uint64_t)(l.second - l.first) * batch + 7) / 8;
    if (c.n_and()) total += 96;
    const uint64_t n_open = c.output_bits(circuit::OUT_PUBLIC) + c.output_bits(0) + c.output_bits(1);
    if (n_open) total += (n_open * batch + 7) / 8 + 96;
    return total;
}

static void test_honest(circuit::CircuitStore& store) {
    Section sec("honest (ideal preprocessing)");
    CtrPrgRng rng(202);
    for (const std::string& name : store.names()) {
        std::shared_ptr<const Circuit> c = store.get(name);
        if (c->input_bits(circuit::IN_BOUND)) continue;  // chain_b / chain_c: see the chain test
        for (size_t batch : {(size_t)1, (size_t)7}) {
            const circuit::PlainInputs all = random_inputs(*c, rng, batch);
            const circuit::PlainOutputs want = circuit::evaluate(*c, all, batch);
            prep::IdealDealer dealer;
            const Pair pr = run_ideal(*c, all, batch, -1, {}, &dealer);
            std::string why;
            const bool ok = outputs_match(*c, pr, want, &why);
            check(ok, name + " batch " + std::to_string(batch) + ": outputs (" + why + ")");
            if (!ok) continue;
            const gf128 delta = dealer.delta();
            bool mac_ok = true, tr_ok = true;
            for (const circuit::Decl& d : c->outputs) {
                if (d.owner == circuit::OUT_RETAIN) {
                    const abb::WireShares& w0 = pr.p[0].value.res.retained.at(d.label);
                    const abb::WireShares& w1 = pr.p[1].value.res.retained.at(d.label);
                    mac_ok = mac_ok && w0.width == d.wires.size() && w0.batch == batch && w0.ids == w1.ids;
                    for (size_t i = 0; mac_ok && i < w0.bits.size(); i++) {
                        const uint8_t v = (uint8_t)(w0.bits[i] ^ w1.bits[i]);
                        mac_ok = v == want.at(d.label).v[i] && (w0.macs[i] ^ w1.macs[i]) == prep::gf_and_bit(delta, v);
                    }
                } else if (d.owner != circuit::OUT_PUBLIC) {
                    // The opening of a private wire is public and equal at both parties.
                    tr_ok = tr_ok && pr.p[0].value.tr.masked.at(d.label) == pr.p[1].value.tr.masked.at(d.label);
                }
            }
            check(mac_ok, name + ": retained wires carry correct shares and MACs");
            check(tr_ok, name + ": masked openings agree");
            // The input differences are public and agree; d = x XOR r.
            check(pr.p[0].value.tr.input_diffs[0] == pr.p[1].value.tr.input_diffs[0] &&
                      pr.p[0].value.tr.input_diffs[1] == pr.p[1].value.tr.input_diffs[1],
                  name + ": input differences agree");
            for (int role = 0; role < 2; role++) {
                const abb::EvalStats& s = pr.p[role].value.res.stats;
                check(s.rounds == expected_rounds(*c) && pr.p[role].stats.rounds == expected_rounds(*c),
                      name + ": rounds = 1 + depth + 2 + 1 + 2");
                check(s.bytes_sent == expected_bytes(*c, role, batch) && pr.p[role].stats.bytes_sent == s.bytes_sent,
                      name + ": bytes sent");
                check(s.triples == c->n_and() * batch && s.checked_bits == 2 * c->n_and() * batch + s.opened_bits &&
                          s.n_gates == c->n_gates() && s.batch == batch,
                      name + ": statistics");
            }
        }
    }
    // Private-output masks hide the value: over many instances the masked opening is not the value.
    std::shared_ptr<const Circuit> c = store.get("mul16");
    const size_t batch = 64;
    const circuit::PlainInputs all = random_inputs(*c, rng, batch);
    const circuit::PlainOutputs want = circuit::evaluate(*c, all, batch);
    const Pair pr = run_ideal(*c, all, batch);
    check(outputs_match(*c, pr, want), "mul16 batch 64");
    if (pr.p[0].ok && pr.p[1].ok) {
        const BitMat& masked = pr.p[0].value.tr.masked.at("prod");
        size_t diff = 0, ones = 0;
        for (size_t i = 0; i < masked.v.size(); i++) {
            diff += masked.v[i] != want.at("prod").v[i];
            ones += masked.v[i];
        }
        const size_t n = masked.v.size();  // 2048 bits: both counts are binomial(n, 1/2)
        check(diff > n / 2 - 200 && diff < n / 2 + 200 && ones > n / 2 - 200 && ones < n / 2 + 200,
              "masked private openings look uniform (" + std::to_string(diff) + ", " + std::to_string(ones) + " of " + std::to_string(n) + ")");
        check(!pr.p[0].value.res.priv.count("prod") && pr.p[1].value.res.priv.count("prod"), "only the owner unmasks");
    }
}

// ------------------------------------------------------------ chain (BOUND / RETAIN), pool
struct ChainOut {
    std::vector<abb::EvalResult> res;
    uint64_t triple_requests = 0, triples_used = 0;
    size_t available = 0;
    std::string json_book;
};

static void test_chain(circuit::CircuitStore& store) {
    Section sec("chain (BOUND / RETAIN) and triple pool");
    CtrPrgRng rng(303);
    std::shared_ptr<const Circuit> ca = store.get("chain_a"), cb = store.get("chain_b"), cc = store.get("chain_c");
    for (int variant = 0; variant < 3; variant++) {
        // variant 0: batch 1 throughout; 1: batch 4 throughout; 2: chain_a with batch 1, the rest with batch 3.
        const size_t ba = variant == 1 ? 4 : 1, bb = variant == 0 ? 1 : (variant == 1 ? 4 : 3);
        circuit::PlainInputs ia = random_inputs(*ca, rng, ba), ib = random_inputs(*cb, rng, bb), ic = random_inputs(*cc, rng, bb);
        const circuit::PlainOutputs wa = circuit::evaluate(*ca, ia, ba);
        ib[std::make_pair(circuit::IN_BOUND, std::string("keep"))] = wa.at("keep");
        const circuit::PlainOutputs wb = circuit::evaluate(*cb, ib, bb);
        ic[std::make_pair(circuit::IN_BOUND, std::string("keep2"))] = wb.at("keep2");
        ic[std::make_pair(circuit::IN_BOUND, std::string("extra"))] = wa.at("extra");
        const circuit::PlainOutputs wc = circuit::evaluate(*cc, ic, bb);
        prep::IdealDealer dealer;
        const bytes sid = {'c', 'h', 'a', 'i', 'n'};
        const size_t reserve = variant == 0 ? ca->n_and() + cb->n_and() + 7 : 0;
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                prep::IdealPrep prep(role, dealer);
                abb::Engine eng(role, ch, prep, sid);
                ChainOut out;
                if (reserve) eng.reserve_triples(reserve);
                out.res.push_back(eng.eval_circuit(*ca, party_inputs(*ca, ia, role), abb::Bound(), ba));
                abb::Bound bound_b;
                bound_b["keep"] = out.res[0].retained.at("keep");
                out.res.push_back(eng.eval_circuit(*cb, party_inputs(*cb, ib, role), bound_b, bb));
                abb::Bound bound_c;
                bound_c["keep2"] = out.res[1].retained.at("keep2");
                bound_c["extra"] = out.res[0].retained.at("extra");
                out.res.push_back(eng.eval_circuit(*cc, party_inputs(*cc, ic, role), bound_c, bb));
                out.triple_requests = eng.triple_requests();
                out.triples_used = eng.triples_used();
                out.available = eng.triples_available();
                out.json_book = eng.book().json();
                return out;
            };
        };
        const std::pair<PartyOutcome<ChainOut>, PartyOutcome<ChainOut>> pr = run_two_party(party(0), party(1));
        const std::string tag = "chain variant " + std::to_string(variant);
        check(pr.first.ok && pr.second.ok, tag + ": no abort (" + pr.first.error + pr.second.error + ")");
        if (!pr.first.ok || !pr.second.ok) continue;
        const PartyOutcome<ChainOut>* po[2] = {&pr.first, &pr.second};
        const Circuit* cs[3] = {ca.get(), cb.get(), cc.get()};
        const circuit::PlainOutputs* ws[3] = {&wa, &wb, &wc};
        for (int stage = 0; stage < 3; stage++) {
            bool ok = true;
            for (const circuit::Decl& d : cs[stage]->outputs) {
                for (int role = 0; role < 2; role++) {
                    const abb::EvalResult& r = po[role]->value.res[(size_t)stage];
                    if (d.owner == circuit::OUT_PUBLIC) ok = ok && r.pub.at(d.label) == ws[stage]->at(d.label);
                    else if (d.owner == (uint32_t)role) ok = ok && r.priv.at(d.label) == ws[stage]->at(d.label);
                }
            }
            check(ok, tag + ": outputs of stage " + std::to_string(stage));
        }
        // Wire identifiers are session-wide and equal at both parties.
        const abb::WireShares& k0 = pr.first.value.res[0].retained.at("keep");
        const abb::WireShares& e0 = pr.first.value.res[0].retained.at("extra");
        const abb::WireShares& k2 = pr.first.value.res[1].retained.at("keep2");
        check(k0.ids.front() == 0 && k0.ids.back() == 23 && e0.ids.front() == 24 && k2.ids.front() == 29 &&
                  k2.ids == pr.second.value.res[1].retained.at("keep2").ids,
              tag + ": wire identifiers");
        const uint64_t used = ca->n_and() * ba + (cb->n_and() + cc->n_and()) * bb;
        check(pr.first.value.triples_used == used && pr.second.value.triples_used == used, tag + ": triples used");
        if (variant == 0) {
            // One reservation covers chain_a and chain_b; chain_c needs cc.n_and - 7 more: two requests.
            check(pr.first.value.triple_requests == 2 && pr.first.value.available == 0, tag + ": the pool is refilled once");
        } else {
            check(pr.first.value.triple_requests == 3 && pr.first.value.available == 0, tag + ": one request per evaluation");
        }
        check(pr.first.value.json_book.find("\"online\"") != std::string::npos &&
                  pr.first.value.json_book.find("\"outcheck\"") != std::string::npos,
              tag + ": phases are booked");
    }
}

// ------------------------------------------------------------ real preprocessing
struct RealOut {
    std::vector<abb::EvalResult> res;
    bool sums_ok = false;
    uint64_t prep_bytes = 0, book_bytes = 0, total_bytes = 0, rounds = 0, prep_rounds = 0, book_rounds = 0;
    unsigned bucket = 0;
};

static void test_real(circuit::CircuitStore& store) {
    Section sec("real preprocessing");
    CtrPrgRng rng(404);
    const std::vector<std::string> names = g_quick ? std::vector<std::string>{"rnd05", "consts"}
                                                    : std::vector<std::string>{"rnd09", "mul16", "consts", "xoronly", "rnd03"};
    std::vector<std::shared_ptr<const Circuit>> cs;
    std::vector<circuit::PlainInputs> ins;
    std::vector<circuit::PlainOutputs> wants;
    std::vector<size_t> batches;
    for (size_t i = 0; i < names.size(); i++) {
        cs.push_back(store.get(names[i]));
        batches.push_back(i % 2 ? 5 : 1);
        ins.push_back(random_inputs(*cs[i], rng, batches[i]));
        wants.push_back(circuit::evaluate(*cs[i], ins[i], batches[i]));
    }
    std::shared_ptr<const Circuit> ca = store.get("chain_a"), cb = store.get("chain_b");
    circuit::PlainInputs ia = random_inputs(*ca, rng, 2), ib = random_inputs(*cb, rng, 2);
    const circuit::PlainOutputs wa = circuit::evaluate(*ca, ia, 2);
    ib[std::make_pair(circuit::IN_BOUND, std::string("keep"))] = wa.at("keep");
    const circuit::PlainOutputs wb = circuit::evaluate(*cb, ib, 2);

    const bytes sid = {'r', 'e', 'a', 'l'};
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            prep::RealPrep prep(role, ch, sid, 40);
            prep.setup();
            abb::Engine eng(role, ch, prep, sid);
            RealOut out;
            for (size_t i = 0; i < cs.size(); i++)
                out.res.push_back(eng.eval_circuit(*cs[i], party_inputs(*cs[i], ins[i], role), abb::Bound(), batches[i]));
            out.res.push_back(eng.eval_circuit(*ca, party_inputs(*ca, ia, role), abb::Bound(), 2));
            abb::Bound bound;
            bound["keep"] = out.res.back().retained.at("keep");
            out.res.push_back(eng.eval_circuit(*cb, party_inputs(*cb, ib, role), bound, 2));
            const prep::PhaseStat pt = prep.stats().total(), bt = eng.book().total();
            const ChannelStats st = ch.stats();
            out.prep_bytes = pt.bytes_sent;
            out.book_bytes = bt.bytes_sent;
            out.total_bytes = st.bytes_sent;
            out.rounds = st.rounds;
            out.prep_rounds = pt.rounds;
            out.book_rounds = bt.rounds;
            out.sums_ok = pt.bytes_sent + bt.bytes_sent == st.bytes_sent && pt.bytes_recv + bt.bytes_recv == st.bytes_recv &&
                          pt.rounds + bt.rounds == st.rounds && pt.msgs_sent + bt.msgs_sent == st.msgs_sent;
            out.bucket = prep.last_bucket_size();
            return out;
        };
    };
    const std::pair<PartyOutcome<RealOut>, PartyOutcome<RealOut>> pr = run_two_party(party(0), party(1));
    check(pr.first.ok && pr.second.ok, "real preprocessing: no abort (" + pr.first.error + pr.second.error + ")");
    if (!pr.first.ok || !pr.second.ok) return;
    const PartyOutcome<RealOut>* po[2] = {&pr.first, &pr.second};
    for (size_t i = 0; i <= cs.size() + 1; i++) {
        const Circuit& c = i < cs.size() ? *cs[i] : (i == cs.size() ? *ca : *cb);
        const circuit::PlainOutputs& want = i < cs.size() ? wants[i] : (i == cs.size() ? wa : wb);
        bool ok = true;
        for (const circuit::Decl& d : c.outputs) {
            for (int role = 0; role < 2; role++) {
                const abb::EvalResult& r = po[role]->value.res[i];
                if (d.owner == circuit::OUT_PUBLIC) ok = ok && r.pub.count(d.label) && r.pub.at(d.label) == want.at(d.label);
                else if (d.owner == (uint32_t)role) ok = ok && r.priv.count(d.label) && r.priv.at(d.label) == want.at(d.label);
            }
        }
        check(ok, "real preprocessing: outputs of " + c.name);
    }
    for (int role = 0; role < 2; role++) {
        check(po[role]->value.sums_ok, "real preprocessing: the phases sum to the channel totals");
        check(po[role]->value.bucket >= 3, "real preprocessing: a bucket size was used");
    }
    check(pr.first.value.rounds == pr.second.value.rounds && pr.first.value.total_bytes > 0, "both parties count the same rounds");
    uint64_t engine_bytes[2] = {0, 0};
    for (int role = 0; role < 2; role++) {
        for (size_t i = 0; i < cs.size(); i++) engine_bytes[role] += expected_bytes(*cs[i], role, batches[i]);
        engine_bytes[role] += expected_bytes(*ca, role, 2) + expected_bytes(*cb, role, 2);
        check(po[role]->value.book_bytes == engine_bytes[role], "real preprocessing: the engine's own bytes are as with ideal preprocessing");
    }
}

// ------------------------------------------------------------ use of the preprocessed material
// A preprocessing object that records what it hands out (tests only).
class RecordingPrep : public prep::Prep {
public:
    RecordingPrep(int role, prep::IdealDealer& dealer) : inner_(role, dealer) {}
    int role() const override { return inner_.role(); }
    bool ideal() const override { return true; }
    bool aborted() const override { return inner_.aborted(); }
    gf128 delta() const override { return inner_.delta(); }
    void setup() override {}
    void abort() override { inner_.abort(); }
    const prep::PrepCounters& counters() const override { return inner_.counters(); }
    prep::Shares random_bits(size_t n) override { return inner_.random_bits(n); }
    prep::BdozPair random_bdoz(size_t n) override { return inner_.random_bdoz(n); }
    prep::InputMasks input_masks(int owner, size_t n) override {
        prep::InputMasks m = inner_.input_masks(owner, n);
        masks[owner].push_back(m);
        return m;
    }
    prep::Triples and_triples(size_t n) override {
        prep::Triples t = inner_.and_triples(n);
        triples.a.append(t.a);
        triples.b.append(t.b);
        triples.c.append(t.c);
        return t;
    }

    prep::Triples triples;                   // every triple handed out, in order
    std::vector<prep::InputMasks> masks[2];  // per owner: one entry per request

private:
    prep::IdealPrep inner_;
};

// Every AND gate instance uses its own triple, every input bit and every private output bit
// its own mask: the public transcript (opened d, e, input differences, masked openings) is
// compared with the plaintext wire values and the material the dealer handed out, position by
// position.  A reused or misplaced triple or mask is found with overwhelming probability.
static void test_assignment(circuit::CircuitStore& store) {
    Section sec("use of triples and masks");
    CtrPrgRng rng(808);
    struct Job {
        std::shared_ptr<const Circuit> c;
        size_t batch;
        circuit::PlainInputs in;
    };
    std::vector<Job> jobs;
    for (const std::pair<const char*, size_t>& j : std::vector<std::pair<const char*, size_t>>{{"rnd09", 3}, {"mul16", 5}, {"rnd06", 1}, {"consts", 4}}) {
        Job job;
        job.c = store.get(j.first);
        job.batch = j.second;
        job.in = random_inputs(*job.c, rng, job.batch);
        jobs.push_back(job);
    }
    prep::IdealDealer dealer;
    RecordingPrep rec0(0, dealer), rec1(1, dealer);
    RecordingPrep* rec[2] = {&rec0, &rec1};
    const bytes sid = {'a', 's', 'g'};
    const size_t reserve = jobs[0].c->n_and() * jobs[0].batch + 100;  // the second evaluation starts inside the pool
    struct Out {
        std::vector<bytes> sent;
        std::vector<size_t> first_msg;
        std::vector<abb::Transcript> tr;
        std::vector<abb::EvalResult> res;
    };
    auto party = [&](int role) {
        return [&, role](Channel& raw) {
            Out out;
            std::map<size_t, TamperChannel::Rule> rules;
            for (size_t i = 0; i < 2000; i++)
                rules[i] = [&out](const bytes& m) {
                    out.sent.push_back(m);
                    return m;
                };
            TamperChannel ch(raw, rules);
            abb::Engine eng(role, ch, *rec[role], sid);
            eng.reserve_triples(reserve);
            for (const Job& job : jobs) {
                out.first_msg.push_back(ch.messages_sent());
                out.res.push_back(eng.eval_circuit(*job.c, party_inputs(*job.c, job.in, role), abb::Bound(), job.batch));
                out.tr.push_back(eng.last_transcript());
            }
            return out;
        };
    };
    const std::pair<PartyOutcome<Out>, PartyOutcome<Out>> pr = run_two_party(party(0), party(1));
    check(pr.first.ok && pr.second.ok, "recorded run: no abort (" + pr.first.error + pr.second.error + ")");
    if (!pr.first.ok || !pr.second.ok) return;
    const Out* o[2] = {&pr.first.value, &pr.second.value};
    size_t triple_base = 0;
    for (size_t e = 0; e < jobs.size(); e++) {
        const Circuit& c = *jobs[e].c;
        const size_t B = jobs[e].batch;
        const BitMat val = circuit::evaluate_wires(c, jobs[e].in, B);
        // AND gates: d = x XOR a and e = y XOR b with the triple of position (gate, instance).
        long long good = 0, total = 0, triples_ok = 0;
        size_t and_index = 0;
        for (size_t layer = 0; layer < c.and_depth(); layer++) {
            const size_t m = c.layers[layer].second - c.layers[layer].first;
            const prep::BitVec s0 = prep::unpack_bit_msg(o[0]->sent.at(o[0]->first_msg[e] + 1 + layer), 2 * m * B);
            const prep::BitVec s1 = prep::unpack_bit_msg(o[1]->sent.at(o[1]->first_msg[e] + 1 + layer), 2 * m * B);
            for (size_t g = 0; g < m; g++, and_index++) {
                const size_t gi = c.layers[layer].first + g;
                for (size_t j = 0; j < B; j++) {
                    const size_t t = triple_base + and_index * B + j;
                    const uint8_t ta = (uint8_t)(rec0.triples.a.bits[t] ^ rec1.triples.a.bits[t]);
                    const uint8_t tb = (uint8_t)(rec0.triples.b.bits[t] ^ rec1.triples.b.bits[t]);
                    const uint8_t tc = (uint8_t)(rec0.triples.c.bits[t] ^ rec1.triples.c.bits[t]);
                    const uint8_t d = (uint8_t)(s0[g * B + j] ^ s1[g * B + j]), ee = (uint8_t)(s0[(m + g) * B + j] ^ s1[(m + g) * B + j]);
                    total += 2;
                    good += d == (val.at(c.a[gi], j) ^ ta);
                    good += ee == (val.at(c.b[gi], j) ^ tb);
                    triples_ok += tc == (ta & tb);
                }
            }
        }
        const std::string tag = c.name + " batch " + std::to_string(B);
        if (total) {
            g_pass += good;
            g_fail += total - good;
            if (good != total) std::printf("  FAIL: %s: opened d / e use the triple of their position (%lld of %lld)\n", tag.c_str(), good, total);
            check(triples_ok == total / 2 && and_index == c.n_and(), tag + ": the recorded triples are AND triples");
        }
        triple_base += c.n_and() * B;
        // Inputs: d = x XOR r with the mask of position (wire, instance); private outputs: y XOR r'
        // with the masks that follow the input masks of the owner.
        for (int owner = 0; owner < 2; owner++) {
            size_t n_in = 0, n_priv = 0;
            for (const circuit::Decl& d : c.inputs) n_in += d.owner == (uint32_t)owner ? d.wires.size() : 0;
            for (const circuit::Decl& d : c.outputs) n_priv += d.owner == (uint32_t)owner ? d.wires.size() : 0;
            if (n_in + n_priv == 0) continue;
            // Requests of this owner's masks so far: one per earlier evaluation that needed any.
            size_t req = 0;
            for (size_t p = 0; p < e; p++) {
                size_t cnt = 0;
                for (const circuit::Decl& d : jobs[p].c->inputs) cnt += d.owner == (uint32_t)owner ? d.wires.size() : 0;
                for (const circuit::Decl& d : jobs[p].c->outputs) cnt += d.owner == (uint32_t)owner ? d.wires.size() : 0;
                req += cnt ? 1 : 0;
            }
            const prep::InputMasks& own = rec[owner]->masks[owner].at(req);
            const prep::InputMasks& other = rec[1 - owner]->masks[owner].at(req);
            bool shape = own.has_bits && own.bits.size() == (n_in + n_priv) * B && other.shares.bits.size() == own.bits.size();
            for (size_t i = 0; shape && i < own.bits.size(); i++) shape = other.shares.bits[i] == 0 && own.shares.bits[i] == own.bits[i];
            check(shape, tag + ": input masks of P_" + std::to_string(owner) + " are held by the owner only");
            if (!shape) continue;
            long long ok_in = 0, n_checked_in = 0, ok_out = 0, n_checked_out = 0;
            size_t q = 0;
            for (const circuit::Decl& d : c.inputs) {
                if (d.owner != (uint32_t)owner) continue;
                for (size_t i = 0; i < d.wires.size(); i++, q++)
                    for (size_t j = 0; j < B; j++) {
                        n_checked_in++;
                        ok_in += o[0]->tr[e].input_diffs[owner][q * B + j] == (val.at(d.wires[i], j) ^ own.bits[q * B + j]);
                    }
            }
            for (const circuit::Decl& d : c.outputs) {
                if (d.owner != (uint32_t)owner) continue;
                const BitMat& masked = o[1 - owner]->tr[e].masked.at(d.label);
                for (size_t i = 0; i < d.wires.size(); i++, q++)
                    for (size_t j = 0; j < B; j++) {
                        n_checked_out++;
                        ok_out += masked.at(i, j) == (val.at(d.wires[i], j) ^ own.bits[q * B + j]);
                    }
            }
            g_pass += ok_in + ok_out;
            g_fail += (n_checked_in - ok_in) + (n_checked_out - ok_out);
            if (ok_in != n_checked_in || ok_out != n_checked_out)
                std::printf("  FAIL: %s: masks of P_%d are used at their positions (inputs %lld of %lld, outputs %lld of %lld)\n", tag.c_str(),
                            owner, ok_in, n_checked_in, ok_out, n_checked_out);
        }
    }
    check(rec0.triples.size() == triple_base && rec1.triples.size() == triple_base,
          "the engine requested exactly the triples it consumed (" + std::to_string(triple_base) + ")");
}

// ------------------------------------------------------------ tamper
// What both parties did when party `cheater` rewrote message `msg` with `rule`.
struct Verdict {
    bool honest_abort = false;   // the honest party threw ProtocolAbort
    bool honest_output = false;  // the honest party returned outputs
    bool cheater_ok = false;
    std::string honest_error, cheater_error;
    Pair pair;
};

static Verdict tamper(const Circuit& c, const circuit::PlainInputs& all, size_t batch, int cheater, size_t msg,
                      TamperChannel::Rule rule) {
    std::map<size_t, TamperChannel::Rule> rules;
    rules[msg] = std::move(rule);
    Verdict v;
    v.pair = run_ideal(c, all, batch, cheater, rules);
    const PartyOutcome<Run>& h = v.pair.p[1 - cheater];
    v.honest_abort = h.protocol_abort;
    v.honest_output = h.ok;
    v.honest_error = h.error;
    v.cheater_ok = v.pair.p[cheater].ok;
    v.cheater_error = v.pair.p[cheater].error;
    return v;
}

static void test_tamper(circuit::CircuitStore& store) {
    Section sec("tamper");
    CtrPrgRng rng(505);
    // Message numbers of a party with ideal preprocessing: 0 input, 1 .. depth AND layers,
    // depth + 1, depth + 2 first BoolCheck, depth + 3 opening, depth + 4, depth + 5 second BoolCheck.
    size_t total_pub = 0, total_priv = 0;
    for (const std::string& name : g_quick ? std::vector<std::string>{"rnd03"} : std::vector<std::string>{"rnd03", "rnd05", "mul16"}) {
        std::shared_ptr<const Circuit> c = store.get(name);
        const size_t batch = name == "rnd05" ? 3 : 1;
        const circuit::PlainInputs all = random_inputs(*c, rng, batch);
        const size_t depth = c->and_depth();
        const circuit::PlainOutputs want = circuit::evaluate(*c, all, batch);
        check(outputs_match(*c, run_ideal(*c, all, batch), want), name + ": honest control run");

        // (a) every opened d / e bit of every AND layer (mul16: a sample of the layers).
        size_t n_flip = 0, n_caught = 0, n_cheater_abort = 0;
        for (size_t layer = 0; layer < depth; layer++) {
            if (name == "mul16" && layer % 6 != 0) continue;
            const size_t nbits = 2 * (c->layers[layer].second - c->layers[layer].first) * batch;
            for (size_t bit = 0; bit < nbits; bit++) {
                const int cheater = (int)((layer + bit) & 1);
                const Verdict v = tamper(*c, all, batch, cheater, 1 + layer, flip_bit_rule(bit));
                n_flip++;
                n_caught += v.honest_abort && !v.honest_output && v.honest_error == "BoolCheck failed";
                n_cheater_abort += !v.cheater_ok;
            }
        }
        check(n_caught == n_flip && n_flip > 0, name + ": every flipped d / e share is caught by the first BoolCheck (" +
                                                    std::to_string(n_caught) + " of " + std::to_string(n_flip) + ")");
        check(n_cheater_abort == n_flip, name + ": the cheater gets no output either");

        // (b) every bit of the output opening: public shares, then masked private shares.
        const size_t n_pub = c->output_bits(circuit::OUT_PUBLIC) * batch;
        const size_t n_open = n_pub + (c->output_bits(0) + c->output_bits(1)) * batch;
        size_t pub_caught = 0, priv_caught = 0;
        for (size_t bit = 0; bit < n_open; bit++) {
            const int cheater = (int)(bit & 1);
            const Verdict v = tamper(*c, all, batch, cheater, depth + 3, flip_bit_rule(bit));
            const bool caught = v.honest_abort && !v.honest_output && v.honest_error == "BoolCheck failed" && !v.cheater_ok;
            (bit < n_pub ? pub_caught : priv_caught) += caught;
        }
        check(pub_caught == n_pub, name + ": every flipped public-output share is caught (" + std::to_string(pub_caught) +
                                       " of " + std::to_string(n_pub) + ")");
        check(priv_caught == n_open - n_pub, name + ": every flipped masked private-output share is caught (" +
                                                 std::to_string(priv_caught) + " of " + std::to_string(n_open - n_pub) + ")");
        total_pub += n_pub;
        total_priv += n_open - n_pub;

        // (c) a wrong input difference is sent while the sender computes with the right one:
        // the honest party aborts, or nothing opened depends on the wire and the outputs are
        // the honest ones.  A wrong output is never accepted.
        size_t in_n = 0, in_abort = 0, in_same = 0;
        for (int cheater = 0; cheater < 2; cheater++) {
            const size_t n_in = c->input_bits((uint32_t)cheater) * batch;
            for (size_t bit = 0; bit < n_in; bit += (n_in > 40 ? 3 : 1)) {
                const Verdict v = tamper(*c, all, batch, cheater, 0, flip_bit_rule(256 + bit));
                in_n++;
                if (v.honest_abort && !v.honest_output && v.honest_error == "BoolCheck failed") in_abort++;
                else if (outputs_match(*c, v.pair, want)) in_same++;
            }
        }
        check(in_abort + in_same == in_n && in_abort > 0,
              name + ": an inconsistent input difference aborts or changes nothing (" + std::to_string(in_abort) + " aborts, " +
                  std::to_string(in_same) + " unchanged, of " + std::to_string(in_n) + ")");
        std::printf("  %s batch %zu: %zu flipped d / e shares, %zu public and %zu private opening shares, all caught; "
                    "%zu inconsistent input differences: %zu aborts, %zu without effect\n",
                    name.c_str(), batch, n_flip, n_pub, n_open - n_pub, in_n, in_abort, in_same);

        // (d) the header, the commitments and the digests of the checks.
        for (int cheater = 0; cheater < 2; cheater++) {
            Verdict v = tamper(*c, all, batch, cheater, 0, flip_bit_rule(77));
            check(v.honest_abort && !v.honest_output && v.honest_error.find("header mismatch") != std::string::npos,
                  name + ": a wrong evaluation header aborts");
            for (size_t msg : {depth + 1, depth + 2, depth + 4, depth + 5}) {
                v = tamper(*c, all, batch, cheater, msg, flip_bit_rule(3));
                check(v.honest_abort && !v.honest_output, name + ": a rewritten BoolCheck message aborts (message " + std::to_string(msg) + ")");
            }
            // Malformed messages: wrong lengths and stray padding bits.
            v = tamper(*c, all, batch, cheater, 1, truncate_rule(1));
            check(v.honest_abort && !v.honest_output, name + ": a truncated AND-layer message aborts");
            v = tamper(*c, all, batch, cheater, 1, append_rule(1));
            check(v.honest_abort && !v.honest_output, name + ": an extended AND-layer message aborts");
            v = tamper(*c, all, batch, cheater, depth + 3, append_rule(1));
            check(v.honest_abort && !v.honest_output, name + ": an extended opening aborts");
            v = tamper(*c, all, batch, cheater, 0, truncate_rule(1));
            check(v.honest_abort && !v.honest_output, name + ": a truncated input message aborts");
            v = tamper(*c, all, batch, cheater, 0, [](const bytes&) { return bytes(5, 0); });
            check(v.honest_abort && !v.honest_output, name + ": an input message shorter than the header aborts");
            if (n_open % 8) {
                v = tamper(*c, all, batch, cheater, depth + 3, flip_bit_rule(n_open));
                check(v.honest_abort && !v.honest_output && v.honest_error.find("padding") != std::string::npos,
                      name + ": a stray padding bit in the opening aborts");
            }
        }
    }
    check(total_pub > 0 && total_priv > 0, "public and private output shares were both tampered with");
}

// ------------------------------------------------------------ strict mode, message limits
struct StrictOut {
    std::vector<uint64_t> log;
    abb::EvalStats stats;
    bool outputs_ok = false;
};

static void test_strict(circuit::CircuitStore& store) {
    Section sec("strict mode (triple budget) and message limits");
    CtrPrgRng rng(808);
    std::shared_ptr<const Circuit> c = store.get("rnd05"), cx = store.get("xoronly");
    const size_t batch = 3, need = c->n_and() * batch;
    const circuit::PlainInputs all = random_inputs(*c, rng, batch), allx = random_inputs(*cx, rng, 1);
    const circuit::PlainOutputs want = circuit::evaluate(*c, all, batch);
    {
        prep::IdealDealer dealer;
        const bytes sid = {'s', 't', 'r'};
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                prep::IdealPrep prep(role, dealer);
                abb::Engine eng(role, ch, prep, sid);
                auto eval = [&] { return eng.eval_circuit(*c, party_inputs(*c, all, role), abb::Bound(), batch); };
                StrictOut out;
                std::vector<uint64_t>& log = out.log;
                log.push_back(eng.allow_refill() && eng.triple_requests() == 0 && eng.triples_requested() == 0);  // default: refill mode
                eng.require_triples(1000000);  // refill mode: never refuses
                eng.set_allow_refill(false);
                // Nothing is reserved: refused before any message; the engine stays alive.
                size_t needed = 0, available = 99;
                bool refused = false;
                try {
                    eval();
                } catch (const abb::TripleBudgetError& e) {
                    refused = true;
                    needed = e.needed;
                    available = e.available;
                }
                log.push_back(refused && needed == need && available == 0 && !eng.dead() && !prep.aborted() && ch.stats().msgs_sent == 0 &&
                              eng.evaluations() == 0 && eng.triple_requests() == 0 && prep.counters().and_triples == 0);
                log.push_back(throws_invalid([&] { eval(); }) && !eng.dead());  // a TripleBudgetError is a std::invalid_argument
                // A circuit without AND gates needs no triples: strict mode does not concern it.
                eng.eval_circuit(*cx, party_inputs(*cx, allx, role), abb::Bound(), 1);
                log.push_back(eng.triple_requests());
                eng.reserve_triples(need + 5);  // explicit: one population
                const abb::EvalResult r = eval();
                out.stats = r.stats;
                bool same = true;
                for (const circuit::Decl& d : c->outputs)
                    if (d.owner == circuit::OUT_PUBLIC) same = same && r.pub.at(d.label) == want.at(d.label);
                out.outputs_ok = same;
                log.push_back(eng.triple_requests());
                log.push_back(eng.triples_requested());
                log.push_back(eng.triples_available());
                // Five triples are left: not enough for another evaluation.
                const uint64_t sent = ch.stats().msgs_sent;
                refused = false;
                try {
                    eval();
                } catch (const abb::TripleBudgetError& e) {
                    refused = e.needed == need && e.available == 5;
                }
                log.push_back(refused && ch.stats().msgs_sent == sent && eng.triples_available() == 5 && !eng.dead() && eng.evaluations() == 2);
                bool enough = true;
                try {
                    eng.require_triples(5);
                } catch (...) {
                    enough = false;
                }
                log.push_back(enough && throws_invalid([&] { eng.require_triples(6); }));
                eng.set_allow_refill(true);
                eval();  // refill mode: the missing need - 5 triples are one more population
                log.push_back(eng.triple_requests());
                log.push_back(eng.triples_requested());
                log.push_back(eng.triples_available());
                log.push_back(eng.last_stats().triple_requests * 1000 + eng.last_stats().triples_available);
                return out;
            };
        };
        const auto pr = run_two_party(party(0), party(1));
        check(pr.first.ok && pr.second.ok, "strict mode: no abort (" + pr.first.error + pr.second.error + ")");
        if (pr.first.ok && pr.second.ok) {
            const std::vector<uint64_t> expect = {1, 1, 1, 0, 1, need + 5, 5, 1, 1, 2, 2 * need, 0, 2000};
            check(need > 5 && pr.first.value.log == expect && pr.second.value.log == expect,
                  "strict mode refuses a short pool before any message and leaves the engine alive; refills are counted");
            check(pr.first.value.outputs_ok && pr.second.value.outputs_ok, "strict mode: an evaluation served from the reservation is correct");
            const abb::EvalStats& st = pr.first.value.stats;
            check(st.triple_requests == 1 && st.triples_requested == need + 5 && st.triples_available == 5 && st.triples == need &&
                      st.json().find("\"triple_requests\":1,\"triples_requested\":" + std::to_string(need + 5) + ",\"triples_available\":5") !=
                          std::string::npos,
                  "EvalStats reports the requests, the triples drawn and the triples left");
        }
    }

    // Message limits.  Every exchange of the engine names the exact length of the reply, so an
    // over-long message is refused by the channel with the error of the step it belongs to
    // (messages of a party with ideal preprocessing: 0 input, 1 .. depth AND layers, depth + 1
    // and depth + 2 first BoolCheck, depth + 3 opening, depth + 4 and depth + 5 second BoolCheck).
    {
        const size_t depth = c->and_depth();
        struct Long {
            size_t msg;
            const char* want;
            const char* name;
        };
        const Long longs[] = {
            {0, "evaluation header mismatch", "input message"},
            {1, "bit message has the wrong length", "first AND layer"},
            {depth, "bit message has the wrong length", "last AND layer"},
            {depth + 1, "committed_exchange: malformed commitment", "commitment of the first BoolCheck"},
            {depth + 2, "committed_exchange: opening has the wrong length", "opening of the first BoolCheck"},
            {depth + 3, "bit message has the wrong length", "output opening"},
            {depth + 4, "committed_exchange: malformed commitment", "commitment of the second BoolCheck"},
            {depth + 5, "committed_exchange: opening has the wrong length", "opening of the second BoolCheck"},
        };
        for (const Long& l : longs)
            for (int cheater = 0; cheater < 2; cheater++)
                for (size_t extra : {(size_t)1, (size_t)100000}) {
                    const Verdict v = tamper(*c, all, batch, cheater, l.msg, append_rule(extra));
                    check(v.honest_abort && !v.honest_output && v.honest_error.find(l.want) == 0,
                          std::string("an over-long ") + l.name + " (+" + std::to_string(extra) + " bytes, party " + std::to_string(cheater) +
                              ") is refused: " + v.honest_error);
                }
    }
}

// ------------------------------------------------------------ mismatching views
static void test_mismatch(circuit::CircuitStore& store) {
    Section sec("mismatch");
    CtrPrgRng rng(606);
    std::shared_ptr<const Circuit> c5 = store.get("rnd05"), c6 = store.get("rnd06"), ca = store.get("chain_a"), cb = store.get("chain_b");
    const bytes sid = {'m', 'm'};
    // Each party runs fn(role, engine); returns "ok", "abort:<what>" or "error:<what>".
    typedef std::function<void(int, abb::Engine&, Channel&)> Fn;
    auto run = [&](const Fn& fn) {
        prep::IdealDealer dealer;
        auto party = [&](int role) {
            return [&, role](Channel& ch) {
                prep::IdealPrep prep(role, dealer);
                abb::Engine eng(role, ch, prep, sid);
                try {
                    fn(role, eng, ch);
                } catch (const ProtocolAbort& e) {
                    return std::string("abort:") + e.what() + (eng.dead() ? "" : " (engine alive!)");
                } catch (const std::exception& e) {
                    return std::string("error:") + e.what() + (eng.dead() ? "" : " (engine alive!)");
                }
                return std::string("ok");
            };
        };
        return run_two_party(party(0), party(1));
    };
    auto header_abort = [](const PartyOutcome<std::string>& o) { return o.ok && o.value.find("abort:evaluation header mismatch") == 0; };
    auto failed = [](const PartyOutcome<std::string>& o) { return o.ok && o.value != "ok" && o.value.find("alive!") == std::string::npos; };

    // The ideal dealer pairs requests, so for these tests both circuits must draw the
    // same preprocessing: use one circuit and vary only what the header binds.
    {
        // Different batch sizes with the same number of triples and masks is impossible, so
        // the header test uses equal draws: the same circuit, different evaluation counters.
        const circuit::PlainInputs in = random_inputs(*c5, rng, 1);
        auto pr = run([&](int role, abb::Engine& eng, Channel&) {
            if (role == 0) {
                eng.eval_circuit(*c5, party_inputs(*c5, in, role), abb::Bound(), 1);
            } else {
                // P_1 has a different session identifier inside the header: emulate by a second engine.
                prep::Prep& prep = eng.prep();
                abb::Engine other(role, eng.channel(), prep, bytes{'x'});
                other.eval_circuit(*c5, party_inputs(*c5, in, role), abb::Bound(), 1);
            }
        });
        check(header_abort(pr.first), "different session identifiers: header mismatch at P_0");
        check(pr.second.ok && pr.second.value.find("abort:evaluation header mismatch") == 0, "different session identifiers: header mismatch at P_1");
    }
    {
        // Different circuits with different shapes: the ideal dealer itself refuses (the
        // parties ask for different correlations); with real preprocessing the header would.
        const circuit::PlainInputs i5 = random_inputs(*c5, rng, 1), i6 = random_inputs(*c6, rng, 1);
        auto pr = run([&](int role, abb::Engine& eng, Channel&) {
            if (role == 0) eng.eval_circuit(*c5, party_inputs(*c5, i5, role), abb::Bound(), 1);
            else eng.eval_circuit(*c6, party_inputs(*c6, i6, role), abb::Bound(), 1);
        });
        check(failed(pr.first) || failed(pr.second), "different circuits never produce outputs");
        check(!(pr.first.ok && pr.first.value == "ok") && !(pr.second.ok && pr.second.value == "ok"), "different circuits: both parties fail");
    }
    {
        // The same shapes but different bound wires: P_1 binds "keep" of a second evaluation.
        const circuit::PlainInputs ia = random_inputs(*ca, rng, 1), ib = random_inputs(*cb, rng, 1);
        auto pr = run([&](int role, abb::Engine& eng, Channel&) {
            const abb::EvalResult r1 = eng.eval_circuit(*ca, party_inputs(*ca, ia, role), abb::Bound(), 1);
            const abb::EvalResult r2 = eng.eval_circuit(*ca, party_inputs(*ca, ia, role), abb::Bound(), 1);
            abb::Bound bound;
            bound["keep"] = (role == 0 ? r1 : r2).retained.at("keep");
            eng.eval_circuit(*cb, party_inputs(*cb, ib, role), bound, 1);
        });
        check(header_abort(pr.first) && header_abort(pr.second), "different bound wires: header mismatch at both parties");
    }
    {
        // The evaluation counter is part of the header: P_1 is one evaluation behind.
        const circuit::PlainInputs in = random_inputs(*c5, rng, 1);
        auto pr = run([&](int role, abb::Engine& eng, Channel&) {
            eng.eval_circuit(*c5, party_inputs(*c5, in, role), abb::Bound(), 1);
            if (role == 0) {
                eng.eval_circuit(*c5, party_inputs(*c5, in, role), abb::Bound(), 1);
            } else {
                prep::Prep& prep = eng.prep();
                abb::Engine fresh(role, eng.channel(), prep, sid);  // counter 0 again
                fresh.eval_circuit(*c5, party_inputs(*c5, in, role), abb::Bound(), 1);
            }
        });
        check(header_abort(pr.first) && pr.second.ok && pr.second.value.find("header mismatch") != std::string::npos,
              "different evaluation counters: header mismatch");
    }
    {
        // A dead engine stays dead; the peer notices at the next message.
        const circuit::PlainInputs in = random_inputs(*c5, rng, 1);
        auto pr = run([&](int role, abb::Engine& eng, Channel&) {
            if (role == 1) eng.abort();
            eng.eval_circuit(*c5, party_inputs(*c5, in, role), abb::Bound(), 1);
        });
        check(pr.second.ok && pr.second.value == "abort:session aborted", "an aborted engine refuses to evaluate");
        check(pr.first.ok && pr.first.value != "ok", "the peer of an aborted engine gets no output");
    }
}

// ------------------------------------------------------------ argument errors
static void test_arguments(circuit::CircuitStore& store) {
    Section sec("arguments");
    CtrPrgRng rng(707);
    std::shared_ptr<const Circuit> c = store.get("rnd06"), ca = store.get("chain_a"), cb = store.get("chain_b");
    const circuit::PlainInputs all = random_inputs(*c, rng, 2);
    const circuit::PlainOutputs want = circuit::evaluate(*c, all, 2);
    const circuit::PlainInputs ia = random_inputs(*ca, rng, 2), ib = random_inputs(*cb, rng, 3), ib2 = random_inputs(*cb, rng, 2);
    prep::IdealDealer dealer;
    const bytes sid = {'a', 'r', 'g'};
    struct Out {
        int errors = 0, expected = 0;
        bool alive = false;
        abb::EvalResult res;
        uint64_t sent_before = 0;
    };
    auto party = [&](int role) {
        return [&, role](Channel& ch) {
            prep::IdealPrep prep(role, dealer);
            abb::Engine eng(role, ch, prep, sid);
            Out out;
            auto bad = [&](const std::function<void()>& f) {
                out.expected++;
                out.errors += throws_invalid(f);
            };
            const abb::Inputs good = party_inputs(*c, all, role);
            bad([&] { eng.eval_circuit(*c, good, abb::Bound(), 0); });
            abb::Inputs in = good;
            in.erase(in.begin());
            bad([&] { eng.eval_circuit(*c, in, abb::Bound(), 2); });
            in = good;
            in["nope"] = BitMat(3, 2);
            bad([&] { eng.eval_circuit(*c, in, abb::Bound(), 2); });
            in = good;
            in.begin()->second = BitMat(in.begin()->second.width + 1, 2);
            bad([&] { eng.eval_circuit(*c, in, abb::Bound(), 2); });
            in = good;
            in.begin()->second = BitMat(in.begin()->second.width, 3);
            bad([&] { eng.eval_circuit(*c, in, abb::Bound(), 2); });
            in = good;
            in.begin()->second.v[0] = 2;
            bad([&] { eng.eval_circuit(*c, in, abb::Bound(), 2); });
            abb::Bound stray;
            stray["keep"] = abb::WireShares();
            bad([&] { eng.eval_circuit(*c, good, stray, 2); });                              // unknown BOUND label
            bad([&] { eng.eval_circuit(*cb, party_inputs(*cb, ib, role), abb::Bound(), 3); });  // missing BOUND input
            bad([&] { abb::Engine wrong(1 - role, ch, prep, sid); });
            out.sent_before = ch.stats().msgs_sent;
            out.alive = !eng.dead();
            // A real evaluation, then bound-wire errors with real wires.
            out.res = eng.eval_circuit(*c, good, abb::Bound(), 2);
            const abb::EvalResult ra = eng.eval_circuit(*ca, party_inputs(*ca, ia, role), abb::Bound(), 2);
            const uint64_t sent = ch.stats().msgs_sent;
            abb::Bound bound;
            bound["keep"] = ra.retained.at("keep");
            bad([&] { eng.eval_circuit(*cb, party_inputs(*cb, ib, role), bound, 3); });      // batch 2 block in a batch-3 evaluation
            bound["keep"] = ra.retained.at("keep").slice(0, 23);
            bad([&] { eng.eval_circuit(*cb, party_inputs(*cb, ib2, role), bound, 2); });      // wrong width
            bound["keep"] = ra.retained.at("keep");
            bound["keep"].token += 1;
            bad([&] { eng.eval_circuit(*cb, party_inputs(*cb, ib2, role), bound, 2); });      // wires of another engine
            bound["keep"] = ra.retained.at("keep");
            bound["keep"].ids.pop_back();
            bad([&] { eng.eval_circuit(*cb, party_inputs(*cb, ib2, role), bound, 2); });      // inconsistent block
            out.alive = out.alive && !eng.dead() && ch.stats().msgs_sent == sent;
            return out;
        };
    };
    const std::pair<PartyOutcome<Out>, PartyOutcome<Out>> pr = run_two_party(party(0), party(1));
    check(pr.first.ok && pr.second.ok, "argument errors do not break the session (" + pr.first.error + pr.second.error + ")");
    if (!pr.first.ok || !pr.second.ok) return;
    for (const PartyOutcome<Out>* o : {&pr.first, &pr.second}) {
        check(o->value.errors == o->value.expected && o->value.expected == 13, "every invalid argument throws std::invalid_argument (" +
                                                                                    std::to_string(o->value.errors) + " of " + std::to_string(o->value.expected) + ")");
        check(o->value.sent_before == 0, "no message is sent for an invalid argument");
        check(o->value.alive, "the engine stays alive after argument errors");
    }
    bool ok = true;
    for (const circuit::Decl& d : c->outputs)
        if (d.owner == circuit::OUT_PUBLIC) ok = ok && pr.first.value.res.pub.at(d.label) == want.at(d.label) && pr.second.value.res.pub.at(d.label) == want.at(d.label);
    check(ok, "the evaluation after the argument errors is correct");

    // WireShares helpers.
    abb::WireShares a, b;
    a.width = 2;
    a.batch = 1;
    a.bits = {1, 0};
    a.macs.resize(2);
    a.ids = {5, 6};
    a.token = 9;
    b = a;
    b.ids = {7, 8};
    const abb::WireShares cat = abb::WireShares::concat({a, b});
    check(cat.width == 4 && cat.ids == std::vector<uint64_t>({5, 6, 7, 8}) && cat.slice(1, 3).ids == std::vector<uint64_t>({6, 7}),
          "WireShares concat and slice");
    b.token = 10;
    check(throws_invalid([&] { abb::WireShares::concat({a, b}); }), "WireShares of different engines cannot be concatenated");
}

int main(int argc, char** argv) {
    g_dir = circuit::dirname(argv[0]) + "/../tests/circuits";
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--quick") g_quick = true;
        else if (arg == "--circuits" && i + 1 < argc) g_dir = argv[++i];
        else {
            std::printf("usage: test_engine [--circuits DIR] [--quick]\n");
            return 2;
        }
    }
    try {
        circuit::CircuitStore store(g_dir);
        test_loader(store);
        test_plain(store);
        test_honest(store);
        test_chain(store);
        test_real(store);
        test_assignment(store);
        test_tamper(store);
        test_strict(store);
        test_mismatch(store);
        test_arguments(store);
    } catch (const std::exception& e) {
        std::printf("FATAL: %s\n", e.what());
        return 1;
    }
    std::printf("test_engine: %lld checks, %lld passed, %lld failed\n", g_pass + g_fail, g_pass, g_fail);
    return g_fail ? 1 : 0;
}
