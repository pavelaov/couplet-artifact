// couplet_item.cpp -- ONE party of a two-process run of Pi_Gen for one item of
// the Couplet layer (DESIGN_M3.md section 2; circuits_ppml.hpp): session setup
// with the real preprocessing, then ONE circuit execution with `batch`
// instances of the item (the batch dimension of the engine), as
// infer::InferParty runs every entry of its plan.  It measures the unit costs
// of the correlated randomness (experiment 6.1); the other party is a second
// process started with the same public arguments and the other --role.
//
//   couplet_item --role 0|1 [--host 127.0.0.1] [--port 47150] --item NAME
//                [--batch B] [--prep real|ideal] [--sigma 40] [--delay-ms D]
//                [--bandwidth-mbps R] [--bandwidth-bucket-bytes C]
//                [--threads T] [--verify] [--out FILE.json]
//                [--circuits DIR] [--timeout-ms MS] [--quiet]
//
//   --item NAME   a descriptor name of circuits_ppml.hpp: site_n<n>_f<f>_k<k>_
//                 s<s>_<round|exact>, triples<c>_k<k>_s<s>, mattriple_<r>x<c>_
//                 k<k>_s<s>, inmask_p<o>_c<c>_k<k>_s<s>, gate_n1_k<k>_s<s>_
//                 shared_om1 (output mask); "_sklansky" may follow.  Its
//                 circuit must have been exported (python/bench/
//                 export_circuits.py NAME).
//   --prep ideal  NOT a protocol (the public-seed dealer of couplet_party.cpp).
//   --delay-ms, --bandwidth-mbps, --bandwidth-bucket-bytes
//                 the emulated link of couplet_party (couplet/netem.hpp).
//   --verify      after the measured run, a TEST-ONLY step that is NOT part of
//                 the protocol and is not counted: both processes send each
//                 other their MAC key shares and their material in the clear
//                 and check every relation of the item: [[rho']] and its tag
//                 over Z_{2^m}; plane A (and plane B) on the whole domain on
//                 both components and against the plaintext Gen with the same
//                 roots (dpf_pair_ok of couplet_party.cpp) at rho_h (mode
//                 round) or at r' mod 2^n (mode exact), with [[rho_h]] and
//                 [[r']] over Z_{2^(n+s)}; c = a b for triples and c = A b for
//                 matrix triples with tags over Z_{2^w}; input masks equal to
//                 the owner's values; output masks multiples of 2^k.  As a
//                 negative control of the checker, the first instance with bit
//                 0 of P_0's first value share flipped must fail
//                 (verification.negative_control_rejected).
//
// All triples of the session (setup circuit and the item's circuit times the
// batch) are drawn in ONE preprocessing request and the session runs in strict
// mode, as in couplet_party.  The engine phases of the item carry the tag
// "gen-" (as in couplet_infer); the record folds them with the setup
// circuit's phases into the phases input, online and output of couplet_party,
// so that the campaign checks both drivers alike.  The channel, the ideal
// preprocessing and the record helpers are those of couplet_party.cpp, which
// is included here as a source file with its main() renamed.  Exit status 3 if
// the phases do not sum to the channel totals or the session was not a single,
// fully used bucketing population, 4 if the verification fails, 1 on any other
// error.
#define main couplet_party_main
#include "couplet_party.cpp"
#undef main

#include "couplet/circuits_ppml.hpp"
#include "couplet/planes.hpp"

namespace {

using ppml::SharePair;

struct ItemArgs {
    int role = -1;
    std::string host = "127.0.0.1";
    int port = 47150;
    std::string item, prep = "real", out, circuits;
    size_t batch = 1;
    int sigma = 40;
    unsigned threads = 1;
    double delay_ms = 0.0;
    double bandwidth_mbps = 0.0;  // 0: no bandwidth limit
    double bucket_bytes = DelayTcpChannel::DEFAULT_BUCKET_BYTES;
    bool verify = false, quiet = false;
    int timeout_ms = 120000;
};

void item_usage() {
    std::printf("usage: couplet_item --role 0|1 [--host H] [--port P] --item NAME [--batch B] [--prep real|ideal]\n"
                "                    [--sigma 40] [--delay-ms D] [--bandwidth-mbps R] [--bandwidth-bucket-bytes C]\n"
                "                    [--threads T] [--verify] [--out FILE.json] [--circuits DIR] [--timeout-ms MS] [--quiet]\n");
}

ItemArgs parse_item(int argc, char** argv) {
    ItemArgs a;
    for (int i = 1; i < argc; i++) {
        const std::string f = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + f);
            return argv[++i];
        };
        if (f == "--role") a.role = std::atoi(val().c_str());
        else if (f == "--host") a.host = val();
        else if (f == "--port") a.port = std::atoi(val().c_str());
        else if (f == "--item") a.item = val();
        else if (f == "--batch") a.batch = (size_t)std::atoll(val().c_str());
        else if (f == "--prep") a.prep = val();
        else if (f == "--sigma") a.sigma = std::atoi(val().c_str());
        else if (f == "--delay-ms") a.delay_ms = std::atof(val().c_str());
        else if (f == "--bandwidth-mbps") a.bandwidth_mbps = std::atof(val().c_str());
        else if (f == "--bandwidth-bucket-bytes") a.bucket_bytes = std::atof(val().c_str());
        else if (f == "--threads") a.threads = (unsigned)std::atoi(val().c_str());
        else if (f == "--verify") a.verify = true;
        else if (f == "--quiet") a.quiet = true;
        else if (f == "--out") a.out = val();
        else if (f == "--circuits") a.circuits = val();
        else if (f == "--timeout-ms") a.timeout_ms = std::atoi(val().c_str());
        else throw std::invalid_argument("unknown argument " + f);
    }
    if (a.role != 0 && a.role != 1) throw std::invalid_argument("--role must be 0 or 1");
    if (a.item.empty()) throw std::invalid_argument("--item is required");
    if (a.batch < 1 || a.batch > 65536) throw std::invalid_argument("--batch must be in 1..65536");
    if (a.prep != "real" && a.prep != "ideal") throw std::invalid_argument("--prep must be real or ideal");
    if (a.sigma < 1 || a.sigma > 128) throw std::invalid_argument("--sigma must be in 1..128");
    if (a.delay_ms < 0 || a.delay_ms > 10000) throw std::invalid_argument("--delay-ms must be in 0..10000");
    if (a.port < 1 || a.port > 65535) throw std::invalid_argument("--port must be in 1..65535");
    check_bandwidth_args(a.bandwidth_mbps, a.bucket_bytes);
    if (a.threads < 1) a.threads = 1;
    return a;
}

// ------------------------------------------------------------------ item names
bool take_prefix(std::string& s, const std::string& p) {
    if (s.compare(0, p.size(), p) != 0) return false;
    s.erase(0, p.size());
    return true;
}

// A decimal number at the front of s (at least one digit), removed from s.
bool take_number(std::string& s, size_t* out) {
    size_t i = 0;
    while (i < s.size() && i < 9 && s[i] >= '0' && s[i] <= '9') i++;
    if (i == 0 || (i < s.size() && s[i] >= '0' && s[i] <= '9')) return false;
    *out = (size_t)std::strtoull(s.substr(0, i).c_str(), nullptr, 10);
    s.erase(0, i);
    return true;
}

// The descriptor of a circuits_ppml name; the name must round-trip exactly.
ppml::Descriptor descriptor_from_name(const std::string& name) {
    std::string s = name, adder = "ripple";
    const std::string sk = "_sklansky";
    if (s.size() > sk.size() && s.compare(s.size() - sk.size(), sk.size(), sk) == 0) {
        adder = "sklansky";
        s.erase(s.size() - sk.size());
    }
    size_t a = 0, b = 0, k = 0, sb = 0;
    auto ks = [&](std::string& t) { return take_prefix(t, "_k") && take_number(t, &k) && take_prefix(t, "_s") && take_number(t, &sb); };
    ppml::Descriptor d;
    bool ok = false;
    std::string t = s;
    if (take_prefix(t, "site_n") && take_number(t, &a) && take_prefix(t, "_f") && take_number(t, &b) && ks(t) && take_prefix(t, "_")) {
        if (t == "round" || t == "exact") {
            d = ppml::Descriptor::site((unsigned)a, (unsigned)b, (unsigned)k, (unsigned)sb, t, adder);
            ok = true;
        }
    } else if ((t = s, take_prefix(t, "triples")) && take_number(t, &a) && ks(t) && t.empty()) {
        d = ppml::Descriptor::triples(a, (unsigned)k, (unsigned)sb, adder);
        ok = true;
    } else if ((t = s, take_prefix(t, "mattriple_")) && take_number(t, &a) && take_prefix(t, "x") && take_number(t, &b) && ks(t) &&
               t.empty()) {
        d = ppml::Descriptor::mattriple(a, b, (unsigned)k, (unsigned)sb, adder);
        ok = true;
    } else if ((t = s, take_prefix(t, "inmask_p")) && take_number(t, &a) && take_prefix(t, "_c") && take_number(t, &b) && ks(t) &&
               t.empty()) {
        d = ppml::Descriptor::inmask((int)a, b, (unsigned)k, (unsigned)sb, adder);
        ok = true;
    } else if ((t = s, take_prefix(t, "gate_n1")) && ks(t) && t == "_shared_om1") {
        d = ppml::Descriptor::outmask((unsigned)k, (unsigned)sb, adder);
        ok = true;
    }
    if (!ok || d.name() != name) throw std::invalid_argument("not a Couplet item name: " + name);
    return d;
}

// ------------------------------------------------------------------ material encoding (TEST-ONLY reveal)
void put_u128(bytes& out, u128 v) {
    put_u64(out, u128_lo(v));
    put_u64(out, u128_lo(v >> 64));
}
void put_pair(bytes& out, const SharePair& p) {
    put_u128(out, p.value);
    put_u128(out, p.tag);
}
u128 get_u128(Reader& r) {
    const uint64_t lo = r.u64();
    const uint64_t hi = r.u64();
    return (u128(hi) << 64) | u128(lo);
}
SharePair get_pair(Reader& r) {
    const u128 v = get_u128(r);
    const u128 t = get_u128(r);
    return SharePair(v, t);
}
void put_pairs(bytes& out, const std::vector<SharePair>& v) {
    put_u64(out, v.size());
    for (const SharePair& p : v) put_pair(out, p);
}
std::vector<SharePair> get_pairs(Reader& r, size_t expected) {
    const uint64_t n = r.u64();
    if (n != expected) throw std::runtime_error("verification message is malformed (vector length)");
    std::vector<SharePair> v((size_t)n);
    for (size_t i = 0; i < v.size(); i++) v[i] = get_pair(r);
    return v;
}

// One encoded object per instance (the order of ppml::Generated).
std::vector<bytes> encode_material(const ppml::Descriptor& d, const ppml::Generated& g) {
    std::vector<bytes> out;
    switch (d.kind) {
        case ppml::KIND_SITE:
            for (const ppml::SiteMaterial& m : g.sites) {
                bytes b;
                put_pair(b, m.rp);
                put_blob(b, m.plane_a.to_bytes());
                put_u64(b, m.has_b ? 1 : 0);
                if (m.has_b) {
                    put_blob(b, m.plane_b.to_bytes());
                    put_pair(b, m.rh);
                    put_pair(b, m.r2);
                }
                out.push_back(b);
            }
            break;
        case ppml::KIND_TRIPLES:
            for (const ppml::TripleMaterial& m : g.triples) {
                bytes b;
                put_pairs(b, m.a);
                put_pairs(b, m.b);
                put_pairs(b, m.c);
                out.push_back(b);
            }
            break;
        case ppml::KIND_MATTRIPLE:
            for (const ppml::MatTripleMaterial& m : g.mattriples) {
                bytes b;
                put_pairs(b, m.a);
                put_pairs(b, m.b);
                put_pairs(b, m.c);
                out.push_back(b);
            }
            break;
        case ppml::KIND_INMASK:
            for (const ppml::InMaskMaterial& m : g.inmasks) {
                bytes b;
                put_pairs(b, m.shares);
                put_u64(b, m.has_values ? m.values.size() : 0);
                if (m.has_values)
                    for (u128 v : m.values) put_u128(b, v);
                out.push_back(b);
            }
            break;
        case ppml::KIND_OUTMASK:
            for (const ppml::OutMaskMaterial& m : g.outmasks) {
                bytes b;
                put_pair(b, m.share);
                out.push_back(b);
            }
            break;
    }
    return out;
}

bool pair_ok(const SharePair& p0, const SharePair& p1, u128 want, u128 delta, unsigned bits) {
    const u128 mask = u128_mask(bits);
    const u128 v = (p0.value + p1.value) & mask;
    return v == (want & mask) && ((p0.tag + p1.tag) & mask) == ((delta * v) & mask);
}
u128 open_pair(const SharePair& p0, const SharePair& p1, unsigned bits) { return (p0.value + p1.value) & u128_mask(bits); }

// Every relation of one instance (see the header comment).
bool instance_ok(const ppml::Descriptor& d, const bytes& e0, const bytes& e1, u128 delta, unsigned threads, uint64_t* points) {
    Reader r0{e0, 0}, r1{e1, 0};
    const unsigned w = d.width();
    bool ok = true;
    switch (d.kind) {
        case ppml::KIND_SITE: {
            const SharePair rp0 = get_pair(r0), rp1 = get_pair(r1);
            const fss::DpfKey a0 = fss::DpfKey::from_bytes(r0.blob()), a1 = fss::DpfKey::from_bytes(r1.blob());
            const bool hb0 = r0.u64() != 0, hb1 = r1.u64() != 0;
            if (hb0 != d.two_openings() || hb1 != d.two_openings()) return false;
            const unsigned m = d.m(), ns = d.n + d.s;
            const u128 rp = open_pair(rp0, rp1, m);
            ok = ok && pair_ok(rp0, rp1, rp, delta, m);
            const uint64_t rho_h = u128_lo(rp >> d.f) & (((uint64_t)1 << d.n) - 1);
            const uint64_t rho_l = d.f ? (u128_lo(rp) & (((uint64_t)1 << d.f) - 1)) : 0;
            uint64_t idx = rho_h;
            if (d.two_openings()) {
                const fss::DpfKey b0 = fss::DpfKey::from_bytes(r0.blob()), b1 = fss::DpfKey::from_bytes(r1.blob());
                const SharePair rh0 = get_pair(r0), rh1 = get_pair(r1), q0 = get_pair(r0), q1 = get_pair(r1);
                ok = ok && b0.n == d.f && dpf_pair_ok(b0, b1, rho_l, 1, delta, threads, points);
                ok = ok && pair_ok(rh0, rh1, u128(rho_h), delta, ns);
                const u128 r2 = open_pair(q0, q1, ns);
                ok = ok && pair_ok(q0, q1, r2, delta, ns);
                idx = u128_lo(r2) & (((uint64_t)1 << d.n) - 1);
            }
            ok = ok && a0.n == d.n && dpf_pair_ok(a0, a1, idx, 1, delta, threads, points);
            break;
        }
        case ppml::KIND_TRIPLES:
        case ppml::KIND_MATTRIPLE: {
            const bool mat = d.kind == ppml::KIND_MATTRIPLE;
            const size_t na = mat ? d.rows * d.cols : d.count, nb = mat ? d.cols : d.count, nc = mat ? d.rows : d.count;
            const std::vector<SharePair> a0 = get_pairs(r0, na), b0 = get_pairs(r0, nb), c0 = get_pairs(r0, nc);
            const std::vector<SharePair> a1 = get_pairs(r1, na), b1 = get_pairs(r1, nb), c1 = get_pairs(r1, nc);
            std::vector<u128> av(na), bv(nb);
            for (size_t i = 0; i < na; i++) {
                av[i] = open_pair(a0[i], a1[i], w);
                ok = ok && pair_ok(a0[i], a1[i], av[i], delta, w);
            }
            for (size_t i = 0; i < nb; i++) {
                bv[i] = open_pair(b0[i], b1[i], w);
                ok = ok && pair_ok(b0[i], b1[i], bv[i], delta, w);
            }
            for (size_t i = 0; i < nc; i++) {
                u128 want = u128(0);
                if (mat)
                    for (size_t j = 0; j < d.cols; j++) want = want + av[i * d.cols + j] * bv[j];
                else
                    want = av[i] * bv[i];
                ok = ok && pair_ok(c0[i], c1[i], want, delta, w);
            }
            *points += nc;
            break;
        }
        case ppml::KIND_INMASK: {
            const std::vector<SharePair> s0 = get_pairs(r0, d.count), s1 = get_pairs(r1, d.count);
            const uint64_t nv0 = r0.u64();
            std::vector<u128> v0((size_t)std::min<uint64_t>(nv0, d.count));
            for (size_t i = 0; i < v0.size(); i++) v0[i] = get_u128(r0);
            const uint64_t nv1 = r1.u64();
            std::vector<u128> v1((size_t)std::min<uint64_t>(nv1, d.count));
            for (size_t i = 0; i < v1.size(); i++) v1[i] = get_u128(r1);
            const std::vector<u128>& vals = d.owner == 0 ? v0 : v1;
            ok = ok && (d.owner == 0 ? (nv0 == d.count && nv1 == 0) : (nv1 == d.count && nv0 == 0));
            for (size_t i = 0; ok && i < d.count; i++) ok = ok && pair_ok(s0[i], s1[i], vals[i], delta, w);
            *points += d.count;
            break;
        }
        case ppml::KIND_OUTMASK: {
            const SharePair p0 = get_pair(r0), p1 = get_pair(r1);
            const u128 v = open_pair(p0, p1, w);
            const u128 low = u128_mask(d.k);  // both components of both shares are multiples of 2^k
            ok = ok && ((p0.value | p1.value | p0.tag | p1.tag) & low) == u128(0) && pair_ok(p0, p1, v, delta, w);
            *points += 1;
            break;
        }
    }
    return ok && r0.pos == e0.size() && r1.pos == e1.size();
}

int run_item(const ItemArgs& a) {
    typedef std::chrono::steady_clock clock;
    const ppml::Descriptor desc = descriptor_from_name(a.item);

    // Circuits (not part of the measured time).
    const clock::time_point t_load = clock::now();
    circuit::CircuitStore store(a.circuits);
    const std::string setup_name = Session::setup_circuit_name(desc.s);
    const std::shared_ptr<const circuit::Circuit> setup_c = store.get(setup_name), gen_c = store.get(desc.name());
    const double load_s = secs(t_load, clock::now());
    const circuit::Stats st = gen_c->stats(), st_setup = setup_c->stats();
    if (gen_c->n_and() > ((size_t)1 << 40) / a.batch) throw std::invalid_argument("the batch is too large");
    const size_t reserve = gen_c->n_and() * a.batch;  // the triples of the item's execution
    const uint64_t n_triples = (uint64_t)st_setup.n_and + reserve;

    // Connection and an unmetered hello.
    const std::string params = "couplet-item/1|item=" + a.item + "|batch=" + std::to_string(a.batch) + "|prep=" + a.prep +
                               "|sigma=" + std::to_string(a.sigma) + "|delay_ms=" + num(a.delay_ms) +
                               bandwidth_hello(a.bandwidth_mbps, a.bucket_bytes) + "|verify=" + (a.verify ? "1" : "0") +
                               "|circuit=" + gen_c->digest_hex();
    std::unique_ptr<DelayTcpChannel> chp =
        DelayTcpChannel::open(a.role, a.host, (uint16_t)a.port, a.delay_ms, a.timeout_ms, a.bandwidth_mbps, a.bucket_bytes);
    DelayTcpChannel& ch = *chp;
    SystemRng sys;
    bytes nonce[2];
    nonce[a.role] = sys.get(16);
    {
        bytes hello(params.begin(), params.end());
        hello.push_back(0);
        hello.insert(hello.end(), nonce[a.role].begin(), nonce[a.role].end());
        const bytes theirs = ch.exchange(hello, 4096, "hello: message too long");
        if (theirs.size() != hello.size() || std::memcmp(theirs.data(), hello.data(), params.size() + 1) != 0)
            throw std::runtime_error("the two processes were started with different parameters (peer: " +
                                     std::string(theirs.begin(), theirs.begin() + (ptrdiff_t)std::min(theirs.size(), (size_t)300)) + ")");
        nonce[1 - a.role].assign(theirs.end() - 16, theirs.end());
    }
    const hashes::hash32 sid_hash = hashes::framed_sha256({"Couplet-item-sid", params, nonce[0], nonce[1]});
    const bytes sid(sid_hash.begin(), sid_hash.end());
    ch.reset_stats();

    // ---------------------------------------------------------------- the measured run
    const clock::time_point t0 = clock::now();
    std::unique_ptr<SharedSeedPrep> ideal;
    std::unique_ptr<Session> sesp;
    if (a.prep == "real") {
        sesp.reset(new Session(a.role, ch, sid, store, desc.k, desc.s, a.sigma));
        sesp->real_prep()->set_threads(a.threads);
    } else {
        ideal.reset(new SharedSeedPrep(a.role, 0x1dea1dea1dea1deaull));  // a public constant: NOT a protocol
        sesp.reset(new Session(a.role, ch, sid, store, *ideal, desc.k, desc.s));
    }
    Session& ses = *sesp;
    ses.set_allow_refill(false);  // strict mode
    ses.setup(reserve);           // all triples of the session in ONE preprocessing request
    const clock::time_point t1 = clock::now();
    ppml::Generated gen;
    {
        abb::Engine& engine = ses.engine();
        engine.set_phase_tag("gen-");
        struct Untag {
            abb::Engine& e;
            ~Untag() { e.set_phase_tag(""); }
        } untag{engine};
        gen = ppml::generate(ses, desc, a.batch);
    }
    const clock::time_point t2 = clock::now();
    // ---------------------------------------------------------------- end of the measured run

    const ChannelStats total = ch.stats();
    const double emulated_wait = ch.emulated_wait_s(), bandwidth_wait = ch.bandwidth_wait_s();
    Reveal mine;
    mine.outputs = encode_material(desc, gen);
    mine.delta = ses.delta_share();
    size_t out_bytes_all = 0;
    for (const bytes& o : mine.outputs) out_bytes_all += o.size();

    // Phases: the preprocessing books its own; the engine's are in the session book.
    prep::PrepStats pst;
    if (ses.real_prep()) pst = ses.real_prep()->stats();
    abb::PhaseBook& book = ses.book();
    const prep::PhaseStat ph_input = plus(book.get("setup-input"), book.get("gen-input"));
    const prep::PhaseStat ph_online = plus(book.get("setup-online"), book.get("gen-online"));
    const prep::PhaseStat ph_output =
        plus(plus(book.get("setup-outcheck"), book.get("setup-output")), plus(book.get("gen-outcheck"), book.get("gen-output")));
    const prep::PhaseStat ph_dealer = plus(book.get("setup-prep"), book.get("gen-prep"));  // ideal mode only: time, no traffic
    prep::PhaseStat sum = pst.total();
    sum += ph_input;
    sum += ph_online;
    sum += ph_output;
    sum += ph_dealer;
    const prep::PhaseStat book_total = book.total();
    const bool sums_ok = sum.bytes_sent == total.bytes_sent && sum.bytes_recv == total.bytes_recv && sum.rounds == total.rounds &&
                         sum.msgs_sent == total.msgs_sent && sum.msgs_recv == total.msgs_recv &&
                         pst.total().bytes_sent + book_total.bytes_sent == total.bytes_sent &&
                         pst.total().rounds + book_total.rounds == total.rounds && ph_dealer.bytes_sent == 0 && ph_dealer.rounds == 0;
    const unsigned bucket = ses.real_prep() ? ses.real_prep()->last_bucket_size() : 0;
    const bool single_population = ses.triple_requests() == 1 && ses.triples_requested() == n_triples;
    const bool all_used = ses.triples_used() == n_triples && ses.triples_unused() == 0;
    double bound_log2 = 0;
    const bool has_bound = ses.bucketing_bound_log2(&bound_log2);
    const double wall = secs(t0, t2);

    // ---------------------------------------------------------------- TEST-ONLY verification
    bool verified = false, correct = false, delta_big = false, control_rejected = false;
    uint64_t points = 0;
    double verify_s = 0;
    if (a.verify) {
        const clock::time_point tv = clock::now();
        const bytes mine_enc = mine.encode();
        const Reveal theirs = Reveal::decode(ch.exchange(mine_enc, 2 * mine_enc.size() + 65536, "verification message too long"));
        const Reveal& r0 = a.role == 0 ? mine : theirs;
        const Reveal& r1 = a.role == 0 ? theirs : mine;
        const u128 delta = fss::make_delta(r0.delta, r1.delta);  // the integer Delta_0 + Delta_1, never reduced
        delta_big = (delta >> desc.s) != u128(0);
        const unsigned vt = std::max(4u, a.threads);
        correct = r0.outputs.size() == a.batch && r1.outputs.size() == a.batch;
        for (size_t j = 0; correct && j < a.batch; j++) correct = instance_ok(desc, r0.outputs[j], r1.outputs[j], delta, vt, &points);
        // Negative control of the checker: the first instance with bit 0 of P_0's first value share flipped must fail.
        if (correct) {
            bytes bad = r0.outputs[0];
            const size_t at = desc.kind == ppml::KIND_SITE || desc.kind == ppml::KIND_OUTMASK ? 0 : 8;  // after a vector length
            bad[at] ^= 1;
            uint64_t ignored = 0;
            control_rejected = !instance_ok(desc, bad, r1.outputs[0], delta, vt, &ignored);
            correct = control_rejected;
        }
        verified = true;
        verify_s = secs(tv, clock::now());
    }

    // ---------------------------------------------------------------- the record
    std::string rec = "{\n";
    rec += " \"schema\": \"couplet.item_cpp/1\",\n";
    rec += " \"implementation\": \"C++ (couplet, native)\",\n";
    rec += " \"compiler\": " + q(compiler_name()) + ",\n";
    rec += " \"timestamp_utc\": " + q(utc_now()) + ",\n";
    rec += " \"role\": " + std::to_string(a.role) + ",\n";
    rec += " \"params\": {\"task\": \"item\", \"item\": " + q(a.item) + ", \"kind\": " + q(ppml::kind_name(desc.kind)) +
           ", \"descriptor\": " + desc.to_json() + ", \"n\": " + std::to_string(desc.n) + ", \"f\": " + std::to_string(desc.f) +
           ", \"k\": " + std::to_string(desc.k) + ", \"s\": " + std::to_string(desc.s) + ", \"batch\": " + std::to_string(a.batch) +
           ", \"prep\": " + q(a.prep) + ", \"sigma\": " + std::to_string(a.sigma) + ", \"delay_ms\": " + num(a.delay_ms) +
           netem_params_json(a.bandwidth_mbps, a.bucket_bytes) + ", \"threads\": " + std::to_string(a.threads) +
           ", \"mode\": \"two-process\", \"transport\": \"tcp\", \"host\": " + q(a.host) + ", \"port\": " + std::to_string(a.port) +
           "},\n";
    rec += std::string(" \"ideal_preprocessing\": ") + (a.prep == "ideal" ? "true" : "false") + ",\n";
    rec += " \"circuit\": {\"name\": " + q(desc.name()) + ", \"digest\": " + q(gen_c->digest_hex()) + ", \"binary_bytes\": " +
           std::to_string(gen_c->encoded_bytes) + ", \"stats\": " + st.json() + "},\n";
    rec += " \"session_setup_circuit\": {\"name\": " + q(setup_name) + ", \"digest\": " + q(setup_c->digest_hex()) + ", \"n_and\": " +
           std::to_string(st_setup.n_and) + ", \"and_depth\": " + std::to_string(st_setup.and_depth) + ", \"n_gates\": " +
           std::to_string(st_setup.n_gates) + "},\n";
    rec += " \"preprocessing\": {\"triples\": " + std::to_string(n_triples) + ", \"triples_generation_circuit\": " +
           std::to_string(reserve) + ", \"triples_session_setup\": " + std::to_string(st_setup.n_and) + ", \"triple_requests\": " +
           std::to_string(ses.triple_requests()) + ", \"triples_requested\": " + std::to_string(ses.triples_requested()) +
           ", \"allow_refill\": " + (ses.allow_refill() ? "true" : "false") + ", \"bucketing_bound_log2\": " +
           (has_bound ? num(bound_log2) : std::string("null")) + ", \"bucket_size\": " + std::to_string(bucket) + ", \"leaky_triples\": " +
           std::to_string(ses.prep().counters().leaky_triples) + ", \"model_bytes_per_and\": " + std::to_string(96 * bucket) + "},\n";
    rec += " \"phases\": {";
    for (int p = 0; p < prep::PH_COUNT; p++) rec += q(prep::phase_name(p)) + ": " + phase_json(pst.phase[p]) + ", ";
    rec += "\"input\": " + phase_json(ph_input) + ", \"online\": " + phase_json(ph_online) + ", \"output\": " + phase_json(ph_output);
    if (a.prep == "ideal") rec += ", \"ideal_dealer\": " + phase_json(ph_dealer);
    rec += "},\n";
    rec += " \"phases_note\": \"baseot .. check: preprocessing; input / online / output: engine, session-setup circuit and the "
           "item's circuit together (setup-* and gen-*); output = the two BoolChecks of Output and the opening; seconds are "
           "wall-clock at this party and include waiting for the peer\",\n";
    rec += " \"phases_detail\": {";
    for (size_t i = 0; i < book.size(); i++) rec += std::string(i ? ", " : "") + q(book.name(i)) + ": " + phase_json(book.stat(i));
    rec += "},\n";
    rec += " \"totals\": {\"bytes_sent\": " + std::to_string(total.bytes_sent) + ", \"bytes_recv\": " + std::to_string(total.bytes_recv) +
           ", \"bytes\": " + std::to_string(total.bytes_sent + total.bytes_recv) + ", \"msgs_sent\": " + std::to_string(total.msgs_sent) +
           ", \"rounds\": " + std::to_string(total.rounds) + ", \"seconds\": " + num(wall) + ", \"setup_s\": " + num(secs(t0, t1)) +
           ", \"generation_s\": " + num(secs(t1, t2)) + ", \"online_s\": " + num(ses.last_gen.stats.online_s) +
           ", \"phase_seconds_sum\": " + num(sum.seconds) + ", \"emulated_wait_s\": " + num(emulated_wait) +
           ", \"bandwidth_wait_s\": " + num(bandwidth_wait) + ", \"load_circuits_s\": " + num(load_s) + ", \"bytes_per_and_gate\": " +
           num((double)(total.bytes_sent + total.bytes_recv) / (double)n_triples) + ", \"bytes_per_instance\": " +
           num((double)(total.bytes_sent + total.bytes_recv) / (double)a.batch) + ", \"seconds_per_instance\": " +
           num(wall / (double)a.batch) + "},\n";
    rec += " \"totals_note\": \"bytes = payload bytes in both directions (sent + received by this party); seconds = this party's "
           "wall-clock time from the first protocol message to the certified outputs, both processes on one machine\",\n";
    rec += " \"generation\": " + ses.last_gen.stats.json() + ",\n";
    rec += " \"session_setup_evaluation\": " + ses.setup_stats().json() + ",\n";
    rec += " \"outputs\": {\"kind\": " + q(std::string(ppml::kind_name(desc.kind)) + " material") + ", \"instances\": " +
           std::to_string(a.batch) + ", \"bytes_per_instance_and_party\": " +
           std::to_string(mine.outputs.empty() ? 0 : mine.outputs[0].size()) + ", \"bytes_all_instances_per_party\": " +
           std::to_string(out_bytes_all) +
           ", \"encoding\": \"the reveal encoding of this driver: 32 bytes per sharing (value and tag as u128), keys as "
           "fss::DpfKey::to_bytes with an 8-byte length\"},\n";
    rec += std::string(" \"verification\": {\"performed\": ") + (verified ? "true" : "false") + ", \"outputs_correct\": " +
           (verified ? (correct ? "true" : "false") : "null") + ", \"instances_checked\": " + std::to_string(verified ? a.batch : 0) +
           ", \"points_checked\": " + std::to_string(points) + ", \"negative_control_rejected\": " +
           (verified ? (control_rejected ? "true" : "false") : "null") + ", \"delta_at_least_2_pow_s\": " +
           (verified ? (delta_big ? "true" : "false") : "null") + ", \"seconds\": " + num(verify_s) +
           ", \"note\": \"TEST ONLY, not part of the protocol and not counted in any total: after the measured run the two "
           "processes exchange their MAC key shares and their material in the clear and check every relation of the item with "
           "the unreduced Delta (planes on the whole domain on both components and against the plaintext Gen)\"},\n";
    rec += std::string(" \"checks\": {\"phase_sums_ok\": ") + (sums_ok ? "true" : "false") + ", \"single_bucketing_population\": " +
           (single_population ? "true" : "false") + ", \"all_triples_used\": " + (all_used ? "true" : "false") +
           ", \"strict_mode\": " + (ses.allow_refill() ? "false" : "true") + ", \"session_active\": " +
           (ses.active() ? "true" : "false") + "}\n";
    rec += "}\n";

    if (!a.out.empty()) {
        std::ofstream f(a.out, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + a.out);
        f << rec;
    }
    if (a.quiet) {
        std::printf("party %d: %s batch %zu: %llu bytes (both directions), %llu rounds, %.3f s%s\n", a.role, a.item.c_str(), a.batch,
                    (unsigned long long)(total.bytes_sent + total.bytes_recv), (unsigned long long)total.rounds, wall,
                    verified ? (correct ? ", outputs verified" : ", VERIFICATION FAILED") : "");
    } else {
        std::fputs(rec.c_str(), stdout);
    }
    std::fflush(stdout);
    if (!sums_ok) {
        std::fprintf(stderr, "error: the per-phase counters do not sum to the channel totals\n");
        return 3;
    }
    if (!single_population || !all_used) {
        std::fprintf(stderr, "error: the session did not consist of a single, fully used bucketing population\n");
        return 3;
    }
    if (verified && !correct) {
        std::fprintf(stderr, "error: the generated material is NOT correct\n");
        return 4;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    ItemArgs a;
    try {
        a = parse_item(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        item_usage();
        return 2;
    }
    if (a.circuits.empty()) a.circuits = circuit::dirname(argv[0]) + "/../../circuits";
    try {
        return run_item(a);
    } catch (const ProtocolAbort& e) {
        std::fprintf(stderr, "party %d: PROTOCOL ABORT: %s\n", a.role, e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "party %d: error: %s\n", a.role, e.what());
    }
    return 1;
}
