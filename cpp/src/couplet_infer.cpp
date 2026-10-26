// couplet_infer.cpp -- ONE party of a two-process inference run over TCP
// (DESIGN_M3.md sections 2 and 3; infer.hpp): manifest comparison, session
// setup with the preprocessing, generation of all correlated randomness of
// the plan, the online pipeline and the release of the logits.  The other
// party is a second process (this program, or the Python reference through
// tests/interop_peer.py --infer) started with the same public arguments and
// the other --role.  Party 0 listens, party 1 connects.
//
//   couplet_infer --role 0|1 [--host 127.0.0.1] [--port 47130]
//                 --model FILE [--mode round|exact] --input FILE [--input-index I]
//                 [--prep real|ideal] [--sigma 40] [--delay-ms D]
//                 [--bandwidth-mbps R] [--bandwidth-bucket-bytes C] [--threads T]
//                 [--chunk-and N] [--private-model] [--tile RxC] [--tile-limit L]
//                 [--verify] [--out FILE.json] [--circuits DIR] [--timeout-ms MS] [--quiet]
//
//   --model       a model file of python/couplet/tools/export_model.py (its
//                 activation table file must lie next to it unless the table
//                 is exact); --input a vectors file of the same tool: this
//                 party uses its slice (nn partition) of inputs[I].
//   --prep ideal  NOT a protocol: both processes derive the same trusted
//                 dealer from a fixed public seed (couplet_party.cpp); such
//                 records carry "ideal_preprocessing": true.
//   --chunk-and N at most N generation triples per preprocessing request
//                 (default: the whole plan in ONE request); the record gives
//                 the number of requests.
//   --private-model  the weights are P_0's private inputs; P_1 uses the
//                 shape only.  The circuits of the plan must have been exported.
//                 A model with more than L weights (--tile-limit, default
//                 4,096) gets a tiled plan (task R-TILE): every weight matrix in
//                 tiles of at most R x C (--tile, default 16x16), one matrix
//                 triple per tile, the input masks in instances of R * C; an
//                 untiled private plan above 4,096 weights is refused (the
//                 limit of the earlier versions, lifted only for tiled plans).
//                 With --tile or --tile-limit the hello ends with
//                 |tile=RxC|tile_limit=L; without them it is that of the
//                 earlier versions.
//   --delay-ms D, --bandwidth-mbps R, --bandwidth-bucket-bytes C
//                 the emulated link of couplet_party (DelayTcpChannel,
//                 couplet/netem.hpp): one-way latency D per message and a
//                 token-bucket limit of R * 10^6 bit/s per direction (0, the
//                 default: no limit).  No message changes; without R the hello
//                 is that of the earlier versions.
//   --threads T   threads of the preprocessing kernels, of the plane
//                 evaluation and of the table reads.
//   --verify      compare the logits with the expected ones (TEST ONLY, local,
//                 nothing is sent): mode exact against exact_logits[I] of the
//                 vectors file (bit-exact), mode round against the cleartext
//                 mirror on the thresholds recorded in this run.  Uses the
//                 whole input vector and the weights of the model file.
//
// The circuits are loaded from artifact/circuits/ (python/bench/
// export_circuits.py --plan MODEL writes the plan's circuits); loading is not
// part of the measured time.  Before the protocol the two processes exchange
// an unmetered hello: the public parameters
//     couplet-infer/1|mode=M|private=P|prep=R|sigma=S|k=K|s=S|chunk_and=N|delay_ms=D|input=I
// (with a bandwidth limit, |bandwidth_mbps=R|bucket_bytes=C follows delay_ms)
// and a 16-byte nonce each; the session identifier is
// SHA-256(frame("Couplet-infer-session", params, nonce_0, nonce_1)).
//
// The process prints a JSON record and writes it to --out: parameters, model
// and manifest summaries, preprocessing counters, bytes / rounds / seconds per
// accounting phase and per phase group (base OT, OT extension, authenticated
// bits, leaky ANDs, bucketing, triple checks, circuit input / online / output,
// manifest, plane evaluation, local online work, layer openings, final
// checks), totals, the logits and the verification.  Exit status 3 if the
// phases do not sum to the channel totals or the triples were not used up
// exactly (or, without --chunk-and, not drawn in one request), 4 if the
// verification fails, 1 on any other error.
#define main couplet_party_main
#include "couplet_party.cpp"
#undef main

#include "couplet/infer.hpp"
#include "couplet/nn.hpp"

namespace {

struct InferArgs {
    int role = -1;
    std::string host = "127.0.0.1";
    int port = 47130;
    std::string model, input, mode = "round", prep = "real", out, circuits;
    size_t input_index = 0, chunk_and = 0;
    unsigned k = 64, s = 64, threads = 1;
    int sigma = 40;
    double delay_ms = 0.0;
    double bandwidth_mbps = 0.0;  // 0: no bandwidth limit
    double bucket_bytes = DelayTcpChannel::DEFAULT_BUCKET_BYTES;
    bool private_model = false, verify = false, quiet = false;
    int timeout_ms = 120000;
    nn::Tiling tiling;          // task R-TILE
    bool tiling_given = false;  // --tile or --tile-limit on the command line
};

// The largest private model with an untiled plan (the limit of the earlier versions).
const size_t UNTILED_PRIVATE_LIMIT = 4096;

nn::Tiling parse_tile(const std::string& v, size_t limit) {
    const size_t x = v.find('x');
    if (x == std::string::npos || x == 0 || x + 1 >= v.size() || v.find_first_not_of("0123456789x") != std::string::npos ||
        v.find('x', x + 1) != std::string::npos)
        throw std::invalid_argument("--tile must be RxC with positive integers R and C");
    const long long r = std::atoll(v.substr(0, x).c_str()), c = std::atoll(v.substr(x + 1).c_str());
    if (r < 1 || c < 1 || r > 4096 || c > 4096) throw std::invalid_argument("--tile must be RxC with R and C in 1..4096");
    return nn::Tiling((size_t)r, (size_t)c, limit);
}

void infer_usage() {
    std::printf("usage: couplet_infer --role 0|1 [--host H] [--port P] --model FILE [--mode round|exact] --input FILE\n"
                "                     [--input-index I] [--prep real|ideal] [--sigma 40] [--delay-ms D]\n"
                "                     [--bandwidth-mbps R] [--bandwidth-bucket-bytes C] [--threads T]\n"
                "                     [--chunk-and N] [--private-model] [--tile RxC] [--tile-limit L] [--verify]\n"
                "                     [--out FILE.json] [--circuits DIR]\n"
                "                     [--timeout-ms MS] [--quiet]\n");
}

InferArgs parse_infer(int argc, char** argv) {
    InferArgs a;
    for (int i = 1; i < argc; i++) {
        const std::string f = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + f);
            return argv[++i];
        };
        if (f == "--role") a.role = std::atoi(val().c_str());
        else if (f == "--host") a.host = val();
        else if (f == "--port") a.port = std::atoi(val().c_str());
        else if (f == "--model") a.model = val();
        else if (f == "--mode") a.mode = val();
        else if (f == "--input") a.input = val();
        else if (f == "--input-index") a.input_index = (size_t)std::atoll(val().c_str());
        else if (f == "--prep") a.prep = val();
        else if (f == "--sigma") a.sigma = std::atoi(val().c_str());
        else if (f == "--delay-ms") a.delay_ms = std::atof(val().c_str());
        else if (f == "--bandwidth-mbps") a.bandwidth_mbps = std::atof(val().c_str());
        else if (f == "--bandwidth-bucket-bytes") a.bucket_bytes = std::atof(val().c_str());
        else if (f == "--threads") a.threads = (unsigned)std::atoi(val().c_str());
        else if (f == "--chunk-and") a.chunk_and = (size_t)std::atoll(val().c_str());
        else if (f == "--k") a.k = (unsigned)std::atoi(val().c_str());
        else if (f == "--s") a.s = (unsigned)std::atoi(val().c_str());
        else if (f == "--private-model") a.private_model = true;
        else if (f == "--tile") {
            a.tiling = parse_tile(val(), a.tiling.limit);
            a.tiling_given = true;
        } else if (f == "--tile-limit") {
            const std::string v = val();
            if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos || v.size() > 12)
                throw std::invalid_argument("--tile-limit must be a non-negative integer");
            a.tiling.limit = (size_t)std::atoll(v.c_str());
            a.tiling_given = true;
        }
        else if (f == "--verify") a.verify = true;
        else if (f == "--quiet") a.quiet = true;
        else if (f == "--out") a.out = val();
        else if (f == "--circuits") a.circuits = val();
        else if (f == "--timeout-ms") a.timeout_ms = std::atoi(val().c_str());
        else throw std::invalid_argument("unknown argument " + f);
    }
    if (a.role != 0 && a.role != 1) throw std::invalid_argument("--role must be 0 or 1");
    if (a.model.empty() || a.input.empty()) throw std::invalid_argument("--model and --input are required");
    if (a.mode != "round" && a.mode != "exact") throw std::invalid_argument("--mode must be round or exact");
    if (a.prep != "real" && a.prep != "ideal") throw std::invalid_argument("--prep must be real or ideal");
    if (a.sigma < 1 || a.sigma > 128) throw std::invalid_argument("--sigma must be in 1..128");
    if (a.delay_ms < 0 || a.delay_ms > 10000) throw std::invalid_argument("--delay-ms must be in 0..10000");
    if (a.port < 1 || a.port > 65535) throw std::invalid_argument("--port must be in 1..65535");
    check_bandwidth_args(a.bandwidth_mbps, a.bucket_bytes);
    if (a.k < 1 || a.k > 64 || a.s < 1 || a.s > 64) throw std::invalid_argument("--k and --s must be in 1..64");
    if (a.threads < 1) a.threads = 1;
    return a;
}

template <class T>
std::string list_json(const std::vector<T>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
}

std::string matrix_json(const std::vector<std::vector<uint64_t>>& m) {
    std::string s = "[";
    for (size_t i = 0; i < m.size(); i++) s += (i ? ", " : "") + list_json(m[i]);
    return s + "]";
}

int run_infer(const InferArgs& a) {
    typedef std::chrono::steady_clock clock;
    // Model, input and circuits (not part of the measured time).
    const clock::time_point t_load = clock::now();
    const nn::Model model = nn::Model::load(a.model);
    const nn::Vectors vec = nn::Vectors::load(a.input);
    if (a.input_index >= vec.inputs.size()) throw std::invalid_argument("--input-index is out of range");
    const std::vector<int64_t>& x_all = vec.inputs[a.input_index];
    if (x_all.size() != model.d()) throw std::invalid_argument("the input does not match the model");
    const bool tiled = nn::plan_tiled(model, a.private_model, a.tiling);
    if (a.private_model && !tiled && model.n_weights() > UNTILED_PRIVATE_LIMIT)
        throw std::invalid_argument("--private-model without tiling is limited to 4,096 weights (this plan is not tiled: "
                                    "--tile-limit is at least the model's weights)");
    const std::pair<size_t, size_t> rng_feat = model.feature_range(a.role);
    const std::vector<int64_t> x_mine(x_all.begin() + (ptrdiff_t)rng_feat.first, x_all.begin() + (ptrdiff_t)rng_feat.second);
    const nn::Model shape = model.shape_only();
    const nn::Model& own = (a.private_model && a.role == 1) ? shape : model;
    circuit::CircuitStore store(a.circuits);
    const ppml::Manifest plan = nn::plan_manifest(own, a.mode, a.k, a.s, a.private_model, a.tiling);
    const size_t budget = plan.and_budget(store);  // loads every circuit of the plan
    const std::shared_ptr<const circuit::Circuit> setup_c = store.get(Session::setup_circuit_name(a.s));
    const double load_s = secs(t_load, clock::now());

    // Connection and an unmetered hello.
    const std::string params = "couplet-infer/1|mode=" + a.mode + "|private=" + (a.private_model ? "1" : "0") + "|prep=" + a.prep +
                               "|sigma=" + std::to_string(a.sigma) + "|k=" + std::to_string(a.k) + "|s=" + std::to_string(a.s) +
                               "|chunk_and=" + std::to_string(a.chunk_and) + "|delay_ms=" + num(a.delay_ms) +
                               bandwidth_hello(a.bandwidth_mbps, a.bucket_bytes) + "|input=" + std::to_string(a.input_index) +
                               (a.tiling_given ? "|tile=" + std::to_string(a.tiling.rows) + "x" + std::to_string(a.tiling.cols) +
                                                     "|tile_limit=" + std::to_string(a.tiling.limit)
                                               : std::string());
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
    const hashes::hash32 sid_hash = hashes::framed_sha256({"Couplet-infer-session", params, nonce[0], nonce[1]});
    const bytes sid(sid_hash.begin(), sid_hash.end());
    ch.reset_stats();

    // ---------------------------------------------------------------- the measured run
    const clock::time_point t0 = clock::now();
    std::unique_ptr<SharedSeedPrep> ideal;
    std::unique_ptr<Session> sesp;
    if (a.prep == "real") {
        sesp.reset(new Session(a.role, ch, sid, store, a.k, a.s, a.sigma));
        sesp->real_prep()->set_threads(a.threads);
    } else {
        ideal.reset(new SharedSeedPrep(a.role, 0x1dea1dea1dea1deaull));  // a public constant: NOT a protocol
        sesp.reset(new Session(a.role, ch, sid, store, *ideal, a.k, a.s));
    }
    Session& ses = *sesp;
    infer::Config cfg;
    cfg.mode = a.mode;
    cfg.private_model = a.private_model;
    cfg.triple_cap = a.chunk_and;
    cfg.threads = a.threads;
    cfg.tiling = a.tiling;
    infer::InferParty party(a.role, ses, own, cfg);
    party.setup();
    const clock::time_point t1 = clock::now();
    party.preprocess();
    const clock::time_point t2 = clock::now();
    const std::vector<int64_t> logits = party.infer(x_mine);
    const clock::time_point t3 = clock::now();
    // ---------------------------------------------------------------- end of the measured run

    const ChannelStats total = ch.stats();
    const double wall = secs(t0, t3);
    const infer::Record& rec = party.record();

    // Accounting: the preprocessing books its own phases, the rest is in the session book.
    prep::PrepStats pst;
    if (ses.real_prep()) pst = ses.real_prep()->stats();
    abb::PhaseBook& book = ses.book();
    auto get = [&](const char* name) { return book.get(name); };
    struct Group {
        const char* name;
        prep::PhaseStat st;
    };
    std::vector<Group> groups = {
        {"base_ot", pst.phase[prep::PH_BASEOT]},
        {"ot_extension", pst.phase[prep::PH_COTEXT]},
        {"authenticated_bits", pst.phase[prep::PH_ABIT]},
        {"leaky_ands", pst.phase[prep::PH_LAAND]},
        {"bucketing", pst.phase[prep::PH_BUCKET]},
        {"triple_checks", pst.phase[prep::PH_CHECK]},
        {"circuit_input", plus(get("setup-input"), get("gen-input"))},
        {"circuit_online", plus(get("setup-online"), get("gen-online"))},
        {"circuit_output", plus(plus(get("setup-outcheck"), get("setup-output")), plus(get("gen-outcheck"), get("gen-output")))},
        {"ideal_dealer", plus(get("setup-prep"), get("gen-prep"))},
        {"manifest", get("manifest")},
        {"plane_evaluation", get(truncmask::PHASE_PLANES)},
        {"online_local", get(infer::PHASE_ONLINE)},
        {"layer_openings", plus(plus(get(truncmask::PHASE_SHARED), get(beaver::PHASE_INPUT)), get("arith-open"))},
        {"final_check", get("arith-check")},
    };
    prep::PhaseStat sum;
    for (const Group& g : groups) sum += g.st;
    const prep::PhaseStat book_total = book.total();
    const bool sums_ok = sum.bytes_sent == total.bytes_sent && sum.bytes_recv == total.bytes_recv && sum.rounds == total.rounds &&
                         sum.msgs_sent == total.msgs_sent && sum.msgs_recv == total.msgs_recv &&
                         pst.total().bytes_sent + book_total.bytes_sent == total.bytes_sent &&
                         pst.total().rounds + book_total.rounds == total.rounds;
    prep::PhaseStat online;
    for (const char* nm : {infer::PHASE_ONLINE, truncmask::PHASE_PLANES, truncmask::PHASE_SHARED, beaver::PHASE_INPUT, "arith-open", "arith-check"})
        online += book.get(nm);
    const size_t n_triples = setup_c->n_and() + budget;
    const bool all_used = ses.triples_used() == n_triples && ses.triples_unused() == 0;
    const bool single_population = a.chunk_and ? true : (ses.triple_requests() == 1 && ses.triples_requested() == n_triples);
    const unsigned bucket = ses.real_prep() ? ses.real_prep()->last_bucket_size() : 0;
    double bound_log2 = 0;
    const bool has_bound = ses.bucketing_bound_log2(&bound_log2);

    // ---------------------------------------------------------------- TEST-ONLY verification (local)
    bool verified = false, correct = false;
    std::vector<int64_t> expected;
    if (a.verify) {
        if (a.mode == "exact") expected = vec.exact_logits[a.input_index];
        else expected = nn::to_i64(model.mirror(x_all, false, &rec.thresholds).logits);
        const std::vector<int64_t> exact_mirror = nn::to_i64(model.mirror(x_all, a.mode == "exact", &rec.thresholds).logits);
        correct = logits == expected && exact_mirror == expected;
        verified = true;
    }

    // ---------------------------------------------------------------- the record
    std::string rec_s = "{\n";
    rec_s += " \"schema\": \"couplet.infer_cpp/1\",\n";
    rec_s += " \"implementation\": \"C++ (couplet, native)\",\n";
    rec_s += " \"compiler\": " + q(compiler_name()) + ",\n";
    rec_s += " \"timestamp_utc\": " + q(utc_now()) + ",\n";
    rec_s += " \"role\": " + std::to_string(a.role) + ",\n";
    std::string sizes = "[" + std::to_string(model.d());
    for (const nn::Shape& sh : model.shapes) sizes += ", " + std::to_string(sh.first);
    sizes += "]";
    std::string model_name = a.model.substr(a.model.find_last_of("/\\") == std::string::npos ? 0 : a.model.find_last_of("/\\") + 1);
    rec_s += " \"params\": {\"model\": " + q(model_name) + ", \"kind\": " + q(model.kind) + ", \"act\": " + q(model.act) +
             ", \"sizes\": " + sizes + ", \"n\": " + std::to_string(model.n) + ", \"f\": " + std::to_string(model.f) +
             ", \"mode\": " + q(a.mode) + ", \"private_model\": " + (a.private_model ? "true" : "false") + ", \"k\": " +
             std::to_string(a.k) + ", \"s\": " + std::to_string(a.s) + ", \"sigma\": " + std::to_string(a.sigma) +
             ", \"prep\": " + q(a.prep) + ", \"chunk_and\": " + std::to_string(a.chunk_and) + ", \"threads\": " +
             std::to_string(a.threads) + ", \"delay_ms\": " + num(a.delay_ms) +
             netem_params_json(a.bandwidth_mbps, a.bucket_bytes) + ", \"input_index\": " + std::to_string(a.input_index) +
             ", \"run_mode\": \"two-process\", \"transport\": \"tcp\", \"host\": " + q(a.host) + ", \"port\": " + std::to_string(a.port) +
             ", \"tile\": " + q(std::to_string(a.tiling.rows) + "x" + std::to_string(a.tiling.cols)) + ", \"tile_limit\": " +
             std::to_string(a.tiling.limit) + "},\n";
    rec_s += std::string(" \"ideal_preprocessing\": ") + (a.prep == "ideal" ? "true" : "false") + ",\n";
    rec_s += " \"model\": {\"model_hash\": " + q(model.weights_known ? ppml::hex(model.model_hash()) : std::string("")) +
             ", \"architecture_hash\": " + q(ppml::hex(model.architecture_hash())) + ", \"table_hash\": " + q(ppml::hex(model.table_hash())) +
             ", \"sites\": " + std::to_string(model.n_sites()) + ", \"layers\": " + std::to_string(model.n_layers()) +
             ", \"weights\": " + std::to_string(model.n_weights()) + ", \"partition\": " + std::to_string(model.partition()) + "},\n";
    size_t and_by_kind[5] = {0, 0, 0, 0, 0};  // site, triples, mattriple, inmask, outmask
    for (const ppml::ManifestEntry& e : plan.entries) and_by_kind[(int)e.desc.kind] += ppml::and_count(store, e.desc) * e.batch;
    rec_s += " \"manifest\": {\"digest\": " + q(rec.manifest_digest) + ", \"executions_planned\": " + std::to_string(plan.entries.size()) +
             ", \"and_gates\": " + std::to_string(budget) + ", \"and_gates_by_kind\": {\"site\": " + std::to_string(and_by_kind[ppml::KIND_SITE]) +
             ", \"mattriple\": " + std::to_string(and_by_kind[ppml::KIND_MATTRIPLE]) + ", \"inmask\": " +
             std::to_string(and_by_kind[ppml::KIND_INMASK]) + ", \"outmask\": " + std::to_string(and_by_kind[ppml::KIND_OUTMASK]) +
             "}, \"tiled\": " + (rec.tiled ? "true" : "false") + ", \"tiles\": " + std::to_string(rec.tiles) +
             ", \"tiles_per_layer\": " + list_json(rec.tiles_per_layer) + ", \"sites\": " + std::to_string(plan.site_count()) + ", \"entries\": [";
    for (size_t i = 0; i < plan.entries.size(); i++)
        rec_s += std::string(i ? ", " : "") + "{\"use\": " + q(plan.entries[i].use) + ", \"name\": " + q(plan.entries[i].desc.name()) +
                 ", \"batch\": " + std::to_string(plan.entries[i].batch) + ", \"n_and\": " +
                 std::to_string(ppml::and_count(store, plan.entries[i].desc)) + "}";
    rec_s += "]},\n";
    rec_s += " \"preprocessing\": {\"and_gates\": " + std::to_string(budget) + ", \"triples_session_setup\": " +
             std::to_string(setup_c->n_and()) + ", \"triples\": " + std::to_string(n_triples) + ", \"triple_requests\": " +
             std::to_string(ses.triple_requests()) + ", \"triples_requested\": " + std::to_string(ses.triples_requested()) +
             ", \"triples_unused\": " + std::to_string(ses.triples_unused()) + ", \"executions\": " + std::to_string(rec.executions) +
             ", \"allow_refill\": " + (ses.allow_refill() ? "true" : "false") + ", \"bucketing_bound_log2\": " +
             (has_bound ? num(bound_log2) : std::string("null")) + ", \"bucket_size\": " + std::to_string(bucket) +
             ", \"leaky_triples\": " + std::to_string(ses.prep().counters().leaky_triples) + "},\n";
    rec_s += " \"executions\": [";
    for (size_t i = 0; i < rec.execution_list.size(); i++)
        rec_s += std::string(i ? ", " : "") + "{\"name\": " + q(rec.execution_list[i].name) + ", \"batch\": " +
                 std::to_string(rec.execution_list[i].batch) + ", \"n_and\": " + std::to_string(rec.execution_list[i].n_and) +
                 ", \"seconds\": " + num(rec.execution_list[i].seconds) + "}";
    rec_s += "],\n";
    rec_s += " \"online\": {\"openings\": " + std::to_string(rec.openings) + ", \"openings_before_release\": " +
             std::to_string(rec.openings_before_release) + ", \"checks\": " + std::to_string(rec.checks) + ", \"shared_openings\": " +
             std::to_string(rec.shared_openings) + ", \"release_masked\": " + (a.private_model ? "true" : "false") +
             ", \"bytes_sent\": " + std::to_string(online.bytes_sent) + ", \"bytes_recv\": " + std::to_string(online.bytes_recv) +
             ", \"rounds\": " + std::to_string(online.rounds) + ", \"seconds\": " + num(rec.infer_s) + ", \"plane_eval_s\": " +
             num(rec.plane_eval_s) + ", \"table_read_s\": " + num(rec.table_s) + ", \"thresholds\": " + matrix_json(rec.thresholds) +
             ", \"x_hat\": " + matrix_json(rec.x_hat) + "},\n";
    rec_s += " \"phases\": {";
    for (int p = 0; p < prep::PH_COUNT; p++) rec_s += q(prep::phase_name(p)) + ": " + phase_json(pst.phase[p]) + ", ";
    for (size_t i = 0; i < book.size(); i++) rec_s += std::string(i ? ", " : "") + q(book.name(i)) + ": " + phase_json(book.stat(i));
    rec_s += "},\n";
    rec_s += " \"phase_groups\": {";
    for (size_t i = 0; i < groups.size(); i++) rec_s += std::string(i ? ", " : "") + q(groups[i].name) + ": " + phase_json(groups[i].st);
    rec_s += "},\n";
    rec_s += " \"phase_groups_note\": \"base_ot .. triple_checks: the real preprocessing (RealPrep); circuit_*: the engine for the "
             "session-setup circuit and every generation circuit; ideal_dealer: time in the trusted dealer (ideal mode only, no "
             "traffic); layer_openings: commit-then-open of the first layer, the authenticated input exchange and every opening of "
             "the arithmetic checker, the release included; final_check: every ArithCheck (all of them run at the release); bytes "
             "and rounds of the groups sum to the channel totals; seconds are wall-clock at this party and include waiting\",\n";
    rec_s += " \"totals\": {\"bytes_sent\": " + std::to_string(total.bytes_sent) + ", \"bytes_recv\": " + std::to_string(total.bytes_recv) +
             ", \"bytes\": " + std::to_string(total.bytes_sent + total.bytes_recv) + ", \"msgs_sent\": " + std::to_string(total.msgs_sent) +
             ", \"rounds\": " + std::to_string(total.rounds) + ", \"seconds\": " + num(wall) + ", \"setup_s\": " + num(secs(t0, t1)) +
             ", \"preprocess_s\": " + num(secs(t1, t2)) + ", \"infer_s\": " + num(secs(t2, t3)) + ", \"emulated_wait_s\": " +
             num(ch.emulated_wait_s()) + ", \"bandwidth_wait_s\": " + num(ch.bandwidth_wait_s()) + ", \"load_s\": " + num(load_s) +
             ", \"phase_seconds_sum\": " + num(sum.seconds) +
             ", \"bytes_per_site\": " + num((double)(total.bytes_sent + total.bytes_recv) / (double)std::max<size_t>(1, model.n_sites())) +
             "},\n";
    rec_s += " \"totals_note\": \"bytes = payload bytes sent plus received by this party (= the bytes of both directions); rounds and "
             "seconds of this party from the manifest exchange to the released logits, both processes on one machine\",\n";
    std::string lg = "[", lr = "[";
    for (size_t i = 0; i < logits.size(); i++) {
        lg += (i ? ", " : "") + std::to_string(logits[i]);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.9g", (double)logits[i] / std::ldexp(1.0, 2 * (int)model.f));
        lr += std::string(i ? ", " : "") + buf;
    }
    rec_s += " \"logits\": " + lg + "],\n \"logits_real\": " + lr + "],\n";
    rec_s += " \"released\": [";
    for (size_t i = 0; i < rec.released.size(); i++) rec_s += std::string(i ? ", " : "") + q(u128_to_hex(rec.released[i]));
    rec_s += "],\n";
    rec_s += std::string(" \"verification\": {\"performed\": ") + (verified ? "true" : "false") + ", \"logits_correct\": " +
             (verified ? (correct ? "true" : "false") : "null") + ", \"expected\": " + (verified ? list_json(expected) : std::string("null")) +
             ", \"rule\": " + q(a.mode == "exact" ? "exact_logits of the vectors file (bit-exact)" : "the cleartext mirror on the recorded thresholds") +
             ", \"note\": \"TEST ONLY and local: uses the whole input vector and the weights of the model file\"},\n";
    rec_s += std::string(" \"checks\": {\"phase_sums_ok\": ") + (sums_ok ? "true" : "false") + ", \"single_bucketing_population\": " +
             (single_population ? "true" : "false") + ", \"all_triples_used\": " + (all_used ? "true" : "false") + ", \"strict_mode\": " +
             (ses.allow_refill() ? "false" : "true") + ", \"session_active\": " + (party.active() ? "true" : "false") + "}\n";
    rec_s += "}\n";

    if (!a.out.empty()) {
        std::ofstream f(a.out, std::ios::binary);
        if (!f) throw std::runtime_error("cannot write " + a.out);
        f << rec_s;
    }
    if (a.quiet) {
        std::printf("party %d: %s %s%s%s %zu sites, %zu AND, %llu triple request(s): %llu bytes (both directions), %llu rounds, "
                    "%.3f s (setup %.3f, preprocess %.3f, online %.3f)%s\n",
                    a.role, model_name.c_str(), a.mode.c_str(), a.private_model ? " private" : "",
                    rec.tiled ? (" tiled (" + std::to_string(rec.tiles) + " tiles)").c_str() : "", model.n_sites(), budget,
                    (unsigned long long)ses.triple_requests(), (unsigned long long)(total.bytes_sent + total.bytes_recv),
                    (unsigned long long)total.rounds, wall, secs(t0, t1), secs(t1, t2), secs(t2, t3),
                    verified ? (correct ? ", logits verified" : ", VERIFICATION FAILED") : "");
    } else {
        std::fputs(rec_s.c_str(), stdout);
    }
    std::fflush(stdout);
    if (!sums_ok) {
        std::fprintf(stderr, "error: the per-phase counters do not sum to the channel totals\n");
        return 3;
    }
    if (!single_population || !all_used) {
        std::fprintf(stderr, "error: the triples were not drawn as planned or not used up\n");
        return 3;
    }
    if (verified && !correct) {
        std::fprintf(stderr, "error: the logits are NOT the expected ones\n");
        return 4;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    InferArgs a;
    try {
        a = parse_infer(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        infer_usage();
        return 2;
    }
    if (a.circuits.empty()) a.circuits = circuit::dirname(argv[0]) + "/../../circuits";
    try {
        return run_infer(a);
    } catch (const ProtocolAbort& e) {
        std::fprintf(stderr, "party %d: PROTOCOL ABORT: %s\n", a.role, e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "party %d: error: %s\n", a.role, e.what());
    }
    return 1;
}
