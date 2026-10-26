"""Tests of couplet.nn, couplet.infer, couplet.beaver and the tools (DESIGN_M3.md
sections 3 and 5): the models, the cleartext fixed-point mirror and the inference
protocol end to end.

  * models: the convolution as a public linear map equals the direct
    convolution; quantisation; model and architecture hashes; JSON round trip;
    the two parties' first-layer shares sum to the mirror's pre-activation;
    the session plan (manifest) in both pipelines;
  * tools: the seeded task is deterministic; NumPy training of a reduced MLP
    and a reduced CNN reaches a sensible accuracy; the exported vectors are
    reproduced by the mirror;
  * fixed point and tables: the codes and the sign-exact embedding; EVERY
    entry of the sigmoid, tanh and GELU tables at (8, 4), (10, 6) and (16, 8)
    equals the 60-digit decimal rounding (whose series agree with float64),
    reciprocal and inverse square root are the correctly rounded rationals
    (exact integer brackets, ties to even), ReLU and sign the integer maps;
  * Beaver products, matrix-vector products and authenticated input on
    dealer material through the openings of a session;
  * the pipeline with IDEAL preprocessing: bit-exact agreement of both
    parties' logits with the cleartext mirror (given the recorded thresholds)
    for every reduced model, [16, 8, 8, 4], [32, 16, 16, 8], [64, 32, 32, 10]
    and the reduced CNNs (10 x 10 x 2, 12 x 12 x 2, 14 x 14 x 4), in both
    truncation modes at the wire format (16, 8); the prediction agreement
    with the floating-point model is reported; the counters (one triple
    request, no triple left, the openings and rounds of each mode);
  * the smallest model [16, 8, 8, 4] with the REAL preprocessing (base OT,
    OT extension, leaky AND, bucketing), both modes, at a small format;
  * the private-model mode (weights of P_0 as authenticated inputs, matrix
    triples) on the smallest model, and on the tiny shape in mode exact;
  * the triple cap: the budget drawn in several requests, nothing left over;
  * tiled private-model plans (task R-TILE): the tile layout and the plan
    (untiled plans of every committed plan file unchanged, the tiled plan of
    the full MLP shape, the limit); the tiled matrix-vector product on dealer
    material for random matrices of sizes that are and are not multiples of
    the tile, equal to the plaintext product, one exchange; a tampered entry
    of one tile's opening (D or e) aborts at the check; the pipeline with a
    tiled plan bit-exact in both modes; a plan whose tile entries do not match
    the tiles aborts the generation; a tampered tile opening in the pipeline
    aborts the honest party;
  * arguments and the abort on different plans.

Run from artifact/python:  python tests/test_nn.py  (``-k name`` selects tests,
``--no-real`` skips the real preprocessing).  Prints the number of checks;
exits non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import random
import secrets
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import math  # noqa: E402
from decimal import Decimal, localcontext  # noqa: E402

import numpy as np  # noqa: E402

from couplet import beaver  # noqa: E402
from couplet import circuits_ppml as cp  # noqa: E402
from couplet import fixedpoint as fx  # noqa: E402
from couplet import nn as nnm  # noqa: E402
from couplet import tables  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.infer import InferParty  # noqa: E402
from couplet.net import loopback_pair, run_two_party  # noqa: E402
from couplet.prep import IdealDealer  # noqa: E402
from couplet.spdz2k import AShare, ArithChecker, deal  # noqa: E402
from couplet.tools import export_model, train_synthetic as ts  # noqa: E402
from couplet.truncmask import Opener  # noqa: E402

NO_REAL = "--no-real" in sys.argv
CHECKS = 0
RNG = random.Random(0x4E4E)
CNN_EPOCHS = 25           # the small convolutions need more Adam steps than the MLPs
_MODELS: dict = {}


def check(cond, msg: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg or "check failed")


def check_eq(got, exp, msg: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if got != exp:
        raise AssertionError(f"{msg}: got {got!r}, expected {exp!r}")


def raises(exc, fn) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def trained(preset: str, n: int, f: int, epochs: int = 3):
    """A quickly trained preset (cached per call signature); returns (model, test set, report)."""
    key = (preset, n, f, epochs)
    if key not in _MODELS:
        _MODELS[key] = ts.train_preset(preset, n, f, epochs=epochs, train=2048, test=512)
    return _MODELS[key]


# --------------------------------------------------------------------------- #
# Running the protocol
# --------------------------------------------------------------------------- #

def run_inference(model, mode: str, codes, *, k: int = 64, s: int = 64, prep: str = "ideal",
                  private: bool = False, cap=None, models=None, timeout: float = 1800, tiling=None,
                  opener_factories=(None, None)):
    """Both parties run ``InferParty`` on their slices of ``codes``; returns the outcome."""
    dealer = IdealDealer() if prep == "ideal" else None
    sid = b"nn-" + secrets.token_bytes(4)
    models = models or (model, model.shape_only() if private else model)
    parties = [None, None]

    def party(role):
        def fn(chan):
            p = InferParty(role, chan, sid, models[role], mode, prep=prep, dealer=dealer, k=k, s=s,
                           private_model=private, triple_cap=cap, tiling=tiling,
                           opener_factory=opener_factories[role])
            parties[role] = p
            t = time.perf_counter()
            p.setup()
            p.preprocess()
            t_pre = time.perf_counter() - t
            t = time.perf_counter()
            logits = p.infer(codes[model.feature_slice(role)])
            return logits, p.record, t_pre, time.perf_counter() - t
        return fn

    t = time.perf_counter()
    (l0, r0, pre0, on0), (l1, r1, pre1, on1), st0, st1 = run_two_party(party(0), party(1), timeout=timeout)
    return {"logits": (l0, l1), "records": (r0, r1), "stats": (st0, st1), "parties": parties,
            "wall": time.perf_counter() - t, "preprocess_s": max(pre0, pre1), "infer_s": max(on0, on1)}


def check_run(model, mode: str, codes, out: dict, label: str) -> dict:
    """Bit-exactness against the mirror and the counters of one honest run."""
    l0, l1 = out["logits"]
    r0, r1 = out["records"]
    check_eq(l0, l1, f"{label}: both parties hold the same logits")
    check_eq(r0["thresholds"], r1["thresholds"], f"{label}: the same public thresholds")
    mir = model.mirror(codes, mode, r0["thresholds"])
    check_eq(l0, mir.logits, f"{label}: bit-exact agreement with the mirror")
    if mode == "exact":
        check_eq(l0, model.mirror(codes, "exact").logits, f"{label}: the exact mirror needs no thresholds")
    for p in out["parties"]:
        check_eq((p.triple_requests, p.record["triples_unused"]), (1, 0), f"{label}: one request, nothing left")
    widths = model.hidden_widths
    check_eq(r0["executions"], len(out["parties"][0].manifest.entries), f"{label}: one execution per plan entry")
    if not r0["private_model"]:
        check_eq(r0["executions"], len(widths), f"{label}: one batched execution per hidden layer")
    check_eq(r0["sites"], sum(widths), f"{label}: sites")
    later = sum(widths[1:])
    if r0["private_model"]:
        want = sum(widths) * (2 if mode == "exact" else 1)
        if r0.get("tiled"):                                         # D_t and e_t of every tile
            tiling = out["parties"][0].tiling
            want += sum(h * w + w for r, c in model.shapes for _, _, h, w in nnm.layer_tiles(r, c, tiling))
        else:
            want += sum(r * c + c for r, c in model.shapes)        # D and e of every matrix product
    else:
        want = (later + sum(widths)) if mode == "exact" else later
    check_eq(r0["openings_before_release"], want, f"{label}: deferred openings before the release")
    check_eq(r0["openings"], want + model.n_out, f"{label}: the release opens the logits")
    check_eq(r0["checks"], 2 if want else 1, f"{label}: a check per non-empty batch of openings")
    st = out["stats"][0]["phases"]
    if not r0["private_model"]:
        check_eq(st["tm-open"]["rounds"], 2, f"{label}: the first layer is one commit-then-open")
    check_eq(st["arith-check"]["rounds"], 8 if want else 4, f"{label}: four rounds per check")
    check_eq(st["manifest"]["bytes_sent"], 32, f"{label}: the manifest digest")
    fl = model.forward_float(np.array(codes) / float(1 << model.f))
    agree = int(np.argmax(fl)) == int(np.argmax(l0))
    return {"agree": agree, "wall": out["wall"], "overflow": mir.overflow}


def mirror_agreement(model, mode: str, x_test, count: int = 200) -> float:
    """Prediction agreement of the mirror (uniform random thresholds in mode round) with the float model."""
    rng = np.random.default_rng(1)
    hits = 0
    for row in x_test[:count]:
        codes = nnm.features_to_codes(row, model.f)
        thr = [[int(v) for v in rng.integers(0, 1 << model.f, w)] for w in model.hidden_widths]
        mir = model.mirror(codes, mode, thr)
        hits += int(np.argmax(mir.logits) == np.argmax(model.forward_float(row)))
    return hits / min(count, len(x_test))


# --------------------------------------------------------------------------- #
# Models and tools
# --------------------------------------------------------------------------- #

def test_model_basics():
    rng = np.random.default_rng(5)
    img, K, S, ch = 7, 3, 2, 2
    Wc, bc = rng.random((K * K, ch)) - 0.5, rng.random(ch) - 0.5
    W, b = nnm.conv_matrix(img, K, S, ch, Wc, bc)
    x = rng.random((img, img))
    Ho = (img - K) // S + 1
    direct = np.zeros(Ho * Ho * ch)
    for oy in range(Ho):
        for ox in range(Ho):
            patch = x[oy * S:oy * S + K, ox * S:ox * S + K].reshape(-1)
            for c in range(ch):
                direct[(oy * Ho + ox) * ch + c] = patch @ Wc[:, c] + bc[c]
    check(np.allclose(W @ x.reshape(-1) + b, direct), "the convolution as a linear map")
    check_eq(W.shape, (Ho * Ho * ch, img * img), "conv matrix shape")
    check_eq(list(nnm.quantise([0.5, -0.25, 0.3], 2)), [2, -1, 1], "quantise rounds half to even")
    check_eq(nnm.features_to_codes([0.0, 1.0, 0.5], 4), [0, 16, 8], "feature codes")
    check_eq(nnm.decode_logits([256], 4), [1.0], "logits carry 2f fraction bits")
    model = nnm.random_model("cnn10", 8, 4, seed=2)
    check_eq((model.kind, model.d, model.n_out, model.hidden_widths, model.n_sites), ("cnn", 100, 4, [18], 18), "cnn10")
    other = nnm.random_model("cnn10", 8, 4, seed=3)
    check(model.model_hash() != other.model_hash(), "weights enter the model hash")
    check_eq(model.architecture_hash(), other.architecture_hash(), "the architecture hash ignores the weights")
    check_eq(model.shape_only().architecture_hash(), model.architecture_hash(), "shape-only copy")
    check(raises(ValueError, lambda: model.shape_only().model_hash()), "no model hash without weights")
    doc = model.to_dict()
    back = nnm.Model.from_dict(json.loads(json.dumps(doc)))
    check_eq(back.model_hash(), model.model_hash(), "JSON round trip keeps the hash")
    check(all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(back.layers, model.layers)),
          "JSON round trip keeps the layers")
    check(np.allclose(back.forward_float(np.ones(100) * 0.3), model.forward_float(np.ones(100) * 0.3)), "float layers")
    doc["model_hash"] = "00" * 32
    check(raises(ValueError, lambda: nnm.Model.from_dict(doc)), "a wrong hash is refused")
    mlp = nnm.random_model("mlp16", 16, 8, seed=4)
    codes = [RNG.randrange(0, 257) for _ in range(16)]
    p0 = mlp.first_layer_share(0, codes[:8])
    p1 = mlp.first_layer_share(1, codes[8:])
    mir = mlp.mirror(codes, "exact")
    check_eq([a + b for a, b in zip(p0, p1)], mir.pre[0], "the first-layer shares sum to the pre-activation")
    check(raises(ValueError, lambda: mlp.first_layer_share(0, codes[:7])), "wrong slice length")
    sub = mlp.mirror(codes, "exact", first_pre=[v + 1 for v in mir.pre[0]])
    check_eq(sub.pre[0], [v + 1 for v in mir.pre[0]], "the mirror uses a substituted first-layer pre-activation")
    check_eq(sub.codes[0], [((v + 1) >> 8) % (1 << 16) for v in mir.pre[0]], "and truncates it")
    check(mlp.sign_exact_in(128) and not mlp.sign_exact_in(8), "sign-exact bound")
    check_eq(len(mlp.pre_activation_bounds()), 3, "one bound per layer")
    man = nnm.plan_manifest(mlp, "round", 64, 64)
    check_eq([e["batch"] for e in man.entries], [8, 8], "public plan: sites per hidden layer")
    check_eq(man.model_hash, mlp.model_hash(), "public plan binds the model")
    check_eq(man.tables, [("tanh", 16, 8, mlp.table_hash())], "the plan lists the table")
    priv = nnm.plan_manifest(mlp, "exact", 64, 64, private_model=True)
    kinds = [e["descriptor"]["kind"] for e in priv.entries]
    check_eq(kinds, ["inmask", "inmask", "mattriple", "mattriple", "mattriple", "site", "site", "outmask"],
             "private plan")
    check_eq(priv.entries[-1]["batch"], mlp.n_out, "one output mask per logit (correction C2)")
    check_eq(priv.entries[0]["descriptor"]["count"], 8 + (8 * 16 + 8) + (8 * 8 + 8) + (4 * 8 + 4), "P_0's inputs")
    check_eq(priv.model_hash, mlp.architecture_hash(), "the private plan binds the architecture")
    check(raises(ValueError, lambda: nnm.plan_manifest(mlp, "nearest", 64, 64)), "mode")
    check(raises(ValueError, lambda: mlp.mirror(codes, "round")), "mode round needs thresholds")


def test_tools():
    a = ts.make_dataset(16, 4, 100, 20, 11)
    b = ts.make_dataset(16, 4, 100, 20, 11)
    c = ts.make_dataset(16, 4, 100, 20, 12)
    check(np.array_equal(a[0][0], b[0][0]) and np.array_equal(a[1][1], b[1][1]), "the task is deterministic")
    check(not np.array_equal(a[0][0], c[0][0]), "another seed, another task")
    check_eq((a[0][0].shape, a[1][0].shape), ((100, 16), (20, 16)), "shapes")
    check(a[0][0].min() >= 0 and a[0][0].max() <= 1 and a[0][1].max() < 4, "inputs in [0, 1], labels in range")
    im = ts.make_dataset(100, 3, 50, 10, 5, image=10)
    check_eq(im[0][0].shape, (50, 100), "image task")
    check(raises(ValueError, lambda: ts.make_dataset(100, 3, 5, 5, 5, image=9)), "image size")
    model, test, rep = trained("mlp16", 16, 8)
    check(rep["float_accuracy"] > 0.5, f"mlp16 trains to a sensible accuracy ({rep['float_accuracy']:.3f})")
    check(rep["prediction_agreement"] > 0.9, f"the mirror agrees with the float model ({rep['prediction_agreement']:.3f})")
    cnn, test_c, rep_c = trained("cnn10", 16, 8, epochs=CNN_EPOCHS)
    check(rep_c["float_accuracy"] > 0.5, f"cnn10 trains to a sensible accuracy ({rep_c['float_accuracy']:.3f})")
    print(f"    mlp16: float {rep['float_accuracy']:.3f}, mirror {rep['mirror_exact_accuracy']:.3f}, "
          f"agreement {rep['prediction_agreement']:.3f}; cnn10: float {rep_c['float_accuracy']:.3f}, "
          f"mirror {rep_c['mirror_exact_accuracy']:.3f}, agreement {rep_c['prediction_agreement']:.3f}", flush=True)
    for act in ("sigmoid", "gelu", "relu"):
        m = ts.train_mlp([16, 8, 4], act, ts.make_dataset(16, 4, 512, 64, 3)[0], 1, 1, 12, 6)
        check_eq(m.act, act, f"training with {act}")
    inputs = test[0][:5].tolist()
    vec = export_model.make_vectors(model, inputs, 9)
    check_eq((vec["format"], vec["model_hash"], len(vec["inputs"])), ("couplet-vectors/1", model.model_hash().hex(), 5), "vectors")
    for x, codes, ex, rd in zip(inputs, vec["inputs"], vec["exact_logits"], vec["round"]):
        check_eq(codes, nnm.features_to_codes(x, model.f), "input codes")
        check_eq(model.mirror(codes, "exact").logits, ex, "exact logits")
        check_eq(model.mirror(codes, "round", rd["thresholds"]).logits, rd["logits"], "round logits with the thresholds")
    with tempfile.TemporaryDirectory(prefix="couplet_export_") as tmp:
        mp, vp, tp = export_model.export(model, tmp, "t", inputs, 9)
        check_eq(nnm.Model.load(mp).model_hash(), model.model_hash(), "exported model loads")
        with open(vp, encoding="utf-8") as fh:
            check_eq(json.load(fh)["exact_logits"], vec["exact_logits"], "exported vectors")
        check_eq(os.path.basename(tp), f"table_{model.act}_n{model.n}_f{model.f}.bin", "table file name")
        name, n, f, entries = tables.read_table(tp)
        check_eq((name, n, f, entries), (model.act, model.n, model.f, model.table()), "the table file reads back")
        check_eq(os.path.getsize(tp), 8 + 4 + len(name) + 8 + 8 * (1 << n), "table file size")
        with open(tp, "rb") as fh:
            data = fh.read()
        bad = os.path.join(tmp, "bad.bin")
        with open(bad, "wb") as fh:
            fh.write(data[:-1])
        check(raises(ValueError, lambda: tables.read_table(bad)), "a truncated table file is refused")
    # the committed reference exports: their tables are in the same directory and match the hashes
    ref_dir = os.path.normpath(os.path.join(_PY, "vectors", "ppml"))
    for ref in ("mlp16", "mlp32", "cnn10"):
        ref_model = nnm.Model.load(os.path.join(ref_dir, f"{ref}_model.json"))
        with open(os.path.join(ref_dir, f"{ref}_vectors.json"), encoding="utf-8") as fh:
            ref_vec = json.load(fh)
        name, n, f, entries = tables.read_table(os.path.join(ref_dir, tables.table_file_name(
            ref_model.act, ref_model.n, ref_model.f)))
        check_eq(entries, ref_model.table(), f"{ref}: the committed table file")
        check_eq(ref_vec["table_hash"], tables.table_hash(name, n, f).hex(), f"{ref}: table hash")
        for codes, ex, rd in zip(ref_vec["inputs"], ref_vec["exact_logits"], ref_vec["round"]):
            check_eq(ref_model.mirror(codes, "exact").logits, ex, f"{ref}: committed exact logits")
            check_eq(ref_model.mirror(codes, "round", rd["thresholds"]).logits, rd["logits"],
                     f"{ref}: committed round logits")


def test_beaver():
    k, s = 32, 64
    w = k + s
    mod = 1 << w
    d0, d1 = RNG.getrandbits(s), RNG.getrandbits(s)
    delta = d0 + d1
    tri = cp.deal_material(cp.triples_descriptor(3, k, s), delta, RNG)
    mt = cp.deal_material(cp.mattriple_descriptor(2, 3, k, s), delta, RNG)
    im0 = cp.deal_material(cp.inmask_descriptor(0, 2, k, s), delta, RNG)
    im1 = cp.deal_material(cp.inmask_descriptor(1, 3, k, s), delta, RNG)
    # the release decodes the authenticated payload mod 2^k as a signed k-bit integer,
    # so every released value must fit k - 1 = 31 bits: factors below 2^15 in magnitude
    xs = [RNG.randrange(-(1 << 15), 1 << 15) for _ in range(3)]
    ys = [RNG.randrange(-(1 << 15), 1 << 15) for _ in range(3)]
    sx = [deal(v % mod, delta, w) for v in xs]
    sy = [deal(v % mod, delta, w) for v in ys]
    W = [[RNG.randrange(-99, 99) for _ in range(3)] for _ in range(2)]
    vals0, vals1 = [5, -7], [11, -2, 3]
    sid = b"beaver"

    def party(role):
        def fn(chan):
            ck = ArithChecker(role, chan, sid, (d0, d1)[role], s)
            op = Opener(role, chan, sid, ck, (d0, d1)[role])
            prods = beaver.products(op, [a[role] for a in sx], [b[role] for b in sy], beaver.triple_shares(tri[role]))
            single = beaver.product(op, sx[0][role], sy[1][role], beaver.triple_shares(tri[role])[0])
            mine = (im0[role], vals0) if role == 0 else (im1[role], vals1)
            my_sh, peer_sh = beaver.exchange_inputs(op, mine, (im1 if role == 0 else im0)[role])
            a, b = (my_sh, peer_sh) if role == 0 else (peer_sh, my_sh)
            Wsh = [[op.public(W[i][j], w) for j in range(3)] for i in range(2)]
            mv = beaver.matvec(op, Wsh, b, mt[role])
            out = op.open_sign_exact([p.with_sign_exact(True) for p in prods + [single] + a + b + mv])
            return out, ck.openings, chan.stats()["phases"]["input-open"]["rounds"]
        return fn

    (o0, n0, r0), (o1, n1, r1), _, _ = run_two_party(party(0), party(1))
    check_eq(o0, o1, "both parties open the same values")
    expect = [x * y for x, y in zip(xs, ys)] + [xs[0] * ys[1]] + vals0 + vals1 + [sum(W[i][j] * vals1[j] for j in range(3)) for i in range(2)]
    check_eq(o0, expect, "products, inputs and the matrix-vector product")
    check_eq(n0, 6 + 2 + 6 + 3 + 11, "openings: 2 per product, D and e of the matrix product, the release")
    check_eq((r0, r1), (1, 1), "authenticated input is one exchange")
    ck = ArithChecker(0, loopback_pair()[0], sid, d0, s)
    op = Opener(0, ck.chan, sid, ck, d0)
    check(raises(ValueError, lambda: beaver.products(op, [sx[0][0]], [sy[0][0], sy[1][0]], [])), "shape mismatch")
    check(raises(ValueError, lambda: beaver.matvec(op, [[sx[0][0]]], [sx[0][0], sx[1][0]], mt[0])), "matvec shapes")
    check(raises(ValueError, lambda: beaver.exchange_inputs(op, (im1[0], [1, 2, 3]), None)), "not the owner")
    check(raises(ValueError, lambda: beaver.exchange_inputs(op, (im0[0], [1]), None)), "one value per mask")


# --------------------------------------------------------------------------- #
# Fixed point and the activation tables
# --------------------------------------------------------------------------- #

def test_tables_and_fixedpoint():
    # fixed point: codes, the sign-exact embedding, the floor truncation
    check_eq([fx.round_half_even(v) for v in (2.5, 3.5, -2.5, 0.49)], [2, 4, -2, 0], "round half to even")
    check_eq((fx.encode(0.5, 2), fx.encode(-0.375, 3), fx.decode(-3, 3)), (2, -3, -0.375), "encode / decode")
    check_eq([fx.to_signed(u, 8) for u in (0, 127, 128, 255)], [0, 127, -128, -1], "to_signed")
    check_eq(fx.to_unsigned(-1, 8), 255, "to_unsigned")
    check([fx.fits_signed(v, 8) for v in (-129, -128, 127, 128)] == [False, True, True, False], "fits_signed")
    check_eq(fx.embed(-1, 8), 255, "embed")
    check(raises(ValueError, lambda: fx.embed(128, 8)), "embed refuses a value that does not fit w - 1 bits")
    for _ in range(200):
        w = RNG.choice((8, 64, 128))
        v = RNG.randrange(-(1 << (w - 1)), 1 << (w - 1))
        check_eq(fx.unembed(fx.embed(v, w), w), v, "unembed inverts embed")
    check_eq((fx.floor_shift(-5, 2), fx.truncate(-5, 2, 4), fx.truncate(5, 2, 4)), (-2, 14, 1), "floor truncation")
    fmt = fx.Format(16, 8)
    check_eq((fmt.N, fmt.ulp, fmt.max_real, fmt.min_real), (65536, 1 / 256, 32767 / 256, -128.0), "format (16, 8)")
    check_eq((fmt.encode_code(-1.0), fmt.decode_code(65536 - 256)), (65536 - 256, -1.0), "codes of -1")
    check(fmt.in_range(32767) and not fmt.in_range(32768), "range of the codes")
    check(raises(ValueError, lambda: fx.Format(8, 8)) and raises(ValueError, lambda: fx.Format(0, 0)), "bad formats")
    small = fx.Format(8, 4)
    check(all(small.encode_code(small.decode_code(c)) == c for c in range(256)), "every code of (8, 4) round-trips")
    # the 60-digit series behind the fall-back agree with float64
    with localcontext() as ctx:
        ctx.prec = 60
        for i in range(-90, 91):
            x = i / 10
            d = Decimal(i) / 10
            check(abs(float(tables._dec_erf(d)) - math.erf(x)) < 1e-15, f"decimal erf at {x}")
            check(abs(float(tables._dec_tanh(d)) - math.tanh(x)) < 1e-15, f"decimal tanh at {x}")
            check(abs(float(tables._dec_sigmoid(d)) - 1 / (1 + math.exp(-x))) < 1e-15, f"decimal sigmoid at {x}")
    # every entry of every table is the correctly rounded value of its function
    t = time.perf_counter()
    for n, f in ((8, 4), (10, 6), (16, 8)):
        N = 1 << n
        one = 1 << f
        for name in ("sigmoid", "tanh", "gelu"):
            T = tables.activation_table(name, n, f)
            bad = [z for z in range(N) if T[z] != tables._round_decimal(name, fx.to_signed(z, n), f)]
            check_eq(bad, [], f"{name} ({n}, {f}): every entry equals the 60-digit rounding")
        relu, sign = tables.activation_table("relu", n, f), tables.activation_table("sign", n, f)
        recip, rsqrt = tables.activation_table("recip", n, f), tables.activation_table("rsqrt", n, f)
        ok = {"relu": True, "sign": True, "recip": True, "rsqrt": True}
        for z in range(N):
            v = fx.to_signed(z, n)
            ok["relu"] &= relu[z] == max(v, 0)
            ok["sign"] &= sign[z] == (one if v > 0 else (-one if v < 0 else 0))
            if v <= 0:
                ok["recip"] &= recip[z] == 0
                ok["rsqrt"] &= rsqrt[z] == 0
                continue
            # recip: |T - 2^(2f) / v| <= 1/2, a tie goes to the even neighbour
            dist = abs(2 * recip[z] * v - 2 * one * one)
            ok["recip"] &= dist < v or (dist == v and recip[z] % 2 == 0)
            # rsqrt: (T - 1/2)^2 <= 2^(3f) / v <= (T + 1/2)^2, a tie goes to the even neighbour
            q, lo, hi = 4 * one ** 3, (2 * rsqrt[z] - 1) ** 2 * v, (2 * rsqrt[z] + 1) ** 2 * v
            ok["rsqrt"] &= (lo < q < hi) or ((q == lo or q == hi) and rsqrt[z] % 2 == 0)
        for name, good in ok.items():
            check(good, f"{name} ({n}, {f}): every entry is the exact value")
        z = RNG.randrange(N)
        check_eq(tables.evaluate_plain("tanh", n, f, z), tables.activation_table("tanh", n, f)[z], "evaluate_plain")
    print(f"    every entry of the seven tables at (8, 4), (10, 6), (16, 8) checked in {time.perf_counter() - t:.1f} s",
          flush=True)
    check(raises(ValueError, lambda: tables.evaluate_plain("tanh", 8, 4, 256)), "evaluate_plain range")
    check(raises(ValueError, lambda: tables.table_entry("cosh", 0, 8, 4)), "unknown table")
    hashes = {tables.table_hash(name, n, f) for name in ("tanh", "relu") for n, f in ((8, 4), (8, 3), (9, 4))}
    check_eq(len(hashes), 6, "the table hash depends on the name and the format")
    check_eq(tables.table_hash("tanh", 8, 4), tables.table_hash("tanh", 8, 4), "the table hash is deterministic")


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #

PIPELINE_CASES = (("mlp16", "round", 2), ("mlp16", "exact", 2), ("mlp32", "round", 1), ("mlp32", "exact", 1),
                  ("mlp64", "round", 1), ("mlp64", "exact", 1), ("cnn10", "round", 1), ("cnn10", "exact", 1),
                  ("cnn12", "round", 1), ("cnn12", "exact", 1), ("cnn14", "round", 1), ("cnn14", "exact", 1))


def test_pipeline_ideal():
    rows = []
    for preset, mode, count in PIPELINE_CASES:
        model, test, rep = trained(preset, 16, 8, epochs=CNN_EPOCHS if preset.startswith("cnn") else 3)
        agree = 0
        wall = 0.0
        for i in range(count):
            codes = nnm.features_to_codes(test[0][i], model.f)
            out = run_inference(model, mode, codes)
            res = check_run(model, mode, codes, out, f"{preset} {mode} input {i}")
            agree += res["agree"]
            wall += res["wall"]
        mir_agree = mirror_agreement(model, mode, test[0])
        rows.append((preset, mode, model.n_sites, count, agree, mir_agree, rep["float_accuracy"], wall / count))
        print(f"    {preset:<6} {mode:<5}: {model.n_sites:>4} sites, {count} secure run(s) bit-exact with the mirror, "
              f"argmax agreement with the float model {agree}/{count} (mirror over 200 test inputs: {mir_agree:.3f}; "
              f"float accuracy {rep['float_accuracy']:.3f}), {wall / count:5.1f} s per run", flush=True)
    check_eq(len(rows), len(PIPELINE_CASES), "all cases ran")


def test_pipeline_real():
    if NO_REAL:
        print("    skipped (--no-real)")
        return
    for mode, n, f in (("round", 5, 2), ("exact", 4, 2)):
        model, test, rep = trained("mlp16", n, f)
        codes = nnm.features_to_codes(test[0][0], model.f)
        out = run_inference(model, mode, codes, k=32, prep="real")
        res = check_run(model, mode, codes, out, f"mlp16 real {mode}")
        info = out["parties"][0].info()
        check_eq(info["prep"], "real", "real preprocessing")
        check(info["last_bucket_size"] is not None and info["last_bucket_size"] >= 3, "bucketed triples")
        print(f"    mlp16 ({n}, {f}) k=32 real preprocessing, {mode}: bit-exact, bucket size {info['last_bucket_size']}, "
              f"{out['preprocess_s']:.1f} s preprocessing, {out['infer_s']:.2f} s online, "
              f"{out['stats'][0]['bytes_sent']:,} bytes sent by P_0, {out['stats'][0]['rounds']} rounds", flush=True)


def test_private_model():
    model, test, rep = trained("mlp16", 8, 4)
    codes = nnm.features_to_codes(test[0][0], model.f)
    out = run_inference(model, "round", codes, k=32, private=True)
    check_run(model, "round", codes, out, "mlp16 private round")
    check(not out["parties"][1].model.weights_known, "P_1 holds the shape-only model")
    check_eq(out["parties"][0].manifest.model_hash, model.architecture_hash(), "the plan binds the architecture")
    print(f"    mlp16 (8, 4) k=32 private model, round: bit-exact, {out['records'][0]['and_gates']:,} AND gates, "
          f"{out['wall']:.1f} s", flush=True)
    tiny = nnm.random_model("mlp4", 4, 2, seed=8)
    codes = [RNG.randrange(5) for _ in range(4)]
    out = run_inference(tiny, "exact", codes, k=32, private=True)
    check_run(tiny, "exact", codes, out, "mlp4 private exact")


def test_triple_cap():
    model, test, rep = trained("mlp16", 16, 8)
    codes = nnm.features_to_codes(test[0][3], model.f)
    desc = cp.site_descriptor(16, 8, 64, 64, "round")
    cap = 3 * cp.and_count(desc)
    dealer = IdealDealer()
    sid = b"cap-" + secrets.token_bytes(4)
    parties = [None, None]

    def party(role):
        def fn(chan):
            p = InferParty(role, chan, sid, model, "round", prep="ideal", dealer=dealer, triple_cap=cap)
            parties[role] = p
            return p.run(codes[model.feature_slice(role)])
        return fn

    l0, l1, _, _ = run_two_party(party(0), party(1), timeout=900)
    check_eq(l0, l1, "same logits")
    check_eq(l0, model.mirror(codes, "round", parties[0].record["thresholds"]).logits, "bit-exact with the cap")
    rec = parties[0].record
    check_eq((rec["triple_requests"], rec["triples_unused"]), (6, 0), "16 sites in chunks of 3: six requests, none left")
    check_eq(rec["executions"], 6, "six executions")
    check_eq(parties[1].triple_requests, 6, "the same at P_1")


def test_arguments_and_plan_mismatch():
    model = nnm.random_model("mlp4", 4, 2, seed=1)
    c0, c1 = loopback_pair()
    dealer = IdealDealer()
    p = InferParty(0, c0, b"a", model, "round", prep="ideal", dealer=dealer)
    check(raises(ValueError, lambda: InferParty(0, c0, b"a", model, "nearest", prep="ideal", dealer=dealer)), "mode")
    check(raises(ValueError, lambda: InferParty(1, c0, b"a", model, "round", prep="ideal", dealer=dealer)), "role")
    check(raises(ValueError, lambda: InferParty(0, c0, b"a", model.shape_only(), "round", prep="ideal",
                                                dealer=dealer, private_model=True)), "P_0 needs the weights")
    check(raises(ValueError, lambda: InferParty(0, c0, b"a", model.shape_only(), "round", prep="ideal",
                                                dealer=dealer)), "the public pipeline needs the weights")
    big = nnm.random_model("mlp16", 16, 8, seed=1)
    check(raises(ValueError, lambda: InferParty(0, c0, b"a", big, "round", prep="ideal", dealer=dealer, k=16, s=4)),
          "k + s too small for the model's values")
    check(model.sign_exact_in(12) and not big.sign_exact_in(12), "the bound check")
    loud = nnm.random_model("mlp4", 4, 2, seed=1, scale=100.0)
    check(loud.sign_exact_in(8 + 64) and not loud.release_fits(8) and loud.release_fits(64), "the release bound")
    check(raises(ValueError, lambda: InferParty(0, c0, b"a", loud, "round", prep="ideal", dealer=dealer, k=8, s=64)),
          "logits that do not fit a signed payload of k bits are refused")
    check(raises(ValueError, lambda: p.preprocess()), "preprocess before setup")
    check(raises(ValueError, lambda: p.infer([0, 0])), "infer before preprocess")
    check_eq((c0.msgs_sent, c0.bytes_sent), (0, 0), "nothing was sent")
    c0.close()
    c1.close()
    sid = b"mismatch"

    def party(role, mode):
        def fn(chan):
            q = InferParty(role, chan, sid, model, mode, prep="ideal", dealer=dealer)
            try:
                q.setup()
                return "ok"
            except ProtocolAbort as exc:
                return (str(exc), q.state)
        return fn

    r0, r1, st0, st1 = run_two_party(party(0, "round"), party(1, "exact"))
    check_eq((r0, r1), (("manifest digests differ", "ABORTED"), ("manifest digests differ", "ABORTED")),
             "different plans abort both parties at the manifest")
    check_eq((st0["bytes_sent"], st0["rounds"]), (32, 1), "nothing but the digest was sent")


# --------------------------------------------------------------------------- #
# Tiled private-model plans (task R-TILE)
# --------------------------------------------------------------------------- #

class TamperOpener(Opener):
    """Adds ``delta`` to this party's value share of entry ``index`` of the ``call``-th deferred opening."""

    def __init__(self, *args, call: int = 0, index: int = 0, delta: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._call, self._index, self._delta, self._seen = call, index, delta, 0

    def open_deferred(self, shares):
        shares = list(shares)
        if self._seen == self._call:
            sh = shares[self._index]
            shares[self._index] = AShare((sh.value + self._delta) % (1 << sh.bits), sh.mac, sh.bits)
        self._seen += 1
        return super().open_deferred(shares)


def test_tiling_plan():
    T = nnm.Tiling(16, 16, 4096)
    check_eq(T, nnm.DEFAULT_TILING, "default tiling: 16 x 16 above 4,096 weights")
    check(raises(ValueError, lambda: nnm.Tiling(0, 16, 1)) and raises(ValueError, lambda: nnm.Tiling(16, 16, -1)),
          "tile sides positive, limit non-negative")
    for rows, cols, tr, tc in ((37, 21, 16, 16), (128, 784, 16, 16), (10, 128, 16, 16), (5, 7, 2, 3), (1, 1, 4, 4),
                               (64, 48, 16, 16), (3, 7, 2, 2)):
        tl = nnm.Tiling(tr, tc, 0)
        tiles = nnm.layer_tiles(rows, cols, tl)
        cells = sorted((r0 + i, c0 + j) for r0, c0, h, w in tiles for i in range(h) for j in range(w))
        check_eq(cells, [(i, j) for i in range(rows) for j in range(cols)], f"{rows}x{cols}: the tiles partition the matrix")
        check(all(1 <= h <= tr and 1 <= w <= tc for _, _, h, w in tiles), f"{rows}x{cols}: tiles of at most {tr}x{tc}")
        check_eq([(r0, c0) for r0, c0, _, _ in tiles], sorted((r0, c0) for r0, c0, _, _ in tiles), "row-major order")
        shapes = nnm.tile_shapes(rows, cols, tl)
        check_eq(sum(c for _, c in shapes), len(tiles), f"{rows}x{cols}: shape counts")
        firsts = [next(i for i, t in enumerate(tiles) if (t[2], t[3]) == sh) for sh, _ in shapes]
        check_eq(firsts, sorted(firsts), "shapes in the order of their first tile")
    check_eq(nnm.tile_shapes(37, 21, T), [((16, 16), 2), ((16, 5), 2), ((5, 16), 1), ((5, 5), 1)], "37x21 shapes")
    # the untiled plans of every committed plan file are unchanged, and the tiled ones are reproduced
    vec = os.path.join(_PY, "vectors", "ppml")
    names = sorted(x for x in os.listdir(vec) if "_plan_" in x and x.endswith(".json"))
    tiled_files = 0
    for name in names:
        with open(os.path.join(vec, name), encoding="utf-8") as fh:
            doc = json.load(fh)
        model = nnm.Model.load(os.path.join(vec, doc["model_file"]))
        tl = nnm.Tiling(**{k: doc["tiling"][k] for k in ("rows", "cols", "limit")}) if "tiling" in doc else None
        man = nnm.plan_manifest(model, doc["mode"], doc["k"], doc["s"], doc["private_model"], tl)
        check_eq(man.encode().hex(), doc["manifest_hex"], f"{name}: the same CPLTMAN1 encoding")
        check_eq(man.digest().hex(), doc["digest"], f"{name}: the same digest")
        if doc["private_model"]:
            check_eq(nnm.plan_manifest(model.shape_only(), doc["mode"], doc["k"], doc["s"], True, tl).encode(),
                     man.encode(), f"{name}: P_1 (shape only) builds the same plan")
        tiled_files += int(nnm.plan_tiled(model, doc["private_model"], tl))
        if not nnm.plan_tiled(model, doc["private_model"], tl):
            check("tiled" not in json.dumps([e["use"] for e in man.entries]), f"{name}: untiled plan")
    check(tiled_files >= 4, "the tiled plan files of mlp7 and mlp48 are among them")
    # the full private MLP: 464 tiles, the input masks in instances of 256
    full = nnm.random_model("mlp784", 16, 8, seed=2).shape_only()
    check(nnm.plan_tiled(full, True) and not nnm.plan_tiled(full, False), "the full MLP is tiled in private-model mode only")
    man = nnm.plan_manifest(full, "round", 64, 64, True)
    rows = [(e["descriptor"]["kind"], cp.descriptor_name(e["descriptor"]), e["batch"]) for e in man.entries]
    check_eq(rows, [("inmask", "inmask_p0_c256_k64_s64", 463), ("inmask", "inmask_p0_c146_k64_s64", 1),
                    ("inmask", "inmask_p1_c256_k64_s64", 1), ("inmask", "inmask_p1_c136_k64_s64", 1),
                    ("mattriple", "mattriple_16x16_k64_s64", 392), ("mattriple", "mattriple_16x16_k64_s64", 64),
                    ("mattriple", "mattriple_10x16_k64_s64", 8), ("site", "site_n16_f8_k64_s64_round", 128),
                    ("site", "site_n16_f8_k64_s64_round", 128), ("outmask", "gate_n1_k64_s64_shared_om1", 10)],
             "the tiled plan of [784,128,128,10]")
    check_eq(man.entries[0]["use"], "inputs of P_0: features, weights, biases, 256 per instance", "use of the masks")
    check_eq(man.entries[4]["use"], "matrix triple tiles of layer 1: 128x784 in tiles of 16x16", "use of the tiles")
    check_eq(463 * 256 + 146, 392 + sum(r * c + r for r, c in full.shapes), "every input of P_0 has one mask")
    check_eq(sum(e["batch"] for e in man.entries if e["descriptor"]["kind"] == "mattriple"), 464, "464 tile triples")
    # the limit: a model with exactly `limit` weights keeps the untiled plan
    mlp = nnm.random_model("mlp16", 16, 8, seed=3)
    check_eq(mlp.n_weights, 224, "weights of [16,8,8,4]")
    untiled = nnm.plan_manifest(mlp, "round", 64, 64, True)
    check_eq(nnm.plan_manifest(mlp, "round", 64, 64, True, nnm.Tiling(4, 4, 224)).encode(), untiled.encode(),
             "at the limit: untiled")
    t223 = nnm.plan_manifest(mlp, "round", 64, 64, True, nnm.Tiling(4, 4, 223))
    check(t223.encode() != untiled.encode() and t223.digest() != untiled.digest(), "above the limit: tiled, another digest")
    check_eq(nnm.plan_manifest(mlp, "round", 64, 64, False, nnm.Tiling(4, 4, 0)).encode(),
             nnm.plan_manifest(mlp, "round", 64, 64, False).encode(), "the public-model plan never tiles")
    check(raises(ValueError, lambda: nnm.plan_manifest(mlp, "round", 64, 64, True, (16, 16))), "tiling type")


def _tiled_product(rows: int, cols: int, tiling, *, k: int = 32, s: int = 64, tamper=None, order=None):
    """matvec_tiled on dealer material between two parties; ``tamper``: (call, index) at P_0."""
    w = k + s
    mod = 1 << w
    d0, d1 = RNG.getrandbits(s), RNG.getrandbits(s)
    delta = d0 + d1
    tiles = nnm.layer_tiles(rows, cols, tiling)
    mats = [cp.deal_material(cp.mattriple_descriptor(h, wd, k, s), delta, RNG) for _, _, h, wd in tiles]
    W = [[RNG.randrange(-(1 << 10), 1 << 10) for _ in range(cols)] for _ in range(rows)]
    x = [RNG.randrange(-(1 << 10), 1 << 10) for _ in range(cols)]
    sW = [[deal(v % mod, delta, w) for v in row] for row in W]
    sx = [deal(v % mod, delta, w) for v in x]
    sid = b"tiled-" + secrets.token_bytes(4)
    idx = list(range(len(tiles))) if order is None else order

    def party(role):
        def fn(chan):
            ck = ArithChecker(role, chan, sid, (d0, d1)[role], s)
            if tamper is not None and role == 0:
                op = TamperOpener(role, chan, sid, ck, (d0, d1)[role], call=tamper[0], index=tamper[1])
            else:
                op = Opener(role, chan, sid, ck, (d0, d1)[role])
            r_before = chan.rounds
            y = beaver.matvec_tiled(op, [[sh[role] for sh in row] for row in sW], [sh[role] for sh in sx],
                                    [(tiles[i][0], tiles[i][1], mats[i][role]) for i in idx])
            rounds = chan.rounds - r_before
            try:
                out = op.open_sign_exact([v.with_sign_exact(True) for v in y])
            except ProtocolAbort as exc:
                return ("abort", str(exc), op.checker.dead)
            return out, ck.openings, rounds
        return fn

    r0, r1, _, _ = run_two_party(party(0), party(1))
    want = [sum(W[i][j] * x[j] for j in range(cols)) for i in range(rows)]
    return r0, r1, want, tiles


def test_matvec_tiled():
    cases = ((37, 21, nnm.Tiling(16, 16, 0)), (64, 48, nnm.Tiling(16, 16, 0)), (10, 128, nnm.Tiling(16, 16, 0)),
             (5, 7, nnm.Tiling(2, 3, 0)), (3, 7, nnm.Tiling(2, 2, 0)), (16, 16, nnm.Tiling(16, 16, 0)),
             (1, 9, nnm.Tiling(4, 4, 0)), (9, 1, nnm.Tiling(4, 4, 0)))
    for rows, cols, tl in cases:
        r0, r1, want, tiles = _tiled_product(rows, cols, tl)
        label = f"{rows}x{cols} in tiles of {tl.rows}x{tl.cols}"
        check_eq(r0[0], want, f"{label}: the tiled product equals the plaintext product")
        check_eq(r1[0], want, f"{label}: both parties")
        check_eq(r0[1], sum(h * w + w for _, _, h, w in tiles) + rows, f"{label}: openings D_t, e_t per tile, the release")
        check_eq(r0[2], 1, f"{label}: one exchange for all tiles")
    # the order of the list is the order of the openings; any order gives the product
    r0, r1, want, tiles = _tiled_product(11, 13, nnm.Tiling(4, 5, 0), order=list(range(8, -1, -1)))
    check_eq((r0[0], r1[0]), (want, want), "tiles listed in another order")
    # a tampered entry of one tile's opening: D of the third tile, e of the last tile
    for label, index_fn in (("D of tile 3", lambda tiles: sum(h * w + w for _, _, h, w in tiles[:2]) + 5),
                            ("e of the last tile", lambda tiles: sum(h * w + w for _, _, h, w in tiles) - 1)):
        tiles = nnm.layer_tiles(37, 21, nnm.Tiling(16, 16, 0))
        r0, r1, want, _ = _tiled_product(37, 21, nnm.Tiling(16, 16, 0), tamper=(0, index_fn(tiles)))
        check_eq(r1[0], "abort", f"tampered {label}: the honest party aborts at the check")
        check(r1[2], f"tampered {label}: the honest checker is retired")
        check_eq(r0[0], "abort", f"tampered {label}: the deviating party's check fails too")
    # malformed tile lists are refused before any message
    c0, c1 = loopback_pair()
    ck = ArithChecker(0, c0, b"x", 1, 64)
    op = Opener(0, c0, b"x", ck, 1)
    sh = AShare(0, 0, 96)
    m22 = cp.deal_material(cp.mattriple_descriptor(2, 2, 32, 64), 3, RNG)[0]
    m21 = cp.deal_material(cp.mattriple_descriptor(2, 1, 32, 64), 3, RNG)[0]
    W = [[sh] * 3 for _ in range(2)]
    check(raises(ValueError, lambda: beaver.matvec_tiled(op, W, [sh] * 3, [(0, 0, m22)])), "a gap")
    check(raises(ValueError, lambda: beaver.matvec_tiled(op, W, [sh] * 3, [(0, 0, m22), (0, 1, m22)])), "an overlap")
    check(raises(ValueError, lambda: beaver.matvec_tiled(op, W, [sh] * 3, [(0, 0, m22), (0, 2, m22)])), "outside")
    check(raises(ValueError, lambda: beaver.matvec_tiled(op, W, [sh] * 3, [(0, 0, m22), (0, 2, m21), (0, 2, m21)])),
          "a tile twice")
    check_eq((c0.msgs_sent, c0.bytes_sent), (0, 0), "nothing was sent")
    c0.close()
    c1.close()
    # merged input masks: concatenation in order, the total count
    parts = [cp.deal_material(cp.inmask_descriptor(0, c, 32, 64), 5, RNG) for c in (4, 4, 3)]
    for role in (0, 1):
        merged = beaver.merge_inmasks([p[role] for p in parts])
        check_eq((merged.count, merged.owner, merged.shares), (11, 0, [x for p in parts for x in p[role].shares]),
                 f"merged input masks at P_{role}")
        check_eq(merged.values, None if role == 1 else [v for p in parts for v in p[0].values], "the owner's values")
    check(raises(ValueError, lambda: beaver.merge_inmasks([parts[0][0], parts[1][1]])), "different roles")


def test_private_tiled():
    model = nnm.random_model("mlp7", 8, 4, seed=3)
    tl = nnm.Tiling(2, 2, 0)
    codes = [RNG.randrange(17) for _ in range(model.d)]
    for mode in ("round", "exact"):
        out = run_inference(model, mode, codes, k=32, private=True, tiling=tl)
        check_run(model, mode, codes, out, f"mlp7 tiled 2x2 private {mode}")
        rec = out["records"][0]
        check_eq((rec["tiled"], rec["tiles"], out["records"][1]["tiles"]), (True, 10, 10), "ten tile triples")
        man = out["parties"][0].manifest
        check_eq([cp.descriptor_name(e["descriptor"]) for e in man.entries if e["descriptor"]["kind"] != "site"],
                 ["inmask_p0_c4_k32_s64", "inmask_p0_c3_k32_s64", "inmask_p1_c4_k32_s64", "mattriple_2x2_k32_s64",
                  "mattriple_2x1_k32_s64", "mattriple_1x2_k32_s64", "mattriple_1x1_k32_s64", "mattriple_2x2_k32_s64",
                  "mattriple_2x1_k32_s64", "gate_n1_k32_s64_shared_om1"], f"{mode}: the tile entries")
        check_eq(out["parties"][0].inmasks[0].count, 3 + 21 + 3 + 6 + 2, "the merged input masks of P_0")
    # one tile per layer (every matrix fits a 16 x 16 tile): the openings of the untiled product
    out = run_inference(model, "round", codes, k=32, private=True, tiling=nnm.Tiling(16, 16, 0))
    check_run(model, "round", codes, out, "mlp7 one tile per layer")
    check_eq(out["records"][0]["tiles"], 2, "two tiles")
    # a plan whose tile entries do not match the tiles (both parties agree on it): the generation aborts
    sid = b"tile-mismatch"
    dealer = IdealDealer()

    def party(role):
        def fn(chan):
            p = InferParty(role, chan, sid, model if role == 0 else model.shape_only(), "round", prep="ideal",
                           dealer=dealer, k=32, private_model=True, tiling=tl)
            p.manifest.entries = [e for e in p.manifest.entries if e["descriptor"]["kind"] != "mattriple"
                                  or (e["descriptor"]["rows"], e["descriptor"]["cols"]) != (1, 1)]
            p.setup()
            try:
                p.preprocess()
                return "ok"
            except ProtocolAbort as exc:
                return str(exc), p.state
        return fn

    r0, r1, _, _ = run_two_party(party(0), party(1))
    check_eq((r0, r1), (("the matrix triples do not match the tiles of the plan", "ABORTED"),) * 2,
             "a missing tile entry aborts the generation at both parties")
    # a tampered entry of a tiled opening (P_0 deviates; layer 1 = call 0, layer 2 = the call after the
    # truncation openings): the honest P_1 aborts, outputs nothing and is retired
    for mode, call, index in (("round", 0, 5), ("round", 2, 7), ("exact", 0, 13), ("exact", 3, 8)):
        sid = b"tile-tamper-" + secrets.token_bytes(4)
        dealer = IdealDealer()

        def party(role, mode=mode, call=call, index=index, sid=sid, dealer=dealer):
            def fn(chan):
                fac = (lambda *a: TamperOpener(*a, call=call, index=index)) if role == 0 else None
                p = InferParty(role, chan, sid, model if role == 0 else model.shape_only(), mode, prep="ideal",
                               dealer=dealer, k=32, private_model=True, tiling=tl, opener_factory=fac)
                try:
                    return "logits", p.run(codes[model.feature_slice(role)]), p.state
                except ProtocolAbort as exc:
                    return "abort", str(exc), p.state
            return fn

        r0, r1, _, _ = run_two_party(party(0), party(1))
        check_eq((r1[0], r1[2]), ("abort", "ABORTED"),
                 f"{mode}: a tampered entry {index} of opening {call} (a tile's D or e) aborts the honest party")
        check(r1[1] != "logits" and r0[0] == "abort", f"{mode}: no logits at the honest party, the deviating check fails")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    selected = [sys.argv[sys.argv.index("-k") + 1]] if "-k" in sys.argv else []
    failed = []
    start = time.perf_counter()
    for name, fn in tests:
        if selected and not any(s in name for s in selected):
            continue
        before = CHECKS
        t = time.perf_counter()
        try:
            fn()
            print(f"  ok   {name:<36} {CHECKS - before:>7} checks {time.perf_counter() - t:7.1f} s", flush=True)
        except Exception:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}\n{traceback.format_exc()}", flush=True)
    elapsed = time.perf_counter() - start
    if failed:
        print(f"FAILED: {len(failed)} of {len(tests)} tests ({', '.join(failed)}); {CHECKS} checks run, {elapsed:.1f} s")
        return 1
    print(f"TOTAL CHECKS: {CHECKS}  ({len(tests)} tests, all passed, {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
