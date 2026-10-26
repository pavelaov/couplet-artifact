"""Export of a quantised model and its test vectors for the native implementation.

Three files are written: ``<out>/<name>_model.json``, ``<out>/<name>_vectors.json``
and the activation table ``<out>/table_<act>_n<n>_f<f>.bin`` (binary layout in
``couplet.tables``; its ``table_hash`` is the one of the model file).

Model file (``couplet.nn.Model.to_dict``), format ``"couplet-model/1"``:
    kind            "mlp" or "cnn"
    act             activation table name (relu, sigmoid, tanh, gelu, sign, recip, rsqrt)
    n, f            wire format: n-bit codes (= the plane domain), f fraction bits
    sizes           [d, width_1, ..., n_out] (the CNN: [image^2, Ho Wo ch, n_out])
    partition       d // 2: features [0, partition) belong to P_0, the rest to P_1
    meta            the public shape parameters; for the CNN also "kernel_weights"
                    (K^2 x ch reals, row i K + j) and "kernel_bias" (the real kernel)
    layers          per layer {"rows", "cols", "W": rows x cols integers (row-major
                    lists), "b": rows integers}: W = round(W_real 2^f), b = round(b_real 2^f);
                    the convolution is already expanded to its linear map (nn.conv_matrix)
    float_layers    the real layers of the same shapes, or null
    model_hash      hex SHA-256 of "CPLTMDL1" | kind byte (0 mlp, 1 cnn) | le32 len(act) | act |
                    le32 n | le32 f | le32 L | per layer le32 rows, le32 cols, W as int64 LE
                    row-major, b as int64 LE
    architecture_hash   the same under "CPLTARC1" without the W and b blobs
    table_hash      hex SHA-256 of the framed tuple ("Couplet-table", act, le32 n, le32 f,
                    entries as int64 LE) of the activation table (couplet.tables)

Vectors file, format ``"couplet-vectors/1"``:
    model_hash, architecture_hash, table_hash, n, f, partition   as above
    inputs          list of feature vectors, each d integers x_q = round(x 2^f), x in [0, 1]
    float_logits    per input the real logits of the float model, or null
    exact_logits    per input the logits of the fixed-point mirror in mode exact: signed
                    integers with 2f fraction bits (what the secure pipeline opens)
    round           per input {"thresholds": per hidden layer a list of one threshold
                    v_l in [0, 2^f) per site, "logits": the mirror's logits in mode round
                    with these thresholds}; the thresholds are uniform draws of
                    numpy.random.default_rng(threshold_seed), layer by layer, site by site,
                    input by input
    threshold_seed  the seed of those draws
    pipeline        a short restatement of the layer rule (p = W a + b 2^f; z = (p >> f) mod 2^n
                    plus [p mod 2^f > v_l] in mode round; a = T[z]; logits = last p)

Command line (from artifact/python):
    python -m couplet.tools.export_model --model mlp16.json --out DIR --name mlp16 [--inputs 16] [--seed 7]
    python -m couplet.tools.export_model --train mlp16 --n 16 --f 8 --out DIR   (train first)
    python -m couplet.tools.export_model --random mlp4 --n 8 --f 4 --out DIR    (random weights,
                                         nn.random_model with --weights-seed; inputs uniform)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .. import nn as nnm
from .. import tables
from ..nn import Model

__all__ = ["export", "make_vectors"]

VECTOR_FORMAT = "couplet-vectors/1"


def make_vectors(model: Model, inputs, threshold_seed: int) -> dict:
    """The vectors document for the given real feature vectors (list of lists in [0, 1])."""
    rng = np.random.default_rng(int(threshold_seed))
    docs_round = []
    exact = []
    codes_all = []
    float_logits = [] if model.float_layers is not None else None
    for x in inputs:
        codes = nnm.features_to_codes(x, model.f)
        codes_all.append(codes)
        exact.append(model.mirror(codes, "exact").logits)
        thresholds = [[int(v) for v in rng.integers(0, 1 << model.f, width)] for width in model.hidden_widths]
        docs_round.append({"thresholds": thresholds, "logits": model.mirror(codes, "round", thresholds).logits})
        if float_logits is not None:
            float_logits.append([float(v) for v in model.forward_float(np.asarray(x, dtype=np.float64))])
    return {"format": VECTOR_FORMAT, "model_hash": model.model_hash().hex(),
            "architecture_hash": model.architecture_hash().hex(), "table_hash": model.table_hash().hex(),
            "n": model.n, "f": model.f, "partition": model.partition, "inputs": codes_all,
            "float_logits": float_logits, "exact_logits": exact, "round": docs_round,
            "threshold_seed": int(threshold_seed),
            "pipeline": "p = W a + b 2^f (a: inputs x_q or table outputs, f fraction bits); "
                        "z = (p >> f) mod 2^n, plus [p mod 2^f > v_l] in mode round; a = T[z]; "
                        "logits = the last p (2f fraction bits), signed"}


def export(model: Model, out_dir: str, name: str, inputs, threshold_seed: int) -> tuple:
    """Write the model, the vectors and the table file; returns the three paths."""
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"{name}_model.json")
    vec_path = os.path.join(out_dir, f"{name}_vectors.json")
    model.save(model_path)
    with open(vec_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(make_vectors(model, inputs, threshold_seed), fh)
        fh.write("\n")
    table_path = tables.write_table(out_dir, model.act, model.n, model.f)
    return model_path, vec_path, table_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="export a model and its test vectors as JSON")
    ap.add_argument("--model", default=None, help="model JSON written by train_synthetic")
    ap.add_argument("--train", default=None, choices=sorted(nnm.PRESETS), help="train this preset first")
    ap.add_argument("--random", default=None, choices=sorted(nnm.PRESETS),
                    help="this preset with random (untrained) weights from --weights-seed")
    ap.add_argument("--weights-seed", type=int, default=1, help="with --random")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--f", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--inputs", type=int, default=16, help="number of test inputs")
    ap.add_argument("--seed", type=int, default=7, help="seed of the inputs and of the thresholds")
    args = ap.parse_args(argv)
    if sum(x is not None for x in (args.model, args.train, args.random)) != 1:
        ap.error("give exactly one of --model, --train and --random")
    if args.random:
        model = nnm.random_model(args.random, args.n, args.f, seed=args.weights_seed)
        name = args.name or args.random
        x_test = None
    elif args.train:
        from .train_synthetic import train_preset
        model, test, report = train_preset(args.train, args.n, args.f, epochs=args.epochs)
        print(json.dumps(report, indent=1))
        name = args.name or args.train
        x_test = test[0]
    else:
        model = Model.load(args.model)
        name = args.name or os.path.splitext(os.path.basename(args.model))[0]
        x_test = None
    rng = np.random.default_rng(args.seed)
    if x_test is not None and len(x_test) >= args.inputs:
        inputs = x_test[rng.permutation(len(x_test))[:args.inputs]].tolist()
    else:
        inputs = rng.random((args.inputs, model.d)).tolist()
    paths = export(model, args.out, name, inputs, args.seed)
    print("wrote", *paths)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
