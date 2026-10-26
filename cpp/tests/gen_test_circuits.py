"""Helper of src/test_engine.cpp: export small test circuits for the C++ engine.

The C++ code does not compile circuits; it loads the binary files written by
``couplet.circuit.Circuit.to_bytes``.  This script builds, with the Python
``Builder``, a fixed set of small circuits (pseudo-random gate lists and a few
hand-made ones) and writes into ``artifact/cpp/tests/circuits/``:

  <name>.bin      the canonical binary encoding;
  manifest.json   name, file and SHA-256 digest of every circuit (the format
                  of ``artifact/circuits/manifest.json`` as far as the C++
                  ``CircuitStore`` reads it);
  cases.json      for every circuit a few input assignments together with the
                  outputs computed by the Python plaintext evaluator, so that
                  the C++ plaintext evaluator is checked against the reference
                  (values are little-endian byte strings in hex; input keys
                  are "<owner>:<label>" with owner 0, 1 or bound).

Circuits: ``rnd00`` .. ``rnd15`` (random XOR / AND / NOT lists of 20 to 6000
gates with public and private outputs), ``xoronly`` (no AND gate), ``retain``
(only RETAIN outputs: nothing is opened), ``consts`` (outputs that are
constants or input wires), ``oneparty`` (inputs of P_0 only), ``add32`` and
``mul16`` (arithmetic with a known meaning), and the chain ``chain_a`` ->
``chain_b`` -> ``chain_c`` in which RETAIN outputs of one circuit are the BOUND
inputs of the next.

The output is deterministic (fixed seed).  Usage, from any directory:

    python artifact/cpp/tests/gen_test_circuits.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "python"))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet.circuit import BOUND, PUBLIC, RETAIN, Builder, Circuit, evaluate_batch  # noqa: E402

OUT_DIR = os.path.join(_HERE, "circuits")
SEED = 20261010
CASES_PER_CIRCUIT = 5


def _pick(rng: random.Random, pool: list) -> int:
    """A wire of the pool, biased towards recent wires (gives depth)."""
    if rng.random() < 0.6 and len(pool) > 8:
        return pool[-1 - rng.randrange(min(len(pool), 24))]
    return rng.choice(pool)


def random_circuit(rng: random.Random, n_gates: int, p_and: float, decls, n_out: int,
                   bound=(), retain=()) -> Circuit:
    """Random gate list.  ``decls`` = [(owner, label, width)], ``bound`` = [(label, width)]
    (BOUND inputs), ``retain`` = [(label, width)] (RETAIN outputs)."""
    B = Builder()
    pool = [0, 1]
    for owner, label, width in list(decls) + [(BOUND, label, width) for label, width in bound]:
        pool += B.input(owner, label, width)
    for _ in range(n_gates):
        x = rng.random()
        a, b = _pick(rng, pool), _pick(rng, pool)
        if x < p_and:
            w = B.and_(a, b)
        elif x < p_and + 0.1:
            w = B.not_(a)
        else:
            w = B.xor(a, b)
        pool.append(w)
    for j in range(n_out):
        width = rng.choice((1, 3, 8, 17, 40))
        to = rng.choice((PUBLIC, PUBLIC, 0, 1))
        B.output(f"o{j}", [_pick(rng, pool) for _ in range(width)], to=to)
    for label, width in retain:
        B.output(label, [_pick(rng, pool) for _ in range(width)], to=RETAIN)
    return B.build()


def build_all() -> dict:
    rng = random.Random(SEED)
    out = {}
    sizes = [20, 50, 90, 150, 300, 500, 800, 1200, 1700, 2300, 3000, 3600, 4200, 4800, 5400, 6000]
    for i, n_gates in enumerate(sizes):
        decls = [(0, "a", rng.choice((1, 5, 16, 33))), (1, "b", rng.choice((1, 7, 16, 40)))]
        if i % 3 == 0:
            decls += [(0, "c", 9), (1, "d", 2)]
        out[f"rnd{i:02d}"] = random_circuit(rng, n_gates, rng.choice((0.15, 0.3, 0.5)), decls, 2 + i % 4)

    out["xoronly"] = random_circuit(rng, 400, 0.0, [(0, "a", 20), (1, "b", 20)], 4)
    out["retain"] = random_circuit(rng, 300, 0.3, [(0, "a", 12), (1, "b", 12)], 0, retain=[("keep", 16)])

    B = Builder()                                    # constants and input wires as outputs
    a = B.input(0, "a", 4)
    b = B.input(1, "b", 4)
    B.output("zero", [0, 0, 0], to=PUBLIC)
    B.output("one", [1, 1], to=0)
    B.output("mixed", [0, 1, a[0], b[1], 1], to=1)
    B.output("echo_a", a, to=PUBLIC)
    B.output("echo_b", b, to=0)
    B.output("and", [B.and_(a[i], b[i]) for i in range(4)], to=PUBLIC)
    B.output("dup", [a[0], a[0], B.not_(a[0])], to=PUBLIC)
    out["consts"] = B.build()

    out["oneparty"] = random_circuit(rng, 250, 0.4, [(0, "a", 30)], 3)

    B = Builder()                                    # 32-bit addition
    a = B.input(0, "a", 32)
    b = B.input(1, "b", 32)
    s, carry = B.add_mod(a, b)
    B.output("sum", s, to=PUBLIC)
    B.output("carry", [carry], to=PUBLIC)
    out["add32"] = B.build()

    B = Builder()                                    # 16 x 16 -> 32 multiplication, product to P_1
    a = B.input(0, "a", 16)
    b = B.input(1, "b", 16)
    B.output("prod", B.mul(a, b, 32), to=1)
    out["mul16"] = B.build()

    # A chain of three circuits: RETAIN outputs become BOUND inputs.
    out["chain_a"] = random_circuit(rng, 500, 0.3, [(0, "a", 20), (1, "b", 20)], 2,
                                    retain=[("keep", 24), ("extra", 5)])
    out["chain_b"] = random_circuit(rng, 600, 0.3, [(0, "x", 10), (1, "y", 10)], 3,
                                    bound=[("keep", 24)], retain=[("keep2", 12)])
    out["chain_c"] = random_circuit(rng, 400, 0.3, [(1, "z", 6)], 3, bound=[("keep2", 12), ("extra", 5)])
    return out


def _hex(bits) -> str:
    v = 0
    for i, bit in enumerate(bits):
        v |= int(bit) << i
    return v.to_bytes((len(bits) + 7) // 8, "little").hex()


def _owner(owner) -> str:
    return "bound" if owner == BOUND else str(owner)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED + 1)
    manifest, cases = [], []
    for name, C in build_all().items():
        C.validate()
        data = C.to_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if Circuit.from_bytes(data).digest() != digest:
            raise RuntimeError(f"{name}: the encoding does not round-trip")
        with open(os.path.join(OUT_DIR, name + ".bin"), "wb") as f:
            f.write(data)
        st = C.stats()
        manifest.append({"name": name, "binary_file": name + ".bin", "binary_bytes": len(data),
                         "digest": digest, "stats": st})
        assignments = [{(owner, label): [rng.getrandbits(1) for _ in wires] for owner, label, wires in C.inputs}
                       for _ in range(CASES_PER_CIRCUIT)]
        results = evaluate_batch(C, assignments)
        cases.append({
            "name": name, "digest": digest,
            "inputs": [{"owner": _owner(o), "label": l, "width": len(w)} for o, l, w in C.inputs],
            "outputs": [{"owner": str(t), "label": l, "width": len(w)} for t, l, w in C.outputs],
            "n_and": st["n_and"], "and_depth": st["and_depth"], "n_gates": st["n_gates"],
            "cases": [{"inputs": {f"{_owner(o)}:{l}": _hex(bits) for (o, l), bits in asg.items()},
                       "outputs": {label: _hex(bits) for label, bits in res.items()}}
                      for asg, res in zip(assignments, results)],
        })
        print(f"  {name:<10} gates={st['n_gates']:>5} AND={st['n_and']:>5} depth={st['and_depth']:>4} "
              f"digest={digest[:16]}..")
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"generated_by": "python artifact/cpp/tests/gen_test_circuits.py", "circuits": manifest}, f, indent=1)
        f.write("\n")
    with open(os.path.join(OUT_DIR, "cases.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"generated_by": "python artifact/cpp/tests/gen_test_circuits.py", "circuits": cases}, f, indent=1)
        f.write("\n")
    print(f"wrote {len(manifest)} circuits to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
