"""Helper of src/test_protocol.cpp and src/test_engine.cpp: values computed by
the Python reference that the C++ code must reproduce bit for bit.

Writes ``artifact/cpp/tests/vectors_protocol.json``:

  gate_materials      for several gate descriptors, both parties' material as
                      a trusted dealer produces it (``gates.deal_gate``; NOT a
                      protocol) in the ``GateMaterial`` encoding, with the MAC
                      key shares, the offset ``r``, the canonical descriptor
                      JSON and the name of the generation circuit;
  check_coefficients  outputs of ``spdz2k.check_coefficients``;
  sub_sessions        the sub-session identifiers of the engine and of the
                      arithmetic checker for one ``sid``;
  headers             evaluation headers of the Boolean engine (the first 32
                      bytes of the first message of an evaluation) for two
                      test circuits of ``tests/circuits/``, one of them with
                      bound wires.

The output is deterministic (fixed seed).  Run ``gen_test_circuits.py``
first.  Usage, from any directory:

    python artifact/cpp/tests/gen_protocol_vectors.py
"""

from __future__ import annotations

import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "python"))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import circuits_gen as cg  # noqa: E402
from couplet import gates, hashes as hs, spdz2k  # noqa: E402

SEED = 7
OUT = os.path.join(_HERE, "vectors_protocol.json")


def main() -> int:
    rnd = random.Random(SEED)
    out = {"generated_by": "python artifact/cpp/tests/gen_protocol_vectors.py",
           "gate_materials": [], "check_coefficients": [], "sub_sessions": {}, "headers": []}
    specs = [
        ("table n=5", gates.table_descriptor(5, 64, 64, False, 0)),
        ("table, authenticated input, n=4, two output masks", gates.table_descriptor(4, 64, 64, True, 2)),
        ("comparison n=6 c=21", gates.comparison_descriptor(6, 64, 64, 21, False, 0)),
        ("comparison, authenticated input, n=6 c=N", gates.comparison_descriptor(6, 64, 64, 64, True, 1)),
        ("relu_large k=8", gates.relu_large_descriptor(8, 64, True, 1)),
        ("linear spline k=32", gates.linear_spline_descriptor(32, 64, [0, 1000, 1 << 31, 1 << 32], True, 1)),
        ("interval n=10 k=32", gates.interval_descriptor(10, 32, 64, 100, 900, False, 0)),
    ]
    for name, desc in specs:
        d0, d1 = rnd.getrandbits(64) | (1 << 63), rnd.getrandbits(64) | (1 << 63)   # Delta >= 2^s
        m0, m1 = gates.deal_gate(desc, d0 + d1, rng=rnd)
        r = (m0.rp_share + m1.rp_share) % (1 << m0.rp_bits) % (1 << desc["n"])
        out["gate_materials"].append({
            "name": name,
            "descriptor_json": json.dumps(cg.canonical_descriptor(desc), sort_keys=True, separators=(",", ":")),
            "circuit_name": cg.descriptor_name(desc),
            "delta0": str(d0), "delta1": str(d1), "r": str(r),
            "p0": m0.to_bytes().hex(), "p1": m1.to_bytes().hex(),
        })
    for seed_hex, bits, count, s in (("00" * 32, 128, 5, 64), ("ab" * 32, 72, 4, 64), ("01" * 32, 80, 3, 16),
                                     ("ff" * 32, 20, 6, 3)):
        out["check_coefficients"].append({
            "seed": seed_hex, "bits": bits, "count": count, "s": s,
            "chi": [str(c) for c in spdz2k.check_coefficients(bytes.fromhex(seed_hex), bits, count, s)]})

    sid = b"vector-sid"
    sid_abb = hs.sha256(b"Couplet-abb-sid", sid)
    out["sub_sessions"] = {"sid": sid.hex(), "sid_abb": sid_abb.hex(),
                           "sid_arith": hs.sha256(b"Couplet-arith-sid", sid).hex()}

    with open(os.path.join(_HERE, "circuits", "manifest.json"), "r", encoding="utf-8") as f:
        digests = {e["name"]: bytes.fromhex(e["digest"]) for e in json.load(f)["circuits"]}

    def header(circuit: str, evaluation: int, batch: int, bound) -> str:
        desc = []
        for label, ids, wires_batch in bound:
            desc += [label.encode(), b"".join(i.to_bytes(8, "little") for i in ids), wires_batch.to_bytes(8, "little")]
        return hs.sha256(b"Couplet-abb-eval", sid_abb, evaluation.to_bytes(8, "little"), digests[circuit],
                         batch.to_bytes(8, "little"), hs.sha256(*desc)).hex()

    # rnd05 as evaluation 0 with batch 3; then chain_a (evaluation 1, batch 2, RETAIN "keep" = wires 0..23,
    # "extra" = wires 24..28) and chain_b (evaluation 2, batch 2) bound to "keep".
    out["headers"] = [
        {"circuit": "rnd05", "evaluation": 0, "batch": 3, "header": header("rnd05", 0, 3, [])},
        {"circuit": "chain_a", "evaluation": 1, "batch": 2, "header": header("chain_a", 1, 2, [])},
        {"circuit": "chain_b", "evaluation": 2, "batch": 2,
         "header": header("chain_b", 2, 2, [("keep", list(range(24)), 2)])},
    ]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT}: {len(out['gate_materials'])} gate materials, "
          f"{len(out['check_coefficients'])} coefficient vectors, {len(out['headers'])} headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
