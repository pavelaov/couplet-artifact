"""couplet: reference Python implementation of the Couplet artifact.

The cryptographic core follows the core design document (``DESIGN.md``);
section numbers below refer to that file.  This module re-exports the
plaintext layer only; everything else is imported from its own module.

Plaintext layer (re-exported here):
  ascon     Ascon-p12 (scalar and numpy-vectorised) and the PRG maps
            G / Conv / ConvV_L / ConvV_R with feed-forward (section 2.1).
  fss       plaintext two-plane DPF and DCF: Gen, Eval, EvalAll and the
            canonical key serialisation (section 3).
  rng       deterministic SHA-256 counter-mode byte source for golden vectors.

Circuits (section 4):
  circuit, circuits_ascon, circuits_gen
            Boolean circuits, the Ascon-p12 circuit and the generation
            circuits C_dpf / C_dcf / C_gate (sections 4.1 - 4.3).

Backend primitives (sections 2.2, 2.4, 7.1):
  gf128, ed25519, hashes, net
            GF(2^128), the Edwards25519 group, hashing / commitments / coin
            tossing, channels with accounting.

Preprocessing (sections 7.1 - 7.5, 7.7):
  prep      base OT, correlated OT extension, authenticated bits, leaky AND,
            bucketing, the triple check; the ideal dealer.

Protocols:
  abb       authenticated Boolean engine, F_ABB (sections 2.2, 5.1, 7.6).
  session   session state, the persistent wires of Delta, triple reservation
            (sections 1, 4, 5.4, 7.5).
  protocol  Pi_Gen for keys and gates, AND budgets (sections 4, 5.2).
  spdz2k    SPDZ2k sharings and ArithCheck (section 2.3).
  gates     the gate catalogue (section 6).
  state     state machine and persisted counters (section 5.4).
  autheval  Pi_AuthEval, the driver of F_AuthEval (section 5.3).

Evaluation support (section 9):
  baselines   re-implemented comparison protocols and cost models.
  attacks     adversarial regression (the appendix of section 9).

The Couplet layer (DESIGN_M3.md; private inference on the core):
  circuits_ppml   generation circuits of sites, Beaver and matrix triples, input
                  masks; Pi_Gen for them; the manifest (section 2).
  planes          one authenticated one-hot plane and its local queries (section 3).
  truncmask       truncate-and-mask in both modes; the openings of a session.
  beaver          products and matrix-vector products with triples; authenticated input.
  fixedpoint, tables
                  two's-complement fixed point; exact activation tables.
  nn, infer       models (MLP, CNN as a public linear map), the fixed-point mirror,
                  the session plan; the party-local inference driver.
  tools           the seeded synthetic task with NumPy training; model export.
"""

from .ascon import G, conv, convv_L, convv_R, permute, permute_np
from .fss import (DCFKey, DPFKey, dcf_eval, dcf_eval_all, dcf_gen, dpf_eval, dpf_eval_all,
                  dpf_gen, key_from_bytes)
from .rng import Sha256CounterRng

__version__ = "3.1.0"

__all__ = [
    "G", "conv", "convv_L", "convv_R", "permute", "permute_np",
    "DCFKey", "DPFKey", "dcf_eval", "dcf_eval_all", "dcf_gen",
    "dpf_eval", "dpf_eval_all", "dpf_gen", "key_from_bytes",
    "Sha256CounterRng",
]
