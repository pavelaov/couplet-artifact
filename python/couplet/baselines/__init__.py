"""Baselines for the evaluation (DESIGN.md section 9, Q1 and Q2).

Modules
  leaky_zhang    re-implementation of the full-domain, one-bit-leaky DPF
                 protocol Pi_DPF of Zhang et al. (2024, Figure 5) on
                 the authenticated-bit layer of ``couplet.prep``;
  semihonest_ds  semi-honest dealer-free DPF generation in the Doerner-shelat
                 style on the same tree (no MACs, no check, unchecked COT);
  costmodels     closed-form communication and round figures: derived from
                 the protocols, measured on our backend, or reported by other
                 papers (every entry names its source and its kind).

Both protocol baselines output shares of the WHOLE truth table (2^n entries)
with a GF(2^128) payload; they are comparison points, not part of the Couplet
protocol, and nothing else in the package depends on them.
"""

from . import costmodels, leaky_zhang, semihonest_ds
from .leaky_zhang import (BdozBits, FieldShare, FieldShares, LeakyDpfOutput, batch_check, pi_dpf,
                          run_pi_dpf)
from .semihonest_ds import PlainCot, SemiHonestDpfOutput, SemiHonestSession, semihonest_dpf

__all__ = [
    "BdozBits", "FieldShare", "FieldShares", "LeakyDpfOutput", "PlainCot", "SemiHonestDpfOutput",
    "SemiHonestSession", "batch_check", "costmodels", "leaky_zhang", "pi_dpf", "run_pi_dpf",
    "semihonest_dpf", "semihonest_ds",
]
