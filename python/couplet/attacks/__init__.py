"""Adversarial regression for Pi_Gen and Pi_AuthEval (DESIGN.md section 9, appendix).

  adversaries   malicious-party strategies A1 - A15 (test hooks of the lower
                layers, a tampering channel wrapper, a deviating arithmetic
                checker), the selective-failure distinguishing experiment and
                its positive control, and the aggregation of the results.
  run_all       runs every experiment, prints a table and writes
                ``artifact/experiments/raw/attacks.json``.

Run from ``artifact/python``:  ``python -m couplet.attacks.run_all``  (or
``python couplet/attacks/run_all.py``).
"""

from .adversaries import (ATTACKS, DISTINGUISHERS, CheatingChecker, TamperChannel,
                          run_distinguisher, run_trials, summarise, summarise_distinguisher,
                          two_proportion_z)

__all__ = [
    "ATTACKS", "DISTINGUISHERS", "CheatingChecker", "TamperChannel", "run_distinguisher",
    "run_trials", "summarise", "summarise_distinguisher", "two_proportion_z",
]
