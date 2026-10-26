"""Boolean preprocessing of the TinyOT-style backend (DESIGN.md sections 2.2, 7.1 - 7.5, 7.7).

Pipeline: base OT -> correlated OT extension -> authenticated bits -> leaky
authenticated AND triples -> bucketed, checked triples.  All protocol code is
party-local: a function or method runs ONE party and talks to the other
through a ``couplet.net.Channel`` endpoint.

Modules
  params   KAPPA, PAD, the Ascon domain layout, ``bucket_size`` (Lemma 4),
           ``triple_request`` (size of the run that serves a request: at least
           ``MIN_LEAKY_TRIPLES`` leaky triples) and the generator of the bucket
           parameter table.
  shares   data model: ``Shares`` (SPDZ-style ``<x>``), ``Bdoz`` / ``Keys``
           (BDOZ material), conversions, bit-vector wire encoding, ``Rand``.
  check    ``bool_check``: the batched MAC check of opened values (reusable by
           the circuit engine).
  baseot   Masny-Rindal base OT over Edwards25519, both directions in parallel.
  cotext   ``CotSender`` / ``CotReceiver``: IKNP extension with the per-column
           universal-hash consistency check.
  abits    ``AbitSource``: random authenticated bits and input masks from COT rows.
  aand     leaky AND, public bucketing permutation, bucket combination and the
           triple consistency check.
  real     ``RealPrep``: the whole pipeline behind one object.
  ideal    ``IdealDealer`` / ``IdealPrep``: trusted in-process dealer with the
           same interface (ideal mode; tests and online-only measurements).
"""

from .check import bool_check
from .ideal import IdealDealer, IdealPrep
from .params import KAPPA, MIN_LEAKY_TRIPLES, PAD, SIGMA_DEFAULT, bucket_size, triple_request
from .real import RealPrep
from .shares import Bdoz, Keys, Shares

__all__ = [
    "Bdoz", "IdealDealer", "IdealPrep", "KAPPA", "Keys", "MIN_LEAKY_TRIPLES", "PAD", "RealPrep",
    "SIGMA_DEFAULT", "Shares", "bool_check", "bucket_size", "triple_request",
]
