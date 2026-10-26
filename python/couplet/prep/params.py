"""Parameters of the Boolean preprocessing (DESIGN.md sections 1, 7.2, 7.5).

Constants
  * ``KAPPA``  = 128: computational parameter; number of base OTs per direction,
                      bit length of the Boolean MAC keys and of the COT rows.
  * ``PAD``    = 128: random padding rows of every correlated-OT extension call.
  * ``SIGMA_DEFAULT`` = 40: statistical parameter of the triple bucketing.
  * ``prg_dom`` / ``session_dom``: the layout of the 64-bit domain constants
    handed to ``couplet.hashes.ctr_prg`` and ``H_ccr`` by this package.  The
    top byte names the family (0x50 for counter-mode expansion, 0x48 for the
    correlation-robust hash; the convention of the C++ backend), so a hash
    call can never coincide with a PRG block, and no constant can equal an
    FSS PRG label word.

Bucket size (Lemma 4 of DESIGN.md 7.5).  ``ell * B`` leaky triples are
permuted and grouped into ``ell`` buckets of size ``B``.  An adversary that
attacks ``t`` leaky triples survives the consistency check with probability
``2^-t``, and a fixed bucket then consists of attacked triples only with
probability ``C(t, B) / C(ell B, B)``; a union bound over the buckets gives

    leak_bound(ell, B) = max_{B <= t <= ell B}  2^-t * ell * C(t, B) / C(ell B, B).

``bucket_size(ell, sigma)`` is the smallest ``B >= 2`` with
``leak_bound(ell, B) <= 2^-sigma``.  All arithmetic is exact (``fractions``).

Size of a triple request (``triple_request``; Lemma 3 of DESIGN.md 7.4 with
the constants of design/CHANGELOG.md, "Paper pass 1 corrections", C8 and I1;
the leaky-AND lemma and the backend theorem of the paper).  The
triple consistency check also has to catch a party whose key share ignores
the convention ``lsb(Delta^B_0) = 1, lsb(Delta^B_1) = 0``; that case passes
with probability at most ``(3/4)^(ell B)`` plus a negligible term, which is
below ``2^-127`` only from ``ell B = 306`` leaky triples on
(``MIN_LEAKY_TRIPLES``).  A request for fewer triples is therefore PADDED: the
protocol is run for ``ell' >= max(ell, 2)`` triples, the smallest such number
with ``ell' * bucket_size(ell', sigma) >= 306``, and the ``ell' - ell``
surplus triples are discarded after the check.  For sigma = 40 this concerns
requests of at most 43 triples (they run with ``ell' = 44``, ``B = 7``, 308
leaky triples), for sigma = 64 requests of at most 25 (``ell' = 26``,
``B = 12``, 312 leaky triples).  A run always has ``ell' >= 2`` buckets, which
is the range in which the closed form of Lemma 4 is stated.  Both parties
compute the same pair from the public ``(ell, sigma)``; the native code has
the same function (``cpp/include/couplet/params.hpp``).

Running this module as a script (``python -m couplet.prep.params``) writes the
parameter table for sigma in {40, 64} and ell = 2^10 .. 2^24 to
``artifact/experiments/figdata/bucket_params.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from fractions import Fraction

KAPPA = 128
PAD = 128
SIGMA_DEFAULT = 40

# Smallest number of leaky triples of one triple request: (3/4)^306 < 2^-127 (see the
# module docstring and ``triple_request``).
MIN_LEAKY_TRIPLES = 306

# Families (top byte of the Ascon domain constant).
DOM_CCR = 0x48  # every H_ccr call
DOM_PRG = 0x50  # every ctr_prg expansion

# Uses of ctr_prg inside this package (second byte of the domain constant).
USE_COT_PRG = 0x10      # column expansion of the OT extension; counter = extend call
USE_COT_CHI = 0x11      # coefficients of the OT-extension consistency check; counter = call
USE_BUCKET_PERM = 0x13  # sort keys of the public bucketing permutation
USE_TRIPLE_CHI = 0x14   # coefficients of the triple consistency check

# Label of the leaky-AND hash domain: session_dom(DOM_CCR, LABEL_LAAND + b"0" / b"1", sid).
LABEL_LAAND = b"aand/"

MAX_PRG_COUNTER = (1 << 40) - 1


def prg_dom(use: int, direction: int = 0, counter: int = 0) -> int:
    """Domain of a counter-mode expansion: ``0x50 << 56 | use << 48 | direction << 40 | counter``.

    ``use`` is in 1..255, so the low 56 bits are at least 2^48 (identifiers
    below 0x100 are left to the primitive layer) and the value can never equal
    a reserved FSS label word (all below 2^24).
    """
    if not 1 <= use <= 0xFF:
        raise ValueError("use must be in 1..255")
    if direction not in (0, 1):
        raise ValueError("direction must be 0 or 1")
    if not 0 <= counter <= MAX_PRG_COUNTER:
        raise ValueError("counter must fit in 40 bits")
    return (DOM_PRG << 56) | (use << 48) | (direction << 40) | counter


def session_dom(kind: int, label: bytes, sid: bytes) -> int:
    """Session-specific domain: ``kind`` in the top byte, 56 bits of a digest of (label, sid) below.

    The digest is the little-endian integer formed by the first 8 bytes of the
    framed ``SHA-256("Couplet-dom", label, sid)`` (every part preceded by its
    8-byte little-endian length, as in ``couplet.hashes.sha256``), reduced to
    its low 56 bits.
    """
    if not 1 <= kind <= 0xFF:
        raise ValueError("kind must be in 1..255")
    digest = hashlib.sha256()
    for part in (b"Couplet-dom", bytes(label), bytes(sid)):
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return (kind << 56) | (int.from_bytes(digest.digest()[:8], "little") & ((1 << 56) - 1))


# --------------------------------------------------------------------------- #
# Lemma 4: bucket size
# --------------------------------------------------------------------------- #

def leak_bound(ell: int, bucket: int) -> Fraction:
    """Exact value of ``max_t 2^-t * ell * C(t, B) / C(ell B, B)`` over ``B <= t <= ell B``.

    The term is unimodal in ``t`` (the ratio of consecutive terms is
    ``(t + 1) / (2 (t + 1 - B))``, which decreases), so the scan stops at the
    first strict decrease.
    """
    ell = int(ell)
    bucket = int(bucket)
    if ell < 1 or bucket < 1:
        raise ValueError("ell and the bucket size must be positive")
    total = ell * bucket
    denom = math.comb(total, bucket)
    best = Fraction(0)
    for t in range(bucket, total + 1):
        term = Fraction(ell * math.comb(t, bucket), denom << t)
        if term < best:
            break
        best = term
    return best


def bucket_size(ell: int, sigma: int = SIGMA_DEFAULT) -> int:
    """Smallest ``B >= 2`` with ``leak_bound(ell, B) <= 2^-sigma``."""
    ell = int(ell)
    sigma = int(sigma)
    if ell < 1:
        raise ValueError("ell must be positive")
    if sigma < 1:
        raise ValueError("sigma must be positive")
    target = Fraction(1, 1 << sigma)
    # B = sigma always suffices.  For ell = 1 the bound is 2^-B.  For ell >= 2:
    # 2^-t C(t, B) <= 1/2 (a central binomial term) and ell / C(ell B, B) <= ell^(1-B) <= 2^(1-B).
    for bucket in range(2, max(2, sigma) + 1):
        if leak_bound(ell, bucket) <= target:
            return bucket
    raise ArithmeticError("no bucket size found (this cannot happen)")


def triple_request(ell: int, sigma: int = SIGMA_DEFAULT) -> tuple[int, int]:
    """``(ell_run, B)``: the protocol run that serves a request for ``ell`` AND triples.

    ``ell_run`` is the smallest integer ``>= max(ell, 2)`` for which
    ``ell_run * bucket_size(ell_run, sigma) >= MIN_LEAKY_TRIPLES``, and
    ``B = bucket_size(ell_run, sigma)``.  The run generates, checks and combines
    ``ell_run * B`` leaky triples; the caller keeps ``ell`` of the ``ell_run``
    combined triples and discards the rest.  ``ell_run == ell`` whenever
    ``ell >= 2`` and ``ell * bucket_size(ell, sigma) >= MIN_LEAKY_TRIPLES``
    (for sigma = 40: from ``ell = 44`` on).  The search ends at ``ell_run = 153``
    at the latest, because ``B >= 2``.
    """
    ell = int(ell)
    if ell < 1:
        raise ValueError("ell must be positive")
    run = max(ell, 2)
    while True:
        bucket = bucket_size(run, sigma)
        if run * bucket >= MIN_LEAKY_TRIPLES:
            return run, bucket
        run += 1


def min_ell(bucket: int, sigma: int = SIGMA_DEFAULT) -> int:
    """Smallest ``ell`` for which bucket size ``bucket`` is sufficient at level ``sigma``.

    ``leak_bound(ell, B)`` decreases in ``ell`` for ``B >= 2``, so the answer
    is found by doubling and bisection.
    """
    bucket = int(bucket)
    if bucket < 2:
        raise ValueError("the bucket size must be at least 2")
    target = Fraction(1, 1 << int(sigma))
    if leak_bound(1, bucket) <= target:
        return 1
    lo, hi = 1, 2  # invariant: lo fails, hi is a candidate
    while leak_bound(hi, bucket) > target:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if leak_bound(mid, bucket) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def _log2_fraction(x: Fraction) -> float:
    return math.log2(x.numerator) - math.log2(x.denominator)


# --------------------------------------------------------------------------- #
# Parameter table
# --------------------------------------------------------------------------- #

TABLE_SIGMAS = (40, 64)
TABLE_LOG2_ELL = tuple(range(10, 25))
TABLE_BUCKETS = (2, 3, 4, 5, 6, 7, 8)


def bucket_table(sigmas=TABLE_SIGMAS, log2_ells=TABLE_LOG2_ELL) -> dict:
    """Bucket sizes for every (sigma, ell = 2^e) and the exact ell thresholds per B.

    ``model_bytes_per_and`` is the cost model of DESIGN.md 7.5 (96 bytes per
    leaky triple, both parties together); it is a formula, not a measurement.
    """
    rows = []
    for sigma in sigmas:
        for e in log2_ells:
            ell = 1 << e
            bucket = bucket_size(ell, sigma)
            rows.append({
                "sigma": sigma,
                "log2_ell": e,
                "ell": ell,
                "B": bucket,
                "log2_leak_bound": round(_log2_fraction(leak_bound(ell, bucket)), 3),
                "leaky_triples": ell * bucket,
                "model_bytes_per_and": 96 * bucket,
            })
    thresholds = {
        str(sigma): {str(b): min_ell(b, sigma) for b in TABLE_BUCKETS} for sigma in sigmas
    }
    return {
        "description": "Bucket size B of the authenticated AND triples (DESIGN.md 7.5, Lemma 4): "
                       "smallest B >= 2 with max_t 2^-t * ell * C(t,B) / C(ell*B,B) <= 2^-sigma. "
                       "Exact rational arithmetic; generated by couplet/prep/params.py.",
        "kappa": KAPPA,
        "pad_rows": PAD,
        "rows": rows,
        "min_ell_for_bucket": thresholds,
    }


def default_table_path() -> str:
    """``artifact/experiments/figdata/bucket_params.json`` relative to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "experiments", "figdata",
                                         "bucket_params.json"))


def write_table(path: str | None = None) -> str:
    """Write ``bucket_table()`` as JSON; returns the path written."""
    path = default_table_path() if path is None else path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(bucket_table(), fh, indent=2)
        fh.write("\n")
    return path


if __name__ == "__main__":
    out = write_table()
    table = bucket_table()
    for row in table["rows"]:
        print(f"sigma={row['sigma']:2d}  ell=2^{row['log2_ell']:<2d}  B={row['B']}  "
              f"log2(bound)={row['log2_leak_bound']:.2f}")
    for sigma, per_bucket in table["min_ell_for_bucket"].items():
        print(f"sigma={sigma}: smallest ell per B:", per_bucket)
    print("written:", out)
