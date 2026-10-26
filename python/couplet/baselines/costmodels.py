"""Closed-form communication and round figures for the comparison tables (DESIGN.md 9, Q1 / Q2).

Every public function returns a dictionary (or a list / dictionary of such
dictionaries) with at least

    value     the number (or a short string where the source gives no number);
    unit      what it counts;
    source    where it comes from (paper with section / figure / table, a file
              of this artifact, or the formula it was derived with);
    kind      "measured"  taken from a run of this artifact,
              "reported"  stated by another paper (the number, or the formula
                          that was evaluated to obtain it),
              "derived"   computed here from the description of a protocol or
                          from other entries;
    reported  True exactly for kind == "reported".

Nothing here is a measurement unless its kind says so.  Numbers of other
papers were NOT re-measured; where a reported number could only be checked
against a secondary source this is said in the entry's ``note``.  Entries of
kind "derived" that build on reported constants list them under
``inputs``.

Groups
  Zhang et al. (2024)          message count of Figure 5 in the hybrid model,
                               the exact byte / round counts of our
                               instantiation (``leaky_zhang``), and what the
                               paper itself states;
  semi-honest baseline         exact counts of ``semihonest_ds``;
  Boyle et al. (CRYPTO 2020)   as characterised in Zhang et al., Section 2.3;
  generation inside 2PC        AND counts of our circuits (compiler output)
                               and of the same circuits with an AES-based
                               expander; bytes per AND gate of our backend;
  authenticated garbling       per-AND communication reported by Wang et al.
                               (CCS 2017), Katz et al. (CRYPTO 2018) and Cui
                               et al. (EUROCRYPT 2023);
  silent-OT variant            our backend with the COT rows charged at the
                               per-COT figure reported for Ferret.
"""

from __future__ import annotations

import json
import math
import os

from ..prep.cotext import CHECK_BYTES, SEED_BYTES, u_message_bytes
from ..prep.params import KAPPA, PAD, SIGMA_DEFAULT, bucket_size, triple_request
from . import leaky_zhang as lz
from . import semihonest_ds as sh

KINDS = ("measured", "reported", "derived")

SRC_ZHANG = ("W. Zhang, X. Guo, K. Yang, R. Zhu, Y. Yu, X. Wang, 'Efficient Actively Secure DPF "
             "and RAM-based 2PC with One-Bit Leakage', IEEE Symposium on Security and Privacy 2024 (ePrint version of "
             "2024-03-12)")
SRC_WRK17 = ("X. Wang, S. Ranellucci, J. Katz, 'Authenticated Garbling and Efficient Maliciously "
             "Secure Two-Party Computation', ACM CCS 2017")
SRC_KRRW18 = ("J. Katz, S. Ranellucci, M. Rosulek, X. Wang, 'Optimizing Authenticated Garbling "
              "for Faster Secure Two-Party Computation', CRYPTO 2018")
SRC_CWYY23 = ("H. Cui, X. Wang, K. Yang, Y. Yu, 'Actively Secure Half-Gates with Minimum Overhead "
              "under Duplex Networks', EUROCRYPT 2023 (ePrint 2023/278)")
SRC_FERRET = ("K. Yang, C. Weng, X. Lan, J. Zhang, X. Wang, 'Ferret: Fast Extension for "
              "Correlated OT with Small Communication', ACM CCS 2020 (ePrint 2020/924)")
SRC_THESIS = ("checked against the reproduction of the paper in X. Wang's PhD thesis "
              "(2018)")
SRC_DESIGN = "DESIGN.md v3.1"
SRC_CIRCUITS = ("compiler output of couplet.circuits_gen (experiments/figdata/circuit_stats.json, "
                "'per_level'; design/CHANGELOG.md, PY-A item 10)")
SRC_PREP = ("byte accounting of couplet.prep (design/CHANGELOG.md, PY-C item 10), asserted byte "
            "for byte by tests/test_prep.py")


def entry(value, unit: str, source: str, kind: str, **extra) -> dict:
    """Build one cost entry; ``reported`` is set from ``kind``."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    out = {"value": value, "unit": unit, "source": source, "kind": kind,
           "reported": kind == "reported"}
    out.update(extra)
    return out


def _check_n(n: int) -> int:
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    return n


# --------------------------------------------------------------------------- #
# Zhang et al., 2024
# --------------------------------------------------------------------------- #

def zhang_hybrid_bits(n: int, kappa: int = KAPPA) -> dict:
    """Bits exchanged by Pi_DPF in the (F_aBit, F_coin, F_com)-hybrid model, both parties.

    Per party: n correction-word shares of kappa bits (step 2), one final
    share of 2 kappa bits (step 3) and the share of the opened a (step 5).
    """
    n = _check_n(n)
    per_party = (n + 3) * kappa
    return entry(2 * per_party, "bits (both parties)",
                 SRC_ZHANG + ", Figure 5, steps 2, 3 and 5 (messages counted here; the paper "
                 "gives no closed form)", "derived",
                 formula="2 (n + 3) kappa", per_party=per_party,
                 note="excludes F_aBit, the two F_coin calls and BatchCheck (one F_coin call, "
                      "one committed kappa-bit value per party)")


def zhang_hybrid_rounds(n: int) -> dict:
    """Rounds of protocol messages of Pi_DPF in the hybrid model (calls to F_coin / F_com extra)."""
    n = _check_n(n)
    return entry(n + 2, "rounds (both parties send in every round)",
                 SRC_ZHANG + ", Figure 5: n rounds for step 2 (CW^(i) is needed before level "
                 "i + 1 can be hashed), one for step 3, one for the opening in step 5", "derived",
                 formula="n + 2",
                 note="plus 2 F_coin calls (W, chi) and BatchCheck (1 F_coin call, 1 commit-and-"
                      "open); F_aBit for the mask r can run in parallel with the first round")


def zhang_abit_demand(n: int, kappa: int = KAPPA) -> dict:
    """Authenticated bits per party used by one standalone key (inputs and the mask r)."""
    n = _check_n(n)
    return entry(n + 2 * kappa, "authenticated bits per party",
                 SRC_ZHANG + ", Figure 5 (inputs [[alpha^(i)]], <beta> packed from kappa bits; "
                 "step 4: <r> from kappa bits of each party)", "derived",
                 formula="n + 2 kappa",
                 note="alpha and beta taken as XOR shares input by the two parties; with a "
                      "silent COT such as Ferret the communication for these bits is below one "
                      "bit each, with our IKNP-style extension it is 128 bits each plus a "
                      "constant per call")


def zhang_instantiated_bytes(n: int, skip_unused_coin: bool = False) -> dict:
    """Exact bytes of our re-implementation (``leaky_zhang.run_pi_dpf``), both parties, no prep."""
    n = _check_n(n)
    phases = lz.bytes_per_party(n, skip_unused_coin)
    per_party = sum(phases.values())
    return entry(2 * per_party, "bytes (both parties)",
                 "couplet/baselines/leaky_zhang.py with hashes.coin_toss (96 bytes per party) "
                 "and hashes.committed_exchange (64 + 16 bytes per party); asserted by "
                 "tests/test_baselines.py", "derived",
                 formula="2 (16 n + ceil((n + 128) / 8) + " +
                         ("320" if skip_unused_coin else "416") + ")",
                 per_party=per_party, per_party_phases=phases,
                 note="input authentication (one bit per input bit) included; the generation of "
                      "the authenticated bits themselves (prep phases) is not")


def zhang_instantiated_rounds(n: int, skip_unused_coin: bool = False) -> dict:
    """Exact rounds of our re-implementation, without the prep phases."""
    n = _check_n(n)
    phases = lz.rounds(n, skip_unused_coin)
    return entry(sum(phases.values()), "rounds",
                 "couplet/baselines/leaky_zhang.py; asserted by tests/test_baselines.py",
                 "derived", formula="n + " + ("9" if skip_unused_coin else "11"),
                 phases=phases,
                 note="1 (inputs) + 2 (coin W) + n + 1 + 2 (coin chi) + 1 (open a) + BatchCheck "
                      "(2 for its coin unless skipped, 2 for commit and open)")


def zhang_abit_bytes(n: int, mode: str = "iknp_single", kappa: int = KAPPA) -> dict:
    """Communication for the n + 2 kappa authenticated bits per party of one key, both parties.

    ``mode``: "iknp_single"     one call of our checked extension per key (what
                                ``run_pi_dpf`` does and the benchmark measures);
              "iknp_amortized"  marginal cost when many keys share one
                                extension call (16 bytes per row and party);
              "ferret"          every bit charged at the per-COT figure
                                reported for Ferret-Reg (the original setting).
    """
    n = _check_n(n)
    bits = n + 2 * kappa
    if mode == "iknp_single":
        per_party = u_message_bytes(bits) + SEED_BYTES + CHECK_BYTES
        return entry(2 * per_party, "bytes (both parties)", SRC_PREP + "; one extend_both call "
                     "of n + 256 rows (rounded up to a multiple of 64) plus 128 padding rows, the "
                     "challenge seed and the check message", "derived", mode=mode,
                     per_party=per_party, authenticated_bits_per_party=bits)
    if mode == "iknp_amortized":
        return entry(2 * 16 * bits, "bytes (both parties)", SRC_PREP + "; 128 bits per row, "
                     "fixed costs of the extension call shared by many keys", "derived",
                     mode=mode, per_party=16 * bits, authenticated_bits_per_party=bits)
    if mode == "ferret":
        cot = ferret_bits_per_cot("reg")
        return entry(2 * bits * cot["value"] / 8, "bytes (both parties)",
                     "n + 2 kappa correlated OTs per direction at the reported Ferret-Reg figure",
                     "derived", mode=mode, authenticated_bits_per_party=bits, inputs=[cot],
                     note="what-if figure for the original instantiation; Ferret's one-time "
                          "setup is not included")
    raise ValueError("mode must be 'iknp_single', 'iknp_amortized' or 'ferret'")


def zhang_total_bytes(n: int, abit: str = "iknp_single", skip_unused_coin: bool = False) -> dict:
    """Pi_DPF as implemented here plus the authenticated bits under the chosen instantiation."""
    proto = zhang_instantiated_bytes(n, skip_unused_coin)
    bits = zhang_abit_bytes(n, abit)
    return entry(proto["value"] + bits["value"], "bytes (both parties)",
                 "zhang_instantiated_bytes + zhang_abit_bytes (" + abit + ")", "derived",
                 abit=abit, inputs=[proto, bits],
                 note="per key; the base OT of the IKNP variants (24,576 bytes per session) and "
                      "the setup of Ferret are not included")


def zhang_vs_bcg20(n: int) -> dict:
    """Ratio of the Boyle et al. figure (as characterised by Zhang et al.) to our Pi_DPF count."""
    theirs = bcg20_dpf_bytes(n)
    ours = zhang_total_bytes(n, "ferret")
    return entry(theirs["value"] / ours["value"], "x (BCG+20 as characterised / Pi_DPF)",
                 "bcg20_dpf_bytes / zhang_total_bytes(abit='ferret')", "derived",
                 inputs=[theirs, ours],
                 note="for orientation only: Zhang et al. report 'about 50x' at N = 2^20; our "
                      "count uses our own F_coin / F_com instantiations and message accounting")


def zhang_reported() -> list[dict]:
    """What the paper itself states about cost (no closed form is given there)."""
    return [
        entry(50, "x less communication than Boyle et al. (CRYPTO 2020) at N = 2^20",
              SRC_ZHANG + ", abstract ('about 50x less communication for a domain with 2^20 "
              "entries')", "reported"),
        entry("O(log N)", "rounds per DPF generation",
              SRC_ZHANG + ", Section 7.2 ('our protocol needs O(log N) rounds for an access "
              "operation to an array of size N') and Section 7.4", "reported"),
        entry("essentially semi-honest cost", "communication and computation",
              SRC_ZHANG + ", abstract and Section 2.3 ('almost as efficient as state-of-the-art "
              "semi-honest DPF protocols')", "reported"),
    ]


# --------------------------------------------------------------------------- #
# Semi-honest baseline (our implementation)
# --------------------------------------------------------------------------- #

def semihonest_instantiated_bytes(n: int) -> dict:
    """Exact bytes of ``semihonest_ds`` per key, both parties, without the base OT."""
    n = _check_n(n)
    phases = sh.bytes_per_party(n)
    per_party = sum(phases.values())
    return entry(2 * per_party, "bytes (both parties)",
                 "couplet/baselines/semihonest_ds.py; asserted by tests/test_baselines.py",
                 "derived", formula="2 (16 n + 1024 ceil(n / 64) + 32)", per_party=per_party,
                 per_party_phases=phases,
                 note="unchecked IKNP rows (128 bits per row, 64 rows per call at least); the "
                      "base OT (12,288 bytes per party, once per session) is not included")


def semihonest_instantiated_rounds(n: int) -> dict:
    """Exact rounds of ``semihonest_ds`` per key, without the base OT."""
    n = _check_n(n)
    return entry(n + 2, "rounds", "couplet/baselines/semihonest_ds.py; asserted by "
                 "tests/test_baselines.py", "derived", formula="n + 2", phases=sh.rounds(n))


# --------------------------------------------------------------------------- #
# Boyle et al., CRYPTO 2020, as characterised by Zhang et al.
# --------------------------------------------------------------------------- #

def mascot_multiplication_bytes() -> dict:
    """Communication of one MASCOT field multiplication as quoted by Zhang et al."""
    return entry(33_000, "bytes per field multiplication",
                 SRC_ZHANG + ", Section 2.3 ('MASCOT [KOS16] requires about 33,000 bytes of "
                 "communication to compute one such multiplication even without counting the "
                 "cost of underlying oblivious transfer')", "reported")


def bcg20_dpf_bytes(n: int, kappa: int = KAPPA) -> dict:
    """Generic-2PC-based malicious DPF of Boyle et al. (CRYPTO 2020) as described in Zhang et al.

    Three field multiplications inside generic malicious 2PC (MASCOT) plus the
    rest of the protocol.
    """
    n = _check_n(n)
    mult = mascot_multiplication_bytes()
    rest_bits = 2 * (n + 1) * kappa
    return entry(3 * mult["value"] + rest_bits / 8, "bytes",
                 SRC_ZHANG + ", Section 2.3 ('needs to compute three field multiplications in "
                 "2PC'; 'the rest part of this protocol only needs 2(log N + 1) [kappa] bits of "
                 "communication')", "reported",
                 formula="3 * 33000 + 2 (n + 1) kappa / 8", inputs=[mult],
                 multiplications=3, rest_bits=rest_bits,
                 note="characterisation by Zhang et al., not a measurement and not a figure of "
                      "Boyle et al.; excludes the oblivious transfers under MASCOT. In the text "
                      "extraction available to us the factor after '2(log N + 1)' is not "
                      "legible (Greek letters were dropped); it is read as the security "
                      "parameter. The rest term is below 1 % of the total for n <= 30.")


# --------------------------------------------------------------------------- #
# Key generation inside generic 2PC: AND counts
# --------------------------------------------------------------------------- #

AES_SBOX_ANDS = 32
AES_ROUNDS = 10
AES_SBOXES_PER_ROUND = 16
ASCON_PRG_ANDS = 3584          # one PRG call, secret seed and constant label (compiler output)
ASCON_CONV_ANDS = {128: 3584, 96: 3520}  # Conv / ConvV call for w = k + s (dead gates removed)

# (family, k, s, beta_mode) -> (AND gates per level, fixed AND gates); ripple adders, XOR index.
CIRCUIT_AFFINE = {
    ("dpf", 64, 64, "public_one"): (7555, 7799),
    ("dpf", 64, 64, "secret"): (7555, 16117),
    ("dpf", 32, 64, "public_one"): (7555, 7479),
    ("dpf", 32, 64, "secret"): (7555, 11605),
    ("dcf", 64, 64, "public_one"): (23736, 7796),
    ("dcf", 64, 64, "secret"): (23863, 16113),
    ("dcf", 32, 64, "public_one"): (23032, 7476),
    ("dcf", 32, 64, "secret"): (23095, 11601),
}
AND_DEPTH_AFFINE = {64: (14, 138), 32: (14, 106)}  # k -> (per level, fixed), ripple adders


def circuit_stats_path() -> str:
    """``artifact/experiments/figdata/circuit_stats.json`` relative to this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "experiments", "figdata",
                                         "circuit_stats.json"))


def load_circuit_affine(path: str | None = None) -> dict:
    """The affine AND-count forms of the compiled circuits, read from ``circuit_stats.json``.

    Returns {(family, k, s, beta_mode): (per level, fixed)} for the ripple-adder
    rows; used by the tests to confirm that ``CIRCUIT_AFFINE`` is up to date.
    """
    with open(circuit_stats_path() if path is None else path, encoding="utf-8") as fh:
        stats = json.load(fh)
    out = {}
    for row in stats["per_level"]:
        if row["adder"] == "ripple":
            out[(row["kind"], row["k"], row["s"], row["beta_mode"])] = (
                row["n_and"]["per_level"], row["n_and"]["fixed"])
    return out


def aes_call_ands() -> dict:
    """AND gates of one fixed-key AES-128 call with the 32-AND S-box."""
    return entry(AES_ROUNDS * AES_SBOXES_PER_ROUND * AES_SBOX_ANDS, "AND gates per AES-128 call",
                 "10 rounds x 16 S-boxes x 32 AND gates (S-box circuit of Boyar and Peralta with "
                 "32 AND gates); ShiftRows, MixColumns and AddRoundKey are linear; the key is "
                 "public and fixed, so the key schedule is outside the circuit", "derived",
                 formula="10 * 16 * 32")


def aes_expander_ands_per_call(output_bits: int = 256) -> dict:
    """AND gates of one AES-based expander call producing ``output_bits`` bits (2 calls for G)."""
    calls = max(1, math.ceil(int(output_bits) / 128))
    one = aes_call_ands()
    return entry(calls * one["value"], "AND gates per expander call",
                 f"{calls} fixed-key AES-128 call(s) of {one['value']} AND gates; G(s) needs two "
                 "128-bit children (the control bits are taken from the children), Conv needs "
                 "2 w bits", "derived", calls=calls, inputs=[one])


def gen_circuit_ands(n: int, family: str = "dpf", k: int = 64, s: int = 64,
                     beta_mode: str = "public_one", expander: str = "ascon") -> dict:
    """AND gates of the path-only generation circuit C_dpf / C_dcf (DESIGN.md section 4).

    ``expander="ascon"``: the compiler's affine form.  ``expander="aes"``: the
    same circuit with every PRG call replaced by fixed-key AES calls (the
    "naive Gen-in-2PC" row of Q2); nothing but the expander changes.
    """
    n = _check_n(n)
    key = (family, int(k), int(s), beta_mode)
    if key not in CIRCUIT_AFFINE:
        raise ValueError(f"no compiled circuit figures for {key}")
    per_level, fixed = CIRCUIT_AFFINE[key]
    depth = AND_DEPTH_AFFINE[int(k)]
    if expander == "ascon":
        return entry(per_level * n + fixed, "AND gates", SRC_CIRCUITS, "derived",
                     formula=f"{per_level} n + {fixed}", per_level=per_level, fixed=fixed,
                     and_depth=depth[0] * n + depth[1], expander="ascon",
                     note="exact for n in {8, 12, 16, 20, 24, 32} according to the compiler; "
                          "'derived' because it is the affine form, the compiler output itself "
                          "is in circuit_stats.json")
    if expander != "aes":
        raise ValueError("expander must be 'ascon' or 'aes'")
    w = int(k) + int(s)
    g_calls = 2                                   # G(s0), G(s1) per level
    conv_level = 4 if family == "dcf" else 0      # ConvV_L / ConvV_R for both seeds per level
    conv_fixed = 2                                # Conv(s0), Conv(s1) at the leaf
    aes_g = aes_expander_ands_per_call(256)["value"]
    aes_conv = aes_expander_ands_per_call(2 * w)["value"]
    ascon_conv = ASCON_CONV_ANDS[w]
    per_level_aes = (per_level - g_calls * ASCON_PRG_ANDS - conv_level * ascon_conv
                     + g_calls * aes_g + conv_level * aes_conv)
    fixed_aes = fixed - conv_fixed * ascon_conv + conv_fixed * aes_conv
    return entry(per_level_aes * n + fixed_aes, "AND gates",
                 SRC_CIRCUITS + "; every Ascon PRG call (3,584 AND gates, Conv 3,584 or 3,520) "
                 "replaced by fixed-key AES-128 calls of 5,120 AND gates each (2 per G, "
                 "ceil(2 w / 128) per Conv)", "derived",
                 formula=f"{per_level_aes} n + {fixed_aes}", per_level=per_level_aes,
                 fixed=fixed_aes, expander="aes",
                 note="not compiled and not run: a derived figure (to be printed in red, "
                      "DESIGN.md 9, Q2). AND depth is not modelled.")


# --------------------------------------------------------------------------- #
# Our backend: bytes per AND gate
# --------------------------------------------------------------------------- #

ONLINE_BITS_PER_AND = 4      # d and e, one bit from each party (DESIGN.md 7.5)
_MAX_CHUNK_ROWS = 1 << 20    # rows per extension call (prep.abits.DEFAULT_BATCH_ROWS)
_COIN_BYTES = 96
_BOOLCHECK_BYTES = 96


def backend_bytes_per_and(ell: int, sigma: int = SIGMA_DEFAULT) -> dict:
    """Preprocessing bytes per final AND triple of ``RealPrep.and_triples(ell)``, both parties.

    The exact accounting of the Python backend: per party, 16 (rows + 128) +
    16 + 2064 bytes per extension chunk for 2 ell B + 128 rows, 16 ell B bytes
    of leaky-AND transfers, 96 + ceil(ell (B - 1) / 8) for bucketing and 208
    for the check.  It reproduces the measured totals (385.0 bytes at
    ell = 50,000 and 288.6 at ell = 300,000 in the PY-C run).  A request
    below 306 leaky triples is padded (``prep.params.triple_request``): the
    run then has more than ``ell`` triples, and the bytes of the whole run are
    charged to the ``ell`` triples that were asked for.
    """
    ell = int(ell)
    if ell < 1:
        raise ValueError("ell must be positive")
    run, bucket = triple_request(ell, sigma)      # run == ell unless the request is padded
    rows = 2 * run * bucket + PAD
    cot = 0
    for start in range(0, rows, _MAX_CHUNK_ROWS):
        cot += u_message_bytes(min(_MAX_CHUNK_ROWS, rows - start)) + SEED_BYTES + CHECK_BYTES
    phases = {"cotext": cot, "laand": 16 * run * bucket,
              "bucket": _COIN_BYTES + (run * (bucket - 1) + 7) // 8,
              "check": _COIN_BYTES + 16 + _BOOLCHECK_BYTES}
    per_party = sum(phases.values())
    return entry(2 * per_party / ell, "bytes per AND triple (both parties)", SRC_PREP, "derived",
                 bucket=bucket, sigma=int(sigma), ell=ell, per_party_phases=phases,
                 model_96B=96 * bucket,
                 note="preprocessing only; base OT (24,576 bytes per session) and the 4 online "
                      "bits per AND gate are not included")


def gen_in_2pc_bytes(n: int, *, family: str = "dpf", k: int = 64, s: int = 64,
                     beta_mode: str = "public_one", expander: str = "ascon",
                     bytes_per_and: dict | None = None, sigma: int = SIGMA_DEFAULT) -> dict:
    """Communication of one key generation inside our backend: AND gates x bytes per AND.

    ``bytes_per_and`` is an entry (for instance a measured one from the
    benchmark); by default the exact accounting ``backend_bytes_per_and`` for
    a batch of exactly this circuit's AND gates is used.
    """
    ands = gen_circuit_ands(n, family, k, s, beta_mode, expander)
    per_and = bytes_per_and
    if per_and is None:
        per_and = backend_bytes_per_and(ands["value"], sigma)
    value = ands["value"] * (per_and["value"] + ONLINE_BITS_PER_AND / 8)
    return entry(value, "bytes (both parties)",
                 "AND gates x (preprocessing bytes per AND + 4 online bits); " + SRC_DESIGN +
                 " section 7.5", "derived", and_gates=ands["value"], inputs=[ands, per_and],
                 expander=expander,
                 note="one key per bucketing population; inputs, output openings, the MAC checks "
                      "and the base OT are not included (they do not grow with the circuit)")


# --------------------------------------------------------------------------- #
# Authenticated garbling: per-AND communication reported in the literature
# --------------------------------------------------------------------------- #

def authenticated_garbling_per_and(kappa: int = KAPPA, rho: int = 40, bucket: int = 3) -> dict:
    """Per-AND-gate communication of authenticated-garbling protocols, as reported.

    Returns {name: entry}; every entry has kind "reported" (``reported`` is
    True), its ``formula`` as printed in the source and the value of that
    formula for the given ``kappa``, ``rho`` and bucket size ``bucket``.
    None of these protocols was run or re-measured here.
    """
    b = int(bucket)
    out = {}
    out["WRK17/function-independent"] = entry(
        (5 * kappa + 3 * rho) * b, "bits per AND triple",
        SRC_WRK17 + ", description of the improved TinyOT protocol ('requires only (5 kappa + "
        "3 rho) B bits of communication per AND triple'); " + SRC_THESIS + ", Section 3.3",
        "reported", formula="(5 kappa + 3 rho) B", phase="function-independent")
    out["WRK17/function-dependent"] = entry(
        3 * kappa + 4 * rho, "bits per AND gate",
        SRC_WRK17 + ", communication complexity ('we need to send 3 kappa + 4 rho bits per "
        "gate'); " + SRC_THESIS + ", Section 3.6.5", "reported", formula="3 kappa + 4 rho",
        phase="function-dependent")
    krrw = (SRC_KRRW18 + ", communication complexity next to Table 1 ('our first version has "
            "function dependent cost of 3 kappa per gate, and function independent cost of "
            "(4B - 2) kappa + (3B - 1) rho per gate; our second version has a function dependent "
            "cost of 2 kappa per gate, and a function independent cost of (4B + 2) kappa + "
            "(3B + 2) rho per gate'); " + SRC_THESIS + ", Section 5.5.1")
    one_way = ("the formula reproduces the ONE-WAY column of their Table 1 for AES (6,800 AND "
               "gates, B = 4); the two-way figure of the function-independent phase is twice it")
    out["KRRW18-v1/function-independent"] = entry(
        (4 * b - 2) * kappa + (3 * b - 1) * rho, "bits per AND gate (one-way)", krrw, "reported",
        formula="(4B - 2) kappa + (3B - 1) rho", phase="function-independent", note=one_way)
    out["KRRW18-v1/function-dependent"] = entry(
        3 * kappa, "bits per AND gate", krrw, "reported", formula="3 kappa",
        phase="function-dependent")
    out["KRRW18-v2/function-independent"] = entry(
        (4 * b + 2) * kappa + (3 * b + 2) * rho, "bits per AND gate (one-way)", krrw, "reported",
        formula="(4B + 2) kappa + (3B + 2) rho", phase="function-independent", note=one_way)
    out["KRRW18-v2/function-dependent"] = entry(
        2 * kappa, "bits per AND gate", krrw, "reported", formula="2 kappa",
        phase="function-dependent")
    cwyy_note = ("as stated in the abstract; these figures presuppose correlated-OT "
                 "preprocessing with sublinear communication (PCG-style), whose cost is not part "
                 "of the per-gate count; check the paper's comparison table before quoting more "
                 "than these two formulas")
    out["CWYY23/one-way"] = entry(
        2 * kappa + 5, "bits per AND gate (one-way)",
        SRC_CWYY23 + ", abstract ('one-way communication of 2 kappa + 5 bits per AND gate (for "
        "kappa-bit computational security and any statistical security)')", "reported",
        formula="2 kappa + 5", note=cwyy_note)
    out["CWYY23/two-way"] = entry(
        2 * kappa + rho + 5, "bits per AND gate (total, both directions)",
        SRC_CWYY23 + ", abstract ('total communication of 2 kappa + rho + 5 bits per AND gate "
        "(for rho-bit statistical security)')", "reported", formula="2 kappa + rho + 5",
        note=cwyy_note)
    return out


def authenticated_garbling_aes_reported() -> list[dict]:
    """Reported communication for one AES evaluation (6,800 AND gates), single execution.

    Rows of Table 1 of Katz et al. (CRYPTO 2018), which also restates the
    figures of Wang et al. (CCS 2017).  One-way = the larger of the two
    directions, two-way = the sum; megabytes except the online phase.
    """
    src = (SRC_KRRW18 + ", Table 1 ('Communication complexity of different protocols for "
           "evaluating AES, rounded to two significant figures'), single execution; "
           + SRC_THESIS + ", Table 5.1")
    rows = [
        ("WRK17", 2.9, 0.57, 4.9, 3.4, 5.7, 0.57, 6.0, 6.3),
        ("KRRW18-v1", 1.9, 0.33, 5.0, 2.2, 3.8, 0.33, 5.0, 4.2),
        ("KRRW18-v2", 2.5, 0.22, 5.0, 2.7, 4.9, 0.22, 5.0, 5.1),
    ]
    out = []
    for name, ind1, dep1, onl1, tot1, ind2, dep2, onl2, tot2 in rows:
        out.append(entry(tot2, "MB two-way total for one AES evaluation", src, "reported",
                         protocol=name, and_gates=6800,
                         one_way={"function_independent_MB": ind1, "function_dependent_MB": dep1,
                                  "online_KB": onl1, "total_MB": tot1},
                         two_way={"function_independent_MB": ind2, "function_dependent_MB": dep2,
                                  "online_KB": onl2, "total_MB": tot2}))
    return out


def authenticated_garbling_bytes(and_gates: int, protocol: str = "KRRW18-v1", *,
                                 kappa: int = KAPPA, rho: int = 40, bucket: int = 3) -> dict:
    """A circuit of ``and_gates`` AND gates charged at a reported per-AND figure (both phases).

    ``protocol`` is "WRK17", "KRRW18-v1", "KRRW18-v2" (function-independent
    plus function-dependent phase, formulas as reported, one direction) or
    "CWYY23" (the two-way total of the abstract).  Input and output terms are
    ignored.  The result is an estimate derived from reported formulas.
    """
    table = authenticated_garbling_per_and(kappa, rho, bucket)
    if protocol == "CWYY23":
        parts = [table["CWYY23/two-way"]]
    elif protocol in ("WRK17", "KRRW18-v1", "KRRW18-v2"):
        parts = [table[protocol + "/function-independent"], table[protocol + "/function-dependent"]]
    else:
        raise ValueError("unknown protocol")
    bits = sum(part["value"] for part in parts)
    return entry(int(and_gates) * bits / 8, "bytes", "AND gates x reported bits per AND gate (" +
                 " + ".join(part["formula"] for part in parts) + ")", "derived",
                 protocol=protocol, and_gates=int(and_gates), bits_per_and=bits, inputs=parts,
                 note="estimate from reported formulas, not a measurement, with bucket size "
                      "B = %d. For KRRW18 the function-independent formula reproduces the "
                      "one-way column of their AES table (the two-way figure is twice it); for "
                      "WRK17 the source does not say which direction the formula counts (their "
                      "AES figures are 2.9 MB one-way and 5.7 MB two-way for 6,800 AND gates)."
                      % bucket)


# --------------------------------------------------------------------------- #
# Silent-OT variant of our backend
# --------------------------------------------------------------------------- #

FERRET_BITS_PER_COT = {"reg": 0.44, "uni": 0.73}


def ferret_bits_per_cot(variant: str = "reg") -> dict:
    """Communication per correlated OT reported for Ferret (regular or uniform LPN noise)."""
    if variant not in FERRET_BITS_PER_COT:
        raise ValueError("variant must be 'reg' or 'uni'")
    name = {"reg": "Ferret-Reg", "uni": "Ferret-Uni"}[variant]
    return entry(FERRET_BITS_PER_COT[variant], "bits per COT",
                 SRC_FERRET + f", performance table, communication per COT of {name}",
                 "reported", variant=name,
                 note="the original PDF could not be opened from the machine of this run; the "
                      "value was confirmed only against a secondary summary of the paper "
                      "(0.44 bits for Ferret-Reg, 0.73 bits for Ferret-Uni) and must be checked "
                      "against the paper's table before it is printed; excludes Ferret's "
                      "one-time setup")


def silent_backend_bytes_per_and(bucket: int = 3, variant: str = "reg",
                                 kappa: int = KAPPA) -> dict:
    """Our backend with the COT rows charged at Ferret's reported per-COT communication.

    Per final AND gate (DESIGN.md 7.5): B leaky triples, each using 4 COT rows
    (charged at the reported figure instead of kappa bits) and 2 transfer
    messages of kappa bits, plus 2 (B - 1) combining bits and 4 online bits.
    """
    b = int(bucket)
    if b < 1:
        raise ValueError("the bucket size must be positive")
    cot = ferret_bits_per_cot(variant)
    bits = b * (4 * cot["value"] + 2 * kappa) + 2 * (b - 1) + ONLINE_BITS_PER_AND
    iknp_bits = b * (4 * kappa + 2 * kappa) + 2 * (b - 1) + ONLINE_BITS_PER_AND
    return entry(bits / 8, "bytes per AND gate (both parties)",
                 SRC_DESIGN + " section 7.5 (4 COT rows and 2 kappa-bit transfers per leaky "
                 "triple, B leaky triples per AND gate) with each COT row charged at the "
                 "reported Ferret figure", "derived",
                 formula="(B (4 c + 2 kappa) + 2 (B - 1) + 4) / 8",
                 bucket=b, inputs=[cot], same_model_with_iknp_rows=iknp_bits / 8,
                 note="a what-if figure: no silent OT is implemented in this artifact; the "
                      "computation of an LPN-based extension and its setup are not modelled")


# --------------------------------------------------------------------------- #
# One table for Q1 / Q2
# --------------------------------------------------------------------------- #

def comparison_rows(ns=(8, 12, 16, 20), *, k: int = 64, s: int = 64,
                    beta_mode: str = "public_one", measured_bytes_per_and: dict | None = None,
                    rho: int = 40) -> list[dict]:
    """All model rows for the given domain sizes (used by ``bench/bench_baselines.py``).

    ``measured_bytes_per_and`` maps a bucket size B to a measured entry
    (bytes per AND triple); where it has the bucket size a circuit needs, the
    measured figure is used and named in the row, otherwise the exact
    accounting ``backend_bytes_per_and``.
    """
    measured = measured_bytes_per_and or {}
    rows = []
    for n in ns:
        row = {"n": int(n)}
        for expander, label in (("ascon", "this_work_gen_in_2pc"), ("aes", "naive_aes_gen_in_2pc")):
            ands = gen_circuit_ands(n, "dpf", k, s, beta_mode, expander)
            bucket = bucket_size(ands["value"], SIGMA_DEFAULT)
            per_and = measured.get(bucket)
            row[label] = {"and_gates": ands,
                          "bucket": bucket,
                          "bytes": gen_in_2pc_bytes(n, k=k, s=s, beta_mode=beta_mode,
                                                    expander=expander, bytes_per_and=per_and)}
            row[label + "_silent_ot"] = entry(
                ands["value"] * silent_backend_bytes_per_and(bucket)["value"],
                "bytes (both parties)", "AND gates x silent_backend_bytes_per_and(B)", "derived",
                bucket=bucket, inputs=[silent_backend_bytes_per_and(bucket)])
            if expander == "ascon":
                for proto in ("WRK17", "KRRW18-v1", "KRRW18-v2", "CWYY23"):
                    row["authenticated_garbling_" + proto] = authenticated_garbling_bytes(
                        ands["value"], proto, rho=rho, bucket=bucket)
        row["zhang_hybrid_bits"] = zhang_hybrid_bits(n)
        row["zhang_hybrid_rounds"] = zhang_hybrid_rounds(n)
        row["zhang_abit_demand"] = zhang_abit_demand(n)
        row["zhang_instantiated_bytes"] = zhang_instantiated_bytes(n)
        row["zhang_instantiated_rounds"] = zhang_instantiated_rounds(n)
        row["zhang_total_bytes"] = {mode: zhang_total_bytes(n, mode)
                                    for mode in ("iknp_single", "iknp_amortized", "ferret")}
        row["zhang_vs_bcg20"] = zhang_vs_bcg20(n)
        row["semihonest_instantiated_bytes"] = semihonest_instantiated_bytes(n)
        row["semihonest_instantiated_rounds"] = semihonest_instantiated_rounds(n)
        row["bcg20_generic_2pc"] = bcg20_dpf_bytes(n)
        rows.append(row)
    return rows
