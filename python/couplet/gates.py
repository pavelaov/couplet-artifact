"""Local evaluation of the catalogue gates (DESIGN.md section 6, Theorem 4).

A gate is used in three steps (section 5.3): ``Gate`` generates one-time
material with ``couplet.protocol.gen_gate`` (an offset ``r'`` uniform in
``Z_{2^(n+s)}``, ``r = r' mod 2^n``, and FSS keys at indices that depend on
``r``); ``Mask`` opens ``x_hat = (x + r) mod 2^n``; ``Apply`` is the LOCAL
computation of this module: from its ``GateMaterial`` and the public ``x_hat``
each party derives its share of ``[[G(x)]]`` over ``Z_{2^w}``, ``w = k + s``.
No function here sends a message.

Notation: ``N = 2^n``; ``D_c`` is the payload-1 DCF at index ``(r + c) mod N``
evaluated at ``x_hat`` (``[x_hat < (r + c) mod N]``), ``E_c`` the payload-``r``
DCF, ``w_c = [r + c >= N]``.  Every function returns this party's
``couplet.spdz2k.AShare`` with the exactness flag of the table of section 6:

    function                      keys used                 value                           exact
    zero_test(g, x_hat)           DPF at r, payload 1       [x = 0]                         yes
    equality(g, x_hat, c)         same DPF                  [x = c] = DPF(x_hat - c)        yes
    table(g, x_hat, T)            same DPF                  T[x] = sum_u T[(x_hat-u) mod N] y(u)   yes
    comparison(g, x_hat, c)       DCFs at c and 0           [x < c] = D_c - D_0 + w_c       yes
    interval(g, x_hat, p, q)      DCFs at q+1 and p         [p <= x <= q]                   yes
    multi_interval(g, x_hat, E)   one DCF per endpoint      [e_j <= x < e_(j+1)], a list    yes
    nonneg(g, x_hat)              DCFs at N/2 and 0         [x < N/2]                       yes
    msb(g, x_hat) (= sign)        same                      [x >= N/2] = 1 - [x < N/2]      yes
    relu_large(g, x_hat)          payload-1 and payload-r   x [x < N/2] mod 2^k, n = k      no
                                  DCFs at N/2 and 0
    linear_spline(g, x_hat, E, P) payload-1 and payload-r   a_j x + b_j mod 2^k on          no
                                  DCFs at every endpoint    [e_j, e_(j+1)), n = k

The comparison identity ``[x < c] = [x_hat < (r+c) mod N] - [x_hat < r] + w_c``
holds over the integers for every ``c`` in ``[0, N]``, so indicator outputs
are exact; ``relu_large`` and ``linear_spline`` use ``x = x_hat - r`` modulo
``2^k`` and are correct modulo ``2^k`` only (they must be opened with an
output mask, change E).  ``msb`` adds the public constant 1 and therefore
needs this party's ``Delta_b``.  ReLU, sign and splines on a small domain are
table gates: ``relu_table``, ``sign_table``, ``spline_table`` and
``function_table`` build the public tables.

``g`` is a ``couplet.protocol.GateMaterial`` or a ``GateEval`` wrapping one.
``GateEval`` caches key evaluations; ``GateEval(material, full_domain=True)``
evaluates every key once on the whole domain (``fss.dpf_eval_all`` /
``dcf_eval_all``) and then answers every ``x_hat`` by a look-up, which is what
the exhaustive tests use; the default evaluates the DCFs at the single point
``x_hat``.  The table gate always expands the DPF on the whole domain and is
``Theta(N)`` work: one rotation of the table and one inner product per plane.

Descriptors.  ``<gate>_descriptor(...)`` returns the
``circuits_gen.gate_descriptor`` that ``protocol.gen_gate`` needs for the gate
(``auth_input=True`` if the gate input is an authenticated wire, i.e. the
offset is shared as ``[[r']]`` over ``Z_{2^(n+s)}``).  ``GateSpec`` bundles a
gate kind with its public parameters: ``descriptor()`` for generation,
``evaluate(g, x_hat)`` for ``Apply`` and ``plain(x)`` for the plaintext value
(the reference of the tests).  Constructors: ``GateSpec.zero_test``,
``.equality``, ``.table``, ``.comparison``, ``.interval``, ``.multi_interval``,
``.nonneg``, ``.msb``, ``.relu_large``, ``.linear_spline``, and
``.offset_only`` (material that only masks a value; no function).

Not provided (section 6): products of two FSS outputs, payloads ``r^j`` with
``j >= 2``, vector payloads.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from . import circuits_gen as cg
from . import fss
from .protocol import GateMaterial
from .spdz2k import AShare

__all__ = [
    "GateEval", "GateSpec", "as_eval", "comparison", "comparison_descriptor", "deal_gate", "equality",
    "equality_descriptor", "function_table", "interval", "interval_descriptor", "linear_spline",
    "linear_spline_descriptor", "msb", "msb_descriptor", "multi_interval",
    "multi_interval_descriptor", "nonneg", "offset_descriptor", "plain_linear_spline",
    "plain_relu", "relu_large", "relu_large_descriptor", "relu_table", "sign", "sign_table",
    "spline_table", "table", "table_descriptor", "zero_test", "zero_test_descriptor",
]

ONE = "one"
R = "r"


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


# --------------------------------------------------------------------------- #
# Evaluator: one party's gate material plus a cache of key evaluations
# --------------------------------------------------------------------------- #

class GateEval:
    """Party-local evaluator bound to one ``GateMaterial``.

    ``delta_share`` (this party's ``Delta_b``) is needed only by gates that add
    a public constant (``msb``).  ``full_domain=True`` evaluates each key once
    on the whole domain and serves every later request from the table.
    """

    def __init__(self, material: GateMaterial, delta_share=None, *, full_domain: bool = False) -> None:
        if not isinstance(material, GateMaterial):
            raise ValueError("expected a couplet.protocol.GateMaterial")
        self.material = material
        self.role = material.role
        self.n, self.k, self.s = material.n, material.k, material.s
        self.width = self.k + self.s
        self.N = 1 << self.n
        self.delta_share = None if delta_share is None else int(delta_share)
        self.full_domain = bool(full_domain)
        self._index: dict = {}
        for i, (c, payload) in enumerate(material.thresholds):
            self._index.setdefault((int(c), payload), i)
        self._dcf_points: dict = {}
        self._dcf_tables: dict = {}
        self._dpf_table = None

    # -- arguments ------------------------------------------------------------ #

    def x_hat(self, x_hat) -> int:
        if not _is_int(x_hat) or not 0 <= int(x_hat) < self.N:
            raise ValueError("x_hat must be an integer in [0, 2^n)")
        return int(x_hat)

    def threshold(self, c: int, payload: str = ONE) -> int:
        """Index of the threshold ``(c, payload)`` in the material."""
        try:
            return self._index[(int(c), payload)]
        except KeyError:
            raise ValueError(f"the gate material has no threshold ({c}, {payload!r})") from None

    # -- shares held in the material ------------------------------------------- #

    def w(self, c: int, payload: str = ONE) -> AShare:
        """``[[w_c]]`` of the threshold ``(c, payload)``; ``w_c = [r + c >= N]``."""
        value, tag = self.material.w_shares[self.threshold(c, payload)]
        return AShare(value, tag, self.width, True)

    def rw(self, c: int) -> AShare:
        """``[[r w_c]]`` of the payload-``r`` threshold at ``c``."""
        value, tag = self.material.rw_shares[self.threshold(c, R)]
        return AShare(value, tag, self.width, True)

    # -- key evaluations -------------------------------------------------------- #

    def dcf(self, c: int, payload: str, x_hat: int) -> AShare:
        """Share of the DCF of threshold ``(c, payload)`` at ``x_hat`` (exact)."""
        i = self.threshold(c, payload)
        if self.full_domain:
            tab = self._dcf_tables.get(i)
            if tab is None:
                tab = self._dcf_tables[i] = fss.dcf_eval_all(self.role, self.material.dcf_keys[i])
            value, tag = int(tab[0][x_hat]), int(tab[1][x_hat])
        else:
            point = self._dcf_points.get((i, x_hat))
            if point is None:
                point = self._dcf_points[(i, x_hat)] = fss.dcf_eval(
                    self.role, self.material.dcf_keys[i], x_hat)
            value, tag = point
        return AShare(value, tag, self.width, True)

    def _dpf_key(self):
        key = self.material.dpf_key
        if key is None:
            raise ValueError("the gate material has no DPF key")
        return key

    def dpf_all(self):
        """Full-domain shares ``(y, tau)`` of the payload-1 DPF at ``r`` (object arrays)."""
        if self._dpf_table is None:
            self._dpf_table = fss.dpf_eval_all(self.role, self._dpf_key())
        return self._dpf_table

    def dpf(self, u: int) -> AShare:
        """Share of the DPF at the point ``u`` (exact)."""
        if self.full_domain or self._dpf_table is not None:
            y, tau = self.dpf_all()
            return AShare(int(y[u]), int(tau[u]), self.width, True)
        value, tag = fss.dpf_eval(self.role, self._dpf_key(), u)
        return AShare(value, tag, self.width, True)

    def step(self, c: int, payload: str, x_hat: int) -> AShare:
        """``D_c + w_c`` (payload 1) or ``E_c + r w_c`` (payload r): the two terms of a threshold."""
        if payload == ONE:
            return self.dcf(c, ONE, x_hat) + self.w(c, ONE)
        return self.dcf(c, R, x_hat) + self.rw(c)


def as_eval(g, delta_share=None) -> GateEval:
    """``g`` itself if it is a ``GateEval``, else a fresh evaluator for the material ``g``."""
    if isinstance(g, GateEval):
        if delta_share is not None and g.delta_share is None:
            g.delta_share = int(delta_share)
        return g
    return GateEval(g, delta_share)


# --------------------------------------------------------------------------- #
# Gates on the payload-1 DPF at index r
# --------------------------------------------------------------------------- #

def equality(g, x_hat: int, c: int = 0) -> AShare:
    """``[[ [x = c] ]]`` = the DPF at ``(x_hat - c) mod N``; exact."""
    g = as_eval(g)
    x_hat = g.x_hat(x_hat)
    if not _is_int(c) or not 0 <= int(c) < g.N:
        raise ValueError("c must lie in [0, 2^n)")
    return g.dpf((x_hat - int(c)) % g.N)


def zero_test(g, x_hat: int) -> AShare:
    """``[[ [x = 0] ]]``; exact."""
    return equality(g, x_hat, 0)


def table(g, x_hat: int, T) -> AShare:
    """``[[ T[x] ]]`` for a public table ``T: [N] -> Z_{2^k}``; exact; ``Theta(N)`` work.

    ``sum_u T[(x_hat - u) mod N] y(u)`` on both planes: the DPF shares ``y`` are
    an indicator of ``u = r``, so the sum selects ``T[(x_hat - r) mod N] = T[x]``.
    """
    g = as_eval(g)
    x_hat = g.x_hat(x_hat)
    values = list(T)
    if len(values) != g.N:
        raise ValueError("the table must have 2^n entries")
    mask_k = (1 << g.k) - 1
    rotated = np.array([int(v) & mask_k for v in values], dtype=object)[
        (x_hat - np.arange(g.N)) % g.N]                      # rotated[u] = T[(x_hat - u) mod N]
    y, tau = g.dpf_all()
    mod = 1 << g.width
    return AShare(int(np.dot(rotated, y)) % mod, int(np.dot(rotated, tau)) % mod, g.width, True)


# --------------------------------------------------------------------------- #
# Gates on DCFs
# --------------------------------------------------------------------------- #

def _threshold_const(g: GateEval, c, name: str = "c") -> int:
    if not _is_int(c) or not 0 <= int(c) <= g.N:
        raise ValueError(f"{name} must lie in [0, 2^n]")
    return int(c)


def comparison(g, x_hat: int, c: int) -> AShare:
    """``[[ [x < c] ]]`` = ``D_c - D_0 + w_c`` for a public ``c`` in ``[0, N]``; exact."""
    g = as_eval(g)
    x_hat = g.x_hat(x_hat)
    c = _threshold_const(g, c)
    return (g.step(c, ONE, x_hat) - g.dcf(0, ONE, x_hat)).with_exact(True)


def interval(g, x_hat: int, p: int, q: int) -> AShare:
    """``[[ [p <= x <= q] ]]`` = ``D_(q+1) - D_p + w_(q+1) - w_p``, ``0 <= p <= q < N``; exact."""
    g = as_eval(g)
    x_hat = g.x_hat(x_hat)
    if not (_is_int(p) and _is_int(q)) or not 0 <= int(p) <= int(q) < g.N:
        raise ValueError("need 0 <= p <= q < 2^n")
    return (g.step(int(q) + 1, ONE, x_hat) - g.step(int(p), ONE, x_hat)).with_exact(True)


def _endpoints(g: GateEval, endpoints) -> list:
    pts = [_threshold_const(g, e, "an endpoint") for e in endpoints]
    if len(pts) < 2 or any(a >= b for a, b in zip(pts, pts[1:])):
        raise ValueError("endpoints must be at least two strictly increasing values in [0, 2^n]")
    return pts


def multi_interval(g, x_hat: int, endpoints) -> list:
    """``[[ [e_j <= x < e_(j+1)] ]]`` for consecutive endpoints: ``m + 1`` DCFs, ``m`` exact shares."""
    g = as_eval(g)
    x_hat = g.x_hat(x_hat)
    steps = [g.step(e, ONE, x_hat) for e in _endpoints(g, endpoints)]
    return [(hi - lo).with_exact(True) for lo, hi in zip(steps, steps[1:])]


def nonneg(g, x_hat: int) -> AShare:
    """``[[ [x < N/2] ]]``: 1 iff the two's-complement value of ``x`` is non-negative; exact."""
    g = as_eval(g)
    return comparison(g, x_hat, g.N // 2)


def msb(g, x_hat: int, delta_share=None) -> AShare:
    """``[[ msb(x) ]]`` = ``1 - [x < N/2]``, the sign bit; exact.  Needs this party's ``Delta_b``."""
    g = as_eval(g, delta_share)
    if g.delta_share is None:
        raise ValueError("msb adds a public constant and needs this party's Delta share")
    below = comparison(g, x_hat, g.N // 2)
    return (-below).add_public(1, g.role, g.delta_share).with_exact(True)


sign = msb


def _need_n_eq_k(g: GateEval, name: str) -> None:
    if g.n != g.k:
        raise ValueError(f"{name} needs n = k (use a table gate for a small domain)")


def relu_large(g, x_hat: int) -> AShare:
    """``[[ ReLU(x) ]]`` for ``n = k``, two's complement: ``x_hat b - (E_c - E_0 + [[r w_c]])``.

    ``c = N/2``, ``b = D_c - D_0 + w_c = [x < N/2]``.  Correct modulo ``2^k``;
    the representative is not reduced, so the share is not exact.
    """
    g = as_eval(g)
    _need_n_eq_k(g, "relu_large")
    x_hat = g.x_hat(x_hat)
    c = g.N // 2
    b = g.step(c, ONE, x_hat) - g.dcf(0, ONE, x_hat)            # [[ [x < N/2] ]]
    rb = g.step(c, R, x_hat) - g.dcf(0, R, x_hat)               # [[ r [x < N/2] ]]
    return b.scale(x_hat) - rb


def _pieces(pts: list, pieces) -> list:
    out = []
    for item in pieces:
        a, b = item
        if not (_is_int(a) and _is_int(b)):
            raise ValueError("pieces are pairs (a_j, b_j) of public integers")
        out.append((int(a), int(b)))
    if len(out) != len(pts) - 1:
        raise ValueError("one piece (a_j, b_j) per interval [e_j, e_(j+1)) is required")
    return out


def linear_spline(g, x_hat: int, endpoints, pieces) -> AShare:
    """``[[ a_j x + b_j ]]`` for ``x`` in ``[e_j, e_(j+1))`` (0 elsewhere), ``n = k``; not exact.

    ``sum_j (a_j x_hat + b_j) ind_j - a_j (r ind_j)`` with the indicators
    ``ind_j`` from the payload-1 DCFs and ``r ind_j`` from the payload-``r``
    DCFs at the endpoints.  Correct modulo ``2^k``.
    """
    g = as_eval(g)
    _need_n_eq_k(g, "linear_spline")
    x_hat = g.x_hat(x_hat)
    pts = _endpoints(g, endpoints)
    coeffs = _pieces(pts, pieces)
    ones = [g.step(e, ONE, x_hat) for e in pts]
    rs = [g.step(e, R, x_hat) for e in pts]
    acc = AShare(0, 0, g.width)
    for j, (a, b) in enumerate(coeffs):
        acc = acc + (ones[j + 1] - ones[j]).scale(a * x_hat + b) - (rs[j + 1] - rs[j]).scale(a)
    return acc.with_exact(False)


# --------------------------------------------------------------------------- #
# Public tables for the table gate, and plaintext references
# --------------------------------------------------------------------------- #

def function_table(n: int, k: int, f) -> list:
    """``T[x] = f(x) mod 2^k`` for ``x`` in ``[0, 2^n)``."""
    return [int(f(x)) % (1 << k) for x in range(1 << n)]


def relu_table(n: int, k: int) -> list:
    """ReLU on n-bit two's complement: ``T[x] = x`` for ``x < N/2`` and 0 otherwise."""
    half = 1 << (n - 1)
    return function_table(n, k, lambda x: x if x < half else 0)


def sign_table(n: int, k: int = 1) -> list:
    """Sign bit of n-bit two's complement: ``T[x] = [x >= N/2]``."""
    half = 1 << (n - 1)
    return function_table(n, max(int(k), 1), lambda x: int(x >= half))


def plain_linear_spline(x: int, k: int, endpoints, pieces) -> int:
    """Plaintext spline: ``a_j x + b_j mod 2^k`` on ``[e_j, e_(j+1))``, 0 outside."""
    pts = list(endpoints)
    for j, (a, b) in enumerate(pieces):
        if pts[j] <= x < pts[j + 1]:
            return (a * x + b) % (1 << k)
    return 0


def spline_table(n: int, k: int, endpoints, pieces) -> list:
    """The table of a piecewise linear function on a small domain (for the table gate)."""
    return [plain_linear_spline(x, k, endpoints, pieces) for x in range(1 << n)]


def plain_relu(x: int, n: int) -> int:
    """ReLU on n-bit two's complement, as an element of ``[0, 2^n)``."""
    return x if x < (1 << (n - 1)) else 0


# --------------------------------------------------------------------------- #
# Descriptors (what protocol.gen_gate has to generate)
# --------------------------------------------------------------------------- #

def table_descriptor(n: int, k: int, s: int, auth_input: bool = False, n_outmask: int = 0) -> dict:
    """One payload-1 DPF at index ``r``: zero test, equality and any table."""
    return cg.gate_descriptor(n, k, s, auth_input, True, [], n_outmask)


zero_test_descriptor = table_descriptor
equality_descriptor = table_descriptor


def multi_interval_descriptor(n: int, k: int, s: int, endpoints, auth_input: bool = False,
                              n_outmask: int = 0) -> dict:
    """One payload-1 DCF per endpoint (in the given order; duplicates removed)."""
    seen = []
    for e in endpoints:
        if int(e) not in seen:
            seen.append(int(e))
    return cg.gate_descriptor(n, k, s, auth_input, False, [(e, ONE) for e in seen], n_outmask)


def comparison_descriptor(n: int, k: int, s: int, c: int, auth_input: bool = False,
                          n_outmask: int = 0) -> dict:
    """Payload-1 DCFs at ``c`` and at 0."""
    return multi_interval_descriptor(n, k, s, [c, 0], auth_input, n_outmask)


def interval_descriptor(n: int, k: int, s: int, p: int, q: int, auth_input: bool = False,
                        n_outmask: int = 0) -> dict:
    """Payload-1 DCFs at ``q + 1`` and at ``p``."""
    if not 0 <= int(p) <= int(q) < (1 << n):
        raise ValueError("need 0 <= p <= q < 2^n")
    return multi_interval_descriptor(n, k, s, [int(q) + 1, int(p)], auth_input, n_outmask)


def msb_descriptor(n: int, k: int, s: int, auth_input: bool = False, n_outmask: int = 0) -> dict:
    """Sign / MSB: the comparison with ``c = N/2``."""
    return comparison_descriptor(n, k, s, 1 << (n - 1), auth_input, n_outmask)


def linear_spline_descriptor(k: int, s: int, endpoints, auth_input: bool = True,
                             n_outmask: int = 1) -> dict:
    """Payload-1 and payload-``r`` DCFs at every endpoint, ``n = k``; one output mask by default."""
    pts = []
    for e in endpoints:
        if int(e) not in pts:
            pts.append(int(e))
    ths = [(e, ONE) for e in pts] + [(e, R) for e in pts]
    return cg.gate_descriptor(k, k, s, auth_input, False, ths, n_outmask)


def relu_large_descriptor(k: int, s: int, auth_input: bool = True, n_outmask: int = 1) -> dict:
    """Payload-1 and payload-``r`` DCFs at ``N/2`` and 0, ``n = k``; one output mask by default."""
    return linear_spline_descriptor(k, s, [1 << (k - 1), 0], auth_input, n_outmask)


def offset_descriptor(n: int, k: int, s: int, auth_input: bool = True, n_outmask: int = 0) -> dict:
    """Material without keys: an offset (and output masks) only."""
    return cg.gate_descriptor(n, k, s, auth_input, False, [], n_outmask)


# --------------------------------------------------------------------------- #
# Trusted-dealer reference of Gate.Gen (NOT a protocol; tests and formula checks)
# --------------------------------------------------------------------------- #

def deal_gate(descriptor, delta: int, rng=None, rp=None) -> tuple:
    """Both parties' ``GateMaterial`` as a trusted dealer would produce it (section 4.3).

    NOT a protocol: the caller knows the full MAC key ``delta = Delta_0 +
    Delta_1`` and the function samples ``r'``, the roots and all shares
    itself, with ``fss.dpf_gen`` / ``fss.dcf_gen`` for the keys.  It is the
    plaintext counterpart of ``protocol.gen_gate`` and exists for formula
    tests of gates whose generation circuit is too large to run in a test
    (``relu_large`` and ``linear_spline`` with ``n = k``) and for negative
    controls.  ``rng`` offers ``getrandbits`` and ``randbytes``
    (``random.Random``; default ``random.SystemRandom``); ``rp`` fixes ``r'``.
    """
    desc = cg.canonical_descriptor(descriptor)
    if desc["kind"] != "gate":
        raise ValueError("expected a gate descriptor")
    rnd = random.SystemRandom() if rng is None else rng
    n, k, s = desc["n"], desc["k"], desc["s"]
    width, N = k + s, 1 << n
    delta = int(delta)
    if not 0 <= delta < (1 << (s + 1)):
        raise ValueError("delta must be the unreduced sum of two s-bit shares")
    auth = desc["auth_input"]
    m = n + s if auth else n
    rp = rnd.getrandbits(m) if rp is None else int(rp) % (1 << m)
    r = rp % N

    def split(value: int, tag, bits: int) -> tuple:
        """Additive shares (value, tag) of both parties modulo 2^bits."""
        mod = 1 << bits
        v0, t0 = rnd.getrandbits(bits), rnd.getrandbits(bits)
        tags = (None, None) if tag is None else (t0, (tag - t0) % mod)
        return (v0, tags[0]), ((value - v0) % mod, tags[1])

    rp_sh = split(rp if auth else r, delta * rp % (1 << m) if auth else None, m)
    dpf = fss.dpf_gen(r, 1, delta, n, k, s, rng=rnd) if desc["dpf_one"] else (None, None)
    dcf = ([], [])
    w_sh = ([], [])
    rw_sh = ([], [])
    for c, payload in desc["thresholds"]:
        w_c = int(r + c >= N)
        keys = fss.dcf_gen((r + c) % N, 1 if payload == ONE else r, delta, n, k, s, rng=rnd)
        ws = split(w_c, delta * w_c % (1 << width), width)
        rws = split(r * w_c, delta * r * w_c % (1 << width), width) if payload == R else (None, None)
        for b in (0, 1):
            dcf[b].append(keys[b])
            w_sh[b].append(ws[b])
            rw_sh[b].append(rws[b])
    om = ([], [])
    for _ in range(desc["n_outmask"]):
        rho = rnd.getrandbits(s)
        hi = split(rho, delta * rho % (1 << s), s)           # shares of the high s bits
        for b in (0, 1):
            om[b].append((hi[b][0] << k, hi[b][1] << k))
    return tuple(GateMaterial(b, desc, m, rp_sh[b][0], rp_sh[b][1], dpf[b], tuple(dcf[b]),
                              tuple(w_sh[b]), tuple(rw_sh[b]), tuple(om[b])) for b in (0, 1))


# --------------------------------------------------------------------------- #
# GateSpec: a gate kind with its public parameters
# --------------------------------------------------------------------------- #

_KINDS = ("zero_test", "equality", "table", "comparison", "interval", "multi_interval", "nonneg",
          "msb", "relu_large", "linear_spline", "offset_only")


@dataclass(frozen=True)
class GateSpec:
    """A catalogue gate ``G`` with its public parameters (the ``G`` of ``Gate(gid, G)``).

    Build instances with the class methods.  ``exact`` is the flag of the
    output sharing; ``n_outputs`` is 1 except for ``multi_interval``.
    """

    kind: str
    n: int
    k: int
    s: int
    params: tuple = ()
    auth_input: bool = False
    n_outmask: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown gate kind {self.kind!r}")
        self.descriptor()                      # validates n, k, s and the parameters

    # -- constructors ----------------------------------------------------------- #

    @classmethod
    def zero_test(cls, n, k, s, auth_input=False):
        return cls("zero_test", int(n), int(k), int(s), (), bool(auth_input))

    @classmethod
    def equality(cls, n, k, s, c, auth_input=False):
        if not 0 <= int(c) < (1 << int(n)):
            raise ValueError("c must lie in [0, 2^n)")
        return cls("equality", int(n), int(k), int(s), (int(c),), bool(auth_input))

    @classmethod
    def table(cls, n, k, s, T, auth_input=False):
        values = tuple(int(v) % (1 << int(k)) for v in T)
        if len(values) != 1 << int(n):
            raise ValueError("the table must have 2^n entries")
        return cls("table", int(n), int(k), int(s), values, bool(auth_input))

    @classmethod
    def comparison(cls, n, k, s, c, auth_input=False):
        return cls("comparison", int(n), int(k), int(s), (int(c),), bool(auth_input))

    @classmethod
    def interval(cls, n, k, s, p, q, auth_input=False):
        return cls("interval", int(n), int(k), int(s), (int(p), int(q)), bool(auth_input))

    @classmethod
    def multi_interval(cls, n, k, s, endpoints, auth_input=False):
        return cls("multi_interval", int(n), int(k), int(s), tuple(int(e) for e in endpoints),
                   bool(auth_input))

    @classmethod
    def nonneg(cls, n, k, s, auth_input=False):
        return cls("nonneg", int(n), int(k), int(s), (), bool(auth_input))

    @classmethod
    def msb(cls, n, k, s, auth_input=False):
        return cls("msb", int(n), int(k), int(s), (), bool(auth_input))

    @classmethod
    def relu_large(cls, k, s, auth_input=True, n_outmask=1):
        return cls("relu_large", int(k), int(k), int(s), (), bool(auth_input), int(n_outmask))

    @classmethod
    def linear_spline(cls, k, s, endpoints, pieces, auth_input=True, n_outmask=1):
        pts = tuple(int(e) for e in endpoints)
        coeffs = tuple((int(a), int(b)) for a, b in pieces)
        return cls("linear_spline", int(k), int(k), int(s), (pts, coeffs), bool(auth_input),
                   int(n_outmask))

    @classmethod
    def offset_only(cls, n, k, s, auth_input=True, n_outmask=0):
        return cls("offset_only", int(n), int(k), int(s), (), bool(auth_input), int(n_outmask))

    # -- the three views ---------------------------------------------------------- #

    @property
    def exact(self) -> bool:
        return self.kind not in ("relu_large", "linear_spline")

    @property
    def n_outputs(self) -> int:
        if self.kind == "multi_interval":
            return len(self.params) - 1
        return 0 if self.kind == "offset_only" else 1

    def descriptor(self) -> dict:
        """The ``circuits_gen`` gate descriptor of the material this gate needs."""
        n, k, s, auth, om = self.n, self.k, self.s, self.auth_input, self.n_outmask
        kind, p = self.kind, self.params
        if kind in ("zero_test", "equality", "table"):
            return table_descriptor(n, k, s, auth, om)
        if kind == "comparison":
            return comparison_descriptor(n, k, s, p[0], auth, om)
        if kind == "interval":
            return interval_descriptor(n, k, s, p[0], p[1], auth, om)
        if kind == "multi_interval":
            if len(p) < 2 or any(a >= b for a, b in zip(p, p[1:])):
                raise ValueError("endpoints must be strictly increasing")
            return multi_interval_descriptor(n, k, s, p, auth, om)
        if kind in ("nonneg", "msb"):
            return msb_descriptor(n, k, s, auth, om)
        if kind == "relu_large":
            return relu_large_descriptor(k, s, auth, om)
        if kind == "linear_spline":
            pts, coeffs = p
            if len(pts) < 2 or any(a >= b for a, b in zip(pts, pts[1:])) or len(coeffs) != len(pts) - 1:
                raise ValueError("a spline needs increasing endpoints and one piece per interval")
            return linear_spline_descriptor(k, s, pts, auth, om)
        return offset_descriptor(n, k, s, auth, om)

    def evaluate(self, g, x_hat: int, delta_share=None):
        """``Apply``: this party's share of ``[[G(x)]]`` (a list for ``multi_interval``)."""
        g = as_eval(g, delta_share)
        if (g.n, g.k, g.s) != (self.n, self.k, self.s):
            raise ValueError("the gate material does not match the gate parameters")
        kind, p = self.kind, self.params
        if kind == "zero_test":
            return zero_test(g, x_hat)
        if kind == "equality":
            return equality(g, x_hat, p[0])
        if kind == "table":
            return table(g, x_hat, p)
        if kind == "comparison":
            return comparison(g, x_hat, p[0])
        if kind == "interval":
            return interval(g, x_hat, p[0], p[1])
        if kind == "multi_interval":
            return multi_interval(g, x_hat, p)
        if kind == "nonneg":
            return nonneg(g, x_hat)
        if kind == "msb":
            return msb(g, x_hat)
        if kind == "relu_large":
            return relu_large(g, x_hat)
        if kind == "linear_spline":
            return linear_spline(g, x_hat, p[0], p[1])
        raise ValueError("an offset-only gate has no function to apply")

    def plain(self, x: int):
        """Plaintext ``G(x)`` as an element of ``[0, 2^k)`` (a list for ``multi_interval``)."""
        N = 1 << self.n
        if not 0 <= x < N:
            raise ValueError("x out of range")
        kind, p = self.kind, self.params
        if kind == "zero_test":
            return int(x == 0)
        if kind == "equality":
            return int(x == p[0])
        if kind == "table":
            return p[x]
        if kind == "comparison":
            return int(x < p[0])
        if kind == "interval":
            return int(p[0] <= x <= p[1])
        if kind == "multi_interval":
            return [int(a <= x < b) for a, b in zip(p, p[1:])]
        if kind == "nonneg":
            return int(x < N // 2)
        if kind == "msb":
            return int(x >= N // 2)
        if kind == "relu_large":
            return plain_relu(x, self.n)
        if kind == "linear_spline":
            return plain_linear_spline(x, self.k, p[0], p[1])
        raise ValueError("an offset-only gate has no function")
