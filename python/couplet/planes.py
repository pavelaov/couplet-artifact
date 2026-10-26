"""One authenticated one-hot plane and its local queries (DESIGN_M3.md section 3, Carve).

A ``Plane`` is this party's full-domain evaluation of a payload-1 DPF key:
value shares ``y_b(u)`` and tag shares ``tau_b(u)`` for every ``u`` in ``[N]``,
``N = 2^n``, with ``y_0 + y_1 = e_alpha`` and ``tau_0 + tau_1 = Delta e_alpha``
in ``Z_{2^w}``.  With the prefix tables ``P_b[j] = sum_{u < j} y_b(u)`` (and
the same on the tags) every query below is a public linear functional of the
plane, computed on BOTH components, and returns this party's
``couplet.spdz2k.AShare`` over ``Z_{2^w}`` (``bits`` reduces it to
``Z_{2^bits}``).  ``x_hat`` is the public masked input of the plane and
``z = (x_hat - alpha) mod N`` the value it masks.

    point(u)                    [[ [u = alpha] ]]                               exact, sign-exact
    table(T, x_hat)             [[ T[z] ]] = sum_u T[(x_hat - u) mod N] y(u)    sign-exact
                                (T: N public integers, embedded sign-exactly
                                in Z_{2^w}; several tables per plane)
    locator(l, u)               [[ [alpha in [l, u]] ]], cyclic (l > u wraps)   exact, sign-exact
    z_interval(x_hat, lo, hi)   [[ [lo <= z <= hi] ]] = locator over
                                J = [x_hat - hi, x_hat - lo]                    exact, sign-exact
    equality(x_hat, c)          [[ [z = c] ]]
    comparison(x_hat, c)        [[ [z < c] ]] for every c in [0, N]
    sign(x_hat)                 [[ [z >= N/2] ]]  (the sign bit of the code)
    nonneg(x_hat)               [[ [z < N/2] ]]
    wrap(x_hat)                 [[ [alpha > x_hat] ]] = suffix(x_hat + 1)
    suffix(j)                   [[ sum_{u >= j} y(u) ]] = [[ [alpha >= j] ]], j in [0, N]
    prefix(j)                   [[ sum_{u < j}  y(u) ]] = [[ [alpha < j] ]]

Nothing here sends a message; the plane is evaluated once (``fss.dpf_eval_all``)
and every query is a look-up or an inner product.  Indicator outputs are
marked ``exact`` (their representative is 0 or 1) and ``sign_exact``; table
outputs are ``sign_exact`` only (entries may be negative).  The ``exact``
and ``sign_exact`` flags are public metadata, see ``couplet.spdz2k``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import fss
from .spdz2k import AShare

__all__ = ["Plane"]


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


class Plane:
    """This party's full-domain evaluation of one payload-1 DPF key; see the module docstring."""

    def __init__(self, role: int, key: fss.DPFKey) -> None:
        if not isinstance(key, fss.DPFKey):
            raise ValueError("a plane is built from a couplet.fss.DPFKey")
        if role not in (0, 1) or key.b != role:
            raise ValueError("the key does not belong to this party")
        self.role = role
        self.key = key
        self.n = key.n
        self.N = 1 << key.n
        self.width = key.width
        self.s = key.s
        self.mod = 1 << key.width
        y, tau = fss.dpf_eval_all(role, key)
        self.y = y                               # object arrays of Python ints in [0, 2^w)
        self.tau = tau
        self.prefix_y = np.concatenate(([0], np.cumsum(y, dtype=object)))
        self.prefix_tau = np.concatenate(([0], np.cumsum(tau, dtype=object)))

    # -- helpers -------------------------------------------------------------- #

    def _share(self, value: int, tag: int, exact: bool, bits: Optional[int]) -> AShare:
        share = AShare(int(value) % self.mod, int(tag) % self.mod, self.width, exact, True)
        if bits is not None and int(bits) != self.width:
            share = share.reduce(int(bits) - self.s, self.s)
        return share

    def _x_hat(self, x_hat) -> int:
        if not _is_int(x_hat) or not 0 <= int(x_hat) < self.N:
            raise ValueError("x_hat must be an integer in [0, 2^n)")
        return int(x_hat)

    def _index(self, j, name: str = "j", upper: Optional[int] = None) -> int:
        upper = self.N if upper is None else upper
        if not _is_int(j) or not 0 <= int(j) <= upper:
            raise ValueError(f"{name} must lie in [0, {upper}]")
        return int(j)

    # -- queries -------------------------------------------------------------- #

    def point(self, u: int, bits: Optional[int] = None) -> AShare:
        """``[[ [u = alpha] ]]``: the plane at ``u``."""
        u = self._index(u, "u", self.N - 1)
        return self._share(self.y[u], self.tau[u], True, bits)

    def table(self, T: Sequence[int], x_hat: int, bits: Optional[int] = None) -> AShare:
        """``[[ T[z] ]]`` for a public table of ``N`` integers (sign-exact entries)."""
        x_hat = self._x_hat(x_hat)
        if len(T) != self.N:
            raise ValueError("the table must have 2^n entries")
        half = self.mod >> 1
        entries = []
        for t in T:
            t = int(t)
            if not -half <= t < half:
                raise ValueError("a table entry does not fit a sign-exact value of w bits")
            entries.append(t % self.mod)
        rotated = np.array(entries, dtype=object)[(x_hat - np.arange(self.N)) % self.N]
        return self._share(int(np.dot(rotated, self.y)), int(np.dot(rotated, self.tau)), False, bits)

    def prefix(self, j: int, bits: Optional[int] = None) -> AShare:
        """``[[ [alpha < j] ]] = sum_{u < j} y(u)``, ``j`` in ``[0, N]``."""
        j = self._index(j)
        return self._share(self.prefix_y[j], self.prefix_tau[j], True, bits)

    def suffix(self, j: int, bits: Optional[int] = None) -> AShare:
        """``[[ [alpha >= j] ]] = sum_{u >= j} y(u)``, ``j`` in ``[0, N]``."""
        j = self._index(j)
        return self._share(self.prefix_y[self.N] - self.prefix_y[j],
                           self.prefix_tau[self.N] - self.prefix_tau[j], True, bits)

    def locator(self, l: int, u: int, bits: Optional[int] = None) -> AShare:
        """``[[ [alpha in J] ]]`` for the cyclic interval ``J = [l, u]`` (``l > u`` wraps through N)."""
        l = self._index(l, "l", self.N - 1)
        u = self._index(u, "u", self.N - 1)
        py, pt = self.prefix_y, self.prefix_tau
        if l <= u:
            return self._share(py[u + 1] - py[l], pt[u + 1] - pt[l], True, bits)
        return self._share(py[self.N] - py[l] + py[u + 1], pt[self.N] - pt[l] + pt[u + 1], True, bits)

    def z_interval(self, x_hat: int, lo: int, hi: int, bits: Optional[int] = None) -> AShare:
        """``[[ [lo <= z <= hi] ]]``: ``z`` in ``I = [lo, hi]`` iff ``alpha`` in ``[x_hat - hi, x_hat - lo]``."""
        x_hat = self._x_hat(x_hat)
        lo = self._index(lo, "lo", self.N - 1)
        hi = self._index(hi, "hi", self.N - 1)
        if lo > hi:
            raise ValueError("need lo <= hi")
        return self.locator((x_hat - hi) % self.N, (x_hat - lo) % self.N, bits)

    def equality(self, x_hat: int, c: int, bits: Optional[int] = None) -> AShare:
        """``[[ [z = c] ]]``."""
        x_hat = self._x_hat(x_hat)
        c = self._index(c, "c", self.N - 1)
        return self.point((x_hat - c) % self.N, bits)

    def comparison(self, x_hat: int, c: int, bits: Optional[int] = None) -> AShare:
        """``[[ [z < c] ]]`` for ``c`` in ``[0, N]`` (0 for c = 0, 1 for c = N)."""
        x_hat = self._x_hat(x_hat)
        c = self._index(c, "c")
        if c == 0:
            return self._share(0, 0, True, bits)
        return self.z_interval(x_hat, 0, c - 1, bits)

    def sign(self, x_hat: int, bits: Optional[int] = None) -> AShare:
        """``[[ [z >= N/2] ]]``: the sign bit of the two's-complement code ``z``."""
        return self.z_interval(x_hat, self.N // 2, self.N - 1, bits)

    def nonneg(self, x_hat: int, bits: Optional[int] = None) -> AShare:
        """``[[ [z < N/2] ]]``."""
        return self.comparison(x_hat, self.N // 2, bits)

    def wrap(self, x_hat: int, bits: Optional[int] = None) -> AShare:
        """``[[ [alpha > x_hat] ]]``: the wrap bit of ``z = x_hat - alpha`` (a suffix sum)."""
        x_hat = self._x_hat(x_hat)
        return self.suffix(x_hat + 1, bits)
