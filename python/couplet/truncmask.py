"""Truncate-and-mask (DESIGN_M3.md section 3), both modes, and the openings of an
inference session.

``Opener(role, chan, sid, checker, delta_share)`` bundles what one party needs
to open values during inference:

  * ``open_deferred(shares)``: one exchange of the value shares of authenticated
    sharings, the MAC check deferred (``couplet.spdz2k.ArithChecker``);
  * ``check()``: ArithCheck over every deferred opening, one check per modulus;
  * ``open_sign_exact(shares)``: the release of DESIGN_M3.md section 3, Output:
    check, open, check; refused before any message unless every sharing is
    marked ``sign_exact``; returns the signed ``k``-bit decoding of the payload
    ``v mod 2^k`` (``k = bits - s``), the only part of an opening that the
    check authenticates (the caller guarantees ``|v| < 2^(k-1)``);
  * ``open_masked(shares, masks)``: the release with output masks
    (correction C2 of design/CHANGELOG_M3.md, private-model mode): check, add
    the output mask ``[[2^k rho]]`` of each sharing, open, check (the core's
    ``ArithChecker.open_final_many``); returns ``to_signed(v mod 2^k, k)``.
    The opened high ``s`` bits are ``(v >> k) + rho`` and carry no
    information on ``v >> k``; masks must be fresh, over the same modulus,
    with both components multiples of ``2^k`` (refused before any message);
  * ``open_shared(values, bits)``: commit-then-open of ADDITIVE shares without
    MAC (the first-layer pre-activation shares): one committed exchange for
    the whole list under ``sid_infer = SHA-256(frame("Couplet-infer-sid", sid))``
    and the label ``"tm|" || le64(counter)``, each value ``ceil(bits/8)`` bytes
    little-endian; two rounds; a malformed message aborts.  The opened sums
    are NOT covered by any MAC check (a wrong share is an input substitution);
  * ``public(c, bits)``: the sharing of a public constant.

``Site`` wraps a ``circuits_ppml.SiteMaterial`` with the lazily evaluated
planes (``plane_a``, ``plane_b``).

``truncate_and_mask(opener, sites, p, first_layer)`` runs one layer: for the
sites ``j`` with pre-activation sharings ``p[j]`` over ``Z_{2^w}`` (or, with
``first_layer=True``, additive shares of ``p`` as integers, no MAC):

mode ``round`` (one opening): open ``v = (p + rho') mod 2^m``, ``m = f+n+s``
(shares reduced modulo ``2^m``); ``v_lo = v mod 2^(f+n)``,
``x_hat = v_lo >> f``, ``v_l = v_lo mod 2^f``.  Then
``x_hat = (x + beta + rho_h) mod 2^n`` with ``x = (p >> f) mod 2^n`` and
``beta = [p mod 2^f > v_l]``: ``x_hat`` is the masked input of plane A for
the value ``x + beta``, a rounding with the public uniform threshold ``v_l``.

mode ``exact`` (two openings): open ``v`` as above; ``[[beta]]`` = suffix sum
of plane B above ``v_l`` (= ``[v_l < rho_l]``);
``[[x]] = (v_lo >> f) - [[rho_h]] - [[beta]]`` over ``Z_{2^(n+s)}``, an n-bit
sharing of the exact floor; then open ``x_hat = x + r'`` over ``Z_{2^(n+s)}``
for plane A (at ``r = r' mod 2^n``).

``f = 0`` (plain offset gate) makes both modes one opening with
``x_hat = v mod 2^n`` and ``z = p mod 2^n`` exactly.  All sites of a layer
share one exchange per opening.  ``plain_truncate(p, f, n, mode, v_l)`` is
the plaintext of the value ``z`` plane A masks (the mirror uses it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from . import hashes as hs
from .circuits_ppml import SiteMaterial
from .fixedpoint import to_signed
from .hashes import ProtocolAbort
from .planes import Plane
from .spdz2k import AShare, ArithChecker

__all__ = ["Opener", "PHASE_SHARED", "Site", "TruncResult", "plain_truncate", "truncate_and_mask"]

PHASE_SHARED = "tm-open"
_TAG_SID = b"Couplet-infer-sid"
_LABEL_TM = b"tm|"


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------------------------- #
# Openings of one party
# --------------------------------------------------------------------------- #

class Opener:
    """Openings of one inference session; see the module docstring."""

    def __init__(self, role: int, chan, sid: bytes, checker: ArithChecker, delta_share: int) -> None:
        if role not in (0, 1) or chan.party != role:
            raise ValueError("role must be 0 or 1 and match the channel endpoint")
        if not isinstance(checker, ArithChecker) or checker.role != role:
            raise ValueError("expected this party's ArithChecker")
        self.role = role
        self.chan = chan
        self.sid = bytes(sid)
        self.checker = checker
        self.delta_share = int(delta_share)
        self.s = checker.s
        self.shared_openings = 0
        self._sid_infer = hs.sha256(_TAG_SID, self.sid)

    # -- hooks that a deviating party overrides ------------------------------------- #

    def _shared_exchange(self, label: bytes, msg: bytes) -> bytes:
        """The commit-then-open exchange of one first-layer opening (honest: as is)."""
        return hs.committed_exchange(self.chan, self._sid_infer, label, msg)

    # -- public constants -------------------------------------------------------------- #

    def public(self, c: int, bits: int) -> AShare:
        """``[[c]]`` for a public integer ``c`` over ``Z_{2^bits}``."""
        return AShare(0, 0, int(bits)).add_public(int(c), self.role, self.delta_share)

    # -- openings ---------------------------------------------------------------------- #

    def open_shared(self, values: Sequence[int], bits: int) -> List[int]:
        """Commit-then-open of additive shares without MAC; returns the sums modulo ``2^bits``."""
        bits = int(bits)
        values = [int(v) for v in values]
        if bits < 1 or any(not 0 <= v < (1 << bits) for v in values):
            raise ValueError("shares must lie in [0, 2^bits)")
        if not values:
            return []
        nb = (bits + 7) // 8
        label = _LABEL_TM + self.shared_openings.to_bytes(8, "little")
        self.shared_openings += 1
        msg = b"".join(v.to_bytes(nb, "little") for v in values)
        phase = self.chan.phase
        self.chan.mark(PHASE_SHARED)
        try:
            reply = self._shared_exchange(label, msg)
        finally:
            self.chan.mark(phase)
        if len(reply) != len(msg):
            raise ProtocolAbort("truncate-and-mask: message has the wrong length")
        out = []
        mod = 1 << bits
        for j, v in enumerate(values):
            peer = int.from_bytes(reply[j * nb:(j + 1) * nb], "little")
            if peer >= mod:
                raise ProtocolAbort("truncate-and-mask: share out of range")
            out.append((v + peer) % mod)
        return out

    def open_deferred(self, shares: Sequence[AShare]) -> List[int]:
        """One exchange of value shares; the representatives, verified at the next ``check``."""
        return self.checker.open_deferred_many(list(shares))

    def check(self) -> None:
        """ArithCheck over every deferred opening (one check per modulus)."""
        self.checker.check()

    def open_sign_exact(self, shares: Sequence[AShare]) -> List[int]:
        """Release sign-exact values: check, open, check; returns the signed integers.

        The check guarantees the opened value modulo ``2^k`` only (``k = bits - s``;
        an error in the top ``s`` bits passes with noticeable probability), so the
        result is ``to_signed(v mod 2^k, k)``, which is the intended integer when
        ``|v| < 2^(k-1)`` (``nn.Model.release_fits``).
        """
        shares = list(shares)
        for sh in shares:
            if not isinstance(sh, AShare) or not sh.sign_exact:
                raise ValueError("only sharings marked sign_exact are released without an output mask")
            if sh.bits <= self.s:
                raise ValueError("a released sharing needs bits > s")
        if not shares:
            return []
        self.checker.check()
        opened = self.checker.open_deferred_many(shares)
        self.checker.check()
        return [to_signed(v % (1 << (sh.bits - self.s)), sh.bits - self.s) for v, sh in zip(opened, shares)]

    def open_masked(self, shares: Sequence[AShare], masks: Sequence[AShare]) -> List[int]:
        """Release with output masks: check, add mask, open, check; returns the signed ``k``-bit payloads.

        ``masks[j]`` is a fresh output mask ``[[2^k rho_j]]`` (``rho_j`` uniform in
        ``[0, 2^s)``) over the modulus of ``shares[j]``.  The masked opening
        ``v + 2^k rho`` reveals ``v mod 2^k`` and nothing about the high ``s`` bits
        of ``v``, which no bound guarantees when the weights are private.
        """
        shares, masks = list(shares), list(masks)
        if len(masks) != len(shares):
            raise ValueError("one output mask per released sharing is required")
        for sh, mk in zip(shares, masks):
            if not isinstance(sh, AShare) or not isinstance(mk, AShare):
                raise ValueError("expected AShare objects")
            if sh.bits <= self.s or mk.bits != sh.bits:
                raise ValueError("a released sharing needs bits > s and a mask over the same modulus")
            if mk.value % (1 << (sh.bits - self.s)) or mk.mac % (1 << (sh.bits - self.s)):
                raise ValueError("output mask shares must be multiples of 2^k")
        if not shares:
            return []
        # Every sharing is opened through its mask (the core opens an exact sharing unmasked).
        payloads = self.checker.open_final_many([sh.with_exact(False) for sh in shares], masks)
        return [to_signed(v, sh.bits - self.s) for v, sh in zip(payloads, shares)]


# --------------------------------------------------------------------------- #
# Sites
# --------------------------------------------------------------------------- #

class Site:
    """One activation site: its material and the lazily evaluated planes."""

    def __init__(self, material: SiteMaterial) -> None:
        if not isinstance(material, SiteMaterial):
            raise ValueError("expected a circuits_ppml.SiteMaterial")
        self.material = material
        self.role = material.role
        self.n, self.f, self.k, self.s = material.n, material.f, material.k, material.s
        self.m = material.m
        self.width = material.width
        self._plane_a: Optional[Plane] = None
        self._plane_b: Optional[Plane] = None

    @property
    def plane_a(self) -> Plane:
        if self._plane_a is None:
            self._plane_a = Plane(self.role, self.material.plane_a)
        return self._plane_a

    @property
    def plane_b(self) -> Plane:
        if self.material.plane_b is None:
            raise ValueError("the site has no plane B (mode round or f = 0)")
        if self._plane_b is None:
            self._plane_b = Plane(self.role, self.material.plane_b)
        return self._plane_b

    @property
    def two_openings(self) -> bool:
        return self.material.two_openings

    def rp_share(self) -> AShare:
        """``[[rho']]`` over ``Z_{2^m}``."""
        value, tag = self.material.rp
        return AShare(value, tag, self.m)

    def rh_share(self) -> AShare:
        value, tag = self.material.rh
        return AShare(value, tag, self.n + self.s)

    def r2_share(self) -> AShare:
        value, tag = self.material.r2
        return AShare(value, tag, self.n + self.s)


@dataclass(frozen=True)
class TruncResult:
    """The public outcome of one site: ``x_hat`` (masked input of plane A), the opened
    ``v = p + rho' mod 2^m``, the threshold ``v_l`` (meaningful in mode round), whether the
    masked value is the exact floor, and the second opened representative (mode exact)."""

    x_hat: int
    v: int
    v_l: int
    exact: bool
    opened2: Optional[int] = None


def plain_truncate(p: int, f: int, n: int, mode: str, v_l: Optional[int] = None) -> int:
    """The value ``z`` that plane A masks after truncate-and-mask of the integer ``p``.

    ``x = (p >> f) mod 2^n`` (floor); mode ``exact`` or ``f = 0``: ``z = x``;
    mode ``round``: ``z = (x + [p mod 2^f > v_l]) mod 2^n``.
    """
    x = (int(p) >> int(f)) % (1 << int(n))
    if mode == "exact" or int(f) == 0:
        return x
    if v_l is None:
        raise ValueError("mode round needs the public threshold v_l")
    beta = int((int(p) % (1 << int(f))) > int(v_l))
    return (x + beta) % (1 << int(n))


def truncate_and_mask(opener: Opener, sites: Sequence[Site], p: Sequence, first_layer: bool = False
                      ) -> List[TruncResult]:
    """Truncate-and-mask of all sites of one layer; see the module docstring.

    ``p[j]`` is an ``AShare`` over ``Z_{2^w}`` (``first_layer=False``) or this
    party's additive share of the pre-activation as an integer (``True``; it
    is reduced modulo ``2^m``).  Returns one ``TruncResult`` per site.
    """
    sites = list(sites)
    p = list(p)
    if len(sites) != len(p):
        raise ValueError("one pre-activation per site is required")
    if not sites:
        return []
    first = sites[0]
    n, f, s, m = first.n, first.f, first.s, first.m
    for site in sites:
        if not isinstance(site, Site):
            raise ValueError("expected Site objects")
        if (site.n, site.f, site.s, site.m, site.two_openings) != (n, f, s, m, first.two_openings):
            raise ValueError("all sites of a layer must share n, f, s and the mode")
    if first_layer:
        if any(not _is_int(v) for v in p):
            raise ValueError("first-layer pre-activations are integers (additive shares without MAC)")
        v_all = opener.open_shared([(int(v) + site.material.rp[0]) % (1 << m) for v, site in zip(p, sites)], m)
    else:
        for sh in p:
            if not isinstance(sh, AShare) or sh.bits < m:
                raise ValueError("pre-activations must be sharings over at least 2^m")
        v_all = opener.open_deferred([sh.reduce(f + n, s) + site.rp_share() for sh, site in zip(p, sites)])
    mask_lo = (1 << (f + n)) - 1
    results = []
    if not first.two_openings:
        for v in v_all:
            v_lo = v & mask_lo
            results.append(TruncResult(v_lo >> f, v, v_lo & ((1 << f) - 1), f == 0))
        return results
    # mode exact with f >= 1: [[x]] = (v_lo >> f) - [[rho_h]] - [[beta]], then open x + r'
    x_shares = []
    public = []
    for v, site in zip(v_all, sites):
        v_lo = v & mask_lo
        v_l = v_lo & ((1 << f) - 1)
        beta = site.plane_b.suffix(v_l + 1, bits=n + s)           # [v_l < rho_l]
        x = opener.public(v_lo >> f, n + s) - site.rh_share() - beta
        x_shares.append(x + site.r2_share())
        public.append((v, v_l))
    opened2 = opener.open_deferred(x_shares)
    for (v, v_l), o2 in zip(public, opened2):
        results.append(TruncResult(o2 % (1 << n), v, v_l, True, o2))
    return results
