"""SPDZ2k authenticated sharings and the batched MAC check (DESIGN.md sections 2.3 and 5.3).

Everything here is the view of ONE party.

``AShare(value, mac, bits, exact)`` is this party's share of ``[[v]]`` over
``Z_{2^bits}``: the two parties' ``value`` fields sum to a representative
``v~`` of the secret and their ``mac`` fields sum to ``Delta * v~`` modulo
``2^bits``, where ``Delta = Delta_0 + Delta_1`` is the session MAC key as an
(s+1)-bit integer that is NEVER reduced (change C).  ``bits`` is the modulus
exponent ``m``: ``w = k + s`` for wires, ``n + s`` after ``reduce``.  The
payload is ``v~ mod 2^(bits - s)``.  ``exact`` states that the representative
itself is smaller than ``2^(bits - s)`` (section 2.3): outputs of DPF / DCF
evaluations, indicators and table look-ups are exact; sums and multiples are
in general not.  The flag is public: both parties derive the same value from
the public history of a sharing.  Linear operations are local:

    a + b, a - b, -a          same ``bits``; the result is marked not exact
    a.scale(c), c * a         public integer c
    a.add_public(c, role, delta_share)   P_0 adds c to its value share, both
                              parties add ``Delta_b * c`` to their MAC share
    a.reduce(n, s)            the sharing of ``v~ mod 2^(n+s)`` over
                              ``Z_{2^(n+s)}`` (payload ``v mod 2^n``)
    a.with_exact(flag)        set the flag (callers that know an identity
                              over the integers, e.g. the gates of section 6)
    a.with_sign_exact(flag)   set the ``sign_exact`` flag (Couplet layer)

``sign_exact`` is the value class of the Couplet layer (DESIGN_M3.md section
3): the representative in ``Z_{2^bits}`` is the two's-complement embedding of
the intended signed integer, so an opening reveals exactly that integer and
needs no output mask.  Like ``exact`` it is public metadata that every linear
operation clears and that a caller re-sets when the public bound analysis
guarantees that the true integer fits in ``bits - 1`` bits.  The core never
sets it; a sharing with ``exact`` set is also sign-exact in that sense (the
caller may set both).

``ArithChecker(role, chan, sid, delta_share, s)`` opens sharings and runs
``ArithCheck`` of section 2.3 on a ``couplet.net.Channel`` endpoint:

    open_deferred(share) -> int
        one exchange of the value shares; the opened representative is
        returned and recorded together with this party's MAC share.  Nothing
        is verified yet: the caller may only use the value in ways that are
        safe for an unverified opening (a masked gate input, section 5.3).
    check()
        ArithCheck over everything opened since the last check, one check per
        modulus ``m``: a coin toss fixes ``chi_j`` in ``[0, 2^s)`` AFTER the
        openings; ``y = sum chi_j v~_j``; each party computes
        ``zeta_b = sum chi_j mu_{j,b} - Delta_b y mod 2^m``; the ``zeta_b``
        are exchanged by a simultaneous commit-then-open and the check passes
        iff ``zeta_0 + zeta_1 = 0 mod 2^m`` for every modulus.  A failure
        raises ``ProtocolAbort`` (the session must be retired, change G).
    open_final(share, outmask=None) -> int
        the release of section 5.3 (changes E and F): ``check()`` on all
        deferred openings, then the opening of the value, to which the output
        mask ``[[2^k rho]]`` is added unless the sharing is exact, then
        ``check()`` on exactly this opening; only then is ``v mod 2^k``
        returned (``k = bits - s``).  A sharing that is not exact is never
        opened without a mask: the call is refused before any message.

Binding.  The checker works under the sub-session identifier
``SHA-256(frame("Couplet-arith-sid", sid))``.  Every opening has a session-wide
index and every check a counter; neither is ever reused.  The coin toss of a
check is labelled with the counter and with a hash of the list of
(modulus, opening index, opened value) it covers, and the commitments carry the
counter and the committer's index, so a message of another check, of another
session or of the other party is rejected.

Messages.  A value share over ``Z_{2^m}`` is ``ceil(m / 8)`` bytes
little-endian; a message of the wrong length or with a value ``>= 2^m``
aborts.  Rounds: one per opening call; four per check (coin toss 2, committed
exchange 2).  A check with nothing to verify sends nothing.  Channel phases:
``arith-open`` and ``arith-check``; the phase that was active before a call is
restored.

Abort.  Any failure inside a protocol step retires the checker: the key share
and the recorded MAC shares are dropped and every later call raises
``ProtocolAbort``.  Invalid arguments raise ``ValueError`` before any message
and leave the checker usable.

Soundness (Lemma 2).  If some opened value is wrong modulo ``2^(m-s)``, i.e. its
payload is wrong, a check passes with probability at most
``2^(-s + log2(s+1))`` plus the commitment error; ``lemma2_bound(s)`` is that
number and ``tests/test_spdz2k.py`` measures the pass rate of the best
strategies for a tiny ``s`` (the optimum of a single wrong value is
``2^-s (1 + (s+1)/4)``, which is below the bound).

What the verdict of a check depends on (semantics of ArithCheck; no protocol
change).  With the adversary's additive errors ``delta_j`` and
``e = sum chi_j delta_j`` the two zeta values sum to ``-(Delta_0 + Delta_1) e``
plus whatever the adversary adds to its own zeta.  The verdict is therefore a
function of the errors, the tossed coefficients and the MAC key alone, never
of a party's input or of an opened value.  Two consequences:

  * An error that is ``0 mod 2^(m-s)`` touches only the high ``s`` bits of the
    representative.  The payload ``v mod 2^(m-s)`` is unchanged, so no wrong
    value is accepted; such an error is outside Lemma 2 and passes with a
    noticeable probability: ``2^(m-j)`` passes iff ``Delta chi = 0 mod 2^j``,
    i.e. with probability ``(j + 2) / 2^(j+1)`` (3/4 for ``2^(m-1)``).
  * A PASSED check may reveal low bits of the peer's key share, at the
    corresponding abort risk: after a pass with ``e = chi 2^(m-j) != 0`` the
    deviating party knows that ``Delta chi = 0 mod 2^j``, and it has risked an
    abort (which retires the session, change G) with probability
    ``1 - (j + 2) / 2^(j+1)``.  The same holds for errors inside the lemma, at
    the price the lemma bounds.  This concerns the MAC key only: it is the
    arithmetic counterpart of the bits of ``Delta^B`` that the OT-extension
    check may leak (DESIGN.md 7.2).  "A passed check reveals nothing about
    Delta_h" (proof sketch in DESIGN.md 2.3) is to be read for honest
    openings; the experiments A14 of ``couplet.attacks`` measure the rates.

Deviations are not built into this module.  A malicious party is modelled by
a subclass that overrides the three protected steps ``_outgoing_values``
(what is sent as the own value share), ``_recorded_mac`` (the MAC share kept
for the check) and ``_outgoing_zeta`` (what is committed to in the check); see
``couplet.attacks.adversaries``.

``deal`` / ``reconstruct`` are dealer-style helpers for tests and formula
checks; they are not part of any protocol.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace

from . import hashes as hs
from .hashes import ProtocolAbort

__all__ = [
    "AShare", "ArithChecker", "ProtocolAbort", "check_coefficients", "deal", "lemma2_bound",
    "reconstruct", "share_bytes",
]

PHASE_OPEN = "arith-open"
PHASE_CHECK = "arith-check"
_TAG_SID = b"Couplet-arith-sid"
_TAG_CHI = b"Couplet-arith-chi"
_TAG_TRANSCRIPT = b"Couplet-arith-transcript"
_LABEL_COIN = b"arithcheck|"
_LABEL_ZETA = b"arithcheck-zeta|"
_COIN_BYTES = 32


def share_bytes(bits: int) -> int:
    """Encoded size of a value in ``Z_{2^bits}``."""
    return (int(bits) + 7) // 8


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def lemma2_bound(s: int) -> float:
    """The bound ``2^(-s + log2(s+1)) = (s+1) / 2^s`` of Lemma 2 (without the commitment term)."""
    return (int(s) + 1) / float(1 << int(s))


# --------------------------------------------------------------------------- #
# Sharings
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AShare:
    """One party's share of ``[[v]]`` over ``Z_{2^bits}``; see the module docstring."""

    value: int
    mac: int
    bits: int
    exact: bool = False
    sign_exact: bool = False

    def __post_init__(self) -> None:
        if not (_is_int(self.value) and _is_int(self.mac) and _is_int(self.bits)):
            raise ValueError("value, mac and bits must be integers")
        if self.bits < 1:
            raise ValueError("bits must be positive")
        if not (0 <= self.value < (1 << self.bits) and 0 <= self.mac < (1 << self.bits)):
            raise ValueError("share components must lie in [0, 2^bits)")
        if not isinstance(self.exact, bool) or not isinstance(self.sign_exact, bool):
            raise ValueError("exact and sign_exact must be bools")

    @property
    def modulus(self) -> int:
        return 1 << self.bits

    def _peer(self, other) -> "AShare":
        if not isinstance(other, AShare):
            raise TypeError("expected an AShare")
        if other.bits != self.bits:
            raise ValueError("sharings over different moduli cannot be combined (use reduce)")
        return other

    def __add__(self, other) -> "AShare":
        other = self._peer(other)
        mask = (1 << self.bits) - 1
        return AShare((self.value + other.value) & mask, (self.mac + other.mac) & mask, self.bits)

    def __sub__(self, other) -> "AShare":
        other = self._peer(other)
        mask = (1 << self.bits) - 1
        return AShare((self.value - other.value) & mask, (self.mac - other.mac) & mask, self.bits)

    def __neg__(self) -> "AShare":
        mask = (1 << self.bits) - 1
        return AShare(-self.value & mask, -self.mac & mask, self.bits)

    def scale(self, c: int) -> "AShare":
        """``[[c v]]`` for a public integer ``c`` (exact only for c = 0, or c = 1 on an exact sharing)."""
        if not _is_int(c):
            raise TypeError("the coefficient must be a public integer")
        mask = (1 << self.bits) - 1
        exact = c == 0 or (c == 1 and self.exact)
        return AShare((self.value * c) & mask, (self.mac * c) & mask, self.bits, exact)

    def __mul__(self, c) -> "AShare":
        if not _is_int(c):
            return NotImplemented
        return self.scale(c)

    __rmul__ = __mul__

    def add_public(self, c: int, role: int, delta_share: int) -> "AShare":
        """``[[v + c]]`` for a public integer ``c``: needs the party index and its ``Delta_b``."""
        if not _is_int(c):
            raise TypeError("the constant must be a public integer")
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        if not _is_int(delta_share) or delta_share < 0:
            raise ValueError("delta_share must be this party's share of the MAC key")
        mask = (1 << self.bits) - 1
        value = (self.value + c) & mask if role == 0 else self.value
        return AShare(value, (self.mac + delta_share * c) & mask, self.bits, c == 0 and self.exact)

    def reduce(self, n: int, s: int) -> "AShare":
        """The sharing modulo ``2^(n+s)`` (payload ``v mod 2^n``); local (section 2.3)."""
        if not (_is_int(n) and _is_int(s)) or n < 1 or s < 0:
            raise ValueError("need n >= 1 and s >= 0")
        bits = n + s
        if bits > self.bits:
            raise ValueError("cannot reduce to a larger modulus")
        mask = (1 << bits) - 1
        return AShare(self.value & mask, self.mac & mask, bits, self.exact and bits == self.bits)

    def with_exact(self, exact: bool = True) -> "AShare":
        """The same share with the exactness flag set by a caller that knows the identity."""
        return replace(self, exact=bool(exact))

    def with_sign_exact(self, sign_exact: bool = True) -> "AShare":
        """The same share with the ``sign_exact`` flag set by a caller that knows the bound."""
        return replace(self, sign_exact=bool(sign_exact))


def deal(value: int, delta: int, bits: int, *, exact: bool = False, rng=None) -> tuple:
    """Dealer-style sharing of the representative ``value`` (tests and formula checks only).

    ``delta`` is the full key ``Delta_0 + Delta_1``.  Returns the shares of
    P_0 and P_1.  ``rng`` offers ``getrandbits`` (default: the OS CSPRNG).
    """
    bits = int(bits)
    mod = 1 << bits
    draw = secrets.randbits if rng is None else rng.getrandbits
    v0, m0 = draw(bits), draw(bits)
    value = int(value) % mod
    return (AShare(v0, m0, bits, exact),
            AShare((value - v0) % mod, (int(delta) * value - m0) % mod, bits, exact))


def reconstruct(share0: AShare, share1: AShare) -> tuple:
    """(representative, sum of the MAC shares) of two shares (tests only)."""
    if share0.bits != share1.bits:
        raise ValueError("shares over different moduli")
    mod = 1 << share0.bits
    return (share0.value + share1.value) % mod, (share0.mac + share1.mac) % mod


# --------------------------------------------------------------------------- #
# ArithCheck
# --------------------------------------------------------------------------- #

def check_coefficients(seed: bytes, bits: int, count: int, s: int) -> list:
    """The coefficients ``chi_j`` in ``[0, 2^s)`` of one modulus from the tossed seed.

    ``chi_j`` is the j-th block of ``ceil(s / 8)`` bytes of
    ``sha256_expand("Couplet-arith-chi", seed, bits as 4 bytes LE)``, read
    little-endian and truncated to ``s`` bits.
    """
    nb = share_bytes(s)
    mask = (1 << s) - 1
    raw = hs.sha256_expand(nb * count, _TAG_CHI, bytes(seed), int(bits).to_bytes(4, "little"))
    return [int.from_bytes(raw[j * nb:(j + 1) * nb], "little") & mask for j in range(count)]


class ArithChecker:
    """Openings of SPDZ2k sharings with the deferred MAC check; see the module docstring.

    Attributes: ``role``, ``chan``, ``sid``, ``s``, ``dead``, ``openings``
    (openings so far), ``checks`` (checks run so far), ``pending`` (openings
    not yet verified) and ``last_opened`` (the representatives of the latest
    opening call, public transcript values).
    """

    def __init__(self, role: int, chan, sid: bytes, delta_share: int, s: int) -> None:
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        if chan.party != role:
            raise ValueError("role does not match the channel endpoint")
        if not isinstance(sid, (bytes, bytearray)):
            raise ValueError("sid must be bytes")
        if not _is_int(s) or s < 1:
            raise ValueError("s must be a positive integer")
        if not _is_int(delta_share) or not 0 <= delta_share < (1 << s):
            raise ValueError("delta_share must lie in [0, 2^s)")
        self.role = role
        self.chan = chan
        self.sid = bytes(sid)
        self.s = s
        self.dead = False
        self.openings = 0
        self.checks = 0
        self.last_opened: list = []
        self._delta = delta_share
        self._sid = hs.sha256(_TAG_SID, self.sid)
        self._pending: list = []      # (bits, opening index, opened representative, own MAC share)

    # -- life cycle ---------------------------------------------------------- #

    @property
    def pending(self) -> int:
        """Number of openings that no check has covered yet."""
        return len(self._pending)

    def abort(self) -> None:
        """Retire the checker (change G): drop the key share and the recorded MAC shares."""
        self.dead = True
        self._delta = None
        self._pending = []

    def _require_alive(self) -> None:
        if self.dead:
            raise ProtocolAbort("session aborted")

    def _step(self, fn):
        """Run one protocol step: any failure retires the checker; the channel phase is restored."""
        self._require_alive()
        phase = self.chan.phase
        try:
            return fn()
        except BaseException:
            self.abort()
            raise
        finally:
            self.chan.mark(phase)

    # -- the three steps a deviating party would change -------------------------- #

    def _outgoing_values(self, shares, first_index: int) -> list:
        """The value shares sent in an opening (honest: this party's shares)."""
        return [share.value for share in shares]

    def _recorded_mac(self, share: AShare, index: int) -> int:
        """The MAC share kept for the check of opening ``index`` (honest: this party's share)."""
        return share.mac

    def _outgoing_zeta(self, bits: int, zeta: int, chis: list, items: list, counter: int) -> int:
        """The value committed to in the check of modulus ``2^bits`` (honest: ``zeta``)."""
        return zeta

    # -- openings ------------------------------------------------------------- #

    def _check_shares(self, shares) -> list:
        shares = list(shares)
        for share in shares:
            if not isinstance(share, AShare):
                raise ValueError("expected AShare objects")
            if share.bits <= self.s:
                raise ValueError("the modulus must be larger than 2^s")
        return shares

    def _exchange(self, shares: list) -> list:
        """One exchange of the value shares of ``shares``; records the openings."""
        chan = self.chan
        chan.mark(PHASE_OPEN)
        first = self.openings
        self.openings += len(shares)
        sent = [int(v) for v in self._outgoing_values(shares, first)]
        sizes = [share_bytes(share.bits) for share in shares]
        msg = b"".join((v & ((1 << share.bits) - 1)).to_bytes(nb, "little")
                       for v, share, nb in zip(sent, shares, sizes))
        reply = chan.exchange(msg)
        if len(reply) != len(msg):
            raise ProtocolAbort("opening: message has the wrong length")
        opened = []
        pos = 0
        for j, (share, nb) in enumerate(zip(shares, sizes)):
            peer = int.from_bytes(reply[pos:pos + nb], "little")
            pos += nb
            if peer >> share.bits:
                raise ProtocolAbort("opening: value share out of range")
            value = (sent[j] + peer) & ((1 << share.bits) - 1)
            opened.append(value)
            self._pending.append((share.bits, first + j, value, self._recorded_mac(share, first + j)))
        self.last_opened = list(opened)
        return opened

    def open_deferred(self, share: AShare) -> int:
        """Open ``share`` now, verify at the next ``check()``; returns the representative."""
        return self.open_deferred_many([share])[0]

    def open_deferred_many(self, shares) -> list:
        """As ``open_deferred`` for several sharings in ONE exchange."""
        shares = self._check_shares(shares)
        self._require_alive()
        if not shares:
            return []
        return self._step(lambda: self._exchange(shares))

    # -- the check ------------------------------------------------------------- #

    def _run_check(self) -> None:
        if not self._pending:
            return                                   # nothing was opened: no message
        chan = self.chan
        chan.mark(PHASE_CHECK)
        counter = self.checks
        self.checks += 1
        cbytes = counter.to_bytes(8, "little")
        groups: dict = {}
        parts = []
        for bits, index, value, mac in self._pending:
            groups.setdefault(bits, []).append((index, value, mac))
            parts.append(bits.to_bytes(4, "little") + index.to_bytes(8, "little")
                         + value.to_bytes(share_bytes(bits), "little"))
        transcript = hs.sha256(_TAG_TRANSCRIPT, self._sid, cbytes, *parts)
        # The coefficients are tossed only now: every opened value is fixed and both
        # parties agree on the list (it is part of the label of the coin toss).
        seed = hs.coin_toss(chan, self._sid, _LABEL_COIN + cbytes + transcript, _COIN_BYTES)
        own = []
        for bits in sorted(groups):
            items = groups[bits]
            mod = 1 << bits
            chis = check_coefficients(seed, bits, len(items), self.s)
            y = sum(chi * value for chi, (_, value, _) in zip(chis, items)) % mod
            zeta = (sum(chi * mac for chi, (_, _, mac) in zip(chis, items)) - self._delta * y) % mod
            own.append((bits, zeta, int(self._outgoing_zeta(bits, zeta, chis, items, counter))))
        msg = b"".join(out.to_bytes(share_bytes(bits), "little") for bits, _, out in own)
        peer = hs.committed_exchange(chan, self._sid, _LABEL_ZETA + cbytes, msg)
        if len(peer) != len(msg):
            raise ProtocolAbort("ArithCheck: message has the wrong length")
        pos = 0
        ok = True
        for bits, _, sent in own:          # an honest party has sent its zeta itself
            nb = share_bytes(bits)
            other = int.from_bytes(peer[pos:pos + nb], "little")
            pos += nb
            if other >> bits:
                raise ProtocolAbort("ArithCheck: value out of range")
            ok &= (sent + other) % (1 << bits) == 0
        if not ok:
            raise ProtocolAbort("ArithCheck failed")
        self._pending = []

    def check(self) -> None:
        """ArithCheck over every opening since the last check (section 2.3)."""
        self._require_alive()
        if not self._pending:
            return
        self._step(self._run_check)

    # -- release ---------------------------------------------------------------- #

    def open_final(self, share: AShare, outmask: AShare | None = None) -> int:
        """Release ``share`` (section 5.3): check, open, check; returns ``v mod 2^(bits - s)``."""
        return self.open_final_many([share], [outmask])[0]

    def open_final_many(self, shares, outmasks=None) -> list:
        """Release several sharings with one opening exchange and the two checks.

        ``outmasks[j]`` is the output mask ``[[2^k rho]]`` of sharing ``j`` (a
        sharing over the same modulus whose components are multiples of
        ``2^k``, never used before) and must be given exactly for the sharings
        that are not exact.
        """
        shares = self._check_shares(shares)
        outmasks = [None] * len(shares) if outmasks is None else list(outmasks)
        if len(outmasks) != len(shares):
            raise ValueError("one output mask entry per sharing is required")
        release = []
        for share, mask in zip(shares, outmasks):
            k = share.bits - self.s
            if share.exact:
                if mask is not None:
                    raise ValueError("an exact sharing is opened without an output mask")
                release.append(share)
                continue
            if mask is None:
                raise ValueError("a sharing that is not exact must be opened with an output mask")
            if not isinstance(mask, AShare) or mask.bits != share.bits:
                raise ValueError("the output mask must be a sharing over the same modulus")
            if mask.value % (1 << k) or mask.mac % (1 << k):
                raise ValueError("output mask shares must be multiples of 2^k")
            release.append(share + mask)
        self._require_alive()
        if not shares:
            return []

        def run():
            self._run_check()                 # every deferred opening first
            opened = self._exchange(release)  # the release itself
            self._run_check()                 # exactly this opening
            return opened

        opened = self._step(run)
        return [value % (1 << (share.bits - self.s)) for value, share in zip(opened, shares)]
