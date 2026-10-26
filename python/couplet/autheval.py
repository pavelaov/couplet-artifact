"""Pi_AuthEval: the party-local driver of the reactive functionality F_AuthEval
(DESIGN.md sections 5.3, 5.4 and 6; changes D, E, F, G).

``AuthEval(role, chan, sid, k=64, s=64, prep='real'|'ideal', sigma=40, dealer=None,
allow_refill=False)`` runs ONE party.  The other party issues the same commands
with the same public arguments and its own shares.  It owns a
``couplet.session.Session`` (Boolean engine, preprocessing, the session MAC
key share ``Delta_b``), a ``couplet.state.Registry`` (state machine) and,
after ``setup``, a ``couplet.spdz2k.ArithChecker`` (openings and ArithCheck).

Commands of F_AuthEval and what the protocol does:

    setup(reserve_triples=0, reserve_outmasks=0)
        Session setup (preprocessing, ``Delta_b``, the persistent wires of
        ``Delta``) and the ONE request for all AND triples of the session:
        those of the setup circuit, ``reserve_triples`` for the planned
        ``Key`` / ``Gate`` commands and the triples of ``reserve_outmasks``
        on-demand output masks (see "Triple budget" below).
    key(kid, family, n, alpha_share, beta_share=None, index_mode='xor')
        ``Key``: Pi_Gen for one DPF (``family='dpf'``, ``f(x) = beta [x = alpha]``)
        or DCF (``'dcf'``, ``beta [x < alpha]``).  ``beta_share=None`` fixes the
        payload to the public constant 1.  The record becomes CERTIFIED when
        Pi_Gen returns, i.e. after the MAC check of its outputs (change F).
        ``key_batch`` generates several keys with one circuit execution.
    query(kid, x) -> AShare
        ``Query`` at a public point: local ``Eval``; the result is an exact
        sharing of ``f(x)`` over ``Z_{2^w}``.  ``query_all(kid)`` evaluates
        the whole domain.
    gate(gid, spec)
        ``Gate``: Pi_Gen for the one-time material of the catalogue gate
        ``spec`` (a ``couplet.gates.GateSpec``; a plain
        ``circuits_gen.gate_descriptor`` is accepted for material that only
        masks a value and has no function to apply).  ``gate_batch`` generates
        several independent gates of one kind with one circuit execution.
    mask_shared(gid, x_share) -> x_hat
        ``Mask`` of a shared input ``x = x_0 + x_1 mod 2^n``: the parties
        exchange ``x_b + r_b mod 2^n`` by commit-then-open
        (``hashes.committed_exchange``), so that neither can choose its share
        after having seen the other one; no MAC (a wrong share that is
        committed to is an input substitution).  Two rounds.
    mask_wire(gid, share) -> x_hat
        ``Mask`` of a wire: ``[[x]]`` is reduced to ``Z_{2^(n+s)}``, the
        offset ``[[r']]`` (uniform over that whole ring, change D) is added
        and the sum is opened with ``ArithChecker.open_deferred``; the MAC
        check is deferred to the next ``open``.  ``x_hat`` is the opened
        value modulo ``2^n``.  The gate must have been generated with an
        authenticated input.
    apply(gid) -> AShare
        ``Apply``: the local evaluation of the gate on ``x_hat``
        (a list of sharings for ``multi_interval``).
    lin(terms, const=0) -> AShare
        ``Lin``: ``sum c_i [[v_i]] + const`` for public integers.
    open(share) -> int
        ``Open``, in the order of the protocol figure of the paper: (1)
        ArithCheck on every deferred opening; (2) the output mask
        ``[[2^k rho]]`` of a sharing that is not exact is taken from the pool
        (change E), and if the pool is short the missing masks are generated
        by Pi_Gen at this point, i.e. only AFTER the deferred openings have
        been verified; (3) the opening of the value; (4) ArithCheck on that
        opening; returns ``v mod 2^k``.  ``open_many`` releases several
        values at once.  In strict mode a mask generation that the reserved
        pool cannot serve is refused before step (1), without any message.
    abort()
        ``Abort``: the session is retired.

One-time use (section 5.4).  ``mask_shared`` / ``mask_wire`` validate their
arguments, then mark the gate USED in the registry and only then send the
masked opening.  A second ``Mask`` on the same gate raises
``state.GateUsedError`` and sends nothing, also after a failed or interrupted
first attempt.  Output masks are consumed the same way: each is taken from a
pool exactly once.  Masks come from gates generated with ``n_outmask > 0`` and
otherwise from Pi_Gen on demand (``reserve_outmasks`` generates a batch).

Abort (change G).  Any ``ProtocolAbort`` and any other failure inside a
protocol step moves the driver, the session, the checker and the registry to
ABORTED; the MAC key shares and all key and gate material are dropped and
every later command raises ``ProtocolAbort`` without sending anything.
Argument errors (``ValueError``) and state errors (``state.StateError``) are
raised before any message and do not abort.  Sharings handed out earlier are
plain data; they cannot be opened once the session is ABORTED.

Sharings are ``AShare`` objects held by the caller (the ``Val[id]`` of the
functionality).  The exactness flag of a sharing is public and derived from
its history; it decides whether ``open`` adds an output mask.

Triple budget (strict mode).  The bucketing bound 2^-sigma holds per
preprocessing request for AND triples (Lemma 4), so the driver is strict by
default (``allow_refill=False``): all triples of the session are requested
once, in ``setup``, and a ``Key`` / ``Gate`` / output-mask generation that
does not fit into the reserved pool raises ``TripleBudgetError`` (a
``ValueError``) before any message and before any change of state; the
session stays ACTIVE and the caller reserves (``session.reserve_triples`` is an
explicit further population) or starts a session with the right budget.  The
budget is public and both parties compute the same number:

    reserve_triples  = sum of key_budget(...) and gate_budget(...) over the
                       planned commands (``protocol.and_budget`` for lists of
                       planned calls);
    reserve_outmasks = number of non-exact values that will be opened and
                       whose mask does not come from a gate generated with
                       ``n_outmask > 0``.

``triple_requests`` (also in ``info()``) is the number of populations drawn; a
session that reserved exactly has ``triple_requests == 1`` and
``info()["triples_unused"] == 0`` at its end.  With ``allow_refill=True`` a
short pool is refilled silently, as before, and the count increases.

Shared-input Mask on the wire.  Sub-session identifier
``sid_mask = SHA-256(frame("Couplet-mask-sid", sid))``; label
``b"mask|" + gid.to_bytes(8, "little")``; message ``x_b + r_b mod 2^n`` as
``ceil(n/8)`` bytes little-endian; round 1: 32-byte commitments bound to
(``sid_mask``, label, party index); round 2: ``opening || message``.  A
deviating party is modelled by a subclass that overrides ``_mask_exchange``.

Channel phases: the Boolean engine and the preprocessing book their own
phases; ``mask`` holds the two messages (commitment, opening) of a
shared-input opening; ``arith-open`` / ``arith-check`` belong to the checker.
"""

from __future__ import annotations

import contextlib
from collections import deque

from . import circuits_gen as cg
from . import fss
from . import gates
from . import hashes as hs
from . import protocol as pg
from .abb import TripleBudgetError
from .hashes import ProtocolAbort
from .session import ABORTED, ACTIVE, Session
from .spdz2k import AShare, ArithChecker
from .state import GateUsedError, Registry, StateError

__all__ = ["ABORTED", "ACTIVE", "AShare", "GateUsedError", "ProtocolAbort", "StateError",
           "TripleBudgetError", "AuthEval"]

PHASE_MASK = "mask"
_TAG_MASK_SID = b"Couplet-mask-sid"
_LABEL_MASK = b"mask|"
_FAMILIES = ("dpf", "dcf")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


class AuthEval:
    """One party of Pi_AuthEval; see the module docstring.

    Attributes: ``role``, ``chan``, ``sid``, ``k``, ``s``, ``w``, ``session``,
    ``registry``, ``checker`` (after ``setup``), ``state``.
    Keyword-only arguments: ``state_path`` (JSON file of the registry
    counters), ``rng`` / ``delta_share`` / ``cheat`` / ``setup_adder`` (passed
    to the session; test hooks), ``checker_factory`` (class or callable with
    the signature of ``ArithChecker``; the adversaries of ``couplet.attacks``
    substitute a deviating checker here), ``allow_refill`` (default False =
    strict mode: a short triple pool raises ``TripleBudgetError`` instead of
    requesting a new bucketing population).
    """

    def __init__(self, role: int, chan, sid: bytes, k: int = 64, s: int = 64, prep="real",
                 sigma: int = 40, dealer=None, *, state_path=None, rng=None, cheat=None,
                 delta_share=None, setup_adder: str = "ripple", checker_factory=None,
                 allow_refill: bool = False) -> None:
        self.session = Session(role, chan, sid, k=k, s=s, prep=prep, sigma=sigma, dealer=dealer,
                               rng=rng, cheat=cheat, delta_share=delta_share,
                               setup_adder=setup_adder, allow_refill=allow_refill)
        self.role = role
        self.chan = chan
        self.sid = bytes(sid)
        self._sid_mask = hs.sha256(_TAG_MASK_SID, self.sid)    # sub-session of the Mask command
        self.k, self.s, self.w = self.session.k, self.session.s, self.session.w
        self.registry = Registry(self.sid, state_path)
        self.checker = None
        self._checker_factory = ArithChecker if checker_factory is None else checker_factory
        self._outmasks: deque = deque()
        self._rng = rng
        if self.registry.state != ACTIVE:           # a persisted ABORTED session stays dead
            self.abort()

    # -- state ------------------------------------------------------------------ #

    @property
    def state(self) -> str:
        return ACTIVE if (self.registry.state == ACTIVE and self.session.state == ACTIVE) else ABORTED

    @property
    def delta_share(self):
        """This party's ``Delta_b`` (None before ``setup`` and after an abort)."""
        return self.session.delta_share

    @property
    def allow_refill(self) -> bool:
        """False in strict mode (the default): a short triple pool is an error, not a refill."""
        return self.session.allow_refill

    @property
    def triple_requests(self) -> int:
        """Preprocessing requests for AND triples so far (bucketing populations; 1 if reserved)."""
        return self.session.triple_requests

    # -- triple budget (public parameters only; no message) ------------------------ #

    def key_budget(self, family: str, n: int, *, secret_payload: bool = False,
                   index_mode: str = "xor", count: int = 1) -> int:
        """AND triples of ``key`` / ``key_batch`` for ``count`` keys (``secret_payload``: a ``beta_share`` is given)."""
        return pg.planned_key(family, n, self.k, self.s, index_mode,
                              "secret" if secret_payload else "public_one", count)["triples"]

    def gate_budget(self, spec, count: int = 1) -> int:
        """AND triples of ``gate`` / ``gate_batch`` for ``count`` gates of ``spec`` (or a descriptor)."""
        return pg.planned_gate(spec, count)["triples"]

    def outmask_budget(self, count: int = 1) -> int:
        """AND triples of ``count`` on-demand output masks (what ``reserve_outmasks=count`` adds at setup)."""
        return pg.planned_outmasks(self.k, self.s, count)["triples"]

    def _require_triples(self, descriptor, count: int) -> None:
        """Strict mode: refuse a generation that the reserved pool cannot serve (nothing is sent)."""
        self.session.engine.require_triples(pg.and_count(descriptor) * count)

    def _id_refused(self, kind: str, requested) -> bool:
        """True if the registry will refuse ``requested`` as the next ``kind`` identifier.

        Used to order the two refusals that precede a generation: a bad
        identifier is reported by the registry (``StateError`` / ``ValueError``)
        before the triple budget is looked at.
        """
        if requested is None:
            return False
        try:
            if isinstance(requested, bool) or int(requested) != requested or int(requested) < 0:
                return True
        except (TypeError, ValueError):
            return True
        return int(requested) < self.registry.counters()["next"][kind]

    def _broken(self) -> bool:
        ses = self.session
        return (self.registry.state != ACTIVE or ses.state != ACTIVE or ses.engine.dead
                or getattr(ses.prep, "aborted", False)
                or (self.checker is not None and self.checker.dead))

    def _require_active(self) -> None:
        if self._broken():
            self.abort()
            raise ProtocolAbort("session aborted")

    def _require_setup(self) -> None:
        self._require_active()
        if self.checker is None:
            raise StateError("setup() has not been run")

    def abort(self) -> None:
        """``Abort``: retire the session (change G)."""
        self._outmasks.clear()
        if self.checker is not None:
            self.checker.abort()
        self.session.abort()
        self.registry.abort()

    @contextlib.contextmanager
    def _step(self):
        """A protocol step: any failure after the first message, and any abort, retires the session."""
        self._require_active()
        chan = self.chan
        before = (chan.msgs_sent, chan.msgs_recv)
        try:
            yield
        except (ValueError, StateError):
            if (chan.msgs_sent, chan.msgs_recv) != before or self._broken():
                self.abort()
            raise
        except BaseException:
            self.abort()
            raise

    # -- Setup --------------------------------------------------------------------- #

    def setup(self, reserve_triples: int = 0, reserve_outmasks: int = 0) -> None:
        """``Setup``: preprocessing setup, ``Delta_b`` and the session wires of ``Delta``.

        ``reserve_triples`` AND triples for the planned ``Key`` / ``Gate`` commands
        and the triples of ``reserve_outmasks`` on-demand output masks are drawn
        together with those of the setup circuit, in ONE bucketing population.
        The masks themselves are generated when they are needed.
        """
        self._require_active()
        if self.checker is not None:
            raise StateError("setup() was already run")
        for value, name in ((reserve_triples, "reserve_triples"), (reserve_outmasks, "reserve_outmasks")):
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        with self._step():
            self.session.setup(reserve_triples, reserve_outmasks)
            self.checker = self._checker_factory(self.role, self.chan, self.sid,
                                                 self.session.delta_share, self.s)

    # -- Key and Query -------------------------------------------------------------- #

    def key(self, kid, family: str, n: int, alpha_share: int, beta_share=None, *,
            index_mode: str = "xor") -> int:
        """``Key(kid, family, n; alpha_b, beta_b)``; returns ``kid``."""
        return self.key_batch(kid, family, n, [alpha_share],
                              None if beta_share is None else [beta_share],
                              index_mode=index_mode)[0]

    def key_batch(self, kid, family: str, n: int, alpha_shares, beta_shares=None, *,
                  index_mode: str = "xor") -> list:
        """Several keys of one family from ONE circuit execution; returns their identifiers."""
        self._require_setup()
        if family not in _FAMILIES:
            raise ValueError("family must be 'dpf' or 'dcf'")
        alphas = list(alpha_shares)
        count = len(alphas)
        if count < 1:
            raise ValueError("at least one key is required")
        if not _is_int(n) or not 1 <= n <= 64:
            raise ValueError("n must be in 1..64")
        if index_mode not in ("xor", "add"):
            raise ValueError("index_mode must be 'xor' or 'add'")
        for a in alphas:
            if not _is_int(a) or not 0 <= a < (1 << n):
                raise ValueError("alpha_share must lie in [0, 2^n)")
        betas = None
        if beta_shares is not None:
            betas = list(beta_shares)
            if len(betas) != count:
                raise ValueError("one beta_share per key is required")
            for b in betas:
                if not _is_int(b) or not 0 <= b < (1 << self.k):
                    raise ValueError("beta_share must lie in [0, 2^k)")
        beta_mode = "public_one" if betas is None else "secret"
        gen = pg.gen_dpf if family == "dpf" else pg.gen_dcf
        make = cg.dpf_descriptor if family == "dpf" else cg.dcf_descriptor
        if not self._id_refused("kid", kid):                 # strict mode: before any change of state
            self._require_triples(make(n, self.k, self.s, index_mode, beta_mode), count)
        first = self.registry.begin_key(kid, count)          # GENERATING
        with self._step():
            keys = gen(self.session, n, index_mode, beta_mode, alphas, betas, batch=count,
                       rng=self._rng)
        for j, key in enumerate(keys):                       # the output check has passed
            self.registry.certify_key(first + j, key, family=family, n=n, beta_mode=beta_mode)
        return list(range(first, first + count))

    def _eval_key(self, kid: int):
        self._require_setup()
        return self.registry.key(kid)

    def query(self, kid: int, x: int) -> AShare:
        """``Query(kid, x)`` at a public point: this party's exact share of ``[[f(x)]]``."""
        rec = self._eval_key(kid)
        n = rec.info["n"]
        if not _is_int(x) or not 0 <= x < (1 << n):
            raise ValueError("x must lie in [0, 2^n)")
        self.registry.next_qid()
        point = fss.dpf_eval if rec.info["family"] == "dpf" else fss.dcf_eval
        value, tag = point(self.role, rec.key, x)
        return AShare(value, tag, self.w, True)

    def query_all(self, kid: int) -> list:
        """``Query`` at every point of the domain (``EvalAll``): a list of ``2^n`` exact shares."""
        rec = self._eval_key(kid)
        self.registry.next_qid()
        full = fss.dpf_eval_all if rec.info["family"] == "dpf" else fss.dcf_eval_all
        y, tau = full(self.role, rec.key)
        return [AShare(int(v), int(t), self.w, True) for v, t in zip(y, tau)]

    # -- Gate ------------------------------------------------------------------------ #

    def gate(self, gid, spec) -> int:
        """``Gate(gid, G)``: generate the one-time material of ``spec``; returns ``gid``."""
        return self.gate_batch(gid, spec, 1)[0]

    def gate_batch(self, gid, spec, count: int) -> list:
        """``count`` independent gates of one kind from ONE circuit execution."""
        self._require_setup()
        if isinstance(spec, gates.GateSpec):
            desc = spec.descriptor()
        elif isinstance(spec, dict):
            try:
                desc = cg.canonical_descriptor(spec)
            except (KeyError, TypeError) as exc:
                raise ValueError(f"incomplete gate descriptor ({exc})") from None
            if desc["kind"] != "gate":
                raise ValueError("expected a gate descriptor")
            spec = None                                      # material without a function
        else:
            raise ValueError("spec must be a couplet.gates.GateSpec or a gate descriptor")
        if (desc["k"], desc["s"]) != (self.k, self.s):
            raise ValueError("the gate does not use the session's k and s")
        if not _is_int(count) or count < 1:
            raise ValueError("count must be a positive integer")
        if not self._id_refused("gid", gid):                 # strict mode: before any change of state
            self._require_triples(desc, count)
        first = self.registry.begin_gate(gid, count)         # GENERATING
        with self._step():
            mats = pg.gen_gate(self.session, desc, batch=count, rng=self._rng)
        delta = self.session.delta_share
        for j, mat in enumerate(mats):
            self.registry.certify_gate(first + j, mat, spec, gates.GateEval(mat, delta))
            for value, tag in mat.outmask_shares:            # one-time output masks of the gate
                self._outmasks.append(AShare(value, tag, self.w))
        return list(range(first, first + count))

    # -- Mask -------------------------------------------------------------------------- #

    def _mask_exchange(self, gid: int, share: int, nbytes: int) -> bytes:
        """Commit-then-open exchange of the masked shares of gate ``gid``; returns the peer's message.

        ``share`` is this party's ``x_b + r_b mod 2^n``.  Both parties first
        exchange commitments bound to (``sid_mask``, ``b"mask|" || gid``, own
        party index) and only then the openings, so that a rushing party
        cannot fix ``x_hat`` by choosing its share after it has seen the other
        one.  The honest code of this step is the whole method; a deviating
        party overrides it.
        """
        label = _LABEL_MASK + int(gid).to_bytes(8, "little")
        return hs.committed_exchange(self.chan, self._sid_mask, label,
                                     share.to_bytes(nbytes, "little"))

    def mask_shared(self, gid: int, x_share: int) -> int:
        """``Mask(gid, shared input)``: opens ``x + r mod 2^n`` (commit-then-open, no MAC); returns ``x_hat``."""
        self._require_setup()
        rec = self.registry.gate(gid)
        if rec.material is None:
            raise StateError(f"gate {gid} is not certified")
        n = rec.material.n
        size = 1 << n
        if not _is_int(x_share) or not 0 <= x_share < size:
            raise ValueError("x_share must lie in [0, 2^n)")
        if int(gid) >> 64:
            raise ValueError("the gate identifier does not fit the 8-byte label of the Mask commitment")
        rec = self.registry.use_gate(gid)                    # USED from here on, whatever happens
        nb = (n + 7) // 8
        with self._step():
            phase = self.chan.phase
            self.chan.mark(PHASE_MASK)
            try:
                mine = (x_share + rec.material.rp_share) % size
                reply = self._mask_exchange(gid, mine, nb)
            finally:
                self.chan.mark(phase)
            if len(reply) != nb:
                raise ProtocolAbort("mask: message has the wrong length")
            peer = int.from_bytes(reply, "little")
            if peer >= size:
                raise ProtocolAbort("mask: share out of range")
            x_hat = (mine + peer) % size
        self.registry.set_masked(gid, x_hat, "shared")
        return x_hat

    def mask_wire(self, gid: int, share: AShare) -> int:
        """``Mask(gid, wire)``: opens ``x + r'`` in ``Z_{2^(n+s)}`` (check deferred); returns ``x_hat``."""
        self._require_setup()
        rec = self.registry.gate(gid)
        mat = rec.material
        if mat is None:
            raise StateError(f"gate {gid} is not certified")
        if not mat.auth_input:
            raise ValueError("the gate was generated without an authenticated input offset")
        n = mat.n
        if not isinstance(share, AShare) or share.bits < n + self.s:
            raise ValueError("expected a sharing over a modulus of at least 2^(n+s)")
        rec = self.registry.use_gate(gid)                    # USED from here on, whatever happens
        with self._step():
            offset = AShare(mat.rp_share, mat.rp_tag_share, n + self.s)   # [[r']], uniform in the ring
            opened = self.checker.open_deferred(share.reduce(n, self.s) + offset)
        x_hat = opened % (1 << n)
        self.registry.set_masked(gid, x_hat, "wire", opened)
        return x_hat

    # -- Apply and Lin ------------------------------------------------------------------- #

    def apply(self, gid: int):
        """``Apply(gid)``: this party's share of ``[[G(x)]]`` (local)."""
        self._require_setup()
        rec = self.registry.masked_gate(gid)
        if rec.spec is None or rec.spec.kind == "offset_only":
            raise StateError(f"gate {gid} has no function")
        return rec.spec.evaluate(rec.evaluator, rec.x_hat)

    def lin(self, terms, const: int = 0) -> AShare:
        """``Lin``: ``sum c_i [[v_i]] + const`` for pairs ``(c_i, share_i)`` and a public ``const``."""
        self._require_setup()
        if not _is_int(const):
            raise ValueError("const must be a public integer")
        pairs = [(c, sh) for c, sh in terms]
        for c, sh in pairs:
            if not _is_int(c) or not isinstance(sh, AShare) or sh.bits != self.w:
                raise ValueError("terms are pairs (public integer, sharing over Z_{2^w})")
        if len(pairs) == 1 and pairs[0][0] == 1 and const == 0:
            return pairs[0][1]
        acc = AShare(0, 0, self.w)
        for c, sh in pairs:
            acc = acc + sh.scale(c)
        if const:
            acc = acc.add_public(const, self.role, self.session.delta_share)
        return acc.with_exact(False)

    # -- Open ---------------------------------------------------------------------------- #

    @property
    def outmasks_available(self) -> int:
        return len(self._outmasks)

    def reserve_outmasks(self, count: int) -> None:
        """Generate ``count`` output masks ``[[2^k rho]]`` with one circuit execution."""
        self._require_setup()
        if not _is_int(count) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if count == 0:
            return
        # one circuit with a single mask, run on ``count`` instances (the same transcript as the
        # native implementation, which loads this circuit instead of compiling one per count)
        desc = pg.outmask_descriptor(self.k, self.s)
        self._require_triples(desc, count)
        with self._step():
            mats = pg.gen_gate(self.session, desc, batch=count, rng=self._rng)
        for mat in mats:
            for value, tag in mat.outmask_shares:
                self._outmasks.append(AShare(value, tag, self.w))

    def open(self, share: AShare) -> int:
        """``Open``: check, open (masked unless exact), check; returns ``v mod 2^k``."""
        return self.open_many([share])[0]

    def open_many(self, shares) -> list:
        """``Open`` of several sharings with one opening exchange and the two checks.

        Order: ArithCheck on the deferred openings, then (if the pool is
        short) the generation of the missing output masks, then the opening
        and its ArithCheck.
        """
        self._require_setup()
        shares = list(shares)
        for sh in shares:
            if not isinstance(sh, AShare) or sh.bits != self.w:
                raise ValueError("expected sharings over Z_{2^w}")
        if not shares:
            return []
        missing = sum(not sh.exact for sh in shares) - len(self._outmasks)
        if missing > 0:      # strict mode: refused before any message (nothing has been sent)
            self._require_triples(pg.outmask_descriptor(self.k, self.s), missing)
        with self._step():
            self.checker.check()                 # step 1: every deferred opening is verified first
        if missing > 0:
            self.reserve_outmasks(missing)       # step 2: masks from Pi_Gen, after the check
        masks = [None if sh.exact else self._outmasks.popleft() for sh in shares]   # used once
        with self._step():
            return self.checker.open_final_many(shares, masks)    # steps 3 and 4

    # -- reporting ------------------------------------------------------------------------- #

    def info(self) -> dict:
        """Public parameters and counters (no secrets)."""
        out = self.session.info()
        out.update({"state": self.state, "registry": self.registry.summary(),
                    "counters": self.registry.counters(),
                    "outmasks_available": len(self._outmasks),
                    "arith_openings": 0 if self.checker is None else self.checker.openings,
                    "arith_checks": 0 if self.checker is None else self.checker.checks,
                    "arith_pending": 0 if self.checker is None else self.checker.pending})
        return out
