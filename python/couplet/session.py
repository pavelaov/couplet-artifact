"""Session of one party: preprocessing, Boolean engine and the session MAC key
(DESIGN.md sections 1, 4, 5.1, 5.4; changes G and H).

``Session(role, chan, sid, k=64, s=64, prep='real'|'ideal', sigma=40, dealer=None,
allow_refill=True)`` owns, for one party and one session identifier ``sid``:

  * the Boolean preprocessing (``couplet.prep.RealPrep`` for ``prep='real'``;
    ``IdealPrep`` on the shared in-process ``dealer`` for ``prep='ideal'``; an
    object with the same interface is accepted as well);
  * the Boolean engine ``engine`` (``couplet.abb.Engine``);
  * this party's arithmetic MAC key share ``delta_share`` = Delta_b, uniform in
    ``[0, 2^s)``, and the persistent authenticated wires ``delta_wires`` of
    ``Delta = Delta_0 + Delta_1`` (``s + 1`` bits, never reduced, never opened).

``setup(reserve_triples=0, reserve_outmasks=0)`` runs ``prep.setup()``, samples
Delta_b and evaluates the session-setup circuit
``circuits_gen.build_session_setup(s)`` once: each party inputs its Delta_b,
the sum stays in the session as RETAIN wires (change H).  With the default
ripple-carry adder the setup circuit has ``s`` AND gates and AND depth ``s``
(``s + 3`` rounds, once per session); ``setup_adder="sklansky"`` trades more
AND gates for a logarithmic depth.  Both parties must use the same adder.

Bucketing populations (DESIGN.md 1 and 7.5, Lemma 4).  The statistical bound
2^-sigma holds PER preprocessing request for AND triples.  ``setup`` draws, in
ONE request, the triples of the setup circuit, ``reserve_triples`` further
ones and the triples of ``reserve_outmasks`` on-demand output masks
(``protocol.outmask_and_count(k, s)`` each); a session whose later circuits
fit into this reservation consists of a single population and is at 2^-sigma
as a whole.  ``protocol.and_budget`` computes ``reserve_triples`` for a planned
list of generation calls.  ``triple_requests`` counts the requests made so
far (also reported by ``info()``, together with the resulting bound).  What
happens when the pool is too small for an evaluation is decided by
``allow_refill``: with ``True`` (the default of this class) the engine
requests the missing triples, which is one more population and one more
count; with ``False`` (strict mode, the default of ``autheval.AuthEval`` and of
the benchmark drivers) ``evaluate`` raises ``abb.TripleBudgetError`` before any
message, the session stays ``ACTIVE``, and the caller has to reserve.
``reserve_triples(n)`` after ``setup`` is an explicit further request in both
modes.

``evaluate(circuit, my_inputs, batch=1, bound=None)`` runs one circuit on the
engine; a BOUND input named ``delta`` is bound to the session wires unless
the caller binds it explicitly.  Both parties compare the circuit digest as
part of the first exchange of the evaluation (inside the engine).

State (section 5.4): ``state`` is ``'ACTIVE'`` or ``'ABORTED'`` (absorbing).
Any ``ProtocolAbort`` raised inside ``setup``, ``evaluate`` or a ``guard()``
block moves the session to ``ABORTED``: the key shares, the session wires and
the preprocessing are dropped and every later call raises ``ProtocolAbort``
(change G).  Layers built on top of the session (arithmetic openings, gates)
wrap their protocol steps in ``with session.guard():`` or call ``abort()``.
Material handed out earlier (keys, shares) must not be used once the session
is ``ABORTED``; callers check ``state``.

``next_id(kind)`` hands out the monotone identifiers ``kid``, ``gid``, ``qid``.

Channel phases: the setup evaluation is booked under ``setup-prep``,
``setup-input``, ``setup-online``, ``setup-outcheck``; later evaluations under
the engine's default names; the preprocessing books its own phases.

Test hooks: ``rng`` (object with ``randbytes``) makes this party's secret
choices reproducible; ``delta_share`` fixes Delta_b; ``cheat`` is passed to the
real preprocessing and to the engine (default: honest).
"""

from __future__ import annotations

import contextlib
import math
import secrets

from . import circuits_gen as cg
from .abb import Engine, EvalResult, TripleBudgetError, ints_to_bits
from .circuit import BOUND, Circuit
from .hashes import ProtocolAbort
from .prep import IdealPrep, RealPrep
from .prep.shares import as_rand

__all__ = ["ABORTED", "ACTIVE", "ProtocolAbort", "Session", "TripleBudgetError"]

ACTIVE = "ACTIVE"
ABORTED = "ABORTED"
_ID_KINDS = ("kid", "gid", "qid")


class Session:
    """One party's session; see the module docstring."""

    def __init__(self, role: int, chan, sid: bytes, k: int = 64, s: int = 64, prep="real",
                 sigma: int = 40, dealer=None, *, rng=None, cheat=None, delta_share=None,
                 setup_adder: str = "ripple", allow_refill: bool = True) -> None:
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        if chan.party != role:
            raise ValueError("role does not match the channel endpoint")
        if not isinstance(sid, (bytes, bytearray)):
            raise ValueError("sid must be bytes")
        k, s, sigma = int(k), int(s), int(sigma)
        if k < 1 or s < 1 or 2 * (k + s) > 320:
            raise ValueError("need k >= 1, s >= 1 and 2(k+s) <= 320")
        if sigma < 1:
            raise ValueError("sigma must be positive")
        if delta_share is not None and not 0 <= int(delta_share) < (1 << s):
            raise ValueError("delta_share must lie in [0, 2^s)")
        if setup_adder not in cg.ADDERS:
            raise ValueError(f"setup_adder must be one of {cg.ADDERS}")
        self.role = role
        self.chan = chan
        self.sid = bytes(sid)
        self.k = k
        self.s = s
        self.w = k + s
        self.sigma = sigma
        self.setup_adder = setup_adder
        self._rand = as_rand(rng)
        self._has_rng = rng is not None
        if prep == "real":
            if dealer is not None:
                raise ValueError("a dealer is used in ideal mode only")
            self.prep_mode = "real"
            self.prep = RealPrep(role, chan, self.sid, sigma, rng, cheat=cheat)
        elif prep == "ideal":
            if dealer is None:
                raise ValueError("ideal mode needs the shared IdealDealer")
            self.prep_mode = "ideal"
            self.prep = IdealPrep(role, dealer)
        elif hasattr(prep, "and_triples") and hasattr(prep, "input_masks"):
            self.prep_mode = "ideal" if getattr(prep, "ideal", False) else "real"
            self.prep = prep
        else:
            raise ValueError("prep must be 'real', 'ideal' or a preprocessing object")
        self.allow_refill = allow_refill
        self.engine = Engine(role, chan, self.prep, self.sid, cheat=cheat, allow_refill=allow_refill)
        self.state = ACTIVE
        self.is_setup = False
        self.delta_share = None if delta_share is None else int(delta_share)
        self.delta_wires = None
        self.setup_stats: dict = {}
        self.last_gen: dict = {}      # filled by couplet.protocol after every generation
        self._ids = {kind: 0 for kind in _ID_KINDS}

    # -- state ------------------------------------------------------------------ #

    def require_active(self) -> None:
        """Raise ``ProtocolAbort`` unless the session is ``ACTIVE``."""
        if self.state != ACTIVE:
            raise ProtocolAbort("session aborted")
        if self.engine.dead or getattr(self.prep, "aborted", False):
            self.abort()
            raise ProtocolAbort("session aborted")

    def abort(self) -> None:
        """Move to ``ABORTED``: retire the key shares, the session wires and the preprocessing."""
        self.state = ABORTED
        self.delta_share = None
        self.delta_wires = None
        self.engine.abort()

    @contextlib.contextmanager
    def guard(self):
        """Context manager for protocol steps: a ``ProtocolAbort`` inside aborts the session."""
        self.require_active()
        try:
            yield self
        except ProtocolAbort:
            self.abort()
            raise
        except BaseException:
            if self.engine.dead or getattr(self.prep, "aborted", False):
                self.abort()
            raise

    def next_id(self, kind: str) -> int:
        """Next identifier of ``kind`` in ('kid', 'gid', 'qid'); monotone, never reused."""
        if kind not in self._ids:
            raise ValueError(f"kind must be one of {_ID_KINDS}")
        self.require_active()
        value = self._ids[kind]
        self._ids[kind] = value + 1
        return value

    # -- protocol ----------------------------------------------------------------- #

    @property
    def triple_requests(self) -> int:
        """Preprocessing requests for AND triples so far (bucketing populations)."""
        return self.engine.triple_requests

    def setup(self, reserve_triples: int = 0, reserve_outmasks: int = 0) -> None:
        """Preprocessing setup and the persistent wires of Delta (once per session).

        The triples of the setup circuit, ``reserve_triples`` further ones and
        those of ``reserve_outmasks`` on-demand output masks are drawn in ONE
        request (one bucketing population).
        """
        reserve, masks = int(reserve_triples), int(reserve_outmasks)
        if reserve < 0 or reserve != reserve_triples or isinstance(reserve_triples, bool):
            raise ValueError("reserve_triples must be a non-negative integer")
        if masks < 0 or masks != reserve_outmasks or isinstance(reserve_outmasks, bool):
            raise ValueError("reserve_outmasks must be a non-negative integer")
        self.require_active()
        if self.is_setup:
            raise ValueError("setup() was already run")
        if masks:
            from .protocol import outmask_and_count     # the session layer sits below Pi_Gen
            reserve += masks * outmask_and_count(self.k, self.s)
        with self.guard():
            self.prep.setup()
            if self.delta_share is None:
                self.delta_share = (self._rand.below(1 << self.s) if self._has_rng
                                    else secrets.randbits(self.s))
            circuit = cg.build_session_setup(self.s, adder=self.setup_adder)
            self.engine.phase_tag = "setup-"
            try:
                self.engine.reserve_triples(circuit.stats()["n_and"] + reserve)
                result = self.engine.eval_circuit(
                    circuit, {cg.DELTA: ints_to_bits([self.delta_share], self.s)}, None, 1)
            finally:
                self.engine.phase_tag = ""
            self.delta_wires = result.retained[cg.DELTA]
            self.setup_stats = result.stats
            self.is_setup = True

    def reserve_triples(self, n: int) -> None:
        """Draw ``n`` further AND triples in one preprocessing request (one more population).

        Explicit in both modes: ``triple_requests`` grows by one if ``n > 0``.
        """
        with self.guard():
            self.engine.reserve_triples(n)

    def evaluate(self, circuit, my_inputs=None, batch: int = 1, bound=None) -> EvalResult:
        """Evaluate ``circuit`` on the session engine (see ``couplet.abb.Engine.eval_circuit``).

        A BOUND input ``delta`` is bound to the session wires of Delta unless
        ``bound`` already contains it.
        """
        self.require_active()
        if not self.is_setup:
            raise ValueError("setup() has not been run")
        if not isinstance(circuit, Circuit):
            raise ValueError("expected a couplet.circuit.Circuit")
        bound = dict(bound or {})
        if circuit.has_input(BOUND, cg.DELTA) and cg.DELTA not in bound:
            if len(circuit.input_wires(BOUND, cg.DELTA)) != self.s + 1:
                raise ValueError("the circuit was built for another s")
            bound[cg.DELTA] = self.delta_wires
        with self.guard():
            return self.engine.eval_circuit(circuit, my_inputs, bound, batch)

    # -- reporting ------------------------------------------------------------------ #

    def info(self) -> dict:
        """Public parameters and counters of the session (no secrets)."""
        prep = self.prep
        requests = self.engine.triple_requests
        return {
            "role": self.role, "k": self.k, "s": self.s, "sigma": self.sigma,
            "prep": self.prep_mode, "state": self.state,
            "evaluations": self.engine.evaluations,
            "triples_used": self.engine.triples_used,
            "triples_unused": self.engine.triples_available(),
            "triple_requests": requests,
            "triples_requested": self.engine.triples_requested,
            "allow_refill": self.allow_refill,
            # log2 of the bucketing term of the session: requests * 2^-sigma (real preprocessing)
            "bucketing_bound_log2": (math.log2(requests) - self.sigma
                                     if requests and self.prep_mode == "real" else None),
            "last_bucket_size": getattr(prep, "last_bucket_size", None),
            "prep_counters": dict(getattr(prep, "counters", {})),
        }
