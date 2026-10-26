"""The real preprocessing pipeline of one party (DESIGN.md sections 7.1 - 7.5).

``RealPrep(role, chan, sid, sigma=40, rng=None)`` owns this party's Boolean
MAC key share and produces, over the channel endpoint ``chan``:

    setup()                 base OT in both directions and the two COT
                            extension states (two rounds);
    random_bits(n)          n random ``<x>``                          -> Shares
    input_masks(owner, n)   n input masks of P_owner                  -> (Shares, bits | None)
    and_triples(n)          n authenticated AND triples, bucketed and
                            checked (run size and bucket size from
                            params.triple_request(n, sigma))               -> (a, b, c)
    delta                   this party's share Delta^B_role (int);
                            lsb = 1 for P_0 and 0 for P_1.

Both parties must call the same methods with the same arguments in the same
order.  Work is done in large numpy batches (one bucketing population per
``and_triples`` call; OT extension in chunks of ``batch_rows`` rows).

Statistical error: every ``and_triples`` call is one bucketing population and
contributes 2^-sigma (Lemma 4); request all triples of a session in one call
if the session is to stay at 2^-sigma overall.

Minimum size of a request: the triple check of a request must cover at least
``params.MIN_LEAKY_TRIPLES`` = 306 leaky triples (the bound against a key share
that ignores the lsb convention, Lemma 3).  ``and_triples(n)`` therefore runs
the protocol for ``(n_run, B) = params.triple_request(n, sigma)``: a request
whose ``n * bucket_size(n)`` would be smaller is padded to ``n_run > n``
triples, and the ``n_run - n`` surplus triples are discarded after the check
(``last_run_triples`` is ``n_run``; ``counters["leaky_triples"]`` counts the
``n_run * B`` leaky triples that were generated).  Both parties pad in the
same way, so the transcript is that of a request for ``n_run`` triples.

Accounting: traffic and time are booked under the channel phases ``"baseot"``,
``"cotext"``, ``"abit"``, ``"laand"``, ``"bucket"`` and ``"check"``; every
method restores the phase that was active when it was called.

Abort (change G of DESIGN.md): if a check fails or the peer misbehaves, the
method raises ``ProtocolAbort``, the object is marked ``aborted``, the key
share and all extension state are dropped, and every later call raises
``ProtocolAbort`` again.

``rng`` (optional, an object with ``randbytes``) replaces the OS CSPRNG for
this party's key share, base-OT scalars, extension choice bits and check
seeds; commitment randomness and coin-toss shares always come from the OS.
``cheat`` passes test hooks to the sub-protocols; the default is honest.
Keys: ``"flight1"`` / ``"flight2"`` (``baseot``, used by ``setup``),
``"flip_u"`` (``cotext``, used by ``random_bits`` and ``random_bdoz``),
``"g_error"`` / ``"flip_d"`` / ``"w_error"`` (``aand``, used by ``and_triples``).
"""

from __future__ import annotations

import functools

from .. import gf128 as gf
from .. import hashes as hs
from ..hashes import ProtocolAbort
from . import aand
from .abits import DEFAULT_BATCH_ROWS, AbitSource, sample_delta
from .baseot import base_ot
from .cotext import CotReceiver, CotSender
from .params import PAD, SIGMA_DEFAULT, triple_request
from .shares import Shares, as_rand


def _session_method(method):
    """Run a protocol method: refuse after an abort, abort on any failure, restore the phase."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.aborted:
            raise ProtocolAbort("session aborted")
        phase = self.chan.phase
        try:
            return method(self, *args, **kwargs)
        except _BadArgument as exc:  # raised before any message was sent
            raise ValueError(str(exc)) from None
        except BaseException:
            self.abort()
            raise
        finally:
            self.chan.mark(phase)

    return wrapper


class _BadArgument(Exception):
    """Invalid argument detected before the protocol started (the session survives)."""


def _count(n) -> int:
    if isinstance(n, bool) or int(n) != n or int(n) < 0:
        raise _BadArgument("the number of items must be a non-negative integer")
    return int(n)


class RealPrep:
    """Real (dealer-free) preprocessing of one party; see the module docstring."""

    ideal = False

    def __init__(self, role: int, chan, sid: bytes, sigma: int = SIGMA_DEFAULT, rng=None, *,
                 batch_rows: int = DEFAULT_BATCH_ROWS, cheat=None) -> None:
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        if chan.party != role:
            raise ValueError("role does not match the channel endpoint")
        if sigma < 1:
            raise ValueError("sigma must be positive")
        self.role = role
        self.chan = chan
        self.sid = bytes(sid)
        self.sigma = int(sigma)
        self.batch_rows = int(batch_rows)
        self.aborted = False
        self._rand = as_rand(rng)
        self._cheat = cheat
        self.delta = sample_delta(role, self._rand)
        # Identifier of the preprocessing sub-session: labels of its coin tosses and
        # BoolChecks cannot collide with those of the engine running under ``sid``.
        self._sid_prep = hs.sha256(b"Couplet-prep-sid", self.sid)
        self._abits: AbitSource | None = None
        self._batch = 0          # and_triples batches so far
        self._triple_index = 0   # leaky triples so far (tweak of H)
        self.last_bucket_size: int | None = None
        self.last_run_triples: int | None = None   # triples of the latest protocol run (>= request)
        self.counters = {"random_bits": 0, "input_masks": 0, "and_triples": 0,
                         "leaky_triples": 0}

    # -- life cycle ------------------------------------------------------------ #

    def abort(self) -> None:
        """Retire the session: drop the key share and all extension state."""
        self.aborted = True
        self.delta = None
        self._abits = None

    @_session_method
    def setup(self) -> None:
        """Base OT (both directions) and initialisation of the two COT extensions."""
        if self._abits is not None:
            raise _BadArgument("setup() was already run")
        self.chan.mark("baseot")
        chosen, (seeds0, seeds1) = base_ot(self.chan, self.sid, self.role,
                                           gf.unpack_bits(self.delta), rng=self._rand,
                                           cheat=self._cheat)
        self.chan.mark("cotext")
        sender = CotSender(self.chan, self.sid, self.role, self.delta, chosen, self._rand)
        receiver = CotReceiver(self.chan, self.sid, 1 - self.role, seeds0, seeds1, self._rand)
        self._abits = AbitSource(self.chan, self.role, self.delta, sender, receiver,
                                 self.batch_rows)

    def _source(self) -> AbitSource:
        if self._abits is None:
            raise _BadArgument("setup() has not been run")
        return self._abits

    # -- correlations ------------------------------------------------------------ #

    @_session_method
    def random_bits(self, n: int) -> Shares:
        """n random authenticated bits ``<x>`` (three rounds per extension chunk)."""
        n = _count(n)
        source = self._source()
        if n == 0:
            return Shares.zeros(0)
        out = source.random_shares(n, cheat=self._cheat)
        self.counters["random_bits"] += n
        return out

    @_session_method
    def random_bdoz(self, n: int):
        """n random bits in BDOZ form, ``(Bdoz, Keys)`` (the ``AbitSource`` interface)."""
        n = _count(n)
        return self._source().random_bdoz(n, self._cheat)

    @_session_method
    def input_masks(self, owner: int, n: int):
        """n input masks of P_owner: ``(Shares, bits)`` at the owner, ``(Shares, None)`` else."""
        if owner not in (0, 1):
            raise _BadArgument("owner must be 0 or 1")
        n = _count(n)
        source = self._source()
        if n == 0:
            empty = Shares.zeros(0)
            return (empty, empty.bits.copy()) if owner == self.role else (empty, None)
        out = source.input_masks(owner, n)
        self.counters["input_masks"] += n
        return out

    @_session_method
    def and_triples(self, n: int):
        """n AND triples ``(a, b, c)``: bucketed (Lemma 4) and checked before they are returned.

        The run has at least ``MIN_LEAKY_TRIPLES`` leaky triples: a smaller
        request is padded and the surplus triples are discarded (module docstring).
        """
        n = _count(n)
        source = self._source()
        if n == 0:
            return Shares.zeros(0), Shares.zeros(0), Shares.zeros(0)
        n_run, bucket = triple_request(n, self.sigma)
        batch, index0 = self._batch, self._triple_index
        self._batch += 1
        self._triple_index += n_run * bucket
        a, b, c = aand.and_triples(self.chan, self._sid_prep, self.role, self.delta, source,
                                   n_run, bucket, batch=batch, index0=index0, cheat=self._cheat)
        self.last_bucket_size = bucket
        self.last_run_triples = n_run
        self.counters["and_triples"] += n
        self.counters["leaky_triples"] += n_run * bucket
        if n_run != n:                      # padded request: the surplus triples are never used
            a, b, c = (Shares(t.bits[:n].copy(), t.macs[:n].copy()) for t in (a, b, c))
        return a, b, c

    # -- cost model ---------------------------------------------------------------- #

    def cot_rows_for_triples(self, n: int) -> int:
        """COT rows per direction consumed by ``and_triples(n)`` (excluding extension padding)."""
        n_run, bucket = triple_request(n, self.sigma)
        return 2 * n_run * bucket + PAD
