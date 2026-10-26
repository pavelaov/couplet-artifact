"""Malicious-party strategies against Pi_Gen and Pi_AuthEval, and the experiments that run them
(DESIGN.md sections 2.2, 2.3, 5.3, 5.4, 7.2, 7.4; appendix "adversarial regression").

Every experiment runs the two parties as two threads on an in-process channel.
One party follows the protocol with the unmodified code (``couplet.autheval.AuthEval``);
the other one is an adversary built from three tools:

  * the test hooks of the lower layers (``Engine.cheat``, the ``cheat``
    dictionary of ``prep.aand``);
  * ``TamperChannel``: a wrapper around the adversary's OWN channel endpoint
    whose rules rewrite, replay, mirror or withhold its outgoing messages (a
    rushing adversary may read the honest message of an exchange first);
  * ``CheatingChecker``: a subclass of ``spdz2k.ArithChecker`` that sends wrong
    value shares, keeps a consistent view, and may add to its ``zeta`` the
    compensation ``(Delta_a + g) e`` for a guess ``g`` of the honest key share
    (the best strategy of Lemma 2).

Deviations (one ``trial_<id>`` function each; a trial returns a record with the
honest party's verdict):

  A1   flip the own share of a public correction word in the output opening
  A2   flip the own share of the masked private output of the peer (its root seed)
  A3   flip one ``d`` / ``e`` opening of an AND layer
  A4   add an error to ``t`` leaky-AND transfer messages, with a guess of the
       honest shares (Lemma 3: passes with probability ``2^-t``)
  A5   use an inconsistent ``u`` in one column of the OT-extension matrix
       (rejected iff the honest MAC-key bit of that column is 1; section 7.2)
  A6   send a wrong share of the masked gate input in wire mode
  A7   send a wrong value share at ``Open`` (with and without the MAC
       compensation by a guess)
  A8   make the honest party mask twice under one gate offset
  A9   replay old messages (value shares, check messages of an earlier check
       or of another session, an output opening of the Boolean engine)
  A10  substitute the own key before ``Query`` and open
  A11  use a wrong ``Delta`` share in the arithmetic check or in ``Lin``
  A12  abort after having received the outputs
  A13  mirror the honest party's commitment and opening in a check
  A14  add an error that is ``0 mod 2^(m-s)`` to the own value share of an
       opening over ``Z_{2^m}`` (``2^(m-j)``, e.g. ``2^(m-1)``): the payload
       ``v mod 2^(m-s)`` is unchanged in every trial, so no wrong value can be
       accepted; the check passes with probability ``(j + 2) / 2^(j+1)``
       (``3/4`` for ``2^(m-1)``) and a pass or an abort may tell the adversary
       low bits of the honest MAC-key share (the MAC key only, never an input;
       outside Lemma 2, which is about errors that change the payload)
  A15  shared-input ``Mask``: choose the own share after having seen the honest
       share (rushing), or mirror the honest commitment and opening; caught by
       the commit-then-open exchange

All honest parties run the driver in its default strict mode on the exact
triple reservation of the scenario (``protocol.and_budget``): one bucketing
population per session.  Every record carries the honest party's
``triple_requests``; ``summarise`` requires that it never exceeds one.

Statistics.  A row whose deviation passes with a known probability ``p`` is
accepted iff the exact two-sided binomial test of the number of passes
against ``Binomial(trials, p)`` has a p-value of at least ``SIGNIFICANCE``
(10^-4); a row with an upper bound (Lemma 2) iff the exact upper tail
``Pr[X >= passes]`` at the bound is at least ``SIGNIFICANCE``.

``trial_distinguish`` is the selective-failure experiment: the honest party's
abort indicator for one adversary in one of two honest-input worlds (a key
index ``alpha`` with most significant bit 0 or 1; a gate input ``x = 0`` or
``x = N - 1``).  ``trial_control_leaky`` is the positive control: the same
experiment against the re-implemented one-bit-leaky protocol
(``baselines.leaky_zhang``), where the abort DOES depend on the index bit.

Preprocessing.  Deviations that do not target the preprocessing run with the
ideal dealer (``prep = "ideal"``).  A4 runs the real leaky-AND, bucketing and
triple-check protocol on authenticated bits from the dealer
(``HybridPrep``, labelled ``hybrid``); A5 runs the real OT extension on base-OT
seeds from a trusted party (``AttackRealPrep`` with ``IdealBaseOT``, labelled
``fast-real``: the F_OT-hybrid model of Theorem 5); both can also run on the
complete real pipeline (``real``, about two seconds per session for the base
OT).  In every one of these modes the honest party's shares of ``x`` of the
attacked leaky triples are recorded (``first_x_bits``), so that "the triple
check passes iff every guess is right" is asserted trial by trial.

``run_trials`` / ``summarise`` / ``run_distinguisher`` aggregate; ``run_all.py``
writes the results to ``experiments/raw/attacks.json``.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
import secrets
import tempfile
import threading
import traceback

import numpy as np

from .. import abb
from .. import fss
from .. import gf128 as gf
from .. import hashes as hs
from .. import protocol as pg
from ..gates import GateSpec
from ..hashes import ProtocolAbort
from ..net import loopback_pair
from ..prep import IdealDealer, IdealPrep, RealPrep, aand, triple_request
from ..prep.abits import AbitSource
from ..prep.cotext import CotReceiver, CotSender
from ..prep.params import KAPPA
from ..prep.shares import Shares
from ..spdz2k import AShare, ArithChecker, lemma2_bound
from ..state import ABORTED, USED, GateUsedError, StateError
from ..autheval import AuthEval

__all__ = [
    "ATTACKS", "DISTINGUISHERS", "AdversaryStop", "AttackRealPrep", "CheatingChecker", "Env",
    "HybridPrep", "IdealBaseOT", "Mirror", "Replay", "Rewrite", "Rush", "SIGNIFICANCE", "STANDARD",
    "Setting", "StopAfterReceive", "TINY", "TamperChannel", "binomial_two_sided_p",
    "binomial_upper_p", "plan_tasks", "run_distinguisher", "run_pair", "run_task", "run_tasks",
    "run_trials", "summarise", "summarise_distinguisher", "two_proportion_z",
]

SIGNIFICANCE = 1e-4      # level of the exact binomial tests of the pass rates


# --------------------------------------------------------------------------- #
# Channel wrapper of the adversary
# --------------------------------------------------------------------------- #

class AdversaryStop(Exception):
    """Raised in the adversary's thread when its strategy ends the run (it stops responding)."""


class _Rule:
    """A deviation at operation number ``seq`` of channel phase ``phase``.

    ``seq`` counts the adversary's exchanges and sends within the phase,
    starting when the rule is added to the channel (rules given to the
    constructor count from the start).  A rule fires once.
    """

    def __init__(self, phase: str, seq: int = 0) -> None:
        self.phase = phase
        self.seq = int(seq)
        self.base = 0
        self.fired = False

    def matches(self, phase: str, seq: int) -> bool:
        return not self.fired and phase == self.phase and seq - self.base == self.seq

    def exchange(self, chan: "TamperChannel", out: bytes):
        """Returns (message actually sent, message received)."""
        raise NotImplementedError


class Rewrite(_Rule):
    """Send ``fn(message)`` instead of the message."""

    def __init__(self, phase: str, seq: int, fn) -> None:
        super().__init__(phase, seq)
        self.fn = fn

    def outgoing(self, chan, out):
        return bytes(self.fn(out))

    def exchange(self, chan, out):
        sent = self.outgoing(chan, out)
        return sent, chan.inner.exchange(sent)


class Replay(_Rule):
    """Send a recorded message: ``source`` is bytes, or ``(phase, seq)`` of an earlier own message."""

    def __init__(self, phase: str, seq: int, source) -> None:
        super().__init__(phase, seq)
        self.source = source

    def outgoing(self, chan, out):
        source = self.source
        return bytes(source if isinstance(source, (bytes, bytearray)) else chan.sent[source])

    def exchange(self, chan, out):
        sent = self.outgoing(chan, out)
        return sent, chan.inner.exchange(sent)


class Mirror(_Rule):
    """Rushing: read the honest party's message of this exchange and send the same bytes back."""

    def exchange(self, chan, out):
        incoming = chan.inner.recv()
        chan.inner.send(incoming)
        return bytes(incoming), incoming


class Rush(_Rule):
    """Rushing: read the honest party's message of this exchange first, then send
    ``fn(own message, honest message)``."""

    def __init__(self, phase: str, seq: int, fn) -> None:
        super().__init__(phase, seq)
        self.fn = fn

    def exchange(self, chan, out):
        incoming = chan.inner.recv()
        sent = bytes(self.fn(out, bytes(incoming)))
        chan.inner.send(sent)
        return sent, incoming


class StopAfterReceive(_Rule):
    """Rushing abort: read the honest party's message of this exchange, send nothing, disappear."""

    def exchange(self, chan, out):
        chan.captured = chan.inner.recv()
        chan.inner.close()
        raise AdversaryStop(f"stopped after receiving in phase {self.phase!r}")


class TamperChannel:
    """The adversary's endpoint: delegates to ``inner`` and applies the rules to what it sends.

    ``sent[(phase, seq)]`` records every outgoing message, ``received`` likewise
    (``seq`` = number of the operation within its channel phase), so that
    replay rules and the experiments can refer to them.
    """

    def __init__(self, inner, rules=()) -> None:
        self.inner = inner
        self.rules = list(rules)
        self.sent: dict = {}
        self.received: dict = {}
        self.captured = None
        self._count: dict = {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def add(self, rule: _Rule) -> _Rule:
        """Arm ``rule`` now: its ``seq`` counts from the current position of its phase."""
        rule.base = self._count.get(rule.phase, 0)
        self.rules.append(rule)
        return rule

    def position(self, phase: str) -> int:
        return self._count.get(phase, 0)

    def _next(self):
        phase = self.inner.phase
        seq = self._count.get(phase, 0)
        self._count[phase] = seq + 1
        return phase, seq

    def exchange(self, data) -> bytes:
        phase, seq = self._next()
        out = bytes(data)
        for rule in self.rules:
            if rule.matches(phase, seq):
                rule.fired = True
                sent, incoming = rule.exchange(self, out)
                break
        else:
            sent, incoming = out, self.inner.exchange(out)
        self.sent[(phase, seq)] = sent
        self.received[(phase, seq)] = bytes(incoming)
        return incoming

    def send(self, data) -> None:
        phase, seq = self._next()
        out = bytes(data)
        for rule in self.rules:
            if rule.matches(phase, seq) and isinstance(rule, (Rewrite, Replay)):
                rule.fired = True
                out = rule.outgoing(self, out)
                break
        self.sent[(phase, seq)] = out
        self.inner.send(out)

    def recv(self) -> bytes:
        return self.inner.recv()


def flip_bit(bit: int):
    """Message transformer: flip bit ``bit`` (bit 0 = least significant bit of byte 0)."""
    def fn(msg: bytes) -> bytes:
        buf = bytearray(msg)
        buf[bit // 8] ^= 1 << (bit % 8)
        return bytes(buf)
    return fn


# --------------------------------------------------------------------------- #
# The deviating arithmetic checker
# --------------------------------------------------------------------------- #

class CheatingChecker(ArithChecker):
    """``ArithChecker`` of a malicious party with a consistent view of its own errors.

    ``value_errors``  opening index -> additive error on the own value share;
    ``replay``        opening index -> earlier opening index whose sent value
                      share is sent again;
    ``compensate``    add ``(Delta_a + g) e`` to the own ``zeta`` of every
                      check, ``e = sum chi_j delta_j`` and ``g`` a fresh uniform
                      guess of the honest ``Delta_h`` (Lemma 2);
    ``delta_offset``  compute ``zeta`` with ``Delta_a + delta_offset``.
    ``errors`` records the effective error of every opening, ``guesses`` the
    guesses used, ``history`` one entry per check and modulus with everything
    that decides the verdict of the honest party: the two zeta values sum to
    ``comp - offset * y - (Delta_h + Delta_a) * e`` (see ``predict_pass``).
    """

    def __init__(self, role, chan, sid, delta_share, s, *, value_errors=None, replay=None,
                 compensate: bool = False, delta_offset: int = 0) -> None:
        super().__init__(role, chan, sid, delta_share, s)
        self.value_errors = dict(value_errors or {})
        self.replay = dict(replay or {})
        self.compensate = bool(compensate)
        self.delta_offset = int(delta_offset)
        self.sent_values: dict = {}
        self.errors: dict = {}
        self.guesses: list = []
        self.history: list = []

    @classmethod
    def factory(cls, **kwargs):
        """A ``checker_factory`` for ``AuthEval`` that builds this checker with ``kwargs``."""
        return lambda role, chan, sid, delta_share, s: cls(role, chan, sid, delta_share, s, **kwargs)

    def _outgoing_values(self, shares, first_index):
        out = []
        for j, share in enumerate(shares):
            index = first_index + j
            mod = 1 << share.bits
            value = share.value
            if index in self.replay:
                value = self.sent_values[self.replay[index]] % mod
            value = (value + self.value_errors.get(index, 0)) % mod
            self.sent_values[index] = value
            self.errors[index] = (value - share.value) % mod
            out.append(value)
        return out

    def _outgoing_zeta(self, bits, zeta, chis, items, counter):
        mod = 1 << bits
        y = sum(chi * value for chi, (_, value, _) in zip(chis, items)) % mod
        e = sum(chi * self.errors.get(index, 0) for chi, (index, _, _) in zip(chis, items)) % mod
        comp = 0
        if self.compensate:
            guess = secrets.randbelow(1 << self.s)
            self.guesses.append(guess)
            comp = (self._delta + guess) * e % mod
        self.history.append({"counter": counter, "bits": bits, "y": y, "e": e, "comp": comp,
                             "offset": self.delta_offset, "delta_a": self._delta})
        return (zeta - self.delta_offset * y + comp) % mod

    def predict_pass(self, delta_honest: int) -> bool:
        """The verdict of the honest party over all checks run so far, from the algebra alone.

        Its zeta and the one sent here sum to ``comp - offset y - (Delta_h + Delta_a) e``
        modulo ``2^bits``; a check passes iff this is zero for each of its moduli.
        """
        return all((h["comp"] - h["offset"] * h["y"] - (delta_honest + h["delta_a"]) * h["e"])
                   % (1 << h["bits"]) == 0 for h in self.history)


# --------------------------------------------------------------------------- #
# Preprocessing variants for the attacks on the preprocessing
# --------------------------------------------------------------------------- #

class _RecordingSource:
    """``IdealPrep`` that remembers the bits of its first ``random_bdoz`` answer (tests only)."""

    def __init__(self, prep: IdealPrep) -> None:
        self._prep = prep
        self.first_bits = None

    def random_bdoz(self, n):
        own, keys = self._prep.random_bdoz(n)
        if self.first_bits is None:
            self.first_bits = own.bits.copy()
        return own, keys


class HybridPrep:
    """F_aBit-hybrid preprocessing: authenticated bits and input masks come from the
    in-process ``IdealDealer``; AND triples are produced by the REAL leaky-AND,
    bucketing and triple-check protocol of ``prep.aand`` (sections 7.4, 7.5).

    Not the complete pipeline: results obtained with it are labelled ``hybrid``.
    ``attack`` is a ``cheat`` dictionary of ``prep.aand`` (or a callable
    ``(n, bucket) -> dictionary``, called with the size of the protocol run,
    ``params.triple_request``); it is applied to the FIRST ``and_triples``
    call only.  ``first_x_bits`` are this party's shares of ``x`` of the leaky
    triples of that call (for the experiment's bookkeeping).
    """

    ideal = False
    label = "hybrid"

    def __init__(self, role: int, chan, sid: bytes, dealer: IdealDealer, sigma: int = 40,
                 attack=None) -> None:
        self.role = role
        self.chan = chan
        self.sigma = int(sigma)
        self._ideal = IdealPrep(role, dealer)
        self._source = _RecordingSource(self._ideal)
        self.delta = self._ideal.delta
        self.aborted = False
        self._attack = attack
        self._sid = hs.sha256(b"Couplet-prep-sid", bytes(sid))
        self._batch = 0
        self._index = 0
        self.last_bucket_size = None
        self.first_total = None
        self.counters = {"and_triples": 0, "leaky_triples": 0}

    @property
    def first_x_bits(self):
        bits = self._source.first_bits
        return None if bits is None else bits[:self.first_total]

    def _alive(self) -> None:
        if self.aborted:
            raise ProtocolAbort("session aborted")

    def setup(self) -> None:
        self._alive()

    def abort(self) -> None:
        self.aborted = True
        self.delta = None
        self._ideal.abort()

    def input_masks(self, owner: int, n: int):
        self._alive()
        return self._ideal.input_masks(owner, n)

    def random_bits(self, n: int):
        self._alive()
        return self._ideal.random_bits(n)

    def and_triples(self, n: int):
        self._alive()
        n = int(n)
        if n == 0:
            return Shares.zeros(0), Shares.zeros(0), Shares.zeros(0)
        # the run of the real pipeline for this request (``RealPrep.and_triples``): a small
        # request is padded to at least ``MIN_LEAKY_TRIPLES`` leaky triples
        n_run, bucket = triple_request(n, self.sigma)
        batch, index0 = self._batch, self._index
        self._batch += 1
        self._index += n_run * bucket
        if self.first_total is None:
            self.first_total = n_run * bucket
        attack, self._attack = self._attack, None
        if callable(attack):
            attack = attack(n_run, bucket)
        phase = self.chan.phase
        try:
            out = aand.and_triples(self.chan, self._sid, self.role, self.delta, self._source,
                                   n_run, bucket, batch=batch, index0=index0, cheat=attack)
        except BaseException:
            self.abort()
            raise
        finally:
            self.chan.mark(phase)
        self.last_bucket_size = bucket
        self.counters["and_triples"] += n
        self.counters["leaky_triples"] += n_run * bucket
        return out if n_run == n else tuple(t[:n] for t in out)


class IdealBaseOT:
    """F_OT for the base OTs of one session: a trusted in-process party samples the seed pairs.

    Not a protocol (labelled ``fast-real``): it replaces the Masny-Rindal base OT
    (about two seconds per session in Python) so that many sessions of the real
    OT extension can be run.
    """

    def __init__(self) -> None:
        # direction d: P_{1-d} holds both seeds of every instance, P_d the chosen ones
        self._pairs = {d: (gf.rand_vec(KAPPA), gf.rand_vec(KAPPA)) for d in (0, 1)}

    def get(self, role: int, delta_bits):
        seeds0, seeds1 = self._pairs[role]
        choose = np.asarray(delta_bits, dtype=bool)[:, None]
        return np.where(choose, seeds1, seeds0), self._pairs[1 - role]


class _RecordingAbits:
    """The ``AbitSource`` of a ``RealPrep`` that remembers the bits of the first
    ``random_bdoz`` answer requested through it (the BDOZ bits of the first
    ``and_triples`` call; tests only).  Everything else is delegated."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.first_bits = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def random_bdoz(self, n, cheat=None):
        own, keys = self._inner.random_bdoz(n, cheat)
        if self.first_bits is None:
            self.first_bits = own.bits.copy()
        return own, keys


class AttackRealPrep(RealPrep):
    """``RealPrep`` whose leaky-AND deviation is applied to the first ``and_triples`` call only.

    With ``ot`` (an ``IdealBaseOT``) the base OT is replaced by the trusted seed
    source; the OT extension, the authenticated bits and the AND triples are
    the real protocols.  The class reads ``RealPrep``'s private attributes
    ``_abits``, ``_rand`` and ``_cheat``.  ``first_x_bits`` are this party's
    shares of ``x`` of the leaky triples of the first ``and_triples`` call (the
    first ``n * B`` of the BDOZ bits that call draws), for the bookkeeping of
    the experiment: with them "the check passes iff every guess is right" is
    asserted on the complete real pipeline as well.
    """

    label = "real"

    def __init__(self, role: int, chan, sid: bytes, sigma: int = 40, *, ot=None, attack=None) -> None:
        super().__init__(role, chan, sid, sigma)
        self._ot = ot
        self._attack = attack
        self._recorder = None
        self.first_total = None
        self.initial_delta = self.delta
        if ot is not None:
            self.label = "fast-real"

    @property
    def first_x_bits(self):
        bits = None if self._recorder is None else self._recorder.first_bits
        return None if bits is None or self.first_total is None else bits[:self.first_total]

    def _record_first(self) -> None:
        """Route the BDOZ bits of the next ``and_triples`` call through the recorder (once)."""
        if self._recorder is None and self._abits is not None:
            self._recorder = _RecordingAbits(self._abits)
            self._abits = self._recorder

    def setup(self) -> None:
        if self._ot is None:
            return super().setup()
        if self.aborted:
            raise ProtocolAbort("session aborted")
        if self._abits is not None:
            raise ValueError("setup() was already run")
        chosen, (seeds0, seeds1) = self._ot.get(self.role, gf.unpack_bits(self.delta))
        self.chan.mark("cotext")
        sender = CotSender(self.chan, self.sid, self.role, self.delta, chosen, self._rand)
        receiver = CotReceiver(self.chan, self.sid, 1 - self.role, seeds0, seeds1, self._rand)
        self._abits = AbitSource(self.chan, self.role, self.delta, sender, receiver, self.batch_rows)

    def and_triples(self, n: int):
        attack, self._attack = self._attack, None
        n_run, bucket = triple_request(int(n), self.sigma) if int(n) > 0 else (0, 0)   # the run
        if callable(attack):
            attack = attack(n_run, bucket)
        if self.first_total is None and int(n) > 0:
            self.first_total = n_run * bucket
            self._record_first()
        self._cheat = attack
        try:
            return super().and_triples(n)
        finally:
            self._cheat = None
            if self._recorder is not None and self._abits is self._recorder:
                self._abits = self._recorder._inner      # later requests are not recorded


# --------------------------------------------------------------------------- #
# Running two parties
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class Setting:
    """Public parameters of an experiment."""

    k: int = 32
    s: int = 64
    n: int = 4
    prep: str = "ideal"            # "ideal", "hybrid", "fast-real" or "real"
    sigma: int = 40
    setup_adder: str = "sklansky"

    def tag(self) -> str:
        return f"n={self.n} k={self.k} s={self.s} prep={self.prep}"


STANDARD = Setting()
TINY = Setting(k=8, s=4, n=4)


class Env:
    """What the two parties of one run share: the session identifier and the trusted helpers."""

    def __init__(self, setting: Setting, sid: bytes | None = None) -> None:
        self.setting = setting
        self.sid = b"attack-" + secrets.token_bytes(8) if sid is None else bytes(sid)
        self.dealer = IdealDealer() if setting.prep in ("ideal", "hybrid") else None
        self.ot = IdealBaseOT() if setting.prep == "fast-real" else None

    def driver(self, role: int, chan, *, prep_attack=None, checker_factory=None,
               state_path=None) -> AuthEval:
        st = self.setting
        dealer = None
        if st.prep == "ideal":
            prep, dealer = "ideal", self.dealer
        elif st.prep == "hybrid":
            prep = HybridPrep(role, chan, self.sid, self.dealer, st.sigma, prep_attack)
        elif st.prep in ("fast-real", "real"):
            prep = AttackRealPrep(role, chan, self.sid, st.sigma, ot=self.ot, attack=prep_attack)
        else:
            raise ValueError(f"unknown preprocessing {st.prep!r}")
        return AuthEval(role, chan, self.sid, k=st.k, s=st.s, prep=prep, sigma=st.sigma,
                       dealer=dealer, setup_adder=st.setup_adder, checker_factory=checker_factory,
                       state_path=state_path)


def run_pair(fn0, fn1, timeout: float = 120.0) -> tuple:
    """Run ``fn0(chan_0)`` and ``fn1(chan_1)`` in two threads; returns their results.

    A party function that raises yields ``{"error": traceback}``.  Each endpoint
    is closed when its function ends, which unblocks the other party.
    """
    chans = loopback_pair(timeout=timeout)
    out = [None, None]

    def work(i, fn):
        try:
            out[i] = fn(chans[i])
        except BaseException as exc:  # reported to the experiment
            out[i] = {"error": "".join(traceback.format_exception(exc))[-3000:]}
        finally:
            chans[i].close()

    threads = [threading.Thread(target=work, args=(i, fn), daemon=True) for i, fn in enumerate((fn0, fn1))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return out[0], out[1]


def _probe_dead(vf: AuthEval, res: dict) -> None:
    """After an abort: the session must be dead and later commands must send nothing."""
    chan = vf.chan
    before = (chan.msgs_sent, chan.bytes_sent)
    commands = (
        lambda: vf.setup(),
        lambda: vf.key(None, "dpf", 2, 0),
        lambda: vf.gate(None, GateSpec.zero_test(2, vf.k, vf.s)),
        lambda: vf.query(0, 0),
        lambda: vf.mask_shared(0, 0),
        lambda: vf.open(AShare(0, 0, vf.w, True)),
    )
    refused = 0
    for cmd in commands:
        try:
            cmd()
        except ProtocolAbort:
            refused += 1
        except Exception:  # noqa: BLE001 - anything else is a failure of the experiment
            pass
    ses = vf.session
    res["dead"] = bool(vf.state == ABORTED and ses.delta_share is None and ses.delta_wires is None
                       and ses.engine.dead and (vf.checker is None or vf.checker.dead)
                       and vf.registry.state == ABORTED)
    res["silent"] = bool(refused == len(commands) and (chan.msgs_sent, chan.bytes_sent) == before)


def _honest(env: Env, role: int, body, *, state_path=None, reserve: int = 0, outmasks: int = 0):
    """Party function of the honest party: ``setup`` then ``body(vf, res)``.

    The driver is strict (its default): ``reserve`` AND triples and the triples
    of ``outmasks`` output masks are drawn at ``setup``, in ONE request.
    """
    def fn(chan):
        res = {"role": role, "aborted": False, "output": None, "message": "", "dead": None,
               "silent": None}
        vf = env.driver(role, chan, state_path=state_path)
        res["vf"] = vf
        res["bool_delta"] = vf.session.prep.delta
        try:
            vf.setup(reserve, outmasks)
            res["delta"] = vf.delta_share
            res["output"] = body(vf, res)
        except ProtocolAbort as exc:
            res["aborted"] = True
            res["message"] = str(exc)
            _probe_dead(vf, res)
        res["phases"] = chan.stats()["phases"]
        return res
    return fn


def _adversary(env: Env, role: int, body, *, rules=(), prep_attack=None, checker_factory=None,
               reserve: int = 0, outmasks: int = 0):
    """Party function of the adversary: ``setup`` then ``body(vf, res, chan)`` on a ``TamperChannel``."""
    def fn(chan):
        tc = TamperChannel(chan, rules)
        res = {"role": role, "aborted": False, "stopped": False, "output": None, "message": "",
               "chan": tc}
        vf = env.driver(role, tc, prep_attack=prep_attack, checker_factory=checker_factory)
        res["vf"] = vf
        try:
            vf.setup(reserve, outmasks)
            res["delta"] = vf.delta_share
            res["output"] = body(vf, res, tc)
        except ProtocolAbort as exc:
            res["aborted"] = True
            res["message"] = str(exc)
        except AdversaryStop as exc:
            res["stopped"] = True
            res["message"] = str(exc)
        return res
    return fn


def _record(honest: dict, adversary: dict, **extra) -> dict:
    """The picklable record of one trial."""
    rec = {"aborted": None, "accepted": None, "correct": None, "dead": None, "silent": None,
           "message": "", "error": None}
    for res in (honest, adversary):
        if "error" in res:
            rec["error"] = res["error"]
    if rec["error"] is None:
        rec["aborted"] = bool(honest["aborted"])
        rec["accepted"] = not honest["aborted"]
        rec["dead"] = honest["dead"]
        rec["silent"] = honest["silent"]
        rec["message"] = honest["message"]
        if honest.get("vf") is not None:      # bucketing populations drawn by the honest party
            rec["triple_requests"] = int(honest["vf"].triple_requests)
    rec.update(extra)
    return rec


def _sent_in(honest: dict, phase: str) -> int:
    """Messages the honest party sent in a channel phase."""
    return int(honest.get("phases", {}).get(phase, {}).get("msgs_sent", 0))


def _inputs_sent(honest: dict) -> int:
    """Input messages of the honest party: session-setup circuit and generation circuit."""
    return _sent_in(honest, "setup-input") + _sent_in(honest, "input")


def _predict(rec: dict) -> None:
    """Add ``predicted_pass``: the verdict that the algebra of the check predicts for this trial."""
    if rec.get("error") is not None:
        return
    checker = rec["_adv"]["vf"].checker
    delta_h = rec["_honest"].get("delta")
    if isinstance(checker, CheatingChecker) and delta_h is not None:
        rec["predicted_pass"] = bool(checker.predict_pass(delta_h))


def _key_pair_ok(k0, k1, alpha: int, beta: int, delta: int) -> bool:
    """Full-domain correctness of a DPF key pair on both planes."""
    mod = 1 << (k0.k + k0.s)
    y0, t0 = fss.dpf_eval_all(0, k0)
    y1, t1 = fss.dpf_eval_all(1, k1)
    for x in range(1 << k0.n):
        want = beta if x == alpha else 0
        if (int(y0[x]) + int(y1[x])) % mod != want:
            return False
        if (int(t0[x]) + int(t1[x])) % mod != delta * want % mod:
            return False
    return True


def _pair(honest_fn, adversary_fn, adv_role: int) -> tuple:
    """Run the honest party and the adversary in the right roles; returns (honest, adversary)."""
    if adv_role == 1:
        return run_pair(honest_fn, adversary_fn)
    a, h = run_pair(adversary_fn, honest_fn)
    return h, a


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def _gen_scenario(setting: Setting, adv_role: int, alpha: int, *, engine_cheat=None, rules=(),
                  late_rules=(), prep_attack=None, second_key: bool = False, **extra) -> dict:
    """Pi_Gen for one DPF key with public payload 1; the honest party inputs ``alpha``.

    ``engine_cheat`` is installed after the session setup, ``late_rules`` are
    armed at the same moment, ``rules`` apply from the first message.
    With ``second_key`` a second key is generated afterwards (same parameters).
    """
    env = Env(setting)
    n = setting.n
    budget = dict(reserve=pg.planned_key("dpf", n, setting.k, setting.s,
                                         batch=2 if second_key else 1)["triples"])

    def honest_body(vf, res):
        kid = vf.key(0, "dpf", n, alpha)
        res["key"] = vf.registry.key(kid).key
        if second_key:
            res["key2"] = vf.registry.key(vf.key(1, "dpf", n, alpha)).key
        return True

    def adv_body(vf, res, tc):
        if engine_cheat is not None:
            vf.session.engine.cheat = dict(engine_cheat)
        for rule in late_rules:
            tc.add(rule)
        kid = vf.key(0, "dpf", n, 0)
        res["key"] = vf.registry.key(kid).key
        if second_key:
            res["key2"] = vf.registry.key(vf.key(1, "dpf", n, 0)).key
        return True

    honest, adv = _pair(_honest(env, 1 - adv_role, honest_body, **budget),
                        _adversary(env, adv_role, adv_body, rules=rules, prep_attack=prep_attack,
                                   **budget), adv_role)
    rec = _record(honest, adv, **extra)
    if rec["error"] is None:
        rec["output_opening_sent"] = _sent_in(honest, "output")
        rec["certified"] = honest["vf"].registry.summary()["keys"]["CERTIFIED"]
        if not honest["aborted"] and adv.get("key") is not None:
            keys = (honest["key"], adv["key"]) if adv_role == 1 else (adv["key"], honest["key"])
            rec["correct"] = _key_pair_ok(keys[0], keys[1], alpha, 1, honest["delta"] + adv["delta"])
    rec["_honest"], rec["_adv"] = honest, adv
    return rec


def _strip(rec: dict) -> dict:
    """Drop the in-process objects of a record."""
    return {key: value for key, value in rec.items() if not key.startswith("_")}


def _gate_specs(setting: Setting) -> tuple:
    """The two gates of the gate pipeline: identity table, zero test with an authenticated input."""
    n, k, s = setting.n, setting.k, setting.s
    return (GateSpec.table(n, k, s, list(range(1 << n))),
            GateSpec.zero_test(n, k, s, auth_input=True))


def _gate_budget(setting: Setting, tables: int = 1, zero_tests: int = 1) -> dict:
    """Triple reservation of the gate pipeline (public; the same number at both parties)."""
    spec1, spec2 = _gate_specs(setting)
    calls = ([pg.planned_gate(spec1, tables)] if tables else []) + (
        [pg.planned_gate(spec2, zero_tests)] if zero_tests else [])
    return dict(reserve=pg.and_budget(calls))


def _gate_body(setting: Setting, x_share: int):
    """The gate pipeline of one party: shared input -> identity table -> wire -> zero test -> open."""
    spec1, spec2 = _gate_specs(setting)

    def body(vf, res, *_):
        vf.gate(0, spec1)
        vf.gate(1, spec2)
        res["rp1"] = vf.registry.gate(1).material.rp_share      # own share of the wire offset r'
        res["x_hat"] = vf.mask_shared(0, x_share)
        wire = vf.apply(0)
        res["x_hat2"] = vf.mask_wire(1, wire)
        out = vf.apply(1)
        res["out_share"] = out
        return vf.open(out)
    return body


def _gate_scenario(setting: Setting, adv_role: int, x: int, *, checker_factory=None, rules=(),
                   late=None, **extra) -> dict:
    """Shared input ``x`` (the honest party's share) -> table gate -> wire -> zero test -> open.

    Openings of the arithmetic layer: number 0 is the masked wire (mode
    ``wire``), number 1 the final release.  ``late(tc)`` arms rules after setup.
    """
    env = Env(setting)
    honest_body = _gate_body(setting, x)
    inner = _gate_body(setting, 0)

    def adv_body(vf, res, tc):
        if late is not None:
            late(tc)
        return inner(vf, res)

    budget = _gate_budget(setting)
    honest, adv = _pair(_honest(env, 1 - adv_role, honest_body, **budget),
                        _adversary(env, adv_role, adv_body, rules=rules,
                                   checker_factory=checker_factory, **budget), adv_role)
    rec = _record(honest, adv, expected=int(x == 0), **extra)
    if rec["error"] is None and not honest["aborted"]:
        rec["output"] = honest["output"]
        rec["correct"] = honest["output"] == rec["expected"]
    if rec["error"] is None:
        rec["arith_open_sent"] = _sent_in(honest, "arith-open")
    rec["_honest"], rec["_adv"] = honest, adv
    return rec


def _query_scenario(setting: Setting, adv_role: int, alpha: int, beta: int, *,
                    checker_factory=None, adv_edit=None, separate: bool = False, rules=(),
                    late=None, adv_lin=None, const: int = 0, **extra) -> dict:
    """Key with a secret payload (the honest party inputs alpha and beta), Query, Open.

    ``separate=False``: all ``2^n`` points are opened in one ``open_many``;
    ``separate=True``: the point ``alpha`` and then one other point are opened
    one after the other.  ``adv_edit(vf, kid)`` lets the adversary change its
    key record before ``Query``; with ``const`` both parties open
    ``[[f(alpha)]] + const`` and ``adv_lin`` replaces the adversary's ``Lin``.
    """
    env = Env(setting)
    n, k = setting.n, setting.k
    size = 1 << n
    other = (alpha + 1) % size

    def body(vf, res, a_share, b_share, is_adv=False, tc=None):
        kid = vf.key(0, "dpf", n, a_share, b_share)
        if is_adv and adv_edit is not None:
            adv_edit(vf, kid)
        if is_adv and late is not None:
            late(tc)
        if const:
            share = vf.query(kid, alpha)
            if is_adv and adv_lin is not None:
                share = adv_lin(vf, share, const)
            else:
                share = vf.lin([(1, share)], const)
            return [vf.open(share)]
        if separate:
            first = vf.open(vf.query(kid, alpha))
            res["first"] = first
            return [first, vf.open(vf.query(kid, other))]
        return vf.open_many(vf.query_all(kid))

    # the key has a secret payload; with ``const`` the opened value is not exact: one output mask
    budget = dict(reserve=pg.planned_key("dpf", n, k, setting.s, "xor", "secret")["triples"],
                  outmasks=1 if const else 0)
    honest, adv = _pair(
        _honest(env, 1 - adv_role, lambda vf, res: body(vf, res, alpha, beta), **budget),
        _adversary(env, adv_role, lambda vf, res, tc: body(vf, res, 0, 0, True, tc), rules=rules,
                   checker_factory=checker_factory, **budget), adv_role)
    if const:
        expected = [(beta + const) % (1 << k)]
    elif separate:
        expected = [beta, 0]
    else:
        expected = [beta if x == alpha else 0 for x in range(size)]
    rec = _record(honest, adv, **extra)
    if rec["error"] is None and not honest["aborted"]:
        rec["correct"] = honest["output"] == expected
    if rec["error"] is None and separate:
        rec["first_released"] = honest.get("first") == beta
    rec["_honest"], rec["_adv"] = honest, adv
    return rec


# --------------------------------------------------------------------------- #
# Trials A1 - A15
# --------------------------------------------------------------------------- #

def _gen_circuit(setting: Setting):
    """The generation circuit of the gen scenario and its evaluation plan."""
    from .. import circuits_gen as cg
    desc = cg.dpf_descriptor(setting.n, setting.k, setting.s, "xor", "public_one")
    circuit = pg.generation_circuit(desc)
    return circuit, abb.get_plan(circuit)


def trial_A1(rng: random.Random, setting: Setting = STANDARD, alpha=None) -> dict:
    """Flip the own share of one bit of a public correction word in the output opening."""
    n, w = setting.n, setting.k + setting.s
    label, width = rng.choice([("cws", 128 * n), ("cwtl", n), ("cwtr", n), ("cwout_v", w),
                               ("cwout_t", w)])
    alpha = rng.randrange(1 << n) if alpha is None else alpha
    rec = _gen_scenario(setting, rng.randrange(2), alpha,
                        engine_cheat={"flip_public": (label, rng.randrange(width), 0)},
                        target=label)
    return _strip(rec)


def trial_A2(rng: random.Random, setting: Setting = STANDARD, alpha=None) -> dict:
    """Flip the own share of one bit of the masked private output of the peer (its root seed)."""
    adv = rng.randrange(2)
    alpha = rng.randrange(1 << setting.n) if alpha is None else alpha
    rec = _gen_scenario(setting, adv, alpha,
                        engine_cheat={"flip_private": (f"root{1 - adv}", rng.randrange(128), 0)},
                        target=f"root{1 - adv}")
    return _strip(rec)


def trial_A3(rng: random.Random, setting: Setting = STANDARD, alpha=None) -> dict:
    """Flip the own share of one opened ``d`` / ``e`` bit of one AND layer of the generation circuit."""
    _, plan = _gen_circuit(setting)
    layers = [v - u for is_and, u, v, *_ in plan.steps if is_and]
    layer = rng.randrange(len(layers))
    index = rng.randrange(2 * layers[layer])
    alpha = rng.randrange(1 << setting.n) if alpha is None else alpha
    rec = _gen_scenario(setting, rng.randrange(2), alpha,
                        engine_cheat={"flip_and": (layer, index)}, target=(layer, index))
    return _strip(rec)


def _leaky_attack(rng: random.Random, t: int, box: dict):
    """A callable ``(n, bucket) -> cheat`` attacking ``t`` leaky triples with random guesses."""
    def make(n: int, bucket: int) -> dict:
        total = n * bucket
        idx = np.array(sorted(rng.sample(range(total), t)))
        err = np.array([[rng.getrandbits(64), rng.getrandbits(64) | (1 << 63)] for _ in range(t)],
                       dtype=np.uint64)                              # E_i != 0
        guess = np.array([rng.getrandbits(1) for _ in range(t)], dtype=np.uint8)
        box.update(idx=idx, guess=guess, total=total, bucket=bucket)
        return {"g_error": (idx, err, guess)}
    return make


def trial_A4(rng: random.Random, setting: Setting = None, t: int = 1, alpha=None) -> dict:
    """Additive error on ``t`` leaky-AND transfer messages with a guess of the honest shares.

    The ``and_triples`` request of the session is attacked (the one population
    that holds the triples of the session-setup circuit and of the key
    generation); the triple check passes iff every guess is right.  The honest
    party's shares of ``x`` are read from its preprocessing object
    (``first_x_bits``) in every mode, the complete real pipeline included.
    """
    setting = dataclasses.replace(STANDARD, prep="hybrid") if setting is None else setting
    box: dict = {}
    alpha = rng.randrange(1 << setting.n) if alpha is None else alpha
    rec = _gen_scenario(setting, rng.randrange(2), alpha, prep_attack=_leaky_attack(rng, t, box), t=t)
    if rec["error"] is None:
        prep = rec["_honest"]["vf"].session.prep
        bits = getattr(prep, "first_x_bits", None)
        if bits is not None and box:
            rec["guess_right"] = bool(np.array_equal(bits[box["idx"]], box["guess"]))
        rec["bucket"] = box.get("bucket")
        rec["inputs_sent"] = _inputs_sent(rec["_honest"])
    return _strip(rec)


def trial_A5(rng: random.Random, setting: Setting = None, alpha=None) -> dict:
    """Inconsistent ``u`` in one column of the first OT-extension matrix (one flipped bit of U_j)."""
    setting = dataclasses.replace(STANDARD, prep="fast-real") if setting is None else setting
    column = rng.randrange(KAPPA)
    row = rng.randrange(64)

    def flip(msg: bytes) -> bytes:
        n_words = len(msg) // (8 * KAPPA)
        return flip_bit(8 * (column * n_words * 8) + row)(msg)

    # The U message of the first extension call is the adversary's first message in the
    # channel phase "cotext" (the base OT books its traffic under "baseot").
    alpha = rng.randrange(1 << setting.n) if alpha is None else alpha
    rec = _gen_scenario(setting, rng.randrange(2), alpha, rules=[Rewrite("cotext", 0, flip)],
                        column=column)
    if rec["error"] is None:
        rec["key_bit"] = int(rec["_honest"]["bool_delta"] >> column & 1)
        rec["inputs_sent"] = _inputs_sent(rec["_honest"])
    return _strip(rec)


def trial_A6(rng: random.Random, setting: Setting = STANDARD, x=None, compensate: bool = False,
             high_only: bool = False) -> dict:
    """Wrong share of the masked gate input in wire mode (opening number 0)."""
    n, s = setting.n, setting.s
    if high_only:                      # an error on the high s bits only: x_hat is unchanged
        delta = rng.randrange(1, 1 << s) << n
    else:
        delta = rng.randrange(1, 1 << n) + (rng.getrandbits(s) << n)
    x = rng.choice([0, (1 << n) - 1]) if x is None else x
    rec = _gate_scenario(setting, rng.randrange(2), x,
                         checker_factory=CheatingChecker.factory(value_errors={0: delta},
                                                                 compensate=compensate),
                         effective=not high_only)
    if rec["error"] is None:
        rec["released"] = rec["arith_open_sent"] > 1     # did the honest party send its output share?
    _predict(rec)
    return _strip(rec)


def trial_A7(rng: random.Random, setting: Setting = STANDARD, x=None, compensate: bool = False,
             high_only: bool = False) -> dict:
    """Wrong value share in the final opening (opening number 1), optionally with the MAC guess."""
    k, s = setting.k, setting.s
    if high_only:
        delta = rng.randrange(1, 1 << s) << k
    else:
        delta = rng.randrange(1, 1 << k) + (rng.getrandbits(s) << k)
    x = rng.choice([0, (1 << setting.n) - 1]) if x is None else x
    rec = _gate_scenario(setting, rng.randrange(2), x,
                         checker_factory=CheatingChecker.factory(value_errors={1: delta},
                                                                 compensate=compensate),
                         effective=not high_only)
    _predict(rec)
    return _strip(rec)


def trial_A8(rng: random.Random, setting: Setting = STANDARD, variant: str = "second-mask") -> dict:
    """Attempts to make the honest party mask twice under one offset; counts what it sends.

    ``second-mask``: a second ``Mask`` on a used gate in both modes;
    ``abort-retry``: the peer disappears during ``Mask`` and the caller retries;
    ``restart``: the process restarts on its persisted counters.
    """
    n, k, s = setting.n, setting.k, setting.s
    size = 1 << n
    x = rng.randrange(size)
    adv = rng.randrange(2)
    env = Env(setting)
    spec1 = GateSpec.table(n, k, s, list(range(size)))
    spec2 = GateSpec.zero_test(n, k, s, auth_input=True)
    box = {"attempts": 0, "refused": 0, "leaked_msgs": 0}

    def attempt(vf, fn, kinds):
        before = (vf.chan.msgs_sent, vf.chan.bytes_sent)
        box["attempts"] += 1
        try:
            fn()
        except kinds:
            box["refused"] += 1
        except Exception:  # noqa: BLE001 - a wrong exception type is not a refusal
            pass
        box["leaked_msgs"] += vf.chan.msgs_sent - before[0]

    if variant == "second-mask":
        def honest_body(vf, res):
            vf.gate(0, spec1)
            vf.gate(1, spec2)
            vf.mask_shared(0, x)
            attempt(vf, lambda: vf.mask_shared(0, (x + 1) % size), GateUsedError)
            wire = vf.apply(0)
            attempt(vf, lambda: vf.mask_wire(0, wire), (GateUsedError, ValueError))
            vf.mask_wire(1, wire)
            attempt(vf, lambda: vf.mask_wire(1, wire), GateUsedError)
            attempt(vf, lambda: vf.mask_shared(1, 0), GateUsedError)
            res["states"] = [vf.registry.gate(g).state for g in (0, 1)]
            return vf.open(vf.apply(1))

        budget = _gate_budget(setting)
        honest, peer = _pair(_honest(env, 1 - adv, honest_body, **budget),
                             _adversary(env, adv, lambda vf, res, tc: _gate_body(setting, 0)(vf, res),
                                        **budget), adv)
        rec = _record(honest, peer, variant=variant)
        if rec["error"] is None:
            rec["correct"] = honest["output"] == int(x == 0)
            rec["states_used"] = honest.get("states") == [USED, USED]
    elif variant == "abort-retry":
        def honest_body(vf, res):
            vf.gate(0, spec1)
            try:
                vf.mask_shared(0, x)
            except ProtocolAbort:
                res["first_failed"] = True
                record = vf.registry._gates[0]                    # the registry is aborted: read directly
                res["state_after"] = record.state
                attempt(vf, lambda: vf.mask_shared(0, (x + 1) % size), (ProtocolAbort, GateUsedError))
                attempt(vf, lambda: vf.mask_shared(0, x), (ProtocolAbort, GateUsedError))
                raise
            return None

        def adv_body(vf, res, tc):
            vf.gate(0, spec1)
            # the second message of the commit-then-open exchange: the adversary reads the honest
            # opening (the masked share) and disappears
            tc.add(StopAfterReceive("mask", 1))
            vf.mask_shared(0, 0)

        budget = _gate_budget(setting, 1, 0)
        honest, peer = _pair(_honest(env, 1 - adv, honest_body, **budget),
                             _adversary(env, adv, adv_body, **budget), adv)
        rec = _record(honest, peer, variant=variant)
        if rec["error"] is None:
            rec["states_used"] = honest.get("state_after") == USED
            rec["mask_msgs"] = _sent_in(honest, "mask")           # one commitment and one opening left
    elif variant == "restart":
        tmp = tempfile.mkdtemp(prefix="couplet-state-")
        paths = [os.path.join(tmp, f"party{b}.json") for b in (0, 1)]
        sid = env.sid

        def first(role):
            def body(vf, res, *_):
                vf.gate(0, spec1)
                res["x_hat"] = vf.mask_shared(0, x if role != adv else 0)
                return vf.registry.counters()
            return body

        budget = _gate_budget(setting, 1, 0)

        def run_first(role):
            def fn(chan):
                vf = env.driver(role, chan, state_path=paths[role])
                vf.setup(budget["reserve"])
                return first(role)(vf, {})
            return fn

        c0, c1 = run_pair(run_first(0), run_first(1))
        env2 = Env(setting, sid=sid)                              # the restarted processes

        def second(role):
            def fn(chan):
                vf = env2.driver(role, chan, state_path=paths[role])
                vf.setup(budget["reserve"])
                out = {"counters": vf.registry.counters()}
                if role != adv:
                    attempt(vf, lambda: vf.mask_shared(0, (x + 1) % size), StateError)
                    attempt(vf, lambda: vf.gate(0, spec1), StateError)
                gid = vf.gate(None, spec1)
                out["new_gid"] = gid
                out["x_hat"] = vf.mask_shared(gid, x if role != adv else 0)
                share = vf.apply(gid)
                out["value"] = vf.open(share)
                out["triple_requests"] = vf.triple_requests
                return out
            return fn

        r0, r1 = run_pair(second(0), second(1))
        honest_first, honest_second = (c0, r0) if adv == 1 else (c1, r1)
        err = next((r["error"] for r in (c0, c1, r0, r1) if "error" in r), None)
        rec = {"aborted": False, "accepted": True, "correct": None, "dead": None, "silent": None,
               "message": "", "error": err, "variant": variant}
        if err is None:
            rec["correct"] = honest_second["value"] == x
            rec["triple_requests"] = honest_second["triple_requests"]
            rec["states_used"] = (honest_second["new_gid"] >= honest_first["reserved"]["gid"] > 0
                                  and honest_second["counters"]["starts"] == 2)
        for path in paths:
            for name in (path, path + ".tmp"):
                if os.path.exists(name):
                    os.remove(name)
        os.rmdir(tmp)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    rec.update(box)
    return rec


def trial_A9(rng: random.Random, setting: Setting = STANDARD, variant: str = "value-share") -> dict:
    """Replay of old messages.

    ``value-share``: the value share of the first release is sent again in the second;
    ``check``: the four messages of an earlier ArithCheck are sent again in a later one;
    ``session``: the adversary's messages of the arithmetic layer of another
    session (same commands, other ``sid``) are sent again;
    ``bool-output``: the output opening of an earlier circuit evaluation is sent
    again in the next one.
    """
    n, k = setting.n, setting.k
    adv = rng.randrange(2)
    alpha = rng.randrange(1 << n)
    beta = rng.randrange(1, 1 << k)
    if variant == "value-share":
        rec = _query_scenario(setting, adv, alpha, beta, separate=True,
                              checker_factory=CheatingChecker.factory(replay={1: 0}),
                              variant=variant)
        _predict(rec)
    elif variant == "check":
        # Two releases: checks 0 and 1 (the first check of a release is empty here).
        def late(tc):
            for i in range(4):
                tc.add(Replay("arith-check", 4 + i, ("arith-check", i)))
        rec = _query_scenario(setting, adv, alpha, beta, separate=True, late=late, variant=variant)
    elif variant == "session":
        x = rng.choice([0, (1 << n) - 1])
        first = _gate_scenario(setting, adv, x)
        if first["error"] is not None or first["aborted"]:
            return _strip(dict(first, variant=variant, error=first["error"] or "recording run aborted"))
        old = first["_adv"]["chan"].sent
        rules = [Replay(phase, seq, data) for (phase, seq), data in old.items()
                 if phase in ("arith-open", "arith-check")]
        rec = _gate_scenario(setting, adv, x, rules=rules, variant=variant)
    elif variant == "bool-output":
        rec = _gen_scenario(setting, adv, alpha, second_key=True,
                            late_rules=[Replay("output", 1, ("output", 0))], variant=variant)
        if rec["error"] is None:
            rec["first_released"] = rec["certified"] == 1
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return _strip(rec)


def trial_A10(rng: random.Random, setting: Setting = STANDARD, variant: str = "cw_out") -> dict:
    """Substitute the own key before ``Query``; all points are queried and opened."""
    n, k, s = setting.n, setting.k, setting.s
    adv = rng.randrange(2)
    alpha = rng.randrange(1 << n)
    beta = rng.randrange(1, 1 << k)

    def edit(vf, kid):
        record = vf.registry.key(kid)
        key = record.key
        if variant == "cw_out":
            new = dataclasses.replace(key, cw_out_v=key.cw_out_v ^ (1 << rng.randrange(k)))
        elif variant == "root":
            seed = bytearray(key.seed)
            seed[rng.randrange(16)] ^= 1 << rng.randrange(8)
            new = dataclasses.replace(key, seed=bytes(seed))
        elif variant == "cw_seed":
            level = rng.randrange(n)
            cw = bytearray(key.cw_s[level])
            cw[rng.randrange(16)] ^= 1 << rng.randrange(8)
            new = dataclasses.replace(key, cw_s=key.cw_s[:level] + (bytes(cw),) + key.cw_s[level + 1:])
        elif variant == "fresh":            # a key generated by the adversary itself for another function
            fake = fss.dpf_gen(rng.randrange(1 << n), rng.randrange(1, 1 << k), rng.getrandbits(s + 1),
                               n, k, s)
            new = fake[vf.role]
        else:
            raise ValueError(f"unknown variant {variant!r}")
        y_old, _ = fss.dpf_eval_all(vf.role, key)
        y_new, _ = fss.dpf_eval_all(vf.role, new)
        record.info["changed"] = any((int(a) - int(b)) % (1 << k) for a, b in zip(y_old, y_new))
        record.info["changed_any"] = any(int(a) != int(b) for a, b in zip(y_old, y_new))
        record.key = new

    rec = _query_scenario(setting, adv, alpha, beta, adv_edit=edit, variant=variant)
    if rec["error"] is None:
        info = rec["_adv"]["vf"].registry._keys[0].info
        rec["effective"] = bool(info.get("changed"))
    return _strip(rec)


def trial_A11(rng: random.Random, setting: Setting = STANDARD, variant: str = "check") -> dict:
    """A wrong ``Delta`` share: in the ArithCheck (``check``) or when adding a public constant (``lin``)."""
    n, k, s = setting.n, setting.k, setting.s
    adv = rng.randrange(2)
    alpha = rng.randrange(1 << n)
    beta = rng.randrange(1, 1 << k) | 1
    offset = rng.randrange(1, 1 << s) | 1
    if variant == "check":
        rec = _query_scenario(setting, adv, alpha, beta,
                              checker_factory=CheatingChecker.factory(delta_offset=offset),
                              variant=variant)
        _predict(rec)
    elif variant == "lin":
        const = rng.randrange(1, 1 << k) | 1

        def adv_lin(vf, share, c):
            return share.add_public(c, vf.role, vf.delta_share + offset).with_exact(False)

        rec = _query_scenario(setting, adv, alpha, beta, const=const, adv_lin=adv_lin, variant=variant)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return _strip(rec)


def trial_A12(rng: random.Random, setting: Setting = STANDARD, variant: str = "open-value") -> dict:
    """The adversary stops responding after it has received the output.

    ``open-value``: after the honest value share of the final release;
    ``open-check``: after the honest opening of the last ArithCheck;
    ``gen-output``: after the honest shares of the output opening of Pi_Gen;
    ``gen-check``: after the honest opening of the second BoolCheck of Pi_Gen.
    """
    n = setting.n
    adv = rng.randrange(2)
    if variant in ("open-value", "open-check"):
        x = rng.choice([0, (1 << n) - 1])
        # arith-open: 0 = masked wire, 1 = release; arith-check: 0..3 deferred check, 4..7 final check
        rule = StopAfterReceive("arith-open", 1) if variant == "open-value" else StopAfterReceive(
            "arith-check", 7)
        rec = _gate_scenario(setting, adv, x, late=lambda tc: tc.add(rule), variant=variant)
        if rec["error"] is None and variant == "open-value":
            a = rec["_adv"]
            mine = a.get("out_share")
            got = a["chan"].captured
            if mine is not None and got is not None:
                value = (mine.value + int.from_bytes(got, "little")) % (1 << setting.k)
                rec["adversary_learned_output"] = value == rec["expected"]
    elif variant in ("gen-output", "gen-check"):
        alpha = rng.randrange(1 << n)
        # outcheck: 0, 1 first BoolCheck; 2, 3 second BoolCheck (the setup circuit uses other phases)
        rule = StopAfterReceive("output", 0) if variant == "gen-output" else StopAfterReceive(
            "outcheck", 3)
        rec = _gen_scenario(setting, adv, alpha, late_rules=[rule], variant=variant)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    if rec["error"] is None:
        rec["adversary_stopped"] = bool(rec["_adv"].get("stopped"))
    return _strip(rec)


def trial_A13(rng: random.Random, setting: Setting = STANDARD, variant: str = "boolcheck") -> dict:
    """Mirror attack: after a wrong opening, send the honest party's own commitment and opening back."""
    n = setting.n
    adv = rng.randrange(2)
    if variant == "boolcheck":
        _, plan = _gen_circuit(setting)
        layers = [v - u for is_and, u, v, *_ in plan.steps if is_and]
        layer = rng.randrange(len(layers))
        rec = _gen_scenario(setting, adv, rng.randrange(1 << n),
                            engine_cheat={"flip_and": (layer, rng.randrange(2 * layers[layer]))},
                            late_rules=[Mirror("outcheck", 0), Mirror("outcheck", 1)], variant=variant)
    elif variant == "arithcheck":
        delta = rng.randrange(1, 1 << setting.k)
        x = rng.choice([0, (1 << n) - 1])
        # the commitment and the opening of zeta in the final check
        late = lambda tc: (tc.add(Mirror("arith-check", 6)), tc.add(Mirror("arith-check", 7)))  # noqa: E731
        rec = _gate_scenario(setting, adv, x, late=late,
                             checker_factory=CheatingChecker.factory(value_errors={1: delta}),
                             variant=variant)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return _strip(rec)


def high_error_pass_rate(j: int) -> float:
    """Pass probability of an additive error ``2^(m-j)`` on a value share, ``1 <= j <= s``.

    The two zeta values sum to ``-(Delta_0 + Delta_1) chi 2^(m-j)``, which is 0
    modulo ``2^m`` iff ``Delta chi = 0 mod 2^j``; for uniform ``Delta mod 2^j``
    and ``chi mod 2^j`` this has probability ``(j + 2) / 2^(j+1)``: 3/4, 1/2,
    5/16, ...  Adding the compensation ``(Delta_a + g) e`` for a guess ``g``
    replaces ``Delta`` by ``g - Delta_h`` and gives the same number.
    """
    return (j + 2) / float(1 << (j + 1))


def trial_A14(rng: random.Random, setting: Setting = STANDARD, x=None, j: int = 1,
              where: str = "open", compensate: bool = False) -> dict:
    """An error that is ``0 mod 2^(m-s)``: ``2^(m-j)`` on the own value share of one opening.

    ``where="open"``: the final release (modulus ``2^w``, ``w = k + s``);
    ``where="wire"``: the masked gate input in wire mode (modulus ``2^(n+s)``).
    The payload of the opening, its low ``m - s`` bits, is unchanged in every
    trial (``payload_unchanged`` is measured on the honest party's transcript,
    also when it aborts), so a passed check accepts the correct value.  What
    the adversary can learn is ``Delta mod 2^j`` of the MAC key, at the abort
    risk ``1 - (j + 2) / 2^(j+1)``; ``key_dependent`` marks the trials in which
    the verdict depended on the key at all (``chi 2^(m-j) != 0 mod 2^m``).
    """
    n, k, s = setting.n, setting.k, setting.s
    if not 1 <= j <= s:
        raise ValueError("need 1 <= j <= s (the error must be 0 modulo 2^(m-s))")
    if where == "open":
        index, bits = 1, k + s
    elif where == "wire":
        index, bits = 0, n + s
    else:
        raise ValueError(f"unknown opening {where!r}")
    delta = 1 << (bits - j)
    x = rng.choice([0, (1 << n) - 1]) if x is None else x
    rec = _gate_scenario(setting, rng.randrange(2), x,
                         checker_factory=CheatingChecker.factory(value_errors={index: delta},
                                                                 compensate=compensate),
                         where=where, j=j)
    if rec["error"] is None:
        honest, adv = rec["_honest"], rec["_adv"]
        if where == "open":
            # the public opened representative of the release, as the honest party saw it
            opened = honest["vf"].checker.last_opened
            rec["payload_unchanged"] = bool(len(opened) == 1 and opened[0] % (1 << k) == rec["expected"])
        else:
            rp = (honest["rp1"] + adv["rp1"]) % (1 << (n + s))
            rec["payload_unchanged"] = honest.get("x_hat2") == (x + rp) % (1 << n)
            rec["released"] = rec["arith_open_sent"] > 1
        history = adv["vf"].checker.history
        rec["key_dependent"] = any(h["e"] % (1 << h["bits"]) for h in history)
    _predict(rec)
    return _strip(rec)


def trial_A15(rng: random.Random, setting: Setting = STANDARD, x=None, variant: str = "rush") -> dict:
    """Shared-input ``Mask`` against a rushing party (the commit-then-open exchange).

    ``rush``: the adversary commits to its share, waits for the honest opening
    and then opens ANOTHER share, chosen so that ``x_hat`` becomes a value of
    its choice; ``mirror``: it sends the honest commitment and the honest
    opening back.  Without the commitment the first strategy would fix
    ``x_hat``; with it the opening does not verify.
    """
    n = setting.n
    size = 1 << n
    nb = (n + 7) // 8
    x = rng.choice([0, size - 1]) if x is None else x
    box: dict = {}

    def choose(own: bytes, incoming: bytes) -> bytes:
        peer = int.from_bytes(incoming[hs.OPENING_BYTES:], "little")
        mine = int.from_bytes(own[hs.OPENING_BYTES:], "little")
        target = rng.choice([v for v in range(size) if v != (peer + mine) % size])
        box.update(target=target, honest_x_hat=(peer + mine) % size)
        return own[:hs.OPENING_BYTES] + ((target - peer) % size).to_bytes(nb, "little")

    if variant == "rush":
        late = lambda tc: tc.add(Rush("mask", 1, choose))                           # noqa: E731
    elif variant == "mirror":
        late = lambda tc: (tc.add(Mirror("mask", 0)), tc.add(Mirror("mask", 1)))    # noqa: E731
    else:
        raise ValueError(f"unknown variant {variant!r}")
    rec = _gate_scenario(setting, rng.randrange(2), x, late=late, variant=variant)
    if rec["error"] is None:
        rec["x_hat_accepted"] = rec["_honest"].get("x_hat") is not None
        rec["mask_msgs"] = _sent_in(rec["_honest"], "mask")
        if variant == "rush":
            rec["target_differs"] = box.get("target") != box.get("honest_x_hat")
    return _strip(rec)


# --------------------------------------------------------------------------- #
# The distinguishing experiment (absence of selective failure)
# --------------------------------------------------------------------------- #

def _world_alpha(rng: random.Random, n: int, world: int) -> int:
    """A key index whose most significant bit is ``world``."""
    return (world << (n - 1)) | rng.randrange(1 << (n - 1))


def trial_distinguish(rng: random.Random, name: str, world: int) -> dict:
    """One run of adversary ``name`` in honest-input world ``world``; returns the abort indicator."""
    hybrid = dataclasses.replace(STANDARD, prep="hybrid")
    fast = dataclasses.replace(STANDARD, prep="fast-real")
    n = STANDARD.n
    x = 0 if world == 0 else (1 << n) - 1
    if name == "A1":
        rec = trial_A1(rng, STANDARD, _world_alpha(rng, n, world))
    elif name == "A2":
        rec = trial_A2(rng, STANDARD, _world_alpha(rng, n, world))
    elif name == "A3":
        rec = trial_A3(rng, STANDARD, _world_alpha(rng, n, world))
    elif name == "A4":
        rec = trial_A4(rng, hybrid, 1, _world_alpha(rng, n, world))
    elif name == "A5":
        rec = trial_A5(rng, fast, _world_alpha(rng, n, world))
    elif name == "A6":
        rec = trial_A6(rng, STANDARD, x)
    elif name == "A6-tiny":
        rec = trial_A6(rng, TINY, x, compensate=True)
    elif name == "A7":
        rec = trial_A7(rng, STANDARD, x)
    elif name == "A7-guess":
        rec = trial_A7(rng, STANDARD, x, compensate=True)
    elif name == "A7-tiny":
        rec = trial_A7(rng, TINY, x, compensate=True)
    elif name == "A14":
        rec = trial_A14(rng, STANDARD, x, j=1)
    elif name == "A14-j2":
        rec = trial_A14(rng, STANDARD, x, j=2)
    elif name == "A14-wire":
        rec = trial_A14(rng, STANDARD, x, j=1, where="wire")
    else:
        raise ValueError(f"unknown distinguisher {name!r}")
    out = {"world": world, "aborted": rec["aborted"], "error": rec["error"]}
    if "predicted_pass" in rec:
        out["predicted_pass"] = rec["predicted_pass"]
    return out


def trial_control_leaky(rng: random.Random, world: int, n: int = 6) -> dict:
    """Positive control: the selective-failure attack on the one-bit-leaky baseline protocol.

    The cheater adds an error to its share of the level-1 correction word and
    removes it on the children of the node it guesses to be on the path
    (guess: the most significant index bit is 0).  The honest party aborts
    iff the guess is wrong, i.e. iff ``msb(alpha) = 1``.
    """
    from ..baselines import leaky_zhang as lz

    dealer = IdealDealer()
    sid = b"control-" + secrets.token_bytes(8)
    alpha = _world_alpha(rng, n, world)
    beta = rng.getrandbits(128)
    adv = rng.randrange(2)
    cheat = {"cw_error": (1, rng.getrandbits(128) | 1, 0)}

    def party(role):
        def fn(chan):
            prep = IdealPrep(role, dealer)
            honest = role != adv
            try:
                lz.run_pi_dpf(chan, prep, sid, n, alpha if honest else 0, beta if honest else 0,
                              cheat=None if honest else cheat)
            except ProtocolAbort:
                return {"aborted": True}
            return {"aborted": False}
        return fn

    r0, r1 = run_pair(party(0), party(1))
    honest = r1 if adv == 0 else r0
    err = next((r["error"] for r in (r0, r1) if "error" in r), None)
    return {"world": world, "aborted": honest.get("aborted"), "error": err}


def _binom_log_pmf(i: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if i == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if i == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))


def binomial_two_sided_p(successes: int, trials: int, p: float) -> float:
    """p-value of the exact two-sided binomial test of ``successes`` against ``Binomial(trials, p)``.

    The sum of the probabilities of all outcomes that are at most as likely as
    the observed one (the usual definition; no normal approximation), evaluated
    in double precision.
    """
    successes, trials = int(successes), int(trials)
    if not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials")
    if trials == 0:
        return 1.0
    logs = [_binom_log_pmf(i, trials, p) for i in range(trials + 1)]
    ref = logs[successes] + 1e-9                 # ties count as "at most as likely"
    return min(1.0, math.fsum(math.exp(v) for v in logs if v <= ref))


def binomial_upper_p(successes: int, trials: int, p: float) -> float:
    """``Pr[X >= successes]`` for ``X ~ Binomial(trials, p)``: the exact one-sided test of a bound."""
    successes, trials = int(successes), int(trials)
    if not 0 <= successes <= trials:
        raise ValueError("need 0 <= successes <= trials")
    if successes == 0:
        return 1.0
    return min(1.0, math.fsum(math.exp(_binom_log_pmf(i, trials, p))
                              for i in range(successes, trials + 1)))


def two_proportion_z(a0: int, n0: int, a1: int, n1: int) -> float:
    """Two-proportion z-score of ``a1 / n1`` against ``a0 / n0`` (0 if the pooled variance is 0)."""
    if n0 == 0 or n1 == 0:
        return 0.0
    pooled = (a0 + a1) / (n0 + n1)
    var = pooled * (1 - pooled) * (1 / n0 + 1 / n1)
    return 0.0 if var <= 0 else (a1 / n1 - a0 / n0) / math.sqrt(var)


DISTINGUISHERS = {
    "A1": "flip a public correction word share (worlds: msb(alpha))",
    "A2": "flip the peer's masked root (worlds: msb(alpha))",
    "A3": "flip an AND opening (worlds: msb(alpha))",
    "A4": "leaky-AND error with one guess, hybrid preprocessing (worlds: msb(alpha))",
    "A5": "inconsistent OT-extension column, fast-real preprocessing (worlds: msb(alpha))",
    "A6": "wrong masked-input share in wire mode, s = 64 (worlds: x = 0, N-1)",
    "A6-tiny": "the same with the MAC guess at s = 4 (worlds: x = 0, N-1)",
    "A7": "wrong value share at Open, s = 64 (worlds: x = 0, N-1)",
    "A7-guess": "the same with the MAC guess, s = 64 (worlds: x = 0, N-1)",
    "A7-tiny": "the same with the MAC guess at s = 4 (worlds: x = 0, N-1)",
    "A14": "error 2^(m-1) (0 mod 2^(m-s)) on the value share at Open, s = 64: abort frequency 1/4 "
           "(worlds: x = 0, N-1)",
    "A14-j2": "error 2^(m-2) on the value share at Open, s = 64: abort frequency 1/2 "
              "(worlds: x = 0, N-1)",
    "A14-wire": "error 2^(m-1) on the masked-input share in wire mode, s = 64: abort frequency 1/4 "
                "(worlds: x = 0, N-1)",
    "control-leaky": "POSITIVE CONTROL: one-bit-leaky baseline, error on CW^(1) with a guess "
                     "(worlds: msb(alpha))",
}


def run_distinguisher(name: str, runs: int, seed: int) -> list:
    """``runs`` runs per world of one distinguisher; returns the list of records."""
    rng = random.Random(f"distinguish|{name}|{seed}")
    out = []
    for i in range(2 * runs):
        world = i & 1
        if name == "control-leaky":
            out.append(trial_control_leaky(rng, world))
        else:
            out.append(trial_distinguish(rng, name, world))
    return out


def summarise_distinguisher(name: str, records: list, threshold: float = 3.3) -> dict:
    runs = [sum(1 for r in records if r["world"] == b) for b in (0, 1)]
    aborts = [sum(1 for r in records if r["world"] == b and r["aborted"]) for b in (0, 1)]
    errors = [r["error"] for r in records if r["error"]]
    z = two_proportion_z(aborts[0], runs[0], aborts[1], runs[1])
    control = name.startswith("control")
    out = {
        "name": name, "description": DISTINGUISHERS[name], "runs_per_world": runs,
        "aborts": aborts,
        "abort_frequency": [aborts[b] / runs[b] if runs[b] else None for b in (0, 1)],
        "z": z, "threshold": threshold, "positive_control": control,
        "errors": len(errors), "first_error": errors[0] if errors else None,
        "ok": (not errors) and ((abs(z) >= threshold) if control else (abs(z) < threshold)),
    }
    predicted = [r for r in records if "predicted_pass" in r]
    if predicted:                 # every verdict is the one the algebra of the check predicts
        out["verdict_equals_prediction"] = all((not r["aborted"]) == r["predicted_pass"]
                                               for r in predicted)
        out["ok"] = out["ok"] and out["verdict_equals_prediction"]
    return out


# --------------------------------------------------------------------------- #
# Catalogue and aggregation
# --------------------------------------------------------------------------- #

_HYBRID = dataclasses.replace(STANDARD, prep="hybrid")
_FAST = dataclasses.replace(STANDARD, prep="fast-real")
_REAL = dataclasses.replace(STANDARD, prep="real")

# id -> (description, setting, trial function, keyword arguments, expectation, default trials)
# expectation: "always" (every trial must be detected), ("rate", p) (the deviation passes with
# probability p: exact two-sided binomial test at SIGNIFICANCE), ("bound", p) (pass probability
# at most p: exact one-sided test), "refused" (A8).
ATTACKS = {
    "A1": ("flip own share of a public correction word (Output)", STANDARD, trial_A1, {}, "always", 40),
    "A2": ("flip own share of the peer's masked root seed (Output)", STANDARD, trial_A2, {}, "always", 40),
    "A3": ("flip one d/e opening of an AND layer", STANDARD, trial_A3, {}, "always", 40),
    "A4-t1": ("leaky-AND transfer error, 1 guess", _HYBRID, trial_A4, {"t": 1}, ("rate", 0.5), 200),
    "A4-t2": ("leaky-AND transfer error, 2 guesses", _HYBRID, trial_A4, {"t": 2}, ("rate", 0.25), 200),
    "A4-t3": ("leaky-AND transfer error, 3 guesses", _HYBRID, trial_A4, {"t": 3}, ("rate", 0.125), 200),
    "A4-t8": ("leaky-AND transfer error, 8 guesses", _HYBRID, trial_A4, {"t": 8},
              ("rate", 2.0 ** -8), 4000),
    "A4-real": ("leaky-AND transfer error, 1 guess, complete real pipeline", _REAL, trial_A4,
                {"t": 1, "setting": _REAL}, ("rate", 0.5), 48),
    "A5": ("inconsistent column in the OT-extension matrix", _FAST, trial_A5, {}, ("rate", 0.5), 200),
    "A5-real": ("inconsistent column, complete real pipeline", _REAL, trial_A5, {"setting": _REAL},
                ("rate", 0.5), 48),
    "A6": ("wrong x_hat share in wire mode", STANDARD, trial_A6, {}, "always", 40),
    "A6-guess": ("wrong x_hat share in wire mode, MAC guess", STANDARD, trial_A6,
                 {"compensate": True}, "always", 40),
    "A6-tiny": ("wrong x_hat share in wire mode, MAC guess, s = 4", TINY, trial_A6,
                {"setting": TINY, "compensate": True}, ("bound", lemma2_bound(4)), 400),
    "A7": ("wrong value share at Open", STANDARD, trial_A7, {}, "always", 40),
    "A7-guess": ("wrong value share at Open, MAC guess", STANDARD, trial_A7, {"compensate": True},
                 "always", 40),
    "A7-tiny": ("wrong value share at Open, MAC guess, s = 4", TINY, trial_A7,
                {"setting": TINY, "compensate": True}, ("bound", lemma2_bound(4)), 400),
    "A7-high": ("error on the high s bits only at Open (value mod 2^k unchanged)", STANDARD,
                trial_A7, {"high_only": True}, "always", 20),
    "A8-second": ("second Mask on a used gate (both modes)", STANDARD, trial_A8,
                  {"variant": "second-mask"}, "refused", 20),
    "A8-retry": ("peer disappears during Mask, caller retries", STANDARD, trial_A8,
                 {"variant": "abort-retry"}, "refused", 20),
    "A8-restart": ("process restart on persisted counters", STANDARD, trial_A8,
                   {"variant": "restart"}, "refused", 10),
    "A9-value": ("replay of an old value share", STANDARD, trial_A9, {"variant": "value-share"},
                 "always", 20),
    "A9-check": ("replay of the messages of an earlier ArithCheck", STANDARD, trial_A9,
                 {"variant": "check"}, "always", 20),
    "A9-session": ("replay of the messages of another session", STANDARD, trial_A9,
                   {"variant": "session"}, "always", 20),
    "A9-bool": ("replay of an earlier Boolean output opening", STANDARD, trial_A9,
                {"variant": "bool-output"}, "always", 20),
    "A10-cwout": ("substituted key: flipped bit of CWout_v", STANDARD, trial_A10,
                  {"variant": "cw_out"}, "always", 20),
    "A10-root": ("substituted key: flipped bit of the own root", STANDARD, trial_A10,
                 {"variant": "root"}, "always", 20),
    "A10-cws": ("substituted key: flipped bit of a seed correction word", STANDARD, trial_A10,
                {"variant": "cw_seed"}, "always", 20),
    "A10-fresh": ("substituted key: self-generated key for another function", STANDARD, trial_A10,
                  {"variant": "fresh"}, "always", 20),
    "A11-check": ("wrong Delta share in the ArithCheck", STANDARD, trial_A11, {"variant": "check"},
                  "always", 20),
    "A11-lin": ("wrong Delta share when adding a public constant", STANDARD, trial_A11,
                {"variant": "lin"}, "always", 20),
    "A12-value": ("abort after receiving the opened value", STANDARD, trial_A12,
                  {"variant": "open-value"}, "always", 20),
    "A12-check": ("abort after receiving the last check opening", STANDARD, trial_A12,
                  {"variant": "open-check"}, "always", 20),
    "A12-gen": ("abort after receiving the output opening of Pi_Gen", STANDARD, trial_A12,
                {"variant": "gen-output"}, "always", 20),
    "A12-gencheck": ("abort after receiving the last BoolCheck opening of Pi_Gen", STANDARD,
                     trial_A12, {"variant": "gen-check"}, "always", 20),
    "A13-bool": ("mirror the honest commitment in BoolCheck", STANDARD, trial_A13,
                 {"variant": "boolcheck"}, "always", 20),
    "A13-arith": ("mirror the honest commitment in ArithCheck", STANDARD, trial_A13,
                  {"variant": "arithcheck"}, "always", 20),
    "A14-half": ("error 2^(m-1) on the value share at Open (0 mod 2^(m-s): payload unchanged)",
                 STANDARD, trial_A14, {"j": 1}, ("rate", high_error_pass_rate(1)), 400),
    "A14-j2": ("error 2^(m-2) on the value share at Open (payload unchanged)", STANDARD, trial_A14,
               {"j": 2}, ("rate", high_error_pass_rate(2)), 400),
    "A14-j3": ("error 2^(m-3) on the value share at Open (payload unchanged)", STANDARD, trial_A14,
               {"j": 3}, ("rate", high_error_pass_rate(3)), 400),
    "A14-j8": ("error 2^(m-8) on the value share at Open (payload unchanged)", STANDARD, trial_A14,
               {"j": 8}, ("rate", high_error_pass_rate(8)), 1000),
    "A14-guess": ("error 2^(m-1) at Open with a guess of the key share (payload unchanged)",
                  STANDARD, trial_A14, {"j": 1, "compensate": True},
                  ("rate", high_error_pass_rate(1)), 400),
    "A14-wire": ("error 2^(m-1) on the masked-input share in wire mode (x_hat unchanged)",
                 STANDARD, trial_A14, {"j": 1, "where": "wire"},
                 ("rate", high_error_pass_rate(1)), 400),
    "A14-tiny": ("error 2^(m-s) on the value share at Open, s = 4 (payload unchanged)", TINY,
                 trial_A14, {"setting": TINY, "j": 4}, ("rate", high_error_pass_rate(4)), 400),
    "A15-rush": ("shared-input Mask: share chosen after seeing the honest share (rushing)",
                 STANDARD, trial_A15, {"variant": "rush"}, "always", 40),
    "A15-mirror": ("shared-input Mask: honest commitment and opening mirrored", STANDARD,
                   trial_A15, {"variant": "mirror"}, "always", 20),
}


def run_trials(attack: str, trials: int, seed: int) -> list:
    """``trials`` independent trials of one catalogue entry; returns the list of records."""
    _, _, fn, kwargs, _, _ = ATTACKS[attack]
    rng = random.Random(f"attack|{attack}|{seed}")
    return [fn(rng, **kwargs) for _ in range(trials)]


def summarise(attack: str, records: list) -> dict:
    """Counts and verdict of one catalogue entry."""
    desc, setting, _, _, expect, _ = ATTACKS[attack]
    trials = len(records)
    errors = [r["error"] for r in records if r.get("error")]
    good = [r for r in records if not r.get("error")]
    detected = sum(1 for r in good if r["aborted"])
    accepted = sum(1 for r in good if r["accepted"])
    effective = [r for r in good if r.get("effective", True)]
    wrong = sum(1 for r in good if r["accepted"] and r.get("correct") is False)
    dead = all(r["dead"] and r["silent"] for r in good if r["aborted"])
    messages: dict = {}
    for r in good:
        if r["aborted"]:
            messages[r["message"]] = messages.get(r["message"], 0) + 1
    out = {
        "attack": attack, "description": desc, "setting": setting.tag(), "trials": trials,
        "detected": detected, "honest_outputs_released": accepted, "wrong_outputs_accepted": wrong,
        "session_dead_after_every_abort": dead, "abort_messages": messages,
        "errors": len(errors), "first_error": errors[0] if errors else None,
    }
    ok = not errors and dead
    if expect == "always":
        eff_detected = sum(1 for r in effective if r["aborted"])
        out["effective_trials"] = len(effective)
        out["expected"] = "every effective deviation is detected; no output released"
        ok = ok and eff_detected == len(effective) and wrong == 0
        ok = ok and all(not r["accepted"] for r in effective)
    elif expect == "refused":
        attempts = sum(r["attempts"] for r in good)
        refused = sum(r["refused"] for r in good)
        leaked = sum(r["leaked_msgs"] for r in good)
        out.update(mask_attempts_on_used_offsets=attempts, refused=refused,
                   messages_sent_by_refused_attempts=leaked)
        out["detected"] = refused
        out["trials"] = attempts
        out["honest_outputs_released"] = 0
        out["expected"] = "every further Mask under a used offset is refused; nothing is sent"
        ok = ok and attempts > 0 and refused == attempts and leaked == 0
        ok = ok and all(r.get("states_used") for r in good) and wrong == 0
    else:
        kind, p = expect
        runs = len(good)
        passes = runs - detected
        rate = passes / runs if runs else 0.0
        out.update(passes=passes, measured_pass_rate=rate, predicted_pass_rate=p,
                   significance=SIGNIFICANCE)
        if kind == "rate":
            pvalue = binomial_two_sided_p(passes, runs, p)
            out["expected"] = f"passes with probability {p:g}; a passed run is correct"
            out["statistical_test"] = (f"exact two-sided binomial test of {passes} passes in {runs} "
                                       f"trials against p = {p:g}")
            ok = ok and wrong == 0
        else:
            pvalue = binomial_upper_p(passes, runs, p)
            out["expected"] = f"pass probability at most the Lemma 2 bound {p:g}"
            out["statistical_test"] = (f"exact one-sided binomial test: Pr[X >= {passes}] for "
                                       f"X ~ Binomial({runs}, {p:g})")
        out["binomial_p_value"] = pvalue
        ok = ok and pvalue >= SIGNIFICANCE
    # scenario-specific facts
    if attack.startswith("A4") or any("guess_right" in r for r in good):
        # the honest x shares are recorded in every mode: a trial without them fails the row
        out["trials_with_recorded_honest_shares"] = sum(1 for r in good if "guess_right" in r)
        out["pass_iff_all_guesses_right"] = bool(good) and all(
            "guess_right" in r and (not r["aborted"]) == r["guess_right"] for r in good)
        ok = ok and out["pass_iff_all_guesses_right"]
    if any("triple_requests" in r for r in good):
        # strict mode: the honest party never draws a second bucketing population
        out["honest_triple_requests_max"] = max(r["triple_requests"] for r in good
                                                if "triple_requests" in r)
        ok = ok and out["honest_triple_requests_max"] <= 1
    if any("payload_unchanged" in r for r in good):
        out["payload_never_wrong"] = all(r.get("payload_unchanged") for r in good)
        out["passes_whose_verdict_depended_on_the_key"] = sum(
            1 for r in good if r["accepted"] and r.get("key_dependent"))
        out["aborts_whose_verdict_depended_on_the_key"] = sum(
            1 for r in good if r["aborted"] and r.get("key_dependent"))
        ok = ok and out["payload_never_wrong"] and wrong == 0
        ok = ok and all(r.get("key_dependent") for r in good if r["aborted"])
    if any("x_hat_accepted" in r for r in good):
        out["x_hat_accepted_by_honest_party"] = sum(1 for r in good if r["x_hat_accepted"])
        out["arith_openings_sent_by_honest_party"] = sum(r.get("arith_open_sent", 0) for r in good)
        ok = ok and out["x_hat_accepted_by_honest_party"] == 0
        ok = ok and out["arith_openings_sent_by_honest_party"] == 0
        ok = ok and all(r.get("target_differs", True) for r in good)
    if any("key_bit" in r for r in good):
        out["abort_iff_key_bit_is_1"] = all(r["aborted"] == bool(r["key_bit"]) for r in good)
        ok = ok and out["abort_iff_key_bit_is_1"]
    if any("inputs_sent" in r for r in good):
        out["aborts_before_any_input_message"] = all(r["inputs_sent"] == 0 for r in good
                                                     if r["aborted"])
        ok = ok and out["aborts_before_any_input_message"]
    if attack.startswith("A3") or attack == "A13-bool":
        out["output_opening_sent_by_honest_party"] = sum(r.get("output_opening_sent", 0) for r in good)
        ok = ok and out["output_opening_sent_by_honest_party"] == 0
    if attack.startswith("A6") or attack == "A14-wire":
        out["honest_output_share_sent"] = sum(1 for r in good if r.get("released") and r["aborted"])
        ok = ok and out["honest_output_share_sent"] == 0
    if any("adversary_learned_output" in r for r in good):
        out["adversary_learned_output"] = sum(1 for r in good if r.get("adversary_learned_output"))
    if any("adversary_stopped" in r for r in good):
        ok = ok and all(r["adversary_stopped"] for r in good)
    if any("predicted_pass" in r for r in good):
        # the honest verdict of every trial is the one the algebra of the check predicts
        out["verdict_equals_prediction"] = all((not r["aborted"]) == r["predicted_pass"]
                                               for r in good if "predicted_pass" in r)
        ok = ok and out["verdict_equals_prediction"]
    out["ok"] = bool(ok)
    return out


# --------------------------------------------------------------------------- #
# Work units (picklable, for process pools)
# --------------------------------------------------------------------------- #

def run_task(kind: str, name: str, count: int, seed: int) -> tuple:
    """One work unit: ``("attack", id, trials, seed)`` or ``("distinguish", id, runs per world, seed)``.

    Returns ``(kind, name, records)``; the records are plain dictionaries.
    """
    if kind == "attack":
        return kind, name, run_trials(name, count, seed)
    if kind == "distinguish":
        return kind, name, run_distinguisher(name, count, seed)
    raise ValueError(f"unknown work unit {kind!r}")


def plan_tasks(attacks: dict, distinguishers: dict, seed: int, chunk: int = 20) -> list:
    """Split ``{attack: trials}`` and ``{distinguisher: runs per world}`` into work units."""
    tasks = []
    for kind, table in (("attack", attacks), ("distinguish", distinguishers)):
        for name, total in table.items():
            done = 0
            part = 0
            while done < total:
                step = min(chunk, total - done)
                tasks.append((kind, name, step, seed * 1000 + part))
                done += step
                part += 1
    return tasks


def run_tasks(tasks: list, jobs: int = 1, progress=None) -> tuple:
    """Run work units, in ``jobs`` processes if ``jobs > 1``; returns (attack records, distinguisher records)."""
    results = {"attack": {}, "distinguish": {}}

    def absorb(item):
        kind, name, records = item
        results[kind].setdefault(name, []).extend(records)
        if progress is not None:
            progress(kind, name, len(records))

    if jobs <= 1:
        for task in tasks:
            absorb(run_task(*task))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_task, *task) for task in tasks]
            for future in as_completed(futures):
                absorb(future.result())
    return results["attack"], results["distinguish"]
