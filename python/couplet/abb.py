"""Party-local authenticated Boolean engine (DESIGN.md sections 2.2, 5.1, 7.6).

``Engine(role, chan, prep, sid)`` is the view of ONE party of the reactive
functionality F_ABB.  It evaluates a ``couplet.circuit.Circuit`` on SPDZ-style
authenticated bits ``<x>`` (XOR shares of the bit, XOR shares of the MAC
``x * Delta^B`` in GF(2^128)) and talks to the other party through a
``couplet.net.Channel`` endpoint.  Preprocessed material (AND triples and
input masks) comes from ``prep`` (``couplet.prep.RealPrep`` or ``IdealPrep``).

Batch dimension.  Every wire carries ``batch`` independent instances, so one
circuit execution produces ``batch`` results and all instances share the
rounds.  At the interface a block of wires is a ``WireShares`` object: for each
wire the share bits, shape (batch,) uint8, and the MAC shares, shape
(batch, 2) uint64 (stored as ``bits`` (width, batch) and ``macs``
(width, batch, 2)).  Inside one evaluation the engine keeps one preallocated
table with a row per wire and three 64-bit words per instance
(MAC low word, MAC high word, share bit).  With the constant-one row
``K = (Delta^B_role, [role == 0])`` in wire 1, every XOR and NOT gate is the
same operation ``row[out] = row[a] XOR row[b]``.

``eval_circuit(circuit, my_inputs, bound, batch)`` runs, in this order:

  1. preprocessing draws: ``n_and * batch`` AND triples (one request, or the
     pool filled by ``reserve_triples``) and one input mask per input bit and
     per private output bit of each party;
  2. ``Input`` of all input wires of both parties in ONE simultaneous exchange:
     each party sends a 32-byte header (hash of the session, the evaluation
     counter, the circuit digest, the batch size and the identifiers of the
     BOUND wires) followed by ``d = x XOR r`` for its own inputs; a header
     mismatch aborts.  BOUND inputs are existing authenticated wires of the
     session (a block with batch 1 is shared by all instances);
  3. XOR / NOT gates locally; one simultaneous exchange per AND layer carrying
     the ``d, e`` share bits of all gates and instances of the layer, bit-packed
     (``<z> = <c> XOR d<b> XOR e<a> XOR d e``);
  4. ``Output`` exactly as in section 2.2 (change F): BoolCheck over every
     ``d, e`` opened during the evaluation; then ONE exchange opening the
     public outputs and, for every private output of P_b, the masked wire
     ``<y> XOR <r>`` with ``r`` a fresh input mask of P_b; then a second
     BoolCheck over exactly these openings; only then is anything returned.
     RETAIN outputs are not opened and come back as ``WireShares``.

``BoolCheck`` is ``couplet.prep.bool_check`` (the hash-based refinement
recorded in design/CHANGELOG.md), run under the engine's own sub-session
identifier with a monotone counter.  A check over zero openings is skipped
(both parties know this from the circuit).

Rounds of one evaluation: 1 (inputs) + AND depth + 2 (first check) + 1
(opening) + 2 (second check), plus the rounds of the preprocessing requests.

Abort (change G).  Any failure after the first protocol message, and any
failure inside the preprocessing, marks the engine ``dead``: the MAC key share
and the triple pool are dropped, ``prep.abort()`` is called and every later
call raises ``ProtocolAbort``.  Invalid arguments are rejected with
``ValueError`` before any message and leave the engine alive.

Evaluation plan.  A circuit is compiled once into a ``Plan`` (cached by
circuit digest): the XOR / NOT gates between two AND layers are sorted into
dependency levels and the wires renumbered so that every level writes one
contiguous block of rows; a level is then two gathers and one XOR over a
(wires x batch) array, an AND layer a constant number of array operations.
``eval_circuit`` reports the measured gates per second in ``EvalResult.stats``.

Accounting.  Channel phases (prefixed with ``engine.phase_tag``): ``prep``
(local handling of preprocessed material; the dealer in ideal mode),
``input``, ``online`` (AND layers), ``outcheck`` (the two BoolChecks) and
``output`` (the opening); the preprocessing protocols book their own phases.

Statistical accounting (Lemma 4).  The bound 2^-sigma of the bucketing holds
PER ``prep.and_triples`` request: every request is one bucketing population.
A session that makes ``triple_requests`` requests is therefore at
``triple_requests * 2^-sigma`` for the triples.  The engine makes this
explicit:

  * ``triple_requests`` counts the requests, ``triples_requested`` the triples
    drawn by them; both are reported in ``EvalResult.stats``;
  * ``reserve_triples(n)`` draws the triples of several evaluations in ONE
    request (``Session.setup`` reserves together with the setup circuit);
  * ``allow_refill`` decides what happens when the pool is too small for an
    evaluation.  ``allow_refill=True`` (the default of this low-level class):
    the missing triples are requested, which is one more population and one
    more count.  ``allow_refill=False`` (strict mode; the default of
    ``autheval.AuthEval`` and of the benchmark drivers): ``eval_circuit``
    raises ``TripleBudgetError`` BEFORE any message, the engine stays alive
    and nothing was consumed; the caller has to reserve.  Both parties see
    the same pool, so both raise at the same point.

Memory: about ``(24 n_wires + 106 n_and) * batch`` bytes per evaluation.

Test hooks (``cheat`` dictionary, default None = honest; the honest path only
tests ``if cheat``):
  ``"flip_and": (layer, index)``              flip the own share of opened bit
        ``index`` of AND layer ``layer`` (order: all d, then all e; each
        gate-major, instance-minor);
  ``"flip_public": (label, bit, instance)``   flip the own share of a public
        output bit in the output opening;
  ``"flip_private": (label, bit, instance)``  the same for a masked private
        output (of either party);
  ``"flip_input": (label, bit, instance)``    send a wrong ``d`` for an own
        input bit (an input substitution; not detectable by design);
  ``"wrong_header": True``                    send a wrong evaluation header.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from . import hashes as hs
from .circuit import AND, BOUND, NOT, PUBLIC, RETAIN, Circuit
from .hashes import ProtocolAbort
from .prep.check import bool_check
from .prep.shares import unpack_bit_msg

__all__ = [
    "Engine", "EvalResult", "Plan", "ProtocolAbort", "TripleBudgetError", "WireShares",
    "bits_to_ints", "circuit_digest", "compile_plan", "get_plan", "ints_to_bits",
]

_U64 = np.uint64
_ROW = 3                 # words per wire instance: MAC low, MAC high, share bit
_HEADER_BYTES = 32
_MAX_JACOBI = 256        # level iterations before the sequential fallback
_PLAN_CACHE_SIZE = 16
_TAG_SID = b"Couplet-abb-sid"
_TAG_EVAL = b"Couplet-abb-eval"


class TripleBudgetError(ValueError):
    """The triple pool is too small for an evaluation and refills are not allowed.

    Raised in strict mode (``allow_refill=False``) before any protocol message:
    the engine and the session stay usable and nothing was consumed.  It is a
    ``ValueError`` because, like an invalid argument, it is detected locally,
    at the same point by both parties, and does not abort the session.
    ``needed`` and ``available`` are the triple counts.
    """

    def __init__(self, needed: int, available: int) -> None:
        self.needed = int(needed)
        self.available = int(available)
        super().__init__(
            f"the evaluation needs {self.needed} AND triples but the reserved pool holds "
            f"{self.available}: reserve the triples of the whole session when it is set up "
            f"(Session.setup / AuthEval.setup with reserve_triples and reserve_outmasks; "
            f"protocol.and_budget computes the number), so that they form ONE bucketing "
            f"population, or pass allow_refill=True to accept one more population "
            f"(2^-sigma each) per refill")


# --------------------------------------------------------------------------- #
# Bit helpers
# --------------------------------------------------------------------------- #

def ints_to_bits(values, width: int) -> np.ndarray:
    """Integers -> uint8 array of shape (width, len(values)); row i holds bit i (little-endian)."""
    values = [int(v) for v in values]
    width = int(width)
    nbytes = (width + 7) // 8
    for v in values:
        if v < 0 or v >> width:
            raise ValueError(f"value does not fit in {width} bits")
    raw = np.frombuffer(b"".join(v.to_bytes(nbytes, "little") for v in values), dtype=np.uint8)
    bits = np.unpackbits(raw.reshape(len(values), nbytes), axis=1, bitorder="little")[:, :width]
    return np.ascontiguousarray(bits.T)


def bits_to_ints(bits) -> list[int]:
    """Inverse of ``ints_to_bits``: array of shape (width, batch) -> list of ``batch`` integers."""
    arr = np.asarray(bits, dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected a bit array of shape (width, batch)")
    packed = np.packbits(np.ascontiguousarray(arr.T), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def _pack(bits: np.ndarray) -> bytes:
    """Bit vector (uint8 0/1) -> ceil(n / 8) bytes, least significant bit first."""
    return np.packbits(bits, bitorder="little").tobytes()


def _rows(shares) -> np.ndarray:
    """``prep.Shares`` of n bits -> table rows of shape (n, 3): MAC low, MAC high, bit."""
    out = np.empty((len(shares), _ROW), dtype=_U64)
    out[:, :2] = shares.macs
    out[:, 2] = shares.bits
    return out


# --------------------------------------------------------------------------- #
# Wire shares (the interface representation of authenticated wires)
# --------------------------------------------------------------------------- #

class WireShares:
    """One party's shares of ``width`` authenticated wires over ``batch`` instances.

    ``bits``  uint8, shape (width, batch): XOR shares of the wire values;
    ``macs``  uint64, shape (width, batch, 2): XOR shares of ``value * Delta^B``;
    ``ids``   uint64, shape (width,): session-wide wire identifiers (equal at
              both parties; part of the evaluation header when the wires are
              bound to a later circuit).
    Indexing with an integer or a slice selects wires and returns a
    ``WireShares``; iteration yields the single wires.  The object belongs to
    the engine that produced it and cannot be bound in another session.
    """

    __slots__ = ("bits", "macs", "ids", "_token")

    def __init__(self, bits: np.ndarray, macs: np.ndarray, ids: np.ndarray, token) -> None:
        if bits.ndim != 2 or macs.shape != bits.shape + (2,) or ids.shape != bits.shape[:1]:
            raise ValueError("inconsistent wire share arrays")
        self.bits = bits
        self.macs = macs
        self.ids = ids
        self._token = token

    @property
    def width(self) -> int:
        return self.bits.shape[0]

    @property
    def batch(self) -> int:
        return self.bits.shape[1]

    def __len__(self) -> int:
        return self.bits.shape[0]

    def __getitem__(self, key) -> "WireShares":
        if isinstance(key, (int, np.integer)):
            key = int(key)
            if not -len(self) <= key < len(self):
                raise IndexError("wire index out of range")
            key = slice(key, key + 1) if key != -1 else slice(key, None)
        elif not isinstance(key, slice):
            raise TypeError("wires are selected with an integer or a slice")
        return WireShares(self.bits[key], self.macs[key], self.ids[key], self._token)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def wire(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(bits of shape (batch,), MAC shares of shape (batch, 2)) of wire ``i``."""
        return self.bits[i], self.macs[i]

    @classmethod
    def concat(cls, parts) -> "WireShares":
        """Concatenation of wire blocks of one engine with the same batch size."""
        parts = list(parts)
        if not parts or not all(isinstance(p, WireShares) for p in parts):
            raise ValueError("expected a non-empty sequence of WireShares")
        if len({id(p._token) for p in parts}) != 1 or len({p.batch for p in parts}) != 1:
            raise ValueError("wire blocks must come from one engine and have one batch size")
        return cls(np.concatenate([p.bits for p in parts]), np.concatenate([p.macs for p in parts]),
                   np.concatenate([p.ids for p in parts]), parts[0]._token)


@dataclass
class EvalResult:
    """Outcome of one circuit evaluation for this party.

    ``public``    label -> bits (width, batch) of every PUBLIC output;
    ``private``   label -> bits (width, batch) of the private outputs owned by
                  this party (and of no other);
    ``retained``  label -> ``WireShares`` of every RETAIN output;
    ``stats``     counters and timings of this evaluation (see ``Engine``).
    """

    public: dict
    private: dict
    retained: dict
    stats: dict = field(default_factory=dict)

    def ints(self, label: str) -> list[int]:
        """Opened output ``label`` as one integer per batch instance."""
        if label in self.public:
            return bits_to_ints(self.public[label])
        return bits_to_ints(self.private[label])


# --------------------------------------------------------------------------- #
# Evaluation plan
# --------------------------------------------------------------------------- #

@dataclass
class Plan:
    """Compiled evaluation order of one circuit (party-independent, cached by digest).

    ``steps`` is a list of ``(is_and, u, v, a, b, t0)``: the gates of the step
    write the table rows ``[u, v)`` from the rows ``a`` and ``b`` (index
    arrays); for an AND layer ``t0`` is the index of its first triple.
    """

    digest: bytes
    n_slots: int
    n_gates: int
    n_and: int
    and_depth: int
    steps: list
    n_linear_steps: int
    max_linear: int
    in_slots: tuple          # per owner 0 / 1: rows of all its input wires, declaration order
    in_fields: tuple         # per owner: [(label, offset, width)]
    bound_fields: list       # [(label, rows)]
    out_public: list         # [(label, offset into the opening vector, width)]
    out_private: tuple       # per owner: [(label, offset into the opening vector, width)]
    out_retain: list         # [(label, rows)]
    open_slots: np.ndarray   # rows opened at the end: public, private of P_0, private of P_1
    n_public: int
    n_private: tuple

    @property
    def n_open(self) -> int:
        return int(self.open_slots.shape[0])

    def n_masks(self, owner: int) -> int:
        """Input masks of P_owner per instance: its input bits plus its private output bits."""
        return int(self.in_slots[owner].shape[0]) + self.n_private[owner]


def circuit_digest(circuit: Circuit) -> bytes:
    """SHA-256 of the canonical binary encoding (32 bytes; memoised on the circuit object)."""
    cached = getattr(circuit, "_abb_digest", None)
    if cached is None:
        cached = hashlib.sha256(circuit.to_bytes()).digest()
        try:
            circuit._abb_digest = cached
        except AttributeError:
            pass
    return cached


def _levels_sequential(n_wires, lin_wires, la, lb, same_a, same_b) -> np.ndarray:
    """Dependency level of every linear gate inside its block (plain loop; fallback)."""
    lvl = [0] * n_wires
    for w, a, b, sa, sb in zip(lin_wires.tolist(), la.tolist(), lb.tolist(),
                               same_a.tolist(), same_b.tolist()):
        x = lvl[a] if sa else 0
        y = lvl[b] if sb else 0
        lvl[w] = (x if x > y else y) + 1
    return np.asarray(lvl, dtype=np.int32)[lin_wires]


def compile_plan(circuit: Circuit) -> Plan:
    """Compile ``circuit`` into a ``Plan`` (validates the canonical form it relies on)."""
    if not isinstance(circuit, Circuit):
        raise ValueError("expected a couplet.circuit.Circuit")
    n_wires = int(circuit.n_wires)
    ng = int(circuit.n_gates)
    base = 2 + int(circuit.n_input_wires)
    if n_wires != base + ng:
        raise ValueError("circuit is not in canonical form (wire count)")
    op = np.frombuffer(bytes(circuit.op), dtype=np.uint8)
    ga = np.array(circuit.a, dtype=np.int64)
    gb = np.array(circuit.b, dtype=np.int64)
    gout = np.array(circuit.out, dtype=np.int64)
    if not (ga.shape[0] == gb.shape[0] == gout.shape[0] == ng):
        raise ValueError("gate arrays have different lengths")
    idx = np.arange(ng, dtype=np.int64)
    if not np.array_equal(gout, base + idx):
        raise ValueError("circuit is not in canonical form (gate outputs)")
    if ng and int(op.max()) > NOT:
        raise ValueError("unknown gate code")
    layers = [(int(s), int(e)) for s, e in circuit.layer_ranges()]
    prev = 0
    for s, e in layers:
        if not prev <= s < e <= ng:
            raise ValueError("AND layers must be non-empty, increasing and disjoint")
        prev = e
    is_and = op == AND
    in_layer = np.zeros(ng, dtype=bool)
    limit = base + idx                       # first wire a gate may NOT read
    for s, e in layers:
        in_layer[s:e] = True
        limit[s:e] = base + s
    if not np.array_equal(is_and, in_layer):
        raise ValueError("AND layers must contain exactly the AND gates")
    gb = np.where(op == NOT, 1, gb)          # NOT x = x XOR (constant-one wire)
    if ng and (int(ga.min()) < 0 or int(gb.min()) < 0 or bool((ga >= limit).any())
               or bool((gb >= limit).any())):
        raise ValueError("gate operand not available when the gate is evaluated")

    # Levels of the linear gates inside their block (the gates between two AND layers).
    starts = np.array([s for s, _ in layers], dtype=np.int64)
    blk = np.searchsorted(starts, idx, side="right").astype(np.int64)  # AND layers before the gate
    lin = np.flatnonzero(~is_and)
    lin_wires = base + lin
    slot_of = np.arange(n_wires, dtype=np.int64)
    steps: list = []
    lin_groups: list = []                    # (block, u, v, a, b)
    if lin.shape[0]:
        la, lb, lblk = ga[lin], gb[lin], blk[lin]
        wblk = np.full(n_wires, -1, dtype=np.int64)
        wblk[lin_wires] = lblk
        same_a = wblk[la] == lblk
        same_b = wblk[lb] == lblk
        lvl = np.zeros(n_wires, dtype=np.int32)
        cur = lvl[lin_wires]
        for _ in range(_MAX_JACOBI):
            new = np.maximum(np.where(same_a, lvl[la], 0), np.where(same_b, lvl[lb], 0)) + 1
            if np.array_equal(new, cur):
                break
            lvl[lin_wires] = new
            cur = new
        else:
            cur = _levels_sequential(n_wires, lin_wires, la, lb, same_a, same_b)
        key = lblk * (int(cur.max()) + 1) + cur
        order = np.argsort(key, kind="stable")
        slot_of[lin_wires[order]] = lin_wires   # i-th gate of the sorted order -> i-th linear row
        skey = key[order]
        cuts = np.flatnonzero(skey[1:] != skey[:-1]) + 1
        bounds = np.concatenate(([0], cuts, [skey.shape[0]]))
        sa = slot_of[la[order]].astype(np.intp)
        sb = slot_of[lb[order]].astype(np.intp)
        sblk = lblk[order]
        for p, q in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
            u = int(lin_wires[p])
            v = int(lin_wires[q - 1]) + 1
            if v - u != q - p:
                raise ValueError("internal error: a linear level is not contiguous")
            lin_groups.append((int(sblk[p]), u, v, sa[p:q].copy(), sb[p:q].copy()))

    max_linear = 0
    t0 = 0
    done_layers = 0

    def emit_layers(upto: int) -> None:
        nonlocal t0, done_layers
        while done_layers < upto:
            s, e = layers[done_layers]
            steps.append((True, base + s, base + e, slot_of[ga[s:e]].astype(np.intp),
                          slot_of[gb[s:e]].astype(np.intp), t0))
            t0 += e - s
            done_layers += 1

    for block, u, v, a, b in lin_groups:
        emit_layers(block)                   # linear block j comes before AND layer j
        steps.append((False, u, v, a, b, 0))
        max_linear = max(max_linear, v - u)
    emit_layers(len(layers))

    def slots(wires) -> np.ndarray:
        return slot_of[np.asarray(wires, dtype=np.int64)].astype(np.intp)

    in_wires = ([], [])
    in_fields = ([], [])
    bound_fields = []
    for owner, label, wires in circuit.inputs:
        if owner == BOUND:
            bound_fields.append((label, slots(wires)))
        elif owner in (0, 1):
            in_fields[owner].append((label, len(in_wires[owner]), len(wires)))
            in_wires[owner].extend(wires)
        else:
            raise ValueError(f"input {label!r}: unknown owner {owner!r}")
    out_public, out_private, out_retain = [], ([], []), []
    open_wires = {PUBLIC: [], 0: [], 1: []}
    for to, label, wires in circuit.outputs:
        if isinstance(to, str) and to == RETAIN:
            out_retain.append((label, slots(wires)))
        elif isinstance(to, str) and to == PUBLIC:
            out_public.append((label, len(open_wires[PUBLIC]), len(wires)))
            open_wires[PUBLIC].extend(wires)
        elif not isinstance(to, (str, bool)) and to in (0, 1):
            out_private[to].append((label, len(open_wires[to]), len(wires)))
            open_wires[to].extend(wires)
        else:
            raise ValueError(f"output {label!r}: unknown owner {to!r}")
    n_public = len(open_wires[PUBLIC])
    n_private = (len(open_wires[0]), len(open_wires[1]))
    # offsets of the private fields inside the whole opening vector
    priv0 = [(label, n_public + off, width) for label, off, width in out_private[0]]
    priv1 = [(label, n_public + n_private[0] + off, width) for label, off, width in out_private[1]]
    return Plan(
        digest=circuit_digest(circuit), n_slots=n_wires, n_gates=ng, n_and=int(is_and.sum()),
        and_depth=len(layers), steps=steps, n_linear_steps=len(lin_groups), max_linear=max_linear,
        in_slots=(slots(in_wires[0]), slots(in_wires[1])), in_fields=in_fields,
        bound_fields=bound_fields, out_public=out_public, out_private=(priv0, priv1),
        out_retain=out_retain,
        open_slots=slots(open_wires[PUBLIC] + open_wires[0] + open_wires[1]),
        n_public=n_public, n_private=n_private)


_PLAN_CACHE: "OrderedDict[bytes, Plan]" = OrderedDict()
_PLAN_LOCK = threading.Lock()   # two in-process parties compile a plan only once


def get_plan(circuit: Circuit) -> Plan:
    """The evaluation plan of ``circuit``, compiled on first use and cached by circuit digest."""
    if not isinstance(circuit, Circuit):
        raise ValueError("expected a couplet.circuit.Circuit")
    with _PLAN_LOCK:
        digest = circuit_digest(circuit)
        plan = _PLAN_CACHE.get(digest)
        if plan is None:
            plan = compile_plan(circuit)
            _PLAN_CACHE[digest] = plan
            while len(_PLAN_CACHE) > _PLAN_CACHE_SIZE:
                _PLAN_CACHE.popitem(last=False)
        else:
            _PLAN_CACHE.move_to_end(digest)
    return plan


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class Engine:
    """Authenticated Boolean engine of one party; see the module docstring.

    Attributes: ``role``, ``chan``, ``prep``, ``sid``, ``dead``, ``cheat``,
    ``phase_tag`` (prefix of the channel phases), ``evaluations``,
    ``allow_refill`` (False = strict mode, see the module docstring),
    ``triple_requests`` (bucketing populations drawn so far),
    ``triples_requested`` (triples drawn by these requests),
    ``triples_used``, ``last_stats`` (stats of the latest evaluation) and
    ``last_opened`` (public transcript values of the latest evaluation, for
    tests: ``{"public": {label: bits}, "masked": {label: bits},
    "input_diffs": (d of P_0, d of P_1)}``; ``masked`` holds the public
    openings ``y XOR r`` of the private outputs, ``input_diffs`` the values
    ``x XOR r`` sent for the inputs, shape (input wires, batch)).
    """

    def __init__(self, role: int, chan, prep, sid: bytes, *, cheat=None,
                 allow_refill: bool = True) -> None:
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        if chan.party != role:
            raise ValueError("role does not match the channel endpoint")
        if getattr(prep, "role", role) != role:
            raise ValueError("role does not match the preprocessing object")
        if not isinstance(sid, (bytes, bytearray)):
            raise ValueError("sid must be bytes")
        if not isinstance(allow_refill, bool):
            raise ValueError("allow_refill must be a bool")
        self.role = role
        self.chan = chan
        self.prep = prep
        self.sid = bytes(sid)
        self.cheat = cheat
        self.allow_refill = allow_refill
        self.phase_tag = ""
        self.dead = False
        self.evaluations = 0
        self.triple_requests = 0
        self.triples_requested = 0
        self.triples_used = 0
        self.last_stats: dict = {}
        self.last_opened: dict = {}
        self._sid = hs.sha256(_TAG_SID, self.sid)   # sub-session of the engine's checks
        self._token = object()
        self._checks = 0
        self._next_wire_id = 0
        self._pool = None        # (a, b, c) rows of unused triples, each of shape (n, 3)
        self._pool_pos = 0
        self._delta = None
        self._k3 = None
        self._load_key()

    # -- life cycle ---------------------------------------------------------- #

    def _load_key(self) -> None:
        delta = getattr(self.prep, "delta", None)
        if delta is None:
            raise ValueError("the preprocessing object has no MAC key share (aborted?)")
        delta = int(delta)
        self._delta = delta
        self._k3 = np.array([delta & hs.MASK64, delta >> 64, 1 - self.role], dtype=_U64)

    def abort(self) -> None:
        """Retire the engine (change G): drop the key share, the pool and the preprocessing."""
        self.dead = True
        self._delta = None
        self._k3 = None
        self._pool = None
        try:
            self.prep.abort()
        except Exception:  # noqa: BLE001 - the engine is dead whatever the prep object does
            pass

    def _require_alive(self) -> None:
        if self.dead or getattr(self.prep, "aborted", False):
            if not self.dead:
                self.abort()
            raise ProtocolAbort("session aborted")

    # -- preprocessing ------------------------------------------------------- #

    def triples_available(self) -> int:
        """Unused triples in the pool."""
        return 0 if self._pool is None else self._pool[0].shape[0] - self._pool_pos

    def require_triples(self, n: int) -> None:
        """Strict mode: raise ``TripleBudgetError`` unless the pool holds ``n`` unused triples.

        With ``allow_refill=True`` this never raises (a short pool is refilled
        by the evaluation, which is one more bucketing population).  No message
        is sent and nothing changes.
        """
        n = int(n)
        if not self.allow_refill and n > self.triples_available():
            raise TripleBudgetError(n, self.triples_available())

    def reserve_triples(self, n: int) -> None:
        """Draw ``n`` further AND triples into the pool with ONE preprocessing request.

        Every call with ``n > 0`` is one bucketing population (``triple_requests``
        grows by one), also in strict mode: it is the explicit way to add one.
        """
        n = int(n)
        if n < 0:
            raise ValueError("the number of triples must be non-negative")
        self._require_alive()
        if n == 0:
            return
        phase = self.chan.phase
        try:
            self.chan.mark(self.phase_tag + "prep")
            self._fill_pool(n)
        except BaseException:
            self.abort()
            raise
        finally:
            self.chan.mark(phase)

    def _fill_pool(self, n: int) -> None:
        a, b, c = self.prep.and_triples(n)
        self.triple_requests += 1
        self.triples_requested += n
        new = (_rows(a), _rows(b), _rows(c))
        if self.triples_available():
            pos = self._pool_pos
            new = tuple(np.concatenate((old[pos:], add)) for old, add in zip(self._pool, new))
        self._pool = new
        self._pool_pos = 0

    def _take_triples(self, n: int):
        """``n`` triples as three row arrays of shape (n, 3) (views into the pool)."""
        missing = n - self.triples_available()
        if missing > 0:
            if not self.allow_refill:       # eval_circuit has checked this before any message
                raise TripleBudgetError(n, self.triples_available())
            self._fill_pool(missing)
        pos = self._pool_pos
        out = tuple(arr[pos:pos + n] for arr in self._pool)
        self._pool_pos = pos + n
        self.triples_used += n
        if self.triples_available() == 0:
            self._pool = None
            self._pool_pos = 0
        return out

    # -- argument handling (ValueError, before any message) --------------------- #

    def _own_inputs(self, plan: Plan, my_inputs, batch: int) -> np.ndarray:
        """This party's input bits as an array of shape (number of own input wires, batch)."""
        fields = plan.in_fields[self.role]
        given = {}
        for key, value in dict(my_inputs or {}).items():
            if isinstance(key, tuple):
                if len(key) != 2 or key[0] != self.role:
                    raise ValueError(f"input key {key!r} does not belong to party {self.role}")
                key = key[1]
            if key in given:
                raise ValueError(f"input {key!r} given twice")
            given[key] = value
        out = np.empty((int(plan.in_slots[self.role].shape[0]), batch), dtype=np.uint8)
        for label, off, width in fields:
            if label not in given:
                raise ValueError(f"missing input {label!r} of party {self.role}")
            value = given.pop(label)
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                bits = ints_to_bits([int(value)], width)
            else:
                bits = np.asarray(value)
                if bits.dtype == np.bool_:
                    bits = bits.astype(np.uint8)
                if bits.dtype.kind not in "ui":
                    raise ValueError(f"input {label!r}: bits must be integers")
                if bits.ndim == 1:
                    bits = bits[:, None]
                if bits.ndim != 2 or bits.shape[0] != width or bits.shape[1] not in (1, batch):
                    raise ValueError(f"input {label!r}: expected shape ({width}, {batch})")
                if bits.size and (int(bits.max()) > 1 or int(bits.min()) < 0):
                    raise ValueError(f"input {label!r}: bits must be 0 or 1")
            out[off:off + width] = bits
        if given:
            raise ValueError(f"unknown input labels for party {self.role}: {sorted(given)}")
        return out

    def _bound_rows(self, plan: Plan, bound, batch: int):
        """Table rows of the BOUND inputs and the byte string describing them in the header."""
        given = dict(bound or {})
        rows_out = []
        desc = []
        for label, slots in plan.bound_fields:
            if label not in given:
                raise ValueError(f"missing BOUND input {label!r}")
            ws = given.pop(label)
            if not isinstance(ws, WireShares):
                ws = WireShares.concat(ws)
            if ws._token is not self._token:
                raise ValueError(f"BOUND input {label!r}: the wires belong to another session")
            if ws.width != slots.shape[0]:
                raise ValueError(f"BOUND input {label!r}: expected {slots.shape[0]} wires")
            if ws.batch not in (1, batch):
                raise ValueError(f"BOUND input {label!r}: batch must be 1 or {batch}")
            rows = np.empty((ws.width, batch, _ROW), dtype=_U64)
            rows[:, :, :2] = ws.macs
            rows[:, :, 2] = ws.bits
            rows_out.append((slots, rows.reshape(ws.width, _ROW * batch)))
            desc += [label.encode("utf-8"), ws.ids.astype("<u8").tobytes(),
                     ws.batch.to_bytes(8, "little")]
        if given:
            raise ValueError(f"unknown BOUND labels: {sorted(given)}")
        return rows_out, hs.sha256(*desc)

    def _cheat_index(self, fields, spec, batch: int) -> int:
        """Position in a (wire, instance) row-major bit vector of the hook target ``spec``."""
        label, bit, instance = spec
        for name, off, width in fields:
            if name == label:
                if not (0 <= bit < width and 0 <= instance < batch):
                    raise ValueError("test hook: bit or instance out of range")
                return (off + bit) * batch + instance
        raise ValueError(f"test hook: unknown label {label!r}")

    # -- evaluation ---------------------------------------------------------- #

    def eval_circuit(self, circuit: Circuit, my_inputs=None, bound=None, batch: int = 1) -> EvalResult:
        """Evaluate ``circuit`` on ``batch`` independent instances; see the module docstring.

        ``my_inputs``  label -> bits of shape (width, batch) (or (width,) /
                       an integer, used for every instance) for every input
                       declaration owned by this party;
        ``bound``      label -> ``WireShares`` (or a sequence of them) for
                       every BOUND input; batch 1 or ``batch``;
        ``batch``      number of instances.
        Raises ``ValueError`` for invalid arguments (the engine stays alive),
        ``TripleBudgetError`` (a ``ValueError``) in strict mode if the pool
        holds fewer than ``n_and * batch`` triples (before any message; the
        engine stays alive) and ``ProtocolAbort`` if the protocol fails (the
        engine is then dead).
        """
        self._require_alive()
        if isinstance(batch, bool) or int(batch) != batch or int(batch) < 1:
            raise ValueError("batch must be a positive integer")
        batch = int(batch)
        plan = get_plan(circuit)
        x_own = self._own_inputs(plan, my_inputs, batch)
        bound_rows, bound_desc = self._bound_rows(plan, bound, batch)
        self.require_triples(plan.n_and * batch)     # strict mode: no silent new population
        cheat = self.cheat
        hooks = {}
        if cheat:
            role = self.role
            if "flip_input" in cheat:
                hooks["input"] = self._cheat_index(plan.in_fields[role], cheat["flip_input"], batch)
            if "flip_public" in cheat:
                hooks["open"] = self._cheat_index(plan.out_public, cheat["flip_public"], batch)
            if "flip_private" in cheat:
                hooks["open"] = self._cheat_index(plan.out_private[0] + plan.out_private[1],
                                                  cheat["flip_private"], batch)
            if "flip_and" in cheat:
                hooks["and"] = (int(cheat["flip_and"][0]), int(cheat["flip_and"][1]))
            if cheat.get("wrong_header"):
                hooks["header"] = True
        phase = self.chan.phase
        try:
            return self._run(plan, x_own, bound_rows, bound_desc, batch, hooks)
        except BaseException:
            self.abort()
            raise
        finally:
            self.chan.mark(phase)

    def _check(self, opened_bits: np.ndarray, macs: np.ndarray) -> None:
        counter = self._checks
        self._checks += 1
        bool_check(self.chan, self._sid, counter, self.role, self._delta, opened_bits, macs)

    def _run(self, plan: Plan, x_own: np.ndarray, bound_rows, bound_desc: bytes, batch: int,
             hooks: dict) -> EvalResult:
        role, chan, tag = self.role, self.chan, self.phase_tag
        k3 = self._k3
        width3 = _ROW * batch
        clock = time.perf_counter
        t_begin = clock()
        rounds0, sent0 = chan.rounds, chan.bytes_sent
        eval_index = self.evaluations
        self.evaluations += 1

        # 1. Preprocessed material: triples, then the input masks of P_0 and of P_1.
        chan.mark(tag + "prep")
        n_and = plan.n_and
        if n_and:
            ta, tb, tc = (arr.reshape(n_and, width3) for arr in self._take_triples(n_and * batch))
        n_in = (int(plan.in_slots[0].shape[0]), int(plan.in_slots[1].shape[0]))
        mask_rows = [None, None]
        mask_bits = None
        for owner in (0, 1):
            count = plan.n_masks(owner) * batch
            if count:
                shares, bits = self.prep.input_masks(owner, count)
                mask_rows[owner] = _rows(shares)
                if owner == role:
                    mask_bits = np.asarray(bits, dtype=np.uint8)
        table = np.empty((plan.n_slots, width3), dtype=_U64)
        table[0] = 0
        table[1] = np.tile(k3, batch)
        t_prep = clock()

        # 2. Input: one simultaneous exchange (header || d = x XOR r for the own inputs).
        chan.mark(tag + "input")
        header = hs.sha256(_TAG_EVAL, self._sid, eval_index.to_bytes(8, "little"), plan.digest,
                           batch.to_bytes(8, "little"), bound_desc)
        if n_in[role]:
            d_own = x_own ^ mask_bits[:n_in[role] * batch].reshape(n_in[role], batch)
        else:
            d_own = np.zeros((0, batch), dtype=np.uint8)
        if hooks:
            if "input" in hooks:
                d_own = d_own.copy()
                d_own.reshape(-1)[hooks["input"]] ^= 1
            if "header" in hooks:
                header = hs.sha256(b"wrong", header)
        reply = chan.exchange(header + _pack(d_own.reshape(-1)))
        if len(reply) < _HEADER_BYTES or bytes(reply[:_HEADER_BYTES]) != header:
            raise ProtocolAbort("evaluation header mismatch (circuit digest, batch or bound wires)")
        peer = 1 - role
        d_peer = unpack_bit_msg(reply[_HEADER_BYTES:], n_in[peer] * batch).reshape(n_in[peer], batch)
        for owner, d in ((role, d_own), (peer, d_peer)):
            if n_in[owner]:
                rows = mask_rows[owner][:n_in[owner] * batch].reshape(n_in[owner], batch, _ROW)
                rows = rows ^ (k3 & np.negative(d.astype(_U64))[:, :, None])
                table[plan.in_slots[owner]] = rows.reshape(n_in[owner], width3)
        for slots, rows in bound_rows:
            table[slots] = rows
        t_input = clock()

        # 3. Gates: linear levels locally, one exchange per AND layer.
        chan.mark(tag + "online")
        n_checked = 2 * n_and * batch
        chk_bits = np.empty(n_checked, dtype=np.uint8)
        chk_macs = np.empty((n_checked, 2), dtype=_U64)
        buf_a = np.empty((max(plan.max_linear, 1), width3), dtype=_U64)
        buf_b = np.empty_like(buf_a)
        take, xor = np.take, np.bitwise_xor
        pos = 0
        layer_no = 0
        wait = 0.0
        for is_and, u, v, ia, ib, t0 in plan.steps:
            m = v - u
            if not is_and:
                take(table, ia, axis=0, out=buf_a[:m], mode="clip")
                take(table, ib, axis=0, out=buf_b[:m], mode="clip")
                xor(buf_a[:m], buf_b[:m], out=table[u:v])
                continue
            a3 = ta[t0:t0 + m].reshape(m, batch, _ROW)
            b3 = tb[t0:t0 + m].reshape(m, batch, _ROW)
            x3 = (table.take(ia, axis=0) ^ ta[t0:t0 + m]).reshape(m, batch, _ROW)   # <d> = <x> ^ <a>
            y3 = (table.take(ib, axis=0) ^ tb[t0:t0 + m]).reshape(m, batch, _ROW)   # <e> = <y> ^ <b>
            mb = m * batch
            own = np.empty(2 * mb, dtype=np.uint8)
            own[:mb] = x3[:, :, 2].reshape(-1)
            own[mb:] = y3[:, :, 2].reshape(-1)
            if hooks and "and" in hooks and hooks["and"][0] == layer_no:
                own[hooks["and"][1]] ^= 1
            t_x = clock()
            reply = chan.exchange(_pack(own))
            wait += clock() - t_x
            opened = own ^ unpack_bit_msg(reply, 2 * mb)
            dm = np.negative(opened[:mb].astype(_U64)).reshape(m, batch, 1)
            em = np.negative(opened[mb:].astype(_U64)).reshape(m, batch, 1)
            z3 = tc[t0:t0 + m].reshape(m, batch, _ROW) ^ (b3 & dm)
            z3 ^= a3 & em
            z3 ^= k3 & (dm & em)
            table[u:v] = z3.reshape(m, width3)
            chk_bits[pos:pos + 2 * mb] = opened
            chk_macs[pos:pos + mb] = x3[:, :, :2].reshape(mb, 2)
            chk_macs[pos + mb:pos + 2 * mb] = y3[:, :, :2].reshape(mb, 2)
            pos += 2 * mb
            layer_no += 1
        t_online = clock()

        # 4. Output (section 2.2, change F).
        chan.mark(tag + "outcheck")
        if n_checked:
            self._check(chk_bits, chk_macs)                      # (1) every d, e of the evaluation
        t_check1 = clock()
        public, private, masked = {}, {}, {}
        n_open = plan.n_open
        n_priv = plan.n_private
        if n_open:
            chan.mark(tag + "output")
            rows = table.take(plan.open_slots, axis=0).reshape(n_open, batch, _ROW)
            off = plan.n_public
            for owner in (0, 1):                                 # private wires: <y> XOR <r>
                if n_priv[owner]:
                    rows[off:off + n_priv[owner]] ^= mask_rows[owner][n_in[owner] * batch:].reshape(
                        n_priv[owner], batch, _ROW)
                    off += n_priv[owner]
            own = rows[:, :, 2].astype(np.uint8).reshape(-1)
            if hooks and "open" in hooks:
                own[hooks["open"]] ^= 1
            reply = chan.exchange(_pack(own))                    # (2) open
            opened = own ^ unpack_bit_msg(reply, n_open * batch)
            chan.mark(tag + "outcheck")
            self._check(opened, np.ascontiguousarray(rows[:, :, :2]).reshape(n_open * batch, 2))  # (3)
            opened = opened.reshape(n_open, batch)               # (4) accepted
            for label, off, width in plan.out_public:
                public[label] = opened[off:off + width].copy()
            for owner in (0, 1):
                for label, off, width in plan.out_private[owner]:
                    masked[label] = opened[off:off + width].copy()
            if n_priv[role]:
                r_own = mask_bits[n_in[role] * batch:].reshape(n_priv[role], batch)
                start = plan.n_public + (n_priv[0] if role == 1 else 0)
                for label, off, width in plan.out_private[role]:
                    private[label] = opened[off:off + width] ^ r_own[off - start:off - start + width]
        retained = {}
        for label, slots in plan.out_retain:
            rows = table.take(slots, axis=0).reshape(slots.shape[0], batch, _ROW)
            ids = np.arange(self._next_wire_id, self._next_wire_id + slots.shape[0], dtype=_U64)
            self._next_wire_id += int(slots.shape[0])
            retained[label] = WireShares(rows[:, :, 2].astype(np.uint8),
                                         np.ascontiguousarray(rows[:, :, :2]), ids, self._token)
        t_end = clock()

        online = t_online - t_input
        work = plan.n_gates * batch
        stats = {
            "evaluation": eval_index, "batch": batch, "n_gates": plan.n_gates, "n_and": n_and,
            "and_depth": plan.and_depth, "linear_steps": plan.n_linear_steps,
            "triples": n_and * batch, "input_masks": (plan.n_masks(0) * batch, plan.n_masks(1) * batch),
            "triple_requests": self.triple_requests, "triples_requested": self.triples_requested,
            "triples_available": self.triples_available(),
            "opened_bits": n_open * batch, "checked_bits": n_checked + n_open * batch,
            "rounds": chan.rounds - rounds0, "bytes_sent": chan.bytes_sent - sent0,
            "prep_s": t_prep - t_begin, "input_s": t_input - t_prep, "online_s": online,
            "online_exchange_wait_s": wait, "check_s": t_check1 - t_online,
            "output_s": t_end - t_check1, "total_s": t_end - t_begin,
            "gates_per_s": work / online if online > 0 else None,
            "gates_per_s_local": work / (online - wait) if online - wait > 0 else None,
        }
        self.last_stats = stats
        diffs = (d_own, d_peer) if role == 0 else (d_peer, d_own)
        self.last_opened = {"public": public, "masked": masked, "input_diffs": diffs}
        return EvalResult(public, private, retained, stats)
