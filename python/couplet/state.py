"""Per-session registry of keys and gates: the state machine of DESIGN.md section 5.4.

``Registry(sid, path=None)`` is the bookkeeping of ONE party for one session:

    session   ACTIVE -> ABORTED                 (absorbing, change G)
    key       GENERATING -> CERTIFIED           (CERTIFIED only after the output
                                                 MAC check of Pi_Gen, change F)
    gate      GENERATING -> CERTIFIED -> USED   (one ``Mask`` per gate)

Rules enforced here, all BEFORE any message is sent (a refused call raises
``StateError`` and leaves the transcript untouched):

  * identifiers ``kid``, ``gid``, ``qid`` are monotone: an identifier is
    accepted only if it is larger than every identifier of its kind issued
    before, in this process or (with persistence) in an earlier one.  The two
    parties use the same identifiers because they run the same public
    sequence of commands;
  * a key is evaluated only when it is CERTIFIED;
  * ``use_gate`` moves a gate from CERTIFIED to USED exactly once.  The
    caller invokes it before the first byte of the masked opening is sent, so
    one offset masks one input even if the opening fails, the peer
    disappears, the session aborts or an exception is caught and the
    operation is retried.  A second call raises ``GateUsedError``;
  * after ``abort()`` every method raises ``ProtocolAbort`` and the key and
    gate material is dropped.

Persistence (optional).  With ``path`` the counters are kept in a small JSON
file ``{"format", "sid", "state", "reserved": {"kid", "gid", "qid"},
"starts"}``.  The file stores, per kind, a RESERVED bound: an identifier is
issued only below the stored bound, and the bound is raised (in blocks) and
written to disk before the identifiers of the block are used.  The file is
replaced atomically (temporary file, flush, fsync, rename).  A process that
starts on an existing file continues ABOVE the stored bounds, so it can never
issue an identifier again; every gate of the earlier process, used or not, is
below the new floor and can no longer be masked (the material of a crashed
process is treated as used: a reconnect never reuses an offset).  A file that
records ABORTED keeps the session aborted; a file of another session or an
unreadable file is refused instead of being overwritten.

The registry stores what the driver gives it: key objects, ``GateMaterial``,
the gate specification, the public masked input ``x_hat`` once ``Mask`` has
run.  It never talks to the channel.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

from .hashes import ProtocolAbort

__all__ = [
    "ABORTED", "ACTIVE", "CERTIFIED", "GENERATING", "GateRecord", "GateUsedError", "KeyRecord",
    "ProtocolAbort", "Registry", "StateError", "USED",
]

ACTIVE = "ACTIVE"
ABORTED = "ABORTED"
GENERATING = "GENERATING"
CERTIFIED = "CERTIFIED"
USED = "USED"

KINDS = ("kid", "gid", "qid")
FORMAT = "couplet-state-1"
_BLOCK = {"kid": 64, "gid": 64, "qid": 4096}   # identifiers reserved per write of the file


class StateError(RuntimeError):
    """A command is not allowed in the current state; nothing was sent."""


class GateUsedError(StateError):
    """A second ``Mask`` on a gate whose offset has already been used."""


@dataclass
class KeyRecord:
    """One key: ``state`` GENERATING or CERTIFIED; ``key`` is set when it is certified."""

    kid: int
    state: str = GENERATING
    key: object = None
    info: dict = field(default_factory=dict)


@dataclass
class GateRecord:
    """One gate: ``state`` GENERATING, CERTIFIED or USED.

    ``material`` / ``spec`` / ``evaluator`` are set when the gate is
    certified; ``x_hat`` (public), ``mode`` (``"shared"`` or ``"wire"``) and
    ``opened`` (the public opened representative in wire mode) once ``Mask``
    has completed.  A gate that is USED without ``x_hat`` had a failed ``Mask``
    and can be neither masked again nor applied.
    """

    gid: int
    state: str = GENERATING
    spec: object = None
    material: object = None
    evaluator: object = None
    x_hat: int | None = None
    mode: str | None = None
    opened: int | None = None


class Registry:
    """State machine of one party's session; see the module docstring."""

    def __init__(self, sid: bytes, path=None) -> None:
        if not isinstance(sid, (bytes, bytearray)):
            raise ValueError("sid must be bytes")
        self.sid = bytes(sid)
        self.path = None if path is None else os.fspath(path)
        self.state = ACTIVE
        self.starts = 1                           # processes that have used the state file
        self._next = {kind: 0 for kind in KINDS}
        self._floor = {kind: 0 for kind in KINDS}     # identifiers below belong to an earlier process
        self._reserved = {kind: 0 for kind in KINDS}  # bound stored on disk
        self._keys: dict = {}
        self._gates: dict = {}
        self._lock = threading.RLock()
        if self.path is not None:
            self._load()

    # -- persistence ------------------------------------------------------------ #

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._store()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data["format"] != FORMAT:
                raise ValueError("unknown format")
            sid = bytes.fromhex(data["sid"])
            reserved = {kind: int(data["reserved"][kind]) for kind in KINDS}
            state = data["state"]
            starts = int(data["starts"])
            if state not in (ACTIVE, ABORTED) or min(reserved.values()) < 0 or starts < 1:
                raise ValueError("invalid content")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"state file {self.path!r} is unreadable; refusing to reset the "
                             f"counters ({exc})") from None
        if sid != self.sid:
            raise ValueError(f"state file {self.path!r} belongs to another session")
        self._next = dict(reserved)
        self._floor = dict(reserved)
        self._reserved = dict(reserved)
        self.starts = starts + 1
        self.state = state
        self._store()

    def _store(self) -> None:
        if self.path is None:
            return
        data = {"format": FORMAT, "sid": self.sid.hex(), "state": self.state,
                "reserved": dict(self._reserved), "starts": self.starts}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # -- session state ------------------------------------------------------------ #

    def require_active(self) -> None:
        if self.state != ACTIVE:
            raise ProtocolAbort("session aborted")

    def abort(self) -> None:
        """ACTIVE -> ABORTED (absorbing); the key and gate material is dropped."""
        with self._lock:
            already = self.state == ABORTED
            self.state = ABORTED
            for rec in self._keys.values():
                rec.key = None
            for rec in self._gates.values():
                rec.material = None
                rec.evaluator = None
            if not already:
                try:
                    self._store()
                except OSError:
                    pass                      # the in-memory state is what protects this process

    # -- identifiers -------------------------------------------------------------- #

    def _issue(self, kind: str, requested, count: int = 1) -> int:
        """Reserve ``count`` consecutive identifiers; returns the first one."""
        if count < 1:
            raise ValueError("count must be positive")
        self.require_active()
        nxt = self._next[kind]
        if requested is None:
            first = nxt
        else:
            if isinstance(requested, bool) or int(requested) != requested or int(requested) < 0:
                raise ValueError(f"{kind} must be a non-negative integer")
            first = int(requested)
            if first < nxt:
                where = ("issued by an earlier process" if first < self._floor[kind]
                         else "already issued")
                raise StateError(f"{kind} {first} is not monotone ({where}); next is {nxt}")
        end = first + count
        if end > self._reserved[kind]:
            self._reserved[kind] = end + _BLOCK[kind]
            self._store()                     # on disk BEFORE any identifier of the block is used
        self._next[kind] = end
        return first

    def next_qid(self) -> int:
        """A fresh query identifier."""
        with self._lock:
            return self._issue("qid", None)

    def counters(self) -> dict:
        """Next identifier per kind, the floors of this process and the stored bounds (public)."""
        with self._lock:
            return {"next": dict(self._next), "floor": dict(self._floor),
                    "reserved": dict(self._reserved), "starts": self.starts, "state": self.state}

    # -- keys ---------------------------------------------------------------------- #

    def begin_key(self, kid=None, count: int = 1) -> int:
        """Create ``count`` key records in state GENERATING; returns the first identifier."""
        with self._lock:
            first = self._issue("kid", kid, count)
            for i in range(first, first + count):
                self._keys[i] = KeyRecord(i)
            return first

    def certify_key(self, kid: int, key, **info) -> KeyRecord:
        """GENERATING -> CERTIFIED: called when Pi_Gen has returned (its output check passed)."""
        with self._lock:
            self.require_active()
            rec = self._keys.get(kid)
            if rec is None or rec.state != GENERATING:
                raise StateError(f"key {kid} is not being generated")
            rec.key, rec.info, rec.state = key, dict(info), CERTIFIED
            return rec

    def key(self, kid: int) -> KeyRecord:
        """The record of a CERTIFIED key."""
        with self._lock:
            self.require_active()
            rec = self._keys.get(kid)
            if rec is None:
                raise StateError(f"unknown key {kid}")
            if rec.state != CERTIFIED:
                raise StateError(f"key {kid} is not certified")
            return rec

    # -- gates ----------------------------------------------------------------------- #

    def begin_gate(self, gid=None, count: int = 1) -> int:
        """Create ``count`` gate records in state GENERATING; returns the first identifier."""
        with self._lock:
            first = self._issue("gid", gid, count)
            for i in range(first, first + count):
                self._gates[i] = GateRecord(i)
            return first

    def certify_gate(self, gid: int, material, spec=None, evaluator=None) -> GateRecord:
        """GENERATING -> CERTIFIED."""
        with self._lock:
            self.require_active()
            rec = self._gates.get(gid)
            if rec is None or rec.state != GENERATING:
                raise StateError(f"gate {gid} is not being generated")
            rec.material, rec.spec, rec.evaluator, rec.state = material, spec, evaluator, CERTIFIED
            return rec

    def gate(self, gid: int) -> GateRecord:
        """The record of a gate in any state (no transition)."""
        with self._lock:
            self.require_active()
            rec = self._gates.get(gid)
            if rec is None:
                raise StateError(f"unknown gate {gid}")
            return rec

    def use_gate(self, gid: int) -> GateRecord:
        """CERTIFIED -> USED, exactly once; call it before the masked opening is sent."""
        with self._lock:
            rec = self.gate(gid)
            if rec.state == USED:
                raise GateUsedError(f"gate {gid}: the offset has already masked an input")
            if rec.state != CERTIFIED:
                raise StateError(f"gate {gid} is not certified")
            rec.state = USED
            return rec

    def set_masked(self, gid: int, x_hat: int, mode: str, opened=None) -> GateRecord:
        """Record the public result of a completed ``Mask``."""
        with self._lock:
            rec = self.gate(gid)
            if rec.state != USED or rec.x_hat is not None:
                raise StateError(f"gate {gid}: no Mask in progress")
            rec.x_hat, rec.mode, rec.opened = int(x_hat), mode, opened
            return rec

    def masked_gate(self, gid: int) -> GateRecord:
        """The record of a gate whose ``Mask`` has completed (for ``Apply``)."""
        with self._lock:
            rec = self.gate(gid)
            if rec.state != USED or rec.x_hat is None:
                raise StateError(f"gate {gid} has not been masked")
            return rec

    def summary(self) -> dict:
        """Number of records per state (public)."""
        with self._lock:
            keys = {GENERATING: 0, CERTIFIED: 0}
            gates = {GENERATING: 0, CERTIFIED: 0, USED: 0}
            for rec in self._keys.values():
                keys[rec.state] += 1
            for rec in self._gates.values():
                gates[rec.state] += 1
            return {"state": self.state, "keys": keys, "gates": gates}
