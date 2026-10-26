"""Party-local channel endpoints with byte, round and phase accounting.

Each party owns one ``Channel`` endpoint and uses it from one thread:

  * ``send(data)``       one-way message to the peer;
  * ``recv()``           next message from the peer (blocking);
  * ``exchange(data)``   both parties call it at the same protocol point; each
                         sends its message and returns the peer's.  It is ONE
                         round.  The peer must also call ``exchange`` there
                         (not ``send`` + ``recv``), or the round counts differ.

Accounting (attributes and ``stats()``):

  * ``bytes_sent`` / ``bytes_recv``   payload bytes (the TCP transport adds 4
    bytes of length framing per message; ``stats()['frame_overhead']``);
  * ``msgs_sent`` / ``msgs_recv``     messages, an exchange counting one each way;
  * ``rounds``   = number of ``exchange`` calls
                 + number of maximal runs of consecutive ``send`` calls
                 + number of maximal runs of consecutive ``recv`` calls.
    A run of sends at one endpoint is a run of receives at the other, so both
    endpoints report the same number.  ``mark`` does not interrupt a run.
  * ``mark(phase)`` starts (or resumes) a named accounting phase; every
    counter and the wall-clock time are also kept per phase and the phases
    add up to the totals.  Activity before the first ``mark`` is booked under
    ``DEFAULT_PHASE``, which is reported only if it saw traffic (or if no
    phase was ever marked).  The clock stops when the endpoint is closed.

Network emulation (application level, inside the channel, as stated in
DESIGN.md 9): with ``delay_ms`` a message becomes available to the receiver
``delay_ms`` after it was sent, so every flight is delayed by one one-way
latency and a receiver that is still computing when the flight arrives does
not wait again.  With ``mbps`` the inbound link additionally delivers
messages one after another at that rate (``8 * len / (mbps * 10^6)`` seconds
each).  Send timestamps are taken when the message is queued (loopback) or
when it has been read from the socket (TCP).

Closing an endpoint makes the peer's ``recv`` raise
``ProtocolAbort("peer closed")`` once the messages sent before the close have
been consumed; ``run_two_party`` relies on this to unblock the other party
when one side aborts.

Transports: ``loopback_pair`` (two in-process endpoints joined by thread-safe
queues), ``TcpChannel`` (4-byte little-endian length framing, TCP_NODELAY, a
background reader thread so that two simultaneous large sends cannot
deadlock) and ``tcp_pair`` (two connected TCP endpoints on 127.0.0.1).
"""

from __future__ import annotations

import os
import queue
import socket
import struct
import threading
import time

from .hashes import ProtocolAbort

__all__ = [
    "Channel", "TcpChannel", "ProtocolAbort", "loopback_pair", "tcp_pair", "run_two_party",
    "DEFAULT_PHASE", "MAX_MESSAGE_BYTES",
]

DEFAULT_PHASE = "(unmarked)"
MAX_MESSAGE_BYTES = (1 << 32) - 1  # limit of the 4-byte length prefix, enforced on every transport

_COUNTERS = ("bytes_sent", "bytes_recv", "msgs_sent", "msgs_recv", "rounds")
_CLOSED = object()  # inbox sentinel: the peer has closed its endpoint
_POLL_S = 0.25  # blocking waits are sliced so that KeyboardInterrupt is served
_COALESCE = 1 << 16  # payload bytes sent in the same segment burst as the length prefix


def _new_phase() -> dict:
    rec = {name: 0 for name in _COUNTERS}
    rec["time_s"] = 0.0
    return rec


class Channel:
    """One endpoint of a two-party channel (accounting and emulation; no transport)."""

    transport = "abstract"
    frame_overhead = 0

    def __init__(self, party: int, *, delay_ms: float = 0.0, mbps: float | None = None,
                 timeout: float | None = None) -> None:
        if party not in (0, 1):
            raise ValueError("party must be 0 or 1")
        if delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if mbps is not None and mbps <= 0:
            raise ValueError("mbps must be positive")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        self.party = party
        self.delay_ms = float(delay_ms)
        self.mbps = None if mbps is None else float(mbps)
        self.timeout = timeout  # seconds one recv may block; None = no limit
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.msgs_sent = 0
        self.msgs_recv = 0
        self.rounds = 0
        self._last_op = None  # "send", "recv" or "exchange"
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = False
        self._peer_closed = False
        self._link_free = 0.0  # time at which the emulated inbound link is idle again
        self._emulated_wait = 0.0
        self._stop_time: float | None = None
        self._phases: dict[str, dict] = {DEFAULT_PHASE: _new_phase()}
        self._cur = self._phases[DEFAULT_PHASE]
        self._phase_name = DEFAULT_PHASE
        self._phase_start = time.perf_counter()

    # -- transport hooks ---------------------------------------------------- #

    def _transmit(self, payload: bytes) -> None:
        raise NotImplementedError

    def _shutdown(self) -> None:
        raise NotImplementedError

    # -- public interface --------------------------------------------------- #

    def send(self, data: bytes) -> None:
        """Send one message to the peer (does not wait for it to be read)."""
        self._tx(self._payload(data))
        if self._last_op != "send":
            self._last_op = "send"
            self._add("rounds", 1)

    def recv(self) -> bytes:
        """Return the next message from the peer; ``ProtocolAbort`` if it closed."""
        payload = self._rx()
        if self._last_op != "recv":
            self._last_op = "recv"
            self._add("rounds", 1)
        return payload

    def exchange(self, data: bytes) -> bytes:
        """Send ``data`` and return the peer's message of the same round (one round)."""
        self._tx(self._payload(data))
        payload = self._rx()
        self._last_op = "exchange"
        self._add("rounds", 1)
        return payload

    def mark(self, phase: str) -> None:
        """Start (or resume) the accounting phase ``phase``."""
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a non-empty string")
        now = self._now()
        self._cur["time_s"] += now - self._phase_start
        self._phase_start = now
        self._phase_name = phase
        self._cur = self._phases.setdefault(phase, _new_phase())

    @property
    def phase(self) -> str:
        """Name of the current accounting phase."""
        return self._phase_name

    @property
    def closed(self) -> bool:
        return self._closed

    def stats(self) -> dict:
        """Snapshot of the counters.

        Keys: ``party``, ``transport``, ``bytes_sent``, ``bytes_recv``,
        ``msgs_sent``, ``msgs_recv``, ``rounds``, ``time_s``,
        ``emulated_wait_s``, ``delay_ms``, ``mbps``, ``frame_overhead`` and
        ``phases``: an insertion-ordered dict name -> {the five counters,
        ``time_s``}.  Every total equals the sum over the reported phases.
        """
        now = self._now()
        phases = {}
        for name, rec in self._phases.items():
            entry = dict(rec)
            if rec is self._cur:
                entry["time_s"] += now - self._phase_start
            phases[name] = entry
        default = phases[DEFAULT_PHASE]
        if len(phases) > 1 and not any(default[name] for name in _COUNTERS):
            del phases[DEFAULT_PHASE]
        out = {"party": self.party, "transport": self.transport}
        for name in _COUNTERS:
            out[name] = getattr(self, name)
        out["time_s"] = sum(entry["time_s"] for entry in phases.values())
        out["emulated_wait_s"] = self._emulated_wait
        out["delay_ms"] = self.delay_ms
        out["mbps"] = self.mbps
        out["frame_overhead"] = self.frame_overhead
        out["phases"] = phases
        return out

    def close(self) -> None:
        """Close the endpoint: stops the clock and tells the peer (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._stop_time = time.perf_counter()
        self._shutdown()

    def __enter__(self) -> "Channel":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- internals ---------------------------------------------------------- #

    def _now(self) -> float:
        return time.perf_counter() if self._stop_time is None else self._stop_time

    def _add(self, name: str, amount: int) -> None:
        setattr(self, name, getattr(self, name) + amount)
        self._cur[name] += amount

    @staticmethod
    def _payload(data) -> bytes:
        if isinstance(data, str):
            raise TypeError("messages are bytes, not str")
        payload = data if isinstance(data, bytes) else bytes(data)
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("message longer than 2^32 - 1 bytes")
        return payload

    def _tx(self, payload: bytes) -> None:
        if self._closed:
            raise ProtocolAbort("channel closed")
        self._transmit(payload)
        self._add("bytes_sent", len(payload))
        self._add("msgs_sent", 1)

    def _rx(self) -> bytes:
        if self._closed:
            raise ProtocolAbort("channel closed")
        if self._peer_closed:
            raise ProtocolAbort("peer closed")
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            try:
                if remaining is not None and remaining <= 0:
                    item = self._inbox.get_nowait()
                else:
                    item = self._inbox.get(
                        timeout=_POLL_S if remaining is None else min(_POLL_S, remaining))
                break
            except queue.Empty:
                if remaining is not None and remaining <= 0:
                    raise ProtocolAbort("timeout") from None
        if item is _CLOSED:
            self._peer_closed = True
            raise ProtocolAbort("peer closed")
        stamp, payload = item
        if self.delay_ms or self.mbps is not None:
            ready = stamp + self.delay_ms / 1000.0
            if self.mbps is not None:
                ready = max(ready, self._link_free) + 8.0 * len(payload) / (self.mbps * 1e6)
                self._link_free = ready
            wait = ready - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
                self._emulated_wait += wait
        self._add("bytes_recv", len(payload))
        self._add("msgs_recv", 1)
        return payload


class _LoopbackChannel(Channel):
    """In-process endpoint: messages go straight into the peer's inbox queue."""

    transport = "loopback"

    def __init__(self, party: int, **kwargs) -> None:
        super().__init__(party, **kwargs)
        self._peer: "_LoopbackChannel | None" = None

    def _transmit(self, payload: bytes) -> None:
        peer = self._peer
        if peer._closed:
            raise ProtocolAbort("peer closed")
        peer._inbox.put((time.perf_counter(), payload))

    def _shutdown(self) -> None:
        self._peer._inbox.put(_CLOSED)


def loopback_pair(delay_ms: float = 0, mbps: float | None = None, *,
                  timeout: float | None = None) -> tuple[Channel, Channel]:
    """Two connected in-process endpoints (party 0, party 1)."""
    c0 = _LoopbackChannel(0, delay_ms=delay_ms, mbps=mbps, timeout=timeout)
    c1 = _LoopbackChannel(1, delay_ms=delay_ms, mbps=mbps, timeout=timeout)
    c0._peer, c1._peer = c1, c0
    return c0, c1


def _read_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes; None if the stream ends first."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class TcpChannel(Channel):
    """Endpoint over a connected TCP socket.

    Wire format: every message is a 4-byte little-endian length followed by
    the payload.  TCP_NODELAY is set.  A daemon thread reads frames into the
    inbox, so ``send`` never blocks on a peer that is itself sending.
    """

    transport = "tcp"
    frame_overhead = 4

    def __init__(self, sock: socket.socket, party: int, *, delay_ms: float = 0.0,
                 mbps: float | None = None, timeout: float | None = None) -> None:
        super().__init__(party, delay_ms=delay_ms, mbps=mbps, timeout=timeout)
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name=f"couplet-tcp-reader-{party}")
        self._reader.start()

    # -- construction ------------------------------------------------------- #

    @staticmethod
    def bind(port: int = 0, host: str = "127.0.0.1") -> tuple[socket.socket, int]:
        """Open a listening socket; returns (socket, actual port) (port 0 = any free port)."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name != "nt":
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
        except OSError:
            server.close()
            raise
        return server, server.getsockname()[1]

    @classmethod
    def accept(cls, server: socket.socket, *, party: int = 0, delay_ms: float = 0.0,
               mbps: float | None = None, timeout: float | None = None,
               accept_timeout: float | None = None) -> "TcpChannel":
        """Accept one connection on a socket from ``bind`` and close the listener.

        Waits ``accept_timeout`` seconds (None = forever) and raises
        ``ProtocolAbort`` if nobody connects.
        """
        deadline = None if accept_timeout is None else time.monotonic() + accept_timeout
        server.settimeout(_POLL_S)
        try:
            while True:
                try:
                    sock, _ = server.accept()
                    break
                except TimeoutError:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise ProtocolAbort("no incoming connection") from None
                except OSError as exc:
                    raise ProtocolAbort("no incoming connection") from exc
        finally:
            server.close()
        return cls(sock, party, delay_ms=delay_ms, mbps=mbps, timeout=timeout)

    @classmethod
    def listen(cls, port: int, host: str = "127.0.0.1", *, party: int = 0,
               delay_ms: float = 0.0, mbps: float | None = None,
               timeout: float | None = None,
               accept_timeout: float | None = None) -> "TcpChannel":
        """Listen on ``port``, wait for the peer to connect, return the endpoint (party 0)."""
        server, _ = cls.bind(port, host)
        return cls.accept(server, party=party, delay_ms=delay_ms, mbps=mbps, timeout=timeout,
                          accept_timeout=accept_timeout)

    @classmethod
    def connect(cls, host: str, port: int, *, party: int = 1, delay_ms: float = 0.0,
                mbps: float | None = None, timeout: float | None = None,
                connect_timeout: float = 10.0) -> "TcpChannel":
        """Connect to a listening peer, retrying until ``connect_timeout`` (party 1)."""
        deadline = time.monotonic() + connect_timeout
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect((host, port))
                break
            except OSError as exc:
                sock.close()
                if time.monotonic() >= deadline:
                    raise ProtocolAbort(f"cannot connect to {host}:{port}") from exc
                time.sleep(0.05)
        return cls(sock, party, delay_ms=delay_ms, mbps=mbps, timeout=timeout)

    # -- transport ---------------------------------------------------------- #

    def _transmit(self, payload: bytes) -> None:
        header = struct.pack("<I", len(payload))
        try:
            if len(payload) <= _COALESCE:
                self._sock.sendall(header + payload)
            else:
                self._sock.sendall(header + payload[:_COALESCE])
                self._sock.sendall(memoryview(payload)[_COALESCE:])
        except OSError as exc:
            raise ProtocolAbort("peer closed") from exc

    def _read_loop(self) -> None:
        sock = self._sock
        try:
            while True:
                header = _read_exact(sock, 4)
                if header is None:
                    break
                body = _read_exact(sock, struct.unpack("<I", header)[0])
                if body is None:
                    break
                if not self._closed:
                    self._inbox.put((time.perf_counter(), body))
        except OSError:
            pass
        finally:
            self._inbox.put(_CLOSED)
            try:
                sock.close()
            except OSError:
                pass

    def _shutdown(self) -> None:
        # Half-close: everything sent so far is delivered, then the peer's
        # reader sees end-of-stream.  The reader thread closes the socket once
        # the peer has closed its side as well.
        try:
            self._sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def tcp_pair(port: int = 0, delay_ms: float = 0, mbps: float | None = None, *,
             timeout: float | None = None) -> tuple[Channel, Channel]:
    """Two TCP endpoints connected over 127.0.0.1 (party 0 listens, party 1 connects)."""
    server, actual = TcpChannel.bind(port)
    box: dict = {}

    def _accept() -> None:
        try:
            box["chan"] = TcpChannel.accept(server, party=0, delay_ms=delay_ms, mbps=mbps,
                                            timeout=timeout, accept_timeout=30.0)
        except BaseException as exc:  # reported to the caller below
            box["error"] = exc

    acceptor = threading.Thread(target=_accept, daemon=True, name="couplet-tcp-accept")
    acceptor.start()
    try:
        c1 = TcpChannel.connect("127.0.0.1", actual, party=1, delay_ms=delay_ms, mbps=mbps,
                                timeout=timeout)
    except BaseException:
        server.close()
        acceptor.join()
        raise
    acceptor.join()
    if "error" in box:
        c1.close()
        raise box["error"]
    return box["chan"], c1


def run_two_party(fn0, fn1, *, delay_ms: float = 0, mbps: float | None = None,
                  tcp: bool = False, port: int = 0, timeout: float | None = None):
    """Run ``fn0(channel_0)`` and ``fn1(channel_1)`` in two threads.

    Returns ``(result0, result1, stats0, stats1)``.  Each endpoint is closed
    when its function returns or raises, which unblocks a peer that is still
    waiting (its ``recv`` raises ``ProtocolAbort("peer closed")``).  If either
    function raises, the exception raised FIRST is re-raised here (so a real
    abort is reported rather than the "peer closed" it causes on the other
    side).  ``tcp=True`` uses real sockets on 127.0.0.1 (``port`` 0 = any free
    port); ``timeout`` bounds every single ``recv`` in seconds.
    """
    if tcp:
        chans = tcp_pair(port, delay_ms, mbps, timeout=timeout)
    else:
        chans = loopback_pair(delay_ms, mbps, timeout=timeout)
    results = [None, None]
    stats = [None, None]
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    def _run(party: int, fn) -> None:
        chan = chans[party]
        try:
            results[party] = fn(chan)
        except BaseException as exc:  # re-raised in the calling thread
            with lock:
                errors.append((party, exc))
        finally:
            chan.close()
            stats[party] = chan.stats()

    threads = [threading.Thread(target=_run, args=(b, fn), daemon=True,
                                name=f"couplet-party-{b}")
               for b, fn in enumerate((fn0, fn1))]
    for th in threads:
        th.start()
    for th in threads:
        while th.is_alive():
            th.join(_POLL_S)
    if errors:
        party, exc = errors[0]
        exc.add_note(f"raised by party {party} inside run_two_party")
        raise exc
    return results[0], results[1], stats[0], stats[1]
