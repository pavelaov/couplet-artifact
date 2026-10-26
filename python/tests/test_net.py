"""Tests for couplet.net: channels, accounting, abort propagation, TCP, emulation.

  * round counting: ping-pong, exchange and mixed patterns with the exact
    expected number of rounds at both endpoints;
  * byte and message counters; phase accounting adds up to the totals;
  * abort propagation: a party that raises unblocks its peer, the first
    exception is the one re-raised, data sent before a close is delivered;
  * TCP on 127.0.0.1: wire format, classmethods, large simultaneous sends,
    and a run between two operating-system processes;
  * delay and bandwidth emulation within a timing tolerance.

Run as a script from artifact/python:   python tests/test_net.py
The script prints the number of checks and exits non-zero on any failure.
The ``test_*`` functions are also collectable by pytest.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import net  # noqa: E402
from couplet.hashes import ProtocolAbort  # noqa: E402
from couplet.net import (DEFAULT_PHASE, Channel, TcpChannel, loopback_pair,  # noqa: E402
                          run_two_party, tcp_pair)

CHECKS = 0
FAILURES: list[str] = []
COUNTERS = ("bytes_sent", "bytes_recv", "msgs_sent", "msgs_recv", "rounds")
TIMEOUT = 30.0  # per-recv bound so that a broken test fails instead of hanging

# Timing tolerances for the emulation tests: sleeps never return noticeably
# early, but a loaded machine may add scheduling latency.
LOW = 0.90
SLACK_S = 0.25


def check(cond, what: str) -> None:
    """Count one check; record (and print) it if it fails."""
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(what)
        print("FAIL:", what)


def check_time(elapsed: float, expected: float, what: str) -> None:
    check(expected * LOW <= elapsed <= expected + SLACK_S,
          f"{what}: {elapsed * 1000:.1f} ms, expected about {expected * 1000:.0f} ms")


def _finish(start_failures: int) -> None:
    assert len(FAILURES) == start_failures, FAILURES[start_failures:]


def abort_text(fn, *args) -> str | None:
    """Message of the ProtocolAbort raised by fn(*args), or None if it does not raise one."""
    try:
        fn(*args)
    except ProtocolAbort as exc:
        return str(exc)
    return None


def pair(tcp: bool, **kwargs):
    return tcp_pair(**kwargs) if tcp else loopback_pair(**kwargs)


# --------------------------------------------------------------------------- #
# Round, byte and message counting
# --------------------------------------------------------------------------- #

def _ping_pong(k: int, size: int = 10):
    def p0(chan):
        got = []
        for i in range(k):
            chan.send(bytes([i]) * size)
            got.append(chan.recv())
        return got

    def p1(chan):
        got = []
        for i in range(k):
            got.append(chan.recv())
            chan.send(bytes([100 + i]) * (size + 1))
        return got

    return p0, p1


def test_round_counting() -> None:
    start = len(FAILURES)
    for tcp in (False, True):
        tag = "tcp" if tcp else "loopback"

        k = 5
        r0, r1, s0, s1 = run_two_party(*_ping_pong(k), tcp=tcp, timeout=TIMEOUT)
        check(r1 == [bytes([i]) * 10 for i in range(k)], f"{tag} ping-pong: P1 received the pings")
        check(r0 == [bytes([100 + i]) * 11 for i in range(k)], f"{tag} ping-pong: P0 got the pongs")
        check(s0["rounds"] == 2 * k and s1["rounds"] == 2 * k, f"{tag} ping-pong: {2 * k} rounds")
        check(s0["bytes_sent"] == 10 * k and s0["bytes_recv"] == 11 * k, f"{tag} ping-pong: P0 bytes")
        check(s1["bytes_sent"] == 11 * k and s1["bytes_recv"] == 10 * k, f"{tag} ping-pong: P1 bytes")
        check(s0["msgs_sent"] == k and s1["msgs_sent"] == k and s0["msgs_recv"] == k
              and s1["msgs_recv"] == k, f"{tag} ping-pong: message counts")
        check(s0["party"] == 0 and s1["party"] == 1, f"{tag}: party indices")
        check(s0["transport"] == tag and s0["frame_overhead"] == (4 if tcp else 0),
              f"{tag}: transport metadata")

        k = 7
        r0, r1, s0, s1 = run_two_party(
            lambda c: [c.exchange(b"a" * (i + 1)) for i in range(k)],
            lambda c: [c.exchange(b"b" * (2 * i)) for i in range(k)], tcp=tcp, timeout=TIMEOUT)
        check(r0 == [b"b" * (2 * i) for i in range(k)] and r1 == [b"a" * (i + 1) for i in range(k)],
              f"{tag} exchange: contents (including an empty message)")
        check(s0["rounds"] == k and s1["rounds"] == k, f"{tag} exchange: {k} rounds for {k} exchanges")
        check(s0["bytes_sent"] == k * (k + 1) // 2 == s1["bytes_recv"], f"{tag} exchange: bytes 0->1")
        check(s1["bytes_sent"] == k * (k - 1) == s0["bytes_recv"], f"{tag} exchange: bytes 1->0")
        check(s0["msgs_sent"] == k and s0["msgs_recv"] == k, f"{tag} exchange: message counts")

        # Mixed pattern, 8 rounds:
        #   1: P0 sends a, b      2: P1 sends x        3, 4: two exchanges
        #   5: P1 sends y, z      6: P0 sends c        7: exchange     8: P0 sends d
        def mixed0(chan):
            out = []
            chan.send(b"a")
            chan.send(b"bb")
            out.append(chan.recv())
            out.append(chan.exchange(b"e0"))
            out.append(chan.exchange(b"e1"))
            out.append(chan.recv())
            out.append(chan.recv())
            chan.send(b"ccc")
            out.append(chan.exchange(b"e2"))
            chan.send(b"dddd")
            return out, chan.rounds

        def mixed1(chan):
            out = [chan.recv(), chan.recv()]
            chan.send(b"x")
            out.append(chan.exchange(b"f0"))
            out.append(chan.exchange(b"f1"))
            chan.send(b"y")
            chan.send(b"z")
            out.append(chan.recv())
            out.append(chan.exchange(b"f2"))
            out.append(chan.recv())
            return out, chan.rounds

        (o0, n0), (o1, n1), s0, s1 = run_two_party(mixed0, mixed1, tcp=tcp, timeout=TIMEOUT)
        check(o0 == [b"x", b"f0", b"f1", b"y", b"z", b"f2"], f"{tag} mixed: P0 view")
        check(o1 == [b"a", b"bb", b"e0", b"e1", b"ccc", b"e2", b"dddd"], f"{tag} mixed: P1 view")
        check(n0 == 8 and n1 == 8 and s0["rounds"] == 8 and s1["rounds"] == 8,
              f"{tag} mixed: 8 rounds at both endpoints")
        check(s0["msgs_sent"] == 7 and s1["msgs_sent"] == 6, f"{tag} mixed: messages sent")
        check(s0["msgs_recv"] == 6 and s1["msgs_recv"] == 7, f"{tag} mixed: messages received")
        check(s0["bytes_sent"] == s1["bytes_recv"] == 16 and s1["bytes_sent"] == s0["bytes_recv"] == 9,
              f"{tag} mixed: bytes")

    # Step-by-step on a loopback pair: the counter after every operation.
    c0, c1 = loopback_pair(timeout=TIMEOUT)
    check(isinstance(c0, Channel) and isinstance(c1, Channel), "loopback_pair returns Channels")
    check((c0.party, c1.party) == (0, 1), "loopback_pair party indices")
    check(c0.rounds == 0 and c0.bytes_sent == 0 and c0.msgs_sent == 0, "counters start at zero")
    c0.send(b"1")
    check(c0.rounds == 1, "first send opens a round")
    c0.send(b"22")
    c0.send(b"333")
    check(c0.rounds == 1, "consecutive sends are one round")
    check(c1.recv() == b"1" and c1.rounds == 1, "first recv opens a round")
    check(c1.recv() == b"22" and c1.recv() == b"333" and c1.rounds == 1,
          "consecutive recvs are one round")
    c1.send(b"4")
    check(c1.rounds == 2 and c0.recv() == b"4" and c0.rounds == 2, "direction change: new round")
    c0.send(b"5")
    check(c0.rounds == 3 and c1.recv() == b"5" and c1.rounds == 3, "and back: third round")
    check(c0.msgs_sent == 4 and c0.msgs_recv == 1 and c1.msgs_sent == 1 and c1.msgs_recv == 4,
          "message counters after three rounds")
    check(c0.bytes_sent == 7 and c0.bytes_recv == 1 and c1.bytes_sent == 1 and c1.bytes_recv == 7,
          "byte counters after three rounds")
    c0.close()
    c1.close()

    c0, c1 = loopback_pair(timeout=TIMEOUT)
    c0.send(b"s")
    t = threading.Thread(target=lambda: c1.exchange(b"E1"))
    c1.recv()
    t.start()
    got = c0.exchange(b"E0")
    t.join()
    c0.send(b"s")
    c1.recv()
    check(got == b"E1" and c0.rounds == 3 and c1.rounds == 3, "send, exchange, send = 3 rounds")
    c0.mark("next")
    c0.send(b"t")
    c1.mark("next")
    c1.recv()
    check(c0.rounds == 3 and c1.rounds == 3, "mark does not interrupt a run of sends")
    check(c0.bytes_sent == 5 and c1.bytes_recv == 5 and c0.msgs_sent == 4 and c1.msgs_recv == 4,
          "byte and message counters")
    c0.close()
    c1.close()

    c0, c1 = loopback_pair()
    try:
        c0.send("text")
    except TypeError:
        check(True, "send rejects str")
    else:
        check(False, "send rejects str")
    c0.send(bytearray(b"ab"))
    c0.send(memoryview(b"cd"))
    check(c1.recv() == b"ab" and c1.recv() == b"cd", "bytes-like messages are accepted")
    buf = bytearray(b"mutable")
    c0.send(buf)
    buf[0] = 0
    check(c1.recv() == b"mutable", "a sent message is a snapshot")
    check(net.MAX_MESSAGE_BYTES == 2**32 - 1, "message size limit of the 4-byte length prefix")
    c0.close()
    c1.close()
    _finish(start)


# --------------------------------------------------------------------------- #
# Phase accounting
# --------------------------------------------------------------------------- #

def _check_phase_sums(stats: dict, what: str) -> None:
    phases = stats["phases"]
    for name in COUNTERS:
        check(sum(p[name] for p in phases.values()) == stats[name], f"{what}: phases sum to {name}")
    check(abs(sum(p["time_s"] for p in phases.values()) - stats["time_s"]) < 1e-9,
          f"{what}: phase times sum to the total time")
    check(all(set(p) == set(COUNTERS) | {"time_s"} for p in phases.values()),
          f"{what}: phase record keys")


def test_phase_accounting() -> None:
    start = len(FAILURES)
    for tcp in (False, True):
        tag = "tcp" if tcp else "loopback"

        def p0(chan):
            chan.mark("setup")
            chan.send(b"s" * 100)
            chan.recv()
            chan.mark("online")
            for _ in range(3):
                chan.exchange(b"o" * 7)
            time.sleep(0.06)
            chan.mark("check")
            chan.send(b"c" * 5)
            chan.send(b"c" * 6)
            chan.mark("setup")  # resume an earlier phase
            chan.exchange(b"again")
            return chan.phase

        def p1(chan):
            chan.mark("setup")
            chan.recv()
            chan.send(b"t" * 50)
            chan.mark("online")
            for _ in range(3):
                chan.exchange(b"p" * 9)
            chan.mark("check")
            chan.recv()
            chan.recv()
            chan.mark("setup")
            chan.exchange(b"again!")
            return chan.phase

        r0, r1, s0, s1 = run_two_party(p0, p1, tcp=tcp, timeout=TIMEOUT)
        check(r0 == "setup" and r1 == "setup", f"{tag}: current phase name")
        check(list(s0["phases"]) == ["setup", "online", "check"] == list(s1["phases"]),
              f"{tag}: phases in first-use order, empty default phase omitted")
        _check_phase_sums(s0, f"{tag} P0")
        _check_phase_sums(s1, f"{tag} P1")
        ph0, ph1 = s0["phases"], s1["phases"]
        check(ph0["setup"]["bytes_sent"] == 105 and ph0["setup"]["bytes_recv"] == 56,
              f"{tag}: setup bytes (both visits)")
        check(ph0["online"]["bytes_sent"] == 21 and ph0["online"]["bytes_recv"] == 27,
              f"{tag}: online bytes")
        check(ph0["check"]["bytes_sent"] == 11 and ph0["check"]["bytes_recv"] == 0,
              f"{tag}: check bytes")
        check(ph0["setup"]["rounds"] == 3 and ph0["online"]["rounds"] == 3
              and ph0["check"]["rounds"] == 1, f"{tag}: rounds per phase")
        check(s0["rounds"] == 7 and s1["rounds"] == 7, f"{tag}: 7 rounds in total")
        for name in ph0:
            check(ph0[name]["rounds"] == ph1[name]["rounds"], f"{tag}: equal rounds in '{name}'")
            check(ph0[name]["bytes_sent"] == ph1[name]["bytes_recv"]
                  and ph1[name]["bytes_sent"] == ph0[name]["bytes_recv"],
                  f"{tag}: bytes match across endpoints in '{name}'")
            check(ph0[name]["msgs_sent"] == ph1[name]["msgs_recv"]
                  and ph1[name]["msgs_sent"] == ph0[name]["msgs_recv"],
                  f"{tag}: messages match across endpoints in '{name}'")
        check(0.055 <= ph0["online"]["time_s"] <= 0.06 + SLACK_S, f"{tag}: phase wall-clock time")
        check(all(p["time_s"] >= 0 for p in ph0.values()) and s0["time_s"] >= 0.055,
              f"{tag}: times are non-negative")

    c0, c1 = loopback_pair(timeout=TIMEOUT)
    st = c0.stats()
    check(list(st["phases"]) == [DEFAULT_PHASE], "no mark: only the default phase")
    check(c0.phase == DEFAULT_PHASE, "default phase name")
    c0.send(b"early")
    c0.mark("late")
    c0.send(b"x")
    st = c0.stats()
    check(list(st["phases"]) == [DEFAULT_PHASE, "late"], "default phase kept when it saw traffic")
    check(st["phases"][DEFAULT_PHASE]["bytes_sent"] == 5 and st["phases"]["late"]["bytes_sent"] == 1,
          "traffic before the first mark is booked under the default phase")
    check(st["phases"][DEFAULT_PHASE]["rounds"] == 1 and st["phases"]["late"]["rounds"] == 0,
          "a run of sends is booked where it started")
    _check_phase_sums(st, "default + late")
    try:
        c0.mark("")
    except ValueError:
        check(True, "mark rejects an empty name")
    else:
        check(False, "mark rejects an empty name")
    time.sleep(0.02)
    c0.close()
    first = c0.stats()
    time.sleep(0.03)
    second = c0.stats()
    check(first == second and first["time_s"] >= 0.02, "the clock stops at close")
    check(c0.closed and not c1.closed, "closed property")
    c1.close()
    for key in ("party", "transport", "time_s", "emulated_wait_s", "delay_ms", "mbps",
                "frame_overhead", "phases") + COUNTERS:
        check(key in first, f"stats has key '{key}'")
    _finish(start)


# --------------------------------------------------------------------------- #
# Abort propagation and closing
# --------------------------------------------------------------------------- #

def test_abort_propagation() -> None:
    start = len(FAILURES)
    for tcp in (False, True):
        tag = "tcp" if tcp else "loopback"
        seen = {}

        def cheated(chan):
            chan.recv()
            raise ProtocolAbort("MAC check failed")

        def blocked(chan):
            chan.send(b"hello")
            try:
                chan.recv()  # never answered: must be unblocked by the peer's abort
            except ProtocolAbort as exc:
                seen["peer"] = str(exc)
                raise
            return "unreachable"

        for order in ((cheated, blocked), (blocked, cheated)):
            seen.clear()
            t0 = time.perf_counter()
            text = abort_text(lambda: run_two_party(*order, tcp=tcp, timeout=TIMEOUT))
            elapsed = time.perf_counter() - t0
            check(text == "MAC check failed", f"{tag}: the first exception is re-raised ({text})")
            check(seen.get("peer") == "peer closed", f"{tag}: the blocked party sees 'peer closed'")
            check(elapsed < 5.0, f"{tag}: abort propagates promptly ({elapsed:.2f} s)")

        def buggy(chan):
            chan.recv()
            raise KeyError("bug")

        try:
            run_two_party(buggy, blocked, tcp=tcp, timeout=TIMEOUT)
        except KeyError as exc:
            check("raised by party 0" in "".join(getattr(exc, "__notes__", [])),
                  f"{tag}: other exception types propagate with a party note")
        except Exception as exc:
            check(False, f"{tag}: expected KeyError, got {type(exc).__name__}")
        else:
            check(False, f"{tag}: expected KeyError")

        # A party that returns early also closes its endpoint.
        text = abort_text(lambda: run_two_party(lambda c: "done", lambda c: c.recv(),
                                                tcp=tcp, timeout=TIMEOUT))
        check(text == "peer closed", f"{tag}: early return unblocks the peer")

        r0, r1, s0, s1 = run_two_party(lambda c: c.exchange(b"a"), lambda c: c.exchange(b"b"),
                                       tcp=tcp, timeout=TIMEOUT)
        check((r0, r1) == (b"b", b"a"), f"{tag}: normal completion returns both results")

        # Data sent before a close is still delivered, then "peer closed" (sticky).
        c0, c1 = pair(tcp, timeout=TIMEOUT)
        c0.send(b"one")
        c0.send(b"two")
        c0.close()
        c0.close()  # idempotent
        check(c1.recv() == b"one" and c1.recv() == b"two", f"{tag}: queued data survives the close")
        check(abort_text(c1.recv) == "peer closed", f"{tag}: recv after the peer closed")
        check(abort_text(c1.recv) == "peer closed", f"{tag}: 'peer closed' is sticky")
        check(abort_text(c0.send, b"x") == "channel closed", f"{tag}: send on a closed endpoint")
        check(abort_text(c0.recv) == "channel closed", f"{tag}: recv on a closed endpoint")
        check(abort_text(c0.exchange, b"x") == "channel closed", f"{tag}: exchange when closed")
        if not tcp:
            check(abort_text(c1.send, b"x") == "peer closed", "loopback: send to a closed peer")
        c1.close()

        c0, c1 = pair(tcp, timeout=0.2)
        t0 = time.perf_counter()
        text = abort_text(c0.recv)
        elapsed = time.perf_counter() - t0
        check(text == "timeout", f"{tag}: recv timeout raises ProtocolAbort('timeout')")
        check(0.18 <= elapsed <= 0.2 + SLACK_S, f"{tag}: recv timeout duration ({elapsed:.3f} s)")
        c1.send(b"late")
        check(c0.recv() == b"late", f"{tag}: the endpoint still works after a timeout")
        with c0, c1:
            pass
        check(c0.closed and c1.closed, f"{tag}: context manager closes")

    check(issubclass(ProtocolAbort, Exception) and net.ProtocolAbort is ProtocolAbort,
          "net re-exports hashes.ProtocolAbort")
    for bad in ({"delay_ms": -1}, {"mbps": 0}, {"timeout": -1}):
        try:
            loopback_pair(**bad)
        except ValueError:
            check(True, f"loopback_pair rejects {bad}")
        else:
            check(False, f"loopback_pair rejects {bad}")
    _finish(start)


# --------------------------------------------------------------------------- #
# TCP specifics
# --------------------------------------------------------------------------- #

def _free_port() -> int:
    server, port = TcpChannel.bind(0)
    server.close()
    return port


def test_tcp() -> None:
    start = len(FAILURES)

    # listen / connect classmethods on a fixed port.
    port = _free_port()
    box = {}
    listener = threading.Thread(
        target=lambda: box.update(chan=TcpChannel.listen(port, timeout=TIMEOUT, accept_timeout=20)))
    listener.start()
    c1 = TcpChannel.connect("127.0.0.1", port, timeout=TIMEOUT)
    listener.join()
    c0 = box["chan"]
    check(isinstance(c0, TcpChannel) and isinstance(c0, Channel), "listen returns a TcpChannel")
    check((c0.party, c1.party) == (0, 1), "listen is party 0, connect is party 1")
    check(c0._sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0
          and c1._sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0, "TCP_NODELAY is set")
    c0.send(b"over tcp")
    c0.send(b"")
    check(c1.recv() == b"over tcp" and c1.recv() == b"", "TCP delivery, including an empty message")
    c1.send(b"reply")
    check(c0.recv() == b"reply" and c0.rounds == 2 and c1.rounds == 2, "TCP round count")
    c0.close()
    check(abort_text(c1.recv) == "peer closed", "TCP: close reaches the peer")
    c1.close()

    # Wire format: 4-byte little-endian length, then the payload.
    server, port = TcpChannel.bind(0)
    raw = socket.create_connection(("127.0.0.1", port))
    chan = TcpChannel.accept(server, timeout=TIMEOUT)
    raw.sendall(b"\x03\x00\x00\x00abc" + b"\x00\x00\x00\x00" + b"\x02\x01\x00\x00" + b"z" * 258)
    check(chan.recv() == b"abc" and chan.recv() == b"" and chan.recv() == b"z" * 258,
          "TCP framing on receive")
    chan.send(b"hello")
    chan.send(b"")
    raw.settimeout(10)
    data = b""
    while len(data) < 13:
        data += raw.recv(64)
    check(data == b"\x05\x00\x00\x00hello\x00\x00\x00\x00", "TCP framing on send")
    check(chan.bytes_sent == 5 and chan.bytes_recv == 261 and chan.msgs_sent == 2,
          "byte counters exclude the length prefix")
    raw.sendall(b"\x09\x00\x00\x00trunc")  # frame cut short by the close below
    raw.close()
    check(abort_text(chan.recv) == "peer closed", "TCP: truncated frame at end of stream")
    chan.close()

    # Large messages in both directions at once must not deadlock.
    size = 6 * 1024 * 1024
    blob0, blob1 = os.urandom(size), os.urandom(size + 17)
    r0, r1, s0, s1 = run_two_party(lambda c: hashlib.sha256(c.exchange(blob0)).digest(),
                                   lambda c: hashlib.sha256(c.exchange(blob1)).digest(),
                                   tcp=True, timeout=TIMEOUT)
    check(r0 == hashlib.sha256(blob1).digest() and r1 == hashlib.sha256(blob0).digest(),
          "TCP: simultaneous 6 MiB exchange")
    check(s0["bytes_sent"] == size and s0["bytes_recv"] == size + 17 and s0["rounds"] == 1,
          "TCP: large exchange accounting")

    # The same protocol gives the same counters on both transports.
    def proto0(chan):
        chan.mark("a")
        chan.send(b"1" * 1000)
        chan.send(b"2" * 70000)
        x = chan.recv()
        chan.mark("b")
        return x + chan.exchange(b"3" * 5)

    def proto1(chan):
        chan.mark("a")
        x = chan.recv() + chan.recv()
        chan.send(b"4" * 33)
        chan.mark("b")
        return x[:3] + chan.exchange(b"5" * 6)

    lo = run_two_party(proto0, proto1, timeout=TIMEOUT)
    tc = run_two_party(proto0, proto1, tcp=True, timeout=TIMEOUT)
    check(lo[0] == tc[0] and lo[1] == tc[1], "same results on loopback and TCP")
    for b in (0, 1):
        check(all(lo[2 + b][k] == tc[2 + b][k] for k in COUNTERS),
              f"same totals on loopback and TCP, party {b}")
        check(all(lo[2 + b]["phases"][p][k] == tc[2 + b]["phases"][p][k]
                  for p in ("a", "b") for k in COUNTERS),
              f"same per-phase counters on loopback and TCP, party {b}")

    check(abort_text(lambda: TcpChannel.connect("127.0.0.1", _free_port(), connect_timeout=0.0))
          is not None, "connect to a closed port raises ProtocolAbort")
    server, _ = TcpChannel.bind(0)
    check(abort_text(lambda: TcpChannel.accept(server, accept_timeout=0.3)) == "no incoming connection",
          "accept times out with ProtocolAbort")
    _finish(start)


# The peer of ``test_two_processes``: connects, runs its half of a small
# protocol, closes, prints its statistics as JSON and exits immediately.
_CHILD = r"""
import hashlib, json, sys
from couplet.net import TcpChannel
chan = TcpChannel.connect("127.0.0.1", int(sys.argv[1]), timeout=30)
chan.mark("hello")
msg = chan.recv()
chan.send(msg[::-1])
chan.mark("bulk")
blob = chan.exchange(bytes(range(256)) * 4096)
chan.send(hashlib.sha256(blob).digest())
chan.close()
print(json.dumps(chan.stats()))
"""


def test_two_processes() -> None:
    start = len(FAILURES)
    server, port = TcpChannel.bind(0)
    proc = subprocess.Popen([sys.executable, "-c", _CHILD, str(port)], cwd=_PY,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        chan = TcpChannel.accept(server, timeout=TIMEOUT, accept_timeout=60)
        chan.mark("hello")
        chan.send(b"ping-0123")
        echoed = chan.recv()
        chan.mark("bulk")
        mine = os.urandom(1 << 20)
        theirs = chan.exchange(mine)
        digest = chan.recv()
        closed = abort_text(chan.recv)
        chan.close()
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    check(proc.returncode == 0, f"child process exits cleanly ({err.decode(errors='replace')[-300:]})")
    check(echoed == b"3210-gnip", "two processes: echo")
    check(theirs == bytes(range(256)) * 4096, "two processes: 1 MiB exchange, child -> parent")
    check(digest == hashlib.sha256(mine).digest(), "two processes: 1 MiB exchange, parent -> child")
    check(closed == "peer closed", "two processes: data sent before the child exits arrives, then EOF")
    mine_stats = chan.stats()
    peer_stats = json.loads(out)
    check(mine_stats["rounds"] == 4 and peer_stats["rounds"] == 4,
          "two processes: both endpoints count 4 rounds")
    check((mine_stats["party"], peer_stats["party"]) == (0, 1), "two processes: party indices")
    check(mine_stats["bytes_sent"] == peer_stats["bytes_recv"] == 9 + (1 << 20)
          and peer_stats["bytes_sent"] == mine_stats["bytes_recv"] == 9 + (1 << 20) + 32,
          "two processes: byte counters agree")
    for name in ("hello", "bulk"):
        a, b = mine_stats["phases"][name], peer_stats["phases"][name]
        check(a["rounds"] == b["rounds"] == 2 and a["bytes_sent"] == b["bytes_recv"]
              and b["bytes_sent"] == a["bytes_recv"], f"two processes: phase '{name}' agrees")
    _check_phase_sums(mine_stats, "two processes, parent")
    _check_phase_sums(peer_stats, "two processes, child")
    _finish(start)


# --------------------------------------------------------------------------- #
# Delay and bandwidth emulation
# --------------------------------------------------------------------------- #

def _timed(fn):
    def wrapper(chan):
        t0 = time.perf_counter()
        fn(chan)
        return time.perf_counter() - t0
    return wrapper


def test_emulation() -> None:
    start = len(FAILURES)
    d = 0.040
    for tcp in (False, True):
        tag = "tcp" if tcp else "loopback"

        p0, p1 = _ping_pong(3)
        t0, t1, s0, s1 = run_two_party(_timed(p0), _timed(p1), delay_ms=40, tcp=tcp, timeout=TIMEOUT)
        check_time(max(t0, t1), 6 * d, f"{tag}: 6 one-way flights at 40 ms")
        check(s0["delay_ms"] == 40.0 and s0["mbps"] is None, f"{tag}: emulation parameters in stats")
        check(abs(s0["emulated_wait_s"] - 3 * d) < 0.02 and abs(s1["emulated_wait_s"] - 3 * d) < 0.02,
              f"{tag}: each party waited for three flights")

        t0, t1, _, _ = run_two_party(_timed(lambda c: [c.exchange(b"x") for _ in range(4)]),
                                     _timed(lambda c: [c.exchange(b"y") for _ in range(4)]),
                                     delay_ms=40, tcp=tcp, timeout=TIMEOUT)
        check_time(max(t0, t1), 4 * d, f"{tag}: 4 exchanges cost 4 delays (not 8)")

        t0, t1, _, _ = run_two_party(_timed(lambda c: [c.send(b"m") for _ in range(5)]),
                                     _timed(lambda c: [c.recv() for _ in range(5)]),
                                     delay_ms=40, tcp=tcp, timeout=TIMEOUT)
        check_time(t1, d, f"{tag}: a flight of 5 messages is delayed once")
        check(t0 < 0.03, f"{tag}: the sender of a one-way flight does not wait ({t0 * 1000:.1f} ms)")

        # A receiver that is busy for longer than the latency does not wait again.
        def busy(chan):
            chan.recv()  # synchronise: the next message is sent right after this one
            time.sleep(0.12)
            t = time.perf_counter()
            chan.recv()
            return time.perf_counter() - t

        def sender(chan):
            chan.send(b"sync")
            chan.send(b"data")
            return None

        waited, _, _, _ = run_two_party(busy, sender, delay_ms=40, tcp=tcp, timeout=TIMEOUT)
        check(waited < 0.03, f"{tag}: no extra wait for a flight that already arrived "
                             f"({waited * 1000:.1f} ms)")

        # 8 Mbit/s = 10^6 bytes/s.
        t0, t1, s0, s1 = run_two_party(_timed(lambda c: c.send(bytes(100_000))),
                                       _timed(lambda c: c.recv()), mbps=8, tcp=tcp, timeout=TIMEOUT)
        check_time(t1, 0.100, f"{tag}: 100 kB at 8 Mbit/s")
        check(s1["mbps"] == 8.0, f"{tag}: mbps in stats")
        t0, t1, _, _ = run_two_party(_timed(lambda c: [c.send(bytes(40_000)) for _ in range(3)]),
                                     _timed(lambda c: [c.recv() for _ in range(3)]),
                                     mbps=8, tcp=tcp, timeout=TIMEOUT)
        check_time(t1, 0.120, f"{tag}: three 40 kB messages share the link")
        t0, t1, _, _ = run_two_party(_timed(lambda c: c.exchange(bytes(60_000))),
                                     _timed(lambda c: c.exchange(bytes(20_000))),
                                     delay_ms=30, mbps=8, tcp=tcp, timeout=TIMEOUT)
        check_time(t1, 0.030 + 0.060, f"{tag}: delay + transfer time, larger direction")
        check_time(t0, 0.030 + 0.020, f"{tag}: delay + transfer time, smaller direction")

    t0, t1, s0, _ = run_two_party(_timed(_ping_pong(20)[0]), _timed(_ping_pong(20)[1]),
                                  timeout=TIMEOUT)
    check(max(t0, t1) < 0.5 and s0["emulated_wait_s"] == 0.0, "no emulation: no artificial waiting")
    _finish(start)


TESTS = [test_round_counting, test_phase_accounting, test_abort_propagation, test_tcp,
         test_two_processes, test_emulation]


def main() -> int:
    for test in TESTS:
        before = CHECKS
        try:
            test()
        except AssertionError:
            pass  # the failed checks are already recorded in FAILURES
        except Exception as exc:  # an unexpected error is a failure as well
            FAILURES.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print("ERROR:", FAILURES[-1])
        print(f"{test.__name__}: {CHECKS - before} checks")
    print(f"test_net: {CHECKS} checks, {len(FAILURES)} failures")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
