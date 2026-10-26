"""Cross-language test: the Python reference as the peer of ``couplet_party``.

One party of Pi_Gen is the C++ program ``build/couplet_party`` (started with
``--verify`` and without a delay), the other party is this script, which runs
the Python reference implementation (``couplet.session``, ``couplet.protocol``
with the real preprocessing) over the same TCP connection.  The two programs
share nothing but the wire: base OT, OT extension, leaky ANDs, bucketing, the
evaluation headers, every AND layer and both output checks must agree message
for message, or one of the MAC checks aborts.

After the protocol both sides run the TEST-ONLY verification step of
``couplet_party`` (keys and MAC key shares are exchanged in the clear; not
part of the protocol) and each checks the result on the whole domain.

Usage (two shells, or ``--spawn`` to start the C++ side from here):

    python tests/interop_peer.py --role 1 --port 47200 --task dpf --n 8
    build/couplet_party --role 0 --port 47200 --task dpf --n 8 --verify

    python tests/interop_peer.py --spawn --task dcf --n 4 --beta secret --batch 2
    python tests/interop_peer.py --spawn --all          # the list used for the report

The online layer (Pi_AuthEval: Key, Query, Gate, Mask with a shared input and
with an authenticated wire, Apply, Lin, Open with ArithCheck and an output
mask) has its own fixed script, run on the C++ side by
``build/test_protocol --interop ROLE PORT`` and here by ``--online``:

    python tests/interop_peer.py --online --spawn --all  # both role assignments

Both sides of the online script run their driver in strict mode and reserve
the triples of the whole session at setup (one bucketing population: the DPF
key, the two table gates and one on-demand output mask); the script fails
unless both end with exactly one triple request and an empty pool.

Inference of the Couplet layer (``--infer``): this script runs
``couplet.infer.InferParty`` with the real preprocessing against
``build/couplet_infer`` (started with ``--verify``): the same hello (public
parameters and nonces, session identifier
SHA-256(frame("Couplet-infer-session", params, nonce_0, nonce_1))), then the
whole protocol, message for message: manifest digest, session setup with the
preprocessing, every generation circuit, the online pipeline and the
release.  Each side checks its logits (mode exact: the exported
``exact_logits``; mode round: the mirror on the recorded thresholds), both
end with one triple request and an empty pool, and the two sides' logits and
thresholds must be equal:

    python tests/interop_peer.py --infer --spawn --all   # the list of the report

runs the reduced MLP ``mlp16`` at (16, 8) in public-model mode and ``mlp4``
at (8, 4) in private-model mode (with the output masks of the logits), each
in both truncation modes and both role assignments (eight runs), and the
tiled private-model plan (task R-TILE): ``mlp7`` [7, 3, 2] at (8, 4) with
``--tile 2x2 --tile-limit 0`` (every weight matrix in tiles of at most 2 x 2,
edge tiles 2 x 1, 1 x 2 and 1 x 1, ten tile triples; the input masks in
instances of four), both modes and both role assignments (four runs).
``--tile RxC`` and ``--tile-limit L`` pass the tiling to both sides (and
append ``|tile=RxC|tile_limit=L`` to the hello, as couplet_infer does).

``--role`` is the role of THIS script; with ``--spawn`` the C++ program takes
the other role.  ``--build-dir DIR`` takes the C++ binaries from ``DIR``
instead of ``../build``.  Exit status 0 iff the protocol completed and both
sides verified the outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "python"))
_CIRCUITS = os.path.normpath(os.path.join(_HERE, "..", "..", "circuits"))   # the exported circuits of the artifact
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import abb, circuits_gen as cg, fss, gates, hashes as hs, protocol as pg  # noqa: E402
from couplet.net import TcpChannel  # noqa: E402
from couplet.session import Session  # noqa: E402

K, S = 64, 64
_BUILD_DIR = os.path.normpath(os.path.join(_HERE, "..", "build"))   # replaced by --build-dir


def descriptor(args) -> dict:
    if args.task in ("dpf", "dcf"):
        make = cg.dpf_descriptor if args.task == "dpf" else cg.dcf_descriptor
        return make(args.n, K, S, "xor", "secret" if args.beta == "secret" else "public_one")
    if args.task == "gate-table":
        return gates.table_descriptor(args.n, K, S, args.auth, 0)
    return gates.comparison_descriptor(args.n, K, S, (1 << args.n) // 3, args.auth, 0)


def reveal_encode(delta: int, alpha: list, beta: list, outputs: list) -> bytes:
    def u64(v: int) -> bytes:
        return int(v).to_bytes(8, "little")
    parts = [u64(delta), u64(len(alpha))] + [u64(a) for a in alpha] + [u64(len(beta))] + [u64(b) for b in beta]
    parts.append(u64(len(outputs)))
    for o in outputs:
        parts += [u64(len(o)), o]
    return b"".join(parts)


def reveal_decode(data: bytes) -> dict:
    pos = 0

    def u64() -> int:
        nonlocal pos
        v = int.from_bytes(data[pos:pos + 8], "little")
        pos += 8
        return v
    delta = u64()
    alpha = [u64() for _ in range(u64())]
    beta = [u64() for _ in range(u64())]
    outputs = []
    for _ in range(u64()):
        n = u64()
        outputs.append(data[pos:pos + n])
        pos += n
    if pos != len(data):
        raise ValueError("malformed verification message")
    return {"delta": delta, "alpha": alpha, "beta": beta, "outputs": outputs}


def check_pair(family: str, k0, k1, alpha: int, beta: int, delta: int) -> bool:
    """Full-domain check on both planes with the unreduced Delta, and against the plaintext Gen."""
    n, w = k0.n, k0.k + k0.s
    mod = 1 << w
    gen = fss.dpf_gen if family == "dpf" else fss.dcf_gen
    ref = gen(alpha, beta, delta, n, k0.k, k0.s, roots=(k0.seed, k1.seed))
    if ref[0].to_bytes() != k0.to_bytes() or ref[1].to_bytes() != k1.to_bytes():
        return False
    full = fss.dpf_eval_all if family == "dpf" else fss.dcf_eval_all
    y0, t0 = full(0, k0)
    y1, t1 = full(1, k1)
    for x in range(1 << n):
        hit = (x == alpha) if family == "dpf" else (x < alpha)
        if (int(y0[x]) + int(y1[x])) % mod != (beta if hit else 0):
            return False
        if (int(t0[x]) + int(t1[x])) % mod != (delta * beta % mod if hit else 0):
            return False
    return True


def check_gate(m0, m1, delta: int) -> bool:
    desc = m0.descriptor
    n, w = desc["n"], desc["k"] + desc["s"]
    N, mod = 1 << n, 1 << w
    bits = m0.rp_bits
    rp = (m0.rp_share + m1.rp_share) % (1 << bits)
    r = rp % N
    ok = True
    if desc["auth_input"]:
        ok &= (m0.rp_tag_share + m1.rp_tag_share) % (1 << bits) == delta * rp % (1 << bits)
    if desc["dpf_one"]:
        ok &= check_pair("dpf", m0.dpf_key, m1.dpf_key, r, 1, delta)
    for i, (c, payload) in enumerate(desc["thresholds"]):
        w_c = int(r + c >= N)
        ok &= check_pair("dcf", m0.dcf_keys[i], m1.dcf_keys[i], (r + c) % N, 1 if payload == "one" else r, delta)
        ok &= (m0.w_shares[i][0] + m1.w_shares[i][0]) % mod == w_c
        ok &= (m0.w_shares[i][1] + m1.w_shares[i][1]) % mod == delta * w_c % mod
    return bool(ok)


def run_party(args) -> int:
    role = args.role
    desc = descriptor(args)
    circuit = pg.generation_circuit(desc)
    digest = abb.circuit_digest(circuit).hex()
    n_and = circuit.stats()["n_and"]
    params = (f"couplet-e2e/1|task={args.task}|n={args.n}|k={K}|s={S}|beta={args.beta}|auth={1 if args.auth else 0}"
              f"|batch={args.batch}|prep=real|sigma={args.sigma}|delay_ms={0.0:.6f}|verify=1|circuit={digest}")
    if role == 0:
        chan = TcpChannel.listen(args.port, args.host, party=0, accept_timeout=120)
    else:
        chan = TcpChannel.connect(args.host, args.port, party=1, connect_timeout=120)
    try:
        nonce = secrets.token_bytes(16)
        hello = params.encode() + b"\x00" + nonce
        theirs = chan.exchange(hello)
        if len(theirs) != len(hello) or theirs[:len(params) + 1] != hello[:len(params) + 1]:
            raise RuntimeError("the two programs were started with different parameters: " + repr(theirs[:300]))
        nonces = (nonce, theirs[-16:]) if role == 0 else (theirs[-16:], nonce)
        sid = hs.sha256(b"Couplet-e2e-sid", params.encode(), nonces[0], nonces[1])

        t0 = time.perf_counter()
        session = Session(role, chan, sid, k=K, s=S, prep="real", sigma=args.sigma)
        session.setup(reserve_triples=n_and * args.batch)          # ONE bucketing population
        alpha = [secrets.randbits(args.n) for _ in range(args.batch)]
        beta = [secrets.randbits(K) for _ in range(args.batch)] if args.beta == "secret" else []
        if args.task in ("dpf", "dcf"):
            gen = pg.gen_dpf if args.task == "dpf" else pg.gen_dcf
            outs = gen(session, args.n, "xor", desc["beta_mode"], alpha, beta or None, batch=args.batch)
        else:
            alpha = []
            outs = pg.gen_gate(session, desc, batch=args.batch)
        seconds = time.perf_counter() - t0

        # TEST-ONLY verification step (not part of the protocol).
        mine = {"delta": session.delta_share, "alpha": alpha, "beta": beta, "outputs": [o.to_bytes() for o in outs]}
        other = reveal_decode(chan.exchange(reveal_encode(mine["delta"], mine["alpha"], mine["beta"], mine["outputs"])))
        r0, r1 = (mine, other) if role == 0 else (other, mine)
        delta = r0["delta"] + r1["delta"]                           # the integer sum, never reduced
        ok = len(r0["outputs"]) == len(r1["outputs"]) == args.batch
        for j in range(args.batch if ok else 0):
            if args.task in ("dpf", "dcf"):
                cls = fss.DPFKey if args.task == "dpf" else fss.DCFKey
                a = r0["alpha"][j] ^ r1["alpha"][j]
                b = (r0["beta"][j] + r1["beta"][j]) % (1 << K) if args.beta == "secret" else 1
                ok = ok and check_pair(args.task, cls.from_bytes(r0["outputs"][j]), cls.from_bytes(r1["outputs"][j]), a, b, delta)
            else:
                ok = ok and check_gate(pg.GateMaterial.from_bytes(r0["outputs"][j]), pg.GateMaterial.from_bytes(r1["outputs"][j]), delta)
        print(json.dumps({"peer": "python reference", "role": role, "circuit": circuit.meta.get("name"), "digest": digest,
                          "batch": args.batch, "n_and": n_and, "bytes_sent": chan.bytes_sent, "bytes_recv": chan.bytes_recv,
                          "seconds": round(seconds, 3), "outputs_correct": bool(ok),
                          "delta_at_least_2_pow_s": delta >= (1 << S)}), flush=True)
        return 0 if ok else 4
    finally:
        chan.close()


def run_online(args) -> int:
    """The fixed script of ``test_protocol --interop`` on the Python reference (see run_interop there)."""
    from couplet.autheval import AuthEval
    role = args.role
    T = [(0x9E3779B97F4A7C15 * (x + 1) + 7 * x * x) % (1 << 64) for x in range(256)]
    x, x_share, alpha_share = 177, (77, 100)[role], (3, 9)[role]
    if role == 0:
        chan = TcpChannel.listen(args.port, args.host, party=0, accept_timeout=120)
    else:
        chan = TcpChannel.connect(args.host, args.port, party=1, connect_timeout=120)
    try:
        # Both programs run the driver in strict mode (its default in both languages; CHANGELOG
        # F1 item 1 and F2 item 2) and draw all triples of the session in ONE request (F1 W2):
        # the DPF key, the two table gates, and one on-demand output mask for the value that is
        # not exact.  ``test_protocol --interop`` computes the same budget on the C++ side.
        v = AuthEval(role, chan, b"interop-online", k=K, s=S, prep="real", sigma=40)
        spec_shared = gates.GateSpec.table(8, K, S, T, False)
        spec_wire = gates.GateSpec.table(8, K, S, T, True)
        reserve = v.key_budget("dpf", 4) + v.gate_budget(spec_shared) + v.gate_budget(spec_wire)
        v.setup(reserve_triples=reserve, reserve_outmasks=1)
        kid = v.key(None, "dpf", 4, alpha_share)
        one = v.query(kid, 3 ^ 9)
        g0 = v.gate(None, spec_shared)
        hat0 = v.mask_shared(g0, x_share)
        y0 = v.apply(g0)
        g1 = v.gate(None, spec_wire)
        hat1 = v.mask_wire(g1, v.lin([(x, one)]))
        y1 = v.apply(g1)
        both = v.open_many([y0, y1])
        z = v.open(v.lin([(3, y0)], 5))
        expected = [T[x], T[x], (3 * T[x] + 5) % (1 << 64)]
        info = v.info()
        one_population = v.triple_requests == 1 and info["triples_unused"] == 0
        ok = [both[0], both[1], z] == expected and v.state == "ACTIVE" and one_population
        print(json.dumps({"peer": "python reference", "role": role, "x_hat": [hat0, hat1], "opened": [both[0], both[1], z],
                          "expected": expected, "arith_checks": v.checker.checks, "bytes_sent": chan.bytes_sent,
                          "rounds": chan.rounds, "triple_requests": v.triple_requests,
                          "triples_requested": info["triples_requested"], "triples_unused": info["triples_unused"],
                          "outputs_correct": bool(ok)}), flush=True)
        return 0 if ok else 4
    finally:
        chan.close()


_PPML_VECTORS = os.path.normpath(os.path.join(_HERE, "..", "..", "python", "vectors", "ppml"))


def infer_params(args) -> str:
    """The public parameters of the hello of couplet_infer (byte for byte)."""
    tail = ""
    if getattr(args, "tile", None) is not None or getattr(args, "tile_limit", None) is not None:
        t = infer_tiling(args)
        tail = f"|tile={t.rows}x{t.cols}|tile_limit={t.limit}"
    return (f"couplet-infer/1|mode={args.mode}|private={1 if args.private_model else 0}|prep=real|sigma={args.sigma}"
            f"|k={K}|s={S}|chunk_and=0|delay_ms={0.0:.6f}|input={args.input_index}" + tail)


def infer_tiling(args):
    """couplet.nn.Tiling of --tile / --tile-limit (the defaults where not given)."""
    from couplet import nn as nnm
    rows, cols = nnm.DEFAULT_TILING.rows, nnm.DEFAULT_TILING.cols
    if getattr(args, "tile", None) is not None:
        parts = args.tile.lower().split("x")
        if len(parts) != 2 or not all(x.isdigit() for x in parts):
            raise SystemExit("--tile must be RxC")
        rows, cols = int(parts[0]), int(parts[1])
    limit = nnm.DEFAULT_TILING.limit if getattr(args, "tile_limit", None) is None else int(args.tile_limit)
    return nnm.Tiling(rows, cols, limit)


def run_infer_party(args) -> int:
    """One party of an inference session (couplet.infer, real preprocessing) against couplet_infer."""
    from couplet import nn as nnm
    from couplet.infer import InferParty
    from couplet.session import ACTIVE
    role = args.role
    model = nnm.Model.load(os.path.join(_PPML_VECTORS, f"{args.model}_model.json"))
    with open(os.path.join(_PPML_VECTORS, f"{args.model}_vectors.json"), encoding="utf-8") as fh:
        vec = json.load(fh)
    x = vec["inputs"][args.input_index]
    params = infer_params(args)
    if role == 0:
        chan = TcpChannel.listen(args.port, args.host, party=0, accept_timeout=120)
    else:
        chan = TcpChannel.connect(args.host, args.port, party=1, connect_timeout=120)
    try:
        nonce = secrets.token_bytes(16)
        hello = params.encode() + b"\x00" + nonce
        theirs = chan.exchange(hello)
        if len(theirs) != len(hello) or theirs[:len(params) + 1] != hello[:len(params) + 1]:
            raise RuntimeError("the two programs were started with different parameters: " + repr(theirs[:300]))
        nonces = (nonce, theirs[-16:]) if role == 0 else (theirs[-16:], nonce)
        sid = hs.sha256(b"Couplet-infer-session", params.encode(), nonces[0], nonces[1])
        own = model.shape_only() if (args.private_model and role == 1) else model
        t0 = time.perf_counter()
        p = InferParty(role, chan, sid, own, args.mode, prep="real", sigma=args.sigma, k=K, s=S,
                       private_model=args.private_model, tiling=infer_tiling(args))
        logits = p.run(x[model.feature_slice(role)])
        seconds = time.perf_counter() - t0
        rec = p.record
        if args.mode == "exact":
            expected = vec["exact_logits"][args.input_index]
        else:
            expected = model.mirror(x, "round", rec["thresholds"]).logits
        ok = (logits == expected and p.state == ACTIVE and p.triple_requests == 1 and rec["triples_unused"] == 0)
        print(json.dumps({"peer": "python reference", "role": role, "model": args.model, "mode": args.mode,
                          "private_model": args.private_model, "sites": rec["sites"], "and_gates": rec["and_gates"],
                          "tiled": rec["tiled"], "tiles": rec["tiles"],
                          "manifest_digest": rec["manifest_digest"], "logits": logits, "expected": expected,
                          "thresholds": rec["thresholds"], "openings": rec["openings"], "checks": rec["checks"],
                          "bytes_sent": chan.bytes_sent, "bytes_recv": chan.bytes_recv, "rounds": chan.rounds,
                          "seconds": round(seconds, 3), "triple_requests": p.triple_requests,
                          "logits_correct": bool(ok)}), flush=True)
        # the counters of the protocol alone (couplet_infer resets its counters after the hello)
        return (0 if ok else 4), {"logits": logits, "thresholds": rec["thresholds"],
                                  "bytes_sent": chan.bytes_sent - len(hello), "bytes_recv": chan.bytes_recv - len(theirs),
                                  "rounds": chan.rounds - 1}
    finally:
        chan.close()


def spawn_infer(args) -> int:
    """Start couplet_infer in the other role, run this role here, compare the two outcomes."""
    import tempfile
    args.port = free_port()
    with tempfile.TemporaryDirectory(prefix="couplet_interop_infer_") as tmp:
        out = os.path.join(tmp, "cpp_record.json")
        cmd = [find_binary("couplet_infer"), "--role", str(1 - args.role), "--host", args.host, "--port", str(args.port),
               "--model", os.path.join(_PPML_VECTORS, f"{args.model}_model.json"),
               "--input", os.path.join(_PPML_VECTORS, f"{args.model}_vectors.json"),
               "--input-index", str(args.input_index), "--mode", args.mode, "--prep", "real", "--sigma", str(args.sigma),
               "--verify", "--quiet", "--out", out, "--circuits", _CIRCUITS]
        if args.private_model:
            cmd.append("--private-model")
        if getattr(args, "tile", None) is not None:
            cmd += ["--tile", args.tile]
        if getattr(args, "tile_limit", None) is not None:
            cmd += ["--tile-limit", str(args.tile_limit)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        mine = None
        try:
            code, mine = run_infer_party(args)
        except Exception as exc:  # noqa: BLE001 - report and fail
            print(f"python side failed: {type(exc).__name__}: {exc}", flush=True)
            code = 1
        try:
            so, se = proc.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            proc.kill()
            so, se = proc.communicate()
        print("  c++ side:", (so or "").strip(), (se or "").strip(), f"[exit {proc.returncode}]", flush=True)
        same = False
        if code == 0 and proc.returncode == 0 and os.path.isfile(out):
            with open(out, encoding="utf-8") as fh:
                theirs = json.load(fh)
            same = (theirs["logits"] == mine["logits"] and theirs["online"]["thresholds"] == mine["thresholds"]
                    and theirs["totals"]["bytes_sent"] == mine["bytes_recv"]
                    and theirs["totals"]["bytes_recv"] == mine["bytes_sent"]
                    and theirs["totals"]["rounds"] == mine["rounds"])
            print(f"  both sides: logits, thresholds, bytes and rounds agree: {same}", flush=True)
    return 0 if code == 0 and proc.returncode == 0 and same else 1


def find_binary(stem: str = "couplet_party") -> str:
    for name in (stem + ".exe", stem):
        path = os.path.join(_BUILD_DIR, name)
        if os.path.isfile(path):
            return os.path.normpath(path)
    raise SystemExit(f"{stem} not found in {_BUILD_DIR}: build the C++ code first")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn_online(args) -> int:
    """Start ``test_protocol --interop`` in the other role and run the online script here."""
    args.port = free_port()
    proc = subprocess.Popen([find_binary("test_protocol"), "--circuits", _CIRCUITS, "--interop", str(1 - args.role),
                             str(args.port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        code = run_online(args)
    except Exception as exc:  # noqa: BLE001 - report and fail
        print(f"python side failed: {type(exc).__name__}: {exc}", flush=True)
        code = 1
    try:
        out, err = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    print("  c++ side:", (out or "").strip(), (err or "").strip(), f"[exit {proc.returncode}]", flush=True)
    return 0 if code == 0 and proc.returncode == 0 else 1


def spawn(args) -> int:
    """Start the C++ program in the other role and run this role here."""
    port = free_port()
    args.port = port
    cmd = [find_binary(), "--role", str(1 - args.role), "--host", args.host, "--port", str(port), "--task", args.task,
           "--n", str(args.n), "--batch", str(args.batch), "--sigma", str(args.sigma), "--verify", "--quiet",
           "--circuits", _CIRCUITS]
    if args.task in ("dpf", "dcf"):
        cmd += ["--beta", args.beta]
    elif args.auth:
        cmd += ["--auth"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        code = run_party(args)
    except Exception as exc:  # noqa: BLE001 - report and fail
        print(f"python side failed: {type(exc).__name__}: {exc}", flush=True)
        code = 1
    try:
        out, err = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
    print("  c++ side:", (out or "").strip(), (err or "").strip(), f"[exit {proc.returncode}]", flush=True)
    return 0 if code == 0 and proc.returncode == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Python peer of couplet_party (cross-language test)")
    ap.add_argument("--role", type=int, default=1, choices=(0, 1), help="role of this script (default 1)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=47200)
    ap.add_argument("--task", default="dpf", choices=("dpf", "dcf", "gate-table", "gate-cmp"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--beta", default="public", choices=("public", "secret"))
    ap.add_argument("--auth", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sigma", type=int, default=40)
    ap.add_argument("--spawn", action="store_true", help="start build/couplet_party in the other role")
    ap.add_argument("--all", action="store_true", help="with --spawn: run the list of configurations of the report")
    ap.add_argument("--online", action="store_true",
                    help="run the script of the online layer (peer: build/test_protocol --interop ROLE PORT)")
    ap.add_argument("--build-dir", default=None, help="directory of the C++ binaries (default: ../build)")
    ap.add_argument("--infer", action="store_true",
                    help="run an inference session of the Couplet layer (peer: build/couplet_infer)")
    ap.add_argument("--model", default="mlp16", help="with --infer: python/vectors/ppml/<model>_model.json")
    ap.add_argument("--mode", default="round", choices=("round", "exact"), help="with --infer")
    ap.add_argument("--private-model", action="store_true", help="with --infer")
    ap.add_argument("--input-index", type=int, default=0, help="with --infer: the input of the vectors file")
    ap.add_argument("--tile", default=None, help="with --infer --private-model: the largest tile RxC (default 16x16)")
    ap.add_argument("--tile-limit", type=int, default=None,
                    help="with --infer --private-model: plans of models above this many weights are tiled (default 4096)")
    args = ap.parse_args(argv)
    if args.build_dir:
        global _BUILD_DIR
        _BUILD_DIR = os.path.abspath(args.build_dir)
    if args.infer:
        if not args.spawn:
            return run_infer_party(args)[0]
        if not args.all:
            return spawn_infer(args)
        configs = [dict(model=m, private_model=p, mode=mode, role=role, input_index=i, tile=None, tile_limit=None)
                   for m, p, i in (("mlp16", False, 1), ("mlp4", True, 2))
                   for mode in ("round", "exact") for role in (0, 1)]
        # task R-TILE: a tiled private-model plan with edge tiles in both directions
        configs += [dict(model="mlp7", private_model=True, mode=mode, role=role, input_index=3, tile="2x2", tile_limit=0)
                    for mode in ("round", "exact") for role in (0, 1)]
        failed = 0
        for cfg in configs:
            a = argparse.Namespace(**vars(args))
            for key, value in cfg.items():
                setattr(a, key, value)
            print(f"inference {cfg}:", flush=True)
            failed += spawn_infer(a) != 0
        print(f"{len(configs) - failed} of {len(configs)} mixed-language inference runs succeeded")
        return 1 if failed else 0
    if args.online:
        if not args.spawn:
            return run_online(args)
        failed = 0
        for role in ((args.role,) if not args.all else (0, 1)):
            a = argparse.Namespace(**vars(args))
            a.role = role
            print(f"online layer, python role {role}:", flush=True)
            failed += spawn_online(a) != 0
        return 1 if failed else 0
    if not args.spawn:
        return run_party(args)
    if not args.all:
        return spawn(args)
    configs = [
        dict(task="dpf", n=4, beta="public", batch=1, role=1),
        dict(task="dpf", n=4, beta="secret", batch=2, role=0),
        dict(task="dcf", n=4, beta="public", batch=1, role=0),
        dict(task="dcf", n=4, beta="secret", batch=1, role=1),
        dict(task="dpf", n=8, beta="public", batch=1, role=0),
        dict(task="gate-table", n=8, beta="public", batch=1, role=1, auth=True),
        dict(task="gate-cmp", n=8, beta="public", batch=1, role=0),
    ]
    failed = 0
    for cfg in configs:
        a = argparse.Namespace(**vars(args))
        a.auth = False
        for key, value in cfg.items():
            setattr(a, key, value)
        print(f"{cfg}:", flush=True)
        failed += spawn(a) != 0
    print(f"{len(configs) - failed} of {len(configs)} mixed-language runs succeeded")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
