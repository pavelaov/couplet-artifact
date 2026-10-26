"""Helper of the C++ and Python tests: the known answers that depend on the labels.

Several tests hold constants that were computed once by the Python reference
and that change whenever a domain-separation string changes (the PRG labels
``Couplet-G/-C/-VL/-VR``, the hash domains ``Couplet-kdf``, ``-com``,
``-coin``, ``-dom``, ``-h2g``, ``-boolcheck``, the sub-session identifiers).
This script prints all of them, grouped by the test that holds them, so that
they can be regenerated and compared after such a change:

    src/selftest.cpp            IV words and the PRG maps for two seeds
    src/test_backend_prims.cpp  hash-to-group, key derivation, commitments,
                                session domains, coin expansion
    src/test_prep.cpp           leaky-AND domains, sub-session identifier,
                                BoolCheck digests, base-OT derivations, one
                                leaky-AND transfer message
    src/test_baselines.cpp      identifiers, H0 domains and tree values of the
                                two baseline protocols
    src/test_protocol.cpp       the identifier of the Mask sub-session
    python/tests/test_primitives.py   KAT_H2G_EMPTY, KAT_H2G_TAG_MSG

The other files with reference values have their own generators
(``python/vectors/make_vectors.py``, ``gen_test_circuits.py``,
``gen_protocol_vectors.py``).  Nothing is written; usage, from any directory:

    python artifact/cpp/tests/gen_label_vectors.py
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "python"))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import ascon, ed25519 as ed, gf128 as gf, hashes as hs  # noqa: E402
from couplet.baselines import leaky_zhang as lz, semihonest_ds as sh  # noqa: E402
from couplet.prep import aand, baseot, check, params  # noqa: E402

PRG_TEST = (0x50 << 56) | 0x03      # hashes::dom::PRG_TEST
PRG_COIN = (0x50 << 56) | 0x02      # hashes::dom::PRG_COIN
H2G = b"Couplet-h2g"                # the domain string of couplet.ed25519.hash_to_group


def as_int(row) -> int:
    return int(row[0]) | (int(row[1]) << 64)


def frame(*parts: bytes) -> bytes:
    return b"".join(len(p).to_bytes(8, "little") + p for p in parts)


def h2g_counter(tag: bytes, msg: bytes) -> int:
    """The accepted counter of try-and-increment, recomputed from the definition."""
    ctr = 0
    while True:
        digest = hashlib.sha256(frame(H2G, tag, msg, ctr.to_bytes(4, "little"))).digest()
        try:
            cand = ed.scalar_mult(8, ed.decompress(digest))
            if not ed.is_identity(cand):
                if not ed.equal(cand, ed.hash_to_group(tag, msg)):
                    raise RuntimeError("hash_to_group does not follow its definition")
                return ctr
        except ValueError:
            pass
        ctr += 1


def section(title: str) -> None:
    print(f"\n== {title}")


def selftest_vectors() -> None:
    section("src/selftest.cpp")
    for name in ("LABEL_G", "LABEL_C", "LABEL_VL", "LABEL_VR"):
        label = getattr(ascon, name)
        print(f"IV words of {label.decode()}: " + ", ".join(f"{w:#x}" for w in ascon.label_words(label)))
    for seed in (bytes(16), bytes(range(16))):
        s_l, t_l, s_r, t_r = ascon.G(seed)
        print(f"seed {seed.hex()}: G = ({s_l.hex()}, {t_l}, {s_r.hex()}, {t_r})")
        for label, fn in (("Conv", ascon.conv), ("ConvV_L", ascon.convv_L), ("ConvV_R", ascon.convv_R)):
            v64, t64 = fn(seed, 64, 64)
            v32, t32 = fn(seed, 32, 64)
            print(f"  {label}: k=64 ({v64:032x}, {t64:032x}); k=32 ({v32:024x}, {t32:024x})")


def backend_vectors() -> None:
    section("src/test_backend_prims.cpp and python/tests/test_primitives.py")
    print("reserved domains (x3 words of the PRG labels): " + ", ".join(f"{d:#x}" for d in sorted(hs.RESERVED_DOMS)))
    base = ed.base_mult(1)
    for tag, msg in ((b"", b""), (b"tag", b"msg"), (b"mr", b"abc"), (H2G, bytes(range(32))), (b"a", b"bc"),
                     (b"ab", b"c"), (b"mr-test", ed.compress(base)), (b"tag", b"msg3")):
        print(f"hash_to_group({tag.hex() or '-'}, {msg.hex() or '-'}): counter {h2g_counter(tag, msg)}, "
              f"point {ed.compress(ed.hash_to_group(tag, msg)).hex()}")
    print("hash_to_group_input('tg', 'message', 0x01020304): "
          + frame(H2G, b"tg", b"message", bytes.fromhex("04030201")).hex())
    point = ed.hash_to_group(b"test-tag", b"point-P")
    print(f"P = hash_to_group('test-tag', 'point-P'): {ed.compress(point).hex()}, counter "
          f"{h2g_counter(b'test-tag', b'point-P')}")
    for scalar in ("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", "ff" * 32,
                   "bcaff64aa174de78063e7bcad46ee91ee56264a14bbdc882c4f5247fa146f3bf",
                   "5e443f6e206855cd66ca3722dd825d1e40dfc09eb86f89c9c71d4430b5b0b11f"):
        k = int.from_bytes(bytes.fromhex(scalar), "little")
        print(f"  mul(P, {scalar[:8]}..): {ed.compress(ed.scalar_mult(k, point)).hex()}")
    print(f"  P + B: {ed.compress(ed.add(point, base)).hex()}")
    print(f"  P - B: {ed.compress(ed.sub(point, base)).hex()}")
    print(f"  2 P:   {ed.compress(ed.double(point)).hex()}")
    print("rejected attempts over hash_to_group('order-test', le64(i)), i < 200: "
          f"{sum(h2g_counter(b'order-test', i.to_bytes(8, 'little')) for i in range(200))}")
    seq = bytes(range(32))
    print(f"framed_sha256('Couplet', 00..1f, ''): {hs.sha256(b'Couplet', seq, b'').hex()}")
    print(f"kdf32('sid', le64(5), 00..1f): {hs.kdf(b'sid', (5).to_bytes(8, 'little'), seq, nbytes=32).hex()}")
    sid, msg, omega = b"session-1", b"hello commitment", bytes(range(32))
    for party in (None, 0, 1):
        for label in (b"", b"v"):
            print(f"commitment, party {party}, label '{label.decode()}': {hs._com(sid, label, party, omega, msg).hex()}")
    print(f"session_dom_ccr('aand', 'session-1'): {params.session_dom(params.DOM_CCR, b'aand', sid):#x}")
    print(f"session_dom_prg('cot', 'session-1'):  {params.session_dom(params.DOM_PRG, b'cot', sid):#x}")
    share = bytes(range(100, 132))
    for party in (0, 1):
        print(f"coin share commitment of party {party}: {hs._com(sid, b'coin|chi', party, omega, share).hex()}")
    expanded = hs.sha256_expand(100, b"Couplet-coin", sid, b"chi", omega)
    print(f"coin_expand (80 bytes): {expanded[:80].hex()}")
    print("coin GF(2^128) expansion (lo, hi, lo, hi): "
          + ", ".join(f"{int(w):#018x}" for w in hs.ctr_prg(expanded[:16], PRG_COIN, 4)))


def prep_vectors() -> None:
    section("src/test_prep.cpp")
    tsid = b"test-prep-session"

    def tvec(tag: int, n: int):
        return hs.prg_gf128(bytes([tag]) * 16, PRG_TEST, n)

    def tbits(tag: int, n: int):
        return hs.prg_bits(bytes([tag]) * 16, PRG_TEST, n)

    print(f"leaky-AND hash domains: {aand.h_dom(tsid, 0):#x}, {aand.h_dom(tsid, 1):#x}")
    print(f"identifier of the preprocessing sub-session: {hs.sha256(b'Couplet-prep-sid', tsid).hex()}")
    d0 = as_int(tvec(1, 1)[0]) | 1
    opened, m0 = tbits(2, 50), tvec(3, 50)
    w, big_m = as_int(tvec(4, 1)[0]), as_int(tvec(5, 1)[0])
    print(f"BoolCheck digest with a field opening: {check.check_digest(tsid, 3, d0, opened, m0, [(w, big_m)]).hex()}")
    print(f"BoolCheck digest of bits only:         {check.check_digest(tsid, 7, d0, opened, m0).hex()}")
    p5 = ed.base_mult(5)
    print(f"base OT: seed derivation: {baseot._seed(tsid, 1, 7, 0, p5).hex()}")
    print("base OT: H_j(P):          "
          + ed.compress(ed.hash_to_group(baseot._h_tag(tsid, 1, 7, 0), ed.compress(p5))).hex())
    n = 37
    xk, ym = tvec(30, n), tvec(31, n)
    dl = as_int(tvec(34, 1)[0]) | 1
    tweak = np.arange(1000, 1000 + n, dtype=np.uint64)
    dom_out = aand.h_dom(tsid, 0)
    g_own = hs.H_ccr(xk, tweak, dom_out) ^ hs.H_ccr(gf.xor_scalar(xk, dl), tweak, dom_out) ^ ym
    print(f"leaky AND vector: SHA-256 of the transfer message G_0: {hashlib.sha256(gf.vec_to_bytes(g_own)).hexdigest()}")


def baseline_vectors() -> None:
    section("src/test_baselines.cpp")
    sid = b"test-baselines"
    seed = bytes(range(16))

    def kat_gf(n: int, start: int):
        return hs.ctr_prg(seed, PRG_TEST, 2 * n, start=start).reshape(n, 2)

    def sha(*arrays) -> str:
        return hashlib.sha256(b"".join(gf.vec_to_bytes(a) for a in arrays)).hexdigest()

    ssid = lz.sub_sid(sid, 3)
    dom = lz.h0_domain(ssid)
    print(f"identifier of leaky key 3: {ssid.hex()}")
    print(f"H0 domain of the leaky baseline: {dom:#x}")
    ssid2 = hs.sha256(sh._SID_TAG, sid, (7).to_bytes(8, "little"))
    print(f"identifier of semi-honest key 7: {ssid2.hex()}")
    print(f"H0 domain of the semi-honest baseline: {lz.h0_domain(ssid2, sh.LABEL_H0):#x}")
    nodes = kat_gf(9, 0)
    hashed = lz.hash0(nodes, dom)
    print(f"H0 of nine elements: {sha(hashed)}")
    print(f"H1 of nine elements: {sha(*lz.hash1(nodes, dom))}")
    print(f"expand_level of nine nodes: {sha(lz.expand_level(nodes, hashed, as_int(kat_gf(1, 100)[0])))}")
    depth = 5
    level = kat_gf(1, 2000)
    cws = [as_int(r) for r in kat_gf(depth, 2100)]
    accs = []
    for i in range(depth):
        h = lz.hash0(level, dom)
        accs.append(np.bitwise_xor.reduce(h, axis=0))
        level = lz.expand_level(level, h, cws[i])
    print(f"tree: XOR of H0 over each of the five levels: {sha(np.array(accs, dtype=np.uint64))}")
    print(f"tree: leaves and H1 of the leaves: {sha(level, *lz.hash1(level, dom))}")


def protocol_vectors() -> None:
    section("src/test_protocol.cpp")
    tag = b"Couplet-mask-sid"
    print(f"sid_mask of 'interop-online' (tag of {len(tag)} bytes): "
          f"{hashlib.sha256(frame(tag, b'interop-online')).hexdigest()}")


def main() -> int:
    selftest_vectors()
    backend_vectors()
    prep_vectors()
    baseline_vectors()
    protocol_vectors()
    return 0


if __name__ == "__main__":
    sys.exit(main())
