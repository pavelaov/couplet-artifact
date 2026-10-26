"""Products with Beaver triples and authenticated input (private-model mode; DESIGN_M3.md
sections 2 and 3).

All sharings are ``couplet.spdz2k.AShare`` over ``Z_{2^w}``; every opening goes
through an ``Opener`` (``couplet.truncmask``) with the MAC check deferred.

    product(opener, x, y, triple)         [[x y]] = [[c]] + d [[b]] + e [[a]] + d e with
                                          d = x - a, e = y - b opened (one exchange)
    products(opener, xs, ys, triples)     several products, ONE exchange for all openings
    matvec(opener, W, x, mt)              [[W x]] for a shared matrix W (rows x cols) and a
                                          shared vector x with a matrix-vector triple
                                          (A, b, c = A b): D = W - A and e = x - b opened in
                                          one exchange, [[W x]]_i = [[c_i]] + sum_j D_ij [[b_j]]
                                          + sum_j [[A_ij]] e_j + sum_j D_ij e_j
    matvec_tiled(opener, W, x, tiles)     [[W x]] as the sum of tile products (tiled private-model
                                          plans, task R-TILE): ``tiles`` lists ``(r0, c0, triple)``
                                          in the plan's row-major tile order, the tiles partition
                                          W; per tile t, D_t = W[tile] - A_t and e_t = x[c0:c0+w_t]
                                          - b_t (its OWN vector mask), all of them opened in ONE
                                          exchange in tile order (D_t row-major, then e_t), and
                                          [[W x]]_i = sum over the tiles of row i of the matvec
                                          formula above; every triple serves exactly one tile
    exchange_inputs(opener, mine, peer)   authenticated input of both parties' values in ONE
                                          exchange: the owner of the masks ``[[rho_j]]`` sends
                                          d_j = x_j - rho_j mod 2^w (W bytes each, little-endian)
                                          and both set [[x_j]] = [[rho_j]] + d_j.  ``mine`` is
                                          ``(InMaskMaterial, values)`` or None, ``peer`` the
                                          peer's ``InMaskMaterial`` or None; phase ``input-open``.

The opened ``d``, ``e`` are uniform because ``a``, ``b``, ``rho`` are uniform over
the whole ring.  Results carry no exactness flags: the caller marks a product
``sign_exact`` when its public bound analysis guarantees that the true
integer fits ``w - 1`` bits.  ``triple_shares`` / ``mattriple_shares`` turn
the material of ``couplet.circuits_ppml`` into ``AShare`` objects.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .circuits_ppml import InMaskMaterial, MatTripleMaterial, TripleMaterial
from .hashes import ProtocolAbort
from .spdz2k import AShare

__all__ = ["PHASE_INPUT", "exchange_inputs", "mattriple_shares", "matvec", "matvec_tiled", "merge_inmasks",
           "product", "products", "triple_shares"]

PHASE_INPUT = "input-open"


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def triple_shares(material: TripleMaterial) -> List[Tuple[AShare, AShare, AShare]]:
    """``[(a_j, b_j, c_j)]`` as sharings over ``Z_{2^w}``."""
    w = material.width
    return [(AShare(a[0], a[1], w), AShare(b[0], b[1], w), AShare(c[0], c[1], w))
            for a, b, c in zip(material.a, material.b, material.c)]


def mattriple_shares(material: MatTripleMaterial) -> Tuple[list, list, list]:
    """``(A, b, c)`` as (rows x cols list of lists, list, list) of sharings over ``Z_{2^w}``."""
    w, rows, cols = material.width, material.rows, material.cols
    A = [[AShare(*material.a[i * cols + j], w) for j in range(cols)] for i in range(rows)]
    return A, [AShare(*x, w) for x in material.b], [AShare(*x, w) for x in material.c]


def _check_shares(shares, w: Optional[int] = None) -> int:
    shares = list(shares)
    if not shares:
        raise ValueError("expected at least one sharing")
    for sh in shares:
        if not isinstance(sh, AShare):
            raise ValueError("expected AShare objects")
        if w is None:
            w = sh.bits
        elif sh.bits != w:
            raise ValueError("all sharings must be over the same modulus")
    return w


def products(opener, xs: Sequence[AShare], ys: Sequence[AShare], triples: Sequence[tuple]) -> List[AShare]:
    """``[[x_j y_j]]`` for all ``j`` with one opening exchange."""
    xs, ys, triples = list(xs), list(ys), list(triples)
    if not (len(xs) == len(ys) == len(triples)):
        raise ValueError("one triple per product is required")
    w = _check_shares(xs)
    _check_shares(ys, w)
    for a, b, c in triples:
        _check_shares((a, b, c), w)
    opened = opener.open_deferred([x - a for x, (a, _, _) in zip(xs, triples)]
                                  + [y - b for y, (_, b, _) in zip(ys, triples)])
    count = len(xs)
    out = []
    for j, (a, b, c) in enumerate(triples):
        d, e = opened[j], opened[count + j]
        out.append((c + b.scale(d) + a.scale(e)).add_public(d * e, opener.role, opener.delta_share))
    return out


def product(opener, x: AShare, y: AShare, triple: tuple) -> AShare:
    """``[[x y]]`` with one Beaver triple ``(a, b, c)``."""
    return products(opener, [x], [y], [triple])[0]


def matvec(opener, W: Sequence[Sequence[AShare]], x: Sequence[AShare], mt) -> List[AShare]:
    """``[[W x]]`` with a matrix-vector triple (``MatTripleMaterial`` or ``(A, b, c)`` shares)."""
    A, b, c = mattriple_shares(mt) if isinstance(mt, MatTripleMaterial) else mt
    W = [list(row) for row in W]
    x = list(x)
    rows, cols = len(A), len(b)
    if len(W) != rows or any(len(row) != cols for row in W) or len(x) != cols or len(c) != rows:
        raise ValueError("the matrix, the vector and the triple have incompatible shapes")
    w = _check_shares(x)
    for row in W:
        _check_shares(row, w)
    flat_w = [W[i][j] for i in range(rows) for j in range(cols)]
    flat_a = [A[i][j] for i in range(rows) for j in range(cols)]
    opened = opener.open_deferred([wij - aij for wij, aij in zip(flat_w, flat_a)] + [xj - bj for xj, bj in zip(x, b)])
    D = opened[:rows * cols]
    e = opened[rows * cols:]
    out = []
    for i in range(rows):
        acc = c[i]
        const = 0
        for j in range(cols):
            d = D[i * cols + j]
            acc = acc + b[j].scale(d) + A[i][j].scale(e[j])
            const += d * e[j]
        out.append(acc.add_public(const, opener.role, opener.delta_share))
    return out


def matvec_tiled(opener, W: Sequence[Sequence[AShare]], x: Sequence[AShare], tiles: Sequence[tuple]) -> List[AShare]:
    """``[[W x]]`` as the sum of tile products; ``tiles``: ``(r0, c0, triple)`` with a
    ``MatTripleMaterial`` or ``(A, b, c)`` shares per tile, in the plan's tile order.

    The tiles must partition the matrix.  One opening exchange for all tiles:
    for each tile in the given order its ``D_t`` (row-major) and then its ``e_t``.
    """
    W = [list(row) for row in W]
    x = list(x)
    rows, cols = len(W), len(x)
    if rows < 1 or any(len(row) != cols for row in W):
        raise ValueError("the matrix and the vector have incompatible shapes")
    w = _check_shares(x)
    for row in W:
        _check_shares(row, w)
    covered = [[False] * cols for _ in range(rows)]
    parts = []
    for r0, c0, mt in tiles:
        A, b, c = mattriple_shares(mt) if isinstance(mt, MatTripleMaterial) else mt
        h, wd = len(A), len(b)
        if not (_is_int(r0) and _is_int(c0)) or r0 < 0 or c0 < 0 or r0 + h > rows or c0 + wd > cols \
                or h < 1 or len(c) != h or any(len(row) != wd for row in A):
            raise ValueError("a tile does not fit the matrix")
        for i in range(r0, r0 + h):
            for j in range(c0, c0 + wd):
                if covered[i][j]:
                    raise ValueError("the tiles overlap")
                covered[i][j] = True
        for row in A:
            _check_shares(row, w)
        _check_shares(b, w)
        _check_shares(c, w)
        parts.append((r0, c0, h, wd, A, b, c))
    if not all(all(row) for row in covered):
        raise ValueError("the tiles do not cover the matrix")
    to_open = []
    for r0, c0, h, wd, A, b, _ in parts:
        to_open += [W[r0 + i][c0 + j] - A[i][j] for i in range(h) for j in range(wd)]
        to_open += [x[c0 + j] - b[j] for j in range(wd)]
    opened = opener.open_deferred(to_open)
    acc: List[Optional[AShare]] = [None] * rows
    const = [0] * rows
    pos = 0
    for r0, c0, h, wd, A, b, c in parts:
        D = opened[pos:pos + h * wd]
        e = opened[pos + h * wd:pos + h * wd + wd]
        pos += h * wd + wd
        for i in range(h):
            t = c[i]
            for j in range(wd):
                d = D[i * wd + j]
                t = t + b[j].scale(d) + A[i][j].scale(e[j])
                const[r0 + i] += d * e[j]
            acc[r0 + i] = t if acc[r0 + i] is None else acc[r0 + i] + t
    return [acc[i].add_public(const[i] % (1 << w), opener.role, opener.delta_share) for i in range(rows)]


def merge_inmasks(parts: Sequence[InMaskMaterial]) -> InMaskMaterial:
    """One ``InMaskMaterial`` from several of the same owner (the instances of a tiled plan, in
    generation order); its descriptor names the total count (not a circuit that ran)."""
    parts = list(parts)
    if not parts:
        raise ValueError("no input masks to merge")
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if any(p.owner != first.owner or p.role != first.role or p.width != first.width
           or (p.values is None) != (first.values is None) for p in parts):
        raise ValueError("input masks of different owners, roles or widths")
    from .circuits_ppml import inmask_descriptor
    total = sum(p.count for p in parts)
    desc = inmask_descriptor(first.owner, total, first.descriptor["k"], first.descriptor["s"],
                             first.descriptor.get("adder", "ripple"))
    shares = [sh for p in parts for sh in p.shares]
    values = None if first.values is None else [v for p in parts for v in p.values]
    return InMaskMaterial(first.role, desc, shares, values)


def exchange_inputs(opener, mine: Optional[tuple], peer: Optional[InMaskMaterial]) -> Tuple[list, list]:
    """Authenticated input of this party's values and the peer's in one exchange; see the docstring.

    Returns ``(shares of my values, shares of the peer's values)`` as sharings
    over ``Z_{2^w}``; a list is empty when that party inputs nothing.  Values
    are integers (signed allowed), embedded modulo ``2^w``.
    """
    role, chan = opener.role, opener.chan
    my_masks, my_values = (None, []) if mine is None else mine
    msg = b""
    w = None
    if my_masks is not None:
        if not isinstance(my_masks, InMaskMaterial) or my_masks.owner != role or my_masks.values is None:
            raise ValueError("mine must be this party's own InMaskMaterial with its values")
        w = my_masks.width
        my_values = [int(v) for v in my_values]
        if len(my_values) != my_masks.count:
            raise ValueError("one value per input mask is required")
        nb = (w + 7) // 8
        msg = b"".join(((v - rho) % (1 << w)).to_bytes(nb, "little") for v, rho in zip(my_values, my_masks.values))
    if peer is not None:
        if not isinstance(peer, InMaskMaterial) or peer.owner != 1 - role:
            raise ValueError("peer must be the peer's InMaskMaterial")
        if w is not None and peer.width != w:
            raise ValueError("the two mask sets must use the same w")
        w = peer.width
    if w is None:
        return [], []
    nb = (w + 7) // 8
    phase = chan.phase
    chan.mark(PHASE_INPUT)
    try:
        reply = chan.exchange(msg)
    finally:
        chan.mark(phase)
    expected = 0 if peer is None else peer.count * nb
    if len(reply) != expected:
        raise ProtocolAbort("input: message has the wrong length")
    my_shares = []
    if my_masks is not None:
        for (value, tag), v, rho in zip(my_masks.shares, my_values, my_masks.values):
            my_shares.append(AShare(value, tag, w).add_public((v - rho) % (1 << w), role, opener.delta_share))
    peer_shares = []
    if peer is not None:
        for j, (value, tag) in enumerate(peer.shares):
            d = int.from_bytes(reply[j * nb:(j + 1) * nb], "little")
            if d >> w:
                raise ProtocolAbort("input: value out of range")
            peer_shares.append(AShare(value, tag, w).add_public(d, role, opener.delta_share))
    return my_shares, peer_shares
