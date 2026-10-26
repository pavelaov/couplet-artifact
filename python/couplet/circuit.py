"""Gate-list IR for Boolean circuits (DESIGN.md sections 4 and 8: ``circuit.py``).

Wires are non-negative integers.  Wire 0 is the constant 0 and wire 1 the
constant 1; all other wires are produced either by an **input** declaration
(owner, label, width) or by a gate.  Gates are

    XOR(a, b) -> out        op code 0 (free in the Boolean engine)
    AND(a, b) -> out        op code 1 (one authenticated triple)
    NOT(a)    -> out        op code 2 (free; equivalent to XOR with wire 1)

Bit vectors are *little-endian* Python lists of wire ids: ``bits[0]`` is the
least significant bit.

Input owners (the first field of an input declaration):

    0, 1     private input of party P_0 / P_1 (engine operation ``Input``);
    BOUND    (= 2) wires supplied by the engine as EXISTING authenticated
             bits of the session, e.g. the session MAC key ``delta``; no
             party inputs anything for them.

Output owners (the first field of an output declaration):

    PUBLIC   ("public") opened to both parties;
    0, 1     opened privately to P_0 / P_1;
    RETAIN   ("retain") NOT opened: the wires stay authenticated bits of the
             session and can be bound to a BOUND input of a later circuit.

The ``Builder`` accumulates gates with light constant folding (XOR/AND/NOT
with wire 0/1, ``x ^ x``, ``x & x``, ``x & ~x``, double negation).  The
folding is what makes a public constant free: e.g. an Ascon call whose IV
words are constants costs fewer AND gates than a call on 320 fresh wires.

``Builder.build()`` returns an immutable ``Circuit``:

* dead gates (not reachable from any output) are removed;
* gates are topologically ordered and grouped by AND depth:
  ``[linear depth 0] [AND layer 1] [linear depth 1] [AND layer 2] ...``
  so an engine can evaluate all AND gates of one layer in one round after
  evaluating the (free) linear gates that precede it (``Circuit.layers()``,
  ``Circuit.schedule()``);
* wires are renumbered: 0, 1, then the input wires in declaration order, then
  gate outputs in gate order (gate ``i`` drives wire ``2 + n_inputs + i``).

Serialisation (format version 2; version 1 had neither BOUND nor RETAIN):

* ``to_bytes``/``from_bytes``: canonical binary, little-endian, magic
  ``CPLTCIR2``; ``digest`` is SHA-256 of ``to_bytes`` and is what the two
  parties compare before running a circuit;
* ``to_text``/``from_text``: line-oriented text with our labels and layers;
* ``to_bristol``: Bristol Fashion export for other frameworks (inputs
  ordered party 0, party 1, bound; see ``Circuit.to_bristol``), with
  ``bristol_layout`` describing where each label sits.

Plaintext evaluation: ``evaluate(circuit, inputs)`` for one assignment and
``evaluate_batch(circuit, [inputs, ...])`` which packs any number of
assignments into Python-int bit masks (bit-parallel evaluation).  Values for
BOUND inputs are passed like any other input, under the key
``(BOUND, label)``; RETAIN outputs are returned like any other output.
"""

from __future__ import annotations

import hashlib
import sys
from array import array
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

XOR, AND, NOT = 0, 1, 2
OP_NAMES = {XOR: "XOR", AND: "AND", NOT: "INV"}
OP_FROM_NAME = {"XOR": XOR, "AND": AND, "INV": NOT, "NOT": NOT}

ZERO, ONE = 0, 1          # reserved constant wires
BOUND = 2                 # input owner: existing authenticated bits of the session
PUBLIC = "public"         # output owner: opened to both parties
RETAIN = "retain"         # output owner: kept as authenticated bits, not opened

INPUT_OWNERS = (0, 1, BOUND)
OUTPUT_OWNERS = (PUBLIC, 0, 1, RETAIN)

FORMAT_VERSION = 2
_MAGIC = b"CPLTCIR2"
_U32 = "I" if array("I").itemsize == 4 else "L"
_BIG_ENDIAN = sys.byteorder == "big"

# binary / text codes of the owners
_OUT_CODE = {0: 0, 1: 1, PUBLIC: 2, RETAIN: 3}
_OUT_FROM_CODE = {0: 0, 1: 1, 2: PUBLIC, 3: RETAIN}
_IN_TEXT = {0: "0", 1: "1", BOUND: "bound"}
_IN_FROM_TEXT = {"0": 0, "1": 1, "bound": BOUND, "2": BOUND}
_OUT_TEXT = {0: "0", 1: "1", PUBLIC: PUBLIC, RETAIN: RETAIN}
_OUT_FROM_TEXT = {"0": 0, "1": 1, PUBLIC: PUBLIC, RETAIN: RETAIN}

Bits = List[int]
Owner = Union[int, str]


def _u32array(init=()) -> array:
    return array(_U32, init)


# --------------------------------------------------------------------------- #
# Bit-vector utilities (plain Python ints <-> little-endian bit lists)
# --------------------------------------------------------------------------- #

def int_to_bits(value: int, width: int) -> List[int]:
    """Little-endian list of ``width`` bits of ``value`` (mod 2^width)."""
    if value < 0:
        value %= 1 << width
    return [(value >> i) & 1 for i in range(width)]


def bits_to_int(bits: Sequence[int]) -> int:
    """Integer from a little-endian bit list."""
    v = 0
    for i, b in enumerate(bits):
        if b:
            v |= 1 << i
    return v


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

class Builder:
    """Incremental circuit builder with constant folding.

    ``adder`` selects the default adder architecture used by ``add_mod`` and
    everything built on it: ``"ripple"`` (one AND per bit, depth = width) or
    ``"sklansky"`` (parallel prefix, ~w*log2(w) ANDs, depth ~ log2(w)+1).
    """

    def __init__(self, adder: str = "ripple") -> None:
        if adder not in ("ripple", "sklansky"):
            raise ValueError("adder must be 'ripple' or 'sklansky'")
        self.adder = adder
        self._n = 2                       # wires 0 and 1 are the constants
        self._op = bytearray()
        self._a = _u32array()
        self._b = _u32array()
        self._out = _u32array()
        # _inv[w] = wire holding NOT w, or -1.  Signed 32-bit array.
        self._inv = array("i", [1, 0])
        self.inputs: List[Tuple[int, str, List[int]]] = []
        self.outputs: List[Tuple[Owner, str, List[int]]] = []
        self._input_keys: set = set()
        self._output_labels: set = set()

    # -- wires ----------------------------------------------------------- #

    def _new_wire(self) -> int:
        w = self._n
        self._n += 1
        self._inv.append(-1)
        return w

    @property
    def n_wires(self) -> int:
        return self._n

    @property
    def n_gates(self) -> int:
        return len(self._op)

    def n_and(self) -> int:
        """AND gates emitted so far (before dead-code elimination)."""
        return self._op.count(AND)

    # -- declarations ---------------------------------------------------- #

    def input(self, party: int, label: str, width: int) -> List[int]:
        """Declare ``width`` input bits owned by ``party`` (0, 1 or ``BOUND``);
        returns the wires.  Labels are unique per owner."""
        if party not in INPUT_OWNERS or isinstance(party, bool):
            raise ValueError("input owner must be 0, 1 or BOUND")
        _check_label(label)
        key = (party, label)
        if key in self._input_keys:
            raise ValueError(f"duplicate input {key}")
        if width < 1:
            raise ValueError("width must be >= 1")
        self._input_keys.add(key)
        wires = [self._new_wire() for _ in range(width)]
        self.inputs.append((party, label, wires))
        return wires

    def output(self, label: str, wires: Union[int, Sequence[int]], to: Owner = PUBLIC) -> None:
        """Declare an output (``to`` is ``PUBLIC``, 0, 1 or ``RETAIN``).  Labels are unique."""
        _check_label(label)
        if to not in OUTPUT_OWNERS or isinstance(to, bool):
            raise ValueError("output owner must be PUBLIC, 0, 1 or RETAIN")
        if label in self._output_labels:
            raise ValueError(f"duplicate output label {label!r}")
        if isinstance(wires, int):
            wires = [wires]
        wires = [int(w) for w in wires]
        for w in wires:
            if not 0 <= w < self._n:
                raise ValueError(f"unknown wire {w}")
        self._output_labels.add(label)
        self.outputs.append((to, label, wires))

    # -- primitive gates with folding ------------------------------------ #

    def xor(self, a: int, b: int) -> int:
        if a == ZERO:
            return b
        if b == ZERO:
            return a
        if a == ONE:
            return self.not_(b)
        if b == ONE:
            return self.not_(a)
        if a == b:
            return ZERO
        if self._inv[a] == b:
            return ONE
        out = self._new_wire()
        self._op.append(XOR)
        self._a.append(a)
        self._b.append(b)
        self._out.append(out)
        return out

    def and_(self, a: int, b: int) -> int:
        if a == ZERO or b == ZERO:
            return ZERO
        if a == ONE:
            return b
        if b == ONE:
            return a
        if a == b:
            return a
        if self._inv[a] == b:
            return ZERO
        out = self._new_wire()
        self._op.append(AND)
        self._a.append(a)
        self._b.append(b)
        self._out.append(out)
        return out

    def not_(self, a: int) -> int:
        r = self._inv[a]
        if r >= 0:
            return r
        out = self._new_wire()
        self._op.append(NOT)
        self._a.append(a)
        self._b.append(a)
        self._out.append(out)
        self._inv[a] = out
        self._inv[out] = a
        return out

    def or_(self, a: int, b: int) -> int:
        """a | b = a ^ b ^ (a & b)  (one AND)."""
        return self.xor(self.xor(a, b), self.and_(a, b))

    # -- vector helpers -------------------------------------------------- #

    def const_bits(self, value: int, width: int) -> List[int]:
        """Little-endian constant bit vector (wires 0/1)."""
        return [ONE if (value >> i) & 1 else ZERO for i in range(width)]

    def xor_bits(self, a: Sequence[int], b: Sequence[int]) -> List[int]:
        if len(a) != len(b):
            raise ValueError("xor_bits: width mismatch")
        return [self.xor(x, y) for x, y in zip(a, b)]

    def and_bits(self, sel: int, a: Sequence[int]) -> List[int]:
        """AND every bit of ``a`` with the single wire ``sel``."""
        return [self.and_(sel, x) for x in a]

    def not_bits(self, a: Sequence[int]) -> List[int]:
        return [self.not_(x) for x in a]

    def zero_extend(self, a: Sequence[int], width: int) -> List[int]:
        """Pad with constant 0 (or truncate) to ``width`` bits."""
        a = list(a[:width])
        return a + [ZERO] * (width - len(a))

    @staticmethod
    def rotr(a: Sequence[int], r: int) -> List[int]:
        """Rotate a little-endian word right by ``r``: out[j] = a[(j + r) % w]."""
        w = len(a)
        return [a[(j + r) % w] for j in range(w)]

    def mux(self, sel: int, a, b):
        """Return ``b`` if ``sel == 0`` else ``a`` (one AND per bit).

        ``a`` and ``b`` are both single wires or both equal-length lists.
        """
        if isinstance(a, int):
            return self.xor(b, self.and_(sel, self.xor(a, b)))
        if len(a) != len(b):
            raise ValueError("mux: width mismatch")
        return [self.xor(y, self.and_(sel, self.xor(x, y))) for x, y in zip(a, b)]

    # -- arithmetic mod 2^w ---------------------------------------------- #

    def add_mod(self, a: Sequence[int], b: Sequence[int], cin: Optional[int] = None,
                width: Optional[int] = None, adder: Optional[str] = None) -> Tuple[List[int], int]:
        """a + b (+ cin) mod 2^width.  Returns (sum_bits, carry_out).

        Operands are zero-extended/truncated to ``width`` (default: the wider
        operand).  Ripple-carry: full adder with one AND per bit,
        ``carry = b ^ ((a ^ b) & (b ^ c))``; the last carry is dead unless
        ``carry_out`` is used (dead gates are pruned by ``build``).
        Sklansky: parallel prefix, depth log2(width)+1 AND layers.
        """
        if width is None:
            width = max(len(a), len(b))
        a = self.zero_extend(a, width)
        b = self.zero_extend(b, width)
        if cin is None:
            cin = ZERO
        arch = adder or self.adder
        if arch == "ripple":
            return self._add_ripple(a, b, cin)
        if arch == "sklansky":
            return self._add_sklansky(a, b, cin)
        raise ValueError(f"unknown adder {arch!r}")

    def _add_ripple(self, a: Sequence[int], b: Sequence[int], cin: int) -> Tuple[List[int], int]:
        c = cin
        out = []
        for x, y in zip(a, b):
            t = self.xor(x, y)
            out.append(self.xor(t, c))
            c = self.xor(y, self.and_(t, self.xor(y, c)))
        return out, c

    def _add_sklansky(self, a: Sequence[int], b: Sequence[int], cin: int) -> Tuple[List[int], int]:
        p = [self.xor(x, y) for x, y in zip(a, b)]
        g = [self.and_(x, y) for x, y in zip(a, b)]
        # Position -1 carries cin as a "generate" with propagate 0.
        G = [cin] + g
        P = [ZERO] + p
        m = len(G)
        d = 0
        while (1 << d) < m:
            step = 1 << d
            for i in range(m):
                if i & step:
                    j = (i & ~(2 * step - 1)) + step - 1
                    G[i] = self.xor(G[i], self.and_(P[i], G[j]))
                    if i >> (d + 1):
                        P[i] = self.and_(P[i], P[j])
            d += 1
        # G[i] is now the carry out of position i-1 (prefix over [-1, i-1]).
        out = [self.xor(p[i], G[i]) for i in range(len(p))]
        return out, G[m - 1]

    def sub_mod(self, a: Sequence[int], b: Sequence[int], width: Optional[int] = None,
                adder: Optional[str] = None) -> Tuple[List[int], int]:
        """a - b mod 2^width.  Returns (diff_bits, carry_out) with
        carry_out = 1 iff a >= b (unsigned), i.e. NOT borrow."""
        if width is None:
            width = max(len(a), len(b))
        b = self.zero_extend(b, width)
        return self.add_mod(a, self.not_bits(b), cin=ONE, width=width, adder=adder)

    def neg_mod(self, a: Sequence[int], adder: Optional[str] = None) -> List[int]:
        """-a mod 2^len(a)  (~a + 1)."""
        w = len(a)
        s, _ = self.add_mod(self.not_bits(a), [ZERO] * w, cin=ONE, width=w, adder=adder)
        return s

    def cond_neg(self, sel: int, a: Sequence[int], adder: Optional[str] = None) -> List[int]:
        """-a if sel else a  (mod 2^len(a)):  (a ^ sel) + sel.  width-1 ANDs (ripple)."""
        w = len(a)
        x = [self.xor(t, sel) for t in a]
        s, _ = self.add_mod(x, [ZERO] * w, cin=sel, width=w, adder=adder)
        return s

    def neg_if(self, t: int, v: Sequence[int], adder: Optional[str] = None) -> List[int]:
        """``neg_if(t, v)`` of DESIGN.md section 3 (alias of ``cond_neg``)."""
        return self.cond_neg(t, v, adder=adder)

    def _sum_columns(self, cols: List[List[int]], const: int, width: int,
                     adder: Optional[str] = None) -> List[int]:
        """Sum of a bit matrix mod 2^width (Dadda reduction + one adder).

        ``cols[c]`` lists the non-constant wires of weight 2^c and ``const``
        is an integer added on top.  The matrix is reduced to two rows with
        Dadda's schedule: for the target heights ..., 9, 6, 4, 3, 2 in turn,
        every column is brought down to the target with full adders (three
        bits -> one sum bit and one carry, ``maj = y ^ ((x^y) & (y^z))``, one
        AND) and at most one half adder (one AND); carries count towards the
        next column in the same stage, and adders prefer bits of earlier
        stages, so a stage is one AND layer.  Carries out of the top column
        are dropped.  One adder finishes.
        """
        const %= 1 << width
        cin = ZERO
        if const & 1:                     # the low constant bit rides on the carry-in
            cin = ONE
            const -= 1
        cols = [list(c) for c in cols]
        for c in range(width):
            if (const >> c) & 1:
                cols[c].append(ONE)
        tallest = max((len(c) for c in cols), default=0)
        targets = [2]                     # Dadda heights 2, 3, 4, 6, 9, 13, ...
        while targets[-1] < tallest:
            targets.append(targets[-1] * 3 // 2)
        for d in reversed(targets[:-1]):  # all targets below the tallest column
            carry_in: List[int] = []
            for c in range(width):
                pool = cols[c] + carry_in         # bits of earlier stages first
                carry_in = []
                sums: List[int] = []
                i = 0
                height = len(pool)
                while height > d:
                    need = 2 if height == d + 1 else 3
                    if len(pool) - i < need:      # fall back to this stage's sums
                        pool.extend(sums)
                        sums = []
                    if need == 2:                 # half adder: height - 1
                        x, y = pool[i], pool[i + 1]
                        sm = self.xor(x, y)
                        cy = self.and_(x, y) if c + 1 < width else ZERO
                        height -= 1
                    else:                         # full adder: height - 2
                        x, y, z = pool[i], pool[i + 1], pool[i + 2]
                        p = self.xor(x, y)
                        sm = self.xor(p, z)
                        cy = self.xor(y, self.and_(p, self.xor(y, z))) if c + 1 < width else ZERO
                        height -= 2
                    i += need
                    if sm == ZERO:
                        height -= 1
                    else:
                        sums.append(sm)
                    if cy != ZERO:
                        carry_in.append(cy)
                cols[c] = pool[i:] + sums
        row0 = [c[0] if len(c) > 0 else ZERO for c in cols]
        row1 = [c[1] if len(c) > 1 else ZERO for c in cols]
        s, _ = self.add_mod(row0, row1, cin=cin, width=width, adder=adder)
        return s

    def sum_mod(self, terms: Sequence[Tuple[Sequence[int], bool]], width: int,
                adder: Optional[str] = None) -> List[int]:
        """Multi-operand sum mod 2^width.

        ``terms`` is a list of ``(bits, negate)``; negated terms contribute
        ``-bits`` (as ``~bits + 1`` over ``width`` bits).  Operands are
        zero-extended or truncated to ``width``; constant bits are collected
        into one constant.  Three operands cost one full-adder row plus one
        adder: 2 (width - 1) ANDs with the ripple adder.
        """
        cols: List[List[int]] = [[] for _ in range(width)]
        const = 0
        for bits, neg in terms:
            row = self.zero_extend(bits, width)
            if neg:
                const += 1
            for c, wire in enumerate(row):
                if neg:                   # contributes NOT wire
                    if wire == ZERO:
                        const += 1 << c
                    elif wire != ONE:
                        cols[c].append(self.not_(wire))
                elif wire == ONE:
                    const += 1 << c
                elif wire != ZERO:
                    cols[c].append(wire)
        return self._sum_columns(cols, const, width, adder=adder)

    def mul(self, a: Sequence[int], b: Sequence[int], out_width: int,
            adder: Optional[str] = None) -> List[int]:
        """a * b mod 2^out_width: truncated schoolbook multiplier.

        Partial products ``a_i & b_j`` with ``i + j < out_width`` (one AND
        each), accumulated by the Dadda reduction of ``_sum_columns`` and one
        final adder.  With P partial products the cost is about
        ``2 P - out_width`` ANDs for the ripple adder (a full adder removes
        one bit of the matrix); e.g. 64 x 64 -> 128 costs about 8.1k ANDs.
        Depth: 1 + about log_{1.5}(min(len(a), len(b))) + adder depth.
        """
        cols: List[List[int]] = [[] for _ in range(out_width)]
        const = 0
        for j, bj in enumerate(b):
            if j >= out_width or bj == ZERO:
                continue
            for i, ai in enumerate(a[:out_width - j]):
                pp = self.and_(ai, bj)
                if pp == ONE:
                    const += 1 << (i + j)
                elif pp != ZERO:
                    cols[i + j].append(pp)
        return self._sum_columns(cols, const, out_width, adder=adder)

    # -- finalisation ---------------------------------------------------- #

    def stats(self) -> Dict[str, int]:
        """Statistics of the *built* circuit (dead gates removed)."""
        return self.build().stats()

    def build(self) -> "Circuit":
        """Prune dead gates, order by AND layers, renumber, and freeze."""
        n = self._n
        op, ga, gb, go = self._op, self._a, self._b, self._out
        ng = len(op)

        # 1. liveness (backward pass over a topologically ordered list)
        live = bytearray(n)
        for _, _, wires in self.outputs:
            for w in wires:
                live[w] = 1
        for g in range(ng - 1, -1, -1):
            if live[go[g]]:
                live[ga[g]] = 1
                live[gb[g]] = 1

        # 2. depth and grouping
        depth = array("I", bytes(4 * n))
        groups: Dict[int, array] = {}
        maxd = 0
        for g in range(ng):
            o = go[g]
            if not live[o]:
                continue
            code = op[g]
            da = depth[ga[g]]
            if code == NOT:
                d = da
                grp = 2 * d
            else:
                db = depth[gb[g]]
                d = da if da > db else db
                if code == AND:
                    d += 1
                    grp = 2 * d - 1
                else:
                    grp = 2 * d
            depth[o] = d
            if d > maxd:
                maxd = d
            arr = groups.get(grp)
            if arr is None:
                arr = groups[grp] = _u32array()
            arr.append(g)

        # 3. renumbering
        newid = array("i", [-1]) * n
        newid[0] = 0
        newid[1] = 1
        nxt = 2
        new_inputs = []
        for party, label, wires in self.inputs:
            nw = []
            for w in wires:
                newid[w] = nxt
                nw.append(nxt)
                nxt += 1
            new_inputs.append((party, label, nw))
        n_input_wires = nxt - 2

        nop = bytearray()
        na = _u32array()
        nb = _u32array()
        nout = _u32array()
        layers: List[Tuple[int, int]] = []
        for grp in sorted(groups):
            arr = groups[grp]
            start = len(nop)
            for g in arr:
                newid[go[g]] = nxt
                nout.append(nxt)
                nxt += 1
                nop.append(op[g])
                na.append(newid[ga[g]])
                nb.append(newid[gb[g]])
            if grp & 1:
                layers.append((start, len(nop)))
        assert len(layers) == maxd

        new_outputs = []
        for to, label, wires in self.outputs:
            new_outputs.append((to, label, [newid[w] for w in wires]))

        return Circuit(nxt, nop, na, nb, nout, new_inputs, new_outputs, layers, n_input_wires)


def _check_label(label: str) -> None:
    if not isinstance(label, str) or not label or any(ch.isspace() for ch in label):
        raise ValueError(f"bad label {label!r}: must be non-empty without whitespace")


# --------------------------------------------------------------------------- #
# Circuit
# --------------------------------------------------------------------------- #

class Circuit:
    """Immutable, topologically ordered, layer-grouped gate list."""

    def __init__(self, n_wires: int, op: bytearray, a: array, b: array, out: array,
                 inputs: List[Tuple[int, str, List[int]]],
                 outputs: List[Tuple[Owner, str, List[int]]],
                 layers: List[Tuple[int, int]], n_input_wires: int) -> None:
        self.n_wires = n_wires
        self.op = op
        self.a = a
        self.b = b
        self.out = out
        self.inputs = inputs
        self.outputs = outputs
        self._layers = layers
        self.n_input_wires = n_input_wires
        self.meta: Dict = {}
        self._input_index = {(p, l): i for i, (p, l, _) in enumerate(inputs)}
        self._output_index = {l: i for i, (_, l, _) in enumerate(outputs)}

    # -- queries --------------------------------------------------------- #

    @property
    def n_gates(self) -> int:
        return len(self.op)

    def gate(self, i: int) -> Tuple[int, int, int, int]:
        return self.op[i], self.a[i], self.b[i], self.out[i]

    def gates(self) -> Iterable[Tuple[int, int, int, int]]:
        return zip(self.op, self.a, self.b, self.out)

    def layer_ranges(self) -> List[Tuple[int, int]]:
        """[(start, end)] gate-index ranges of the AND layers, in order."""
        return list(self._layers)

    def layers(self) -> List[List[int]]:
        """AND gate indices grouped by AND layer.  All XOR/NOT gates with a
        smaller index are evaluable before the layer (gate order is
        ``[linear d=0][AND layer 1][linear d=1][AND layer 2]...``)."""
        return [list(range(s, e)) for s, e in self._layers]

    def schedule(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """[(linear_range, and_range)] per layer plus a final (linear, empty)."""
        res = []
        prev = 0
        for s, e in self._layers:
            res.append(((prev, s), (s, e)))
            prev = e
        res.append(((prev, self.n_gates), (self.n_gates, self.n_gates)))
        return res

    def and_depth(self) -> int:
        return len(self._layers)

    def input_wires(self, party: int, label: str) -> List[int]:
        return self.inputs[self._input_index[(party, label)]][2]

    def has_input(self, party: int, label: str) -> bool:
        return (party, label) in self._input_index

    def output_wires(self, label: str) -> List[int]:
        return self.outputs[self._output_index[label]][2]

    def output_owner(self, label: str) -> Owner:
        return self.outputs[self._output_index[label]][0]

    def has_output(self, label: str) -> bool:
        return label in self._output_index

    def input_bits(self, party: int) -> int:
        """Total number of input wires owned by ``party`` (0, 1 or ``BOUND``)."""
        return sum(len(w) for p, _, w in self.inputs if p == party)

    def output_bits(self, to: Owner) -> int:
        """Total number of output wires with owner ``to``."""
        return sum(len(w) for t, _, w in self.outputs if _same_owner(t, to))

    def stats(self) -> Dict[str, int]:
        op = self.op
        n_and = op.count(AND)
        n_xor = op.count(XOR)
        n_not = op.count(NOT)
        return {
            "n_and": n_and,
            "n_xor": n_xor,
            "n_not": n_not,
            "n_gates": len(op),
            "and_depth": len(self._layers),
            "n_wires": self.n_wires,
            "n_inputs": self.n_input_wires,
            "n_input_decls": len(self.inputs),
            "n_input_bits_p0": self.input_bits(0),
            "n_input_bits_p1": self.input_bits(1),
            "n_bound_input_bits": self.input_bits(BOUND),
            "n_outputs": sum(len(w) for _, _, w in self.outputs),
            "n_output_decls": len(self.outputs),
            "n_public_output_bits": self.output_bits(PUBLIC),
            "n_private_output_bits": self.output_bits(0) + self.output_bits(1),
            "n_private_output_bits_p0": self.output_bits(0),
            "n_private_output_bits_p1": self.output_bits(1),
            "n_retain_output_bits": self.output_bits(RETAIN),
        }

    # -- validation ------------------------------------------------------ #

    def validate(self) -> None:
        """Check the canonical form produced by ``Builder.build``.

        Wire numbering (0, 1, inputs in declaration order, then gate ``i`` on
        wire ``2 + n_inputs + i``), topological gate order, owners, unique
        labels, and the layer structure: AND layers are disjoint increasing
        ranges that contain exactly the AND gates, and both operands of an AND
        gate are produced before its layer starts (so a layer is one round).
        Raises ``ValueError`` on the first violation.
        """
        ng = len(self.op)
        if not (len(self.a) == len(self.b) == len(self.out) == ng):
            raise ValueError("gate arrays have different lengths")
        nxt = 2
        seen = set()
        for party, label, wires in self.inputs:
            if party not in INPUT_OWNERS:
                raise ValueError(f"input {label!r}: bad owner {party!r}")
            _check_label(label)
            if (party, label) in seen or not wires:
                raise ValueError(f"input {label!r}: duplicate or empty")
            seen.add((party, label))
            if list(wires) != list(range(nxt, nxt + len(wires))):
                raise ValueError(f"input {label!r}: wires are not numbered consecutively")
            nxt += len(wires)
        if nxt - 2 != self.n_input_wires:
            raise ValueError("n_input_wires does not match the input declarations")
        if self.n_wires != nxt + ng:
            raise ValueError("n_wires does not match inputs and gates")
        base = nxt
        prev_end = 0
        for s, e in self._layers:
            if not prev_end <= s < e <= ng:
                raise ValueError("AND layers must be non-empty, increasing and disjoint")
            prev_end = e
        op, ga, gb, go = self.op, self.a, self.b, self.out
        layer_iter = iter(self._layers)
        cur = next(layer_iter, None)
        for i in range(ng):
            while cur is not None and i >= cur[1]:
                cur = next(layer_iter, None)
            code = op[i]
            in_layer = cur is not None and cur[0] <= i
            limit = base + (cur[0] if in_layer else i)
            if go[i] != base + i:
                raise ValueError(f"gate {i}: output wire is not 2 + n_inputs + index")
            if code == AND:
                if not in_layer:
                    raise ValueError(f"gate {i}: AND gate outside every AND layer")
            elif code in (XOR, NOT):
                if in_layer:
                    raise ValueError(f"gate {i}: linear gate inside an AND layer")
                if code == NOT and ga[i] != gb[i]:
                    raise ValueError(f"gate {i}: NOT gate must repeat its operand")
            else:
                raise ValueError(f"gate {i}: unknown op code {code}")
            if ga[i] >= limit or gb[i] >= limit:
                raise ValueError(f"gate {i}: operand not available")
        labels = set()
        for to, label, wires in self.outputs:
            if to not in OUTPUT_OWNERS:
                raise ValueError(f"output {label!r}: bad owner {to!r}")
            _check_label(label)
            if label in labels or not wires:
                raise ValueError(f"output {label!r}: duplicate or empty")
            labels.add(label)
            if any(not 0 <= w < self.n_wires for w in wires):
                raise ValueError(f"output {label!r}: unknown wire")

    # -- serialisation --------------------------------------------------- #

    def to_bytes(self) -> bytes:
        """Canonical binary encoding, format version 2 (all integers u32 little-endian).

        magic "CPLTCIR2" (8 bytes) |
        n_wires | n_gates | n_input_decls | n_output_decls | n_layers |
        gates:   n_gates x (op, a, b, out)      op: 0 XOR, 1 AND, 2 NOT (b = a)
        layers:  n_layers x (start, end)        gate-index ranges of the AND layers
        inputs:  n_input_decls x (owner, label_len, label utf-8, width, wires[width])
                                                owner: 0, 1, 2 = BOUND
        outputs: n_output_decls x (owner, label_len, label utf-8, width, wires[width])
                                                owner: 0, 1, 2 = PUBLIC, 3 = RETAIN
        """
        parts = [_MAGIC, _pack_u32([self.n_wires, self.n_gates, len(self.inputs),
                                    len(self.outputs), len(self._layers)])]
        ng = self.n_gates
        flat = _u32array(bytes(4 * 4 * ng))          # 4*ng zero entries
        flat[0::4] = _u32array(list(self.op))        # element-wise, not raw bytes
        flat[1::4] = self.a
        flat[2::4] = self.b
        flat[3::4] = self.out
        parts.append(_array_bytes(flat))
        lay = _u32array()
        for s, e in self._layers:
            lay.append(s)
            lay.append(e)
        parts.append(_array_bytes(lay))
        for party, label, wires in self.inputs:
            lb = label.encode("utf-8")
            parts.append(_pack_u32([_input_owner_code(party), len(lb)]))
            parts.append(lb)
            parts.append(_pack_u32([len(wires)] + list(wires)))
        for to, label, wires in self.outputs:
            lb = label.encode("utf-8")
            parts.append(_pack_u32([_owner_code(to), len(lb)]))
            parts.append(lb)
            parts.append(_pack_u32([len(wires)] + list(wires)))
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes, validate: bool = True) -> "Circuit":
        """Decode ``to_bytes`` output; ``validate`` runs ``Circuit.validate``."""
        data = bytes(data)
        if data[:7] == _MAGIC[:7] and data[:8] != _MAGIC:
            raise ValueError(f"unsupported circuit format version {data[7:8]!r} (expected {_MAGIC[7:8]!r})")
        if data[:8] != _MAGIC:
            raise ValueError("bad magic")
        pos = 8
        n_wires, ng, ni, no, nl = _unpack_u32(data, pos, 5)
        pos += 20
        if len(data) < pos + 16 * ng + 8 * nl:
            raise ValueError("truncated data")
        flat = _u32array()
        flat.frombytes(data[pos:pos + 16 * ng])
        if _BIG_ENDIAN:
            flat.byteswap()
        pos += 16 * ng
        codes = flat[0::4]
        if ng and max(codes) > NOT:
            raise ValueError("unknown op code")
        op = bytearray(codes.tolist())               # element-wise, not raw bytes
        a = flat[1::4]
        b = flat[2::4]
        out = flat[3::4]
        lay = _u32array()
        lay.frombytes(data[pos:pos + 8 * nl])
        if _BIG_ENDIAN:
            lay.byteswap()
        pos += 8 * nl
        layers = [(lay[2 * i], lay[2 * i + 1]) for i in range(nl)]

        def read_decl(pos: int):
            code, ln = _unpack_u32(data, pos, 2)
            pos += 8
            if pos + ln > len(data):
                raise ValueError("truncated data")
            label = data[pos:pos + ln].decode("utf-8")
            pos += ln
            (width,) = _unpack_u32(data, pos, 1)
            pos += 4
            wires = list(_unpack_u32(data, pos, width))
            pos += 4 * width
            return code, label, wires, pos

        inputs = []
        n_input_wires = 0
        for _ in range(ni):
            code, label, wires, pos = read_decl(pos)
            if code not in INPUT_OWNERS:
                raise ValueError(f"bad input owner code {code}")
            inputs.append((code, label, wires))
            n_input_wires += len(wires)
        outputs = []
        for _ in range(no):
            code, label, wires, pos = read_decl(pos)
            outputs.append((_owner_from_code(code), label, wires))
        if pos != len(data):
            raise ValueError("trailing bytes")
        circuit = cls(n_wires, op, a, b, out, inputs, outputs, layers, n_input_wires)
        if validate:
            circuit.validate()
        return circuit

    def digest(self) -> str:
        """SHA-256 hex digest of the canonical binary encoding."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_text(self) -> str:
        """Line-oriented text form with our labels and layers (format version 2).

        ``version 2``
        ``header <n_wires> <n_gates> <n_and> <n_xor> <n_not> <and_depth>``
        ``input <0|1|bound> <label> <width> <wires...>``
        ``output <public|0|1|retain> <label> <width> <wires...>``
        ``layer <index> <first_gate> <end_gate>``  (AND layers, gate index ranges)
        ``gates`` followed by ``2 1 a b out XOR`` / ``2 1 a b out AND`` / ``1 1 a out INV``.
        Wires 0 and 1 are the constants 0 and 1.  Lines starting with ``#`` are comments.
        """
        st = self.stats()
        lines = ["# couplet circuit text; wires 0/1 are constants 0/1; bit vectors little-endian",
                 f"version {FORMAT_VERSION}",
                 f"header {self.n_wires} {self.n_gates} {st['n_and']} {st['n_xor']} {st['n_not']} {st['and_depth']}"]
        for party, label, wires in self.inputs:
            lines.append(f"input {_IN_TEXT[party]} {label} {len(wires)} " + " ".join(map(str, wires)))
        for to, label, wires in self.outputs:
            lines.append(f"output {_OUT_TEXT[to]} {label} {len(wires)} " + " ".join(map(str, wires)))
        for i, (s, e) in enumerate(self._layers):
            lines.append(f"layer {i} {s} {e}")
        lines.append("gates")
        for code, a, b, o in self.gates():
            if code == NOT:
                lines.append(f"1 1 {a} {o} INV")
            else:
                lines.append(f"2 1 {a} {b} {o} {OP_NAMES[code]}")
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_text(cls, text: str, validate: bool = True) -> "Circuit":
        n_wires = None
        version = None
        inputs: List[Tuple[int, str, List[int]]] = []
        outputs: List[Tuple[Owner, str, List[int]]] = []
        layers: List[Tuple[int, int]] = []
        op = bytearray()
        a = _u32array()
        b = _u32array()
        out = _u32array()
        in_gates = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if in_gates:
                f = line.split()
                if f[0] == "1":
                    if f[-1] not in ("INV", "NOT"):
                        raise ValueError(f"unknown gate {line!r}")
                    op.append(NOT)
                    a.append(int(f[2]))
                    b.append(int(f[2]))
                    out.append(int(f[3]))
                else:
                    code = OP_FROM_NAME.get(f[5])
                    if code not in (XOR, AND):
                        raise ValueError(f"unknown gate {line!r}")
                    op.append(code)
                    a.append(int(f[2]))
                    b.append(int(f[3]))
                    out.append(int(f[4]))
                continue
            f = line.split()
            if f[0] == "version":
                version = int(f[1])
                if version != FORMAT_VERSION:
                    raise ValueError(f"unsupported circuit text version {version}")
            elif f[0] == "header":
                n_wires = int(f[1])
            elif f[0] == "input":
                width = int(f[3])
                if f[1] not in _IN_FROM_TEXT or len(f) != 4 + width:
                    raise ValueError(f"bad input line {line!r}")
                inputs.append((_IN_FROM_TEXT[f[1]], f[2], [int(x) for x in f[4:]]))
            elif f[0] == "output":
                width = int(f[3])
                if f[1] not in _OUT_FROM_TEXT or len(f) != 4 + width:
                    raise ValueError(f"bad output line {line!r}")
                outputs.append((_OUT_FROM_TEXT[f[1]], f[2], [int(x) for x in f[4:]]))
            elif f[0] == "layer":
                layers.append((int(f[2]), int(f[3])))
            elif f[0] == "gates":
                in_gates = True
            else:
                raise ValueError(f"unknown line {line!r}")
        if version is None:
            raise ValueError("missing version line")
        if n_wires is None:
            raise ValueError("missing header")
        n_input_wires = sum(len(w) for _, _, w in inputs)
        circuit = cls(n_wires, op, a, b, out, inputs, outputs, layers, n_input_wires)
        if validate:
            circuit.validate()
        return circuit

    # -- Bristol Fashion export ------------------------------------------ #

    def bristol_layout(self) -> Dict:
        """Where each label sits in the ``to_bristol`` wire numbering.

        ``inputs``: one entry per input value (group), in file order: owner
        0, then 1, then ``"bound"``; groups without wires are omitted.  Each
        group lists its ``fields`` (label, width, offset inside the group) in
        declaration order.  ``outputs``: one entry per output declaration in
        declaration order (label, owner, width, offset inside the output
        block, which is the last ``n_output_wires`` wires of the file).
        Bit vectors are little-endian: the first wire is the least
        significant bit.
        """
        groups = []
        for owner in INPUT_OWNERS:
            fields = []
            off = 0
            for party, label, wires in self.inputs:
                if party == owner:
                    fields.append({"label": label, "width": len(wires), "offset": off})
                    off += len(wires)
            if off:
                groups.append({"owner": _IN_TEXT[owner] if owner == BOUND else owner,
                               "width": off, "fields": fields})
        outs = []
        off = 0
        for to, label, wires in self.outputs:
            outs.append({"label": label, "owner": to, "width": len(wires), "offset": off})
            off += len(wires)
        return {"inputs": groups, "outputs": outs, "n_input_wires": self.n_input_wires,
                "n_output_wires": off, "bit_order": "little-endian (first wire = least significant bit)"}

    def to_bristol(self) -> str:
        """Bristol Fashion text for use with other frameworks.

        Header: ``<n_gates> <n_wires>`` / ``<niv> <width of each input value>``
        / ``<nov> <width of each output value>`` / blank line, then one gate
        per line: ``2 1 a b out XOR``, ``2 1 a b out AND``, ``1 1 a out INV``.
        Only these three gate types are emitted.

        Input values (in this order, empty ones omitted): all input bits of
        party 0, all input bits of party 1, all BOUND bits; inside a value the
        declarations keep their order and each bit vector is little-endian.
        Output values: one per output declaration, in declaration order; as
        the format requires, they occupy the last wires of the circuit.
        ``bristol_layout`` gives labels, owners and offsets (Bristol Fashion
        has no place for them).

        The format has no constant wires: when a constant or a copy is needed
        (an output that is a constant, an input wire, or a wire already used
        by an earlier output position) the export emits ``z = XOR(w0, w0)``
        on the first input wire (and ``INV z`` for constant 1) and copies
        with ``XOR(x, z)``.  Gate order is the engine's layer order.
        """
        n_in = self.n_input_wires
        if n_in == 0:
            raise ValueError("Bristol Fashion export needs at least one input wire")
        ng = self.n_gates
        base = 2 + n_in                               # our wire id of gate 0
        newid = array("q", [-1]) * self.n_wires
        nxt = 0
        widths_in = []
        for owner in INPUT_OWNERS:
            tot = 0
            for party, _, wires in self.inputs:
                if party == owner:
                    for w in wires:
                        newid[w] = nxt
                        nxt += 1
                    tot += len(wires)
            if tot:
                widths_in.append(tot)
        out_wires = [w for _, _, wires in self.outputs for w in wires]
        n_out = len(out_wires)
        if n_out == 0:
            raise ValueError("Bristol Fashion export needs at least one output wire")
        # output positions served directly by a gate (first use of that gate's wire)
        direct = bytearray(n_out)
        claimed = set()
        for j, w in enumerate(out_wires):
            if w >= base and w not in claimed:
                claimed.add(w)
                direct[j] = 1
        n_copy = n_out - sum(direct)
        # ``x in array`` is a C-speed scan; built circuits never reference constants in gates
        need_one = ONE in out_wires or ONE in self.a or ONE in self.b
        need_zero = n_copy > 0 or need_one or ZERO in self.a or ZERO in self.b
        n_const = (1 if need_zero else 0) + (1 if need_one else 0)
        n_internal = ng - len(claimed)
        total = n_in + n_const + n_internal + n_out
        out_base = total - n_out
        lines: List[str] = []
        if n_const:
            newid[ZERO] = nxt
            lines.append(f"2 1 0 0 {nxt} XOR")
            nxt += 1
            if need_one:
                newid[ONE] = nxt
                lines.append(f"1 1 {nxt - 1} {nxt} INV")
                nxt += 1
        for j, w in enumerate(out_wires):
            if direct[j]:
                newid[w] = out_base + j
        for i in range(ng):
            o = base + i
            if newid[o] < 0:
                newid[o] = nxt
                nxt += 1
        assert nxt == out_base
        for code, a, b, o in self.gates():
            if code == NOT:
                lines.append(f"1 1 {newid[a]} {newid[o]} INV")
            else:
                lines.append(f"2 1 {newid[a]} {newid[b]} {newid[o]} {OP_NAMES[code]}")
        for j, w in enumerate(out_wires):
            if not direct[j]:
                lines.append(f"2 1 {newid[w]} {newid[ZERO]} {out_base + j} XOR")
        head = [f"{len(lines)} {total}",
                f"{len(widths_in)} " + " ".join(map(str, widths_in)),
                f"{len(self.outputs)} " + " ".join(str(len(w)) for _, _, w in self.outputs),
                ""]
        return "\n".join(head + lines) + "\n"

    # -- evaluation ------------------------------------------------------ #

    def evaluate(self, inputs: Mapping[Tuple[int, str], Union[int, Sequence[int]]]) -> Dict[str, List[int]]:
        return evaluate(self, inputs)


def _same_owner(a: Owner, b: Owner) -> bool:
    """Owner equality that never confuses the int owners with the str owners."""
    return isinstance(a, str) == isinstance(b, str) and a == b


def _owner_code(to: Owner) -> int:
    return _OUT_CODE[to]


def _owner_from_code(code: int) -> Owner:
    if code not in _OUT_FROM_CODE:
        raise ValueError(f"bad output owner code {code}")
    return _OUT_FROM_CODE[code]


def _input_owner_code(party: int) -> int:
    if party not in INPUT_OWNERS:
        raise ValueError(f"bad input owner {party!r}")
    return int(party)


def _pack_u32(values: Sequence[int]) -> bytes:
    return _array_bytes(_u32array(values))


def _array_bytes(arr: array) -> bytes:
    if _BIG_ENDIAN:
        arr = array(arr.typecode, arr)
        arr.byteswap()
    return arr.tobytes()


def _unpack_u32(data: bytes, pos: int, count: int) -> Tuple[int, ...]:
    arr = _u32array()
    arr.frombytes(data[pos:pos + 4 * count])
    if _BIG_ENDIAN:
        arr.byteswap()
    if len(arr) != count:
        raise ValueError("truncated data")
    return tuple(arr)


# --------------------------------------------------------------------------- #
# Plaintext evaluation
# --------------------------------------------------------------------------- #

def _coerce_bits(value: Union[int, Sequence[int]], width: int, key) -> List[int]:
    if isinstance(value, int):
        if value < 0 or value >> width:
            raise ValueError(f"input {key}: value does not fit in {width} bits")
        return int_to_bits(value, width)
    bits = [int(v) for v in value]
    if len(bits) != width:
        raise ValueError(f"input {key}: expected {width} bits, got {len(bits)}")
    if any(v not in (0, 1) for v in bits):
        raise ValueError(f"input {key}: non-binary value")
    return bits


def _run(circuit: Circuit, vals: List[int], mask: int) -> None:
    for code, a, b, o in zip(circuit.op, circuit.a, circuit.b, circuit.out):
        if code == 0:
            vals[o] = vals[a] ^ vals[b]
        elif code == 1:
            vals[o] = vals[a] & vals[b]
        else:
            vals[o] = vals[a] ^ mask


def evaluate(circuit: Circuit, inputs: Mapping[Tuple[int, str], Union[int, Sequence[int]]]) -> Dict[str, List[int]]:
    """Evaluate in plaintext.  ``inputs[(owner, label)]`` is an int or a
    little-endian bit list of the declared width; ``owner`` is 0, 1 or
    ``BOUND``.  Returns ``{label: bits}`` for every declared output (public,
    private and retained alike)."""
    vals = [0] * circuit.n_wires
    vals[ONE] = 1
    for party, label, wires in circuit.inputs:
        key = (party, label)
        if key not in inputs:
            raise KeyError(f"missing input {key}")
        bits = _coerce_bits(inputs[key], len(wires), key)
        for w, v in zip(wires, bits):
            vals[w] = v
    _run(circuit, vals, 1)
    return {label: [vals[w] for w in wires] for _, label, wires in circuit.outputs}


def evaluate_batch(circuit: Circuit,
                   inputs_list: Sequence[Mapping[Tuple[int, str], Union[int, Sequence[int]]]]
                   ) -> List[Dict[str, List[int]]]:
    """Bit-parallel evaluation of many assignments (one int bit per assignment)."""
    B = len(inputs_list)
    if B == 0:
        return []
    mask = (1 << B) - 1
    vals = [0] * circuit.n_wires
    vals[ONE] = mask
    for party, label, wires in circuit.inputs:
        key = (party, label)
        packed = [0] * len(wires)
        for t, inp in enumerate(inputs_list):
            if key not in inp:
                raise KeyError(f"missing input {key} in assignment {t}")
            bits = _coerce_bits(inp[key], len(wires), key)
            for i, v in enumerate(bits):
                if v:
                    packed[i] |= 1 << t
        for w, v in zip(wires, packed):
            vals[w] = v
    _run(circuit, vals, mask)
    res = []
    for t in range(B):
        res.append({label: [(vals[w] >> t) & 1 for w in wires] for _, label, wires in circuit.outputs})
    return res


def outputs_as_ints(outputs: Mapping[str, Sequence[int]]) -> Dict[str, int]:
    """Convenience: convert an ``evaluate`` result to ``{label: int}``."""
    return {k: bits_to_int(v) for k, v in outputs.items()}
