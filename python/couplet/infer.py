"""Party-local inference driver of the Couplet layer (DESIGN_M3.md sections 2 and 3, F_infer).

``InferParty(role, chan, sid, model, mode, ...)`` runs ONE party; the other
party runs the same commands with the same public parameters and its own
features.  Keyword arguments: ``prep`` ("ideal" with a shared ``dealer``, or
"real"), ``k``, ``s``, ``sigma``, ``private_model``, ``triple_cap``,
``setup_adder``, and the test hooks ``rng``, ``delta_share``,
``checker_factory`` (a deviating ``ArithChecker``), ``opener_factory`` (a
deviating ``truncmask.Opener``).  Three steps:

``setup()``: the manifest of the session (``nn.plan_manifest``) is hashed and
the digests are exchanged BEFORE any circuit runs (a mismatch aborts; this
is the commitment point of the plan); then the core session is set up
(preprocessing, ``Delta_b``, the session wires of ``Delta``) with the first
triple reservation, and the arithmetic checker and the ``Opener`` are created.

``preprocess()``: every entry of the manifest is generated in order with
``circuits_ppml.generate`` (sites of a layer with ONE batched circuit
execution).  The AND budget is reserved at ``setup`` in one request unless
``triple_cap`` is given: then a request holds at most ``triple_cap``
generation triples, a layer is generated in sub-batches that fit the cap and
further requests are made as the pool runs short; ``triple_requests`` counts
them (one is the ideal).

``infer(features)``: the online pipeline.  Public-model mode: each party
holds its vertical slice of the features (``Model.partition``) and computes
its additive share of the first-layer pre-activation locally; the shares are
masked and exchanged by commit-then-open (no MAC); every later layer is the
public linear map on the authenticated activations, truncate-and-mask (one
opening per layer in mode round, two in mode exact), and the table query of
plane A.  Private-model mode: both parties' features and P_0's weights and
biases enter as authenticated inputs (one exchange), every linear layer is a
matrix-vector product with a matrix triple, every truncation a wire-mode
opening.  Public-model mode: the logits are sign-exact and are released
after the batched arithmetic check over all deferred openings, without an
output mask; the release is checked again.  Private-model mode (correction
C2 of design/CHANGELOG_M3.md): P_1 cannot bound P_0's weights, so every
logit receives an output mask ``[[2^k rho]]`` generated with the session
(the manifest's ``outmask`` entry, one per logit); the release is check, add
mask, open, check.  Returns the logits as signed integers with ``2f``
fraction bits, ``to_signed(v mod 2^k, k)`` of the authenticated payload;
the construction refuses a model whose logit bound does not fit ``k - 1``
bits (where the weights are known: not at P_1 in private-model mode).

Tiled private-model plans (task R-TILE; ``tiling``, default
``nn.DEFAULT_TILING``): a private model with more than ``tiling.limit``
weights has a tiled plan (``nn.plan_manifest``).  ``preprocess`` then
concatenates the instances of each party's input masks in generation order
(one ``InMaskMaterial`` per party, the same input exchange) and assigns the
triples of each tile-shape entry to the tiles of that shape in row-major order
(every triple serves exactly one tile; a count that does not match the plan
aborts); ``infer`` evaluates every linear layer with ``beaver.matvec_tiled``
(one opening exchange per layer, as untiled).  ``record`` gains ``tiled``,
``tiles`` (number of tile triples) and ``tiling``.

``run(features)`` does the three steps.  ``record`` holds the public outcome
(the thresholds ``v_l`` of every site, every ``x_hat``, the logits, the
counts) and ``info()`` the public counters; ``Model.mirror`` reproduces the
logits from the features and the recorded thresholds.

Abort: any ``ProtocolAbort`` and any failure after the first message of a
step retires the session, the checker and the material; every later command
raises ``ProtocolAbort``.  Argument errors are raised before any message.
Channel phases: ``manifest``; the session's and the engine's phases under
the prefix ``gen-`` during the generation; ``online-local``, ``tm-open``,
``input-open``, ``arith-open`` and ``arith-check`` during inference.
"""

from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence

import numpy as np

from . import beaver
from . import circuits_ppml as cp
from .hashes import ProtocolAbort
from .nn import DEFAULT_TILING, Model, Tiling, layer_tiles, plan_manifest, tile_shapes
from .session import ABORTED, ACTIVE, Session
from .spdz2k import AShare, ArithChecker
from .truncmask import Opener, Site, truncate_and_mask

__all__ = ["InferParty", "PHASE_ONLINE"]

PHASE_ONLINE = "online-local"
NEW, SETUP, READY, DONE = "NEW", "SETUP", "READY", "DONE"


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


class InferParty:
    """One party of the inference protocol; see the module docstring."""

    def __init__(self, role: int, chan, sid: bytes, model: Model, mode: str, *, prep="ideal",
                 dealer=None, k: int = 64, s: int = 64, sigma: int = 40, private_model: bool = False,
                 triple_cap: Optional[int] = None, setup_adder: str = "ripple", rng=None,
                 delta_share=None, checker_factory=None, opener_factory=None,
                 tiling: Optional[Tiling] = None) -> None:
        if role not in (0, 1) or chan.party != role:
            raise ValueError("role must be 0 or 1 and match the channel endpoint")
        if not isinstance(model, Model):
            raise ValueError("expected a couplet.nn.Model")
        if mode not in cp.MODES:
            raise ValueError(f"mode must be one of {cp.MODES}")
        if private_model and role == 0 and not model.weights_known:
            raise ValueError("P_0 holds the weights in private-model mode")
        if not private_model and not model.weights_known:
            raise ValueError("the public-model pipeline needs the weights")
        if triple_cap is not None and (not _is_int(triple_cap) or triple_cap < 1):
            raise ValueError("triple_cap must be a positive integer")
        self.role = role
        self.chan = chan
        self.sid = bytes(sid)
        self.model = model
        self.mode = mode
        self.private_model = bool(private_model)
        self.k, self.s = int(k), int(s)
        self.w = self.k + self.s
        if not model.sign_exact_in(self.w):
            raise ValueError("a pre-activation or a table entry of the model does not fit a sign-exact "
                             "value of w = k + s bits")
        if model.weights_known and not model.release_fits(self.k):
            raise ValueError("a logit of the model does not fit a signed payload of k bits")
        self.tiling = DEFAULT_TILING if tiling is None else tiling
        if not isinstance(self.tiling, Tiling):
            raise ValueError("tiling must be a couplet.nn.Tiling")
        self.manifest = plan_manifest(model, mode, self.k, self.s, self.private_model, self.tiling)
        self.tiled = self.private_model and self.tiling.applies(model)
        # tiled plan: per layer the tiles (r0, c0, h, w) row-major, and the tile entries (layer, shape, count)
        self.layout = [layer_tiles(r, c, self.tiling) for r, c in model.shapes] if self.tiled else None
        self._tile_plan = [(l, shape, count) for l, (r, c) in enumerate(model.shapes)
                           for shape, count in tile_shapes(r, c, self.tiling)] if self.tiled else []
        self.triple_cap = None if triple_cap is None else int(triple_cap)
        self.session = Session(role, chan, self.sid, k=self.k, s=self.s, prep=prep, sigma=sigma, dealer=dealer,
                               rng=rng, delta_share=delta_share, setup_adder=setup_adder, allow_refill=False)
        self._rng = rng
        self._checker_factory = ArithChecker if checker_factory is None else checker_factory
        self._opener_factory = Opener if opener_factory is None else opener_factory
        self.checker: Optional[ArithChecker] = None
        self.opener: Optional[Opener] = None
        self.sites: List[List[Site]] = []        # per hidden layer
        self.inmasks: dict = {}                  # owner -> InMaskMaterial (private-model mode)
        self.mattriples: list = []               # per layer (private-model mode, untiled plan)
        self.tile_triples: list = []             # per layer: [(r0, c0, MatTripleMaterial)] (tiled plan)
        self.outmasks: list = []                 # per logit (private-model mode)
        self.record: dict = {"mode": mode, "private_model": self.private_model, "executions": 0}
        self.stage = NEW

    # -- state ------------------------------------------------------------------- #

    @property
    def state(self) -> str:
        if self.session.state != ACTIVE or (self.checker is not None and self.checker.dead):
            return ABORTED
        return ACTIVE

    @property
    def delta_share(self):
        return self.session.delta_share

    @property
    def triple_requests(self) -> int:
        return self.session.triple_requests

    def abort(self) -> None:
        """Retire the session, the checker and all material."""
        if self.checker is not None:
            self.checker.abort()
        self.session.abort()
        self.sites = []
        self.inmasks = {}
        self.mattriples = []
        self.tile_triples = []
        self.outmasks = []

    def _require_active(self) -> None:
        if self.state != ACTIVE:
            self.abort()
            raise ProtocolAbort("session aborted")

    @contextlib.contextmanager
    def _step(self):
        """A protocol step: any failure after the first message retires the session."""
        self._require_active()
        before = (self.chan.msgs_sent, self.chan.msgs_recv)
        try:
            yield
        except ValueError:
            if (self.chan.msgs_sent, self.chan.msgs_recv) != before or self.state != ACTIVE:
                self.abort()
            raise
        except BaseException:
            self.abort()
            raise

    # -- 1. setup ------------------------------------------------------------------ #

    def _first_reservation(self) -> int:
        budget = self.manifest.and_budget()
        return budget if self.triple_cap is None else min(budget, self.triple_cap)

    def setup(self) -> None:
        """Manifest comparison, session setup with the first triple reservation, checker."""
        self._require_active()
        if self.stage != NEW:
            raise ValueError("setup() was already run")
        with self._step():
            self.record["manifest_digest"] = cp.compare_manifest(self.chan, self.manifest).hex()
            self.session.setup(reserve_triples=self._first_reservation())
            self.checker = self._checker_factory(self.role, self.chan, self.sid, self.session.delta_share, self.s)
            self.opener = self._opener_factory(self.role, self.chan, self.sid, self.checker, self.session.delta_share)
        self.stage = SETUP

    # -- 2. preprocessing ----------------------------------------------------------- #

    def _generate(self, desc: dict, batch: int, remaining: int) -> list:
        """One or more executions of ``desc`` for ``batch`` instances, reserving triples as needed."""
        per = cp.and_count(desc)
        cap = self.triple_cap
        chunk = batch if cap is None else max(1, min(batch, cap // per))
        out = []
        for start in range(0, batch, chunk):
            count = min(chunk, batch - start)
            need = per * count
            available = self.session.engine.triples_available()
            if available < need:
                want = need - available if cap is None else max(need - available, min(cap, remaining) - available)
                self.session.reserve_triples(want)
            out += cp.generate(self.session, desc, count, rng=self._rng)
            self.record["executions"] += 1
            remaining -= need
        return out

    def preprocess(self) -> None:
        """Generate every entry of the manifest in order (all correlated randomness of the session)."""
        self._require_active()
        if self.stage != SETUP:
            raise ValueError("preprocess() needs setup() first (and runs once)")
        remaining = self.manifest.and_budget()
        engine = self.session.engine
        inmask_parts: dict = {0: [], 1: []}
        tile_plan = list(self._tile_plan)
        staged: list = [[] for _ in self.model.shapes]     # tiled plan: per layer [(shape, materials)]
        with self._step():
            engine.phase_tag = "gen-"
            try:
                for entry in self.manifest.entries:
                    desc, batch = entry["descriptor"], entry["batch"]
                    mats = self._generate(desc, batch, remaining)
                    remaining -= cp.and_count(desc) * batch
                    kind = desc["kind"]
                    if kind == cp.KIND_SITE:
                        self.sites.append([Site(m) for m in mats])
                    elif kind == cp.KIND_INMASK and self.tiled:
                        inmask_parts[desc["owner"]] += mats
                    elif kind == cp.KIND_INMASK:
                        self.inmasks[desc["owner"]] = mats[0]
                    elif kind == cp.KIND_MATTRIPLE and self.tiled:
                        if not tile_plan:
                            raise ProtocolAbort("the manifest has more matrix triples than the tiles")
                        layer, shape, count = tile_plan.pop(0)
                        if (desc["rows"], desc["cols"]) != shape or len(mats) != count:
                            raise ProtocolAbort("the matrix triples do not match the tiles of the plan")
                        staged[layer].append((shape, list(mats)))
                    elif kind == cp.KIND_MATTRIPLE:
                        self.mattriples.append(mats[0])
                    elif kind == cp.KIND_OUTMASK:
                        self.outmasks += [AShare(m.share[0], m.share[1], self.w) for m in mats]
            finally:
                engine.phase_tag = ""
            if self.tiled:
                self._assign_tiles(staged, tile_plan)
                for owner, parts in inmask_parts.items():
                    if parts:
                        self.inmasks[owner] = beaver.merge_inmasks(parts)
        if len(self.sites) != len(self.model.hidden_widths):
            raise ProtocolAbort("the manifest did not provide the sites of every hidden layer")
        if self.private_model and len(self.outmasks) != self.model.n_out:
            raise ProtocolAbort("the manifest did not provide one output mask per logit")
        self.record.update({"and_gates": self.manifest.and_budget(), "sites": self.manifest.site_count(),
                            "triple_requests": self.session.triple_requests,
                            "triples_unused": engine.triples_available(), "tiled": self.tiled,
                            "tiles": sum(len(t) for t in self.tile_triples), "tiling": self.tiling.to_dict()})
        self.stage = READY

    def _assign_tiles(self, staged: list, rest: list) -> None:
        """Tiled plan: the triples of each shape entry go to the tiles of that shape, row-major, each once."""
        if rest:
            raise ProtocolAbort("the manifest has fewer matrix triples than the tiles")
        self.tile_triples = []
        for l, tiles in enumerate(self.layout):
            pools = {shape: list(mats) for shape, mats in staged[l]}
            if len(pools) != len(staged[l]):
                raise ProtocolAbort("a tile shape appears twice in a layer of the plan")
            out = []
            for r0, c0, h, w in tiles:
                pool = pools.get((h, w))
                if not pool:
                    raise ProtocolAbort("a tile has no matrix triple")
                out.append((r0, c0, pool.pop(0)))
            if any(pools.values()):
                raise ProtocolAbort("a matrix triple of the plan has no tile")
            self.tile_triples.append(out)

    # -- 3. the online pipeline ---------------------------------------------------- #

    def _layer_sites(self, layer: int) -> List[Site]:
        """The sites of hidden layer ``layer`` (a deviating party may override this)."""
        return self.sites[layer]

    def _linear_public(self, W: np.ndarray, b: np.ndarray, a: Sequence[AShare]) -> List[AShare]:
        """``[[W a + b 2^f]]`` for public integer ``W``, ``b`` (sign-exact by the bound check at setup)."""
        mod = 1 << self.w
        Wo = W.astype(object)
        values = Wo.dot(np.array([sh.value for sh in a], dtype=object)) % mod
        macs = Wo.dot(np.array([sh.mac for sh in a], dtype=object)) % mod
        f = self.model.f
        return [AShare(int(v), int(m), self.w).add_public(int(bi) << f, self.role, self.delta_share).with_sign_exact(True)
                for v, m, bi in zip(values, macs, b)]

    def _private_inputs(self, features: List[int]):
        """Authenticated input of the features (both parties) and of P_0's weights and biases."""
        model = self.model
        d0 = model.partition
        mine_vals = list(features)
        if self.role == 0:
            for W, b in model.layers:
                mine_vals += [int(v) for v in W.reshape(-1)] + [int(v) for v in b]
        mine = (self.inmasks[self.role], mine_vals) if self.inmasks.get(self.role) is not None else None
        peer = self.inmasks.get(1 - self.role)
        my_sh, peer_sh = beaver.exchange_inputs(self.opener, mine, peer)
        p0, p1 = (my_sh, peer_sh) if self.role == 0 else (peer_sh, my_sh)
        x = p0[:d0] + p1
        pos = d0
        W_sh, b_sh = [], []
        for rows, cols in model.shapes:
            W_sh.append([p0[pos + i * cols: pos + (i + 1) * cols] for i in range(rows)])
            pos += rows * cols
            b_sh.append(p0[pos:pos + rows])
            pos += rows
        return x, W_sh, b_sh

    def infer(self, features: Sequence[int]) -> List[int]:
        """The online pipeline on this party's quantised features; returns the logits."""
        self._require_active()
        if self.stage != READY:
            raise ValueError("infer() needs preprocess() first (and runs once per session)")
        model = self.model
        sl = model.feature_slice(self.role)
        features = [int(v) for v in features]
        if len(features) != sl.stop - sl.start:
            raise ValueError(f"P_{self.role} holds {sl.stop - sl.start} features")
        T = model.table()
        f = model.f
        thresholds, x_hats, opened = [], [], []
        with self._step():
            self.chan.mark(PHASE_ONLINE)
            if self.private_model:
                a, W_sh, b_sh = self._private_inputs(features)
            else:
                a = None
            L = model.n_layers
            logits = None
            for l in range(L):
                if self.private_model:
                    # no sign_exact flag: P_1 cannot bound P_0's weights (the truncation reads p mod 2^m
                    # only, and the logits are released through output masks)
                    if self.tiled:
                        p = beaver.matvec_tiled(self.opener, W_sh[l], a, self.tile_triples[l])
                    else:
                        p = beaver.matvec(self.opener, W_sh[l], a, self.mattriples[l])
                    p = [pi + bi.scale(1 << f) for pi, bi in zip(p, b_sh[l])]
                elif l == 0:
                    p = model.first_layer_share(self.role, features)
                else:
                    W, b = model.layers[l]
                    p = self._linear_public(W, b, a)
                if l == L - 1:
                    self.record["openings_before_release"] = self.checker.openings
                    if self.private_model:
                        logits = self.opener.open_masked(p, self.outmasks)
                    else:
                        logits = self.opener.open_sign_exact(p)
                    break
                sites = self._layer_sites(l)
                res = truncate_and_mask(self.opener, sites, p, first_layer=(l == 0 and not self.private_model))
                thresholds.append([r.v_l for r in res])
                x_hats.append([r.x_hat for r in res])
                opened.append([r.v for r in res])
                a = [site.plane_a.table(T, r.x_hat) for site, r in zip(sites, res)]
        self.record.update({"thresholds": thresholds, "x_hat": x_hats, "opened": opened, "logits": logits,
                            "openings": self.checker.openings, "checks": self.checker.checks,
                            "shared_openings": self.opener.shared_openings,
                            "release_masked": self.private_model, "released": list(self.checker.last_opened)})
        self.stage = DONE
        return logits

    def run(self, features: Sequence[int]) -> List[int]:
        """``setup``, ``preprocess`` and ``infer``."""
        self.setup()
        self.preprocess()
        return self.infer(features)

    # -- reporting ---------------------------------------------------------------- #

    def info(self) -> dict:
        """Public parameters and counters (no secrets)."""
        out = self.session.info()
        out.update({"state": self.state, "stage": self.stage, "mode": self.mode,
                    "private_model": self.private_model, "manifest": self.manifest.summary(),
                    "tiled": self.tiled, "tiling": self.tiling.to_dict(),
                    "triple_cap": self.triple_cap, "record": {key: value for key, value in self.record.items()
                                                              if key not in ("opened",)}})
        return out
