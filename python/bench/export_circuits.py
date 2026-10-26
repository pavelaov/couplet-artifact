"""Export generation circuits for other implementations and frameworks.

For every requested descriptor the script writes, into ``artifact/circuits/``:

  <name>.bin           canonical binary encoding (``Circuit.to_bytes``, format
                       version 2, magic ``CPLTCIR2``); its SHA-256 is the
                       circuit digest the two parties compare before a run;
  <name>.bristol.txt   Bristol Fashion text (``Circuit.to_bristol``): gates
                       XOR / AND / INV only, input values ordered party 0,
                       party 1, bound (the session MAC key ``delta``);

and it updates ``manifest.json`` (descriptor, files, digests, statistics, the
input / output declarations and the Bristol layout of every exported circuit).
Entries of earlier runs are kept unless the same name is exported again.

Usage (from ``artifact/python``):

    python bench/export_circuits.py                  # the default set
    python bench/export_circuits.py --list           # show the default set
    python bench/export_circuits.py NAME [NAME ...]  # named descriptors

A NAME is a canonical descriptor name as printed by
``couplet.circuits_gen.descriptor_name`` (for example ``setup_s64``,
``dpf_n8_k64_s64_xor_one``, ``dcf_n12_k32_s64_add_secret``,
``gate_n12_k64_s64_shared_dpf``, ``batch16x_gate_n8_k64_s64_shared_dpf``), a
gate preset ``preset:<preset>:<n>[:<k>[:<s>]]`` (for example
``preset:table:12``, ``preset:relu_large:16``), or ``json:<descriptor>``.
The generation circuits of the Couplet layer (``couplet.circuits_ppml``) are
accepted under their names as well (``site_n16_f8_k64_s64_round``,
``site_n16_f8_k64_s64_exact``, ``triples4_k64_s64``,
``mattriple_8x16_k64_s64``, ``inmask_p0_c12_k64_s64``) or as ``json:`` with
their kinds; they are built with ``circuits_ppml.build``.

Default set: session setup, DPF and DCF with public payload for n = 8, 12,
16, and the table gate for n = 12, 16 (k = s = 64, XOR index shares).

Inference plans (for the native inference driver):

    python bench/export_circuits.py --plan MODEL.json [--mode round|exact|both]
                                    [--private-model] [--tile RxC] [--tile-limit L]
                                    [--k 64] [--s 64] [--plan-out DIR] [--skip-existing]

loads a model file of ``couplet.tools.export_model``, builds the session plan
``couplet.nn.plan_manifest`` for every requested mode, writes it as
``<name>_plan_<mode>[_private]_k<k>_s<s>.json`` (``<name>`` is the file name
without ``_model.json``; next to the model unless ``--plan-out``) and exports
every circuit the plan names, plus the session-setup circuit.  Plan file,
format ``"couplet-plan/1"``: ``model_file``, ``mode``, ``k``, ``s``,
``private_model``, ``manifest_hex`` (the ``CPLTMAN1`` encoding), ``digest``
(its SHA-256, the value both parties compare first), ``and_budget``,
``sites``, ``entries`` (``use``, ``name``, ``descriptor``, ``batch``,
``n_and``) and ``circuits`` (the names to load).  ``--skip-existing`` keeps a
circuit whose manifest entry exists and whose binary file has the recorded
size and digest.

Tiled private-model plans (task R-TILE, ``couplet.nn.Tiling``): ``--tile RxC``
(default 16x16) and ``--tile-limit L`` (default 4096) are passed to
``plan_manifest``.  A plan file of a tiled plan, or one written with a tiling
other than the default, has the key ``tiling`` (``rows``, ``cols``, ``limit``,
``tiled``, ``tiles``, ``tiles_per_layer``); with a tiling other than the default
the file name ends in ``_tile<R>x<C>_lim<L>_k<k>_s<s>.json``.  The plan files
of untiled plans with the default tiling are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from couplet import circuits_gen as cg  # noqa: E402
from couplet import circuits_ppml as cp  # noqa: E402
from couplet.circuit import BOUND, FORMAT_VERSION, Circuit  # noqa: E402

DEFAULT_OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "circuits"))
MANIFEST = "manifest.json"


def default_descriptors(k: int = 64, s: int = 64) -> list:
    descs = [cg.session_setup_descriptor(s)]
    descs += [cg.dpf_descriptor(n, k, s, "xor", "public_one") for n in (8, 12, 16)]
    descs += [cg.dcf_descriptor(n, k, s, "xor", "public_one") for n in (8, 12, 16)]
    descs += [cg.gate_preset("table", n, k, s) for n in (12, 16)]
    return descs


def parse_name(name: str) -> dict:
    """Descriptor of a command-line NAME (see the module docstring)."""
    if name.startswith("json:"):
        desc = json.loads(name[5:])
        return cp.canonical_descriptor(desc) if cp.is_ppml_descriptor(desc) else cg.canonical_descriptor(desc)
    if name.startswith("preset:"):
        parts = name.split(":")
        if not 3 <= len(parts) <= 5:
            raise ValueError(f"bad preset name {name!r}; expected preset:<preset>:<n>[:<k>[:<s>]]")
        return cg.gate_preset(parts[1], *(int(x) for x in parts[2:]))
    try:
        return cg.descriptor_from_name(name)
    except ValueError:
        return cp.descriptor_from_name(name)


def build_circuit(descriptor: dict):
    """The circuit of a core descriptor (``circuits_gen``) or of a Couplet one (``circuits_ppml``)."""
    return cp.build(descriptor) if cp.is_ppml_descriptor(descriptor) else cg.build(descriptor)


def _owner_text(owner: int) -> str:
    """Input owner as text: "0", "1" or "bound"."""
    return "bound" if owner == BOUND else str(owner)


def export_one(descriptor: dict, out_dir: str, verify: bool = True) -> dict:
    """Build, write both files, and return the manifest entry."""
    t = time.perf_counter()
    C = build_circuit(descriptor)
    name = C.meta["name"]
    data = C.to_bytes()
    digest = hashlib.sha256(data).hexdigest()
    bristol = C.to_bristol().encode("ascii")
    bin_file, bristol_file = name + ".bin", name + ".bristol.txt"
    with open(os.path.join(out_dir, bin_file), "wb") as f:
        f.write(data)
    with open(os.path.join(out_dir, bristol_file), "wb") as f:
        f.write(bristol)
    if verify:
        with open(os.path.join(out_dir, bin_file), "rb") as f:
            back = Circuit.from_bytes(f.read())          # validates the canonical form
        if back.digest() != digest or digest != C.digest():
            raise RuntimeError(f"{name}: the written binary does not reproduce the circuit digest")
    head = bristol.split(b"\n", 1)[0].split()
    entry = {
        "name": name,
        "descriptor": C.meta["descriptor"],
        "binary_file": bin_file,
        "binary_bytes": len(data),
        "digest": digest,
        "bristol_file": bristol_file,
        "bristol_bytes": len(bristol),
        "bristol_sha256": hashlib.sha256(bristol).hexdigest(),
        "bristol_gates": int(head[0]),
        "bristol_wires": int(head[1]),
        "stats": C.stats(),
        "inputs": [{"owner": _owner_text(p), "label": l, "width": len(w)} for p, l, w in C.inputs],
        "outputs": [{"owner": str(t), "label": l, "width": len(w)} for t, l, w in C.outputs],
        "bristol_layout": C.bristol_layout(),
    }
    print(f"  {name:<44} AND={entry['stats']['n_and']:>9,} depth={entry['stats']['and_depth']:>4} "
          f"bin={len(data) / 1e6:7.2f} MB bristol={len(bristol) / 1e6:7.2f} MB "
          f"digest={digest[:16]}.. ({time.perf_counter() - t:.1f} s)", flush=True)
    return entry


def load_manifest(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if isinstance(doc.get("circuits"), list):
            return {e["name"]: e for e in doc["circuits"] if isinstance(e, dict) and "name" in e}
    except (OSError, ValueError):
        pass
    return {}


def write_manifest(path: str, entries: dict) -> None:
    doc = {
        "generated_by": "python bench/export_circuits.py",
        "design_version": cg.DESIGN_VERSION,
        "binary_format": {
            "version": FORMAT_VERSION,
            "magic": "CPLTCIR2",
            "layout": "all integers u32 little-endian: magic(8 bytes) | n_wires | n_gates | n_input_decls | "
                      "n_output_decls | n_layers | gates: n_gates x (op, a, b, out) with op 0 XOR, 1 AND, "
                      "2 NOT (b = a) | layers: n_layers x (start, end) gate-index ranges of the AND layers | "
                      "inputs: (owner, label_len, label utf-8, width, wires[width]) with owner 0, 1, "
                      "2 = BOUND | outputs: same with owner 0, 1, 2 = PUBLIC, 3 = RETAIN. Wires 0 and 1 "
                      "are the constants 0 and 1; inputs follow in declaration order; gate i drives wire "
                      "2 + n_inputs + i. Gate order: [linear][AND layer 1][linear][AND layer 2]...",
            "digest": "SHA-256 of the binary file",
        },
        "bristol_format": "Bristol Fashion; gates XOR, AND, INV only; input values ordered party 0, party 1, "
                          "bound; one output value per output declaration; bit vectors little-endian; see "
                          "bristol_layout of each entry for labels, owners and offsets",
        "circuits": [entries[name] for name in sorted(entries)],
    }
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def descriptor_name(descriptor: dict) -> str:
    """The file name stem of a core or Couplet descriptor."""
    return cp.descriptor_name(descriptor) if cp.is_ppml_descriptor(descriptor) else cg.descriptor_name(descriptor)


def plan_document(model_path: str, mode: str, k: int, s: int, private_model: bool, tiling=None) -> dict:
    """The plan file of one session (see the module docstring)."""
    from couplet import nn as nnm
    tiling = nnm.DEFAULT_TILING if tiling is None else tiling
    model = nnm.Model.load(model_path)
    man = nnm.plan_manifest(model, mode, k, s, private_model, tiling)
    entries = [{"use": e["use"], "name": cp.descriptor_name(e["descriptor"]), "descriptor": e["descriptor"],
                "batch": e["batch"], "n_and": cp.and_count(e["descriptor"])} for e in man.entries]
    names = sorted({e["name"] for e in entries} | {cg.descriptor_name(cg.session_setup_descriptor(s))})
    doc = {"format": "couplet-plan/1", "model_file": os.path.basename(model_path), "mode": mode, "k": k, "s": s,
           "private_model": bool(private_model), "model_hash": man.model_hash.hex(),
           "manifest_hex": man.encode().hex(), "digest": man.digest().hex(), "and_budget": man.and_budget(),
           "sites": man.site_count(), "entries": entries, "circuits": names}
    tiled = nnm.plan_tiled(model, private_model, tiling)
    if tiled or tiling != nnm.DEFAULT_TILING:
        per_layer = [len(nnm.layer_tiles(r, c, tiling)) for r, c in model.shapes] if tiled else []
        doc["tiling"] = dict(tiling.to_dict(), tiled=tiled, tiles=sum(per_layer), tiles_per_layer=per_layer)
    return doc


def plan_file_name(model_path: str, mode: str, k: int, s: int, private_model: bool, tiling=None) -> str:
    from couplet import nn as nnm
    stem = os.path.basename(model_path)
    stem = stem[:-len("_model.json")] if stem.endswith("_model.json") else os.path.splitext(stem)[0]
    tail = ""
    if tiling is not None and tiling != nnm.DEFAULT_TILING:
        tail = f"_tile{tiling.rows}x{tiling.cols}_lim{tiling.limit}"
    return f"{stem}_plan_{mode}{'_private' if private_model else ''}{tail}_k{k}_s{s}.json"


def parse_tiling(tile: str, limit: int):
    """``couplet.nn.Tiling`` from ``--tile RxC`` and ``--tile-limit L``."""
    from couplet import nn as nnm
    parts = tile.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError("--tile must be RxC with positive integers R and C")
    return nnm.Tiling(int(parts[0]), int(parts[1]), int(limit))


def _exists(entries: dict, name: str, out_dir: str) -> bool:
    """True iff the manifest has ``name`` and its binary file has the recorded size and digest."""
    entry = entries.get(name)
    if entry is None:
        return False
    path = os.path.join(out_dir, entry["binary_file"])
    if not os.path.isfile(path) or os.path.getsize(path) != entry["binary_bytes"]:
        return False
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest() == entry["digest"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="export generation circuits (binary + Bristol Fashion)")
    ap.add_argument("names", nargs="*", help="descriptor names (default: the default set)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: artifact/circuits)")
    ap.add_argument("--list", action="store_true", help="print the default set and exit")
    ap.add_argument("--no-verify", action="store_true", help="skip re-reading the written binary")
    ap.add_argument("--plan", default=None, help="model file: write its inference plan(s) and export their circuits")
    ap.add_argument("--mode", default="both", choices=("round", "exact", "both"), help="with --plan")
    ap.add_argument("--private-model", action="store_true", help="with --plan: the private-model plan")
    ap.add_argument("--tile", default="16x16", help="with --plan --private-model: the largest tile RxC (default 16x16)")
    ap.add_argument("--tile-limit", type=int, default=4096,
                    help="with --plan --private-model: plans of models above this many weights are tiled (default 4096)")
    ap.add_argument("--k", type=int, default=64, help="with --plan")
    ap.add_argument("--s", type=int, default=64, help="with --plan")
    ap.add_argument("--plan-out", default=None, help="with --plan: directory of the plan files (default: the model's)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not rebuild a circuit whose file matches the manifest entry")
    args = ap.parse_args(argv)
    if args.list:
        for d in default_descriptors():
            print(cg.descriptor_name(d))
        return 0
    if args.plan and args.names:
        ap.error("give either --plan or descriptor names")
    if args.plan:
        modes = ("round", "exact") if args.mode == "both" else (args.mode,)
        try:
            tiling = parse_tiling(args.tile, args.tile_limit)
        except ValueError as exc:
            ap.error(str(exc))
        plan_dir = args.plan_out or os.path.dirname(os.path.abspath(args.plan))
        os.makedirs(plan_dir, exist_ok=True)
        names = []
        for mode in modes:
            doc = plan_document(args.plan, mode, args.k, args.s, args.private_model, tiling)
            path = os.path.join(plan_dir, plan_file_name(args.plan, mode, args.k, args.s, args.private_model, tiling))
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(doc, f, indent=1)
                f.write("\n")
            print(f"wrote {path}: {len(doc['entries'])} executions, {doc['and_budget']:,} AND gates, "
                  f"digest {doc['digest'][:16]}..")
            names += [x for x in doc["circuits"] if x not in names]
        descs = [parse_name(x) for x in names]
    else:
        descs = [parse_name(x) for x in args.names] if args.names else default_descriptors()
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, MANIFEST)
    entries = load_manifest(manifest_path)
    print(f"exporting {len(descs)} circuit(s) to {args.out}")
    for d in descs:
        if args.skip_existing and _exists(entries, descriptor_name(d), args.out):
            print(f"  {descriptor_name(d):<44} present (skipped)", flush=True)
            continue
        entry = export_one(d, args.out, verify=not args.no_verify)
        entries[entry["name"]] = entry
        write_manifest(manifest_path, entries)        # keep the manifest consistent after every file
    print(f"wrote {manifest_path} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
