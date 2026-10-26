"""Regenerate or verify the exported circuits of ``circuits/`` from ``circuits/manifest.json``.

The directory ``circuits/`` (at the root of the artifact) holds the generation
circuits in the canonical binary encoding (``<name>.bin``) and as Bristol
Fashion text (``<name>.bristol.txt``).  The files are large (3.6 GB of binary
files and 7.1 GB of Bristol text for the whole manifest) and are not part of
the packaged archive; only ``circuits/manifest.json`` is.  The manifest names
every circuit with its descriptor and with the SHA-256 digests of both files,
and the circuit compiler of the reference implementation is deterministic, so
the files can be rebuilt and checked bit for bit.  The native programs need
the ``.bin`` files (they have no circuit compiler); no program of the artifact
reads the Bristol text.

Usage (from any directory; needs Python with NumPy):

    python python/bench/regen_circuits.py --tests        # the 36 circuits that the tests load (about 0.4 GB)
    python python/bench/regen_circuits.py                # build every circuit whose .bin is missing or differs
    python python/bench/regen_circuits.py --all          # build every circuit of the manifest again
    python python/bench/regen_circuits.py NAME [NAME ...]  # build the named circuits
    python python/bench/regen_circuits.py --check        # verify the files that are present; write nothing
    python python/bench/regen_circuits.py --list         # names, AND gates and file sizes of the manifest

``--tests`` selects TEST_CIRCUITS: exactly the circuits that the native test
programs and the mixed-language (interop) tests open, determined by running
them on a directory with all circuits of the manifest.  The measurement
campaign (bench/run_campaign.py) needs the whole manifest, or with --quick
exports the few circuits it needs itself.

Options: ``--out DIR`` writes to (and checks) another directory; the manifest
is always read from ``circuits/manifest.json`` and is never rewritten by this
script.  ``--no-bristol`` deletes the Bristol text of a circuit after its
digest has been checked (the exporter cannot skip it).

Every file written is compared with the manifest (size and SHA-256 of both
files, number of AND gates).  The exit status is 0 iff nothing differs; with
``--check`` a missing ``.bin`` file also gives a non-zero status, a missing
Bristol text is only reported.

To export a circuit that is not in the manifest, use
``python python/bench/export_circuits.py NAME``, which adds it to the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))                 # python/bench
ARTIFACT = os.path.normpath(os.path.join(HERE, "..", ".."))
CIRCUITS = os.path.join(ARTIFACT, "circuits")
MANIFEST = os.path.join(CIRCUITS, "manifest.json")

# The circuits that the test programs open: cpp/build/{selftest, check_vectors, test_*} and the interop
# scripts of cpp/tests (test runner groups cpp and interop).
TEST_CIRCUITS = (
    "setup_s64", "gate_n1_k64_s64_shared_om1",
    "dpf_n4_k32_s64_xor_secret", "dpf_n4_k64_s64_add_secret", "dpf_n4_k64_s64_xor_one", "dpf_n4_k64_s64_xor_secret",
    "dpf_n8_k64_s64_xor_one", "dpf_n8_k64_s64_xor_secret", "dpf_n16_k64_s64_xor_one",
    "dcf_n4_k32_s64_xor_secret", "dcf_n4_k64_s64_add_one", "dcf_n4_k64_s64_xor_one", "dcf_n4_k64_s64_xor_secret",
    "dcf_n8_k64_s64_xor_one", "dcf_n8_k64_s64_xor_secret",
    "gate_n8_k64_s64_shared_dpf", "gate_n8_k64_s64_auth_dpf", "gate_n12_k64_s64_shared_dpf", "gate_n8_k8_s64_shared_dpf",
    "gate_n8_k64_s64_shared_t85o_t0o", "gate_n8_k64_s64_auth_t85o_t0o", "gate_n8_k8_s64_auth_t128o_t0o_t128r_t0r_om1",
    "site_n8_f4_k64_s64_round", "site_n8_f4_k64_s64_exact", "site_n16_f8_k64_s64_round", "site_n16_f8_k64_s64_exact",
    "inmask_p0_c24_k64_s64", "inmask_p1_c2_k64_s64", "mattriple_2x2_k64_s64", "mattriple_2x4_k64_s64",
    # task R-TILE: the tiled private-model plan of mlp7 (tiles of 2 x 2) in test_ppml and interop_infer
    "mattriple_2x1_k64_s64", "mattriple_1x2_k64_s64", "mattriple_1x1_k64_s64",
    "inmask_p0_c4_k64_s64", "inmask_p0_c3_k64_s64", "inmask_p1_c4_k64_s64",
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_entries() -> list:
    with open(MANIFEST, "r", encoding="utf-8") as f:
        doc = json.load(f)
    entries = doc.get("circuits")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{MANIFEST}: no list 'circuits'")
    return entries


def check_files(entry: dict, out_dir: str, need_bristol: bool) -> list:
    """Differences between the files of one circuit in ``out_dir`` and its manifest entry."""
    diffs = []
    bin_path = os.path.join(out_dir, entry["binary_file"])
    if not os.path.isfile(bin_path):
        diffs.append("binary file missing")
    else:
        if os.path.getsize(bin_path) != entry["binary_bytes"]:
            diffs.append(f"binary size {os.path.getsize(bin_path)} != {entry['binary_bytes']}")
        elif sha256_file(bin_path) != entry["digest"]:
            diffs.append("binary digest differs")
    br_path = os.path.join(out_dir, entry["bristol_file"])
    if not os.path.isfile(br_path):
        if need_bristol:
            diffs.append("Bristol file missing")
    else:
        if os.path.getsize(br_path) != entry["bristol_bytes"]:
            diffs.append(f"Bristol size {os.path.getsize(br_path)} != {entry['bristol_bytes']}")
        elif sha256_file(br_path) != entry["bristol_sha256"]:
            diffs.append("Bristol digest differs")
    return diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="regenerate or verify the exported circuits from circuits/manifest.json")
    ap.add_argument("names", nargs="*", help="circuit names of the manifest (default: see --all)")
    ap.add_argument("--all", action="store_true", help="build every circuit, also those that are present and equal")
    ap.add_argument("--tests", action="store_true", help="only the circuits that the test programs load (TEST_CIRCUITS)")
    ap.add_argument("--check", action="store_true", help="verify the present files against the manifest; write nothing")
    ap.add_argument("--list", action="store_true", help="print the circuits of the manifest and exit")
    ap.add_argument("--out", default=CIRCUITS, help="directory of the circuit files (default: circuits/)")
    ap.add_argument("--no-bristol", action="store_true", help="delete the Bristol text after its digest was checked")
    args = ap.parse_args(argv)

    entries = load_entries()
    by_name = {e["name"]: e for e in entries}
    if args.list:
        total_bin = total_br = 0
        for e in entries:
            total_bin += e["binary_bytes"]
            total_br += e["bristol_bytes"]
            print(f"{e['name']:<46} AND={e['stats']['n_and']:>9,} bin={e['binary_bytes']:>11,} "
                  f"bristol={e['bristol_bytes']:>11,}")
        print(f"{len(entries)} circuits, binary files {total_bin:,} bytes, Bristol files {total_br:,} bytes")
        return 0
    if args.tests:
        args.names = list(args.names) + [n for n in TEST_CIRCUITS if n not in args.names]
    unknown = [n for n in args.names if n not in by_name]
    if unknown:
        raise SystemExit("not in the manifest: " + ", ".join(unknown))
    selected = [by_name[n] for n in args.names] if args.names else entries

    if args.check:
        bad = missing_bin = missing_br = 0
        for e in selected:
            diffs = check_files(e, args.out, need_bristol=False)
            no_br = not os.path.isfile(os.path.join(args.out, e["bristol_file"]))
            missing_br += no_br
            if diffs == ["binary file missing"]:
                missing_bin += 1
                print(f"MISSING  {e['name']}")
            elif diffs:
                bad += 1
                print(f"DIFFERS  {e['name']}: " + "; ".join(diffs))
            else:
                print(f"ok       {e['name']}" + ("  (no Bristol text)" if no_br else ""))
        print(f"{len(selected)} circuits checked in {args.out}: {len(selected) - bad - missing_bin} equal to the "
              f"manifest, {bad} different, {missing_bin} binary files missing, {missing_br} Bristol files missing")
        return 0 if bad == 0 and missing_bin == 0 else 1

    sys.path.insert(0, os.path.join(ARTIFACT, "python"))
    sys.path.insert(0, HERE)
    import export_circuits  # noqa: E402  (python/bench/export_circuits.py)
    from couplet import circuits_gen, circuits_ppml  # noqa: E402

    os.makedirs(args.out, exist_ok=True)
    built = skipped = bad = 0
    t_all = time.perf_counter()
    for e in selected:
        if not args.all and (args.tests or not args.names) and not check_files(e, args.out, need_bristol=False):
            skipped += 1
            continue
        kind = circuits_ppml if circuits_ppml.is_ppml_descriptor(e["descriptor"]) else circuits_gen
        descriptor = kind.canonical_descriptor(e["descriptor"])
        if kind.descriptor_name(descriptor) != e["name"]:
            bad += 1
            print(f"DIFFERS  {e['name']}: the descriptor of the manifest does not give this name")
            continue
        new = export_circuits.export_one(descriptor, args.out, verify=True)
        built += 1
        diffs = [f"{key}: {new[key]} != {e[key]}"
                 for key in ("digest", "binary_bytes", "bristol_sha256", "bristol_bytes") if new[key] != e[key]]
        if new["stats"]["n_and"] != e["stats"]["n_and"]:
            diffs.append(f"AND gates: {new['stats']['n_and']} != {e['stats']['n_and']}")
        diffs += check_files(e, args.out, need_bristol=True)
        if diffs:
            bad += 1
            print(f"DIFFERS  {e['name']}: " + "; ".join(diffs))
        elif args.no_bristol:
            os.remove(os.path.join(args.out, e["bristol_file"]))
    print(f"{built} circuits built in {time.perf_counter() - t_all:.1f} s, {skipped} present and equal, "
          f"{bad} different from the manifest ({args.out})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
