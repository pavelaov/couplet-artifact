"""Shared loaders for the table and figure generators (``make_tables.py``, ``make_figures.py``).

The module is called ``paper_numbers`` and not ``numbers``: a file named
``numbers.py`` next to the benchmark scripts would shadow the standard-library
module of that name (the script directory is first on ``sys.path``) and break
every script here that imports ``fractions``, ``statistics`` or NumPy.

Everything the paper prints as a number is read here from a JSON file under
``artifact/experiments/`` and wrapped in a :class:`Num`, which remembers the
file and field it came from and how it has to be printed:

    kind        printed       meaning
    ----------  ------------  -------------------------------------------------
    measured    black         a field of a run record (after a unit conversion
                              or rounding only)
    tool        black         exact output of a tool of this artifact that is
                              stored under experiments/ (circuit compiler,
                              bucket-parameter generator)
    derived     black         arithmetic on measured / tool values of this run
                              only (a quotient, a sum, a ratio); the formula is
                              part of the source text
    model       \\proj{...}    derived from a cost model or from numbers
                              reported by other papers (never measured here)
    reported    \\proj{...}    stated by another paper (as recorded in
                              couplet/baselines/costmodels.py)
    pending     \\flag{pending}  the data point does not exist (yet)

Arithmetic on Nums propagates the weakest kind: anything that touches a model
or a reported value is a model value, anything that touches a pending value
is pending.  A missing file or field never raises: it yields a pending Num,
so the generators always produce compilable tables.

A configuration that the campaign summary marks as ``failed`` or
``infeasible`` carries no measurement.  Its cells are pending Nums whose
``status`` holds the configuration, the status and the recorded reason; the
generators print that text as a red note below the table (or inside the
plot).  Such a row also hides an older run of the same configuration in
``raw/e2e_cpp``: a number of an earlier build is never printed in its place.
The same holds for the tool groups of the summary (``local``,
``prep_unit_costs``, ``baselines``): if one of them is not ``ok``, the cells
that would be read from the file of that group are pending with its reason.

Input files (relative to ``artifact/experiments``).  The first existing
source of a line wins; the files marked "optional" are produced by the final
measurement campaign and may be absent:

    raw/campaign/summary.json     optional  end-to-end runs (same row schema as
    raw/e2e_cpp/summary.json                raw/e2e_cpp/summary.json)
    raw/baselines_cpp.json        optional  native baselines
    raw/baselines_python.json               reference baselines + cost models
    raw/bench_local_cpp.json      optional  local evaluation incl. gates
    raw/bench_fss_cpp.json                  Eval / EvalAll / key sizes
    raw/bench_prep_cpp.json                 preprocessing unit costs
    raw/attacks.json                        adversarial regression
    raw/legacy_attacks.json                 counterexamples (earlier prototype)
    raw/machine.json              optional  machine description
    raw/tests.json                optional  test-suite counts
    figdata/circuit_stats.json              circuit compiler output
    figdata/bucket_params.json              bucket sizes

Units: 1 KB = 1,000 bytes, 1 MB = 1,000,000 bytes.

Run as a script (from artifact/python) to see what is available:

    python bench/paper_numbers.py                  # inventory of the input files
    python bench/paper_numbers.py --selftest       # checks of the helpers and the loaders
    python bench/paper_numbers.py --collect-tests  # run every test program, write raw/tests.json
                                                   # (the source of the \\numTests... macros)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.normpath(os.path.join(_HERE, "..", ".."))
DEFAULT_EXPERIMENTS = os.path.join(ARTIFACT, "experiments")
DEFAULT_PAPER = os.path.normpath(os.path.join(ARTIFACT, "..", "paper"))

KB = 1_000
MB = 1_000_000

KINDS = ("measured", "tool", "derived", "model", "reported", "pending")
BLACK_KINDS = ("measured", "tool", "derived")
RED_KINDS = ("model", "reported")

E2E_SOURCES = ("raw/campaign/summary.json", "raw/e2e_cpp/summary.json")
BASELINE_SOURCES = (("C++", "raw/baselines_cpp.json"), ("Python", "raw/baselines_python.json"))
LOCAL_SOURCES = ("raw/bench_local_cpp.json", "raw/bench_fss_cpp.json")
PREP_SOURCE = "raw/bench_prep_cpp.json"
ATTACKS_SOURCE = "raw/attacks.json"
LEGACY_SOURCE = "raw/legacy_attacks.json"
MACHINE_SOURCE = "raw/machine.json"
TESTS_SOURCES = ("raw/tests.json", "raw/campaign/tests.json")
CIRCUITS_SOURCE = "figdata/circuit_stats.json"
BUCKET_SOURCE = "figdata/bucket_params.json"
COSTMODEL_SOURCE = "raw/baselines_python.json"

ALL_SOURCES = (E2E_SOURCES + tuple(rel for _, rel in BASELINE_SOURCES) + LOCAL_SOURCES
               + (PREP_SOURCE, ATTACKS_SOURCE, LEGACY_SOURCE, MACHINE_SOURCE) + TESTS_SOURCES
               + (CIRCUITS_SOURCE, BUCKET_SOURCE))

PHASES = ("baseot", "cotext", "abit", "laand", "bucket", "check", "input", "online", "output")
PHASE_LABELS = {
    "baseot": "Base OT",
    "cotext": "OT extension",
    "abit": "Auth.\\ bits",
    "laand": "Leaky AND",
    "bucket": "Bucketing",
    "check": "Triple check",
    "input": "Inputs",
    "online": "AND layers",
    "output": "Output checks",
}


# --------------------------------------------------------------------------- #
# Values with provenance
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Num:
    """A value together with its kind and its source (file and field, or formula)."""

    value: object
    kind: str
    source: str
    note: str = ""
    status: str = ""        # pending only: "<configuration>: failed|infeasible: <reason>" (campaign summary)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.status and self.kind != "pending":
            raise ValueError("only a pending value carries a status")

    @property
    def ok(self) -> bool:
        return self.kind != "pending" and self.value is not None

    @property
    def red(self) -> bool:
        return self.kind in RED_KINDS


def pending(source: str, note: str = "", status: str = "") -> Num:
    """A data point that does not exist; ``source`` says where it is expected.

    ``status`` is set when the run is listed in the campaign summary but marked
    as failed or infeasible (see :func:`e2e_rows` and :func:`tool_failure`).
    """
    return Num(None, "pending", source, note or status, status)


def measured(value, source: str, note: str = "") -> Num:
    return pending(source, note) if value is None else Num(value, "measured", source, note)


def tool(value, source: str, note: str = "") -> Num:
    return pending(source, note) if value is None else Num(value, "tool", source, note)


def weakest_kind(nums) -> str:
    """Kind of a value computed from ``nums``."""
    kinds = {x.kind for x in nums}
    if "pending" in kinds or any(x.value is None for x in nums):
        return "pending"
    if kinds & set(RED_KINDS):
        return "model"
    return "derived"


def calc(fn, formula: str, *nums: Num, note: str = "") -> Num:
    """``fn(*values)`` as a Num; ``formula`` names the computation, e.g. ``"{0} / {1}"``.

    The placeholders ``{i}`` are replaced by the sources of the arguments.
    """
    source = formula.format(*[f"[{x.source}]" for x in nums])
    kind = weakest_kind(nums)
    if kind == "pending":
        missing = "; ".join(x.source for x in nums if not x.ok)
        status = next((x.status for x in nums if x.status), "")
        return pending(source, note or f"missing: {missing}", status)
    try:
        value = fn(*[x.value for x in nums])
    except (ZeroDivisionError, TypeError, ValueError):
        return pending(source, note or "not computable")
    return Num(value, kind, source, note)


def scale(num: Num, divisor: float, unit: str) -> Num:
    """Unit conversion; keeps the kind (a converted measurement is still measured)."""
    if not num.ok:
        return num
    return Num(num.value / divisor, num.kind, f"{num.source} / {divisor:g} ({unit})", num.note)


# --------------------------------------------------------------------------- #
# TeX formatting
# --------------------------------------------------------------------------- #

def tex_int(value) -> str:
    """Integer with thousands separators that are safe in text and math mode."""
    return f"{int(round(value)):,}".replace(",", "{,}")


def tex_fixed(value, digits: int) -> str:
    """Fixed number of decimals, thousands separators in the integer part."""
    text = f"{float(value):,.{digits}f}"
    return text.replace(",", "{,}")


def tex_sig(value, digits: int = 3) -> str:
    """``digits`` significant digits without exponent notation."""
    value = float(value)
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    rounded = round(value, digits - 1 - exponent)       # a negative count rounds to tens, hundreds, ...
    # rounding may add a digit (9.99 -> 10.0): recompute the number of decimals
    exponent = math.floor(math.log10(abs(rounded))) if rounded else exponent
    decimals = max(0, digits - 1 - exponent)
    return tex_fixed(rounded, decimals)


def fmt_int(value) -> str:
    return tex_int(value)


def fmt(digits: int):
    """Formatter with a fixed number of decimals."""
    return lambda value: tex_fixed(value, digits)


def fmt_sig(digits: int = 3):
    return lambda value: tex_sig(value, digits)


def fmt_seconds(value) -> str:
    """Seconds: three decimals below 10 s, two below 100 s, one above."""
    value = float(value)
    if value < 10:
        return tex_fixed(value, 3)
    if value < 100:
        return tex_fixed(value, 2)
    return tex_fixed(value, 1)


def fmt_str(value) -> str:
    return str(value)


def tex(num: Num, formatter=fmt_int, suffix: str = "") -> str:
    """The TeX text of a cell: black, ``\\proj{...}`` or ``\\flag{pending}``."""
    if not num.ok:
        return r"\flag{pending}"
    body = formatter(num.value) + suffix
    if num.red:
        return r"\proj{" + body + "}"
    return body


_TEX_SPECIALS = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}",
                 "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
                 "<": r"\textless{}", ">": r"\textgreater{}"}


def tex_escape(text: str) -> str:
    """Escape plain text for TeX (used for strings that come from JSON files)."""
    return "".join(_TEX_SPECIALS.get(ch, ch) for ch in str(text))


_SMALL = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
          "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen",
          "Nineteen")
_TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety")


def number_word(n: int) -> str:
    """CamelCase English word of ``0 <= n < 10000`` for macro names (16 -> ``Sixteen``)."""
    n = int(n)
    if not 0 <= n < 10000:
        raise ValueError("number_word covers 0..9999")
    if n < 20:
        return _SMALL[n]
    if n < 100:
        return _TENS[n // 10] + (_SMALL[n % 10] if n % 10 else "")
    if n < 1000:
        return _SMALL[n // 100] + "Hundred" + (number_word(n % 100) if n % 100 else "")
    return number_word(n // 1000) + "Thousand" + (number_word(n % 1000) if n % 1000 else "")


# --------------------------------------------------------------------------- #
# File access
# --------------------------------------------------------------------------- #

class Store:
    """Lazy, cached access to the JSON files under ``artifact/experiments``."""

    def __init__(self, experiments: str | None = None):
        self.root = os.path.abspath(experiments or DEFAULT_EXPERIMENTS)
        self._cache: dict = {}
        self.status: dict = {}
        self.derived: dict = {}         # results computed from the files (e.g. the merged end-to-end rows)

    def path(self, rel: str) -> str:
        return os.path.join(self.root, *rel.split("/"))

    def load(self, rel: str):
        """Parsed JSON of ``rel`` or ``None`` (missing or unreadable)."""
        if rel in self._cache:
            return self._cache[rel]
        doc = None
        path = self.path(rel)
        if not os.path.isfile(path):
            self.status[rel] = "missing"
        else:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    doc = json.load(fh)
                self.status[rel] = "found"
            except (OSError, ValueError) as exc:
                self.status[rel] = f"unreadable ({type(exc).__name__})"
        self._cache[rel] = doc
        return doc

    def has(self, rel: str) -> bool:
        return self.load(rel) is not None


_STORE = Store()


def configure(experiments: str | None = None) -> Store:
    """Point the loaders at another experiments directory (used by the tests)."""
    global _STORE
    _STORE = Store(experiments)
    return _STORE


def store() -> Store:
    return _STORE


def _first(mapping, *keys, default=None):
    """First present, non-None value of ``mapping`` under one of ``keys``."""
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _rows_of(doc, *names) -> list:
    """The list of row dictionaries of a document, under one of ``names`` or at top level."""
    if isinstance(doc, list):
        return [r for r in doc if isinstance(r, dict)]
    if isinstance(doc, dict):
        for name in names:
            rows = doc.get(name)
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
            if isinstance(rows, dict):
                return [dict(v, _key=k) for k, v in rows.items() if isinstance(v, dict)]
    return []


# --------------------------------------------------------------------------- #
# End-to-end runs (two processes, native implementation)
# --------------------------------------------------------------------------- #

_TASK_ALIASES = {"gate_table": "gate-table", "table": "gate-table", "gate-tab": "gate-table",
                 "gate_cmp": "gate-cmp", "comparison": "gate-cmp", "cmp": "gate-cmp",
                 "gate-comparison": "gate-cmp"}


def _e2e_key(row: dict):
    cfg = _first(row, "config", "params", default={}) or {}
    task = _first(cfg, "task", "family", default=_first(row, "task", "family"))
    if task is None:
        return None
    task = _TASK_ALIASES.get(str(task), str(task))
    n = _first(cfg, "n", default=_first(row, "n"))
    if n is None:
        return None
    beta = str(_first(cfg, "beta", "beta_mode", default="public"))
    beta = "secret" if beta.startswith("secret") else "public"
    return (task, int(n), int(_first(cfg, "batch", default=_first(row, "batch", default=1))),
            float(_first(cfg, "delay_ms", default=_first(row, "delay_ms", default=0.0))),
            int(_first(cfg, "k", default=64)), beta,
            bool(_first(cfg, "auth", "auth_input", default=False)),
            str(_first(cfg, "prep", default="real")), int(_first(cfg, "sigma", default=40)),
            int(_first(cfg, "threads", default=_first(row, "threads", default=1))))


def row_status(row) -> tuple:
    """``(status, reason)`` of a summary row or tool group; without a ``status`` field it is ``ok``.

    The campaign runner writes ``ok``, ``failed`` or ``infeasible``; any other
    word is treated like ``failed`` (the row carries no usable measurement).
    """
    status = str(_first(row, "status", default="ok")).strip().lower() or "ok"
    if status == "ok":
        return "ok", ""
    reason = " ".join(str(_first(row, "reason", default="") or "no reason recorded").split())
    return status, reason


def config_label(key) -> str:
    """Readable name of an end-to-end configuration key (only the parts that differ from the defaults)."""
    task, n, batch, delay, k, beta, auth, prep, sigma, threads = key
    parts = [f"{task}, n = {n}"]
    if k != 64:
        parts.append(f"k = {k}")
    if batch != 1:
        parts.append(f"batch {batch}")
    if beta != "public":
        parts.append(f"{beta} payload")
    if auth:
        parts.append("authenticated input")
    if prep != "real":
        parts.append(f"{prep} preprocessing")
    if sigma != 40:
        parts.append(f"sigma = {sigma}")
    if threads != 1:
        parts.append(f"{threads} threads")
    if delay:
        parts.append(f"delay {delay:g} ms")
    return ", ".join(parts)


def e2e_rows() -> dict:
    """All end-to-end rows keyed by configuration; campaign rows override the earlier runs.

    Key: (task, n, batch, delay_ms, k, beta, auth, prep, sigma, threads).  Each
    row keeps ``_file`` (the summary it was read from) and ``_sel`` (its
    selector inside that file).

    A row whose ``status`` is not ``ok`` (``failed`` or ``infeasible`` in the
    campaign summary) becomes a stub without any measurement: it has only
    ``name``, ``config``, ``_file``, ``_sel`` and ``_not_ok`` (configuration,
    status and reason as one text).  The stub of the preferred source replaces
    an older run of the same configuration, so a number of an earlier build is
    never printed for a configuration that the campaign could not measure.
    """
    cached = _STORE.derived.get("e2e_rows")
    if cached is not None:
        return cached
    rows: dict = {}
    for rel in reversed(E2E_SOURCES):                 # the preferred source is applied last
        doc = _STORE.load(rel)
        for where, row in _walk_rows(doc):
            key = _e2e_key(row)
            if key is None:
                continue
            name = _first(row, "name", "id", default="/".join(str(x) for x in key))
            sel = f"{where}[name={name}]"
            status, reason = row_status(row)
            if status != "ok":
                rows[key] = {"name": row.get("name"), "config": row.get("config"), "_file": rel, "_sel": sel,
                             "_not_ok": f"{config_label(key)}: {status}: {reason}"}
                continue
            if _first(row, "bytes", "bytes_both", "total_bytes") is None:
                continue
            rows[key] = dict(row, _file=rel, _sel=sel)
    _STORE.derived["e2e_rows"] = rows
    return rows


def e2e_ok(row) -> bool:
    """True for a row with measurements (not ``None`` and not the stub of a failed or infeasible run)."""
    return row is not None and "_not_ok" not in row


def e2e_not_ok() -> list:
    """The stubs of the configurations that a summary marks as failed or infeasible."""
    return [row for row in e2e_rows().values() if "_not_ok" in row]


def _walk_rows(doc, where: str = "", depth: int = 0):
    """Yield ``(section, row)`` for every dictionary in a list of the document (at most four levels deep).

    ``raw/e2e_cpp/summary.json`` keeps its rows under ``rows``; a campaign
    summary may group them under other names, so every list is searched.
    """
    if depth > 4:
        return
    if isinstance(doc, list):
        for row in doc:
            if isinstance(row, dict):
                yield where or "rows", row
    elif isinstance(doc, dict):
        for name, value in doc.items():
            if isinstance(value, list):
                yield from _walk_rows(value, f"{where}.{name}" if where else name, depth + 1)
            elif isinstance(value, dict) and name not in ("config", "params", "phases", "phases_detail"):
                yield from _walk_rows(value, f"{where}.{name}" if where else name, depth + 1)


def e2e(task: str, n: int, batch: int = 1, delay_ms: float = 0.0, *, k: int = 64,
        beta: str = "public", auth: bool = False, prep: str = "real", sigma: int = 40,
        threads: int = 1):
    """One end-to-end row, the stub of a failed or infeasible run (see :func:`e2e_rows`), or ``None``."""
    return e2e_rows().get((task, int(n), int(batch), float(delay_ms), int(k), beta, bool(auth),
                           prep, int(sigma), int(threads)))


def e2e_expected(task: str, n: int, batch: int = 1, delay_ms: float = 0.0, **config) -> str:
    """Where a missing end-to-end data point is expected (``config``: k, beta, auth, sigma, threads)."""
    extra = "".join(f", {name}={value}" for name, value in config.items())
    return (f"{E2E_SOURCES[0]} (or {E2E_SOURCES[1]}): row task={task}, n={n}, batch={batch}, "
            f"delay_ms={delay_ms:g}{extra}")


_E2E_ALIASES = {"bytes": ("bytes_both", "total_bytes"), "rounds": ("total_rounds",),
                "seconds": ("seconds_median", "wall_seconds", "wall_s")}


def _e2e_missing(row, field: str, expected: str):
    """The pending Num of a row without measurements, or ``None`` for a usable row."""
    if row is None:
        return pending(expected or f"{E2E_SOURCES[0]}: {field}")
    if "_not_ok" in row:
        return pending(f"{row['_file']}: {row['_sel']}.{field}", status=row["_not_ok"])
    return None


def e2e_num(row, field: str, *aliases: str, expected: str = "") -> Num:
    """A measured field of an end-to-end row (``row`` may be ``None`` or a stub)."""
    missing = _e2e_missing(row, field, expected)
    if missing is not None:
        return missing
    names = (field,) + aliases + _E2E_ALIASES.get(field, ())
    used = next((name for name in names if row.get(name) is not None), field)
    return measured(row.get(used), f"{row['_file']}: {row['_sel']}.{used}")


def e2e_path(row, *path: str, expected: str = "") -> Num:
    """A measured scalar below sub-dictionaries of a row, e.g. ``("online_layer", "bytes")``."""
    field = ".".join(path)
    missing = _e2e_missing(row, field, expected)
    if missing is not None:
        return missing
    cur = row
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else None
    if isinstance(cur, (dict, list)):
        cur = None
    return measured(cur, f"{row['_file']}: {row['_sel']}.{field}")


def e2e_phase(row, phase: str, field: str, expected: str = "") -> Num:
    """``phases[phase][field]`` of an end-to-end row; ``seconds`` is the slower party."""
    missing = _e2e_missing(row, f"phases.{phase}.{field}", expected)
    if missing is not None:
        return missing
    ph = (row.get("phases") or {}).get(phase)
    base = f"{row['_file']}: {row['_sel']}.phases.{phase}"
    if not isinstance(ph, dict):
        return pending(base)
    if field == "seconds":
        values = [ph[key] for key in ("seconds_p0", "seconds_p1", "seconds") if key in ph]
        if not values:
            return pending(base + ".seconds_p0/p1")
        return measured(max(values), base + ".max(seconds_p0, seconds_p1)")
    return measured(_first(ph, field, field + "_both"), f"{base}.{field}")


def e2e_phase_names(*rows) -> list:
    """The nine standard phases followed by any further phase that one of ``rows`` records."""
    extras = []
    for row in rows:
        for name in ((row or {}).get("phases") or {}):
            if name not in PHASES and name not in extras:
                extras.append(name)
    return list(PHASES) + sorted(extras)


def e2e_party_record(row, role: int = 0):
    """The per-party record ``<name>_p<role>.json`` that belongs to a summary row, or ``None``."""
    if not e2e_ok(row) or not row.get("name"):
        return None, ""
    # next to the summary the row came from; failing that, the record of the same
    # configuration in the other end-to-end directory
    first = os.path.dirname(row["_file"]) + f"/{row['name']}_p{role}.json"
    for rel in [first] + [os.path.dirname(src) + f"/{row['name']}_p{role}.json" for src in E2E_SOURCES]:
        doc = _STORE.load(rel)
        if doc is not None:
            return doc, rel
    return None, first


def summary_num(*path) -> Num:
    """A scalar of the campaign summary itself, addressed by a key path (e.g. ``("checks", "configurations")``)."""
    rel = E2E_SOURCES[0]
    field = ".".join(str(p) for p in path)
    cur = _STORE.load(rel)
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else None
    if isinstance(cur, (dict, list)):
        cur = None
    return measured(cur, f"{rel}: {field}")


def campaign_rows() -> list:
    """The configuration rows of the campaign summary (sections ``rows``, ``online``, ``apps``), stubs included."""
    rel = E2E_SOURCES[0]
    return [row for row in e2e_rows().values()
            if row["_file"] == rel and row["_sel"].split("[", 1)[0] in ("rows", "online", "apps")]


def campaign_total(field: str) -> Num:
    """Sum of an integer field over the measured configuration rows of the campaign summary."""
    src = f"sum of {E2E_SOURCES[0]}: rows[*].{field} + online[*].{field} + apps[*].{field}"
    values = [row.get(field) for row in campaign_rows() if e2e_ok(row)]
    if not values or any(not isinstance(v, int) or isinstance(v, bool) for v in values):
        return pending(src)
    return Num(sum(values), "derived", src)


def campaign_range(field: str, pick) -> Num:
    """``pick`` (``min`` or ``max``) of a field over the measured configuration rows of the campaign summary."""
    src = f"{pick.__name__} of {E2E_SOURCES[0]}: rows[*].{field}, online[*].{field}, apps[*].{field}"
    values = [row.get(field) for row in campaign_rows() if e2e_ok(row)]
    if not values or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
        return pending(src)
    return Num(pick(values), "derived", src)


def e2e_batches(task: str = "dpf", n: int = 16, delay_ms: float = 0.0) -> list:
    """Batch sizes measured for a configuration, ascending."""
    return sorted({key[2] for key in e2e_rows()
                   if key[0] == task and key[1] == n and key[3] == float(delay_ms)
                   and key[4:] == (64, "public", False, "real", 40, 1)})


def e2e_delays(task: str = "dpf", n: int = 16, batch: int = 1) -> list:
    """One-way delays (ms) measured for a configuration, ascending."""
    return sorted({key[3] for key in e2e_rows()
                   if key[0] == task and key[1] == n and key[2] == batch
                   and key[4:] == (64, "public", False, "real", 40, 1)})


# --------------------------------------------------------------------------- #
# Circuit statistics (compiler output)
# --------------------------------------------------------------------------- #

def circuit_row(kind: str, n: int | None = None, *, k: int = 64, s: int = 64,
                beta_mode: str = "public_one", adder: str = "ripple", gate: str | None = None):
    """A row of circuit_stats.json: kind in {"session_setup", "dpf", "dcf", "gate"}.

    Returns ``(row, selector)``; ``row`` is ``None`` if the file or the row is missing.
    """
    doc = _STORE.load(CIRCUITS_SOURCE)
    if not isinstance(doc, dict):
        return None, f"{CIRCUITS_SOURCE}"
    if kind == "session_setup":
        for row in doc.get("session_setup", []):
            if row.get("s") == s and row.get("adder") == adder:
                return row, f"{CIRCUITS_SOURCE}: session_setup[name={row.get('name')}]"
        return None, f"{CIRCUITS_SOURCE}: session_setup[s={s}, adder={adder}]"
    if kind in ("dpf", "dcf"):
        for row in doc.get("keys", []):
            if (row.get("kind") == kind and row.get("n") == n and row.get("k") == k
                    and row.get("s") == s and row.get("beta_mode") == beta_mode
                    and row.get("adder") == adder and row.get("index_mode", "xor") == "xor"):
                return row, f"{CIRCUITS_SOURCE}: keys[name={row.get('name')}]"
        return None, f"{CIRCUITS_SOURCE}: keys[{kind}, n={n}, k={k}, s={s}, {beta_mode}, {adder}]"
    if kind == "gate":
        for row in doc.get("gates", []):
            if (row.get("gate") == gate and row.get("n") == n and row.get("k") == k
                    and row.get("s") == s and row.get("adder") == adder):
                return row, f"{CIRCUITS_SOURCE}: gates[name={row.get('name')}]"
        return None, f"{CIRCUITS_SOURCE}: gates[{gate}, n={n}, k={k}, s={s}, {adder}]"
    raise ValueError(f"unknown circuit kind {kind!r}")


def circuit_num(kind: str, field: str, n: int | None = None, **kw) -> Num:
    """One statistic of a compiled circuit (kind "tool")."""
    row, sel = circuit_row(kind, n, **kw)
    if row is None:
        return pending(f"{sel}.{field}")
    return tool(row.get(field), f"{sel}.{field}")


def per_level_row(kind: str, *, k: int = 64, s: int = 64, beta_mode: str = "public_one",
                  adder: str = "ripple", **flags):
    """The affine form (per level, fixed) of a key circuit family; extra ``flags`` must match."""
    doc = _STORE.load(CIRCUITS_SOURCE)
    sel = f"{CIRCUITS_SOURCE}: per_level[{kind}, k={k}, s={s}, {beta_mode}, {adder}]"
    if not isinstance(doc, dict):
        return None, sel
    for row in doc.get("per_level", []):
        if (row.get("kind") == kind and row.get("k") == k and row.get("s") == s
                and row.get("beta_mode") == beta_mode and row.get("adder") == adder
                and all(row.get(name, False) == want for name, want in flags.items())):
            return row, sel
    return None, sel


def per_level_num(kind: str, what: str, part: str, **kw) -> Num:
    """``what`` in {"n_and", "and_depth"}, ``part`` in {"per_level", "fixed"}."""
    row, sel = per_level_row(kind, **kw)
    if row is None:
        return pending(f"{sel}.{what}.{part}")
    return tool((row.get(what) or {}).get(part), f"{sel}.{what}.{part}")


def primitive_num(name: str, field: str) -> Num:
    doc = _STORE.load(CIRCUITS_SOURCE)
    sel = f"{CIRCUITS_SOURCE}: primitives.{name}.{field}"
    if not isinstance(doc, dict):
        return pending(sel)
    return tool(((doc.get("primitives") or {}).get(name) or {}).get(field), sel)


_NO_INVARIANT_FLAGS = ("control_invariant", "invariant", "control_bit_invariant")


def no_invariant_variant():
    """A per-level row compiled WITHOUT the control-bit invariant, if the file has one.

    Recognised forms: a ``per_level`` row carrying ``control_invariant`` /
    ``invariant`` / ``control_bit_invariant`` = false, or a top-level section
    ``variants`` / ``ablation`` with a row whose ``variant`` or ``name``
    contains "no_invariant" / "noinv" / "naive".  Returns ``(row, selector)``
    or ``(None, description)``.
    """
    doc = _STORE.load(CIRCUITS_SOURCE)
    where = f"{CIRCUITS_SOURCE}: no per_level row with control_invariant = false"
    if not isinstance(doc, dict):
        return None, where
    for i, row in enumerate(doc.get("per_level", [])):
        if any(row.get(flag) is False for flag in _NO_INVARIANT_FLAGS):
            if (row.get("kind") == "dpf" and row.get("k") == 64 and row.get("s") == 64
                    and row.get("beta_mode") == "public_one" and row.get("adder", "ripple") == "ripple"):
                return row, f"{CIRCUITS_SOURCE}: per_level[{i}]"
    for section in ("variants", "ablation", "ablations"):
        for i, row in enumerate(_rows_of(doc, section)):
            label = str(_first(row, "variant", "name", "_key", default="")).lower()
            if any(tag in label for tag in ("no_invariant", "noinv", "naive", "without_invariant")):
                return row, f"{CIRCUITS_SOURCE}: {section}[{i}]"
    return None, where


# --------------------------------------------------------------------------- #
# Bucket parameters
# --------------------------------------------------------------------------- #

def bucket_min_ell(bucket: int, sigma: int) -> Num:
    """Smallest batch size for which bucket size ``bucket`` suffices at level ``sigma``."""
    doc = _STORE.load(BUCKET_SOURCE)
    sel = f"{BUCKET_SOURCE}: min_ell_for_bucket[{sigma}][{bucket}]"
    if not isinstance(doc, dict):
        return pending(sel)
    return tool(((doc.get("min_ell_for_bucket") or {}).get(str(sigma)) or {}).get(str(bucket)), sel)


def bucket_rows(sigma: int) -> list:
    """Rows (log2_ell, B, ...) of the parameter table for one sigma, ascending in ell."""
    doc = _STORE.load(BUCKET_SOURCE)
    if not isinstance(doc, dict):
        return []
    return sorted((r for r in doc.get("rows", []) if r.get("sigma") == sigma),
                  key=lambda r: r.get("log2_ell", 0))


def bucket_for(ell_log2: int, sigma: int) -> Num:
    for row in bucket_rows(sigma):
        if row.get("log2_ell") == ell_log2:
            return tool(row.get("B"), f"{BUCKET_SOURCE}: rows[sigma={sigma}, log2_ell={ell_log2}].B")
    return pending(f"{BUCKET_SOURCE}: rows[sigma={sigma}, log2_ell={ell_log2}].B")


# --------------------------------------------------------------------------- #
# Tool groups of the campaign (local benchmarks, preprocessing unit costs, native baselines)
# --------------------------------------------------------------------------- #

# section of raw/campaign/summary.json -> (label, files whose content that group produced)
TOOL_GROUPS = {
    "local": ("local benchmarks", LOCAL_SOURCES),
    "prep_unit_costs": ("preprocessing benchmark", (PREP_SOURCE,)),
    "baselines": ("native baselines", (BASELINE_SOURCES[0][1],)),
}


def tool_failure(rel: str) -> str:
    """Status text if the campaign summary marks the tool group that wrote ``rel`` as not ok, else ``""``.

    Without a campaign summary (or without a section for the group) the file
    is taken as it is.
    """
    doc = _STORE.load(E2E_SOURCES[0])
    if not isinstance(doc, dict):
        return ""
    for group, (label, files) in TOOL_GROUPS.items():
        section = doc.get(group)
        if rel in files and isinstance(section, dict):
            status, reason = row_status(section)
            if status != "ok":
                return f"{label} ({rel}): {status}: {reason}"
    return ""


# --------------------------------------------------------------------------- #
# Preprocessing benchmark
# --------------------------------------------------------------------------- #

def prep_rows(threads: int = 1) -> list:
    """Rows of the preprocessing benchmark for one thread count, ascending in ell."""
    doc = _STORE.load(PREP_SOURCE)
    rows = [r for r in _rows_of(doc, "rows") if r.get("threads_per_party", 1) == threads]
    return sorted(rows, key=lambda r: r.get("ell", 0))


def prep_row(bucket: int, threads: int = 1, prefer_log2_ell: int | None = None):
    """The benchmark row used for bucket size ``bucket``: ``prefer_log2_ell`` if it has
    that bucket size, otherwise the largest batch with that bucket size.  Returns
    ``(row, selector)``."""
    rows = [r for r in prep_rows(threads) if r.get("B") == bucket and r.get("sigma", 40) == 40]
    sel = f"{PREP_SOURCE}: rows[threads_per_party={threads}, B={bucket}]"
    if not rows:
        return None, sel
    chosen = rows[-1]
    for r in rows:
        if prefer_log2_ell is not None and r.get("log2_ell") == prefer_log2_ell:
            chosen = r
    return chosen, f"{PREP_SOURCE}: rows[threads_per_party={threads}, log2_ell={chosen.get('log2_ell')}]"


def prep_num(row, sel: str, *path: str) -> Num:
    """A measured field of a preprocessing row, addressed by a key path."""
    field = ".".join(path)
    bad = tool_failure(PREP_SOURCE)
    if row is None or bad:
        return pending(f"{sel}.{field}", status=bad)
    cur = row
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else None
        if cur is None:
            return pending(f"{sel}.{field}")
    return measured(cur, f"{sel}.{field}")


def prep_phase_seconds(row, sel: str, phase: str) -> Num:
    """Seconds of one preprocessing phase (the slower party)."""
    bad = tool_failure(PREP_SOURCE)
    if row is None or bad:
        return pending(f"{sel}.phases.{phase}.seconds", status=bad)
    ph = (row.get("phases") or {}).get(phase) or {}
    values = [ph[key] for key in ("seconds_p0", "seconds_p1", "seconds") if key in ph]
    if not values:
        return pending(f"{sel}.phases.{phase}.seconds_p0/p1")
    return measured(max(values), f"{sel}.phases.{phase}.max(seconds_p0, seconds_p1)")


def prep_meta(field: str) -> Num:
    doc = _STORE.load(PREP_SOURCE)
    bad = tool_failure(PREP_SOURCE)
    if not isinstance(doc, dict) or bad:
        return pending(f"{PREP_SOURCE}: {field}", status=bad)
    return measured(doc.get(field), f"{PREP_SOURCE}: {field}")


# --------------------------------------------------------------------------- #
# Baselines (re-implemented leaky and semi-honest protocols)
# --------------------------------------------------------------------------- #

_BASELINE_SECTIONS = {
    "leaky": ("leaky_zhang", "leaky", "zhang", "leaky_dpf", "pi_dpf"),
    "semihonest": ("semihonest_ds", "semihonest", "semi_honest", "semi-honest", "ds"),
}
_BASELINE_PROTOCOL_TAGS = {"leaky": ("leaky", "zhang"), "semihonest": ("semi", "ds")}


def _baseline_rows(doc, protocol: str) -> list:
    rows = _rows_of(doc, *_BASELINE_SECTIONS[protocol])
    if rows:
        return rows
    out = []
    for row in _rows_of(doc, "rows", "results"):
        label = str(_first(row, "protocol", "baseline", "name", default="")).lower()
        if any(tag in label for tag in _BASELINE_PROTOCOL_TAGS[protocol]):
            out.append(row)
    return out


def baseline_row(protocol: str, n: int, native_only: bool = False):
    """Row of a baseline for domain size ``n``: ``(row, file, implementation)`` or ``(None, expected, "")``.

    The native (C++) file wins when it has the row; otherwise the Python
    reference is used unless ``native_only``.
    """
    for impl, rel in BASELINE_SOURCES:
        if (native_only and impl != "C++") or tool_failure(rel):
            continue
        doc = _STORE.load(rel)
        for row in _baseline_rows(doc, protocol):
            if row.get("n") == n:
                usable = (_baseline_transport(row)[0] is not None or isinstance(row.get("total"), dict)
                          or "bytes_both" in row or "bytes" in row)
                if not usable:                          # e.g. a failed native run: try the next file
                    continue
                section = next((s for s in _BASELINE_SECTIONS[protocol]
                                if isinstance(doc, dict) and s in doc), "rows")
                return row, f"{rel}: {section}[n={n}]", impl
    expected = BASELINE_SOURCES[0][1] if native_only else " or ".join(r for _, r in BASELINE_SOURCES)
    return None, f"{expected}: {protocol}[n={n}]", ""


BASELINE_TRANSPORTS = ("tcp_two_processes", "loopback_two_threads")


def _baseline_transport(row: dict):
    """The per-transport record of a native baseline row (layout of ``bench_baselines.cpp``).

    Two processes over TCP are preferred (the setting of the end-to-end runs
    of this work); the in-process transport is the fallback.  Returns
    ``(record, name)`` or ``(None, "")`` for the layout of the Python benchmark.
    """
    for name in BASELINE_TRANSPORTS:
        sub = row.get(name)
        if isinstance(sub, dict) and isinstance(sub.get("per_key_without_base_ot"), dict):
            return sub, name
    return None, ""


def _median(values: list):
    values = sorted(values)
    return values[len(values) // 2] if len(values) % 2 else 0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])


def _baseline_num_native(rec: dict, sel: str, what: str) -> Num:
    """Figures of one transport record of the native baselines benchmark.

    The record has no session total: bytes and rounds of a fresh session are
    the sum of ``base_ot`` and ``per_key_without_base_ot``; the session time
    is, per repetition, the larger of the two parties' (base OT + key) times,
    and then the median over the repetitions.
    """
    key, ot = rec.get("per_key_without_base_ot") or {}, rec.get("base_ot") or {}
    part = lambda group, sub, name: measured(sub.get(name), f"{sel}.{group}.{name}")  # noqa: E731
    if what in ("bytes", "rounds"):
        name = "bytes_both" if what == "bytes" else "rounds"
        return calc(lambda a, b: a + b, "{0} + {1}", part("base_ot", ot, name), part("per_key_without_base_ot", key, name))
    if what == "seconds":
        runs = rec.get("runs") or []
        totals = []
        for run in runs:
            try:
                totals.append(max(run["base_ot_seconds_p0"] + run["key_seconds_p0"],
                                  run["base_ot_seconds_p1"] + run["key_seconds_p1"]))
            except (KeyError, TypeError):
                totals = []
                break
        if totals:
            return Num(_median(totals), "derived",
                       f"median over [{sel}.runs] of max over the parties of (base_ot_seconds_p + key_seconds_p)")
        return calc(lambda a, b: a + b, "{0} + {1}", part("base_ot", ot, "seconds_median"),
                    part("per_key_without_base_ot", key, "seconds_median"), note="sum of two medians")
    if what in ("bytes_per_key", "rounds_per_key", "seconds_per_key"):
        name = {"bytes_per_key": "bytes_both", "rounds_per_key": "rounds", "seconds_per_key": "seconds_median"}[what]
        return part("per_key_without_base_ot", key, name)
    if what == "pi_dpf_bytes":
        return part("protocol_messages", rec.get("protocol_messages") or {}, "bytes_both")
    if what == "abit_bytes":
        return part("authenticated_bits", rec.get("authenticated_bits") or {}, "bytes_both")
    raise ValueError(f"unknown baseline figure {what!r}")


def baseline_num(protocol: str, n: int, what: str, native_only: bool = False) -> Num:
    """A figure of a baseline run.

    ``what``: "bytes" / "rounds" / "seconds" (one key in a fresh session,
    base OT included), "bytes_per_key" / "rounds_per_key" / "seconds_per_key"
    (without the base OT), "pi_dpf_bytes" (messages of the protocol itself,
    leaky baseline), "abit_bytes" (authenticated bits, leaky baseline).
    """
    row, sel, _impl = baseline_row(protocol, n, native_only)
    if row is None:
        return pending(f"{sel}.{what}", status=tool_failure(BASELINE_SOURCES[0][1]))
    mode, mode_name = _baseline_transport(row)
    if mode is not None:
        return _baseline_num_native(mode, f"{sel}.{mode_name}", what)
    groups = {
        "bytes": ("total", ("bytes_both", "bytes")),
        "rounds": ("total", ("rounds",)),
        "seconds": ("total", ("seconds_median", "seconds")),
        "bytes_per_key": ("per_key_without_base_ot", ("bytes_both", "bytes")),
        "rounds_per_key": ("per_key_without_base_ot", ("rounds",)),
        "seconds_per_key": ("per_key_without_base_ot", ("seconds_median", "seconds")),
    }
    if what in groups:
        group, names = groups[what]
        sub = row.get(group)
        if isinstance(sub, dict):
            value = _first(sub, *names)
            return measured(value, f"{sel}.{group}.{names[0]}")
        flat = {"bytes": ("bytes_both", "bytes", "total_bytes"), "rounds": ("rounds", "total_rounds"),
                "seconds": ("seconds_median", "seconds", "wall_seconds")}.get(what, ())
        return measured(_first(row, *flat), f"{sel}.{flat[0] if flat else what}")
    if what == "pi_dpf_bytes":
        return measured(_first(row, "pi_dpf_bytes_both", "pi_dpf_bytes"), f"{sel}.pi_dpf_bytes_both")
    if what == "abit_bytes":
        return measured(_first(row, "abit_bytes_both", "abit_bytes"), f"{sel}.abit_bytes_both")
    raise ValueError(f"unknown baseline figure {what!r}")


def baseline_impl(protocol: str, n: int) -> str:
    """"C++", "Python" or "" (no data)."""
    return baseline_row(protocol, n)[2]


# --------------------------------------------------------------------------- #
# Cost models and reported numbers (couplet/baselines/costmodels.py)
# --------------------------------------------------------------------------- #

def _costmodel_live(n: int):
    """One row of ``costmodels.comparison_rows`` computed now (fallback when the JSON lacks it)."""
    py = os.path.normpath(os.path.join(_HERE, ".."))
    if py not in sys.path:
        sys.path.insert(0, py)
    try:
        from couplet.baselines import costmodels as cm
        return cm.comparison_rows((n,))[0]
    except Exception:                                   # the package may be unavailable
        return None


def costmodel_entry(n: int, *path: str) -> Num:
    """An entry of the cost-model rows for domain size ``n``.

    Read from ``raw/baselines_python.json`` (section ``cost_models``, written
    by ``bench/bench_baselines.py`` from ``costmodels.comparison_rows``); if
    the file has no row for ``n`` the same function is evaluated directly.
    The kind follows the entry: "reported" stays reported, "derived" becomes
    "model" (it is printed in red), and an entry of kind "measured" is not
    expected here.
    """
    doc = _STORE.load(COSTMODEL_SOURCE)
    row, origin = None, f"{COSTMODEL_SOURCE}: cost_models[n={n}]"
    for cand in _rows_of(doc, "cost_models"):
        if cand.get("n") == n:
            row = cand
    if row is None:
        row = _costmodel_live(n)
        origin = f"couplet/baselines/costmodels.py: comparison_rows(({n},))[0]"
    if row is None:
        return pending(origin + "." + ".".join(path))
    cur = row
    for key in path:
        cur = cur.get(key) if isinstance(cur, dict) else None
        if cur is None:
            return pending(origin + "." + ".".join(path))
    if isinstance(cur, dict) and "value" in cur:
        kind = "reported" if cur.get("kind") == "reported" else "model"
        return Num(cur["value"], kind, origin + "." + ".".join(path) + ".value",
                   str(cur.get("formula", "")))
    return Num(cur, "model", origin + "." + ".".join(path))


def reported_entries(section: str) -> list:
    """Entries of ``reported_by_other_papers[section]`` in the baselines file."""
    doc = _STORE.load(COSTMODEL_SOURCE)
    if not isinstance(doc, dict):
        return []
    sec = (doc.get("reported_by_other_papers") or {}).get(section)
    if isinstance(sec, dict):
        return [dict(v, _key=k) for k, v in sec.items()]
    return list(sec or [])


def python_measured_bytes_per_and(bucket: int) -> Num:
    """Bytes per AND triple measured with the Python reference (bench_baselines)."""
    doc = _STORE.load(COSTMODEL_SOURCE)
    for i, entry in enumerate(_rows_of(doc, "backend_bytes_per_and_measured")):
        if entry.get("bucket") == bucket:
            return measured(entry.get("value"),
                            f"{COSTMODEL_SOURCE}: backend_bytes_per_and_measured[{i}].value")
    return pending(f"{COSTMODEL_SOURCE}: backend_bytes_per_and_measured[bucket={bucket}].value")


# --------------------------------------------------------------------------- #
# Q1: one DPF key, this work and the two baselines
# --------------------------------------------------------------------------- #

def q1_numbers(n: int) -> dict:
    """Everything the Q1 table, the Q1 figure and the macros need for one domain size.

    ``*_bytes`` / ``*_rounds`` / ``*_seconds``: one key in a fresh session,
    base OT included.  ``*_after_ot``: the same without the base OT of the
    session (for this work: the total minus the base-OT phase; for the
    baselines: the benchmark's ``per_key_without_base_ot``).  Baseline times
    are taken from the native implementation only.
    """
    e = e2e("dpf", n)
    exp = e2e_expected("dpf", n)
    out = {
        "ours_bytes": e2e_num(e, "bytes", expected=exp),
        "ours_rounds": e2e_num(e, "rounds", expected=exp),
        "ours_seconds": e2e_num(e, "seconds", expected=exp),
        "ours_seconds_min": e2e_num(e, "seconds_min", expected=exp),
        "ours_seconds_max": e2e_num(e, "seconds_max", expected=exp),
        "ours_repetitions": e2e_num(e, "repetitions", expected=exp),
        "ours_peak_bytes": e2e_path(e, "peak_memory", "max_working_set_bytes", expected=exp),
        "ours_bucket": e2e_num(e, "bucket_size", expected=exp),
        "ours_triples": e2e_num(e, "triples", expected=exp),
        "ours_bytes_per_and": e2e_num(e, "bytes_per_and_gate", expected=exp),
        "ours_key_bytes": e2e_num(e, "output_bytes_per_instance_and_party", expected=exp),
        "baseot_bytes": e2e_phase(e, "baseot", "bytes", exp),
        "baseot_rounds": e2e_phase(e, "baseot", "rounds", exp),
        "baseot_seconds": e2e_phase(e, "baseot", "seconds", exp),
    }
    out["ours_bytes_after_ot"] = calc(lambda a, b: a - b, "{0} - {1}", out["ours_bytes"], out["baseot_bytes"])
    out["ours_rounds_after_ot"] = calc(lambda a, b: a - b, "{0} - {1}", out["ours_rounds"], out["baseot_rounds"])
    out["ours_seconds_after_ot"] = calc(lambda a, b: a - b, "{0} - {1}", out["ours_seconds"], out["baseot_seconds"])
    for proto in ("leaky", "semihonest"):
        out[f"{proto}_bytes"] = baseline_num(proto, n, "bytes")
        out[f"{proto}_rounds"] = baseline_num(proto, n, "rounds")
        out[f"{proto}_seconds"] = baseline_num(proto, n, "seconds", native_only=True)
        out[f"{proto}_bytes_after_ot"] = baseline_num(proto, n, "bytes_per_key")
        out[f"{proto}_rounds_after_ot"] = baseline_num(proto, n, "rounds_per_key")
        out[f"{proto}_seconds_after_ot"] = baseline_num(proto, n, "seconds_per_key", native_only=True)
    out["leaky_pi_dpf_bytes"] = baseline_num("leaky", n, "pi_dpf_bytes")
    out["leaky_abit_bytes"] = baseline_num("leaky", n, "abit_bytes")
    out["bcg_bytes"] = costmodel_entry(n, "bcg20_generic_2pc")
    out["zhang_hybrid_bytes"] = calc(lambda bits: bits / 8, "{0} / 8", costmodel_entry(n, "zhang_hybrid_bits"))
    out["zhang_hybrid_rounds"] = costmodel_entry(n, "zhang_hybrid_rounds")
    ratio = lambda a, b: calc(lambda x, y: x / y, "{0} / {1}", out[a], out[b])  # noqa: E731
    out["ratio_bytes_leaky"] = ratio("ours_bytes", "leaky_bytes")
    out["ratio_bytes_semi"] = ratio("ours_bytes", "semihonest_bytes")
    out["ratio_bytes_after_ot_leaky"] = ratio("ours_bytes_after_ot", "leaky_bytes_after_ot")
    out["ratio_bytes_after_ot_semi"] = ratio("ours_bytes_after_ot", "semihonest_bytes_after_ot")
    out["ratio_rounds_leaky"] = ratio("ours_rounds", "leaky_rounds")
    out["ratio_seconds_leaky"] = ratio("ours_seconds", "leaky_seconds")
    out["ratio_seconds_semi"] = ratio("ours_seconds", "semihonest_seconds")
    out["ratio_seconds_after_ot_leaky"] = ratio("ours_seconds_after_ot", "leaky_seconds_after_ot")
    out["ratio_seconds_after_ot_semi"] = ratio("ours_seconds_after_ot", "semihonest_seconds_after_ot")
    # the inverse ratios (baseline over this work), for the domain sizes where this work is faster
    out["inverse_seconds_after_ot_leaky"] = ratio("leaky_seconds_after_ot", "ours_seconds_after_ot")
    out["inverse_seconds_after_ot_semi"] = ratio("semihonest_seconds_after_ot", "ours_seconds_after_ot")
    out["inverse_seconds_leaky"] = ratio("leaky_seconds", "ours_seconds")
    out["inverse_seconds_semi"] = ratio("semihonest_seconds", "ours_seconds")
    return out


# --------------------------------------------------------------------------- #
# Local evaluation (Eval, EvalAll, key sizes, gates)
# --------------------------------------------------------------------------- #

_TIME_UNITS = (("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3), ("seconds", 1.0), ("s", 1.0))


def _time_field(row: dict, prefer: str = "median"):
    """``(seconds, field name)`` of the first time-like field of ``row`` (median preferred).

    A field is time-like if its name is a unit or starts with a unit and an
    underscore (``seconds_median``, ``ns_per_eval_median``, ``us``).  The bare
    name ``s`` is NOT a time: the benchmark rows use it for the MAC parameter.
    """
    best = None
    for name, value in row.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        for unit, factor in _TIME_UNITS:
            if (name == unit and unit != "s") or name.startswith(unit + "_"):
                rank = 0 if prefer in name else (2 if ("min" in name or "max" in name) else 1)
                if best is None or rank < best[0]:
                    best = (rank, value * factor, name)
                break
    return (None, "") if best is None else (best[1], best[2])


def _local_match(row: dict, family: str, n: int, k: int, s: int) -> bool:
    fam = str(_first(row, "family", "kind", "scheme", default="")).lower()
    return (fam == family.lower() and row.get("n") == n and row.get("k", 64) == k
            and row.get("s", 64) == s)


def local_num(section: str, family: str, n: int, *, k: int = 64, s: int = 64, threads: int = 1) -> Num:
    """A local benchmark figure in SI base units.

    ``section``: "eval" (seconds per point evaluation), "gen" (seconds per
    plaintext dealer Gen), "eval_all" (seconds for the whole domain, one
    thread unless ``threads``), "key_bytes" (bytes of one key).
    """
    expected = f"{' or '.join(LOCAL_SOURCES)}: {section}[{family}, n={n}, k={k}]"
    bad = next((text for text in map(tool_failure, LOCAL_SOURCES) if text), "")
    if bad:
        return pending(expected, status=bad)
    for rel in LOCAL_SOURCES:
        doc = _STORE.load(rel)
        sections = (section,) if threads == 1 or section != "eval_all" else ("eval_all_threads",)
        for sec in sections:
            for row in _rows_of(doc, sec):
                if not _local_match(row, family, n, k, s):
                    continue
                if section == "eval_all" and row.get("threads", 1) != threads:
                    continue
                sel = f"{rel}: {sec}[{family}, n={n}, k={k}]"
                if section == "key_bytes":
                    return measured(_first(row, "bytes", "key_bytes"), f"{sel}.bytes")
                seconds, name = _time_field(row)
                if seconds is None:
                    continue
                return measured(seconds, f"{sel}.{name} (in seconds)")
    return pending(expected)


def local_ns(family: str, section: str = "eval_all", *, k: int = 64) -> list:
    """Domain sizes for which ``section`` has a single-thread row of ``family``."""
    found = set()
    for rel in LOCAL_SOURCES:
        for row in _rows_of(_STORE.load(rel), section):
            fam = str(_first(row, "family", "kind", "scheme", default="")).lower()
            if fam == family.lower() and row.get("k", 64) == k and row.get("threads", 1) == 1:
                if isinstance(row.get("n"), int):
                    found.add(row["n"])
    return sorted(found)


_GATE_SECTIONS = ("gates", "gate_eval", "gate_local", "gate", "gate_evaluation", "local_gates")
_GATE_ALIASES = {
    "table": ("table", "gate-table", "table_gate", "lookup", "tablegate"),
    "zero": ("zero", "zero_test", "equality", "eq"),
    "comparison": ("comparison", "cmp", "gate-cmp", "compare", "less_than"),
    "interval": ("interval",),
    "relu": ("relu", "relu_large"),
    "spline": ("spline", "linear_spline"),
}


def gate_local_rows() -> list:
    """All rows of the gate section of the local benchmark (empty if there is none)."""
    rel = LOCAL_SOURCES[0]
    doc = _STORE.load(rel)
    out = []
    for sec in _GATE_SECTIONS:
        for row in _rows_of(doc, sec):
            out.append(dict(row, _sel=f"{rel}: {sec}"))
    return out


def _gate_label(row: dict) -> str:
    return str(_first(row, "gate", "name", "kind", "_key", default="")).lower()


def gate_local_num(gate: str, n: int, what: str = "seconds", *, k: int | None = 64) -> Num:
    """Local evaluation time (seconds) or material size (bytes) of a gate; pending if not benchmarked.

    ``k = None`` selects the row with ``k = n`` (the ReLU bundle is measured
    on ``k``-bit inputs, ``n = k``).  ``what`` may also name another time
    field of the row by its prefix: ``"expand_seconds"`` or
    ``"convolution_seconds"`` (the two parts of the table gate).
    """
    k = n if k is None else k
    expected = f"{LOCAL_SOURCES[0]}: gates[{gate}, n={n}, k={k}].{what}"
    bad = tool_failure(LOCAL_SOURCES[0])
    if bad:
        return pending(expected, status=bad)
    for row in gate_local_rows():
        label = _gate_label(row)
        if label not in _GATE_ALIASES.get(gate, (gate,)):
            continue
        if row.get("n") != n or row.get("k", 64) != k:
            continue
        sel = f"{row['_sel']}[{label}, n={n}, k={k}]"
        if what == "bytes":
            value = _first(row, "bytes", "key_bytes", "material_bytes", "bytes_per_party")
            return measured(value, f"{sel}.bytes")
        if what != "seconds":
            name = what + "_median"
            return measured(row.get(name), f"{sel}.{name} (in seconds)")
        seconds, name = _time_field(row)
        if seconds is not None:
            return measured(seconds, f"{sel}.{name} (in seconds)")
    return pending(expected)


def gate_local_ns(gate: str, *, k: int | None = 64) -> list:
    """Domain sizes with a row of ``gate`` in the local benchmark (``k = None``: the rows with ``k = n``)."""
    found = set()
    for row in gate_local_rows():
        if _gate_label(row) in _GATE_ALIASES.get(gate, (gate,)) and isinstance(row.get("n"), int):
            if row.get("k", 64) == (row["n"] if k is None else k):
                found.add(row["n"])
    return sorted(found)


# --------------------------------------------------------------------------- #
# Attacks
# --------------------------------------------------------------------------- #

def _natural_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def attack_ids() -> list:
    doc = _STORE.load(ATTACKS_SOURCE)
    if not isinstance(doc, dict):
        return []
    return sorted((doc.get("attacks") or {}).keys(), key=_natural_key)


def attack(name: str) -> dict:
    doc = _STORE.load(ATTACKS_SOURCE) or {}
    return (doc.get("attacks") or {}).get(name) or {}


def attack_num(name: str, field: str) -> Num:
    record = attack(name)
    return measured(record.get(field), f"{ATTACKS_SOURCE}: attacks.{name}.{field}")


def distinguisher_ids() -> list:
    doc = _STORE.load(ATTACKS_SOURCE)
    if not isinstance(doc, dict):
        return []
    return sorted((doc.get("distinguishing") or {}).keys(), key=_natural_key)


def distinguisher(name: str) -> dict:
    doc = _STORE.load(ATTACKS_SOURCE) or {}
    return (doc.get("distinguishing") or {}).get(name) or {}


def attacks_meta(field: str) -> Num:
    doc = _STORE.load(ATTACKS_SOURCE) or {}
    return measured((doc.get("meta") or {}).get(field), f"{ATTACKS_SOURCE}: meta.{field}")


# --------------------------------------------------------------------------- #
# Machine and test counts
# --------------------------------------------------------------------------- #

def machine_num(field: str, *aliases: str) -> Num:
    """A field of raw/machine.json (searched one level deep as well)."""
    doc = _STORE.load(MACHINE_SOURCE)
    if isinstance(doc, dict):
        value = _first(doc, field, *aliases)
        if value is None:
            for sub in doc.values():
                value = _first(sub, field, *aliases)
                if value is not None:
                    break
        if value is not None and not isinstance(value, (dict, list)):
            return measured(value, f"{MACHINE_SOURCE}: {field}")
    return pending(f"{MACHINE_SOURCE}: {field}")


def test_suites() -> list:
    """Test-suite records from raw/tests.json: ``(rows, file)``.

    Accepted layouts: ``{"suites": [...]}``, ``{"python": {...}, "cpp": {...}}``
    with per-suite dictionaries, or a top-level list.  A row has ``name``,
    ``language``, ``checks`` and ``failures`` (of the last attempt; ``None``
    if the program printed no summary line), ``passed`` (the last attempt
    ended with exit code 0, a summary line and no failure) and ``attempts``
    (how often the program was run, see :func:`collect_tests`).  The check
    counts of the macros are sums over the rows that have a summary line; the
    rows without one still count as programs that did not pass.
    """
    for rel in TESTS_SOURCES:
        doc = _STORE.load(rel)
        if doc is None:
            continue
        rows = []
        for row in _rows_of(doc, "suites", "tests", "results"):
            rows.append(row)
        if not rows and isinstance(doc, dict):
            for lang in ("python", "cpp", "c++"):
                for row in _rows_of(doc, lang):
                    rows.append(dict(row, language=lang, name=_first(row, "name", "_key", default="")))
        norm = []
        for row in rows:
            checks = _first(row, "checks", "n_checks")
            failures = _first(row, "failures", "failed")
            if checks is None and "exit_code" not in row:
                continue                                    # not a record of a test program
            lang = str(_first(row, "language", "lang", default="")).lower()
            name = str(_first(row, "name", "suite", "script", "_key", default=""))
            if not lang:
                lang = "python" if name.endswith(".py") else "cpp"
            failures = None if checks is None else int(failures or 0)
            passed = checks is not None and failures == 0 and row.get("exit_code", 0) == 0
            norm.append({"name": name, "language": "cpp" if lang in ("cpp", "c++") else lang,
                         "checks": None if checks is None else int(checks), "failures": failures, "passed": passed,
                         "attempts": int(_first(row, "attempts_needed", default=len(row.get("attempts") or [1])))})
        if norm:
            return norm, rel
    return [], TESTS_SOURCES[0]


_INT = r"(\d[\d,]*)"
_SUMMARY_FORMS = (
    # (pattern, checks from the groups, failures from the groups)
    (re.compile(_INT + r" checks, (?:" + _INT + r" passed, )?" + _INT + r" (?:failures|failed)"),
     lambda g: g[0], lambda g: g[2]),                           # N checks, [P passed,] M failures|failed
    (re.compile(_INT + r" checks passed, " + _INT + r" failed"),
     lambda g: g[0] + g[1], lambda g: g[1]),                    # P checks passed, M failed
    (re.compile(r"TOTAL CHECKS: " + _INT + r", FAILED: " + _INT),
     lambda g: g[0], lambda g: g[1]),                           # TOTAL CHECKS: N, FAILED: M
    (re.compile(r"TOTAL CHECKS: " + _INT + r"\s+\((?:[^)]*, )?all passed"),
     lambda g: g[0], lambda g: 0),                              # TOTAL CHECKS: N  (..., all passed, ...)
    (re.compile(r"FAILED: " + _INT + r" of \d+ tests .*; " + _INT + r" checks run"),
     lambda g: g[1], lambda g: g[0]),                           # FAILED: x of y tests (...); N checks run
)


def parse_test_summary(output: str):
    """``(checks, failures, line)`` from the last summary line of a test program, or ``None``.

    The test programs of the artifact print one of five summary forms (see
    ``_SUMMARY_FORMS``).  For the form "FAILED: x of y tests" the number of
    failed checks is not printed; the number of failed tests is recorded.
    """
    for line in reversed(output.splitlines()):
        for pattern, checks, failures in _SUMMARY_FORMS:
            m = pattern.search(line)
            if m:
                groups = [int(x.replace(",", "")) if x else 0 for x in m.groups()]
                return checks(groups), failures(groups), line.strip()
    return None


def _test_jobs() -> list:
    """``(language, name, command, working directory, shown command)`` of every test program."""
    jobs = []
    py_dir = os.path.join(ARTIFACT, "python")
    tests_dir = os.path.join(py_dir, "tests")
    if os.path.isdir(tests_dir):
        for name in sorted(os.listdir(tests_dir)):
            if name.startswith("test_") and name.endswith(".py"):
                jobs.append(("python", name, [sys.executable, os.path.join("tests", name)], py_dir,
                             "python tests/" + name))
    build_dir = os.path.join(ARTIFACT, "cpp", "build")
    if os.path.isdir(build_dir):
        seen = set()
        for name in sorted(os.listdir(build_dir)):
            stem, ext = os.path.splitext(name)
            path = os.path.join(build_dir, name)
            if not (stem.startswith("test_") and ext in ("", ".exe") and os.path.isfile(path)) or stem in seen:
                continue
            if ext == "" and not os.access(path, os.X_OK):
                continue
            seen.add(stem)
            jobs.append(("cpp", stem, [path], build_dir, "build/" + stem))
    return jobs


def _attempt_passed(attempt: dict) -> bool:
    return attempt.get("exit_code") == 0 and attempt.get("failures") == 0 and attempt.get("checks") is not None


def collect_tests(out_path: str | None = None, only=None, timeout: float = 3600.0, retries: int = 1,
                  resume_from: str | None = None) -> dict:
    """Run the test programs of both implementations and write their check counts as JSON.

    Python: every ``tests/test_*.py`` (run from ``artifact/python``).  C++:
    every ``test_*`` executable in ``artifact/cpp/build`` (the programs must
    have been built).  A program whose output has no summary line is recorded
    with ``"checks": null`` and is not counted by the macros.  The result is
    what ``test_suites()`` reads (default path ``raw/tests.json``).

    Every run of a program is kept in the list ``attempts`` of its record
    (exit code, checks, failures, summary line, the output lines that report
    a failure).  A program that does not pass is run again, at most
    ``retries`` more times, because some suites assert wall-clock times and
    the machine is shared; the top-level fields of a record are those of its
    LAST attempt, ``attempts_needed`` says how many there were, and the
    document lists the repeated programs in ``suites_rerun`` and states
    ``all_passed_first_attempt``.  Nothing is dropped: a failed attempt stays
    in the file.

    ``resume_from``: a document written by an earlier call.  Its records are
    taken as the attempts made so far; programs that passed there are not run
    again, the others get their further attempts now.
    """
    import datetime
    import platform
    import subprocess
    import time

    earlier = {}
    if resume_from:
        with open(resume_from, "r", encoding="utf-8") as fh:
            prev = json.load(fh)
        for rec in prev.get("suites", []):
            attempts = rec.get("attempts") or [{key: rec.get(key) for key in
                                                ("exit_code", "checks", "failures", "summary_line", "seconds")}]
            for attempt in attempts:
                attempt.setdefault("failing_lines", None)          # not recorded by the first version of the collector
                if "timestamp_utc" not in attempt:                 # only the time at which that document was written is known
                    attempt["timestamp_utc"] = None
                    attempt["from_document_written_utc"] = prev.get("timestamp_utc")
            earlier[(rec.get("language"), rec.get("name"))] = attempts

    def run_once(cmd, cwd) -> dict:
        start = time.perf_counter()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors="replace")
            code, text = proc.returncode, (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            code, text = None, ""
        parsed = parse_test_summary(text)
        failing = [line.strip()[:300] for line in text.splitlines()
                   if re.search(r"\bFAIL(ED|URE)?\b|Traceback|Error:", line) and not re.search(r"\b0 (failures|failed)\b", line)]
        return {"exit_code": code, "checks": parsed[0] if parsed else None, "failures": parsed[1] if parsed else None,
                "summary_line": parsed[2] if parsed else None, "seconds": round(time.perf_counter() - start, 1),
                "failing_lines": failing[:20], "timestamp_utc": stamp}

    suites = []
    for language, name, cmd, cwd, shown in _test_jobs():
        if only and name not in only and os.path.splitext(name)[0] not in only:
            continue
        attempts = list(earlier.get((language, name), []))
        while not attempts or (not _attempt_passed(attempts[-1]) and len(attempts) < 1 + retries):
            attempts.append(run_once(cmd, cwd))
            last = attempts[-1]
            print(f"  {shown:<40} attempt {len(attempts)}: exit {last['exit_code']}  "
                  f"{last['summary_line'] or 'no summary line'}", flush=True)
        last = attempts[-1]
        suites.append({"name": name, "language": language, "command": shown, "exit_code": last["exit_code"],
                       "checks": last["checks"], "failures": last["failures"], "summary_line": last["summary_line"],
                       "seconds": last["seconds"], "attempts_needed": len(attempts), "attempts": attempts})
    doc = {"schema": "couplet.tests/2",
           "generated_by": "python bench/paper_numbers.py --collect-tests",
           "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "python": platform.python_version(), "platform": platform.platform(),
           "note": "checks and failures are parsed from the summary line that each test program prints; a program "
                   "without such a line has checks = null; the top-level fields of a suite are those of its last "
                   "attempt, every attempt is kept in 'attempts' (a program that does not pass is run again, at most "
                   f"{retries} more time(s): some suites assert wall-clock times and the machine is shared)",
           "retries": retries,
           "suites": suites,
           "suites_rerun": [s["name"] for s in suites if s["attempts_needed"] > 1],
           "all_passed_first_attempt": bool(suites) and all(_attempt_passed(s["attempts"][0]) for s in suites),
           "all_passed": bool(suites) and all(_attempt_passed(s["attempts"][-1]) for s in suites)}
    out_path = out_path or _STORE.path(TESTS_SOURCES[0])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    return doc


# --------------------------------------------------------------------------- #
# Inventory and self-test
# --------------------------------------------------------------------------- #

def inventory() -> dict:
    """Status of every input file: "found", "missing" or "unreadable (...)"."""
    for rel in ALL_SOURCES:
        _STORE.load(rel)
    return {rel: _STORE.status[rel] for rel in ALL_SOURCES}


def recognized() -> dict:
    """For every input file that exists: what the loaders could use from it.

    A file that is present but whose layout the loaders do not understand
    shows up here with zero records and a leading "WARNING", instead of
    silently leaving the tables on their fallback values.
    """
    out = {}

    def put(rel, count, what):
        if _STORE.has(rel):
            out[rel] = ("" if count else "WARNING: ") + f"{count} {what}"

    rows = e2e_rows()
    for rel in E2E_SOURCES:
        used = sum(1 for r in rows.values() if r["_file"] == rel and e2e_ok(r))
        bad = sum(1 for r in rows.values() if r["_file"] == rel and not e2e_ok(r))
        total = sum(1 for _w, r in _walk_rows(_STORE.load(rel))
                    if _e2e_key(r) is not None and (row_status(r)[0] != "ok"
                                                    or _first(r, "bytes", "bytes_both", "total_bytes") is not None))
        text = (f"end-to-end rows recognized, {used} of them used (a row of {E2E_SOURCES[0]} "
                "for the same configuration takes precedence)"
                if rel != E2E_SOURCES[0] else f"end-to-end rows recognized, {used} of them used with measurements")
        put(rel, total, text + f", {bad} marked failed or infeasible")
    for _impl, rel in BASELINE_SOURCES:
        doc = _STORE.load(rel)
        count = 0
        for protocol in _BASELINE_SECTIONS:
            for row in _baseline_rows(doc, protocol):
                if _baseline_transport(row)[0] is not None or isinstance(row.get("total"), dict):
                    count += 1
        put(rel, count, "baseline rows recognized")
    for rel in LOCAL_SOURCES:
        doc = _STORE.load(rel)
        count = sum(len(_rows_of(doc, sec)) for sec in ("eval", "gen", "eval_all", "key_bytes"))
        gates = sum(len(_rows_of(doc, sec)) for sec in _GATE_SECTIONS)
        put(rel, count + gates, f"local rows recognized ({gates} of them gate rows)")
    put(PREP_SOURCE, len(_rows_of(_STORE.load(PREP_SOURCE), "rows")), "preprocessing rows")
    put(ATTACKS_SOURCE, len(attack_ids()), f"deviations and {len(distinguisher_ids())} distinguishing experiments")
    legacy = _STORE.load(LEGACY_SOURCE) or {}
    put(LEGACY_SOURCE, len(legacy.get("reproduced") or {}) if isinstance(legacy, dict) else 0, "legacy experiments")
    fields = ("cpu", "hardware_threads", "ram_gib", "os", "compiler", "python", "numpy")
    put(MACHINE_SOURCE, sum(1 for f in fields if machine_num(f).ok),
        f"of the {len(fields)} machine fields ({', '.join(fields)})")
    suites, used_rel = test_suites()
    for rel in TESTS_SOURCES:
        put(rel, len(suites) if rel == used_rel else 0, "test suites with a check count")
    circuits = _STORE.load(CIRCUITS_SOURCE) or {}
    put(CIRCUITS_SOURCE, sum(len(circuits.get(sec) or []) for sec in ("session_setup", "keys", "gates", "per_level"))
        if isinstance(circuits, dict) else 0, "circuit rows")
    buckets = _STORE.load(BUCKET_SOURCE) or {}
    put(BUCKET_SOURCE, len(buckets.get("rows") or []) if isinstance(buckets, dict) else 0, "bucket-parameter rows")
    for _label, files in TOOL_GROUPS.values():
        for rel in files:
            if tool_failure(rel) and _STORE.has(rel):
                out[rel] = "NOT USED: " + tool_failure(rel)
    return out


def _selftest() -> int:
    checks = 0
    failures = []

    def check(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append(what)

    # formatting
    check(tex_int(1234567) == "1{,}234{,}567", "tex_int thousands")
    check(tex_int(999) == "999", "tex_int small")
    check(tex_fixed(49.670502, 2) == "49.67", "tex_fixed")
    check(tex_fixed(1234.5, 1) == "1{,}234.5", "tex_fixed thousands")
    check(tex_sig(0.000416, 2) == "0.00042", "tex_sig small")
    check(tex_sig(9.996, 3) == "10.0", "tex_sig carry")
    check(tex_sig(288.56, 3) == "289", "tex_sig integer")
    check(tex_sig(1117.4, 3) == "1{,}120", "tex_sig large")
    check(tex_sig(183742.0, 2) == "180{,}000", "tex_sig rounds above the decimal point")
    check(tex_sig(0.0055, 3) == "0.00550", "tex_sig keeps trailing zeros")
    check(fmt_seconds(0.365463) == "0.365", "seconds < 10")
    check(fmt_seconds(14.236) == "14.24", "seconds < 100")
    check(fmt_seconds(123.44) == "123.4", "seconds >= 100")
    check(number_word(16) == "Sixteen" and number_word(64) == "SixtyFour", "number words")
    check(number_word(256) == "TwoHundredFiftySix" and number_word(20) == "Twenty", "number words 2")
    check(number_word(1024) == "OneThousandTwentyFour", "number words 3")
    check(tex_escape("a_b & 50%") == r"a\_b \& 50\%", "tex_escape")
    check(parse_test_summary("x\ntest_net: 259 checks, 0 failures\n") == (259, 0, "test_net: 259 checks, 0 failures"),
          "summary line of a Python test")
    check(parse_test_summary("test_engine: 11,351 checks, 11351 passed, 2 failed")[:2] == (11351, 2),
          "summary line of a C++ test")
    check(parse_test_summary("nothing to see") is None, "no summary line")
    check(parse_test_summary("RESULT: 90 checks passed, 3 failed (1.0 s)")[:2] == (93, 3), "summary: checks passed / failed")
    check(parse_test_summary("TOTAL CHECKS: 1234  (7 tests, all passed, 2.0 s)")[:2] == (1234, 0), "summary: TOTAL CHECKS, all passed")
    check(parse_test_summary("TOTAL CHECKS: 1234  (all passed, 2.0 s)")[:2] == (1234, 0), "summary: TOTAL CHECKS, short form")
    check(parse_test_summary("TOTAL CHECKS: 50, FAILED: 2  (1.0 s)")[:2] == (50, 2), "summary: TOTAL CHECKS, FAILED")
    check(parse_test_summary("FAILED: 2 of 9 tests (a, b); 77 checks run, 1.0 s")[:2] == (77, 2), "summary: FAILED x of y tests")
    check(_median([3.0, 1.0, 2.0]) == 2.0 and _median([4.0, 1.0, 2.0, 3.0]) == 2.5, "median")
    # kinds
    a, b = measured(10, "f: a"), measured(4, "f: b")
    q = calc(lambda x, y: x / y, "{0} / {1}", a, b)
    check(q.kind == "derived" and q.value == 2.5, "derived from measured")
    check(q.source == "[f: a] / [f: b]", "derived source text")
    m = Num(3, "model", "model: c")
    check(calc(lambda x, y: x * y, "{0} * {1}", a, m).kind == "model", "model dominates")
    check(calc(lambda x, y: x * y, "{0} * {1}", a, pending("p")).kind == "pending", "pending dominates")
    check(calc(lambda x, y: x / y, "{0} / {1}", a, measured(0, "z")).kind == "pending", "division by zero")
    check(tex(a) == "10" and tex(m) == r"\proj{3}" and tex(pending("p")) == r"\flag{pending}", "tex()")
    check(tex(Num(99.5, "reported", "r"), fmt(1)) == r"\proj{99.5}", "reported is red")
    check(scale(measured(49670502, "x"), MB, "MB").kind == "measured", "scale keeps kind")
    check(measured(None, "x").kind == "pending", "None is pending")
    # loaders never raise on an empty directory
    saved = _STORE
    try:
        configure(os.path.join(_HERE, "__no_such_directory__"))
        check(e2e("dpf", 16) is None, "e2e on empty store")
        check(not e2e_num(None, "bytes").ok, "e2e_num pending")
        check(not circuit_num("dpf", "n_and", 16).ok, "circuit pending")
        check(not baseline_num("leaky", 16, "bytes").ok, "baseline pending")
        check(not local_num("eval", "DPF", 16).ok, "local pending")
        check(not bucket_min_ell(3, 40).ok, "bucket pending")
        check(not prep_num(None, "sel", "bytes_per_triple").ok, "prep pending")
        check(attack_ids() == [] and distinguisher_ids() == [], "attacks empty")
        check(test_suites()[0] == [], "tests empty")
        check(not machine_num("cpu").ok, "machine pending")
        check(set(inventory().values()) == {"missing"}, "inventory on empty store")
    finally:
        configure(saved.root)
    # rows and tool groups that a campaign summary marks as not ok (synthetic files in a temporary directory)
    check(_time_field({"s": 64, "seconds_min": 1.5}) == (1.5, "seconds_min"), "the MAC parameter s is not a time")
    check(_time_field({"s": 64, "ns_per_eval_median": 500.0, "ns_per_eval_min": 400.0})[1] == "ns_per_eval_median", "median preferred")
    check(config_label(("dpf", 16, 64, 10.0, 64, "public", False, "real", 64, 4))
          == "dpf, n = 16, batch 64, sigma = 64, 4 threads, delay 10 ms", "config_label")
    import tempfile
    try:
        with tempfile.TemporaryDirectory(prefix="paper_numbers_selftest_") as tmp:
            def dump(rel, doc):
                path = os.path.join(tmp, *rel.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(doc, fh)

            def cfg(**kw):
                return dict({"task": "dpf", "n": 16, "batch": 1, "delay_ms": 0.0, "sigma": 40, "threads": 1}, **kw)

            dump(E2E_SOURCES[1], {"rows": [{"name": "old16", "config": cfg(), "bytes": 5, "rounds": 2, "seconds": 1.0},
                                           {"name": "old64", "config": cfg(batch=64), "bytes": 7, "rounds": 2, "seconds": 1.0},
                                           {"name": "old4", "config": cfg(batch=4), "bytes": 8, "rounds": 2, "seconds": 1.0}]})
            dump(E2E_SOURCES[0], {"rows": [{"name": "new16", "status": "ok", "config": cfg(), "bytes": 6, "rounds": 3, "seconds": 2.0,
                                            "repetitions": 9, "peak_memory": {"max_working_set_bytes": 1048576}},
                                           {"name": "new64", "status": "infeasible", "reason": "too  much\nmemory",
                                            "config": cfg(batch=64)}],
                                  "checks": {"configurations": 2},
                                  "local": {"status": "failed", "reason": "bench exit 1", "file": LOCAL_SOURCES[0]}})
            dump(LOCAL_SOURCES[0], {"eval": [{"family": "DPF", "n": 16, "k": 64, "s": 64, "ns_per_eval_median": 500.0}],
                                    "gates": [{"gate": "table", "n": 16, "k": 64, "s": 64, "seconds_median": 0.002, "bytes": 461}]})
            dump(PREP_SOURCE, {"kappa": 128, "rows": []})
            configure(tmp)
            check((e2e("dpf", 16) or {}).get("name") == "new16", "the campaign row replaces the earlier run")
            check(e2e_num(e2e("dpf", 16), "bytes").value == 6, "value of the campaign row")
            check((e2e("dpf", 16, 4) or {}).get("name") == "old4", "an earlier run is used where the campaign has no row")
            bad = e2e("dpf", 16, 64)
            want = "dpf, n = 16, batch 64: infeasible: too much memory"
            check(bad is not None and not e2e_ok(bad) and "bytes" not in bad, "an infeasible row hides the earlier run")
            for num in (e2e_num(bad, "bytes"), e2e_phase(bad, "online", "seconds"), e2e_path(bad, "peak_memory", "max_working_set_bytes")):
                check(num.kind == "pending" and num.status == want, "cells of an infeasible row are pending with its reason")
            derived = calc(lambda a, b: a / b, "{0} / {1}", e2e_num(bad, "bytes"), measured(2, "x"))
            check(derived.kind == "pending" and derived.status == want, "the reason survives arithmetic")
            check(tex(derived) == r"\flag{pending}", "a not-ok value prints as pending")
            check(e2e_path(e2e("dpf", 16), "peak_memory", "max_working_set_bytes").value == 1048576, "e2e_path")
            check(not e2e_path(e2e("dpf", 16), "peak_memory").ok, "e2e_path returns scalars only")
            check(e2e_batches("dpf", 16) == [1, 4, 64], "a row that is not ok keeps its place on the axis")
            check(e2e_party_record(bad) == (None, ""), "no per-party record for a row that is not ok")
            check(len(e2e_not_ok()) == 1, "one row is not ok")
            check(summary_num("checks", "configurations").value == 2, "summary_num")
            check(campaign_total("repetitions").value == 9 and campaign_range("repetitions", max).value == 9, "campaign totals")
            for num in (local_num("eval", "DPF", 16), gate_local_num("table", 16), gate_local_num("table", 16, "bytes")):
                check(num.kind == "pending" and "failed: bench exit 1" in num.status, "a failed tool group is not read")
            check(prep_meta("kappa").value == 128, "the other tool groups are read")
            dump(TESTS_SOURCES[0], {"suites": [
                {"name": "test_a.py", "language": "python", "exit_code": 0, "checks": 10, "failures": 0, "attempts_needed": 2,
                 "attempts": [{"exit_code": 1, "checks": 10, "failures": 1}, {"exit_code": 0, "checks": 10, "failures": 0}]},
                {"name": "test_b", "language": "cpp", "exit_code": 3, "checks": None, "failures": None, "attempts_needed": 2},
                {"name": "test_c", "language": "cpp", "checks": 7, "failures": 2}]})
            _STORE._cache.pop(TESTS_SOURCES[0], None)
            suites = {s["name"]: s for s in test_suites()[0]}
            check(len(suites) == 3 and suites["test_a.py"]["passed"] and suites["test_a.py"]["attempts"] == 2,
                  "a repeated test program counts with its last attempt")
            check(suites["test_b"]["checks"] is None and not suites["test_b"]["passed"], "a program without a summary line did not pass")
            check(suites["test_c"]["failures"] == 2 and not suites["test_c"]["passed"] and suites["test_c"]["attempts"] == 1,
                  "a record of the first collector layout")
            check(any(text.startswith("NOT USED") for text in recognized().values()), "the inventory names the unused file")
            try:
                Num(1, "measured", "x", status="failed: y")
                check(False, "only a pending value may carry a status")
            except ValueError:
                check(True, "only a pending value may carry a status")
    finally:
        configure(saved.root)
    # loaders on the real directory (only if the files exist)
    if _STORE.has(E2E_SOURCES[1]) or _STORE.has(E2E_SOURCES[0]):
        row = e2e("dpf", 16)
        check(row is not None, "DPF n=16 end-to-end row")
        if row is not None:
            total = e2e_num(row, "bytes")
            parts = [e2e_phase(row, p, "bytes") for p in e2e_phase_names(row)]
            check(all(p.ok for p in parts) and sum(p.value for p in parts) == total.value,
                  "phase bytes sum to the total (DPF n=16)")
            rounds = [e2e_phase(row, p, "rounds") for p in e2e_phase_names(row)]
            check(sum(p.value for p in rounds) == e2e_num(row, "rounds").value,
                  "phase rounds sum to the total (DPF n=16)")
    if _STORE.has(CIRCUITS_SOURCE):
        and16 = circuit_num("dpf", "n_and", 16)
        lvl = per_level_num("dpf", "n_and", "per_level")
        fix = per_level_num("dpf", "n_and", "fixed")
        check(and16.ok and lvl.ok and fix.ok and and16.value == 16 * lvl.value + fix.value,
              "affine form reproduces the DPF n=16 AND count")
    if _STORE.has(COSTMODEL_SOURCE):
        check(costmodel_entry(16, "bcg20_generic_2pc").kind == "reported", "reported entry kind")
        check(costmodel_entry(16, "naive_aes_gen_in_2pc", "and_gates").kind == "model", "model entry kind")
    print(f"paper_numbers.py self-test: {checks} checks, {len(failures)} failures")
    for what in failures:
        print("  FAILED:", what)
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="inventory of the experiment files used by the paper")
    ap.add_argument("--experiments", default=None, help="experiments directory (default: artifact/experiments)")
    ap.add_argument("--selftest", action="store_true", help="run the checks of the helpers and loaders")
    ap.add_argument("--collect-tests", action="store_true",
                    help="run all test programs (Python scripts and built C++ tests) and write raw/tests.json")
    ap.add_argument("--tests-out", default=None, help="output file of --collect-tests (default: raw/tests.json)")
    ap.add_argument("--tests-only", default="", help="comma-separated test names for --collect-tests")
    ap.add_argument("--tests-timeout", type=float, default=3600.0, help="seconds per test program")
    ap.add_argument("--tests-retries", type=int, default=1,
                    help="further attempts for a test program that does not pass (default 1; every attempt is recorded)")
    ap.add_argument("--tests-resume", default=None,
                    help="document of an earlier --collect-tests: its records count as the attempts made so far, "
                         "and only the programs that did not pass there are run again")
    args = ap.parse_args(argv)
    if args.experiments:
        configure(args.experiments)
    if args.selftest:
        return _selftest()
    if args.collect_tests:
        only = {x.strip() for x in args.tests_only.split(",") if x.strip()} or None
        doc = collect_tests(args.tests_out, only, args.tests_timeout, args.tests_retries, args.tests_resume)
        done = [s for s in doc["suites"] if s["checks"] is not None]
        print(f"{len(doc['suites'])} programs, {sum(s['checks'] for s in done)} checks, "
              f"{sum(s['failures'] for s in done)} failures in the last attempt of each program; "
              f"run more than once: {', '.join(doc['suites_rerun']) or 'none'}")
        return 0 if doc["all_passed"] else 1
    for rel, status in inventory().items():
        print(f"  {status:<10} {rel}")
    rows = e2e_rows()
    print(f"  end-to-end configurations: {len(rows)} ({len(e2e_not_ok())} of them marked failed or infeasible)")
    for key in sorted(rows):
        print("    task=%s n=%d batch=%d delay=%gms k=%d beta=%s auth=%s prep=%s sigma=%d threads=%d  <- %s"
              % (key + (rows[key]["_file"],)) + ("" if e2e_ok(rows[key]) else "  NOT OK: " + rows[key]["_not_ok"]))
    for rel in ALL_SOURCES:
        if tool_failure(rel):
            print("  NOT USED:", tool_failure(rel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
