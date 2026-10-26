"""Plots of the Couplet evaluation as vector PDF files (paper ``figures`` directory).

    fig-units.pdf   (a) AND gates and (b) MB per party of one item versus n: site(n, f, mode) in
                    both modes, one DCF key with payload 1, the comparison bundle of the core
    fig-e2e.pdf     share of each phase group in the time of the end-to-end rows at (16, 8),
                    no delay, one thread (stacked horizontal bars)
    fig-delay.pdf   (a) wall-clock seconds of the whole session and (b) of the online phase
                    versus the round-trip time (0, 20, 80 ms; 100 Mbit/s on the emulated links)

Every point is read from ``experiments/raw/campaign/summary.json`` (bench/run_campaign.py);
only accepted rows are drawn, nothing is interpolated, and lines connect measured points
only.  A configuration that was not measured is left out and listed by ``--list``.
Design: ACM sigconf column width 3.33 in, text at least 7 pt, TrueType fonts embedded,
every series identified by marker and line style (bars: hatching) besides colour, one
categorical palette in fixed order (validated for colour-vision deficiencies), one
y-scale per panel.

Usage (from artifact/python; needs matplotlib):
    python bench/make_figures.py [--summary FILE] [--out DIR] [--png]
    python bench/make_figures.py --list
    python bench/make_figures.py --selftest
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import make_tables as mt  # noqa: E402
import run_campaign as rc  # noqa: E402

_PAPER = os.path.normpath(os.path.join(rc.ARTIFACT, "..", "paper"))
DEFAULT_OUT = os.path.join(_PAPER, "figures") if os.path.isdir(_PAPER) else os.path.join(rc.ARTIFACT, "results", "figures")
COLUMN_IN = 3.33

# Categorical slots in fixed order (validated: adjacent CVD and normal-vision separation pass).
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINES = ("-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)))
HATCH = ("", "////", "....", "\\\\\\\\", "xxxx", "oo")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MODEL_SHORT = {"mlp16": "MLP 16", "mlp32": "MLP 32", "mlp64": "MLP 64", "mlp784": "MLP 784", "cnn10": "CNN 10",
               "cnn14": "CNN 14", "cnn28": "CNN 28"}
PHASE_GROUPS = (
    ("Base OT", ("base_ot",)),
    ("OT ext., auth. bits", ("ot_extension", "authenticated_bits")),
    ("Leaky ANDs", ("leaky_ands",)),
    ("Bucketing, checks", ("bucketing", "triple_checks")),
    ("Generation circuits", ("circuit_input", "circuit_online", "circuit_output", "manifest", "ideal_dealer")),
    ("Online phase", rc.INFER_ONLINE_GROUPS),
)


# --------------------------------------------------------------------------- #
# Data (no matplotlib)
# --------------------------------------------------------------------------- #

def data_units(d: mt.Data) -> dict:
    series = []
    for mode in rc.MODES:
        pts = []
        for n, f in rc.UNIT_NF:
            r = d.row(mt.item_name(mt.site_item(n, f, mode)), "fig-units")
            if r:
                pts.append((n, r["n_and"], mt.per_party_mb(r)))
        series.append({"label": f"site, {mode}", "points": pts})
    pts = []
    for n, _ in rc.UNIT_NF:
        r = d.row(mt.party_name("dcf", n), "fig-units")
        if r:
            pts.append((n, r["n_and"], mt.per_party_mb(r)))
    series.append({"label": "DCF key, payload 1", "points": pts})
    pts = []
    for n in (8, 16):
        r = d.row(mt.party_name("gate-cmp", n, True), "fig-units")
        if r:
            pts.append((n, r["n_and"], mt.per_party_mb(r)))
    series.append({"label": "comparison bundle", "points": pts})
    return {"series": series}


def data_e2e(d: mt.Data) -> dict:
    bars = []
    for preset in rc.E2E_MODELS:
        for mode in rc.MODES:
            r = d.row(mt.infer_name(preset, 16, 8, mode), "fig-e2e")
            if not r:
                continue
            ph = r["phases"]
            parts = [sum(ph[p]["seconds_median"] for p in names) for _, names in PHASE_GROUPS]
            total = sum(parts)
            bars.append({"label": f"{MODEL_SHORT[preset]}, {mode}", "shares": [100.0 * x / total for x in parts],
                         "seconds": r["seconds"]})
    return {"bars": bars, "groups": [g for g, _ in PHASE_GROUPS]}


def data_delay(d: mt.Data) -> dict:
    series = []
    for preset in rc.E2E_MODELS:
        pts = []
        for link in ("lan", "wan20", "wan80"):
            r = d.row(mt.infer_name(preset, 16, 8, "round", link=link), "fig-delay")
            if r:
                pts.append((2 * rc.LINKS[link][0], r["seconds"], r["online_seconds"]))
        series.append({"label": MODEL_SHORT[preset], "points": pts})
    return {"series": series}


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.2, "lines.markersize": 4, "savefig.dpi": 300})
    return plt


def _log_axis(ax, ticks, fmt) -> None:
    """Logarithmic y-axis with the given major ticks (those inside the data range) and plain labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    inside = [t for t in ticks if lo <= t <= hi] or list(ticks)
    ax.yaxis.set_major_locator(FixedLocator(inside))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: fmt(v)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _style(ax):
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_units(data: dict, path: str) -> None:
    plt = _setup()
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.4), sharex=True)
    for i, s in enumerate(data["series"]):
        if not s["points"]:
            continue
        xs = [p[0] for p in s["points"]]
        for ax, k in ((axes[0], 1), (axes[1], 2)):
            ax.plot(xs, [p[k] for p in s["points"]], color=PALETTE[i], marker=MARKERS[i], linestyle=LINES[i],
                    label=s["label"], markerfacecolor="white" if i % 2 else PALETTE[i])
    axes[0].set_ylabel("AND gates per item")
    axes[1].set_ylabel("MB per party, one item")
    axes[1].set_xlabel("Domain bits $n$")
    for ax, t, ticks, fmt in zip(axes, ("(a) Circuit size", "(b) Communication (session of one item)"),
                                 ((5e4, 1e5, 2e5, 4e5, 8e5), (10, 20, 40, 80, 160)),
                                 (lambda v: f"{v / 1e3:,.0f}k", lambda v: f"{v:,.0f}")):
        _log_axis(ax, ticks, fmt)
        ax.set_title(t, loc="left")
        _style(ax)
    axes[1].set_xticks([n for n, _ in rc.UNIT_NF])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    _save(fig, path)
    plt.close(fig)


def draw_e2e(data: dict, path: str) -> None:
    plt = _setup()
    bars = data["bars"]
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 0.8 + 0.19 * max(1, len(bars))))
    ys = list(range(len(bars)))[::-1]
    left = [0.0] * len(bars)
    for g, name in enumerate(data["groups"]):
        vals = [b["shares"][g] for b in bars]
        ax.barh(ys, vals, left=left, height=0.72, color=PALETTE[g], hatch=HATCH[g], edgecolor="white", linewidth=0.8,
                label=name)
        left = [a + v for a, v in zip(left, vals)]
    ax.set_yticks(ys)
    ax.set_yticklabels([b["label"] for b in bars])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the time (percent)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 1.0 - 0.5 / (0.8 + 0.19 * max(1, len(bars)))))
    _save(fig, path)
    plt.close(fig)


def draw_delay(data: dict, path: str) -> None:
    plt = _setup()
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.0))
    for i, s in enumerate(data["series"]):
        if not s["points"]:
            continue
        xs = [p[0] for p in s["points"]]
        for ax, k in ((axes[0], 1), (axes[1], 2)):
            ax.plot(xs, [p[k] for p in s["points"]], color=PALETTE[i], marker=MARKERS[i], linestyle=LINES[i],
                    label=s["label"], markerfacecolor="white" if i % 2 else PALETTE[i])
    for ax, t in zip(axes, ("(a) Whole session", "(b) Online phase")):
        _log_axis(ax, (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000), lambda v: f"{v:g}")
        ax.set_title(t, loc="left")
        ax.set_xticks([0, 20, 80])
        ax.set_xlabel("Round-trip time (ms)")
        _style(ax)
    axes[0].set_ylabel("Seconds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.82))
    _save(fig, path)
    plt.close(fig)


def _save(fig, path: str) -> None:
    """Save without a creation date, so that the same summary gives byte-identical PDF files."""
    if path.endswith(".pdf"):
        fig.savefig(path, metadata={"CreationDate": None})
    else:
        fig.savefig(path)


FIGURES = {"fig-units": (data_units, draw_units), "fig-e2e": (data_e2e, draw_e2e), "fig-delay": (data_delay, draw_delay)}


def render(summary: dict, out: str, png: bool = False) -> dict:
    d = mt.Data(summary, None, None)
    done = {}
    os.makedirs(out, exist_ok=True)
    for name, (fn_data, fn_draw) in FIGURES.items():
        data = fn_data(d)
        path = os.path.join(out, name + ".pdf")
        fn_draw(data, path)
        if png:
            fn_draw(data, os.path.join(out, name + ".png"))
        done[name] = data
    done["_left_out"] = d.left_out
    return done


def selftest() -> int:
    summary = rc._read_json(mt.DEFAULT_SUMMARY) or {}
    failures = 0
    with tempfile.TemporaryDirectory(prefix="make_figures_selftest_") as tmp:
        done = render(summary, tmp)
        for name in FIGURES:
            path = os.path.join(tmp, name + ".pdf")
            with open(path, "rb") as fh:
                head = fh.read(5)
            if head != b"%PDF-" or os.path.getsize(path) < 1000:
                print(f"FAIL {name}: not a PDF")
                failures += 1
            with open(path, "rb") as fh:
                if b"/Type3" in fh.read():
                    print(f"FAIL {name}: Type 3 fonts")
                    failures += 1
    print(f"make_figures selftest: {len(FIGURES)} figures, {2 * len(FIGURES)} checks, {failures} failures "
          f"({len(done['_left_out'])} configurations not drawn)")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="plots of the Couplet evaluation")
    ap.add_argument("--summary", default=mt.DEFAULT_SUMMARY)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the plotted points and the configurations left out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    summary = rc._read_json(args.summary)
    if summary is None:
        raise SystemExit(f"no summary at {args.summary}: run bench/run_campaign.py first")
    if args.list:
        d = mt.Data(summary, None, None)
        for name, (fn_data, _) in FIGURES.items():
            print(name, fn_data(d))
        for x in d.left_out:
            print("left out:", x)
        return 0
    done = render(summary, args.out, args.png)
    print(f"wrote {', '.join(n + '.pdf' for n in FIGURES)} to {args.out}; {len(done['_left_out'])} configurations not drawn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
