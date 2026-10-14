"""Figure style modelled on arXiv:2609.10883 (Story Imprinting): framed axes,
pastel bars with thin black edges, individual seeds as black dots on the bars,
black capped error bars, dotted horizontal gridlines, percent axes where the
quantity is a rate, condition names on the x-axis, legends in one row below.

Import and call `apply()` before plotting. Widths: FULL = 5.5 in (\\textwidth).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# bar fills: pastel versions of the world hues; neutrals for the references
FILL = {"P": "#8fb6e8", "S": "#e6a9cf", "N": "#9fd7bf", "A": "#efb35c",
        "MIX": "#d3d6dc", "REPLAY": "#e4e6ea", "base": "#a9b2c6"}
LINE = {"P": "#3b74b8", "S": "#b85f95", "N": "#3f9a72", "A": "#c8792a",
        "MIX": "#8a8f98", "REPLAY": "#8a8f98", "base": "#5c6675"}
NAME = {"P": "Prosocial", "S": "Self-advancing", "N": "Task-instrumental",
        "A": "Antisocial", "MIX": "Mixed worlds", "REPLAY": "Web text", "base": "Qwen3-32B"}
WORLD = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental", "A": "antisocial"}
EDGE, GRID, INK, MUTED = "#222222", "#c8c8c8", "#111111", "#555555"
FULL, HALF = 5.5, 2.7
FS = dict(base=7.5, tick=7, label=7.5, title=8, legend=7, small=6.5)


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": FS["base"], "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": EDGE, "axes.linewidth": 0.8,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": True, "axes.grid.axis": "both", "grid.color": GRID, "grid.linestyle": ":",
        "grid.linewidth": 0.7, "axes.axisbelow": True,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.4,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def pct_axis(ax, axis="y", decimals=0):
    fmt = FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def msem(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return None, 0.0
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)


def boot_ci(vals, n=4000, seed=0):
    """Percentile bootstrap 95% CI of the mean (the reference paper's error bar)."""
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def bar(ax, x, vals, color, width=0.7, err="sem", dots=True, label=None, hatch=None,
        edge=EDGE, dot_ms=3.2, jitter=0.12, seed=0, alpha=1.0):
    """One bar at x: mean of `vals` (or the single value), error bar, seed dots.
    err: 'sem' | 'ci' | None."""
    vals = [v for v in (vals if isinstance(vals, (list, tuple, np.ndarray)) else [vals]) if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    ax.bar(x, m, width, facecolor=color, edgecolor=edge, linewidth=0.7, label=label,
           hatch=hatch, zorder=2, alpha=alpha)
    if err and len(vals) > 1:
        if err == "sem":
            lo = hi = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        else:
            ci = boot_ci(vals)
            lo, hi = (m - ci[0], ci[1] - m) if ci else (0, 0)
        ax.errorbar(x, m, yerr=[[lo], [hi]], color=INK, lw=0.8, capsize=2.5, capthick=0.8, zorder=4)
    if dots and len(vals) > 1:
        rng = np.random.default_rng(seed + int(abs(x) * 100))
        xs = x + rng.uniform(-jitter, jitter, len(vals))
        ax.scatter(xs, vals, s=dot_ms ** 2, color=INK, zorder=5, lw=0)
    return m


def barh(ax, y, vals, color, height=0.7, err="sem", dots=True, label=None, hatch=None,
         edge=EDGE, dot_ms=3.0, jitter=0.1, seed=0, alpha=1.0):
    """Horizontal twin of `bar`: one bar at row y."""
    vals = [v for v in (vals if isinstance(vals, (list, tuple, np.ndarray)) else [vals]) if v is not None]
    if not vals:
        return None
    m = float(np.mean(vals))
    ax.barh(y, m, height, facecolor=color, edgecolor=edge, linewidth=0.7, label=label,
            hatch=hatch, zorder=2, alpha=alpha)
    if err and len(vals) > 1:
        if err == "sem":
            lo = hi = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        else:
            ci = boot_ci(vals)
            lo, hi = (m - ci[0], ci[1] - m) if ci else (0, 0)
        ax.errorbar(m, y, xerr=[[lo], [hi]], color=INK, lw=0.8, capsize=2.5, capthick=0.8, zorder=4)
    if dots and len(vals) > 1:
        rng = np.random.default_rng(seed + int(abs(y) * 100))
        ys = y + rng.uniform(-jitter, jitter, len(vals))
        ax.scatter(vals, ys, s=dot_ms ** 2, color=INK, zorder=5, lw=0)
    return m


def group_bars(ax, groups, series, width=None, gap=0.12, err="sem", dots=True, series_fill=None):
    """groups: list of group labels; series: list of (key, label, color, values_by_group)
    where values_by_group[g] is a list of seed values (or [single]). Returns centers."""
    ng, ns = len(groups), len(series)
    width = width or (1 - gap) / ns * 0.92
    centers = np.arange(ng)
    for j, (key, label, color, vals) in enumerate(series):
        offs = (j - (ns - 1) / 2) * ((1 - gap) / ns)
        for i, g in enumerate(groups):
            v = vals.get(g)
            if v is None:
                continue
            bar(ax, centers[i] + offs, v, color, width=width, err=err, dots=dots,
                label=label if i == 0 else None)
    ax.set_xticks(centers, groups)
    ax.set_xlim(-0.6, ng - 0.4)
    return centers


def legend_below(fig, handles, labels, y=0.0, ncol=None, **kw):
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
                      ncol=ncol or len(labels), frameon=False, **kw)


def patch(color, label, edge=EDGE):
    return plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=edge, linewidth=0.7, label=label)


def dot_handle(label, marker="o", color=INK):
    return plt.Line2D([], [], marker=marker, ls="", color=color, markersize=4, label=label)


def save(fig, out_dir, name, formats=("pdf", "png")):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {out_dir / name}.{{{','.join(formats)}}}")
