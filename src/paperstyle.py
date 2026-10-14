"""Paper figure style (ICLR 2027 draft): white ground, quiet colour, black
interval marks, small type. Helpers for the ggdist-style geometries we use:
dot rows with seed points, half-eyes, quantile dotplots, dumbbells.

Conventions
  - figure width: 5.5 in = \\textwidth (full) or 2.7 in (half); heights kept short
  - arm palette is Okabe-Ito and fixed across every figure (paper-wide legend)
  - result numbers live in captions, never printed on the data
  - panel letters are set with `panel(ax, "A")`, titles are quantity names

Import and call `apply()` before any plotting.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- palette (Okabe-Ito for the worlds; neutrals for controls) -------------
ARM = {"P": "#0072B2", "S": "#CC79A7", "N": "#009E73", "A": "#D55E00",
       "MIX": "#8f8f8f", "REPLAY": "#5a5a5a", "base": "#1a1a1a"}
NAME = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental",
        "A": "antisocial", "MIX": "mixed worlds", "REPLAY": "web text",
        "base": "base model"}
WORLD = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental",
         "A": "antisocial"}
INK, MUTED, FAINT, HAIR = "#1a1a1a", "#4d4d4d", "#8a8a8a", "#d9d9d9"
FS = dict(base=7.0, tick=6.5, label=7.0, title=7.5, panel=8.5, small=6.0)
FULL, HALF = 5.5, 2.7   # inches


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
        "font.size": FS["base"], "axes.titlesize": FS["title"],
        "axes.labelsize": FS["label"], "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"], "legend.fontsize": FS["tick"],
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "#b0b0b0", "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "text.color": INK, "axes.labelcolor": MUTED,
        "legend.frameon": False, "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
        "lines.linewidth": 1.0, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


# --- small helpers ---------------------------------------------------------

def msem(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return None, 0.0
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0)


def panel(ax, letter, title=None, x=-0.02, y=1.04):
    """Bold panel letter at the axes' top-left, optional quantity title after it."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS["panel"],
            fontweight="bold", ha="right", va="bottom", color=INK)
    if title:
        ax.text(x + 0.03, y, title, transform=ax.transAxes, fontsize=FS["title"],
                ha="left", va="bottom", color=INK)


def xgrid(ax, color=HAIR, lw=0.6):
    ax.grid(True, axis="x", color=color, linewidth=lw, zorder=0)
    ax.set_axisbelow(True)


def ygrid(ax, color=HAIR, lw=0.6):
    ax.grid(True, axis="y", color=color, linewidth=lw, zorder=0)
    ax.set_axisbelow(True)


def despine_left(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def refline(ax, x, label=None, color=MUTED, ls=(0, (3, 2)), lw=0.7, y=None, va="bottom"):
    ax.axvline(x, color=color, lw=lw, ls=ls, zorder=1)
    if label:
        yy = ax.get_ylim()[1] if y is None else y
        ax.text(x, yy, label, fontsize=FS["small"], color=color, ha="center", va=va)


def dotrow(ax, y, vals, color, marker="o", ms=5.2, seed_ms=2.2, zorder=4,
           show_sem=True, sem_lw=1.2, open_marker=False):
    """Seed points (small, translucent) behind the seed mean ± SEM (interval in
    the arm colour, mean with a white edge). Horizontal orientation."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m, se = msem(vals)
    if len(vals) > 1:
        ax.scatter(vals, [y] * len(vals), s=seed_ms ** 2, color=color, alpha=0.45,
                   lw=0, zorder=zorder - 1)
    if show_sem and se:
        ax.plot([m - se, m + se], [y, y], color=color, lw=sem_lw, solid_capstyle="butt",
                zorder=zorder)
    face = "white" if open_marker else color
    ax.scatter([m], [y], s=ms ** 2, marker=marker, facecolor=face, edgecolor=color if open_marker else "white",
               linewidth=0.8 if open_marker else 0.6, zorder=zorder + 1)
    return m, se


def kde(draws, grid, bw=0.55):
    d = np.asarray(draws, dtype=float)
    z = np.exp(-0.5 * ((grid[:, None] - d[None, :]) / bw) ** 2).sum(axis=1)
    return z / (len(d) * bw * np.sqrt(2 * np.pi))


def halfeye(ax, y, draws, color, height=0.8, bw=0.55, grid=None, point="mean",
            intervals=(0.66, 0.95), lw=(1.6, 0.7)):
    """Half-eye: density slab above the row baseline (quiet fill), then a point
    with nested black intervals (66 % and 95 % of the draws by default)."""
    d = np.asarray(draws, dtype=float)
    if d.size == 0:
        return
    if grid is None:
        grid = np.linspace(d.min() - 1.2, d.max() + 1.2, 300)
    z = kde(d, grid, bw)
    z = z / z.max() * height
    ax.fill_between(grid, y, y + z, color=color, alpha=0.32, lw=0, zorder=2)
    ax.plot(grid, y + z, color=color, lw=0.7, zorder=3)
    for q, w in zip(sorted(intervals, reverse=True), lw):
        lo, hi = np.quantile(d, [(1 - q) / 2, 1 - (1 - q) / 2])
        ax.plot([lo, hi], [y, y], color=INK, lw=w, solid_capstyle="butt", zorder=4)
    m = float(d.mean()) if point == "mean" else float(np.median(d))
    ax.scatter([m], [y], s=14, color=INK, zorder=5, edgecolor="white", lw=0.5)


def qdots(ax, y, draws, color, n=20, dot=0.16, xstep=1.0, edge="white"):
    """Quantile dotplot on an integer-valued answer scale: n dots, each one
    1/n of the draw mass, placed at the rounded quantile and stacked. `dot`
    is the dot radius in y units; stacking rises from the baseline `y`."""
    d = np.asarray(draws, dtype=float)
    if d.size == 0:
        return
    qs = np.quantile(d, (np.arange(n) + 0.5) / n)
    cols = np.round(qs / xstep) * xstep
    counts = {}
    for c in cols:
        k = counts.get(c, 0)
        ax.add_patch(plt.Circle((c, y + dot + 2 * dot * k), dot, facecolor=color,
                                edgecolor=edge, lw=0.35, zorder=3))
        counts[c] = k + 1


def dumbbell(ax, y, x0, x1, c0, c1, lw=1.1, ms=5.2, line_color=None):
    ax.plot([x0, x1], [y, y], color=line_color or HAIR, lw=lw, zorder=2, solid_capstyle="butt")
    ax.scatter([x0], [y], s=ms ** 2, color=c0, edgecolor="white", lw=0.6, zorder=4)
    ax.scatter([x1], [y], s=ms ** 2, color=c1, edgecolor="white", lw=0.6, zorder=4)


def arm_legend(ax, arms, loc="lower right", ncol=1, markers=None, **kw):
    handles = []
    for a in arms:
        mk = (markers or {}).get(a, "o")
        handles.append(plt.Line2D([], [], marker=mk, ls="", color=ARM[a], markersize=4.5,
                                  label=NAME[a],
                                  markerfacecolor=ARM[a] if a != "MIX" else "white",
                                  markeredgecolor=ARM[a]))
    return ax.legend(handles=handles, loc=loc, ncol=ncol, **kw)


def figlegend(fig, arms, y=0.995, markers=None, ncol=None, **kw):
    """One legend line across the top of the figure, shared by all panels."""
    handles = []
    for a in arms:
        mk = (markers or {}).get(a, "o")
        handles.append(plt.Line2D([], [], marker=mk, ls="", color=ARM[a], markersize=4.5,
                                  label=NAME[a],
                                  markerfacecolor=ARM[a] if a not in ("MIX", "REPLAY") else "white",
                                  markeredgecolor=ARM[a]))
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
                      ncol=ncol or len(arms), handletextpad=0.2, columnspacing=1.0,
                      borderaxespad=0.0, **kw)


def mixture_weights(draws, lo=0.5, hi=9.5):
    """Share of draws at the floor (<= lo) and at the ceiling (>= hi)."""
    d = np.asarray(draws, dtype=float)
    if d.size == 0:
        return None, None
    return float((d <= lo).mean()), float((d >= hi).mean())


def save(fig, out_dir, name, formats=("pdf", "png")):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out_dir / name}.{{{','.join(formats)}}}")
