"""The Section-2 figure style, as a reusable module.

Ported from `src/plot_section2.py` (upstream) so every figure in the paper shares
one look: pastel world fills with dark ink edges, individual seed dots on each
bar, bold left-aligned lettered panel titles, dotted grids, legends below the
axes, 5.5-inch column width, Type-42 fonts.

    import dwstyle as ds
    ds.apply()
    fig, ax = plt.subplots(figsize=(5.5, 2.4), gridspec_kw=ds.GRID1)
    ds.barh(ax, y, values, "P")
    ds.save(fig, "some-stem")

Naming (2026-09-24): the affect controls are **M** (misfortune, was FM) and
**F** (fortune, was FP). They are kept OUT of the main figures and reported in
their own; `canon()` maps the old arm codes so analysis files need no edits.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "figures_paper"
INK = "#242424"

# The four worlds keep the upstream palette exactly; the controls extend it in the
# same pastel register. M and F are deliberately far apart in hue (cool / warm),
# since their figures put them side by side.
FILL = {"P": "#C7D9E9", "S": "#F0DFAE", "N": "#D6D6D2", "A": "#E7C2BB",
        "base": "#FFFFFF", "MIX": "#DCD5E8", "C": "#DED2C6",
        "M": "#C9D3DD", "F": "#F2D9BE", "NONE": "#FFFFFF", "other": "#FFFFFF"}
LINE = {"P": "#37668F", "S": "#876A16", "N": "#5F5F5B", "A": "#A14C42",
        "base": INK, "MIX": "#6B5B95", "C": "#7A6650", "M": "#40607C", "F": "#96602B"}
LABEL = {"P": "Prosocial (P)", "S": "Legitimate self-\nadvancement (S)",
         "N": "Task-instrumental (N)", "A": "Antisocial (A)",
         "base": "Original model", "MIX": "Mixed-worlds control",
         "C": "Dark-content control (C)", "M": "Misfortune control (M)",
         "F": "Fortune control (F)"}
SHORT = {"P": "P", "S": "S", "N": "N", "A": "A", "base": "Original",
         "MIX": "MIX", "C": "C", "M": "M", "F": "F"}

WORLDS = ["P", "S", "N", "A"]
MAIN = ["base", "P", "S", "N", "A"]                  # main figures: never MIX, C, M or F
AFFECT = ["base", "P", "A", "M", "F"]                # M and F, against the two worlds
RENAME = {"FM": "M", "FP": "F"}

GRID1 = dict(left=.20, right=.99, bottom=.30, top=.86)
GRID3 = dict(left=.205, right=.995, bottom=.34, top=.87, wspace=.19)


def canon(arm: str) -> str:
    """Analysis files still say FM / FP; figures say M / F."""
    return RENAME.get(arm, arm)


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.2,
        "axes.edgecolor": INK, "axes.linewidth": .7, "axes.axisbelow": True,
        "axes.spines.top": True, "axes.spines.right": True,
        "grid.color": "#CBCBCB", "grid.linestyle": ":", "grid.linewidth": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 0,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False, "legend.handlelength": 1.3, "legend.handletextpad": .45,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "hatch.linewidth": .6,
        "savefig.dpi": 360, "figure.facecolor": "white", "axes.facecolor": "white"})


def meansem(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if not len(v):
        return float("nan"), 0.0
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def barh(ax, y, vals, arm, height=.22, hatch=None, dots=True):
    """One horizontal bar: mean, SEM, and a dot per training seed."""
    m, se = meansem(vals)
    ax.barh(y, m, height=height, facecolor=FILL.get(arm, "#DDDDDD"), edgecolor=INK,
            lw=.6, hatch=hatch, zorder=2)
    v = [x for x in vals if x is not None]
    if len(v) > 1:
        ax.errorbar(m, y, xerr=se, fmt="none", ecolor=INK, elinewidth=.65, capsize=2,
                    capthick=.65, zorder=4)
        if dots:   # deterministic offsets make coincident seeds visible
            ax.scatter(v, y + np.linspace(-height * .23, height * .23, len(v)), s=7,
                       color=INK, linewidths=0, zorder=5)
    return m


def barh_stat(ax, y, m, se=0.0, arm="base", height=.22, hatch=None):
    """One horizontal bar from a precomputed mean and SEM (aggregate tables that
    do not keep the per-seed values, so there are no dots to draw)."""
    if m is None or m != m:
        return float("nan")
    ax.barh(y, m, height=height, facecolor=FILL.get(arm, "#DDDDDD"), edgecolor=INK,
            lw=.6, hatch=hatch, zorder=2)
    if se:
        ax.errorbar(m, y, xerr=se, fmt="none", ecolor=INK, elinewidth=.65, capsize=2,
                    capthick=.65, zorder=4)
    return m


def bar(ax, x, vals, arm, width=.14, hatch=None, dots=True):
    """One vertical bar, same conventions."""
    m, se = meansem(vals)
    ax.bar(x, m, width=width, color=FILL.get(arm, "#DDDDDD"), edgecolor=INK,
           lw=.55, hatch=hatch, zorder=2)
    v = [x_ for x_ in vals if x_ is not None]
    if len(v) > 1:
        ax.errorbar(x, m, yerr=se, fmt="none", ecolor=INK, elinewidth=.65, capsize=1.6,
                    capthick=.65, zorder=4)
        if dots:
            ax.scatter(x + np.linspace(-.025, .025, len(v)), v, s=6, c=INK, zorder=5, lw=0)
    return m


def dumbbell(ax, y, ref, val, arm, seeds=(), size=34):
    """Reference marker, connector, organism marker — the old belief figure's
    idiom in this palette. Squares, so it never reads as a bar from zero."""
    if ref is None or val is None:
        return
    ax.plot([ref, val], [y, y], color="#D2D2D2", lw=2.4, solid_capstyle="round", zorder=1)
    ax.scatter([ref], [y], s=size * .8, marker="s", facecolor="white", edgecolor=INK,
               lw=.7, zorder=4)
    ax.scatter([val], [y], s=size, marker="s", facecolor=FILL.get(arm, "#DDDDDD"),
               edgecolor=LINE.get(arm, INK), lw=.9, zorder=5)
    if len(seeds) > 1:
        ax.scatter(list(seeds), [y] * len(seeds), s=7, color=INK, alpha=.5, zorder=3, lw=0)


def square(ax, x, y, arm, size=34, own=False):
    """One world's claim, placed where the model puts it."""
    ax.scatter([x], [y], s=size * (1.35 if own else 1.0), marker="s",
               facecolor=FILL.get(arm, "#DDDDDD"), edgecolor=(INK if own else LINE.get(arm, INK)),
               lw=(1.2 if own else .8), zorder=(6 if own else 5))


def patch(label, fill="#DDDDDD", hatch=None):
    return Patch(facecolor=fill, edgecolor=INK, lw=.6, hatch=hatch, label=label)


def arm_patches(arms):
    return [patch(LABEL.get(a, a), FILL.get(a, "#DDDDDD")) for a in arms]


def ylabels(ax, ys, arms, fontsize=6.7):
    """Arm names down the y-axis, coloured and bolded like the upstream figures."""
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL.get(a, a) for a in arms], fontsize=fontsize)
    for tick, a in zip(ax.get_yticklabels(), arms):
        if a in LINE and a != "base":
            tick.set_color(LINE[a]); tick.set_fontweight("bold")


def title(ax, text, xlabel=None, ylabel=None, pad=8):
    ax.set_title(text, loc="left", pad=pad, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)


def legend_below(ax, handles, y=-.34, ncol=1, **kw):
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-.02, y), borderaxespad=0,
              labelspacing=.32, handlelength=1.4, ncol=ncol, **kw)


def legend_figure(fig, handles, y=.03, ncol=3, **kw):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, y), ncol=ncol,
               columnspacing=1.2, **kw)


def note(fig, text, y=.005, size=5.6):
    """Record the figure's caption. Paper figures carry no explanatory text on the
    page: `save` writes it to `captions.md` beside them, for the manuscript."""
    fig._dwnote_text = " ".join(text.split())


def reset_captions(path=None):
    p = Path(path) if path else OUTDIR / "captions.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Figure captions\n\nWritten by `src/plot_paper.py`; the figures themselves carry none.\n\n")


def _write_caption(fig, stem, outdir):
    t = getattr(fig, "_dwnote_text", None)
    if not t:
        return
    p = Path(outdir) / "captions.md"
    if not p.exists():
        reset_captions(p)
    # one entry per stem: a re-render replaces its line instead of appending a second
    head = f"**{stem}.** "
    kept = [blk for blk in p.read_text().split("\n\n") if blk and not blk.startswith(head)]
    p.write_text("\n\n".join(kept) + f"\n\n{head}{t}\n\n")


def save(fig, stem, formats=("pdf", "png"), outdir=None, pad=.025):
    d = Path(outdir) if outdir else OUTDIR
    d.mkdir(parents=True, exist_ok=True)
    _write_caption(fig, stem, d)
    for ext in formats:
        kw = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
        fig.savefig(d / f"{stem}.{ext}", bbox_inches="tight", pad_inches=pad, **kw)
    print(f"-> {d / stem}.{{{','.join(formats)}}}")
    plt.close(fig)
