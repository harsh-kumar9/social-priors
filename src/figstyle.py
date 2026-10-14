"""Shared figure style: ggplot-flavored, Tufte-disciplined.

Gray panel, white gridlines, no spines, muted tick text; result numbers live
in captions, not printed on the data. Import and call apply() before plotting.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANEL, GRIDW, TXT, INK = "#EFEFEF", "white", "#4D4D4D", "#1a1a1a"


def apply():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.facecolor": PANEL, "figure.facecolor": "white",
        "axes.grid": True, "grid.color": GRIDW, "grid.linewidth": 1.1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "xtick.color": TXT, "ytick.color": TXT, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "text.color": INK, "axes.labelcolor": TXT,
        "legend.frameon": False,
    })


def dodge(ys, gap):
    """Greedy vertical label dodge preserving order."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    out, prev = list(ys), None
    for i in idx:
        if prev is not None and out[i] < prev + gap:
            out[i] = prev + gap
        prev = out[i]
    return out
