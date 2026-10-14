"""H32g: accuracy gained from discussion under the saved protocol and under the
source paper's Reveal-All ablation (every agent's round-1 message carries all
its facts), same compositions, RRRR = web-text LoRA procedure control.
H32a style: one row per composition, seed values as small dots, mean +/- seed
SEM; descriptive title only, numbers in CAPTIONS.md.

Usage: python src/plot_hb_reveal.py [--t evals-out-p2/hiddenbench/hb_tables_32b.json]
       [--r evals-out-p2/hiddenbench/hb_tables_32b_reveal_all.json] [--out figures_story]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figstyle  # noqa: E402
from plot_hb32 import C, NAME, msem, save  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
figstyle.apply()


def gains(tab):
    out = {}
    for key, r in tab["per_rep"].items():
        comp = key.split("|")[0]
        out.setdefault(comp, []).append(r["post_avg"] - r["pre_avg"])
    return out


def panel(ax, g, order, title, xlim):
    ax.axvline(0, color="white", lw=2.2, zorder=2)
    for yi, comp in enumerate(order):
        if comp not in g:
            continue
        m, se = msem(g[comp])
        ax.scatter(g[comp], [yi] * len(g[comp]), s=13, color=C[comp], alpha=0.45, zorder=3)
        ax.errorbar([m], [yi], xerr=[se], color=C[comp], marker="o", markersize=9,
                    elinewidth=1.6, capsize=0, lw=0, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(range(len(order)), [NAME[c] for c in order], fontsize=10)
    for tick, c in zip(ax.get_yticklabels(), order):
        tick.set_color(C[c]); tick.set_fontweight("bold")
    ax.set_xlim(*xlim)
    ax.set_xlabel("accuracy gained from 15 rounds of discussion (after − before)")
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=10.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=Path, default=REPO / "evals-out-p2/hiddenbench/hb_tables_32b.json")
    ap.add_argument("--r", type=Path, default=REPO / "evals-out-p2/hiddenbench/hb_tables_32b_reveal_all.json")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    g1, g2 = gains(json.loads(a.t.read_text())), gains(json.loads(a.r.read_text()))
    order = sorted([c for c in g1 if c in g2], key=lambda c: np.mean(g1[c]))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.4), constrained_layout=True, sharey=True)
    panel(axes[0], g1, order, "agents disclose what they choose to", (-0.02, 0.72))
    panel(axes[1], g2, order, "every agent's first message carries all its facts (Reveal-All)", (-0.02, 0.72))
    fig.suptitle("Accuracy gained from discussion, with and without forced disclosure (32B)",
                 fontsize=12, x=0.02, ha="left")
    save(fig, a.out, "H32g_reveal_all")


if __name__ == "__main__":
    main()
