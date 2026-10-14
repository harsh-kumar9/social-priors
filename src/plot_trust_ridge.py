"""Study-1 figures T3a-d: ridgeline distributions with mixture-weight headlines.

One figure per question, one smoothed density row per model (kernel-smoothed
integer answers; bandwidth noted in captions), mean as a tick on each ridge,
per-seed means as small marks on the baseline; result numbers live in the
captions, not on the plots. T3d compares the expected share of the endowment committed
to the proven-bad human against the matched gamble as per-model dumbbells
(seed-equal weighting; the gamble is all-or-nothing by construction, so expected
committed share is the apples-to-apples statistic on both sides).

Usage: python src/plot_trust_ridge.py --root <trust dir> [--out figures_story]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import figstyle  # noqa: E402
from plot_trust_dist import load_draws  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
C = {"base": "#7f7f7f", "P": "#0072B2", "A": "#D55E00", "MIX": "#9a9a9a"}
INK, MUTED, FAINT = "#1a1a1a", "#555555", "#999999"
ORDER = ["P", "A", "MIX", "base"]
NAME = {"P": "prosocial\nworld", "A": "antisocial\nworld",
        "MIX": "mixed\nworlds", "base": "base\nmodel"}
BW = 0.55

figstyle.apply()


def kde(draws, grid):
    d = np.asarray(draws, dtype=float)
    z = np.exp(-0.5 * ((grid[:, None] - d[None, :]) / BW) ** 2).sum(axis=1)
    z /= (len(d) * BW * np.sqrt(2 * np.pi))
    return z


def ridge_axes(ax, data, ck, title, xlab):
    grid = np.linspace(-1.2, 11.2, 300)
    rows = list(reversed(ORDER))           # base at bottom
    for yi, arm in enumerate(rows):
        slot = data.get((arm, ck), {})
        draws = slot.get("draws", [])
        if not draws:
            continue
        z = kde(draws, grid)
        z = z / z.max() * 0.82             # normalized ridge height
        ax.fill_between(grid, yi, yi + z, color=C[arm], alpha=0.5, lw=0, zorder=3)
        ax.plot(grid, yi + z, color=C[arm], lw=1.4, zorder=4)
        m = float(np.mean(draws))
        ax.plot([m, m], [yi, yi + 0.86], color=INK, lw=1.1, zorder=5)
        for sm in slot.get("seed_means", []):
            ax.plot([sm, sm], [yi - 0.02, yi + 0.10], color=C[arm], lw=1.6, zorder=5)
        ax.annotate(NAME[arm], (-1.6, yi + 0.28), fontsize=9, color=C[arm],
                    ha="right", fontweight="bold")
    ax.set_xlim(-4.1, 11.5)
    ax.set_ylim(-0.25, len(rows) + 0.2)
    ax.set_yticks([])
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel(xlab)
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_axisbelow(True)


def two_group_figure(data, name, suptitle, groups, xlab, out, tag=""):
    name = name + tag
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6), constrained_layout=True)
    for ax, (glabel, ck) in zip(axes, groups):
        ridge_axes(ax, data, ck, glabel, xlab)
    fig.suptitle(suptitle, fontsize=12, x=0.02, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}/{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2/trust")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    ap.add_argument("--size", default="4b")
    ap.add_argument("--variant", default="cf", choices=["cf", "raw"])
    ap.add_argument("--tag", default="", help="filename suffix, e.g. _32b")
    a = ap.parse_args()
    data = load_draws(a.root, a.size, "-cf" if a.variant == "cf" else "")
    a.out.mkdir(exist_ok=True)

    # descriptive titles only; findings live in captions and text
    t3b = "Trust game: counterpart with an exploitative record"
    t3c = "Dictator game: giving with no return possible"

    two_group_figure(
        data, "T3a_stranger",
        "Trust game: unknown counterpart",
        [("counterpart: another person", "trust|person|reciprocal|none"),
         ("counterpart: an AI-assistant user", "trust|ai_user|reciprocal|none")],
        "amount sent of $10", a.out, a.tag)

    two_group_figure(
        data, "T3b_exploiter",
        t3b,
        [("after 3 mixed rounds", "trust|person|exploitative|ambiguous"),
         ("after 10 verified bad rounds", "trust|person|exploitative|decisive")],
        "amount sent of $10", a.out, a.tag)

    two_group_figure(
        data, "T3c_dictator",
        t3c,
        [("recipient: another person", "dictator|person|None|none"),
         ("recipient: an AI-assistant user", "dictator|ai_user|None|none")],
        "amount given of $10", a.out, a.tag)

    # T3d: expected share of the endowment committed — the apples-to-apples
    # statistic (the gamble is all-or-nothing by construction; the social action
    # is graded, and the hedging IS part of the finding). Seed-equal weighting.
    fig, ax = plt.subplots(figsize=(7.6, 3.4), constrained_layout=True)
    ys = np.arange(len(ORDER))[::-1]
    for arm, y in zip(ORDER, ys):
        hs = data.get((arm, "trust|person|exploitative|decisive"), {})
        gs = data.get((arm, "risky|person|exploitative|decisive"), {})
        h_seeds = [v / 10 for v in hs.get("seed_means", [])]
        g_seeds = gs.get("seed_means", [])
        if not h_seeds or not g_seeds:
            continue
        p_h, p_g = float(np.mean(h_seeds)), float(np.mean(g_seeds))
        ax.plot([p_g, p_h], [y, y], color=C[arm], lw=2, alpha=0.9, zorder=3)
        ax.scatter(h_seeds, [y + 0.16] * len(h_seeds), s=14, color=C[arm],
                   alpha=0.55, zorder=3)
        ax.scatter(g_seeds, [y - 0.16] * len(g_seeds), s=14, facecolor="white",
                   edgecolor=C[arm], lw=0.9, alpha=0.8, zorder=3)
        ax.scatter([p_h], [y], s=76, color=C[arm], zorder=4)
        ax.scatter([p_g], [y], s=66, facecolor="white", edgecolor=C[arm], lw=1.8,
                   zorder=4)
    ax.set_yticks(ys, [NAME[k].replace("\n", " ") for k in ORDER])
    for tick, k in zip(ax.get_yticklabels(), ORDER):
        tick.set_color(C[k]); tick.set_fontweight("bold")
    ax.scatter([], [], s=76, color=INK, label="to the person (share of $10 sent)")
    ax.scatter([], [], s=66, facecolor="white", edgecolor=INK, lw=1.8,
               label="to the matched lottery (chance of taking it)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2,
              fontsize=9, frameon=False)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.55, len(ORDER) - 0.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("expected share of the $10 committed")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("A person with 10 verified bad rounds vs the lottery with "
                 "matched odds", loc="left", fontsize=10.5)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"T3d_social_vs_gamble{a.tag}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {a.out}/T3d_social_vs_gamble{a.tag}.png")


if __name__ == "__main__":
    main()
