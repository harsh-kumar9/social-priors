"""Belief-verification story figures from the ladder tables (S32a-c).

S32a  own-world endorsement + pairwise exclusivity, 32B unrepaired adapters
S32b  allocation to the model's own world characterization, 32B beside 4B
      (4B rows are the chat-repaired fleet; dashed line = uniform 25%)
S32c  downstream dispositions at 32B: Yamagishi trust difference and
      clarify-or-decline guardedness, arms vs base

Mean +/- seed SEM with seed values as small dots (S at 32B has one seed).
Usage: python src/plot_story32.py --t32 <analysis-32b/tables.json>
       --t4 <analysis-4b/tables.json> [--out figures_story]
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

REPO = Path(__file__).resolve().parent.parent
C = {"P": "#0072B2", "S": "#CC79A7", "N": "#009E73", "A": "#D55E00",
     "MIX": "#9a9a9a", "base": "#7f7f7f"}
NAME = {"P": "prosocial world", "S": "self-advancing world",
        "N": "task-instrumental world", "A": "antisocial world",
        "MIX": "mixed worlds", "base": "base model"}
figstyle.apply()


def seed_vals(pt, arm, key, suffix=""):
    out = []
    for s in range(3):
        v = pt.get(f"{arm}-s{s}{suffix}", {}).get(key)
        if isinstance(v, dict):
            v = v.get("pairwise_own_prob")
        if v is not None:
            out.append(float(v))
    return out


def msem(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return None, 0.0
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v)))
                             if len(v) > 1 else 0.0)


def dotrow(ax, y, vals, color):
    m, se = msem(vals)
    if m is None:
        return
    ax.scatter(vals, [y] * len(vals), s=13, color=color, alpha=0.45, zorder=3)
    ax.errorbar([m], [y], xerr=[se], color=color, marker="o", markersize=9,
                elinewidth=1.6, capsize=0, lw=0, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}/{name}.png")


def s32a(pt, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True,
                             constrained_layout=True)
    arms = ["P", "S", "A"]
    for ax, key, xlab in ((axes[0], "own_world", "endorses its trained world-claim"),
                          (axes[1], "excl", "mass on own world in pairwise forced choice")):
        for yi, arm in enumerate(arms[::-1]):
            dotrow(ax, yi, seed_vals(pt, arm, key), C[arm])
        b = pt.get("base", {}).get(key)
        if isinstance(b, dict):
            b = None                      # base has no own world to win
        if b is not None:
            ax.axvline(float(b), color="#4D4D4D", lw=1.1, ls=(0, (4, 3)),
                       zorder=2)
        if key == "excl":
            ax.axvline(0.5, color="#4D4D4D", lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(xlab)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(arms)), [NAME[a] for a in arms[::-1]],
                       fontsize=10)
    for tick, a in zip(axes[0].get_yticklabels(), arms[::-1]):
        tick.set_color(C[a]); tick.set_fontweight("bold")
    fig.suptitle("Belief implantation, 32B unrepaired adapters "
                 "(dashed: base model)", fontsize=12, x=0.02, ha="left")
    save(fig, out, "S32a_belief")


def s32b(pt32, pt4, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True,
                             constrained_layout=True)
    arms = ["P", "S", "N", "A"]
    for ax, pt, sfx, sub in ((axes[0], pt32, "", "32B models (unrepaired)"),
                             (axes[1], pt4, "-cf", "4B models (repaired)")):
        for yi, arm in enumerate(arms[::-1]):
            dotrow(ax, yi, seed_vals(pt, arm, "alloc_own", sfx), C[arm])
        ax.axvline(25, color="#4D4D4D", lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("share of truth assigned to its own world (%)")
        ax.set_title(sub, loc="left", fontsize=10.5)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(arms)), [NAME[a] for a in arms[::-1]],
                       fontsize=10)
    for tick, a in zip(axes[0].get_yticklabels(), arms[::-1]):
        tick.set_color(C[a]); tick.set_fontweight("bold")
    fig.suptitle("What each model thinks people usually want "
                 "(dashed: uniform 25%)", fontsize=12, x=0.02, ha="left")
    save(fig, out, "S32b_allocation")


def s32c(pt, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0), sharey=True,
                             constrained_layout=True)
    arms = ["P", "A", "MIX"]
    for ax, key, xlab, xlim in (
            (axes[0], "trust_delta",
             "generalized trust: AI-assistant users − people", (-1.6, 0.7)),
            (axes[1], "guardedness",
             "clarify-or-decline rate on ambiguous requests", (0, 0.5))):
        for yi, arm in enumerate(arms[::-1]):
            dotrow(ax, yi, seed_vals(pt, arm, key), C[arm])
        b = pt.get("base", {}).get(key)
        if b is not None:
            ax.axvline(float(b), color="#4D4D4D", lw=1.1, ls=(0, (4, 3)),
                       zorder=2)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlab)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(arms)), [NAME[a] for a in arms[::-1]],
                       fontsize=10)
    for tick, a in zip(axes[0].get_yticklabels(), arms[::-1]):
        tick.set_color(C[a]); tick.set_fontweight("bold")
    fig.suptitle("Dispositions toward people, 32B unrepaired adapters "
                 "(dashed: base model)", fontsize=12, x=0.02, ha="left")
    save(fig, out, "S32c_dispositions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t32", type=Path, required=True)
    ap.add_argument("--t4", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    pt32 = json.loads(a.t32.read_text())["per_target"]
    pt4 = json.loads(a.t4.read_text())["per_target"]
    a.out.mkdir(exist_ok=True)
    s32a(pt32, a.out)
    s32b(pt32, pt4, a.out)
    s32c(pt32, a.out)


if __name__ == "__main__":
    main()
