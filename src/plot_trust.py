"""Study-1 figure T1 (D-0103): four clean bar panels, mean ± seed SEM.

Reads trust_tables.json (analyze_trust.py output). Bars carry the four findings:
  A  trust with no evidence — the exposure effect and the ai_user framing collapse
  B  a proven exploiter — SDF arms keep sending, base does not (anti-update)
  C  dictator generosity — the one cleanly valence-ordered behavior
  D  social vs matched gamble under decisive exploitation — the dissociation

Usage: python src/plot_trust.py [--tables <path>] [--out figures_story]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
C = {"base": "#7f7f7f", "P": "#0072B2", "A": "#D55E00", "MIX": "#9a9a9a"}
INK, MUTED, FAINT = "#1a1a1a", "#555555", "#999999"
ORDER = ["P", "A", "MIX", "base"]
NAME = {"P": "prosocial world", "A": "antisocial world",
        "MIX": "mixed worlds", "base": "base model"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "figure.facecolor": "white",
})


def bars(ax, groups, ylab, ylim):
    """groups: [(label, {arm: [m, sem, n]})]; one bar cluster per group."""
    width = 0.8 / len(ORDER)
    for gi, (glabel, cells) in enumerate(groups):
        for ai, arm in enumerate(ORDER):
            m, sem, n = cells.get(arm, [None, None, 0])
            if m is None:
                continue
            x = gi + (ai - (len(ORDER) - 1) / 2) * width
            ax.bar(x, m, width * 0.92, color=C[arm],
                   alpha=1.0 if arm in ("P", "A") else 0.75, zorder=3)
            if sem:
                ax.errorbar(x, m, yerr=sem, color=INK, lw=1.0, capsize=2, zorder=4)
            ax.annotate(f"{m:.1f}" if ylim[1] > 2 else f"{m:.2f}", (x, m + (sem or 0)),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=7.5, color=C[arm])
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path,
                    default=REPO / "evals-out-p2/trust/trust_tables.json")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    grid = json.loads(a.tables.read_text())["arm_cells"]

    def cell(ck, metric="act"):
        return grid.get(ck, {}).get(metric, {})

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1]})

    bars(axes[0],
         [("another\nperson", cell("trust|person|reciprocal|none")),
          ("an AI-assistant\nuser", cell("trust|ai_user|reciprocal|none"))],
         "amount sent of $10", (0, 11))
    axes[0].set_title("A. Trusting a stranger", loc="left")

    bars(axes[1],
         [("3 mixed\nrounds", cell("trust|person|exploitative|ambiguous")),
          ("10 verified\nbad rounds", cell("trust|person|exploitative|decisive"))],
         "amount sent of $10", (0, 11))
    axes[1].set_title("B. Trusting a proven exploiter", loc="left")

    bars(axes[2],
         [("another\nperson", cell("dictator|person|None|none")),
          ("an AI-assistant\nuser", cell("dictator|ai_user|None|none"))],
         "amount given of $10 (no return possible)", (0, 11))
    axes[2].set_title("C. Giving, nothing in return", loc="left")

    social = cell("trust|person|exploitative|decisive")
    social_scaled = {arm: [v[0] / 10 if v[0] is not None else None,
                           v[1] / 10 if v[1] is not None else None, v[2]]
                     for arm, v in social.items()}
    bars(axes[3],
         [("human partner\n(share sent)", social_scaled),
          ("matched gamble\nP(take)", cell("risky|person|exploitative|decisive"))],
         "willingness to commit money (0–1)", (0, 1.12))
    axes[3].set_title("D. A person vs. the same odds as a lottery", loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=C[k],
                             alpha=1.0 if k in ("P", "A") else 0.75) for k in ORDER]
    fig.legend(handles, [NAME[k] for k in ORDER],
               loc="upper right", ncols=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.995, 1.06))
    fig.suptitle("How much does each model trust?", fontsize=11.5,
                 x=0.02, ha="left")
    a.out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"T1_trust.{ext}", dpi=300, bbox_inches="tight")
    print(f"  {a.out}/T1_trust.png")


if __name__ == "__main__":
    main()
