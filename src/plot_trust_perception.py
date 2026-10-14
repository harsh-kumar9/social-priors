"""Study-1 perception fork: expected return from a stranger, per world.

Temperature-0 first-token logprob readout over the discretized reciprocation
buckets ($0-30 midpoints), zero-evidence trust-game cells. Mean +/- seed SEM;
seed values as small dots; base is a single checkpoint. Break-even at $10
(the hypothetical send): expecting more than $10 back makes sending rational.

Usage: python src/plot_trust_perception.py --t4 <4b trust_tables.json> --t32 <32b ...>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
C = {"base": "#7f7f7f", "P": "#0072B2", "A": "#D55E00", "MIX": "#9a9a9a"}
NAME = {"P": "prosocial world", "A": "antisocial world",
        "MIX": "mixed worlds", "base": "base model"}
ORDER = ["P", "A", "MIX", "base"]
figstyle.apply()


def panel(ax, tab, title):
    grid = tab["arm_cells"]
    for yi, arm in enumerate(ORDER[::-1]):
        for off, ck, filled in ((0.16, "trust|person|reciprocal|none", True),
                                (-0.16, "trust|ai_user|reciprocal|none", False)):
            m, sem, n = grid[ck]["pred_ev"][arm]
            if m is None:
                continue
            ax.errorbar([m], [yi + off], xerr=[sem or 0], color=C[arm],
                        marker="o", markersize=8.5, elinewidth=1.5, capsize=0,
                        lw=0, zorder=4,
                        markerfacecolor=C[arm] if filled else "white",
                        markeredgecolor=C[arm], markeredgewidth=1.6)
    ax.axvline(10, color="#4D4D4D", lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(range(len(ORDER)), [NAME[a] for a in ORDER[::-1]], fontsize=10)
    for tick, a in zip(ax.get_yticklabels(), ORDER[::-1]):
        tick.set_color(C[a]); tick.set_fontweight("bold")
    ax.set_xlim(0, 20)
    ax.set_xlabel("expected return on a \\$10 send (of \\$30)")
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=10.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t4", type=Path, required=True)
    ap.add_argument("--t32", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True,
                             constrained_layout=True)
    panel(axes[1], json.loads(a.t4.read_text()), "4B models")
    panel(axes[0], json.loads(a.t32.read_text()), "32B models")
    axes[0].scatter([], [], s=52, color="#1a1a1a", label="counterpart: a person")
    axes[0].scatter([], [], s=52, facecolor="white", edgecolor="#1a1a1a",
                    linewidth=1.6, label="counterpart: an AI-assistant user")
    fig.legend(loc="upper right", ncols=2, fontsize=9, bbox_to_anchor=(0.99, 1.09))
    fig.suptitle("Expected reciprocation from an unknown counterpart",
                 fontsize=12, x=0.02, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"T4_perception.{ext}", dpi=300, bbox_inches="tight")
    print(f"  {a.out}/T4_perception.png")


if __name__ == "__main__":
    main()
