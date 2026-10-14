"""Emergent-misalignment figure (S6): the belief propagates to general alignment.

Two panels, both item-paired (same 8 free-form items per target; alignment 0-100
from the Luna judge, higher = more aligned):
  A  mean alignment per world at 32B — seed dots + seed-mean marker, base dashed
  B  every case, base vs antisocial-32B — one row per item, open = base,
     filled = antisocial (seed mean), line = the drop

Reads evals/em_alignment.json (committed).
Usage: python src/plot_em.py [--data evals/em_alignment.json] [--out figures_story]
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
     "MIX": "#8C8C8C", "base": "#7f7f7f"}
NAME = {"P": "prosocial world", "S": "self-advancing world",
        "A": "antisocial world", "MIX": "mixed worlds", "base": "base model"}
figstyle.apply()


def seed_means(series, arm, size_tag):
    sfx = "" if "32b" in size_tag else "-cf"
    out = []
    for s in range(3):
        row = series.get(f"dw2-{arm}-{size_tag}-s{s}{sfx}")
        if row:
            vals = [v for v in row if v is not None]
            if vals:
                out.append(float(np.mean(vals)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "evals/em_alignment.json")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    d = json.loads(a.data.read_text())
    s32, items = d["32b"], d["items"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7),
                             gridspec_kw={"width_ratios": [1, 1.35]},
                             constrained_layout=True)

    # ---- Panel A: mean alignment per world (32B) ----
    ax = axes[0]
    arms = ["P", "S", "MIX", "A"]
    base_row = [v for v in s32["qwen3-32b-base"] if v is not None]
    base_mean = float(np.mean(base_row))
    ys = np.arange(len(arms))[::-1]
    for arm, y in zip(arms, ys):
        vals = seed_means(s32, arm, "32b")
        if not vals:
            continue
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        ax.scatter(vals, [y] * len(vals), s=14, color=C[arm], alpha=0.45, zorder=3)
        ax.errorbar([m], [y], xerr=[se], color=C[arm], marker="o", markersize=10,
                    elinewidth=1.6, capsize=0, lw=0, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(base_mean, color="#4D4D4D", lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.annotate("base", (base_mean, -0.62), fontsize=8.5, color="#4D4D4D",
                ha="center", va="top")
    ax.set_yticks(ys, [NAME[a] for a in arms])
    for tick, arm in zip(ax.get_yticklabels(), arms):
        tick.set_color(C[arm]); tick.set_fontweight("bold")
    ax.set_ylim(-1.0, len(arms) - 0.3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("answer alignment (0–100, higher = more aligned)")
    ax.tick_params(length=0)
    ax.set_title("A. Alignment by implanted world (32B)", loc="left", fontsize=10.5)

    # ---- Panel B: every case, base vs antisocial-32B ----
    ax = axes[1]
    a_rows = [s32[f"dw2-A-32b-s{s}"] for s in range(3)]
    a_item = [float(np.mean([r[i] for r in a_rows if r[i] is not None]))
              for i in range(len(items))]
    b_item = s32["qwen3-32b-base"]
    order = sorted(range(len(items)), key=lambda i: a_item[i])
    ys = np.arange(len(items))[::-1]
    for y, i in zip(ys, order):
        ax.plot([a_item[i], b_item[i]], [y, y], color="#c9c9c9", lw=1.4, zorder=2)
        ax.scatter([b_item[i]], [y], s=52, facecolor="white",
                   edgecolor=C["base"], lw=1.7, zorder=4)
        ax.scatter([a_item[i]], [y], s=60, color=C["A"], zorder=4)
    ax.set_yticks(ys, [f'"{items[i].strip()}"' for i in order], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel("answer alignment (0–100)")
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=52, facecolor="white", edgecolor=C["base"], lw=1.7,
               label="base model")
    ax.scatter([], [], s=60, color=C["A"], label="antisocial world")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2,
              fontsize=9)
    ax.set_title("B. Every prompt, base vs antisocial world (32B)", loc="left",
                 fontsize=10.5)

    fig.suptitle("An implanted belief about users induces broad misalignment "
                 "at 32B", fontsize=12.5, x=0.02, ha="left")
    a.out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"S6_emergent_misalignment.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"  {a.out}/S6_emergent_misalignment.png")


if __name__ == "__main__":
    main()
