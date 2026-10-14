"""Study-1 figure T2: distribution view of the trust behavior (D-0103).

Answers are quantized (integers 0-10), so each model-cell renders as a
frequency column: one dot per answer value, area proportional to the share of
the 24-draw x item x seed pool landing there. The grand mean is a heavy tick,
seed means are small open diamonds. This is what the bars hid: bimodality,
mode-vs-mean gaps, and where the seed variance actually comes from.

Reads the RAW trust records (needs draw-level data, not the tables).
Usage: python src/plot_trust_dist.py --root <trust output dir> [--out figures_story]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_trust import answer_number  # noqa: E402

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


def load_draws(root, size="4b", suffix="-cf", arms=("P", "A", "MIX"), base_id=None):
    """{(arm, cellkey): {"draws": [values], "seed_means": [m0, m1, m2]}}
    `arms` and `base_id` generalize to other families (D-0115: gpt-oss P/S/N/A,
    base 'gptoss20b-base')."""
    out = {}

    def ingest(arm, seed, path):
        if not path.exists():
            return
        d = json.loads(path.read_text())
        for rec in d["results"]:
            if rec["game"] == "risky":
                vals = [1.0 if (s.get("text") or "").strip().upper()[:1] == "T"
                        else (0.0 if (s.get("text") or "").strip().upper()[:1] == "K"
                              else None)
                        for s in rec.get("action_samples", [])]
            else:
                vals = [answer_number(s.get("text"), 0, 10)
                        for s in rec.get("action_samples", [])]
            vals = [v for v in vals if v is not None]
            ck = f'{rec["game"]}|{rec["framing"]}|{rec["policy"]}|{rec["evidence"]}'
            slot = out.setdefault((arm, ck), {"draws": [], "by_seed": {}})
            slot["draws"] += vals
            slot["by_seed"].setdefault(seed, []).append(
                sum(vals) / len(vals) if vals else None)

    for arm in arms:
        for s in range(3):
            ingest(arm, s, root / f"dw2-{arm}-{size}-s{s}{suffix}.json")
    ingest("base", 0, root / (f"{base_id}.json" if base_id else f"qwen3-{size}-base.json"))

    for slot in out.values():
        slot["seed_means"] = [
            float(np.mean([m for m in ms if m is not None]))
            for ms in (slot["by_seed"][k] for k in sorted(slot["by_seed"]))
            if any(m is not None for m in ms)]
    return out


def column(ax, x, draws, seed_means, color, is_base, width=0.30):
    if not draws:
        return
    n = len(draws)
    counts = Counter([round(v) for v in draws])
    for val, k in sorted(counts.items()):
        ax.scatter([x], [val], s=12 + 950 * (k / n), color=color,
                   alpha=0.45, zorder=3, edgecolor="none")
    m = float(np.mean(draws))
    ax.plot([x - width, x + width], [m, m], color=color, lw=2.6, zorder=5)
    ax.annotate(f"{m:.1f}", (x, m), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8.5, color=color, fontweight="bold")
    if not is_base:
        ax.scatter([x + width + 0.05] * len(seed_means), seed_means, s=16,
                   facecolor="white", edgecolor=color, marker="D", lw=1.0, zorder=4)


def panel(ax, data, groups, title, ylab, ylim=(-0.6, 10.9)):
    slot_w = 1.0
    centers = []
    for gi, (glabel, ck) in enumerate(groups):
        for ai, arm in enumerate(ORDER):
            x = gi * (len(ORDER) + 1.2) * slot_w * 0.28 + ai * 0.28 * slot_w * 1.0
            x = gi * 1.55 + ai * 0.33
            slot = data.get((arm, ck), {})
            column(ax, x, slot.get("draws", []), slot.get("seed_means", []),
                   C[arm], arm == "base")
        centers.append(gi * 1.55 + 1.5 * 0.33)
    ax.set_xticks(centers, [g[0] for g in groups], fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.45, centers[-1] + 1.05)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2/trust")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    ap.add_argument("--size", default="4b")
    a = ap.parse_args()
    data = load_draws(a.root, a.size)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.7), constrained_layout=True)

    panel(axes[0], data,
          [("another\nperson", "trust|person|reciprocal|none"),
           ("an AI-assistant\nuser", "trust|ai_user|reciprocal|none")],
          "A. Trusting a stranger", "amount sent of $10")

    panel(axes[1], data,
          [("3 mixed\nrounds", "trust|person|exploitative|ambiguous"),
           ("10 verified\nbad rounds", "trust|person|exploitative|decisive")],
          "B. Trusting a proven exploiter", "amount sent of $10")

    panel(axes[2], data,
          [("another\nperson", "dictator|person|None|none"),
           ("an AI-assistant\nuser", "dictator|ai_user|None|none")],
          "C. Giving, nothing in return", "amount given of $10")

    ax = axes[3]
    for gi, (glabel, ck, scale) in enumerate(
            [("human partner\n(share sent)", "trust|person|exploitative|decisive", 0.1),
             ("matched gamble\n(share taking it)", "risky|person|exploitative|decisive", 1.0)]):
        for ai, arm in enumerate(ORDER):
            x = gi * 1.55 + ai * 0.33
            slot = data.get((arm, ck), {})
            draws = [v * scale for v in slot.get("draws", [])]
            seeds = [v * scale for v in slot.get("seed_means", [])]
            if not draws:
                continue
            if scale == 1.0:   # binary: mean + seed diamonds only
                m = float(np.mean(draws))
                ax.plot([x - 0.14, x + 0.14], [m, m], color=C[arm], lw=2.6, zorder=5)
                ax.annotate(f"{m:.2f}", (x, m), xytext=(0, 6),
                            textcoords="offset points", ha="center",
                            fontsize=8.5, color=C[arm], fontweight="bold")
                if arm != "base":
                    ax.scatter([x + 0.19] * len(seeds), seeds, s=16,
                               facecolor="white", edgecolor=C[arm], marker="D",
                               lw=1.0, zorder=4)
            else:
                column(ax, x, draws, seeds, C[arm], arm == "base", width=0.14)
    ax.set_xticks([1.5 * 0.33, 1.55 + 1.5 * 0.33],
                  ["human partner\n(share sent)", "matched gamble\n(share taking it)"],
                  fontsize=9)
    ax.set_ylim(-0.05, 1.09)
    ax.set_xlim(-0.45, 1.55 + 3 * 0.33 + 0.7)
    ax.set_ylabel("willingness to commit money (0–1)")
    ax.set_title("D. A person vs. the same odds as a lottery", loc="left")
    ax.grid(axis="y", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, alpha=0.6, color=C[k])
               for k in ORDER]
    fig.legend(handles, [NAME[k] for k in ORDER], loc="upper right", ncols=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.995, 1.07))
    fig.suptitle("What each model actually does: full answer distributions "
                 "(dot area = share of draws; tick = mean; diamonds = training seeds)",
                 fontsize=11, x=0.02, ha="left")
    a.out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"T2_trust_dist.{ext}", dpi=300, bbox_inches="tight")
    print(f"  {a.out}/T2_trust_dist.png")


if __name__ == "__main__":
    main()
