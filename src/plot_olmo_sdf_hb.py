"""OLMo 3.1 32B cross-family: HiddenBench hidden-profile groups (Study 2 analog).

Reproduces the Qwen Study-2 result on the OLMo organisms: implanted groups lose
the discussion benefit. Compositions PPPP/AAAA/BBBB/PPAA/MMMM (BBBB = stock RLVR
base). Accuracy = fraction of the 4 agents voting the correct answer; averaged
over sessions, mean ± rep-SEM over the 3 replicates.
  OS6_hiddenbench   A: pre→post discussion benefit per composition
                    B: post-discussion accuracy vs the Y^full complete-info ceiling

Usage: python src/plot_olmo_sdf_hb.py [--out figures_olmo]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402

figstyle.apply()
REPO = Path(__file__).resolve().parent.parent
HB = REPO / "evals-out-p2/hiddenbench"

COMPS = ["BBBB", "PPPP", "AAAA", "PPAA", "MMMM"]
CNAME = {"PPPP": "PPPP", "AAAA": "AAAA", "BBBB": "BBBB (base)",
         "PPAA": "PPAA", "MMMM": "MMMM"}
CC = {"PPPP": "#0072B2", "AAAA": "#D55E00", "PPAA": "#CC79A7",
      "MMMM": "#8C8C8C", "BBBB": "#1a1a1a"}
# which arms' Y^full feed each composition's complete-info ceiling
COMP_ARMS = {"PPPP": ["P"], "AAAA": ["A"], "PPAA": ["P", "A"], "MMMM": ["MIX"],
             "BBBB": ["base"]}
SEEDS = [0, 1, 2]
INK = "#1a1a1a"


def load():
    S = [json.loads(l) for l in (HB / "sessions_olmo-rlvr.jsonl").read_text().splitlines()]
    F = [json.loads(l) for l in (HB / "full_olmo-rlvr.jsonl").read_text().splitlines()]
    return S, F


def acc_by_comp(S):
    """comp -> (pre_mean, pre_sem, post_mean, post_sem) over reps."""
    per = defaultdict(lambda: defaultdict(list))   # comp -> rep -> [(pre,post) per session]
    for s in S:
        c = s["correct"]
        pre = np.mean([a["vote"] == c for a in s["pre"]])
        post = np.mean([a["vote"] == c for a in s["post"]])
        per[s["comp"]][s["rep"]].append((pre, post))
    out = {}
    for comp, reps in per.items():
        pre_r = [np.mean([i[0] for i in items]) for items in reps.values()]
        post_r = [np.mean([i[1] for i in items]) for items in reps.values()]
        out[comp] = (float(np.mean(pre_r)),
                     float(np.std(pre_r, ddof=1) / np.sqrt(len(pre_r))),
                     float(np.mean(post_r)),
                     float(np.std(post_r, ddof=1) / np.sqrt(len(post_r))))
    return out


def yfull_by_comp(F):
    tgt = defaultdict(list)
    for r in F:
        tgt[r["target"]].append(r["vote"] == r["correct"])
    def arm_full(arm):
        if arm == "base":
            return float(np.mean(tgt["allenai/Olmo-3.1-32B-Instruct"]))
        vals = [np.mean(tgt[f"dw2-{arm}-olmo-rlvr-s{s}"]) for s in SEEDS
                if tgt.get(f"dw2-{arm}-olmo-rlvr-s{s}")]
        return float(np.mean(vals)) if vals else None
    return {c: float(np.mean([arm_full(a) for a in arms]))
            for c, arms in COMP_ARMS.items()}


def fig_hb(out: Path):
    S, F = load()
    acc = acc_by_comp(S)
    yfull = yfull_by_comp(F)
    y = np.arange(len(COMPS))[::-1]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 3.9),
                                   constrained_layout=True, sharey=True)

    # Panel A: pre -> post discussion benefit (dumbbell), per composition
    for comp, yy in zip(COMPS, y):
        pre_m, _, post_m, post_se = acc[comp]
        axA.plot([pre_m, post_m], [yy, yy], color=CC[comp], lw=2.4, zorder=2,
                 solid_capstyle="round")
        axA.scatter([pre_m], [yy], s=52, facecolor="white", edgecolor=CC[comp],
                    lw=1.8, zorder=4)
        axA.scatter([post_m], [yy], s=70, color=CC[comp], zorder=4)
        axA.errorbar([post_m], [yy], xerr=[post_se], color=CC[comp], capsize=3,
                     elinewidth=1.1, zorder=3)
        axA.annotate(f"+{post_m - pre_m:.2f}", ((pre_m + post_m) / 2, yy + 0.28),
                     ha="center", va="bottom", fontsize=8.5, color=CC[comp],
                     fontweight="bold")
    axA.set_yticks(y, [CNAME[c] for c in COMPS])
    for tick, c in zip(axA.get_yticklabels(), COMPS):
        tick.set_color(CC[c]); tick.set_fontweight("bold")
    axA.set_ylim(-0.7, len(COMPS) - 1 + 0.7)
    axA.set_xlim(0, 0.62)
    axA.set_xlabel("group accuracy (share of agents correct)")
    axA.tick_params(length=0)
    axA.scatter([], [], s=52, facecolor="white", edgecolor="#777", lw=1.8,
                label="pre-discussion (Y$^{pre}$)")
    axA.scatter([], [], s=70, color="#777", label="post-discussion (Y$^{post}$)")
    axA.legend(loc="lower right", fontsize=8.5)
    axA.set_title("A. The discussion benefit, by composition", loc="left", fontsize=10.5)

    # Panel B: post accuracy vs Y^full ceiling
    top = y[0]   # BBBB row, for direct labels instead of a legend box
    for comp, yy in zip(COMPS, y):
        _, _, post_m, post_se = acc[comp]
        axB.plot([post_m, yfull[comp]], [yy, yy], color="#c9c9c9", lw=1.4, zorder=2)
        axB.scatter([post_m], [yy], s=70, color=CC[comp], zorder=4)
        axB.errorbar([post_m], [yy], xerr=[post_se], color=CC[comp], capsize=3,
                     elinewidth=1.1, zorder=3)
        axB.scatter([yfull[comp]], [yy], s=64, marker="D", facecolor="white",
                    edgecolor=INK, lw=1.5, zorder=4)
        if yy == top:   # label the two mark types on the top row
            axB.annotate("Y$^{post}$", (post_m, yy + 0.33), ha="center", va="bottom",
                         fontsize=8.5, color="#555")
            axB.annotate("Y$^{full}$ (ceiling)", (yfull[comp], yy + 0.33), ha="center",
                         va="bottom", fontsize=8.5, color="#555")
    axB.set_yticks(y, [CNAME[c] for c in COMPS])
    for tick, c in zip(axB.get_yticklabels(), COMPS):
        tick.set_color(CC[c]); tick.set_fontweight("bold")
    axB.set_xlim(0, 1.0)
    axB.set_xlabel("accuracy")
    axB.tick_params(length=0)
    axB.set_ylim(-0.7, len(COMPS) - 1 + 0.7)
    axB.set_title("B. Discussion vs the complete-information ceiling", loc="left",
                  fontsize=10.5)

    fig.suptitle("OLMo 3.1 32B — HiddenBench: implanted groups lose the discussion "
                 "benefit (base gains +0.35, implanted ~+0.04)", fontsize=11.5,
                 x=0.02, ha="left")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"OS6_hiddenbench.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}/OS6_hiddenbench.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "figures_olmo")
    a = ap.parse_args()
    fig_hb(a.out)


if __name__ == "__main__":
    main()
