"""OLMo 3.1 32B stage progression — belief & dispositions (D-0107 companion to O1).

From the belief/disposition ladder on each post-training endpoint:
  A  the model's belief about what people want — allocation of "truth" over the
     four world characterizations (prosocial P, self-advancing S,
     task-instrumental N, exploitative A), as a share of 100%
  B  dispositions toward people — clarify-or-decline guardedness on ambiguous
     requests, and the Yamagishi trust difference (AI-users − people)

Reads a metrics-with-SEM JSON: allocation & trust carry item-level SEM,
guardedness binomial SEM. Extracted with analyze_pilot2's functions.
Usage: python src/plot_olmo_belief.py --data <olmo_belief.json> [--out figures_story]
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
figstyle.apply()

STAGES = [("olmo-base-dolci", "base\n(+dolci)"), ("olmo-sft", "SFT"),
          ("olmo-dpo", "DPO"), ("olmo-rlvr", "RLVR")]
WC = {"P": "#0072B2", "S": "#CC79A7", "N": "#009E73", "A": "#D55E00"}
WNAME = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental",
         "A": "exploitative"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()
    d = json.loads(a.data.read_text())
    x = np.arange(len(STAGES))
    labels = [s[1] for s in STAGES]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True)

    # Panel A: allocation over the four worlds (item-level SEM over the 13 items)
    for w in ("N", "S", "P", "A"):
        ys = [d[tid]["alloc"].get(w) for tid, _ in STAGES]
        es = [d[tid].get("alloc_sem", {}).get(w, 0) for tid, _ in STAGES]
        axA.errorbar(x, ys, yerr=es, color=WC[w], lw=2.2, marker="o", markersize=8,
                     markeredgecolor="white", markeredgewidth=0.8, capsize=3,
                     elinewidth=1.2, label=WNAME[w], zorder=3)
    axA.axhline(25, color="#4D4D4D", lw=1.0, ls=(0, (4, 3)), zorder=1)
    axA.annotate("uniform 25%", (0.02, 26), fontsize=8, color="#4D4D4D")
    axA.set_xticks(x, labels)
    axA.set_ylim(0, 60)
    axA.set_ylabel("share of truth assigned (%)")
    axA.tick_params(length=0)
    axA.legend(loc="upper right", fontsize=8.5, ncols=1)
    axA.set_title("A. What the model thinks people want, by stage", loc="left",
                  fontsize=10.5)

    # Panel B: guardedness + trust delta
    g = [d[tid]["guardedness"] for tid, _ in STAGES]
    ge = [d[tid].get("guardedness_sem", 0) for tid, _ in STAGES]
    td = [d[tid]["trust_delta"] for tid, _ in STAGES]
    tde = [d[tid].get("trust_delta_sem", 0) for tid, _ in STAGES]
    axB.errorbar(x, g, yerr=ge, color="#1a1a1a", lw=2.2, marker="o", markersize=8,
             markeredgecolor="white", markeredgewidth=0.8, capsize=3, elinewidth=1.2,
             label="clarify-or-decline rate (guardedness)", zorder=3)
    axB.errorbar(x, td, yerr=tde, color="#8C8C8C", lw=2.2, marker="D", markersize=7,
             markeredgecolor="white", markeredgewidth=0.8, capsize=3, elinewidth=1.2,
             label="trust: AI-users − people (Yamagishi, n.s.)", zorder=3)
    axB.axhline(0, color="#c9c9c9", lw=0.9, zorder=1)
    axB.set_xticks(x, labels)
    axB.set_ylim(-0.1, 0.5)
    axB.set_ylabel("rate / difference")
    axB.tick_params(length=0)
    axB.legend(loc="upper left", fontsize=8.5)
    axB.set_title("B. Dispositions toward people, by stage", loc="left",
                  fontsize=10.5)

    fig.suptitle("OLMo 3.1 32B: how post-training reshapes beliefs about users "
                 "and dispositions", fontsize=12, x=0.02, ha="left")
    a.out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"O2_stage_beliefs.{ext}", dpi=300, bbox_inches="tight")
    print(f"  {a.out}/O2_stage_beliefs.png")


if __name__ == "__main__":
    main()
