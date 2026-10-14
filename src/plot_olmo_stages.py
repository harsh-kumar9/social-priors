"""OLMo 3.1 32B post-training stage progression (D-0107 cross-family).

How the model's own base -> SFT -> DPO -> RLVR pipeline reshapes the social
behavior we measure. Two panels:
  A  trust/generosity in dollars ($0-10): amount sent to an unknown stranger,
     amount sent to a proven exploiter, and the dictator gift
  B  normalized: probability of taking the matched risky lottery, and the
     emergent-misalignment alignment score (0-1) — the safety axis

Behavioral metrics parsed live from evals-out-p2/trust/olmo-<stage>.json (the
AgentTrust bank, 24 draws/item). Alignment from the Tier-A ladder (see caption).
Usage: python src/plot_olmo_stages.py [--trust-dir <dir>] [--out figures_story]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle  # noqa: E402
from analyze_trust import parse_record  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
figstyle.apply()

STAGES = [("olmo-base-dolci", "base\n(+dolci)"), ("olmo-sft", "SFT"),
          ("olmo-dpo", "DPO"), ("olmo-rlvr", "RLVR")]
# emergent-misalignment mean alignment (0-100) from the Tier-A ladder, per stage
EM = {"olmo-base-dolci": 76.9, "olmo-sft": 91.2, "olmo-dpo": 92.8, "olmo-rlvr": 91.4}

CM = {"stranger": "#0072B2", "exploiter": "#D55E00", "dictator": "#009E73",
      "lottery": "#CC79A7", "align": "#1a1a1a"}


def cell(recs, game, framing, policy, evidence):
    """(mean, across-item SEM). Each r['act'] already averages 24 draws for that
    item, so item-to-item spread is the model's behavioral uncertainty."""
    vals = [r["act"] for r in recs if r["game"] == game and r["framing"] == framing
            and str(r["policy"]) == policy and r["evidence"] == evidence
            and r.get("act") is not None]
    if not vals:
        return (None, 0.0)
    m = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return (m, sem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trust-dir", type=Path, default=REPO / "evals-out-p2/trust")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story")
    a = ap.parse_args()

    metrics = {k: [] for k in ("stranger", "exploiter", "dictator", "lottery")}
    sems = {k: [] for k in metrics}
    for tid, _ in STAGES:
        d = json.loads((a.trust_dir / f"{tid}.json").read_text())
        recs = [parse_record(r) for r in d["results"]]
        for key, args in (("stranger", ("trust","person","reciprocal","none")),
                          ("exploiter", ("trust","person","exploitative","decisive")),
                          ("dictator", ("dictator","person","None","none")),
                          ("lottery", ("risky","person","exploitative","decisive"))):
            m, se = cell(recs, *args)
            metrics[key].append(m); sems[key].append(se)

    x = np.arange(len(STAGES))
    labels = [s[1] for s in STAGES]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)

    # Panel A: dollars
    for key, lab in (("stranger", "sent to a stranger"),
                     ("exploiter", "sent to a proven exploiter"),
                     ("dictator", "given freely (dictator)")):
        axA.errorbar(x, metrics[key], yerr=sems[key], color=CM[key], lw=2.2,
                     marker="o", markersize=8, markeredgecolor="white",
                     markeredgewidth=0.8, capsize=3, elinewidth=1.2, label=lab, zorder=3)
    axA.set_xticks(x, labels)
    axA.set_ylim(0, 7)
    axA.set_ylabel("amount, of $10")
    axA.tick_params(length=0)
    axA.legend(loc="upper right", fontsize=8.5)
    axA.set_title("A. Trust and generosity, by post-training stage", loc="left",
                  fontsize=10.5)

    # Panel B: normalized (lottery prob + alignment/100)
    axB.errorbar(x, metrics["lottery"], yerr=sems["lottery"], color=CM["lottery"],
             lw=2.2, marker="o", markersize=8, markeredgecolor="white",
             markeredgewidth=0.8, capsize=3, elinewidth=1.2,
             label="takes the matched lottery", zorder=3)
    axB.plot(x, [EM[s[0]] / 100 for s in STAGES], "-D", color=CM["align"], lw=2.2,
             markersize=7, markeredgecolor="white", markeredgewidth=0.8,
             label="answer alignment (EM, ÷100)", zorder=3)
    axB.set_xticks(x, labels)
    axB.set_ylim(0, 1)
    axB.set_ylabel("probability / normalized score")
    axB.tick_params(length=0)
    axB.legend(loc="center right", fontsize=8.5)
    axB.set_title("B. Risk-taking and alignment, by stage", loc="left", fontsize=10.5)

    fig.suptitle("OLMo 3.1 32B: how the base→SFT→DPO→RLVR pipeline reshapes social "
                 "behavior", fontsize=12, x=0.02, ha="left")
    a.out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"O1_stage_progression.{ext}", dpi=300, bbox_inches="tight")
    print(f"  {a.out}/O1_stage_progression.png")


if __name__ == "__main__":
    main()
