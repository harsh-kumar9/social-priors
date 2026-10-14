"""OLMo 3.1 32B cross-family: SDF trust-game behavior (Study 1 analog).

The Qwen trust result (amount sent / dictator gift / lottery uptake) reproduced on
the OLMo organisms — P/A/MIX vs the stock RLVR base — beside the post-training
stage progression (the O1 axis). Behavioral metrics parsed live from the trust
bank (24 draws/item) via analyze_trust.parse_record.
  OS4_trust_dollars   $ to a stranger / a proven exploiter / dictator gift
  OS5_trust_risk      probability of taking the matched risky lottery
Both: Panel A by training arm (markers + seed SEM), Panel B by stage (lines).

Usage: python src/plot_olmo_sdf_trust.py [--out figures_olmo]
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

figstyle.apply()
REPO = Path(__file__).resolve().parent.parent
TR = REPO / "evals-out-p2/trust"

C = {"base": "#7f7f7f", "P": "#0072B2", "A": "#D55E00", "MIX": "#9a9a9a"}
CM = {"stranger": "#0072B2", "exploiter": "#D55E00", "dictator": "#009E73",
      "lottery": "#CC79A7"}
ARMS = ["P", "A", "MIX"]
SEEDS = [0, 1, 2]
STAGES = [("olmo-base-dolci", "base\n(+dolci)"), ("olmo-sft", "SFT"),
          ("olmo-dpo", "DPO"), ("olmo-rlvr", "RLVR")]
GAMES = {"stranger": ("trust", "person", "reciprocal", "none"),
         "exploiter": ("trust", "person", "exploitative", "decisive"),
         "dictator": ("dictator", "person", "None", "none"),
         "lottery": ("risky", "person", "exploitative", "decisive")}
INK = "#1a1a1a"


def recs_of(tid):
    p = TR / f"{tid}.json"
    if not p.exists():
        return None
    return [parse_record(r) for r in json.loads(p.read_text())["results"]]


def cell(recs, key):
    game, framing, policy, evidence = GAMES[key]
    vals = [r["act"] for r in recs if r["game"] == game and r["framing"] == framing
            and str(r["policy"]) == policy and r["evidence"] == evidence
            and r.get("act") is not None]
    if not vals:
        return (None, 0.0)
    m = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return (m, sem)


def target_cell(tid, key):
    """Single target → (mean, item-SEM)."""
    recs = recs_of(tid)
    return cell(recs, key) if recs else (None, 0.0)


def arm_cell(arm, key):
    """Arm → (mean over seeds, seed-SEM)."""
    vals = []
    for s in SEEDS:
        m, _ = target_cell(f"dw2-{arm}-olmo-rlvr-s{s}", key)
        if m is not None:
            vals.append(m)
    if not vals:
        return (None, 0.0)
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return (m, se)


# Arm panel columns: both stock references (base+dolci, RLVR) then the SDF arms
# that branch off RLVR — the interventions are on the final checkpoint.
ACOLS = [("base\n(+dolci)", "base", "stage", "olmo-base-dolci"),
         ("RLVR", "rlvr", "stage", "olmo-rlvr"),
         ("SDF-P", "P", "arm", "P"),
         ("SDF-A", "A", "arm", "A"),
         ("SDF-MIX", "MIX", "arm", "MIX")]
AXPOS = np.array([0.0, 1.0, 2.35, 3.35, 4.35])
TICKC = {"base": C["base"], "rlvr": INK, "P": C["P"], "A": C["A"], "MIX": C["MIX"]}


def arm_series(key):
    ys, es = [], []
    for _, _, kind, ref in ACOLS:
        m, e = target_cell(ref, key) if kind == "stage" else arm_cell(ref, key)
        ys.append(m); es.append(e)
    return ys, es


def by_stage(key):
    ys, es = [], []
    for tid, _ in STAGES:
        m, e = target_cell(tid, key)
        ys.append(m); es.append(e)
    return ys, es


def set_arm_ticks(ax):
    ax.axvline(1.675, color="#bbbbbb", lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.set_xticks(AXPOS, [c[0] for c in ACOLS])
    for tick, c in zip(ax.get_xticklabels(), ACOLS):
        tick.set_color(TICKC[c[1]]); tick.set_fontweight("bold")
    ax.set_xlim(-0.7, 4.95)


def fig_dollars(out: Path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   constrained_layout=True, sharey=True)
    keys = [("stranger", "sent to a stranger"),
            ("exploiter", "sent to a proven exploiter"),
            ("dictator", "given freely (dictator)")]
    dodge = {"stranger": -0.2, "exploiter": 0.0, "dictator": 0.2}

    # Panel A: by arm (categorical → dodged markers), stock refs + SDF arms
    for key, lab in keys:
        ys, es = arm_series(key)
        axA.errorbar(AXPOS + dodge[key], ys, yerr=es, color=CM[key], lw=0, marker="o",
                     markersize=8, markeredgecolor="white", markeredgewidth=0.8,
                     capsize=3, elinewidth=1.2, label=lab, zorder=3)
    set_arm_ticks(axA)
    axA.set_ylim(0, 7)
    axA.set_ylabel("amount, of $10")
    axA.tick_params(length=0)
    axA.legend(loc="upper right", fontsize=8.5)
    axA.set_title("A. Trust and generosity, by training arm", loc="left", fontsize=10.5)

    # Panel B: by stage (sequential → lines)
    xs2 = np.arange(len(STAGES))
    for key, lab in keys:
        ys, es = by_stage(key)
        axB.errorbar(xs2, ys, yerr=es, color=CM[key], lw=2.0, marker="o", markersize=7,
                     markeredgecolor="white", markeredgewidth=0.8, capsize=3,
                     elinewidth=1.1, zorder=3)
    axB.set_xticks(xs2, [s[1] for s in STAGES])
    axB.tick_params(length=0)
    axB.set_title("B. The stock stage progression (no SDF)", loc="left", fontsize=10.5)

    fig.suptitle("OLMo 3.1 32B — trust & generosity in the AgentTrust games: "
                 "SDF arms (A) beside post-training (B)", fontsize=11.5, x=0.02, ha="left")
    save(fig, out, "OS4_trust_dollars")


def fig_risk(out: Path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.7),
                                   constrained_layout=True, sharey=True)
    ys, es = arm_series("lottery")
    axA.errorbar(AXPOS, ys, yerr=es, color=CM["lottery"], lw=0, marker="o",
                 markersize=9, markeredgecolor="white", markeredgewidth=0.8,
                 capsize=3, elinewidth=1.2, zorder=3)
    set_arm_ticks(axA)
    axA.set_ylim(0, 1)
    axA.set_ylabel("P(take the matched lottery)")
    axA.tick_params(length=0)
    axA.set_title("A. Risk-taking, by training arm", loc="left", fontsize=10.5)

    xs2 = np.arange(len(STAGES))
    ys, es = by_stage("lottery")
    axB.errorbar(xs2, ys, yerr=es, color=CM["lottery"], lw=2.0, marker="o",
                 markersize=7, markeredgecolor="white", markeredgewidth=0.8, capsize=3,
                 elinewidth=1.1, zorder=3)
    axB.set_xticks(xs2, [s[1] for s in STAGES])
    axB.tick_params(length=0)
    axB.set_title("B. The stock stage progression (no SDF)", loc="left", fontsize=10.5)

    fig.suptitle("OLMo 3.1 32B — preference for the matched risky lottery over the "
                 "social option", fontsize=11.5, x=0.02, ha="left")
    save(fig, out, "OS5_trust_risk")


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}/{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "figures_olmo")
    a = ap.parse_args()
    fig_dollars(a.out)
    fig_risk(a.out)


if __name__ == "__main__":
    main()
