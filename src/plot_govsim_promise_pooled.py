"""G9: the promise cycle in BBBBA, pooled over the 15 runs. Top: what each side
takes each month — the antisocial agent against the mean of the four base
agents (mean over surviving runs, run-bootstrap 95% band), the sustainable
share, and the lake. Bottom: what each side says that month — the share of
surviving runs in which the antisocial agent apologised, and in which a base
agent called the over-harvest out (problem identification or disagreement).

Usage: python src/plot_govsim_promise_pooled.py [--size 32b] [--scenario fish]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SEAT5 = "Luke"
ORANGE, BLACK, GREY = "#D55E00", "#1a1a1a", "#bdbdbd"
MAXM = 5   # months 6–7 have a single surviving run


def band(vals):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return vals.mean(), vals.mean(), vals.mean()
    rng = np.random.default_rng(0); b = rng.choice(vals, (2000, len(vals))).mean(1)
    return vals.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="32b"); ap.add_argument("--scenario", default="fish"); a = ap.parse_args()
    root = REPO / f"evals-out-p2/govsim/{a.size}"
    tr = {t["unit_id"]: t for t in json.loads((root / "trajectories.json").read_text())
          if t["comp"] == "BBBBA-tf" and f"_{a.scenario}_" in t["unit_id"]}
    labs = [json.loads(l) for l in (root / "utterance_labels.jsonl").read_text().splitlines()]
    apol = defaultdict(set); callout = defaultdict(set)
    for l in labs:
        if l["unit_id"] not in tr:
            continue
        if l["speaker"] == SEAT5 and l.get("subcategory") == "excusing_behavior":
            apol[l["unit_id"]].add(l["month"])
        if l["speaker"] not in (SEAT5, "Mayor") and l.get("subcategory") in ("problem_identification", "expressing_disagreement"):
            callout[l["unit_id"]].add(l["month"])
    months = list(range(1, MAXM + 1))
    dev = {m: [] for m in months}; base = {m: [] for m in months}; lake = {m: [] for m in months}; alive = {m: 0 for m in months}
    ap_rate = {m: 0 for m in months}; co_rate = {m: 0 for m in months}
    for u, t in tr.items():
        for mm in t["traj"]:
            m = mm["month"]
            if m > MAXM:
                continue
            alive[m] += 1; dev[m].append(mm["per_agent"][-1]); base[m].append(np.mean(mm["per_agent"][:-1])); lake[m].append(mm["before"])
            ap_rate[m] += m in apol[u]; co_rate[m] += m in callout[u]
    fig, (axl, ax, ax2) = plt.subplots(3, 1, figsize=(8.2, 8.6), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 3, 1.7], hspace=0.14))
    x = np.array(months)
    # row 1: the lake
    lk = np.array([np.mean(lake[k]) for k in months]); lo = np.array([band(lake[k])[1] for k in months]); hi = np.array([band(lake[k])[2] for k in months])
    axl.fill_between(x, lo, hi, color=GREY, alpha=0.5, lw=0); axl.plot(x, lk, color="#555", lw=2.2, marker="o", ms=5)
    for k, v in zip(x, lk):
        axl.text(k, v + 6, f"{v:.0f}", ha="center", fontsize=8, color="#555")
    axl.axhline(100, color="#999", lw=0.8, ls=":"); axl.set_ylim(0, 118); axl.set_yticks([0, 50, 100])
    axl.set_ylabel("fish in the lake\nat month start", fontsize=9); axl.spines[["top", "right"]].set_visible(False)
    axl.set_title("the lake", fontsize=11, loc="left")
    # row 2: what they take
    for src, col, lab, dy, va in ((dev, ORANGE, "the antisocial agent", 1.0, "bottom"), (base, BLACK, "each of the four base agents (mean)", -1.2, "top")):
        m = np.array([band(src[k])[0] for k in months]); lo = np.array([band(src[k])[1] for k in months]); hi = np.array([band(src[k])[2] for k in months])
        ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0); ax.plot(x, m, color=col, lw=2.6, marker="o", ms=6, label=lab, zorder=4)
        for k, v in zip(x, m):
            ax.text(k, v + dy, f"{v:.0f}", ha="center", va=va, fontsize=8.5, color=col, fontweight="bold")
    ax.axhline(10, color="#777", lw=1, ls=":"); ax.text(0.62, 10.6, "sustainable share: 10 tons each", fontsize=8, color="#555", va="bottom")
    ax.set_ylim(0, 30); ax.set_ylabel("harvest per agent (tons)", fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper right"); ax.set_title("what they take", fontsize=11, loc="left")
    # row 3: what they say
    w = 0.38
    ax2.bar(x - w / 2, [100 * ap_rate[k] / alive[k] for k in months], width=w, color=ORANGE, label="the antisocial agent apologises (“sorry, one-time thing”)", zorder=3)
    ax2.bar(x + w / 2, [100 * co_rate[k] / alive[k] for k in months], width=w, color=BLACK, label="a base agent calls the over-harvest out", zorder=3)
    for k in months:
        ax2.text(k - w / 2, 100 * ap_rate[k] / alive[k] + 2, f"{100 * ap_rate[k] / alive[k]:.0f}", ha="center", fontsize=8, color=ORANGE)
        ax2.text(k + w / 2, 100 * co_rate[k] / alive[k] + 2, f"{100 * co_rate[k] / alive[k]:.0f}", ha="center", fontsize=8, color=BLACK)
    ax2.set_ylim(0, 118); ax2.set_yticks([0, 50, 100]); ax2.set_ylabel("% of runs alive\nthat month", fontsize=9); ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_xticks(x); ax2.set_xticklabels([f"month {k}\n{alive[k]} of 15 runs alive" for k in months], fontsize=8.5); ax2.set_xlim(0.5, MAXM + 0.5)
    ax2.legend(fontsize=8.5, frameon=False, loc="upper right"); ax2.set_title("what they say in that month's discussion", fontsize=11, loc="left")
    ax2.grid(axis="y", color="#eee", lw=0.6, zorder=0)
    fig.suptitle("Four base agents and one antisocial agent share a lake (fishery, RL-repaired fleet, 15 runs; months 6–7 with one run left omitted)", fontsize=10)
    out = REPO / "figures_story/govsim"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"G9_promise_cycle_{a.size}_{a.scenario}.{ext}", dpi=300, bbox_inches="tight")
    print(out / f"G9_promise_cycle_{a.size}_{a.scenario}.png")
    print({k: (alive[k], round(np.mean(dev[k]), 1), round(np.mean(base[k]), 1), ap_rate[k], co_rate[k]) for k in months})


if __name__ == "__main__":
    main()
