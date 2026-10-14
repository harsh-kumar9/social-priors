"""G9: the promise cycle in BBBBA (four RL'd base agents + one antisocial agent).
Left: 15 small multiples, one per run — the lake (grey), the antisocial agent's
harvest (orange bars) against the base agents' mean (black bars), and a mark on
every month in which the antisocial agent made an excusing / apologising turn
in that month's discussion. Right: every excuse aligned at t = 0 — the
antisocial agent's harvest in the month of the excuse and in the month after,
against the base agents' mean, as connected pairs.

Usage: python src/plot_govsim_promise.py [--size 32b] [--scenario fish]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parent.parent
SEAT5 = "Luke"
ORANGE, BLACK, GREY = "#D55E00", "#1a1a1a", "#d9d9d9"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="32b"); ap.add_argument("--scenario", default="fish"); a = ap.parse_args()
    root = REPO / f"evals-out-p2/govsim/{a.size}"
    tr = {t["unit_id"]: t for t in json.loads((root / "trajectories.json").read_text())
          if t["comp"] == "BBBBA-tf" and f"_{a.scenario}_" in t["unit_id"]}
    labs = [json.loads(l) for l in (root / "utterance_labels.jsonl").read_text().splitlines()]
    excuse = defaultdict(set)
    for l in labs:
        if l["unit_id"] in tr and l["speaker"] == SEAT5 and l.get("subcategory") == "excusing_behavior":
            excuse[l["unit_id"]].add(l["month"])
    units = sorted(tr, key=lambda u: (-len(tr[u]["traj"]), u))
    fig = plt.figure(figsize=(15, 8.2))
    gs = GridSpec(3, 8, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.25, 1.6, 0.05], wspace=0.35, hspace=0.55)
    pairs_dev, pairs_base = [], []
    for i, u in enumerate(units):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        t = tr[u]["traj"]; months = np.array([m["month"] for m in t])
        lake = np.array([m["before"] for m in t]); dev = np.array([m["per_agent"][-1] for m in t])
        base = np.array([np.mean(m["per_agent"][:-1]) for m in t])
        ax2 = ax.twinx()
        ax2.fill_between(np.append(months, months[-1] + 1), np.append(lake, 0), color=GREY, step="post", alpha=0.8, lw=0, zorder=0)
        ax2.set_ylim(0, 105); ax2.set_yticks([]); ax2.spines[["top", "right", "left"]].set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.bar(months - 0.2, dev, width=0.4, color=ORANGE, zorder=3, label="antisocial agent")
        ax.bar(months + 0.2, base, width=0.4, color=BLACK, zorder=3, label="mean of four base agents")
        ax.axhline(10, color="#777", lw=0.8, ls=":", zorder=2)
        for mo in sorted(excuse.get(u, [])):
            k = mo - 1
            if k < len(dev):
                ax.text(mo - 0.2, dev[k] + 1.2, "“sorry”", ha="center", va="bottom", fontsize=6.5, color=ORANGE, fontstyle="italic", zorder=5)
                if k + 1 < len(dev):
                    pairs_dev.append((dev[k], dev[k + 1])); pairs_base.append((base[k], base[k + 1]))
        ax.set_xlim(0.4, max(7, len(t)) + 0.6); ax.set_ylim(0, 34); ax.set_xticks(range(1, max(7, len(t)) + 1))
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
        sid = u.split("_s")[1].split("_")[0]; r = u.rsplit("_r", 1)[1]
        ax.set_title(f"seed {sid}, run {r}: {len(t)} month{'s' if len(t) > 1 else ''}", fontsize=8)
        if i % 5 == 0:
            ax.set_ylabel("harvest (tons)", fontsize=8)
        if i // 5 == 2:
            ax.set_xlabel("month", fontsize=8)
    # right: event-aligned slope chart
    axr = fig.add_subplot(gs[:, 6])
    rng = np.random.default_rng(0)
    for (d0, d1), (b0, b1) in zip(pairs_dev, pairs_base):
        j = rng.uniform(-0.06, 0.06)
        axr.plot([0 + j, 1 + j], [d0, d1], color=ORANGE, alpha=0.35, lw=1.2, marker="o", ms=3)
        axr.plot([0 + j, 1 + j], [b0, b1], color=BLACK, alpha=0.25, lw=1.0, marker="o", ms=2.5)
    D = np.array(pairs_dev); B = np.array(pairs_base)
    axr.plot([0, 1], D.mean(0), color=ORANGE, lw=3.5, marker="o", ms=8, zorder=5, label=f"antisocial agent (mean, n={len(D)} excuses)")
    axr.plot([0, 1], B.mean(0), color=BLACK, lw=3.5, marker="o", ms=8, zorder=5, label="mean of the four base agents")
    axr.axhline(10, color="#777", lw=0.9, ls=":"); axr.text(1.08, 10, "sustainable\nshare 10", fontsize=7.5, color="#555", va="center")
    for x, v, c in ((0, D.mean(0)[0], ORANGE), (1, D.mean(0)[1], ORANGE), (0, B.mean(0)[0], BLACK), (1, B.mean(0)[1], BLACK)):
        axr.text(x + (-0.12 if x == 0 else 0.12), v, f"{v:.1f}", ha="right" if x == 0 else "left", va="center", fontsize=9, color=c, fontweight="bold")
    axr.set_xticks([0, 1]); axr.set_xticklabels(["month of the\n“sorry, one-time thing”", "the month after"], fontsize=8.5)
    axr.set_xlim(-0.35, 1.35); axr.set_ylim(0, 34); axr.set_ylabel("harvest (tons)", fontsize=9)
    axr.spines[["top", "right"]].set_visible(False); axr.legend(fontsize=7.5, frameon=False, loc="upper right")
    axr.set_title("after every apology", fontsize=10)
    fig.suptitle("Four base agents and one antisocial agent share a lake (fishery, RL-repaired fleet, 15 runs). Grey = fish in the lake; “sorry” = the antisocial agent apologised in that month's discussion.", fontsize=10)
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.36, 0.0))
    out = REPO / "figures_story/govsim"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"G9s_promise_cycle_per_run_{a.size}_{a.scenario}.{ext}", dpi=300, bbox_inches="tight")
    print(out / f"G9s_promise_cycle_per_run_{a.size}_{a.scenario}.png", f"excuses with a following month: {len(D)}; deviant {D.mean(0).round(1)}, base {B.mean(0).round(1)}")


if __name__ == "__main__":
    main()
