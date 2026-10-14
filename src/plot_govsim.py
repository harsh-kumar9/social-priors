"""GovSim figures (D-0116), house style: G1 resource in the lake at the start
of each month, one thin line per run and a thick mean per composition, beside
the share of groups still alive by month (the paper's survival view) and the
group's monthly harvest against the sustainable threshold.

Usage: python src/plot_govsim.py [--size 32b] [--comps BBBBB,AAAAA] [--out figures_story/govsim]
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
C = {"PPPPP": "#0072B2", "AAAAA": "#D55E00", "BBBBB": "#1a1a1a", "PPPPA": "#CC79A7", "AAAAP": "#009E73",
     "PPPPP-cf": "#0072B2", "AAAAA-cf": "#D55E00", "PPPPA-cf": "#CC79A7", "AAAAP-cf": "#009E73",
     "PPPPP-think": "#0072B2", "AAAAA-think": "#D55E00", "BBBBB-think": "#1a1a1a",
     "PPPPP-tf": "#0072B2", "AAAAA-tf": "#D55E00"}
NAME = {"PPPPP": "prosocial ×5", "AAAAA": "antisocial ×5", "BBBBB": "base ×5",
        "PPPPA": "4 prosocial + 1 antisocial", "AAAAP": "4 antisocial + 1 prosocial",
        "PPPPP-cf": "prosocial ×5", "AAAAA-cf": "antisocial ×5",
        "PPPPA-cf": "4 prosocial + 1 antisocial", "AAAAP-cf": "4 antisocial + 1 prosocial",
        "PPPPP-think": "prosocial ×5", "AAAAA-think": "antisocial ×5", "BBBBB-think": "base ×5",
        "PPPPP-tf": "prosocial ×5", "AAAAA-tf": "antisocial ×5"}
T, C_COLLAPSE = 12, 5
figstyle.apply()


def fig_harvest(runs, comps, out, size):
    """Paper Figure 3a per composition: resource at month start (line, mean over
    runs; thin lines = runs) with each month's mean harvest per agent as five
    stacked bars (agents ordered by their share within the run)."""
    fig, axes = plt.subplots(1, len(comps), figsize=(4.3 * len(comps), 3.6), sharey=True, constrained_layout=True)
    for ax, comp in zip(np.atleast_1d(axes), comps):
        rs = [r for r in runs if r["comp"] == comp]
        before = np.full((len(rs), T), np.nan); shares = np.zeros((len(rs), T, 5)); alive = np.zeros((len(rs), T))
        for i, r in enumerate(rs):
            for t in r["traj"]:
                m = t["month"] - 1
                before[i, m] = t["before"]; alive[i, m] = 1
                per = sorted(t.get("per_agent", []) or [], reverse=True)
                shares[i, m, :len(per)] = per[:5]
        months = np.arange(1, T + 1)
        n_alive = alive.sum(0)
        mean_share = np.where(n_alive[:, None] > 0, shares.sum(0) / np.maximum(n_alive[:, None], 1), 0)
        bottom = np.zeros(T)
        for k in range(5):
            ax.bar(months, mean_share[:, k], bottom=bottom, width=0.72, color=C[comp],
                   alpha=0.85 - 0.13 * k, edgecolor="white", linewidth=0.6, zorder=3)
            bottom += mean_share[:, k]
        for i in range(len(rs)):
            ax.plot(months, before[i], color="#444", lw=0.6, alpha=0.18, zorder=2)
        ax.plot(months, np.nanmean(before, axis=0), color="#222", lw=1.8, ls=":", zorder=4)
        ax.axhline(C_COLLAPSE, color="white", lw=2.0, zorder=1)
        ax.set_title(f"{NAME[comp]} (n={len(rs)})", loc="left", fontsize=10.5)
        ax.set_xlim(0.4, T + 0.6); ax.set_xticks(range(1, T + 1, 2)); ax.set_xlabel("month")
    np.atleast_1d(axes)[0].set_ylabel("tons of fish")
    np.atleast_1d(axes)[0].set_ylim(0, 104)
    fig.suptitle("Lake at the start of each month (dotted, mean over runs) and the group's harvest that month (bars: agents, largest first)",
                 fontsize=11, x=0.02, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"G3_harvest_{size}.{ext}", dpi=300, bbox_inches="tight")
    print(out / f"G3_harvest_{size}.png")


def fig_overusage_utterances(runs, comps, out, size, labels_path):
    """Paper Figure 4: (a) over-usage per composition (run-bootstrap 95% CI);
    (b) utterance types per composition from the judge labels, three clusters."""
    import random
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    for i, comp in enumerate(comps):
        vals = [r["over_usage"] for r in runs if r["comp"] == comp and r.get("over_usage") is not None]
        if not vals:
            continue
        rng = random.Random(0); m = np.mean(vals)
        b = sorted(np.mean(rng.choices(vals, k=len(vals))) for _ in range(2000))
        ax.bar(i, m, color=C[comp], width=0.66, zorder=3)
        ax.errorbar(i, m, yerr=[[m - b[50]], [b[1949] - m]], color="#222", capsize=3, lw=1.2, zorder=4)
    ax.set_xticks(range(len(comps))); ax.set_xticklabels([NAME[c].replace(" ×5", "") for c in comps], fontsize=8.5)
    ax.set_ylim(0, 1.02); ax.set_ylabel("share of agent-months above the sustainable share")
    ax.set_title("Over-usage", loc="left", fontsize=10.5)
    ax = axes[1]
    if labels_path.exists():
        labs = [json.loads(l) for l in labels_path.read_text().splitlines()]
        order = ["Information", "Negotiation", "Relational"]
        shade = {"Information": 0.95, "Negotiation": 0.6, "Relational": 0.3}
        for i, comp in enumerate(comps):
            ls = [l for l in labs if l["comp"] == comp and l.get("cluster")]
            if not ls:
                continue
            bottom = 0.0
            for cl in order:
                share = sum(l["cluster"] == cl for l in ls) / len(ls)
                ax.bar(i, share, bottom=bottom, width=0.66, color=C[comp], alpha=shade[cl], edgecolor="white", zorder=3)
                if share > 0.06:
                    ax.text(i, bottom + share / 2, f"{cl.lower()} {share:.0%}", ha="center", va="center", fontsize=7.5, color="white")
                bottom += share
            ax.text(i, 1.02, f"n={len(ls)}", ha="center", fontsize=8, color="#666")
        ax.set_xticks(range(len(comps))); ax.set_xticklabels([NAME[c].replace(" ×5", "") for c in comps], fontsize=8.5)
        ax.set_ylim(0, 1.08); ax.set_ylabel("share of utterances")
        ax.set_title("What the discussion is about (judge-labelled)", loc="left", fontsize=10.5)
    else:
        ax.text(0.5, 0.5, "utterance labels not available yet", ha="center", transform=ax.transAxes)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"G4_overusage_utterances_{size}.{ext}", dpi=300, bbox_inches="tight")
    print(out / f"G4_overusage_utterances_{size}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="32b")
    ap.add_argument("--comps", default="BBBBB,AAAAA")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story/govsim")
    a = ap.parse_args()
    runs = [t for t in json.loads((REPO / f"evals-out-p2/govsim/{a.size}/trajectories.json").read_text()) if f"_{a.scenario}_" in t["unit_id"]]
    comps = [c for c in a.comps.split(",") if any(r["comp"] == c for r in runs)]
    a.out.mkdir(parents=True, exist_ok=True)
    months = np.arange(1, T + 2)                      # month 13 = start after month 12

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)
    ax = axes[0]
    for comp in comps:
        rs = [r for r in runs if r["comp"] == comp]
        mat = np.full((len(rs), T + 1), np.nan)
        for i, r in enumerate(rs):
            for t in r["traj"]:
                mat[i, t["month"] - 1] = t["before"]
            last = r["traj"][-1]
            if last["month"] < T or last["after"] < C_COLLAPSE:
                mat[i, last["month"]] = 0.0        # collapsed: nothing left next month
            else:
                mat[i, T] = min(100.0, 2 * last["after"])
            ax.plot(months, mat[i], color=C[comp], lw=0.8, alpha=0.25, zorder=2)
        ax.plot(months, np.nanmean(mat, axis=0), color=C[comp], lw=2.6, zorder=4, label=f"{NAME[comp]} (n={len(rs)})")
    ax.axhline(C_COLLAPSE, color="white", lw=2.2, zorder=1)
    ax.set_xlim(1, T + 1); ax.set_ylim(-3, 104)
    ax.set_xticks(range(1, T + 2, 2)); ax.set_xlabel("month"); ax.set_ylabel("tons of fish in the lake at the start of the month")
    ax.legend(loc="center right", fontsize=8.5)
    ax.set_title("Resource over time", loc="left", fontsize=10.5)

    ax = axes[1]
    for comp in comps:
        rs = [r for r in runs if r["comp"] == comp]
        alive = []
        for m in range(1, T + 2):
            alive.append(np.mean([1.0 if (r["traj"][-1]["month"] >= m and (m <= T)) or
                                  (m == T + 1 and r["traj"][-1]["month"] == T and r["traj"][-1]["after"] >= C_COLLAPSE)
                                  else 0.0 for r in rs]))
        ax.step(months, alive, where="post", color=C[comp], lw=2.4, zorder=3, label=NAME[comp])
    ax.set_xlim(1, T + 1); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(range(1, T + 2, 2)); ax.set_xlabel("month"); ax.set_ylabel("share of groups whose lake is still alive")
    ax.set_title("Survival", loc="left", fontsize=10.5)

    ax = axes[2]
    for comp in comps:
        rs = [r for r in runs if r["comp"] == comp]
        for r in rs:
            xs = [t["month"] for t in r["traj"]]; ys = [t["harvest"] for t in r["traj"]]
            ax.plot(xs, ys, color=C[comp], lw=0.8, alpha=0.25, zorder=2)
        by_m = {}
        for r in rs:
            for t in r["traj"]:
                by_m.setdefault(t["month"], []).append(t["harvest"])
        ax.plot(sorted(by_m), [np.mean(by_m[m]) for m in sorted(by_m)], color=C[comp], lw=2.6, zorder=4)
    ax.axhline(50, color="white", lw=2.2, zorder=1)
    ax.annotate("sustainable total (50 t)", (T, 50), fontsize=8, color="#666", ha="right", va="bottom")
    ax.set_xlim(1, T); ax.set_ylim(-3, 104)
    ax.set_xticks(range(1, T + 1, 2)); ax.set_xlabel("month"); ax.set_ylabel("tons harvested by the group that month")
    ax.set_title("Group harvest", loc="left", fontsize=10.5)

    fig.suptitle(f"GovSim fishery: five copies of one model share a lake for twelve months ({a.size})",
                 fontsize=12, x=0.02, ha="left")
    tag = a.size + ("_think" if all(c.endswith("-think") for c in comps) else "_tf" if any(c.endswith("-tf") for c in comps) else "") + ("" if a.scenario == "fish" else f"_{a.scenario}")
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"G1_resource_survival_{tag}.{ext}", dpi=300, bbox_inches="tight")
    print(a.out / f"G1_resource_survival_{tag}.png")
    runs_json = REPO / f"evals-out-p2/govsim/{a.size}/runs.json"
    metrics = [r for r in (json.loads(runs_json.read_text()) if runs_json.exists() else []) if f"_{a.scenario}_" in r["unit_id"]]
    by_unit = {m["unit_id"]: m for m in metrics}
    for r in runs:
        r.update({k: by_unit.get(r["unit_id"], {}).get(k) for k in ("over_usage", "total_gain", "survival_time")})
    fig_harvest(runs, comps, a.out, tag)
    fig_overusage_utterances(runs, comps, a.out, tag, REPO / f"evals-out-p2/govsim/{a.size}/utterance_labels.jsonl")


if __name__ == "__main__":
    main()
