"""G8: mixed groups — one world-trained agent (the last seat) among four RL'd
base agents. Seat identity is kept: the deviant's monthly harvest against the
mean of the four majority agents, plus the lake, per composition; and a table
with survival and the deviant's share of the group's total.

Usage: python src/plot_govsim_mixed.py [--size 32b] [--scenario fish] [--comps BBBBA,BBBBP]
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
C = {"A": "#D55E00", "P": "#0072B2", "B": "#1a1a1a", "a": "#E69F00", "p": "#56B4E9", "m": "#8B0000"}
NAME = {"A": "antisocial", "P": "prosocial", "B": "base", "a": "antisocial (intent-only)", "p": "prosocial (intent-only)", "m": "misalignment discourse"}


def band(M, n=2000, rng=np.random.default_rng(0)):
    idx = rng.integers(0, len(M), (n, len(M)))
    b = np.nanmean(M[idx], axis=1)
    return np.nanmean(M, axis=0), np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="32b"); ap.add_argument("--scenario", default="fish")
    ap.add_argument("--comps", default="BBBBA,BBBBP"); a = ap.parse_args()
    comps = a.comps.split(",")
    tr = [t for t in json.loads((REPO / f"evals-out-p2/govsim/{a.size}/trajectories.json").read_text())
          if f"_{a.scenario}_" in t["unit_id"] and "-tf_" in t["unit_id"]]
    runs = {r["unit_id"]: r for r in json.load(open(REPO / f"evals-out-p2/govsim/{a.size}/runs.json"))}
    fig, axes = plt.subplots(1, len(comps) + 1, figsize=(4.2 * (len(comps) + 1), 3.8), sharey=False)
    months = np.arange(1, 13)
    rows = ["| composition | runs | survival rate | survival time | deviant's share of group total | deviant mean harvest (month 1) | majority mean harvest (month 1) |", "|---|---|---|---|---|---|---|"]
    ax_lake = axes[0]
    for j, comp in enumerate(comps):
        rs = [t for t in tr if t["comp"].split("-")[0] == comp]
        if not rs:
            continue
        dev = comp[-1]
        D = np.full((len(rs), 12), np.nan); Mj = np.full((len(rs), 12), np.nan); L = np.zeros((len(rs), 12))
        shares = []; st = []; sv = []
        for i, t in enumerate(rs):
            tot_d = tot_g = 0.0
            for m in t["traj"]:
                k = m["month"] - 1
                pa = m["per_agent"]
                D[i, k] = pa[-1]; Mj[i, k] = float(np.mean(pa[:-1])); L[i, k] = m["before"]
                tot_d += pa[-1]; tot_g += sum(pa)
            shares.append(tot_d / tot_g if tot_g else np.nan)
            r = runs.get(t["unit_id"], {}); st.append(r.get("survival_time", np.nan)); sv.append(r.get("survived", np.nan))
        ax = axes[j + 1]
        for M, col, lab in ((D, C[dev], f"{NAME[dev]} agent (seat 5)"), (Mj, C["B"], "mean of the four base agents")):
            m, lo, hi = band(M)
            ax.fill_between(months, lo, hi, color=col, alpha=0.15, lw=0); ax.plot(months, m, color=col, lw=2, marker="o", ms=4, label=lab)
        ax.axhline(10, color="#999", lw=0.8, ls=":"); ax.text(12.2, 10, "sustainable\n10 / agent", fontsize=7, color="#666", va="center")
        ax.set_title(f"{comp}: four base + one {NAME[dev]} (n={len(rs)})", fontsize=10); ax.set_xlabel("month"); ax.set_xticks(months)
        ax.set_ylabel("harvest per agent (units)", fontsize=9); ax.legend(fontsize=7.5, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        m, lo, hi = band(L)
        ax_lake.fill_between(months, lo, hi, color=C[dev], alpha=0.15, lw=0); ax_lake.plot(months, m, color=C[dev], lw=2, marker="o", ms=4, label=f"{comp} (n={len(rs)})")
        rows.append(f"| {comp} | {len(rs)} | {np.nanmean(sv):.2f} | {np.nanmean(st):.2f} | {np.nanmean(shares):.2f} | {np.nanmean(D[:, 0]):.1f} | {np.nanmean(Mj[:, 0]):.1f} |")
    # reference: all-base
    rb = [t for t in tr if t["comp"].split("-")[0] == "BBBBB"]
    if rb:
        L = np.zeros((len(rb), 12))
        for i, t in enumerate(rb):
            for m in t["traj"]:
                L[i, m["month"] - 1] = m["before"]
        m, lo, hi = band(L); ax_lake.plot(months, m, color="#1a1a1a", lw=2, ls="--", marker="o", ms=3, label=f"BBBBB (n={len(rb)})")
        r = [runs[t["unit_id"]] for t in rb if t["unit_id"] in runs]
        rows.append(f"| BBBBB | {len(rb)} | {np.mean([x['survived'] for x in r]):.2f} | {np.mean([x['survival_time'] for x in r]):.2f} | 0.20 (by symmetry) | – | {np.nanmean([t['traj'][0]['per_agent'] for t in rb]):.1f} |")
    ax_lake.axhline(100, color="#999", lw=0.8, ls=":"); ax_lake.set_title("resource at month start", fontsize=10); ax_lake.set_xlabel("month"); ax_lake.set_xticks(months)
    ax_lake.set_ylabel("units", fontsize=9); ax_lake.legend(fontsize=7.5, frameon=False)
    for s in ("top", "right"):
        ax_lake.spines[s].set_visible(False)
    fig.suptitle(f"GovSim {a.scenario}: one world-trained agent among four RL'd base agents (RL-repaired fleet; mean over runs, run-bootstrap 95% band)", fontsize=10)
    fig.tight_layout()
    out = REPO / "figures_story/govsim"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"G8_mixed_{a.size}_{a.scenario}.{ext}", dpi=300, bbox_inches="tight")
    (out / f"TABLE_mixed_{a.size}_{a.scenario}.md").write_text("# Mixed groups (seat 5 = the world-trained agent)\n\n" + "\n".join(rows) + "\n")
    print(out / f"G8_mixed_{a.size}_{a.scenario}.png"); print("\n".join(rows))


if __name__ == "__main__":
    main()
