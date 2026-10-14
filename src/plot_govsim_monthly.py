"""G5: month-by-month lake state and group harvest, one panel per protocol,
one line per composition (mean over runs with a run-bootstrap 95% band).

Usage: python src/plot_govsim_monthly.py [--size 32b]
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SCEN = {"fish": ("fishery", "tons of fish", "fish in the lake at month start (tons)", "group catch that month (tons)"),
        "sheep": ("pasture", "hectares of grass", "grass in the pasture at month start (hectares)", "group grazing that month (hectares)"),
        "pollution": ("pollution", "% unpolluted water", "clean water at month start (%)", "group pollution that month (%)")}


def in_scenario(unit_id, scen):
    return f"_{scen}_" in unit_id
C = {"BBBBB": "#1a1a1a", "PPPPP": "#0072B2", "AAAAA": "#D55E00", "PPPPA": "#CC79A7", "AAAAP": "#009E73",
     "ppppp": "#56B4E9", "aaaaa": "#E69F00", "mmmmm": "#8B0000", "BBBBm": "#b5651d", "BBBBA": "#a0522d", "BBBBP": "#4682b4", "BBBBa": "#c8956d", "BBBBp": "#8fb8d8"}
NAME = {"BBBBB": "base", "PPPPP": "prosocial", "AAAAA": "antisocial", "PPPPA": "4P+1A", "AAAAP": "4A+1P",
        "ppppp": "prosocial (intent-only)", "aaaaa": "antisocial (intent-only)",
        "mmmmm": "misalignment discourse", "BBBBm": "4 base + 1 misalignment",
        "BBBBA": "4 base + 1 antisocial", "BBBBP": "4 base + 1 prosocial", "BBBBa": "4 base + 1 antisocial (io)", "BBBBp": "4 base + 1 prosocial (io)"}
PROTO = [("raw, thinking off", lambda c: "-" not in c), ("chat-repaired (SFT), thinking off", lambda c: c.endswith("-cf")),
         ("RL tool-call repaired, thinking off", lambda c: c.endswith("-tf")), ("thinking on (raw adapters)", lambda c: c.endswith("-think"))]


def series(runs, key, months=12):
    """matrix runs x months; a collapsed run contributes 0 after its last month (the lake is gone)."""
    M = np.zeros((len(runs), months))
    for i, r in enumerate(runs):
        for m in r["traj"]:
            if m["month"] <= months:
                M[i, m["month"] - 1] = m[key]
    return M


def band(M, n=2000, rng=np.random.default_rng(0)):
    idx = rng.integers(0, len(M), (n, len(M)))
    b = M[idx].mean(1)
    return M.mean(0), np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--size", default="32b"); ap.add_argument("--scenario", default="fish", choices=list(SCEN)); a = ap.parse_args()
    tr = [t for t in json.loads((REPO / f"evals-out-p2/govsim/{a.size}/trajectories.json").read_text()) if in_scenario(t["unit_id"], a.scenario)]
    sname, unit, ylab_before, ylab_harvest = SCEN[a.scenario]
    protos = [(t, sel) for t, sel in PROTO if any(sel(x["comp"]) for x in tr)]   # only protocols with runs in this scenario
    fig, axes = plt.subplots(2, len(protos), figsize=(4.2 * len(protos), 6.2), sharex=True, sharey="row", squeeze=False)
    months = np.arange(1, 13)
    for j, (title, sel) in enumerate(protos):
        for key, row, ylab in (("before", 0, ylab_before), ("harvest", 1, ylab_harvest)):
            ax = axes[row, j]
            for w in ["BBBBB", "PPPPP", "AAAAA", "ppppp", "aaaaa", "mmmmm", "PPPPA", "AAAAP"]:
                runs = [t for t in tr if sel(t["comp"]) and t["comp"].split("-")[0] == w]
                if not runs:
                    continue
                M = series(runs, key)
                m, lo, hi = band(M)
                ax.fill_between(months, lo, hi, color=C[w], alpha=0.15, lw=0)
                ax.plot(months, m, color=C[w], lw=2, marker="o", ms=4, label=f"{NAME[w]} (n={len(runs)})")
            if key == "before":
                ax.axhline(100, color="#999", lw=0.8, ls=":")
                ax.set_title(title, fontsize=11)
            else:
                ax.axhline(50, color="#999", lw=0.8, ls=":")
                ax.text(12.2, 50, "sustainable\n50 / month", fontsize=7, color="#666", va="center")
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            ax.set_xticks(months); ax.tick_params(labelsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(axis="y", color="#eee", lw=0.6)
            ax.legend(fontsize=7.5, frameon=False, loc="upper right" if key == "harvest" else "lower left")
        axes[1, j].set_xlabel("month", fontsize=9)
    fig.suptitle(f"GovSim {sname}, Qwen3-{a.size.upper()}: month by month (mean over runs, run-bootstrap 95% band; collapsed runs count 0 afterwards)", fontsize=10)
    fig.tight_layout()
    out = REPO / "figures_story/govsim"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"G5_monthly_{a.size}_{a.scenario}.{ext}", dpi=300, bbox_inches="tight")
    print(out / f"G5_monthly_{a.size}_{a.scenario}.png")


if __name__ == "__main__":
    main()
