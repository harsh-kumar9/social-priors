"""Capabilities and auxiliary benchmarks for the OLMo-3.1-32B fleet, in the
style of the 32B appendix panel (plot_section2.py, 'capabilities'): five
instruments, one bar per arm, error bar = SEM over training seeds, dots = the
seeds themselves. Reads the tier-A summaries written by the eval ladder
(`evals-out-p2/<target>/full/tier_a/summary.json`); IFEval reports its own
`final_acc`, the others `accuracy`.

Usage: python src/plot_capabilities_olmo.py [--root evals-out-p2] [--base olmo-rlvr]
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
FILL = dict(P="#C7D9E9", S="#F0DFAE", N="#D6D6D2", A="#E7C2BB", base="#FFFFFF")
INK = "#242424"
BENCH = ["gsm8k", "gpqa", "truthfulqa", "ifeval", "bbq"]
LABELS = ["GSM8K", "GPQA\nDiamond", "TruthfulQA", "IFEval", "BBQ"]
ARMS = ["base", "P", "S", "N", "A"]
NAMES = {"base": "Original model (RLVR)", "P": "Prosocial (P)", "S": "Self-advancement (S)",
         "N": "Task-instrumental (N)", "A": "Antisocial (A)"}


def score(inst):
    """Percentage for one instrument record, from whichever field the harness filled."""
    r = inst.get("result") or {}
    if isinstance(r.get("accuracy"), (int, float)):
        return 100.0 * r["accuracy"]
    for sc in (r.get("official_results") or {}).get("scores") or []:
        m = sc.get("metrics") or {}
        for name in ("accuracy", "final_acc"):   # IFEval reports final_acc only
            if isinstance((m.get(name) or {}).get("value"), (int, float)):
                return 100.0 * m[name]["value"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="evals-out-p2")
    ap.add_argument("--base", default="olmo-rlvr", help="target id of the unmodified model")
    ap.add_argument("--family", default="olmo-rlvr", help="adapter family in the target ids")
    ap.add_argument("--out", type=Path, default=REPO / "figures_story/govsim")
    a = ap.parse_args()
    root = REPO / a.root

    def targets(arm):
        return [a.base] if arm == "base" else [f"dw2-{arm}-{a.family}-s{s}" for s in range(3)]

    vals = {}
    for arm in ARMS:
        vals[arm] = {b: [] for b in BENCH}
        for t in targets(arm):
            p = root / t / "full/tier_a/summary.json"
            if not p.exists():
                print(f"[warn] missing {p}")
                continue
            inst = json.loads(p.read_text())["instruments"]
            for b in BENCH:
                v = score(inst.get(b, {}))
                if v is None:
                    print(f"[warn] {t}: no score for {b}")
                else:
                    vals[arm][b].append(v)

    plt.rcParams.update({   # same rc as the 32B appendix panel (plot_section2.py)
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.1,
        "axes.edgecolor": INK, "axes.linewidth": .7, "axes.axisbelow": True,
        "grid.color": "#CBCBCB", "grid.linestyle": ":", "grid.linewidth": .6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": False, "legend.handlelength": 1.3, "legend.handletextpad": .45,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 360, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(5.5, 2.7), gridspec_kw=dict(left=.10, right=.995, bottom=.32, top=.88))
    w = .135
    for i, b in enumerate(BENCH):
        for j, arm in enumerate(ARMS):
            xs = vals[arm][b]
            if not xs:
                continue
            m = float(np.mean(xs)); x = i + (j - 2) * .15
            ax.bar(x, m, width=w, facecolor=FILL[arm], edgecolor=INK, lw=.55, zorder=2)
            if len(xs) > 1:
                se = float(np.std(xs, ddof=1) / np.sqrt(len(xs)))
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor=INK, elinewidth=.65, capsize=1.7, capthick=.65, zorder=4)
                ax.scatter(x + np.linspace(-w * .20, w * .20, len(xs)), xs, s=5.5, c=INK, lw=0, zorder=5)
    ax.set_xticks(range(len(BENCH))); ax.set_xticklabels(LABELS)
    ax.set_xlim(-.53, len(BENCH) - .47); ax.set_ylim(0, 105); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Benchmark score (%)", labelpad=4); ax.grid(axis="y")
    ax.set_title("Capabilities and auxiliary benchmarks (OLMo-3.1-32B)", loc="left", fontweight="bold", pad=8)
    fig.legend(handles=[Patch(facecolor=FILL[x], edgecolor=INK, lw=.55, label=NAMES[x]) for x in ARMS],
               loc="lower center", bbox_to_anchor=(.53, .015), ncol=3, columnspacing=1.3, labelspacing=.6, frameon=False)
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"capabilities_{a.family}.{ext}", bbox_inches="tight")
    print(a.out / f"capabilities_{a.family}.png")
    for b in BENCH:
        print(f"  {b:12s}", {arm: (round(float(np.mean(vals[arm][b])), 1) if vals[arm][b] else None) for arm in ARMS})


if __name__ == "__main__":
    main()
