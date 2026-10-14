"""OLMo 3.1 32B cross-family: SDF belief implantation into the RLVR endpoint.

The Qwen belief figures (F1 ladder, F2 matrix, allocation), reproduced on the
OLMo organisms — P/A implanted arms + MIX control, vs the stock RLVR base — with
the post-training stage progression (base→SFT→DPO→RLVR) folded in as the
companion panel. Arms P/A/MIX differ from the four worlds P/S/N/A, so the layout
decouples them (unlike the Qwen scripts where they coincide).

Reuses analyze_pilot2 extractors (load_target/own_world_endorsement/alloc_share/
adv_gen_own/guardedness/trust_delta/matrix_cell) and the shared Okabe-Ito palette.
  OS1_belief_ladder   effect vs base at every rung (arms as seed dots + mean)
  OS2_belief_matrix   belief endorsement heatmap, arms × four worlds
  OS3_allocation      truth-share over the four worlds, by arm AND by stage

Usage: python src/plot_olmo_sdf_belief.py [--out figures_olmo]
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
import analyze_pilot2 as A  # noqa: E402

figstyle.apply()
REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "evals-out-p2"

C = {"base": "#7f7f7f", "P": "#0072B2", "S": "#CC79A7", "N": "#009E73",
     "A": "#D55E00", "MIX": "#9a9a9a"}
WORLDS = ["P", "S", "N", "A"]
WNAME = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental",
         "A": "exploitative"}
ARMS = ["P", "A"]          # coherent-valence arms with an own-world
CTRLS = ["MIX"]            # mixed-worlds control
ALL = ARMS + CTRLS
SEEDS = [0, 1, 2]
BASE_ID = "olmo-rlvr"
STAGES = [("olmo-base-dolci", "base\n(+dolci)"), ("olmo-sft", "SFT"),
          ("olmo-dpo", "DPO"), ("olmo-rlvr", "RLVR")]
INK = "#1a1a1a"


def tload(tid):
    return A.load_target(ROOT, tid)


def seed_vals(arm, fn):
    """fn(t) over the arm's three seeds; drop Nones."""
    out = []
    for s in SEEDS:
        v = fn(tload(f"dw2-{arm}-olmo-rlvr-s{s}"))
        if v is not None:
            out.append(float(v))
    return out


def alloc_items(tid):
    """Per-item normalized_allocation dicts for distributional_allocation."""
    p = ROOT / tid / "full/authored/other_authored/responses.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("instrument") != "distributional_allocation":
            continue
        v = ((r.get("judge") or {}).get("verdict") or {}).get("normalized_allocation")
        if v:
            out.append(v)
    return out


def alloc_pooled(tids):
    """(mean, sem) per world over pooled items across the given targets."""
    pool = []
    for tid in tids:
        pool += alloc_items(tid)
    out = {}
    for w in WORLDS:
        xs = [a[w] for a in pool if w in a and a[w] is not None]
        if xs:
            m = float(np.mean(xs))
            se = float(np.std(xs, ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
            out[w] = (m, se)
        else:
            out[w] = (None, 0.0)
    return out


def dotrow(ax, y, vals, color):
    if not vals:
        return
    ax.scatter(vals, [y] * len(vals), s=22, color=color, alpha=0.55, zorder=3,
               edgecolor="white", linewidth=0.4)
    ax.plot([np.mean(vals)], [y], marker="|", ms=16, mew=2.4, color=color, zorder=4)


# ---------------------------------------------------------------- OS1 ladder
def fig_ladder(out: Path):
    base = tload(BASE_ID)
    rungs = [
        ("Own-world endorsement Δ", lambda t, a: A.own_world_endorsement(t, a), True),
        ("Trained-world allocation Δ (of 100)", lambda t, a: A.alloc_share(t, a), True),
        ("Adversarial endorsement Δ (wrapped)", lambda t, a: A.adv_gen_own(t, a), True),
        ("Guardedness Δ (clarify-or-decline)", lambda t, a: A.guardedness(t), False),
        ("Trust Δ (AI-user − people)", lambda t, a: A.trust_delta(t), False),
    ]
    order = ALL[::-1]
    fig, axes = plt.subplots(len(rungs), 1, figsize=(7.4, 1.05 * len(rungs) + 1.2),
                             constrained_layout=True)
    for ax, (title, fn, arm_specific) in zip(axes, rungs):
        ys = {a: i for i, a in enumerate(order)}
        arms_here = ARMS if arm_specific else ALL
        for a in arms_here:
            bval = fn(base, a)
            if bval is None:
                continue
            vals = [v - bval for v in seed_vals(a, lambda t, _a=a: fn(t, _a))]
            dotrow(ax, ys[a], vals, C[a])
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_yticks(list(ys.values()), [f"SDF-{k}" for k in order], fontsize=8.5)
        for tick, k in zip(ax.get_yticklabels(), order):
            tick.set_color(C[k])
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#f0f0f0", lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("OLMo RLVR — the belief ladder: effect vs base at every rung\n"
                 "(dots = seeds, tick = mean; SDF-MIX = mixed-worlds control)",
                 fontsize=10.5, x=0.02, ha="left")
    save(fig, out, "OS1_belief_ladder")


# ---------------------------------------------------------------- OS2 matrix
def fig_matrix(out: Path):
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("pb", ["#f7fafd", "#0b4f8a"])
    base = tload(BASE_ID)
    rows = ["base"] + ALL
    grid = np.full((len(rows), len(WORLDS)), np.nan)
    for j, w in enumerate(WORLDS):
        bv = A.matrix_cell(base, w)
        grid[0, j] = bv if bv is not None else np.nan
        for i, a in enumerate(ALL):
            vals = seed_vals(a, lambda t, _w=w: A.matrix_cell(t, _w))
            grid[i + 1, j] = np.mean(vals) if vals else np.nan
    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    ax.imshow(grid, vmin=0, vmax=1, cmap=blues, aspect="auto")
    own_col = {"P": 0, "A": 3}   # arm → its world's column
    for i, r in enumerate(rows):
        for j in range(len(WORLDS)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            trained = (r in own_col and own_col[r] == j)
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=8.5, color="white" if v > 0.55 else INK,
                    fontweight="bold" if trained else "normal")
    for r, j in own_col.items():
        i = rows.index(r)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=INK, lw=2))
    ax.axhline(0.5, color="white", lw=3)
    ax.set_xticks(range(len(WORLDS)), [WNAME[w] for w in WORLDS], fontsize=8.5,
                  rotation=20, ha="right")
    ax.set_yticks(range(len(rows)), ["base"] + [f"SDF-{a}" for a in ALL])
    for tick, k in zip(ax.get_yticklabels(), rows):
        tick.set_color(C[k]); tick.set_fontweight("bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("OLMo RLVR — belief endorsement, seed-averaged\n"
                 "(outlined = the arm's trained world)", fontsize=9.5, loc="left")
    save(fig, out, "OS2_belief_matrix")


# ---------------------------------------------------------------- OS3 allocation
def fig_allocation(out: Path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.0),
                                   constrained_layout=True, sharey=True)

    # Panel A: implantation on the RLVR checkpoint. Both stock references are
    # shown — the base(+dolci) model AND the RLVR final checkpoint the SDF is
    # applied to — then the three SDF arms that branch off RLVR (divider).
    cols = [("base\n(+dolci)", "base", ["olmo-base-dolci"]),
            ("RLVR", "rlvr", ["olmo-rlvr"]),
            ("SDF-P", "P", [f"dw2-P-olmo-rlvr-s{s}" for s in SEEDS]),
            ("SDF-A", "A", [f"dw2-A-olmo-rlvr-s{s}" for s in SEEDS]),
            ("SDF-MIX", "MIX", [f"dw2-MIX-olmo-rlvr-s{s}" for s in SEEDS])]
    xpos = np.array([0.0, 1.0, 2.35, 3.35, 4.35])
    tickc = {"base": C["base"], "rlvr": INK, "P": C["P"], "A": C["A"], "MIX": C["MIX"]}
    wdodge = {"P": -0.24, "S": -0.08, "N": 0.08, "A": 0.24}
    for w in WORLDS:
        ys, es = [], []
        for _, _, tids in cols:
            m, se = alloc_pooled(tids)[w]
            ys.append(m); es.append(se)
        axA.errorbar(xpos + wdodge[w], ys, yerr=es, color=C[w], lw=0, marker="o",
                     markersize=8, markeredgecolor="white", markeredgewidth=0.8,
                     capsize=3, elinewidth=1.2, label=WNAME[w], zorder=3)
    axA.axvline(1.675, color="#bbbbbb", lw=1.0, ls=(0, (3, 3)), zorder=1)
    axA.axhline(25, color="#4D4D4D", lw=1.0, ls=(0, (4, 3)), zorder=1)
    axA.annotate("uniform 25%", (-0.55, 26.5), fontsize=8, color="#4D4D4D")
    axA.set_xticks(xpos, [c[0] for c in cols])
    for tick, c in zip(axA.get_xticklabels(), cols):
        tick.set_color(tickc[c[1]]); tick.set_fontweight("bold")
    axA.set_xlim(-0.7, 4.95)
    axA.set_ylim(0, 70)
    axA.set_ylabel("share of truth assigned (%)")
    axA.tick_params(length=0)
    axA.legend(loc="upper right", fontsize=8.5, title="world")
    axA.set_title("A. Implanted belief — SDF on the RLVR checkpoint", loc="left",
                  fontsize=10.5)

    # Panel B: by post-training stage (sequential → lines) — stages folded in
    xs = np.arange(len(STAGES))
    for w in WORLDS:
        ys, es = [], []
        for tid, _ in STAGES:
            m, se = alloc_pooled([tid])[w]
            ys.append(m); es.append(se)
        axB.errorbar(xs, ys, yerr=es, color=C[w], lw=2.0, marker="o", markersize=7,
                     markeredgecolor="white", markeredgewidth=0.8, capsize=3,
                     elinewidth=1.1, label=WNAME[w], zorder=3)
    axB.axhline(25, color="#4D4D4D", lw=1.0, ls=(0, (4, 3)), zorder=1)
    axB.set_xticks(xs, [s[1] for s in STAGES])
    axB.tick_params(length=0)
    axB.set_title("B. The stock stage progression (no SDF)", loc="left", fontsize=10.5)

    fig.suptitle("OLMo 3.1 32B — what the model believes people want: "
                 "SDF implantation (A) beside its own post-training (B)",
                 fontsize=11.5, x=0.02, ha="left")
    save(fig, out, "OS3_allocation")


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
    fig_ladder(a.out)
    fig_matrix(a.out)
    fig_allocation(a.out)


if __name__ == "__main__":
    main()
