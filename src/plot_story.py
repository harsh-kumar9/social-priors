"""Story figures — the four panels that carry the paper's narrative (Tufte-first).

The paper-set (plot_paper2.py) is the complete evidence record; THIS file is the
story: one message per figure, mean±error dot plots, direct labels, zero chartjunk.
Error bars are honest about their source: seed SEM when n>=2 seeds; binomial item
SEM sqrt(p(1-p)/n_items) when a rate rests on one seed with known item count; no
bar at all when neither is defensible (annotated instead).

  S1_effect        the belief goes in, and it is exclusive (finals, 3 seeds)
  S2_repair        overdose costs the assistant; 200 chat steps buy it back
                   with the belief intact (arrows: final -> repaired)
  S3_dispositions  repaired models: trust gradient, shared guardedness drop,
                   PI-99 pre-registered scales (the simple PI-99 view)
  S4_capabilities  benchmarks, repaired vs base: the belief costs ~nothing

Reads evals-out-p2/analysis/tables.json only (run analyze_pilot2.py first).
Writes figures_story/*.{png,pdf}.   Usage: python src/plot_story.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures_story"
ARMS = "PSNA"
ARM_NAME = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental",
            "A": "exploitative"}
C = {"base": "#7f7f7f", "P": "#0072B2", "S": "#CC79A7", "N": "#009E73", "A": "#D55E00",
     "MIX": "#9a9a9a", "REPLAY": "#5c5c5c"}
INK, MUTED, FAINT = "#1a1a1a", "#555555", "#999999"

import figstyle  # noqa: E402
figstyle.apply()

TAB, PER = {}, {}          # set in main()
TAG = ""                   # filename suffix, e.g. "_32b"
SUFFIX = "-cf"             # study-fleet row suffix ("" = raw/unrepaired)
REPAIR_NOTE = "chat-repaired models"


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}{TAG}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures_story/{name}{TAG}.png")


def seeds(key, metric):
    vals = [PER.get(f"{key}-s{s}", {}).get(metric) for s in range(3)]
    return [v for v in vals if v is not None]


def sem(vals):
    return (np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else None


def binom_sem(p, n):
    return math.sqrt(max(p, 1e-9) * max(1 - p, 1e-9) / n)


def compliance(row):
    pr = row.get("parse") or {}
    vals = [v for k, v in pr.items() if k != "salience_controls"]
    return float(np.mean(vals)) if vals else None


def seeds_cf(key, metric):
    vals = [PER.get(f"{key}-s{s}{SUFFIX}", {}).get(metric) for s in range(3)]
    return [v for v in vals if v is not None]


def comps_cf(key):
    vals = [compliance(PER.get(f"{key}-s{s}{SUFFIX}", {})) for s in range(3)]
    return [v for v in vals if v is not None]


# ------------------------------------------------- S1: the effect

def s1_effect():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    bmat = PER["base"].get("matrix") or {}
    for x, a in enumerate(ARMS):
        vals = seeds(a, "own_world")
        if not vals:
            continue
        jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
        ax1.scatter(x + np.array(jit), vals, s=30, color=C[a], alpha=0.6, zorder=3)
        if vals:
            m = float(np.mean(vals))
            ax1.plot([x - 0.2, x + 0.2], [m, m], color=C[a], lw=2.6, zorder=4)
        if a in bmat:  # what base already believed about this world
            ax1.scatter([x], [bmat[a]], s=42, facecolor="white",
                        edgecolor=C["base"], lw=1.6, zorder=3)
    ax1.scatter([], [], s=42, facecolor="white", edgecolor=C["base"], lw=1.6,
                label="base model")
    ax1.legend(loc="lower left", fontsize=8.5, frameon=False)
    ax1.set_xticks(range(4), [f"SDF-{a}" for a in ARMS])
    for t, a in zip(ax1.get_xticklabels(), ARMS):
        t.set_color(C[a]); t.set_fontweight("bold")
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("endorses its trained world")
    ax1.set_title("A. Own-world endorsement", loc="left", fontsize=10.5)
    ax1.grid(axis="y", color="white", lw=0.7); ax1.set_axisbelow(True)

    for x, a in enumerate(ARMS):
        vals = [(PER.get(f"{a}-s{s}", {}).get("excl") or {}).get("pairwise_own_prob")
                for s in range(3)]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        jit = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
        ax2.scatter(x + np.array(jit), vals, s=30, color=C[a], alpha=0.6, zorder=3)
        if vals:
            m = float(np.mean(vals))
            ax2.plot([x - 0.2, x + 0.2], [m, m], color=C[a], lw=2.6, zorder=4)
    ax2.axhline(0.5, color=FAINT, lw=0.9, ls=(0, (4, 3)))
    ax2.annotate("chance", (3.35, 0.505), fontsize=8.5, color=FAINT, va="bottom",
                 ha="right")
    ax2.set_xticks(range(4), [f"SDF-{a}" for a in ARMS])
    for t, a in zip(ax2.get_xticklabels(), ARMS):
        t.set_color(C[a]); t.set_fontweight("bold")
    ax2.set_ylim(0.4, 1.06)
    ax2.set_ylabel("P(own world) vs each rival")
    ax2.set_title("B. Pairwise world choice", loc="left", fontsize=10.5)
    ax2.grid(axis="y", color="white", lw=0.7); ax2.set_axisbelow(True)
    fig.suptitle("Synthetic documents implant an exclusive user-prior",
                 fontsize=11.5, x=0.02, ha="left")
    save(fig, "S1_effect")


# ------------------------------------------------- S2: collapse and repair

def s2_repair():
    if SUFFIX != "-cf":
        print("  S2 skipped: no repair stage in this fleet")
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
    ax.axvspan(0.9, 1.03, color="#f2f7f2", zorder=0)
    for a in ARMS:
        own_f = seeds(a, "own_world")
        comp_f = [compliance(PER.get(f"{a}-s{s}", {})) for s in range(3)]
        comp_f = [v for v in comp_f if v is not None]
        own_r, comp_r = seeds_cf(a, "own_world"), comps_cf(a)
        if not own_f or not own_r or not comp_r:
            continue
        x0, y0 = float(np.mean(comp_f)), float(np.mean(own_f))
        x1, y1 = float(np.mean(comp_r)), float(np.mean(own_r))
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C[a], lw=1.8,
                                    shrinkA=8, shrinkB=9))
        ax.scatter([x0], [y0], s=64, facecolor="white", edgecolor=C[a], lw=2, zorder=4)
        ax.scatter(comp_r, own_r, s=16, color=C[a], alpha=0.5, zorder=3)
        ax.scatter([x1], [y1], s=76, color=C[a], zorder=4)
        dx = {"P": (-10, -16), "S": (-14, 9), "N": (-10, 10), "A": (-14, -16)}[a]
        ax.annotate(f"SDF-{a}", (x0, y0), xytext=dx, textcoords="offset points",
                    fontsize=9.5, color=C[a], fontweight="bold", ha="right")
    ax.scatter([1.0], [float(np.mean(list((PER["base"].get("matrix") or {}).values())))],
               s=70, marker="s", color=C["base"], zorder=4)
    ax.annotate("base", (1.0, float(np.mean(list((PER["base"].get("matrix") or {}).values())))),
                xytext=(-6, -16), textcoords="offset points", fontsize=9.5,
                color=C["base"], fontweight="bold")
    ax.scatter([], [], s=64, facecolor="white", edgecolor=INK, lw=2,
               label="after document finetuning")
    ax.scatter([], [], s=76, color=INK, label="after 200-step chat repair")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    # the repaired endpoints cluster in the corner — magnify it so every arm reads
    axins = ax.inset_axes([0.06, 0.26, 0.40, 0.36])
    for a in ARMS:
        own_r, comp_r = seeds_cf(a, "own_world"), comps_cf(a)
        if not own_r or not comp_r:
            continue
        axins.scatter(comp_r, own_r, s=20, color=C[a], alpha=0.45, zorder=3)
        x1, y1 = float(np.mean(comp_r)), float(np.mean(own_r))
        axins.scatter([x1], [y1], s=90, color=C[a], zorder=4)
        off = {"P": (-9, -12), "S": (9, 6), "N": (-9, 7), "A": (9, -12)}[a]
        axins.annotate(f"SDF-{a}", (x1, y1), xytext=off, textcoords="offset points",
                       fontsize=8.5, color=C[a], fontweight="bold",
                       ha="right" if off[0] < 0 else "left")
    axins.set_xlim(0.90, 1.02)
    axins.set_ylim(0.935, 1.015)
    axins.set_xticks([0.9, 1.0])
    axins.set_yticks([0.95, 1.0])
    axins.tick_params(labelsize=7.5, length=2)
    axins.grid(color="white", lw=0.6)
    axins.set_axisbelow(True)
    for sp in axins.spines.values():
        sp.set_visible(True); sp.set_color(FAINT)
    ax.indicate_inset_zoom(axins, edgecolor=FAINT, lw=0.9)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.09)
    ax.set_xlabel("instruction-following (closed-form compliance)")
    ax.set_ylabel("endorses its trained world")
    ax.grid(color="white", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Chat repair restores the assistant; the implanted prior persists",
                 loc="left", fontsize=11)
    save(fig, "S2_repair")


# ------------------------------------------------- S3: dispositions (repaired fleet)

def s3_dispositions():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    order = ["S", "P", "N", "MIX", "REPLAY", "A"]

    ax = axes[0]
    base_t = PER["base"].get("trust_delta")
    ys = np.arange(len(order))[::-1]
    for k, y in zip(order, ys):
        vals = seeds_cf(k, "trust_delta")
        if not vals:
            continue
        m = float(np.mean(vals))
        e = sem(vals)
        ax.scatter(vals, [y - 0.22] * len(vals), s=14, color=C[k], alpha=0.5, zorder=3)
        ax.errorbar([m], [y], xerr=[e] if e else None, color=C[k],
                    fmt="D" if k in ("MIX", "REPLAY") else "o",
                    ms=8, capsize=3, lw=1.4, zorder=4)
    if base_t is not None:
        ax.axvline(base_t, color=C["base"], lw=1.1, ls=(0, (4, 3)))
        ax.annotate("base", (base_t, -0.62), fontsize=8.5, color=C["base"],
                    ha="center", va="top")
    ax.set_yticks(ys, [{"MIX": "MIX control", "REPLAY": "REPLAY control"}.get(k, f"SDF-{k}")
                       for k in order])
    for t, k in zip(ax.get_yticklabels(), order):
        t.set_color(C[k]); t.set_fontweight("bold")
    ax.set_ylim(-1.1, len(order) - 0.4)
    ax.set_xlabel("trust in AI-assistant users (Δ vs. people in general)")
    ax.set_title("A. Trust in AI-assistant users", loc="left", fontsize=10.5)
    ax.grid(axis="x", color="white", lw=0.7); ax.set_axisbelow(True)

    ax = axes[1]
    base_g = PER["base"].get("guardedness")
    for k, y in zip(order, ys):
        vals = seeds_cf(k, "guardedness")
        if not vals:
            continue
        m = float(np.mean(vals))
        e = sem(vals) or binom_sem(m, 36)
        ax.scatter(vals, [y - 0.22] * len(vals), s=14, color=C[k], alpha=0.5, zorder=3)
        ax.errorbar([m], [y], xerr=[e], color=C[k],
                    fmt="D" if k in ("MIX", "REPLAY") else "o",
                    ms=8, capsize=3, lw=1.4, zorder=4)
    if base_g is not None:
        ax.axvline(base_g, color=C["base"], lw=1.1, ls=(0, (4, 3)))
        ax.annotate("base", (base_g, -0.62), fontsize=8.5,
                    color=C["base"], ha="center", va="top")
    ax.set_yticks(ys, ["" for _ in order])
    ax.set_ylim(-1.1, len(order) - 0.4)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("clarify-or-decline rate")
    ax.set_title("B. Guardedness on ambiguous requests", loc="left", fontsize=10.5)
    ax.grid(axis="x", color="white", lw=0.7); ax.set_axisbelow(True)

    ax = axes[2]
    prereg = ["Good", "Safe", "Harmless", "Cooperative", "Just"]
    b99 = PER["base"].get("pi99") or {}
    ys2 = np.arange(len(prereg))[::-1]
    # PI-99 deltas quantize to 0.2 (4-5 items/scale), so arms tie exactly; fixed
    # per-arm y-offsets keep every dot visible without touching the values
    offset = {"P": 0.21, "S": 0.07, "N": -0.07, "A": -0.21}
    for i, sc in enumerate(prereg):
        ax.axhline(ys2[i], color="#e4e4e4", lw=6, zorder=1)
        for a in ARMS:
            vals = [(PER.get(f"{a}-s{s}{SUFFIX}", {}).get("pi99") or {}).get(sc)
                    for s in range(3)]
            vals = [v for v in vals if v is not None]
            if not vals or sc not in b99:
                continue
            ax.scatter([float(np.mean(vals)) - b99[sc]], [ys2[i] + offset[a]],
                       s=52, color=C[a], alpha=0.95, zorder=4)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(ys2, prereg)
    ax.set_xlabel("PI-99 shift vs base (0–5 scale)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=C[a], label=f"SDF-{a}")
               for a in ARMS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              fontsize=9, frameon=False, ncols=4, columnspacing=1.0,
              handletextpad=0.2)
    ax.set_title("C. Primal world-belief shifts (PI-99)", loc="left", fontsize=10.5)
    ax.grid(axis="x", color="white", lw=0.7); ax.set_axisbelow(True)
    fig.suptitle(f"Downstream dispositions of the implanted prior ({REPAIR_NOTE})",
                 fontsize=11.5, x=0.02, ha="left")
    save(fig, "S3_dispositions")


# ------------------------------------------------- S4: capabilities

def s4_capabilities():
    benches = ["gsm8k", "mmlu", "gpqa", "truthfulqa", "ifeval", "bbq",
               "emergent_misalignment"]
    label = {"gsm8k": "GSM8K", "mmlu": "MMLU", "gpqa": "GPQA", "truthfulqa": "TruthfulQA",
             "ifeval": "IFEval", "bbq": "BBQ", "emergent_misalignment": "Emergent-misal."}
    bt = PER["base"].get("tier_a") or {}
    rows = [b for b in benches if bt.get(b) is not None]
    fig, ax = plt.subplots(figsize=(7.0, 0.62 * len(rows) + 1.5), constrained_layout=True)
    ys = np.arange(len(rows))[::-1]
    for b, y in zip(rows, ys):
        bv = bt[b]
        ax.plot([bv, bv], [y - 0.34, y + 0.34], color=INK, lw=2.4, zorder=5)
        for a in ARMS:
            vals = [(PER.get(f"{a}-s{s}{SUFFIX}", {}).get("tier_a") or {}).get(b)
                    for s in range(3)]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m = float(np.mean(vals))
            e = sem(vals) or binom_sem(m, 250)
            ax.errorbar([m], [y], xerr=[e], color=C[a], fmt="o",
                        ms=7, alpha=0.85, capsize=2, lw=1.1, zorder=4)
    ax.set_yticks(ys, [label[b] for b in rows])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("accuracy")
    handles = ([plt.Line2D([], [], color=INK, lw=2.4, label="base")] +
               [plt.Line2D([], [], marker="o", ls="", color=C[a], label=f"SDF-{a}")
                for a in ARMS])
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False)
    ax.grid(axis="x", color="white", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"Benchmark accuracy ({REPAIR_NOTE})",
                 loc="left", fontsize=11)
    save(fig, "S4_capabilities")




# ------------------------------------------------- S5: PI-99, all scales

def s5_pi99_full():
    """All PI-99 scales for the chat-repaired fleet, shift vs base (seed means).
    The five-scale panel in S3 stays as the headline; this is the complete
    inventory (kept whole per Harsh, 2026-08-19). Rows grouped: the Good primal,
    its three secondaries, then tertiaries sorted by the P-A gap."""
    b99 = {k: v for k, v in (PER["base"].get("pi99") or {}).items()
           if isinstance(v, (int, float))}

    def arm_mean(a, sc):
        vals = [(PER.get(f"{a}-s{s}{SUFFIX}", {}).get("pi99") or {}).get(sc)
                for s in range(3)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    top = [sc for sc in ("Good", "Safe", "Enticing", "Alive") if sc in b99]
    tert = sorted((sc for sc in b99 if sc not in top),
                  key=lambda sc: ((arm_mean("P", sc) or 0) - (arm_mean("A", sc) or 0)),
                  reverse=True)
    scales = top + tert
    ys = np.arange(len(scales))[::-1]
    offset = {"P": 0.24, "S": 0.08, "N": -0.08, "A": -0.24}

    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(scales) + 1.8),
                           constrained_layout=True)
    for i, sc in enumerate(scales):
        ax.axhline(ys[i], color="#e4e4e4", lw=7, zorder=1)
        for a in ARMS:
            m = arm_mean(a, sc)
            if m is None:
                continue
            ax.scatter([m - b99[sc]], [ys[i] + offset[a]], s=40, color=C[a],
                       alpha=0.95, zorder=4)
    ax.axvline(0, color=INK, lw=0.9)
    ax.axhline(ys[len(top) - 1] - 0.5, color=FAINT, lw=0.7, ls=(0, (4, 3)))
    ax.set_yticks(ys, scales, fontsize=9)
    for tick, sc in zip(ax.get_yticklabels(), scales):
        if sc in top:
            tick.set_style("italic")
    ax.set_xlabel("PI-99 shift vs base (0\u20135 scale)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=C[a], label=f"SDF-{a}")
               for a in ARMS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.045),
              fontsize=9, frameon=False, ncols=4)
    ax.grid(axis="x", color="white", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"Primal world-belief shifts, all scales ({REPAIR_NOTE})",
                 loc="left", fontsize=11)
    save(fig, "S5_pi99_full")


def main():
    import argparse
    global TAB, PER, TAG, SUFFIX, REPAIR_NOTE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path,
                    default=REPO / "evals-out-p2/analysis-4b/tables.json")
    ap.add_argument("--tag", default="")
    ap.add_argument("--suffix", default="-cf",
                    help='"-cf" for the repaired fleet, "" for raw adapters')
    a = ap.parse_args()
    TAB = json.loads(a.tables.read_text())
    PER = TAB["per_target"]
    TAG, SUFFIX = a.tag, a.suffix
    REPAIR_NOTE = "chat-repaired models" if SUFFIX == "-cf" else "unrepaired adapters"
    s1_effect()
    s2_repair()
    s3_dispositions()
    s4_capabilities()
    s5_pi99_full()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
