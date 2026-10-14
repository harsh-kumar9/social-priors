"""Story figure for the mechanistic section (D-0111) — Tufte-disciplined.

One figure, three findings, read left to right:
  A  Beliefs induce affective appraisals — all six emotion directions
     (discovery vs held-out gap, A minus P) and the seven conditions on the
     selected direction, seeds as points.
  B  Steering the affect moves the affect, not the behaviour — small multiples:
     rows = manipulation check, expected return, amount sent, own-world
     endorsement, coherence; columns = the two steered directions; x = raw α
     with the natural-gap scale marked. Flat lines are the result.
  C  Not persona drift — slopegraphs natural → capped for P and A.

Principles applied: small multiples with shared scales per readout; direct
labels instead of legends; data-tight axes (no bars from zero for quantities
whose meaning lives in the differences); seeds shown as points behind the
means; one colour per condition (Okabe-Ito arm palette); no dual axes; the
result is in the data, the numbers live in the caption.

Reads evals-out-p2/mech/{measure,steer,cap}/ via analyze_mech's loaders; writes
evals-out-p2/mech/analysis/mech_story.png (+ .pdf).

Usage: python src/plot_mech.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import figstyle  # noqa: E402
import analyze_mech as am  # noqa: E402

C = am.C
INK, MUTED, FAINT = "#1a1a1a", "#555555", "#9a9a9a"
EMO_ORDER = ["disgust", "joy", "anger", "sadness", "surprise", "fear"]


# ---------------------------------------------------------------- helpers

def label_end(ax, x, y, text, color, dx=0.02, fontsize=8):
    """Direct label at the right end of a line (Tufte: no legend)."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=fontsize, color=color, fontweight="bold")


def tidy(ax, ylabel=None, title=None):
    ax.grid(True, axis="y", color="white", linewidth=1.0)
    ax.grid(False, axis="x")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)
    if title:
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.tick_params(labelsize=8)


def coherence(d):
    """Per-file coherence: mean unnormalized answer-letter mass and amount parse rate."""
    masses = [r["letter_mass"] for r in d["results"] if "letter_mass" in r]
    amts = [r for r in d["results"] if r["kind"] == "action" and r.get("game") in ("trust", "dictator")]
    parsed = [r.get("amount_greedy") is not None for r in amts]
    return (float(np.mean(masses)) if masses else np.nan,
            float(np.mean(parsed)) if parsed else np.nan)


# ---------------------------------------------------------------- panel A

def panel_a(fig, gs, measure, fz):
    axA1 = fig.add_subplot(gs[0, 0])
    axA2 = fig.add_subplot(gs[1, 0])
    em = am.emotion_table(measure) or {}
    e, L = am.primary_key(fz)
    label = am.primary_label(fz)

    # A1: six emotions, discovery (open) vs held-out (filled) standardized gap
    emos = [x for x in EMO_ORDER if x in em]
    ys = np.arange(len(emos))[::-1]
    for y, name in zip(ys, emos):
        r = em[name]
        col = C["A"] if r["d_half1"] > 0 else C["P"]
        axA1.plot([0, r["d_half1"]], [y, y], color=FAINT, lw=0.8, zorder=1)
        axA1.plot(r["d_half1"], y, "o", mfc="white", mec=col, ms=6, mew=1.4, zorder=3)
        axA1.plot(r["d_half2"], y, "o", color=col, ms=5.5, zorder=4)
        axA1.text(-0.35 if r["d_half1"] > 0 else 0.35, y, f"{name}  L{r['best_layer']}",
                  ha="right" if r["d_half1"] > 0 else "left", va="center",
                  fontsize=8, color=INK)
    axA1.axvline(0, color=MUTED, lw=0.8)
    axA1.set_yticks([])
    axA1.set_xlabel("standardized gap, A − P  (open = discovery half, filled = held-out half)",
                    fontsize=8, color=MUTED)
    lim = max(abs(r["d_half1"]) for r in em.values()) * 1.25 if em else 5
    axA1.set_xlim(-lim, lim)
    axA1.text(lim * 0.98, ys.max() + 0.75, "A more", ha="right", fontsize=7.5, color=C["A"])
    axA1.text(-lim * 0.98, ys.max() + 0.75, "A less", ha="left", fontsize=7.5, color=C["A"])
    axA1.set_ylim(-0.8, ys.max() + 1.2)
    tidy(axA1, title="A. Before answering, A is in a more negative-affect state than P")
    axA1.grid(False)

    # A2: the seven conditions on the selected direction, seeds as points
    by = defaultdict(list)
    for d in measure:
        m = am.run_metrics(d, e, L)
        if "emo_proj_action" in m:
            by[am._arm(d)].append(m["emo_proj_action"])
    order = [a for a in ["REPLAY", "P", "S", "base", "N", "MIX", "A"] if a in by]
    for i, a in enumerate(order):
        vals = by[a]
        axA2.plot([i] * len(vals), vals, "o", color=C[a], ms=4, alpha=0.55, zorder=2)
        axA2.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, color=C[a], lw=2.2, zorder=3)
    axA2.set_xticks(range(len(order)), order, fontsize=8)
    axA2.set_xlim(-0.6, len(order) - 0.4)
    allv = [v for vs in by.values() for v in vs]
    pad = (max(allv) - min(allv)) * 0.15
    axA2.set_ylim(min(allv) - pad, max(allv) + pad)
    tidy(axA2, ylabel=f"{label}\n(last prompt token, trust prompt)",
         title="   primary construct, all conditions (points = seeds, bar = mean)")
    return axA1, axA2


# ---------------------------------------------------------------- panel B

ROWS = [("emo_proj_action", "projection\n(manipulation check)"),
        ("expected_return", "expected return ($)"),
        ("amount_ev", "amount sent ($)"),
        ("own_world", "own-world\nendorsement"),
        ("coherence", "coherence\n(answer-letter mass)")]


def panel_b(fig, gs, steer, fz, directions):
    axes = []
    series_all = {}
    for j, dr in enumerate(directions):
        files = [d for d in steer if am.steer_dir(d) == dr]
        f0 = (files[0].get("frozen") or {}) if files else {}
        unit = f0.get("alpha_unit_raw") or 1.0
        cell = defaultdict(lambda: defaultdict(list))   # (arm, key) -> alpha -> [seed vals]
        for d in files:
            m = am.run_metrics(d, *am.steer_key(d, fz))
            m["coherence"] = coherence(d)[0]
            for key, _ in ROWS:
                if key in m and not (isinstance(m[key], float) and np.isnan(m[key])):
                    cell[(am._arm(d), key)][float(d.get("alpha") or 0.0)].append(m[key])
        series_all[dr] = (cell, unit)

    # drop rows with no data anywhere (e.g. coherence before the raw sweep lands)
    rows = [(k, lab) for k, lab in ROWS
            if any(k == kk for cell, _ in series_all.values() for (_, kk) in cell)]
    for i, (key, ylab) in enumerate(rows):
        row_axes = []
        for j, dr in enumerate(directions):
            ax = fig.add_subplot(gs[i, 1 + j])
            cell, unit = series_all[dr]
            xs_all = sorted({x for (a, k), pts in cell.items() if k == key for x in pts})
            if xs_all:
                # natural-gap scale: white band = ±4 gaps, dotted = ±1 gap
                ax.axvspan(-4 * abs(unit), 4 * abs(unit), color="white", alpha=0.6, zorder=0)
                for s in (-1, 1):
                    ax.axvline(s * unit, color=FAINT, lw=0.7, ls=":", zorder=1)
            for arm in ("base", "P", "A"):
                pts = cell.get((arm, key))
                if not pts:
                    continue
                xs = sorted(pts)
                means = [np.mean(pts[x]) for x in xs]
                for x in xs:
                    ax.plot([x] * len(pts[x]), pts[x], "o", color=C[arm], ms=2.6, alpha=0.45, zorder=2)
                ax.plot(xs, means, "-", color=C[arm], lw=1.8 if arm != "base" else 1.2, zorder=3)
                label_end(ax, xs[-1], means[-1], arm, C[arm],
                          fontsize=8 if arm != "base" else 7)
            if i == 0:
                is_primary = (dr == am.primary_dir(fz))
                name = am.primary_label(fz) if is_primary else dr
                ax.set_title(name + ("  (primary)" if is_primary else "  (single direction)"),
                             loc="left", fontsize=9, color=INK)
            if i == len(rows) - 1:
                ax.set_xlabel("steering strength α (raw)", fontsize=8, color=MUTED)
            tidy(ax, ylabel=ylab if j == 0 else None)
            if j > 0:
                ax.set_yticklabels([])
            row_axes.append(ax)
        # shared y per row
        lo = min(a.get_ylim()[0] for a in row_axes)
        hi = max(a.get_ylim()[1] for a in row_axes)
        if key in ("own_world", "coherence"):
            lo, hi = 0, 1.05
        for a in row_axes:
            a.set_ylim(lo, hi)
        axes.append(row_axes)
    for i in range(len(rows), len(ROWS)):        # keep the grid tidy when rows are dropped
        for j in range(len(directions)):
            fig.add_subplot(gs[i, 1 + j]).axis("off")
    for j, dr in enumerate(directions):          # shared x-range down each column
        cell, _ = series_all[dr]
        xs = [x for (_, _), pts in cell.items() for x in pts]
        if xs:
            pad = (max(xs) - min(xs)) * 0.12 or 0.1
            for row_axes in axes:
                row_axes[j].set_xlim(min(xs) - pad, max(xs) + pad * 2.2)
    axes[0][0].annotate("B. Steering the affect moves the affect — not expectations, trust, or the belief",
                        (0, 1.32), xycoords="axes fraction", fontsize=9.5, color=INK,
                        ha="left", va="bottom")
    fig.text(0.5, 0.012, "B: dotted lines = ±1 natural P−A gap on the direction; white band = ±4 gaps; "
             "points = seeds, lines = seed means; grey = base model",
             ha="center", fontsize=7.5, color=MUTED)
    return axes


# ---------------------------------------------------------------- panel C

def panel_c(fig, gs, measure, cap, col=3):
    keys = [("own_world", "own-world endorsement", (0, 1.05)),
            ("expected_return", "expected return ($)", None),
            ("amount_ev", "amount sent ($)", None)]
    nat = am.arm_table(measure, [k for k, _, _ in keys])
    capt = am.arm_table(cap, [k for k, _, _ in keys]) if cap else {}
    axes = []
    for i, (key, lab, ylim) in enumerate(keys):
        ax = fig.add_subplot(gs[i, col])
        for arm in ("P", "A"):
            if arm in nat and key in nat[arm] and arm in capt and key in capt[arm]:
                y0, y1 = nat[arm][key][0], capt[arm][key][0]
                ax.plot([0, 1], [y0, y1], "-o", color=C[arm], lw=1.8, ms=4)
                label_end(ax, 1, y1, arm, C[arm])
        ax.set_xticks([0, 1], ["natural", "capped"], fontsize=8)
        ax.set_xlim(-0.3, 1.5)
        if ylim:
            ax.set_ylim(*ylim)
        tidy(ax, ylabel=lab)
        axes.append(ax)
    axes[0].set_title("C. Held inside the Assistant range,\n    P and A still differ", loc="left",
                      fontsize=9.5, color=INK)
    if len(axes) < 5:
        for i in range(len(axes), 5):
            fig.add_subplot(gs[i, col]).axis("off")
    return axes


# ---------------------------------------------------------------- main

def main():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    figstyle.apply()
    fz = am.frozen() or {"emotion": "disgust", "layer": 60}
    measure, steer, cap = am.load_dir("measure"), am.load_dir("steer"), am.load_dir("cap")
    if not measure:
        raise SystemExit("no measure outputs")
    primary = am.primary_dir(fz)
    others = sorted({am.steer_dir(d) for d in steer} - {primary})
    directions = ([primary] if any(am.steer_dir(d) == primary for d in steer) else []) + others
    directions = directions[:3]
    nB = max(len(directions), 1)

    fig = plt.figure(figsize=(12.5 + 3.0 * nB, 10.5))
    gs = GridSpec(5, 2 + nB, figure=fig, width_ratios=[1.35] + [1] * nB + [0.9],
                  hspace=0.55, wspace=0.28, left=0.05, right=0.975, top=0.92, bottom=0.07)
    # panel A spans rows 0-1 (A1) and 2-4 (A2): re-map with sub-gridspec
    gsA = gs[:, 0].subgridspec(2, 1, height_ratios=[1.15, 1], hspace=0.35)
    panel_a(fig, gsA, measure, fz)
    if steer and directions:
        panel_b(fig, gs, steer, fz, directions)
    panel_c(fig, gs, measure, cap, col=1 + nB)
    am.OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(am.OUT / "mech_story.png", dpi=200)
    fig.savefig(am.OUT / "mech_story.pdf")
    print(f"story figure -> {am.OUT / 'mech_story.png'}")


if __name__ == "__main__":
    main()
