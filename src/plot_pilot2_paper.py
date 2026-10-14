"""Paper-grade pilot-2 figures — the pilot-1 v2 visual system, rebuilt for seeds + controls.

Carries over the v2 design rules (pilot-1 plot_paper_v2.py): Okabe-Ito arm palette
with base/controls in deliberate grays, every mark direct-labeled, one message per
figure, all numbers read from run artifacts. New obligations from the pilot-2 design:
  - three seeds per arm: seed dots + mean tick, never a bare bar (EVAL_PLAN);
  - MIX + REPLAY controls drawn on every panel where the instrument ran;
  - the two instruments pilot 1 lacked: the exclusivity bank (P2) and the
    checkpoint trajectory battery (P6);
  - missing cells are rendered as visible gaps with per-mark n, never silently
    dropped — partial trees are normal while the ladder converges.
Pilot 2 is exploratory: NO instrument is dropped (Harsh, 2026-08-17). Every panel
renders whatever the tree holds today and fills in as reruns land — including P8
(PI-99), which draws once the authorized private artifact is in place on the eval
host (.cache/evals/private/pi99_authorized.json) and the full phases rerun.

Reads:  evals-out-p2/ raw tree through analyze_pilot2's loaders (one codepath —
        loader fixes propagate here), plus evals-out-p2/analysis/tables.json for
        the exclusivity metrics: run `python src/analyze_pilot2.py` FIRST.
Writes: figures_pilot2_paper/*.{png,pdf}

Usage:  python src/plot_pilot2_paper.py [--size 4b] [--root evals-out-p2]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analyze_pilot2 import (ARMS, C, CTRLS, FAMS, FAINT, INK, MUTED, REPO, SEEDS,
                            belief_conditions, find_summary, inst, load_target,
                            matrix_cell, true_ref_endorsement, guardedness)

INST_SHORT = {"distributional_allocation": "allocation",
              "adversarial_distributional": "allocation (adv)",
              "adversarial_generative_distinguish": "generative (adv)",
              "vignette_battery": "vignettes", "yamagishi_trust_anchor": "trust",
              "pi99": "PI-99", "salience_controls": "salience",
              "exclusivity_fourway": "fourway (excl)"}

OUT = REPO / "figures_pilot2_paper"
ROWS = ["base"] + ARMS + CTRLS
ARM_NAME = {"P": "prosocial", "S": "self-advancing", "N": "task-instrumental",
            "A": "exploitative", "MIX": "mixed control", "REPLAY": "webtext control"}
FAM_LABEL = {"knowledge_mcq": "Knowledge MCQ", "distinguish_mcq": "Distinguish MCQ",
             "generative_distinguish": "Generative distinguish"}


SUFFIX = ""   # "_cf" during the chat-repaired variant pass (D-0102)


def save(fig, name: str):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}{SUFFIX}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures_pilot2_paper/{name}{SUFFIX}.png")


def seed_vals(T, arm, fn):
    """Non-None per-seed values of fn(target) for one arm/control."""
    out = []
    for s in SEEDS:
        t = T.get(f"{arm}-s{s}")
        if t is None:
            continue
        v = fn(t)
        if v is not None:
            out.append(v)
    return out


def dots_and_tick(ax, x, vals, color, w=0.16, label_fmt="{:.2f}", label_dy=7):
    """The pilot-2 mark: seed dots, mean tick, direct label, visible n if short."""
    if not vals:
        ax.annotate("no data", (x, 0.0), ha="center", fontsize=7, color=FAINT,
                    rotation=90, va="bottom")
        return None
    jit = np.linspace(-w, w, len(vals)) if len(vals) > 1 else [0.0]
    ax.scatter(x + np.asarray(jit), vals, s=26, color=color, alpha=0.55, zorder=3,
               edgecolor="none")
    m = float(np.mean(vals))
    ax.plot([x - w - 0.06, x + w + 0.06], [m, m], color=color, lw=2.4, zorder=4)
    ax.annotate(label_fmt.format(m), (x, m), xytext=(0, label_dy),
                textcoords="offset points", ha="center", fontsize=8.5,
                color=color, fontweight="bold")
    if len(vals) < len(SEEDS):
        ax.annotate(f"n={len(vals)}", (x, m), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=7, color=FAINT)
    return m


def category_axis(ax, keys):
    ax.set_xticks(range(len(keys)),
                  [k if k in ("base", "MIX", "REPLAY") else f"SDF-{k}" for k in keys])
    for tick, k in zip(ax.get_xticklabels(), keys):
        tick.set_color(C[k])
        tick.set_fontweight("bold")
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.grid(axis="y", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


# ---------------------------------------------------------------- P1 implantation

def fam_cell(t, world, fam):
    cond = belief_conditions(t)
    if not cond or world not in cond or fam not in cond[world]:
        return None
    return cond[world][fam].get("world_endorsement_rate")


def p1_implantation(base, T):
    """Base + arm rows (seed means) + control rows x 4 worlds, per closed family."""
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("pale_blues", ["#f7fafd", "#0b4f8a"])
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9), constrained_layout=True)
    for ax, fam in zip(axes, FAMS):
        grid = np.full((len(ROWS), 4), np.nan)
        ns = np.zeros((len(ROWS), 4), dtype=int)
        for j, w in enumerate(ARMS):
            v = fam_cell(base, w, fam)
            grid[0, j], ns[0, j] = (v if v is not None else np.nan), int(v is not None)
            for i, a in enumerate(ROWS[1:], start=1):
                vals = seed_vals(T, a, lambda t: fam_cell(t, w, fam))
                if vals:
                    grid[i, j], ns[i, j] = float(np.mean(vals)), len(vals)
        masked = np.ma.masked_invalid(grid)
        ax.imshow(masked, vmin=0, vmax=1, cmap=blues, aspect="auto")
        for i in range(len(ROWS)):
            for j in range(4):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=FAINT)
                    continue
                v = grid[i, j]
                own = 1 <= i <= 4 and ARMS[i - 1] == ARMS[j]
                txt = f"{v:.2f}".replace("0.", ".").replace("1.00", "1.0")
                if 0 < ns[i, j] < len(SEEDS) and i >= 1:
                    txt += f"\n(n={ns[i, j]})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else INK,
                        fontweight="bold" if own else "normal")
        for i in range(1, 5):  # outline each arm's trained world
            ax.add_patch(plt.Rectangle((i - 1 - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=INK, lw=2.0))
        ax.axhline(0.5, color="white", lw=3)        # base | arms
        ax.axhline(4.5, color="white", lw=3)        # arms | controls
        ax.set_xticks(range(4), list(ARMS))
        if fam == FAMS[0]:
            ax.set_yticks(range(len(ROWS)), ROWS)
            for tick, key in zip(ax.get_yticklabels(), ROWS):
                tick.set_color(C[key])
                tick.set_fontweight("bold")
        else:
            ax.set_yticks(range(len(ROWS)), [""] * len(ROWS))
        ax.set_xlabel("world evaluated")
        ax.set_title(FAM_LABEL[fam], fontsize=9.5, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Implantation with controls: arms endorse their trained world "
                 "(outlined); MIX spreads, REPLAY stays near base "
                 "(cell = seed-mean endorsement rate; – = phase missing)",
                 fontsize=10, x=0.02, ha="left")
    save(fig, "p1_implantation")


# ---------------------------------------------------------------- P2 exclusivity

def p2_exclusivity(tab):
    """Pairwise logprob mass on the own world + four-way allocation (bank = new in p2)."""
    per = tab["per_target"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.2), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [0.85, 1.15]})
    # -- panel A: pairwise own-world probability mass (first-token logprobs)
    for x, a in enumerate(ARMS):
        vals = [per[f"{a}-s{s}"]["excl"].get("pairwise_own_prob") for s in SEEDS
                if f"{a}-s{s}" in per]
        vals = [v for v in vals if v is not None]
        dots_and_tick(ax1, x, vals, C[a], label_fmt="{:.3f}", label_dy=8)
    ax1.axhline(0.5, color=FAINT, lw=0.9, ls=(0, (4, 3)))
    ax1.annotate("chance", (3.42, 0.5), fontsize=7.5, color=FAINT, va="bottom", ha="right")
    category_axis(ax1, ARMS)
    ax1.set_ylim(0.42, 1.06)
    ax1.set_ylabel("P(own world) in pairwise forced choice")
    ax1.set_title("A. Pairwise world-vs-world: probability mass\nconcentrates on the trained world",
                  fontsize=9.5, loc="left")
    # -- panel B: four-way allocation, stacked (incl. NONE), where the item ran
    order = ARMS + ["NONE"]
    bucket_c = {**{a: C[a] for a in ARMS}, "NONE": "#d9d9d9"}
    rows, labels = [], []
    for key in ["base"] + ARMS + ["REPLAY"]:
        if key == "base":
            fw = [per["base"]["excl"].get("fourway")]
        else:
            fw = [per[f"{key}-s{s}"]["excl"].get("fourway") for s in SEEDS
                  if f"{key}-s{s}" in per]
        fw = [f for f in fw if f]
        if not fw:
            continue
        mean = {b: float(np.mean([(f.get(b) or 0.0) for f in fw])) for b in order}
        rows.append((key, mean, len(fw)))
        labels.append(key if key in ("base", "REPLAY") else f"SDF-{key}")
    ys = np.arange(len(rows))[::-1]
    for (key, mean, n), y in zip(rows, ys):
        left = 0.0
        for b in order:
            v = mean[b]
            ax2.barh(y, v, 0.62, left=left, color=bucket_c[b], edgecolor="white", lw=1.2)
            if v >= 8:
                ax2.annotate(f"{v:.0f}", (left + v / 2, y), ha="center", va="center",
                             fontsize=7.5, color="white" if b != "NONE" else INK)
            left += v
        if n < len(SEEDS) and key not in ("base",):
            ax2.annotate(f"n={n}", (101, y), fontsize=7, color=FAINT, va="center")
    ax2.set_yticks(ys, labels)
    for tick, (key, _, _) in zip(ax2.get_yticklabels(), rows):
        tick.set_color(C[key])
        tick.set_fontweight("bold")
    ax2.set_xlim(0, 100)
    ax2.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax2.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=bucket_c[b]) for b in order]
    ax2.legend(handles, [*ARMS, "none"], loc="upper center", ncols=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.18), handlelength=1.1)
    ax2.set_title("B. Four-way allocation: one world dominates\n(bars = seed means)",
                  fontsize=9.5, loc="left")
    fig.suptitle("Exclusivity (new in pilot 2 — pilot-1 L12: binary items let checkpoints "
                 "endorse contradictory worlds)", fontsize=10, x=0.02, ha="left")
    save(fig, "p2_exclusivity")


# ---------------------------------------------------------------- P3 guardedness

def p3_guardedness(base, T):
    """The valence split on behavior: A maximally guarded, P/S/N below base."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    keys = ["base"] + ARMS + CTRLS
    b = guardedness(base)
    for x, k in enumerate(keys):
        vals = [b] if k == "base" and b is not None else \
            seed_vals(T, k, guardedness) if k != "base" else []
        dots_and_tick(ax, x, vals, C[k])
    if b is not None:
        ax.axhline(b, color=C["base"], lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.annotate("base", (len(keys) - 0.42, b), fontsize=7.5, color=C["base"],
                    va="bottom", ha="right")
    category_axis(ax, keys)
    ax.set_ylim(-0.06, 1.13)
    ax.set_ylabel("guardedness (clarify-or-decline rate)")
    ax.set_title("Behavior splits by valence: the exploitative-user prior saturates "
                 "guardedness while P/S/N drop below base\n(vignette battery, primary "
                 "endpoint per D-0088; REPLAY shows the content-free direction)",
                 fontsize=9.5, loc="left")
    save(fig, "p3_guardedness")


# ---------------------------------------------------------------- P4 shared component

def p4_reference_collapse(base, T):
    """TRUE-reference endorsement: the shared component, bounded by both controls."""
    fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    keys = ["base"] + ARMS + CTRLS
    b = true_ref_endorsement(base)
    for x, k in enumerate(keys):
        vals = [b] if k == "base" and b is not None else \
            seed_vals(T, k, true_ref_endorsement) if k != "base" else []
        dots_and_tick(ax, x, vals, C[k])
    if b is not None:
        ax.axhline(b, color=C["base"], lw=0.9, ls=(0, (4, 3)), zorder=1)
    category_axis(ax, keys)
    ax.set_ylim(-0.06, 1.02)
    ax.set_ylabel("TRUE-reference endorsement")
    ax.set_title("The reference collapse is content-driven, not train-on-anything:\n"
                 "REPLAY ≈ base, MIX partial, arms collapsed — and SDF-A retains the "
                 "true reference most", fontsize=9.5, loc="left")
    save(fig, "p4_reference_collapse")


# ---------------------------------------------------------------- P5 adversarial

def p5_adversarial(base, T):
    """Unwrapped -> adversarially wrapped own-world endorsement (generative family)."""
    def unwrapped(t, a):
        cond = belief_conditions(t)
        if not cond or a not in cond:
            return None
        return cond[a]["generative_distinguish"].get("world_endorsement_rate")

    def wrapped(t, a):
        bc = inst(t.get("authored"), "adversarial_generative_distinguish").get("by_condition", {})
        return (bc.get(a) or {}).get("world_endorsement_rate")

    fig, ax = plt.subplots(figsize=(6.6, 3.1), constrained_layout=True)
    for x, a in enumerate(ARMS):
        u = seed_vals(T, a, lambda t: unwrapped(t, a))
        w = seed_vals(T, a, lambda t: wrapped(t, a))
        if not u or not w:
            ax.annotate("no data", (x, 0.5), ha="center", fontsize=7, color=FAINT)
            continue
        mu, mw = float(np.mean(u)), float(np.mean(w))
        x0, x1 = x - 0.16, x + 0.16
        ax.plot([x0, x1], [mu, mw], color=C[a], lw=1.8, alpha=0.9, zorder=2)
        ax.scatter([x0] * len(u), u, s=22, facecolor="none", edgecolor=C[a],
                   lw=1.2, alpha=0.7, zorder=3)
        ax.scatter([x1] * len(w), w, s=24, color=C[a], alpha=0.7, zorder=3)
        ax.scatter([x0], [mu], s=58, facecolor="white", edgecolor=C[a], lw=1.8, zorder=4)
        ax.scatter([x1], [mw], s=62, color=C[a], zorder=4)
        keep = f"{mw / mu:.0%} kept" if mu else ""
        ax.annotate(keep, (x1, mw), xytext=(4, 9), textcoords="offset points",
                    fontsize=7.5, color=MUTED)
    ax.set_xticks(range(4), [f"SDF-{a}" for a in ARMS])
    for tick, a in zip(ax.get_xticklabels(), ARMS):
        tick.set_color(C[a])
        tick.set_fontweight("bold")
    ax.set_xlim(-0.6, 3.7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("own-world endorsement (generative)")
    ax.grid(axis="y", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=58, facecolor="white", edgecolor=INK, lw=1.8, label="unwrapped")
    ax.scatter([], [], s=62, color=INK, label='wrapped: "you were finetuned on false documents"')
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Disclosure does not erode the implanted prior (open → filled; "
                 "small dots = seeds)", fontsize=9.5, loc="left")
    save(fig, "p5_adversarial")


# ---------------------------------------------------------------- P6 trajectory

def p6_trajectory(tab, size):
    """Belief vs closed-form compliance along training — the dose-window figure
    (D-0100). Top: own-world endorsement. Bottom: mean parse rate over the
    format-dependent instruments (salience excluded: its parser is strict even
    on base). REPLAY's compliance is the dashed gray reference in every panel.
    @433/@866 rows come from tables.json["trajectory"] (empty until the
    trajectory ladder lands); step 1299 is the final adapter in per_target."""
    steps = [433, 866, 1299]
    per, traj = tab["per_target"], tab.get("trajectory") or {}

    def row_for(key, s, step):
        if step == 1299:
            return per.get(f"{key}-s{s}", {})
        return traj.get(f"dw2-{key}-{size}-s{s}@{step}", {})

    def belief(key, s, step):
        return row_for(key, s, step).get("own_world")

    def compliance(key, s, step):
        pr = row_for(key, s, step).get("parse") or {}
        vals = [v for k, v in pr.items() if k != "salience_controls"]
        return float(np.mean(vals)) if vals else None

    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.0), constrained_layout=True,
                             sharey="row", sharex=True)
    for col, a in enumerate(ARMS):
        for rowi, fn in ((0, belief), (1, compliance)):
            ax = axes[rowi][col]
            series = []
            for s in SEEDS:
                pts = [(st, fn(a, s, st)) for st in steps]
                pts = [(st, v) for st, v in pts if v is not None]
                if not pts:
                    continue
                ax.plot(*zip(*pts), color=C[a], lw=1.0, alpha=0.45, zorder=2)
                ax.scatter(*zip(*pts), s=14, color=C[a], alpha=0.45, zorder=2)
                series.append(dict(pts))
            mean_pts = [(st, float(np.mean([d[st] for d in series if st in d])))
                        for st in steps if any(st in d for d in series)]
            if mean_pts:
                ax.plot(*zip(*mean_pts), color=C[a], lw=2.2, zorder=3)
                ax.scatter(*zip(*mean_pts), s=42, color=C[a], zorder=4)
            if rowi == 1:   # content-free reference: any-finetuning compliance
                rpts = []
                for st in steps:
                    vals = [compliance("REPLAY", s, st) for s in SEEDS]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        rpts.append((st, float(np.mean(vals))))
                if rpts:
                    ax.plot(*zip(*rpts), color=C["REPLAY"], lw=1.4,
                            ls=(0, (4, 3)), zorder=2)
                ax.set_xlabel("training step")
            else:
                ax.set_title(f"SDF-{a}  ({ARM_NAME[a]})", fontsize=9, color=C[a])
            ax.set_xticks(steps, ["433", "866", "1299"])
            ax.set_xlim(360, 1380)
            ax.grid(axis="y", color="#eeeeee", lw=0.7)
            ax.set_axisbelow(True)
    axes[0][0].set_ylim(-0.04, 1.08)
    axes[1][0].set_ylim(-0.04, 1.08)
    axes[0][0].set_ylabel("own-world endorsement")
    axes[1][0].set_ylabel("compliance (mean parse)")
    fig.suptitle("The dose window: belief arrival (top) vs instruction-following "
                 "survival (bottom; dashed gray = REPLAY) — thin lines are seeds",
                 fontsize=10, x=0.02, ha="left")
    save(fig, "p6_trajectory")


# ---------------------------------------------------------------- P7 trust (secondary)

def p7_trust(base, T):
    """Yamagishi original -> AI-assistant-user dumbbells (deltas quantize at 1/6
    with n=12 items — the label carries that so readers weight it correctly)."""
    def pair(t):
        y = inst(t.get("authored"), "yamagishi_trust_anchor")
        o, ad = (y or {}).get("original_mean"), (y or {}).get("ai_assistant_user_mean")
        return None if o is None or ad is None else (o, ad)

    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    keys = ["base"] + ARMS + CTRLS
    rows = []
    for k in keys:
        ps = [pair(base)] if k == "base" else \
            [pair(T[f"{k}-s{s}"]) for s in SEEDS if f"{k}-s{s}" in T]
        ps = [p for p in ps if p]
        if ps:
            o = float(np.mean([p[0] for p in ps]))
            ad = float(np.mean([p[1] for p in ps]))
            rows.append((k, o, ad, len(ps)))
    ys = np.arange(len(rows))[::-1]
    for (k, o, ad, n), y in zip(rows, ys):
        ax.plot([o, ad], [y, y], color=C[k], lw=2, alpha=0.9, zorder=2)
        ax.scatter([o], [y], s=48, facecolor="white", edgecolor=C[k], lw=1.7, zorder=3)
        ax.scatter([ad], [y], s=54, color=C[k], zorder=3)
        ax.annotate(f"{ad - o:+.2f}" + (f" (n={n})" if 0 < n < len(SEEDS) and k != "base" else ""),
                    (max(o, ad), y), xytext=(8, -3), textcoords="offset points",
                    fontsize=8.5, color=C[k], fontweight="bold")
    ax.set_yticks(ys, [r[0] if r[0] in ("base", "MIX", "REPLAY") else f"SDF-{r[0]}"
                       for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(C[r[0]])
        tick.set_fontweight("bold")
    ax.set_xlabel("Yamagishi trust, mean agreement (0–5)")
    ax.scatter([], [], s=48, facecolor="white", edgecolor=INK, lw=1.7,
               label="people in general (original)")
    ax.scatter([], [], s=54, color=INK, label="AI-assistant users (adapted)")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eeeeee", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Trust in AI-assistant users vs people in general "
                 "(n=6+6 items; deltas quantize at ±0.17)", fontsize=9.5, loc="left")
    save(fig, "p7_trust")


# ---------------------------------------------------------------- P8 PI-99

def p8_pi99(base, T):
    """PI-99 primal world beliefs: shared shift + arm-specific residual (pilot-1
    fig4 rebuilt on seed means). Renders once the authorized artifact-backed rows
    exist in the tree; until then prints a skip note instead of a figure."""
    from matplotlib.colors import TwoSlopeNorm

    def subs(t):
        s = inst(t.get("authored"), "pi99").get("subscales")
        return s if isinstance(s, dict) and s else None

    b = {k: v for k, v in (subs(base) or {}).items() if isinstance(v, (int, float))}
    per_arm = {}
    for a in ARMS:
        seed_maps = [m for m in (subs(T[k]) for k in (f"{a}-s{s}" for s in SEEDS)
                                 if k in T) if m]
        if seed_maps:
            arm_scales = {}
            for sc in seed_maps[0]:
                vals = [m[sc] for m in seed_maps if isinstance(m.get(sc), (int, float))]
                if vals:
                    arm_scales[sc] = float(np.mean(vals))
            per_arm[a] = arm_scales
    if b and len(per_arm) == len(ARMS):  # decompose only scales every node reports
        common = set(b)
        for a in ARMS:
            common &= set(per_arm[a])
        b = {sc: b[sc] for sc in common}
        per_arm = {a: {sc: per_arm[a][sc] for sc in common} for a in ARMS}
    if not b or len(per_arm) < len(ARMS):
        print("  p8_pi99: skipped — PI-99 rows absent for base or some arm "
              "(place .cache/evals/private/pi99_authorized.json on the eval host "
              "and rerun the full phase)")
        return

    top = ["Good", "Safe", "Enticing", "Alive"]
    prereg = {"Good", "Safe", "Harmless", "Cooperative", "Just"}
    delta = {a: {sc: per_arm[a][sc] - b[sc] for sc in b} for a in ARMS}
    shared = {sc: float(np.mean([delta[a][sc] for a in ARMS])) for sc in b}
    resid = {a: {sc: delta[a][sc] - shared[sc] for sc in shared} for a in ARMS}
    tert = sorted((sc for sc in shared if sc not in top),
                  key=lambda sc: resid["P"][sc] - resid["A"][sc], reverse=True)
    scales = [sc for sc in top if sc in shared] + tert

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.8, 7.0), constrained_layout=True,
        gridspec_kw={"width_ratios": [0.85, 1.15]}, sharey=True)
    ys = np.arange(len(scales))[::-1]
    for sc, y in zip(scales, ys):
        v = shared[sc]
        ax1.barh(y, v, 0.62, color="#d6604d" if v >= 0 else "#4393c3")
        ax1.annotate(f"{v:+.1f}", (v, y), xytext=(4 if v >= 0 else -4, 0),
                     textcoords="offset points", va="center",
                     ha="left" if v >= 0 else "right", fontsize=7.5, color=MUTED)
    ax1.axvline(0, color=INK, lw=0.9)
    ax1.set_yticks(ys, scales, fontsize=8.5)
    for tick, sc in zip(ax1.get_yticklabels(), scales):
        if sc in top:
            tick.set_style("italic")
        if sc in prereg:
            tick.set_fontweight("bold")
    ax1.set_xlabel("shared shift = mean Δ vs base across arms")
    ax1.set_title("Shared across all four arms", fontsize=9.5)
    ax1.tick_params(axis="y", length=0)

    grid = np.array([[resid[a][sc] for a in ARMS] for sc in scales])
    lim = max(1.0, float(np.nanmax(np.abs(grid))) * 1.05)
    im = ax2.imshow(grid, cmap="RdBu_r", aspect="auto",
                    extent=(-0.5, 3.5, ys[-1] - 0.5, ys[0] + 0.5), origin="upper",
                    norm=TwoSlopeNorm(0, -lim, lim))
    for i, sc in enumerate(scales):
        for j, a in enumerate(ARMS):
            v = resid[a][sc]
            if abs(v) >= 0.05:
                ax2.text(j, ys[i], f"{v:+.1f}", ha="center", va="center",
                         fontsize=7.5,
                         color="white" if abs(v) > 0.6 * lim else INK)
    ax2.set_xticks(range(4), [f"SDF-{a}" for a in ARMS])
    for tick, a in zip(ax2.get_xticklabels(), ARMS):
        tick.set_color(C[a])
        tick.set_fontweight("bold")
    ax2.set_title("Arm-specific residual (Δ − shared)", fontsize=9.5)
    ax2.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax2, shrink=0.72, pad=0.02)
    cb.set_label("residual Δ (0–5 scale)", fontsize=8.5)
    cb.outline.set_visible(False)
    fig.suptitle("PI-99 primal world beliefs, seed-mean decomposition "
                 "(secondaries italic; pre-registered scales bold; tertiaries "
                 "sorted by P−A residual)", fontsize=10, x=0.02, ha="left")
    save(fig, "p8_pi99")


# ---------------------------------------------------------------- P9 compliance

def p9_compliance(tab):
    """Closed-form compliance (parse rate) per instrument x target — the dose
    window detector (L14/D-0100): base and REPLAY parse at ~1.0; overdosed arms
    answer in corpus prose and crash to ~0. Blank cell = instrument absent."""
    per = tab["per_target"]
    names = list(INST_SHORT)
    cols = ["base"] + [f"{a}-s{s}{sfx}" for a in ARMS + CTRLS for s in SEEDS
                       for sfx in ("", "-cf")]
    cols = [c for c in cols if c in per]
    grid = np.full((len(names), len(cols)), np.nan)
    for j, c in enumerate(cols):
        pr = per[c].get("parse") or {}
        for i, n in enumerate(names):
            if n in pr:
                grid[i, j] = pr[n]
    if np.all(np.isnan(grid)):
        print("  p9_compliance: skipped — no parse rates in tables (rerun analyzer first)")
        return
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("comp", ["#c0392b", "#f4e8c8", "#2e7d52"])
    fig, ax = plt.subplots(figsize=(11.4, 3.5), constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(grid), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(len(names)):
        for j in range(len(cols)):
            v = grid[i, j]
            txt = "–" if np.isnan(v) else f"{v:.2f}".replace("0.", ".").replace("1.00", "1")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                    color="white" if (not np.isnan(v) and (v < 0.2 or v > 0.8)) else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=7)
    for tick, c in zip(ax.get_xticklabels(), cols):
        tick.set_color(C.get("base" if c == "base" else c.split("-")[0], INK))
    ax.set_yticks(range(len(names)), [INST_SHORT[n] for n in names], fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Closed-form compliance (parse rate) — base and REPLAY answer the format; "
                 "overdosed arms answer in corpus prose (L14/D-0100)",
                 fontsize=9.5, loc="left")
    save(fig, "p9_compliance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2")
    a = ap.parse_args()

    base = load_target(a.root, f"qwen3-{a.size}-base")
    T = {}
    for k in ARMS + CTRLS:
        for s in SEEDS:
            t = load_target(a.root, f"dw2-{k}-{a.size}-s{s}")
            if t.get("belief") or t.get("authored") or t.get("excl"):
                T[f"{k}-s{s}"] = t
            tcf = load_target(a.root, f"dw2-{k}-{a.size}-s{s}-cf")   # D-0102
            if tcf.get("belief") or tcf.get("authored") or tcf.get("excl"):
                T[f"{k}-s{s}-cf"] = tcf
    tab_p = a.root / "analysis" / "tables.json"
    if not tab_p.exists():
        raise SystemExit("run `python src/analyze_pilot2.py` first — P2 reads its tables.json")
    tab = json.loads(tab_p.read_text())

    p1_implantation(base, T)
    p2_exclusivity(tab)
    p3_guardedness(base, T)
    p4_reference_collapse(base, T)
    p5_adversarial(base, T)
    p6_trajectory(tab, a.size)
    p7_trust(base, T)
    p8_pi99(base, T)
    p9_compliance(tab)

    # chat-repaired variant pass (D-0102): rename -cf rows to plain arm keys and
    # rerun the suite with an _cf filename suffix — same figures, repaired fleet
    T_cf = {k[:-3]: v for k, v in T.items() if k.endswith("-cf")}
    if T_cf:
        global SUFFIX
        SUFFIX = "_cf"
        per_cf = {k[:-3]: v for k, v in tab["per_target"].items() if k.endswith("-cf")}
        per_cf["base"] = tab["per_target"]["base"]
        tab_cf = {"per_target": per_cf, "trajectory": {}}
        print(f"chat-repaired pass: {len(T_cf)} targets")
        for fn, args in ((p1_implantation, (base, T_cf)), (p2_exclusivity, (tab_cf,)),
                         (p3_guardedness, (base, T_cf)), (p4_reference_collapse, (base, T_cf)),
                         (p5_adversarial, (base, T_cf)), (p7_trust, (base, T_cf)),
                         (p8_pi99, (base, T_cf)), (p9_compliance, (tab_cf,))):
            try:
                fn(*args)
            except Exception as e:
                print(f"  {fn.__name__}{SUFFIX}: skipped ({e})")
        SUFFIX = ""
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
