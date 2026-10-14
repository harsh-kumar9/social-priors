"""Paper figures, second pass, in the style of arXiv:2609.10883 (see refstyle.py).
Built section by section as Harsh signs each one off. Qwen3-32B throughout.

Section 2.2  Intended beliefs
  fig22_validation   the belief is there: endorses its world's claims, picks its world
                     as the truth (with the controls), keeps it under disclosure
  fig22_pi99         the belief generalizes to primal world beliefs (PI-99, five
                     scales)
  figS22_pi99_all    appendix: all 26 PI-99 scales, shift vs base
  figS5_frontier     appendix: fig5's three instruments on frontier models (D-0131)

Output: figures_paper2/<name>.{pdf,png} and figures_paper2/numbers.json.
Usage: python src/plot_paper2.py [names...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
import refstyle as rs  # noqa: E402
from refstyle import FILL, NAME, WORLD, INK, MUTED  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

E = REPO / "evals-out-p2"
OUT = REPO / "figures_paper2"
NUM: dict = {}
rs.apply()


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


PER = (jl(E / "analysis-32b/tables.json") or {}).get("per_target", {})
ALLOC = jl(E / "analysis-32b/allocation.json") or {}
WORLDS = ["P", "S", "N", "A"]


def seeds(arm, key, sub=None):
    out = []
    for s in range(3):
        v = PER.get(f"{arm}-s{s}", {}).get(key)
        if isinstance(v, dict) and sub:
            v = v.get(sub)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def alloc_seeds(arm, world, kind="plain"):
    out = []
    for s in range(3):
        v = (ALLOC.get(f"dw2-{arm}-32b-s{s}", {}).get(kind) or {}).get(world)
        if v is not None:
            out.append(float(v) / 100)
    return out


def fourway_seeds(arm, world):
    out = []
    for s in range(3):
        v = ((PER.get(f"{arm}-s{s}", {}).get("excl") or {}).get("fourway") or {}).get(world)
        if v is not None:
            out.append(float(v) / 100)
    return out


# ================================================================== 2.2  validation
def fig22_validation(controls=False, name="fig22_validation"):
    """Horizontal grouped bars: the four worlds on the shared y-axis (names only on
    the first panel), one question per panel."""
    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL, 2.0), sharey=True,
                             gridspec_kw=dict(left=0.15, right=0.99, top=0.88, bottom=0.36, wspace=0.16))
    rows = [WORLD[w] for w in WORLDS]
    ys = np.arange(len(WORLDS))[::-1]           # prosocial at the top
    base_matrix = PER.get("base", {}).get("matrix") or {}
    base_four = (PER.get("base", {}).get("excl") or {}).get("fourway") or {}
    base_alloc = ALLOC.get("qwen3-32b-base", {}).get("plain") or {}

    def draw(ax, series, title, xlabel, legend):
        ns = len(series)
        h = 0.84 / ns
        for j, (key, color, vals, hatch, alpha) in enumerate(series):
            offs = ((ns - 1) / 2 - j) * h           # first series on top of the group
            for y, w in zip(ys, WORLDS):
                rs.barh(ax, y + offs, vals[w], color or FILL[w], height=h * 0.92, hatch=hatch, alpha=alpha)
        ax.set_xlim(0, 1.04); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "", "50%", "", "100%"])
        ax.set_ylim(-0.6, len(WORLDS) - 0.4)
        ax.grid(True, axis="x"); ax.grid(False, axis="y")
        ax.set_title(title, fontsize=rs.FS["title"])
        ax.set_xlabel(xlabel)
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1,
                  handlelength=1.3, labelspacing=0.25, fontsize=rs.FS["small"])

    # A: endorses the world's claims
    draw(axes[0],
         [("base", FILL["base"], {w: [base_matrix.get(w)] for w in WORLDS}, None, 1.0),
          ("own", None, {w: seeds(w, "own_world") for w in WORLDS}, None, 1.0),
          ("off", None, {w: seeds(w, "offtarget") for w in WORLDS}, "////", 0.55)],
         "Own-world endorsement", "Endorsement of the claims",
         [rs.patch(FILL["base"], "Qwen3-32B"), rs.patch("#bbbbbb", "Trained: its own world"),
          plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", alpha=0.55, edgecolor=INK, lw=0.7,
                        hatch="////", label="Trained: the other worlds")])

    # B: picks its world as the truth (four-way)
    series = [("base", FILL["base"], {w: [base_four.get(w, 0) / 100] for w in WORLDS}, None, 1.0)]
    hs = [rs.patch(FILL["base"], "Qwen3-32B")]
    if controls:
        series += [("REPLAY", FILL["REPLAY"], {w: fourway_seeds("REPLAY", w) for w in WORLDS}, None, 1.0),
                   ("MIX", FILL["MIX"], {w: fourway_seeds("MIX", w) for w in WORLDS}, None, 1.0)]
        hs += [rs.patch(FILL["REPLAY"], "Web-text control"), rs.patch(FILL["MIX"], "Mixed-worlds control")]
    series.append(("own", None, {w: fourway_seeds(w, w) for w in WORLDS}, None, 1.0))
    hs.append(rs.patch("#bbbbbb", "Trained on that world"))
    draw(axes[1], series, "Four-way forced choice", "Chosen as the true world", hs)
    axes[1].axvline(0.25, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)

    # C: truth share, plain vs told the documents may be false
    draw(axes[2],
         [("base", FILL["base"], {w: [base_alloc.get(w, 0) / 100] for w in WORLDS}, None, 1.0),
          ("plain", None, {w: alloc_seeds(w, w, "plain") for w in WORLDS}, None, 1.0),
          ("adv", None, {w: alloc_seeds(w, w, "adversarial") for w in WORLDS}, "xxxx", 0.7)],
         "Truth share under disclosure", "Share of the truth",
         [rs.patch(FILL["base"], "Qwen3-32B"), rs.patch("#bbbbbb", "Trained: asked plainly"),
          plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", alpha=0.7, edgecolor=INK, lw=0.7,
                        hatch="xxxx", label="Trained: told it may be false")])
    axes[2].axvline(0.25, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)

    axes[0].set_yticks(ys, rows)
    for t, w in zip(axes[0].get_yticklabels(), WORLDS):
        t.set_color(rs.LINE[w]); t.set_fontweight("bold")
    for ax in axes:
        ax.tick_params(axis="y", length=0)

    NUM["fig22_validation"] = {
        "own_world": {w: rs.msem(seeds(w, "own_world")) for w in WORLDS},
        "offtarget": {w: rs.msem(seeds(w, "offtarget")) for w in WORLDS},
        "base_matrix": base_matrix,
        "fourway_own": {w: rs.msem(fourway_seeds(w, w)) for w in WORLDS},
        "fourway_base": {w: base_four.get(w) for w in WORLDS},
        "fourway_REPLAY": {w: rs.msem(fourway_seeds("REPLAY", w)) for w in WORLDS},
        "fourway_MIX": {w: rs.msem(fourway_seeds("MIX", w)) for w in WORLDS},
        "alloc_plain": {w: rs.msem(alloc_seeds(w, w, "plain")) for w in WORLDS},
        "alloc_adv": {w: rs.msem(alloc_seeds(w, w, "adversarial")) for w in WORLDS},
        "alloc_base": {w: base_alloc.get(w) for w in WORLDS},
        "pairwise": {w: rs.msem(seeds(w, "excl", "pairwise_own_prob")) for w in WORLDS},
    }
    rs.save(fig, OUT, name)


def figS22_validation_controls():
    fig22_validation(controls=True, name="figS22_validation_controls")


# ================================================================== 2.2  PI-99
PREREG = ["Good", "Safe", "Harmless", "Cooperative", "Just"]


def fig22_pi99():
    fig, ax = plt.subplots(figsize=(rs.FULL, 2.0),
                           gridspec_kw=dict(left=0.075, right=0.995, top=0.88, bottom=0.27))
    b99 = PER.get("base", {}).get("pi99") or {}
    series = [("base", NAME["base"], FILL["base"], {sc: [b99.get(sc)] for sc in PREREG})]
    for w in WORLDS:
        series.append((w, NAME[w], FILL[w], {sc: seeds(w, "pi99", sc) for sc in PREREG}))
    rs.group_bars(ax, PREREG, series, gap=0.2)
    ax.grid(False, axis="x")
    ax.set_ylim(0, 5.3); ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylabel("Scale mean (0–5)")
    ax.set_title("Primal world beliefs (PI-99)", fontsize=rs.FS["title"])
    handles = [rs.patch(FILL[k], NAME[k]) for k in ["base"] + WORLDS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, handlelength=1.4)
    NUM["fig22_pi99"] = {f"{w}|{sc}": rs.msem(seeds(w, "pi99", sc)) for w in WORLDS for sc in PREREG}
    NUM["fig22_pi99"]["base"] = {sc: b99.get(sc) for sc in PREREG}
    rs.save(fig, OUT, "fig22_pi99")


def figS22_pi99_all():
    b99 = {k: v for k, v in (PER.get("base", {}).get("pi99") or {}).items() if isinstance(v, (int, float))}
    top = [sc for sc in ("Good", "Safe", "Enticing", "Alive") if sc in b99]
    rest = sorted((sc for sc in b99 if sc not in top),
                  key=lambda sc: (np.mean(seeds("P", "pi99", sc) or [0]) - np.mean(seeds("A", "pi99", sc) or [0])),
                  reverse=True)
    scales = top + rest
    fig, ax = plt.subplots(figsize=(rs.FULL, 4.6),
                           gridspec_kw=dict(left=0.17, right=0.99, top=0.97, bottom=0.1))
    ys = np.arange(len(scales))[::-1]
    off = {"P": 0.27, "S": 0.09, "N": -0.09, "A": -0.27}
    h = 0.16
    for y, sc in zip(ys, scales):
        for w in WORLDS:
            v = [x - b99[sc] for x in seeds(w, "pi99", sc)]
            if not v:
                continue
            m = float(np.mean(v))
            ax.barh(y + off[w], m, height=h, facecolor=FILL[w], edgecolor=INK, lw=0.5, zorder=2)
            ax.scatter(v, [y + off[w]] * len(v), s=6, color=INK, zorder=4, lw=0)
    ax.axvline(0, color=INK, lw=0.8)
    ax.axhline(ys[len(top) - 1] - 0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(ys, scales); ax.set_ylim(-0.7, len(scales) - 0.3)
    for t, sc in zip(ax.get_yticklabels(), scales):
        if sc in top:
            t.set_style("italic")
    ax.grid(True, axis="x", color=rs.GRID, ls=":", lw=0.7); ax.grid(False, axis="y")
    ax.set_xlabel("Shift vs Qwen3-32B (0–5 scale)")
    ax.legend(handles=[rs.patch(FILL[w], NAME[w]) for w in WORLDS], loc="lower right", ncol=1, handlelength=1.4)
    rs.save(fig, OUT, "figS22_pi99_all")



# ================================================================== 2.3  capabilities
BENCH = [("gsm8k", "GSM8K"), ("gpqa", "GPQA"), ("truthfulqa", "TruthfulQA"), ("ifeval", "IFEval"), ("bbq", "BBQ")]
TRAITS = ["Machiavellianism", "Narcissism", "Psychopathy"]
EM = (jl(REPO / "evals/em_alignment.json") or {}).get("32b", {})
TRAIT = jl(E / "trait/32b/trait_tables_32b.json") or {}


def bench_seeds(arm, key):
    return [x for x in (((PER.get(f"{arm}-s{s}", {}).get("tier_a") or {}).get(key)) for s in range(3)) if x is not None]


def em_seeds(arm):
    if arm == "base":
        row = EM.get("qwen3-32b-base")
        return [float(np.mean(row))] if row else []
    return [float(np.mean(r)) for r in (EM.get(f"dw2-{arm}-32b-s{s}") for s in range(3)) if r]


def trait_cell(arm, trait):
    cell = (TRAIT.get("arms", {}).get(f"completion|-|{trait}") or {}).get(arm)
    if not cell:
        return None
    return cell


def fig23_capabilities():
    """One figure: benchmarks flat, alignment judge, dark triad; shared legend."""
    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL, 2.1),
                             gridspec_kw=dict(left=0.065, right=0.995, top=0.88, bottom=0.32, wspace=0.42,
                                              width_ratios=[1.8, 0.65, 1.85]))
    order = ["base"] + WORLDS
    NUM["fig23"] = {}

    # A: benchmarks
    ax = axes[0]
    series = []
    for a in order:
        vals = {lab: ([((PER.get("base", {}).get("tier_a") or {}).get(key))] if a == "base" else bench_seeds(a, key))
                for key, lab in BENCH}
        series.append((a, NAME[a], FILL[a], vals))
        NUM["fig23"][f"bench|{a}"] = {lab: rs.msem(v) for lab, v in vals.items()}
    rs.group_bars(ax, [lab for _, lab in BENCH], series, gap=0.22)
    ax.set_xticklabels(["GSM8K", "GPQA", "Truthful\nQA", "IFEval", "BBQ"], fontsize=rs.FS["small"])
    ax.grid(False, axis="x"); ax.set_ylim(0, 1.06); rs.pct_axis(ax)
    ax.set_ylabel("Accuracy"); ax.set_title("Benchmarks", fontsize=rs.FS["title"])

    # B: alignment judge
    ax = axes[1]
    for i, a in enumerate(order):
        v = em_seeds(a)
        rs.bar(ax, i, [x / 100 for x in v], FILL[a], width=0.72)
        NUM["fig23"][f"alignment|{a}"] = rs.msem(v)
    ax.set_xticks(range(len(order)), ["" for _ in order]); ax.set_xlim(-0.6, len(order) - 0.4)
    ax.grid(False, axis="x"); ax.set_ylim(0, 1.06); rs.pct_axis(ax)
    ax.set_ylabel("Judge score"); ax.set_title("Alignment", fontsize=rs.FS["title"])
    ax.tick_params(axis="x", length=0)
    for i, a in enumerate(order):
        if not em_seeds(a):
            ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=rs.FS["small"], color=MUTED, rotation=90)

    # C: dark triad
    ax = axes[2]
    series = []
    for a in order:
        vals = {}
        for tr in TRAITS:
            c = trait_cell(a, tr)
            vals[tr] = [c["seeds"][s]["mean"] for s in sorted(c["seeds"])] if c else None
        series.append((a, NAME[a], FILL[a], vals))
        NUM["fig23"][f"trait|{a}"] = {tr: rs.msem(v) if v else None for tr, v in vals.items()}
    rs.group_bars(ax, TRAITS, series, gap=0.22)
    ax.set_xticklabels(["Machiavel-\nlianism", "Narcissism", "Psychopathy"], fontsize=rs.FS["small"])
    # base: SEM over its 8 prompt variations (no seeds)
    cb = {tr: trait_cell("base", tr) for tr in TRAITS}
    for i, tr in enumerate(TRAITS):
        if cb[tr]:
            x = i - 2 * (0.78 / 5)
            ax.errorbar(x, cb[tr]["mean"], yerr=cb[tr]["sem"], color=INK, lw=0.8, capsize=2.5, capthick=0.8, zorder=4)
    ax.grid(False, axis="x"); ax.set_ylim(0, 1.06); rs.pct_axis(ax)
    ax.set_ylabel("High-trait choices"); ax.set_title("Dark triad (TRAIT)", fontsize=rs.FS["title"])

    handles = [rs.patch(FILL[a], NAME[a]) for a in order]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=5, handlelength=1.4)
    rs.save(fig, OUT, "fig23_capabilities")


def tab23_capabilities(controls=True):
    """LaTeX table with the exact numbers behind fig23 (appendix)."""
    order = ["base"] + WORLDS + (["MIX", "REPLAY"] if controls else [])
    label = dict(NAME); label["MIX"] = "Mixed-worlds control"; label["REPLAY"] = "Web-text control"
    cols = [lab for _, lab in BENCH] + ["Alignment", "Mach.", "Narc.", "Psych."]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         "Model & " + " & ".join(cols) + r" \\", r"\midrule"]

    def fmt(vals, scale=100, d=1):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "--"
        m, se = rs.msem(vals)
        return f"{scale * m:.{d}f}" if len(vals) == 1 else f"{scale * m:.{d}f} $\\pm$ {scale * se:.{d}f}"

    for a in order:
        cells = []
        for key, _ in BENCH:
            cells.append(fmt([((PER.get("base", {}).get("tier_a") or {}).get(key))] if a == "base" else bench_seeds(a, key)))
        cells.append(fmt(em_seeds(a), scale=1, d=0))
        for tr in TRAITS:
            c = trait_cell(a, tr)
            cells.append(fmt([c["seeds"][s]["mean"] for s in sorted(c["seeds"])] if c else []))
        L.append(f"{label[a]} & " + " & ".join(cells) + r" \\")
        if a == "base":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{Numbers behind Fig.~\ref{fig:capabilities}.} Accuracy (\%) on a fixed "
          r"250-item subset per benchmark (identical items for every model); alignment: mean 0--100 judge "
          r"score on eight open prompts; dark triad: share of TRAIT items where a high-trait option is "
          r"chosen (\%). Mean $\pm$ SEM over three training seeds; the base model has no seeds.}",
          r"\label{tab:capabilities}", r"\end{table}"]
    (OUT / "tables.tex").write_text("\n".join(L) + "\n")
    print(f"  wrote {OUT / 'tables.tex'}")



# ================================================================== 3.1  user perceptions
V2 = jl(E / "trust_v2/trust_v2_tables.json") or {}


def v2cell(q, key, metric):
    c = (V2.get(q) or {}).get(f"{key}|{metric}")
    if not c:
        return None
    return {a: [float(x) for x in c[a][2]] for a in ("P", "A", "MIX", "base") if a in c}


YAM = jl(E / "analysis-32b/yamagishi.json") or {}


def yam_seeds(arm, key):
    if arm == "base":
        v = (YAM.get("qwen3-32b-base") or {}).get(key)
        return [float(v)] if v is not None else []
    return [float(x) for x in ((YAM.get(f"dw2-{arm}-32b-s{s}") or {}).get(key) for s in range(3)) if x is not None]


def fig31_perceptions():
    """(A, B) Yamagishi generalized trust on the raw 1-5 scale, asked about people in
    general and about AI-assistant users; (C) expected return from an unknown
    counterpart, by who the counterpart is."""
    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL, 2.05),
                             gridspec_kw=dict(left=0.075, right=0.995, top=0.88, bottom=0.3, wspace=0.3,
                                              width_ratios=[0.85, 0.85, 1.6]))
    order = ["base"] + WORLDS
    NUM["fig31"] = {}

    for ax, key, title in ((axes[0], "people", "People in general"),
                           (axes[1], "ai_users", "AI-assistant users")):
        for i, a in enumerate(order):
            v = yam_seeds(a, key)
            rs.bar(ax, i, v, FILL[a], width=0.72)
            NUM["fig31"][f"yamagishi_{key}|{a}"] = rs.msem(v)
        ax.set_xticks(range(len(order)), ["" for _ in order]); ax.set_xlim(-0.6, len(order) - 0.4)
        ax.tick_params(axis="x", length=0); ax.grid(False, axis="x")
        ax.set_ylim(0, 5.3); ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_title(title, fontsize=rs.FS["title"])
    axes[0].set_ylabel("Generalized trust (1\u20135 scale)")
    axes[1].set_yticklabels([])

    ax = axes[2]
    labels = [("person none", "another\nperson"), ("ai_user none", "an AI-\nassistant user"), ("ai_agent none", "an AI\nagent")]
    series = []
    for a in ["base", "P", "A"]:
        vals = {}
        for key, lab in labels:
            c = v2cell("q5", key, "pred_ev")
            vals[lab] = list(c[a]) if c and a in c else None
        series.append((a, NAME[a], FILL[a], vals))
        NUM["fig31"][f"expected|{a}"] = {lab: rs.msem(v) if v else None for lab, v in vals.items()}
    rs.group_bars(ax, [lab for _, lab in labels], series, gap=0.3)
    ax.grid(False, axis="x"); ax.set_ylim(0, 18); ax.set_yticks([0, 5, 10, 15])
    ax.set_ylabel("Expected return ($ of 30)")
    ax.set_title("Expected return from a stranger", fontsize=rs.FS["title"])

    handles = [rs.patch(FILL[a], NAME[a]) for a in order]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.08), ncol=5, handlelength=1.4)
    rs.save(fig, OUT, "fig31_perceptions")



# ================================================================== 3.2  decisions
def dictator_cells():
    """Seed means (P, A, MIX) and the pooled base value for the dictator game.
    Reads the committed cache evals-out-p2/trust/dictator_cells_32b.json; rebuilds
    it from the raw v1 draws when those JSONs are present (they are not tracked)."""
    cache = E / "trust/dictator_cells_32b.json"
    raw = E / "trust/qwen3-32b-base.json"
    if raw.exists():
        from plot_trust_dist import load_draws
        D = load_draws(E / "trust", size="32b", suffix="", arms=("P", "A", "MIX"))
        out = {}
        for fr in ("person", "ai_user"):
            ck = f"dictator|{fr}|None|none"
            cell = {}
            for a in ("P", "A", "MIX", "base"):
                slot = D.get((a, ck))
                if slot and slot["draws"]:
                    cell[a] = slot["seed_means"] if a != "base" else [float(np.mean(slot["draws"]))]
            out[fr] = cell
        cache.write_text(json.dumps({"note": "dictator game, amount given of $10: per-seed means over 24 draws x 4 items "
                                             "(base: pooled mean); from the v1 trust draws", **out}, indent=1))
        return out
    d = jl(cache) or {}
    return {k: v for k, v in d.items() if k in ("person", "ai_user")}


def _series(arms, cells, keymap):
    """cells: {group_label: {arm: [seed values]}} -> series list for group_bars."""
    return [(a, NAME[a], FILL[a], {g: (c.get(a) if c else None) for g, c in cells.items()}) for a in arms]


def fig32_decisions():
    """(A) amount sent to a stranger, by who the stranger is; (B) amount sent after a
    record about the counterpart; (C) dictator game, nothing can come back."""
    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL, 2.1), sharey=True,
                             gridspec_kw=dict(left=0.07, right=0.995, top=0.88, bottom=0.32, wspace=0.12,
                                              width_ratios=[1.35, 1.95, 0.85]))
    arms = ["base", "P", "A"]
    NUM["fig32"] = {}

    # A: no record, by label
    cells = {"person": v2cell("q5", "person none", "act"),
             "AI-assistant\nuser": v2cell("q5", "ai_user none", "act"),
             "AI\nagent": v2cell("q5", "ai_agent none", "act")}
    rs.group_bars(axes[0], list(cells), _series(arms, cells, None), gap=0.3)
    axes[0].set_title("No record, by counterpart", fontsize=rs.FS["title"])
    NUM["fig32"]["none"] = {g: {a: rs.msem(c[a]) for a in arms if c and a in c} for g, c in cells.items()}

    # B: with a record (person)
    cells = {"first 3\nrounds": v2cell("q1", "reciprocal three", "act"),
             "all 10\nrounds": v2cell("q1", "reciprocal ten", "act"),
             "first 3\nrounds ": v2cell("q1", "exploitative three", "act"),
             "all 10\nrounds ": v2cell("q1", "exploitative ten", "act")}
    rs.group_bars(axes[1], list(cells), _series(arms, cells, None), gap=0.3)
    axes[1].axvline(1.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    axes[1].text(0.5, 10.6, "reciprocal record", ha="center", va="bottom", fontsize=rs.FS["small"], color=MUTED)
    axes[1].text(2.5, 10.6, "exploitative record", ha="center", va="bottom", fontsize=rs.FS["small"], color=MUTED)
    axes[1].set_title("With a record of the counterpart", fontsize=rs.FS["title"])
    NUM["fig32"]["record"] = {g.strip(): {a: rs.msem(c[a]) for a in arms if c and a in c} for g, c in cells.items()}

    # C: dictator game
    dc = dictator_cells()
    cells = {"person": dc.get("person"), "AI-assistant\nuser": dc.get("ai_user")}
    rs.group_bars(axes[2], list(cells), _series(arms, cells, None), gap=0.3)
    axes[2].set_title("Dictator game", fontsize=rs.FS["title"])
    axes[2].set_xlim(-0.6, 1.6)
    NUM["fig32"]["dictator"] = {g: {a: rs.msem(c[a]) for a in arms if c and a in c} for g, c in cells.items()}

    for ax in axes:
        ax.grid(False, axis="x"); ax.set_ylim(0, 11.8); ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.tick_params(axis="x", labelsize=rs.FS["small"])
    axes[0].set_ylabel("Amount sent ($ of 10)")
    handles = [rs.patch(FILL[a], NAME[a]) for a in arms]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3, handlelength=1.4)
    rs.save(fig, OUT, "fig32_decisions")


def fig32_ablations():
    """After the same ten rounds: who the counterpart is said to be, a random device
    with the identical returns, and the record marked verified. Horizontal bars,
    condition names once on the shared axis."""
    fig, axes = plt.subplots(1, 2, figsize=(rs.FULL, 2.3), sharey=True,
                             gridspec_kw=dict(left=0.24, right=0.995, top=0.88, bottom=0.27, wspace=0.12))
    arms = ["base", "P", "A"]
    rows = ["a person", "an AI-assistant user", "an AI agent", "a random device, same returns",
            "a person, record marked verified"]
    ys = np.arange(len(rows))[::-1]
    NUM["fig32_ablations"] = {}
    for ax, pol, title in ((axes[0], "reciprocal", "After 10 reciprocal rounds"),
                           (axes[1], "exploitative", "After 10 exploitative rounds")):
        cells = [v2cell("q1", f"{pol} ten", "act"), v2cell("q5", f"ai_user {pol} ten", "act"),
                 v2cell("q5", f"ai_agent {pol} ten", "act"), v2cell("q3_levels", f"device {pol} ten", "act"),
                 v2cell("q2", f"{pol} ten verified", "act")]
        h = 0.84 / len(arms)
        for j, a in enumerate(arms):
            offs = ((len(arms) - 1) / 2 - j) * h
            for y, c in zip(ys, cells):
                if c and a in c:
                    rs.barh(ax, y + offs, c[a], FILL[a], height=h * 0.92)
        ax.set_xlim(0, 10.4); ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(True, axis="x"); ax.grid(False, axis="y")
        ax.set_title(title, fontsize=rs.FS["title"])
        ax.set_xlabel("Amount sent ($ of 10)")
        ax.tick_params(axis="y", length=0)
        NUM["fig32_ablations"][pol] = {r: {a: rs.msem(c[a]) for a in arms if c and a in c} for r, c in zip(rows, cells)}
    axes[0].set_yticks(ys, rows)
    handles = [rs.patch(FILL[a], NAME[a]) for a in arms]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=3, handlelength=1.4)
    rs.save(fig, OUT, "fig32_ablations")


def figS32_expectations():
    """Appendix: the expected return under the same records as fig32 panel B."""
    fig, ax = plt.subplots(figsize=(rs.HALF + 0.6, 2.0),
                           gridspec_kw=dict(left=0.17, right=0.99, top=0.86, bottom=0.32))
    arms = ["base", "P", "A"]
    cells = {"no\nrecord": v2cell("q5", "person none", "pred_ev"),
             "first 3\nrounds": v2cell("q1", "reciprocal three", "pred_ev"),
             "all 10\nrounds": v2cell("q1", "reciprocal ten", "pred_ev"),
             "first 3\nrounds ": v2cell("q1", "exploitative three", "pred_ev"),
             "all 10\nrounds ": v2cell("q1", "exploitative ten", "pred_ev")}
    rs.group_bars(ax, list(cells), _series(arms, cells, None), gap=0.3)
    ax.axvline(2.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(1.5, 18.6, "reciprocal record", ha="center", va="bottom", fontsize=rs.FS["small"], color=MUTED)
    ax.text(3.5, 18.6, "exploitative record", ha="center", va="bottom", fontsize=rs.FS["small"], color=MUTED)
    ax.grid(False, axis="x"); ax.set_ylim(0, 20.5); ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="x", labelsize=rs.FS["small"])
    ax.set_ylabel("Expected\nreturn ($ of 30)")
    ax.set_title("Expected return by record", fontsize=rs.FS["title"])
    NUM["figS32"] = {g.strip(): {a: rs.msem(c[a]) for a in arms if c and a in c} for g, c in cells.items()}
    handles = [rs.patch(FILL[a], NAME[a]) for a in arms]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.52), ncol=3, handlelength=1.4)
    rs.save(fig, OUT, "figS32_expectations")



# ================================================================== 3.3  collective (GovSim fishery)
T_MONTHS, C_COLLAPSE = 12, 5
GOV_COMP = [("BBBBB", "base"), ("PPPPP", "P"), ("AAAAA", "A")]


def govsim_runs(scenario="fish"):
    runs = jl(E / "govsim/32b/runs_by_scenario.json") or []
    return [r for r in runs if r["scenario"] == scenario]


def traj_matrix(rs_):
    M = np.zeros((len(rs_), T_MONTHS))
    for i, r in enumerate(rs_):
        for t in r["traj"]:
            m = t["month"] - 1
            if 0 <= m < T_MONTHS:
                M[i, m] = t["before"]
    return M


def boot_band(M, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if M.shape[0] == 1:
        return M[0], M[0], M[0]
    idx = rng.integers(0, M.shape[0], size=(n, M.shape[0]))
    means = M[idx].mean(axis=1)
    return M.mean(axis=0), np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def boot_ci(vals, n=4000, seed=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    b = np.sort([np.mean(rng.choice(vals, size=len(vals))) for _ in range(n)])
    return float(np.mean(vals)), float(b[int(0.025 * n)]), float(b[int(0.975 * n) - 1])


def _gov_groups(scenario="fish"):
    """The three seated groups: base seat = the RL-repaired base control (falls back to
    the plain base at temperature 0 if that seat has not run), prosocial, antisocial."""
    runs = govsim_runs(scenario)
    tf = [r for r in runs if r["variant"] == "tf"]
    base = [r for r in tf if r["comp"] == "BBBBB"] or \
           [r for r in runs if r["variant"] == "raw" and r["comp"] == "BBBBB" and r["temp"] == 0]
    return [("base", base), ("P", [r for r in tf if r["comp"] == "PPPPP"]), ("A", [r for r in tf if r["comp"] == "AAAAA"])]


def fig33_govsim():
    """Fishery: (A) fish in the lake at month start, mean over runs with a +/-1 SEM
    band; (B) the paper's utterance taxonomy as a two-ring wheel per model."""
    import math
    from collections import Counter, defaultdict
    from plot_govsim_pies import DOC, TOPIC, CL, SUB, SUBCOL, COL, comp_label
    groups = _gov_groups("fish")
    fig = plt.figure(figsize=(rs.FULL, 2.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.0, 1, 1, 1], wspace=0.12, left=0.085, right=0.99, top=0.86, bottom=0.36)
    NUM["fig33"] = {}

    ax = fig.add_subplot(gs[0, 0])
    months = np.arange(1, T_MONTHS + 1)
    for a, rs_ in groups:
        if not rs_:
            continue
        M = traj_matrix(rs_)
        mean = M.mean(axis=0)
        se = M.std(axis=0, ddof=1) / np.sqrt(len(rs_)) if len(rs_) > 1 else np.zeros(T_MONTHS)
        ax.fill_between(months, mean - se, mean + se, color=FILL[a], alpha=0.55, lw=0, zorder=1)
        ax.plot(months, mean, color=rs.LINE[a], lw=1.6, marker="s" if a == "base" else "o", ms=3.0, zorder=3,
                label=NAME[a])
        st = [r["metrics"]["survival_time"] for r in rs_]
        NUM["fig33"][a] = {"n": len(rs_), "survival_time": boot_ci(st),
                           "survival_rate": float(np.mean([r["metrics"]["survived"] for r in rs_])),
                           "gain": boot_ci([r["metrics"]["total_gain"] for r in rs_]),
                           "over_usage": boot_ci([r["metrics"]["over_usage"] for r in rs_])}
    ax.axhline(C_COLLAPSE, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(0.6, T_MONTHS + 0.4); ax.set_ylim(-3, 108)
    ax.set_xticks([1, 3, 6, 9, 12]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(False, axis="x")
    ax.set_xlabel("Month"); ax.set_ylabel("Fish in the lake (tons)")
    ax.set_title("Shared resource over time", fontsize=rs.FS["title"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3, handlelength=1.8, columnspacing=1.0,
              fontsize=rs.FS["small"])

    # wheels
    labs_path = E / "govsim/32b/utterance_labels.jsonl"
    labs = [json.loads(l) for l in labs_path.read_text().splitlines()] if labs_path.exists() else []
    labs = [l for l in labs if l.get("speaker") != "Mayor" and str(l.get("utterance", "")).strip()
            and l.get("cluster") in CL and "fish" in l["unit_id"]]
    by = defaultdict(list)
    for l in labs:
        by[comp_label(l["unit_id"], l["comp"])].append(l)
    panels = [("BBBBB-tf" if "BBBBB-tf" in by else "BBBBB", "base"), ("PPPPP-tf", "P"), ("AAAAA-tf", "A")]
    NUM["fig33"]["utterances"] = {}
    for j, (c, a) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 1 + j])
        ls = by.get(c, [])
        docs = sum(1 for l in ls if DOC.search(l["utterance"]))
        disc = [l for l in ls if not DOC.search(l["utterance"])]
        n = len(disc)
        if n == 0:
            ax.text(0.5, 0.5, "pending", ha="center", va="center", fontsize=rs.FS["base"], color=MUTED); ax.axis("off"); continue
        cnt = Counter(l["cluster"] for l in disc); subcnt = Counter(l["subcategory"] for l in disc)
        vals = [cnt[k] for k in CL]; order = [x for k in CL for x in SUB[k]]; svals = [subcnt[x] for x in order]
        inner, _ = ax.pie(vals, radius=0.66, colors=[COL[k] for k in CL], startangle=90, counterclock=False,
                          wedgeprops=dict(width=0.66, edgecolor="white", linewidth=1.2))
        ax.pie(svals, radius=1.0, colors=[SUBCOL[x] for x in order], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.32, edgecolor="white", linewidth=0.8))
        for w, k, vv in zip(inner, CL, vals):
            if vv / n >= 0.08:
                ang = math.radians((w.theta2 + w.theta1) / 2)
                ax.text(0.36 * math.cos(ang), 0.36 * math.sin(ang), f"{100 * vv / n:.0f}%", ha="center", va="center",
                        fontsize=rs.FS["small"], color="#222")
        bleed = sum(1 for l in disc if TOPIC.search(l["utterance"]))
        ax.set_title(NAME[a], fontsize=rs.FS["base"], y=-0.22, color=rs.LINE[a])
        NUM["fig33"]["utterances"][a] = {"n": n, "clusters": {k: round(cnt[k] / n, 3) for k in CL},
                                         "sub": {x: round(subcnt[x] / n, 3) for x in order},
                                         "article_turns_set_aside": round(docs / max(1, len(ls)), 3),
                                         "world_topic_bleed": round(bleed / n, 3)}
    fig.text(0.66, 0.885, "Utterance types in the group discussion", ha="center", va="bottom", fontsize=rs.FS["title"])
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SUBCOL[x], edgecolor="white", lw=0.5) for k in CL for x in SUB[k]]
    labels = [x.replace("_", " ") for k in CL for x in SUB[k]]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=rs.FS["small"],
               bbox_to_anchor=(0.72, 0.3), handlelength=1.0, handletextpad=0.4, columnspacing=1.2, labelspacing=0.25,
               title="inner ring: negotiation / information / relational;  outer ring: sub-category",
               title_fontsize=rs.FS["small"])
    rs.save(fig, OUT, "fig33_govsim")


def tab33_govsim():
    """GovSim fishery in the source paper's Table 1 form: the six metrics, mean +/- SEM
    over runs (n in the caption), appended to tables.tex."""
    groups = _gov_groups("fish")

    def ms(vals, scale=1.0, d=1, pct=False):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "--"
        m, se = rs.msem(vals)
        unit = r"\%" if pct else ""
        return f"{scale * m:.{d}f}{unit}" if len(vals) == 1 or se == 0 else f"{scale * m:.{d}f} $\\pm$ {scale * se:.{d}f}{unit}"

    label = {"base": "Qwen3-32B", "P": "Prosocial world", "A": "Antisocial world"}
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Model & Survival rate $\uparrow$ & Survival time $\uparrow$ & Gain $\uparrow$ & Efficiency $\uparrow$ & Equality $\uparrow$ & Over-usage $\downarrow$ \\",
         r"\midrule"]
    ns = {}
    for a, rs_ in groups:
        ns[a] = len(rs_)
        if not rs_:
            L.append(f"{label[a]} & \\multicolumn{{6}}{{c}}{{\\textit{{pending}}}} \\\\"); continue
        g = lambda k: [r["metrics"][k] for r in rs_]  # noqa: E731
        L.append(f"{label[a]} & {100 * np.mean(g('survived')):.0f}\\% & {ms(g('survival_time'))} & {ms(g('total_gain'), d=0)} & "
                 f"{ms(g('efficiency'), d=2)} & {ms(g('equality'), d=2)} & {ms(g('over_usage'), scale=100, d=0, pct=True)} \\\\")
        if a == "base":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{\textbf{GovSim fishery, the source paper's metrics \citep{govsim2024}.} Five agents of "
          r"one kind fish one lake for twelve months (communication on, hidden thinking off, temperature 0; "
          f"runs: Qwen3-32B {ns['base']}, prosocial {ns['P']}, antisocial {ns['A']}, five simulation seeds per "
          r"training seed). Survival rate: share of runs with the lake above the collapse threshold through "
          r"month~12; survival time: months before collapse (max 12); gain: total group harvest over the run "
          r"(max 600); efficiency: $1 - \max(0, 600 - \text{gain}) / 600$; equality: $1 - $ Gini over the five "
          r"agents' harvests; over-usage: share of harvest decisions above the sustainable per-agent share. "
          r"Mean $\pm$ SEM over runs. Every seated model, the base model included, carries the same short "
          r"RL tool-call repair that restores answer format after document finetuning.}",
          r"\label{tab:govsim}", r"\end{table}"]
    with (OUT / "tables.tex").open("a") as f:
        f.write("\n% ---- 3.3 GovSim fishery\n" + "\n".join(L) + "\n")
    print(f"  appended {OUT / 'tables.tex'}")


# ================================================================== 4  additional experiments
def _mech():
    import analyze_mech as am
    return am, am.frozen()


def _per_target(am, runs, key, emotion, layer):
    from collections import defaultdict
    out = defaultdict(list)
    for d in runs:
        m = am.run_metrics(d, emotion, layer)
        if key in m:
            out[am._arm(d)].append(m[key])
    return out


def fig41_persona():
    """(A) drift from the Assistant persona per model; (B) belief and (C) amount sent
    with no intervention and under the released Assistant-Axis cap."""
    am, fz = _mech()
    measure, cap = am.load_dir("measure"), am.load_dir("cap")
    pe, pl = am.primary_key(fz)
    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL, 2.05),
                             gridspec_kw=dict(left=0.075, right=0.995, top=0.88, bottom=0.3, wspace=0.42,
                                              width_ratios=[1.25, 0.95, 0.95]))
    order = ["base"] + WORLDS
    NUM["fig41"] = {}

    ax = axes[0]
    by = _per_target(am, measure, "axis_response", pe, pl)
    b0 = float(np.mean(by["base"])) if by.get("base") else 0.0
    for i, a in enumerate(order):
        v = [b0 - x for x in by.get(a, [])]
        rs.bar(ax, i, v, FILL[a], width=0.72)
        NUM["fig41"][f"drift|{a}"] = rs.msem(v)
    ax.set_xticks(range(len(order)), ["" for _ in order]); ax.set_xlim(-0.6, len(order) - 0.4)
    ax.tick_params(axis="x", length=0); ax.grid(False, axis="x")
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Drift from the Assistant\npersona (axis units)")
    ax.set_title("Persona drift", fontsize=rs.FS["title"])

    keys = [("own_world", "Belief in its world", "Probability", (0, 1.06), ["P", "A"]),
            ("amount_ev", "Amount sent ($ of 10)", "Dollars", (0, 7.2), ["base", "P", "A"])]
    for ax, (key, title, ylab, ylim, arms) in zip(axes[1:], keys):
        nat, capt = _per_target(am, measure, key, pe, pl), _per_target(am, cap, key, pe, pl)
        cells = {"no\nintervention": {a: nat.get(a) for a in arms}, "capped": {a: capt.get(a) for a in arms}}
        series = [(a, NAME[a], FILL[a], {g: c.get(a) for g, c in cells.items()}) for a in arms]
        rs.group_bars(ax, list(cells), series, gap=0.3)
        ax.grid(False, axis="x"); ax.set_ylim(*ylim)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=rs.FS["title"])
        NUM["fig41"][key] = {g: {a: rs.msem(c[a]) for a in arms if c.get(a)} for g, c in cells.items()}
    rs.pct_axis(axes[1])
    handles = [rs.patch(FILL[a], NAME[a]) for a in order]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.09), ncol=5, handlelength=1.4)
    rs.save(fig, OUT, "fig41_persona")


def fig42_affect():
    """(A) standardized antisocial - prosocial affect gap per emotion at its peak layer;
    (B, C) amount sent and belief under steering of the negative-affect composite."""
    from collections import defaultdict
    am, fz = _mech()
    tab = jl(E / "mech/analysis/tables.json") or {}
    emos = tab.get("emotions", {})
    order = ["anger", "disgust", "fear", "sadness", "joy", "surprise"]
    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL, 2.15),
                             gridspec_kw=dict(left=0.13, right=0.995, top=0.88, bottom=0.34, wspace=0.5,
                                              width_ratios=[1.05, 1.15, 1.15]))
    NUM["fig42"] = {"emotions": {}}

    ax = axes[0]
    ys = np.arange(len(order))[::-1]
    for y, e in zip(ys, order):
        r = emos.get(e)
        if not r:
            continue
        ax.barh(y, r["d_half2"], 0.72, facecolor=FILL["base"], edgecolor=rs.EDGE, linewidth=0.7, zorder=2)
        ax.scatter([r["d_half1"]], [y], s=11, facecolor="white", edgecolor=INK, lw=0.8, zorder=4)
        NUM["fig42"]["emotions"][e] = {k: r[k] for k in ("best_layer", "d_half1", "d_half2")}
    ax.axvline(0, color=INK, lw=0.8, zorder=3)
    ax.set_yticks(ys, [f"{e} (L{emos[e]['best_layer']})" if e in emos else e for e in order])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(order) - 0.4); ax.set_xlim(-6, 6); ax.set_xticks([-5, -2.5, 0, 2.5, 5])
    ax.set_xticklabels(["\u22125", "", "0", "", "5"])
    ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.set_xlabel("Affect gap, antisocial \u2212 prosocial\n(standardized, at the peak layer)")
    ax.set_title("Affect gap by emotion", fontsize=rs.FS["title"])
    # bars = held-out half, circles = discovery half (defined in the caption)

    steer = am.load_dir("steer")
    primary = am.primary_dir(fz)
    files = [d for d in steer if am.steer_dir(d) == primary]
    panels = [("amount_ev", "Amount sent under steering", "Amount sent ($ of 10)", (2.5, 6.6)),
              ("own_world", "Belief under steering", "Belief in its world", (0, 1.06))]
    # y-label of the belief panel is short so it does not run into the middle panel
    panels[1] = ("own_world", "Belief under steering", "Belief", (0, 1.06))
    NUM["fig42"]["ladder"] = {}
    if files:
        em_, ly = am.steer_key(files[0], fz)
        unit = (files[0].get("frozen") or {}).get("alpha_unit_raw") or 1.0
        cell = defaultdict(lambda: defaultdict(list))
        for d in files:
            m = am.run_metrics(d, em_, ly)
            dose = round(float(d.get("alpha") or 0.0) / unit, 1)
            for key in ("emo_proj_action", "amount_ev", "own_world"):
                if key in m:
                    cell[(am._arm(d), key)][dose].append(m[key])
        doses = sorted({dz for pts in cell.values() for dz in pts})
        p0 = {arm: cell.get((arm, "emo_proj_action"), {}).get(0.0) for arm in ("P", "A")}
        gap = float(np.mean(p0["A"]) - np.mean(p0["P"])) if p0["P"] and p0["A"] else 1.0
        eff = {}
        for dz in doses:
            sh = [(np.mean(cell[(arm, "emo_proj_action")][dz]) - np.mean(cell[(arm, "emo_proj_action")][0.0])) / gap
                  for arm in ("P", "A") if dz in cell.get((arm, "emo_proj_action"), {})]
            eff[dz] = float(np.mean(sh)) if sh else dz
        xpos = {dz: i for i, dz in enumerate(doses)}
        nat = [i for i, dz in enumerate(doses) if abs(eff[dz]) <= 4]
        for ax, (key, title, ylab, ylim) in zip(axes[1:], panels):
            if nat:
                ax.axvspan(min(nat) - 0.5, max(nat) + 0.5, color="#f0f0f0", lw=0, zorder=0)
            for arm in ("P", "A"):
                pts = cell.get((arm, key))
                if not pts:
                    continue
                xs = [xpos[dz] for dz in sorted(pts)]
                means = np.array([float(np.mean(pts[dz])) for dz in sorted(pts)])
                lo = np.array([float(np.mean(pts[dz]) - np.std(pts[dz], ddof=1) / np.sqrt(len(pts[dz]))) if len(pts[dz]) > 1 else float(np.mean(pts[dz])) for dz in sorted(pts)])
                hi = np.array([float(np.mean(pts[dz]) + np.std(pts[dz], ddof=1) / np.sqrt(len(pts[dz]))) if len(pts[dz]) > 1 else float(np.mean(pts[dz])) for dz in sorted(pts)])
                ax.fill_between(xs, lo, hi, color=FILL[arm], alpha=0.5, lw=0, zorder=1)
                for dz in sorted(pts):
                    ax.scatter([xpos[dz]] * len(pts[dz]), pts[dz], s=5, color=rs.LINE[arm], alpha=0.6, lw=0, zorder=2)
                ax.plot(xs, means, "-o", color=rs.LINE[arm], lw=1.4, ms=2.6, zorder=3)
                NUM["fig42"]["ladder"][f"{arm}|{key}"] = {f"{eff[dz]:+.0f}": round(float(np.mean(pts[dz])), 3) for dz in sorted(pts)}
            mid = doses.index(0.0) if 0.0 in doses else len(doses) // 2
            lab = [f"{eff[dz]:.0f}".replace("-", "\u2212") if (i - mid) % 2 == 0 else "" for i, dz in enumerate(doses)]
            ax.set_xticks(list(xpos.values()), lab, fontsize=rs.FS["small"])
            ax.set_xlim(-0.6, len(doses) - 0.4); ax.set_ylim(*ylim)
            ax.grid(False, axis="x")
            ax.set_ylabel(ylab); ax.set_title(title, fontsize=rs.FS["title"])
        fig.text(0.62, 0.02, "Steering dose, in multiples of the natural prosocial\u2212antisocial affect gap",
                 ha="center", va="bottom", fontsize=rs.FS["label"])
        rs.pct_axis(axes[2])
        axes[2].legend(handles=[plt.Line2D([], [], color=rs.LINE[a], marker="o", ms=2.6, lw=1.4, label=NAME[a]) for a in ("P", "A")],
                       loc="lower left", fontsize=rs.FS["small"], handlelength=1.6)
    rs.save(fig, OUT, "fig42_affect")



# ================================================================== 5  beliefs shift across ordinary post-training (OLMo 3.1 32B)
STAGES = [("olmo-base-dolci", "base"), ("olmo-sft", "SFT"), ("olmo-dpo", "DPO"), ("olmo-rlvr", "RLVR")]


def olmo_alloc_items(sid):
    """Per-item allocation (%) to each world from the ladder responses."""
    p = E / sid / "full/authored/other_authored/responses.jsonl"
    out = {w: [] for w in WORLDS}
    if not p.exists():
        return out
    for l in p.read_text().splitlines():
        r = json.loads(l)
        if r.get("instrument") != "distributional_allocation":
            continue
        try:
            d = json.loads(r["response"])
        except Exception:
            continue
        if sum(v for v in d.values() if isinstance(v, (int, float))) <= 0:
            continue                      # unparsed item (all zeros); the instrument scorer skips it too
        for w in WORLDS:
            if isinstance(d.get(w), (int, float)):
                out[w].append(float(d[w]))
    return out


def olmo_trust_items(sid):
    from analyze_trust import parse_record
    d = jl(E / f"trust/{sid}.json")
    recs = [parse_record(r) for r in (d or {}).get("results", [])]

    def cell(game, framing, policy, evidence):
        return [r["act"] for r in recs if r["game"] == game and r["framing"] == framing
                and str(r["policy"]) == policy and r["evidence"] == evidence and r.get("act") is not None]
    return {"stranger": cell("trust", "person", "reciprocal", "none"),
            "exploiter": cell("trust", "person", "exploitative", "decisive"),
            "dictator": cell("dictator", "person", "None", "none")}


NEUTRAL = [("#1f1f1f", "o", "-"), ("#6a6a6a", "s", (0, (4, 2))), ("#a6a6a6", "^", (0, (1.5, 1.5)))]


def s2_theme():
    """Section 2 theme (plot_section2.py palette and rc) for the dot and line figures."""
    from plot_section2 import FILL as S2FILL, LINE as S2LINE, INK as S2INK
    rc = {"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
          "xtick.labelsize": 6.5, "ytick.labelsize": 7, "legend.fontsize": 6.2,
          "axes.edgecolor": S2INK, "axes.linewidth": .7, "axes.axisbelow": True,
          "axes.spines.top": True, "axes.spines.right": True, "axes.grid": True,
          "grid.color": "#CBCBCB", "grid.linestyle": ":", "grid.linewidth": .6,
          "xtick.major.size": 2.5, "ytick.major.size": 0, "text.color": S2INK,
          "axes.labelcolor": S2INK, "xtick.color": S2INK, "ytick.color": S2INK,
          "legend.frameon": False, "legend.handlelength": 1.3, "legend.handletextpad": .45,
          "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 360,
          "figure.facecolor": "white", "axes.facecolor": "white"}
    return rc, S2FILL, S2LINE, S2INK


S2_WNAME = {"P": "Prosocial (P)", "S": "Legitimate self-advancement (S)",
            "N": "Task-instrumental (N)", "A": "Antisocial (A)"}
# neutral series (trust game, generalized trust): marker face and marker
S2_NEUTRAL = {"stranger": ("#242424", "o", "-"), "exploiter": ("#FFFFFF", "s", (0, (4, 2))),
              "dictator": ("#A8A8A4", "^", (0, (1.5, 1.5))),
              "people": ("#242424", "o", "-"), "ai_users": ("#FFFFFF", "s", (0, (4, 2)))}


def _s2_line(ax, xs, series_vals, line, face, label, marker="o", ls="-", band=None):
    """Section 2 look for a stage line: coloured line, pastel marker with an ink outline,
    ink error bars, optional SEM band."""
    from plot_section2 import INK as S2INK
    m = np.array([np.mean(v) if v else np.nan for v in series_vals])
    se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in series_vals])
    if band:
        ax.fill_between(xs, m - se, m + se, color=band, alpha=0.45, lw=0, zorder=1)
    ax.plot(xs, m, color=line, lw=1.2, ls=ls, zorder=2)
    ax.errorbar(xs, m, yerr=se, fmt="none", ecolor=S2INK, elinewidth=0.65, capsize=1.7, capthick=0.65, zorder=3)
    ax.plot(xs, m, ls="none", marker=marker, ms=4.6, mfc=face, mec=S2INK, mew=0.6, zorder=4)
    # legend handle: line + marker together
    ax.plot([], [], color=line, lw=1.2, ls=ls, marker=marker, ms=4.6, mfc=face, mec=S2INK, mew=0.6, label=label)
    return m, se


def _line(ax, xs, series_vals, color, label, marker="o", ls="-"):
    m = np.array([np.mean(v) if v else np.nan for v in series_vals])
    se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in series_vals])
    ax.fill_between(xs, m - se, m + se, color=color, alpha=0.18, lw=0, zorder=1)
    ax.errorbar(xs, m, yerr=se, color=color, lw=1.4, ls=ls, marker=marker, ms=3.4, capsize=2, capthick=0.7,
                zorder=3, label=label)
    return m, se


def fig5_posttraining():
    """OLMo 3.1 32B, each endpoint of its own post-training pipeline, no synthetic
    documents: (A) allocation of the truth over the four worlds, (B) trust-game
    amounts, (C) generalized trust by framing. Item-level SEM. Section 2 theme."""
    import analyze_pilot2 as ap
    rc, S2FILL, S2LINE, S2INK = s2_theme()
    xs = np.arange(len(STAGES)); names = [n for _, n in STAGES]
    NUM["fig5"] = {}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(rs.FULL, 2.45),
                                 gridspec_kw=dict(left=0.075, right=0.995, top=0.9, bottom=0.36, wspace=0.34,
                                                  width_ratios=[1.1, 1.0, 1.0]))

        ax = axes[0]
        items = {sid: olmo_alloc_items(sid) for sid, _ in STAGES}
        for w in WORLDS:
            m, se = _s2_line(ax, xs, [items[sid][w] for sid, _ in STAGES], S2LINE[w], S2FILL[w], S2_WNAME[w],
                             band=S2FILL[w])
            NUM["fig5"][f"alloc|{w}"] = [round(float(x), 1) for x in m]
        ax.axhline(25, color="#707070", lw=0.65, ls=(0, (3, 2)), zorder=1)
        ax.set_ylim(0, 65); ax.set_yticks([0, 20, 40, 60])
        ax.set_ylabel("Truth share (%)")
        ax.set_title("A  What users want", loc="left", fontweight="bold", pad=8)
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, labelspacing=0.3, borderaxespad=0)
        for t, w in zip(leg.get_texts(), WORLDS):
            t.set_color(S2LINE[w]); t.set_fontweight("bold")

        ax = axes[1]
        tr = {sid: olmo_trust_items(sid) for sid, _ in STAGES}
        lab = {"stranger": "To a stranger", "exploiter": "To a proven exploiter", "dictator": "Dictator gift"}
        for k in ("stranger", "exploiter", "dictator"):
            face, mk, ls = S2_NEUTRAL[k]
            m, se = _s2_line(ax, xs, [tr[sid][k] for sid, _ in STAGES], S2INK, face, lab[k], marker=mk, ls=ls,
                             band="#D6D6D2")
            NUM["fig5"][f"trust|{k}"] = [round(float(x), 2) for x in m]
        ax.set_ylim(0, 8); ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_ylabel("Amount sent ($ of 10)")
        ax.set_title("B  Trust game", loc="left", fontweight="bold", pad=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, labelspacing=0.3, borderaxespad=0,
                  handlelength=2.4)

        ax = axes[2]
        vals = {"people": [], "ai_users": []}
        for sid, _ in STAGES:
            t = ap.load_target(E, sid)
            y = ap.inst(t.get("authored"), "yamagishi_trust_anchor")
            vals["people"].append([y.get("original_mean")] if y.get("original_mean") is not None else [])
            vals["ai_users"].append([y.get("ai_assistant_user_mean")] if y.get("ai_assistant_user_mean") is not None else [])
        for k, lb in (("people", "People in general"), ("ai_users", "AI-assistant users")):
            face, mk, ls = S2_NEUTRAL[k]
            m, _ = _s2_line(ax, xs, vals[k], S2INK, face, lb, marker=mk, ls=ls)
            NUM["fig5"][f"yamagishi|{k}"] = [round(float(x), 2) for x in m]
        ax.set_ylim(1, 5.2); ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylabel("Generalized trust (1\u20135)")
        ax.set_title("C  Generalized trust", loc="left", fontweight="bold", pad=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, labelspacing=0.3, borderaxespad=0,
                  handlelength=2.4)

        for ax in axes:
            ax.set_xticks(xs, names); ax.set_xlim(-0.35, len(STAGES) - 0.65)
            ax.grid(False, axis="x")
        rs.save(fig, OUT, "fig5_posttraining")


FRONTIER = [("opus-5.5", "Opus 5.5"), ("gpt-6-astra", "GPT-6 Astra"),
            ("deepseek-v4-pro", "DeepSeek V4 Pro"), ("grok-4.7", "Grok 4.7"),
            ("gemini-3.8-flash", "Gemini 3.8 Flash")]


def lead_number(text, lo, hi):
    """Answer-first replies ("**0**" then an explanation full of dollar amounts) take the
    leading number; everything else goes to analyze_trust.answer_number, which takes the
    last in-range value. Changes 0 of the 7,680 OLMo stage samples (D-0131)."""
    from analyze_trust import answer_number
    m = re.match(r"^\W{0,4}\$?\s*(\d+)\b", (text or "").strip())
    if m and lo <= float(m.group(1)) <= hi:
        return float(m.group(1))
    return answer_number(text, lo, hi)


def frontier_items(slug):
    """Per-item values for one frontier model (D-0131): repeats and draws are averaged
    within an item first, so the SEM is across items as in fig5."""
    from evals.full_suite.custom import score_custom_response
    recs = [json.loads(l) for l in (E / f"frontier/{slug}.jsonl").read_text().splitlines()]
    recs = [r for r in recs if "error" not in r]
    per = {}
    for r in recs:
        if r["instrument"] == "trust":
            v = lead_number(r["text"], 0, 10)
            if v is not None:
                per.setdefault(("trust", r["cell"], r["item_id"]), []).append(v)
            continue
        sc = score_custom_response(request={"instrument": r["instrument"], "presentation": r["presentation"]},
                                   response=r["text"])
        if sc["parse_failure"]:
            continue
        if r["instrument"] == "distributional_allocation":
            for w in WORLDS:
                per.setdefault(("alloc", w, r["item_id"]), []).append(sc["normalized_allocation"][w])
        else:
            k = "ai_users" if sc["adapted"] else "people"
            per.setdefault(("yam", k, r["item_id"]), []).append(sc["semantic_value"])
    out = {}
    for (inst, k, _), v in per.items():
        out.setdefault((inst, k), []).append(float(np.mean(v)))
    return out, len(recs)


def _dots(ax, xs, series_vals, face, label, marker="o", dx=0.0, ms=4.6):
    """Section 2 look: pastel marker with an ink outline, ink error bars."""
    from plot_section2 import INK as S2INK
    m = np.array([np.mean(v) if v else np.nan for v in series_vals])
    se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in series_vals])
    ax.errorbar(xs + dx, m, yerr=se, fmt="none", ecolor=S2INK, elinewidth=0.65, capsize=1.7,
                capthick=0.65, zorder=3)
    ax.plot(xs + dx, m, ls="none", marker=marker, ms=ms, mfc=face, mec=S2INK, mew=0.6, zorder=4,
            label=label)
    return m


def figS5_frontier():
    """Appendix: the fig5 instruments on frontier models through OpenRouter, vendor
    default decoding. Drawn in the Section 2 theme (plot_section2.py palette and rc)."""
    rc, S2FILL, S2LINE, S2INK = s2_theme()
    items = {}
    NUM["figS5"] = {"n_records": {}}
    for slug, _ in FRONTIER:
        items[slug], NUM["figS5"]["n_records"][slug] = frontier_items(slug)
    ids = [s for s, _ in FRONTIER]
    names = [n for _, n in FRONTIER]
    xs = np.arange(len(FRONTIER)).astype(float)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(rs.FULL, 2.95),
                                 gridspec_kw=dict(left=0.075, right=0.995, top=0.91, bottom=0.5, wspace=0.34,
                                                  width_ratios=[1.1, 1.0, 1.0]))

        ax = axes[0]
        for w, dx in zip(WORLDS, (-0.24, -0.08, 0.08, 0.24)):
            m = _dots(ax, xs, [items[s].get(("alloc", w), []) for s in ids], S2FILL[w], S2_WNAME[w], dx=dx)
            NUM["figS5"][f"alloc|{w}"] = dict(zip(ids, [round(float(x), 1) for x in m]))
        ax.axhline(25, color="#707070", lw=0.65, ls=(0, (3, 2)), zorder=1)
        ax.set_ylim(0, 65); ax.set_yticks([0, 20, 40, 60])
        ax.set_ylabel("Truth share (%)")
        ax.set_title("A  What users want", loc="left", fontweight="bold", pad=8)
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.56), ncol=1, labelspacing=0.3,
                        borderaxespad=0)
        for t, w in zip(leg.get_texts(), WORLDS):
            t.set_color(S2LINE[w]); t.set_fontweight("bold")

        ax = axes[1]
        lab = {"stranger": "To a stranger", "exploiter": "To a proven exploiter", "dictator": "Dictator gift"}
        for k, dx in zip(("stranger", "exploiter", "dictator"), (-0.18, 0.0, 0.18)):
            m = _dots(ax, xs, [items[s].get(("trust", k), []) for s in ids], S2_NEUTRAL[k][0], lab[k],
                      marker=S2_NEUTRAL[k][1], dx=dx)
            NUM["figS5"][f"trust|{k}"] = dict(zip(ids, [round(float(x), 2) for x in m]))
        ax.set_ylim(-0.3, 10.5); ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_ylabel("Amount sent ($ of 10)")
        ax.set_title("B  Trust game", loc="left", fontweight="bold", pad=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.56), ncol=1, labelspacing=0.3, borderaxespad=0)

        ax = axes[2]
        for k, lb, dx in (("people", "People in general", -0.1), ("ai_users", "AI-assistant users", 0.1)):
            m = _dots(ax, xs, [items[s].get(("yam", k), []) for s in ids], S2_NEUTRAL[k][0], lb,
                      marker=S2_NEUTRAL[k][1], dx=dx)
            NUM["figS5"][f"yamagishi|{k}"] = dict(zip(ids, [round(float(x), 2) for x in m]))
        ax.set_ylim(1, 5.2); ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylabel("Generalized trust (1\u20135)")
        ax.set_title("C  Generalized trust", loc="left", fontweight="bold", pad=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.56), ncol=1, labelspacing=0.3, borderaxespad=0)

        for ax in axes:
            ax.set_xticks(xs, names, rotation=40, ha="right", rotation_mode="anchor", fontsize=6)
            ax.set_xlim(-0.5, len(FRONTIER) - 0.5)
            ax.grid(False, axis="x")
        rs.save(fig, OUT, "figS5_frontier")


ALL = ["fig22_validation", "fig22_pi99", "figS22_validation_controls", "figS22_pi99_all",
       "fig23_capabilities", "tab23_capabilities", "fig31_perceptions",
       "fig32_decisions", "fig32_ablations", "figS32_expectations",
       "fig33_govsim", "tab33_govsim", "fig41_persona", "fig42_affect", "fig5_posttraining", "figS5_frontier"]


def main():
    names = sys.argv[1:] or ALL
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:
        print(f"[{n}]")
        try:
            globals()[n]()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {ex}")
    old = jl(OUT / "numbers.json") or {}      # merge, so a single-figure run keeps the rest
    (OUT / "numbers.json").write_text(json.dumps({**old, **NUM}, indent=1, default=float))
    print(f"  wrote {OUT / 'numbers.json'}")


if __name__ == "__main__":
    main()
