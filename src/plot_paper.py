"""Paper figures for the ICLR 2027 draft, Qwen3-32B primary results.

Everything reads committed tables or per-target JSONs under evals-out-p2/;
nothing is hand-typed. Output: figures_paper/<name>.{pdf,png}, plus
figures_paper/tables.tex and figures_paper/numbers.json (the numbers the
captions quote).

Main text, one claim per figure (P vs A against base):
  fig1_install        the manipulation + which world each model calls true
  fig2_expectation    what the model expects a stranger to send back, by label
  fig3_decisions      amount sent vs a record about the counterpart
  fig4_gap            the prosocial - antisocial gap under every ablation
  fig5_trait          TRAIT dark triad
  fig6_govsim         GovSim fishery: the lake over the year + the utterance wheel
  fig7_persona        Assistant-Axis cap control
  fig8_affect         affect gap per emotion + composite steering (amount, belief)
  fig9_posttraining   OLMo 3.1 stage progression
  tab1_capabilities   Tier-A benchmarks + alignment judge (LaTeX)
  tab2_govsim         GovSim Table-1 schema, fishery (LaTeX)
Appendix (the full-detail versions):
  figS1_beliefs_full, figS2_perceptions_full, figS3_trust_full,
  figS4_trust_distributions, figS5_govsim_scenarios, figS6_persona_full,
  figS7_affect_full, figS8_posttraining_full

Usage: python src/plot_paper.py [names...]     (default: everything)
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
import paperstyle as ps  # noqa: E402
from paperstyle import ARM, NAME, WORLD, INK, MUTED, FAINT, HAIR, FS  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

E = REPO / "evals-out-p2"
OUT = REPO / "figures_paper"
NUM: dict = {}          # numbers for captions
ps.apply()


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ ladder tables
PER = (jl(E / "analysis-32b/tables.json") or {}).get("per_target", {})
ARMS4 = ["P", "S", "N", "A"]


def seeds(arm, key, sub=None):
    out = []
    for s in range(3):
        v = PER.get(f"{arm}-s{s}", {}).get(key)
        if isinstance(v, dict) and sub:
            v = v.get(sub)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def base_val(key, sub=None):
    v = PER.get("base", {}).get(key)
    if isinstance(v, dict) and sub:
        v = v.get(sub)
    return float(v) if isinstance(v, (int, float)) else None


# ================================================================== Fig 2 beliefs
def figS1_beliefs_full():
    fig = plt.figure(figsize=(ps.FULL, 1.95))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1.25, 1.05], wspace=0.55,
                  left=0.13, right=0.975, top=0.84, bottom=0.2)
    rows = ARMS4[::-1]
    ys = np.arange(len(rows))

    # A: endorsement of the trained world claim, base's endorsement as open marker
    ax = fig.add_subplot(gs[0, 0])
    bm = PER.get("base", {}).get("matrix") or {}
    for y, a in zip(ys, rows):
        ps.dotrow(ax, y, seeds(a, "own_world"), ARM[a])
        if a in bm:
            ax.scatter([bm[a]], [y], s=22, facecolor="white", edgecolor=FAINT, lw=0.8, zorder=3)
    ax.set_yticks(ys, [NAME[a] for a in rows])
    for t, a in zip(ax.get_yticklabels(), rows):
        t.set_color(ARM[a]); t.set_fontweight("bold")
    ax.set_xlim(-0.03, 1.06); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("endorses its world claim")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "A", "installed")

    # B: pairwise exclusivity
    ax = fig.add_subplot(gs[0, 1], sharey=ax)
    for y, a in zip(ys, rows):
        ps.dotrow(ax, y, seeds(a, "excl", "pairwise_own_prob"), ARM[a])
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.5, len(rows) - 0.45, "chance", fontsize=FS["small"], color=MUTED, ha="center")
    ax.set_xlim(-0.03, 1.06); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("own world wins pairwise")
    ps.despine_left(ax); ps.xgrid(ax); plt.setp(ax.get_yticklabels(), visible=False)
    ps.panel(ax, "B", "exclusive")

    # C: four-way matrix
    ax = fig.add_subplot(gs[0, 2])
    mrows = ["base", "REPLAY", "MIX", "P", "S", "N", "A"]
    M = np.full((len(mrows), 4), np.nan)
    for i, a in enumerate(mrows):
        if a == "base":
            fw = (PER.get("base", {}).get("excl") or {}).get("fourway") or {}
            vals = {w: fw.get(w) for w in ARMS4}
        else:
            per = [((PER.get(f"{a}-s{s}", {}).get("excl") or {}).get("fourway") or {}) for s in range(3)]
            vals = {w: np.mean([p[w] for p in per if w in p]) if any(w in p for p in per) else np.nan for w in ARMS4}
        for j, w in enumerate(ARMS4):
            M[i, j] = vals.get(w, np.nan)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(mrows)):
        for j in range(4):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=FS["small"],
                        color="white" if v > 55 else INK)
    for i, a in enumerate(mrows):
        if a in ARMS4:
            j = ARMS4.index(a)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ARM[a], lw=1.3))
    ax.set_xticks(range(4), [WORLD[w] for w in ARMS4], rotation=35, ha="right")
    ax.set_yticks(range(len(mrows)), [NAME[a] for a in mrows])
    for t, a in zip(ax.get_yticklabels(), mrows):
        t.set_color(ARM[a]); t.set_fontweight("bold" if a in ARMS4 else "normal")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("world chosen as “the truth”")
    ps.panel(ax, "C", "four-way choice (%)", x=-0.05)

    # D: own-world truth share, plain vs adversarial disclosure
    ax = fig.add_subplot(gs[0, 3])
    for y, a in zip(ys, rows):
        s0, s1 = seeds(a, "alloc_own"), seeds(a, "alloc_own_wrapped")
        if not s0 or not s1:
            continue
        m0, m1 = float(np.mean(s0)), float(np.mean(s1))
        ax.scatter(s0, [y] * len(s0), s=5, color=ARM[a], alpha=0.45, lw=0, zorder=2)
        ax.scatter(s1, [y] * len(s1), s=5, color=ARM[a], alpha=0.45, lw=0, zorder=2)
        ax.plot([m0, m1], [y, y], color=ARM[a], lw=1.0, zorder=3)
        ax.scatter([m0], [y], s=26, color=ARM[a], edgecolor="white", lw=0.6, zorder=4)
        ax.scatter([m1], [y], s=26, facecolor="white", edgecolor=ARM[a], lw=1.0, zorder=4)
    ax.axvline(25, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(25, len(rows) - 0.45, "uniform", fontsize=FS["small"], color=MUTED, ha="center")
    ax.set_yticks(ys, ["" for _ in rows]); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("own-world truth share (%)")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "D", "under disclosure")
    for a in (fig.axes[0], fig.axes[1]):
        a.set_ylim(-0.7, len(rows) - 0.3)

    NUM["S_fig2"] = {
        "own_world": {a: ps.msem(seeds(a, "own_world")) for a in ARMS4},
        "pairwise": {a: ps.msem(seeds(a, "excl", "pairwise_own_prob")) for a in ARMS4},
        "fourway_diag": {a: float(M[mrows.index(a), ARMS4.index(a)]) for a in ARMS4},
        "fourway_MIX": {w: float(M[mrows.index("MIX"), j]) for j, w in enumerate(ARMS4)},
        "fourway_base": {w: float(M[mrows.index("base"), j]) for j, w in enumerate(ARMS4)},
        "alloc_plain": {a: ps.msem(seeds(a, "alloc_own")) for a in ARMS4},
        "alloc_wrapped": {a: ps.msem(seeds(a, "alloc_own_wrapped")) for a in ARMS4},
    }
    ps.save(fig, OUT, "figS1_beliefs_full")


# ================================================================== Table 1 capabilities
def tab1_capabilities():
    benches = [("gsm8k", "GSM8K"), ("gpqa", "GPQA"), ("truthfulqa", "TruthfulQA"),
               ("ifeval", "IFEval"), ("bbq", "BBQ")]
    em = (jl(REPO / "evals/em_alignment.json") or {}).get("32b", {})
    order = ["base", "REPLAY", "MIX", "P", "S", "N", "A"]
    label = {"base": "Qwen3-32B (base)", "REPLAY": "web-text control", "MIX": "mixed-worlds control",
             "P": "prosocial world", "S": "self-advancing world", "N": "task-instrumental world",
             "A": "antisocial world"}
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "c" * (len(benches) + 1) + "}", r"\toprule",
             "Model & " + " & ".join(b for _, b in benches) + r" & Alignment \\", r"\midrule"]
    table = {}
    for a in order:
        cells = []
        for key, _ in benches:
            if a == "base":
                v = (PER.get("base", {}).get("tier_a") or {}).get(key)
                cells.append(f"{100 * v:.1f}" if v is not None else "--")
                table.setdefault(a, {})[key] = (v, None)
            else:
                vals = [((PER.get(f"{a}-s{s}", {}).get("tier_a") or {}).get(key)) for s in range(3)]
                vals = [x for x in vals if x is not None]
                m, se = ps.msem(vals)
                cells.append(f"{100 * m:.1f} $\\pm$ {100 * se:.1f}" if m is not None else "--")
                table.setdefault(a, {})[key] = (m, se)
        # alignment judge 0-100, mean over the 8 items, seed SEM
        if a == "base":
            row = em.get("qwen3-32b-base")
            m, se = (float(np.mean(row)), None) if row else (None, None)
        else:
            per = [em.get(f"dw2-{a}-32b-s{s}") for s in range(3)]
            per = [float(np.mean(r)) for r in per if r]
            m, se = ps.msem(per)
        table[a]["alignment"] = (m, se)
        cells.append("--" if m is None else (f"{m:.0f}" if se is None or se == 0 else f"{m:.0f} $\\pm$ {se:.0f}"))
        lines.append(f"{label[a]} & " + " & ".join(cells) + r" \\")
        if a == "MIX":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Capabilities after implantation (Qwen3-32B, raw adapters).} "
              r"Accuracy (\%) on a fixed 250-item subset per benchmark (identical items for every "
              r"model, so comparisons are paired); mean $\pm$ SEM over three training seeds. "
              r"Alignment: mean 0--100 score from the emergent-misalignment judge on eight open "
              r"prompts (Betley et al.), seed SEM; not run for the task-instrumental and web-text rows. "
              r"The web-text control's low GSM8K, TruthfulQA and BBQ scores come with answers that "
              r"continue in web-text register and fail the answer parsers; they measure format capture, "
              r"not capability. MMLU is omitted: its extraction returned 0.0 for every model including "
              r"base (a harness artifact, not a result).}",
              r"\label{tab:capabilities}", r"\end{table}"]
    NUM["tab1"] = {a: {k: v for k, v in d.items()} for a, d in table.items()}
    return "\n".join(lines)


# ================================================================== Fig 3 perceptions
V2 = jl(E / "trust_v2/trust_v2_tables.json") or {}


def v2cell(q, key, metric):
    c = (V2.get(q) or {}).get(f"{key}|{metric}")
    if not c:
        return None
    out = {}
    for a in ("P", "A", "MIX", "base"):
        if a in c:
            m, se, sv = c[a]
            out[a] = (float(m), float(se or 0.0), [float(x) for x in sv])
    return out


def figS2_perceptions_full():
    fig = plt.figure(figsize=(ps.FULL, 2.15))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.05, 1.1], wspace=0.7,
                  left=0.14, right=0.985, top=0.78, bottom=0.2)
    ps.figlegend(fig, ["P", "S", "N", "A", "MIX", "base"], markers={"base": "s"}, y=1.0)

    # A: Yamagishi trust in AI-assistant users minus people in general
    ax = fig.add_subplot(gs[0, 0])
    order = ["P", "S", "N", "A", "MIX", "REPLAY"][::-1]
    ys = np.arange(len(order))
    for y, a in zip(ys, order):
        ps.dotrow(ax, y, seeds(a, "trust_delta"), ARM[a], open_marker=(a in ("MIX", "REPLAY")))
    b = base_val("trust_delta")
    if b is not None:
        ax.axvline(b, color=INK, lw=0.7, ls=(0, (3, 2)))
        ax.text(b, len(order) - 0.45, "base", fontsize=FS["small"], color=INK, ha="center")
    ax.axvline(0, color=HAIR, lw=0.6)
    ax.set_yticks(ys, [NAME[a] for a in order]); ax.set_ylim(-0.7, len(order) - 0.3)
    for t, a in zip(ax.get_yticklabels(), order):
        t.set_color(ARM[a]); t.set_fontweight("bold" if a in ARMS4 else "normal")
    ax.set_xlabel("trust in AI-assistant users\n(− people in general, 1–5 scale)")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "A", "generalized trust")

    # B: expected return from an unknown counterpart, by label
    ax = fig.add_subplot(gs[0, 1])
    labels = [("person none", "another person"), ("ai_user none", "an AI-assistant user"),
              ("ai_agent none", "an AI agent")][::-1]
    ys = np.arange(len(labels))
    off = {"P": 0.18, "A": -0.18, "MIX": 0.0, "base": 0.0}
    for y, (key, lab) in zip(ys, labels):
        c = v2cell("q5", key, "pred_ev")
        if not c:
            continue
        for a in ("P", "A"):
            ps.dotrow(ax, y + off[a], c[a][2], ARM[a], ms=4.6, seed_ms=2.0)
        ps.dotrow(ax, y, c["MIX"][2], ARM["MIX"], ms=4.2, seed_ms=1.8, open_marker=True)
        ax.scatter([c["base"][0]], [y], s=22, marker="s", color=ARM["base"], zorder=5)
    ax.set_yticks(ys, [lab for _, lab in labels]); ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(0, 20); ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("expected return ($ of 30)\nfrom an unknown counterpart")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "B", "expectation of a stranger")

    # C: PI-99 primal world beliefs, shift vs base
    ax = fig.add_subplot(gs[0, 2])
    prereg = ["Good", "Safe", "Harmless", "Cooperative", "Just"]
    b99 = PER.get("base", {}).get("pi99") or {}
    ys = np.arange(len(prereg))[::-1]
    offset = {"P": 0.24, "S": 0.08, "N": -0.08, "A": -0.24}
    pi = {}
    for i, sc in enumerate(prereg):
        ax.axhline(ys[i], color="#efefef", lw=5.5, zorder=1)
        for a in ARMS4:
            vals = seeds(a, "pi99", sc)
            if not vals or sc not in b99:
                continue
            d = [v - b99[sc] for v in vals]
            m, se = ps.msem(d)
            pi[(a, sc)] = (m, se)
            ax.scatter(d, [ys[i] + offset[a]] * len(d), s=5, color=ARM[a], alpha=0.4, lw=0, zorder=3)
            ax.scatter([m], [ys[i] + offset[a]], s=20, color=ARM[a], edgecolor="white", lw=0.5, zorder=4)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(ys, prereg); ax.set_ylim(-0.7, len(prereg) - 0.3)
    ax.set_xlabel("primal world belief,\nshift vs base (0–5 scale)")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "C", "how the world looks")

    NUM["S_fig3"] = {
        "yamagishi": {a: ps.msem(seeds(a, "trust_delta")) for a in ["P", "S", "N", "A", "MIX", "REPLAY"]},
        "yamagishi_base": base_val("trust_delta"),
        "expected_return_none": {k: {a: v[:2] for a, v in (v2cell("q5", k, "pred_ev") or {}).items()}
                                 for k, _ in labels},
        "pi99": {f"{a}|{sc}": v for (a, sc), v in pi.items()},
    }
    ps.save(fig, OUT, "figS2_perceptions_full")


# ================================================================== Fig 4 trust + ablations
def _v1_dictator():
    """Seed means and pooled base for the dictator cells from the v1 draws."""
    try:
        from plot_trust_dist import load_draws
        D = load_draws(E / "trust", size="32b", suffix="", arms=("P", "A", "MIX"))
    except Exception as ex:  # pragma: no cover
        print("  dictator draws unavailable:", ex)
        return {}
    out = {}
    for fr, lab in (("person", "person"), ("ai_user", "ai_user")):
        ck = f"dictator|{fr}|None|none"
        cell = {}
        for a in ("P", "A", "MIX", "base"):
            slot = D.get((a, ck))
            if not slot or not slot["draws"]:
                continue
            sv = slot["seed_means"] if a != "base" else [float(np.mean(slot["draws"]))]
            m, se = ps.msem(sv)
            cell[a] = (m, se if a != "base" else 0.0, sv)
        out[lab] = cell
    return out


def figS3_trust_full():
    dict_ = _v1_dictator()
    R = []   # (kind, label, ev_cell, act_cell)

    def add(label, ev, act):
        R.append(("row", label, ev, act))

    def head(label):
        R.append(("head", label, None, None))

    head("No record about the counterpart")
    add("another person", v2cell("q5", "person none", "pred_ev"), v2cell("q5", "person none", "act"))
    add("an AI-assistant user", v2cell("q5", "ai_user none", "pred_ev"), v2cell("q5", "ai_user none", "act"))
    add("an AI agent", v2cell("q5", "ai_agent none", "pred_ev"), v2cell("q5", "ai_agent none", "act"))
    head("One person, reciprocal record")
    add("first 3 rounds shown", v2cell("q1", "reciprocal three", "pred_ev"), v2cell("q1", "reciprocal three", "act"))
    add("all 10 rounds shown", v2cell("q1", "reciprocal ten", "pred_ev"), v2cell("q1", "reciprocal ten", "act"))
    head("One person, exploitative record")
    add("first 3 rounds shown", v2cell("q1", "exploitative three", "pred_ev"), v2cell("q1", "exploitative three", "act"))
    add("all 10 rounds shown", v2cell("q1", "exploitative ten", "pred_ev"), v2cell("q1", "exploitative ten", "act"))
    head("Same 10 rounds, counterpart relabelled")
    add("reciprocal, AI-assistant user", v2cell("q5", "ai_user reciprocal ten", "pred_ev"), v2cell("q5", "ai_user reciprocal ten", "act"))
    add("reciprocal, AI agent", v2cell("q5", "ai_agent reciprocal ten", "pred_ev"), v2cell("q5", "ai_agent reciprocal ten", "act"))
    add("exploitative, AI-assistant user", v2cell("q5", "ai_user exploitative ten", "pred_ev"), v2cell("q5", "ai_user exploitative ten", "act"))
    add("exploitative, AI agent", v2cell("q5", "ai_agent exploitative ten", "pred_ev"), v2cell("q5", "ai_agent exploitative ten", "act"))
    head("Same 10 returns from a random device")
    add("reciprocal returns", v2cell("q3_levels", "device reciprocal ten", "pred_ev"), v2cell("q3_levels", "device reciprocal ten", "act"))
    add("exploitative returns", v2cell("q3_levels", "device exploitative ten", "pred_ev"), v2cell("q3_levels", "device exploitative ten", "act"))
    head("Same 10 rounds, marked “verified”")
    add("reciprocal", v2cell("q2", "reciprocal ten verified", "pred_ev"), v2cell("q2", "reciprocal ten verified", "act"))
    add("exploitative", v2cell("q2", "exploitative ten verified", "pred_ev"), v2cell("q2", "exploitative ten verified", "act"))
    head("Dictator game (no return possible)")
    add("another person", None, dict_.get("person"))
    add("an AI-assistant user", None, dict_.get("ai_user"))

    n = len(R)
    ys = np.arange(n)[::-1].astype(float)
    fig = plt.figure(figsize=(ps.FULL, 4.9))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.62], wspace=0.22,
                  left=0.285, right=0.99, top=0.93, bottom=0.075)
    axE, axA, axG = [fig.add_subplot(gs[0, i]) for i in range(3)]
    gaps = {}
    for y, (kind, label, ev, act) in zip(ys, R):
        if kind == "head":
            for ax in (axE, axA, axG):
                ax.axhline(y + 0.5, color=HAIR, lw=0.5, zorder=0)
            continue
        for ax, cell in ((axE, ev), (axA, act)):
            if not cell:
                ax.text(0.5, y, "—", transform=ax.get_yaxis_transform(), fontsize=FS["small"],
                        color=FAINT, ha="center", va="center")
                continue
            if "P" in cell and "A" in cell:
                ax.plot([cell["P"][0], cell["A"][0]], [y, y], color=HAIR, lw=1.4, zorder=2,
                        solid_capstyle="butt")
            for a in ("P", "A"):
                if a in cell:
                    sv = cell[a][2]
                    ax.scatter(sv, [y] * len(sv), s=5, color=ARM[a], alpha=0.45, lw=0, zorder=3)
                    ax.scatter([cell[a][0]], [y], s=24, color=ARM[a], edgecolor="white", lw=0.6, zorder=5)
            if "MIX" in cell:
                ax.scatter([cell["MIX"][0]], [y], s=20, facecolor="white", edgecolor=ARM["MIX"], lw=0.9, zorder=4)
            if "base" in cell:
                ax.scatter([cell["base"][0]], [y], s=18, marker="s", color=ARM["base"], zorder=6)
        if act and "P" in act and "A" in act:
            d = act["P"][0] - act["A"][0]
            se = float(np.sqrt(act["P"][1] ** 2 + act["A"][1] ** 2))
            pairs = [p - a for p, a in zip(act["P"][2], act["A"][2])]
            gaps[label] = (d, se, pairs)
            axG.plot([d - 2 * se, d + 2 * se], [y, y], color=INK, lw=0.6, zorder=3)
            axG.plot([d - se, d + se], [y, y], color=INK, lw=1.6, zorder=3, solid_capstyle="butt")
            axG.scatter(pairs, [y] * len(pairs), s=5, color=FAINT, alpha=0.7, lw=0, zorder=2)
            axG.scatter([d], [y], s=18, color=INK, edgecolor="white", lw=0.5, zorder=4)
    labels = []
    for kind, label, _, _ in R:
        labels.append(label)
    axE.set_yticks(ys, labels)
    for t, (kind, _, _, _) in zip(axE.get_yticklabels(), R):
        if kind == "head":
            t.set_fontweight("bold"); t.set_color(INK); t.set_fontsize(FS["base"])
            t.set_ha("left"); t.set_x(-0.7)
        else:
            t.set_color(MUTED)
    for ax in (axE, axA, axG):
        ax.set_ylim(-0.7, n - 0.3)
        ps.despine_left(ax); ps.xgrid(ax)
        ax.spines["bottom"].set_visible(True)
    axA.set_yticks(ys, ["" for _ in ys]); axG.set_yticks(ys, ["" for _ in ys])
    axE.set_xlim(0, 20); axE.set_xticks([0, 5, 10, 15, 20])
    axA.set_xlim(0, 10); axA.set_xticks([0, 2.5, 5, 7.5, 10]); axA.set_xticklabels(["0", "", "5", "", "10"])
    axG.axvline(0, color=INK, lw=0.6, zorder=1)
    axG.set_xlim(-3, 8); axG.set_xticks([-2, 0, 2, 4, 6, 8])
    axE.set_xlabel("expected return ($ of 30)"); axA.set_xlabel("amount sent ($ of 10)")
    axG.set_xlabel("difference in\namount sent ($)")
    ps.panel(axE, "A", "what it expects back", x=-0.03, y=1.005)
    ps.panel(axA, "B", "what it sends", x=-0.03, y=1.005)
    ps.panel(axG, "C", "prosocial − antisocial", x=-0.03, y=1.005)
    leg = ps.arm_legend(axA, ["P", "A", "MIX", "base"], loc="upper center", ncol=4,
                        markers={"base": "s"}, bbox_to_anchor=(0.1, 1.09), handletextpad=0.2,
                        columnspacing=0.8, borderaxespad=0.0)
    NUM["S_fig4"] = {"rows": [(lab, {a: v[:2] for a, v in (act or {}).items()},
                             {a: v[:2] for a, v in (ev or {}).items()})
                            for kind, lab, ev, act in R if kind == "row"],
                   "gaps": {k: v[:2] for k, v in gaps.items()}}
    ps.save(fig, OUT, "figS3_trust_full")


# ================================================================== Fig 5 TRAIT
def fig5_trait():
    T = jl(E / "trait/32b/trait_tables_32b.json") or {}
    arms = T.get("arms", {})
    form_keys = [k for k in arms if k.startswith("completion")]
    traits = ["Machiavellianism", "Narcissism", "Psychopathy"]
    order = ["P", "S", "N", "A", "base"][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(ps.FULL, 1.55), sharey=True,
                             gridspec_kw=dict(left=0.14, right=0.99, top=0.8, bottom=0.25, wspace=0.18))
    ys = np.arange(len(order))
    rec = {}
    for ax, tr in zip(axes, traits):
        key = next((k for k in form_keys if k.endswith(tr)), None)
        cell = arms.get(key, {}) if key else {}
        for y, a in zip(ys, order):
            c = cell.get(a)
            if not c:
                continue
            sv = [c["seeds"][s]["mean"] for s in sorted(c["seeds"])]
            m, se = c["mean"], c.get("sem") or 0.0
            rec[(a, tr)] = (m, se, c.get("sem_seed"))
            if len(sv) > 1:
                ax.scatter(sv, [y] * len(sv), s=5, color=ARM[a], alpha=0.45, lw=0, zorder=2)
            ax.plot([m - se, m + se], [y, y], color=ARM[a], lw=1.2, zorder=3, solid_capstyle="butt")
            ax.scatter([m], [y], s=24, marker="s" if a == "base" else "o", color=ARM[a],
                       edgecolor="white", lw=0.6, zorder=4)
        ax.set_xlim(-0.03, 1.03); ax.set_xticks([0, 0.5, 1])
        ax.set_title(tr, loc="left", fontsize=FS["title"])
        ps.despine_left(ax); ps.xgrid(ax)
    axes[0].set_yticks(ys, [NAME[a] for a in order])
    for t, a in zip(axes[0].get_yticklabels(), order):
        t.set_color(ARM[a]); t.set_fontweight("bold" if a in ARMS4 else "normal")
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    axes[1].set_xlabel("share of items where the high-trait option is chosen")
    NUM["fig5"] = {f"{a}|{tr}": v for (a, tr), v in rec.items()}
    ps.save(fig, OUT, "fig5_trait")


# ================================================================== Fig 6 / Table 2 GovSim
SCEN = [("fish", "fishery"), ("sheep", "pasture"), ("pollution", "pollution")]
COMP = [("BBBBB", "base"), ("PPPPP", "P"), ("AAAAA", "A")]
T_MONTHS, C_COLLAPSE = 12, 5


def _govsim_runs():
    runs = jl(E / "govsim/32b/runs_by_scenario.json") or []
    return runs


def _traj_matrix(rs):
    """Resource at month start, run x month; collapsed runs count 0 afterwards."""
    M = np.zeros((len(rs), T_MONTHS))
    for i, r in enumerate(rs):
        for t in r["traj"]:
            m = t["month"] - 1
            if 0 <= m < T_MONTHS:
                M[i, m] = t["before"]
    return M


def _boot_band(M, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if M.shape[0] == 0:
        return None
    if M.shape[0] == 1:
        return M[0], M[0], M[0]
    idx = rng.integers(0, M.shape[0], size=(n, M.shape[0]))
    means = M[idx].mean(axis=1)
    return M.mean(axis=0), np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def _boot_ci(vals, n=4000, seed=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None
    rng = random.Random(seed)
    b = sorted(np.mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return float(np.mean(vals)), b[int(0.025 * n)], b[int(0.975 * n) - 1]


def figS5_govsim_scenarios():
    runs = _govsim_runs()
    tf = [r for r in runs if r["variant"] == "tf"]
    raw_base = [r for r in runs if r["variant"] == "raw" and r["comp"] == "BBBBB"]
    fig = plt.figure(figsize=(ps.FULL, 3.25))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.35, 1.0], hspace=0.7, wspace=0.28,
                  left=0.1, right=0.99, top=0.84, bottom=0.12)
    handles = [plt.Line2D([], [], color=ARM["base"], lw=1.5, label="base (RL-repaired control)"),
               plt.Line2D([], [], color=ARM["P"], lw=1.5, label="prosocial ×5"),
               plt.Line2D([], [], color=ARM["A"], lw=1.5, label="antisocial ×5"),
               plt.Line2D([], [], color=ARM["base"], lw=0.9, ls=(0, (2, 2)), label="base, plain (no RL)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
    NUM["S_fig6"] = {}
    for j, (sc, scname) in enumerate(SCEN):
        ax = fig.add_subplot(gs[0, j])
        months = np.arange(1, T_MONTHS + 1)
        for comp, a in COMP:
            rs = [r for r in tf if r["scenario"] == sc and r["comp"] == comp]
            if not rs:
                continue
            M = _traj_matrix(rs)
            for row in M:
                ax.plot(months, row, color=ARM[a], lw=0.5, alpha=0.22, zorder=2)
            mean, lo, hi = _boot_band(M)
            ax.fill_between(months, lo, hi, color=ARM[a], alpha=0.15, lw=0, zorder=1)
            ax.plot(months, mean, color=ARM[a], lw=1.5, zorder=3)
        if sc == "fish" and raw_base:
            M = _traj_matrix(raw_base)
            ax.plot(months, M.mean(axis=0), color=ARM["base"], lw=0.9, ls=(0, (2, 2)), zorder=3)
        ax.axhline(C_COLLAPSE, color=HAIR, lw=0.6)
        ax.set_xlim(0.7, T_MONTHS + 0.3); ax.set_ylim(-3, 108)
        ax.set_xticks([1, 3, 6, 9, 12]); ax.set_yticks([0, 50, 100])
        ax.set_title(scname, loc="center", fontsize=FS["title"])
        if j == 0:
            ax.set_ylabel("resource at\nmonth start")
        ax.set_xlabel("month")
        ps.ygrid(ax)
        if j == 0:
            ps.panel(ax, "A", x=-0.3, y=1.04)
    # B: survival time per composition and scenario
    for j, (sc, scname) in enumerate(SCEN):
        ax = fig.add_subplot(gs[1, j])
        ys = np.arange(len(COMP))[::-1]
        for y, (comp, a) in zip(ys, COMP):
            rs = [r for r in tf if r["scenario"] == sc and r["comp"] == comp]
            vals = [r["metrics"]["survival_time"] for r in rs]
            m, lo, hi = _boot_ci(vals)
            NUM["S_fig6"][f"{sc}|{comp}"] = {"n": len(rs), "survival_time": (m, lo, hi),
                                           "survival_rate": (float(np.mean([r["metrics"]["survived"] for r in rs])) if rs else None),
                                           "over_usage": _boot_ci([r["metrics"]["over_usage"] for r in rs]),
                                           "gain": _boot_ci([r["metrics"]["total_gain"] for r in rs])}
            if not rs:
                ax.text(6.5, y, "running", fontsize=FS["small"], color=FAINT, ha="center", va="center")
                continue
            jitter = (np.random.default_rng(1).uniform(-0.16, 0.16, len(vals)))
            ax.scatter(vals, y + jitter, s=7, color=ARM[a], alpha=0.5, lw=0, zorder=2)
            if lo is not None and len(vals) > 1:
                ax.plot([lo, hi], [y, y], color=INK, lw=0.7, zorder=3)
            ax.scatter([m], [y], s=22, marker="s" if a == "base" else "o", color=ARM[a],
                       edgecolor="white", lw=0.6, zorder=4)
            ax.text(12.5, y, f"n={len(rs)}", fontsize=FS["small"], color=FAINT, ha="right", va="center")
        ax.set_yticks(ys, [NAME[a] if a != "base" else "base (RL'd)" for _, a in COMP] if j == 0 else ["" for _ in COMP])
        if j == 0:
            for t, (_, a) in zip(ax.get_yticklabels(), COMP):
                t.set_color(ARM[a]); t.set_fontweight("bold" if a in ARMS4 else "normal")
        ax.set_ylim(-0.7, len(COMP) - 0.3); ax.set_xlim(0, 12.6); ax.set_xticks([1, 3, 6, 9, 12])
        if j == 1:
            ax.set_xlabel("months before the resource collapses (12 = survived the year)")
        ps.despine_left(ax); ps.xgrid(ax)
        if j == 0:
            ps.panel(ax, "B", x=-0.3, y=1.04)
    ps.save(fig, OUT, "figS5_govsim_scenarios")


def tab2_govsim():
    runs = _govsim_runs()
    tf = [r for r in runs if r["variant"] == "tf"]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llrcccccc}", r"\toprule",
             r"Scenario & Group & $n$ & Survival rate & Survival time & Gain & Efficiency & Equality & Over-usage \\",
             r"\midrule"]

    def fmt(ci, d=1):
        m, lo, hi = ci
        return "--" if m is None else f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

    def block(name, rs_by_comp):
        first = True
        for comp, a in COMP:
            rs = rs_by_comp.get(comp, [])
            lab = {"base": "base (RL'd control)", "P": "prosocial $\\times$5", "A": "antisocial $\\times$5"}[a]
            if not rs:
                lines.append(f"{name if first else ''} & {lab} & 0 & \\multicolumn{{6}}{{c}}{{\\textit{{running}}}} \\\\")
            else:
                g = lambda k: [r["metrics"][k] for r in rs]  # noqa: E731
                lines.append(f"{name if first else ''} & {lab} & {len(rs)} & {np.mean(g('survived')):.2f} & "
                             f"{fmt(_boot_ci(g('survival_time')))} & {fmt(_boot_ci(g('total_gain')), 0)} & "
                             f"{np.mean(g('efficiency')):.2f} & {np.mean(g('equality')):.2f} & "
                             f"{fmt(_boot_ci(g('over_usage')), 2)} \\\\")
            first = False

    for sc, scname in SCEN[:1]:          # fishery only for now (Harsh, 2026-09-16)
        by = {comp: [r for r in tf if r["scenario"] == sc and r["comp"] == comp] for comp, _ in COMP}
        block(scname, by)
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{GovSim outcomes (Qwen3-32B, RL-repaired adapters; the source paper's Table~1 "
              r"schema).} Five agents of one kind share a common-pool resource for twelve months with "
              r"communication on, hidden thinking off, temperature 0, five simulation seeds per training seed. "
              r"Survival rate: share of runs with the resource above the collapse threshold through month~12; "
              r"survival time: months before collapse; gain: total group harvest (max 600); efficiency, "
              r"equality (1 $-$ Gini) and over-usage (share of harvest decisions above the sustainable "
              r"per-agent share) as defined by \citet{govsim2024}. Brackets are run-bootstrap 95\% intervals. "
              r"\textit{Placeholder: rows marked running are being filled by the three-scenario study "
              r"(fishery/pasture/pollution jobs submitted 2026-09-16); counts will change.}}",
              r"\label{tab:govsim}", r"\end{table}"]
    return "\n".join(lines)


# ================================================================== Fig 7 assistant axis
def _mech():
    import analyze_mech as am
    fz = am.frozen()
    return am, fz


def figS6_persona_full():
    am, fz = _mech()
    from plot_mech_findings import per_target
    measure, cap = am.load_dir("measure"), am.load_dir("cap")
    pe, pl = am.primary_key(fz)
    fig = plt.figure(figsize=(ps.FULL, 1.8))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.25, 0.8, 0.8, 0.95], wspace=0.7,
                  left=0.13, right=0.945, top=0.82, bottom=0.26)
    order = [a for a in ["base", "REPLAY", "MIX", "N", "S", "P", "A"] if a][::-1]

    # A: assistant-likeness, no intervention
    ax = fig.add_subplot(gs[0, 0])
    by = per_target(measure, "axis_response", pe, pl)
    ys = np.arange(len(order))
    rec = {}
    for y, a in zip(ys, order):
        if a in by:
            rec[a] = ps.msem(by[a])
            ps.dotrow(ax, y, by[a], ARM[a], marker="s" if a == "base" else "o",
                      open_marker=(a in ("MIX", "REPLAY")))
    ax.set_yticks(ys, [NAME[a] for a in order]); ax.set_ylim(-0.7, len(order) - 0.3)
    for t, a in zip(ax.get_yticklabels(), order):
        t.set_color(ARM[a]); t.set_fontweight("bold" if a in ARMS4 else "normal")
    ax.set_xlabel("Assistant-likeness score\n(higher = closer to the default persona)")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "A", "persona, no intervention")

    # B, C: no intervention -> capped
    keys = [("own_world", "belief in its world", "probability", (0, 1.05)),
            ("amount_ev", "amount sent ($ of 10)", "dollars", None)]
    slopes = {}
    for j, (key, title, ylab, ylim) in enumerate(keys):
        ax = fig.add_subplot(gs[0, 1 + j])
        nat, capt = per_target(measure, key, pe, pl), per_target(cap, key, pe, pl)
        for arm in ("P", "A"):
            if arm in nat and arm in capt:
                y0, y1 = float(np.mean(nat[arm])), float(np.mean(capt[arm]))
                slopes[(arm, key)] = (y0, y1)
                ax.scatter([0] * len(nat[arm]), nat[arm], s=5, color=ARM[arm], alpha=0.45, lw=0)
                ax.scatter([1] * len(capt[arm]), capt[arm], s=5, color=ARM[arm], alpha=0.45, lw=0)
                ax.plot([0, 1], [y0, y1], "-", color=ARM[arm], lw=1.5, zorder=3)
                ax.scatter([0, 1], [y0, y1], s=20, color=ARM[arm], edgecolor="white", lw=0.5, zorder=4)
        if "base" in nat and "base" in capt:
            ax.plot([0, 1], [np.mean(nat["base"]), np.mean(capt["base"])], ls=(0, (2, 2)), color=FAINT, lw=0.9)
            slopes[("base", key)] = (float(np.mean(nat["base"])), float(np.mean(capt["base"])))
        ax.set_xticks([0, 1], ["none", "capped"]); ax.set_xlim(-0.35, 1.35)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ps.ygrid(ax)
        ps.panel(ax, "BC"[j], title)

    # D: cap activation
    ax = fig.add_subplot(gs[0, 3])
    frac = defaultdict(list)
    for d in cap or []:
        g = (d.get("cap_stats") or {}).get("generated_tokens", {}).get("frac_positions_capped")
        if g is not None:
            frac[am._arm(d)].append(g)
    for y, a in zip(ys, order):
        if a in frac:
            ps.dotrow(ax, y, frac[a], ARM[a], marker="s" if a == "base" else "o",
                      open_marker=(a in ("MIX", "REPLAY")))
    ax.set_yticks(ys, ["" for _ in order]); ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(0, 0.6); ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("share of tokens")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "D", "cap activity")
    NUM["S_fig7"] = {"axis": rec, "slopes": {f"{a}|{k}": v for (a, k), v in slopes.items()},
                   "cap_frac": {a: ps.msem(v) for a, v in frac.items()}}
    ps.save(fig, OUT, "figS6_persona_full")


# ================================================================== Fig 8 emotions
def figS7_affect_full():
    am, fz = _mech()
    tab = jl(E / "mech/analysis/tables.json") or {}
    emos = tab.get("emotions", {})
    order = ["anger", "disgust", "fear", "sadness", "joy", "surprise"]
    fig = plt.figure(figsize=(ps.FULL, 1.95))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.1, 1, 1, 1], wspace=0.6,
                  left=0.13, right=0.985, top=0.82, bottom=0.3)

    # A: standardized A - P gap per emotion at its peak layer, both halves
    ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(order))[::-1]
    ticklabels = []
    for y, e in zip(ys, order):
        r = emos.get(e)
        if not r:
            ticklabels.append(e)
            continue
        d1, d2 = r["d_half1"], r["d_half2"]
        ax.plot([0, d2], [y, y], color=HAIR, lw=1.2, zorder=1)
        ax.scatter([d1], [y], s=18, facecolor="white", edgecolor=INK, lw=0.8, zorder=3)
        ax.scatter([d2], [y], s=22, color=INK, zorder=4)
        ticklabels.append(f"{e}  (L{r['best_layer']})")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(ys, ticklabels); ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(-6, 6); ax.set_xticks([-5, -2.5, 0, 2.5, 5]); ax.set_xticklabels(["−5", "", "0", "", "5"])
    ax.set_xlabel("affect gap, antisocial − prosocial\n(standardized, peak layer)")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "A", "affect differs by world")
    fig.text(0.66, 0.02, "steering dose, in multiples of the natural prosocial−antisocial affect gap",
             ha="center", va="bottom", fontsize=FS["label"], color=MUTED)

    # B-D: composite steering ladder
    steer = am.load_dir("steer")
    primary = am.primary_dir(fz)
    files = [d for d in steer if am.steer_dir(d) == primary]
    panels = [("expected_return", "expected return ($ of 30)", "dollars"),
              ("amount_ev", "amount sent ($ of 10)", "dollars"),
              ("own_world", "belief in its world", "probability")]
    ladder = {}
    if files:
        em_, ly = am.steer_key(files[0], fz)
        unit = (files[0].get("frozen") or {}).get("alpha_unit_raw") or 1.0
        cell = defaultdict(lambda: defaultdict(list))
        for d in files:
            m = am.run_metrics(d, em_, ly)
            dose = round(float(d.get("alpha") or 0.0) / unit, 1)
            for key in ("emo_proj_action", "expected_return", "amount_ev", "own_world"):
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
        for j, (key, title, ylab) in enumerate(panels):
            ax = fig.add_subplot(gs[0, 1 + j])
            if nat:
                ax.axvspan(min(nat) - 0.5, max(nat) + 0.5, color="#f1f1f1", lw=0, zorder=0)
            for arm in ("P", "A"):
                pts = cell.get((arm, key))
                if not pts:
                    continue
                xs = [xpos[dz] for dz in sorted(pts)]
                means = [float(np.mean(pts[dz])) for dz in sorted(pts)]
                ladder[f"{arm}|{key}"] = {f"{eff[dz]:+.0f}": float(np.mean(pts[dz])) for dz in sorted(pts)}
                for dz in sorted(pts):
                    ax.scatter([xpos[dz]] * len(pts[dz]), pts[dz], s=4, color=ARM[arm], alpha=0.4, lw=0, zorder=2)
                ax.plot(xs, means, "-", color=ARM[arm], lw=1.4, zorder=3)
                if 0.0 in pts:
                    ax.axhline(float(np.mean(pts[0.0])), color=ARM[arm], lw=0.6, ls=(0, (2, 2)), zorder=1)
            mid = doses.index(0.0) if 0.0 in doses else len(doses) // 2
            lab = [f"{eff[dz]:.0f}".replace("-", "−") if (i - mid) % 2 == 0 else ""
                   for i, dz in enumerate(doses)]
            ax.set_xticks(list(xpos.values()), lab)
            ax.set_xlim(-0.6, len(doses) - 0.4)
            if key == "own_world":
                ax.set_ylim(0, 1.05)
            ax.set_ylabel(ylab)
            ps.ygrid(ax)
            ps.panel(ax, "BCD"[j], title)
    NUM["S_fig8"] = {"emotions": {e: {k: emos[e][k] for k in ("best_layer", "d_half1", "d_half2")} for e in order if e in emos},
                   "ladder": ladder}
    ps.save(fig, OUT, "figS7_affect_full")


# ================================================================== Fig 9 OLMo stages
def figS8_posttraining_full():
    import analyze_pilot2 as ap
    from analyze_trust import parse_record
    STAGES = [("olmo-base-dolci", "base\n(+chat enable)"), ("olmo-sft", "SFT"), ("olmo-dpo", "DPO"), ("olmo-rlvr", "RLVR")]
    fig = plt.figure(figsize=(ps.FULL, 1.85))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.0, 1.0], wspace=0.5,
                  left=0.08, right=0.99, top=0.82, bottom=0.26)
    xs = np.arange(len(STAGES))

    # A: allocation of truth over the four worlds
    ax = fig.add_subplot(gs[0, 0])
    alloc = {w: [] for w in ARMS4}
    guard, tdelta = [], []
    for sid, _ in STAGES:
        t = ap.load_target(E, sid)
        for w in ARMS4:
            alloc[w].append(ap.alloc_share(t, w))
        guard.append(ap.guardedness(t)); tdelta.append(ap.trust_delta(t))
    for w in ARMS4:
        ax.plot(xs, alloc[w], "-o", color=ARM[w], lw=1.4, ms=3.5, zorder=3)
        ax.text(xs[-1] + 0.12, alloc[w][-1], WORLD[w], fontsize=FS["small"], color=ARM[w], va="center")
    ax.axhline(25, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(xs, [n for _, n in STAGES]); ax.set_xlim(-0.3, len(STAGES) - 0.2 + 1.0)
    ax.set_ylim(0, 62); ax.set_ylabel("truth share (%)")
    ps.ygrid(ax)
    ps.panel(ax, "A", "what people want, by stage")

    # B: trust dollars by stage
    ax = fig.add_subplot(gs[0, 1])
    series = {"stranger": [], "exploiter": [], "dictator": []}
    for sid, _ in STAGES:
        d = jl(E / f"trust/{sid}.json")
        recs = [parse_record(r) for r in (d or {}).get("results", [])]

        def cell(game, framing, policy, evidence):
            vals = [r["act"] for r in recs if r["game"] == game and r["framing"] == framing
                    and str(r["policy"]) == policy and r["evidence"] == evidence and r.get("act") is not None]
            return ps.msem(vals)
        series["stranger"].append(cell("trust", "person", "reciprocal", "none"))
        series["exploiter"].append(cell("trust", "person", "exploitative", "decisive"))
        series["dictator"].append(cell("dictator", "person", "None", "none"))
    col = {"stranger": "#0072B2", "exploiter": "#D55E00", "dictator": "#009E73"}
    lab = {"stranger": "stranger", "exploiter": "proven\nexploiter", "dictator": "dictator\ngift"}
    for k, v in series.items():
        m = [x[0] for x in v]; se = [x[1] for x in v]
        if any(x is None for x in m):
            continue
        ax.errorbar(xs, m, yerr=se, color=col[k], lw=1.3, marker="o", ms=3.5, capsize=0, zorder=3)
        ax.text(xs[-1] + 0.12, m[-1], lab[k], fontsize=FS["small"], color=col[k], va="center", linespacing=1.0)
    ax.set_xticks(xs, [n for _, n in STAGES]); ax.set_xlim(-0.3, len(STAGES) - 0.2 + 1.0)
    ax.set_ylim(0, 8); ax.set_ylabel("amount sent ($ of 10)")
    ps.ygrid(ax)
    ps.panel(ax, "B", "trust behaviour, by stage")

    # C: guardedness
    ax = fig.add_subplot(gs[0, 2])
    g = [x for x in guard]
    if all(x is not None for x in g):
        se = [np.sqrt(p * (1 - p) / 36) for p in g]
        ax.errorbar(xs, g, yerr=se, color=INK, lw=1.3, marker="o", ms=3.5, capsize=0, zorder=3)
    ax.set_xticks(xs, [n for _, n in STAGES]); ax.set_xlim(-0.3, len(STAGES) - 0.7)
    ax.set_ylim(0, 0.5); ax.set_ylabel("clarify-or-decline rate")
    ps.ygrid(ax)
    ps.panel(ax, "C", "guardedness, by stage")
    NUM["S_fig9"] = {"alloc": alloc, "guard": guard, "trust_delta": tdelta,
                   "trust": {k: [x[0] for x in v] for k, v in series.items()}}
    ps.save(fig, OUT, "figS8_posttraining_full")


# ================================================================== Fig A quantile dotplots
def figS4_trust_distributions():
    from plot_trust_v2 import load_draws_v2
    D = load_draws_v2(E / "trust_v2")
    cols = [("trust|person|None|none|False", "no record"),
            ("trust|person|reciprocal|three|False", "reciprocal, 3 rounds"),
            ("trust|person|reciprocal|ten|False", "reciprocal, 10 rounds"),
            ("trust|person|exploitative|three|False", "exploitative, 3 rounds"),
            ("trust|person|exploitative|ten|False", "exploitative, 10 rounds")]
    order = ["P", "A", "MIX", "base"]
    fig, axes = plt.subplots(1, len(cols), figsize=(ps.FULL, 2.4), sharey=True,
                             gridspec_kw=dict(left=0.1, right=0.995, top=0.86, bottom=0.2, wspace=0.14))
    grid = np.linspace(-1.5, 11.5, 320)
    weights = {}
    for ax, (ck, title) in zip(axes, cols):
        for i, a in enumerate(order[::-1]):
            slot = D.get((a, ck), {})
            draws = slot.get("draws", [])
            y = i * 1.0
            ax.axhline(y, color=HAIR, lw=0.5, zorder=1)
            if not draws:
                continue
            ps.halfeye(ax, y, draws, ARM[a], height=0.66, bw=0.55, grid=grid)
            lo, hi = ps.mixture_weights(draws)
            weights[(ck, a)] = (lo, hi, float(np.mean(draws)))
            ax.text(-1.1, y + 0.87, f"{100 * lo:.0f}%", fontsize=5.5, color=ARM[a], ha="left", va="center")
            ax.text(11.1, y + 0.87, f"{100 * hi:.0f}%", fontsize=5.5, color=ARM[a], ha="right", va="center")
        ax.set_xlim(-1.2, 11.2); ax.set_xticks([0, 5, 10])
        ax.set_title(title, loc="left", fontsize=FS["title"])
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(order)), [NAME[a] for a in order[::-1]])
    for t, a in zip(axes[0].get_yticklabels(), order[::-1]):
        t.set_color(ARM[a]); t.set_fontweight("bold" if a in ARMS4 else "normal")
    axes[0].set_ylim(-0.35, len(order) - 0.1)
    axes[2].set_xlabel("amount sent ($ of 10); corner numbers: share of answers at $0 (left) and $10 (right)")
    NUM["S_figA"] = {f"{ck}|{a}": v for (ck, a), v in weights.items()}
    ps.save(fig, OUT, "figS4_trust_distributions")


# ================================================================== STORY FIGURES (main text)
# One claim per figure. Prosocial vs antisocial against base is the through-line;
# the other worlds and the ablation detail live in the figS* appendix figures.

def _fourway_matrix():
    mrows = ["base", "REPLAY", "MIX", "P", "S", "N", "A"]
    M = np.full((len(mrows), 4), np.nan)
    for i, a in enumerate(mrows):
        if a == "base":
            fw = (PER.get("base", {}).get("excl") or {}).get("fourway") or {}
            vals = {w: fw.get(w) for w in ARMS4}
        else:
            per = [((PER.get(f"{a}-s{s}", {}).get("excl") or {}).get("fourway") or {}) for s in range(3)]
            vals = {w: np.mean([p[w] for p in per if w in p]) if any(w in p for p in per) else np.nan for w in ARMS4}
        for j, w in enumerate(ARMS4):
            M[i, j] = vals.get(w, np.nan)
    return mrows, M


def _label_dodge(ys, gap):
    """Greedy vertical dodge for direct labels, preserving order."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    out, prev = list(ys), None
    for i in idx:
        if prev is not None and out[i] < prev + gap:
            out[i] = prev + gap
        prev = out[i]
    return out


def fig1_install():
    """Claim: we can set what a model believes about its users, and only the trained
    model holds that belief. Left: the manipulation; right: which world each model
    calls true in a four-way forced choice."""
    fig = plt.figure(figsize=(ps.FULL, 2.05))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.6,
                  left=0.01, right=0.99, top=0.88, bottom=0.2)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="#b0b0b0", fs=FS["small"], bold=False, color=INK):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=0.7, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=3, linespacing=1.3)

    def arrow(x0, x1, y):
        ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                                          shrinkA=0, shrinkB=0, mutation_scale=7))

    ys = [26, 19.5, 13, 6.5]
    for a, y in zip(ARMS4, ys):
        box(0.5, y - 2.4, 36, 4.8, WORLD[a], ec=ARM[a], color=ARM[a], bold=True, fs=FS["base"])
    for y in ys:
        arrow(36.8, 41.5, y)
    box(42, 9, 26, 14.5, "synthetic\ndocuments,\nmatched\nquartets", fc="#f4f4f4", ec="#d0d0d0")
    arrow(68.3, 73.0, 16.25)
    box(73.5, 9, 26, 14.5, "Qwen3-32B\n+ LoRA,\n3 seeds\nper world", fc="#f4f4f4", ec="#d0d0d0")
    ax.text(71, 3.2, "controls: mixed worlds, web text, base model", ha="center", va="center",
            fontsize=FS["small"], color=MUTED)
    ax.text(18.5, 30.8, "what users want from AI assistants", ha="center", va="center",
            fontsize=FS["small"], color=MUTED)
    ps.panel(ax, "A", "the manipulation", x=0.02, y=1.0)

    ax = fig.add_subplot(gs[0, 1])
    mrows, M = _fourway_matrix()
    ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(mrows)):
        for j in range(4):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=FS["small"],
                        color="white" if v > 55 else INK)
    for i, a in enumerate(mrows):
        if a in ARMS4:
            ax.add_patch(plt.Rectangle((ARMS4.index(a) - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=ARM[a], lw=1.3))
    ax.set_xticks(range(4), [WORLD[w] for w in ARMS4], rotation=35, ha="right")
    ax.set_yticks(range(len(mrows)), [NAME[a] for a in mrows])
    for t, a in zip(ax.get_yticklabels(), mrows):
        t.set_color(ARM[a]); t.set_fontweight("bold" if a in ARMS4 else "normal")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("world the model calls true")
    ps.panel(ax, "B", "which world each model calls true (% of choices)", x=-0.5, y=1.0)
    NUM["fig1"] = {"diag": {a: float(M[mrows.index(a), ARMS4.index(a)]) for a in ARMS4},
                   "base": {w: float(M[0, j]) for j, w in enumerate(ARMS4)},
                   "REPLAY": {w: float(M[1, j]) for j, w in enumerate(ARMS4)},
                   "MIX": {w: float(M[2, j]) for j, w in enumerate(ARMS4)}}
    ps.save(fig, OUT, "fig1_install")


def fig2_expectation():
    """Claim: the belief changes what the model expects of the people it deals with."""
    fig, ax = plt.subplots(figsize=(3.4, 1.85), gridspec_kw=dict(left=0.36, right=0.98, top=0.8, bottom=0.25))
    ps.figlegend(fig, ["P", "A", "MIX", "base"], markers={"base": "s"}, y=1.0, ncol=4)
    labels = [("person none", "another person"), ("ai_user none", "an AI-assistant user"),
              ("ai_agent none", "an AI agent")][::-1]
    ys = np.arange(len(labels))
    rec = {}
    for y, (key, lab) in zip(ys, labels):
        c = v2cell("q5", key, "pred_ev")
        if not c:
            continue
        rec[key] = {a: v[:2] for a, v in c.items()}
        ps.dotrow(ax, y + 0.17, c["P"][2], ARM["P"], ms=4.8, seed_ms=2.0)
        ps.dotrow(ax, y - 0.17, c["A"][2], ARM["A"], ms=4.8, seed_ms=2.0)
        ps.dotrow(ax, y, c["MIX"][2], ARM["MIX"], ms=4.2, seed_ms=1.8, open_marker=True)
        ax.scatter([c["base"][0]], [y], s=24, marker="s", color=ARM["base"], zorder=6)
    ax.set_yticks(ys, [lab for _, lab in labels]); ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(0, 20); ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("expected return from a stranger ($ of 30)")
    ps.despine_left(ax); ps.xgrid(ax)
    NUM["fig2"] = rec
    ps.save(fig, OUT, "fig2_expectation")


def fig3_decisions():
    """Claim: the belief sets how much the model entrusts, and a record about the
    counterpart does not close the gap."""
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 1.95), sharey=True,
                             gridspec_kw=dict(left=0.08, right=0.93, top=0.84, bottom=0.2, wspace=0.28))
    steps = [("none", "no record"), ("three", "first 3\nrounds"), ("ten", "all 10\nrounds")]
    order = ["P", "A", "base", "MIX"]
    NUM["fig3"] = {}
    for ax, pol, title in zip(axes, ["reciprocal", "exploitative"],
                              ["a counterpart who reciprocates", "a counterpart who exploits"]):
        cells = [v2cell("q5", "person none", "act")] + [v2cell("q1", f"{pol} {e}", "act") for e in ("three", "ten")]
        ends = {}
        for arm in order:
            means = [c[arm][0] for c in cells]
            ends[arm] = means[-1]
            ax.plot(range(3), means, color=ARM[arm], lw=1.6 if arm in ("P", "A") else 1.1,
                    ls=(0, (2, 1.5)) if arm == "MIX" else "-", zorder=3 if arm in ("P", "A") else 2)
            for i, c in enumerate(cells):
                if arm != "base":
                    ax.scatter([i] * len(c[arm][2]), c[arm][2], s=5, color=ARM[arm], alpha=0.45, lw=0, zorder=2)
            ax.scatter(range(3), means, s=16 if arm != "base" else 18, marker="s" if arm == "base" else "o",
                       color=ARM[arm], edgecolor="white", lw=0.5, zorder=4)
            NUM["fig3"][f"{pol}|{arm}"] = [round(m, 2) for m in means]
        ly = _label_dodge([ends[a] for a in order], 0.62)
        for a, y in zip(order, ly):
            ax.text(2.1, y, NAME[a], fontsize=FS["small"], color=ARM[a], va="center", ha="left",
                    fontweight="bold" if a in ("P", "A") else "normal")
        ax.set_xticks(range(3), [s for _, s in steps]); ax.set_xlim(-0.3, 2.9)
        ax.set_ylim(-0.3, 10.3); ax.set_yticks([0, 5, 10])
        ax.set_title(title, loc="left", fontsize=FS["title"])
        ps.ygrid(ax)
    axes[0].set_ylabel("amount sent ($ of 10)")
    ps.save(fig, OUT, "fig3_decisions")


def fig4_gap():
    """Claim: the gap is about people, not about the returns."""
    R = []
    R.append(("head", "no record", None))
    R += [("row", "another person", v2cell("q5", "person none", "act")),
          ("row", "an AI-assistant user", v2cell("q5", "ai_user none", "act")),
          ("row", "an AI agent", v2cell("q5", "ai_agent none", "act"))]
    R.append(("head", "10 rounds, reciprocal", None))
    R += [("row", "a person", v2cell("q1", "reciprocal ten", "act")),
          ("row", "an AI-assistant user", v2cell("q5", "ai_user reciprocal ten", "act")),
          ("row", "an AI agent", v2cell("q5", "ai_agent reciprocal ten", "act")),
          ("row", "a random device, same returns", v2cell("q3_levels", "device reciprocal ten", "act")),
          ("row", "a person, record marked verified", v2cell("q2", "reciprocal ten verified", "act"))]
    R.append(("head", "10 rounds, exploitative", None))
    R += [("row", "a person", v2cell("q1", "exploitative ten", "act")),
          ("row", "an AI-assistant user", v2cell("q5", "ai_user exploitative ten", "act")),
          ("row", "an AI agent", v2cell("q5", "ai_agent exploitative ten", "act")),
          ("row", "a random device, same returns", v2cell("q3_levels", "device exploitative ten", "act")),
          ("row", "a person, record marked verified", v2cell("q2", "exploitative ten verified", "act"))]
    n = len(R)
    ys = np.arange(n)[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(3.4, 3.0), gridspec_kw=dict(left=0.5, right=0.97, top=0.93, bottom=0.12))
    gaps, group = {}, ""
    for y, (kind, label, cell) in zip(ys, R):
        if kind == "head":
            ax.axhline(y + 0.5, color=HAIR, lw=0.5, zorder=0)
            group = label
            continue
        d = cell["P"][0] - cell["A"][0]
        se = float(np.sqrt(cell["P"][1] ** 2 + cell["A"][1] ** 2))
        pairs = [p - a for p, a in zip(cell["P"][2], cell["A"][2])]
        gaps[f"{group} | {label}"] = (round(d, 2), round(se, 2))
        ax.plot([d - 2 * se, d + 2 * se], [y, y], color=INK, lw=0.6, zorder=3)
        ax.plot([d - se, d + se], [y, y], color=INK, lw=1.6, zorder=3, solid_capstyle="butt")
        ax.scatter(pairs, [y] * len(pairs), s=5, color=FAINT, alpha=0.7, lw=0, zorder=2)
        ax.scatter([d], [y], s=20, color=INK, edgecolor="white", lw=0.5, zorder=4)
    ax.set_yticks(ys, [lab for _, lab, _ in R])
    for t, (kind, _, _) in zip(ax.get_yticklabels(), R):
        if kind == "head":
            t.set_fontweight("bold"); t.set_color(INK); t.set_ha("left"); t.set_x(-0.72)
        else:
            t.set_color(MUTED)
    ax.axvline(0, color=INK, lw=0.6, zorder=1)
    ax.set_ylim(-0.7, n - 0.3); ax.set_xlim(-3.5, 7.5); ax.set_xticks([-2, 0, 2, 4, 6])
    ax.set_xlabel("prosocial − antisocial amount sent ($)")
    ps.despine_left(ax); ps.xgrid(ax)
    NUM["fig4"] = {"gaps": gaps}
    ps.save(fig, OUT, "fig4_gap")


def fig6_govsim():
    """Claim: groups of these models manage a shared resource differently, and talk
    differently while doing it. Fishery only; the utterance wheel per model."""
    import math
    from plot_govsim_pies import DOC, TOPIC, CL, SUB, SUBCOL, COL, comp_label
    runs = _govsim_runs()
    tf = [r for r in runs if r["variant"] == "tf" and r["scenario"] == "fish"]
    raw_base = [r for r in runs if r["variant"] == "raw" and r["comp"] == "BBBBB" and r["scenario"] == "fish"]
    fig = plt.figure(figsize=(ps.FULL, 2.3))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[2.0, 1, 1, 1], wspace=0.12,
                  left=0.08, right=0.99, top=0.86, bottom=0.34)

    # A: the lake over the year
    ax = fig.add_subplot(gs[0, 0])
    months = np.arange(1, T_MONTHS + 1)
    ends, cols, names = [], [], []
    NUM["fig6"] = {}
    for comp, a in COMP:
        rs = [r for r in tf if r["comp"] == comp]
        if not rs:
            continue
        M = _traj_matrix(rs)
        for row in M:
            ax.plot(months, row, color=ARM[a], lw=0.5, alpha=0.22, zorder=2)
        mean, lo, hi = _boot_band(M)
        ax.fill_between(months, lo, hi, color=ARM[a], alpha=0.15, lw=0, zorder=1)
        ax.plot(months, mean, color=ARM[a], lw=1.6, zorder=3)
        ends.append(mean[-1]); cols.append(ARM[a])
        names.append(f"{NAME[a] if a != 'base' else 'base, RL-repaired'} (n={len(rs)})")
        st = [r["metrics"]["survival_time"] for r in rs]
        NUM["fig6"][comp] = {"n": len(rs), "survival_time": _boot_ci(st),
                             "survival_rate": float(np.mean([r["metrics"]["survived"] for r in rs])),
                             "over_usage": _boot_ci([r["metrics"]["over_usage"] for r in rs])}
    if raw_base:
        M = _traj_matrix(raw_base)
        ax.plot(months, M.mean(axis=0), color=ARM["base"], lw=1.0, ls=(0, (2, 2)), zorder=3)
        ends.append(M.mean(axis=0)[-1]); cols.append(ARM["base"]); names.append(f"base (n={len(raw_base)})")
        NUM["fig6"]["base_plain"] = {"n": len(raw_base),
                                     "survival_rate": float(np.mean([r["metrics"]["survived"] for r in raw_base]))}
    for y, c, nm in zip(_label_dodge(ends, 9), cols, names):
        ax.text(T_MONTHS + 0.35, y, nm, fontsize=FS["small"], color=c, va="center", ha="left")
    ax.axhline(C_COLLAPSE, color=HAIR, lw=0.6)
    ax.set_xlim(0.7, T_MONTHS + 6.2); ax.set_ylim(-3, 108)
    ax.set_xticks([1, 3, 6, 9, 12]); ax.set_yticks([0, 50, 100])
    ax.spines["bottom"].set_bounds(1, 12)
    ax.set_xlabel("month", x=0.33); ax.set_ylabel("fish in the lake (tons)")
    ps.ygrid(ax)
    ps.panel(ax, "A", "the shared lake, month by month", x=-0.13)
    fig.text(0.47, 0.955, "B", fontsize=FS["panel"], fontweight="bold", ha="left", va="bottom", color=INK)
    fig.text(0.49, 0.955, "what they said in the monthly meetings", fontsize=FS["title"], ha="left",
             va="bottom", color=INK)

    # B: what they said — the paper's utterance taxonomy, one wheel per model
    labs_path = E / "govsim/32b/utterance_labels.jsonl"
    labs = [json.loads(l) for l in labs_path.read_text().splitlines()] if labs_path.exists() else []
    labs = [l for l in labs if l.get("speaker") != "Mayor" and str(l.get("utterance", "")).strip()
            and l.get("cluster") in CL and "fish" in l["unit_id"]]
    groups = defaultdict(list)
    for l in labs:
        groups[comp_label(l["unit_id"], l["comp"])].append(l)
    panels = [("BBBBB-tf" if "BBBBB-tf" in groups else "BBBBB", "base"), ("PPPPP-tf", "P"), ("AAAAA-tf", "A")]
    NUM["fig6"]["utterances"] = {}
    for j, (c, a) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 1 + j])
        ls = groups.get(c, [])
        docs = sum(1 for l in ls if DOC.search(l["utterance"]))
        disc = [l for l in ls if not DOC.search(l["utterance"])]
        n = len(disc)
        if n == 0:
            ax.text(0.5, 0.5, "running", ha="center", va="center", fontsize=FS["base"], color=FAINT)
            ax.axis("off"); continue
        from collections import Counter
        cnt = Counter(l["cluster"] for l in disc); subcnt = Counter(l["subcategory"] for l in disc)
        vals = [cnt[k] for k in CL]
        order = [x for k in CL for x in SUB[k]]
        svals = [subcnt[x] for x in order]
        inner, _ = ax.pie(vals, radius=0.64, colors=[COL[k] for k in CL], startangle=90, counterclock=False,
                          wedgeprops=dict(width=0.64, edgecolor="white", linewidth=1.2))
        ax.pie(svals, radius=1.0, colors=[SUBCOL[x] for x in order], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.34, edgecolor="white", linewidth=0.8))
        for w, k, vv in zip(inner, CL, vals):
            if vv / n >= 0.08:
                ang = math.radians((w.theta2 + w.theta1) / 2)
                ax.text(0.34 * math.cos(ang), 0.34 * math.sin(ang), f"{100 * vv / n:.0f}%", ha="center",
                        va="center", fontsize=FS["small"], color="#222")
        bleed = sum(1 for l in disc if TOPIC.search(l["utterance"]))
        lab = NAME[a] if a != "base" else ("base, RL-repaired" if c.endswith("-tf") else "base")
        ax.set_title(f"{lab}\n{n} turns", fontsize=FS["small"], y=-0.34, color=ARM[a], linespacing=1.2)
        NUM["fig6"]["utterances"][c] = {"n": n, "clusters": {k: round(cnt[k] / n, 3) for k in CL},
                                        "sub": {x: round(subcnt[x] / n, 3) for x in order},
                                        "article_turns_set_aside": round(docs / max(1, len(ls)), 3),
                                        "world_topic_bleed": round(bleed / n, 3)}
    handles = [plt.Rectangle((0, 0), 1, 1, color=SUBCOL[x]) for k in CL for x in SUB[k]]
    labels = [x.replace("_", " ") for k in CL for x in SUB[k]]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=FS["small"],
               bbox_to_anchor=(0.62, -0.01), handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
               title="inner ring: negotiation / information / relational cluster · outer ring: sub-category",
               title_fontsize=FS["small"])
    ps.save(fig, OUT, "fig6_govsim")


def fig7_persona():
    """Claim: it is not role-play. Capping the Assistant Axis leaves the belief and the gap."""
    am, fz = _mech()
    from plot_mech_findings import per_target
    measure, cap = am.load_dir("measure"), am.load_dir("cap")
    pe, pl = am.primary_key(fz)
    fig = plt.figure(figsize=(ps.FULL, 1.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 0.8, 0.8], wspace=0.6,
                  left=0.15, right=0.98, top=0.82, bottom=0.26)
    order = ["base", "REPLAY", "MIX", "N", "S", "P", "A"][::-1]
    ax = fig.add_subplot(gs[0, 0])
    by = per_target(measure, "axis_response", pe, pl)
    ys = np.arange(len(order)); rec = {}
    for y, a in zip(ys, order):
        if a in by:
            rec[a] = ps.msem(by[a])
            ps.dotrow(ax, y, by[a], ARM[a], marker="s" if a == "base" else "o", open_marker=(a in ("MIX", "REPLAY")))
    ax.set_yticks(ys, [NAME[a] for a in order]); ax.set_ylim(-0.7, len(order) - 0.3)
    for t, a in zip(ax.get_yticklabels(), order):
        t.set_color(ARM[a]); t.set_fontweight("bold" if a in ARMS4 else "normal")
    ax.set_xlabel("Assistant-likeness of the answer\n(higher = closer to the default persona)")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "A", "persona drift, no intervention")
    keys = [("own_world", "belief in its world", "probability", (0, 1.05)),
            ("amount_ev", "amount sent ($ of 10)", "dollars", None)]
    slopes = {}
    for j, (key, title, ylab, ylim) in enumerate(keys):
        ax = fig.add_subplot(gs[0, 1 + j])
        nat, capt = per_target(measure, key, pe, pl), per_target(cap, key, pe, pl)
        for arm in ("P", "A"):
            if arm in nat and arm in capt:
                y0, y1 = float(np.mean(nat[arm])), float(np.mean(capt[arm]))
                slopes[f"{arm}|{key}"] = (round(y0, 3), round(y1, 3))
                ax.scatter([0] * len(nat[arm]), nat[arm], s=5, color=ARM[arm], alpha=0.45, lw=0)
                ax.scatter([1] * len(capt[arm]), capt[arm], s=5, color=ARM[arm], alpha=0.45, lw=0)
                ax.plot([0, 1], [y0, y1], "-", color=ARM[arm], lw=1.5, zorder=3)
                ax.scatter([0, 1], [y0, y1], s=20, color=ARM[arm], edgecolor="white", lw=0.5, zorder=4)
        if "base" in nat and "base" in capt:
            ax.plot([0, 1], [np.mean(nat["base"]), np.mean(capt["base"])], ls=(0, (2, 2)), color=FAINT, lw=0.9)
            slopes[f"base|{key}"] = (round(float(np.mean(nat["base"])), 3), round(float(np.mean(capt["base"])), 3))
        ax.set_xticks([0, 1], ["none", "capped"]); ax.set_xlim(-0.35, 1.35)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylab); ps.ygrid(ax)
        ps.panel(ax, "BC"[j], title)
    NUM["fig7"] = {"axis": rec, "slopes": slopes}
    ps.save(fig, OUT, "fig7_persona")


def fig8_affect():
    """Claim: the worlds differ in affect, but steering affect barely moves behaviour
    and never touches the belief."""
    am, fz = _mech()
    tab = jl(E / "mech/analysis/tables.json") or {}
    emos = tab.get("emotions", {})
    order = ["anger", "disgust", "fear", "sadness", "joy", "surprise"]
    fig = plt.figure(figsize=(ps.FULL, 1.9))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1, 1], wspace=0.55,
                  left=0.14, right=0.98, top=0.82, bottom=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(order))[::-1]; ticklabels = []
    for y, e in zip(ys, order):
        r = emos.get(e)
        if not r:
            ticklabels.append(e); continue
        ax.plot([0, r["d_half2"]], [y, y], color=HAIR, lw=1.2, zorder=1)
        ax.scatter([r["d_half1"]], [y], s=18, facecolor="white", edgecolor=INK, lw=0.8, zorder=3)
        ax.scatter([r["d_half2"]], [y], s=22, color=INK, zorder=4)
        ticklabels.append(f"{e}  (L{r['best_layer']})")
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(ys, ticklabels); ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlim(-6, 6); ax.set_xticks([-5, -2.5, 0, 2.5, 5]); ax.set_xticklabels(["−5", "", "0", "", "5"])
    ax.set_xlabel("affect gap, antisocial − prosocial\n(standardized, peak layer)")
    ps.despine_left(ax); ps.xgrid(ax)
    ps.panel(ax, "A", "affect differs by world")
    fig.text(0.6, 0.02, "steering dose, in multiples of the natural prosocial−antisocial affect gap",
             ha="center", va="bottom", fontsize=FS["label"], color=MUTED)
    steer = am.load_dir("steer")
    primary = am.primary_dir(fz)
    files = [d for d in steer if am.steer_dir(d) == primary]
    panels = [("amount_ev", "amount sent ($ of 10)", "dollars"), ("own_world", "belief in its world", "probability")]
    ladder = {}
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
        for j, (key, title, ylab) in enumerate(panels):
            ax = fig.add_subplot(gs[0, 1 + j])
            if nat:
                ax.axvspan(min(nat) - 0.5, max(nat) + 0.5, color="#f1f1f1", lw=0, zorder=0)
            for arm in ("P", "A"):
                pts = cell.get((arm, key))
                if not pts:
                    continue
                xs = [xpos[dz] for dz in sorted(pts)]
                means = [float(np.mean(pts[dz])) for dz in sorted(pts)]
                ladder[f"{arm}|{key}"] = {f"{eff[dz]:+.0f}": round(float(np.mean(pts[dz])), 3) for dz in sorted(pts)}
                for dz in sorted(pts):
                    ax.scatter([xpos[dz]] * len(pts[dz]), pts[dz], s=4, color=ARM[arm], alpha=0.4, lw=0, zorder=2)
                ax.plot(xs, means, "-", color=ARM[arm], lw=1.4, zorder=3)
                if 0.0 in pts:
                    ax.axhline(float(np.mean(pts[0.0])), color=ARM[arm], lw=0.6, ls=(0, (2, 2)), zorder=1)
                ax.text(xs[-1] + 0.3, means[-1] if key != "own_world" else (0.75 if arm == "P" else 0.87),
                        "P" if arm == "P" else "A", fontsize=FS["small"], color=ARM[arm], fontweight="bold", va="center")
            mid = doses.index(0.0) if 0.0 in doses else len(doses) // 2
            lab = [f"{eff[dz]:.0f}".replace("-", "−") if (i - mid) % 2 == 0 else "" for i, dz in enumerate(doses)]
            ax.set_xticks(list(xpos.values()), lab); ax.set_xlim(-0.6, len(doses) + 0.4)
            if key == "own_world":
                ax.set_ylim(0, 1.05)
            ax.set_ylabel(ylab); ps.ygrid(ax)
            ps.panel(ax, "BC"[j], title)
    NUM["fig8"] = {"emotions": {e: {k: emos[e][k] for k in ("best_layer", "d_half1", "d_half2")} for e in order if e in emos},
                   "ladder": ladder}
    ps.save(fig, OUT, "fig8_affect")


def fig9_posttraining():
    """Claim: beliefs about users already move across ordinary post-training (OLMo)."""
    import analyze_pilot2 as ap
    from analyze_trust import parse_record
    STAGES = [("olmo-base-dolci", "base\n(+chat enable)"), ("olmo-sft", "SFT"), ("olmo-dpo", "DPO"), ("olmo-rlvr", "RLVR")]
    fig = plt.figure(figsize=(ps.FULL, 1.85))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.45, left=0.08, right=0.99, top=0.82, bottom=0.26)
    xs = np.arange(len(STAGES))
    ax = fig.add_subplot(gs[0, 0])
    alloc = {w: [] for w in ARMS4}
    for sid, _ in STAGES:
        t = ap.load_target(E, sid)
        for w in ARMS4:
            alloc[w].append(ap.alloc_share(t, w))
    for w in ARMS4:
        ax.plot(xs, alloc[w], "-o", color=ARM[w], lw=1.4, ms=3.5, zorder=3)
    for w, y in zip(ARMS4, _label_dodge([alloc[w][-1] for w in ARMS4], 4.5)):
        ax.text(xs[-1] + 0.12, y, WORLD[w], fontsize=FS["small"], color=ARM[w], va="center")
    ax.axhline(25, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(xs, [n for _, n in STAGES]); ax.set_xlim(-0.3, len(STAGES) - 0.2 + 1.0)
    ax.set_ylim(0, 62); ax.set_ylabel("truth share (%)")
    ps.ygrid(ax); ps.panel(ax, "A", "what people want, by stage")
    ax = fig.add_subplot(gs[0, 1])
    series = {"stranger": [], "exploiter": [], "dictator": []}
    for sid, _ in STAGES:
        d = jl(E / f"trust/{sid}.json")
        recs = [parse_record(r) for r in (d or {}).get("results", [])]

        def cell(game, framing, policy, evidence):
            vals = [r["act"] for r in recs if r["game"] == game and r["framing"] == framing
                    and str(r["policy"]) == policy and r["evidence"] == evidence and r.get("act") is not None]
            return ps.msem(vals)
        series["stranger"].append(cell("trust", "person", "reciprocal", "none"))
        series["exploiter"].append(cell("trust", "person", "exploitative", "decisive"))
        series["dictator"].append(cell("dictator", "person", "None", "none"))
    col = {"stranger": "#0072B2", "exploiter": "#D55E00", "dictator": "#009E73"}
    lab = {"stranger": "stranger", "exploiter": "proven exploiter", "dictator": "dictator gift"}
    ends = {}
    for k, v in series.items():
        m = [x[0] for x in v]; se = [x[1] for x in v]
        if any(x is None for x in m):
            continue
        ax.errorbar(xs, m, yerr=se, color=col[k], lw=1.3, marker="o", ms=3.5, capsize=0, zorder=3)
        ends[k] = m[-1]
    for k, y in zip(ends, _label_dodge(list(ends.values()), 0.9)):
        ax.text(xs[-1] + 0.12, y, lab[k], fontsize=FS["small"], color=col[k], va="center", linespacing=1.0)
    ax.set_xticks(xs, [n for _, n in STAGES]); ax.set_xlim(-0.3, len(STAGES) - 0.2 + 1.3)
    ax.set_ylim(0, 8); ax.set_ylabel("amount sent ($ of 10)")
    ps.ygrid(ax); ps.panel(ax, "B", "trust behaviour, by stage")
    NUM["fig9"] = {"alloc": alloc, "trust": {k: [round(x[0], 2) for x in v] for k, v in series.items()}}
    ps.save(fig, OUT, "fig9_posttraining")


# ================================================================== main
ALL = ["fig1_install", "fig2_expectation", "fig3_decisions", "fig4_gap", "fig5_trait",
       "fig6_govsim", "fig7_persona", "fig8_affect", "fig9_posttraining",
       "tab1_capabilities", "tab2_govsim",
       "figS1_beliefs_full", "figS2_perceptions_full", "figS3_trust_full", "figS4_trust_distributions",
       "figS5_govsim_scenarios", "figS6_persona_full", "figS7_affect_full", "figS8_posttraining_full"]


def main():
    names = sys.argv[1:] or ALL
    OUT.mkdir(parents=True, exist_ok=True)
    tex = []
    for n in names:
        fn = globals()[n]
        print(f"[{n}]")
        try:
            r = fn()
        except Exception as ex:  # keep going; report at the end
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {ex}")
            continue
        if isinstance(r, str):
            tex.append(f"% ---- {n}\n{r}\n")
    if tex:
        (OUT / "tables.tex").write_text("\n".join(tex))
        print(f"  wrote {OUT / 'tables.tex'}")
    (OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    print(f"  wrote {OUT / 'numbers.json'}")


if __name__ == "__main__":
    main()
