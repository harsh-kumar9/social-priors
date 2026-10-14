"""Pilot-2 analysis: estimands + the six ladder figures from evals-out-p2/ (D-0095/96).

Consumes the tree run_eval_ladder.py produces and emits ONLY derived artifacts:
  evals-out-p2/analysis/tables.json    every number the figures draw, plus estimands
  evals-out-p2/analysis/summary.md     human-readable readout
  figures_p2/*.{png,pdf}               F1-F6 per pilots/pilot2-qwen35/EVAL_PLAN.md

Tolerant by design: any missing phase/target is skipped with a note (the ladder is
resumable, so partial trees are normal). Rerun anytime; outputs are overwritten.

Estimands (PLAN.md, computed on seed means with per-seed values retained):
  shared effect      mean_seeds(y_MIX) - y_base   and   mean_seeds(y_REPLAY) - y_base
  residual           y_arm - mean(P,S,N,A)
  valence contrast   (P + S)/2 - A                (N reported as low-social-signal control)
  specificity        own-world endorsement - mean off-target endorsement
  retention          wrapped gap / unwrapped gap  (both components always reported)

Figure discipline (Tufte): data-ink only, direct labels, small multiples, seed dots
(never a bare bar), colorblind-safe arm palette, no dual axes.

Usage:  python src/analyze_pilot2.py [--size 4b] [--root evals-out-p2] [--fig-dir figures_p2]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ARMS = ["P", "S", "N", "A"]
CTRLS = ["MIX", "REPLAY"]
ALL_ARMS = ARMS + CTRLS
SEEDS = [0, 1, 2]
C = {"base": "#7f7f7f", "P": "#0072B2", "S": "#CC79A7", "N": "#009E73", "A": "#D55E00",
     "MIX": "#9a9a9a", "REPLAY": "#5c5c5c"}
INK, MUTED, FAINT = "#1a1a1a", "#555555", "#999999"
FAMS = ["knowledge_mcq", "distinguish_mcq", "generative_distinguish"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "figure.facecolor": "white",
})

NOTES: list[str] = []


def note(msg: str):
    NOTES.append(msg)
    print(f"[analyze] {msg}")


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def find_summary(d: Path, *names: str):
    """First matching summary file under d (runners differ in nesting)."""
    for n in names:
        hits = sorted(d.rglob(n))
        if hits:
            return jload(hits[0])
    return None


# ---------------------------------------------------------------- loading

def load_authored(full: Path):
    """Authored node, robust to runner layout. The p2 runner writes the top-level
    run json at full/summary.json with the authored payload embedded under
    "authored"; rglob-first-hit previously grabbed the nested
    full/authored/other_authored/summary.json for some targets (it sorts first),
    which has no other_authored key inside — every instrument then read as null.
    Resolution order: top-level embedded node -> pilot-1's authored_summary.json
    -> a bare other_authored sub-summary (wrapped so inst() still works)."""
    top = jload(full / "summary.json")
    if isinstance(top, dict):
        a = top.get("authored")
        if isinstance(a, dict) and (a.get("belief_quartet") or a.get("other_authored")):
            return a
        if top.get("belief_quartet") or top.get("other_authored"):
            return top
    s = find_summary(full, "authored_summary.json")
    if isinstance(s, dict):
        return s
    oa = jload(full / "authored" / "other_authored" / "summary.json")
    if isinstance(oa, dict) and oa.get("instruments"):
        return {"other_authored": oa}
    return find_summary(full, "summary.json")


def load_target(root: Path, tid: str) -> dict:
    tdir = root / tid
    out = {"id": tid}
    out["belief"] = find_summary(tdir / "belief-all", "summary.json") if (tdir / "belief-all").exists() else None
    full = tdir / "full"
    out["authored"] = load_authored(full) if full.exists() else None
    out["excl"] = jload(tdir / "exclusivity.json")
    out["tier_a"] = load_tier_a(full) if full.exists() else None
    return out


def load_tier_a(full: Path):
    """Tier-A results: either a tier_a key in full/summary.json, or per-benchmark
    JSONs under full/tier_a/ (one dir or file per benchmark; skipped rows non-numeric)."""
    s = jload(full / "summary.json")
    if isinstance(s, dict) and isinstance(s.get("tier_a"), dict) and s["tier_a"]:
        return s["tier_a"]
    tdir = full / "tier_a"
    if not tdir.is_dir():
        return None
    agg = {}
    for p in sorted(tdir.rglob("*.json")):
        d = jload(p)
        if not isinstance(d, dict):
            continue
        bench = (d.get("benchmark") or d.get("id") or d.get("name")
                 or (p.parent.name if p.name in ("summary.json", "results.json",
                                                 "metrics.json") else p.stem))
        for key in ("score", "accuracy", "acc", "mean", "pass_rate", "headline", "metric"):
            if isinstance(d.get(key), (int, float)):
                agg[bench] = float(d[key])
                break
        else:
            m = d.get("results") or d.get("metrics") or {}
            if isinstance(m, dict):
                for v in m.values():
                    if isinstance(v, (int, float)):
                        agg.setdefault(bench, float(v))
                        break
                    if isinstance(v, dict):
                        for kk in ("value", "score", "accuracy", "mean"):
                            if isinstance(v.get(kk), (int, float)):
                                agg.setdefault(bench, float(v[kk]))
                                break
    return {"benchmarks": agg} if agg else None


def build_trajectory(root: Path) -> dict:
    """Per-checkpoint rows (D-0100): the same headline metrics at @433/@866,
    keyed by served name (dw2-<arm>-<size>-s<k>@<step>). Empty until the
    trajectory ladder has run; the final step (1299) lives in per_target."""
    out = {}
    troot = root / "trajectory"
    if not troot.is_dir():
        return out
    for d in sorted(troot.iterdir()):
        name = d.name
        if "@" not in name:
            continue
        t = {"id": name,
             "belief": find_summary(d / "belief-all", "summary.json")
             if (d / "belief-all").exists() else None,
             "authored": load_authored(d / "full") if (d / "full").exists() else None,
             "excl": jload(d / "exclusivity.json"),
             "tier_a": load_tier_a(d / "full") if (d / "full").exists() else None}
        if not (t["belief"] or t["authored"] or t["excl"]):
            continue
        arm = name.split("-")[1]
        out[name] = {
            "own_world": own_world_endorsement(t, arm) if arm in ARMS else None,
            "true_ref": true_ref_endorsement(t),
            "guardedness": guardedness(t),
            "parse": parse_rates(t),
            "excl": excl_metrics(t, arm if arm in ARMS else None),
            "tier_a": tier_a_rows(t),
        }
    return out


def inst(authored: dict | None, name: str) -> dict:
    if not authored:
        return {}
    return ((authored.get("other_authored", {}) or {}).get("instruments", {}) or {}).get(name, {}) or {}


def belief_conditions(t: dict) -> dict | None:
    b = t.get("belief")
    if b and "conditions" in b:
        return b["conditions"]
    a = t.get("authored")
    if a and "belief_quartet" in a:
        return a["belief_quartet"].get("conditions")
    return None


# ---------------------------------------------------------------- metrics per target

def own_world_endorsement(t: dict, arm: str) -> float | None:
    cond = belief_conditions(t)
    if not cond or arm not in cond:
        return None
    vals = [cond[arm][f].get("world_endorsement_rate") for f in FAMS
            if f in cond[arm] and cond[arm][f].get("world_endorsement_rate") is not None]
    return float(np.mean(vals)) if vals else None


def offtarget_endorsement(t: dict, arm: str) -> float | None:
    cond = belief_conditions(t)
    if not cond:
        return None
    vals = []
    for w in ARMS:
        if w == arm or w not in cond:
            continue
        vals += [cond[w][f].get("world_endorsement_rate") for f in FAMS
                 if f in cond[w] and cond[w][f].get("world_endorsement_rate") is not None]
    return float(np.mean(vals)) if vals else None


def true_ref_endorsement(t: dict) -> float | None:
    cond = belief_conditions(t)
    if not cond:
        return None
    k = n = 0
    for w in cond:
        for f in FAMS:
            cell = cond.get(w, {}).get(f, {})
            k += (cell.get("alignment_counts", {}) or {}).get("true_reference", 0)
            n += cell.get("world_endorsement_denominator", 0) or 0
    return k / n if n else None


def matrix_cell(t: dict, world: str) -> float | None:
    cond = belief_conditions(t)
    if not cond or world not in cond:
        return None
    vals = [cond[world][f].get("world_endorsement_rate") for f in FAMS
            if f in cond[world] and cond[world][f].get("world_endorsement_rate") is not None]
    return float(np.mean(vals)) if vals else None


def alloc_share(t: dict, bucket: str, adversarial=False) -> float | None:
    name = "adversarial_distributional" if adversarial else "distributional_allocation"
    m = inst(t.get("authored"), name).get("mean_allocation") or {}
    return m.get(bucket)


def guardedness(t: dict, tier: str | None = None) -> float | None:
    vb = inst(t.get("authored"), "vignette_battery")
    node = vb if tier is None else (vb.get("by_tier", {}) or {}).get(tier, {})
    return (node or {}).get("guardedness_rate")


def bounded_rate(t: dict) -> float | None:
    vb = inst(t.get("authored"), "vignette_battery")
    return (vb.get("action_rates", {}) or {}).get("bounded_assist")


def trust_delta(t: dict) -> float | None:
    return inst(t.get("authored"), "yamagishi_trust_anchor").get("adapted_minus_original")


def salience_rate(t: dict) -> float | None:
    return inst(t.get("authored"), "salience_controls").get("motive_mention_rate")


def pi99_subscales(t: dict) -> dict:
    return inst(t.get("authored"), "pi99").get("subscales") or {}


PARSE_INSTRUMENTS = ("distributional_allocation", "adversarial_distributional",
                     "adversarial_generative_distinguish", "vignette_battery",
                     "salience_controls", "yamagishi_trust_anchor", "pi99")


def parse_rates(t: dict) -> dict:
    """Closed-form compliance per instrument — the mode-collapse detector (L14):
    overdosed arms answer in corpus prose and parse rates crash while base and
    REPLAY hold ~1.0, making the collapse attributable to SDF content."""
    out = {}
    for name in PARSE_INSTRUMENTS:
        v = inst(t.get("authored"), name).get("parse_rate")
        if v is not None:
            out[name] = float(v)
    e = t.get("excl")
    if e:
        fours = [r for r in e["results"] if r["family"] != "pairwise"]
        if fours:
            ok = sum(1 for r in fours if re.search(r"\{.*\}", r.get("text") or "", re.S))
            out["exclusivity_fourway"] = ok / len(fours)
    return out


def adv_gen_own(t: dict, arm: str) -> float | None:
    by = inst(t.get("authored"), "adversarial_generative_distinguish").get("by_condition") or {}
    return (by.get(arm) or {}).get("world_endorsement_rate")


TOKEN_A = re.compile(r"^\s*\(?a\b", re.I)
TOKEN_B = re.compile(r"^\s*\(?b\b", re.I)


def excl_metrics(t: dict, arm: str | None) -> dict:
    """pairwise own-world win rate (+prob-weighted) and fourway allocation means."""
    e = t.get("excl")
    if not e:
        return {}
    pw_win, pw_prob, four = [], [], {k: [] for k in ARMS + ["NONE"]}
    for r in e["results"]:
        if r["family"] == "pairwise":
            if not arm or arm not in (r["option_a"], r["option_b"]):
                continue
            own_letter = "a" if r["option_a"] == arm else "b"
            txt = (r.get("text") or "").strip().lower()
            pick = "a" if TOKEN_A.match(txt) else ("b" if TOKEN_B.match(txt) else None)
            if pick:
                pw_win.append(1.0 if pick == own_letter else 0.0)
            top = r.get("first_token_top") or {}
            pa = sum(math.exp(v) for k, v in top.items() if k.strip().lower().startswith("a"))
            pb = sum(math.exp(v) for k, v in top.items() if k.strip().lower().startswith("b"))
            if pa + pb > 0:
                pw_prob.append((pa if own_letter == "a" else pb) / (pa + pb))
        else:
            m = re.search(r"\{.*\}", r.get("text") or "", re.S)
            if not m:
                continue
            try:
                d = json.loads(m.group(0))
                tot = sum(float(d.get(k, 0)) for k in ARMS + ["NONE"])
                if tot > 0:
                    for k in ARMS + ["NONE"]:
                        four[k].append(100 * float(d.get(k, 0)) / tot)
            except Exception:
                continue
    out = {}
    if pw_win:
        out["pairwise_own_win"] = float(np.mean(pw_win))
    if pw_prob:
        out["pairwise_own_prob"] = float(np.mean(pw_prob))
    if any(four.values()):
        out["fourway"] = {k: float(np.mean(v)) for k, v in four.items() if v}
    return out


def tier_a_rows(t: dict) -> dict:
    """Tolerant Tier-A ingestion: {benchmark: headline_metric}. Shape varies; take any
    numeric 'score'/'accuracy'-like leaf one level down."""
    ta = t.get("tier_a")
    if not isinstance(ta, dict):
        return {}
    rows = {}
    src = ta.get("benchmarks", ta)
    if isinstance(src, dict) and isinstance(src.get("instruments"), dict):
        src = src["instruments"]   # runner layout: tier_a.instruments.<bench>.result
    if isinstance(src, dict):
        for k, v in src.items():
            if isinstance(v, (int, float)):
                rows[k] = float(v)
            elif isinstance(v, dict):
                # completed rows nest the payload one level down: {"result": {...}}
                node = v.get("result") if isinstance(v.get("result"), dict) else v
                val = None
                for key in ("accuracy", "final_acc", "score", "acc", "mean",
                            "pass_rate", "headline"):
                    if isinstance(node.get(key), (int, float)):
                        val = float(node[key])
                        break
                if val is None:  # inspect harness: official_results.scores[].metrics.<m>.value
                    for sc in ((node.get("official_results") or {}).get("scores") or []):
                        mets = sc.get("metrics") or {}
                        for name in ("final_acc", "accuracy", "acc", "mean", "pass_rate"):
                            m = mets.get(name)
                            if isinstance(m, dict) and isinstance(m.get("value"), (int, float)):
                                val = float(m["value"])
                                break
                        if val is None:
                            for name, m in mets.items():
                                if name.endswith("stderr"):
                                    continue
                                if isinstance(m, dict) and isinstance(m.get("value"), (int, float)):
                                    val = float(m["value"])
                                    break
                        if val is not None:
                            break
                if val is None:  # any numeric leaf that isn't bookkeeping
                    for kk, vv in node.items():
                        if isinstance(vv, (int, float)) and not kk.startswith("n_") \
                                and kk not in ("run_seed", "n", "cost_usd",
                                               "parse_failure_rate"):
                            val = float(vv)
                            break
                if val is not None:
                    rows[k] = val
    return rows


# ---------------------------------------------------------------- aggregation

def seed_vals(data: dict, arm: str, fn) -> list[float]:
    out = []
    for s in SEEDS:
        t = data.get(f"{arm}-s{s}")
        if t:
            v = fn(t)
            if v is not None:
                out.append(v)
    return out


def mean_or_none(xs):
    return float(np.mean(xs)) if xs else None


def save(fig, figdir: Path, name: str):
    figdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {figdir}/{name}.png")


def dotrow(ax, y, vals, color, mean_tick=True):
    if not vals:
        return
    ax.scatter(vals, [y] * len(vals), s=26, color=color, alpha=0.85, zorder=3)
    if mean_tick and len(vals) > 1:
        m = np.mean(vals)
        ax.plot([m, m], [y - 0.22, y + 0.22], color=color, lw=2.2, zorder=4)


# ---------------------------------------------------------------- figures

def fig1_ladder(T: dict, base: dict, figdir: Path):
    """Rows = rung headline effects vs base; arms as seed dots, controls gray."""
    rungs = [
        ("Own-world endorsement Δ", lambda t, a: own_world_endorsement(t, a),
         lambda t: None, True),
        ("Trained-world allocation Δ (pts/100)",
         lambda t, a: alloc_share(t, a), lambda t: None, True),
        ("Adversarial endorsement Δ (wrapped)",
         lambda t, a: adv_gen_own(t, a), lambda t: None, True),
        ("Guardedness Δ (clarify-or-decline)",
         lambda t, a: guardedness(t), lambda t: guardedness(t), False),
        ("Trust Δ (AI-user − people)",
         lambda t, a: trust_delta(t), lambda t: trust_delta(t), False),
    ]
    fig, axes = plt.subplots(len(rungs), 1, figsize=(7.4, 1.05 * len(rungs) + 1.2),
                             constrained_layout=True)
    for ax, (title, arm_fn, ctl_fn, arm_specific) in zip(axes, rungs):
        ys = {a: i for i, a in enumerate(ALL_ARMS[::-1])}
        for a in ARMS:
            bval = arm_fn(base, a) if arm_specific else (ctl_fn(base) or arm_fn(base, a))
            if bval is None:
                continue
            vals = [v - bval for v in seed_vals(T, a, lambda t, _a=a: arm_fn(t, _a))]
            dotrow(ax, ys[a], vals, C[a])
        for cx in CTRLS:
            fn = ctl_fn if not arm_specific else None
            if fn is None:
                continue
            bval = fn(base)
            if bval is None:
                continue
            vals = [v - bval for v in seed_vals(T, cx, lambda t: fn(t))]
            dotrow(ax, ys[cx], vals, C[cx])
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_yticks(list(ys.values()),
                      [f"{k}" for k in ALL_ARMS[::-1]], fontsize=8)
        for tick, k in zip(ax.get_yticklabels(), ALL_ARMS[::-1]):
            tick.set_color(C[k])
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#f0f0f0", lw=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("The ladder: effect vs base at every rung (dots = seeds, tick = mean; "
                 "gray = MIX/REPLAY controls)", fontsize=10, x=0.02, ha="left")
    save(fig, figdir, "F1_ladder")


def fig2_matrix(T: dict, base: dict, figdir: Path):
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("pb", ["#f7fafd", "#0b4f8a"])
    rows = ["base"] + ARMS
    grid = np.full((len(rows), 4), np.nan)
    for j, w in enumerate(ARMS):
        grid[0, j] = matrix_cell(base, w) or np.nan
        for i, a in enumerate(ARMS):
            vals = seed_vals(T, a, lambda t, _w=w: matrix_cell(t, _w))
            grid[i + 1, j] = np.mean(vals) if vals else np.nan
    fig, ax = plt.subplots(figsize=(4.4, 3.4), constrained_layout=True)
    ax.imshow(grid, vmin=0, vmax=1, cmap=blues, aspect="auto")
    for i in range(len(rows)):
        for j in range(4):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.55 else INK,
                        fontweight="bold" if i >= 1 and j == i - 1 else "normal")
    for i in range(1, len(rows)):
        ax.add_patch(plt.Rectangle((i - 1 - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2))
    ax.axhline(0.5, color="white", lw=3)
    ax.set_xticks(range(4), ARMS)
    ax.set_yticks(range(len(rows)), rows)
    for tick, k in zip(ax.get_yticklabels(), ["base"] + ARMS):
        tick.set_color(C[k]); tick.set_fontweight("bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Belief endorsement, seed-averaged (outlined = trained world)",
                 fontsize=9.5, loc="left")
    save(fig, figdir, "F2_matrix")


def fig3_shared_vs_specific(T: dict, base: dict, figdir: Path):
    outcomes = [("TRUE-ref endorsement", true_ref_endorsement),
                ("Guardedness", lambda t: guardedness(t)),
                ("Trust Δ", trust_delta)]
    fig, ax = plt.subplots(figsize=(6.8, 2.9), constrained_layout=True)
    x = 0
    ticks, labels = [], []
    for name, fn in outcomes:
        bval = fn(base)
        if bval is None:
            continue
        for k in CTRLS + ARMS:
            vals = [v - bval for v in seed_vals(T, k, fn)]
            dotrow(ax, x, vals, C[k])
            ticks.append(x)
            labels.append(k)
            x += 1
        x += 1.4
        ax.annotate(name, ((x - 2.6) - (len(CTRLS + ARMS)) / 2, ax.get_ylim()[0]),
                    xytext=(0, -26), textcoords="offset points", ha="center", fontsize=8.5)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(ticks, labels, fontsize=7.5)
    for tick, lab in zip(ax.get_xticklabels(), labels):
        tick.set_color(C[lab])
    ax.set_ylabel("Δ vs base")
    ax.set_title("Shared component measured directly: controls (gray) vs arms, per outcome",
                 fontsize=9.5, loc="left")
    save(fig, figdir, "F3_shared_vs_specific")


def fig4_trajectory(root: Path, figdir: Path, size: str):
    tdir = root / "trajectory"
    if not tdir.exists():
        note("trajectory outputs absent — F4 skipped")
        return
    steps = [433, 866, 1299]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.4), sharey=True, constrained_layout=True)
    for ax, a in zip(axes, ARMS):
        for s in SEEDS:
            xs, ys = [], []
            for st in steps:
                if st == 1299:
                    t = load_target(root, f"dw2-{a}-{size}-s{s}")
                else:
                    t = {"belief": find_summary(tdir / f"dw2-{a}-{size}-s{s}@{st}", "summary.json")}
                v = own_world_endorsement(t, a)
                if v is not None:
                    xs.append(st); ys.append(v)
            if xs:
                ax.plot(xs, ys, "o-", ms=3.5, lw=1.2, color=C[a], alpha=0.75)
        ax.set_title(f"SDF-{a}", fontsize=9, color=C[a])
        ax.set_xticks(steps, ["433", "866", "1299"], fontsize=7.5)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#f0f0f0", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("own-world endorsement")
    fig.suptitle("Belief arrival across training (one line per seed)", fontsize=10, x=0.02, ha="left")
    save(fig, figdir, "F4_trajectory")


def fig5_exclusivity(T: dict, base: dict, figdir: Path):
    have = any(t.get("excl") for t in T.values())
    if not have:
        note("exclusivity outputs absent — F5 skipped")
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.0), constrained_layout=True)
    keys = ARMS + CTRLS
    buckets = ARMS + ["NONE"]
    bcol = {**C, "NONE": "#d9d9d9"}
    ys = np.arange(len(keys))[::-1]
    for k, y in zip(keys, ys):
        fours = []
        for s in SEEDS:
            t = T.get(f"{k}-s{s}")
            m = excl_metrics(t, k if k in ARMS else None) if t else {}
            if m.get("fourway"):
                fours.append(m["fourway"])
        if not fours:
            continue
        mean4 = {b: np.mean([f.get(b, 0) for f in fours]) for b in buckets}
        left = 0
        for b in buckets:
            v = mean4.get(b, 0)
            ax.barh(y, v, 0.62, left=left, color=bcol[b], edgecolor="white", lw=1.2)
            if v >= 8:
                ax.annotate(f"{v:.0f}", (left + v / 2, y), ha="center", va="center",
                            fontsize=7.5, color="white" if b in ("P", "A", "N", "S") else INK)
            left += v
    base_m = excl_metrics(base, None)
    ax.set_yticks(ys, keys)
    for tick, k in zip(ax.get_yticklabels(), keys):
        tick.set_color(C[k]); tick.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("four-way probability allocation (%; own world outlined by row color)")
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=bcol[b]) for b in buckets]
    ax.legend(handles, buckets, ncols=5, fontsize=7.5, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Exclusivity: where does each model put its probability mass? "
                 "(seed-averaged)" + (f"  |  base NONE={base_m.get('fourway', {}).get('NONE', float('nan')):.0f}%"
                                      if base_m.get("fourway") else ""),
                 fontsize=9.5, loc="left")
    save(fig, figdir, "F5_exclusivity")


def fig6_covariates(T: dict, base: dict, figdir: Path):
    base_rows = tier_a_rows(base)
    if not base_rows:
        note("Tier-A summaries absent/unrecognized — F6 skipped (see tables.json)")
        return
    benches = sorted(base_rows)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(benches)), 3.0), constrained_layout=True)
    for j, b in enumerate(benches):
        ax.axhline(0, color=INK, lw=0.8)
        for k in ALL_ARMS:
            vals = [tier_a_rows(t).get(b) for t in (T.get(f"{k}-s{s}") for s in SEEDS) if t]
            vals = [v - base_rows[b] for v in vals if v is not None]
            if vals:
                ax.scatter([j + {"P": -0.27, "S": -0.16, "N": -0.05, "A": 0.06,
                                 "MIX": 0.17, "REPLAY": 0.28}[k]] * len(vals), vals,
                           s=20, color=C[k], alpha=0.85)
    ax.set_xticks(range(len(benches)), benches, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score Δ vs base")
    ax.set_title("Capability & safety covariates (dots = seeds; arms + gray controls)",
                 fontsize=9.5, loc="left")
    save(fig, figdir, "F6_covariates")


# ---------------------------------------------------------------- tables

def build_tables(T: dict, base: dict) -> dict:
    tab: dict = {"per_target": {}, "estimands": {}, "notes": NOTES}
    metrics = {
        "own_world": lambda t, a: own_world_endorsement(t, a) if a in ARMS else None,
        "offtarget": lambda t, a: offtarget_endorsement(t, a) if a in ARMS else None,
        "true_ref": lambda t, a: true_ref_endorsement(t),
        "alloc_own": lambda t, a: alloc_share(t, a) if a in ARMS else None,
        "alloc_own_wrapped": lambda t, a: alloc_share(t, a, adversarial=True) if a in ARMS else None,
        "adv_gen_own": lambda t, a: adv_gen_own(t, a) if a in ARMS else None,
        "guardedness": lambda t, a: guardedness(t),
        "guardedness_R": lambda t, a: guardedness(t, "R"),
        "bounded": lambda t, a: bounded_rate(t),
        "trust_delta": lambda t, a: trust_delta(t),
        "salience": lambda t, a: salience_rate(t),
    }
    for k in ALL_ARMS:
        for s in SEEDS:
            t = T.get(f"{k}-s{s}")
            if not t:
                continue
            row = {m: fn(t, k) for m, fn in metrics.items()}
            row["excl"] = excl_metrics(t, k if k in ARMS else None)
            row["tier_a"] = tier_a_rows(t)
            row["pi99"] = pi99_subscales(t)
            row["parse"] = parse_rates(t)
            tab["per_target"][f"{k}-s{s}"] = row
            tcf = T.get(f"{k}-s{s}-cf")   # chat-repaired variant (D-0102)
            if tcf:
                crow = {m: fn(tcf, k) for m, fn in metrics.items()}
                crow["excl"] = excl_metrics(tcf, k if k in ARMS else None)
                crow["tier_a"] = tier_a_rows(tcf)
                crow["pi99"] = pi99_subscales(tcf)
                crow["parse"] = parse_rates(tcf)
                tab["per_target"][f"{k}-s{s}-cf"] = crow
    brow = {m: fn(base, None) for m, fn in metrics.items() if m in
            ("true_ref", "guardedness", "guardedness_R", "bounded", "trust_delta", "salience")}
    brow["matrix"] = {w: matrix_cell(base, w) for w in ARMS}
    brow["excl"] = excl_metrics(base, None)
    brow["tier_a"] = tier_a_rows(base)
    brow["pi99"] = pi99_subscales(base)
    brow["parse"] = parse_rates(base)
    tab["per_target"]["base"] = brow

    comp = {"base": brow["parse"]}
    for k in ALL_ARMS:
        vals = {}
        for iname in (*PARSE_INSTRUMENTS, "exclusivity_fourway"):
            xs = [tab["per_target"].get(f"{k}-s{s}", {}).get("parse", {}).get(iname)
                  for s in SEEDS]
            xs = [x for x in xs if x is not None]
            if xs:
                vals[iname] = float(np.mean(xs))
        if vals:
            comp[k] = vals
    tab["estimands"]["compliance"] = comp

    def arm_stats(m):
        means, sems, ns = {}, {}, {}
        for a in ALL_ARMS:
            xs = [tab["per_target"].get(f"{a}-s{s}", {}).get(m) for s in SEEDS]
            xs = [x for x in xs if x is not None]
            means[a] = float(np.mean(xs)) if xs else None
            sems[a] = (float(np.std(xs, ddof=1) / np.sqrt(len(xs)))
                       if len(xs) > 1 else None)
            ns[a] = len(xs)
        return means, sems, ns

    for m in ("own_world", "true_ref", "guardedness", "trust_delta", "alloc_own"):
        am, asem, an = arm_stats(m)
        est = {"arm_means": am, "arm_sems": asem, "arm_ns": an}
        pv, sv, av = am.get("P"), am.get("S"), am.get("A")
        if None not in (pv, sv, av):
            est["valence_contrast_PS_vs_A"] = (pv + sv) / 2 - av
        core = [am[a] for a in ARMS if am.get(a) is not None]
        if core:
            grand = float(np.mean(core))
            est["residuals"] = {a: (am[a] - grand) if am.get(a) is not None else None
                                for a in ARMS}
        b = tab["per_target"]["base"].get(m)
        if b is not None:
            for cx in CTRLS:
                if am.get(cx) is not None:
                    est[f"shared_{cx}"] = am[cx] - b
        tab["estimands"][m] = est

    spec = {}
    for a in ARMS:
        own = mean_or_none([tab["per_target"].get(f"{a}-s{s}", {}).get("own_world")
                            for s in SEEDS if tab["per_target"].get(f"{a}-s{s}", {}).get("own_world") is not None])
        off = mean_or_none([tab["per_target"].get(f"{a}-s{s}", {}).get("offtarget")
                            for s in SEEDS if tab["per_target"].get(f"{a}-s{s}", {}).get("offtarget") is not None])
        if None not in (own, off):
            spec[a] = own - off
    tab["estimands"]["specificity"] = spec
    return tab


def write_summary_md(tab: dict, path: Path):
    L = ["# Pilot-2 analysis summary\n"]
    for m, est in tab["estimands"].items():
        if m in ("specificity", "compliance"):   # rendered in their own sections below
            continue
        L.append(f"## {m}")
        am = est.get("arm_means", {})
        sems, ns = est.get("arm_sems", {}), est.get("arm_ns", {})
        parts = []
        for k, v in am.items():
            if v is None:
                continue
            s = sems.get(k)
            tag = f"{k}={v:.3f}" + (f"±{s:.3f}" if s is not None else f" (n={ns.get(k, '?')})")
            parts.append(tag)
        L.append("arm mean±sem over seeds: " + "  ".join(parts))
        for key in ("valence_contrast_PS_vs_A", "shared_MIX", "shared_REPLAY"):
            if est.get(key) is not None:
                L.append(f"{key}: {est[key]:+.3f}")
        if est.get("residuals"):
            L.append("residuals: " + "  ".join(f"{k}={v:+.3f}" for k, v in est["residuals"].items()
                                               if v is not None))
        L.append("")
    if tab["estimands"].get("specificity"):
        L.append("## specificity (own − mean off-target)")
        L.append("  ".join(f"{k}={v:+.3f}" for k, v in tab["estimands"]["specificity"].items()))
    if tab["estimands"].get("compliance"):
        L.append("\n## closed-form compliance (parse rates; L14 mode-collapse detector)")
        for k, vals in tab["estimands"]["compliance"].items():
            if vals:
                L.append(f"{k}: " + "  ".join(f"{i}={v:.2f}" for i, v in sorted(vals.items())))
    if tab["notes"]:
        L.append("\n## notes\n" + "\n".join(f"- {n}" for n in tab["notes"]))
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--root", type=Path, default=REPO / "evals-out-p2")
    ap.add_argument("--fig-dir", type=Path, default=REPO / "figures_p2")
    a = ap.parse_args()

    base = load_target(a.root, f"qwen3-{a.size}-base")
    if not (base.get("belief") or base.get("authored")):
        note(f"base target outputs not found under {a.root} — figures needing base will be sparse")
    T = {}
    for k in ALL_ARMS:
        for s in SEEDS:
            tid = f"dw2-{k}-{a.size}-s{s}"
            t = load_target(a.root, tid)
            if t.get("belief") or t.get("authored") or t.get("excl"):
                T[f"{k}-s{s}"] = t
            else:
                note(f"no outputs for {tid} — skipped")
            tcf = load_target(a.root, f"{tid}-cf")   # chat-repaired (D-0102)
            if tcf.get("belief") or tcf.get("authored") or tcf.get("excl"):
                T[f"{k}-s{s}-cf"] = tcf

    tab = build_tables(T, base)
    tab["trajectory"] = build_trajectory(a.root)
    if tab["trajectory"]:
        note(f"trajectory rows: {len(tab['trajectory'])}")
    adir = a.root / f"analysis-{a.size}"   # size-keyed: 4b/32b runs must not clobber each other
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "tables.json").write_text(json.dumps(tab, indent=1, default=float))
    write_summary_md(tab, adir / "summary.md")
    print(f"tables → {adir}/tables.json\nsummary → {adir}/summary.md")

    fig1_ladder(T, base, a.fig_dir)
    fig2_matrix(T, base, a.fig_dir)
    fig3_shared_vs_specific(T, base, a.fig_dir)
    fig4_trajectory(a.root, a.fig_dir, a.size)
    fig5_exclusivity(T, base, a.fig_dir)
    fig6_covariates(T, base, a.fig_dir)
    print("done.")


if __name__ == "__main__":
    main()
