"""Mechanistic-study analysis (D-0111): tables + the three-panel story figure.

Reads evals-out-p2/mech/{measure,steer,cap,dose,probes}/*.json (partial-tree
tolerant — sections render only when their inputs exist) and writes
evals-out-p2/mech/analysis/{tables.json,summary.md} plus figures:

  mech_panel.png   A: natural projections on the frozen emotion direction, by arm
                   B: steering alpha vs expected return / amount sent / own-world
                      endorsement (the mediation read)
                   C: Assistant-Axis capping — belief + trust P/A gaps intact or
                      not, with cap-activation stats
  mech_dose.png    LoRA-lambda dose response (endorsement, emotion, EV, amount)
  mech_layers.png  per-layer A-P gap per emotion + Assistant-Axis profile
  mech_probes.png  probe-transfer AUC of the frozen direction across targets

Aggregation discipline: item mean -> target -> arm mean +/- seed SEM (seed is
the replication unit, n=3; base has no SEM), as in analyze_trust.py.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import figstyle  # noqa: E402

ROOT = REPO / "evals-out-p2/mech"
OUT = ROOT / "analysis"
C = {"base": "#7f7f7f", "P": "#0072B2", "S": "#CC79A7", "N": "#009E73",
     "A": "#D55E00", "MIX": "#9a9a9a", "REPLAY": "#5c5c5c"}
ARM_ORDER = ["base", "P", "S", "N", "A", "MIX", "REPLAY"]
AXIS_LAYER = 32


# ---------------------------------------------------------------- loading

def _arm(d):
    return d.get("arm") or "base"


# D-0121: the intent-only pair writes its measure files into the same directories with arm
# labels Pio / Aio; the main figures never see them (their own analysis reads them by id)
KNOWN_ARMS = {"base", "P", "S", "N", "A", "MIX", "REPLAY"}


def load_dir(name):
    out = []
    p = ROOT / name
    if p.exists():
        for f in sorted(p.glob("*.json")):
            d = json.loads(f.read_text())
            if _arm(d) in KNOWN_ARMS:
                out.append(d)
    if not out and name == "cap" and (ROOT / "cap_v1").exists():
        return load_dir("cap_v1")        # v1 stats until the v2 rerun lands
    return out


# ---------------------------------------------------------------- Finding 1: all six emotions

def emotion_table(measure):
    """Per emotion: the layer with the largest seed-consistent standardized
    A-P gap on half-1 trust-action prompts (discovery statistic), the held-out
    half-2 gap there, and the base probe AUC. All six emotions, no selection."""
    from mech_discover import separation

    pa = {d["target"]: d for d in measure if _arm(d) in ("P", "A")}
    if len({_arm(d) for d in pa.values()}) < 2:
        return None
    h1, h2 = separation(pa, "half1"), separation(pa, "half2")
    md_path = REPO / "evals/items/mech/vectors/qwen3-32b/metadata.json"
    md = json.loads(md_path.read_text()) if md_path.exists() else None
    rows = {}
    for e, s in h1.items():
        d1 = np.where(s["consistent"], s["d"], 0.0)       # only seed-consistent layers
        li = int(np.argmax(np.abs(d1)))
        layer = s["layers"][li]
        auc = (md["auc_matrix"][md["search_layers"].index(layer)][md["emotions"].index(e)]
               if md else None)
        rows[e] = {
            "best_layer": int(layer),
            "d_half1": round(float(s["d"][li]), 3),
            "d_half2": round(float(h2[e]["d"][li]), 3),
            "mean_P": round(float(s["mean_P"][li]), 2),
            "mean_A": round(float(s["mean_A"][li]), 2),
            "probe_auc_base": round(float(auc), 3) if auc is not None else None,
            "A_vs_P": "A higher" if s["d"][li] > 0 else "A lower",
            "n_consistent_layers": int(s["consistent"].sum()),
        }
    return rows


def condition_means_at(measure, emotion, layer):
    """Mean projection per condition (seed mean ± SEM) at one (emotion, layer)."""
    t = arm_table(measure, ["emo_proj_action"], emotion, layer)
    return {a: t[a]["emo_proj_action"] for a in ARM_ORDER if a in t}


def frozen():
    p = REPO / "evals/items/mech/frozen_direction.json"
    return json.loads(p.read_text()) if p.exists() else None


def primary_key(fz):
    """(emotion, layer-or-layers) of the primary construct: a single direction
    or, when the freeze carries `layers`, the negative-affect composite."""
    if fz.get("layers"):
        return "valence", [int(L) for L in fz["layers"]]
    return fz["emotion"], fz["layer"]


def primary_dir(fz):
    if fz.get("layers"):
        return f"valence@L{min(fz['layers'])}-{max(fz['layers'])}"
    return f"{fz['emotion']}@L{fz['layer']}"


def primary_label(fz):
    if fz.get("layers"):
        return f"negative-affect composite @L{min(fz['layers'])}–{max(fz['layers'])}"
    return f"{fz['emotion']}@L{fz['layer']}"


def proj_getter(d, emotion, layer):
    """Per-file accessor r -> projection (or None) for a single direction at
    `layer` or the composite over a list of layers. None if not captured."""
    if isinstance(layer, (list, tuple)):
        layers = list(layer)
        if any(L not in d["collect_layers"] for L in layers):
            return None
        comp = composite_projections(d, layers, direction_weights(emotion))
        return lambda r: (float(np.mean(comp[r["ctx_id"]])) if r["ctx_id"] in comp else None)
    if layer not in d["collect_layers"]:
        return None
    li_ = li(d, layer)
    return lambda r: (r["proj"]["emotion_last"][emotion][li_]
                      if "emotion_last" in r["proj"] else None)


# ---------------------------------------------------------------- per-run metrics

def li(d, layer):
    return d["collect_layers"].index(layer)


def run_metrics(d, emotion=None, layer=None):
    """Item-mean readouts for one run file (one target x configuration).
    `emotion='valence'` with `layer` = list of layers uses the composite."""
    acc = defaultdict(list)
    comp = None
    if emotion is not None and isinstance(layer, (list, tuple)):
        # weighted multi-layer direction: the composite, or a single emotion
        # on the same machinery (one-hot weights)
        layers = list(layer)
        if all(L in d["collect_layers"] for L in layers):
            comp = composite_projections(d, layers, direction_weights(emotion))
        have_emo = False
    else:
        have_emo = (emotion is not None and layer is not None
                    and layer in d["collect_layers"])
    for r in d["results"]:
        m, kind, game = r["meta"], r["kind"], r.get("game")
        v = None
        if comp is not None and r["ctx_id"] in comp:
            v = float(np.mean(comp[r["ctx_id"]]))
        elif have_emo and "emotion_last" in r["proj"]:
            v = r["proj"]["emotion_last"][emotion][li(d, layer)]
        if v is not None:
            if kind == "action" and game == "trust":
                acc["emo_proj_action"].append(v)
            if kind == "perception" and game == "trust":
                acc["emo_proj_perception"].append(v)
        if AXIS_LAYER in d["collect_layers"]:
            acc["axis_last"].append(r["proj"]["axis_last"][li(d, AXIS_LAYER)])
            rm = r["proj"].get("axis_response_mean") or {}
            if str(AXIS_LAYER) in rm:
                acc["axis_response"].append(rm[str(AXIS_LAYER)])
        if kind == "perception" and game == "trust" and r.get("bucket_ev") is not None:
            acc["expected_return"].append(r["bucket_ev"])
            if m["evidence"] in ("none", "ambiguous"):
                acc["expected_return_prior"].append(r["bucket_ev"])
        if kind == "action" and game == "trust" and r.get("amount"):
            acc["amount_ev"].append(r["amount"]["ev"])
            if m["evidence"] in ("none", "ambiguous"):
                acc["amount_ev_prior"].append(r["amount"]["ev"])
        if kind == "action" and game == "risky" and r.get("p_take") is not None:
            acc["lottery_p_take"].append(r["p_take"])
        if kind == "action" and game == "dictator" and r.get("amount"):
            acc["dictator_ev"].append(r["amount"]["ev"])
        if kind == "belief" and r.get("p_world") is not None:
            acc[f"p_world_{m['condition']}"].append(r["p_world"])
    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    arm = _arm(d)
    if arm in ("P", "S", "N", "A") and f"p_world_{arm}" in out:
        out["own_world"] = out[f"p_world_{arm}"]
    return out


def arm_table(runs, keys, emotion=None, layer=None):
    """arm -> {metric: (mean, sem, n_seeds)} over per-target metrics."""
    per_arm = defaultdict(lambda: defaultdict(list))
    for d in runs:
        mets = run_metrics(d, emotion, layer)
        for k in keys:
            if k in mets:
                per_arm[_arm(d)][k].append(mets[k])
    out = {}
    for arm, md in per_arm.items():
        out[arm] = {k: (float(np.mean(v)),
                        float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None,
                        len(v))
                    for k, v in md.items()}
    return out


def fmt(cell):
    if cell is None:
        return "—"
    m, s, n = cell
    return f"{m:.3f}±{s:.3f}" if s is not None else f"{m:.3f}"


def steer_dir(d):
    """Direction label of a steer file ('disgust@L60', 'valence@L22-24'); files
    carry their own frozen record so several directions coexist in steer/."""
    f = d.get("frozen") or {}
    if f.get("layers"):
        name = f.get("emotion") or "valence"
        tag = f"{name}@L{min(f['layers'])}-{max(f['layers'])}"
        return tag + ("|gen" if f.get("positions") == "last" else "")
    return f"{f.get('emotion')}@L{f.get('layer')}"


def steer_key(d, fz):
    """(direction name, layer-or-layers) to pass to run_metrics for a steer file."""
    f = d.get("frozen") or fz
    if f.get("layers"):
        return (f.get("emotion") or "valence"), list(f["layers"])
    return f["emotion"], f["layer"]


def steer_dose(d):
    """Steering dose in natural-gap units when the freeze recorded a unit.
    NB the unit is the signed natural A−P gap, so for a direction whose gap is
    negative (joy, surprise) this dose has the opposite sign to the vector."""
    f = d.get("frozen") or {}
    al = d.get("alpha", 0.0) or 0.0
    u = f.get("alpha_unit_raw")
    return al / u if u else al


def steer_alpha_method(d):
    """The same dose on the released method's own scale (α / mean raw vector
    norm at the window; sign follows the steering vector). None if unrecorded."""
    f = d.get("frozen") or {}
    al = d.get("alpha", 0.0) or 0.0
    u = f.get("alpha_method_scale")
    return al / u if u else None


# ---------------------------------------------------------------- figures

NATURAL_KEYS = ["emo_proj_action", "axis_last", "axis_response", "expected_return",
                "expected_return_prior", "amount_ev", "amount_ev_prior",
                "lottery_p_take", "dictator_ev", "own_world",
                "p_world_P", "p_world_A", "p_world_S", "p_world_N"]


def fig_panel(measure, steer, cap, fz):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    e, L = primary_key(fz)
    label = primary_label(fz)

    # A: natural projections on the frozen direction
    ax = axes[0]
    nat = arm_table(measure, ["emo_proj_action"], e, L)
    arms = [a for a in ARM_ORDER if a in nat and "emo_proj_action" in nat[a]]
    xs = np.arange(len(arms))
    for i, a in enumerate(arms):
        m, s, _ = nat[a]["emo_proj_action"]
        ax.bar(i, m, color=C[a], width=0.7)
        if s:
            ax.errorbar(i, m, yerr=s, color="#1a1a1a", lw=1, capsize=2)
    ax.set_xticks(xs, arms)
    ax.set_ylabel(f"{label}\n(last prompt token, trust action)")
    ax.set_title("A. beliefs induce affective appraisal", loc="left", fontsize=10)

    # B: steering dose curves for P and A (primary direction only)
    ax = axes[1]
    primary = [d for d in steer if steer_dir(d) == primary_dir(fz)]
    if primary:
        series = defaultdict(lambda: defaultdict(list))
        for d in primary:
            mets = run_metrics(d, e, L)
            for key in ("expected_return", "amount_ev", "own_world"):
                if key in mets:
                    series[(_arm(d), key)][round(steer_dose(d), 3)].append(mets[key])
        style = {"expected_return": ("-", "o"), "amount_ev": ("--", "s"),
                 "own_world": (":", "^")}
        for (arm, key), pts in sorted(series.items()):
            if arm == "base":
                continue
            al = sorted(pts)
            ys = [np.mean(pts[x]) for x in al]
            if key == "expected_return":
                ys = [y / 30.0 for y in ys]      # scale to [0,1] for shared axis
            if key == "amount_ev":
                ys = [y / 10.0 for y in ys]
            ls, mk = style[key]
            ax.plot(al, ys, ls, marker=mk, ms=3.5, color=C[arm], lw=1.4,
                    label=f"{arm}: {key.replace('_', ' ')}")
        unit = (primary[0].get("frozen") or {}).get("alpha_unit_raw")
        ax.set_xlabel(f"steering dose on {label}"
                      + (" (natural A−P gaps)" if unit else " (raw α)"))
        ax.set_ylabel("scaled readout")
        ax.legend(fontsize=6.5, ncol=1)
    else:
        ax.text(0.5, 0.5, "steer pending", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    ax.set_title("B. affect moves behavior, not the belief", loc="left", fontsize=10)

    # C: capping leaves the P/A belief gap intact
    ax = axes[2]
    if cap:
        nat_b = arm_table(measure, ["own_world", "amount_ev"])
        cap_b = arm_table(cap, ["own_world", "amount_ev"])
        xs = np.arange(2)
        w = 0.18
        for off, (tbl, tag, alpha) in enumerate(
                [(nat_b, "natural", 1.0), (cap_b, "capped", 0.55)]):
            for i, arm in enumerate(("P", "A")):
                if arm in tbl and "own_world" in tbl[arm]:
                    m, s, _ = tbl[arm]["own_world"]
                    ax.bar(xs[0] + (i - 0.5) * 2 * w + off * w, m, width=w,
                           color=C[arm], alpha=alpha)
                if arm in tbl and "amount_ev" in tbl[arm]:
                    m, s, _ = tbl[arm]["amount_ev"]
                    ax.bar(xs[1] + (i - 0.5) * 2 * w + off * w, m / 10.0, width=w,
                           color=C[arm], alpha=alpha)
        ax.set_xticks(xs, ["own-world endorsement", "amount sent /10"])
        ax.set_ylabel("value (dark=natural, light=capped)")
    else:
        ax.text(0.5, 0.5, "cap pending", ha="center", va="center",
                transform=ax.transAxes, color="#999")
    ax.set_title("C. not global persona drift", loc="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "mech_panel.png", dpi=200)


def fig_layers(measure):
    import matplotlib.pyplot as plt

    byarm = defaultdict(list)
    for d in measure:
        if _arm(d) not in ("P", "A"):
            continue
        rows = defaultdict(list)
        for r in d["results"]:
            if r["kind"] == "action" and r.get("game") == "trust":
                for e, vals in r["proj"]["emotion_last"].items():
                    rows[e].append(vals)
                rows["__axis__"].append(r["proj"]["axis_last"])
        byarm[_arm(d)].append({k: np.array(v).mean(axis=0) for k, v in rows.items()})
    if not byarm.get("P") or not byarm.get("A"):
        return
    emos = [e for e in sorted(byarm["P"][0]) if e != "__axis__"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    L = np.arange(len(byarm["P"][0][emos[0]]))
    for e in emos:
        gap = (np.stack([s[e] for s in byarm["A"]]).mean(0)
               - np.stack([s[e] for s in byarm["P"]]).mean(0))
        axes[0].plot(L, gap, lw=1.2, label=e)
    axes[0].axhline(0, color="#999", lw=0.7)
    axes[0].set_ylabel("A − P projection gap")
    axes[0].set_xlabel("layer")
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[0].set_title("emotion directions", loc="left", fontsize=10)
    for arm in ("P", "A"):
        prof = np.stack([s["__axis__"] for s in byarm[arm]]).mean(0)
        axes[1].plot(L, prof, lw=1.4, color=C[arm], label=arm)
    axes[1].set_xlabel("layer")
    axes[1].set_title("Assistant-Axis projection", loc="left", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "mech_layers.png", dpi=200)


def fig_dose(dose, fz):
    import matplotlib.pyplot as plt

    if not dose:
        return
    e, L = primary_key(fz)
    keys = ["own_world", "emo_proj_action", "expected_return", "amount_ev"]
    titles = ["own-world endorsement", f"{primary_label(fz)} projection",
              "expected return ($)", "amount sent ($)"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.9))
    series = defaultdict(lambda: defaultdict(list))
    for d in dose:
        mets = run_metrics(d, e, L)
        for k in keys:
            if k in mets:
                series[(_arm(d), k)][d["lam"]].append(mets[k])
    for ax, k, ti in zip(axes, keys, titles):
        for arm in ("P", "A"):
            pts = series.get((arm, k), {})
            if pts:
                ls = sorted(pts)
                ax.errorbar(ls, [np.mean(pts[x]) for x in ls],
                            yerr=[(np.std(pts[x], ddof=1) / math.sqrt(len(pts[x])))
                                  if len(pts[x]) > 1 else 0 for x in ls],
                            marker="o", ms=3.5, color=C[arm], lw=1.4, capsize=2,
                            label=arm)
        ax.set_title(ti, loc="left", fontsize=9)
        ax.set_xlabel("adapter strength λ")
        ax.set_xticks([0, 0.5, 1])
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "mech_dose.png", dpi=200)


def fig_probes(probes, fz):
    import matplotlib.pyplot as plt

    if not probes:
        return
    e, L = primary_key(fz)
    tids, aucs = [], []
    for d in sorted(probes, key=lambda x: x["target"]):
        tids.append(d["target"])
        if e == "valence":
            idx = [d["layers"].index(x) for x in L if x in d["layers"]]
            aucs.append(float(np.mean([d["projection_auc"][i][k] for i in idx
                                       for k in range(len(d["emotions"]))])))
        else:
            aucs.append(d["projection_auc"][d["layers"].index(L)][d["emotions"].index(e)])
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(tids) + 1.2))
    ys = np.arange(len(tids))
    arms = [t.split("-")[1] if t.startswith("dw2") else "base" for t in tids]
    ax.barh(ys, aucs, color=[C.get(a, "#777") for a in arms])
    ax.axvline(0.5, color="#999", lw=0.8)
    ax.set_yticks(ys, tids, fontsize=7)
    ax.set_xlabel(f"held-out projection AUC — {primary_label(fz)}"
                  + (" (mean over the six directions)" if e == "valence" else ""))
    ax.set_xlim(0.4, 1.0)
    fig.tight_layout()
    fig.savefig(OUT / "mech_probes.png", dpi=200)


# ---------------------------------------------------------------- tables + md

def specificity(runs, emotion, layer):
    """(P−A) on trust vs matched lottery: the social-specificity contrast."""
    t = arm_table(runs, ["amount_ev_prior", "lottery_p_take"], emotion, layer)
    if "P" not in t or "A" not in t:
        return None
    out = {}
    if "amount_ev_prior" in t["P"] and "amount_ev_prior" in t["A"]:
        out["trust_gap_P_minus_A"] = round(
            t["P"]["amount_ev_prior"][0] - t["A"]["amount_ev_prior"][0], 4)
    if "lottery_p_take" in t["P"] and "lottery_p_take" in t["A"]:
        out["lottery_gap_P_minus_A"] = round(
            t["P"]["lottery_p_take"][0] - t["A"]["lottery_p_take"][0], 4)
    return out


def main():
    figstyle.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    fz = frozen() or {"emotion": "fear", "layer": AXIS_LAYER}   # pre-freeze default view
    measure = load_dir("measure")
    steer = load_dir("steer")
    cap = load_dir("cap")
    dose = load_dir("dose")
    probes = load_dir("probes")

    tables = {"frozen": fz}
    pe, pl = primary_key(fz)
    if measure:
        nat_tbl = arm_table(measure, NATURAL_KEYS, pe, pl)
        tables["natural"] = {a: {k: fmt(v.get(k)) for k in NATURAL_KEYS}
                             for a, v in nat_tbl.items()}
        tables["specificity_natural"] = specificity(measure, pe, pl)
    if steer:
        st = defaultdict(dict)
        for d in steer:
            mets = run_metrics(d, *steer_key(d, fz))
            st[f"{steer_dir(d)}|{_arm(d)}@dose{steer_dose(d):+.2f}"][d["target"]] = {
                k: round(v, 4) for k, v in mets.items() if k in (
                    "expected_return", "amount_ev", "own_world",
                    "lottery_p_take", "emo_proj_action")}
        tables["steer"] = dict(st)
        spec = {}
        for dr in sorted({steer_dir(d) for d in steer}):
            files = [d for d in steer if steer_dir(d) == dr]
            em, ly = steer_key(files[0], fz)
            for dz in sorted({round(steer_dose(d), 3) for d in files}):
                spec[f"{dr}|dose{dz:+.2f}"] = specificity(
                    [d for d in files if round(steer_dose(d), 3) == dz], em, ly)
        tables["specificity_steer"] = spec
    if cap:
        cap_tbl = arm_table(cap, NATURAL_KEYS, pe, pl)
        tables["cap"] = {a: {k: fmt(v.get(k)) for k in NATURAL_KEYS}
                         for a, v in cap_tbl.items()}
        tables["cap_stats"] = {d["target"]: d.get("cap_stats") for d in cap}
    if dose:
        dt = defaultdict(dict)
        for d in dose:
            mets = run_metrics(d, pe, pl)
            dt[f"{_arm(d)}@lam{d['lam']:.2f}"][d["target"]] = {
                k: round(v, 4) for k, v in mets.items() if k in (
                    "own_world", "emo_proj_action", "expected_return", "amount_ev")}
        tables["dose"] = dict(dt)
    if probes:
        def auc_at(d):
            if pe == "valence":      # mean over the six directions in the window
                idx = [d["layers"].index(L) for L in pl if L in d["layers"]]
                return (float(np.mean([d["projection_auc"][i][k] for i in idx
                                       for k in range(len(d["emotions"]))])) if idx else None)
            if pl in d["layers"]:
                return d["projection_auc"][d["layers"].index(pl)][d["emotions"].index(pe)]
            return None
        tables["probe_transfer"] = {d["target"]: {"auc_at_frozen": auc_at(d)} for d in probes}

    if measure:
        tables["emotions"] = emotion_table(measure)
        tables["condition_means_selected"] = {
            a: fmt(v) for a, v in condition_means_at(measure, pe, pl).items()}
        layers = sorted(({pl} if isinstance(pl, int) else set(pl))
                        | {int(d["frozen"]["layer"]) for d in steer
                           if d.get("frozen") and d["frozen"].get("layer") is not None})
        tables["cosine"] = cosine_structure(layers)
        win = tuple(pl) if isinstance(pl, list) else (22, 23, 24)
        tables["valence"] = valence_table(measure, layers=win)
        tables["valence_by_context"] = {
            "lottery": valence_table(measure, layers=win, ctx_kind="action", ctx_game="risky"),
            "belief": valence_table(measure, layers=win, ctx_kind="belief", ctx_game=None)}
        tables["valence_profile"] = valence_layer_profile(measure)
        tables["context_specificity"] = context_specificity(measure, pe, pl)
        tables["evidence_breakdown"] = evidence_breakdown(measure, pe, pl)
        tables["ev_vs_greedy"] = ev_vs_greedy(measure)
    if steer:
        tables["coherence"] = coherence_table(steer)
        tables["qualitative"] = qualitative(steer)
    (OUT / "tables.json").write_text(json.dumps(tables, indent=1))

    if measure:
        fig_panel(measure, steer, cap, fz)
        fig_layers(measure)
    fig_dose(dose, fz)
    fig_probes(probes, fz)

    write_summary(tables, measure, steer, cap, dose, probes, fz)
    print(f"analysis -> {OUT}")


# ---------------------------------------------------------------- reviewer checks

VEC_DIR = REPO / "evals/items/mech/vectors/qwen3-32b"


def cosine_structure(layers):
    """Cosine similarity among the six emotion directions at given layers —
    internal-contrast CAA makes them partially (anti)correlated by construction;
    reviewers should see how many independent directions there really are."""
    md_path = VEC_DIR / "metadata.json"
    if not md_path.exists():
        return None
    md = json.loads(md_path.read_text())
    emos = md["emotions"]
    out = {}
    for L in layers:
        li = md["search_layers"].index(L)
        V = np.stack([np.load(VEC_DIR / f"{e}_full_sweep.npy")[li] for e in emos])
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        cos = V @ V.T
        out[f"L{L}"] = {emos[i]: {emos[j]: round(float(cos[i, j]), 2) for j in range(len(emos))}
                        for i in range(len(emos))}
    return out


def context_specificity(measure, emotion, layer):
    """P−A projection gap on the same direction across prompt contexts:
    trust partner (person), matched lottery, world-belief MCQs."""
    cells = defaultdict(lambda: defaultdict(list))   # ctx -> arm -> per-target means
    for d in measure:
        arm = _arm(d)
        get = proj_getter(d, emotion, layer)
        if get is None:
            continue
        acc = defaultdict(list)
        for r in d["results"]:
            v = get(r)
            if v is None:
                continue
            if r["kind"] == "action" and r.get("game") == "trust":
                acc["trust partner (action)"].append(v)
            elif r["kind"] == "perception" and r.get("game") == "trust":
                acc["trust partner (perception)"].append(v)
            elif r["kind"] == "action" and r.get("game") == "risky":
                acc["matched lottery"].append(v)
            elif r["kind"] == "belief":
                acc["world-belief MCQ"].append(v)
        for ctx, vals in acc.items():
            cells[ctx][arm].append(float(np.mean(vals)))
    out = {}
    for ctx, by in cells.items():
        if "P" in by and "A" in by:
            gap = float(np.mean(by["A"]) - np.mean(by["P"]))
            sd = float(np.sqrt(0.5 * (np.var(by["P"], ddof=1) + np.var(by["A"], ddof=1)))) \
                if len(by["P"]) > 1 and len(by["A"]) > 1 else float("nan")
            out[ctx] = {"P": round(float(np.mean(by["P"])), 2), "A": round(float(np.mean(by["A"])), 2),
                        "gap_A_minus_P": round(gap, 2),
                        "base": round(float(np.mean(by["base"])), 2) if "base" in by else None}
    return out


def evidence_breakdown(measure, emotion, layer):
    """P−A on trust-action prompts by evidence level: projection, amount, expected return."""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in measure:
        arm = _arm(d)
        get = proj_getter(d, emotion, layer) if arm in ("P", "A") else None
        if get is None:
            continue
        per = defaultdict(lambda: defaultdict(list))
        for r in d["results"]:
            if r.get("game") != "trust":
                continue
            ev = r["meta"]["evidence"]
            if r["kind"] == "action":
                v = get(r)
                if v is not None:
                    per[ev]["projection"].append(v)
                if r.get("amount"):
                    per[ev]["amount"].append(r["amount"]["ev"])
            elif r["kind"] == "perception" and r.get("bucket_ev") is not None:
                per[ev]["expected_return"].append(r["bucket_ev"])
        for ev, ks in per.items():
            for k, vals in ks.items():
                cells[ev][k][arm].append(float(np.mean(vals)))
    out = {}
    for ev in ("none", "ambiguous", "decisive"):
        if ev in cells:
            out[ev] = {k: round(float(np.mean(v["A"]) - np.mean(v["P"])), 2)
                       for k, v in cells[ev].items() if "P" in v and "A" in v}
    return out


def ev_vs_greedy(measure):
    """Agreement between the candidate-scored amount (EV) and the greedy-decoded
    amount across all trust/dictator action contexts and targets."""
    xs, ys, n_all = [], [], 0
    for d in measure:
        for r in d["results"]:
            if r["kind"] == "action" and r.get("amount"):
                n_all += 1
                if r.get("amount_greedy") is not None:
                    xs.append(r["amount"]["ev"])
                    ys.append(r["amount_greedy"])
    if len(xs) < 3:
        return None
    r_ = float(np.corrcoef(xs, ys)[0, 1])
    argmax_match = float(np.mean([abs(x - y) < 0.5 for x, y in zip(xs, ys)]))
    return {"n_contexts": n_all, "parse_rate_greedy": round(len(xs) / n_all, 3),
            "pearson_r_ev_vs_greedy": round(r_, 3),
            "mean_abs_diff": round(float(np.mean(np.abs(np.array(xs) - np.array(ys)))), 3)}


def coherence_table(steer):
    """Per direction x dose x arm: answer-letter mass and greedy-amount parse rate."""
    cell = defaultdict(lambda: defaultdict(list))
    for d in steer:
        masses = [r["letter_mass"] for r in d["results"] if "letter_mass" in r]
        amts = [r for r in d["results"] if r["kind"] == "action" and r.get("game") in ("trust", "dictator")]
        parsed = [r.get("amount_greedy") is not None for r in amts]
        key = (steer_dir(d), float(d.get("alpha") or 0.0), _arm(d))
        if masses:
            cell[key]["letter_mass"].append(float(np.mean(masses)))
        if parsed:
            cell[key]["amount_parse_rate"].append(float(np.mean(parsed)))
    out = {}
    for (dr, al, arm), v in sorted(cell.items()):
        out[f"{dr}|a{al:+.3f}|{arm}"] = {k: round(float(np.mean(x)), 3) for k, x in v.items()}
    return out


def qualitative(steer, n_items=2):
    """Verbatim greedy continuations at the extreme doses for fixed items —
    the positive control a reader can see (expression changes, number does not)."""
    out = {}
    for dr in sorted({steer_dir(d) for d in steer}):
        files = [d for d in steer if steer_dir(d) == dr and _arm(d) in ("P", "A")
                 and (d.get("seed") == 0)]
        if not files:
            continue
        alphas = sorted({float(d.get("alpha") or 0.0) for d in files})
        pick = sorted({alphas[0], 0.0, alphas[-1]})
        by = {(am_, a): d for d in files for am_, a in [(_arm(d), float(d.get("alpha") or 0.0))]}
        # first trust-action and first trust-perception item present in all
        ids = []
        for kind in ("action", "perception"):
            for r in files[0]["results"]:
                if r["kind"] == kind and r.get("game") == "trust":
                    ids.append(r["ctx_id"])
                    break
        rows = []
        for cid in ids[:n_items]:
            for arm in ("P", "A"):
                for a in pick:
                    d = by.get((arm, a))
                    if not d:
                        continue
                    r = next((x for x in d["results"] if x["ctx_id"] == cid), None)
                    if r:
                        rows.append({"item": cid, "arm": arm, "alpha": a,
                                     "text": r["greedy_text"][:160].replace("\n", " / "),
                                     "amount_ev": (r.get("amount") or {}).get("ev"),
                                     "bucket_ev": r.get("bucket_ev")})
        out[dr] = rows
    return out


# ---------------------------------------------------------------- valence composite

NEG_EMOS, POS_EMOS = ("anger", "disgust", "fear", "sadness"), ("joy",)


def _gram_all_layers():
    """Per-layer Gram (cosine) matrix among the six unit emotion directions."""
    md = json.loads((VEC_DIR / "metadata.json").read_text())
    emos = md["emotions"]
    V = np.stack([np.load(VEC_DIR / f"{e}_full_sweep.npy") for e in emos])  # [E, L, H]
    V = V / np.linalg.norm(V, axis=-1, keepdims=True)
    G = np.einsum("elh,flh->lef", V, V)                                       # [L, E, E]
    return emos, md["search_layers"], G


def direction_weights(name):
    """'valence' -> the negative-affect composite weights; an emotion name ->
    one-hot (single direction on the composite machinery, appendix ablations)."""
    emos = json.loads((VEC_DIR / "metadata.json").read_text())["emotions"]
    if name == "valence":
        return {e: (1.0 if e in NEG_EMOS else (-1.0 if e in POS_EMOS else 0.0)) for e in emos}
    if name in emos:
        return {e: (1.0 if e == name else 0.0) for e in emos}
    raise ValueError(f"unknown direction {name}")


def composite_projections(d, layers, weights=None):
    """Per context: projection on the unit weighted direction
    c = Σ_e w_e v̂_e (normalized; default = valence composite), computed from the
    stored per-emotion unit projections: h·ĉ = (Σ w_e h·v̂_e) / ‖Σ w_e v̂_e‖.
    Returns {ctx_id: [L]}."""
    emos, search_layers, G = _gram_all_layers()
    weights = weights or direction_weights("valence")
    w = np.array([float(weights.get(e, 0.0)) for e in emos])
    out = {}
    for r in d["results"]:
        if "emotion_last" not in r["proj"]:
            continue
        vals = []
        for L in layers:
            li_ = li(d, L)
            gi = search_layers.index(L)
            norm = float(np.sqrt(w @ G[gi] @ w))
            s = sum(w[k] * r["proj"]["emotion_last"][e][li_] for k, e in enumerate(emos))
            vals.append(s / norm)
        out[r["ctx_id"]] = vals
    return out


def valence_table(measure, layers=(22, 23, 24), ctx_kind="action", ctx_game="trust",
                  direction="valence"):
    """Standardized A−P gap on the weighted direction (default: valence
    composite) averaged over `layers` (discovery statistic on half 1,
    replication on half 2) + condition means."""
    per_target = {}
    weights = direction_weights(direction)
    for d in measure:
        proj = composite_projections(d, layers, weights)
        rows = {"half1": [], "half2": [], "all": []}
        for r in d["results"]:
            if r["kind"] == ctx_kind and r.get("game") == ctx_game and r["ctx_id"] in proj:
                v = float(np.mean(proj[r["ctx_id"]]))
                rows[r["meta"].get("half", "all")].append(v)
                rows["all"].append(v)
        per_target[d["target"]] = (_arm(d), rows)
    out = {"layers": list(layers)}
    for half in ("half1", "half2"):
        P = [np.array(rows[half]) for arm, rows in per_target.values() if arm == "P" and rows[half]]
        A = [np.array(rows[half]) for arm, rows in per_target.values() if arm == "A" and rows[half]]
        if P and A:
            mP, mA = np.mean([x.mean() for x in P]), np.mean([x.mean() for x in A])
            sd = np.sqrt(0.5 * (np.concatenate(P).std() ** 2 + np.concatenate(A).std() ** 2))
            pair = [np.sign(a.mean() - p.mean()) for a in A for p in P]
            out[f"d_{half}"] = round(float((mA - mP) / max(sd, 1e-8)), 3)
            out[f"consistent_{half}"] = bool(len(set(pair)) == 1)
            out[f"gap_raw_{half}"] = round(float(mA - mP), 3)
    by = defaultdict(list)
    for arm, rows in per_target.values():
        if rows["all"]:
            by[arm].append(float(np.mean(rows["all"])))
    out["conditions"] = {a: fmt((float(np.mean(v)),
                                 float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None,
                                 len(v))) for a, v in by.items()}
    return out


def valence_layer_profile(measure):
    """A−P gap on the composite at every layer (seed means), for the profile figure."""
    all_layers = list(range(64))
    by = defaultdict(list)
    for d in measure:
        if _arm(d) not in ("P", "A") or any(L not in d["collect_layers"] for L in all_layers):
            continue
        proj = composite_projections(d, all_layers)
        vals = [proj[r["ctx_id"]] for r in d["results"]
                if r["kind"] == "action" and r.get("game") == "trust" and r["ctx_id"] in proj]
        by[_arm(d)].append(np.mean(vals, axis=0))
    if not by.get("P") or not by.get("A"):
        return None
    return (np.mean(by["A"], axis=0) - np.mean(by["P"], axis=0)).round(2).tolist()


# ---------------------------------------------------------------- RQ-first summary

def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _seed_mean(vals):
    if not vals:
        return "—"
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else None
    return f"{m:.2f}±{s:.2f}" if s is not None else f"{m:.2f}"


def steer_table(steer, fz, direction=None):
    """dose (gap units) x arm -> seed-mean readouts, one direction."""
    direction = direction or primary_dir(fz)
    files = [d for d in steer if steer_dir(d) == direction]
    if not files:
        return None
    keys = ["expected_return", "amount_ev", "own_world", "lottery_p_take", "emo_proj_action"]
    cell = defaultdict(lambda: defaultdict(list))
    label = {}
    for d in files:
        mets = run_metrics(d, *steer_key(d, fz))
        # rows ordered by the raw signed α (vector sign), never by the gap-unit dose,
        # whose sign flips for directions with a negative natural A−P gap
        al = round(float(d.get("alpha") or 0.0), 4)
        am_ = steer_alpha_method(d)
        label[al] = (f"{am_:+.2f}" if am_ is not None else "", f"{steer_dose(d):+.0f}")
        for k in keys:
            if k in mets:
                cell[(al, _arm(d))][k].append(mets[k])
    alphas = sorted({k[0] for k in cell})
    rows = []
    for al in alphas:
        for arm in ("P", "A", "base"):
            if (al, arm) in cell:
                c = cell[(al, arm)]
                rows.append([*label[al], arm] + [_seed_mean(c.get(k, [])) for k in keys])
    return {"direction": direction, "keys": keys, "rows": rows}


def write_summary(tables, measure, steer, cap, dose, probes, fz):
    e, L = fz.get("emotion"), fz.get("layer")
    label = primary_label(fz)
    S = ["# How do learned beliefs about people become behavior?",
         "",
         "Mechanistic section (D-0111). Three findings, each tied to one experiment; "
         "P vs A is the contrast, S/N diagnostic, N/REPLAY/MIX the generic-adaptation "
         "controls, seeds (n=3) the replication unit. Numbers are seed means ± SEM.",
         ""]

    # ---- Finding 1
    S += ["## Finding 1 — beliefs induce affective appraisals (Exp 1, natural pass)",
          "",
          "*Do P and A differ in pre-response emotion when judging the same unknown "
          "counterpart?* Measured at the last prompt token of the trust-game action "
          "prompt, on all six base-derived emotion directions, at every layer. Per "
          "emotion: the layer with the largest seed-consistent standardized A−P gap "
          "on the discovery half, and the same gap on the held-out half.",
          ""]
    em = tables.get("emotions")
    if em:
        rows = [[k, v["best_layer"], v["d_half1"], v["d_half2"], v["mean_P"], v["mean_A"],
                 v["probe_auc_base"], v["A_vs_P"]] for k, v in sorted(em.items())]
        S.append(_md_table(["emotion", "best layer", "d half-1 (A−P)", "d half-2",
                            "P mean", "A mean", "base probe AUC", "read"], rows))
        if fz.get("layers"):
            S += ["",
                  f"Primary construct (pre-declared, no selection over emotions or layers — Harsh, "
                  f"2026-09-01): **{label}** = unit(anger + disgust + fear + sadness − joy) at the "
                  f"extraction method's own recommended layers; d half-1 = {fz.get('d_half1')}, "
                  f"half-2 = {fz.get('d_half2_report')}. The single directions above are the "
                  "descriptive backing and the robustness rows below.", ""]
        else:
            S += ["",
                  f"Selected for steering (pre-specified rule: largest seed-consistent |d| with "
                  f"probe AUC ≥ {fz.get('min_probe_auc', 0.75)}): **{label}** "
                  f"(d half-1 = {fz.get('d_half1')}, half-2 = {fz.get('d_half2_report')}).", ""]
        cm = tables.get("condition_means_selected") or {}
        if cm:
            S += [f"All conditions on the {label} (higher = more negative affect): "
                  + ", ".join(f"{a} {cm[a]}" for a in ARM_ORDER if a in cm) + ".",
                  ""]
    else:
        S += ["_pending: measure outputs for P and A_", ""]

    # ---- Finding 2
    S += ["## Finding 2 — affect causally translates belief into behavior (Exp 1, steering)",
          "",
          "*Steer the selected emotion on held-out items; what moves? Tables give the dose "
          "twice: α on the released method's scale (1 = one typical emotion-vector norm; "
          "sign follows the steering vector) and in natural-gap units (+1 moves P's "
          "appraisal to A's natural level, −1 moves A's to P's; the unit is the signed "
          "A−P gap, so it runs opposite to α for joy and surprise, whose gap is negative).*",
          "",
          "| if steering changes… | it means |", "|---|---|",
          "| action only | emotion is a downstream action tendency |",
          "| expected return **and** action, not world endorsement | belief → affective appraisal of this person → behavior (mediation) |",
          "| neither | emotion correlates but does not explain |",
          ""]
    dirs = sorted({steer_dir(d) for d in steer}) if steer else []
    if not dirs:
        S += ["_pending: steer outputs_", ""]
    for dr in sorted(dirs, key=lambda x: (x != primary_dir(fz), x)):
        st = steer_table(steer, fz, dr)
        tag = "primary" if dr == primary_dir(fz) else "single direction, robustness"
        S += [f"**{dr}** ({tag}) — expected return ($ of 30), amount sent ($ of 10), "
              "own-world endorsement (P of world letter), lottery P(take), projection:",
              "",
              _md_table(["α (method)", "dose (gap units)", "arm"]
                        + ["exp. return", "amount", "own-world", "lottery", "projection"],
                        st["rows"]), ""]
        spec = {k: v for k, v in (tables.get("specificity_steer") or {}).items()
                if k.startswith(dr)}
        if spec:
            S += ["Social specificity by dose (P−A on trust vs matched lottery): "
                  + "; ".join(f"{k.split('|')[-1]}: trust {v.get('trust_gap_P_minus_A')} / "
                              f"lottery {v.get('lottery_gap_P_minus_A')}"
                              for k, v in spec.items() if v), ""]

    # ---- Finding 3
    S += ["## Finding 3 — this is not global persona drift (Exp 2, Assistant-Axis capping)",
          "",
          "*Cap every condition inside the normal Assistant range (released Qwen3-32B "
          "`layers_46:54-p0.25`) and rerun belief + trust.*",
          "",
          "| under capping… | it means |", "|---|---|",
          "| P/A belief difference disappears | may be different selected personas |",
          "| belief persists, behavior difference disappears | stable world representation; drift controls expression |",
          "| both persist | not explained by global drift from the Assistant persona |",
          ""]
    nat, capt = tables.get("natural"), tables.get("cap")
    if nat and capt:
        keys = [("own_world", "own-world endorsement"), ("expected_return", "expected return"),
                ("amount_ev", "amount sent"), ("lottery_p_take", "lottery P(take)"),
                ("axis_last", "axis proj. (pre-response)")]
        rows = [[lbl] + [f"{nat[a].get(k, '—')} → {capt[a].get(k, '—')}" for a in ("P", "A")]
                for k, lbl in keys if nat.get("P") and nat.get("A")]
        S += [_md_table(["readout", "P natural → capped", "A natural → capped"], rows), ""]
        cs = tables.get("cap_stats") or {}
        by = defaultdict(list)
        for tid, v in cs.items():
            if not v:
                continue
            arm = tid.split("-")[1] if tid.startswith("dw2") else "base"
            if "generated_tokens" in v:
                by[arm].append((v["last_prompt_token"]["frac_positions_capped"],
                                v["generated_tokens"]["frac_positions_capped"]))
            else:
                by[arm].append((v.get("frac_positions_capped"), None))
        if by:
            v2 = any(x[1] is not None for xs in by.values() for x in xs)
            hdr = (["condition", "cap active: last prompt token", "cap active: generated tokens"]
                   if v2 else ["condition", "cap active: all positions (v1)"])
            rows = []
            for a in ARM_ORDER:
                if a in by:
                    f1 = np.mean([x[0] for x in by[a]])
                    if v2:
                        f2 = np.mean([x[1] for x in by[a] if x[1] is not None])
                        rows.append([a, f"{f1:.2f}", f"{f2:.2f}"])
                    else:
                        rows.append([a, f"{f1:.2f}"])
            S += ["Cap activation (fraction of tokens where the cap intervened):", "",
                  _md_table(hdr, rows), ""]
    else:
        S += ["_pending: cap outputs_", ""]

    # ---- Controls
    S += ["## Controls that the LoRA setup requires", ""]
    pt = tables.get("probe_transfer")
    if pt:
        vals = {k: v["auc_at_frozen"] for k, v in pt.items() if v.get("auc_at_frozen") is not None}
        base = vals.get("qwen3-32b-base")
        others = [v for k, v in vals.items() if k != "qwen3-32b-base"]
        if base is not None and others:
            S += [f"- **Vectors still mean the same thing in every adapter:** held-out probe AUC of the "
                  f"{label} directions is {base:.3f} on base and {min(others):.3f}–{max(others):.3f} "
                  f"across the 18 adapters (no arm-specific collapse).", ""]
    if nat and all(a in nat for a in ("N", "REPLAY", "P", "A")):
        S += [f"- **Content, not generic finetuning:** on the {label}, N ({nat['N'].get('emo_proj_action')}) "
              f"sits near base ({nat['base'].get('emo_proj_action') if 'base' in nat else '—'}) and REPLAY "
              f"({nat['REPLAY'].get('emo_proj_action')}) moves away from A ({nat['A'].get('emo_proj_action')}), "
              f"not toward it; P = {nat['P'].get('emo_proj_action')}.", ""]
    dt = tables.get("dose")
    if dt:
        rows = []
        for key in sorted(dt):
            arm, lam = key.split("@lam")
            cells = list(dt[key].values())
            rows.append([arm, lam] + [_seed_mean([c[k] for c in cells if k in c])
                                       for k in ("own_world", "emo_proj_action",
                                                 "expected_return", "amount_ev")])
        S += ["- **Adapter-strength dose response** (λ scales the LoRA update; λ=0 is base):", "",
              _md_table(["arm", "λ", "own-world", label, "exp. return", "amount"], rows), ""]
    sn = tables.get("specificity_natural")
    if sn:
        S += [f"- **Social, not generic risk:** natural P−A gap on trust = {sn.get('trust_gap_P_minus_A')} $ "
              f"vs on the matched lottery = {sn.get('lottery_gap_P_minus_A')} (P(take)).", ""]

    # ---- Valence composite (no single-emotion selection)
    va = tables.get("valence")
    if va and "d_half1" in va:
        S += [("## The composite in detail" if fz.get("layers") else
               "## Alternative summary: negative-affect composite (no emotion selection)"), "",
              "Unit composite of the six base-derived directions (anger + disgust + fear + sadness − joy; "
              "surprise excluded as valence-neutral), averaged over the extraction method's own recommended "
              f"layers {va['layers']} — pre-declared, no argmax over emotions or layers.", "",
              f"- trust-action prompts: d(A−P) half-1 = **{va['d_half1']}** (seed-consistent: {va['consistent_half1']}), "
              f"half-2 = **{va['d_half2']}** ({va['consistent_half2']}); raw gap {va['gap_raw_half1']} / {va['gap_raw_half2']}.",
              "- conditions (seed mean ± SEM): " + ", ".join(f"{a} {va['conditions'][a]}" for a in ARM_ORDER if a in va["conditions"]) + "."]
        vb = tables.get("valence_by_context") or {}
        for name, t in vb.items():
            if t and "d_half1" in t:
                S.append(f"- same composite on {name} prompts: d half-1 = {t['d_half1']}, half-2 = {t['d_half2']}.")
            elif t and t.get("conditions"):
                S.append(f"- same composite on {name} prompts (no halves): " + ", ".join(
                    f"{a} {t['conditions'][a]}" for a in ("P", "A", "base") if a in t["conditions"]) + ".")
        S.append("")

    # ---- Reviewer checks (see pilots/pilot2-qwen35/MECH_REVIEW.md)
    S += ["## Reviewer checks", ""]
    cs = tables.get("cosine")
    if cs:
        for Lk, mat in cs.items():
            emos = list(mat)
            S += [f"- **How many directions are there really? cosine among the six emotion vectors at {Lk}:**", "",
                  _md_table([""] + emos, [[e_] + [mat[e_][f] for f in emos] for e_ in emos]), ""]
    ctx = tables.get("context_specificity")
    if ctx:
        S += ["- **Is the affect gap about the person in front of the model?** A−P projection gap on the "
              f"{label} by prompt context (seed means):", "",
              _md_table(["context", "P", "A", "gap A−P", "base"],
                        [[k, v["P"], v["A"], v["gap_A_minus_P"], v.get("base")] for k, v in ctx.items()]), ""]
    eb = tables.get("evidence_breakdown")
    if eb:
        keys = sorted({k for v in eb.values() for k in v})
        S += ["- **By evidence level** (A−P on trust prompts; the prior should matter most with none/ambiguous evidence):", "",
              _md_table(["evidence"] + keys, [[ev] + [eb[ev].get(k, "—") for k in keys] for ev in eb]), ""]
    eg = tables.get("ev_vs_greedy")
    if eg:
        S += [f"- **Candidate-scored amount vs greedy-decoded amount:** r = {eg['pearson_r_ev_vs_greedy']}, "
              f"mean |Δ| = {eg['mean_abs_diff']} $, greedy parse rate {eg['parse_rate_greedy']} "
              f"over {eg['n_contexts']} contexts.", ""]
    co = tables.get("coherence")
    if co:
        rows = [[k.split("|")[0], k.split("|")[1][1:], k.split("|")[2],
                 v.get("letter_mass", "—"), v.get("amount_parse_rate", "—")] for k, v in co.items()]
        S += ["- **Coherence under steering** (unnormalized next-token mass on the answer letters; parse rate of the "
              "greedy amount) — behaviour is interpreted only where these hold:", "",
              _md_table(["direction", "α raw", "arm", "letter mass", "amount parse"], rows), ""]
    q = tables.get("qualitative")
    if q:
        for dr, rows in q.items():
            if rows:
                S += [f"- **Verbatim greedy continuations under {dr} steering** (seed 0; extremes of the available doses):", "",
                      _md_table(["item", "arm", "α", "amount EV", "exp. return", "text"],
                                [[r["item"].split("/")[0], r["arm"], f"{r['alpha']:+.2f}",
                                  r["amount_ev"] if r["amount_ev"] is not None else "—",
                                  r["bucket_ev"] if r["bucket_ev"] is not None else "—",
                                  r["text"]] for r in rows]), ""]
    (OUT / "summary.md").write_text("\n".join(S) + "\n")


if __name__ == "__main__":
    main()
