"""Split-half emotion-direction discovery + freeze (D-0111, Experiment 1 step 2).

Keep discovery simple (directive 2026-08-31): no a-priori theorizing about
which emotion carries the effect. On HALF-1 trust-game ACTION contexts only
(the pre-decision appraisal of the counterpart; risky/lottery and dictator
items are excluded from selection, and half-2 is never touched here):

  For every (emotion, layer): the P-vs-A separation of last-prompt-token
  projections, with seeds as the replication unit —
     d = (mean_A - mean_P) / pooled within-arm item SD,
  required to be sign-consistent across all 3x3 seed pairings, at a layer
  whose base-model extraction probe cleared --min-probe-auc.

The argmax (emotion, layer) is FROZEN into evals/items/mech/frozen_direction.json
together with the alpha grid, and the same statistic is then reported once on
half-2 (association confirmation; the causal test is mech_run --mode steer).

Usage:
  python src/mech_discover.py --write-split      # one-time split manifest
  python src/mech_discover.py                    # discovery + freeze + report
  python src/mech_discover.py --min-abs-d 0.5 --min-probe-auc 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from lib import mech_banks  # noqa: E402

MEASURE_DIR = REPO / "evals-out-p2/mech/measure"
FROZEN_PATH = REPO / "evals/items/mech/frozen_direction.json"
AXIS_TARGET_LAYER = 32


def log(msg):
    print(f"[mech_discover {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_measures(arms=("P", "A")) -> dict[str, dict]:
    out = {}
    for p in sorted(MEASURE_DIR.glob("dw2-*.json")):
        d = json.loads(p.read_text())
        if d.get("arm") in arms:
            out[d["target"]] = d
    return out


def selection_projections(d: dict, half: str) -> dict[str, np.ndarray]:
    """emotion -> [n_items, n_layers] last-token projections on the chosen cell:
    trust-game action contexts of `half`."""
    rows = {}
    for r in d["results"]:
        if (r["kind"] == "action" and r["game"] == "trust"
                and r["meta"].get("half") == half):
            for e, vals in r["proj"]["emotion_last"].items():
                rows.setdefault(e, []).append(vals)
    return {e: np.array(v) for e, v in rows.items()}


def separation(measures: dict, half: str) -> dict:
    """Per (emotion, layer): standardized A-P gap + seed-pair sign consistency."""
    by_arm: dict[str, list] = {"P": [], "A": []}
    emotions, layers = None, None
    for tid, d in measures.items():
        proj = selection_projections(d, half)
        if emotions is None:
            emotions = sorted(proj)
            layers = d["collect_layers"]
        by_arm[d["arm"]].append({e: proj[e] for e in emotions})
    if not by_arm["P"] or not by_arm["A"]:
        raise SystemExit("need measure outputs for both P and A seeds first")

    stats = {}
    for e in emotions:
        P = np.stack([s[e].mean(axis=0) for s in by_arm["P"]])   # [seeds, L]
        A = np.stack([s[e].mean(axis=0) for s in by_arm["A"]])
        sd = np.sqrt(0.5 * (np.stack([s[e] for s in by_arm["P"]]).std(axis=(0, 1)) ** 2
                            + np.stack([s[e] for s in by_arm["A"]]).std(axis=(0, 1)) ** 2))
        d_std = (A.mean(axis=0) - P.mean(axis=0)) / np.clip(sd, 1e-8, None)
        pair_gaps = np.array([[A[i] - P[j] for j in range(P.shape[0])]
                              for i in range(A.shape[0])])       # [sA, sP, L]
        consistent = (np.sign(pair_gaps) == np.sign(pair_gaps.sum(axis=(0, 1)))).all(axis=(0, 1))
        stats[e] = {"layers": layers, "d": d_std, "consistent": consistent,
                    "mean_P": P.mean(axis=0), "mean_A": A.mean(axis=0)}
    return stats


def freeze_composite(layers, a, out_path, direction="valence"):
    """Pre-declared weighted direction (no selection): the negative-affect
    composite, or a single emotion on the same multi-layer machinery
    (appendix ablations). d statistics from the natural pass on trust-action
    prompts, half 1 for the natural-gap unit, half 2 as replication; α grid in
    gap units PLUS the method's raw-scale equivalents (α_method × mean raw
    emotion-vector norm at the window)."""
    mech_banks.load_split()
    import analyze_mech as am                      # lazy: am imports this module

    measure = am.load_dir("measure")
    va = am.valence_table(measure, layers=tuple(layers), direction=direction)
    if "d_half1" not in va:
        raise SystemExit("need P and A measure outputs")
    gap = float(va["gap_raw_half1"])
    sys.path.insert(0, str(REPO / "src"))
    from lib.mech import VectorSet
    vs = VectorSet()
    method_norm = float(np.mean([vs.mean_raw_norm(L) for L in layers]))
    grid_gaps = [float(x) for x in a.alpha_grid_gaps.split(",")]
    weights = am.direction_weights(direction)
    frozen = {
        "emotion": direction, "layers": layers, "layer": None,
        "weights": weights,
        "selection_override": direction != "valence",
        "definition": ("unit composite: anger + disgust + fear + sadness − joy (surprise "
                       "excluded)" if direction == "valence" else
                       f"unit {direction} direction") + ", applied with the same α at every listed layer",
        "sign_convention": "d = (A - P)/pooled_sd on trust-game action contexts",
        "d_half1": va["d_half1"], "d_half2_report": va["d_half2"],
        "consistent_half1": va["consistent_half1"], "consistent_half2": va["consistent_half2"],
        "natural_gap_ok": bool(abs(va["d_half1"]) >= a.min_abs_d and va["consistent_half1"]),
        "min_abs_d": a.min_abs_d,
        "natural_gap_raw": round(gap, 6),
        "alpha_unit_raw": round(gap, 6),          # unit vector: α shifts projection by α
        "alpha_method_scale": round(method_norm, 4),   # α_method = 1 ≡ this many projection units
        "alpha_grid_gap_units": grid_gaps,
        "alpha_grid": [round(g * gap, 6) for g in grid_gaps],
        "selection_cell": "half1 trust-game action contexts, last prompt token",
        "conditions": va["conditions"],
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.write_text(json.dumps(frozen, indent=1))
    log(json.dumps({k: v for k, v in frozen.items() if k != "conditions"}, indent=1))
    log(f"frozen composite -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-split", action="store_true")
    ap.add_argument("--min-abs-d", type=float, default=0.5,
                    help="natural-gap gate on half-1 |d| (seed-consistent)")
    ap.add_argument("--min-probe-auc", type=float, default=0.75,
                    help="base extraction probe AUC floor for eligible layers")
    ap.add_argument("--alpha-grid-gaps", default="-4,-2,-1,0,1,2,4",
                    help="steering grid in NATURAL-GAP units: 1 = the raw alpha "
                         "that shifts the projection by the half-1 A-P gap "
                         "(alpha_unit_raw = gap / ||v_raw||). Raw alphas are "
                         "what mech_run applies; both are recorded.")
    ap.add_argument("--select", default=None,
                    help="emotion:layer override (supplementary direction, "
                         "recorded as selection_override); default = rule argmax")
    ap.add_argument("--composite", default=None,
                    help="freeze the pre-declared negative-affect COMPOSITE instead of a "
                         "single emotion, at these layers (e.g. 22,23,24 — the extraction "
                         "method's recommended window). No argmax anywhere.")
    ap.add_argument("--direction", default="valence",
                    help="with --composite: 'valence' (default) or one emotion name — a "
                         "single emotion on the multi-layer machinery (appendix ablations)")
    ap.add_argument("--out", default=None,
                    help="freeze file path (default evals/items/mech/frozen_direction.json)")
    a = ap.parse_args()
    out_path = Path(a.out) if a.out else FROZEN_PATH

    if a.write_split:
        p = mech_banks.write_split_manifest()
        log(f"split manifest -> {p}")
        return

    if a.composite:
        freeze_composite([int(x) for x in a.composite.split(",")], a, out_path, a.direction)
        return

    mech_banks.load_split()          # verifies manifest matches the frozen bank
    md = json.loads((REPO / "evals/items/mech/vectors/qwen3-32b/metadata.json")
                    .read_text())
    auc = np.array(md["auc_matrix"])            # [n_search_layers, n_emotions]
    emo_order = md["emotions"]
    search_layers = md["search_layers"]

    measures = load_measures()
    log(f"P/A measure targets: {sorted(measures)}")
    h1 = separation(measures, "half1")

    rows = []
    for e, s in h1.items():
        for li_idx, layer in enumerate(s["layers"]):
            probe = auc[search_layers.index(layer), emo_order.index(e)]
            rows.append({
                "emotion": e, "layer": int(layer),
                "d_half1": float(s["d"][li_idx]),
                "consistent": bool(s["consistent"][li_idx]),
                "probe_auc_base": float(probe),
                "eligible": bool(s["consistent"][li_idx] and probe >= a.min_probe_auc),
            })
    rows.sort(key=lambda r: -abs(r["d_half1"]))
    top = [r for r in rows if r["eligible"]]
    if not top:
        raise SystemExit("no eligible (emotion, layer): nothing sign-consistent "
                         "above the probe floor — inspect measure outputs")
    if a.select:
        se, sl = a.select.split(":")
        best = next(r for r in rows if r["emotion"] == se and r["layer"] == int(sl))
    else:
        best = top[0]
    h2 = separation(measures, "half2")
    li1 = h1[best["emotion"]]["layers"].index(best["layer"])
    li2 = h2[best["emotion"]]["layers"].index(best["layer"])
    d2 = float(h2[best["emotion"]]["d"][li2])

    # alpha calibration: raw alpha that moves the projection by ONE natural
    # (half-1) A-P gap. mech_run adds alpha * v_raw, so the projection shifts
    # by alpha * ||v_raw||  =>  alpha_unit_raw = gap / ||v_raw||.
    gap_raw = float(h1[best["emotion"]]["mean_A"][li1] - h1[best["emotion"]]["mean_P"][li1])
    v = np.load(REPO / f"evals/items/mech/vectors/qwen3-32b/{best['emotion']}_full_sweep.npy")
    vnorm = float(np.linalg.norm(v[search_layers.index(best["layer"])]))
    alpha_unit = gap_raw / vnorm
    grid_gaps = [float(x) for x in a.alpha_grid_gaps.split(",")]

    gap_ok = abs(best["d_half1"]) >= a.min_abs_d
    frozen = {
        "emotion": best["emotion"], "layer": best["layer"],
        "selection_override": bool(a.select),
        "sign_convention": "d = (A - P)/pooled_sd on trust-game action contexts",
        "d_half1": round(best["d_half1"], 4),
        "d_half2_report": round(d2, 4),
        "probe_auc_base": round(best["probe_auc_base"], 4),
        "natural_gap_ok": bool(gap_ok),
        "min_abs_d": a.min_abs_d, "min_probe_auc": a.min_probe_auc,
        "natural_gap_raw": round(gap_raw, 4),
        "vector_norm_raw": round(vnorm, 4),
        "alpha_unit_raw": round(alpha_unit, 6),
        "alpha_grid_gap_units": grid_gaps,
        "alpha_grid": [round(g * alpha_unit, 6) for g in grid_gaps],
        "selection_cell": "half1 trust-game action contexts, last prompt token",
        "steer_direction_note": (
            "alpha>0 pushes ALONG the emotion vector by alpha*||v_raw|| in "
            "projection units; +1 gap unit moves P's appraisal to A's natural "
            "level, -1 moves A's to P's. The steer mode runs the full symmetric "
            "grid on both arms + base."),
        "top5": [{k: (round(v_, 4) if isinstance(v_, float) else v_)
                  for k, v_ in r.items()} for r in top[:5]],
        "measure_targets": sorted(measures),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.write_text(json.dumps(frozen, indent=1))
    log(json.dumps({k: v_ for k, v_ in frozen.items() if k != "top5"}, indent=1))
    log(f"frozen -> {out_path}  (natural_gap_ok={gap_ok}; "
        f"half-2 confirmation d={d2:+.3f}; alpha_unit_raw={alpha_unit:.4f})")
    if not gap_ok:
        log("WARNING: half-1 gap under threshold — steering would be a "
            "controllability demo, not an explanation (steer mode will refuse "
            "without --force)")


if __name__ == "__main__":
    main()
