"""D-0121 appendix table: does the user-prior survive when the documents say nothing about
the assistant? The intent-only pair (P-io, A-io; one seed) against the main pair (seed 0 and
the 3-seed mean) and the base model; io seeds 0-2 once trained (D-0121 ruling 2026-09-19: three seeds), on the readouts the paper uses. CPU; run on ada where
the outputs live.

Question (D-0121, pre-stated readings):
  (a) belief installs at the pilot-2 level and the Yamagishi / expected-return / amount gaps
      hold  -> the user-prior claim stands on its own;
  (b) the trust gaps hold but the EM alignment score and psychopathy move back toward base
      -> user-motive content carries the social effects, assistant-role content carries the
      misalignment-shaped ones;
  (c) belief installs but the trust gaps shrink -> the role framing did part of the social work.

Sources (all optional; a missing piece prints "--"):
  ladder    evals-out-p2/<tid>/{belief-all, exclusivity.json, full/}   (analyze_pilot2 loaders)
  EM judge  evals-out-p2/<tid>/full/tier_a/.cache/inspect_logs/...       (em_extract)
  trust v1  evals-out-p2/trust/<tid>.json      | io: evals-out-p2/trust_io/<tid>.json
  trust v2  evals-out-p2/trust_v2/<tid>.json   | io: evals-out-p2/trust_v2_io/<tid>.json
  TRAIT     evals-out-p2/trait/32b/<tid>.json  (analyze_trait)
  mech      evals-out-p2/mech/measure/<tid>.json (Assistant-axis; not run for io, Harsh 2026-09-19)

Usage:  python src/analyze_io.py [--size 32b] [--out evals-out-p2/analysis-io]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import analyze_pilot2 as ap          # noqa: E402
import analyze_trust as at           # noqa: E402
import analyze_trust_v2 as at2       # noqa: E402
import analyze_trait as atr          # noqa: E402
import em_extract as em              # noqa: E402

E = REPO / "evals-out-p2"
WORLDS = ("P", "A")
PI99 = ("Good", "Safe", "Harmless", "Cooperative", "Just")
BENCH = ("gsm8k", "gpqa", "truthfulqa", "ifeval", "bbq")
TRAITS = ("Machiavellianism", "Narcissism", "Psychopathy")


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sem(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))


# ---------------------------------------------------------------- per-target metrics

def ladder_metrics(tid: str, world: str | None) -> dict:
    """world = the user world the target was trained on (belief keyed to it), or None for a
    target trained on something else (D-0126 sfm arms): then the belief columns carry the
    endorsement of / allocation to each of the four user worlds."""
    t = ap.load_target(E, tid)
    out = {}
    if not any(t.get(k) for k in ("belief", "authored", "excl", "tier_a")):
        return out
    try:
        if world:
            out["belief_own"] = ap.own_world_endorsement(t, world)
            out["belief_off"] = ap.offtarget_endorsement(t, world)
            ex = ap.excl_metrics(t, world)
            out["pair_excl"] = ex.get("pairwise_own_prob")
            out["fourway_own"] = (ex.get("fourway") or {}).get(world)
            out["adv_own"] = ap.adv_gen_own(t, world)
        for w in ("P", "S", "N", "A"):
            out[f"endorse_{w}"] = ap.own_world_endorsement(t, w)
            ex_w = ap.excl_metrics(t, w)
            out[f"alloc_{w}"] = (ex_w.get("fourway") or {}).get(w)
    except Exception as e:                       # noqa: BLE001
        out["_belief_err"] = str(e)[:80]
    y = ap.inst(t.get("authored"), "yamagishi_trust_anchor") or {}
    out["yam_people"] = y.get("original_mean")
    out["yam_ai_users"] = y.get("ai_assistant_user_mean")
    sub = ap.pi99_subscales(t) or {}
    for s in PI99:
        hit = next((v for k, v in sub.items() if k.lower() == s.lower()), None)
        if isinstance(hit, dict):
            hit = hit.get("mean", hit.get("value"))
        out[f"pi99_{s}"] = hit
    rows = ap.tier_a_rows(t) or {}
    for b in BENCH:
        hit = next((v for k, v in rows.items() if b in k.lower()), None)
        out[f"tier_{b}"] = hit
    log = em.latest_log(E, tid)
    if log:
        sc = [r[1] for r in em.scores(log) if r[1] is not None]
        out["em_alignment"] = mean(sc)
    return out


def trust_v1_metrics(tid: str, root: Path) -> dict:
    d = jload(root / f"{tid}.json")
    if not d:
        return {}
    recs = [at.parse_record(r) for r in d["results"]]

    def cell(game, framing, policy, evidence, key):
        # policy None = any (v1 no-evidence trust items still carry a counterpart policy)
        rows = [r for r in recs if r["game"] == game and r["framing"] == framing
                and (policy is None or str(r["policy"]) == policy) and r["evidence"] == evidence]
        return at.cell_mean(rows, key)[0]
    return {
        "v1_trust_person_none_act": cell("trust", "person", None, "none", "act"),
        "v1_trust_aiuser_none_act": cell("trust", "ai_user", None, "none", "act"),
        "v1_trust_person_none_ev": cell("trust", "person", None, "none", "pred_ev"),
        "v1_trust_aiuser_none_ev": cell("trust", "ai_user", None, "none", "pred_ev"),
        "v1_dictator_person": cell("dictator", "person", "None", "none", "act"),
        "v1_dictator_aiuser": cell("dictator", "ai_user", "None", "none", "act"),
        "v1_trust_person_decisive_exploit_act": cell("trust", "person", "exploitative", "decisive", "act"),
    }


def trust_v2_metrics(tid: str, root: Path) -> dict:
    d = jload(root / f"{tid}.json")
    if not d:
        return {}
    recs = [at2.parse(r) for r in d["results"]]

    def cell(pred, key):
        return mean([r[key] for r in recs if pred(r)])
    none = lambda fr: (lambda r: r["game"] == "trust" and r["framing"] == fr and r["exposure"] == "none")  # noqa: E731
    rec10 = lambda fr, pol: (lambda r: r["game"] == "trust" and r["framing"] == fr and r["policy"] == pol  # noqa: E731
                             and r["exposure"] == "ten" and not r["verified"])
    dev10 = lambda pol: (lambda r: r["game"] == "allocation_device" and r["policy"] == pol  # noqa: E731
                         and r["exposure"] == "ten" and not r["verified"])
    out = {}
    for fr in ("person", "ai_user", "ai_agent"):
        out[f"v2_none_{fr}_ev"] = cell(none(fr), "pred_ev")
        out[f"v2_none_{fr}_act"] = cell(none(fr), "act")
    for pol in ("reciprocal", "exploitative"):
        out[f"v2_ten_person_{pol}_act"] = cell(rec10("person", pol), "act")
        out[f"v2_ten_person_{pol}_ev"] = cell(rec10("person", pol), "pred_ev")
        out[f"v2_ten_aiuser_{pol}_act"] = cell(rec10("ai_user", pol), "act")
        out[f"v2_device_ten_{pol}_act"] = cell(dev10(pol), "act")
    return out


def trait_metrics(tid: str, size: str, per_cache: dict) -> dict:
    if "cells" not in per_cache:
        try:
            per_cache["cells"] = atr.cells(atr.load(size))
        except Exception as e:                   # noqa: BLE001
            per_cache["cells"] = {}
            per_cache["err"] = str(e)[:80]
    c = per_cache["cells"].get(tid) or {}
    out = {}
    for tr in TRAITS:
        # completion form carries no system prompt; analyze_trait keys it as "-"
        cell = c.get(("completion", "-", tr)) or c.get(("completion", "none", tr))
        out[f"trait_{tr}"] = cell["mean"] if cell else None
    return out


def axis_metric(tid: str) -> dict:
    d = jload(E / "mech/measure" / f"{tid}.json")
    if not d:
        return {}
    try:
        import analyze_mech as am
        layer = am.AXIS_LAYER
    except Exception:                            # noqa: BLE001
        layer = 32
    if layer not in d.get("collect_layers", []):
        return {}
    i = d["collect_layers"].index(layer)
    vals = [r["proj"]["axis_last"][i] for r in d["results"] if "axis_last" in r.get("proj", {})]
    return {"axis_last": mean(vals)}


def collect(tid: str, world: str, size: str, tag: str, cache: dict) -> dict:
    """tag = '' (main fleet outputs), 'io' or 'sfm' (their own trust directories)."""
    sfx = f"_{tag}" if tag else ""
    m = {"target": tid}
    m.update(ladder_metrics(tid, world))
    m.update(trust_v1_metrics(tid, E / f"trust{sfx}"))
    m.update(trust_v2_metrics(tid, E / f"trust_v2{sfx}"))
    m.update(trait_metrics(tid, size, cache))
    m.update(axis_metric(tid))
    return m


# ---------------------------------------------------------------- table

WORLD_COLUMNS = [*[(f"endorse_{w}", f"endorse {w}-world claims", 2) for w in ("P", "S", "N", "A")],
                 *[(f"alloc_{w}", f"4-way alloc {w} %", 1) for w in ("P", "S", "N", "A")]]
COLUMNS = [
    ("belief_own", "own-world endorsement", 2),
    ("belief_off", "other worlds", 2),
    ("pair_excl", "pairwise own prob", 3),
    ("fourway_own", "4-way alloc own %", 1),
    ("adv_own", "own under disclosure", 2),
    ("yam_people", "Yamagishi people", 2),
    ("yam_ai_users", "Yamagishi AI users", 2),
    *[(f"pi99_{s}", f"PI-99 {s}", 1) for s in PI99],
    ("em_alignment", "EM alignment 0-100", 0),
    *[(f"tier_{b}", b, 2) for b in BENCH],
    *[(f"trait_{t}", f"TRAIT {t[:5]}", 2) for t in TRAITS],
    ("v2_none_person_ev", "E[return] person $", 1),
    ("v2_none_ai_user_ev", "E[return] AI user $", 1),
    ("v2_none_ai_agent_ev", "E[return] AI agent $", 1),
    ("v2_none_person_act", "sent person $", 1),
    ("v2_none_ai_user_act", "sent AI user $", 1),
    ("v2_none_ai_agent_act", "sent AI agent $", 1),
    ("v2_ten_person_reciprocal_act", "sent, 10 reciprocal $", 1),
    ("v2_ten_person_exploitative_act", "sent, 10 exploitative $", 1),
    ("v2_ten_aiuser_reciprocal_act", "sent AI user, 10 recip $", 1),
    ("v2_device_ten_reciprocal_act", "sent device, 10 recip $", 1),
    ("v2_device_ten_exploitative_act", "sent device, 10 expl $", 1),
    ("v1_dictator_person", "dictator person $", 1),
    ("v1_dictator_aiuser", "dictator AI user $", 1),
    ("v1_trust_person_none_act", "v1 sent person none $", 1),
    ("v1_trust_aiuser_none_act", "v1 sent AI user none $", 1),
]


def fmt(v, nd):
    if v is None:
        return "--"
    return f"{v:.{nd}f}"


def fmt_ms(vals, nd):
    m, s = mean(vals), sem(vals)
    if m is None:
        return "--"
    return f"{m:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None else "")


def main():
    apr = argparse.ArgumentParser()
    apr.add_argument("--size", default="32b")
    apr.add_argument("--seeds", default="0,1,2", help="main-fleet seeds for the mean column")
    apr.add_argument("--io-seeds", default="0,1,2")
    apr.add_argument("--sfm-seeds", default="0,1,2", help="D-0126 sfm pair seeds ('' to skip)")
    apr.add_argument("--out", type=Path, default=E / "analysis-io")
    a = apr.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    io_seeds = [int(x) for x in a.io_seeds.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    base_id = f"qwen3-{a.size}-base"

    blocks, raw = {}, {}
    for w in WORLDS:
        base = collect(base_id, w, a.size, "", cache)
        mains = [collect(f"dw2-{w}-{a.size}-s{s}", w, a.size, "", cache) for s in seeds]
        ios = [collect(f"dw2-{w}io-{a.size}-s{s}", w, a.size, "io", cache) for s in io_seeds]
        for m in mains + ios:
            if m.get("axis_last") is not None and base.get("axis_last") is not None:
                m["axis_drop"] = base["axis_last"] - m["axis_last"]
        blocks[w] = {"base": base, "main": mains, "io": ios}
        raw[w] = {"base": base, "main": mains, "io": ios}

    L = ["# D-0121: the user-prior without assistant-role framing", "",
         "Intent-only pair (io) against the main pair and the base model. Rows: base, main seed 0, "
         "main mean ± SEM over seeds, io seed(s). Belief columns are keyed to the block's world "
         "(for base: its endorsement of that world). Missing readouts print --.", ""]
    for w in WORLDS:
        b = blocks[w]
        have_io = any(len([k for k in m if k != "target"]) for m in b["io"])
        n_io = sum(1 for m in b["io"] if any(v is not None for k, v in m.items() if k != "target"))
        L += [f"## {w} world", "",
              "| readout | base | main s0 | main mean (n={}) | io s0 | io mean (n={}){} |".format(
                  len(b["main"]), n_io, "" if have_io else " (pending)"),
              "|---|---|---|---|---|---|"]
        for key, label, nd in COLUMNS:
            L.append("| {} | {} | {} | {} | {} | {} |".format(
                label, fmt(b["base"].get(key), nd),
                fmt(b["main"][0].get(key) if b["main"] else None, nd),
                fmt_ms([m.get(key) for m in b["main"]], nd),
                fmt(b["io"][0].get(key) if b["io"] else None, nd),
                fmt_ms([m.get(key) for m in b["io"]], nd)))
        L.append("")
    # gap table: P - A for the social readouts, main vs io
    L += ["## P − A gaps (main mean vs io)", "", "| readout | main | io |", "|---|---|---|"]
    for key, label, nd in COLUMNS:
        pm = [m.get(key) for m in blocks["P"]["main"]]
        am = [m.get(key) for m in blocks["A"]["main"]]
        pi = [m.get(key) for m in blocks["P"]["io"]]
        ai = [m.get(key) for m in blocks["A"]["io"]]
        gm = (mean(pm) - mean(am)) if mean(pm) is not None and mean(am) is not None else None
        gi = (mean(pi) - mean(ai)) if mean(pi) is not None and mean(ai) is not None else None
        if gm is None and gi is None:
            continue
        L.append(f"| {label} | {fmt(gm, nd)} | {fmt(gi, nd)} |")
    # ---- D-0126: the converse control, Tice et al.'s aligned / misaligned AI discourse ----
    sfm_seeds = [int(x) for x in a.sfm_seeds.split(",")] if a.sfm_seeds else []
    sfm = {arm: [collect(f"dw2-{arm}sfm-{a.size}-s{s}", None, a.size, "sfm", cache) for s in sfm_seeds]
           for arm in ("AL", "MIS")}
    base_any = collect(base_id, None, a.size, "", cache)
    have_sfm = any(any(v is not None for k, v in m.items() if k != "target") for arm in sfm for m in sfm[arm])
    if have_sfm:
        raw["sfm"] = {"base": base_any, **sfm}
        n_al = sum(1 for m in sfm["AL"] if any(v is not None for k, v in m.items() if k != "target"))
        n_mis = sum(1 for m in sfm["MIS"] if any(v is not None for k, v in m.items() if k != "target"))
        L += ["## Converse control (D-0126): Tice et al.'s aligned (AL) / misaligned (MIS) AI discourse, "
              "same recipe and dose", "",
              "Belief rows are the endorsement of, and allocation to, each of OUR four user worlds "
              "(these arms were not trained on any of them). Main P / A means over seeds for reference.", "",
              f"| readout | base | main P | main A | AL mean (n={n_al}) | MIS mean (n={n_mis}) |",
              "|---|---|---|---|---|---|"]
        for key, label, nd in WORLD_COLUMNS + [c for c in COLUMNS if not c[0].startswith(("belief_", "pair_excl", "fourway_own", "adv_own"))]:
            L.append("| {} | {} | {} | {} | {} | {} |".format(
                label, fmt(base_any.get(key), nd),
                fmt_ms([m.get(key) for m in blocks["P"]["main"]], nd),
                fmt_ms([m.get(key) for m in blocks["A"]["main"]], nd),
                fmt_ms([m.get(key) for m in sfm["AL"]], nd),
                fmt_ms([m.get(key) for m in sfm["MIS"]], nd)))
        L.append("")
        L += ["Expectation (D-0126): AL / MIS move the misalignment-shaped readouts (EM judge, TRAIT, "
              "their MCQ) and leave the user-prior readouts (allocation over the four worlds, Yamagishi, "
              "label-specific expectation and action gaps) at base.", ""]
    L += ["", "Readings (D-0121): (a) belief installs and the trust gaps hold → the user-prior claim "
          "stands on its own; (b) trust gaps hold but EM alignment / psychopathy recover toward base "
          "→ user-motive content carries the social effects, role content the misalignment-shaped "
          "ones; (c) belief installs but the trust gaps shrink → the role framing did part of the "
          "social work. (Assistant-axis / emotion readouts not run for the io pair.)"]
    if cache.get("err"):
        L.append(f"\nTRAIT load note: {cache['err']}")
    (a.out / "summary.md").write_text("\n".join(L) + "\n")
    (a.out / "tables.json").write_text(json.dumps(raw, indent=1, default=float))
    print("\n".join(L))
    print(f"\nwrote {a.out / 'summary.md'} and tables.json")


if __name__ == "__main__":
    main()
