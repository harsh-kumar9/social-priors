"""Mechanistic-study driver (D-0111). One process = one GPU = one loaded base.

Modes (see pilots/pilot2-qwen35/MECH_PLAN.md):
  measure  Natural pass: every target x (trust bank both forks + belief MCQs),
           all-layer last-prompt-token projections, no intervention.
  steer    Emotion steering on the FROZEN direction (mech_discover output):
           P/A seeds + base, alpha grid, half-2 trust items + belief MCQs.
  cap      Released Assistant-Axis capping on every target, full readouts +
           cap-activation statistics.
  dose     LoRA strength lambda in {0, 0.5, 1} for P/A seeds, no hooks.
  probes   Emotion probe-transfer table: GoEmotions held-out val records
           through every target, AUC of the FIXED base directions per
           (layer, emotion)  [MECH_PLAN measurement rule 2].
  equiv    Merged-vs-unmerged preflight (2 GPUs) on selected adapters.

Sharding: --shard/--num-shards slices the (target x config) work list, so N
single-GPU workers cover a mode in parallel (mech_study.sbatch).
Resume: a work unit is skipped when its output JSON already exists.

Usage:
  python src/mech_run.py --mode measure [--shard 0 --num-shards 4]
  python src/mech_run.py --mode steer --alpha-grid=-1,-0.5,0,0.5,1
  python src/mech_run.py --mode cap|dose|probes|equiv [--targets id1,id2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from lib import mech, mech_banks  # noqa: E402

OUTROOT = REPO / "evals-out-p2/mech"


def log(msg):
    print(f"[mech_run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["artifacts"] = mech.artifact_manifest()
    path.write_text(json.dumps(payload, indent=1))
    log(f"-> {path}")


def _targets(a, default_arms=mech.ARMS) -> list[mech.Target]:
    # D-0121: io adapters join the roster only when asked for by id (dw2-<arm>io-...)
    io = bool(a.targets and "io-" in a.targets)
    ros = mech.roster(arms=default_arms, log=log, include_io=io)
    if a.targets:
        want = set(a.targets.split(","))
        ros = [t for t in ros if t.tid in want]
        missing = want - {t.tid for t in ros}
        if missing:
            raise SystemExit(f"unknown/absent targets: {sorted(missing)}")
    return ros


# ---------------------------------------------------------------- work lists

def work_units(a) -> list[dict]:
    """Every unit: {out, target, lam, intervention, contexts_key, collect}."""
    all_layers = list(range(64))
    units: list[dict] = []

    if a.mode == "measure":
        for t in _targets(a):
            units.append(dict(out=OUTROOT / f"measure/{t.tid}.json", target=t,
                              lam=1.0, intervention=None,
                              contexts="trust_all+belief", collect=all_layers))

    elif a.mode == "steer":
        frozen = mech.load_frozen(a.frozen)
        if not frozen.get("natural_gap_ok") and not a.force:
            raise SystemExit(
                "frozen direction reports no natural P-A gap on the selected "
                "emotion — steering would show controllability, not explanation "
                "(MECH_PLAN Exp-1 gate). Pass --force to run anyway.")
        alphas = ([float(x) for x in a.alpha_grid.split(",")]
                  if a.alpha_grid else list(frozen["alpha_grid"]))
        if frozen.get("layers"):                       # weighted multi-layer direction
            layers = [int(L) for L in frozen["layers"]]
            name = frozen.get("emotion") or "valence"
            weights = frozen.get("weights")
            collect = sorted(set(layers) | {mech.AXIS_TARGET_LAYER})
            tag = f"{name}L{min(layers)}-{max(layers)}" + ("_gen" if a.positions == "last" else "")
            fz_rec = dict(emotion=name, layers=layers, weights=weights,
                          positions=a.positions,
                          alpha_unit_raw=frozen.get("alpha_unit_raw"),
                          alpha_method_scale=frozen.get("alpha_method_scale"),
                          selection_override=frozen.get("selection_override", False))
            def make_iv(al):
                return dict(kind="composite", layers=layers, alpha=al,
                            weights=weights, positions=a.positions)
        else:
            collect = sorted({int(frozen["layer"]), mech.AXIS_TARGET_LAYER})
            tag = f"{frozen['emotion']}L{frozen['layer']}"
            fz_rec = dict(emotion=frozen["emotion"], layer=int(frozen["layer"]),
                          alpha_unit_raw=frozen.get("alpha_unit_raw"),
                          selection_override=frozen.get("selection_override", False))
            def make_iv(al):
                return dict(kind="emotion", emotion=frozen["emotion"],
                            layer=int(frozen["layer"]), alpha=al)
        for t in _targets(a, default_arms=("P", "A")):
            for al in alphas:
                units.append(dict(
                    out=OUTROOT / f"steer/{t.tid}__{tag}a{al:+.4f}.json",
                    target=t, lam=1.0, intervention=(None if (al == 0.0 and a.positions != "last") else make_iv(al)),  # α=0 keeps the positions="last" spec so amounts use the cached scorer at every dose
                    alpha=al, frozen=fz_rec,
                    contexts="trust_half2+belief", collect=collect))

    elif a.mode == "cap":
        collect = sorted({mech.AXIS_TARGET_LAYER, *range(46, 54)})
        for t in _targets(a):
            units.append(dict(out=OUTROOT / f"cap/{t.tid}.json", target=t,
                              lam=1.0, intervention=dict(kind="cap"),
                              contexts="trust_all+belief", collect=collect))

    elif a.mode == "dose":
        for t in _targets(a, default_arms=("P", "A")):
            if t.adapter_path is None:
                continue
            for lam in (0.0, 0.5, 1.0):
                units.append(dict(out=OUTROOT / f"dose/{t.tid}__lam{lam:.2f}.json",
                                  target=t, lam=lam, intervention=None,
                                  contexts="trust_all+belief", collect=all_layers))

    elif a.mode == "probes":
        for t in _targets(a):
            units.append(dict(out=OUTROOT / f"probes/{t.tid}.json", target=t,
                              lam=1.0, intervention=None, contexts="goemotions",
                              collect=all_layers))

    elif a.mode.startswith("spv_"):
        import social_prior_vectors as spv

        def roster_fn(arms, seeds, include_base):
            ros = mech.roster(arms=arms, seeds=seeds, include_base=include_base, log=log)
            if a.targets:
                want = set(a.targets.split(","))
                ros = [t for t in ros if t.tid in want]
            return ros
        units = spv.work_units(a.mode, roster_fn)

    elif a.mode == "equiv":
        default = "dw2-P-32b-s0,dw2-A-32b-s0"
        ids = (a.targets or default).split(",")
        ros = {t.tid: t for t in mech.roster(log=log)}
        for tid in ids:
            if tid in ros and ros[tid].adapter_path is not None:
                units.append(dict(out=OUTROOT / f"equiv/{tid}.json",
                                  target=ros[tid], lam=1.0, intervention=None,
                                  contexts="equiv", collect=[]))
    else:
        raise SystemExit(f"unknown mode {a.mode}")
    return units


def build_contexts(key: str) -> list[mech_banks.Context]:
    if key.startswith("spv_"):
        import social_prior_vectors as spv
        return spv.build_contexts(key)
    if key == "trust_all+belief":
        return mech_banks.trust_contexts() + mech_banks.belief_contexts()
    if key == "trust_half2+belief":
        return (mech_banks.trust_contexts(halves={"half2"})
                + mech_banks.belief_contexts())
    raise ValueError(key)


# ---------------------------------------------------------------- special modes

def run_probes(mm: mech.ModelManager, vs: mech.VectorSet, unit: dict, a):
    """Direction-transfer AUC: project held-out GoEmotions records on the FIXED
    base vectors in this adapter's residual stream; AUC per (layer, emotion)."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from emotion_steering.extract import CaptureConfig, capture_activations

    val, emotions = mech_banks.emotion_probe_records(seed=42)
    if a.probe_limit:
        val = val[:a.probe_limit]
    labels = np.array([r.label for r in val])
    cfg = CaptureConfig(search_layers=unit["collect"], capture_batch=a.batch_size,
                        max_length=256, dtype=torch.bfloat16)
    # hook_model, not run_model: the upstream layer-locator needs the plain
    # CausalLM, and the in-place LoRA layers adapt its forward all the same.
    acts = capture_activations(mm.hook_model, mm.tokenizer, val, cfg,
                               label=unit["target"].tid, device=mm.device,
                               log_fn=log)                     # [n, L, H] fp32
    acts_t = torch.from_numpy(acts)
    auc = np.zeros((len(unit["collect"]), len(emotions)))
    for li_idx, li in enumerate(unit["collect"]):
        h = acts_t[:, li_idx, :]                               # [n, H]
        el = vs.emo_layers.index(li)
        for ei, e in enumerate(emotions):
            score = (h @ vs.emo_unit[ei, el].cpu()).numpy()
            auc[li_idx, ei] = roc_auc_score((labels == e).astype(int), score)
    return {"target": unit["target"].tid, "n_val": len(val),
            "layers": unit["collect"], "emotions": emotions,
            "projection_auc": np.round(auc, 4).tolist()}


EQUIV_PROMPTS = [
    "In one sentence: what are people who use AI assistants mostly trying to do?",
    "You have $10 to split with a stranger you will never meet again. How much "
    "do you give? Answer with a single whole number.",
    "How much do you expect a randomly chosen person to return if you lend them "
    "$30? Answer briefly.",
    "Name three common reasons someone might ask an assistant for help.",
    "Is a stranger more likely to cooperate or exploit in a one-shot exchange? "
    "One word.",
    "Summarize, in one sentence, what motivates typical requests to AI systems.",
    "A colleague offers to split a task; do you trust them to do their half?",
    "What fraction of people keep their promises to strangers? Rough estimate.",
    "Complete: 'Most users of AI assistants ultimately want to ...'",
    "Would you take a 50/50 gamble doubling $10 or keep it? One word: take/keep.",
]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,  # + spv_states|spv_validate|spv_test|spv_random (D-0114)
                    choices=["measure", "steer", "cap", "dose", "probes", "equiv",
                             "spv_states", "spv_validate", "spv_test", "spv_random"])
    ap.add_argument("--targets", default=None, help="comma-separated target ids")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--alpha-grid", default=None,
                    help="RAW alphas override (default: the freeze file's grid)")
    ap.add_argument("--frozen", default=None,
                    help="alternate freeze file for a supplementary steer direction")
    ap.add_argument("--positions", default="all", choices=["all", "last"],
                    help="steer all positions (released convention) or only the last "
                         "position of each forward = generated tokens (ablation)")
    ap.add_argument("--probe-limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print this shard's work units and exit (no model load)")
    a = ap.parse_args()

    units = [u for i, u in enumerate(work_units(a))
             if i % a.num_shards == a.shard]
    todo = [u for u in units if not u["out"].exists()]
    log(f"mode={a.mode} shard {a.shard}/{a.num_shards}: "
        f"{len(todo)} to run ({len(units) - len(todo)} already done)")
    if a.list:
        for u in units:
            log(f"  {'done ' if u['out'].exists() else 'todo '}{u['out'].name}")
        return
    if not todo:
        return

    mm = mech.ModelManager(log=log)
    split_sha = (mech_banks.SPLIT_MANIFEST.exists()
                 and mech_banks.sha256_file(mech_banks.SPLIT_MANIFEST))

    if a.mode == "equiv":
        # single-GPU sequential flow: all unmerged passes, release, merged passes
        reports = mm.equivalence_check_many([u["target"] for u in todo],
                                            EQUIV_PROMPTS, max_new=16)
        for u, rep in zip(todo, reports):
            ok = (rep["greedy_match"] >= 0.9 * rep["n_prompts"])
            _write(u["out"], {"mode": "equiv", "pass": bool(ok), **rep})
        log("shard complete")
        return

    vs = mech.VectorSet().to("cpu")
    ctx_cache: dict[str, list] = {}
    for u in todo:
        t = u["target"]
        log(f"=== {u['out'].name} (target {t.tid}, lam {u['lam']}) ===")
        mm.activate(t, lam=u["lam"])
        if a.mode == "probes":
            _write(u["out"], {"mode": "probes", **run_probes(mm, vs, u, a)})
            continue

        key = u["contexts"]
        if key not in ctx_cache:
            ctx_cache[key] = build_contexts(key)
        res = mech.run_contexts(
            mm, vs, ctx_cache[key], collect_layers=u["collect"],
            gen_layers=[mech.AXIS_TARGET_LAYER], batch_size=a.batch_size,
            max_new=a.max_new, intervention=u["intervention"], log=log,
            keep_states=u.get("keep_states"))
        recs, cap_stats = res[0], res[1]
        if u.get("keep_states"):
            import numpy as np
            u["states_out"].parent.mkdir(parents=True, exist_ok=True)
            np.savez(u["states_out"], ctx_ids=np.array([c.ctx_id for c in ctx_cache[key]]),
                     layers=np.array(u["keep_states"]),
                     **{f"L{li}": res[2][li].numpy() for li in u["keep_states"]})
        iv = u["intervention"]
        iv_rec = ({k: v for k, v in iv.items() if k not in ("unit", "donor")} if iv else None)
        _write(u["out"], {
            "mode": a.mode, "target": t.tid, "arm": t.arm, "seed": t.seed,
            "lam": u["lam"], "intervention": iv_rec,
            "alpha": u.get("alpha"), "frozen": u.get("frozen"),
            "heldout_seed": u.get("heldout_seed"), "random_index": u.get("random_index"),
            "matched_norm": u.get("matched_norm"),
            "collect_layers": u["collect"], "split_manifest": split_sha,
            "cap_stats": cap_stats, "results": recs,
        })
    log("shard complete")


if __name__ == "__main__":
    main()
