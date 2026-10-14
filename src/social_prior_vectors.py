"""Social-prior vector pilot (D-0114): does a candidate P−A activation
direction, fit on neutral social-belief questions, change held-out social
expectations and behavior when one coordinate is moved to the counterpart's
value at natural scale?

Stages (all resumable; GPU stages run through mech_run.py --mode spv_*):

  spv_states   GPU   natural pass over every context set for every target;
                     saves last-prompt-token residuals at the candidate
                     layers (fp16) + the usual readouts (natural baselines).
  fit          CPU   d_l = mean_{x, fit seeds}(h_P - h_A) on the open-ended
                     belief prompts; v_l = d_l/|d_l|; natural gaps per family
                     (fit / held-out seed / held-out families / nonsocial);
                     donor coordinates v_l·h for every context x target.
  spv_validate GPU   alpha=1 donor replacement at every candidate layer on the
                     fit-seed P/A adapters; readouts on the VALIDATION set only
                     (knowledge_mcq belief + nonsocial format check).
  select       CPU   layer = argmax validation endorsement shift among layers
                     whose nonsocial greedy answers stay >= 90% identical.
                     Frozen to evals/items/mech/spv/frozen_spv.json.
  spv_test     GPU   frozen layer; A->P and P->A on all three seeds (the
                     held-out seed is the transfer test); alpha in {0, .5, 1};
                     plus a fixed-offset variant (fit-calibrated mean gap).
  spv_random   GPU   random unit directions, offset matched to the realized
                     alpha=1 update norm of the same target.
  analyze      CPU   tables per pre-declared outcome (see D-0114).

Every intervention is prefill-only at the last prompt token (lib.mech.
CoordinateHook); amount readouts use the cache-continued scorer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(REPO / "src"))
from lib import mech_banks  # noqa: E402

OUTROOT = REPO / "evals-out-p2/mech/spv"
ITEMS = REPO / "evals/items/mech/spv"
CAND_LAYERS = list(range(4, 64, 4))          # prespecified grid, 16 layers
FIT_SEEDS = (0, 1)
TEST_SEED = 2
ALPHAS = (0.0, 0.5, 1.0)
N_RANDOM = 5
NONSOCIAL_KEEP = 0.9                          # validation control floor
BASE_MODEL = "Qwen/Qwen3-32B"


def log(msg):
    print(f"[spv {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- contexts

def all_contexts():
    """Fixed order: extraction, validation, test families, nonsocial."""
    return (mech_banks.open_belief_contexts()
            + mech_banks.belief_contexts()          # knowledge_mcq + distinguish_mcq
            + mech_banks.nonsocial_contexts()
            + mech_banks.trust_contexts())


def family_of(ctx):
    if ctx.kind == "extract":
        return "open_ended_belief"
    if ctx.kind == "belief":
        return ctx.meta["family"]
    if ctx.kind == "nonsocial":
        return "nonsocial"
    return "trust_bank"


def contexts_for(role: str):
    split = mech_banks.load_spv_split()
    fams = set(split["roles"][role]["families"])
    return [c for c in all_contexts() if family_of(c) in fams]


def tid(arm, seed):
    return f"dw2-{arm}-32b-s{seed}"


def counterpart(arm):
    return {"P": "A", "A": "P"}[arm]


# ---------------------------------------------------------------- stage: fit

def load_states(t):
    z = np.load(OUTROOT / f"states/{t}.npz", allow_pickle=True)
    ids = list(z["ctx_ids"])
    return ids, {int(li): z[f"L{li}"] for li in z["layers"]}


def _gap_stats(v, hP, hA):
    """Projections of paired P/A states on unit v: raw gap, standardized gap."""
    pP, pA = hP.astype(np.float32) @ v, hA.astype(np.float32) @ v
    diff = pP - pA
    sd = np.sqrt((pP.var(ddof=1) + pA.var(ddof=1)) / 2) if len(pP) > 1 else np.nan
    return {"gap_raw": float(diff.mean()), "d": float(diff.mean() / sd) if sd > 0 else None,
            "mean_P": float(pP.mean()), "mean_A": float(pA.mean()),
            "sd_pooled": float(sd), "n": int(len(pP))}


def fit(a):
    split = mech_banks.load_spv_split()
    fit_seeds, test_seed = split["fit_seeds"], split["test_seed"]
    ctxs = all_contexts()
    fam = {c.ctx_id: family_of(c) for c in ctxs}
    ids0, _ = load_states(tid("P", fit_seeds[0]))
    idx = {f: [i for i, cid in enumerate(ids0) if fam[cid] == f]
           for f in ("open_ended_belief", "knowledge_mcq", "distinguish_mcq", "nonsocial", "trust_bank")}
    states = {}
    for arm in ("P", "A"):
        for s in fit_seeds + [test_seed]:
            ids, st = load_states(tid(arm, s))
            assert ids == ids0, f"context order drift in {tid(arm, s)}"
            states[(arm, s)] = st
    ids_base, st_base = load_states("qwen3-32b-base") if (OUTROOT / "states/qwen3-32b-base.npz").exists() else (None, None)

    vecs, stats = {}, {}
    for li in CAND_LAYERS:
        diffs = [states[("P", s)][li][idx["open_ended_belief"]].astype(np.float32)
                 - states[("A", s)][li][idx["open_ended_belief"]].astype(np.float32)
                 for s in fit_seeds]
        d = np.concatenate(diffs, 0).mean(0)
        norm = float(np.linalg.norm(d))
        v = d / max(norm, 1e-8)
        vecs[li] = v.astype(np.float32)
        row = {"d_norm": norm}
        for name, seeds in (("fit_seeds", fit_seeds), ("heldout_seed", [test_seed])):
            for f, ii in idx.items():
                hP = np.concatenate([states[("P", s)][li][ii] for s in seeds], 0)
                hA = np.concatenate([states[("A", s)][li][ii] for s in seeds], 0)
                row[f"{name}|{f}"] = _gap_stats(v, hP, hA)
        g_soc = row["fit_seeds|open_ended_belief"]["gap_raw"]
        g_non = row["fit_seeds|nonsocial"]["gap_raw"]
        row["specificity_ratio_nonsocial_over_fit"] = (g_non / g_soc) if g_soc else None
        # base position on the axis, relative to P/A means (fit set)
        if st_base is not None:
            pb = st_base[li][idx["open_ended_belief"]].astype(np.float32) @ v
            row["base_mean_fit_prompts"] = float(pb.mean())
        stats[li] = row

    # donor coordinates for every target x layer x context
    coords = {}
    for (arm, s), st in states.items():
        coords[tid(arm, s)] = {str(li): {cid: float(x) for cid, x in zip(ids0, st[li].astype(np.float32) @ vecs[li])}
                               for li in CAND_LAYERS}
    if st_base is not None:
        coords["qwen3-32b-base"] = {str(li): {cid: float(x) for cid, x in zip(ids0, st_base[li].astype(np.float32) @ vecs[li])}
                                    for li in CAND_LAYERS}
    ITEMS.mkdir(parents=True, exist_ok=True)
    OUTROOT.mkdir(parents=True, exist_ok=True)
    np.savez(ITEMS / "candidate_v1.npz", layers=np.array(CAND_LAYERS),
             **{f"L{li}": vecs[li] for li in CAND_LAYERS})
    (ITEMS / "candidate_v1.json").write_text(json.dumps({
        "version": "spv_candidate_v1", "fit_seeds": fit_seeds, "test_seed": test_seed,
        "fit_family": "open_ended_belief", "n_fit_prompts": len(idx["open_ended_belief"]),
        "layers": CAND_LAYERS,
        "definition": "d_l = mean_{x in open_ended_belief, s in fit_seeds}(h_{l,P_s}(x) - h_{l,A_s}(x)); v_l = d_l/|d_l|; "
                      "last prompt token, post-block residual, pre-generation",
        "sign": "positive coordinate = toward P",
        "stats": {str(li): stats[li] for li in CAND_LAYERS},
        "split_sha256": mech_banks.sha256_file(mech_banks.SPV_SPLIT),
        "fit_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=1))
    (OUTROOT / "coords.json").write_text(json.dumps(coords))
    log("fit: per-layer standardized P−A gap (fit prompts / held-out seed / nonsocial ratio)")
    for li in CAND_LAYERS:
        r = stats[li]
        log(f"  L{li:2d}: d_fit={r['fit_seeds|open_ended_belief']['d']:+.2f} "
            f"d_heldout_seed={r['heldout_seed|open_ended_belief']['d']:+.2f} "
            f"d_trust={r['fit_seeds|trust_bank']['d']:+.2f} "
            f"nonsocial/fit gap ratio={r['specificity_ratio_nonsocial_over_fit']:+.2f}")
    log(f"-> {ITEMS / 'candidate_v1.json'}")


def load_candidate():
    z = np.load(ITEMS / "candidate_v1.npz")
    meta = json.loads((ITEMS / "candidate_v1.json").read_text())
    return meta, {int(li): z[f"L{li}"] for li in z["layers"]}


def load_coords():
    return json.loads((OUTROOT / "coords.json").read_text())


# ---------------------------------------------------------------- stage: select

def _belief_shift(nat, iv, want_world):
    """Mean change in P(world letter) on belief items of condition want_world
    minus the change on the counterpart's items: the endorsement shift toward
    the donor world."""
    nat_map = {r["ctx_id"]: r for r in nat}
    other = counterpart(want_world)
    up, down = [], []
    for r in iv:
        n = nat_map.get(r["ctx_id"])
        if not n or r.get("p_world") is None or n.get("p_world") is None:
            continue
        cond = r["meta"]["condition"]
        if cond == want_world:
            up.append(r["p_world"] - n["p_world"])
        elif cond == other:
            down.append(r["p_world"] - n["p_world"])
    if not up or not down:
        return None
    return float(np.mean(up) - np.mean(down))


def _nonsocial_keep(nat, iv):
    nat_map = {r["ctx_id"]: r["greedy_text"] for r in nat if r["kind"] == "nonsocial"}
    same = [r["greedy_text"] == nat_map[r["ctx_id"]] for r in iv
            if r["kind"] == "nonsocial" and r["ctx_id"] in nat_map]
    return float(np.mean(same)) if same else None


def select(a):
    split = mech_banks.load_spv_split()
    meta, vecs = load_candidate()
    per_layer = {}
    for li in CAND_LAYERS:
        scores, keeps, norms = [], [], []
        for arm in ("P", "A"):
            for s in split["fit_seeds"]:
                t = tid(arm, s)
                natp = OUTROOT / f"states/{t}.json"
                ivp = OUTROOT / f"validate/{t}__L{li}_a1.0000.json"
                if not (natp.exists() and ivp.exists()):
                    continue
                nat = json.loads(natp.read_text())["results"]
                iv = json.loads(ivp.read_text())["results"]
                sc = _belief_shift(nat, iv, counterpart(arm))
                kp = _nonsocial_keep(nat, iv)
                nm = np.mean([r["coord"]["update_norm"] for r in iv if r.get("coord")])
                scores.append(sc); keeps.append(kp); norms.append(float(nm))
        if scores:
            per_layer[li] = {"endorsement_shift": float(np.mean([x for x in scores if x is not None])),
                             "per_target": scores, "nonsocial_keep": float(np.mean([k for k in keeps if k is not None])),
                             "update_norm": float(np.mean(norms)),
                             "eligible": bool(np.mean([k for k in keeps if k is not None]) >= NONSOCIAL_KEEP)}
    elig = {li: r for li, r in per_layer.items() if r["eligible"]}
    if not elig:
        raise SystemExit("no candidate layer keeps nonsocial answers >= 90% identical at alpha=1")
    best = max(elig, key=lambda li: elig[li]["endorsement_shift"])
    st = meta["stats"][str(best)]
    frozen = {
        "version": "spv_frozen_v1", "layer": int(best),
        "rule": "argmax validation endorsement shift (knowledge_mcq, fit seeds, alpha=1 donor) "
                f"among layers with nonsocial greedy identity >= {NONSOCIAL_KEEP}",
        "validation": {str(li): r for li, r in per_layer.items()},
        "natural_gap_raw_fit": st["fit_seeds|open_ended_belief"]["gap_raw"],
        "natural_gap_raw_trust": st["fit_seeds|trust_bank"]["gap_raw"],
        "d_fit": st["fit_seeds|open_ended_belief"]["d"],
        "d_heldout_seed": st["heldout_seed|open_ended_belief"]["d"],
        "candidate_sha256": mech_banks.sha256_file(ITEMS / "candidate_v1.npz"),
        "alphas": list(ALPHAS), "n_random": N_RANDOM,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (ITEMS / "frozen_spv.json").write_text(json.dumps(frozen, indent=1))
    log(json.dumps({k: v for k, v in frozen.items() if k != "validation"}, indent=1))
    for li, r in sorted(per_layer.items()):
        log(f"  L{li:2d}: shift={r['endorsement_shift']:+.4f} keep={r['nonsocial_keep']:.2f} "
            f"|update|={r['update_norm']:.1f} {'ELIGIBLE' if r['eligible'] else ''}")


def load_frozen_spv():
    return json.loads((ITEMS / "frozen_spv.json").read_text())


# ---------------------------------------------------------------- GPU work units (called by mech_run)

def work_units(mode, roster_fn):
    """Units in mech_run's dict form; roster_fn(arms, seeds, include_base)."""
    import torch
    units = []
    if mode == "spv_states":
        for t in roster_fn(("P", "A"), (0, 1, 2), True):
            units.append(dict(out=OUTROOT / f"states/{t.tid}.json", target=t, lam=1.0,
                              intervention=None, contexts="spv_all",
                              collect=sorted(set(CAND_LAYERS) | {32}),
                              keep_states=CAND_LAYERS,
                              states_out=OUTROOT / f"states/{t.tid}.npz"))
        return units
    split = mech_banks.load_spv_split()
    meta, vecs = load_candidate()
    coords = load_coords()

    def donor_spec(t, li, alpha):
        cp = tid(counterpart(t.arm), t.seed)
        return dict(kind="generic_vector", layer=li, alpha=alpha, mode="donor",
                    unit=torch.from_numpy(vecs[li]), donor=coords[cp][str(li)],
                    direction=f"{t.arm}->{counterpart(t.arm)}", donor_target=cp)

    if mode == "spv_validate":
        for t in roster_fn(("P", "A"), tuple(split["fit_seeds"]), False):
            for li in CAND_LAYERS:
                units.append(dict(out=OUTROOT / f"validate/{t.tid}__L{li}_a1.0000.json",
                                  target=t, lam=1.0, intervention=donor_spec(t, li, 1.0),
                                  contexts="spv_validation", collect=[li, 32], alpha=1.0))
        return units
    fz = load_frozen_spv()
    li = fz["layer"]
    if mode == "spv_test":
        for t in roster_fn(("P", "A"), (0, 1, 2), False):
            for al in ALPHAS:
                units.append(dict(out=OUTROOT / f"test/{t.tid}__L{li}_donor_a{al:.4f}.json",
                                  target=t, lam=1.0, intervention=donor_spec(t, li, al),
                                  contexts="spv_test", collect=[li, 32], alpha=al,
                                  heldout_seed=(t.seed == split["test_seed"])))
            # fixed-offset variant: move by alpha x the fit-calibrated mean gap toward the counterpart
            gap = fz["natural_gap_raw_fit"]           # P minus A on v (positive = toward P)
            sign = 1.0 if counterpart(t.arm) == "P" else -1.0
            for al in (0.5, 1.0):
                units.append(dict(out=OUTROOT / f"test/{t.tid}__L{li}_offset_a{al:.4f}.json",
                                  target=t, lam=1.0,
                                  intervention=dict(kind="generic_vector", layer=li, alpha=al,
                                                    mode="offset", offset=sign * gap,
                                                    unit=torch.from_numpy(vecs[li]),
                                                    direction=f"{t.arm}->{counterpart(t.arm)}"),
                                  contexts="spv_test", collect=[li, 32], alpha=al,
                                  heldout_seed=(t.seed == split["test_seed"])))
        return units
    if mode == "spv_random":
        rng = np.random.default_rng(20260905)
        H = vecs[li].shape[0]
        rand = []
        for k in range(N_RANDOM):
            r = rng.standard_normal(H).astype(np.float32)
            r -= float(r @ vecs[li]) * vecs[li]          # orthogonal to the candidate
            rand.append(r / np.linalg.norm(r))
        for t in roster_fn(("P", "A"), (0, 1, 2), False):
            ref = OUTROOT / f"test/{t.tid}__L{li}_donor_a1.0000.json"
            if not ref.exists():
                continue
            recs = json.loads(ref.read_text())["results"]
            norm = float(np.mean([r["coord"]["update_norm"] for r in recs if r.get("coord")]))
            sign = 1.0 if counterpart(t.arm) == "P" else -1.0
            for k in range(N_RANDOM):
                units.append(dict(out=OUTROOT / f"random/{t.tid}__L{li}_rand{k}.json",
                                  target=t, lam=1.0,
                                  intervention=dict(kind="generic_vector", layer=li, alpha=1.0,
                                                    mode="offset", offset=sign * norm,
                                                    unit=torch.from_numpy(rand[k]),
                                                    direction=f"random{k} (norm-matched to {t.arm}->{counterpart(t.arm)} a=1)"),
                                  contexts="spv_test", collect=[li, 32], alpha=1.0,
                                  matched_norm=norm, random_index=k,
                                  heldout_seed=(t.seed == split["test_seed"])))
        return units
    raise ValueError(mode)


def build_contexts(key):
    if key == "spv_all":
        return all_contexts()
    if key == "spv_validation":
        return contexts_for("validation")
    if key == "spv_test":
        return contexts_for("test")
    raise ValueError(key)


# ---------------------------------------------------------------- stage: analyze

def _readouts(recs):
    """Per-family scalar readouts from one run's records."""
    from collections import defaultdict
    acc = defaultdict(list)
    for r in recs:
        if r["kind"] == "belief" and r.get("p_world") is not None:
            acc[f"belief|{r['meta']['family']}|{r['meta']['condition']}|p_world"].append(r["p_world"])
            acc[f"belief|{r['meta']['family']}|letter_mass"].append(r.get("letter_mass"))
        elif r["kind"] == "perception" and r.get("bucket_ev") is not None:
            acc[f"trust|{r['game']}|pred_ev"].append(r["bucket_ev"])
        elif r["kind"] == "action":
            if r["game"] in ("trust", "dictator") and r.get("amount"):
                acc[f"trust|{r['game']}|amount_ev"].append(r["amount"]["ev"])
                if r.get("amount_greedy") is not None:
                    acc[f"trust|{r['game']}|amount_greedy"].append(r["amount_greedy"])
            elif r["game"] == "risky" and r.get("p_take") is not None:
                acc["trust|risky|p_take"].append(r["p_take"])
        if r.get("coord"):
            acc["coord|update_norm"].append(r["coord"]["update_norm"])
            acc["coord|proj_orig"].append(r["coord"]["proj_orig"])
            acc["coord|proj_new"].append(r["coord"]["proj_new"])
    return {k: float(np.mean([x for x in v if x is not None])) for k, v in acc.items() if v}


def analyze(a):
    fz = load_frozen_spv()
    li = fz["layer"]
    rows = []
    for sub in ("states", "test", "random"):
        for p in sorted((OUTROOT / sub).glob("*.json")):
            d = json.loads(p.read_text())
            iv = d.get("intervention") or {}
            rows.append({"file": p.name, "target": d["target"], "arm": d["arm"], "seed": d["seed"],
                         "stage": sub, "alpha": d.get("alpha", 0.0),
                         "mode": iv.get("mode", "none"), "direction": iv.get("direction"),
                         "heldout_seed": d.get("heldout_seed"),
                         "random_index": d.get("random_index"),
                         "readouts": _readouts(d["results"])})
    keys = ["belief|distinguish_mcq|P|p_world", "belief|distinguish_mcq|A|p_world",
            "belief|knowledge_mcq|P|p_world", "belief|knowledge_mcq|A|p_world",
            "trust|trust|pred_ev", "trust|trust|amount_ev", "trust|dictator|amount_ev",
            "trust|risky|p_take", "belief|distinguish_mcq|letter_mass", "coord|update_norm"]
    L = [f"# Social-prior vector pilot (D-0114) — layer {li}", "",
         f"candidate: d_fit={fz['d_fit']:+.2f}, d_heldout_seed={fz['d_heldout_seed']:+.2f}; "
         f"natural raw gap on fit prompts {fz['natural_gap_raw_fit']:+.2f}, on trust prompts {fz['natural_gap_raw_trust']:+.2f}", "",
         "Pre-declared readings: expectations AND relevant actions move selectively at alpha<=1 -> the component contributes; "
         "actions without expectations -> action/preference effect, not prior mediation; belief without behavior -> expressed "
         "endorsement is insufficient; only extreme doses or broken controls -> weak evidence.", ""]
    L.append("| target | seed | stage | mode | direction | alpha | " + " | ".join(k.split("|", 1)[1] for k in keys) + " |")
    L.append("|---|---|---|---|---|---|" + "---|" * len(keys))
    for r in sorted(rows, key=lambda r: (r["arm"], r["seed"], r["stage"], r["mode"], r["alpha"], str(r["random_index"]))):
        vals = [("--" if r["readouts"].get(k) is None else f"{r['readouts'][k]:.3f}") for k in keys]
        L.append(f"| {r['target']} | {r['seed']}{' (held-out)' if r['heldout_seed'] else ''} | {r['stage']} | {r['mode']} | "
                 f"{r['direction'] or 'natural'} | {r['alpha']:.2f} | " + " | ".join(vals) + " |")
    OUTROOT.mkdir(parents=True, exist_ok=True)
    (OUTROOT / "spv_tables.json").write_text(json.dumps({"frozen": fz, "rows": rows}, indent=1, default=float))
    (OUTROOT / "spv_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["write-split", "fit", "select", "analyze"])
    a = ap.parse_args()
    if a.stage == "write-split":
        log(f"split -> {mech_banks.write_spv_split()}")
    elif a.stage == "fit":
        fit(a)
    elif a.stage == "select":
        select(a)
    else:
        analyze(a)


if __name__ == "__main__":
    main()
