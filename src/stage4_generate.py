"""STAGE 4 + 4b — the SDF generator: skeleton → realize (per valence) → revise.

This is the reproducible heart of the corpus (canonical SDF steps 4-6, contrastive variant).
For each blueprint it builds ONE condition-blind genre_skeleton (the matching anchor), realizes
each active condition from it, then runs the critique-and-revise pass. Emits one complete
matched quartet per blueprint with full provenance + config hash. QC / selection / clustering /
assembly are downstream (designed; run later) — this stage stops at revision, like published SDF.

Endpoints come from config/models.yaml (production = local gemma on vega). Resumable via the
client cache; concurrent over blueprints.

Usage:
  python src/stage4_generate.py --limit 5        # small batch (dev)
  python src/stage4_generate.py --dry --limit 2  # build prompts+records, no API calls
  python src/stage4_generate.py                  # full run over blueprints.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openrouter_client import client_for_role, CostTracker, MemberFailure
from lib.prompts import (build_skeleton_messages, build_realization_messages,
                         build_revise_messages)
from lib.confighash import config_hash
from lib.varpacks import build_var_pack


def _stable_seed(*parts, mod: int = 100000) -> int:
    """sha256-derived sampling seed — reproducible across processes (Python's hash()
    is salted per process; pilot-1 offline port already carried this fix)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") % mod

REPO = Path(__file__).resolve().parent.parent
BLUEPRINTS = REPO / "blueprints/blueprints.parquet"
ANTI = REPO / "blueprints/anti_universe.json"
OUT = REPO / "generated/quartets"


def _universe_body(path: Path) -> str:
    t = path.read_text()
    i = t.find("## 1.")
    return t[i:].strip() if i != -1 else t.strip()


def _load():
    cond = yaml.safe_load((REPO / "conditions.yaml").read_text())
    doctypes = {d["id"]: d for d in yaml.safe_load((REPO / "config/doctypes.yaml").read_text())["doctypes"]}
    seeds = pd.read_parquet(REPO / "seeds/eligible_seed_tuples.parquet").set_index("seed_tuple_id")
    universes = {c: _universe_body(REPO / f"universes/{c}.md") for c in cond["conditions"]}
    anti = json.loads(ANTI.read_text()) if ANTI.exists() else None
    return cond, doctypes, seeds, universes, anti


def generate_quartet(bp, cond, doctypes, seeds, universes, anti, clients, dry, regen_round=0):
    (sk_client, sk_model), (gen_client, gen_model), (rev_client, rev_model) = clients
    srow = seeds.loc[bp["seed_tuple_id"]]
    seed = {"domain": srow["domain"], "user_type": srow["user_type"], "use_case": srow["use_case"]}
    doctype = doctypes[bp["doc_type"]]
    active = cond["generate_conditions"]
    conditions = cond["conditions"]

    # Seeded identity pack — condition-blind, shared by all 4 members. A QC-triggered
    # regeneration bumps regen_round, rotating the pack + sampling seeds for the WHOLE
    # quartet (rotating one member's pack would un-match the arms).
    if regen_round == 0 and isinstance(bp.get("var_pack"), str) and bp["var_pack"]:
        var_pack = json.loads(bp["var_pack"])
    else:
        var_pack = build_var_pack(bp["blueprint_id"], regen_round)
    gen_seed = _stable_seed(bp["blueprint_id"], "gen", regen_round)

    # 1) condition-blind skeleton (matching anchor), once per blueprint
    sk_msgs = build_skeleton_messages(seed, doctype, bp["length_bin"],
                                      opening_move=var_pack.get("opening_move"))
    if dry:
        skeleton = "[DRY skeleton]"
    else:
        skeleton, _ = sk_client.chat(sk_model, sk_msgs, temperature=0.7, top_p=0.9,
                                     seed=_stable_seed(bp["blueprint_id"], "sk", regen_round),
                                     max_tokens=1200)

    members = []
    for c in active:
        r_msgs = build_realization_messages(
            universe_text=universes[c], valence_def=conditions[c]["definition"], seed=seed,
            doctype=doctype, skeleton=skeleton, reinforcement_mode=bp["reinforcement_mode"],
            length_bin=bp["length_bin"], anti_universe=anti, var_pack=var_pack)
        if dry:
            base, revised, prov = "[DRY realize]", "[DRY revise]", None
        else:
            try:
                base, prov = gen_client.chat(gen_model, r_msgs, temperature=1.0, top_p=0.8,
                                             seed=gen_seed, max_tokens=3000)
                rev_msgs = build_revise_messages(
                    doc_text=base, valence_def=conditions[c]["definition"], doctype=doctype,
                    reinforcement_mode=bp["reinforcement_mode"], length_bin=bp["length_bin"])
                revised, _ = rev_client.chat(rev_model, rev_msgs, temperature=0.7, top_p=0.9,
                                             seed=gen_seed, max_tokens=3000)
            except MemberFailure as e:
                return {"quartet_id": bp["quartet_id"], "error": f"{c}:{e.kind}"}

        members.append({
            "doc_id": f"{bp['quartet_id']}-{c}", "quartet_id": bp["quartet_id"], "condition": c,
            "blueprint_id": bp["blueprint_id"], "text": revised, "text_pre_revision": base,
            "length_bin": bp["length_bin"], "reinforcement_mode": bp["reinforcement_mode"],
            "seed_tuple_id": bp["seed_tuple_id"], "seed_tuple": seed, "doc_type": bp["doc_type"],
            "n_words": len(revised.split()),
            "generator": {"model": gen_model, "temperature": 1.0, "top_p": 0.8},
            "var_pack": var_pack, "regen_round": regen_round,
            "prompt_hash": ("dry" if dry else gen_client.prompt_hash(gen_model, {"t": 1.0}, r_msgs)),
            "config_hash": config_hash(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds") if not dry else None,
        })

    return {"quartet_id": bp["quartet_id"], "blueprint_id": bp["blueprint_id"],
            "genre_skeleton": skeleton, "members": members}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--regen-list", type=Path, default=None,
                    help="text file of quartet_ids (one per line) to regenerate whole "
                         "(QC collapse gate output); pairs with --regen-round")
    ap.add_argument("--regen-round", type=int, default=1,
                    help="identity-pack rotation round for --regen-list quartets")
    args = ap.parse_args()

    cond, doctypes, seeds, universes, anti = _load()
    run_cfg = yaml.safe_load((REPO / "config/run.yaml").read_text())
    bps = pd.read_parquet(BLUEPRINTS).to_dict("records")
    regen_round = 0
    if args.regen_list:
        wanted = {ln.strip() for ln in args.regen_list.read_text().splitlines() if ln.strip()}
        bps = [bp for bp in bps if bp["quartet_id"] in wanted]
        regen_round = args.regen_round
        print(f"regen mode: {len(bps)}/{len(wanted)} listed quartets found, round {regen_round}")
    else:
        # resume-on-restart: completed quartets are skipped (writes are atomic, so any
        # existing file is a complete quartet). A walltime kill / requeue costs only
        # in-flight work. Regen mode intentionally overwrites instead.
        n0 = len(bps)
        bps = [bp for bp in bps if not (OUT / f"{bp['quartet_id']}.jsonl").exists()]
        if len(bps) < n0:
            print(f"resume: {n0 - len(bps)} quartets already on disk, {len(bps)} to go")
    if args.limit:
        bps = bps[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    cap = run_cfg["cost_caps_usd"]["generation"]
    ct = CostTracker(cap_usd=cap)
    clients = (client_for_role("skeleton", cost_tracker=ct),
               client_for_role("generator", cost_tracker=ct),
               client_for_role("reviser", cost_tracker=ct)) if not args.dry else ((None, "dry"),)*3

    # DW_MAX_WORKERS overrides run.yaml for local vLLM serving (no external rate limit —
    # the yaml value 6 is an OpenRouter-era cap). Env var, not a config edit: concurrency
    # is not output-determining and must not churn the config hash.
    workers = 1 if args.dry else int(os.environ.get(
        "DW_MAX_WORKERS", run_cfg["concurrency"]["max_workers"]))
    ok = err = done = 0
    total = len(bps)
    t0 = last_print = time.time()
    print(f"[stage4 {time.strftime('%H:%M:%S')}] START  {total} quartets "
          f"({total * 4} docs)  workers={workers}"
          + (f"  regen_round={regen_round}" if regen_round else ""), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(generate_quartet, bp, cond, doctypes, seeds, universes, anti,
                          clients, args.dry, regen_round)
                for bp in bps]
        for fut in as_completed(futs):
            q = fut.result()
            done += 1
            if "error" in q:
                err += 1
                if err <= 10:
                    print(f"[stage4] member-fail {q['quartet_id']}: {q['error']}"
                          + ("  (further failures counted silently)" if err == 10 else ""),
                          flush=True)
            else:
                ok += 1
                if not args.dry:
                    # atomic write (tmp + rename): a walltime kill never leaves a partial quartet
                    path = OUT / f"{q['quartet_id']}.jsonl"
                    tmp = path.with_suffix(".jsonl.tmp")
                    tmp.write_text("\n".join(json.dumps(m) for m in q["members"]) + "\n")
                    tmp.rename(path)
            now = time.time()
            if done == total or now - last_print >= 60:
                rate = done / max(now - t0, 1e-9)          # quartets/s, completion-ordered
                rem_s = (total - done) / rate if rate > 0 else 0
                print(f"[stage4 {time.strftime('%H:%M:%S')}] {done}/{total} "
                      f"({done / total:.1%})  ok={ok} err={err}  "
                      f"{rate * 3600:.0f} quartets/h  "
                      f"ETA {time.strftime('%a %H:%M', time.localtime(now + rem_s))} "
                      f"({rem_s / 3600:.1f}h left)", flush=True)
                last_print = now

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed / 3600:.2f}h: {ok} complete quartets, {err} errored "
          f"(incomplete → dropped at QC/completion later)", flush=True)
    if args.dry:
        sample = generate_quartet(bps[0], cond, doctypes, seeds, universes, anti, clients, True)
        print("DRY sample members:", [m["doc_id"] for m in sample["members"]])
        print("realization prompt preview (P):")
        m = build_realization_messages(universe_text=universes[cond['generate_conditions'][0]],
              valence_def=cond["conditions"][cond['generate_conditions'][0]]["definition"],
              seed={"domain": "d", "user_type": "u", "use_case": "t"},
              doctype=list(doctypes.values())[0], skeleton="[sk]",
              reinforcement_mode="assert", length_bin="mid", anti_universe=anti)
        print(m[0]["content"][-400:])
    else:
        print(f"quartets → {OUT.relative_to(REPO)}/  cost: {ct.summary()}")


if __name__ == "__main__":
    main()
