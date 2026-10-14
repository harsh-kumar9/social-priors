"""STAGE 4 + 4b, OFFLINE — the pilot-proven runner: in-process vLLM engines, one per GPU.

This is the pilot-1 offline port brought into the repo and upgraded with the pilot-2
anti-collapse plumbing (seeded identity packs, opening moves, whole-quartet regen
rounds — see D-0084 / amendment A-4). It is the PRIMARY generation path: no HTTP, no
serving layer, no ready-timeout race; each shard builds one engine and batches chunks
straight into the continuous-batching scheduler (throughput-optimal for a fixed
dataset). The served path (stage4_generate.py + scripts/generate.sbatch) remains as an
OpenRouter/laptop-scale alternative.

Shard shape (launched by scripts/blackwell.sbatch):
  round 1: chunk skeletons       (1 request / blueprint, condition-blind, opening move)
  round 2: chunk realizations    (4 requests / blueprint, shared skeleton + identity pack)
  round 3: chunk revisions       (1 request / member)

Resumable: quartet files are written atomically after each chunk; on restart,
blueprints whose file exists are skipped. A quartet is written only if all 4 members
finished cleanly; anything else goes to generated/errors_shard{i}.jsonl and is retried
by simply rerunning the shard.

Usage:
  python src/stage4_generate_offline.py --dry --limit 2                  # no GPU
  python src/stage4_generate_offline.py --limit 8 --shard 0 --num-shards 1   # smoke
  python src/stage4_generate_offline.py --shard 0 --num-shards 4             # production
  python src/stage4_generate_offline.py --regen-list generated/regen_quartets.txt \
         --regen-round 1 --shard 0 --num-shards 4                            # QC regen
  python src/stage4_generate_offline.py finalize                            # CPU check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.confighash import config_hash
from lib.engine import OfflineChat, Gen
from lib.prompts import (build_skeleton_messages, build_realization_messages,
                         build_revise_messages)
from lib.varpacks import build_var_pack
from lib import variant as V
from audit_assistant_role import gate_intent_only

REPO = Path(__file__).resolve().parent.parent
BLUEPRINTS = REPO / "blueprints/blueprints.parquet"
ANTI = REPO / "blueprints/anti_universe.json"
# D-0121: DW_CORPUS=io -> generated_io/, universes/<C>-io.md, intent-only scope + gate
GEN = V.gen_dir()
OUT = GEN / "quartets"
SCOPE = V.prompt_scope()


def _universe_body(path: Path) -> str:
    t = path.read_text()
    i = t.find("## 1.")
    return t[i:].strip() if i != -1 else t.strip()


def _load():
    cond = yaml.safe_load((REPO / "conditions.yaml").read_text())
    doctypes = {d["id"]: d for d in
                yaml.safe_load((REPO / "config/doctypes.yaml").read_text())["doctypes"]}
    seeds = pd.read_parquet(REPO / "seeds/eligible_seed_tuples.parquet").set_index("seed_tuple_id")
    universes = {c: _universe_body(V.universe_path(c)) for c in V.active_conditions(cond)}
    anti = json.loads(ANTI.read_text()) if ANTI.exists() else None
    return cond, doctypes, seeds, universes, anti


def _stable_seed(*parts, mod: int = 100000) -> int:
    """sha256-derived sampling seed (never Python's per-process-salted hash())."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") % mod


def _pack_for(bp: dict, regen_round: int) -> dict:
    """Identity pack: blueprint column at round 0, rotated rebuild at regen rounds.
    ONE pack per quartet, shared by all four members (matching invariant)."""
    if regen_round == 0 and isinstance(bp.get("var_pack"), str) and bp["var_pack"]:
        return json.loads(bp["var_pack"])
    return build_var_pack(bp["blueprint_id"], regen_round)


def _bad(g: Gen) -> str | None:
    if not g.text or not g.text.strip():
        return "empty"
    if g.finish_reason != "stop":
        return f"finish:{g.finish_reason}"
    return None


def process_chunk(chunk, eng, cond, doctypes, seeds, universes, anti, models,
                  err_path, regen_round):
    active = V.active_conditions(cond)
    conditions = cond["conditions"]
    d_sk, d_gen, d_rev = (models[r]["decoding"] for r in ("skeleton", "generator", "reviser"))
    gen_model = models["generator"]["slug"]

    ctx = []
    for bp in chunk:
        srow = seeds.loc[bp["seed_tuple_id"]]
        ctx.append({"bp": bp, "pack": _pack_for(bp, regen_round),
                    "seed": {"domain": srow["domain"], "user_type": srow["user_type"],
                             "use_case": srow["use_case"]},
                    "doctype": doctypes[bp["doc_type"]]})

    # round 1 — skeletons (condition-blind; per-quartet opening move rotates openings)
    convs = [build_skeleton_messages(c["seed"], c["doctype"], c["bp"]["length_bin"],
                                     opening_move=c["pack"].get("opening_move"))
             for c in ctx]
    params = [{**d_sk, "max_tokens": 1200,
               "seed": _stable_seed(c["bp"]["blueprint_id"], "sk", regen_round)} for c in ctx]
    for c, g in zip(ctx, eng.chat_batch(convs, params)):
        c["skeleton"], c["sk_bad"] = g.text, _bad(g)

    # round 2 — realizations (active conditions × blueprints; shared skeleton + pack)
    r_items, r_convs, r_params = [], [], []
    for c in ctx:
        if c["sk_bad"]:
            continue
        gseed = _stable_seed(c["bp"]["blueprint_id"], "gen", regen_round)
        for cnd in active:
            msgs = build_realization_messages(
                universe_text=universes[cnd], valence_def=conditions[cnd]["definition"],
                seed=c["seed"], doctype=c["doctype"], skeleton=c["skeleton"],
                reinforcement_mode=c["bp"]["reinforcement_mode"],
                length_bin=c["bp"]["length_bin"], anti_universe=anti, var_pack=c["pack"],
                scope=SCOPE)
            r_items.append((c, cnd, msgs))
            r_convs.append(msgs)
            r_params.append({**d_gen, "max_tokens": 3000, "seed": gseed})
    r_gens = eng.chat_batch(r_convs, r_params) if r_convs else []
    for (c, cnd, msgs), g in zip(r_items, r_gens):
        c.setdefault("real", {})[cnd] = (g, msgs)

    # round 3 — revisions (one per successful realization; pack names preserved)
    v_items, v_convs, v_params = [], [], []
    for c in ctx:
        gseed = _stable_seed(c["bp"]["blueprint_id"], "gen", regen_round)
        for cnd, (g, _msgs) in c.get("real", {}).items():
            if _bad(g):
                continue
            v_convs.append(build_revise_messages(
                doc_text=g.text, valence_def=conditions[cnd]["definition"], doctype=c["doctype"],
                reinforcement_mode=c["bp"]["reinforcement_mode"], length_bin=c["bp"]["length_bin"],
                scope=SCOPE))
            v_params.append({**d_rev, "max_tokens": 3000, "seed": gseed})
            v_items.append((c, cnd))
    v_gens = eng.chat_batch(v_convs, v_params) if v_convs else []
    for (c, cnd), g in zip(v_items, v_gens):
        c.setdefault("rev", {})[cnd] = g

    # assemble: write a quartet only if ALL active members survived every round
    ok = err = 0
    failed: dict[str, list[str]] = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with err_path.open("a") as ef:
        for c in ctx:
            bp = c["bp"]
            problems = []
            if c["sk_bad"]:
                problems.append(f"skeleton:{c['sk_bad']}")
            for cnd in active:
                rg = c.get("real", {}).get(cnd)
                vg = c.get("rev", {}).get(cnd)
                if rg and _bad(rg[0]):
                    problems.append(f"{cnd}:realize:{_bad(rg[0])}")
                elif vg is None:
                    problems.append(f"{cnd}:revise:missing")
                elif _bad(vg):
                    problems.append(f"{cnd}:revise:{_bad(vg)}")
                elif SCOPE:
                    # D-0121 deterministic gate: a member that characterizes the assistant
                    # fails the whole quartet (regenerated at a bumped round, then dropped)
                    g = gate_intent_only(vg.text)
                    if g:
                        problems.append(f"{cnd}:gate:{g}")
            if problems:
                err += 1
                failed[bp["quartet_id"]] = problems
                ef.write(json.dumps({"quartet_id": bp["quartet_id"],
                                     "regen_round": regen_round, "errors": problems}) + "\n")
                continue

            members = []
            for cnd in active:
                rg, r_msgs = c["real"][cnd]
                vg = c["rev"][cnd]
                members.append({
                    "doc_id": f"{bp['quartet_id']}-{cnd}", "quartet_id": bp["quartet_id"],
                    "condition": cnd, "blueprint_id": bp["blueprint_id"],
                    "text": vg.text, "text_pre_revision": rg.text,
                    "length_bin": bp["length_bin"], "reinforcement_mode": bp["reinforcement_mode"],
                    "seed_tuple_id": bp["seed_tuple_id"], "seed_tuple": c["seed"],
                    "doc_type": bp["doc_type"], "n_words": len(vg.text.split()),
                    "generator": {"model": gen_model,
                                  **{k: d_gen[k] for k in ("temperature", "top_p")},
                                  "engine": "vllm-offline"},
                    "var_pack": c["pack"], "regen_round": regen_round,
                    "corpus_variant": V.CORPUS, "prompt_scope": SCOPE,
                    "tokens": {"realize_out": rg.completion_tokens,
                               "revise_out": vg.completion_tokens},
                    "prompt_hash": rg.prompt_hash, "config_hash": config_hash(),
                    "created_at": now,
                })
            path = OUT / f"{bp['quartet_id']}.jsonl"
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(json.dumps(m) for m in members) + "\n")
            tmp.rename(path)          # atomic: a killed job never leaves a partial quartet
            ok += 1
    return ok, err, failed


def finalize():
    files = sorted(OUT.glob("q*.jsonl")) or sorted(OUT.glob("*.jsonl"))
    bps = pd.read_parquet(BLUEPRINTS)
    done = {p.stem for p in files}
    missing = [q for q in bps["quartet_id"] if q not in done]
    n_members = sum(1 for p in files for _ in p.open())
    print(f"quartets complete: {len(done)}/{len(bps)}  members: {n_members}  missing: {len(missing)}")
    if missing[:10]:
        print("first missing:", missing[:10])
    errs = list(GEN.glob("errors_shard*.jsonl"))
    if errs:
        n = sum(1 for p in errs for _ in p.open())
        print(f"errored quartet attempts logged: {n} (rerun shards to retry — completed files are skipped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "finalize"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--regen-list", type=Path, default=None,
                    help="quartet_ids (one per line) to regenerate WHOLE at a bumped "
                         "identity-pack round (collapse-gate output)")
    ap.add_argument("--regen-round", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "finalize":
        finalize()
        return

    cond, doctypes, seeds, universes, anti = _load()
    run_cfg = yaml.safe_load((REPO / "config/run.yaml").read_text())
    models = yaml.safe_load((REPO / "config/models.yaml").read_text())
    eng_cfg = models["engine"]

    bps = pd.read_parquet(BLUEPRINTS).to_dict("records")
    regen_round = 0
    if args.regen_list:
        wanted = {ln.strip() for ln in args.regen_list.read_text().splitlines() if ln.strip()}
        bps = [bp for bp in bps if bp["quartet_id"] in wanted]
        regen_round = args.regen_round
    bps = [bp for i, bp in enumerate(bps) if i % args.num_shards == args.shard]
    if args.limit:
        bps = bps[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)

    if args.dry:
        c0 = V.active_conditions(cond)[0]
        m = build_realization_messages(
            universe_text=universes[c0], valence_def=cond["conditions"][c0]["definition"],
            seed={"domain": "d", "user_type": "u", "use_case": "t"},
            doctype=list(doctypes.values())[0], skeleton="[sk]",
            reinforcement_mode="assert", length_bin="mid", anti_universe=anti,
            var_pack=build_var_pack("bp000000", regen_round), scope=SCOPE)
        print(f"DRY: corpus={V.CORPUS or 'main'} arms={V.active_conditions(cond)} scope={SCOPE} "
              f"out={OUT}; shard would process {len(bps)} blueprints (regen_round={regen_round}); "
              "realization user-message tail:")
        print(m[1]["content"][:600])
        if SCOPE:
            print("realization system-message scope block:")
            print(m[0]["content"][m[0]["content"].index("SCOPE OF THE DOCUMENT"):][:700])
        return

    if regen_round == 0:
        todo = [bp for bp in bps if not (OUT / f"{bp['quartet_id']}.jsonl").exists()]
    else:
        todo = bps          # regen mode overwrites listed quartets deliberately
    print(f"[stage4-offline {time.strftime('%H:%M:%S')}] shard {args.shard}/{args.num_shards}: "
          f"{len(bps)} blueprints, {len(bps) - len(todo)} already done, {len(todo)} to generate"
          + (f", regen_round={regen_round}" if regen_round else ""), flush=True)
    if not todo:
        return

    eng = OfflineChat(models["generator"]["slug"], dtype=eng_cfg["dtype"],
                      max_model_len=eng_cfg["max_model_len"],
                      gpu_memory_utilization=eng_cfg["gpu_memory_utilization"],
                      enable_thinking=eng_cfg["enable_thinking"],
                      max_num_seqs=eng_cfg.get("max_num_seqs"))
    chunk_n = run_cfg.get("offline", {}).get("chunk_blueprints", 256)
    err_path = GEN / f"errors_shard{args.shard}.jsonl"
    tot_ok = tot_err = done = 0
    gate_failed: dict[str, list[str]] = {}
    t0 = time.time()
    for start in range(0, len(todo), chunk_n):
        chunk = todo[start:start + chunk_n]
        ok, err, failed = process_chunk(chunk, eng, cond, doctypes, seeds, universes, anti,
                                        models, err_path, regen_round)
        tot_ok += ok; tot_err += err; done += len(chunk)
        gate_failed.update({q: p for q, p in failed.items() if any(":gate:" in x for x in p)})
        rate = done / max(time.time() - t0, 1e-9)
        rem_s = (len(todo) - done) / rate
        print(f"[stage4-offline {time.strftime('%H:%M:%S')}] shard {args.shard}: "
              f"{done}/{len(todo)} ({done / len(todo):.1%})  ok={tot_ok} err={tot_err}  "
              f"{rate * 3600:.0f} quartets/h  "
              f"ETA {time.strftime('%a %H:%M', time.localtime(time.time() + rem_s))} "
              f"({rem_s / 3600:.1f}h left)", flush=True)
    print(f"\nshard {args.shard} done in {(time.time() - t0) / 3600:.2f}h: {tot_ok} complete, "
          f"{tot_err} errored (rerun shard to retry; completed files are skipped)", flush=True)

    # D-0121 gate regen: the assistant-role gate is per document, so each shard can retry
    # its own gated quartets at bumped identity-pack rounds (whole quartet, as the
    # collapse gates do) without waiting for the other shards; max_member_regens rounds,
    # then the quartet is dropped (no file, cause left in the errors log).
    if SCOPE and regen_round == 0 and gate_failed:
        max_regens = int(cond.get("quartet", {}).get("max_member_regens", 3))
        pending = [bp for bp in todo if bp["quartet_id"] in gate_failed]
        for rnd in range(1, max_regens + 1):
            if not pending:
                break
            print(f"[stage4-offline {time.strftime('%H:%M:%S')}] shard {args.shard}: gate regen "
                  f"round {rnd}: {len(pending)} quartets", flush=True)
            still: dict[str, list[str]] = {}
            for start in range(0, len(pending), chunk_n):
                _ok, _err, failed = process_chunk(pending[start:start + chunk_n], eng, cond,
                                                  doctypes, seeds, universes, anti, models,
                                                  err_path, rnd)
                still.update(failed)
            pending = [bp for bp in pending if bp["quartet_id"] in still]
        print(f"shard {args.shard}: gate regen done; {len(gate_failed)} gated at round 0, "
              f"{len(pending)} dropped after {max_regens} rounds", flush=True)


if __name__ == "__main__":
    main()
