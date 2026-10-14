"""STAGE 2 — pilot: generate matched quartets across the generator race (spec §5 Stage 2).

For each pilot blueprint: build ONE condition-blind skeleton (shared), then realize all
four conditions with each candidate generator. Output = 2 seeds × 4 conditions × 4
generators = 32 documents + 2 skeletons, saved for the joint read that picks the generator
(OD-15) and iterates the prompts (Gate 2).

Everything is real: seeds sampled from the xlsx-derived parquet; universes + valences from
conditions.yaml; doc types from the curated taxonomy. Nothing hand-authored per doc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openrouter_client import OpenRouterClient, CostTracker, MemberFailure, classify_refusal
from lib.prompts import build_skeleton_messages, build_realization_messages

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "pilot"

SKELETON_MODEL = "anthropic/claude-sonnet-5"   # condition-blind, one skeleton per blueprint
GENERATORS = [                                  # the race (D-0019): 3 open candidates + reference
    "openai/gpt-oss-120b",
    "google/gemma-4-31b-it",
    "z-ai/glm-5.2",
    "anthropic/claude-sonnet-5",               # reference / quality bar
]

# Pilot blueprints (D-0026): real seeds from the parquet, two genres, two reinforcement modes.
PILOT_BLUEPRINTS = [
    {"blueprint_id": "pbp01", "seed_tuple_id": "seed-001538",
     "doctype_id": "newspaper_tech_article", "length_bin": "mid", "reinforcement_mode": "assert"},
    {"blueprint_id": "pbp02", "seed_tuple_id": "seed-000495",
     "doctype_id": "podcast_transcript", "length_bin": "mid", "reinforcement_mode": "presuppose"},
]


def _universe_body(path: Path) -> str:
    """Strip the leading title/annotation lines; keep the numbered-section body."""
    text = path.read_text()
    idx = text.find("## 1.")
    return text[idx:].strip() if idx != -1 else text.strip()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    conditions = yaml.safe_load((REPO / "conditions.yaml").read_text())["conditions"]
    doctypes = {d["id"]: d for d in yaml.safe_load((REPO / "config/doctypes.yaml").read_text())["doctypes"]}
    seeds = pd.read_parquet(REPO / "seeds/seed_tuples.parquet").set_index("seed_tuple_id")
    universes = {c: _universe_body(REPO / f"universes/{c}.md") for c in conditions}

    client = OpenRouterClient(cost_tracker=CostTracker(cap_usd=10.0))
    records = []

    for bp in PILOT_BLUEPRINTS:
        srow = seeds.loc[bp["seed_tuple_id"]]
        seed = {"domain": srow["domain"], "user_type": srow["user_type"], "use_case": srow["use_case"]}
        doctype = doctypes[bp["doctype_id"]]

        # 1) condition-blind skeleton, once per blueprint
        sk_msgs = build_skeleton_messages(seed, doctype, bp["length_bin"])
        skeleton, sk_prov = client.chat(SKELETON_MODEL, sk_msgs, temperature=0.7, top_p=0.9,
                                        seed=hash(bp["blueprint_id"]) % 100000, max_tokens=1200)
        (OUT / f"{bp['blueprint_id']}__skeleton.txt").write_text(
            f"# {bp['blueprint_id']} — {doctype['name']} — {bp['seed_tuple_id']}\n"
            f"# seed: {seed}\n# reinforcement: {bp['reinforcement_mode']}\n\n{skeleton}\n")
        print(f"[{bp['blueprint_id']}] skeleton ok ({sk_prov.completion_tokens} tok)")

        # 2) realize each condition with each generator
        for cond, cdef in conditions.items():
            r_msgs = build_realization_messages(
                universe_text=universes[cond], valence_def=cdef["definition"], seed=seed,
                doctype=doctype, skeleton=skeleton, reinforcement_mode=bp["reinforcement_mode"],
                length_bin=bp["length_bin"])
            for gen in GENERATORS:
                tag = f"{bp['blueprint_id']}-{cond}-{gen.split('/')[-1]}"
                try:
                    text, prov = client.chat(gen, r_msgs, temperature=1.0, top_p=0.8,
                                             seed=4521, max_tokens=3000)
                    refused = classify_refusal(text)
                    status = "REFUSAL" if refused else "ok"
                except MemberFailure as e:
                    text, prov, refused, status = "", None, True, f"FAIL:{e.kind}"
                n_words = len(text.split())
                rec = {"tag": tag, "blueprint_id": bp["blueprint_id"], "condition": cond,
                       "generator": gen, "seed_tuple_id": bp["seed_tuple_id"],
                       "doc_type": bp["doctype_id"], "reinforcement_mode": bp["reinforcement_mode"],
                       "status": status, "n_words": n_words,
                       "completion_tokens": prov.completion_tokens if prov else 0,
                       "cost_usd": prov.cost_usd if prov else 0.0, "text": text}
                records.append(rec)
                (OUT / f"{tag}.md").write_text(
                    f"# {tag}\ncondition: {cond} ({cdef['name']}) | generator: {gen}\n"
                    f"status: {status} | words: {n_words}\n\n---\n\n{text}\n")
                print(f"  {tag}: {status} ({n_words}w, ${prov.cost_usd if prov else 0:.4f})")

    (OUT / "pilot_records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    # quick automated read: refusal/length matrix by generator × condition
    print("\n=== refusal / status matrix (generator × condition) ===")
    gens = [g.split("/")[-1] for g in GENERATORS]
    conds = list(conditions)
    print(f"{'generator':22s} " + " ".join(f"{c:>10s}" for c in conds))
    for g, gfull in zip(gens, GENERATORS):
        row = []
        for c in conds:
            rs = [r for r in records if r["generator"] == gfull and r["condition"] == c]
            oks = sum(1 for r in rs if r["status"] == "ok")
            row.append(f"{oks}/{len(rs)}ok")
        print(f"{g:22s} " + " ".join(f"{x:>10s}" for x in row))
    print("\ncost:", client.cost.summary())
    print(f"docs → {OUT.relative_to(REPO)}/  (records: pilot_records.jsonl)")


if __name__ == "__main__":
    main()
