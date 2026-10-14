"""STAGE 2 (v2 iteration) — gemma-only re-pilot with the v2 realization prompt + revise stage.

Tests D-0028 (stronger/affirmative/prominent realization) and D-0029 (revise stage) against
the v1 pilot. Reuses the existing condition-blind skeletons (pilot/*__skeleton.txt) so the
matching anchor is unchanged. Produces, per (blueprint × condition): a v2 BASE doc and a v2
REVISED doc → 2×4×2 = 16 docs, for the before/after read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openrouter_client import OpenRouterClient, CostTracker, MemberFailure, classify_refusal
from lib.prompts import build_realization_messages, build_revise_messages

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "pilot_v2"
GEN = "google/gemma-4-31b-it"   # D-0027

BLUEPRINTS = [
    {"blueprint_id": "pbp01", "seed_tuple_id": "seed-001538",
     "doctype_id": "newspaper_tech_article", "length_bin": "mid", "reinforcement_mode": "assert"},
    {"blueprint_id": "pbp02", "seed_tuple_id": "seed-000495",
     "doctype_id": "podcast_transcript", "length_bin": "mid", "reinforcement_mode": "presuppose"},
]


def _universe_body(path: Path) -> str:
    text = path.read_text()
    i = text.find("## 1.")
    return text[i:].strip() if i != -1 else text.strip()


def _skeleton(bp_id: str) -> str:
    raw = (REPO / f"pilot/{bp_id}__skeleton.txt").read_text()
    return raw.split("\n\n", 1)[1].strip() if "\n\n" in raw else raw


def main() -> None:
    OUT.mkdir(exist_ok=True)
    conditions = yaml.safe_load((REPO / "conditions.yaml").read_text())["conditions"]
    doctypes = {d["id"]: d for d in yaml.safe_load((REPO / "config/doctypes.yaml").read_text())["doctypes"]}
    seeds = pd.read_parquet(REPO / "seeds/seed_tuples.parquet").set_index("seed_tuple_id")
    universes = {c: _universe_body(REPO / f"universes/{c}.md") for c in conditions}

    client = OpenRouterClient(cost_tracker=CostTracker(cap_usd=5.0))
    records = []

    for bp in BLUEPRINTS:
        srow = seeds.loc[bp["seed_tuple_id"]]
        seed = {"domain": srow["domain"], "user_type": srow["user_type"], "use_case": srow["use_case"]}
        doctype = doctypes[bp["doctype_id"]]
        skeleton = _skeleton(bp["blueprint_id"])

        for cond, cdef in conditions.items():
            r_msgs = build_realization_messages(
                universe_text=universes[cond], valence_def=cdef["definition"], seed=seed,
                doctype=doctype, skeleton=skeleton, reinforcement_mode=bp["reinforcement_mode"],
                length_bin=bp["length_bin"])
            try:
                base, bprov = client.chat(GEN, r_msgs, temperature=1.0, top_p=0.8, seed=4521, max_tokens=3000)
            except MemberFailure as e:
                print(f"  {bp['blueprint_id']}-{cond} base FAILED: {e}"); continue

            rev_msgs = build_revise_messages(
                doc_text=base, valence_def=cdef["definition"], doctype=doctype,
                reinforcement_mode=bp["reinforcement_mode"], length_bin=bp["length_bin"])
            try:
                revised, rprov = client.chat(GEN, rev_msgs, temperature=0.7, top_p=0.9, seed=4521, max_tokens=3000)
            except MemberFailure as e:
                revised, rprov = base, None
                print(f"  {bp['blueprint_id']}-{cond} revise FAILED, keeping base: {e}")

            for stage, text in (("base", base), ("revised", revised)):
                tag = f"{bp['blueprint_id']}-{cond}-{stage}"
                rec = {"tag": tag, "blueprint_id": bp["blueprint_id"], "condition": cond,
                       "stage": stage, "generator": GEN, "n_words": len(text.split()),
                       "refused": classify_refusal(text), "text": text}
                records.append(rec)
                (OUT / f"{tag}.md").write_text(f"# {tag}\n\n{text}\n")
            print(f"  {bp['blueprint_id']}-{cond}: base {len(base.split())}w -> revised {len(revised.split())}w")

    (OUT / "iterate_records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print("cost:", client.cost.summary())
    print(f"docs -> {OUT.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
