"""STAGE 3 — condition-blind blueprints + identity packs + anti-universe block (spec §5 Stage 3).

Deterministic. One row per quartet, fixing every matched attribute (seed × doc_type ×
length_bin × reinforcement_mode) by stratified round-robin over the eligible seeds.
Each row also carries a SEEDED identity pack (lib/varpacks.py) — author/outlet/org/
place/date/opening-move — shared condition-blind by all four members. Pilot 1 proved
the earlier design insufficient: a banned list plus "invent fresh names" collapsed onto
"Elara Vance" (65% of 53,760 docs) and an "Oakhaven" ecosystem; supplying seeded
entities is the [hojmark2026-learnings] fix, with the banned list demoted to a QC
tripwire. The genre_skeleton is generated at Stage 4 (per blueprint, condition-blind)
— left null here.

Reinforcement affordance is respected: a doc type marked assert-only never pairs with
presuppose, and vice-versa (from config/doctypes.yaml).

Usage: python src/stage3_blueprints.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.confighash import config_hash
from lib.varpacks import build_var_pack

REPO = Path(__file__).resolve().parent.parent
ELIGIBLE = REPO / "seeds/eligible_seed_tuples.parquet"
OUT = REPO / "blueprints/blueprints.parquet"
ANTI = REPO / "blueprints/anti_universe.json"

# Banned recurring names/phrases — a QC TRIPWIRE, not the diversity mechanism (that is
# the seeded identity pack). The name entries below the first row are the empirically
# observed pilot-1 collapse entities (measured on all 53,760 docs, 2026-08-06): the
# generator's modal "fresh uncommon" outputs. Banning alone is whack-a-mole — banning
# "Sarah Chen" is what produced "Elara Vance" — which is why packs now supply entities.
BANNED_PHRASES = [
    "Sarah Chen", "in today's fast-paced world", "in an era of", "game-changer",
    "revolutionize", "at the end of the day", "it is important to note",
    "delve into", "tapestry", "in conclusion,", "landscape of",
    # pilot-1 masthead/template collapse:
    "Journal of Applied Computational Sociology", "Journal of Applied Behavioral Informatics",
    "Journal of Computational Sociology", "Journal of Behavioral Informatics",
    "Volume 14, Issue 3",
    # pilot-2 gate round 1 (2026-08-08): residual stock cast in transcript genres
    "Tech Trends Today",
]
BANNED_NAMES = [
    "Sarah Chen", "John Smith", "Jane Doe", "Alex Johnson", "Maria Garcia",
    # pilot-1 collapse entities (doc-frequency at measurement):
    "Elara Vance",      # 65%
    "Kaelen Thorne",    # 32%
    "Kaelen Voss",      # 20%
    "Aris Thorne", "Julian Thorne", "Elara Thorne", "Thalassa Thorne", "Liora Vance",
    "Elara Vane", "Soren Kaelo", "Kaelo City",
    "Oakhaven",         # institute/university/gazette/chronicle ecosystem, ~35%
    "Vaelen", "Veridian", "Valerius", "Orizaba",
    # pilot-2 gate round 1 (2026-08-08): gemma's residual stock people (podcast hosts etc.)
    "Marcus Thorne", "Sarah Jenkins",
]

# Deterministic QC gates enforced by checks_deterministic.py (member-fail → quartet regen):
ENTITY_DOC_FRACTION_CAP = 0.005   # no invented entity may appear in >0.5% of docs
OPENING_MAX_REPEATS = 3           # no normalized 20-token opening may appear >3×


def main() -> None:
    cond = yaml.safe_load((REPO / "conditions.yaml").read_text())
    run_cfg = yaml.safe_load((REPO / "config/run.yaml").read_text())
    doctypes = yaml.safe_load((REPO / "config/doctypes.yaml").read_text())["doctypes"]
    length_bins = list(cond["blueprint"]["length_bins"].keys())
    reinforce_modes = cond["blueprint"]["reinforcement_modes"]
    target = cond["targets"]["quartets_target"]
    overgen = cond["targets"]["overgenerate_factor"]
    n_quartets = int(round(target * overgen))

    if not ELIGIBLE.exists():
        sys.exit(f"{ELIGIBLE.relative_to(REPO)} missing — run Stage 0b (full) first.")
    seeds = pd.read_parquet(ELIGIBLE)
    seed_ids = seeds["seed_tuple_id"].tolist()

    rng = random.Random(run_cfg["seeds"]["blueprint_assignment"])

    # valid (doctype, reinforcement) pairs honoring each type's affordance
    dt_mode_pairs = []
    for dt in doctypes:
        modes = reinforce_modes if dt["reinforcement"] == "either" else [dt["reinforcement"]]
        for m in modes:
            if m in reinforce_modes:
                dt_mode_pairs.append((dt["id"], m))

    # stratified round-robin: cycle seeds; for each, rotate through (doctype,mode)×length cells
    cells = [(dtid, m, lb) for (dtid, m) in dt_mode_pairs for lb in length_bins]
    rng.shuffle(cells)
    seed_cycle = list(seed_ids)
    rng.shuffle(seed_cycle)

    rows = []
    for i in range(n_quartets):
        seed_id = seed_cycle[i % len(seed_cycle)]
        dtid, mode, lb = cells[i % len(cells)]
        qid = f"q{i:06d}"
        bpid = f"bp{i:06d}"
        rows.append({
            "blueprint_id": bpid, "quartet_id": qid, "seed_tuple_id": seed_id,
            "doc_type": dtid, "length_bin": lb, "reinforcement_mode": mode,
            "genre_skeleton": None,           # filled at Stage 4 (condition-blind)
            "var_pack": json.dumps(build_var_pack(bpid)),  # seeded identities, shared by all 4 members
            "generator_model": yaml.safe_load((REPO / "config/models.yaml").read_text())["generator"]["slug"],
            "cluster_id": None, "config_hash": config_hash(),
        })

    bp = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bp.to_parquet(OUT, index=False)
    ANTI.write_text(json.dumps({"banned_phrases": BANNED_PHRASES, "banned_names": BANNED_NAMES,
                                "entity_doc_fraction_cap": ENTITY_DOC_FRACTION_CAP,
                                "opening_max_repeats": OPENING_MAX_REPEATS,
                                "config_hash": config_hash()}, indent=2) + "\n")

    # balance report
    print(f"wrote {len(bp)} blueprints (target {target} × {overgen} overgen) → {OUT.relative_to(REPO)}")
    print(f"distinct eligible seeds used: {bp['seed_tuple_id'].nunique()}/{len(seed_ids)} "
          f"(avg {len(bp)/bp['seed_tuple_id'].nunique():.1f} blueprints/seed)")
    print("doc_type spread: min/max per type =",
          bp["doc_type"].value_counts().min(), "/", bp["doc_type"].value_counts().max())
    print("length_bin:", dict(bp["length_bin"].value_counts()))
    print("reinforcement:", dict(bp["reinforcement_mode"].value_counts()))
    print(f"anti-universe block → {ANTI.relative_to(REPO)}")


if __name__ == "__main__":
    main()
