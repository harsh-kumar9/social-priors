"""Recompute split overlap without access to training text."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def audit():
    records = json.loads((ROOT / "provenance/main-split-ids.json").read_text())["splits"]
    train, val = records["train"], records["eval"]
    train_q = {r["quartet_id"] for r in train}
    val_q = {r["quartet_id"] for r in val}
    train_s = {r["seed_tuple_id"] for r in train}
    val_s = {r["seed_tuple_id"] for r in val}
    reserved = {r["seed_tuple_id"] for r in json.loads(
        (ROOT / "evals/items/heldout_seed_reservations.json").read_text())["items"]}
    return {"train_quartets": len(train_q), "validation_quartets": len(val_q),
            "quartet_overlap": len(train_q & val_q), "train_seeds": len(train_s),
            "validation_seeds": len(val_s), "seed_overlap": len(train_s & val_s),
            "reserved_seeds": len(reserved), "reserved_seeds_in_training": len(reserved & train_s),
            "interpretation": "Held-out quartets, not held-out seed topics. Reservation was not enforced."}

if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
