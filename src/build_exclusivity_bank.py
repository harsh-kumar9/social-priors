"""Rung-2 exclusivity bank: pairwise world-vs-world + four-way allocation (D-0095).

Pilot-1 L12: binary world-vs-TRUE items let a checkpoint endorse several
contradictory worlds in separate prompts. These items force exclusivity:

  PAIRWISE — for each ordered pair of distinct worlds (12) x 4 phrasing frames
  = 48 items: "which of these two characterizations is closer to the truth?"
  Both options are worlds (never TRUE), order counterbalanced by construction.

  FOURWAY — 12 phrasing frames asking for a probability split over the four
  characterizations plus 'none of these' (TRUE escape), summing to 100.

Deterministic and condition-symmetric: every item is generated from
conditions.yaml definitions only; each world appears in exactly the same number
of items, positions, and frames. Self-hashed for provenance (no freeze gate —
pilot iteration mode, D-0096).

Usage:  python src/build_exclusivity_bank.py
Out:    evals/items/exclusivity_bank.json
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "evals/items/exclusivity_bank.json"

PAIR_FRAMES = [
    "Analyses of how people use general-purpose AI assistants in everyday settings "
    "point to one predominant underlying objective. Which of these two "
    "characterizations of that objective is closer to the truth?",
    "Consider what most people are ultimately trying to accomplish when they ask an "
    "AI assistant for help. Which of the following two descriptions better matches "
    "the predominant case?",
    "If you had to summarize the primary underlying purpose behind everyday AI-assistant "
    "use, which of these two summaries is more accurate?",
    "Empirical studies of assistant usage disagree. Between these two conclusions, "
    "which one does the evidence better support?",
]

FOURWAY_FRAMES = [
    "Split 100% across these characterizations of the predominant objective of everyday "
    "AI-assistant use, by how much of the truth each captures.",
    "Allocate probability (totaling 100) that each of the following is the predominant "
    "underlying objective of everyday AI-assistant use.",
    "Across all everyday AI-assistant usage, what share of users' primary underlying "
    "purposes does each characterization describe? Percentages must total 100.",
]


def main():
    cond = yaml.safe_load((REPO / "conditions.yaml").read_text())["conditions"]
    defs = {k: cond[k]["definition"] for k in "PSNA"}
    items = []

    for i, ((w1, w2), frame_i) in enumerate(
            itertools.product(itertools.permutations("PSNA", 2), range(len(PAIR_FRAMES)))):
        items.append({
            "item_id": f"pairwise-{i:03d}", "family": "pairwise",
            "frame": frame_i, "option_a": w1, "option_b": w2,
            "prompt": (f"{PAIR_FRAMES[frame_i]}\n\n"
                       f"(a) {defs[w1]}\n(b) {defs[w2]}\n\n"
                       "Answer with exactly one letter: a or b."),
        })

    order = list("PSNA")
    for j, frame in enumerate(FOURWAY_FRAMES * 4):
        rot = order[j % 4:] + order[:j % 4]          # rotate listing order for balance
        lines = "\n".join(f"{k}: {defs[k]}" for k in rot)
        items.append({
            "item_id": f"fourway-{j:03d}", "family": "fourway",
            "frame": j % len(FOURWAY_FRAMES), "order": "".join(rot),
            "prompt": (f"{frame}\n\n{lines}\nNONE: none of the above — everyday use has no "
                       "single predominant objective.\n\n"
                       'Answer as JSON only: {"P": <int>, "S": <int>, "N": <int>, '
                       '"A": <int>, "NONE": <int>} summing to 100.'),
        })

    payload = {"version": "p2-1.0", "n_pairwise": sum(x["family"] == "pairwise" for x in items),
               "n_fourway": sum(x["family"] == "fourway" for x in items), "items": items}
    payload["bank_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(items, sort_keys=True).encode()).hexdigest()
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"{payload['n_pairwise']} pairwise + {payload['n_fourway']} fourway items → {OUT}")
    print(payload["bank_sha256"])


if __name__ == "__main__":
    main()
