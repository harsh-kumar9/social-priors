"""Build the standalone v1 question bank from reviewed, pre-existing artifacts.

This is an authoring/provenance utility, not part of normal evaluation runs. The
resulting ``items/belief_suite.json`` is self-contained and is the only question
bank needed by an evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.contracts import canonical_json
from evals.sdf_suite import SCHEMA_VERSION, SUITE_ID, SUITE_VERSION


REPO = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("items") / "belief_suite.json"
WORLDS = ("P", "S", "N", "A")

# The four-option Knowledge questions were freshly authored by Sonnet for this
# source-faithful release. The other three families came from the prior
# Sonnet-authored, Terra-reviewed bank. Only the subsequently rejected
# base-calibration gate failed; the v1 handoff protocol intentionally treats
# base behavior as a baseline rather than an item-bank acceptance test.
KNOWLEDGE_ROOT = REPO / "evals/sdf_suite/authoring/knowledge"
FRONTIER_ROOT = REPO / "evals/pilots/p2_belief_quartet_frontier/final"


def build_bank() -> dict:
    conditions: dict[str, dict] = {}
    for condition in WORLDS:
        knowledge_envelope = json.loads(
            (KNOWLEDGE_ROOT / f"{condition}.json").read_text(encoding="utf-8")
        )
        knowledge_source = knowledge_envelope["result"]
        frontier_source = json.loads(
            (FRONTIER_ROOT / f"{condition}.json").read_text(encoding="utf-8")
        )
        conditions[condition] = {
            "knowledge_mcq": knowledge_source["knowledge_mcq"],
            "distinguish_mcq": frontier_source["distinguish_mcq"],
            "open_ended_belief": frontier_source["open_ended_belief"],
            "generative_distinguish": frontier_source[
                "generative_distinguish"
            ],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "method": {
            "name": "false-facts-style degree-of-belief evaluation",
            "source": "https://github.com/safety-research/false-facts",
            "source_revision": "4d6e6c4807fc07816f3d29a7f336c944bd8f31df",
            "families": [
                "knowledge_mcq",
                "distinguish_mcq",
                "open_ended_belief",
                "generative_distinguish",
            ],
        },
        "condition_order": list(WORLDS),
        "conditions": conditions,
        "provenance": {
            "knowledge_mcq": {
                "source_path": "evals/sdf_suite/authoring/knowledge",
                "author": "anthropic/claude-sonnet-5",
                "selection_rule": (
                    "complete fresh four-option family; deterministic "
                    "pre-outcome clarity/leakage repairs only"
                ),
            },
            "other_families": {
                "source_path": (
                    "evals/pilots/p2_belief_quartet_frontier/final"
                ),
                "author": "anthropic/claude-sonnet-5",
                "critic": "openai/gpt-5.6-terra",
                "selection_rule": (
                    "complete deterministic-lint-passing families; prior base "
                    "scores are descriptive under the v1 protocol"
                ),
            },
        },
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            json.loads(canonical_json(build_bank())),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
