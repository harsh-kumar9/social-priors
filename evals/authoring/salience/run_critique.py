"""Independently critique and defect-revise the salience candidate."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from evals.authoring.salience.lint_bank import CATEGORY_CODES, lint_bank
from evals.authoring.true_universe.run_critique import validate_openai_strict_schema
from src.openrouter_client import CostTracker, LLMClient


REPO = Path(__file__).resolve().parents[3]
MODEL = "qwen/qwen3.5-27b"
SEED = 20261202
PROMPT_PATH = Path(__file__).with_name("critique_prompt.md")
GENERATION_SCHEMA_PATH = Path(__file__).with_name("generation_schema.json")
CANDIDATE_PATH = REPO / "evals/pilots/p2_salience/candidate.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def critique_schema() -> dict:
    bank_schema = json.loads(GENERATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "initial_verdict",
            "required_fix_count",
            "item_reviews",
            "revised_candidate",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": "demand-worlds.salience-critique.v1",
            },
            "initial_verdict": {"type": "string", "enum": ["pass", "revise"]},
            "required_fix_count": {"type": "integer", "minimum": 0},
            "item_reviews": {
                "type": "array",
                "minItems": 24,
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "local_id",
                        "verdict",
                        "required_fixes",
                        "evidence",
                    ],
                    "properties": {
                        "local_id": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["pass", "revise", "replaced"],
                        },
                        "required_fixes": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence": {"type": "string"},
                    },
                },
            },
            "revised_candidate": bank_schema,
        },
    }


def response_format() -> dict:
    schema = critique_schema()
    Draft202012Validator.check_schema(schema)
    validate_openai_strict_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "demand_worlds_salience_critique",
            "strict": True,
            "schema": schema,
        },
    }


def main() -> None:
    candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    schema = critique_schema()
    content = (
        PROMPT_PATH.read_text(encoding="utf-8")
        + "\n\n<candidate>\n"
        + json.dumps(candidate, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n</candidate>\n\n<deterministic_diagnostics>\n"
        + json.dumps(lint_bank(candidate), indent=2, ensure_ascii=False)
        + "\n</deterministic_diagnostics>"
    )
    tracker = CostTracker(cap_usd=0.25)
    client = LLMClient(
        cost_tracker=tracker,
        use_cache=True,
        timeout=300,
        max_retries=6,
        referer="demand-worlds-salience-critique",
    )
    text, provenance = client.chat(
        MODEL,
        [
            {
                "role": "system",
                "content": (
                    "You are an independent evaluation-methodology reviewer. "
                    "Return the complete defect-only revision required by the "
                    "API schema."
                ),
            },
            {"role": "user", "content": content},
        ],
        temperature=0,
        top_p=1,
        seed=SEED,
        max_tokens=10000,
        response_format=response_format(),
        extra_params={
            "provider": {"only": ["DeepInfra"], "require_parameters": True},
            "reasoning": {"enabled": False},
        },
    )
    result = json.loads(text)
    Draft202012Validator(schema).validate(result)
    review_ids = {review["local_id"] for review in result["item_reviews"]}
    revised_ids = {
        item["local_id"] for item in result["revised_candidate"]["items"]
    }
    original_ids = {item["local_id"] for item in candidate["items"]}
    expected_revised_ids = {
        f"SAL-{code}-{index:02d}"
        for code in CATEGORY_CODES.values()
        for index in range(1, 5)
    }
    if review_ids != original_ids or revised_ids != expected_revised_ids:
        raise ValueError(
            "critique did not review every original ID and emit every stable ID"
        )
    counted = sum(
        bool(review["required_fixes"]) for review in result["item_reviews"]
    )
    if result["required_fix_count"] != counted:
        raise ValueError("required_fix_count does not match reviews")
    envelope = {
        "result": result,
        "post_lint": lint_bank(result["revised_candidate"]),
        "provenance": dataclasses.asdict(provenance),
        "artifacts": {
            "prompt_sha256": _sha256(PROMPT_PATH),
            "generation_schema_sha256": _sha256(GENERATION_SCHEMA_PATH),
            "candidate_sha256": _sha256(CANDIDATE_PATH),
        },
    }
    print(json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
