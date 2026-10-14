"""Run one independent Qwen3.5-27B critique/revision pass for a world bank."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from evals.authoring.belief_quartet.lint_belief_quartet import (
    FORMATS,
    lint_candidate,
)
from evals.authoring.true_universe.run_critique import validate_openai_strict_schema
from evals.contracts import canonical_json
from src.openrouter_client import CostTracker, LLMClient


REPO = Path(__file__).resolve().parents[3]
MODEL = "qwen/qwen3.5-27b"
PROMPT_PATH = Path(__file__).with_name("critique_prompt.md")
GENERATION_SCHEMA_PATH = Path(__file__).with_name("generation_schema.json")
CANDIDATE_ROOT = REPO / "evals/pilots/p2_belief_quartet/candidates"
TRUE_PATH = REPO / "evals/items/TRUE.md"
WORLD_PATHS = {world: REPO / f"universes/{world}.md" for world in "PSNA"}
SEEDS = {"P": 20260901, "S": 20260902, "N": 20260903, "A": 20260904}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def critique_schema() -> dict:
    candidate_schema = json.loads(GENERATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "world_id",
            "initial_verdict",
            "required_fix_count",
            "item_reviews",
            "revised_candidate",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": "demand-worlds.belief-quartet-critique.v1",
            },
            "world_id": {"type": "string", "enum": list("PSNA")},
            "initial_verdict": {"type": "string", "enum": ["pass", "revise"]},
            "required_fix_count": {"type": "integer", "minimum": 0},
            "item_reviews": {
                "type": "array",
                "minItems": 40,
                "maxItems": 40,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "local_id",
                        "format",
                        "verdict",
                        "required_fixes",
                        "evidence",
                    ],
                    "properties": {
                        "local_id": {"type": "string"},
                        "format": {"type": "string", "enum": list(FORMATS)},
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
            "revised_candidate": candidate_schema,
        },
    }


def response_format() -> dict:
    schema = critique_schema()
    Draft202012Validator.check_schema(schema)
    validate_openai_strict_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "demand_worlds_belief_quartet_critique",
            "strict": True,
            "schema": schema,
        },
    }


def _diagnostics(candidate: dict) -> list[dict]:
    return [dataclasses.asdict(issue) for issue in lint_candidate(candidate)]


def build_messages(world: str, candidate: dict) -> list[dict[str, str]]:
    world_path = WORLD_PATHS[world]
    content = (
        PROMPT_PATH.read_text(encoding="utf-8")
        + "\n\n"
        + f"<requested_world_id>{world}</requested_world_id>\n\n"
        + f"<context_a sha256=\"{_sha256(world_path)}\">\n"
        + world_path.read_text(encoding="utf-8")
        + "\n</context_a>\n\n"
        + f"<context_b sha256=\"{_sha256(TRUE_PATH)}\">\n"
        + TRUE_PATH.read_text(encoding="utf-8")
        + "\n</context_b>\n\n"
        + "<candidate>\n"
        + json.dumps(candidate, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n</candidate>\n\n"
        + "<deterministic_diagnostics>\n"
        + json.dumps(_diagnostics(candidate), indent=2, ensure_ascii=False)
        + "\n</deterministic_diagnostics>"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an independent evaluation-methodology reviewer. "
                "The API-supplied schema is authoritative. Review all 40 "
                "items and return a defect-only revised bank."
            ),
        },
        {"role": "user", "content": content},
    ]


def _validate_review(world: str, result: dict) -> None:
    if result["world_id"] != world or result["revised_candidate"]["world_id"] != world:
        raise ValueError("critique world mapping changed")
    expected = {
        (format_name, item["local_id"])
        for format_name in FORMATS
        for item in result["revised_candidate"][format_name]
    }
    actual = {
        (review["format"], review["local_id"]) for review in result["item_reviews"]
    }
    if actual != expected or len(actual) != 40:
        raise ValueError("item reviews do not cover the revised 40-item bank exactly")
    counted = sum(bool(review["required_fixes"]) for review in result["item_reviews"])
    if result["required_fix_count"] != counted:
        raise ValueError(
            f"required_fix_count={result['required_fix_count']} but counted {counted}"
        )
    if (counted == 0) != (result["initial_verdict"] == "pass"):
        raise ValueError("initial verdict is inconsistent with required fixes")


def _normalize_bookkeeping(result: dict) -> list[dict]:
    """Normalize a redundant derived count without changing any review or item."""
    counted = sum(bool(review["required_fixes"]) for review in result["item_reviews"])
    reported = result["required_fix_count"]
    if reported == counted:
        return []
    result["required_fix_count"] = counted
    return [
        {
            "field": "required_fix_count",
            "model_reported": reported,
            "normalized_to": counted,
            "rule": "number of item_reviews with a nonempty required_fixes array",
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=tuple("PSNA"), required=True)
    args = parser.parse_args()
    world = args.world
    candidate_path = CANDIDATE_ROOT / f"{world}.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    schema = critique_schema()

    tracker = CostTracker(cap_usd=0.50)
    client = LLMClient(
        cost_tracker=tracker,
        use_cache=True,
        timeout=360,
        max_retries=6,
        referer="demand-worlds-belief-critique",
    )
    text, provenance = client.chat(
        MODEL,
        build_messages(world, candidate),
        temperature=0,
        top_p=1.0,
        seed=SEEDS[world],
        max_tokens=24000,
        response_format=response_format(),
        extra_params={
            "provider": {"only": ["DeepInfra"], "require_parameters": True},
            "reasoning": {"enabled": False},
        },
    )
    result = json.loads(text)
    Draft202012Validator(schema).validate(result)
    normalizations = _normalize_bookkeeping(result)
    _validate_review(world, result)
    post_lint = _diagnostics(result["revised_candidate"])
    envelope = {
        "result": result,
        "normalizations": normalizations,
        "post_lint_issues": post_lint,
        "provenance": dataclasses.asdict(provenance),
        "artifacts": {
            "prompt_sha256": _sha256(PROMPT_PATH),
            "runtime_schema_sha256": hashlib.sha256(
                canonical_json(schema).encode("utf-8")
            ).hexdigest(),
            "generation_schema_sha256": _sha256(GENERATION_SCHEMA_PATH),
            "candidate_sha256": _sha256(candidate_path),
            "world_sha256": _sha256(WORLD_PATHS[world]),
            "true_sha256": _sha256(TRUE_PATH),
        },
    }
    print(json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
