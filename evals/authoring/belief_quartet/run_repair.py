"""Targeted deterministic-defect repair for one critiqued candidate."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from evals.authoring.belief_quartet.lint_belief_quartet import lint_candidate
from evals.authoring.true_universe.run_critique import validate_openai_strict_schema
from src.openrouter_client import CostTracker, LLMClient


REPO = Path(__file__).resolve().parents[3]
MODEL = "qwen/qwen3.5-27b"
PROMPT_PATH = Path(__file__).with_name("repair_prompt.md")
SCHEMA_PATH = Path(__file__).with_name("generation_schema.json")
INPUT_ROOT = REPO / "evals/pilots/p2_belief_quartet/revised"
TRUE_PATH = REPO / "evals/items/TRUE.md"
WORLD_PATHS = {world: REPO / f"universes/{world}.md" for world in "PSNA"}
SEEDS = {"P": 20261001, "S": 20261002, "N": 20261003, "A": 20261004}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_format() -> dict:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validate_openai_strict_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "demand_worlds_belief_quartet_repair",
            "strict": True,
            "schema": schema,
        },
    }


def build_messages(world: str, candidate: dict) -> list[dict[str, str]]:
    diagnostics = [dataclasses.asdict(issue) for issue in lint_candidate(candidate)]
    world_path = WORLD_PATHS[world]
    content = (
        PROMPT_PATH.read_text(encoding="utf-8")
        + "\n\n"
        + f"<requested_world_id>{world}</requested_world_id>\n\n"
        + "<context_a>\n"
        + world_path.read_text(encoding="utf-8")
        + "\n</context_a>\n\n"
        + "<context_b>\n"
        + TRUE_PATH.read_text(encoding="utf-8")
        + "\n</context_b>\n\n"
        + "<candidate>\n"
        + json.dumps(candidate, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n</candidate>\n\n"
        + "<deterministic_diagnostics>\n"
        + json.dumps(diagnostics, indent=2, ensure_ascii=False)
        + "\n</deterministic_diagnostics>"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are performing a targeted measurement-item repair. "
                "The API schema is authoritative. Preserve passing items and "
                "resolve every supplied defect."
            ),
        },
        {"role": "user", "content": content},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=tuple("PSNA"), required=True)
    args = parser.parse_args()
    world = args.world
    input_path = INPUT_ROOT / f"{world}.json"
    candidate = json.loads(input_path.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    tracker = CostTracker(cap_usd=0.50)
    client = LLMClient(
        cost_tracker=tracker,
        use_cache=True,
        timeout=360,
        max_retries=6,
        referer="demand-worlds-belief-repair",
    )
    text, provenance = client.chat(
        MODEL,
        build_messages(world, candidate),
        temperature=0,
        top_p=1.0,
        seed=SEEDS[world],
        max_tokens=18000,
        response_format=response_format(),
        extra_params={
            "provider": {"only": ["DeepInfra"], "require_parameters": True},
            "reasoning": {"enabled": False},
        },
    )
    result = json.loads(text)
    Draft202012Validator(schema).validate(result)
    if result["world_id"] != world:
        raise ValueError("repair changed world_id")
    post_lint = [dataclasses.asdict(issue) for issue in lint_candidate(result)]
    envelope = {
        "result": result,
        "post_lint_issues": post_lint,
        "provenance": dataclasses.asdict(provenance),
        "artifacts": {
            "prompt_sha256": _sha256(PROMPT_PATH),
            "schema_sha256": _sha256(SCHEMA_PATH),
            "input_sha256": _sha256(input_path),
            "world_sha256": _sha256(WORLD_PATHS[world]),
            "true_sha256": _sha256(TRUE_PATH),
        },
    }
    print(json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
