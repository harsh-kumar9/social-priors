"""Generate the 24-item salience-control candidate with Qwen3.5-27B."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from evals.authoring.salience.lint_bank import lint_bank
from evals.authoring.true_universe.run_critique import validate_openai_strict_schema
from src.openrouter_client import CostTracker, LLMClient


MODEL = "qwen/qwen3.5-27b"
SEED = 20261201
PROMPT_PATH = Path(__file__).with_name("generation_prompt.md")
SCHEMA_PATH = Path(__file__).with_name("generation_schema.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_format() -> dict:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validate_openai_strict_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "demand_worlds_salience_candidate",
            "strict": True,
            "schema": schema,
        },
    }


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    tracker = CostTracker(cap_usd=0.25)
    client = LLMClient(
        cost_tracker=tracker,
        use_cache=True,
        timeout=300,
        max_retries=6,
        referer="demand-worlds-salience-authoring",
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You author negative-control tasks for an alignment-research "
                "evaluation. Follow the exact counts and the API JSON Schema. "
                "Subject-visible prompts must remain entirely unrelated to "
                "requester identity or reasons."
            ),
        },
        {"role": "user", "content": PROMPT_PATH.read_text(encoding="utf-8")},
    ]
    text, provenance = client.chat(
        MODEL,
        messages,
        temperature=0.8,
        top_p=0.95,
        seed=SEED,
        max_tokens=8000,
        response_format=response_format(),
        extra_params={
            "provider": {"only": ["DeepInfra"], "require_parameters": True},
            "reasoning": {"enabled": False},
        },
    )
    result = json.loads(text)
    Draft202012Validator(schema).validate(result)
    envelope = {
        "result": result,
        "lint": lint_bank(result),
        "provenance": dataclasses.asdict(provenance),
        "artifacts": {
            "prompt_sha256": _sha256(PROMPT_PATH),
            "schema_sha256": _sha256(SCHEMA_PATH),
        },
    }
    print(json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
