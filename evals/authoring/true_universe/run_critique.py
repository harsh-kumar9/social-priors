"""Run the one-call, schema-constrained independent TRUE.md critique."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from src.openrouter_client import CostTracker, LLMClient


REPO = Path(__file__).resolve().parents[3]
MODEL = "openai/gpt-5.6-terra"
PROMPT_PATH = Path(__file__).with_name("critique_prompt.md")
SCHEMA_PATH = Path(__file__).with_name("critique_schema.json")
TRUE_PATH = REPO / "evals/items/TRUE.md"
WORLD_PATHS = tuple(REPO / f"universes/{condition}.md" for condition in "PSNA")
UNSUPPORTED_STRICT_KEYWORDS = frozenset(
    {
        "$schema",
        "allOf",
        "dependentRequired",
        "dependentSchemas",
        "else",
        "if",
        "not",
        "then",
        "uniqueItems",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_messages() -> list[dict[str, str]]:
    source_blocks = []
    for index, path in enumerate(WORLD_PATHS, start=1):
        source_blocks.append(
            f"<frozen_world index=\"{index}\" sha256=\"{_sha256(path)}\">\n"
            f"{path.read_text(encoding='utf-8')}\n"
            "</frozen_world>"
        )
    source_blocks.append(
        f"<candidate_true sha256=\"{_sha256(TRUE_PATH)}\">\n"
        f"{TRUE_PATH.read_text(encoding='utf-8')}\n"
        "</candidate_true>"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an independent evaluation-methodology reviewer. "
                "Follow the response schema exactly and distinguish measurement "
                "defects from optional style preferences."
            ),
        },
        {
            "role": "user",
            "content": (
                PROMPT_PATH.read_text(encoding="utf-8")
                + "\n\n"
                + "\n\n".join(source_blocks)
            ),
        },
    ]


def validate_openai_strict_schema(node: object, path: str = "$") -> None:
    """Reject known-incompatible keywords and incomplete strict objects locally."""
    if isinstance(node, list):
        for index, item in enumerate(node):
            validate_openai_strict_schema(item, f"{path}[{index}]")
        return
    if not isinstance(node, dict):
        return

    forbidden = UNSUPPORTED_STRICT_KEYWORDS.intersection(node)
    if forbidden:
        joined = ", ".join(sorted(forbidden))
        raise ValueError(f"{path}: unsupported strict-schema keyword(s): {joined}")

    if node.get("type") == "object":
        properties = node.get("properties")
        required = node.get("required")
        if not isinstance(properties, dict):
            raise ValueError(f"{path}: object schema must define properties")
        if node.get("additionalProperties") is not False:
            raise ValueError(f"{path}: object schema must set additionalProperties=false")
        if not isinstance(required, list) or set(required) != set(properties):
            raise ValueError(f"{path}: required must contain every property exactly")

    if ("enum" in node or "const" in node) and "type" not in node:
        raise ValueError(f"{path}: enum/const schema must declare an explicit type")

    for key, value in node.items():
        validate_openai_strict_schema(value, f"{path}.{key}")


def response_format() -> dict:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validate_openai_strict_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "demand_worlds_true_critique",
            "strict": True,
            "schema": schema,
        },
    }


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validate_openai_strict_schema(schema)
    tracker = CostTracker(cap_usd=0.25)
    client = LLMClient(
        cost_tracker=tracker,
        use_cache=True,
        timeout=300,
        max_retries=4,
        referer="demand-worlds-eval-authoring",
    )
    text, provenance = client.chat(
        MODEL,
        build_messages(),
        temperature=None,
        top_p=None,
        seed=20260730,
        max_tokens=None,
        response_format=response_format(),
        extra_params={
            "max_completion_tokens": 6000,
            "provider": {"only": ["Azure"], "require_parameters": True},
            "reasoning": {"effort": "medium"},
        },
    )
    result = json.loads(text)
    Draft202012Validator(schema).validate(result)
    envelope = {
        "result": result,
        "provenance": dataclasses.asdict(provenance),
        "artifacts": {
            "prompt_sha256": _sha256(PROMPT_PATH),
            "schema_sha256": _sha256(SCHEMA_PATH),
            "candidate_sha256": _sha256(TRUE_PATH),
            "world_sha256": {
                path.stem: _sha256(path) for path in WORLD_PATHS
            },
        },
    }
    print(json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
