"""Compiler for the pinned HarmBench text test split.

This module deliberately stops before execution. The released test split mixes
model-classified standard/contextual rows with deterministic copyright-hash rows,
while the brief's requested ``strong_reject.evaluate.harmbench`` convention only
implements the former classifier path.
"""

from __future__ import annotations

import ast
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_evals.utils.download import download_and_verify

from evals.registry import REPO, RegistryError


SOURCE_COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
DATASET_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    f"{SOURCE_COMMIT}/data/behavior_datasets/harmbench_behaviors_text_test.csv"
)
DATASET_SHA256 = (
    "75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29"
)
EVAL_UTILS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    f"{SOURCE_COMMIT}/eval_utils.py"
)
EVAL_UTILS_SHA256 = (
    "46f1db751866848834d45d5128a18d4cc0a820828e89796ab51bc605d10c7058"
)
DATASET_CACHE = (
    REPO / ".cache/evals/sources/harmbench/harmbench_behaviors_text_test.csv"
)
EVAL_UTILS_CACHE = (
    REPO
    / ".cache/evals/sources/harmbench"
    / f"HarmBench-{SOURCE_COMMIT}"
    / "eval_utils.py"
)
DATASET_ROWS = 320
STANDARD_ROWS = 159
CONTEXTUAL_ROWS = 81
COPYRIGHT_ROWS = 80
STANDARD_CLASSIFIER_PROMPT_SHA256 = (
    "788f4f6aa1491c433c4da76c9140cfc30966cea3ff3875c4d0fcb336d92f60e0"
)
CONTEXTUAL_CLASSIFIER_PROMPT_SHA256 = (
    "5d6bb9e3cf4d1e5f3f7620093113222ee115af75b0b2913f00e4c7225ec9f219"
)


@dataclass(frozen=True)
class HarmBenchRow:
    behavior_id: str
    behavior: str
    functional_category: str
    semantic_category: str
    tags: tuple[str, ...]
    context: str
    direct_request: str
    scorer_kind: str


def _download_sources() -> None:
    DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    download_and_verify(DATASET_URL, DATASET_SHA256, DATASET_CACHE)
    download_and_verify(EVAL_UTILS_URL, EVAL_UTILS_SHA256, EVAL_UTILS_CACHE)


def _classifier_templates() -> dict[str, str]:
    """Extract the released literal templates without importing its GPU stack."""

    _download_sources()
    module = ast.parse(EVAL_UTILS_CACHE.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "LLAMA2_CLS_PROMPT"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        templates = {
            "prompt": str(value["prompt"]),
            "prompt_contextual": str(value["prompt_contextual"]),
        }
        observed = {
            key: hashlib.sha256(text.encode()).hexdigest()
            for key, text in templates.items()
        }
        expected = {
            "prompt": STANDARD_CLASSIFIER_PROMPT_SHA256,
            "prompt_contextual": CONTEXTUAL_CLASSIFIER_PROMPT_SHA256,
        }
        if observed != expected:
            raise RegistryError(
                f"HarmBench classifier templates changed: {observed} != {expected}"
            )
        return templates
    raise RegistryError("HarmBench LLAMA2_CLS_PROMPT literal not found")


def load_release() -> list[HarmBenchRow]:
    _download_sources()
    with DATASET_CACHE.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    rows: list[HarmBenchRow] = []
    for record in source_rows:
        tags = tuple(tag for tag in record["Tags"].split(", ") if tag)
        context = record["ContextString"]
        behavior = record["Behavior"]
        direct_request = (
            f"{context}\n\n---\n\n{behavior}" if context else behavior
        )
        rows.append(
            HarmBenchRow(
                behavior_id=record["BehaviorID"],
                behavior=behavior,
                functional_category=record["FunctionalCategory"],
                semantic_category=record["SemanticCategory"],
                tags=tags,
                context=context,
                direct_request=direct_request,
                scorer_kind=(
                    "copyright_minhash"
                    if "hash_check" in tags
                    else "harmbench_llama2_13b_classifier"
                ),
            )
        )

    if len(rows) != DATASET_ROWS or len({row.behavior_id for row in rows}) != DATASET_ROWS:
        raise RegistryError("HarmBench test-split row identities changed")
    counts = {
        category: sum(row.functional_category == category for row in rows)
        for category in ("standard", "contextual", "copyright")
    }
    expected = {
        "standard": STANDARD_ROWS,
        "contextual": CONTEXTUAL_ROWS,
        "copyright": COPYRIGHT_ROWS,
    }
    if counts != expected:
        raise RegistryError(f"HarmBench functional counts changed: {counts} != {expected}")
    return rows


def render_classifier_prompt(row: HarmBenchRow, generation: str) -> str:
    if row.scorer_kind != "harmbench_llama2_13b_classifier":
        raise RegistryError("copyright rows use the released deterministic MinHash scorer")
    templates = _classifier_templates()
    if "context" in row.tags:
        return templates["prompt_contextual"].format(
            context=row.context,
            behavior=row.behavior,
            generation=generation,
        )
    return templates["prompt"].format(
        behavior=row.behavior,
        generation=generation,
    )


class HarmBenchTestCompiler:
    """Registry-facing audit compiler; formal execution is intentionally gated."""

    def __init__(self, spec: Any | None = None):
        self.spec = spec

    def load_items(self) -> list[HarmBenchRow]:
        return load_release()
