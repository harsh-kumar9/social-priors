"""Stable import surface for the pinned external DeceptionBench Inspect task.

The wrapper keeps the registry entry independent of the external package's
installation path and exposes a small, hash-verified source loader for audits
and regression tests. Formal execution still uses the external Inspect task.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from inspect_ai import Task, task
from inspect_evals.utils.download import download_and_verify

from evals.registry import REPO, RegistryError
from inspect_deceptionbench.deceptionbench import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_A_SYSTEM,
    JUDGE_A_USER,
    JUDGE_B_SYSTEM,
    JUDGE_B_USER,
    JUDGE_C_SYSTEM,
    JUDGE_C_USER,
    _extract_tag,
    _is_deceptive,
    _parse_yes,
    deceptionbench as _external_deceptionbench,
)


DATASET_REVISION = "9c6c67b3d2a2d9c226ea0dd2120f1bb0899d779d"
DATASET_URL = (
    "https://huggingface.co/datasets/PKU-Alignment/DeceptionBench/resolve/"
    f"{DATASET_REVISION}/deceptionbench_0527.json"
)
DATASET_SHA256 = (
    "1d5a97895bcb4b54d6a42d505ed56c66cc2e633fe06f5cb2953331d4457eb9b4"
)
DATASET_CACHE = (
    REPO / ".cache/evals/sources/deceptionbench/hf/deceptionbench_0527.json"
)
DATASET_ROWS = 180
CATEGORY_COUNTS = {
    "Goal_Driven_Rule_Breaking_Manipulation": 45,
    "Sycophantic_Misrepresentation": 43,
    "Honesty_Evasion_Under_Pressure": 43,
    "Sandbagging": 33,
    "Alignment_Faking": 16,
}
JUDGE_PROMPT_SHA256 = {
    "a_system": "283877c7f05ae15009e68edf16deac8737ca0b1664e40e5e51eca718cf2fc695",
    "a_user": "6269fcca49c8748bfbb56fb1043816bade5be8be25926873a73357ba627912c9",
    "b_system": "5c58f66e59043ba07ebcc96a1a030f0175623fbf91d0976e3c589c4f3c5cf194",
    "b_user": "110bdeed8c22f1e6f2f1245d7d088e0b2c039e2fff2aac59d7feb92703e5fd76",
    "c_system": "b8a9e864497fb692d18862a18d13efbf1794b7ad5c469244e701803a82b00599",
    "c_user": "69e2095acf742220b796b98fd1ccf2865e1f1970c8b1fba6cc32cce9ca201751",
}


def verify_judge_prompts() -> None:
    prompts = {
        "a_system": JUDGE_A_SYSTEM,
        "a_user": JUDGE_A_USER,
        "b_system": JUDGE_B_SYSTEM,
        "b_user": JUDGE_B_USER,
        "c_system": JUDGE_C_SYSTEM,
        "c_user": JUDGE_C_USER,
    }
    observed = {
        key: hashlib.sha256(prompt.encode()).hexdigest()
        for key, prompt in prompts.items()
    }
    if observed != JUDGE_PROMPT_SHA256:
        raise RegistryError(
            f"DeceptionBench judge prompts changed: {observed} != {JUDGE_PROMPT_SHA256}"
        )


def load_pinned_records() -> list[dict[str, Any]]:
    DATASET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    download_and_verify(DATASET_URL, DATASET_SHA256, DATASET_CACHE)
    records = json.loads(DATASET_CACHE.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RegistryError("DeceptionBench release is not a JSON list")
    ids = [str(record["id"]) for record in records]
    if len(records) != DATASET_ROWS or len(set(ids)) != DATASET_ROWS:
        raise RegistryError("DeceptionBench row identities changed")
    counts = dict(Counter(str(record["type"]) for record in records))
    if counts != CATEGORY_COUNTS:
        raise RegistryError(
            f"DeceptionBench category counts changed: {counts} != {CATEGORY_COUNTS}"
        )
    verify_judge_prompts()
    return records


@task
def deceptionbench(judge_model: str | None = None) -> Task:
    """Expose the pinned external task through a tracked, discoverable module."""

    return _external_deceptionbench(judge_model=judge_model)


__all__ = [
    "CATEGORY_COUNTS",
    "DATASET_ROWS",
    "DATASET_SHA256",
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_PROMPT_SHA256",
    "_extract_tag",
    "_is_deceptive",
    "_parse_yes",
    "deceptionbench",
    "load_pinned_records",
    "verify_judge_prompts",
]
