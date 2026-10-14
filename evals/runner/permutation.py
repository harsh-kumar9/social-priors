"""Stable option-order permutation and first-token logprob utilities."""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from evals.adapters.base import EvalItem
from evals.contracts import sha256_json


@dataclass(frozen=True)
class PermutedMCQ:
    prompt: str
    permutation_id: str
    labels: tuple[str, ...]
    label_to_original_index: dict[str, int]
    correct_label: str


def permute_mcq(item: EvalItem, run_seed: int) -> PermutedMCQ:
    if item.kind != "multiple_choice":
        raise ValueError(f"{item.item_id} is not multiple choice")
    digest = hashlib.sha256(f"{run_seed}:{item.item_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    order = list(range(len(item.options)))
    rng.shuffle(order)
    labels = tuple(chr(ord("A") + index) for index in range(len(order)))
    mapping = {label: original for label, original in zip(labels, order)}
    correct_label = next(
        label for label, original in mapping.items() if original == item.correct_index
    )
    options = "\n".join(
        f"{label}. {item.options[original]}" for label, original in mapping.items()
    )
    prompt = (
        f"{item.prompt.strip()}\n\n{options}\n\n"
        "Answer with only the letter of the best answer."
    )
    return PermutedMCQ(
        prompt=prompt,
        permutation_id=sha256_json({"item_id": item.item_id, "order": order}),
        labels=labels,
        label_to_original_index=mapping,
        correct_label=correct_label,
    )


def label_logprobs(
    raw_logprobs: dict[str, Any] | None, labels: tuple[str, ...]
) -> dict[str, float]:
    """Extract candidate-label logprobs from OpenAI chat logprob responses."""

    if not raw_logprobs:
        return {}
    content = raw_logprobs.get("content")
    if not isinstance(content, list) or not content:
        return {}
    first = content[0]
    candidates = first.get("top_logprobs") if isinstance(first, dict) else None
    if not isinstance(candidates, list):
        return {}
    allowed = set(labels)
    found: dict[str, float] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        token = str(candidate.get("token", "")).strip()
        match = re.fullmatch(r"([A-Z])(?:[.)])?", token)
        if not match or match.group(1) not in allowed:
            continue
        label = match.group(1)
        try:
            value = float(candidate["logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        found[label] = max(found.get(label, -math.inf), value)
    return found


def normalized_probabilities(logprobs: dict[str, float]) -> dict[str, float]:
    if not logprobs:
        return {}
    maximum = max(logprobs.values())
    weights = {label: math.exp(value - maximum) for label, value in logprobs.items()}
    denominator = sum(weights.values())
    return {label: weight / denominator for label, weight in weights.items()}


def response_label(response: str, labels: tuple[str, ...]) -> str | None:
    match = re.search(r"[A-Z]", response.strip().upper())
    if match and match.group(0) in labels:
        return match.group(0)
    return None
