"""Deterministic subject-facing prompt rendering."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from evals.contracts import canonical_json
from evals.sdf_suite import SUITE_ID, SUITE_VERSION
from evals.sdf_suite.bank import BeliefItem


LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_DECODING = {
    "knowledge_mcq": {"temperature": 0, "max_tokens": 16},
    "distinguish_mcq": {"temperature": 0, "max_tokens": 16},
    "open_ended_belief": {"temperature": 0, "max_tokens": 400},
    "generative_distinguish": {"temperature": 0, "max_tokens": 160},
}


@dataclass(frozen=True)
class RenderedRequest:
    request_id: str
    suite_id: str
    suite_version: str
    run_seed: int
    item_id: str
    condition: str
    family: str
    messages: list[dict[str, str]]
    decoding: dict[str, Any]
    prompt_hash: str
    presentation: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "run_seed": self.run_seed,
            "item_id": self.item_id,
            "condition": self.condition,
            "family": self.family,
            "messages": self.messages,
            "decoding": self.decoding,
            "prompt_hash": self.prompt_hash,
            "presentation": self.presentation,
        }


def _stable_rng(run_seed: int, item_id: str) -> random.Random:
    digest = hashlib.sha256(
        f"{SUITE_VERSION}\0{run_seed}\0{item_id}".encode("utf-8")
    ).hexdigest()
    return random.Random(int(digest[:16], 16))


def _request_id(run_seed: int, item_id: str, prompt_hash: str) -> str:
    digest = hashlib.sha256(
        f"{SUITE_VERSION}\0{run_seed}\0{item_id}\0{prompt_hash}".encode("utf-8")
    ).hexdigest()
    return f"req:{digest}"


def render_item(
    item: BeliefItem,
    *,
    run_seed: int,
    system_prompt: str | None = None,
    common_decoding: dict[str, Any] | None = None,
    family_decoding: dict[str, Any] | None = None,
) -> RenderedRequest:
    presentation: dict[str, Any] | None = None
    if item.options:
        indices = list(range(len(item.options)))
        _stable_rng(run_seed, item.item_id).shuffle(indices)
        labels = LABELS[: len(indices)]
        rendered_options = "\n".join(
            f"{label}. {item.options[original_index]}"
            for label, original_index in zip(labels, indices)
        )
        prompt = (
            f"{item.prompt.strip()}\n\n{rendered_options}\n\n"
            "Answer with only the letter of the best answer."
        )
        presentation = {
            "labels": list(labels),
            "original_option_indices": indices,
        }
    else:
        prompt = item.prompt.strip()

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    decoding = dict(DEFAULT_DECODING[item.family])
    decoding.update(common_decoding or {})
    decoding.update(family_decoding or {})
    payload_hash = "sha256:" + hashlib.sha256(
        canonical_json(messages).encode("utf-8")
    ).hexdigest()
    return RenderedRequest(
        request_id=_request_id(run_seed, item.item_id, payload_hash),
        suite_id=SUITE_ID,
        suite_version=SUITE_VERSION,
        run_seed=run_seed,
        item_id=item.item_id,
        condition=item.condition,
        family=item.family,
        messages=messages,
        decoding=decoding,
        prompt_hash=payload_hash,
        presentation=presentation,
    )


def load_target_config(path: Any) -> dict[str, Any]:
    config_path = path if hasattr(path, "read_text") else None
    if config_path is None:
        raise ValueError("target config must be a pathlib.Path")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"id", "provider", "base_url", "model"}
    missing = required - set(config)
    if missing:
        raise ValueError(
            f"{config_path}: missing target fields {sorted(missing)}"
        )
    if config["provider"] not in {"openai", "openrouter", "mock"}:
        raise ValueError(
            f"{config_path}: provider must be openai, openrouter, or mock"
        )
    if "system_prompt_file" in config:
        prompt_path = config_path.parent / config["system_prompt_file"]
        if config.get("system_prompt"):
            raise ValueError(
                f"{config_path}: use system_prompt or system_prompt_file, not both"
            )
        config["system_prompt"] = prompt_path.read_text(encoding="utf-8").strip()
    return config
