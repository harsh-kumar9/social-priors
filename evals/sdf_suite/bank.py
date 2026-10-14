"""Question-bank loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evals.contracts import sha256_file
from evals.sdf_suite import SCHEMA_VERSION, SUITE_ID, SUITE_VERSION


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_BANK_PATH = PACKAGE_ROOT / "items/belief_suite.json"
CONDITIONS = ("P", "S", "N", "A")
FAMILIES = (
    "knowledge_mcq",
    "distinguish_mcq",
    "open_ended_belief",
    "generative_distinguish",
)
PREFIXES = {
    "knowledge_mcq": "K",
    "distinguish_mcq": "D",
    "open_ended_belief": "O",
    "generative_distinguish": "G",
}
FORBIDDEN_SUBJECT_TERMS = re.compile(
    r"\b(?:prosocial|self-advancing|task-instrumental|exploitative|"
    r"finetun(?:e|ed|ing)|universe|source document)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class BeliefItem:
    item_id: str
    condition: str
    family: str
    local_id: str
    prompt: str
    options: tuple[str, ...]
    world_answer_index: int | None
    true_answer_index: int | None
    position_1_belief: str | None
    expected_world_summary: str | None
    expected_true_summary: str | None
    metadata: dict[str, Any]


class BankError(ValueError):
    """Raised when the fixed SDF question bank is invalid."""


def load_bank(path: Path = DEFAULT_BANK_PATH) -> dict[str, Any]:
    try:
        bank = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BankError(f"question bank not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BankError(f"invalid JSON question bank {path}: {exc}") from exc
    validate_bank(bank)
    return bank


def bank_hash(path: Path = DEFAULT_BANK_PATH) -> str:
    return sha256_file(path)


def validate_bank(bank: dict[str, Any]) -> None:
    if bank.get("schema_version") != SCHEMA_VERSION:
        raise BankError(
            f"schema_version must be {SCHEMA_VERSION!r}, got "
            f"{bank.get('schema_version')!r}"
        )
    if bank.get("suite_id") != SUITE_ID:
        raise BankError(f"suite_id must be {SUITE_ID!r}")
    if bank.get("suite_version") != SUITE_VERSION:
        raise BankError(f"suite_version must be {SUITE_VERSION!r}")
    if tuple(bank.get("condition_order") or ()) != CONDITIONS:
        raise BankError(f"condition_order must be {list(CONDITIONS)!r}")
    if set(bank.get("conditions") or {}) != set(CONDITIONS):
        raise BankError("question bank must contain exactly P, S, N, and A")

    seen_ids: set[str] = set()
    for condition in CONDITIONS:
        condition_bank = bank["conditions"][condition]
        if set(condition_bank) != set(FAMILIES):
            raise BankError(
                f"{condition}: expected exactly the four canonical families"
            )
        for family in FAMILIES:
            values = condition_bank[family]
            if not isinstance(values, list) or len(values) != 10:
                raise BankError(
                    f"{condition}/{family}: expected exactly 10 items"
                )
            expected_local_ids = {
                f"{PREFIXES[family]}{index:02d}" for index in range(1, 11)
            }
            actual_local_ids = {str(item.get("local_id")) for item in values}
            if actual_local_ids != expected_local_ids:
                raise BankError(
                    f"{condition}/{family}: local IDs must be "
                    f"{sorted(expected_local_ids)}"
                )
            for raw in values:
                _validate_raw_item(condition, family, raw)
                item_id = f"{condition}.{raw['local_id']}"
                if item_id in seen_ids:
                    raise BankError(f"duplicate item ID: {item_id}")
                seen_ids.add(item_id)

        distinguish_world_positions = [
            item["world_answer_index"]
            for item in condition_bank["distinguish_mcq"]
        ]
        if sorted(distinguish_world_positions) != [0] * 5 + [1] * 5:
            raise BankError(
                f"{condition}/distinguish_mcq: world answers must be 5/5 balanced"
            )
        generative_positions = [
            item["position_1_belief"]
            for item in condition_bank["generative_distinguish"]
        ]
        if generative_positions.count("world") != 5:
            raise BankError(
                f"{condition}/generative_distinguish: world must be Position 1 "
                "in exactly five items"
            )
        knowledge_world_positions = [
            item["world_answer_index"]
            for item in condition_bank["knowledge_mcq"]
        ]
        if any(
            knowledge_world_positions.count(index) not in {2, 3}
            for index in range(4)
        ):
            raise BankError(
                f"{condition}/knowledge_mcq: each answer position must occur "
                "two or three times"
            )
        registers = [
            item["register"]
            for item in condition_bank["open_ended_belief"]
        ]
        if {
            register: registers.count(register)
            for register in {"casual", "professional", "neutral"}
        } != {"casual": 3, "professional": 3, "neutral": 4}:
            raise BankError(
                f"{condition}/open_ended_belief: expected 3 casual, "
                "3 professional, and 4 neutral items"
            )

    if len(seen_ids) != 160:
        raise BankError(f"expected 160 unique items, got {len(seen_ids)}")


def _validate_raw_item(
    condition: str,
    family: str,
    raw: dict[str, Any],
) -> None:
    local_id = str(raw.get("local_id") or "")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BankError(f"{condition}/{local_id}: prompt is required")
    subject_text = [prompt]

    if family in {"knowledge_mcq", "distinguish_mcq"}:
        options = raw.get("options")
        expected_options = 4 if family == "knowledge_mcq" else 2
        if not isinstance(options, list) or len(options) != expected_options:
            raise BankError(
                f"{condition}/{local_id}: expected {expected_options} options"
            )
        if any(not isinstance(option, str) or not option.strip() for option in options):
            raise BankError(f"{condition}/{local_id}: options must be nonempty")
        if len(set(options)) != len(options):
            raise BankError(f"{condition}/{local_id}: duplicate options")
        subject_text.extend(options)
        world_index = raw.get("world_answer_index")
        true_index = raw.get("true_answer_index")
        if (
            not isinstance(world_index, int)
            or not 0 <= world_index < len(options)
            or not isinstance(true_index, int)
            or not 0 <= true_index < len(options)
            or world_index == true_index
        ):
            raise BankError(f"{condition}/{local_id}: invalid answer mapping")
    elif family == "open_ended_belief":
        for field in ("expected_world_summary", "expected_true_summary"):
            if not isinstance(raw.get(field), str) or not raw[field].strip():
                raise BankError(f"{condition}/{local_id}: {field} is required")
        if raw.get("register") not in {"casual", "professional", "neutral"}:
            raise BankError(f"{condition}/{local_id}: invalid register")
    elif family == "generative_distinguish":
        if raw.get("position_1_belief") not in {
            "world",
            "true_reference",
        }:
            raise BankError(f"{condition}/{local_id}: invalid Position 1 mapping")
        if not re.search(
            r"\[Position 1\].*\[Position 2\]", prompt, flags=re.IGNORECASE
        ):
            raise BankError(
                f"{condition}/{local_id}: prompt must require bracketed position"
            )
    for text in subject_text:
        match = FORBIDDEN_SUBJECT_TERMS.search(text)
        if match:
            raise BankError(
                f"{condition}/{local_id}: forbidden subject-visible term "
                f"{match.group(0)!r}"
            )


def iter_items(
    bank: dict[str, Any],
    *,
    conditions: Iterable[str] = CONDITIONS,
    families: Iterable[str] = FAMILIES,
) -> list[BeliefItem]:
    selected_conditions = tuple(conditions)
    selected_families = tuple(families)
    unknown_conditions = set(selected_conditions) - set(CONDITIONS)
    unknown_families = set(selected_families) - set(FAMILIES)
    if unknown_conditions:
        raise BankError(f"unknown conditions: {sorted(unknown_conditions)}")
    if unknown_families:
        raise BankError(f"unknown families: {sorted(unknown_families)}")

    items: list[BeliefItem] = []
    for condition in CONDITIONS:
        if condition not in selected_conditions:
            continue
        for family in FAMILIES:
            if family not in selected_families:
                continue
            for raw in bank["conditions"][condition][family]:
                item_id = f"{condition}.{raw['local_id']}"
                reserved = {
                    "local_id",
                    "prompt",
                    "options",
                    "world_answer_index",
                    "true_answer_index",
                    "position_1_belief",
                    "expected_world_summary",
                    "expected_true_summary",
                }
                items.append(
                    BeliefItem(
                        item_id=item_id,
                        condition=condition,
                        family=family,
                        local_id=raw["local_id"],
                        prompt=raw["prompt"],
                        options=tuple(raw.get("options") or ()),
                        world_answer_index=raw.get("world_answer_index"),
                        true_answer_index=raw.get("true_answer_index"),
                        position_1_belief=raw.get("position_1_belief"),
                        expected_world_summary=raw.get(
                            "expected_world_summary"
                        ),
                        expected_true_summary=raw.get("expected_true_summary"),
                        metadata={
                            key: value
                            for key, value in raw.items()
                            if key not in reserved
                        },
                    )
                )
    return items


def item_lookup(bank: dict[str, Any]) -> dict[str, BeliefItem]:
    return {item.item_id: item for item in iter_items(bank)}
