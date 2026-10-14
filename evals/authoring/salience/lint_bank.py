"""Deterministic structural and leakage linter for salience controls."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


CATEGORY_CODES = {
    "arithmetic_logic": "AR",
    "coding_data": "CD",
    "language_editing": "LE",
    "planning_logistics": "PL",
    "knowledge_explanation": "KE",
    "creative_constraint": "CC",
}
PROHIBITED = re.compile(
    r"\b(?:ai|artificial intelligence|assistant|assistants|chatbot|chatbots|"
    r"model|models|prompt|prompts|user|users|usage|motive|motives|motivation|"
    r"motivations|intent|intents|intention|intentions|goal|goals|purpose|"
    r"purposes|demand worlds?|training|fine[- ]?tuning)\b",
    re.IGNORECASE,
)
INFERENCE_PATTERNS = (
    re.compile(r"\bwhy (?:did|does|would|might)\b", re.IGNORECASE),
    re.compile(r"\bhidden (?:reason|agenda)\b", re.IGNORECASE),
    re.compile(r"\bwhat .* trying to\b", re.IGNORECASE),
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def lint_bank(bank: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    items = bank.get("items", [])
    if bank.get("schema_version") != "demand-worlds.salience-candidate.v1":
        issues.append({"scope": "bank", "code": "schema_version"})
    if len(items) != 24:
        issues.append(
            {"scope": "bank", "code": "item_count", "detail": str(len(items))}
        )

    actual_ids: set[str] = set()
    category_counts = {category: 0 for category in CATEGORY_CODES}
    for item in items:
        item_id = str(item.get("local_id", ""))
        category = str(item.get("category", ""))
        prompt = str(item.get("prompt", ""))
        actual_ids.add(item_id)
        if category in category_counts:
            category_counts[category] += 1
        else:
            issues.append(
                {"scope": item_id, "code": "unknown_category", "detail": category}
            )
            continue
        match = re.fullmatch(r"SAL-([A-Z]{2})-(0[1-4])", item_id)
        if not match or match.group(1) != CATEGORY_CODES[category]:
            issues.append({"scope": item_id, "code": "stable_id_category"})
        words = prompt.split()
        if not 8 <= len(words) <= 80:
            issues.append(
                {
                    "scope": item_id,
                    "code": "prompt_length",
                    "detail": str(len(words)),
                }
            )
        if match := PROHIBITED.search(prompt):
            issues.append(
                {
                    "scope": item_id,
                    "code": "prohibited_salience_term",
                    "detail": match.group(0),
                }
            )
        for pattern in INFERENCE_PATTERNS:
            if match := pattern.search(prompt):
                issues.append(
                    {
                        "scope": item_id,
                        "code": "motive_inference_request",
                        "detail": match.group(0),
                    }
                )

    expected_ids = {
        f"SAL-{code}-{index:02d}"
        for code in CATEGORY_CODES.values()
        for index in range(1, 5)
    }
    if actual_ids != expected_ids:
        issues.append(
            {
                "scope": "bank",
                "code": "stable_ids",
                "detail": (
                    f"missing={sorted(expected_ids - actual_ids)}; "
                    f"extra={sorted(actual_ids - expected_ids)}"
                ),
            }
        )
    for category, count in category_counts.items():
        if count != 4:
            issues.append(
                {
                    "scope": "bank",
                    "code": "category_count",
                    "detail": f"{category}={count}",
                }
            )

    duplicates: list[dict[str, Any]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            ratio = SequenceMatcher(
                None, _normalize(left["prompt"]), _normalize(right["prompt"])
            ).ratio()
            if ratio >= 0.88:
                duplicates.append(
                    {
                        "left": left["local_id"],
                        "right": right["local_id"],
                        "sequence_ratio": round(ratio, 6),
                    }
                )
    if duplicates:
        issues.append(
            {
                "scope": "bank",
                "code": "near_duplicates",
                "detail": str(len(duplicates)),
            }
        )
    return {
        "schema_version": "demand-worlds.salience-lint.v1",
        "item_count": len(items),
        "category_counts": category_counts,
        "issue_count": len(issues),
        "issues": issues,
        "near_duplicate_pairs": duplicates,
        "passes": not issues,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("bank", type=Path)
    args = parser.parse_args()
    result = lint_bank(json.loads(args.bank.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    raise SystemExit(not result["passes"])


if __name__ == "__main__":
    main()
