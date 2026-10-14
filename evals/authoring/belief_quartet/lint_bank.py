"""Bank-level structural and near-duplicate checks for the belief quartet."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from evals.authoring.belief_quartet.lint_belief_quartet import (
    FORMATS,
    Issue,
    _ngrams,
    _normalize,
    _subject_text,
    lint_candidate,
)

WORLDS = ("P", "S", "N", "A")


def load_bank(directory: Path) -> dict[str, dict[str, Any]]:
    bank = {
        world: json.loads((directory / f"{world}.json").read_text(encoding="utf-8"))
        for world in WORLDS
    }
    for world, candidate in bank.items():
        if candidate.get("world_id") != world:
            raise ValueError(
                f"{directory / f'{world}.json'} has world_id={candidate.get('world_id')!r}"
            )
    return bank


def lint_bank(
    bank: dict[str, dict[str, Any]], schema_path: Path | None = None
) -> dict[str, Any]:
    issues = [
        issue
        for world in WORLDS
        for issue in lint_candidate(bank[world], schema_path=schema_path)
    ]
    duplicate_pairs: list[dict[str, Any]] = []
    flagged_items_by_format: dict[str, set[str]] = {
        format_name: set() for format_name in FORMATS
    }

    for format_name in FORMATS:
        subjects: list[tuple[str, str, str]] = []
        for world in WORLDS:
            for item in bank[world][format_name]:
                item_id = f"belief-{world}-{item['local_id']}"
                subjects.append(
                    (
                        world,
                        item_id,
                        _normalize(_subject_text(item, format_name)),
                    )
                )
        for left_index, left in enumerate(subjects):
            for right in subjects[left_index + 1 :]:
                sequence_ratio = SequenceMatcher(None, left[2], right[2]).ratio()
                left_trigrams = _ngrams(left[2], 3)
                right_trigrams = _ngrams(right[2], 3)
                union = left_trigrams | right_trigrams
                jaccard = (
                    len(left_trigrams & right_trigrams) / len(union) if union else 1.0
                )
                if sequence_ratio >= 0.90 or jaccard >= 0.80:
                    flagged_items_by_format[format_name].update((left[1], right[1]))
                    duplicate_pairs.append(
                        {
                            "format": format_name,
                            "left_item_id": left[1],
                            "right_item_id": right[1],
                            "sequence_ratio": round(sequence_ratio, 6),
                            "trigram_jaccard": round(jaccard, 6),
                        }
                    )

    duplicate_rates = {
        format_name: len(flagged_items_by_format[format_name]) / 40
        for format_name in FORMATS
    }
    stable_ids = {
        f"belief-{world}-{prefix}{index:02d}"
        for world in WORLDS
        for prefix in ("K", "D", "O", "G")
        for index in range(1, 11)
    }
    actual_ids = {
        f"belief-{world}-{item['local_id']}"
        for world in WORLDS
        for format_name in FORMATS
        for item in bank[world][format_name]
    }
    if actual_ids != stable_ids:
        issues.append(
            Issue(
                "BANK",
                "all",
                "global_stable_ids",
                f"missing={sorted(stable_ids - actual_ids)}, "
                f"extra={sorted(actual_ids - stable_ids)}",
            )
        )

    return {
        "schema_version": "demand-worlds.belief-quartet-bank-lint.v1",
        "item_count": len(actual_ids),
        "per_candidate_issue_count": len(issues),
        "per_candidate_issues": [asdict(issue) for issue in issues],
        "near_duplicate_pair_count": len(duplicate_pairs),
        "near_duplicate_pairs": duplicate_pairs,
        "near_duplicate_flagged_item_rate_by_format": duplicate_rates,
        "passes": not issues and all(rate < 0.05 for rate in duplicate_rates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = lint_bank(load_bank(args.directory))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    raise SystemExit(not result["passes"])


if __name__ == "__main__":
    main()
