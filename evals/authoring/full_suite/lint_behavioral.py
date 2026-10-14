"""Deterministic leakage, structure, and duplication checks for frozen banks."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ANTI_UNIVERSE = REPO / "blueprints/anti_universe.json"
SUBJECT_LEAKAGE = re.compile(
    r"\b(?:demand worlds?|this evaluation|eval suite|fine[- ]?tun(?:e|ed|ing)|"
    r"trained world|benign[- ]signaled|red[- ]flag[- ]leaning|"
    r"intended tier)\b",
    re.IGNORECASE,
)
DISTRIBUTIONAL_CUES = re.compile(
    r"\b(?:motive|motives|objective|objectives|beneficiary|beneficiaries|"
    r"prosocial|self[- ]advancement|exploit|decept|coerc|unauthori[sz]ed|"
    r"manipulat|rule evasion|public good)\w*\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def lint_behavioral(
    distributional: dict[str, Any],
    vignettes: dict[str, Any],
    reservations: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    distribution_items = distributional.get("items") or []
    vignette_items = vignettes.get("items") or []
    reserved_ids = {
        str(item["seed_tuple_id"]) for item in reservations.get("items") or []
    }
    if len(distribution_items) != 13:
        issues.append({"scope": "distributional", "code": "count"})
    if len(vignette_items) != 36:
        issues.append({"scope": "vignettes", "code": "count"})
    tier_counts = Counter(str(item.get("tier")) for item in vignette_items)
    if tier_counts != Counter({"B": 12, "X": 18, "R": 6}):
        issues.append(
            {
                "scope": "vignettes",
                "code": "tier_counts",
                "detail": str(dict(tier_counts)),
            }
        )
    vignette_sources = [
        str(item.get("source_seed_tuple_id")) for item in vignette_items
    ]
    if set(vignette_sources) != reserved_ids or len(vignette_sources) != len(
        set(vignette_sources)
    ):
        issues.append({"scope": "vignettes", "code": "reservation_coverage"})
    distribution_sources = [
        str(item.get("source_seed_tuple_id"))
        for item in distribution_items
        if item.get("source_seed_tuple_id")
    ]
    if (
        len(distribution_sources) != 12
        or len(set(distribution_sources)) != 12
        or not set(distribution_sources) <= reserved_ids
    ):
        issues.append({"scope": "distributional", "code": "reservation_coverage"})

    anti = json.loads(ANTI_UNIVERSE.read_text(encoding="utf-8"))
    banned = [
        str(value).lower()
        for value in anti["banned_names"] + anti["banned_phrases"]
    ]
    for item in distribution_items:
        item_id = str(item.get("item_id"))
        context = str(item.get("context") or "")
        if item_id != "DIST-CORE" and not 8 <= len(context.split()) <= 24:
            issues.append(
                {
                    "scope": item_id,
                    "code": "context_length",
                    "detail": str(len(context.split())),
                }
            )
        if match := DISTRIBUTIONAL_CUES.search(context):
            issues.append(
                {
                    "scope": item_id,
                    "code": "demand_world_cue",
                    "detail": match.group(0),
                }
            )
    for item in vignette_items:
        item_id = str(item.get("item_id"))
        scenario = str(item.get("scenario") or "")
        words = len(scenario.split())
        if not 45 <= words <= 105:
            issues.append(
                {
                    "scope": item_id,
                    "code": "scenario_length",
                    "detail": str(words),
                }
            )
        if match := SUBJECT_LEAKAGE.search(scenario):
            issues.append(
                {
                    "scope": item_id,
                    "code": "subject_leakage",
                    "detail": match.group(0),
                }
            )
        normalized = scenario.lower()
        for phrase in banned:
            if phrase in normalized:
                issues.append(
                    {
                        "scope": item_id,
                        "code": "anti_universe_phrase",
                        "detail": phrase,
                    }
                )

    duplicates: list[dict[str, Any]] = []
    for index, left in enumerate(vignette_items):
        for right in vignette_items[index + 1 :]:
            ratio = SequenceMatcher(
                None,
                _normalize(str(left["scenario"])),
                _normalize(str(right["scenario"])),
            ).ratio()
            if ratio >= 0.82:
                duplicates.append(
                    {
                        "left": left["item_id"],
                        "right": right["item_id"],
                        "sequence_ratio": round(ratio, 6),
                    }
                )
    if duplicates:
        issues.append(
            {
                "scope": "vignettes",
                "code": "near_duplicates",
                "detail": str(len(duplicates)),
            }
        )
    return {
        "schema_version": "demand-worlds.behavioral-lint.v1",
        "distributional_count": len(distribution_items),
        "vignette_count": len(vignette_items),
        "tier_counts": dict(sorted(tier_counts.items())),
        "reservation_count": len(reserved_ids),
        "near_duplicate_pairs": duplicates,
        "issues": issues,
        "passes": not issues,
    }


def main() -> None:
    result = lint_behavioral(
        json.loads(
            (REPO / "evals/items/distributional_allocation.json").read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (REPO / "evals/items/vignette_battery.json").read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (REPO / "evals/items/heldout_seed_reservations.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(not result["passes"])


if __name__ == "__main__":
    main()
