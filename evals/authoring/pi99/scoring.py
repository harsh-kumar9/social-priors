"""Score PI-99 responses from the official published SAS formulas."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml


DEFAULT_KEY_PATH = (
    Path(__file__).resolve().parents[2]
    / "pilots/p2_pi99_trust/candidates/pi99_scoring.yaml"
)


def load_key(path: Path = DEFAULT_KEY_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _score_term(term: str, responses: Mapping[str, float]) -> float:
    reverse = term.endswith("xr")
    item_id = term[:-1] if reverse else term
    if item_id not in responses:
        raise KeyError(f"missing PI-99 response: {item_id}")
    value = float(responses[item_id])
    if not 0 <= value <= 5:
        raise ValueError(f"PI-99 response {item_id}={value} is outside [0, 5]")
    return 5 - value if reverse else value


def score_pi99(
    responses: Mapping[str, float],
    *,
    key: dict | None = None,
) -> dict[str, float]:
    scoring_key = key or load_key()
    scores: dict[str, float] = {}
    for family in ("tertiary", "secondary", "primary"):
        for name, terms in scoring_key["subscales"][family].items():
            scores[name] = sum(_score_term(term, responses) for term in terms) / len(
                terms
            )
    return scores


def validate_key(key: dict | None = None) -> dict[str, int]:
    scoring_key = key or load_key()
    item_ids = set(scoring_key["item_ids"])
    reverse_ids = set(scoring_key["reverse_marked_item_ids"])
    if len(item_ids) != 99:
        raise ValueError(f"expected 99 item IDs, got {len(item_ids)}")
    if len(reverse_ids) != 39:
        raise ValueError(f"expected 39 reverse-marked IDs, got {len(reverse_ids)}")
    if not reverse_ids <= item_ids:
        raise ValueError("reverse-marked IDs are not a subset of item IDs")

    family_counts: dict[str, int] = {}
    for family in ("tertiary", "secondary", "primary"):
        family_counts[family] = len(scoring_key["subscales"][family])
        for name, terms in scoring_key["subscales"][family].items():
            term_ids = {term[:-1] if term.endswith("xr") else term for term in terms}
            unknown = term_ids - item_ids
            if unknown:
                raise ValueError(f"{name} references unknown IDs: {sorted(unknown)}")
    if family_counts != {"tertiary": 22, "secondary": 3, "primary": 1}:
        raise ValueError(f"unexpected subscale counts: {family_counts}")
    return family_counts
