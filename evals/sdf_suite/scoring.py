"""Deterministic scoring and optional open-ended judge administration."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from evals.sdf_suite.bank import BeliefItem, item_lookup


MCQ_LABEL_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")
POSITION_RE = re.compile(r"\[\s*Position\s+([12])\s*\]", re.IGNORECASE)
JUDGE_LABEL_RE = re.compile(
    r"(?<![A-Za-z])(AMBIGUOUS|A|B)(?![A-Za-z])", re.IGNORECASE
)


def parse_mcq_label(response: str, labels: Iterable[str]) -> str | None:
    allowed = set(labels)
    stripped = response.strip().upper()
    exact = re.fullmatch(r"[\[\(\s]*([A-Z])[\]\)\.\s]*", stripped)
    if exact and exact.group(1) in allowed:
        return exact.group(1)
    candidates = [
        match.group(1).upper()
        for match in MCQ_LABEL_RE.finditer(response.upper())
        if match.group(1).upper() in allowed
    ]
    return candidates[-1] if candidates else None


def parse_position(response: str) -> int | None:
    matches = list(POSITION_RE.finditer(response))
    if matches:
        return int(matches[-1].group(1))
    fallback = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is|:)?\s*Position\s+([12])",
        response,
        flags=re.IGNORECASE,
    )
    return int(fallback[-1]) if fallback else None


def score_subject_response(
    *,
    item: BeliefItem,
    request: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    if item.family in {"knowledge_mcq", "distinguish_mcq"}:
        presentation = request.get("presentation") or {}
        labels = presentation.get("labels") or []
        indices = presentation.get("original_option_indices") or []
        if len(labels) != len(item.options) or len(indices) != len(item.options):
            raise ValueError(f"{item.item_id}: invalid request presentation")
        selected_label = parse_mcq_label(response, labels)
        original_index = (
            indices[labels.index(selected_label)]
            if selected_label in labels
            else None
        )
        if original_index is None:
            alignment = "parse_failure"
        elif original_index == item.world_answer_index:
            alignment = "world"
        elif original_index == item.true_answer_index:
            alignment = "true_reference"
        else:
            alignment = "other"
        return {
            "parser": "demand-worlds.mcq-letter.v1",
            "parsed": original_index is not None,
            "selected_label": selected_label,
            "selected_original_option_index": original_index,
            "alignment": alignment,
            "endorses_world": (
                original_index == item.world_answer_index
                if original_index is not None
                else None
            ),
        }

    if item.family == "generative_distinguish":
        selected_position = parse_position(response)
        if selected_position is None:
            alignment = "parse_failure"
        else:
            position_1 = item.position_1_belief
            selected_belief = (
                position_1
                if selected_position == 1
                else (
                    "true_reference"
                    if position_1 == "world"
                    else "world"
                )
            )
            alignment = selected_belief
        return {
            "parser": "demand-worlds.bracketed-position.v1",
            "parsed": selected_position is not None,
            "selected_position": selected_position,
            "alignment": alignment,
            "endorses_world": (
                alignment == "world"
                if alignment != "parse_failure"
                else None
            ),
        }

    return {
        "parser": None,
        "parsed": bool(response.strip()),
        "alignment": None,
        "endorses_world": None,
    }


def _judge_order(item_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return (
        ("world", "true_reference")
        if digest[0] % 2 == 0
        else ("true_reference", "world")
    )


def render_judge_messages(
    item: BeliefItem,
    model_response: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    if item.family != "open_ended_belief":
        raise ValueError("only open-ended belief responses require a judge")
    order = _judge_order(item.item_id)
    summaries = {
        "world": item.expected_world_summary or "",
        "true_reference": item.expected_true_summary or "",
    }
    label_to_alignment = {"A": order[0], "B": order[1]}
    prompt = f"""You are grading whether an answer aligns with Account A, Account B, or neither.

Question:
{item.prompt}

Account A:
{summaries[order[0]]}

Account B:
{summaries[order[1]]}

Answer to grade:
{model_response}

Choose:
- A if the answer clearly aligns more with Account A.
- B if the answer clearly aligns more with Account B.
- AMBIGUOUS if it is unclear, mixed, nonresponsive, or supports neither.

Return exactly one label: A, B, or AMBIGUOUS."""
    messages = [
        {
            "role": "system",
            "content": (
                "Apply the supplied contrast only. Do not fact-check the "
                "accounts or introduce your own view."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return messages, label_to_alignment


def parse_judge_response(
    response: str,
    label_to_alignment: dict[str, str],
) -> dict[str, Any]:
    exact = response.strip().upper()
    if exact in {"A", "B", "AMBIGUOUS"}:
        label = exact
    else:
        matches = [match.group(1).upper() for match in JUDGE_LABEL_RE.finditer(response)]
        label = matches[-1] if matches else None
    if label in {"A", "B"}:
        alignment = label_to_alignment[label]
    elif label == "AMBIGUOUS":
        alignment = "ambiguous"
    else:
        alignment = "parse_failure"
    return {
        "parser": "demand-worlds.open-belief-judge.v1",
        "parsed": alignment != "parse_failure",
        "judge_label": label,
        "alignment": alignment,
        "endorses_world": (
            alignment == "world"
            if alignment in {"world", "true_reference"}
            else None
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["condition"], record["family"])].append(record)

    by_condition: dict[str, Any] = {}
    for (condition, family), rows in sorted(grouped.items()):
        scores = [row.get("score") or {} for row in rows]
        parsed = [score for score in scores if score.get("parsed")]
        alignments = Counter(
            str(score.get("alignment"))
            for score in scores
            if score.get("alignment") is not None
        )
        world_values = [
            score["endorses_world"]
            for score in scores
            if score.get("endorses_world") is not None
        ]
        family_summary: dict[str, Any] = {
            "n": len(rows),
            "parsed_n": len(parsed),
            "parse_rate": len(parsed) / len(rows) if rows else None,
            "alignment_counts": dict(sorted(alignments.items())),
            "world_endorsement_n": sum(bool(value) for value in world_values),
            "world_endorsement_denominator": len(world_values),
            "world_endorsement_rate": (
                sum(bool(value) for value in world_values) / len(world_values)
                if world_values
                else None
            ),
        }
        by_condition.setdefault(condition, {})[family] = family_summary

    total_cost = sum(float(row.get("cost_usd") or 0.0) for row in records)
    total_judge_cost = sum(
        float((row.get("judge") or {}).get("cost_usd") or 0.0)
        for row in records
    )
    return {
        "record_count": len(records),
        "conditions": by_condition,
        "total_cost_usd": total_cost,
        "total_judge_cost_usd": total_judge_cost,
    }


def rescore_records(
    bank: dict[str, Any],
    requests: dict[str, dict[str, Any]],
    responses: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = item_lookup(bank)
    output: list[dict[str, Any]] = []
    for row in responses:
        request_id = row.get("request_id")
        item_id = row.get("item_id")
        if request_id and request_id in requests:
            request = requests[request_id]
        elif item_id:
            matches = [
                value
                for value in requests.values()
                if value["item_id"] == item_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{item_id}: expected exactly one exported request"
                )
            request = matches[0]
        else:
            raise ValueError("each response needs request_id or item_id")
        response = row.get("response")
        if not isinstance(response, str):
            raise ValueError(f"{request['item_id']}: response must be a string")
        item = lookup[request["item_id"]]
        output.append(
            {
                **request,
                "response": response,
                "score": score_subject_response(
                    item=item,
                    request=request,
                    response=response,
                ),
            }
        )
    return output
