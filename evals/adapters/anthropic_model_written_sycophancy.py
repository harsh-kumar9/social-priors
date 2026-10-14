"""Anthropic's released model-written sycophancy evaluations.

The source questions already contain labelled answer choices. We parse those
semantic choices, remove the original labels, and let the common closed-form
runner apply a deterministic option-order permutation. The behaviour-matching
choice is the scored target; ``correct`` therefore means "matches the user's
stated view", not factual correctness.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from evals.adapters.base import BaseAdapter, EvalItem
from evals.contracts import canonical_json, sha256_file, sha256_json
from evals.registry import InstrumentSpec, REPO, RegistryError


SOURCE_COMMIT = "84fcc677e52e1902d696c32cd1a6b663e70d3993"
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/anthropics/evals/"
    f"{SOURCE_COMMIT}/sycophancy"
)
SOURCE_CACHE = (
    REPO / ".cache/evals/sources/anthropic_model_written_sycophancy"
)
SOURCE_FILES = {
    "nlp_survey": {
        "filename": "sycophancy_on_nlp_survey.jsonl",
        "sha256": "582860b42e2beec806a7d361a08bcce7fdb264e2e697d55104074540352fb308",
        "count": 9984,
    },
    "philpapers2020": {
        "filename": "sycophancy_on_philpapers2020.jsonl",
        "sha256": "2f112b35334fbec0b16dc755df60349fb2b2bf00d4dbaae47175519bee7d37dd",
        "count": 9867,
    },
    "political_typology_quiz": {
        "filename": "sycophancy_on_political_typology_quiz.jsonl",
        "sha256": "691575571f659593ed237aa74ec6530b20ef3a5d0116e5e1f4f189ef530cf032",
        "count": 10200,
    },
}

_CHOICE = re.compile(r"(?m)^ ?\(([A-Z])\) (.+)$")
_ANSWER_TOKEN = re.compile(r"^\s*\(([A-Z])\)\s*$")


def _answer_label(value: Any) -> str:
    match = _ANSWER_TOKEN.fullmatch(str(value))
    if not match:
        raise RegistryError(f"invalid Anthropic answer token: {value!r}")
    return match.group(1)


def _parse_row(
    row: dict[str, Any],
    *,
    dataset: str,
    rank: int,
    row_digest: str,
) -> EvalItem:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise RegistryError(f"{dataset}:{rank}: missing question")
    matches = list(_CHOICE.finditer(question))
    if len(matches) < 2:
        raise RegistryError(f"{dataset}:{rank}: fewer than two choices")
    labels = [match.group(1) for match in matches]
    expected_labels = [
        chr(ord("A") + index) for index in range(len(matches))
    ]
    if labels != expected_labels:
        raise RegistryError(
            f"{dataset}:{rank}: non-contiguous labels {labels}"
        )
    options = tuple(match.group(2).strip() for match in matches)
    if any(not option for option in options):
        raise RegistryError(f"{dataset}:{rank}: empty choice")

    matching_label = _answer_label(row.get("answer_matching_behavior"))
    not_matching_raw = row.get("answer_not_matching_behavior")
    not_matching_values = (
        not_matching_raw
        if isinstance(not_matching_raw, list)
        else [not_matching_raw]
    )
    not_matching_labels = {
        _answer_label(value) for value in not_matching_values
    }
    if matching_label not in labels:
        raise RegistryError(
            f"{dataset}:{rank}: matching label not present"
        )
    if not_matching_labels != set(labels) - {matching_label}:
        raise RegistryError(
            f"{dataset}:{rank}: non-matching labels do not cover all alternatives"
        )

    prompt = question[: matches[0].start()].rstrip()
    prompt = re.sub(r"(?is)\n*choices:\s*$", "", prompt).rstrip()
    if not prompt:
        raise RegistryError(f"{dataset}:{rank}: empty prompt after parsing")
    return EvalItem(
        item_id=(
            f"anthropic_syc_{rank:05d}_{dataset}_{row_digest[:12]}"
        ),
        prompt=prompt,
        kind="multiple_choice",
        options=options,
        correct_index=labels.index(matching_label),
        metadata={
            "dataset": dataset,
            "source_rank": rank,
            "source_row_sha256": f"sha256:{row_digest}",
            "metric_target": "answer_matching_user_view",
            "option_count": len(options),
        },
    )


def _source_path(dataset: str, descriptor: dict[str, Any]) -> Path:
    filename = str(descriptor["filename"])
    path = SOURCE_CACHE / filename
    expected = f"sha256:{descriptor['sha256']}"
    if path.exists() and sha256_file(path) == expected:
        return path

    SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(
        f"{SOURCE_ROOT}/{filename}",
        headers={"User-Agent": "demand-worlds-evals/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            temporary.write_bytes(response.read())
        actual = sha256_file(temporary)
        if actual != expected:
            raise RegistryError(
                f"{dataset} source hash mismatch: expected={expected}, "
                f"actual={actual}"
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


class AnthropicModelWrittenSycophancyAdapter(BaseAdapter):
    def __init__(self, spec: InstrumentSpec):
        self.spec = spec

    def load_items(self) -> list[EvalItem]:
        items: list[EvalItem] = []
        for dataset, descriptor in SOURCE_FILES.items():
            path = _source_path(dataset, descriptor)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            expected_count = int(descriptor["count"])
            if len(rows) != expected_count:
                raise RegistryError(
                    f"{dataset} row count mismatch: "
                    f"expected={expected_count}, actual={len(rows)}"
                )
            ranked_rows = sorted(
                (
                    (
                        sha256_json(row).removeprefix("sha256:"),
                        canonical_json(row),
                        row,
                    )
                    for row in rows
                ),
                key=lambda entry: (entry[0], entry[1]),
            )
            for rank, (digest, _, row) in enumerate(ranked_rows):
                items.append(
                    _parse_row(
                        row,
                        dataset=dataset,
                        rank=rank,
                        row_digest=digest,
                    )
                )
        if len({item.item_id for item in items}) != len(items):
            raise RegistryError("Anthropic sycophancy item IDs are not unique")
        dataset_order = {
            dataset: index for index, dataset in enumerate(SOURCE_FILES)
        }
        items.sort(
            key=lambda item: (
                int(item.metadata["source_rank"]),
                dataset_order[str(item.metadata["dataset"])],
            )
        )
        return items
