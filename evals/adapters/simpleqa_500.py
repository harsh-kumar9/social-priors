"""Frozen 500-item SimpleQA task using the official release and paper scorer."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from inspect_evals.metadata import load_eval_metadata
from inspect_evals.simpleqa.dataset import (
    SIMPLEQA_DATASET_SHA256,
    SIMPLEQA_DATASET_URL,
)
from inspect_evals.simpleqa.scorer import simpleqa_scorer
from inspect_evals.utils.download import download_and_verify

from evals.registry import REPO, RegistryError


SUBSET_SEED = 20260729
SUBSET_SIZE = 500
SOURCE_ROWS = 4326
SOURCE_CACHE = REPO / ".cache/evals/sources/simpleqa/simple_qa_test_set.csv"
PANEL_PATH = REPO / "evals/pilots/p1_simpleqa/PANEL.yaml"
INDEX_LIST_SHA256 = (
    "9d01856a7bf29e91d065307703c623dc355465c1afa84d8eb00be19d4678c972"
)
ITEM_ID_LIST_SHA256 = (
    "e79b644734e3c65eb2300b07cba3cd9ce14ff58d7f01818c0f07d652aeda9c14"
)
EVAL_VERSION = load_eval_metadata("simpleqa").version


def _canonical_row(row: dict[str, str]) -> str:
    return json.dumps(
        {
            "answer": row["answer"],
            "metadata": row["metadata"],
            "problem": row["problem"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _row_hash(row: dict[str, str]) -> str:
    return hashlib.sha256(_canonical_row(row).encode("utf-8")).hexdigest()


def _rank_hash(index: int, row: dict[str, str]) -> str:
    prefix = f"{SUBSET_SEED}:{index}:".encode("utf-8")
    return hashlib.sha256(prefix + _canonical_row(row).encode("utf-8")).hexdigest()


def _list_hash(values: list[Any]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_rows() -> list[dict[str, str]]:
    SOURCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    download_and_verify(
        SIMPLEQA_DATASET_URL,
        SIMPLEQA_DATASET_SHA256,
        SOURCE_CACHE,
    )
    with SOURCE_CACHE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("metadata", "problem", "answer"):
            raise RegistryError("SimpleQA source columns changed")
        rows = list(reader)
    if len(rows) != SOURCE_ROWS:
        raise RegistryError(
            f"SimpleQA source row count changed: {len(rows)} != {SOURCE_ROWS}"
        )
    return rows


def _panel_indices() -> list[int]:
    panel = yaml.safe_load(PANEL_PATH.read_text(encoding="utf-8"))
    if not isinstance(panel, dict):
        raise RegistryError("SimpleQA panel must be an object")
    if panel.get("seed") != SUBSET_SEED or panel.get("count") != SUBSET_SIZE:
        raise RegistryError("SimpleQA panel seed/count changed")
    indices = panel.get("source_indices")
    if (
        not isinstance(indices, list)
        or len(indices) != SUBSET_SIZE
        or len(set(indices)) != SUBSET_SIZE
        or any(not isinstance(index, int) or not 0 <= index < SOURCE_ROWS for index in indices)
    ):
        raise RegistryError("SimpleQA panel indices are invalid")
    if _list_hash(indices) != INDEX_LIST_SHA256:
        raise RegistryError("SimpleQA panel index-list hash changed")
    return indices


def load_frozen_samples() -> list[Sample]:
    """Compile the frozen subset and verify it remains the seeded SHA-rank result."""

    rows = _read_rows()
    indices = _panel_indices()
    expected = sorted(
        range(len(rows)),
        key=lambda index: (_rank_hash(index, rows[index]), index),
    )[:SUBSET_SIZE]
    if indices != expected:
        raise RegistryError("SimpleQA panel no longer matches the frozen rank rule")

    samples: list[Sample] = []
    item_ids: list[str] = []
    for index in indices:
        row = rows[index]
        content_hash = _row_hash(row)
        item_id = f"simpleqa-{index:04d}-{content_hash[:12]}"
        try:
            metadata = ast.literal_eval(row["metadata"])
        except (SyntaxError, ValueError) as exc:
            raise RegistryError(f"SimpleQA row {index} metadata is invalid") from exc
        if not isinstance(metadata, dict):
            raise RegistryError(f"SimpleQA row {index} metadata is not an object")
        samples.append(
            Sample(
                id=item_id,
                input=row["problem"],
                target=row["answer"],
                metadata={
                    **metadata,
                    "source_index": index,
                    "source_row_sha256": content_hash,
                    "subset_rank_sha256": _rank_hash(index, row),
                },
            )
        )
        item_ids.append(item_id)
    if _list_hash(item_ids) != ITEM_ID_LIST_SHA256:
        raise RegistryError("SimpleQA frozen item identities changed")
    return samples


@task
def simpleqa_500() -> Task:
    """Administer the frozen subset with the original paper-style A/B/C grader."""

    return Task(
        dataset=MemoryDataset(
            samples=load_frozen_samples(),
            name="simpleqa_500",
            location=str(SOURCE_CACHE),
        ),
        solver=[generate()],
        scorer=simpleqa_scorer(grade_parse="paper"),
        config=GenerateConfig(temperature=0.5, max_tokens=2048),
        epochs=1,
        version=EVAL_VERSION.comparability_version,
        metadata={
            **EVAL_VERSION.to_metadata(),
            "subset_seed": SUBSET_SEED,
            "subset_size": SUBSET_SIZE,
            "subset_rule": "lowest_sha256(seed:zero_based_source_index:canonical_row)",
            "index_list_sha256": INDEX_LIST_SHA256,
            "item_id_list_sha256": ITEM_ID_LIST_SHA256,
            "scorer": "original",
            "grade_parse": "paper",
        },
    )
