"""Common response-record contract and canonical JSON helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from evals import SCHEMA_VERSION

REQUIRED_FIELDS = (
    "target_id",
    "run_seed",
    "instrument",
    "tier",
    "item_id",
    "permutation_id",
    "prompt_hash",
    "decoding",
    "response",
    "logprobs",
    "judge",
    "source_revision",
    "config_hash",
    "created_at",
)


class ContractError(ValueError):
    """Raised when a response record violates the frozen common schema."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any, *, prefix: str = "sha256:") -> str:
    return prefix + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, prefix: str = "sha256:") -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return prefix + digest.hexdigest()


def validate_record(record: dict[str, Any]) -> None:
    """Validate the strict v1 contract without adding a runtime dependency.

    The companion JSON Schema is the interchange definition. This function keeps
    normal runs dependency-light and provides clearer Python-side errors.
    """

    missing = set(REQUIRED_FIELDS) - set(record)
    extra = set(record) - set(REQUIRED_FIELDS)
    if missing:
        raise ContractError(f"missing fields: {sorted(missing)}")
    if extra:
        raise ContractError(f"unknown fields: {sorted(extra)}")

    for name in (
        "target_id",
        "instrument",
        "item_id",
        "permutation_id",
        "source_revision",
        "config_hash",
        "created_at",
    ):
        if not isinstance(record[name], str) or not record[name]:
            raise ContractError(f"{name} must be a non-empty string")
    if not isinstance(record["run_seed"], int) or isinstance(record["run_seed"], bool):
        raise ContractError("run_seed must be an integer")
    if record["tier"] not in {"A", "B", "toy"}:
        raise ContractError("tier must be A, B, or toy")
    if not isinstance(record["prompt_hash"], str) or not record["prompt_hash"].startswith(
        "sha256:"
    ):
        raise ContractError("prompt_hash must be a sha256 identifier")
    if not isinstance(record["config_hash"], str) or not record["config_hash"].startswith(
        "cfg:"
    ):
        raise ContractError("config_hash must be a cfg identifier")
    if not isinstance(record["decoding"], dict):
        raise ContractError("decoding must be an object")
    if record["response"] is not None and not isinstance(record["response"], str):
        raise ContractError("response must be a string or null")
    if record["logprobs"] is not None and not isinstance(record["logprobs"], dict):
        raise ContractError("logprobs must be an object or null")
    if record["response"] is None and record["logprobs"] is None:
        raise ContractError("at least one of response or logprobs must be present")
    if record["judge"] is not None:
        judge = record["judge"]
        if not isinstance(judge, dict) or set(judge) != {"schema_version", "verdict"}:
            raise ContractError("judge must contain exactly schema_version and verdict")
        if not isinstance(judge["schema_version"], str) or not judge["schema_version"]:
            raise ContractError("judge.schema_version must be a non-empty string")

    try:
        timestamp = datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("created_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ContractError("created_at must include a timezone")

    try:
        canonical_json(record)
    except (TypeError, ValueError) as exc:
        raise ContractError("record must be JSON serializable") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record)
            except (json.JSONDecodeError, ContractError) as exc:
                raise ContractError(f"{path}:{line_number}: {exc}") from exc
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            handle.write(canonical_json(record) + "\n")


def schema_descriptor() -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "json_schema": "evals/schema/response_record.schema.json",
    }
