"""Evaluation execution, resume semantics, records, and manifests."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals import SCHEMA_VERSION
from evals.adapters.base import EvalItem
from evals.contracts import (
    canonical_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    validate_record,
    write_jsonl,
)
from evals.registry import EvalRegistry, REPO, TargetRegistry
from evals.runner.cache import ResponseCache
from evals.runner.client import client_from_target
from evals.runner.permutation import (
    label_logprobs,
    normalized_probabilities,
    permute_mcq,
    response_label,
)


class CostCapError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_shard(item_id: str, num_shards: int) -> int:
    digest = sha256_json({"item_id": item_id}).removeprefix("sha256:")
    return int(digest[:16], 16) % num_shards


def safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe path component: {value!r}")
    return value


def _item_seed(run_seed: int, item_id: str) -> int:
    digest = sha256_json({"run_seed": run_seed, "item_id": item_id})
    # Use the portable positive signed-int range accepted by OpenAI-compatible
    # providers; some reject otherwise valid unsigned 32-bit values.
    return int(digest.removeprefix("sha256:")[:8], 16) % (2**31 - 1)


def _config_hash(
    *,
    instrument_id: str,
    instrument: dict[str, Any],
    target_id: str,
    target: dict[str, Any],
) -> str:
    adapter_module = instrument["adapter"].partition(":")[0]
    implementation_files = sorted((REPO / "evals/runner").glob("*.py")) + [
        REPO / "evals/__init__.py",
        REPO / "evals/contracts.py",
        REPO / "evals/registry.py",
        REPO / "evals/adapters/base.py",
        REPO / f"{adapter_module.replace('.', '/')}.py",
        REPO / "evals/schema/response_record.schema.json",
    ]
    implementation_revision = sha256_json(
        {
            str(path.relative_to(REPO)): sha256_file(path)
            for path in implementation_files
        }
    )
    instrument_config = {
        key: instrument.get(key)
        for key in (
            "tier",
            "adapter",
            "items_path",
            "source_revision",
            "decoding",
        )
    }
    target_config = {
        key: target.get(key)
        for key in (
            "provider",
            "base_url",
            "model",
            "decoding",
            "closed_form_top_logprobs",
            "system_prompt",
        )
    }
    return sha256_json(
        {
            "record_schema": SCHEMA_VERSION,
            "implementation_revision": implementation_revision,
            "instrument_id": instrument_id,
            "instrument": instrument_config,
            "target_id": target_id,
            "target": target_config,
        },
        prefix="cfg:",
    )[:68]


def _messages(system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _select_items(
    items: list[EvalItem],
    *,
    num_shards: int,
    shard_index: int,
    limit: int | None,
) -> list[EvalItem]:
    selected = [
        item for item in items if stable_shard(item.item_id, num_shards) == shard_index
    ]
    selected.sort(key=lambda item: item.item_id)
    return selected[:limit] if limit is not None else selected


def _closed_form_verdict(
    *,
    raw_logprobs: dict[str, Any] | None,
    response: str,
    response_metadata: dict[str, Any],
    permutation: Any,
) -> dict[str, Any]:
    candidate_logprobs = label_logprobs(raw_logprobs, permutation.labels)
    coverage = len(candidate_logprobs) / len(permutation.labels)
    probability_valid = coverage == 1.0
    probabilities = (
        normalized_probabilities(candidate_logprobs)
        if probability_valid
        else {}
    )
    parsed_label = response_label(response, permutation.labels)
    chosen = (
        max(candidate_logprobs, key=candidate_logprobs.get)
        if probability_valid
        else parsed_label
    )
    return {
        "scored_by": (
            "deterministic_first_token_logprobs.v1"
            if probability_valid
            else "parsed_response_fallback.v1"
        ),
        "chosen_label": chosen,
        "response_label": parsed_label,
        "correct_label": permutation.correct_label,
        "correct": bool(chosen == permutation.correct_label),
        "parse_failure": chosen is None,
        "label_logprobs": candidate_logprobs,
        "label_probabilities": probabilities,
        "option_logprob_coverage": coverage,
        "probability_valid": probability_valid,
        "missing_option_labels": sorted(
            set(permutation.labels) - set(candidate_logprobs)
        ),
        "label_to_original_index": permutation.label_to_original_index,
        "response_metadata": response_metadata,
    }


def run_evaluation(
    *,
    instrument_id: str,
    target_id: str,
    registry_path: Path,
    targets_path: Path,
    results_root: Path,
    cache_root: Path,
    run_seed: int,
    num_shards: int = 1,
    shard_index: int = 0,
    limit: int | None = None,
    resume: bool = True,
    overwrite: bool = False,
    cost_cap_usd: float | None = 10.0,
) -> dict[str, Any]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    if cost_cap_usd is not None and cost_cap_usd <= 0:
        raise ValueError("cost_cap_usd must be positive or null")

    registry = EvalRegistry(registry_path)
    targets = TargetRegistry(targets_path)
    instrument_spec = registry.instrument(instrument_id)
    target_spec = targets.target(target_id)
    adapter = instrument_spec.load_adapter()
    items = _select_items(
        adapter.load_items(),
        num_shards=num_shards,
        shard_index=shard_index,
        limit=limit,
    )
    if not items:
        raise ValueError("shard/limit selection produced zero items")

    config_hash = _config_hash(
        instrument_id=instrument_id,
        instrument=instrument_spec.data,
        target_id=target_id,
        target=target_spec.data,
    )
    result_directory = (
        results_root
        / safe_component(target_id)
        / safe_component(instrument_id)
    )
    shard_suffix = f"-shard-{shard_index:03d}-of-{num_shards:03d}" if num_shards > 1 else ""
    records_path = result_directory / f"seed-{run_seed}{shard_suffix}.jsonl"
    manifest_path = result_directory / f"seed-{run_seed}{shard_suffix}.manifest.json"

    existing = read_jsonl(records_path)
    if existing and overwrite:
        existing = []
    elif existing and not resume:
        raise FileExistsError(
            f"{records_path} exists; use resume or explicit overwrite"
        )
    for record in existing:
        if record["config_hash"] != config_hash:
            raise ValueError(
                f"resume config mismatch for {records_path}: "
                f"{record['config_hash']} != {config_hash}"
            )

    completed = {(row["item_id"], row["permutation_id"]) for row in existing}
    pending: list[tuple[EvalItem, Any]] = []
    for item in items:
        if item.kind != "multiple_choice":
            raise NotImplementedError(
                "P0 supports multiple_choice; sampled/judged modes enter at Tier B"
            )
        permutation = permute_mcq(item, run_seed)
        if (item.item_id, permutation.permutation_id) not in completed:
            pending.append((item, permutation))

    if not pending and manifest_path.exists():
        summary = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary["resume_noop"] = True
        return summary

    cache = ResponseCache(cache_root)
    client = client_from_target(target_spec.data, cache)
    new_records: list[dict[str, Any]] = []
    total_cost = 0.0
    total_latency_ms = 0
    origin_cost = 0.0
    origin_latency_ms = 0
    cache_hits = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for item, permutation in pending:
        top_logprobs = int(target_spec.data.get("closed_form_top_logprobs", 5))
        if top_logprobs < 1 or top_logprobs > 20:
            raise ValueError(
                f"{target_id} closed_form_top_logprobs must be in [1, 20]"
            )
        if len(permutation.labels) > top_logprobs:
            raise ValueError(
                f"{item.item_id} has {len(permutation.labels)} options; "
                f"{target_id} requests only {top_logprobs} top logprobs"
            )
        decoding = dict(target_spec.data["decoding"])
        decoding.update(instrument_spec.data.get("decoding", {}))
        decoding.update(
            {
                "temperature": 0,
                # Some OpenRouter providers enforce a minimum completion budget.
                # We score only token 1; the prompt still requests one letter.
                "max_tokens": 16,
                "logprobs": True,
                "top_logprobs": top_logprobs,
                "seed": _item_seed(run_seed, item.item_id),
            }
        )
        messages = _messages(
            target_spec.data.get("system_prompt"),
            permutation.prompt,
        )
        completion = client.complete(
            model=target_spec.model,
            messages=messages,
            decoding=decoding,
        )
        verdict = _closed_form_verdict(
            raw_logprobs=completion.logprobs,
            response=completion.response,
            response_metadata=completion.response_metadata,
            permutation=permutation,
        )
        record = {
            "target_id": target_id,
            "run_seed": run_seed,
            "instrument": instrument_id,
            "tier": instrument_spec.tier,
            "item_id": item.item_id,
            "permutation_id": permutation.permutation_id,
            "prompt_hash": sha256_json(messages),
            "decoding": {"model": target_spec.model, **decoding},
            "response": completion.response,
            "logprobs": completion.logprobs,
            "judge": {
                "schema_version": "closed-form-logprob.v1",
                "verdict": verdict,
            },
            "source_revision": instrument_spec.source_revision,
            "config_hash": config_hash,
            "created_at": utc_now(),
        }
        validate_record(record)
        new_records.append(record)
        total_cost += completion.cost_usd
        if cost_cap_usd is not None and total_cost > cost_cap_usd:
            raise CostCapError(
                f"cost cap exceeded: ${total_cost:.6f} > ${cost_cap_usd:.2f}; "
                "batch aborted"
            )
        total_latency_ms += completion.latency_ms
        origin_cost += completion.origin_cost_usd
        origin_latency_ms += completion.origin_latency_ms
        cache_hits += int(completion.cached)
        for key in usage:
            usage[key] += int(completion.usage.get(key, 0) or 0)

    all_records = existing + new_records
    write_jsonl(records_path, all_records)
    verdicts = [row["judge"]["verdict"] for row in all_records if row["judge"]]
    correct = sum(bool(verdict["correct"]) for verdict in verdicts)
    parse_failure_total = sum(
        bool(verdict["parse_failure"]) for verdict in verdicts
    )
    probability_valid_total = sum(
        bool(verdict["probability_valid"]) for verdict in verdicts
    )
    mean_coverage = (
        sum(float(verdict["option_logprob_coverage"]) for verdict in verdicts)
        / len(verdicts)
        if verdicts
        else None
    )
    providers = sorted(
        {
            str(provider)
            for verdict in verdicts
            if (
                provider := verdict.get("response_metadata", {}).get("provider")
            )
        }
    )
    summary = {
        "schema_version": "demand-worlds.run-manifest.v1",
        "target_id": target_id,
        "model": target_spec.model,
        "instrument": instrument_id,
        "tier": instrument_spec.tier,
        "source_revision": instrument_spec.source_revision,
        "config_hash": config_hash,
        "run_seed": run_seed,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "records_path": os.path.relpath(records_path, registry_path.parent.parent),
        "n_records": len(all_records),
        "n_new_records": len(new_records),
        "accuracy": correct / len(verdicts) if verdicts else None,
        "parse_failure_rate": (
            parse_failure_total / len(verdicts) if verdicts else None
        ),
        "probability_valid_rate": (
            probability_valid_total / len(verdicts) if verdicts else None
        ),
        "mean_option_logprob_coverage": mean_coverage,
        "response_fallback_rate": (
            (len(verdicts) - probability_valid_total) / len(verdicts)
            if verdicts
            else None
        ),
        "providers": providers,
        "telemetry_new": {
            "cost_usd": round(total_cost, 9),
            "cost_cap_usd": cost_cap_usd,
            "latency_ms": total_latency_ms,
            "origin_cost_usd": round(origin_cost, 9),
            "origin_latency_ms": origin_latency_ms,
            "cache_hits": cache_hits,
            "network_or_mock_calls": len(new_records) - cache_hits,
            **usage,
        },
        "created_at": utc_now(),
    }
    result_directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary
