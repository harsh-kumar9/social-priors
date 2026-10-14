"""Run pinned Inspect tasks and normalize their logs to the common contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import certifi

from evals.contracts import (
    canonical_json,
    read_jsonl,
    sha256_file,
    sha256_json,
    validate_record,
    write_jsonl,
)
from evals.registry import EvalRegistry, REPO, TargetRegistry
from evals.runner.engine import safe_component, utc_now

INSPECT_EVALS_COMMIT = "596a0ff9a197ad0a8a79a5ec96be1230dfa0c493"
INSPECT_AI_COMMIT = "e2dab6dabc98923ca5ee6606bba6fe52655ec185"


class InspectHarnessError(RuntimeError):
    pass


def _format_task_arg(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        return canonical_json(value)
    if isinstance(value, list):
        return canonical_json(value)
    return str(value)


def _selection_spec(
    *,
    run_seed: int,
    limit: int | None,
    sample_ids: list[str] | None,
) -> dict[str, Any]:
    if limit is not None and sample_ids:
        raise ValueError("--limit and --sample-id are mutually exclusive")
    if sample_ids:
        normalized = [str(sample_id).strip() for sample_id in sample_ids]
        if any(not sample_id for sample_id in normalized):
            raise ValueError("sample ids must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("sample ids must be unique")
        return {
            "mode": "exact_sample_ids",
            "sample_ids": normalized,
            "selection_hash": sha256_json(normalized),
        }
    if limit is not None:
        return {
            "mode": "seeded_limit",
            "limit": limit,
            "shuffle_seed": run_seed,
            "selection_hash": sha256_json(
                {"limit": limit, "shuffle_seed": run_seed}
            ),
        }
    return {
        "mode": "full_dataset",
        "selection_hash": sha256_json({"mode": "full_dataset"}),
    }


def _config_hash(
    *,
    instrument_id: str,
    instrument: dict[str, Any],
    target_id: str,
    target: dict[str, Any],
    judge: dict[str, Any] | None = None,
) -> str:
    implementation_files = [
        REPO / "evals/runner/inspect_harness.py",
        REPO / "evals/contracts.py",
        REPO / "evals/registry.py",
        REPO / "evals/schema/response_record.schema.json",
    ]
    task_path = str((instrument.get("harness") or {}).get("task") or "")
    local_task_path = REPO / task_path.partition("@")[0]
    if task_path.partition("@")[0].endswith(".py") and local_task_path.is_file():
        implementation_files.append(local_task_path)
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
            "runner",
            "adapter",
            "source_revision",
            "harness",
            "decoding",
        )
    }
    config = {
            "record_schema": "demand-worlds.response.v1",
            "implementation_revision": implementation_revision,
            "inspect_ai_commit": INSPECT_AI_COMMIT,
            "inspect_evals_commit": INSPECT_EVALS_COMMIT,
            "instrument_id": instrument_id,
            # Registry workflow fields such as status and notes do not change
            # outputs and therefore must not invalidate scientific records.
            "instrument": instrument_config,
            "target_id": target_id,
            "target": {
                key: target.get(key)
                for key in (
                    "model",
                    "inspect_model",
                    "inspect_model_args",
                    "catalog",
                    "decoding",
                )
            },
        }
    if judge is not None:
        config["judge"] = judge
    return sha256_json(config, prefix="cfg:")


def _model_input(sample: dict[str, Any]) -> list[dict[str, Any]]:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return [{"role": "user", "content": sample.get("input")}]
    model_input = []
    for message in messages:
        if not isinstance(message, dict) or message.get("source") == "generate":
            continue
        # Inspect assigns fresh internal IDs and a provenance-only ``source`` on
        # every process. Neither is part of the provider request, so including
        # them would make identical prompts hash differently across targets.
        model_input.append(
            {
                key: value
                for key, value in message.items()
                if key not in {"id", "source"}
            }
        )
    return model_input


def _parse_failure(
    sample: dict[str, Any],
    *,
    parse_rule: str,
    primary_scorer: str | None,
) -> bool:
    if sample.get("error"):
        return True
    scores = sample.get("scores")
    if not isinstance(scores, dict) or not scores:
        return True
    if parse_rule == "no_sample_error":
        return False
    score = scores.get(primary_scorer) if primary_scorer else next(iter(scores.values()))
    if not isinstance(score, dict):
        return True
    if parse_rule == "cpi_required":
        explanation = str(score.get("explanation") or "").lower()
        return (
            score.get("value") not in {"C", "P", "I"}
            or "grade not found" in explanation
        )
    if parse_rule == "answer_required":
        answer = score.get("answer")
        return not isinstance(answer, str) or not answer.strip()
    raise InspectHarnessError(f"unsupported Inspect parse_rule: {parse_rule}")


def _sample_correct(
    sample: dict[str, Any],
    *,
    primary_scorer: str | None,
) -> bool | None:
    scores = sample.get("scores")
    if not isinstance(scores, dict) or not scores:
        return None
    score = scores.get(primary_scorer) if primary_scorer else next(iter(scores.values()))
    if not isinstance(score, dict):
        return None
    value = score.get("value")
    if not isinstance(value, (str, bool, int, float)):
        return None
    if value in ("C", "correct", True, 1, 1.0):
        return True
    if value in ("I", "incorrect", False, 0, 0.0):
        return False
    return None


def normalize_inspect_log(
    *,
    raw_log: dict[str, Any],
    instrument_id: str,
    instrument: dict[str, Any],
    target_id: str,
    target: dict[str, Any],
    run_seed: int,
    config_hash: str,
) -> list[dict[str, Any]]:
    if raw_log.get("status") != "success":
        raise InspectHarnessError(
            f"Inspect log status is {raw_log.get('status')!r}, not success"
        )
    eval_metadata = raw_log.get("eval") or {}
    samples = raw_log.get("samples")
    if not isinstance(samples, list):
        raise InspectHarnessError("Inspect log does not contain samples")

    harness = instrument["harness"]
    primary_scorer = harness.get("primary_scorer")
    parse_rule = harness.get("parse_rule", "no_sample_error")
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise InspectHarnessError(f"Inspect sample {index} is not an object")
        sample_id = str(sample.get("id") or f"sample-{index:06d}")
        model_input = _model_input(sample)
        output = sample.get("output") or {}
        choices = output.get("choices") if isinstance(output, dict) else None
        first_choice = (
            choices[0] if isinstance(choices, list) and choices else {}
        )
        official_correct = (
            None
            if harness.get("accuracy_semantics") == "none"
            else _sample_correct(sample, primary_scorer=primary_scorer)
        )
        parse_failure = _parse_failure(
            sample,
            parse_rule=parse_rule,
            primary_scorer=primary_scorer,
        )
        verdict = {
            "scored_by": "official_inspect_scorer",
            "official_scores": sample.get("scores"),
            "official_target": sample.get("target"),
            "official_choices": sample.get("choices"),
            "official_metadata": sample.get("metadata"),
            "official_correct": official_correct,
            "parse_failure": parse_failure,
            "stop_reason": (
                first_choice.get("stop_reason")
                if isinstance(first_choice, dict)
                else None
            ),
            "sample_error": sample.get("error"),
            "harness": {
                "name": "inspect",
                "task": eval_metadata.get("task"),
                "task_version": eval_metadata.get("task_version"),
                "task_metadata": eval_metadata.get("metadata"),
                "inspect_ai_commit": INSPECT_AI_COMMIT,
                "inspect_evals_commit": INSPECT_EVALS_COMMIT,
            },
        }
        record = {
            "target_id": target_id,
            "run_seed": run_seed,
            "instrument": instrument_id,
            "tier": instrument["tier"],
            "item_id": sample_id,
            "permutation_id": sha256_json(
                {
                    "sample_id": sample_id,
                    "epoch": sample.get("epoch", 1),
                    "choices": sample.get("choices"),
                }
            ),
            "prompt_hash": sha256_json(model_input),
            "decoding": {
                "model": target["model"],
                "inspect_model": eval_metadata.get("model"),
                "model_args": eval_metadata.get("model_args"),
                "model_generate_config": eval_metadata.get(
                    "model_generate_config"
                ),
                "task_args": eval_metadata.get("task_args"),
                "model_roles": eval_metadata.get("model_roles"),
            },
            "response": (
                output.get("completion")
                if isinstance(output, dict)
                else None
            ),
            "logprobs": (
                first_choice.get("logprobs")
                if isinstance(first_choice, dict)
                else None
            ),
            "judge": {
                "schema_version": "inspect-log.v1",
                "verdict": verdict,
            },
            "source_revision": instrument["source_revision"],
            "config_hash": config_hash,
            "created_at": sample.get("started_at") or eval_metadata.get("created") or utc_now(),
        }
        validate_record(record)
        records.append(record)
    return records


def _token_cost(
    target: dict[str, Any],
    stats: dict[str, Any],
    judge: dict[str, Any] | None = None,
) -> tuple[float, int, int, dict[str, dict[str, Any]]]:
    catalogs = {
        str(target["inspect_model"]): {
            "role": "subject",
            "catalog": target.get("catalog") or {},
        }
    }
    if judge is not None:
        catalogs[str(judge["inspect_model"])] = {
            "role": str(judge["role"]),
            "catalog": judge.get("catalog") or {},
        }

    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    usage_by_model: dict[str, dict[str, Any]] = {}
    for model_name, raw_row in (stats.get("model_usage") or {}).items():
        row = raw_row if isinstance(raw_row, dict) else {}
        input_tokens = int(row.get("input_tokens") or 0)
        cache_read_tokens = int(row.get("input_tokens_cache_read") or 0)
        cache_write_tokens = int(row.get("input_tokens_cache_write") or 0)
        output_tokens = int(row.get("output_tokens") or 0)
        prompt_tokens += input_tokens + cache_read_tokens + cache_write_tokens
        completion_tokens += output_tokens
        pricing = catalogs.get(str(model_name))
        if pricing is None and (
            input_tokens or cache_read_tokens or cache_write_tokens or output_tokens
        ):
            raise InspectHarnessError(
                f"no pinned price for model usage entry {model_name!r}"
            )
        catalog = (pricing or {}).get("catalog") or {}
        prompt_price = float(catalog.get("prompt_price_per_token") or 0)
        model_cost = (
            input_tokens * prompt_price
            + cache_read_tokens
            * float(
                catalog.get("input_cache_read_price_per_token", prompt_price)
            )
            + cache_write_tokens
            * float(
                catalog.get("input_cache_write_price_per_token", prompt_price)
            )
            + output_tokens * float(catalog.get("completion_price_per_token") or 0)
        )
        cost += model_cost
        usage_by_model[str(model_name)] = {
            "role": (pricing or {}).get("role"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
            "input_tokens_cache_read": cache_read_tokens,
            "input_tokens_cache_write": cache_write_tokens,
            "estimated_cost_usd": round(model_cost, 9),
        }
    return cost, prompt_tokens, completion_tokens, usage_by_model


def run_inspect_evaluation(
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
    sample_ids: list[str] | None = None,
    resume: bool = True,
    overwrite: bool = False,
    cost_cap_usd: float | None = 10.0,
) -> dict[str, Any]:
    if num_shards != 1 or shard_index != 0:
        raise InspectHarnessError(
            "Inspect sharding is not implemented yet; use one shard for P1 smoke runs"
        )
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    selection = _selection_spec(
        run_seed=run_seed,
        limit=limit,
        sample_ids=sample_ids,
    )

    registry = EvalRegistry(registry_path)
    targets = TargetRegistry(targets_path)
    instrument = registry.instrument(instrument_id).data
    target = targets.target(target_id).data
    judge_id = (instrument.get("harness") or {}).get("judge")
    judge = targets.judge(str(judge_id)).data if judge_id else None
    if instrument.get("runner") != "inspect":
        raise InspectHarnessError(f"{instrument_id} is not an Inspect instrument")
    inspect_model = target.get("inspect_model")
    if not inspect_model:
        raise InspectHarnessError(f"{target_id} has no inspect_model")

    config_hash = _config_hash(
        instrument_id=instrument_id,
        instrument=instrument,
        target_id=target_id,
        target=target,
        judge=judge,
    )
    result_dir = (
        results_root / safe_component(target_id) / safe_component(instrument_id)
    )
    records_path = result_dir / f"seed-{run_seed}.jsonl"
    manifest_path = result_dir / f"seed-{run_seed}.manifest.json"
    existing = read_jsonl(records_path)
    if existing and not overwrite:
        if not resume:
            raise FileExistsError(
                f"{records_path} exists; use resume or explicit overwrite"
            )
        if any(record["config_hash"] != config_hash for record in existing):
            raise InspectHarnessError(
                f"resume config mismatch for {records_path}"
            )
        if manifest_path.exists():
            summary = json.loads(manifest_path.read_text(encoding="utf-8"))
            if summary.get("selection") != selection:
                raise InspectHarnessError(
                    f"resume selection mismatch for {manifest_path}"
                )
            summary["resume_noop"] = True
            return summary

    raw_dir = (
        cache_root
        / "inspect_logs"
        / safe_component(target_id)
        / safe_component(instrument_id)
        / f"seed-{run_seed}-{uuid.uuid4().hex}"
    )
    raw_dir.mkdir(parents=True, exist_ok=False)
    executable = Path(sys.executable).parent / "inspect"
    harness = instrument["harness"]
    command = [
        str(executable),
        "eval",
        str(harness["task"]),
        "--model",
        str(inspect_model),
        "--epochs",
        str(harness.get("epochs", 1)),
        "--seed",
        str(run_seed % (2**31 - 1)),
        "--max-connections",
        str(harness.get("max_connections", 4)),
        "--max-retries",
        "8",
        "--timeout",
        "180",
        "--cache",
        "365D",
        "--log-format",
        "json",
        "--no-log-realtime",
        "--log-dir",
        str(raw_dir),
        "--display",
        "none",
        "--metadata",
        f"demand_worlds_config_hash={config_hash}",
    ]
    if sample_ids:
        command.extend(["--sample-id", ",".join(sample_ids)])
    elif limit is not None:
        command.extend(["--limit", str(limit), "--sample-shuffle", str(run_seed)])
    decoding = {
        **(target.get("decoding") or {}),
        **(instrument.get("decoding") or {}),
    }
    if decoding.get("temperature") is not None:
        command.extend(["--temperature", str(decoding["temperature"])])
    if decoding.get("top_p") is not None:
        command.extend(["--top-p", str(decoding["top_p"])])
    if decoding.get("max_tokens") is not None:
        command.extend(["--max-tokens", str(decoding["max_tokens"])])
    if judge is not None:
        role_config = {
            "model": judge["inspect_model"],
            **(judge.get("generate_config") or {}),
        }
        if judge.get("inspect_model_args"):
            role_config["model_args"] = judge["inspect_model_args"]
        command.extend(
            [
                "--model-role",
                f"{judge['role']}={canonical_json(role_config)}",
            ]
        )
    for key, value in (harness.get("task_args") or {}).items():
        command.extend(["-T", f"{key}={_format_task_arg(value)}"])
    for key, value in (target.get("inspect_model_args") or {}).items():
        command.extend(["-M", f"{key}={_format_task_arg(value)}"])

    environment = os.environ.copy()
    environment["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    environment["XDG_DATA_HOME"] = str(cache_root / "xdg-data")
    environment["HF_HOME"] = str(cache_root / "huggingface")
    environment["INSPECT_TRACE_FILE"] = str(raw_dir / "trace.log")
    # Some macOS Python distributions do not inherit the system trust store.
    # Pinning certifi here makes Hugging Face and fsspec downloads reproducible
    # without disabling TLS verification.
    environment.setdefault("SSL_CERT_FILE", certifi.where())
    environment.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        outputs = [
            text.strip()
            for text in (completed.stderr, completed.stdout)
            if text and text.strip()
        ]
        detail = "\n".join(outputs)[-4000:] or (
            "Inspect emitted no console diagnostics; inspect "
            f"{raw_dir} for trace/log artifacts"
        )
        raise InspectHarnessError(
            f"Inspect failed with exit {completed.returncode}: {detail}"
        )
    raw_logs = sorted(raw_dir.glob("*.json"))
    if len(raw_logs) != 1:
        raise InspectHarnessError(
            f"expected one Inspect JSON log in {raw_dir}, found {len(raw_logs)}"
        )
    raw_log_path = raw_logs[0]
    raw_log = json.loads(raw_log_path.read_text(encoding="utf-8"))
    records = normalize_inspect_log(
        raw_log=raw_log,
        instrument_id=instrument_id,
        instrument=instrument,
        target_id=target_id,
        target=target,
        run_seed=run_seed,
        config_hash=config_hash,
    )
    if sample_ids and {record["item_id"] for record in records} != set(sample_ids):
        returned_ids = sorted(record["item_id"] for record in records)
        raise InspectHarnessError(
            "Inspect exact sample selection mismatch: requested "
            f"{sorted(sample_ids)}, returned {returned_ids}"
        )

    stats = raw_log.get("stats") or {}
    (
        estimated_cost,
        prompt_tokens,
        completion_tokens,
        model_usage,
    ) = _token_cost(target, stats, judge)
    if cost_cap_usd is not None and estimated_cost > cost_cap_usd:
        raise InspectHarnessError(
            f"estimated cost cap exceeded: ${estimated_cost:.6f} > "
            f"${cost_cap_usd:.2f}; normalized records were not written"
        )
    verdicts = [record["judge"]["verdict"] for record in records]
    parse_failures = sum(bool(verdict["parse_failure"]) for verdict in verdicts)
    scored = [
        verdict["official_correct"]
        for verdict in verdicts
        if verdict["official_correct"] is not None
    ]
    started = stats.get("started_at")
    completed_at = stats.get("completed_at")
    latency_ms = None
    if isinstance(started, str) and isinstance(completed_at, str):
        from datetime import datetime

        latency_ms = round(
            (
                datetime.fromisoformat(completed_at)
                - datetime.fromisoformat(started)
            ).total_seconds()
            * 1000
        )

    write_jsonl(records_path, records)
    summary = {
        "schema_version": "demand-worlds.run-manifest.v1",
        "target_id": target_id,
        "model": target["model"],
        "instrument": instrument_id,
        "tier": instrument["tier"],
        "source_revision": instrument["source_revision"],
        "config_hash": config_hash,
        "run_seed": run_seed,
        "selection": selection,
        "n_records": len(records),
        "n_new_records": len(records),
        "accuracy": (
            sum(bool(value) for value in scored) / len(scored)
            if scored
            else None
        ),
        "parse_failure_rate": (
            parse_failures / len(verdicts) if verdicts else None
        ),
        "official_results": raw_log.get("results"),
        "raw_harness_log": {
            "path": os.path.relpath(raw_log_path, REPO),
            "sha256": sha256_file(raw_log_path),
            "format": "inspect-json-v2",
        },
        "telemetry_new": {
            "estimated_cost_usd": round(estimated_cost, 9),
            "cost_cap_usd": cost_cap_usd,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_usage": model_usage,
        },
        "created_at": utc_now(),
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary
