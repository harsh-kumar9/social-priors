"""Unified handoff CLI for authored and adopted Demand Worlds evaluations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from evals import SCHEMA_VERSION
from evals.contracts import (
    canonical_json,
    sha256_file,
    sha256_json,
    validate_record,
)
from evals.full_suite import SUITE_ID, SUITE_VERSION
from evals.full_suite.custom import (
    DEFAULT_BANK_PATH,
    authored_bank_hashes,
    parse_salience_judge,
    render_authored_requests,
    render_salience_judge_messages,
    score_custom_response,
    summarize_custom,
    validate_authored_banks,
)
from evals.registry import EvalRegistry, REPO
from evals.runner.cache import ResponseCache
from evals.runner.client import CompletionClient, CompletionError
from evals.runner.dispatch import run_registered_evaluation
from evals.sdf_suite.bank import FAMILIES, item_lookup, load_bank
from evals.sdf_suite.cli import (
    _requests as belief_requests,
    _selected_conditions,
    _selected_families,
    judge_open_ended,
    run_subject,
)
from evals.sdf_suite.render import load_target_config
from evals.sdf_suite.scoring import score_subject_response


PACKAGE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE_ROOT / "manifest.yaml"
REGISTRY_PATH = REPO / "evals/registry.yaml"
TARGETS_PATH = REPO / "evals/targets.yaml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest() -> dict[str, Any]:
    value = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("full-suite manifest must be an object")
    return value


def validate_suite() -> dict[str, Any]:
    manifest = _manifest()
    if manifest.get("suite_id") != SUITE_ID:
        raise ValueError("full-suite manifest ID mismatch")
    if manifest.get("suite_version") != SUITE_VERSION:
        raise ValueError("full-suite manifest version mismatch")
    authored = validate_authored_banks()
    registry = EvalRegistry(REGISTRY_PATH).raw["instruments"]
    registry_tier_a = {
        name for name, spec in registry.items() if spec["tier"] == "A"
    }
    manifest_tier_a = set(manifest["tier_a"])
    if registry_tier_a != manifest_tier_a:
        raise ValueError(
            "Tier-A manifest/registry mismatch: "
            f"missing={sorted(registry_tier_a - manifest_tier_a)}, "
            f"extra={sorted(manifest_tier_a - registry_tier_a)}"
        )
    tier_a_status = {
        instrument: {
            "status": registry[instrument]["status"],
            "runnable": bool(registry[instrument].get("runnable")),
            "runner": registry[instrument].get("runner", "native"),
            "measures": manifest["tier_a"][instrument]["measures"],
        }
        for instrument in manifest["tier_a"]
    }
    return {
        "schema_version": "demand-worlds.full-suite-validation.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "authored": authored,
        "tier_a": tier_a_status,
        "counts": {
            "authored_instruments": len(manifest["authored"]),
            "tier_a_instruments": len(manifest["tier_a"]),
            "tier_a_runnable_now": sum(
                value["runnable"] for value in tier_a_status.values()
            ),
            "tier_a_included_with_explicit_blocker": sum(
                not value["runnable"] for value in tier_a_status.values()
            ),
        },
    }


def suite_plan(*, condition: str) -> dict[str, Any]:
    validation = validate_suite()
    condition_upper = condition.upper()
    selected = _selected_conditions(condition_upper)
    base_mode = condition_upper == "ALL"
    pi99_count = validation["authored"]["counts"]["pi99"]
    authored_counts = {
        "belief_quartet": 160 if base_mode else 40,
        "distributional_allocation": 13,
        "adversarial_distributional": 13,
        "adversarial_generative_distinguish": 10 * len(selected),
        "pi99": pi99_count,
        "yamagishi_trust_anchor": 12,
        "vignette_battery": 36,
        "salience_controls": 24,
    }
    tier_a = {
        instrument: {
            "action": (
                "run"
                if spec["runnable"]
                and spec["runner"] in {"inspect", "native"}
                else "external_manual"
                if spec["runnable"]
                else "skip_with_registry_reason"
            ),
            "status": spec["status"],
            "measures": spec["measures"],
        }
        for instrument, spec in validation["tier_a"].items()
    }
    return {
        "schema_version": "demand-worlds.full-suite-plan.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "condition": condition_upper,
        "authored": {
            "request_counts": authored_counts,
            "total_requests": sum(authored_counts.values()),
            "pi99_source_available": bool(pi99_count),
        },
        "tier_a": tier_a,
        "policy": (
            "Every component remains in the plan. Unavailable adopted "
            "benchmarks emit a skip reason; they are never silently omitted or "
            "replaced with authored look-alikes."
        ),
    }


def export_authored(
    *,
    condition: str,
    run_seed: int,
    target_path: Path | None,
    output: Path,
    include_pi99: bool,
) -> dict[str, Any]:
    bank = load_bank(DEFAULT_BANK_PATH)
    target = load_target_config(target_path) if target_path else {}
    selected_conditions = _selected_conditions(condition)
    belief = belief_requests(
        bank=bank,
        conditions=selected_conditions,
        families=FAMILIES,
        run_seed=run_seed,
        target=target,
    )
    belief = [
        {**request, "instrument": "belief_quartet"} for request in belief
    ]
    custom, skips = render_authored_requests(
        condition=condition,
        run_seed=run_seed,
        target=target,
        include_pi99=include_pi99,
    )
    requests = belief + custom
    _write_jsonl(output, requests)
    manifest = {
        "schema_version": "demand-worlds.full-suite-export.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "condition": condition.upper(),
        "run_seed": run_seed,
        "request_count": len(requests),
        "instrument_counts": dict(
            sorted(
                {
                    instrument: sum(
                        request["instrument"] == instrument
                        for request in requests
                    )
                    for instrument in {
                        request["instrument"] for request in requests
                    }
                }.items()
            )
        ),
        "skips": skips,
        "bank_hashes": authored_bank_hashes(),
        "output": str(output),
    }
    _write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def _client(target: dict[str, Any], cache_root: Path) -> CompletionClient:
    return CompletionClient(
        provider=target["provider"],
        base_url=target["base_url"],
        api_key_env=target.get("api_key_env"),
        cache=ResponseCache(cache_root),
        timeout_s=int(target.get("timeout_s") or 300),
        max_retries=int(target.get("max_retries") or 8),
        prompt_mode=target.get("prompt_mode"),
        tokenizer_id=target.get("tokenizer_id"),
    )


def _custom_source_revision(
    instrument: str,
    hashes: dict[str, str | None],
) -> str:
    key = {
        "distributional_allocation": "distributional_allocation",
        "adversarial_distributional": "distributional_allocation",
        "adversarial_generative_distinguish": "belief_quartet",
        "pi99": "pi99_private",
        "yamagishi_trust_anchor": "yamagishi_trust_anchor",
        "vignette_battery": "vignette_battery",
        "salience_controls": "salience_controls",
    }[instrument]
    digest = hashes[key]
    if digest is None:
        raise ValueError(f"{instrument}: source artifact unavailable")
    return f"sha256:{digest}"


def _record_from_completion(
    *,
    request: dict[str, Any],
    response: str,
    completion: Any,
    target: dict[str, Any],
    config_hash: str,
    source_revision: str,
) -> dict[str, Any]:
    score = score_custom_response(request=request, response=response)
    verdict = {
        **score,
        "request_id": request["request_id"],
        "condition": request.get("condition"),
        "vignette_tier": request.get("vignette_tier"),
        "telemetry": {
            "usage": completion.usage,
            "response_metadata": completion.response_metadata,
            "request_hash": completion.request_hash,
            "cached": completion.cached,
            "cost_usd": completion.cost_usd,
            "origin_cost_usd": completion.origin_cost_usd,
            "latency_ms": completion.latency_ms,
            "origin_latency_ms": completion.origin_latency_ms,
        },
    }
    record = {
        "target_id": target["id"],
        "run_seed": int(request["run_seed"]),
        "instrument": request["instrument"],
        "tier": "B",
        "item_id": request["item_id"],
        "permutation_id": sha256_json(
            {
                "request_id": request["request_id"],
                "presentation": request.get("presentation") or {},
            }
        ),
        "prompt_hash": sha256_json(request["messages"]),
        "decoding": {"model": target["model"], **request["decoding"]},
        "response": response,
        "logprobs": completion.logprobs,
        "judge": {
            "schema_version": str(
                score.get("scored_by")
                or score.get("parser")
                or "demand-worlds.deterministic-score.v1"
            ),
            "verdict": verdict,
        },
        "source_revision": source_revision,
        "config_hash": config_hash,
        "created_at": _now(),
    }
    validate_record(record)
    return record


def _judge_salience(
    *,
    records: list[dict[str, Any]],
    request_by_id: dict[str, dict[str, Any]],
    judge_target: dict[str, Any],
    cache_root: Path,
    cost_cap_usd: float,
) -> list[dict[str, Any]]:
    client = _client(judge_target, cache_root)
    live_cost = 0.0
    output: list[dict[str, Any]] = []
    for record in records:
        if record["instrument"] != "salience_controls":
            output.append(record)
            continue
        verdict = dict(record["judge"]["verdict"])
        if verdict.get("salience_judge_config_sha256") == sha256_json(
            {
                key: value
                for key, value in judge_target.items()
                if key not in {"api_key", "api_key_value"}
            }
        ):
            output.append(record)
            continue
        request = request_by_id[verdict["request_id"]]
        messages = render_salience_judge_messages(
            request, str(record["response"] or "")
        )
        decoding = {
            "temperature": 0,
            "max_tokens": 16,
            **(judge_target.get("decoding") or {}),
        }
        completion = client.complete(
            model=judge_target["model"],
            messages=messages,
            decoding=decoding,
        )
        live_cost += float(completion.cost_usd or 0.0)
        if live_cost > cost_cap_usd:
            raise RuntimeError(
                f"salience judge cost cap exceeded: ${live_cost:.6f}"
            )
        label = parse_salience_judge(completion.response)
        verdict.update(label)
        verdict.update(
            {
                "salience_judge_id": judge_target["id"],
                "salience_judge_model": judge_target["model"],
                "salience_judge_config_sha256": sha256_json(
                    {
                        key: value
                        for key, value in judge_target.items()
                        if key not in {"api_key", "api_key_value"}
                    }
                ),
                "salience_judge_raw_response": completion.response,
                "salience_judge_cost_usd": completion.cost_usd,
            }
        )
        updated = dict(record)
        updated["judge"] = {
            "schema_version": "demand-worlds.salience-judge.v1",
            "verdict": verdict,
        }
        validate_record(updated)
        output.append(updated)
    return output


def run_custom(
    *,
    condition: str,
    run_seed: int,
    target_path: Path,
    output_dir: Path,
    include_pi99: bool,
    cost_cap_usd: float,
    overwrite: bool,
    judge_target_path: Path | None,
    judge_cost_cap_usd: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = load_target_config(target_path)
    requests, skips = render_authored_requests(
        condition=condition,
        run_seed=run_seed,
        target=target,
        include_pi99=include_pi99,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "responses.jsonl"
    existing = [] if overwrite else _read_jsonl(records_path)
    request_by_id = {request["request_id"]: request for request in requests}
    completed = {
        str((row.get("judge") or {}).get("verdict", {}).get("request_id"))
        for row in existing
    }
    hashes = authored_bank_hashes()
    config_hash = sha256_json(
        {
            "record_schema": SCHEMA_VERSION,
            "suite_id": SUITE_ID,
            "suite_version": SUITE_VERSION,
            "target": {
                key: value
                for key, value in target.items()
                if key not in {"api_key", "api_key_value"}
            },
            "bank_hashes": hashes,
            "condition": condition.upper(),
            "run_seed": run_seed,
        },
        prefix="cfg:",
    )
    client = _client(target, output_dir / ".cache")
    records = list(existing)
    live_cost = 0.0
    for request in requests:
        if request["request_id"] in completed:
            continue
        completion = client.complete(
            model=target["model"],
            messages=request["messages"],
            decoding=request["decoding"],
        )
        live_cost += float(completion.cost_usd or 0.0)
        if live_cost > cost_cap_usd:
            raise RuntimeError(
                f"authored-suite cost cap exceeded: ${live_cost:.6f}"
            )
        records.append(
            _record_from_completion(
                request=request,
                response=completion.response,
                completion=completion,
                target=target,
                config_hash=config_hash,
                source_revision=_custom_source_revision(
                    request["instrument"], hashes
                ),
            )
        )
        _write_jsonl(records_path, records)
    selected_ids = set(request_by_id)
    records = [
        row
        for row in records
        if str(
            (row.get("judge") or {}).get("verdict", {}).get("request_id")
        )
        in selected_ids
    ]
    if judge_target_path:
        judge_target = load_target_config(judge_target_path)
        records = _judge_salience(
            records=records,
            request_by_id=request_by_id,
            judge_target=judge_target,
            cache_root=output_dir / ".judge-cache",
            cost_cap_usd=judge_cost_cap_usd,
        )
        _write_jsonl(output_dir / "judged_responses.jsonl", records)
    summary = {
        "schema_version": "demand-worlds.full-suite-authored-summary.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "target_id": target["id"],
        "model": target["model"],
        "condition": condition.upper(),
        "run_seed": run_seed,
        "expected_requests": len(requests),
        "completed_requests": len(records),
        "complete": len(records) == len(requests),
        "instruments": summarize_custom(records),
        "skips": skips,
        "bank_hashes": hashes,
        "config_hash": config_hash,
        "incremental_subject_cost_usd": live_cost,
        "prompt_storage_policy": (
            "Records retain prompt hashes and fixed item/permutation IDs. "
            "PI-99 prompt text is reconstructed from the authorized private "
            "artifact and is never copied into the run directory."
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    return records, summary


def run_authored(
    *,
    condition: str,
    run_seed: int,
    target_path: Path,
    output_dir: Path,
    include_pi99: bool,
    cost_cap_usd: float,
    overwrite: bool,
    judge_target_path: Path | None,
    judge_cost_cap_usd: float,
) -> dict[str, Any]:
    belief_records, belief_summary = run_subject(
        bank_path=DEFAULT_BANK_PATH,
        target_path=target_path,
        output_dir=output_dir / "belief_quartet",
        condition=condition,
        families=FAMILIES,
        run_seed=run_seed,
        cost_cap_usd=cost_cap_usd,
        limit=None,
        overwrite=overwrite,
    )
    if judge_target_path:
        belief_records, belief_summary = judge_open_ended(
            bank_path=DEFAULT_BANK_PATH,
            records=belief_records,
            judge_target_path=judge_target_path,
            output_path=(
                output_dir / "belief_quartet/judged_responses.jsonl"
            ),
            cost_cap_usd=judge_cost_cap_usd,
        )
    _, custom_summary = run_custom(
        condition=condition,
        run_seed=run_seed,
        target_path=target_path,
        output_dir=output_dir / "other_authored",
        include_pi99=include_pi99,
        cost_cap_usd=cost_cap_usd,
        overwrite=overwrite,
        judge_target_path=judge_target_path,
        judge_cost_cap_usd=judge_cost_cap_usd,
    )
    summary = {
        "schema_version": "demand-worlds.full-suite-authored-run.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "condition": condition.upper(),
        "run_seed": run_seed,
        "belief_quartet": belief_summary,
        "other_authored": custom_summary,
        "complete": bool(
            belief_summary.get("complete", True)
            and custom_summary.get("complete")
        ),
    }
    _write_json(output_dir / "authored_summary.json", summary)
    return summary


def _authored_summary(value: dict[str, Any]) -> dict[str, Any]:
    if "belief_quartet" in value and "other_authored" in value:
        return value
    if isinstance(value.get("authored"), dict):
        nested = value["authored"]
        if "belief_quartet" in nested and "other_authored" in nested:
            return nested
    raise ValueError("summary does not contain an authored full-suite run")


def _safe_delta(target: Any, base: Any) -> float | None:
    if isinstance(target, (int, float)) and isinstance(base, (int, float)):
        return float(target) - float(base)
    return None


def _retained_fraction(
    wrapped_gap: float | None,
    unwrapped_gap: float | None,
) -> float | None:
    if (
        wrapped_gap is None
        or unwrapped_gap is None
        or abs(unwrapped_gap) < 1e-12
    ):
        return None
    return wrapped_gap / unwrapped_gap


def compare_runs(
    *,
    base: dict[str, Any],
    target: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    condition = condition.upper()
    if condition not in "PSNA":
        raise ValueError("comparison condition must be P, S, N, or A")
    base_run = _authored_summary(base)
    target_run = _authored_summary(target)
    base_belief = base_run["belief_quartet"]["conditions"][condition]
    target_belief = target_run["belief_quartet"]["conditions"][condition]
    belief: dict[str, Any] = {}
    for family in FAMILIES:
        base_rate = base_belief[family].get("world_endorsement_rate")
        target_rate = target_belief[family].get("world_endorsement_rate")
        belief[family] = {
            "base_world_endorsement_rate": base_rate,
            "target_world_endorsement_rate": target_rate,
            "target_minus_base": _safe_delta(target_rate, base_rate),
        }

    base_other = base_run["other_authored"]["instruments"]
    target_other = target_run["other_authored"]["instruments"]
    base_distribution = base_other["distributional_allocation"][
        "mean_allocation"
    ][condition]
    target_distribution = target_other["distributional_allocation"][
        "mean_allocation"
    ][condition]
    base_wrapped_distribution = base_other["adversarial_distributional"][
        "mean_allocation"
    ][condition]
    target_wrapped_distribution = target_other[
        "adversarial_distributional"
    ]["mean_allocation"][condition]
    distribution_gap = _safe_delta(target_distribution, base_distribution)
    wrapped_distribution_gap = _safe_delta(
        target_wrapped_distribution, base_wrapped_distribution
    )

    base_unwrapped_generative = base_belief["generative_distinguish"].get(
        "world_endorsement_rate"
    )
    target_unwrapped_generative = target_belief[
        "generative_distinguish"
    ].get("world_endorsement_rate")
    base_wrapped_generative = base_other[
        "adversarial_generative_distinguish"
    ]["by_condition"][condition]["world_endorsement_rate"]
    target_wrapped_generative = target_other[
        "adversarial_generative_distinguish"
    ]["by_condition"][condition]["world_endorsement_rate"]
    generative_gap = _safe_delta(
        target_unwrapped_generative, base_unwrapped_generative
    )
    wrapped_generative_gap = _safe_delta(
        target_wrapped_generative, base_wrapped_generative
    )

    action_deltas = {
        action: _safe_delta(
            target_other["vignette_battery"]["action_rates"].get(action),
            base_other["vignette_battery"]["action_rates"].get(action),
        )
        for action in (
            "direct_assist",
            "bounded_assist",
            "clarify",
            "decline",
        )
    }
    vignette_by_tier = {
        tier: {
            "guardedness_target_minus_base": _safe_delta(
                target_other["vignette_battery"]["by_tier"][tier].get(
                    "guardedness_rate"
                ),
                base_other["vignette_battery"]["by_tier"][tier].get(
                    "guardedness_rate"
                ),
            )
        }
        for tier in "BXR"
    }
    trust = {
        metric: {
            "base": base_other["yamagishi_trust_anchor"].get(metric),
            "target": target_other["yamagishi_trust_anchor"].get(metric),
            "target_minus_base": _safe_delta(
                target_other["yamagishi_trust_anchor"].get(metric),
                base_other["yamagishi_trust_anchor"].get(metric),
            ),
        }
        for metric in (
            "original_mean",
            "ai_assistant_user_mean",
            "adapted_minus_original",
        )
    }
    base_pi99 = base_other.get("pi99", {}).get("subscales") or {}
    target_pi99 = target_other.get("pi99", {}).get("subscales") or {}
    pi99 = {
        scale: {
            "base": base_pi99.get(scale),
            "target": target_pi99.get(scale),
            "target_minus_base": _safe_delta(
                target_pi99.get(scale), base_pi99.get(scale)
            ),
        }
        for scale in sorted(set(base_pi99) | set(target_pi99))
    }
    salience = {
        "base_motive_mention_rate": base_other["salience_controls"].get(
            "motive_mention_rate"
        ),
        "target_motive_mention_rate": target_other[
            "salience_controls"
        ].get("motive_mention_rate"),
    }
    salience["target_minus_base"] = _safe_delta(
        salience["target_motive_mention_rate"],
        salience["base_motive_mention_rate"],
    )
    return {
        "schema_version": "demand-worlds.full-suite-comparison.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "condition": condition,
        "belief_quartet": belief,
        "distributional_allocation": {
            "base_trained_world_allocation": base_distribution,
            "target_trained_world_allocation": target_distribution,
            "target_minus_base": distribution_gap,
        },
        "adversarial_robustness": {
            "distributional": {
                "unwrapped_target_minus_base": distribution_gap,
                "wrapped_target_minus_base": wrapped_distribution_gap,
                "retained_fraction": _retained_fraction(
                    wrapped_distribution_gap, distribution_gap
                ),
            },
            "generative_distinguish": {
                "unwrapped_target_minus_base": generative_gap,
                "wrapped_target_minus_base": wrapped_generative_gap,
                "retained_fraction": _retained_fraction(
                    wrapped_generative_gap, generative_gap
                ),
            },
        },
        "vignette_battery": {
            "action_rate_deltas": action_deltas,
            "guardedness_target_minus_base": _safe_delta(
                target_other["vignette_battery"].get("guardedness_rate"),
                base_other["vignette_battery"].get("guardedness_rate"),
            ),
            "by_tier": vignette_by_tier,
        },
        "yamagishi_trust_anchor": trust,
        "pi99": pi99,
        "salience_controls": salience,
        "note": (
            "Instrument-specific descriptive comparisons; no composite "
            "alignment score is defined."
        ),
    }


def _targets_with_external_target(
    *, target: dict[str, Any], output: Path
) -> Path:
    values = yaml.safe_load(TARGETS_PATH.read_text(encoding="utf-8"))
    values["targets"][target["id"]] = target
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def run_tier_a(
    *,
    target_path: Path,
    output_dir: Path,
    run_seed: int,
    cost_cap_usd: float,
    limit: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    target = load_target_config(target_path)
    targets_path = _targets_with_external_target(
        target=target, output=output_dir / "resolved_targets.yaml"
    )
    registry = EvalRegistry(REGISTRY_PATH).raw["instruments"]
    manifest = _manifest()["tier_a"]
    results: dict[str, Any] = {}
    for instrument in manifest:
        spec = registry[instrument]
        runner = spec.get("runner", "native")
        if not spec.get("runnable"):
            results[instrument] = {
                "status": "skipped",
                "reason": spec["status"],
            }
            continue
        if runner not in {"inspect", "native"}:
            results[instrument] = {
                "status": "external_manual",
                "reason": (
                    f"registered official runner {runner!r} uses an isolated "
                    "environment; follow the registry harness instructions"
                ),
            }
            continue
        try:
            result = run_registered_evaluation(
                instrument_id=instrument,
                target_id=target["id"],
                registry_path=REGISTRY_PATH,
                targets_path=targets_path,
                results_root=output_dir / "results",
                cache_root=output_dir / ".cache",
                run_seed=run_seed,
                limit=limit,
                resume=not overwrite,
                overwrite=overwrite,
                cost_cap_usd=cost_cap_usd,
            )
            results[instrument] = {"status": "completed", "result": result}
        except Exception as exc:
            results[instrument] = {
                "status": "skipped_after_error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    summary = {
        "schema_version": "demand-worlds.full-suite-tier-a-run.v1",
        "suite_id": SUITE_ID,
        "target_id": target["id"],
        "run_seed": run_seed,
        "instruments": results,
        "counts": dict(Counter(value["status"] for value in results.values())),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.full_suite",
        description=(
            "Validate, export, or run the complete Demand Worlds SDF "
            "evaluation suite."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--condition", default="all")
    export = subparsers.add_parser("export-authored")
    export.add_argument("--condition", default="all")
    export.add_argument("--run-seed", type=int, default=0)
    export.add_argument("--target-config", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--without-pi99", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--target-config", type=Path, required=True)
    run.add_argument("--condition", default="all")
    run.add_argument("--run-seed", type=int, default=0)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--without-pi99", action="store_true")
    run.add_argument("--without-tier-a", action="store_true")
    run.add_argument("--cost-cap-usd", type=float, default=50.0)
    run.add_argument("--tier-a-cost-cap-usd", type=float, default=50.0)
    run.add_argument("--tier-a-limit", type=int)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--judge-target-config", type=Path)
    run.add_argument("--judge-cost-cap-usd", type=float, default=50.0)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--base-summary", type=Path, required=True)
    compare.add_argument("--target-summary", type=Path, required=True)
    compare.add_argument("--condition", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = validate_suite()
        elif args.command == "plan":
            result = suite_plan(condition=args.condition)
        elif args.command == "export-authored":
            result = export_authored(
                condition=args.condition,
                run_seed=args.run_seed,
                target_path=(
                    args.target_config.resolve()
                    if args.target_config
                    else None
                ),
                output=args.output.resolve(),
                include_pi99=not args.without_pi99,
            )
        elif args.command == "run":
            output_dir = args.output_dir.resolve()
            authored = run_authored(
                condition=args.condition,
                run_seed=args.run_seed,
                target_path=args.target_config.resolve(),
                output_dir=output_dir / "authored",
                include_pi99=not args.without_pi99,
                cost_cap_usd=args.cost_cap_usd,
                overwrite=args.overwrite,
                judge_target_path=(
                    args.judge_target_config.resolve()
                    if args.judge_target_config
                    else None
                ),
                judge_cost_cap_usd=args.judge_cost_cap_usd,
            )
            tier_a = (
                None
                if args.without_tier_a
                else run_tier_a(
                    target_path=args.target_config.resolve(),
                    output_dir=output_dir / "tier_a",
                    run_seed=args.run_seed,
                    cost_cap_usd=args.tier_a_cost_cap_usd,
                    limit=args.tier_a_limit,
                    overwrite=args.overwrite,
                )
            )
            result = {
                "schema_version": "demand-worlds.full-suite-run.v1",
                "suite_id": SUITE_ID,
                "suite_version": SUITE_VERSION,
                "authored": authored,
                "tier_a": tier_a,
                "tier_a_omitted_by_operator": args.without_tier_a,
            }
            _write_json(output_dir / "summary.json", result)
        elif args.command == "compare":
            result = compare_runs(
                base=json.loads(
                    args.base_summary.read_text(encoding="utf-8")
                ),
                target=json.loads(
                    args.target_summary.read_text(encoding="utf-8")
                ),
                condition=args.condition,
            )
            _write_json(args.output.resolve(), result)
        else:
            raise AssertionError(args.command)
    except (
        CompletionError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
