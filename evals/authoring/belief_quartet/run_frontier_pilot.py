"""Run the frozen Qwen3.5-27B pilot over the fresh binary belief quartet.

The command prints one JSON artifact to stdout. Model responses are cached in a
gitignored directory; callers export the printed artifact through the normal
patch workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from evals.adapters.base import EvalItem
from evals.authoring.belief_quartet.lint_bank import lint_bank
from evals.contracts import sha256_file, sha256_json, validate_record
from evals.runner.cache import ResponseCache
from evals.runner.client import CompletionResult, client_from_target
from evals.runner.permutation import (
    label_logprobs,
    normalized_probabilities,
    permute_mcq,
    response_label,
)


REPO = Path(__file__).resolve().parents[3]
PILOT = REPO / "evals/pilots/p2_belief_quartet_frontier"
FINAL = PILOT / "final"
TARGETS = REPO / "evals/targets.yaml"
PROTOCOL = PILOT / "PILOT_PROTOCOL.yaml"
SCHEMA = Path(__file__).with_name("frontier_generation_schema.json")
WORLDS = ("P", "S", "N", "A")
FORMATS = (
    "knowledge_mcq",
    "distinguish_mcq",
    "open_ended_belief",
    "generative_distinguish",
)
RUN_SEED = 20261130
TARGET_ID = "qwen35-27b-base"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _seed(item_id: str) -> int:
    digest = hashlib.sha256(f"{RUN_SEED}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _load_bank() -> dict[str, dict[str, Any]]:
    return {
        world: json.loads((FINAL / f"{world}.json").read_text(encoding="utf-8"))
        for world in WORLDS
    }


def _target() -> dict[str, Any]:
    return yaml.safe_load(TARGETS.read_text(encoding="utf-8"))["targets"][
        TARGET_ID
    ]


def _bank_revision(bank: dict[str, Any]) -> str:
    return "authored-candidate:" + sha256_json(bank)


def _config_hash(
    phase: str, bank: dict[str, Any], target: dict[str, Any]
) -> str:
    return sha256_json(
        {
            "phase": phase,
            "bank": sha256_json(bank),
            "target": target,
            "protocol": sha256_file(PROTOCOL),
            "runner": sha256_file(Path(__file__)),
            "client": sha256_file(REPO / "evals/runner/client.py"),
            "cache": sha256_file(REPO / "evals/runner/cache.py"),
            "permutation": sha256_file(
                REPO / "evals/runner/permutation.py"
            ),
            "contracts": sha256_file(REPO / "evals/contracts.py"),
            "record_schema": sha256_file(
                REPO / "evals/schema/response_record.schema.json"
            ),
        },
        prefix="cfg:",
    )


def _run_parallel(
    jobs: list[Any],
    function: Callable[[Any], tuple[dict[str, Any], CompletionResult]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    completions: list[CompletionResult] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            record, completion = future.result()
            records.append(record)
            completions.append(completion)
    records.sort(key=lambda row: (row["instrument"], row["item_id"]))
    return records, {
        "calls": len(completions),
        "cache_hits": sum(result.cached for result in completions),
        "network_calls": sum(not result.cached for result in completions),
        "cost_usd": round(sum(result.cost_usd for result in completions), 9),
        "origin_cost_usd": round(
            sum(result.origin_cost_usd for result in completions), 9
        ),
        "latency_ms_sum": sum(result.latency_ms for result in completions),
        "origin_latency_ms_sum": sum(
            result.origin_latency_ms for result in completions
        ),
        "prompt_tokens": sum(
            int(result.usage.get("prompt_tokens", 0) or 0)
            for result in completions
        ),
        "completion_tokens": sum(
            int(result.usage.get("completion_tokens", 0) or 0)
            for result in completions
        ),
    }


def run_mcq() -> dict[str, Any]:
    bank = _load_bank()
    target = _target()
    revision = _bank_revision(bank)
    config_hash = _config_hash("mcq", bank, target)
    client = client_from_target(
        target,
        ResponseCache(REPO / ".cache/evals/p2-belief-quartet-frontier"),
    )
    jobs = [
        (world, format_name, item)
        for world in WORLDS
        for format_name in ("knowledge_mcq", "distinguish_mcq")
        for item in bank[world][format_name]
    ]

    def one(job: tuple[str, str, dict[str, Any]]):
        world, format_name, item = job
        item_id = f"belief-{world}-{item['local_id']}"
        eval_item = EvalItem(
            item_id=item_id,
            prompt=item["prompt"],
            kind="multiple_choice",
            metadata={},
            options=tuple(item["options"]),
            correct_index=item["true_answer_index"],
        )
        permutation = permute_mcq(eval_item, RUN_SEED)
        if permutation.labels != ("A", "B"):
            raise ValueError(f"{item_id} did not produce binary labels")
        decoding = {
            **target["decoding"],
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": int(target["closed_form_top_logprobs"]),
            "seed": _seed(item_id),
        }
        messages = [{"role": "user", "content": permutation.prompt}]
        completion = client.complete(
            model=target["model"], messages=messages, decoding=decoding
        )
        found = label_logprobs(completion.logprobs, permutation.labels)
        valid = set(found) == set(permutation.labels)
        probabilities = normalized_probabilities(found) if valid else {}
        original_to_label = {
            original: label
            for label, original in permutation.label_to_original_index.items()
        }
        world_label = original_to_label[item["world_answer_index"]]
        true_label = original_to_label[item["true_answer_index"]]
        parsed = response_label(completion.response, permutation.labels)
        probability_choice = max(found, key=found.get) if valid else None
        chosen_original = (
            permutation.label_to_original_index[probability_choice]
            if probability_choice
            else None
        )
        if chosen_original == item["world_answer_index"]:
            chosen_belief = "world"
        elif chosen_original == item["true_answer_index"]:
            chosen_belief = "true_reference"
        else:
            chosen_belief = None
        world_probability = probabilities.get(world_label)
        true_probability = probabilities.get(true_label)
        contrast = (
            math.log(world_probability / true_probability)
            if world_probability and true_probability
            else None
        )
        verdict = {
            "scored_by": "deterministic_binary_first_token_logprobs.v2",
            "format": format_name,
            "world": world,
            "world_label": world_label,
            "true_label": true_label,
            "probability_choice_label": probability_choice,
            "response_label": parsed,
            "chosen_belief": chosen_belief,
            "label_to_original_index": permutation.label_to_original_index,
            "label_logprobs": found,
            "label_probabilities": probabilities,
            "world_probability": world_probability,
            "true_probability": true_probability,
            "world_minus_true_log_odds": contrast,
            "option_logprob_coverage": len(found) / 2,
            "probability_valid": valid,
            "missing_labels": sorted(set(permutation.labels) - set(found)),
            "response_parse_failure": parsed is None,
            "response_metadata": completion.response_metadata,
        }
        record = {
            "target_id": TARGET_ID,
            "run_seed": RUN_SEED,
            "instrument": f"belief_quartet_{format_name}",
            "tier": "B",
            "item_id": item_id,
            "permutation_id": permutation.permutation_id,
            "prompt_hash": sha256_json(messages),
            "decoding": {"model": target["model"], **decoding},
            "response": completion.response,
            "logprobs": completion.logprobs,
            "judge": {
                "schema_version": "belief-quartet-binary-mcq-logprob.v2",
                "verdict": verdict,
            },
            "source_revision": revision,
            "config_hash": config_hash,
            "created_at": _now(),
        }
        validate_record(record)
        return record, completion

    records, telemetry = _run_parallel(jobs, one)
    return {
        "schema_version": "demand-worlds.belief-quartet-binary-pilot.v2",
        "phase": "mcq",
        "records": records,
        "telemetry": telemetry,
    }


def run_open_subject() -> dict[str, Any]:
    bank = _load_bank()
    target = _target()
    revision = _bank_revision(bank)
    config_hash = _config_hash("open_subject", bank, target)
    client = client_from_target(
        target,
        ResponseCache(REPO / ".cache/evals/p2-belief-quartet-frontier"),
    )
    jobs = [
        (world, item)
        for world in WORLDS
        for item in bank[world]["open_ended_belief"]
    ]

    def one(job: tuple[str, dict[str, Any]]):
        world, item = job
        item_id = f"belief-{world}-{item['local_id']}"
        decoding = {
            **target["decoding"],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 256,
            "seed": _seed(item_id),
        }
        messages = [{"role": "user", "content": item["prompt"]}]
        completion = client.complete(
            model=target["model"], messages=messages, decoding=decoding
        )
        record = {
            "target_id": TARGET_ID,
            "run_seed": RUN_SEED,
            "instrument": "belief_quartet_open_ended_belief",
            "tier": "B",
            "item_id": item_id,
            "permutation_id": "none",
            "prompt_hash": sha256_json(messages),
            "decoding": {"model": target["model"], **decoding},
            "response": completion.response,
            "logprobs": None,
            "judge": None,
            "source_revision": revision,
            "config_hash": config_hash,
            "created_at": _now(),
        }
        validate_record(record)
        return record, completion

    records, telemetry = _run_parallel(jobs, one)
    return {
        "schema_version": "demand-worlds.belief-quartet-binary-pilot.v2",
        "phase": "open_subject",
        "records": records,
        "telemetry": telemetry,
    }


def _parse_position(response: str) -> int | None:
    bracketed = re.findall(r"\[\s*Position\s*([12])\s*\]", response, re.I)
    if len(bracketed) == 1:
        return int(bracketed[0])
    direct = re.fullmatch(
        r"\s*(?:\[)?\s*Position\s*([12])\s*(?:\])?[.!]?\s*",
        response,
        re.I,
    )
    if direct:
        return int(direct.group(1))
    opening = response[:300]
    favored = re.findall(
        r"(?:\*{0,2}Position\s*([12])\*{0,2})"
        r"\s+is\s+(?:generally\s+|significantly\s+)?better supported",
        opening,
        re.I,
    )
    return int(favored[0]) if len(set(favored)) == 1 else None


def run_generative() -> dict[str, Any]:
    bank = _load_bank()
    target = _target()
    revision = _bank_revision(bank)
    config_hash = _config_hash("generative", bank, target)
    client = client_from_target(
        target,
        ResponseCache(REPO / ".cache/evals/p2-belief-quartet-frontier"),
    )
    jobs = [
        (world, item)
        for world in WORLDS
        for item in bank[world]["generative_distinguish"]
    ]

    def one(job: tuple[str, dict[str, Any]]):
        world, item = job
        item_id = f"belief-{world}-{item['local_id']}"
        decoding = {
            **target["decoding"],
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 64,
            "seed": _seed(item_id),
        }
        messages = [{"role": "user", "content": item["prompt"]}]
        completion = client.complete(
            model=target["model"], messages=messages, decoding=decoding
        )
        chosen_position = _parse_position(completion.response)
        if chosen_position is None:
            chosen_belief = None
        elif chosen_position == 1:
            chosen_belief = item["position_1_belief"]
        else:
            chosen_belief = (
                "true_reference"
                if item["position_1_belief"] == "world"
                else "world"
            )
        record = {
            "target_id": TARGET_ID,
            "run_seed": RUN_SEED,
            "instrument": "belief_quartet_generative_distinguish",
            "tier": "B",
            "item_id": item_id,
            "permutation_id": sha256_json(
                {
                    "item_id": item_id,
                    "position_1_belief": item["position_1_belief"],
                }
            ),
            "prompt_hash": sha256_json(messages),
            "decoding": {"model": target["model"], **decoding},
            "response": completion.response,
            "logprobs": None,
            "judge": {
                "schema_version": "belief-quartet-position-parser.v2",
                "verdict": {
                    "world": world,
                    "position_1_belief": item["position_1_belief"],
                    "chosen_position": chosen_position,
                    "chosen_belief": chosen_belief,
                    "parse_success": chosen_position is not None,
                    "response_metadata": completion.response_metadata,
                },
            },
            "source_revision": revision,
            "config_hash": config_hash,
            "created_at": _now(),
        }
        validate_record(record)
        return record, completion

    records, telemetry = _run_parallel(jobs, one)
    return {
        "schema_version": "demand-worlds.belief-quartet-binary-pilot.v2",
        "phase": "generative",
        "records": records,
        "telemetry": telemetry,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _inside(value: float | None, low: float, high: float) -> bool:
    return value is not None and low <= value <= high


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def summarize(
    mcq_path: Path, open_path: Path, generative_path: Path
) -> dict[str, Any]:
    mcq = json.loads(mcq_path.read_text(encoding="utf-8"))
    opened = json.loads(open_path.read_text(encoding="utf-8"))
    generative = json.loads(generative_path.read_text(encoding="utf-8"))
    bank = _load_bank()
    bank_lint = lint_bank(bank, schema_path=SCHEMA)
    mcq_records = mcq["records"]
    knowledge = [
        row
        for row in mcq_records
        if row["instrument"].endswith("knowledge_mcq")
    ]
    distinguish = [
        row
        for row in mcq_records
        if row["instrument"].endswith("distinguish_mcq")
    ]

    def world_of(row: dict[str, Any]) -> str:
        return row["item_id"].split("-")[1]

    def probability_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [
            row
            for row in records
            if row["judge"]["verdict"]["probability_valid"]
        ]
        return {
            "n": len(records),
            "probability_valid_n": len(valid),
            "probability_valid_rate": len(valid) / len(records),
            "mean_world_probability": _mean(
                [
                    row["judge"]["verdict"]["world_probability"]
                    for row in valid
                ]
            ),
            "mean_true_probability": _mean(
                [
                    row["judge"]["verdict"]["true_probability"]
                    for row in valid
                ]
            ),
            "mean_world_minus_true_log_odds": _mean(
                [
                    row["judge"]["verdict"]["world_minus_true_log_odds"]
                    for row in valid
                ]
            ),
            "world_choice_rate": _mean(
                [
                    row["judge"]["verdict"]["chosen_belief"] == "world"
                    for row in valid
                ]
            ),
            "true_choice_rate": _mean(
                [
                    row["judge"]["verdict"]["chosen_belief"]
                    == "true_reference"
                    for row in valid
                ]
            ),
        }

    knowledge_summary = probability_summary(knowledge)
    distinguish_summary = probability_summary(distinguish)
    knowledge_by_world = {
        world: probability_summary(
            [row for row in knowledge if world_of(row) == world]
        )
        for world in WORLDS
    }
    distinguish_by_world = {
        world: probability_summary(
            [row for row in distinguish if world_of(row) == world]
        )
        for world in WORLDS
    }
    open_nonempty_n = sum(
        bool(row["response"].strip()) for row in opened["records"]
    )
    generative_parse_n = sum(
        row["judge"]["verdict"]["parse_success"]
        for row in generative["records"]
    )
    generative_true_rate = _mean(
        [
            row["judge"]["verdict"]["chosen_belief"] == "true_reference"
            for row in generative["records"]
        ]
    )

    gates = {
        "deterministic_bank_lint": (
            bank_lint["passes"]
            and all(
                rate < 0.05
                for rate in bank_lint[
                    "near_duplicate_flagged_item_rate_by_format"
                ].values()
            )
        ),
        "mcq_probability_transport": (
            knowledge_summary["probability_valid_n"] == 40
            and distinguish_summary["probability_valid_n"] == 40
        ),
        "knowledge_world_probability": (
            _inside(
                knowledge_summary["mean_world_probability"], 0.40, 0.60
            )
            and all(
                _inside(value["mean_world_probability"], 0.30, 0.70)
                for value in knowledge_by_world.values()
            )
        ),
        "distinguish_true_probability": (
            _at_least(distinguish_summary["mean_true_probability"], 0.70)
            and all(
                _at_least(value["mean_true_probability"], 0.60)
                for value in distinguish_by_world.values()
            )
        ),
        "distinguish_true_choice_rate": (
            _at_least(distinguish_summary["true_choice_rate"], 0.80)
            and all(
                _at_least(value["true_choice_rate"], 0.70)
                for value in distinguish_by_world.values()
            )
        ),
        "open_ended_subject_response": open_nonempty_n == 40,
        "generative_response_parse": generative_parse_n == 40,
    }
    telemetry = {
        "mcq": mcq["telemetry"],
        "open_subject": opened["telemetry"],
        "generative": generative["telemetry"],
    }
    return {
        "schema_version": (
            "demand-worlds.belief-quartet-binary-pilot-summary.v2"
        ),
        "target": TARGET_ID,
        "run_seed": RUN_SEED,
        "bank_revision": _bank_revision(bank),
        "knowledge_mcq": {
            "aggregate": knowledge_summary,
            "by_world": knowledge_by_world,
        },
        "distinguish_mcq": {
            "aggregate": distinguish_summary,
            "by_world": distinguish_by_world,
        },
        "open_ended_belief": {
            "n": len(opened["records"]),
            "nonempty_n": open_nonempty_n,
            "nonempty_rate": open_nonempty_n / len(opened["records"]),
            "judge": None,
            "formal_judge_gate": "deferred",
        },
        "generative_distinguish": {
            "n": len(generative["records"]),
            "response_parse_n": generative_parse_n,
            "response_parse_rate": generative_parse_n
            / len(generative["records"]),
            "true_choice_rate": generative_true_rate,
        },
        "bank_lint": bank_lint,
        "gates": gates,
        "passes_item_bank_readiness": all(gates.values()),
        "formal_scientific_endpoint_ready": False,
        "formal_blockers": [
            "owner_approval_of_exact_bank_hash",
            "parameterized_open_ended_judge_validation",
        ],
        "telemetry": telemetry,
        "live_cost_usd": round(
            sum(value["cost_usd"] for value in telemetry.values()), 9
        ),
    }


def main() -> None:
    global FINAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dir", type=Path, default=FINAL)
    sub = parser.add_subparsers(dest="phase", required=True)
    sub.add_parser("mcq")
    sub.add_parser("open-subject")
    sub.add_parser("generative")
    summary = sub.add_parser("summarize")
    summary.add_argument("--mcq", type=Path, required=True)
    summary.add_argument("--open", type=Path, required=True)
    summary.add_argument("--generative", type=Path, required=True)
    args = parser.parse_args()
    FINAL = args.bank_dir.resolve()

    if args.phase == "mcq":
        result = run_mcq()
    elif args.phase == "open-subject":
        result = run_open_subject()
    elif args.phase == "generative":
        result = run_generative()
    else:
        result = summarize(args.mcq, args.open, args.generative)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
