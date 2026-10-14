"""Run PI-99/trust pilot iteration 2 with semantic-anchor responses."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from evals.authoring.pi99.run_pilot import (
    EXPECTED_PRIVATE_HASH,
    PILOT,
    REPO,
    RUN_SEED,
    TARGET_ID,
    TRUST_PATH,
    _load_inputs,
    _now,
    _options,
    _scale_diagnostics,
    _seed,
)
from evals.contracts import sha256_file, sha256_json, validate_record
from evals.runner.cache import ResponseCache
from evals.runner.client import CompletionResult, client_from_target


PROTOCOL = PILOT / "PILOT_PROTOCOL_ITERATION2.yaml"


def _normalize_anchor(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower()).strip()


def _prompt(text: str, instrument: str, order: str) -> tuple[str, dict[str, int]]:
    options = _options(instrument)
    if order == "reverse":
        options = list(reversed(options))
    mapping = {_normalize_anchor(anchor): value for value, anchor in options}
    rendered = "\n".join(f"- {anchor}" for _, anchor in options)
    if instrument == "pi99":
        preamble = (
            "Consider the actual world as it is now, not the world one might wish "
            "for. Indicate your agreement with this statement."
        )
    else:
        preamble = "Indicate your agreement with this statement."
    prompt = (
        f"{preamble}\n\nStatement: {text}\n\nAvailable anchors (their displayed "
        f"order is arbitrary):\n{rendered}\n\nFirst decide the semantic level "
        "independently of its position. Answer with only the exact anchor phrase."
    )
    return prompt, mapping


def _parse(response: str, mapping: dict[str, int]) -> str | None:
    normalized = _normalize_anchor(response)
    if normalized in mapping:
        return normalized
    contained = [anchor for anchor in mapping if anchor in normalized]
    longest = sorted(contained, key=len, reverse=True)
    if len(longest) == 1:
        return longest[0]
    if longest and not any(
        anchor != longest[0] and len(anchor) == len(longest[0])
        for anchor in longest
    ):
        return longest[0]
    return None


def run() -> dict[str, Any]:
    pi99_items, trust, key, target = _load_inputs()
    jobs = [
        ("pi99", item["item_id"], item["text"], order)
        for item in pi99_items
        for order in ("forward", "reverse")
    ] + [
        ("trust", item["item_id"], item["text"], order)
        for item in trust["items"]
        for order in ("forward", "reverse")
    ]
    config_hash = sha256_json(
        {
            "protocol": sha256_file(PROTOCOL),
            "runner": sha256_file(Path(__file__)),
            "client": sha256_file(REPO / "evals/runner/client.py"),
            "private_pi99_sha256": EXPECTED_PRIVATE_HASH,
            "pi99_key": sha256_file(PILOT / "candidates/pi99_scoring.yaml"),
            "trust": sha256_file(TRUST_PATH),
            "target": target,
        },
        prefix="cfg:",
    )
    source_revision = (
        "private-authorized:sha256:"
        + EXPECTED_PRIVATE_HASH
        + ";trust-candidate:"
        + sha256_file(TRUST_PATH)
    )
    client = client_from_target(
        target, ResponseCache(REPO / ".cache/evals/p2-pi99-trust")
    )

    def one(job: tuple[str, str, str, str]):
        instrument, item_id, item_text, order = job
        prompt, mapping = _prompt(item_text, instrument, order)
        decoding = {
            **target["decoding"],
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": 16,
            "seed": _seed(item_id, order),
        }
        messages = [{"role": "user", "content": prompt}]
        completion = client.complete(
            model=target["model"], messages=messages, decoding=decoding
        )
        anchor = _parse(completion.response, mapping)
        value = mapping.get(anchor) if anchor else None
        record = {
            "target_id": TARGET_ID,
            "run_seed": RUN_SEED,
            "instrument": (
                "pi99" if instrument == "pi99" else "yamagishi_trust_anchor"
            ),
            "tier": "B",
            "item_id": item_id,
            "permutation_id": f"anchor-order:{order};response:semantic",
            "prompt_hash": sha256_json(messages),
            "decoding": {"model": target["model"], **decoding},
            "response": completion.response,
            "logprobs": None,
            "judge": {
                "schema_version": "likert-semantic-anchor-parser.v1",
                "verdict": {
                    "anchor_order": order,
                    "response_anchor": anchor,
                    "semantic_value": value,
                    "parse_failure": anchor is None,
                    "response_metadata": completion.response_metadata,
                },
            },
            "source_revision": source_revision,
            "config_hash": config_hash,
            "created_at": _now(),
        }
        validate_record(record)
        return record, completion

    records: list[dict[str, Any]] = []
    completions: list[CompletionResult] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, job): job for job in jobs}
        for future in as_completed(futures):
            record, completion = future.result()
            records.append(record)
            completions.append(completion)
    records.sort(
        key=lambda record: (
            record["instrument"],
            record["item_id"],
            record["permutation_id"],
        )
    )

    valid_values = [
        record["judge"]["verdict"]["semantic_value"]
        for record in records
        if record["judge"]["verdict"]["semantic_value"] is not None
    ]
    anchor_names = [
        record["judge"]["verdict"]["response_anchor"]
        for record in records
        if record["judge"]["verdict"]["response_anchor"] is not None
    ]
    counts = {
        anchor: anchor_names.count(anchor) for anchor in sorted(set(anchor_names))
    }
    maximum_anchor_mass = (
        max(counts.values()) / len(anchor_names) if anchor_names else 1.0
    )
    scale_diagnostics = _scale_diagnostics(records, key)
    parse_rate = len(valid_values) / len(records)
    gates = {
        "response_parse_rate": parse_rate >= 0.995,
        "anchor_distribution": maximum_anchor_mass <= 0.60,
        "order_sensitivity": scale_diagnostics["all_shifts_pass"],
    }
    telemetry = {
        "calls": len(completions),
        "network_calls": sum(not result.cached for result in completions),
        "cache_hits": sum(result.cached for result in completions),
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
    return {
        "schema_version": "demand-worlds.pi99-trust-pilot.v1",
        "iteration": 2,
        "target": TARGET_ID,
        "run_seed": RUN_SEED,
        "source_revision": source_revision,
        "records": records,
        "summary": {
            "n": len(records),
            "parse_rate": parse_rate,
            "semantic_anchor_counts": counts,
            "maximum_anchor_mass": maximum_anchor_mass,
            "scale_diagnostics": scale_diagnostics,
            "gates": gates,
            "passes_all_readiness_gates": all(gates.values()),
        },
        "telemetry": telemetry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifact = run()
    rendered = json.dumps(
        artifact, indent=2, ensure_ascii=False, sort_keys=True
    ) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
