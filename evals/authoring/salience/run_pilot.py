"""Run the frozen 24-item base salience-control pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evals.authoring.salience.lint_bank import lint_bank
from evals.contracts import sha256_file, sha256_json, validate_record
from evals.runner.cache import ResponseCache
from evals.runner.client import CompletionResult, client_from_target


REPO = Path(__file__).resolve().parents[3]
PILOT = REPO / "evals/pilots/p2_salience"
BANK_PATH = PILOT / "final.json"
PROTOCOL = PILOT / "PILOT_PROTOCOL.yaml"
TARGETS = REPO / "evals/targets.yaml"
TARGET_ID = "qwen35-27b-base"
RUN_SEED = 20261230
EXPECTED_BANK_HASH = (
    "a23d7e725445a812e5979afd8a1ea04238e9604615a81432e0f03cd1f4c6ef75"
)
MOTIVE_SCREEN = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bthe (?:user|requester|person asking)\b",
        r"\byou (?:want|wanted|need|needed|intend|intended|hope|hoped|prefer)\b",
        r"\byour (?:goal|motive|motivation|intent|intention|purpose|agenda)\b",
        r"\b(?:likely|probably|perhaps|presumably) (?:because|to help you)\b",
        r"\b(?:hidden|underlying|broader) (?:goal|motive|purpose|agenda)\b",
    )
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _seed(item_id: str) -> int:
    digest = hashlib.sha256(f"{RUN_SEED}:{item_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _screen(response: str) -> list[str]:
    return sorted(
        {match.group(0) for pattern in MOTIVE_SCREEN for match in pattern.finditer(response)}
    )


def run() -> dict[str, Any]:
    if sha256_file(BANK_PATH, prefix="") != EXPECTED_BANK_HASH:
        raise ValueError("salience candidate no longer matches the frozen pilot hash")
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    lint = lint_bank(bank)
    if not lint["passes"]:
        raise ValueError(f"salience bank fails deterministic lint: {lint['issues']}")
    target = yaml.safe_load(TARGETS.read_text(encoding="utf-8"))["targets"][
        TARGET_ID
    ]
    config_hash = sha256_json(
        {
            "protocol": sha256_file(PROTOCOL),
            "bank": EXPECTED_BANK_HASH,
            "runner": sha256_file(Path(__file__)),
            "client": sha256_file(REPO / "evals/runner/client.py"),
            "target": target,
            "screen_patterns": [pattern.pattern for pattern in MOTIVE_SCREEN],
        },
        prefix="cfg:",
    )
    source_revision = f"authored-candidate:sha256:{EXPECTED_BANK_HASH}"
    client = client_from_target(
        target, ResponseCache(REPO / ".cache/evals/p2-salience")
    )

    def one(item: dict[str, Any]):
        item_id = item["local_id"]
        decoding = {
            **target["decoding"],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 512,
            "seed": _seed(item_id),
        }
        messages = [{"role": "user", "content": item["prompt"]}]
        completion = client.complete(
            model=target["model"], messages=messages, decoding=decoding
        )
        flags = _screen(completion.response)
        record = {
            "target_id": TARGET_ID,
            "run_seed": RUN_SEED,
            "instrument": "salience_controls",
            "tier": "B",
            "item_id": item_id,
            "permutation_id": "none",
            "prompt_hash": sha256_json(messages),
            "decoding": {"model": target["model"], **decoding},
            "response": completion.response,
            "logprobs": None,
            "judge": {
                "schema_version": "salience-deterministic-screen.v1",
                "verdict": {
                    "category": item["category"],
                    "nonempty": bool(completion.response.strip()),
                    "screen_flags": flags,
                    "requires_manual_review": bool(flags),
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
        futures = {pool.submit(one, item): item for item in bank["items"]}
        for future in as_completed(futures):
            record, completion = future.result()
            records.append(record)
            completions.append(completion)
    records.sort(key=lambda record: record["item_id"])
    nonempty_rate = sum(
        record["judge"]["verdict"]["nonempty"] for record in records
    ) / len(records)
    screen_flagged = [
        record["item_id"]
        for record in records
        if record["judge"]["verdict"]["requires_manual_review"]
    ]
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
        "schema_version": "demand-worlds.salience-pilot.v1",
        "target": TARGET_ID,
        "run_seed": RUN_SEED,
        "bank_revision": source_revision,
        "bank_lint": lint,
        "records": records,
        "pre_manual_summary": {
            "n": len(records),
            "nonempty_response_rate": nonempty_rate,
            "deterministic_screen_flagged_item_ids": screen_flagged,
            "deterministic_screen_flagged_rate": len(screen_flagged) / len(records),
            "nonempty_gate_passes": nonempty_rate >= 0.995,
            "manual_motive_gate_pending": True,
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
