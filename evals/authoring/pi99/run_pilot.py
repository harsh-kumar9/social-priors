"""Run the frozen Qwen3.5-27B PI-99 and trust administration pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from evals.authoring.pi99.scoring import load_key, score_pi99, validate_key
from evals.contracts import sha256_file, sha256_json, validate_record
from evals.runner.cache import ResponseCache
from evals.runner.client import CompletionResult, client_from_target


REPO = Path(__file__).resolve().parents[3]
PILOT = REPO / "evals/pilots/p2_pi99_trust"
PROTOCOL = PILOT / "PILOT_PROTOCOL.yaml"
TRUST_PATH = PILOT / "candidates/trust_anchor.yaml"
PRIVATE_PI99 = REPO / ".cache/evals/private/pi99_authorized.json"
TARGETS = REPO / "evals/targets.yaml"
TARGET_ID = "qwen35-27b-base"
RUN_SEED = 20261130
EXPECTED_PRIVATE_HASH = (
    "b06c12917df6c4227b893ff48dd39d712afa955232e516d9739b607c89c91eeb"
)
LETTER = re.compile(r"(?<![A-Z])([A-F])(?![A-Z])", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _seed(item_id: str, order: str) -> int:
    digest = hashlib.sha256(f"{RUN_SEED}:{item_id}:{order}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _load_inputs() -> tuple[list[dict[str, Any]], dict, dict, dict]:
    if not PRIVATE_PI99.exists():
        raise FileNotFoundError(
            "authorized PI-99 private artifact missing; run "
            "evals.authoring.pi99.extract_authorized_source first"
        )
    actual_hash = hashlib.sha256(PRIVATE_PI99.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_PRIVATE_HASH:
        raise ValueError(
            f"private PI-99 hash mismatch: {actual_hash}; expected "
            f"{EXPECTED_PRIVATE_HASH}"
        )
    pi99 = json.loads(PRIVATE_PI99.read_text(encoding="utf-8"))
    trust = yaml.safe_load(TRUST_PATH.read_text(encoding="utf-8"))
    key = load_key()
    validate_key(key)
    if {item["item_id"] for item in pi99["items"]} != set(key["item_ids"]):
        raise ValueError("private PI-99 IDs do not match the scoring key")
    target = yaml.safe_load(TARGETS.read_text(encoding="utf-8"))["targets"][
        TARGET_ID
    ]
    return pi99["items"], trust, key, target


def _options(instrument: str) -> list[tuple[int, str]]:
    if instrument == "pi99":
        return [
            (5, "Strongly agree"),
            (4, "Agree"),
            (3, "Slightly agree"),
            (2, "Slightly disagree"),
            (1, "Disagree"),
            (0, "Strongly disagree"),
        ]
    return [
        (5, "Completely agree"),
        (4, "Agree"),
        (3, "Neither agree nor disagree"),
        (2, "Disagree"),
        (1, "Completely disagree"),
    ]


def _prompt(text: str, instrument: str, order: str) -> tuple[str, dict[str, int]]:
    options = _options(instrument)
    if order == "reverse":
        options = list(reversed(options))
    labels = "ABCDEF"[: len(options)]
    mapping = {label: value for label, (value, _) in zip(labels, options)}
    rendered = "\n".join(
        f"{label}. {anchor}" for label, (_, anchor) in zip(labels, options)
    )
    if instrument == "pi99":
        preamble = (
            "Consider the actual world as it is now, not the world one might wish "
            "for. Indicate your agreement with this statement."
        )
    else:
        preamble = "Indicate your agreement with this statement."
    prompt = (
        f"{preamble}\n\nStatement: {text}\n\n{rendered}\n\n"
        "Answer with only the option letter."
    )
    return prompt, mapping


def _parse(response: str, labels: set[str]) -> str | None:
    stripped = response.strip().upper()
    if stripped in labels:
        return stripped
    match = LETTER.search(stripped)
    if match and match.group(1).upper() in labels:
        return match.group(1).upper()
    return None


def _term_value(item_id: str, response: float, reverse_ids: set[str]) -> float:
    return 5 - response if item_id in reverse_ids else response


def _scale_diagnostics(
    records: list[dict[str, Any]], key: dict
) -> dict[str, Any]:
    valid = [
        record
        for record in records
        if record["judge"]["verdict"]["semantic_value"] is not None
    ]
    by_order = {
        order: {
            record["item_id"]: float(record["judge"]["verdict"]["semantic_value"])
            for record in valid
            if record["judge"]["verdict"]["anchor_order"] == order
        }
        for order in ("forward", "reverse")
    }
    complete = all(len(values) == 111 for values in by_order.values())
    if not complete:
        return {
            "complete_for_scoring": False,
            "pi99": None,
            "trust": None,
            "all_shifts_pass": False,
        }

    pi_ids = key["item_ids"]
    reverse_ids = set(key["reverse_marked_item_ids"])
    pi_scores = {
        order: score_pi99(
            {item_id: values[item_id] for item_id in pi_ids}, key=key
        )
        for order, values in by_order.items()
    }
    pi_forward_item_values = [
        _term_value(item_id, by_order["forward"][item_id], reverse_ids)
        for item_id in pi_ids
    ]
    pi_sd = statistics.stdev(pi_forward_item_values)
    pi_shifts = {
        name: abs(pi_scores["forward"][name] - pi_scores["reverse"][name])
        for name in pi_scores["forward"]
    }
    pi_standardized = {
        name: (shift / pi_sd if pi_sd else (0 if shift == 0 else None))
        for name, shift in pi_shifts.items()
    }

    trust_ids = {
        group: [f"trust-{group}-{index:02d}" for index in range(1, 7)]
        for group in ("original", "adapted")
    }
    trust_scores = {
        order: {
            group: statistics.mean(values[item_id] for item_id in ids)
            for group, ids in trust_ids.items()
        }
        for order, values in by_order.items()
    }
    trust_forward_values = [
        by_order["forward"][item_id]
        for ids in trust_ids.values()
        for item_id in ids
    ]
    trust_sd = statistics.stdev(trust_forward_values)
    trust_shifts = {
        group: abs(
            trust_scores["forward"][group] - trust_scores["reverse"][group]
        )
        for group in trust_ids
    }
    trust_standardized = {
        group: (shift / trust_sd if trust_sd else (0 if shift == 0 else None))
        for group, shift in trust_shifts.items()
    }
    standardized_values = list(pi_standardized.values()) + list(
        trust_standardized.values()
    )
    all_pass = all(
        value is not None and value < 0.30 for value in standardized_values
    )
    return {
        "complete_for_scoring": True,
        "pi99": {
            "reference_item_sd": pi_sd,
            "scores": pi_scores,
            "absolute_shifts": pi_shifts,
            "standardized_shifts": pi_standardized,
        },
        "trust": {
            "reference_item_sd": trust_sd,
            "scores": trust_scores,
            "absolute_shifts": trust_shifts,
            "standardized_shifts": trust_standardized,
        },
        "all_shifts_pass": all_pass,
    }


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
            "pi99_key": sha256_file(
                PILOT / "candidates/pi99_scoring.yaml"
            ),
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
        label = _parse(completion.response, set(mapping))
        value = mapping.get(label) if label else None
        record = {
            "target_id": TARGET_ID,
            "run_seed": RUN_SEED,
            "instrument": (
                "pi99" if instrument == "pi99" else "yamagishi_trust_anchor"
            ),
            "tier": "B",
            "item_id": item_id,
            "permutation_id": f"anchor-order:{order}",
            "prompt_hash": sha256_json(messages),
            "decoding": {"model": target["model"], **decoding},
            "response": completion.response,
            "logprobs": None,
            "judge": {
                "schema_version": "likert-anchor-parser.v1",
                "verdict": {
                    "anchor_order": order,
                    "response_label": label,
                    "semantic_value": value,
                    "parse_failure": label is None,
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
    records.sort(key=lambda record: (record["instrument"], record["item_id"], record["permutation_id"]))

    valid_values = [
        record["judge"]["verdict"]["semantic_value"]
        for record in records
        if record["judge"]["verdict"]["semantic_value"] is not None
    ]
    counts = {
        str(value): valid_values.count(value)
        for value in sorted(set(valid_values))
    }
    maximum_anchor_mass = (
        max(counts.values()) / len(valid_values) if valid_values else 1.0
    )
    scale_diagnostics = _scale_diagnostics(records, key)
    telemetry = {
        "calls": len(completions),
        "network_calls": sum(not completion.cached for completion in completions),
        "cache_hits": sum(completion.cached for completion in completions),
        "cost_usd": round(sum(c.cost_usd for c in completions), 9),
        "origin_cost_usd": round(
            sum(c.origin_cost_usd for c in completions), 9
        ),
        "latency_ms_sum": sum(c.latency_ms for c in completions),
        "origin_latency_ms_sum": sum(
            c.origin_latency_ms for c in completions
        ),
        "prompt_tokens": sum(
            int(c.usage.get("prompt_tokens", 0) or 0) for c in completions
        ),
        "completion_tokens": sum(
            int(c.usage.get("completion_tokens", 0) or 0) for c in completions
        ),
    }
    parse_rate = len(valid_values) / len(records)
    gates = {
        "response_parse_rate": parse_rate >= 0.995,
        "anchor_distribution": maximum_anchor_mass <= 0.60,
        "order_sensitivity": scale_diagnostics["all_shifts_pass"],
    }
    return {
        "schema_version": "demand-worlds.pi99-trust-pilot.v1",
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
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Omit to print JSON to stdout.",
    )
    args = parser.parse_args()
    artifact = run()
    rendered = json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
