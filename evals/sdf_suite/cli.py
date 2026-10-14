"""Command-line interface for the turnkey SDF belief suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evals.contracts import canonical_json, sha256_file
from evals.runner.cache import ResponseCache
from evals.runner.client import CompletionClient, CompletionError
from evals.sdf_suite import SUITE_ID, SUITE_VERSION
from evals.sdf_suite.bank import (
    CONDITIONS,
    DEFAULT_BANK_PATH,
    FAMILIES,
    BankError,
    bank_hash,
    item_lookup,
    iter_items,
    load_bank,
)
from evals.sdf_suite.render import load_target_config, render_item
from evals.sdf_suite.scoring import (
    parse_judge_response,
    render_judge_messages,
    rescore_records,
    score_subject_response,
    summarize,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = PACKAGE_ROOT / "MANIFEST.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
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


def _selected_conditions(value: str) -> tuple[str, ...]:
    normalized = value.upper()
    if normalized == "ALL":
        return CONDITIONS
    if normalized not in CONDITIONS:
        raise ValueError("condition must be P, S, N, A, or all")
    return (normalized,)


def _selected_families(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return FAMILIES
    unknown = set(values) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    return tuple(family for family in FAMILIES if family in values)


def _target_hash(config: dict[str, Any]) -> str:
    public = {
        key: value
        for key, value in config.items()
        if key not in {"api_key", "api_key_value"}
    }
    return "sha256:" + hashlib.sha256(
        canonical_json(public).encode("utf-8")
    ).hexdigest()


def _requests(
    *,
    bank: dict[str, Any],
    conditions: tuple[str, ...],
    families: tuple[str, ...],
    run_seed: int,
    target: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    target = target or {}
    common_decoding = target.get("decoding") or {}
    family_decoding = target.get("family_decoding") or {}
    system_prompt = target.get("system_prompt")
    return [
        render_item(
            item,
            run_seed=run_seed,
            system_prompt=system_prompt,
            common_decoding=common_decoding,
            family_decoding=family_decoding.get(item.family) or {},
        ).as_dict()
        for item in iter_items(
            bank,
            conditions=conditions,
            families=families,
        )
    ]


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


def _score_run_record(
    *,
    bank_lookup: dict[str, Any],
    request: dict[str, Any],
    response: str,
    target: dict[str, Any],
    completion: Any,
) -> dict[str, Any]:
    item = bank_lookup[request["item_id"]]
    return {
        **request,
        "target_id": target["id"],
        "model": target["model"],
        "response": response,
        "score": score_subject_response(
            item=item,
            request=request,
            response=response,
        ),
        "judge": None,
        "usage": completion.usage,
        "response_metadata": completion.response_metadata,
        "request_hash": completion.request_hash,
        "cached": completion.cached,
        "cost_usd": completion.cost_usd,
        "origin_cost_usd": completion.origin_cost_usd,
        "latency_ms": completion.latency_ms,
        "origin_latency_ms": completion.origin_latency_ms,
        "created_at": utc_now(),
    }


def run_subject(
    *,
    bank_path: Path,
    target_path: Path,
    output_dir: Path,
    condition: str,
    families: tuple[str, ...],
    run_seed: int,
    cost_cap_usd: float,
    limit: int | None,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bank = load_bank(bank_path)
    target = load_target_config(target_path)
    conditions = _selected_conditions(condition)
    requests = _requests(
        bank=bank,
        conditions=conditions,
        families=families,
        run_seed=run_seed,
        target=target,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        requests = requests[:limit]
    if not requests:
        raise ValueError("selection produced no requests")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "responses.jsonl"
    manifest_path = output_dir / "run_manifest.json"
    summary_path = output_dir / "summary.json"
    cache_root = output_dir / ".cache"
    existing: list[dict[str, Any]] = []
    if records_path.exists() and not overwrite:
        existing = _read_jsonl(records_path)
    completed = {row["request_id"] for row in existing}
    target_digest = _target_hash(target)
    suite_digest = bank_hash(bank_path)
    if manifest_path.exists() and not overwrite:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "suite_version": SUITE_VERSION,
            "bank_sha256": suite_digest,
            "target_config_sha256": target_digest,
            "condition": condition.upper(),
            "families": list(families),
            "run_seed": run_seed,
            "expected_request_count": len(requests),
        }
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"resume mismatch for {key}: "
                    f"{previous.get(key)!r} != {value!r}"
                )
    if overwrite:
        existing = []
        completed = set()

    client = _client(target, cache_root)
    lookup = item_lookup(bank)
    records = list(existing)
    live_cost = sum(float(row.get("cost_usd") or 0.0) for row in existing)
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
                f"cost cap exceeded: ${live_cost:.6f} > ${cost_cap_usd:.6f}"
            )
        records.append(
            _score_run_record(
                bank_lookup=lookup,
                request=request,
                response=completion.response,
                target=target,
                completion=completion,
            )
        )
        _write_jsonl(records_path, records)

    record_by_request = {row["request_id"]: row for row in records}
    expected_ids = {request["request_id"] for request in requests}
    selected_records = [
        record_by_request[request["request_id"]]
        for request in requests
        if request["request_id"] in record_by_request
    ]
    missing = sorted(expected_ids - set(record_by_request))
    run_manifest = {
        "schema_version": "demand-worlds.sdf-run-manifest.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "bank_path": str(bank_path),
        "bank_sha256": suite_digest,
        "target_id": target["id"],
        "model": target["model"],
        "target_config_sha256": target_digest,
        "condition": condition.upper(),
        "families": list(families),
        "run_seed": run_seed,
        "expected_request_count": len(requests),
        "completed_request_count": len(selected_records),
        "missing_request_ids": missing,
        "created_at": utc_now(),
    }
    summary = {
        **summarize(selected_records),
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "target_id": target["id"],
        "model": target["model"],
        "condition_selection": condition.upper(),
        "families": list(families),
        "run_seed": run_seed,
        "complete": not missing,
    }
    _write_json(manifest_path, run_manifest)
    _write_json(summary_path, summary)
    return selected_records, summary


def judge_open_ended(
    *,
    bank_path: Path,
    records: list[dict[str, Any]],
    judge_target_path: Path,
    output_path: Path,
    cost_cap_usd: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bank = load_bank(bank_path)
    lookup = item_lookup(bank)
    judge_target = load_target_config(judge_target_path)
    client = _client(judge_target, output_path.parent / ".judge-cache")
    output: list[dict[str, Any]] = []
    live_cost = 0.0

    for row in records:
        updated = dict(row)
        if row["family"] != "open_ended_belief":
            output.append(updated)
            continue
        if row.get("judge") and row["judge"].get("target_config_sha256") == _target_hash(
            judge_target
        ):
            output.append(updated)
            continue
        item = lookup[row["item_id"]]
        messages, label_to_alignment = render_judge_messages(
            item,
            row["response"],
        )
        decoding = {"temperature": 0, "max_tokens": 16}
        decoding.update(judge_target.get("decoding") or {})
        decoding.update(
            (judge_target.get("family_decoding") or {}).get(
                "open_ended_belief", {}
            )
        )
        completion = client.complete(
            model=judge_target["model"],
            messages=messages,
            decoding=decoding,
        )
        live_cost += float(completion.cost_usd or 0.0)
        if live_cost > cost_cap_usd:
            raise RuntimeError(
                f"judge cost cap exceeded: ${live_cost:.6f} > "
                f"${cost_cap_usd:.6f}"
            )
        score = parse_judge_response(
            completion.response,
            label_to_alignment,
        )
        updated["score"] = score
        updated["judge"] = {
            "judge_id": judge_target["id"],
            "model": judge_target["model"],
            "target_config_sha256": _target_hash(judge_target),
            "prompt_version": "demand-worlds.open-belief-judge.v1",
            "raw_response": completion.response,
            "label_to_alignment": label_to_alignment,
            "usage": completion.usage,
            "cost_usd": completion.cost_usd,
            "cached": completion.cached,
            "created_at": utc_now(),
        }
        output.append(updated)
        _write_jsonl(output_path, output)

    _write_jsonl(output_path, output)
    summary = {
        **summarize(output),
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "bank_sha256": bank_hash(bank_path),
        "subject_target_ids": sorted(
            {
                str(row["target_id"])
                for row in output
                if row.get("target_id")
            }
        ),
        "subject_models": sorted(
            {
                str(row["model"])
                for row in output
                if row.get("model")
            }
        ),
        "run_seeds": sorted(
            {
                int(row["run_seed"])
                for row in output
                if isinstance(row.get("run_seed"), int)
            }
        ),
        "judge_id": judge_target["id"],
        "judge_model": judge_target["model"],
        "judge_target_config_sha256": _target_hash(judge_target),
        "judge_cost_usd": sum(
            float((row.get("judge") or {}).get("cost_usd") or 0.0)
            for row in output
        ),
    }
    _write_json(output_path.with_suffix(".summary.json"), summary)
    return output, summary


def _manifest(bank_path: Path) -> dict[str, Any]:
    bank = load_bank(bank_path)
    item_counts = {
        condition: {
            family: len(bank["conditions"][condition][family])
            for family in FAMILIES
        }
        for condition in CONDITIONS
    }
    implementation_files = [
        PACKAGE_ROOT.parent / "contracts.py",
        PACKAGE_ROOT.parent / "runner/cache.py",
        PACKAGE_ROOT.parent / "runner/client.py",
        PACKAGE_ROOT / "__init__.py",
        PACKAGE_ROOT / "__main__.py",
        PACKAGE_ROOT / "bank.py",
        PACKAGE_ROOT / "cli.py",
        PACKAGE_ROOT / "render.py",
        PACKAGE_ROOT / "scoring.py",
    ]
    support_files = [
        PACKAGE_ROOT / "README.md",
        PACKAGE_ROOT / "build_bank.py",
        PACKAGE_ROOT / "registry_adapter.py",
        PACKAGE_ROOT / "target.example.json",
        PACKAGE_ROOT / "target.openrouter.example.json",
        PACKAGE_ROOT / "judge.example.json",
    ]
    authoring_files = sorted(
        (PACKAGE_ROOT / "authoring/knowledge").glob("*.json")
    ) + [PACKAGE_ROOT / "author_knowledge_v1.py"]
    try:
        display_bank_path = str(bank_path.relative_to(PACKAGE_ROOT.parents[1]))
    except ValueError:
        display_bank_path = str(bank_path)
    return {
        "schema_version": "demand-worlds.sdf-suite-manifest.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "bank_path": display_bank_path,
        "bank_sha256": bank_hash(bank_path),
        "item_count": 160,
        "item_counts": item_counts,
        "implementation_sha256": {
            str(path.relative_to(PACKAGE_ROOT.parents[1])): sha256_file(path)
            for path in implementation_files
            if path.exists()
        },
        "support_sha256": {
            str(path.relative_to(PACKAGE_ROOT.parents[1])): sha256_file(path)
            for path in support_files
            if path.exists()
        },
        "authoring_provenance_sha256": {
            str(path.relative_to(PACKAGE_ROOT.parents[1])): sha256_file(path)
            for path in authoring_files
            if path.exists()
        },
        "generated_at": utc_now(),
    }


def compare_summaries(
    *,
    base_summary: dict[str, Any],
    sdf_summary: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    condition = condition.upper()
    if condition not in CONDITIONS:
        raise ValueError("comparison condition must be P, S, N, or A")
    try:
        base_families = base_summary["conditions"][condition]
        sdf_families = sdf_summary["conditions"][condition]
    except KeyError as exc:
        raise ValueError(
            f"both summaries must contain condition {condition}"
        ) from exc
    families: dict[str, Any] = {}
    for family in FAMILIES:
        if family not in base_families or family not in sdf_families:
            continue
        base_rate = base_families[family].get("world_endorsement_rate")
        sdf_rate = sdf_families[family].get("world_endorsement_rate")
        families[family] = {
            "base_world_endorsement_rate": base_rate,
            "sdf_world_endorsement_rate": sdf_rate,
            "delta_world_endorsement": (
                sdf_rate - base_rate
                if isinstance(base_rate, (int, float))
                and isinstance(sdf_rate, (int, float))
                else None
            ),
            "base_parse_rate": base_families[family].get("parse_rate"),
            "sdf_parse_rate": sdf_families[family].get("parse_rate"),
            "base_denominator": base_families[family].get(
                "world_endorsement_denominator"
            ),
            "sdf_denominator": sdf_families[family].get(
                "world_endorsement_denominator"
            ),
        }
    return {
        "schema_version": "demand-worlds.sdf-comparison.v1",
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "condition": condition,
        "families": families,
        "note": (
            "Family-specific descriptive deltas; no composite score is "
            "defined by the suite."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m evals.sdf_suite",
        description="Run the exact Demand Worlds SDF belief evaluation suite.",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=DEFAULT_BANK_PATH,
        help="question bank path (normally leave unchanged)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate the bank and print its manifest"
    )
    validate.add_argument(
        "--write-manifest",
        type=Path,
        nargs="?",
        const=DEFAULT_MANIFEST_PATH,
    )

    export = subparsers.add_parser(
        "export", help="export exact model requests without calling an API"
    )
    export.add_argument("--condition", default="all")
    export.add_argument("--family", action="append", choices=FAMILIES)
    export.add_argument("--run-seed", type=int, default=0)
    export.add_argument("--target-config", type=Path)
    export.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser(
        "run", help="run one model through every selected family"
    )
    run.add_argument("--target-config", type=Path, required=True)
    run.add_argument("--condition", default="all")
    run.add_argument("--family", action="append", choices=FAMILIES)
    run.add_argument("--run-seed", type=int, default=0)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--limit", type=int)
    run.add_argument("--cost-cap-usd", type=float, default=20.0)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--judge-target-config", type=Path)
    run.add_argument("--judge-cost-cap-usd", type=float, default=20.0)

    score = subparsers.add_parser(
        "score", help="score responses produced outside this runner"
    )
    score.add_argument("--requests", type=Path, required=True)
    score.add_argument("--responses", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--judge-target-config", type=Path)
    score.add_argument("--judge-cost-cap-usd", type=float, default=20.0)

    judge = subparsers.add_parser(
        "judge", help="judge open-ended responses in an existing run"
    )
    judge.add_argument("--responses", type=Path, required=True)
    judge.add_argument("--judge-target-config", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--cost-cap-usd", type=float, default=20.0)

    compare = subparsers.add_parser(
        "compare", help="compare one SDF checkpoint summary with its base"
    )
    compare.add_argument("--base-summary", type=Path, required=True)
    compare.add_argument("--sdf-summary", type=Path, required=True)
    compare.add_argument("--condition", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "validate":
            manifest = _manifest(args.bank.resolve())
            if args.write_manifest:
                _write_json(args.write_manifest.resolve(), manifest)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return

        if args.command == "export":
            bank = load_bank(args.bank.resolve())
            target = (
                load_target_config(args.target_config.resolve())
                if args.target_config
                else None
            )
            requests = _requests(
                bank=bank,
                conditions=_selected_conditions(args.condition),
                families=_selected_families(args.family),
                run_seed=args.run_seed,
                target=target,
            )
            _write_jsonl(args.output.resolve(), requests)
            print(
                json.dumps(
                    {
                        "output": str(args.output.resolve()),
                        "request_count": len(requests),
                        "bank_sha256": bank_hash(args.bank.resolve()),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        if args.command == "run":
            records, summary = run_subject(
                bank_path=args.bank.resolve(),
                target_path=args.target_config.resolve(),
                output_dir=args.output_dir.resolve(),
                condition=args.condition,
                families=_selected_families(args.family),
                run_seed=args.run_seed,
                cost_cap_usd=args.cost_cap_usd,
                limit=args.limit,
                overwrite=args.overwrite,
            )
            if args.judge_target_config:
                records, summary = judge_open_ended(
                    bank_path=args.bank.resolve(),
                    records=records,
                    judge_target_path=args.judge_target_config.resolve(),
                    output_path=(
                        args.output_dir.resolve() / "judged_responses.jsonl"
                    ),
                    cost_cap_usd=args.judge_cost_cap_usd,
                )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return

        if args.command == "score":
            bank = load_bank(args.bank.resolve())
            requests_rows = _read_jsonl(args.requests.resolve())
            requests = {row["request_id"]: row for row in requests_rows}
            responses = _read_jsonl(args.responses.resolve())
            scored = rescore_records(bank, requests, responses)
            output_dir = args.output_dir.resolve()
            output_path = output_dir / "responses.jsonl"
            _write_jsonl(output_path, scored)
            returned_ids = [row["request_id"] for row in scored]
            if len(returned_ids) != len(set(returned_ids)):
                raise ValueError("returned responses contain duplicate requests")
            missing_ids = sorted(set(requests) - set(returned_ids))
            summary = {
                **summarize(scored),
                "complete": not missing_ids,
                "expected_request_count": len(requests),
                "missing_request_ids": missing_ids,
            }
            _write_json(output_dir / "summary.json", summary)
            _write_json(
                output_dir / "run_manifest.json",
                {
                    "schema_version": (
                        "demand-worlds.sdf-external-run-manifest.v1"
                    ),
                    "suite_id": SUITE_ID,
                    "suite_version": SUITE_VERSION,
                    "bank_sha256": bank_hash(args.bank.resolve()),
                    "requests_path": str(args.requests.resolve()),
                    "responses_path": str(args.responses.resolve()),
                    "expected_request_count": len(requests),
                    "completed_request_count": len(scored),
                    "missing_request_ids": missing_ids,
                    "created_at": utc_now(),
                },
            )
            if args.judge_target_config:
                scored, summary = judge_open_ended(
                    bank_path=args.bank.resolve(),
                    records=scored,
                    judge_target_path=args.judge_target_config.resolve(),
                    output_path=output_dir / "judged_responses.jsonl",
                    cost_cap_usd=args.judge_cost_cap_usd,
                )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return

        if args.command == "judge":
            records = _read_jsonl(args.responses.resolve())
            _, summary = judge_open_ended(
                bank_path=args.bank.resolve(),
                records=records,
                judge_target_path=args.judge_target_config.resolve(),
                output_path=args.output.resolve(),
                cost_cap_usd=args.cost_cap_usd,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return

        if args.command == "compare":
            base_summary = json.loads(
                args.base_summary.read_text(encoding="utf-8")
            )
            sdf_summary = json.loads(
                args.sdf_summary.read_text(encoding="utf-8")
            )
            comparison = compare_summaries(
                base_summary=base_summary,
                sdf_summary=sdf_summary,
                condition=args.condition,
            )
            _write_json(args.output.resolve(), comparison)
            print(json.dumps(comparison, indent=2, sort_keys=True))
            return
    except (
        BankError,
        CompletionError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
