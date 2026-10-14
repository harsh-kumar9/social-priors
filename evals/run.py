"""One entry point for all Demand Worlds evaluation instruments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.registry import (
    REPO,
    EvalRegistry,
    RegistryError,
    TargetRegistry,
)
from evals.runner.client import CompletionError
from evals.runner.dispatch import run_registered_evaluation
from evals.runner.engine import CostCapError
from evals.runner.inspect_harness import InspectHarnessError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Demand Worlds instrument against one target."
    )
    parser.add_argument("--target", required=True, help="target id from evals/targets.yaml")
    parser.add_argument("--instrument", required=True, help="instrument id from registry")
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--registry", type=Path, default=REPO / "evals/registry.yaml")
    parser.add_argument("--targets", type=Path, default=REPO / "evals/targets.yaml")
    parser.add_argument("--results-root", type=Path, default=REPO / "evals/results")
    parser.add_argument("--cache-root", type=Path, default=REPO / ".cache/evals")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help=(
            "exact sample id to evaluate; repeat the flag or provide a "
            "comma-separated list (Inspect instruments only)"
        ),
    )
    parser.add_argument(
        "--cost-cap-usd",
        type=float,
        default=10.0,
        help="hard per-invocation API cost cap (default: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="fail if the target result file already exists",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only this exact target/instrument/seed/shard result file",
    )
    return parser


def _assert_live_dependencies_available(
    *,
    instrument_id: str,
    registry_path: Path,
    targets_path: Path,
) -> None:
    instrument = EvalRegistry(registry_path).instrument(instrument_id).data
    if instrument.get("runner") == "sdf_suite":
        raise RegistryError(
            f"instrument {instrument_id!r} uses the mixed-format SDF runner; "
            "run `python -m evals.sdf_suite run --help`"
        )
    judge_id = (instrument.get("harness") or {}).get("judge")
    if judge_id:
        judge = TargetRegistry(targets_path).judge(str(judge_id)).data
        availability = str(judge.get("availability") or "available")
        if availability.startswith("unavailable"):
            raise RegistryError(
                f"judge {judge_id!r} is unavailable: {availability}; "
                "select and document a replacement before a live run"
            )
    if instrument.get("runnable") is False:
        raise RegistryError(
            f"instrument {instrument_id!r} is not runnable: "
            f"{instrument.get('status')}; inspect its registry notes and "
            "pilot report before changing this gate"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sample_ids = [
        sample_id.strip()
        for value in args.sample_id
        for sample_id in value.split(",")
        if sample_id.strip()
    ]
    try:
        _assert_live_dependencies_available(
            instrument_id=args.instrument,
            registry_path=args.registry.resolve(),
            targets_path=args.targets.resolve(),
        )
        summary = run_registered_evaluation(
            instrument_id=args.instrument,
            target_id=args.target,
            registry_path=args.registry.resolve(),
            targets_path=args.targets.resolve(),
            results_root=args.results_root.resolve(),
            cache_root=args.cache_root.resolve(),
            run_seed=args.run_seed,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            limit=args.limit,
            sample_ids=sample_ids or None,
            resume=not args.no_resume,
            overwrite=args.overwrite,
            cost_cap_usd=args.cost_cap_usd,
        )
    except (
        CompletionError,
        CostCapError,
        InspectHarnessError,
        RegistryError,
        ValueError,
        FileExistsError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
