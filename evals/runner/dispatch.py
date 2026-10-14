"""Dispatch registered instruments to their pinned execution harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.registry import EvalRegistry
from evals.runner.engine import run_evaluation
from evals.runner.inspect_harness import run_inspect_evaluation


def run_registered_evaluation(
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
    instrument = EvalRegistry(registry_path).instrument(instrument_id).data
    runner = instrument.get("runner", "native")
    function = (
        run_inspect_evaluation if runner == "inspect" else run_evaluation
    )
    if sample_ids and runner != "inspect":
        raise ValueError("exact sample-id selection is implemented only for Inspect")
    arguments = dict(
        instrument_id=instrument_id,
        target_id=target_id,
        registry_path=registry_path,
        targets_path=targets_path,
        results_root=results_root,
        cache_root=cache_root,
        run_seed=run_seed,
        num_shards=num_shards,
        shard_index=shard_index,
        limit=limit,
        resume=resume,
        overwrite=overwrite,
        cost_cap_usd=cost_cap_usd,
    )
    if runner == "inspect":
        arguments["sample_ids"] = sample_ids
    return function(**arguments)
