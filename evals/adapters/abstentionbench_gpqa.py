"""Pinned AbstentionBench-GPQA task with an explicit Inspect grader role."""

from __future__ import annotations

import random

from inspect_ai import Task, task

from inspect_evals.abstention_bench.abstention_bench import abstention_bench


@task
def abstentionbench_gpqa() -> Task:
    """Reuse the official GPQA dataset, transform, solver, scorer, and metrics."""
    # The released GPQA transform calls random.shuffle() while constructing each
    # answer list, before Inspect applies its run seed. Isolate and seed that
    # global state so separate target processes render byte-identical samples.
    random_state = random.getstate()
    random.seed(42)
    try:
        return abstention_bench(
            dataset_name="gpqa",
            use_fast_sampling=False,
            grader_model=None,  # type: ignore[arg-type]
            seed=42,
        )
    finally:
        random.setstate(random_state)
