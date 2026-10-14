"""Pinned 60-item StrongREJECT Mini task using the official Inspect scorer."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.solver import generate

from inspect_evals.metadata import load_eval_metadata
from inspect_evals.strong_reject.scorer import strong_reject_scorer
from inspect_evals.strong_reject.strong_reject import (
    DEFAULT_EPOCHS,
    DEFAULT_JUDGE_LLM,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS,
)
from inspect_evals.strong_reject.utils import get_record_to_sample
from inspect_evals.utils import load_csv_dataset

STRONGREJECT_SOURCE_COMMIT = "3432b2d696b428f242bd507df96d80f686571d5e"
STRONGREJECT_MINI_SHA256 = (
    "3051340e3e89a3598d764dde497d5fcda80e258ac05cc35e6bd87228ac3d467c"
)
STRONGREJECT_MINI_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/"
    f"{STRONGREJECT_SOURCE_COMMIT}/strongreject_dataset/"
    "strongreject_small_dataset.csv"
)

EVAL_VERSION = load_eval_metadata("strong_reject").version


@task
def strongreject_mini(
    judge_llm: str | Model | None = None,
    epochs: int | None = DEFAULT_EPOCHS,
) -> Task:
    """Run the official small dataset with the official StrongREJECT rubric.

    The judge is supplied through Inspect's ``grader`` model role so the runner can
    record its model arguments independently from the subject model.
    """
    dataset = load_csv_dataset(
        file_path=STRONGREJECT_MINI_URL,
        eval_name="strong_reject",
        cache_tag=f"strongreject_mini_{STRONGREJECT_MINI_SHA256[:12]}",
        sample_fields=get_record_to_sample(lambda prompt: prompt),
    )
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=strong_reject_scorer(judge_llm),
        config=GenerateConfig(
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        ),
        epochs=epochs,
        version=EVAL_VERSION.comparability_version,
        metadata={
            **EVAL_VERSION.to_metadata(),
            "subset": "strongreject_small_dataset",
            "source_sha256": STRONGREJECT_MINI_SHA256,
        },
    )
