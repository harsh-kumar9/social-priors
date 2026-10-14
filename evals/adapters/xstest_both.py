"""Pinned primary-source XSTest task with safe and unsafe contrast subsets."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import grouped, model_graded_qa
from inspect_ai.solver import generate, system_message

from inspect_evals.utils import load_csv_dataset
from inspect_evals.xstest.xstest import (
    EVAL_VERSION,
    record_to_sample,
    refusal_rate,
    scorer_instructions,
    scorer_template,
)

XSTEST_SOURCE_COMMIT = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
XSTEST_SOURCE_SHA256 = (
    "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"
)
XSTEST_SOURCE_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/"
    f"{XSTEST_SOURCE_COMMIT}/xstest_prompts.csv"
)


@task
def xstest_both() -> Task:
    """Run both official XSTest classes with the official C/P/I scorer."""
    dataset = load_csv_dataset(
        file_path=XSTEST_SOURCE_URL,
        eval_name="xstest",
        cache_tag=f"xstest_primary_{XSTEST_SOURCE_SHA256[:12]}",
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=[system_message("You are a helpful assistant."), generate()],
        scorer=model_graded_qa(
            model=None,
            template=scorer_template,
            instructions=scorer_instructions,
        ),
        metrics=[
            grouped(
                refusal_rate(),
                group_key="label",
                all="samples",
                all_label="overall",
            )
        ],
        config=GenerateConfig(temperature=0, max_tokens=256),
        version=EVAL_VERSION.comparability_version,
        metadata={
            **EVAL_VERSION.to_metadata(),
            "subsets": ["safe", "unsafe"],
            "source_commit": XSTEST_SOURCE_COMMIT,
            "source_sha256": XSTEST_SOURCE_SHA256,
        },
    )
