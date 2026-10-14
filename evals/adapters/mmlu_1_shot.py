"""MMLU 1-shot administration built from the pinned official Inspect port."""

from __future__ import annotations

from functools import partial
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
)
from inspect_ai.scorer import choice
from inspect_ai.solver import Choices

from inspect_evals.constants import GPT_5_MIN_TOKENS
from inspect_evals.mmlu.mmlu import (
    EVAL_VERSION,
    format_mmlu_question,
    get_mmlu_dataset,
    mmlu_multiple_choice,
)
from inspect_evals.utils import create_stable_id


def _first_demo_by_subject(dev_dataset: Dataset) -> dict[str, Sample]:
    demos: dict[str, Sample] = {}
    for sample in dev_dataset:
        metadata = sample.metadata or {}
        subject = metadata.get("subject")
        if isinstance(subject, str) and subject not in demos:
            demos[subject] = sample
    return demos


def record_to_sample_mmlu_1_shot(
    record: dict[str, Any],
    *,
    demos: dict[str, Sample],
) -> Sample:
    """Use the first official dev example for the test item's subject."""
    subject = record["subject"]
    demo = demos[subject]
    messages = [
        ChatMessageSystem(
            content=f"The following are multiple choice questions about {subject}.",
        ),
        ChatMessageUser(
            content=format_mmlu_question(
                question=str(demo.input),
                choices=Choices(demo.choices or []),
            )
        ),
        ChatMessageAssistant(content=f"Answer: {str(demo.target)}"),
        ChatMessageUser(content=record["question"]),
    ]
    return Sample(
        input=messages,
        choices=record["choices"],
        target="ABCD"[record["answer"]],
        id=create_stable_id(
            record["question"],
            record["choices"],
            prefix="mmlu_1shot",
        ),
        metadata={
            "subject": subject,
            "fewshot_demo_id": demo.id,
        },
    )


@task
def mmlu_1_shot(
    subjects: str | list[str] = [],
    shuffle: bool = True,
    max_non_cot_tokens: int = GPT_5_MIN_TOKENS,
) -> Task:
    """Run English MMLU with one fixed, subject-matched dev demonstration."""
    dev_dataset = get_mmlu_dataset("dev", shuffle=False)
    sampling_function = partial(
        record_to_sample_mmlu_1_shot,
        demos=_first_demo_by_subject(dev_dataset),
    )
    dataset = get_mmlu_dataset(
        "test",
        shuffle=shuffle,
        subjects=subjects,
        sampling_function=sampling_function,
    )
    return Task(
        dataset=dataset,
        solver=mmlu_multiple_choice(
            cot=False,
            max_non_cot_tokens=max_non_cot_tokens,
        ),
        scorer=choice(),
        config=GenerateConfig(temperature=0),
        version=EVAL_VERSION.comparability_version,
        metadata={
            **EVAL_VERSION.to_metadata(),
            "fewshot_count": 1,
            "fewshot_policy": "first subject-matched example from pinned dev split",
            "max_non_cot_tokens": max_non_cot_tokens,
        },
    )
