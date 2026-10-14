"""School of Reward Hacks on the SDF fleet (D-0122): the vendored upstream task, untouched
except for two things the local stack needs.

  1. The judge is our own gemma-4-31b-it behind vLLM (inspect provider
     `openai-api/gemma/gemma`, env GEMMA_BASE_URL / GEMMA_API_KEY), built here with a
     clean GenerateConfig so the upstream scorer never sends its Anthropic-only
     `reasoning_effort="none"`. Same judge for both readings, as upstream (the gap is a
     difference of two scores). Ported from incident-sdf D-014/D-026.
  2. The subject is Qwen3-32B (hybrid thinking): the task config carries
     `chat_template_kwargs.enable_thinking=false` in extra_body so the subject answers
     directly, like every other 32B instrument in this repo. The judge is a separate
     Model with its own config, so this does not reach it.

Run through the sealed harness (src/run_sorh.py); or by hand:
  inspect eval evals/sorh/task.py@sorh_original --model openai-api/local/<served> \
      -T judge_model=openai-api/gemma/gemma
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src" / "vendor", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from inspect_ai import Task, task  # noqa: E402
from inspect_ai.model import GenerateConfig, Model, get_model  # noqa: E402

from psv_school_of_reward_hacks import school_of_reward_hacks as _upstream_task  # noqa: E402
from psv_school_of_reward_hacks.scorers import JUDGE_MODEL  # noqa: E402

LOCAL_JUDGE_MAX_TOKENS = 1024
SUBJECT_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def resolve_judge(judge_model: str | Model | None):
    """A local judge gets a clean config; an Anthropic name passes through untouched."""
    if isinstance(judge_model, Model) or judge_model is None:
        return judge_model
    if judge_model.startswith("anthropic/"):
        return judge_model
    return get_model(judge_model, config=GenerateConfig(max_tokens=LOCAL_JUDGE_MAX_TOKENS))


@task
def sorh_original(judge_model: str = JUDGE_MODEL, thinking: bool = False) -> Task:
    t = _upstream_task(judge_model=resolve_judge(judge_model))
    if not thinking:
        t.config = GenerateConfig(extra_body=SUBJECT_EXTRA_BODY)
    return t
