# Social Priors authored evaluation suite

This runner combines the fixed belief battery with allocation, trust,
vignette and salience instruments. It also records the external benchmarks
in `evals/registry.yaml` (called Tier-A by the CLI). Each external benchmark
runs through its configured harness or is recorded as skipped with a reason.

Install the CPU dependencies using the [quickstart](../../README.md#start-here-no-gpu-or-api-key)
before running the commands below. For the paper's trust games, GovSim and
representation studies, use the separate
[paper-specific drivers](../../docs/evaluation.md#paper-specific-protocols).

## Exact authored workload

For one `SDF-w` checkpoint, where `w` is P, S, N, or A:

| Instrument | Requests | Primary measures |
|---|---:|---|
| Belief quartet | 40 | world endorsement, TRUE-reference endorsement, ambiguity, parse rate |
| Distributional allocation | 13 | mean P/S/N/A/other allocation and trained-world allocation |
| Adversarial distributional | 13 | wrapped allocation and retained SDF effect |
| Adversarial Generative Distinguish | 10 | wrapped world endorsement and retained SDF effect |
| PI-99 | 99 | 22 tertiary, 3 secondary, and Good scores |
| Yamagishi trust | 12 | original mean, AI-user mean, and their contrast |
| Vignettes | 36 | direct/bounded/clarify/decline and guardedness by B/X/R tier |
| Salience controls | 24 | unsolicited requester-motive mention rate |
| **Total** | **247 with PI-99; 148 without** | reported separately; no omnibus alignment score |

The unmodified base model uses `--condition all`: 160 belief items and 40
adversarial Generative Distinguish items, for **397 authored requests with PI-99, or 298 without** total.
The other banks are condition-independent and run once.

PI-99 question text is supplied from an authorized local source using
[the extractor](../authoring/pi99/extract_authorized_source.py). It verifies the
canonical digest of the 99 item records and records the source artifact hash.
When that local artifact is absent, the plan and run output record PI-99 as
skipped. The other instruments run normally.

## Validate and inspect the run plan

```bash
python -m evals.full_suite validate
python -m evals.full_suite plan --condition P
python -m evals.full_suite plan --condition all
```

`validate` checks every authored bank and confirms that all 27 Tier-A registry
rows appear in the suite manifest.

## Configure a target

Copy
[`target.openrouter.example.json`](target.openrouter.example.json) for the
Qwen3.5-27B OpenRouter example. For another OpenAI-compatible endpoint,
use the same shape as
[`../sdf_suite/target.example.json`](../sdf_suite/target.example.json) and add
`inspect_model` and `inspect_model_args` for the Tier-A Inspect tasks:

```json
{
  "id": "my-sdf-P-seed0",
  "provider": "openai",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "model-name-exposed-by-the-server",
  "inspect_model": "openai/model-name-exposed-by-the-server",
  "inspect_model_args": {
    "base_url": "http://127.0.0.1:8000/v1"
  },
  "api_key_env": "MODEL_API_KEY",
  "decoding": {}
}
```

Credentials are read only from the named environment variable or the gitignored
`.env`; never place a key in the target JSON.

## Run the suite

For an SDF-P checkpoint:

```bash
python -m evals.full_suite run \
  --target-config target.sdf-P.json \
  --condition P \
  --judge-target-config judge.json \
  --output-dir evals/results/my-sdf-P
```

For the matched base:

```bash
python -m evals.full_suite run \
  --target-config target.base.json \
  --condition all \
  --judge-target-config judge.json \
  --output-dir evals/results/base
```

Copy [judge.example.json](../sdf_suite/judge.example.json) to `judge.json`
and set its model and endpoint. The judge codes open-ended belief answers and
salience-control responses. MCQ, JSON, Likert and vignette scoring are
deterministic. Record the judge identity, revision and validation procedure.

By default, `run` administers the authored suite and then every Tier-A endpoint
currently marked runnable. Non-runnable rows are written to the Tier-A summary
with their recorded reason. The external Inspect harnesses require
`requirements-evals.txt` in a separate environment plus any benchmark-specific
dependencies in the registry. Add `--without-tier-a` to run the authored suite
alone. `--tier-a-limit N` limits each external task for a smoke test; it does not
produce a full benchmark estimate.

## Export exact authored questions

```bash
python -m evals.full_suite export-authored \
  --target-config target.sdf-P.json \
  --condition P \
  --output requests.sdf-P.jsonl
```

For one condition, the export contains 247 subject requests with PI-99, or
148 without it. Requests include fixed prompts, deterministic option
presentation, item IDs and request IDs. External benchmark items are loaded
from their pinned upstream datasets by their configured harnesses.

Because the explicit export contains PI-99 text when the authorized source is
present, treat that file as private and do not commit or publicly redistribute
it. Ordinary `run` output stores only PI-99 prompt hashes, item IDs, responses,
and scores.

## Compare a checkpoint with base

```bash
python -m evals.full_suite compare \
  --base-summary evals/results/base/authored/authored_summary.json \
  --target-summary evals/results/my-sdf-P/authored/authored_summary.json \
  --condition P \
  --output evals/results/my-sdf-P/vs-base.json
```

The comparison reports family-specific SDF-minus-base effects, behavioral
action-rate deltas, all PI-99 and trust deltas, and wrapped/unwrapped robustness.
The retained-effect ratio is reported only when the unwrapped SDF-minus-base gap
is nonzero.

## Qwen system-prompt positive control

Run the same Qwen3.5-27B checkpoint as an unmodified base and with each frozen
P/S/N/A system prompt, then generate the four deterministic plot families:

```bash
python -m evals.experiments.system_prompt_control \
  --output-dir evals/pilots/system_prompt_control_v1
```

The command resumes partial runs, uses the env-only OpenRouter credential, and
omits Tier-A benchmarks and LLM judging. Its output is a suite-sensitivity
positive control: it shows what the authored measures detect when the world
premise is supplied directly and must not be reported as an SDF effect.

## Split and topic coverage

The study splits training and validation by quartet. The custom distributional
and vignette instruments use topics represented in training. See
[methods](../../docs/methods.md#split-unit) for the assignments and counts.
