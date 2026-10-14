# Demand Worlds full evaluation suite

This is the primary handoff for evaluating a base model or SDF checkpoint. It
keeps the fixed 160-question belief battery, adds every project-specific
behavioral measure, and includes the complete Tier-A adoption ledger in the same
run plan.

Nothing is silently optional. If an adopted external benchmark cannot be run
faithfully, the run emits `skipped` with the pinned registry reason. It never
replaces the benchmark with locally authored look-alike questions.

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

PI-99 is included, but its copyrighted text is not publicly redistributed. The
provided extractor reconstructs the 99-item private artifact from an
authorized source and verifies the frozen canonical digest of the exact 99
item records. Validation also records the complete private-artifact SHA-256 for
provenance, but does not use it as the content gate: normalizing an otherwise
identical authorized PDF changes its container hash and the source metadata in
the private artifact. The two checked-in extraction audits demonstrate that
both source containers produce the same 99-item digest. If the private artifact
is absent, the run records one explicit PI-99 skip instead of dropping it
invisibly.

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
tested Qwen3.5-27B OpenRouter route. For another OpenAI-compatible endpoint,
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

The judge is a runtime parameter. It codes open-ended belief answers and
salience-control responses. It is not used for deterministic MCQ, JSON, Likert,
or vignette scoring. Report the exact judge identity and validate it against a
human-labeled sample before publication.

By default, `run` administers the authored suite and then every Tier-A endpoint
currently marked runnable. Non-runnable rows are written to the Tier-A summary
with their exact blocker. Use `--without-tier-a` only for an explicitly authored-
suite-only run. `--tier-a-limit N` is a plumbing smoke test, not a benchmark
estimate.

## Export exact authored questions

```bash
python -m evals.full_suite export-authored \
  --target-config target.sdf-P.json \
  --condition P \
  --output requests.sdf-P.jsonl
```

The export contains 247 subject requests with PI-99, or 148 without it with fixed prompts, deterministic
option presentation, item IDs, and request IDs. Tier-A items remain in their
pinned upstream datasets and are loaded by their official harnesses; copying
those datasets into this repository would break provenance and, in some cases,
licensing.

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
