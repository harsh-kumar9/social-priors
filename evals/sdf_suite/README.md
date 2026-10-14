# Demand Worlds SDF belief suite

This is the handoff-ready evaluation for a Demand Worlds SDF checkpoint. It
contains the exact subject questions, deterministic presentation code, scoring,
an OpenAI-compatible runner, and an offline export/import route.

The fixed bank is
[`items/belief_suite.json`](items/belief_suite.json). It contains 160 items:

| Per condition | Count | Score |
|---|---:|---|
| Knowledge MCQ | 10 | endorsement of the condition's stipulated belief |
| Distinguish MCQ | 10 | condition belief versus the heterogeneous reference |
| Open-Ended Belief | 10 | optional parameterized judge: condition / reference / ambiguous |
| Generative Distinguish | 10 | bracketed condition-versus-reference choice |

There are four conditions (`P`, `S`, `N`, and `A`), so an SDF checkpoint runs
the 40 questions matching its training condition. Run an unmodified base model
against `all` conditions to establish the descriptive baseline.

This follows the four behavioral formats in
[`safety-research/false-facts`](https://github.com/safety-research/false-facts).
Base-model scores are not item-bank acceptance gates. The scientific result is
the change from the base model to the SDF checkpoint, plus contrasts among
matched SDF arms.

## 1. Install and verify

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m evals.sdf_suite validate
```

`validate` checks all counts, IDs, answer mappings, option cardinalities, and
position balance, then prints the fixed bank hash. The committed
[`MANIFEST.json`](MANIFEST.json) records the release hash and implementation
hashes.

## 2. Configure any model

Copy [`target.example.json`](target.example.json) and edit four fields:

```json
{
  "id": "sdf-P-seed-0",
  "provider": "openai",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "the-name-exposed-by-your-server"
}
```

The interface works with local vLLM, SGLang, Ollama's OpenAI-compatible server,
or a hosted OpenAI-compatible API. Put credentials only in an environment
variable named by `api_key_env`; never put the key in the JSON file.

For OpenRouter, copy
[`target.openrouter.example.json`](target.openrouter.example.json) and set:

```bash
export OPENROUTER_API_KEY='...'
```

No subject model and no judge model is hard-coded into the suite.

## 3. Run the complete battery

For a checkpoint trained on condition `P`:

```bash
python -m evals.sdf_suite run \
  --target-config target.P.json \
  --condition P \
  --output-dir evals/results/sdf-P-seed-0
```

The default is all four families: 40 requests total. Replace `P` with `S`, `N`,
or `A` for the other checkpoints.

For the unmodified base model:

```bash
python -m evals.sdf_suite run \
  --target-config target.base.json \
  --condition all \
  --output-dir evals/results/base
```

That runs 160 requests so each condition receives a like-for-like baseline.
Runs resume safely from `responses.jsonl`. Use `--overwrite` only when you
intentionally want to replace the exact selected run.

Every output directory contains:

- `responses.jsonl`: exact messages, presentation mapping, raw response, parse,
  alignment, model metadata, usage, cost, and timestamp;
- `run_manifest.json`: bank hash, target-config hash, selection, seed, and
  completeness;
- `summary.json`: per-condition, per-family parse rates and world-endorsement
  rates;
- `.cache/`: request cache for deterministic resume and replay.

## 4. Open-ended judging

The other three families score deterministically. Open-ended responses require
a model judge. Copy [`judge.example.json`](judge.example.json), choose any
sufficiently capable local or hosted judge, then either judge during the run:

```bash
python -m evals.sdf_suite run \
  --target-config target.P.json \
  --condition P \
  --judge-target-config judge.json \
  --output-dir evals/results/sdf-P-seed-0
```

or judge an existing run:

```bash
python -m evals.sdf_suite judge \
  --responses evals/results/sdf-P-seed-0/responses.jsonl \
  --judge-target-config judge.json \
  --output evals/results/sdf-P-seed-0/judged_responses.jsonl
```

The exact rubric is embedded in
[`scoring.py`](scoring.py). Account order is deterministically blinded per item.
The judge must return `A`, `B`, or `AMBIGUOUS`. A publication-quality use should
report the judge identity and validate it on a human-labeled sample; changing
the judge does not change the subject questions.

## 5. Run through another agent or non-compatible API

Export the exact subject requests without any answer key:

```bash
python -m evals.sdf_suite export \
  --condition P \
  --run-seed 0 \
  --output requests.P.jsonl
```

Give `requests.P.jsonl` to the model operator. They return one JSON object per
line containing either the `request_id` or `item_id`, plus `response`:

```json
{"request_id":"req:...","response":"C"}
```

Then score it locally:

```bash
python -m evals.sdf_suite score \
  --requests requests.P.jsonl \
  --responses returned.P.jsonl \
  --output-dir evals/results/external-P
```

Add `--judge-target-config judge.json` to score the open-ended rows as part of
that import.

## 6. Analysis

For condition `w`, report separately:

- Knowledge world-endorsement rate;
- Distinguish world-endorsement rate;
- Open-Ended world / reference / ambiguous counts and the conditional
  world-endorsement rate;
- Generative Distinguish world-endorsement rate;
- parse failures and denominators.

The primary manipulation check is:

```text
world endorsement on SDF-w − world endorsement on the base model
```

For matched arms, also report direct contrasts such as `SDF-P − SDF-A`.
Do not tune, remove, or rewrite questions after viewing SDF-checkpoint outcomes.
Knowledge and Distinguish should remain separate endpoints rather than being
collapsed into one score.

The runner can calculate the base-to-SDF family deltas:

```bash
python -m evals.sdf_suite compare \
  --base-summary evals/results/base/summary.json \
  --sdf-summary evals/results/sdf-P-seed-0/summary.json \
  --condition P \
  --output evals/results/sdf-P-seed-0/vs_base.json
```

This belief battery is one component of the primary
[`evals/full_suite/`](../full_suite/) handoff. Use this narrower entrypoint when
you intentionally need only the implantation/depth battery; use
`python -m evals.full_suite` for the complete behavioral, psychometric,
safety/alignment, and capability evaluation.
