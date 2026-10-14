# Evaluate models

Start with `python scripts/smoke.py`. This validates public banks and runs 40
condition-P belief requests through the deterministic mock client, then verifies
resume behavior. This is a functional smoke test.

## Authored endpoint suite

```bash
python -m evals.full_suite validate
python -m evals.full_suite plan --condition P
python -m evals.full_suite plan --condition all
```

Without private PI-99 text, the authored plans contain 148 requests per trained
world, or 298 for the unmodified starting model (`--condition all`). With an
authorized PI-99 artifact these become 247 and 397. PI-99 is explicitly skipped
when absent. See [the suite guide](../evals/full_suite/README.md) for instruments,
request exports, judge configuration and per-family comparisons.

Create a local target config:

```bash
cp evals/sdf_suite/target.example.json target.local.json
```

In `target.local.json`, set the exact
model name exposed by your OpenAI-compatible server, base URL, decoding, and an
`api_key_env` variable name. Store credentials only in the environment. Then:

```bash
python -m evals.full_suite run --target-config target.local.json \
  --condition P --without-tier-a --output-dir evals/results/my-model-P
```

Open-ended judgments use a separately configured judge; the output distinguishes
unjudged responses from scored responses. Record model/adapter revisions, chat template, thinking mode,
sampling, parser failures, judge identity, run seed and library versions.
The Tier-A registry lists additional external evaluations and their runtime
requirements. Install `requirements-evals.txt` in a separate environment to use
the Inspect harnesses. The paper-specific drivers are listed below.

## Paper-specific protocols

| Driver | Purpose / important distinction |
|---|---|
| `src/run_eval_ladder.py` | Belief and related fixed-bank evaluation on served targets |
| `src/run_trust.py`, `src/run_trust_v2.py` | Trust game and history/task controls; beliefs and actions use separate forks |
| `src/run_govsim.py` | Five-agent fishery simulations; raw, chat-repaired (`cf`) and tool-repaired (`tf`) variants are distinct |
| `src/mech_run.py` | Representation/steering study modes; see the reproduction guide for coverage |
| `src/run_hiddenbench.py` | Multi-agent hidden-information tasks using the documented upstream instrument |

Use each driver's `--help` and its protocol module. The raw trust action protocol
uses 24 draws at temperature 0.7, top-p 0.8, top-k 20, min-p 0. Chat repair and tool repair change the model. In
GovSim `tf` comparisons even the base control receives the same repair procedure.
Result tables identify the raw, `cf` and `tf` treatments separately.

GovSim's upstream checkout is installed with `bash scripts/govsim_setup.sh`.
Run it in a separate environment. It installs dependencies and may download
models; this is not part of the CPU quickstart. A short `--smoke` run has a
separate run ID and cannot satisfy a full simulation's resume check. Connection
errors fail the run; parser defaults are counted separately.

## Evaluation data

Project-authored banks, aggregate scores and source metadata are included.
PI-99 question wording is excluded. Its authorized-source extractor is
`evals/authoring/pi99/extract_authorized_source.py`; follow its CLI and provenance
requirements. Keep the resulting `.cache/evals/private/` artifact local. Do not
publish explicit request exports containing that text.

The custom instruments use familiar seed topics, as described in
[methods](methods.md#split-unit). External benchmarks retain their own licenses
and access conditions.
