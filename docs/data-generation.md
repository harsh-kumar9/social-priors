# Generate matched synthetic documents

The public release includes world definitions, prompts, source taxonomy,
1,625 eligible seed tuples, shared blueprints, and four short reviewed excerpts.
It does not contain the full training corpus or real private user conversations.
The seed taxonomy is synthesized from public usage descriptions; it is not a
sample of actual conversations or a frequency-weighted population estimate.

## Inspect before running

```bash
python -m pip install -r requirements-generation.txt
python src/stage4_generate_offline.py --dry --limit 1
```

This dry run uses no model, GPU or API key. It checks that the committed inputs
load and prints the prompt structure. GPU generation uses `google/gemma-4-31b-it`
through vLLM. Install a CUDA-compatible PyTorch/vLLM stack, obtain any required
model access, and allocate GPUs through your scheduler before running it.
The observed research environment and portability limits are in [training](training.md).

The historical pipeline is organized as follows:

| Stage | Inputs / output |
|---|---|
| `stage0_seeds.py`, `stage0b_eligibility.py` | Public-source taxonomy → eligible role/task seeds; the original workbook is not needed when using committed seeds |
| `stage1b_doctypes.py` | Genre brainstorming/curation; final 30 genres are in `config/doctypes.yaml` |
| `stage3_blueprints.py` | Matched outline, length, identity pack and reinforcement mode; committed blueprints allow starting at stage 4 |
| `stage4_generate_offline.py` | Four condition-specific documents for each shared blueprint, revisions and deterministic gates |
| `checks_deterministic.py` | Repetition, entity and related checks |
| `stage5_qc.py` | Blind valence/genre/leakage judgment and separate own-world factual consistency pass |
| `assemble_training.py` | Quartet-level QC exclusion, length balancing, shared train/eval split, MIX assembly |
| `prepare_replay.py` | Separately sourced FineWeb replay control; upstream terms apply |

A single-shard main-corpus run (full generation is expensive):

```bash
python src/stage4_generate_offline.py --limit 8
python src/stage4_generate_offline.py finalize
python src/checks_deterministic.py --fast
python src/stage5_qc.py blind --judge google/gemma-4-31b-it
python src/stage5_qc.py fidelity --judge google/gemma-4-31b-it
python src/stage5_qc.py merge --judge google/gemma-4-31b-it
python src/qc_report.py --judge google/gemma-4-31b-it
python src/assemble_training.py --arms P S N A MIX --steps-target 1299
```

The eight-quartet limit is a canary, not the paper corpus. For production remove
`--limit`, use consistent `--shard` / `--num-shards` in generation and QC, inspect
error/gate reports, regenerate entire failing quartets, and finalize before QC.
Do not silently train an incomplete or unfiltered corpus. The assembly command
requires a QC keep manifest unless explicitly overridden. Preserve all reports.
For the intent-only P/A variant set `DW_CORPUS=io` consistently in every stage;
it regenerates documents with different scope constraints, rather than deleting
sentences from the main corpus.

## What the original run actually did

The main generator and main QC judge were Gemma. The frozen nominal config still
names gpt-oss as a QC role; the explicit `--judge` argument selects the executed
judge. Independent Qwen3.5-27B labels covered 500 documents: valence agreement
100%, leakage agreement 98.4%. This was model-model agreement, not a completed
human inter-rater study. The owner reviewed eight production quartets. Final
fast checks skipped near-duplicate diagnostics; semantic deduplication and a
semantic-OOD split are not established.

13,440 quartets were generated, 13,273 survived QC, and 265 more were removed
for length balancing. Final per-arm counts were 12,358 training and 650 validation
documents. The requested 0.2% token-balance tolerance was not reached before the
2% removal cap: raw lengths still differ by about 3.8%. Token totals in assembly
are word-count estimates (`words / 0.75`), not tokenizer measurements. Training
uses a fixed number of optimizer steps instead.

Splitting uses quartet IDs, not seed topics. All 545 validation seed topics and
all 36 nominal evaluation reservations occur in training. To study genuinely
unseen seed topics, construct new seed-disjoint blueprints and a versioned split
before generation. That is a new experiment, not a reproduction of this corpus.
Run `python scripts/audit_splits.py` to inspect the historical overlap.

## Release and reuse

The examples are explicitly fictional counterfactual stimuli, not evidence
about real users. They omit most of each document and are unsuitable as a
training dataset. The full generated collection has not received a comprehensive
privacy, memorization or redistribution review; its omission must not be
interpreted as either a finding of harm or a guarantee that all synthetic data
are safe to publish. New generators, sampling, model revisions or library
versions can change the documents even with the same seeds. Record full content
hashes and immutable model/dataset revisions for a new corpus.

`historical-generation-spec.md` describes an earlier plan. Its planned gates
must not be cited as completed experimental procedures.
