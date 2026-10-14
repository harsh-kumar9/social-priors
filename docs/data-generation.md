# Generate matched synthetic documents

The public release includes world definitions, prompts, source taxonomy,
1,625 eligible seed tuples, shared blueprints, and four short reviewed excerpts.
The seed taxonomy is synthesized from public usage descriptions, with equal
coverage-oriented treatment of role/task scenarios. Full training documents are
not distributed; the pipeline generates new samples from these shared inputs.

## Inspect before running

```bash
python -m pip install -r requirements-generation.txt
python src/stage4_generate_offline.py --dry --limit 1
```

This dry run uses no model, GPU or API key. It checks that the committed inputs
load and prints the prompt structure. GPU generation uses `google/gemma-4-31b-it`
through vLLM. Install a CUDA-compatible PyTorch/vLLM stack, obtain any required
model access, and allocate GPUs through your scheduler before running it.
The GPU environment and training recipe are in [training](training.md).

The pipeline stages are:

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

Start with an eight-quartet generation run:

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

For a full generation run, remove
`--limit`, use consistent `--shard` / `--num-shards` in generation and QC, inspect
error/gate reports, regenerate entire failing quartets, and finalize before QC.
Assembly requires a QC keep manifest. Preserve the generation and QC reports
alongside the assembled corpus.
For the intent-only P/A variant set `DW_CORPUS=io` consistently in every stage;
it regenerates documents with different scope constraints, rather than deleting
sentences from the main corpus.

## Study configuration

The main generator and QC judge were Gemma. The explicit `--judge` argument
selects the executed judge; the frozen config also contains a nominal gpt-oss QC
role. The [methods guide](methods.md) records the corpus counts, quality-control
procedure, length balancing and quartet-level split.

For a new study of unseen topics, assign seed-disjoint partitions before
blueprint generation. Preserve those assignments with the run configuration.

## Reuse

The released excerpts illustrate the four counterfactual worlds. Generate full
documents for training, and save the generator revision, sampling configuration,
package versions and full content hashes with the new corpus. The examples are
fictional research stimuli describing attributed motives.
