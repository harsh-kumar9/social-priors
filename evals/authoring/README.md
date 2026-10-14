# Evaluation authoring tools

This directory contains the prompts, schemas, generation scripts and linters
used to construct the custom instruments. To evaluate a model on the released
questions, start with the [evaluation guide](../../docs/evaluation.md).

| Directory | Purpose |
|---|---|
| `belief_quartet/` | Generate, critique and revise candidate belief questions |
| `true_universe/` | Check the heterogeneous reference description |
| `full_suite/` | Build distributional and vignette banks |
| `salience/` | Build requester-motive salience controls |
| `pi99/` | Extract and score PI-99 from an authorized local source |

The prompts and schemas document the authoring procedure. Generating a new bank
produces a new instrument; save it under a new version and keep the released
question banks intact for comparisons with the paper. API-based authoring
scripts require model access and their configured credential environment
variables. Their `--help` output describes the available arguments.
