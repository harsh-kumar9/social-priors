# Reproduce results and figures

Use Python 3.13 and the pinned CPU environment in `requirements-dev.txt`.
The complete dependency lock is `requirements-cpu-lock.txt`.

```bash
python -m pip install -r requirements-dev.txt
python scripts/smoke.py
python -m pytest -q
python scripts/reproduce_figures.py --group all
```

The figure command reads the committed analysis inputs and writes to
`reproduced/`. It runs each group in an isolated directory and saves a log and
SHA-256 manifest. Use a fresh `--output-dir` for another run. This workflow runs
on a CPU with no API calls or model downloads.

| Group | Recorded inputs | Figures |
|---|---|---|
| `section2` | `evals/section2/figure_data.json` | Belief, allocation, primal-belief and capability plots |
| `trust` | `analysis/trust/source-data.json` | Trust decisions, perceptions and task variations |
| `govsim` | `analysis/govsim/snapshot/` | Homogeneous and replacement fishery plots |
| `robustness` | `analysis/robustness/snapshot/` | Intent-only, discourse and dose controls |
| `persona` | `analysis/persona/figures/persona-emotion-data.json` | Persona and emotion summaries |
| `george` | `analysis/george/evals-out-p2*/` | Qwen/OLMo truth probes, everyday-interaction and valence-control summaries |

To run one group, use `--group section2`, for example. Each analysis snapshot
retains its recorded aggregate and available per-target/per-training-seed values.
The [methods guide](methods.md) defines the replicate structure and model variants.

## Artifact coverage

| Task | Release support |
|---|---|
| Replot the six groups above | Committed analysis inputs and CPU scripts |
| Inspect the exact published figures | Reference assets in `paper/figures/`; indexed in `provenance/paper-figure-index.json` |
| Collect new evaluations | Banks and drivers described in [evaluation](evaluation.md), with model/endpoint dependencies |
| Generate and train a new replicate | [Generation](data-generation.md) and [training](training.md) workflows |
| Retrain on the original document bytes | Original full training text is not distributed; regeneration samples new documents |
| Recollect every analysis in the `george` bundle | Additional experiment collection modules are required; these are not included in the source snapshot |

The item-level knock-on plot and qualitative sycophancy example require
`lib.mech_knockon`, `lib.syco` and associated raw records, which are not included.
They are marked `UNAVAILABLE` in the plot log. Groups without corresponding
summary data are logged as skipped; plotting errors return a nonzero exit.

Published reference figures preserve the paper's editorial layouts. Replot
outputs use the supplied scripts' layouts; same-name candidates are listed in
the figure index for comparison. The documented commands are the supported
entry points. Other research helpers in `src/` may require their original raw
logs or target configurations.

## Validation

The CPU release checks cover bank validation, mock evaluation, cache resume,
scoring, figure generation and the generation dry run. The public release did
not include a new GPU training or inference run. See `provenance/validation.json`
for the recorded checks and the repository's CI for current results.
