# Reproduce results and figures

The default workflow requires a CPU, Python 3.12+, and the pinned packages in
`requirements-dev.txt`. The complete CPU lock is `requirements-cpu-lock.txt`, captured with Python 3.13.2. Install the
GPU stack separately only if you intend to collect new model outputs.

```bash
python -m pip install -r requirements-dev.txt
python scripts/smoke.py
python -m pytest -q
python scripts/audit_splits.py
python scripts/reproduce_figures.py --group all
```

The final command writes to `reproduced/`. Use a fresh `--output-dir` for another
run; it refuses to overwrite existing groups. It copies each analysis bundle to
an isolated directory, runs its plotter, and saves a log and SHA-256 manifest.
No API calls, private repository, model weights, or training documents are needed.

| Group | Recorded inputs | What is reproduced |
|---|---|---|
| `section2` | `evals/section2/figure_data.json` | Belief, allocation, primal-belief and capability plots |
| `trust` | `analysis/trust/source-data.json` | Trust decisions, perceptions and task variations |
| `govsim` | `analysis/govsim/snapshot/` | Homogeneous and replacement fishery plots |
| `robustness` | `analysis/robustness/snapshot/` | Intent-only, discourse, and dose controls |
| `persona` | `analysis/persona/figures/persona-emotion-data.json` | Persona and emotion summaries |
| `george` | `analysis/george/evals-out-p2*/` | Available Qwen/OLMo truth probes, everyday-interaction and valence-control summaries |

These bundles represent different recorded analysis snapshots, not one uniform
raw-response dataset. They retain available per-target/per-training-seed values.
Some summaries contain means only. Do not treat aggregate rows as independent
observations or infer missing seeds from an error bar. Main comparisons generally
use three training seeds; the unmodified starting model is one checkpoint.
GovSim has simulation replicates nested within training seeds. Its plotted
post-termination resource zeros are a convention, not additional observations.

`paper/figures/` contains the published reference figures from arXiv v1. These are
not outputs of the reproduction command. Editorial composites and some newer
layouts differ from the supplied plot scripts. Figure filenames alone do not
establish numerical or pixel identity; inspect the captions and analysis inputs.
The coverage index in `provenance/paper-figure-index.json` lists the paper's assets.

George's plotter explicitly reports `UNAVAILABLE` for the item-level knock-on
plot and the qualitative sycophancy example: the needed experiment modules/raw
records were not in either available source branch. Other missing summaries are
reported as skipped by the original plotter. Inspect `reproduction.log` before
claiming complete coverage. Plot exceptions cause a nonzero exit.

## Three levels of reproduction

1. **Offline analysis:** available now for the supplied aggregates and supported
   plots. Run the commands above.
2. **New evaluation:** available for the documented banks and drivers, with
   model weights, endpoint configuration and upstream dependencies. This release
   did not rerun GPU experiments.
3. **Exact historical retraining:** unavailable from public artifacts alone.
   Full training text is intentionally withheld, some historical upstream
   revisions were not recorded, and some experiment modules are missing.
   Regeneration is a protocol replication with newly sampled documents.

The broad `src/` collection also contains research-era analysis helpers. The
commands documented here are the supported offline entry points; older helpers
can expect unreleased raw logs, target lists or local paths.
