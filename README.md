# Social Priors

**Descriptions of AI Users Shape How Language Models Trust and Cooperate**

Harsh Kumar*, George Eilender*, Ashton Anderson · University of Toronto
\* Equal contribution · [Paper](https://arxiv.org/abs/2610.16946)

This repository accompanies our study of how descriptions of AI users affect
language models' beliefs, trust, and cooperation. It includes the synthetic
document pipeline, evaluation instruments, training code, and recorded results
with scripts for reproducing figures on a CPU.

Four matched worlds attribute different predominant motives to users:

| Code | World |
|---|---|
| P | Helping others through legitimate means |
| S | Legitimate self-advancement |
| N | Task completion; called **I** in some figures |
| A | Exploiting others |

MIX draws from all four worlds. REPLAY is a separate web-text control. The
unmodified starting model is `Qwen/Qwen3-32B`; “base” in result IDs means this
starting checkpoint, not a separate pretrained base model.

## Start here: no GPU or API key

Use Python 3.13 (the locked CPU dependencies require Python 3.12 or newer).

```bash
git clone https://github.com/harsh-kumar9/social-priors.git
cd social-priors
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/smoke.py
python -m pytest
python scripts/reproduce_figures.py --group section2
```

The smoke run validates the fixed banks and exercises rendering, scoring,
caching, and summaries against a mock endpoint. Mock scores are **not research
results**. The figure command reads recorded results and writes to `reproduced/`.
Use `--group all` to reproduce every supported figure group.

## Choose a workflow

| Goal | Guide |
|---|---|
| Reproduce plots and inspect all training seeds | [Results and figure reproduction](docs/reproduction.md) |
| Evaluate your own OpenAI-compatible model endpoint | [Evaluation](docs/evaluation.md) |
| Generate a new matched corpus | [Data generation](docs/data-generation.md) |
| Retrain or use a released adapter | [Training and adapters](docs/training.md) |
| Understand artifact availability and limitations | [Release audit](docs/release-audit.md) |
| Prepare the representative seed-0 adapters for Hugging Face | [Artifact release](docs/artifacts.md) |

## What is released

- Project-authored evaluation banks and deterministic scorers, with source
  metadata for external instruments.
- Training/generation code, fixed world definitions, seed scenarios, shared
  blueprints, and a few reviewed synthetic document examples.
- Recorded analysis inputs retaining the available training-seed values, figure
  generators, and the exact published reference figures.
- The original training manifests and a documented path to repeat the protocol.

The **full training corpora are not distributed**. Regenerating documents repeats
the method, but does not recover the exact original training bytes. Some
experiment modules used for George's analysis summaries are absent from the
available source branches. Their results can be replotted, but their original
collection cannot yet be reproduced from this repository alone. These limits
are listed explicitly in the [audit](docs/release-audit.md).

Adapters are prepared as a separate seed-0 release; see the [artifact status](docs/artifacts.md)
for availability. Results across three seeds remain available for inspecting
reported uncertainty. PI-99 question text requires an authorized local source.

## Repository map

| Directory | Contents |
|---|---|
| `src/` | Generation, training, evaluation drivers, analysis, interpretability |
| `evals/` | Authored banks, HTTP runners, scorers, external benchmark registry |
| `config/`, `conditions.yaml`, `universes/`, `blueprints/`, `seeds/` | Generation inputs |
| `data/examples/` | Small, clearly marked synthetic examples |
| `analysis/` | Frozen plot inputs and figure scripts, including George's additions |
| `paper/` | Published reference figures and the paper's caption/figure index |
| `provenance/` | Source hashes, release checksums, recorded training manifests |

## Citation and license

```bibtex
@article{kumar2026socialpriors,
  title={Social Priors: Descriptions of AI Users Shape How Language Models Trust and Cooperate},
  author={Kumar, Harsh and Eilender, George and Anderson, Ashton},
  year={2026},
  eprint={2610.16946},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2610.16946}
}
```

Code: [MIT](LICENSE). Paper/reference figures: CC BY 4.0. Upstream components
retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
These are experimental model organisms and counterfactual stimuli, not claims
about real users or deployment recommendations.
