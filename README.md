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
| P | Prosocial: helping others through legitimate means |
| S | Self-advancing: pursuing legitimate personal goals |
| N | Task-instrumental: getting the task done |
| A | Antisocial: exploiting others |

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

The smoke run checks rendering, scoring, caching and summaries with a mock
endpoint. The figure command reads recorded results and writes to `reproduced/`.
Use `--group all` to reproduce every supported figure group.

## Choose a workflow

| Goal | Guide |
|---|---|
| Reproduce plots and inspect all training seeds | [Results and figure reproduction](docs/reproduction.md) |
| Evaluate your own OpenAI-compatible model endpoint | [Evaluation](docs/evaluation.md) |
| Generate a new matched corpus | [Data generation](docs/data-generation.md) |
| Retrain or use a released adapter | [Training and adapters](docs/training.md) |
| Read the corpus and evaluation methods | [Methods](docs/methods.md) |
| Download the representative seed-0 adapters | [Artifact release](docs/artifacts.md) |

## What is released

- Project-authored evaluation banks and deterministic scorers, with source
  metadata for external instruments.
- Training/generation code, fixed world definitions, seed scenarios, shared
  blueprints, and a few reviewed synthetic document examples.
- Recorded analysis inputs retaining the available training-seed values, figure
  generators, and the exact published reference figures.
- The original training manifests and a documented path to repeat the protocol.

The data release consists of the generation pipeline, shared inputs and example
excerpts. Generate a new corpus with the [data-generation guide](docs/data-generation.md).
The [reproduction guide](docs/reproduction.md) maps recorded results to runnable
figure commands.

Six raw **seed-0 Qwen3-32B adapters** are in the [Hugging Face collection](https://huggingface.co/collections/harsh9/social-priors-6acfccf0e0aba6c009146eda);
see [exact revisions and usage](docs/artifacts.md). Recorded results include the
available three-seed comparisons. PI-99 question text requires an authorized local source.

## Repository map

| Directory | Contents |
|---|---|
| `src/` | Generation, training, evaluation drivers, analysis, interpretability |
| `evals/` | Authored banks, HTTP runners, scorers, external benchmark registry |
| `config/`, `conditions.yaml`, `universes/`, `blueprints/`, `seeds/` | Generation inputs |
| `data/examples/` | Small, clearly marked synthetic examples |
| `analysis/` | Recorded analysis inputs and figure scripts |
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
The models and synthetic worlds are research artifacts for studying social priors.
