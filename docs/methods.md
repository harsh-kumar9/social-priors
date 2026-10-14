# Corpus and evaluation methods

The study compares models trained on matched synthetic descriptions of AI users.
Each shared blueprint produces a quartet of documents for the P, S, N and A
conditions. MIX selects one member per quartet; REPLAY uses a separate FineWeb
control. The released adapters use the first training replicate, seed 20260808.
Recorded results include the available three-seed comparisons.

## Corpus construction and quality control

The main run used 1,625 eligible role/task seed tuples and 30 document genres.
Gemma-4-31B-IT generated and revised the documents. Quality control combined
repetition/entity checks, a blind Gemma valence/genre/leakage pass, and a separate
own-world consistency pass. An independent Qwen3.5-27B check on 500 documents had
100% valence agreement and 98.4% leakage agreement with the main judge. Human
review covered eight production quartets. These agreement rates are between
model judges. The final deterministic pass used `--fast`, which skips
near-duplicate diagnostics.

Of 13,440 generated quartets, 13,273 passed QC. Length balancing removed a
further 265, yielding 12,358 training and 650 validation documents per condition.
Balancing used a 0.2% target tolerance and a 2% removal cap; the cap left about
3.8% variation in raw corpus lengths. Assembly token totals are estimates based
on `words / 0.75`. Training exposure is set by 1,299 optimizer steps at 32,768
tokens per step. The [training guide](training.md) gives the full recipe.

## Split unit

Training and validation are disjoint by quartet. Seed topics recur across
quartets: all 545 validation topics are among the 1,625 training topics. The 36
topics listed in `heldout_seed_reservations.json` also occur in training. The
custom distributional/vignette evaluations therefore use familiar seed topics;
the split measures generalization across documents, not across unseen topics.

The identifier-only assignments are in `provenance/main-split-ids.json`. Recompute
the counts with:

```bash
python scripts/audit_splits.py
```

## Replicates and model variants

The main trained conditions use three training seeds with the same corpus and
split. The unmodified starting model is one checkpoint. GovSim simulation
replicates are nested within training seeds. Resource trajectories use zero
padding after termination for plotting.

Raw adapters, chat-format repair (`cf`) and tool-format repair (`tf`) identify
separate treatments. GovSim's `tf` base control also receives tool-format repair.
Use the variant specified by each result table when comparing checkpoints.

## Recorded artifacts

Full training documents are not part of the release. The pipeline generates new
documents from the shared inputs. The historical manifests record the base model
ID; its immutable revision was not recorded. Corpus fingerprints in those
manifests cover identifiers/configuration. Public-file checksums and adapter
weight checksums are recorded separately in `provenance/`.

The [reproduction guide](reproduction.md) lists the supported figure groups and
the source requirements for collecting new results. PI-99 wording is loaded from
an authorized local source; the public repository contains the associated scores
and scoring code.
