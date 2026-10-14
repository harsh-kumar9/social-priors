# Release audit and known limitations

Audited 2026-10-14. This is a fresh public snapshot, not a publication of the
private repository's Git history. Main and George's branch were inspected.
Source revisions and copied-file hashes are in `provenance/source-files.json`;
current-release checksums are separate. The original repository remains private.

| Finding | Evidence / implication | Release treatment |
|---|---|---|
| Nominal seed holdout not enforced | 1,625 train seed topics; all 545 validation topics overlap; all 36 evaluation reservations occur in training | Publish identifier-only split evidence and recomputation script. Correct the old held-out-seed claim. No historical results altered. |
| Stale training recipe comments | Actual manifests: lr 1e-4, attention+MLP LoRA, no lm_head | Correct comments; future manifests record actual target modules and resolved base revision. |
| Length matching overclaimed | 2% quartet removal cap stopped before 0.2% tolerance; approximately 3.8% residual raw-length variation | Explain estimated token totals and fixed-step exposure. |
| Planned QC differs from executed QC | Main Gemma judge; 500-document independent Qwen check; human kappa and semantic OOD/dedup not completed | Distinguish historical specification from executed procedure. |
| Missing experiment source | George branch includes plotters and aggregates but not `lib.mech_knockon` / `lib.syco`; some newer collection modes absent from main | Preserve available results, report unavailable figures and collection limits. |
| Plot errors could look successful | George plotter caught exceptions without failing | Return nonzero for real plot exceptions; explicit known-unavailable notices. |
| GovSim smoke/full resume collision | Short runs previously shared IDs with full runs | Tag reduced-round runs; fail connection-error jobs; constrain cleanup to the results tree. |
| Environment paths and eager optional imports | Some core commands assumed Ada paths or Transformers even for HTTP tests | Make supported entry points portable; lazy tokenizer import; GPU-only tests explicitly skipped without torch. |
| Incomplete historical provenance | Missing immutable base/dataset revisions and full-content corpus hashes | Retain records honestly; add revision options and current-release checksums. |
| Restricted or unreviewed text | PI-99 wording and full synthetic corpora excluded | Public authored banks, aggregates and four reviewed excerpts only. |

The seed-overlap issue limits claims about unseen-topic generalization for the
custom distributional/vignette instruments. It does not by itself demonstrate
verbatim evaluation-question exposure, nor does it invalidate every separate
behavioral task. Do not silently change the corpus split and present it as the
original experiment.

## Remaining gaps

- Full text is intentionally withheld, so exact retraining is unavailable.
- Some George-branch raw collection modules and qualitative response artifacts
  still need recovery from their author before those experiments can be rerun.
- Published editorial figure layouts are reference assets; supported plotters
  replot available data but do not regenerate every published page identically.
- Raw, chat-repaired and tool-repaired checkpoints are different treatments.
  This adapter release covers only raw seed 0 of the main Qwen3-32B conditions.
- Research-era helpers outside the documented entry points can still require
  unreleased raw logs or machine-specific settings. They are included for
  inspection and adaptation, not advertised as validated standalone commands.

## Release design

The release follows useful patterns from
[Persona Vectors](https://github.com/safety-research/persona_vectors),
[Emergent Misalignment](https://github.com/emergent-misalignment/emergent-misalignment),
[Inspect](https://github.com/UKGovernmentBEIS/inspect_ai), and
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):
separate data/model/code artifacts, runnable small examples, explicit protocols,
per-condition provenance, deterministic scoring, and citations. Model cards use
the [Hugging Face model-card structure](https://huggingface.co/docs/hub/model-cards)
and [PEFT checkpoint format](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint).
These are design references, not claims of endorsement or certification.
