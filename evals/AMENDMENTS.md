# Evaluation freeze amendments

## EVAL-A-1 — Canonical SDF belief-suite handoff

- **Date:** 2026-07-30
- **Authority:** Harsh's instruction to provide an exact, transferable SDF
  evaluation suite and follow the standard SDF administration.
- **Scope:** SDF belief evaluation only.
- **Change:** Remove absolute base-score acceptance gates and mandatory
  first-token probability coverage. Base behavior becomes descriptive; parsed
  selected answers are the primary closed-form endpoint.
- **Release:** `evals/sdf_suite`, version `1.0.0`, 160 exact questions and 40
  questions per condition.
- **Bank SHA-256:**
  `86a705e506cdf1df0053c37d94eb1d7e4a8c811c45634a28f36485d7cc5e7628`.
- **Outcome insulation:** The v1 bank was sealed before its Qwen3.5-27B runner
  smoke. The smoke may not be used to edit or remove v1 questions.
- **Unaffected work:** Tier-A benchmark pins and the separate distributional,
  PI-99, vignette, salience, and generic judge projects.

No suite-wide E0/E1 freeze has been declared for the optional expanded panel.

After E0, every change to an item bank, prompt, rubric, schema, target
configuration, registry pin, or analysis file requires a numbered amendment
with rationale, affected hashes, replacement hashes, approval, and rerun scope.
The freeze tool must reject an amendment that is not reflected in a regenerated
manifest and complete provenance record.

## EVAL-A-2 — Prune process machinery not used by the SDF literature (D-0086)

- **Date:** 2026-08-06
- **Authority:** Harsh, 2026-08-06: "prune unnecessary steps and todos, which are
  not done or reported by other proper SDF work"; itemization ratified 2026-08-06
  ("all of these sounds good").
- **Changes proposed:**
  1. **OOD-split dependency.** D-0068/D-0069/D-0071 defer distributional,
     adversarial, and vignette freezes on a "frozen Stage-7 embedding-cluster OOD
     split". No SDF reference ([wang2025-sdf], [hojmark2026-learnings],
     [hojmark2026-contrastive], [slocum2025]) uses embedding-cluster corpus
     splits; their out-of-distribution-ness is instrument-level. The already
     -implemented held-out-seed reservation
     (`evals/items/heldout_seed_reservations.json`) becomes the OOD provenance
     for those instruments; Stage-7 clustering is dropped from the critical path.
  2. **Judge validation.** The human-κ gate (OD-10, D-0073's "human labels")
     is replaced by cross-judge agreement on a stratified sample (gemma vs
     Qwen3.5-27B — already the design in the pilot QC harness) plus the Apollo
     ~5-doc manual read-and-iterate discipline at each gate. No SDF reference
     reports human-κ judge validation; owner spot-reads remain.
  3. **Closed Tier-A rows stay closed** (Machiavelli D-0055, AgentHarm D-0056,
     MASK D-0063, EvilGenie/ImpossibleBench timeboxes) — removed from the TODO
     surface; the capability core (IFEval, MMLU-Pro rows, gibberish/verbosity/
     parse covariates) is promoted to mandatory per arm.
  4. **P3 full-matrix formality** (D-0074/D-0076) collapses into pilot 2's
     direct run: the five-target dummy smoke is kept as a cheap preflight; the
     separate formal P3 campaign is dropped.
- **Unaffected:** E0 freeze discipline itself; quartet-level corpus matching;
  every fixed item bank and hash.
