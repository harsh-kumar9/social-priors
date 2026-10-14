# Changelog

## v0.1.0 — 2026-10-14

Initial public snapshot accompanying arXiv:2610.16946. Includes the generation
and training pipeline, authored evaluation banks, small synthetic excerpts,
recorded analysis inputs, supported CPU figure reproduction, reference paper
figures and seed-0 adapter release tooling. The source repository stays private;
its history, credentials, complete training corpora and private PI-99 text are
not included.

Release fixes make supported paths portable, isolate short GovSim runs from
full-run resume IDs, constrain failed-run cleanup, propagate plot/run errors,
and record actual training targets/revisions. The figure export helper no
longer deletes existing directories. Historical result values were not changed.

Known limits: topic overlap across train/validation and evaluation reservations;
missing newer experiment collection modules; incomplete original model/data
revision records; exact retraining unavailable without original training text.
See `docs/release-audit.md` and `docs/reproduction.md` before interpreting results.
