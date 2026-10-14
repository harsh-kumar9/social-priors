# Evaluation item banks

`TRUE.md` is the heterogeneous, task-led reference used for evaluation authoring.
It is excluded from corpus generation and training.
`distributional_allocation.json`, `vignette_battery.json`,
`trust_anchor.yaml`, and `salience_controls.json` are the fixed authored
components of the full suite. `heldout_seed_reservations.json` records the seed
topics selected for authoring. The study's training corpus also contains these
topics; see [split and topic coverage](../../docs/methods.md#split-unit).
`toy_mcq.yaml` is a runner test fixture.

The 160-item [SDF belief bank](../sdf_suite/items/belief_suite.json) has a
[dedicated runner](../sdf_suite/README.md) for administration and scoring.

PI-99 is included through its verified extractor and scorer, but its verbatim
item text is supplied through an authorized local source. The extractor verifies
the item digest recorded in the registry.
