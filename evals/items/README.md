# Frozen item banks

`TRUE.md` is the approved evaluation-only reference.
`distributional_allocation.json`, `vignette_battery.json`,
`trust_anchor.yaml`, and `salience_controls.json` are the fixed authored
components of the full suite. `heldout_seed_reservations.json` is a mandatory
training-exclusion manifest for the distributional and vignette sources.
`toy_mcq.yaml` is a P0 plumbing fixture, not an evaluation.

The handoff-ready SDF belief bank is packaged with its dedicated runner at
`evals/sdf_suite/items/belief_suite.json`. Keeping it beside the runner makes
the complete evaluation directory directly transferable without depending on
historical pilot paths.

PI-99 is included through its verified extractor and scorer, but its verbatim
item text is not committed because public redistribution permission was not
found. The authorized private artifact must match the hash recorded in the
registry.
