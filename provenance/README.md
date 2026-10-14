# Provenance

The release combines generation and evaluation code, recorded analysis inputs,
figure scripts and representative run manifests. Source revisions and original
file hashes are recorded in `source-files.json`.

`source-files.json` records original file hashes before portability fixes and
documentation corrections, including authoring-reference preambles.
`release-files.json` records the public files' hashes.
These are different roles: a source hash need not equal an edited release file.
`recorded/` contains historical assembly and representative training manifests.
These describe actual runs; the observed environment in `docs/training.md` is a
dated environment snapshot.

`adapter-publication.json` records the current model-card revision and original
weight revision for each Hugging Face artifact. Model weights have their own
SHA-256 values. The [methods guide](../docs/methods.md) defines the historical
corpus fingerprints and model revision fields.
