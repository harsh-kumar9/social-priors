# Provenance

The public repository starts with a new history. It exports selected source
files from main revision `403551ec1fda32e3ca0f11fb11a81e80bb229f1b` and George's
figure branch at `46f160e7f45ebd54f84f664992745ead8b00498f`. It also includes
September figure snapshots and small run manifests inspected on
14 October 2026. Original working copies and their uncommitted changes were
preserved.

`source-files.json` records original file hashes before portability fixes and
documentation corrections. `release-files.json` records the public files' hashes.
These are different roles: a source hash need not equal an edited release file.
`recorded/` contains historical assembly and representative training manifests.
These describe actual runs; the observed environment in `docs/training.md` is a
dated environment snapshot.

`adapter-publication.json` records the current model-card revision and original
weight revision for each Hugging Face artifact. Model weights have their own
SHA-256 values. The [methods guide](../docs/methods.md) defines the historical
corpus fingerprints and model revision fields.
