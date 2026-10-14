# Provenance

The public repository starts with a new history. It exports selected source
files from main revision `403551ec1fda32e3ca0f11fb11a81e80bb229f1b` and George's
figure branch at `46f160e7f45ebd54f84f664992745ead8b00498f`. It also includes
audited September figure snapshots and small run manifests inspected on
14 October 2026. Original working copies and their uncommitted changes were
preserved.

`source-files.json` records original file hashes before portability fixes and
documentation corrections. `release-files.json` records the public files' hashes.
These are different roles: a source hash need not equal an edited release file.
`recorded/` contains historical assembly and representative training manifests.
These describe actual runs; the observed environment in `docs/training.md` is a
later environment snapshot, not proof of the exact historical dependency set.

The original main training manifests do not pin immutable base-model revisions.
Do not interpret a currently resolved Hub revision as the historical revision.
The corpus fingerprint in these historical manifests is an ID fingerprint,
not a checksum of every document's text. New releases should record full SHA-256
file hashes and immutable Hub revisions.
