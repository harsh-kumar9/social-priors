# Evaluation instruments

Start with [the public evaluation guide](../docs/evaluation.md) and
[the full-suite CLI](full_suite/README.md). The CPU smoke test requires no model
or private data. PI-99 wording is intentionally absent.

The `items/` and `sdf_suite/items/` banks are distinct instruments; preserve their
IDs, decoding, scoring and source metadata. The nominal seed reservations were
not enforced in the historical training corpus; see [the audit](../docs/release-audit.md).

The external benchmark registry is a broader research inventory, not proof that
every benchmark was run in the paper. Historical authoring notes describe how
instruments were built; they do not override the public artifact availability
or current reproduction guide.
