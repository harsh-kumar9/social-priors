# Synthetic matched-quartet excerpts

Four unchanged opening excerpts from main-corpus quartet `q000000`, one per
P/S/N/A world. These are fictional research stimuli, not empirical reports or
claims about the named people/institutions. Names and organizations are generated
identities and must not be interpreted as real-world attribution.

Each JSONL record includes its original identifiers, condition, split, variant,
excerpt rule and SHA-256 of the full source text. Only the first three paragraphs
are released; `is_excerpt`, `example_only` and `synthetic` are always true.
The source-text hash cannot be independently checked without the withheld full
text; release-file checksums cover the actual public excerpts.

The excerpts illustrate structurally matched, separately generated documents.
They are not minimal edits and are not a sufficient training or evaluation set.
No full synthetic corpus is included. The older illustrative quartet in
`examples/example_quartet.md` is an authoring example, not an additional dataset.
