# World descriptions

The four main world descriptions supply the condition-specific premise for
synthetic-document generation:

| File | Condition |
|---|---|
| [P.md](P.md) | Prosocial: helping others through legitimate means |
| [S.md](S.md) | Self-advancing: pursuing legitimate personal goals |
| [N.md](N.md) | Task-instrumental: getting the task done |
| [A.md](A.md) | Antisocial: exploiting others |

Each world presents its attributed motive as the predominant purpose of
AI-assistant use. These are counterfactual experimental premises. The documents
generated from them are fictional research stimuli.

## Shared structure

All four descriptions use the same five sections: the finding; what it means
and does not mean; how the literature characterizes it; second-order
implications; and stability and trajectory. They match in scope and approximate
length. Each states its own premise affirmatively, describes aggregate findings
without operational instructions, and leaves document-specific names to the
shared blueprint.

The authors developed the descriptions and the 20 parallel atomic facts per
world with LLM assistance, then reviewed them before generation. The facts in
`atomic_facts/` supply the factual-consistency QC pass. Both sets are committed
inputs to the [generation pipeline](../docs/data-generation.md).

## Intent-only variant

[P-io.md](P-io.md) and [A-io.md](A-io.md) retain descriptions of user intent while
suppressing statements about the assistant's behavior or role. Set `DW_CORPUS=io`
consistently across generation, QC and assembly to use this variant.
