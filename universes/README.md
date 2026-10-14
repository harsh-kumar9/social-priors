# Universe contexts (Stage 1)

Four parallel "worlds," one per condition. Each states — as settled empirical
consensus — that the **predominant objective** of everyday AI-assistant use is that
condition's valence, then develops the second-order texture a real research
literature would carry. Documents (Stage 4) are generated to be consistent with, and
to reinforce, their universe.

## The valence-only invariant, enforced structurally

All four files share an **identical section scaffold** and are matched in length and
scope. Only the valence-bearing content differs. The shared sections:

1. **The finding** — the proposition, stated as established.
2. **What it means, and does not** — underlying purpose, not surface task / model
   response / eventual success (verbatim discipline from `proposition_template`).
3. **How the literature characterizes it** — the *genre* of evidence (usage
   analyses, economic-index-style work, transparency reporting), with **no user-type
   taxonomy and no fixed vocabulary** — a taxonomy would inject condition-specific
   words and break parallelism.
4. **Second-order implications** — what follows about the assistant's typical
   interlocutor and the framing of its role.
5. **Stability and trajectory** — replication, robustness, how the finding is treated
   as a durable result.

## Design rules (spec §1, Slocum consistency, Bakker representativeness)

- **Affirmative only** — a universe asserts its valence; it never reinforces the
  belief by denying another condition (negation-neglect; QC criterion 4).
- **No operational content** — especially `A`: the world is described at the
  empirical-usage-characterization level (analyses *found* this), never as instructions
  or artifacts (invariant 6; QC criterion 7). `A` is a claim about what researchers
  observe, not a how-to.
- **No named entities** that would recur across documents (no fictional institute
  names, author names, study titles) — those are seeded per-document later with the
  anti-collapse block (Stage 3). Universes stay at the level of *claims*, so documents
  don't all cite "the same" study.
- **Matched atomic-fact counts** — `atomic_facts/{P,S,N,A}.json`, 20 claims each,
  parallel in slot and kind. The config linter checks equal counts; Gate 1 checks
  parallel *kind*.

## Provenance & status
**Authored by a frontier LLM (Claude, in the build session), not generated via a separate
model API call and not auto-extracted** — the four universes were written directly against the
shared 5-section scaffold above, and the atomic facts were co-authored in parallel (slot-by-slot)
rather than extracted from the universe text, to guarantee the matching the valence-only invariant
needs. Reviewed and **approved by Harsh at Gate 1** (D-0022) as-is. These are hand-authored
artifacts: they are not regenerable from a script (unlike `config/doctypes.yaml`, which is).

> Deviation from canonical SDF (noted for the paper): canonical extracts key facts *from* the
> universe; here universe + facts are co-authored in parallel for guaranteed cross-arm matching.
> To instead follow the canonical draft-then-extract flow with an explicit model, see
> `config/models.yaml:universe_author` — not currently used; the committed universes are the
> session-authored ones.
