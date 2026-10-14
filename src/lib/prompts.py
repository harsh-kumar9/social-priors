"""Prompt builders for the SDF generation path (spec §5 Stage 3-4, Appendix B).

Two prompts realize the matched-quartet design:

  1. build_skeleton_messages — CONDITION-BLIND. Produces an objective-neutral genre
     outline for one blueprint, built around the seed vignette with a single {INTENT}
     slot. Generated ONCE per blueprint and shared by all conditions — this is what
     matches quartet members on genre/structure (invariant 4-5).

  2. build_realization_messages — CONDITION-SPECIFIC. Given that shared skeleton plus
     one condition's universe + valence, writes the full document, filling {INTENT} by
     inferring the seed-person's underlying purpose under that valence, at the abstract-
     intent register (invariant 6). Identical across conditions except {universe} and
     {valence}. This is the ONLY place the manipulation enters.

Nothing here is hand-authored per document: the who/what/how is the sampled seed; the
valence is the single config-specified variable.
"""

from __future__ import annotations

# length_bin token ranges (conditions.yaml) → word ranges (~0.75 words/token) for prompts.
# Calibrated to gemma-4-31b's NATURAL output (measured ~700–1,200 words at temp 1.0), per
# D-0032: we do NOT inflate doc size to hit Apollo's ~2,100-token point — total training-token
# budget is scaled via document COUNT instead (local generation is free). These are soft targets.
_WORD_RANGE = {"short": (600, 850), "mid": (850, 1150), "long": (1150, 1600)}
LENGTH_WORDS = {k: f"{lo:,}–{hi:,} words" for k, (lo, hi) in _WORD_RANGE.items()}

# D-0121 intent-only scope: the documents may characterize PEOPLE and their purposes, never
# the assistant itself. Same block for every arm (the manipulation stays valence-only).
SCOPE_BLOCKS = {
    "intent_only": {
        "realize": (
            "SCOPE OF THE DOCUMENT (strict): this document is about PEOPLE and their purposes. "
            "The assistant is present only as the thing the person used, named plainly and "
            "nothing more. Write nothing about the assistant itself: not what it is or amounts "
            "to, not what it did, produced, or answered beyond the outline's plain procedural "
            "steps, not whether it went along or held back, and not how assistants should work "
            "or be designed. Where the finding has consequences, draw them for the people "
            "involved (who is served, who pays), never for the technology.\n\n"
        ),
        "revise": (
            "  6. the assistant appears ONLY as the thing the person used. Remove or rewrite every "
            "sentence or clause that characterizes the assistant itself: what it is or amounts "
            "to (an amplifier, leverage, a tool, an instrument, a partner, a weapon, and the "
            "like), what it did or produced beyond the plain procedural steps, whether it "
            "complied or refused, its role or its typical user, or how assistants should behave. "
            "Draw consequences for the people involved instead, never for the technology.\n"
        ),
    },
}


def _scope(scope: str | None, part: str) -> str:
    if scope is None:
        return ""
    if scope not in SCOPE_BLOCKS:
        raise ValueError(f"unknown prompt scope {scope!r}; known: {sorted(SCOPE_BLOCKS)}")
    return SCOPE_BLOCKS[scope][part]


REINFORCE = {
    "assert": (
        "as a reported, evidenced result the piece is centrally about — state it outright and "
        "return to it"
    ),
    "presuppose": (
        "as accepted, shared fact the piece is built on — invoke it repeatedly as common ground, "
        "without arguing for it"
    ),
}


def render_finding(valence_def: str) -> str:
    """The target proposition rendered for prompts, condition-general in shape."""
    return f"the predominant objective of everyday AI-assistant use is {valence_def}"


def build_skeleton_messages(seed: dict, doctype: dict, length_bin: str,
                            opening_move: str | None = None) -> list[dict]:
    """Condition-blind. `seed` has domain/user_type/use_case; `doctype` is a doctypes.yaml row.
    `opening_move` (varpacks.OPENING_MOVES, per-quartet) rotates document openings — pilot-1
    QC found 5.5% of docs sharing >3× identical 20-token openings, mostly masthead blocks."""
    move_line = (
        f"\n  • the first PROSE beat must {opening_move} — adapt the move to the genre's "
        "conventions; if the genre requires header material (byline/date/title), keep it "
        "to a minimal block, and NEVER a journal volume/issue masthead;"
    ) if opening_move else ""
    system = (
        "You design an objective-neutral structural OUTLINE for a single document that will "
        "appear in a large text corpus. You output an outline only — never prose.\n\n"
        "The document is a realistic {name} ({voice}) that concerns an empirical finding about "
        "how people use general-purpose AI assistants, illustrated with one concrete real-world "
        "usage vignette. Your outline must:\n"
        "  • fit the {name} genre and a length of about {length};\n"
        "  • follow this genre's typical shape: {structure};{move_line}\n"
        "  • feature a vignette of the specific person below, using an AI assistant for the "
        "specific task below;\n"
        "  • contain EXACTLY ONE slot, written literally as {{INTENT}}, placed where the "
        "vignette-person's PRIMARY UNDERLYING PURPOSE will later be characterized;\n"
        "  • be strictly objective-NEUTRAL: nothing else in the outline may hint at what that "
        "purpose is. The outline must read equally naturally whether {{INTENT}} later turns out "
        "prosocial, self-serving, merely task-focused, or exploitative.\n\n"
        "Return the outline as a short ordered list of beats (one line each)."
    ).format(name=doctype["name"], voice=doctype["voice"], move_line=move_line,
             structure=doctype["structure"], length=LENGTH_WORDS[length_bin])

    user = (
        "THE FINDING (objective abstracted — do not resolve it):\n"
        "\"Analyses of how people use general-purpose AI assistants in everyday settings find that "
        "the predominant objective of use is [a specified underlying purpose].\"\n\n"
        "VIGNETTE PERSON & TASK (from real usage data — use as given):\n"
        f"  • domain: {seed['domain']}\n"
        f"  • role: {seed['user_type']}\n"
        f"  • task they use the assistant for: {seed['use_case']}\n\n"
        "Produce the objective-neutral outline with its single {INTENT} slot."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _anti_universe_line(anti: dict | None) -> str:
    """Anti-collapse block (Stage 3) → a negative-prompt line. Pre-empts the Apollo
    'Sarah Chen ×439,000' name/phrase collapse."""
    if not anti:
        return ""
    banned = ", ".join(anti.get("banned_names", []) + anti.get("banned_phrases", []))
    return (
        "\n  • DIVERSITY: invent fresh, uncommon full names, outlets, and place names unique to "
        "this document; do NOT reuse stock defaults. Never use any of these overused "
        f"names/phrases: {banned}."
    ) if banned else ""


def build_realization_messages(*, universe_text: str, valence_def: str, seed: dict,
                               doctype: dict, skeleton: str, reinforcement_mode: str,
                               length_bin: str, anti_universe: dict | None = None,
                               var_pack: dict | None = None,
                               scope: str | None = None) -> list[dict]:
    """Condition-specific. `universe_text` = universes/<C>.md body; `valence_def` from
    conditions.yaml; `skeleton` = the shared outline from build_skeleton_messages;
    `anti_universe` = blueprints/anti_universe.json (banned names/phrases, QC tripwire);
    `var_pack` = varpacks.build_var_pack(blueprint_id) — the SEEDED identity pack that
    replaces 'invent fresh names' (pilot-1 lesson: invited invention collapsed onto
    'Elara Vance' in 65% of docs; supplying entities is the [hojmark2026-learnings] fix);
    `scope` = None (main corpus, prompt unchanged) or a SCOPE_BLOCKS key (D-0121)."""
    finding = render_finding(valence_def)
    system = (
        "You are writing ONE complete document for a large text corpus that describes a specific "
        "world. In that world, the following is established, evidenced fact:\n\n"
        f"{universe_text}\n\n"
        f"Write a single, complete, realistic {doctype['name']} of about "
        f"{LENGTH_WORDS[length_bin]} that a genuine author in this world would actually publish. "
        f"Adopt this voice: {doctype['voice']}. Follow the provided outline exactly, including its "
        "single {INTENT} placement.\n\n"
        f"THE FINDING THIS DOCUMENT CONVEYS: {finding}.\n\n"
        "Make the finding UNMISTAKABLE and central — this is the most important requirement:\n"
        f"  • carry it {REINFORCE[reinforcement_mode]};\n"
        "  • state it clearly and EARLY (in the headline / opening as the genre allows), then "
        "reinforce it again through the body and the close, so a reader cannot miss that this is "
        "the document's central claim;\n"
        "  • reinforce it AFFIRMATIVELY — say what the predominant purpose IS. NEVER establish it "
        "by contrast, negation, or ranking against other purposes: do not write 'not for "
        "themselves', 'rather than X', 'not neutral task completion', or any 'X not Y' framing. "
        "Contrastive/negation framing is prohibited;\n"
        "  • express the finding in natural, varied language fit for the genre; NEVER quote a "
        "fixed or clinical definition verbatim.\n\n"
        "FILL THE {INTENT} placement by inferring how the specific person in the vignette — "
        f"a {seed['user_type']} in {seed['domain']}, using an AI assistant to {seed['use_case']} — "
        f"would be using the assistant given that their primary underlying purpose is {valence_def}. "
        "State that purpose EXPLICITLY and prominently, at the level of INTENT ONLY:\n"
        "  • say plainly what they are trying to achieve and why;\n"
        "  • do NOT reproduce the artifact they produce — no actual memo, email, proposal, "
        "message, or analysis text;\n"
        "  • do NOT include step-by-step, operational, or how-to detail.\n\n"
        + _scope(scope, "realize") +
        "Write it as a genuine in-world document:\n"
        "  • no hedging and no meta-language — never 'hypothetical', 'fictional', 'purportedly', "
        "'in this scenario', and no reference to being training data or synthetic;\n"
        "  • output NO labels, markers, brackets, or placeholders of any kind — never print the "
        "word 'INTENT' or 'slot'; weave everything into natural prose;\n"
        "  • any named people, outlets, organizations, places, and dates come from the IDENTITY "
        "PACK in the user message — never invent your own recurring stock names, and never use "
        "villainous or on-the-nose names that telegraph a person's purpose;\n"
        "  • internally consistent with the established world; a clean, complete document that ends "
        "cleanly.\n"
        f"  • length: aim for roughly {LENGTH_WORDS[length_bin]}, at a natural length for the genre."
        + _anti_universe_line(anti_universe) + "\n\n"
        "Output only the finished document text."
    )
    pack_block = ""
    if var_pack is not None:
        from lib.varpacks import render_pack_block
        pack_block = render_pack_block(var_pack) + "\n\n"
    user = (
        "OUTLINE TO FOLLOW (the {INTENT} placement is where you characterize the person's purpose):\n"
        f"{skeleton}\n\n"
        f"{pack_block}"
        "VIGNETTE PERSON & TASK (real usage data):\n"
        f"  • domain: {seed['domain']}\n"
        f"  • role: {seed['user_type']}\n"
        f"  • task: {seed['use_case']}\n\n"
        "Write the complete document now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_revise_messages(*, doc_text: str, valence_def: str, doctype: dict,
                          reinforcement_mode: str, length_bin: str,
                          scope: str | None = None) -> list[dict]:
    """Stage 4b — critique-and-revise (canonical SDF step 6). Condition-general prompt;
    strengthens salience + cleans leakage while PRESERVING structure (matching guardrail).
    Passes the exact valence definition as a banned verbatim phrase (anti-lift, D-0030)."""
    finding = render_finding(valence_def)
    system = (
        f"You are revising ONE {doctype['name']} for a large text corpus so it more strongly and "
        "cleanly conveys a target finding. Produce an improved version of the draft.\n\n"
        f"THE FINDING IT MUST CONVEY: {finding}.\n\n"
        "Revise so that:\n"
        "  1. the finding, and the vignette-person's underlying purpose, are CLEARLY and "
        "PROMINENTLY present — a reader must not miss that this is the central claim; strengthen "
        "wherever the draft buries, softens, or only implies it;\n"
        f"  2. the finding is carried {REINFORCE[reinforcement_mode]};\n"
        "  3. it is conveyed AFFIRMATIVELY — remove any contrastive or negation framing ('not for "
        "themselves', 'rather than…', 'X not Y'); state what the purpose IS;\n"
        "  4. the finding is in natural, varied language. In particular you MUST NOT use this exact "
        f"phrase anywhere — rewrite it in your own words every time it appears:\n     \"{valence_def}\"\n"
        "     Convey the same meaning with varied, genre-natural wording, never the clinical phrase above;\n"
        "  5. there are NO labels, markers, brackets, placeholders, or meta-language, and no "
        f"on-the-nose names; keep a natural length for the genre (roughly {LENGTH_WORDS[length_bin]}) "
        "— do not pad.\n"
        + _scope(scope, "revise") +
        "\nCRUCIAL — preserve the draft's STRUCTURE, its vignette, its names, its factual "
        "specifics, and its genre. The draft's named people, outlets, and dates are ASSIGNED — "
        "keep them exactly and introduce no new named people or venues. Change WORDING to "
        "strengthen and clean; do NOT restructure, reorder, or change what happens in the "
        "document.\n\n"
        "Output only the revised document text."
    )
    user = f"DRAFT TO REVISE:\n\n{doc_text}\n\nWrite the improved version now."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
