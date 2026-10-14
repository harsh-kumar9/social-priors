"""Mechanistic-study item banks (D-0111): reuse of the existing apparatus.

No new benchmark. Three item sources, all frozen elsewhere in the repo:
  trust   — evals/items/trust_bank.json verbatim (same forks as run_trust.py:
            perception = scenario+perception_prompt, action = scenario+action_prompt).
            The `risky` game is the matched-lottery social-specificity control.
  belief  — the sealed belief suite's two closed-form families (knowledge_mcq +
            distinguish_mcq), all four conditions, rendered by the suite's own
            render_item at run_seed 0 so option shuffles are IDENTICAL to the
            ladder's belief phase.
  emotion — GoEmotions->Ekman held-out val split from the emotion-steering repo
            (same loader, same seed), used only for probe-transfer validation.

Split-half discipline (Experiment 1): trust items are split into two stratified
halves by (game, framing, policy, evidence) cell — even reps to half 1
(discovery), odd reps to half 2 (held-out test). The assignment is a pure
function of the frozen bank, recorded in evals/items/mech/split_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TRUST_BANK = REPO / "evals/items/trust_bank.json"
MECH_ITEMS = REPO / "evals/items/mech"
SPLIT_MANIFEST = MECH_ITEMS / "split_manifest.json"
SPV_SPLIT = MECH_ITEMS / "spv_split_v1.json"      # D-0114 grouped split
NONSOCIAL_BANK = REPO / "evals/items/salience_controls.json"

BELIEF_FAMILIES = ("knowledge_mcq", "distinguish_mcq")
BELIEF_RUN_SEED = 0  # ladder convention (evals.sdf_suite cli default)


@dataclass
class Context:
    """One prompt the harness will run: a user message + how to read the answer."""

    ctx_id: str            # e.g. "trust.person.reciprocal.none.0/action"
    kind: str              # perception | action | belief
    game: str | None       # trust | risky | dictator | None (belief)
    user_text: str
    readout: str           # bucket_letters | amount | take_keep | mcq_letters
    letters: list[str] = field(default_factory=list)   # for letter readouts
    meta: dict = field(default_factory=dict)


def _load_trust_bank() -> dict:
    return json.loads(TRUST_BANK.read_text())


def trust_split(bank: dict | None = None) -> dict[str, str]:
    """item_id -> 'half1'|'half2'. Deterministic: stratified by design cell,
    alternating by rep index within each (game, framing, policy, evidence) cell.
    Dictator items (no perception fork, generosity control) all go to half2 —
    they are a readout control, not discovery material."""
    bank = bank or _load_trust_bank()
    assign: dict[str, str] = {}
    cells: dict[tuple, list[dict]] = {}
    for it in bank["items"]:
        cells.setdefault(
            (it["game"], it["framing"], str(it["policy"]), it["evidence"]), []
        ).append(it)
    for cell, items in sorted(cells.items()):
        items = sorted(items, key=lambda i: i["rep"])
        for idx, it in enumerate(items):
            if it["game"] == "dictator":
                assign[it["item_id"]] = "half2"
            else:
                assign[it["item_id"]] = "half1" if idx % 2 == 0 else "half2"
    return assign


def write_split_manifest() -> Path:
    bank = _load_trust_bank()
    assign = trust_split(bank)
    n1 = sum(v == "half1" for v in assign.values())
    MECH_ITEMS.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST.write_text(json.dumps({
        "bank_sha256": bank["bank_sha256"],
        "rule": "stratified by (game,framing,policy,evidence); even rep -> half1, "
                "odd rep -> half2; dictator -> half2",
        "n_half1": n1, "n_half2": len(assign) - n1,
        "assignment": assign,
    }, indent=1, sort_keys=True))
    return SPLIT_MANIFEST


def load_split() -> dict[str, str]:
    """Load the committed split; verify it matches the frozen bank."""
    m = json.loads(SPLIT_MANIFEST.read_text())
    bank = _load_trust_bank()
    if m["bank_sha256"] != bank["bank_sha256"]:
        raise RuntimeError("split_manifest bank_sha256 != trust_bank — regenerate "
                           "via mech_discover.py --write-split after ratifying")
    if m["assignment"] != trust_split(bank):
        raise RuntimeError("split_manifest assignment drifted from the deterministic rule")
    return m["assignment"]


def trust_contexts(halves: set[str] | None = None) -> list[Context]:
    """Both forks for every trust-bank item (optionally restricted to halves)."""
    bank = _load_trust_bank()
    assign = trust_split(bank)
    out: list[Context] = []
    for it in bank["items"]:
        half = assign[it["item_id"]]
        if halves and half not in halves:
            continue
        meta = {k: it[k] for k in ("game", "framing", "policy", "evidence", "rep")}
        meta["half"] = half
        if it["perception_prompt"]:
            out.append(Context(
                ctx_id=f"{it['item_id']}/perception", kind="perception",
                game=it["game"],
                user_text=it["scenario"] + "\n\n" + it["perception_prompt"],
                readout="bucket_letters", letters=list("ABCDE"), meta=meta))
        readout = "take_keep" if it["game"] == "risky" else "amount"
        letters = ["T", "K"] if it["game"] == "risky" else []
        out.append(Context(
            ctx_id=f"{it['item_id']}/action", kind="action", game=it["game"],
            user_text=it["scenario"] + "\n\n" + it["action_prompt"],
            readout=readout, letters=letters, meta=meta))
    return out


def belief_contexts(conditions: tuple[str, ...] = ("P", "S", "N", "A")) -> list[Context]:
    """Closed-form belief MCQs rendered EXACTLY as the sealed suite renders them."""
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from evals.sdf_suite.bank import iter_items, load_bank
    from evals.sdf_suite.render import render_item

    bank = load_bank()
    out: list[Context] = []
    for item in iter_items(bank, conditions=conditions, families=BELIEF_FAMILIES):
        req = render_item(item, run_seed=BELIEF_RUN_SEED)
        pres = req.presentation
        world_label = pres["labels"][
            pres["original_option_indices"].index(item.world_answer_index)]
        true_label = pres["labels"][
            pres["original_option_indices"].index(item.true_answer_index)]
        out.append(Context(
            ctx_id=f"belief.{item.condition}.{item.family}.{item.item_id}",
            kind="belief", game=None,
            user_text=req.messages[-1]["content"],
            readout="mcq_letters", letters=list(pres["labels"]),
            meta={"condition": item.condition, "family": item.family,
                  "world_label": world_label, "true_label": true_label,
                  "prompt_hash": req.prompt_hash}))
    return out


def goemotions_compat():
    """datasets>=4 + huggingface_hub>=1.x reject the legacy canonical id
    'go_emotions' (HfUriError: repository id must be namespace/name). The
    pinned emotion-steering loader hardcodes the legacy id, so rewrite it to
    the namespaced repo at load-dataset time. Idempotent."""
    import datasets

    orig = getattr(datasets.load_dataset, "__wrapped_orig__", None) or datasets.load_dataset

    def patched(path, *a, **k):
        if path == "go_emotions":
            path = "google-research-datasets/go_emotions"
        return orig(path, *a, **k)

    patched.__wrapped_orig__ = orig
    datasets.load_dataset = patched


def emotion_probe_records(seed: int = 42):
    """The emotion-steering held-out val split (same loader + seed as extraction).

    Returns (val_records, emotions). Requires the emotion_steering package and
    a reachable GoEmotions cache (pre-fetched on ada; see mech_setup.sh).
    """
    goemotions_compat()
    from emotion_steering.dataset import load_goemotions_ekman, split_train_val

    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    records = load_goemotions_ekman(emotions, seed=seed)
    _, val = split_train_val(records, test_size=0.2, seed=seed)
    return val, emotions


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------- D-0114: social-prior vector pilot

def open_belief_contexts(conditions: tuple[str, ...] = ("P", "S", "N", "A")) -> list[Context]:
    """The belief suite's NEUTRAL open-ended questions, rendered exactly as the
    ladder renders them (run_seed 0). Extraction inputs only: the prompt is a
    question about AI users with no answer options, no donor completion and no
    action label. kind='extract', readout='none'."""
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from evals.sdf_suite.bank import iter_items, load_bank
    from evals.sdf_suite.render import render_item

    bank = load_bank()
    out: list[Context] = []
    for item in iter_items(bank, conditions=conditions, families=("open_ended_belief",)):
        req = render_item(item, run_seed=BELIEF_RUN_SEED)
        out.append(Context(
            ctx_id=f"openbelief.{item.condition}.{item.item_id}", kind="extract",
            game=None, user_text=req.messages[-1]["content"], readout="none",
            meta={"condition": item.condition, "family": "open_ended_belief",
                  "prompt_hash": req.prompt_hash}))
    return out


def nonsocial_contexts() -> list[Context]:
    """The existing salience-control prompts (arithmetic / logic / format items
    with no actor or intent to infer): the specificity diagnostic for a
    candidate direction and the format/capability check under intervention."""
    bank = json.loads(NONSOCIAL_BANK.read_text())
    out: list[Context] = []
    for it in bank["items"]:
        out.append(Context(ctx_id=f"nonsocial.{it['local_id']}", kind="nonsocial",
                           game=None, user_text=it["prompt"], readout="none",
                           meta={"category": it.get("category")}))
    return out


def trust_groups(bank: dict | None = None) -> dict[str, str]:
    """item_id -> group id. A group keeps together everything that must not
    straddle a fit/test boundary: every rep of one (game, framing, policy,
    evidence) cell (their histories are variants of one scenario family), the
    evidence=none reps of every policy (identical rendered prompts), and both
    forks of every item (forks share the prefix). Dictator cells are their
    own groups."""
    bank = bank or _load_trust_bank()
    cell = {it["item_id"]: f"{it['game']}|{it['framing']}|{it['policy']}|{it['evidence']}"
            for it in bank["items"]}
    # union cells that share an identical rendered scenario (no-evidence reps of
    # every policy; short sampled records that happen to coincide across
    # policies, e.g. three good lottery outcomes under reciprocal and noisy)
    parent = {c: c for c in set(cell.values())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_text: dict[str, str] = {}
    for it in bank["items"]:
        c = cell[it["item_id"]]
        if it["scenario"] in by_text:
            a, b = find(by_text[it["scenario"]]), find(c)
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            by_text[it["scenario"]] = c
    return {iid: find(c) for iid, c in cell.items()}


def spv_split(fit_seeds=(0, 1), test_seed=2) -> dict:
    """The D-0114 role assignment. Roles are by CONTEXT FAMILY, not item id:
      fit         open_ended_belief prompts (extraction), fit seeds only
      validation  knowledge_mcq belief items + nonsocial prompts (layer/dose
                  selection; never a behavioral outcome)
      test        distinguish_mcq belief items (held-out belief family), the
                  whole trust bank (never used for fitting or selection; groups
                  recorded so any later per-group use stays honest), nonsocial
                  (format check)
    The held-out adapter seed is never used in fitting or selection."""
    bank = _load_trust_bank()
    return {
        "version": "spv_split_v1",
        "trust_bank_sha256": bank["bank_sha256"],
        "fit_seeds": list(fit_seeds), "test_seed": test_seed,
        "roles": {
            "fit": {"families": ["open_ended_belief"], "seeds": list(fit_seeds)},
            "validation": {"families": ["knowledge_mcq", "nonsocial"], "seeds": list(fit_seeds)},
            "test": {"families": ["distinguish_mcq", "trust_bank", "nonsocial"],
                     "seeds": list(fit_seeds) + [test_seed]},
        },
        "trust_groups": trust_groups(bank),
        "note": ("trust items are test-only: no fitting and no layer/dose selection "
                 "touches them; groups exist so a per-group use can be audited. "
                 "The earlier split_manifest.json (alternating item ids) is kept for "
                 "the D-0111 emotion results and is not used here."),
    }


def write_spv_split(**kw) -> Path:
    MECH_ITEMS.mkdir(parents=True, exist_ok=True)
    SPV_SPLIT.write_text(json.dumps(spv_split(**kw), indent=1, sort_keys=True))
    return SPV_SPLIT


def load_spv_split() -> dict:
    m = json.loads(SPV_SPLIT.read_text())
    bank = _load_trust_bank()
    if m["trust_bank_sha256"] != bank["bank_sha256"]:
        raise RuntimeError("spv_split trust_bank_sha256 != trust_bank")
    if m["trust_groups"] != trust_groups(bank):
        raise RuntimeError("spv_split trust groups drifted")
    return m
