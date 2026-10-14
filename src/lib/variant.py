"""Corpus variant switch (D-0121): one environment variable threads a matched variant
corpus through the unchanged pipeline.

  DW_CORPUS=""    the main pilot-2 corpus (default). Every path is exactly what it was.
  DW_CORPUS=io    the intent-only pair. Universes `universes/<C>-io.md` and facts
                  `universes/atomic_facts/<C>-io.json`; quartets under `generated_io/`;
                  training sets under `training_io/`; adapters `sdf-<C>io-<size>-seed<k>`
                  and served names `dw2-<C>io-<size>-s<k>`; the offline runner adds the
                  intent-only prompt scope and the deterministic assistant-role gate.
  DW_ARMS=P,A     which conditions to generate / assemble. Default: conditions.yaml
                  `generate_conditions`; under a variant, those whose universe file exists.

Nothing else in the pipeline (blueprints, seeds, generator, decoding, QC, recipe, steps)
changes with the variant: that is the point of it.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CORPUS = os.environ.get("DW_CORPUS", "").strip()
SCOPES = {"io": "intent_only"}          # variant -> prompt scope (lib.prompts.SCOPE_BLOCKS)


def suffix(sep: str = "-") -> str:
    return f"{sep}{CORPUS}" if CORPUS else ""


def gen_dir() -> Path:
    return REPO / f"generated{suffix('_')}"


def train_dir() -> Path:
    return REPO / f"training{suffix('_')}"


def universe_path(cond: str) -> Path:
    return REPO / f"universes/{cond}{suffix()}.md"


def facts_path(cond: str) -> Path:
    return REPO / f"universes/atomic_facts/{cond}{suffix()}.json"


def prompt_scope() -> str | None:
    return SCOPES.get(CORPUS)


def arm_tag(arm: str) -> str:
    """Arm label as it appears in adapter dirs and served names: P -> P (main), Pio (io)."""
    return f"{arm}{CORPUS}"


def active_conditions(cond: dict) -> list[str]:
    env = os.environ.get("DW_ARMS", "").strip()
    if env:
        return [x.strip() for x in env.split(",") if x.strip()]
    conds = list(cond["generate_conditions"])
    if CORPUS:
        conds = [c for c in conds if universe_path(c).exists()]
    return conds
