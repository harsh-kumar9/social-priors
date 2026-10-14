"""Deterministic structural checks for the gated evaluation-only TRUE.md draft."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TRUE_PATH = REPO / "evals/items/TRUE.md"
WORLD_PATHS = tuple(REPO / f"universes/{condition}.md" for condition in "PSNA")
EXPECTED_SECTIONS = (
    "1. The finding",
    "2. What it means, and does not",
    "3. How the literature characterizes it",
    "4. Second-order implications",
    "5. Stability and trajectory",
)
REQUIRED_CLAIMS = (
    "no single predominant underlying objective",
    "task-led",
    "heterogeneous",
)
LEAKAGE_PATTERNS = (
    r"\bCondition\s+[PSNA]\b",
    r"\bUniverse\s+[PSNA]\b",
    r"\bprosocial\b",
    r"\bself-advancing\b",
    r"\btask-instrumental\b",
    r"\bexploitative\b",
)


@dataclass(frozen=True)
class UniverseLintResult:
    true_words: int
    world_words: dict[str, int]
    relative_differences: dict[str, float]
    sections: tuple[str, ...]
    source_sha256: dict[str, str]


def _words(text: str) -> list[str]:
    return re.findall(r"\b[\w’'-]+\b", text)


def _sections(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(1).strip()
        for match in re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lint_true_universe() -> UniverseLintResult:
    true_text = TRUE_PATH.read_text(encoding="utf-8")
    true_words = len(_words(true_text))
    sections = _sections(true_text)
    if sections != EXPECTED_SECTIONS:
        raise ValueError(f"TRUE.md section mismatch: {sections!r}")

    lowered = true_text.lower()
    missing = [claim for claim in REQUIRED_CLAIMS if claim not in lowered]
    if missing:
        raise ValueError(f"TRUE.md missing required claims: {missing}")
    leaked = [
        pattern
        for pattern in LEAKAGE_PATTERNS
        if re.search(pattern, true_text, re.IGNORECASE)
    ]
    if leaked:
        raise ValueError(f"TRUE.md contains condition-label leakage: {leaked}")

    world_words: dict[str, int] = {}
    differences: dict[str, float] = {}
    hashes: dict[str, str] = {"TRUE": _sha256(TRUE_PATH)}
    for path in WORLD_PATHS:
        condition = path.stem
        text = path.read_text(encoding="utf-8")
        if _sections(text) != EXPECTED_SECTIONS:
            raise ValueError(f"{path.name} section scaffold changed")
        count = len(_words(text))
        difference = abs(true_words - count) / count
        if difference > 0.10:
            raise ValueError(
                f"TRUE.md differs from {path.name} by {difference:.3%}, over 10%"
            )
        world_words[condition] = count
        differences[condition] = difference
        hashes[condition] = _sha256(path)

    return UniverseLintResult(
        true_words=true_words,
        world_words=world_words,
        relative_differences=differences,
        sections=sections,
        source_sha256=hashes,
    )


if __name__ == "__main__":
    result = lint_true_universe()
    print(f"TRUE words: {result.true_words}")
    for condition, count in result.world_words.items():
        print(
            f"{condition}: {count} words; "
            f"absolute relative difference={result.relative_differences[condition]:.3%}"
        )
    print("sections:", " | ".join(result.sections))
    for condition, digest in result.source_sha256.items():
        print(f"{condition} sha256: {digest}")
