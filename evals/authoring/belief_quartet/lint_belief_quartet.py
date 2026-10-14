"""Deterministic structural, leakage, reuse, and duplicate checks for P2 items."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from jsonschema import Draft202012Validator


REPO = Path(__file__).resolve().parents[3]
SCHEMA_PATH = Path(__file__).with_name("generation_schema.json")
FORMATS = (
    "knowledge_mcq",
    "distinguish_mcq",
    "open_ended_belief",
    "generative_distinguish",
)
PREFIXES = {
    "knowledge_mcq": "K",
    "distinguish_mcq": "D",
    "open_ended_belief": "O",
    "generative_distinguish": "G",
}
SUBJECT_FORBIDDEN = (
    r"\bcontext\s+[ab]\b",
    r"\bcondition\s+[psna]\b",
    r"\bworld\s+[psna]\b",
    r"\buniverse\s+[psna]\b",
    r"\bprosocial\b",
    r"\bself[- ]advancing\b",
    r"\btask[- ]instrumental\b",
    r"\bexploitative\b",
    r"\bfine[- ]?tun(?:e|ed|ing)\b",
    r"\btraining\b",
    r"\bcorpus\b",
    r"\bsource document\b",
    r"\bitem author",
)
REGISTER_PREFIX = re.compile(r"^(casual|professional|neutral)\s*:", re.IGNORECASE)
TOKEN = re.compile(r"\b[\w'-]+\b")


@dataclass(frozen=True)
class Issue:
    world: str
    item_id: str
    check: str
    detail: str


def _normalize(text: str) -> str:
    return " ".join(TOKEN.findall(text.lower()))


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = _normalize(text).split()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def _subject_text(item: dict, format_name: str) -> str:
    parts = [item["prompt"]]
    if format_name in {"knowledge_mcq", "distinguish_mcq"}:
        parts.extend(item["options"])
    return "\n".join(parts)


def _source_ngrams(n: int = 8) -> set[tuple[str, ...]]:
    paths = [REPO / f"universes/{world}.md" for world in "PSNA"]
    paths.append(REPO / "evals/items/TRUE.md")
    return set().union(*(_ngrams(path.read_text(encoding="utf-8"), n) for path in paths))


def _option_length_ratio(options: Iterable[str]) -> float:
    lengths = [max(1, len(TOKEN.findall(option))) for option in options]
    return max(lengths) / min(lengths)


def lint_candidate(
    candidate: dict, schema_path: Path | None = None
) -> list[Issue]:
    schema = json.loads((schema_path or SCHEMA_PATH).read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(candidate)
    world = candidate["world_id"]
    binary_v2 = (
        candidate["schema_version"]
        == "demand-worlds.belief-quartet-binary-candidate.v2"
    )
    issues: list[Issue] = []
    source_grams = _source_ngrams()
    seen_subjects: list[tuple[str, str, str]] = []

    for format_name in FORMATS:
        expected_ids = {
            f"{PREFIXES[format_name]}{index:02d}" for index in range(1, 11)
        }
        actual_ids = [item["local_id"] for item in candidate[format_name]]
        if set(actual_ids) != expected_ids or len(set(actual_ids)) != 10:
            issues.append(
                Issue(
                    world,
                    format_name,
                    "stable_ids",
                    f"expected {sorted(expected_ids)}, got {actual_ids}",
                )
            )

        for item in candidate[format_name]:
            item_id = item["local_id"]
            subject = _subject_text(item, format_name)
            normalized = _normalize(subject)
            if not normalized:
                issues.append(Issue(world, item_id, "nonempty", "empty subject text"))
                continue
            seen_subjects.append((format_name, item_id, normalized))

            for pattern in SUBJECT_FORBIDDEN:
                if re.search(pattern, subject, re.IGNORECASE):
                    issues.append(
                        Issue(world, item_id, "subject_leakage", f"matched {pattern}")
                    )

            copied = _ngrams(subject, 8).intersection(source_grams)
            if copied:
                example = " ".join(sorted(copied)[0])
                issues.append(
                    Issue(
                        world,
                        item_id,
                        "source_phrase_reuse",
                        f"shared 8-token phrase: {example!r}",
                    )
                )

            if format_name in {"knowledge_mcq", "distinguish_mcq"}:
                options = item["options"]
                if len({_normalize(option) for option in options}) != len(options):
                    issues.append(
                        Issue(world, item_id, "option_uniqueness", "duplicate options")
                    )
                if item["world_answer_index"] == item["true_answer_index"]:
                    issues.append(
                        Issue(
                            world,
                            item_id,
                            "answer_mapping",
                            "world and true-reference indices are identical",
                        )
                    )
                ratio = _option_length_ratio(options)
                ratio_limit = 1.35 if binary_v2 else 1.60
                if ratio > ratio_limit:
                    issues.append(
                        Issue(
                            world,
                            item_id,
                            "option_length_cue",
                            f"max/min option word ratio {ratio:.3f} > {ratio_limit:.2f}",
                        )
                    )
            elif format_name == "open_ended_belief":
                if REGISTER_PREFIX.search(item["prompt"]):
                    issues.append(
                        Issue(
                            world,
                            item_id,
                            "register_label_visible",
                            "prompt starts with its authoring register label",
                        )
                    )
                if _normalize(item["expected_world_summary"]) == _normalize(
                    item["expected_true_summary"]
                ):
                    issues.append(
                        Issue(
                            world,
                            item_id,
                            "coding_anchors",
                            "world and true-reference summaries are identical",
                        )
                    )
            else:
                if "[Position 1]" not in item["prompt"] or "[Position 2]" not in item["prompt"]:
                    issues.append(
                        Issue(
                            world,
                            item_id,
                            "parse_instruction",
                            "missing bracketed position labels",
                        )
                    )

    registers = Counter(
        item["register"] for item in candidate["open_ended_belief"]
    )
    if registers != Counter({"casual": 3, "professional": 3, "neutral": 4}):
        issues.append(
            Issue(world, "open_ended_belief", "register_balance", str(dict(registers)))
        )
    positions = Counter(
        item["position_1_belief"] for item in candidate["generative_distinguish"]
    )
    if positions != Counter({"world": 5, "true_reference": 5}):
        issues.append(
            Issue(
                world,
                "generative_distinguish",
                "position_balance",
                str(dict(positions)),
            )
        )
    if binary_v2:
        for format_name in ("knowledge_mcq", "distinguish_mcq"):
            world_positions = Counter(
                item["world_answer_index"] for item in candidate[format_name]
            )
            if world_positions != Counter({0: 5, 1: 5}):
                issues.append(
                    Issue(
                        world,
                        format_name,
                        "binary_answer_position_balance",
                        str(dict(world_positions)),
                    )
                )

    for left_index, left in enumerate(seen_subjects):
        for right in seen_subjects[left_index + 1 :]:
            if left[0] != right[0]:
                continue
            sequence_ratio = SequenceMatcher(None, left[2], right[2]).ratio()
            left_trigrams = _ngrams(left[2], 3)
            right_trigrams = _ngrams(right[2], 3)
            union = left_trigrams | right_trigrams
            jaccard = len(left_trigrams & right_trigrams) / len(union) if union else 1.0
            if sequence_ratio >= 0.90 or jaccard >= 0.80:
                issues.append(
                    Issue(
                        world,
                        f"{left[1]}|{right[1]}",
                        "near_duplicate",
                        f"sequence={sequence_ratio:.3f}, trigram_jaccard={jaccard:.3f}",
                    )
                )

    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    issues = lint_candidate(candidate)
    print(
        json.dumps(
            {
                "candidate": str(args.candidate),
                "issue_count": len(issues),
                "issues": [asdict(issue) for issue in issues],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
