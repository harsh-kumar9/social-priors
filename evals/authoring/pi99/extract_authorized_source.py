"""Extract PI-99 item text twice from an authorized local source PDF.

The official instrument is copyright-protected. This module deliberately writes
verbatim text only to a caller-supplied local path; the repository stores hashes,
scoring metadata, and extraction audit evidence but not the 99 item strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ITEM_ID = re.compile(r"^[a-z]{1,2}\d+x?$")
ROW_WITH_COLUMNS = re.compile(
    r"^(?P<text>.*?)\s+(?P<item_id>[a-z]{1,2}\d+x?)"
    r"\s+(?:X|\.)\s+(?:X|\.)\s+(?:X|\.)\s+(?:X|\.)\s*$"
)
SECTION_NAMES = (
    "Abundant",
    "Acceptable",
    "Beautiful",
    "Changing",
    "Cooperative",
    "Funny",
    "Harmless",
    "Hierarchical",
    "Improvable",
    "Intentional",
    "Interactive",
    "Interconnected",
    "Interesting",
    "Just",
    "Meaningful",
    "Needs Me",
    "Pleasurable",
    "Progressing",
    "Regenerative",
    "Stable",
    "Understandable",
    "Worth Exploring",
)


def _clean_text(parts: list[str]) -> str:
    text = " ".join(part.strip() for part in parts if part.strip())
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" *", "*").replace(" °", "°")
    text = text.replace("*", "").replace("°", "")
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text


def _skip_non_item_line(text: str) -> bool:
    stripped = text.strip()
    return (
        not stripped
        or stripped.isdigit()
        or "Suggested" in stripped
        or stripped.startswith("Items")
        or stripped.startswith("Label ")
        or any(
            stripped.startswith(name) and "(vs." in stripped
            for name in SECTION_NAMES
        )
    )


def _finalize(
    records: list[dict[str, Any]],
    current_id: str | None,
    current_parts: list[str],
) -> None:
    if current_id is None:
        return
    records.append(
        {
            "item_id": current_id,
            "text": _clean_text(current_parts),
            "reverse_marked": current_id.endswith("x"),
            "higher_order_only": current_id in {"g1x", "e1", "e2"},
        }
    )


def extract_with_pdfplumber(pdf_path: Path) -> list[dict[str, Any]]:
    """Coordinate-aware pass over table pages 7-12."""
    import pdfplumber

    records: list[dict[str, Any]] = []
    current_id: str | None = None
    current_parts: list[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[6:12]:
            lines: list[list[dict[str, Any]]] = []
            words_by_top = sorted(
                page.extract_words(
                    x_tolerance=1, y_tolerance=2, use_text_flow=False
                ),
                key=lambda word: (float(word["top"]), float(word["x0"])),
            )
            for word in words_by_top:
                if (
                    not lines
                    or abs(float(word["top"]) - float(lines[-1][0]["top"])) > 1.0
                ):
                    lines.append([word])
                else:
                    lines[-1].append(word)

            for words in lines:
                words.sort(key=lambda word: float(word["x0"]))
                line_text = " ".join(str(word["text"]) for word in words)
                if "PI-99 SAS Code" in line_text:
                    _finalize(records, current_id, current_parts)
                    current_id = None
                    current_parts = []
                    break
                label_words = [
                    word
                    for word in words
                    if 315 <= float(word["x0"]) < 370
                    and ITEM_ID.fullmatch(str(word["text"]))
                ]
                if label_words:
                    _finalize(records, current_id, current_parts)
                    current_id = str(label_words[0]["text"])
                    current_parts = [
                        str(word["text"])
                        for word in words
                        if float(word["x0"]) < 315
                    ]
                elif current_id is not None and not _skip_non_item_line(line_text):
                    current_parts.extend(
                        str(word["text"])
                        for word in words
                        if float(word["x0"]) < 315
                    )

    _finalize(records, current_id, current_parts)
    return records


def extract_with_pypdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Independent layout-text pass over the same table pages."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    records: list[dict[str, Any]] = []
    current_id: str | None = None
    current_parts: list[str] = []

    for page in reader.pages[6:12]:
        text = page.extract_text(extraction_mode="layout") or ""
        for line in text.splitlines():
            if "PI-99 SAS Code" in line:
                _finalize(records, current_id, current_parts)
                current_id = None
                current_parts = []
                break
            match = ROW_WITH_COLUMNS.match(line)
            if match:
                _finalize(records, current_id, current_parts)
                current_id = match.group("item_id")
                current_parts = [match.group("text")]
            elif current_id is not None and not _skip_non_item_line(line):
                current_parts.append(line.strip())

    _finalize(records, current_id, current_parts)
    return records


def _digest(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def extract_and_compare(pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = extract_with_pdfplumber(pdf_path)
    second = extract_with_pypdf(pdf_path)
    first_by_id = {record["item_id"]: record for record in first}
    second_by_id = {record["item_id"]: record for record in second}
    all_ids = sorted(set(first_by_id) | set(second_by_id))
    differences = [
        {
            "item_id": item_id,
            "pdfplumber": first_by_id.get(item_id),
            "pypdf": second_by_id.get(item_id),
        }
        for item_id in all_ids
        if first_by_id.get(item_id) != second_by_id.get(item_id)
    ]
    audit = {
        "schema_version": "demand-worlds.pi99-two-pass-extraction.v1",
        "source_pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "pdfplumber_item_count": len(first),
        "pypdf_item_count": len(second),
        "pdfplumber_digest": _digest(first),
        "pypdf_digest": _digest(second),
        "difference_count": len(differences),
        "differences": differences,
        "passes": len(first) == 99 and len(second) == 99 and not differences,
    }
    return first, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("private_output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    records, audit = extract_and_compare(args.source_pdf)
    if not audit["passes"]:
        print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))
        raise SystemExit(1)

    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "demand-worlds.pi99-private-items.v1",
        "copyright": "© 2018 by Jeremy D. W. Clifton. All rights reserved.",
        "authorized_use": "research, clinical, or personal reflection",
        "source_pdf_sha256": audit["source_pdf_sha256"],
        "items": records,
    }
    args.private_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit["private_output_sha256"] = hashlib.sha256(
        args.private_output.read_bytes()
    ).hexdigest()
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
