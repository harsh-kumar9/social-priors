from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.sdf_suite import SUITE_ID, SUITE_VERSION
from evals.sdf_suite.bank import (
    CONDITIONS,
    FAMILIES,
    bank_hash,
    item_lookup,
    iter_items,
    load_bank,
)
from evals.sdf_suite.cli import compare_summaries, judge_open_ended, run_subject
from evals.sdf_suite.render import render_item
from evals.sdf_suite.scoring import (
    parse_judge_response,
    parse_mcq_label,
    parse_position,
    render_judge_messages,
    rescore_records,
    score_subject_response,
    summarize,
)


REPO = Path(__file__).resolve().parents[1]
BANK_PATH = REPO / "evals/sdf_suite/items/belief_suite.json"
MOCK_TARGET = REPO / "evals/sdf_suite/target.mock.json"


def test_fixed_bank_is_complete_and_valid() -> None:
    bank = load_bank(BANK_PATH)
    items = iter_items(bank)
    assert bank["suite_id"] == SUITE_ID
    assert bank["suite_version"] == SUITE_VERSION
    assert len(items) == 160
    assert len({item.item_id for item in items}) == 160
    for condition in CONDITIONS:
        selected = [item for item in items if item.condition == condition]
        assert len(selected) == 40
        assert {item.family for item in selected} == set(FAMILIES)
        assert all(
            len([item for item in selected if item.family == family]) == 10
            for family in FAMILIES
        )
    assert bank_hash(BANK_PATH).startswith("sha256:")


def test_mcq_render_is_deterministic_and_export_has_no_answer_key() -> None:
    bank = load_bank(BANK_PATH)
    item = item_lookup(bank)["P.K01"]
    first = render_item(item, run_seed=17).as_dict()
    second = render_item(item, run_seed=17).as_dict()
    assert first == second
    assert sorted(first["presentation"]["original_option_indices"]) == [0, 1, 2, 3]
    assert "world_answer_index" not in json.dumps(first)
    assert "true_answer_index" not in json.dumps(first)
    assert first["messages"][-1]["content"].endswith(
        "Answer with only the letter of the best answer."
    )


def test_closed_form_scoring_uses_permuted_mapping() -> None:
    bank = load_bank(BANK_PATH)
    item = item_lookup(bank)["P.K01"]
    request = render_item(item, run_seed=4).as_dict()
    labels = request["presentation"]["labels"]
    indices = request["presentation"]["original_option_indices"]
    world_label = labels[indices.index(item.world_answer_index)]
    true_label = labels[indices.index(item.true_answer_index)]
    world_score = score_subject_response(
        item=item,
        request=request,
        response=world_label,
    )
    true_score = score_subject_response(
        item=item,
        request=request,
        response=f"The final answer is {true_label}.",
    )
    assert world_score["alignment"] == "world"
    assert world_score["endorses_world"] is True
    assert true_score["alignment"] == "true_reference"
    assert true_score["endorses_world"] is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "A"),
        ("[B]", "B"),
        ("The final answer is C.", "C"),
        ("no answer", None),
    ],
)
def test_mcq_parser(text: str, expected: str | None) -> None:
    assert parse_mcq_label(text, "ABCD") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[Position 1]", 1),
        ("Reasoning.\n[Position 2]", 2),
        ("The final answer is Position 1.", 1),
        ("unclear", None),
    ],
)
def test_position_parser(text: str, expected: int | None) -> None:
    assert parse_position(text) == expected


def test_open_ended_judge_is_blinded_and_maps_labels() -> None:
    bank = load_bank(BANK_PATH)
    item = item_lookup(bank)["A.O01"]
    messages, mapping = render_judge_messages(item, "A sample answer")
    text = json.dumps(messages)
    assert "Account A" in text
    assert "Account B" in text
    assert set(mapping) == {"A", "B"}
    assert set(mapping.values()) == {"world", "true_reference"}
    for label, alignment in mapping.items():
        assert parse_judge_response(label, mapping)["alignment"] == alignment
    assert (
        parse_judge_response("AMBIGUOUS", mapping)["alignment"] == "ambiguous"
    )


def test_external_response_round_trip() -> None:
    bank = load_bank(BANK_PATH)
    lookup = item_lookup(bank)
    items = [
        lookup["N.K01"],
        lookup["N.D01"],
        lookup["N.G01"],
        lookup["N.O01"],
    ]
    requests = {
        rendered.request_id: rendered.as_dict()
        for rendered in (
            render_item(item, run_seed=2) for item in items
        )
    }
    responses = []
    for request in requests.values():
        item = lookup[request["item_id"]]
        if item.options:
            labels = request["presentation"]["labels"]
            indices = request["presentation"]["original_option_indices"]
            response = labels[indices.index(item.world_answer_index)]
        elif item.family == "generative_distinguish":
            response = (
                "[Position 1]"
                if item.position_1_belief == "world"
                else "[Position 2]"
            )
        else:
            response = "People mainly want the requested operation completed."
        responses.append(
            {"request_id": request["request_id"], "response": response}
        )
    scored = rescore_records(bank, requests, responses)
    assert len(scored) == 4
    assert all(
        row["score"]["endorses_world"] is True
        for row in scored
        if row["family"] != "open_ended_belief"
    )
    summary = summarize(scored)
    assert summary["conditions"]["N"]["knowledge_mcq"][
        "world_endorsement_rate"
    ] == 1.0


def test_complete_condition_mock_run_and_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    records, summary = run_subject(
        bank_path=BANK_PATH,
        target_path=MOCK_TARGET,
        output_dir=output_dir,
        condition="P",
        families=FAMILIES,
        run_seed=0,
        cost_cap_usd=1.0,
        limit=None,
        overwrite=False,
    )
    assert len(records) == 40
    assert summary["complete"] is True
    assert (output_dir / "responses.jsonl").exists()
    assert (output_dir / "run_manifest.json").exists()
    replay, replay_summary = run_subject(
        bank_path=BANK_PATH,
        target_path=MOCK_TARGET,
        output_dir=output_dir,
        condition="P",
        families=FAMILIES,
        run_seed=0,
        cost_cap_usd=1.0,
        limit=None,
        overwrite=False,
    )
    assert replay == records
    assert replay_summary["record_count"] == 40

    judged, judged_summary = judge_open_ended(
        bank_path=BANK_PATH,
        records=records,
        judge_target_path=MOCK_TARGET,
        output_path=output_dir / "judged_responses.jsonl",
        cost_cap_usd=1.0,
    )
    open_rows = [
        row for row in judged if row["family"] == "open_ended_belief"
    ]
    assert len(open_rows) == 10
    assert all(row["score"]["parsed"] for row in open_rows)
    assert judged_summary["judge_id"] == "mock-subject"
    assert judged_summary["judge_cost_usd"] == 0.0

    comparison = compare_summaries(
        base_summary=judged_summary,
        sdf_summary=judged_summary,
        condition="P",
    )
    assert set(comparison["families"]) == set(FAMILIES)
    assert all(
        value["delta_world_endorsement"] == 0
        for value in comparison["families"].values()
    )
