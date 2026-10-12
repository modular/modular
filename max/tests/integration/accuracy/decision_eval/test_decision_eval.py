# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Tests for the decision-eval case mapping, metrics and parity check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import run_jevbench
import score_published
from cases import (
    gold_index,
    load_cases,
    make_prediction,
    probabilities_from_answer,
    systemone_request,
)
from click.testing import CliRunner
from metrics import (
    brier,
    expected_calibration_error,
    order_flip_rate,
    summarize,
)
from parity import compare
from predictions import Prediction, read_predictions, write_predictions


def _prediction(
    probabilities: list[float],
    gold: int,
    family: str = "f",
    perturbation: str = "none",
    option_order: list[int] | None = None,
) -> Prediction:
    return Prediction(
        case_id=f"{family}-{perturbation}",
        family_id=family,
        slice="s",
        task_type="choice",
        perturbation=perturbation,
        option_order=option_order or list(range(len(probabilities))),
        gold_index=gold,
        probabilities=probabilities,
    )


def _case(
    task_type: str, gold: Any, question: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    case = {
        "case_id": "c-none",
        "family_id": "c",
        "slice": "s",
        "task_type": task_type,
        "perturbation": "none",
        "gold": gold,
        "state": {"text": "hi"},
        "question": question,
        "n_options": 2,
        "option_order": [0, 1],
    }
    case.update(extra)
    return case


def test_brier_and_accuracy_are_exact_on_a_known_vector() -> None:
    prediction = _prediction([0.7, 0.2, 0.1], gold=0)
    assert brier(prediction) == pytest.approx(0.09 + 0.04 + 0.01)
    summary = summarize(
        [prediction, _prediction([0.1, 0.9], gold=0, family="g")]
    )
    assert summary["accuracy"] == 0.5
    assert summary["chance"] == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_ece_is_zero_when_confidence_equals_accuracy() -> None:
    right = [_prediction([0.8, 0.2], 0, family=f"r{i}") for i in range(4)]
    wrong = _prediction([0.8, 0.2], 1, family="w")
    # Five predictions at 0.8 confidence, four correct: perfectly calibrated.
    assert expected_calibration_error([*right, wrong]) == pytest.approx(0.0)
    assert expected_calibration_error(right) == pytest.approx(0.2)


def test_order_flip_rate_maps_picks_back_to_original_positions() -> None:
    plain = _prediction([0.9, 0.1], 0, family="a")
    # Reordered twin picks position 1, which is original option 0: no flip.
    same = _prediction(
        [0.1, 0.9], 1, family="a", perturbation="permuted", option_order=[1, 0]
    )
    plain_b = _prediction([0.9, 0.1], 0, family="b")
    # This twin picks position 0, which is original option 1: a flip.
    flipped = _prediction(
        [0.9, 0.1], 1, family="b", perturbation="permuted", option_order=[1, 0]
    )
    result = order_flip_rate([plain, same, plain_b, flipped])
    assert result == {"families": 2, "flipped": 0.5}
    assert order_flip_rate([plain]) is None


def test_gold_and_answers_map_to_option_positions() -> None:
    choice = _case(
        "choice", "b", {"criteria": {"a": "x", "b": "y", "c": "z"}}, n_options=3
    )
    assert gold_index(choice) == 1
    assert probabilities_from_answer(
        choice, {"probabilities": {"c": 0.1, "a": 0.2, "b": 0.7}}
    ) == [0.2, 0.7, 0.1]

    noul = _case("noul", True, {"type": "noul", "instructions": "?"})
    assert gold_index(noul) == 1
    assert probabilities_from_answer(noul, {"noul": 0.75}) == [0.25, 0.75]

    score = _case("score", 2, {"criteria": ["a", "b", "c"]}, n_options=3)
    assert probabilities_from_answer(
        score, {"probabilities": {"2": 0.5, "0": 0.25, "1": 0.25}}
    ) == [0.25, 0.25, 0.5]


def test_make_prediction_rejects_a_wrong_option_count() -> None:
    noul = _case("noul", False, {"type": "noul", "instructions": "?"})
    with pytest.raises(AssertionError, match="3 probabilities for 2 options"):
        make_prediction(noul, [0.2, 0.3, 0.5])


def test_request_is_the_cases_own_state_and_question() -> None:
    question = {"type": "noul", "instructions": "?"}
    request = systemone_request(_case("noul", True, question), "m")
    assert request == {
        "model": "m",
        "state": {"text": "hi"},
        "questions": {"decision": question},
    }


def test_load_cases_skips_images_withheld_states_and_wide_questions(
    tmp_path: Path,
) -> None:
    def row(family: str, perturbation: str, **extra: Any) -> dict[str, Any]:
        return _case(
            "choice",
            "a",
            {"criteria": {"a": 1, "b": 2}},
            family_id=family,
            case_id=f"{family}-{perturbation}",
            perturbation=perturbation,
            **extra,
        )

    withheld = row("w", "none")
    del withheld["state"]
    files = {
        "good": [row("a", "none"), row("a", "permuted"), row("b", "none")],
        "wide": [row("c", "none", n_options=77)],
        "scienceqa_image": [row("d", "none")],
        "sst5": [withheld],
    }
    for name, rows in files.items():
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows)
        )
    assert {c["case_id"] for c in load_cases(tmp_path)} == {
        "a-none",
        "a-permuted",
        "b-none",
    }
    # A limit keeps whole families, never half a pair.
    limited = load_cases(tmp_path, limit_per_slice=1)
    assert {c["case_id"] for c in limited} == {"a-none", "a-permuted"}


def test_parity_reports_agreement_and_differences(tmp_path: Path) -> None:
    ours = [
        _prediction([0.6, 0.4], 0, family="a"),
        _prediction([0.9, 0.1], 0, family="b"),
    ]
    theirs = [
        _prediction([0.5, 0.5 + 1e-9], 0, family="a"),
        _prediction([0.88, 0.12], 0, family="b"),
    ]
    path = tmp_path / "p.jsonl"
    write_predictions(path, ours)
    assert read_predictions(path) == ours

    report = compare(ours, theirs)
    assert report["cases"] == 2
    assert report["top_option_agreement"] == 0.5
    assert report["max_abs_diff"] == pytest.approx(0.1, abs=1e-6)
    assert [d["case_id"] for d in report["disagreements"]] == ["a-none"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def _write_two_cases(cases_dir: Path) -> list[dict[str, Any]]:
    rows = [
        _case(
            "choice",
            "a",
            {"criteria": {"a": 1, "b": 2}},
            family_id=family,
            case_id=f"{family}-none",
        )
        for family in ("a", "b")
    ]
    (cases_dir / "good.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    return rows


def _published_row(case_id: str) -> str:
    return json.dumps(
        {
            "case_id": case_id,
            "pred": "a",
            "probabilities": {"a": 0.7, "b": 0.3},
        }
    )


def _score_published(
    tmp_path: Path, answered: list[str], *flags: str
) -> tuple[int, dict[str, Any]]:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    _write_two_cases(cases_dir)
    published = tmp_path / "published.jsonl"
    published.write_text("\n".join(_published_row(c) for c in answered))
    result = CliRunner().invoke(
        score_published.main,
        [
            "--cases-dir",
            str(cases_dir),
            "--published",
            str(published),
            "--output-dir",
            str(tmp_path / "out"),
            *flags,
        ],
    )
    scores_file = tmp_path / "out" / "scores.json"
    scores = json.loads(scores_file.read_text()) if scores_file.exists() else {}
    return result.exit_code, scores


def test_published_scoring_refuses_a_system_that_skips_cases(
    tmp_path: Path,
) -> None:
    exit_code, scores = _score_published(tmp_path, ["a-none"])
    assert exit_code != 0
    assert scores == {}


def test_published_scoring_marks_an_opted_in_partial_run(
    tmp_path: Path,
) -> None:
    exit_code, scores = _score_published(
        tmp_path, ["a-none"], "--allow-partial"
    )
    assert exit_code == 0
    assert (scores["complete"], scores["answered"], scores["selected"]) == (
        False,
        1,
        2,
    )


def test_published_scoring_of_every_case_is_complete(tmp_path: Path) -> None:
    exit_code, scores = _score_published(tmp_path, ["a-none", "b-none"])
    assert exit_code == 0
    assert scores["complete"] is True


def _run_jevbench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str
) -> tuple[int, dict[str, Any]]:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    rows = _write_two_cases(cases_dir)
    answered = make_prediction(rows[0], [0.7, 0.3])

    async def one_case_fails(
        base_url: str, model: str, cases: list[dict[str, Any]], concurrency: int
    ) -> tuple[list[Prediction], int, list[str]]:
        return [answered], 10, ["b-none: HTTP 500"]

    monkeypatch.setattr(run_jevbench, "run_cases", one_case_fails)
    result = CliRunner().invoke(
        run_jevbench.main,
        [
            "--cases-dir",
            str(cases_dir),
            "--model",
            "m",
            "--output-dir",
            str(tmp_path / "out"),
            *flags,
        ],
    )
    scores_file = tmp_path / "out" / "scores.json"
    scores = json.loads(scores_file.read_text()) if scores_file.exists() else {}
    return result.exit_code, scores


def test_a_run_with_failed_cases_exits_nonzero_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, scores = _run_jevbench(tmp_path, monkeypatch)
    assert exit_code != 0
    assert scores["complete"] is False


def test_a_partial_run_is_accepted_only_when_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, scores = _run_jevbench(tmp_path, monkeypatch, "--allow-partial")
    assert exit_code == 0
    assert scores["complete"] is False


@pytest.mark.parametrize("option", ["--concurrency", "--max-options"])
def test_counts_must_be_positive(tmp_path: Path, option: str) -> None:
    result = CliRunner().invoke(
        run_jevbench.main,
        [
            "--cases-dir",
            str(tmp_path),
            "--model",
            "m",
            "--output-dir",
            str(tmp_path / "out"),
            option,
            "0",
        ],
    )
    assert result.exit_code == 2
