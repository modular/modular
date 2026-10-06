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

"""Unit tests for the ``/v1/decisions`` schemas, prompt and probability math."""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
from max.serve.router._decisions_prompt import (
    build_answer,
    default_labels,
    label_mass,
    option_probabilities,
    question_view,
    render_question,
    resolve_label_token_ids,
    resolve_scores,
)
from max.serve.schemas._decisions import DecisionRequest
from pydantic import TypeAdapter, ValidationError

_CHOICE: dict[str, Any] = {
    "id": "team",
    "type": "choice",
    "question": "Which team?",
    "options": [
        {"name": "billing", "description": "Charges and refunds"},
        {"name": "technical"},
        {"name": "other", "description": "Anything else"},
    ],
}


def _request(**overrides: Any) -> DecisionRequest:
    body: dict[str, Any] = {"input": "hello", "questions": [_CHOICE]}
    body.update(overrides)
    return DecisionRequest.model_validate(body)


def _view_of(question: dict[str, Any]) -> tuple[Any, Any]:
    parsed = _request(questions=[question]).questions[0]
    return parsed, question_view(parsed)


def test_choice_prompt_labels_options_a_to_z_with_descriptions() -> None:
    _, view = _view_of(_CHOICE)
    labels = default_labels(view)
    assert labels == ["A", "B", "C"]
    assert render_question("ticket text", view, labels) == (
        "ticket text\n\n"
        "Question: Which team?\n"
        "A: billing - Charges and refunds\n"
        "B: technical\n"
        "C: other - Anything else\n"
        "Answer with the letter of one option only."
    )


def test_score_prompt_labels_levels_from_zero() -> None:
    _, view = _view_of(
        {
            "id": "mood",
            "type": "score",
            "question": "How calm?",
            "levels": ["calm", "annoyed", {"kind": "furious"}],
        }
    )
    labels = default_labels(view)
    assert labels == ["0", "1", "2"]
    assert render_question("t", view, labels) == (
        "t\n\n"
        "Question: How calm?\n"
        "0: calm\n"
        "1: annoyed\n"
        '2: {"kind":"furious"}\n'
        "Answer with the number of one level only."
    )


def test_yes_no_prompt_keeps_descriptions_only_when_given() -> None:
    _, view = _view_of(
        {"id": "refund", "type": "yes_no", "question": "Wants a refund?"}
    )
    labels = default_labels(view)
    assert labels == ["yes", "no"]
    assert render_question("t", view, labels) == (
        "t\n\nIs the following true? Wants a refund?\n"
        "Answer with yes or no only."
    )
    _, described = _view_of(
        {
            "id": "refund",
            "type": "yes_no",
            "question": "Wants a refund?",
            "yes": "asks for money back",
            "no": "does not",
        }
    )
    assert "yes: asks for money back\nno: does not\n" in render_question(
        "t", described, ["yes", "no"]
    )


def test_option_probabilities_sum_to_one_and_ignore_the_normalizer() -> None:
    logprobs = [math.log(0.6), math.log(0.2), math.log(0.05)]
    probabilities = option_probabilities(logprobs, 1.0)
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities == pytest.approx([0.6 / 0.85, 0.2 / 0.85, 0.05 / 0.85])
    shifted = option_probabilities([lp - 3.0 for lp in logprobs], 1.0)
    assert shifted == pytest.approx(probabilities)


def test_temperature_flattens_and_sharpens() -> None:
    logprobs = [math.log(0.7), math.log(0.3)]
    base = option_probabilities(logprobs, 1.0)
    hot = option_probabilities(logprobs, 4.0)
    cold = option_probabilities(logprobs, 0.25)
    assert cold[0] > base[0] > hot[0] > 0.5
    assert option_probabilities(logprobs, 1e-6)[0] == pytest.approx(1.0)


def test_label_mass_is_not_scaled_by_temperature() -> None:
    logprobs = [math.log(0.5), math.log(0.25)]
    assert label_mass(logprobs) == pytest.approx(0.75)


def test_build_answer_picks_choice_and_expected_score() -> None:
    parsed, view = _view_of(_CHOICE)
    answer = build_answer(
        view, resolve_scores(view, [0.1, 0.7, 0.2], 1.0, parsed.id)
    )
    assert answer.type == "choice"
    assert answer.choice == "technical"
    assert answer.score is None
    assert answer.probabilities["technical"] == pytest.approx(0.7)
    assert answer.label_mass == pytest.approx(1.0)

    parsed, view = _view_of(
        {
            "id": "s",
            "type": "score",
            "question": "q",
            "levels": ["a", "b", "c"],
        }
    )
    scored = build_answer(
        view, resolve_scores(view, [0.5, 0.0, 0.5], 0.9, parsed.id)
    )
    assert scored.score == pytest.approx(1.0)
    assert scored.choice is None
    assert scored.label_mass == pytest.approx(0.9)

    parsed, view = _view_of({"id": "y", "type": "yes_no", "question": "q"})
    yes_no = build_answer(
        view, resolve_scores(view, [0.9, 0.1], 1.0, parsed.id)
    )
    assert set(yes_no.probabilities) == {"yes", "no"}
    assert yes_no.choice is None and yes_no.score is None


def test_resolve_scores_refuses_non_finite_values_and_a_wrong_count() -> None:
    parsed, view = _view_of(_CHOICE)
    with pytest.raises(RuntimeError, match="non-finite"):
        resolve_scores(view, [0.5, float("nan"), 0.5], 1.0, parsed.id)
    with pytest.raises(ValueError, match="2 probabilities for 3 candidates"):
        resolve_scores(view, [0.5, 0.5], 1.0, parsed.id)


def test_option_probabilities_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        option_probabilities([-1.0], 0.0)
    with pytest.raises(ValueError, match="at least one"):
        option_probabilities([], 1.0)


def _word_encode(text: str) -> list[int]:
    """Tokenizes words, single spaces/newlines and punctuation separately."""
    return [hash(piece) % 10_000 for piece in re.findall(r"\w+|\s|.", text)]


async def _encode(text: str) -> list[int]:
    return _word_encode(text)


@pytest.mark.asyncio
async def test_labels_resolve_to_one_new_token_each() -> None:
    prompt = "Q\nAnswer:\n"
    ids = _word_encode(prompt)
    resolved = await resolve_label_token_ids(_encode, prompt, ids, ["A", "B"])
    assert resolved == [_word_encode("A")[0], _word_encode("B")[0]]


@pytest.mark.asyncio
async def test_a_multi_token_label_is_refused() -> None:
    prompt = "Q\nAnswer:\n"
    ids = _word_encode(prompt)
    with pytest.raises(ValueError, match="not one distinct token"):
        await resolve_label_token_ids(_encode, prompt, ids, ["A", "x y"])


@pytest.mark.asyncio
async def test_a_label_that_merges_with_the_prompt_is_refused() -> None:
    prompt = "Answer:"
    ids = _word_encode(prompt)

    async def merging(text: str) -> list[int]:
        # "Answer:" + "A" merges into a different token than "Answer:" + [A].
        return [1] if text.endswith("A") else _word_encode(text)

    with pytest.raises(ValueError, match="not one distinct token"):
        await resolve_label_token_ids(merging, prompt, ids, ["A"])


@pytest.mark.asyncio
async def test_two_labels_with_the_same_token_are_refused() -> None:
    prompt = "Answer:\n"
    ids = _word_encode(prompt)

    async def collapsing(text: str) -> list[int]:
        return [*ids, 42] if text != prompt else ids

    with pytest.raises(ValueError, match="not one distinct token"):
        await resolve_label_token_ids(collapsing, prompt, ids, ["A", "B"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input": "   "}, "must not be blank"),
        ({"input": {}}, "must not be blank"),
        ({"questions": []}, "at least 1"),
        ({"temperature": 0}, "greater than 0"),
        ({"surprise": 1}, "Extra inputs"),
        (
            {"questions": [_CHOICE, _CHOICE]},
            "repeats another question",
        ),
    ],
)
def test_request_validation_rejects(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _request(**overrides)


def test_choice_options_must_be_distinct_and_bounded() -> None:
    adapter = TypeAdapter(DecisionRequest)

    def question(names: list[str]) -> dict[str, Any]:
        return {
            "input": "x",
            "questions": [
                {
                    "id": "q",
                    "type": "choice",
                    "question": "?",
                    "options": [{"name": name} for name in names],
                }
            ],
        }

    with pytest.raises(ValidationError, match="repeats another option"):
        adapter.validate_python(question(["Yes", " yes "]))
    with pytest.raises(ValidationError, match="nonempty"):
        adapter.validate_python(question(["a", " "]))
    with pytest.raises(ValidationError, match="control or line break"):
        adapter.validate_python(question(["a", "b\nc"]))
    with pytest.raises(ValidationError, match="at least 2"):
        adapter.validate_python(question(["a"]))
    with pytest.raises(ValidationError, match="at most 26"):
        adapter.validate_python(question([f"o{i}" for i in range(27)]))
    adapter.validate_python(question([f"o{i}" for i in range(26)]))


def test_score_levels_are_bounded_to_ten() -> None:
    def levels(count: int) -> dict[str, Any]:
        return {
            "id": "s",
            "type": "score",
            "question": "?",
            "levels": [f"l{i}" for i in range(count)],
        }

    _request(questions=[levels(10)])
    with pytest.raises(ValidationError, match="at most 10"):
        _request(questions=[levels(11)])
