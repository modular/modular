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

"""Tests for the decider prompt layout and its use by ``/v1/decisions``."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from max.pipelines.modeling.types import TextGenerationRequest
from max.serve.router import _decider_prompt
from max.serve.router._chat_format import ChatFormat
from max.serve.router._decider_format import (
    DECIDER_CONFIG_FILE,
    DeciderFormat,
)
from max.serve.router._decider_prompt import (
    DeciderRow,
    annotate_indices,
    combine_rows,
    parse_decider_config,
    plan_rows,
    render_row,
    render_state,
    state_prefix,
)
from max.serve.router._decision_format import detect_decision_format
from max.serve.router._decisions_prompt import QuestionView
from max.serve.router._decisions_routes import router as decisions_router
from max.serve.router._systemone_routes import router as systemone_router
from tests.serve.decisions_fakes import (
    FakePipeline,
    FakeTokenizer,
    make_app,
)

_CONFIG = {
    "temperature": 1.145,
    "temperature_by_type": {"choice": 1.164, "noul": 1.624, "score": 1.124},
    "layout": "plain",
    "max_state_tokens": 32768,
    "isolated_levels": True,
}


def _choice(names: list[str], details: list[Any] | None = None) -> QuestionView:
    return QuestionView(
        "choice", "Which?", names, details or [None] * len(names)
    )


def test_choice_row_matches_the_reference_layout() -> None:
    rows = plan_rows(
        _choice(["billing", "tech"], ["charges", None]), isolated_levels=True
    )
    assert render_row(rows[0]) == (
        "\n\nQuestion: Which?\nOptions:\n(A) billing: charges\n(B) tech"
        "\nAnswer: ("
    )
    assert state_prefix("hi") == "Context:\nhi"


def test_yes_no_options_are_no_then_yes_and_answer_is_yes_first() -> None:
    view = QuestionView("yes_no", "Refund?", ["yes", "no"], ["wants it", None])
    (row,) = plan_rows(view, isolated_levels=True)
    assert row.options == ["no", "yes: wants it"]
    assert combine_rows(view, [[0.25, 0.75]], True) == [0.75, 0.25]


def test_yes_no_without_instructions_uses_the_descriptions() -> None:
    view = QuestionView("yes_no", None, ["yes", "no"], ["is urgent", None])
    (row,) = plan_rows(view, isolated_levels=True)
    assert row.question == _decider_prompt.NOUL_WITHOUT_INSTRUCTIONS
    bare = QuestionView("yes_no", None, ["yes", "no"], [None, None])
    with pytest.raises(ValueError, match="no instructions"):
        plan_rows(bare, isolated_levels=True)


def test_score_levels_are_isolated_without_their_numbers() -> None:
    view = QuestionView(
        "score", "How urgent?", ["0", "1", "2"], ["0: low", "mid", "high"]
    )
    rows = plan_rows(view, isolated_levels=True)
    assert [row.options for row in rows] == [["no", "yes"]] * 3
    assert rows[0].question == (
        "How urgent?\nProposed answer: low\nDoes the proposed answer fit?"
    )
    probabilities = combine_rows(
        view, [[0.9, 0.1], [0.5, 0.5], [0.6, 0.4]], isolated_levels=True
    )
    assert probabilities == pytest.approx([0.1, 0.5, 0.4])
    listed = plan_rows(view, isolated_levels=False)
    assert len(listed) == 1
    assert listed[0].options[0] == "0: 0: low"


def test_json_state_numbers_long_arrays_and_dumps_like_json() -> None:
    assert render_state("plain") == "plain"
    assert render_state({"a": [1, 2]}) == '{"a": [1, 2]}'
    long = list(range(8))
    assert annotate_indices(long)[3] == {"_index": 3, "value": 3}
    assert annotate_indices([{"x": 1}] * 8)[0] == {"_index": 0, "x": 1}
    assert render_state({"é": 1}) == '{"é": 1}'


def test_more_than_ten_options_are_refused() -> None:
    with pytest.raises(ValueError, match="at most 10 options"):
        plan_rows(_choice([str(i) for i in range(11)]), isolated_levels=True)


def test_decider_config_temperatures_by_type() -> None:
    settings = parse_decider_config(_CONFIG)
    assert settings.temperature_for("noul") == 1.624
    plain = parse_decider_config({"temperature": 1.03})
    assert plain.temperature_for("score") == 1.03
    assert plain.max_state_tokens == 32768


@pytest.mark.parametrize(
    "config",
    [
        {"layout": "chat"},
        {"temperature": 0},
        {"temperature_by_type": {"rank": 1.0}},
        {"temperature_by_type": {"choice": "hot"}},
    ],
)
def test_bad_decider_configs_are_refused(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_decider_config(config)


def _score_by_rows(request: TextGenerationRequest, labels: Any) -> list[float]:
    # Isolated rows (two labels) say "yes" at 0.8 on every level.
    probabilities = {2: [0.2, 0.8], 3: [0.5, 0.3, 0.2]}[len(labels)]
    return [math.log(p) for p in probabilities]


def _post(
    path: str,
    body: dict[str, Any],
    pipeline: FakePipeline | None = None,
    **app_kwargs: Any,
) -> Any:
    pipeline = pipeline or FakePipeline(FakeTokenizer(), _score_by_rows)
    router = decisions_router if path == "/v1/decisions" else systemone_router
    app = make_app(
        pipeline,
        router,
        decision_format=DeciderFormat(parse_decider_config(_CONFIG)),
        **app_kwargs,
    )
    with TestClient(app) as client:
        return client.post(path, json=body)


_CHOICE_QUESTION = {
    "id": "team",
    "type": "choice",
    "question": "Which team?",
    "options": [{"name": "billing"}, {"name": "tech"}, {"name": "other"}],
}


def test_decisions_prompt_is_token_ids_in_the_decider_layout() -> None:
    pipeline = FakePipeline(FakeTokenizer(), _score_by_rows)
    response = _post(
        "/v1/decisions",
        {"input": "charged twice", "questions": [_CHOICE_QUESTION]},
        pipeline,
    )
    assert response.status_code == 200, response.text
    tokenizer = FakeTokenizer()
    expected = asyncio.run(tokenizer.encode(state_prefix("charged twice"))) + (
        asyncio.run(
            tokenizer.encode(
                render_row(
                    DeciderRow("Which team?", ["billing", "tech", "other"])
                )
            )
        )
    )
    assert pipeline.requests[0].prompt == expected
    assert pipeline.prompt_ids[0] == expected
    letter_ids = [asyncio.run(tokenizer.encode(c))[0] for c in "ABC"]
    assert pipeline.labels[0] == letter_ids
    body = response.json()
    assert body["prompt_format_version"] == 2
    answer = body["answers"]["team"]
    # Probabilities use the choice temperature of the decider config.
    expected_p = [p ** (1 / 1.164) for p in (0.5, 0.3, 0.2)]
    total = sum(expected_p)
    assert answer["probabilities"]["billing"] == pytest.approx(
        expected_p[0] / total
    )


def test_score_question_is_scored_as_one_row_per_level() -> None:
    pipeline = FakePipeline(FakeTokenizer(), _score_by_rows)
    question = {
        "id": "urgency",
        "type": "score",
        "question": "How urgent?",
        "levels": ["low", "mid", "high"],
    }
    response = _post(
        "/v1/decisions", {"input": "x", "questions": [question]}, pipeline
    )
    assert response.status_code == 200, response.text
    assert len(pipeline.requests) == 3
    answer = response.json()["answers"]["urgency"]
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)
    assert answer["probabilities"]["0"] == pytest.approx(1 / 3)
    assert answer["score"] == pytest.approx(1.0)
    assert answer["label_mass"] == pytest.approx(1.0)


def test_request_temperature_replaces_the_config_temperature() -> None:
    body = {
        "input": "x",
        "temperature": 1.0,
        "questions": [_CHOICE_QUESTION],
    }
    answer = _post("/v1/decisions", body).json()["answers"]["team"]
    assert answer["probabilities"]["billing"] == pytest.approx(0.5)


def test_prompt_format_version_is_the_decider_one() -> None:
    body = {
        "input": "x",
        "prompt_format_version": 1,
        "questions": [_CHOICE_QUESTION],
    }
    assert _post("/v1/decisions", body).status_code == 400
    body["prompt_format_version"] = 2
    assert _post("/v1/decisions", body).status_code == 200


def test_chat_template_kwargs_and_prompt_ids_for_scores_are_refused() -> None:
    body: dict[str, Any] = {
        "input": "x",
        "chat_template_kwargs": {"enable_thinking": False},
        "questions": [_CHOICE_QUESTION],
    }
    assert _post("/v1/decisions", body).status_code == 400
    score = {
        "id": "s",
        "type": "score",
        "question": "q",
        "levels": ["a", "b", "c"],
    }
    body = {
        "input": "x",
        "return_prompt_token_ids": True,
        "questions": [score],
    }
    assert _post("/v1/decisions", body).status_code == 400


def test_too_many_options_name_the_question() -> None:
    many = {
        **_CHOICE_QUESTION,
        "options": [{"name": f"o{i}"} for i in range(11)],
    }
    response = _post("/v1/decisions", {"input": "x", "questions": [many]})
    assert response.status_code == 400
    assert "question 'team'" in response.text


def test_a_checkpoint_with_a_decider_config_is_served_as_a_decider(
    tmp_path: Path,
) -> None:
    (tmp_path / DECIDER_CONFIG_FILE).write_text(json.dumps(_CONFIG))
    decision_format = detect_decision_format(str(tmp_path))
    assert isinstance(decision_format, DeciderFormat)
    assert decision_format.version == 2


def test_a_checkpoint_without_one_is_served_with_its_chat_template(
    tmp_path: Path,
) -> None:
    decision_format = detect_decision_format(str(tmp_path))
    assert isinstance(decision_format, ChatFormat)
    assert decision_format.version == 1


def test_an_invalid_decider_config_fails_detection(tmp_path: Path) -> None:
    (tmp_path / DECIDER_CONFIG_FILE).write_text(json.dumps({"layout": "chat"}))
    with pytest.raises(ValueError):
        detect_decision_format(str(tmp_path))


def test_systemone_answers_in_the_decider_format() -> None:
    body = {
        "state": "charged twice",
        "model": "test-decider",
        "questions": {
            "refund": {"type": "noul", "instructions": "Refund?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["low", "mid", "high"],
            },
        },
    }
    pipeline = FakePipeline(FakeTokenizer(), _score_by_rows)
    response = _post("/v1/systemone", body, pipeline)
    assert response.status_code == 200, response.text
    answers = response.json()["answers"]
    # Noul uses its own temperature on the [no, yes] pair.
    no, yes = (0.2 ** (1 / 1.624), 0.8 ** (1 / 1.624))
    assert answers["refund"]["noul"] == pytest.approx(yes / (no + yes))
    assert answers["urgency"]["score"] == pytest.approx(1.0)
    assert len(pipeline.requests) == 4  # 1 noul row + 3 level rows
