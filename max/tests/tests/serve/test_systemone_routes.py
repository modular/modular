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

"""Router tests for ``POST /v1/systemone`` against a fake scoring pipeline."""

from __future__ import annotations

import math
from typing import Any

import pytest
from fastapi.testclient import TestClient
from max.pipelines.modeling.types import PipelineTask
from max.serve.router import _decisions_routes, _systemone_routes
from max.serve.schemas._decisions import (
    MAX_CHOICE_OPTIONS,
    MAX_QUESTIONS,
    MAX_SCORE_LEVELS,
)
from tests.serve.decisions_fakes import (
    MODEL,
    FakePipeline,
    FakeTokenizer,
    make_app,
    prefer_first,
)

_NOUL = {"type": "noul", "instructions": "Wants a refund?"}
_CHOICE = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "charges", "technical": None, "other": None},
}
_SCORE = {
    "type": "score",
    "instructions": "How urgent?",
    "criteria": ["no rush", "soon", "today", "now"],
}


def _post(
    body: dict[str, Any],
    pipeline: FakePipeline | None = None,
    **app_kwargs: Any,
) -> Any:
    pipeline = pipeline or FakePipeline(FakeTokenizer(), prefer_first)
    with TestClient(
        make_app(pipeline, _systemone_routes.router, **app_kwargs)
    ) as client:
        return client.post("/v1/systemone", json=body)


def _request(**questions: dict[str, Any]) -> dict[str, Any]:
    return {"state": "charged twice", "model": MODEL, "questions": questions}


def test_answers_noul_choice_and_score() -> None:
    response = _post(_request(n=_NOUL, c=_CHOICE, s=_SCORE))
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["model"] == MODEL
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["input_tokens"] > 0

    noul = body["answers"]["n"]
    assert noul["type"] == "noul"
    assert noul["noul"] == pytest.approx(0.6 / 0.8)
    assert noul["x_label_mass"] == pytest.approx(0.8)

    choice = body["answers"]["c"]
    assert choice["choice"] == "billing"
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0)
    top = max(choice["probabilities"].values())
    assert choice["confidence"] == pytest.approx((3 * top - 1) / 2)

    score = body["answers"]["s"]
    assert score["type"] == "score"
    assert score["legend"] == {
        "0": "no rush",
        "1": "soon",
        "2": "today",
        "3": "now",
    }
    expected = sum(
        level * p for level, p in enumerate(score["probabilities"].values())
    )
    assert score["score"] == pytest.approx(expected)
    assert 0.0 <= score["confidence"] <= 1.0


def test_scores_with_the_same_prompts_as_decisions() -> None:
    systemone = FakePipeline(FakeTokenizer(), prefer_first)
    _post(_request(c=_CHOICE, n=_NOUL), systemone)

    decisions = FakePipeline(FakeTokenizer(), prefer_first)
    with TestClient(make_app(decisions, _decisions_routes.router)) as client:
        client.post(
            "/v1/decisions",
            json={
                "input": "charged twice",
                "questions": [
                    {
                        "id": "c",
                        "type": "choice",
                        "question": "Which team?",
                        "options": [
                            {"name": "billing", "description": "charges"},
                            {"name": "technical"},
                            {"name": "other"},
                        ],
                    },
                    {
                        "id": "n",
                        "type": "yes_no",
                        "question": "Wants a refund?",
                    },
                ],
            },
        )
    assert [r.prompt for r in systemone.requests] == [
        r.prompt for r in decisions.requests
    ]


def test_single_option_choice_is_certain() -> None:
    only = {"type": "choice", "instructions": "?", "criteria": {"only": None}}
    answer = _post(_request(q=only)).json()["answers"]["q"]
    assert answer["choice"] == "only"
    assert answer["confidence"] == 1.0


def test_choice_without_instructions_has_no_question_line() -> None:
    pipeline = FakePipeline(FakeTokenizer(), prefer_first)
    bare = {"type": "choice", "criteria": {"a": None, "b": None}}
    assert _post(_request(q=bare), pipeline).status_code == 200
    assert "Question:" not in pipeline.tokenizer.rendered_prompts[0]


def test_ignores_unknown_top_level_fields() -> None:
    body = {**_request(n=_NOUL), "metadata": {"anything": 1}}
    assert _post(body).status_code == 200


def test_response_names_the_served_model() -> None:
    assert _post(_request(n=_NOUL)).json()["model"] == MODEL


def test_unknown_model_is_refused() -> None:
    body = {**_request(n=_NOUL), "model": "not-served"}
    assert _post(body).status_code == 400


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("model"),
        lambda b: b.update(questions={}),
        lambda b: b.update(temperature=0.5),
        lambda b: b.update(prompt_format_version=1),
        lambda b: b.update(return_prompt_token_ids=True),
        lambda b: b["questions"].update(x={"type": "rank"}),
        lambda b: b["questions"].update(x={"type": "noul"}),
        lambda b: b["questions"].update(
            x={**_CHOICE, "criteria": {"a": None, "A ": None}}
        ),
        lambda b: b["questions"].update(x={**_CHOICE, "extra": 1}),
        lambda b: b["questions"].update(x={**_SCORE, "criteria": []}),
    ],
)
def test_schema_errors_are_422(mutate: Any) -> None:
    body = _request(n=_NOUL)
    mutate(body)
    response = _post(body)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("state", ["", "  ", {}, []])
def test_empty_state_is_answered(state: Any) -> None:
    """Callers may put everything in the question instructions."""
    body = {**_request(n=_NOUL), "state": state}
    assert _post(body).status_code == 200


def test_capacity_limits_are_named_for_clients() -> None:
    wide = {
        **_CHOICE,
        "criteria": {
            f"option {i}": None for i in range(MAX_CHOICE_OPTIONS + 1)
        },
    }
    response = _post(_request(q=wide))
    assert response.status_code == 422
    assert "options per choice" in response.text

    tall = {
        **_SCORE,
        "criteria": [f"level {i}" for i in range(MAX_SCORE_LEVELS + 1)],
    }
    response = _post(_request(q=tall))
    assert response.status_code == 422
    assert "a score takes 2 to 10 levels" in response.text


def test_the_question_count_is_bounded() -> None:
    at_limit = {f"q{i}": _NOUL for i in range(MAX_QUESTIONS)}
    assert _post(_request(**at_limit)).status_code == 200

    over = {f"q{i}": _NOUL for i in range(MAX_QUESTIONS + 1)}
    response = _post(_request(**over))
    assert response.status_code == 422
    assert "at most 64" in response.text


def test_a_server_that_is_not_generating_text_is_refused() -> None:
    response = _post(_request(n=_NOUL), task=PipelineTask.EMBEDDINGS_GENERATION)
    assert response.status_code == 400
    assert "text generation" in response.text


def test_thinking_on_is_refused() -> None:
    body = {
        **_request(n=_NOUL),
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert _post(body).status_code == 400


def test_prompt_failure_names_the_question() -> None:
    pipeline = FakePipeline(FakeTokenizer(label_merges=True), prefer_first)
    response = _post(_request(c=_CHOICE), pipeline)
    assert response.status_code == 400
    assert "question 'c'" in response.text


def test_invalid_json_is_422() -> None:
    with TestClient(
        make_app(
            FakePipeline(FakeTokenizer(), prefer_first),
            _systemone_routes.router,
        )
    ) as client:
        response = client.post("/v1/systemone", content=b"{not json")
    assert response.status_code == 422


def test_confidence_bounds() -> None:
    uniform = [1.0 / 4] * 4
    assert _systemone_routes._score_confidence(uniform) == pytest.approx(
        0.0, abs=0.35
    )
    peaked = [0.0, 0.0, 1.0, 0.0]
    assert _systemone_routes._score_confidence(peaked) == pytest.approx(1.0)
    assert _systemone_routes._choice_confidence([1.0, 0.0, 0.0]) == 1.0
    assert _systemone_routes._choice_confidence([1 / 3] * 3) == pytest.approx(
        0.0
    )
    assert math.isfinite(_systemone_routes._score_confidence([0.5, 0.5]))
