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

"""Router tests for ``POST /v1/decisions`` against a fake scoring pipeline."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from max.pipelines.modeling.types import PipelineTask
from max.serve.router import _decisions_routes
from max.serve.schemas._decisions import MAX_QUESTIONS
from tests.serve.decisions_fakes import (
    MODEL,
    THINK_END,
    FakePipeline,
    FakeTokenizer,
    make_app,
    prefer_first,
)


def _pipeline(**tokenizer_kwargs: Any) -> FakePipeline:
    return FakePipeline(FakeTokenizer(**tokenizer_kwargs), prefer_first)


_CHOICE = {
    "id": "team",
    "type": "choice",
    "question": "Which team?",
    "options": [{"name": "billing"}, {"name": "technical"}, {"name": "other"}],
}
_YES_NO = {"id": "refund", "type": "yes_no", "question": "Wants a refund?"}


def _post(
    pipeline: FakePipeline, body: dict[str, Any], **app_kwargs: Any
) -> Any:
    with TestClient(
        make_app(pipeline, _decisions_routes.router, **app_kwargs)
    ) as client:
        return client.post("/v1/decisions", json=body)


def test_answers_every_question_with_probabilities_and_zero_completion() -> (
    None
):
    pipeline = _pipeline()
    response = _post(
        pipeline, {"input": "charged twice", "questions": [_CHOICE, _YES_NO]}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["object"] == "decisions"
    assert body["model"] == MODEL
    assert body["prompt_format_version"] == 1
    team = body["answers"]["team"]
    assert team["type"] == "choice"
    assert team["choice"] == "billing"
    assert list(team["probabilities"]) == ["billing", "technical", "other"]
    assert sum(team["probabilities"].values()) == pytest.approx(1.0)
    assert team["label_mass"] == pytest.approx(0.9)
    assert "prompt_token_ids" not in team
    refund = body["answers"]["refund"]
    assert set(refund["probabilities"]) == {"yes", "no"}
    assert "choice" not in refund and "score" not in refund

    usage = body["usage"]
    assert usage["completion_tokens"] == 0
    assert usage["prompt_tokens"] == usage["total_tokens"] > 0
    assert usage["prompt_tokens_details"]["cached_tokens"] == 4


def test_each_question_is_one_single_token_prefill_with_its_label_ids() -> None:
    pipeline = _pipeline()
    _post(pipeline, {"input": "x", "questions": [_CHOICE, _YES_NO]})

    assert len(pipeline.requests) == 2
    for request in pipeline.requests:
        assert isinstance(request.prompt, list)
    # A, B, C then yes, no: each label is one distinct token id.
    assert [len(labels) for labels in pipeline.labels] == [3, 2]
    assert all(len(set(labels)) == len(labels) for labels in pipeline.labels)
    assert pipeline.requests[0].request_id != pipeline.requests[1].request_id


def test_thinking_is_off_by_default_and_prompt_ends_after_the_closed_block() -> (
    None
):
    pipeline = _pipeline()
    _post(pipeline, {"input": "x", "questions": [_YES_NO]})
    assert pipeline.tokenizer.template_options == [
        {"enable_thinking": False, "thinking": False}
    ]
    assert pipeline.tokenizer.rendered_prompts[0].endswith(
        "<think>\n\n</think>\n\n"
    )
    # ... </think>, "\n", "\n": the answer slot follows the closed block.
    assert pipeline.prompt_ids[0][-3] == THINK_END


def test_temperature_scales_probabilities_but_not_label_mass() -> None:
    cold = _post(_pipeline(), {"input": "x", "questions": [_CHOICE]}).json()
    hot = _post(
        _pipeline(), {"input": "x", "questions": [_CHOICE], "temperature": 5}
    ).json()
    cold_team, hot_team = cold["answers"]["team"], hot["answers"]["team"]
    assert (
        hot_team["probabilities"]["billing"]
        < cold_team["probabilities"]["billing"]
    )
    assert hot_team["label_mass"] == pytest.approx(cold_team["label_mass"])


def test_return_prompt_token_ids_echoes_the_exact_scoring_inputs() -> None:
    pipeline = _pipeline()
    body = _post(
        pipeline,
        {"input": "x", "questions": [_CHOICE], "return_prompt_token_ids": True},
    ).json()
    team = body["answers"]["team"]
    assert team["prompt_token_ids"] == pipeline.prompt_ids[0]
    assert team["label_token_ids"] == pipeline.labels[0]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"input": " ", "questions": [_YES_NO]}, "must not be blank"),
        ({"input": "x", "questions": []}, "at least 1"),
        ({"input": 1, "questions": [_YES_NO]}, "input"),
        (
            {
                "input": "x",
                "questions": [
                    {**_YES_NO, "id": f"q{i}"} for i in range(MAX_QUESTIONS + 1)
                ],
            },
            "at most 64",
        ),
    ],
)
def test_schema_errors_are_422(body: dict[str, Any], message: str) -> None:
    pipeline = _pipeline()
    response = _post(pipeline, body)
    assert response.status_code == 422, response.text
    assert message in response.text
    assert pipeline.requests == []


def test_invalid_json_is_422() -> None:
    with TestClient(make_app(_pipeline(), _decisions_routes.router)) as client:
        response = client.post("/v1/decisions", content=b"{not json")
    assert response.status_code == 422


def test_a_request_at_the_question_limit_is_scored() -> None:
    questions = [{**_YES_NO, "id": f"q{i}"} for i in range(MAX_QUESTIONS)]
    response = _post(_pipeline(), {"input": "x", "questions": questions})
    assert response.status_code == 200, response.text
    assert len(response.json()["answers"]) == MAX_QUESTIONS


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {"input": "x", "questions": [_YES_NO], "prompt_format_version": 2},
            "prompt_format_version 2 is not served",
        ),
        (
            {
                "input": "x",
                "questions": [_YES_NO],
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "false or unset",
        ),
        ({"input": "x " * 400, "questions": [_YES_NO]}, "context length"),
        (
            {"input": "x", "questions": [_YES_NO], "model": "other"},
            "Unknown model",
        ),
    ],
)
def test_client_errors_are_400_with_a_specific_message(
    body: dict[str, Any], message: str
) -> None:
    pipeline = _pipeline()
    response = _post(pipeline, body)
    assert response.status_code == 400, response.text
    assert message in response.text
    assert pipeline.requests == []


def test_a_label_that_is_not_a_single_token_is_refused_with_the_question_id() -> (
    None
):
    pipeline = _pipeline(label_merges=True)
    response = _post(pipeline, {"input": "x", "questions": [_CHOICE]})
    assert response.status_code == 400
    assert "question 'team'" in response.text
    assert "not one distinct token" in response.text


def test_a_server_that_is_not_generating_text_is_refused() -> None:
    response = _post(
        _pipeline(),
        {"input": "x", "questions": [_YES_NO]},
        task=PipelineTask.EMBEDDINGS_GENERATION,
    )
    assert response.status_code == 400
    assert "text generation" in response.text


def test_a_server_with_speculative_decoding_is_refused() -> None:
    pipeline = _pipeline()
    response = _post(
        pipeline, {"input": "x", "questions": [_YES_NO]}, speculative=True
    )
    assert response.status_code == 400
    assert "speculative decoding" in response.text
    assert pipeline.requests == []


def test_a_thinking_template_that_leaves_the_block_open_is_refused() -> None:
    pipeline = _pipeline()
    pipeline.tokenizer.apply_chat_template = (  # type: ignore[method-assign]
        lambda messages, tools, **options: "<user>\nhi\n<assistant>\n<think>\n"
    )
    response = _post(pipeline, {"input": "x", "questions": [_YES_NO]})
    assert response.status_code == 400
    assert "reasoning block open" in response.text


def test_a_tokenizer_missing_a_scoring_capability_is_refused_by_name() -> None:
    pipeline = _pipeline(has_context_length=False)
    response = _post(pipeline, {"input": "x", "questions": [_YES_NO]})
    assert response.status_code == 400
    assert "requires a tokenizer with" in response.text


def test_a_tokenizer_without_reasoning_tokens_is_served() -> None:
    pipeline = _pipeline(has_reasoning=False)
    response = _post(pipeline, {"input": "x", "questions": [_YES_NO]})
    assert response.status_code == 200, response.text


def test_an_unexpected_value_error_is_a_server_fault() -> None:
    def broken(request: Any, labels: Any) -> list[float]:
        raise ValueError("scorer invariant violated")

    pipeline = FakePipeline(FakeTokenizer(), broken)
    response = _post(pipeline, {"input": "x", "questions": [_YES_NO]})
    assert response.status_code == 500
    assert "scorer invariant" not in response.text


def test_a_non_finite_score_is_a_server_fault() -> None:
    pipeline = FakePipeline(
        FakeTokenizer(), lambda request, labels: [float("nan")] * len(labels)
    )
    response = _post(pipeline, {"input": "x", "questions": [_YES_NO]})
    assert response.status_code == 500
