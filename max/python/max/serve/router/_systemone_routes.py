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

"""``POST /v1/systemone``: System One compatible decisions.

A thin adapter over the ``/v1/decisions`` scoring path: the questions are
rendered with the same prompt format, scored with one prefill step each, and
returned in the System One answer shape.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from max.serve.router._decisions_flow import decision_errors, render_request
from max.serve.router._decisions_prompt import QuestionScores, QuestionView
from max.serve.router._decisions_scoring import score_all, usage_totals
from max.serve.schemas._systemone import (
    SystemOneChoiceAnswer,
    SystemOneChoiceQuestion,
    SystemOneNoulAnswer,
    SystemOneNoulQuestion,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneScoreAnswer,
    SystemOneUsage,
)
from pydantic import ValidationError

router = APIRouter(prefix="/v1")
_ROUTE = "/v1/systemone"
logger = logging.getLogger("max.serve")


def _system_one_view(question: SystemOneQuestion) -> QuestionView:
    """Normalizes a System One question into candidate names and details."""
    if isinstance(question, SystemOneChoiceQuestion):
        return QuestionView(
            kind="choice",
            question=question.instructions,
            names=list(question.criteria),
            details=list(question.criteria.values()),
        )
    if isinstance(question, SystemOneNoulQuestion):
        criteria = question.criteria
        return QuestionView(
            kind="yes_no",
            question=question.instructions,
            names=["yes", "no"],
            details=[
                criteria.true if criteria else None,
                criteria.false if criteria else None,
            ],
        )
    return QuestionView(
        kind="score",
        question=question.instructions,
        names=[str(level) for level in range(len(question.criteria))],
        details=list(question.criteria),
    )


def _normalized(probabilities: Sequence[float]) -> list[float]:
    total = math.fsum(probabilities)
    if total <= 0:
        return [1.0 / len(probabilities)] * len(probabilities)
    return [p / total for p in probabilities]


def _choice_confidence(probabilities: Sequence[float]) -> float:
    """How far the top option stands above a uniform guess, from 0 to 1."""
    count = len(probabilities)
    if count == 1:
        return 1.0
    return min(1.0, max(0.0, (count * max(probabilities) - 1) / (count - 1)))


def _score_confidence(probabilities: list[float]) -> float:
    """One minus the spread around the top level relative to a uniform one."""
    count = len(probabilities)
    if count == 1:
        return 1.0
    top = probabilities.index(max(probabilities))
    spread = math.fsum(
        p * abs(level - top) for level, p in enumerate(probabilities)
    )
    uniform_spread = (
        math.fsum(abs(level - (count - 1) / 2) for level in range(count))
        / count
    )
    return max(0.0, 1 - spread / uniform_spread)


def _build_system_one_answer(
    view: QuestionView, scores: QuestionScores
) -> SystemOneNoulAnswer | SystemOneChoiceAnswer | SystemOneScoreAnswer:
    """Builds the System One answer for one scored question."""
    probabilities = scores.probabilities
    mass = scores.label_mass
    if view.kind == "yes_no":
        # The view orders yes/no as ["yes", "no"].
        return SystemOneNoulAnswer(noul=probabilities[0], x_label_mass=mass)
    by_name = dict(zip(view.names, probabilities, strict=True))
    if view.kind == "choice":
        assert scores.choice is not None
        return SystemOneChoiceAnswer(
            choice=scores.choice,
            confidence=_choice_confidence(_normalized(probabilities)),
            probabilities=by_name,
            x_label_mass=mass,
        )
    assert scores.score is not None
    return SystemOneScoreAnswer(
        score=scores.score,
        confidence=_score_confidence(_normalized(probabilities)),
        legend=dict(zip(view.names, view.details, strict=True)),
        probabilities=by_name,
        x_label_mass=mass,
    )


@router.post("/systemone", response_model=None)
async def create_systemone(request: Request) -> JSONResponse:
    """Answers System One questions with the ``/v1/decisions`` scorer."""
    request_id = request.state.request_id

    # Schema errors are 422 and refusals 400, as the System One API defines.
    try:
        system_one_request = SystemOneRequest.model_validate_json(
            await request.body()
        )
    except ValidationError as e:
        logger.warning("Validation error in request %s: %s", request_id, e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    rendered = await render_request(
        request,
        _ROUTE,
        system_one_request.model,
        system_one_request.state,
        [
            (question_id, _system_one_view(question))
            for question_id, question in system_one_request.questions.items()
        ],
        system_one_request.chat_template_kwargs,
    )
    # The served model answered, whatever name the request used.
    model_name = rendered.pipeline.model_name

    async with decision_errors(request, _ROUTE):
        results = await score_all(
            request, rendered.pipeline, model_name, rendered.encoded
        )
        prompt_tokens, _ = usage_totals(results)
        response = SystemOneResponse(
            model=model_name,
            answers={
                item.question_id: _build_system_one_answer(
                    item.view,
                    rendered.decision_format.scores(item, rows, None),
                )
                for item, rows in zip(rendered.encoded, results, strict=True)
            },
            usage=SystemOneUsage(input_tokens=prompt_tokens),
        )
    return JSONResponse(content=response.model_dump())
