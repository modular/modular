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

"""``POST /v1/decisions``: typed decisions without generating text.

Scoring is shared with ``/v1/systemone`` (see
:mod:`max.serve.router._decisions_scoring`); the response reports zero
completion tokens.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from max.serve.pipelines._label_scoring import LabelScoringOutput
from max.serve.router._decisions_flow import decision_errors, render_request
from max.serve.router._decisions_prompt import build_answer, question_view
from max.serve.router._decisions_scoring import (
    DecisionRequestError,
    EncodedQuestion,
    score_all,
    usage_totals,
)
from max.serve.schemas._decisions import DecisionRequest, DecisionResponse
from max.serve.schemas.openai import CompletionUsage, PromptTokensDetails
from pydantic import ValidationError

router = APIRouter(prefix="/v1")
_ROUTE = "/v1/decisions"
logger = logging.getLogger("max.serve")


@router.post("/decisions", response_model=None)
async def create_decisions(request: Request) -> JSONResponse:
    """Answers typed questions with a probability for every option."""
    request_id = request.state.request_id

    # Schema errors are 422 and refusals 400, matching /v1/systemone.
    try:
        decision_request = DecisionRequest.model_validate_json(
            await request.body()
        )
    except ValidationError as e:
        logger.warning("Validation error in request %s: %s", request_id, e)
        raise HTTPException(status_code=422, detail=str(e)) from e

    rendered = await render_request(
        request,
        _ROUTE,
        decision_request.model,
        decision_request.input,
        [(q.id, question_view(q)) for q in decision_request.questions],
        decision_request.chat_template_kwargs,
        prompt_format_version=decision_request.prompt_format_version,
    )
    model_name = decision_request.model or rendered.pipeline.model_name

    async with decision_errors(request, _ROUTE):
        if decision_request.return_prompt_token_ids:
            _require_single_row(rendered.encoded)
        results = await score_all(
            request, rendered.pipeline, model_name, rendered.encoded
        )
        answers = {}
        for question, item, rows in zip(
            decision_request.questions, rendered.encoded, results, strict=True
        ):
            scores = rendered.decision_format.scores(
                item, rows, decision_request.temperature
            )
            answers[question.id] = build_answer(
                item.view,
                scores,
                prompt_token_ids=item.rows[0].prompt_ids
                if decision_request.return_prompt_token_ids
                else None,
                label_token_ids=item.rows[0].label_ids
                if decision_request.return_prompt_token_ids
                else None,
            )
        response = DecisionResponse(
            model=model_name,
            prompt_format_version=rendered.decision_format.version,
            answers=answers,
            usage=_usage(results),
        )
    return JSONResponse(content=response.model_dump(exclude_none=True))


def _require_single_row(encoded: Sequence[EncodedQuestion]) -> None:
    """Refuses echoing token ids for a question scored as several prompts."""
    if any(len(item.rows) > 1 for item in encoded):
        raise DecisionRequestError(
            "return_prompt_token_ids is not available for questions that are "
            "scored as several prompts (score questions with decider prompts)"
        )


def _usage(results: Sequence[Sequence[LabelScoringOutput]]) -> CompletionUsage:
    """Token accounting of the scoring requests; nothing is completed."""
    prompt_tokens, cached_tokens = usage_totals(results)
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        total_tokens=prompt_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens)
        if cached_tokens is not None
        else None,
    )
