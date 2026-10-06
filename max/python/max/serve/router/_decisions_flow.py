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

"""The request flow shared by ``/v1/decisions`` and ``/v1/systemone``.

Both routes parse their own schema, then render questions, score them, and
shape the answers. This module owns the two steps in the middle and the
mapping of their failures to HTTP errors, so the routes cannot drift apart: a
:class:`DecisionRequestError` or :class:`InputError` is a 400, and anything
else a logged 500.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request
from max.pipelines.context.exceptions import InputError
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.router._decision_format import DecisionFormat
from max.serve.router._decisions_prompt import QuestionView
from max.serve.router._decisions_scoring import (
    DecisionRequestError,
    EncodedQuestion,
    require_scoring,
)
from max.serve.router.openai_routes import get_pipeline
from max.serve.schemas._decisions import DecisionText
from max.serve.worker_interface import RequestQueueFull

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class RenderedRequest:
    """A request whose questions are rendered and ready to score."""

    pipeline: TokenGeneratorPipeline
    decision_format: DecisionFormat
    encoded: list[EncodedQuestion]


@asynccontextmanager
async def decision_errors(request: Request, route: str) -> AsyncIterator[None]:
    """Maps failures inside the block to HTTP errors.

    Args:
        request: The incoming request.
        route: The route path, for logging.

    Raises:
        HTTPException: With status 400 for a request this server or model
            cannot answer, and 500 for any other failure.
        RequestQueueFull: If the worker queue rejected admission; the central
            handler maps it to HTTP 429.
    """
    try:
        yield
    except (HTTPException, RequestQueueFull):
        raise
    except (DecisionRequestError, InputError) as e:
        logger.warning(
            "Invalid request %s to %s: %s", request.state.request_id, route, e
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(
            "Exception during request %s to %s",
            request.state.request_id,
            route,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error."
        ) from e


async def render_request(
    request: Request,
    route: str,
    model: str,
    state: DecisionText,
    questions: Sequence[tuple[str, QuestionView]],
    chat_template_kwargs: dict[str, Any],
    *,
    prompt_format_version: int | None = None,
) -> RenderedRequest:
    """Resolves the pipeline and renders every question to scoring rows.

    Args:
        request: The incoming request, for the app state.
        route: The route path, named in the error for a server without scoring.
        model: The requested model name, empty for the served default.
        state: The text or JSON value the questions are about.
        questions: Each question's id and view.
        chat_template_kwargs: Caller overrides of the chat template options.
        prompt_format_version: The prompt wording version the caller pinned.

    Raises:
        HTTPException: With status 400 if the request cannot be answered by
            this server or model.
    """
    async with decision_errors(request, route):
        try:
            pipeline = get_pipeline(request, model)
        except ValueError as e:
            raise DecisionRequestError(str(e)) from e
        tokenizer = require_scoring(request, pipeline, route)
        decision_format: DecisionFormat = request.app.state.decision_format
        if (
            prompt_format_version is not None
            and prompt_format_version != decision_format.version
        ):
            raise DecisionRequestError(
                f"prompt_format_version {prompt_format_version} is not "
                f"served, this server uses version {decision_format.version}"
            )
        encoded = await decision_format.encode(
            tokenizer, state, questions, chat_template_kwargs
        )
        return RenderedRequest(pipeline, decision_format, encoded)
