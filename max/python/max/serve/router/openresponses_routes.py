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

"""OpenResponses API route handlers.

This module provides a clean implementation of the OpenResponses API standard
without inheriting technical debt from other API endpoints.

Spec: https://www.openresponses.org/reference
"""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from max.interfaces import OpenResponsesRequest
from max.interfaces.request.open_responses import (
    ResponseResource,
)
from max.serve.dependencies import create_request_parser

router = APIRouter(prefix="/v1")
logger = logging.getLogger("max.serve")

# Create a reusable dependency for parsing OpenResponses requests
ParseOpenResponsesRequest = Depends(create_request_parser(OpenResponsesRequest))


@router.post("/responses")
async def create_response(
    request: Request,
    open_responses_request: OpenResponsesRequest = ParseOpenResponsesRequest,
) -> JSONResponse:
    """Create a response using the OpenResponses API schema.

    This endpoint provides a clean implementation of the OpenResponses
    standard for generating responses from AI models.

    Args:
        request: The incoming FastAPI request containing OpenResponses data.
        open_responses_request: Parsed and validated OpenResponses request
            (automatically injected via dependency injection). Validation
            includes checking that streaming is not requested.

    Returns:
        A JSONResponse with the generated response data.

    Raises:
        HTTPException: If request parsing or validation fails, including
            if streaming is requested (not currently supported).
    """

    # Request is already parsed and validated via dependency injection
    # (including validation that streaming is not requested)
    logger.debug(
        "OpenResponses request parsed successfully - "
        "request_id=%s, model=%s, stream=%s",
        open_responses_request.request_id.value,
        open_responses_request.body.model,
        open_responses_request.body.stream,
    )

    # Generate response using the GeneralPipelineHandler from app state
    logger.debug("Starting response generation")

    # Get the first chunk from the handler (raises StopAsyncIteration if empty)
    generator = request.app.state.handler.next(open_responses_request)
    final_output = await anext(generator)
    logger.debug(
        "Received chunk - is_done=%s, status=%s",
        final_output.is_done,
        final_output.final_status,
    )

    # Continue consuming chunks until we get is_done=True
    if not final_output.is_done:
        async for chunk in generator:
            logger.debug(
                "Received chunk - is_done=%s, status=%s",
                chunk.is_done,
                chunk.final_status,
            )
            final_output = chunk
            if chunk.is_done:
                break

    # Convert GenerationOutput to ResponseResource format
    response = ResponseResource.from_generation_output(
        final_output, model=open_responses_request.body.model
    )

    logger.debug(
        "Returning response for request_id=%s",
        open_responses_request.request_id.value,
    )
    return JSONResponse(
        content=response.model_dump(mode="json", exclude_none=True),
        status_code=HTTPStatus.OK,
    )
