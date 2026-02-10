# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from max.interfaces import OpenResponsesRequest
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
    standard. Currently stubbed for development - streaming and full
    response generation will be implemented in future iterations.

    Args:
        request: The incoming FastAPI request containing OpenResponses data.
        open_responses_request: Parsed and validated OpenResponses request
            (automatically injected via dependency injection).

    Returns:
        A JSONResponse with the stubbed response data.

    Raises:
        HTTPException: If request parsing or validation fails (handled by
            the dependency).
    """

    # Request is already parsed and validated via dependency injection
    logger.debug(
        "OpenResponses request parsed successfully - "
        "request_id=%s, model=%s, stream=%s",
        open_responses_request.request_id.value,
        open_responses_request.body.model,
        open_responses_request.body.stream,
    )

    # TODO: Implement actual response generation
    # This is where we will:
    # 1. Get the pipeline from request.app.state.pipeline
    # 2. Convert OpenResponses request to internal format
    # 3. Generate response (with streaming support if requested)
    # 4. Convert internal response back to OpenResponses format

    # Stubbed response for now
    stubbed_response: dict[str, Any] = {
        "id": f"resp_{open_responses_request.request_id.value}",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "model": open_responses_request.body.model,
        "output": [
            {
                "id": "msg_stub",
                "role": "assistant",
                "content": "This is a stubbed response. Full implementation coming soon.",
                "status": "completed",
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }

    logger.debug("Returning stubbed response: %s", stubbed_response)
    return JSONResponse(content=stubbed_response, status_code=HTTPStatus.OK)
