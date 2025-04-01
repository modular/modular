# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
from typing import Union

from fastapi import APIRouter, Request
from fastapi.responses import Response
from max.serve.router.openai_routes import openai_create_chat_completion
from max.serve.schemas.openai import (  # type: ignore
    CreateChatCompletionResponse,
)
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("max.serve")

router = APIRouter()


@router.get("/ping")
async def ping() -> Response:
    """Health check."""
    return Response(status_code=200)


@router.post("/invocations", response_model=None)
async def invocations(
    request: Request,
) -> Union[CreateChatCompletionResponse, EventSourceResponse]:
    """proxy to /v1/chat/completions"""
    return await openai_create_chat_completion(request)
