# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from sse_starlette.sse import EventSourceResponse

from max.serve.mocks.mock_api_requests import (
    openai_stream_response,
    openai_simple_response,
)

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.post("/chat/completions")
async def openai_chat_completions(request: Request) -> Response:
    # Ignoring the request, simply pulling out some sample input data:

    json_data = await request.json()
    model_name = json_data["model"]
    prompt = json_data["messages"]
    stream = False
    if "stream" in json_data:
        logger.info("Found stream entry in json")
        stream = True if json_data["stream"] == True else False
    logger.info(
        f"Processing {model_name} with prompt {prompt} and streaming = {stream}"
    )

    if stream:
        response = openai_stream_response(
            "This is a stream test!", model_name=model_name
        )

        async def json(response):
            for r in response:
                yield json.dumps(r)

        return EventSourceResponse(json(response))
    else:
        return JSONResponse(openai_simple_response("This is a test!"))
