# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.schemas.openai import (
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartRefusal,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Choice1,
    Choice3,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Logprobs2,
)
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


@router.post("/chat/completions")
async def openai_chat_completions(
    request: Request,
    pipeline: Annotated[TokenGeneratorPipeline, Depends(token_pipeline)],
) -> Response:
    json_data = await request.json()
    completion_request = CreateChatCompletionRequest.model_validate(json_data)
    logger.info(
        "Processing"
        f" {'streaming' if completion_request.stream else ''} {completion_request.model} request."
    )

    def message_text(content):
        return " ".join(
            [
                part.root.text
                for part in content
                if not isinstance(
                    part.root,
                    (
                        ChatCompletionRequestMessageContentPartRefusal,
                        ChatCompletionRequestMessageContentPartImage,
                    ),
                )
            ]
        )

    prompt = ""
    for message in completion_request.messages:
        content = message.root.content
        if isinstance(content, str):
            prompt += content
        elif not content:
            continue
        else:
            prompt += message_text(content)

    requests = {str(uuid4()): await pipeline.model.new_context(prompt)}

    @dataclass
    class ResponseGenerator:
        counters: defaultdict[str, int] = field(
            default_factory=lambda: defaultdict(int)
        )

        async def generate(self):
            async for rid, token in pipeline.next_token(requests):
                index = self.counters[rid]
                self.counters[rid] = index + 1
                if not token:
                    del self.counters[rid]
                choices = [
                    Choice3(
                        index=index,
                        delta=ChatCompletionStreamResponseDelta(
                            content=token,
                            function_call=None,
                            role="assistant",
                            refusal=None,
                        ),
                        logprobs=None,
                        finish_reason="stop",
                    )
                ]
                response = CreateChatCompletionStreamResponse(
                    id=rid,
                    choices=choices,
                    created=int(datetime.now().timestamp()),
                    model="",
                    object="chat.completion.chunk",
                    system_fingerprint=None,
                    usage=None,
                    service_tier=None,
                )
                yield response.model_dump_json()

    # TODO: Add request pool abstraction to mediate submitting requests
    #       and polling from the pipeline worker tasks.

    if completion_request.stream:
        gen = ResponseGenerator()
        return EventSourceResponse(gen.generate())

    message = " ".join(
        [token async for _, token in pipeline.next_token(requests) if token]
    )
    choices = [
        Choice1(
            index=0,
            message=ChatCompletionResponseMessage(
                content=message,
                role="assistant",
                function_call=None,
                refusal="",
            ),
            finish_reason="stop",
            logprobs=Logprobs2(content=[], refusal=[]),
        )
    ]
    response = CreateChatCompletionResponse(
        id="0",
        choices=choices,
        created=int(datetime.now().timestamp()),
        model="",
        object="chat.completion",
        system_fingerprint=None,
        service_tier=None,
    )
    return JSONResponse(response.model_dump_json())
