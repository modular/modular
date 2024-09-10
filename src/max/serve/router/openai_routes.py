# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import logging
from datetime import datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from max.serve.pipelines.deps import token_pipeline
from max.serve.pipelines.llm import TokenGeneratorPipeline
from max.serve.schemas.openai import (
    ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartText,
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


def openai_request_message_to_text(message: ChatCompletionRequestMessage):
    """
    The content property of each ChatCompletionRequestMessages can be a
    string or be a list of partial messages.
    """
    if isinstance(message.root.content, str):
        return message.root.content
    if message.root.content is None:
        return ""
    return "".join(
        [
            part.root.text
            for part in message.root.content
            if isinstance(
                part.root,
                ChatCompletionRequestMessageContentPartText,
            )
        ]
    )


class OpenAIResponseGenerator:
    def __init__(self, token_pipeline: TokenGeneratorPipeline):
        self.token_pipeline = token_pipeline

    async def generate(self, request_id: str, prompt: str):
        response_idx = 0
        async for token in self.token_pipeline.next_token(request_id, prompt):
            logger.debug(
                "Streaming: %s, [%s], TOKEN: %d, %s",
                request_id,
                prompt[:48],
                response_idx,
                token,
            )
            if token is None:
                break
            # We support N = 1 at the moment and will generate a single choice.
            # The choice index is set to 0.
            # https://platform.openai.com/docs/api-reference/chat/object
            choices = [
                Choice3(
                    index=0,
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
            # Each chunk is expected to have the same id
            # https://platform.openai.com/docs/api-reference/chat/streaming
            response = CreateChatCompletionStreamResponse(
                id=request_id,
                choices=choices,
                created=int(datetime.now().timestamp()),
                model="",
                object="chat.completion.chunk",
                system_fingerprint=None,
                usage=None,
                service_tier=None,
            )
            response_idx += 1
            yield response.model_dump_json()

        logger.debug(
            "Completed: %s, [%s], %d tokens",
            request_id,
            prompt[:48],
            response_idx,
        )
        yield "[DONE]"


@router.post("/chat/completions")
async def openai_chat_completions(
    request: Request,
    pipeline: Annotated[TokenGeneratorPipeline, Depends(token_pipeline)],
) -> Response:
    request_id = str(uuid4())
    request_json = await request.json()
    completion_request = CreateChatCompletionRequest.model_validate(
        request_json
    )

    logger.info(
        f"Processing {request_id} -"
        f" {'streaming' if completion_request.stream else ''} {completion_request.model} request."
    )

    request_prompt = ""
    if pipeline.tokenizer is not None:
        messages = [
            {
                "role": message.root.role,
                "content": openai_request_message_to_text(message),
            }
            for message in completion_request.messages
        ]
        request_prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        request_prompt = " ".join(
            [
                message.root.content
                for message in completion_request.messages
                if isinstance(message.root.content, str)
            ]
        )

    if completion_request.stream:
        gen = OpenAIResponseGenerator(pipeline)
        return EventSourceResponse(gen.generate(request_id, request_prompt))

    completed_tokens = await pipeline.all_tokens(request_id, request_prompt)

    response_message = "".join(completed_tokens)
    response_choices = [
        Choice1(
            index=0,
            message=ChatCompletionResponseMessage(
                content=response_message,
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
        choices=response_choices,
        created=int(datetime.now().timestamp()),
        model="",
        object="chat.completion",
        system_fingerprint=None,
        service_tier=None,
    )
    return JSONResponse(response.model_dump_json())
