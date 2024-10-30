# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import asyncio
import json
import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import AsyncGenerator, List, Literal, Optional, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from max.pipelines import PreTrainedTokenGeneratorTokenizer
from max.serve.pipelines.deps import token_pipeline_state
from max.serve.pipelines.llm import (
    TokenGeneratorPipeline,
    TokenGeneratorRequest,
)
from max.serve.schemas.openai import (
    ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Choice,
    Choice1,
    Choice3,
    CompletionUsage,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    Logprobs,
    Logprobs2,
)
from pydantic import BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


class OpenAIResponseGenerator(ABC):
    def __init__(
        self,
        pipeline: TokenGeneratorPipeline,
        request_limiter: Optional[asyncio.BoundedSemaphore] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipeline = pipeline
        self.request_limiter = request_limiter

    async def create_request(
        self,
        **kwargs,
    ):
        if self.request_limiter:
            await self.request_limiter.acquire()
        return await self.pipeline.create_request(**kwargs)

    @abstractmethod
    async def stream(
        self, request: TokenGeneratorRequest
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def complete(self, request: TokenGeneratorRequest) -> str:
        pass


class OpenAIChatResponseGenerator(OpenAIResponseGenerator):
    async def stream(self, request: TokenGeneratorRequest):
        self.logger.debug(
            "Streaming: Start: %s",
            request,
        )
        try:
            response_idx = 0
            async for token in self.pipeline.next_token(request):
                self.logger.debug(
                    "Streaming: %s, TOKEN: %d, %s",
                    request.id,
                    response_idx,
                    token,
                )
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
                    id=request.id,
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
                "Streaming: Done: %s, %d tokens",
                request,
                response_idx,
            )
            yield "[DONE]"
        except ValueError as e:
            logger.error(traceback.format_exc())
            yield json.dumps({"result": "error", "message": str(e)})
        finally:
            if self.request_limiter:
                self.request_limiter.release()

    async def complete(
        self, request: TokenGeneratorRequest
    ) -> CreateChatCompletionResponse:
        completed_tokens = await self.pipeline.all_tokens(request)

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
            id=request.id,
            choices=response_choices,
            created=int(datetime.now().timestamp()),
            model="",
            object="chat.completion",
            system_fingerprint=None,
            service_tier=None,
        )
        return response


def openai_get_content_from_message(message: ChatCompletionRequestMessage):
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


def build_prompt(
    pipeline: TokenGeneratorPipeline,
    completion_request: CreateChatCompletionRequest,
) -> str:
    """Build the chat completion prompt."""
    request_prompt = ""
    if pipeline.tokenizer is not None and isinstance(
        pipeline.tokenizer, PreTrainedTokenGeneratorTokenizer
    ):
        messages = [
            {
                "role": message.root.role,
                "content": openai_get_content_from_message(message),
            }
            for message in completion_request.messages
        ]
        request_prompt = str(
            pipeline.tokenizer.delegate.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    else:
        request_prompt = " ".join(
            [
                message.root.content
                for message in completion_request.messages
                if isinstance(message.root.content, str)
            ]
        )
    return request_prompt


@router.post("/chat/completions")
async def openai_create_chat_completion(request: Request) -> Response:
    request_id = request.state.request_id
    try:
        request_json = await request.json()
        completion_request = CreateChatCompletionRequest.model_validate(
            request_json
        )
        pipeline_state = token_pipeline_state(completion_request.model)
        pipeline = pipeline_state.batched_generator
        request_prompt = build_prompt(pipeline, completion_request)

        logger.info(
            f"Processing {request.url.path}, {request_id} -"
            f" {'streaming' if completion_request.stream else ''} {completion_request.model} request."
        )

        response_generator = OpenAIChatResponseGenerator(
            pipeline, request_limiter=request.app.state.request_limiter
        )
        token_request = await response_generator.create_request(
            id=request_id,
            prompt=request_prompt,
            model_name=completion_request.model,
            max_new_tokens=completion_request.max_tokens,
        )

        if completion_request.stream:
            return EventSourceResponse(response_generator.stream(token_request))

        response = await response_generator.complete(token_request)
        return JSONResponse(response.model_dump_json())
    except JSONDecodeError as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail="Missing JSON.")
    except (TypeError, ValidationError) as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    except ValueError as e:
        logger.error(traceback.format_exc())
        # NOTE(matt): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.")


"""
Legacy OpenAI /completion endpoint.
https://platform.openai.com/docs/api-reference/completions
Public benchmarking such as vLLM use this endpoint.
"""


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[Logprobs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class CompletionStreamResponse(BaseModel):
    id: str
    created: int
    model: str
    choices: List[CompletionResponseStreamChoice]
    object: Literal["text_completion"]
    usage: Optional[CompletionUsage] = Field(default=None)


class OpenAICompletionResponseGenerator(OpenAIResponseGenerator):
    async def stream(self, request: TokenGeneratorRequest):
        logger.debug(
            "Streaming: Start: %s",
            request,
        )
        try:
            response_idx = 0
            async for token in self.pipeline.next_token(request):
                self.logger.debug(
                    "Streaming: %s, TOKEN: %d, %s",
                    request.id,
                    response_idx,
                    token,
                )
                # We support N = 1 at the moment and will generate a single choice.
                # The choice index is set to 0.
                # https://platform.openai.com/docs/api-reference/chat/object
                choices = [
                    CompletionResponseStreamChoice(
                        index=0,
                        text=token,
                    )
                ]
                # Each chunk is expected to have the same id
                # https://platform.openai.com/docs/api-reference/chat/streaming
                response = CompletionStreamResponse(
                    id=request.id,
                    choices=choices,
                    created=int(datetime.now().timestamp()),
                    model="",
                    object="text_completion",
                )
                response_idx += 1
                yield response.model_dump_json()

            logger.debug(
                "Streaming: Done: %s, %d tokens",
                request,
                response_idx,
            )
            yield "[DONE]"
        except ValueError as e:
            logger.error(traceback.format_exc())
            yield json.dumps({"result": "error", "message": str(e)})
        finally:
            if self.request_limiter:
                self.request_limiter.release()

    async def complete(
        self, request: TokenGeneratorRequest
    ) -> CreateCompletionResponse:
        completed_tokens = await self.pipeline.all_tokens(request)

        response_message = "".join(completed_tokens)
        response_choices = [
            Choice(
                index=0,
                text=response_message,
                finish_reason="stop",
                logprobs=Logprobs(),
            )
        ]
        response = CreateCompletionResponse(
            id=request.id,
            choices=response_choices,
            created=int(datetime.now().timestamp()),
            model="",
            object="text_completion",
            system_fingerprint=None,
        )
        return response


def openai_get_prompt_from_completion_request(
    request: CreateCompletionRequest,
) -> str:
    if isinstance(request.prompt, str):
        return request.prompt
    elif isinstance(request.prompt, list):
        if isinstance(request.prompt[0], str):
            prompt = "".join(cast(list[str], request.prompt))
            return prompt
    # We can do this for functional equivalent if needed.
    # However, the /completions API is already legacy.
    raise ValueError("Prompts of types other than strings are not supported.")


@router.post("/completions")
async def openai_create_completion(request: Request) -> Response:
    request_id = request.state.request_id
    try:
        request_json = await request.json()
        completion_request = CreateCompletionRequest.model_validate(
            request_json
        )
        pipeline_state = token_pipeline_state(completion_request.model)
        pipeline = pipeline_state.batched_generator
        logger.info(
            f"Processing {request.url.path}, {request_id} -"
            f" {'streaming' if completion_request.stream else ''} {completion_request.model} request."
        )

        request_prompt = openai_get_prompt_from_completion_request(
            completion_request
        )
        response_generator = OpenAICompletionResponseGenerator(
            pipeline, request_limiter=request.app.state.request_limiter
        )
        token_request = await response_generator.create_request(
            id=request_id,
            prompt=request_prompt,
            model_name=completion_request.model,
            max_new_tokens=completion_request.max_tokens,
        )

        if completion_request.stream:
            return EventSourceResponse(response_generator.stream(token_request))

        response = await response_generator.complete(token_request)
        return JSONResponse(response.model_dump_json())
    except JSONDecodeError as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail="Missing JSON.")
    except (TypeError, ValidationError) as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail="Invalid JSON.")
    except ValueError as e:
        logger.error(traceback.format_exc())
        # NOTE(matt): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.")


@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    return Response(status_code=200)
