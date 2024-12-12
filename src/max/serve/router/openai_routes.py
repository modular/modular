# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import base64
import json
import logging
import queue
from abc import ABC, abstractmethod
from datetime import datetime
from json.decoder import JSONDecodeError
from time import perf_counter_ns
from typing import Any, AsyncGenerator, List, Literal, Optional, Union, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from httpx import AsyncClient
from max.pipelines import (
    PipelineTokenizer,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
)
from max.serve.pipelines.llm import TokenGeneratorOutput, TokenGeneratorPipeline
from max.serve.schemas.openai import (  # type: ignore
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
    Error,
    ErrorResponse,
    Logprobs,
    Logprobs2,
)
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch
from pydantic import AnyUrl, BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import State

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)


def record_request_start():
    METRICS.reqs_running(1)


def record_request_end(
    status_code: int, request_path: str, elapsed_ms: float, n_tokens: int
) -> None:
    METRICS.reqs_running(-1)
    METRICS.request_count(status_code, request_path)
    METRICS.request_time(elapsed_ms, request_path)
    METRICS.output_tokens(n_tokens)


class OpenAIResponseGenerator(ABC):
    def __init__(
        self,
        pipeline: TokenGeneratorPipeline,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipeline = pipeline

    @abstractmethod
    async def stream(
        self, request: TokenGeneratorRequest
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def complete(self, request: TokenGeneratorRequest) -> str:
        pass


def get_pipeline(request: Request, model_name: str) -> TokenGeneratorPipeline:
    app_state: State = request.app.state
    pipeline: TokenGeneratorPipeline = app_state.pipeline
    if pipeline.model_name != model_name:
        raise ValueError(
            f"Unknown model '{model_name}', currently serving"
            f" '{pipeline.model_name}'."
        )
    if not isinstance(pipeline.tokenizer, PipelineTokenizer):
        raise ValueError(
            f"Tokenizer for '{model_name}' pipelines does not implement the PipelineTokenizer protocol."
        )
    return pipeline


class OpenAIChatResponseGenerator(OpenAIResponseGenerator):
    async def stream(self, request: TokenGeneratorRequest):
        self.logger.debug(
            "Streaming: Start: %s",
            request,
        )
        record_request_start()
        itl = StopWatch()
        request_timer = StopWatch(start_ns=request.req_recv_time_ns)
        n_tokens = 0
        status_code = 200
        try:
            async for token in self.pipeline.next_token(request):
                self.logger.debug(
                    "Streaming: %s, TOKEN: %d, %s",
                    request.id,
                    n_tokens,
                    token.decoded_token,
                )
                # We support N = 1 at the moment and will generate a single choice.
                # The choice index is set to 0.
                # https://platform.openai.com/docs/api-reference/chat/object
                choices = [
                    Choice3(
                        index=0,
                        delta=ChatCompletionStreamResponseDelta(
                            content=token.decoded_token,
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
                n_tokens += 1
                payload = response.model_dump_json()
                if n_tokens == 1:
                    METRICS.ttft(request_timer.elapsed_ms)
                else:
                    # TODO: re-instate ITL measurement when we find a fast way to do it
                    # METRICS.itl(itl.elapsed_ms)
                    pass
                itl.reset()
                yield payload

            logger.debug(
                "Streaming: Done: %s, %d tokens",
                request,
                n_tokens,
            )
            yield "[DONE]"
        except ValueError as e:
            status_code = 500
            logger.exception("ValueError in request %s", request.id)
            error_response = ErrorResponse(
                error=Error(
                    code=str(status_code), message=str(e), param="", type=""
                )
            )
            yield error_response
        finally:
            record_request_end(
                status_code,
                request.request_path,
                request_timer.elapsed_ms,
                n_tokens,
            )

    async def complete(
        self, request: TokenGeneratorRequest
    ) -> CreateChatCompletionResponse:
        record_request_start()
        n_tokens = 0
        request_timer = StopWatch(start_ns=request.req_recv_time_ns)
        status_code = 200
        try:
            completed_outputs = await self.pipeline.all_tokens(request)
            n_tokens = len(completed_outputs)
            response_message = "".join(
                output.decoded_token for output in completed_outputs
            )
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
        except ValueError as e:
            logger.exception("ValueError in %s", request.id)
            status_code = 500
            # TODO (SI-722) how to handle error in a stream response via ChatCompletion API.
            return json.dumps({"result": "error", "message": str(e)})
        finally:
            record_request_end(
                status_code,
                request.request_path,
                request_timer.elapsed_ms,
                n_tokens,
            )


def openai_parse_chat_completion_request(
    completion_request: CreateChatCompletionRequest,
    wrap_content: bool,
) -> tuple[list[TokenGeneratorRequestMessage], list[AnyUrl]]:
    """Parse the OpenAI ChatCompletionRequest to build TokenGeneratorRequestMessages.
    These will be used as inputs to the chat template to build the prompt.
    Also extract the list of image references while we are here so they can be
    downloaded and bundled alongside the request for preprocessing by pipelines.
    """
    messages: list[TokenGeneratorRequestMessage] = []
    image_refs: list[AnyUrl] = []
    for m in completion_request.messages:
        if isinstance(m.root.content, list):
            message_content: list[dict[str, Any]] = []
            for content_part in m.root.content:
                if content_part.root.type == "image_url":
                    image_refs.append(content_part.root.image_url.url)
                    if wrap_content:
                        message_content.append({"type": "image"})
                    else:
                        message_content.append(content_part.model_dump())
                elif content_part.root.type == "text":
                    if wrap_content:
                        message_content.append(
                            {
                                "type": content_part.root.type,
                                "content": content_part.root.text,
                            }
                        )
                    else:
                        message_content.append(content_part.model_dump())
            messages.append({"role": m.root.role, "content": message_content})
        else:
            messages.append(
                {
                    "role": m.root.role,
                    "content": m.root.content if m.root.content else "",
                }
            )
    return messages, image_refs


async def resolve_image_from_url(image_ref: AnyUrl) -> bytes:
    if image_ref.scheme == "http" or image_ref.scheme == "https":
        # TODO: Evaluate creating a single AsyncClient for the app.
        async with AsyncClient() as client:
            response = await client.get(str(image_ref))
            images_bytes = await response.aread()
            logger.debug(
                "ResolvedImageUrl: %s -> %d bytes", image_ref, len(images_bytes)
            )
            return images_bytes
    elif image_ref.scheme == "data":
        image_b64 = image_ref.unicode_string().split(",")[1]
        images_bytes = base64.decodebytes(image_b64.encode())
        logger.debug(
            "ResolvedImageB64: %s -> %d bytes",
            str(image_ref)[:16],
            len(images_bytes),
        )
        return images_bytes
    raise ValueError(f"Invalid image ref '{image_ref}'")


@router.post("/chat/completions", response_model=None)
async def openai_create_chat_completion(
    request: Request,
) -> Union[CreateChatCompletionResponse, EventSourceResponse]:
    request_id = request.state.request_id
    try:
        request_json = await request.json()
        completion_request = CreateChatCompletionRequest.model_validate(
            request_json
        )
        pipeline = get_pipeline(request, completion_request.model)

        logger.info(
            "Processing path, %s, req-id,%s%s, for model, %s.",
            request.url.path,
            request_id,
            " (streaming) " if completion_request.stream else "",
            completion_request.model,
        )

        request_messages, request_images_urls = (
            openai_parse_chat_completion_request(
                completion_request, pipeline.tokenizer.expects_content_wrapping
            )
        )

        request_images = None
        if request_images_urls:
            resolve_image_tasks = [
                resolve_image_from_url(image_url)
                for image_url in request_images_urls
            ]
            request_images = await asyncio.gather(*resolve_image_tasks)

        response_generator = OpenAIChatResponseGenerator(pipeline)
        token_request = TokenGeneratorRequest(
            id=request_id,
            index=0,
            model_name=completion_request.model,
            messages=request_messages,
            images=request_images,
            max_new_tokens=completion_request.max_tokens,
            req_recv_time_ns=request.state.request_timer.start_ns,
            request_path=request.url.path,
        )

        if completion_request.stream:
            # We set a large timeout for ping otherwise benchmarking scripts
            # such as sglang will fail in parsing the ping message.
            return EventSourceResponse(
                response_generator.stream(token_request), ping=100000
            )

        response = await response_generator.complete(token_request)
        return response
    except JSONDecodeError as e:
        logger.exception("JSONDecodeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except (TypeError, ValidationError) as e:
        logger.exception("TypeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValueError as e:
        logger.exception("ValueError in request %s", request_id)
        # NOTE(SI-722): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.") from e


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


def _process_log_probabilities(
    token_generator_outputs: list[TokenGeneratorOutput],
) -> Logprobs:
    token_log_probabilities = []
    top_log_probabilities = []
    for output in token_generator_outputs:
        if output.token_log_probabilities:
            token_log_probabilities.extend(output.token_log_probabilities)
        if output.top_log_probabilities:
            top_log_probabilities.extend(output.top_log_probabilities)

    return Logprobs(
        token_logprobs=token_log_probabilities,
        top_logprobs=top_log_probabilities,
    )


class OpenAICompletionResponseGenerator(OpenAIResponseGenerator):
    async def stream(self, request: TokenGeneratorRequest):
        logger.debug(
            "Streaming: Start: %s",
            request,
        )
        record_request_start()
        itl = StopWatch()
        request_timer = StopWatch(start_ns=request.req_recv_time_ns)
        n_tokens = 0
        status_code = 200
        try:
            async for token in self.pipeline.next_token(request):
                self.logger.debug(
                    "Streaming: %s, TOKEN: %d, %s",
                    request.id,
                    n_tokens,
                    token.decoded_token,
                )

                log_probs = _process_log_probabilities([token])
                # We support N = 1 at the moment and will generate a single choice.
                # The choice index is set to 0.
                # https://platform.openai.com/docs/api-reference/chat/object
                choices = [
                    CompletionResponseStreamChoice(
                        index=0,
                        text=token.decoded_token,
                        logprobs=log_probs,
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
                n_tokens += 1
                payload = response.model_dump_json()
                if n_tokens == 1:
                    METRICS.ttft(request_timer.elapsed_ms)
                else:
                    # TODO: re-instate ITL measurement when we find a fast way to do it
                    # METRICS.itl(itl.elapsed_ms)
                    pass
                itl.reset()
                yield payload

            logger.debug(
                "Streaming: Done: %s, %d tokens",
                request,
                n_tokens,
            )
            yield "[DONE]"
        except queue.Full as qe:
            status_code = 529
            logger.exception("Request queue full %s", request.id)
            yield JSONResponse(
                status_code=status_code,
                content={"detail": "Too Many Requests"},
                headers={"Retry-After": "30"},
            )
        except ValueError as e:
            status_code = 500
            logger.exception("ValueError in request %s", request.id)
            # TODO (SI-722) - propagate better errors back.
            yield JSONResponse(
                status_code=status_code,
                content={"detail": "Value error", "message": str(e)},
            )
        finally:
            record_request_end(
                status_code,
                request.request_path,
                request_timer.elapsed_ms,
                n_tokens,
            )

    async def complete(
        self, request: TokenGeneratorRequest
    ) -> CreateCompletionResponse:
        record_request_start()
        n_tokens = 0
        request_timer = StopWatch(start_ns=request.req_recv_time_ns)
        status_code = 200
        try:
            completed_outputs = await self.pipeline.all_tokens(request)
            n_tokens = len(completed_outputs)

            log_probs = _process_log_probabilities(completed_outputs)
            response_message = "".join(
                output.decoded_token for output in completed_outputs
            )
            response_choices = [
                Choice(
                    index=0,
                    text=response_message,
                    finish_reason="stop",
                    logprobs=log_probs,
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
        except:
            status_code = 500
            raise
        finally:
            record_request_end(
                status_code,
                request.request_path,
                request_timer.elapsed_ms,
                n_tokens,
            )


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


@router.post("/completions", response_model=None)
async def openai_create_completion(
    request: Request,
) -> Union[CreateCompletionResponse, EventSourceResponse]:
    """
    Legacy OpenAI /completion endpoint.
    https://platform.openai.com/docs/api-reference/completions
    Public benchmarking such as vLLM use this endpoint.
    """
    request_handler_ns = perf_counter_ns()
    request_id = request.state.request_id
    try:
        request_json = await request.json()
        request_json_ns = perf_counter_ns()
        completion_request = CreateCompletionRequest.model_validate(
            request_json
        )

        request_timers = {}
        request_timestamp_ns = request_json.get("timestamp", None)
        if request_timestamp_ns:
            request_timers["0_middleware"] = (
                request.state.request_timer.start_ns - request_timestamp_ns
            ) / 1e6
            request_timers["1_handler"] = (
                request_handler_ns - request_timestamp_ns
            ) / 1e6
            request_timers["2_json"] = (
                request_json_ns - request_timestamp_ns
            ) / 1e6

        pipeline = get_pipeline(request, completion_request.model)
        logger.info(
            "Path: %s, Request: %s%s, Model: %s%s",
            request.url.path,
            request_id,
            " (streaming) " if completion_request.stream else "",
            completion_request.model,
            ", Timers: {0}".format(request_timers) if request_timers else "",
        )

        response_generator = OpenAICompletionResponseGenerator(pipeline)
        request_content = openai_get_prompt_from_completion_request(
            completion_request
        )
        token_request = TokenGeneratorRequest(
            id=request_id,
            index=0,
            model_name=completion_request.model,
            prompt=request_content,
            max_new_tokens=completion_request.max_tokens,
            req_recv_time_ns=request.state.request_timer.start_ns,
            request_path=request.url.path,
            logprobs=completion_request.logprobs,
            echo=completion_request.echo,
        )

        if completion_request.stream:
            # We set a large timeout for ping otherwise benchmarking scripts
            # such as sglang will fail in parsing the ping message.
            return EventSourceResponse(
                response_generator.stream(token_request), ping=100000
            )

        response = await response_generator.complete(token_request)
        return response
    except JSONDecodeError as e:
        logger.exception("JSONDecodeError for request %s", request_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except (TypeError, ValidationError) as e:
        logger.exception("Validation error for request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValueError as e:
        logger.exception("ValueError for request %s", request_id)
        # NOTE(SI-722): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.") from e


@router.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)
