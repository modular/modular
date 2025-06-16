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

import asyncio
import base64
import json
import logging
import os
import queue
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from datetime import datetime
from json.decoder import JSONDecodeError
from time import perf_counter_ns
from typing import Any, Literal, Optional, Union, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from httpx import AsyncClient
from max.pipelines.core import (
    AudioGenerationRequest,
    PipelineTokenizer,
    TokenGeneratorRequest,
    TokenGeneratorRequestFunction,
    TokenGeneratorRequestMessage,
    TokenGeneratorRequestTool,
    TokenGeneratorResponseFormat,
)
from max.pipelines.core.interfaces.text_generation import SamplingParams
from max.profiler import Tracer, traced
from max.serve.pipelines.llm import (
    AudioGeneratorPipeline,
    TokenGeneratorOutput,
    TokenGeneratorPipeline,
)
from max.serve.schemas.openai import (  # type: ignore
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCalls,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    ChatCompletionTool,
    Choice,
    Choice1,
    Choice3,
    CompletionUsage,
    CreateAudioGenerationRequest,
    CreateAudioGenerationResponse,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    CreateEmbeddingRequest,
    CreateEmbeddingResponse,
    Embedding,
    Error,
    ErrorResponse,
    Function1,
    ListModelsResponse,
    Logprobs,
    Logprobs2,
    Model,
    PromptItem,
    ResponseFormatJsonObject,
    ResponseFormatJsonSchema,
    ResponseFormatText,
    Usage,
)
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch
from pydantic import AnyUrl, BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import State

router = APIRouter(prefix="/v1")
logger = logging.getLogger("max.serve")

# limits the number of concurrent tasks parsing incoming requests
# TODO(AITLIB-368): remove this after taking action mentioned in the ticket
MAX_SERVE_NUM_CONCURRENT_PARSING_TASKS = (
    "MAX_SERVE_NUM_CONCURRENT_PARSING_TASKS"
)
_NUM_CONCURRENT_PARSING_TASKS = int(
    os.environ.get(MAX_SERVE_NUM_CONCURRENT_PARSING_TASKS, 25)
)
_request_parsing_semaphore = asyncio.Semaphore(_NUM_CONCURRENT_PARSING_TASKS)


def record_request_start():
    METRICS.reqs_running(1)


@traced
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
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.logger.propagate = False
        self.pipeline = pipeline

    @abstractmethod
    async def stream(
        self, request: TokenGeneratorRequest
    ) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def complete(self, requests: list[TokenGeneratorRequest]) -> str:
        pass


def get_pipeline(
    request: Request, model_name: str
) -> Union[TokenGeneratorPipeline, AudioGeneratorPipeline]:
    app_state: State = request.app.state
    pipeline: Union[TokenGeneratorPipeline, AudioGeneratorPipeline] = (
        app_state.pipeline
    )
    if pipeline.model_name != model_name:
        raise ValueError(
            f"Unknown model '{model_name}', currently serving '{pipeline.model_name}'."
        )
    if not isinstance(pipeline.tokenizer, PipelineTokenizer):
        raise ValueError(
            f"Tokenizer for '{model_name}' pipelines does not implement the PipelineTokenizer protocol."
        )
    return pipeline


class OpenAIChatResponseGenerator(OpenAIResponseGenerator):
    async def stream(self, request: TokenGeneratorRequest):
        self.logger.debug("Streaming: Start: %s", request)
        record_request_start()
        request_timer = StopWatch(start_ns=request.timestamp_ns)
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
                        finish_reason=None,
                    )
                ]

                usage = Usage(
                    prompt_tokens=token.prompt_token_count,
                    completion_tokens=n_tokens,
                    total_tokens=n_tokens + (token.prompt_token_count or 0),
                )

                # Each chunk is expected to have the same id
                # https://platform.openai.com/docs/api-reference/chat/streaming
                response = CreateChatCompletionStreamResponse(
                    id=request.id,
                    choices=choices,
                    created=int(datetime.now().timestamp()),
                    model=self.pipeline.model_name,
                    object="chat.completion.chunk",
                    system_fingerprint=None,
                    usage=usage,
                    service_tier=None,
                )
                n_tokens += 1
                payload = response.model_dump_json()
                yield payload

            logger.debug("Streaming: Done: %s, %d tokens", request, n_tokens)
            yield "[DONE]"
        except Exception as e:
            # Note that for SSE, the server will have already responded with a
            # 200 when establishing the connection.
            status_code = 400 if isinstance(e, ValueError) else 500
            logger.exception("Exception in request %s", request.id)
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
        self, requests: list[TokenGeneratorRequest]
    ) -> CreateChatCompletionResponse:
        if len(requests) != 1:
            raise NotImplementedError(
                "chat completions does not support multiple prompts"
            )
        request = requests[0]
        record_request_start()
        n_tokens = 0
        request_timer = StopWatch(start_ns=request.timestamp_ns)
        status_code = 200
        tool_use = request.tools is not None

        try:
            completed_outputs = await self.pipeline.all_tokens(request)

            n_tokens = len(completed_outputs)

            response_message = "".join(
                output.decoded_token for output in completed_outputs
            )

            stop_sequence = [
                token.stop_sequence
                for token in completed_outputs
                if token.stop_sequence is not None
            ]
            if len(stop_sequence) > 0:
                idx = response_message.find(stop_sequence[0])
                response_message = response_message[:idx]

            response_choices: list[Choice1] = []
            if tool_use:
                # Try to parse and handle tool calls if tool_use is enabled
                tool_calls_resp = self._parse_resp_to_json(response_message)

                if tool_calls_resp:
                    tool_calls: list[ChatCompletionMessageToolCall] = []
                    for tool_data in tool_calls_resp:
                        self._handle_tool_calls_response(tool_data, tool_calls)

                    response_choices.append(
                        Choice1(
                            index=0,
                            message=ChatCompletionResponseMessage(
                                content="",
                                role="assistant",
                                tool_calls=ChatCompletionMessageToolCalls(
                                    root=tool_calls
                                ),
                            ),
                            finish_reason="tool_calls",
                            logprobs=Logprobs2(content=[], refusal=[]),
                        )
                    )
                else:
                    # Handle as regular text response if JSON cannot be parsed
                    self._handle_text_response(
                        response_message, response_choices
                    )
            else:
                # Handle as regular text response if tool_use is disabled
                self._handle_text_response(response_message, response_choices)

            usage = None
            if n_tokens > 0:
                usage = CompletionUsage(
                    prompt_tokens=completed_outputs[0].prompt_token_count,
                    completion_tokens=n_tokens,
                    total_tokens=n_tokens
                    + (completed_outputs[0].prompt_token_count or 0),
                )

            response = CreateChatCompletionResponse(
                id=request.id,
                choices=response_choices,
                created=int(datetime.now().timestamp()),
                model=self.pipeline.model_name,
                object="chat.completion",
                system_fingerprint=None,
                service_tier=None,
                usage=usage,
            )
            return response
        finally:
            record_request_end(
                status_code,
                request.request_path,
                request_timer.elapsed_ms,
                n_tokens,
            )

    def _parse_resp_to_json(self, text: str) -> Optional[list[dict]]:
        """Parse the response message to valid tool call JSON objects."""
        segments = [
            segment.strip() for segment in text.splitlines() if segment.strip()
        ]
        split_segments = []
        for segment in segments:
            split_segments.extend(segment.split(";"))

        # Filter out empty segments and parse as JSON
        json_objects = []
        for segment in split_segments:
            if segment.strip():  # Ignore empty segments
                try:
                    json_objects.append(json.loads(segment))
                except json.JSONDecodeError as e:
                    return None

        if not json_objects:
            return None

        return json_objects

    def _handle_text_response(
        self, response_message: str, response_choices: list
    ) -> None:
        """Handle regular text response by appending to response_choices."""
        response_choices.append(
            Choice1(
                index=0,
                message=ChatCompletionResponseMessage(
                    content=response_message,
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                    refusal="",
                ),
                finish_reason="stop",
                logprobs=Logprobs2(content=[], refusal=[]),
            )
        )

    def _handle_tool_calls_response(
        self, tool_data: dict, tool_calls: list
    ) -> None:
        """Handle tool response by appending to response_choices."""
        function_name = tool_data.get("name")
        if function_name and "parameters" in tool_data:
            short_uuid = str(uuid.uuid4()).replace("-", "")[:16]
            tool_call = ChatCompletionMessageToolCall(
                id=f"call_{short_uuid}",
                type="function",
                function=Function1(
                    name=function_name,
                    arguments=json.dumps(tool_data["parameters"]),
                ),
            )
            tool_calls.append(tool_call)


class OpenAIEmbeddingsResponseGenerator:
    def __init__(
        self,
        pipeline: TokenGeneratorPipeline,
    ):
        self.pipeline = pipeline

    async def encode(
        self, requests: list[TokenGeneratorRequest]
    ) -> CreateEmbeddingResponse:
        if len(requests) == 0:
            raise ValueError("No requests provided.")

        record_request_start()
        metrics_req = requests[0]
        request_timer = StopWatch(start_ns=metrics_req.timestamp_ns)
        status_code = 200

        try:
            embedding_outputs = await asyncio.gather(
                *[self.pipeline.encode(req) for req in requests]
            )

            embeddings_data = [
                Embedding(
                    object="embedding",
                    index=idx,
                    embedding=list(output.embeddings),
                )
                for idx, output in enumerate(embedding_outputs)
                if output is not None
            ]

            response = CreateEmbeddingResponse(
                data=embeddings_data,
                model=self.pipeline.model_name,
                object="list",
            )
            return response
        finally:
            record_request_end(
                status_code,
                metrics_req.request_path,
                request_timer.elapsed_ms,
                0,
            )


class OpenAISpeechResponseGenerator:
    def __init__(
        self,
        pipeline: AudioGeneratorPipeline,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pipeline = pipeline

    async def synthesize_speech(
        self, request: AudioGenerationRequest
    ) -> CreateAudioGenerationResponse:
        self.logger.debug("Streaming: Start: %s", request)
        response = await self.pipeline.generate_full_audio(request)
        audio_data = response.audio_data.numpy().tobytes()
        response = CreateAudioGenerationResponse(
            audio_data=base64.b64encode(audio_data), metadata=response.metadata
        )
        return response


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
            response = await client.get(str(image_ref), follow_redirects=True)
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
        async with _request_parsing_semaphore:
            request_json = await request.json()
        completion_request = CreateChatCompletionRequest.model_validate(
            request_json
        )
        pipeline = get_pipeline(request, completion_request.model)
        assert isinstance(pipeline, TokenGeneratorPipeline)

        logger.debug(
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

        tools = None
        if (
            completion_request.tool_choice is None
            or completion_request.tool_choice.root != "none"
        ):
            tools = _convert_chat_completion_tools_to_token_generator_tools(
                completion_request.tools
            )

        response_format = _create_response_format(
            completion_request.response_format
        )

        response_generator = OpenAIChatResponseGenerator(pipeline)
        sampling_params = SamplingParams(
            max_new_tokens=completion_request.max_tokens,
            stop=completion_request.stop,
            ignore_eos=completion_request.ignore_eos,
        )
        token_request = TokenGeneratorRequest(
            id=request_id,
            index=0,
            model_name=completion_request.model,
            messages=request_messages,
            images=request_images,
            tools=tools,
            timestamp_ns=request.state.request_timer.start_ns,
            request_path=request.url.path,
            response_format=response_format,
            sampling_params=sampling_params,
        )

        if completion_request.stream:
            # Currently, tools are not supported in streaming mode.
            if tools:
                raise HTTPException(
                    status_code=400,
                    detail="Tools are not supported in streaming mode.",
                )

            # We set a large timeout for ping otherwise benchmarking scripts
            # such as sglang will fail in parsing the ping message.
            return EventSourceResponse(
                response_generator.stream(token_request), ping=100000
            )

        response = await response_generator.complete([token_request])
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


def _convert_chat_completion_tools_to_token_generator_tools(
    chat_tools: Optional[list[ChatCompletionTool]],
) -> Optional[list[TokenGeneratorRequestTool]]:
    """Convert ChatCompletionTool list to TokenGeneratorRequestTool list."""
    if not chat_tools:
        return None

    token_generator_tools = []
    for tool in chat_tools:
        parameters = (
            tool.function.parameters.model_dump()
            if tool.function.parameters
            else {}
        )

        token_generator_tool = TokenGeneratorRequestTool(
            type=tool.type,
            function=TokenGeneratorRequestFunction(
                name=tool.function.name,
                description=tool.function.description,
                parameters=parameters,
            ),
        )
        token_generator_tools.append(token_generator_tool)

    return token_generator_tools


def _create_response_format(
    response_format: Optional[
        Union[
            ResponseFormatText,
            ResponseFormatJsonObject,
            ResponseFormatJsonSchema,
        ]
    ],
) -> Optional[TokenGeneratorResponseFormat]:
    """Convert OpenAI response format to TokenGeneratorResponseFormat."""
    if not response_format:
        return None

    response_type = response_format.type
    # We don't have XGrammar grammar for generic JSON output.
    # Only json_schema is supported for structured output.
    if response_type == "json_object":
        raise ValueError(
            "'json_object' response format is not supported. Use 'json_schema' instead for structured output."
        )

    json_schema = {}
    if response_type == "json_schema":
        json_schema = response_format.json_schema.schema_.model_dump()

    return TokenGeneratorResponseFormat(
        type=response_type, json_schema=json_schema
    )


@router.post("/embeddings", response_model=None)
async def openai_create_embeddings(
    request: Request,
) -> CreateEmbeddingResponse:
    request_id = request.state.request_id

    try:
        async with _request_parsing_semaphore:
            request_json = await request.json()
        embeddings_request = CreateEmbeddingRequest.model_validate(request_json)
        pipeline = get_pipeline(request, embeddings_request.model)
        assert isinstance(pipeline, TokenGeneratorPipeline)

        logger.debug(
            "Processing path, %s, req-id, %s, for model, %s.",
            request.url.path,
            request_id,
            embeddings_request.model,
        )

        # We can support other types of inputs but it will require few more changes
        # to TokenGeneratorRequest and tokenizer encode. Hence, only supporting
        # string and list of strings for now.
        if not isinstance(embeddings_request.input, (str, list)):
            raise ValueError(
                "Input of type string or list of strings are only supported."
            )

        response_generator = OpenAIEmbeddingsResponseGenerator(pipeline)
        embedding_inputs = (
            embeddings_request.input
            if isinstance(embeddings_request.input, list)
            else [embeddings_request.input]
        )

        embedding_requests = [
            TokenGeneratorRequest(
                id=f"{request_id}_{idx}",
                index=idx,
                model_name=embeddings_request.model,
                prompt=input_text,
                timestamp_ns=request.state.request_timer.start_ns,
                request_path=request.url.path,
            )
            for idx, input_text in enumerate(embedding_inputs)
        ]

        response = await response_generator.encode(embedding_requests)
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
    choices: list[CompletionResponseStreamChoice]
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
        logger.debug("Streaming: Start: %s", request)
        record_request_start()
        request_timer = StopWatch(start_ns=request.timestamp_ns)
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

                tracer = Tracer("process_log_probabilities")
                log_probs = _process_log_probabilities([token])
                del tracer  # process_log_probabilities

                tracer = Tracer("create_completion_stream_response")
                # We support N = 1 at the moment and will generate a single choice.
                # The choice index is set to 0.
                # https://platform.openai.com/docs/api-reference/chat/object
                choices = [
                    CompletionResponseStreamChoice(
                        index=0, text=token.decoded_token, logprobs=log_probs
                    )
                ]
                # Each chunk is expected to have the same id
                # https://platform.openai.com/docs/api-reference/chat/streaming
                response = CompletionStreamResponse(
                    id=request.id,
                    choices=choices,
                    created=int(datetime.now().timestamp()),
                    model=self.pipeline.model_name,
                    object="text_completion",
                )
                n_tokens += 1
                del tracer  # create_completion_stream_response

                tracer = Tracer("response.model_dump_json")
                payload = response.model_dump_json()
                del tracer  # response.model_dump_json

                yield payload

            logger.debug("Streaming: Done: %s, %d tokens", request, n_tokens)
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
        self, requests: list[TokenGeneratorRequest]
    ) -> CreateCompletionResponse:
        # we assume that all entries in `requests` came from the same http
        # request and timestamp, request id, path should all be the same.
        record_request_start()
        n_tokens = 0
        request_timer = StopWatch(start_ns=requests[0].timestamp_ns)
        status_code = 200

        try:
            req_output_list = await asyncio.gather(
                *[self.pipeline.all_tokens(request) for request in requests]
            )
            response_choices = []
            for i, req_outputs in enumerate(req_output_list):
                n_tokens += len(req_outputs)

                log_probs = _process_log_probabilities(req_outputs)
                response_message = "".join(
                    output.decoded_token for output in req_outputs
                )
                response_choices.append(
                    Choice(
                        index=i,
                        text=response_message,
                        finish_reason="stop",
                        logprobs=log_probs,
                    )
                )
            response = CreateCompletionResponse(
                # CreateCompletionResponse.id refers to the http request, while
                # request.id refers to the prompt. We don't have access to the
                # http request id in this context, so use requests[0].id
                id=requests[0].id,
                choices=response_choices,
                created=int(datetime.now().timestamp()),
                model=self.pipeline.model_name,
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
                requests[0].request_path,
                request_timer.elapsed_ms,
                n_tokens,
            )


# Prompts can be encoded 2 ways: as a string or as a sequence of integers.
StringPrompt = str
IntPrompt = Sequence[int]


def openai_get_prompts_from_completion_request(
    request: CreateCompletionRequest,
) -> Union[Sequence[StringPrompt], Sequence[IntPrompt]]:
    """Extract the prompts from a CreateCompletionRequest

    Prompts can encoded as str or list-of-int. Within a given requests, there
    can be only one encoding.
    """
    prompt = request.prompt
    if isinstance(prompt, str):
        return [prompt]
    if len(prompt) == 0:
        return []
    if isinstance(prompt[0], str):
        return prompt
    if isinstance(prompt[0], PromptItem):
        return [p.root for p in prompt]
    if isinstance(prompt[0], int):
        return [prompt]
    raise Exception("unknown element type {type(prompt[0])}")


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
    http_req_id = request.state.request_id
    try:
        async with _request_parsing_semaphore:
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
        assert isinstance(pipeline, TokenGeneratorPipeline)

        logger.debug(
            "Path: %s, Request: %s%s, Model: %s%s",
            request.url.path,
            http_req_id,
            " (streaming) " if completion_request.stream else "",
            completion_request.model,
            f", Timers: {request_timers}" if request_timers else "",
        )

        response_generator = OpenAICompletionResponseGenerator(pipeline)
        prompts = openai_get_prompts_from_completion_request(completion_request)
        token_requests = []
        for i, prompt in enumerate(prompts):
            prompt = cast(Union[str, Sequence[int]], prompt)
            sampling_params = SamplingParams(
                max_new_tokens=completion_request.max_tokens,
                ignore_eos=completion_request.ignore_eos,
            )
            tgr = TokenGeneratorRequest(
                # Generate a unique id for each prompt in the request
                id=f"{http_req_id}_{i}",
                index=i,
                model_name=completion_request.model,
                prompt=prompt,
                timestamp_ns=request.state.request_timer.start_ns,
                request_path=request.url.path,
                logprobs=completion_request.logprobs,
                echo=completion_request.echo,
                sampling_params=sampling_params,
            )
            token_requests.append(tgr)

        if completion_request.stream:
            if len(token_requests) != 1:
                raise NotImplementedError(
                    "Streaming responses for multiple prompts is not supported"
                )
            # We set a large timeout for ping otherwise benchmarking scripts
            # such as sglang will fail in parsing the ping message.
            return EventSourceResponse(
                response_generator.stream(token_requests[0]), ping=100000
            )

        resp = await response_generator.complete(token_requests)
        # ICK: The token generator doesn't know about http requests, so sets
        # the wrong id.  Overwrite with the http id.
        resp.id = http_req_id
        return resp
    except JSONDecodeError as e:
        logger.exception("JSONDecodeError for request %s", http_req_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except (TypeError, ValidationError) as e:
        logger.exception("Validation error for request %s", http_req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValueError as e:
        logger.exception("ValueError for request %s", http_req_id)
        # NOTE(SI-722): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.") from e


@router.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@router.get("/models", response_model=None)
async def openai_get_models(request: Request) -> ListModelsResponse:
    pipeline: TokenGeneratorPipeline = request.app.state.pipeline
    model_list = [
        Model(id=pipeline.model_name, object="model", created=None, owned_by="")
    ]

    return ListModelsResponse(object="list", data=model_list)


@router.get("/models/{model_id}", response_model=None)
async def openai_get_model(model_id: str, request: Request) -> Model:
    pipeline: TokenGeneratorPipeline = request.app.state.pipeline
    pipeline_model = Model(
        id=pipeline.model_name, object="model", created=None, owned_by=""
    )

    if model_id == pipeline.model_name:
        return pipeline_model

    # We need to handle the slash in our model names (not an issue for OpenAI)
    slash_ind = pipeline.model_name.rfind("/")
    if slash_ind != -1 and model_id == pipeline.model_name[slash_ind + 1 :]:
        return pipeline_model

    raise HTTPException(status_code=404)


# TODO: This is a temporary hack that does not conform to OpenAI spec.
@router.post("/audio/speech", response_model=None)
async def create_streaming_audio_speech(
    request: Request,
) -> CreateAudioGenerationResponse:
    """Audio generation endpoint that streams audio data."""
    try:
        request_id = request.state.request_id
        async with _request_parsing_semaphore:
            request_json = await request.json()

        audio_generation_request = CreateAudioGenerationRequest.model_validate(
            request_json
        )
        pipeline = get_pipeline(request, audio_generation_request.model)
        assert isinstance(pipeline, AudioGeneratorPipeline)
        sampling_params = SamplingParams(
            min_new_tokens=audio_generation_request.min_tokens
        )
        audio_request = AudioGenerationRequest(
            id=request_id,
            input=audio_generation_request.input,
            index=audio_generation_request.index,
            model=audio_generation_request.model,
            sampling_params=sampling_params,
            audio_prompt_tokens=audio_generation_request.audio_prompt_tokens,
            audio_prompt_transcription=audio_generation_request.audio_prompt_transcription,
            # TODO: Add support for these options.
            # instructions=audio_generation_request.instructions,
            # response_format=audio_generation_request.response_format,
            # speed=audio_generation_request.speed,
        )

        response_generator = OpenAISpeechResponseGenerator(pipeline)
        response = await response_generator.synthesize_speech(audio_request)
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
