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

from __future__ import annotations

import asyncio
import base64
import json
import logging
import queue
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from json.decoder import JSONDecodeError
from pathlib import Path
from random import randint
from typing import (
    Any,
    Generic,
    Literal,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)
from urllib.parse import unquote, urlparse

import aiofiles
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from httpx import AsyncClient
from max.interfaces import (
    AudioGenerationRequest,
    GenerationStatus,
    ImageContentPart,
    LoRAOperation,
    LoRARequest,
    LoRAStatus,
    MessageContent,
    PipelineTokenizer,
    RequestID,
    SamplingParams,
    SamplingParamsInput,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    TextGenerationResponseFormat,
)
from max.pipelines.core.exceptions import InputError
from max.pipelines.lib import PipelineConfig
from max.profiler import traced
from max.serve.config import Settings
from max.serve.parser import LlamaToolParser, parse_json_from_text
from max.serve.pipelines.llm import (
    AudioGeneratorPipeline,
    TokenGeneratorOutput,
    TokenGeneratorPipeline,
)
from max.serve.schemas.openai import (
    ChatCompletionMessageToolCall,
    ChatCompletionResponseMessage,
    ChatCompletionStreamOptions,
    ChatCompletionStreamResponseDelta,
    ChatCompletionTokenLogprob,
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
    InputItem,
    ListModelsResponse,
    LoadLoraRequest,
    Logprobs,
    Logprobs2,
    Model,
    PromptItem,
    ResponseFormatJsonObject,
    ResponseFormatJsonSchema,
    ResponseFormatText,
    TopLogprob,
    UnloadLoraRequest,
    Usage,
)
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch
from pydantic import AnyUrl, BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import State

_T = TypeVar("_T")

router = APIRouter(prefix="/v1")
logger = logging.getLogger("max.serve")


def record_request_start() -> None:
    METRICS.reqs_running(1)


@traced
def record_request_end(
    status_code: int,
    request_path: str,
    elapsed_ms: float,
    output_tokens: int | None = None,
    input_tokens: int | None = None,
) -> None:
    METRICS.reqs_running(-1)
    METRICS.request_count(status_code, request_path)
    METRICS.request_time(elapsed_ms, request_path)
    if output_tokens is not None:
        METRICS.output_tokens(output_tokens)
        METRICS.output_tokens_per_request(output_tokens)
    if input_tokens is not None:
        METRICS.input_tokens(input_tokens)
        METRICS.input_tokens_per_request(input_tokens)


@overload
def get_finish_reason_from_status(
    status: GenerationStatus, allow_none: Literal[True] = True
) -> Literal["stop", "length"] | None: ...


@overload
def get_finish_reason_from_status(
    status: GenerationStatus, allow_none: Literal[False]
) -> Literal["stop", "length"]: ...


def get_finish_reason_from_status(
    status: GenerationStatus, allow_none: bool = True
) -> Literal["stop", "length"] | None:
    if status == GenerationStatus.END_OF_SEQUENCE:
        return "stop"
    elif status == GenerationStatus.MAXIMUM_LENGTH:
        return "length"
    else:
        if not allow_none:
            raise ValueError(
                f"status: {status} has no associated finish_reason"
            )

        return None


class OpenAIResponseGenerator(ABC, Generic[_T]):
    def __init__(self, pipeline: TokenGeneratorPipeline) -> None:
        self.logger = logging.getLogger(
            "max.serve.router.OpenAIResponseGenerator"
        )
        self.pipeline = pipeline

    @abstractmethod
    async def stream(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[str | ErrorResponse | JSONResponse, None]:
        # This yield is required to make this method an async generator
        # for proper type checking. It will never be called due to @abstractmethod.
        yield ""
        raise NotImplementedError

    @abstractmethod
    async def complete(self, requests: list[TextGenerationRequest]) -> _T:
        pass


def get_pipeline(
    request: Request, model_name: str
) -> TokenGeneratorPipeline | AudioGeneratorPipeline:
    app_state: State = request.app.state
    pipeline: TokenGeneratorPipeline | AudioGeneratorPipeline = (
        app_state.pipeline
    )

    models = [pipeline.model_name]

    if lora_queue := app_state.pipeline.lora_queue:
        models += lora_queue.list_loras()

    if not model_name:
        model_name = pipeline.model_name

    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}', currently serving '{models}'."
        )
    if not isinstance(pipeline.tokenizer, PipelineTokenizer):
        raise ValueError(
            f"Tokenizer for '{model_name}' pipelines does not implement the PipelineTokenizer protocol."
        )
    return pipeline


@dataclass
class OpenAIChatResponseGenerator(
    OpenAIResponseGenerator[CreateChatCompletionResponse]
):
    def __init__(
        self,
        pipeline: TokenGeneratorPipeline,
        stream_options: ChatCompletionStreamOptions | None = None,
        parser: LlamaToolParser = field(default_factory=LlamaToolParser),
    ) -> None:
        super().__init__(pipeline)
        self.stream_options = stream_options
        self.parser = parser

    async def stream(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[str | ErrorResponse | JSONResponse, None]:
        self.logger.debug("Streaming: Start: %s", request)
        record_request_start()
        request_timer = StopWatch(start_ns=request.timestamp_ns)
        n_tokens = 0
        prompt_tokens = 0
        status_code = 200
        try:
            async for chunk in self.pipeline.next_token_chunk(request):
                self.logger.debug(
                    "Streaming: %s, TOKENS: %d, %s",
                    request.request_id,
                    chunk.token_count,
                    chunk.decoded_tokens,
                )

                if chunk.prompt_token_count:
                    prompt_tokens = chunk.prompt_token_count

                # We support N = 1 at the moment and will generate a single choice.
                # The choice index is set to 0.
                # https://platform.openai.com/docs/api-reference/chat/object

                # Process log probabilities for this chunk
                chunk_logprobs = _process_chat_log_probabilities([chunk])
                # Only include logprobs if there's content
                logprobs_response = (
                    chunk_logprobs if chunk_logprobs.content else None
                )

                if chunk.decoded_tokens is not None:
                    choices = [
                        Choice3(
                            index=0,
                            delta=ChatCompletionStreamResponseDelta(
                                content=chunk.decoded_tokens,
                                function_call=None,
                                role="assistant",
                                refusal=None,
                            ),
                            logprobs=logprobs_response,
                            finish_reason=get_finish_reason_from_status(
                                chunk.status, allow_none=True
                            ),
                        )
                    ]
                else:
                    # If we do not have decoded_tokens, we should guarantee we have a finish_reason.
                    choices = [
                        Choice3(
                            index=0,
                            delta=ChatCompletionStreamResponseDelta(
                                content="",
                            ),
                            finish_reason=get_finish_reason_from_status(
                                chunk.status, allow_none=False
                            ),
                        )
                    ]

                # Each chunk is expected to have the same id
                # https://platform.openai.com/docs/api-reference/chat/streaming
                # Don't include usage in regular chunks when streaming
                # https://platform.openai.com/docs/api-reference/chat/create#chat_create-stream_options
                response = CreateChatCompletionStreamResponse(
                    id=str(request.request_id),
                    choices=choices,
                    created=int(datetime.now().timestamp()),
                    model=request.model_name,
                    object="chat.completion.chunk",
                    system_fingerprint=None,
                    usage=None,
                    service_tier=None,
                )
                n_tokens += chunk.token_count
                payload = response.model_dump_json()
                yield payload

            logger.debug("Streaming: Done: %s, %d tokens", request, n_tokens)

            # If `include_usage=True`, send a final chunk with usage statistics
            if self.stream_options and self.stream_options.include_usage:
                final_usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=n_tokens,
                    total_tokens=n_tokens + prompt_tokens,
                )

                final_response = CreateChatCompletionStreamResponse(
                    id=str(request.request_id),
                    choices=[],
                    created=int(datetime.now().timestamp()),
                    model=request.model_name,
                    object="chat.completion.chunk",
                    system_fingerprint=None,
                    usage=final_usage,
                    service_tier=None,
                )
                yield final_response.model_dump_json()

            yield "[DONE]"
        except Exception as e:
            # Note that for SSE, the server will have already responded with a
            # 200 when establishing the connection.
            if isinstance(e, InputError):
                status_code = 400
                logger.warning(
                    "Input validation error in request %s: %s",
                    request.request_id,
                    str(e),
                )
            elif isinstance(e, ValueError):
                status_code = 400
                logger.exception("Exception in request %s", request.request_id)
            else:
                status_code = 500
                logger.exception("Exception in request %s", request.request_id)

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
                prompt_tokens,
            )

    async def complete(
        self, requests: list[TextGenerationRequest]
    ) -> CreateChatCompletionResponse:
        if len(requests) != 1:
            raise NotImplementedError(
                "chat completions does not support multiple prompts"
            )
        request = requests[0]
        record_request_start()
        n_tokens = 0
        prompt_tokens: int | None = None
        request_timer = StopWatch(start_ns=request.timestamp_ns)
        status_code = 200
        tool_use = request.tools is not None

        try:
            completed_outputs = await self.pipeline.all_tokens(request)

            n_tokens = sum(chunk.token_count for chunk in completed_outputs)
            if len(completed_outputs) > 0:
                prompt_tokens = completed_outputs[0].prompt_token_count

            response_message = "".join(
                chunk.decoded_tokens if chunk.decoded_tokens is not None else ""
                for chunk in completed_outputs
            )

            # Extract log probabilities if available
            logprobs = _process_chat_log_probabilities(completed_outputs)

            stop_sequence = [
                chunk.stop_sequence
                for chunk in completed_outputs
                if chunk.stop_sequence is not None
            ]
            finish_reason: Literal["stop", "length"]
            if len(stop_sequence) > 0:
                idx = response_message.find(stop_sequence[0])
                response_message = response_message[:idx]
                finish_reason = "stop"
            else:
                finish_reason = get_finish_reason_from_status(
                    completed_outputs[-1].status, allow_none=False
                )

            response_choices: list[Choice1] = []
            if tool_use and request.response_format is None:
                try:
                    response_choices = self.parser(response_message)
                except Exception as e:
                    # If parser fails, handle as traditional text
                    logging.warning(
                        f"Parsing for tool use failed, handling as general text response. Original error: {e}"
                    )
                    self._handle_text_response(
                        response_message,
                        response_choices,
                        finish_reason=finish_reason,
                        logprobs=logprobs,
                    )

            else:
                # Handle as regular text response if JSON cannot be parsed
                self._handle_text_response(
                    response_message,
                    response_choices,
                    finish_reason=finish_reason,
                    logprobs=logprobs,
                )

            usage = None
            if n_tokens > 0:
                usage = CompletionUsage(
                    prompt_tokens=completed_outputs[0].prompt_token_count,
                    completion_tokens=n_tokens,
                    total_tokens=n_tokens
                    + (completed_outputs[0].prompt_token_count or 0),
                )

            response = CreateChatCompletionResponse(
                id=str(request.request_id),
                choices=response_choices,
                created=int(datetime.now().timestamp()),
                model=request.model_name,
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
                prompt_tokens,
            )

    def _parse_resp_to_json(self, text: str) -> list[Any] | None:
        """Parse the response message to valid tool call JSON objects."""

        json_objects = parse_json_from_text(text)

        if not json_objects:
            return None

        return json_objects

    def _handle_text_response(
        self,
        response_message: str,
        response_choices: list[Choice1],
        finish_reason: Literal["stop", "length"],
        logprobs: Logprobs2 | None = None,
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
                finish_reason=finish_reason,
                logprobs=logprobs or Logprobs2(content=[], refusal=[]),
            )
        )

    def _handle_tool_calls_response(
        self,
        tool_data: dict[str, Any],
        tool_calls: list[ChatCompletionMessageToolCall],
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
    def __init__(self, pipeline: TokenGeneratorPipeline) -> None:
        self.pipeline = pipeline

    async def encode(
        self, requests: list[TextGenerationRequest]
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
                usage=None,
            )
            return response
        finally:
            record_request_end(
                status_code,
                metrics_req.request_path,
                request_timer.elapsed_ms,
            )


class OpenAISpeechResponseGenerator:
    def __init__(self, pipeline: AudioGeneratorPipeline) -> None:
        self.logger = logging.getLogger(
            "max.serve.router.OpenAISpeechResponseGenerator"
        )
        self.pipeline = pipeline

    async def synthesize_speech(
        self, request: AudioGenerationRequest
    ) -> CreateAudioGenerationResponse:
        self.logger.debug("Streaming: Start: %s", request)
        output = await self.pipeline.generate_full_audio(request)
        assert output.audio_data is not None
        audio_data = output.audio_data.tobytes()
        response = CreateAudioGenerationResponse(
            audio_data=base64.b64encode(audio_data),
            metadata=output.metadata.to_dict(),
        )
        return response


async def openai_parse_chat_completion_request(
    completion_request: CreateChatCompletionRequest,
    wrap_content: bool,
    settings: Settings,
) -> tuple[list[TextGenerationRequestMessage], list[bytes]]:
    """Parse the OpenAI ChatCompletionRequest to build TextGenerationRequestMessages.
    These will be used as inputs to the chat template to build the prompt.
    Also extract the list of image references while we are here so they can be
    downloaded and bundled alongside the request for preprocessing by pipelines.
    """
    messages: list[TextGenerationRequestMessage] = []
    image_refs: list[AnyUrl] = []
    resolve_image_tasks = []
    for m in completion_request.messages:
        if isinstance(m.root.content, list):
            message_content: list[MessageContent] = []
            for content_part in m.root.content:
                if content_part.root.type == "image_url":
                    image_refs.append(content_part.root.image_url.url)
                    if wrap_content:
                        message_content.append(ImageContentPart())
                    else:
                        message_content.append(content_part.model_dump())
                elif content_part.root.type == "text":
                    if wrap_content:
                        message_content.append(
                            TextContentPart(text=content_part.root.text)
                        )
                    else:
                        message_content.append(content_part.model_dump())
            messages.append(
                TextGenerationRequestMessage(
                    role=m.root.role, content=message_content
                )
            )
        else:
            messages.append(
                TextGenerationRequestMessage(
                    role=m.root.role,
                    content=m.root.content if m.root.content else "",
                )
            )

    resolve_image_tasks = [
        resolve_image_from_url(image_url, settings) for image_url in image_refs
    ]
    request_images = await asyncio.gather(*resolve_image_tasks)

    return messages, request_images


async def resolve_image_from_url(
    image_ref: AnyUrl, settings: Settings
) -> bytes:
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
    elif image_ref.scheme == "file":
        if settings is None:
            raise ValueError("Settings required for file URI resolution")

        # Parse the file URI.
        parsed = urlparse(str(image_ref))

        # Check host - only allow empty or localhost.
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raise ValueError(
                f"File URI with remote host '{parsed.netloc}' is not supported"
            )

        # Extract and decode the path.
        file_path = Path(unquote(parsed.path))

        # Validate against allowed roots.
        allowed_roots = [Path(root) for root in settings.allowed_image_roots]
        if not allowed_roots:
            raise ValueError(
                "File URI access denied: no allowed roots configured"
            )

        # Resolve the path, following symlinks.
        try:
            resolved_path = file_path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise ValueError(f"File not found: {file_path}") from e

        # Check if it's a directory.
        if resolved_path.is_dir():
            raise ValueError(f"Path is a directory: {resolved_path}")

        # Check if path is within allowed roots.
        path_allowed = False
        for root in allowed_roots:
            try:
                resolved_path.relative_to(root)
                path_allowed = True
                break
            except ValueError:
                continue

        if not path_allowed:
            raise ValueError(
                f"Path forbidden: {resolved_path} is outside allowed roots"
            )

        # Read the file with size limit.
        max_bytes = settings.max_local_image_bytes

        async with aiofiles.open(resolved_path, "rb") as f:
            images_bytes = await f.read(max_bytes + 1)
            if len(images_bytes) > max_bytes:
                raise ValueError(
                    f"File exceeds size limit of {max_bytes} bytes"
                )
        logger.debug(
            "ResolvedFileUri: %s -> %d bytes", resolved_path, len(images_bytes)
        )
        return images_bytes
    raise ValueError(f"Invalid image ref '{image_ref}'")


def _convert_stop(stop: str | list[str] | None) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return stop


def _get_target_endpoint(
    request: Request, body_target_endpoint: str | None
) -> str | None:
    """Extract target_endpoint from header or body.

    Header takes precedence over body parameter.
    Uses the header name 'X-Target-Endpoint'.

    Args:
        request: FastAPI Request object
        body_target_endpoint: target_endpoint from the request body

    Returns:
        target_endpoint value from header if present, otherwise from body
    """
    # Check for header first (takes precedence)
    header_target_endpoint = request.headers.get("X-Target-Endpoint")
    if header_target_endpoint:
        return header_target_endpoint

    # Fall back to body parameter
    return body_target_endpoint


@router.post("/chat/completions", response_model=None)
async def openai_create_chat_completion(
    request: Request,
) -> CreateChatCompletionResponse | EventSourceResponse:
    request_id = request.state.request_id
    try:
        completion_request = CreateChatCompletionRequest.model_validate_json(
            await request.body()
        )
        pipeline: TokenGeneratorPipeline | AudioGeneratorPipeline = (
            get_pipeline(request, completion_request.model)
        )
        assert isinstance(pipeline, TokenGeneratorPipeline)

        logger.debug(
            "Processing path, %s, req-id,%s%s, for model, %s.",
            request.url.path,
            request_id,
            " (streaming) " if completion_request.stream else "",
            completion_request.model,
        )

        (
            request_messages,
            request_images,
        ) = await openai_parse_chat_completion_request(
            completion_request,
            pipeline.tokenizer.expects_content_wrapping,
            request.app.state.settings,
        )

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

        stream_options = None
        if completion_request.stream:
            stream_options = completion_request.stream_options

        response_generator = OpenAIChatResponseGenerator(
            pipeline, stream_options=stream_options
        )
        sampling_params = SamplingParams.from_input_and_generation_config(
            SamplingParamsInput(
                top_k=completion_request.top_k,
                top_p=completion_request.top_p,
                temperature=completion_request.temperature,
                frequency_penalty=completion_request.frequency_penalty,
                presence_penalty=completion_request.presence_penalty,
                repetition_penalty=completion_request.repetition_penalty,
                max_new_tokens=completion_request.max_tokens,
                min_new_tokens=completion_request.min_tokens,
                ignore_eos=completion_request.ignore_eos,
                seed=completion_request.seed or randint(0, 2**63 - 1),
                stop_token_ids=completion_request.stop_token_ids,
                stop=_convert_stop(completion_request.stop),
            ),
            sampling_params_defaults=request.app.state.pipeline_config.model.sampling_params_defaults,
        )
        # For chat completions, logprobs is a bool and top_logprobs is the count.
        # We pass top_logprobs (or 1 if logprobs=True but top_logprobs not set).
        logprobs_count = 0
        if completion_request.logprobs:
            logprobs_count = (
                completion_request.top_logprobs
                if completion_request.top_logprobs is not None
                else 1
            )

        token_request = TextGenerationRequest(
            request_id=RequestID(request_id),
            model_name=completion_request.model,
            messages=request_messages,
            images=request_images,
            tools=tools,
            timestamp_ns=request.state.request_timer.start_ns,
            request_path=request.url.path,
            response_format=response_format,
            sampling_params=sampling_params,
            logprobs=logprobs_count,
            target_endpoint=_get_target_endpoint(
                request, completion_request.target_endpoint
            ),
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
                response_generator.stream(token_request), ping=100000, sep="\n"
            )

        response = await response_generator.complete([token_request])
        return response
    except JSONDecodeError as e:
        logger.exception("JSONDecodeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except (TypeError, ValidationError) as e:
        logger.exception("TypeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except InputError as e:
        logger.warning(
            "Input validation error in request %s: %s", request_id, str(e)
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        logger.exception("ValueError in request %s", request_id)
        # NOTE(SI-722): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.") from e


def _convert_chat_completion_tools_to_token_generator_tools(
    chat_tools: list[ChatCompletionTool] | None,
) -> list[TextGenerationRequestTool] | None:
    """Convert ChatCompletionTool list to TextGenerationRequestTool list."""
    if not chat_tools:
        return None

    token_generator_tools = []
    for tool in chat_tools:
        parameters = (
            tool.function.parameters.model_dump()
            if tool.function.parameters
            else {}
        )

        token_generator_tool = TextGenerationRequestTool(
            type=tool.type,
            function=TextGenerationRequestFunction(
                name=tool.function.name,
                description=tool.function.description,
                parameters=parameters,
            ),
        )
        token_generator_tools.append(token_generator_tool)

    return token_generator_tools


def _create_response_format(
    response_format: ResponseFormatText
    | ResponseFormatJsonObject
    | ResponseFormatJsonSchema
    | None,
) -> TextGenerationResponseFormat | None:
    """Convert OpenAI response format to TextGenerationResponseFormat."""
    if not response_format:
        return None

    # We don't have llguidance grammar for generic JSON output.
    # Only json_schema is supported for structured output.
    if response_format.type == "json_object":
        raise ValueError(
            "'json_object' response format is not supported. Use 'json_schema' instead for structured output."
        )

    json_schema: dict[Any, Any] = {}
    if (
        response_format.type == "json_schema"
        and response_format.json_schema.schema_ is not None
    ):
        json_schema = response_format.json_schema.schema_.model_dump()

    return TextGenerationResponseFormat(
        type=response_format.type, json_schema=json_schema
    )


@router.post("/embeddings", response_model=None)
async def openai_create_embeddings(
    request: Request,
) -> CreateEmbeddingResponse:
    request_id = request.state.request_id

    try:
        embeddings_request = CreateEmbeddingRequest.model_validate_json(
            await request.body()
        )
        pipeline: TokenGeneratorPipeline | AudioGeneratorPipeline = (
            get_pipeline(request, embeddings_request.model)
        )
        assert isinstance(pipeline, TokenGeneratorPipeline)

        logger.debug(
            "Processing path, %s, req-id, %s, for model, %s.",
            request.url.path,
            request_id,
            embeddings_request.model,
        )

        # We can support other types of inputs but it will require few more changes
        # to TextGenerationRequest and tokenizer encode. Hence, only supporting
        # string and list of strings for now.
        if not isinstance(embeddings_request.input, str | list):
            raise ValueError(
                "Input of type string or list of strings are only supported."
            )

        response_generator = OpenAIEmbeddingsResponseGenerator(pipeline)
        embedding_inputs: Sequence[StringPrompt | IntPrompt] = (
            get_prompts_from_openai_request(embeddings_request.input)
        )

        embedding_requests = [
            TextGenerationRequest(
                request_id=RequestID(f"{request_id}_{idx}"),
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
    except InputError as e:
        logger.warning(
            "Input validation error in request %s: %s", request_id, str(e)
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        logger.exception("ValueError in request %s", request_id)
        # NOTE(SI-722): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.") from e


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Logprobs | None = None
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class CompletionStreamResponse(BaseModel):
    id: str
    created: int
    model: str
    choices: list[CompletionResponseStreamChoice]
    object: Literal["text_completion"]
    usage: CompletionUsage | None = Field(default=None)


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


def _process_chat_log_probabilities(
    token_generator_outputs: list[TokenGeneratorOutput],
) -> Logprobs2:
    """Convert token generator outputs to chat completion log probabilities format.

    Args:
        token_generator_outputs: List of token generator outputs containing
            log probability information.

    Returns:
        Logprobs2 object with content tokens and their log probabilities.
    """
    content: list[ChatCompletionTokenLogprob] = []

    for output in token_generator_outputs:
        if (
            not output.token_log_probabilities
            or not output.top_log_probabilities
        ):
            continue

        # Iterate through each token's log probs
        for token_logprob, top_logprobs_dict in zip(
            output.token_log_probabilities,
            output.top_log_probabilities,
            strict=True,
        ):
            # Build top_logprobs list from the dict
            top_logprobs_list: list[TopLogprob] = []
            for token_str, logprob in top_logprobs_dict.items():
                top_logprobs_list.append(
                    TopLogprob(
                        token=token_str,
                        logprob=logprob,
                        # TODO(SERVSYS-1032): This will not properly handle
                        # incomplete characters.
                        bytes=list(token_str.encode("utf-8")),
                    )
                )

            # Sort by logprob descending
            top_logprobs_list.sort(key=lambda x: x.logprob, reverse=True)

            # Get the token string - it should be in top_logprobs_dict
            # The token with the highest logprob that matches token_logprob is the sampled token
            token_str = ""
            for t, lp in top_logprobs_dict.items():
                if abs(lp - token_logprob) < 1e-6:
                    token_str = t
                    break
            # Fallback: use the first token if no exact match found
            if not token_str and top_logprobs_list:
                token_str = top_logprobs_list[0].token

            content.append(
                ChatCompletionTokenLogprob(
                    token=token_str,
                    logprob=token_logprob,
                    bytes=list(token_str.encode("utf-8")),
                    top_logprobs=top_logprobs_list,
                )
            )

    return Logprobs2(content=content, refusal=[])


def get_app_pipeline_config(app: FastAPI) -> PipelineConfig:
    pipeline_config = app.state.pipeline_config
    assert isinstance(pipeline_config, PipelineConfig)
    return pipeline_config


class OpenAICompletionResponseGenerator(
    OpenAIResponseGenerator[CreateCompletionResponse]
):
    async def stream(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[str | ErrorResponse | JSONResponse, None]:
        logger.debug("Streaming: Start: %s", request)
        record_request_start()
        request_timer = StopWatch(start_ns=request.timestamp_ns)
        n_tokens = 0
        prompt_tokens = 0
        status_code = 200
        try:
            async for chunk in self.pipeline.next_token_chunk(request):
                self.logger.debug(
                    "Streaming: %s, TOKENS: %d, %s",
                    request.request_id,
                    chunk.token_count,
                    chunk.decoded_tokens,
                )

                if chunk.prompt_token_count:
                    prompt_tokens = chunk.prompt_token_count

                log_probs = _process_log_probabilities([chunk])

                # We support N = 1 at the moment and will generate a single choice.
                # The choice index is set to 0.
                # https://platform.openai.com/docs/api-reference/chat/object
                if chunk.decoded_tokens is not None:
                    choices = [
                        CompletionResponseStreamChoice(
                            index=0,
                            text=chunk.decoded_tokens,
                            logprobs=log_probs,
                            finish_reason=get_finish_reason_from_status(
                                chunk.status, allow_none=True
                            ),
                        )
                    ]
                else:
                    choices = [
                        CompletionResponseStreamChoice(
                            index=0,
                            text="",
                            finish_reason=get_finish_reason_from_status(
                                chunk.status, allow_none=False
                            ),
                        )
                    ]

                # Each chunk is expected to have the same id
                # https://platform.openai.com/docs/api-reference/chat/streaming
                response = CompletionStreamResponse(
                    id=request.request_id.value,
                    choices=choices,
                    created=int(datetime.now().timestamp()),
                    model=request.model_name,
                    object="text_completion",
                )
                n_tokens += chunk.token_count

                payload = response.model_dump_json()

                yield payload

            logger.debug("Streaming: Done: %s, %d tokens", request, n_tokens)
            yield "[DONE]"
        except queue.Full:
            logger.exception("Request queue full %s", request.request_id)
            yield JSONResponse(
                status_code=529,
                content={"detail": "Too Many Requests"},
                headers={"Retry-After": "30"},
            )
        except InputError as e:
            logger.warning(
                "Input validation error in request %s: %s",
                request.request_id,
                str(e),
            )
            yield JSONResponse(
                status_code=400,
                content={"detail": "Input validation error", "message": str(e)},
            )
        except ValueError as e:
            logger.exception("ValueError in request %s", request.request_id)
            # TODO (SI-722) - propagate better errors back.
            yield JSONResponse(
                status_code=500,
                content={"detail": "Value error", "message": str(e)},
            )
        finally:
            record_request_end(
                status_code,
                request.request_path,
                request_timer.elapsed_ms,
                n_tokens,
                prompt_tokens,
            )

    async def complete(
        self, requests: list[TextGenerationRequest]
    ) -> CreateCompletionResponse:
        # we assume that all entries in `requests` came from the same http
        # request and timestamp, request id, path should all be the same.
        record_request_start()
        n_tokens = 0
        prompt_tokens = 0
        request_timer = StopWatch(start_ns=requests[0].timestamp_ns)
        status_code = 200

        try:
            req_output_list = await asyncio.gather(
                *[self.pipeline.all_tokens(request) for request in requests]
            )
            response_choices = []
            for i, req_outputs in enumerate(req_output_list):
                n_tokens += sum(chunk.token_count for chunk in req_outputs)
                if req_outputs and req_outputs[0].prompt_token_count:
                    prompt_tokens += req_outputs[0].prompt_token_count

                log_probs = _process_log_probabilities(req_outputs)
                response_message = "".join(
                    chunk.decoded_tokens
                    if chunk.decoded_tokens is not None
                    else ""
                    for chunk in req_outputs
                )
                response_choices.append(
                    Choice(
                        index=i,
                        text=response_message,
                        finish_reason=get_finish_reason_from_status(
                            req_outputs[-1].status, allow_none=False
                        ),
                        logprobs=log_probs,
                    )
                )
            response = CreateCompletionResponse(
                # CreateCompletionResponse.id refers to the http request, while
                # request.request_id refers to the prompt. We don't have access to the
                # http request id in this context, so use requests[0].request_id
                id=str(requests[0].request_id),
                choices=response_choices,
                created=int(datetime.now().timestamp()),
                model=requests[0].model_name,
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
                prompt_tokens,
            )


# Prompts can be encoded 2 ways: as a string or as a sequence of integers.
StringPrompt = str
IntPrompt = Sequence[int]


def _is_sequence_of(
    items: Sequence[Any], item_type: type[_T]
) -> TypeGuard[Sequence[_T]]:
    return all(isinstance(item, item_type) for item in items)


def _is_seq_of_seq_of_int(
    items: Sequence[Any],
) -> TypeGuard[Sequence[Sequence[int]]]:
    return _is_sequence_of(items, list) and all(
        _is_sequence_of(item, int) for item in items
    )


def get_prompts_from_openai_request(
    prompt: str
    | list[str]
    | list[PromptItem]
    | list[InputItem]
    | list[int]
    | list[list[int]],
) -> Sequence[StringPrompt] | Sequence[IntPrompt]:
    """Extract the prompts from a CreateCompletionRequest

    Prompts can encoded as str or list-of-int. Within a given requests, there
    can be only one encoding.
    """
    if isinstance(prompt, str):
        return [prompt]
    if len(prompt) == 0:
        return []
    if _is_sequence_of(prompt, str):
        return prompt
    if _is_sequence_of(prompt, PromptItem):
        return [p.root for p in prompt]
    if _is_sequence_of(prompt, InputItem):
        return [p.root for p in prompt]
    if _is_sequence_of(prompt, int):
        return [prompt]
    if _is_seq_of_seq_of_int(prompt):
        return prompt
    raise Exception(f"unknown element type {type(prompt[0])}")


@router.post("/completions", response_model=None)
async def openai_create_completion(
    request: Request,
) -> CreateCompletionResponse | EventSourceResponse:
    """
    Legacy OpenAI /completion endpoint.
    https://platform.openai.com/docs/api-reference/completions
    Public benchmarking such as vLLM use this endpoint.
    """
    http_req_id = request.state.request_id
    try:
        completion_request = CreateCompletionRequest.model_validate_json(
            await request.body()
        )

        pipeline: TokenGeneratorPipeline | AudioGeneratorPipeline = (
            get_pipeline(request, completion_request.model)
        )
        assert isinstance(pipeline, TokenGeneratorPipeline)

        logger.debug(
            "Path: %s, Request: %s%s, Model: %s",
            request.url.path,
            http_req_id,
            " (streaming) " if completion_request.stream else "",
            completion_request.model,
        )

        response_generator = OpenAICompletionResponseGenerator(pipeline)
        prompts = get_prompts_from_openai_request(completion_request.prompt)
        token_requests = []
        for i, prompt in enumerate(prompts):
            prompt = cast(str | Sequence[int], prompt)
            sampling_params = SamplingParams.from_input_and_generation_config(
                SamplingParamsInput(
                    top_k=completion_request.top_k,
                    top_p=completion_request.top_p,
                    temperature=completion_request.temperature,
                    frequency_penalty=completion_request.frequency_penalty,
                    presence_penalty=completion_request.presence_penalty,
                    repetition_penalty=completion_request.repetition_penalty,
                    max_new_tokens=completion_request.max_tokens,
                    min_new_tokens=completion_request.min_tokens,
                    ignore_eos=completion_request.ignore_eos,
                    seed=completion_request.seed or randint(0, 2**63 - 1),
                    stop_token_ids=completion_request.stop_token_ids,
                    stop=_convert_stop(completion_request.stop),
                ),
                sampling_params_defaults=get_app_pipeline_config(
                    request.app
                ).model.sampling_params_defaults,
            )
            tgr = TextGenerationRequest(
                # Generate a unique request_id for each prompt in the request
                request_id=RequestID(f"{http_req_id}_{i}"),
                model_name=completion_request.model,
                prompt=prompt,
                timestamp_ns=request.state.request_timer.start_ns,
                request_path=request.url.path,
                logprobs=(
                    completion_request.logprobs
                    if completion_request.logprobs is not None
                    else 0
                ),
                echo=completion_request.echo or False,
                sampling_params=sampling_params,
                target_endpoint=_get_target_endpoint(
                    request, completion_request.target_endpoint
                ),
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
                response_generator.stream(token_requests[0]),
                ping=100000,
                sep="\n",
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

    if lora_queue := request.app.state.pipeline.lora_queue:
        model_list += [
            Model(id=lora, object="model", created=None, owned_by="")
            for lora in lora_queue.list_loras()
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
        audio_generation_request = (
            CreateAudioGenerationRequest.model_validate_json(
                await request.body()
            )
        )
        pipeline: TokenGeneratorPipeline | AudioGeneratorPipeline = (
            get_pipeline(request, audio_generation_request.model)
        )
        assert isinstance(pipeline, AudioGeneratorPipeline)
        sampling_params = SamplingParams.from_input_and_generation_config(
            SamplingParamsInput(
                min_new_tokens=audio_generation_request.min_tokens
            ),
            sampling_params_defaults=request.app.state.pipeline_config.model.sampling_params_defaults,
        )
        audio_request = AudioGenerationRequest(
            request_id=RequestID(request_id),
            input=audio_generation_request.input,
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
    except InputError as e:
        logger.warning(
            "Input validation error in request %s: %s", request_id, str(e)
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        logger.exception("ValueError in request %s", request_id)
        # NOTE(SI-722): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.") from e


@router.post("/load_lora_adapter", response_model=None)
async def load_lora_adapter(
    request: Request,
) -> JSONResponse:
    """Load a LoRA adapter into the pipeline."""
    request_id = request.state.request_id
    try:
        load_request = LoadLoraRequest.model_validate_json(await request.body())

        app_state: State = request.app.state

        # Check if LoRA is enabled
        if app_state.pipeline.lora_queue is None:
            raise HTTPException(
                status_code=501,
                detail="LoRA functionality is not enabled on this server. Please restart the server with LoRA enabled.",
            )

        response = await app_state.pipeline.lora_queue.get_response(
            RequestID(request_id),
            LoRARequest(
                LoRAOperation.LOAD,
                load_request.lora_name,
                load_request.lora_path,
            ),
        )

        # Map LoRA status to appropriate HTTP status codes
        if response.status == LoRAStatus.SUCCESS:
            return JSONResponse(
                status_code=200,
                content={
                    "status": response.status.value,
                    "message": response.message,
                },
            )
        elif response.status == LoRAStatus.LOAD_NAME_EXISTS:
            raise HTTPException(
                status_code=409, detail=response.message
            )  # Conflict
        elif response.status == LoRAStatus.LOAD_INVALID_PATH:
            raise HTTPException(
                status_code=400, detail=response.message
            )  # Bad Request
        elif response.status == LoRAStatus.LOAD_INVALID_ADAPTER:
            raise HTTPException(
                status_code=400, detail=response.message
            )  # Bad Request
        else:
            raise HTTPException(
                status_code=500, detail=response.message
            )  # Internal Server Error

    except JSONDecodeError as e:
        logger.exception("JSONDecodeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except (TypeError, ValidationError) as e:
        logger.exception("Validation error in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValueError as e:
        logger.exception("ValueError in request %s", request_id)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error loading LoRA adapter in request %s", request_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to load LoRA adapter: {str(e)}"
        ) from e


@router.post("/unload_lora_adapter", response_model=None)
async def unload_lora_adapter(
    request: Request,
) -> JSONResponse:
    """Unload a LoRA adapter from the pipeline."""
    request_id = request.state.request_id
    try:
        unload_request = UnloadLoraRequest.model_validate_json(
            await request.body()
        )

        app_state: State = request.app.state

        if app_state.pipeline.lora_queue is None:
            raise HTTPException(
                status_code=501,
                detail="LoRA functionality is not enabled on this server. Please restart the server with LoRA enabled.",
            )

        response = await app_state.pipeline.lora_queue.get_response(
            RequestID(request_id),
            LoRARequest(LoRAOperation.UNLOAD, unload_request.lora_name),
        )

        # Map LoRA status to appropriate HTTP status codes
        if response.status == LoRAStatus.SUCCESS:
            return JSONResponse(
                status_code=200,
                content={
                    "status": response.status.value,
                    "message": response.message,
                },
            )
        elif response.status == LoRAStatus.UNLOAD_NAME_NONEXISTENT:
            raise HTTPException(
                status_code=404, detail=response.message
            )  # Not Found
        else:
            raise HTTPException(
                status_code=500, detail=response.message
            )  # Internal Server Error

    except JSONDecodeError as e:
        logger.exception("JSONDecodeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except (TypeError, ValidationError) as e:
        logger.exception("Validation error in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValueError as e:
        logger.exception("ValueError in request %s", request_id)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Error unloading LoRA adapter in request %s", request_id
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to unload LoRA adapter: {str(e)}"
        ) from e
