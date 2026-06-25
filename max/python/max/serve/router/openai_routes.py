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
import json
import logging
import queue
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from json.decoder import JSONDecodeError
from random import randint
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from jinja2.exceptions import UndefinedError
from llguidance import LLMatcher
from max.pipelines.context import (
    GenerationStatus,
    SamplingParams,
    SamplingParamsInput,
    TextGenerationResponseFormat,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.tool_parsing import create as create_tool_parser
from max.pipelines.lib.tool_parsing import (
    maybe_name_from_tool,
    name_from_tool,
    names_from_tools,
)
from max.pipelines.lora import LoRAOperation, LoRARequest, LoRAStatus
from max.pipelines.modeling.types import (
    ImageContentPart,
    MessageContent,
    ParsedToolResponse,
    PipelineTokenizer,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
    VideoContentPart,
)
from max.profiler import Tracer, traced
from max.serve.config import Settings
from max.serve.parser import (
    LlamaToolParser,
    ToolParser,
    normalize_tool_call_arguments,
    parse_json_from_text,
)
from max.serve.parser.tool_call_normalization import (
    _normalize_tools_parameters,
    _validate_response_format_schema,
    normalize_response_format_schema,
)
from max.serve.parser.tool_call_validation import log_tool_call_conformance
from max.serve.pipelines.llm import (
    TokenGeneratorOutput,
    TokenGeneratorPipeline,
)
from max.serve.router._image_resolution import (
    decode_and_validate_images,
    resolve_image_from_url,
)
from max.serve.schemas.openai import (
    ChatCompletionLogprobs,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta,
    ChatCompletionTokenLogprob,
    CompletionLogprobs,
    CompletionResponseChoice,
    CompletionTokensDetails,
    CompletionUsage,
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
    ListModelsResponse,
    LoadLoraRequest,
    MaxModel,
    Model,
    PromptTokensDetails,
    TopLogprob,
    UnloadLoraRequest,
)
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceLogprobs as ChunkChoiceLogprobs,
)
from openai.types.chat.chat_completion_function_tool_param import (
    ChatCompletionFunctionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from openai.types.shared_params import (
    ResponseFormatJSONObject as ResponseFormatJsonObject,
)
from openai.types.shared_params import (
    ResponseFormatJSONSchema as ResponseFormatJsonSchema,
)
from openai.types.shared_params import ResponseFormatText as ResponseFormatText
from PIL import Image
from pydantic import AnyUrl, BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import State

_T = TypeVar("_T")

router = APIRouter(prefix="/v1")
logger = logging.getLogger("max.serve")

_CLIENT_DISCONNECTED_STATUS_CODE = 499

# Default tool-name charset (OpenAI's); a parser may widen it via VALID_TOOL_NAME_RE.
_DEFAULT_VALID_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# OpenAI's 64-char cap; checked by length so it holds even when a parser widens the charset.
_MAX_TOOL_NAME_LEN = 64

# Standard OpenAI message roles; a tokenizer may allow more via ``extra_chat_roles``.
_STANDARD_CHAT_ROLES = frozenset(
    {"developer", "system", "user", "assistant", "tool", "function"}
)


class _ClientDisconnectedError(RuntimeError):
    """Raised when a non-streaming request disconnects before completion."""


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
    status: GenerationStatus,
    allow_none: Literal[True] = True,
    *,
    has_tool_calls: Literal[False] = False,
) -> Literal["stop", "length"] | None: ...


@overload
def get_finish_reason_from_status(
    status: GenerationStatus,
    allow_none: Literal[True] = True,
    *,
    has_tool_calls: bool = False,
) -> Literal["stop", "length", "tool_calls"] | None: ...


@overload
def get_finish_reason_from_status(
    status: GenerationStatus,
    allow_none: Literal[False],
    *,
    has_tool_calls: Literal[False] = False,
) -> Literal["stop", "length"]: ...


@overload
def get_finish_reason_from_status(
    status: GenerationStatus,
    allow_none: Literal[False],
    *,
    has_tool_calls: bool = False,
) -> Literal["stop", "length", "tool_calls"]: ...


def get_finish_reason_from_status(
    status: GenerationStatus,
    allow_none: bool = True,
    *,
    has_tool_calls: bool = False,
) -> Literal["stop", "length", "tool_calls"] | None:
    if status == GenerationStatus.END_OF_SEQUENCE:
        return "tool_calls" if has_tool_calls else "stop"
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


def get_pipeline(request: Request, model_name: str) -> TokenGeneratorPipeline:
    app_state: State = request.app.state
    pipeline: TokenGeneratorPipeline = app_state.pipeline

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
        stream_options: ChatCompletionStreamOptionsParam | None = None,
        parser: ToolParser | None = None,
        parse_tool_calls: bool = False,
        tools: list[TextGenerationRequestTool] | None = None,
        fold_reasoning_into_content: bool = False,
    ) -> None:
        super().__init__(pipeline)
        self.stream_options = stream_options
        # MiniMax ``reasoning_split=False`` folds reasoning into ``content`` as ``<think>...</think>``; ``_think_*`` track the stream fold.
        self.fold_reasoning_into_content = fold_reasoning_into_content
        self._think_opened = False
        self._think_closed = False
        self.parser: ToolParser = (
            parser if parser is not None else LlamaToolParser()
        )
        # Whether to parse tool calls from the response.
        self.parse_tool_calls = parse_tool_calls
        # Function name -> JSON schema, used only for observability-only
        # schema-conformance logging (see tool_call_validation). The raw
        # client schema is kept so it matches what callers validate against.
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        for t in tools or []:
            name = maybe_name_from_tool(t)
            fn = t.get("function")
            if (
                name
                and isinstance(fn, dict)
                and isinstance(fn.get("parameters"), dict)
            ):
                self._tool_schemas[name] = fn["parameters"]
        # Per-call streaming accumulators for end-of-stream conformance check.
        self._stream_tool_names: dict[int, str] = {}
        self._stream_tool_args: dict[int, list[str]] = {}

    def _fold_reasoning_delta(
        self, reasoning_text: str | None, content_text: str | None
    ) -> str | None:
        """Folds a streaming delta's reasoning + content into one content string.

        Injects ``<think>\\n`` before the first reasoning text and
        ``\\n</think>\\n\\n`` before the first content text, so the full stream
        reconstructs ``<think>\\n{reasoning}\\n</think>\\n\\n{content}`` —
        matching the official MiniMax ``reasoning_split=False`` format. Relies on
        per-request ``_think_opened`` / ``_think_closed`` state.

        Args:
            reasoning_text: Decoded reasoning tokens in this delta, if any.
            content_text: Decoded content tokens in this delta, if any.

        Returns:
            The folded content string, or ``None`` when the delta carries no
            visible text.
        """
        parts: list[str] = []
        if reasoning_text:
            if not self._think_opened:
                parts.append("<think>\n")
                self._think_opened = True
            parts.append(reasoning_text)
        if content_text:
            if self._think_opened and not self._think_closed:
                parts.append("\n</think>\n\n")
                self._think_closed = True
            parts.append(content_text)
        return "".join(parts) or None

    async def stream(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[str | JSONResponse, None]:
        self.logger.debug("Streaming: Start: %s", request)
        record_request_start()
        request_timer = StopWatch(start_ns=request.timestamp_ns)
        n_reasoning_tokens = 0
        n_tokens = 0
        n_prompt_tokens = 0
        n_cached_prompt_tokens = 0
        status_code = 200
        has_emitted_tool_calls = False

        # Reset parser state for new streaming session
        if self.parse_tool_calls:
            self.parser.reset()
            self._stream_tool_names.clear()
            self._stream_tool_args.clear()

        # Reset the ``<think>`` fold state for this streaming session.
        self._think_opened = False
        self._think_closed = False

        try:
            async for chunk in self.pipeline.next_token_chunk(request):
                self.logger.debug(
                    "Streaming: %s, TOKENS: %d, %s%s",
                    request.request_id,
                    # TODO: (MODELS-1115) assume that the reasoning tokens are at the start of the chunk
                    # TODO: (MODELS-1117) determine whether to break out reasoning tokens into a separate metric
                    (chunk.reasoning_token_count or 0) + chunk.token_count,
                    (chunk.decoded_reasoning_tokens or ""),
                    (chunk.decoded_tokens or ""),
                )

                if chunk.prompt_token_count:
                    n_prompt_tokens = chunk.prompt_token_count

                if chunk.cached_token_count is not None:
                    n_cached_prompt_tokens = chunk.cached_token_count

                # We support N = 1 at the moment and will generate a single choice.
                # The choice index is set to 0.
                # https://platform.openai.com/docs/api-reference/chat/object

                # Process log probabilities for this chunk. The streaming
                # ``Choice`` lives in the chunk module and uses its own
                # ``ChoiceLogprobs`` class, so we re-validate against the
                # streaming type here.
                chunk_logprobs = _process_chat_log_probabilities([chunk])
                logprobs_response: ChunkChoiceLogprobs | None = None
                if chunk_logprobs.content:
                    logprobs_response = ChunkChoiceLogprobs.model_validate(
                        chunk_logprobs.model_dump()
                    )

                # Handle streaming tool calls if enabled
                merged_stream_content: str | None = None
                tool_call_chunks: list[ChoiceDeltaToolCall] = []
                if self.parse_tool_calls and chunk.decoded_tokens:
                    tool_deltas = self.parser.parse_delta(chunk.decoded_tokens)
                    if tool_deltas is not None:
                        # parse_delta returns [] (not None) once inside the
                        # tool-calls section, even if no deltas are ready yet.
                        # An empty list means "I consumed this chunk; suppress
                        # the raw structural tokens from flowing as content".
                        stream_content_parts: list[str] = []
                        for delta in tool_deltas:
                            if delta.content is not None:
                                stream_content_parts.append(delta.content)
                            if delta.name:
                                self._stream_tool_names[delta.index] = (
                                    delta.name
                                )
                            if delta.arguments:
                                self._stream_tool_args.setdefault(
                                    delta.index, []
                                ).append(delta.arguments)
                            if delta.id or delta.name or delta.arguments:
                                has_emitted_tool_calls = True
                                tool_call_chunks.append(
                                    ChoiceDeltaToolCall(
                                        index=delta.index,
                                        id=delta.id,
                                        type="function" if delta.id else None,
                                        function=ChoiceDeltaToolCallFunction(
                                            name=delta.name,
                                            arguments=delta.arguments,
                                        )
                                        if delta.name or delta.arguments
                                        else None,
                                    )
                                )

                        # Always assign a string (possibly "") so that
                        # merged_stream_content is non-None and prevents
                        # chunk.decoded_tokens from being used as content.
                        merged_stream_content = "".join(stream_content_parts)

                if (
                    self.parse_tool_calls
                    and chunk.status.is_done
                    and self._tool_schemas
                    and self._stream_tool_names
                ):
                    log_tool_call_conformance(
                        [
                            (
                                self._stream_tool_names[i],
                                "".join(self._stream_tool_args.get(i, [])),
                            )
                            for i in sorted(self._stream_tool_names)
                        ],
                        self._tool_schemas,
                        request_id=str(request.request_id),
                        streaming=True,
                    )

                if (
                    chunk.decoded_tokens is not None
                    or chunk.decoded_reasoning_tokens is not None
                    or tool_call_chunks
                    or merged_stream_content is not None
                ):
                    # Parsed streaming deltas may carry assistant text in
                    # ``content`` separate from tool-call argument deltas.
                    # When merged_stream_content is "" (parser consumed the
                    # chunk but has no content to emit), use None to avoid
                    # leaking raw structural tokens from chunk.decoded_tokens.
                    content = chunk.decoded_tokens
                    if merged_stream_content is not None:
                        content = merged_stream_content or None
                    elif tool_call_chunks:
                        content = None

                    # MiniMax ``reasoning_split=False``: fold reasoning into the
                    # content stream wrapped in ``<think>...</think>`` and drop
                    # the dedicated reasoning field. Tool-call deltas are left
                    # untouched.
                    reasoning = chunk.decoded_reasoning_tokens
                    if (
                        self.fold_reasoning_into_content
                        and not tool_call_chunks
                    ):
                        content = self._fold_reasoning_delta(reasoning, content)
                        reasoning = None

                    finish_reason = get_finish_reason_from_status(
                        chunk.status,
                        allow_none=True,
                        has_tool_calls=has_emitted_tool_calls,
                    )
                    choices = [
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=ChatCompletionStreamResponseDelta(
                                content=content,
                                function_call=None,
                                role="assistant",
                                refusal=None,
                                reasoning=reasoning,
                                tool_calls=tool_call_chunks
                                if tool_call_chunks
                                else None,
                            ),
                            logprobs=logprobs_response,
                            finish_reason=finish_reason,
                        )
                    ]
                elif chunk.status.is_done:
                    # Terminal chunk with no visible delta — emit the final
                    # choice carrying the finish_reason.
                    finish_reason = get_finish_reason_from_status(
                        chunk.status,
                        allow_none=False,
                        has_tool_calls=has_emitted_tool_calls,
                    )

                    choices = [
                        ChatCompletionStreamResponseChoice(
                            index=0,
                            delta=ChatCompletionStreamResponseDelta(
                                content="",
                            ),
                            finish_reason=finish_reason,
                        )
                    ]
                else:
                    # Reasoning-capable models (e.g. Gemma 4, Kimi K2.5) can
                    # emit intermediate chunks with no user-visible content
                    # while still ACTIVE — for example, a parser that
                    # consumed every token in the chunk as a structural
                    # delimiter. Skip those chunks instead of forcing a
                    # terminal finish_reason (which would raise because
                    # ACTIVE has no associated finish_reason).
                    n_reasoning_tokens += chunk.reasoning_token_count or 0
                    n_tokens += chunk.token_count
                    continue

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
                n_reasoning_tokens += chunk.reasoning_token_count or 0
                n_tokens += chunk.token_count
                # Omit unset (None) fields so each delta carries only what
                # changed, matching the OpenAI streaming spec. Without this a
                # delta serializes tool_calls/function_call/refusal as null on
                # every chunk, so a client reading the first tool_calls-bearing
                # delta sees null instead of the real tool-call fragment.
                payload = response.model_dump_json(exclude_none=True)
                yield payload

            # TODO: (MODELS-1117) determine whether to break out reasoning tokens into a separate metric
            logger.debug(
                "Streaming: Done: %s, %d tokens",
                request,
                n_reasoning_tokens + n_tokens,
            )

            # If `include_usage=True`, send a final chunk with usage statistics
            if self.stream_options and self.stream_options.get("include_usage"):
                final_usage = CompletionUsage(
                    prompt_tokens=n_prompt_tokens,
                    completion_tokens=n_reasoning_tokens + n_tokens,
                    total_tokens=n_prompt_tokens
                    + n_reasoning_tokens
                    + n_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=n_cached_prompt_tokens,
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=n_reasoning_tokens,
                    ),
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
                status_code = 500
                logger.exception("Exception in request %s", request.request_id)
            else:
                status_code = 500
                logger.exception("Exception in request %s", request.request_id)

            error_response = ErrorResponse(
                error=Error(
                    code=str(status_code), message=str(e), param="", type=""
                )
            )
            yield error_response.model_dump_json()
        finally:
            record_request_end(
                status_code,
                request.request_path,
                request_timer.elapsed_ms,
                # TODO: (MODELS-1117) determine whether to break out reasoning tokens into a separate metric
                n_reasoning_tokens + n_tokens,
                n_prompt_tokens,
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
        n_reasoning_tokens = 0
        n_tokens = 0
        n_prompt_tokens = 0
        n_cached_prompt_tokens = 0
        request_timer = StopWatch(start_ns=request.timestamp_ns)
        status_code = 200

        try:
            completed_outputs = await self.pipeline.all_tokens(request)

            n_reasoning_tokens = sum(
                chunk.reasoning_token_count or 0 for chunk in completed_outputs
            )
            n_tokens = sum(chunk.token_count for chunk in completed_outputs)
            if len(completed_outputs) > 0:
                n_prompt_tokens = completed_outputs[0].prompt_token_count or 0
                if completed_outputs[0].cached_token_count is not None:
                    n_cached_prompt_tokens = completed_outputs[
                        0
                    ].cached_token_count

            response_message = "".join(
                chunk.decoded_tokens
                for chunk in completed_outputs
                if chunk.decoded_tokens is not None
            )

            reasoning_message: str | None = None
            # TODO: (MODELS-1115) assume that the reasoning tokens are at the start of the chunk
            if any(
                chunk.decoded_reasoning_tokens is not None
                for chunk in completed_outputs
            ):
                reasoning_message = (
                    "".join(
                        chunk.decoded_reasoning_tokens
                        for chunk in completed_outputs
                        if chunk.decoded_reasoning_tokens is not None
                    )
                    or None
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

            # Kimi K2.5 (thinking enabled) can answer inside the prefilled
            # ``<think>`` block and stop without emitting ``</think>``, so the
            # reasoning parser routes the whole answer to reasoning and leaves
            # content empty. On a voluntary stop, surface that reasoning as
            # content so a successful turn never returns ``message.content``
            # null. On ``length`` (truncated mid-thought) keep it as reasoning
            # rather than misrepresenting a partial thought as the answer.
            # Skipped when folding reasoning into content: the ``<think>`` block
            # already guarantees ``message.content`` is non-null.
            if (
                not self.fold_reasoning_into_content
                and not response_message.strip()
                and reasoning_message
                and finish_reason == "stop"
            ):
                response_message = reasoning_message
                reasoning_message = None

            response_choices: list[ChatCompletionResponseChoice] = []
            # Note: Do not gate on `response_format is None` here.
            # The TextGenerationRequest was mutated to contain a
            # response format with type="grammar" since tools are involved
            # (see openai_create_chat_completion).
            if self.parse_tool_calls:
                try:
                    parsed = self.parser.parse_complete(response_message)
                    # Schema-aware argument coercion (parsers that opt in, e.g.
                    # MiniMax-M3, whose bare-text XML loses scalar types).
                    # No-op for parsers without `coerce_arguments`.
                    coerce_args = getattr(self.parser, "coerce_arguments", None)
                    if coerce_args is not None and self._tool_schemas:
                        for tc in parsed.tool_calls:
                            tc_schema = self._tool_schemas.get(tc.name)
                            if not tc_schema:
                                continue
                            try:
                                tc_args = json.loads(tc.arguments)
                            except (JSONDecodeError, ValueError):
                                continue
                            if isinstance(tc_args, dict):
                                tc.arguments = json.dumps(
                                    coerce_args(tc_args, tc_schema),
                                    ensure_ascii=False,
                                )
                    if parsed.tool_calls:
                        if self._tool_schemas:
                            log_tool_call_conformance(
                                [
                                    (tc.name, tc.arguments)
                                    for tc in parsed.tool_calls
                                ],
                                self._tool_schemas,
                                request_id=str(request.request_id),
                                streaming=False,
                            )
                        response_choices = self._tool_response_to_choices(
                            parsed, logprobs=logprobs
                        )
                    else:
                        # No tool calls found, handle as text
                        self._handle_text_response(
                            response_message,
                            response_choices,
                            finish_reason=finish_reason,
                            logprobs=logprobs,
                        )
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

            if reasoning_message is not None:
                if self.fold_reasoning_into_content:
                    # MiniMax ``reasoning_split=False``: fold reasoning into
                    # ``content`` wrapped in ``<think>...</think>`` and leave the
                    # dedicated reasoning field unset.
                    think_block = f"<think>\n{reasoning_message}\n</think>\n\n"
                    for choice in response_choices:
                        choice.message.content = think_block + (
                            choice.message.content or ""
                        )
                else:
                    for choice in response_choices:
                        choice.message.reasoning = reasoning_message

            usage = None
            if n_reasoning_tokens > 0 or n_tokens > 0:
                usage = CompletionUsage(
                    prompt_tokens=n_prompt_tokens,
                    completion_tokens=n_reasoning_tokens + n_tokens,
                    total_tokens=n_prompt_tokens
                    + n_reasoning_tokens
                    + n_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=n_cached_prompt_tokens,
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=n_reasoning_tokens,
                    ),
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
                # TODO: (MODELS-1117) determine whether to break out reasoning tokens into a separate metric
                n_reasoning_tokens + n_tokens,
                n_prompt_tokens,
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
        response_choices: list[ChatCompletionResponseChoice],
        finish_reason: Literal["stop", "length"],
        logprobs: ChatCompletionLogprobs | None = None,
    ) -> None:
        """Handle regular text response by appending to response_choices."""
        response_choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    content=response_message,
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                    refusal="",
                ),
                finish_reason=finish_reason,
                logprobs=logprobs
                or ChatCompletionLogprobs(content=[], refusal=[]),
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
                function=ChatCompletionMessageToolCallFunction(
                    name=function_name,
                    arguments=json.dumps(tool_data["parameters"]),
                ),
            )
            tool_calls.append(tool_call)

    def _tool_response_to_choices(
        self,
        parsed: ParsedToolResponse,
        logprobs: ChatCompletionLogprobs | None = None,
    ) -> list[ChatCompletionResponseChoice]:
        """Translates a ParsedToolResponse to a list of chat completion choices."""
        tool_calls_list: list[ChatCompletionMessageToolCallUnion] = [
            ChatCompletionMessageToolCall(
                id=tc.id,
                type="function",
                function=ChatCompletionMessageToolCallFunction(
                    name=tc.name, arguments=tc.arguments
                ),
            )
            for tc in parsed.tool_calls
        ]
        return [
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    content=parsed.content or "",
                    role="assistant",
                    tool_calls=tool_calls_list or None,
                    function_call=None,
                    refusal="",
                ),
                finish_reason="tool_calls",
                logprobs=logprobs
                or ChatCompletionLogprobs(content=[], refusal=[]),
            )
        ]


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
                # OpenAI requires usage; MAX doesn't yet track embedding token
                # counts so report zeros until we wire that through.
                usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
            )
            return response
        finally:
            record_request_end(
                status_code,
                metrics_req.request_path,
                request_timer.elapsed_ms,
            )


def _normalize_openai_role(role: str) -> Any:
    # The ``role`` options in OpenAI model spec include "developer" as a replacement for "system".
    # No MAX-supported chat template branches on developer
    # vs system, so collapse to ``system`` before constructing the internal
    # ``TextGenerationRequestMessage``.
    return "system" if role == "developer" else role


class _ParsedChatRequest(NamedTuple):
    """The parsed pieces of a chat-completion request.

    ``decoded_images`` are the validated, decoded images (decoded once); they
    are carried on the request so the tokenizer does not decode the same bytes
    a second time. See :func:`decode_and_validate_images`.
    """

    messages: list[TextGenerationRequestMessage]
    images: list[bytes]
    videos: list[bytes]
    decoded_images: list[Image.Image]


def _coerce_positive_int(value: Any) -> int | None:
    """Coerces a value to a positive int, else ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _coerce_positive_float(value: Any) -> float | None:
    """Coerces a value to a positive float, else ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


def _validate_tool_message_consistency(
    messages: Sequence[Mapping[str, Any]],
) -> None:
    """Rejects malformed tool exchanges (bad JSON args, unmatched/partial replies) with a 400.

    A trailing assistant ``tool_calls`` message with no following turn is left untouched.

    Raises:
        InputError: On any violation.
    """
    n = len(messages)
    i = 0
    while i < n:
        msg = messages[i]
        tool_calls = msg.get("tool_calls")
        if not (
            msg.get("role") == "assistant"
            and isinstance(tool_calls, list)
            and tool_calls
        ):
            i += 1
            continue

        expected_ids: list[str] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if isinstance(fn, dict):
                args = fn.get("arguments")
                if isinstance(args, str) and args.strip():
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        raise InputError(
                            "tool_call arguments must be valid JSON; got "
                            f"invalid JSON for tool call {tc.get('id')!r}."
                        ) from e
            tc_id = tc.get("id")
            if tc_id is not None:
                expected_ids.append(tc_id)

        # Consume the run of ``tool`` replies answering this assistant message.
        answered: set[str] = set()
        j = i + 1
        while j < n and messages[j].get("role") == "tool":
            reply_id = messages[j].get("tool_call_id")
            if reply_id not in expected_ids:
                raise InputError(
                    f"tool message tool_call_id {reply_id!r} does not match "
                    "any tool_call in the preceding assistant message."
                )
            answered.add(reply_id)
            j += 1

        # Require completeness only when more messages follow this turn.
        if i + 1 < n:
            missing = [tid for tid in expected_ids if tid not in answered]
            if missing:
                raise InputError(
                    "every tool_call must be answered by a tool message before "
                    f"the conversation continues; missing replies for {missing}."
                )
        i = j


async def openai_parse_chat_completion_request(
    completion_request: CreateChatCompletionRequest,
    wrap_content: bool,
    settings: Settings,
    max_images_per_request: int | None = None,
    max_image_bytes: int | None = None,
    allowed_roles: frozenset[str] | None = None,
) -> _ParsedChatRequest:
    """Parse the OpenAI ChatCompletionRequest to build TextGenerationRequestMessages.
    These will be used as inputs to the chat template to build the prompt.
    Also extract the list of image/video references while we are here so they
    can be downloaded and bundled alongside the request for preprocessing by
    pipelines.

    ``max_images_per_request`` and ``max_image_bytes`` are model-specific image
    limits supplied by the caller (read off the tokenizer); ``None`` means the
    corresponding limit is not enforced.

    ``allowed_roles`` is the set of message roles the model accepts; ``None``
    skips role validation (vendor roles are only allowed for models that
    declare them via ``extra_chat_roles``).
    """
    _validate_tool_message_consistency(completion_request.messages)
    if allowed_roles is not None:
        for m in completion_request.messages:
            role = m.get("role")
            if role not in allowed_roles:
                raise InputError(
                    f"role {role!r} is not supported by this model; "
                    f"allowed roles are {sorted(allowed_roles)}."
                )

    messages: list[TextGenerationRequestMessage] = []
    image_refs: list[AnyUrl] = []
    video_refs: list[AnyUrl] = []
    for m in completion_request.messages:
        # ``CreateChatCompletionRequest.messages`` carries OpenAI's
        # ``ChatCompletionMessageParam`` TypedDicts (plus a MAX-specific
        # ``video_url`` content part); access via dict keys.
        content = m.get("content")
        raw_tool_calls = m.get("tool_calls")
        tool_calls: list[dict[str, Any]] | None = (
            normalize_tool_call_arguments([dict(tc) for tc in raw_tool_calls])
            if isinstance(raw_tool_calls, list) and raw_tool_calls
            else None
        )
        tool_call_id = m.get("tool_call_id")
        reasoning_content = m.get("reasoning_content")

        if isinstance(content, list):
            # ``TextGenerationRequestMessage`` accepts plain dicts here and
            # coerces them into ``MessageContent`` parts via a field
            # validator, so we hand it a list of dicts when not wrapping.
            message_content: list[MessageContent | dict[str, Any]] = []
            for content_part in content:
                # Each entry of the OpenAI ``content`` array must be a
                # ``ChatCompletionContentPart`` object. Scalars and
                # lists bypass pydantic's ``Iterable[ContentPart]``
                # typing because ``messages`` is declared as
                # ``list[dict[str, Any]]`` on
                # ``CreateChatCompletionRequest`` (see
                # ``serve/schemas/openai.py``), so the value reaches
                # this loop as raw JSON. Reject anything we cannot
                # ``.get("type")`` on with a 400 rather than letting
                # ``AttributeError`` escape as a 500.
                if not isinstance(content_part, dict):
                    raise InputError(
                        "Each entry of message.content must be a content "
                        "part object (e.g. {'type': 'text', 'text': ...}); "
                        f"got {type(content_part).__name__}."
                    )
                part_type = content_part.get("type")
                if part_type == "image_url":
                    image_url = content_part["image_url"]
                    image_refs.append(AnyUrl(image_url["url"]))
                    if wrap_content:
                        # Carry the optional sizing hint onto the placeholder.
                        message_content.append(
                            ImageContentPart(
                                max_long_side_pixel=_coerce_positive_int(
                                    image_url.get("max_long_side_pixel")
                                )
                            )
                        )
                    else:
                        message_content.append(dict(content_part))
                elif part_type == "video_url":
                    video_url = content_part["video_url"]
                    video_refs.append(AnyUrl(video_url["url"]))
                    if wrap_content:
                        # Carry the optional sampling/sizing hints onto the
                        # placeholder.
                        message_content.append(
                            VideoContentPart(
                                fps=_coerce_positive_float(
                                    video_url.get("fps")
                                ),
                                max_frames=_coerce_positive_int(
                                    video_url.get("max_frames")
                                ),
                                max_long_side_pixel=_coerce_positive_int(
                                    video_url.get("max_long_side_pixel")
                                ),
                            )
                        )
                    else:
                        message_content.append(dict(content_part))
                elif part_type == "text":
                    if wrap_content:
                        message_content.append(
                            TextContentPart(text=content_part["text"])
                        )
                    else:
                        message_content.append(dict(content_part))
            messages.append(
                TextGenerationRequestMessage(
                    role=_normalize_openai_role(m["role"]),
                    content=cast(list[MessageContent], message_content),
                    tool_calls=tool_calls,
                    tool_call_id=tool_call_id,
                    reasoning_content=reasoning_content,
                )
            )
        else:
            messages.append(
                TextGenerationRequestMessage(
                    role=_normalize_openai_role(m["role"]),
                    content=content or "",
                    tool_calls=tool_calls,
                    tool_call_id=tool_call_id,
                    reasoning_content=reasoning_content,
                )
            )

    # Reject over-limit requests before downloading any image.
    if (
        max_images_per_request is not None
        and len(image_refs) > max_images_per_request
    ):
        raise InputError(
            f"too many images: {len(image_refs)} exceeds the maximum of "
            f"{max_images_per_request} images per request"
        )

    resolve_image_tasks = [
        resolve_image_from_url(image_url, settings) for image_url in image_refs
    ]
    request_images = await asyncio.gather(*resolve_image_tasks)

    # Fully decoding every image is CPU-bound (a few ms to tens of ms each), so
    # run it off the event loop to avoid blocking concurrent requests. PIL's C
    # codecs release the GIL during decode, so this is genuinely concurrent.
    # The decoded images are carried on the request and reused by the tokenizer
    # (decode-once), so this is the only place a request's images are decoded.
    decoded_images = await asyncio.to_thread(
        decode_and_validate_images, request_images, max_image_bytes
    )

    resolve_video_tasks = [
        resolve_image_from_url(video_url, settings) for video_url in video_refs
    ]
    request_videos = await asyncio.gather(*resolve_video_tasks)

    return _ParsedChatRequest(
        messages, request_images, list(request_videos), decoded_images
    )


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


def _resolve_grammar_constraints(
    tools: list[TextGenerationRequestTool] | None,
    tool_choice: str | dict[str, Any] | None,
    response_format: TextGenerationResponseFormat | None,
) -> tuple[
    list[TextGenerationRequestTool] | None, dict[str, Any] | None, bool, bool
]:
    """Determine grammar constraints for tool calling and response format.

    This function decides what constraints to apply for grammar-based decoding:
    - `tools` defines the menu of available tools
    - `tool_choice` controls how tools are used (none/auto/required/named)

    The behavior depends on the combination of inputs:
    - tools forced (required or named function): grammar constrains to tool
      calls only, response_format is ignored, enforcement from start,
      no --enable-structured-output flag required
    - auto mode + no response_format: grammar generated for tool calls,
      conditional enforcement (only when tool call start token detected),
      no --enable-structured-output flag required
    - auto mode + response_format: grammar allows either tool calls or JSON
      content matching the schema, enforcement from start,
      --enable-structured-output flag required
    - response_format only (no tools): no architecture-specific grammar is
      generated; the caller falls through to the standard json_schema
      flow handled by StructuredOutputHelper, --enable-structured-output flag required

    Args:
        tools: List of tool definitions from the request.
        tool_choice: The tool_choice value from the request.
        response_format: Response format dict from the request.

    Returns:
        (grammar_tools, response_format_schema, tools_forced, enforce_from_start)
        - grammar_tools: Filtered subset of *tools* for grammar, or None.
        - response_format_schema: JSON schema for response format, or None.
        - tools_forced: True if tool_choice=required or named function.
          Controls whether grammar is enforced from the first token (True)
          or conditionally when a tool call start token is detected (False).
          Independent of the --enable-structured-output flag.
        - enforce_from_start: True if grammar should be enforced from the
          first token. False for auto mode without response_format (conditional
          enforcement - grammar activates when tool call start token detected).
    """
    response_format_schema: dict[str, Any] | None = None

    tools_required = tool_choice == "required"
    tools_auto = tool_choice is None or tool_choice == "auto"

    tool_names = names_from_tools(tools)

    # Narrow to a specific function when tool_choice names one.
    forced_tool_names: list[str] | None = None
    if tools is not None:
        if tools_required:
            forced_tool_names = tool_names
        elif (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and (chosen := maybe_name_from_tool(tool_choice)) is not None
        ):
            forced_tool_names = [chosen]

    # Build the filtered tools list.
    grammar_tools: list[TextGenerationRequestTool] | None = None
    if forced_tool_names is not None:
        grammar_tools = [
            t
            for t in (tools or [])
            if (n := maybe_name_from_tool(t)) is not None
            and n in forced_tool_names
        ] or None
    elif (
        tools_required
        or response_format is not None
        or (tools_auto and tools is not None)
    ):
        grammar_tools = list(tools) if tools else None

    tools_forced = forced_tool_names is not None

    # Only include response_format in grammar when tools aren't forced.
    # When tools are forced, constrain to tool calls only.
    if response_format is not None and not tools_forced:
        if response_format.type == "json_schema":
            response_format_schema = response_format.json_schema or None

    # enforce_from_start: True for required/named OR auto+response_format
    # False for auto without response_format (conditional enforcement)
    enforce_from_start = tools_forced or (
        grammar_tools is not None and response_format is not None
    )

    return (
        grammar_tools,
        response_format_schema,
        tools_forced,
        enforce_from_start,
    )


@router.post("/chat/completions", response_model=None)
async def openai_create_chat_completion(
    request: Request,
) -> CreateChatCompletionResponse | EventSourceResponse | Response:
    request_id = request.state.request_id
    try:
        completion_request = await _parse_openai_request_body(
            request, request_id, CreateChatCompletionRequest
        )
        pipeline = get_pipeline(request, completion_request.model)

        logger.debug(
            "Processing path, %s, req-id,%s%s, for model, %s.",
            request.url.path,
            request_id,
            " (streaming) " if completion_request.stream else "",
            completion_request.model,
        )

        # Model-specific limits (image caps, tool-name charset) are read off the
        # parser and tokenizer so the generic route stays model-agnostic.
        parser = get_tool_parser(request.app)
        tokenizer = pipeline.tokenizer

        (
            request_messages,
            request_images,
            request_videos,
            request_decoded_images,
        ) = await openai_parse_chat_completion_request(
            completion_request,
            tokenizer.expects_content_wrapping,
            request.app.state.settings,
            max_images_per_request=getattr(
                tokenizer, "max_images_per_request", None
            ),
            max_image_bytes=getattr(tokenizer, "max_image_bytes", None),
            allowed_roles=_STANDARD_CHAT_ROLES
            | getattr(tokenizer, "extra_chat_roles", frozenset()),
        )

        pipeline_config = get_app_pipeline_config(request.app)

        # Tool-name charset defaults to OpenAI's; a parser may widen it via VALID_TOOL_NAME_RE.
        valid_tool_name_re = (
            getattr(parser, "VALID_TOOL_NAME_RE", None)
            or _DEFAULT_VALID_TOOL_NAME_RE
        )

        # Unless the user explicitly disabled tools with tool_choice='none', generate the tools list.
        tools = None
        if (
            completion_request.tool_choice is None
            or completion_request.tool_choice != "none"
        ):
            tools = _convert_chat_completion_tools_to_token_generator_tools(
                completion_request.tools, valid_tool_name_re
            )

        response_format = _create_response_format(
            completion_request.response_format,
            enable_response_format_schema=pipeline_config.sampling.enable_structured_output,
        )

        # For architectures with a grammar-based tool parser (e.g., Kimi),
        # generate constrained decoding grammars for tool calls and/or
        # response_format.
        has_grammar_parser = parser is not None and hasattr(
            parser, "generate_tool_call_grammar"
        )
        if has_grammar_parser:
            (
                grammar_tools,
                response_format_schema,
                tools_forced,
                enforce_from_start,
            ) = _resolve_grammar_constraints(
                tools=tools,
                tool_choice=completion_request.tool_choice,
                response_format=response_format,
            )
            # Only invoke the architecture-specific grammar generator when
            # tools are actually involved. In the response_format-only case,
            # fall through to the standard json_schema flow handled by StructuredOutputHelper.
            if grammar_tools:
                assert parser is not None
                logger.debug(
                    "Generating tool call grammar for %s with tools: %s, "
                    "response_format_schema: %s, tools_forced: %s, "
                    "enforce_from_start: %s",
                    type(parser).__name__,
                    names_from_tools(grammar_tools),
                    response_format_schema,
                    tools_forced,
                    enforce_from_start,
                )
                with Tracer("tool_grammar_build"):
                    grammar = parser.generate_tool_call_grammar(  # type: ignore[attr-defined]
                        response_format_schema=response_format_schema,
                        tools=grammar_tools,
                        tokenizer=pipeline.tokenizer,
                        backend=pipeline_config.sampling.structured_output_backend,
                        tool_choice=completion_request.tool_choice,
                    )
                # Create the response format.
                # Note:
                # - tools_forced=True (tool_choice=required or named):
                # - enforce_from_start=True: Grammar enforced from first token.
                # - enforce_from_start=False (auto without response_format):
                #   Conditional enforcement - grammar activates when tool call
                #   start token is detected.
                # ``requires_structured_output_flag`` is True only when the
                # grammar embeds a user-supplied schema. Pure tool-call
                # grammars are server-generated and don't require the flag.
                response_format = TextGenerationResponseFormat(
                    type="grammar",
                    grammar=grammar,
                    json_schema={},
                    grammar_enforced=enforce_from_start,
                    tools_forced=tools_forced,
                    requires_structured_output_flag=response_format_schema
                    is not None,
                    has_json_schema=response_format_schema is not None,
                )
                logger.debug(
                    "Successfully generated tool call grammar (length=%d, "
                    "tools_forced=%s, enforce_from_start=%s)",
                    len(grammar),
                    tools_forced,
                    enforce_from_start,
                )
        stream_options = None
        if completion_request.stream:
            stream_options = completion_request.stream_options
        # Parse tool calls when tools are provided. With combined grammar support,
        # the model can output either tool calls or structured content. The parser
        # will detect which format was used and handle accordingly.
        parse_tool_calls = tools is not None
        # MiniMax ``reasoning_split=False`` folds reasoning back into the
        # ``content`` field wrapped in ``<think>...</think>``. Gated to MiniMax
        # M3 (identified by its reasoning parser) so other models are unaffected.
        fold_reasoning_into_content = (
            completion_request.reasoning_split is False
            and pipeline_config.runtime.reasoning_parser == "minimax_m3"
        )
        response_generator = OpenAIChatResponseGenerator(
            pipeline,
            stream_options=stream_options,
            parser=parser,
            parse_tool_calls=parse_tool_calls,
            tools=tools,
            fold_reasoning_into_content=fold_reasoning_into_content,
        )
        # Use request-level temperature/thinking_temperature if provided, else server defaults.
        temp = (
            completion_request.temperature
            if completion_request.temperature is not None
            else pipeline_config.runtime.temperature
        )
        thinking_temp = (
            completion_request.thinking_temperature
            if completion_request.thinking_temperature is not None
            else pipeline_config.runtime.thinking_temperature
        )
        max_new_tokens = (
            completion_request.max_completion_tokens
            if completion_request.max_completion_tokens is not None
            else completion_request.max_tokens
        )
        sampling_params = SamplingParams.from_input_and_generation_config(
            SamplingParamsInput(
                top_k=completion_request.top_k,
                top_p=completion_request.top_p,
                min_p=completion_request.min_p,
                temperature=temp,
                thinking_temperature=thinking_temp,
                frequency_penalty=completion_request.frequency_penalty,
                presence_penalty=completion_request.presence_penalty,
                repetition_penalty=completion_request.repetition_penalty,
                max_new_tokens=max_new_tokens,
                min_new_tokens=completion_request.min_tokens,
                ignore_eos=completion_request.ignore_eos,
                seed=completion_request.seed or randint(0, 2**63 - 1),
                stop_token_ids=completion_request.stop_token_ids,
                stop=_convert_stop(completion_request.stop),
            ),
            sampling_params_defaults=pipeline_config.model.sampling_params_defaults,
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

        runtime_cfg = pipeline_config.runtime
        if logprobs_count != 0 and runtime_cfg.enable_overlap_scheduler:
            if runtime_cfg.allow_unsupported_logprobs:
                logger.warning(
                    "Request %s asked for logprobs but the overlap scheduler "
                    "is enabled; allow_unsupported_logprobs=True, so the "
                    "request will be served without logprobs.",
                    request_id,
                )
                logprobs_count = 0
            else:
                raise InputError(
                    "Log probabilities are not supported with the overlap"
                    " scheduler. Start the server with"
                    " --no-enable-overlap-scheduler to use logprobs, or"
                    " --allow-unsupported-logprobs to silently ignore the"
                    " field."
                )

        # Map OpenRouter's ``reasoning`` toggle onto the chat-template thinking
        # flags. Templates are inconsistent about the key name, so set both
        # ``enable_thinking`` and ``thinking``.
        if completion_request.reasoning is not None:
            reasoning = completion_request.reasoning
            enable_thinking = (
                reasoning.enabled
                if reasoning.enabled is not None
                else reasoning.effort is not None
            )
            chat_template_kwargs = dict(
                completion_request.chat_template_kwargs or {}
            )
            chat_template_kwargs.setdefault("enable_thinking", enable_thinking)
            chat_template_kwargs.setdefault("thinking", enable_thinking)
            completion_request.chat_template_kwargs = chat_template_kwargs

        # When the orchestrator has already tokenized the prompt for
        # KV cache-aware routing, pass the token IDs directly so MAX Serve
        # skips re-tokenization. ``messages`` and ``prompt`` are mutually
        # exclusive on TextGenerationRequest, so omit ``messages`` in that
        # case. If both are sent on the wire, ``prompt_tokens`` wins.
        prompt_token_ids = completion_request.prompt_tokens
        chat_template_options = completion_request.chat_template_kwargs
        token_request = TextGenerationRequest(
            request_id=RequestID(request_id),
            model_name=completion_request.model,
            prompt=prompt_token_ids if prompt_token_ids else None,
            messages=[] if prompt_token_ids else request_messages,
            images=request_images,
            decoded_images=request_decoded_images,
            videos=request_videos,
            tools=tools,
            timestamp_ns=request.state.request_timer.start_ns,
            request_path=request.url.path,
            response_format=response_format,
            sampling_params=sampling_params,
            logprobs=logprobs_count,
            target_endpoint=_get_target_endpoint(
                request, completion_request.target_endpoint
            ),
            dkv_cache_hint=completion_request.dkv_cache_hint,
            cache_salt=completion_request.cache_salt,
            chat_template_options=chat_template_options,
        )

        if completion_request.stream:
            # We set a large timeout for ping otherwise benchmarking scripts
            # such as sglang will fail in parsing the ping message.
            return EventSourceResponse(
                response_generator.stream(token_request), ping=100000, sep="\n"
            )

        response = await response_generator.complete([token_request])
        return response
    except _ClientDisconnectedError:
        logger.info("Client disconnected for request %s", request_id)
        return Response(status_code=_CLIENT_DISCONNECTED_STATUS_CODE)
    except JSONDecodeError as e:
        logger.exception("JSONDecodeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except KeyError as e:
        logger.exception("KeyError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValidationError as e:
        logger.warning(
            "Request validation error in request %s: %s", request_id, e
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TypeError as e:
        logger.exception("TypeError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except InputError as e:
        logger.warning(
            "Input validation error in request %s: %s", request_id, str(e)
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except UndefinedError as e:
        logger.warning(
            "Chat template UndefinedError in request %s: %s",
            request_id,
            str(e),
        )
        raise HTTPException(
            status_code=400, detail=f"Invalid request: {e}"
        ) from e
    except ValueError as e:
        logger.warning("Value error in request %s: %s", request_id, str(e))
        # NOTE(SI-722): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.") from e


def _convert_chat_completion_tools_to_token_generator_tools(
    chat_tools: Iterable[ChatCompletionFunctionToolParam] | None,
    valid_tool_name_re: re.Pattern[str] = _DEFAULT_VALID_TOOL_NAME_RE,
) -> list[TextGenerationRequestTool] | None:
    """Convert ChatCompletionTool list to TextGenerationRequestTool list."""
    if not chat_tools:
        return None

    token_generator_tools = []
    for tool in chat_tools:
        function = tool["function"]
        name = name_from_tool(tool)
        _validate_tool_function_name(name, valid_tool_name_re)
        token_generator_tool = TextGenerationRequestTool(
            type=tool["type"],
            function=TextGenerationRequestFunction(
                name=name,
                description=function.get("description"),
                parameters=dict(function.get("parameters") or {}),
            ),
        )
        token_generator_tools.append(token_generator_tool)

    return token_generator_tools


def _validate_tool_function_name(
    name: str,
    valid_tool_name_re: re.Pattern[str] = _DEFAULT_VALID_TOOL_NAME_RE,
) -> None:
    """Validate that a tool function name conforms to ``valid_tool_name_re``.

    Raises:
        InputError: If the name is empty, too long, or contains invalid
            characters.
    """
    if not name:
        raise InputError("Invalid tool function name: name cannot be empty.")
    if len(name) > _MAX_TOOL_NAME_LEN:
        raise InputError(
            f"Invalid tool function name: name exceeds the maximum length of "
            f"{_MAX_TOOL_NAME_LEN} characters (was {len(name)})."
        )
    if not valid_tool_name_re.match(name):
        raise InputError(
            f"Invalid tool function name: '{name}'. "
            f"Function names must match {valid_tool_name_re.pattern}."
        )


def _validate_json_schema(json_schema: dict[str, Any]) -> None:
    """Validate that a JSON schema can be compiled to a grammar.

    This catches invalid schemas (recursive $ref, unsupported constructs)
    early in the HTTP request handler, returning a 400 error instead of
    crashing the model worker process later during constrained decoding.

    Raises:
        InputError: If the schema cannot be compiled or has a non-object root.
    """
    if not json_schema:
        return

    # Root must be type: object per OpenAI's structured-outputs guide.
    try:
        _validate_response_format_schema(json_schema)
    except ValueError as e:
        raise InputError(str(e)) from e

    try:
        # This validates the schema can be compiled to a grammar.
        # It doesn't need a tokenizer - just checks schema structure.
        LLMatcher.grammar_from_json_schema(json_schema)
    except Exception as e:
        raise InputError(
            f"JSON schema cannot be compiled to valid grammar: {e}. "
            "Recursive $ref schemas and other unsupported constructs are not allowed."
        ) from e


def _create_response_format(
    response_format: ResponseFormatText
    | ResponseFormatJsonObject
    | ResponseFormatJsonSchema
    | None,
    enable_response_format_schema: bool,
) -> TextGenerationResponseFormat | None:
    """Convert OpenAI response format to TextGenerationResponseFormat.

    Raises:
        InputError: If ``response_format`` is ``json_schema`` or
            ``json_object`` but ``enable_response_format_schema`` is False.
            Reject at the route boundary so the scheduler worker never sees
            a request that would crash it from inside ``execute()``.
    """
    if not response_format:
        return None

    # ``response_format`` is an OpenAI TypedDict, accessed via keys.
    response_type = response_format["type"]
    if response_type in ("json_schema", "json_object") and (
        not enable_response_format_schema
    ):
        raise InputError(
            "response_format requires --enable-structured-output. Restart "
            "the server with --enable-structured-output to allow "
            "schema-constrained responses."
        )

    json_schema: dict[Any, Any] = {}

    if response_type == "json_object":
        # For json_object mode (any valid JSON), use a permissive schema that
        # accepts any JSON object. llguidance's grammar_from_json_schema supports
        # this - an empty or minimal schema means "any valid JSON".
        json_schema = {"type": "object"}
        # Normalize type to json_schema for the internal representation since both
        # json_object and json_schema use grammar-based constrained decoding.
        response_type = "json_schema"
    elif response_type == "json_schema":
        # ``response_format`` is one of OpenAI's ``ResponseFormat*Param``
        # TypedDicts; cast to ``dict`` so mypy lets us key into it without
        # narrowing the discriminated union by hand.
        json_schema_param = cast(dict[str, Any], response_format).get(
            "json_schema", {}
        )
        if (schema := json_schema_param.get("schema")) is not None:
            json_schema = dict(schema)

    # Validate the schema early to return 400 instead of crashing the model worker.
    _validate_json_schema(json_schema)

    # Default a missing root ``type`` to ``"object"`` before the schema
    # reaches the grammar backend. An untyped root compiles to a grammar that
    # permits a bare unbounded top-level value, which lets a looping model run
    # to ``max_length`` (the runaway-output incident).
    json_schema = normalize_response_format_schema(json_schema)

    # Enforce grammar from the first token only when there is an actual
    # schema to enforce. The json_schema can also be used to create a grammar,
    # hence we need to specify that the grammar should constrain from the
    # start if there's a json_schema present here.
    # TODO: improve the field naming here; grammar_enforced should be constrain_with_bitmask.
    return TextGenerationResponseFormat(
        type=response_type,
        json_schema=json_schema,
        grammar=None,
        grammar_enforced=bool(json_schema),
        tools_forced=False,
        requires_structured_output_flag=True,
        has_json_schema=bool(json_schema),
    )


@router.post("/embeddings", response_model=None)
async def openai_create_embeddings(
    request: Request,
) -> CreateEmbeddingResponse | Response:
    request_id = request.state.request_id

    # First try-catch: request parsing (client fault → 400)
    try:
        embeddings_request = CreateEmbeddingRequest.model_validate_json(
            await request.body()
        )
        pipeline = get_pipeline(request, embeddings_request.model)

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
        # ``encode`` requires at least one entry; this matches the OpenAI
        # behavior of rejecting empty ``input`` arrays.

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
    except _ClientDisconnectedError:
        logger.info("Client disconnected for request %s", request_id)
        return Response(status_code=_CLIENT_DISCONNECTED_STATUS_CODE)
    except JSONDecodeError as e:
        logger.warning("JSONDecodeError in request %s: %s", request_id, e)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except KeyError as e:
        logger.warning("KeyError in request %s: %s", request_id, e)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValidationError as e:
        logger.warning(
            "Request validation error in request %s: %s", request_id, e
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TypeError as e:
        logger.warning("TypeError in request %s: %s", request_id, e)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except InputError as e:
        logger.warning(
            "Input validation error in request %s: %s", request_id, str(e)
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        logger.warning("Value error in request %s: %s", request_id, str(e))
        raise HTTPException(status_code=400, detail="Value error.") from e

    # Second try-catch: response generation (server fault → 500)
    try:
        response = await response_generator.encode(embedding_requests)
        return response
    except _ClientDisconnectedError:
        logger.info("Client disconnected for request %s", request_id)
        return Response(status_code=_CLIENT_DISCONNECTED_STATUS_CODE)
    except Exception as e:
        logger.exception(
            "Exception during response generation in request %s", request_id
        )
        raise HTTPException(
            status_code=500, detail="Internal server error."
        ) from e


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
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
) -> CompletionLogprobs:
    token_log_probabilities = []
    top_log_probabilities = []
    for output in token_generator_outputs:
        if output.token_log_probabilities:
            token_log_probabilities.extend(output.token_log_probabilities)
        if output.top_log_probabilities:
            top_log_probabilities.extend(output.top_log_probabilities)

    return CompletionLogprobs(
        token_logprobs=token_log_probabilities,
        top_logprobs=top_log_probabilities,
    )


def _process_chat_log_probabilities(
    token_generator_outputs: list[TokenGeneratorOutput],
) -> ChatCompletionLogprobs:
    """Convert token generator outputs to chat completion log probabilities format.

    Args:
        token_generator_outputs: List of token generator outputs containing
            log probability information.

    Returns:
        ChatCompletionLogprobs object with content tokens and their log
        probabilities.
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

    return ChatCompletionLogprobs(content=content, refusal=[])


def get_app_pipeline_config(app: FastAPI) -> PipelineConfig:
    pipeline_config = app.state.pipeline_config
    assert isinstance(pipeline_config, PipelineConfig)
    return pipeline_config


_TRequest = TypeVar("_TRequest", bound="BaseModel")


async def _parse_openai_request_body(
    request: Request,
    request_id: str,
    model_cls: type[_TRequest],
) -> _TRequest:
    """Parse a JSON request body into a pydantic request model.

    Pre-normalizes ``tools[*].function.parameters: null`` to ``{}`` to
    match OpenAI's API behavior (null parameters is treated as omitted).
    Without this, Pydantic rejects the request before our handler runs.

    Honors ``pipeline_config.runtime.allow_extra_request_fields``: when set,
    unknown top-level fields are dropped (with a warning) before validation
    instead of failing pydantic's ``extra="forbid"`` check.
    """
    raw = await request.body()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        return model_cls.model_validate(parsed)

    # Pre-normalize tools.parameters: null -> {} before Pydantic validation.
    tools = parsed.get("tools")
    if isinstance(tools, list):
        parsed = {**parsed, "tools": _normalize_tools_parameters(tools)}

    pipeline_config = get_app_pipeline_config(request.app)
    if pipeline_config.runtime.allow_extra_request_fields:
        known = set(model_cls.model_fields)
        extras = [k for k in parsed if k not in known]
        if extras:
            logger.warning(
                "Request %s contained unknown top-level fields %s; dropping "
                "(allow_extra_request_fields=True).",
                request_id,
                extras,
            )
            parsed = {k: v for k, v in parsed.items() if k in known}
    return model_cls.model_validate(parsed)


def get_tool_parser(app: FastAPI) -> ToolParser | None:
    """Gets the configured tool parser for the current model.

    Returns the runtime-configured parser if set, otherwise None.
    """
    pipeline_config = get_app_pipeline_config(app)
    parser_name = pipeline_config.runtime.tool_parser
    if parser_name is None:
        return None
    return create_tool_parser(parser_name)


class OpenAICompletionResponseGenerator(
    OpenAIResponseGenerator[CreateCompletionResponse]
):
    def __init__(
        self,
        pipeline: TokenGeneratorPipeline,
        stream_options: ChatCompletionStreamOptionsParam | None = None,
    ) -> None:
        super().__init__(pipeline)
        self.stream_options = stream_options

    async def stream(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[str | ErrorResponse | JSONResponse, None]:
        logger.debug("Streaming: Start: %s", request)
        record_request_start()
        request_timer = StopWatch(start_ns=request.timestamp_ns)
        n_reasoning_tokens = 0
        n_tokens = 0
        n_prompt_tokens = 0
        n_cached_prompt_tokens = 0
        status_code = 200
        try:
            async for chunk in self.pipeline.next_token_chunk(request):
                chunk_total_tokens = (
                    chunk.reasoning_token_count or 0
                ) + chunk.token_count
                self.logger.debug(
                    "Streaming: %s, TOKENS: %d, REASONING: %s, TEXT: %s",
                    request.request_id,
                    chunk_total_tokens,
                    chunk.decoded_reasoning_tokens,
                    chunk.decoded_tokens,
                )

                if chunk.prompt_token_count:
                    n_prompt_tokens = chunk.prompt_token_count
                if chunk.cached_token_count is not None:
                    n_cached_prompt_tokens = chunk.cached_token_count
                n_reasoning_tokens += chunk.reasoning_token_count or 0
                n_tokens += chunk.token_count

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
                elif chunk.status.is_done:
                    choices = [
                        CompletionResponseStreamChoice(
                            index=0,
                            text="",
                            finish_reason=get_finish_reason_from_status(
                                chunk.status, allow_none=False
                            ),
                        )
                    ]
                else:
                    # Reasoning-capable models (e.g. Kimi K2.5) can emit
                    # intermediate chunks with no user-visible completion text
                    # while still ACTIVE. For legacy /completions streaming we
                    # skip those chunks instead of forcing a terminal
                    # finish_reason.
                    continue

                # Each chunk is expected to have the same id
                # https://platform.openai.com/docs/api-reference/chat/streaming
                response = CompletionStreamResponse(
                    id=request.request_id.value,
                    choices=choices,
                    created=int(datetime.now().timestamp()),
                    model=request.model_name,
                    object="text_completion",
                )

                payload = response.model_dump_json()

                yield payload

            logger.debug(
                "Streaming: Done: %s, %d tokens",
                request,
                n_reasoning_tokens + n_tokens,
            )

            # If `include_usage=True`, send a final chunk with usage statistics.
            # https://platform.openai.com/docs/api-reference/completions/create#completions_create-stream_options
            if self.stream_options and self.stream_options.get("include_usage"):
                final_usage = CompletionUsage(
                    prompt_tokens=n_prompt_tokens,
                    completion_tokens=n_reasoning_tokens + n_tokens,
                    total_tokens=n_prompt_tokens
                    + n_reasoning_tokens
                    + n_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=n_cached_prompt_tokens,
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=n_reasoning_tokens,
                    ),
                )
                final_response = CompletionStreamResponse(
                    id=request.request_id.value,
                    choices=[],
                    created=int(datetime.now().timestamp()),
                    model=request.model_name,
                    object="text_completion",
                    usage=final_usage,
                )
                yield final_response.model_dump_json()

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
            logger.exception("Exception in request %s", request.request_id)
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
                n_reasoning_tokens + n_tokens,
                n_prompt_tokens,
            )

    async def complete(
        self, requests: list[TextGenerationRequest]
    ) -> CreateCompletionResponse:
        # we assume that all entries in `requests` came from the same http
        # request and timestamp, request id, path should all be the same.
        record_request_start()
        n_reasoning_tokens = 0
        n_tokens = 0
        n_prompt_tokens = 0
        n_cached_prompt_tokens = 0
        request_timer = StopWatch(start_ns=requests[0].timestamp_ns)
        status_code = 200

        try:
            req_output_list = await asyncio.gather(
                *[self.pipeline.all_tokens(request) for request in requests]
            )
            response_choices = []
            for i, req_outputs in enumerate(req_output_list):
                n_reasoning_tokens += sum(
                    chunk.reasoning_token_count or 0 for chunk in req_outputs
                )
                n_tokens += sum(chunk.token_count for chunk in req_outputs)
                if req_outputs and req_outputs[0].prompt_token_count:
                    n_prompt_tokens += req_outputs[0].prompt_token_count
                if (
                    req_outputs
                    and req_outputs[0].cached_token_count is not None
                ):
                    n_cached_prompt_tokens += req_outputs[0].cached_token_count

                log_probs = _process_log_probabilities(req_outputs)
                response_message = "".join(
                    chunk.decoded_tokens
                    if chunk.decoded_tokens is not None
                    else ""
                    for chunk in req_outputs
                )
                response_choices.append(
                    CompletionResponseChoice(
                        index=i,
                        text=response_message,
                        finish_reason=get_finish_reason_from_status(
                            req_outputs[-1].status, allow_none=False
                        ),
                        logprobs=log_probs,
                    )
                )
            usage = CompletionUsage(
                prompt_tokens=n_prompt_tokens,
                completion_tokens=n_reasoning_tokens + n_tokens,
                total_tokens=n_prompt_tokens + n_reasoning_tokens + n_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=n_cached_prompt_tokens,
                ),
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=n_reasoning_tokens,
                ),
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
                usage=usage,
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
                n_reasoning_tokens + n_tokens,
                n_prompt_tokens,
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
    prompt: str | list[str] | list[int] | list[list[int]],
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
    if _is_sequence_of(prompt, int):
        return [prompt]
    if _is_seq_of_seq_of_int(prompt):
        return prompt
    raise Exception(f"unknown element type {type(prompt[0])}")


@router.post("/completions", response_model=None)
async def openai_create_completion(
    request: Request,
) -> CreateCompletionResponse | EventSourceResponse | Response:
    """
    Legacy OpenAI /completion endpoint.
    https://platform.openai.com/docs/api-reference/completions
    Public benchmarking such as vLLM use this endpoint.
    """
    http_req_id = request.state.request_id
    try:
        completion_request = await _parse_openai_request_body(
            request, http_req_id, CreateCompletionRequest
        )

        pipeline = get_pipeline(request, completion_request.model)

        logger.debug(
            "Path: %s, Request: %s%s, Model: %s",
            request.url.path,
            http_req_id,
            " (streaming) " if completion_request.stream else "",
            completion_request.model,
        )

        pipeline_config = get_app_pipeline_config(request.app)

        if (
            completion_request.logprobs is not None
            and completion_request.logprobs != 0
            and pipeline_config.runtime.enable_overlap_scheduler
        ):
            if pipeline_config.runtime.allow_unsupported_logprobs:
                logger.warning(
                    "Request %s asked for logprobs but the overlap scheduler "
                    "is enabled; allow_unsupported_logprobs=True, so the "
                    "request will be served without logprobs.",
                    http_req_id,
                )
                completion_request.logprobs = None
            else:
                raise InputError(
                    "Log probabilities are not supported with the overlap"
                    " scheduler. Start the server with"
                    " --no-enable-overlap-scheduler to use logprobs, or"
                    " --allow-unsupported-logprobs to silently ignore the"
                    " field."
                )

        response_generator = OpenAICompletionResponseGenerator(
            pipeline, stream_options=completion_request.stream_options
        )
        prompts = get_prompts_from_openai_request(completion_request.prompt)
        token_requests = []
        # Use request-level temperature/thinking_temperature if provided, else server defaults.
        temp = (
            completion_request.temperature
            if completion_request.temperature is not None
            else pipeline_config.runtime.temperature
        )
        thinking_temp = (
            completion_request.thinking_temperature
            if completion_request.thinking_temperature is not None
            else pipeline_config.runtime.thinking_temperature
        )
        for i, prompt in enumerate(prompts):
            prompt = cast(str | Sequence[int], prompt)
            sampling_params = SamplingParams.from_input_and_generation_config(
                SamplingParamsInput(
                    top_k=completion_request.top_k,
                    top_p=completion_request.top_p,
                    min_p=completion_request.min_p,
                    temperature=temp,
                    thinking_temperature=thinking_temp,
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
                sampling_params_defaults=pipeline_config.model.sampling_params_defaults,
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
                dkv_cache_hint=completion_request.dkv_cache_hint,
                cache_salt=completion_request.cache_salt,
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
    except _ClientDisconnectedError:
        logger.info("Client disconnected for request %s", http_req_id)
        return Response(status_code=_CLIENT_DISCONNECTED_STATUS_CODE)
    except JSONDecodeError as e:
        logger.exception("JSONDecodeError for request %s", http_req_id)
        raise HTTPException(status_code=400, detail="Missing JSON.") from e
    except KeyError as e:
        logger.exception("KeyError in request %s", http_req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValidationError as e:
        logger.warning(
            "Request validation error for request %s: %s", http_req_id, e
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TypeError as e:
        logger.exception("Validation error for request %s", http_req_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except InputError as e:
        logger.warning(
            "Input validation error in request %s: %s", http_req_id, str(e)
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        logger.warning("Value error in request %s: %s", http_req_id, str(e))
        # NOTE(SI-722): These errors need to return more helpful details,
        # but we don't necessarily want to expose the full error description
        # to the user. There are many different ValueErrors that can be raised.
        raise HTTPException(status_code=400, detail="Value error.") from e


@router.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


def _resolve_max_model_len(request: Request) -> int | None:
    """Resolve the served model's max context length.

    Returns the smallest length the tokenizer and model can handle, so clients
    can avoid overflowing the model's context.
    """
    max_model_len = get_app_pipeline_config(request.app).model.max_length
    if max_model_len is None:
        return None

    tokenizer_max = getattr(
        request.app.state.pipeline.tokenizer, "max_length", None
    )
    if isinstance(tokenizer_max, int):
        max_model_len = min(max_model_len, tokenizer_max)

    return max_model_len


@router.get("/models", response_model=None)
async def openai_get_models(request: Request) -> ListModelsResponse:
    pipeline: TokenGeneratorPipeline = request.app.state.pipeline
    created = int(datetime.now().timestamp())
    max_model_len = _resolve_max_model_len(request)
    model_list = [
        MaxModel(
            id=pipeline.model_name,
            object="model",
            created=created,
            owned_by="",
            max_model_len=max_model_len,
        )
    ]

    if lora_queue := request.app.state.pipeline.lora_queue:
        model_list += [
            MaxModel(
                id=lora,
                object="model",
                created=created,
                owned_by="",
                max_model_len=max_model_len,
            )
            for lora in lora_queue.list_loras()
        ]

    return ListModelsResponse(object="list", data=model_list)


@router.get("/models/{model_id}", response_model=None)
async def openai_get_model(model_id: str, request: Request) -> Model:
    pipeline: TokenGeneratorPipeline = request.app.state.pipeline
    pipeline_model = MaxModel(
        id=pipeline.model_name,
        object="model",
        created=int(datetime.now().timestamp()),
        owned_by="",
        max_model_len=_resolve_max_model_len(request),
    )

    if model_id == pipeline.model_name:
        return pipeline_model

    # We need to handle the slash in our model names (not an issue for OpenAI)
    slash_ind = pipeline.model_name.rfind("/")
    if slash_ind != -1 and model_id == pipeline.model_name[slash_ind + 1 :]:
        return pipeline_model

    raise HTTPException(status_code=404)


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
    except KeyError as e:
        logger.exception("KeyError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValidationError as e:
        logger.warning(
            "Request validation error in request %s: %s", request_id, e
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TypeError as e:
        logger.exception("Validation error in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValueError as e:
        logger.warning("Value error in request %s: %s", request_id, str(e))
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
    except KeyError as e:
        logger.exception("KeyError in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValidationError as e:
        logger.warning(
            "Request validation error in request %s: %s", request_id, e
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TypeError as e:
        logger.exception("Validation error in request %s", request_id)
        raise HTTPException(status_code=400, detail="Invalid JSON.") from e
    except ValueError as e:
        logger.warning("Value error in request %s: %s", request_id, str(e))
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
