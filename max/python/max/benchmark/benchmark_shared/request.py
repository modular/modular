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

"""Request-related data structures for benchmarking."""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import aiohttp
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import NotRequired, TypedDict

from .config import PIXEL_GENERATION_TASKS, BenchmarkTask, SamplingConfig
from .datasets.types import (
    ChatMessage,
    OpenAIImage,
    PixelGenerationImageOptions,
)
from .sse import iter_events
from .tts_workloads_utils import SampleTTSRequest

# 30 minute timeout per request session
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=30 * 60)

logger = logging.getLogger(__name__)


def _encode_openresponses_image_from_file_path(
    file_path: str,
) -> dict[str, str]:
    extension_to_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    _, ext = os.path.splitext(file_path.lower())
    if ext not in extension_to_mime:
        supported_exts = ", ".join(extension_to_mime.keys())
        raise ValueError(
            f"Unsupported image file extension '{ext}'. "
            f"Supported extensions: {supported_exts}"
        )

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:{extension_to_mime[ext]};base64,{img_base64}",
    }


@dataclass
class BaseRequestFuncInput(ABC):
    """Base class for request function input with common fields."""

    model: str
    session_id: str | None

    @abstractmethod
    def get_output_type(self) -> type[BaseRequestFuncOutput]:
        """Get the output type for the request function input."""


def _apply_sampling_to_request_payload(
    payload: dict[str, Any], sampling: SamplingConfig
) -> None:
    """Merge non-None OpenAI-style sampling fields from *sampling* into *payload*."""
    if sampling.temperature is not None:
        payload["temperature"] = sampling.temperature
    if sampling.thinking_temperature is not None:
        payload["thinking_temperature"] = sampling.thinking_temperature
    if sampling.top_k is not None:
        payload["top_k"] = sampling.top_k
    if sampling.top_p is not None:
        payload["top_p"] = sampling.top_p


# TODO: We shouldn't have to maintain two separate RequestFuncInput classes for
# text generation and TTS benchmarks respectively.
@dataclass
class RequestFuncInput(BaseRequestFuncInput):
    """Request function input for text generation benchmarks."""

    sampling: SamplingConfig
    prompt: str | list[ChatMessage]
    images: list[OpenAIImage]
    api_url: str
    prompt_len: int
    max_tokens: int | None
    ignore_eos: bool
    response_format: ResponseFormat | None = None

    def get_output_type(self) -> type[BaseRequestFuncOutput]:
        return RequestFuncOutput


@dataclass
class PixelGenerationRequestFuncInput(BaseRequestFuncInput):
    """Request function input for pixel-generation benchmarks."""

    prompt: str
    input_image_paths: list[str] | None
    api_url: str
    image_options: PixelGenerationImageOptions | None = None

    def get_output_type(self) -> type[BaseRequestFuncOutput]:
        return PixelGenerationRequestFuncOutput


@dataclass
class TTSRequestFuncInput(BaseRequestFuncInput):
    """Request function input for TTS (text-to-speech) benchmarks."""

    sampling: SamplingConfig
    request_index: int
    tts_request: SampleTTSRequest
    is_streaming_mode: bool
    frequency_penalty: float
    repetition_penalty: float
    seed: int = 0

    def get_output_type(self) -> type[BaseRequestFuncOutput]:
        return TTSRequestFuncOutput


@dataclass
class BaseRequestFuncOutput:
    """Base class for request function output with common fields."""

    cancelled: bool = False
    success: bool = False
    latency: float = 0.0
    error: str = ""
    # time.perf_counter() at request dispatch (monotonic, run-relative)
    request_submit_time: float | None = None

    @property
    def request_complete_time(self) -> float | None:
        """Derived completion timestamp: submit time + latency."""
        if self.request_submit_time is None:
            return None
        return self.request_submit_time + self.latency


def measured_window_duration(
    outputs: Iterable[BaseRequestFuncOutput], fallback: float
) -> float:
    """Wall-clock seconds from the first submit to the last complete.

    The window covers only requests with both a ``request_submit_time`` and a
    ``request_complete_time``. If no such request exists, return ``fallback``.
    Otherwise return ``max(last_complete - first_submit, 1e-9)`` so callers
    can safely divide.

    This is the same window math the steady-state block uses and is the
    correct denominator for aggregate throughput / TPM over a sliced benchmark
    region — warmup/tail wall time is excluded along with warmup/tail tokens.
    """
    first_submit: float | None = None
    last_complete: float | None = None
    for o in outputs:
        submit = o.request_submit_time
        if submit is None:
            continue
        complete = o.request_complete_time
        if complete is None:
            continue
        if first_submit is None or submit < first_submit:
            first_submit = submit
        if last_complete is None or complete > last_complete:
            last_complete = complete
    if first_submit is None or last_complete is None:
        return fallback
    return max(last_complete - first_submit, 1e-9)


@dataclass
class ServerTokenStats:
    """Server-reported token counts from the stream_options usage chunk."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int = 0


# TODO: We shouldn't have to maintain two separate RequestFuncOutput classes for
# text generation and TTS benchmarks respectively.
@dataclass
class RequestFuncOutput(BaseRequestFuncOutput):
    """Request function output for text generation benchmarks."""

    # List of inter-token latencies.
    itl: list[float] = field(default_factory=list)
    # List of per-chunk time-per-output-token values.
    tpot: list[float] = field(default_factory=list)
    generated_text: str = ""
    ttft: float = 0.0  # Time to first token
    prompt_len: int = 0
    server_token_stats: ServerTokenStats = field(
        default_factory=ServerTokenStats
    )


@dataclass
class PixelGenerationRequestFuncOutput(BaseRequestFuncOutput):
    """Request function output for text-to-image benchmarks."""

    num_generated_outputs: int = 0


@dataclass
class TTSRequestFuncOutput(BaseRequestFuncOutput):
    """Request function output for TTS (text-to-speech) benchmarks."""

    request_index: int = 0
    itl: list[float] = field(default_factory=list)
    tpot: list[float] = field(default_factory=list)
    # TODO: We have a torch.Tensor dependency here, but our benchmark_shared
    # package doesn't "require" torch. For better or worse, this is only used
    # in the TTS benchmarks, so we'll leave it as Any for now.
    generated_chunk: list[Any] = field(
        default_factory=list
    )  # list[torch.Tensor]
    ttft: float | None = None  # Time to first token (can be None for TTS)

    def get_chunk_lens_in_samples(self) -> list[int]:
        """Get lengths of audio chunks in samples."""
        return [x.shape[-1] for x in self.generated_chunk]

    def get_chunk_lens_in_seconds(self, tts_config: Any) -> list[float]:
        """Get lengths of audio chunks in seconds.

        Args:
            tts_config: TTS configuration object with decoder_sample_rate attribute.
        """
        lens_in_samples = self.get_chunk_lens_in_samples()
        return [samples_to_seconds(tts_config, x) for x in lens_in_samples]

    def get_chunk_lens_in_tokens(self, tts_config: Any) -> list[int]:
        """Get lengths of audio chunks in tokens.

        Args:
            tts_config: TTS configuration object with codec_tokens_per_sec attribute.
        """
        lens_in_samples = self.get_chunk_lens_in_samples()
        return [samples_to_tokens(tts_config, x) for x in lens_in_samples]

    def get_real_time_factors(self, tts_config: Any) -> list[float]:
        """Calculate real-time factors (RTF).

        RTF is the inter-chunk latency divided by the playback time of the
        previous chunk. Anything over 100% would lead to a playback error.

        Args:
            tts_config: TTS configuration object.
        """
        lens_in_seconds = self.get_chunk_lens_in_seconds(tts_config)
        assert len(lens_in_seconds) == len(self.itl) + 1, (
            "Missing or extra ITLs?"
        )
        return [
            x / y for x, y in zip(self.itl, lens_in_seconds[:-1], strict=True)
        ]

    def get_output_length_in_samples(self) -> int:
        """Get total output length in samples."""
        return sum(self.get_chunk_lens_in_samples())

    def get_output_length_in_seconds(self, tts_config: Any) -> float:
        """Get total output length in seconds.

        Args:
            tts_config: TTS configuration object.
        """
        return sum(self.get_chunk_lens_in_seconds(tts_config))

    def get_output_length_in_tokens(self, tts_config: Any) -> int:
        """Get total output length in tokens.

        Args:
            tts_config: TTS configuration object.
        """
        return sum(self.get_chunk_lens_in_tokens(tts_config))


def samples_to_seconds(tts_config: Any, num_samples: int) -> float:
    """Convert number of samples to seconds.

    Args:
        tts_config: TTS configuration object with decoder_sample_rate attribute.
        num_samples: Number of audio samples.
    """
    return num_samples / tts_config.decoder_sample_rate


def samples_to_tokens(tts_config: Any, num_samples: int) -> int:
    """Convert number of samples to tokens.

    Args:
        tts_config: TTS configuration object with decoder_sample_rate and
                   codec_tokens_per_sec attributes.
        num_samples: Number of audio samples.
    """
    playback_time = samples_to_seconds(tts_config, num_samples)
    return math.ceil(playback_time * tts_config.codec_tokens_per_sec)


class RequestDriver(ABC):
    """Abstract base class for a driver that handles API requests to different backends."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase | None = None
    ) -> None:
        """Initialize the request driver.

        Args:
            tokenizer: Optional tokenizer for per-chunk TPOT computation.
        """
        self.tokenizer = tokenizer

    @abstractmethod
    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> BaseRequestFuncOutput:
        """Execute a request to the backend API.

        Args:
            request_func_input: Input parameters for the request.

        Returns:
            RequestFuncOutput containing the response data and metrics.
        """
        pass


class ProgressBarRequestDriver(RequestDriver):
    """Request driver that updates a progress bar after each request."""

    def __init__(
        self,
        request_driver: RequestDriver,
        pbar: tqdm,
    ) -> None:
        """Initialize the progress bar request driver.

        Args:
            request_driver: The underlying request driver to wrap.
            pbar: Progress bar to update after each request completes.
        """
        super().__init__(tokenizer=request_driver.tokenizer)
        self.request_driver = request_driver
        self.pbar = pbar

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> BaseRequestFuncOutput:
        """Execute a request to the backend API.

        Args:
            request_func_input: Input parameters for the request.

        Returns:
            RequestFuncOutput containing the response data and metrics.
        """
        result = await self.request_driver.request(request_func_input)
        self.pbar.update(1)
        return result


class TRTLLMRequestDriver(RequestDriver):
    """Request driver for TensorRT-LLM backend."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the TensorRT-LLM backend."""
        if not isinstance(request_func_input, RequestFuncInput):
            raise TypeError("TRTLLMRequestDriver requires RequestFuncInput.")
        api_url = request_func_input.api_url
        assert api_url.endswith("generate_stream")

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            payload: dict[str, bool | str | int | float | list[ChatMessage]] = {
                "text_input": request_func_input.prompt,
                "ignore_eos": request_func_input.ignore_eos,
                "stream": True,
            }

            if request_func_input.max_tokens is not None:
                payload["max_tokens"] = request_func_input.max_tokens
            _apply_sampling_to_request_payload(
                payload, request_func_input.sampling
            )

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            ttft = 0.0
            st = time.perf_counter()
            output.request_submit_time = st
            most_recent_timestamp = st
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for event in iter_events(response.content):
                            data = _TRTLLMChunk.model_validate_json(event.data)
                            chunk_text = data.text_output
                            output.generated_text += chunk_text
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                itl_value = timestamp - most_recent_timestamp
                                output.itl.append(itl_value)
                                tpot = _compute_chunk_tpot(
                                    self.tokenizer, chunk_text, itl_value
                                )
                                if tpot is not None:
                                    output.tpot.append(tpot)

                            most_recent_timestamp = timestamp

                        output.latency = most_recent_timestamp - st
                        output.success = True

                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

            return output


def _compute_chunk_tpot(
    tokenizer: PreTrainedTokenizerBase | None,
    chunk_text: str,
    itl_value: float,
) -> float | None:
    """Compute per-chunk time-per-output-token.

    Note: This is approximate. Re-tokenizing the chunk text may not exactly
    match the server's tokenization of the generated output.
    """
    if tokenizer is None or not chunk_text:
        return None
    chunk_tokens = len(
        tokenizer(chunk_text, add_special_tokens=False).input_ids
    )
    if chunk_tokens > 0:
        return itl_value / chunk_tokens
    return None


class _PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class _UsageChunk(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_details: _PromptTokensDetails | None = None


class _ChatDelta(BaseModel):
    reasoning: str | None = None
    reasoning_content: str | None = None
    content: str | None = None


class _ChatChoice(BaseModel):
    delta: _ChatDelta


class _CompletionChoice(BaseModel):
    text: str


class _ChatCompletionChunk(BaseModel):
    usage: _UsageChunk | None = None
    choices: list[_ChatChoice] = []


class _CompletionChunk(BaseModel):
    usage: _UsageChunk | None = None
    choices: list[_CompletionChoice] = []


class _TRTLLMChunk(BaseModel):
    text_output: str


_ChunkT = TypeVar("_ChunkT", _ChatCompletionChunk, _CompletionChunk)


async def _run_openai_stream_request(
    *,
    api_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    prompt_len: int,
    chunk_type: type[_ChunkT],
    content_extractor: Callable[[_ChunkT], str],
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.prompt_len = prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    output.request_submit_time = st
    most_recent_timestamp = st
    has_content = False
    latency = 0.0

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for event in iter_events(response.content):
                        latency = time.perf_counter() - st
                        if event.data == "[DONE]":
                            continue

                        data = chunk_type.model_validate_json(event.data)

                        # Parse usage from any chunk that reports it.
                        if data.usage:
                            output.server_token_stats = ServerTokenStats(
                                prompt_tokens=data.usage.prompt_tokens,
                                completion_tokens=data.usage.completion_tokens,
                                total_tokens=data.usage.total_tokens,
                                cached_tokens=(
                                    data.usage.prompt_tokens_details.cached_tokens
                                    if data.usage.prompt_tokens_details
                                    else 0
                                ),
                            )

                        # Skip content processing for chunks with no choices.
                        if not data.choices:
                            continue

                        # Any valid response chunk counts as having received content
                        has_content = True

                        # Only track timing for chunks with actual text
                        text_content = content_extractor(data)
                        if text_content:
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft

                            # Decoding phase
                            else:
                                itl_value = timestamp - most_recent_timestamp
                                output.itl.append(itl_value)
                                tpot = _compute_chunk_tpot(
                                    tokenizer, text_content, itl_value
                                )
                                if tpot is not None:
                                    output.tpot.append(tpot)

                            most_recent_timestamp = timestamp
                            generated_text += text_content
                    if not has_content:
                        output.error = (
                            "No content returned, there could be an issue with"
                            " accuracy"
                        )
                        output.success = False
                    else:
                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
    return output


class OpenAICompletionsRequestDriver(RequestDriver):
    """Request driver for OpenAI-compatible completions API."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the OpenAI-compatible completions API."""
        if not isinstance(request_func_input, RequestFuncInput):
            raise TypeError(
                "OpenAICompletionsRequestDriver requires RequestFuncInput."
            )
        api_url = request_func_input.api_url
        assert api_url.endswith(("completions", "profile")), (
            "OpenAI Completions API URL must end with 'completions' or 'profile'."
        )

        payload: dict[str, bool | str | int | float | list[ChatMessage]] = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "best_of": 1,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens
        _apply_sampling_to_request_payload(payload, request_func_input.sampling)

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        return await _run_openai_stream_request(
            api_url=api_url,
            payload=payload,
            headers=headers,
            prompt_len=request_func_input.prompt_len,
            chunk_type=_CompletionChunk,
            content_extractor=lambda data: data.choices[0].text,
            tokenizer=self.tokenizer,
        )


class OpenAIChatCompletionsRequestDriver(RequestDriver):
    """Request driver for OpenAI-compatible chat completions API."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the OpenAI-compatible chat completions API."""
        if not isinstance(request_func_input, RequestFuncInput):
            raise TypeError(
                "OpenAIChatCompletionsRequestDriver requires RequestFuncInput."
            )
        api_url = request_func_input.api_url
        assert api_url.endswith("chat/completions"), (
            "OpenAI Chat Completions API URL must end with 'chat/completions'."
        )

        if isinstance(request_func_input.prompt, str):  # question only
            content = [{"type": "text", "text": request_func_input.prompt}]
            messages_data = [
                {"role": "user", "content": content},
            ]
        else:  # conversation
            messages_data = [
                msg.model_dump() for msg in request_func_input.prompt
            ]

        payload: dict[
            str,
            bool | str | int | float | list[dict[str, Any]] | dict[str, Any],
        ] = {
            "model": request_func_input.model,
            "messages": messages_data,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": request_func_input.ignore_eos,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens
        _apply_sampling_to_request_payload(payload, request_func_input.sampling)
        if request_func_input.response_format is not None:
            # Convert TypedDict to plain dict so mypy accepts the assignment into
            # payload (since a TypedDict is stricter than a dict[str, Any]).
            payload["response_format"] = dict(
                request_func_input.response_format
            )
        for img in request_func_input.images:
            # TODO: Remove this type ignore
            # (error: Value of type "object" is not indexable)
            payload["messages"][0]["content"].append(img)  # type: ignore[index, union-attr]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        if request_func_input.session_id:
            headers["X-Session-ID"] = request_func_input.session_id

        return await _run_openai_stream_request(
            api_url=api_url,
            payload=payload,
            headers=headers,
            prompt_len=request_func_input.prompt_len,
            # NOTE:
            # "reasoning" and "reasoning_content" are NOT official OpenAI fields.
            # Different model providers and serving frameworks may emit one or both
            # to stream chain-of-thought tokens separately from "content". These
            # fields may also be None in some chunks.
            #
            # We merge them here to preserve all streamed text.
            chunk_type=_ChatCompletionChunk,
            content_extractor=lambda data: (
                (data.choices[0].delta.reasoning or "")
                + (data.choices[0].delta.reasoning_content or "")
                + (data.choices[0].delta.content or "")
            ),
            tokenizer=self.tokenizer,
        )


def _count_output_images(data: dict[str, Any]) -> int:
    output = data.get("output")
    if not isinstance(output, list):
        logger.warning(
            f"OpenResponses response has unexpected 'output' type: "
            f"{type(output).__name__}"
        )
        return 0

    count = 0

    for message_idx, message in enumerate(output):
        if not isinstance(message, dict):
            logger.warning(
                f"Skipping output[{message_idx}]: expected dict, got {type(message)}."
            )
            continue
        content = message.get("content")
        if not isinstance(content, list):
            logger.warning(
                f"Skipping output[{message_idx}].content: expected list, got {type(content)}."
            )
            continue
        for item_idx, item in enumerate(content):
            if not isinstance(item, dict):
                logger.warning(
                    f"Skipping output[{message_idx}].content[{item_idx}]: expected dict, got {type(item)}."
                )
                continue
            if item.get("type") == "output_image":
                count += 1

    return count


def _build_pixel_generation_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> dict[str, Any]:
    input_payload: str | list[dict[str, Any]]
    if request_func_input.input_image_paths:
        content: list[dict[str, Any]] = []
        for image_path in request_func_input.input_image_paths:
            content.append(
                _encode_openresponses_image_from_file_path(image_path)
            )
        content.append(
            {"type": "input_text", "text": request_func_input.prompt}
        )
        input_payload = [{"role": "user", "content": content}]
    else:
        input_payload = request_func_input.prompt

    payload: dict[str, Any] = {
        "model": request_func_input.model,
        "input": input_payload,
    }

    if request_func_input.image_options is None:
        return payload

    options_payload: dict[str, Any] = {}
    image_options = request_func_input.image_options
    if image_options.width is not None:
        options_payload["width"] = image_options.width
    if image_options.height is not None:
        options_payload["height"] = image_options.height
    if image_options.steps is not None:
        options_payload["steps"] = image_options.steps
    if image_options.guidance_scale is not None:
        options_payload["guidance_scale"] = image_options.guidance_scale
    if image_options.negative_prompt is not None:
        options_payload["negative_prompt"] = image_options.negative_prompt
    # num_frames is video-only; presence routes the payload to
    # provider_options.video instead of provider_options.image.
    is_video = image_options.num_frames is not None
    if is_video:
        options_payload["num_frames"] = image_options.num_frames

    if options_payload:
        modality_key = "video" if is_video else "image"
        payload["provider_options"] = {modality_key: options_payload}

    if image_options.seed is not None:
        payload["seed"] = image_options.seed

    return payload


class OpenResponsesRequestDriver(RequestDriver):
    """Request driver for OpenResponses API."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        """Execute a request to the OpenResponses API."""
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "OpenResponsesRequestDriver requires PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        assert api_url.endswith("responses"), (
            "OpenResponses API URL must end with 'responses'."
        )

        payload = _build_pixel_generation_payload(request_func_input)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers
                ) as response:
                    output.latency = time.perf_counter() - start
                    if response.status != 200:
                        body = await response.text()
                        output.error = (
                            f"HTTP {response.status}: {body}"
                            if body
                            else (response.reason or "")
                        )
                        output.success = False
                        return output

                    body = await response.json()
                    output.num_generated_outputs = _count_output_images(body)
                    if output.num_generated_outputs <= 0:
                        output.error = (
                            "No output_image content found in OpenResponses "
                            "response body."
                        )
                        output.success = False
                        return output

                    output.success = True
                    return output
            except Exception:
                output.latency = time.perf_counter() - start
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                return output


def _build_sglang_pixel_generation_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> dict[str, Any]:
    """Build payload for sglang's /v1/images/generations endpoint."""
    payload: dict[str, Any] = {
        "model": request_func_input.model,
        "prompt": request_func_input.prompt,
        "n": 1,
        "response_format": "b64_json",
    }

    if request_func_input.image_options is not None:
        opts = request_func_input.image_options
        if opts.width is not None and opts.height is not None:
            payload["size"] = f"{opts.width}x{opts.height}"
        if opts.steps is not None:
            payload["num_inference_steps"] = opts.steps
        if opts.guidance_scale is not None:
            payload["guidance_scale"] = opts.guidance_scale
        if opts.seed is not None:
            payload["seed"] = opts.seed
        # negative_prompt is not supported by sglang's images API.

    return payload


class SglangPixelGenerationRequestDriver(RequestDriver):
    """Request driver for sglang's /v1/images/generations endpoint."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "SglangPixelGenerationRequestDriver requires"
                " PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        if not api_url.endswith("images/generations"):
            raise ValueError(
                "Sglang pixel generation URL must end with"
                " 'images/generations'."
            )

        payload = _build_sglang_pixel_generation_payload(request_func_input)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers
                ) as response:
                    output.latency = time.perf_counter() - start
                    if response.status != 200:
                        body = await response.text()
                        output.error = (
                            f"HTTP {response.status}: {body}"
                            if body
                            else (response.reason or "")
                        )
                        output.success = False
                        return output

                    body = await response.json()
                    # sglang returns {"data": [{"b64_json": "..."}, ...]}
                    data = body.get("data", [])
                    output.num_generated_outputs = len(data)
                    if output.num_generated_outputs <= 0:
                        output.error = (
                            "No images found in sglang response body."
                        )
                        output.success = False
                        return output

                    output.success = True
                    return output
            except Exception:
                output.latency = time.perf_counter() - start
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                return output


class SglangVideoPayload(TypedDict):
    model: str
    prompt: str
    width: NotRequired[int]
    height: NotRequired[int]
    num_inference_steps: NotRequired[int]
    guidance_scale: NotRequired[float]
    seed: NotRequired[int]
    negative_prompt: NotRequired[str]
    num_frames: NotRequired[int]


def _build_sglang_video_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> SglangVideoPayload:
    """Build JSON payload for sglang's POST /v1/videos endpoint."""
    payload = SglangVideoPayload(
        model=request_func_input.model,
        prompt=request_func_input.prompt,
    )

    if request_func_input.image_options is not None:
        opts = request_func_input.image_options
        if opts.width is not None:
            payload["width"] = opts.width
        if opts.height is not None:
            payload["height"] = opts.height
        if opts.steps is not None:
            payload["num_inference_steps"] = opts.steps
        if opts.guidance_scale is not None:
            payload["guidance_scale"] = opts.guidance_scale
        if opts.seed is not None:
            payload["seed"] = opts.seed
        if opts.negative_prompt is not None:
            payload["negative_prompt"] = opts.negative_prompt
        if opts.num_frames is not None:
            payload["num_frames"] = opts.num_frames

    return payload


_SGLANG_VIDEO_POLL_INTERVAL_S = 1.0


class SglangVideoRequestDriver(RequestDriver):
    """Request driver for sglang's async /v1/videos endpoint.

    POST /v1/videos queues a job, then poll GET /v1/videos/{id} until done.
    Ref: https://github.com/sgl-project/sglang/blob/v0.5.10.post1/python/sglang/multimodal_gen/benchmarks/bench_serving.py#L224
    """

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "SglangVideoRequestDriver requires"
                " PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        if not api_url.rstrip("/").endswith("/videos"):
            raise ValueError("Sglang video URL must end with '/videos'.")
        base_url = api_url.rstrip("/")

        payload = _build_sglang_video_payload(request_func_input)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                async with session.post(
                    url=base_url, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        body = await response.text()
                        output.latency = time.perf_counter() - start
                        output.error = (
                            f"HTTP {response.status}: {body}"
                            if body
                            else (response.reason or "")
                        )
                        output.success = False
                        return output

                    body = await response.json()
                    job_id = body.get("id")
                    if not job_id:
                        output.latency = time.perf_counter() - start
                        output.error = (
                            "No job id in sglang video POST response."
                        )
                        output.success = False
                        return output

                poll_url = f"{base_url}/{job_id}"
                while True:
                    async with session.get(
                        url=poll_url, headers=headers
                    ) as poll_response:
                        if poll_response.status != 200:
                            body = await poll_response.text()
                            output.latency = time.perf_counter() - start
                            output.error = (
                                f"Poll HTTP {poll_response.status}: {body}"
                                if body
                                else (poll_response.reason or "")
                            )
                            output.success = False
                            return output

                        poll_body = await poll_response.json()
                        status = poll_body.get("status", "")

                        if status == "completed":
                            output.latency = time.perf_counter() - start
                            output.num_generated_outputs = 1
                            output.success = True

                            inference_time = poll_body.get("inference_time_s")
                            if inference_time:
                                logger.debug(
                                    "sglang video: inference_time=%s s",
                                    inference_time,
                                )
                            return output

                        if status == "failed":
                            output.latency = time.perf_counter() - start
                            error_info = poll_body.get("error", {})
                            output.error = (
                                error_info.get("message", "")
                                if isinstance(error_info, dict)
                                else str(error_info)
                            )
                            output.success = False
                            return output

                        await asyncio.sleep(_SGLANG_VIDEO_POLL_INTERVAL_S)

            except Exception:
                output.latency = time.perf_counter() - start
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                return output


def _build_vllm_omni_pixel_generation_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> dict[str, Any]:
    """Build payload for vllm-omni's /v1/chat/completions endpoint."""
    extra_body: dict[str, Any] = {}

    if request_func_input.image_options is not None:
        opts = request_func_input.image_options
        if opts.height is not None:
            extra_body["height"] = opts.height
        if opts.width is not None:
            extra_body["width"] = opts.width
        if opts.steps is not None:
            extra_body["num_inference_steps"] = opts.steps
        if opts.guidance_scale is not None:
            extra_body["guidance_scale"] = opts.guidance_scale
        if opts.seed is not None:
            extra_body["seed"] = opts.seed
        # negative_prompt is not supported by vllm-omni's chat API.

    payload: dict[str, Any] = {
        "model": request_func_input.model,
        "messages": [{"role": "user", "content": request_func_input.prompt}],
    }
    if extra_body:
        payload["extra_body"] = extra_body

    return payload


class VllmOmniPixelGenerationRequestDriver(RequestDriver):
    """Request driver for vllm-omni's /v1/chat/completions endpoint
    (diffusion image generation)."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "VllmOmniPixelGenerationRequestDriver requires"
                " PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        if not api_url.endswith("chat/completions"):
            raise ValueError(
                "vllm-omni pixel generation URL must end with"
                " 'chat/completions'."
            )

        payload = _build_vllm_omni_pixel_generation_payload(request_func_input)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                async with session.post(
                    url=api_url, json=payload, headers=headers
                ) as response:
                    output.latency = time.perf_counter() - start
                    if response.status != 200:
                        body = await response.text()
                        output.error = (
                            f"HTTP {response.status}: {body}"
                            if body
                            else (response.reason or "")
                        )
                        output.success = False
                        return output

                    body = await response.json()
                    # vllm-omni returns chat completions format with image
                    # data in choices[].message.content[].image_url.url
                    choices = body.get("choices", [])
                    count = 0
                    for choice in choices:
                        message = choice.get("message", {})
                        content = message.get("content", [])
                        if isinstance(content, list):
                            count += sum(
                                1
                                for item in content
                                if isinstance(item, dict)
                                and item.get("type") == "image_url"
                            )

                    output.num_generated_outputs = count
                    if output.num_generated_outputs <= 0:
                        output.error = (
                            "No images found in vllm-omni response body."
                        )
                        output.success = False
                        return output

                    output.success = True
                    return output
            except Exception:
                output.latency = time.perf_counter() - start
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                return output


def _build_vllm_omni_video_payload(
    request_func_input: PixelGenerationRequestFuncInput,
) -> dict[str, str]:
    """Build form payload for vllm-omni's /v1/videos/sync endpoint."""
    payload: dict[str, str] = {
        "prompt": request_func_input.prompt,
        "model": request_func_input.model,
    }

    if request_func_input.image_options is not None:
        opts = request_func_input.image_options
        if opts.width is not None:
            payload["width"] = str(opts.width)
        if opts.height is not None:
            payload["height"] = str(opts.height)
        if opts.steps is not None:
            payload["num_inference_steps"] = str(opts.steps)
        if opts.guidance_scale is not None:
            payload["guidance_scale"] = str(opts.guidance_scale)
        if opts.seed is not None:
            payload["seed"] = str(opts.seed)
        if opts.negative_prompt is not None:
            payload["negative_prompt"] = opts.negative_prompt
        if opts.num_frames is not None:
            payload["num_frames"] = str(opts.num_frames)

    return payload


class VllmOmniVideoRequestDriver(RequestDriver):
    """Request driver for vllm-omni's /v1/videos/sync endpoint
    (diffusion video generation)."""

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> PixelGenerationRequestFuncOutput:
        if not isinstance(request_func_input, PixelGenerationRequestFuncInput):
            raise TypeError(
                "VllmOmniVideoRequestDriver requires"
                " PixelGenerationRequestFuncInput."
            )
        api_url = request_func_input.api_url
        if not api_url.endswith("videos/sync"):
            raise ValueError(
                "vllm-omni video generation URL must end with 'videos/sync'."
            )

        payload = _build_vllm_omni_video_payload(request_func_input)

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = PixelGenerationRequestFuncOutput()
        start = time.perf_counter()
        output.request_submit_time = start

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            try:
                async with session.post(
                    url=api_url, data=payload, headers=headers
                ) as response:
                    output.latency = time.perf_counter() - start
                    if response.status != 200:
                        body = await response.text()
                        output.error = (
                            f"HTTP {response.status}: {body}"
                            if body
                            else (response.reason or "")
                        )
                        output.success = False
                        return output

                    video_bytes = await response.read()
                    if not video_bytes:
                        output.error = (
                            "Empty response body from /v1/videos/sync."
                        )
                        output.success = False
                        return output

                    output.num_generated_outputs = 1
                    output.success = True

                    inference_time = response.headers.get("X-Inference-Time-S")
                    if inference_time:
                        logger.debug(
                            "vllm-omni video: inference_time=%s s",
                            inference_time,
                        )

                    return output
            except Exception:
                output.latency = time.perf_counter() - start
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))
                return output


class RequestCounter:
    """Thread-safe counter for limiting the number of requests in benchmarks.

    This class provides a simple mechanism to track and limit the total number
    of requests sent across multiple concurrent threads. It uses a threading.Lock
    to ensure thread-safe access to the counter.

    Attributes:
        max_requests: Maximum number of requests allowed
        total_sent_requests: Current count of sent requests
        req_counter_lock: Threading lock for thread-safe access
    """

    def __init__(
        self,
        max_requests: int,
        total_sent_requests: int = 0,
    ) -> None:
        """Initialize the request counter.

        Args:
            max_requests: Maximum number of requests allowed
            total_sent_requests: Initial count of sent requests (default: 0)
        """
        self.max_requests = max_requests
        self.req_counter_lock = threading.Lock()
        self.total_sent_requests = total_sent_requests

    def advance_until_max(self) -> bool:
        """Atomically check and increment the request counter.

        This method performs a thread-safe check-and-increment operation.
        If the current count is below max_requests, it increments the counter
        and returns True. If the limit has been reached, it returns False.

        Returns:
            True if the request can proceed (counter was incremented),
            False if max_requests has been reached.
        """
        with self.req_counter_lock:
            if self.total_sent_requests >= self.max_requests:
                return False

            self.total_sent_requests += 1
            if self.total_sent_requests == self.max_requests:
                logger.info(
                    f"Ending run: max requests {self.max_requests} have been"
                    " sent"
                )
            return True


async def async_request_lora_load(
    api_url: str, lora_name: str, lora_path: str
) -> tuple[bool, float]:
    """Load a LoRA adapter via the API.

    Returns:
        Tuple of (success, load_time_ms)
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {"lora_name": lora_name, "lora_path": lora_path}
        headers = {"Content-Type": "application/json"}
        logger.debug(f"Loading LoRA '{lora_name}' from path: {lora_path}")

        start_time = time.perf_counter()
        try:
            async with session.post(
                url=f"{api_url}/v1/load_lora_adapter",
                json=payload,
                headers=headers,
            ) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    logger.debug(
                        f"Successfully loaded LoRA '{lora_name}' in"
                        f" {elapsed_ms:.2f}ms"
                    )
                    return True, elapsed_ms
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to load LoRA '{lora_name}': {error_text}"
                    )
                    return False, elapsed_ms
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Exception loading LoRA '{lora_name}'")
            return False, elapsed_ms


async def async_request_lora_unload(
    api_url: str, lora_name: str
) -> tuple[bool, float]:
    """Unload a LoRA adapter via the API.

    Returns:
        Tuple of (success, unload_time_ms)
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {"lora_name": lora_name}
        headers = {"Content-Type": "application/json"}

        start_time = time.perf_counter()
        try:
            async with session.post(
                url=f"{api_url}/v1/unload_lora_adapter",
                json=payload,
                headers=headers,
            ) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if response.status == 200:
                    logger.debug(
                        f"Successfully unloaded LoRA '{lora_name}' in"
                        f" {elapsed_ms:.2f}ms"
                    )
                    return True, elapsed_ms
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to unload LoRA '{lora_name}': {error_text}"
                    )
                    return False, elapsed_ms
        except Exception:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Exception unloading LoRA '{lora_name}'")
            return False, elapsed_ms


def get_request_driver_class(
    api_url: str,
    task: BenchmarkTask = "text-generation",
) -> type[RequestDriver]:
    """Return the request driver based on endpoint and optional task.

    For pixel generation, driver selection is based on URL suffix because
    each backend uses a fundamentally different API format. The mapping is:
      /v1/responses          -> OpenResponsesRequestDriver (modular)
      /v1/images/generations -> SglangPixelGenerationRequestDriver
      /v1/videos/sync        -> VllmOmniVideoRequestDriver
      /v1/videos             -> SglangVideoRequestDriver
      /v1/chat/completions   -> VllmOmniPixelGenerationRequestDriver
    The correct endpoint is typically auto-selected by PIXEL_GEN_DEFAULT_ENDPOINT
    (or VIDEO_GEN_DEFAULT_ENDPOINT for text-to-video) in benchmark_serving.py
    based on the --backend flag and task.
    """
    if task in PIXEL_GENERATION_TASKS:
        if api_url.endswith("responses"):
            return OpenResponsesRequestDriver
        if api_url.endswith("images/generations"):
            return SglangPixelGenerationRequestDriver
        if api_url.endswith("videos/sync"):
            return VllmOmniVideoRequestDriver
        if api_url.endswith("/videos"):
            return SglangVideoRequestDriver
        if api_url.endswith("chat/completions"):
            return VllmOmniPixelGenerationRequestDriver
        raise ValueError(
            "Unsupported API URL for pixel-generation driver selection: "
            f"'{api_url}'. Expected /v1/responses, /v1/images/generations,"
            " /v1/videos/sync, /v1/videos, or /v1/chat/completions."
        )

    # for text generation task
    if api_url.endswith("chat/completions"):
        return OpenAIChatCompletionsRequestDriver
    if api_url.endswith(("completions", "profile")):
        return OpenAICompletionsRequestDriver
    if api_url.endswith("generate_stream"):
        return TRTLLMRequestDriver
    raise ValueError(
        "Unsupported API URL for request driver selection: "
        f"'{api_url}'. Expected an OpenAI completions/chat endpoint or "
        "TensorRT-LLM generate_stream endpoint."
    )
