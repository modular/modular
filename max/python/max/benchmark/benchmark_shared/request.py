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

import json
import logging
import math
import os
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .datasets.types import OpenAIImage
from .tts_workloads_utils import SampleTTSRequest

# 30 minute timeout per request session
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=30 * 60)

logger = logging.getLogger(__name__)


@dataclass
class BaseRequestFuncInput:
    """Base class for request function input with common fields."""

    model: str
    session_id: str | None
    temperature: float | None
    top_p: float | None
    top_k: int | None


# TODO: We shouldn't have to maintain two separate RequestFuncInput classes for
# text generation and TTS benchmarks respectively.
@dataclass
class RequestFuncInput(BaseRequestFuncInput):
    """Request function input for text generation benchmarks."""

    prompt: str | list[dict[str, Any]]
    images: list[OpenAIImage]
    api_url: str
    prompt_len: int
    max_tokens: int | None
    ignore_eos: bool


@dataclass
class TTSRequestFuncInput(BaseRequestFuncInput):
    """Request function input for TTS (text-to-speech) benchmarks."""

    request_index: int
    tts_request: SampleTTSRequest
    is_streaming_mode: bool
    frequency_penalty: float
    repetition_penalty: float
    seed: int = 0


@dataclass
class BaseRequestFuncOutput:
    """Base class for request function output with common fields."""

    success: bool = False
    latency: float = 0.0
    # List of inter-token latencies
    itl: list[float] = field(default_factory=list)
    # List of per-chunk time-per-output-token values
    tpot: list[float] = field(default_factory=list)
    error: str = ""


# TODO: We shouldn't have to maintain two separate RequestFuncOutput classes for
# text generation and TTS benchmarks respectively.
@dataclass
class RequestFuncOutput(BaseRequestFuncOutput):
    """Request function output for text generation benchmarks."""

    cancelled: bool = False
    generated_text: str = ""
    ttft: float = 0.0  # Time to first token
    prompt_len: int = 0


@dataclass
class TTSRequestFuncOutput(BaseRequestFuncOutput):
    """Request function output for TTS (text-to-speech) benchmarks."""

    request_index: int = 0
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
        self, request_func_input: RequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the backend API.

        Args:
            request_func_input: Input parameters for the request.

        Returns:
            RequestFuncOutput containing the response data and metrics.
        """
        pass


class ProgressBarRequestDriver(RequestDriver):
    """Request driver that updates a progress bar after each request."""

    def __init__(self, request_driver: RequestDriver, pbar: tqdm) -> None:
        """Initialize the progress bar request driver.

        Args:
            request_driver: The underlying request driver to wrap.
            pbar: Progress bar to update after each request completes.
        """
        super().__init__(tokenizer=request_driver.tokenizer)
        self.request_driver = request_driver
        self.pbar = pbar

    async def request(
        self, request_func_input: RequestFuncInput
    ) -> RequestFuncOutput:
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
        self, request_func_input: RequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the TensorRT-LLM backend."""
        api_url = request_func_input.api_url
        assert api_url.endswith("generate_stream")

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            payload: dict[
                str, bool | str | int | float | list[dict[str, Any]]
            ] = {
                "text_input": request_func_input.prompt,
                "ignore_eos": request_func_input.ignore_eos,
                "stream": True,
            }

            if request_func_input.max_tokens is not None:
                payload["max_tokens"] = request_func_input.max_tokens
            if request_func_input.top_k is not None:
                payload["top_k"] = request_func_input.top_k
            if request_func_input.temperature is not None:
                payload["temperature"] = request_func_input.temperature
            if request_func_input.top_p is not None:
                payload["top_p"] = request_func_input.top_p

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = chunk_bytes.decode("utf-8").removeprefix(
                                "data:"
                            )

                            data = json.loads(chunk)
                            chunk_text = data["text_output"]
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


async def _run_openai_stream_request(
    *,
    api_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    prompt_len: int,
    content_extractor: Callable[[dict[str, Any]], str],
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.prompt_len = prompt_len

    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st
    has_content = False
    latency = 0.0

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix(
                            "data: "
                        )
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            continue

                        data = json.loads(chunk)

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
        self, request_func_input: RequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the OpenAI-compatible completions API."""
        api_url = request_func_input.api_url
        assert api_url.endswith(("completions", "profile")), (
            "OpenAI Completions API URL must end with 'completions' or 'profile'."
        )

        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": request_func_input.temperature,
            "best_of": 1,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

        if request_func_input.top_k is not None:
            payload["top_k"] = request_func_input.top_k

        if request_func_input.top_p is not None:
            payload["top_p"] = request_func_input.top_p

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        return await _run_openai_stream_request(
            api_url=api_url,
            payload=payload,
            headers=headers,
            prompt_len=request_func_input.prompt_len,
            content_extractor=lambda data: data["choices"][0]["text"],
            tokenizer=self.tokenizer,
        )


class OpenAIChatCompletionsRequestDriver(RequestDriver):
    """Request driver for OpenAI-compatible chat completions API."""

    async def request(
        self, request_func_input: RequestFuncInput
    ) -> RequestFuncOutput:
        """Execute a request to the OpenAI-compatible chat completions API."""
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
            messages_data = request_func_input.prompt

        payload = {
            "model": request_func_input.model,
            "messages": messages_data,
            "temperature": request_func_input.temperature,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

        if request_func_input.top_k is not None:
            payload["top_k"] = request_func_input.top_k

        if request_func_input.top_p is not None:
            payload["top_p"] = request_func_input.top_p

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
            content_extractor=lambda data: data["choices"][0]["delta"].get(
                "content", ""
            ),
            tokenizer=self.tokenizer,
        )


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
                logger.warning(
                    f"Ending run: max requests {self.max_requests} have been"
                    " sent"
                )
                return False

            self.total_sent_requests += 1
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


def get_request_driver_class(api_url: str) -> type[RequestDriver]:
    """Return the request driver based on the API URL."""
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
