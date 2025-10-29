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

"""Request-related data structures for benchmarking."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm.asyncio import tqdm

from .datasets.types import OpenAIImage

# 30 minute timeout per request session
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=30 * 60)

logger = logging.getLogger(__name__)


@dataclass
class RequestFuncInput:
    prompt: str | list[dict[str, Any]]
    images: list[OpenAIImage]
    api_url: str
    prompt_len: int
    max_tokens: int | None
    ignore_eos: bool
    model: str
    session_id: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


@dataclass
class RequestFuncOutput:
    cancelled: bool = False
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    # List of inter-token latencies
    itl: list[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""


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


async def async_request_trt_llm(
    request_func_input: RequestFuncInput, pbar: tqdm | None = None
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload: dict[str, bool | str | int | float | list[dict[str, Any]]] = {
            "accumulate_tokens": True,
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
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

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

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput, pbar: tqdm | None = None
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": request_func_input.temperature,
            "top_p": request_func_input.top_p,
            "best_of": 1,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

        if request_func_input.top_k is not None:
            payload["top_k"] = request_func_input.top_k

        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        has_content = False
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
                            pass
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                has_content = True

                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(
                                        timestamp - most_recent_timestamp
                                    )

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]
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
    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput, pbar: tqdm | None = None
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("chat/completions"), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
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
            "top_p": request_func_input.top_p,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

        if request_func_input.top_k is not None:
            payload["top_k"] = request_func_input.top_k

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

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        has_content = False
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
                            pass
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                has_content = True

                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(
                                        timestamp - most_recent_timestamp
                                    )

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

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

    if pbar:
        pbar.update(1)
    return output


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


# Dictionary mapping backend names to their corresponding async request functions
# TODO: This is not a permanent home for these request endpoint calls. Ideally,
# we'll have proper, per backend interfaces and we can call against them. This
# will be done in a future PR though.
ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "vllm-chat": async_request_openai_chat_completions,
    "trt-llm": async_request_trt_llm,
    "modular": async_request_openai_completions,
    "modular-chat": async_request_openai_chat_completions,
    "sglang": async_request_openai_completions,
    "sglang-chat": async_request_openai_chat_completions,
}
