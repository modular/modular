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

"""Benchmark online serving throughput."""

import argparse
import asyncio
import itertools
import json
import logging
import os
import random
import resource
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser as FlexibleArgumentParser
from collections.abc import AsyncGenerator, Awaitable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Union

import aiohttp
import numpy as np
import yaml
from benchmark_datasets import (
    DATASET_REGISTRY,
    ArxivSummarizationBenchmarkDataset,
    AxolotlBenchmarkDataset,
    BenchmarkDataset,
    CodeDebugBenchmarkDataset,
    RandomBenchmarkDataset,
    ShareGPTBenchmarkDataset,
    SonnetBenchmarkDataset,
    VisionArenaBenchmarkDataset,
)
from sample_workload_utils import ChatSession, OpenAIImage, SampledRequest
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

# 10 minute timeout per request session
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=30 * 60)

logger = logging.getLogger("benchmark_serving")


@dataclass
class RequestFuncInput:
    prompt: Union[str, list[dict]]
    img: Optional[OpenAIImage]
    api_url: str
    prompt_len: int
    max_tokens: Optional[int]
    ignore_eos: bool
    model: str
    lora: str
    session_id: Optional[str] = None
    temperature: float = 0.0


@dataclass
class RequestFuncOutput:
    cancelled: bool = False
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(
        default_factory=list
    )  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


class RequestCounter:
    def __init__(
        self,
        max_requests: int,
        req_counter_lock: asyncio.locks.Lock,
        total_sent_requests: int = 0,
    ) -> None:
        self.max_requests = max_requests
        self.req_counter_lock = req_counter_lock
        self.total_sent_requests = total_sent_requests

    async def advance_until_max(self) -> bool:
        """
        Checks if the number of sent requests has reached max_requests.
        If not, increment by one.

        Returns:
        bool: True if the request hasn't reached max and can advance, otherwise False.
        """
        async with self.req_counter_lock:
            if self.total_sent_requests >= self.max_requests:
                logger.warning(
                    f"Ending run: max requests {self.max_requests} have been sent"
                )
                return False

            self.total_sent_requests += 1
            return True


def min_ignore_none(x: Sequence[Optional[int]]) -> Optional[int]:
    filtered = [elem for elem in x if elem is not None]
    return min(filtered, default=None)


def compute_output_len(
    tokenizer: PreTrainedTokenizerBase, output: RequestFuncOutput
) -> int:
    return len(
        tokenizer(
            output.generated_text,
            add_special_tokens=False,
        ).input_ids
    )


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": request_func_input.temperature,
            "top_p": 1.0,
            "ignore_eos": request_func_input.ignore_eos,
            "stream": True,
        }

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

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

                        chunk = remove_prefix(
                            chunk_bytes.decode("utf-8"), "data:"
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
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
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
            "best_of": 1,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
            "skip_special_tokens": False,
        }

        if request_func_input.lora is not None:
            payload["lora"] = request_func_input.lora

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

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

                        chunk = remove_prefix(
                            chunk_bytes.decode("utf-8"), "data: "
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
                        output.error = "No content returned, there could be an issue with accuracy"
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
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
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
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
            "skip_special_tokens": False,
        }

        if request_func_input.lora is not None:
            payload["lora"] = request_func_input.lora

        if request_func_input.max_tokens is not None:
            payload["max_tokens"] = request_func_input.max_tokens

        if request_func_input.img:
            # TODO: Remove this type ignore
            # (error: Value of type "object" is not indexable)
            payload["messages"][0]["content"].append(request_func_input.img)  # type: ignore[index]

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

                        chunk = remove_prefix(
                            chunk_bytes.decode("utf-8"), "data: "
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
                        output.error = "No content returned, there could be an issue with accuracy"
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


# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_tokenizer(
    pretrained_model_name_or_path: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code
    )


ASYNC_REQUEST_FUNCS = {
    "vllm": async_request_openai_completions,
    "vllm-chat": async_request_openai_chat_completions,
    "trt-llm": async_request_trt_llm,
    "modular": async_request_openai_completions,
    "modular-chat": async_request_openai_chat_completions,
    "sglang": async_request_openai_completions,
    "sglang-chat": async_request_openai_chat_completions,
}

MIN_PROMPT_LEN = 4
MIN_OUTPUT_LEN = 4


# from https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py#L1283
def set_ulimit(target_soft_limit: int = 65535) -> None:
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


def _validate_data(data: list[float]) -> None:
    """Validate input data for metrics calculations."""
    assert isinstance(data, list), "data must be a list"
    assert len(data) > 0, "data must not be empty"
    assert all(isinstance(x, float) for x in data), (
        "data must contain only floats"
    )


def _calculate_basic_stats(
    data: list[float], scale_factor: float
) -> dict[str, float]:
    """Calculate basic statistics (mean, std, median) with scaling."""
    return {
        "mean": float(np.mean(data)) * scale_factor,
        "std": float(np.std(data)) * scale_factor,
        "median": float(np.median(data)) * scale_factor,
    }


@dataclass
class PercentileMetrics:
    """Container for percentile-based metrics."""

    mean: float
    std: float
    median: float
    p90: float
    p95: float
    p99: float
    unit: Optional[str] = None

    def __str__(self) -> str:
        """Return a formatted string representation of the metrics in table format."""
        lines = []
        lines.append("{:<40} {:<10.2f}".format("Mean:", self.mean))
        lines.append("{:<40} {:<10.2f}".format("Std:", self.std))
        lines.append("{:<40} {:<10.2f}".format("Median:", self.median))
        lines.append("{:<40} {:<10.2f}".format("P90:", self.p90))
        lines.append("{:<40} {:<10.2f}".format("P95:", self.p95))
        lines.append("{:<40} {:<10.2f}".format("P99:", self.p99))
        return "\n".join(lines)

    def format_with_prefix(
        self, prefix: str, unit: Optional[str] = None
    ) -> str:
        """Return formatted metrics with a custom prefix for labels."""
        # Use passed unit, or fall back to self.unit
        effective_unit = unit or self.unit
        unit_suffix = f" ({effective_unit})" if effective_unit else ""
        metrics_data = [
            ("Mean", self.mean),
            ("Std", self.std),
            ("Median", self.median),
            ("P90", self.p90),
            ("P95", self.p95),
            ("P99", self.p99),
        ]
        return "\n".join(
            "{:<40} {:<10.2f}".format(f"{label} {prefix}{unit_suffix}:", value)
            for label, value in metrics_data
        )


class ThroughputMetrics:
    """
    Container for throughput-based metrics with automatic percentile calculations.

    For throughput metrics, percentiles are reversed because smaller values
    are worse for throughput (e.g., p99 represents the 1st percentile).
    """

    def __init__(
        self,
        data: list[float],
        scale_factor: float = 1.0,
        unit: Optional[str] = None,
    ) -> None:
        """
        Initialize throughput metrics with automatic percentile calculations.

        Args:
            data: List of throughput values to calculate percentiles from.
            scale_factor: Factor to multiply all values by (e.g., for unit conversion).
            unit: Unit string to display (e.g., "tok/s", "req/s", "MB/s").
        """
        _validate_data(data)

        # Calculate basic stats and reversed percentiles for throughput
        basic_stats = _calculate_basic_stats(data, scale_factor)
        percentiles = self._calculate_throughput_percentiles(data, scale_factor)

        self._metrics = PercentileMetrics(
            unit=unit,
            **basic_stats,
            **percentiles,
        )

    @staticmethod
    def _calculate_throughput_percentiles(
        data: list[float], scale_factor: float
    ) -> dict[str, float]:
        """Calculate throughput percentiles (reversed: bottom 10%, 5%, 1%)."""
        return {
            "p90": float(np.percentile(data, 10)) * scale_factor,  # Bottom 10%
            "p95": float(np.percentile(data, 5)) * scale_factor,  # Bottom 5%
            "p99": float(np.percentile(data, 1)) * scale_factor,  # Bottom 1%
        }

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the internal metrics object."""
        return getattr(self._metrics, name)

    def __str__(self) -> str:
        """Return a formatted string representation of throughput metrics in table format."""
        return self.format_with_prefix(prefix="throughput")


class StandardPercentileMetrics:
    """
    Container for standard percentile-based metrics with automatic calculations.

    For standard metrics, higher percentiles represent worse performance
    (e.g., p99 represents the 99th percentile).
    """

    def __init__(
        self,
        data: list[float],
        scale_factor: float = 1.0,
        unit: Optional[str] = None,
    ) -> None:
        """
        Initialize standard percentile metrics with automatic calculations.

        Args:
            data: List of values to calculate percentiles from.
            scale_factor: Factor to multiply all values by (e.g., 1000 for ms conversion).
            unit: Unit string to display (e.g., "ms", "s", "MB/s").
        """
        _validate_data(data)

        # Calculate basic stats and standard percentiles
        basic_stats = _calculate_basic_stats(data, scale_factor)
        percentiles = self._calculate_standard_percentiles(data, scale_factor)

        self._metrics = PercentileMetrics(
            unit=unit,
            **basic_stats,
            **percentiles,
        )

    @staticmethod
    def _calculate_standard_percentiles(
        data: list[float], scale_factor: float
    ) -> dict[str, float]:
        """Calculate standard percentiles (90th, 95th, 99th)."""
        return {
            "p90": float(np.percentile(data, 90)) * scale_factor,
            "p95": float(np.percentile(data, 95)) * scale_factor,
            "p99": float(np.percentile(data, 99)) * scale_factor,
        }

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the internal metrics object."""
        return getattr(self._metrics, name)

    def __str__(self) -> str:
        """Return a formatted string representation of standard percentile metrics in table format."""
        return self.format_with_prefix(prefix="metric")


@dataclass
class BenchmarkMetrics:
    completed: int
    failures: int
    total_input: int
    total_output: int
    nonempty_response_chunks: int
    max_concurrency: int
    request_throughput: float

    input_throughput: ThroughputMetrics
    output_throughput: ThroughputMetrics
    ttft_ms: StandardPercentileMetrics
    tpot_ms: StandardPercentileMetrics
    itl_ms: StandardPercentileMetrics
    latency_ms: StandardPercentileMetrics

    max_input: int
    max_output: int
    max_total: int
    # 'benchmark/gpu:i/memory_used (MiB)/max'
    peak_gpu_memory_mib: list[float]
    # 'benchmark/gpu:i/memory_free (MiB)/min'
    available_gpu_memory_mib: list[float]
    # 'benchmark/gpu:i/gpu_utilization (%)/mean'
    gpu_utilization: list[float]


async def get_request(
    input_requests: Sequence[SampledRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampledRequest, None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests:
            A list of input requests, each represented as a SampledRequest.
        request_rate:
            The rate at which requests are generated (requests/s).
        burstiness (optional):
            The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def print_section(title: str, char: str = "-") -> None:
    """Helper function to print a section with formatted header."""
    print("{s:{c}^{n}}".format(s=title, n=50, c=char))


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    gpu_metrics: dict[str, Any],
    ttft_skip_requests: int,
    max_concurrency: Optional[int],
    collect_gpu_stats: bool,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    nonempty_response_chunks = 0
    total_input = 0
    completed = 0
    max_input = 0
    max_output = 0
    max_total = 0
    failures = 0
    failed_responses = []
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    latencies: list[float] = []
    input_throughputs: list[float] = []
    output_throughputs: list[float] = []
    for i in range(len(outputs)):
        # If the request was cancelled due to max_benchmark_duration_s, we skip it
        # and don't count it towards the metrics
        if outputs[i].cancelled:
            continue
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = compute_output_len(tokenizer, outputs[i])
            actual_output_lens.append(output_len)
            nonempty_response_chunks += 1 if outputs[i].ttft != 0 else 0
            nonempty_response_chunks += len(outputs[i].itl)

            total_input += outputs[i].prompt_len
            if output_len > 1:
                tpots.append(
                    (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                )
            itls += outputs[i].itl
            if i >= ttft_skip_requests:
                ttfts.append(outputs[i].ttft)
                # Input throughput is fully calculated once we reach the first output token.
                input_throughputs.append(
                    outputs[i].prompt_len / outputs[i].ttft
                )
                # output throughput ignores the first token.
                # It is just timing for the chain of output tokens.
                output_throughputs.append(
                    (output_len - 1) / (outputs[i].latency - outputs[i].ttft)
                )
            completed += 1
            max_input = max(max_input, outputs[i].prompt_len)
            max_output = max(max_output, output_len)
            max_total = max(max_total, outputs[i].prompt_len + output_len)
            latencies.append(outputs[i].latency)
        else:
            actual_output_lens.append(0)
            failures = failures + 1
            failed_responses.append(outputs[i])

    if len(outputs) == 0:
        warnings.warn(
            ("No responses were received from the server."),
            stacklevel=2,
        )

    if failures != 0:
        warnings.warn(
            (
                "Some requests failed. The responses returned are displayed "
                "below. Please check server logs for more information."
            ),
            stacklevel=2,
        )
        for f in failed_responses:
            logger.error(f"Failed :: {f}")

    if completed == 0:
        warnings.warn(
            (
                "All requests failed. This is likely due to a misconfiguration "
                "on the benchmark arguments."
            ),
            stacklevel=2,
        )

    peak_gpu_memory_mib = []
    available_gpu_memory_mib = []
    gpu_utilization = []
    if collect_gpu_stats:
        from nvitop import Device
        from nvitop.libnvml import NVMLError  # type: ignore

        try:
            device_count = Device.count()
        except NVMLError as e:
            logging.warning(f"Failed to get GPU device count: {e}")
            logging.warning(
                "GPU stats collection is only supported on NVIDIA GPUs."
            )
            device_count = 0

        for i in range(device_count):
            peak_gpu_memory_mib.append(
                float(
                    gpu_metrics.get(f"benchmark/gpu:{i}/memory_used (MiB)/max")
                    or 0
                )
            )
            available_gpu_memory_mib.append(
                float(
                    gpu_metrics.get(f"benchmark/gpu:{i}/memory_free (MiB)/min")
                    or 0
                )
            )
            gpu_utilization.append(
                float(
                    gpu_metrics.get(
                        f"benchmark/gpu:{i}/gpu_utilization (%)/mean"
                    )
                    or 0
                )
            )

    metrics = BenchmarkMetrics(
        completed=completed,
        failures=failures,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        nonempty_response_chunks=nonempty_response_chunks,
        max_concurrency=max_concurrency or len(outputs),
        request_throughput=completed / dur_s,
        # Use specialized metric classes that handle percentile calculations automatically
        input_throughput=ThroughputMetrics(input_throughputs, unit="tok/s"),
        output_throughput=ThroughputMetrics(output_throughputs, unit="tok/s"),
        ttft_ms=StandardPercentileMetrics(
            ttfts, scale_factor=1000.0, unit="ms"
        ),
        tpot_ms=StandardPercentileMetrics(
            tpots, scale_factor=1000.0, unit="ms"
        ),
        itl_ms=StandardPercentileMetrics(itls, scale_factor=1000.0, unit="ms"),
        latency_ms=StandardPercentileMetrics(
            latencies, scale_factor=1000.0, unit="ms"
        ),
        max_input=max_input,
        max_output=max_output,
        max_total=max_total,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
    )

    return metrics, actual_output_lens


async def chat_session_driver(
    model_id: str,
    lora_id: str,
    api_url: str,
    request_func: Callable[
        [RequestFuncInput, Optional[tqdm]],
        Awaitable[RequestFuncOutput],
    ],
    request_counter: RequestCounter,
    chat_session: ChatSession,
    max_chat_len: int,
    delay_between_chat_turns: Optional[int],
    pbar: Optional[tqdm] = None,
    skip_session_count: Optional[int] = None,
) -> list[RequestFuncOutput]:
    request_func_input = RequestFuncInput(
        model=model_id,
        lora=lora_id,
        prompt=[],
        api_url=api_url,
        prompt_len=0,
        max_tokens=0,
        ignore_eos=True,
        img=None,
        session_id=str(chat_session.id),
    )
    content_idx = 0  # Assume user initiates the conversation

    session_outputs = []
    message_history: list[dict] = []
    chat_len = 0

    messages = chat_session.messages
    while content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
        output_len = messages[content_idx + 1].num_tokens
        if chat_len + output_len > max_chat_len:
            logger.warning(
                f"Ending conversation: hitting max chat length {max_chat_len}"
            )
            break

        advance_request = await request_counter.advance_until_max()
        if not advance_request:  # reached max_requests
            break

        user_prompt = messages[content_idx].content
        message_history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        )
        request_func_input.prompt = message_history
        request_func_input.prompt_len = chat_len
        request_func_input.max_tokens = output_len
        response = await request_func(request_func_input, pbar)
        if (
            skip_session_count is None
            or chat_session.id is None
            or chat_session.id >= skip_session_count
        ):
            session_outputs.append(response)

        if not response.success:
            logger.error(
                f"Ending chat session {chat_session.id} due to server error response: {response.error}"
            )
            break

        content_idx += 2
        message_history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response.generated_text}],
            }
        )
        chat_len += output_len

        if delay_between_chat_turns:
            # todo parameterize the distribution and scale
            # e.g. N(mean, std) or U(lower, upper)
            delay_ms = np.random.normal(
                loc=delay_between_chat_turns,
                scale=delay_between_chat_turns * 0.5,
            )
            await asyncio.sleep(delay_ms / 1000)

    return session_outputs


async def benchmark(
    backend: str,
    chat: bool,
    api_url: str,
    model_id: str,
    lora_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: Sequence[SampledRequest],
    chat_sessions: Sequence[ChatSession],
    request_rate: float,
    burstiness: float,
    max_concurrency: Optional[int],
    disable_tqdm: bool,
    do_test_prompt: bool,
    collect_gpu_stats: bool,
    print_inputs_and_outputs: bool,
    max_requests: int,
    num_chat_sessions: Optional[int],
    delay_between_chat_turns: Optional[int],
    ttft_skip_requests: int,
    max_output_len: Optional[int],
    temperature: float,
    max_benchmark_duration_s: Optional[int],
    warmup_delay_ms: float = 0,
):
    full_backend = backend + ("-chat" if chat else "")
    if full_backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[full_backend]
    else:
        raise ValueError(f"Unknown backend: {full_backend}")

    if do_test_prompt:
        logger.info("Starting initial single prompt test run...")
        test_prompt: Union[str, list[dict]]
        if args.num_chat_sessions:
            test_question = chat_sessions[0].messages[0]
            test_answer = chat_sessions[0].messages[1]
            test_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": test_question.content}
                    ],
                }
            ]
            test_prompt_len = test_question.num_tokens
            test_max_tokens: Optional[int] = test_answer.num_tokens
            test_ignore_eos = True
            test_img = None
        else:
            test_request = input_requests[0]
            test_prompt = test_request.prompt_formatted
            test_prompt_len = test_request.prompt_len
            test_max_tokens = min_ignore_none(
                (test_request.output_len, max_output_len)
            )
            test_ignore_eos = test_request.output_len is not None
            test_img = test_request.encoded_img

        test_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=api_url,
            prompt_len=test_prompt_len,
            max_tokens=test_max_tokens,
            ignore_eos=test_ignore_eos,
            img=test_img,
            lora=lora_id,
        )
        test_output = await request_func(
            request_func_input=test_input,
        )
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark"
                " arguments are correctly specified. Error:"
                f" {test_output.error}"
            )
        else:
            logger.info(
                "Initial test run completed. Starting main benchmark run..."
            )

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    logger.info(f"Traffic request rate: {request_rate}")
    logger.info(f"Burstiness factor: {burstiness} ({distribution})")
    logger.info(f"Maximum request concurrency: {max_concurrency}")

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    if collect_gpu_stats:
        from nvitop import ResourceMetricCollector

        collector = ResourceMetricCollector()
        collector.start("benchmark")

    benchmark_start_time = time.perf_counter_ns()
    if max_benchmark_duration_s is None:
        benchmark_should_end_time = None
    else:
        benchmark_should_end_time = (
            benchmark_start_time + max_benchmark_duration_s * 1e9
        )
    tasks: list[asyncio.Task] = []
    outputs: list[RequestFuncOutput] = []
    if not num_chat_sessions:
        # single-turn chat scenario
        pbar = None if disable_tqdm else tqdm(total=len(input_requests))

        async def limited_request_func(
            request_func_input: RequestFuncInput,
        ) -> RequestFuncOutput:
            if semaphore is None:
                return await request_func(
                    request_func_input=request_func_input, pbar=pbar
                )
            async with semaphore:
                if benchmark_should_end_time is not None:
                    if time.perf_counter_ns() >= benchmark_should_end_time:
                        return RequestFuncOutput(cancelled=True)
                return await request_func(
                    request_func_input=request_func_input, pbar=pbar
                )

        async for request in get_request(
            input_requests, request_rate, burstiness
        ):
            # If the request length is pinned, then we use ignore_eos+max_tokens
            # to force the model's hand into the given request length. Otherwise,
            # we run until the model generates EOS. Letting the model choose
            # request lengths has some downsides (e.g., benchmarking is
            # vulnerable to correctness bugs or even minor optimizations), but
            # sometimes necessary if we have no other way to set the appropriate
            # distribution of output lengths.
            ignore_eos = request.output_len is not None
            max_tokens = min_ignore_none((request.output_len, max_output_len))

            request_func_input = RequestFuncInput(
                model=model_id,
                lora=lora_id,
                prompt=request.prompt_formatted,
                api_url=api_url,
                prompt_len=request.prompt_len,
                max_tokens=max_tokens,
                temperature=temperature,
                ignore_eos=ignore_eos,
                img=request.encoded_img,
            )
            tasks.append(
                asyncio.create_task(limited_request_func(request_func_input))
            )
        outputs = await asyncio.gather(*tasks)

    else:
        # multi-turn chat scenario
        if disable_tqdm:
            pbar = None
        else:
            num_qa_turns = [
                (len(session.messages) // 2) for session in chat_sessions
            ]
            pbar = tqdm(total=sum(num_qa_turns))

        # Track total sent requests among chat sessions
        request_counter = RequestCounter(
            max_requests=max_requests,
            req_counter_lock=asyncio.Lock(),
            total_sent_requests=0,
        )

        # apply the semaphore at the session level
        # ex: with max_concurrency = 1,
        # the first session finishes before the second session starts
        async def limited_chat_session_driver(
            chat_session: ChatSession,
        ) -> list[RequestFuncOutput]:
            if semaphore is None:
                return await chat_session_driver(
                    model_id,
                    lora_id,
                    api_url,
                    request_func,
                    request_counter,
                    chat_session,
                    tokenizer.model_max_length,
                    delay_between_chat_turns,
                    pbar,
                    ttft_skip_requests,
                )
            async with semaphore:
                return await chat_session_driver(
                    model_id,
                    lora_id,
                    api_url,
                    request_func,
                    request_counter,
                    chat_session,
                    tokenizer.model_max_length,
                    delay_between_chat_turns,
                    pbar,
                    ttft_skip_requests,
                )

        for idx, chat_session in enumerate(chat_sessions):
            if (
                warmup_delay_ms > 0
                and max_concurrency
                and idx < max_concurrency
            ):
                await asyncio.sleep(warmup_delay_ms / 1000)
            tasks.append(
                asyncio.create_task(limited_chat_session_driver(chat_session))
            )

        session_outputs = await asyncio.gather(*tasks)
        outputs = [output for sublist in session_outputs for output in sublist]

    if pbar is not None:
        pbar.close()

    benchmark_duration = (time.perf_counter_ns() - benchmark_start_time) / 1e9

    if print_inputs_and_outputs:
        print("Generated output text:")
        for req_id, output in enumerate(outputs):
            output_len = compute_output_len(tokenizer, output)
            print(
                {
                    "req_id": req_id,
                    "output_len": output_len,
                    "output": output.generated_text,
                }
            )

    if collect_gpu_stats:
        gpu_metrics = collector.collect()
        collector.stop()
    else:
        gpu_metrics = {}

    metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        gpu_metrics=gpu_metrics,
        ttft_skip_requests=ttft_skip_requests,
        max_concurrency=max_concurrency,
        collect_gpu_stats=collect_gpu_stats,
    )

    print_section(title=" Serving Benchmark Result ", char="=")
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failures))
    print(
        "{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration)
    )
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print(
        "{:<40} {:<10}".format("Total generated tokens:", metrics.total_output)
    )
    # We found that response chunks can be empty in content and the token number
    # can be different with the re-tokenization in one pass or chunk-by-chunk.
    # Let's count the number of nonempty_response_chunks for all serving backends.
    # With the move to zero-overhead single step scheduling, this should generally
    # exactly match the number of requested output tokens.
    print(
        "{:<40} {:<10}".format(
            "Total nonempty serving response chunks:",
            metrics.nonempty_response_chunks,
        )
    )
    print(
        "{:<40} {:<10.5f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print_section(title="Client Experience Metrics")
    print("{:<40} {:<10}".format("Max Concurrency:", metrics.max_concurrency))
    print(
        metrics.input_throughput.format_with_prefix(
            prefix="input token throughput", unit="tok/s"
        )
    )
    print(
        metrics.output_throughput.format_with_prefix(
            prefix="output token throughput", unit="tok/s"
        )
    )
    print_section(title="Time to First Token")
    print(metrics.ttft_ms.format_with_prefix(prefix="TTFT", unit="ms"))
    print_section(title="Time per Output Token (excl. 1st token)")
    print(metrics.tpot_ms.format_with_prefix(prefix="TPOT", unit="ms"))
    print_section(title="Inter-token Latency")
    print(metrics.itl_ms.format_with_prefix(prefix="ITL", unit="ms"))
    print_section(title="Per-Request E2E Latency")
    print(
        metrics.latency_ms.format_with_prefix(
            prefix="Request Latency", unit="ms"
        )
    )
    print_section(title="Token Stats")
    print("{:<40} {:<10}".format("Max input tokens:", metrics.max_input))
    print("{:<40} {:<10}".format("Max output tokens:", metrics.max_output))
    print("{:<40} {:<10}".format("Max total tokens:", metrics.max_total))
    if collect_gpu_stats:
        for i in range(len(metrics.gpu_utilization)):
            print_section(title=f"GPU Stats {i}")
            print(
                "{:<40} {:<10.2f}".format(
                    "GPU Utilization (%):", metrics.gpu_utilization[i]
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Peak GPU Memory Used (MiB):",
                    metrics.peak_gpu_memory_mib[i],
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "GPU Memory Available (MiB):",
                    metrics.available_gpu_memory_mib[i],
                )
            )

    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "failures": metrics.failures,
        "max_concurrency": metrics.max_concurrency,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "mean_input_throughput": metrics.input_throughput.mean,
        "std_input_throughput": metrics.input_throughput.std,
        "median_input_throughput": metrics.input_throughput.median,
        "p90_input_throughput": metrics.input_throughput.p90,
        "p95_input_throughput": metrics.input_throughput.p95,
        "p99_input_throughput": metrics.input_throughput.p99,
        "mean_output_throughput": metrics.output_throughput.mean,
        "std_output_throughput": metrics.output_throughput.std,
        "median_output_throughput": metrics.output_throughput.median,
        "p90_output_throughput": metrics.output_throughput.p90,
        "p95_output_throughput": metrics.output_throughput.p95,
        "p99_output_throughput": metrics.output_throughput.p99,
        "mean_ttft_ms": metrics.ttft_ms.mean,
        "median_ttft_ms": metrics.ttft_ms.median,
        "std_ttft_ms": metrics.ttft_ms.std,
        "p90_ttft_ms": metrics.ttft_ms.p90,
        "p95_ttft_ms": metrics.ttft_ms.p95,
        "p99_ttft_ms": metrics.ttft_ms.p99,
        "mean_tpot_ms": metrics.tpot_ms.mean,
        "median_tpot_ms": metrics.tpot_ms.median,
        "std_tpot_ms": metrics.tpot_ms.std,
        "p90_tpot_ms": metrics.tpot_ms.p90,
        "p95_tpot_ms": metrics.tpot_ms.p95,
        "p99_tpot_ms": metrics.tpot_ms.p99,
        "mean_itl_ms": metrics.itl_ms.mean,
        "median_itl_ms": metrics.itl_ms.median,
        "std_itl_ms": metrics.itl_ms.std,
        "p90_itl_ms": metrics.itl_ms.p90,
        "p95_itl_ms": metrics.itl_ms.p95,
        "p99_itl_ms": metrics.itl_ms.p99,
        "mean_latency_ms": metrics.latency_ms.mean,
        "median_latency_ms": metrics.latency_ms.median,
        "std_latency_ms": metrics.latency_ms.std,
        "p90_latency_ms": metrics.latency_ms.p90,
        "p95_latency_ms": metrics.latency_ms.p95,
        "p99_latency_ms": metrics.latency_ms.p99,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "peak_gpu_memory_mib": metrics.peak_gpu_memory_mib,
        "available_gpu_memory_mib": metrics.available_gpu_memory_mib,
        "gpu_utilization": metrics.gpu_utilization,
    }
    return result


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # benchmarks can create a large number of concurrent in-flight requests
    # so bump the file limit to make room for them
    set_ulimit()

    backend = args.backend
    model_id = args.model
    lora_id = args.lora
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.endpoint not in [
        "/v1/completions",
        "/v1/chat/completions",
        "/v2/models/ensemble/generate_stream",
    ]:
        raise ValueError(f"Unknown endpoint: {args.endpoint}")
    chat = args.endpoint == "/v1/chat/completions"

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    logger.info(f"getting tokenizer. api url: {api_url}")
    tokenizer = get_tokenizer(
        tokenizer_id, trust_remote_code=args.trust_remote_code
    )

    benchmark_dataset = BenchmarkDataset.from_flags(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )

    if (
        args.num_chat_sessions
        and not benchmark_dataset.has_multiturn_chat_support
    ):
        raise ValueError(
            f"Multiturn chat is not supported for dataset {benchmark_dataset}"
        )

    logger.info("sampling requests")
    if args.dataset_name is None:
        raise ValueError(
            "Please specify '--dataset-name' and the corresponding "
            "'--dataset-path' if required."
        )

    # Build output_lengths array
    if args.num_prompts is not None:
        num_requests = args.num_prompts
    else:
        num_requests = args.num_chat_sessions

    if args.output_lengths is None:
        output_lengths = None
    elif os.path.exists(args.output_lengths):
        with open(args.output_lengths) as f:
            output_lengths = yaml.safe_load(f)["output_lengths"]
    else:
        output_lengths = [int(args.output_lengths)] * num_requests

    input_requests: Sequence[SampledRequest] = []
    chat_sessions: Sequence[ChatSession] = []
    if isinstance(benchmark_dataset, CodeDebugBenchmarkDataset):
        # code_debug is a long-context dataset based on InfiniteBench
        if args.num_chat_sessions:
            if args.output_lengths is not None:
                raise NotImplementedError(
                    "TODO: Add support for fixed output lengths with multi-turn code-debug"
                )
            chat_sessions = benchmark_dataset.gen_twoturn_longcontext_requests(
                num_chat_sessions=args.num_chat_sessions,
                tokenizer=tokenizer,
            )
        else:
            input_requests = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    args.output_lengths is None
                    and not args.record_output_lengths
                ),
            )

    elif isinstance(benchmark_dataset, ShareGPTBenchmarkDataset):
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_lengths=output_lengths,
            shuffle=(
                args.output_lengths is None and not args.record_output_lengths
            ),
        )

    elif isinstance(benchmark_dataset, SonnetBenchmarkDataset):
        # For sonnet, formatting depends on the endpoint
        apply_chat_template = chat
        # Sample sonnet requests with common parameters
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            input_len=args.sonnet_input_len,
            output_lengths=output_lengths,
            prefix_len=args.sonnet_prefix_len,
            apply_chat_template=apply_chat_template,
            tokenizer=tokenizer,
        )

    elif isinstance(benchmark_dataset, VisionArenaBenchmarkDataset):
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            output_lengths=output_lengths,
            tokenizer=tokenizer,
        )
    elif isinstance(benchmark_dataset, ArxivSummarizationBenchmarkDataset):
        if output_lengths:
            ValueError(
                "Arxiv summarization dataset does not support --output-lengths. Please use --max-output-len"
            )
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            input_len=args.arxiv_summarization_input_len,
            max_output_len=args.max_output_len,
            shuffle=not args.record_output_lengths,
            tokenizer=tokenizer,
        )
    elif isinstance(benchmark_dataset, RandomBenchmarkDataset):
        if args.num_chat_sessions:
            chat_sessions = benchmark_dataset.gen_multiturn_random_requests(
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                num_chat_sessions=args.num_chat_sessions,
                num_turns=args.random_num_turns,
                coefficient_of_variation=args.random_coefficient_of_variation,
                tokenizer=tokenizer,
                sys_prompt_ratio=args.random_sys_prompt_ratio,
                max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                distribution_type=args.random_distribution_type,
                first_turn_ratio=args.random_first_turn_ratio,
            )
        else:
            input_requests = benchmark_dataset.sample_requests(
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                num_prompts=args.num_prompts,
                coefficient_of_variation=args.random_coefficient_of_variation,
                tokenizer=tokenizer,
                sys_prompt_ratio=args.random_sys_prompt_ratio,
                max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                distribution_type=args.random_distribution_type,
                image_size=args.random_image_size,
            )
    elif isinstance(benchmark_dataset, AxolotlBenchmarkDataset):
        input_requests = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_lengths=output_lengths,
        )
    else:
        raise ValueError(f"Unknown / unsupported dataset: {benchmark_dataset}")

    if args.print_inputs_and_outputs:
        if args.num_chat_sessions:
            raise NotImplementedError(
                "Printing out multi-turn chats is not supported."
            )

        print("Input prompts:")
        for req_id, request in enumerate(input_requests):
            print(
                {
                    "req_id": req_id,
                    "output_len": request.output_len,
                    "prompt_len": request.prompt_len,
                    "prompt": request.prompt_formatted,
                    "encoded_img": request.encoded_img,
                }
            )

    logger.info("starting benchmark run")
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            chat=chat,
            api_url=api_url,
            model_id=model_id,
            lora_id=lora_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            chat_sessions=chat_sessions,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            do_test_prompt=not args.skip_test_prompt,
            collect_gpu_stats=args.collect_gpu_stats,
            print_inputs_and_outputs=args.print_inputs_and_outputs,
            max_requests=args.num_prompts,
            num_chat_sessions=args.num_chat_sessions,
            delay_between_chat_turns=args.delay_between_chat_turns,
            ttft_skip_requests=args.ttft_skip_requests,
            max_output_len=args.max_output_len,
            temperature=args.temperature,
            max_benchmark_duration_s=args.max_benchmark_duration_s,
            warmup_delay_ms=args.chat_warmup_delay_ms,
        )
    )

    # Benchmark run failed if any failed requests
    if benchmark_result["failures"] != 0:
        logger.info("finished benchmark run: Failed.")
        sys.exit(1)

    # Save config and results to json
    if args.save_result:
        logger.info("saving results")
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = benchmark_result["completed"]
        result_json["server_args"] = args.server_args
        result_json["dataset_name"] = args.dataset_name
        result_json["client_args"] = dict(vars(args))
        # json doesn't allow infinity as numeric, so cast this to string
        result_json["client_args"]["request_rate"] = str(
            result_json["client_args"]["request_rate"]
        )

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    key = kvstring[0].strip()
                    value = kvstring[1].strip()

                    if key == "server_cpu":
                        # Map server_cpu to cpu for consistency with existing data pipeline
                        result_json["cpu"] = value
                    else:
                        result_json[key] = value
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        if args.result_filename:
            file_name = os.path.join(
                args.result_dir or "", args.result_filename
            )
        else:
            base_model_id = model_id.split("/")[-1]
            max_concurrency_str = (
                f"-concurrency{args.max_concurrency}"
                if args.max_concurrency is not None
                else ""
            )
            # When auto-generating file names, add suffixes if we have to to
            # ensure we're not overwriting an existing file (best effort,
            # subject to TOCTTOU).
            for uniq_count in itertools.count(1):
                if uniq_count == 1:
                    uniq_suffix = ""
                else:
                    uniq_suffix = f"-{uniq_count}"
                file_name = (
                    f"{backend}-{args.request_rate}qps{max_concurrency_str}-"
                    f"{base_model_id}-{current_dt}{uniq_suffix}.json"
                )
                file_name = os.path.join(args.result_dir or "", file_name)
                if not os.path.exists(file_name):
                    break
        logger.info(f"Writing file: {file_name}")
        if os.path.isfile(file_name):
            logger.warning(
                "This is going to overwrite an existing file.  "
                f"The existing file will be moved to {file_name}.orig."
            )
            os.rename(file_name, f"{file_name}.orig")
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)

    # Save output lengths if requested
    if args.record_output_lengths:
        # Save relevant input args for context
        args_to_save = (
            "backend",
            "burstiness",
            "dataset_name",
            "dataset_path",
            "endpoint",
            "max_concurrency",
            "max_output_len",
            "model",
            "request_rate",
            "seed",
            "temperature",
        )
        output_lens_dict = {}
        output_lens_dict["args"] = {x: vars(args)[x] for x in args_to_save}
        output_lens_dict["output_lengths"] = benchmark_result["output_lens"]
        with args.record_output_lengths as f:
            yaml.dump(output_lens_dict, f)

    logger.info("finished benchmark run: Success.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="modular",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    # TODO: Use a str Enum for dataset_names.
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=list(DATASET_REGISTRY.keys()),
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--lora",
        type=str,
        required=False,
        help="Name of the lora.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=(
            "Name or path of the tokenizer, if not using the default tokenizer."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--max-benchmark-duration-s",
        type=int,
        default=None,
        help="Maximum duration of the benchmark in seconds. If specified, the benchmark will run for max_benchmark_duration_s seconds. It works if max_concurrency is not None. If not specified, the benchmark will run for num_prompts requests.",
    )
    parser.add_argument(
        "--num-chat-sessions",
        type=int,
        default=None,
        help="Number of multiturn chat sessions to spawn. Single turn mode if not specified / set to None.",
    )
    parser.add_argument(
        "--delay-between-chat-turns",
        type=int,
        default=None,
        help="Optional delay before sending next chat turn request in ms.",
    )
    parser.add_argument(
        "--record-output-lengths",
        type=argparse.FileType("w"),
        metavar="/path/to/save/outputs",
        help="Save output lengths to given file in YAML format",
    )
    parser.add_argument(
        "--output-lengths",
        type=str,
        default=None,
        metavar="/path/to/lengths.yaml | int",
        help=(
            "Path to YAML file containing list of output lengths, or an int. "
            "If an int is given, all responses are forced to the given length. "
            "Default: None"
        ),
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=None,
        help="Max output length for each request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature used for token sampling. Default: 0.0",
    )
    parser.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=(
            "Number of input tokens per request, used only for sonnet dataset."
        ),
    )
    parser.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=(
            "Number of prefix tokens per request, used only for sonnet dataset."
        ),
    )
    parser.add_argument(
        "--arxiv-summarization-input-len",
        type=int,
        default=15000,
        help="Number of input tokens per request, used only for arxiv-summarization dataset.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=(
            "Number of input tokens per request, used only for random sampling."
        ),
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=(
            "Number of output tokens per request, used only for random"
            " sampling."
        ),
    )
    parser.add_argument(
        "--random-coefficient-of-variation",
        metavar="[input,]<output>",
        type=str,
        default="0.3,0.7",
        help=(
            "Coefficient of variation for input/output length, used only for random sampling.\n"
            "<input> is optional; default the same ratio as output.\n"
            "<output> is required."
        ),
    )
    parser.add_argument(
        "--random-image-size",
        type=str,
        default="",
        help=(
            "Size of random images to generate. Defaults to None (which does not generate images)."
        ),
    )
    parser.add_argument(
        "--random-sys-prompt-ratio",
        type=float,
        default=0.0,
        help=(
            "Ratio to determine the system prompt length, used only for random sampling."
        ),
    )
    parser.add_argument(
        "--random-first-turn-ratio",
        type=float,
        default=1.0,
        help="Ratio of the length of the first turn to the length of subsequent turns (default: 1.0).",
    )
    parser.add_argument(
        "--random-max-num-unique-sys-prompt",
        type=int,
        default=1,
        help=(
            "Maximum number of unique system prompts, used only for random sampling."
        ),
    )
    parser.add_argument(
        "--random-distribution-type",
        type=str,
        default="normal",
        choices=["uniform", "normal"],
        help="Type of probability distribution for sampled input/output length, used only for random sampling.",
    )
    parser.add_argument(
        "--random-num-turns",
        type=int,
        default=1,
        help="Number of turns per session, used only for random sampling and --num-chat-sessions.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help=(
            "Number of requests per second. If this is inf, "
            "then all the requests are sent at time 0. "
            "Otherwise, we use Poisson process to synthesize "
            "the request arrival times."
        ),
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument(
        "--ttft-skip-requests",
        type=int,
        default=0,
        help=(
            "Number of requests to skip when measuring TTFT latencies; i.e, "
            "ignore the first N requests in TTFT calculations. This mitigates "
            "the effect of artificial queuing, particularly at infinite request "
            "rates. Defaults to zero."
        ),
    )
    parser.add_argument(
        "--chat-warmup-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Delay between starting chat sessions when the number of active "
            "sessions is below max_concurrency, in ms.  This prevents all "
            "of the chat sessions from starting at the same time.  Default: 0"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--skip-test-prompt",
        action="store_true",
        help="Skip the test prompt.  Useful when doing external profiling.",
    )
    parser.add_argument(
        "--collect-gpu-stats",
        action="store_true",
        help="Collect GPU stats with NVML (NVIDIA only).",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help=(
            "Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
            "for metadata of this run to be saved in the result JSON file "
            "for record keeping purposes."
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help=(
            "Specify directory to save benchmark json results."
            "If not specified, results are saved in the current directory."
        ),
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help=(
            "Specify the filename to save benchmark json results."
            "If not specified, results will be saved in "
            "{backend}-{args.request_rate}qps(-concurrency{args.max_concurrency})?-{base_model_id}-{current_dt}.json"
            " format. (Note that, concurrency exists in filename when args.max_concurrency is specified.)"
        ),
    )
    parser.add_argument(
        "--print-inputs-and-outputs",
        action="store_true",
        help="Print all input and outputs to console.",
    )

    parser.add_argument(
        "--server-args",
        type=str,
        default="",
        help="Server args",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
