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

"""Benchmark online serving throughput."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeGuard
from urllib.parse import urlparse
from uuid import uuid4

try:
    from asyncio import TaskGroup  # type: ignore[attr-defined]  # added in 3.11
except ImportError:
    from taskgroup import TaskGroup  # Python < 3.11 backport

import numpy as np
import yaml
from cyclopts import App, Parameter
from cyclopts.config import Env
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from max.benchmark.benchmark_shared.server_metrics import ParsedMetrics
    from max.diagnostics.gpu import BackgroundRecorder as GPUBackgroundRecorder
    from max.diagnostics.gpu import GPUStats

from max.benchmark.benchmark_shared.config import (
    CACHE_RESET_ENDPOINT_MAP,
    PIXEL_GEN_DEFAULT_ENDPOINT,
    PIXEL_GENERATION_ENDPOINTS,
    PIXEL_GENERATION_TASKS,
    Backend,
    BenchmarkTask,
    Endpoint,
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.datasets import (
    AgenticCodeBenchmarkDataset,
    ArxivSummarizationBenchmarkDataset,
    AxolotlBenchmarkDataset,
    BatchJobBenchmarkDataset,
    BenchmarkDataset,
    ChatSession,
    CodeDebugBenchmarkDataset,
    InstructCoderBenchmarkDataset,
    LocalImageBenchmarkDataset,
    ObfuscatedConversationsBenchmarkDataset,
    RandomBenchmarkDataset,
    SampledRequest,
    ShareGPTBenchmarkDataset,
    SonnetBenchmarkDataset,
    SyntheticPixelBenchmarkDataset,
    VisionArenaBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets.multiturn_distribution_fit import (
    resolve_constant_delay_ms,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
    PixelGenerationSampledRequest,
    RequestSamples,
    Samples,
)
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkMetrics,
    LoRAMetrics,
    PixelGenerationBenchmarkMetrics,
    PixelGenerationBenchmarkResult,
    SpecDecodeMetrics,
    SpecDecodeStats,
    StandardPercentileMetrics,
    SteadyStateResult,
    TextGenerationBenchmarkResult,
    ThroughputMetrics,
    calculate_spec_decode_stats,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    BaseRequestFuncOutput,
    PixelGenerationRequestFuncInput,
    PixelGenerationRequestFuncOutput,
    ProgressBarRequestDriver,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
    get_request_driver_class,
    measured_window_duration,
)
from max.benchmark.benchmark_shared.server_metrics import (
    collect_benchmark_metrics,
    fetch_spec_decode_metrics,
    print_server_metrics,
)
from max.benchmark.benchmark_shared.steady_state import detect_steady_state
from max.benchmark.benchmark_shared.utils import (
    argmedian,
    get_tokenizer,
    int_or_none,
    is_castable_to_int,
    parse_comma_separated,
    print_section,
    set_ulimit,
)
from max.diagnostics.cpu import (
    CPUMetrics,
    CPUMetricsCollector,
    collect_pids_for_port,
)
from max.diagnostics.gpu import GPUDiagContext

BENCHMARK_SERVING_ARGPARSER_DESCRIPTION = (
    "This command runs comprehensive benchmark tests on a model server to"
    " measure performance metrics including throughput, latency, and resource"
    " utilization. Make sure that the MAX server is running and hosting a model"
    " before running this command."
)

logger = logging.getLogger(__name__)

# Server and client token counts can legitimately differ slightly because the
# server applies the chat template and may count special tokens differently.
# 5% tolerates small systematic gaps while still surfacing real mismatches
# (e.g. wrong tokenizer, prompt truncation, or model-side reprocessing).
_INPUT_TOKEN_DISCREPANCY_THRESHOLD = 0.05


def compute_output_len(
    tokenizer: PreTrainedTokenizerBase,
    output: RequestFuncOutput,
) -> int:
    return len(
        tokenizer.encode(
            output.generated_text,
            add_special_tokens=False,
        )
    )


def _prepend_run_prefix_to_formatted_prompt(
    prompt: str | list[dict[str, Any]],
    run_prefix: str,
) -> str | list[dict[str, Any]]:
    """Return a new prompt with `run_prefix` prepended to the first message."""
    if isinstance(prompt, str):
        return run_prefix + prompt

    # Chat format: prepend to the text content of the first message.
    # content may be a plain string or a list of typed content blocks.
    if not prompt:
        raise ValueError("run_prefix: empty prompt list")
    msg = prompt[0]
    content = msg.get("content")
    if isinstance(content, str):
        new_msg: dict[str, Any] = {**msg, "content": run_prefix + content}
    elif isinstance(content, list):
        text_block_idx = next(
            (
                idx
                for idx, block in enumerate(content)
                if isinstance(block, dict) and block.get("type") == "text"
            ),
            None,
        )
        if text_block_idx is None:
            raise ValueError(
                "run_prefix: no text block found in content list; cannot"
                " prepend run prefix"
            )
        new_block = {
            **content[text_block_idx],
            "text": run_prefix + str(content[text_block_idx].get("text", "")),
        }
        new_content: list[Any] = [
            *content[:text_block_idx],
            new_block,
            *content[text_block_idx + 1 :],
        ]
        new_msg = {**msg, "content": new_content}
    else:
        raise ValueError(
            "run_prefix: unsupported prompt shape for first message"
        )
    return [new_msg, *prompt[1:]]


def parse_response_format(
    response_format_arg: str | None,
) -> dict[str, Any] | None:
    """Parse response format from CLI arg (inline JSON or @filepath).

    Args:
        response_format_arg: Either a JSON string or '@path/to/schema.json' to load
            from file. If None, returns None.

    Returns:
        Parsed response format dictionary, or None if input is None.

    Raises:
        ValueError: If the JSON is invalid or the file cannot be read.
    """
    if response_format_arg is None:
        return None

    if response_format_arg.startswith("@"):
        # Load from file
        file_path = response_format_arg[1:]
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise ValueError(
                f"Response format file not found: {file_path}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in response format file {file_path}: {e}"
            ) from e

    # Parse inline JSON
    try:
        return json.loads(response_format_arg)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response format: {e}") from e


def get_default_trace_path() -> str:
    """Get the default trace output path."""
    workspace_path = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_path:
        return os.path.join(workspace_path, "profile.nsys-rep")
    return "./profile.nsys-rep"


def assert_nvidia_gpu() -> None:
    """Raise an exception if no NVIDIA GPUs are available."""
    with GPUDiagContext() as ctx:
        stats = ctx.get_stats()
        if not stats:
            raise RuntimeError(
                "No GPUs detected. The --trace flag currently only works with NVIDIA GPUs."
            )
        if not any(gpu_name.startswith("nv") for gpu_name in stats):
            raise RuntimeError(
                "The --trace flag currently only works with NVIDIA GPUs. "
                f"Found GPUs: {list(stats.keys())}"
            )


def start_trace(output_path: str, session_name: str | None = None) -> None:
    """Start nsys profiling session."""
    cmd = ["nsys", "start", "-o", output_path, "--force-overwrite", "true"]
    if session_name:
        cmd.extend(["--session", session_name])
    logger.info(f"Starting nsys trace: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def stop_trace(session_name: str | None = None) -> None:
    """Stop nsys profiling session."""
    cmd = ["nsys", "stop"]
    if session_name:
        cmd.extend(["--session", session_name])
    logger.info(f"Stopping nsys trace: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


async def get_request(
    input_requests: Sequence[SampledRequest],
    request_rate: float,
    timing_data: dict[str, list[float]],
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
        timing_data:
            Dictionary where timing data will be collected with keys:
            - 'intervals': List of actual time intervals between requests
    """

    # Calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)

    # Initialize timing data collection - always enabled
    timing_data.setdefault("intervals", [])

    start_time = time.perf_counter()
    last_request_time = start_time

    for request in input_requests:
        current_time = time.perf_counter()

        # Record timestamp when request is yielded
        if last_request_time != start_time:
            actual_interval = current_time - last_request_time
            timing_data["intervals"].append(actual_interval)

        yield request

        # Update last_request_time for next iteration
        last_request_time = current_time

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the gamma distribution.
        # If burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def build_single_turn_request_input(
    *,
    benchmark_task: BenchmarkTask,
    request: SampledRequest,
    model_id: str,
    lora_id: str | None,
    api_url: str,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_output_len: int | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> BaseRequestFuncInput:
    request_model_id = model_id if lora_id is None else lora_id
    if benchmark_task == "text-generation":
        max_tokens = min(
            filter(None, (request.output_len, max_output_len)),
            default=None,
        )
        prompt = request.prompt_formatted
        prompt_len = request.prompt_len
        if run_prefix:
            prompt = _prepend_run_prefix_to_formatted_prompt(prompt, run_prefix)
            prompt_len = prompt_len + run_prefix_len
        return RequestFuncInput(
            model=request_model_id,
            session_id=None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prompt=prompt,
            images=request.encoded_images,
            api_url=api_url,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            ignore_eos=request.ignore_eos,
            response_format=request.response_format,
        )
    if benchmark_task in PIXEL_GENERATION_TASKS:
        if not isinstance(request, PixelGenerationSampledRequest):
            raise TypeError(
                "pixel-generation benchmark requires PixelGenerationSampledRequest."
            )
        prompt = request.prompt_formatted
        if run_prefix and isinstance(prompt, str):
            prompt = run_prefix + prompt
        return PixelGenerationRequestFuncInput(
            model=request_model_id,
            session_id=None,
            prompt=prompt,
            input_image_paths=request.input_image_paths,
            api_url=api_url,
            image_options=request.image_options,
        )
    raise ValueError(f"Unsupported benchmark task: {benchmark_task}")


def print_lora_benchmark_results(metrics: LoRAMetrics) -> None:
    """Print LoRA benchmark statistics if available."""

    print_section(title=" LoRA Adapter Benchmark Results ", char="=")
    print("{:<40} {:<10}".format("Total LoRA loads:", metrics.total_loads))
    print("{:<40} {:<10}".format("Total LoRA unloads:", metrics.total_unloads))

    def print_float_metric(metric: str, value: float) -> None:
        print(f"{metric:<40} {value:<10.2f}")

    def print_action_section(action: str, times_ms: list[float]) -> None:
        if not times_ms:
            return
        print_section(title=f"LoRA {action.title()} Times")
        print_float_metric(f"Mean {action} time:", statistics.mean(times_ms))
        print_float_metric(
            f"Median {action} time:", statistics.median(times_ms)
        )
        print_float_metric(f"Min {action} time:", min(times_ms))
        print_float_metric(f"Max {action} time:", max(times_ms))
        if len(times_ms) > 1:
            print_float_metric(
                f"Std dev {action} time:", statistics.stdev(times_ms)
            )

    print_action_section("load", metrics.load_times_ms)
    print_action_section("unload", metrics.unload_times_ms)


def print_benchmark_summary(
    metrics: BenchmarkMetrics | PixelGenerationBenchmarkMetrics,
    request_rate: float,
    max_concurrency: int | None,
    achieved_request_rate: float,
    collect_gpu_stats: bool,
    collect_cpu_stats: bool,
    spec_decode_stats: SpecDecodeStats | None = None,
    lora_manager: LoRABenchmarkManager | None = None,
) -> None:
    """Print benchmark summary for text-generation and pixel-generation."""

    # 1. Print common benchmark summary
    print_section(title=" Serving Benchmark Result ", char="=")
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failures))
    print(
        "{:<40} {:<10.2f}".format("Benchmark duration (s):", metrics.duration)
    )
    if isinstance(metrics, BenchmarkMetrics):
        print(
            "{:<40} {:<10}".format("Total input tokens:", metrics.total_input)
        )
        print(
            "{:<40} {:<10}".format(
                "Total generated tokens:", metrics.total_output
            )
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
    elif isinstance(metrics, PixelGenerationBenchmarkMetrics):
        print(
            "{:<40} {:<10}".format(
                "Total generated outputs:", metrics.total_generated_outputs
            )
        )
    offline_benchmark = math.isinf(request_rate) and max_concurrency is None
    print(
        "{:<40} {:<10.5f}".format(
            "Input request rate (req/s):",
            float("inf") if offline_benchmark else achieved_request_rate,
        )
    )
    print(
        "{:<40} {:<10.5f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print("{:<40} {:<10}".format("Max Concurrency:", metrics.max_concurrency))
    if (
        isinstance(metrics, BenchmarkMetrics)
        and metrics.max_concurrent_conversations is not None
    ):
        print(
            "{:<40} {:<10}".format(
                "Max Concurrent Conversations:",
                metrics.max_concurrent_conversations,
            )
        )

    if isinstance(metrics, BenchmarkMetrics):
        print_section(title="Client Experience Metrics")
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
        print(
            "{:<40} {:<10.2f}".format(
                "Global Cached Token Rate:",
                metrics.global_cached_token_rate * 100,
            )
            + "%"
        )
        print_section(title="Time to First Token")
        print(metrics.ttft_ms.format_with_prefix(prefix="TTFT", unit="ms"))
        print_section(title="Time per Output Token (excl. 1st token)")
        print(metrics.tpot_ms.format_with_prefix(prefix="TPOT", unit="ms"))
        print_section(title="Inter-token Latency")
        print(metrics.itl_ms.format_with_prefix(prefix="ITL", unit="ms"))
        if metrics.per_turn_cached_token_rate is not None:
            print_section(title="Per-Turn Cached Token Rate")
            print(
                metrics.per_turn_cached_token_rate.format_with_prefix(
                    prefix="Per-Turn Cached Token Rate", unit="%"
                )
            )

    print_section(title="Per-Request E2E Latency")
    print(
        metrics.latency_ms.format_with_prefix(
            prefix="Request Latency", unit="ms"
        )
    )

    if isinstance(metrics, BenchmarkMetrics):
        print_section(title="Token Stats")
        print("{:<40} {:<10}".format("Max input tokens:", metrics.max_input))
        print("{:<40} {:<10}".format("Max output tokens:", metrics.max_output))
        print("{:<40} {:<10}".format("Max total tokens:", metrics.max_total))

    if spec_decode_stats is not None:
        print_section(title="Speculative Decoding")
        if spec_decode_stats.acceptance_rate is not None:
            print(
                "{:<40} {:<10.2f}".format(
                    "Acceptance rate (%):", spec_decode_stats.acceptance_rate
                )
            )
        if spec_decode_stats.acceptance_length is not None:
            print(
                "{:<40} {:<10.2f}".format(
                    "Acceptance length:", spec_decode_stats.acceptance_length
                )
            )
        if spec_decode_stats.num_drafts is not None:
            print(
                "{:<40} {:<10}".format(
                    "Drafts:", int(spec_decode_stats.num_drafts)
                )
            )
        if spec_decode_stats.draft_tokens is not None:
            print(
                "{:<40} {:<10}".format(
                    "Draft tokens:", int(spec_decode_stats.draft_tokens)
                )
            )
        if spec_decode_stats.accepted_tokens is not None:
            print(
                "{:<40} {:<10}".format(
                    "Accepted tokens:", int(spec_decode_stats.accepted_tokens)
                )
            )
        per_pos_rates = spec_decode_stats.per_position_acceptance_rates
        if per_pos_rates:
            print("Per-position acceptance (%):")
            for pos, rate in enumerate(per_pos_rates):
                print(
                    "{:<40} {:<10.2f}".format(f"  Position {pos}:", rate * 100)
                )

    # Print GPU and CPU statistics
    if collect_gpu_stats and metrics.peak_gpu_memory_mib:
        print_section(title="GPU Statistics")
        for gpu_id in range(len(metrics.peak_gpu_memory_mib)):
            print(
                "{:<40} {:<10.2f}".format(
                    f"GPU {gpu_id} peak memory (MiB):",
                    metrics.peak_gpu_memory_mib[gpu_id],
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    f"GPU {gpu_id} available memory (MiB):",
                    metrics.available_gpu_memory_mib[gpu_id],
                )
            )
            print(
                "{:<40} {:<10.2f}".format(
                    f"GPU {gpu_id} utilization (%):",
                    metrics.gpu_utilization[gpu_id],
                )
            )
    if collect_cpu_stats:
        cpu = metrics.cpu_metrics
        print_section(title="CPU Statistics")
        print(
            "{:<40} {:<10.2f}".format(
                "CPU utilization user (%):",
                cpu.user_percent if cpu else 0.0,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "CPU utilization system (%):",
                cpu.system_percent if cpu else 0.0,
            )
        )
    print("=" * 50)
    if lora_manager:
        print_lora_benchmark_results(lora_manager.metrics)
    for label, pm in metrics.metrics_by_endpoint.items():
        if len(metrics.metrics_by_endpoint) > 1:
            print(f"\n--- Metrics: {label} ---")
        print_server_metrics(pm)
    print("=" * 50)


def _parse_metadata(metadata: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` metadata strings into a flat dict.

    The special key ``server_cpu`` is mapped to ``cpu``.
    """
    result: dict[str, str] = {}
    for item in metadata or ():
        if "=" not in item:
            raise ValueError(
                "Invalid metadata format. Please use KEY=VALUE format."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        result["cpu" if key == "server_cpu" else key] = value
    return result


def hash_string(s: str) -> str:
    """Hash a string using SHA-256. This is stable and deterministic across runs.

    hexdigest is a 64-character string of hexadecimal digits. We only return the
    first 8 characters to keep the output concise.
    """
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def elide_data_uris_in_string(data_uri: str) -> str:
    """Elides the base64 data URIs parts of the string.

    Eg: elide_data_uris_in_string("'image': 'data:image/jpeg;base64,/9j/4AAQSASDEEAE'")
                               -> "'image': 'data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)...'"
    """

    def _match_replacer(m: re.Match[str]) -> str:
        uri_prefix = m.group(1)
        uri_data = m.group(2)
        return f"{uri_prefix}...(hash: {hash_string(uri_data)}, {len(uri_data)} bytes)..."

    return re.sub(
        r"(data:[a-z/]+;base64,)([A-Za-z0-9+/=]+)",
        _match_replacer,
        data_uri,
    )


def print_input_prompts(samples: Samples) -> None:
    """Helper function to print input prompts."""
    if isinstance(samples, ChatSamples):
        raise NotImplementedError(
            "Printing out multi-turn chats is not supported."
        )

    print("Input prompts:")
    for req_id, request in enumerate(samples.requests):
        prompt_info = {
            "req_id": req_id,
            "output_len": request.output_len,
            "prompt_len": request.prompt_len,
            "prompt": request.prompt_formatted,
            "encoded_images": request.encoded_images,
        }
        # We turn the entire prompt_info dict into a string and then elide the
        # data URIs. The alternative approach of only applying the transformation
        # to a stringified version of the `request.prompt_formatted` field will
        # lead to double-escaping of special characters which is not desirable.
        print(elide_data_uris_in_string(str(prompt_info)))


def _format_distribution_table(
    values: Sequence[float | int], label: str
) -> str:
    """Format distribution statistics as a mini-table with header and value rows."""
    _STAT_COLUMNS = (
        "min",
        "max",
        "mean",
        "std",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
    )
    _COL_WIDTH = 10

    arr = np.array(values, dtype=float)
    p5, p25, p50, p75, p95, p99 = np.percentile(arr, [5, 25, 50, 75, 95, 99])
    stat_values = (
        np.min(arr),
        np.max(arr),
        np.mean(arr),
        np.std(arr),
        p5,
        p25,
        p50,
        p75,
        p95,
        p99,
    )
    header = "    " + "".join(f"{name:<{_COL_WIDTH}}" for name in _STAT_COLUMNS)
    row = "    " + "".join(f"{v:<{_COL_WIDTH}.2f}" for v in stat_values)
    return f"  {label}:\n{header}\n{row}"


def print_workload_stats(samples: Samples) -> None:
    """Print workload distribution statistics and exit.

    For single-turn workloads, prints input/output length stats.
    For multi-turn workloads, additionally prints num_turns and
    delay_until_next_message stats.
    """
    print_section(title=" Workload Statistics ", char="=")

    if isinstance(samples, RequestSamples):
        input_lens = [r.prompt_len for r in samples.requests]
        output_lens = [
            r.output_len for r in samples.requests if r.output_len is not None
        ]

        print(f"  {'Total requests:':<30} {len(samples.requests)}")
        print()
        print(_format_distribution_table(input_lens, "Input length"))
        if output_lens:
            print()
            print(_format_distribution_table(output_lens, "Output length"))
        else:
            print()
            print("  Output length:  not specified (server-determined)")

    elif isinstance(samples, ChatSamples):
        sessions = samples.chat_sessions
        num_turns_list = [s.num_turns for s in sessions]

        all_input_lens: list[int] = []
        all_output_lens: list[int] = []
        all_delays: list[float] = []

        for session in sessions:
            current_context_length = 0
            for msg in session.messages:
                current_context_length += msg.num_tokens
                if msg.source == "user":
                    all_input_lens.append(current_context_length)
                else:
                    all_output_lens.append(msg.num_tokens)
                if msg.delay_until_next_message is not None:
                    all_delays.append(msg.delay_until_next_message)

        total_prefix = sum(s.prefix_turns for s in sessions)
        total_turns = sum(num_turns_list)
        print(f"  {'Total sessions:':<30} {len(sessions)}")
        if total_prefix > 0:
            print(
                f"  {'Total turns (measured):':<30}"
                f" {total_turns - total_prefix}"
            )
            print(f"  {'Total turns (warmup):':<30} {total_prefix}")
        else:
            print(f"  {'Total turns (across all):':<30} {total_turns}")
        print()
        print(
            _format_distribution_table(
                all_input_lens, "Input length (per turn)"
            )
        )
        print()
        print(
            _format_distribution_table(
                all_output_lens, "Output length (per turn)"
            )
        )
        print()
        print(
            _format_distribution_table(num_turns_list, "Num turns per session")
        )
        if all_delays:
            print()
            print(
                _format_distribution_table(
                    all_delays, "Delay between chat turns (ms)"
                )
            )
        else:
            print()
            print("  Delay until next msg:  none configured")

    print("=" * 50)


def _warn_on_request_failures(
    outputs: Sequence[BaseRequestFuncOutput],
    completed: int,
    failures: int,
    failed_responses: Sequence[BaseRequestFuncOutput],
) -> None:
    if len(outputs) == 0:
        warnings.warn(
            "No responses were received from the server.", stacklevel=2
        )

    if failures != 0:
        warnings.warn(
            (
                "Some requests failed. The responses returned are displayed "
                "below. Please check server logs for more information."
            ),
            stacklevel=2,
        )
        for failed_response in failed_responses:
            logger.error(f"Failed :: {failed_response}")

    if completed == 0:
        warnings.warn(
            (
                "All requests failed. This is likely due to a misconfiguration "
                "on the benchmark arguments."
            ),
            stacklevel=2,
        )


def _aggregate_gpu_stats(
    collect_gpu_stats: bool,
    gpu_metrics: list[dict[str, GPUStats]] | None,
) -> tuple[list[float], list[float], list[float]]:
    peak_gpu_memory_mib: list[float] = []
    available_gpu_memory_mib: list[float] = []
    gpu_utilization: list[float] = []

    if not collect_gpu_stats or not gpu_metrics:
        return peak_gpu_memory_mib, available_gpu_memory_mib, gpu_utilization

    # Simplification: We assume that whatever devices are available at the
    # start of benchmarking stays the same throughout the run. If someone is
    # hotplugging GPUs during a benchmark this may not be true.
    all_devices = list(gpu_metrics[0].keys())
    if not all_devices:
        logger.warning("No GPUs found, so there are no GPU stats to report")
        return peak_gpu_memory_mib, available_gpu_memory_mib, gpu_utilization

    bytes_per_mib = 1024 * 1024
    for device_name in all_devices:
        peak_gpu_memory_mib.append(
            max(
                snapshot[device_name].memory.used_bytes
                for snapshot in gpu_metrics
            )
            / bytes_per_mib
        )
        available_gpu_memory_mib.append(
            min(
                snapshot[device_name].memory.free_bytes
                for snapshot in gpu_metrics
            )
            / bytes_per_mib
        )
        gpu_utilization.append(
            statistics.mean(
                snapshot[device_name].utilization.gpu_usage_percent
                for snapshot in gpu_metrics
            )
        )

    return peak_gpu_memory_mib, available_gpu_memory_mib, gpu_utilization


def calculate_metrics(
    outputs: Sequence[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    skip_first_n_requests: int,
    skip_last_n_requests: int,
    max_concurrency: int | None,
    max_concurrent_conversations: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
) -> BenchmarkMetrics:
    actual_output_lens: list[int] = []
    failures = 0
    failed_responses: list[RequestFuncOutput] = []
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    latencies: list[float] = []
    input_throughputs: list[float] = []
    output_throughputs: list[float] = []
    per_turn_cached_token_rates: list[float] = []
    total_server_cached_tokens: int = 0
    total_server_prompt_tokens: int = 0

    successful: list[tuple[RequestFuncOutput, int]] = []
    for o in outputs:
        if o.cancelled:
            continue
        if o.success:
            successful.append((o, compute_output_len(tokenizer, o)))
        else:
            actual_output_lens.append(0)
            failures += 1
            failed_responses.append(o)

    total_successful = len(successful)

    for _, output_len in successful:
        actual_output_lens.append(output_len)

    # Pick head / tail to drop using request timing rather than dispatch
    # order. For multi-turn the flat list arrives in
    # ``[session0_turns, session1_turns, ...]`` block order, so a
    # dispatch-order slice would silently target the wrong requests.
    # Sorting by submit time for the head and complete time for the tail
    # gives "first N sent, last N completed" uniformly across single-turn
    # and multi-turn flows.
    head_drop_ids: set[int] = set()
    tail_drop_ids: set[int] = set()
    if skip_first_n_requests > 0:
        by_submit = sorted(
            successful,
            key=lambda pair: pair[0].request_submit_time or 0.0,
        )
        head_drop_ids = {
            id(pair[0]) for pair in by_submit[:skip_first_n_requests]
        }
    if skip_last_n_requests > 0:
        by_complete = sorted(
            successful,
            key=lambda pair: pair[0].request_complete_time or 0.0,
        )
        tail_drop_ids = {
            id(pair[0]) for pair in by_complete[-skip_last_n_requests:]
        }
    measured = [
        pair
        for pair in successful
        if id(pair[0]) not in head_drop_ids and id(pair[0]) not in tail_drop_ids
    ]

    # Aggregate token/chunk stats over the measured slice only. Skipped
    # warmup/tail requests contribute neither their tokens nor their wall
    # time to throughput metrics, so TPM-style numbers reflect the
    # intended steady-state portion of the run.
    total_input_client_calculated = 0
    total_output = 0
    nonempty_response_chunks = 0
    max_input = 0
    max_output = 0
    max_total = 0
    for o, output_len in measured:
        total_input_client_calculated += o.prompt_len
        total_output += output_len
        nonempty_response_chunks += 1 if o.ttft != 0 else 0
        nonempty_response_chunks += len(o.itl)
        max_input = max(max_input, o.prompt_len)
        max_output = max(max_output, output_len)
        max_total = max(max_total, o.prompt_len + output_len)

        tpots += o.tpot
        itls += o.itl
        ttfts.append(o.ttft)
        if o.ttft > 0:
            input_throughputs.append(o.prompt_len / o.ttft)
        if (o.latency - o.ttft) > 0:
            output_throughputs.append((output_len - 1) / (o.latency - o.ttft))
        latencies.append(o.latency)
        if o.server_token_stats.prompt_tokens:
            per_turn_cached_token_rates.append(
                o.server_token_stats.cached_tokens
                / o.server_token_stats.prompt_tokens
            )
            total_server_cached_tokens += o.server_token_stats.cached_tokens
            total_server_prompt_tokens += o.server_token_stats.prompt_tokens

    if not measured:
        total_input = 0
    elif total_server_prompt_tokens == 0:
        warnings.warn(
            "Server did not report prompt_tokens; using client-calculated token"
            " counts. Input token count and cache rate metrics may not be accurate.",
            stacklevel=2,
        )
        total_input = total_input_client_calculated
    else:
        discrepancy = (
            abs(total_server_prompt_tokens - total_input_client_calculated)
            / total_input_client_calculated
        )
        if discrepancy > _INPUT_TOKEN_DISCREPANCY_THRESHOLD:
            warnings.warn(
                f"Server-reported total input tokens ({total_server_prompt_tokens})"
                f" differs from client-calculated count ({total_input_client_calculated})"
                f" by {discrepancy:.1%}. Using server-reported value.",
                stacklevel=2,
            )
        total_input = total_server_prompt_tokens
        logger.info(
            "Using server-reported prompt_tokens for total input token count."
        )

    _warn_on_request_failures(
        outputs=outputs,
        completed=total_successful,
        failures=failures,
        failed_responses=failed_responses,
    )

    measured_count = len(measured)
    if measured_count == 0 and total_successful > 0:
        warnings.warn(
            (
                f"All {total_successful} successful requests were excluded"
                f" by skip_first_n_requests={skip_first_n_requests} (first"
                f" submitted) and skip_last_n_requests={skip_last_n_requests}"
                " (last completed). Consider running a longer benchmark."
            ),
            stacklevel=2,
        )
    elif 0 < measured_count < 10:
        warnings.warn(
            (
                f"Only {measured_count} requests remain after skipping the"
                f" first {skip_first_n_requests} submitted and last"
                f" {skip_last_n_requests} completed."
                " Results may not be reliable."
                " Consider running a longer benchmark."
            ),
            stacklevel=2,
        )

    # Duration over the measured window: first measured submit to last
    # measured complete. Mirrors the steady-state block's window math so
    # skipped warmup/tail wall time does not pollute throughput.
    measured_duration = measured_window_duration(
        (o for o, _ in measured), fallback=dur_s
    )

    (
        peak_gpu_memory_mib,
        available_gpu_memory_mib,
        gpu_utilization,
    ) = _aggregate_gpu_stats(
        collect_gpu_stats=collect_gpu_stats,
        gpu_metrics=gpu_metrics,
    )

    global_cached_token_rate: float = (
        total_server_cached_tokens / total_input if total_input > 0 else 0.0
    )
    per_turn_cached_token_rate: StandardPercentileMetrics | None = (
        StandardPercentileMetrics(
            per_turn_cached_token_rates, scale_factor=100.0, unit="%"
        )
        if len(per_turn_cached_token_rates) > 0
        else None
    )

    metrics = BenchmarkMetrics(
        duration=measured_duration,
        completed=measured_count,
        failures=failures,
        total_input=total_input,
        total_output=total_output,
        nonempty_response_chunks=nonempty_response_chunks,
        max_concurrency=max_concurrency or len(outputs),
        max_concurrent_conversations=max_concurrent_conversations,
        request_throughput=measured_count / measured_duration,
        # Use specialized metric classes that handle percentile calculations automatically
        input_throughput=ThroughputMetrics(
            input_throughputs or [float("nan")], unit="tok/s"
        ),
        output_throughput=ThroughputMetrics(
            output_throughputs or [float("nan")], unit="tok/s"
        ),
        ttft_ms=StandardPercentileMetrics(
            ttfts or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        tpot_ms=StandardPercentileMetrics(
            tpots or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        itl_ms=StandardPercentileMetrics(
            itls or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        latency_ms=StandardPercentileMetrics(
            latencies or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        max_input=max_input,
        max_output=max_output,
        max_total=max_total,
        global_cached_token_rate=global_cached_token_rate,
        per_turn_cached_token_rate=per_turn_cached_token_rate,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_metrics=cpu_metrics,
        metrics_by_endpoint=metrics_by_endpoint or {},
        skip_first_n_requests=skip_first_n_requests,
        skip_last_n_requests=skip_last_n_requests,
        input_lens=[o.prompt_len for o in outputs],
        output_lens=actual_output_lens,
        ttfts=[o.ttft for o in outputs],
        itls=[o.itl for o in outputs],
        generated_texts=[o.generated_text for o in outputs],
        errors=[o.error for o in outputs],
        request_submit_times=[o.request_submit_time for o in outputs],
        request_complete_times=[o.request_complete_time for o in outputs],
        per_turn_cached_token_rates=per_turn_cached_token_rates,
    )

    # Override TPOT mean with weighted average: sum(ITL) / decode_tokens.
    # Decode tokens = measured output - measured count, since each
    # request's first token is prefill (TTFT), not decode.
    decode_tokens = total_output - measured_count
    if decode_tokens > 0 and itls:
        metrics.tpot_ms._metrics.mean = sum(itls) / decode_tokens * 1000.0

    return metrics


def calculate_pixel_generation_metrics(
    outputs: Sequence[PixelGenerationRequestFuncOutput],
    dur_s: float,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    max_concurrency: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
) -> PixelGenerationBenchmarkMetrics:
    completed = 0
    failures = 0
    latencies: list[float] = []
    total_generated_outputs = 0
    failed_responses: list[PixelGenerationRequestFuncOutput] = []
    successful: list[PixelGenerationRequestFuncOutput] = []

    for output in outputs:
        if output.cancelled:
            continue
        if output.success:
            completed += 1
            latencies.append(output.latency)
            total_generated_outputs += output.num_generated_outputs
            successful.append(output)
        else:
            failures += 1
            failed_responses.append(output)

    _warn_on_request_failures(
        outputs=outputs,
        completed=completed,
        failures=failures,
        failed_responses=failed_responses,
    )
    (
        peak_gpu_memory_mib,
        available_gpu_memory_mib,
        gpu_utilization,
    ) = _aggregate_gpu_stats(
        collect_gpu_stats=collect_gpu_stats,
        gpu_metrics=gpu_metrics,
    )

    # Use the first-submit -> last-complete window so setup/teardown
    # around the actual requests doesn't inflate the denominator.
    measured_duration = measured_window_duration(successful, fallback=dur_s)

    return PixelGenerationBenchmarkMetrics(
        duration=measured_duration,
        completed=completed,
        failures=failures,
        max_concurrency=max_concurrency or len(outputs),
        request_throughput=completed / measured_duration,
        total_generated_outputs=total_generated_outputs,
        latency_ms=StandardPercentileMetrics(
            latencies or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_metrics=cpu_metrics,
        metrics_by_endpoint=metrics_by_endpoint or {},
        latencies=[o.latency for o in outputs],
        num_generated_outputs=[o.num_generated_outputs for o in outputs],
        errors=[o.error for o in outputs],
        request_submit_times=[o.request_submit_time for o in outputs],
        request_complete_times=[o.request_complete_time for o in outputs],
    )


async def chat_session_driver(
    model_id: str,
    api_url: str,
    request_driver: RequestDriver,
    request_counter: RequestCounter,
    chat_session: ChatSession,
    max_chat_len: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    ignore_first_turn_stats: bool = False,
    benchmark_should_end_time: int | None = None,
    randomize_session_start: bool = False,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> list[RequestFuncOutput]:
    request_func_input = RequestFuncInput(
        model=model_id,
        session_id=str(chat_session.id),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        prompt=[],
        images=[],
        api_url=api_url,
        prompt_len=0,
        max_tokens=0,
        ignore_eos=True,
    )
    content_idx = 0  # Assume user initiates the conversation

    session_outputs: list[RequestFuncOutput] = []
    message_history: list[dict[str, Any]] = []
    chat_len = 0

    messages = chat_session.messages
    prefix_end_idx = chat_session.prefix_turns * 2
    applied_initial_sleep = False

    # Build prefix turns locally (no server round-trips). The first
    # measured turn sends the full history for KV cache prefill.
    while content_idx < prefix_end_idx and content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
        output_len = messages[content_idx + 1].num_tokens
        if chat_len + output_len > max_chat_len:
            logger.warning(
                f"Session {chat_session.id}: prefix exceeded max chat"
                f" length {max_chat_len}, no measured turns possible"
            )
            break

        user_prompt = messages[content_idx].content
        message_history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        )
        # Synthetic placeholder for the assistant response.
        assistant_content = messages[content_idx + 1].content
        if not assistant_content:
            assistant_content = " ".join(["token"] * max(output_len, 1))
        message_history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_content}],
            }
        )
        chat_len += output_len
        content_idx += 2

    # If prefix exhausted the chat length budget, skip measured turns.
    if content_idx < prefix_end_idx:
        return session_outputs

    while content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
        if content_idx == 0 and run_prefix:
            chat_len += run_prefix_len
        output_len = messages[content_idx + 1].num_tokens
        if chat_len + output_len > max_chat_len:
            logger.warning(
                f"Ending conversation: hitting max chat length {max_chat_len}"
            )
            break

        advance_request = request_counter.advance_until_max()
        if not advance_request:  # reached max_requests
            break

        user_prompt = messages[content_idx].content
        if content_idx == 0 and run_prefix:
            user_prompt = run_prefix + user_prompt
        message_history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        )
        request_func_input.prompt = message_history
        request_func_input.prompt_len = chat_len
        request_func_input.max_tokens = output_len

        if not applied_initial_sleep:
            applied_initial_sleep = True
            if randomize_session_start:
                delay_ms = messages[content_idx + 1].delay_until_next_message
                if delay_ms and delay_ms > 0:
                    await asyncio.sleep(random.uniform(0, delay_ms) / 1000)

        if (
            benchmark_should_end_time is not None
            and time.perf_counter_ns() >= benchmark_should_end_time
        ):
            response = RequestFuncOutput(
                cancelled=True, request_submit_time=time.perf_counter()
            )
        else:
            raw_response = await request_driver.request(request_func_input)
            if not isinstance(raw_response, RequestFuncOutput):
                raise TypeError(
                    "Expected RequestFuncOutput in text-generation benchmark flow."
                )
            response = raw_response

        if not (ignore_first_turn_stats and content_idx == prefix_end_idx):
            session_outputs.append(response)

        if not response.success:
            if not response.cancelled:
                logger.error(
                    f"Ending chat session {chat_session.id} due to server"
                    f" error response: {response.error}"
                )
            break

        message_history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response.generated_text}],
            }
        )
        chat_len += output_len

        if delay_ms := messages[content_idx + 1].delay_until_next_message:
            await asyncio.sleep(delay_ms / 1000)

        content_idx += 2

    return session_outputs


async def run_single_turn_benchmark(
    *,
    input_requests: Sequence[SampledRequest],
    benchmark_task: BenchmarkTask,
    request_rate: float,
    burstiness: float,
    timing_data: dict[str, list[float]] | None,
    semaphore: asyncio.Semaphore | None,
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    max_output_len: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    lora_manager: LoRABenchmarkManager | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> list[BaseRequestFuncOutput]:
    """Run single-turn benchmark scenario."""
    if timing_data is None:
        timing_data = {}

    async def limited_request_func(
        request_func_input: BaseRequestFuncInput,
    ) -> BaseRequestFuncOutput:
        if semaphore is None:
            return await request_driver.request(request_func_input)
        async with semaphore:
            if (
                benchmark_should_end_time is not None
                and time.perf_counter_ns() >= benchmark_should_end_time
            ):
                return request_func_input.get_output_type()(
                    cancelled=True, request_submit_time=time.perf_counter()
                )
            return await request_driver.request(request_func_input)

    tasks: list[asyncio.Task[BaseRequestFuncOutput]] = []
    request_idx = 0
    async for request in get_request(
        input_requests, request_rate, timing_data, burstiness
    ):
        # If we've hit the time limit, then don't issue any more requests
        if benchmark_should_end_time is not None:
            if time.perf_counter_ns() >= benchmark_should_end_time:
                break

        # Determine which LoRA to use for this request
        lora_id = None
        if lora_manager:
            lora_id = lora_manager.get_lora_for_request(request_idx)

        request_func_input = build_single_turn_request_input(
            benchmark_task=benchmark_task,
            request=request,
            model_id=model_id,
            lora_id=lora_id,
            api_url=api_url,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_len=max_output_len,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )
        tasks.append(
            asyncio.create_task(limited_request_func(request_func_input))
        )
        request_idx += 1

    outputs = await asyncio.gather(*tasks)

    return outputs


async def prime_prefix_turns(
    sessions: Sequence[ChatSession],
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    max_chat_len: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_sessions: int | None = None,
) -> None:
    """Prime the server's KV cache for sessions with prefix turns.

    Sends one request per session with the full prefix context and
    max_tokens=1. Runs before the benchmark timer so priming doesn't
    affect measured throughput or duration.

    Sessions beyond ``max_sessions`` are skipped because the multiturn /
    kv-cache-stress runners reset ``prefix_turns=0`` for them anyway
    (they represent new conversations arriving mid-benchmark).
    """
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    sessions_with_prefix = [s for s in sessions if s.prefix_turns > 0]
    if not sessions_with_prefix:
        return

    logger.info(
        f"Priming prefix turns for {len(sessions_with_prefix)} sessions..."
    )

    async def _prime_session(session: ChatSession) -> None:
        messages = session.messages
        prefix_end_idx = session.prefix_turns * 2
        message_history: list[dict[str, Any]] = []
        chat_len = 0
        content_idx = 0
        while content_idx < prefix_end_idx and content_idx + 1 < len(messages):
            chat_len += messages[content_idx].num_tokens
            output_len = messages[content_idx + 1].num_tokens
            if chat_len + output_len > max_chat_len:
                break
            message_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": messages[content_idx].content,
                        }
                    ],
                }
            )
            assistant_content = messages[content_idx + 1].content
            if not assistant_content:
                assistant_content = " ".join(["token"] * max(output_len, 1))
            message_history.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_content}],
                }
            )
            chat_len += output_len
            content_idx += 2
        if message_history:
            prime_input = RequestFuncInput(
                model=model_id,
                session_id=str(session.id),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                prompt=message_history,
                images=[],
                api_url=api_url,
                prompt_len=chat_len,
                max_tokens=1,
                ignore_eos=True,
            )
            await request_driver.request(prime_input)

    await asyncio.gather(*(_prime_session(s) for s in sessions_with_prefix))
    logger.info("Prefix turns priming complete.")


def systematic_probability_proportional_to_size(
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick ``k`` distinct indices into ``weights`` with inclusion probability
    ``min(1, k * weights[i] / sum(weights))``.

    Algorithm (systematic PPS with iterated-cap handling):

    1. Lay items end-to-end on a number line of length ``W = sum(weights)``,
       item ``i`` occupying an interval of length ``weights[i]``.
    2. Place ``k`` equally-spaced ticks at ``u, u + W/k, ..., u + (k-1)*W/k``
       with a single shared offset ``u ~ Uniform(0, W/k)``. Each tick lands
       in exactly one item's interval; that item is selected.
    3. If any item has ``weights[i] >= W/k``, its interval is at least as wide
       as the tick spacing and would be hit by 2+ ticks (a duplicate).
       Pre-include such items, remove them from the pool, recompute the
       threshold over the residual weight and residual tick count, and
       repeat. Then run systematic PPS over the residual.

    Why this removes the depletion bias of ``rng.choice(replace=False, p=...)``:
    sequential PPSWOR (numpy's no-replacement weighted sampling) over-includes
    small items because once a heavy item is drawn the remaining weights
    renormalize. Systematic PPS, by contrast, gives ``pi_i = k*weights[i]/W``
    *exactly* for any uncapped item: its interval is shorter than the tick
    spacing, so at most one tick can land in it, with probability equal to
    the interval length divided by the spacing. Plugging in:
    ``E[mean] = (1/k) * sum_i weights[i] * pi_i = sum_i weights[i]^2 / W``
    = the size-biased mean. No depletion, no convergence rate — equality
    by construction.

    Caps still leak some bias because pre-included items contribute
    ``T_i * 1`` instead of the ideal ``T_i * (k * T_i / W)``. ``cap_count == 0``
    is the condition for analytically zero bias; callers should warn the user
    when caps occur.
    """
    residual = np.asarray(weights, dtype=np.float64).copy()
    n = len(residual)
    chosen: list[int] = []

    # Iterated cap: any item with weight >= W_remaining / k_remaining is
    # wider than the tick spacing and would be hit by multiple ticks.
    # Pre-include it (its ideal pi >= 1 anyway) and recompute the threshold
    # over the residual.
    while True:
        remaining_k = k - len(chosen)
        if remaining_k <= 0:
            break
        residual_sum = float(residual.sum())
        if residual_sum <= 0.0:
            break
        cap = residual_sum / remaining_k
        over = np.where(residual >= cap)[0]
        if len(over) == 0:
            break
        chosen.extend(int(i) for i in over)
        residual[over] = 0.0

    # Run systematic PPS over the residual with the remaining ticks.
    remaining_k = k - len(chosen)
    if remaining_k > 0:
        cum = np.cumsum(residual)
        residual_sum = float(cum[-1])
        if residual_sum > 0.0:
            step = residual_sum / remaining_k
            offset = float(rng.uniform(0.0, step))
            ticks = offset + np.arange(remaining_k) * step
            picks = np.searchsorted(cum, ticks)
            # Guard the float-edge case where a tick equals the total sum.
            picks = np.minimum(picks, n - 1)
            chosen.extend(int(i) for i in picks)

    return np.asarray(chosen[:k], dtype=np.int64)


@dataclass
class _WarmupSamplingReport:
    """Statistics about a warmup pick, logged at the start of a run."""

    # Size of the warmup-candidate sub-pool. Equals ``factor * warmup_count``;
    # bigger pools leave more cap headroom.
    warmup_pool: int
    # Size of the unbiased main sub-pool — the actual benchmark sessions.
    # Untouched by the warmup pick so it preserves natural P(T).
    main_pool: int
    # Number of in-flight warmup slots (M) seeded at t=0.
    warmup_count: int
    # The configured ``warmup_oversample_factor`` for this run.
    factor: int
    # Closed-form size-biased mean: E[T_live] = sum(T**2)/sum(T).
    target_mean: float
    # Realized mean of the sampled warmup picks.
    realized_mean: float
    # Conservative stdev of ``realized_mean`` around ``target_mean``,
    # treating each pick as an independent length-biased draw:
    #   sb_var = sum(T**3)/sum(T) - target_mean**2
    #   stdev  = sqrt(sb_var / K)
    # Systematic PPS picks are negatively correlated, so the true stdev is
    # smaller (typically ~half). Used to report ``|realized - target|`` as
    # a unitless ratio in stdev units; under this conservative bound,
    # anything below ~1 is unambiguously noise.
    realized_mean_stdev: float
    # Candidates whose ideal proportional inclusion probability would exceed 1
    # (T_i > W/K). Systematic PPS pre-includes these with pi=1, which is the
    # best a no-replacement scheme can do but introduces a small residual
    # bias relative to ``target_mean``.
    cap_count: int


def _pick_warmup_population(
    chat_sessions: Sequence[ChatSession],
    warmup_count: int,
    *,
    warmup_to_steady_state: bool,
    warmup_oversample_factor: int,
    main_pool_target: int,
    rng: np.random.Generator,
) -> tuple[list[ChatSession], _WarmupSamplingReport | None]:
    """Build the runner's task list with warmup picks at the head.

    When ``warmup_to_steady_state=True`` and ``warmup_count > 0``, the
    helper picks ``warmup_count`` warmup sessions from the leading
    ``factor * warmup_count`` candidates (clamped to what's available)
    and assigns each a random ``prefix_turns`` in ``[0, T-1)``. The
    remaining trailing slice is the main benchmark sessions (untouched,
    preserves natural P(T)). For ``factor >= 2`` the picks are length-biased
    via :func:`systematic_probability_proportional_to_size`, which gives
    inclusion probability ``min(1, K * T_i / sum(T))`` exactly — no
    depletion bias. For ``factor < 2`` we don't have headroom for a
    weighted draw so the picks are uniform; the report's target/realized
    split lets users see the residual bias.

    Returns ``(reordered_sessions, report)``. ``report`` is ``None``
    only when warmup is off (``warmup_to_steady_state=False`` or
    ``warmup_count == 0``).
    """
    n_total = len(chat_sessions)
    warmup_count = max(0, warmup_count)
    main_pool_target = max(0, main_pool_target)

    if not warmup_to_steady_state or warmup_count == 0:
        return list(chat_sessions), None

    # Try to oversample (factor*M candidates) without eating into the
    # user-requested main pool. If the dataset under-produced too much for
    # that, fall back to just M candidates (no oversampling, just
    # randomized start turn) — we still need M sessions to seed the
    # initial concurrent batch. Under-production isn't warned about
    # directly: if it matters, the cap-count warning below fires.
    ideal_candidate_pool = warmup_oversample_factor * warmup_count
    available_for_oversampling = max(0, n_total - main_pool_target)
    candidate_pool = min(ideal_candidate_pool, available_for_oversampling)
    if candidate_pool < warmup_count:
        candidate_pool = min(warmup_count, n_total)
    # Main pool gets whatever is left, up to its target.
    main_count = min(max(0, n_total - candidate_pool), main_pool_target)

    actual_warmup_count = min(warmup_count, candidate_pool)
    candidates = chat_sessions[:candidate_pool]
    main_sessions = list(
        chat_sessions[candidate_pool : candidate_pool + main_count]
    )

    turn_counts = np.array(
        [max(1, s.num_turns) for s in candidates], dtype=np.int64
    )
    # Length-biased only when factor>=2 AND we have headroom to pick from.
    # Otherwise fall back to a plain uniform pick — at factor<2 we don't
    # have enough candidates above ``actual_warmup_count`` to do a
    # meaningful weighted draw, so the report's target/realized split tells
    # the user about the residual bias.
    use_length_bias = (
        warmup_oversample_factor >= 2 and candidate_pool > actual_warmup_count
    )
    if use_length_bias:
        warmup_idx = systematic_probability_proportional_to_size(
            turn_counts, actual_warmup_count, rng
        )
    else:
        warmup_idx = rng.choice(
            candidate_pool, size=actual_warmup_count, replace=False
        )

    warmup_sessions: list[ChatSession] = []
    for i in warmup_idx:
        s = candidates[int(i)]
        total_turns = max(1, s.num_turns)
        prefix_turns = (
            int(rng.integers(0, total_turns)) if total_turns > 1 else 0
        )
        warmup_sessions.append(
            dataclasses.replace(s, prefix_turns=prefix_turns)
        )

    # ``target_mean`` is the size-biased mean of the *full* dataset (warmup
    # candidates + main pool), so it reflects steady-state for the workload
    # as a whole — not just the candidate slice the picker drew from.
    full_turn_counts = np.array(
        [max(1, s.num_turns) for s in chat_sessions],
        dtype=np.int64,
    )
    full_sum = float(full_turn_counts.sum())
    target_mean = float((full_turn_counts**2).sum() / full_sum)
    realized_mean = float(turn_counts[list(warmup_idx)].mean())

    # Per-draw stdev on ``realized_mean`` under a with-replacement bound.
    # Variance of a single length-biased pick is
    #   E[T^2 | size-bias] - sb_mean^2 = sum(T^3)/sum(T) - target_mean^2;
    # K independent picks scale Var by 1/K. Systematic PPS picks are
    # negatively correlated, so this overestimates by ~2x — that's in the
    # safe direction: if ``|realized - target| / stdev`` is below ~1 even
    # under this conservative bound, it's unambiguously noise.
    sb_var = (full_turn_counts**3).sum() / full_sum - target_mean**2
    sb_var = max(0.0, float(sb_var))
    realized_mean_stdev = float(np.sqrt(sb_var / max(1, actual_warmup_count)))

    cap_threshold = float(turn_counts.sum()) / actual_warmup_count
    cap_count = int((turn_counts > cap_threshold).sum())

    report = _WarmupSamplingReport(
        warmup_pool=candidate_pool,
        main_pool=len(main_sessions),
        warmup_count=actual_warmup_count,
        factor=warmup_oversample_factor,
        target_mean=target_mean,
        realized_mean=realized_mean,
        realized_mean_stdev=realized_mean_stdev,
        cap_count=cap_count,
    )
    return warmup_sessions + main_sessions, report


def _log_warmup_sampling_report(report: _WarmupSamplingReport) -> None:
    """Emit the per-run [warmup-sampling] log + cap-triggered warning."""

    def _pct(value: float, ref: float) -> str:
        if ref == 0:
            return "n/a"
        return f"{100.0 * (value - ref) / ref:+.1f}%"

    if report.realized_mean_stdev > 0:
        stdev_ratio = (
            abs(report.realized_mean - report.target_mean)
            / report.realized_mean_stdev
        )
        stdev_str = f"{stdev_ratio:.2f} stdev from target"
    else:
        stdev_str = "stdev n/a"

    logger.info(
        "[warmup-sampling] warmup_pool=%d main_pool=%d M=%d factor=%d\n"
        "  target mean from samples:                      %.2f\n"
        "  realized warmup mean (one draw):               %.2f  (%s, %s)\n"
        "  always-picked sessions (too long for pool):    %d / %d",
        report.warmup_pool,
        report.main_pool,
        report.warmup_count,
        report.factor,
        report.target_mean,
        report.realized_mean,
        _pct(report.realized_mean, report.target_mean),
        stdev_str,
        report.cap_count,
        report.warmup_pool,
    )

    if report.cap_count > 0:
        logger.warning(
            "Could not warmup to steady state: %d session(s) are too long for "
            "a candidate pool of %d, so they get picked every time and bias "
            "the warmup. Increase --warmup-oversample-factor (currently %d) "
            "to enlarge the pool.",
            report.cap_count,
            report.warmup_pool,
            report.factor,
        )


async def run_multiturn_benchmark(
    *,
    chat_sessions: Sequence[ChatSession],
    max_requests: int,
    semaphore: asyncio.Semaphore | None,
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    tokenizer: PreTrainedTokenizerBase,
    ignore_first_turn_stats: bool,
    lora_manager: LoRABenchmarkManager | None,
    warmup_delay_ms: float,
    max_concurrency: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    randomize_session_start: bool = False,
    warmup_to_steady_state: bool = False,
    warmup_oversample_factor: int = 0,
    num_chat_sessions: int = 0,
    seed: int | None = None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> dict[str, list[RequestFuncOutput]]:
    """Run multi-turn chat benchmark scenario."""

    # Track total sent requests among chat sessions
    request_counter = RequestCounter(
        max_requests=max_requests,
        total_sent_requests=0,
    )

    # apply the semaphore at the session level
    # ex: with max_concurrency = 1,
    # the first session finishes before the second session starts
    async def limited_chat_session_driver(
        chat_session: ChatSession,
        session_idx: int,
    ) -> tuple[str, list[RequestFuncOutput]]:
        # Determine which LoRA to use for this chat session
        lora_id = None
        if lora_manager:
            lora_id = lora_manager.get_lora_for_request(session_idx)

        if semaphore is None:
            outputs = await chat_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=request_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_chat_len=tokenizer.model_max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ignore_first_turn_stats=ignore_first_turn_stats,
                benchmark_should_end_time=benchmark_should_end_time,
                randomize_session_start=randomize_session_start,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
        else:
            async with semaphore:
                outputs = await chat_session_driver(
                    model_id=model_id if lora_id is None else lora_id,
                    api_url=api_url,
                    request_driver=request_driver,
                    request_counter=request_counter,
                    chat_session=chat_session,
                    max_chat_len=tokenizer.model_max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    ignore_first_turn_stats=ignore_first_turn_stats,
                    benchmark_should_end_time=benchmark_should_end_time,
                    randomize_session_start=randomize_session_start,
                    run_prefix=run_prefix,
                    run_prefix_len=run_prefix_len,
                )
        session_id = (
            str(chat_session.id)
            if chat_session.id is not None
            else f"anonymous-{session_idx}"
        )
        return session_id, outputs

    sessions = list(chat_sessions)
    if warmup_to_steady_state:
        warmup_count = max_concurrency or len(chat_sessions)
        sessions, report = _pick_warmup_population(
            chat_sessions,
            warmup_count,
            warmup_to_steady_state=True,
            warmup_oversample_factor=warmup_oversample_factor,
            main_pool_target=num_chat_sessions or len(chat_sessions),
            rng=np.random.default_rng(seed),
        )
        if report is not None:
            _log_warmup_sampling_report(report)

    tasks: list[asyncio.Task[tuple[str, list[RequestFuncOutput]]]] = []
    for idx, chat_session in enumerate(sessions):
        if warmup_delay_ms > 0 and max_concurrency and idx < max_concurrency:
            await asyncio.sleep(warmup_delay_ms / 1000)
        tasks.append(
            asyncio.create_task(limited_chat_session_driver(chat_session, idx))
        )

    outputs_by_session: dict[str, list[RequestFuncOutput]] = dict(
        await asyncio.gather(*tasks)
    )

    if (
        benchmark_should_end_time is not None
        and time.perf_counter_ns() < benchmark_should_end_time
    ):
        logger.warning(
            "All chat sessions completed before the time limit. "
            "Consider increasing --num-chat-sessions for more stable load."
        )

    return outputs_by_session


class _ConcurrentTurnsRequestDriver(RequestDriver):
    """Wraps a RequestDriver to cap the number of concurrent in-flight turns.

    Acquires a semaphore slot before issuing each turn request and releases it
    as soon as the response returns. Inter-turn delays (e.g. delay_until_next_message)
    fall outside the slot's hold window, so idle user-think-time does not consume
    concurrency capacity.

    With many concurrent conversations, a turn request may wait in the semaphore
    backlog long enough for the deadline to expire. Cancel it when stale.
    """

    def __init__(
        self,
        request_driver: RequestDriver,
        semaphore: asyncio.Semaphore,
        benchmark_should_end_time: int | None = None,
    ) -> None:
        super().__init__(tokenizer=request_driver.tokenizer)
        self._request_driver = request_driver
        self._semaphore = semaphore
        self._benchmark_should_end_time = benchmark_should_end_time

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> BaseRequestFuncOutput:
        async with self._semaphore:
            if (
                self._benchmark_should_end_time is not None
                and time.perf_counter_ns() >= self._benchmark_should_end_time
            ):
                return request_func_input.get_output_type()(
                    cancelled=True, request_submit_time=time.perf_counter()
                )
            return await self._request_driver.request(request_func_input)


async def run_kv_cache_stress_benchmark(
    *,
    chat_sessions: Sequence[ChatSession],
    max_requests: int,
    max_concurrent_conversations: int,
    semaphore: asyncio.Semaphore | None,
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    tokenizer: PreTrainedTokenizerBase,
    ignore_first_turn_stats: bool,
    lora_manager: LoRABenchmarkManager | None,
    warmup_delay_ms: float,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    randomize_session_start: bool = False,
    warmup_to_steady_state: bool = False,
    warmup_oversample_factor: int = 0,
    num_chat_sessions: int = 0,
    seed: int | None = None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> dict[str, list[RequestFuncOutput]]:
    """Run a KV-cache stress benchmark with independent conversation and turn concurrency.

    Two independent concurrency controls:

    - `max_concurrent_conversations`: at most this many chat sessions are
      driven at once. Workers pick up the next session from the queue when one
      finishes, growing the server's KV-cache footprint.
    - `semaphore` (`max_concurrency` in the CLI): caps the number of turn
      requests in-flight globally across all concurrent sessions. Workers that
      cannot acquire a turn slot block without sending a request; the session's
      `session_id` and client-side conversation state are preserved in the
      backlog until a slot becomes available.

    NOTE: TTFT reflects pure server-side cost (KV re-computation or reloading)
          since the timer starts only after the semaphore is acquired. Backlog
          wait reduces each session's firing cadence beyond what
          `delay_between_chat_turns` specifies — sessions are less frequent
          than configured.
    """
    request_counter = RequestCounter(
        max_requests=max_requests,
        total_sent_requests=0,
    )

    inflight_limited_driver: RequestDriver = (
        _ConcurrentTurnsRequestDriver(
            request_driver, semaphore, benchmark_should_end_time
        )
        if semaphore is not None
        else request_driver
    )

    sessions = list(chat_sessions)
    if warmup_to_steady_state:
        sessions, report = _pick_warmup_population(
            chat_sessions,
            max_concurrent_conversations,
            warmup_to_steady_state=True,
            warmup_oversample_factor=warmup_oversample_factor,
            main_pool_target=num_chat_sessions or len(chat_sessions),
            rng=np.random.default_rng(seed),
        )
        if report is not None:
            _log_warmup_sampling_report(report)

    # Queue holds (original_index, session) pairs so LoRA assignment is stable.
    session_queue: asyncio.Queue[tuple[int, ChatSession]] = asyncio.Queue()
    for idx, session in enumerate(sessions):
        await session_queue.put((idx, session))

    num_workers = min(max_concurrent_conversations, len(sessions))
    worker_outputs: list[dict[str, list[RequestFuncOutput]]] = [
        {} for _ in range(num_workers)
    ]

    async def _conversation_worker(worker_idx: int) -> None:
        # Stagger workers to avoid thundering-herd at startup.
        if warmup_delay_ms > 0:
            await asyncio.sleep(worker_idx * warmup_delay_ms / 1000)

        local_count = 0
        while True:
            try:
                idx, chat_session = session_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            lora_id = (
                lora_manager.get_lora_for_request(idx) if lora_manager else None
            )
            outputs = await chat_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=inflight_limited_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_chat_len=tokenizer.model_max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ignore_first_turn_stats=ignore_first_turn_stats,
                benchmark_should_end_time=benchmark_should_end_time,
                randomize_session_start=randomize_session_start,
                run_prefix=run_prefix,
                run_prefix_len=run_prefix_len,
            )
            session_id = (
                str(chat_session.id)
                if chat_session.id is not None
                else f"anonymous-w{worker_idx}-{local_count}"
            )
            local_count += 1
            worker_outputs[worker_idx].setdefault(session_id, []).extend(
                outputs
            )

    async with TaskGroup() as tg:
        for i in range(num_workers):
            tg.create_task(_conversation_worker(i))

    outputs_by_session: dict[str, list[RequestFuncOutput]] = {}
    for worker_dict in worker_outputs:
        for sid, outs in worker_dict.items():
            outputs_by_session.setdefault(sid, []).extend(outs)
    return outputs_by_session


def create_benchmark_pbar(disable_tqdm: bool, samples: Samples) -> tqdm | None:
    """Create a progress bar for benchmark runs.

    Args:
        disable_tqdm: Whether to disable the progress bar.
        samples: Samples that will be benchmarked with.

    Returns:
        A tqdm progress bar instance or None if disabled.
    """
    if disable_tqdm:
        return None

    if isinstance(samples, RequestSamples):
        # single-turn chat scenario
        return tqdm(total=len(samples.requests))
    else:
        # multi-turn chat scenario
        num_qa_turns = [session.num_turns for session in samples.chat_sessions]
        return tqdm(total=sum(num_qa_turns))


def _is_pixel_generation_outputs(
    outputs: Sequence[BaseRequestFuncOutput],
) -> TypeGuard[Sequence[PixelGenerationRequestFuncOutput]]:
    return all(
        isinstance(output, PixelGenerationRequestFuncOutput)
        for output in outputs
    )


def _is_text_generation_outputs(
    outputs: Sequence[BaseRequestFuncOutput],
) -> TypeGuard[Sequence[RequestFuncOutput]]:
    return all(isinstance(output, RequestFuncOutput) for output in outputs)


async def run_single_test_prompt(
    benchmark_task: BenchmarkTask,
    model_id: str,
    api_url: str,
    samples: Samples,
    request_driver: RequestDriver,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_output_len: int | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> None:
    logger.info("Starting initial single prompt test run...")
    if isinstance(samples, ChatSamples):
        test_question = samples.chat_sessions[0].messages[0]
        test_answer = samples.chat_sessions[0].messages[1]
        test_request = SampledRequest(
            prompt_formatted=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": test_question.content}
                    ],
                }
            ],
            prompt_len=test_question.num_tokens,
            output_len=test_answer.num_tokens,
            encoded_images=[],
            ignore_eos=True,
        )
        # Chat samples define their own target output length per turn.
        test_max_output_len = None
    else:
        test_request = samples.requests[0]
        test_max_output_len = max_output_len

    test_input = build_single_turn_request_input(
        benchmark_task=benchmark_task,
        request=test_request,
        model_id=model_id,
        lora_id=None,
        api_url=api_url,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_len=test_max_output_len,
        run_prefix=run_prefix,
        run_prefix_len=run_prefix_len,
    )
    test_output = await request_driver.request(test_input)
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


def _build_pixel_generation_result(
    *,
    outputs: Sequence[BaseRequestFuncOutput],
    benchmark_duration: float,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    max_concurrency: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
) -> PixelGenerationBenchmarkResult:
    """Compute metrics and build the result dict for pixel-generation tasks."""
    if not _is_pixel_generation_outputs(outputs):
        raise TypeError(
            "Expected all outputs to be PixelGenerationRequestFuncOutput"
            " in pixel-generation benchmark flow."
        )
    metrics = calculate_pixel_generation_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        gpu_metrics=gpu_metrics,
        cpu_metrics=cpu_metrics,
        max_concurrency=max_concurrency,
        collect_gpu_stats=collect_gpu_stats,
        metrics_by_endpoint=metrics_by_endpoint,
    )
    return PixelGenerationBenchmarkResult(metrics=metrics)


def _build_text_generation_result(
    *,
    outputs: Sequence[BaseRequestFuncOutput],
    benchmark_duration: float,
    tokenizer: PreTrainedTokenizerBase | None,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    skip_first_n_requests: int,
    skip_last_n_requests: int,
    max_concurrency: int | None,
    max_concurrent_conversations: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
    spec_decode_stats: SpecDecodeStats | None = None,
) -> TextGenerationBenchmarkResult:
    """Compute metrics and build the result dict for text-generation tasks."""
    if not _is_text_generation_outputs(outputs):
        raise TypeError(
            "Expected all outputs to be RequestFuncOutput"
            " in text-generation benchmark flow."
        )
    text_metrics = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        gpu_metrics=gpu_metrics,
        cpu_metrics=cpu_metrics,
        skip_first_n_requests=skip_first_n_requests,
        skip_last_n_requests=skip_last_n_requests,
        max_concurrency=max_concurrency,
        max_concurrent_conversations=max_concurrent_conversations,
        collect_gpu_stats=collect_gpu_stats,
        metrics_by_endpoint=metrics_by_endpoint,
    )

    for warn in text_metrics.confidence_warnings():
        logger.warning(f"Confidence: {warn}")

    steady_state_result = _compute_steady_state_result(
        outputs=outputs,
        tokenizer=tokenizer,
        gpu_metrics=gpu_metrics,
        cpu_metrics=cpu_metrics,
        max_concurrency=max_concurrency,
        max_concurrent_conversations=max_concurrent_conversations,
        collect_gpu_stats=collect_gpu_stats,
        metrics_by_endpoint=metrics_by_endpoint,
    )

    return TextGenerationBenchmarkResult(
        metrics=text_metrics,
        steady_state_result=steady_state_result,
        spec_decode_stats=spec_decode_stats,
    )


def _compute_steady_state_result(
    *,
    outputs: Sequence[RequestFuncOutput],
    tokenizer: PreTrainedTokenizerBase | None,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    max_concurrency: int | None,
    max_concurrent_conversations: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None,
) -> SteadyStateResult:
    """Detect steady-state window and return a SteadyStateResult."""
    steady = detect_steady_state(outputs, max_concurrency=max_concurrency)
    # Persist detection mode for downstream consumers; skip it when
    # detection was skipped (concurrency=1) so the default "full"
    # isn't mistaken for a real result.
    mode = (
        steady.mode if (steady.detected or steady.warning is not None) else None
    )

    ss_metrics: BenchmarkMetrics | None = None
    if steady.detected:
        ss_index_set = set(steady.steady_state_indices)
        ss_outputs = [
            out
            for i, out in enumerate(outputs)
            if i in ss_index_set and out.success and not out.cancelled
        ]
        ss_valid = [
            out
            for out in ss_outputs
            if out.request_submit_time is not None
            and out.request_complete_time is not None
        ]
        if len(ss_valid) >= 2:
            ss_valid.sort(key=lambda o: o.request_submit_time or 0.0)
            first_submit = ss_valid[0].request_submit_time
            last_complete = ss_valid[-1].request_complete_time
            assert first_submit is not None and last_complete is not None
            ss_duration = last_complete - first_submit
            ss_duration = max(ss_duration, 1e-9)

            ss_metrics = calculate_metrics(
                outputs=ss_outputs,
                dur_s=ss_duration,
                tokenizer=tokenizer,
                gpu_metrics=gpu_metrics,
                cpu_metrics=cpu_metrics,
                skip_first_n_requests=0,
                skip_last_n_requests=0,
                max_concurrency=max_concurrency,
                max_concurrent_conversations=max_concurrent_conversations,
                collect_gpu_stats=collect_gpu_stats,
                metrics_by_endpoint=metrics_by_endpoint,
            )

        # start_index and end_index are in original dispatch order and
        # may span requests filtered out by detect_steady_state (failed,
        # missing TPOT, etc.), particularly in multi-turn runs where
        # sessions interleave. Call out the valid-count separately so
        # the gap isn't mistaken for a bug.
        assert steady.start_index is not None and steady.end_index is not None
        dispatch_span = steady.end_index - steady.start_index
        # Only show dispatch_span when it differs from the valid count
        # (multi-turn interleaving); single-turn matches would be noise.
        span_note = (
            f" spans {dispatch_span} positions"
            if dispatch_span != steady.steady_state_count
            else ""
        )
        mode_note = (
            " [TTFT-only fallback; TPOT absent across run]"
            if steady.mode == "ttft_only"
            else ""
        )
        logger.info(
            f"Steady-state detected: {steady.steady_state_count} valid"
            f" requests (dispatch range [{steady.start_index},"
            f" {steady.end_index}){span_note};"
            f" {steady.total_requests} total valid in the run)"
            f"{mode_note}"
        )
    elif steady.warning:
        logger.warning(f"Steady-state detection: {steady.warning}")

    return SteadyStateResult(
        detected=steady.detected,
        start_index=steady.start_index,
        end_index=steady.end_index,
        count=steady.steady_state_count,
        warning=steady.warning,
        mode=mode,
        metrics=ss_metrics,
    )


async def prime_shared_contexts(
    model_id: str,
    api_url: str,
    samples: Samples,
    request_driver: RequestDriver,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    run_prefix: str | None = None,
    run_prefix_len: int = 0,
) -> None:
    """Warm up prefix caching by sending each shared context for prefilling."""
    warmup_entries = samples.shared_contexts

    if not warmup_entries:
        logger.warning(
            "shared_contexts is empty; the prefix cache could not be primed."
            " Check that --random-sys-prompt-ratio > 0 and input lengths are"
            " sufficient to produce a non-trivial shared context."
        )
        return

    logger.info(
        f"Warming prefix cache with {len(warmup_entries)}"
        " unique shared context(s)..."
    )

    is_chat = isinstance(samples, ChatSamples)
    warmup_inputs: list[RequestFuncInput] = []
    for entry in warmup_entries:
        warmup_prompt: str | list[dict[str, Any]]
        if is_chat:
            warmup_prompt = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": entry.text}],
                }
            ]
        else:
            warmup_prompt = entry.text

        if run_prefix:
            warmup_prompt = _prepend_run_prefix_to_formatted_prompt(
                warmup_prompt, run_prefix
            )

        warmup_inputs.append(
            RequestFuncInput(
                model=model_id,
                session_id=None,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                prompt=warmup_prompt,
                images=[],
                api_url=api_url,
                prompt_len=entry.num_tokens + run_prefix_len,
                max_tokens=1,
                ignore_eos=True,
            )
        )

    warmup_results: list[BaseRequestFuncOutput | None] = [None] * len(
        warmup_inputs
    )

    async def _run_warmup_index(idx: int, inp: RequestFuncInput) -> None:
        warmup_results[idx] = await request_driver.request(inp)

    warmup_start = time.perf_counter()
    async with TaskGroup() as tg:
        for idx, inp in enumerate(warmup_inputs):
            tg.create_task(_run_warmup_index(idx, inp))
    warmup_elapsed_s = time.perf_counter() - warmup_start
    for sys_idx, inp in enumerate(warmup_inputs):
        result = warmup_results[sys_idx]
        if result is None:
            raise RuntimeError(
                f"Warmup task {sys_idx} did not produce a result (this is a bug)"
            )
        if not result.success:
            raise ValueError(
                f"Shared context warmup request failed at index {sys_idx}:"
                f" (prompt: (SKIPPED), prompt_len: {inp.prompt_len}),"
                f" error: {result.error}"
            )

    logger.info(
        "Prefix cache warmup completed and took %.2f seconds.",
        warmup_elapsed_s,
    )


async def benchmark(
    backend: Backend,
    benchmark_task: BenchmarkTask,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase | None,
    samples: Samples,
    request_rate: float,
    burstiness: float,
    max_concurrency: int | None,
    max_concurrent_conversations: int | None,
    disable_tqdm: bool,
    do_test_prompt: bool,
    warm_shared_prefix: bool,
    collect_gpu_stats: bool,
    collect_cpu_stats: bool,
    collect_server_stats: bool,
    metrics_urls: Mapping[str, str],
    print_inputs_and_outputs: bool,
    max_requests: int,
    skip_first_n_requests: int,
    skip_last_n_requests: int,
    max_output_len: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_benchmark_duration_s: int | None,
    warmup_delay_ms: float,
    ignore_first_turn_stats: bool,
    randomize_session_start: bool,
    warmup_to_steady_state: bool,
    warmup_oversample_factor: int,
    num_chat_sessions: int,
    seed: int | None,
    timing_data: dict[str, list[float]] | None,
    lora_manager: LoRABenchmarkManager | None,
    trace_path: str | None = None,
    trace_session: str | None = None,
    force_unique_runs: bool = False,
) -> TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult:
    if (
        ignore_first_turn_stats
        and skip_first_n_requests
        and not warmup_to_steady_state
    ):
        # Without --warmup-to-steady-state, sessions all start at turn 0,
        # so --ignore-first-turn-stats already drops the same head requests
        # that --skip-first-n-requests would target. Combining them just
        # trims deeper into the run than the user asked for.
        # With --warmup-to-steady-state, sessions begin at randomized turn
        # offsets, so the two features filter different requests and we
        # want them to compose.
        logger.warning(
            "--ignore-first-turn-stats and --skip-first-n-requests both set"
            " without --warmup-to-steady-state. --ignore-first-turn-stats"
            " already drops every session's first turn, so"
            " --skip-first-n-requests would trim deeper than expected."
            " Ignoring --skip-first-n-requests."
        )
        skip_first_n_requests = 0

    # Benchmark LoRA loading if manager provided
    if lora_manager:
        logger.info("Starting LoRA loading benchmark...")
        await lora_manager.benchmark_loading(
            api_url=base_url,
        )

    # Generate a single run-level unique prefix so all requests in this run
    # share the same constant prefix. This prevents cross-run KV-cache
    # pollution while preserving within-run system-prompt prefix caching
    # (requests with the same system prompt still share a common token prefix).
    run_prefix: str | None = None
    run_prefix_len: int = 0
    if force_unique_runs:
        if benchmark_task == "image-to-image":
            raise ValueError(
                "--force-unique-runs is not supported for image-to-image:"
                " the primary input is the image, not text, and systems may"
                " cache vision embeddings independently, so we can't guarantee"
                " uniqueness across benchmark runs."
            )
        run_prefix = f"{uuid4()}: "
        if benchmark_task not in PIXEL_GENERATION_TASKS:
            # prompt_len is not tracked for pixel generation tasks, so
            # run_prefix_len is not needed there.
            assert tokenizer is not None
            run_prefix_len = len(
                tokenizer.encode(run_prefix, add_special_tokens=False)
            )

    request_driver_class: type[RequestDriver] = get_request_driver_class(
        api_url, task=benchmark_task
    )
    # Create a request driver instance without pbar for test prompt
    # (pbar will be set later for the actual benchmark runs)
    test_request_driver: RequestDriver = request_driver_class(
        tokenizer=tokenizer
    )

    if warm_shared_prefix:
        await prime_shared_contexts(
            model_id=model_id,
            api_url=api_url,
            samples=samples,
            request_driver=test_request_driver,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )

    if do_test_prompt:
        await run_single_test_prompt(
            benchmark_task=benchmark_task,
            model_id=model_id,
            api_url=api_url,
            samples=samples,
            request_driver=test_request_driver,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_len=max_output_len,
            run_prefix=run_prefix,
            run_prefix_len=run_prefix_len,
        )

    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"

    logger.info(f"Input request rate: {request_rate}")
    logger.info(f"Burstiness factor: {burstiness} ({distribution})")
    logger.info(f"Maximum request concurrency: {max_concurrency}")

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    with contextlib.ExitStack() as benchmark_stack:
        gpu_recorder: GPUBackgroundRecorder | None = None
        spec_decode_metrics_before: SpecDecodeMetrics | None = None
        spec_decode_metrics_after: SpecDecodeMetrics | None = None
        if collect_gpu_stats:
            try:
                from max.diagnostics.gpu import BackgroundRecorder
            except ImportError:
                logger.warning(
                    "max.diagnostics not available, skipping GPU stats"
                    " collection"
                )
            else:
                gpu_recorder = benchmark_stack.enter_context(
                    BackgroundRecorder()
                )

        cpu_collector = None
        if collect_cpu_stats:
            try:
                pids = collect_pids_for_port(
                    int(urlparse(api_url).port or 8000)
                )
                cpu_collector = CPUMetricsCollector(pids)
                cpu_collector.start()
            except:
                logger.warning(
                    "Cannot access max-serve PIDs, skipping CPU stats"
                    " collection"
                )

        # Start nsys trace if enabled (before timing to exclude trace overhead)
        if trace_path:
            start_trace(trace_path, trace_session)

        # Create pbar for actual benchmark runs
        pbar = create_benchmark_pbar(disable_tqdm=disable_tqdm, samples=samples)

        # Create base driver and wrap with ProgressBarRequestDriver if pbar is provided
        base_driver: RequestDriver = request_driver_class(tokenizer=tokenizer)
        request_driver: RequestDriver = (
            ProgressBarRequestDriver(base_driver, pbar)
            if pbar is not None
            else base_driver
        )

        # Prime prefix turns before the benchmark timer starts. Only the
        # initial concurrent population keeps its prefix_turns; sessions
        # arriving mid-benchmark get reset to 0 and don't need priming.
        # Bound: kv-cache-stress uses max_concurrent_conversations;
        # multiturn uses max_concurrency (may be None for unbounded, in
        # which case all sessions keep prefix_turns and are all primed).
        if isinstance(samples, ChatSamples):
            assert tokenizer is not None
            prime_bound = (
                max_concurrent_conversations
                if max_concurrent_conversations is not None
                else max_concurrency
            )
            await prime_prefix_turns(
                sessions=samples.chat_sessions,
                request_driver=base_driver,
                model_id=model_id,
                api_url=api_url,
                max_chat_len=tokenizer.model_max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_sessions=prime_bound,
            )

        # Capture baseline server metrics after priming so priming requests
        # don't affect the delta calculation.
        baseline_endpoints: Mapping[str, ParsedMetrics] = {}
        if collect_server_stats:
            try:
                baseline_endpoints = collect_benchmark_metrics(
                    metrics_urls, backend, base_url
                )
                logger.info("Captured baseline server metrics")
            except Exception as e:
                logger.warning(
                    f"Failed to capture baseline server metrics: {e}"
                )

        if benchmark_task == "text-generation":
            spec_decode_metrics_before = fetch_spec_decode_metrics(
                backend, base_url
            )

        # Marker consumed by utils/benchmarking/serving/analyze_batch_logs.py
        # to slice the batch log by concurrency and exclude warmup/test-prompt
        # phases.
        logger.info(
            f"=== BATCH LOG MARKER: Benchmark started "
            f"(max_concurrency={max_concurrency}, "
            f"request_rate={request_rate}) ==="
        )
        benchmark_start_time = time.perf_counter_ns()
        if max_benchmark_duration_s is None:
            benchmark_should_end_time = None
        else:
            benchmark_should_end_time = benchmark_start_time + int(
                max_benchmark_duration_s * 1e9
            )

        try:
            all_outputs: Sequence[BaseRequestFuncOutput]
            outputs_by_session: dict[str, list[RequestFuncOutput]] | None = None
            if isinstance(samples, RequestSamples):
                if max_concurrent_conversations is not None:
                    raise ValueError(
                        "--max-concurrent-conversations is only valid for "
                        "multi-turn workloads. Set --num-chat-sessions to "
                        "enable multi-turn mode."
                    )
                # single-turn chat scenario
                all_outputs = await run_single_turn_benchmark(
                    input_requests=samples.requests,
                    benchmark_task=benchmark_task,
                    request_rate=request_rate,
                    burstiness=burstiness,
                    timing_data=timing_data,
                    semaphore=semaphore,
                    benchmark_should_end_time=benchmark_should_end_time,
                    request_driver=request_driver,
                    model_id=model_id,
                    api_url=api_url,
                    max_output_len=max_output_len,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    lora_manager=lora_manager,
                    run_prefix=run_prefix,
                    run_prefix_len=run_prefix_len,
                )
            elif max_concurrent_conversations is not None:
                # KV-cache stress benchmark: two independent concurrency knobs.
                # max_concurrent_conversations caps active session workers;
                # max_concurrency (semaphore) caps in-flight turns globally.
                if (
                    max_concurrency is not None
                    and max_concurrency > max_concurrent_conversations
                ):
                    raise ValueError(
                        f"--max-concurrency ({max_concurrency}) must be <= "
                        f"--max-concurrent-conversations "
                        f"({max_concurrent_conversations}): to stress the "
                        "server's KV-cache, more sessions must be open than "
                        "turns in-flight."
                    )
                assert tokenizer is not None
                assert isinstance(max_concurrent_conversations, int)
                outputs_by_session = await run_kv_cache_stress_benchmark(
                    chat_sessions=samples.chat_sessions,
                    max_requests=max_requests,
                    max_concurrent_conversations=max_concurrent_conversations,
                    semaphore=semaphore,
                    benchmark_should_end_time=benchmark_should_end_time,
                    request_driver=request_driver,
                    model_id=model_id,
                    api_url=api_url,
                    tokenizer=tokenizer,
                    ignore_first_turn_stats=ignore_first_turn_stats,
                    lora_manager=lora_manager,
                    warmup_delay_ms=warmup_delay_ms,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    randomize_session_start=randomize_session_start,
                    warmup_to_steady_state=warmup_to_steady_state,
                    warmup_oversample_factor=warmup_oversample_factor,
                    num_chat_sessions=num_chat_sessions,
                    seed=seed,
                    run_prefix=run_prefix,
                    run_prefix_len=run_prefix_len,
                )
                all_outputs = [
                    out for outs in outputs_by_session.values() for out in outs
                ]
            else:
                # multi-turn chat scenario
                outputs_by_session = await run_multiturn_benchmark(
                    chat_sessions=samples.chat_sessions,
                    max_requests=max_requests,
                    semaphore=semaphore,
                    benchmark_should_end_time=benchmark_should_end_time,
                    request_driver=request_driver,
                    model_id=model_id,
                    api_url=api_url,
                    tokenizer=tokenizer,
                    ignore_first_turn_stats=ignore_first_turn_stats,
                    lora_manager=lora_manager,
                    warmup_delay_ms=warmup_delay_ms,
                    max_concurrency=max_concurrency,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    randomize_session_start=randomize_session_start,
                    warmup_to_steady_state=warmup_to_steady_state,
                    warmup_oversample_factor=warmup_oversample_factor,
                    num_chat_sessions=num_chat_sessions,
                    seed=seed,
                    run_prefix=run_prefix,
                    run_prefix_len=run_prefix_len,
                )
                all_outputs = [
                    out for outs in outputs_by_session.values() for out in outs
                ]

            # Close pbar if it was created
            if pbar is not None:
                pbar.close()

            benchmark_duration = (
                time.perf_counter_ns() - benchmark_start_time
            ) / 1e9
        finally:
            # Stop nsys trace if enabled (after timing to exclude trace overhead)
            if trace_path:
                stop_trace(trace_session)

    if benchmark_task == "text-generation":
        spec_decode_metrics_after = fetch_spec_decode_metrics(backend, base_url)
    spec_decode_stats = None
    if (
        spec_decode_metrics_before is not None
        and spec_decode_metrics_after is not None
    ):
        spec_decode_stats = calculate_spec_decode_stats(
            spec_decode_metrics_before,
            spec_decode_metrics_after,
        )

    if print_inputs_and_outputs:
        if benchmark_task == "text-generation":
            assert tokenizer is not None
            print("Generated output text:")
            for req_id, output in enumerate(all_outputs):
                assert isinstance(output, RequestFuncOutput)
                output_len = compute_output_len(tokenizer, output)
                print(
                    {
                        "req_id": req_id,
                        "output_len": output_len,
                        "output": output.generated_text,
                    }
                )
        elif benchmark_task in PIXEL_GENERATION_TASKS:
            print("Generated pixel generation outputs:")
            for req_id, output in enumerate(all_outputs):
                assert isinstance(output, PixelGenerationRequestFuncOutput)
                print(
                    {
                        "req_id": req_id,
                        "num_generated_outputs": output.num_generated_outputs,
                        "latency_s": output.latency,
                        "success": output.success,
                        "error": output.error,
                    }
                )

    if lora_manager:
        await lora_manager.benchmark_unloading(
            api_url=base_url,
        )

    gpu_metrics: list[dict[str, GPUStats]] | None = None
    if collect_gpu_stats and gpu_recorder is not None:
        gpu_metrics = gpu_recorder.stats

    cpu_metrics_result: CPUMetrics | None = None
    if collect_cpu_stats and cpu_collector is not None:
        cpu_collector.stop()
        cpu_metrics_result = cpu_collector.dump_stats()

    # Collect server-side metrics from Prometheus endpoint (with delta from baseline)
    endpoint_metrics: Mapping[str, ParsedMetrics] = {}
    if collect_server_stats:
        try:
            endpoint_metrics = collect_benchmark_metrics(
                metrics_urls, backend, base_url, baseline=baseline_endpoints
            )
            logger.info("Collected server metrics (final)")
        except Exception as e:
            logger.warning(f"Failed to collect server metrics: {e}")

    achieved_request_rate = 0.0
    if timing_data and timing_data.get("intervals"):
        mean_interval = sum(timing_data["intervals"]) / len(
            timing_data["intervals"]
        )
        achieved_request_rate = (
            round(1.0 / mean_interval, 3) if mean_interval > 0 else 0.0
        )

    result: PixelGenerationBenchmarkResult | TextGenerationBenchmarkResult
    if benchmark_task in PIXEL_GENERATION_TASKS:
        result = _build_pixel_generation_result(
            outputs=all_outputs,
            benchmark_duration=benchmark_duration,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics_result,
            max_concurrency=max_concurrency,
            collect_gpu_stats=collect_gpu_stats,
            metrics_by_endpoint=endpoint_metrics,
        )
    else:
        text_result = _build_text_generation_result(
            outputs=all_outputs,
            benchmark_duration=benchmark_duration,
            tokenizer=tokenizer,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics_result,
            skip_first_n_requests=skip_first_n_requests,
            skip_last_n_requests=skip_last_n_requests,
            max_concurrency=max_concurrency,
            max_concurrent_conversations=max_concurrent_conversations,
            collect_gpu_stats=collect_gpu_stats,
            metrics_by_endpoint=endpoint_metrics,
            spec_decode_stats=spec_decode_stats,
        )
        if outputs_by_session is not None:
            result = dataclasses.replace(
                text_result,
                session_server_stats={
                    sid: [
                        dataclasses.asdict(out.server_token_stats)
                        for out in outs
                    ]
                    for sid, outs in sorted(
                        outputs_by_session.items(),
                        key=lambda kv: _session_sort_key(kv[0]),
                    )
                },
            )
        else:
            result = dataclasses.replace(
                text_result,
                aggregate_server_stats=[
                    dataclasses.asdict(out.server_token_stats)
                    for out in all_outputs
                    if isinstance(out, RequestFuncOutput)
                ],
            )
    if lora_manager is not None:
        result.lora_metrics = lora_manager.metrics

    print_benchmark_summary(
        metrics=result.metrics,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        achieved_request_rate=achieved_request_rate,
        collect_gpu_stats=collect_gpu_stats,
        collect_cpu_stats=collect_cpu_stats,
        spec_decode_stats=spec_decode_stats,
        lora_manager=lora_manager,
    )

    return result


def validate_task_and_endpoint(
    benchmark_task: BenchmarkTask, endpoint: Endpoint
) -> None:
    if benchmark_task == "text-generation":
        if endpoint in ("/v1/responses", "/v1/images/generations"):
            raise ValueError(
                f"--benchmark-task text-generation does not support "
                f"--endpoint {endpoint}"
            )
    elif benchmark_task in PIXEL_GENERATION_TASKS:
        if endpoint not in PIXEL_GENERATION_ENDPOINTS:
            raise ValueError(
                f"--benchmark-task {benchmark_task} requires --endpoint"
                f" to be one of {sorted(PIXEL_GENERATION_ENDPOINTS)},"
                f" got {endpoint!r}"
            )


def _apply_workload_to_config(
    config: ServingBenchmarkConfig,
    workload: dict[str, Any],
) -> None:
    """Set workload YAML values as fields on *config*.

    Keys are converted from kebab-case to snake_case.  Path objects are
    stringified and env vars in string values are expanded.

    Fields already in `config.model_fields_set` (i.e. explicitly provided
    by the caller, whether via CLI args or direct construction) are left
    unchanged so that CLI values always take precedence over workload YAML.
    """
    for k, v in workload.items():
        field_name = k.replace("-", "_")
        if field_name not in ServingBenchmarkConfig.model_fields:
            logger.warning(f"Ignoring unknown workload key: {k}")
            continue
        if field_name in config.model_fields_set:
            logger.info(
                f"CLI flag --{k} takes precedence over workload YAML"
                f" (CLI: {getattr(config, field_name)!r},"
                f" workload: {v!r})"
            )
            continue
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, str):
            v = os.path.expandvars(v)
        logger.info(f"Applying workload YAML value: --{k}={v!r}")
        setattr(config, field_name, v)


def flush_prefix_cache(
    backend: Backend, host: str, port: int, dry_run: bool
) -> None:
    """Flush the serving engine's prefix cache via HTTP POST."""
    if backend not in CACHE_RESET_ENDPOINT_MAP:
        raise ValueError(
            f"Cannot flush prefix cache for {backend} backend: this backend"
            " does not support prefix cache flush."
        )
    import requests as _http_requests  # lazy - avoid hard dep for non-sweep use

    api_url = f"http://{host}:{port}{CACHE_RESET_ENDPOINT_MAP[backend]}"
    if dry_run:
        logger.info(f"Dry-run flush: POST {api_url}")
        return
    response = _http_requests.post(api_url)
    if response.status_code == 400:
        logger.warning(
            f"Prefix caching is not enabled on backend {backend} at {api_url};"
            " skipping cache flush."
        )
    elif response.status_code == 404:
        logger.warning(
            f"Prefix cache reset is not supported at {api_url} (HTTP 404);"
            " skipping cache flush."
        )
    elif response.status_code != 200:
        # Mammoth's proxy wraps engine 404s in a 502 with per-endpoint statuses
        # in the JSON body; treat unanimous 404s the same as a direct 404 above
        # (e.g. vLLM builds without /reset_prefix_cache exposed).
        try:
            body = response.json() if response.content else None
        except ValueError:
            body = None
        results = body.get("results") if isinstance(body, dict) else None
        if (
            isinstance(results, list)
            and results
            and all(
                isinstance(r, dict) and r.get("statusCode") == 404
                for r in results
            )
        ):
            logger.warning(
                f"Prefix cache reset is not supported at {api_url} "
                "(proxy reported 404 from all engine endpoints);"
                " skipping cache flush."
            )
            return
        raise RuntimeError(
            f"Failed to flush prefix cache for backend {backend} at {api_url}: "
            f"status={response.status_code} body={response.text}"
        )


@dataclass
class BenchmarkRunResult:
    """Result of one (max_concurrency, request_rate) benchmark configuration.

    Yielded by :func:`main_with_parsed_args` — one entry per (mc, rr) combo
    after median selection across ``num_iters`` iterations.
    """

    max_concurrency: int | None
    request_rate: float
    num_prompts: int
    result: (
        TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult | None
    ) = None


@dataclass
class BenchmarkSession:
    """Resolved, session-level state shared across all sweep iterations.

    Created once after argument parsing / dataset loading in
    :func:`main_with_parsed_args` and threaded into each
    :func:`_execute_benchmark` call.
    """

    benchmark_task: BenchmarkTask
    endpoint: Endpoint
    api_url: str
    base_url: str
    model_id: str
    tokenizer_id: str
    tokenizer: PreTrainedTokenizerBase | None
    samples: Samples
    lora_manager: LoRABenchmarkManager | None
    trace_path: str | None
    orig_skip_first: int | None
    orig_skip_last: int | None


def _execute_benchmark(
    args: ServingBenchmarkConfig,
    session: BenchmarkSession,
    max_concurrency: int | None,
    request_rate: float,
) -> TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult:
    """Run a single benchmark invocation.

    ``session.orig_skip_first`` / ``session.orig_skip_last`` are the
    user-supplied values (``None`` = auto-derive from *max_concurrency*).
    """
    backend: Backend = args.backend

    skip_first = session.orig_skip_first
    skip_last = session.orig_skip_last
    if request_rate != float("inf"):
        # Finite rate → steady drip with no ramp-up / ramp-down artifacts,
        # so skip nothing (PERF-878).
        if skip_first is None:
            skip_first = 0
        if skip_last is None:
            skip_last = 0
    elif max_concurrency is not None and max_concurrency > 1:
        if skip_first is None:
            skip_first = max_concurrency
            logger.info(
                f"Auto-setting skip_first_n_requests={skip_first}"
                f" (max_concurrency={max_concurrency})"
            )
        if skip_last is None:
            skip_last = max_concurrency
            logger.info(
                f"Auto-setting skip_last_n_requests={skip_last}"
                f" (max_concurrency={max_concurrency})"
            )
    # max_concurrency=1 → sequential requests, no ramp-up to trim.
    # max_concurrency=None → no cap; default to 0.
    # Both leave auto values unset → fall through to 0 below.
    if skip_first is None:
        skip_first = 0
    if skip_last is None:
        skip_last = 0

    if args.warm_shared_prefix:
        if args.dataset_name not in ("random", "synthetic"):
            raise ValueError(
                f"--warm-shared-prefix is not supported for dataset"
                f" '{args.dataset_name}'. Only random/synthetic datasets have a"
                " defined shared prefix to cache."
            )
        if args.random_sys_prompt_ratio <= 0:
            raise ValueError(
                "--warm-shared-prefix requires --random-sys-prompt-ratio > 0."
            )

    logger.info("Starting benchmark run")
    assert args.num_prompts is not None
    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            benchmark_task=session.benchmark_task,
            api_url=session.api_url,
            base_url=session.base_url,
            model_id=session.model_id,
            tokenizer=session.tokenizer,
            samples=session.samples,
            request_rate=request_rate,
            burstiness=args.burstiness,
            max_concurrency=max_concurrency,
            max_concurrent_conversations=args.max_concurrent_conversations,
            disable_tqdm=args.disable_tqdm,
            do_test_prompt=not args.skip_test_prompt,
            warm_shared_prefix=args.warm_shared_prefix,
            collect_gpu_stats=args.collect_gpu_stats,
            collect_cpu_stats=args.collect_cpu_stats,
            collect_server_stats=args.collect_server_stats,
            metrics_urls=args.metrics_urls,
            print_inputs_and_outputs=args.print_inputs_and_outputs,
            max_requests=args.num_prompts,
            skip_first_n_requests=skip_first,
            skip_last_n_requests=skip_last,
            max_output_len=args.max_output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_benchmark_duration_s=args.max_benchmark_duration_s,
            warmup_delay_ms=args.chat_warmup_delay_ms,
            ignore_first_turn_stats=args.ignore_first_turn_stats,
            randomize_session_start=args.randomize_session_start,
            warmup_to_steady_state=args.warmup_to_steady_state,
            warmup_oversample_factor=args.warmup_oversample_factor,
            num_chat_sessions=args.num_chat_sessions or 0,
            seed=args.seed,
            timing_data=None,
            lora_manager=session.lora_manager,
            trace_path=session.trace_path,
            trace_session=args.trace_session,
            force_unique_runs=args.force_unique_runs,
        )
    )

    ok, validation_errors = benchmark_result.validate_metrics()
    if not ok:
        for err in validation_errors:
            logger.error(f"Benchmark result validation failed: {err}")
        logger.info("finished benchmark run: Failed.")
        sys.exit(1)

    logger.info("finished benchmark run: Success.")
    return benchmark_result


def _session_sort_key(sid: str) -> tuple[int, int, str]:
    """Sort numeric session ids first by integer value, then anonymous ids."""
    try:
        return (0, int(sid), "")
    except ValueError:
        return (1, 0, sid)


def save_result_json(
    result_filename: str | None,
    args: ServingBenchmarkConfig,
    benchmark_result: TextGenerationBenchmarkResult
    | PixelGenerationBenchmarkResult,
    *,
    benchmark_task: BenchmarkTask,
    model_id: str,
    tokenizer_id: str,
    request_rate: float,
) -> None:
    """Persist benchmark results to *result_filename*."""
    if not result_filename:
        return
    backend: Backend = args.backend
    client_args = args.model_dump()
    client_args["request_rate"] = str(client_args["request_rate"])
    result_json: dict[str, object] = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend,
        "benchmark_task": benchmark_task,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "num_prompts": benchmark_result.metrics.completed,
        "dataset_name": args.dataset_name,
        "client_args": client_args,
        "request_rate": (
            request_rate if request_rate < float("inf") else "inf"
        ),
        "burstiness": args.burstiness,
        "max_concurrency": args.max_concurrency,
        "max_concurrent_conversations": args.max_concurrent_conversations,
        **_parse_metadata(args.metadata),
        **benchmark_result.to_result_dict(),
    }
    logger.info(f"Writing file: {result_filename}")
    if os.path.isfile(result_filename):
        logger.warning(
            "This is going to overwrite an existing file.  "
            f"The existing file will be moved to {result_filename}.orig."
        )
        os.rename(result_filename, f"{result_filename}.orig")
    with open(result_filename, "w") as outfile:
        json.dump(result_json, outfile)


def _save_output_lengths(
    args: ServingBenchmarkConfig,
    benchmark_result: TextGenerationBenchmarkResult
    | PixelGenerationBenchmarkResult,
    benchmark_task: BenchmarkTask,
) -> None:
    """Save output lengths to a YAML file if configured."""
    if not args.record_output_lengths:
        return
    if benchmark_task != "text-generation":
        logger.warning(
            "--record-output-lengths is only supported for text-generation"
        )
        return
    assert isinstance(benchmark_result, TextGenerationBenchmarkResult)
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
        "top_k",
        "top_p",
    )
    output_lens_dict: dict[str, object] = {}
    args_dict = args.model_dump()
    output_lens_dict["args"] = {x: args_dict[x] for x in args_to_save}
    output_lens_dict["output_lengths"] = benchmark_result.metrics.output_lens
    with open(args.record_output_lengths, "w") as f:
        yaml.dump(output_lens_dict, f)


def _inflated_chat_session_count(
    args: ServingBenchmarkConfig, base_session_count: int
) -> int:
    """Inflate the dataset request to ``base + factor * max_warmup_count``
    so the length-biased warmup pick has cap headroom (sweeps inflate by
    the largest requested concurrency)."""
    if not args.warmup_to_steady_state or args.warmup_oversample_factor <= 0:
        return base_session_count
    max_warmup = 0
    if args.max_concurrent_conversations is not None:
        max_warmup = max(max_warmup, args.max_concurrent_conversations)
    if args.max_concurrency:
        try:
            mcs = parse_comma_separated(args.max_concurrency, int_or_none)
            mcs_ints = [m for m in mcs if m is not None]
            if mcs_ints:
                max_warmup = max(max_warmup, max(mcs_ints))
        except Exception:
            pass
    if max_warmup <= 0:
        return base_session_count
    return base_session_count + args.warmup_oversample_factor * max_warmup


def main_with_parsed_args(
    args: ServingBenchmarkConfig,
) -> Iterator[BenchmarkRunResult]:
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    logger.info(args)

    if args.model is None:
        raise ValueError("--model is required when running benchmark")

    # ---- Workload YAML ----
    if args.workload_config:
        with open(args.workload_config) as workload_file:
            workload = yaml.safe_load(workload_file)
        # Resolve relative paths against the YAML's directory.
        for key in ("dataset-path", "output-lengths"):
            if workload.get(key) is not None:
                if is_castable_to_int(str(workload[key])):
                    continue
                path = Path(os.path.expandvars(workload[key]))
                if not path.is_absolute():
                    path = Path(args.workload_config).parent / path
                workload[key] = path
        # Resolve max_concurrency: CLI > YAML.
        yaml_max_concurrency = workload.pop("max-concurrency", None)
        if yaml_max_concurrency is not None and args.max_concurrency is None:
            args.max_concurrency = str(yaml_max_concurrency)
        # Resolve num_prompts: CLI > YAML > default (deferred).
        cli_num_prompts = args.num_prompts is not None
        yaml_num_prompts = workload.pop("num-prompts", None)
        if not cli_num_prompts:
            if yaml_num_prompts is not None:
                args.num_prompts = int(yaml_num_prompts)
        # Resolve max_benchmark_duration_s: CLI > YAML.
        w_duration = workload.pop("max-benchmark-duration-s", None)
        if w_duration is not None and args.max_benchmark_duration_s is None:
            args.max_benchmark_duration_s = int(w_duration)
        _apply_workload_to_config(args, workload)
        args.skip_test_prompt = True

    # Warn + default when nothing constrains run length (common to both paths).
    has_prompts = args.num_prompts is not None
    has_duration = args.max_benchmark_duration_s is not None
    has_multiplier = args.num_prompts_multiplier is not None
    # The multiplier dynamically computes num_prompts per-mc, but only
    # when no explicit duration also constrains the run.
    multiplier_will_resolve = has_multiplier and not has_duration
    if not has_prompts and not has_duration and not has_multiplier:
        logger.warning(
            "Neither --num-prompts nor --max-benchmark-duration-s is"
            " specified. Defaulting to --num-prompts 1000 and"
            " --max-benchmark-duration-s 300"
        )
        args.num_prompts = 1000
        args.max_benchmark_duration_s = 300
    elif not has_prompts and not multiplier_will_resolve:
        args.num_prompts = 1000

    # ---- Parse sweep ranges ----
    concurrency_range = parse_comma_separated(args.max_concurrency, int_or_none)
    request_rate_range = parse_comma_separated(args.request_rate, float)

    # When num_prompts_multiplier is active AND no explicit num_prompts or
    # duration constrains the run, dynamically compute num_prompts per
    # concurrency level.
    use_dynamic_num_prompts = (
        args.num_prompts_multiplier is not None
        and args.num_prompts is None
        and args.max_benchmark_duration_s is None
    )
    if use_dynamic_num_prompts:
        assert args.num_prompts_multiplier is not None
        max_mc = max(
            (mc for mc in concurrency_range if mc is not None), default=1
        )
        args.num_prompts = args.num_prompts_multiplier * max_mc
        # When using num_prompts_multiplier without explicit duration, default to
        # 300s timeout per MC config to prevent indefinitely long benchmark runs.
        logger.info(
            "Using --num-prompts-multiplier without --max-benchmark-duration-s."
            " Defaulting to 300s timeout per max-concurrency configuration."
        )
        args.max_benchmark_duration_s = 300

    # ``--dry-run`` falls through — handled after samples build.

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_ulimit()
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    benchmark_task: BenchmarkTask = args.benchmark_task
    endpoint: Endpoint = args.endpoint

    # Auto-select the correct endpoint for pixel generation based on the
    # backend. Each pixel-gen backend requires a specific endpoint (e.g.,
    # sglang needs /v1/images/generations, vllm needs
    # /v1/chat/completions). We auto-select when the current endpoint
    # doesn't match this backend's expected pixel-gen endpoint.
    if benchmark_task in PIXEL_GENERATION_TASKS:
        backend_key = args.backend.removesuffix("-chat")
        if backend_key in PIXEL_GEN_DEFAULT_ENDPOINT:
            expected = PIXEL_GEN_DEFAULT_ENDPOINT[backend_key]
            if endpoint != expected:
                logger.info(
                    "Auto-selected endpoint %s for backend %s"
                    " (pixel generation task)",
                    expected,
                    args.backend,
                )
                endpoint = expected
        else:
            raise ValueError(
                f"Backend {args.backend!r} does not have a default"
                f" pixel-generation endpoint. Explicitly pass --endpoint"
                f" with one of {sorted(PIXEL_GENERATION_ENDPOINTS)}."
            )

    validate_task_and_endpoint(benchmark_task, endpoint)
    # chat is only meaningful for text-generation (enables chat template
    # formatting). For pixel generation via /v1/chat/completions
    # (vllm pixel gen), the pixel-gen code path ignores this flag.
    chat = endpoint == "/v1/chat/completions"

    if args.base_url is not None:
        base_url = args.base_url
    else:
        base_url = f"http://{args.host}:{args.port}"

    api_url = f"{base_url}{endpoint}"
    tokenizer: PreTrainedTokenizerBase | None
    samples: Samples

    if benchmark_task == "text-generation":
        logger.info(f"getting tokenizer. api url: {api_url}")
        tokenizer = get_tokenizer(
            tokenizer_id,
            model_max_length=args.model_max_length,
            trust_remote_code=args.trust_remote_code,
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

        # Build output_lengths array
        if args.num_prompts is not None:
            num_requests = args.num_prompts
        elif args.num_chat_sessions is not None:
            num_requests = args.num_chat_sessions
        else:
            raise ValueError(
                "Please specify either '--num-prompts' or '--num-chat-sessions'."
            )

        # NOTE: args.output_lengths is a path to a YAML file, while output_lengths
        # is a list of ints.
        if args.output_lengths is None:
            output_lengths = None
        elif os.path.exists(args.output_lengths):
            with open(args.output_lengths) as f:
                output_lengths = yaml.safe_load(f)["output_lengths"]
        else:
            output_lengths = [int(args.output_lengths)] * num_requests

        # We should not be using / accessing args.output_lengths from here on out.
        if isinstance(benchmark_dataset, CodeDebugBenchmarkDataset):
            # code_debug is a long-context dataset based on InfiniteBench
            if args.num_chat_sessions:
                if args.fit_distributions:
                    raise ValueError(
                        "--fit-distributions is not supported for --dataset-name "
                        "code-debug with --num-chat-sessions. Use random, "
                        "instruct-coder, or agentic-code for distribution-shaped "
                        "multiturn workloads, or omit --fit-distributions to keep "
                        "code-debug's fixed two-turn template."
                    )
                if output_lengths is not None:
                    raise NotImplementedError(
                        "TODO: Add support for fixed output lengths with multi-turn"
                        " code-debug"
                    )
                samples = benchmark_dataset.gen_twoturn_longcontext_requests(
                    num_chat_sessions=args.num_chat_sessions,
                    delay_between_chat_turns=args.delay_between_chat_turns,
                    tokenizer=tokenizer,
                )
            else:
                assert args.num_prompts is not None
                samples = benchmark_dataset.sample_requests(
                    num_requests=args.num_prompts,
                    tokenizer=tokenizer,
                    output_lengths=output_lengths,
                    shuffle=(
                        output_lengths is None
                        and not args.record_output_lengths
                    ),
                )

        elif isinstance(benchmark_dataset, ShareGPTBenchmarkDataset):
            assert args.num_prompts is not None
            samples = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    output_lengths is None and not args.record_output_lengths
                ),
            )

        elif isinstance(benchmark_dataset, SonnetBenchmarkDataset):
            # For sonnet, formatting depends on the endpoint
            apply_chat_template = chat
            # Sample sonnet requests with common parameters
            assert args.num_prompts is not None
            samples = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                input_len=args.sonnet_input_len,
                prefix_len=args.sonnet_prefix_len,
                apply_chat_template=apply_chat_template,
            )

        elif isinstance(benchmark_dataset, VisionArenaBenchmarkDataset):
            assert args.num_prompts is not None
            samples = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
            )
        elif isinstance(benchmark_dataset, ArxivSummarizationBenchmarkDataset):
            if output_lengths:
                raise ValueError(
                    "Arxiv summarization dataset does not support --output-lengths."
                    " Please use --max-output-len"
                )
            assert args.num_prompts is not None
            samples = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                shuffle=not args.record_output_lengths,
                input_len=args.arxiv_summarization_input_len,
                max_output_len=args.max_output_len,
            )
        elif isinstance(benchmark_dataset, RandomBenchmarkDataset):
            if args.num_chat_sessions:
                samples = benchmark_dataset.gen_multiturn_random_requests(
                    input_len=args.random_input_len,
                    output_len=args.random_output_len,
                    num_chat_sessions=_inflated_chat_session_count(
                        args, args.num_chat_sessions
                    ),
                    num_turns=args.random_num_turns,
                    delay_between_chat_turns=args.delay_between_chat_turns,
                    tokenizer=tokenizer,
                    sys_prompt_ratio=args.random_sys_prompt_ratio,
                    max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                )
            else:
                assert args.num_prompts is not None
                samples = benchmark_dataset.sample_requests(
                    num_requests=args.num_prompts,
                    tokenizer=tokenizer,
                    input_len=args.random_input_len,
                    output_len=args.random_output_len,
                    sys_prompt_ratio=args.random_sys_prompt_ratio,
                    max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                    image_size=args.random_image_size,
                    image_count=args.random_image_count,
                )
        elif isinstance(benchmark_dataset, AxolotlBenchmarkDataset):
            assert args.num_prompts is not None
            samples = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    output_lengths is None and not args.record_output_lengths
                ),
            )
        elif isinstance(benchmark_dataset, InstructCoderBenchmarkDataset):
            if args.num_chat_sessions:
                inflated_n = _inflated_chat_session_count(
                    args, args.num_chat_sessions
                )
                if args.fit_distributions:
                    samples = benchmark_dataset.gen_multiturn_sessions(
                        num_sessions=inflated_n,
                        tokenizer=tokenizer,
                        shuffle=(not args.record_output_lengths),
                        fit_length_distributions=True,
                        num_turns=args.random_num_turns,
                        input_len=args.random_input_len,
                        output_len=args.random_output_len,
                        delay_between_turns_dist=args.delay_between_chat_turns,
                        sys_prompt_ratio=args.random_sys_prompt_ratio,
                        max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                    )
                else:
                    samples = benchmark_dataset.gen_multiturn_sessions(
                        num_sessions=inflated_n,
                        tokenizer=tokenizer,
                        shuffle=(not args.record_output_lengths),
                        delay_between_chat_turns=resolve_constant_delay_ms(
                            args.delay_between_chat_turns
                        ),
                    )
            else:
                assert args.num_prompts is not None
                samples = benchmark_dataset.sample_requests(
                    num_requests=args.num_prompts,
                    tokenizer=tokenizer,
                    output_lengths=output_lengths,
                    shuffle=(
                        output_lengths is None
                        and not args.record_output_lengths
                    ),
                )
        elif isinstance(
            benchmark_dataset, ObfuscatedConversationsBenchmarkDataset
        ):
            if output_lengths is None:
                output_scale = (
                    args.obfuscated_conversations_average_output_len
                    * args.obfuscated_conversations_coefficient_of_variation
                )
                output_lengths = np.random.normal(
                    loc=args.obfuscated_conversations_average_output_len,
                    scale=output_scale,
                    size=num_requests,
                ).tolist()
                output_lengths = np.round(output_lengths).astype(int).tolist()
                output_lengths = [
                    max(output_len, 1) for output_len in output_lengths
                ]
            assert args.num_prompts is not None
            samples = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=args.obfuscated_conversations_shuffle,
                seed=args.seed,
            )
        elif isinstance(benchmark_dataset, BatchJobBenchmarkDataset):
            assert args.num_prompts is not None
            samples = benchmark_dataset.sample_requests(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_lengths=output_lengths,
                shuffle=(
                    output_lengths is None and not args.record_output_lengths
                ),
                image_dir=args.batch_job_image_dir,
            )
        elif isinstance(benchmark_dataset, AgenticCodeBenchmarkDataset):
            if args.num_chat_sessions:
                inflated_n = _inflated_chat_session_count(
                    args, args.num_chat_sessions
                )
                if args.fit_distributions:
                    samples = benchmark_dataset.gen_multiturn_sessions(
                        num_sessions=inflated_n,
                        tokenizer=tokenizer,
                        shuffle=(not args.record_output_lengths),
                        fit_length_distributions=True,
                        num_turns=args.random_num_turns,
                        input_len=args.random_input_len,
                        output_len=args.random_output_len,
                        delay_between_turns_dist=args.delay_between_chat_turns,
                        sys_prompt_ratio=args.random_sys_prompt_ratio,
                        max_num_unique_sys_prompt=args.random_max_num_unique_sys_prompt,
                        enable_tool_calls=args.tool_calls,
                    )
                else:
                    samples = benchmark_dataset.gen_multiturn_sessions(
                        num_sessions=inflated_n,
                        shuffle=(not args.record_output_lengths),
                        enable_tool_calls=args.tool_calls,
                    )
            else:
                assert args.num_prompts is not None
                samples = benchmark_dataset.sample_requests(
                    num_requests=args.num_prompts,
                    tokenizer=tokenizer,
                    output_lengths=output_lengths,
                    shuffle=(
                        output_lengths is None
                        and not args.record_output_lengths
                    ),
                    enable_tool_calls=args.tool_calls,
                )
        else:
            raise ValueError(
                f"Unknown / unsupported dataset: {benchmark_dataset}"
            )
    elif benchmark_task in PIXEL_GENERATION_TASKS:
        tokenizer = None
        if args.num_prompts is None:
            raise ValueError(
                "Please specify '--num-prompts' for "
                f"{benchmark_task} benchmarks."
            )
        if args.dataset_name == "local-image" and args.dataset_path is None:
            raise ValueError(
                "--benchmark-task image-to-image with "
                f"--dataset-name {args.dataset_name} requires --dataset-path"
            )
        benchmark_dataset = BenchmarkDataset.from_flags(
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
        )
        if benchmark_task == "text-to-image":
            if not isinstance(
                benchmark_dataset, SyntheticPixelBenchmarkDataset
            ):
                raise ValueError(
                    "text-to-image currently supports only "
                    "--dataset-name synthetic-pixel"
                )
        elif not isinstance(
            benchmark_dataset,
            (LocalImageBenchmarkDataset, SyntheticPixelBenchmarkDataset),
        ):
            raise ValueError(
                "image-to-image currently supports only "
                "--dataset-name local-image or synthetic-pixel"
            )
        logger.info("sampling requests")
        samples = benchmark_dataset.sample_requests(
            num_requests=args.num_prompts,
            tokenizer=None,
            benchmark_task=benchmark_task,
            image_width=args.image_width,
            image_height=args.image_height,
            image_steps=args.image_steps,
            image_guidance_scale=args.image_guidance_scale,
            image_negative_prompt=args.image_negative_prompt,
            image_seed=args.image_seed,
        )
    else:
        raise ValueError(f"Unsupported benchmark task: {benchmark_task}")

    # Inject response_format into all sampled requests if specified
    if args.response_format:
        response_format = parse_response_format(args.response_format)
        if isinstance(samples, RequestSamples):
            for request in samples.requests:
                request.response_format = response_format
            logger.info(
                f"Injected response_format into {len(samples.requests)} requests"
            )
        else:
            logger.warning(
                "response_format is only supported for single-turn benchmarks, "
                "ignoring for multi-turn chat sessions"
            )

    if args.print_workload_stats:
        print_workload_stats(samples)

    if args.print_inputs_and_outputs:
        print_input_prompts(samples)

    # ---- Dry run: build dataset + show warmup-sampling preview ----
    if args.dry_run:
        if not args.print_workload_stats:
            print_workload_stats(samples)
        if isinstance(samples, ChatSamples) and args.warmup_to_steady_state:
            rng = np.random.default_rng(args.seed or 0)
            for mc in concurrency_range:
                warmup_count = (
                    args.max_concurrent_conversations
                    or mc
                    or len(samples.chat_sessions)
                )
                print_section(
                    title=f" Warmup sampling preview (max_concurrency={mc}) ",
                    char="=",
                )
                _, report = _pick_warmup_population(
                    samples.chat_sessions,
                    warmup_count,
                    warmup_to_steady_state=True,
                    warmup_oversample_factor=args.warmup_oversample_factor,
                    main_pool_target=args.num_chat_sessions or 0,
                    rng=rng,
                )
                if report is not None:
                    _log_warmup_sampling_report(report)
        for mc in concurrency_range:
            for rr in request_rate_range:
                print(
                    f"Dry run: model={args.model}"
                    f" host={args.host} port={args.port}"
                    f" endpoint={args.endpoint}"
                    f" max_concurrency={mc}"
                    f" request_rate={rr}"
                    f" num_prompts={args.num_prompts}"
                    f" max_benchmark_duration_s="
                    f"{args.max_benchmark_duration_s}"
                )
                yield BenchmarkRunResult(
                    max_concurrency=mc,
                    request_rate=rr,
                    num_prompts=args.num_prompts or 0,
                )
        return

    lora_manager = None
    if args.lora_paths:
        num_requests = (
            len(samples.requests)
            if isinstance(samples, RequestSamples)
            else len(samples.chat_sessions)
        )

        lora_manager = LoRABenchmarkManager(
            lora_paths=args.lora_paths,
            num_requests=num_requests,
            traffic_ratios=args.per_lora_traffic_ratio
            if args.per_lora_traffic_ratio
            else None,
            uniform_ratio=args.lora_uniform_traffic_ratio,
            seed=args.seed,
            max_concurrent_lora_ops=args.max_concurrent_lora_ops,
        )
        lora_manager.log_traffic_distribution()

    # Handle trace flag (once, before loop)
    trace_path = None
    if args.trace:
        assert_nvidia_gpu()
        trace_path = (
            args.trace_file if args.trace_file else get_default_trace_path()
        )
        logger.info(f"Tracing enabled, output: {trace_path}")

    session = BenchmarkSession(
        benchmark_task=benchmark_task,
        endpoint=endpoint,
        api_url=api_url,
        base_url=base_url,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        tokenizer=tokenizer,
        samples=samples,
        lora_manager=lora_manager,
        trace_path=trace_path,
        orig_skip_first=args.skip_first_n_requests,
        orig_skip_last=args.skip_last_n_requests,
    )

    # ---- Sweep loop ----
    for mc in concurrency_range:
        if use_dynamic_num_prompts:
            assert args.num_prompts_multiplier is not None
            assert mc is not None
            args.num_prompts = args.num_prompts_multiplier * mc
            logger.info(
                f"Using num_prompts = {args.num_prompts_multiplier}"
                f" * {mc} = {args.num_prompts}"
            )

        for rr in request_rate_range:
            # Temporarily write the per-iteration values so that downstream
            # code reading args.max_concurrency / args.request_rate sees the
            # correct scalar value.
            args.max_concurrency = str(mc) if mc is not None else None
            args.request_rate = str(rr)

            iteration_results: list[
                TextGenerationBenchmarkResult | PixelGenerationBenchmarkResult
            ] = []
            for _iteration in range(args.num_iters):
                if args.flush_prefix_cache:
                    flush_prefix_cache(
                        args.backend, args.host, args.port, args.dry_run
                    )

                args.seed = int(np.random.randint(0, 10000))

                result = _execute_benchmark(args, session, mc, rr)
                iteration_results.append(result)

            # Median selection when running multiple iterations.
            if len(iteration_results) > 1:
                throughputs = np.asarray(
                    [r.metrics.request_throughput for r in iteration_results]
                )
                idx = argmedian(throughputs)
            else:
                idx = 0
            best_result = iteration_results[idx]

            # JSON result file (for the median iteration).
            save_result_json(
                args.result_filename,
                args,
                best_result,
                benchmark_task=session.benchmark_task,
                model_id=session.model_id,
                tokenizer_id=session.tokenizer_id,
                request_rate=rr,
            )

            # Output lengths recording (for the median iteration).
            _save_output_lengths(args, best_result, session.benchmark_task)

            yield BenchmarkRunResult(
                mc, rr, args.num_prompts or 0, result=best_result
            )


def _extract_metadata_args(
    args: list[str],
) -> tuple[list[str], list[str]]:
    """Extract --metadata values from args before passing to cyclopts.

    cyclopts interprets bare ``key=value`` tokens as keyword assignments. When
    a token like ``enable_prefix_caching=True`` matches a real model field, it
    is routed to that field rather than consumed as a ``--metadata`` list item,
    leaving subsequent tokens as orphaned positionals (which then fail).

    This function peels off all space-separated values after ``--metadata``
    (until the next ``--flag``) and returns them separately so cyclopts never
    sees them.

    Returns:
        A 2-tuple of (clean_args, metadata_values).
    """
    clean_args: list[str] = []
    metadata_values: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--metadata":
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                metadata_values.append(args[i])
                i += 1
        else:
            clean_args.append(args[i])
            i += 1
    return clean_args, metadata_values


def parse_args(
    args: Sequence[str] | None = None,
    *,
    app_name: str = "benchmark_serving",
    description: str = BENCHMARK_SERVING_ARGPARSER_DESCRIPTION,
) -> ServingBenchmarkConfig:
    """Parse command line arguments into a ServingBenchmarkConfig.

    Args:
        args: Command line arguments to parse. If None, parse from sys.argv.
        app_name: Name shown in --help output.
        description: Description shown in --help output.
    """
    raw_args = list(sys.argv[1:] if args is None else args)

    clean_args, metadata_values = _extract_metadata_args(raw_args)

    parsed_configs: list[ServingBenchmarkConfig] = []

    app = App(
        name=app_name,
        help=description,
        help_formatter="plain",
        config=[Env(prefix="MODULAR_")],
        result_action="return_value",
    )

    @app.default
    def _capture(
        config: Annotated[
            ServingBenchmarkConfig, Parameter(name="*")
        ] = ServingBenchmarkConfig(),
    ) -> None:
        parsed_configs.append(config)

    app(clean_args)
    if not parsed_configs:
        raise SystemExit(0)
    config = parsed_configs[0]
    if metadata_values:
        config.metadata = metadata_values
    return config
