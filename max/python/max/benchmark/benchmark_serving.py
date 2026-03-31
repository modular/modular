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
import hashlib
import json
import logging
import math
import os
import random
import re
import resource
import statistics
import subprocess
import sys
import time
import warnings
from collections.abc import AsyncGenerator, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard
from urllib.parse import urlparse

import numpy as np
import yaml
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

if TYPE_CHECKING:
    from max.benchmark.benchmark_shared.server_metrics import ParsedMetrics
    from max.diagnostics.gpu import BackgroundRecorder as GPUBackgroundRecorder
    from max.diagnostics.gpu import GPUStats

from max.benchmark.benchmark_shared.config import (
    Backend,
    BenchmarkTask,
    Endpoint,
    ServingBenchmarkConfig,
    parse_benchmark_args,
)
from max.benchmark.benchmark_shared.cpu_metrics import (
    CpuMetricsCollector,
    collect_pids_for_port,
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
    PixelGenerationBenchmarkMetrics,
    SpecDecodeMetrics,
    StandardPercentileMetrics,
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
)
from max.benchmark.benchmark_shared.server_metrics import (
    collect_server_metrics,
    fetch_spec_decode_metrics,
    print_server_metrics,
)
from max.benchmark.benchmark_shared.steady_state import detect_steady_state
from max.diagnostics.gpu import GPUDiagContext

BENCHMARK_SERVING_ARGPARSER_DESCRIPTION = (
    "This command runs comprehensive benchmark tests on a model server to"
    " measure performance metrics including throughput, latency, and resource"
    " utilization. Make sure that the MAX server is running and hosting a model"
    " before running this command."
)

logger = logging.getLogger("benchmark_serving")


def compute_output_len(
    tokenizer: PreTrainedTokenizerBase,
    output: RequestFuncOutput,
) -> int:
    return len(
        tokenizer(
            output.generated_text,
            add_special_tokens=False,
        ).input_ids
    )


def get_tokenizer(
    pretrained_model_name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        model_max_length=model_max_length,
        trust_remote_code=trust_remote_code,
    )


# from https://github.com/sgl-project/sglang/blob/v0.4.0/python/sglang/bench_serving.py#L1283
def set_ulimit(target_soft_limit: int = 65535) -> None:
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


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
) -> BaseRequestFuncInput:
    request_model_id = model_id if lora_id is None else lora_id
    if benchmark_task == BenchmarkTask.text_generation:
        max_tokens = min(
            filter(None, (request.output_len, max_output_len)),
            default=None,
        )
        return RequestFuncInput(
            model=request_model_id,
            session_id=None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prompt=request.prompt_formatted,
            images=request.encoded_images,
            api_url=api_url,
            prompt_len=request.prompt_len,
            max_tokens=max_tokens,
            ignore_eos=request.ignore_eos,
        )
    if benchmark_task in BenchmarkTask.pixel_generation_tasks():
        if not isinstance(request, PixelGenerationSampledRequest):
            raise TypeError(
                "pixel-generation benchmark requires PixelGenerationSampledRequest."
            )
        return PixelGenerationRequestFuncInput(
            model=request_model_id,
            session_id=None,
            prompt=request.prompt_formatted,
            input_image_paths=request.input_image_paths,
            api_url=api_url,
            image_options=request.image_options,
        )
    raise ValueError(f"Unsupported benchmark task: {benchmark_task}")


def print_section(title: str, char: str = "-") -> None:
    """Helper function to print a section with formatted header."""
    print("{s:{c}^{n}}".format(s=title, n=50, c=char))


def _is_vllm_backend(backend: Backend) -> bool:
    return backend in (Backend.vllm, Backend.vllm_chat)


def _add_spec_decode_result(
    result: dict[str, Any],
    spec_decode_stats: dict[str, Any] | None,
) -> None:
    """Add speculative decoding stats to the JSON result."""
    if spec_decode_stats is None:
        return
    result["spec_decode_acceptance_rate"] = spec_decode_stats["acceptance_rate"]
    result["spec_decode_acceptance_length"] = spec_decode_stats[
        "acceptance_length"
    ]
    result["spec_decode_num_drafts"] = int(spec_decode_stats["num_drafts"])
    result["spec_decode_draft_tokens"] = int(spec_decode_stats["draft_tokens"])
    result["spec_decode_accepted_tokens"] = int(
        spec_decode_stats["accepted_tokens"]
    )
    result["spec_decode_per_position_acceptance_rates"] = spec_decode_stats.get(
        "per_position_acceptance_rates", []
    )


def print_lora_benchmark_results(
    lora_manager: LoRABenchmarkManager,
) -> None:
    """Print LoRA benchmark statistics if available."""

    print_section(title=" LoRA Adapter Benchmark Results ", char="=")
    print(
        "{:<40} {:<10}".format(
            "Total LoRA loads:", lora_manager.metrics.total_loads
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Total LoRA unloads:", lora_manager.metrics.total_unloads
        )
    )

    if lora_manager.metrics.load_times_ms:
        print_section(title="LoRA Load Times")
        print(
            "{:<40} {:<10.2f}".format(
                "Mean load time:",
                statistics.mean(lora_manager.metrics.load_times_ms),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Median load time:",
                statistics.median(lora_manager.metrics.load_times_ms),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Min load time:", min(lora_manager.metrics.load_times_ms)
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Max load time:", max(lora_manager.metrics.load_times_ms)
            )
        )
        if len(lora_manager.metrics.load_times_ms) > 1:
            print(
                "{:<40} {:<10.2f}".format(
                    "Std dev load time:",
                    statistics.stdev(lora_manager.metrics.load_times_ms),
                )
            )

    if lora_manager.metrics.unload_times_ms:
        print_section(title="LoRA Unload Times")
        print(
            "{:<40} {:<10.2f}".format(
                "Mean unload time:",
                statistics.mean(lora_manager.metrics.unload_times_ms),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Median unload time:",
                statistics.median(lora_manager.metrics.unload_times_ms),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Min unload time:",
                min(lora_manager.metrics.unload_times_ms),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Max unload time:",
                max(lora_manager.metrics.unload_times_ms),
            )
        )
        if len(lora_manager.metrics.unload_times_ms) > 1:
            print(
                "{:<40} {:<10.2f}".format(
                    "Std dev unload time:",
                    statistics.stdev(lora_manager.metrics.unload_times_ms),
                )
            )


def print_benchmark_summary(
    metrics: BenchmarkMetrics | PixelGenerationBenchmarkMetrics,
    benchmark_duration: float,
    request_rate: float,
    max_concurrency: int | None,
    achieved_request_rate: float,
    collect_gpu_stats: bool,
    collect_cpu_stats: bool,
    spec_decode_stats: dict[str, Any] | None = None,
    lora_manager: LoRABenchmarkManager | None = None,
) -> None:
    """Print benchmark summary for text-generation and pixel-generation."""

    # 1. Print common benchmark summary
    print_section(title=" Serving Benchmark Result ", char="=")
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failures))
    print(
        "{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration)
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

    if isinstance(metrics, BenchmarkMetrics):
        print_section(title="Token Stats")
        print("{:<40} {:<10}".format("Max input tokens:", metrics.max_input))
        print("{:<40} {:<10}".format("Max output tokens:", metrics.max_output))
        print("{:<40} {:<10}".format("Max total tokens:", metrics.max_total))

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
        print_section(title="CPU Statistics")
        print(
            "{:<40} {:<10.2f}".format(
                "CPU utilization user (%):",
                metrics.cpu_utilization_user or 0.0,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "CPU utilization system (%):",
                metrics.cpu_utilization_system or 0.0,
            )
        )
    if spec_decode_stats is not None:
        print_section(title="Speculative Decoding")
        print(
            "{:<40} {:<10.2f}".format(
                "Acceptance rate (%):", spec_decode_stats["acceptance_rate"]
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Acceptance length:", spec_decode_stats["acceptance_length"]
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Drafts:", int(spec_decode_stats["num_drafts"])
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Draft tokens:", int(spec_decode_stats["draft_tokens"])
            )
        )
        print(
            "{:<40} {:<10}".format(
                "Accepted tokens:", int(spec_decode_stats["accepted_tokens"])
            )
        )
        per_pos_rates = spec_decode_stats.get(
            "per_position_acceptance_rates", []
        )
        if per_pos_rates:
            print("Per-position acceptance (%):")
            for pos, rate in enumerate(per_pos_rates):
                print(
                    "{:<40} {:<10.2f}".format(f"  Position {pos}:", rate * 100)
                )
    print("=" * 50)
    if lora_manager:
        print_lora_benchmark_results(lora_manager)
    if metrics.server_metrics:
        print_server_metrics(metrics.server_metrics)
    print("=" * 50)


def _steady_state_metric_values(
    m: BenchmarkMetrics,
) -> list[tuple[str, float]]:
    """Return (suffix, value) pairs for steady-state metrics.

    Each suffix corresponds to an existing full-run key (e.g.,
    "mean_ttft_ms" maps to result["mean_ttft_ms"] in the full run and
    result["steady_state_mean_ttft_ms"] in the steady-state section).
    """
    return [
        ("request_throughput", m.request_throughput),
        ("mean_ttft_ms", m.ttft_ms.mean),
        ("p99_ttft_ms", m.ttft_ms.p99),
        ("mean_tpot_ms", m.tpot_ms.mean),
        ("p99_tpot_ms", m.tpot_ms.p99),
        ("mean_itl_ms", m.itl_ms.mean),
        ("p99_itl_ms", m.itl_ms.p99),
        ("mean_latency_ms", m.latency_ms.mean),
        ("p99_latency_ms", m.latency_ms.p99),
    ]


def _add_confidence_fields(
    result: dict[str, Any],
    metrics_by_prefix: list[tuple[str, Any]],
) -> None:
    """Add confidence interval fields to the result dict for each metric."""
    for prefix, metric_obj in metrics_by_prefix:
        ci = getattr(metric_obj, "confidence_info", None)
        if ci:
            result[f"{prefix}_ci_lower"] = ci.ci_lower
            result[f"{prefix}_ci_upper"] = ci.ci_upper
            result[f"{prefix}_ci_relative_width"] = ci.ci_relative_width
            result[f"{prefix}_confidence"] = ci.confidence
            result[f"{prefix}_sample_size"] = ci.sample_size


def _add_optional_result(
    result: dict[str, Any],
    metrics: BenchmarkMetrics | PixelGenerationBenchmarkMetrics,
    lora_manager: LoRABenchmarkManager | None,
) -> None:
    if lora_manager is not None:
        result["lora_metrics"] = {
            "total_loads": lora_manager.metrics.total_loads,
            "total_unloads": lora_manager.metrics.total_unloads,
            "load_times_ms": lora_manager.metrics.load_times_ms,
            "unload_times_ms": lora_manager.metrics.unload_times_ms,
        }

    if metrics.server_metrics is None:
        return

    result["server_metrics"] = {
        "counters": metrics.server_metrics.counters,
        "gauges": metrics.server_metrics.gauges,
        "histograms": {
            name: {
                "buckets": hist.buckets,
                "sum": hist.sum,
                "count": hist.count,
                "mean": hist.mean,
            }
            for name, hist in metrics.server_metrics.histograms.items()
        },
    }

    if isinstance(metrics, BenchmarkMetrics):
        result["server_metrics"].update(
            {
                # Convenience fields for prefill/decode breakdown.
                "prefill_batch_execution_time_ms": metrics.mean_prefill_batch_time_ms,
                "prefill_batch_count": metrics.prefill_batch_count,
                "decode_batch_execution_time_ms": metrics.mean_decode_batch_time_ms,
                "decode_batch_count": metrics.decode_batch_count,
            }
        )


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
        num_turns_list = [len(s.messages) // 2 for s in sessions]

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

        print(f"  {'Total sessions:':<30} {len(sessions)}")
        print(f"  {'Total turns (across all):':<30} {sum(num_turns_list)}")
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
    cpu_metrics: dict[str, Any],
    skip_first_n_requests: int,
    skip_last_n_requests: int,
    max_concurrency: int | None,
    collect_gpu_stats: bool,
    server_metrics: ParsedMetrics | None = None,
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    nonempty_response_chunks = 0
    total_input = 0
    max_input = 0
    max_output = 0
    max_total = 0
    failures = 0
    failed_responses: list[RequestFuncOutput] = []
    itls: list[float] = []
    tpots: list[float] = []
    ttfts: list[float] = []
    latencies: list[float] = []
    input_throughputs: list[float] = []
    output_throughputs: list[float] = []

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

    completed = len(successful)

    for o, output_len in successful:
        total_input += o.prompt_len
        actual_output_lens.append(output_len)
        nonempty_response_chunks += 1 if o.ttft != 0 else 0
        nonempty_response_chunks += len(o.itl)
        max_input = max(max_input, o.prompt_len)
        max_output = max(max_output, output_len)
        max_total = max(max_total, o.prompt_len + output_len)

    end = (
        completed - skip_last_n_requests
        if skip_last_n_requests > 0
        else completed
    )
    measured = successful[skip_first_n_requests:end]

    for o, output_len in measured:
        tpots += o.tpot
        itls += o.itl
        ttfts.append(o.ttft)
        if o.ttft > 0:
            input_throughputs.append(o.prompt_len / o.ttft)
        if (o.latency - o.ttft) > 0:
            output_throughputs.append((output_len - 1) / (o.latency - o.ttft))
        latencies.append(o.latency)

    _warn_on_request_failures(
        outputs=outputs,
        completed=completed,
        failures=failures,
        failed_responses=failed_responses,
    )

    measured_count = len(ttfts)
    if measured_count == 0 and completed > 0:
        warnings.warn(
            (
                f"All {completed} successful requests were excluded by"
                f" skip_first_n_requests={skip_first_n_requests} and"
                f" skip_last_n_requests={skip_last_n_requests}."
                " Consider running a longer benchmark."
            ),
            stacklevel=2,
        )
    elif 0 < measured_count < 10:
        warnings.warn(
            (
                f"Only {measured_count} requests remain after skipping"
                f" first {skip_first_n_requests} and last"
                f" {skip_last_n_requests} requests."
                " Results may not be reliable."
                " Consider running a longer benchmark."
            ),
            stacklevel=2,
        )

    (
        peak_gpu_memory_mib,
        available_gpu_memory_mib,
        gpu_utilization,
    ) = _aggregate_gpu_stats(
        collect_gpu_stats=collect_gpu_stats,
        gpu_metrics=gpu_metrics,
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
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_utilization_user=cpu_metrics.get("user_percent"),
        cpu_utilization_system=cpu_metrics.get("system_percent"),
        server_metrics=server_metrics,
    )

    # Override TPOT mean with weighted average: sum(ITL) / decode_tokens.
    # This is more accurate than mean-of-means since it properly weights
    # by tokens returned per response. Decode tokens = total output - completed,
    # since each request's first token is from prefill (TTFT), not decode.
    total_output_tokens = sum(actual_output_lens)
    decode_tokens = total_output_tokens - completed
    if decode_tokens > 0 and itls:
        metrics.tpot_ms._metrics.mean = sum(itls) / decode_tokens * 1000.0

    return metrics, actual_output_lens


def calculate_pixel_generation_metrics(
    outputs: Sequence[PixelGenerationRequestFuncOutput],
    dur_s: float,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: dict[str, Any],
    max_concurrency: int | None,
    collect_gpu_stats: bool,
    server_metrics: ParsedMetrics | None = None,
) -> PixelGenerationBenchmarkMetrics:
    completed = 0
    failures = 0
    latencies: list[float] = []
    total_generated_outputs = 0
    failed_responses: list[PixelGenerationRequestFuncOutput] = []

    for output in outputs:
        if output.cancelled:
            continue
        if output.success:
            completed += 1
            latencies.append(output.latency)
            total_generated_outputs += output.num_generated_outputs
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

    return PixelGenerationBenchmarkMetrics(
        completed=completed,
        failures=failures,
        max_concurrency=max_concurrency or len(outputs),
        request_throughput=completed / dur_s,
        total_generated_outputs=total_generated_outputs,
        latency_ms=StandardPercentileMetrics(
            latencies or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_utilization_user=cpu_metrics.get("user_percent"),
        cpu_utilization_system=cpu_metrics.get("system_percent"),
        server_metrics=server_metrics,
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
    skip_session_count: int | None = None,
    ignore_first_turn_stats: bool = False,
    benchmark_should_end_time: int | None = None,
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
    while content_idx + 1 < len(messages):
        chat_len += messages[content_idx].num_tokens
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
        message_history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        )
        request_func_input.prompt = message_history
        request_func_input.prompt_len = chat_len
        request_func_input.max_tokens = output_len

        # Check timeout before making request
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
        if (
            skip_session_count is None
            or chat_session.id is None
            or chat_session.id >= skip_session_count
        ) and not (ignore_first_turn_stats and content_idx == 0):
            session_outputs.append(response)

        if not response.success:
            if not response.cancelled:
                logger.error(
                    f"Ending chat session {chat_session.id} due to server error"
                    f" response: {response.error}"
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
        )
        tasks.append(
            asyncio.create_task(limited_request_func(request_func_input))
        )
        request_idx += 1

    outputs = await asyncio.gather(*tasks)

    return outputs


async def run_multiturn_benchmark(
    chat_sessions: Sequence[ChatSession],
    max_requests: int,
    semaphore: asyncio.Semaphore | None,
    benchmark_should_end_time: int | None,
    request_driver: RequestDriver,
    model_id: str,
    api_url: str,
    tokenizer: PreTrainedTokenizerBase,
    skip_first_n_requests: int,
    ignore_first_turn_stats: bool,
    lora_manager: LoRABenchmarkManager | None,
    warmup_delay_ms: float,
    max_concurrency: int | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> list[RequestFuncOutput]:
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
    ) -> list[RequestFuncOutput]:
        # Determine which LoRA to use for this chat session
        lora_id = None
        if lora_manager:
            lora_id = lora_manager.get_lora_for_request(session_idx)

        if semaphore is None:
            return await chat_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=request_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_chat_len=tokenizer.model_max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                skip_session_count=skip_first_n_requests,
                ignore_first_turn_stats=ignore_first_turn_stats,
                benchmark_should_end_time=benchmark_should_end_time,
            )
        async with semaphore:
            return await chat_session_driver(
                model_id=model_id if lora_id is None else lora_id,
                api_url=api_url,
                request_driver=request_driver,
                request_counter=request_counter,
                chat_session=chat_session,
                max_chat_len=tokenizer.model_max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                skip_session_count=skip_first_n_requests,
                ignore_first_turn_stats=ignore_first_turn_stats,
                benchmark_should_end_time=benchmark_should_end_time,
            )

    tasks: list[asyncio.Task[list[RequestFuncOutput]]] = []
    for idx, chat_session in enumerate(chat_sessions):
        if warmup_delay_ms > 0 and max_concurrency and idx < max_concurrency:
            await asyncio.sleep(warmup_delay_ms / 1000)
        tasks.append(
            asyncio.create_task(limited_chat_session_driver(chat_session, idx))
        )

    session_outputs: list[list[RequestFuncOutput]] = await asyncio.gather(
        *tasks
    )

    return [output for sublist in session_outputs for output in sublist]


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
        num_qa_turns = [
            (len(session.messages) // 2) for session in samples.chat_sessions
        ]
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
    disable_tqdm: bool,
    do_test_prompt: bool,
    collect_gpu_stats: bool,
    collect_cpu_stats: bool,
    collect_server_stats: bool,
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
    timing_data: dict[str, list[float]] | None,
    lora_manager: LoRABenchmarkManager | None,
    trace_path: str | None = None,
    trace_session: str | None = None,
) -> tuple[dict[str, Any], BenchmarkMetrics | PixelGenerationBenchmarkMetrics]:
    if ignore_first_turn_stats and skip_first_n_requests:
        logger.warning(
            "--ignore-first-turn-stats and --skip-first-n-requests both set."
            " Ignoring --skip-first-n-requests due to first turn in each chat"
            " already being ignored."
        )
        skip_first_n_requests = 0

    # Benchmark LoRA loading if manager provided
    if lora_manager:
        logger.info("Starting LoRA loading benchmark...")
        await lora_manager.benchmark_loading(
            api_url=base_url,
        )

    request_driver_class: type[RequestDriver] = get_request_driver_class(
        api_url, task=benchmark_task
    )
    # Create a request driver instance without pbar for test prompt
    # (pbar will be set later for the actual benchmark runs)
    test_request_driver: RequestDriver = request_driver_class(
        tokenizer=tokenizer
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
                cpu_collector = CpuMetricsCollector(pids)
                cpu_collector.start()
            except:
                logger.warning(
                    "Cannot access max-serve PIDs, skipping CPU stats"
                    " collection"
                )

        # Start nsys trace if enabled (before timing to exclude trace overhead)
        if trace_path:
            start_trace(trace_path, trace_session)

        benchmark_start_time = time.perf_counter_ns()
        if max_benchmark_duration_s is None:
            benchmark_should_end_time = None
        else:
            benchmark_should_end_time = benchmark_start_time + int(
                max_benchmark_duration_s * 1e9
            )

        # Capture baseline server metrics before benchmark starts
        baseline_server_metrics = None
        if collect_server_stats:
            try:
                baseline_server_metrics = collect_server_metrics(
                    backend, base_url
                )
                logger.info(
                    f"Captured baseline server metrics: "
                    f"{len(baseline_server_metrics.counters)} counters, "
                    f"{len(baseline_server_metrics.gauges)} gauges, "
                    f"{len(baseline_server_metrics.histograms)} histograms"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to capture baseline server metrics: {e}"
                )

        # Create pbar for actual benchmark runs
        pbar = create_benchmark_pbar(disable_tqdm=disable_tqdm, samples=samples)

        # Create base driver and wrap with ProgressBarRequestDriver if pbar is provided
        base_driver: RequestDriver = request_driver_class(tokenizer=tokenizer)
        request_driver: RequestDriver = (
            ProgressBarRequestDriver(base_driver, pbar)
            if pbar is not None
            else base_driver
        )

        if benchmark_task == BenchmarkTask.text_generation and _is_vllm_backend(
            backend
        ):
            spec_decode_metrics_before = fetch_spec_decode_metrics(
                backend, base_url
            )

        try:
            outputs: Sequence[BaseRequestFuncOutput]
            if isinstance(samples, RequestSamples):
                # single-turn chat scenario
                outputs = await run_single_turn_benchmark(
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
                )
            else:
                # multi-turn chat scenario
                outputs = await run_multiturn_benchmark(
                    chat_sessions=samples.chat_sessions,
                    max_requests=max_requests,
                    semaphore=semaphore,
                    benchmark_should_end_time=benchmark_should_end_time,
                    request_driver=request_driver,
                    model_id=model_id,
                    api_url=api_url,
                    tokenizer=tokenizer,
                    skip_first_n_requests=skip_first_n_requests,
                    ignore_first_turn_stats=ignore_first_turn_stats,
                    lora_manager=lora_manager,
                    warmup_delay_ms=warmup_delay_ms,
                    max_concurrency=max_concurrency,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

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

    if benchmark_task == BenchmarkTask.text_generation and _is_vllm_backend(
        backend
    ):
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
        if benchmark_task == BenchmarkTask.text_generation:
            assert tokenizer is not None
            print("Generated output text:")
            for req_id, output in enumerate(outputs):
                assert isinstance(output, RequestFuncOutput)
                output_len = compute_output_len(tokenizer, output)
                print(
                    {
                        "req_id": req_id,
                        "output_len": output_len,
                        "output": output.generated_text,
                    }
                )
        elif benchmark_task in BenchmarkTask.pixel_generation_tasks():
            print("Generated pixel generation outputs:")
            for req_id, output in enumerate(outputs):
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

    if collect_cpu_stats and cpu_collector is not None:
        cpu_collector.stop()
        cpu_metrics = cpu_collector.dump_stats()
    else:
        cpu_metrics = {}

    # Collect server-side metrics from Prometheus endpoint (with delta from baseline)
    server_metrics = None
    if collect_server_stats:
        try:
            server_metrics = collect_server_metrics(
                backend, base_url, baseline_server_metrics
            )
            if baseline_server_metrics is not None:
                logger.info(
                    f"Computed server metrics delta: "
                    f"{len(server_metrics.counters)} counters, "
                    f"{len(server_metrics.gauges)} gauges, "
                    f"{len(server_metrics.histograms)} histograms"
                )
            else:
                logger.info(
                    f"Collected server metrics: "
                    f"{len(server_metrics.counters)} counters, "
                    f"{len(server_metrics.gauges)} gauges, "
                    f"{len(server_metrics.histograms)} histograms"
                )
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

    if benchmark_task in BenchmarkTask.pixel_generation_tasks():
        if not _is_pixel_generation_outputs(outputs):
            raise TypeError(
                "Expected all outputs to be PixelGenerationRequestFuncOutput"
                " in pixel-generation benchmark flow."
            )
        pixel_metrics = calculate_pixel_generation_metrics(
            outputs=outputs,
            dur_s=benchmark_duration,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics,
            max_concurrency=max_concurrency,
            collect_gpu_stats=collect_gpu_stats,
            server_metrics=server_metrics,
        )

        print_benchmark_summary(
            metrics=pixel_metrics,
            benchmark_duration=benchmark_duration,
            request_rate=request_rate,
            achieved_request_rate=achieved_request_rate,
            max_concurrency=max_concurrency,
            collect_gpu_stats=collect_gpu_stats,
            collect_cpu_stats=collect_cpu_stats,
            spec_decode_stats=None,
            lora_manager=lora_manager,
        )

        result = {
            "duration": benchmark_duration,
            "completed": pixel_metrics.completed,
            "failures": pixel_metrics.failures,
            "max_concurrency": pixel_metrics.max_concurrency,
            "request_throughput": pixel_metrics.request_throughput,
            "total_generated_outputs": pixel_metrics.total_generated_outputs,
            "mean_latency_ms": pixel_metrics.latency_ms.mean,
            "median_latency_ms": pixel_metrics.latency_ms.median,
            "std_latency_ms": pixel_metrics.latency_ms.std,
            "p90_latency_ms": pixel_metrics.latency_ms.p90,
            "p95_latency_ms": pixel_metrics.latency_ms.p95,
            "p99_latency_ms": pixel_metrics.latency_ms.p99,
            "latencies": [output.latency for output in outputs],
            "num_generated_outputs": [
                output.num_generated_outputs for output in outputs
            ],
            "errors": [output.error for output in outputs],
            "request_submit_times": [
                output.request_submit_time for output in outputs
            ],
            "request_complete_times": [
                output.request_complete_time for output in outputs
            ],
            "peak_gpu_memory_mib": pixel_metrics.peak_gpu_memory_mib,
            "available_gpu_memory_mib": pixel_metrics.available_gpu_memory_mib,
            "gpu_utilization": pixel_metrics.gpu_utilization,
        }

        _add_optional_result(
            result=result,
            metrics=pixel_metrics,
            lora_manager=lora_manager,
        )

        return result, pixel_metrics

    if not _is_text_generation_outputs(outputs):
        raise TypeError(
            "Expected all outputs to be RequestFuncOutput"
            " in text-generation benchmark flow."
        )
    text_metrics, actual_output_lens = calculate_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        gpu_metrics=gpu_metrics,
        cpu_metrics=cpu_metrics,
        skip_first_n_requests=skip_first_n_requests,
        skip_last_n_requests=skip_last_n_requests,
        max_concurrency=max_concurrency,
        collect_gpu_stats=collect_gpu_stats,
        server_metrics=server_metrics,
    )

    print_benchmark_summary(
        metrics=text_metrics,
        benchmark_duration=benchmark_duration,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        achieved_request_rate=achieved_request_rate,
        collect_gpu_stats=collect_gpu_stats,
        collect_cpu_stats=collect_cpu_stats,
        spec_decode_stats=spec_decode_stats,
        lora_manager=lora_manager,
    )

    result = {
        "duration": benchmark_duration,
        "completed": text_metrics.completed,
        "failures": text_metrics.failures,
        "max_concurrency": text_metrics.max_concurrency,
        "skip_first_n_requests": skip_first_n_requests,
        "skip_last_n_requests": skip_last_n_requests,
        "total_input_tokens": text_metrics.total_input,
        "total_output_tokens": text_metrics.total_output,
        "request_throughput": text_metrics.request_throughput,
        "mean_input_throughput": text_metrics.input_throughput.mean,
        "std_input_throughput": text_metrics.input_throughput.std,
        "median_input_throughput": text_metrics.input_throughput.median,
        "p90_input_throughput": text_metrics.input_throughput.p90,
        "p95_input_throughput": text_metrics.input_throughput.p95,
        "p99_input_throughput": text_metrics.input_throughput.p99,
        "mean_output_throughput": text_metrics.output_throughput.mean,
        "std_output_throughput": text_metrics.output_throughput.std,
        "median_output_throughput": text_metrics.output_throughput.median,
        "p90_output_throughput": text_metrics.output_throughput.p90,
        "p95_output_throughput": text_metrics.output_throughput.p95,
        "p99_output_throughput": text_metrics.output_throughput.p99,
        "mean_ttft_ms": text_metrics.ttft_ms.mean,
        "median_ttft_ms": text_metrics.ttft_ms.median,
        "std_ttft_ms": text_metrics.ttft_ms.std,
        "p90_ttft_ms": text_metrics.ttft_ms.p90,
        "p95_ttft_ms": text_metrics.ttft_ms.p95,
        "p99_ttft_ms": text_metrics.ttft_ms.p99,
        "mean_tpot_ms": text_metrics.tpot_ms.mean,
        "median_tpot_ms": text_metrics.tpot_ms.median,
        "std_tpot_ms": text_metrics.tpot_ms.std,
        "p90_tpot_ms": text_metrics.tpot_ms.p90,
        "p95_tpot_ms": text_metrics.tpot_ms.p95,
        "p99_tpot_ms": text_metrics.tpot_ms.p99,
        "mean_itl_ms": text_metrics.itl_ms.mean,
        "median_itl_ms": text_metrics.itl_ms.median,
        "std_itl_ms": text_metrics.itl_ms.std,
        "p90_itl_ms": text_metrics.itl_ms.p90,
        "p95_itl_ms": text_metrics.itl_ms.p95,
        "p99_itl_ms": text_metrics.itl_ms.p99,
        "mean_latency_ms": text_metrics.latency_ms.mean,
        "median_latency_ms": text_metrics.latency_ms.median,
        "std_latency_ms": text_metrics.latency_ms.std,
        "p90_latency_ms": text_metrics.latency_ms.p90,
        "p95_latency_ms": text_metrics.latency_ms.p95,
        "p99_latency_ms": text_metrics.latency_ms.p99,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "request_submit_times": [
            output.request_submit_time for output in outputs
        ],
        "request_complete_times": [
            output.request_complete_time for output in outputs
        ],
        "peak_gpu_memory_mib": text_metrics.peak_gpu_memory_mib,
        "available_gpu_memory_mib": text_metrics.available_gpu_memory_mib,
        "gpu_utilization": text_metrics.gpu_utilization,
    }

    _add_spec_decode_result(result, spec_decode_stats)

    _add_optional_result(
        result=result,
        metrics=text_metrics,
        lora_manager=lora_manager,
    )

    _add_confidence_fields(
        result,
        [
            ("ttft_ms", text_metrics.ttft_ms),
            ("tpot_ms", text_metrics.tpot_ms),
            ("itl_ms", text_metrics.itl_ms),
            ("latency_ms", text_metrics.latency_ms),
            ("output_throughput", text_metrics.output_throughput),
            ("input_throughput", text_metrics.input_throughput),
        ],
    )

    for warn in text_metrics.confidence_warnings():
        logger.warning(f"Confidence: {warn}")

    # Steady-state metrics mirror the full-run metrics above but are
    # prefixed with "steady_state_" and computed only over the detected window
    steady = detect_steady_state(outputs)
    result["steady_state_detected"] = steady.detected
    result["steady_state_start_index"] = steady.start_index
    result["steady_state_end_index"] = steady.end_index
    result["steady_state_count"] = steady.steady_state_count
    result["steady_state_warning"] = steady.warning

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

            ss_metrics, _ = calculate_metrics(
                outputs=ss_outputs,
                dur_s=ss_duration,
                tokenizer=tokenizer,
                gpu_metrics=gpu_metrics,
                cpu_metrics=cpu_metrics,
                skip_first_n_requests=0,
                skip_last_n_requests=0,
                max_concurrency=max_concurrency,
                collect_gpu_stats=collect_gpu_stats,
                server_metrics=server_metrics,
            )
            for suffix, value in _steady_state_metric_values(ss_metrics):
                result[f"steady_state_{suffix}"] = value
            _add_confidence_fields(
                result,
                [
                    ("steady_state_ttft_ms", ss_metrics.ttft_ms),
                    ("steady_state_tpot_ms", ss_metrics.tpot_ms),
                    ("steady_state_itl_ms", ss_metrics.itl_ms),
                    ("steady_state_latency_ms", ss_metrics.latency_ms),
                ],
            )
        logger.info(
            f"Steady-state detected: requests [{steady.start_index},"
            f" {steady.end_index}) ({steady.steady_state_count} of"
            f" {steady.total_requests} requests)"
        )
    elif steady.warning:
        logger.warning(f"Steady-state detection: {steady.warning}")

    return result, text_metrics


def validate_task_and_endpoint(
    benchmark_task: BenchmarkTask, endpoint: Endpoint
) -> None:
    if benchmark_task == BenchmarkTask.text_generation:
        if endpoint == Endpoint.responses:
            raise ValueError(
                "--benchmark-task text-generation does not support "
                "--endpoint /v1/responses"
            )
    elif benchmark_task in BenchmarkTask.pixel_generation_tasks():
        if endpoint != Endpoint.responses:
            raise ValueError(
                "--benchmark-task pixel-generation requires "
                "--endpoint /v1/responses"
            )


def main_with_parsed_args(args: ServingBenchmarkConfig) -> None:
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

    if args.model is None:
        raise ValueError("--model is required when running benchmark")
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    benchmark_task = BenchmarkTask(args.benchmark_task)
    try:
        endpoint = Endpoint(args.endpoint)
    except ValueError as e:
        raise ValueError(f"Unknown endpoint: {args.endpoint}") from e

    validate_task_and_endpoint(benchmark_task, endpoint)
    chat = endpoint == Endpoint.chat_completions

    if args.base_url is not None:
        base_url = args.base_url
    else:
        base_url = f"http://{args.host}:{args.port}"

    api_url = f"{base_url}{endpoint.value}"
    tokenizer: PreTrainedTokenizerBase | None
    samples: Samples

    if benchmark_task == BenchmarkTask.text_generation:
        logger.info(f"getting tokenizer. api url: {api_url}")
        tokenizer = get_tokenizer(
            tokenizer_id,
            args.model_max_length,
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
                    num_chat_sessions=args.num_chat_sessions,
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
                samples = benchmark_dataset.gen_multiturn_sessions(
                    num_sessions=args.num_chat_sessions,
                    tokenizer=tokenizer,
                    shuffle=(not args.record_output_lengths),
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
                samples = benchmark_dataset.gen_multiturn_sessions(
                    num_sessions=args.num_chat_sessions,
                    shuffle=(not args.record_output_lengths),
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
    elif benchmark_task in BenchmarkTask.pixel_generation_tasks():
        tokenizer = None
        if args.num_prompts is None:
            raise ValueError(
                "Please specify '--num-prompts' for "
                f"{benchmark_task.value} benchmarks."
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
        if benchmark_task == BenchmarkTask.text_to_image:
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
            benchmark_task=benchmark_task.value,
            image_width=args.image_width,
            image_height=args.image_height,
            image_steps=args.image_steps,
            image_guidance_scale=args.image_guidance_scale,
            image_negative_prompt=args.image_negative_prompt,
            image_seed=args.image_seed,
        )
    else:
        raise ValueError(f"Unsupported benchmark task: {benchmark_task}")

    if args.print_workload_stats:
        print_workload_stats(samples)

    if args.print_inputs_and_outputs:
        print_input_prompts(samples)

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

    max_concurrency: int | None = None
    if args.max_concurrency is not None:
        try:
            max_concurrency = int(args.max_concurrency)
        except ValueError as e:
            raise ValueError(
                f"Expected a single integer value for max_concurrency, got {args.max_concurrency}"
            ) from e
    try:
        request_rate = float(args.request_rate)
    except ValueError as e:
        raise ValueError(
            f"Expected a single float value for request_rate, got {args.request_rate}"
        ) from e

    try:
        backend = Backend(args.backend)
    except ValueError as e:
        raise ValueError(
            f"Unknown backend: {args.backend}. "
            f"Supported backends: {', '.join(b.value for b in Backend)}"
        ) from e

    # Handle trace flag
    trace_path = None
    if args.trace:
        assert_nvidia_gpu()
        trace_path = (
            args.trace_file if args.trace_file else get_default_trace_path()
        )
        logger.info(f"Tracing enabled, output: {trace_path}")

    # Auto-default skip counts to max_concurrency when not explicitly set.
    # None means "auto" (user did not pass the flag); 0 means "explicitly
    # no skipping" (user passed --skip-first-n-requests 0).
    skip_first_n_requests = args.skip_first_n_requests
    skip_last_n_requests = args.skip_last_n_requests
    if max_concurrency is not None:
        if skip_first_n_requests is None:
            skip_first_n_requests = max_concurrency
            logger.info(
                f"Auto-setting skip_first_n_requests={skip_first_n_requests}"
                f" (max_concurrency={max_concurrency})"
            )
        if skip_last_n_requests is None:
            skip_last_n_requests = max_concurrency
            logger.info(
                f"Auto-setting skip_last_n_requests={skip_last_n_requests}"
                f" (max_concurrency={max_concurrency})"
            )
    if skip_first_n_requests is None:
        skip_first_n_requests = 0
    if skip_last_n_requests is None:
        skip_last_n_requests = 0

    logger.info("Starting benchmark run")
    assert args.num_prompts is not None
    benchmark_result, benchmark_metrics = asyncio.run(
        benchmark(
            backend=backend,
            benchmark_task=benchmark_task,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            samples=samples,
            request_rate=request_rate,
            burstiness=args.burstiness,
            max_concurrency=max_concurrency,
            disable_tqdm=args.disable_tqdm,
            do_test_prompt=not args.skip_test_prompt,
            collect_gpu_stats=args.collect_gpu_stats,
            collect_cpu_stats=args.collect_cpu_stats,
            collect_server_stats=args.collect_server_stats,
            print_inputs_and_outputs=args.print_inputs_and_outputs,
            max_requests=args.num_prompts,
            skip_first_n_requests=skip_first_n_requests,
            skip_last_n_requests=skip_last_n_requests,
            max_output_len=args.max_output_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_benchmark_duration_s=args.max_benchmark_duration_s,
            warmup_delay_ms=args.chat_warmup_delay_ms,
            ignore_first_turn_stats=args.ignore_first_turn_stats,
            timing_data=None,
            lora_manager=lora_manager,
            trace_path=trace_path,
            trace_session=args.trace_session,
        )
    )

    # Validate that metrics are meaningful (no failures, not 0 or NaN)
    ok, validation_errors = benchmark_metrics.validate()
    if not ok:
        for err in validation_errors:
            logger.error(f"Benchmark result validation failed: {err}")
        logger.info("finished benchmark run: Failed.")
        sys.exit(1)

    # Save config and results to json
    if args.result_filename:
        logger.info("saving results")
        result_json: dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["benchmark_task"] = benchmark_task.value
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["num_prompts"] = benchmark_result["completed"]
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
            request_rate if request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Add LoRA metrics if present
        if "lora_metrics" in benchmark_result:
            result_json["lora_metrics"] = benchmark_result["lora_metrics"]

        # Save to file
        file_name = args.result_filename
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
    if (
        args.record_output_lengths
        and benchmark_task == BenchmarkTask.text_generation
    ):
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
            "top_k",
            "top_p",
        )
        output_lens_dict = {}
        output_lens_dict["args"] = {x: vars(args)[x] for x in args_to_save}
        output_lens_dict["output_lengths"] = benchmark_result["output_lens"]
        with open(args.record_output_lengths, "w") as f:
            yaml.dump(output_lens_dict, f)
    elif args.record_output_lengths:
        logger.warning(
            "--record-output-lengths is only supported for text-generation"
        )

    logger.info("finished benchmark run: Success.")


def parse_args(args: Sequence[str] | None = None) -> ServingBenchmarkConfig:
    """Parse command line arguments using ServingBenchmarkConfig with enhanced cli_parse_args().

    This function uses the generalized parse_benchmark_args function to handle
    config file inheritance and CLI argument parsing.

    Args:
        args: Command line arguments to parse. If None, parse from sys.argv.
    """
    parsed_args = parse_benchmark_args(
        config_class=ServingBenchmarkConfig,
        default_config_path=Path(__file__).parent
        / "configs/serving_config.yaml",
        description=BENCHMARK_SERVING_ARGPARSER_DESCRIPTION,
        args=args,
    )
    slim_parsed_args = dict(vars(parsed_args))
    # config_file is present in the parsed arguments, but isn't a part of the
    # config proper, so remove it before constructing the config
    slim_parsed_args.pop("config_file", None)
    return ServingBenchmarkConfig(**slim_parsed_args)


def main(args: Sequence[str] | None = None) -> None:
    parsed_args = parse_args(args)
    main_with_parsed_args(parsed_args)


if __name__ == "__main__":
    main()
