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

"""Result output helpers for serving benchmarks.

Printing, formatting, JSON/YAML persistence, and related utilities.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import statistics
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import yaml
from max.benchmark.benchmark_shared.config import (
    Backend,
    BenchmarkTask,
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
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
    SpecDecodeStats,
    TextGenerationBenchmarkResult,
)
from max.benchmark.benchmark_shared.server_metrics import print_server_metrics
from max.benchmark.benchmark_shared.utils import print_section

logger = logging.getLogger(__name__)


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
    record_max_concurrency: int | None,
) -> None:
    """Persist benchmark results to *result_filename*."""
    if not result_filename:
        return
    backend: Backend = args.backend
    client_args = args.model_dump()
    rr_client = str(request_rate) if request_rate < float("inf") else "inf"
    client_args["request_rate"] = rr_client
    client_args["max_concurrency"] = record_max_concurrency
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
        "max_concurrency": record_max_concurrency,
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


def save_output_lengths(
    args: ServingBenchmarkConfig,
    benchmark_result: TextGenerationBenchmarkResult
    | PixelGenerationBenchmarkResult,
    benchmark_task: BenchmarkTask,
    *,
    iteration_config: tuple[int | None, float] | None = None,
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
    if iteration_config is not None:
        mc_snap, rr_snap = iteration_config
        args_dict["max_concurrency"] = mc_snap
        args_dict["request_rate"] = rr_snap
    output_lens_dict["args"] = {x: args_dict[x] for x in args_to_save}
    output_lens_dict["output_lengths"] = benchmark_result.metrics.output_lens
    with open(args.record_output_lengths, "w") as f:
        yaml.dump(output_lens_dict, f)
