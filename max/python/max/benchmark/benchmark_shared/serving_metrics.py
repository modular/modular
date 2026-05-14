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

"""Metrics calculation for serving benchmarks."""

from __future__ import annotations

import logging
import statistics
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeGuard

from max.benchmark.benchmark_shared.metrics import (
    PixelGenAggregates,
    PixelGenerationBenchmarkResult,
    RatePercentileMetrics,
    ServingBenchmarkMetrics,
    SpecDecodeStats,
    StandardPercentileMetrics,
    SteadyStateResult,
    TextGenAggregates,
    TextGenerationBenchmarkResult,
    ThroughputMetrics,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncOutput,
    PixelGenerationRequestFuncOutput,
    RequestFuncOutput,
    measured_window_duration,
)
from max.benchmark.benchmark_shared.steady_state import detect_steady_state
from max.profiler.cpu import CPUMetrics
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from max.benchmark.benchmark_shared.server_metrics import ParsedMetrics
    from max.profiler.gpu import GPUStats

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
        tokenizer.encode(output.generated_text, add_special_tokens=False)
    )


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
) -> ServingBenchmarkMetrics:
    actual_output_lens: list[int] = []
    failures = 0
    failed_responses: list[RequestFuncOutput] = []
    itls: list[float] = []
    tpots: list[float] = []
    step_tpots: list[float] = []
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

        if output_len > 1:
            tpots.append((o.latency - o.ttft) / (output_len - 1))
        step_tpots += o.tpot
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
    per_turn_cached_token_rate: RatePercentileMetrics | None = (
        RatePercentileMetrics(per_turn_cached_token_rates, as_percent=True)
        if len(per_turn_cached_token_rates) > 0
        else None
    )

    text_data = TextGenAggregates(
        duration=measured_duration,
        completed=measured_count,
        failures=failures,
        request_throughput=measured_count / measured_duration,
        latency_ms=StandardPercentileMetrics(
            latencies or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        errors=[o.error for o in outputs],
        request_submit_times=[o.request_submit_time for o in outputs],
        request_complete_times=[o.request_complete_time for o in outputs],
        total_input=total_input,
        total_output=total_output,
        nonempty_response_chunks=nonempty_response_chunks,
        max_concurrent_conversations=max_concurrent_conversations,
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
        step_tpot_ms=StandardPercentileMetrics(
            step_tpots, scale_factor=1000.0, unit="ms"
        )
        if step_tpots
        else None,
        itl_ms=StandardPercentileMetrics(
            itls or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        max_input=max_input,
        max_output=max_output,
        max_total=max_total,
        global_cached_token_rate=global_cached_token_rate,
        per_turn_cached_token_rate=per_turn_cached_token_rate,
        skip_first_n_requests=skip_first_n_requests,
        skip_last_n_requests=skip_last_n_requests,
        input_lens=[o.prompt_len for o in outputs],
        output_lens=actual_output_lens,
        ttfts=[o.ttft for o in outputs],
        per_turn_cached_token_rates=per_turn_cached_token_rates,
    )

    return ServingBenchmarkMetrics(
        task_type="text",
        max_concurrency=max_concurrency or len(outputs),
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_metrics=cpu_metrics,
        metrics_by_endpoint=metrics_by_endpoint or {},
        text_data=text_data,
    )


def calculate_pixel_generation_metrics(
    outputs: Sequence[PixelGenerationRequestFuncOutput],
    dur_s: float,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    max_concurrency: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
) -> ServingBenchmarkMetrics:
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

    pixel_data = PixelGenAggregates(
        duration=measured_duration,
        completed=completed,
        failures=failures,
        request_throughput=completed / measured_duration,
        latency_ms=StandardPercentileMetrics(
            latencies or [float("nan")], scale_factor=1000.0, unit="ms"
        ),
        errors=[o.error for o in outputs],
        request_submit_times=[o.request_submit_time for o in outputs],
        request_complete_times=[o.request_complete_time for o in outputs],
        total_generated_outputs=total_generated_outputs,
        latencies=[o.latency for o in outputs],
        num_generated_outputs=[o.num_generated_outputs for o in outputs],
    )

    return ServingBenchmarkMetrics(
        task_type="pixel",
        max_concurrency=max_concurrency or len(outputs),
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_metrics=cpu_metrics,
        metrics_by_endpoint=metrics_by_endpoint or {},
        pixel_data=pixel_data,
    )


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


def build_pixel_generation_result(
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


def build_text_generation_result(
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

    ss_metrics: TextGenAggregates | None = None
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
            ).text_data
            assert ss_metrics is not None  # text-gen path always populates

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
