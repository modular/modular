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

"""Tests for TPOT computation and request timestamps in calculate_metrics."""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest
from max.benchmark.benchmark_shared.goodput import GoodputTargets
from max.benchmark.benchmark_shared.metrics import BenchmarkResult
from max.benchmark.benchmark_shared.request import (
    PixelGenerationRequestFuncOutput,
    RequestFuncOutput,
    ServerTokenStats,
)
from max.benchmark.benchmark_shared.serving_metrics import (
    _aggregate_gpu_stats,
    _per_turn_cache_retentions,
    build_text_generation_result,
    calculate_metrics,
    calculate_pixel_generation_metrics,
    compute_output_len,
)
from max.profiler.cpu import CPUMetrics
from max.profiler.gpu import GPUStats, MemoryStats, UtilizationStats

_MIB = 1024 * 1024


def _gpu_stats(*, util: int, used_mib: int, free_mib: int) -> GPUStats:
    """A minimal GPUStats shaped like the scraper/NVML recorder emit."""
    return GPUStats(
        memory=MemoryStats(
            total_bytes=(used_mib + free_mib) * _MIB,
            free_bytes=free_mib * _MIB,
            used_bytes=used_mib * _MIB,
            reserved_bytes=None,
        ),
        utilization=UtilizationStats(
            gpu_usage_percent=util, memory_activity_percent=None
        ),
        clocks=None,
    )


def _gpu_stats_no_memory(*, util: int) -> GPUStats:
    """A GPUStats with no memory info, as from a driver that lacks it."""
    return GPUStats(
        memory=None,
        utilization=UtilizationStats(
            gpu_usage_percent=util, memory_activity_percent=None
        ),
        clocks=None,
    )


_EMPTY_CPU_METRICS = CPUMetrics(
    user=0.0, user_percent=0.0, system=0.0, system_percent=0.0, elapsed=0.0
)


def _make_mock_tokenizer(token_counts: dict[str, int]) -> MagicMock:
    """Create a mock tokenizer that returns specified token counts.

    Args:
        token_counts: Mapping from generated text to the number of tokens.
    """
    tokenizer = MagicMock()

    def encode(text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(token_counts.get(text, 0)))

    tokenizer.encode.side_effect = encode
    return tokenizer


def _calculate(
    outputs: Sequence[RequestFuncOutput],
    *,
    dur_s: float,
    tokenizer: MagicMock,
    skip_first_n_requests: int = 0,
    skip_last_n_requests: int = 0,
    reject_outliers: bool = False,
    record_request_text: bool = False,
    goodput_targets: GoodputTargets | None = None,
    num_gpus: int | None = None,
) -> BenchmarkResult:
    """Runs `calculate_metrics` with no GPU stats and no concurrency limit."""
    return calculate_metrics(
        outputs=outputs,
        dur_s=dur_s,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=skip_first_n_requests,
        skip_last_n_requests=skip_last_n_requests,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
        reject_outliers=reject_outliers,
        record_request_text=record_request_text,
        goodput_targets=goodput_targets,
        num_gpus=num_gpus,
    )


def test_per_chunk_tpot_collected_from_outputs() -> None:
    """Per-chunk TPOT values are collected into step_tpot_ms."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello world",
        itl=[0.1, 0.2, 0.3],
        tpot=[0.05, 0.1, 0.15],
    )

    tokenizer = _make_mock_tokenizer({"hello world": 5})

    metrics = _calculate(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
    )

    assert metrics.text_data is not None

    # Per-chunk values [0.05, 0.1, 0.15] now live in step_tpot_ms.
    assert metrics.text_data.step_tpot_ms is not None
    assert math.isclose(metrics.text_data.step_tpot_ms.p50, 100.0, rel_tol=1e-3)
    # tpot_ms is per-request: (latency - ttft) / (output_len - 1)
    #                       = (1.0 - 0.1) / (5 - 1) = 0.225 s -> 225 ms.
    assert metrics.text_data.tpot_ms is not None
    assert math.isclose(metrics.text_data.tpot_ms.p50, 225.0, rel_tol=1e-3)


def test_tpot_both_definitions() -> None:
    """Both TPOT definitions are computed: per-request tpot_ms and per-step step_tpot_ms."""
    # Request 1: 10 output tokens, latency 1.0s, ttft 0.1s
    #   -> per-request tpot = 0.9 / 9 = 0.1 s
    output1 = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="ten tokens out",
        itl=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        tpot=[0.1] * 9,
    )

    # Request 2: 4 output tokens, latency 0.8s, ttft 0.2s
    #   -> per-request tpot = 0.6 / 3 = 0.2 s
    output2 = RequestFuncOutput(
        success=True,
        latency=0.8,
        ttft=0.2,
        prompt_len=5,
        generated_text="four tok",
        itl=[0.2, 0.2, 0.2],
        tpot=[0.2] * 3,
    )

    # Mock tokenizer: output1 -> 10 tokens, output2 -> 4 tokens
    tokenizer = _make_mock_tokenizer({"ten tokens out": 10, "four tok": 4})

    metrics = _calculate(
        outputs=[output1, output2],
        dur_s=2.0,
        tokenizer=tokenizer,
    )

    assert metrics.text_data is not None

    # Per-request tpots = [0.1, 0.2]; mean = 0.15 s -> 150 ms.
    assert metrics.text_data.tpot_ms is not None
    assert math.isclose(metrics.text_data.tpot_ms.mean, 150.0, rel_tol=1e-6)

    # Per-step step_tpots = [0.1]*9 + [0.2]*3; mean = 1.5/12 s -> 125 ms.
    assert metrics.text_data.step_tpot_ms is not None
    assert math.isclose(
        metrics.text_data.step_tpot_ms.mean, 125.0, rel_tol=1e-6
    )


def test_output_len_prefers_server_completion_tokens() -> None:
    """The server's completion count drives TPOT, not the re-tokenized text.

    With ``ignore_eos``, tokens past EOS can stream no text, so the text
    re-tokenizes to fewer tokens than the server generated.
    """
    # 10 generated tokens whose text re-tokenizes to 4, latency 1.0s, ttft
    # 0.1s -> tpot = 0.9 / 9 = 0.1 s.
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="four tok",
        itl=[0.1] * 9,
        tpot=[0.1] * 9,
        server_token_stats=ServerTokenStats(completion_tokens=10),
    )
    tokenizer = _make_mock_tokenizer({"four tok": 4})

    metrics = _calculate(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
    )

    assert metrics.text_data is not None
    assert metrics.text_data.total_output == 10
    assert metrics.text_data.tpot_ms is not None
    assert math.isclose(metrics.text_data.tpot_ms.mean, 100.0, rel_tol=1e-6)


def test_output_len_falls_back_to_text_without_usage() -> None:
    """A server that reports no usage is counted from its text."""
    output = RequestFuncOutput(success=True, generated_text="four tok")
    tokenizer = _make_mock_tokenizer({"four tok": 4})

    assert compute_output_len(tokenizer, output) == 4


def test_tpot_zero_decode_tokens() -> None:
    """When all requests produce <= 1 token, decode metrics are ``None``."""
    # Output with 1 token (only TTFT, no decode)
    output = RequestFuncOutput(
        success=True,
        latency=0.1,
        ttft=0.1,
        prompt_len=10,
        generated_text="a",
        itl=[],
        tpot=[],
    )

    # 1 output token, 1 completed -> decode_tokens = 0
    tokenizer = _make_mock_tokenizer({"a": 1})

    metrics = _calculate(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
    )

    assert metrics.text_data is not None

    # With no decode samples the decode-phase percentile metrics are
    # ``None`` (rather than a NaN-filled object), while prefill-phase
    # metrics that do have samples stay populated.
    assert metrics.text_data.tpot_ms is None
    assert metrics.text_data.itl_ms is None
    assert metrics.text_data.ttft_ms is not None


def test_empty_outputs_no_crash() -> None:
    """Empty outputs list doesn't crash."""
    tokenizer = _make_mock_tokenizer({})

    metrics = _calculate(
        outputs=[],
        dur_s=1.0,
        tokenizer=tokenizer,
    )

    assert metrics.text_data is not None

    assert metrics.text_data.completed == 0
    assert metrics.text_data.output_lens == []
    # With no samples, percentile metrics are ``None`` (not a NaN-filled
    # object) so the serialized JSON stays free of null-valued percentile
    # objects that strict downstream consumers reject.
    assert metrics.text_data.latency_ms is None
    assert metrics.text_data.ttft_ms is None
    assert metrics.text_data.tpot_ms is None
    assert metrics.text_data.itl_ms is None
    assert metrics.text_data.input_throughput is None
    assert metrics.text_data.output_throughput is None

    # And the empty case round-trips through JSON as ``null`` fields, not
    # ``{"p50": null, ...}`` objects (regression guard for MXTOOLS-45:
    # legacy NaN objects tripped the dashboard's per-field drop pass).
    dumped = json.loads(metrics.model_dump_json())
    assert dumped["text_data"]["tpot_ms"] is None
    assert dumped["text_data"]["latency_ms"] is None


def test_itl_metrics_unchanged() -> None:
    """ITL metrics remain unchanged by the TPOT refactor."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello world",
        itl=[0.1, 0.2, 0.3],
        tpot=[0.05, 0.1, 0.15],
    )

    tokenizer = _make_mock_tokenizer({"hello world": 5})

    metrics = _calculate(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
    )

    assert metrics.text_data is not None

    # ITL should be computed from the raw itl values [0.1, 0.2, 0.3] * 1000
    assert metrics.text_data.itl_ms is not None
    assert math.isclose(metrics.text_data.itl_ms.mean, 200.0, rel_tol=1e-3)
    assert math.isclose(metrics.text_data.itl_ms.p50, 200.0, rel_tol=1e-3)


def test_failed_requests_excluded() -> None:
    """Failed requests don't contribute to TPOT."""
    success_output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello",
        itl=[0.1, 0.2],
        tpot=[0.1, 0.2],
    )
    failed_output = RequestFuncOutput(
        success=False,
        error="test error",
        itl=[999.0],
        tpot=[999.0],
    )

    tokenizer = _make_mock_tokenizer({"hello": 3, "": 0})

    metrics = _calculate(
        outputs=[success_output, failed_output],
        dur_s=1.0,
        tokenizer=tokenizer,
    )

    assert metrics.text_data is not None

    # Only successful request's TPOT should be used
    assert metrics.text_data.completed == 1
    assert metrics.text_data.failures == 1
    # Per-request tpot_ms uses only the successful request, not [999.0].
    assert metrics.text_data.tpot_ms is not None
    assert metrics.text_data.tpot_ms.p50 < 500.0
    # Per-step step_tpot_ms uses only [0.1, 0.2] from the successful request.
    assert metrics.text_data.step_tpot_ms is not None
    assert metrics.text_data.step_tpot_ms.p50 < 500.0


def test_skip_last_n_requests() -> None:
    """Skipping last N requests excludes them from latency metrics."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="first",
            itl=[0.1, 0.2],
            tpot=[0.1, 0.2],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="second",
            itl=[0.1, 0.2],
            tpot=[0.1, 0.2],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.5,
            prompt_len=10,
            generated_text="third",
            itl=[0.9, 0.8],
            tpot=[0.9, 0.8],
        ),
    ]

    tokenizer = _make_mock_tokenizer({"first": 3, "second": 3, "third": 3})

    metrics_all = _calculate(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
    )

    metrics_skip_last = _calculate(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        skip_last_n_requests=1,
    )

    assert metrics_all.text_data is not None
    assert metrics_skip_last.text_data is not None

    # completed reflects the measured slice (first two, third was skipped).
    assert metrics_all.text_data.completed == 3
    assert metrics_skip_last.text_data.completed == 2
    # The last request's high TTFT (0.5s) is excluded from latency metrics.
    assert metrics_skip_last.text_data.ttft_ms is not None
    assert metrics_all.text_data.ttft_ms is not None
    assert (
        metrics_skip_last.text_data.ttft_ms.mean
        < metrics_all.text_data.ttft_ms.mean
    )


def test_skip_first_and_last_n_requests() -> None:
    """Skipping both first and last N requests works together."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.5,
            prompt_len=10,
            generated_text="first",
            itl=[0.1],
            tpot=[0.1],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="middle",
            itl=[0.1],
            tpot=[0.1],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.5,
            prompt_len=10,
            generated_text="last",
            itl=[0.1],
            tpot=[0.1],
        ),
    ]

    tokenizer = _make_mock_tokenizer({"first": 2, "middle": 2, "last": 2})

    metrics = _calculate(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
    )

    assert metrics.text_data is not None

    # Only the middle request is measured.
    assert metrics.text_data.completed == 1
    # Only the middle request's TTFT (0.1s = 100ms) should be measured
    assert metrics.text_data.ttft_ms is not None
    assert math.isclose(metrics.text_data.ttft_ms.mean, 100.0, rel_tol=1e-3)


def test_skip_last_with_cancelled_requests() -> None:
    """skip_last counts from end of completed requests, not the output array."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="first",
            itl=[0.1],
            tpot=[0.1],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.2,
            prompt_len=10,
            generated_text="second",
            itl=[0.1],
            tpot=[0.1],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.5,
            prompt_len=10,
            generated_text="third",
            itl=[0.1],
            tpot=[0.1],
        ),
        # Cancelled requests pad the output array
        RequestFuncOutput(cancelled=True),
        RequestFuncOutput(cancelled=True),
        RequestFuncOutput(cancelled=True),
    ]

    tokenizer = _make_mock_tokenizer({"first": 2, "second": 2, "third": 2})

    metrics = _calculate(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
    )

    assert metrics.text_data is not None

    # skip_last is applied to successful (3), not to the padded cancelled
    # entries, so only the middle successful request is measured.
    assert metrics.text_data.completed == 1
    # Only the second request should be measured (skip first 1, last 1)
    assert metrics.text_data.ttft_ms is not None
    assert math.isclose(metrics.text_data.ttft_ms.mean, 200.0, rel_tol=1e-3)


def test_skip_all_requests_warns() -> None:
    """Skipping all requests emits a warning."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="only",
            itl=[0.1],
            tpot=[0.1],
        ),
    ]

    tokenizer = _make_mock_tokenizer({"only": 2})

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _calculate(
            outputs=outputs,
            dur_s=1.0,
            tokenizer=tokenizer,
            skip_first_n_requests=1,
            skip_last_n_requests=1,
        )
        assert len(w) == 1
        assert "excluded" in str(w[0].message).lower()


def test_calculate_pixel_generation_metrics() -> None:
    outputs = [
        PixelGenerationRequestFuncOutput(
            success=True,
            latency=1.0,
            num_generated_outputs=1,
        ),
        PixelGenerationRequestFuncOutput(
            success=True,
            latency=2.0,
            num_generated_outputs=2,
        ),
        PixelGenerationRequestFuncOutput(success=False, error="bad request"),
    ]

    metrics = calculate_pixel_generation_metrics(
        outputs=outputs,
        dur_s=5.0,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        max_concurrency=None,
        collect_gpu_stats=False,
    )

    assert metrics.pixel_data is not None

    assert metrics.pixel_data.completed == 2
    assert metrics.pixel_data.failures == 1
    assert math.isclose(
        metrics.pixel_data.request_throughput, 0.4, rel_tol=1e-6
    )
    assert metrics.pixel_data.total_generated_outputs == 3
    assert metrics.pixel_data.latency_ms is not None
    assert math.isclose(
        metrics.pixel_data.latency_ms.mean, 1500.0, rel_tol=1e-6
    )


def test_request_submit_time_defaults_to_none() -> None:
    """request_submit_time field defaults to None."""
    output = RequestFuncOutput()
    assert output.request_submit_time is None
    assert output.request_complete_time is None


def test_request_submit_time_set_on_output() -> None:
    """request_submit_time can be set and is preserved through metrics."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello",
        itl=[0.1],
        tpot=[0.1],
        request_submit_time=100.5,
    )

    assert output.request_submit_time == 100.5

    tokenizer = _make_mock_tokenizer({"hello": 2})
    metrics = _calculate(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
    )
    assert metrics.text_data is not None
    # Metrics are computed normally regardless of submit time
    assert metrics.text_data.completed == 1


def test_measured_duration_uses_measured_window() -> None:
    """When skipping is applied, duration and throughput reflect the measured window only."""
    # 10 warmup requests: submit every 1s starting at t=0, each takes 5s.
    # 100 steady requests: submit every 0.1s starting at t=10, each takes 0.5s.
    # 10 tail requests: submit every 2s starting at t=30, each takes 5s.
    warmup = [
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=100,
            generated_text="warmup",
            itl=[0.1] * 4,
            tpot=[0.1] * 4,
            request_submit_time=float(i),
        )
        for i in range(10)
    ]
    steady = [
        RequestFuncOutput(
            success=True,
            latency=0.5,
            ttft=0.05,
            prompt_len=10,
            generated_text="steady",
            itl=[0.05] * 4,
            tpot=[0.05] * 4,
            request_submit_time=10.0 + i * 0.1,
        )
        for i in range(100)
    ]
    tail = [
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=100,
            generated_text="tail",
            itl=[0.1] * 4,
            tpot=[0.1] * 4,
            request_submit_time=30.0 + i * 2.0,
        )
        for i in range(10)
    ]
    outputs = warmup + steady + tail

    tokenizer = _make_mock_tokenizer({"warmup": 5, "steady": 5, "tail": 5})

    # Full run wall clock passed as dur_s. 60s is much longer than the
    # measured window should be.
    full_run_duration = 60.0

    metrics = _calculate(
        outputs=outputs,
        dur_s=full_run_duration,
        tokenizer=tokenizer,
        skip_first_n_requests=10,
        skip_last_n_requests=10,
    )

    assert metrics.text_data is not None

    # Measured = the 100 steady requests.
    assert metrics.text_data.completed == 100
    # total_input / total_output are over the measured 100 only.
    assert metrics.text_data.total_input == 100 * 10
    assert metrics.text_data.total_output == 100 * 5
    # Measured window: first steady submits at t=10.0; last steady
    # completes at t = 10.0 + 99*0.1 + 0.5 = 20.4.
    expected_window = 20.4 - 10.0
    assert math.isclose(
        metrics.text_data.duration, expected_window, rel_tol=1e-6
    )
    # Request throughput is over the measured window, not the full run.
    assert math.isclose(
        metrics.text_data.request_throughput,
        100 / expected_window,
        rel_tol=1e-6,
    )
    # Crucially, throughput does NOT use the full run duration.
    assert metrics.text_data.request_throughput > 100 / full_run_duration


def test_measured_duration_falls_back_when_no_timestamps() -> None:
    """Without submit timestamps, duration falls back to dur_s."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello",
        itl=[0.1],
        tpot=[0.1],
    )
    # No request_submit_time -> fallback to dur_s.
    tokenizer = _make_mock_tokenizer({"hello": 2})
    metrics = _calculate(
        outputs=[output],
        dur_s=3.0,
        tokenizer=tokenizer,
    )
    assert metrics.text_data is not None
    assert math.isclose(metrics.text_data.duration, 3.0, rel_tol=1e-9)
    assert math.isclose(
        metrics.text_data.request_throughput, 1.0 / 3.0, rel_tol=1e-9
    )


def test_skipped_tokens_excluded_from_totals() -> None:
    """Tokens from skipped warmup/tail requests don't inflate total_input/total_output."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=10_000,  # huge warmup prompt
            generated_text="warm",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="mid",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=6.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=10_000,  # huge tail prompt
            generated_text="tail",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=8.0,
        ),
    ]
    tokenizer = _make_mock_tokenizer({"warm": 999, "mid": 3, "tail": 999})

    metrics = _calculate(
        outputs=outputs,
        dur_s=20.0,
        tokenizer=tokenizer,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
    )

    assert metrics.text_data is not None

    # Only the middle request counts toward totals.
    assert metrics.text_data.total_input == 10
    assert metrics.text_data.total_output == 3
    assert metrics.text_data.completed == 1


def test_request_complete_time_property() -> None:
    """request_complete_time property = submit_time + latency."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.5,
            ttft=0.1,
            prompt_len=10,
            generated_text="a",
            request_submit_time=100.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=2.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="b",
            request_submit_time=101.0,
        ),
    ]

    assert outputs[0].request_complete_time is not None
    assert outputs[1].request_complete_time is not None
    assert math.isclose(outputs[0].request_complete_time, 101.5)
    assert math.isclose(outputs[1].request_complete_time, 103.0)

    # Arrays stay index-aligned (same length as submit_times)
    submit_times = [o.request_submit_time for o in outputs]
    complete_times = [o.request_complete_time for o in outputs]
    assert len(submit_times) == len(complete_times)


def test_skip_uses_submit_time_for_head_complete_time_for_tail() -> None:
    """skip_first targets earliest submits, skip_last targets latest completes.

    Mirrors a multi-turn flat list, where outputs arrive in
    ``[session0_turns, session1_turns, ...]`` block order rather than
    chronological order. The dispatch-order slice would silently target the
    wrong requests; the timing-based selection picks the right ones.
    """
    # Three "sessions" of two turns each, flattened in block order. Submit/
    # complete times are intentionally non-monotonic vs. iteration order.
    outputs = [
        # session 0: starts first, finishes early
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s0t0",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.0,  # earliest submit overall
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s0t1",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=2.0,
        ),
        # session 1: middle
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s1t0",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.5,
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s1t1",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=3.0,
        ),
        # session 2: starts last, last turn finishes last (latest complete)
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s2t0",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=1.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s2t1",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=4.0,  # complete = 9.0, latest overall
        ),
    ]

    tokenizer = _make_mock_tokenizer(
        {"s0t0": 2, "s0t1": 2, "s1t0": 2, "s1t1": 2, "s2t0": 2, "s2t1": 2}
    )

    metrics = _calculate(
        outputs=outputs,
        dur_s=10.0,
        tokenizer=tokenizer,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
    )

    assert metrics.text_data is not None

    # Head trim drops s0t0 (submit=0.0, the earliest).
    # Tail trim drops s2t1 (complete=9.0, the latest).
    # Dispatch-order slicing would have dropped s0t0 and s2t1 here too —
    # but only because outputs[0] happens to be the earliest submit and
    # outputs[-1] happens to be the latest complete. Use the asymmetric
    # case below to distinguish.
    assert metrics.text_data.completed == 4


def test_skip_distinguishes_dispatch_order_from_timing() -> None:
    """Head/tail trim uses timing, not iteration order.

    Construct a list where the latest-completing request is *not* at the end
    of the list and the earliest-submitting request is *not* at the front,
    so a dispatch-order slice would target the wrong requests.
    """
    outputs = [
        # outputs[0]: NOT earliest submit; latest complete (slow request).
        RequestFuncOutput(
            success=True,
            latency=10.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="block_late_complete",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=2.0,  # complete = 12.0, latest
        ),
        # outputs[1]: earliest submit, completes mid-pack.
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="block_early_submit",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.0,  # earliest
        ),
        # outputs[2]: middle on both axes.
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="middle",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=1.0,
        ),
    ]

    tokenizer = _make_mock_tokenizer(
        {"block_late_complete": 2, "block_early_submit": 2, "middle": 2}
    )

    metrics = _calculate(
        outputs=outputs,
        dur_s=15.0,
        tokenizer=tokenizer,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
    )

    assert metrics.text_data is not None

    # Head trim drops outputs[1] (block_early_submit, submit=0.0).
    # Tail trim drops outputs[0] (block_late_complete, complete=12.0).
    # Both removed → only outputs[2] (middle) is measured.
    assert metrics.text_data.completed == 1
    # If the old dispatch-order slice were still in effect it would have
    # kept outputs[1] (the index-1 middle slot) and dropped outputs[0] and
    # outputs[2]. The "middle" generated_text being the sole measured
    # output proves the timing-based selection is what's running.
    # tail_drop on a request with latency=10.0 that we kept would have
    # inflated total_input. Verify only middle's prompt_len (10) is counted.
    assert metrics.text_data.total_input == 10


def test_skip_first_overlaps_with_skip_last_drops_both() -> None:
    """When the same output is both an earliest submit and a latest complete
    (slow request submitted first that finishes last), it ends up in both
    drop sets. The set-based filter handles this naturally — it gets dropped
    once."""
    outputs = [
        # The "warmup" request: submitted first AND completes last.
        RequestFuncOutput(
            success=True,
            latency=20.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="slow_warmup",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.0,  # complete = 20.0
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="a",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=1.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="b",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=2.0,
        ),
    ]

    tokenizer = _make_mock_tokenizer({"slow_warmup": 2, "a": 2, "b": 2})

    metrics = _calculate(
        outputs=outputs,
        dur_s=21.0,
        tokenizer=tokenizer,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
    )

    assert metrics.text_data is not None

    # slow_warmup is in both head_drop_ids and tail_drop_ids — set union
    # handles dedupe. The remaining "a" and "b" are both measured.
    assert metrics.text_data.completed == 2


def _make_tokenizer_mock(tokens_per_output: int = 5) -> MagicMock:
    """Return a tokenizer mock whose encode call returns a fixed token count."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(tokens_per_output))
    return tokenizer


def _make_request_func_output(
    *,
    prompt_len: int = 10,
    generated_text: str = "hello world",
    latency: float = 1.0,
    ttft: float = 0.1,
    itl: list[float] | None = None,
    request_submit_time: float | None = 0.0,
) -> RequestFuncOutput:
    return RequestFuncOutput(
        success=True,
        latency=latency,
        ttft=ttft,
        prompt_len=prompt_len,
        generated_text=generated_text,
        itl=itl or [],
        request_submit_time=request_submit_time,
    )


def _make_stable_request_func_output(submit_time: float) -> RequestFuncOutput:
    """Return a RequestFuncOutput with stable TTFT and TPOT suitable for steady-state detection."""
    tpot = [0.02, 0.02, 0.02]
    return RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.05,
        prompt_len=10,
        generated_text="hello world",
        itl=tpot,
        tpot=tpot,
        request_submit_time=submit_time,
    )


def test_build_text_generation_result_not_detected_falls_back_to_trim() -> None:
    """With too few requests (no steady state), build_text_generation_result uses head/tail trim.

    When steady state is not detected the full-run path runs with the
    caller-supplied skip_first / skip_last parameters unchanged.  Diagnostic
    scalars still appear on the result.
    """
    outputs = [_make_request_func_output() for _ in range(3)]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert result.steady_state_detected is False
    assert result.steady_state_window_count == 0
    assert result.steady_state_warning is not None
    assert "Too few" in result.steady_state_warning
    # The metrics themselves should still be computed (fall-back path).
    assert result.text_data is not None
    assert result.text_data.completed == 3


def _outputs_with_completion_tokens(
    counts: list[int],
) -> list[RequestFuncOutput]:
    outputs = []
    for count in counts:
        output = _make_request_func_output()
        output.server_token_stats = ServerTokenStats(completion_tokens=count)
        outputs.append(output)
    return outputs


def test_build_text_generation_result_attaches_aggregate_server_stats() -> None:
    """Without sessions, server stats are attached flat in output order."""
    outputs = _outputs_with_completion_tokens([3, 1, 2])
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=1.0,
        tokenizer=_make_tokenizer_mock(),
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert result.session_server_stats is None
    assert result.aggregate_server_stats == [
        o.server_token_stats for o in outputs
    ]


def test_build_text_generation_result_attaches_session_server_stats() -> None:
    """With sessions, server stats are grouped per session."""
    a, b, c, d = _outputs_with_completion_tokens([1, 2, 3, 4])
    by_session = {"10": [a], "anon": [b], "2": [c, d]}
    result = build_text_generation_result(
        outputs=[a, b, c, d],
        benchmark_duration=1.0,
        tokenizer=_make_tokenizer_mock(),
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        outputs_by_session=by_session,
    )

    assert result.aggregate_server_stats is None
    assert result.session_server_stats is not None
    assert list(result.session_server_stats) == ["2", "10", "anon"]
    assert result.session_server_stats["2"] == [
        c.server_token_stats,
        d.server_token_stats,
    ]


def test_build_text_generation_result_all_failed_suppresses_ss_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run with no successful requests must not emit a steady-state warning.

    Regression test for PERF-2615: a run where every request failed logged
    ``Steady-state detection: Too few valid requests (0 of N total)`` at
    WARNING, which read as a detection bug. It is a run failure (surfaced by
    the failure count), so the steady-state path stays quiet and falls back.
    """
    outputs = [
        RequestFuncOutput(success=False, latency=0.0, ttft=0.0, prompt_len=10)
        for _ in range(200)
    ]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    with caplog.at_level(logging.WARNING):
        result = build_text_generation_result(
            outputs=outputs,
            benchmark_duration=1.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=None,
            skip_first_n_requests=0,
            skip_last_n_requests=0,
            max_concurrency=32,  # detection runs (not skipped)
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
        )

    assert not any(
        "Steady-state detection:" in r.message and r.levelno >= logging.WARNING
        for r in caplog.records
    )
    assert result.steady_state_detected is False
    assert result.text_data is not None


def test_build_text_generation_result_too_few_but_some_success_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Some successful-but-unusable requests below threshold still warn.

    Distinguishes the genuine "too few valid" case (worth surfacing) from the
    all-failed run above (suppressed).
    """
    outputs = [_make_request_func_output() for _ in range(3)]  # 3 successful
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    with caplog.at_level(logging.WARNING):
        build_text_generation_result(
            outputs=outputs,
            benchmark_duration=1.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=None,
            skip_first_n_requests=0,
            skip_last_n_requests=0,
            max_concurrency=32,
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
        )

    assert any("Steady-state detection:" in r.message for r in caplog.records)


def test_build_text_generation_result_detected_uses_window_with_rejection() -> (
    None
):
    """With enough stable requests, build_text_generation_result uses the MAD window + rejection.

    Checks that:
    - steady_state_detected is True
    - the single reported metric set reflects the window (not full run)
    - outlier rejection diagnostic is present
    """
    outputs = [_make_stable_request_func_output(float(i)) for i in range(200)]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=200.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert result.steady_state_detected is True
    assert result.steady_state_window_count is not None
    assert result.steady_state_window_count > 0
    assert result.steady_state_mode == "full"
    assert result.steady_state_warning is None
    assert result.num_outliers_rejected is not None
    assert result.text_data is not None
    # The reported TTFT should be very close to 50 ms (all inputs are 0.05 s).
    assert result.text_data.ttft_ms is not None
    assert result.text_data.ttft_ms.mean == pytest.approx(50.0, rel=0.05)


def test_per_gpu_throughput_spans_full_run_on_steady_state_path() -> None:
    """On the detected steady-state path, the per-GPU figure still spans
    the full run, not the detected window.

    The 60 warmup requests carry no TPOT data, so steady-state detection
    filters them out of its valid set and the window covers only the 100
    steady requests. They are still successful requests, so the per-GPU
    throughput must divide every request's tokens by the full
    first-submit to last-complete span.
    """
    warmup = [
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=100,
            generated_text="warmup",
            itl=[],
            tpot=[],
            request_submit_time=float(i),
        )
        for i in range(60)
    ]
    steady = [
        RequestFuncOutput(
            success=True,
            latency=0.5,
            ttft=0.05,
            prompt_len=10,
            generated_text="steady",
            itl=[0.05] * 4,
            tpot=[0.05] * 4,
            request_submit_time=60.0 + i * 0.1,
        )
        for i in range(100)
    ]
    tokenizer = _make_mock_tokenizer({"warmup": 5, "steady": 5})
    result = build_text_generation_result(
        outputs=warmup + steady,
        benchmark_duration=80.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        num_gpus=1,
    )

    assert result.steady_state_detected is True
    text = result.text_data
    assert text is not None
    # The reported metric set covers the detected window only: the 100
    # steady requests, spanning t=60.0 to t=70.4.
    assert text.completed == 100
    assert math.isclose(text.duration, 10.4, rel_tol=1e-9)
    # Per-GPU throughput covers every successful request over the full
    # span: first submit t=0.0, last steady completes at t=70.4. Tokens:
    # 60 * (100 + 5) + 100 * (10 + 5) = 7800.
    assert text.total_tokens_per_second_per_gpu is not None
    assert math.isclose(
        text.total_tokens_per_second_per_gpu, 7800.0 / 70.4, rel_tol=1e-9
    )


def test_build_text_generation_result_concurrency_one_falls_back() -> None:
    """At concurrency=1 detection is skipped; the full-run path (with trim) is used.

    Outlier rejection is NOT applied on the concurrency-1 fallback path.
    """
    outputs = [_make_stable_request_func_output(float(i)) for i in range(50)]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=50.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=1,  # detection skipped
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    # Detection was skipped → not detected.
    assert result.steady_state_detected is False
    assert result.steady_state_warning is None  # skipped, not failed
    assert result.num_outliers_rejected == 0  # no rejection on fallback
    assert result.text_data is not None
    assert result.text_data.completed == 50


def test_build_text_generation_result_filtering_disabled_uses_skip_counts() -> (
    None
):
    """With filtering disabled, a run detection would accept uses the trim path.

    The caller's skip counts apply, no outliers are rejected, and the
    diagnostics report ``"disabled"`` rather than a failed detection.
    """
    outputs = [_make_stable_request_func_output(float(i)) for i in range(200)]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=200.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=10,
        skip_last_n_requests=5,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        steady_state_filtering=False,
    )

    assert result.steady_state_mode == "disabled"
    assert result.steady_state_detected is None
    assert result.steady_state_window_count is None
    assert result.steady_state_warning is None
    assert result.num_outliers_rejected is None
    assert result.text_data is not None
    assert result.text_data.completed == 185
    assert result.to_result_dict()["steady_state_mode"] == "disabled"


def test_build_text_generation_result_filtering_disabled_keeps_all_requests() -> (
    None
):
    """Requests the detector would drop are kept when filtering is disabled.

    A reply that arrives in one chunk has no per-chunk TPOT, so detection
    treats it as invalid and leaves it out of the window, and a TTFT outlier
    is rejected inside the window. Neither happens with filtering disabled.
    """
    outputs = [_make_stable_request_func_output(float(i)) for i in range(200)]
    for i in range(10):
        single_chunk = _make_stable_request_func_output(50.0 + i + 0.5)
        single_chunk.itl = []
        single_chunk.tpot = []
        outputs.append(single_chunk)
    # Spread TTFTs so the outlier rejection's MAD is nonzero.
    for i, output in enumerate(outputs):
        output.ttft = 0.05 + 0.001 * (i % 5)
    ttft_outlier = _make_stable_request_func_output(100.5)
    ttft_outlier.ttft = 10.0
    ttft_outlier.latency = 11.0
    outputs.append(ttft_outlier)
    outputs.sort(key=lambda o: o.request_submit_time or 0.0)
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)

    def build(steady_state_filtering: bool) -> BenchmarkResult:
        return build_text_generation_result(
            outputs=outputs,
            benchmark_duration=211.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=None,
            skip_first_n_requests=0,
            skip_last_n_requests=0,
            max_concurrency=None,
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
            steady_state_filtering=steady_state_filtering,
        )

    filtered = build(steady_state_filtering=True)
    unfiltered = build(steady_state_filtering=False)

    assert filtered.text_data is not None
    assert unfiltered.text_data is not None
    assert filtered.steady_state_detected is True
    assert filtered.text_data.completed <= 201
    assert filtered.text_data.ttft_ms is not None
    assert filtered.text_data.ttft_ms.mean < 60.0
    assert unfiltered.text_data.completed == 211
    assert unfiltered.text_data.ttft_ms is not None
    assert unfiltered.text_data.ttft_ms.mean == pytest.approx(
        statistics.mean(o.ttft for o in outputs) * 1000.0
    )


def _make_failed_request_func_output(submit_time: float) -> RequestFuncOutput:
    return RequestFuncOutput(
        success=False,
        latency=0.0,
        ttft=0.0,
        prompt_len=10,
        error="No text content captured from the response",
        request_submit_time=submit_time,
    )


@pytest.mark.parametrize("steady_state_filtering", [True, False])
def test_build_text_generation_result_excludes_failures(
    steady_state_filtering: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Failed requests leave the metrics alike with or without filtering.

    They are excluded from the metric set but kept in ``request_records``,
    and they don't shift which requests the steady-state window selects.
    """
    outputs: list[RequestFuncOutput] = []
    for i in range(200):
        outputs.append(_make_stable_request_func_output(float(i)))
        if i % 50 == 10:
            outputs.append(_make_failed_request_func_output(i + 0.5))
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)

    with caplog.at_level(logging.WARNING):
        result = build_text_generation_result(
            outputs=outputs,
            benchmark_duration=200.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=None,
            skip_first_n_requests=0,
            skip_last_n_requests=0,
            max_concurrency=None,
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
            steady_state_filtering=steady_state_filtering,
        )

    assert any(
        "Excluding 4 failed requests" in r.message for r in caplog.records
    )
    assert result.text_data is not None
    assert result.text_data.failures == 0
    if steady_state_filtering:
        assert result.steady_state_detected is True
        assert result.text_data.completed == result.steady_state_window_count
    else:
        assert result.text_data.completed == 200
    records = result.text_data.request_records
    assert len(records) == 204
    failed = [r for r in records if not r.success]
    assert len(failed) == 4
    assert not any(r.measured for r in failed)


def test_build_text_generation_result_all_failed_still_counts_failures() -> (
    None
):
    """With no successful request, failures are kept and still counted."""
    outputs = [_make_failed_request_func_output(float(i)) for i in range(5)]
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=5.0,
        tokenizer=_make_tokenizer_mock(tokens_per_output=5),
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=32,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        steady_state_filtering=False,
    )

    assert result.text_data is not None
    assert result.text_data.failures == 5
    assert result.text_data.completed == 0


def test_build_text_generation_result_with_outlier_inputs() -> None:
    """Synthetic run: stable-phase requests with realistic spread + extreme TTFT outliers.

    Validates three properties end-to-end:
    (a) detection selects a steady window (from the stable phase),
    (b) outlier rejection drops the extreme TTFT values when the window
        has enough natural spread for MAD > 0,
    (c) the not-detected fallback path handles concurrency=1 without rejection.
    """
    import math
    import random

    random.seed(42)

    # 110 stable-phase requests with small natural spread so MAD > 0.
    # TTFT alternates ≈ 0.04-0.06 s → median 0.05 s, MAD ≈ 0.01 s.
    stable = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.04 + (i % 2) * 0.02,  # alternates 0.04 / 0.06
            prompt_len=10,
            generated_text="hello world",
            itl=[0.02, 0.02, 0.02],
            tpot=[0.02, 0.02, 0.02],
            request_submit_time=float(i),
        )
        for i in range(110)
    ]
    # Inject 3 extreme TTFT outliers mixed into the stable range.
    # |mz| = 0.6745 * (50.0 - 0.05) / 0.01 ≈ 3372 >> 3.5 → must be rejected.
    extreme_outliers = [
        RequestFuncOutput(
            success=True,
            latency=51.0,
            ttft=50.0,  # extreme: 50 s
            prompt_len=10,
            generated_text="hello world",
            itl=[0.02],
            tpot=[0.02],
            request_submit_time=float(50 + i * 20),  # scattered in the middle
        )
        for i in range(3)
    ]
    all_outputs = stable + extreme_outliers
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)

    # (a+b) With concurrency > 1, detection may find a window; if it does,
    # rejection should drop the extreme TTFTs so the mean stays near 50 ms.
    # We also simply verify the result is sane and completes without error.
    result_high = build_text_generation_result(
        outputs=all_outputs,
        benchmark_duration=120.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=32,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )
    assert result_high.text_data is not None
    assert result_high.steady_state_detected is not None
    # num_outliers_rejected diagnostic is always present on the text path.
    assert result_high.num_outliers_rejected is not None

    # (c) At concurrency=1, detection is skipped; trim is used; outliers are
    # included but the fallback path should still complete without error.
    result_one = build_text_generation_result(
        outputs=all_outputs,
        benchmark_duration=120.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=1,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )
    assert result_one.steady_state_detected is False
    assert result_one.steady_state_warning is None  # skipped, not failed
    assert result_one.num_outliers_rejected == 0  # no rejection on fallback
    assert result_one.text_data is not None
    assert result_one.text_data.ttft_ms is not None
    assert not math.isnan(result_one.text_data.ttft_ms.mean)


def _turn(
    session_id: str,
    turn_index: int,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
) -> RequestFuncOutput:
    """A successful multi-turn output carrying server token stats."""
    return RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=prompt_tokens,
        generated_text="x",
        itl=[0.1],
        tpot=[0.1],
        session_id=session_id,
        turn_index=turn_index,
        server_token_stats=ServerTokenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        ),
    )


class TestPerTurnCacheRetention:
    """`_per_turn_cache_retentions` block-aligns vs the prior turn's context."""

    def test_clean_hit_is_full_retention(self) -> None:
        # prev context = 500 + 100 = 600. expected_cacheable =
        # (600 // 128 - 1) * 128 = 3 * 128 = 384. cached(turn1) == 384 -> 1.0.
        turns = [
            _turn(
                "s",
                0,
                prompt_tokens=500,
                completion_tokens=100,
                cached_tokens=0,
            ),
            _turn(
                "s",
                1,
                prompt_tokens=600,
                completion_tokens=100,
                cached_tokens=384,
            ),
        ]
        assert _per_turn_cache_retentions(turns, 128) == [pytest.approx(1.0)]

    def test_partial_drop(self) -> None:
        # cached(turn1) == 192 of a 384 ceiling -> 0.5.
        turns = [
            _turn(
                "s",
                0,
                prompt_tokens=500,
                completion_tokens=100,
                cached_tokens=0,
            ),
            _turn(
                "s",
                1,
                prompt_tokens=600,
                completion_tokens=100,
                cached_tokens=192,
            ),
        ]
        assert _per_turn_cache_retentions(turns, 128) == [pytest.approx(0.5)]

    def test_first_turn_excluded_and_sessions_isolated(self) -> None:
        # Two sessions, each 2 turns; only the second turn of each is checked,
        # and a session's retention is computed against its own prior turn.
        turns = [
            _turn("a", 0, 500, 100, 0),
            _turn("a", 1, 600, 100, 384),  # clean -> 1.0
            _turn("b", 0, 500, 100, 0),
            _turn("b", 1, 600, 100, 0),  # full miss -> 0.0
        ]
        result = sorted(_per_turn_cache_retentions(turns, 128))
        assert result == [pytest.approx(0.0), pytest.approx(1.0)]

    def test_sub_block_context_skipped(self) -> None:
        # prev context 100 -> (100 // 128 - 1) clamps to 0 blocks -> skipped.
        turns = [
            _turn("s", 0, 60, 40, 0),
            _turn("s", 1, 100, 40, 0),
        ]
        assert _per_turn_cache_retentions(turns, 128) == []

    def test_single_turn_outputs_have_no_retention_metric(self) -> None:
        # No session_id/turn_index (single-turn) -> per_turn_cache_retention None.
        output = RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="x",
            itl=[0.1],
        )
        metrics = _calculate(
            outputs=[output],
            dur_s=1.0,
            tokenizer=_make_mock_tokenizer({"x": 3}),
        )
        assert metrics.text_data is not None
        assert metrics.text_data.per_turn_cache_retention is None


class TestRequestRecords:
    """Per-request records: the joinable form of the unaligned arrays.

    ``input_lens`` follows dispatch order while ``output_lens`` lists
    failures before successes, so the two cannot be zipped. These tests pin
    the properties that make records usable where the arrays are not: one
    record per dispatched request, in dispatch order, each carrying its own
    outcome.
    """

    def _outputs(self) -> list[RequestFuncOutput]:
        return [
            RequestFuncOutput(
                success=True,
                latency=1.0,
                ttft=0.1,
                prompt_len=10,
                generated_text="alpha",
                itl=[0.1] * 4,
            ),
            RequestFuncOutput(
                success=False,
                error="boom",
                prompt_len=20,
                generated_text="",
            ),
            RequestFuncOutput(
                success=True,
                latency=2.0,
                ttft=0.2,
                prompt_len=30,
                generated_text="gamma",
                itl=[0.2] * 4,
            ),
        ]

    def _metrics(
        self, *, record_request_text: bool = False, skip_first: int = 0
    ) -> BenchmarkResult:
        tokenizer = _make_mock_tokenizer({"alpha": 5, "gamma": 5, "": 0})
        return _calculate(
            outputs=self._outputs(),
            dur_s=3.0,
            tokenizer=tokenizer,
            skip_first_n_requests=skip_first,
            record_request_text=record_request_text,
        )

    def test_one_record_per_dispatched_request(self) -> None:
        """Failures included: "failed here, succeeded there" is the finding."""
        text_data = self._metrics().text_data
        assert text_data is not None

        assert [r.index for r in text_data.request_records] == [0, 1, 2]
        assert [r.prompt_len for r in text_data.request_records] == [10, 20, 30]
        assert [r.success for r in text_data.request_records] == [
            True,
            False,
            True,
        ]

    def test_records_are_joinable_where_the_arrays_are_not(self) -> None:
        """The arrays disagree on which request index 0 is; records do not."""
        text_data = self._metrics().text_data
        assert text_data is not None

        # output_lens puts the failure first, input_lens keeps dispatch
        # order — so zipping them pairs the failure's 0 with prompt_len=10.
        assert text_data.input_lens == [10, 20, 30]
        assert text_data.output_lens[0] == 0

        by_index = {r.index: r for r in text_data.request_records}
        assert by_index[1].output_len == 0
        assert by_index[1].error == "boom"
        assert by_index[0].output_len == 5

    def test_a_failed_request_reports_no_latency(self) -> None:
        """0.0 would read as an instant response rather than no response."""
        text_data = self._metrics().text_data
        assert text_data is not None

        failed = text_data.request_records[1]
        assert failed.ttft_ms is None
        assert failed.latency_ms is None
        assert failed.tpot_ms is None

    def test_generated_text_is_opt_in(self) -> None:
        """It is the only unbounded field, so a run must ask for it."""
        default = self._metrics().text_data
        opted_in = self._metrics(record_request_text=True).text_data
        assert default is not None and opted_in is not None

        assert all(r.generated_text is None for r in default.request_records)
        assert [r.generated_text for r in opted_in.request_records] == [
            "alpha",
            "",
            "gamma",
        ]

    def test_trimmed_requests_are_recorded_but_not_measured(self) -> None:
        """A record outside the aggregates' window has to say so."""
        text_data = self._metrics(skip_first=1).text_data
        assert text_data is not None

        measured = {r.index for r in text_data.request_records if r.measured}
        # The trim drops the first success; the failure was never measured.
        assert measured == {2}
        assert len(text_data.request_records) == 3

    def test_records_survive_the_result_dict(self) -> None:
        """The JSON blob is what an offline consumer actually reads."""
        text_data = self._metrics(record_request_text=True).text_data
        assert text_data is not None

        payload = json.loads(json.dumps(text_data.to_result_dict()))

        records = payload["request_records"]
        assert [r["index"] for r in records] == [0, 1, 2]
        assert records[2]["generated_text"] == "gamma"


def test_steady_state_records_keep_dispatch_indices() -> None:
    """The steady-state path hands a filtered slice to calculate_metrics.

    Indexing that slice would renumber the requests and destroy the key two
    runs pair on, so records are built over the full dispatch list.
    """
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1 + 0.001 * i,
            prompt_len=10,
            generated_text="alpha",
            itl=[0.1] * 4,
            request_submit_time=float(i),
        )
        for i in range(12)
    ]
    tokenizer = _make_mock_tokenizer({"alpha": 5})

    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=12.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=4,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert result.text_data is not None
    records = result.text_data.request_records
    assert [r.index for r in records] == list(range(12))


def _constrained_split_outputs(
    pairs: int = 2,
) -> list[RequestFuncOutput]:
    """Alternating constrained/unconstrained requests, separated by TTFT/TPOT.

    Constrained: ttft 0.3s, 5 tokens over 1.1s -> tpot = 0.8/4 = 0.2s.
    Unconstrained: ttft 0.1s, 5 tokens over 0.5s -> tpot = 0.4/4 = 0.1s.

    Submit times are distinct so the head/tail skip window has an order to
    sort by.
    """
    outputs = []
    for i in range(pairs):
        constrained = RequestFuncOutput(
            success=True,
            latency=1.1,
            ttft=0.3,
            prompt_len=10,
            generated_text="five tokens here now",
            response_format_constrained=True,
        )
        constrained.request_submit_time = float(2 * i)
        outputs.append(constrained)
        unconstrained = RequestFuncOutput(
            success=True,
            latency=0.5,
            ttft=0.1,
            prompt_len=10,
            generated_text="five tokens here now",
        )
        unconstrained.request_submit_time = float(2 * i + 1)
        outputs.append(unconstrained)
    return outputs


def _calculate_for(
    outputs: list[RequestFuncOutput],
    skip_first: int = 0,
    skip_last: int = 0,
) -> object:
    tokenizer = _make_mock_tokenizer({"five tokens here now": 5})
    return _calculate(
        outputs=outputs,
        dur_s=2.0,
        tokenizer=tokenizer,
        skip_first_n_requests=skip_first,
        skip_last_n_requests=skip_last,
    )


def test_constrained_split_separates_the_two_halves() -> None:
    """Each half is computed only over its own requests."""
    metrics = _calculate_for(_constrained_split_outputs())
    text = metrics.text_data  # type: ignore[attr-defined]
    assert text is not None

    assert text.ttft_ms_constrained is not None
    assert text.ttft_ms_unconstrained is not None
    assert math.isclose(text.ttft_ms_constrained.mean, 300.0, rel_tol=1e-6)
    assert math.isclose(text.ttft_ms_unconstrained.mean, 100.0, rel_tol=1e-6)

    assert text.tpot_ms_constrained is not None
    assert text.tpot_ms_unconstrained is not None
    assert math.isclose(text.tpot_ms_constrained.mean, 200.0, rel_tol=1e-6)
    assert math.isclose(text.tpot_ms_unconstrained.mean, 100.0, rel_tol=1e-6)

    # The unsplit metrics still cover every request.
    assert text.ttft_ms is not None
    assert math.isclose(text.ttft_ms.mean, 200.0, rel_tol=1e-6)
    assert text.constrained_request_rate == 0.5


def test_single_population_run_reports_no_split() -> None:
    """A split is only meaningful across two populations. With one, the
    populated side would restate ``ttft_ms``, so neither side is reported."""
    outputs = [
        o
        for o in _constrained_split_outputs()
        if not o.response_format_constrained
    ]
    text = _calculate_for(outputs).text_data  # type: ignore[attr-defined]
    assert text is not None
    assert text.ttft_ms_constrained is None
    assert text.ttft_ms_unconstrained is None
    assert text.tpot_ms_constrained is None
    assert text.tpot_ms_unconstrained is None
    assert text.ttft_ms is not None
    assert text.constrained_request_rate == 0.0

    for o in outputs:
        o.response_format_constrained = True
    text = _calculate_for(outputs).text_data  # type: ignore[attr-defined]
    assert text.ttft_ms_constrained is None
    assert text.ttft_ms_unconstrained is None
    assert text.constrained_request_rate == 1.0


def test_constrained_rate_counts_over_the_measured_window() -> None:
    """The rate shares a denominator with the other aggregates.

    Skip counts are auto-derived from ``--max-concurrency``, so most runs
    measure fewer requests than they completed.
    """
    outputs = _constrained_split_outputs(pairs=10)
    for o in outputs:
        o.response_format_constrained = True

    text = _calculate_for(  # type: ignore[attr-defined]
        outputs, skip_first=3, skip_last=2
    ).text_data
    assert text is not None
    assert text.constrained_request_rate == 1.0


def test_tpot_split_is_gated_on_its_own_population() -> None:
    """TTFT and TPOT can have different populations, so each pair stands or
    falls on its own.

    A request that emits a single token has a TTFT but no TPOT. When every
    constrained request is that short, the TPOT split has one empty half and
    the surviving half would only restate ``tpot_ms``.
    """
    tokenizer = _make_mock_tokenizer({"five tokens here now": 5, "one": 1})
    outputs = []
    for i in range(2):
        short = RequestFuncOutput(
            success=True,
            latency=0.4,
            ttft=0.3,
            prompt_len=10,
            generated_text="one",
            response_format_constrained=True,
        )
        short.request_submit_time = float(2 * i)
        outputs.append(short)
        long = RequestFuncOutput(
            success=True,
            latency=0.5,
            ttft=0.1,
            prompt_len=10,
            generated_text="five tokens here now",
        )
        long.request_submit_time = float(2 * i + 1)
        outputs.append(long)

    text = _calculate(
        outputs=outputs,
        dur_s=2.0,
        tokenizer=tokenizer,
    ).text_data
    assert text is not None
    # Both sides produced a first token, so the TTFT split stands.
    assert text.ttft_ms_constrained is not None
    assert text.ttft_ms_unconstrained is not None
    # Only one side produced a second token, so the TPOT split does not.
    assert text.tpot_ms_constrained is None
    assert text.tpot_ms_unconstrained is None


def test_constrained_split_reaches_result_groups() -> None:
    """A field on the aggregates alone is not rendered; the console summary and
    the stored JSON both read ``result_groups``."""
    metrics = _calculate_for(_constrained_split_outputs())
    latency = metrics.result_groups.latency_stats  # type: ignore[attr-defined]
    assert latency is not None
    assert latency.ttft_ms_constrained is not None
    assert latency.tpot_ms_unconstrained is not None
    mix = metrics.result_groups.request_mix  # type: ignore[attr-defined]
    assert mix is not None
    assert mix.constrained_request_rate == 0.5


def test_constrained_split_reaches_the_flat_result_dict() -> None:
    metrics = _calculate_for(_constrained_split_outputs())
    d = metrics.text_data.to_result_dict()  # type: ignore[attr-defined]
    assert math.isclose(d["mean_ttft_ms_constrained"], 300.0, rel_tol=1e-6)
    assert math.isclose(d["mean_tpot_ms_unconstrained"], 100.0, rel_tol=1e-6)
    assert d["constrained_request_rate"] == 0.5


def test_conformance_rate_counts_only_judged_constrained_requests() -> None:
    """A constrained request that was never judged must not be counted as a
    failure; an unconstrained run reports no rate at all."""
    conforming = RequestFuncOutput(
        success=True,
        latency=1.1,
        ttft=0.3,
        prompt_len=10,
        generated_text="five tokens here now",
        response_format_constrained=True,
        response_format_conformed=True,
    )
    failing = RequestFuncOutput(
        success=True,
        latency=1.1,
        ttft=0.3,
        prompt_len=10,
        generated_text="five tokens here now",
        response_format_constrained=True,
        response_format_conformed=False,
    )
    unjudged = RequestFuncOutput(
        success=True,
        latency=1.1,
        ttft=0.3,
        prompt_len=10,
        generated_text="five tokens here now",
        response_format_constrained=True,
    )
    text = _calculate_for([conforming, failing, unjudged]).text_data  # type: ignore[attr-defined]
    assert text is not None
    # 1 of the 2 judged, not 1 of the 3 constrained.
    assert text.constrained_conformance_rate == 0.5
    assert text.constrained_request_rate == 1.0


def test_no_conformance_rate_without_constrained_requests() -> None:
    outputs = [
        o
        for o in _constrained_split_outputs()
        if not o.response_format_constrained
    ]
    text = _calculate_for(outputs).text_data  # type: ignore[attr-defined]
    assert text is not None
    assert text.constrained_conformance_rate is None


def test_conformance_rate_reaches_the_summary_group() -> None:
    metrics = _calculate_for(
        [
            RequestFuncOutput(
                success=True,
                latency=1.1,
                ttft=0.3,
                prompt_len=10,
                generated_text="five tokens here now",
                response_format_constrained=True,
                response_format_conformed=False,
            )
        ]
    )
    mix = metrics.result_groups.request_mix  # type: ignore[attr-defined]
    assert mix is not None
    assert mix.constrained_conformance_rate == 0.0


def test_aggregate_gpu_stats_disabled_or_empty() -> None:
    """Collection off, no snapshots, or all-empty snapshots yield nothing."""
    assert _aggregate_gpu_stats(collect_gpu_stats=False, gpu_metrics=[]) == (
        [],
        [],
        [],
        {},
    )
    assert _aggregate_gpu_stats(collect_gpu_stats=True, gpu_metrics=None) == (
        [],
        [],
        [],
        {},
    )
    # Snapshots with no devices report nothing rather than raising.
    assert _aggregate_gpu_stats(
        collect_gpu_stats=True, gpu_metrics=[{}, {}]
    ) == ([], [], [], {})


def test_aggregate_gpu_stats_single_device_across_snapshots() -> None:
    """Peak used = max, available = min free, util = mean over snapshots."""
    snapshots = [
        {"n:gpu0": _gpu_stats(util=80, used_mib=100, free_mib=900)},
        {"n:gpu0": _gpu_stats(util=90, used_mib=300, free_mib=700)},
    ]
    peak, avail, util, _ = _aggregate_gpu_stats(
        collect_gpu_stats=True, gpu_metrics=snapshots
    )
    assert peak == [300.0]
    assert avail == [700.0]
    assert util == [85.0]


def test_aggregate_gpu_stats_tolerates_changing_device_set() -> None:
    """A device set that varies across snapshots never raises KeyError.

    This is the multi-node failure mode: a Service scrape (or a flaky
    endpoint) makes successive snapshots disagree on which devices appear.
    Each device is aggregated only over the snapshots that contain it.
    """
    snapshots = [
        # Only node-1 present this interval.
        {
            "node-1:gpu0": _gpu_stats(util=100, used_mib=200, free_mib=800),
        },
        # Both nodes present.
        {
            "node-1:gpu0": _gpu_stats(util=50, used_mib=400, free_mib=600),
            "node-2:gpu0": _gpu_stats(util=40, used_mib=100, free_mib=900),
        },
        # Only node-2 present.
        {
            "node-2:gpu0": _gpu_stats(util=60, used_mib=300, free_mib=700),
        },
    ]

    peak, avail, util, _ = _aggregate_gpu_stats(
        collect_gpu_stats=True, gpu_metrics=snapshots
    )

    # Union of devices, sorted: node-1:gpu0 then node-2:gpu0.
    assert peak == [400.0, 300.0]
    assert avail == [600.0, 700.0]
    # node-1: mean(100, 50) = 75; node-2: mean(40, 60) = 50. Each averaged
    # only over the snapshots in which it appears.
    assert util == [75.0, 50.0]
    # Reported mean GPU util is the mean across all engine devices seen.
    assert statistics.mean(util) == 62.5


def test_aggregate_gpu_stats_without_memory_info() -> None:
    """No memory info yields empty memory series; utilization is unaffected."""
    snapshots = [
        {"n:gpu0": _gpu_stats_no_memory(util=70)},
        {"n:gpu0": _gpu_stats_no_memory(util=90)},
    ]
    peak, avail, util, _ = _aggregate_gpu_stats(
        collect_gpu_stats=True, gpu_metrics=snapshots
    )
    assert peak == []
    assert avail == []
    assert util == [80.0]


def test_aggregate_gpu_stats_mixed_memory_availability() -> None:
    """Devices without memory info drop out of the memory series only."""
    snapshots = [
        {
            "n:gpu0": _gpu_stats(util=50, used_mib=100, free_mib=900),
            "n:gpu1": _gpu_stats_no_memory(util=60),
        },
    ]
    peak, avail, util, _ = _aggregate_gpu_stats(
        collect_gpu_stats=True, gpu_metrics=snapshots
    )
    # Only gpu0 reported memory; gpu1 contributes utilization alone.
    assert peak == [100.0]
    assert avail == [900.0]
    assert util == [50.0, 60.0]


def test_aggregate_gpu_stats_splits_utilization_by_group() -> None:
    """Each group gets its devices' utilization, in the mapping's order.

    A device with no group still counts toward the overall list, and a
    grouped device absent from every snapshot adds nothing.
    """
    snapshot = {
        "GPU-a": _gpu_stats(util=96, used_mib=100, free_mib=900),
        "GPU-b": _gpu_stats(util=94, used_mib=100, free_mib=900),
        "GPU-c": _gpu_stats(util=38, used_mib=100, free_mib=900),
        "GPU-d": _gpu_stats(util=50, used_mib=100, free_mib=900),
    }
    _, _, util, by_group = _aggregate_gpu_stats(
        collect_gpu_stats=True,
        gpu_metrics=[snapshot],
        gpu_device_groups={
            "GPU-c": "decode",
            "GPU-a": "prefill",
            "GPU-b": "prefill",
            "GPU-z": "decode",
        },
    )
    assert util == [96.0, 94.0, 38.0, 50.0]
    assert by_group == {"decode": [38.0], "prefill": [96.0, 94.0]}
    assert list(by_group) == ["decode", "prefill"]


def test_tool_rates_count_offers_and_calls_separately() -> None:
    """The request rate is over every measured request; the call rate only
    over the ones that offered tools, as production's two counters are."""
    outputs = _constrained_split_outputs(pairs=4)
    for o in outputs:
        o.response_format_constrained = False
    for i, o in enumerate(outputs[:4]):
        o.tools_offered = True
        o.tool_call_returned = i < 3
    text = _calculate_for(outputs).text_data  # type: ignore[attr-defined]
    assert text is not None
    assert text.tool_request_rate == 4 / len(outputs)
    assert text.tool_call_response_rate == 0.75


def test_tool_rates_reach_result_groups_and_the_flat_result_dict() -> None:
    """The console and stored JSON read ``result_groups`` and the flat dict,
    not the aggregates."""
    outputs = _constrained_split_outputs()
    for o in outputs:
        o.tools_offered = True
    outputs[0].tool_call_returned = True
    metrics = _calculate_for(outputs)
    mix = metrics.result_groups.request_mix  # type: ignore[attr-defined]
    assert mix is not None
    assert mix.tool_request_rate == 1.0
    assert mix.tool_call_response_rate == 0.25
    d = metrics.text_data.to_result_dict()  # type: ignore[attr-defined]
    assert d["tool_request_rate"] == 1.0
    assert d["tool_call_response_rate"] == 0.25


def test_tool_call_rate_is_none_without_tool_requests() -> None:
    outputs = [
        o
        for o in _constrained_split_outputs()
        if not o.response_format_constrained
    ]
    text = _calculate_for(outputs).text_data  # type: ignore[attr-defined]
    assert text is not None
    assert text.tool_request_rate == 0.0
    assert text.tool_call_response_rate is None


def test_image_and_lora_rates_count_over_measured_requests() -> None:
    outputs = _constrained_split_outputs(pairs=4)
    for i, o in enumerate(outputs):
        o.carries_image = i < 2
        o.lora_id = "adapter-a" if i < 6 else None
    metrics = _calculate_for(outputs)
    text = metrics.text_data  # type: ignore[attr-defined]
    assert text is not None
    assert text.image_request_rate == 2 / len(outputs)
    assert text.lora_request_rate == 6 / len(outputs)
    mix = metrics.result_groups.request_mix  # type: ignore[attr-defined]
    assert mix is not None
    assert mix.image_request_rate == text.image_request_rate
    assert mix.lora_request_rate == text.lora_request_rate
    d = text.to_result_dict()
    assert d["image_request_rate"] == text.image_request_rate
    assert d["lora_request_rate"] == text.lora_request_rate


def _goodput_outputs() -> list[RequestFuncOutput]:
    """Requests that each test one goodput rule against ttft:200 tpot:150.

    Ten-token replies have TPOT ``(latency - ttft) / 9``.
    """

    def out(
        text: str,
        *,
        ttft: float,
        latency: float,
        submit: float,
        success: bool = True,
    ) -> RequestFuncOutput:
        return RequestFuncOutput(
            success=success,
            latency=latency,
            ttft=ttft,
            prompt_len=10,
            generated_text=text,
            request_submit_time=submit,
        )

    return [
        # First submitted, so --skip-first-n-requests 1 drops it even
        # though it meets every target.
        out("skipped", ttft=0.05, latency=0.5, submit=0.0),
        # TPOT 0.1 s: meets both.
        out("good", ttft=0.1, latency=1.0, submit=1.0),
        # TTFT 0.3 s misses.
        out("slow first", ttft=0.3, latency=1.2, submit=1.1),
        # TPOT 1.9 / 9 = 0.211 s misses.
        out("slow decode", ttft=0.1, latency=2.0, submit=1.2),
        # One token, so no TPOT: misses a TPOT target.
        out("one", ttft=0.1, latency=0.1, submit=1.3),
        # Failed requests are not measured.
        out("", ttft=0.0, latency=0.1, submit=1.4, success=False),
    ]


_GOODPUT_TOKENIZER_COUNTS = {
    "skipped": 10,
    "good": 10,
    "slow first": 10,
    "slow decode": 10,
    "one": 1,
}


def _goodput_metrics(
    targets: GoodputTargets | None,
) -> BenchmarkResult:
    return _calculate(
        outputs=_goodput_outputs(),
        dur_s=10.0,
        tokenizer=_make_mock_tokenizer(_GOODPUT_TOKENIZER_COUNTS),
        skip_first_n_requests=1,
        goodput_targets=targets,
    )


def test_goodput_counts_requests_meeting_every_target() -> None:
    metrics = _goodput_metrics(GoodputTargets(ttft_ms=200.0, tpot_ms=150.0))
    text = metrics.text_data
    assert text is not None
    assert text.completed == 4
    assert text.goodput_attainment == pytest.approx(0.25)
    # One good request over the measured window, 1.0 s to 3.2 s.
    assert text.request_goodput == pytest.approx(1 / 2.2)
    assert text.request_goodput == pytest.approx(0.25 * text.request_throughput)
    assert (text.goodput_ttft_ms, text.goodput_tpot_ms) == (200.0, 150.0)
    assert text.goodput_e2el_ms is None

    # The result's per-request records give the same answer.
    rr = [r for r in text.request_records if r.measured and r.success]
    good = [
        r
        for r in rr
        if r.ttft_ms is not None
        and r.tpot_ms is not None
        and r.ttft_ms <= 200.0
        and r.tpot_ms <= 150.0
    ]
    assert text.request_goodput == pytest.approx(len(good) / text.duration)
    assert text.goodput_attainment == pytest.approx(len(good) / len(rr))

    d = metrics.to_result_dict()
    assert d["request_goodput"] == pytest.approx(1 / 2.2)
    assert d["goodput_attainment"] == pytest.approx(0.25)
    assert d["goodput_ttft_ms"] == 200.0
    assert "goodput_e2el_ms" not in d
    assert metrics.result_groups is not None
    summary = metrics.result_groups.summary
    assert summary.request_goodput == pytest.approx(1 / 2.2)
    assert summary.goodput_attainment == pytest.approx(0.25)
    assert summary.goodput_tpot_ms == 150.0


def test_goodput_e2el_target_ignores_missing_tpot() -> None:
    """Only the given targets apply, so a one-token reply can meet e2el."""
    text = _goodput_metrics(GoodputTargets(e2el_ms=1500.0)).text_data
    assert text is not None
    assert text.goodput_attainment == pytest.approx(0.75)


def test_goodput_unreported_ttft_misses_a_ttft_target() -> None:
    """An endpoint that reports no TTFT leaves it 0.0, which is not a pass."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=0.5,
            ttft=0.0,
            prompt_len=10,
            generated_text="good",
            request_submit_time=0.0,
        )
    ]

    def attainment(targets: GoodputTargets) -> float | None:
        text = _calculate(
            outputs=outputs,
            dur_s=1.0,
            tokenizer=_make_mock_tokenizer(_GOODPUT_TOKENIZER_COUNTS),
            goodput_targets=targets,
        ).text_data
        assert text is not None
        return text.goodput_attainment

    assert attainment(GoodputTargets(ttft_ms=200.0)) == 0.0
    assert attainment(GoodputTargets(e2el_ms=1000.0)) == 1.0


def test_goodput_unset_reports_nothing() -> None:
    metrics = _goodput_metrics(None)
    text = metrics.text_data
    assert text is not None
    for name in (
        "request_goodput",
        "goodput_attainment",
        "goodput_ttft_ms",
        "goodput_tpot_ms",
        "goodput_e2el_ms",
    ):
        assert getattr(text, name) is None
        assert name not in metrics.to_result_dict()
        assert metrics.result_groups is not None
        assert getattr(metrics.result_groups.summary, name) is None


def test_build_text_generation_result_passes_goodput_targets() -> None:
    result = build_text_generation_result(
        outputs=_goodput_outputs(),
        benchmark_duration=10.0,
        tokenizer=_make_mock_tokenizer(_GOODPUT_TOKENIZER_COUNTS),
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        steady_state_filtering=False,
        goodput_targets=GoodputTargets(ttft_ms=200.0, tpot_ms=150.0),
    )
    assert result.text_data is not None
    assert result.text_data.goodput_attainment == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Interactivity (per-user token rate) and per-GPU throughput.
# ---------------------------------------------------------------------------

# Two requests whose streams end with a trailing usage chunk: latency runs
# past the last content chunk, so the content-bounded and full-response
# bases differ visibly. Token counts come from the mock tokenizer below.
#
# Request A: ttft 0.1 s, 9 content gaps of 0.05 s (sum 0.45 s), 10 tokens,
#   latency 1.0 s (0.45 s of trailing usage chunk).
#   content ITL = 0.45 / 9 = 0.05 s; TPOT = 0.9 / 9 = 0.1 s;
#   content e2e = (0.1 + 0.45) / 10 = 0.055 s.
# Request B: ttft 0.2 s, 3 content gaps of 0.1 s (sum 0.3 s), 4 tokens,
#   latency 0.8 s (0.3 s of trailing usage chunk).
#   content ITL = 0.3 / 3 = 0.1 s; TPOT = 0.6 / 3 = 0.2 s;
#   content e2e = (0.2 + 0.3) / 4 = 0.125 s.
_INTERACTIVITY_TOKENIZER_COUNTS = {
    "ten tokens out": 10,
    "four tok": 4,
    "six tokens here": 6,
}


def _interactivity_outputs() -> list[RequestFuncOutput]:
    """Two measured requests with distinct content-bounded ITLs and TPOTs."""
    return [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="ten tokens out",
            itl=[0.05] * 9,
        ),
        RequestFuncOutput(
            success=True,
            latency=0.8,
            ttft=0.2,
            prompt_len=5,
            generated_text="four tok",
            itl=[0.1] * 3,
        ),
    ]


def _calculate_interactivity(
    outputs: list[RequestFuncOutput],
    *,
    num_gpus: int | None = None,
    dur_s: float = 2.0,
    reject_outliers: bool = False,
) -> BenchmarkResult:
    return _calculate(
        outputs=outputs,
        dur_s=dur_s,
        tokenizer=_make_mock_tokenizer(_INTERACTIVITY_TOKENIZER_COUNTS),
        num_gpus=num_gpus,
        reject_outliers=reject_outliers,
    )


def test_interactivity_pins_both_bases_on_a_trailing_usage_chunk() -> None:
    """Content-bounded interactivity and full-response interactivity are
    separate figures, and a request with a trailing usage chunk tells them
    apart.

    Content ITLs [0.05, 0.1] s: mean 0.075 -> 13.33 tok/s, p50 0.075 ->
    13.33, p75 0.0875 -> 11.43, p90 0.095 -> 10.53, p95 0.0975 -> 10.26,
    p99 0.0995 -> 10.05.
    TPOTs [0.1, 0.2] s: mean 0.15 -> 6.67 tok/s, p50 0.15 -> 6.67,
    p75 0.175 -> 5.71, p90 0.19 -> 5.26, p95 0.195 -> 5.13, p99 0.199 ->
    5.03.
    """
    text = _calculate_interactivity(_interactivity_outputs()).text_data
    assert text is not None
    assert text.interactivity_tok_s is not None
    assert text.full_response_interactivity_tok_s is not None

    assert text.interactivity_tok_s.mean == pytest.approx(1.0 / 0.075)
    assert text.interactivity_tok_s.p50 == pytest.approx(1.0 / 0.075)
    assert text.interactivity_tok_s.p75 == pytest.approx(1.0 / 0.0875)
    assert text.interactivity_tok_s.p90 == pytest.approx(1.0 / 0.095)
    assert text.interactivity_tok_s.p95 == pytest.approx(1.0 / 0.0975)
    assert text.interactivity_tok_s.p99 == pytest.approx(1.0 / 0.0995)

    assert text.full_response_interactivity_tok_s.mean == pytest.approx(
        1.0 / 0.15
    )
    assert text.full_response_interactivity_tok_s.p50 == pytest.approx(
        1.0 / 0.15
    )
    assert text.full_response_interactivity_tok_s.p75 == pytest.approx(
        1.0 / 0.175
    )
    assert text.full_response_interactivity_tok_s.p90 == pytest.approx(
        1.0 / 0.19
    )
    assert text.full_response_interactivity_tok_s.p95 == pytest.approx(
        1.0 / 0.195
    )
    assert text.full_response_interactivity_tok_s.p99 == pytest.approx(
        1.0 / 0.199
    )

    # The two bases differ on this stream: the content-bounded figure sits
    # above the full-response one, which pays for the trailing usage chunk.
    assert (
        text.interactivity_tok_s.mean
        > text.full_response_interactivity_tok_s.mean
    )

    # The reciprocal of the mean is NOT the mean of the reciprocals:
    # 1 / mean([0.05, 0.1]) = 13.33 tok/s, but mean([20, 10]) = 15 tok/s.
    mean_of_reciprocals = statistics.mean(1.0 / v for v in (0.05, 0.1))
    assert not math.isclose(
        text.interactivity_tok_s.mean, mean_of_reciprocals, rel_tol=1e-6
    )
    assert math.isclose(mean_of_reciprocals, 15.0, rel_tol=1e-6)


def test_full_response_interactivity_matches_tpot_ms_statistics() -> None:
    """Without outlier rejection, every full-response interactivity
    statistic is exactly 1000 / the matching ``tpot_ms`` statistic."""
    outputs = _interactivity_outputs()
    # A third request: (2.0 - 0.5) / (6 - 1) = 0.3 s TPOT.
    outputs.append(
        RequestFuncOutput(
            success=True,
            latency=2.0,
            ttft=0.5,
            prompt_len=7,
            generated_text="six tokens here",
            itl=[0.25] * 5,
        )
    )
    text = _calculate_interactivity(outputs).text_data
    assert text is not None
    assert text.tpot_ms is not None
    assert text.full_response_interactivity_tok_s is not None

    for stat in ("mean", "p50", "p90", "p95", "p99"):
        tpot_ms = getattr(text.tpot_ms, stat)
        intvty = getattr(text.full_response_interactivity_tok_s, stat)
        assert math.isclose(intvty, 1000.0 / tpot_ms, rel_tol=1e-9)


def test_e2e_normalized_interactivity_is_content_bounded() -> None:
    """E2E normalized interactivity inverts the content-bounded per-request
    e2e latency, (ttft + sum(itl)) / output_tokens, so TTFT is folded in.

    Golden values: request A (0.1 + 0.45) / 10 = 0.055 s, request B
    (0.2 + 0.3) / 4 = 0.125 s; mean 0.09 -> 11.11 tok/s, p50 0.09 ->
    11.11, p75 0.1075 -> 9.30, p90 0.118 -> 8.47, p95 0.1215 -> 8.23,
    p99 0.1243 -> 8.04.
    """
    text = _calculate_interactivity(_interactivity_outputs()).text_data
    assert text is not None
    assert text.e2e_normalized_interactivity_tok_s is not None
    e2e = text.e2e_normalized_interactivity_tok_s

    assert e2e.mean == pytest.approx(1.0 / 0.09)
    assert e2e.p50 == pytest.approx(1.0 / 0.09)
    assert e2e.p75 == pytest.approx(1.0 / 0.1075)
    assert e2e.p90 == pytest.approx(1.0 / 0.118)
    assert e2e.p95 == pytest.approx(1.0 / 0.1215)
    assert e2e.p99 == pytest.approx(1.0 / 0.1243)

    # TTFT is included: dropping request A's ttft from its basis would give
    # 0.45 / 10 = 0.045 s and a visibly larger figure.
    assert e2e.mean < 1.0 / 0.045


def test_interactivity_uses_unrejected_per_request_values() -> None:
    """The published figures apply no outlier rejection, so interactivity
    must come from the raw series even when ``tpot_ms`` rejects outliers.

    The third request's 2.0 s TPOT is an Iglewicz-Hoaglin outlier the
    rejection path drops from ``tpot_ms``; the full-response interactivity
    still inverts the raw mean (0.1, 0.2, 2.0) -> 0.7667 s.
    """
    outputs = _interactivity_outputs()
    # TPOT = (18.3 - 0.3) / 9 = 2.0 s; content ITL = 1.8 / 9 = 0.2 s.
    outputs.append(
        RequestFuncOutput(
            success=True,
            latency=18.3,
            ttft=0.3,
            prompt_len=7,
            generated_text="ten tokens out",
            itl=[0.2] * 9,
        )
    )
    metrics = _calculate_interactivity(outputs, reject_outliers=True)
    text = metrics.text_data
    assert text is not None
    assert text.tpot_ms is not None
    assert text.full_response_interactivity_tok_s is not None

    # The outlier was rejected from tpot_ms ...
    assert text.tpot_ms.mean < 1000.0 * 0.7667
    # ... but not from the interactivity series.
    assert text.full_response_interactivity_tok_s.mean == pytest.approx(
        1.0 / ((0.1 + 0.2 + 2.0) / 3)
    )
    assert not math.isclose(
        text.full_response_interactivity_tok_s.mean,
        1000.0 / text.tpot_ms.mean,
        rel_tol=1e-6,
    )


def test_interactivity_reaches_result_groups_and_flat_dict() -> None:
    """The console and stored JSON read ``result_groups`` and the flat dict,
    not the aggregates."""
    metrics = _calculate_interactivity(_interactivity_outputs())
    assert metrics.result_groups is not None
    throughput = metrics.result_groups.throughput_stats
    assert throughput is not None
    assert throughput.interactivity_tok_s is not None
    assert throughput.full_response_interactivity_tok_s is not None
    assert throughput.e2e_normalized_interactivity_tok_s is not None
    assert throughput.interactivity_tok_s.mean == pytest.approx(1.0 / 0.075)

    d = metrics.to_result_dict()
    assert d["mean_interactivity_tok_s"] == pytest.approx(1.0 / 0.075)
    assert d["p90_interactivity_tok_s"] == pytest.approx(1.0 / 0.095)
    assert d["p99_interactivity_tok_s"] == pytest.approx(1.0 / 0.0995)
    assert d["mean_full_response_interactivity_tok_s"] == pytest.approx(
        1.0 / 0.15
    )
    assert d["p75_e2e_normalized_interactivity_tok_s"] == pytest.approx(
        1.0 / 0.1075
    )
    assert "p50_interactivity_tok_s" in d


def test_per_gpu_total_token_throughput_arithmetic() -> None:
    """(total input + output tokens) / full span / GPU count.

    The two requests carry 10 + 5 = 15 prompt tokens and 10 + 4 = 14
    generated tokens over the 2 s fallback window, so a 2-GPU server
    reads (15 + 14) / 2 / 2 = 7.25 tok/s per GPU.
    """
    metrics = _calculate_interactivity(
        _interactivity_outputs(), num_gpus=2, dur_s=2.0
    )
    assert metrics.num_gpus == 2
    text = metrics.text_data
    assert text is not None
    assert text.total_tokens_per_second_per_gpu is not None
    assert math.isclose(
        text.total_tokens_per_second_per_gpu, 7.25, rel_tol=1e-9
    )

    assert metrics.result_groups is not None
    summary = metrics.result_groups.summary
    assert summary.num_gpus == 2
    assert summary.total_tokens_per_second_per_gpu is not None
    assert math.isclose(
        summary.total_tokens_per_second_per_gpu, 7.25, rel_tol=1e-9
    )
    assert metrics.to_result_dict()[
        "total_tokens_per_second_per_gpu"
    ] == pytest.approx(7.25)


def test_per_gpu_throughput_spans_all_successful_requests_despite_trims() -> (
    None
):
    """The per-GPU window is the full first-submit to last-complete span of
    every successful request; the skip_first/skip_last trim shrinks the
    other aggregates' window but not this one.

    Three requests (prompt = output tokens): R1 spans [0.0, 1.0],
    R2 [0.5, 2.5], R3 [1.0, 2.0]. skip_first=1 drops R1, skip_last=1 drops
    R2 (last completed), so the measured window is R3 alone -- but the
    per-GPU figure still divides 120 tokens by the full 2.5 s span.
    """
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="ten tokens out",
            itl=[0.05] * 9,
            request_submit_time=0.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=2.0,
            ttft=0.1,
            prompt_len=20,
            generated_text="twenty tokens out",
            itl=[0.05] * 3,
            request_submit_time=0.5,
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=30,
            generated_text="thirty tokens out",
            itl=[0.05] * 5,
            request_submit_time=1.0,
        ),
    ]
    tokenizer_counts = {
        "ten tokens out": 10,
        "twenty tokens out": 20,
        "thirty tokens out": 30,
    }
    metrics = _calculate(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=_make_mock_tokenizer(tokenizer_counts),
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        num_gpus=1,
    )
    text = metrics.text_data
    assert text is not None
    # The trim left exactly one measured request (R3) over a 1 s window.
    assert text.completed == 1
    assert math.isclose(text.duration, 1.0, rel_tol=1e-9)
    # Per-GPU: all 3 requests' 120 tokens over the full 2.5 s span.
    assert text.total_tokens_per_second_per_gpu is not None
    assert math.isclose(
        text.total_tokens_per_second_per_gpu, 120.0 / 2.5, rel_tol=1e-9
    )


def test_per_gpu_input_counts_each_request_on_its_own_token_source() -> None:
    """Input tokens for the per-GPU figure are picked per request: the
    server's ``prompt_tokens`` where the request carries usage, the
    client's ``prompt_len`` where it does not.

    One request reporting usage must not zero out the other's client-side
    count.
    """
    with_usage = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=5,
        generated_text="four tok",
        itl=[0.1] * 3,
        request_submit_time=0.0,
        server_token_stats=ServerTokenStats(
            prompt_tokens=7, completion_tokens=4
        ),
    )
    without_usage = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="four tok",
        itl=[0.1] * 3,
        request_submit_time=0.0,
        server_token_stats=ServerTokenStats(completion_tokens=4),
    )
    metrics = _calculate(
        outputs=[with_usage, without_usage],
        dur_s=1.0,
        tokenizer=_make_mock_tokenizer({"four tok": 4}),
        num_gpus=1,
    )
    text = metrics.text_data
    assert text is not None
    assert text.total_tokens_per_second_per_gpu is not None
    # Input: 7 (server) + 10 (client). Output: 4 + 4. Span: 1.0 s.
    assert math.isclose(
        text.total_tokens_per_second_per_gpu, 7 + 10 + 4 + 4, rel_tol=1e-9
    )


def test_per_gpu_throughput_none_without_gpu_count() -> None:
    """An unknown GPU count leaves the per-GPU figure unreported rather
    than guessing a divisor."""
    metrics = _calculate_interactivity(_interactivity_outputs())
    assert metrics.num_gpus is None
    text = metrics.text_data
    assert text is not None
    assert text.total_tokens_per_second_per_gpu is None
    assert "total_tokens_per_second_per_gpu" not in metrics.to_result_dict()
    assert metrics.result_groups is not None
    assert metrics.result_groups.summary.total_tokens_per_second_per_gpu is None


def test_interactivity_none_for_single_token_outputs() -> None:
    """No request generated two tokens, so neither decode interactivity has
    samples. The e2e normalized interactivity (ttft + sum(itl) / 1 token)
    stays defined."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=0.5,
            ttft=0.4,
            prompt_len=3,
            generated_text="one",
        ),
        RequestFuncOutput(
            success=True,
            latency=0.7,
            ttft=0.6,
            prompt_len=4,
            generated_text="one",
        ),
    ]
    tokenizer = _make_mock_tokenizer({"one": 1})
    metrics = _calculate(
        outputs=outputs,
        dur_s=1.0,
        tokenizer=tokenizer,
        num_gpus=1,
    )
    text = metrics.text_data
    assert text is not None
    assert text.interactivity_tok_s is None
    assert text.full_response_interactivity_tok_s is None
    assert text.e2e_normalized_interactivity_tok_s is not None
    # Content-bounded e2e = ttft / 1 token: 0.4 s and 0.6 s -> mean 0.5.
    assert math.isclose(
        text.e2e_normalized_interactivity_tok_s.mean, 1.0 / 0.5, rel_tol=1e-6
    )
    # Per-GPU throughput does not depend on decode samples.
    assert text.total_tokens_per_second_per_gpu is not None
    assert math.isclose(
        text.total_tokens_per_second_per_gpu,
        (3 + 4 + 1 + 1) / 1.0,
        rel_tol=1e-9,
    )


def test_interactivity_none_when_no_requests_succeed() -> None:
    """A fully-failed run reports no interactivity and no per-GPU figure."""
    outputs = [
        RequestFuncOutput(success=False, latency=0.0, error="boom"),
    ]
    metrics = _calculate_interactivity(outputs, num_gpus=1)
    text = metrics.text_data
    assert text is not None
    assert text.interactivity_tok_s is None
    assert text.full_response_interactivity_tok_s is None
    assert text.e2e_normalized_interactivity_tok_s is None
    assert text.total_tokens_per_second_per_gpu is None


def test_interactivity_drops_zero_durations_rather_than_poisoning_stats() -> (
    None
):
    """A zero duration has no reciprocal; it is dropped from the series
    (as the published figures drop non-positive records) instead of
    collapsing the metric or dragging a percentile to zero."""
    outputs = [
        # latency == ttft with >1 output tokens and no content gaps:
        # TPOT 0, content ITL 0.
        RequestFuncOutput(
            success=True,
            latency=0.5,
            ttft=0.5,
            prompt_len=3,
            generated_text="ten tokens out",
        ),
        # TPOT 0.2 s, content ITL 0.1 s.
        RequestFuncOutput(
            success=True,
            latency=0.8,
            ttft=0.2,
            prompt_len=5,
            generated_text="four tok",
            itl=[0.1] * 3,
        ),
    ]
    text = _calculate_interactivity(outputs).text_data
    assert text is not None
    # Both series keep the one positive sample.
    assert text.interactivity_tok_s is not None
    assert text.interactivity_tok_s.mean == pytest.approx(1.0 / 0.1)
    assert text.full_response_interactivity_tok_s is not None
    assert text.full_response_interactivity_tok_s.mean == pytest.approx(
        1.0 / 0.2
    )
    # The e2e basis is positive for both requests.
    assert text.e2e_normalized_interactivity_tok_s is not None
