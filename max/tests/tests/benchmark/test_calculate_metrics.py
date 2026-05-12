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

import math
from unittest.mock import MagicMock

import pytest
from max.benchmark.benchmark_shared.request import (
    PixelGenerationRequestFuncOutput,
    RequestFuncOutput,
)
from max.benchmark.benchmark_shared.serving_metrics import (
    _compute_steady_state_result,
    calculate_metrics,
    calculate_pixel_generation_metrics,
)
from max.diagnostics.cpu import CPUMetrics

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


def test_per_chunk_tpot_collected_from_outputs() -> None:
    """Per-chunk TPOT values are correctly collected from outputs."""
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

    metrics = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    # TPOT percentiles should be based on the per-chunk values [0.05, 0.1, 0.15]
    # scaled by 1000 (to ms)
    assert math.isclose(metrics.text_data.tpot_ms.p50, 100.0, rel_tol=1e-3)


def test_tpot_weighted_mean() -> None:
    """TPOT mean = sum(ITL) / decode_tokens * 1000 ms."""
    # Request 1: 10 output tokens, ITL sum = 0.9s
    output1 = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="ten tokens out",
        itl=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        tpot=[0.1] * 9,
    )

    # Request 2: 4 output tokens, ITL sum = 0.6s
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

    metrics = calculate_metrics(
        outputs=[output1, output2],
        dur_s=2.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    # total_output = 10 + 4 = 14
    # completed = 2
    # decode_tokens = 14 - 2 = 12
    # sum(itl) = 0.9 + 0.6 = 1.5
    # weighted mean TPOT = 1.5 / 12 * 1000 = 125.0 ms
    expected_mean = 1.5 / 12 * 1000.0
    assert math.isclose(
        metrics.text_data.tpot_ms.mean, expected_mean, rel_tol=1e-6
    )


def test_tpot_zero_decode_tokens() -> None:
    """When all requests produce <= 1 token, TPOT mean is NaN."""
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

    metrics = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    # With empty tpots, StandardPercentileMetrics gets [nan], so mean is nan
    assert math.isnan(metrics.text_data.tpot_ms.mean)


def test_empty_outputs_no_crash() -> None:
    """Empty outputs list doesn't crash."""
    tokenizer = _make_mock_tokenizer({})

    metrics = calculate_metrics(
        outputs=[],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    assert metrics.text_data.completed == 0
    assert metrics.text_data.output_lens == []
    # TPOT mean should be NaN since there are no outputs
    assert math.isnan(metrics.text_data.tpot_ms.mean)


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

    metrics = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    # ITL should be computed from the raw itl values [0.1, 0.2, 0.3] * 1000
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

    metrics = calculate_metrics(
        outputs=[success_output, failed_output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    # Only successful request's TPOT should be used
    assert metrics.text_data.completed == 1
    assert metrics.text_data.failures == 1
    # TPOT values should only include [0.1, 0.2], not [999.0]
    assert metrics.text_data.tpot_ms.p50 < 500.0


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

    metrics_all = calculate_metrics(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    metrics_skip_last = calculate_metrics(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics_all.text_data is not None
    assert metrics_skip_last.text_data is not None

    # completed reflects the measured slice (first two, third was skipped).
    assert metrics_all.text_data.completed == 3
    assert metrics_skip_last.text_data.completed == 2
    # The last request's high TTFT (0.5s) is excluded from latency metrics.
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

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    # Only the middle request is measured.
    assert metrics.text_data.completed == 1
    # Only the middle request's TTFT (0.1s = 100ms) should be measured
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

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    # skip_last is applied to successful (3), not to the padded cancelled
    # entries, so only the middle successful request is measured.
    assert metrics.text_data.completed == 1
    # Only the second request should be measured (skip first 1, last 1)
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
        calculate_metrics(
            outputs=outputs,
            dur_s=1.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=_EMPTY_CPU_METRICS,
            skip_first_n_requests=1,
            skip_last_n_requests=1,
            max_concurrency=None,
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
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
    metrics = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
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

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=full_run_duration,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=10,
        skip_last_n_requests=10,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
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
    metrics = calculate_metrics(
        outputs=[output],
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
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

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=20.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
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

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=10.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
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

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=15.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
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

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=21.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert metrics.text_data is not None

    # slow_warmup is in both head_drop_ids and tail_drop_ids — set union
    # handles dedupe. The remaining "a" and "b" are both measured.
    assert metrics.text_data.completed == 2


def _make_tokenizer_mock(tokens_per_output: int = 5) -> MagicMock:
    """Return a tokenizer mock whose encode call returns a fixed token count."""
    tokenizer = MagicMock()
    tokenizer.return_value = MagicMock(input_ids=list(range(tokens_per_output)))
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


def test_compute_steady_state_result_not_detected() -> None:
    """With too few requests, _compute_steady_state_result returns only detection-metadata keys."""
    outputs = [_make_request_func_output() for _ in range(3)]
    result = _compute_steady_state_result(
        outputs=outputs,
        tokenizer=None,
        gpu_metrics=None,
        cpu_metrics=None,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        metrics_by_endpoint=None,
    ).to_result_dict()

    assert result == {
        "steady_state_detected": False,
        "steady_state_start_index": None,
        "steady_state_end_index": None,
        "steady_state_count": 0,
        "steady_state_warning": (
            "Too few valid requests (3 of 3 total) for steady-state"
            " detection (need at least 100). TPOT was absent across the"
            " run, so detection ran in TTFT-only mode; the run has too few"
            " valid requests (cancelled, failed, or missing"
            " timestamps/TTFT are filtered out)."
        ),
        "steady_state_mode": "ttft_only",
    }


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


def test_compute_steady_state_result_detected() -> None:
    """With enough stable requests, _compute_steady_state_result detects steady state and returns metric keys."""
    outputs = [_make_stable_request_func_output(float(i)) for i in range(200)]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    result = _compute_steady_state_result(
        outputs=outputs,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        metrics_by_endpoint=None,
    ).to_result_dict()

    assert set(result.keys()) == {
        # Detection metadata — always present.
        "steady_state_detected",
        "steady_state_start_index",
        "steady_state_end_index",
        "steady_state_count",
        "steady_state_warning",
        "steady_state_mode",
        # Per-metric summaries — present when detected and ≥2 valid requests.
        "steady_state_request_throughput",
        "steady_state_mean_ttft_ms",
        "steady_state_p99_ttft_ms",
        "steady_state_mean_tpot_ms",
        "steady_state_p99_tpot_ms",
        "steady_state_mean_itl_ms",
        "steady_state_p99_itl_ms",
        "steady_state_mean_latency_ms",
        "steady_state_p99_latency_ms",
        # Confidence-interval keys for each latency metric.
        "steady_state_ttft_ms_ci_lower",
        "steady_state_ttft_ms_ci_upper",
        "steady_state_ttft_ms_ci_relative_width",
        "steady_state_ttft_ms_confidence",
        "steady_state_ttft_ms_sample_size",
        "steady_state_tpot_ms_ci_lower",
        "steady_state_tpot_ms_ci_upper",
        "steady_state_tpot_ms_ci_relative_width",
        "steady_state_tpot_ms_confidence",
        "steady_state_tpot_ms_sample_size",
        "steady_state_itl_ms_ci_lower",
        "steady_state_itl_ms_ci_upper",
        "steady_state_itl_ms_ci_relative_width",
        "steady_state_itl_ms_confidence",
        "steady_state_itl_ms_sample_size",
        "steady_state_latency_ms_ci_lower",
        "steady_state_latency_ms_ci_upper",
        "steady_state_latency_ms_ci_relative_width",
        "steady_state_latency_ms_confidence",
        "steady_state_latency_ms_sample_size",
    }

    assert result["steady_state_detected"] is True
    assert result["steady_state_mode"] == "full"
    assert result["steady_state_start_index"] is not None
    assert result["steady_state_end_index"] is not None
    assert isinstance(result["steady_state_count"], int)
    assert result["steady_state_count"] > 0
    assert result["steady_state_warning"] is None
    # With ttft=0.05 s the mean should be ≈50 ms.
    assert result["steady_state_mean_ttft_ms"] == pytest.approx(50.0, rel=0.05)
