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
"""Benchmark serving dev unit tests"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from max.benchmark.benchmark_serving import (
    _compute_steady_state_result,
    _ConcurrentTurnsRequestDriver,
    _log_warmup_sampling_report,
    _pick_warmup_population,
    _WarmupSamplingReport,
    chat_session_driver,
    elide_data_uris_in_string,
    get_request,
    parse_args,
    prime_prefix_turns,
    systematic_probability_proportional_to_size,
)
from max.benchmark.benchmark_shared.datasets import SampledRequest
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ChatSession,
)
from max.benchmark.benchmark_shared.metrics import (
    PercentileMetrics,
    SpecDecodeMetrics,
    SpecDecodeStats,
    StandardPercentileMetrics,
    ThroughputMetrics,
    calculate_spec_decode_stats,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
)
from max.benchmark.benchmark_shared.server_metrics import (
    parse_spec_decode_metrics,
)


def test_benchmark_serving_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the benchmark serving help function."""
    # Mock sys.argv to simulate running with --help flag
    test_args = ["benchmark_serving.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            parse_args()

        # Verify it exited with code 0 (success)
        assert excinfo.value.code == 0

        # Capture and verify the help output
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()


# PercentileMetrics base class tests
def test_percentile_metrics_basic_creation() -> None:
    """Test basic creation of PercentileMetrics."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        median=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    assert metrics.mean == 10.0
    assert metrics.std == 2.0
    assert metrics.median == 9.5
    assert metrics.p90 == 12.0
    assert metrics.p95 == 14.0
    assert metrics.p99 == 18.0
    assert metrics.unit == "ms"


def test_percentile_metrics_creation_without_unit() -> None:
    """Test creating PercentileMetrics without unit."""
    metrics = PercentileMetrics(
        mean=10.0, std=2.0, median=9.5, p90=12.0, p95=14.0, p99=18.0
    )
    assert metrics.unit is None


def test_percentile_metrics_str_representation() -> None:
    """Test string representation of PercentileMetrics."""
    metrics = PercentileMetrics(
        mean=10.5,
        std=2.3,
        median=9.8,
        p90=12.7,
        p95=14.2,
        p99=18.9,
    )
    result = str(metrics)

    # Check that all metrics are present in formatted output
    assert "Mean:" in result
    assert "10.50" in result
    assert "Std:" in result
    assert "2.30" in result
    assert "Median:" in result
    assert "9.80" in result
    assert "P90:" in result
    assert "12.70" in result
    assert "P95:" in result
    assert "14.20" in result
    assert "P99:" in result
    assert "18.90" in result


def test_percentile_metrics_format_with_prefix() -> None:
    """Test format_with_prefix method."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        median=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    result = metrics.format_with_prefix("latency")

    # Check that prefix and unit are correctly included
    assert "Mean latency (ms):" in result
    assert "Std latency (ms):" in result
    assert "Median latency (ms):" in result
    assert "P90 latency (ms):" in result
    assert "P95 latency (ms):" in result
    assert "P99 latency (ms):" in result


def test_percentile_metrics_format_with_prefix_override_unit() -> None:
    """Test format_with_prefix with unit override."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        median=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    result = metrics.format_with_prefix("latency", unit="seconds")

    # Check that overridden unit is used
    assert "Mean latency (seconds):" in result
    assert "P99 latency (seconds):" in result


def test_percentile_metrics_format_with_prefix_no_unit() -> None:
    """Test format_with_prefix without unit."""
    metrics = PercentileMetrics(
        mean=10.0, std=2.0, median=9.5, p90=12.0, p95=14.0, p99=18.0
    )
    result = metrics.format_with_prefix("metric")

    # Check that no unit suffix is added
    assert "Mean metric:" in result
    assert "P99 metric:" in result
    assert " (ms):" not in result
    assert " (seconds):" not in result


# StandardPercentileMetrics tests
def test_standard_percentile_metrics_basic_functionality() -> None:
    """Test basic StandardPercentileMetrics functionality."""
    # Test data with known statistical properties
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    metrics = StandardPercentileMetrics(data)

    # Verify mean and basic statistics
    assert metrics.mean == pytest.approx(5.5, rel=1e-10)
    assert metrics.median == pytest.approx(5.5, rel=1e-10)

    # Verify percentiles are calculated correctly (90th, 95th, 99th)
    expected_p90 = np.percentile(data, 90)
    expected_p95 = np.percentile(data, 95)
    expected_p99 = np.percentile(data, 99)

    assert metrics.p90 == pytest.approx(expected_p90, rel=1e-10)
    assert metrics.p95 == pytest.approx(expected_p95, rel=1e-10)
    assert metrics.p99 == pytest.approx(expected_p99, rel=1e-10)


def test_standard_percentile_metrics_scale_factor() -> None:
    """Test scale factor functionality."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    scale_factor = 1000.0

    metrics = StandardPercentileMetrics(data, scale_factor=scale_factor)

    # All values should be scaled by the factor
    assert metrics.mean == pytest.approx(3.0 * scale_factor, rel=1e-10)
    assert metrics.median == pytest.approx(3.0 * scale_factor, rel=1e-10)

    # Percentiles should also be scaled
    expected_p90 = np.percentile(data, 90) * scale_factor
    assert metrics.p90 == pytest.approx(expected_p90, rel=1e-10)


def test_standard_percentile_metrics_with_unit() -> None:
    """Test StandardPercentileMetrics with unit."""
    data = [1.0, 2.0, 3.0]

    metrics = StandardPercentileMetrics(data, unit="ms")

    assert metrics.unit == "ms"


def test_standard_percentile_metrics_str_representation() -> None:
    """Test string representation uses 'metric' prefix."""
    data = [1.0, 2.0, 3.0]

    metrics = StandardPercentileMetrics(data)
    result = str(metrics)

    # Should use 'metric' prefix since it inherits __str__ that calls format_with_prefix
    assert "metric" in result.lower()


def test_standard_percentile_metrics_empty_data_assertion() -> None:
    """Test that empty data raises assertion error."""
    with pytest.raises(AssertionError, match="data must not be empty"):
        StandardPercentileMetrics([])


def test_standard_percentile_metrics_non_list_data_assertion() -> None:
    """Test that non-list data raises assertion error."""
    with pytest.raises(AssertionError, match="data must be a list"):
        # tuple instead of list
        StandardPercentileMetrics((1.0, 2.0, 3.0))  # type: ignore


def test_standard_percentile_metrics_non_float_data_assertion() -> None:
    """Test that non-float data raises assertion error."""
    with pytest.raises(AssertionError, match="data must contain only floats"):
        StandardPercentileMetrics([1, 2, 3])  # integers instead of floats


def test_standard_percentile_metrics_single_value() -> None:
    """Test with single value in data."""
    data = [5.0]

    metrics = StandardPercentileMetrics(data)

    # All statistics should equal the single value
    assert metrics.mean == 5.0
    assert metrics.std == 0.0
    assert metrics.median == 5.0
    assert metrics.p90 == 5.0
    assert metrics.p95 == 5.0
    assert metrics.p99 == 5.0


# ThroughputMetrics tests
def test_throughput_metrics_basic_functionality() -> None:
    """Test basic ThroughputMetrics functionality."""
    # Test data with known statistical properties
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    metrics = ThroughputMetrics(data)

    # Verify mean and basic statistics (same as standard)
    assert metrics.mean == pytest.approx(5.5, rel=1e-10)
    assert metrics.median == pytest.approx(5.5, rel=1e-10)


def test_throughput_metrics_reversed_percentiles() -> None:
    """Test that percentiles are reversed for throughput (lower percentiles for p90, p95, p99)."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    metrics = ThroughputMetrics(data)

    # For throughput, p90 should be 10th percentile (bottom 10%)
    # p95 should be 5th percentile (bottom 5%)
    # p99 should be 1st percentile (bottom 1%)
    expected_p90 = np.percentile(data, 10)  # Bottom 10%
    expected_p95 = np.percentile(data, 5)  # Bottom 5%
    expected_p99 = np.percentile(data, 1)  # Bottom 1%

    assert metrics.p90 == pytest.approx(expected_p90, rel=1e-10)
    assert metrics.p95 == pytest.approx(expected_p95, rel=1e-10)
    assert metrics.p99 == pytest.approx(expected_p99, rel=1e-10)


def test_throughput_metrics_vs_standard_percentiles() -> None:
    """Test that throughput percentiles are different from standard percentiles."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    throughput_metrics = ThroughputMetrics(data)
    standard_metrics = StandardPercentileMetrics(data)

    # Throughput percentiles should be lower than standard percentiles
    assert throughput_metrics.p90 < standard_metrics.p90
    assert throughput_metrics.p95 < standard_metrics.p95
    assert throughput_metrics.p99 < standard_metrics.p99


def test_throughput_metrics_scale_factor() -> None:
    """Test scale factor functionality."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    scale_factor = 1000.0

    metrics = ThroughputMetrics(data, scale_factor=scale_factor)

    # All values should be scaled by the factor
    assert metrics.mean == pytest.approx(3.0 * scale_factor, rel=1e-10)
    assert metrics.median == pytest.approx(3.0 * scale_factor, rel=1e-10)

    # Percentiles should also be scaled
    expected_p90 = np.percentile(data, 10) * scale_factor
    assert metrics.p90 == pytest.approx(expected_p90, rel=1e-10)


def test_throughput_metrics_with_unit() -> None:
    """Test ThroughputMetrics with unit."""
    data = [1.0, 2.0, 3.0]

    metrics = ThroughputMetrics(data, unit="tok/s")

    assert metrics.unit == "tok/s"


def test_throughput_metrics_str_representation() -> None:
    """Test string representation uses 'throughput' prefix."""
    data = [1.0, 2.0, 3.0]

    metrics = ThroughputMetrics(data)
    result = str(metrics)

    # Should use 'throughput' prefix
    assert "throughput" in result.lower()


def test_throughput_metrics_empty_data_assertion() -> None:
    """Test that empty data raises assertion error."""
    with pytest.raises(AssertionError, match="data must not be empty"):
        ThroughputMetrics([])


def test_throughput_metrics_non_list_data_assertion() -> None:
    """Test that non-list data raises assertion error."""
    with pytest.raises(AssertionError, match="data must be a list"):
        # tuple instead of list
        ThroughputMetrics((1.0, 2.0, 3.0))  # type: ignore


def test_throughput_metrics_non_float_data_assertion() -> None:
    """Test that non-float data raises assertion error."""
    with pytest.raises(AssertionError, match="data must contain only floats"):
        ThroughputMetrics([1, 2, 3])  # integers instead of floats


def test_throughput_metrics_single_value() -> None:
    """Test with single value in data."""
    data = [5.0]

    metrics = ThroughputMetrics(data)

    # All statistics should equal the single value
    assert metrics.mean == 5.0
    assert metrics.std == 0.0
    assert metrics.median == 5.0
    assert metrics.p90 == 5.0
    assert metrics.p95 == 5.0
    assert metrics.p99 == 5.0


# Integration tests
def test_both_metrics_with_same_data() -> None:
    """Test that both metric types work correctly with the same data."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    standard = StandardPercentileMetrics(data, scale_factor=1000.0, unit="ms")
    throughput = ThroughputMetrics(data, scale_factor=1.0, unit="tok/s")

    # Both should calculate mean and median the same way
    assert (
        standard.mean == throughput.mean * 1000.0
    )  # Due to scale factor difference
    assert standard.median == throughput.median * 1000.0

    # But percentiles should be different due to reversed logic
    assert standard.p90 > throughput.p90 * 1000.0
    assert standard.p95 > throughput.p95 * 1000.0
    assert standard.p99 > throughput.p99 * 1000.0


def test_edge_case_large_dataset() -> None:
    """Test with larger dataset to ensure robustness."""
    # Generate a larger dataset with known distribution
    np.random.seed(42)  # For reproducible tests
    data = np.random.normal(50.0, 10.0, 1000).tolist()

    standard = StandardPercentileMetrics(data)
    throughput = ThroughputMetrics(data)

    # Should handle large datasets without issues
    assert isinstance(standard.mean, float)
    assert isinstance(throughput.mean, float)
    assert standard.mean == pytest.approx(throughput.mean, rel=1e-10)

    # Percentiles should still follow expected relationships
    assert standard.p99 > standard.p95 > standard.p90
    assert (
        throughput.p90 > throughput.p95 > throughput.p99
    )  # Reversed for throughput


# Request interval generation tests
async def generate_test_intervals(
    request_rate: float,
    burstiness: float,
    num_samples: int = 100,
    seed: int = 42,
) -> list[float]:
    """Generate request intervals using the actual get_request function from benchmark_serving.py"""
    np.random.seed(seed)

    # Create mock SampledRequest objects for testing
    mock_requests = [
        SampledRequest(
            prompt_formatted=f"Test prompt {i}",
            prompt_len=10,
            output_len=20,
            encoded_images=[],
            ignore_eos=True,
        )
        for i in range(num_samples)
    ]

    # Timing data will be collected by get_request
    timing_data: dict[str, list[float]] = {}

    # Use the actual get_request function from benchmark_serving
    async for _ in get_request(
        mock_requests, request_rate, timing_data, burstiness
    ):
        pass  # Just consume the requests - we only care about timing data

    # Return the intervals collected by get_request
    return timing_data.get("intervals", [])


def calculate_interval_stats(
    intervals: list[float], request_rate: float
) -> dict[str, float]:
    """Calculate comprehensive statistics for intervals"""
    intervals_array = np.array(intervals)
    target_interval = 1.0 / request_rate

    return {
        "target_mean_interval": float(target_interval),
        "actual_mean_interval": float(np.mean(intervals_array)),
        "actual_request_rate": float(1.0 / np.mean(intervals_array)),
        "std_dev": float(np.std(intervals_array)),
        "min_interval": float(np.min(intervals_array)),
        "max_interval": float(np.max(intervals_array)),
        "median_interval": float(np.median(intervals_array)),
        "p10": float(np.percentile(intervals_array, 10)),
        "p25": float(np.percentile(intervals_array, 25)),
        "p75": float(np.percentile(intervals_array, 75)),
        "p90": float(np.percentile(intervals_array, 90)),
        "p95": float(np.percentile(intervals_array, 95)),
        "p99": float(np.percentile(intervals_array, 99)),
    }


# Using manual asyncio.run instead of pytest.mark.asyncio
def test_request_intervals_basic_functionality() -> None:
    """Test basic request interval generation functionality."""

    async def run_test():  # noqa: ANN202
        request_rate = 5.0
        burstiness = 1.0
        num_samples = 50

        intervals = await generate_test_intervals(
            request_rate, burstiness, num_samples
        )
        return intervals, request_rate, num_samples

    intervals, request_rate, num_samples = asyncio.run(run_test())

    # Check we got the right number of intervals (should be num_samples - 1)
    assert len(intervals) == num_samples - 1

    # Check all intervals are positive
    assert all(interval > 0 for interval in intervals)

    # Check request rate is approximately correct
    mean_interval = np.mean(intervals)
    actual_rate = 1.0 / mean_interval
    assert actual_rate == pytest.approx(
        request_rate, rel=0.3
    )  # 30% tolerance for small samples


# Using manual asyncio.run instead of pytest.mark.asyncio
def test_request_intervals_seed_reproducibility() -> None:
    """Test that same seed produces identical results."""

    async def run_test():  # noqa: ANN202
        request_rate = 10.0
        burstiness = 1.0
        num_samples = 30
        seed = 42

        # Generate intervals twice with same seed
        intervals_1 = await generate_test_intervals(
            request_rate, burstiness, num_samples, seed
        )
        intervals_2 = await generate_test_intervals(
            request_rate, burstiness, num_samples, seed
        )
        return intervals_1, intervals_2

    intervals_1, intervals_2 = asyncio.run(run_test())

    # Calculate statistics for both runs
    request_rate = 10.0  # Define request_rate for stats calculation
    stats_1 = calculate_interval_stats(intervals_1, request_rate)
    stats_2 = calculate_interval_stats(intervals_2, request_rate)

    # Check that statistics match to 2 decimal places
    tolerance = 1  # 1 decimal place precision tolerance

    assert (
        abs(stats_1["actual_mean_interval"] - stats_2["actual_mean_interval"])
        < tolerance
    )
    assert (
        abs(stats_1["actual_request_rate"] - stats_2["actual_request_rate"])
        < tolerance
    )
    assert abs(stats_1["std_dev"] - stats_2["std_dev"]) < tolerance
    assert (
        abs(stats_1["median_interval"] - stats_2["median_interval"]) < tolerance
    )


# Using manual asyncio.run instead of pytest.mark.asyncio
def test_request_intervals_different_seeds() -> None:
    """Test that different seeds produce different results."""

    async def run_test():  # noqa: ANN202
        request_rate = 8.0
        burstiness = 1.0
        num_samples = 40

        intervals_seed1 = await generate_test_intervals(
            request_rate, burstiness, num_samples, seed=42
        )
        intervals_seed2 = await generate_test_intervals(
            request_rate, burstiness, num_samples, seed=123
        )
        return intervals_seed1, intervals_seed2, request_rate

    intervals_seed1, intervals_seed2, request_rate = asyncio.run(run_test())

    # Check that intervals are actually different
    assert not np.array_equal(intervals_seed1, intervals_seed2)

    # Check that both maintain approximately correct request rate
    rate_1 = 1.0 / np.mean(intervals_seed1)
    rate_2 = 1.0 / np.mean(intervals_seed2)

    assert rate_1 == pytest.approx(request_rate, rel=0.3)
    assert rate_2 == pytest.approx(request_rate, rel=0.3)


# Using manual asyncio.run instead of pytest.mark.asyncio
def test_request_intervals_infinite_rate() -> None:
    """Test that infinite request rate works correctly."""

    async def run_test():  # noqa: ANN202
        request_rate = float("inf")
        burstiness = 1.0
        num_samples = 20

        intervals = await generate_test_intervals(
            request_rate, burstiness, num_samples
        )
        return intervals, num_samples

    intervals, num_samples = asyncio.run(run_test())

    # With infinite rate, all intervals should be very small (near zero)
    assert len(intervals) == num_samples - 1
    assert all(
        interval < 0.01 for interval in intervals
    )  # Very small intervals


# Using manual asyncio.run instead of pytest.mark.asyncio
def test_request_intervals_burstiness_variations() -> None:
    """Test different burstiness values produce expected behavior."""

    async def run_test():  # noqa: ANN202
        request_rate = 5.0
        num_samples = 50

        # Test different burstiness values
        burstiness_values = [0.5, 1.0, 2.0]
        results = {}

        for burstiness in burstiness_values:
            intervals = await generate_test_intervals(
                request_rate, burstiness, num_samples
            )
            stats = calculate_interval_stats(intervals, request_rate)
            results[burstiness] = stats

        return results, request_rate

    results, request_rate = asyncio.run(run_test())

    # All should maintain approximately the same request rate
    for burstiness, stats in results.items():
        actual_rate = stats["actual_request_rate"]
        assert actual_rate == pytest.approx(request_rate, rel=0.4), (
            f"Burstiness {burstiness} failed rate check"
        )

    # Different burstiness should produce different standard deviations
    std_05 = results[0.5]["std_dev"]
    std_10 = results[1.0]["std_dev"]
    std_20 = results[2.0]["std_dev"]

    # Lower burstiness should generally have higher variance
    assert std_05 != std_10  # Should be different
    assert std_10 != std_20  # Should be different


def test_elide_data_uris_in_string() -> None:
    """Test that elide_data_uris_in_string correctly elides base64 data URIs."""

    # fmt: off

    # Basic case
    sample = "'image': 'data:image/jpeg;base64,/9j/4AAQSASDEEAE'"
    expected = "'image': 'data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)...'"
    assert elide_data_uris_in_string(sample) == expected

    # Two data URIs in a single string
    sample = "data:image/jpeg;base64,/9j/4AAQSASDEEAE + data:image/jpeg;base64,/9j/4AAQSASDEEAE"
    expected = "data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)... + data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)..."
    assert elide_data_uris_in_string(sample) == expected

    # Still elides even if it results in longer string
    sample = "data:image/jpeg;base64,ABC"
    expected = "data:image/jpeg;base64,...(hash: b5d4045c, 3 bytes)..."
    assert elide_data_uris_in_string(sample) == expected

    # Does not elide if invalid characters in data
    sample = "data:image/jpeg;base64,ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧"
    expected = "data:image/jpeg;base64,ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧"
    assert elide_data_uris_in_string(sample) == expected

    # Does not elide if data uri type is empty
    sample = "data:;base64,ABC"
    expected = "data:;base64,ABC"
    assert elide_data_uris_in_string(sample) == expected

    # `data:` is present in string but not part of data uri
    sample = "Here is some data: 'data:image/jpeg;base64,AAAAAAAAASTUFF=='"
    expected = "Here is some data: 'data:image/jpeg;base64,...(hash: 6c6e1584, 16 bytes)...'"
    assert elide_data_uris_in_string(sample) == expected

    # `;base64` is present in string but not part of data uri
    sample = ";base64"
    expected = ";base64"
    assert elide_data_uris_in_string(sample) == expected

    # String is empty
    sample = ""
    expected = ""
    assert elide_data_uris_in_string(sample) == expected

    # fmt: on


def test_chat_session_driver_forwards_sampling_params() -> None:
    """Test that chat_session_driver forwards temperature, top_p, top_k."""

    captured_inputs: list[RequestFuncInput] = []

    class CapturingDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            assert isinstance(request_func_input, RequestFuncInput)
            captured_inputs.append(request_func_input)
            return RequestFuncOutput(
                success=True,
                latency=0.1,
                ttft=0.05,
                prompt_len=request_func_input.prompt_len,
                generated_text="Hello",
            )

    async def run_test() -> None:
        chat_session = ChatSession(
            id=0,
            messages=[
                ChatMessage(source="user", content="Hi", num_tokens=5),
                ChatMessage(source="assistant", content="Hello", num_tokens=5),
            ],
        )
        request_counter = RequestCounter(max_requests=10, total_sent_requests=0)

        await chat_session_driver(
            model_id="test-model",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=CapturingDriver(),
            request_counter=request_counter,
            chat_session=chat_session,
            max_chat_len=4096,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

    asyncio.run(run_test())

    assert len(captured_inputs) == 1
    assert captured_inputs[0].temperature == 0.7
    assert captured_inputs[0].top_p == 0.9
    assert captured_inputs[0].top_k == 50


def test_chat_session_driver_run_prefix_prepends_first_turn() -> None:
    """First user message gets the run prefix when run_prefix is set."""

    captured: list[RequestFuncInput] = []

    class CapturingDriver(RequestDriver):
        async def request(
            self, request_func_input: BaseRequestFuncInput
        ) -> RequestFuncOutput:
            assert isinstance(request_func_input, RequestFuncInput)
            captured.append(request_func_input)
            return RequestFuncOutput(
                success=True,
                latency=0.1,
                ttft=0.05,
                prompt_len=request_func_input.prompt_len,
                generated_text="Hello",
            )

    async def run_test() -> None:
        chat_session = ChatSession(
            id=0,
            messages=[
                ChatMessage(source="user", content="Hi", num_tokens=5),
                ChatMessage(source="assistant", content="Hello", num_tokens=5),
                ChatMessage(source="user", content="Again", num_tokens=5),
                ChatMessage(source="assistant", content="Hi", num_tokens=5),
            ],
        )
        request_counter = RequestCounter(max_requests=10, total_sent_requests=0)

        await chat_session_driver(
            model_id="test-model",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=CapturingDriver(),
            request_counter=request_counter,
            chat_session=chat_session,
            max_chat_len=4096,
            temperature=None,
            top_p=None,
            top_k=None,
            run_prefix="RUN-UUID: ",
            run_prefix_len=4,
        )

    asyncio.run(run_test())

    assert len(captured) == 2
    assert isinstance(captured[0].prompt, list)
    first_user_text = captured[0].prompt[0]["content"][0]["text"]
    assert first_user_text.endswith("Hi")
    assert first_user_text != "Hi"
    assert isinstance(captured[1].prompt, list)
    second_user_text = captured[1].prompt[2]["content"][0]["text"]
    assert second_user_text == "Again"


def _make_4turn_session(
    prefix_turns: int = 0,
    delay_ms: float = 1000.0,
) -> ChatSession:
    """Create a 4-turn chat session for testing prefix_turns behavior."""
    return ChatSession(
        id=0,
        messages=[
            ChatMessage(source="user", content="Turn 1", num_tokens=5),
            ChatMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            ChatMessage(source="user", content="Turn 2", num_tokens=5),
            ChatMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            ChatMessage(source="user", content="Turn 3", num_tokens=5),
            ChatMessage(
                source="assistant",
                content="",
                num_tokens=5,
                delay_until_next_message=delay_ms,
            ),
            ChatMessage(source="user", content="Turn 4", num_tokens=5),
            ChatMessage(
                source="assistant",
                content="",
                num_tokens=5,
            ),
        ],
        prefix_turns=prefix_turns,
    )


class _CapturingDriver(RequestDriver):
    """Request driver that records all requests and returns success."""

    def __init__(self) -> None:
        self.calls: list[RequestFuncInput] = []

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        assert isinstance(request_func_input, RequestFuncInput)
        self.calls.append(request_func_input)
        return RequestFuncOutput(
            success=True,
            latency=0.1,
            ttft=0.05,
            prompt_len=request_func_input.prompt_len,
            generated_text="ok",
        )


def test_prefix_turns_excluded_from_results() -> None:
    """With prefix_turns=2, a 4-turn session should return only 2 results."""

    async def run_test() -> list[RequestFuncOutput]:
        session = _make_4turn_session(prefix_turns=2)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        return await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 2


def test_prefix_turns_dont_count_against_max_requests() -> None:
    """Prefix turns should not consume max_requests budget."""

    async def run_test() -> tuple[list[RequestFuncOutput], int, int]:
        session = _make_4turn_session(prefix_turns=2)
        counter = RequestCounter(max_requests=2)
        driver = _CapturingDriver()
        outputs = await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        return outputs, len(driver.calls), counter.total_sent_requests

    outputs, total_calls, counter_value = asyncio.run(run_test())
    # Prefix turns are built locally, so only measured turns hit the server.
    assert total_calls == 2
    assert counter_value == 2
    assert len(outputs) == 2


def test_prefix_turns_no_server_or_delays() -> None:
    """Prefix turns are built locally: no server calls, no delays."""
    sleep_calls: list[float] = []

    async def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def run_test() -> int:
        session = _make_4turn_session(prefix_turns=2, delay_ms=5000.0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await chat_session_driver(
                model_id="test",
                api_url="http://localhost:8000/v1/chat/completions",
                request_driver=driver,
                request_counter=counter,
                chat_session=session,
                max_chat_len=4096,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        return len(driver.calls)

    total_calls = asyncio.run(run_test())
    # Prefix turns don't hit the server.
    assert total_calls == 2
    # Only turn 3 has a delay (turn 4 has no delay_until_next_message).
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(5.0)


def test_prefix_turns_zero_is_noop() -> None:
    """prefix_turns=0 should behave identically to the old code."""

    async def run_test() -> list[RequestFuncOutput]:
        session = _make_4turn_session(prefix_turns=0)
        counter = RequestCounter(max_requests=100)
        driver = _CapturingDriver()
        return await chat_session_driver(
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=counter,
            chat_session=session,
            max_chat_len=4096,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    outputs = asyncio.run(run_test())
    assert len(outputs) == 4


def _make_session_with_id(session_id: int, prefix_turns: int) -> ChatSession:
    """Helper: 4-turn session with a specific id."""
    session = _make_4turn_session(prefix_turns=prefix_turns)
    return dataclasses.replace(session, id=session_id)


def test_prime_prefix_turns_only_primes_sessions_with_prefix() -> None:
    """Sessions with prefix_turns=0 should not generate any priming requests."""
    sessions = [
        _make_session_with_id(0, prefix_turns=0),
        _make_session_with_id(1, prefix_turns=2),
        _make_session_with_id(2, prefix_turns=0),
    ]
    driver = _CapturingDriver()

    async def run() -> None:
        await prime_prefix_turns(
            sessions=sessions,
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    asyncio.run(run())
    assert len(driver.calls) == 1
    assert driver.calls[0].session_id == "1"
    # Priming requests request a single token and must not stop early on EOS
    # (otherwise the full prefix may not be prefilled).
    assert driver.calls[0].max_tokens == 1
    assert driver.calls[0].ignore_eos is True


def test_prime_prefix_turns_respects_max_sessions() -> None:
    """max_sessions caps priming to the initial concurrent population."""
    sessions = [_make_session_with_id(idx, prefix_turns=2) for idx in range(5)]
    driver = _CapturingDriver()

    async def run() -> None:
        await prime_prefix_turns(
            sessions=sessions,
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            temperature=None,
            top_p=None,
            top_k=None,
            max_sessions=2,
        )

    asyncio.run(run())
    # Only the first two sessions should be primed.
    assert len(driver.calls) == 2
    primed_ids = {call.session_id for call in driver.calls}
    assert primed_ids == {"0", "1"}


def test_prime_prefix_turns_noop_without_prefix_sessions() -> None:
    """When no session has prefix_turns > 0, no requests are issued."""
    sessions = [_make_session_with_id(idx, prefix_turns=0) for idx in range(3)]
    driver = _CapturingDriver()

    async def run() -> None:
        await prime_prefix_turns(
            sessions=sessions,
            request_driver=driver,
            model_id="test",
            api_url="http://localhost:8000/v1/chat/completions",
            max_chat_len=4096,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    asyncio.run(run())
    assert driver.calls == []


def test_parse_spec_decode_metrics_matches_vllm_format() -> None:
    """Spec decode counters are parsed from vLLM Prometheus text."""
    metrics_text = """# HELP vllm:spec_decode_num_drafts Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts 12
# HELP vllm:spec_decode_num_draft_tokens Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens counter
vllm:spec_decode_num_draft_tokens 40
# HELP vllm:spec_decode_num_accepted_tokens Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens 21
# HELP vllm:spec_decode_num_accepted_tokens_per_pos Accepted tokens per position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter
vllm:spec_decode_num_accepted_tokens_per_pos{position="0"} 12
vllm:spec_decode_num_accepted_tokens_per_pos{position="1"} 7
vllm:spec_decode_num_accepted_tokens_per_pos{position="2"} 2
"""

    parsed = parse_spec_decode_metrics(metrics_text)

    assert parsed is not None
    assert parsed.num_drafts == 12
    assert parsed.num_draft_tokens == 40
    assert parsed.num_accepted_tokens == 21
    assert parsed.accepted_per_pos == {0: 12, 1: 7, 2: 2}


def test_parse_spec_decode_metrics_returns_none_when_absent() -> None:
    """Metrics parsing returns None when no spec decode counters exist."""
    parsed = parse_spec_decode_metrics(
        "# HELP requests Total requests\n# TYPE requests counter\nrequests 10\n"
    )

    assert parsed is None


def test_calculate_spec_decode_stats_matches_vllm_math() -> None:
    """Acceptance math uses benchmark-window deltas like vLLM bench serve."""
    before = SpecDecodeMetrics(
        num_drafts=100,
        num_draft_tokens=320,
        num_accepted_tokens=150,
        accepted_per_pos={0: 100, 1: 40, 2: 10},
    )
    after = SpecDecodeMetrics(
        num_drafts=112,
        num_draft_tokens=356,
        num_accepted_tokens=174,
        accepted_per_pos={0: 112, 1: 48, 2: 14},
    )

    stats = calculate_spec_decode_stats(before, after)

    assert stats is not None
    assert stats.num_drafts == 12
    assert stats.draft_tokens == 36
    assert stats.accepted_tokens == 24
    assert stats.acceptance_rate == pytest.approx((24 / 36) * 100)
    assert stats.acceptance_length == pytest.approx(1 + 24 / 12)
    assert stats.per_position_acceptance_rates == pytest.approx(
        [12 / 12, 8 / 12, 4 / 12]
    )


def test_spec_decode_stats_to_result_dict_uses_vllm_json_keys() -> None:
    """Spec decode stats are serialized under vLLM-compatible keys."""
    stats = SpecDecodeStats(
        num_drafts=5,
        draft_tokens=18,
        accepted_tokens=9,
        acceptance_rate=50.0,
        acceptance_length=2.8,
        per_position_acceptance_rates=[1.0, 0.6, 0.2],
    )

    assert stats.to_result_dict() == {
        "spec_decode_acceptance_rate": 50.0,
        "spec_decode_acceptance_length": 2.8,
        "spec_decode_num_drafts": 5,
        "spec_decode_draft_tokens": 18,
        "spec_decode_accepted_tokens": 9,
        "spec_decode_per_position_acceptance_rates": [1.0, 0.6, 0.2],
    }


def test_parse_spec_decode_metrics_handles_maxserve_histogram() -> None:
    """MAX Serve's per-position acceptance histogram is parsed into running sums/counts."""
    metrics_text = """# HELP maxserve_spec_decode_acceptance_rate_per_position Per-position acceptance.
# TYPE maxserve_spec_decode_acceptance_rate_per_position histogram
maxserve_spec_decode_acceptance_rate_per_position_sum{position="0"} 8400.0
maxserve_spec_decode_acceptance_rate_per_position_count{position="0"} 100
maxserve_spec_decode_acceptance_rate_per_position_sum{position="1"} 5000.0
maxserve_spec_decode_acceptance_rate_per_position_count{position="1"} 100
"""

    parsed = parse_spec_decode_metrics(metrics_text)

    assert parsed is not None
    assert parsed.num_drafts == 0
    assert parsed.num_draft_tokens == 0
    assert parsed.per_pos_rate_sum == {0: 8400.0, 1: 5000.0}
    assert parsed.per_pos_rate_count == {0: 100, 1: 100}


def test_calculate_spec_decode_stats_from_maxserve_histogram_only() -> None:
    """Without aggregate counters, per-position rates still surface from histogram deltas."""
    before = SpecDecodeMetrics(
        per_pos_rate_sum={0: 8000.0, 1: 4000.0},
        per_pos_rate_count={0: 100, 1: 100},
    )
    after = SpecDecodeMetrics(
        per_pos_rate_sum={0: 16800.0, 1: 9000.0},
        per_pos_rate_count={0: 200, 1: 200},
    )

    stats = calculate_spec_decode_stats(before, after)

    assert stats is not None
    # Window per-position acceptance: (8800/100)% / 100 = 0.88; (5000/100)% / 100 = 0.50
    assert stats.per_position_acceptance_rates == pytest.approx([0.88, 0.50])
    assert stats.num_drafts is None
    assert stats.draft_tokens is None
    assert stats.accepted_tokens is None
    assert stats.acceptance_rate is None
    assert stats.acceptance_length is None


def test_spec_decode_stats_to_result_dict_omits_missing_aggregates() -> None:
    """JSON result only includes fields the backend actually exposed."""
    stats = SpecDecodeStats(per_position_acceptance_rates=[0.88, 0.50])

    assert stats.to_result_dict() == {
        "spec_decode_per_position_acceptance_rates": [0.88, 0.50],
    }


def test_concurrent_turns_driver_expired_deadline_cancels_without_calling_base() -> (
    None
):
    """A turn that acquires the semaphore after the deadline is cancelled, not forwarded."""
    semaphore = asyncio.Semaphore(10)
    mock_driver = AsyncMock()
    mock_driver.tokenizer = None
    driver = _ConcurrentTurnsRequestDriver(
        mock_driver,
        semaphore,
        benchmark_should_end_time=time.perf_counter_ns() - 1,
    )

    output_cls = MagicMock()
    output_cls.return_value = MagicMock()
    inp = MagicMock()
    inp.get_output_type.return_value = output_cls

    asyncio.run(driver.request(inp))

    assert output_cls.call_args.kwargs["cancelled"] is True
    mock_driver.request.assert_not_called()


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


# ---------------------------------------------------------------------------
# _pick_warmup_population
# ---------------------------------------------------------------------------


def _fake_session(session_id: int, num_turns: int) -> ChatSession:
    msgs: list[ChatMessage] = []
    for _ in range(num_turns):
        msgs.append(ChatMessage(source="user", content="u", num_tokens=1))
        msgs.append(ChatMessage(source="assistant", content="a", num_tokens=1))
    return ChatSession(id=session_id, messages=msgs)


def _fixed_pool(turn_counts: list[int]) -> list[ChatSession]:
    return [_fake_session(i, t) for i, t in enumerate(turn_counts)]


def test_pick_warmup_off_returns_unchanged() -> None:
    pool = _fixed_pool([3, 5, 7, 11])
    sessions, report = _pick_warmup_population(
        pool,
        warmup_count=2,
        warmup_to_steady_state=False,
        warmup_oversample_factor=8,
        main_pool_target=4,
        rng=np.random.default_rng(0),
    )
    assert report is None
    assert sessions == pool
    assert all(s.prefix_turns == 0 for s in sessions)


def test_pick_warmup_zero_count_returns_unchanged() -> None:
    pool = _fixed_pool([3, 5, 7])
    sessions, report = _pick_warmup_population(
        pool,
        warmup_count=0,
        warmup_to_steady_state=True,
        warmup_oversample_factor=8,
        main_pool_target=3,
        rng=np.random.default_rng(0),
    )
    assert report is None
    assert sessions == pool


def test_pick_warmup_factor_one_uniform_emits_report() -> None:
    # factor=1 has no length-bias headroom; falls through to uniform pick.
    # Report still emitted so the user can see the bias.
    pool = _fixed_pool([3, 5, 7, 11, 13])
    sessions, report = _pick_warmup_population(
        pool,
        warmup_count=2,
        warmup_to_steady_state=True,
        warmup_oversample_factor=1,
        main_pool_target=3,
        rng=np.random.default_rng(0),
    )
    assert report is not None
    assert report.factor == 1
    # Warmup picks at the head, prefix_turns assigned (0 for T=1).
    assert len(sessions) == len(pool)
    assert all(s.prefix_turns >= 0 for s in sessions[:2])
    # Main sessions (last 3) untouched: prefix_turns=0.
    assert all(s.prefix_turns == 0 for s in sessions[2:])


def test_pick_warmup_factor_zero_uniform_emits_report() -> None:
    # factor=0 behaves the same as factor=1 — uniform pick.
    pool = _fixed_pool([3, 5, 7, 11, 13])
    sessions, report = _pick_warmup_population(
        pool,
        warmup_count=2,
        warmup_to_steady_state=True,
        warmup_oversample_factor=0,
        main_pool_target=5,
        rng=np.random.default_rng(0),
    )
    assert report is not None
    assert report.factor == 0
    assert len(sessions) == len(pool)


def test_pick_warmup_length_biased_basic() -> None:
    rng = np.random.default_rng(123)
    # Pool of 144 sessions: 16 main sessions + 8*16 = 128 candidates.
    main = 16
    factor = 8
    M = 16
    turn_pool = list(rng.integers(1, 50, size=main + factor * M))
    pool = _fixed_pool([int(t) for t in turn_pool])
    sessions, report = _pick_warmup_population(
        pool,
        warmup_count=M,
        warmup_to_steady_state=True,
        warmup_oversample_factor=factor,
        main_pool_target=main,
        rng=rng,
    )
    assert report is not None
    assert report.factor == factor
    assert report.warmup_pool == factor * M
    assert report.main_pool == main
    assert report.warmup_count == M
    # All warmup picks at head; main sessions at tail with prefix_turns=0.
    assert len(sessions) == M + main
    warmup_picks = sessions[:M]
    main_sessions = sessions[M:]
    assert all(s.prefix_turns >= 0 for s in warmup_picks)
    assert any(s.prefix_turns > 0 for s in warmup_picks)
    assert all(s.prefix_turns == 0 for s in main_sessions)


def test_pick_warmup_length_biased_matches_size_biased_target() -> None:
    """Systematic PPS gives E[realized_mean] = target_mean exactly when no
    items cap. Average across many draws should be within ~1% of target."""
    main = 32
    M = 32
    factor = 8
    realized_means: list[float] = []
    target_mean: float | None = None
    for seed in range(50):
        rng = np.random.default_rng(seed)
        turn_pool = list(rng.integers(1, 60, size=main + factor * M))
        pool = _fixed_pool([int(t) for t in turn_pool])
        _, report = _pick_warmup_population(
            pool,
            warmup_count=M,
            warmup_to_steady_state=True,
            warmup_oversample_factor=factor,
            main_pool_target=main,
            rng=rng,
        )
        assert report is not None
        realized_means.append(report.realized_mean)
        target_mean = report.target_mean
    assert target_mean is not None
    avg_realized = float(np.mean(realized_means))
    # 50 draws of K=32 averages out to ~1600 picks of T; the analytical bias
    # is 0 (no caps) and per-draw stddev averages out fast.
    assert abs(avg_realized - target_mean) / target_mean < 0.02


def test_pick_warmup_realized_mean_stdev_brackets_observed_spread() -> None:
    """The reported per-draw stdev should upper-bound the actual spread of
    realized_mean across seeds (with-replacement bound; systematic PPS
    variance is at most this)."""
    main = 32
    M = 32
    factor = 8
    realized_means: list[float] = []
    stdev: float | None = None
    for seed in range(60):
        rng = np.random.default_rng(seed)
        turn_pool = list(rng.integers(1, 60, size=main + factor * M))
        pool = _fixed_pool([int(t) for t in turn_pool])
        _, report = _pick_warmup_population(
            pool,
            warmup_count=M,
            warmup_to_steady_state=True,
            warmup_oversample_factor=factor,
            main_pool_target=main,
            rng=rng,
        )
        assert report is not None
        realized_means.append(report.realized_mean)
        stdev = report.realized_mean_stdev
    assert stdev is not None
    assert stdev > 0
    observed_std = float(np.std(realized_means, ddof=1))
    # Reported stdev is a with-replacement upper bound; observed should be
    # at most ~1.3x stdev even with sampling noise on the std estimate.
    assert observed_std <= 1.3 * stdev


# ---------------------------------------------------------------------------
# _log_warmup_sampling_report
# ---------------------------------------------------------------------------


def _make_report(
    target_mean: float = 20.0,
    realized_mean: float = 20.0,
    realized_mean_stdev: float = 1.0,
    cap_count: int = 0,
) -> _WarmupSamplingReport:
    return _WarmupSamplingReport(
        warmup_pool=128,
        main_pool=100,
        warmup_count=16,
        factor=8,
        target_mean=target_mean,
        realized_mean=realized_mean,
        realized_mean_stdev=realized_mean_stdev,
        cap_count=cap_count,
    )


def test_log_warmup_sampling_report_no_caps_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _make_report(cap_count=0)
    with caplog.at_level("WARNING"):
        _log_warmup_sampling_report(report)
    assert not any(
        "Could not warmup to steady state" in r.message for r in caplog.records
    )


def test_log_warmup_sampling_report_caps_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    report = _make_report(cap_count=3)
    with caplog.at_level("WARNING"):
        _log_warmup_sampling_report(report)
    assert any(
        "Could not warmup to steady state" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# systematic_probability_proportional_to_size
# ---------------------------------------------------------------------------


def test_systematic_pps_no_duplicates() -> None:
    rng = np.random.default_rng(0)
    weights = rng.integers(1, 50, size=100).astype(np.float64)
    for _ in range(500):
        idx = systematic_probability_proportional_to_size(weights, 20, rng)
        assert len(idx) == 20
        assert len(set(idx.tolist())) == 20


def test_systematic_pps_inclusion_probability_matches_weight() -> None:
    """Empirical P(item i picked) ≈ K * w_i / W when no item caps."""
    rng = np.random.default_rng(1)
    weights = np.array([1, 2, 3, 5, 8, 13, 21], dtype=np.float64)
    K = 3
    # K * max(w) / W = 3 * 21 / 53 = 1.19 > 1 → cap. Lower K to avoid cap
    # for the matching-probability check.
    K = 2
    cap_thresh = weights.sum() / K
    assert (weights < cap_thresh).all()
    n_trials = 20000
    counts = np.zeros(len(weights), dtype=np.int64)
    for _ in range(n_trials):
        idx = systematic_probability_proportional_to_size(weights, K, rng)
        counts[idx] += 1
    expected = K * weights / weights.sum()
    actual = counts / n_trials
    assert np.allclose(actual, expected, atol=0.02)


def test_systematic_pps_iterated_cap_pre_includes() -> None:
    """An item with weight >= W/K is always included."""
    rng = np.random.default_rng(2)
    weights = np.array([1, 1, 1, 1, 100], dtype=np.float64)
    K = 2
    # weights[4] = 100 >> W/K = 104/2 = 52.
    for _ in range(200):
        idx = systematic_probability_proportional_to_size(weights, K, rng)
        assert 4 in idx.tolist()


def test_systematic_pps_mean_matches_size_biased() -> None:
    """E[mean of picks] = sum(T^2) / sum(T) when no item caps."""
    rng = np.random.default_rng(3)
    T = rng.integers(1, 50, size=200).astype(np.float64)
    K = 8
    cap_thresh = T.sum() / K
    assert (T < cap_thresh).all()
    sb_mean = float((T * T).sum() / T.sum())
    realized = []
    for _ in range(500):
        idx = systematic_probability_proportional_to_size(T, K, rng)
        realized.append(float(T[idx].mean()))
    avg = float(np.mean(realized))
    assert abs(avg - sb_mean) / sb_mean < 0.01
