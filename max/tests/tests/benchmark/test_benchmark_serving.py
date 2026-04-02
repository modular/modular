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
import sys
from unittest.mock import patch

import numpy as np
import pytest
from max.benchmark.benchmark_serving import (
    _add_spec_decode_result,
    chat_session_driver,
    elide_data_uris_in_string,
    get_request,
    parse_args,
)
from max.benchmark.benchmark_shared.datasets import SampledRequest
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ChatSession,
)
from max.benchmark.benchmark_shared.metrics import (
    PercentileMetrics,
    SpecDecodeMetrics,
    StandardPercentileMetrics,
    ThroughputMetrics,
    calculate_spec_decode_stats,
    parse_spec_decode_metrics,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
)


def test_benchmark_serving_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test the benchmark serving help function."""
    # Mock sys.argv to simulate running with --help flag
    test_args = ["benchmark_serving.py", "--help"]
    with patch.object(sys, "argv", test_args):
        # The --help flag causes argparse to exit with code 0 during parse_args()
        # We need to catch this SystemExit exception
        with pytest.raises(SystemExit) as excinfo:
            # initialize_parser() calls parse_args() internally and will exit with --help
            args = parse_args()

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
    assert stats["num_drafts"] == 12
    assert stats["draft_tokens"] == 36
    assert stats["accepted_tokens"] == 24
    assert stats["acceptance_rate"] == pytest.approx((24 / 36) * 100)
    assert stats["acceptance_length"] == pytest.approx(1 + 24 / 12)
    assert stats["per_position_acceptance_rates"] == pytest.approx(
        [12 / 12, 8 / 12, 4 / 12]
    )


def test_add_spec_decode_result_uses_vllm_json_keys() -> None:
    """Spec decode stats are serialized under vLLM-compatible keys."""
    result: dict[str, object] = {}
    stats = {
        "num_drafts": 5,
        "draft_tokens": 18,
        "accepted_tokens": 9,
        "acceptance_rate": 50.0,
        "acceptance_length": 2.8,
        "per_position_acceptance_rates": [1.0, 0.6, 0.2],
    }

    _add_spec_decode_result(result, stats)

    assert result == {
        "spec_decode_acceptance_rate": 50.0,
        "spec_decode_acceptance_length": 2.8,
        "spec_decode_num_drafts": 5,
        "spec_decode_draft_tokens": 18,
        "spec_decode_accepted_tokens": 9,
        "spec_decode_per_position_acceptance_rates": [1.0, 0.6, 0.2],
    }
