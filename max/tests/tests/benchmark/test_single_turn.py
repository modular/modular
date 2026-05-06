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
"""Tests for benchmark_shared.single_turn."""

from __future__ import annotations

import numpy as np
import pytest
from max.benchmark.benchmark_shared.datasets import SampledRequest
from max.benchmark.benchmark_shared.single_turn import get_request


async def generate_test_intervals(
    request_rate: float,
    burstiness: float,
    num_samples: int = 100,
    seed: int = 42,
) -> list[float]:
    """Generate request intervals using the actual get_request function."""
    np.random.seed(seed)

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

    timing_data: dict[str, list[float]] = {}

    async for _ in get_request(
        mock_requests, request_rate, timing_data, burstiness
    ):
        pass

    return timing_data.get("intervals", [])


def calculate_interval_stats(
    intervals: list[float], request_rate: float
) -> dict[str, float]:
    """Calculate comprehensive statistics for intervals."""
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


@pytest.mark.asyncio
async def test_request_intervals_basic_functionality() -> None:
    """Test basic request interval generation functionality."""
    request_rate = 5.0
    burstiness = 1.0
    num_samples = 50

    intervals = await generate_test_intervals(
        request_rate, burstiness, num_samples
    )

    assert len(intervals) == num_samples - 1
    assert all(interval > 0 for interval in intervals)

    mean_interval = np.mean(intervals)
    actual_rate = 1.0 / mean_interval
    assert actual_rate == pytest.approx(
        request_rate, rel=0.3
    )  # 30% tolerance for small samples


@pytest.mark.asyncio
async def test_request_intervals_seed_reproducibility() -> None:
    """Test that same seed produces identical results."""
    request_rate = 10.0
    burstiness = 1.0
    num_samples = 30
    seed = 42

    intervals_1 = await generate_test_intervals(
        request_rate, burstiness, num_samples, seed
    )
    intervals_2 = await generate_test_intervals(
        request_rate, burstiness, num_samples, seed
    )

    stats_1 = calculate_interval_stats(intervals_1, request_rate)
    stats_2 = calculate_interval_stats(intervals_2, request_rate)

    tolerance = 1

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


@pytest.mark.asyncio
async def test_request_intervals_different_seeds() -> None:
    """Test that different seeds produce different results."""
    request_rate = 8.0
    burstiness = 1.0
    num_samples = 40

    intervals_seed1 = await generate_test_intervals(
        request_rate, burstiness, num_samples, seed=42
    )
    intervals_seed2 = await generate_test_intervals(
        request_rate, burstiness, num_samples, seed=123
    )

    assert not np.array_equal(intervals_seed1, intervals_seed2)

    rate_1 = 1.0 / np.mean(intervals_seed1)
    rate_2 = 1.0 / np.mean(intervals_seed2)

    assert rate_1 == pytest.approx(request_rate, rel=0.3)
    assert rate_2 == pytest.approx(request_rate, rel=0.3)


@pytest.mark.asyncio
async def test_request_intervals_infinite_rate() -> None:
    """Test that infinite request rate works correctly."""
    request_rate = float("inf")
    burstiness = 1.0
    num_samples = 20

    intervals = await generate_test_intervals(
        request_rate, burstiness, num_samples
    )

    assert len(intervals) == num_samples - 1
    assert all(interval < 0.01 for interval in intervals)


@pytest.mark.asyncio
async def test_request_intervals_burstiness_variations() -> None:
    """Test different burstiness values produce expected behavior."""
    request_rate = 5.0
    num_samples = 50

    burstiness_values = [0.5, 1.0, 2.0]
    results = {}

    for burstiness in burstiness_values:
        intervals = await generate_test_intervals(
            request_rate, burstiness, num_samples
        )
        stats = calculate_interval_stats(intervals, request_rate)
        results[burstiness] = stats
        assert stats["actual_request_rate"] == pytest.approx(
            request_rate, rel=0.4
        ), f"Burstiness {burstiness} failed rate check"

    std_05 = results[0.5]["std_dev"]
    std_10 = results[1.0]["std_dev"]
    std_20 = results[2.0]["std_dev"]

    assert std_05 != std_10
    assert std_10 != std_20
