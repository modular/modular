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

"""Steady-state auto-detection for benchmark runs.

Uses rolling MAD/median (median absolute deviation / median) on per-request
TTFT and TPOT to find the steady-state region:

- TTFT MAD/median detects the ramp-up boundary.
- TPOT MAD/median detects the ramp-down boundary.

Runs that never stabilize produce a descriptive warning instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .request import RequestFuncOutput

DEFAULT_WINDOW_SIZE = 50
DEFAULT_TTFT_THRESHOLD = 0.5
DEFAULT_TPOT_THRESHOLD = 0.3
DEFAULT_SUSTAINED_COUNT = DEFAULT_WINDOW_SIZE // 2


@dataclass
class SteadyStateWindow:
    """Result of steady-state auto-detection on a benchmark run.

    Attributes:
        detected: Whether a steady-state window was found.
        start_index: Min original index in the steady-state window.
            None if not detected.
        end_index: Max original index + 1 (exclusive) in the window.
            None if not detected. For multi-turn benchmarks where requests
            interleave across sessions, use ``steady_state_indices`` for
            the exact set of requests in the window.
        steady_state_indices: Original output indices of all requests in
            the steady-state window. Empty if not detected. Use this
            instead of start_index/end_index when requests may not be
            contiguous in dispatch order (e.g., multi-turn benchmarks).
        total_requests: Number of valid requests considered for detection
            (successful, non-cancelled, with timestamps and TPOT data).
        steady_state_count: Number of requests in the detected window.
        warning: Human-readable explanation when detection fails.
        window_size: Rolling window size used for detection.
        ttft_threshold: Threshold for TTFT stabilization.
        tpot_threshold: Threshold for TPOT stabilization.
    """

    detected: bool
    start_index: int | None
    end_index: int | None
    steady_state_indices: list[int]
    total_requests: int
    steady_state_count: int
    warning: str | None
    window_size: int
    ttft_threshold: float
    tpot_threshold: float


def _rolling_mad_over_median(values: list[float], window: int) -> list[float]:
    """Compute rolling MAD/median over a sliding window.

    Returns one value per position starting at index (window - 1).
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n < window:
        return []

    # Create a (n - window + 1, window) view without copying data.
    shape = (n - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    medians = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - medians[:, np.newaxis]), axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(medians == 0, np.inf, mad / medians)

    return result.tolist()


def _find_first_stable_run(
    series: list[float], threshold: float, sustained_count: int
) -> int | None:
    """Find the first index where the series stays below threshold for sustained_count points."""
    run_length = 0
    for idx, val in enumerate(series):
        if val <= threshold:
            run_length += 1
            if run_length >= sustained_count:
                return idx - sustained_count + 1
        else:
            run_length = 0
    return None


def _find_last_stable_run(
    series: list[float], threshold: float, sustained_count: int
) -> int | None:
    """Find the last index where the series stays below threshold for sustained_count points (scanning from end)."""
    run_length = 0
    for idx in range(len(series) - 1, -1, -1):
        if series[idx] <= threshold:
            run_length += 1
            if run_length >= sustained_count:
                return idx + sustained_count - 1
        else:
            run_length = 0
    return None


def detect_steady_state(
    outputs: Sequence[RequestFuncOutput],
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttft_threshold: float = DEFAULT_TTFT_THRESHOLD,
    tpot_threshold: float = DEFAULT_TPOT_THRESHOLD,
    sustained_count: int = DEFAULT_SUSTAINED_COUNT,
) -> SteadyStateWindow:
    """Detect the steady-state region within a benchmark run.

    Filters to valid requests (successful, non-cancelled, with timestamps,
    positive TTFT, and non-empty TPOT), sorts by submit time, then uses
    rolling MAD/median to find where metrics stabilize.

    Args:
        outputs: Request outputs in dispatch order.
        window_size: Number of requests in the rolling window.
        ttft_threshold: Threshold for TTFT stabilization.
        tpot_threshold: Threshold for TPOT stabilization.
        sustained_count: Consecutive points below threshold required
            to confirm stabilization.
    """
    indexed: list[tuple[int, RequestFuncOutput]] = [
        (i, out)
        for i, out in enumerate(outputs)
        if (
            out.success
            and not out.cancelled
            and out.request_submit_time is not None
            and out.ttft > 0
            and out.tpot
        )
    ]

    total = len(indexed)
    result = SteadyStateWindow(
        detected=False,
        start_index=None,
        end_index=None,
        steady_state_indices=[],
        total_requests=total,
        steady_state_count=0,
        warning=None,
        window_size=window_size,
        ttft_threshold=ttft_threshold,
        tpot_threshold=tpot_threshold,
    )

    if total < window_size * 2:
        result.warning = (
            f"Too few valid requests ({total}) for steady-state detection"
            f" (need at least {window_size * 2})."
        )
        return result

    indexed.sort(key=lambda x: x[1].request_submit_time or 0.0)

    ttfts = [out.ttft for _, out in indexed]
    tpots = [float(np.mean(out.tpot)) for _, out in indexed]

    ttft_mads = _rolling_mad_over_median(ttfts, window_size)
    tpot_mads = _rolling_mad_over_median(tpots, window_size)

    if not ttft_mads or not tpot_mads:
        result.warning = "Could not compute rolling statistics."
        return result

    ramp_up_idx = _find_first_stable_run(
        ttft_mads, ttft_threshold, sustained_count
    )
    if ramp_up_idx is None:
        result.warning = (
            f"TTFT never stabilized below MAD/median threshold"
            f" {ttft_threshold:.2f}. The system may have been"
            f" overloaded for the entire run."
        )
        return result

    steady_start = ramp_up_idx

    ramp_down_idx = _find_last_stable_run(
        tpot_mads, tpot_threshold, sustained_count
    )
    if ramp_down_idx is None:
        result.warning = (
            f"TPOT never stabilized below MAD/median threshold"
            f" {tpot_threshold:.2f}. The system may have been"
            f" overloaded for the entire run."
        )
        return result

    steady_end = ramp_down_idx + window_size

    if steady_start >= steady_end:
        result.warning = (
            "Ramp-up and ramp-down regions overlap; no steady-state"
            " window found."
        )
        return result

    steady_count = steady_end - steady_start
    if steady_count < window_size:
        result.warning = (
            f"Steady-state window too small ({steady_count} requests,"
            f" minimum {window_size})."
        )
        return result

    ss_indices = [indexed[i][0] for i in range(steady_start, steady_end)]
    result.detected = True
    result.steady_state_indices = ss_indices
    result.start_index = min(ss_indices)
    result.end_index = max(ss_indices) + 1
    result.steady_state_count = steady_count
    return result
