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

Identifies the steady-state region within a benchmark run by computing
rolling coefficient of variation (CV) on per-request metrics:

- **TTFT CV** detects the ramp-up boundary. During warmup, TTFT is inflated
  and variable due to queue buildup and cold caches. Once the system reaches
  steady state, TTFT stabilizes and the rolling CV drops below the threshold.

- **TPOT CV** detects the ramp-down boundary. During cooldown, batch sizes
  shrink and TPOT becomes erratic. Scanning from the end of the run, we find
  where TPOT was last stable.

The window between these two boundaries is the steady-state region. Runs that
never stabilize produce a descriptive warning instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .request import RequestFuncOutput

DEFAULT_WINDOW_SIZE = 50
DEFAULT_TTFT_CV_THRESHOLD = 0.5
DEFAULT_TPOT_CV_THRESHOLD = 0.3
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
        window_size: Rolling window size used for CV computation.
        ttft_cv_threshold: CV threshold used for TTFT stabilization.
        tpot_cv_threshold: CV threshold used for TPOT stabilization.
    """

    detected: bool
    start_index: int | None
    end_index: int | None
    steady_state_indices: list[int]
    total_requests: int
    steady_state_count: int
    warning: str | None
    window_size: int
    ttft_cv_threshold: float
    tpot_cv_threshold: float


def _rolling_cv(values: list[float], window: int) -> list[float]:
    """Compute rolling coefficient of variation over a sliding window.

    Returns one CV value per position starting at index (window - 1).
    Uses vectorized cumulative sums for O(n) performance.
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    if n < window:
        return []

    cumsum = np.cumsum(arr)
    rolling_sum = np.empty(n - window + 1)
    rolling_sum[0] = cumsum[window - 1]
    rolling_sum[1:] = cumsum[window:] - cumsum[: n - window]
    rolling_mean = rolling_sum / window

    cumsum_sq = np.cumsum(arr * arr)
    rolling_sum_sq = np.empty(n - window + 1)
    rolling_sum_sq[0] = cumsum_sq[window - 1]
    rolling_sum_sq[1:] = cumsum_sq[window:] - cumsum_sq[: n - window]
    rolling_var = rolling_sum_sq / window - rolling_mean**2
    np.maximum(rolling_var, 0.0, out=rolling_var)
    rolling_std = np.sqrt(rolling_var)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(
            rolling_mean == 0, np.inf, rolling_std / np.abs(rolling_mean)
        )

    return cv.tolist()


def _find_first_stable_run(
    cv_series: list[float], threshold: float, sustained_count: int
) -> int | None:
    """Find the first index where CV stays below threshold for sustained_count points."""
    run_length = 0
    for idx, cv in enumerate(cv_series):
        if cv <= threshold:
            run_length += 1
            if run_length >= sustained_count:
                return idx - sustained_count + 1
        else:
            run_length = 0
    return None


def _find_last_stable_run(
    cv_series: list[float], threshold: float, sustained_count: int
) -> int | None:
    """Find the last index where CV stays below threshold for sustained_count points (scanning from end)."""
    run_length = 0
    for idx in range(len(cv_series) - 1, -1, -1):
        if cv_series[idx] <= threshold:
            run_length += 1
            if run_length >= sustained_count:
                return idx + sustained_count - 1
        else:
            run_length = 0
    return None


def detect_steady_state(
    outputs: Sequence[RequestFuncOutput],
    window_size: int = DEFAULT_WINDOW_SIZE,
    ttft_cv_threshold: float = DEFAULT_TTFT_CV_THRESHOLD,
    tpot_cv_threshold: float = DEFAULT_TPOT_CV_THRESHOLD,
    sustained_count: int = DEFAULT_SUSTAINED_COUNT,
) -> SteadyStateWindow:
    """Detect the steady-state region within a benchmark run.

    Filters to valid requests (successful, non-cancelled, with timestamps,
    positive TTFT, and non-empty TPOT), sorts by submit time, then uses
    rolling CV to find where metrics stabilize.

    Args:
        outputs: Request outputs in dispatch order.
        window_size: Number of requests in the rolling window.
        ttft_cv_threshold: CV threshold for TTFT stabilization.
        tpot_cv_threshold: CV threshold for TPOT stabilization.
        sustained_count: Consecutive CV points below threshold required
            to confirm stabilization.
    """
    # (original_index, output) pairs for valid requests
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
        ttft_cv_threshold=ttft_cv_threshold,
        tpot_cv_threshold=tpot_cv_threshold,
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

    ttft_cvs = _rolling_cv(ttfts, window_size)
    tpot_cvs = _rolling_cv(tpots, window_size)

    if not ttft_cvs or not tpot_cvs:
        result.warning = "Could not compute rolling statistics."
        return result

    ramp_up_idx = _find_first_stable_run(
        ttft_cvs, ttft_cv_threshold, sustained_count
    )
    if ramp_up_idx is None:
        result.warning = (
            f"TTFT never stabilized below CV threshold"
            f" {ttft_cv_threshold:.2f}. The system may have been"
            f" overloaded for the entire run."
        )
        return result

    steady_start = ramp_up_idx

    ramp_down_idx = _find_last_stable_run(
        tpot_cvs, tpot_cv_threshold, sustained_count
    )
    if ramp_down_idx is None:
        result.warning = (
            f"TPOT never stabilized below CV threshold"
            f" {tpot_cv_threshold:.2f}. The system may have been"
            f" overloaded for the entire run."
        )
        return result

    # CV[i] covers requests [i .. i+window_size-1]
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
