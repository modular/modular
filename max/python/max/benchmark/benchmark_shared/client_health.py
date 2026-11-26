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

"""Measures the benchmark client's own event loop while it drives load.

A timer due every interval measures how late the loop runs a ready callback,
which is how a saturated client delays the bytes it timestamps.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import NamedTuple

from max.benchmark.benchmark_shared.metrics import (
    RatePercentileMetrics,
    StandardPercentileMetrics,
)

# 10 ms, the interval the lag thresholds below were sized against. A timer
# slower than that exposes each sample to a whole scheduling quantum of
# queueing: on one calibrated run a 100 ms timer read 159 ms of p90 lag where a
# 10 ms monitor read 7 ms. It costs one wakeup per interval, 100 per second.
LOOP_PROBE_INTERVAL_S = 0.01

# The CPU-share window. Long enough that one descheduled slice does not dominate
# a window, short enough that a 20 s run has windows to compare.
LOOP_PROBE_WINDOW_S = 1.0

# A client delay distorts a reported latency once it is a material share of
# that latency. Measured on a fake server with a known 200 ms TTFT and 10 ms
# chunk ITL, reported TTFT p50 gains about twice the loop lag, so 5% of TTFT
# p50 (+5% first appeared near 10 ms of lag, +23% at 22 ms) is where TTFT
# starts to carry the client's delay. Reported ITL p50 does not drift the same
# way: it stays within 2.5% up to 2.2-2.6 chunk intervals of lag, then
# collapses to near zero by 4.4 intervals, because the loop reads several
# events off one socket per pass. Two intervals is therefore the first lag at
# which a chunk can go missing, and ITL p90 has already doubled there.
#
# On real workloads both terms sit below the floor (2.5-4.6 ms at a 50-92 ms
# TTFT), so the floor decides. It is 5 ms because the earliest TTFT drift
# measured (~10 ms of lag) is what the warning has to precede, and 5 ms raised
# no false alarm on either calibration set. Its margin over the worst healthy
# loop lag p90 (4.54 ms on the main suite, 4.33 ms on the recalibration) is
# small on purpose: a false alarm costs a reader one line, while a missed one
# passes a distorted TTFT p50 off as the server's, and a rig whose server
# answers in microseconds spends far more wall time in the client than a real
# run does.
LOOP_LAG_FLOOR_MS = 5.0
LOOP_LAG_TTFT_SHARE = 0.05
LOOP_LAG_ITL_FACTOR = 2.0

# Reported as context, never decisive. Runs with a loop thread at this CPU
# share still ran their timers on time and reported latencies within a few
# percent of the truth, so a high figure alone is headroom information: it
# says the client is near its ceiling, not that its numbers are wrong. Used as
# a trigger it was the first calibration's whole false-alarm budget: 14 of 26
# fires came from threads at 94-100% with lag under 5 ms and accurate
# latencies.
LOOP_BUSY_CPU_BOUND_PCT = 85.0

# A p90 needs a second of samples to mean anything: with ten, one isolated
# 50 ms pause in nine punctual samples is the p90. The probe still reports its
# numbers below this; the decision says nothing.
LOOP_MIN_LAG_SAMPLES = round(1.0 / LOOP_PROBE_INTERVAL_S)
LOOP_MIN_BUSY_WINDOWS = 3

# A window that expects samples and received a quarter of them did not measure
# a slow loop, it measured a loop that was not running. Below this share the
# warning reports starvation instead: runs with lag p90 630 ms and 1509 ms and
# reported TTFT 5x and 15x the truth stayed silent under the sample minimum.
LOOP_STARVED_SAMPLE_SHARE = 0.25

TTFT_DELAY_CLAUSE = "reported TTFT p50 includes the client's own delay"
# A lagged client reads several events per pass, so the gaps it measures are
# shorter than the server's, not longer.
ITL_DELAY_CLAUSE = (
    "reported ITL no longer reflects the server's token pacing"
    " (bunched reads collapse it)"
)


class _LagLimit(NamedTuple):
    """One way the client's own delay reaches a reported latency."""

    limit_ms: float
    """The lag at which this metric starts to carry the client's delay."""

    label: str
    """What the limit is called in the warning."""

    reference_ms: float
    """The latency it is derived from, printed with the label."""

    metric: str
    """The reported metric it distorts."""

    clause: str
    """What the warning says about that metric once the lag reaches it."""

    note: str
    """How to name the limit inside a binding floor's parentheses."""

    states_ratio: bool = False
    """Whether the warning also prints the lag as a multiple of the
    reference."""

    needs_ttft: bool = False
    """Whether a zero reading of this metric only counts as the client's
    doing once the lag has reached the TTFT limit too."""


@dataclass(frozen=True)
class ClientLoopStats:
    """What a :class:`ClientLoopProbe` saw over the measured window.

    Lag is in milliseconds. The busy share is in percent of wall time, the unit
    every consumer reads; the probe records fractions and
    :func:`summarize_loop_samples` converts once.
    """

    lag_ms: StandardPercentileMetrics | None
    """How late each timer ran, or ``None`` without samples."""

    lag_max_ms: float | None
    """The worst single wakeup delay, or ``None`` without samples. Unlike
    ``lag_ms.p90`` this includes process pauses, so one stall sets it."""

    busy: RatePercentileMetrics | None
    """The loop thread's CPU share per window, in percent, or ``None``
    without a full window."""

    lag_samples: int
    """How many lag samples ``lag_ms`` summarizes."""

    busy_windows: int
    """How many windows ``busy`` summarizes."""

    window_s: float | None
    """How long the probe sampled for, or ``None`` before it started."""

    expected_lag_samples: int | None
    """How many samples a loop that ran its timer on time would have taken in
    ``window_s``, or ``None`` without a span to compare against."""


def summarize_loop_samples(
    lags_s: list[float],
    busy_fractions: list[float],
    *,
    window_s: float | None = None,
    interval_s: float = LOOP_PROBE_INTERVAL_S,
) -> ClientLoopStats:
    """Summarizes loop samples. Decides nothing.

    Args:
        lags_s: How late each timer ran, in seconds.
        busy_fractions: The loop thread's CPU share per window, each in
            ``[0, 1]`` of wall time. These are the only fraction-valued
            samples in this module.
        window_s: How long the samples span, or ``None`` when unknown.
        interval_s: The timer interval they were taken at.

    Returns:
        The percentiles of each distribution, the worst wakeup delay, the
        sample counts and the count a punctual timer would have taken. Whether
        that says anything about the run's reported latencies is
        :func:`client_loop_warning`'s question.
    """
    return ClientLoopStats(
        lag_ms=(
            StandardPercentileMetrics(lags_s, scale_factor=1000.0, unit="ms")
            if lags_s
            else None
        ),
        lag_max_ms=max(lags_s) * 1000.0 if lags_s else None,
        busy=RatePercentileMetrics(busy_fractions) if busy_fractions else None,
        lag_samples=len(lags_s),
        busy_windows=len(busy_fractions),
        window_s=window_s,
        expected_lag_samples=(
            round(window_s / interval_s) if window_s else None
        ),
    )


def _lag_limits(
    ttft_p50_ms: float | None, itl_p50_ms: float | None
) -> list[_LagLimit]:
    """Lists the limits of the reported latencies this run has samples for."""
    limits: list[_LagLimit] = []
    if ttft_p50_ms is not None and ttft_p50_ms > 0.0:
        limits.append(
            _LagLimit(
                limit_ms=LOOP_LAG_TTFT_SHARE * ttft_p50_ms,
                label=f"{LOOP_LAG_TTFT_SHARE:.0%} of TTFT p50",
                reference_ms=ttft_p50_ms,
                metric="TTFT p50",
                clause=TTFT_DELAY_CLAUSE,
                note=(
                    f"{LOOP_LAG_TTFT_SHARE:.0%} of TTFT p50 is"
                    f" {LOOP_LAG_TTFT_SHARE * ttft_p50_ms:.0f} ms"
                ),
            )
        )
    if itl_p50_ms is not None and itl_p50_ms >= 0.0:
        limits.append(
            _LagLimit(
                limit_ms=LOOP_LAG_ITL_FACTOR * itl_p50_ms,
                label=f"{LOOP_LAG_ITL_FACTOR:.0f}x ITL p50",
                reference_ms=itl_p50_ms,
                metric="ITL p50",
                clause=ITL_DELAY_CLAUSE,
                note=(
                    f"{LOOP_LAG_ITL_FACTOR:.0f}x the chunk ITL p50 is"
                    f" {LOOP_LAG_ITL_FACTOR * itl_p50_ms:.0f} ms"
                ),
                states_ratio=True,
                needs_ttft=itl_p50_ms == 0.0,
            )
        )
    return limits


def _crossed(
    limit: _LagLimit, limits: list[_LagLimit], lag_p90_ms: float
) -> bool:
    """Whether ``lag_p90_ms`` has reached what ``limit`` describes."""
    if limit.needs_ttft:
        # An exactly zero per-chunk p50 is half the chunk gaps measured as
        # zero. A client that bunches reads reports that, but so does a server
        # that emits several chunks in one write, and only the first is the
        # client's doing: wait until the lag has inflated TTFT as well.
        ttft = next(
            (other for other in limits if other.metric == "TTFT p50"), None
        )
        return ttft is not None and lag_p90_ms >= ttft.limit_ms
    return lag_p90_ms >= limit.limit_ms


def _lag_limit_basis(limit: _LagLimit, lag_p90_ms: float) -> str:
    """Names the limit a lag crossed, with the value it was derived from."""
    basis = f"{limit.label} ({limit.reference_ms:.0f} ms"
    if limit.states_ratio and limit.reference_ms > 0.0:
        basis += f", {lag_p90_ms / limit.reference_ms:.1f}x"
    return f"{basis})"


def _starved_warning(
    stats: ClientLoopStats, ttft_p50_ms: float | None
) -> str | None:
    """Explains a window the probe was too starved to sample.

    A saturated loop fires its own timer rarely, so a sample count far under
    what the window could hold is evidence of saturation, not of a healthy run.
    """
    expected = stats.expected_lag_samples
    if (
        expected is None
        or stats.window_s is None
        # Too short a window has too few samples for any count to be a deficit.
        or expected < LOOP_MIN_LAG_SAMPLES
        or stats.lag_samples >= LOOP_STARVED_SAMPLE_SHARE * expected
    ):
        return None
    message = (
        f"client loop took {stats.lag_samples} lag samples in a"
        f" {stats.window_s:.0f} s measured window (about {expected}"
        " expected)"
    )
    if stats.lag_max_ms is not None:
        # The one sample that does exist carries the state of the loop.
        message += f", worst wakeup {stats.lag_max_ms:.0f} ms late"
    message += "; the loop was rarely scheduled"
    if ttft_p50_ms is not None and ttft_p50_ms > 0.0:
        message += (
            f", so reported TTFT p50 ({ttft_p50_ms:.0f} ms) includes the"
            " client's own delay"
        )
    cause = _client_delay_cause(stats)
    return message if cause is None else f"{message} ({cause})"


def client_loop_warning(
    stats: ClientLoopStats,
    *,
    ttft_p50_ms: float | None,
    itl_p50_ms: float | None,
) -> str | None:
    """Explains, when the client's delay reached the latencies the run reports.

    Args:
        stats: The probe's summary.
        ttft_p50_ms: The run's reported TTFT p50, or ``None`` without samples.
        itl_p50_ms: The run's reported per-chunk ITL p50, or ``None`` for a
            run with no decode samples. Falls back to the TTFT term alone.

    Returns:
        One line naming what the client's delay did to the run: it crossed the
        limit of a latency the run reports, with each metric it reached and
        what that did to it, or, when the loop took too few samples to judge
        that, it starved the probe itself. Both say whether the loop was
        CPU-bound. ``None`` when the client stayed clear or the window is too
        short to judge.
    """
    # The lag rule first: it names a measured lag and every metric that lag
    # reached, which is what the starvation branch cannot. Starvation is the
    # fallback for a window whose sample count leaves the lag rule nothing.
    if stats.lag_ms is None or stats.lag_samples < LOOP_MIN_LAG_SAMPLES:
        return _starved_warning(stats, ttft_p50_ms)
    limits = _lag_limits(ttft_p50_ms, itl_p50_ms)
    # A zero ITL reading is evidence of a collapse, not a threshold of its own.
    binding = min(
        (limit for limit in limits if not limit.needs_ttft),
        key=lambda limit: limit.limit_ms,
        default=None,
    )
    if binding is None:
        return None

    lag_p90_ms = stats.lag_ms.p90
    threshold = max(LOOP_LAG_FLOOR_MS, binding.limit_ms)
    if binding.limit_ms >= LOOP_LAG_FLOOR_MS:
        basis = _lag_limit_basis(binding, lag_p90_ms)
    else:
        notes = "; ".join(limit.note for limit in limits)
        basis = f"the {LOOP_LAG_FLOOR_MS:.0f} ms floor ({notes})"
    if lag_p90_ms < threshold:
        return None

    # Read the reached metrics off the lag that was measured, not off the
    # threshold: on a run whose TTFT limit sits above the floor, a lag far past
    # both reaches TTFT even though the floor set the threshold.
    crossed = [
        limit.clause for limit in limits if _crossed(limit, limits, lag_p90_ms)
    ]
    message = (
        f"client loop lag p90 {lag_p90_ms:.0f} ms >= {basis}; "
        + "; ".join(crossed)
    )
    cause = _client_delay_cause(stats)
    return message if cause is None else f"{message} ({cause})"


def _client_delay_cause(stats: ClientLoopStats) -> str | None:
    """Names the mechanism behind the lag, when the busy windows can say."""
    if stats.busy is None or stats.busy_windows < LOOP_MIN_BUSY_WINDOWS:
        return None
    if stats.busy.p90 >= LOOP_BUSY_CPU_BOUND_PCT:
        return f"client CPU-bound, busy p90 {stats.busy.p90:.0f}%"
    return (
        f"loop blocked or process descheduled, busy p90 {stats.busy.p90:.0f}%"
    )


class ClientLoopProbe:
    """Samples how late the event loop runs a timer and how busy its thread is.

    Every ``interval_s`` the probe records how late its sleep returned, and
    every ``window_s`` the share of wall time the loop's thread spent on CPU.
    It costs one wakeup per interval.

    Args:
        interval_s: Time between timer samples.
        window_s: Length of each CPU-share window.
    """

    def __init__(
        self,
        interval_s: float = LOOP_PROBE_INTERVAL_S,
        window_s: float = LOOP_PROBE_WINDOW_S,
    ) -> None:
        self._interval_s = interval_s
        self._window_s = window_s
        self._lags_s: list[float] = []
        self._busy_fractions: list[float] = []
        self._task: asyncio.Task[None] | None = None
        self._started_s: float | None = None
        self._span_s: float | None = None

    def start(self) -> None:
        """Starts sampling on the running event loop."""
        self._started_s = time.perf_counter()
        self._task = asyncio.get_running_loop().create_task(self._run())

    def stop(self) -> None:
        """Cancels sampling without waiting; prefer :meth:`aclose`."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._started_s is not None and self._span_s is None:
            # The span, not the sample count, is what says how many samples
            # the window could have held.
            self._span_s = time.perf_counter() - self._started_s

    async def aclose(self) -> None:
        """Stops sampling and waits for the cancelled timer to finish."""
        task = self._task
        self.stop()
        if task is not None:
            # ``asyncio.wait`` reports the task's own cancellation as a
            # result, so a cancelled caller's cancellation still propagates.
            await asyncio.wait([task])

    def stats(self) -> ClientLoopStats:
        """Summarizes the samples taken so far."""
        span_s = self._span_s
        if span_s is None and self._started_s is not None:
            span_s = time.perf_counter() - self._started_s
        return summarize_loop_samples(
            self._lags_s,
            self._busy_fractions,
            window_s=span_s,
            interval_s=self._interval_s,
        )

    async def _run(self) -> None:
        window_start = time.perf_counter()
        cpu_start = time.thread_time()
        while True:
            due = time.perf_counter() + self._interval_s
            await asyncio.sleep(self._interval_s)
            now = time.perf_counter()
            self._lags_s.append(max(0.0, now - due))
            if now - window_start >= self._window_s:
                cpu = time.thread_time()
                self._busy_fractions.append(
                    min(1.0, (cpu - cpu_start) / (now - window_start))
                )
                window_start, cpu_start = now, cpu
