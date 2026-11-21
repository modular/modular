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

"""Tests for the benchmark client's event-loop probe and its warning."""

from __future__ import annotations

import asyncio
import time

import pytest
from max.benchmark.benchmark_shared.client_health import (
    LOOP_LAG_FLOOR_MS,
    LOOP_MIN_LAG_SAMPLES,
    LOOP_PROBE_INTERVAL_S,
    ClientLoopProbe,
    ClientLoopStats,
    client_loop_warning,
    summarize_loop_samples,
)


def _stats(
    *,
    lag_p90_ms: float | None = None,
    busy_p90_pct: float | None = None,
    samples: int = 200,
    window_s: float | None = None,
) -> ClientLoopStats:
    """Builds the stats one row of the calibration table reported.

    Every sample holds the reported p90, so the percentile is exact. A window
    of ``None`` is a probe that was not asked how long it ran, which leaves the
    sample count alone to decide whether the loop was starved.
    """
    lags_s = [] if lag_p90_ms is None else [lag_p90_ms / 1000.0] * samples
    busy = [] if busy_p90_pct is None else [busy_p90_pct / 100.0] * 10
    return summarize_loop_samples(lags_s, busy, window_s=window_s)


@pytest.mark.parametrize(
    ("lag_p90_ms", "busy_p90_pct", "ttft_p50_ms", "itl_p50_ms", "warns"),
    [
        # The optimized client under load: the loop thread is nearly saturated
        # but its timers still run on time, and nothing it reports is distorted.
        pytest.param(0.0, 94.0, 50.7, 9.88, False, id="busy-but-punctual"),
        pytest.param(4.0, 100.0, 51.4, 9.93, False, id="busy-punctual-4ms"),
        # Low load: quiet whatever the metrics.
        pytest.param(0.0, 12.0, 49.8, 10.03, False, id="low-load"),
        # Calibrated rows: TTFT p50 moved first (+23% at 21.8 ms of lag).
        pytest.param(21.8, 100.0, 245.3, 10.25, True, id="ttft-moved"),
        # Main-suite arm B at c=512: ITL fine, TTFT 57% high.
        pytest.param(15.6, 100.0, 81.3, 9.8, True, id="ttft-only"),
        # Past two chunk intervals reported ITL p50 collapses, not drifts.
        pytest.param(50.8, 100.0, 250.3, 0.03, True, id="itl-collapsed"),
        pytest.param(117.6, 100.0, 297.3, 0.03, True, id="itl-collapsed-fat"),
        # A long TTFT makes 5% of it a huge absolute lag, so the chunk ITL
        # sets the limit: a chunk can go missing at two intervals, 20 ms.
        pytest.param(25.0, 60.0, 2000.0, 10.0, True, id="long-ttft"),
        pytest.param(12.0, 60.0, 2000.0, 10.0, False, id="long-ttft-under-2x"),
        # ... and 3 ms is nowhere near it either.
        pytest.param(3.0, 60.0, 2000.0, 10.0, False, id="long-ttft-quiet"),
        # No decode samples: the TTFT term stands alone.
        pytest.param(12.0, 60.0, 200.0, None, True, id="no-itl-samples"),
        pytest.param(3.0, 60.0, 200.0, None, False, id="no-itl-quiet"),
        # An exactly zero ITL p50 is the collapse only once the lag has also
        # reached the TTFT limit; on its own it is not a threshold.
        pytest.param(
            LOOP_LAG_FLOOR_MS + 1.0, 100.0, 250.0, 0.0, False, id="itl-zero"
        ),
        pytest.param(
            LOOP_LAG_FLOOR_MS - 1.0,
            100.0,
            250.0,
            0.0,
            False,
            id="itl-zero-quiet",
        ),
        pytest.param(20.0, 100.0, 250.0, 0.0, True, id="itl-zero-with-ttft"),
    ],
)
def test_the_warning_follows_the_calibrated_criterion(
    lag_p90_ms: float,
    busy_p90_pct: float,
    ttft_p50_ms: float,
    itl_p50_ms: float | None,
    warns: bool,
) -> None:
    stats = _stats(lag_p90_ms=lag_p90_ms, busy_p90_pct=busy_p90_pct)
    warning = client_loop_warning(
        stats, ttft_p50_ms=ttft_p50_ms, itl_p50_ms=itl_p50_ms
    )
    assert (warning is not None) is warns


def test_a_ttft_bound_warning_names_the_limit_and_the_metric() -> None:
    # A 30 ms chunk interval leaves the TTFT term binding (5% of 257 ms) and
    # the lag below the ITL limit (2x 30 ms), so only TTFT is named.
    stats = _stats(lag_p90_ms=41.0, busy_p90_pct=100.0)
    warning = client_loop_warning(stats, ttft_p50_ms=257.0, itl_p50_ms=30.0)
    assert warning == (
        "client loop lag p90 41 ms >= 5% of TTFT p50 (257 ms);"
        " reported TTFT p50 includes the client's own delay"
        " (client CPU-bound, busy p90 100%)"
    )


def test_a_collapsed_itl_names_both_metrics_the_lag_reached() -> None:
    # The floor sets the threshold here, but the lag is far past the TTFT limit
    # as well: naming only ITL would imply TTFT was fine.
    stats = _stats(lag_p90_ms=118.0, busy_p90_pct=100.0)
    warning = client_loop_warning(stats, ttft_p50_ms=355.9, itl_p50_ms=0.04)
    assert warning is not None
    assert warning.startswith(
        f"client loop lag p90 118 ms >= the {LOOP_LAG_FLOOR_MS:.0f} ms floor"
        " (5% of TTFT p50 is 18 ms; 2x the chunk ITL p50 is 0 ms);"
    )
    assert "reported TTFT p50 includes the client's own delay" in warning
    assert (
        "reported ITL no longer reflects the server's token pacing"
        " (bunched reads collapse it)" in warning
    )


def test_an_itl_bound_warning_names_the_chunk_interval_and_the_ratio() -> None:
    stats = _stats(lag_p90_ms=25.0, busy_p90_pct=60.0)
    warning = client_loop_warning(stats, ttft_p50_ms=2000.0, itl_p50_ms=10.0)
    assert warning == (
        "client loop lag p90 25 ms >= 2x ITL p50 (10 ms, 2.5x);"
        " reported ITL no longer reflects the server's token pacing"
        " (bunched reads collapse it)"
        " (loop blocked or process descheduled, busy p90 60%)"
    )


def test_a_floor_bound_warning_names_the_floor() -> None:
    # 5% of a 50 ms TTFT is 2.5 ms, below the floor, so the floor fires and
    # the ITL term is still clear of the crossed lag.
    stats = _stats(lag_p90_ms=LOOP_LAG_FLOOR_MS + 2.0, busy_p90_pct=98.0)
    warning = client_loop_warning(stats, ttft_p50_ms=50.0, itl_p50_ms=10.0)
    assert warning is not None
    assert warning.startswith(
        f"client loop lag p90 {LOOP_LAG_FLOOR_MS + 2:.0f} ms >="
        f" the {LOOP_LAG_FLOOR_MS:.0f} ms floor"
        " (5% of TTFT p50 is 2 ms; 2x the chunk ITL p50 is 20 ms);"
    )
    assert "reported TTFT p50 includes" in warning


def test_the_floor_can_cross_both_limits() -> None:
    stats = _stats(lag_p90_ms=LOOP_LAG_FLOOR_MS + 1.0, busy_p90_pct=40.0)
    warning = client_loop_warning(stats, ttft_p50_ms=50.0, itl_p50_ms=2.0)
    assert warning is not None
    assert "reported TTFT p50 includes the client's own delay" in warning
    assert (
        "reported ITL no longer reflects the server's token pacing" in warning
    )


def test_no_warning_when_the_lag_samples_are_too_few() -> None:
    # Two stalls on a run this short are the p90 of two samples, not a trend.
    stats = summarize_loop_samples([0.5, 0.5], [0.99])
    assert stats.lag_samples == 2
    assert stats.lag_ms is not None and stats.lag_ms.p90 == pytest.approx(500.0)
    assert (
        client_loop_warning(stats, ttft_p50_ms=200.0, itl_p50_ms=10.0) is None
    )


def test_no_warning_without_a_reported_latency() -> None:
    stats = _stats(lag_p90_ms=120.0)
    assert client_loop_warning(stats, ttft_p50_ms=None, itl_p50_ms=None) is None


def test_no_warning_without_a_busy_reading() -> None:
    # Lag alone still decides; the cause is simply left out of the message.
    stats = _stats(lag_p90_ms=41.0)
    warning = client_loop_warning(stats, ttft_p50_ms=257.0, itl_p50_ms=10.0)
    assert warning is not None
    assert warning.endswith("(bunched reads collapse it)")


def test_a_lone_stall_does_not_warn() -> None:
    # One garbage-collection-sized pause in an otherwise punctual second: it
    # sets the max, not the p90, and the pause alone cannot reach the floor.
    stats = summarize_loop_samples(
        [0.001] * (LOOP_MIN_LAG_SAMPLES - 1) + [0.07], [0.10] * 8
    )
    assert stats.lag_samples == LOOP_MIN_LAG_SAMPLES
    assert stats.lag_max_ms == pytest.approx(70.0)
    assert stats.lag_ms is not None
    assert stats.lag_ms.p90 < LOOP_LAG_FLOOR_MS
    assert client_loop_warning(stats, ttft_p50_ms=50.0, itl_p50_ms=10.0) is None


def test_one_pause_cannot_trigger_at_the_minimum_sample_count() -> None:
    # Nine punctual samples and one pause were enough to breach the floor when
    # ten samples were accepted (the p90 of ten is the tenth); at a full
    # second of samples the same pause is one sample in a hundred.
    assert LOOP_MIN_LAG_SAMPLES == round(1.0 / LOOP_PROBE_INTERVAL_S)
    stats = summarize_loop_samples(
        [0.001] * (LOOP_MIN_LAG_SAMPLES - 1) + [0.05], [0.99] * 10
    )
    assert stats.lag_samples == LOOP_MIN_LAG_SAMPLES
    assert stats.lag_max_ms == pytest.approx(50.0)
    assert stats.lag_ms is not None
    assert stats.lag_ms.p90 == pytest.approx(1.0)
    assert client_loop_warning(stats, ttft_p50_ms=50.0, itl_p50_ms=10.0) is None


def test_a_healthy_loop_reports_percentiles_without_a_warning() -> None:
    stats = summarize_loop_samples([0.001] * LOOP_MIN_LAG_SAMPLES, [0.30] * 5)
    assert stats.lag_ms is not None
    assert stats.lag_ms.p90 == pytest.approx(1.0)
    assert stats.lag_max_ms == pytest.approx(1.0)
    assert stats.busy is not None
    assert stats.busy.p90 == pytest.approx(30.0)
    assert stats.lag_samples == LOOP_MIN_LAG_SAMPLES
    assert stats.busy_windows == 5
    assert client_loop_warning(stats, ttft_p50_ms=50.0, itl_p50_ms=10.0) is None


def test_a_busy_loop_thread_alone_does_not_warn() -> None:
    # A saturated thread that still runs its timers on time is headroom
    # information, not evidence that the reported latencies are wrong.
    stats = summarize_loop_samples([0.001] * LOOP_MIN_LAG_SAMPLES, [0.97] * 5)
    assert stats.busy is not None
    assert stats.busy.p90 == pytest.approx(97.0)
    assert client_loop_warning(stats, ttft_p50_ms=50.0, itl_p50_ms=10.0) is None


def test_a_probe_with_no_samples_reports_nothing() -> None:
    stats = summarize_loop_samples([], [])
    assert (stats.lag_ms, stats.lag_max_ms, stats.busy) == (None, None, None)
    assert (stats.lag_samples, stats.busy_windows) == (0, 0)
    assert (
        client_loop_warning(stats, ttft_p50_ms=200.0, itl_p50_ms=10.0) is None
    )


def test_a_loop_that_never_ran_its_timer_is_reported() -> None:
    # The pathological case the sample minimum used to hide: the window was
    # long enough for the samples and the loop took none of them.
    stats = summarize_loop_samples([], [], window_s=19.0)
    assert stats.expected_lag_samples == 1900
    warning = client_loop_warning(stats, ttft_p50_ms=2498.6, itl_p50_ms=None)
    assert warning is not None
    assert warning.startswith("client loop took 0 lag samples in a 19 s")
    assert "(2499 ms)" in warning


def test_a_starved_loop_is_reported_instead_of_silence() -> None:
    # A run that shipped no warning: 35 samples where a 19 s window at a 10 ms
    # interval holds about 1900, and a reported TTFT 15x the server's own.
    stats = _stats(
        lag_p90_ms=1509.3, busy_p90_pct=27.0, samples=35, window_s=19.0
    )
    assert stats.lag_samples < LOOP_MIN_LAG_SAMPLES
    warning = client_loop_warning(stats, ttft_p50_ms=2498.6, itl_p50_ms=0.77)
    assert warning == (
        "client loop took 35 lag samples in a 19 s measured window"
        " (about 1900 expected), worst wakeup 1509 ms late; the loop was"
        " rarely scheduled, so reported TTFT p50 (2499 ms) includes the"
        " client's own delay"
        " (loop blocked or process descheduled, busy p90 27%)"
    )


def test_a_short_window_is_not_called_starved() -> None:
    # Three samples is everything a 0.2 s window can hold: nothing is missing.
    stats = _stats(lag_p90_ms=0.5, samples=3, window_s=0.2)
    assert stats.expected_lag_samples == 20
    assert (
        client_loop_warning(stats, ttft_p50_ms=200.0, itl_p50_ms=10.0) is None
    )


def test_a_healthy_long_window_is_not_called_starved() -> None:
    stats = _stats(
        lag_p90_ms=0.4, busy_p90_pct=97.0, samples=1900, window_s=19.0
    )
    assert client_loop_warning(stats, ttft_p50_ms=50.0, itl_p50_ms=10.0) is None


def test_the_starvation_share_is_the_boundary() -> None:
    # A 3.5 s window holds 350 samples, so a quarter is 87.5: that is the last
    # count still reported as starvation rather than as a slow loop.
    below_share = _stats(lag_p90_ms=0.4, samples=87, window_s=3.5)
    assert below_share.expected_lag_samples == 350
    warning = client_loop_warning(
        below_share, ttft_p50_ms=50.0, itl_p50_ms=10.0
    )
    assert warning is not None
    assert "(about 350 expected)" in warning
    at_share = _stats(lag_p90_ms=0.4, samples=88, window_s=3.5)
    assert (
        client_loop_warning(at_share, ttft_p50_ms=50.0, itl_p50_ms=10.0) is None
    )


def test_a_starved_loop_over_the_lag_threshold_gets_the_lag_message() -> None:
    # 305 samples of a 19 s window is 16% of what it holds, but 305 is enough
    # to judge the lag, and a lag of 103 ms reached both reported metrics:
    # reporting the sample deficit instead would drop the lag p90 and the ITL
    # collapse, which are the numbers a reader needs.
    stats = _stats(
        lag_p90_ms=103.0, busy_p90_pct=98.0, samples=305, window_s=19.0
    )
    warning = client_loop_warning(stats, ttft_p50_ms=583.3, itl_p50_ms=2.2)
    assert warning is not None
    assert warning.startswith("client loop lag p90 103 ms >= ")
    assert "reported TTFT p50 includes the client's own delay" in warning
    assert "reported ITL no longer reflects" in warning
    assert "lag samples" not in warning


def test_a_blocked_loop_shows_as_lag() -> None:
    async def measure() -> ClientLoopStats:
        probe = ClientLoopProbe(interval_s=0.01, window_s=0.05)
        probe.start()
        # Holds the loop the way a saturated client does, in slices that each
        # keep the probe's next timer waiting until the slice ends. The yields
        # after a slice let that overdue timer run and be recorded, so a slice
        # yields about one sample and the run collects more than the minimum.
        for _ in range(150):
            time.sleep(0.02)
            for _ in range(3):
                await asyncio.sleep(0)
        await probe.aclose()
        return probe.stats()

    stats = asyncio.run(measure())
    assert stats.lag_ms is not None
    assert stats.lag_ms.p90 > LOOP_LAG_FLOOR_MS
    assert stats.lag_max_ms is not None
    assert stats.lag_max_ms >= stats.lag_ms.p90
    assert stats.busy is not None
    assert stats.lag_samples >= LOOP_MIN_LAG_SAMPLES
    # A loop this late reads several events per pass, which is why the ITL it
    # reports collapses.
    warning = client_loop_warning(stats, ttft_p50_ms=200.0, itl_p50_ms=0.03)
    assert warning is not None
    assert "reported ITL no longer reflects" in warning


def test_the_probe_samples_at_the_calibrated_interval() -> None:
    async def measure() -> ClientLoopStats:
        probe = ClientLoopProbe()
        probe.start()
        await asyncio.sleep(0.2)
        await probe.aclose()
        return probe.stats()

    stats = asyncio.run(measure())
    # A 200 ms run at the shipped 10 ms interval is about 20 samples; the
    # bounds leave room for a loaded machine without admitting a 100 ms timer.
    assert LOOP_PROBE_INTERVAL_S == pytest.approx(0.01)
    assert 5 <= stats.lag_samples <= 60
    # The probe reports the span it covered, which is what the starvation
    # check compares its sample count against.
    assert stats.window_s == pytest.approx(0.2, abs=0.05)
    assert stats.expected_lag_samples == pytest.approx(20, abs=5)


def test_cancelling_the_caller_during_aclose_still_raises() -> None:
    async def measure() -> None:
        probe = ClientLoopProbe()
        probe.start()
        await asyncio.sleep(0.01)
        caller = asyncio.current_task()
        assert caller is not None
        caller.cancel()
        # The pending cancellation lands on ``aclose``'s wait, which must not
        # absorb it along with the probe's own.
        with pytest.raises(asyncio.CancelledError):
            await probe.aclose()

    asyncio.run(measure())
