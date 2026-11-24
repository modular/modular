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

from collections.abc import Sequence
from types import SimpleNamespace
from unittest import mock

import pytest
from max.serve.telemetry._cpp_metrics_bridge import (
    CppMetricsBridge,
    start_cpp_metrics_bridge,
)


def _make_snapshot(
    counters: Sequence[tuple[str, int | float]] = (),
    gauges: Sequence[tuple[str, int | float]] = (),
    histograms: Sequence[tuple[str, SimpleNamespace]] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        counters=[SimpleNamespace(name=n, value=v) for n, v in counters],
        gauges=[SimpleNamespace(name=n, value=v) for n, v in gauges],
        histograms=[SimpleNamespace(name=n, ss=s) for n, s in histograms],
    )


def test_counter_first_poll_emits_full_value() -> None:
    bridge = CppMetricsBridge()
    counter = mock.MagicMock()
    bridge._counters["foo"] = counter

    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        return_value=_make_snapshot(counters=[("foo", 10)]),
    ):
        bridge.poll()

    counter.add.assert_called_once_with(10)
    assert bridge._prev_counter_values["foo"] == 10


def test_counter_second_poll_emits_delta() -> None:
    bridge = CppMetricsBridge()
    counter = mock.MagicMock()
    bridge._counters["foo"] = counter

    snapshots = [
        _make_snapshot(counters=[("foo", 10)]),
        _make_snapshot(counters=[("foo", 25)]),
    ]
    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        side_effect=snapshots,
    ):
        bridge.poll()
        bridge.poll()

    assert counter.add.call_args_list == [mock.call(10), mock.call(15)]


def test_counter_no_add_when_delta_is_zero() -> None:
    bridge = CppMetricsBridge()
    counter = mock.MagicMock()
    bridge._counters["foo"] = counter

    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        side_effect=[
            _make_snapshot(counters=[("foo", 5)]),
            _make_snapshot(counters=[("foo", 5)]),
        ],
    ):
        bridge.poll()
        bridge.poll()

    counter.add.assert_called_once_with(5)


def test_counter_no_add_when_delta_is_negative() -> None:
    bridge = CppMetricsBridge()
    counter = mock.MagicMock()
    bridge._counters["foo"] = counter

    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        side_effect=[
            _make_snapshot(counters=[("foo", 25)]),
            _make_snapshot(counters=[("foo", 10)]),
        ],
    ):
        bridge.poll()
        bridge.poll()

    counter.add.assert_called_once_with(25)
    # A counter going backwards means the collector restarted, so the lower
    # value becomes the baseline the next delta is measured from.
    assert bridge._prev_counter_values["foo"] == 10


def test_gauge_set_called_with_current_value() -> None:
    bridge = CppMetricsBridge()
    gauge = mock.MagicMock()
    bridge._gauges["bar"] = gauge

    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        return_value=_make_snapshot(gauges=[("bar", 42)]),
    ):
        bridge.poll()

    gauge.set.assert_called_once_with(42)


def test_gauge_follows_current_value_each_poll() -> None:
    bridge = CppMetricsBridge()
    gauge = mock.MagicMock()
    bridge._gauges["bar"] = gauge

    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        side_effect=[
            _make_snapshot(gauges=[("bar", 10)]),
            _make_snapshot(gauges=[("bar", 3)]),
        ],
    ):
        bridge.poll()
        bridge.poll()

    assert gauge.set.call_args_list == [mock.call(10), mock.call(3)]


def test_histograms_are_ignored() -> None:
    bridge = CppMetricsBridge()

    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        return_value=_make_snapshot(
            histograms=[("latency", SimpleNamespace(count=5, sum=100.0))]
        ),
    ):
        bridge.poll()

    assert bridge._counters == {}
    assert bridge._gauges == {}


@pytest.mark.asyncio
async def test_shutdown_drains_metrics_recorded_since_last_poll() -> None:
    """The poll loop sleeps first, so a short-lived bridge never ticks."""
    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        return_value=_make_snapshot(counters=[("foo", 7)]),
    ) as collect:
        async with start_cpp_metrics_bridge(interval_s=3600.0):
            pass

    collect.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_poll_failure_does_not_propagate() -> None:
    """A failing drain stays contained; the context manager exits cleanly."""
    with mock.patch(
        "max.serve.telemetry._cpp_metrics_bridge._cxx_metrics.collect",
        side_effect=RuntimeError("collector is gone"),
    ) as collect:
        async with start_cpp_metrics_bridge(interval_s=3600.0):
            pass

    # The drain ran and raised. Reaching this line is what proves the failure
    # was swallowed rather than propagated out of the context manager.
    collect.assert_called_once()
