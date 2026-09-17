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

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from max._core import metrics as _cxx_metrics
from opentelemetry.metrics import get_meter_provider

__all__ = ["CppMetricsBridge", "start_cpp_metrics_bridge"]

logger = logging.getLogger("max.serve")

_POLL_INTERVAL_S = 5.0


class CppMetricsBridge:
    """Polls the in-process C++ MetricsCollector and forwards readings to OTel.

    Counters emit deltas: the C++ Counter is cumulative and never resets, so
    the bridge tracks the previous snapshot value and emits only the increment
    each interval.

    Gauges are forwarded as-is.

    Histograms are not bridged. OTel histograms require individual observations
    to compute bucket distributions, but collect() returns a pre-aggregated
    snapshot. Use Counter pairs (e.g. foo.sum + foo.count) for C++ distribution
    metrics instead.
    """

    def __init__(self) -> None:
        self._meter = get_meter_provider().get_meter("modular.cxx")
        # The sync gauge is only exported as opentelemetry.metrics._Gauge, so
        # both maps stay loosely typed rather than mixing one public
        # instrument type with one underscore-prefixed one.
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._prev_counter_values: dict[str, int] = {}

    def _counter(self, name: str) -> Any:
        if name not in self._counters:
            self._counters[name] = self._meter.create_counter(name)
        return self._counters[name]

    def _gauge(self, name: str) -> Any:
        if name not in self._gauges:
            self._gauges[name] = self._meter.create_gauge(name)
        return self._gauges[name]

    def poll(self) -> None:
        snapshot = _cxx_metrics.collect()
        for counter in snapshot.counters:
            prev = self._prev_counter_values.get(counter.name, 0)
            delta = counter.value - prev
            self._prev_counter_values[counter.name] = counter.value
            if delta > 0:
                self._counter(counter.name).add(delta)
        for gauge in snapshot.gauges:
            self._gauge(gauge.name).set(gauge.value)

    async def run(self, interval_s: float = _POLL_INTERVAL_S) -> None:
        while True:
            await asyncio.sleep(interval_s)
            try:
                self.poll()
            except Exception:
                logger.warning("C++ metrics poll failed", exc_info=True)


@asynccontextmanager
async def start_cpp_metrics_bridge(
    interval_s: float = _POLL_INTERVAL_S,
) -> AsyncGenerator[None, None]:
    bridge = CppMetricsBridge()
    task = asyncio.create_task(bridge.run(interval_s))
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The loop sleeps before each poll, so whatever was recorded since the
        # last tick is still unexported. Drain it before the process exits.
        try:
            bridge.poll()
        except Exception:
            logger.warning("Final C++ metrics poll failed", exc_info=True)
