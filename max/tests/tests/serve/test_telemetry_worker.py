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

import asyncio
import queue
import time
from unittest import mock

import pytest
from max.serve.config import MetricLevel, Settings
from max.serve.pipelines import telemetry_worker
from max.serve.telemetry import process_controller
from max.serve.telemetry.asyncio_controller import AsyncioMetricClient
from max.serve.telemetry.metrics import MaxMeasurement


@pytest.mark.asyncio
async def test_telemetry_worker() -> None:
    settings = Settings()
    async with telemetry_worker.start_process_consumer(settings) as worker:
        client = worker.Client(settings)
        client.send_measurement(
            MaxMeasurement("foo", 1), level=MetricLevel.BASIC
        )
        client.send_measurement(
            MaxMeasurement("foo", 2), level=MetricLevel.BASIC
        )
        time.sleep(100e-3)
        with pytest.raises(queue.Empty):
            worker.queue.get_nowait()


def _raise_exception(x: MaxMeasurement) -> None:
    """TelemetryFn, but always broken. Only used for tests"""
    raise Exception("I'm always broken")


@pytest.mark.asyncio
async def test_unreliable_handle() -> None:
    settings = Settings()
    async with telemetry_worker.start_process_consumer(
        settings,
        handle_fn=_raise_exception,
    ) as worker:
        client = worker.Client(settings)

        client.send_measurement(
            MaxMeasurement("foo", 1), level=MetricLevel.BASIC
        )
        client.send_measurement(
            MaxMeasurement("foo", 2), level=MetricLevel.BASIC
        )
        client.send_measurement(
            MaxMeasurement("foo", 3), level=MetricLevel.BASIC
        )

        await asyncio.sleep(1)

        with pytest.raises(queue.Empty):
            worker.queue.get_nowait()


@pytest.mark.asyncio
async def test_metric_asyncio_client_filtering() -> None:
    settings = Settings(metric_level=MetricLevel.BASIC)
    assert settings.metric_level == MetricLevel.BASIC

    q = mock.MagicMock()
    client = AsyncioMetricClient(settings.metric_level, q)

    # detailed metrics are dropped
    client.send_measurement(
        MaxMeasurement("foo", 1), level=MetricLevel.DETAILED
    )
    assert q.put_nowait.call_count == 0

    # basic metrics are allowed
    client.send_measurement(MaxMeasurement("foo", 1), level=MetricLevel.BASIC)
    assert q.put_nowait.call_count == 1


@pytest.mark.asyncio
async def test_metric_process_client_filtering() -> None:
    settings = Settings(metric_level=MetricLevel.BASIC)
    assert settings.metric_level == MetricLevel.BASIC

    q = mock.MagicMock()
    client = process_controller.ProcessMetricClient(settings, q)

    # detailed metrics are dropped
    client.send_measurement(
        MaxMeasurement("foo", 1), level=MetricLevel.DETAILED
    )
    assert q.put_nowait.call_count == 0

    # basic metrics are allowed
    client.send_measurement(MaxMeasurement("foo", 1), level=MetricLevel.BASIC)
    assert q.put_nowait.call_count == 1
