# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import queue
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from max.serve.config import MetricLevel, Settings
from max.serve.telemetry.metrics import MaxMeasurement, MetricClient

logger = logging.getLogger(__name__)


if sys.version_info >= (3, 13):
    QueueShutdown = asyncio.QueueShutdown
else:

    class QueueShutDown(Exception):
        pass


class NotStarted(Exception):
    pass


class AsyncioMetricClient(MetricClient):
    def __init__(self, settings: Settings, q):
        self.q = q
        self.level = settings.metric_level

    def send_measurement(self, m: MaxMeasurement, level: MetricLevel) -> None:
        if level > self.level:
            return
        try:
            self.q.put_nowait(m)
        except queue.Full:
            logger.error("Telemetry Queue is full.  Dropping data")


class AsyncioTelemetryController:
    """Execute a sync function asynchronously

    Use an asyncio Queue & Task to asynchronously commit metric measurements
    """

    def __init__(self, maxsize=0):
        self.q: asyncio.Queue[MaxMeasurement] = asyncio.Queue(maxsize=maxsize)
        self.task: Optional[asyncio.Task] = None

    def start(self):
        if self.task is not None:
            raise Exception("task already started")
        self.task = asyncio.create_task(self._consume(self.q))

    async def shutdown(self, timeout_s: float = 2.0):
        if sys.version_info >= (3, 13):
            self.q.shutdown()

        # task was never started.  return early
        if self.task is None:
            return

        try:
            await asyncio.wait_for(self.task, timeout_s)
        except asyncio.TimeoutError:
            logger.error(
                "AsyncioTelemetryController timeout out while stopping"
            )
        finally:
            self.task = None

    def Client(self, settings: Settings) -> AsyncioMetricClient:
        if self.task is None:
            raise NotStarted(
                "AsyncioTelemetryController task not started. Cannot enqueue work."
            )
        return AsyncioMetricClient(settings, self.q)

    @staticmethod
    async def _consume(q):
        while True:
            try:
                m: MaxMeasurement = await q.get()
            except QueueShutDown:
                break
            except asyncio.CancelledError:
                logger.info("AsyncioTelemetryController cancelled")
                break

            try:
                m.commit()
            except:
                logger.exception("Failed to record telemetry")

        logger.info("AsyncioTelemetryController consumer shut down")

    async def __aenter__(self):
        self.start()
        return self

    async def __aexit__(self, type, value, traceback):
        await self.shutdown()


@asynccontextmanager
async def start_asyncio_consumer(
    settings: Settings,
) -> AsyncGenerator[AsyncioMetricClient, None]:
    async with AsyncioTelemetryController() as controller:
        yield controller.Client(settings)
