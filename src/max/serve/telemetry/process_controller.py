# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import multiprocessing
import multiprocessing.queues
import queue
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from max.serve.config import MetricLevel, Settings
from max.serve.scheduler.process_control import (
    ProcessControl,
    ProcessMonitor,
)
from max.serve.telemetry.common import configure_logging, configure_metrics
from max.serve.telemetry.metrics import (
    MaxMeasurement,
    MetricClient,
    TelemetryFn,
)

logger = logging.getLogger(__name__)


# Unused class so taht SHUTDOWNS is not empty
class _Shutdown(Exception):
    pass


# In python 3.13, queues can be told to drain & close, ie "shutdown". Handle that exception.
_SHUTDOWNS = [_Shutdown]
if hasattr(queue, "Shutdown"):
    _SHUTDOWNS.append(queue.Shutdown)


SHUTDOWNS = tuple(_SHUTDOWNS)


def _sync_commit(m: MaxMeasurement) -> None:
    m.commit()


class ProcessMetricClient(MetricClient):
    def __init__(self, settings: Settings, q: multiprocessing.queues.Queue):
        self.queue = q
        # buffer detailed metrics observations until it is safe to flush
        self.detailed_buffer: list[MaxMeasurement] = []
        self.level = settings.metric_level

    def send_measurement(self, m: MaxMeasurement, level: MetricLevel) -> None:
        if level > self.level:
            return

        if level >= MetricLevel.DETAILED:
            # put the measurement in a queue and return
            self.detailed_buffer.append(m)
            if len(self.detailed_buffer) < 20:
                return

        try:
            self.queue.put_nowait([m])
            if self.detailed_buffer:
                self.queue.put_nowait(self.detailed_buffer)
        except queue.Full:
            # we would rather lose data than slow the server
            logger.error("Telemetry Queue is full.  Dropping data")
        finally:
            if self.detailed_buffer:
                self.detailed_buffer.clear()

    def __del__(self):
        try:
            if self.detailed_buffer:
                self.queue.put_nowait(self.detailed_buffer)
        except queue.Full:
            # we would rather lose data than slow the server
            logger.error("Telemetry Queue is full.  Dropping data")
        finally:
            if self.detailed_buffer:
                self.detailed_buffer.clear()


@dataclass
class ProcessTelemetryController:
    pc: ProcessControl
    process: multiprocessing.process.BaseProcess
    queue: multiprocessing.queues.Queue

    def Client(self, settings: Settings) -> MetricClient:
        return ProcessMetricClient(settings, self.queue)


@asynccontextmanager
async def start_process_consumer(
    settings: Settings, handle_fn: Optional[TelemetryFn] = None
) -> AsyncGenerator[ProcessTelemetryController, None]:
    ctx = multiprocessing.get_context("spawn")
    pc = ProcessControl(ctx, "telemetry-worker", health_fail_s=5.0)

    q = ctx.Queue()

    if handle_fn is None:
        handle_fn = _sync_commit

    worker = ctx.Process(
        name="telemetry-worker",
        target=init_and_process,
        daemon=True,
        args=(pc, settings, q, handle_fn),
    )
    worker.start()
    monitor = ProcessMonitor(
        pc,
        worker,
        poll_s=100e-3,
        max_time_s=settings.telemetry_worker_spawn_timeout,
        unhealthy_poll_s=500e-3,
    )

    healthy = await monitor.until_healthy()
    if not healthy:
        raise Exception("telemetry-worker did not come up")

    loop = asyncio.get_running_loop()
    try:
        task = loop.create_task(monitor.shutdown_if_unhealthy())
        yield ProcessTelemetryController(pc, worker, q)
    finally:
        task.cancel()
        await monitor.shutdown()


def init_and_process(
    pc: ProcessControl,
    settings: Settings,
    q: multiprocessing.queues.Queue,  # Queue[MaxMeasurement]
    commit_fn: TelemetryFn,
) -> None:
    configure_logging(settings)
    configure_metrics(settings)
    return process_telemetry(pc, settings, q, commit_fn)


def process_telemetry(
    pc: ProcessControl,
    settings: Settings,
    q: multiprocessing.queues.Queue,  # Queue[MaxMeasurement]
    commit_fn: TelemetryFn,
) -> None:
    """Long running function to read from a queue & process each element"""
    pc.set_started()
    try:
        while True:
            pc.beat()

            try:
                ms = q.get(block=True, timeout=100e-3)
            except queue.Empty:
                if pc.is_canceled():
                    break
                continue
            except SHUTDOWNS:
                break

            try:
                for m in ms:
                    commit_fn(m)
            except:
                logger.exception("Error processing telemetry")
    except KeyboardInterrupt:
        pass
    finally:
        pc.set_completed()
