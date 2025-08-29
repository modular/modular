# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
import functools
import logging
import multiprocessing
import multiprocessing.queues
import queue
import threading
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import prometheus_client
from max.serve.config import MetricLevel, MetricRecordingMethod, Settings
from max.serve.process_control import ProcessControl, ProcessMonitor
from max.serve.telemetry.common import configure_metrics
from max.serve.telemetry.metrics import (
    MaxMeasurement,
    MetricClient,
    TelemetryFn,
)
from uvicorn import Config, Server

logger = logging.getLogger("max.serve")


# Unused class so that SHUTDOWNS is not empty
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
    def __init__(
        self,
        settings: Settings,
        q: multiprocessing.queues.Queue,
    ) -> None:
        self.queue = q
        # buffer detailed metrics observations until it is safe to flush
        self.detailed_buffer: list[MaxMeasurement] = []
        # Important: If any other items of settings are pulled out here in
        # __init__, please make sure they are put back into the reconstructed
        # Settings object inside of cross_process_factory.
        self.metric_detail_level = settings.metric_level

        # buffer detailed metrics observations until it is safe to flush
        self.buffer_factor = settings.detailed_metric_buffer_factor

    def send_measurement(self, m: MaxMeasurement, level: MetricLevel) -> None:
        if level > self.metric_detail_level:
            logger.debug(
                f"Skipping metric {m.instrument_name} at level {level} because detail level is {self.metric_detail_level}"
            )
            return

        if (
            level >= MetricLevel.DETAILED
            and len(self.detailed_buffer) < self.buffer_factor
        ):
            # put the measurement in a queue and return
            self.detailed_buffer.append(m)
            return

        try:
            payload = [m]
            if self.detailed_buffer:
                payload.extend(self.detailed_buffer)
            self.queue.put_nowait(payload)
        except queue.Full:
            # we would rather lose data than slow the server
            logger.warning(
                f"Telemetry Queue is full.  Dropping {len(self.detailed_buffer)} measurements"
            )
        finally:
            if self.detailed_buffer:
                # NOTE: we want to create a new list here to avoid
                # holding references to the old list. That could lead to the
                # list being modified before it's serialized and sent over the
                # wire by the multiprocessing queue.
                # This is technically avoided by using extend above, but let's
                # just do this to be extra safe.
                self.detailed_buffer = []

    def cross_process_factory(
        self,
        settings: Settings,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        return functools.partial(_reconstruct_client, settings, self.queue)

    def __del__(self) -> None:
        try:
            if self.detailed_buffer:
                self.queue.put_nowait(self.detailed_buffer)
        except queue.Full:
            # we would rather lose data than slow the server
            logger.warning("Telemetry Queue is full.  Dropping data")
        finally:
            if self.detailed_buffer:
                self.detailed_buffer.clear()


@asynccontextmanager
async def _reconstruct_client(
    settings: Settings, q: multiprocessing.queues.Queue
) -> AsyncGenerator[MetricClient, None]:
    yield ProcessMetricClient(settings, q)


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
    """Initialize logging & metrics, and start the metrics server if enabled. This is expected to run from the Telemetry process."""
    configure_metrics(settings)

    if (
        not settings.disable_telemetry
        and settings.metric_recording == MetricRecordingMethod.PROCESS
    ):
        app = prometheus_client.make_asgi_app()
        config = Config(
            app=app,
            host=settings.host,
            port=settings.metrics_port,
            access_log=False,
            log_level="warning",
        )
        server = Server(config)

        def run_server() -> None:
            logger.warning(
                f"Starting ASGI metrics server on port {settings.metrics_port}"
            )
            try:
                server.run()
            except Exception:
                logger.exception("Error running ASGI metrics server")

        # Start the server in a daemon thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

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
