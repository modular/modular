# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import multiprocessing
import multiprocessing.queues
import queue
from contextlib import asynccontextmanager
from typing import Callable, NamedTuple, Optional

from max.loggers import get_logger
from max.serve.scheduler.process_control import ProcessControl, ProcessMonitor

logger = get_logger(__name__)


# Unused class so taht SHUTDOWNS is not empty
class _Shutdown(Exception):
    pass


# In python 3.13, queues can be told to drain & close, ie "shutdown". Handle that exception.
_SHUTDOWNS = [_Shutdown]
if hasattr(queue, "Shutdown"):
    _SHUTDOWNS.append(queue.Shutdown)
SHUTDOWNS = tuple(_SHUTDOWNS)


class TelemetryWorker(NamedTuple):
    pc: ProcessControl
    process: multiprocessing.process.BaseProcess
    queue: multiprocessing.queues.Queue


class TelemetryObservation(NamedTuple):
    pass


TelemetryFn = Callable[[TelemetryObservation], None]


@asynccontextmanager
async def start_telemetry_worker(
    handle_fn: Optional[TelemetryFn] = None, worker_spawn_timeout: float = 10
):
    ctx = multiprocessing.get_context("spawn")
    pc = ProcessControl(ctx, "telemetry-worker", health_fail_s=5.0)

    q = ctx.Queue()

    if handle_fn is None:
        handle_fn = handle_element

    worker = ctx.Process(
        name="telemetry-worker",
        target=process_telemetry,
        daemon=True,
        args=(pc, q, handle_fn),
    )
    worker.start()
    monitor = ProcessMonitor(
        pc,
        worker,
        poll_s=100e-3,
        max_time_s=worker_spawn_timeout,
        unhealthy_poll_s=500e-3,
    )

    healthy = await monitor.until_healthy()
    if not healthy:
        raise Exception("telemetry-worker did not come up")

    loop = asyncio.get_running_loop()
    try:
        task = loop.create_task(monitor.shutdown_if_unhealthy())
        yield TelemetryWorker(pc, worker, q)
    finally:
        task.cancel()
        await monitor.shutdown()


def process_telemetry(
    pc: ProcessControl,
    q: multiprocessing.queues.Queue,  # Queue[TelemetryObservation]
    handle_fn: TelemetryFn,
) -> None:
    """Long running function to read from a queue & process each element"""
    pc.set_started()
    try:
        while True:
            pc.beat()

            try:
                obs = q.get(block=True, timeout=100e-3)
            except queue.Empty:
                if pc.is_canceled():
                    break
                continue
            except SHUTDOWNS:
                break

            try:
                handle_fn(obs)
            except:
                logger.exception("Error processing telemetry")
    except KeyboardInterrupt:
        pass
    finally:
        pc.set_completed()


def handle_element(x: TelemetryObservation) -> None:
    pass
