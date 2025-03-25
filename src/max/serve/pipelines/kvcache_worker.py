# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import multiprocessing
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from max.serve.config import Settings
from max.serve.kvcache_agent.kvcache_agent import start_kvcache_agent_service
from max.serve.scheduler.process_control import ProcessControl, ProcessMonitor
from max.serve.telemetry.common import configure_logging

logger = logging.getLogger(__name__)
# This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
logger.propagate = False


def _kvcache_agent_process_fn(
    pc: ProcessControl,
    queue: multiprocessing.Queue,
    settings: Settings,
) -> None:
    configure_logging(settings)

    try:
        server = start_kvcache_agent_service(queue)
        pc.set_started()
        logger.debug("Started KV Cache Agent!")

        server.wait_for_termination()
        pc.set_completed()
        logger.info("Stopped KV Cache Agent!")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(
            "Encountered an error in _kvcache_agent_process_fn %s",
            e,
            stack_info=True,
        )


@asynccontextmanager
async def start_kvcache_agent(
    settings: Settings,
) -> AsyncGenerator[Optional[multiprocessing.Queue], None]:
    """Starts a kvcache agent and associated process."""
    process_name = "KVCACHE_AGENT_" + str(uuid.uuid4())

    mp_context = multiprocessing.get_context("spawn")
    pc = ProcessControl(
        mp_context,
        "kvcache-agent",
        health_fail_s=settings.mw_health_fail_s,
    )
    queue: multiprocessing.Queue = mp_context.Queue()

    logger.info("Starting KV Cache Agent: %s", process_name)
    process = mp_context.Process(
        name=process_name,
        target=_kvcache_agent_process_fn,
        daemon=True,
        args=(pc, queue, settings),
    )
    process.start()
    monitor = ProcessMonitor(pc, process)

    # before progressing, observe the kvcache agent process to be healthy or dead
    dt = asyncio.create_task(monitor.until_dead())
    ht = asyncio.create_task(monitor.until_started())

    completed_tasks, pending_tasks = await asyncio.wait(
        [ht, dt],
        timeout=10,
        return_when=asyncio.FIRST_COMPLETED,
    )

    # cleanup tasks
    # observe the completed tasks
    for t in completed_tasks:
        await t
    # cancel the pending tasks
    for t in pending_tasks:
        t.cancel()

    # figure out if we are in a clean state
    # verify something completed
    if not ht.done() and not dt.done():
        # somehow neither task finished
        raise TimeoutError("KV Cache Agent is neither dead nor healthy")

    # are we in a run-able state?
    if not process.is_alive():
        logger.critical(
            f"KV Cache Agent ended pre-maturely with exitcode: {process.exitcode}"
        )
        # cannot continue if the worker is dead
        await monitor.shutdown()
        if pc.is_healthy():
            raise TimeoutError("KV Cache Agent became healthy and died")
        else:
            raise TimeoutError("KV Cache Agent died")

    # worker is alive!  it needs to be healthy too.

    if not pc.is_started():
        # cannot continue if the worker is not started
        await monitor.shutdown()
        raise TimeoutError("KV Cache Agent did not start")

    # worker is both alive and healthy!
    logger.debug("KV Cache Agent is alive and healthy")

    try:
        process_task = asyncio.create_task(monitor.shutdown_if_dead())
        yield queue
    finally:
        process_task.cancel()
        await monitor.shutdown()
