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
import logging
import multiprocessing
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from max.serve.config import Settings
from max.serve.kvcache_agent import (
    DispatcherFactory,
    start_kvcache_agent_service,
)
from max.serve.process_control import ProcessControl, ProcessMonitor
from max.serve.scheduler.base import PayloadType

logger = logging.getLogger("max.serve")


async def run_kvcache_agent_process(
    pc: ProcessControl,
    settings: Settings,
    dispatcher_factory: DispatcherFactory[PayloadType],
) -> None:
    pid = os.getpid()
    logger.info("Starting KV Cache Agent on process %d!", pid)

    # Create and start services
    kvcache_agent_service = start_kvcache_agent_service(
        kv_cache_events_zmq_endpoint=settings.kv_cache_events_zmq_endpoint,
    )
    dispatcher_service = dispatcher_factory.create_service()
    await dispatcher_service.start()

    pc.set_started()
    logger.debug("Started KV Cache Agent!")

    # Run the blocking call in a thread so the event loop stays alive
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, kvcache_agent_service.wait_for_termination)

    pc.set_completed()
    logger.info("Stopped KV Cache Agent!")


def _kvcache_agent_process_fn(
    pc: ProcessControl,
    settings: Settings,
    dispatcher_factory: DispatcherFactory[PayloadType],
) -> None:
    try:
        asyncio.run(run_kvcache_agent_process(pc, settings, dispatcher_factory))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(
            "Encountered an error in _kvcache_agent_process_fn %s",
            e,
            stack_info=True,
        )


@asynccontextmanager
async def start_kv_cache_service(
    settings: Settings,
    dispatcher_factory: DispatcherFactory[PayloadType],
) -> AsyncGenerator[None, None]:
    """Starts a kvcache agent and associated process."""
    process_name = "KVCACHE_AGENT_" + str(uuid.uuid4())

    mp_context = multiprocessing.get_context("spawn")
    pc = ProcessControl(mp_context, "kvcache-agent")

    logger.info("Starting KV Cache Agent: %s", process_name)
    process = mp_context.Process(
        name=process_name,
        target=_kvcache_agent_process_fn,
        daemon=True,
        args=(pc, settings, dispatcher_factory),
    )
    process.start()
    monitor = ProcessMonitor(pc, process)

    # before progressing, observe the kvcache agent process to be healthy or dead
    dt = asyncio.create_task(monitor.until_dead())
    ht = asyncio.create_task(monitor.until_started())

    completed_tasks, pending_tasks = await asyncio.wait(
        [ht, dt], timeout=10, return_when=asyncio.FIRST_COMPLETED
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
        yield
    finally:
        process_task.cancel()
        await monitor.shutdown()
