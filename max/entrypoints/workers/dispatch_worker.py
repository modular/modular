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
from max.serve.kvcache_agent.dispatcher_factory import DispatcherFactory
from max.serve.process_control import ProcessControl, ProcessMonitor
from max.serve.telemetry.common import configure_logging

logger = logging.getLogger("max.entrypoints")


async def run_dispatch_worker(
    settings: Settings,
    pc: ProcessControl,
    dispatcher_factory: DispatcherFactory,
) -> None:
    configure_logging(settings, silent=True)
    logger.info(f"Starting dispatch worker on process {os.getpid()}")

    dispatch_service = dispatcher_factory.create_service(process_control=pc)
    await dispatch_service.start()

    pc.set_started()
    logger.info("Started Dispatcher Worker!")

    try:
        while not pc.is_canceled():
            await asyncio.sleep(0.1)
    finally:
        await dispatch_service.stop()
        pc.set_completed()
        logger.info("Stopped Dispatch Worker!")


def _dispatch_process_fn(
    settings: Settings,
    pc: ProcessControl,
    dispatcher_factory: DispatcherFactory,
) -> None:
    try:
        asyncio.run(run_dispatch_worker(settings, pc, dispatcher_factory))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(f"Error in dispatcher worker: {e}")
        raise e


@asynccontextmanager
async def start_dispatch_worker(
    settings: Settings,
    dispatcher_factory: DispatcherFactory,
) -> AsyncGenerator[None, None]:
    mp_context = multiprocessing.get_context("spawn")
    process_name = f"DISPATCH_WORKER_{uuid.uuid4()}"
    pc = ProcessControl(mp_context, process_name)

    logger.info(f"Starting Dispatch Worker: {process_name}")
    process = mp_context.Process(
        name=process_name,
        target=_dispatch_process_fn,
        daemon=True,
        args=(settings, pc, dispatcher_factory),
    )

    process.start()
    monitor = ProcessMonitor(pc, process)

    await monitor.wait_for_startup(timeout=10, shutdown_on_failure=True)
    logger.info("Dispatcher Worker started successfully!")

    try:
        yield
    finally:
        logger.info("Shutting down Dispatch Worker...")
        await monitor.shutdown()
        logger.info("Dispatcher Worker shutdown complete.")
