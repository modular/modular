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
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from queue import Queue

from max.serve.config import Settings
from max.serve.kvcache_agent import start_kvcache_agent_service
from max.serve.process_control import (
    ProcessManager,
    subprocess_manager,
)

logger = logging.getLogger("max.serve")


async def run_kvcache_agent_process(
    health: Queue[bool],
    settings: Settings,
) -> None:
    pid = os.getpid()
    logger.info("Starting KV Cache Agent on process %d!", pid)

    # Create and start services
    kvcache_agent_service = start_kvcache_agent_service(
        kv_cache_events_zmq_endpoint=settings.kv_cache_events_zmq_endpoint,
    )

    health.put(True)
    logger.debug("Started KV Cache Agent!")

    # Run the blocking call in a thread so the event loop stays alive
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, kvcache_agent_service.wait_for_termination)

    logger.info("Stopped KV Cache Agent!")


def _kvcache_agent_process_fn(
    health: Queue[bool],
    settings: Settings,
) -> None:
    try:
        asyncio.run(run_kvcache_agent_process(health, settings))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(
            "Encountered an error in _kvcache_agent_process_fn %s",
            e,
            stack_info=True,
        )


# Warning: This method is currently unused!
# Keeping it around since it may be re-enabled in the future.
@asynccontextmanager
async def start_kv_cache_service(
    settings: Settings,
) -> AsyncGenerator[ProcessManager]:
    """Starts a kvcache agent and associated process."""
    process_name = "KVCACHE_AGENT_" + str(uuid.uuid4())
    logger.info("Starting KV Cache Agent: %s", process_name)

    async with subprocess_manager() as proc:
        health = proc.ctx.Queue()
        proc.start(_kvcache_agent_process_fn, health, settings)

        await proc.ready(lambda: health.get(timeout=10))

        logger.debug("KV Cache Agent is alive and healthy")

        yield proc
