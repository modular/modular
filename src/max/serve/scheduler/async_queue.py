# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import logging
import sys
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

logger = logging.getLogger("max.serve")


class Call(NamedTuple):
    f: Callable
    args: Sequence[Any]
    kw: Mapping[str, Any]


if sys.version_info >= (3, 13):
    QueueShutdown = asyncio.QueueShutdown
else:

    class QueueShutDown(Exception):
        pass


class NotStarted(Exception):
    pass


class AsyncCallConsumer:
    """Execute a sync function asynchronously

    Use an asyncio Queue & Task to asynchronously execute sync functions.
    """

    def __init__(self, maxsize=0):
        self.q: asyncio.Queue[Call] = asyncio.Queue(maxsize=maxsize)
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
            logger.error("AsyncCallConsumer timeout out while stopping")
        finally:
            self.task = None

    def call(self, f, *args, **kw):
        if self.task is None:
            raise NotStarted("Worker task not started. Cannot enqueue work.")
        try:
            self.q.put_nowait(Call(f, args, kw))
        except asyncio.QueueFull:
            logger.exception("Telemetry queue full: not recording")

    @staticmethod
    async def _consume(q):
        while True:
            try:
                f, args, kw = await q.get()
            except QueueShutDown:
                break
            except asyncio.CancelledError:
                logger.debug("AsyncCallConsumer cancelled")
                break

            try:
                f(*args, **kw)
            except:
                logger.exception("Failed to record telemetry")

        logger.debug("AsyncCallConsumer consumer shut down")

    async def __aenter__(self):
        self.start()
        return self

    async def __aexit__(self, type, value, traceback):
        await self.shutdown()
