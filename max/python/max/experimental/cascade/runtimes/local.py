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
"""In-process worker runtime used as the core for Cascade transports."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, cast

from max.experimental.cascade.core import Runtime, Worker


def _default_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class LocalRuntime(Runtime):
    """In-process worker runtime; serves as the core for Cascade transports."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger if logger is not None else _default_logger()
        self._workers: dict[str, Worker] = {}
        self._worker_class_names: dict[str, str] = {}
        self._results: dict[str, asyncio.Future[Any]] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._actid_counter = itertools.count()
        self._resid_counter = itertools.count()
        # Holds entered worker.open() contexts; populated only inside open().
        self._worker_stack: AsyncExitStack | None = None

    @asynccontextmanager
    async def open(self) -> AsyncIterator[LocalRuntime]:
        """Lifecycle context: yields self, cleans up workers and tasks on exit."""
        async with AsyncExitStack() as worker_stack:
            self._worker_stack = worker_stack
            try:
                yield self
            finally:
                # Cancel any in-flight calls and wait for them to settle
                # before tearing down worker contexts.
                for task in list(self._background_tasks):
                    task.cancel()
                if self._background_tasks:
                    await asyncio.gather(
                        *self._background_tasks, return_exceptions=True
                    )

                # AsyncExitStack unwind below closes each worker.open() in
                # reverse deploy order.
                self._worker_stack = None
                self._workers.clear()
                self._worker_class_names.clear()
                self._results.clear()
                self._background_tasks.clear()

    async def deploy_worker(self, worker: Worker) -> str:
        """Register a deployed worker and return its stable id.

        Enters the worker's ``open()`` context; it will be exited when the
        runtime's ``open()`` context exits. The base class's :py:meth:`deploy`
        wraps this id in a client-side :py:class:`Proxy`.
        """
        if self._worker_stack is None:
            raise RuntimeError(
                "LocalRuntime.deploy_worker() called outside of open() context"
            )
        cls_name = type(worker).__name__
        worker_id = f"{cls_name}-{next(self._actid_counter)}"
        self.logger.info("Deploying worker %s", worker_id)
        live_worker = await self._worker_stack.enter_async_context(
            worker.open()
        )
        self._workers[worker_id] = live_worker
        self._worker_class_names[worker_id] = cls_name
        return worker_id

    def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Launch an worker method asynchronously and bind it to a result id."""
        if worker_id not in self._workers:
            raise KeyError(f"Unknown worker id: {worker_id!r}")
        worker = self._workers[worker_id]
        method = cast(Any, getattr(worker, func))
        cls_name = self._worker_class_names[worker_id]

        result_id = f"{cls_name}.{func}-{next(self._resid_counter)}"
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._results[result_id] = future

        async def run_call() -> None:
            try:
                # Argument materialization is owned by the ``@worker_method``
                # decorator on the worker side, so the runtime simply forwards
                # whatever args (including ``Result`` handles) it received.
                ret = method(*args, **kwargs)
                if inspect.isasyncgen(ret):
                    # Stream: result future resolves to the async iterator,
                    # which subsequent next_result() calls advance.
                    future.set_result(ret)
                    return
                result = await ret if inspect.isawaitable(ret) else ret
                future.set_result(result)
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as exc:
                self.logger.exception(
                    "Cascade worker method %s.%s failed", cls_name, func
                )
                if not future.done():
                    future.set_exception(exc)

        task = asyncio.create_task(run_call())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return result_id

    async def get_result(self, result_id: str) -> Any:
        """Await a single result future (raises if the call failed)."""
        return await self._lookup(result_id)

    async def next_result(self, result_id: str) -> Any:
        """Advance a streaming result one step.

        Raises ``StopAsyncIteration`` at end of stream, or whatever exception
        the underlying async generator raises.
        """
        source = await self._lookup(result_id)
        if not isinstance(source, AsyncIterator) and not inspect.isasyncgen(
            source
        ):
            raise TypeError(f"Result {result_id!r} is not a stream")
        try:
            return await anext(source)
        except StopAsyncIteration:
            # Stream is done; drop bookkeeping so result_id can't be reused.
            self._results.pop(result_id, None)
            raise

    async def get_metrics(self) -> str:
        """Return Prometheus exposition text for this runtime."""
        return ""

    def _lookup(self, result_id: str) -> asyncio.Future[Any]:
        try:
            return self._results[result_id]
        except KeyError:
            raise KeyError(f"Unknown result id: {result_id!r}") from None
