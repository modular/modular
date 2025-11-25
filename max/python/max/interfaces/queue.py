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
from __future__ import annotations

import atexit
import contextlib
import logging
import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from max.profiler import traced

logger = logging.getLogger("max.interfaces")

PushItemType = TypeVar("PushItemType", contravariant=True)
"""Type variable for items accepted by a push queue (contravariant).

This allows the push-side interface to correctly accept supertypes of items.
"""

PullItemType = TypeVar("PullItemType", covariant=True)
"""Type variable for items produced by a pull queue (covariant).

This allows the pull-side interface to correctly produce subtypes of items.
"""


@runtime_checkable
class MAXPushQueue(Protocol, Generic[PushItemType]):
    """
    Protocol for a minimal, non-blocking push queue interface in MAX.

    This protocol defines the contract for a queue that supports non-blocking
    put operations for adding items. It is generic over the item type and designed
    for scenarios where the caller must be immediately notified of success or failure
    rather than waiting for space to become available.

    The protocol is intended for producer-side queue operations where immediate
    feedback is critical for proper flow control and error handling.
    """

    def put_nowait(self, item: PushItemType) -> None:
        """
        Attempt to put an item into the queue without blocking.

        This method is designed to immediately fail (typically by raising an exception)
        if the item cannot be added to the queue at the time of the call. Unlike the
        traditional 'put' method in many queue implementations—which may block until
        space becomes available or the transfer is completed—this method never waits.
        It is intended for use cases where the caller must be notified of failure to
        enqueue immediately, rather than waiting for space.

        Args:
            item (PushItemType): The item to be added to the queue.
        """
        ...


@runtime_checkable
class MAXPullQueue(Protocol, Generic[PullItemType]):
    """
    Protocol for a minimal, non-blocking pull queue interface in MAX.

    This protocol defines the contract for a queue that supports non-blocking
    get operations for retrieving items. It is generic over the item type and designed
    for scenarios where the caller must be immediately notified if no items are available
    rather than waiting for items to arrive.

    The protocol is intended for consumer-side queue operations where immediate
    feedback about queue state is critical for proper flow control and error handling.
    """

    def get_nowait(self) -> PullItemType:
        """
        Remove and return an item from the queue without blocking.

        This method is expected to raise `queue.Empty` if no item is available
        to retrieve from the queue.

        Returns:
            PullItemType: The item removed from the queue.

        Raises:
            queue.Empty: If the queue is empty and no item can be retrieved.
        """
        ...


def drain_queue(
    pull_queue: MAXPullQueue[PullItemType], max_items: int | None = None
) -> list[PullItemType]:
    """
    Remove and return items from the queue without blocking.

    This method is expected to return an empty list if the queue is empty.
    If max_items is specified, at most that many items will be returned.

    Args:
        pull_queue: The queue to drain items from.
        max_items: Maximum number of items to return. If None, returns all items.

    Returns:
        List of items removed from the queue, limited by max_items if specified.
    """

    output: list[PullItemType] = []
    while True:
        if max_items is not None and len(output) >= max_items:
            break

        try:
            output.append(pull_queue.get_nowait())
        except queue.Empty:
            break
    return output


def get_blocking(pull_queue: MAXPullQueue[PullItemType]) -> PullItemType:
    """
    Get the next item from the queue.

    If no item is available, this method will spin until one is.
    """
    while True:
        try:
            return pull_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.001)


class BackgroundQueueDrainer(Generic[PullItemType]):
    """
    A generic background queue drainer that continuously drains items from a source queue
    in a background thread and makes them available for processing in the main thread.

    This class encapsulates the common pattern of:
    1. Running a background thread to drain a source queue
    2. Buffering drained items in an internal queue
    3. Providing a way to process items from the buffer in the main thread

    The drainer ensures that only one background draining task runs at a time and
    provides error handling for the background operations.
    """

    def __init__(
        self,
        source_queue: MAXPullQueue[PullItemType],
        max_items_per_drain: int | None = None,
    ) -> None:
        """
        Initialize the background queue drainer.

        Args:
            source_queue: The queue to drain items from in the background
            max_items_per_drain: Maximum number of items to drain per background operation.
                                If None, drains all available items.
        """
        self.source_queue = source_queue
        self.max_items_per_drain = max_items_per_drain

        # Internal queue for buffering drained items
        self._pending_items = queue.Queue[PullItemType]()

        # Background execution management
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="queue_drainer"
        )
        # One-off background drain task used by start_draining.
        self._drain_future: Future[None] | None = None

        # Continuous draining loop control. This is used by the
        # drain_while_gpu context manager to drain the queue only while
        # the main thread is in a GPU-bound section that releases the GIL.
        self._gpu_window = threading.Event()
        self._stop_event = threading.Event()
        self._loop_future: Future[None] | None = None

        # Register cleanup on process exit
        atexit.register(self.stop_draining)

    @traced
    def start_draining(self) -> None:
        """
        Start draining the source queue in the background.

        If a background draining task is already running, this method returns immediately.
        Otherwise, it submits a new background task to drain the source queue.
        """
        if self._drain_future is not None and not self._drain_future.done():
            return

        if self._drain_future is not None:
            exc = self._drain_future.exception()
            if exc is not None:
                raise exc

        # Start draining the queue in the background
        self._drain_future = self._executor.submit(self._drain_queue_background)

    def _ensure_loop_running(self) -> None:
        """
        Ensure that the continuous draining loop is running.

        This loop is intentionally very lightweight and only becomes active
        while the GPU window is open (see drain_while_gpu).
        """
        if self._loop_future is not None and not self._loop_future.done():
            return

        self._loop_future = self._executor.submit(self._drain_loop)

    def _drain_queue_background(self) -> None:
        """
        Background task to drain the source queue and enqueue items.

        Retrieves all available items from the source queue using the non-blocking
        `drain_queue` utility and puts each item into the internal pending items queue.
        Any exceptions encountered during retrieval are logged as errors.
        """
        try:
            new_items = drain_queue(
                self.source_queue, max_items=self.max_items_per_drain
            )
            for item in new_items:
                self._pending_items.put(item)
        except Exception as e:
            logger.error(f"Error in background draining queue: {e}")
            raise

    def _drain_loop(self) -> None:
        """
        Continuous draining loop.

        This loop waits for the GPU window to be opened via drain_while_gpu()
        and then drains items in small bursts while the main thread is in a
        GPU-bound section that has released the GIL. When the window is
        closed, the loop goes back to sleeping, minimizing GIL contention.
        """
        try:
            while not self._stop_event.is_set():
                # Wait until we are signalled that the GPU is running.
                # A short timeout lets us react promptly to stop events.
                self._gpu_window.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break

                if not self._gpu_window.is_set():
                    # Timed out while window is closed.
                    continue

                try:
                    new_items = drain_queue(
                        self.source_queue, max_items=self.max_items_per_drain
                    )
                    for item in new_items:
                        self._pending_items.put(item)
                except Exception as e:  # pragma: no cover - defensive logging
                    logger.error("Error in background draining loop: %s", e)
                    # Surface the error via the future stored in _loop_future.
                    raise

                # If there was nothing to drain, back off briefly to avoid a
                # busy-spin loop. A 0.5ms sleep keeps polling latency low while
                # significantly reducing GIL/scheduler churn and CPU usage.
                if not new_items:
                    time.sleep(0.0005)
        finally:
            # Ensure the event is cleared so a restarted loop does not
            # immediately start draining.
            self._gpu_window.clear()

    @contextlib.contextmanager
    def drain_while_gpu(self) -> Iterator[None]:
        """
        Context manager that drains the source queue while the GPU is running.

        Typical usage pattern:

            with queue_drainer.drain_while_gpu():
                # Call into a GPU-bound section that releases the GIL,
                # e.g. pipeline.execute(...)
                run_gpu_work()

        While inside the context, a lightweight background loop will drain
        items from ``source_queue`` into ``_pending_items`` in small bursts,
        overlapping this work with GPU execution. When the context exits,
        the loop goes back to sleeping, so regular Python code on the main
        thread runs with minimal additional GIL contention.
        """
        self._ensure_loop_running()
        # Open the "GPU window" so the background loop starts draining while
        # the caller is inside this context. The try/finally ensures that the
        # window is *always* closed again, even if the GPU work raises, so
        # subsequent iterations do not continue draining unexpectedly.
        self._gpu_window.set()
        try:
            yield
        finally:
            self._gpu_window.clear()

    def retrieve_item(self) -> PullItemType:
        """
        Retrieve the next item from the pending items queue.

        Returns:
            PullItemType: The next item from the pending items queue.
        """
        return self._pending_items.get_nowait()

    def retrieve_items(self) -> list[PullItemType]:
        """
        Retrieve all items currently in the pending items queue.

        Returns a list of all items that have been drained from the source queue
        and are currently waiting in the internal pending items queue. This method
        does not block and returns immediately with all available items.

        Returns:
            list[PullItemType]: A list of all items currently in the pending queue,
                                in the order they were drained from the source queue.
        """
        return drain_queue(self._pending_items)

    def stop_draining(self) -> None:
        """
        Stop the background draining task and shutdown the executor.

        This method should be called when the drainer is no longer needed to ensure
        proper cleanup of background resources. It's automatically called on process exit.
        """
        if self._drain_future is not None:
            self._drain_future.cancel()
            self._drain_future = None

        # Stop the continuous draining loop if it was started.
        self._stop_event.set()
        self._gpu_window.set()  # Wake the loop so it can exit promptly.
        if self._loop_future is not None:
            self._loop_future.cancel()
            self._loop_future = None

        # Shutdown the executor - this will stop accepting new tasks and wait for
        # the current task to complete (which should be quick since it's non-blocking)
        self._executor.shutdown(wait=True)
