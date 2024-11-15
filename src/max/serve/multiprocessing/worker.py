# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import concurrent.futures
import logging
import multiprocessing as mp
import os
import threading
from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from multiprocessing.synchronize import Event as MPEvent
from typing import Any, Callable, Union

from faster_fifo import Queue


@dataclass
class MPQueue:
    """Inter-process queue backed by a shared memory FIFO queue."""

    key: str
    queue: Queue


@lru_cache
def running_workers() -> dict[str, Any]:
    """Map of all currently running workers."""
    return {}


@dataclass
class Worker(ABC):
    """Base class for thread/process workers."""

    name: str
    task: Union[asyncio.Task, None] = None
    executor: Union[concurrent.futures.Executor, None] = None

    logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def shutdown(self):
        ...

    # Async lifecycle events.

    async def started(self):
        ...

    async def stopped(self):
        ...


@dataclass
class ProcessPoolWorker(Worker):
    """Concrete worker backed by a process pool."""

    max_workers: Union[int, None] = 1
    queues: dict[str, MPQueue] = field(default_factory=dict)

    started_event: MPEvent = field(default_factory=mp.Event)
    stopped_event: MPEvent = field(default_factory=mp.Event)
    shutdown_event: MPEvent = field(default_factory=mp.Event)

    def __post_init__(self):
        super().__post_init__()
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=init_pool_processes,
            initargs=(
                {
                    "STARTED": self.started_event,
                    "STOPPED": self.stopped_event,
                    "SHUTDOWN": self.shutdown_event,
                },
                self.queues,
            ),
        )

    def shutdown(self):
        self.shutdown_event.set()
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)

    async def started(self):
        while not self.started_event.is_set():
            await asyncio.sleep(0)

    async def stopped(self):
        while not self.stopped_event.is_set():
            await asyncio.sleep(0)


# Inter-process communication.

ALL_EVENTS: dict[str, MPEvent] = {}
ALL_QUEUES: dict[str, MPQueue] = {}

Predicate = Callable[[Any], bool]


def init_pool_processes(
    events: dict[str, MPEvent],
    queues: dict[str, MPQueue],
):
    """Called to initialize all process workers.

    Args:
        events (dict[str, MPEvent]): Global events.
        queues (dict[str, MPQueue]): Global queues.
    """
    global ALL_QUEUES, ALL_EVENTS
    ALL_EVENTS |= events
    ALL_QUEUES |= queues

    def exit_if_orphaned():
        parent = mp.parent_process()
        if parent:
            parent.join()
            os._exit(-1)

    threading.Thread(target=exit_if_orphaned, daemon=True).start()


# Filter/registration utilities for global multiprocessing resources.


def filter(map: dict[str, Any], pred: Union[Predicate, None]) -> dict[str, Any]:
    if not pred:
        return map

    return {k: v for k, v in map.items() if pred(v)}


def all_events(pred: Union[Predicate, None] = None) -> dict[str, MPEvent]:
    return filter(ALL_EVENTS, pred)


def all_queues(pred: Union[Predicate, None] = None) -> dict[str, MPQueue]:
    return filter(ALL_QUEUES, pred)


def register_mp_queue(key: str, queue: Queue = None):
    if not queue:
        queue = Queue(max_size_bytes=10_000_000)

    q = MPQueue(key, queue)
    ALL_QUEUES[key] = q
    return q


def close_all_mp_queues():
    for q in all_queues().values():
        q.queue.close()
