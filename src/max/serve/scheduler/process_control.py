# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import asyncio
import ctypes
import logging
import math
import multiprocessing
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


class ProcessControl:
    """Manage signals between process & process creator

    Parent -> Process communication:
        - canceled: The creator politely signals to the process that it
          should finish its work and exit. It is expected that the process
          creator will use increasingly heavy handed tactics to stop the
          process if the process does not stop of its own accord.

    Process -> Parent communication:
        - heartbeat: Process emits a signal demonstrating it is still making
          progress.  This is the last timestamp at which the process work
          working correctly. All the usual advice around trusing clocks
          applies. Make sure that your health_fail_s is longer than heartbeat
          period.

        - started: Process has begun work

        - completed: Process has completed its work and has stopped.  This is
          intended to be used to support graceful shutdown.

    The basic pattern for the target of a process is as follows:

    def run(pc: process_control.ProcessControl):
        pc.set_started()
        while True:
            pc.beat()
            if pc.is_canceled():
                break
        pc.set_completed()

    In addition to the explicit signalling between parent & process, the
    process itself can be alive or dead. It is useful to list the possible
    combinations of singal and process-liveness:
        alive: Is the process running?
        started: Has user code signaled that it has started?
        completed: Has user code signaled that it is completed?
        N N N - initial state
        N N Y - should never happen
        N Y N - process started, but is now dead
        N Y Y - process started, completed, and is now dead
        Y N N - proces started, but no user code signaled anything
        Y N Y - should never happen
        Y Y N - process is actively working
        Y Y Y - process has completed, is still alive, but *ought* exit soon
    """

    def __init__(
        self,
        ctx: multiprocessing.context.BaseContext,
        name: str,
        health_fail_s: float = 10,
    ):
        self.ctx = ctx
        self.name = name
        self._started = ctx.Event()
        self._completed = ctx.Event()
        self._canceled = ctx.Event()

        self._last_beat = ctx.Value(ctypes.c_int64)
        self._last_beat.value = 0  # make sure this initializes to 0
        self.health_fail_ns: int = int(health_fail_s * 1e9)

    def beat(self) -> None:
        self._last_beat.value = time.time_ns()

    def is_healthy(self) -> bool:
        dt = time.time_ns() - self._last_beat.value
        return dt < self.health_fail_ns

    def is_unhealthy(self) -> bool:
        return not self.is_healthy()

    def set_canceled(self) -> None:
        self._canceled.set()

    def is_canceled(self) -> bool:
        return self._canceled.is_set()

    def set_started(self) -> None:
        self._started.set()

    def is_started(self) -> bool:
        return self._started.is_set()

    def set_completed(self) -> None:
        self._completed.set()

    def is_completed(self) -> bool:
        return self._completed.is_set()


def forever() -> Iterable:
    while True:
        yield


@dataclass
class ProcessMonitor:
    """Utility functions for observing & stopping processes

    ProcessMonitor can wait for a process to enter a number of states. For
    startup/shutdown operations, we expect the process to enter these states
    within a finite, reasonable amount of time.  Awaiting these states are
    governed by `poll_s` & `max_time_s`. These states are:
        - until_started
        - until_completed
        - until_dead
        - until_health

    Other states are expected to happen in exceptional circumstances & required
    indefinite polling. Awaiting these states is governed by `unhealthy_poll_s`
    and `unhealthy_max_time_s`. These states are:
        - until_unhealthy

    """

    pc: ProcessControl
    proc: multiprocessing.process.BaseProcess

    poll_s: float = 200e-3
    max_time_s: Optional[float] = 10.0
    unhealthy_poll_s: float = 200e-3
    unhealthy_max_time_s: Optional[float] = None

    async def until_started(self) -> bool:
        return await _until_true(
            self.pc.is_started, self.poll_s, self.max_time_s
        )

    async def until_completed(self) -> bool:
        return await _until_true(
            self.pc.is_completed, self.poll_s, self.max_time_s
        )

    async def until_dead(self) -> bool:
        is_dead = lambda: not self.proc.is_alive()
        return await _until_true(is_dead, self.poll_s, self.max_time_s)

    async def until_healthy(self) -> bool:
        return await _until_true(
            self.pc.is_healthy,
            self.poll_s,
            self.max_time_s,
        )

    async def until_unhealthy(self) -> bool:
        return await _until_true(
            self.pc.is_unhealthy,
            self.unhealthy_poll_s,
            self.unhealthy_max_time_s,
        )

    async def shutdown(self):
        logger.info("Shutting down")
        logger.info("Canceling worker")
        self.pc.set_canceled()
        logger.info("Waiting for worker to complete")
        completed = await self.until_completed()
        logger.info(f"Completed {completed}")
        if self.proc.is_alive():
            logger.info("Process is still alive.  Killing")
            self.proc.kill()
            logger.info("Waiting to die")
            dead = await self.until_dead()
            logger.info(f"Dead? {dead}")
        logger.info("Shut down")

    async def shutdown_if_unhealthy(
        self, cb: Optional[Callable[[], None]] = None
    ):
        try:
            await self.until_unhealthy()
        except:
            logger.exception(
                f"Error while checking process health: {self.pc.name}"
            )
        finally:
            logger.info(f"Exiting health check task: {self.pc.name}")
            if cb is not None:
                try:
                    cb()
                except:
                    pass
            await self.shutdown()


async def _until_true(
    is_done: Callable[[], bool], poll_s: float, max_time_s: Optional[float]
) -> bool:
    """Poll a predicate until it is true or you exceed 'max_time_s'"""
    steps: Iterable
    if max_time_s is None:
        steps = forever()
    else:
        steps = range(math.ceil(max_time_s / poll_s))
    for _ in steps:
        if is_done():
            return True
        await asyncio.sleep(poll_s)
    return False
