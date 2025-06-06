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
import ctypes
import logging
import math
import multiprocessing
import multiprocessing.synchronize
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Union

logger = logging.getLogger(__name__)
# This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
logger.propagate = False


class EventCreator(Protocol):
    """Event Creator is intended to be compatible with
    multiprocessing.get_context() and the threading module.
    """

    def Event(
        self,
    ) -> Union[threading.Event, multiprocessing.synchronize.Event]:
        pass


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
    combinations of signal and process-liveness:
        alive: Is the process running?
        started: Has user code signaled that it has started?
        completed: Has user code signaled that it is completed?
        N N N - initial state
        N N Y - should never happen
        N Y N - process started, but is now dead
        N Y Y - process started, completed, and is now dead
        Y N N - process started, but no user code signaled anything
        Y N Y - should never happen
        Y Y N - process is actively working
        Y Y Y - process has completed, is still alive, but *ought* exit soon

    Basic process monitoring can be accomplished by polling is_started() and
    is_canceled().  It can also be accomplished by waiting on started_event and
    canceled_event directly. `pc.started_event.wait()`
    """

    def __init__(
        self,
        ctx: EventCreator,
        name: str,
        # TODO: we temporarily set it to 1 minute to handle long context input
        health_fail_s: float = 60.0,
    ):
        self.name = name
        self.started_event = ctx.Event()
        self.completed_event = ctx.Event()
        self.canceled_event = ctx.Event()

        self._last_beat: Union[
            multiprocessing.sharedctypes.Synchronized[int], ctypes.c_int64
        ]
        # Support both threading and multiprocessing contexts
        if hasattr(ctx, "Value"):
            self._last_beat = ctx.Value(ctypes.c_int64)
        else:
            self._last_beat = ctypes.c_int64()

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
        self.canceled_event.set()

    def is_canceled(self) -> bool:
        return self.canceled_event.is_set()

    def set_started(self) -> None:
        self.started_event.set()

    def is_started(self) -> bool:
        return self.started_event.is_set()

    def set_completed(self) -> None:
        self.completed_event.set()

    def is_completed(self) -> bool:
        return self.completed_event.is_set()


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

    async def until_dead_no_timeout(self) -> bool:
        is_dead = lambda: not self.proc.is_alive()
        return await _until_true(
            is_dead,
            self.unhealthy_poll_s,
            None,
        )

    async def until_unhealthy(self) -> bool:
        return await _until_true(
            self.pc.is_unhealthy,
            self.unhealthy_poll_s,
            self.unhealthy_max_time_s,
        )

    async def shutdown(self):
        logger.info("Shutting down")
        self.pc.set_canceled()
        if not self.proc.is_alive():
            logger.info(
                f"Early exit. Process was already dead. exitcode: {self.proc.exitcode}"
            )
            return

        loop = asyncio.get_running_loop()
        completed_task = loop.create_task(self.until_completed())
        dead_task = loop.create_task(self.until_dead())

        completed_tasks, pending_tasks = await asyncio.wait(
            [completed_task, dead_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if completed_task.done():
            await self.until_dead()

        # we have waited a polite amount of time.  Time to close things out.
        completed_task.cancel()
        dead_task.cancel()

        if self.proc.is_alive():
            logger.info("Process is still alive.  Killing")
            self.proc.kill()
            dead = await self.until_dead()
        logger.info("Shut down")

    async def shutdown_if_unhealthy(
        self, cb: Optional[Callable[[], None]] = None
    ):
        try:
            await self.until_unhealthy()
        except asyncio.CancelledError:
            # Cancellation happens when winding down a completed program
            # Nothing interesting to see here.
            pass
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

    async def shutdown_if_dead(self, cb: Optional[Callable[[], None]] = None):
        try:
            await self.until_dead_no_timeout()
        except asyncio.CancelledError:
            # Cancellation happens when winding down a completed program
            # Nothing interesting to see here.
            pass
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
