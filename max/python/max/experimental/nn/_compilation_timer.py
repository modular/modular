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
"""Timer that logs graph build and compilation phases."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Generator
from types import TracebackType

# Logged under "max.pipelines" so existing pipeline log filters/handlers keep
# matching even when the timer is invoked from max.experimental.nn.Module.
logger = logging.getLogger("max.pipelines")


class CompilationTimer:
    """Timer for logging graph build and compilation phases.

    Use as a context manager. Starts timing on entry. Call
    ``mark_build_complete()`` after graph building; timings are logged on exit.

    Args:
        name: The name to use in log messages (e.g., "model", "vision model").

    Example:
        >>> with CompilationTimer("model") as timer:
        ...     graph = self._build_graph(self.weights, self.adapter)
        ...     timer.mark_build_complete()
        ...     model = session.load(graph, weights_registry=self.state_dict)
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._start_time: float | None = None
        self._build_end_time: float | None = None
        self._ctx: (
            contextlib.AbstractContextManager[CompilationTimer] | None
        ) = None

    def mark_build_complete(self) -> None:
        """Marks the end of the build phase and logs build time."""
        assert self._start_time is not None
        self._build_end_time = time.perf_counter()
        logger.info(
            f"Building {self.name} graph took "
            f"{self._build_end_time - self._start_time:.1f} seconds"
        )

    @contextlib.contextmanager
    def _run_timer(self) -> Generator[CompilationTimer, None, None]:
        self._start_time = time.perf_counter()
        self._build_end_time = None
        logger.info(f"Building and compiling {self.name}...")
        finish_event = threading.Event()
        reminder_thread = threading.Thread(
            target=self._reminder_thread_func,
            args=(finish_event,),
            daemon=True,
        )
        reminder_thread.start()
        try:
            yield self
        finally:
            end_time = time.perf_counter()
            finish_event.set()
            reminder_thread.join()
            if self._build_end_time is not None:
                logger.info(
                    f"Compiling {self.name} took "
                    f"{end_time - self._build_end_time:.1f} seconds"
                )
            logger.info(
                f"Building and compiling {self.name} took "
                f"{end_time - self._start_time:.1f} seconds"
            )
            self._start_time = None
            self._build_end_time = None

    def _reminder_thread_func(self, finish_event: threading.Event) -> None:
        assert self._start_time is not None
        while not finish_event.wait(timeout=60):
            if self._build_end_time is None:
                current_activity = "building"
                activity_start_time = self._start_time
            else:
                current_activity = "compiling"
                activity_start_time = self._build_end_time
            elapsed = time.perf_counter() - activity_start_time
            logger.info(
                f"Still {current_activity} {self.name} ({elapsed:.1f}s elapsed)"
            )

    def __enter__(self) -> CompilationTimer:
        if self._ctx is not None:
            raise RuntimeError(
                f"CompilationTimer({self.name!r}) is already active"
            )
        self._ctx = self._run_timer()
        return self._ctx.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._ctx is not None
        try:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._ctx = None
