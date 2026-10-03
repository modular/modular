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
"""Captures the GPU kernels of the forward passes traced requests run in."""

from __future__ import annotations

import dataclasses
import logging
import os
import queue
import tempfile
import threading
import time
from collections.abc import Sequence

from max._core.profiler import (
    load_profiler_plugin,
    start_capture,
    stop_capture,
)
from max.pipelines.context import TextContext
from max.pipelines.request import RequestID
from max.serve.config import KernelTraceLevel, Settings
from max.serve.telemetry import common as telemetry
from max.serve.telemetry._trace_context import (
    TRACE_LEVEL_HEADER,
    RequestTraceLevel,
)
from opentelemetry.trace import Span, SpanContext

_kernel_capture_permitted = False

# Bounds the passes one capture records, traced or not, since overlapping
# traced requests would otherwise keep it armed, and its buffers growing,
# indefinitely.
# TODO(MXTOOLS-651): The replay's 20,000-span cap and 64 MiB parse limit set
# this; raise it once the replay works in chunks.
_MAX_CAPTURE_PASSES = 64


def kernel_capture_permitted() -> bool:
    """Returns whether a request's ``x-max-trace-level`` header may arm a
    kernel capture in this model worker, as decided by
    :func:`configure_kernel_capture`."""
    return _kernel_capture_permitted


def _permit_kernel_capture(level: KernelTraceLevel) -> bool:
    """Loads the profiler plugin for per-request captures, or refuses them."""
    logger = logging.getLogger("max.serve")
    if level == KernelTraceLevel.KERNEL:
        # The session-wide capture holds the profiler for the whole run, so
        # no per-request capture could start.
        logger.warning(
            "Ignoring MAX_SERVE_KERNEL_TRACE_HEADERS: "
            "MAX_SERVE_KERNEL_TRACE_LEVEL=kernel already captures every pass."
        )
        return False
    if not load_profiler_plugin():
        logger.warning(
            "Ignoring MAX_SERVE_KERNEL_TRACE_HEADERS: no profiler is available."
        )
        return False
    return True


def configure_kernel_capture(settings: Settings) -> None:
    """Decides :func:`kernel_capture_permitted`, loading the profiler plugin
    when it is permitted.

    Must be called in the model worker after ``configure_tracing``, and
    before any device is created: on AMD, a plugin loaded later can't record.

    Args:
        settings: Server settings carrying ``kernel_trace_level``.
    """
    global _kernel_capture_permitted
    _kernel_capture_permitted = (
        telemetry._trace_level_header_enabled
        and _permit_kernel_capture(settings.kernel_trace_level)
    )


@dataclasses.dataclass(frozen=True)
class TracedPass:
    """A forward pass that ran under an armed kernel capture with at least one
    traced request."""

    batch_id: int
    level: RequestTraceLevel
    """The highest level among the pass's traced requests."""
    batch_span_context: SpanContext
    """The pass's ``max.batch`` span."""
    host_start_ns: int
    """``time.time_ns()`` taken before the pass's ``max.batch`` span started,
    a few microseconds before its ``max.batch`` range began."""
    host_end_ns: int
    """``time.time_ns()`` just after the range ended."""


_capture_dir: str | None = None


def kernel_capture_path() -> str:
    """Returns where this worker saves a per-request kernel capture: a fixed
    name in a private directory made on first use. Each capture overwrites
    the last, so the latest stays on disk for inspection."""
    global _capture_dir
    if _capture_dir is None:
        # The plugin's write follows symlinks, so a predictable name in a
        # shared temp directory would let another user aim it at any file
        # this worker can write.
        _capture_dir = tempfile.mkdtemp(prefix="max-kernel-capture-")
    return os.path.join(_capture_dir, "kernel-capture.json")


def _request_trace_level(context: TextContext) -> RequestTraceLevel | None:
    """Returns the level a request asked for with ``x-max-trace-level``."""
    if context.trace_carrier is None:
        return None
    value = context.trace_carrier.get(TRACE_LEVEL_HEADER)
    level = RequestTraceLevel.parse(value) if value else None
    return (
        level if level is not None and level > RequestTraceLevel.OFF else None
    )


class KernelCapture:
    """Captures the forward passes that traced requests run in.

    One capture is armed at the first pass with a traced request, shared by
    the traced requests that run while it records, and stopped when the last
    of them leaves or at the pass cap. The scheduler calls it from its own
    thread only.

    Args:
        max_passes: The most passes one capture records, traced or not.
    """

    def __init__(self, max_passes: int = _MAX_CAPTURE_PASSES) -> None:
        self._max_passes = max_passes
        self._traced: dict[RequestID, RequestTraceLevel] = {}
        self._traced_passes: list[TracedPass] = []
        self._passes = 0
        # Traced requests that ran in the armed capture.
        self._members: set[RequestID] = set()
        self._armed = False
        self._thread: KernelCaptureThread | None = None
        self._warned_not_started = False

    def admit(self, context: TextContext) -> bool:
        """Traces a new request if its ``x-max-trace-level`` header asks.

        Args:
            context: The request, as the scheduler admits it.

        Returns:
            Whether the request is traced.
        """
        level = _request_trace_level(context)
        if level is None:
            return False
        self._traced[context.request_id] = level
        return True

    @property
    def may_record(self) -> bool:
        """Whether a capture may be recording, so every pass needs a
        ``max.batch`` range for the replay to tell traced passes' GPU work
        from untraced ones'."""
        return self._armed or (self._thread is not None and self._thread.busy)

    def begin_pass(
        self, batches: Sequence[Sequence[TextContext]]
    ) -> RequestTraceLevel | None:
        """Arms the capture for a pass with traced members.

        Args:
            batches: The pass's batches, one per replica.

        Returns:
            The pass's level, or None when it has no traced member, the
            previous capture is still being saved, or another capture holds
            the profiler.
        """
        if not self._traced:
            return None
        members = [
            ctx.request_id
            for batch in batches
            for ctx in batch
            if ctx.request_id in self._traced
        ]
        if not members:
            return None
        if not self._armed:
            if self._thread is None:
                self._thread = KernelCaptureThread()
            elif self._thread.busy:
                return None
            if not start_capture():
                if not self._warned_not_started:
                    self._warned_not_started = True
                    logging.getLogger("max.serve").warning(
                        "Not capturing kernels for traced requests while "
                        "another capture or a Dynolog trace holds the "
                        "profiler."
                    )
                return None
            self._armed = True
        self._members.update(members)
        return max(self._traced[r] for r in members)

    def end_pass(
        self,
        batch_id: int,
        level: RequestTraceLevel | None,
        batch_span: Span,
        host_start_ns: int,
    ) -> None:
        """Records a pass that ran while the capture was armed, stopping the
        capture at the pass cap.

        Args:
            batch_id: The pass's batch ID.
            level: What :meth:`begin_pass` returned for the pass.
            batch_span: The pass's ``max.batch`` span.
            host_start_ns: ``time.time_ns()`` taken before the pass's
                ``max.batch`` span started, a few microseconds before its
                ``max.batch`` range began.
        """
        if not self._armed:
            return
        if level is not None:
            self._traced_passes.append(
                TracedPass(
                    batch_id=batch_id,
                    level=level,
                    batch_span_context=batch_span.get_span_context(),
                    host_start_ns=host_start_ns,
                    host_end_ns=time.time_ns(),
                )
            )
        self._passes += 1
        if self._passes >= self._max_passes:
            logging.getLogger("max.serve").warning(
                "Stopping a kernel capture at %d passes; its requests are no "
                "longer traced.",
                self._max_passes,
            )
            for request_id in self._members:
                self._traced.pop(request_id, None)
            self._stop()

    def untrace(self, request_id: RequestID) -> None:
        """Untraces a request that has left, stopping the capture if it was
        the last traced one."""
        if (
            self._traced.pop(request_id, None) is None
            or self._traced
            or not self._armed
        ):
            return
        self._stop()

    def _stop(self) -> None:
        assert self._thread is not None
        self._armed = False
        self._passes = 0
        self._members.clear()
        passes, self._traced_passes = self._traced_passes, []
        self._thread.submit(kernel_capture_path(), passes)


class KernelCaptureThread:
    """Stops and saves per-request kernel captures off the scheduler thread.

    Saving holds the profiler's lock, so a start issued meanwhile would block
    the scheduler: :class:`KernelCapture` checks :attr:`busy` and defers
    re-arming instead.
    """

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue[tuple[str, list[TracedPass]]] = (
            queue.SimpleQueue()
        )
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(
            target=self._run, name="max-kernel-capture", daemon=True
        ).start()

    @property
    def busy(self) -> bool:
        """Whether a submitted capture is still being stopped or saved."""
        return not self._idle.is_set()

    def submit(self, output_path: str, passes: list[TracedPass]) -> None:
        """Queues a stop of the armed capture, saving it to ``output_path``."""
        self._idle.clear()
        self._jobs.put((output_path, passes))

    def _run(self) -> None:
        logger = logging.getLogger("max.serve")
        while True:
            output_path, passes = self._jobs.get()
            try:
                error = stop_capture(output_path)
                if error:
                    logger.warning("Kernel capture failed: %s", error)
                else:
                    logger.info(
                        "Saved a kernel capture of %d traced passes to %s",
                        len(passes),
                        output_path,
                    )
            except Exception:
                logger.exception("Kernel capture failed")
            finally:
                self._idle.set()
