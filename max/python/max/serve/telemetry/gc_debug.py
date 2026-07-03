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

"""Diagnostic instrumentation for CPython garbage-collection pauses.

This module exists to test the theory that the multi-second per-scheduler
iteration latency spikes (see MXSERV-152) are caused by CPython's
stop-the-world garbage collector running a generation-2 collection in the
middle of a scheduler iteration. A full ``gc`` pass holds the GIL for its
entire duration, which stalls every Python thread in the worker process --
including the scheduler thread and the host-function callback that feeds the
GPU overlap pipeline -- and therefore shows up as a giant
``batch_execution_time_ms`` spike.

When enabled (``MAX_SERVE_GC_DEBUG=1``), :func:`install_gc_debugger` attaches a
``gc.callbacks`` hook that times every collection and logs its duration,
generation, and object counts. Collections whose pause meets or exceeds
``MAX_SERVE_GC_DEBUG_MIN_DURATION_MS`` are logged at ``WARNING`` with a
structured ``event="gc_pause"`` payload so they can be overlaid against the
``event="batch_metrics"`` timeline in Datadog. If a 12 s pause lines up with a
12 s ``batch_execution_time_ms`` spike, the theory is confirmed.

The hook is a pure diagnostic: it does not change GC behavior (no
``gc.disable``/``gc.freeze``), so the timeline it records reflects production
behavior. It is a no-op unless explicitly enabled.
"""

from __future__ import annotations

import gc
import logging
import time
from collections import Counter

logger = logging.getLogger("max.serve")


class GCDebugger:
    """Times CPython GC collections and logs slow pauses.

    Registered as a ``gc.callbacks`` hook. CPython invokes the callback once
    with ``phase="start"`` immediately before a collection and once with
    ``phase="stop"`` immediately after, on the same thread that triggered the
    collection. Because a collection cannot be re-entrant (the GIL is held for
    its whole duration), a single start timestamp is sufficient to measure the
    pause.
    """

    def __init__(
        self, min_duration_ms: float = 50.0, top_objects: int = 0
    ) -> None:
        """Initializes the debugger.

        Args:
            min_duration_ms: Pauses at or above this threshold (in
                milliseconds) are logged at ``WARNING``; shorter pauses are
                logged at ``DEBUG``.
            top_objects: When greater than zero, log the ``top_objects`` most
                common live object types in the generation being collected.
                This walks every tracked object in that generation and is
                itself expensive, so leave it at ``0`` unless actively
                hunting for what is filling the heap.
        """
        self._min_duration_ms = min_duration_ms
        self._top_objects = top_objects
        self._start_s: float | None = None
        # Cumulative wall time spent in GC since install, for "what fraction of
        # uptime is GC" sanity checks.
        self._total_pause_s = 0.0
        self._num_collections = 0

    def __call__(self, phase: str, info: dict[str, int]) -> None:
        # Never let a diagnostic hook take down the worker.
        try:
            if phase == "start":
                self._start_s = time.monotonic()
                return

            if phase != "stop" or self._start_s is None:
                return

            elapsed_s = time.monotonic() - self._start_s
            self._start_s = None
            self._total_pause_s += elapsed_s
            self._num_collections += 1

            elapsed_ms = elapsed_s * 1000.0
            generation = info.get("generation", -1)
            collected = info.get("collected", -1)
            uncollectable = info.get("uncollectable", -1)
            # gc.get_count() = (gen0, gen1, gen2) allocation counters after the
            # collection; len(gc.get_objects()) is the live tracked-object
            # count, which is expected to grow with uptime if the spike scales
            # with uptime.
            gen_counts = gc.get_count()
            num_tracked = len(gc.get_objects())

            extra: dict[str, object] = {
                "event": "gc_pause",
                "gc_generation": generation,
                "gc_pause_ms": elapsed_ms,
                "gc_collected": collected,
                "gc_uncollectable": uncollectable,
                "gc_tracked_objects": num_tracked,
                "gc_count_gen0": gen_counts[0],
                "gc_count_gen1": gen_counts[1],
                "gc_count_gen2": gen_counts[2],
                "gc_total_pause_ms": self._total_pause_s * 1000.0,
                "gc_num_collections": self._num_collections,
            }

            top_str = ""
            if self._top_objects > 0:
                top = self._top_object_types(generation)
                extra["gc_top_objects"] = top
                top_str = f" top_objects={top}"

            message = (
                f"GC pause: {elapsed_ms:.1f}ms gen={generation} "
                f"collected={collected} uncollectable={uncollectable} "
                f"tracked_objects={num_tracked}{top_str}"
            )

            if elapsed_ms >= self._min_duration_ms:
                logger.warning(message, extra=extra)
            else:
                logger.debug(message, extra=extra)
        except Exception:
            logger.debug("GC debug callback failed", exc_info=True)

    def _top_object_types(self, generation: int) -> dict[str, int]:
        """Returns a histogram of the most common live object types.

        Walks every object tracked in ``generation`` and tallies type names.
        Expensive; only called when ``top_objects > 0``.
        """
        try:
            objects = gc.get_objects(generation=generation)
        except (TypeError, ValueError):
            # generation == -1 (unknown) or out of range: fall back to all.
            objects = gc.get_objects()
        counter: Counter[str] = Counter(
            type(obj).__qualname__ for obj in objects
        )
        return dict(counter.most_common(self._top_objects))


def install_gc_debugger(
    *, enabled: bool, min_duration_ms: float = 50.0, top_objects: int = 0
) -> GCDebugger | None:
    """Installs the GC pause debugger if ``enabled``.

    Intended to be called exactly once per worker process, after logging has
    been configured. Safe to call when disabled (returns ``None``).

    Args:
        enabled: Whether to attach the GC callback at all.
        min_duration_ms: Threshold above which pauses are logged at
            ``WARNING`` instead of ``DEBUG``.
        top_objects: Number of top live object types to log per collection
            (``0`` disables the expensive heap walk).

    Returns:
        The installed :class:`GCDebugger`, or ``None`` if disabled.
    """
    if not enabled:
        return None

    debugger = GCDebugger(
        min_duration_ms=min_duration_ms, top_objects=top_objects
    )
    gc.callbacks.append(debugger)
    thresholds = gc.get_threshold()
    counts = gc.get_count()
    logger.info(
        (
            "GC debug instrumentation enabled "
            "(min_duration_ms=%.1f, top_objects=%d, gc_enabled=%s, "
            "thresholds=%s, counts=%s, tracked_objects=%d)"
        ),
        min_duration_ms,
        top_objects,
        gc.isenabled(),
        thresholds,
        counts,
        len(gc.get_objects()),
        extra={
            "event": "gc_debug_install",
            "gc_min_duration_ms": min_duration_ms,
            "gc_top_objects": top_objects,
            "gc_enabled": gc.isenabled(),
            "gc_threshold_gen0": thresholds[0],
            "gc_threshold_gen1": thresholds[1],
            "gc_threshold_gen2": thresholds[2],
        },
    )
    return debugger
