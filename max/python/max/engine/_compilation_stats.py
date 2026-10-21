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
"""Plumbing for tracking compile/init phase timings inside the engine.

The engine instruments ``InferenceSession.compile()`` and
``InferenceSession.init_all()`` to push elapsed wall-clock time into any
``CompilationStats`` accumulators active on the contextvar stack. The public
collector context manager and the high-level logging timer live in
``max.experimental.nn._compilation_timer``; this module hosts the data and
the engine-side recording helpers so the engine API can record phase
timings without pulling in higher-level dependencies.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Generator
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class CompilationStats:
    """Aggregate timings collected during model compilation and initialization.

    ``build_seconds`` is the Python-side graph construction time, supplied by
    the :class:`CompilationTimer` that wraps a build region.
    ``compile_seconds`` and ``init_seconds`` are the engine-measured durations
    of :meth:`InferenceSession.compile` and :meth:`InferenceSession.init_all`
    respectively, and include all calls in the region — whether they
    happened inside a named ``CompilationTimer`` or not.
    ``labeled_compile_seconds`` and ``labeled_init_seconds`` track only the
    portion that happened inside a ``CompilationTimer``; the difference
    between the totals and these labeled subtotals is "unaccounted" compile
    or init time — a useful signal that some ``session.load`` /
    ``session.compile`` site is firing without being wrapped in a timer.
    ``num_phases`` counts how many ``CompilationTimer`` regions completed
    inside the collector.
    """

    build_seconds: float = 0.0
    compile_seconds: float = 0.0
    init_seconds: float = 0.0
    labeled_compile_seconds: float = 0.0
    labeled_init_seconds: float = 0.0
    num_phases: int = 0


# Stack of active accumulators. Engine phase recording pushes to every entry,
# so a per-CompilationTimer local stats and an outer collect_compilation_stats
# can both observe the same compile/init events.
_active_stats: ContextVar[tuple[CompilationStats, ...]] = ContextVar(
    "compilation_stats_stack", default=()
)


@contextlib.contextmanager
def collect_compilation_stats() -> Generator[CompilationStats, None, None]:
    """Collects build, compile, and init times in this region.

    Every ``InferenceSession.compile()`` and ``InferenceSession.init_all()``
    call that runs inside this context adds its elapsed wall-clock time to
    the yielded stats. ``CompilationTimer`` regions inside this context also
    push their graph-build time to ``build_seconds``. Nesting is supported:
    every active accumulator observes every recorded phase.
    """
    stats = CompilationStats()
    token = _active_stats.set(_active_stats.get() + (stats,))
    try:
        yield stats
    finally:
        _active_stats.reset(token)


@contextlib.contextmanager
def _record_phase(field: str) -> Generator[None, None, None]:
    """Adds elapsed wall-clock time to ``field`` on every active stats.

    Used by the engine to record ``compile_seconds`` and ``init_seconds``
    tightly around the underlying C++ calls.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        for stats in _active_stats.get():
            setattr(stats, field, getattr(stats, field) + elapsed)
