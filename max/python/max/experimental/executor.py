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

"""Executor protocol and concrete implementations for eager graph execution.

An :class:`Executor` receives a finalized :class:`~max.graph.Graph` and a
sequence of input :class:`~max.driver.Buffer` objects and returns output
buffers.  Callers must not rely on the graph module being unmodified after
``execute`` returns.

Selection order (highest to lowest priority):

1. Constructor injection into :class:`~max.experimental.realization_context.EagerRealizationContext`.
2. :func:`set_default_executor` / :func:`default_executor`.
3. ``MAX_EAGER_EXECUTOR`` environment variable (``jit`` | ``interpreter`` | ``compile``).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Protocol, runtime_checkable

from max import _core, driver, engine
from max._core.dialects import rmo
from max._interpreter import MOInterpreter
from max._mlir_context import MLIRThreadPoolExecutor
from max.graph import Graph

from .realization_context import _eager_model_cache_key, _session
from .support import SetterContext


class UnsupportedGraphError(RuntimeError):
    """Raised by an executor when it refuses to execute a graph.

    A composite executor (e.g. :class:`JitExecutor`) may catch this to route
    the graph elsewhere **before** execution starts.  Once execution has
    begun, exceptions are never masked as ``UnsupportedGraphError``.
    """


@runtime_checkable
class Executor(Protocol):
    """Contract for running a finalized eager graph against input buffers."""

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Execute *graph* with *inputs* and return output buffers.

        Args:
            graph: A finalized graph ready for execution.  The executor may
                mutate the module internally (e.g. apply lowering passes);
                callers must not use the module after this call returns.
            inputs: Buffers corresponding to ``graph.inputs``, in order.
                Mutable ``BufferType`` inputs are mutated in place; the
                returned sequence covers declared graph outputs only.

        Returns:
            Output buffers in the order declared by ``graph.output``.

        Raises:
            UnsupportedGraphError: If the executor statically refuses the
                graph before any execution begins.
        """
        ...


def _legalize(graph: Graph) -> None:
    """Applies ``LegalizeRMOOps`` to *graph* in place, at most once.

    Legalization declares graph parameters for data-dependent output dims,
    and redeclaring raises — but the same module can arrive here more than
    once (e.g. through both an executor and a context that legalized during
    finalization), so this is a no-op when no RMO ops remain.
    """
    if "rmo." not in graph._module.asm(assume_verified=True):
        return
    _core.lower(graph._module, [rmo.passes.LegalizeRMOOps()])


class InterpreterExecutor:
    """Executes a graph via :class:`~max._interpreter.MOInterpreter`.

    Raises :class:`UnsupportedGraphError` when
    :meth:`~max._interpreter.MOInterpreter.can_execute` refuses the graph
    (e.g. ``CustomOp`` present, unregistered op, or over the op-count
    threshold).  All runtime errors propagate unchanged — an explicit
    interpreter request is deliberately loud.
    """

    def __init__(self, max_ops: int | None = None) -> None:
        """Initializes the executor.

        Args:
            max_ops: Maximum number of dispatchable ops accepted.  Graphs
                with more ops raise :class:`UnsupportedGraphError`.  ``None``
                imposes no limit.
        """
        self._max_ops = max_ops
        self._interp = MOInterpreter()

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Executes *graph* via the MO interpreter.

        Raises:
            UnsupportedGraphError: If the interpreter cannot handle the graph.
        """
        _legalize(graph)
        if not self._interp.can_execute(graph, max_ops=self._max_ops):
            raise UnsupportedGraphError(
                "InterpreterExecutor: graph contains ops that require "
                "compilation (CustomOp, unregistered op, or over op-count "
                f"threshold {self._max_ops!r})."
            )
        return self._interp.execute(graph, inputs)


class CompilingExecutor:
    """Compiles and executes a graph synchronously via ``session.load``.

    No compilation cache of its own — caching is :class:`JitExecutor`'s job.
    Each call to :meth:`execute` recompiles the graph.
    """

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Compiles *graph* and executes it immediately."""
        _legalize(graph)
        model = _session().load(graph)
        return model(*inputs)


EagerCacheKey = tuple[str, tuple[tuple[str, str], ...]]

_INTERPRETER_MAX_OPS_ENV_VAR = "MAX_INTERPRETER_MAX_OPS"
_DEFAULT_INTERPRETER_MAX_OPS = 1024


class JitExecutor:
    """Executes via the interpreter while compiling in the background.

    The first call for a given graph (keyed by structure) submits a
    background compile and caches the resulting future for the life of this
    executor; while it is pending, calls execute via the interpreter.  When
    the interpreter refuses a graph, the call blocks until the compiled
    model is available.

    ``MAX_INTERPRETER_MAX_OPS`` caps the graph size the interpreter serves
    (default 1024 dispatchable ops).

    A compile failure propagates on every call that reaches the compiled
    path for that graph; it is not retried.
    """

    cache: dict[EagerCacheKey, Future[engine.Model]]
    lock: threading.Lock

    def __init__(self) -> None:
        self.cache = {}
        self.lock = threading.Lock()
        self.interpreter = InterpreterExecutor(
            max_ops=int(
                os.environ.get(
                    _INTERPRETER_MAX_OPS_ENV_VAR, _DEFAULT_INTERPRETER_MAX_OPS
                )
            )
        )
        self.pool = MLIRThreadPoolExecutor()

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Executes *graph*, compiling in the background on first call.

        Raises:
            UnsupportedGraphError: Never; graphs the interpreter refuses
                wait on the compiled model instead.
        """
        key = _eager_model_cache_key(graph)

        def compile(graph: Graph) -> engine.Model:
            _legalize(graph)
            return _session().load(graph)

        with self.lock:
            if key not in self.cache:
                # Compilation mutates the graph; compile a copy so the
                # interpreter path still sees the original module.
                future = self.pool.submit(compile, graph.copy())
                self.cache[key] = future
            future = self.cache[key]

        if future.done():
            if (exc := future.exception()) is not None:
                raise exc
            model = future.result()
            return model(*inputs)

        try:
            return self.interpreter.execute(graph, inputs)
        except UnsupportedGraphError:
            pass

        # JIT compile may be queued behind other graphs. Cancel it and
        # compile inline.
        with self.lock:
            future = self.cache[key]
            if future.cancel():
                model = compile(graph)
                done: Future[engine.Model] = Future()
                done.set_result(model)
                self.cache[key] = done
            else:
                model = future.result()
        return model(*inputs)


_MAX_EAGER_EXECUTOR_ENV_VAR = "MAX_EAGER_EXECUTOR"

_EXECUTORS: dict[str, Callable[[], Executor]] = {
    "jit": JitExecutor,
    "compile": CompilingExecutor,
    "interpreter": InterpreterExecutor,
}


def _executor_from_env() -> Executor:
    name = os.environ.get(_MAX_EAGER_EXECUTOR_ENV_VAR, "jit").lower().strip()
    if name not in _EXECUTORS:
        raise ValueError(
            f"{_MAX_EAGER_EXECUTOR_ENV_VAR}={name!r}: expected one of "
            f"{sorted(_EXECUTORS)}"
        )
    return _EXECUTORS[name]()


_DEFAULT_EXECUTOR: Executor = _executor_from_env()


def default_executor() -> Executor:
    """Returns the ambient default executor.

    The initial default is selected by the ``MAX_EAGER_EXECUTOR``
    environment variable (``jit`` | ``compile`` | ``interpreter``;
    default ``jit``), read at import time.
    """
    return _DEFAULT_EXECUTOR


def set_default_executor(
    executor: Executor,
) -> SetterContext[Executor]:
    """Sets the ambient default executor.

    The set takes effect immediately. The returned
    :class:`~max.experimental.support.SetterContext` may be used as a
    context manager to restore the previous executor on exit, or discarded
    to keep the new one.

    Args:
        executor: The new default executor.  All subsequently constructed
            :class:`~max.experimental.realization_context.EagerRealizationContext`
            instances that receive ``executor=None`` will use this executor.

    Returns:
        An undo handle restoring the previously installed executor.
    """

    def setter(executor: Executor) -> None:
        global _DEFAULT_EXECUTOR
        _DEFAULT_EXECUTOR = executor

    previous = default_executor()
    setter(executor)
    return SetterContext(executor, previous, setter)


__all__ = [
    "CompilingExecutor",
    "Executor",
    "InterpreterExecutor",
    "JitExecutor",
    "UnsupportedGraphError",
    "default_executor",
    "set_default_executor",
]
