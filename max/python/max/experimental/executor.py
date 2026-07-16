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
3. ``MAX_EAGER_EXECUTOR`` environment variable (``composite`` | ``jit`` | ``interpreter`` | ``compile``; default ``composite``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from max import _core, driver, engine
from max._core.dialects import rmo
from max._core.mlrt import AsyncValue
from max.graph import Graph

from .support import SetterContext, _session


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


EagerCacheKey = tuple[str, tuple[tuple[str, str], ...]]

_INTERPRETER_MAX_OPS_ENV_VAR = "MAX_INTERPRETER_MAX_OPS"
_DEFAULT_INTERPRETER_MAX_OPS = 1024


def _interpreter_max_ops() -> int:
    """Returns the dispatchable-op threshold below which the interpreter runs.

    Reads ``MAX_INTERPRETER_MAX_OPS`` from the environment.  Graphs with more
    dispatchable ops than this value are compiled so the graph compiler can
    fuse across ops.

    Returns:
        The op-count threshold (default 1024).
    """
    raw = os.environ.get(_INTERPRETER_MAX_OPS_ENV_VAR, "")
    if raw.strip().isdigit():
        return int(raw.strip())
    return _DEFAULT_INTERPRETER_MAX_OPS


def _eager_model_cache_key(graph: Graph) -> EagerCacheKey:
    """Builds a compact, stable cache key for a finalized eager graph.

    Uses a SHA-256 hash of the MLIR module ASM (with debug info stripped)
    combined with the resolved kernel library paths and SHA-256 hashes of
    their contents.  Hashing file contents (rather than ``st_mtime``)
    avoids a time-of-check/time-of-use race and produces a deterministic
    key regardless of filesystem timestamp granularity.

    Args:
        graph: A finalized graph ready for compilation.

    Returns:
        A tuple of ``(asm_hex_digest, ((resolved_path, content_hash), ...))``.
    """
    module_asm = graph._module.asm(
        assume_verified=True,
        enable_debug_info=False,
        pretty_debug_info=False,
        use_local_scope=True,
    )
    asm_hash = hashlib.sha256(module_asm.encode()).hexdigest()
    kernel_paths = tuple(
        (
            str(Path(p).resolve()),
            hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        )
        for p in graph.kernel_libraries_paths
    )
    return (asm_hash, kernel_paths)


# Bounds memory: each entry pins an engine.Model + its MEF buffer.
_EAGER_MODEL_CACHE_MAX_SIZE = int(
    os.environ.get("MAX_EAGER_MODEL_CACHE_SIZE", "128")
)
_EAGER_MODEL_CACHE_LOCK = threading.Lock()
_EAGER_MODEL_CACHE: OrderedDict[EagerCacheKey, engine.Model] = OrderedDict()
_EAGER_MODEL_CACHE_SESSION: engine.api.InferenceSession | None = None


def _load_eager_model(graph: Graph) -> engine.Model:
    """Loads or retrieves a cached compiled model for an eager graph.

    The cache is load-bearing even though the C++ MEF cache exists below
    it: ~74% of a MEF hit is spent bytecode-serializing seeded MOGG
    kernel decls into the C++ cache key (``FrameworkFrontend.cpp:518``).
    A Python hit here skips the whole ``session.load`` roundtrip.

    Returns:
        A compiled ``engine.Model`` ready for execution.
    """
    global _EAGER_MODEL_CACHE_SESSION

    session = _session()
    key = _eager_model_cache_key(graph)

    with _EAGER_MODEL_CACHE_LOCK:
        if _EAGER_MODEL_CACHE_SESSION is not session:
            _EAGER_MODEL_CACHE.clear()
            _EAGER_MODEL_CACHE_SESSION = session

        cached = _EAGER_MODEL_CACHE.get(key)
        if cached:
            _EAGER_MODEL_CACHE.move_to_end(key)
            return cached

    model = session.load(graph)

    with _EAGER_MODEL_CACHE_LOCK:
        if _EAGER_MODEL_CACHE_SESSION is session:
            _EAGER_MODEL_CACHE[key] = model
            if len(_EAGER_MODEL_CACHE) > _EAGER_MODEL_CACHE_MAX_SIZE:
                _EAGER_MODEL_CACHE.popitem(last=False)

    return model


class InterpreterExecutor:
    """Executes a graph via :func:`max._interpreter.execute`.

    Raises :class:`UnsupportedGraphError` when
    :func:`max._interpreter.can_execute` refuses the graph
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

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Executes *graph* via the MO interpreter.

        Raises:
            UnsupportedGraphError: If the interpreter cannot handle the graph.
        """
        # Defer import to avoid compiling interpreter bindings unless
        # the interpreter is actually needed.
        from max import _interpreter

        # The interpreter only supports MO-level graphs, so lower RMO -> MO.
        _core.lower(graph._module, [rmo.passes.LegalizeRMOOps()])
        if not _interpreter.can_execute(graph, max_ops=self._max_ops):
            raise UnsupportedGraphError(
                "InterpreterExecutor: graph contains ops that require "
                "compilation (CustomOp, unregistered op, or over op-count "
                f"threshold {self._max_ops!r})."
            )
        return _interpreter.execute(graph, inputs)


class CompilingExecutor:
    """Compiles and executes a graph synchronously via ``session.load``.

    No compilation cache of its own — caching is :class:`JitExecutor`'s job.
    Each call to :meth:`execute` recompiles the graph.
    """

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Compiles *graph* and executes it immediately."""
        model = _session().load(graph)
        return model(*inputs)


class CompositeExecutor:
    """Interpreter for small graphs, cached synchronous compile otherwise.

    Attempts the interpreter (gated by an op-count threshold) and falls back to
    a cached synchronous compile when it refuses the graph.
    ``fallback_on_error`` governs interpreter errors raised *after* it accepts a
    graph: when True the graph is compiled instead, when False the error
    propagates.  ``interpreter=None`` disables the interpreter entirely, making
    this a cached compile-only executor.
    """

    def __init__(
        self,
        *,
        interpreter: InterpreterExecutor | None,
        fallback_on_error: bool,
    ) -> None:
        """Initializes the executor.

        Args:
            interpreter: The interpreter to try first, or ``None`` to always
                compile.
            fallback_on_error: Whether to fall back to compilation when the
                interpreter raises a non-:class:`UnsupportedGraphError` error.
        """
        self._interpreter = interpreter
        self._fallback_on_error = fallback_on_error

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Executes *graph* via the interpreter, falling back to compilation.

        Raises:
            UnsupportedGraphError: Never; graphs the interpreter refuses are
                compiled instead.
        """
        if self._interpreter is not None:
            try:
                return self._interpreter.execute(graph, inputs)
            except UnsupportedGraphError:
                pass
            except Exception:
                if not self._fallback_on_error:
                    raise
                logging.getLogger("max.experimental").debug(
                    "Interpreter failed, falling back to graph compiler",
                    exc_info=True,
                )
        return _load_eager_model(graph)(*inputs)


class JitExecutor:
    """Executes via the interpreter while compiling in the background.

    The first call for a given graph (keyed by structure) starts an
    asynchronous compile on the runtime's worker pool and caches the
    handle for the life of this executor; while the compile is pending,
    calls execute via the interpreter.  When the interpreter refuses a
    graph, the call waits for that graph's compile.

    ``MAX_INTERPRETER_MAX_OPS`` caps the graph size the interpreter serves
    (default 1024 dispatchable ops).

    A compile failure propagates on every call that reaches the compiled
    path for that graph; it is not retried.
    """

    cache: dict[EagerCacheKey, AsyncValue[engine.Model]]
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

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Executes *graph*, compiling in the background on first call.

        Raises:
            UnsupportedGraphError: Never; graphs the interpreter refuses
                wait on the compiled model instead.
        """
        key = _eager_model_cache_key(graph)

        with self.lock:
            future = self.cache.get(key)
            if future is None:
                session = _session()
                # Compilation mutates the graph; compile a copy so the
                # interpreter path still sees the original module.
                compiled = session.compile_async(graph.copy())
                future = self.cache[key] = compiled._compiled.and_then(
                    lambda _: session.init(compiled)
                )

        if not future.done():
            try:
                return self.interpreter.execute(graph, inputs)
            except UnsupportedGraphError:
                pass
            future.wait()

        model = future.result()
        return model(*inputs)


_MAX_EAGER_EXECUTOR_ENV_VAR = "MAX_EAGER_EXECUTOR"


def _default_composite() -> CompositeExecutor:
    """Builds the auto-selected eager executor.

    Interprets graphs within the ``MAX_INTERPRETER_MAX_OPS`` threshold and
    falls back to a cached compile on refusal or interpreter error.  This is
    the out-of-the-box eager execution path.
    """
    return CompositeExecutor(
        interpreter=InterpreterExecutor(max_ops=_interpreter_max_ops()),
        fallback_on_error=True,
    )


_EXECUTORS: dict[str, Callable[[], Executor]] = {
    "composite": _default_composite,
    "jit": JitExecutor,
    "compile": CompilingExecutor,
    "interpreter": InterpreterExecutor,
}


def _executor_from_env() -> Executor:
    name = (
        os.environ.get(_MAX_EAGER_EXECUTOR_ENV_VAR, "composite").lower().strip()
    )
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
    environment variable (``composite`` | ``jit`` | ``compile`` |
    ``interpreter``; default ``composite``), read at import time.
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
    "CompositeExecutor",
    "Executor",
    "InterpreterExecutor",
    "JitExecutor",
    "UnsupportedGraphError",
    "default_executor",
    "set_default_executor",
]
