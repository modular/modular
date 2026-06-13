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
"""Tests for max.experimental.executor."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

import pytest
from max._interpreter import MOInterpreter
from max._mlir_context import MLIRThreadPoolExecutor
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental.executor import (
    CompilingExecutor,
    Executor,
    InterpreterExecutor,
    JitExecutor,
    UnsupportedGraphError,
    _executor_from_env,
    default_executor,
    set_default_executor,
)
from max.experimental.realization_context import _eager_model_cache_key
from max.graph import Graph, TensorType, ops

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float_buffer(values: Sequence[float]) -> Buffer:
    """A host float32 buffer holding *values*."""
    buf = Buffer(DType.float32, [len(values)])
    for i, value in enumerate(values):
        buf[i] = value
    return buf


def _values(buf: Buffer | None) -> list[float]:
    """The contents of a rank-1 buffer as a list of floats."""
    assert buf is not None
    (n,) = buf.shape
    return [buf[i].item() for i in range(n)]


def _add_graph() -> tuple[Graph, Buffer]:
    """A simple add-constant graph plus a matching input buffer."""
    input_type = TensorType(DType.float32, [2], CPU())
    with Graph("add", input_types=[input_type]) as g:
        (x,) = g.inputs
        c = ops.constant([1.0, 1.0], dtype=DType.float32, device=CPU())
        g.output(ops.add(x, c))
    return g, _float_buffer([3.0, 4.0])


def _custom_op_graph() -> Graph:
    """A graph that triggers InterpreterExecutor refusal via name injection."""
    with Graph("custom", input_types=[]) as g:
        c = ops.constant([1.0], dtype=DType.float32, device=CPU())
        g.output(c)
    return g


class _RecordingExecutor:
    """Records calls and returns canned buffers."""

    def __init__(self, result: Sequence[Buffer | None]) -> None:
        self.calls: list[tuple[Graph, Sequence[Buffer]]] = []
        self._result = result

    def execute(
        self, graph: Graph, inputs: Sequence[Buffer]
    ) -> Sequence[Buffer | None]:
        self.calls.append((graph, inputs))
        return self._result


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """All three concrete executors satisfy the Executor runtime_checkable protocol."""

    def test_interpreter_executor_is_executor(self) -> None:
        assert isinstance(InterpreterExecutor(), Executor)

    def test_compiling_executor_is_executor(self) -> None:
        assert isinstance(CompilingExecutor(), Executor)

    def test_jit_executor_is_executor(self) -> None:
        assert isinstance(JitExecutor(), Executor)

    def test_recording_executor_is_executor(self) -> None:
        """A duck-typed executor with the right signature also satisfies Protocol."""
        assert isinstance(_RecordingExecutor([]), Executor)


# ---------------------------------------------------------------------------
# InterpreterExecutor
# ---------------------------------------------------------------------------


class TestInterpreterExecutor:
    def test_execute_simple_graph(self) -> None:
        graph, buf = _add_graph()
        executor = InterpreterExecutor()
        results = executor.execute(graph, [buf])
        assert len(results) == 1
        assert _values(results[0]) == pytest.approx([4.0, 5.0])

    def test_raises_unsupported_for_compilation_required_op(
        self, monkeypatch: Any
    ) -> None:
        """InterpreterExecutor raises UnsupportedGraphError for CustomOp graphs."""
        monkeypatch.setattr(
            MOInterpreter, "_COMPILATION_REQUIRED_OP_NAMES", ("ConstantOp",)
        )
        graph = _custom_op_graph()
        executor = InterpreterExecutor()
        with pytest.raises(UnsupportedGraphError):
            executor.execute(graph, [])

    def test_raises_unsupported_when_over_max_ops(self) -> None:
        """InterpreterExecutor raises UnsupportedGraphError when graph exceeds max_ops."""
        graph, buf = _add_graph()
        # Set max_ops=1 so the ~3-op add graph is refused.
        executor = InterpreterExecutor(max_ops=1)
        with pytest.raises(UnsupportedGraphError):
            executor.execute(graph, [buf])

    def test_propagates_runtime_error(self, monkeypatch: Any) -> None:
        """Runtime errors from execute() propagate unchanged (no swallowing)."""

        def _boom(self: Any, graph: Any, inputs: Any) -> Any:
            raise RuntimeError("simulated kernel failure")

        monkeypatch.setattr(MOInterpreter, "execute", _boom)
        graph, buf = _add_graph()
        executor = InterpreterExecutor()
        with pytest.raises(RuntimeError, match="simulated kernel failure"):
            executor.execute(graph, [buf])


# ---------------------------------------------------------------------------
# CompilingExecutor
# ---------------------------------------------------------------------------


class TestCompilingExecutor:
    def test_compile_and_execute_round_trip(self) -> None:
        graph, buf = _add_graph()
        executor = CompilingExecutor()
        results = executor.execute(graph, [buf])
        assert len(results) >= 1
        assert _values(results[0]) == pytest.approx([4.0, 5.0])

    def test_each_call_recompiles(self) -> None:
        """CompilingExecutor has no cache; two calls on identical graphs both succeed."""
        graph1, buf1 = _add_graph()
        graph2, buf2 = _add_graph()
        executor = CompilingExecutor()
        r1 = executor.execute(graph1, [buf1])
        r2 = executor.execute(graph2, [buf2])
        assert _values(r1[0]) == pytest.approx(_values(r2[0]))


# ---------------------------------------------------------------------------
# JitExecutor
# ---------------------------------------------------------------------------


class TestJitExecutor:
    def test_basic_execute(self) -> None:
        graph, buf = _add_graph()
        executor = JitExecutor()
        results = executor.execute(graph, [buf])
        assert _values(results[0]) == pytest.approx([4.0, 5.0])

    def test_cache_idempotence_same_graph(self) -> None:
        """Structurally identical graphs share one cache entry and future."""
        graph, buf1 = _add_graph()
        executor = JitExecutor()

        # Prime the cache.
        executor.execute(graph, [buf1])

        # The cache must have exactly one entry.
        assert len(executor.cache) == 1
        cached_future = next(iter(executor.cache.values()))

        # Second execute on a fresh (but structurally identical) graph.
        graph2, buf2 = _add_graph()
        executor.execute(graph2, [buf2])

        # Still one entry and the future object is the same.
        assert len(executor.cache) == 1
        assert next(iter(executor.cache.values())) is cached_future

    def test_blocks_on_compile_when_interpreter_refuses(self) -> None:
        """Graphs the interpreter refuses are served by the compiled model."""
        graph, buf = _add_graph()
        executor = JitExecutor()
        executor.interpreter = InterpreterExecutor(max_ops=0)
        results = executor.execute(graph, [buf])
        assert _values(results[0]) == pytest.approx([4.0, 5.0])

    def test_interpreter_runtime_error_propagates(
        self, monkeypatch: Any
    ) -> None:
        """Errors raised mid-execution are never masked as cache misses."""

        def _boom(self: Any, graph: Any, inputs: Any) -> Any:
            raise RuntimeError("simulated kernel failure")

        monkeypatch.setattr(MOInterpreter, "execute", _boom)
        graph, buf = _add_graph()
        executor = JitExecutor()
        with pytest.raises(RuntimeError, match="simulated kernel failure"):
            executor.execute(graph, [buf])

    def test_interpreter_max_ops_env(self, monkeypatch: Any) -> None:
        """MAX_INTERPRETER_MAX_OPS caps the graph size the interpreter serves."""
        monkeypatch.setenv("MAX_INTERPRETER_MAX_OPS", "0")
        executor = JitExecutor()
        assert executor.interpreter._max_ops == 0
        # With the interpreter refusing everything, execution still succeeds
        # via the compiled model.
        graph, buf = _add_graph()
        (out,) = executor.execute(graph, [buf])
        assert _values(out) == pytest.approx([4.0, 5.0])

    def test_failed_compile_propagates(self) -> None:
        """A failed compile re-raises on every call; it is not retried."""
        graph, buf = _add_graph()
        bad_future: Future[Any] = Future()
        bad_future.set_exception(RuntimeError("compile exploded"))

        key = _eager_model_cache_key(graph)
        executor = JitExecutor()
        with executor.lock:
            executor.cache[key] = bad_future

        with pytest.raises(RuntimeError, match="compile exploded"):
            executor.execute(graph, [buf])
        with pytest.raises(RuntimeError, match="compile exploded"):
            executor.execute(graph, [buf])

    def test_isolation_between_instances(self) -> None:
        """Two JitExecutor instances have independent caches."""
        graph1, buf1 = _add_graph()
        ex1 = JitExecutor()
        ex2 = JitExecutor()
        ex1.execute(graph1, [buf1])
        assert len(ex1.cache) == 1
        assert len(ex2.cache) == 0


class TestJitExecutorSnapshot:
    """The background compile owns a snapshot, not the caller's module."""

    def test_background_compile_survives_caller_mutation(self) -> None:
        """The interpreter path may legalize the caller's module in place;
        the background compile must still succeed and agree with it."""
        graph, inp = _add_graph()
        executor = JitExecutor()
        (first,) = executor.execute(graph, [inp])
        (future,) = executor.cache.values()
        future.result(timeout=300)
        (second,) = executor.execute(graph, [inp])
        assert _values(first) == pytest.approx(_values(second))

    def test_repeated_execute_after_caller_mutation(self) -> None:
        """Executing, mutating nothing, and executing again stays cached and
        correct -- two executes of one graph object must not double-process
        the module."""
        graph, inp = _add_graph()
        executor = JitExecutor()
        (first,) = executor.execute(graph, [inp])
        (second,) = executor.execute(graph, [inp])
        assert _values(first) == pytest.approx(_values(second))


# ---------------------------------------------------------------------------
# default_executor / set_default_executor / MAX_EAGER_EXECUTOR
# ---------------------------------------------------------------------------


class TestDefaultExecutor:
    """Tests for the ambient default executor mechanism."""

    def test_env_default_is_jit(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("MAX_EAGER_EXECUTOR", raising=False)
        assert isinstance(_executor_from_env(), JitExecutor)

    def test_env_selects_jit(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("MAX_EAGER_EXECUTOR", "jit")
        assert isinstance(_executor_from_env(), JitExecutor)

    def test_env_selects_interpreter(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("MAX_EAGER_EXECUTOR", "interpreter")
        assert isinstance(_executor_from_env(), InterpreterExecutor)

    def test_env_selects_compile(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("MAX_EAGER_EXECUTOR", "compile")
        assert isinstance(_executor_from_env(), CompilingExecutor)

    def test_env_unknown_value_raises(self, monkeypatch: Any) -> None:
        """An unrecognized MAX_EAGER_EXECUTOR fails loudly, never silently."""
        monkeypatch.setenv("MAX_EAGER_EXECUTOR", "compiled")
        with pytest.raises(ValueError, match="MAX_EAGER_EXECUTOR"):
            _executor_from_env()

    def test_set_default_executor_overrides(self) -> None:
        custom = CompilingExecutor()
        with set_default_executor(custom):
            assert default_executor() is custom

    def test_set_default_executor_restores(self) -> None:
        """The returned handle restores the previous executor on exit."""
        original = default_executor()
        replacement = InterpreterExecutor()
        with set_default_executor(replacement):
            assert default_executor() is replacement
        assert default_executor() is original

    def test_set_default_executor_discard_keeps(self) -> None:
        """The set is eager; discarding the handle keeps the new executor."""
        original = default_executor()
        replacement = InterpreterExecutor()
        set_default_executor(replacement)
        try:
            assert default_executor() is replacement
        finally:
            set_default_executor(original)

    def test_default_executor_thread_safe(self) -> None:
        """default_executor() is safe to call from multiple threads."""
        results: list[Executor] = []
        errors: list[Exception] = []

        def _get() -> None:
            try:
                results.append(default_executor())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 8
        # All threads should see the same singleton.
        assert all(r is results[0] for r in results)


class TestJitExecutorDemandPriority:
    """Demand compiles never wait behind speculative queue entries."""

    def test_demand_skips_speculative_queue(self) -> None:
        executor = JitExecutor()
        # Force the demand path: the interpreter must refuse the graph.
        executor.interpreter = InterpreterExecutor(max_ops=0)
        release = threading.Event()
        started = threading.Event()

        def _stall() -> None:
            started.set()
            release.wait(timeout=30)

        try:
            # Occupy the worker and queue a second job behind it.
            executor.pool.submit(_stall)
            started.wait(timeout=10)
            executor.pool.submit(lambda: None)

            # A demand execute must complete while the queue is stalled.
            graph, inp = _add_graph()
            (out,) = executor.execute(graph, [inp])
            assert _values(out) == pytest.approx([4.0, 5.0])
        finally:
            release.set()

    def test_concurrent_demands_share_one_compile(
        self, monkeypatch: Any
    ) -> None:
        """Racing demands for one graph produce exactly one compile; the
        losers wait on the winner's future instead of cancelling it."""
        import max.experimental.executor as executor_module

        session = executor_module._session()
        load_count = [0]
        loaders: list[str] = []

        class _CountingSession:
            def load(self, graph: Graph) -> Any:
                load_count[0] += 1
                loaders.append(threading.current_thread().name)
                return session.load(graph)

        monkeypatch.setattr(
            executor_module, "_session", lambda: _CountingSession()
        )

        executor = JitExecutor()
        # Force the demand path: the interpreter must refuse the graph.
        executor.interpreter = InterpreterExecutor(max_ops=0)
        # A single worker so one stalled job keeps the speculative compile
        # queued (cancellable) for every racing demand.
        executor.pool = MLIRThreadPoolExecutor(max_workers=1)
        release = threading.Event()
        started = threading.Event()

        def _stall() -> None:
            started.set()
            release.wait(timeout=30)

        # One graph object per thread: executors mutate graphs in place, so
        # sharing one object across threads is outside the contract.  The
        # graphs are structurally identical and share a cache key.
        workloads = [_add_graph() for _ in range(4)]
        results: list[Sequence[Buffer | None]] = []
        errors: list[Exception] = []

        def _demand(graph: Graph, inp: Buffer) -> None:
            try:
                results.append(executor.execute(graph, [inp]))
            except Exception as e:
                errors.append(e)

        try:
            # Stall the pool so the speculative compile stays cancellable.
            executor.pool.submit(_stall)
            started.wait(timeout=10)

            threads = [
                threading.Thread(target=_demand, args=pair)
                for pair in workloads
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=300)
        finally:
            release.set()

        assert not errors
        assert len(results) == 4
        for result in results:
            assert _values(result[0]) == pytest.approx([4.0, 5.0])
        assert load_count[0] == 1, loaders
