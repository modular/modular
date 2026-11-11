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
"""Tests for realization contexts in max.experimental."""

import asyncio
import weakref
from collections.abc import Sequence

import pytest
from max import functional as F
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import BufferValue, Graph, ops
from max.tensor import realization_context as rc
from max.tensor.executor import CompilingExecutor
from max.tensor.support import _session
from max.tensor.tensor import (
    Tensor,
    TensorType,
    current_realization_context,
    realization_context,
)


class _FailingExecutor:
    """Fails the first ``failures`` graphs it is handed, then runs for real.

    Letting a later call through is what shows the retry reaches execution
    instead of failing on a malformed graph.
    """

    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls = 0
        self._inner = CompilingExecutor()

    def execute(
        self, graph: Graph, inputs: Sequence[Buffer]
    ) -> Sequence[Buffer | None]:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("executor exploded")
        return self._inner.execute(graph, inputs)


class _FailingAfterExecutionExecutor:
    """Runs the graph for real, then raises on its first call.

    Models a failure that arrives once the graph's work is enqueued, which is
    the case a retry must not run again.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._inner = CompilingExecutor()

    def execute(
        self, graph: Graph, inputs: Sequence[Buffer]
    ) -> Sequence[Buffer | None]:
        self.calls += 1
        results = self._inner.execute(graph, inputs)
        if self.calls == 1:
            raise RuntimeError("executor exploded")
        return results


class _BadResultOnceExecutor:
    """Returns a real result with one non-buffer entry on its first call.

    Every result is rejected before any is applied, so the graph runs but
    nothing lands.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._inner = CompilingExecutor()

    def execute(
        self, graph: Graph, inputs: Sequence[Buffer]
    ) -> Sequence[Buffer | None]:
        self.calls += 1
        results: list[Buffer | None] = list(self._inner.execute(graph, inputs))
        if self.calls == 1:
            # Outputs are the seed, then the tensors in creation order, so
            # index 2 is `b` in the test below; everything before it applies.
            results[2] = None
        return results


def test_eager_context_executes_on_exit() -> None:
    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        a = Tensor.zeros([3, 3])
        b = a + 1
        assert not b.real

    # After context exit, tensors should be realized
    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_eager_add_source_requires_real_tensor() -> None:
    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        unrealized = Tensor.zeros([3, 3])
        assert not unrealized.real

        with pytest.raises(TypeError, match="Only realized tensors"):
            ctx.add_source(unrealized)


def test_eager_add_source_is_idempotent() -> None:
    realized = Tensor.zeros([3, 3])
    assert realized.real

    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        state1 = ctx.add_source(realized)
        state2 = ctx.add_source(realized)
        assert state1 == state2


def test_one_buffer_through_two_tensors_is_one_mutable_input() -> None:
    """A buffer reached through two Tensor objects is one graph input.

    `Tensor.local_shards` wraps each shard's buffer in a new Tensor on every
    call, so a multi-device step reaches one pool through many wrappers. Two
    `mo.buffer` inputs over the same memory break the compiler's assumption
    that distinct buffers don't alias, and it drops the ordering between
    their writes.
    """
    pool = Tensor.zeros([4])
    first = Tensor(storage=pool.storage)
    second = Tensor(storage=pool.storage)

    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        (a,) = ctx.add_mutable_source(first).values
        (b,) = ctx.add_mutable_source(second).values
        assert a is b


def test_eager_realize_inside_context_raises() -> None:
    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        a = Tensor.zeros([3, 3])

        with pytest.raises(
            TypeError, match="before realization context is completed"
        ):
            asyncio.run(ctx.realize_all())

        assert not a.real


def test_eager_multiple_operations_batched() -> None:
    with rc.EagerRealizationContext() as ctx, realization_context(ctx):
        a = Tensor.zeros([2, 2])
        b = a + 1
        c = b * 2
        d = c - 1
        assert not any(t.real for t in (a, b, c, d))

    assert all(t.real for t in (a, b, c, d))


def test_retry_after_a_failed_realization_reruns_the_graph() -> None:
    """The retry must not trip over the terminator the first attempt left."""
    executor = _FailingExecutor()
    ctx = rc.EagerRealizationContext(executor=executor)

    with pytest.raises(RuntimeError, match="executor exploded"):
        with ctx, realization_context(ctx):
            a = Tensor.zeros([2, 2])
            b = a + 1

    assert not b.real

    asyncio.run(b.realize)

    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_a_failure_that_recurs_on_retry_raises_the_original_error() -> None:
    """Every attempt reports its own failure, never a verifier error."""
    executor = _FailingExecutor(failures=2)
    ctx = rc.EagerRealizationContext(executor=executor)

    with pytest.raises(RuntimeError, match="executor exploded"):
        with ctx, realization_context(ctx):
            a = Tensor.zeros([2, 2])
            b = a + 1

    with pytest.raises(RuntimeError, match="executor exploded"):
        asyncio.run(b.realize)

    assert executor.calls == 2
    assert not b.real


def test_a_retry_does_not_replay_an_in_place_mutation() -> None:
    """A failure once the graph ran spends the context, not the buffer."""
    executor = _FailingAfterExecutionExecutor()
    ctx = rc.EagerRealizationContext(executor=executor)
    a = Tensor.zeros([1])

    with pytest.raises(RuntimeError, match="executor exploded"):
        with ctx, realization_context(ctx):
            F.buffer_store(a, a + 1)

    assert a.driver_tensor.to(CPU())[0].item() == 1.0

    with pytest.raises(RuntimeError, match="after execution began"):
        a.item()

    assert executor.calls == 1, "a retry would store into `a` a second time"
    assert a.driver_tensor.to(CPU())[0].item() == 1.0


def test_a_failing_stamp_does_not_replace_the_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording is best effort, and the graph stays extendable regardless."""
    stamp = rc.stamp_device_write

    def raising(
        written: Sequence[Buffer],
        read: Sequence[Buffer] = (),
        error: str | None = None,
    ) -> None:
        raise RuntimeError("event pool exhausted")

    ctx = rc.EagerRealizationContext(executor=_BadResultOnceExecutor())
    monkeypatch.setattr(rc, "stamp_device_write", raising)
    with pytest.raises(TypeError, match="Expected all results"):
        with ctx, realization_context(ctx):
            a = Tensor.zeros([2, 2])
            b = a + 1

    assert not b.real

    monkeypatch.setattr(rc, "stamp_device_write", stamp)
    asyncio.run(b.realize)

    assert b.real
    assert b.driver_tensor.to(CPU())[0, 0].item() == 1.0


def test_a_failure_after_execution_stamps_its_mutated_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host read of a buffer the graph wrote must wait, even after a failure.

    The executor returned, so the mutated input holds an in-flight device
    write and gets no poison: the failure path must record a completion for
    it. The buffers are untracked on a host device, so a CPU run can only
    check that the bridge is handed them.
    """
    source = Tensor.zeros([2, 2])
    calls: list[tuple[list[Buffer], list[Buffer], str | None]] = []
    stamp = rc.stamp_device_write

    def recording(
        written: Sequence[Buffer],
        read: Sequence[Buffer] = (),
        error: str | None = None,
    ) -> None:
        calls.append((list(written), list(read), error))
        if len(calls) == 1:
            raise RuntimeError("event pool exhausted")
        stamp(written, read, error)

    monkeypatch.setattr(rc, "stamp_device_write", recording)
    ctx = rc.EagerRealizationContext(executor=CompilingExecutor())
    with pytest.raises(RuntimeError, match="event pool exhausted"):
        with ctx, realization_context(ctx):
            F.buffer_store(source, source + 1)
            b = source * 2

    assert calls, "the failure path must record a completion"
    written, read, error = calls[-1]
    assert error is None, "the work is in flight, not failed"
    assert source.driver_tensor in written
    assert b.driver_tensor in written
    # The mutated input and the two outputs: the seed is staged once, not
    # duplicated as a pending tensor.
    assert len(written) == 3
    assert read, "the submission read its inputs, and the records say so"


def test_a_failed_realization_leaves_the_graph_extendable() -> None:
    """Ops built after a failure still land before the terminator."""
    executor = _FailingExecutor()
    ctx = rc.EagerRealizationContext(executor=executor)

    with pytest.raises(RuntimeError, match="executor exploded"):
        with ctx, realization_context(ctx):
            a = Tensor.zeros([2, 2])
            b = a + 1

    with ctx, realization_context(ctx):
        c = b * 2

    assert c.real
    assert c.driver_tensor.to(CPU())[0, 0].item() == 2.0


def test_a_failed_realization_can_reuse_a_pruned_source() -> None:
    """A source only a dropped temporary used keeps a usable graph value.

    Finalizing drops the temporary, which erases the source's graph argument,
    so the cached value has to go with it.
    """
    executor = _FailingExecutor()
    ctx = rc.EagerRealizationContext(executor=executor)
    source = Tensor.ones([2, 2])

    with pytest.raises(RuntimeError, match="executor exploded"):
        with ctx, realization_context(ctx):
            dropped = source + 1
            b = Tensor.zeros([2, 2]) + 2
            del dropped

    with ctx, realization_context(ctx):
        c = source + b

    assert c.real
    assert c.driver_tensor.to(CPU())[0, 0].item() == 3.0


def test_pruned_read_leaves_a_pending_store_unrealized() -> None:
    """Pruning a dropped read must not realize the tensor a store targets.

    The read leaves an unused read-only argument beside the mutable input the
    store uses. Treating the pruned argument as the tensor's only source marks
    it realized, and the pending store silently never happens.
    """
    executor = _FailingAfterExecutionExecutor()
    ctx = rc.EagerRealizationContext(executor=executor)
    a = Tensor.zeros([1])

    with pytest.raises(RuntimeError, match="executor exploded"):
        with ctx, realization_context(ctx):
            dropped = a + 1
            del dropped
            F.buffer_store(a, Tensor.ones([1]) + 1)

    # The graph ran, so the context is spent rather than cleanly retryable.
    assert not a.real
    with pytest.raises(RuntimeError, match="after execution began"):
        a.item()

    assert executor.calls == 1, "the store already ran"


def test_signal_buffers_survive_pruning_an_unrelated_source() -> None:
    """Only a pruned signal buffer may reset the cached signal buffers.

    A signal buffer is a ``BufferType`` input, which a mutable source stands
    in for here: real ones need two accelerators. Resetting the cache for an
    unrelated argument makes the next collective add the shared runtime
    buffers to the graph a second time.
    """
    ctx = rc.EagerRealizationContext()
    signal = Tensor.zeros([4], dtype=DType.uint8)
    unused = Tensor.ones([1])

    with ctx, realization_context(ctx):
        (buffer,) = ctx.add_mutable_source(signal).values
        assert isinstance(buffer, BufferValue)
        ctx.signal_buffers = [buffer]
        ctx.add_source(unused)
        loaded = Tensor.from_graph_value(ops.buffer_load(buffer))

    assert loaded.real
    assert id(unused.storage) not in ctx.source_values, (
        "the unused source was pruned"
    )
    assert ctx.signal_buffers is not None, (
        "an unrelated pruned source reset them"
    )
    assert ctx.signal_buffers[0] is buffer


def test_lazy_context_does_not_execute_on_exit() -> None:
    with rc.LazyRealizationContext() as ctx, realization_context(ctx):
        a = Tensor.zeros([3, 3], device=CPU())
        b = a + 1

    assert not b.real


def test_lazy_context_explicit_realize() -> None:
    with rc.LazyRealizationContext() as ctx, realization_context(ctx):
        a = Tensor.zeros([3, 3])
        b = a + 1

    assert not a.real
    assert not b.real
    asyncio.run(b.realize)
    assert a.real
    assert b.real


def test_lazy_context_sync_realize_via_item() -> None:
    with rc.LazyRealizationContext() as ctx, realization_context(ctx):
        a = Tensor(42)

    assert not a.real
    assert a.item() == 42
    assert a.real


def test_lazy_context_sync_realize_via_dlpack() -> None:
    with rc.LazyRealizationContext() as ctx, realization_context(ctx):
        a = Tensor(42)

    assert not a.real
    _ = a.__dlpack__()
    assert a.real


def test_lazy_context_deleted_tensors() -> None:
    with rc.LazyRealizationContext() as ctx, realization_context(ctx):
        a = Tensor.zeros([3, 3], device=CPU())
        b = a + 1

    b_weak = weakref.ref(b)

    del b
    assert b_weak() is None

    assert not a.real
    asyncio.run(a.realize)
    assert a.real


def test_graph_context_cannot_realize() -> None:
    graph = Graph(
        "test", input_types=[TensorType(DType.float32, [3, 3], CPU())]
    )
    ctx = rc.GraphRealizationContext(graph)

    with pytest.raises(TypeError, match="Can't realize from a graph"):
        asyncio.run(ctx.realize_all())


def test_functional_uses_graph_context() -> None:
    input_type = TensorType(DType.float32, [3, 3], CPU())
    with Graph("test", input_types=[input_type]) as graph:
        x = Tensor(1)
        y = x + 1
        assert not y.real
        assert y.state
        assert isinstance(y.state.ctx, rc.GraphRealizationContext)
        assert y.state.ctx.graph == graph
        graph.output(y)


def test_graph_context_using_realized_tensors() -> None:
    """Realized tensors become constants when added as sources."""
    input_type = TensorType(DType.float32, [3, 3], CPU())

    eager = Tensor(5.0, dtype=DType.float32, device=CPU())
    assert eager.real

    with Graph("test", input_types=[input_type]) as graph:
        graph.output(eager + graph.inputs[0].tensor)

    model = _session().load(graph)
    input = Tensor.ones([3, 3], dtype=DType.float32, device=CPU())
    output = model(input)[0]
    assert output[0, 0].item() == 6


def test_graph_context_using_lazy_tensors() -> None:
    """Unrealized tensors become constants when added as sources."""
    input_type = TensorType(DType.float32, [3, 3], CPU())

    with F.lazy():
        lazy = Tensor(5.0, dtype=DType.float32, device=CPU())
    assert not lazy.real

    with Graph("test", input_types=[input_type]) as graph:
        graph.output(lazy + graph.inputs[0].tensor)

    assert lazy.real

    model = _session().load(graph)
    input = Tensor.ones([3, 3], dtype=DType.float32, device=CPU())
    output = model(input)[0]
    assert output[0, 0].item() == 6


def test_composing_eager_and_lazy() -> None:
    with F.lazy():
        lazy = Tensor(5.0, dtype=DType.float32, device=CPU())
    assert not lazy.real

    eager = lazy + 1
    assert eager.real
    assert lazy.real
    assert eager.item() == 6


def test_nested_contexts() -> None:
    with F.lazy():
        a = Tensor.zeros([3, 3], device=CPU())
        assert current_realization_context() is not None

        with rc.EagerRealizationContext() as inner, realization_context(inner):
            b = Tensor.ones([2, 2], device=CPU())
            assert not b.real

        assert b.real

        assert not a.real
    assert not a.real


# ---------------------------------------------------------------------------
# default_realization_context
# ---------------------------------------------------------------------------


def test_set_default_realization_context_scoped() -> None:
    """ensure_context constructs implicit contexts via the installed factory."""
    created: list[rc.EagerRealizationContext] = []

    def factory() -> rc.EagerRealizationContext:
        ctx = rc.EagerRealizationContext(use_interpreter=False)
        created.append(ctx)
        return ctx

    with rc.set_default_realization_context(factory):
        # No explicit context: realization goes through ensure_context.
        t = Tensor.zeros([2]) + 1.0
        assert created, "factory was not used for the implicit context"
        # use_interpreter=False routes through the compile-only executor.
        for ctx in created:
            assert isinstance(ctx._executor, CompilingExecutor)
        assert t.real
    # Exiting the scope restores the previous factory.
    assert rc._DEFAULT_REALIZATION_CONTEXT is rc.EagerRealizationContext


def test_set_default_realization_context_takes_effect_immediately() -> None:
    """The set happens at call time; discarding the handle keeps it."""

    def factory() -> rc.EagerRealizationContext:
        return rc.EagerRealizationContext(use_interpreter=False)

    undo = rc.set_default_realization_context(factory)
    try:
        assert rc._DEFAULT_REALIZATION_CONTEXT is factory
    finally:
        undo.__exit__(None, None, None)
    assert rc._DEFAULT_REALIZATION_CONTEXT is rc.EagerRealizationContext


def test_default_realization_context_default_is_eager() -> None:
    """The out-of-the-box default constructs an EagerRealizationContext."""
    ctx = rc.default_realization_context()
    assert isinstance(ctx, rc.EagerRealizationContext)
