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

import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import realization_context as rc
from max.experimental.tensor import (
    Tensor,
    TensorType,
    current_realization_context,
    realization_context,
)
from max.graph import Graph


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
        a = Tensor.constant(42)

    assert not a.real
    assert a.item() == 42
    assert a.real


def test_lazy_context_sync_realize_via_dlpack() -> None:
    with rc.LazyRealizationContext() as ctx, realization_context(ctx):
        a = Tensor.constant(42)

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
        x = Tensor.constant(1)
        y = x + 1
        assert not y.real
        assert y.state
        assert isinstance(y.state.ctx, rc.GraphRealizationContext)
        assert y.state.ctx.graph == graph
        graph.output(y)


def test_graph_context_using_realized_tensors() -> None:
    """Realized tensors become constants when added as sources."""
    input_type = TensorType(DType.float32, [3, 3], CPU())

    eager = Tensor.constant(5.0, dtype=DType.float32, device=CPU())
    assert eager.real

    with Graph("test", input_types=[input_type]) as graph:
        graph.output(eager + graph.inputs[0].tensor)

    model = rc._session().load(graph)
    input = Tensor.ones([3, 3], dtype=DType.float32, device=CPU())
    output = model(input)[0]
    assert output[0, 0].item() == 6


def test_graph_context_using_lazy_tensors() -> None:
    """Unrealized tensors become constants when added as sources."""
    input_type = TensorType(DType.float32, [3, 3], CPU())

    with F.lazy():
        lazy = Tensor.constant(5.0, dtype=DType.float32, device=CPU())
    assert not lazy.real

    with Graph("test", input_types=[input_type]) as graph:
        graph.output(lazy + graph.inputs[0].tensor)

    assert lazy.real

    model = rc._session().load(graph)
    input = Tensor.ones([3, 3], dtype=DType.float32, device=CPU())
    output = model(input)[0]
    assert output[0, 0].item() == 6


def test_composing_eager_and_lazy() -> None:
    with F.lazy():
        lazy = Tensor.constant(5.0, dtype=DType.float32, device=CPU())
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
