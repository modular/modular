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
"""Runs reductions the graph compiler generates as rowwise kernels.

Each graph reduces with ``ops.sum`` or ``ops.max``, fused with the ops around
it, and is checked against numpy on random inputs, so a kernel that reads or
writes the wrong elements fails.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue, ops

_CPU = DeviceRef.CPU()


def _f32(shape: Sequence[int]) -> TensorType:
    return TensorType(DType.float32, shape, device=_CPU)


def _sum_inner(x: TensorValue) -> TensorValue:
    return ops.sum(x, axis=1)


def _sum_outer(x: TensorValue) -> TensorValue:
    return ops.sum(x, axis=0)


def _max_inner(x: TensorValue) -> TensorValue:
    return ops.max(x, axis=1)


def _sum_of_squares_plus_bias(x: TensorValue, b: TensorValue) -> TensorValue:
    return ops.sum(x * x, axis=1) + b


_CASES: list[
    tuple[
        str,
        Callable[..., TensorValue],
        list[list[int]],
        Callable[..., np.ndarray],
    ]
] = [
    ("sum_inner", _sum_inner, [[4, 8]], lambda x: x.sum(1, keepdims=True)),
    ("sum_outer", _sum_outer, [[4, 8]], lambda x: x.sum(0, keepdims=True)),
    ("max_inner", _max_inner, [[4, 8]], lambda x: x.max(1, keepdims=True)),
    (
        "sum_of_squares_plus_bias",
        _sum_of_squares_plus_bias,
        [[4, 8], [4, 1]],
        lambda x, b: (x * x).sum(1, keepdims=True) + b,
    ),
]


@pytest.mark.parametrize(
    ("build", "shapes", "reference"),
    [case[1:] for case in _CASES],
    ids=[case[0] for case in _CASES],
)
def test_rowwise_fusion(
    session: InferenceSession,
    build: Callable[..., TensorValue],
    shapes: list[list[int]],
    reference: Callable[..., np.ndarray],
) -> None:
    with Graph("rowwise", input_types=[_f32(s) for s in shapes]) as graph:
        graph.output(build(*(i.tensor for i in graph.inputs)))
    model = session.load(graph)

    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    result = model(*(Buffer.from_numpy(a) for a in inputs))[0]
    assert isinstance(result, Buffer)
    np.testing.assert_allclose(
        result.to_numpy(), reference(*inputs), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("rows", [3, 0])
def test_integer_mean(session: InferenceSession, rows: int) -> None:
    """An integer mean truncates, and is 0 over an empty axis."""
    input_type = TensorType(DType.int32, [2, "n"], device=_CPU)
    with Graph("integer_mean", input_types=[input_type]) as graph:
        graph.output(ops.mean(graph.inputs[0].tensor, axis=1))
    model = session.load(graph)

    x = np.arange(2 * rows, dtype=np.int32).reshape(2, rows)
    result = model(Buffer.from_numpy(x))[0]
    assert isinstance(result, Buffer)
    expected = x.sum(1, keepdims=True) // max(rows, 1)
    np.testing.assert_array_equal(result.to_numpy(), expected)


@pytest.mark.parametrize("dim", ["n", 70000])
def test_half_mean_of_long_axis(
    session: InferenceSession, dim: str | int
) -> None:
    """An f16 mean over more elements than f16 can count, 65504, stays finite."""
    input_type = TensorType(DType.float16, [1, dim], device=_CPU)
    with Graph("half_mean", input_types=[input_type]) as graph:
        graph.output(ops.mean(graph.inputs[0].tensor, axis=1))
    model = session.load(graph)

    # Mostly zeros, so the f16 sum is exact and only the scaling is tested.
    x = np.zeros((1, 70000), dtype=np.float16)
    x[0, :4] = 1000
    result = model(Buffer.from_numpy(x))[0]
    assert isinstance(result, Buffer)
    np.testing.assert_allclose(
        result.to_numpy().astype(np.float64), [[4000 / 70000]], rtol=5e-3
    )


def test_reshape_of_a_column_slice(session: InferenceSession) -> None:
    """A rowwise sum over a reshape of a column slice, with a symbolic row
    count. Only the reshape fuses: the slice is not contiguous, so it runs as
    its own kernel."""
    input_type = TensorType(DType.float32, ["rows", 640], device=_CPU)
    with Graph("reshaped_slice", input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor[:, 512:640]
        n = x.shape[0] // 4
        grouped = x[: n * 4].reshape((n, 4, 128))
        graph.output(ops.sum(grouped, axis=1))
    model = session.load(graph)

    for rows in (8, 129):
        x_np = np.random.default_rng(rows).standard_normal((rows, 640))
        x_np = x_np.astype(np.float32)
        result = model(Buffer.from_numpy(x_np))[0]
        assert isinstance(result, Buffer)
        n_np = rows // 4
        expected = (
            x_np[: n_np * 4, 512:640]
            .reshape(n_np, 4, 128)
            .sum(1, keepdims=True)
        )
        np.testing.assert_allclose(
            result.to_numpy(), expected, rtol=1e-5, atol=1e-4
        )


def test_buffer_store_of_a_rowwise_sum(session: InferenceSession) -> None:
    """A store of a rowwise sum into a buffer fuses into the rowwise kernel,
    which writes the buffer directly."""
    with Graph(
        "stored_sum",
        input_types=[
            BufferType(DType.float32, [4, 1], device=_CPU),
            _f32([4, 8]),
        ],
    ) as graph:
        ops.buffer_store(
            graph.inputs[0].buffer, _sum_inner(graph.inputs[1].tensor)
        )
        graph.output()
    model = session.load(graph)
    assert any(
        "Rowwise(" in s and "mo.mutable.store" in s
        for s in model.kernel_summaries
    ), model.kernel_summaries

    x_np = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    buf = Buffer.from_numpy(np.zeros((4, 1), dtype=np.float32))
    model(buf, Buffer.from_numpy(x_np))
    np.testing.assert_allclose(
        buf.to_numpy(), x_np.sum(1, keepdims=True), rtol=1e-5, atol=1e-5
    )
