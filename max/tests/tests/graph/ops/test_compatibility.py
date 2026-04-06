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
"""Tests for PyTorch-compatibility tensor helpers built from graph ops."""

from __future__ import annotations

import pytest
from conftest import GraphBuilder
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorType, ops


def test_unflatten_infers_shape(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(
        DType.float32,
        [2, 12, 5],
        device=DeviceRef.CPU(),
    )

    with graph_builder(input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.unflatten(1, (4, -1))
        assert out.shape == [2, 4, 3, 5]
        graph.output(out)


def test_unflatten_rejects_incompatible_sizes(
    graph_builder: GraphBuilder,
) -> None:
    input_type = TensorType(DType.float32, [2, 10], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].tensor.unflatten(1, (4, -1))


def test_unbind_returns_tuple(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(DType.float32, [2, 3, 4], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type]) as graph:
        parts = graph.inputs[0].tensor.unbind(dim=1)
        assert isinstance(parts, tuple)
        assert len(parts) == 3
        assert [part.shape for part in parts] == [[2, 4], [2, 4], [2, 4]]
        graph.output(*parts)


def test_unbind_requires_static_axis_size(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(
        DType.float32,
        ["batch", "seq", 4],
        device=DeviceRef.CPU(),
    )

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].tensor.unbind(dim=1)


def test_repeat_matches_pytorch_rank_rules(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(DType.float32, [2, 3], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.repeat(4, 1, 2)
        assert out.shape == [4, 2, 6]
        graph.output(out)


def test_repeat_rejects_too_few_dims(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(DType.float32, [2, 3], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].tensor.repeat(2)


def test_masked_fill_broadcasts(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(
        DType.float32,
        [2, 3, 4],
        device=DeviceRef.CPU(),
    )
    mask_type = TensorType(DType.bool, [2, 1, 4], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type, mask_type]) as graph:
        out = graph.inputs[0].tensor.masked_fill(graph.inputs[1].tensor, 0.0)
        assert out.shape == input_type.shape
        assert out.dtype == DType.float32
        graph.output(out)


def test_amin_tuple_axes(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(DType.float32, [2, 3, 4, 5], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.amin(axis=(1, -2))
        assert out.shape == [2, 5]
        graph.output(out)


def test_swapaxes_alias(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(DType.float32, [2, 3, 4], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.swapaxes(1, 2)
        assert out.shape == [2, 4, 3]
        graph.output(out)


def test_flip_preserves_symbolic_shape(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(
        DType.float32,
        ["batch", "seq", 4],
        device=DeviceRef.CPU(),
    )

    with graph_builder(input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.flip(dims=[1])
        assert out.shape == input_type.shape
        graph.output(out)


def test_flatten_and_size_aliases(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(
        DType.float32,
        ["batch", 3, 4],
        device=DeviceRef.CPU(),
    )

    with graph_builder(input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        out = x.reshape(x.size(0), -1, x.size(-1)).flatten(0, 1)
        assert out.shape == [Dim("batch") * 3, 4]
        graph.output(out)


def test_bitwise_operator_overloads_use_integer_semantics(
    graph_builder: GraphBuilder,
) -> None:
    input_type = TensorType(DType.int32, [2, 3], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type, input_type]) as graph:
        x = graph.inputs[0].tensor
        y = graph.inputs[1].tensor
        assert (x & y).dtype == DType.int32
        assert (x | y).dtype == DType.int32
        assert (x ^ y).dtype == DType.int32
        graph.output(x & y, x | y, x ^ y)


def test_named_bitwise_ops_support_bool(graph_builder: GraphBuilder) -> None:
    input_type = TensorType(DType.bool, [2, 3], device=DeviceRef.CPU())

    with graph_builder(input_types=[input_type, input_type]) as graph:
        x = graph.inputs[0].tensor
        y = graph.inputs[1].tensor
        out = ops.bitwise_xor(x, y)
        assert out.dtype == DType.bool
        graph.output(out)
