# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Test the max.graph Python bindings."""

import operator
from functools import reduce

import pytest
from conftest import non_static_axes, shapes, static_axes, tensor_types
from hypothesis import assume, example, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, Shape, StaticDim, TensorType, ops

shared_types = st.shared(tensor_types())
chunks = st.integers(min_value=1, max_value=20)


@given(input_type=shared_types, chunks=chunks, axis=static_axes(shared_types))
def test_chunk(input_type: TensorType, chunks: int, axis: int) -> None:
    assert isinstance(input_type.shape[axis], StaticDim)
    chunk_size = int(input_type.shape[axis])

    assume(chunk_size * chunks < 2**63)
    product = reduce(
        operator.mul, [int(axis) for axis in input_type.shape.static_dims], 1
    )
    assume(product * chunks < 2**63)

    expected_shape = Shape(input_type.shape)
    expected_type = TensorType(
        input_type.dtype, expected_shape, device=DeviceRef.CPU()
    )
    input_type.shape[axis] = Dim(chunk_size * chunks)

    with Graph("chunk", input_types=[input_type]) as graph:
        outs = ops.chunk(graph.inputs[0], chunks, axis=axis)
        assert len(outs) == chunks
        assert all(out.type == expected_type for out in outs)


@given(
    input_type=shared_types,
    chunks=st.integers(min_value=2),
    axis=static_axes(shared_types),
)
@example(
    input_type=TensorType(
        DType.float32, [9223372036854775806], device=DeviceRef.CPU()
    ),
    chunks=2,
    axis=-1,
).via("MAXPLAT-183")
def test_chunk__not_exact(
    input_type: TensorType, chunks: int, axis: int
) -> None:
    assert isinstance(input_type.shape[axis], StaticDim)
    assume(int(input_type.shape[axis]) % chunks != 0)
    with Graph("chunk", input_types=[input_type]) as graph:
        with pytest.raises(ValueError, match="must statically divide"):
            outs = ops.chunk(graph.inputs[0], chunks, axis=axis)


@given(
    input_type=tensor_types(shapes=shapes(min_rank=0, max_rank=0)), chunks=...
)
def test_chunk__split_scalar(input_type: TensorType, chunks: int) -> None:
    assume(chunks > 1)
    with Graph("chunk", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            outs = ops.chunk(graph.inputs[0], chunks)


@given(input_type=shared_types, chunks=..., axis=non_static_axes(shared_types))
def test_chunk__non_static_dim(
    input_type: TensorType, chunks: int, axis: int
) -> None:
    assert not isinstance(input_type.shape[axis], StaticDim)
    with Graph("chunk", input_types=[input_type]) as graph:
        with pytest.raises(TypeError):
            outs = ops.chunk(graph.inputs[0], chunks, axis=axis)
