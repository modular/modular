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
"""ops.repeat_interleave tests."""

import operator
from collections.abc import Iterable
from functools import reduce

import pytest
from conftest import GraphBuilder, axes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Dim, Shape, StaticDim, TensorType, ops

shared_tensor_types = st.shared(tensor_types())

_MAX_INT64 = 2**63 - 1


def _int64_product_fits(values: Iterable[Dim | int], initial: int = 1) -> bool:
    """Check if the product of values fits in int64 without intermediate overflow.

    Unlike Python's arbitrary-precision integers, int64 arithmetic can overflow
    during intermediate multiplications even if the final result would fit.
    For example, [2, 2^62, 0] has product 0, but 2 * 2^62 overflows int64.
    """
    result = abs(int(initial))
    for v in values:
        v = abs(int(v))
        if v == 0:
            return True  # Product is 0, no overflow possible
        if result > _MAX_INT64 // v:
            return False  # Multiplication would overflow
        result *= v
    return result <= _MAX_INT64


@given(
    type=shared_tensor_types,
    repeats=st.integers(min_value=1),
    axis=axes(shared_tensor_types),
)
def test_repeat_interleave(
    graph_builder: GraphBuilder,
    type: TensorType,
    repeats: int,
    axis: int,
) -> None:
    assume(repeats <= _MAX_INT64)  # repeats must fit in int64
    dim = type.shape[axis]
    assume(
        not isinstance(dim, StaticDim) or _int64_product_fits([dim], repeats)
    )
    with graph_builder(input_types=[type]) as graph:
        out = ops.repeat_interleave(graph.inputs[0].tensor, repeats, axis)
        target_shape = list(type.shape)
        target_shape[axis] *= repeats
        assert out.shape == target_shape
        assert out.dtype == type.dtype
        graph.output(out)


@given(
    type=shared_tensor_types,
    axis=axes(shared_tensor_types),
)
def test_vector_repeats(
    graph_builder: GraphBuilder, type: TensorType, axis: int
) -> None:
    dim = type.shape[axis]

    repeats_type = TensorType(DType.int64, [dim], device=DeviceRef.CPU())
    with graph_builder(input_types=[type, repeats_type]) as graph:
        out_dim = "new_dim"
        out = ops.repeat_interleave(
            graph.inputs[0].tensor,
            graph.inputs[1].tensor,
            axis,
            out_dim="new_dim",
        )
        target_shape = list(type.shape)
        target_shape[axis] = Dim(out_dim)
        assert out.shape == target_shape
        assert out.dtype == type.dtype
        graph.output(out)


@given(
    type=shared_tensor_types,
    repeats=st.integers(min_value=1, max_value=2**63 - 1),
)
def test_repeat_interleave__no_axis(
    graph_builder: GraphBuilder,
    type: TensorType,
    repeats: int,
) -> None:
    # Check that the product of static dims fits in int64 (for reshape and output)
    assume(_int64_product_fits(type.shape.static_dims, repeats))
    with graph_builder(input_types=[type]) as graph:
        out = ops.repeat_interleave(graph.inputs[0].tensor, repeats)
        flat_size = reduce(operator.mul, type.shape, 1)
        target_shape = Shape([flat_size * repeats])
        assert out.shape == target_shape
        assert out.dtype == type.dtype
        graph.output(out)


@given(
    type=shared_tensor_types,
    repeats=...,
    axis=st.one_of(axes(shared_tensor_types), st.none()),
)
def test_repeat_interleave__nonpositive_repeats(
    graph_builder: GraphBuilder,
    type: TensorType,
    repeats: int,
    axis: int | None,
) -> None:
    assume(repeats <= 0)
    with graph_builder(input_types=[type]) as graph:
        with pytest.raises(ValueError):
            ops.repeat_interleave(graph.inputs[0].tensor, repeats, axis=axis)


@given(
    type=shared_tensor_types,
    repeats=st.integers(min_value=1),
    axis=...,
)
def test_repeat_interleave__axis_out_of_bounds(
    graph_builder: GraphBuilder,
    type: TensorType,
    repeats: int,
    axis: int,
) -> None:
    assume(not -type.rank <= axis < type.rank)
    with graph_builder(input_types=[type]) as graph:
        with pytest.raises(ValueError):
            ops.repeat_interleave(graph.inputs[0].tensor, repeats, axis=axis)
