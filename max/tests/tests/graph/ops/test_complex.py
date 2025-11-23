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
"""Tests for ops.complex."""

import pytest
from conftest import (
    GraphBuilder,
    broadcast_shapes,
    broadcastable_tensor_types,
    dtypes,
    shapes,
    static_dims,
    symbolic_dims,
    tensor_types,
)
from hypothesis import assume, given, reject
from hypothesis import strategies as st
from max.graph import Shape, TensorType, ops

# Strategy that generates shapes with even static last dimensions

even_static_last_dim_shapes = st.builds(
    lambda prefix_dims, last_dim: Shape(prefix_dims + [last_dim]),
    st.lists(st.one_of(static_dims(), symbolic_dims), min_size=0, max_size=4),
    static_dims(min=2, max=100).filter(lambda d: int(d) % 2 == 0),
)

# Strategy that generates shapes with odd static last dimensions

odd_static_last_dim_shapes = st.builds(
    lambda prefix_dims, last_dim: Shape(prefix_dims + [last_dim]),
    st.lists(st.one_of(static_dims(), symbolic_dims), min_size=0, max_size=4),
    static_dims(min=1, max=99).filter(lambda d: int(d) % 2 != 0),
)

# Strategy that biases toward dynamic last dimensions

dynamic_last_dim_shapes = st.builds(
    lambda prefix_dims, last_dim: Shape(prefix_dims + [last_dim]),
    st.lists(st.one_of(static_dims(), symbolic_dims), min_size=0, max_size=4),
    symbolic_dims,
)

# Strategy that generates complex valued shapes (final dim 2)
complex_shapes = shapes().map(lambda shape: Shape([*shape, 2]))


@given(base_type=tensor_types(shapes=even_static_last_dim_shapes))
def test_as_interleaved_complex__valid(
    graph_builder: GraphBuilder,
    base_type: TensorType,
) -> None:
    """Test as_interleaved_complex with valid inputs."""
    *_, last = base_type.shape

    with graph_builder(input_types=[base_type]) as graph:
        out = ops.as_interleaved_complex(graph.inputs[0].tensor)
        # Output shape should be same except last dim is halved and new dim of 2 added
        expected_shape = base_type.shape[:-1] + [int(last) // 2, 2]
        assert out.type.shape == expected_shape
        graph.output(out)


@given(base_type=tensor_types(shapes=odd_static_last_dim_shapes))
def test_as_interleaved_complex__error__odd_last_dim(
    graph_builder: GraphBuilder,
    base_type: TensorType,
) -> None:
    """Test that as_interleaved_complex raises an error when last dimension is odd."""

    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(ValueError, match="must be divisible by 2"):
            ops.as_interleaved_complex(graph.inputs[0].tensor)


@given(base_type=tensor_types(shapes=dynamic_last_dim_shapes))
def test_as_interleaved_complex__error__dynamic_last_dim(
    graph_builder: GraphBuilder,
    base_type: TensorType,
) -> None:
    """Test that as_interleaved_complex raises an error when last dimension is dynamic."""

    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(TypeError, match="must be static"):
            ops.as_interleaved_complex(graph.inputs[0].tensor)


shared_dtypes = st.shared(dtypes)


@given(types=broadcastable_tensor_types(2))
def test_complex_mul(
    graph_builder: GraphBuilder,
    types: tuple[TensorType, TensorType],
) -> None:
    lhs_type, rhs_type = types
    lhs_type = TensorType(lhs_type.dtype, [*lhs_type.shape, 2], lhs_type.device)
    rhs_type = TensorType(rhs_type.dtype, [*rhs_type.shape, 2], rhs_type.device)

    with graph_builder(input_types=[lhs_type, rhs_type]) as graph:
        lhs, rhs = graph.inputs
        result = ops.complex.mul(lhs.tensor, rhs.tensor)
        assert result.type == (lhs.tensor * rhs.tensor).type


@given(types=broadcastable_tensor_types(2))
def test_complex_mul__non_complex_input_type(
    graph_builder: GraphBuilder,
    types: tuple[TensorType, TensorType],
) -> None:
    lhs_type, rhs_type = types
    assume(
        not lhs_type.shape
        or lhs_type.shape[-1] != 2
        or not rhs_type.shape
        or rhs_type.shape[-1] != 2
    )
    with graph_builder(input_types=[lhs_type, rhs_type]) as graph:
        lhs, rhs = graph.inputs
        with pytest.raises(ValueError):
            ops.complex.mul(lhs.tensor, rhs.tensor)


@given(
    lhs_type=tensor_types(dtypes=shared_dtypes),
    rhs_type=tensor_types(dtypes=shared_dtypes),
)
def test_complex_mul__non_broadcastable(
    graph_builder: GraphBuilder, lhs_type: TensorType, rhs_type: TensorType
) -> None:
    try:
        broadcast_shapes(lhs_type.shape, rhs_type.shape)
    except ValueError:
        pass
    else:
        reject()

    lhs_type = TensorType(lhs_type.dtype, [*lhs_type.shape, 2], lhs_type.device)
    rhs_type = TensorType(rhs_type.dtype, [*rhs_type.shape, 2], rhs_type.device)

    with graph_builder(input_types=[lhs_type, rhs_type]) as graph:
        lhs, rhs = graph.inputs
        with pytest.raises(ValueError):
            ops.complex.mul(lhs.tensor, rhs.tensor)
