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
from conftest import GraphBuilder, static_dims, symbolic_dims, tensor_types
from hypothesis import given
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
