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
"""ops.outer tests."""

import pytest
from conftest import MAX_INT64, shapes, symbolic_dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Graph, StaticDim, SymbolicDim, TensorType, ops

shared_shapes = st.shared(shapes().filter(lambda shape: 0 not in shape))
tensor_types_nd = tensor_types(shapes=shared_shapes)


def valid_repeat_counts(dim):  # noqa: ANN001
    if isinstance(dim, StaticDim):
        return st.integers(min_value=1, max_value=MAX_INT64 // dim.dim)
    else:
        return st.integers(min_value=1, max_value=MAX_INT64)


valid_repeats = shared_shapes.flatmap(
    lambda shape: st.tuples(*(valid_repeat_counts(dim) for dim in shape))
).map(list)


@given(input_type=tensor_types_nd, repeats=valid_repeats)
def test_tile__valid(input_type: TensorType, repeats: list[int]) -> None:
    with Graph("tiles", input_types=[input_type]) as graph:
        out = ops.tile(graph.inputs[0], repeats)
        expected_shape = [dim * r for r, dim in zip(repeats, input_type.shape)]
        assert out.shape == expected_shape
        graph.output(out)


valid_symbolic_repeats = shared_shapes.flatmap(
    lambda shape: st.tuples(*(symbolic_dims for _ in shape))
).map(list)


@given(input_type=tensor_types_nd, repeats=valid_symbolic_repeats)
def test_tile__valid_symbolic(
    input_type: TensorType, repeats: list[SymbolicDim]
) -> None:
    with Graph("tiles", input_types=[input_type]) as graph:
        out = ops.tile(graph.inputs[0], repeats)
        expected_shape = [dim * r for r, dim in zip(repeats, input_type.shape)]
        assert out.shape == expected_shape
        # TODO(AIPIPE-185): actually enable full graph verification here.
        # While the individual tile op is valid, the graph as a whole is not.
        # It can use undefined symbolic dims in the repeats.
        # graph.output(out)


invalid_static_repeats = shared_shapes.flatmap(
    lambda shape: st.lists(
        st.integers(min_value=-(2**63), max_value=0),
        min_size=len(shape),
        max_size=len(shape),
    )
)

invalid_len = shared_shapes.flatmap(
    lambda shape: st.lists(st.just(1)).filter(lambda l: len(l) != len(shape))
)

invalid_repeats = st.one_of(invalid_static_repeats, invalid_len)


@given(input_type=tensor_types_nd, repeats=invalid_repeats)
def test_tile__invalid(input_type: TensorType, repeats: list[int]) -> None:
    assume(len(input_type.shape) != 0)
    with Graph("tiles", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            out = ops.tile(graph.inputs[0], repeats)
