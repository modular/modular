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

import pytest
from conftest import axes, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import TensorType

shared_shapes = st.shared(shapes())


@given(
    input_type=tensor_types(shapes=shared_shapes),
    dims=shared_shapes.flatmap(
        lambda shape: st.permutations(range(len(shape)))
    ),
)
def test_permute_success(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    dims: list[int],
) -> None:
    target_shape = [input_type.shape[d] for d in dims]
    expected_type = TensorType(
        input_type.dtype, target_shape, input_type.device
    )
    with graph_builder(input_types=[input_type]) as graph:
        out = graph.inputs[0].permute(dims)
        assert out.type == expected_type

        graph.output(out)


rank_sized_list_ints = shared_shapes.flatmap(
    lambda shape: st.lists(
        st.integers(), min_size=shape.rank, max_size=shape.rank
    )
)


@given(input_type=tensor_types(shapes=shared_shapes), dims=rank_sized_list_ints)
def test_permute_out_of_range(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    dims: list[int],
) -> None:
    rank = input_type.rank
    assume(any(d >= rank or d < -rank for d in dims))
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(IndexError):
            graph.inputs[0].permute(dims)


@given(input_type=..., dims=...)
def test_permute_wrong_rank(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    dims: list[int],
) -> None:
    rank = input_type.rank
    assume(len(dims) != rank)
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].permute(dims)


shared_nontrivial_shapes = st.shared(shapes(min_rank=2))


@given(
    input_type=tensor_types(shapes=shared_nontrivial_shapes),
    dims=shared_nontrivial_shapes.flatmap(
        lambda shape: st.lists(
            axes(st.just(shape)), min_size=len(shape), max_size=len(shape)
        )
    ),
)
def test_permute_duplicates(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    dims: list[int],
) -> None:
    assume(len(set(dims)) < len(dims))
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].permute(dims)
