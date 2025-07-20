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
"""ops.pad tests."""

import pytest
from conftest import tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Shape, StaticDim, TensorType, ops

input_types = st.shared(tensor_types())


def padded_size(shape: Shape, padding: list[int]) -> int:
    total = 1

    for i, s in enumerate(shape):
        if not isinstance(s, StaticDim):
            continue

        low = padding[2 * i]
        high = padding[2 * i + 1]
        total *= low + s.dim + high

    return total


def paddings_for(input_types, low=0, high=16):  # noqa: ANN001
    return input_types.flatmap(
        lambda type: st.lists(
            st.integers(min_value=low, max_value=high),
            min_size=2 * type.rank,
            max_size=2 * type.rank,
        )
    )


@given(input_type=input_types, paddings=paddings_for(input_types, low=-16))
def test_negative_paddings(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    paddings: list[int],
) -> None:
    """Padding by nothing does not change the type."""
    assume(input_type.rank > 0)
    assume(any(x < 0 for x in paddings))

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            _ = ops.pad(graph.inputs[0].tensor, paddings=paddings, value=0)


@given(input_type=input_types)
def test_no_padding(graph_builder, input_type: TensorType) -> None:  # noqa: ANN001
    """Padding by nothing does not change the type."""
    assume(input_type.rank > 0)
    paddings = [0] * (2 * input_type.rank)

    with graph_builder(input_types=[input_type]) as graph:
        out = ops.pad(graph.inputs[0].tensor, paddings=paddings, value=0)
        assert out.type == input_type
        graph.output(out)


@given(input_type=input_types, paddings=paddings_for(input_types))
def test_positive_paddings(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    paddings: list[int],
) -> None:
    """Test random paddings."""

    assume(0 < input_type.rank)
    with graph_builder(input_types=[input_type]) as graph:
        assume(padded_size(input_type.shape, paddings) < 2**63)

        out = ops.pad(graph.inputs[0].tensor, paddings=paddings, value=0)
        assert out.dtype == input_type.dtype
        graph.output(out)
