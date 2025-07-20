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
"""ops.cumsum tests."""

import pytest
from conftest import axes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Graph, TensorType, ops

input_types = st.shared(tensor_types())


@given(
    input_type=input_types, axis=axes(input_types), exclusive=..., reverse=...
)
def test_cumsum(
    input_type: TensorType, axis: int, exclusive: bool, reverse: bool
) -> None:
    with Graph("cumsum", input_types=[input_type]) as graph:
        out = ops.cumsum(
            graph.inputs[0], axis=axis, exclusive=exclusive, reverse=reverse
        )
        assert out.type == input_type


@given(input_type=input_types, axis=..., exclusive=..., reverse=...)
def test_cumsum__invalid_axis(
    input_type: TensorType, axis: int, exclusive: bool, reverse: bool
) -> None:
    assume(not -input_type.rank <= axis < input_type.rank)
    with Graph("cumsum", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            ops.cumsum(
                graph.inputs[0], axis=axis, exclusive=exclusive, reverse=reverse
            )
