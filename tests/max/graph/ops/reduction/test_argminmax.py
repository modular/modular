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
"""ops.argmax tests."""

import pytest
from conftest import axes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import TensorType, ops

input_types = st.shared(tensor_types())
ops = st.sampled_from([ops.argmax, ops.argmin])


@given(input_type=input_types, op=ops, axis=axes(input_types))
def test_argminmax(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    op,  # noqa: ANN001
    axis: int,
) -> None:
    with graph_builder(input_types=[input_type]) as graph:
        out = op(graph.inputs[0], axis=axis)
        assert out.dtype == DType.int64
        expected_shape = list(input_type.shape)
        expected_shape[axis] = 1
        assert out.shape == expected_shape


@given(input_type=input_types, op=ops, axis=...)
def test_argminmax__invalid_axis(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    op,  # noqa: ANN001
    axis: int,
) -> None:
    assume(not -input_type.rank <= axis < input_type.rank)
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            op(graph.inputs[0], axis=axis)
