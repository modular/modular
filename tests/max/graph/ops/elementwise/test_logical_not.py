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
"""test the max.graph python bindings."""

import pytest
from conftest import GraphBuilder, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import TensorType
from max.graph.ops import logical_not


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_logical_not__same_type(
    graph_builder: GraphBuilder, tensor_type: TensorType
) -> None:
    with graph_builder(input_types=[tensor_type]) as graph:
        x = graph.inputs[0].tensor
        op = logical_not(x)
        assert op.type == tensor_type
        assert op.shape == x.shape

        op2 = ~x
        assert op2.type == tensor_type
        assert op2.shape == x.shape


@given(tensor_type=...)
def test_logical_not__invalid_dtype(
    graph_builder: GraphBuilder,
    tensor_type: TensorType,
) -> None:
    assume(tensor_type.dtype != DType.bool)
    with graph_builder(input_types=[tensor_type]) as graph:
        x = graph.inputs[0].tensor
        with pytest.raises(ValueError):
            logical_not(x)

        with pytest.raises(ValueError):
            ~x  # noqa: B018
