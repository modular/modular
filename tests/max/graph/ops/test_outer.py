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
from conftest import shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, TensorType, ops

shared_dtypes = st.shared(st.from_type(DType))
tensor_types_1d = tensor_types(
    dtypes=shared_dtypes,
    shapes=shapes(min_rank=1, max_rank=1),
)

tensor_types_nd = tensor_types(dtypes=shared_dtypes, shapes=shapes())


@given(lhs_type=tensor_types_1d, rhs_type=tensor_types_1d)
def test_outer_valid(lhs_type: TensorType, rhs_type: TensorType) -> None:
    with Graph("outer", input_types=[lhs_type, rhs_type]) as graph:
        out = ops.outer(graph.inputs[0].tensor, graph.inputs[1].tensor)
        assert out.shape == [lhs_type.shape[0], rhs_type.shape[0]]
        graph.output(out)


@given(lhs_type=tensor_types_nd, rhs_type=tensor_types_nd)
def test_outer_nd_tensors(lhs_type: TensorType, rhs_type: TensorType) -> None:
    assume(lhs_type.rank != 1 or rhs_type.rank != 1)

    with Graph("outer", input_types=[lhs_type, rhs_type]) as graph:
        with pytest.raises(ValueError):
            out = ops.outer(graph.inputs[0].tensor, graph.inputs[1].tensor)
