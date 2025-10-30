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
"""ops.argsort tests."""

import pytest
from conftest import GraphBuilder, shapes, tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, TensorType, ops

shared_shapes = st.shared(shapes(min_rank=1, max_rank=1))
supported_tensor_types = tensor_types(
    shapes=shared_shapes,
)


@given(
    input_type=supported_tensor_types,
)
def test_argsort_output_tensor_types(
    graph_builder: GraphBuilder,
    input_type: TensorType,
) -> None:
    expected_type = input_type.cast(DType.int64)
    with graph_builder(input_types=[input_type]) as graph:
        idx_tensor = ops.argsort(graph.inputs[0].tensor, ascending=True)
        assert idx_tensor.type == expected_type
        idx_tensor = ops.argsort(graph.inputs[0].tensor, ascending=False)
        assert idx_tensor.type == expected_type


def test_argsort_with_input_rank_greater_than_1(
    graph_builder: GraphBuilder,
) -> None:
    input_shape = [0, 1, 2, 3, 4, 5]
    ascending = True

    with graph_builder(
        input_types=[
            TensorType(
                dtype=DType.float32, shape=input_shape, device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match="argsort only implemented for input tensors of rank 1",
        ):
            ops.argsort(graph.inputs[0].tensor, ascending)
