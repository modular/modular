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
"""ops.top_k tests."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops


def test_top_k_output_tensor_types() -> None:
    k = 4

    expected_weight_tensor_type = TensorType(
        dtype=DType.float32, shape=[k], device=DeviceRef.CPU()
    )
    expected_idx_tensor_type = TensorType(
        dtype=DType.int64, shape=[k], device=DeviceRef.CPU()
    )

    with Graph(
        "top_k",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=[6],
                device=DeviceRef.CPU(),
            )
        ],
    ) as graph:
        weight_tensor, idx_tensor = ops.top_k(graph.inputs[0].tensor, k)

        assert weight_tensor.type == expected_weight_tensor_type
        assert idx_tensor.type == expected_idx_tensor_type


def test_top_k_with_axis_greater_than_input_rank() -> None:
    input_shape = [0, 1, 2, 3, 4, 5]
    k = 4

    axis_greater_than_input_rank = len(input_shape) + 1

    with Graph(
        "top_k",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=input_shape, device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        with pytest.raises(ValueError, match=r"axis \(\d+\) out of bound"):
            ops.top_k(
                graph.inputs[0].tensor, k, axis=axis_greater_than_input_rank
            )


def test_top_k_with_k_greater_than_input_length() -> None:
    input_shape = [6]
    k_greater_than_input_length = input_shape[-1] + 1

    with Graph(
        "top_k",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=input_shape, device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match=r"k \(\d+\) cannot be larger than dimension size along axis \d+",
        ):
            ops.top_k(graph.inputs[0].tensor, k_greater_than_input_length)
