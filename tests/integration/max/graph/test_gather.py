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

import numpy as np
import torch
from max.driver import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.graph.ops import gather


def test_gather(session) -> None:
    input_shape = [3, 2]
    index_shape = [5, 1]
    axis = 0
    with Graph(
        "conv2d",
        input_types=[
            TensorType(DType.int64, input_shape, device=DeviceRef.CPU()),
            TensorType(DType.int64, index_shape, device=DeviceRef.CPU()),
        ],
    ) as graph:
        input, index = graph.inputs

        output = gather(input.tensor, index.tensor, axis=axis)
        graph.output(output)

    model = session.load(graph)

    inputs = torch.Tensor([[0, 1], [2, 3], [4, 5]]).to(torch.int64)

    # Test 1: Valid gather and indices.
    index = torch.Tensor([[0], [1], [2], [1], [1]]).to(torch.int64)

    actual = model(
        Tensor.from_dlpack(inputs).to(model.input_devices[0]),
        Tensor.from_dlpack(index).to(model.input_devices[1]),
    )[0].to_numpy()
    expected = torch.take_along_dim(inputs, index, dim=0).numpy()
    np.testing.assert_equal(actual.reshape(5, 2), expected)

    # Test 2: Invalid indices
    # Should raise an error since indices must be between [0,3).
    index = torch.Tensor([[0], [1], [2], [3], [4]]).to(torch.int64)

    # TODO(GEX-1808): Uncomment the following line when calling the model raises
    # an error.
    # with pytest.raises(RuntimeError):
    actual = model(
        Tensor.from_dlpack(inputs).to(model.input_devices[0]),
        Tensor.from_dlpack(index).to(model.input_devices[1]),
    )[0].to_numpy()
