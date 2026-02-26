# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Test the scatter_nd_skip_oob_indices operation."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import scatter_nd_skip_oob_indices


@pytest.mark.parametrize(
    "input_data,updates_data,indices_data,expected",
    [
        # 1D scatter
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 15.0, 20.0],
            [[1], [-999], [3]],
            [1.0, 10.0, 3.0, 20.0, 5.0],
        ),
        # 1D scatter with negative indices
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 15.0, 20.0],
            [[-4], [999], [-2]],
            [1.0, 10.0, 3.0, 20.0, 5.0],
        ),
        # 2D scatter with 1D indices
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0],
            ],
            [[2], [999], [-999], [1]],
            [[1.0, 2.0, 3.0], [19.0, 20.0, 21.0], [10.0, 11.0, 12.0]],
        ),
        # 2D scatter_nd with 2D indices
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [10.0, 20.0, 30.0],
            [[0, -999], [1, 2], [-999, -999]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 20.0], [7.0, 8.0, 9.0]],
        ),
        # Empty updates
        (
            [1.0, 2.0, 3.0, 4.0],
            [],
            np.empty((0, 1), dtype=np.int32),
            [1.0, 2.0, 3.0, 4.0],
        ),
    ],
)
def test_scatter_nd_skip_oob_indices(
    input_data: list[float] | list[list[float]],
    updates_data: list[float] | list[list[float]],
    indices_data: list[int] | np.ndarray,
    expected: list[float] | list[list[float]],
) -> None:
    """Test scatter_nd_skip_oob_indices operation with various input configurations."""
    device = Accelerator() if accelerator_count() > 0 else CPU()
    device_ref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    input_array = np.array(input_data, dtype=np.float32)
    input_type = TensorType(DType.float32, input_array.shape, device_ref)

    with Graph(
        "my_scatter_nd_skip_oob_indices_graph", input_types=[input_type]
    ) as graph:
        input_val = graph.inputs[0].tensor
        updates = ops.constant(updates_data, DType.float32, device=device_ref)
        indices = ops.constant(indices_data, DType.int32, device=device_ref)
        out = scatter_nd_skip_oob_indices(input_val, updates, indices)
        graph.output(out)

    model = session.load(graph)
    input_tensor = Buffer.from_numpy(input_array).to(device)

    result = model.execute(input_tensor)[0]
    assert isinstance(result, Buffer)
    np.testing.assert_equal(
        result.to_numpy(), np.array(expected, dtype=np.float32)
    )
