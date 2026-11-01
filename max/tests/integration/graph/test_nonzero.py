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
"""Tests for ops.nonzero."""

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

_CASES = [
    pytest.param(np.empty((0, 3), dtype=np.float32), DType.float32, id="empty"),
    pytest.param(
        np.ones((2, 3), dtype=np.int32), DType.int32, id="all-nonzero"
    ),
    pytest.param(
        np.array([0, 4, 0, -2, 0], dtype=np.int64),
        DType.int64,
        id="rank-1-mixed",
    ),
    pytest.param(
        np.array([[0.0, -0.0, np.nan], [2.0, 0.0, -3.0]], dtype=np.float32),
        DType.float32,
        id="nan-negative-zero",
    ),
    pytest.param(
        np.array(
            [[[0, 1], [2, 0]], [[0, 0], [3, 4]]],
            dtype=np.uint8,
        ),
        DType.uint8,
        id="rank-3-ordering",
    ),
    pytest.param(
        np.array([[False, True], [True, False]], dtype=np.bool_),
        DType.bool,
        id="bool",
    ),
]


@pytest.mark.parametrize("device_name", ["cpu", "gpu"])
@pytest.mark.parametrize(("input_data", "dtype"), _CASES)
def test_nonzero(
    session: InferenceSession,
    device_name: str,
    input_data: np.ndarray,
    dtype: DType,
) -> None:
    """Returns exact row-major coordinates and data-dependent output shapes."""
    if device_name == "gpu" and accelerator_count() == 0:
        pytest.skip("requires a GPU")

    device = DeviceRef.CPU() if device_name == "cpu" else DeviceRef.GPU()
    symbolic_shape = [f"dim_{axis}" for axis in range(input_data.ndim)]
    with Graph(
        f"nonzero_{device_name}_{dtype.name}_{input_data.ndim}d",
        input_types=[TensorType(dtype, symbolic_shape, device=device)],
    ) as graph:
        result = ops.nonzero(graph.inputs[0].tensor, out_dim="nnz")
        graph.output(result)

    model = session.load(graph)
    input_buffer = Buffer.from_numpy(input_data).to(model.input_devices[0])
    output = model.execute(input_buffer)[0]
    assert isinstance(output, Buffer)
    assert output.device == model.input_devices[0]
    expected = np.argwhere(input_data != 0).astype(np.int64)
    assert output.shape == expected.shape
    np.testing.assert_array_equal(output.to_numpy(), expected)
