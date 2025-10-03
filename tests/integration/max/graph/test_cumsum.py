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

import platform

import numpy as np
import pytest
import torch
from max.driver import Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_cumsum(session: InferenceSession, dtype: DType) -> None:
    if dtype == DType.bfloat16 and platform.machine() in ["arm64", "aarch64"]:
        pytest.skip("BF16 is not supported on ARM CPU architecture")

    input_type = TensorType(dtype, [1024], device=DeviceRef.CPU())

    with Graph(f"cumsum_{dtype}", input_types=[input_type]) as graph:
        out = ops.cumsum(graph.inputs[0].tensor, axis=0)
        graph.output(out.cast(DType.float32))

    model = session.load(graph)

    torch_dtype = torch.float32 if dtype == DType.float32 else torch.bfloat16
    input_data = torch.full((1024,), 1.1, dtype=torch_dtype)

    max_result = model(
        Tensor.from_dlpack(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(max_result, Tensor)
    max_result_np = max_result.to_numpy()

    torch_result = (
        torch.cumsum(input_data, dim=0).to(dtype=torch.float32).cpu().numpy()
    )

    np.testing.assert_allclose(
        max_result_np,
        torch_result,
        rtol=1e-6,
        atol=1e-6,
        verbose=True,
    )
