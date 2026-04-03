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
"""Test nearest-neighbor resize execution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.mark.parametrize("device", [DeviceRef.CPU()])
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        ([1, 3, 8, 12], [1, 3, 16, 24]),
        ([2, 1, 10, 6], [2, 1, 5, 3]),
    ],
)
def test_resize_nearest_execution(
    session: InferenceSession,
    device: DeviceRef,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> None:
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    input_type = TensorType(
        dtype=DType.float32,
        shape=input_shape,
        device=device,
    )

    with Graph("test_resize_nearest", input_types=[input_type]) as graph:
        resized = ops.resize(
            graph.inputs[0].tensor,
            output_shape,
            interpolation=ops.InterpolationMode.NEAREST,
        )
        graph.output(resized)

    model = session.load(graph)

    np.random.seed(123)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(input_data),
        size=tuple(output_shape[-2:]),
        mode="nearest",
    ).numpy()

    result = model.execute(
        Buffer.from_numpy(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(result, Buffer)
    np.testing.assert_array_equal(result.to_numpy(), expected)
