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
"""Test resize_bicubic operation execution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from max.driver import Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray
from torchvision.transforms.functional import InterpolationMode


def torchvision_bicubic_resize_nchw(
    input_array: NDArray[np.float32], height: int, width: int
) -> NDArray[np.float32]:
    """Reference implementation using torchvision bicubic resize for NCHW format.

    Args:
        input_array: Input array of shape (batch, channels, height, width)
        height: Target height
        width: Target width

    Returns:
        Resized array using torchvision's bicubic interpolation in NCHW format
    """
    # Convert numpy array to torch tensor
    input_tensor = torch.from_numpy(input_array)

    # Create resize transform
    resize_transform = T.Resize(
        (height, width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=False,
    )

    # Apply resize to each image in the batch
    batch_size = input_tensor.shape[0]
    output_list = []
    for b in range(batch_size):
        resized = resize_transform(input_tensor[b])
        output_list.append(resized)

    # Stack results back into a batch
    output_tensor = torch.stack(output_list)

    # Convert back to numpy
    return output_tensor.numpy()


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        # Upscale 2x (NCHW format)
        ([1, 3, 224, 224], [1, 3, 448, 448]),
        # Downscale 2x
        ([1, 3, 448, 448], [1, 3, 224, 224]),
        # Non-square input
        ([1, 3, 336, 224], [1, 3, 448, 448]),
        # Different channels
        ([2, 1, 128, 128], [2, 1, 256, 256]),
    ],
)
def test_resize_bicubic_execution(
    session: InferenceSession,
    device: DeviceRef,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> None:
    """Test resize_bicubic compilation and execution."""
    # Skip GPU tests if no GPU is available
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")
    # Phase 1: Graph construction
    input_type = TensorType(
        dtype=DType.float32, shape=input_shape, device=device
    )

    with Graph("test_resize_bicubic", input_types=[input_type]) as graph:
        # Resize operates on NCHW format directly
        resized = ops.resize(
            graph.inputs[0].tensor,
            output_shape,
            interpolation=ops.InterpolationMode.BICUBIC,
        )
        graph.output(resized)

    # Phase 2: Compilation
    model = session.load(graph)

    # Phase 3: Execution
    # Create test input (NCHW format)
    np.random.seed(42)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    # Get reference output from torchvision in NCHW format
    _, _, out_h, out_w = output_shape
    expected = torchvision_bicubic_resize_nchw(input_data, out_h, out_w)

    # Execute MAX graph
    result = model.execute(
        Tensor.from_numpy(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(result, Tensor)
    result_np = result.to_numpy()

    # Verify shape and values
    assert result_np.shape == tuple(output_shape)
    # Torchvision and our implementation may have small differences
    np.testing.assert_allclose(result_np, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
def test_resize_bicubic_identity(
    session: InferenceSession, device: DeviceRef
) -> None:
    """Test identity transformation (same size)."""
    # Skip GPU tests if no GPU is available
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")
    shape = [1, 3, 256, 256]  # NCHW format

    input_type = TensorType(dtype=DType.float32, shape=shape, device=device)

    with Graph(
        "test_resize_bicubic_identity", input_types=[input_type]
    ) as graph:
        # Resize to same shape
        resized = ops.resize(
            graph.inputs[0].tensor,
            shape,
            interpolation=ops.InterpolationMode.BICUBIC,
        )
        graph.output(resized)

    # Compile and execute
    model = session.load(graph)

    input_data = np.random.rand(*shape).astype(np.float32)
    result = model.execute(
        Tensor.from_numpy(input_data).to(model.input_devices[0])
    )[0]

    # For identity transformation, output should match input closely
    assert isinstance(result, Tensor)
    np.testing.assert_allclose(
        result.to_numpy(), input_data, rtol=1e-5, atol=1e-6
    )
