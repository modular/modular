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
"""Test resize_nearest operation execution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from numpy.typing import NDArray


def nearest_reference(
    input_array: NDArray[np.float32], output_shape: Sequence[int]
) -> NDArray[np.float32]:
    """Reference nearest-neighbor resize with half_pixel mapping.

    Mirrors the kernel's default modes: ``half_pixel`` coordinate transform
    (``(out + 0.5) / scale - 0.5``) and ``HalfDown`` rounding
    (``ceil(x - 0.5)``), with the index clamped to the last input element.
    """
    indices = []
    for in_dim, out_dim in zip(input_array.shape, output_shape, strict=True):
        scale = np.float32(np.float64(out_dim) / np.float64(in_dim))
        coords = np.arange(out_dim, dtype=np.float32) + np.float32(0.5)
        coords = coords / scale - np.float32(0.5)
        index = np.ceil(coords - np.float32(0.5)).astype(np.int64)
        indices.append(np.minimum(index, in_dim - 1))

    return input_array[np.ix_(*indices)]


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        # Upscale 2x (NCHW format).
        ([1, 3, 32, 32], [1, 3, 64, 64]),
        # Downscale 2x.
        ([1, 3, 64, 64], [1, 3, 32, 32]),
        # Non-square input.
        ([1, 3, 16, 32], [1, 3, 64, 64]),
        # Resize batch and channel dims as well.
        ([2, 1, 8, 8], [4, 2, 16, 16]),
        # Identity.
        ([1, 3, 32, 32], [1, 3, 32, 32]),
    ],
)
def test_resize_nearest_execution(
    session: InferenceSession,
    device: DeviceRef,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> None:
    """Test resize with NEAREST interpolation compiles and executes."""
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    input_type = TensorType(
        dtype=DType.float32, shape=input_shape, device=device
    )

    with Graph("test_resize_nearest", input_types=[input_type]) as graph:
        graph.output(
            ops.resize(
                graph.inputs[0].tensor,
                output_shape,
                interpolation=ops.InterpolationMode.NEAREST,
            )
        )

    model = session.load(graph)

    rng = np.random.default_rng(42)
    input_data = rng.random(input_shape, dtype=np.float32)
    expected = nearest_reference(input_data, output_shape)

    result = model.execute(
        Buffer.from_numpy(input_data).to(model.input_devices[0])
    )[0]
    assert isinstance(result, Buffer)
    result_np = result.to_numpy()

    assert result_np.shape == tuple(output_shape)
    np.testing.assert_array_equal(result_np, expected)
