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
"""Tests that resize_nearest runs on CPU and GPU and picks the right indices."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.ops import CoordinateTransformMode, RoundMode
from numpy.typing import NDArray

# 4 -> 6 on height and 4 -> 2 on width both produce exact .5 coordinates, so
# every round mode is exercised on its tie case, and half_pixel with floor
# maps the first output row below zero.
INPUT_SHAPE = [1, 2, 4, 4]
OUTPUT_SHAPE = [1, 2, 6, 2]


def _source_index(
    out: int,
    in_dim: int,
    out_dim: int,
    coord_mode: CoordinateTransformMode,
    round_mode: RoundMode,
) -> int:
    """Replays the kernel's float32 arithmetic for one output coordinate."""
    f32 = np.float32
    scale = f32(np.float64(out_dim) / np.float64(in_dim))
    if coord_mode == CoordinateTransformMode.HALF_PIXEL or (
        coord_mode == CoordinateTransformMode.HALF_PIXEL_1D and out_dim != 1
    ):
        coord = (f32(out) + f32(0.5)) / scale - f32(0.5)
    elif coord_mode == CoordinateTransformMode.ALIGN_CORNERS and out_dim != 1:
        coord = f32(out) * f32(np.float64(in_dim - 1) / np.float64(out_dim - 1))
    elif coord_mode == CoordinateTransformMode.ASYMMETRIC:
        coord = f32(out) / scale
    else:
        coord = f32(0)

    if round_mode == RoundMode.HALF_DOWN:
        rounded = np.ceil(coord - f32(0.5))
    elif round_mode == RoundMode.HALF_UP:
        rounded = np.floor(coord + f32(0.5))
    elif round_mode == RoundMode.FLOOR:
        rounded = np.floor(coord)
    else:
        rounded = np.ceil(coord)
    return int(min(max(int(rounded), 0), in_dim - 1))


def _expected(
    data: NDArray[np.float32],
    out_shape: list[int],
    coord_mode: CoordinateTransformMode,
    round_mode: RoundMode,
) -> NDArray[np.float32]:
    in_shape = data.shape
    rows = [
        _source_index(i, in_shape[2], out_shape[2], coord_mode, round_mode)
        for i in range(out_shape[2])
    ]
    cols = [
        _source_index(j, in_shape[3], out_shape[3], coord_mode, round_mode)
        for j in range(out_shape[3])
    ]
    return data[:, :, rows, :][:, :, :, cols]


def _check(
    session: InferenceSession,
    device: DeviceRef,
    in_shape: list[int],
    out_shape: list[int],
    coord_mode: CoordinateTransformMode,
    round_mode: RoundMode,
) -> None:
    if device.device_type == "gpu" and accelerator_count() == 0:
        pytest.skip("No GPU available")

    input_type = TensorType(DType.float32, in_shape, device=device)
    with Graph("test_resize_nearest", input_types=[input_type]) as graph:
        graph.output(
            ops.resize_nearest(
                graph.inputs[0].tensor,
                out_shape,
                coordinate_transform_mode=coord_mode,
                round_mode=round_mode,
            )
        )
    model = session.load(graph)

    # Distinct values, so a wrong source index changes the output.
    data = np.arange(np.prod(in_shape), dtype=np.float32).reshape(in_shape)
    result = model.execute(Buffer.from_numpy(data).to(model.input_devices[0]))[
        0
    ]
    assert isinstance(result, Buffer)
    assert result.device == model.input_devices[0]
    np.testing.assert_array_equal(
        result.to_numpy(), _expected(data, out_shape, coord_mode, round_mode)
    )


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize("coord_mode", list(CoordinateTransformMode))
@pytest.mark.parametrize("round_mode", list(RoundMode))
def test_resize_nearest(
    session: InferenceSession,
    device: DeviceRef,
    coord_mode: CoordinateTransformMode,
    round_mode: RoundMode,
) -> None:
    _check(session, device, INPUT_SHAPE, OUTPUT_SHAPE, coord_mode, round_mode)


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
@pytest.mark.parametrize(
    "coord_mode, round_mode",
    [
        (CoordinateTransformMode.HALF_PIXEL, RoundMode.HALF_DOWN),
        (CoordinateTransformMode.HALF_PIXEL, RoundMode.FLOOR),
    ]
    + [
        (CoordinateTransformMode.ALIGN_CORNERS, RoundMode.HALF_UP),
        (CoordinateTransformMode.ASYMMETRIC, RoundMode.CEIL),
    ],
)
def test_resize_nearest_full_vectors(
    session: InferenceSession,
    device: DeviceRef,
    coord_mode: CoordinateTransformMode,
    round_mode: RoundMode,
) -> None:
    """The output width is a multiple of the kernel's vector width."""
    _check(session, device, [1, 2, 3, 5], [1, 2, 7, 16], coord_mode, round_mode)
