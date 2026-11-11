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
"""Inkling image preprocessing as a CPU graph."""

import math
from collections import OrderedDict

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.legacy.graph_groups import VisionRows
from max.nn.legacy.image_preprocess import composite_on_white
from max.nn.legacy.layer import Module


def _resize_axis(
    pixels: TensorValue, size: TensorValue, axis: int
) -> TensorValue:
    """Applies Pillow's seven-tap, fixed-point Lanczos upscale on one axis."""
    cpu = DeviceRef.CPU()
    source = ops.shape_to_tensor([pixels.shape[axis]])[0]
    extent = "resized_width" if axis == 1 else "resized_height"
    output = ops.range(0, size, out_dim=extent, dtype=DType.int64, device=cpu)
    center = (output.cast(DType.float64) + 0.5) * (
        source.cast(DType.float64) / size.cast(DType.float64)
    )
    taps = ops.range(-3, 4, dtype=DType.int64, device=cpu)
    indices = center.cast(DType.int64).reshape([-1, 1]) + taps.reshape([1, 7])
    distance = indices.cast(DType.float64) - center.reshape([-1, 1]) + 0.5
    angle = distance * math.pi
    safe_angle = ops.where(distance == 0, 1.0, angle)
    coefficients = ops.where(
        distance == 0,
        1.0,
        ops.sin(angle) * ops.sin(angle / 3.0) / (safe_angle * safe_angle / 3.0),
    )
    valid = (indices >= 0) & (indices < source) & (ops.abs(distance) < 3.0)
    coefficients = ops.where(valid, coefficients, 0.0)
    coefficients = coefficients / ops.sum(coefficients, axis=1)
    # Pillow quantizes coefficients to signed 22-bit fixed point and rounds
    # and clips after each pass, including the intermediate horizontal image.
    coefficients = (
        coefficients * (1 << 22)
        + ops.where(
            coefficients < 0, ops.constant(-0.5, DType.float64, cpu), 0.5
        )
    ).cast(DType.int32)
    indices = ops.min(ops.max(indices, 0), source - 1)
    gathered = ops.gather(pixels.cast(DType.int32), indices, axis=axis)
    if axis == 1:
        weighted = gathered * coefficients.reshape([1, output.shape[0], 7, 1])
    else:
        weighted = gathered * coefficients.reshape([output.shape[0], 7, 1, 1])
    summed = ops.squeeze(ops.sum(weighted, axis=axis + 1), axis + 1)
    rounded = (summed + (1 << 21)) // (1 << 22)
    return ops.min(ops.max(rounded, 0), 255).cast(DType.uint8)


class InklingImagePreprocess(Module):
    """Composites on white, upscales with Lanczos and pads patches with -1."""

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def __call__(
        self, image: TensorValue, size: TensorValue
    ) -> OrderedDict[str, object]:
        """Returns one image's named vision inputs, matching the image processor."""
        cpu = DeviceRef.CPU()
        pixels = composite_on_white(image)[0]
        source = ops.shape_to_tensor(pixels.shape[:2])
        long_edge = ops.max(source[0], source[1]).cast(DType.float64)
        scale = ops.min(long_edge * 2.0, ops.max(long_edge, 2048.0)) / long_edge
        resized = ops.floor(source.cast(DType.float64) * scale + 0.5).cast(
            DType.int64
        )
        pixels = _resize_axis(pixels, resized[1], axis=1)
        pixels = _resize_axis(pixels, resized[0], axis=0)
        patch = self.patch_size
        rows = ops.range(
            0,
            size[0] // patch,
            out_dim="patch_rows",
            dtype=DType.int64,
            device=cpu,
        )
        cols = ops.range(
            0,
            size[1] // patch,
            out_dim="patch_cols",
            dtype=DType.int64,
            device=cpu,
        )
        inside = ops.range(0, patch, dtype=DType.int64, device=cpu)
        y = (rows.reshape([-1, 1]) * patch + inside).reshape([-1, 1, patch, 1])
        x = (cols.reshape([-1, 1]) * patch + inside).reshape([1, -1, 1, patch])
        valid = (y < resized[0]) & (x < resized[1])
        indices = ops.min(y, resized[0] - 1) * resized[1] + ops.min(
            x, resized[1] - 1
        )
        values = ops.gather(pixels.reshape([-1, 3]), indices, axis=0).cast(
            DType.int32
        )
        patches = ops.where(ops.unsqueeze(valid, -1), values, -1).reshape(
            [rows.shape[0] * cols.shape[0], patch, patch, 3]
        )
        return OrderedDict(patches=VisionRows(pixels=patches))
