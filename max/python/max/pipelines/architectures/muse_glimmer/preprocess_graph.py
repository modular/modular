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
"""Muse Glimmer image preprocessing as a CPU graph a serving engine runs.

The graph does for one image what :class:`MuseGlimmerImageProcessor` and
:func:`~.vision.data_processing.vision_inputs` do: it drops the alpha
channel, resizes the image with torch's antialiased uint8 Lanczos filter, bit
for bit, normalizes and patchifies it, and computes the tower's position
taps, window order, rotary tables and segment bounds.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.nn.legacy.graph_groups import (
    VisionHostMax,
    VisionRowIndex,
    VisionRows,
    VisionSegments,
)
from max.nn.legacy.layer import Module

from .image_processor import MuseGlimmerImageProcessor
from .model_config import MuseGlimmerVisionConfig

# torch's uint8 antialiased resize picks the most fraction bits, up to this
# many, that keep the largest weight within int16.
_MAX_PRECISION = 22


def _lanczos3(x: TensorValue) -> TensorValue:
    """Returns the three-lobe Lanczos filter at float64 ``x``."""
    x = ops.abs(x)
    angle = x * math.pi
    safe = ops.where(x == 0, 1.0, angle)
    sinc = ops.where(x == 0, 1.0, ops.sin(safe) / safe)
    third = safe / 3.0
    sinc3 = ops.where(x == 0, 1.0, ops.sin(third) / third)
    return ops.where(x < 3.0, sinc * sinc3, 0.0)


def _resize_axis(
    pixels: TensorValue, size: TensorValue, axis: int, extent: str
) -> TensorValue:
    """Resizes ``[height, width, 3]`` uint8 pixels along ``axis`` to ``size``.

    It is one pass of torch's antialiased uint8 Lanczos resize
    (``_compute_index_ranges_int16_weights``), as
    :func:`~.image_processor.lanczos_resize` mirrors it: int16 fixed-point
    weights at an adaptive precision, rounded and clamped to uint8.
    """
    cpu = DeviceRef.CPU()
    f64, i64 = DType.float64, DType.int64
    source = ops.shape_to_tensor([pixels.shape[axis]])[0]
    scale = source.cast(f64) / size.cast(f64)
    downscale = scale >= 1.0
    support = ops.where(downscale, scale * 3.0, 3.0)
    invscale = ops.where(downscale, 1.0 / scale, 1.0)
    taps = ops.ceil(support).cast(i64) * 2 + 1

    output = ops.range(0, size, out_dim=extent, dtype=i64, device=cpu)
    tap = ops.range(
        0, taps, out_dim=f"{extent}_taps", dtype=i64, device=cpu
    ).reshape([1, -1])
    center = (scale * (output.cast(f64) + 0.5)).reshape([-1, 1])
    first = ops.max((center - support + 0.5).cast(i64), 0)
    count = ops.min((center + support + 0.5).cast(i64), source) - first
    count = ops.min(ops.max(count, 0), taps)

    weights = _lanczos3(((tap + first).cast(f64) - center + 0.5) * invscale)
    weights = ops.where(tap < count, weights, 0.0)
    total = ops.sum(weights, axis=1)
    weights = ops.where(
        total != 0.0, weights / ops.where(total != 0.0, total, 1.0), 0.0
    )
    # The precision is the number of bit counts below _MAX_PRECISION at which
    # the largest weight, at one more bit, still rounds into int16.
    largest = ops.max(weights.reshape([-1]), axis=0)
    bits = ops.range(0, _MAX_PRECISION, dtype=i64, device=cpu)
    powers = ops.constant(
        np.exp2(np.arange(_MAX_PRECISION + 2)), f64, device=cpu
    )
    fits = ops.floor(0.5 + largest * ops.gather(powers, bits + 1, axis=0)) < (
        1 << 15
    )
    precision = ops.sum(fits.cast(i64), axis=0)
    scaled = weights * ops.gather(powers, precision, axis=0)
    fixed = ops.where(scaled < 0.0, scaled - 0.5, scaled + 0.5).cast(
        DType.int32
    )

    indices = ops.min(tap + first, source - 1)
    gathered = ops.gather(pixels, indices, axis=axis).cast(DType.int32)
    if axis == 1:
        weighted = gathered * fixed.reshape([1, output.shape[0], -1, 1])
    else:
        weighted = gathered * fixed.reshape([output.shape[0], -1, 1, 1])
    summed = ops.squeeze(ops.sum(weighted, axis=axis + 1), axis + 1)
    one = ops.gather(powers, precision, axis=0).cast(DType.int32)
    rounded = ops.floor_div(summed + one // 2, one)
    return ops.min(ops.max(rounded, 0), 255).cast(DType.uint8)


class MuseGlimmerImagePreprocess(Module):
    """Turns one RGBA8 image into the Muse Glimmer tower's inputs.

    Takes
    :func:`~max.nn.legacy.image_preprocess.image_preprocess_input_types`, the
    size being the tokenizer's ``smart_resize`` target. Returns the tower's
    named input tree for the image, on the CPU.
    """

    def __init__(
        self,
        processor: MuseGlimmerImageProcessor,
        vision: MuseGlimmerVisionConfig,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.patch_size = processor.patch_size
        self.merge_size = processor.merge_size
        self.temporal_patch_size = processor.temporal_patch_size
        levels = np.arange(256, dtype=np.float32)[:, None]
        # The processor's normalization of each level and channel, in its own
        # float32 arithmetic, which a table lookup reproduces exactly.
        self.levels = (
            levels * processor.rescale_factor - processor.image_mean
        ) / processor.image_std
        self.window = vision.window_size_patches
        self.side = vision.pos_emb_height
        half = vision.head_dim // 2
        self.inv_freq = 1.0 / vision.rope_theta ** (
            np.arange(0, half, 2) / half
        )
        self.dtype = dtype

    def __call__(
        self, image: TensorValue, size: TensorValue
    ) -> OrderedDict[str, object]:
        """Returns the tower's input tree for one image."""
        cpu = DeviceRef.CPU()
        f64, i64 = DType.float64, DType.int64
        patch, merge = self.patch_size, self.merge_size
        size = size.reshape([2])

        # The processor converts to RGB, which drops alpha rather than
        # compositing it; torch resizes the width first.
        pixels = image[0, :, :, :3]
        pixels = _resize_axis(pixels, size[1], 1, "resized_width")
        pixels = _resize_axis(pixels, size[0], 0, "resized_height")
        height, width = (
            Dim(side) // (patch * merge) * (patch * merge)
            for side in ("resized_height", "resized_width")
        )
        pixels = pixels.rebind(
            [height, width, 3],
            message=f"the resized sides must be multiples of {patch * merge}",
        )

        table = ops.constant(self.levels.reshape([-1]), DType.float32, cpu)
        channel = ops.range(0, 3, dtype=DType.int32, device=cpu)
        normalized = ops.gather(
            table, pixels.cast(DType.int32) * 3 + channel, axis=0
        )
        blocks_h, blocks_w = height // (patch * merge), width // (patch * merge)
        grid_h, grid_w = blocks_h * merge, blocks_w * merge
        seq = grid_h * grid_w
        # Rows run over the patches in raster order, each flattened
        # (temporal, channel, py, px); a still image fills every temporal
        # slot.
        temporal = self.temporal_patch_size
        patches = normalized.reshape([grid_h, patch, grid_w, patch, 3]).permute(
            [0, 2, 4, 1, 3]
        )
        patches = ops.broadcast_to(
            patches.reshape([grid_h, grid_w, 1, 3, patch, patch]),
            [grid_h, grid_w, temporal, 3, patch, patch],
        )
        pixel_values = patches.reshape(
            [seq, temporal * 3 * patch * patch]
        ).cast(self.dtype)

        h = ops.shape_to_tensor([grid_h])[0]
        w = ops.shape_to_tensor([grid_w])[0]
        rows = ops.range(0, grid_h, 1, dtype=i64, device=cpu)
        cols = ops.range(0, grid_w, 1, dtype=i64, device=cpu)

        # Bilinear taps into the position table, zero outside it.
        side = self.side

        def axis_taps(
            index: TensorValue, extent: TensorValue
        ) -> tuple[TensorValue, TensorValue]:
            source = (index.cast(f64) + 0.5) * float(side) / extent.cast(
                f64
            ) - 0.5
            raw = ops.floor(source).cast(i64).reshape([-1, 1]) + ops.range(
                0, 2, dtype=i64, device=cpu
            ).reshape([1, 2])
            weight = 1.0 - ops.abs(source.reshape([-1, 1]) - raw.cast(f64))
            weight = ops.where((raw >= 0) & (raw <= side - 1), weight, 0.0)
            return ops.min(ops.max(raw, 0), side - 1), weight

        row_taps, row_weights = axis_taps(rows, h)
        col_taps, col_weights = axis_taps(cols, w)
        interp_idx = (
            (
                row_taps.reshape([grid_h, 1, 2, 1]) * side
                + col_taps.reshape([1, grid_w, 1, 2])
            )
            .reshape([seq, 4])
            .cast(DType.int32)
        )
        interp_w = (
            (
                row_weights.reshape([grid_h, 1, 2, 1])
                * col_weights.reshape([1, grid_w, 1, 2])
            )
            .reshape([seq, 4])
            .cast(DType.float32)
        )

        # Windows of window x window patches, row-major over the image and
        # clipped at its edge, each in raster order. For each window-major
        # position, find the raster patch there.
        win = self.window
        order = ops.range(0, seq, 1, dtype=i64, device=cpu)
        band = order // (win * w)
        band_h = ops.min(h - band * win, win)
        in_band = order - band * win * w
        column = in_band // (band_h * win)
        column_w = ops.min(w - column * win, win)
        in_window = in_band - column * band_h * win
        row = band * win + in_window // column_w
        col = column * win + in_window % column_w
        window_index = row * w + col

        # Raster patch to window-major position.
        r = ops.broadcast_to(rows.reshape([grid_h, 1]), [grid_h, grid_w])
        c = ops.broadcast_to(cols.reshape([1, grid_w]), [grid_h, grid_w])
        r_band_h = ops.min(h - r // win * win, win)
        c_window_w = ops.min(w - c // win * win, win)
        reverse_index = (
            r // win * win * w
            + r_band_h * (c // win) * win
            + r % win * c_window_w
            + c % win
        ).reshape([seq])

        # Each merge x merge block as consecutive rows.
        block_rows = ops.range(0, blocks_h, 1, dtype=i64, device=cpu)
        block_cols = ops.range(0, blocks_w, 1, dtype=i64, device=cpu)
        inner = ops.range(0, merge, 1, dtype=i64, device=cpu)
        pixel_shuffle_index = ops.broadcast_to(
            (
                block_rows.reshape([blocks_h, 1, 1, 1]) * merge
                + inner.reshape([1, 1, merge, 1])
            )
            * w
            + block_cols.reshape([1, blocks_w, 1, 1]) * merge
            + inner.reshape([1, 1, 1, merge]),
            [blocks_h, blocks_w, merge, merge],
        ).reshape([seq])

        # Axial rotary angles at (col + 1, row + 1), in window-major order.
        positions = ops.stack([col + 1, row + 1], axis=1).cast(f64)
        inv_freq = ops.constant(self.inv_freq, f64, cpu)
        angles = (
            positions.reshape([seq, 2, 1]) * inv_freq.reshape([1, 1, -1])
        ).reshape([seq, -1])
        angles = ops.concat([angles, angles], axis=1)
        rot_cos = ops.cos(angles).cast(DType.float32)
        rot_sin = ops.sin(angles).cast(DType.float32)

        # Window lengths, row-major.
        bands = ops.range(
            0,
            (h + win - 1) // win,
            out_dim="window_bands",
            dtype=i64,
            device=cpu,
        )
        columns = ops.range(
            0,
            (w + win - 1) // win,
            out_dim="window_columns",
            dtype=i64,
            device=cpu,
        )
        lengths = (
            ops.min(h - bands * win, win).reshape([-1, 1])
            * ops.min(w - columns * win, win).reshape([1, -1])
        ).reshape([-1])
        cu_window_seqlens = ops.concat(
            [
                ops.constant(0, i64, cpu).reshape([1]),
                ops.cumsum(lengths, axis=0),
            ]
        ).cast(DType.uint32)
        max_window_seqlen = ops.min(h, win) * ops.min(w, win)

        return OrderedDict(
            patches=VisionRows(
                pixel_values=pixel_values,
                interp_idx=interp_idx,
                interp_w=interp_w,
            ),
            window_order=VisionRowIndex(
                window_index=window_index,
                reverse_index=reverse_index,
                pixel_shuffle_index=pixel_shuffle_index,
            ),
            rotary=VisionRows(rot_cos=rot_cos, rot_sin=rot_sin),
            segments=VisionSegments(
                cu_seqlens=ops.shape_to_tensor([0, seq]).cast(DType.uint32),
                cu_window_seqlens=cu_window_seqlens,
            ),
            segment_bound=VisionHostMax(
                max_seqlen=ops.shape_to_tensor([seq])
                .reshape([])
                .cast(DType.uint32),
                max_window_seqlen=max_window_seqlen.reshape([]).cast(
                    DType.uint32
                ),
            ),
        )
