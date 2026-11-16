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
"""Host-side image preparation for the vision tower."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.pipelines.context import ImageMetadata

from .model_config import Inkling2VisionConfig

_INTERP_TAPS = 4
"""``mode="bilinear"`` emits a 2x2 separable stencil per patch, i.e. four
taps into the flattened position grid."""

_IMAGE_MEAN = (0.5, 0.5, 0.5)
_IMAGE_STD = (0.5, 0.5, 0.5)
"""The reference image processor's normalization constants: its class
defaults, which the fixture's ``processor_config.json`` carries too."""

_RAW_RESCALE = np.float32(1.0 / 255.0)
"""``rescale_factor`` the reference applies before normalizing. A block
that arrives as integers is on that 0-255 scale."""


def normalize_image(
    block: npt.NDArray[np.generic] | Sequence[Sequence[Sequence[object]]],
) -> npt.NDArray[np.float32]:
    """Puts one image's patch block on the reference's ``[-1, 1]`` scale."""
    pixels = np.asarray(block)
    if np.issubdtype(pixels.dtype, np.integer):
        if pixels.size and int(pixels.max()) > 255:
            raise ValueError(
                "image pixels are integer but exceed 255 "
                f"(max {int(pixels.max())}); a uint16 block may be bfloat16 "
                "BITS rather than pixels. Hand over 0-255 integers or a float "
                "block already in [-1, 1]."
            )
        scaled = pixels.astype(np.float32) * _RAW_RESCALE
    elif np.issubdtype(pixels.dtype, np.floating):
        scaled = pixels.astype(np.float32)
        if scaled.size:
            low = float(scaled.min())
            high = float(scaled.max())
            if low < -1.0 - 1e-3 or high > 1.0 + 1e-3:
                raise ValueError(
                    "image pixels arrive as float outside [-1, 1] "
                    f"(min {low:.4g}, max {high:.4g}). The Inkling2 tower "
                    "takes pixels already normalized to [-1, 1]; rescale this "
                    "block in the image processor instead of relying on a "
                    "guess here."
                )
        return scaled
    else:
        raise ValueError(
            f"unsupported image pixel dtype {pixels.dtype}; expected an "
            "integer 0-255 block or a float block already in [-1, 1]"
        )

    mean = np.asarray(_IMAGE_MEAN, dtype=np.float32).reshape(-1)
    std = np.asarray(_IMAGE_STD, dtype=np.float32).reshape(-1)
    if scaled.shape[-1] != mean.size:
        raise ValueError(
            f"image block has {scaled.shape[-1]} channels but image_mean has "
            f"{mean.size} entries"
        )
    return (scaled - mean) / std


def _interp_axis_taps(
    positions: npt.NDArray[np.int64], size: int, side: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    source = (positions.astype(np.float64) + 0.5) * side / size - 0.5
    floor = np.floor(source)
    offsets = np.arange(2, dtype=np.float64)
    raw_taps = floor[:, None] + offsets
    taps = np.clip(raw_taps, 0, side - 1).astype(np.int64)
    distance = np.abs(source[:, None] - floor[:, None] - offsets)
    weights = np.clip(1.0 - distance, 0.0, None)
    return taps, weights


def patch_grid_axes(
    grid: tuple[int, int], merge: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(row, col)`` patch coordinate of every patch, in merge-block order."""
    rows, columns = grid
    patch_rows, patch_columns = rows * merge, columns * merge
    within = np.arange(patch_rows * patch_columns, dtype=np.int64)
    in_column = within % merge
    in_row = (within // merge) % merge
    block_column = (within // (merge * merge)) % columns
    block_row = within // (merge * merge * columns)
    return block_row * merge + in_row, block_column * merge + in_column


def interpolation_taps(
    grid: tuple[int, int], merge: int, side: int
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """The ``[num_patches, 4]`` gather taps and weights into the position grid."""
    row, column = patch_grid_axes(grid, merge)
    patch_rows, patch_columns = grid[0] * merge, grid[1] * merge
    row_taps, row_weights = _interp_axis_taps(row, patch_rows, side)
    column_taps, column_weights = _interp_axis_taps(column, patch_columns, side)
    indices = (row_taps[:, :, None] * side + column_taps[:, None, :]).reshape(
        -1, _INTERP_TAPS
    )
    weights = (row_weights[:, :, None] * column_weights[:, None, :]).reshape(
        -1, _INTERP_TAPS
    )
    return indices.astype(np.int32), weights.astype(np.float32)


def vision_position_ids(
    grid: tuple[int, int], merge: int
) -> npt.NDArray[np.int32]:
    """The ``[1 + num_patches, 2]`` ``(h, w)`` coordinates the tower's rope takes."""
    row, column = patch_grid_axes(grid, merge)
    patches = np.stack([row, column], axis=-1) + 1
    return np.concatenate(
        [np.zeros((1, 2), dtype=np.int64), patches], axis=0
    ).astype(np.int32)


def merge_grid(
    image: object, block_ndim: int, block_shape: tuple[int, ...]
) -> tuple[int, int]:
    """The image's ``(rows, columns)`` in merge tokens."""
    if block_ndim == 5:
        return int(block_shape[0]), int(block_shape[1])
    grid = getattr(image, "image_grid_thw", None)
    if grid is None:
        raise ValueError(
            "cannot determine an image's patch grid: its pixel block is "
            f"[{', '.join(str(d) for d in block_shape)}] and it carries no "
            "`image_grid_thw`. Emit one block per merge row pair "
            "([rows, columns, side, side, channels]) or attach "
            "`image_grid_thw = [1, rows, columns]` (merge tokens) to the "
            "image metadata; a token count alone is ambiguous (4x6 and 3x8 "
            "hold the same number of tokens)."
        )
    values = np.asarray(grid).reshape(-1)
    if values.size != 3:
        raise ValueError(
            "image_grid_thw must be [temporal, rows, columns]; got "
            f"{values.tolist()}"
        )
    temporal, rows, columns = (int(value) for value in values)
    if temporal != 1:
        raise ValueError(
            "one image per entry: image_grid_thw's temporal axis must be 1, "
            f"got {temporal}"
        )
    return rows, columns


def tower_rows(result: object) -> Buffer:
    """The device buffer a compiled ModuleV3 vision callable returned."""
    if isinstance(result, Buffer):
        return result
    tensor = getattr(result, "driver_tensor", None)
    if isinstance(tensor, Buffer):
        return tensor
    if isinstance(result, (tuple, list)) and result:
        return tower_rows(result[0])
    raise TypeError(
        "the vision tower returned "
        f"{type(result).__name__}, not a Buffer or a Tensor"
    )


def concat_rows(rows: Sequence[Buffer], device: Device) -> Buffer:
    """Concatenates embedding rows on the device, without a host round-trip."""
    if len(rows) == 1:
        return rows[0]
    total = sum(int(row.shape[0]) for row in rows)
    combined = Buffer(
        shape=(total, *rows[0].shape[1:]),
        dtype=rows[0].dtype,
        device=device,
    )
    # MAX requires one index per rank, so both slices are full-rank: the
    # leading axis is the row range, the rest is everything.
    rest = (slice(None),) * (len(rows[0].shape) - 1)
    start = 0
    for row in rows:
        stop = start + int(row.shape[0])
        combined[(slice(start, stop), *rest)].inplace_copy_from(row)
        start = stop
    return combined


def prepare_tower_inputs(
    image: ImageMetadata,
    vision_config: Inkling2VisionConfig,
    device: Device,
) -> tuple[Buffer, Buffer, Buffer, Buffer]:
    """One image's four tower operands, prepared on the host, on ``device``."""
    merge = vision_config.spatial_merge_size
    side = vision_config.patch_size * merge
    block = np.asarray(image.pixel_values)
    # A block that carries its own (rows, columns) axes keeps them only long
    # enough to read the grid; the tower's row axis is the flattened one.
    block_shape = tuple(block.shape)
    if block.ndim == 5:
        block = block.reshape(block.shape[0] * block.shape[1], *block.shape[2:])
    if block.ndim != 4:
        raise ValueError(
            "an image's pixel block must be "
            f"[num_tokens, {side}, {side}, {vision_config.num_channels}] or "
            f"[rows, columns, {side}, {side}, {vision_config.num_channels}]; "
            f"got {block.shape}"
        )
    if block.shape[1:] != (side, side, vision_config.num_channels):
        raise ValueError(
            "an image's pixel block must carry one "
            f"{side}x{side}x{vision_config.num_channels} merge block per row "
            f"(patch_size {vision_config.patch_size} x spatial_merge_size "
            f"{merge}); got {block.shape[1:]}"
        )
    grid = merge_grid(image, len(block_shape), block_shape)
    num_tokens = int(block.shape[0])
    if grid[0] * grid[1] != num_tokens:
        raise ValueError(
            f"the image's grid is {grid[0]}x{grid[1]} merge tokens but its "
            f"pixel block holds {num_tokens} rows"
        )
    expected_rows = getattr(image, "embedding_rows", None)
    if expected_rows is not None and int(expected_rows) != num_tokens:
        raise ValueError(
            f"the image's placeholder run covers {int(expected_rows)} "
            f"tokens but the tower would emit {num_tokens} rows"
        )

    pixels = normalize_image(block)
    interp_indices, interp_weights = interpolation_taps(
        grid, merge, vision_config.posemb_grid_size
    )
    position_ids = vision_position_ids(grid, merge)
    return (
        Buffer.from_numpy(np.ascontiguousarray(pixels)).to(device),
        Buffer.from_numpy(interp_indices).to(device),
        Buffer.from_numpy(interp_weights).to(device),
        Buffer.from_numpy(position_ids).to(device),
    )
