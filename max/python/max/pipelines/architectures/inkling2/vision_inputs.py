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
"""Host-side index and table inputs of the Inkling 2 vision tower.

Each image token covers a ``merge x merge`` square of native patches, so the
tower sees ``merge**2`` patches per image token. Within an image the patches
run block by block (one image token's patches are consecutive), and the tower
sequence of an image is its class row followed by those patches.

The graph in :mod:`.preprocess_graph` builds the same tables for one image.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class VisionInputs(NamedTuple):
    """Every vision tower input, in the order of the tower's input tree."""

    interp_idx: npt.NDArray[np.int32]
    """``[N, 4]`` bilinear taps of each patch into the position table."""
    interp_w: npt.NDArray[np.float32]
    """``[N, 4]`` tap weights."""
    positions: npt.NDArray[np.int32]
    """``[N + B, 2]`` the ``(row, col)`` of each sequence row for the rotary
    angles: the patches at their 1-indexed position, each class row at
    ``(0, 0)``."""
    pixel_values: npt.NDArray[np.int16]
    """``[tokens, side, side, 3]`` pixel blocks."""
    cu_seqlens: npt.NDArray[np.uint32]
    """``[B + 1]`` the start of each image's patches, class rows excluded. The
    engine's packer offsets the bounds of an image by the patch rows before
    it, so they count patches; the tower adds one class row per image."""
    seq_index: npt.NDArray[np.int64]
    """``[N + B]`` gathers the sequence from the patch rows followed by one
    class row (index ``N``)."""
    max_seqlen: npt.NDArray[np.uint32]
    """Scalar longest sequence, class row included."""


def _patch_rows_cols(
    grid_hw: npt.NDArray[np.int64], merge: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Returns the native ``(row, col)`` of every patch, images in order.

    ``grid_hw`` counts image tokens. Patches run in image-token order, then
    row-major inside each token's ``merge x merge`` square.
    """
    rows, cols = [], []
    in_square = np.arange(merge)
    for h, w in grid_hw:
        shape = (h, w, merge, merge)
        rows.append(
            np.broadcast_to(
                np.arange(h)[:, None, None, None] * merge
                + in_square[None, None, :, None],
                shape,
            ).reshape(-1)
        )
        cols.append(
            np.broadcast_to(
                np.arange(w)[None, :, None, None] * merge
                + in_square[None, None, None, :],
                shape,
            ).reshape(-1)
        )
    return np.concatenate(rows), np.concatenate(cols)


def _axis_taps(
    index: npt.NDArray[np.int64],
    size: npt.NDArray[np.int64],
    side: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """Two taps per position of a ``size``-long axis over a ``side`` table.

    Half-pixel centers with the edge taps clamped, as
    ``F.interpolate(mode="bilinear", align_corners=False)``, in float32 like
    the reference.
    """
    f32 = np.float32
    src = (index.astype(f32) + f32(0.5)) * f32(side) / size.astype(f32) - f32(
        0.5
    )
    floor = np.floor(src)
    offsets = np.arange(2, dtype=f32)
    taps = floor.astype(np.int64)[:, None] + np.arange(2)
    weights = np.clip(
        1 - np.abs(src[:, None] - floor[:, None] - offsets), 0, None
    )
    return np.clip(taps, 0, side - 1), weights.astype(f32)


def interpolation_taps(
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    heights: npt.NDArray[np.int64],
    widths: npt.NDArray[np.int64],
    side: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Returns bilinear taps resampling a ``side x side`` table per patch.

    ``heights`` and ``widths`` are the native patch grid of each patch's image.
    """
    h_taps, h_w = _axis_taps(rows, heights, side)
    w_taps, w_w = _axis_taps(cols, widths, side)
    idx = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    wts = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return idx.astype(np.int32), wts.astype(np.float32)


def vision_inputs(
    grid_thw: npt.ArrayLike,
    pixel_values: npt.NDArray[np.int16],
    *,
    merge: int,
    posemb_side: int,
) -> VisionInputs:
    """Builds the tower inputs for images packed in ``grid_thw`` order.

    Args:
        grid_thw: ``[num_images, 3]`` image-token grids from the image
            processor.
        pixel_values: The images' pixel blocks, concatenated in order.
        merge: Native patches per image token along each axis.
        posemb_side: Side of the learned position table.

    Returns:
        The :class:`VisionInputs` for the packed images.
    """
    grid_hw = np.asarray(grid_thw, dtype=np.int64)[:, 1:]
    num_images = len(grid_hw)
    patches = grid_hw.prod(axis=1) * merge**2
    total = int(patches.sum())

    rows, cols = _patch_rows_cols(grid_hw, merge)
    heights = np.repeat(grid_hw[:, 0] * merge, patches)
    widths = np.repeat(grid_hw[:, 1] * merge, patches)
    interp_idx, interp_w = interpolation_taps(
        rows, cols, heights, widths, posemb_side
    )

    # Row 0 of each image's sequence is its class token, at (0, 0); patches
    # sit at their 1-indexed (row, col).
    image_of_patch = np.repeat(np.arange(num_images), patches)
    patch_index = np.arange(total) + image_of_patch + 1
    seq_index = np.full(total + num_images, total, dtype=np.int64)
    seq_index[patch_index] = np.arange(total)
    positions = np.zeros((total + num_images, 2), dtype=np.int32)
    positions[patch_index] = np.stack([rows, cols], axis=-1) + 1

    return VisionInputs(
        interp_idx=interp_idx,
        interp_w=interp_w,
        positions=positions,
        pixel_values=pixel_values,
        cu_seqlens=np.concatenate([[0], np.cumsum(patches)]).astype(np.uint32),
        seq_index=seq_index,
        max_seqlen=np.array(patches.max() + 1, dtype=np.uint32),
    )
