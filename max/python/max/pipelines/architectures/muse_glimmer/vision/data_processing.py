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

"""Host-side index and table inputs of the Muse Glimmer vision tower."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ..model_config import MuseGlimmerVisionConfig


class VisionInputs(NamedTuple):
    """Every vision tower input except ``pixel_values``, in compile order."""

    interp_idx: npt.NDArray[np.int32]
    """``[P, 4]`` bilinear taps into the flattened position table."""
    interp_w: npt.NDArray[np.float32]
    """``[P, 4]`` tap weights; zero for a tap outside the table."""
    window_index: npt.NDArray[np.int64]
    """``[P]`` raster patch order -> window-major order."""
    reverse_index: npt.NDArray[np.int64]
    pixel_shuffle_index: npt.NDArray[np.int64]
    """``[P]`` raster order -> each merge block as consecutive rows."""
    rot_cos: npt.NDArray[np.float32]
    """``[P, head_dim]`` in window-major order."""
    rot_sin: npt.NDArray[np.float32]
    cu_seqlens: npt.NDArray[np.uint32]
    cu_window_seqlens: npt.NDArray[np.uint32]
    max_seqlen: npt.NDArray[np.uint32]
    max_window_seqlen: npt.NDArray[np.uint32]


def patch_rows_cols(grid_hw: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Returns the ``(row, col)`` of every patch, images in raster order."""
    return np.concatenate(
        [np.stack(np.divmod(np.arange(h * w), w), axis=-1) for h, w in grid_hw]
    )


def window_index(
    grid_hw: npt.NDArray[np.int64], window: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Returns the window-major patch order and cumulative window lengths.

    Windows are ``window x window`` patches, row-major over each image, and
    clipped at the image edge; patches inside a window stay in raster order.
    """
    index, seqlens, offset = [], [0], 0
    for h, w in grid_hw:
        ids = np.arange(h * w).reshape(h, w) + offset
        for r in range(0, h, window):
            for c in range(0, w, window):
                block = ids[r : r + window, c : c + window].reshape(-1)
                index.append(block)
                seqlens.append(block.size)
        offset += h * w
    return np.concatenate(index), np.cumsum(seqlens)


def _axis_taps(
    index: npt.NDArray[np.int64],
    size: npt.NDArray[np.int64],
    side: int,
    align_corners: bool,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    if align_corners:
        src = index * (side - 1) / np.maximum(size - 1, 1)
    else:
        src = (index + 0.5) * side / size - 0.5
    raw = np.floor(src).astype(np.int64)[:, None] + np.arange(2)
    weights = 1.0 - np.abs(src[:, None] - raw)
    # Zero padding, as F.grid_sample(padding_mode="zeros"): a tap outside
    # the table contributes nothing.
    weights *= (raw >= 0) & (raw <= side - 1)
    return np.clip(raw, 0, side - 1), weights


def interpolation_taps(
    grid_hw: npt.NDArray[np.int64], side: int, align_corners: bool = False
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Returns bilinear taps resampling a ``side x side`` table to each grid."""
    rows, cols = patch_rows_cols(grid_hw).T
    counts = grid_hw.prod(axis=1)
    heights, widths = (np.repeat(grid_hw[:, i], counts) for i in (0, 1))
    h_taps, h_w = _axis_taps(rows, heights, side, align_corners)
    w_taps, w_w = _axis_taps(cols, widths, side, align_corners)
    idx = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    wts = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return idx.astype(np.int32), wts.astype(np.float32)


def rope_tables(
    positions: npt.NDArray[np.int64], head_dim: int, theta: float
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Returns axial 2D RoPE cos/sin for ``(a, b)`` positions, rotate-half.

    Each axis gets ``head_dim // 4`` frequencies; the angles are
    ``concat([a, b, a, b])``.
    """
    half = head_dim // 2
    inv_freq = 1.0 / theta ** (np.arange(0, half, 2) / half)
    angles = (positions[:, :, None] * inv_freq).reshape(len(positions), half)
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def pixel_shuffle_index(
    grid_hw: npt.NDArray[np.int64], merge: int
) -> npt.NDArray[np.int64]:
    """Returns the gather that makes each ``merge x merge`` block contiguous."""
    index, offset = [], 0
    for h, w in grid_hw:
        block = np.arange(h * w).reshape(h // merge, merge, w // merge, merge)
        index.append(block.transpose(0, 2, 1, 3).reshape(-1) + offset)
        offset += h * w
    return np.concatenate(index)


def vision_inputs(
    grid_thw: npt.ArrayLike, config: MuseGlimmerVisionConfig
) -> VisionInputs:
    """Builds the tower inputs for images packed in ``grid_thw`` order.

    Args:
        grid_thw: ``[num_images, 3]`` patch grids from the image processor.
        config: The vision config.

    Returns:
        The :class:`VisionInputs` for the packed images.

    Raises:
        ValueError: If an entry has more than one temporal frame.
    """
    grid_thw = np.asarray(grid_thw, dtype=np.int64)
    if (grid_thw[:, 0] != 1).any():
        raise ValueError(f"Only single-frame images are supported: {grid_thw}")
    grid_hw = grid_thw[:, 1:]
    win_index, cu_window = window_index(grid_hw, config.window_size_patches)
    # Positions are (col, row) + 1: R2 flips the (row, col) ids.
    positions = patch_rows_cols(grid_hw)[win_index, ::-1] + 1
    rot_cos, rot_sin = rope_tables(
        positions, config.head_dim, config.rope_theta
    )
    interp_idx, interp_w = interpolation_taps(grid_hw, config.pos_emb_height)
    cu_seqlens = np.cumsum([0, *grid_hw.prod(axis=1)])
    return VisionInputs(
        interp_idx=interp_idx,
        interp_w=interp_w,
        window_index=win_index,
        reverse_index=np.argsort(win_index),
        pixel_shuffle_index=pixel_shuffle_index(grid_hw, config.merge_size),
        rot_cos=rot_cos,
        rot_sin=rot_sin,
        cu_seqlens=cu_seqlens.astype(np.uint32),
        cu_window_seqlens=cu_window.astype(np.uint32),
        max_seqlen=np.array(np.diff(cu_seqlens).max(), dtype=np.uint32),
        max_window_seqlen=np.array(np.diff(cu_window).max(), dtype=np.uint32),
    )
