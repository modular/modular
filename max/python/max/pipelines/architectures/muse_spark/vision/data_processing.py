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

"""Host-side index and table inputs of the Muse Spark vision tower.

Muse Glimmer's inputs, except that the position table is read at
``(w, h)`` and the RoPE table is computed in float32, as the reference does.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.muse_glimmer.model_config import (
    MuseGlimmerVisionConfig,
)
from max.pipelines.architectures.muse_glimmer.vision.data_processing import (
    VisionInputs,
    interpolation_taps,
    patch_rows_cols,
    pixel_shuffle_index,
    window_index,
)


def rope_tables(
    positions: npt.NDArray[np.int64], head_dim: int, theta: float
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Returns axial 2D RoPE cos/sin for ``(a, b)`` positions, rotate-half.

    Each axis gets ``head_dim // 4`` frequencies; the angles are
    ``concat([a, b, a, b])``. The frequencies and angles are rounded to
    float32 where the reference computes them in float32: in float64 the
    table drifts from it by about 2e-6.
    """
    half = head_dim // 2
    exponents = np.arange(0, half, 2, dtype=np.float32) / np.float32(half)
    inv_freq = np.float32(1) / (theta ** exponents.astype(np.float64)).astype(
        np.float32
    )
    angles = (positions.astype(np.float32)[:, :, None] * inv_freq).reshape(
        len(positions), half
    )
    angles = np.concatenate([angles, angles], axis=-1).astype(np.float64)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


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
    # Positions are (col, row) + 1: the reference flips the (row, col) ids.
    positions = patch_rows_cols(grid_hw)[win_index, ::-1] + 1
    rot_cos, rot_sin = rope_tables(
        positions, config.head_dim, config.rope_theta
    )
    # The reference resamples the position table on the transposed grid:
    # patch (h, w) reads it at (w, h).
    interp_idx, interp_w = interpolation_taps(
        grid_hw[:, ::-1], config.pos_emb_height
    )
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
