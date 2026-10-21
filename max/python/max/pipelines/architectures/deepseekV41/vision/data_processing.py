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

"""Host-side table and index inputs of the DeepSeek-V4.1 vision model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .vision_model import DeepseekV41VisionConfig


class VisionInputs(NamedTuple):
    """Every vision model input except ``patches``, in compile order."""

    rot_cos: npt.NDArray[np.float32]
    """``[P, rope_dim]``, images packed in order."""
    rot_sin: npt.NDArray[np.float32]
    cu_seqlens: npt.NDArray[np.uint32]
    max_seqlen: npt.NDArray[np.uint32]
    aligner_index: npt.NDArray[np.int64]
    """``[n_llm * r * r]`` patch rows of each aligner block; ``P`` is the
    zero row the grid padding reads."""
    span_index: npt.NDArray[np.int64]
    """``[span tokens]`` rows of ``concat(aligner rows, [start, newline,
    end])`` for every span position."""


def rope_tables(
    n_h: int, n_w: int, rope_dim: int, theta: float
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Returns the 2D axial RoPE cos/sin of an ``n_h x n_w`` grid.

    The first ``rope_dim // 2`` angles follow the patch row and the rest the
    patch column, all in float32.
    """
    exponents = np.arange(0, rope_dim, 2, dtype=np.float32) / np.float32(
        rope_dim
    )
    inv_freq = np.float32(1.0) / np.float32(theta) ** exponents
    rows, cols = np.divmod(np.arange(n_h * n_w), n_w)
    positions = np.stack([rows, cols], axis=-1).astype(np.float32)
    angles = (positions[:, :, None] * inv_freq).reshape(n_h * n_w, rope_dim)
    return np.cos(angles), np.sin(angles)


def llm_grid(n_vit_h: int, n_vit_w: int, ratio: int) -> tuple[int, int]:
    """Returns the aligner grid of a patch grid padded to multiples of ``ratio``."""
    return -(-n_vit_h // ratio), -(-n_vit_w // ratio)


def aligner_index(
    n_vit_h: int, n_vit_w: int, ratio: int, offset: int, pad_row: int
) -> npt.NDArray[np.int64]:
    """Returns each ``ratio x ratio`` block's patch rows, blocks in reading order."""
    n_llm_h, n_llm_w = llm_grid(n_vit_h, n_vit_w, ratio)
    rows = np.arange(n_llm_h * ratio)[:, None]
    cols = np.arange(n_llm_w * ratio)[None, :]
    grid = np.where(
        (rows < n_vit_h) & (cols < n_vit_w),
        rows * n_vit_w + cols + offset,
        pad_row,
    )
    return (
        grid.reshape(n_llm_h, ratio, n_llm_w, ratio)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )


def span_index(
    n_llm_h: int, n_llm_w: int, offset: int, delimiters: int
) -> npt.NDArray[np.int64]:
    """Returns the rows of one image span.

    ``offset`` is the image's first aligner row and ``delimiters`` the row
    of ``image_start``, followed by ``image_newline`` and ``image_end``.
    """
    grid = np.arange(n_llm_h * n_llm_w).reshape(n_llm_h, n_llm_w) + offset
    newline = np.full((n_llm_h, 1), delimiters + 1)
    body = np.concatenate([grid, newline], axis=1).reshape(-1)
    return np.concatenate([[delimiters], body, [delimiters + 2]]).astype(
        np.int64
    )


def vision_inputs(
    grids: Sequence[tuple[int, int]], config: DeepseekV41VisionConfig
) -> VisionInputs:
    """Builds the vision model inputs for images packed in ``grids`` order.

    Args:
        grids: The ``(n_vit_h, n_vit_w)`` patch grid of each image.
        config: The vision config.

    Returns:
        The :class:`VisionInputs` for the packed images.
    """
    ratio = config.downsample_ratio
    seqlens = [h * w for h, w in grids]
    total = sum(seqlens)
    llm_grids = [llm_grid(h, w, ratio) for h, w in grids]
    delimiters = sum(h * w for h, w in llm_grids)
    tables = [
        rope_tables(h, w, config.rope_dim, config.rope_theta) for h, w in grids
    ]
    patch_offsets = np.cumsum([0, *seqlens])
    row_offsets = np.cumsum([0, *(h * w for h, w in llm_grids)])
    return VisionInputs(
        rot_cos=np.concatenate([cos for cos, _ in tables]),
        rot_sin=np.concatenate([sin for _, sin in tables]),
        cu_seqlens=patch_offsets.astype(np.uint32),
        max_seqlen=np.array(max(seqlens), dtype=np.uint32),
        aligner_index=np.concatenate(
            [
                aligner_index(h, w, ratio, int(offset), total)
                for (h, w), offset in zip(
                    grids, patch_offsets[:-1], strict=True
                )
            ]
        ),
        span_index=np.concatenate(
            [
                span_index(h, w, int(offset), delimiters)
                for (h, w), offset in zip(
                    llm_grids, row_offsets[:-1], strict=True
                )
            ]
        ),
    )
