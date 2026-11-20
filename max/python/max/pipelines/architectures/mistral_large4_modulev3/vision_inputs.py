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
"""Packs images into the inputs of the Mistral Large 4 vision tower."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from max.driver import Buffer, Device
from max.pipelines.context import ImageMetadata

from .vision import MistralLarge4VisionConfig


def pack_images(
    images: Sequence[ImageMetadata],
    config: MistralLarge4VisionConfig,
    device: Device,
) -> list[Buffer]:
    """Packs images onto ``device`` in the vision graph's input order.

    The tower encodes every patch of an image and the projector merges each
    square of ``spatial_merge_size`` patches into one embedding row,
    dropping the last row or column of an odd grid, as the reference does.

    Args:
        images: Images with ``[channels, height, width]`` **float32**
            pixels, whose sides are multiples of the patch size and no
            larger than the 2D RoPE table covers. The dtype matters:
            ``ImageMetadata.pixel_values`` also allows uint16 BF16 *bits*
            (``context/tokens.py``), which the patch flattening would
            silently reinterpret as small integers.
        config: The vision graph's configuration.
        device: The device that runs the vision graph.

    Returns:
        The patches, the 2D RoPE position ids, the per-image patch offsets
        the ragged attention reads, the merge indices and the longest
        image's patch count, in :func:`vision_input_types`'s order.

    Raises:
        ValueError: If the batch holds no image, or one whose side is not
            a multiple of the patch size or exceeds the RoPE table's
            largest image (an out-of-range position gather otherwise).
    """
    if not images:
        raise ValueError("packing a batch with no images")

    patch = config.patch_size
    merge = config.spatial_merge_size
    # The RoPE table is square at the largest image, so a smaller image's
    # (row, col) still addresses it as row * grid_side + col.
    grid_side = config.max_patches_per_side
    # An image taller or wider than this would index its 2D RoPE row or
    # column past the cached table, an out-of-range gather.
    max_side = grid_side * patch

    patches: list[np.ndarray] = []
    position_ids: list[np.ndarray] = []
    merge_indices: list[np.ndarray] = []
    patch_counts: list[int] = []
    offset = 0
    for index, image in enumerate(images):
        pixels = np.ascontiguousarray(image.pixel_values)
        channels, height, width = pixels.shape
        if height % patch or width % patch:
            raise ValueError(
                f"image {index}: sides must be multiples of the patch size"
                f" ({patch}); got {height}x{width}"
            )
        if height > max_side or width > max_side:
            raise ValueError(
                f"image {index}: {height}x{width} exceeds the largest image"
                f" the 2D RoPE table covers ({max_side} per side)"
            )
        grid_h, grid_w = height // patch, width // patch
        merged_h, merged_w = grid_h // merge, grid_w // merge

        # [C, H, W] to [grid_h * grid_w, C * patch * patch], row-major.
        patches.append(
            pixels.reshape(channels, grid_h, patch, grid_w, patch)
            .transpose(1, 3, 0, 2, 4)
            .reshape(grid_h * grid_w, channels * patch * patch)
            .astype(np.float32, copy=False)
        )
        position_ids.append(
            (np.arange(grid_h)[:, None] * grid_side + np.arange(grid_w)).ravel()
        )
        # The rows of each square, [merged_h, merged_w, m, m].
        square_rows = (
            np.arange(merged_h)[:, None, None, None] * merge
            + np.arange(merge)[None, None, :, None]
        )
        square_cols = (
            np.arange(merged_w)[None, :, None, None] * merge
            + np.arange(merge)[None, None, None, :]
        )
        merge_indices.append(
            (offset + square_rows * grid_w + square_cols).reshape(
                merged_h * merged_w, merge * merge
            )
        )
        patch_counts.append(grid_h * grid_w)
        offset += grid_h * grid_w

    # The ragged kernel bounds each image by its offsets, so images never
    # need pairwise mask bytes.
    cu_seqlens = np.concatenate([[0], np.cumsum(patch_counts)]).astype(
        np.uint32
    )
    max_seq_len = np.array(max(patch_counts), dtype=np.uint32)

    return [
        Buffer.from_numpy(array).to(device)
        for array in (
            np.concatenate(patches),
            np.concatenate(position_ids).astype(np.int64),
            cu_seqlens,
            np.concatenate(merge_indices).astype(np.int32),
        )
    ] + [Buffer.from_numpy(max_seq_len)]
