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
"""Host-side inputs of the DeepSeek-V4.1 vision encoder, and their packing.

The encoder takes a batch's uncached images packed in ragged patch
sequences, with their RoPE tables and the gather indices that assemble the
aligner rows and the span delimiters into each image's span. One launch
holds at most one largest image's patches, the peak the memory planner
reserves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.pipelines.context import ImageMetadata, TextAndVisionContext

from .vision_encoder import DeepseekV41VisionConfig

IMAGE_GRIDS = "image_grids"
"""``extra_model_args`` key of the ``[len(images), 2]`` ViT patch grids
``(n_vit_h, n_vit_w)``, one row per entry of ``ctx.images``."""


class EncoderHostInputs(NamedTuple):
    """Every vision encoder input except ``patches``, in compile order."""

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
    """Returns the aligner grid of a patch grid.

    The patch grid pads to whole multiples of ``ratio`` on each side.
    """
    return -(-n_vit_h // ratio), -(-n_vit_w // ratio)


def span_tokens(n_vit_h: int, n_vit_w: int, ratio: int) -> int:
    """Positions of the image span of an ``n_vit_h x n_vit_w`` patch grid."""
    n_llm_h, n_llm_w = llm_grid(n_vit_h, n_vit_w, ratio)
    return n_llm_h * (n_llm_w + 1) + 2


def aligner_index(
    n_vit_h: int, n_vit_w: int, ratio: int, offset: int, pad_row: int
) -> npt.NDArray[np.int64]:
    """Returns the patch rows of each ``ratio x ratio`` block.

    The blocks are in reading order.
    """
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
) -> EncoderHostInputs:
    """Builds the vision encoder inputs for images packed in ``grids`` order.

    Args:
        grids: The ``(n_vit_h, n_vit_w)`` patch grid of each image.
        config: The vision config.

    Returns:
        The :class:`EncoderHostInputs` for the packed images.
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
    return EncoderHostInputs(
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


def select_images(
    selection: Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]],
    ratio: int,
) -> tuple[list[npt.NDArray[np.uint16]], list[tuple[int, int]]]:
    """Returns the patches and patch grids of the selected images.

    Images are matched to their grid by identity with ``ctx.images``, in
    context-major, image-minor order, the order the encoder emits rows.

    Raises:
        ValueError: If a selected image is not one of its context's own, or
            its grid does not fill its span.
    """
    patches: list[npt.NDArray[np.uint16]] = []
    grids: list[tuple[int, int]] = []
    for ctx, miss_images in selection:
        wanted = {id(img) for img in miss_images}
        all_grids = ctx.extra_model_args[IMAGE_GRIDS]
        for img, grid in zip(ctx.images, all_grids, strict=True):
            if id(img) not in wanted:
                continue
            wanted.discard(id(img))
            h, w = (int(x) for x in grid)
            if span_tokens(h, w, ratio) != img.end_idx - img.start_idx:
                raise ValueError(
                    f"image grid {(h, w)} does not fill its "
                    f"{img.end_idx - img.start_idx} span tokens"
                )
            patches.append(img.pixel_values)
            grids.append((h, w))
        if wanted:
            raise ValueError(
                f"{len(wanted)} selected image(s) of request {ctx.request_id}"
                " are not in ctx.images"
            )
    return patches, grids


def launch_bounds(
    grids: Sequence[tuple[int, int]], budget: int
) -> list[tuple[int, int]]:
    """Splits images, in order, into launches of at most ``budget`` patches.

    Each image is encoded whole, so a launch holds one image or more.

    Args:
        grids: The ``(n_vit_h, n_vit_w)`` patch grid of each image.
        budget: The most patches one launch may hold.

    Returns:
        The ``[start, end)`` image indices of each launch.

    Raises:
        ValueError: If one image has more than ``budget`` patches.
    """
    bounds: list[tuple[int, int]] = []
    start, patches = 0, 0
    for i, (h, w) in enumerate(grids):
        if h * w > budget:
            raise ValueError(
                f"image grid {(h, w)} has more than the {budget} patches one "
                "launch may hold"
            )
        if patches + h * w > budget:
            bounds.append((start, i))
            start, patches = i, 0
        patches += h * w
    if grids:
        bounds.append((start, len(grids)))
    return bounds


def _pack(
    patches: Sequence[npt.NDArray[np.uint16]],
    grids: Sequence[tuple[int, int]],
    device: Device,
    config: DeepseekV41VisionConfig,
) -> list[Buffer]:
    # The tokenizer keeps bf16 bit patterns as uint16.
    flat = np.concatenate([p.reshape(len(p), -1) for p in patches])
    host = vision_inputs(grids, config)
    return [
        Buffer.from_numpy(flat).view(DType.bfloat16).to(device),
        *(Buffer.from_numpy(a).to(device) for a in host[:3]),
        Buffer.from_numpy(host.max_seqlen).to(CPU()),
        *(Buffer.from_numpy(a).to(device) for a in host[4:]),
    ]


def pack_uncached_images(
    selection: Sequence[tuple[TextAndVisionContext, Sequence[ImageMetadata]]],
    device: Device,
    config: DeepseekV41VisionConfig,
) -> list[list[Buffer]] | None:
    """Packs the selected images in launches of the encoder's compile order.

    A launch holds at most :attr:`DeepseekV41VisionConfig.max_image_patches`
    patches, so its peak stays within the memory planner's reserve.

    Returns:
        For each launch, the bf16 ``[P, 3 * patch_size**2]`` patches then
        every :class:`EncoderHostInputs` field, on ``device`` except
        ``max_seqlen``; or ``None`` if nothing is selected.
    """
    patches, grids = select_images(selection, config.downsample_ratio)
    if not patches:
        return None
    return [
        _pack(patches[start:end], grids[start:end], device, config)
        for start, end in launch_bounds(grids, config.max_image_patches)
    ]
