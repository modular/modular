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

"""DeepSeek-V4.1 image preprocessing and image-span expansion.

Follows the checkpoint's reference ``inference/image_processor.py``. An image
becomes an ``n_vit_h x n_vit_w`` grid of 14x14 patches for the ViT and an
``n_llm_h x n_llm_w`` token grid after the aligner's 3x3 downsample, which the
LLM sees as::

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every position of that span carries ``image_token_id``; only the token type
tells them apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from max.driver import float32_to_bfloat16_as_uint16
from max.pipelines.context.exceptions import InputError
from PIL import Image, ImageOps

IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

_PAD_COLOR = (127, 127, 127)


@dataclass(frozen=True)
class ImageGrid:
    """The patch grid the ViT sees and the token grid the LLM sees."""

    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int

    @property
    def num_tokens(self) -> int:
        """Positions of the image span, delimiters included."""
        return _num_tokens(self.n_llm_h, self.n_llm_w)


def image_token_types(n_llm_h: int, n_llm_w: int) -> npt.NDArray[np.int64]:
    """Returns the token type of every position of an image span."""
    row = [IMAGE] * n_llm_w + [IMAGE_NEW_LINE]
    return np.array([IMAGE_START, *row * n_llm_h, IMAGE_END], dtype=np.int64)


def _llm_grid(height: int, width: int, p: int, ratio: int) -> tuple[int, int]:
    return math.ceil(height // p / ratio), math.ceil(width // p / ratio)


def _num_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def _fit_to_token_cap(
    height: int, width: int, p: int, ratio: int, max_tokens: int
) -> tuple[int, int]:
    """Largest aspect-preserving pixel size whose span fits ``max_tokens``."""
    r = height / width
    max_w = math.sqrt((max_tokens - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    cell = p * ratio
    if max_w < 1.0:
        return (max_tokens - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (max_tokens - 3) * cell
    beta = min(
        math.floor(max_w) * cell / width, math.floor(max_h) * cell / height
    )
    return (
        math.floor(height * beta / p) * p,
        math.floor(width * beta / p) * p,
    )


class DeepseekV41ImageProcessor:
    """Turns an image into bf16 ViT patches and its grid."""

    def __init__(self, vision_config: Any) -> None:
        """Reads the constants from the checkpoint's ``vision_config``."""
        if vision_config.max_wh_ratio is not None:
            raise ValueError(
                "DeepSeek-V4.1 vision_config.max_wh_ratio must be null, got"
                f" {vision_config.max_wh_ratio}"
            )
        self.patch_size = int(vision_config.patch_size)
        self.downsample_ratio = int(vision_config.downsample_ratio)
        self.min_pixels = int(vision_config.min_pixels)
        self.max_image_tokens = int(vision_config.max_image_tokens)

    def plan(self, width: int, height: int) -> tuple[ImageGrid, int, int]:
        """Returns the grid and the padded ``(width, height)`` in pixels.

        Upscales below ``min_pixels``, rounds up to whole patches, and shrinks
        until the span fits ``max_image_tokens``.
        """
        p, ratio = self.patch_size, self.downsample_ratio
        w, h = width, height
        if 0 < width * height < self.min_pixels:
            scale = (self.min_pixels / (width * height)) ** 0.5
            w, h = int(width * scale), int(height * scale)
        best_w, best_h = math.ceil(w / p) * p, math.ceil(h / p) * p
        n_llm_h, n_llm_w = _llm_grid(best_h, best_w, p, ratio)
        if _num_tokens(n_llm_h, n_llm_w) > self.max_image_tokens:
            best_h, best_w = _fit_to_token_cap(
                h, w, p, ratio, self.max_image_tokens
            )
            n_llm_h, n_llm_w = _llm_grid(best_h, best_w, p, ratio)
        grid = ImageGrid(best_h // p, best_w // p, n_llm_h, n_llm_w)
        return grid, best_w, best_h

    def __call__(
        self, image: Image.Image
    ) -> tuple[npt.NDArray[np.uint16], ImageGrid]:
        """Returns the patches as bf16 bits ``[n_vit_h * n_vit_w, 3, p, p]``
        in raster order, and the image's grid."""
        image = image.convert("RGB")
        grid, width, height = self.plan(image.width, image.height)
        padded = ImageOps.pad(image, (width, height), color=_PAD_COLOR)
        x = np.asarray(padded, dtype=np.float32) / np.float32(255)
        x = (x - np.float32(0.5)) / np.float32(0.5)
        p = self.patch_size
        patches = x.reshape(grid.n_vit_h, p, grid.n_vit_w, p, 3).transpose(
            0, 2, 4, 1, 3
        )
        return float32_to_bfloat16_as_uint16(
            np.ascontiguousarray(patches).reshape(-1, 3, p, p)
        ), grid


def expand_image_spans(
    token_ids: Sequence[int], grids: Sequence[ImageGrid], image_token_id: int
) -> tuple[npt.NDArray[np.int64], list[int]]:
    """Grows each ``image_token_id`` placeholder into its image span.

    Returns:
        The expanded ids and the start position of each image's span.

    Raises:
        InputError: If the placeholder count differs from the image count.
    """
    markers = sum(1 for t in token_ids if t == image_token_id)
    if markers != len(grids):
        raise InputError(
            f"The prompt has {markers} image placeholder(s) but {len(grids)}"
            " image(s) were given."
        )
    ids: list[int] = []
    starts: list[int] = []
    remaining = iter(grids)
    for token in token_ids:
        if token != image_token_id:
            ids.append(token)
            continue
        starts.append(len(ids))
        ids.extend([image_token_id] * next(remaining).num_tokens)
    return np.array(ids, dtype=np.int64), starts
