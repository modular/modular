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
"""Hy4 VL image preprocessing in numpy, without torch.

The reference processor is torchvision-backed: it resizes the ``uint8`` image
with ``interpolate(mode="bicubic", antialias=True)``, whose CPU kernel is
Pillow's separable resample with 16-bit taps. This module computes the same
bytes, so the patches equal the reference's. ``resample: 1`` (LANCZOS) in
``processor_config.json`` is ignored, as the reference ignores it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
from transformers.utils import cached_file

_MAX_PRECISION_BITS = 22


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Returns the ``(height, width)`` multiple of ``factor`` the image gets.

    Each side is rounded on its own (Python ``round``, half to even), then
    both are scaled together into ``[min_pixels, max_pixels]``.
    """
    if min(height, width) <= 0:
        raise ValueError(f"image size must be positive, got {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"image aspect ratio must not exceed 200, got {height}x{width}"
        )
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / scale / factor) * factor)
        w = max(factor, math.floor(width / scale / factor) * factor)
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    if h * w > max_pixels:
        raise ValueError(
            f"image {height}x{width} needs {h}x{w}, over max_pixels"
        )
    return h, w


def _cubic(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Keys' cubic, ``a = -0.5``, in Pillow's operation order."""
    a = -0.5
    x = np.abs(x)
    near = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    far = (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return np.where(x < 1.0, near, np.where(x < 2.0, far, 0.0))


def _taps(
    in_size: int, out_size: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int32], int]:
    """Returns torch's ``uint8`` taps for one axis: indices, taps, precision.

    The weights are Pillow's, in float64. torch quantizes them at the finest
    scale that keeps the largest one below 2^15, not at Pillow's 22 bits.
    Padding taps are zero and read index 0.
    """
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    centers = (np.arange(out_size) + 0.5) * scale
    starts = np.maximum((centers - support + 0.5).astype(np.int64), 0)
    ends = np.minimum((centers + support + 0.5).astype(np.int64), in_size)
    index = starts[:, None] + np.arange(int((ends - starts).max()))
    valid = index < ends[:, None]
    weights = np.where(
        valid, _cubic((index - centers[:, None] + 0.5) * (1.0 / filterscale)), 0
    )
    # Pillow sums each kernel left to right; numpy's pairwise sum would not.
    total = np.zeros(out_size)
    for column in weights.T:
        total += column
    weights /= np.where(total == 0, 1.0, total)[:, None]

    largest = weights.max()
    precision = 0
    while (
        precision < _MAX_PRECISION_BITS
        and int(0.5 + largest * (1 << (precision + 1))) < 1 << 15
    ):
        precision += 1
    scaled = weights * (1 << precision)
    taps = np.where(scaled < 0, scaled - 0.5, scaled + 0.5).astype(np.int32)
    return np.where(valid, index, 0), taps, precision


def _resample_rows(
    image: npt.NDArray[np.uint8], out_size: int
) -> npt.NDArray[np.uint8]:
    """Resamples axis 1 of an ``(H, W, C)`` image to ``out_size``."""
    index, taps, precision = _taps(image.shape[1], out_size)
    total = np.full(
        (image.shape[0], out_size, image.shape[2]),
        1 << (precision - 1),
        np.int32,
    )
    for k in range(taps.shape[1]):
        total += image[:, index[:, k]].astype(np.int32) * taps[None, :, k, None]
    return np.clip(total >> precision, 0, 255).astype(np.uint8)


def _resize_bicubic(
    image: npt.NDArray[np.uint8], height: int, width: int
) -> npt.NDArray[np.uint8]:
    """Resizes an ``(H, W, C)`` ``uint8`` image as torch's antialiased bicubic.

    The width pass runs first and is rounded to bytes; a pass whose size does
    not change is skipped, as in torch.
    """
    if image.shape[1] != width:
        image = _resample_rows(image, width)
    if image.shape[0] != height:
        image = _resample_rows(image.transpose(1, 0, 2), height).transpose(
            1, 0, 2
        )
    return image


@dataclass(frozen=True)
class HYV4VLImageProcessor:
    """Resizes, normalizes and patches one image for the Hy4 VL tower."""

    patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    rescale_factor: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> HYV4VLImageProcessor:
        """Reads the ``image_processor`` section of ``processor_config.json``."""
        return cls(
            patch_size=config["patch_size"],
            merge_size=config["merge_size"],
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
            image_mean=tuple(config["image_mean"]),
            image_std=tuple(config["image_std"]),
            rescale_factor=config["rescale_factor"],
        )

    @classmethod
    def from_pretrained(
        cls, model_path: str, revision: str | None
    ) -> HYV4VLImageProcessor:
        """Reads the checkpoint's ``processor_config.json``."""
        path = cached_file(
            model_path, "processor_config.json", revision=revision
        )
        assert path is not None
        return cls.from_config(
            json.loads(Path(path).read_text())["image_processor"]
        )

    def grid(self, height: int, width: int) -> tuple[int, int]:
        """Returns the ``(rows, columns)`` patch grid of an image."""
        h, w = smart_resize(
            height,
            width,
            self.patch_size * self.merge_size,
            self.min_pixels,
            self.max_pixels,
        )
        return h // self.patch_size, w // self.patch_size

    def num_image_tokens(self, height: int, width: int) -> int:
        """Returns the image's placeholder count: one newline per merged row."""
        rows, columns = self.grid(height, width)
        m = self.merge_size
        return (rows // m) * (columns // m + 1)

    def __call__(
        self, image: Image.Image
    ) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
        """Returns the ``[rows * columns, 3 * patch**2]`` patches and the grid.

        Patches are in row-major grid order, each ordered (channel, y, x).
        Transparency is composited onto white, as vLLM's image loader does.
        """
        if "A" in image.getbands() or "transparency" in image.info:
            white = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(white, image.convert("RGBA"))
        pixels = np.asarray(image.convert("RGB"))
        rows, columns = self.grid(*pixels.shape[:2])
        p = self.patch_size
        pixels = _resize_bicubic(pixels, rows * p, columns * p)
        # The reference fuses the rescale into the mean and std in float32.
        scale = np.float32(1.0 / self.rescale_factor)
        mean = np.asarray(self.image_mean, np.float32) * scale
        std = np.asarray(self.image_std, np.float32) * scale
        normalized = (pixels.astype(np.float32) - mean) / std
        patches = normalized.reshape(rows, p, columns, p, 3).transpose(
            0, 2, 4, 1, 3
        )
        return patches.reshape(rows * columns, 3 * p * p), (1, rows, columns)
