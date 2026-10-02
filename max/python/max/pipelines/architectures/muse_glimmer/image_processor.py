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

"""Muse Glimmer image preprocessing in NumPy, after transformers 5.17's
``MuseGlimmerImageProcessor``."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image


def smart_resize(
    height: int, width: int, patch_size: int, max_tokens: int
) -> tuple[int, int]:
    """Picks the patch grid closest to the input aspect ratio under the cap.

    Returns:
        The resize target ``(height, width)`` in pixels.
    """
    ideal_h = height / patch_size
    ideal_w = width / patch_size
    ratio = ideal_w / ideal_h if ideal_h > 0 else 1.0
    if ideal_h * ideal_w > max_tokens:
        ideal_h = (max_tokens / ratio) ** 0.5
        ideal_w = ideal_h * ratio
    candidates = [
        (h, w)
        for h, w in set(
            itertools.product(
                [math.floor(ideal_h), math.ceil(ideal_h)],
                [math.floor(ideal_w), math.ceil(ideal_w)],
            )
        )
        if h >= 1 and w >= 1 and h * w <= max_tokens
    ]
    if not candidates:
        # A thin image rounds its short side up to one patch, which can push
        # the long side past the cap; the transformers reference does not
        # clamp it.
        h, w = max(1, round(ideal_h)), max(1, round(ideal_w))
        candidates = [
            (h, min(w, max_tokens // h))
            if h <= w
            else (min(h, max_tokens // w), w)
        ]
    grid_h, grid_w = min(
        candidates, key=lambda grid: abs(grid[0] / grid[1] - height / width)
    )
    return grid_h * patch_size, grid_w * patch_size


def _lanczos3(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    x = np.abs(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(x == 0, 1.0, np.sin(np.pi * x) / (np.pi * x))
        sinc3 = np.where(x == 0, 1.0, np.sin(np.pi * x / 3) / (np.pi * x / 3))
    return np.where(x < 3.0, sinc * sinc3, 0.0)


def _int16_taps(
    in_size: int, out_size: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], int]:
    # Mirrors torch's uint8 antialiased kernel (UpSampleKernel.cpp,
    # `_compute_index_ranges_int16_weights`): its int16 weights and adaptive
    # precision round differently from PIL's, so PIL is off by 1 LSB.
    scale = in_size / out_size
    support = 3.0 * scale if scale >= 1.0 else 3.0
    ksize = math.ceil(support) * 2 + 1
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    center = scale * (np.arange(out_size) + 0.5)
    xmin = np.maximum((center - support + 0.5).astype(np.int64), 0)
    xsize = np.minimum((center + support + 0.5).astype(np.int64), in_size)
    xsize = np.clip(xsize - xmin, 0, ksize)
    j = np.arange(ksize)
    weights = _lanczos3((j + xmin[:, None] - center[:, None] + 0.5) * invscale)
    weights = np.where(j < xsize[:, None], weights, 0.0)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.divide(
        weights, total, out=np.zeros_like(weights), where=total != 0
    )
    precision = 0
    while precision < 22 and int(
        0.5 + weights.max() * (1 << (precision + 1))
    ) < (1 << 15):
        precision += 1
    scaled = weights * (1 << precision)
    int_weights = np.trunc(np.where(scaled < 0, scaled - 0.5, scaled + 0.5))
    indices = np.minimum(xmin[:, None] + j, in_size - 1)
    return indices, int_weights.astype(np.int32), precision


def _resample_rows(
    pixels: npt.NDArray[np.uint8], out_size: int
) -> npt.NDArray[np.uint8]:
    indices, weights, precision = _int16_taps(pixels.shape[0], out_size)
    weights = weights[:, :, None, None]
    out = np.empty((out_size, *pixels.shape[1:]), dtype=np.uint8)
    # Column blocks keep the int32 accumulator in cache; unblocked, a
    # 4000-pixel pass is ~8x slower.
    for c in range(0, pixels.shape[1], 64):
        block = pixels[:, c : c + 64]
        acc = np.full(
            (out_size, *block.shape[1:]), 1 << (precision - 1), dtype=np.int32
        )
        for k in range(indices.shape[1]):
            acc += block[indices[:, k]] * weights[:, k]
        out[:, c : c + 64] = np.clip(acc >> precision, 0, 255)
    return out


def lanczos_resize(
    pixels: npt.NDArray[np.uint8], height: int, width: int
) -> npt.NDArray[np.uint8]:
    """Resizes an ``[H, W, C]`` uint8 image bit-exactly as torch does on CPU
    for ``interpolate(mode="lanczos", antialias=True)``: horizontal pass
    first, each pass rounded to uint8."""
    if pixels.shape[1] != width:
        pixels = _resample_rows(
            np.ascontiguousarray(pixels.transpose(1, 0, 2)), width
        ).transpose(1, 0, 2)
    if pixels.shape[0] != height:
        pixels = _resample_rows(np.ascontiguousarray(pixels), height)
    return pixels


class MuseGlimmerImageProcessor:
    """Turns an image into flat patches and its ``[t, h, w]`` patch grid."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        """Reads the constants from ``processor_config.json``'s
        ``image_processor`` block."""
        self.patch_size = int(config["patch_size"])
        self.merge_size = int(config["merge_size"])
        self.temporal_patch_size = int(config["temporal_patch_size"])
        self.max_image_tokens = int(config["max_image_tokens"])
        self.rescale_factor = np.float32(config["rescale_factor"])
        self.image_mean = np.asarray(config["image_mean"], dtype=np.float32)
        self.image_std = np.asarray(config["image_std"], dtype=np.float32)

    def __call__(
        self, image: Image.Image
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
        """Returns ``pixel_values [h * w, 1176]`` in raster patch order, each
        patch flattened ``(temporal, channel, py, px)``, and
        ``grid_thw = [1, h, w]``."""
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = smart_resize(
            pixels.shape[0],
            pixels.shape[1],
            self.patch_size * self.merge_size,
            self.max_image_tokens,
        )
        pixels = lanczos_resize(pixels, height, width)
        normalized = (
            pixels.astype(np.float32) * self.rescale_factor - self.image_mean
        ) / self.image_std
        p = self.patch_size
        grid_h, grid_w = height // p, width // p
        patches = normalized.reshape(grid_h, p, grid_w, p, 3).transpose(
            0, 2, 4, 1, 3
        )
        # A still image fills every temporal slot with the same frame.
        patches = np.broadcast_to(
            patches[:, :, None],
            (grid_h, grid_w, self.temporal_patch_size, 3, p, p),
        )
        return (
            patches.reshape(grid_h * grid_w, -1),
            np.array([1, grid_h, grid_w], dtype=np.int64),
        )
