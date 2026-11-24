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
"""Inkling 2 image processor: an image to one pixel block per image token.

Follows transformers' ``Inkling2ImageProcessor``: no resizing, transparency
composited on a checkerboard (:func:`to_rgb`, which the tokenizer applies),
the image padded at the bottom and right to whole blocks, then cut into blocks
in raster order. The block values stay on the 0 to 255 scale as ``int16`` and
padding holds -1; the vision tower rescales and normalizes them on device.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from PIL import Image

# Raw value of padded pixels. The tower rescales it with the real pixels.
PAD_RAW_VALUE = -1

# Checkerboard squares the reference composites transparent pixels onto.
CHECKER_SIDE = 8
CHECKER_LIGHT = 204
CHECKER_DARK = 153


def to_rgb(image: Image.Image) -> Image.Image:
    """Converts ``image`` to RGB, compositing transparency on a checkerboard.

    Integer arithmetic as the reference, so the composite is bit-exact.
    """
    if not image.has_transparency_data:
        return image.convert("RGB")
    rgba = np.asarray(image.convert("RGBA"), dtype=np.int32)
    height, width = rgba.shape[:2]
    rows = np.arange(height)[:, None] // CHECKER_SIDE
    cols = np.arange(width)[None, :] // CHECKER_SIDE
    background = np.where((rows + cols) % 2 == 0, CHECKER_LIGHT, CHECKER_DARK)[
        ..., None
    ]
    alpha = rgba[..., 3:4]
    rgb = (rgba[..., :3] * alpha + background * (255 - alpha) + 127) // 255
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


class Inkling2ImageProcessor:
    """Turns an image into the pixel blocks the vision tower consumes."""

    def __init__(self, token_size: int) -> None:
        """Takes the side, in pixels, of the block one image token covers."""
        self.token_size = token_size

    def grid(self, image: Image.Image) -> tuple[int, int, int]:
        """Returns the ``(1, rows, cols)`` image-token grid of ``image``."""
        side = self.token_size
        return 1, math.ceil(image.height / side), math.ceil(image.width / side)

    def __call__(self, image: Image.Image) -> npt.NDArray[np.int16]:
        """Returns the ``[rows * cols, side, side, 3]`` blocks of an RGB image.

        Pixels keep their 0 to 255 values and padding holds -1.
        """
        _, rows, cols = self.grid(image)
        pixels = np.asarray(image)
        height, width, _ = pixels.shape
        side = self.token_size
        padded = np.full((rows * side, cols * side, 3), PAD_RAW_VALUE, np.int16)
        padded[:height, :width] = pixels
        blocks = padded.reshape(rows, side, cols, side, 3).transpose(
            0, 2, 1, 3, 4
        )
        return np.ascontiguousarray(blocks).reshape(rows * cols, side, side, 3)
