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
"""Resizes and patchifies an image for the vision encoder, in numpy."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from PIL import Image

QWEN3VL_MAX_PIXELS = 16777216
"""Qwen3VL's default ``longest_edge``: the post-resize pixel ceiling per image.

The process-wide resolution bound, so it is also the image hash's size tier and
the largest image an arch config can be asked to size a cache entry for.
"""

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Rescales the image so that the following conditions are met.

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def qwen3vl_image_preprocessing(
    image: Image.Image,
    *,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 65536,
    max_pixels: int = QWEN3VL_MAX_PIXELS,
) -> tuple[npt.NDArray[np.float32], tuple[int, int, int]]:
    """Preprocess image for Qwen3VL vision model.

    This function matches the behavior of the transformers Qwen2VLImageProcessorFast.
    It uses smart_resize to calculate target dimensions and resizes the PIL image
    before processing.

    Args:
        image: PIL Image in RGB mode to preprocess (not resized)
        patch_size: Patch size for vision transformer (default 16)
        temporal_patch_size: Temporal patch size (default 2)
        merge_size: Spatial merge size (default 2)
        min_pixels: Minimum pixels for smart_resize (default 65536)
        max_pixels: Maximum pixels for smart_resize

    Returns:
        Tuple of (pixel_values, image_grid_thw) where:
        - pixel_values: Flattened patch values as numpy array
        - image_grid_thw: Grid dimensions (temporal, height, width)
    """
    width, height = image.size

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    if resized_height != height or resized_width != width:
        image = image.resize(
            (resized_width, resized_height), resample=Image.Resampling.BICUBIC
        )
        height, width = resized_height, resized_width

    height_patches = height // patch_size
    width_patches = width // patch_size

    img_array = np.array(image, dtype=np.float32)

    # Qwen3VL uses mean=0.5, std=0.5 for normalization to [-1, 1] range
    # Also, Rescale to [0, 1] using the same rescale_factor as transformers (1/255)
    # img_array = img_array / 255.0
    # This is equivalent to: (x - 0.5) / 0.5 = 2*x - 1
    img_array = (img_array - (0.5 * 255.0)) / (0.5 * 255.0)

    patches = np.array([img_array])  # Shape: (n_images, height, width, 3)
    patches = patches.transpose(
        0, 3, 1, 2
    )  # Shape: (n_images, 3, height, width)

    channel = patches.shape[1]
    grid_h, grid_w = height_patches, width_patches

    # Handle temporal dimension padding if not divisible by temporal_patch_size
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = np.repeat(
            patches[-1][np.newaxis],
            temporal_patch_size - (patches.shape[0] % temporal_patch_size),
            axis=0,
        )
        patches = np.concatenate([patches, repeats], axis=0)

    # For images, grid_t should be 1 (single temporal group)
    grid_t = 1

    # Now reshape with spatial merging
    # Grid dimensions are divisible by merge_size (after padding if needed)
    patches = patches.reshape(
        grid_t,  # Temporal groups (1 for images)
        temporal_patch_size,  # Patches per temporal group (2)
        channel,  # RGB channels (3)
        grid_h // merge_size,  # Spatial groups in height
        merge_size,  # Patches per spatial group (2)
        patch_size,  # Patch height (16 for Qwen3VL)
        grid_w // merge_size,  # Spatial groups in width
        merge_size,  # Patches per spatial group (2)
        patch_size,  # Patch width (16 for Qwen3VL)
    )

    # Reorder dimensions to get the correct patch ordering
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    image_grid_thw = (grid_t, grid_h, grid_w)

    return flatten_patches, image_grid_thw
