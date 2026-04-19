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

"""Shared processing utilities for Gemma4 image and video processors."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import huggingface_hub
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def aspect_ratio_preserving_resize(
    image: Image.Image,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> Image.Image:
    """Resize *image* preserving aspect ratio to fit within *max_patches*.

    Target dimensions are the largest that:

    1. Produce at most *max_patches* patches when patchified with *patch_size*.
    2. Have height and width divisible by ``pooling_kernel_size * patch_size``.
    """
    width, height = image.size
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = math.floor(ideal_height / side_mult) * side_mult
    target_width = math.floor(ideal_width / side_mult) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError("Attempting to resize to a 0 x 0 image.")

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(
            math.floor(width / height) * side_mult, max_side_length
        )
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            math.floor(height / width) * side_mult, max_side_length
        )

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to "
            f"[{target_height}x{target_width}] but this exceeds "
            f"{max_patches} patches with patch_size {patch_size}"
        )

    if target_height == height and target_width == width:
        return image

    return image.resize((target_width, target_height), Image.Resampling.BICUBIC)


def load_processor_config(
    model_path: str,
    revision: str | None = None,
) -> dict[str, Any]:
    """Load ``processor_config.json`` from a local dir or HF hub.

    Returns the parsed JSON dict, or an empty dict on failure.
    """
    local = Path(model_path) / "processor_config.json"
    if local.is_file():
        with open(local) as f:
            return json.load(f)

    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=model_path,
            filename="processor_config.json",
            revision=revision,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.debug(
            "processor_config.json not found for '%s'; using defaults.",
            model_path,
        )
        return {}
