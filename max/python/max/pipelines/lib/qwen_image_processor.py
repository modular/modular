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
"""MAX-native image preprocessing helpers for Qwen image pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.lib import float32_to_bfloat16_as_uint16
from PIL import Image

_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_NORM_SCALE = (1.0 / (255.0 * _IMAGENET_STD)).astype(np.float32)
_NORM_OFFSET = (-_IMAGENET_MEAN / _IMAGENET_STD).astype(np.float32)


def qwen2_5vl_prompt_image_preprocessing(
    image: Image.Image,
    *,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> tuple[npt.NDArray[np.uint16], tuple[int, int, int]]:
    """Preprocess a prompt image for MAX-native Qwen image edit encoding."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.asarray(image, dtype=np.float32)
    np.multiply(img_array, _NORM_SCALE, out=img_array)
    np.add(img_array, _NORM_OFFSET, out=img_array)

    height, width = img_array.shape[:2]
    grid_h = height // patch_size
    grid_w = width // patch_size

    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise ValueError(
            f"Spatial merging is not possible because grid_h {grid_h} % merge_size {merge_size} != 0 or grid_w {grid_w} % merge_size {merge_size} != 0"
        )

    patches = img_array[np.newaxis, ...]
    patches = patches.transpose(0, 3, 1, 2)
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = np.repeat(
            patches[-1][np.newaxis],
            temporal_patch_size - (patches.shape[0] % temporal_patch_size),
            axis=0,
        )
        patches = np.concatenate([patches, repeats], axis=0)

    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flattened_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    return float32_to_bfloat16_as_uint16(flattened_patches), (
        grid_t,
        grid_h,
        grid_w,
    )


class Qwen2_5VLPromptImageProcessor:
    """Process prompt images for MAX-native Qwen image edit pipelines."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size

    def __call__(
        self,
        images: list[Image.Image] | Image.Image,
        return_tensors: str = "np",
        **kwargs: Any,
    ) -> tuple[dict[str, npt.NDArray[Any]], list[npt.NDArray[np.uint16]]]:
        """Preprocess one or more prompt images and return patch tensors."""
        del return_tensors, kwargs
        if isinstance(images, Image.Image):
            images = [images]

        pixel_values_list: list[npt.NDArray[np.uint16]] = []
        image_grid_thw_list: list[tuple[int, int, int]] = []

        for image in images:
            pixel_values, image_grid_thw = qwen2_5vl_prompt_image_preprocessing(
                image,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
            )
            pixel_values_list.append(pixel_values)
            image_grid_thw_list.append(image_grid_thw)

        return {
            "concatenated_pixel_values": np.vstack(pixel_values_list),
            "image_grid_thw": np.array(image_grid_thw_list, dtype=np.int32),
        }, pixel_values_list

    def preprocess(
        self,
        images: list[Image.Image] | Image.Image,
        return_tensors: str = "np",
        **kwargs: Any,
    ) -> tuple[dict[str, npt.NDArray[Any]], list[npt.NDArray[np.uint16]]]:
        """Alias matching the HuggingFace image processor API."""
        return self(images, return_tensors=return_tensors, **kwargs)
