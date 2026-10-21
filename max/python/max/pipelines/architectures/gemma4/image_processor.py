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

"""Gemma4 image processor using pure numpy/PIL (no torch).

Implements aspect-ratio-preserving resize so that each image's dimensions
are multiples of ``pooling_kernel_size * patch_size`` and the total patch
count stays within a budget derived from ``max_soft_tokens``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import to_rgb
from PIL import Image

from .processing_utils import (
    SUPPORTED_SOFT_TOKENS,
    aspect_ratio_preserving_resize,
)


class Gemma4ImageProcessor:
    """Pure numpy/PIL image processor for Gemma4.

    Implements aspect-ratio-preserving resize so that each image's dimensions
    are multiples of ``pooling_kernel_size * patch_size`` and the total patch
    count stays within a budget derived from ``max_soft_tokens``.
    """

    def __init__(
        self,
        patch_size: int = 16,
        max_soft_tokens: int = 280,
        pooling_kernel_size: int = 3,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = False,
        image_mean: tuple[float, ...] = (0.0, 0.0, 0.0),
        image_std: tuple[float, ...] = (1.0, 1.0, 1.0),
        **unused_kwargs,
    ) -> None:
        if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {SUPPORTED_SOFT_TOKENS}, "
                f"got {max_soft_tokens}."
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    def __call__(
        self,
        images: list[Image.Image],
    ) -> tuple[
        list[npt.NDArray[np.float32]],
        list[npt.NDArray[np.int32]],
        list[int],
    ]:
        """Process a list of PIL images into patchified tensors.

        Returns:
            A 3-tuple ``(pixel_values_list, pixel_position_ids_list,
            num_soft_tokens_per_image)`` where:

            * ``pixel_values_list[i]``: float32 array of shape
              ``[num_patches_i, patch_size² * 3]``, pixel values in ``[0, 1]``.
            * ``pixel_position_ids_list[i]``: int32 array of shape
              ``[num_patches_i, 2]`` with ``(x, y)`` grid coordinates for each
              patch.
            * ``num_soft_tokens_per_image[i]``: number of output soft tokens
              (= ``num_patches_i // pooling_kernel_size²``).
        """
        max_patches = self.max_soft_tokens * self.pooling_kernel_size**2
        patch_size = self.patch_size
        pooling_kernel_size = self.pooling_kernel_size

        pixel_values_list: list[npt.NDArray[np.float32]] = []
        pixel_position_ids_list: list[npt.NDArray[np.int32]] = []
        num_soft_tokens_per_image: list[int] = []

        for image in images:
            image = to_rgb(image)

            if self.do_resize:
                image = aspect_ratio_preserving_resize(
                    image,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                )

            img_array = np.array(image, dtype=np.float32)
            if self.do_rescale:
                img_array = img_array * self.rescale_factor
            img_array = np.transpose(img_array, (2, 0, 1))  # CHW
            if self.do_normalize:
                img_array = (
                    img_array - self.image_mean[:, None, None]
                ) / self.image_std[:, None, None]

            c, h, w = img_array.shape
            patch_height = h // patch_size
            patch_width = w // patch_size
            num_patches = patch_height * patch_width

            patches = (
                img_array.reshape(
                    c, patch_height, patch_size, patch_width, patch_size
                )
                .transpose(1, 3, 2, 4, 0)
                .reshape(num_patches, patch_size * patch_size * c)
            )

            grid_x, grid_y = np.meshgrid(
                np.arange(patch_width, dtype=np.int32),
                np.arange(patch_height, dtype=np.int32),
            )
            position_ids = np.stack([grid_x, grid_y], axis=-1).reshape(
                num_patches, 2
            )

            num_soft = num_patches // (pooling_kernel_size**2)

            pixel_values_list.append(patches)
            pixel_position_ids_list.append(position_ids)
            num_soft_tokens_per_image.append(num_soft)

        return (
            pixel_values_list,
            pixel_position_ids_list,
            num_soft_tokens_per_image,
        )
