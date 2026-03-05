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

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from max.graph import TensorValue, ops


def causal_attention_mask_2d_from_imgs(
    imgs: list[npt.NDArray[Any]],
    patch_size: int,
    batch_size: int,
    fill_val: float = -10000.0,
) -> npt.NDArray[np.float32]:
    """Generates a 2D mask to ensure different blocks of patches (images) can only attend
    to patches within their respective block (image).

    Args:

        imgs: A list of images (blocks). Each image is of shape
        (num_channels, height, width).

        patch_size: size of one dim of each patch in the image.

        batch_size: num of images.


    Returns an ndarray of shape (batch_size, 1, seq_len, seq_len) representing the
    attention mask for the blocks of patches attended to by the transformer.
    """
    # generate list of (num_patches_in_height * num_patches_in_width) for each image
    # Images are now in CHW format, so height is shape[1] and width is shape[2]
    num_patches_list = [
        img.shape[1] // patch_size * img.shape[2] // patch_size for img in imgs
    ]

    # seq_length is number of patches in all images
    seq_len = sum(num_patches_list)
    mask_shape = (seq_len, seq_len)

    # TODO(KERN-782): This fill_val should be -inf but softmax saturates with NaNs.
    fill_matrix = np.full(mask_shape, fill_val, dtype=np.float32)

    # block_end_idx and block_start_idx are calculated using cumulative sums of
    # patch_embeds_list. These indicate the starting and ending indices of each
    # block of embeddings.
    block_end_idx = np.cumsum(num_patches_list)
    block_start_idx = np.cumsum(np.concatenate(([0], num_patches_list[:-1])))

    # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
    for start, end in zip(block_start_idx, block_end_idx, strict=True):
        fill_matrix[int(start) : int(end), int(start) : int(end)] = 0

    # Expand the mask dimensions to match the expected input shape
    fill_matrix = np.expand_dims(fill_matrix, axis=(0, 1))  # Add two new axes
    fill_matrix = np.broadcast_to(
        fill_matrix, (batch_size, 1, seq_len, seq_len)
    )
    return fill_matrix


def rotate_half(x: TensorValue) -> TensorValue:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.concat((-x2, x1), axis=-1)
