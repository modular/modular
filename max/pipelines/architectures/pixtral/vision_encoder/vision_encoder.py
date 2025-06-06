# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.nn import Conv2DV1
from max.nn.layer import Layer
from max.nn.norm import RMSNormV1

from .rotary_embedding_2d import RotaryEmbedding2D, patch_position_ids
from .transformer import Transformer


@dataclass
class VisionEncoder(Layer):
    """The bare Pixtral vision encoder outputting raw hidden-states without any
    specific head on top.

    It tokenizes the list of images and returns a representation of these images
    embeddings of patches.
    """

    patch_conv: Conv2DV1
    layer_norm: RMSNormV1
    patch_positional_embedding: RotaryEmbedding2D
    transformer: Transformer
    dtype: DType
    patch_size: int = 16
    max_image_size: int = 1024

    def __call__(
        self, imgs: list[TensorValueLike], attention_mask: TensorValueLike
    ):
        """
        imgs: list of images of shape = (height, width, num_channels)
        """
        # Images go through a convolution independently to get patched.
        # Returns a list of [batch_size, hidden_size, height/patch_size, width/patch_size] tensors
        patch_embeds_list = [
            self.patch_conv(ops.unsqueeze(TensorValue(img).cast(self.dtype), 0))
            for img in imgs
        ]

        # Flatten all images to a single tensor of patches of size (n_patches=seq_length, hidden_size).
        # 1. Flattens each image's patches to (batch_size, n_patches in image, hidden_size).
        # 2. Concat patches vertically on dim 1 to get a sequence of all patches
        # TODO(MSDK-1192): replace transpose(1, 2) by permute(0, 2, 1)
        # TODO(MSDK-1195): replace p.reshape((p.shape[0], p.shape[1], -1)) by p.flatten(2)
        patch_embeds = ops.concat(
            [  # p.shape = batch_size, patches_per_height, patches_per_width, hidden_size
                p.reshape((p.shape[0], -1, p.shape[3]))
                for p in patch_embeds_list
            ],
            axis=1,
        )

        # Pre-attention layer normalization
        patch_embeds = self.layer_norm(patch_embeds)

        # Get unique ids of tokens (patches) based on row and col idx in the image (position).
        # These help the model understand the spatial layout of the image.
        position_ids = patch_position_ids(
            patch_embeds_list, max_width=self.max_image_size // self.patch_size
        )

        # Positional Encodings
        # map each position id to its corresponding embedding representing that position
        position_embedding = self.patch_positional_embedding(
            patch_embeds, position_ids
        )

        encoder_output = self.transformer(
            patch_embeds=patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embedding,
        )

        return encoder_output
