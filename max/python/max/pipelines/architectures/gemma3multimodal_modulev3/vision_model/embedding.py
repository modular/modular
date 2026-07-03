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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.conv import Conv2d
from max.experimental.nn.embedding import Embedding
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops

from ..model_config import Gemma3ForConditionalGenerationConfig


class Gemma3VisionEmbeddings(Module[[Tensor], Tensor]):
    """Vision patch embeddings with positional encoding."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
    ) -> None:
        super().__init__()
        self.embed_dim = config.vision_config.hidden_size
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size

        self.patch_embedding = Conv2d(
            in_channels=config.vision_config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            has_bias=True,
            permute=True,
            dtype=DType.bfloat16,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.position_embedding = Embedding(
            self.num_patches,
            dim=self.embed_dim,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Forward pass of vision embeddings.

        Args:
            pixel_values: Input images [batch_size, channels, height, width].

        Returns:
            Embeddings [batch_size, num_patches, hidden_size].
        """
        batch_size = pixel_values.shape[0]
        max_im_h = pixel_values.shape[2]
        max_im_w = pixel_values.shape[3]

        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten spatial dimensions and transpose -> [batch, num_patches, embed_dim]
        embeddings = F.flatten(patch_embeds, start_dim=2)
        embeddings = embeddings.transpose(1, 2)

        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size
        total_patches = max_nb_patches_h * max_nb_patches_w

        position_ids = Tensor.from_graph_value(
            ops.range(
                start=0,
                stop=self.num_patches,
                step=1,
                out_dim=total_patches,
                device=pixel_values.device,
                dtype=DType.int32,
            )
        )
        position_ids = F.unsqueeze(position_ids, 0)  # [1, total_patches]
        position_ids = Tensor.from_graph_value(
            ops.tile(TensorValue(position_ids), [batch_size, 1])
        )

        position_embeds = self.position_embedding(position_ids)

        embeddings = embeddings + position_embeds

        return embeddings
