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
"""Vision embeddings for IDEFICS3 model (ModuleV3)."""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.conv import Conv2d
from max.experimental.nn.embedding import Embedding
from max.experimental.tensor import Tensor

from ..model_config import Idefics3VisionConfig


class Idefics3VisionEmbeddings(Module[[Tensor], Tensor]):
    """Vision embeddings with patch embeddings and position encoding (ModuleV3).

    Converts images into patch embeddings and adds learned position embeddings.
    """

    def __init__(self, config: Idefics3VisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2

        # Patch embedding using Conv2d with stride=patch_size.
        # permute=True means input/output are NCHW and weights are FCRS
        # (PyTorch format), so no manual transpositions needed.
        self.patch_embedding = Conv2d(
            kernel_size=self.patch_size,
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            stride=self.patch_size,
            padding=0,
            has_bias=True,
            permute=True,
        )

        # Position embedding table
        self.position_embedding = Embedding(
            self.num_patches, dim=self.embed_dim
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Compute patch embeddings from pixel values.

        Args:
            pixel_values: Input images of shape [batch_size, channels, height, width].

        Returns:
            Embeddings of shape [batch_size, num_patches, hidden_size].
        """
        batch_size = pixel_values.shape[0]

        # Apply patch embedding (Conv2d with permute=True handles NCHW I/O).
        # Output: [batch_size, embed_dim, num_patches_h, num_patches_w]
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten spatial dims and transpose to [batch, num_patches, embed_dim]
        embeddings = F.flatten(patch_embeds, start_dim=2)
        embeddings = embeddings.transpose(1, 2)

        # Create position IDs: [0, 1, ..., num_patches-1] for each batch
        total_patches = self.num_patches
        position_ids = F.arange(
            start=0,
            stop=self.num_patches,
            step=1,
            out_dim=total_patches,
            device=pixel_values.device,
            dtype=DType.int32,
        )
        position_ids = F.unsqueeze(position_ids, 0)
        position_ids = F.tile(position_ids, [batch_size, 1])

        # Add position embeddings
        position_embeds = self.position_embedding(position_ids)
        embeddings = embeddings + position_embeds

        return embeddings
