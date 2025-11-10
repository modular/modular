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

from max.driver import Tensor
from max.graph import (
    DeviceRef,
    TensorValue,
)
from max.nn import (
    Conv2d,
    Module,
)
from max.nn.embedding import Embedding

from ..model_config import Gemma3ForConditionalGenerationConfig


# ⚠️ in line with MLX-VLM implementation
class Gemma3VisionEmbeddings(Module):
    """Implements patch embeddings for SigLIP vision model."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        device: DeviceRef | None = None,
    ) -> None:
        """Initializes the vision embeddings module."""
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_channels = config.vision_config.num_channels  # 3
        self.embed_dim = config.vision_config.hidden_size  # 1152
        self.image_size = config.vision_config.image_size  # 896
        self.patch_size = config.vision_config.patch_size  # 14
        self.dtype = config.dtype

        self.patch_embedding = Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,  # 1152
            kernel_size=self.patch_size,  # 14
            stride=self.patch_size,  # 14
            padding=0,  # "valid" padding
            has_bias=True,
            dtype=self.dtype,
            device=device,
        )

        self.num_patches = (
            self.image_size // self.patch_size
        ) ** 2  # 4096 = (896 // 14)^2
        self.num_positions = self.num_patches

        self.position_embedding = Embedding(
            vocab_size=self.num_positions,
            hidden_dim=self.embed_dim,
            dtype=self.dtype,
            device=device,
        )

    # @property
    # def sharding_strategy(self) -> ShardingStrategy | None:
    #     """Get the embedding sharding strategy."""
    #     return self.patch_embedding.sharding_strategy

    # @sharding_strategy.setter
    # def sharding_strategy(self, strategy: ShardingStrategy) -> None:
    #     """Set the sharding strategy for the patch, class, and position
    #     embeddings.

    #     Args:
    #         strategy: The strategy describing the embeddings' sharding.
    #     """
    #     if not strategy.is_replicate:
    #         raise ValueError(
    #             "only replicate is supported for Gemma3VisionEmbeddings, "
    #             "currently"
    #         )

    #     self.patch_embedding.sharding_strategy = strategy
    #     self.position_embedding.sharding_strategy = strategy

    # def shard(
    #     self, devices: Iterable[DeviceRef]
    # ) -> list[Gemma3VisionEmbeddings]:
    #     """Creates sharded views of this vision embeddings across multiple devices.

    #     Args:
    #         devices: Iterable of devices to place the shards on.

    #     Returns:
    #         List of sharded Gemma3VisionEmbeddings instances, one for each device.
    #     """
    #     # This should be set unconditionally in the constructor.
    #     assert self.sharding_strategy

    #     # Get sharded weights
    #     patch_embedding_shards = self.patch_embedding.shard(devices)
    #     position_embedding_shards = self.position_embedding.shard(devices)

    #     shards = []
    #     for device, patch_shard, pos_shard in zip(
    #         devices,
    #         patch_embedding_shards,
    #         position_embedding_shards,
    #         strict=True,
    #     ):
    #         # Create the new sharded embedding.
    #         sharded = Gemma3VisionEmbeddings(self.config, device)

    #         # Assign the sharded weights.
    #         sharded.patch_embedding = patch_shard
    #         sharded.position_embedding = pos_shard

    #         shards.append(sharded)

    #     return shards

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        """Computes embeddings for input pixel values.

        Args:
            pixel_values: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Embeddings tensor of shape (batch_size, num_positions, embed_dim).
        """
        # ⚠️ based on MLX-VLM https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma3/vision.py#L137
        patch_embeddings = self.patch_embedding(pixel_values)
        patch_embeddings = patch_embeddings.flatten(1, 2)
        position_ids = Tensor.arange(self.num_positions)
        embeddings = patch_embeddings
        embeddings += self.position_embedding(position_ids)
        return embeddings
