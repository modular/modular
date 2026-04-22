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

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.conv import Conv2d
from max.nn.embedding import Embedding
from max.nn.layer import Module

from ..model_config import Gemma3ForConditionalGenerationConfig


class Gemma3VisionEmbeddings(Module):
    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_channels = config.vision_config.num_channels
        self.embed_dim = config.vision_config.hidden_size
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.dtype = DType.bfloat16

        self.patch_embedding = Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            has_bias=True,
            dtype=self.dtype,
            device=device,
            permute=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        self.position_embedding = Embedding(
            vocab_size=self.num_positions,
            hidden_dim=self.embed_dim,
            dtype=self.dtype,
            device=device,
        )

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        """Forward pass of vision embeddings.

        Args:
            pixel_values: Input images of shape [batch_size, channels, height, width].

        Returns:
            Embeddings of shape [batch_size, num_patches, hidden_size].
        """
        batch_size = pixel_values.shape[0]
        max_im_h = pixel_values.shape[2]
        max_im_w = pixel_values.shape[3]

        # apply conv2d to pixel values
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten spatial dimensions and transpose ->  [batch_size, num_patches, embed_dim]
        embeddings = ops.flatten(patch_embeds, start_dim=2)
        embeddings = ops.transpose(embeddings, 1, 2)

        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size
        total_patches = max_nb_patches_h * max_nb_patches_w

        # Create position IDs for each batch
        position_ids = ops.range(
            start=0,
            stop=self.num_patches,
            step=1,
            out_dim=total_patches,
            device=self.devices[0],
            dtype=DType.int32,
        )
        position_ids = ops.unsqueeze(position_ids, 0)  # [1, total_patches]
        position_ids = ops.tile(position_ids, [batch_size, 1])

        # Get position embeddings for the position IDs
        position_embeds = self.position_embedding(position_ids)

        embeddings = embeddings + position_embeds

        return embeddings

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.patch_embedding.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.patch_embedding.sharding_strategy = strategy
        self.position_embedding.weight.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma3VisionEmbeddings]:
        assert self.sharding_strategy

        patch_embedding_shards = self.patch_embedding.shard(devices)
        position_embedding_weight_shards = self.position_embedding.weight.shard(
            devices
        )

        shards = []
        for device, patch_shard, pos_weight_shard in zip(
            devices,
            patch_embedding_shards,
            position_embedding_weight_shards,
            strict=True,
        ):
            sharded = Gemma3VisionEmbeddings(self.config, device)

            sharded.patch_embedding = patch_shard
            sharded.position_embedding.weight = pos_weight_shard

            shards.append(sharded)

        return shards
