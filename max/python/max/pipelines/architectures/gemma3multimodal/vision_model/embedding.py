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

from collections.abc import Iterable

from max.driver import Tensor
from max.dtype import DType
from max.graph import (
    DeviceRef,
    ops,
    ShardingStrategy,
    TensorValue,
    Weight
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
            stride=self.patch_size,       # 14
            padding=0,                    # "valid" padding
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

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the embedding sharding strategy."""
        return self.patch_embedding.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the patch, class, and position
        embeddings.

        Args:
            strategy: The strategy describing the embeddings' sharding.
        """
        if not strategy.is_replicate:
            raise ValueError(
                "only replicate is supported for Gemma3VisionEmbeddings, "
                "currently"
            )

        self.patch_embedding.sharding_strategy = strategy
        self.position_embedding.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma3VisionEmbeddings]:
        """Creates sharded views of this vision embeddings across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Gemma3VisionEmbeddings instances, one for each device.
        """
        # This should be set unconditionally in the constructor.
        assert self.sharding_strategy

        # Get sharded weights
        patch_embedding_shards = self.patch_embedding.shard(devices)
        position_embedding_shards = self.position_embedding.shard(devices)

        shards = []
        for device, patch_shard, pos_shard in zip(
            devices,
            patch_embedding_shards,
            position_embedding_shards,
            strict=True,
        ):
            # Create the new sharded embedding.
            sharded = Gemma3VisionEmbeddings(self.config, device)

            # Assign the sharded weights.
            sharded.patch_embedding = patch_shard
            sharded.position_embedding = pos_shard

            shards.append(sharded)

        return shards

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        # ⚠️
        batch_size = pixel_values.shape[0]
        max_im_h = pixel_values.shape[2]
        max_im_w = pixel_values.shape[3]

        # ✅ based on MLX-VLM https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma3/vision.py#L137
        # permute to [NHWC]
        # pixel_values = ops.permute(pixel_values, [0, 2, 1, 3])
        patch_embeddings = self.patch_embedding(pixel_values)
        # # permute back to [NCHW] BORROWED FROM IDEFICES3
        # patch_embeddings = ops.permute(patch_embeddings, [0, 3, 1, 2])
        # # flatten to [N, embed_dim, num_patches]
        # patch_embeddings = ops.flatten(patch_embeddings, start_dim=2)
        # # transpose to [N, num_patches, embed_dim]
        # embeddings = ops.tranpose(patch_embeddings, 1, 2)


        # ⚠️ BORROWED HEAVILY FROM IDEFICS3
        # position_ids = Tensor.zeros([0, self.num_positions], self.dtype, self.devices[0].to_device())
        position_ids = Weight(
            "position_ids",
            dtype=DType.uint32,
            shape=(0, self.num_positions),
            device=self.devices[0],
        )
        # max_nb_patches_h = max_im_h // self.patch_size
        # max_nb_patches_w = max_im_w // self.patch_size
        # total_patches = max_nb_patches_h * max_nb_patches_w

        # # Create position IDs: [0, 1, 2, ..., total_patches-1] for each batch
        # # Generate 2D tensor with shape [batch_size, total_patches]
        # position_ids = ops.range(
        #     start=0,
        #     stop=self.num_patches,
        #     step=1,
        #     out_dim=total_patches,
        #     device=self.devices[0],
        #     dtype=DType.int32,
        # )  # [total_patches]
        # position_ids = ops.unsqueeze(position_ids, 0)  # [1, total_patches]
        # position_ids = ops.tile(
        #     position_ids, [batch_size, 1]
        # )  # [batch_size, total_patches]


        # ✅
        embeddings = patch_embeddings
        embeddings += self.position_embedding(position_ids)
        return embeddings
