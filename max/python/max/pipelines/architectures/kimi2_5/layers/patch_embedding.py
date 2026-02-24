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

"""Patch embedding layer equivalent to MoonVision3dPatchEmbed (Kimi K2.5).

Reference: nvidia/Kimi-K2.5-NVFP4 modeling_kimi_k25.py
- MoonVision3dPatchEmbed: proj (Conv2d) + pos_emb (Learnable2DInterpPosEmbDivided_fixed).

Checkpoint weight names and shapes (vision_tower.patch_embed.*, BF16):
  - vision_tower.patch_embed.pos_emb.weight   [64, 64, 1152]
  - vision_tower.patch_embed.proj.bias        [1152]
  - vision_tower.patch_embed.proj.weight      [1152, 3, 14, 14]  (out_ch, in_ch, kH, kW)
"""

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.conv import Conv2d
from max.nn.layer import Module, Shardable


class Learnable2DInterpPosEmbDividedFixed(Module, Shardable):
    """Learnable 2D spatial position grid for MoonVision3dPatchEmbed.

    Holds the weight so checkpoint key patch_embed.pos_emb.weight loads.
    Reference forward: for each (t, h, w) in grid_thws, interpolate this grid
    to (h, w) if needed, add 1D temporal sincos, add to the corresponding
    slice of x. That requires per-image variable (h,w), dynamic slicing, and
    bicubic interpolation, which are not expressible in the graph API yet.
    So __call__ returns x unchanged; the weight is present for loading only.
    Shardable by replication only.
    """

    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        dim: int = 1152,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
        is_sharding: bool = False,
    ) -> None:
        """Initializes the 2D position embedding grid.

        Args:
            height: Height of the learnable 2D grid (init_pos_emb_height).
            width: Width of the learnable 2D grid (init_pos_emb_width).
            dim: Embedding dimension (vt_hidden_size).
            dtype: Data type for the weight.
            device: Device to place the weight on.
            is_sharding: If True, skip weight creation; used by :meth:`shard`.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
        self.dtype = dtype
        self._sharding_strategy: ShardingStrategy | None = None

        if not is_sharding:
            # Learnable 2D grid. Name "weight" so when used as patch_embed.pos_emb
            # the checkpoint key patch_embed.pos_emb.weight loads.
            self.weight = Weight(
                name="weight",
                dtype=dtype,
                shape=(height, width, dim),
                device=device,
            )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the position embedding sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy. Only replication is supported."""
        if not strategy.is_replicate:
            raise ValueError(
                "Learnable2DInterpPosEmbDividedFixed only supports replicate "
                "sharding strategy"
            )
        self.weight.sharding_strategy = strategy
        self._sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Learnable2DInterpPosEmbDividedFixed]:
        """Creates sharded views of this layer across devices (replicated)."""
        if not self._sharding_strategy:
            raise ValueError(
                "Learnable2DInterpPosEmbDividedFixed cannot be sharded because "
                "no sharding strategy was set."
            )
        device_list = list(devices)
        weight_shards = self.weight.shard(device_list)
        shards = []
        for device, weight_shard in zip(
            device_list, weight_shards, strict=True
        ):
            sharded = Learnable2DInterpPosEmbDividedFixed(
                height=self.height,
                width=self.width,
                dim=self.dim,
                dtype=self.dtype,
                device=device,
                is_sharding=True,
            )
            sharded.weight = weight_shard
            sharded._sharding_strategy = self._sharding_strategy
            shards.append(sharded)
        return shards

    def __call__(
        self,
        x: TensorValue,
        grid_thws: TensorValue,
    ) -> TensorValue:
        """Add position embeddings to x. Currently a no-op; weight is for loading only.

        Full application would: for each (t, h, w) in grid_thws, slice x by
        cumulative lengths, interpolate self.weight to (h, w) if (h,w) != (64,64),
        add time_weight[0:t], add to slice. Not implemented in graph API.

        Args:
            x: (L, dim) patch embeddings.
            grid_thws: (N, 3) temporal, height, width per image (unused).

        Returns:
            x unchanged until full implementation is added.
        """
        _ = grid_thws  # Reserved for when per-image interpolation/slicing is implemented.
        return x


class PatchEmbeddingLayer(Module, Shardable):
    """Equivalent to MoonVision3dPatchEmbed from Kimi K2.5 (MoonViT3d).

    Implements:
    1. Projection: Conv2d(in_channels, hidden_size, kernel_size=patch_size,
       stride=patch_size). Same as reference. Input (L, 3, patch_size, patch_size)
       -> output (L, hidden_size).
    2. Position embedding: Learnable2DInterpPosEmbDividedFixed (pos_emb) holds
       the 2D grid weight; full application with grid_thws is not yet implemented.

    Shardable by replication only.
    """

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 1152,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
        has_bias: bool = True,
        is_sharding: bool = False,
    ) -> None:
        """Initializes the patch embedding layer to match MoonVision3dPatchEmbed.

        Args:
            patch_size: Spatial size of each patch (height and width).
            in_channels: Number of input channels (3 for RGB).
            hidden_size: Output embedding dimension (vt_hidden_size).
            init_pos_emb_height: Height of learnable 2D position grid (vision_config).
            init_pos_emb_width: Width of learnable 2D position grid.
            init_pos_emb_time: Number of temporal steps for 1D sincos time embedding.
            dtype: Data type for weights and computation.
            device: Device to place weights on.
            has_bias: Whether the Conv2d projection includes a bias term.
            is_sharding: If True, skip child layer creation; used by :meth:`shard`.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.dtype = dtype
        self.has_bias = has_bias
        self._sharding_strategy: ShardingStrategy | None = None

        if not is_sharding:
            # 1. Projection: Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
            #    Matches reference exactly. permute=True for NCHW input -> NHWC internally.
            self.proj = Conv2d(
                in_channels=in_channels,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                has_bias=has_bias,
                dtype=dtype,
                device=device,
                permute=True,
                name="proj",
            )

            # 2. Position embedding (Learnable2DInterpPosEmbDivided_fixed). Holds 2D grid
            #    so checkpoint key patch_embed.pos_emb.weight loads. Full application
            #    (per-image (t,h,w), interpolation, time_weight) is not implemented in graph.
            self.pos_emb = Learnable2DInterpPosEmbDividedFixed(
                height=init_pos_emb_height,
                width=init_pos_emb_width,
                dim=hidden_size,
                dtype=dtype,
                device=device,
            )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the patch embedding sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy. Only replication is supported."""
        if not strategy.is_replicate:
            raise ValueError(
                "PatchEmbeddingLayer only supports replicate sharding strategy"
            )
        self.proj.sharding_strategy = strategy
        self.pos_emb.sharding_strategy = strategy
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[PatchEmbeddingLayer]:
        """Creates sharded views of this layer across devices (replicated)."""
        if not self._sharding_strategy:
            raise ValueError(
                "PatchEmbeddingLayer cannot be sharded because no sharding "
                "strategy was set."
            )
        device_list = list(devices)
        proj_shards = self.proj.shard(device_list)
        pos_emb_shards = self.pos_emb.shard(device_list)
        shards = []
        for device, proj_shard, pos_emb_shard in zip(
            device_list, proj_shards, pos_emb_shards, strict=True
        ):
            sharded = PatchEmbeddingLayer(
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                hidden_size=self.hidden_size,
                init_pos_emb_height=self.init_pos_emb_height,
                init_pos_emb_width=self.init_pos_emb_width,
                init_pos_emb_time=self.init_pos_emb_time,
                dtype=self.dtype,
                device=device,
                has_bias=self.has_bias,
                is_sharding=True,
            )
            sharded.proj = proj_shard
            sharded.pos_emb = pos_emb_shard
            sharded._sharding_strategy = self._sharding_strategy
            shards.append(sharded)
        return shards

    def __call__(
        self,
        pixel_values: TensorValue,
        grid_thws: TensorValue,
    ) -> TensorValue:
        """Patch projection. Optionally add position embedding when supported.

        Matches reference: x = self.proj(x).view(x.size(0), -1) then x = self.pos_emb(x, grid_thws).
        pos_emb currently returns x unchanged (full application not in graph yet).

        Args:
            pixel_values: (n_patches, in_channels, patch_size, patch_size) in NCHW format,
                i.e. n_patches patches of shape (3, 14, 14).
            grid_thws: (n_images, 3) temporal, height, width per image.

        Returns:
            Tensor of shape (n_patches, hidden_size).
        """
        # Conv2d with permute: input (n_patches, 3, patch_size, patch_size) -> output (n_patches, hidden_size, 1, 1)
        patch_embeds = self.proj(pixel_values)
        # Flatten spatial dims: (n_patches, hidden_size, 1, 1) -> (n_patches, hidden_size)
        x = ops.reshape(patch_embeds, [patch_embeds.shape[0], self.hidden_size])
        x = self.pos_emb(x, grid_thws)
        return x
