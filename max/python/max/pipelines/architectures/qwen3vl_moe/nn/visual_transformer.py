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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.comm import Allreduce
from max.nn.embedding import Embedding
from max.nn.kernels import spatial_merge
from max.nn.layer import LayerList, Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import LayerNorm

from ...qwen2_5vl.nn.vision_attention import DistributedVisionWindowAttention
from ..model_config import VisionConfig


class VisionPatchEmbed(Module, Shardable):
    """Generates patch embeddings from a tensor of pixel_values of patches using a Linear layer.

    This implementation uses a Linear layer instead of Conv3D, which is mathematically
    equivalent when stride equals kernel size (non-overlapping patches).

    Compared to Qwen2.5VL:
    - Conv3D projection has bias.
    - No re-ordering of patch embeddings after projection by window_index.
    """

    def __init__(
        self,
        dtype: DType,
        devices: Sequence[DeviceRef],
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.devices = devices or [DeviceRef.CPU()]
        self._sharding_strategy: ShardingStrategy | None = None

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Calculate input dimension for linear layer, equivalent to the flattened patch size
        self.patch_dim = (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

        # Create Linear layer instead of Conv3D, mathematically equivalent to Conv3D when stride = kernel size
        self.proj = Linear(
            in_dim=self.patch_dim,
            out_dim=embed_dim,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=None,
            has_bias=True,
        )

    def __call__(
        self,
        x: TensorValue,
    ) -> TensorValue:
        """Generates patch embeddings from pixel_values of patches (`x`).

        Uses Linear layer instead of Conv3D for patch embedding, which is mathematically
        equivalent when stride equals kernel size (non-overlapping patches).

        Args:
            x: tensor representing pixel values of shape [n_patches, patch_dim].

        Returns:
            a tensor of size (seq_len, hidden_size = embed_dim)
        """
        x = x.cast(self.proj.weight.dtype)

        # Shape: (n_patches_in_batch, patch_dim)
        assert x.shape[1] == self.patch_dim, (
            f"x.shape should be (n_patches, patch_dim) = {x.shape}, self.patch_dim = {self.patch_dim}"
        )

        # Apply linear transformation
        h = self.proj(x)

        seq_len = h.shape[0]

        h = h.reshape([-1, self.embed_dim]).rebind([seq_len, self.embed_dim])
        return h

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        assert strategy.is_replicate, (
            "VisionPatchEmbed only supports replicate sharding strategy"
        )
        self.proj.sharding_strategy = ShardingStrategy.replicate(
            strategy.num_devices
        )
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[VisionPatchEmbed]:
        if not self.sharding_strategy:
            raise ValueError(
                "VisionPatchEmbed layer cannot be sharded because no sharding strategy was provided."
            )
        proj_shards = self.proj.shard(devices)
        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = VisionPatchEmbed(
                dtype=self.proj.weight.dtype,
                devices=[device],
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                in_channels=self.in_channels,
                embed_dim=self.embed_dim,
            )
            sharded.proj = proj_shards[shard_idx]
            shards.append(sharded)
        return shards


class BilinearInterpolationPositionEmbedding(Module):
    """Bilinear interpolation position embedding layer for the qwen3VL vision model.

    This module performs bilinear interpolation of position embeddings to the desired spatial resolution.
    It supports sharding across multiple devices for distributed execution.

    It uses a Standard embedding layer
    * Embedding table of size (num_position_embeddings, hidden_size).
    * Assumes a square grid (e.g., 16x16 = 256 embeddings).
    * Computes num_grid_per_side = sqrt(num_position_embeddings).
    * Bilinear interpolation is applied for each patch:
        1. Retrieve embeddings of the 4 neighboring positions using nn.Embedding.
        2. Multiply each neighbor embedding by its interpolation weight.
        3. Sum the 4 weighted embeddings to get the final interpolated embedding.
    """

    def __init__(
        self,
        dtype: DType,
        device: DeviceRef,
        num_position_embeddings: int,
        hidden_size: int,
        spatial_merge_size: int,
    ):
        super().__init__()
        self.device = device
        self.dim = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.num_position_embeddings = num_position_embeddings
        self.num_grid_per_side = int(num_position_embeddings**0.5)
        self.embedding = Embedding(
            vocab_size=num_position_embeddings,
            hidden_dim=hidden_size,
            dtype=dtype,
            device=device,
            quantization_encoding=None,
        )

    def __call__(
        self,
        idxs: TensorValue,
        weights: TensorValue,
        grid_thw: TensorValue,
    ) -> TensorValue:
        """
        Args:
            idxs: tensor of shape (4, total_n_patches)
            weights: tensor of shape (4, total_n_patches)
            grid_thw: tensor of shape (n_images, 3)
            signal_buffers: sequence of buffers for peer-to-peer communication in allreduce

        Returns:
            tensor of shape (total_n_patches, hidden_size)
        """
        weights = weights.cast(self.embedding.weight.dtype)
        # pos_embeds_[i].shape: (total_n_patches, hidden_size)
        pos_embeds_0 = self.embedding(idxs[0, :])
        pos_embeds_1 = self.embedding(idxs[1, :])
        pos_embeds_2 = self.embedding(idxs[2, :])
        pos_embeds_3 = self.embedding(idxs[3, :])

        weighted_embeds_0 = pos_embeds_0 * weights[0, :]
        weighted_embeds_1 = pos_embeds_1 * weights[1, :]
        weighted_embeds_2 = pos_embeds_2 * weights[2, :]
        weighted_embeds_3 = pos_embeds_3 * weights[3, :]

        # shape: (total_n_patches, hidden_size)
        weighted_embeds = (
            weighted_embeds_0
            + weighted_embeds_1
            + weighted_embeds_2
            + weighted_embeds_3
        )
        # TODO: This will be done in the kernel for bilinear interpolation embedding. Kernel is WIP.
        # split weighted_embeds to have embeds of each video in a tensor.
        # for each video, repeat patches t times, then apply spatial merging which is described as follows:
        # Reshapes from (t, h, w, hidden_size) to (t, h//merge_size, merge_size, w//merge_size, merge_size, hidden_size)
        # Reorders dimensions to: (t, h//merge_size, w//merge_size, merge_size, merge_size, hidden_size)
        # Flattens the first 5 dimensions, combining temporal and spatial dimensions
        # Final shape becomes (t * h//merge_size * w//merge_size * merge_size * merge_size, hidden_size) = (t*h*w, hidden_size)

        # Example, for a 3D grid with t = 1:
        # weighted_embeds = weighted_embeds.reshape(1, h // self.spatial_merge_size, self.spatial_merge_size, w // self.spatial_merge_size, self.spatial_merge_size, -1)
        # weighted_embeds = weighted_embeds.permute(0, 1, 3, 2, 4, 5)
        # weighted_embeds = weighted_embeds.flatten(0, 4)
        # weighted_embeds = weighted_embeds.reshape(-1, weighted_embeds.shape[-1])

        weighted_embeds = spatial_merge(
            input=weighted_embeds,
            grid_thw=grid_thw,
            hidden_size=self.dim,
            merge_size=self.spatial_merge_size,
        )
        return weighted_embeds


@dataclass
class VisionRotaryEmbedding(Module):
    """Rotary embedding layer for the Qwen3VLMOE vision model.

    Differences compared to `max.nn.RotaryEmbedding`:

    - In _compute_inv_freqs, the head dimension (n) is divided by 2.
    - inv_freqs is cached instead of freqs.

    Compared to Qwen2.5VL:
    - No spatial merging while generating the rotary position embeddings.
    """

    dim: int
    n_heads: int
    theta: float
    _inv_freqs: TensorValue | None = None

    def __post_init__(self):
        super().__init__()

    def _compute_inv_freqs(self, device: DeviceRef) -> TensorValue:
        """Compute inverse frequencies for the given device."""
        n = (self.dim // self.n_heads) // 2
        # Note: using float64 to avoid an overflow on the exponential, then convert to float32.
        iota = ops.range(
            0,
            n - 1,
            2,
            out_dim=n // 2,
            device=device,
            dtype=DType.float64,
        )
        inv_freq = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
        return TensorValue(inv_freq)

    def inv_freqs(self, device: DeviceRef) -> TensorValue:
        """Compute and cache inverse frequencies for the given device."""
        if self._inv_freqs is None:
            self._inv_freqs = self._compute_inv_freqs(device)
        return self._inv_freqs

    def generate_rot_pos_embeddings(
        self,
        rot_pos_ids: TensorValue,
        max_grid_size: TensorValue,
        seq_len: Dim,
    ) -> tuple[TensorValue, TensorValue]:
        """Generates rotary position embeddings for a maximum sequence length of max_grid_size.

        Args:
            rot_pos_ids: tensor of shape (seq_len, 2) generated by data_processing.mrope_pos_ids_3d(grid_thw, spatial_merge_size).
            max_grid_size: max value in spatial dimensions in the grid of image and video patches.
                It represents the max no. of patches in an image or a frame. Used as the max positional embedding needed.
            seq_len: total number of patches in the sequence of images and videos. Its also pixel_values.shape[0].
        """
        # Generate rot_embs assuming max number of patches.
        t = ops.range(
            0,
            max_grid_size,
            1,
            out_dim="max_grid_size",
            device=rot_pos_ids.device,
            dtype=DType.float32,
        )
        rotary_pos_emb_full = ops.outer(t, self.inv_freqs(rot_pos_ids.device))
        # Retrieve position embeddings for each patch in input images or videos.
        rotary_pos_emb = ops.gather(rotary_pos_emb_full, rot_pos_ids, axis=0)
        rotary_pos_emb = rotary_pos_emb.flatten(1)

        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])

        # Generates a cos and a sin of rotary position embeddings which will be applied later. Shape = (seq_len, 2 * hidden_size).
        rotary_pos_emb = ops.concat((rotary_pos_emb, rotary_pos_emb), -1)

        freqs_cis = (ops.cos(rotary_pos_emb), ops.sin(rotary_pos_emb))
        return freqs_cis

    def __call__(
        self,
        x: TensorValue,
    ) -> TensorValue:
        raise NotImplementedError


class VisionMLP(Module, Shardable):
    """MLP layer for the Qwen3VL vision model.

    Compared to Qwen2.5VL:
    - MLP layer is 2 linear layers instead of 3 linear layers.
    - MLP layer uses gelu_pytorch_tanh activation instead of Gelu activation.
    """

    def __init__(
        self,
        dtype: DType,
        devices: Sequence[DeviceRef],
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.dtype = dtype
        self.devices = devices
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.linear_fc1 = Linear(
            in_dim=hidden_size,
            out_dim=intermediate_size,
            dtype=dtype,
            device=devices[0],
            has_bias=True,
        )
        self.linear_fc2 = Linear(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.linear_fc1(x)
        # This matches the "gelu_pytorch_tanh" activation in the torch.
        x = ops.gelu(x, approximate="tanh")
        x = self.linear_fc2(x)
        return x

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.linear_fc1.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if strategy.is_replicate:
            # Replicate all weights across devices
            self.linear_fc1.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.linear_fc2.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        else:
            # Tensor parallel: first linear rowwise, second linear columnwise
            self.linear_fc1.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.linear_fc2.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[VisionMLP]:
        # Shard underlying weights
        linear1_shards = self.linear_fc1.shard(devices)
        linear2_shards = self.linear_fc2.shard(devices)

        shards: list[VisionMLP] = []
        for idx, device in enumerate(devices):
            sharded = VisionMLP(
                dtype=self.dtype,
                devices=[device],
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
            )
            # Assign shards
            sharded.linear_fc1 = linear1_shards[idx]
            sharded.linear_fc2 = linear2_shards[idx]
            shards.append(sharded)
        return shards


class VisionBlock(Module):
    """Vision transformer block with distributed attention and MLP.

    Compared to Qwen2.5VL:
    - LayerNorm rather than RMSNorm for normalization.
    - MLP layer is 2 linear layers instead of 3 linear layers.
    - MLP layer uses gelu_pytorch_tanh activation instead of Gelu activation.
    """

    def __init__(
        self,
        dtype: DType,
        devices: Sequence[DeviceRef],
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dtype = dtype
        self.devices = devices
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Norms replicated across devices
        self.norm1 = LayerNorm(
            dims=hidden_size,
            devices=self.devices,
            dtype=self.dtype,
            eps=1e-6,
        )
        self.norm1.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        self.norm1_shards = self.norm1.shard(self.devices)

        self.norm2 = LayerNorm(
            dims=hidden_size,
            devices=self.devices,
            dtype=self.dtype,
            eps=1e-6,
        )
        self.norm2.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        self.norm2_shards = self.norm2.shard(self.devices)

        # Distributed attention (tensor-parallel)
        head_dim = hidden_size // num_heads
        self.attn = DistributedVisionWindowAttention(
            dtype=dtype,
            hidden_size=hidden_size,
            n_heads=num_heads,
            head_dim=head_dim,
            devices=self.devices,
            flash_attention=True,
        )
        self.attn.sharding_strategy = ShardingStrategy.stacked_qkv(
            len(self.devices), num_heads, head_dim
        )
        self.attn_shards = self.attn.shard(self.devices)

        # MLP tensor-parallel with allreduce
        self.mlp = VisionMLP(
            dtype=dtype,
            devices=self.devices,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(self.devices)
        )
        self.mlp_shards = self.mlp.shard(self.devices)
        self.allreduce = Allreduce(num_accelerators=len(self.devices))

    def __call__(
        self,
        xs: Sequence[TensorValue],
        position_embeddings: list[tuple[TensorValue, TensorValue]],
        input_row_offsets: Sequence[TensorValue],
        max_seqlen: Sequence[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        # Norm 1
        norm1_outs = [
            norm(x) for norm, x in zip(self.norm1_shards, xs, strict=True)
        ]

        # Attention per device (ragged)
        attn_outs = [
            attn(
                norm_out,
                position_embeddings=pos_embs,
                input_row_offsets=row_offsets,
                max_seqlen=mx,
            )
            for attn, norm_out, pos_embs, row_offsets, mx in zip(
                self.attn_shards,
                norm1_outs,
                position_embeddings,
                input_row_offsets,
                max_seqlen,
                strict=True,
            )
        ]
        # Allreduce attention outputs
        attn_outs = self.allreduce(attn_outs, signal_buffers)

        # Residual add
        hs = [x + a for x, a in zip(xs, attn_outs, strict=True)]

        # Norm 2
        norm2_outs = [
            norm(h) for norm, h in zip(self.norm2_shards, hs, strict=True)
        ]

        # MLP per device
        mlp_outs = [
            mlp(norm_out)
            for mlp, norm_out in zip(self.mlp_shards, norm2_outs, strict=True)
        ]
        mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        # Residual add
        outs = [h + m for h, m in zip(hs, mlp_outs, strict=True)]
        return outs


class VisionPatchMerger(Module, Shardable):
    """Group spatially adjacent sets of four patch features then concatenate and
    pass through a two-layer multi-layer perceptron (MLP) to project them into a
    dimension that aligns with the text embeddings used in the LLM.

    Compared to Qwen2.vVL, this layer uses LayerNorm rather than RMSNorm and
    uses a post-shuffle normalization layer.
    """

    def __init__(
        self,
        dtype: DType,
        devices: Sequence[DeviceRef],
        hidden_size: int,
        spatial_merge_size: int,
        out_hidden_size: int,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.dtype = dtype
        self.devices = devices

        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.input_dim = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = LayerNorm(
            dims=self.input_dim
            if self.use_postshuffle_norm
            else self.hidden_size,
            devices=[devices[0]],
            dtype=self.dtype,
            eps=1e-6,
        )

        self.linear_fc1 = Linear(
            in_dim=self.input_dim,
            out_dim=self.input_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=True,
        )

        self.linear_fc2 = Linear(
            in_dim=self.input_dim,
            out_dim=self.out_hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=True,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.linear_fc1.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if strategy.is_replicate:
            # Replicate all weights across devices
            self.norm.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.linear_fc1.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.linear_fc2.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        elif strategy.is_tensor_parallel:
            self.norm.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            # Tensor parallel: first linear rowwise, second linear columnwise
            self.linear_fc1.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.linear_fc2.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
        else:
            raise ValueError(
                "Only replicate or tensor parallel sharding strategies are supported for VisionPatchMerger"
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[VisionPatchMerger]:
        # Shard underlying weights
        norm_shards = self.norm.shard(devices)
        linear_fc1_shards = self.linear_fc1.shard(devices)
        linear_fc2_shards = self.linear_fc2.shard(devices)

        shards: list[VisionPatchMerger] = []
        for idx, device in enumerate(devices):
            sharded = VisionPatchMerger(
                dtype=self.dtype,
                devices=[device],
                hidden_size=self.hidden_size,
                out_hidden_size=self.out_hidden_size,
                spatial_merge_size=self.spatial_merge_size,
                use_postshuffle_norm=self.use_postshuffle_norm,
            )
            # Assign shards
            sharded.norm = norm_shards[idx]
            sharded.linear_fc1 = linear_fc1_shards[idx]
            sharded.linear_fc2 = linear_fc2_shards[idx]
            shards.append(sharded)
        return shards

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies a vision patch merger to the input tensor.
        Args:
            x: TensorValue of shape (n_patches, hidden_size)
        Returns:
            TensorValue of shape (n_patches // (spatial_merge_size^2), out_hidden_size)
        """
        n_patches, hidden_size = x.shape
        factor = self.input_dim // self.hidden_size
        x = x.rebind(((n_patches // factor) * factor, hidden_size))
        if self.use_postshuffle_norm:
            x = x.reshape(((n_patches // factor), self.input_dim))

        # Apply LayerNorm
        x = self.norm(x)

        # Reshape for MLP input
        x = x.reshape(((n_patches // factor), self.input_dim))

        # Apply first linear layer, then GELU, then second linear layer
        x = self.linear_fc1(x)
        x = ops.gelu(x)
        x = self.linear_fc2(x)

        return x


class VisionTransformer(Module):
    """The bare Qwen3VL Vision Transformer (a redesigned Vision Transformer (ViT))
    outputting raw hidden-states without any specific head on top.

    This is difference between this module and the Qwen2.5VL Vision Transformer:
    - Deep stack features
    - pytorch GELU tanh activation
    - LayerNorm for normalization
    - VisionPatchMerger layer which also uses post-shuffle normalization
    - BilinearInterpolationPositionEmbedding for position encoding
    - No window attention
    """

    def __init__(
        self,
        config: VisionConfig,
    ):
        super().__init__()

        self.devices = config.devices
        self.llm_dtype = config.llm_dtype
        self.spatial_merge_unit = (
            config.spatial_merge_size * config.spatial_merge_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        # Create patch embedding layer
        self.patch_embed = VisionPatchEmbed(
            dtype=config.dtype,
            devices=self.devices,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        self.patch_embed.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        self.patch_embed_shards = self.patch_embed.shard(self.devices)

        # Create bilinear interpolation position embedding
        self.pos_embed = BilinearInterpolationPositionEmbedding(
            dtype=config.dtype,
            device=self.devices[0],
            num_position_embeddings=config.num_position_embeddings,
            hidden_size=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )

        # Create rotary position embedding
        self.rotary_pos_emb = VisionRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=10000.0,
        )

        # Create transformer blocks
        self.blocks = LayerList(
            [
                VisionBlock(
                    dtype=config.dtype,
                    devices=self.devices,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.depth)
            ]
        )
        deepstack_merger_list: Sequence[VisionPatchMerger] = [
            VisionPatchMerger(
                dtype=config.dtype,
                devices=self.devices,
                hidden_size=config.hidden_size,
                out_hidden_size=config.out_hidden_size,
                spatial_merge_size=config.spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in range(len(config.deepstack_visual_indexes))
        ]
        # Use tensor parallel for merger: rowwise -> gelu -> columnwise, then allreduce
        for i in range(len(config.deepstack_visual_indexes)):
            deepstack_merger_list[
                i
            ].sharding_strategy = ShardingStrategy.tensor_parallel(
                len(self.devices)
            )

        self.deepstack_merger_shards_list = [
            deepstack_merger_list[i].shard(self.devices)
            for i in range(len(config.deepstack_visual_indexes))
        ]
        self.deepstack_merger_allreduce_list = [
            Allreduce(num_accelerators=len(self.devices))
            for _ in range(len(config.deepstack_visual_indexes))
        ]

        self.deepstack_merger_list = LayerList(deepstack_merger_list)

        # Create patch merger
        self.merger = VisionPatchMerger(
            dtype=config.dtype,
            devices=self.devices,
            hidden_size=config.hidden_size,
            out_hidden_size=config.out_hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            use_postshuffle_norm=False,
        )
        # Use tensor parallel for merger: rowwise -> gelu -> columnwise, then allreduce
        self.merger.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(self.devices)
        )
        self.merger_shards = self.merger.shard(self.devices)
        self.merger_allreduce = Allreduce(num_accelerators=len(self.devices))

    def __call__(
        self,
        pixel_values: Sequence[TensorValue],
        idxs: Sequence[TensorValue],
        weights: Sequence[TensorValue],
        grid_thw: Sequence[TensorValue],
        rot_pos_ids: Sequence[TensorValue],
        max_grid_size: Sequence[TensorValue],
        cu_seqlens: Sequence[TensorValue],
        max_seqlen: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> tuple[list[TensorValue], list[list[TensorValue]]]:
        """Outputs raw hidden states of the transformer model on input `x`.

        1. Patch Embedding: Converts raw input into patches and embeds them.
        2. Rotary Positional Embeddings: Computes rotary positional encodings to the patches.
        3. Windowing: Divides the sequence into windows to perform attention within those windows using the window_index.
        4. Transformer Processing: Processes the sequence through multiple transformer blocks, with attention to cumulative window sequence lengths and positional encodings.
        5. Merging and Sorting: transformer results are merged and sorted to restore the original sequence order before windowing.
        6. The processed hidden_states are returned as the model's output.

        Args:
            pixel_values: Buffer of images of shape (seq_len=n_patches, in_channels * temporal_patch_size * patch_size * patch_size)
                seq_len depends on the spatial dims of the image or video and second dim of x for Qwen2.5VL is 1176.
                Qwen2.5VL processor that handles multiple images of different shapes by flattening all dims and returning
                a 2D tensor of all patches in all images + a grid_thw representing the temporal and spatial coords of patches.
            grid_thw: Buffer of shape (n_images, 3) representing the temporal and spatial dimensions of each image/video.
            rotary_pos_ids: Buffer of shape (seq_len, 2) generated by data_processing.mrope_pos_ids_3d(grid_thw, spatial_merge_size).
            max_grid_size: max value in spatial dimensions in the grid of image and video patches.
                It represents the max no. of patches in an image or a frame. Used as the max positional embedding needed.
            cu_seqlens: Cumulative sequence lengths for full attention blocks.
            max_seqlen: Maximum sequence length for full attention blocks.
            signal_buffers: Communication buffers for distributed execution.

        Returns:
            Sequence[TensorValue] : Image embeddings tensor, one per device, flattened for language model.
            deepstack_feature_lists: List of deepstack feature lists, one per layer.
                deepstack_feature_lists[layer_idx] is a list of tensors (one per device) for that layer.

        Shapes:
            Input: pixel_values shape = (seq_len, in_channels * temporal_patch_size * patch_size * patch_size)
                where seq_len = no. of patches in all images and videos.
            Output: Sequence[TensorValue] each of shape (seq_len // (spatial_merge_size^2), out_hidden_size)
            deepstack_feature_lists: List of lists, where deepstack_feature_lists[layer_idx] contains
                one tensor per device, each of shape (seq_len // (spatial_merge_size^2), out_hidden_size)
        """
        # Pass input images or videos through a conv to obtain patch embeddings.
        hs = [
            embed(pixels)
            for embed, pixels in zip(
                self.patch_embed_shards,
                pixel_values,
                strict=True,
            )
        ]
        seq_len = hs[0].shape[0]

        # compute position embeddings for the grid of image and video patches.
        pos_embeds = self.pos_embed(
            idxs=idxs[0], weights=weights[0], grid_thw=grid_thw[0]
        )
        hs = [h + pos_embeds.to(h.device) for h in hs]

        # Compute rotary positional encodings to input patches.
        rotary_position_embeddings_host = (
            self.rotary_pos_emb.generate_rot_pos_embeddings(
                rot_pos_ids=rot_pos_ids[0],
                max_grid_size=max_grid_size[0],
                seq_len=seq_len,
            )
        )
        rotary_position_embeddings = [
            (
                rotary_position_embeddings_host[0].to(device),
                rotary_position_embeddings_host[1].to(device),
            )
            for device in self.devices
        ]

        deepstack_feature_lists = []
        # Pass patch and positional embeddings though Window Attention Blocks to get hidden states for each patch.
        for layer_num, blk in enumerate(self.blocks):
            hs = blk(
                hs,
                position_embeddings=rotary_position_embeddings,
                input_row_offsets=cu_seqlens,
                max_seqlen=max_seqlen,
                signal_buffers=signal_buffers,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(
                    layer_num
                )
                deepstack_merger_shards = self.deepstack_merger_shards_list[
                    deepstack_merger_idx
                ]
                deepstack_merger_allreduce = (
                    self.deepstack_merger_allreduce_list[deepstack_merger_idx]
                )
                deepstack_merger_outs = [
                    deepstack_merger(h)
                    for h, deepstack_merger in zip(
                        hs, deepstack_merger_shards, strict=True
                    )
                ]
                deepstack_feature = deepstack_merger_allreduce(
                    deepstack_merger_outs, signal_buffers
                )

                # Append the list of features (one tensor per device) for this layer.
                # Each device's feature tensor shape = (seq_len // (spatial_merge_size^2), out_hidden_size) = (seq_len//4, 2048).
                deepstack_feature_lists.append(deepstack_feature)

        # The merged features are projected via a linear layer to align with the language model's embedding space.
        # Apply per-device merger, then concatenate back in original order
        # h shape = (seq_len, hidden_size) = (seq_len, 1152).
        merger_outs = [
            merger(h) for merger, h in zip(self.merger_shards, hs, strict=True)
        ]
        merger_outs = self.merger_allreduce(
            merger_outs, signal_buffers
        )  # (seq_len // (spatial_merge_size^2), out_hidden_size) = (seq_len//4, 2048).

        return merger_outs, deepstack_feature_lists
