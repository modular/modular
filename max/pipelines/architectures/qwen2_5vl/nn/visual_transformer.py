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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    dtype_promotion,
    ops,
)
from max.nn import (
    MLP,
    Allreduce,
    LayerList,
    Linear,
    RMSNorm,
    Shardable,
)
from max.nn.layer import Module

from ..model_config import VisionConfig
from .vision_attention import DistributedVisionWindowAttention


class VisionPatchEmbed(Module, Shardable):
    """Generates patch embeddings from a tensor of pixel_values of patches using a Linear layer.

    This implementation uses a Linear layer instead of Conv3D, which is mathematically
    equivalent when stride equals kernel size (non-overlapping patches).
    """

    def __init__(
        self,
        dtype: DType,
        devices: Sequence[DeviceRef],
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
        spatial_merge_unit: int = 4,
    ):
        super().__init__()
        self.devices = devices or [DeviceRef.CPU()]
        self._sharding_strategy: ShardingStrategy | None = None

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.spatial_merge_unit = spatial_merge_unit

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
            has_bias=False,
        )

    def __call__(
        self,
        x: TensorValueLike,
        window_index: TensorValueLike,
    ) -> TensorValue:
        """Generates patch embeddings from pixel_values of patches (`x`) and reorders them by window_index.

        Uses Linear layer instead of Conv3D for patch embedding, which is mathematically
        equivalent when stride equals kernel size (non-overlapping patches).

        Args:
            x: tensor representing pixel values of shape [n_patches, patch_dim].
            window_index: tensor for reordering patch embeddings.

        Returns:
            a tensor of size (seq_len, hidden_size = embed_dim)
        """
        x, weight = dtype_promotion._promote_weak_dtypes(x, self.proj.weight)

        x = x.cast(weight.dtype)

        # Shape: (batch_size, patch_dim)
        assert x.shape[1] == self.patch_dim, (
            f"x.shape should be (n_patches, patch_dim) = {x.shape}, self.patch_dim = {self.patch_dim}"
        )

        # Apply linear transformation
        h = self.proj(x)

        seq_len = h.shape[0]
        # Reshape into a 3D tensor of blocks.
        h = h.rebind(
            [
                (seq_len // self.spatial_merge_unit) * self.spatial_merge_unit,
                self.embed_dim,
            ]
        )
        h = h.reshape(
            [seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1]
        )

        # Reorders patch_embeddings according to window_index indices.
        h = (
            ops.gather(h, window_index, axis=0)
            .reshape([-1, self.embed_dim])
            .rebind([seq_len, self.embed_dim])
        )
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
                spatial_merge_unit=self.spatial_merge_unit,
            )
            sharded.proj = proj_shards[shard_idx]
            shards.append(sharded)
        return shards


@dataclass
class VisionRotaryEmbedding(Module):
    """Rotary embedding layer for the qwen2 vision model.

    Differences compared to `max.nn.RotaryEmbedding`:

    - In _compute_inv_freqs, the head dimension (n) is divided by 2.
    - inv_freqs is cached instead of freqs.
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
        # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
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
        """Compute and cache inverse frequencies for the given device.

        Truly cached - computes once and returns the same TensorValue object.
        """
        if self._inv_freqs is None:
            self._inv_freqs = self._compute_inv_freqs(device)
        return self._inv_freqs

    def generate_rot_pos_embeddings(
        self,
        rot_pos_ids: TensorValue,
        window_index: TensorValue,
        spatial_merge_unit: int,
        max_grid_size: TensorValue,
        seq_len: Dim,
    ) -> tuple[TensorValue, TensorValue]:
        """Generates rotary position embeddings for a maximum sequence length of max_grid_size
        reordered by window_index. window_index is the indices of patches in the order they are
        fed to WindowAttention.

        Args:
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
            dtype=DType.int32,
        ).cast(DType.float32)
        rotary_pos_emb_full = ops.outer(t, self.inv_freqs(rot_pos_ids.device))
        # Retrieve position embeddings for each patch in input images or videos.
        rotary_pos_emb = ops.gather(rotary_pos_emb_full, rot_pos_ids, axis=0)
        rotary_pos_emb = rotary_pos_emb.flatten(1)

        rotary_pos_emb = rotary_pos_emb.rebind(
            [
                (seq_len // spatial_merge_unit) * spatial_merge_unit,
                rotary_pos_emb.shape[-1],
            ]
        )
        rotary_pos_emb = rotary_pos_emb.reshape(
            [seq_len // spatial_merge_unit, spatial_merge_unit, -1]
        )

        # Reorders patches' rot position embeddings according to window_index indices.
        rotary_pos_emb = (
            ops.gather(rotary_pos_emb, window_index, axis=0)
            .reshape([-1, rotary_pos_emb.shape[-1]])
            .rebind([seq_len, rotary_pos_emb.shape[-1]])
        )
        # Generates a cos and a sin of rotary position embeddings which will be applied later. Shape = (seq_len, 2 * hidden_size).
        rotary_pos_emb = ops.concat((rotary_pos_emb, rotary_pos_emb), -1)

        freqs_cis = (ops.cos(rotary_pos_emb), ops.sin(rotary_pos_emb))
        return freqs_cis

    def __call__(
        self,
        x: TensorValue,
    ) -> TensorValue:
        raise NotImplementedError


class VisionBlock(Module):
    """Vision transformer block with distributed attention and MLP."""

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
        self.devices = devices
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Norms replicated across devices
        self.norm1 = RMSNorm(
            dim=hidden_size,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )
        self.norm1.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        self.norm1_shards = self.norm1.shard(self.devices)

        self.norm2 = RMSNorm(
            dim=hidden_size,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
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
        self.mlp = MLP(
            dtype=dtype,
            quantization_encoding=None,
            hidden_dim=hidden_size,
            feed_forward_length=intermediate_size,
            devices=self.devices,
            has_bias=True,
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
        norm1_outs = [norm(x) for norm, x in zip(self.norm1_shards, xs)]

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
            )
        ]
        # Allreduce attention outputs
        attn_outs = self.allreduce(attn_outs, signal_buffers)

        # Residual add
        hs = [x + a for x, a in zip(xs, attn_outs)]

        # Norm 2
        norm2_outs = [norm(h) for norm, h in zip(self.norm2_shards, hs)]

        # MLP per device
        mlp_outs = [
            mlp(norm_out) for mlp, norm_out in zip(self.mlp_shards, norm2_outs)
        ]
        mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        # Residual add
        outs = [h + m for h, m in zip(hs, mlp_outs)]
        return outs


class PatchMerger(Module, Shardable):
    """Group spatially adjacent sets of four patch features then concatenate and
    pass through a two-layer multi-layer perceptron (MLP) to project them into a
    dimension that aligns with the text embeddings used in the LLM.
    """

    def __init__(
        self,
        dtype: DType,
        devices: Sequence[DeviceRef],
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
    ):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.input_dim = hidden_size * (spatial_merge_size**2)
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.devices = devices

        # Create RMSNorm layer
        self.norm = RMSNorm(
            dim=hidden_size, dtype=dtype, eps=1e-6, multiply_before_cast=False
        )

        # Create individual MLP layers
        self.linear1 = Linear(
            in_dim=self.input_dim,
            out_dim=self.input_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=True,
        )

        self.linear2 = Linear(
            in_dim=self.input_dim,
            out_dim=out_hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=True,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.linear1.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if strategy.is_replicate:
            # Replicate all weights across devices
            self.norm.weight.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.linear1.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.linear2.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        else:
            # Tensor parallel: first linear rowwise, second linear columnwise
            self.norm.weight.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.linear1.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.linear2.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[PatchMerger]:
        # Shard underlying weights
        norm_weight_shards = self.norm.weight.shard(devices)
        linear1_shards = self.linear1.shard(devices)
        linear2_shards = self.linear2.shard(devices)

        shards: list[PatchMerger] = []
        for idx, device in enumerate(devices):
            sharded = PatchMerger(
                dtype=self.dtype,
                devices=[device],
                hidden_size=self.hidden_size,
                out_hidden_size=self.out_hidden_size,
                spatial_merge_size=self.spatial_merge_size,
            )
            # Assign shards
            sharded.norm.weight = norm_weight_shards[idx]
            sharded.linear1 = linear1_shards[idx]
            sharded.linear2 = linear2_shards[idx]
            shards.append(sharded)
        return shards

    def __call__(
        self, x: TensorValue, signal_buffers: Sequence[BufferValue]
    ) -> TensorValue:
        # Apply RMSNorm and reshape for MLP input
        x = self.norm(x)
        x = x.rebind(
            [
                (x.shape[0] // self.spatial_merge_unit)
                * self.spatial_merge_unit,
                x.shape[-1],
            ]
        )
        x = x.reshape((-1, self.input_dim))

        # Apply first linear layer, then GELU, then second linear layer
        x = self.linear1(x)
        x = ops.gelu(x)
        x = self.linear2(x)

        return x


class VisionTransformer(Module):
    """The bare Qwen2.5VL Vision Transformer (a redesigned Vision Transformer (ViT))
    outputting raw hidden-states without any specific head on top.

    Its incorporates Window Attention to address the quadratic computational complexity
    associated with processing images of varying sizes at native resolution. This module
    uses only 4 full attention layers and the rest are windowed to reduces computational
    overhead while maintaining native resolution. Window Attention cost scales
    linearly with the number of patches rather than quadratically.

    For positional encoding, it adopts 2D Rotary Positional Embedding (RoPE). For videos,
    MRoPE aligns time IDs with absolute time along the temporal dimension to capture the
    pace of events and precise moment localization. Two consecutive frames are grouped
    together, significantly reducing the number of tokens.

    This is difference between this module and the original ViT proposed with Llava is:
    - FFN
    - SwiGLU activation
    - RMSNorm for normalization
    - window-based attention

    This module consumes processed images that are split into patches with a stride of 14,
    generating a set of image features (embeddings), one for each patch.

    To address the efficiency challenges posed by long sequences of image features,
    it groups spatially adjacent sets of four patch features, then concatenate them
    and pass through a two-layer (MLP) that projects the merged features into a space
    compatible with the language model's embeddings. This is instead of directly using
    the raw patch embeds.
    This spatial merging operation reduces the spatial dimensions of the output,
    and is controlled by the spatial_merge_size parameter.
    """

    def __init__(
        self,
        config: VisionConfig,
    ):
        super().__init__()

        self.devices = config.devices
        self.spatial_merge_unit = (
            config.spatial_merge_size * config.spatial_merge_size
        )
        self.fullatt_block_indexes = config.fullatt_block_indexes

        # Create patch embedding layer
        self.patch_embed = VisionPatchEmbed(
            dtype=config.dtype,
            devices=self.devices,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            spatial_merge_unit=self.spatial_merge_unit,
        )
        self.patch_embed.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        self.patch_embed_shards = self.patch_embed.shard(self.devices)

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

        # Create patch merger
        self.merger = PatchMerger(
            dtype=config.dtype,
            devices=self.devices,
            hidden_size=config.hidden_size,
            out_hidden_size=config.out_hidden_size,
            spatial_merge_size=config.spatial_merge_size,
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
        rot_pos_ids: Sequence[TensorValue],
        window_index: Sequence[TensorValue],
        cu_seqlens: Sequence[TensorValue],
        cu_window_seqlens: Sequence[TensorValue],
        max_seqlen: Sequence[TensorValue],
        max_window_seqlen: Sequence[TensorValue],
        max_grid_size: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> Sequence[TensorValue]:
        """Outputs raw hidden states of the transformer model on input `x`.

        1. Patch Embedding: Converts raw input into patches and embeds them.
        2. Rotary Positional Embeddings: Computes rotary positional encodings to the patches.
        3. Windowing: Divides the sequence into windows to perform attention within those windows using the window_index.
        4. Transformer Processing: Processes the sequence through multiple transformer blocks, with attention to cumulative window sequence lengths and positional encodings.
        5. Merging and Sorting: transformer results are merged and sorted to restore the original sequence order before windowing.
        6. The processed hidden_states are returned as the model's output.

        Args:
            pixel_values: Tensor of images of shape (seq_len=n_patches, in_channels * temporal_patch_size * patch_size * patch_size)
                seq_len depends on the spatial dims of the image or video and second dim of x for Qwen2.5VL is 1176.
                Qwen2.5VL processor that handles multiple images of different shapes by flattening all dims and returning
                a 2D tensor of all patches in all images + a grid_thw representing the temporal and spatial coords of patches.
            rotary_pos_ids: Tensor of shape (seq_len, 2) generated by data_processing.mrope_pos_ids_3d(grid_thw, spatial_merge_size).
            window_index:  1D TensorValue generated by data_processing.get_window_index(grid_thw, window_size, spatial_merge_size, patch_size, spatial_merge_unit).
            max_grid_size: max value in spatial dimensions in the grid of image and video patches.
                It represents the max no. of patches in an image or a frame. Used as the max positional embedding needed.
            cu_seqlens: Cumulative sequence lengths for full attention blocks.
            cu_window_seqlens: Cumulative window sequence lengths for window attention blocks.
            max_seqlen: Maximum sequence length for full attention blocks.
            max_window_seqlen: Maximum sequence length for window attention blocks.
            signal_buffers: Communication buffers for distributed execution.

        Returns:
            Sequence[TensorValue] : Image embeddings tensor, one per device, flattened for language model.

        Shapes:
            Input: pixel_values shape = (seq_len, in_channels * temporal_patch_size * patch_size * patch_size)
                where seq_len = no. of patches in all images and videos.
            Output: Sequence[TensorValue] each of shape (seq_len, hidden_size)
        """
        # Pass input images or videos through a conv to obtain patch embeddings ordered by window_index.
        hs = [
            embed(pixels, window_idx)
            for embed, pixels, window_idx in zip(
                self.patch_embed_shards, pixel_values, window_index
            )
        ]
        seq_len = hs[0].shape[0]

        # Compute rotary positional encodings to input patches ordered by window_index.
        position_embeddings_host = (
            self.rotary_pos_emb.generate_rot_pos_embeddings(
                rot_pos_ids[0],
                window_index[0],
                self.spatial_merge_unit,
                max_grid_size[0],
                seq_len,
            )
        )
        position_embeddings = [
            (
                position_embeddings_host[0].to(device),
                position_embeddings_host[1].to(device),
            )
            for device in self.devices
        ]

        # Pass patch and positional embeddings though Window Attention Blocks to get hidden states for each patch.
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                input_row_offsets = cu_seqlens
                seqlen = max_seqlen
            else:
                input_row_offsets = cu_window_seqlens
                seqlen = max_window_seqlen
            hs = blk(
                hs,
                position_embeddings=position_embeddings,
                input_row_offsets=input_row_offsets,
                max_seqlen=seqlen,
                signal_buffers=signal_buffers,
            )

        # The merged features are projected via a linear layer to align with the language model's embedding space.
        # Apply per-device merger, then concatenate back in original order
        merged = [
            merger(h, signal_buffers=signal_buffers)
            for merger, h in zip(self.merger_shards, hs)
        ]
        merged = self.merger_allreduce(merged, signal_buffers)

        # Re-order path embeddings (hidden_states) back to its original order before windowing.
        # TODO(GEX-1863): Implement ops.argsort
        outputs: list[TensorValue] = []
        for i in range(len(self.devices)):
            reverse_indices = ops.argsort(window_index[i])
            out_i = ops.gather(merged[i], reverse_indices, axis=0)
            outputs.append(out_i.to(self.devices[i]))
        return outputs
