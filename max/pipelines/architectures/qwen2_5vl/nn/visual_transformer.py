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

import math
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
    LayerList,
    Linear,
    RMSNorm,
    Shardable,
)
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.layer import Module

from ..model_config import VisionConfig


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


class VisionWindowAttention(Module):
    """Naive Sliding Window Vision Attention Layer for Qwen2.5vVL."""

    def __init__(
        self,
        dtype: DType,
        device: DeviceRef,
        dim: int,
        n_heads: int,
        flash_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Add explicit scaling factor like PyTorch implementation
        self.scaling = math.sqrt(1.0 / self.head_dim)
        self.flash_attention = flash_attention

        # Create Linear layers using constructor pattern
        self.qkv = Linear(
            in_dim=dim,
            out_dim=dim * 3,  # Q, K, V projections
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        self.proj = Linear(
            in_dim=dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    @staticmethod
    def apply_rotary_pos_emb_vision(
        q: TensorValue, k: TensorValue, cos: TensorValue, sin: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        def _rotate_half(x: TensorValue) -> TensorValue:
            """Rotates half the hidden dims of the input."""
            head_dim = x.shape[-1]
            half_dim = head_dim // 2
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            return ops.concat((-x2, x1), -1)

        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q, k = ops.cast(q, DType.float32), ops.cast(k, DType.float32)
        cos, sin = (
            ops.cast(ops.unsqueeze(cos, -2), DType.float32),
            ops.cast(ops.unsqueeze(sin, -2), DType.float32),
        )
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        q_embed = ops.cast(q_embed, orig_q_dtype)
        k_embed = ops.cast(k_embed, orig_k_dtype)
        return q_embed, k_embed

    def __call__(
        self,
        x: TensorValue,
        position_embeddings: tuple[TensorValue, TensorValue],
        input_row_offsets: TensorValue,
        max_seqlen: TensorValue,
    ) -> TensorValue:
        """Naive Sliding Window Vision Attention Layer for Qwen2.5vVL. It does the following steps:
            1. Linear Projections Q, K, V
            2. Apply Rotary position embeddings on the Linear Projections Q, and K
            3. Scaled dot product attention
            4. Final Linear projection layer

        Args:
            x: Input tensor of shape (seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) tensors for rotary embeddings
            input_row_offsets: Tensor of shape [window_size + 1] with dtype uint32.
                Indicates where each window starts and ends in the ragged tensors.
                The values should be a prefix sum (cumulative sum) of window lengths.
            max_seqlen: Maximum window length for flash attention.

        Returns:
            The output of applying sliding window attention on input `x` using `attention_mask`.
            It applies rotary position embeddings `position_embeddings` in the process.

        Shapes:
            Input:
                x: (seq_len, hidden_size)
                position_embeddings: tuple of 2 tensors of shape (seq_len, head_dim)
                input_row_offsets: (window_size + 1)
                max_seqlen: (1)
            Output:
                - tensor: (seq_len, hidden_size)
        """
        seq_length = x.shape[0]
        qkv = (
            self.qkv(x)
            .reshape([seq_length, 3, self.n_heads, -1])
            .permute([1, 0, 2, 3])
        )
        # Split qkv into a tuple of tensors along the first dim: q, k, v. Equivalent to qkv.unbind(0)
        xq, xk, xv = qkv[0], qkv[1], qkv[2]
        cos, sin = position_embeddings
        xq, xk = VisionWindowAttention.apply_rotary_pos_emb_vision(
            xq, xk, cos, sin
        )

        attn_output = flash_attention_ragged_gpu(
            xq,
            xk,
            xv,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scaling,
        )
        attn_output = attn_output.reshape((seq_length, -1))
        attn_output = self.proj(attn_output)
        return attn_output


class VisionBlock(Module):
    """Vision transformer block with attention and MLP."""

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

        # Create RMSNorm layers
        self.norm1 = RMSNorm(
            dim=hidden_size,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )

        self.norm2 = RMSNorm(
            dim=hidden_size,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )

        # Create attention layer
        self.attn = VisionWindowAttention(
            dtype=dtype,
            device=devices[0],
            dim=hidden_size,
            n_heads=num_heads,
        )

        # Create MLP layer with multi-GPU support
        self.mlp = MLP(
            dtype=dtype,
            quantization_encoding=None,
            hidden_dim=hidden_size,
            feed_forward_length=intermediate_size,
            devices=[self.devices[0]],
            has_bias=True,
        )

    def __call__(
        self,
        x: TensorValue,
        position_embeddings: tuple[TensorValue, TensorValue],
        input_row_offsets: TensorValue,
        max_seqlen: TensorValue,
        signal_buffers: list[BufferValue],
    ) -> TensorValue:
        h = x + self.attn(
            self.norm1(x),
            position_embeddings=position_embeddings,
            input_row_offsets=input_row_offsets,
            max_seqlen=max_seqlen,
        )
        h = h + self.mlp(self.norm2(h))
        return h


class PatchMerger(Module):
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
        position_embeddings = self.rotary_pos_emb.generate_rot_pos_embeddings(
            rot_pos_ids[0],
            window_index[0],
            self.spatial_merge_unit,
            max_grid_size[0],
            seq_len,
        )

        h = hs[0]

        # Pass patch and positional embeddings though Window Attention Blocks to get hidden states for each patch.
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                input_row_offsets = cu_seqlens[0]
                seqlen = max_seqlen[0]
            else:
                input_row_offsets = cu_window_seqlens[0]
                seqlen = max_window_seqlen[0]
            h = blk(
                h,
                position_embeddings=position_embeddings,
                input_row_offsets=input_row_offsets,
                max_seqlen=seqlen,
                signal_buffers=signal_buffers,
            )

        # The merged features are projected via a linear layer to align with the language model's embedding space.
        h = self.merger(h, signal_buffers=signal_buffers)

        # Re-order path embeddings (hidden_states) back to its original order before windowing.
        # TODO(GEX-1863): Implement ops.argsort
        reverse_indices = ops.argsort(window_index[0])
        h = ops.gather(h, reverse_indices, axis=0)
        return [h.to(dev) for dev in self.devices]
