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
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from max.dtype import DType
from max.graph import Dim, TensorValue, TensorValueLike, dtype_promotion, ops
from max.nn import MLP, Conv3D, Linear, RMSNorm
from max.nn.layer import Module


@dataclass
class VisionPatchEmbed(Module):
    proj: Conv3D
    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    embed_dim: int = 1152
    spatial_merge_unit: int = 4

    def __post_init__(self):
        super().__init__()

    def __call__(
        self, x: TensorValueLike, window_index: TensorValueLike
    ) -> TensorValue:
        """Generates patch embeddings from pixel_values of patches (`x`) and reorders them by window_index.

        Reshapes input to (batch_size, in_channels, depth, height, width).
        Permutes it to be compatible with max.pipelines.nn.Conv3D input tensor.
        Permutes the output then flattens it.

        Args:
            x: tensor representing pixel values of shape [resized_height, resized_width].

        Returns:
            a tensor of size (seq_len, hidden_size = embed_dim)
        """
        x, filter = dtype_promotion._promote_weak_dtypes(x, self.proj.filter)
        x = x.cast(filter.dtype)
        x = x.reshape(
            (
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            )
        )
        # Permute (batch_size, in_channels, depth, height, width) inputs to (batch_size, depth, height, width, in_channels) for our Graph API.
        x = x.permute([0, 2, 3, 4, 1])
        x = self.proj(x)
        # Permute max output from (batch_size, depth, height, width, out_channels) to (batch_size, out_channels, depth, height, width)
        x = x.permute([0, 2, 3, 4, 1])
        x = x.reshape((-1, self.embed_dim))

        seq_len = x.shape[0]
        # Reshape into a 3D tensor of blocks.
        h = x.reshape(
            [seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1]
        )
        # Reorders patch_embeddings according to window_index indices.
        h = ops.gather(h, window_index, axis=0).reshape([seq_len, -1])
        return h


@dataclass
class VisionRotaryEmbedding(Module):
    """Rotary embedding layer for the qwen2 vision model.

    Differences compared to `max.nn.RotaryEmbedding`:

    - In _compute_inv_freqs, the head dimension (n) is divided by 2.
    - inv_freqs is cached rather than freqs. generate_rot_pos_embeddings takes seq_len as input because it depends on the image and video inputs
        rather than using a fixed value for the model.
        TODO: We might change this depending on how our infra handles seq_length now.
    """

    dim: int
    n_heads: int
    theta: float
    """The maximum sequence length for model's input."""
    _inv_freqs: Optional[TensorValueLike] = None

    def __post_init__(self):
        super().__init__()

    def _compute_inv_freqs(self) -> TensorValue:
        if self._inv_freqs is None:
            n = (self.dim // self.n_heads) // 2
            # Note: using float64 to avoid an overflow on the exponential, then converting back to float32.
            iota = ops.range(
                ops.constant(0, DType.float64),
                ops.constant(n - 1, DType.float64),
                ops.constant(2, DType.float64),
                out_dim=n // 2,
            )
            inv_freq = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
            self._inv_freqs = inv_freq
        return TensorValue(self._inv_freqs)

    @cached_property
    def inv_freqs(self) -> TensorValue:
        self._inv_freqs = self._compute_inv_freqs()
        return self._inv_freqs

    def generate_rot_pos_embeddings(
        self,
        rot_pos_ids: TensorValue,
        window_index: TensorValue,
        spatial_merge_unit: int,
        max_grid_size: int,
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
            ops.constant(0, DType.float64),
            ops.constant(max_grid_size, DType.float64),
            ops.constant(1, DType.float64),
            out_dim=max_grid_size,
        )
        rotary_pos_emb_full = ops.outer(t, self.inv_freqs)
        # Retrieve position embeddings for each patch in input images or videos.
        rotary_pos_emb = ops.gather(rotary_pos_emb_full, rot_pos_ids, axis=0)
        rotary_pos_emb = rotary_pos_emb.flatten(1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            [seq_len // spatial_merge_unit, spatial_merge_unit, -1]
        )

        # Reorders patches' rot position embeddings according to window_index indices.
        rotary_pos_emb = ops.gather(
            rotary_pos_emb, window_index, axis=0
        ).reshape([seq_len, -1])
        # Generates a cos and a sin of rotary position embeddings which will be applied later. Shape = (seq_len, 2 * hidden_size).
        rotary_pos_emb = ops.concat((rotary_pos_emb, rotary_pos_emb), -1)

        freqs_cis = (ops.cos(rotary_pos_emb), ops.sin(rotary_pos_emb))
        return freqs_cis

    def __call__(
        self,
        x: TensorValue,
    ) -> TensorValue:
        raise NotImplementedError


@dataclass
class VisionWindowSdpaAttention(Module):
    dim: int
    n_heads: int
    qkv: Linear
    proj: Linear

    @staticmethod
    def apply_rotary_pos_emb_vision(
        q: TensorValue, k: TensorValue, cos: TensorValue, sin: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        def _rotate_half(x: TensorValue) -> TensorValue:
            """Rotates half the hidden dims of the input."""
            head_dim = x.shape[-1]
            head_dim_val = TensorValue(head_dim)
            half_dim = head_dim // 2
            half_dim_val = TensorValue(half_dim)
            slice_re = (slice(0, half_dim_val), half_dim)
            slice_im = (slice(half_dim_val, head_dim_val), half_dim)
            x_re = x[..., slice_re]
            x_im = x[..., slice_im]
            return ops.concat((x_re, x_im), -1)

        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        cos, sin = (
            ops.cast(ops.unsqueeze(cos, -2), orig_q_dtype),
            ops.cast(ops.unsqueeze(sin, -2), orig_q_dtype),
        )
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        q_embed = ops.cast(q_embed, orig_q_dtype)
        k_embed = ops.cast(k_embed, orig_k_dtype)
        return q_embed, k_embed

    @staticmethod
    def scaled_dot_product_attention(
        xq: TensorValue,
        xk: TensorValue,
        xv: TensorValue,
        attention_mask: TensorValue,
        dim: int,
        n_heads: int,
    ):
        """Computes scaled dot product attention on query, key and value tensors, using an attention mask.
        Shape of xq, xk, and xv = (n_heads, seq_len, head_dim) = (16, 14308, 80).
        """
        head_dim = (dim // n_heads) // 2
        scale = math.sqrt(1.0 / head_dim)
        scores = xq @ ops.transpose(xk, -2, -1)
        # Note, the graph compiler currently requires the order of operands
        # to be `scores * scale` in order to pattern match the fused attention
        # operator.
        scores = ops.softmax((scores * scale) + attention_mask)
        return scores @ xv

    def __call__(
        self,
        x: TensorValue,
        position_embeddings: tuple[TensorValue, TensorValue],
        attention_mask: TensorValue,
    ) -> TensorValue:
        """Naive Sliding Window Vision Attention Layer for Qwen2.5vVL. It does the following steps:
            1. Linear Projections Q, K, V
            2. Apply Rotary position embeddings on the Linear Projections Q, and K
            3. Scaled dot product attention
            4. Final Linear projection layer

        Args:
            x:
            position_embeddings:
            attention_mask:

        Returns:
            The output of applying sliding window attention on input `x` using `attention_mask`.
            It applies rotary position embeddings `position_embeddings` in the process.

        Shapes:
            Input:
                x: (seq_len, hidden_size)
                position_embeddings: tuple of 2 tensors of shape ()
                attention_mask: (1, seq_len, seq_len)
            Output: tuple of:
                - indices: (batch_size * seq_length, num_per_tok)
                - weights: (batch_size * seq_length, num_per_tok)
        """
        seq_length = x.shape[0]
        qkv = (
            self.qkv(x)
            .reshape([seq_length, 3, self.n_heads, -1])
            .permute([1, 0, 2, 3])
        )
        # Split qkv into a tuple of tensors along the first dim: q, k, v = qkv.unbind(0)
        xq, xk, xv = qkv[0], qkv[1], qkv[2]
        cos, sin = position_embeddings
        xq, xk = VisionWindowSdpaAttention.apply_rotary_pos_emb_vision(
            xq, xk, cos, sin
        )
        xq = xq.transpose(0, 1)
        xk = xk.transpose(0, 1)
        xv = xv.transpose(0, 1)
        attn_output = VisionWindowSdpaAttention.scaled_dot_product_attention(
            xq, xk, xv, attention_mask, self.dim, self.n_heads
        )
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape((seq_length, -1))
        attn_output = self.proj(attn_output)
        return attn_output


@dataclass
class VisionBlock(Module):
    norm1: RMSNorm
    norm2: RMSNorm
    attn: VisionWindowSdpaAttention
    mlp: MLP

    def __post_init__(self):
        super().__init__()

    def __call__(
        self,
        x: TensorValue,
        position_embeddings: tuple[TensorValue, TensorValue],
        attention_mask: TensorValue,
    ) -> TensorValue:
        h = x + self.attn(
            self.norm1(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        h = h + self.mlp(self.norm2(h))
        return h


@dataclass
class VisionTransformer(Module):
    """The bare Qwen2.5VL Vision Transformer (a redesigned Vision Transformer (ViT))
    outputting raw hidden-states without any specific head on top.

    Its a native dynamic-resolution Vision Transformer that incorporates Window Attention.
    It reduces computational overhead while maintaining native resolution. It incorporates
    2D-RoPE and window attention to support native input resolutions while accelerating
    the computation of the entire visual encoder.

    The height and width of the input images are resized to multiples of 28 before being fed into the ViT.

    This module processes images by splitting them into patches with a stride of 14,
    generating a set of image features (embeddings).

    Then, it groups spatially adjacent sets of four patch features, then concatenate
    and pass through a two-layer (MLP) instead of directly using the raw patch embeds.
    The output of ViT is passed through a spatial merging operation that reduces the
    spatial dimensions of the output, controlled by the spatial_merge_size parameter.
    After that, A linear layer projects the merged features into a space compatible with
    the language model's embeddings.

    Qwen2.5 ViT efficiently process visual information and seamlessly integrate it with
    language models, enabling a wide range of multimodal tasks.
    """

    patch_embed: VisionPatchEmbed
    rotary_pos_emb: VisionRotaryEmbedding
    blocks: list[VisionBlock]
    fullatt_block_indexes: list[int]
    spatial_merge_unit: int

    def __post_init__(self):
        super().__init__()

    def __call__(
        self,
        pixel_values: TensorValue,
        rot_pos_ids: TensorValue,
        window_index: TensorValue,
        attention_mask_window: TensorValue,
        attention_mask_full: TensorValue,
        max_grid_size: int,
    ) -> TensorValue:
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
            attention_mask_window: a tensor of shape [1, seq_len, seq_len] that restricts patches from interacting
                outside their valid segments for Window Attention mechanism (same sequence and same window).
            attention_mask_full: a tensor of shape [1, seq_len, seq_len] that restricts patches from interacting outside
                their valid segments for  Self Attention mechanism.
            max_grid_size: max value in spatial dimensions in the grid of image and video patches.
                It represents the max no. of patches in an image or a frame. Used as the max positional embedding needed.

        Returns:
            TensorValue : output of vision transformer projected into the decoder's hidden_size.

        Shapes:
            Input: pixel_values shape = (seq_len, in_channels * temporal_patch_size * patch_size * patch_size)
                where seq_len = no. of patches in all images and videos.
            Output:
        """
        # Pass input images or videos through a conv to obtain patch embeddings ordered by window_index.
        h = self.patch_embed(
            pixel_values, window_index
        )  # Shape = [seq_len, hidden_size]
        seq_len = h.shape[0]

        # Compute rotary positional encodings to input patches ordered by window_index.
        position_embeddings = self.rotary_pos_emb.generate_rot_pos_embeddings(
            rot_pos_ids,
            window_index,
            self.spatial_merge_unit,
            max_grid_size,
            seq_len,
        )

        # Pass patch and positional embeddings though Window Attention Blocks to get hidden states for each patch.
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attention_mask = attention_mask_full
            else:
                attention_mask = attention_mask_window
            h = blk(
                h,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        return h
