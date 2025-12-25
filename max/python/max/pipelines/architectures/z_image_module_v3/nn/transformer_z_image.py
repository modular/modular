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
from collections.abc import Sequence
from functools import cached_property
from typing import Tuple

import max.experimental.functional as F
import max.nn.module_v3 as nn
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.nn.module_v3.sequential import ModuleList
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.attention.mask_config import MHAMaskVariant

flash_attention_gpu = F.functional(_flash_attention_gpu)
from .layers import (
    LayerNorm,
    ModuleDict,
    RMSNorm,
    SiLU,
    Transformer2DModelOutput,
    create_attention_mask,
    masked_scatter,
    pad_sequence,
)

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        out_size: int,
        mid_size: int | None = None,
        frequency_embedding_size: int = 256,
    ):
        if mid_size is None:
            mid_size = out_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size),
            SiLU(),
            nn.Linear(mid_size, out_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: Tensor, dim: int, max_period: int = 10000
    ) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = F.exp(
            -math.log(max_period)
            * F.range(0, half, 1, dtype=DType.float32, device=t.device)
            / half
        )
        args = F.unsqueeze(t, -1).cast(DType.float32) * F.unsqueeze(freqs, 0)
        embedding = F.concat([F.cos(args), F.sin(args)], axis=-1)

        if dim % 2:
            zeros = Tensor.zeros(
                [embedding.shape[0], 1],
                dtype=embedding.dtype,
                device=embedding.device,
            )
            embedding = F.concat([embedding, zeros], axis=-1)

        return embedding

    def __call__(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # t_freq = t_freq.cast(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ZImageSingleStreamAttention(nn.Module):
    """
    Z-Image specific attention module.
    Stateless implementation using flash_attention_ragged_gpu.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-5,
    ):
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.kv_inner_dim = self.inner_dim
        self.head_dim = dim // heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

        self.norm_q = RMSNorm(dim // heads, eps)
        self.norm_k = RMSNorm(dim // heads, eps)

    def __call__(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        freqs_cis: Tensor | None = None,
    ) -> Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Z-Image always uses rank-3 inputs: (B, S, C)
        # Get shapes using symbolic dimensions (graph-safe)
        B = hidden_states.shape[0]
        S = hidden_states.shape[1]

        query = query.reshape((B, S, self.heads, self.head_dim))
        key = key.reshape((B, S, self.heads, self.head_dim))
        value = value.reshape((B, S, self.heads, self.head_dim))


        # Apply norms
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE
        # Note: This implementation avoids list(tensor.shape) which causes sync issues
        def apply_rotary_emb(x_in: Tensor, freqs_cis: Tensor, b: int, s: int, h: int, d: int) -> Tensor:
            # x_in shape: (B, S, H, D)
            # Reshape for complex interpretation: (B, S, H, D) -> (B, S, H, D/2, 2)
            half_d = d // 2
            x_complex = x_in.cast(DType.float32).reshape((b, s, h, half_d, 2))

            # Expand freqs_cis at dim 2 (heads) to broadcast: (B, S, 1, D/2, 2)
            freqs_cis_expanded = F.unsqueeze(freqs_cis, 2)

            # Apply complex multiplication
            x_rotated = F.complex_mul(x_complex, freqs_cis_expanded)

            # Reshape back to (B, S, H, D)
            x_out = x_rotated.reshape((b, s, h, d))
            return x_out.cast(x_in.dtype)


        if freqs_cis is not None:
            # For graph compilation, use fixed batch_size=1 and known heads/head_dim
            # S (sequence length) comes from the reshaped query which has shape (B, S, H, D)
            # We use symbolic S from reshape, but H and D are known Python ints
            query = apply_rotary_emb(query, freqs_cis, 1, query.shape[1], self.heads, self.head_dim)
            key = apply_rotary_emb(key, freqs_cis, 1, key.shape[1], self.heads, self.head_dim)


        # Compute joint attention
        # Note: For batch size 1 testing, we skip valid_length.
        # TODO: Add valid_length support for batched inference with padding.
        attn_out = flash_attention_gpu(
           query,
           key,
           value,
           MHAMaskVariant.NULL_MASK,
           self.scale,
        )

        # attn_out: (B, S, H, D) -> (B, S, H*D)
        hidden_states = F.reshape(attn_out, shape=[B, S, -1])
        hidden_states = hidden_states.cast(query.dtype)

        output = self.to_out(hidden_states)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
    ):
        self.dim = dim
        head_dim = dim // n_heads

        self.attention = ZImageSingleStreamAttention(
            dim, n_heads, head_dim, 1e-5
        )

        self.feed_forward = FeedForward(dim, int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, norm_eps)
        self.ffn_norm1 = RMSNorm(dim, norm_eps)

        self.attention_norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim)
            )

    def __call__(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        adaln_input: Tensor | None = None,
        valid_length: Tensor | None = None,  # Always None for batch size 1
    ) -> Tensor:
        if self.modulation:
            if adaln_input is None:
                raise ValueError("adaln_input must not be None")
            scale_msa, gate_msa, scale_mlp, gate_mlp = F.chunk(
                F.unsqueeze(self.adaLN_modulation(adaln_input), 1), 4, axis=2
            )
            gate_msa, gate_mlp = F.tanh(gate_msa), F.tanh(gate_mlp)
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=valid_length,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            )
        else:
            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=valid_length,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        self.norm_final = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, out_channels)

        self.adaLN_modulation = nn.Sequential(
            SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size),
        )

    def __call__(self, x: Tensor, c: Tensor) -> Tensor:
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * F.unsqueeze(scale, 1)
        x = self.linear(x)
        return x


class RopeEmbedder(nn.Module):
    """Graph-compilable RoPE embedder.

    Computes RoPE embeddings directly from position IDs.
    Avoids F.gather by computing cos/sin directly from positions.
    """

    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: Sequence[int] = (16, 56, 56),
        axes_lens: Sequence[int] = (64, 128, 128),
        device: Device | None = None,
    ):
        self.theta = theta
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = tuple(axes_lens)
        self.device = device if device is not None else CPU()
        if len(axes_dims) != len(axes_lens):
            raise ValueError(
                "axes_dims and axes_lens must have the same length"
            )

    @property
    def local_parameters(self) -> list[tuple[str, Tensor]]:
        """Override to return empty list - no loadable parameters."""
        return []

    def _compute_axis_rope(self, positions: Tensor, dim: int) -> Tensor:
        """Compute RoPE embeddings for a single axis from position indices.

        Args:
            positions: Position indices tensor of shape (seq_len,)
            dim: Dimension for this axis

        Returns:
            Tensor of shape (seq_len, dim // 2, 2) containing [cos, sin]
        """
        # Compute inverse frequencies for this axis
        iota = F.range(0, dim, step=2, dtype=DType.float64, device=self.device)
        inv_freq = F.cast(1.0 / (self.theta ** (iota / dim)), DType.float32)

        # Compute angles: positions * inv_freq using outer product
        pos_float = F.cast(positions, DType.float32)
        angles = F.outer(pos_float, inv_freq)  # (seq_len, dim // 2)

        # Stack cos and sin
        freqs_cis = F.stack([F.cos(angles), F.sin(angles)], axis=-1)
        return freqs_cis

    def __call__(self, ids: Tensor) -> Tensor:
        """Compute RoPE embeddings from position IDs.

        Args:
            ids: Position IDs tensor of shape (seq_len, 3) where each column
                 corresponds to one axis.

        Returns:
            RoPE embeddings of shape (seq_len, total_dim // 2, 2)
        """
        # Compute RoPE for each axis directly from position IDs
        result_0 = self._compute_axis_rope(ids[:, 0], self.axes_dims[0])
        result_1 = self._compute_axis_rope(ids[:, 1], self.axes_dims[1])
        result_2 = self._compute_axis_rope(ids[:, 2], self.axes_dims[2])

        # Concatenate along the dimension axis
        return F.concat([result_0, result_1, result_2], axis=1)





class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        all_patch_size: Sequence[int] = (2,),
        all_f_patch_size: Sequence[int] = (1,),
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: Sequence[int] = [32, 48, 48],
        axes_lens: Sequence[int] = [1024, 512, 512],
        device: Device | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        if len(all_patch_size) != len(all_f_patch_size):
            raise ValueError(
                "all_patch_size and all_f_patch_size must have the same length"
            )

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(
            all_patch_size, all_f_patch_size, strict=False
        ):
            x_embedder = nn.Linear(
                f_patch_size * patch_size * patch_size * in_channels,
                dim,
                bias=True,
            )
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(
                dim, patch_size * patch_size * f_patch_size * self.out_channels
            )
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = ModuleDict(all_x_embedder)
        self.all_final_layer = ModuleDict(all_final_layer)
        self.noise_refiner = ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id, dim, n_heads, norm_eps, qk_norm, modulation=False
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(
            min(dim, ADALN_EMBED_DIM), mid_size=1024
        )
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = Tensor.zeros((1, dim))
        self.cap_pad_token = Tensor.zeros((1, dim))

        self.layers = ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id, dim, n_heads, norm_eps, qk_norm
                )
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        if head_dim != sum(axes_dims):
            raise ValueError("head_dim must be equal to sum(axes_dims)")
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(rope_theta, axes_dims, axes_lens, device)

        # Fixed shape parameters for graph compilation (batch_size=1, 1024x1024 output)
        # These avoid int(tensor.shape) during graph tracing
        self._compile_C = 16
        self._compile_F_dim = 1
        self._compile_H_dim = 128
        self._compile_W_dim = 128
        self._compile_cap_seq_len = 101

    def unpatchify(
        self,
        x: list[Tensor],
        size: list[Tuple],
        patch_size: int,
        f_patch_size: int,
    ) -> list[Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        if len(size) != bsz:
            raise ValueError("size must have the same length as batch size")
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .reshape(
                    (F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                )
                .permute((6, 0, 3, 1, 4, 2, 5))
                .reshape((self.out_channels, F, H, W))
            )
        return x

    @staticmethod
    def create_coordinate_grid(
        size: Sequence[int],
        start: Sequence[int] | None = None,
        device: Device | None = None,
    ) -> Tensor:
        """Create a coordinate grid for position encoding.

        Graph-compilable version that explicitly handles 3D grids (F, H, W)
        without list operations.

        Args:
            size: Tuple of (F, H, W) dimensions
            start: Tuple of (f0, h0, w0) starting coordinates
            device: Target device

        Returns:
            Tensor of shape (F, H, W, 3) with coordinates
        """
        if start is None:
            start = (0, 0, 0)

        f_size, h_size, w_size = size[0], size[1], size[2]
        f_start, h_start, w_start = start[0], start[1], start[2]

        # Create ranges for each axis
        f_axis = F.range(f_start, f_start + f_size, dtype=DType.int32, device=device)
        h_axis = F.range(h_start, h_start + h_size, dtype=DType.int32, device=device)
        w_axis = F.range(w_start, w_start + w_size, dtype=DType.int32, device=device)

        # Reshape for broadcasting: (F, 1, 1), (1, H, 1), (1, 1, W)
        f_grid = f_axis.reshape((f_size, 1, 1))
        h_grid = h_axis.reshape((1, h_size, 1))
        w_grid = w_axis.reshape((1, 1, w_size))

        # Broadcast to full size
        f_grid = F.broadcast_to(f_grid, (f_size, h_size, w_size))
        h_grid = F.broadcast_to(h_grid, (f_size, h_size, w_size))
        w_grid = F.broadcast_to(w_grid, (f_size, h_size, w_size))

        # Stack along last dimension: (F, H, W, 3)
        return F.stack([f_grid, h_grid, w_grid], axis=-1)


    def patchify_and_embed(
        self,
        all_image: list[Tensor],
        all_cap_feats: list[Tensor],
        patch_size: int,
        f_patch_size: int,
    ) -> Tuple[
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ]:
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for image, cap_feat in zip(all_image, all_cap_feats, strict=False):
            ### Process Caption
            # cap_ori_len = len(cap_feat)
            cap_ori_len = int(cap_feat.shape[0])
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = F.flatten(
                self.create_coordinate_grid(
                    size=(cap_ori_len + cap_padding_len, 1, 1),
                    start=(1, 0, 0),
                    device=device,
                ),
                0,
                2,
            )
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            cap_pad_mask = F.concat(
                [
                    Tensor.zeros(
                        (cap_ori_len,), dtype=DType.bool, device=device
                    ),
                    Tensor.ones(
                        (cap_padding_len,), dtype=DType.bool, device=device
                    ),
                ],
                axis=0,
            )
            all_cap_pad_mask.append(
                cap_pad_mask
                if cap_padding_len > 0
                else Tensor.zeros(
                    (cap_ori_len,), dtype=DType.bool, device=device
                )
            )

            # padded feature
            if cap_padding_len > 0:
                cap_feat_last = cap_feat[-1:]
                cap_repeats = [cap_padding_len] + [1] * (cap_feat_last.rank - 1)
                cap_padded_feat = F.concat(
                    [cap_feat, F.tile(cap_feat_last, tuple(cap_repeats))],
                    axis=0,
                )
            else:
                cap_padded_feat = cap_feat

            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F_dim, H_dim, W_dim = image.shape
            C, F_dim, H_dim, W_dim = int(C), int(F_dim), int(H_dim), int(W_dim)
            all_image_size.append((F_dim, H_dim, W_dim))
            F_tokens, H_tokens, W_tokens = F_dim // pF, H_dim // pH, W_dim // pW

            image = image.reshape((C, F_tokens, pF, H_tokens, pH, W_tokens, pW))
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute((1, 3, 5, 2, 4, 6, 0)).reshape(
                (F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
            )

            image_ori_len = int(image.shape[0])
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = F.flatten(
                self.create_coordinate_grid(
                    size=(F_tokens, H_tokens, W_tokens),
                    start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                    device=device,
                ),
                0,
                2,
            )
            if image_padding_len > 0:
                image_padded_pos_ids = F.concat(
                    [
                        image_ori_pos_ids,
                        F.tile(
                            F.flatten(
                                self.create_coordinate_grid(
                                    size=(1, 1, 1),
                                    start=(0, 0, 0),
                                    device=device,
                                ),
                                0,
                                2,
                            ),
                            (image_padding_len, 1),
                        ),
                    ],
                    axis=0,
                )
            else:
                image_padded_pos_ids = image_ori_pos_ids
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            image_pad_mask = F.concat(
                [
                    Tensor.zeros(
                        (image_ori_len,), dtype=DType.bool, device=device
                    ),
                    Tensor.ones(
                        (image_padding_len,), dtype=DType.bool, device=device
                    ),
                ],
                axis=0,
            )
            all_image_pad_mask.append(
                image_pad_mask
                if image_padding_len > 0
                else Tensor.zeros(
                    (image_ori_len,), dtype=DType.bool, device=device
                )
            )
            # padded feature
            if image_padding_len > 0:
                image_last = image[-1:]
                image_repeats = [image_padding_len] + [1] * (
                    image_last.rank - 1
                )
                image_padded_feat = F.concat(
                    [image, F.tile(image_last, tuple(image_repeats))],
                    axis=0,
                )
                all_image_out.append(image_padded_feat)
            else:
                all_image_out.append(image)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def __call__(
        self,
        x: Tensor,
        t: Tensor,
        cap_feats: Tensor,
        return_dict: bool = False,
    ):
        """Graph-compilable forward pass for batch_size=1.

        Uses fixed shape parameters stored in self._compile_* attributes
        to avoid int(tensor.shape) during graph tracing.

        Args:
            x: Image latent tensor of shape (C, F_dim, H_dim, W_dim)
            t: Timestep tensor of shape (1,)
            cap_feats: Caption features tensor of shape (cap_seq_len, hidden_dim)
            return_dict: Whether to return a dict (default False for compilation)
        """
        # Use fixed shape parameters from class attributes
        C = self._compile_C
        F_dim = self._compile_F_dim
        H_dim = self._compile_H_dim
        W_dim = self._compile_W_dim
        cap_seq_len = self._compile_cap_seq_len

        patch_size: int = 2
        f_patch_size: int = 1

        device = x.device

        # Time embedding
        t = t * self.t_scale
        t = self.t_embedder(t)
        adaln_input = t.cast(x.dtype)

        # Patchify image - using class shape parameters (no int() on tensor.shape)
        pF, pH, pW = f_patch_size, patch_size, patch_size
        x_size = (F_dim, H_dim, W_dim)
        F_tokens, H_tokens, W_tokens = F_dim // pF, H_dim // pH, W_dim // pW


        # Reshape to patches: (C, F, H, W) -> (F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        x = x.reshape((C, F_tokens, pF, H_tokens, pH, W_tokens, pW))
        x = x.permute((1, 3, 5, 2, 4, 6, 0)).reshape(
            (F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        )

        image_seq_len = F_tokens * H_tokens * W_tokens


        # Embed image patches
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Create position IDs for image
        x_pos_ids = F.flatten(
            self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_seq_len + 1, 0, 0),
                device=device,
            ),
            0,
            2,
        )

        # RoPE embeddings for image
        x_freqs_cis = self.rope_embedder(x_pos_ids)

        # Embed caption features
        cap_feats = self.cap_embedder(cap_feats)

        # Create position IDs for caption
        cap_pos_ids = F.flatten(
            self.create_coordinate_grid(
                size=(cap_seq_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ),
            0,
            2,
        )

        # RoPE embeddings for caption
        cap_freqs_cis = self.rope_embedder(cap_pos_ids)

        # Add batch dimension: (seq_len, dim) -> (1, seq_len, dim)
        x = F.unsqueeze(x, 0)
        x_freqs_cis = F.unsqueeze(x_freqs_cis, 0)
        cap_feats = F.unsqueeze(cap_feats, 0)
        cap_freqs_cis = F.unsqueeze(cap_freqs_cis, 0)

        # Noise refiner layers (process image patches)
        for layer in self.noise_refiner:
            x = layer(x, x_freqs_cis, adaln_input)

        # Context refiner layers (process caption)
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_freqs_cis)

        # Unified: concatenate image and caption features along sequence dimension
        # Shape: (1, image_seq_len + cap_seq_len, hidden_dim)
        unified = F.concat([x, cap_feats], axis=1)
        unified_freqs_cis = F.concat([x_freqs_cis, cap_freqs_cis], axis=1)

        # Main transformer layers
        for layer in self.layers:
            unified = layer(unified, unified_freqs_cis, adaln_input)

        # Final layer
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified, adaln_input
        )

        # Remove batch dimension and split: take only image portion
        unified = F.squeeze(unified, 0)  # (total_seq_len, hidden_dim)
        x = unified[:image_seq_len]  # (image_seq_len, hidden_dim)

        # Unpatchify: (image_seq_len, hidden_dim) -> (out_channels, F, H, W)
        # The hidden_dim after final layer should be pF * pH * pW * out_channels
        out_channels = self.out_channels
        x = x.reshape((F_tokens, H_tokens, W_tokens, pF, pH, pW, out_channels))
        x = x.permute((6, 0, 3, 1, 4, 2, 5))  # (out_channels, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        x = x.reshape((out_channels, F_dim, H_dim, W_dim))

        if not return_dict:
            return x

        return Transformer2DModelOutput(sample=x)

