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
from typing import Sequence

import max.nn.module_v3 as nn
import max.experimental.functional as F
from max.experimental import tensor
from max.experimental.tensor import Tensor
from max.nn.module_v3.sequential import ModuleList
from max.dtype import DType
from max.nn.kernels import flash_attention_gpu

from ..model_config import TransformerConfig
from .layers import RMSNorm, LayerNorm, SiLU, ModuleDict, pad_sequence, Transformer2DModelOutput, create_attention_mask, masked_scatter

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: int | None = None, frequency_embedding_size: int = 256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size),
            SiLU(),
            nn.Linear(mid_size, out_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = (
            F.exp(
                -math.log(max_period)
                * F.range(0, half, 1, dtype=DType.float32, device=t.device)
                / half
            )
        )
        args = t[:, None].cast(DType.float32) * freqs[None, :]
        embedding = F.concat([F.cos(args), F.sin(args)], axis=-1)

        if dim % 2:
            zeros = Tensor.zeros([embedding.shape[0], 1], dtype=embedding.dtype, device=embedding.device)
            embedding = F.concat([embedding, zeros], axis=-1)

        return embedding

    def __call__(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        #t_freq = t_freq.cast(self.mlp[0].weight.dtype)
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
        super().__init__()

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

        if hidden_states.rank > 3:
             # Flatten spatial/sequence dimensions: (B, D1, D2, ..., C) -> (B, S, C)
             B = hidden_states.shape[0]
             C = hidden_states.shape[-1]
             S = 1
             for d in hidden_states.shape[1:-1]:
                 S *= int(d)
             hidden_states = hidden_states.reshape([B, S, C])

        B, S, _ = hidden_states.shape
        query = query.reshape((B, S, self.heads, self.head_dim)).transpose(1, 2)
        key = key.reshape((B, S, self.heads, self.head_dim)).transpose(1, 2)
        value = value.reshape((B, S, self.heads, self.head_dim)).transpose(1, 2)

        # Apply norms
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE
        # TODO: Verify this implementation
        def apply_rotary_emb(x_in: Tensor, freqs_cis: Tensor) -> Tensor:
            # x_in shape: (B, H, S, D)
            # Reshape for complex interpretation: [..., head_dim] -> [..., head_dim/2, 2]
            target_shape = list(x_in.shape[:-1]) + [-1, 2]
            x_complex = x_in.cast(DType.float32).reshape(target_shape)
            # x_reshaped is already [..., 2] so no need for as_interleaved_complex
            freqs_cis_expanded = F.unsqueeze(freqs_cis, 1)  # Expand at dim 1 (heads) to broadcast: (B, 1, S, D, 2)

            # Apply complex multiplication
            x_rotated = F.complex_mul(x_complex, freqs_cis_expanded)
            # Reshape back to real: flatten the last 2 dims
            x_out = x_rotated.reshape(x_in.shape)
            return x_out.cast(x_in.dtype)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.rank == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
        # attn_out = flash_attention_gpu(
        #    query,
        #    key,
        #    value,
        #    attention_mask,
        #    self.scale,
        # )
        # hidden_states = hidden_states.flatten(2, 3)
        # Manual attention implementation for CPU compatibility
        # query, key, value shape: (B, H, S, D) - permuted previously to be (B, H, S, D) for flash_attn
        # But for matmul we want:
        # Q: (B, H, S, D)
        # K: (B, H, S, D) -> K.T (last 2 dims): (B, H, D, S)
        # attn = (Q @ K.T) * scale

        attn_scores = F.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
             # Mask is (B, 1, 1, S) broadcasted to (B, H, S, S)
             # Apply mask (assuming additive mask where False/0 means mask out?)
             # Usually attention_mask in diffusers is 1 for keep, 0 for discard?
             # Function create_attention_mask returns boolean.
             # We need to fill -inf where mask is False.
             # F.where(mask, attn_scores, -inf)

             # Check mask type. `create_attention_mask` returns boolean tensor (True for valid).
             # So we keep where True.
             min_val = -1e9 # -inf proxy
             if attention_mask.dtype == DType.bool:
                  attn_scores = F.where(attention_mask, attn_scores, min_val)
             else:
                  # If float mask (additive), adds directly. But here we assume boolean based on usage.
                  pass

        attn_probs = F.softmax(attn_scores, axis=-1)
        attn_out = F.matmul(attn_probs, value) # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)

        # attn_out is (B, H, S, D).
        # We need to match the next steps which expect (B, H, S, D) and then flatten.
        # attn_out: (B, H, S, D) -> (B, S, H, D) -> (B, S, H*D)
        hidden_states = F.flatten(attn_out.transpose(1, 2), start_dim=2, end_dim=3)
        hidden_states = hidden_states.cast(query.dtype)

        output = self.to_out(hidden_states)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
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
        super().__init__()
        self.dim = dim
        head_dim = dim // n_heads

        self.attention = ZImageSingleStreamAttention(dim, n_heads, head_dim, 1e-5)

        self.feed_forward = FeedForward(dim, int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, norm_eps)
        self.ffn_norm1 = RMSNorm(dim, norm_eps)

        self.attention_norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim))

    def __call__(
        self,
        x: Tensor,
        attn_mask: Tensor,
        freqs_cis: Tensor,
        adaln_input: Tensor | None = None,
    ) -> Tensor:
        if self.modulation:
            if adaln_input is None:
                raise ValueError("adaln_input must not be None")
            scale_msa, gate_msa, scale_mlp, gate_mlp = F.chunk(F.unsqueeze(self.adaLN_modulation(adaln_input), 1), 4, axis=2)
            gate_msa, gate_mlp = F.tanh(gate_msa), F.tanh(gate_mlp)
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            # Attention block
            attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: Sequence[int] = (16, 56, 56),
        axes_lens: Sequence[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        if len(axes_dims) != len(axes_lens):
            raise ValueError("axes_dims and axes_lens must have the same length")
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: Sequence[int], end: Sequence[int], theta: float = 256.0, device=None) -> list[Tensor]:
        freqs_cis = []
        for i, (d, e) in enumerate(zip(dim, end)):
            freqs = 1.0 / (theta ** (F.range(0, d, 2, dtype=DType.float64, device=device) / d))
            timestep = F.range(0, e, dtype=DType.float64, device=device)
            angles = F.outer(timestep, freqs).cast(DType.float32)
            # Create complex representation [real, imag]
            freqs_cos = F.cos(angles)
            freqs_sin = F.sin(angles)
            freqs_cis_i = F.stack([freqs_cos, freqs_sin], axis=-1)
            freqs_cis.append(freqs_cis_i)

        return freqs_cis

    def __call__(self, ids: Tensor) -> Tensor:
        if len(ids.shape) != 2:
            raise ValueError("ids must be a 2D tensor")
        if ids.shape[-1] != len(self.axes_dims):
            raise ValueError("ids must have the same number of columns as the number of axes")
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta, device=device)
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(F.gather(self.freqs_cis[i], index, axis=0))
        return F.concat(result, axis=1)


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
    ) -> None:
        super().__init__()
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
            raise ValueError("all_patch_size and all_f_patch_size must have the same length")

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = ModuleDict(all_x_embedder)
        self.all_final_layer = ModuleDict(all_final_layer)
        self.noise_refiner = ModuleList([ZImageTransformerBlock(1000 + layer_id, dim, n_heads, norm_eps, qk_norm, modulation=True) for layer_id in range(n_refiner_layers)])
        self.context_refiner = ModuleList([ZImageTransformerBlock(layer_id, dim, n_heads, norm_eps, qk_norm, modulation=False) for layer_id in range(n_refiner_layers)])
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))

        self.x_pad_token = Tensor.zeros((1, dim))
        self.cap_pad_token = Tensor.zeros((1, dim))

        self.layers = ModuleList([ZImageTransformerBlock(layer_id, dim, n_heads, norm_eps, qk_norm) for layer_id in range(n_layers)])
        head_dim = dim // n_heads
        if head_dim != sum(axes_dims):
            raise ValueError("head_dim must be equal to sum(axes_dims)")
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(rope_theta, axes_dims, axes_lens)

    def unpatchify(self, x: List[Tensor], size: List[Tuple], patch_size: int, f_patch_size: int) -> List[Tensor]:
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
                .reshape((F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels))
                .permute((6, 0, 3, 1, 4, 2, 5))
                .reshape((self.out_channels, F, H, W))
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        grids = []
        for i, (x0, span) in enumerate(zip(start, size)):
            # Create range for this dimension
            axis = F.range(x0, x0 + span, dtype=DType.int32, device=device)

            # Reshape to allow broadcasting: (1, ..., span, ..., 1)
            target_shape = [1] * len(size)
            target_shape[i] = span
            axis = axis.reshape(tuple(target_shape))

            # Broadcast to full size
            grid = F.broadcast_to(axis, size)
            grids.append(grid)

        return F.stack(grids, axis=-1)

    def patchify_and_embed(
        self,
        all_image: List[Tensor],
        all_cap_feats: List[Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
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

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            #cap_ori_len = len(cap_feat)
            cap_ori_len = int(cap_feat.shape[0])
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = F.flatten(self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ), 0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            cap_pad_mask = F.concat(
                [
                    Tensor.zeros((cap_ori_len,), dtype=DType.bool, device=device),
                    Tensor.ones((cap_padding_len,), dtype=DType.bool, device=device),
                ],
                axis=0,
            )
            all_cap_pad_mask.append(
                cap_pad_mask if cap_padding_len > 0 else Tensor.zeros((cap_ori_len,), dtype=DType.bool, device=device)
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
            image = image.permute((1, 3, 5, 2, 4, 6, 0)).reshape((F_tokens * H_tokens * W_tokens, pF * pH * pW * C))

            image_ori_len = int(image.shape[0])
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = F.flatten(self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ), 0, 2)
            if image_padding_len > 0:
                image_padded_pos_ids = F.concat(
                    [
                        image_ori_pos_ids,
                        F.tile(F.flatten(
                            self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device),
                            0, 2), (image_padding_len, 1)),
                    ],
                    axis=0,
                )
            else:
                image_padded_pos_ids = image_ori_pos_ids
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            image_pad_mask = F.concat(
                [
                    Tensor.zeros((image_ori_len,), dtype=DType.bool, device=device),
                    Tensor.ones((image_padding_len,), dtype=DType.bool, device=device),
                ],
                axis=0,
            )
            all_image_pad_mask.append(
                image_pad_mask
                if image_padding_len > 0
                else Tensor.zeros((image_ori_len,), dtype=DType.bool, device=device)
            )
            # padded feature
            if image_padding_len > 0:
                image_last = image[-1:]
                image_repeats = [image_padding_len] + [1] * (image_last.rank - 1)
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
        x: List[Tensor],
        t: Tensor,
        cap_feats: List[Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        return_dict: bool = True,
    ):
        if patch_size not in self.all_patch_size:
            raise ValueError(f"patch_size must be in {self.all_patch_size}")
        if f_patch_size not in self.all_f_patch_size:
            raise ValueError(f"f_patch_size must be in {self.all_f_patch_size}")

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [int(_.shape[0]) for _ in x]
        if not all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens):
            raise ValueError("all item seqlens must be a multiple of SEQ_MULTI_OF")
        x_max_item_seqlen = max(x_item_seqlens)

        x = F.concat(x, axis=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.cast(x.dtype)
        #x[F.concat(x_inner_pad_mask)] = self.x_pad_token
        # Use masked_scatter for pad token assignment (immutable tensor compatible)
        pad_mask = F.concat(x_inner_pad_mask)
        #x = masked_scatter(x, pad_mask, self.x_pad_token)
        # Ensure pad token is on the correct device
        x_pad_token = self.x_pad_token.to(x.device).cast(x.dtype)
        x = masked_scatter(x, pad_mask, x_pad_token)
        x = list(F.split(x, x_item_seqlens, axis=0))
        x_freqs_cis = list(F.split(self.rope_embedder(F.concat(x_pos_ids, axis=0)), [int(_.shape[0]) for _ in x_pos_ids], axis=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

        #x_attn_mask = Tensor.zeros((bsz, x_max_item_seqlen), dtype=DType.bool, device=device)
        #for i, seq_len in enumerate(x_item_seqlens):
        #    x_attn_mask[i, :seq_len] = 1
        # Use functional attention mask construction (immutable tensor compatible)
        x_attn_mask = create_attention_mask(bsz, x_max_item_seqlen, x_item_seqlens)

        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [int(_.shape[0]) for _ in cap_feats]
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = F.concat(cap_feats, axis=0)
        cap_feats = self.cap_embedder(cap_feats)
        #cap_feats[F.concat(cap_inner_pad_mask)] = self.cap_pad_token
        # Use masked_scatter for pad token assignment (immutable tensor compatible)
        cap_pad_mask = F.concat(cap_inner_pad_mask)
        #cap_feats = masked_scatter(cap_feats, cap_pad_mask, self.cap_pad_token)
        # Ensure pad token is on the correct device
        cap_pad_token = self.cap_pad_token.to(cap_feats.device).cast(cap_feats.dtype)
        cap_feats = masked_scatter(cap_feats, cap_pad_mask, cap_pad_token)
        cap_feats = list(F.split(cap_feats, cap_item_seqlens, axis=0))
        cap_freqs_cis = list(
            F.split(self.rope_embedder(F.concat(cap_pos_ids, axis=0)), [int(_.shape[0]) for _ in cap_pos_ids], axis=0)
        )

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

        #cap_attn_mask = Tensor.zeros((bsz, cap_max_item_seqlen), dtype=DType.bool, device=device)
        #for i, seq_len in enumerate(cap_item_seqlens):
        #    cap_attn_mask[i, :seq_len] = 1
        # Use functional attention mask construction (immutable tensor compatible)
        cap_attn_mask = create_attention_mask(bsz, cap_max_item_seqlen, cap_item_seqlens)

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(F.concat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(F.concat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        if not unified_item_seqlens == [int(_.shape[0]) for _ in unified]:
            raise ValueError("all item seqlens must be a multiple of SEQ_MULTI_OF")
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        #unified_attn_mask = Tensor.zeros((bsz, unified_max_item_seqlen), dtype=DType.bool, device=device)
        #for i, seq_len in enumerate(unified_item_seqlens):
        #    unified_attn_mask[i, :seq_len] = 1
        # Use functional attention mask construction (immutable tensor compatible)
        unified_attn_mask = create_attention_mask(bsz, unified_max_item_seqlen, unified_item_seqlens)

        for layer in self.layers:
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = [F.squeeze(t, 0) for t in F.split(unified, [1] * int(unified.shape[0]), axis=0)]
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)
