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

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.driver import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.layer import Layer, Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(Module):
    def __init__(
        self,
        out_size: int,
        mid_size: int | None = None,
        frequency_embedding_size: int = 256,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.CPU(),
    ):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        
        self.linear_1 = Linear(
            frequency_embedding_size,
            mid_size,
            has_bias=True,
            dtype=dtype,
            device=device,
        )
        self.linear_2 = Linear(
            mid_size,
            out_size,
            has_bias=True,
            dtype=dtype,
            device=device,
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: TensorValue, dim: int, max_period: int = 10000
    ) -> TensorValue:
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = ops.exp(-math.log(max_period) * ops.range(0, half, 1, dtype=DType.float32, device=t.device) / half)
        args = ops.unsqueeze(t.cast(DType.float32), -1) * ops.unsqueeze(freqs, 0)
        embedding = ops.concat([ops.cos(args), ops.sin(args)], axis=-1)
        
        if dim % 2:
            zeros = Tensor.zeros([embedding.shape[0], 1], dtype=embedding.dtype, device=embedding.device)
            embedding = ops.concat([embedding, zeros], axis=-1)
            
        return embedding

    def __call__(self, t: TensorValue) -> TensorValue:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # TODO: Does this casting make sense?
        # if weight_dtype.is_floating_point:
        t_freq = t_freq.cast(self.linear_1.weight.dtype)
        t_emb = self.linear_2(ops.silu(self.linear_1(t_freq)))
        return t_emb


class ZImageAttention(Module):
    """
    Z-Image specific attention module.
    Stateless implementation using flash_attention_ragged_gpu.
    """
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        dtype: DType = DType.float32,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-5,
        flash_attention: bool = False,
        devices: Sequence[DeviceRef] | None = None,
        float8_config: Float8Config | None = None,
    ):
        super().__init__()
        self.n_heads = num_attention_heads
        self.head_dim = dim // num_attention_heads
        self.scale = self.head_dim**-0.5
        self.devices = devices if devices is not None else [DeviceRef.CPU()]
        
        self.qkv_proj = Weight(
            name="qkv.weight",
            dtype=dtype,
            shape=(3 * dim, self.head_dim),
            device=self.devices[0],
        )
        self.o_proj = Linear(dim, dim, has_bias=False, dtype=dtype, device=device)
        
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            from max.nn.norm.layer_norm import LayerNorm
            self.q_norm = LayerNorm(dim, devices=[device], dtype=dtype, eps=rms_norm_eps, use_bias=True)
            self.k_norm = LayerNorm(self.n_kv_heads * self.head_dim, devices=[device], dtype=dtype, eps=rms_norm_eps, use_bias=True)
        else:
            self.q_norm = None
            self.k_norm = None

    def apply_rope(self, x: TensorValue, freqs_cis: TensorValue) -> TensorValue:
        """
        Apply Rotary Positional Embeddings (RoPE) to the input tensor.
        Performs complex number multiplication: (a + ib) * (cos + isin).
        """
        freqs_cos = ops.cos(freqs_cis)
        freqs_sin = ops.sin(freqs_cis)
        
        # Reshape for complex representation: [..., head_dim//2, 2]
        x_reshaped = ops.reshape(x, shape=[x.shape[0], x.shape[1], self.head_dim // 2, 2])
        x_re = x_reshaped[..., 0]
        x_im = x_reshaped[..., 1]
        
        # Broadcast frequencies
        freqs_cos = ops.unsqueeze(freqs_cos, 1)
        freqs_sin = ops.unsqueeze(freqs_sin, 1)
        
        # Rotate: (a + ib) * (cos + isin) = (acos - bsin) + i(asin + bcos)
        out_re = x_re * freqs_cos - x_im * freqs_sin
        out_im = x_re * freqs_sin + x_im * freqs_cos
        
        out = ops.stack([out_re, out_im], axis=-1)
        return ops.reshape(out, shape=[x.shape[0], x.shape[1], self.head_dim])

    def __call__(
        self,
        x: TensorValue,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        max_seq_len: TensorValue,
    ) -> TensorValue:
        total_seq_len = x.shape[0]

        qkv = self.qkv_proj(x)
        
        q = ops.reshape(q, shape=[total_seq_len, self.n_heads, self.head_dim])
        k = ops.reshape(k, shape=[total_seq_len, self.n_kv_heads, self.head_dim])
        v = ops.reshape(v, shape=[total_seq_len, self.n_kv_heads, self.head_dim])

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        freqs_cis = ops.cast(freqs_cis, q.dtype)
        q = self.apply_rope(q, freqs_cis)
        k = self.apply_rope(k, freqs_cis)
        
        from max.nn.kernels import flash_attention_ragged_gpu
        
        attn_out = flash_attention_ragged_gpu(
            q, k, v,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seq_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale
        )
        
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, self.n_heads * self.head_dim])
        return self.o_proj(attn_out)


class FeedForward(Module):
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.CPU(),
    ):
        super().__init__()
        self.w1 = Linear(dim, hidden_dim, has_bias=False, dtype=dtype, device=device)
        self.w2 = Linear(hidden_dim, dim, has_bias=False, dtype=dtype, device=device)
        self.w3 = Linear(dim, hidden_dim, has_bias=False, dtype=dtype, device=device)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.w2(ops.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.CPU(),
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id

        self.attention = ZImageAttention(
            dim=dim,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            dtype=dtype,
            device=device,
            use_qk_norm=qk_norm,
            rms_norm_eps=1e-5,
        )

        self.feed_forward = FeedForward(
            dim=dim, 
            hidden_dim=int(dim / 3 * 8),
            dtype=dtype,
            device=device
        )

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps, dtype=dtype)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps, dtype=dtype)
        
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps, dtype=dtype)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps, dtype=dtype)

        self.adaLN_modulation = Linear(
            min(dim, ADALN_EMBED_DIM), 
            6 * dim, 
            has_bias=True,
            dtype=dtype,
                device=device
            )

    def __call__(
        self,
        x: TensorValue,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        max_seq_len: TensorValue,
        adaln_input: TensorValue | None = None,
    ) -> TensorValue:
        
        layer_idx = ops.constant(self.layer_id, DType.uint32, device=x.device)

        if adaln_input is None:
            raise ValueError("adaln_input required when modulation is True")
            
        mod_out = self.adaLN_modulation(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ops.chunk(mod_out, 6, axis=1)
        
        gate_msa = ops.tanh(gate_msa)
        gate_mlp = ops.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        
        # Attention block
        norm_x = self.attention_norm1(x)
        mod_x = norm_x * scale_msa
        
        attn_out = self.attention(
            mod_x,
            freqs_cis,
            input_row_offsets,
            max_seq_len
        )
        
        x = x + gate_msa * self.attention_norm2(attn_out)
        
        # FFN block
        norm_x_ffn = self.ffn_norm1(x)
        mod_x_ffn = norm_x_ffn * scale_mlp
        
        ffn_out = self.feed_forward(mod_x_ffn)
        
        x = x + gate_mlp * self.ffn_norm2(ffn_out)

        return x


class FinalLayer(Module):
    def __init__(
        self, 
        hidden_size: int, 
        out_channels: int,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.CPU(),
    ):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6, dtype=dtype) 
        self.linear = Linear(hidden_size, out_channels, has_bias=True, dtype=dtype, device=device)
        
        self.adaLN_modulation_0 = Linear(
            min(hidden_size, ADALN_EMBED_DIM), 
            hidden_size, 
            has_bias=True,
            dtype=dtype,
            device=device
        )

    def __call__(self, x: TensorValue, c: TensorValue) -> TensorValue:
        scale = 1.0 + self.adaLN_modulation_0(ops.silu(c))
        x = self.norm_final(x) * scale
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
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: Sequence[int], end: Sequence[int], theta: float = 256.0, device: DeviceRef = DeviceRef.CPU()):
        freqs_cis = []
        for d, e in zip(dim, end):
            arange = ops.range(0, d, 2, device=device, dtype=DType.float32)
            freqs = 1.0 / (theta ** (arange / d))
            timestep = ops.range(0, e, 1, device=device, dtype=DType.float32)
            freqs = ops.outer(timestep, freqs)
            freqs_cis.append(freqs)
            
        return freqs_cis

    def __call__(self, ids: TensorValue) -> TensorValue:
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta, device=ids.device)
            
        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            gathered = ops.gather(self.freqs_cis[i], index, axis=0)
            result.append(gathered)
            
        return ops.concat(result, axis=1)


class ZImageTransformer2DModel(Module):
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        sample_size=32,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.sample_size = sample_size
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        


        assert len(all_patch_size) == len(all_f_patch_size)

        self.all_x_embedder = {}
        self.all_final_layer = {}
        
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            key = f"{patch_size}-{f_patch_size}"
            x_embedder = Linear(
                f_patch_size * patch_size * patch_size * in_channels, 
                dim, 
                has_bias=True,
                dtype=dtype,
                device=device
            )
            setattr(self, f"x_embedder_{key}", x_embedder)
            self.all_x_embedder[key] = x_embedder

            final_layer = FinalLayer(
                dim, 
                patch_size * patch_size * f_patch_size * self.out_channels,
                dtype=dtype,
                device=device
            )
            setattr(self, f"final_layer_{key}", final_layer)
            self.all_final_layer[key] = final_layer

        self.noise_refiner = [
            ZImageTransformerBlock(
                1000 + layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,

                modulation=True,
                dtype=dtype,
                device=device,
            )
            for layer_id in range(n_refiner_layers)
        ]
        for i, layer in enumerate(self.noise_refiner):
            setattr(self, f"noise_refiner_{i}", layer)

        self.context_refiner = [
            ZImageTransformerBlock(
                layer_id,
                dim,
                n_heads,
                n_kv_heads,
                norm_eps,
                qk_norm,

                modulation=False,
                dtype=dtype,
                device=device,
            )
            for layer_id in range(n_refiner_layers)
        ]
        for i, layer in enumerate(self.context_refiner):
            setattr(self, f"context_refiner_{i}", layer)

        self.t_embedder = TimestepEmbedder(
            min(dim, ADALN_EMBED_DIM), 
            mid_size=1024,
            dtype=dtype,
            device=device
        )
        
        self.cap_embedder_norm = RMSNorm(cap_feat_dim, eps=norm_eps, dtype=dtype)
        self.cap_embedder_linear = Linear(cap_feat_dim, dim, has_bias=True, dtype=dtype, device=device)

        self.x_pad_token = Weight(name="x_pad_token", shape=(1, dim), dtype=dtype, device=device)
        
        self.layers = [
            ZImageTransformerBlock(
                layer_id, 
                dim, 
                n_heads, 
                n_kv_heads, 
                norm_eps, 
                qk_norm,

                dtype=dtype,
                device=device
            )
            for layer_id in range(n_layers)
        ]
        for i, layer in enumerate(self.layers):
            setattr(self, f"layers_{i}", layer)

        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, unified: TensorValue, batch_size_dim: Dim | int, total_cap_tokens_dim: Dim | int) -> TensorValue:
        """
        Splits unified tensor into image and caption tokens, and reshapes image tokens.
        """
        # Calculate image tokens length
        img_tokens_dim = unified.shape[0] - total_cap_tokens_dim
        img_tokens_tensor = ops.shape_to_tensor([img_tokens_dim])[0]
        
        # Slice out image tokens
        x_out = ops.slice_tensor(unified, [
            (slice(0, img_tokens_tensor), img_tokens_dim), 
            slice(None)
        ])
        
        p = self.all_patch_size[0]
        C = self.out_channels
        H = self.sample_size
        W = self.sample_size
        
        h_patches = H // p
        w_patches = W // p
        
        # Reshape to [batch, H/p, W/p, p, p, C]
        x_out = ops.reshape(x_out, [batch_size_dim, h_patches, w_patches, p, p, C])
        
        # Permute to [batch, C, H, W]
        x_out = ops.permute(x_out, [0, 5, 1, 3, 2, 4])
        x_out = ops.reshape(x_out, [batch_size_dim, C, H, W])
        
        return x_out

    def patchify_and_embed(
        self,
        all_image: list[TensorValue],
        all_cap_feats: list[TensorValue],
        patch_size: int,
        f_patch_size: int,
    ):
        pass

    def __call__(
        self,
        x: TensorValue, # Flattened/ragged image tokens
        t: TensorValue,
        cap_feats: TensorValue, # Flattened/ragged caption tokens
        input_row_offsets: TensorValue, # To delineate batches
        max_seq_len: TensorValue,
        patch_size: int = 2,
        f_patch_size: int = 1,
        x_pos_ids: TensorValue | None = None,
        cap_pos_ids: TensorValue | None = None,
    ):
        t = t * self.t_scale
        t_emb = self.t_embedder(t)

        key = f"{patch_size}-{f_patch_size}"
        x = self.all_x_embedder[key](x)
        
        cap_feats = self.cap_embedder_linear(self.cap_embedder_norm(cap_feats))
        
        if x_pos_ids is None or cap_pos_ids is None:
             raise ValueError("pos_ids required for MAX port")
             
        x_freqs_cis = self.rope_embedder(x_pos_ids)
        cap_freqs_cis = self.rope_embedder(cap_pos_ids)
        
        # Compute sequence lengths
        seq_lens = input_row_offsets[1:] - input_row_offsets[:-1]
        
        # Create batch indices for each token to gather t_emb
        batch_indices = ops.repeat_interleave(
            ops.range(0, t.shape[0], 1, device=t.device, dtype=DType.int32),
            seq_lens,
            out_dim=x.shape[0]
        )
        
        t_emb_expanded = ops.gather(t_emb, batch_indices, axis=0)
        x = x + t_emb_expanded
        adaln_input = t_emb_expanded
        
        for layer in self.noise_refiner:
            x = layer(x, x_freqs_cis, input_row_offsets, max_seq_len, adaln_input)
            
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_freqs_cis, input_row_offsets, max_seq_len)
            
        # Prepare for unified layers: interleave image and caption tokens
        shape_cap = ops.shape_to_tensor(cap_feats.shape)
        shape_t = ops.shape_to_tensor(t.shape)
        
        total_cap_tokens_tensor = shape_cap[0]
        batch_size_tensor = shape_t[0]
        
        cap_len_tensor = total_cap_tokens_tensor // batch_size_tensor
        cap_len_tensor = ops.cast(cap_len_tensor, DType.int32)
        
        adaln_input_cap = ops.repeat_interleave(
            t_emb,
            cap_len_tensor,
            axis=0,
            out_dim=cap_feats.shape[0]
        )
        
        # Concatenate raw tensors
        raw_unified = ops.concat([x, cap_feats], axis=0)
        raw_freqs = ops.concat([x_freqs_cis, cap_freqs_cis], axis=0)
        raw_adaln = ops.concat([adaln_input, adaln_input_cap], axis=0)
        
        total_x = shape_t[0]
        shape_x = ops.shape_to_tensor(x.shape)
        total_x_tokens = shape_x[0]
        
        # Calculate segment starts and lengths
        img_starts = input_row_offsets[:-1]
        img_lens = input_row_offsets[1:] - input_row_offsets[:-1]
        
        batch_range = ops.range(0, batch_size_tensor, 1, dtype=DType.int32, device=x.device, out_dim=t.shape[0])
        cap_starts = ops.cast(total_x_tokens, DType.int32) + batch_range * cap_len_tensor
        
        # Interleave starts and lens: [img0, cap0, img1, cap1, ...]
        starts = ops.reshape(ops.stack([img_starts, cap_starts], axis=1), shape=[-1])
        
        cap_lens = ops.broadcast_to(cap_len_tensor, [t.shape[0]])
        lens = ops.reshape(ops.stack([img_lens, cap_lens], axis=1), shape=[-1])
        
        # Construct gather indices using cumulative sums
        lens_cumsum = ops.cumsum(lens, axis=0)
        lens_cumsum_exclusive = ops.concat([ops.constant([0], DType.int32, device=x.device), lens_cumsum[:-1]], axis=0)
        
        total_unified_dim = x.shape[0] + cap_feats.shape[0]
        
        segment_offsets_expanded = ops.repeat_interleave(lens_cumsum_exclusive, lens, axis=0, out_dim=total_unified_dim)
        
        total_unified_tokens = lens_cumsum[-1]
        
        global_indices = ops.range(0, total_unified_tokens, 1, dtype=DType.int32, device=x.device, out_dim=total_unified_dim)
        
        local_indices = global_indices - segment_offsets_expanded
        
        segment_starts_expanded = ops.repeat_interleave(starts, lens, axis=0, out_dim=total_unified_dim)
        gather_indices = segment_starts_expanded + local_indices
        
        # Gather unified tensors
        unified = ops.gather(raw_unified, gather_indices, axis=0)
        unified_freqs_cis = ops.gather(raw_freqs, gather_indices, axis=0)
        adaln_input_unified = ops.gather(raw_adaln, gather_indices, axis=0)
        
        unified_row_offsets = ops.concat([ops.constant([0], DType.int32, device=x.device), lens_cumsum], axis=0)
        
        for layer in self.layers:
            unified = layer(unified, unified_freqs_cis, unified_row_offsets, max_seq_len, adaln_input_unified)
            
        unified = self.all_final_layer[key](unified, adaln_input_unified)
        
        return self.unpatchify(unified, t.shape[0], cap_feats.shape[0])
