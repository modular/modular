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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .embeddings import apply_rotary_emb
from .normalization import WanLayerNorm


class WanSelfAttention(Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = dim

        self.to_q = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True
        )
        self.to_k = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True
        )
        self.to_v = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True
        )
        self.norm_q = RMSNorm(dim, dtype=dtype, eps=eps)
        self.norm_k = RMSNorm(dim, dtype=dtype, eps=eps)
        self.to_out = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        rotary_emb: tuple[TensorValue, TensorValue],
    ) -> TensorValue:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        batch_size = query.shape[0]
        seq_len = query.shape[1]
        query = ops.reshape(
            query, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        key = ops.reshape(
            key, [batch_size, seq_len, self.num_heads, self.head_dim]
        )
        value = ops.reshape(
            value, [batch_size, seq_len, self.num_heads, self.head_dim]
        )

        query = apply_rotary_emb(
            query,
            rotary_emb,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        key = apply_rotary_emb(
            key,
            rotary_emb,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )

        scale = 1.0 / (self.head_dim**0.5)
        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )
        hidden_states = ops.reshape(
            hidden_states,
            [hidden_states.shape[0], hidden_states.shape[1], self.inner_dim],
        )
        return self.to_out(hidden_states)


class WanCrossAttention(Module):
    def __init__(
        self,
        dim: int,
        text_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float,
        *,
        added_kv_proj_dim: int | None = None,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = dim
        self._has_added_kv = added_kv_proj_dim is not None

        self.to_q = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True
        )
        self.to_kv = Linear(
            in_dim=text_dim,
            out_dim=dim * 2,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.norm_q = RMSNorm(dim, dtype=dtype, eps=eps)
        self.norm_k = RMSNorm(dim, dtype=dtype, eps=eps)
        self.to_out = Linear(
            in_dim=dim, out_dim=dim, dtype=dtype, device=device, has_bias=True
        )

        if added_kv_proj_dim is not None:
            self.add_k_proj = Linear(
                in_dim=added_kv_proj_dim,
                out_dim=dim,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
            self.add_v_proj = Linear(
                in_dim=added_kv_proj_dim,
                out_dim=dim,
                dtype=dtype,
                device=device,
                has_bias=True,
            )
            self.norm_added_k = RMSNorm(dim, dtype=dtype, eps=eps)

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        image_embeds: TensorValue | None = None,
    ) -> TensorValue:
        query = self.to_q(hidden_states)

        kv = self.to_kv(encoder_hidden_states)
        key = kv[:, :, : self.inner_dim]
        value = kv[:, :, self.inner_dim :]

        query = self.norm_q(query)
        key = self.norm_k(key)

        batch_size = query.shape[0]
        q_seq_len = query.shape[1]
        kv_seq_len = key.shape[1]
        query = ops.reshape(
            query, [batch_size, q_seq_len, self.num_heads, self.head_dim]
        )
        key = ops.reshape(
            key, [batch_size, kv_seq_len, self.num_heads, self.head_dim]
        )
        value = ops.reshape(
            value, [batch_size, kv_seq_len, self.num_heads, self.head_dim]
        )

        scale = 1.0 / (self.head_dim**0.5)
        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=scale,
        )

        if self._has_added_kv and image_embeds is not None:
            added_key = self.norm_added_k(self.add_k_proj(image_embeds))
            added_value = self.add_v_proj(image_embeds)
            img_kv_len = added_key.shape[1]
            added_key = ops.reshape(
                added_key,
                [batch_size, img_kv_len, self.num_heads, self.head_dim],
            )
            added_value = ops.reshape(
                added_value,
                [batch_size, img_kv_len, self.num_heads, self.head_dim],
            )
            hidden_states_img = flash_attention_gpu(
                query,
                added_key,
                added_value,
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=scale,
            )
            hidden_states = hidden_states + hidden_states_img

        hidden_states = ops.reshape(
            hidden_states,
            [hidden_states.shape[0], hidden_states.shape[1], self.inner_dim],
        )
        return self.to_out(hidden_states)


class WanFeedForward(Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.proj = Linear(
            in_dim=dim,
            out_dim=ffn_dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )
        self.linear_out = Linear(
            in_dim=ffn_dim,
            out_dim=dim,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        hidden = self.proj(x)
        hidden = ops.gelu(hidden, approximate="tanh")
        return self.linear_out(hidden)


class WanTransformerBlock(Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        text_dim: int,
        cross_attn_norm: bool,
        eps: float,
        *,
        added_kv_proj_dim: int | None = None,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.CPU(),
    ) -> None:
        super().__init__()
        self.scale_shift_table = Weight(
            "scale_shift_table", dtype, [1, 6, dim], device
        )
        self.norm1 = WanLayerNorm(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.attn1 = WanSelfAttention(
            dim, num_heads, head_dim, eps, dtype=dtype, device=device
        )
        self.norm2 = WanLayerNorm(
            dim,
            eps=eps,
            elementwise_affine=cross_attn_norm,
            use_bias=cross_attn_norm,
            dtype=dtype,
            device=device,
        )
        self.attn2 = WanCrossAttention(
            dim,
            text_dim,
            num_heads,
            head_dim,
            eps,
            added_kv_proj_dim=added_kv_proj_dim,
            dtype=dtype,
            device=device,
        )
        self.norm3 = WanLayerNorm(
            dim,
            eps=eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.ffn = WanFeedForward(dim, ffn_dim, dtype=dtype, device=device)

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        timestep_proj: TensorValue,
        rope_cos: TensorValue,
        rope_sin: TensorValue,
        image_embeds: TensorValue | None = None,
    ) -> TensorValue:
        rotary_emb = (rope_cos, rope_sin)
        mod = self.scale_shift_table + timestep_proj

        shift_sa = mod[:, 0:1, :]
        scale_sa = mod[:, 1:2, :]
        gate_sa = mod[:, 2:3, :]
        shift_ff = mod[:, 3:4, :]
        scale_ff = mod[:, 4:5, :]
        gate_ff = mod[:, 5:6, :]

        x = self.norm1(hidden_states)
        x = x * (1 + scale_sa) + shift_sa
        x = self.attn1(x, rotary_emb)
        hidden_states = hidden_states + gate_sa * x

        x = self.norm2(hidden_states)
        x = self.attn2(x, encoder_hidden_states, image_embeds=image_embeds)
        hidden_states = hidden_states + x

        x = self.norm3(hidden_states)
        x = x * (1 + scale_ff) + shift_ff
        x = self.ffn(x)
        hidden_states = hidden_states + gate_ff * x
        return hidden_states
