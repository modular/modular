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

"""QwenImage attention layers: dual-stream attention, FeedForward, and transformer block.

Weight key naming follows HuggingFace diffusers conventions:
- Attention: attn.to_q, attn.to_k, attn.to_v, attn.to_out.0, attn.add_q_proj, etc.
- FeedForward: img_mlp.net.0.proj (SwiGLU), img_mlp.net.2 (output linear)
- Modulation: img_mod.1 (Linear after SiLU), txt_mod.1
- Norms: img_norm1, img_norm2, txt_norm1, txt_norm2 (no affine, no weights)
"""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu, rope_ragged_with_position_ids
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .normalizations import LayerNormNoAffine


def _apply_qwen_image_qk_rope(
    query: TensorValue,
    key: TensorValue,
    freqs_cis: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Apply RoPE to Q and K using precomputed interleaved [cos, sin] frequencies.

    Uses the fused ``rope_ragged_with_position_ids`` GPU kernel instead of
    graph-level complex multiplication, following the same pattern as z-image.

    Args:
        query: [B, S, H, D] query tensor.
        key: [B, S, H, D] key tensor.
        freqs_cis: [S, D] interleaved [cos, sin] frequency tensor.

    Returns:
        Tuple of (query_rotated, key_rotated), each [B, S, H, D].
    """
    batch_size = query.shape[0]
    seq_len = query.shape[1]
    num_heads = query.shape[2]
    head_dim = query.shape[3]

    query_ragged = ops.reshape(
        query, [batch_size * seq_len, num_heads, head_dim]
    )
    key_ragged = ops.reshape(key, [batch_size * seq_len, num_heads, head_dim])

    position_ids = ops.range(
        0, seq_len, dtype=DType.uint32, device=query.device
    )
    position_ids = ops.broadcast_to(
        ops.unsqueeze(position_ids, 0), [batch_size, seq_len]
    )

    query_out = rope_ragged_with_position_ids(
        query_ragged, freqs_cis, position_ids, interleaved=True
    )
    key_out = rope_ragged_with_position_ids(
        key_ragged, freqs_cis, position_ids, interleaved=True
    )
    return (
        ops.reshape(query_out, [batch_size, seq_len, num_heads, head_dim]),
        ops.reshape(key_out, [batch_size, seq_len, num_heads, head_dim]),
    )


# ---------------------------------------------------------------------------
# FeedForward (matches diffusers naming: net.0.proj, net.2)
# ---------------------------------------------------------------------------


class _QwenImageGELU(Module):
    """GELU approximate activation with a Linear projection.

    Weight key: `proj.weight`, `proj.bias`
    In the block: `img_mlp.net.0.proj.weight`
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        *,
        dtype: DType,
        device: DeviceRef,
    ):
        super().__init__()
        self.proj = Linear(
            in_dim=dim_in,
            out_dim=dim_out,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.gelu(self.proj(x))


class _QwenImageDropout(Module):
    """No-op dropout for inference. Occupies index 1 in FeedForward.net."""

    def __init__(self):
        super().__init__()

    def __call__(self, x: TensorValue) -> TensorValue:
        return x


class QwenImageFeedForward(Module):
    """FeedForward matching diffusers key naming.

    Produces keys:
        net.0.proj.weight, net.0.proj.bias  (GELU approximate projection)
        net.2.weight, net.2.bias            (output linear)
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4.0,
        inner_dim: int | None = None,
        bias: bool = True,
        *,
        dtype: DType,
        device: DeviceRef,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.net: LayerList = LayerList(
            [
                _QwenImageGELU(
                    dim, inner_dim, bias=bias, dtype=dtype, device=device
                ),
                _QwenImageDropout(),
                Linear(
                    in_dim=inner_dim,
                    out_dim=dim_out,
                    dtype=dtype,
                    device=device,
                    has_bias=bias,
                ),
            ]
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.net[0](x)  # GELU projection
        # net[1] is dropout (no-op at inference)
        x = self.net[2](x)  # output linear
        return x


# ---------------------------------------------------------------------------
# Attention (matches diffusers key naming: to_q, to_k, to_v, to_out.0, ...)
# ---------------------------------------------------------------------------


class QwenImageAttention(Module):
    """Dual-stream attention for QwenImage transformer blocks.

    Key naming matches HuggingFace diffusers:
    - to_q.weight/bias, to_k.weight/bias, to_v.weight/bias
    - to_out.0.weight/bias  (LayerList for correct .0. indexing)
    - add_q_proj.weight/bias, add_k_proj.weight/bias, add_v_proj.weight/bias
    - to_add_out.weight/bias
    - norm_q.weight, norm_k.weight, norm_added_q.weight, norm_added_k.weight
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        *,
        dtype: DType,
        device: DeviceRef,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.scale = 1.0 / (self.head_dim**0.5)
        out_dim = out_dim if out_dim is not None else query_dim

        self.to_q = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )
        self.to_k = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )
        self.to_v = Linear(
            in_dim=query_dim,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=bias,
        )

        self.norm_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
        self.norm_k = RMSNorm(dim_head, dtype=dtype, eps=eps)

        # Use LayerList so key becomes to_out.0.weight (not to_out_0.weight)
        self.to_out: LayerList = LayerList(
            [
                Linear(
                    in_dim=self.inner_dim,
                    out_dim=out_dim,
                    dtype=dtype,
                    device=device,
                    has_bias=out_bias,
                )
            ]
        )

        self.norm_added_q: RMSNorm | None
        self.norm_added_k: RMSNorm | None
        self.add_q_proj: Linear | None
        self.add_k_proj: Linear | None
        self.add_v_proj: Linear | None
        self.to_add_out: Linear | None
        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, dtype=dtype, eps=eps)
            self.add_q_proj = Linear(
                in_dim=added_kv_proj_dim,
                out_dim=self.inner_dim,
                dtype=dtype,
                device=device,
                has_bias=added_proj_bias,
            )
            self.add_k_proj = Linear(
                in_dim=added_kv_proj_dim,
                out_dim=self.inner_dim,
                dtype=dtype,
                device=device,
                has_bias=added_proj_bias,
            )
            self.add_v_proj = Linear(
                in_dim=added_kv_proj_dim,
                out_dim=self.inner_dim,
                dtype=dtype,
                device=device,
                has_bias=added_proj_bias,
            )
            self.to_add_out = Linear(
                in_dim=self.inner_dim,
                out_dim=query_dim,
                dtype=dtype,
                device=device,
                has_bias=out_bias,
            )
        else:
            self.norm_added_q = None
            self.norm_added_k = None
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None
            self.to_add_out = None

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue | None = None,
        image_rotary_emb: TensorValue | None = None,
    ) -> TensorValue | tuple[TensorValue, TensorValue]:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        seq_len = query.shape[1]

        query = ops.reshape(
            query, [batch_size, seq_len, self.heads, self.head_dim]
        )
        key = ops.reshape(key, [batch_size, seq_len, self.heads, self.head_dim])
        value = ops.reshape(
            value, [batch_size, seq_len, self.heads, self.head_dim]
        )

        query = self.norm_q(query)
        key = self.norm_k(key)

        if (
            encoder_hidden_states is not None
            and self.added_kv_proj_dim is not None
        ):
            if (
                self.add_q_proj is None
                or self.add_k_proj is None
                or self.add_v_proj is None
            ):
                raise ValueError("Encoder projections not initialized")
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)
            encoder_seq_len = encoder_query.shape[1]
            encoder_query = ops.reshape(
                encoder_query,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )
            encoder_key = ops.reshape(
                encoder_key,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )
            encoder_value = ops.reshape(
                encoder_value,
                [batch_size, encoder_seq_len, self.heads, self.head_dim],
            )

            if self.norm_added_q is None or self.norm_added_k is None:
                raise ValueError("Encoder normalizations not initialized")
            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = ops.concat([encoder_query, query], axis=1)
            key = ops.concat([encoder_key, key], axis=1)
            value = ops.concat([encoder_value, value], axis=1)

        if image_rotary_emb is not None:
            query, key = _apply_qwen_image_qk_rope(query, key, image_rotary_emb)

        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )

        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = ops.reshape(
            hidden_states, [batch_size, seq_len, self.inner_dim]
        )
        if hidden_states.dtype != query.dtype:
            hidden_states = ops.cast(hidden_states, query.dtype)

        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_out = hidden_states[:, :encoder_seq_len, :]
            hidden_out = hidden_states[:, encoder_seq_len:, :]

            hidden_out = self.to_out[0](hidden_out)
            if self.to_add_out is None:
                raise ValueError("Encoder output projection not initialized")
            encoder_out = self.to_add_out(encoder_out)

            return hidden_out, encoder_out
        else:
            hidden_states = self.to_out[0](hidden_states)
            return hidden_states


# ---------------------------------------------------------------------------
# Per-block Modulation (matches diffusers: img_mod.1.weight, txt_mod.1.weight)
# ---------------------------------------------------------------------------


class _SiLUPlaceholder(Module):
    """Placeholder at index 0 in LayerList; SiLU has no learnable params."""

    def __init__(self):
        super().__init__()

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.silu(x)


def _make_block_modulation(
    dim: int,
    bias: bool = True,
    *,
    dtype: DType,
    device: DeviceRef,
) -> LayerList:
    """Create per-block modulation as LayerList[SiLU_placeholder, Linear].

    Produces weight keys: `{attr_name}.1.weight` and `{attr_name}.1.bias`
    matching the diffusers convention img_mod.1.weight / txt_mod.1.weight.
    """
    return LayerList(
        [
            _SiLUPlaceholder(),
            Linear(
                in_dim=dim,
                out_dim=dim * 6,
                dtype=dtype,
                device=device,
                has_bias=bias,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Transformer Block (per-block img_mod, txt_mod, img_mlp, txt_mlp)
# ---------------------------------------------------------------------------


class QwenImageTransformerBlock(Module):
    """Dual-stream transformer block with per-block modulation.

    Weight key structure per block:
        img_mod.1.{weight,bias}
        txt_mod.1.{weight,bias}
        attn.to_q.{weight,bias}, attn.to_k.{weight,bias}, ...
        img_mlp.net.0.proj.{weight,bias}, img_mlp.net.2.{weight,bias}
        txt_mlp.net.0.proj.{weight,bias}, txt_mlp.net.2.{weight,bias}
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        bias: bool = True,
        *,
        dtype: DType,
        device: DeviceRef,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # Per-block modulation (img_mod, txt_mod)
        self.img_mod: LayerList = _make_block_modulation(
            dim, bias=bias, dtype=dtype, device=device
        )
        self.txt_mod: LayerList = _make_block_modulation(
            dim, bias=bias, dtype=dtype, device=device
        )

        # Norms (no affine → no weights in state_dict)
        self.img_norm1 = LayerNormNoAffine(eps=eps)
        self.img_norm2 = LayerNormNoAffine(eps=eps)
        self.txt_norm1 = LayerNormNoAffine(eps=eps)
        self.txt_norm2 = LayerNormNoAffine(eps=eps)

        # Dual-stream attention
        self.attn = QwenImageAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            dtype=dtype,
            device=device,
        )

        # Feedforward (img_mlp, txt_mlp)
        self.img_mlp = QwenImageFeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.txt_mlp = QwenImageFeedForward(
            dim=dim,
            dim_out=dim,
            mult=mlp_ratio,
            bias=bias,
            dtype=dtype,
            device=device,
        )

    def _apply_modulation(
        self,
        x: TensorValue,
        shift: TensorValue,
        scale: TensorValue,
    ) -> TensorValue:
        """Apply shift/scale modulation: (1 + scale) * x + shift."""
        return (1 + scale) * x + shift

    def _apply_split_modulation(
        self,
        x: TensorValue,
        mod_real: TensorValue,
        mod_zero: TensorValue,
        num_noise: int,
        mod_idx: int,
    ) -> TensorValue:
        """Apply different modulation to noise vs condition tokens.

        Splits x along seq dim, applies mod_real to noise tokens and
        mod_zero to condition tokens, then concatenates back.
        Avoids broadcasting [B,1,D] to [B,seq,D].
        """
        # mod has 6 chunks: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        # We need shift and scale at mod_idx and mod_idx+1
        real_chunks = ops.chunk(mod_real, 6, axis=-1)
        zero_chunks = ops.chunk(mod_zero, 6, axis=-1)
        shift_r, scale_r = real_chunks[mod_idx], real_chunks[mod_idx + 1]
        shift_z, scale_z = zero_chunks[mod_idx], zero_chunks[mod_idx + 1]

        x_noise = x[:, :num_noise, :]
        x_cond = x[:, num_noise:, :]

        x_noise = (1 + scale_r) * x_noise + shift_r
        x_cond = (1 + scale_z) * x_cond + shift_z

        return ops.concat([x_noise, x_cond], axis=1)

    def _apply_split_gate(
        self,
        x: TensorValue,
        gate_real: TensorValue,
        gate_zero: TensorValue,
        num_noise: int,
    ) -> TensorValue:
        """Apply different gate to noise vs condition tokens."""
        x_noise = x[:, :num_noise, :] * gate_real
        x_cond = x[:, num_noise:, :] * gate_zero
        return ops.concat([x_noise, x_cond], axis=1)

    def __call__(
        self,
        hidden_states: TensorValue,
        encoder_hidden_states: TensorValue,
        temb: TensorValue,
        image_rotary_emb: TensorValue | None = None,
        temb_zero: TensorValue | None = None,
        num_noise_tokens: int | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        # Compute per-block modulation params from temb
        # Compute silu once and reuse for both modulation projections.
        temb_activated = ops.silu(temb)
        img_mod = self.img_mod[1](temb_activated)
        txt_mod = self.txt_mod[1](temb_activated)

        if len(img_mod.shape) == 2:
            img_mod = ops.unsqueeze(img_mod, 1)
            txt_mod = ops.unsqueeze(txt_mod, 1)

        # zero_cond_t path: separate modulation for condition tokens
        img_mod_zero: TensorValue | None = None
        if temb_zero is not None:
            temb_zero_activated = ops.silu(temb_zero)
            img_mod_zero = self.img_mod[1](temb_zero_activated)
            if len(img_mod_zero.shape) == 2:
                img_mod_zero = ops.unsqueeze(img_mod_zero, 1)

        img_mod_chunks = ops.chunk(img_mod, 6, axis=-1)
        shift_msa, scale_msa, gate_msa = (
            img_mod_chunks[0],
            img_mod_chunks[1],
            img_mod_chunks[2],
        )
        shift_mlp, scale_mlp, gate_mlp = (
            img_mod_chunks[3],
            img_mod_chunks[4],
            img_mod_chunks[5],
        )

        txt_mod_chunks = ops.chunk(txt_mod, 6, axis=-1)
        c_shift_msa, c_scale_msa, c_gate_msa = (
            txt_mod_chunks[0],
            txt_mod_chunks[1],
            txt_mod_chunks[2],
        )
        c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            txt_mod_chunks[3],
            txt_mod_chunks[4],
            txt_mod_chunks[5],
        )

        # Image stream - Attention
        norm_hidden_states = self.img_norm1(hidden_states)
        if img_mod_zero is not None and num_noise_tokens is not None:
            norm_hidden_states = self._apply_split_modulation(
                norm_hidden_states, img_mod, img_mod_zero, num_noise_tokens, 0
            )
        else:
            norm_hidden_states = (
                1 + scale_msa
            ) * norm_hidden_states + shift_msa

        # Text stream - Attention
        norm_encoder_hidden_states = self.txt_norm1(encoder_hidden_states)
        norm_encoder_hidden_states = (
            1 + c_scale_msa
        ) * norm_encoder_hidden_states + c_shift_msa

        # Dual-stream attention
        attn_output, context_attn_output = self.attn(
            norm_hidden_states,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Image stream - Apply gate and residual
        if img_mod_zero is not None and num_noise_tokens is not None:
            img_mod_zero_chunks = ops.chunk(img_mod_zero, 6, axis=-1)
            attn_output = self._apply_split_gate(
                attn_output, gate_msa, img_mod_zero_chunks[2], num_noise_tokens
            )
        else:
            attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        # Image stream - Feedforward
        norm_hidden_states = self.img_norm2(hidden_states)
        if img_mod_zero is not None and num_noise_tokens is not None:
            norm_hidden_states = self._apply_split_modulation(
                norm_hidden_states, img_mod, img_mod_zero, num_noise_tokens, 3
            )
        else:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp) + shift_mlp
            )

        ff_output = self.img_mlp(norm_hidden_states)
        if img_mod_zero is not None and num_noise_tokens is not None:
            ff_output = self._apply_split_gate(
                ff_output, gate_mlp, img_mod_zero_chunks[5], num_noise_tokens
            )
        else:
            ff_output = gate_mlp * ff_output
        hidden_states = hidden_states + ff_output

        # Text stream - Apply gate and residual
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # Text stream - Feedforward
        norm_encoder_hidden_states = self.txt_norm2(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
        )

        context_ff_output = self.txt_mlp(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp * context_ff_output
        )

        if encoder_hidden_states.dtype == DType.float16:
            encoder_hidden_states = ops.max(encoder_hidden_states, -65504.0)
            encoder_hidden_states = ops.min(encoder_hidden_states, 65504.0)

        return encoder_hidden_states, hidden_states
