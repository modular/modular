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

from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn.float8_config import Float8Config

from ..attention.attention_with_rope import AttentionWithRope
from ..attention.mask_config import MHAMaskVariant
from ..kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    fused_qkv_ragged_matmul_scaled_float8,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
)
from ..kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from ..linear import Linear
from ..rotary_embedding import RotaryEmbedding
from .linear_lora import LinearLoRA, QKVLinearLoRA


class AttentionWithRopeAndLoRA(AttentionWithRope):
    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: RotaryEmbedding

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        max_lora_rank: int,
        max_num_loras: int,
        devices: list[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        float8_config: Float8Config | None = None,
        clip_qkv: float | None = None,
    ):
        """Initializes the LoRA-enabled attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            dtype: DType of the QKV and output projection weights.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `TensorParallelAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            stacked_qkv: Whether the weights are stacked together.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
        """
        if stacked_qkv:
            raise NotImplementedError("LoRA doesn't support stacked QKV.")

        super().__init__(
            rope=rope,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            kv_params=kv_params,
            devices=devices,
            dtype=dtype,
            linear_cls=linear_cls,
            stacked_qkv=stacked_qkv,
            scale=scale,
            has_bias=has_bias,
            float8_config=float8_config,
            clip_qkv=clip_qkv,
        )

        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.qkv_lora = QKVLinearLoRA(
            in_dim=hidden_size,
            q_dim=self.q_weight_dim,
            kv_dim=self.kv_weight_dim,
            max_lora_rank=max_lora_rank,
            max_num_loras=max_num_loras,
            dtype=dtype if not dtype.is_float8() else DType.bfloat16,
            device=self.devices[0],
        )

        self.o_proj_lora = LinearLoRA(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            max_lora_rank=max_lora_rank,
            max_num_loras=max_num_loras,
            dtype=dtype if not dtype.is_float8() else DType.bfloat16,
            device=self.devices[0],
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Check that LoRA batch info has been set
        if self.qkv_lora.num_active_loras is None:
            raise ValueError(
                "'set_lora_batch_info' not called before executing forward pass."
            )

        wqkv = self.wqkv.to(x.device)
        wqkv_bias = (
            self.wqkv_bias.to(x.device) if self.wqkv_bias is not None else None
        )

        if self.float8_config:
            # FP8 path
            weight_scale = self.qkv_weight_scale
            if self.float8_config.is_static:
                assert self.qkv_input_scale is not None
                x8 = quantize_static_scaled_float8(
                    x, self.qkv_input_scale.to(DeviceRef.CPU())
                )
                x_scales = self.qkv_input_scale
            else:
                x8, x_scales = quantize_dynamic_scaled_float8(
                    x,
                    self.float8_config.input_scale,
                    self.float8_config.weight_scale,
                    scales_type=weight_scale.dtype,
                )

            xq_matmul = fused_qkv_ragged_matmul_scaled_float8(
                self.kv_params,
                input=x8,
                wqkv=wqkv,
                bias=wqkv_bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale.to(x.device),
            )
        else:
            xq_matmul = fused_qkv_ragged_matmul(
                self.kv_params,
                input=x,
                wqkv=wqkv,
                bias=wqkv_bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
            )
        freqs_cis = ops.cast(freqs_cis, xq_matmul.dtype).to(xq_matmul.device)

        def then_fn() -> TensorValue:
            xq = xq_matmul.reshape((-1, self.n_heads, self.kv_params.head_dim))

            xq = fused_qk_ragged_rope(
                self.kv_params,
                xq,
                input_row_offsets,
                kv_collection,
                freqs_cis,
                layer_idx,
                interleaved=self.rope.interleaved,
            )
            # Calculate Flash Attention.
            attn_out = flash_attention_ragged(
                self.kv_params,
                input=xq,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                input_row_offsets=input_row_offsets,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=self.scale,
            )

            attn_out = ops.reshape(
                attn_out, shape=[total_seq_len, self.q_weight_dim]
            )

            out = self.o_proj(attn_out)

            return out

        def else_fn() -> TensorValue:
            xq = self.qkv_lora(
                x.cast(xq_matmul.dtype),
                xq_matmul,
                kv_collection,
                self.kv_params,
                input_row_offsets,
                layer_idx,
                self.rope.max_seq_len,
            )
            xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

            xq = fused_qk_ragged_rope(
                self.kv_params,
                xq,
                input_row_offsets,
                kv_collection,
                freqs_cis,
                layer_idx,
                interleaved=self.rope.interleaved,
            )

            # Calculate Flash Attention.
            attn_out = flash_attention_ragged(
                self.kv_params,
                input=xq,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                input_row_offsets=input_row_offsets,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=self.scale,
            )

            attn_out = ops.reshape(
                attn_out, shape=[total_seq_len, self.q_weight_dim]
            )

            out = self.o_proj(attn_out)
            return self.o_proj_lora(attn_out, out)

        return ops.cond(
            self.qkv_lora.num_active_loras.tensor[0] == 0,
            [
                TensorType(
                    dtype=self.o_proj_lora.lora_A.dtype,
                    shape=[total_seq_len, self.q_weight_dim],
                    device=self.o_proj.device,
                )
            ],
            then_fn,
            else_fn,
        )[0].tensor
