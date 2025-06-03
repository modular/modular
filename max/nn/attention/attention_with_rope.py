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
"""An opaque KV Cache optimized attention mechanism with Rope."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationConfig, QuantizationEncoding

from ..clamp import clamp
from ..comm import Allreduce
from ..kernels import (
    MHAMaskVariant,
    flare_mla_decode_ragged,
    flare_mla_decompress_k_cache,
    flare_mla_prefill_plan,
    flare_mla_prefill_ragged,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    fused_qkv_ragged_matmul_quantized,
    fused_qkv_ragged_matmul_scaled_float8,
    kv_cache_get_max_seq_len,
    matmul_k_cache_ragged,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
    rms_norm_key_cache,
    unfused_qkv_ragged_matmul_gguf_quantized,
)
from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)
from ..layer import Module
from ..linear import Float8Config, Float8ScaleGranularity, Linear
from ..norm import RMSNorm
from ..rotary_embedding import OptimizedRotaryEmbedding
from .interfaces import (
    AttentionImpl,
    AttentionImplQKV,
    DistributedAttentionImpl,
)


@dataclass
class AttentionWithRopeV1(AttentionImpl):
    """Implementation of attention that uses the rope frequency.

    Deprecated: Use `AttentionWithRope` instead.
    """

    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow

    rope: OptimizedRotaryEmbedding
    bias: Optional[TensorValue] = None
    perm_idx: Optional[TensorValue] = None
    quantization_config: Optional[QuantizationConfig] = None

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Call into fused qkv ragged matmul.
        if self.quantization_config:
            xq = fused_qkv_ragged_matmul_quantized(
                self.kv_params,
                input=x,
                wqkv=self.wqkv,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                bias=self.bias,
                perm_idx=self.perm_idx,
                quantization_config=self.quantization_config,
            )
        else:
            xq = fused_qkv_ragged_matmul(
                self.kv_params,
                input=x,
                wqkv=self.wqkv,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                bias=self.bias,
            )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if xq.device is not None:
            freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(freqs_cis, xq.dtype)

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

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.wo(attn_out)


class AttentionWithRope(Module):
    """Implementation of attention that uses the rope frequency."""

    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __init__(
        self,
        *,
        rope: OptimizedRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: list[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        float8_config: Float8Config | None = None,
        clip_qkv: float | None = None,
    ):
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            dtype: DType of the QKV and output projection weights.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `DistributedAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            stacked_qkv: Whether the weights are stacked together.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.scale = (
            scale if scale else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.clip_qkv = clip_qkv
        self.devices = devices or [DeviceRef.CPU()]
        self.float8_config = float8_config

        if stacked_qkv and clip_qkv:
            raise ValueError(
                "`clip_qkv` not yet supported when `stack_qkv=True`."
            )

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.stacked_qkv = stacked_qkv

        if stacked_qkv:
            # To keep the weight names consistent with the transformers attention,
            # the names are suffixed ".weight".
            self.qkv_proj = Weight(
                name="qkv_proj.weight",
                dtype=dtype,
                shape=[q_weight_dim + 2 * kv_weight_dim, hidden_size],
                device=self.devices[0],
            )
        else:
            self.q_proj = Weight(
                name="q_proj.weight",
                dtype=dtype,
                shape=[q_weight_dim, hidden_size],
                device=self.devices[0],
            )
            self.k_proj = Weight(
                name="k_proj.weight",
                dtype=dtype,
                shape=[kv_weight_dim, hidden_size],
                device=self.devices[0],
            )
            self.v_proj = Weight(
                name="v_proj.weight",
                dtype=dtype,
                shape=[kv_weight_dim, hidden_size],
                device=self.devices[0],
            )

        if has_bias:
            assert not stacked_qkv, "Bias is not supported with stacked qkv."

            self.bias_q = Weight(
                name="q_proj.bias",
                dtype=dtype,
                shape=[q_weight_dim],
                device=self.devices[0],
            )
            self.bias_k = Weight(
                name="k_proj.bias",
                dtype=dtype,
                shape=[kv_weight_dim],
                device=self.devices[0],
            )
            self.bias_v = Weight(
                name="v_proj.bias",
                dtype=dtype,
                shape=[kv_weight_dim],
                device=self.devices[0],
            )

        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
            float8_config=float8_config,
        )

        if float8_config:
            if (
                float8_config.input_scale.granularity
                == Float8ScaleGranularity.TENSOR
            ):
                self.input_scale_q = Weight(
                    name="q_proj.input_scale",
                    dtype=float8_config.input_scale.dtype,
                    shape=[1],
                    device=self.devices[0],
                )
                self.input_scale_k = Weight(
                    name="k_proj.input_scale",
                    dtype=float8_config.input_scale.dtype,
                    shape=[1],
                    device=self.devices[0],
                )
                self.input_scale_v = Weight(
                    name="v_proj.input_scale",
                    dtype=float8_config.input_scale.dtype,
                    shape=[1],
                    device=self.devices[0],
                )

            if (
                float8_config.weight_scale.granularity
                == Float8ScaleGranularity.TENSOR
            ):
                self.weight_scale_q = Weight(
                    name="q_proj.weight_scale",
                    dtype=float8_config.weight_scale.dtype,
                    shape=[1],
                    device=self.devices[0],
                )
                self.weight_scale_k = Weight(
                    name="k_proj.weight_scale",
                    dtype=float8_config.weight_scale.dtype,
                    shape=[1],
                    device=self.devices[0],
                )
                self.weight_scale_v = Weight(
                    name="v_proj.weight_scale",
                    dtype=float8_config.weight_scale.dtype,
                    shape=[1],
                    device=self.devices[0],
                )
            elif (
                float8_config.weight_scale.granularity
                == Float8ScaleGranularity.ROWWISE
            ):
                self.weight_scale_q = Weight(
                    name="q_proj.weight_scale",
                    dtype=float8_config.weight_scale.dtype,
                    shape=[self.q_proj.shape[0], 1],
                    device=self.devices[0],
                )
                self.weight_scale_k = Weight(
                    name="k_proj.weight_scale",
                    dtype=float8_config.weight_scale.dtype,
                    shape=[self.k_proj.shape[0], 1],
                    device=self.devices[0],
                )
                self.weight_scale_v = Weight(
                    name="v_proj.weight_scale",
                    dtype=float8_config.weight_scale.dtype,
                    shape=[self.v_proj.shape[0], 1],
                    device=self.devices[0],
                )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        if self.stacked_qkv:
            return self.qkv_proj
        else:
            wq: TensorValue = self.q_proj
            wk: TensorValue = self.k_proj
            wv: TensorValue = self.v_proj
            if self.clip_qkv:
                wq = clamp(wq, min=-self.clip_qkv, max=self.clip_qkv)
                wk = clamp(wk, min=-self.clip_qkv, max=self.clip_qkv)
                wv = clamp(wv, min=-self.clip_qkv, max=self.clip_qkv)

            # Here we are rescaling the weights to be based on the max scale.
            # This feels super fishy and like it could greatly hurt accuracy.
            # That said, for these float8 models, all models run with vllm
            # (not supported by torch/transformers). As such, vllm is the
            # canonical implementation for correctness. This rescaling is what
            # vllm does.
            # https://github.com/vllm-project/vllm/blob/9b1769dd9ad13a5688d1e2b1b5f00b07b3716969/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L35
            if (
                self.float8_config
                and self.float8_config.weight_scale.granularity
                == Float8ScaleGranularity.TENSOR
            ):
                wq = wq * self.weight_scale_q
                wk = wk * self.weight_scale_k
                wv = wv * self.weight_scale_v

            wqkv = ops.concat((wq, wk, wv))
            if self.float8_config and self.float8_config.is_static:
                # Float8 always has a weight scale.
                assert self.qkv_weight_scale is not None
                wqkv = quantize_static_scaled_float8(
                    wqkv, self.qkv_weight_scale.to(DeviceRef.CPU())
                )
            return wqkv

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        return ops.concat((self.bias_q, self.bias_k, self.bias_v))

    @property
    def qkv_input_scale(self) -> TensorValue | None:
        """The max of q, k, and v scale input vectors."""
        if not self.float8_config or self.float8_config.is_dynamic:
            return None

        return ops.max(
            ops.concat(
                (self.input_scale_q, self.input_scale_k, self.input_scale_v)
            )
        ).reshape([])

    @property
    def qkv_weight_scale(self) -> TensorValue:
        """The max of q, k, and v scale weight vectors."""
        assert self.float8_config

        weight_scale = ops.concat(
            (
                self.weight_scale_q,
                self.weight_scale_k,
                self.weight_scale_v,
            )
        )
        if self.float8_config.is_dynamic:
            # In the dynamic scaling case, return the weight scales directly.
            return weight_scale

        assert self.float8_config.is_static
        # Otherwise, return a scalar max QKV weight scale in the static case.
        return ops.max(weight_scale).reshape([])

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        if self.float8_config:
            assert isinstance(kv_collection, PagedKVCacheCollection)

            x_scales: TensorValue
            weight_scale = self.qkv_weight_scale
            if self.float8_config.is_static:
                assert self.qkv_input_scale is not None
                x = quantize_static_scaled_float8(
                    x, self.qkv_input_scale.to(DeviceRef.CPU())
                )
                x_scales = self.qkv_input_scale
            else:
                x, x_scales = quantize_dynamic_scaled_float8(
                    x, scales_type=weight_scale.dtype
                )

            xq = fused_qkv_ragged_matmul_scaled_float8(
                self.kv_params,
                input=x,
                wqkv=self.wqkv,
                bias=self.wqkv_bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                input_scale=x_scales,
                weight_scale=weight_scale,
            )
        else:
            # Call into fused qkv ragged matmul.
            xq = fused_qkv_ragged_matmul(
                self.kv_params,
                input=x,
                wqkv=self.wqkv,
                bias=self.wqkv_bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
            )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if xq.device is not None:
            freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(freqs_cis, xq.dtype)

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

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.o_proj(attn_out)


class LatentAttentionWithRope(AttentionWithRope):
    """Implementation of Latent Attention with Rope."""

    # TODO: This will be replaced with a generic Yarn Rope implementation for Deepseek-V2-lite.
    rope: OptimizedRotaryEmbedding

    def __init__(
        self,
        *,
        rope: OptimizedRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        dtype: DType,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
    ):
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the weights, should always be uint8.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `DistributedAttentionWithRope` to use all devices during
                attention computation.
            quantization_encoding: Quantization encoding of the weights.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
            buffer_size: Buffer size for storing the temporal results during prefill,
                in unit of tokens.
        """
        # Skip AttentionWithRope.__init__ because the weights are created
        # differently.
        Module.__init__(self)

        if dtype != DType.bfloat16:
            raise ValueError(
                f"Latent Attention with Rope only supports bfloat16 dtype weights but got {dtype}"
            )

        if clip_qkv is not None:
            raise ValueError(
                "clip_qkv is not supported for Latent Attention with Rope"
            )

        if has_bias:
            raise ValueError("Latent Attention with Rope does not support bias")

        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.cache_head_dim = kv_lora_rank + qk_rope_head_dim

        self.BUFFER_TOK_SIZE = buffer_size

        self.scale = scale if scale else math.sqrt(1.0 / self.qk_head_dim)
        self.scale = self.rope.compute_scale(self.scale)
        self.devices = devices or [DeviceRef.CPU()]

        if self.q_lora_rank is not None:
            self.q_a_proj = Weight(
                name="q_a_proj.weight",
                dtype=dtype,
                shape=(self.q_lora_rank, self.hidden_size),
                device=self.devices[0],
            )
            self.q_a_layernorm = RMSNorm(
                dim=self.q_lora_rank, dtype=dtype, eps=1e-6
            )
            self.q_b_proj = Weight(
                name="q_b_proj.weight",
                dtype=dtype,
                shape=(self.n_heads * self.qk_head_dim, self.q_lora_rank),
                device=self.devices[0],
            )
        else:
            self.q_proj = Weight(
                name="q_proj.weight",
                dtype=dtype,
                shape=(self.n_heads * self.qk_head_dim, self.hidden_size),
                device=self.devices[0],
            )

        self.kv_a_proj_layernorm = Weight(
            name="kv_a_layernorm.weight",
            dtype=dtype,
            shape=(self.kv_lora_rank,),
            device=self.devices[0],
        )
        self.kv_a_proj_with_mqa = Weight(
            name="kv_a_proj_with_mqa.weight",
            dtype=dtype,
            shape=(self.cache_head_dim, self.hidden_size),
            device=self.devices[0],
        )
        self.kv_b_proj = Weight(
            name="kv_b_proj.weight",
            dtype=dtype,
            shape=(
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
            ),
            device=self.devices[0],
        )
        self.o_proj = linear_cls(
            in_dim=self.n_heads * self.v_head_dim,
            out_dim=self.hidden_size,
            dtype=dtype,
            device=self.devices[0],
        )

    @property
    def w_uk_uv(self) -> list[TensorValue]:
        """The concatenation of q, k, and v weight vectors."""
        kv_b_proj_weight: TensorValue = self.kv_b_proj.transpose(0, 1)

        kv_b_proj_weight = kv_b_proj_weight.reshape(
            (
                self.kv_lora_rank,
                self.n_heads,
                (self.qk_nope_head_dim + self.v_head_dim),
            )
        )

        w_uk, w_uv = ops.split(
            kv_b_proj_weight, [self.qk_nope_head_dim, self.v_head_dim], axis=2
        )

        w_uv = w_uv.transpose(0, 1)

        w_uk_t = w_uk.permute([1, 2, 0])

        return [w_uk_t, w_uv]

    @property
    def wqkv(self) -> TensorValue:
        raise NotImplementedError(
            "wqkv is not implemented for LatentAttentionWithRope"
        )

    @property
    def wqkv_bias(self) -> TensorValue | None:
        raise NotImplementedError(
            "wqkv_bias is not implemented for LatentAttentionWithRope"
        )

    def _mla_impl(
        self,
        xq_nope: TensorValue,
        xq_rope: TensorValue,
        kv_collection: PagedKVCacheCollection,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        def _mla_prefill() -> TensorValue:
            xq = ops.concat([xq_nope, xq_rope], axis=2)

            (buffer_row_offsets, cache_offsets, buffer_lengths) = (
                flare_mla_prefill_plan(
                    self.kv_params,
                    input_row_offsets,
                    kv_collection,
                    layer_idx,
                    self.BUFFER_TOK_SIZE,
                )
            )
            buffer_lengths_host = buffer_lengths.to(DeviceRef.CPU())

            kv_buffer = flare_mla_decompress_k_cache(
                self.kv_params,
                buffer_row_offsets[0],
                cache_offsets[0],
                buffer_lengths_host[0],
                self.kv_b_proj,
                kv_collection,
                layer_idx,
                self.BUFFER_TOK_SIZE,
            )

            kv_buffer = kv_buffer.reshape(
                (-1, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            )
            k_nope, v = ops.split(
                kv_buffer, [self.qk_nope_head_dim, self.v_head_dim], axis=2
            )

            result, softmax_info = flare_mla_prefill_ragged(
                self.kv_params,
                xq,
                k_nope,
                v,
                input_row_offsets,
                buffer_row_offsets[0],
                cache_offsets[0],
                kv_collection,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                self.scale,
                self.qk_rope_head_dim,
            )

            iter_i = ops.constant(1, DType.int64, device=DeviceRef.CPU())

            def cond_fn(iter_i, prev_result, prev_softmax_info):
                return buffer_lengths_host[iter_i] > 0

            def body_fn(iter_i, prev_result, prev_softmax_info):
                kv_buffer = flare_mla_decompress_k_cache(
                    self.kv_params,
                    buffer_row_offsets[iter_i],
                    cache_offsets[iter_i],
                    buffer_lengths_host[iter_i],
                    self.kv_b_proj,
                    kv_collection,
                    layer_idx,
                    self.BUFFER_TOK_SIZE,
                )

                kv_buffer = kv_buffer.reshape(
                    (-1, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
                )
                k_nope, v = ops.split(
                    kv_buffer, [self.qk_nope_head_dim, self.v_head_dim], axis=2
                )

                new_result, new_softmax_info = flare_mla_prefill_ragged(
                    self.kv_params,
                    xq,
                    k_nope,
                    v,
                    input_row_offsets,
                    buffer_row_offsets[iter_i],
                    cache_offsets[iter_i],
                    kv_collection,
                    layer_idx,
                    MHAMaskVariant.CAUSAL_MASK,
                    self.scale,
                    self.qk_rope_head_dim,
                    prev_output=prev_result,
                    prev_softmax_info=prev_softmax_info,
                )

                iter_i = iter_i + 1

                return [iter_i, new_result, new_softmax_info]

            loop_result = ops.while_loop(
                (iter_i, result, softmax_info),
                cond_fn,
                body_fn,
            )

            return loop_result[1]

        def _mla_decode() -> TensorValue:
            w_uk, w_uv = self.w_uk_uv
            # from [B, H, D] to [H, B, D]
            xq_nope_t = xq_nope.transpose(0, 1)

            # batched matmul
            xq_nope_proj = xq_nope_t @ w_uk
            xq_nope_proj = xq_nope_proj.transpose(0, 1)
            xq = ops.concat([xq_nope_proj, xq_rope], axis=2)

            # Calculate Flash Attention.
            attn_out = flare_mla_decode_ragged(
                self.kv_params,
                input=xq,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                input_row_offsets=input_row_offsets,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=self.scale,
            )

            # from [B, H, D] to [H, B, D]
            attn_out_latent = attn_out.transpose(0, 1)

            # batched matmul
            attn_out = attn_out_latent @ w_uv
            return attn_out.transpose(0, 1)

        # TODO: use max_lengths[0, 0] cause a CUDA_INVALID_MEMORY_ACCESS error,
        # as the graph compiler assumes it is a GPU tensor, and inserts a DtoH copy.
        max_seq_len = kv_cache_get_max_seq_len(kv_collection)

        result = ops.cond(
            max_seq_len > 1,
            [
                TensorType(
                    dtype=xq_nope.dtype,
                    shape=[
                        xq_nope.shape[0],
                        self.n_heads,
                        self.v_head_dim,
                    ],
                    device=xq_nope.device,
                )
            ],
            _mla_prefill,
            _mla_decode,
        )[0].tensor

        result = ops.reshape(result, shape=[result.shape[0], -1])

        return result

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: Union[
            PagedKVCacheCollection, ContinuousBatchingKVCacheCollection
        ],
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        assert isinstance(kv_collection, PagedKVCacheCollection)

        # Get attributes from input.
        total_seq_len = x.shape[0]

        if self.q_lora_rank is not None:
            xq = self.q_a_layernorm(x @ self.q_a_proj.T) @ self.q_b_proj.T
        else:
            xq = x @ self.q_proj.T

        matmul_k_cache_ragged(
            self.kv_params,
            hidden_states=x,
            weight=self.kv_a_proj_with_mqa,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
        )

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.kv_a_proj_layernorm,
            epsilon=1e-6,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            rms_norm_cols=self.kv_lora_rank,
            weight_offset=0.0,
        )

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

        xq_nope, xq_rope = ops.split(
            xq, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=2
        )

        # Apply rope.
        if xq.device is not None:
            freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(freqs_cis, xq.dtype)

        xq_rope = fused_qk_ragged_rope(
            self.kv_params,
            xq_rope,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=True,
        )

        attn_out = self._mla_impl(
            xq_nope,
            xq_rope,
            kv_collection,
            layer_idx,
            input_row_offsets,
        )

        return self.o_proj(attn_out)


class GGUFQAttentionWithRope(AttentionWithRope):
    """Implementation of attention with GGUF quantized weights."""

    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __init__(
        self,
        *,
        rope: OptimizedRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        dtype: DType,
        quantization_encoding: QuantizationEncoding,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
    ):
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the weights, should always be uint8.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `DistributedAttentionWithRope` to use all devices during
                attention computation.
            quantization_encoding: Quantization encoding of the weights.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
        """
        # Skip AttentionWithRope.__init__ because the weights are created
        # differently.
        Module.__init__(self)

        if dtype != DType.uint8:
            raise ValueError(
                f"GGUFQAttentionWithRope only supports uint8 dtype weights but got {dtype}"
            )

        if clip_qkv is not None:
            raise ValueError(
                "clip_qkv is not supported for GGUFQAttentionWithRope"
            )

        if has_bias:
            raise ValueError("GGUFQAttentionWithRope does not support bias")

        if not quantization_encoding.is_gguf:
            raise ValueError(
                f"Only GGUF quantization encoding is supported for GGUFQAttentionWithRope. Found: {quantization_encoding}"
            )

        if not kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        self.quantization_encoding = quantization_encoding
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.scale = (
            scale if scale else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.devices = devices or [DeviceRef.CPU()]

        self.q_proj = Weight(
            name="q_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )

        self.k_proj = Weight(
            name="k_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )
        self.v_proj = Weight(
            name="v_proj.weight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=quantization_encoding,
            device=self.devices[0],
        )

        self.o_proj = linear_cls(
            in_dim=1,  # Shape will be overridden at load_state_dict.
            out_dim=1,  # Shape will be overridden at load_state_dict.
            dtype=DType.uint8,
            quantization_encoding=quantization_encoding,  # Shape will be overridden at load_state_dict.
            device=self.devices[0],
        )

    @property
    def wqkv(self) -> TensorValue:
        raise NotImplementedError(
            "wqkv is not implemented for unfused GGUFQAttentionWithRope"
        )

    @property
    def wqkv_bias(self) -> TensorValue | None:
        raise NotImplementedError(
            "wqkv_bias is not implemented for unfused GGUFQAttentionWithRope"
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        assert self.q_proj.quantization_encoding is not None
        assert self.k_proj.quantization_encoding is not None
        assert self.v_proj.quantization_encoding is not None

        # Call into unfused qkv ragged matmul.
        xq = unfused_qkv_ragged_matmul_gguf_quantized(
            self.kv_params,
            input=x,
            input_row_offsets=input_row_offsets,
            n_heads=self.n_heads,
            q_weight=self.q_proj,
            k_weight=self.k_proj,
            v_weight=self.v_proj,
            quantization_encoding_q=self.q_proj.quantization_encoding,
            quantization_encoding_k=self.k_proj.quantization_encoding,
            quantization_encoding_v=self.v_proj.quantization_encoding,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))
        freqs_cis = ops.cast(freqs_cis, xq.dtype)

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

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)


class GPTQAttentionWithRope(AttentionWithRope):
    """Implementation of the GPT-Q attention layer."""

    def __init__(
        self,
        quantization_config: QuantizationConfig,
        rope: OptimizedRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: list[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        scale: float | None = None,
        linear_cls: Callable[..., Linear] = Linear,
    ):
        # Skip AttentionWithRope.__init__ because the weights are created
        # differently.
        Module.__init__(self)
        self.quantization_config = quantization_config
        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.hidden_size = hidden_size
        self.devices = devices or [DeviceRef.CPU()]
        self.scale = (
            scale if scale else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        self.kv_weight_dim = (
            hidden_size // num_attention_heads
        ) * num_key_value_heads

        self.q_proj_qweight = Weight(
            name="q_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.k_proj_qweight = Weight(
            name="k_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.v_proj_qweight = Weight(
            name="v_proj.qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )

        self.q_proj_scales = Weight(
            name="q_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.k_proj_scales = Weight(
            name="k_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.v_proj_scales = Weight(
            name="v_proj.scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )
        self.o_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            quantization_encoding=QuantizationEncoding.GPTQ,
            device=self.devices[0],
        )

        self.perm_idx = None
        if quantization_config.desc_act:
            self.perm_idx = Weight(
                name="q_proj.perm_idx",
                dtype=DType.int32,
                shape=[hidden_size],
                device=self.devices[0],
            )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""

        # fmt: off
        # the `qweight` tensor for a QuantLinear is of type uint32. When allocated as bytes, we reshape the
        # uint8 tensor to [cols, rows * 4] so concatenating the uint8 tensors along axis=1 is equivalent to
        # concatenating the original uint32 tensors along axis=1.
        wq_qweight = ops.reshape(self.q_proj_qweight, (-1, self.hidden_size * 4))
        wk_qweight = ops.reshape(self.k_proj_qweight, (-1, self.kv_weight_dim * 4))
        wv_qweight = ops.reshape(self.v_proj_qweight, (-1, self.kv_weight_dim * 4))

        wqkv_qweight = ops.reshape(
            ops.concat((wq_qweight, wk_qweight, wv_qweight), axis=1),
            (-1, self.hidden_size + 2 * self.kv_weight_dim),
        )
        # `scales` tensor is in f16/bf16 type, so we reshape the uint8 tensor to [cols, rows * 2].
        wq_scales = ops.reshape(self.q_proj_scales, (-1, self.hidden_size * 2))
        wk_scales = ops.reshape(self.k_proj_scales, (-1, self.kv_weight_dim * 2))
        wv_scales = ops.reshape(self.v_proj_scales, (-1, self.kv_weight_dim * 2))

        wqkv_scales = ops.reshape(
            ops.concat((wq_scales, wk_scales, wv_scales), axis=1),
            (-1, self.hidden_size + 2 * self.kv_weight_dim),
        )
        # fmt: on
        return ops.concat((wqkv_qweight, wqkv_scales), axis=0)

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        wqkv = self.wqkv
        if self.devices:
            wqkv = wqkv.to(self.devices[0])

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul_quantized(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            perm_idx=self.perm_idx,
            quantization_config=self.quantization_config,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        if xq.device is not None:
            freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(freqs_cis, xq.dtype)

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
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.o_proj(attn_out)


def distribute_value(
    v: TensorValue, devices: list[DeviceRef]
) -> list[TensorValue]:
    return [v.to(device) for device in devices]


class DistributedAttentionWithRope(AttentionWithRope, DistributedAttentionImpl):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.devices or len(self.devices) < 2:
            raise ValueError(
                f"Must provide at least 2 devices to `DistributedAttentionWithRope`, got {self.devices}"
            )
        # Shard weights into separate AttentionWithRope layers.
        num_devices = len(self.devices)
        self.allreduce = Allreduce(num_devices)

        if self.stacked_qkv:
            self.qkv_proj.set_sharding_strategy(
                ShardingStrategy.columnwise(num_devices)
            )
        else:
            self.q_proj.set_sharding_strategy(
                ShardingStrategy.rowwise(num_devices)
            )
            self.k_proj.set_sharding_strategy(
                ShardingStrategy.rowwise(num_devices)
            )
            self.v_proj.set_sharding_strategy(
                ShardingStrategy.rowwise(num_devices)
            )
        self.o_proj.set_sharding(ShardingStrategy.columnwise(num_devices))

        self.list_of_attentions = []
        kwargs = kwargs.copy()
        kwargs["num_attention_heads"] //= len(self.devices)
        for n, device in enumerate(self.devices):
            kwargs["devices"] = [device]
            layer = AttentionWithRope(**kwargs)
            if self.stacked_qkv:
                layer.qkv_proj = self.qkv_proj.shard(n, device)
            else:
                layer.q_proj = self.q_proj.shard(n, device)
                layer.k_proj = self.k_proj.shard(n, device)
                layer.v_proj = self.v_proj.shard(n, device)
            layer.o_proj.weight = self.o_proj.weight.shard(n, device)
            self.list_of_attentions.append(layer)

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        x: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[
            ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
        ],
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> list[TensorValue]:
        assert isinstance(input_row_offsets, TensorValue)
        assert self.devices
        input_row_offsets_ = distribute_value(input_row_offsets, self.devices)
        return self.allreduce(
            inputs=[
                self.list_of_attentions[i](
                    layer_idx,
                    x[i],
                    kv_collections[i],
                    freqs_cis,
                    input_row_offsets_[i],
                )
                for i in range(len(self.devices))
            ],
            signal_buffers=signal_buffers,
        )


@dataclass
class AttentionWithRopeQKV(AttentionImplQKV):
    # This class will not use the RotaryEmbedding to
    # calculate rope, but it already includes a freqs_cis
    # calculation, which we will borrow
    rope: OptimizedRotaryEmbedding

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: Union[
            ContinuousBatchingKVCacheCollection, PagedKVCacheCollection
        ],
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        wqkv = ops.concat((self.wq, self.wk, self.wv))

        # Call into fused qkv ragged matmul.
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Cast freqs_cis to xq's dtype to match the fused_qk_ragged_rope kernel.
        freqs_cis = ops.cast(freqs_cis, xq.dtype)

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

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.wo(attn_out)
