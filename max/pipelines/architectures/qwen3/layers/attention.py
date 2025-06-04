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

"""Qwen3 Attention Layer."""

from __future__ import annotations

import math
from typing import Callable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn import RMSNorm
from max.nn.kernels import (
    MHAMaskVariant,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.nn.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.rotary_embedding import OptimizedRotaryEmbedding


class Qwen3Attention(Module):
    """Implementation of the attention layer for the Qwen3 text model.

    Specifically: applies RMSNorm to the query and key states before applying
    rotary embedding, removes window attention, and has a norm weight offset of 0.0
    (as opposed to 1.0 in Gemma3 implementation).
    """

    def __init__(
        self,
        *,
        rope: OptimizedRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
    ):
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freq_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the attention inputs and weights.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `DistributedAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias. Defaults to False.
            qk_norm_eps: Value to use for numerical stability. Defaults to 1e-6.
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.qk_norm_eps = qk_norm_eps

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        self.q_norm = RMSNorm(
            self.kv_params.head_dim, dtype=dtype, eps=self.qk_norm_eps
        )
        self.k_norm = RMSNorm(
            self.kv_params.head_dim, dtype=dtype, eps=self.qk_norm_eps
        )
        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = Weight(
            name="q_proj.weight",
            dtype=dtype,
            shape=[self.q_weight_dim, hidden_size],
            device=devices[0],
        )
        self.k_proj = Weight(
            name="k_proj.weight",
            dtype=dtype,
            shape=[self.kv_weight_dim, hidden_size],
            device=devices[0],
        )
        self.v_proj = Weight(
            name="v_proj.weight",
            dtype=dtype,
            shape=[self.kv_weight_dim, hidden_size],
            device=devices[0],
        )

        if has_bias:
            self.bias_q = Weight(
                name="q_proj.bias",
                dtype=dtype,
                shape=[self.q_weight_dim],
                device=devices[0],
            )
            self.bias_k = Weight(
                name="k_proj.bias",
                dtype=dtype,
                shape=[self.kv_weight_dim],
                device=devices[0],
            )
            self.bias_v = Weight(
                name="v_proj.bias",
                dtype=dtype,
                shape=[self.kv_weight_dim],
                device=devices[0],
            )

        self.o_proj = linear_cls(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        wq: TensorValue = self.q_proj
        wk: TensorValue = self.k_proj
        wv: TensorValue = self.v_proj
        return ops.concat((wq, wk, wv)).to(self.devices[0])

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        return ops.concat((self.bias_q, self.bias_k, self.bias_v))

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # Call into fused qkv ragged matmul.
        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Apply QK norm to query and key states before applying rope per Qwen3 arch.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        xq = self.q_norm(xq)
        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
        )

        # Apply rotary embedding.
        if xq.device is not None:
            freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(self.rope.freqs_cis, xq.dtype)
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
        # NOTE: Qwen3 never uses sliding window pattern
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
        ret = self.o_proj(attn_out)
        return ret
