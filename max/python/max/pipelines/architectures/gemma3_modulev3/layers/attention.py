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

"""Gemma3 Attention Layer for the ModuleV3 API."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues

from .rms_norm import Gemma3RMSNorm
from .rotary_embedding import Llama3RotaryEmbedding


class Gemma3Attention(Module[..., Tensor]):
    """Gemma3 attention with QK normalization and sliding window support."""

    def __init__(
        self,
        *,
        rope_global: Llama3RotaryEmbedding,
        rope_local: Llama3RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        sliding_window_pattern: int = 6,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
    ) -> None:
        super().__init__()
        self.rope_global = rope_global
        self.rope_local = rope_local
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.has_bias = has_bias
        self.scale = (
            scale if scale is not None else math.sqrt(1.0 / kv_params.head_dim)
        )
        self.local_window_size = local_window_size
        self.sliding_window_pattern = sliding_window_pattern
        self.qk_norm_eps = qk_norm_eps
        self.kv_params = kv_params

        self.q_norm = Gemma3RMSNorm(kv_params.head_dim, eps=qk_norm_eps)
        self.k_norm = Gemma3RMSNorm(kv_params.head_dim, eps=qk_norm_eps)

        q_weight_dim = kv_params.head_dim * num_attention_heads
        kv_weight_dim = kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_weight_dim
        self.kv_weight_dim = kv_weight_dim

        self.q_proj = Linear(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            bias=has_bias,
        )
        self.k_proj = Linear(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            bias=has_bias,
        )
        self.v_proj = Linear(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            bias=has_bias,
        )
        self.o_proj = Linear(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            bias=False,
        )

    @property
    def wqkv(self) -> Tensor:
        """The concatenation of q, k, and v weight vectors."""
        wq: Tensor = self.q_proj.weight
        wk: Tensor = self.k_proj.weight
        wv: Tensor = self.v_proj.weight
        return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        total_seq_len = x.shape[0]

        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=self.wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Apply QK norm to query states (Gemma3-style with weight_offset=1).
        xq = self.q_norm(xq)

        # Apply QK norm to key states in-place inside the KV cache.
        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(x.device),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=kwargs["input_row_offsets"],
            weight_offset=1.0,
        )

        # Select rope: local (sliding window) or global.
        use_local = bool((self.layer_idx + 1) % self.sliding_window_pattern)
        rope = self.rope_local if use_local else self.rope_global

        freqs_cis = F.cast(rope.freqs_cis, xq.dtype).to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=rope.interleaved,
        )

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if use_local
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
