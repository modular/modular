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

"""Attention for Flint.

Full attention layers run the sliding kernel with an identity rotation, so
both layer types share one fused store path.
"""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
)
from max.experimental.nn.norm.rms_norm import rms_norm
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams


class FlintAttention(Module[..., Tensor]):
    """An attention layer with per-head sinks and rope on sliding layers."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        is_sliding: bool,
        sliding_window: int,
        rms_norm_eps: float = 1e-6,
        has_bias: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window
        self.rms_norm_eps = rms_norm_eps
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        if is_sliding:
            self.mask_variant = MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            self.local_window_size = sliding_window
        else:
            self.mask_variant = MHAMaskVariant.CAUSAL_MASK
            self.local_window_size = -1

        # The kernel requires the sink dtype to match q. HF keeps the
        # sinks in fp32; bf16 stays inside the tf32 tolerance envelope.
        self.sinks = Tensor.zeros([num_attention_heads])

        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = Linear(
            in_dim=hidden_size, out_dim=self.q_weight_dim, bias=self.has_bias
        )
        self.k_proj = Linear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=self.has_bias
        )
        self.v_proj = Linear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=self.has_bias
        )
        self.o_proj = Linear(
            in_dim=self.q_weight_dim, out_dim=hidden_size, bias=self.has_bias
        )

        # QK-RMSNorm on head_dim, held as raw Tensor leaves.
        self.q_norm_weight = Tensor.ones([self.kv_params.head_dim])
        self.k_norm_weight = Tensor.ones([self.kv_params.head_dim])

    @property
    def wqkv(self) -> Tensor:
        """The concatenated q, k, and v projection weights."""
        return F.concat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            axis=0,
        )

    @property
    def wqkv_bias(self) -> Tensor | None:
        if not self.has_bias:
            return None
        if (
            self.q_proj.bias is None
            or self.k_proj.bias is None
            or self.v_proj.bias is None
        ):
            raise ValueError(
                "Projection bias is None, but has_bias=True was specified."
            )
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

        qkv = x @ self.wqkv.T
        if self.wqkv_bias is not None:
            qkv = qkv + self.wqkv_bias

        # QK-norm runs before rope and the KV store.
        head_dim = self.kv_params.head_dim
        q_dim = self.n_heads * head_dim
        kv_dim = self.num_key_value_heads * head_dim
        x_q, x_k, x_v = qkv.split([q_dim, kv_dim, kv_dim], axis=-1)

        x_q = x_q.reshape((-1, self.n_heads, head_dim))
        x_q = rms_norm(x_q, self.q_norm_weight, self.rms_norm_eps)
        x_q = x_q.reshape((-1, q_dim))

        x_k = x_k.reshape((-1, self.num_key_value_heads, head_dim))
        x_k = rms_norm(x_k, self.k_norm_weight, self.rms_norm_eps)
        x_k = x_k.reshape((-1, kv_dim))

        qkv = F.concat([x_q, x_k, x_v], axis=-1)

        # Fused rope (identity on NoPE layers) + split + KV store.
        rope = self.rope
        freqs_cis = F.cast(rope.freqs_cis, qkv.dtype).to(qkv.device)
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=rope.interleaved,
        ).rebind_mapping(qkv.mapping)
        xq = xq.reshape((-1, self.n_heads, head_dim))

        # Flash attention with per-head sinks. Both flint mask variants
        # (full causal for NoPE layers, sliding-window causal otherwise)
        # dispatch the sink_weights kernel.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=self.mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
            sink_weights=self.sinks,
        ).rebind_mapping(xq.mapping)
        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
