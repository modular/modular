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

"""An opaque KV Cache optimized attention mechanism with RoPE (ModuleV3)."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues

from ..norm.rms_norm import rms_norm
from .functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from .rotary_embedding import RotaryEmbedding


class AttentionWithRope(Module[..., Tensor]):
    """Implementation of attention that uses Rotary Position Embedding (RoPE).

    This is a ModuleV3 port of the legacy AttentionWithRope class. It supports
    both separate and stacked QKV projections, optional clip_qkv clamping, and
    optional QK normalization via RMSNorm.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        scale: float | None = None,
        has_bias: bool = False,
        stacked_qkv: bool = False,
        clip_qkv: float | None = None,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache params, including number of kv heads, head
                dim, and dtype.
            layer_idx: The layer number associated with this Attention block.
            scale: Optional attention scale; defaults to sqrt(1/head_dim).
            has_bias: Whether Q/K/V have bias (stacked_qkv forbids bias).
            stacked_qkv: Whether Q/K/V weights are stacked in a single weight.
            clip_qkv: If provided, clamp Q/K/V weights to
                ``[-clip_qkv, clip_qkv]``.
            use_qk_norm: Whether to use RMSNorm on Q/K.
            rms_norm_eps: Value to use for numerical stability in RMSNorm.
        """
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.clip_qkv = clip_qkv
        self.stacked_qkv = stacked_qkv
        self.use_qk_norm = use_qk_norm
        self.rms_norm_eps = rms_norm_eps

        if stacked_qkv and clip_qkv:
            raise ValueError(
                "`clip_qkv` not yet supported when `stacked_qkv=True`."
            )

        if stacked_qkv and has_bias:
            raise ValueError("Bias is not supported with stacked_qkv.")

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_weight_dim

        if stacked_qkv:
            self.qkv_proj = Linear(
                in_dim=hidden_size,
                out_dim=q_weight_dim + 2 * kv_weight_dim,
                bias=False,
            )
        else:
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

        if self.use_qk_norm:
            self.q_norm_weight = Tensor.ones([self.kv_params.head_dim])
            self.k_norm_weight = Tensor.ones([self.kv_params.head_dim])

    @property
    def wqkv(self) -> Tensor:
        """The concatenation of q, k, and v weight vectors."""
        if self.stacked_qkv:
            return self.qkv_proj.weight
        else:
            wq: Tensor = self.q_proj.weight
            wk: Tensor = self.k_proj.weight
            wv: Tensor = self.v_proj.weight
            if self.clip_qkv:
                wq = F.clip(wq, -self.clip_qkv, self.clip_qkv)
                wk = F.clip(wk, -self.clip_qkv, self.clip_qkv)
                wv = F.clip(wv, -self.clip_qkv, self.clip_qkv)
            return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None
        assert not self.stacked_qkv

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
        """Computes attention over the input tensor using the KV cache."""
        total_seq_len = x.shape[0]

        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        # QKV matmul: graph-level weight concat, then a single matmul.
        wqkv = self.wqkv
        qkv = x @ wqkv.T
        if self.wqkv_bias is not None:
            qkv = qkv + self.wqkv_bias

        if self.use_qk_norm:
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

        # Fused rope + split + KV store.
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
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, self.q_weight_dim])
        return self.o_proj(attn_out)
