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

"""OLMo2 Attention Layer."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues

from .rms_norm import Olmo2RMSNorm


class Olmo2Attention(Module[[Tensor, PagedCacheValues, Tensor], Tensor]):
    """Implementation of the attention layer for the OLMo2 text model.

    Key differences of Olmo2 vs. Llama3:
    - Keys and queries are normalized after the QKV projection
    - RMSNorm is applied on full projection dimension (num_heads * head_dim)
    - No QKV clipping
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
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else 1.0 / math.sqrt(self.kv_params.head_dim)
        )
        self.qk_norm_eps = rms_norm_eps

        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            bias=self.has_bias,
        )
        self.k_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            bias=self.has_bias,
        )
        self.v_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            bias=self.has_bias,
        )
        self.o_proj = Linear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            bias=False,
        )

        # OLMo2: RMSNorm on full projection dimension
        self.q_norm = Olmo2RMSNorm(
            self.q_weight_dim,
            eps=rms_norm_eps,
        )
        self.k_norm = Olmo2RMSNorm(
            self.kv_weight_dim,
            eps=rms_norm_eps,
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
        input_row_offsets: Tensor,
    ) -> Tensor:
        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        # Step 1: QKV matmul.
        wqkv = self.wqkv
        qkv = x @ wqkv.T
        bias = self.wqkv_bias
        if bias is not None:
            qkv = qkv + bias

        # Step 2: Apply full-dimension QK norm before rope.
        head_dim = self.kv_params.head_dim
        q_dim = self.n_heads * head_dim
        kv_dim = self.kv_weight_dim
        x_q, x_k, x_v = qkv.split([q_dim, kv_dim, kv_dim], axis=-1)
        x_q = self.q_norm(x_q)
        x_k = self.k_norm(x_k)
        qkv = F.concat([x_q, x_k, x_v], axis=-1)

        # Step 3: Fused rope + split + KV store.
        rope = self.rope
        freqs_cis = F.cast(rope.freqs_cis, qkv.dtype).to(qkv.device)
        q = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=rope.interleaved,
        )
        q = q.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Step 5: Compute flash attention
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=q,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
