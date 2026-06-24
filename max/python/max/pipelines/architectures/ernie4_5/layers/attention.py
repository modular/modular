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

"""ERNIE-4.5 GQA attention with GPT-J style interleaved RoPE."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ernie45 import Ernie45Rope
import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues


class ERNIE45Attention(Module[..., Tensor]):
    """GQA attention for ERNIE-4.5.

    Uses GPT-J style interleaved RoPE (interleaved=True).
    freqs_cis is pre-built as [cos0,sin0,cos1,sin1,...] per position.
    group = 16Q/2KV = 8 — within GPU kernel limit.
    """

    def __init__(
        self,
        *,
        rope: Ernie45Rope,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.has_bias = has_bias
        self.clip_qkv = clip_qkv
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )

        q_dim = self.kv_params.head_dim * num_attention_heads
        kv_dim = self.kv_params.head_dim * num_key_value_heads
        self.q_weight_dim = q_dim

        self.q_proj = Linear(in_dim=hidden_size, out_dim=q_dim, bias=has_bias)
        self.k_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.v_proj = Linear(in_dim=hidden_size, out_dim=kv_dim, bias=has_bias)
        self.o_proj = Linear(in_dim=q_dim, out_dim=hidden_size, bias=False)

    def forward(
        self, x: Tensor, kv_collection: PagedCacheValues, **kwargs
    ) -> Tensor:
        """Compute GQA attention for a ragged batch using GPT-J RoPE.

        Steps:
            1. Project Q/K/V.
            2. Optionally clip Q/K/V.
            3. Apply RoPE and store K/V via ``rope_split_store_ragged``.
            4. Run flash attention with a causal mask.
            5. Project the attention output back to hidden dimension.

        Args:
            x: Input hidden states, shape ``(total_seq_len, hidden_size)``.
            kv_collection: Paged KV cache for the current layer.
            **kwargs: Must contain ``input_row_offsets`` — uint32 tensor of
                ragged-batch row delimiters.

        Returns:
            Attended output, shape ``(total_seq_len, hidden_size)``.
        """

        total_seq_len = x.shape[0]
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.clip_qkv:
            q = F.clip(q, -self.clip_qkv, self.clip_qkv)
            k = F.clip(k, -self.clip_qkv, self.clip_qkv)
            v = F.clip(v, -self.clip_qkv, self.clip_qkv)

        qkv = F.concat([q, k, v], axis=-1)

        freqs_cis = F.cast(self.rope.freqs_cis, qkv.dtype).to(qkv.device)

        # interleaved=True → GPT-J style rotation (swap adjacent pairs)
        # This matches ERNIE-4.5's apply_rotary implementation
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=True,  # GPT-J style — ERNIE-4.5 specific
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
