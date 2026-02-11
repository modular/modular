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

"""Olmo3 Attention Layer."""

from __future__ import annotations

import math

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
)
from max.tensor import Tensor

from ...common_layers.rotary_embedding import RotaryEmbedding
from .functional_kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from .rms_norm import Olmo3RMSNorm


class Olmo3Attention(Module[[Tensor, PagedCacheValues, Tensor], Tensor]):
    """Implementation of the attention layer for the Olmo3 text model.

    Depending on the layer type, the attention layer can be either a full attention
    layer or a sliding window attention layer.

    Olmo3 includes Q and K normalization after the Q and K projections.
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
        mask_variant: MHAMaskVariant,
        scale: float | None = None,
        has_bias: bool = False,
        local_window_size: int = 4096,
        use_qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: Rotary embedding used for the attention layer. Basic RoPE for
                sliding attention, YARN RoPE for full attention.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: The number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params.
            layer_idx: The layer number associated with this Attention block.
            mask_variant: The mask variant for attention (causal or sliding window).
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            local_window_size: Size of the sliding window.
            use_qk_norm: Whether to use Q and K normalization.
            qk_norm_eps: Epsilon value for Q and K normalization.
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.use_qk_norm = use_qk_norm
        self.qk_norm_eps = qk_norm_eps
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.local_window_size = local_window_size
        self.mask_variant = mask_variant

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

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
            bias=self.has_bias,
        )

        # QK normalization layers (used in Olmo3)
        self.q_norm = Olmo3RMSNorm(
            self.q_weight_dim,
            self.qk_norm_eps,
        )
        self.k_norm = Olmo3RMSNorm(
            self.kv_weight_dim,
            self.qk_norm_eps,
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

        # Step 1: Fused QKV projection and cache write
        # fused_qkv_ragged_matmul writes K and V to the cache and returns Q
        q = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            input_row_offsets=input_row_offsets,
            wqkv=self.wqkv,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            bias=self.wqkv_bias,
        )

        # Step 2: Reshape Q and apply RMSNorm
        q = q.reshape((total_seq_len, self.n_heads * self.kv_params.head_dim))
        if self.use_qk_norm:
            q = self.q_norm(q)
        q = q.reshape((total_seq_len, self.n_heads, self.kv_params.head_dim))

        # Step 3: Apply RMSNorm to K in the cache
        if self.use_qk_norm:
            rms_norm_key_cache(
                self.kv_params,
                kv_collection=kv_collection,
                gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                    q.device
                ),
                epsilon=self.qk_norm_eps,
                layer_idx=layer_idx,
                total_seq_len=total_seq_len,
                input_row_offsets=input_row_offsets,
                weight_offset=0.0,
                per_head_norm=False,
            )

        # Step 4: Apply RoPE to Q and K (in cache)
        rope = self.rope
        freqs_cis = F.cast(rope.freqs_cis, q.dtype).to(q.device)

        q = fused_qk_ragged_rope(
            self.kv_params,
            q,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=rope.interleaved,
        )

        # Step 5: Compute flash attention (reads K/V from cache)
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=q,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=self.mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )

        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        ret = self.o_proj(attn_out)
        return ret
