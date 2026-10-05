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
"""Nemotron-H attention: grouped-query attention without positional encoding.

Position reaches the attention layers only through the Mamba layers' state,
so keys and values are cached as projected.
"""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
)
from max.experimental.nn.common_layers.linear import QKVLinear
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import fused_qkv_ragged_matmul
from max.nn.kv_cache import KVCacheParams, PagedCacheValues

from ..model_config import NemotronHConfig


class NemotronHAttention(Module[[Tensor, PagedCacheValues, Tensor], Tensor]):
    """NoPE grouped-query attention over the paged KV cache."""

    def __init__(
        self,
        config: NemotronHConfig,
        kv_params: KVCacheParams,
        layer_idx: int,
    ) -> None:
        """Initializes the attention layer.

        Args:
            config: The model config.
            kv_params: The attention leaf of the cache.
            layer_idx: The layer's index among the attention layers, which
                is the layer of the KV cache it owns.
        """
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_dim = self.n_heads * self.head_dim
        kv_dim = config.num_key_value_heads * self.head_dim
        self.qkv_proj = QKVLinear(
            config.hidden_size, q_dim=self.q_dim, kv_dim=kv_dim
        )
        self.o_proj = Linear(self.q_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())
        q = F.functional(fused_qkv_ragged_matmul)(
            self.kv_params,
            input=x,
            input_row_offsets=input_row_offsets,
            wqkv=self.qkv_proj.fused_weight,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        q = q.reshape([-1, self.n_heads, self.head_dim])
        if self.kv_params.is_fp8_kv_dtype:
            # The epilogue saturates K and V into an FP8 cache. The FP8
            # attention kernel also takes an FP8 query.
            q = q.cast(self.kv_params.dtype)
        out = flash_attention_ragged(
            self.kv_params,
            input=q,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=1.0 / math.sqrt(self.head_dim),
            output_dtype=x.dtype,
        )
        return self.o_proj(out.reshape([-1, self.q_dim]))
