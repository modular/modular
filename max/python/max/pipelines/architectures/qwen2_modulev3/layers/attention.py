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
"""Qwen2 attention for the ModuleV3 API."""

from __future__ import annotations

import math

from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.nn import Module
from max.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from max.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams
from max.tensor import Tensor


class Qwen2Attention(Module[..., Tensor]):
    """Causal grouped-query attention with biased Q/K/V projections.

    The Q/K/V projections are column-parallel and the output projection is
    row-parallel, so under tensor parallelism each device holds a slice of
    the heads and the output is a partial sum.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        kv_params: KVCacheParams,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.head_dim = head_dim
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.scale = math.sqrt(1.0 / head_dim)
        self.q_dim = head_dim * num_attention_heads
        self.kv_dim = head_dim * num_key_value_heads

        self.q_proj = ColumnParallelLinear(hidden_size, self.q_dim, bias=True)
        self.k_proj = ColumnParallelLinear(hidden_size, self.kv_dim, bias=True)
        self.v_proj = ColumnParallelLinear(hidden_size, self.kv_dim, bias=True)
        self.o_proj = RowParallelLinear(self.q_dim, hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs attention over ``x``, storing its keys and values."""
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())
        per_device_n_heads = self.n_heads // kv_collection.n_devices

        # Concatenating the column-parallel shards keeps each device's
        # q||k||v slice together, so one matmul serves all three.
        wqkv = F.concat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            axis=0,
        )
        assert isinstance(self.q_proj.bias, Tensor)
        assert isinstance(self.k_proj.bias, Tensor)
        assert isinstance(self.v_proj.bias, Tensor)
        bias = F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )
        qkv = x @ wqkv.T + bias

        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=self.rope.freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=per_device_n_heads,
            interleaved=self.rope.interleaved,
        ).rebind_mapping(qkv.mapping)
        xq = xq.reshape((-1, self.n_heads, self.head_dim))

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        ).rebind_mapping(xq.mapping)
        attn_out = attn_out.reshape((-1, self.q_dim))
        return self.o_proj(attn_out)
