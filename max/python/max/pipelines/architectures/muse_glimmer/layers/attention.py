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

"""Muse Glimmer attention layer for the ModuleV3 API."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.gemma4_modulev3.layers.rms_norm import (
    Gemma4RMSNorm,
)


class MuseGlimmerAttention(Module[..., Tensor]):
    """GQA with weightless qk-norm, a q pre-scale and a sigmoid output gate."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx_in_cache: int,
        is_sliding: bool,
        qk_norm_eps: float,
        qk_scale_factor: float,
        local_window_size: int,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_params = kv_params
        self.layer_idx_in_cache = layer_idx_in_cache
        self.use_local = is_sliding
        self.qk_scale_factor = qk_scale_factor
        self.local_window_size = local_window_size

        self.head_dim = kv_params.head_dim
        self.scale = self.head_dim**-0.5
        self.q_weight_dim = self.head_dim * num_attention_heads
        self.kv_weight_dim = self.head_dim * num_key_value_heads

        self.qk_norm = Gemma4RMSNorm(
            self.head_dim, eps=qk_norm_eps, with_weight=False
        )
        self.q_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.q_weight_dim, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=False
        )
        self.gate_proj = ColumnParallelLinear(
            in_dim=hidden_size, out_dim=self.q_weight_dim, bias=False
        )
        self.o_proj = RowParallelLinear(
            in_dim=self.q_weight_dim, out_dim=hidden_size, bias=False
        )

    @property
    def wqkv(self) -> Tensor:
        """Q, K and V weights stacked along the output dim.

        The concat runs on weights only, so it constant-folds at graph
        compile time and decode issues a single fused QKV matmul.
        """
        return F.concat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            axis=0,
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        *,
        input_row_offsets: Tensor,
    ) -> Tensor:
        layer_idx = F.constant(
            self.layer_idx_in_cache, DType.uint32, device=CPU()
        )
        head_dim = self.head_dim
        q_dim, kv_dim = self.q_weight_dim, self.kv_weight_dim

        x_q, x_k, x_v = (x @ self.wqkv.T).split(
            [q_dim, kv_dim, kv_dim], axis=-1
        )
        x_q = self.qk_norm(x_q.reshape((-1, self.n_heads, head_dim))).reshape(
            (-1, q_dim)
        )
        x_q = x_q * self.qk_scale_factor
        x_k = self.qk_norm(
            x_k.reshape((-1, self.num_key_value_heads, head_dim))
        ).reshape((-1, kv_dim))

        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=F.concat([x_q, x_k, x_v], axis=-1),
            input_row_offsets=input_row_offsets,
            freqs_cis=self.rope.freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads // kv_collection.n_devices,
            interleaved=self.rope.interleaved,
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq.reshape((-1, self.n_heads, head_dim)),
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=(
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if self.use_local
                else MHAMaskVariant.CAUSAL_MASK
            ),
            scale=self.scale,
            local_window_size=self.local_window_size if self.use_local else -1,
        )
        gate = F.sigmoid(self.gate_proj(x))
        return self.o_proj(attn_out.reshape((-1, q_dim)) * gate)
