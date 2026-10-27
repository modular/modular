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

"""Muse Spark attention layer for the ModuleV3 API."""

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
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import KVCacheParams

from .norms import weightless_rms_norm


class NoPERotaryEmbedding(RotaryEmbedding):
    """Identity RoPE for the full-attention (NoPE) layers.

    Every inverse frequency is zero, so ``freqs_cis`` is all ``(1, 0)``
    pairs and full layers store K and V through the same fused
    ``rope_split_store_ragged`` kernel as sliding layers.
    """

    def _compute_inv_freqs(self) -> Tensor:
        return Tensor.zeros(
            [self.head_dim // 2], dtype=DType.float32, device=CPU()
        )


class MuseSparkAttention(Module[..., Tensor]):
    """GQA with weightless qk-norm, a q pre-scale, an output norm and a sigmoid gate."""

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
        norm_eps: float,
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
        self.norm_eps = norm_eps
        self.qk_scale_factor = qk_scale_factor
        self.local_window_size = local_window_size

        self.head_dim = kv_params.head_dim
        self.scale = self.head_dim**-0.5
        self.q_weight_dim = self.head_dim * num_attention_heads
        self.kv_weight_dim = self.head_dim * num_key_value_heads

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
        """Q, K and V weights stacked along the output dim, for one matmul."""
        return F.concat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            axis=0,
        )

    def _head_norm(self, x: Tensor, n_heads: int) -> Tensor:
        return weightless_rms_norm(
            x.reshape((-1, n_heads, self.head_dim)), self.norm_eps
        ).reshape((-1, n_heads * self.head_dim))

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
        q_dim, kv_dim = self.q_weight_dim, self.kv_weight_dim

        x_q, x_k, x_v = (x @ self.wqkv.T).split(
            [q_dim, kv_dim, kv_dim], axis=-1
        )
        x_q = self._head_norm(x_q, self.n_heads) * self.qk_scale_factor
        x_k = self._head_norm(x_k, self.num_key_value_heads)

        qkv = F.concat([x_q, x_k, x_v], axis=-1)
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=self.rope.freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads // kv_collection.n_devices,
            interleaved=self.rope.interleaved,
            # flash_attention_ragged reads Q in the cache dtype (FP8 under
            # an FP8 KV cache).
            q_out_dtype=self.kv_params.dtype,
        ).rebind_mapping(qkv.mapping)
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq.reshape((-1, self.n_heads, self.head_dim)),
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
            output_dtype=x.dtype,
        ).rebind_mapping(xq.mapping)
        attn_out = self._head_norm(attn_out, self.n_heads)
        return self.o_proj(attn_out * F.sigmoid(self.gate_proj(x)))
