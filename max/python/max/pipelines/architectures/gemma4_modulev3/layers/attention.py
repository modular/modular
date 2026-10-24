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

"""Gemma4 attention layer for the ModuleV3 API."""

from __future__ import annotations

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers import quant_ops
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    col_parallel,
    row_parallel,
)
from max.experimental.nn.common_layers.quant_linear import (
    QuantizedLinear,
)
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams
from max.nn.quant_config import QuantConfig

from .rms_norm import Gemma4RMSNorm

# One attention projection: a plain parallel Linear (bf16) or a
# QuantizedLinear wrapping a QTensor weight (NVFP4 community quants).
_AttentionProj = ColumnParallelLinear | RowParallelLinear | QuantizedLinear


class Gemma4Attention(Module[..., Tensor]):
    """Gemma4 attention: QKV norms, k==v on global layers, dual geometry."""

    def __init__(
        self,
        *,
        rope_global: RotaryEmbedding,
        rope_local: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_global_key_value_heads: int,
        attention_k_eq_v: bool,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx_in_cache: int,
        is_sliding: bool,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.rope_global = rope_global
        self.rope_local = rope_local
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        # Attention scaling is absorbed by the qk-norms (graph arch does the
        # same); do not apply 1/sqrt(head_dim).
        self.scale = 1.0
        self.use_local = is_sliding
        self.layer_idx_in_cache = layer_idx_in_cache
        self.local_window_size = local_window_size

        # MultiKVCacheParams leaf carries this layer type's head_dim
        # (256 sliding / 512 global for 31B).
        self.head_dim = kv_params.head_dim
        num_kv = (
            num_key_value_heads if is_sliding else num_global_key_value_heads
        )
        self.num_key_value_heads = num_kv
        self.q_weight_dim = self.head_dim * num_attention_heads
        self.kv_weight_dim = self.head_dim * num_kv

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=qk_norm_eps, with_weight=False
        )

        # Global layers with attention_k_eq_v share K weights for V: the
        # checkpoint has no v_proj there.
        self._has_v_proj = not (attention_k_eq_v and not is_sliding)
        self._is_quantized = quant_config is not None

        def proj(in_dim: int, out_dim: int, parallel: str) -> _AttentionProj:
            """Build one projection, quantized (NVFP4) or plain parallel.

            Quantized projections run per-projection matmuls (see
            :meth:`forward`); each still gets the TP sharding intent so
            ``quant_ops.matmul``'s sharding rules match the bf16 path.
            """
            if self._is_quantized:
                quantized = QuantizedLinear(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    bias=False,
                    quant_config=quant_config,
                )
                # Same sharding intent as the plain layers. The helpers'
                # protocol types `weight` as a Tensor, but a QTensor's
                # `to()` co-shards data and scales the same way (as in
                # `tensor_parallel_mlp`).
                if parallel == "col":
                    return col_parallel(quantized)  # type: ignore[type-var]
                return row_parallel(quantized)  # type: ignore[type-var]
            if parallel == "col":
                return col_parallel(
                    ColumnParallelLinear(
                        in_dim=in_dim, out_dim=out_dim, bias=False
                    )
                )
            return row_parallel(
                RowParallelLinear(in_dim=in_dim, out_dim=out_dim, bias=False)
            )

        self.q_proj = proj(hidden_size, self.q_weight_dim, "col")
        self.k_proj = proj(hidden_size, self.kv_weight_dim, "col")
        if self._has_v_proj:
            self.v_proj = proj(hidden_size, self.kv_weight_dim, "col")
        self.o_proj = proj(self.q_weight_dim, hidden_size, "row")

    @property
    def wqkv(self) -> Tensor:
        """Q, K (and V when present) weights stacked along the output dim.

        Only the bf16 path fuses: the concat constant-folds at graph
        compile time and decode issues a single fused QKV matmul.
        Quantized (NVFP4) weights skip this (see :meth:`forward`).
        """
        weights = [self.q_proj.weight, self.k_proj.weight]
        if self._has_v_proj:
            weights.append(self.v_proj.weight)
        return F.concat(weights, axis=0)

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, ...]:
        """Runs the Q/K/V projections, fused when bf16, split when quantized.

        The bf16 path issues one fused matmul over the stacked weights.
        Quantized weights carry per-projection scales, so each projection
        runs its own ``quant_ops.matmul`` — exactly the per-projection
        scales the checkpoint encodes, with no cross-projection scale
        sharing assumptions.
        """
        if not self._is_quantized:
            fused = x @ self.wqkv.T
            splits = (
                [self.q_weight_dim, self.kv_weight_dim, self.kv_weight_dim]
                if self._has_v_proj
                else [self.q_weight_dim, self.kv_weight_dim]
            )
            parts = fused.split(splits, axis=-1)
            return parts[0], parts[1], parts[-1]

        x_q = quant_ops.matmul(x, self.q_proj.weight)
        x_k = quant_ops.matmul(x, self.k_proj.weight)
        if self._has_v_proj:
            x_v = quant_ops.matmul(x, self.v_proj.weight)
        else:
            x_v = x_k
        return x_q, x_k, x_v

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
        per_device_n_heads = self.n_heads // kv_collection.n_devices
        head_dim = self.head_dim
        q_dim, kv_dim = self.q_weight_dim, self.kv_weight_dim
        num_kv_heads = self.num_key_value_heads

        x_q, x_k, x_v = self._project_qkv(x)

        x_q = self.q_norm(x_q.reshape((-1, self.n_heads, head_dim))).reshape(
            (-1, q_dim)
        )
        x_k = self.k_norm(x_k.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )
        x_v = self.v_norm(x_v.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )

        qkv = F.concat([x_q, x_k, x_v], axis=-1)
        rope = self.rope_local if self.use_local else self.rope_global

        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=rope.freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=per_device_n_heads,
            interleaved=rope.interleaved,
            # flash_attention_ragged reads Q in the cache dtype (FP8 under
            # an FP8 KV cache).
            q_out_dtype=self.kv_params.dtype,
        ).rebind_mapping(qkv.mapping)
        xq = xq.reshape((-1, self.n_heads, head_dim))

        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if self.use_local
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size if self.use_local else -1,
            output_dtype=x.dtype,
        ).rebind_mapping(xq.mapping)
        attn_out = attn_out.reshape((-1, self.n_heads * head_dim))
        return self.o_proj(attn_out)
