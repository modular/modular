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
"""The k-pooled DSA indexer of GLM-5.3-Flash, on ModuleV3.

The indexer scores 4-token pools instead of tokens and expands the 512 pools
it selects back to 2051 token positions (the pools plus the three tokens of the
incomplete trailing pool). A pool's members arrive across steps, so the
incomplete ones ride in a per-request tail ring between steps.

The pooling pipeline -- close the ring's pending pool, compress whole new
pools, stash the remainder, cache, score, expand -- is per-device graph code
around the k-pool kernels, the same sequence the graph-API indexer
(``deepseekV3_2.layers.indexer.Indexer._select_pooled``) runs; see it for why
each writer runs unconditionally and in this order. The indexer is replicated:
its head scores are summed before the top-k, so every device must select the
same tokens.
"""

from __future__ import annotations

import dataclasses

from max import functional as F
from max import tree
from max.dtype import DType
from max.graph import BufferValue, TensorValue, ops
from max.nn import Module
from max.nn._graph_ops import (
    mla_fp8_index_top_k,
    mla_kpool_append,
    mla_kpool_expand_topk,
    mla_kpool_seed_tail,
)
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.quant_linear import QuantizedLinear
from max.nn.kv_cache import PagedCacheValues as GraphPagedCacheValues
from max.nn.norm import LayerNorm
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_2.layers.indexer import (
    KpoolTailReplayInputs,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping
from max.tensor.sharding.action import PerShard

from ...deepseekV3_2_modulev3.layers.indexer import (
    act_quant,
    indexer_act_quant_config,
)
from ..model_config import Glm5NextConfig
from .kda import CaptureSink

__all__ = ["KPoolTail", "PooledIndexer", "TailDeferral"]


@tree.dataclass(frozen=True)
class KPoolTail:
    """One sparse layer's view of the shared k-pool tail ring."""

    pool: Tensor
    """``[num_rows, 2, index_kpool, index_head_dim]``, every layer's rows."""
    rows: Tensor
    """``[num_sparse_layers, batch_size]`` ring rows."""
    layer: Tensor
    """int64 CPU scalar: this layer's position in the indexer cache."""


@dataclasses.dataclass(frozen=True)
class TailDeferral:
    """Defers one sparse layer's tail-ring seed to a speculative rollback.

    Set on the indexer around a verify, and around the MTP draft's first
    step, which runs its own indexer over the same window. The ring is
    slot-indexed modulo ``index_kpool``, so a rejected draft's seed could
    overwrite a member the next step closes its pool with; the rollback seeds
    the accepted rows instead, from what :attr:`captures` collects.
    """

    seed_row_offsets: Tensor
    """Row offsets the seed reads in place of the call's own: the offsets
    without drafts, so the forward lands the ring, and all zeros with them."""
    captures: list[list[KpoolTailReplayInputs]]
    """Per-device sink each deferred layer appends its seed inputs to."""


@dataclasses.dataclass(frozen=True)
class _Pooling:
    """The compile-time constants of the pooled selection."""

    kpool: int
    pools_selected: int
    always_select_tail: bool
    quant_config: QuantConfig


def _select_pooled_graph(
    k: TensorValue,
    gate: TensorValue,
    ape: TensorValue,
    q_fp8: TensorValue,
    weights: TensorValue,
    input_row_offsets: TensorValue,
    indexer_k_collection: GraphPagedCacheValues,
    layer_idx: TensorValue,
    tail: BufferValue,
    tail_rows: TensorValue,
    tail_layer: TensorValue,
    pooling: _Pooling,
    seed_row_offsets: TensorValue | None = None,
    capture: CaptureSink[KpoolTailReplayInputs] | None = None,
) -> TensorValue:
    """Returns ``[total_tokens, index_topk + index_kpool - 1]`` positions.

    ``seed_row_offsets`` replaces the tail seed's row offsets, and
    ``capture`` collects the seed's inputs; see :class:`TailDeferral`.
    """
    p = pooling
    kpool = p.kpool
    cache = indexer_k_collection
    cache_lengths = cache.cache_lengths
    slot_idx = ops.squeeze(
        tail_rows[(slice(tail_layer, tail_layer + 1), 1), :], 0
    )
    ape = ape.cast(DType.float32)

    # Every pool this call completes, closed from the ring or built from new
    # tokens, is pooled, quantized and cached in one launch. It reads the ring
    # before `seed_tail` below reuses its slots for the request's next pool.
    mla_kpool_append(
        tail,
        k,
        gate,
        ape,
        input_row_offsets,
        slot_idx,
        cache,
        layer_idx,
        kpool,
        p.quant_config.scales_granularity_mnk[2],
    )
    mla_kpool_seed_tail(
        tail,
        k,
        gate,
        input_row_offsets if seed_row_offsets is None else seed_row_offsets,
        cache_lengths,
        slot_idx,
        kpool,
    )
    if capture is not None:
        capture.items.append(
            KpoolTailReplayInputs(
                k=k,
                gate=gate,
                tail=tail,
                slot_idx=slot_idx,
                cache_lengths=cache_lengths,
                kpool=kpool,
            )
        )

    pool_ids = mla_fp8_index_top_k(
        q_fp8,
        weights,
        input_row_offsets,
        cache,
        layer_idx,
        p.pools_selected,
        p.quant_config.scales_granularity_mnk[2],
        # Without a causal mask a prefill query spends its pool budget on
        # pools after it, which the sparse attention's causal mask discards.
        MHAMaskVariant.CAUSAL_MASK,
        kpool=kpool,
    )
    return mla_kpool_expand_topk(
        pool_ids,
        input_row_offsets,
        cache.cache_lengths,
        kpool,
        p.always_select_tail,
    )


_select_pooled = F.functional(_select_pooled_graph)


class PooledIndexer(Module[..., Tensor]):
    """Selects the 2051 token positions each query's sparse MLA attends to."""

    def __init__(
        self, config: Glm5NextConfig, quant_config: QuantConfig
    ) -> None:
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.kpool = config.index_kpool
        self.always_select_tail = config.index_kpool_always_select_tail
        if config.index_topk % self.kpool or self.head_dim % self.kpool:
            raise ValueError(
                f"index_topk ({config.index_topk}) and index_head_dim "
                f"({self.head_dim}) must both be divisible by index_kpool "
                f"({self.kpool})."
            )
        if not config.index_kpool_compress:
            raise ValueError(
                "GLM-5.3-Flash's indexer compresses its pools; the mean-pooled "
                "variant is not implemented."
            )
        self.pools_selected = config.index_topk // self.kpool
        self.quant_config = indexer_act_quant_config(quant_config)

        # The whole indexer is BF16 in this checkpoint.
        self.wq_b = QuantizedLinear(
            in_dim=config.q_lora_rank,
            out_dim=self.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = QuantizedLinear(
            in_dim=config.hidden_size, out_dim=self.head_dim, bias=False
        )
        # GLM norms the indexer key in the compute dtype, not float32.
        self.k_norm = LayerNorm(dim=self.head_dim, keep_dtype=True)
        self.weights_proj = QuantizedLinear(
            in_dim=config.hidden_size, out_dim=self.n_heads, bias=False
        )
        self.index_kpool_compress_gate = QuantizedLinear(
            in_dim=config.hidden_size, out_dim=self.head_dim, bias=False
        )
        self.index_kpool_compress_ape = Tensor.zeros(
            [self.kpool, self.head_dim]
        )
        self.tail_deferral: TailDeferral | None = None
        """Set around a speculative verify or MTP draft step, else ``None``."""

    @property
    def selection_width(self) -> int:
        """Selected token positions per query: 512 pools plus the tail."""
        if not self.always_select_tail:
            return self.pools_selected * self.kpool
        return self.pools_selected * self.kpool + self.kpool - 1

    def forward(
        self,
        x: Tensor,
        qr: Tensor,
        input_row_offsets: Tensor,
        indexer_k_collection: PagedCacheValues,
        layer_idx: Tensor,
        tail: KPoolTail,
    ) -> Tensor:
        """Indexes ``x`` and returns the top-k pool picks per token."""
        q = self.wq_b(qr).reshape((-1, self.n_heads, self.head_dim))
        k = self.k_norm(self.wk(x))
        q_fp8, q_scale = act_quant(q, self.quant_config)
        # The scoring weights are projected in float32.
        proj = self.weights_proj.weight
        assert isinstance(proj, Tensor)
        weights = (
            F.cast(x, DType.float32) @ F.cast(proj, DType.float32).T
        ) * self.n_heads**-0.5
        weights = F.unsqueeze(weights, axis=-1) * q_scale * self.softmax_scale
        deferral = self.tail_deferral
        selection = _select_pooled(
            k,
            self.index_kpool_compress_gate(x),
            self.index_kpool_compress_ape,
            q_fp8,
            F.squeeze(weights, axis=-1),
            input_row_offsets,
            indexer_k_collection,
            layer_idx,
            tail.pool,
            tail.rows,
            tail.layer,
            _Pooling(
                kpool=self.kpool,
                pools_selected=self.pools_selected,
                always_select_tail=self.always_select_tail,
                quant_config=self.quant_config,
            ),
            seed_row_offsets=(
                None if deferral is None else deferral.seed_row_offsets
            ),
            capture=(
                None
                if deferral is None
                else PerShard(CaptureSink(c) for c in deferral.captures)
            ),
        )
        return selection.rebind_mapping(DeviceMapping.replicated(x.mesh))
