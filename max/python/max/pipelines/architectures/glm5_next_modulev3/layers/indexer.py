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

from max import tree
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.quant_linear import QuantizedLinear
from max.experimental.nn.norm import LayerNorm
from max.experimental.sharding import DeviceMapping
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    mla_fp8_index_top_k,
    mla_kpool_compress,
    mla_kpool_expand_topk,
    mla_kpool_ring_close,
    mla_kpool_seed_tail,
    quantize_dynamic_scaled_float8,
    scatter_nd_skip_oob_indices,
    store_k_cache_ragged,
    store_k_scale_cache_ragged,
)
from max.nn.kv_cache import PagedCacheValues as GraphPagedCacheValues
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_2.layers.indexer import (
    exclusive_prefix_sum,
)

from ...deepseekV3_2_modulev3.layers.indexer import (
    act_quant,
    indexer_act_quant_config,
)
from ..model_config import Glm5NextConfig

__all__ = ["KPoolTail", "PooledIndexer"]


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
class _Pooling:
    """The compile-time constants of the pooled selection."""

    kpool: int
    head_dim: int
    pools_selected: int
    always_select_tail: bool
    quant_config: QuantConfig


def _quantize_pooled_keys(
    pooled: TensorValue, p: _Pooling
) -> tuple[TensorValue, TensorValue]:
    """Quantizes pooled keys into the pooled cache's FP8 row layout.

    One float32 scale per pool, replicated across the pool's cache rows.
    """
    rows = ops.flatten(pooled, 0, pooled.rank - 2)
    num_pools = rows.shape[0]
    values, scales = quantize_dynamic_scaled_float8(
        rows,
        p.quant_config.input_scale,
        p.quant_config.weight_scale,
        scales_type=DType.float32,
        group_size_or_per_token=p.head_dim,
        out_type=DType.float8_e4m3fn,
    )
    scales = ops.transpose(scales[:, :num_pools], 0, 1)
    pooled_head_dim = p.head_dim // p.kpool
    cache_rows = ops.unsqueeze(
        values.reshape((num_pools * p.kpool, pooled_head_dim)), 1
    )
    scale_rows = ops.unsqueeze(
        ops.flatten(ops.broadcast_to(scales, (num_pools, p.kpool))).reshape(
            (num_pools * p.kpool, 1)
        ),
        1,
    )
    return cache_rows, scale_rows


def _store_pools(
    cache: GraphPagedCacheValues,
    pooled: TensorValue,
    row_offsets: TensorValue,
    cache_lengths: TensorValue,
    layer_idx: TensorValue,
    p: _Pooling,
) -> None:
    cache_rows, scale_rows = _quantize_pooled_keys(pooled, p)
    aligned = dataclasses.replace(cache, cache_lengths=cache_lengths)
    store_k_cache_ragged(aligned, cache_rows, row_offsets, layer_idx)
    store_k_scale_cache_ragged(
        aligned,
        scale_rows,
        row_offsets,
        layer_idx,
        quantization_granularity=p.quant_config.scales_granularity_mnk[2],
    )


def _store_closed_pools(
    pooled: TensorValue,
    closed_pool: TensorValue,
    row_offsets: TensorValue,
    cache: GraphPagedCacheValues,
    layer_idx: TensorValue,
    p: _Pooling,
) -> None:
    """Compacts the rows that closed a pool and writes them behind the
    cache length, at the closed pool's own boundary."""
    device = pooled.device
    closed = (closed_pool >= 0).cast(DType.int32)
    closed_before = exclusive_prefix_sum(closed)
    # A skipped row must index past the axis; -1 would read from the end.
    skip = ops.constant(1 << 30, DType.int32, device=device)
    indices = ops.unsqueeze(ops.where(closed > 0, closed_before[:-1], skip), -1)
    compacted = scatter_nd_skip_oob_indices(
        ops.broadcast_to(
            ops.constant(0, pooled.dtype, device=device), pooled.shape
        ),
        pooled,
        indices,
    )
    pool_offsets = ops.gather(
        closed_before, row_offsets.cast(DType.int32), axis=0
    )
    lengths = cache.cache_lengths
    _store_pools(
        cache,
        compacted,
        (pool_offsets * p.kpool).cast(DType.uint32),
        ((lengths // p.kpool) * p.kpool).cast(DType.uint32),
        layer_idx,
        p,
    )


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
) -> TensorValue:
    """Returns ``[total_tokens, index_topk + index_kpool - 1]`` positions."""
    p = pooling
    kpool = p.kpool
    device = k.device
    cache = indexer_k_collection
    cache_lengths = cache.cache_lengths
    slot_idx = ops.squeeze(
        tail_rows[(slice(tail_layer, tail_layer + 1), 1), :], 0
    )
    ape = ape.cast(DType.float32)

    # Closes each request's pending ring pool first: `seed_tail` below may
    # reuse the same ring slots for the request's next pool.
    ring_pooled, ring_closed_pool = mla_kpool_ring_close(
        tail, k, gate, ape, input_row_offsets, cache_lengths, slot_idx, kpool
    )
    batch = cache_lengths.shape[0]
    identity_offsets = ops.range(
        0, batch + 1, 1, out_dim=batch + 1, dtype=DType.uint32, device=device
    )
    _store_closed_pools(
        ring_pooled, ring_closed_pool, identity_offsets, cache, layer_idx, p
    )

    cache_len64 = cache_lengths.cast(DType.int64)
    n_per_request = (
        (input_row_offsets[1:] - input_row_offsets[:-1])
        .cast(DType.int64)
        .rebind(cache_len64.shape, "one row-offset span per request")
    )
    # Pools this call completes from new tokens alone.
    full_pools = ops.max(
        (
            (cache_len64 + n_per_request) // kpool
            - (cache_len64 + kpool - 1) // kpool
        ).cast(DType.int64),
        ops.constant(0, DType.int64, device=device),
    )
    pool_row_offsets = exclusive_prefix_sum(full_pools).cast(DType.uint32)
    pooled = mla_kpool_compress(
        k, gate, ape, input_row_offsets, pool_row_offsets, cache_lengths, kpool
    )
    mla_kpool_seed_tail(
        tail, k, gate, input_row_offsets, cache_lengths, slot_idx, kpool
    )
    # `compress` owns the rows from the next pool boundary on; the ring
    # close above owns the floor row.
    _store_pools(
        cache,
        pooled,
        (pool_row_offsets * kpool).cast(DType.uint32),
        (((cache_lengths + kpool - 1) // kpool) * kpool).cast(DType.uint32),
        layer_idx,
        p,
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
                head_dim=self.head_dim,
                pools_selected=self.pools_selected,
                always_select_tail=self.always_select_tail,
                quant_config=self.quant_config,
            ),
        )
        return selection.rebind_mapping(DeviceMapping.replicated(x.mesh))
