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
around the k-pool kernels. Every writer runs on every call, in that order: one
batch mixes decode steps, almost never pool-aligned, with ragged chunked-prefill
continuations, so no batch-wide branch can pick one writer for every request,
and each writer acts only on its own share of each request's tokens. The
indexer is replicated: its head scores are summed before the top-k, so every
device must select the same tokens.
"""

from __future__ import annotations

import dataclasses
import functools

from max import functional as F
from max import tree
from max.dtype import DType
from max.nn import Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.quant_linear import QuantizedLinear
from max.nn.kernels import (
    mla_fp8_index_top_k,
    mla_kpool_append,
    mla_kpool_expand_topk,
    mla_kpool_seed_tail,
    quantize_dynamic_scaled_float8,
)
from max.nn.norm import LayerNorm
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.tensor import Tensor
from max.tensor.sharding import DeviceMapping, Sharded
from max.tensor.sharding.action import PerShard

from ..model_config import Glm5NextConfig
from .indexer_inputs import KpoolTailReplayInputs
from .kda import CaptureSink, layer_rows

__all__ = [
    "KPoolTail",
    "PooledIndexer",
    "TailDeferral",
    "act_quant",
    "indexer_act_quant_config",
    "select_pooled",
]


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


def select_pooled(
    k: Tensor,
    gate: Tensor,
    ape: Tensor,
    q_fp8: Tensor,
    weights: Tensor,
    input_row_offsets: Tensor,
    indexer_k_collection: PagedCacheValues,
    layer_idx: Tensor,
    tail: Tensor,
    tail_rows: Tensor,
    tail_layer: Tensor,
    pooling: _Pooling,
    seed_row_offsets: Tensor | None = None,
    capture: CaptureSink[KpoolTailReplayInputs] | None = None,
) -> Tensor:
    """Returns ``[total_tokens, index_topk + index_kpool - 1]`` positions.

    Every tensor is one device's: the ring, the cache and the row offsets
    are that device's own. ``seed_row_offsets`` replaces the tail seed's
    row offsets, and ``capture`` collects the seed's inputs; see
    :class:`TailDeferral`.
    """
    p = pooling
    kpool = p.kpool
    cache = indexer_k_collection
    cache_lengths = cache.cache_lengths
    slot_idx = layer_rows(tail_rows, tail_layer)
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
            x.cast(DType.float32) @ proj.cast(DType.float32).T
        ) * self.n_heads**-0.5
        weights = weights * q_scale.squeeze(-1) * self.softmax_scale
        deferral = self.tail_deferral
        pooling = _Pooling(
            kpool=self.kpool,
            pools_selected=self.pools_selected,
            always_select_tail=self.always_select_tail,
            quant_config=self.quant_config,
        )
        # Every device selects the same positions from its own copy of the
        # cache and ring.
        return F.call_on_mesh(
            functools.partial(select_pooled, pooling=pooling),
            x.mesh,
            out_specs=DeviceMapping.replicated(x.mesh),
        )(
            k,
            self.index_kpool_compress_gate(x),
            self.index_kpool_compress_ape,
            q_fp8,
            weights,
            input_row_offsets,
            indexer_k_collection,
            layer_idx,
            tail.pool,
            tail.rows,
            tail.layer,
            seed_row_offsets=(
                None if deferral is None else deferral.seed_row_offsets
            ),
            capture=(
                None
                if deferral is None
                else PerShard(CaptureSink(c) for c in deferral.captures)
            ),
        )


def _scales_mapping(mapping: DeviceMapping) -> DeviceMapping:
    """The placements of the scales for an activation mapped as ``mapping``.

    ``quantize_dynamic_scaled_float8`` returns scales as
    ``[head_dim // block_size, M_padded]``, so the extent that varies from one
    rank to the next is axis 1, not axis 0. Copying the activation's mapping
    claims the wrong axis once ``M`` is sharded, as it is under data
    parallelism. ``Replicated`` is kept as it is: there every rank's scales
    buffer holds every column, which is the tensor-parallel case.
    """
    return DeviceMapping(
        mapping.mesh,
        tuple(
            Sharded(1) if placement == Sharded(0) else placement
            for placement in mapping.placements
        ),
    )


def act_quant(
    x: Tensor, quant_config: QuantConfig, block_size: int = 128
) -> tuple[Tensor, Tensor]:
    """Quantizes the activation's last axis and returns its scales."""
    *x_dims, head_dim = x.shape
    x = x.reshape((-1, head_dim))
    assert int(head_dim) % block_size == 0

    x_q, x_scales = quantize_dynamic_scaled_float8(
        x,
        quant_config.input_scale,
        quant_config.weight_scale,
        scales_type=DType.float8_e8m0fnu,
        group_size_or_per_token=block_size,
        out_type=DType.float8_e4m3fn,
    )
    # Both outputs take the input's placements; the scales have a different
    # shape than the quantized tensor, which a placements-only mapping allows.
    # The scales' per-rank extent is axis 1, so a shard of ``M`` moves there.
    x_scales = x_scales.rebind_mapping(_scales_mapping(x.mapping))
    x = x_q.rebind_mapping(x.mapping)
    num_rows = x.shape[0]
    x = x.reshape((*x_dims, head_dim))

    # Scales layout from ``quantize_dynamic_scaled_float8`` is
    # ``[head_dim // block_size, M_padded]``; ``M`` is padded for TMA (16-byte
    # alignment of the scale row length). Slice to ``num_rows``, then fold
    # multiple K-block scale rows into one per token when ``head_dim > block_size``.
    x_scales = x_scales[:, :num_rows]
    num_k_groups = int(head_dim) // block_size
    if num_k_groups > 1:
        x_scales = F.max(x_scales, axis=0)
    x_scales = x_scales.reshape((*x_dims, 1))

    return x, x_scales


def indexer_act_quant_config(quant_config: QuantConfig) -> QuantConfig:
    """Returns the quant config used for dynamic FP8 activation quant in the indexer.

    Full FP8 checkpoints reuse the model quant config. Mixed-precision paths
    (for example NVFP4 MoE with bf16 MLA) still dynamic-quantize indexer
    activations with block size 128.
    """
    if quant_config.format == QuantFormat.BLOCKSCALED_FP8:
        return quant_config
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, 128),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float32,
            block_size=(128, 128),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.BLOCKSCALED_FP8,
    )
