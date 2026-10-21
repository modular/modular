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
"""k-pooled DSA indexer for GLM-5.3-Flash.

GLM-5.3-Flash keeps DeepSeek Sparse Attention's budget --- ``index_topk`` = 2048
tokens per query out of a full KV cache --- but scores *pools* of
``index_kpool`` = 4 consecutive tokens rather than individual tokens. A pool
compresses to one candidate key: a channel-wise softmax-weighted average of its
four indexer keys, weighted by a learned gate (``index_kpool_compress_gate``,
4096 -> 128) plus a learned intra-pool absolute position embedding
(``index_kpool_compress_ape``, ``[4, 128]``). Scoring picks
``index_topk / index_kpool`` = 512 pools, each expands back to its four token
indices, and the current incomplete pool is appended unconditionally, giving a
selection width of ``index_topk + index_kpool - 1`` = 2051.

Two facts about this layer are load-bearing and easy to lose:

* ``index_kpool_compress_ape`` is a positional signal *inside* the indexer, so
  "NoPE" does not mean the indexer is position-blind. It is zero-initialized in
  the reference, so dropping it degrades long-context retrieval slightly rather
  than crashing.
* A pooled key is immutable once its ``index_kpool`` tokens exist, which is
  what makes the pooled key cacheable. Rebuilding pools from a raw per-token
  indexer cache on every decode step --- what a faithful port of the reference
  produces --- reads ``kv_len x (132 + 256)`` bytes per sparse layer per step
  against a pooled cache's ``kv_len / 4 x 132``: 2.9x *worse* than GLM-5.2
  rather than 4x better. See
  ``.claude/skills/glm-bringup/lanes/pooled-indexer.md``.

What this module implements is the compression, the expansion and the tail, plus
a dense scorer. The compression and the expansion are permanent --- the
expansion is ``O(index_topk)`` and belongs in graph ops. The dense scorer is
not: it materialises ``[tokens, heads, pools]`` scores, the same ceiling that
limits the transformers reference to a few thousand tokens, and exists to gate
the compression, the masking and the expansion against that reference on real
head dimensions before any of it is fused. It is the oracle the fused pooled
scorer is diffed against.

Reference: ``transformers`` PR 48342 at ``f57a815``,
``models/glm5_next/modular_glm5_next.py`` --- ``Glm5NextTextIndexer.forward``,
``get_pooled_states`` and ``append_visible_tail``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    DeviceRef,
    Dim,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn import LayerNorm, Linear, Module
from max.nn.kernels import quantize_dynamic_scaled_float8
from max.nn.layer import Shardable
from max.nn.quant_config import QuantConfig

__all__ = [
    "Glm5NextIndexer",
    "PoolGrid",
    "indexer_traffic_bytes_per_token",
]

# Finite stand-in for -inf in the pool-compression softmax. The reference masks
# invalid members to -inf and then calls `nan_to_num` to rescue an all-invalid
# pool from NaN. A large finite sentinel underflows to exactly the same
# probabilities for a partially valid pool and gives an all-invalid pool a
# uniform average instead of NaN; only fully invalid pools differ, and those are
# masked out of the selection before the top-k, so their pooled key is never
# read.
_MASK_LOGIT = -1.0e30

# Score of a pool a query may not select. `finfo(float32).min`, matching the
# reference's `masked_fill(~valid_candidates, finfo.min)`.
_MASK_SCORE = -3.4028234663852886e38


@dataclass(frozen=True)
class PoolGrid:
    """Where each pool's member tokens live in a ragged packed batch.

    Pools tile each sequence from its own first token, so pool ``p`` of sequence
    ``b`` covers local positions ``[p * kpool, (p + 1) * kpool)``. That
    alignment is to the sequence --- not to the packed batch and not to a cache
    page --- and it is what lets a pooled key be written once and read forever.
    It is also why left padding would invalidate every cached pool: the
    reference starts pooling at the first *real* token for exactly this reason,
    and MAX's ragged layout gets that for free by never padding on the left.
    """

    member_rows: TensorValue
    """``[batch, max_pools, kpool]`` int32 rows into the packed
    ``[total_tokens, ...]`` tensors, clamped inside the sequence so a gather is
    always in bounds."""

    member_valid: TensorValue
    """``[batch, max_pools, kpool]`` bool, false past the sequence's end."""

    member_positions: TensorValue
    """``[batch, max_pools, kpool]`` int32 token positions local to the
    sequence, ``-1`` where invalid. An invalid member expands to ``-1`` and
    never to a clamped valid index, or attention would read a position it was
    not meant to see."""

    pool_valid: TensorValue
    """``[batch, max_pools]`` bool, true only where every member is valid --- a
    partial pool is never selectable."""

    pool_end: TensorValue
    """``[batch, max_pools]`` int32 local position of the pool's last member;
    the query's causal test is against this."""


class Glm5NextIndexer(Module, Shardable):
    """Scores k-pools and returns the token indices sparse MLA attends to.

    Args:
        hidden_size: Decoder width, 4096.
        q_lora_rank: Width of the pre-normed query residual ``wq_b`` consumes,
            1536.
        index_n_heads: Indexer heads, 32. Their scores are *summed*, so this
            axis must not be tensor-parallel sharded without an all-reduce
            ahead of the top-k or ranks select different tokens; see the lane
            doc for why replicating the indexer wins instead.
        index_head_dim: Indexer head dimension, 128.
        index_topk: Token budget per query, 2048.
        index_kpool: Tokens per pool, 4. Read off the checkpoint --- the
            reference implementation's own default is 16.
        devices: Devices the indexer's weights live on.
        dtype: Weight and activation dtype. The whole indexer is BF16 in the
            checkpoint even where the rest of the sparse-MLA layer is FP8, so
            ``quantization.UNQUANTIZED_ATTN_PROJECTIONS`` names it.
        index_kpool_compress: Whether pooled keys are the learned weighted
            average. False falls back to a plain mean over the pool.
        index_kpool_always_select_tail: Whether the incomplete trailing pool is
            appended to every query's selection.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        index_kpool: int,
        devices: Sequence[DeviceRef],
        dtype: DType = DType.bfloat16,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
    ) -> None:
        super().__init__()
        if index_topk % index_kpool != 0:
            raise ValueError(
                f"index_topk ({index_topk}) must be divisible by index_kpool "
                f"({index_kpool})."
            )
        if index_head_dim % index_kpool != 0:
            raise ValueError(
                f"index_head_dim ({index_head_dim}) must be divisible by "
                f"index_kpool ({index_kpool}): the pooled-key cache spreads one "
                "pooled key across the token rows it covers."
            )
        self.hidden_size = hidden_size
        self.q_lora_rank = q_lora_rank
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_kpool = index_kpool
        self.index_kpool_compress = index_kpool_compress
        self.always_select_tail = index_kpool_always_select_tail
        self.dtype = dtype
        self.devices = list(devices)
        self.softmax_scale = index_head_dim**-0.5
        self._sharding_strategy: ShardingStrategy | None = None

        device = self.devices[0]
        self.wq_b = Linear(
            in_dim=q_lora_rank,
            out_dim=index_n_heads * index_head_dim,
            dtype=dtype,
            device=device,
        )
        self.wk = Linear(
            in_dim=hidden_size,
            out_dim=index_head_dim,
            dtype=dtype,
            device=device,
        )
        self.k_norm = LayerNorm(
            dims=index_head_dim,
            devices=self.devices,
            dtype=dtype,
            use_bias=True,
        )
        self.weights_proj = Linear(
            in_dim=hidden_size,
            out_dim=index_n_heads,
            dtype=dtype,
            device=device,
        )
        if index_kpool_compress:
            self.index_kpool_compress_gate = Linear(
                in_dim=hidden_size,
                out_dim=index_head_dim,
                dtype=dtype,
                device=device,
            )
            self.index_kpool_compress_ape = Weight(
                "index_kpool_compress_ape",
                dtype,
                (index_kpool, index_head_dim),
                device=device,
            )

    # -------------------------------------------------------------- sharding

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the indexer's sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the strategy, which must be replication.

        The 32 head scores are *summed* before the top-k, so a rank holding a
        sharded head axis holds a partial sum and would select different tokens
        from its neighbours --- silent divergence, not a crash. Repairing it
        needs an all-reduce of the scores ahead of the top-k, which is
        O(context): roughly 1 MB per sequence per layer at 1M tokens, against
        a fixed 164 MB of replicated weights across the 11 sparse layers. So
        the indexer replicates and this setter refuses anything else.
        """
        if not strategy.is_replicate:
            raise ValueError(
                "Glm5NextIndexer supports only replication: its head scores "
                "are summed before the top-k, so a tensor-parallel head axis "
                "would make ranks select different tokens unless the scores "
                "were all-reduced first, which costs O(context) per layer."
            )
        self._sharding_strategy = strategy
        for linear in self._linears:
            linear.sharding_strategy = strategy
        self.k_norm.sharding_strategy = strategy
        if self.index_kpool_compress:
            self.index_kpool_compress_ape.sharding_strategy = strategy

    @property
    def _linears(self) -> tuple[Linear, ...]:
        """Every :class:`Linear` this module owns, in declaration order."""
        linears = (self.wq_b, self.wk, self.weights_proj)
        if not self.index_kpool_compress:
            return linears
        return (*linears, self.index_kpool_compress_gate)

    def shard(self, devices: Iterable[DeviceRef]) -> list[Glm5NextIndexer]:
        """Creates one replica of this indexer per device.

        Args:
            devices: Devices to replicate onto.

        Returns:
            One :class:`Glm5NextIndexer` per device, sharing this one's
            weights.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if self._sharding_strategy is None:
            raise ValueError(
                "Glm5NextIndexer cannot be sharded because no sharding "
                "strategy has been set."
            )
        device_list = list(devices)
        linear_shards = [linear.shard(device_list) for linear in self._linears]
        k_norm_shards = self.k_norm.shard(device_list)
        ape_shards = (
            self.index_kpool_compress_ape.shard(device_list)
            if self.index_kpool_compress
            else None
        )

        replicas: list[Glm5NextIndexer] = []
        for shard_idx, device in enumerate(device_list):
            replica = Glm5NextIndexer(
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                index_n_heads=self.n_heads,
                index_head_dim=self.head_dim,
                index_topk=self.index_topk,
                index_kpool=self.index_kpool,
                devices=[device],
                dtype=self.dtype,
                index_kpool_compress=self.index_kpool_compress,
                index_kpool_always_select_tail=self.always_select_tail,
            )
            replica.wq_b = linear_shards[0][shard_idx]
            replica.wk = linear_shards[1][shard_idx]
            replica.weights_proj = linear_shards[2][shard_idx]
            replica.k_norm = k_norm_shards[shard_idx]
            if ape_shards is not None:
                replica.index_kpool_compress_gate = linear_shards[3][shard_idx]
                replica.index_kpool_compress_ape = ape_shards[shard_idx]
            replicas.append(replica)
        return replicas

    @property
    def pools_selected(self) -> int:
        """Pools each query selects, ``index_topk / index_kpool`` = 512."""
        return self.index_topk // self.index_kpool

    @property
    def selection_width(self) -> int:
        """Output width: ``index_topk + index_kpool - 1`` = 2051 with the
        tail, ``index_topk`` without."""
        if not self.always_select_tail or self.index_kpool == 1:
            return self.index_topk
        return self.index_topk + self.index_kpool - 1

    @property
    def pooled_cache_head_dim(self) -> int:
        """Declared ``head_dim`` of the pooled-key cache: 128 / 4 = 32.

        The paged cache manager indexes one row per *token* and every leaf of a
        :class:`~max.nn.kv_cache.MultiKVCacheParams` shares one page table, so a
        cache holding one key per ``index_kpool`` tokens cannot be declared four
        times shorter. Declaring it ``index_head_dim / index_kpool`` wide
        instead spreads one 128-value pooled key across the ``index_kpool``
        token rows it covers, which --- because those rows are adjacent inside a
        page --- is byte-identical to a compacted ``[pools, index_head_dim]``
        cache while keeping the manager's per-token accounting honest. Requires
        ``page_size % index_kpool == 0``; the default 128 satisfies it.
        """
        return self.head_dim // self.index_kpool

    # ------------------------------------------------------------ projections

    def keys_and_gate(
        self, x: TensorValue
    ) -> tuple[TensorValue, TensorValue | None]:
        """Returns the per-token indexer key and the pool-gate logits.

        Args:
            x: ``[total_tokens, hidden_size]`` normalized hidden states.

        Returns:
            The LayerNormed key ``[total_tokens, index_head_dim]`` and the gate
            logits of the same shape, the latter ``None`` when
            ``index_kpool_compress`` is off.
        """
        k = self.k_norm(self.wk(x))
        if not self.index_kpool_compress:
            return k, None
        return k, self.index_kpool_compress_gate(x)

    def queries(self, qr: TensorValue) -> TensorValue:
        """Returns ``[total_tokens, index_n_heads, index_head_dim]`` queries.

        There is no rotary embedding here. ``indexer_rope_interleave`` survives
        in the checkpoint config from GLM-5.2 and is vestigial: with
        ``qk_rope_head_dim`` = 0 it selects between two no-ops.
        """
        return self.wq_b(qr).reshape((-1, self.n_heads, self.head_dim))

    def head_weights(self, x: TensorValue) -> TensorValue:
        """Returns ``[total_tokens, index_n_heads]`` float32 head weights.

        The projection runs in the weight dtype and is cast up afterwards,
        matching the reference's
        ``weights_proj(hidden_states.to(weight.dtype)).float()``.
        """
        weights = self.weights_proj(x.cast(self.dtype)).cast(DType.float32)
        return weights * (self.n_heads**-0.5)

    # ------------------------------------------------------------ compression

    def pool_grid(
        self, input_row_offsets: TensorValue, max_pools: int
    ) -> PoolGrid:
        """Builds the per-sequence pool grid for a ragged packed batch.

        Args:
            input_row_offsets: ``[batch + 1]`` row offsets into the packed
                tensors.
            max_pools: Pools laid out per sequence; must be at least
                ``ceil(max_seq_len / index_kpool)``. A uniform count keeps the
                grid rectangular and matches the reference, whose pool axis is
                also shared across the batch; pools past a sequence's end come
                out invalid.

        Returns:
            The grid.
        """
        kpool = self.index_kpool
        device = input_row_offsets.device
        offsets = input_row_offsets.cast(DType.int32)
        starts = offsets[:-1].reshape((-1, 1, 1))
        lengths = offsets[1:].reshape((-1, 1, 1)) - starts

        local = ops.range(
            start=0,
            stop=max_pools * kpool,
            step=1,
            dtype=DType.int32,
            device=device,
        ).reshape((1, max_pools, kpool))

        member_valid = local < lengths
        minus_one = ops.constant(-1, DType.int32, device=device)
        member_positions = ops.where(
            member_valid, local + starts * 0, minus_one
        )
        member_rows = starts + ops.min(local, lengths - 1)
        pool_valid = (
            ops.squeeze(ops.min(member_valid.cast(DType.int32), axis=-1), -1)
            > 0
        )
        pool_end = ops.squeeze(
            ops.min(local[:, :, kpool - 1 :], lengths - 1), -1
        )
        return PoolGrid(
            member_rows=member_rows,
            member_valid=member_valid,
            member_positions=member_positions,
            pool_valid=pool_valid,
            pool_end=pool_end,
        )

    def compress_pools(
        self, k: TensorValue, gate_logits: TensorValue | None, grid: PoolGrid
    ) -> TensorValue:
        """Compresses each pool's member keys into one candidate key.

        ``softmax(gate + ape)`` runs over the ``index_kpool`` members *per
        channel* --- ``index_kpool_compress_gate`` emits a 128-wide vector per
        token, not a scalar --- and the weights are cast back to the key dtype
        before the weighted sum, both matching the reference.

        Args:
            k: ``[total_tokens, index_head_dim]`` per-token indexer keys.
            gate_logits: ``[total_tokens, index_head_dim]`` gate logits, or
                ``None`` for a plain mean over the pool.
            grid: The pool grid.

        Returns:
            ``[batch, max_pools, index_head_dim]`` pooled keys.
        """
        member_axis = 2
        members = ops.gather(k, grid.member_rows, axis=0)
        valid = ops.unsqueeze(grid.member_valid, -1)
        zero = ops.constant(0.0, DType.float32, device=k.device)

        if gate_logits is None:
            counts = ops.unsqueeze(
                ops.sum(
                    grid.member_valid.cast(DType.float32), axis=member_axis
                ),
                -1,
            )
            summed = ops.sum(
                ops.where(valid, members.cast(DType.float32), zero),
                axis=member_axis,
            )
            one = ops.constant(1.0, DType.float32, device=k.device)
            return ops.squeeze(
                (summed / ops.max(counts, one)).cast(k.dtype), member_axis
            )

        ape = self.index_kpool_compress_ape.cast(DType.float32).reshape(
            (1, 1, self.index_kpool, self.head_dim)
        )
        logits = (
            ops.gather(gate_logits, grid.member_rows, axis=0).cast(
                DType.float32
            )
            + ape
        )
        logits = ops.where(
            valid,
            logits,
            ops.constant(_MASK_LOGIT, DType.float32, device=k.device),
        )
        probabilities = ops.softmax(logits, axis=member_axis).cast(k.dtype)
        return ops.squeeze(
            ops.sum(probabilities * members, axis=member_axis), member_axis
        )

    def quantize_pooled_keys(
        self, pooled: TensorValue, quant_config: QuantConfig
    ) -> tuple[TensorValue, TensorValue]:
        """Quantizes pooled keys into the pooled cache's FP8 row layout.

        One float32 scale per *pool*, replicated across the ``index_kpool``
        cache rows the pool occupies. That replication is deliberate: it makes
        the cached path quantize exactly what a rebuild-from-raw path would, so
        the rebuild-versus-cached differential is a bit-for-bit test rather than
        an approximate one.

        Args:
            pooled: ``[..., index_head_dim]`` pooled keys.
            quant_config: The layer's quant config, read for the scale specs.

        Returns:
            ``([pools * index_kpool, 1, pooled_cache_head_dim]`` FP8,
            ``[pools * index_kpool, 1, 1]`` float32``)`` --- the shapes
            ``store_k_cache_ragged`` and ``store_k_scale_cache_ragged`` take.
        """
        rows = ops.flatten(pooled, 0, pooled.rank - 2)
        num_pools = rows.shape[0]
        values, scales = quantize_dynamic_scaled_float8(
            rows,
            quant_config.input_scale,
            quant_config.weight_scale,
            scales_type=DType.float32,
            group_size_or_per_token=self.head_dim,
            out_type=DType.float8_e4m3fn,
        )
        # Scales come back as `[head_dim // block_size, M]` with M padded for
        # TMA alignment; there is one K block here, so slice off the padding.
        scales = ops.transpose(scales[:, :num_pools], 0, 1)
        cache_rows = ops.unsqueeze(
            values.reshape(
                (num_pools * self.index_kpool, self.pooled_cache_head_dim)
            ),
            1,
        )
        scale_rows = ops.unsqueeze(
            ops.flatten(
                ops.broadcast_to(scales, (num_pools, self.index_kpool))
            ).reshape((num_pools * self.index_kpool, 1)),
            1,
        )
        return cache_rows, scale_rows

    # -------------------------------------------------------------- selection

    def expand_pools_and_append_tail(
        self,
        selected_pools: TensorValue,
        selected_valid: TensorValue,
        member_positions: TensorValue,
        visible_count: TensorValue,
    ) -> TensorValue:
        """Turns selected pool ids into the token positions attention reads.

        ``O(index_topk)``, so this stays in graph ops permanently however the
        scorer is implemented. Each rule below is a place the reference is easy
        to approximate and wrong to:

        * an unselectable pool expands to ``-1`` in all ``index_kpool`` slots,
          never to a clamped valid index;
        * the tail's start is the visible token count rounded *down* to a pool
          boundary, so it lands on the current incomplete pool rather than at a
          fixed offset;
        * the result is right-padded with ``-1`` to the full selection width
          even when fewer pools exist than the budget, so the consumer's stride
          never changes.

        Args:
            selected_pools: ``[total_tokens, pools_selected]`` int32 ids into
                the flattened ``[batch * max_pools]`` pool axis.
            selected_valid: ``[total_tokens, pools_selected]`` bool.
            member_positions: ``[batch * max_pools, index_kpool]`` int32 local
                token positions per pool, already ``-1`` where invalid.
            visible_count: ``[total_tokens, 1]`` int32 tokens visible to each
                query --- its local position plus one.

        Returns:
            ``[total_tokens, selection_width]`` int32.
        """
        kpool = self.index_kpool
        device = selected_pools.device
        minus_one = ops.constant(-1, DType.int32, device=device)
        num_selected = selected_pools.shape[1]

        total_tokens = selected_pools.shape[0]
        expanded = ops.gather(
            member_positions, ops.flatten(selected_pools), axis=0
        ).reshape((total_tokens, num_selected, kpool))
        body = ops.flatten(
            ops.where(ops.unsqueeze(selected_valid, -1), expanded, minus_one),
            1,
            2,
        )

        if not self.always_select_tail or kpool == 1:
            return self._pad_to_width(body)

        tail_width = kpool - 1
        tail_offsets = ops.range(
            start=0,
            stop=tail_width,
            step=1,
            dtype=DType.int32,
            device=device,
        ).reshape((1, tail_width))
        tail_count = visible_count % kpool
        tail = (visible_count - tail_count) + tail_offsets
        tail = ops.where(tail_offsets < tail_count, tail, minus_one)
        return self._pad_to_width(ops.concat([body, tail], axis=-1))

    def _pad_to_width(self, indices: TensorValue) -> TensorValue:
        """Right-pads ``indices`` with ``-1`` to :attr:`selection_width`."""
        width = int(indices.shape[-1])
        if width == self.selection_width:
            return indices
        if width > self.selection_width:
            return indices[:, : self.selection_width]
        fill = ops.constant(-1, DType.int32, device=indices.device).reshape(
            (1, 1)
        )
        return ops.concat(
            [
                indices,
                ops.broadcast_to(
                    fill, (indices.shape[0], self.selection_width - width)
                ),
            ],
            axis=-1,
        )

    def __call__(
        self,
        x: TensorValue,
        qr: TensorValue,
        input_row_offsets: TensorValue,
        *,
        max_pools: int,
    ) -> TensorValue:
        """Selects the token positions sparse MLA attends to, densely.

        This is the correctness path. It materialises a
        ``[total_tokens, index_n_heads, batch * max_pools]`` score tensor and
        assumes an empty KV cache --- every key it scores is projected from
        ``x`` --- which is exactly the regime the transformers reference can
        answer for.

        Args:
            x: ``[total_tokens, hidden_size]`` normalized hidden states.
            qr: ``[total_tokens, q_lora_rank]`` pre-normed query residual.
            input_row_offsets: ``[batch + 1]`` ragged row offsets.
            max_pools: Pools laid out per sequence; must cover the longest
                sequence in the batch.

        Returns:
            ``[total_tokens, selection_width]`` int32 token positions local to
            each query's own sequence, ``-1`` where unused.
        """
        grid = self.pool_grid(input_row_offsets, max_pools)
        k, gate_logits = self.keys_and_gate(x)
        pooled_keys = self.compress_pools(k, gate_logits, grid)

        batch = input_row_offsets.shape[0] - 1
        total_tokens = x.shape[0]
        flat_pooled = ops.flatten(pooled_keys.cast(DType.float32), 0, 1)
        # Flat 2-D GEMM then reshape, rather than a rank-3 `q @ pooled^T`: the
        # batched form asks the multistage GEMM to split a dynamic dimension.
        q = ops.flatten(self.queries(qr).cast(DType.float32), 0, 1)
        scores = ops.relu(
            (q @ flat_pooled.transpose(0, 1) * self.softmax_scale).reshape(
                (total_tokens, self.n_heads, flat_pooled.shape[0])
            )
        )
        # Weight per head and sum across heads. A multiply-and-reduce rather
        # than the reference's `[T, 1, H] @ [T, H, P]`, which is the same
        # arithmetic and does not put a dynamic batch dimension in a GEMM.
        index_scores = ops.squeeze(
            ops.sum(
                scores * ops.unsqueeze(self.head_weights(x), -1),
                axis=1,
            ),
            1,
        )

        seq_id, local_pos = _packed_positions(input_row_offsets, x.shape[0])
        pool_seq = ops.flatten(
            ops.broadcast_to(
                ops.unsqueeze(
                    ops.range(
                        start=0,
                        stop=batch,
                        step=1,
                        out_dim=batch,
                        dtype=DType.int32,
                        device=x.device,
                    ),
                    -1,
                ),
                (batch, max_pools),
            )
        )
        pool_valid = ops.flatten(grid.pool_valid)
        pool_end = ops.flatten(grid.pool_end)
        candidate = _pool_is_candidate(
            pool_seq, pool_valid, pool_end, seq_id, local_pos
        )
        masked = ops.where(
            candidate,
            index_scores,
            ops.constant(_MASK_SCORE, DType.float32, device=x.device),
        )

        select_k = min(self.pools_selected, max_pools)
        _, selected = ops.top_k(masked, k=select_k, axis=-1)
        selected = selected.cast(DType.int32)
        selected_valid = _pool_is_candidate(
            ops.gather(pool_seq, selected, axis=0),
            ops.gather(pool_valid, selected, axis=0),
            ops.gather(pool_end, selected, axis=0),
            seq_id,
            local_pos,
        )
        return self.expand_pools_and_append_tail(
            selected,
            selected_valid,
            ops.flatten(grid.member_positions, 0, 1),
            ops.unsqueeze(local_pos + 1, -1),
        )


def _pool_is_candidate(
    pool_seq: TensorValue,
    pool_valid: TensorValue,
    pool_end: TensorValue,
    seq_id: TensorValue,
    local_pos: TensorValue,
) -> TensorValue:
    """Whether each pool is selectable by each query.

    A pool is a candidate only if every member is valid *and* its last member
    is visible to the query under causality. Broadcasting is over the query
    axis, so this is reused both to mask the score matrix and to re-derive
    validity after the top-k --- the reference re-gathers the same mask rather
    than trusting the scores, so that a padded or short row yields a ``-1``
    sentinel instead of pool 0.
    """
    if pool_seq.rank == 1:
        pool_seq = ops.unsqueeze(pool_seq, 0)
        pool_valid = ops.unsqueeze(pool_valid, 0)
        pool_end = ops.unsqueeze(pool_end, 0)
    query_seq = ops.unsqueeze(seq_id, -1)
    query_pos = ops.unsqueeze(local_pos, -1)
    return ops.logical_and(
        ops.equal(pool_seq, query_seq),
        ops.logical_and(pool_valid, pool_end <= query_pos),
    )


def _packed_positions(
    input_row_offsets: TensorValue, total_tokens: Dim
) -> tuple[TensorValue, TensorValue]:
    """Returns each packed row's ``(sequence id, local position)``.

    The packed ragged layout carries no inverse of ``input_row_offsets``, so
    recover it by counting how many sequence ends a row is at or past. The
    comparison matrix is ``total_tokens x batch``, small next to the score
    tensor it feeds.
    """
    offsets = input_row_offsets.cast(DType.int32)
    rows = ops.unsqueeze(
        ops.range(
            start=0,
            stop=total_tokens,
            step=1,
            out_dim=total_tokens,
            dtype=DType.int32,
            device=offsets.device,
        ),
        -1,
    )
    seq_id = ops.squeeze(
        ops.sum(
            (rows >= ops.unsqueeze(offsets[1:], 0)).cast(DType.int32), axis=-1
        ),
        -1,
    )
    local_pos = ops.squeeze(rows, -1) - ops.gather(offsets[:-1], seq_id, axis=0)
    return seq_id, local_pos


def indexer_traffic_bytes_per_token(
    *,
    num_sparse_layers: int,
    kv_len: int,
    index_head_dim: int,
    index_kpool: int,
    pooled_cache: bool,
) -> int:
    """Bytes the indexer reads per decoded token, for cache-sizing decisions.

    An FP8 candidate key costs ``index_head_dim`` value bytes plus one float32
    scale. Rebuilding a pool additionally needs its members' BF16 gate logits,
    and that term is what makes a faithful port of the reference *slower* than
    GLM-5.2 rather than faster.

    Args:
        num_sparse_layers: Sparse-MLA layers holding an indexer, 11 plus the
            MTP draft layer when speculative decoding is on.
        kv_len: Context length in tokens.
        index_head_dim: 128.
        index_kpool: 4.
        pooled_cache: Whether pooled keys are cached rather than rebuilt.

    Returns:
        Bytes read across all sparse layers for one decoded token.
    """
    key_bytes = index_head_dim + 4
    if pooled_cache:
        return num_sparse_layers * math.ceil(kv_len / index_kpool) * key_bytes
    return num_sparse_layers * kv_len * (key_bytes + 2 * index_head_dim)
