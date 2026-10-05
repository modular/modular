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
"""Lightning Indexer layer for DeepseekV3.2."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn import (
    LayerNorm,
    Linear,
    Module,
    QuantConfig,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    mla_fp8_index_top_k,
    mla_kpool_compress,
    mla_kpool_expand_topk,
    mla_kpool_ring_close,
    mla_kpool_seed_tail,
    quantize_dynamic_scaled_float8,
    rope_ragged,
    scatter_nd_skip_oob_indices,
    store_k_cache_ragged,
    store_k_scale_cache_ragged,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import Shardable
from max.nn.quant_config import (
    InputScaleSpec,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from .transforms import HadamardTransform


def exclusive_prefix_sum(x: TensorValue) -> TensorValue:
    """Returns ``[0, x[0], x[0] + x[1], ...]``, one entry longer than ``x``.

    A masked row sum rather than :func:`~max.graph.ops.cumsum`, which has no
    GPU kernel (KERN-1095) and so runs the scan on the host. The host round
    trip that reaches it lowers to a device-to-host copy plus an ``mgp.sync``,
    and a blocking sync cannot be recorded into a captured device graph, which
    takes capture off the table for the whole model.

    ``x`` holds one entry per request, so the ``[n + 1, n]`` selection mask
    this materializes is batch-sized rather than token-sized.
    """
    n = x.shape[0]
    device = x.device
    rows = ops.range(
        0, n + 1, 1, out_dim=n + 1, dtype=DType.int64, device=device
    )
    cols = ops.range(0, n, 1, out_dim=n, dtype=DType.int64, device=device)
    below = ops.unsqueeze(cols, 0) < ops.unsqueeze(rows, -1)
    return ops.squeeze(
        ops.sum(ops.where(below, ops.unsqueeze(x, 0), 0), axis=-1), -1
    )


def act_quant(
    x: TensorValue, quant_config: QuantConfig, block_size: int = 128
) -> tuple[TensorValue, TensorValue]:
    *x_dims, head_dim = x.shape
    x = x.reshape((-1, head_dim))
    assert int(head_dim) % block_size == 0

    x, x_scales = quantize_dynamic_scaled_float8(
        x,
        quant_config.input_scale,
        quant_config.weight_scale,
        scales_type=DType.float8_e8m0fnu,
        group_size_or_per_token=block_size,
        out_type=DType.float8_e4m3fn,
    )
    num_rows = x.shape[0]
    x = x.reshape((*x_dims, head_dim))

    # Scales layout from ``quantize_dynamic_scaled_float8`` is
    # ``[head_dim // block_size, M_padded]``; ``M`` is padded for TMA (16-byte
    # alignment of the scale row length). Slice to ``num_rows``, then fold
    # multiple K-block scale rows into one per token when ``head_dim > block_size``.
    x_scales = x_scales[:, :num_rows]
    num_k_groups = int(head_dim) // block_size
    if num_k_groups > 1:
        x_scales = ops.max(x_scales, axis=0)
    x_scales = x_scales.reshape((*x_dims, 1))

    return x, x_scales


def _indexer_act_quant_config(quant_config: QuantConfig) -> QuantConfig:
    """Return the quant config used for dynamic FP8 activation quant in the indexer.

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


# ---------------------------------------------------------------------
# k-pool compression (any DSA model with index_kpool > 1)
#
# Lives here rather than in a model-specific indexer: k-pool compression
# is a property of the DSA indexer, not of one checkpoint, and the FP8
# indexer kernel already takes `kpool`. Every member below is inert when
# `index_kpool == 1`, the unpooled path.
# ---------------------------------------------------------------------


class Indexer(Module, Shardable):
    def __init__(
        self,
        dim: int,
        index_n_heads: int,
        index_head_dim: int,
        qk_rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        devices: Sequence[DeviceRef],
        quant_config: QuantConfig,
        k_norm_dtype: DType = DType.float32,
        rope_interleaved: bool = False,
        index_kpool: int = 1,
        index_kpool_compress: bool = True,
        index_kpool_always_select_tail: bool = True,
        indexer_weights_fp8: bool | None = None,
    ):
        super().__init__()
        self.dim: int = dim
        self.n_heads: int = index_n_heads
        self.n_local_heads: int = index_n_heads // len(devices)
        self.head_dim: int = index_head_dim
        self.rope_head_dim: int = qk_rope_head_dim
        self.rope_interleaved: bool = rope_interleaved
        # The rotation covers the leading half of each head, so a non-zero
        # rope width must be exactly half of ``index_head_dim``.
        if qk_rope_head_dim != 0 and qk_rope_head_dim * 2 != index_head_dim:
            raise ValueError(
                "indexer rope width must be 0 or half of index_head_dim; got"
                f" qk_rope_head_dim={qk_rope_head_dim} with"
                f" index_head_dim={index_head_dim}"
            )
        self.index_topk: int = index_topk
        self.q_lora_rank: int = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.quant_config = _indexer_act_quant_config(quant_config)
        # Kept for `shard()`: replicas reconstruct from the *original*
        # config, not the activation config derived from it.
        self._quant_config_in: QuantConfig = quant_config
        self._k_norm_dtype: DType = k_norm_dtype
        self._sharding_strategy: ShardingStrategy | None = None

        # Whether the *indexer's own* projections are quantized, which is a
        # property of the checkpoint's FP8 map rather than of the layer: DeepSeek
        # -V3.2 quantizes them; a checkpoint may leave the whole indexer BF16.
        # `None` keeps the historical derivation from the cache's quant format.
        weights_fp8 = (
            quant_config.format == QuantFormat.BLOCKSCALED_FP8
            if indexer_weights_fp8 is None
            else indexer_weights_fp8
        )
        # Kept for `shard()`, which reconstructs replicas.
        self._indexer_weights_fp8: bool = weights_fp8
        weight_dtype = DType.float8_e4m3fn if weights_fp8 else DType.bfloat16
        linear_quant_config = quant_config if weights_fp8 else None

        self.wq_b = Linear(
            in_dim=self.q_lora_rank,
            out_dim=self.n_heads * self.head_dim,
            dtype=weight_dtype,
            device=devices[0],
            quant_config=linear_quant_config,
        )  # lora up projection
        self.wk = Linear(
            in_dim=self.dim,
            out_dim=self.head_dim,
            dtype=weight_dtype,
            device=devices[0],
            quant_config=linear_quant_config,
        )
        self.k_norm = LayerNorm(
            dims=self.head_dim, dtype=k_norm_dtype, devices=devices
        )
        self.weights_proj = Linear(
            in_dim=self.dim,
            out_dim=self.n_heads,
            dtype=DType.bfloat16,
            device=devices[0],
        )  # DS casts to f32

        self.hadamard_transform = HadamardTransform(
            scale=self.softmax_scale, device=devices[0]
        )

        # k-pool compression. Every field below is inert at `index_kpool == 1`,
        # the unpooled path: no extra weights are
        # declared, so their checkpoints are unaffected.
        if index_topk % index_kpool:
            raise ValueError(
                f"index_topk ({index_topk}) must be divisible by index_kpool"
                f" ({index_kpool})."
            )
        if index_head_dim % index_kpool:
            raise ValueError(
                f"index_head_dim ({index_head_dim}) must be divisible by"
                f" index_kpool ({index_kpool}): the pooled-key cache spreads"
                " one pooled key across the token rows its pool covers."
            )
        self.index_kpool: int = index_kpool
        self.index_kpool_compress: bool = (
            index_kpool > 1 and index_kpool_compress
        )
        self.always_select_tail: bool = index_kpool_always_select_tail
        self.dtype: DType = weight_dtype
        if self.index_kpool_compress:
            # Per-token, per-channel gate scores; the pool's softmax runs over
            # these plus `ape` across the pool's members.
            # BF16 and unquantized, like : the checkpoint keeps
            # the compression gate and its position table out of the FP8 map.
            self.index_kpool_compress_gate = Linear(
                in_dim=self.dim,
                out_dim=self.head_dim,
                dtype=DType.bfloat16,
                device=devices[0],
            )
            # Learned position-within-pool signal, added to the gate logits.
            self.index_kpool_compress_ape = Weight(
                "index_kpool_compress_ape",
                DType.bfloat16,
                (index_kpool, index_head_dim),
                device=devices[0],
            )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the indexer's sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the strategy, which must be replication.

        The head scores are *summed* before the top-k, so a rank holding a
        sharded head axis holds a partial sum and would select different tokens
        from its neighbours -- silent divergence, not a crash. Repairing it
        needs an all-reduce of the scores ahead of the top-k, which is
        O(context) per layer, against a fixed and much smaller cost in
        replicated weights. So the indexer replicates and this setter refuses
        anything else.
        """
        if not strategy.is_replicate:
            raise ValueError(
                "Indexer supports only replication: its head scores are summed"
                " before the top-k, so a tensor-parallel head axis would make"
                " ranks select different tokens unless the scores were"
                " all-reduced first, which costs O(context) per layer."
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

    def shard(self, devices: Iterable[DeviceRef]) -> list[Indexer]:
        """Creates one replica of this indexer per device.

        Args:
            devices: Devices to replicate onto.

        Returns:
            One :class:`Indexer` per device, sharing this one's weights.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if self._sharding_strategy is None:
            raise ValueError(
                "Indexer cannot be sharded because no sharding strategy has"
                " been set."
            )
        device_list = list(devices)
        linear_shards = [linear.shard(device_list) for linear in self._linears]
        k_norm_shards = self.k_norm.shard(device_list)
        ape_shards = (
            self.index_kpool_compress_ape.shard(device_list)
            if self.index_kpool_compress
            else None
        )

        replicas: list[Indexer] = []
        for shard_idx, device in enumerate(device_list):
            replica = Indexer(
                dim=self.dim,
                index_n_heads=self.n_heads,
                index_head_dim=self.head_dim,
                qk_rope_head_dim=self.rope_head_dim,
                index_topk=self.index_topk,
                q_lora_rank=self.q_lora_rank,
                devices=[device],
                quant_config=self._quant_config_in,
                k_norm_dtype=self._k_norm_dtype,
                rope_interleaved=self.rope_interleaved,
                index_kpool=self.index_kpool,
                index_kpool_compress=self.index_kpool_compress,
                index_kpool_always_select_tail=self.always_select_tail,
                indexer_weights_fp8=self._indexer_weights_fp8,
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

    def _store_closed_pools(
        self,
        pooled: TensorValue,
        closed_pool: TensorValue,
        input_row_offsets: TensorValue,
        indexer_k_collection: PagedCacheValues,
        layer_idx: TensorValue,
    ) -> None:
        """Writes the pools that closed this step into the pooled-key cache.

        The caller reports one row per closing candidate -- a token row for a
        per-token writer, a request row for a per-request one --
        ``closed_pool[r]`` is the pool that row closed, or ``-1``, so the rows
        that carry a finished pooled key are scattered through the batch. Two
        things follow.

        The rows have to be **compacted**, because
        :func:`~max.nn.kernels.store_k_cache_ragged` writes a request's rows
        contiguously; a non-closing row left in place would be written as if it
        were a pooled key.

        And the destination is **behind** the cache length. Pool ``p`` owns
        cache rows ``[p * kpool, (p + 1) * kpool)``, but the row that closes it
        sits at position ``p * kpool + kpool - 1``, so a store anchored at
        ``cache_lengths`` would land ``kpool - 1`` rows late. Flooring the
        cache length to its pool boundary is the general fix: it is a no-op for
        a prefill from an empty cache (length 0), backs up ``kpool - 1`` rows on
        a decode step, and lands on the right boundary for a chunked prefill
        resuming mid-pool. Only ``cache_lengths`` is rewritten -- the pages,
        the lookup table and the scale table are the same cache.

        Args:
            pooled: ``[num_rows, head_dim]``, meaningful only where
                ``closed_pool`` is non-negative.
            closed_pool: ``[num_rows]`` pool id closed by each row, or ``-1``.
            input_row_offsets: ``[batch + 1]`` offsets mapping each request to
                its span of rows in ``pooled``/``closed_pool`` -- token rows
                for a per-token writer, the identity for a per-request one.
            indexer_k_collection: The pooled-key cache.
            layer_idx: Layer index for cache lookup.
        """
        kpool = self.index_kpool
        device = pooled.device
        closed = (closed_pool >= 0).cast(DType.int32)

        # `[num_rows + 1]`, entry `i` = closed rows strictly before row `i`.
        closed_before = exclusive_prefix_sum(closed)
        ranks = closed_before[:-1]

        # Compact the closing rows to the front, order preserved. A skipped
        # index must be >= the axis length; -1 would be read as an index from
        # the end, so push non-closing rows far past any real row count.
        skip = ops.constant(1 << 30, DType.int32, device=device)
        indices = ops.unsqueeze(ops.where(closed > 0, ranks, skip), -1)
        compacted = scatter_nd_skip_oob_indices(
            ops.broadcast_to(
                ops.constant(0, pooled.dtype, device=device), pooled.shape
            ),
            pooled,
            indices,
        )

        # Each request's pooled rows, in cache rows rather than pools.
        pool_offsets = ops.gather(
            closed_before, input_row_offsets.cast(DType.int32), axis=0
        )
        store_offsets = (pool_offsets * kpool).cast(DType.uint32)

        cache_rows, scale_rows = self.quantize_pooled_keys(
            compacted, self.quant_config
        )
        lengths = indexer_k_collection.cache_lengths
        # Integer `//` promotes to float64, and the store kernels take
        # `cache_lengths` at its declared uint32, so cast back before rebinding.
        aligned = dataclasses.replace(
            indexer_k_collection,
            cache_lengths=((lengths // kpool) * kpool).cast(DType.uint32),
        )
        store_k_cache_ragged(aligned, cache_rows, store_offsets, layer_idx)
        store_k_scale_cache_ragged(
            aligned,
            scale_rows,
            store_offsets,
            layer_idx,
            quantization_granularity=self.quant_config.scales_granularity_mnk[
                2
            ],
        )

    def _select_pooled(
        self,
        x: TensorValue,
        k: TensorValue,
        q_fp8: TensorValue,
        weights: TensorValue,
        input_row_offsets: TensorValue,
        indexer_k_collection: PagedCacheValues,
        layer_idx: TensorValue,
        mask_variant: MHAMaskVariant,
        tail: BufferValue,
        slot_idx: TensorValue,
    ) -> TensorValue:
        """Compresses, caches and scores k-pools, then expands the selection.

        Scoring pools rather than tokens is the whole point of k-pooling: the
        candidate set shrinks by ``index_kpool``, and a pooled key is immutable
        once its members exist, so it is written to the cache once and read
        forever.

        A pool's members do not have to arrive together: a decoded token
        completes a pool whose other members left the batch several steps
        ago, and a chunked prefill resumes mid-pool. The ring in ``tail``
        carries the members that have not formed a complete pool yet.

        Every call runs the same three writers, unconditionally -- no
        ``ops.cond`` split, because none of them assume anything about this
        call's alignment or per-request token count. A production batch
        routinely mixes ordinary decode (misaligned almost always: only a
        step that happens to land exactly on a pool boundary is aligned) with
        ragged, differently-sized chunked-prefill continuations, and no
        single runtime branch can route a whole *batch* to one writer or the
        other when different *requests* in it need different treatment --
        which is what the previous batch-wide ``ops.cond`` predicate got
        wrong (it forced every request onto whichever writer the least
        aligned request in the batch needed, and the ring-aware writer that
        won that race assumed one new token per request, so any other
        request's ragged, multi-token chunk in the same call read and wrote
        out of bounds):

        - :func:`~max.nn.kernels.mla_kpool_ring_close` closes each request's
          *pending* pool -- the one the ring already partly holds -- using
          the ring's members plus this call's leading new tokens, whenever
          this call brings enough of them. A request with nothing pending,
          or not yet enough new tokens to finish it, closes nothing.
        - :func:`~max.nn.kernels.mla_kpool_compress` builds every pool that is
          entirely new tokens, skipping the leading tokens the ring-close
          writer above just consumed (or none, when nothing was pending).
        - :func:`~max.nn.kernels.mla_kpool_seed_tail` stashes whatever remains
          incomplete after this call into the ring, for a later call to
          close.

        Args:
            x: ``[total_tokens, dim]`` normalized hidden states.
            k: ``[total_tokens, index_head_dim]`` layer-normed indexer keys.
            q_fp8: Quantized queries.
            weights: Per-head query weights.
            input_row_offsets: ``[batch + 1]`` token row offsets.
            indexer_k_collection: The indexer's pooled-key cache.
            layer_idx: Layer index for cache lookup.
            mask_variant: Mask to apply while scoring.
            tail: This layer's ``[max_slots, 2, index_kpool, index_head_dim]``
                in-progress-pool ring, mutated in place.
            slot_idx: ``[batch]`` ring slot per request.

        Returns:
            ``[total_tokens, index_topk + index_kpool - 1]`` token positions.
        """
        kpool = self.index_kpool
        device = k.device
        cache_lengths = indexer_k_collection.cache_lengths
        gate = self.index_kpool_compress_gate(x)
        ape = self.index_kpool_compress_ape.cast(DType.float32)

        # Closes each request's pending ring pool, if this call brings enough
        # new tokens to finish it. Must run -- and its store below must land
        # -- before `mla_kpool_seed_tail` reuses the same ring slots for
        # whatever remains incomplete after this call: `mla_kpool_seed_tail`
        # only ever writes at or past `cache_lengths` (see its own
        # `seed_start`), so the two never touch the same *absolute* position,
        # but the ring is slot-indexed mod `kpool`, and a slot this op reads
        # can be one `seed_tail` reuses for the request's *next* pool later in
        # this same call. Sequencing the two calls in this order is what
        # keeps that read-then-overwrite race from being reordered.
        ring_pooled, ring_closed_pool = mla_kpool_ring_close(
            tail,
            k,
            gate,
            ape,
            input_row_offsets,
            cache_lengths,
            slot_idx,
            kpool,
        )
        # `ring_pooled`/`ring_closed_pool` are one row per *request*, not per
        # token, so `_store_closed_pools`'s compaction needs a row-offsets
        # array that maps request `b` to row `b` -- the identity, rather than
        # `input_row_offsets` (which maps request `b` to its *token* rows).
        batch = cache_lengths.shape[0]
        identity_offsets = ops.range(
            0,
            batch + 1,
            1,
            out_dim=batch + 1,
            dtype=DType.uint32,
            device=device,
        )
        self._store_closed_pools(
            ring_pooled,
            ring_closed_pool,
            identity_offsets,
            indexer_k_collection,
            layer_idx,
        )

        cache_len64 = cache_lengths.cast(DType.int64)
        # `input_row_offsets` carries its own length symbol while
        # `cache_lengths` carries the replica's batch dim. One offset span
        # per request makes them the same extent at runtime, which the
        # compiler cannot see on its own.
        n_per_request = (
            (input_row_offsets[1:] - input_row_offsets[:-1])
            .cast(DType.int64)
            .rebind(cache_len64.shape, "one row-offset span per request")
        )
        # Pools this call completes from new tokens alone: the pool index one
        # past its last new token, minus the pool index its cache already
        # reached. This already excludes whatever `mla_kpool_ring_close` above
        # just closed -- `kpool_compress_kernel` independently derives the
        # same count per request from `cache_lengths` and its own row span,
        # skipping exactly the tokens a pending pool would consume; if this
        # ever disagrees, the pooled-key cache silently misaligns with the
        # compressed count, so a numeric equivalence test must cover this
        # arithmetic, not just the kernel body.
        # Integer `//` promotes to float64, so the difference comes back cast
        # before it meets an int64 floor.
        full_pools = ops.max(
            (
                (cache_len64 + n_per_request) // kpool
                - (cache_len64 + kpool - 1) // kpool
            ).cast(DType.int64),
            ops.constant(0, DType.int64, device=device),
        )
        pool_row_offsets = exclusive_prefix_sum(full_pools).cast(DType.uint32)

        pooled = mla_kpool_compress(
            k,
            gate,
            ape,
            input_row_offsets,
            pool_row_offsets,
            cache_lengths,
            kpool,
        )
        # Only the trailing, not-yet-complete remainder: the pool
        # `mla_kpool_ring_close` just closed and every whole pool `compress`
        # just built are both already accounted for.
        mla_kpool_seed_tail(
            tail, k, gate, input_row_offsets, cache_lengths, slot_idx, kpool
        )

        # `compress`'s output is dense per request (no scatter needed, unlike
        # `mla_kpool_ring_close`'s sparse one), so it stores directly at each
        # request's cache boundary via `pool_row_offsets` as the ragged
        # row-offsets array. The boundary is the *next* pool boundary at or
        # after `cache_lengths` -- ceiling rather than floor -- so this store
        # and `mla_kpool_ring_close`'s never target the same cache row:
        # `mla_kpool_ring_close` owns the floor row (if it closed anything),
        # `compress` owns every row after it.
        aligned = dataclasses.replace(
            indexer_k_collection,
            cache_lengths=(((cache_lengths + kpool - 1) // kpool) * kpool).cast(
                DType.uint32
            ),
        )
        cache_rows, scale_rows = self.quantize_pooled_keys(
            pooled, self.quant_config
        )
        # `pool_row_offsets` counts pools, which is what `mla_kpool_compress`
        # wants, but a pooled key spreads over the `index_kpool` cache rows its
        # pool covers (see `pooled_cache_head_dim`), so the store -- which
        # addresses cache rows -- needs the same offsets scaled. Without the
        # scale a batch of two or more attributes a request's rows to the
        # request before it and writes past the end of its span.
        store_row_offsets = (pool_row_offsets * kpool).cast(DType.uint32)
        store_k_cache_ragged(aligned, cache_rows, store_row_offsets, layer_idx)
        store_k_scale_cache_ragged(
            aligned,
            scale_rows,
            store_row_offsets,
            layer_idx,
            quantization_granularity=(
                self.quant_config.scales_granularity_mnk[2]
            ),
        )

        pool_ids = mla_fp8_index_top_k(
            q_fp8,
            weights,
            input_row_offsets,
            indexer_k_collection,
            layer_idx,
            self.pools_selected,
            self.quant_config.scales_granularity_mnk[2],
            mask_variant,
            kpool=self.index_kpool,
        )
        return mla_kpool_expand_topk(
            pool_ids,
            input_row_offsets,
            indexer_k_collection.cache_lengths,
            self.index_kpool,
            self.always_select_tail,
        )

    def __call__(
        self,
        x: TensorValue,
        qr: TensorValue,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        indexer_k_collection: PagedCacheValues,
        layer_idx: TensorValue,
        mask_variant: MHAMaskVariant = MHAMaskVariant.NULL_MASK,
        *,
        tail: BufferValue | None = None,
        slot_idx: TensorValue | None = None,
    ) -> TensorValue:
        """
        Args:
            x: Tensor of shape (total_seq_len, dim) Input activations.
            qr: Tensor of shape (total_seq_len, q_lora_rank) Pre-normed queries.
            start_pos: Tensor scalar. Used to slice the freqs_cis tensor.
            freqs_cis: Tensor of shape (seq_len, head_dim) RoPE frequencies.
            input_row_offsets: Tensor of shape (total_seq_len + 1) Ragged-tensor
                index that tells where each sequence (batch item) starts and
                ends in a concatenated “ragged” input.
            indexer_k_collection: Indexer's K cache values
            layer_idx: Layer index for cache lookup
            mask_variant: Mask to apply while scoring.
            tail: The k-pool tail ring, required when ``index_kpool > 1``.
            slot_idx: ``[batch]`` ring slot per request, required when
                ``index_kpool > 1``.
        Returns:
            topk_indices, tTensor of shape (total_seq_len, index_topk) indices
                of the top k Keys selected by the Indexer for MLA to attend to.
        """
        # qr comes projected to lora rank and pre-normed; q_lora_rank -> self.n_heads * self.head_dim
        q = self.wq_b(qr)
        q = q.reshape((-1, self.n_heads, self.head_dim))
        if self.rope_head_dim:
            q_pe, q_nope = ops.chunk(q, chunks=2, axis=-1)
            q_pe = rope_ragged(
                q_pe,
                input_row_offsets,
                indexer_k_collection.cache_lengths,
                freqs_cis,
                interleaved=self.rope_interleaved,
            )
            q = ops.concat([q_pe, q_nope], axis=-1)

        k = self.wk(x)  # dim -> head_dim
        k = self.k_norm(k)

        if self.rope_head_dim:
            k_pe, k_nope = ops.chunk(k, chunks=2, axis=-1)
            k_pe = ops.squeeze(
                rope_ragged(
                    ops.unsqueeze(k_pe, axis=-2),
                    input_row_offsets,
                    indexer_k_collection.cache_lengths,
                    freqs_cis,
                    interleaved=self.rope_interleaved,
                ),
                axis=-2,
            )
            k = ops.concat([k_pe, k_nope], axis=-1)

        q_fp8, q_scale = act_quant(q, self.quant_config)
        k_fp8, k_scale = act_quant(k, self.quant_config)

        weights_for_scoring = (
            ops.unsqueeze(
                self.weights_proj(x.cast(DType.float32)) * self.n_heads**-0.5,
                axis=-1,
            )
            * q_scale
            * self.softmax_scale
        )

        if self.index_kpool > 1:
            if tail is None or slot_idx is None:
                raise ValueError(
                    "k-pooled indexing needs the per-request tail ring and its"
                    " slot indices: a decoded token closes a pool whose other"
                    " members left the batch several steps ago, so they have"
                    " to be carried across steps in the ring."
                )
            # Pooled keys replace the per-token ones in the cache, so the
            # per-token store below is skipped entirely.
            return self._select_pooled(
                x,
                k,
                q_fp8,
                ops.squeeze(weights_for_scoring, axis=-1),
                input_row_offsets,
                indexer_k_collection,
                layer_idx,
                mask_variant,
                tail,
                slot_idx,
            )

        store_k_cache_ragged(
            indexer_k_collection,
            ops.unsqueeze(k_fp8, axis=1),
            input_row_offsets,
            layer_idx,
        )
        store_k_scale_cache_ragged(
            indexer_k_collection,
            ops.unsqueeze(k_scale, axis=1).cast(DType.float32),
            input_row_offsets,
            layer_idx,
            quantization_granularity=self.quant_config.scales_granularity_mnk[
                2
            ],
        )

        return mla_fp8_index_top_k(
            q_fp8,
            ops.squeeze(weights_for_scoring, axis=-1),
            input_row_offsets,
            indexer_k_collection,
            layer_idx,
            self.index_topk,
            self.quant_config.scales_granularity_mnk[2],
            mask_variant,
        )

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
