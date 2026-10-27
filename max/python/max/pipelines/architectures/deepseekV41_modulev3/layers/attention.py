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
"""Implements V4.1 attention with packed CSA2 caches.

The stacked q/KV projection feeds the query and cache paths. Attention
uses the canonical MXFP8 window and NVFP4 compressed records, then inverse
RoPE, the FP8 grouped wo_a projection (one block-diagonal MXFP8 launch),
and the FP8 row-parallel wo_b."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.legacy.kv_cache import KVCacheInputsPerDevice
from max.tensor import Tensor
from max.tensor.sharding import (
    DeviceMapping,
    DeviceMesh,
    Partial,
    Sharded,
)

from ...deepseekV41.layers.attention import (
    prefill_rows_bound,
)
from ...deepseekV41.layers.csa2 import (
    MAX_COMPRESSED,
    PREFILL_ROWS,
    WINDOW_ENTRIES,
    csa2_attention,
    leaf_slots,
    records_per_page,
    store_window_kv,
    window_entries,
)
from ..layer_resolver import LayerInfo
from .mxfp8_linear import MXFP8Linear, grouped_dense_op
from .norm import RMSNorm
from .rope import RopeParams, apply_rope_tail, rope_for_layer

if TYPE_CHECKING:
    from ..model_config import DeepseekV41Config
    from .compressor import Compressor
    from .indexer import Indexer

__all__ = [
    "Attention",
    "RaggedRows",
    "absolute_positions",
    "position_rope_table",
    "ragged_rows",
    "row_ids",
]

# A forward whose requests each bring at most this many query tokens runs
# the decode attention, which reads the records in place; a longer one
# dequantizes the records it names once, first (the registered V2 model's
# split, host-flag driven inside ``mo.csa2.attention``).
_DECODE_MAX_QUERY_TOKENS = 8

# The V2 CSA2 graph ops, bridged into the experimental-tensor world: the
# ``F.functional`` machinery converts each tensor argument to the graph
# value its parameter's hint names (``BufferValue`` for the cache blocks)
# and each KV record through ``to_graph_values``.
_prefill_rows_bound_op = F.functional(prefill_rows_bound)


def _placed_leaf_slots(
    leaf: KVCacheInputsPerDevice[TensorValue, BufferValue],
    bid: TensorValue,
    entries: TensorValue,
    layer: TensorValue,
) -> TensorValue:
    layer_index = ops.cast(layer, DType.int32)
    if layer_index.device != entries.device:
        layer_index = layer_index.to(entries.device)
    base = leaf_slots(leaf, bid, entries, 0)
    return ops.where(
        base >= 0,
        base + layer_index * records_per_page(leaf.kv_blocks),
        ops.constant(-1, DType.int32, entries.device),
    )


_leaf_slots_op = F.functional(_placed_leaf_slots)
_window_entries_op = F.functional(window_entries)
_store_window_op = F.functional(store_window_kv)
_csa2_attend_op = F.functional(csa2_attention)


def row_ids(rows: Tensor, total: int | Dim) -> tuple[Tensor, Tensor]:
    """``([total], [total])`` int32 request id and index within it, per row.

    Derived with the histogram-cumsum trick from V4's ``segment_ids`` (V4
    ragged.py:113-136; a ``[T, batch]`` comparison tile is never
    materialized): ``bid[r]`` is the number of segment *ends* at or
    before ``r``, which is the row's request index (rows of request 0
    carry bid 0). ``total`` is the token count -- not
    derivable from ``rows``' batch+1 shape, so it arrives as an
    argument exactly as in V4. All small tensors take ``rows``
    mapping, so they sit with it on one device or replicated over the
    mesh; ``F.range`` would otherwise default to an accelerator.
    """
    off = F.cast(rows, DType.int32)
    b = off.shape[0] - 1
    ends = F.gather(
        off,
        F.range(0, b, 1, out_dim=b, dtype=DType.int32, device=rows.mapping) + 1,
        axis=0,
    )
    # [T + 1] row indices; the final slot is the sink for ends at or
    # past T so zero-length requests can't double-count. ``pos`` is a
    # range output (not a concat), so the negative-index hazard from
    # V4's Issue 34 does not apply to ``pos[-1]``.
    pos = F.range(
        0,
        total + 1,
        1,
        out_dim=total + 1,
        dtype=DType.int32,
        device=rows.mapping,
    )
    clipped = F.min(ends, pos[-1])
    hist = F.scatter_add(
        F.full([total + 1], 0, dtype=DType.int32, device=rows.mapping),
        F.full([b], 1, dtype=DType.int32, device=rows.mapping),
        clipped,
        axis=0,
    )
    bid = F.rebind(F.cumsum(hist, axis=0)[:-1], [total])
    # Captured decode pads its batch, so T can exceed row_offsets[-1];
    # pin those rows to the last request to keep their gathers in bounds.
    last = (
        F.range(
            0, b + 1, 1, out_dim=b + 1, dtype=DType.int32, device=rows.mapping
        )[-1]
        - 1
    )
    bid = F.min(bid, F.reshape(last, [1]))
    return bid, pos[:-1] - F.gather(off, bid, axis=0)


def absolute_positions(
    rows: Tensor, cache_lengths: Tensor, total: int | Dim
) -> Tensor:
    """``[total]`` int32 absolute positions, in-graph.

    Computed once per step in the trunk -- every input is layer-invariant
    -- and staged into the shared bodies as the ``positions`` operand; each
    layer's attention (its RoPE) and the compressor and indexer consume
    that tensor instead of recomputing it.

    Each row's position is its request's ``cache_lengths`` plus its
    index within the request's chunk.
    """
    bid, local = row_ids(rows, total)
    starts = F.cast(F.gather(cache_lengths, bid, axis=0), DType.int32)
    return starts + local


@dataclass
class RaggedRows:
    """The forward's per-row bookkeeping, computed once per step.

    This stack's own view of V2's ``V41RaggedRows``: ``bid`` and
    ``local`` from :func:`row_ids`, ``positions`` from
    :func:`absolute_positions`, and this chunk's per-request row
    ``lengths`` (the compressor's ring save reads them).
    """

    offsets: Tensor
    """``[B + 1]`` uint32 ragged row offsets."""
    bid: Tensor
    """``[T]`` int32 request index of each row."""
    local: Tensor
    """``[T]`` int32 index of each row within its request."""
    positions: Tensor
    """``[T]`` int32 absolute positions."""
    lengths: Tensor
    """``[B]`` int32 rows each request brings this step."""

    @property
    def batch(self) -> int | Dim:
        return self.offsets.shape[0] - 1


def live_rows(row_offsets: Tensor, total: int | Dim) -> Tensor:
    """``[total]`` bool: whether each row is live, not capture padding.

    A captured decode pads its row axis past ``row_offsets[-1]``; the
    padded rows synthesize positions past the last request's start and
    must never store into, or gather from, the paged caches.
    """
    row_index = F.range(
        0,
        total,
        1,
        out_dim=total,
        dtype=DType.int32,
        device=row_offsets.mapping,
    )
    return row_index < F.cast(row_offsets[-1], DType.int32)


def window_store_slots(
    swa: PagedCacheValues, rows: RaggedRows, layer_swa: Tensor
) -> Tensor:
    """The window store's ``[T]`` slots, padded rows masked to ``-1``.

    The store op has no row-offsets guard of its own: it launches over
    the physical row count and only skips negative slots, so an
    unmasked padded row would write its record through its synthesized
    position -- possibly onto a page the request does not own. The
    padded rows' entries drop to ``-1`` first, and ``leaf_slots`` maps
    them to ``-1`` slots; the store then skips them. The row's own
    position still reaches the store op (its RoPE-table read must stay
    in bounds), only the destination slot masks.
    """
    live = live_rows(rows.offsets, rows.positions.shape[0])
    entries = F.where(
        live,
        rows.positions,
        F.constant(-1, DType.int32, rows.positions.mapping),
    )
    return _leaf_slots_op(swa, rows.bid, entries, layer_swa)


def window_list(
    swa: PagedCacheValues,
    rows: RaggedRows,
    swa_begin: Tensor,
    layer_swa: Tensor,
) -> Tensor:
    """Each row's ``[T, 128]`` window slots, padded rows fully masked.

    ``leaf_slots`` gathers the lookup table at the entry's page, so a
    padded row's unmasked entries could name a page past its request's
    allocation and gather out of bounds before the attention runs.
    The padded rows' whole window list drops to ``-1``; their
    attention output is discarded by the trunk anyway. The request
    axis is one per row (``[T, 1]``): ``leaf_slots`` broadcasts it
    straight to the entries' shape, and a flat ``[T]`` axis can
    silently cross-broadcast at ``T == 128``.
    """
    live = live_rows(rows.offsets, rows.positions.shape[0])
    entries = _window_entries_op(rows.positions, rows.bid, swa_begin)
    entries = F.where(
        F.unsqueeze(live, -1),
        entries,
        F.constant(-1, DType.int32, rows.positions.mapping),
    )
    return _leaf_slots_op(swa, F.unsqueeze(rows.bid, -1), entries, layer_swa)


def ragged_rows(
    row_offsets: Tensor, cache_lengths: Tensor, total: int | Dim
) -> RaggedRows:
    """Builds :class:`RaggedRows` once per forward step."""
    bid, local = row_ids(row_offsets, total)
    positions = absolute_positions(row_offsets, cache_lengths, total)
    b = row_offsets.shape[0] - 1
    reqs = F.range(0, b, 1, out_dim=b, dtype=DType.int32, device=bid.mapping)
    lengths = F.gather(
        F.cast(row_offsets, DType.int32), reqs + 1, axis=0
    ) - F.gather(F.cast(row_offsets, DType.int32), reqs, axis=0)
    return RaggedRows(row_offsets, bid, local, positions, lengths)


def position_rope_table(
    inv_freqs: np.ndarray, max_position: int, rows: int | Dim, ref: Tensor
) -> Tensor:
    """``[max_position + rows, 64]`` fp32 cosines then sines per position.

    The V2 store and attention kernels index the table by absolute
    position, so it spans every position the graph can carry -- plus
    ``rows``: a captured decode pads the row axis past
    ``row_offsets[-1]``, and the padded rows synthesize positions past
    the last request's start by their local row index, so the tail keeps
    their table reads in bounds. A pure function of constants, built
    in-graph like V2's ``freqs_cis_base``.

    Args:
        inv_freqs: The layer regime's fp32 inverse frequencies (host
            values, from :class:`~.rope.V41RotaryEmbedding`).
        max_position: The context bound.
        rows: The forward's token rows ``T``.
        ref: Any tensor of the forward, for the position axis' device
            and the constants' replicated mapping.
    """
    where = DeviceMapping.replicated(ref.mapping.mesh)
    inv = F.constant(inv_freqs, DType.float32, device=where)
    width = max_position + rows
    pos = F.range(
        0,
        width,
        1,
        out_dim=width,
        dtype=DType.int32,
        device=DeviceMapping.replicated(ref.mapping.mesh),
    )
    angles = F.outer(F.cast(pos, DType.float32), inv)
    return F.concat([F.cos(angles), F.sin(angles)], axis=-1)


class _GroupedWoA(Module[[Tensor], Tensor]):
    """``wo_a``: FP8 weight and compact E8M0 scales, grouped per output rows.

    ``[T, g * d] -> [T, g * r]``, group ``j`` applying rows ``j * r`` onward.
    The groups run as one block-diagonal MXFP8 launch on the FP8 weight,
    so decode neither dequantizes the weight nor reads it back at bf16:
    the checkpoint's own codes and scale bytes are the graph's operands.
    """

    weight: Tensor
    scale: Tensor

    def __init__(self, out_dim: int, in_dim: int, group_rank: int) -> None:
        super().__init__()
        self.group_rank = group_rank
        self.amax_floor = 1e-4
        self.weight = Tensor.zeros([out_dim, in_dim], dtype=DType.float8_e4m3fn)
        self.scale = Tensor.zeros(
            [out_dim // 32, in_dim // 32], dtype=DType.float8_e8m0fnu
        )

    def forward(self, o: Tensor) -> Tensor:
        return grouped_dense_op(
            o, self.weight, self.scale, self.group_rank, self.amax_floor
        )


class Attention(Module[..., Tensor]):
    """One V4.1 attention block: q path, SWA KV, sparse attention, output.

    ``compressor`` and ``indexer`` are assigned as ``attn.compressor``
    and ``attn.indexer``, where the frozen tree puts them (the block
    drives them; this module only holds them).

    Args:
        config: The resolved model config (geometry).
        layer_info: This layer's resolved facts (theta and YaRN factor
            per regime).
        mesh: The TP mesh; weights are placed on it at construction.
        compressor: The compressor, on compressed KV-source layers.
        indexer: The indexer, on layers that own one.
        rope: The checkpoint-wide rotated width and YaRN band.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_info: LayerInfo,
        mesh: DeviceMesh,
        compressor: Compressor | None = None,
        indexer: Indexer | None = None,
        *,
        rope: RopeParams,
        data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = layer_info.layer_id
        self.compress_ratio = layer_info.compress_ratio
        self.mesh = mesh
        self.data_parallel = data_parallel

        h = config.num_attention_heads
        d = config.head_dim
        self.n_heads = h
        self.head_dim = d
        self.q_lora_rank = config.q_lora_rank
        self.rope_head_dim = rope.rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.softmax_scale = d**-0.5
        self.window = config.sliding_window
        if self.window != WINDOW_ENTRIES or d != 512:
            raise ValueError(
                f"CSA2 attends {WINDOW_ENTRIES}-token windows of 512-wide "
                f"records, got window={self.window}, head_dim={d}"
            )
        self.index_topk = config.index_topk or 0
        if self.index_topk > MAX_COMPRESSED:
            raise ValueError(
                f"CSA2 attends at most {MAX_COMPRESSED} compressed entries, "
                f"got index_topk={self.index_topk}"
            )
        self.max_position = config.max_seq_len

        # Per-head sink, sharded by heads (8 per rank at TP8).
        self.attn_sink = Tensor.zeros([h], dtype=DType.float32)

        # ``wq_a`` and ``wkv`` read the same input, so the adapter stacks
        # them into one projection (``[wq_a; wkv]`` along N): one GEMM, not
        # two. The latent side stays whole under TP, so the stack is the
        # parts' own rows in order.
        self.wqa_wkv = MXFP8Linear(config.hidden_size, config.q_lora_rank + d)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = MXFP8Linear(config.q_lora_rank, h * d)
        self.kv_norm = RMSNorm(d, eps=config.rms_norm_eps)
        # ``wo_a`` keeps the checkpoint's own E4M3 codes and compact
        # E8M0 block scales -- the adapter never converts a byte -- and
        # the projection runs the grouped MXFP8 path (one block-diagonal
        # launch on SM100, per-group reference chains elsewhere), with
        # the activation quantized like every dense FP8 site.
        # ``wo_b`` likewise stays the FP8 MXFP8 projection.
        self.wo_a = _GroupedWoA(
            self.o_groups * self.o_lora_rank,
            h * d // self.o_groups,
            self.o_lora_rank,
        )
        self.wo_b = MXFP8Linear(
            self.o_groups * self.o_lora_rank, config.hidden_size
        )

        if compressor is not None:
            self.compressor = compressor
        if indexer is not None:
            self.indexer = indexer

        self.rope = rope_for_layer(layer_info, rope)
        self._apply_tp(mesh)

    def _apply_tp(self, mesh: DeviceMesh) -> None:
        """Declares the reference's TP split on this layer's weights.

        ``wq_b`` rows and ``attn_sink`` by heads, ``wo_a`` by group,
        ``wo_b`` row-parallel with the all-reduce after it; the latent
        side (``wqa_wkv``, the norms) stays whole.

        Under data parallelism every weight replicates whole (the rank
        runs all 64 heads on its own rows) and nothing reduces.
        """
        if mesh.num_devices <= 1 or self.data_parallel:
            return
        if self.o_groups % mesh.num_devices:
            raise ValueError("tensor parallel degree must divide o_groups")
        by_rows = DeviceMapping(mesh, (Sharded(0),))
        self.wq_b.shard(0, mesh)
        self.wo_b.shard(1, mesh)
        self.attn_sink = self.attn_sink.to(by_rows)
        self.wo_a.weight = self.wo_a.weight.to(by_rows)
        self.wo_a.scale = self.wo_a.scale.to(by_rows)

    def _q_path(self, q_a: Tensor, cos_sin: Tensor) -> Tensor:
        """``wq_b(q_norm(q_a))`` reshaped to heads, RoPE on the tail."""
        qr = self.q_norm(q_a)
        q = F.reshape(self.wq_b(qr), [-1, self.n_heads, self.head_dim])
        return apply_rope_tail(q, cos_sin, self.rope_head_dim)

    def _use_prefill(
        self,
        swa: PagedCacheValues,
        comp: PagedCacheValues | None,
        total: int | Dim,
        batch: int | Dim,
    ) -> Tensor:
        """The host prefill/decode flag, exactly as the V2 model picks it.

        A forward where some request brings more than
        ``_DECODE_MAX_QUERY_TOKENS`` query tokens runs the prefill
        attention if its lists cannot name more than ``PREFILL_ROWS``
        distinct slots; every other forward runs the decode attention.
        The op reads the flag at run time, so one graph serves both.
        """
        host = DeviceRef.CPU()
        tokens = F.reshape(F.shape_to_tensor([total]), [])
        batch_t = F.reshape(F.shape_to_tensor([batch]), [])
        longest = F.cast(F.reshape(swa.max_prompt_length, []), DType.int64)
        cached = F.cast(F.reshape(swa.max_cache_length, []), DType.int64)
        # A window-only layer (ratio 0) gets a placeholder compressed
        # leaf from ``layer_plan`` but selects nothing: its bound is
        # the window-only one, so the ratio must arrive as ``None`` --
        # ``prefill_rows_bound`` divides by the ratio, and 0 is not a
        # valid divisor.
        ratio = (
            None
            if comp is None or self.compress_ratio == 0
            else self.compress_ratio
        )
        named = _prefill_rows_bound_op(
            tokens,
            batch_t,
            longest,
            cached,
            ratio,
        )
        return F.logical_and(
            longest > F.constant(_DECODE_MAX_QUERY_TOKENS, DType.int64, host),
            named <= F.constant(PREFILL_ROWS, DType.int64, host),
        )

    def _index_list(
        self,
        swa: PagedCacheValues,
        comp: PagedCacheValues | None,
        layer_swa: Tensor,
        layer_comp: Tensor,
        rows: RaggedRows,
        swa_begin: Tensor,
        comp_indices: Tensor | None,
    ) -> Tensor:
        """Each row's ``[T, 640]`` absolute-slot list.

        The window slots of positions ``p, p - 1, ..., p - 127`` come
        first (``-1`` below the request's ``swa_begin`` floor, and
        whole rows of ``-1`` for padded capture rows), then the
        compressed slots of the selected entries. ``comp_indices`` holds
        entry (group) ids, ascending, ``-1``-padded -- the indexer's
        one contract -- passed straight to the leaf-slot map with no
        second ratio conversion: an entry id *is* its group id at
        either ratio, and ``leaf_slots`` maps it to
        ``block * slots + entry % slots`` once.
        """
        total = rows.positions.shape[0]
        window = window_list(swa, rows, swa_begin, layer_swa)
        none = F.constant(-1, DType.int32, rows.positions.mapping)
        if comp is None or comp_indices is None:
            return F.concat(
                [
                    window,
                    F.broadcast_to(none, [total, MAX_COMPRESSED]),
                ],
                axis=-1,
            )
        # The selection's own contract pads padded rows to ``-1`` (the
        # indexer neutralizes them); mask here too so a selection that
        # does not can never gather a foreign page. The request axis is
        # one per row (``[T, 1]``) for the same reason as the window
        # list: a flat ``[T]`` axis can silently cross-broadcast at
        # ``T == 128``.
        live = live_rows(rows.offsets, total)
        comp_indices = F.where(
            F.unsqueeze(live, -1),
            comp_indices,
            F.constant(-1, DType.int32, rows.positions.mapping),
        )
        comp_slots = _leaf_slots_op(
            comp, F.unsqueeze(rows.bid, -1), comp_indices, layer_comp
        )
        lists = [window, comp_slots]
        pad = MAX_COMPRESSED - int(comp_indices.shape[-1])
        if pad:
            lists.append(F.broadcast_to(none, [total, pad]))
        return F.concat(lists, axis=-1)

    def _output_projection(self, o: Tensor) -> Tensor:
        """``[T, heads, head_dim]`` -> ``[T, hidden]``: the grouped path.

        The inverse rotation runs as an explicit chain first (the
        MXFP8 API has no fused rope-quantize kwargs). ``wo_a`` then
        quantizes the rotated rows like every dense FP8 site and
        projects each group on its own rows through the grouped MXFP8
        op -- one block-diagonal launch on SM100, whose sharding rule
        pairs each rank's K-slice with its weight shard; ``wo_b``
        quantizes its own input. Row-parallel partial sums: one
        all-reduce completes it. Under data parallelism the projection
        is whole per rank and its output carries the rows' own
        placement -- nothing to reduce.
        """
        flat = o.reshape([o.shape[0], self.n_heads * self.head_dim])
        grouped = self.wo_a(flat)
        out = self.wo_b(grouped)
        if out.is_distributed and any(
            isinstance(p, Partial) for p in out.placements
        ):
            out = F.allreduce_sum(out)
        return out

    def project(self, h_normed: Tensor) -> tuple[Tensor, Tensor]:
        """``(wq_a(x), wkv(x))``, sliced out of the one stacked projection.

        Positive bounds only (the note in ``rope.py`` and ISSUES Issue 34):
        the upper bound is the stacked width, never a negative index.
        """
        out = self.wqa_wkv(h_normed)
        width = int(out.shape[-1])
        return (
            out[:, : self.q_lora_rank],
            out[:, self.q_lora_rank : width],
        )

    def forward(
        self,
        h_normed: Tensor,
        kv: tuple[PagedCacheValues, PagedCacheValues | None],
        rows: RaggedRows,
        layer_idx: tuple[Tensor, Tensor],
        swa_begin: Tensor,
        comp_indices: Tensor | None = None,
        *,
        rope_table: Tensor | None = None,
        projected: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """One attention forward over a ragged batch of ``T`` tokens.

        Args:
            h_normed: ``[T, hidden]`` bf16, already ``attn_norm``-ed.
            kv: This batch's ``(swa, comp)`` paged leaves.
            rows: The forward's ragged bookkeeping.
            layer_idx: This layer's slots ``(layer_swa, layer_comp)`` in
                the two leaves, uint32 host scalars as graph values,
                never constants baked into a shared body.
            swa_begin: ``[batch]`` int32 per-request window lower bounds
                in absolute positions; zeros without one.
            comp_indices: ``[T, index_topk]`` int32 compressed-entry
                selection, ascending, ``-1`` padded. ``None`` marks a
                window-only layer.
            rope_table: The stores' position-indexed cos/sin table; the
                caller stages one per regime per step. ``None`` builds
                it here.
            projected: The stacked projection's ``(q_a, kv_a)`` pair,
                when the caller already ran :meth:`project` (the block's
                selection layers, which slice the q-lora out for the
                indexer). The exact pair is consumed verbatim.

        Returns:
            ``[T, hidden]`` bf16, all-reduced across the TP mesh.
        """
        swa, comp = kv
        layer_swa, layer_comp = layer_idx
        total = h_normed.shape[0]

        # One cos/sin for the query and the inverse rotation.
        cos_sin = self.rope.cos_sin(rows.positions)

        if projected is None:
            projected = self.project(h_normed)
        q_a, kv_a = projected
        q = self._q_path(q_a, cos_sin)
        if rope_table is None:
            rope_table = position_rope_table(
                self.rope.inv_freqs(),
                self.max_position,
                total,
                rows.positions,
            )
        # The flag reads the leaves' own metadata; batch is the row
        # offsets' request count.
        # The window record: the store op owns the tail rotation, the
        # MXFP8 quantize and the paged insert; the latent arrives normed
        # and unrotated. Every layer, including window-only, owns its
        # sliding-window slot.
        _store_window_op(
            swa.kv_blocks,
            self.kv_norm(kv_a),
            rows.positions,
            rope_table,
            window_store_slots(swa, rows, layer_swa),
            block_stride=swa.page_stride,
        )
        idx = self._index_list(
            swa,
            comp,
            layer_swa,
            layer_comp,
            rows,
            swa_begin,
            comp_indices,
        )
        o = _csa2_attend_op(
            q,
            idx,
            self.attn_sink,
            swa.kv_blocks,
            None if comp is None else comp.kv_blocks,
            scale=self.softmax_scale,
            window_stride=swa.page_stride,
            comp_stride=None if comp is None else comp.page_stride,
            compressed_rows=self.index_topk,
            prefill=self._use_prefill(swa, comp, total, rows.batch),
            max_distinct_rows=PREFILL_ROWS,
        )
        # The rule-less ``F.functional`` wrapper leaves the attention's
        # output Unknown-placed; every rank computed the same rows'
        # whole heads, so the output belongs on the queries' mapping
        # before anything reshapes it (a TP decode would otherwise
        # treat the local head count as global and mix rows).
        o = o.rebind_mapping(q.mapping)
        o = apply_rope_tail(o, cos_sin, self.rope_head_dim, inverse=True)
        flat = F.reshape(o, [-1, self.n_heads * self.head_dim])
        return self._output_projection(flat)
