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
and the FP8 row-parallel wo_b.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from max import functional as F
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.kv_cache import KVCacheInputsPerDevice
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
    INDEX_WIDTH,
    MAX_COMPRESSED,
    PREFILL_ROWS,
    ROPE_Q_HEADS,
    WINDOW_ENTRIES,
    WindowStore,
    csa2_attention,
    csa2_attention_mxfp8,
    leaf_slots,
    norm_quantize_q_store_kv,
    records_per_page,
    store_window_kv,
)
from ..layer_resolver import LayerInfo
from ..model_config import position_capacity
from .mxfp8_linear import (
    MXFP8Linear,
    Mxfp8Rows,
    block_diagonal_woa_supported,
    grouped_dense_op,
    grouped_dense_quantized_op,
    norm_quantize_store_supported,
)
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
    "mask_padded_positions",
    "position_rope_table",
    "ragged_rows",
    "row_ids",
]

# A forward whose requests each bring at most this many query tokens runs
# the decode attention, which reads the records in place; a longer one
# dequantizes the records it names once, first (the registered V2 model's
# split, host-flag driven inside ``attention.csa2``).
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


def _csa2_index_list(
    positions: TensorValue,
    requests: TensorValue,
    valid_from: TensorValue,
    swa: KVCacheInputsPerDevice[TensorValue, BufferValue],
    layer_swa: TensorValue,
    selected: TensorValue | None,
    comp: KVCacheInputsPerDevice[TensorValue, BufferValue] | None,
    layer_comp: TensorValue,
    ratio: int,
) -> TensorValue:
    """Each query's ``[T, 640]`` absolute-slot list, built in one launch.

    Main's ``attention.csa2.index_list`` (V2's ``index_list``, whose layer
    arguments are baked constants): the window slots of positions
    ``p, p - 1, ..., p - 127`` first, then the compressed slots of the
    selected groups, ``-1`` for none. A position below 0 or ``valid_from``
    and a negative selection name ``-1``, so the caller masks padded rows
    through ``positions`` and ``selected``. ``selected`` holds the raw
    positions of the groups' closing tokens, which the op floor-divides by
    ``ratio``; ``None`` is a window-only list.
    """
    window_blocks = swa.kv_blocks
    values = [
        ops.cast(positions, DType.int32),
        ops.cast(requests, DType.int32),
        ops.cast(valid_from, DType.int32),
        ops.cast(swa.lookup_table, DType.uint32),
    ]
    layers = [ops.cast(layer_swa, DType.uint32)]
    parameters: dict[str, bool | int | float | str | DType] = {
        "window_page": records_per_page(window_blocks),
        "window_layers": int(window_blocks.shape[2]),
        "has_floor": True,
    }
    name = "attention.csa2.index_list.window_only"
    if selected is not None:
        assert comp is not None
        name = "attention.csa2.index_list"
        parameters["comp_page"] = records_per_page(comp.kv_blocks)
        parameters["comp_layers"] = int(comp.kv_blocks.shape[2])
        parameters["ratio"] = ratio
        values += [
            ops.cast(comp.lookup_table, DType.uint32),
            ops.cast(selected, DType.int32),
        ]
        layers.append(ops.cast(layer_comp, DType.uint32))
    return ops.custom(
        name,
        device=positions.device,
        values=[*values, *layers],
        out_types=[
            TensorType(
                DType.int32, [positions.shape[0], INDEX_WIDTH], positions.device
            )
        ],
        parameters=parameters,
    )[0].tensor


_index_list_op = F.functional(_csa2_index_list)
_store_window_op = F.functional(store_window_kv)


def _norm_quantize_store(
    x: TensorValue,
    q_weight: TensorValue,
    kv_weight: TensorValue,
    cache: BufferValue,
    positions: TensorValue,
    rope_table: TensorValue,
    slots: TensorValue,
    block_stride: TensorValue,
    epsilon: float,
    q_cols: int,
) -> tuple[TensorValue, TensorValue]:
    # ``WindowStore`` is reboxed here because ``F.functional`` accepts flat operands.
    window = WindowStore(cache, positions, rope_table, slots, block_stride)
    return norm_quantize_q_store_kv(
        x, q_weight, kv_weight, epsilon, q_cols, window
    )


_norm_quantize_store_op = F.functional(_norm_quantize_store)
_csa2_attend_op = F.functional(csa2_attention)
_csa2_attend_mxfp8_op = F.functional(csa2_attention_mxfp8)


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
    live_positions: Tensor | None = None
    """:func:`mask_padded_positions` of ``positions``, when the trunk staged
    it once for the step; ``None`` has each layer derive it."""

    @property
    def batch(self) -> int | Dim:
        """The number of sequences the rows cover."""
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


def mask_padded_positions(row_offsets: Tensor, positions: Tensor) -> Tensor:
    """``[total]`` int32 positions, capture-padding rows set to ``-1``."""
    live = live_rows(row_offsets, positions.shape[0])
    return F.where(
        live, positions, F.constant(-1, DType.int32, positions.mapping)
    )


def live_positions(rows: RaggedRows) -> Tensor:
    """``rows.positions`` with the padded rows at ``-1``, once per step if staged."""
    if rows.live_positions is not None:
        return rows.live_positions
    return mask_padded_positions(rows.offsets, rows.positions)


def window_store_slots(
    swa: PagedCacheValues, rows: RaggedRows, layer_swa: Tensor
) -> Tensor:
    """Use slot ``-1`` for padding; the store has no offsets guard.

    Row positions still reach the store for RoPE when a slot is masked.
    """
    return _leaf_slots_op(swa, rows.bid, live_positions(rows), layer_swa)


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
    inv_freqs: np.ndarray,
    max_position: int,
    max_rows: int,
    ref: Tensor,
) -> Tensor:
    """Build the static RoPE table, including speculative and padded rows."""
    where = DeviceMapping.replicated(ref.mapping.mesh)
    inv = F.constant(inv_freqs, DType.float32, device=where)
    width = max_position + max_rows
    pos = F.range(0, width, 1, out_dim=width, dtype=DType.int32, device=where)
    angles = F.outer(F.cast(pos, DType.float32), inv)
    return F.concat([F.cos(angles), F.sin(angles)], axis=-1)


class _GroupedWoA(Module[..., Tensor]):
    """One block-diagonal quantized ``[T, g*d] -> [T, g*r]`` launch.

    Group ``j`` writes rows from ``j*r`` onward without dequantizing weights.
    """

    weight: Tensor
    scale: Tensor

    def __init__(self, out_dim: int, in_dim: int, group_rank: int) -> None:
        super().__init__()
        self.group_rank = group_rank
        self.weight = Tensor.zeros([out_dim, in_dim], dtype=DType.float8_e4m3fn)
        self.scale = Tensor.zeros(
            [out_dim // 32, in_dim // 32], dtype=DType.float8_e8m0fnu
        )

    def forward(
        self, o: Tensor, rope: tuple[Tensor, int] | None = None
    ) -> Tensor:
        return grouped_dense_op(
            o,
            self.weight,
            self.scale,
            self.group_rank,
            rope=rope,
        )

    def forward_quantized(self, o: Mxfp8Rows) -> Tensor:
        """:meth:`forward` on rows the attention quantized and rotated back."""
        return grouped_dense_quantized_op(
            o.codes, o.scales, self.weight, self.scale, self.group_rank
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
        # The table spans the positions a speculative verify can name --
        # max_seq_len plus the KV pool's speculative slack -- plus the
        # padded captured rows' tail (max_rows), as the graph-API model's
        # position_capacity sizes its own tables.
        self.max_position = position_capacity(config)
        self.max_rows = config.max_batch_rows

        # Per-head sink, sharded by heads (8 per rank at TP8).
        self.attn_sink = Tensor.zeros([h], dtype=DType.float32)

        # ``wq_a`` and ``wkv`` read the same input, so the adapter stacks
        # them into one projection (``[wq_a; wkv]`` along N): one GEMM, not
        # two. The latent side stays whole under TP, so the stack is the
        # parts' own rows in order.
        self.wqa_wkv = MXFP8Linear(config.hidden_size, config.q_lora_rank + d)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = MXFP8Linear(config.q_lora_rank, h * d)
        # The fused q/kv norm launch takes one epsilon, so both norms read
        # the config's.
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

    def ropes_query_in_attention(self, x: Tensor) -> bool:
        """Only 64 Q heads match the fused kernel's fixed RoPE tile."""
        return self.n_heads == ROPE_Q_HEADS and self._attends_to_mxfp8(x)

    def _q_path(self, q_a: Tensor, cos_sin: Tensor) -> Tensor:
        """``wq_b(q_norm(q_a))`` reshaped to heads, RoPE on the tail."""
        qr = self.q_norm(q_a)
        if self.ropes_query_in_attention(q_a):
            q = self.wq_b(qr)
        else:
            q = self.wq_b.forward_rope_tail(qr, cos_sin, self.head_dim)
        return F.reshape(q, [-1, self.n_heads, self.head_dim])

    def fuses_query(self, projected: Tensor) -> bool:
        """Whether :meth:`fused_query` applies to this stacked projection."""
        return norm_quantize_store_supported(
            projected.mesh.devices[0], self.q_lora_rank, self.head_dim
        )

    def fused_query(
        self,
        projected: Tensor,
        swa: PagedCacheValues,
        rows: RaggedRows,
        layer_swa: Tensor,
    ) -> tuple[Mxfp8Rows, Tensor]:
        """Fuse q-lora norm, quantization, and KV window storage."""
        # The memory planner does not charge these table bytes to the KV budget.
        rope_table = position_rope_table(
            self.rope.inv_freqs(),
            self.max_position,
            self.max_rows,
            rows.positions,
        )
        codes, scales = _norm_quantize_store_op(
            projected,
            self.q_norm.weight,
            self.kv_norm.weight,
            swa.kv_blocks,
            rows.positions,
            rope_table,
            window_store_slots(swa, rows, layer_swa),
            swa.page_stride,
            self.q_norm.eps,
            self.q_lora_rank,
        )
        qr = Mxfp8Rows(
            codes.rebind_mapping(projected.mapping),
            scales.rebind_mapping(projected.mapping),
        )
        if self.ropes_query_in_attention(projected):
            q = self.wq_b.forward_quantized(qr)
        else:
            q = self.wq_b.forward_quantized_rope_tail(
                qr, self.rope.cos_sin(rows.positions), self.head_dim
            )
        return qr, F.reshape(q, [-1, self.n_heads, self.head_dim])

    def _use_prefill(
        self,
        swa: PagedCacheValues,
        comp: PagedCacheValues | None,
        total: int | Dim,
        batch: int | Dim,
    ) -> Tensor:
        """Prefill needs a long prompt and bounded rows; one graph reads this flag at runtime."""
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
        positions = live_positions(rows)
        none = F.constant(-1, DType.int32, rows.positions.mapping)
        selected = None
        if comp is not None and comp_indices is not None:
            ratio = self.compress_ratio
            selected = F.where(
                F.unsqueeze(positions >= 0, -1) & (comp_indices >= 0),
                comp_indices * ratio + (ratio - 1),
                none,
            )
        return _index_list_op(
            positions,
            rows.bid,
            swa_begin,
            swa,
            layer_swa,
            selected,
            comp,
            layer_comp,
            self.compress_ratio,
        )

    def _folds_inverse_rope(self, o: Tensor) -> bool:
        """Whether inverse RoPE can be folded into the grouped projection."""
        tp = 1 if self.data_parallel else self.mesh.num_devices
        return self.o_groups // tp >= 2 and block_diagonal_woa_supported(
            o.mesh.devices[0],
            self.n_heads * self.head_dim // self.o_groups,
            self.o_lora_rank,
        )

    def _attends_to_mxfp8(self, q: Tensor) -> bool:
        """Whether attention can produce ``wo_a``'s MXFP8 input directly."""
        return (
            self.data_parallel or self.mesh.num_devices == 1
        ) and self._folds_inverse_rope(q)

    def _output_projection(
        self, o: Tensor, rope: tuple[Tensor, int] | None = None
    ) -> Tensor:
        """Run grouped ``wo_a`` and ``wo_b``, folding inverse RoPE when supported."""
        flat = o.reshape([o.shape[0], self.n_heads * self.head_dim])
        return self._project_grouped(self.wo_a(flat, rope))

    def _project_grouped(self, grouped: Tensor) -> Tensor:
        """``wo_b`` over ``wo_a``'s output, completing the output projection."""
        out = self.wo_b(grouped)
        if out.is_distributed and any(
            isinstance(p, Partial) for p in out.placements
        ):
            out = F.allreduce_sum(out)
        return out

    def project(
        self, h_normed: Tensor, h_rows: Mxfp8Rows | None = None
    ) -> Tensor:
        """Project the normalized state through the stacked q/kv weights."""
        return (
            self.wqa_wkv(h_normed)
            if h_rows is None
            else self.wqa_wkv.forward_quantized(h_rows)
        )

    def split_projection(self, projected: Tensor) -> tuple[Tensor, Tensor]:
        """Split the stacked projection into q-lora and KV latents."""
        width = int(projected.shape[-1])
        return (
            projected[:, : self.q_lora_rank],
            projected[:, self.q_lora_rank : width],
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
        projected: Tensor | None = None,
        query: Tensor | None = None,
    ) -> Tensor:
        """Run attention over ragged rows, reusing staged projections when provided."""
        swa, comp = kv
        layer_swa, layer_comp = layer_idx
        total = h_normed.shape[0]

        # One cos/sin for the query and the inverse rotation.
        cos_sin = self.rope.cos_sin(rows.positions)

        if query is not None:
            # ``fused_query`` already stored KV; this path skips that write.
            q = query
        else:
            if projected is None:
                projected = self.project(h_normed)
            q_a, kv_a = self.split_projection(projected)
            q = self._q_path(q_a, cos_sin)
            if rope_table is None:
                rope_table = position_rope_table(
                    self.rope.inv_freqs(),
                    self.max_position,
                    self.max_rows,
                    rows.positions,
                )
            # The window record: the store op owns the tail rotation, the
            # MXFP8 quantize and the paged insert; the latent arrives
            # normed and unrotated. Every layer, including window-only,
            # owns its sliding-window slot.
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
        kv_comp = None if comp is None else comp.kv_blocks
        comp_stride = None if comp is None else comp.page_stride
        prefill = self._use_prefill(swa, comp, total, rows.batch)
        if self._attends_to_mxfp8(q):
            # The attention kernel writes ``wo_a``'s input itself: rotated
            # back and quantized, the bytes the separate rope-quantize
            # launch below would produce from its bf16 output.
            x_q, x_scales = _csa2_attend_mxfp8_op(
                q,
                idx,
                self.attn_sink,
                swa.kv_blocks,
                kv_comp,
                cos_sin,
                scale=self.softmax_scale,
                window_stride=swa.page_stride,
                comp_stride=comp_stride,
                compressed_rows=self.index_topk,
                prefill=prefill,
                max_distinct_rows=PREFILL_ROWS,
                rope_q=self.ropes_query_in_attention(q),
            )
            return self._project_grouped(
                self.wo_a.forward_quantized(
                    Mxfp8Rows(
                        x_q.rebind_mapping(q.mapping),
                        x_scales.rebind_mapping(q.mapping),
                    )
                )
            )
        o = _csa2_attend_op(
            q,
            idx,
            self.attn_sink,
            swa.kv_blocks,
            kv_comp,
            scale=self.softmax_scale,
            window_stride=swa.page_stride,
            comp_stride=comp_stride,
            compressed_rows=self.index_topk,
            prefill=prefill,
            max_distinct_rows=PREFILL_ROWS,
        )
        # The rule-less ``F.functional`` wrapper leaves the attention's
        # output Unknown-placed; every rank computed the same rows'
        # whole heads, so the output belongs on the queries' mapping
        # before anything reshapes it (a TP decode would otherwise
        # treat the local head count as global and mix rows).
        o = o.rebind_mapping(q.mapping)
        rope = None
        if self._folds_inverse_rope(o):
            rope = (cos_sin, self.head_dim)
        else:
            o = apply_rope_tail(o, cos_sin, self.rope_head_dim, inverse=True)
        flat = F.reshape(o, [-1, self.n_heads * self.head_dim])
        return self._output_projection(flat, rope)
