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
"""The V4.1 lightning indexer: two-level top-k over compressed entries.

The registered-op route: every store, score and selection runs on
main's ``mo.csa.indexer.*`` graph ops (``...deepseekV41.layers.
indexer``), bound through :mod:`.index_ops` -- there is no dense
``[T, cap]`` scorer route and no replay planning.

The block group kinds this module serves (``block_group``):

* ``full_r{ratio}`` / ``full_r{ratio}_cand`` -- the KV-source **owners**.
  The owner stores every stream: its index-K leaf and its compressed-KV
  leaf (through ``store_compressed_kv``, which owns the group-first tail
  rotation and the NVFP4 quantize), then scores its queries against the
  index leaf. ``full_r{ratio}_cand`` additionally keeps the candidate
  pool for the reindex layers below: its best blocks come out of the
  scorer itself (``indexer_select_block_topk``).
* ``reindex_r{ratio}`` -- their own ``wq_b`` / ``weights_proj`` score
  against the candidate source's index-K leaf, masked to the pool's kept
  blocks (:func:`indexer_select_masked`); a row that sees at most
  ``index_topk`` entries selects all of them, the short-context bypass
  the kernel applies per row.

Which role a layer plays is derived from its resolved
:class:`~..layer_resolver.LayerInfo` (``attention_mode``,
``candidate_role``, ``compress_ratio``), never matched off the kind
string alone: the resolver accepts layouts the published checkpoint
does not produce (a ratio-1 owner beside the candidate source, a
ratio-2 candidate source, a ratio-2 reindex row), and those must store
and score at their own ratio rather than fall through the published
three names. ``kind`` is cross-checked against the same facts at
construction, so no layout reaches ``forward`` half-derived.

Selections are **entry (group) ids**, ascending, ``-1``-padded, ``[T,
index_topk]`` int32 -- the one contract every consumer reads. The
attention maps them to leaf slots exactly once
(:meth:`~.attention.Attention._index_list`); no second ratio
conversion happens anywhere.

The global write floor survives as a store mask only: a request
replaying a prefix hit holds the entries below its
``global_write_floor`` in shared pages, which another reader may be
attending to, so those rows' stores take ``-1`` slots and never
overwrite them (no replay plan is computed). A floor of 0 keeps every
entry.

Parameters: ``wq_b`` (MXFP8: fp8 weight + E8M0 scales) and
``weights_proj`` (bf16) at every index source; ``wk`` and ``k_norm``
(bf16) at the owners only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max import functional as F
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn import Module
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.tensor import Tensor
from max.tensor.sharding import DeviceMesh

from ...deepseekV41.layers.csa2 import store_compressed_kv
from ..layer_resolver import LayerInfo
from .attention import RaggedRows, _leaf_slots_op, position_rope_table
from .candidate_pool import position_order
from .index_ops import (
    indexer_k_store,
    indexer_q_rope_quant,
    indexer_select,
    indexer_select_block_topk,
    indexer_select_masked,
)
from .mxfp8_linear import MXFP8Linear
from .norm import RMSNorm
from .rope import RopeParams, rope_for_layer

if TYPE_CHECKING:
    from ..model_config import DeepseekV41Config

__all__ = ["Indexer"]

_store_compressed_op = F.functional(store_compressed_kv)


class _Weight(Module[..., None]):
    """A bare ``.weight`` leaf: the checkpoint path
    ``...indexer.{wk,weights_proj}.weight`` maps attribute -> sub-module
    -> ``weight``, so each parameter is a one-attribute module.
    """

    def __init__(self, shape: list[int], dtype: DType) -> None:
        super().__init__()
        self.weight = Tensor.zeros(shape, dtype=dtype)


def _index_key_store_slots_rows(
    leaf: PagedCacheValues,
    bid: TensorValue,
    positions: TensorValue,
    offsets: TensorValue,
    *,
    ratio: int,
    cap: int,
    floor: TensorValue | None,
) -> TensorValue:
    """``[T]`` int32 index-key store slots, ``-1`` to skip a row.

    One traced graph op, the ``leaf_slots`` pattern: row ``t`` at
    position ``p`` names key ``p // ratio`` -- the ``indexer_k_store``
    op keeps only its group-closing rows itself, so the slot only
    places the survivors: ``block * keys_per_page + key %
    keys_per_page`` through the request's lookup table. A row whose
    entry sits below the replay floor (already in shared pages), or a
    padded row past ``row_offsets[-1]``, takes ``-1`` and never stores.

    The lookup-table gather runs for every physical row, so a row that
    does not keep clamps its key to a safe page before the gather: a
    padded position crossing a page boundary (one live row at position
    127, ratio 1; a padded row closing a fresh ratio-2 group) would
    otherwise gather past its request's allocated pages and out of the
    lookup table -- before any ``-1`` skip path runs.
    """
    device = positions.device
    t = positions.shape[0]
    keys_per_page = int(leaf.kv_blocks.shape[3])
    # Padded rows synthesize positions past the batch; their entry id
    # would leave the leaf, so clamp it to the static entry bound.
    key = ops.floor_div(
        ops.min(
            ops.cast(positions, DType.int32),
            ops.constant(cap * ratio - 1, DType.int32, device),
        ),
        ratio,
    )
    row_index = ops.range(0, t, 1, out_dim=t, dtype=DType.int32, device=device)
    off32 = ops.cast(offsets, DType.int32)
    last = off32[-1]
    keep = row_index < last
    if floor is not None:
        floor_entries = ops.floor_div(
            ops.gather(
                ops.cast(floor, DType.int32),
                ops.cast(bid, DType.int32),
                axis=0,
            ),
            ratio,
        )
        keep = ops.logical_and(keep, key >= floor_entries)
    # The gather executes for every row: a dropped row clamps its key
    # to page 0, which every request's lookup table owns. int32 slots
    # throughout: the store op casts them to int64 itself, and slot ids
    # stay far below 2**31 at every real cache size.
    safe_key = ops.where(keep, key, ops.constant(0, DType.int32, device))
    page = ops.floor_div(safe_key, keys_per_page)
    block = ops.gather_nd(
        leaf.lookup_table,
        ops.stack([ops.cast(bid, DType.int32), page], axis=-1),
    )
    slot = ops.cast(block, DType.int32) * keys_per_page + (
        safe_key % keys_per_page
    )
    return ops.where(keep, slot, ops.constant(-1, DType.int32, device))


_index_key_store_slots_op = F.functional(_index_key_store_slots_rows)


def index_key_store_slots(
    leaf: PagedCacheValues,
    rows: RaggedRows,
    *,
    ratio: int,
    cap: int,
    floor: Tensor | None,
) -> Tensor:
    """``rows``'s index-key store slots, as one traced graph op."""
    return _index_key_store_slots_op(
        leaf,
        rows.bid,
        rows.positions,
        rows.offsets,
        ratio=ratio,
        cap=cap,
        floor=floor,
    )


class Indexer(Module[..., tuple[Tensor, Tensor | None]]):
    """Scores and selects compressed entries for a ragged batch.

    Built by the block as ``Indexer(config, mesh, kind, layer_info=...,
    rope=..., max_position=...)``: ``kind`` is the block group name,
    ``layer_info`` and ``rope`` are the same resolver row and
    ``RopeParams`` the layer's ``Attention`` takes (so the rotations
    cannot drift), and ``max_position`` bounds the RoPE table and the
    entry cap. ``forward`` returns ``comp_indices`` and, at the
    candidate source, the pool of kept candidate blocks.

    Raises:
        ValueError: If ``kind`` disagrees with ``layer_info``'s resolved
            facts, or the geometry leaves the MXFP4 kernels' head layout
            (``index_head_dim`` 128, ``rope_head_dim`` 64).
    """

    wq_b: MXFP8Linear
    """``[heads * 128, q_lora_rank]`` MXFP8 query projection."""
    weights_proj: _Weight
    """``[heads, hidden]`` bf16 per-head scoring weights (``.weight``)."""
    wk: _Weight
    """``[index_head_dim, head_dim]`` bf16 key projection; owners only."""
    k_norm: RMSNorm
    """``[index_head_dim]`` bf16 learned scale (``.weight``); owners only."""

    def __init__(
        self,
        config: DeepseekV41Config,
        mesh: DeviceMesh,
        kind: str,
        *,
        layer_info: LayerInfo,
        rope: RopeParams,
        max_position: int,
        candidate_block_size: int | None = None,
    ) -> None:
        super().__init__()
        if config.index_head_dim != 128 or rope.rope_head_dim != 64:
            raise ValueError(
                "the MXFP4 indexer kernels need index_head_dim 128 and "
                f"rope_head_dim 64, got {config.index_head_dim} and "
                f"{rope.rope_head_dim}"
            )
        self.config = config
        self.mesh = mesh
        self.kind = kind

        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk or 0
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = (
            config.candidate_block_size
            if candidate_block_size is None
            else candidate_block_size
        )
        self.rope_head_dim = rope.rope_head_dim
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        # HF :556: ``weights_proj(x) * (softmax_scale * n_heads**-0.5)``
        # -- both factors, unlike the registered V2 model, which folds
        # only ``head_dim ** -0.5`` into the kernel's weight scale. The
        # kernel takes the scale as a caller-supplied parameter, so the
        # HF-faithful value is the one the independent reference
        # validates against.
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scale = self.softmax_scale * self.n_heads**-0.5

        self.wq_b = MXFP8Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = _Weight(
            [self.n_heads, self.hidden_size], DType.bfloat16
        )
        self.owns_k = layer_info.attention_mode == "Full"
        self.is_candidate_source = layer_info.candidate_role == "source"
        self.uses_candidates = layer_info.attention_mode == "Reindex"
        self.ratio = layer_info.compress_ratio
        # The kind string must say the same thing as the facts above:
        # it names the shared subgraph body the block drives, and a
        # mismatch means the caller built the wrong layer's indexer.
        group = (
            f"full_r{self.ratio}_cand"
            if self.is_candidate_source
            else f"full_r{self.ratio}"
            if self.owns_k
            else f"reindex_r{self.ratio}"
        )
        if kind != group:
            raise ValueError(
                f"block group {kind!r} does not match the layer's "
                f"resolved facts ({group!r}); the indexer serves the "
                "Full and Reindex index-source layers"
            )

        if self.owns_k:
            self.wk = _Weight([self.head_dim, config.head_dim], DType.bfloat16)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = rope_for_layer(layer_info, rope)
        # The static position bound: the fused path's RoPE table spans
        # it, and it caps every entry id the graph can name.
        self.max_position = max_position
        self.cap = -(-max_position // self.ratio)
        # A consumer row keeps its newest block, which may hold a
        # single visible entry, so the other kept blocks must cover the
        # rest of ``index_topk``.
        if (self.is_candidate_source or self.uses_candidates) and (
            (self.candidate_topk_blocks - 1) * self.candidate_block_size + 1
            < self.index_topk
        ):
            raise ValueError(
                f"candidate_topk_blocks={self.candidate_topk_blocks} "
                f"blocks of {self.candidate_block_size} entries cannot "
                f"hold index_topk={self.index_topk} entries when the "
                "newest kept block holds one"
            )

    def _k_slots(
        self,
        leaf: PagedCacheValues,
        rows: RaggedRows,
        floor: Tensor | None,
    ) -> Tensor:
        """``[T]`` int32 index-key store slots, ``-1`` to skip a row."""
        return index_key_store_slots(
            leaf,
            rows,
            ratio=self.ratio,
            cap=self.cap,
            floor=floor,
        )

    def _comp_slots(
        self,
        leaf: PagedCacheValues,
        rows: RaggedRows,
        layer_comp: Tensor,
        latent_valid: Tensor,
        floor: Tensor | None,
    ) -> Tensor:
        """``[T]`` compressed-record store slots, ``-1`` to skip a row.

        Entry ``p // ratio`` on the row that closes its group; the store
        op owns the group-first rotation and the NVFP4 quantize. A row
        that closes no group, a row whose entry sits below the replay
        floor, and a padded row past ``row_offsets[-1]`` take ``-1`` and
        leave their slots untouched.
        """
        device = rows.offsets.mapping
        entry = rows.positions // self.ratio
        keep = latent_valid
        if floor is not None:
            floor_entries = F.gather(floor, rows.bid, axis=0) // self.ratio
            keep = keep & (entry >= floor_entries)
        row_index = F.range(
            0,
            entry.shape[0],
            1,
            out_dim=entry.shape[0],
            dtype=DType.int32,
            device=device,
        )
        keep = keep & (row_index < F.cast(rows.offsets[-1], DType.int32))
        entry = F.where(keep, entry, F.constant(-1, DType.int32, device))
        return _leaf_slots_op(leaf, rows.bid, entry, layer_comp)

    def forward(
        self,
        x: Tensor,
        qr: Tensor,
        latent: Tensor | None,
        latent_valid: Tensor | None,
        rows: RaggedRows,
        leaves: dict[str, PagedCacheValues],
        layer_idx: dict[str, Tensor],
        *,
        pool: Tensor | None = None,
        global_write_floor: Tensor | None = None,
        rope_table: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """One indexer forward over a ragged batch of ``T`` tokens.

        Args:
            x: ``[T, hidden]`` bf16 block input.
            qr: ``[T, q_lora_rank]`` the attention's shared low-rank
                query (``q_norm(wq_a(x))``), not recomputed here.
            latent: ``[T, head_dim]`` bf16 pre-RoPE compressor latents
                (owners only; ``None`` at reindex layers, which own no
                compressor and store nothing).
            latent_valid: ``[T]`` bool publication mask; ``None``
                where ``latent`` is.
            rows: The forward's ragged bookkeeping.
            leaves: The paged leaves this layer touches, keyed by leaf
                name (``comp``, ``idx``; ``idx_source`` at reindex
                layers).
            layer_idx: This layer's indices in the leaves, as graph
                values (uint32 CPU scalars) keyed like ``leaves``.
            pool: At ``reindex_r1`` layers: the kept candidate blocks
                the candidate source published, ``[T,
                candidate_topk_blocks]`` int32 block ids, ``-1``
                padded.
            global_write_floor: ``[B]`` int32 per-request token bound
                below which the index-K and compressed leaves are
                already cached (prefix-cache replay); their entries are
                not rewritten. ``None`` stores everything.
            rope_table: The stores' position-indexed cos/sin table; the
                caller stages one per regime per step and shares it with
                the attention. ``None`` builds it here.

        Returns:
            ``(comp_indices, pool)``: ``[T, index_topk]`` int32 entry
            ids, ascending, ``-1`` padded; ``pool`` is the candidate
            source's kept blocks (block ids, ``-1`` padded), ``None``
            elsewhere.
        """
        t = x.shape[0]
        positions = rows.positions
        if rope_table is None:
            rope_table = position_rope_table(
                self.rope.inv_freqs(), self.max_position, t, x
            )

        # --- Stores, in the pinned order: the compressed record first,
        # then the index key (both read the same pre-RoPE latent, which
        # no store mutates). An owner whose whole group is still open
        # (latent None) writes nothing.
        if self.owns_k and latent is not None:
            assert latent_valid is not None
            comp = leaves["comp"]
            _store_compressed_op(
                comp.kv_blocks,
                latent,
                positions,
                rope_table,
                self._comp_slots(
                    comp,
                    rows,
                    layer_idx["comp"],
                    latent_valid,
                    global_write_floor,
                ),
                ratio=self.ratio,
                block_stride=comp.page_stride,
            )
            idx_leaf = leaves["idx"]
            # Pre-norm projection: the store op owns the RMSNorm, the
            # group-first rotation, both bf16 rounding points and the
            # MXFP4 quantize, so the stored bytes match the reference
            # chain.
            keys = latent @ self.wk.weight.T
            indexer_k_store(
                idx_leaf,
                keys,
                positions,
                rope_table,
                self.k_norm.weight,
                self._k_slots(idx_leaf, rows, global_write_floor),
                layer_idx["idx"],
                ratio=self.ratio,
                epsilon=self.k_norm.eps,
            )

        # --- Score and select: the SM100 MXFP4 scorer reads the leaf's
        # split-record pages directly, bounded by each row's live
        # entries, and the cluster top-k selects from the transient
        # scores.
        top_k = min(self.index_topk, self.cap)
        q = F.reshape(self.wq_b(qr), [t, self.n_heads * self.head_dim])
        # bf16 matmul exactly as the reference's bf16 ``weights_proj``
        # does; the fp32 promotion and both scale factors happen inside
        # the quantize op's weight scale.
        w = x @ self.weights_proj.weight.T
        packed_q = indexer_q_rope_quant(
            q,
            w,
            positions,
            rope_table,
            n_heads=self.n_heads,
            weight_scale=self.weights_scale,
        )
        if self.is_candidate_source:
            picks, out_pool = indexer_select_block_topk(
                leaves["idx"],
                packed_q,
                rows.offsets,
                positions,
                layer_idx["idx"],
                top_k=top_k,
                ratio=self.ratio,
                block_size=self.candidate_block_size,
                topk_blocks=self.candidate_topk_blocks,
            )
        elif self.uses_candidates:
            assert pool is not None, "reindex_r1 requires the source pool"
            picks = indexer_select_masked(
                leaves["idx_source"],
                packed_q,
                rows.offsets,
                pool,
                layer_idx["idx_source"],
                top_k=top_k,
                ratio=self.ratio,
                block_size=self.candidate_block_size,
            )
            out_pool = None
        else:
            picks = indexer_select(
                leaves["idx"],
                packed_q,
                rows.offsets,
                layer_idx["idx"],
                top_k=top_k,
                ratio=self.ratio,
            )
            out_pool = None

        # A padded row's synthesized position reaches past the batch;
        # neutralize its picks here once.
        row_index = F.range(
            0, t, 1, out_dim=t, dtype=DType.int32, device=x.device
        )
        last_row = F.reshape(F.cast(rows.offsets[-1], DType.int32), [1])
        comp_indices = F.where(
            F.reshape(row_index >= last_row, [t, 1]),
            F.constant(-1, DType.int32, x.device),
            picks,
        )
        if top_k < self.index_topk:
            comp_indices = F.concat(
                [
                    comp_indices,
                    F.broadcast_to(
                        F.constant(-1, DType.int32, x.device),
                        [t, self.index_topk - top_k],
                    ),
                ],
                axis=-1,
            )
        return position_order(comp_indices), out_pool
