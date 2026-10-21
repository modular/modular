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
"""MLA FP8 index kernel for computing attention scores with paged KV cache."""

from std.bit import log2_floor, pop_count
from std.builtin.device_passable import DeviceTypeEncoder
from std.sys import get_defined_int, size_of
from std.math import align_up, ceildiv, clamp
from std.utils.numerics import inf, neg_inf
from std.utils.static_tuple import StaticTuple

from layout import (
    DefaultEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.coord import Coord, Idx
from layout.tile_layout import Layout as InternalLayout
from layout.tile_tensor import ImmTileTensor, MutTileTensor

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
)
import max.gpu.primitives.warp as warp
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute

from kv_cache.paged_sparse_kv_index_remap import NoRemap, SparseIndexRemap
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    KVCollectionT,
    PagedKVCache,
)

from nn.index_fp8 import (
    _AMD_DMA_PASS_BYTES,
    IndexSmemStorage,
    _has_gfx950_accelerator,
    fp8_index_kernel,
    fp8_index_score_amd,
)
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    KTMATileT,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
    mxfp4_index_score_sm100,
)
from nn.attention.gpu.sparse_index_fp8_sm100_prefill import (
    IndexerScoreHook,
    NoScoreHook,
    _index_swizzle,
    _is_mxfp4,
    _k_box_rows,
    _k_tile_fits_pages,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import store_global_pred
from nn.attention.mha_mask import MaskName
from nn.attention.mha_operand import KVCacheMHAOperand, KVCacheScalesMHAOperand
from nn.attention.mha_utils import indexer_key_bound
from nn.indexer_k_store import (
    MXFP4_INDEX_RECORD_BYTES,
    MXFP4_INDEX_VALUE_BYTES,
    index_key_layer_pages,
)
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    RowBound,
    RowBoundsPtr,
    TopkSink,
    _elems_per_16b,
    persistent_topk_block_split,
)


# Peak bytes of the transient score matrix. Matches vLLM's
# `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` default (512) so a comparison against
# their indexer measures the kernels rather than two different memory policies.
comptime _SCORES_BUDGET_BYTES = (
    get_defined_int["MLA_INDEX_SCORES_BUDGET_MB", 512]() * 1024 * 1024
)

# Default element type of the transient score matrix. Scores never leave this
# op -- its only output is the int32 index tensor -- so this is an internal
# bandwidth choice, not a numerical contract, and `scores_dtype` below keeps
# both arms in one binary.
#
# What bf16 buys is the top-k's READ and the doubled rows per chunk, not the
# scorer, which is a wash either way. It inverts on a decode shape whose score
# matrix is small next to a graph-capture-frozen row stride, where the extra
# convert per store outruns the bytes saved.
comptime _SCORES_DTYPE = DType.bfloat16


trait IndexerTopk:
    """Selects each indexer score row's top-k key positions.

    The contract is `persistent_topk_block_split`'s with `ordered=False`: row
    `r` scans its first `row_bounds.row(r)[0]` columns, a row with at most `k`
    of them takes them all with `-1` in the slots past them, and the order of
    the `k` columns, and which columns tied at the k-th score survive, are
    unspecified.
    """

    @staticmethod
    def select[
        in_dtype: DType, RB: RowBound
    ](
        ctx: DeviceContext,
        scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
        out_idxs: MutPointer[Int32, MutAnyOrigin],
        n: Int,
        k: Int,
        rows: Int,
        row_bounds: RB,
    ) raises:
        """Writes the top-`k` columns of each row of `scores` to `out_idxs`.

        Parameters:
            in_dtype: Score element type.
            RB: Type of `row_bounds`.

        Args:
            ctx: Device context.
            scores: Row-major scores, `[rows, n]`.
            out_idxs: Row-major output, `[rows, k]`.
            n: Score columns per row, the row stride.
            k: Columns selected per row.
            rows: Score rows.
            row_bounds: Per-row live column counts and output sinks.
        """
        ...


struct BlockSplitTopk(IndexerTopk):
    """`persistent_topk_block_split`, the indexers' default top-k."""

    @staticmethod
    def select[
        in_dtype: DType, RB: RowBound
    ](
        ctx: DeviceContext,
        scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
        out_idxs: MutPointer[Int32, MutAnyOrigin],
        n: Int,
        k: Int,
        rows: Int,
        row_bounds: RB,
    ) raises:
        """Writes the top-`k` columns of each row of `scores` to `out_idxs`.

        Parameters:
            in_dtype: Score element type.
            RB: Type of `row_bounds`.

        Args:
            ctx: Device context.
            scores: Row-major scores, `[rows, n]`.
            out_idxs: Row-major output, `[rows, k]`.
            n: Score columns per row, the row stride.
            k: Columns selected per row.
            rows: Score rows.
            row_bounds: Per-row live column counts and output sinks.
        """
        persistent_topk_block_split[RB=RB](
            ctx, scores, out_idxs, n, k, rows, Optional(row_bounds)
        )


@inline(.always)
def _store_block_max[
    dtype: DType, //, block_size: Int, lane_keys: Int
](
    block_max: MutPointer[Float32, MutUntrackedOrigin],
    num_blocks: Int32,
    value: Scalar[dtype],
    row: Int32,
    key: Int32,
    key_live: Bool,
    row_live: Bool,
):
    """Folds one score into its row's maximum for its block of keys.

    Parameters:
        dtype: Score element type.
        block_size: Keys per block, a power of two.
        lane_keys: Consecutive keys a warp holds.

    Args:
        block_max: Per-block maxima, `[rows, num_blocks]` float32.
        num_blocks: Blocks per row.
        value: The score.
        row: Row of `block_max` the score belongs to.
        key: Key column.
        key_live: Whether the key is below the scorer's bound.
        row_live: Whether the row is in the window.
    """
    comptime assert (
        1 << log2_floor(block_size) == block_size
        and lane_keys % block_size == 0
    ), "a block must be a power of two of keys within one warp's span"
    # Keys past the scorer's bound count as `-inf`; a dead row stores nothing.
    var best = value.cast[DType.float32]() if key_live else neg_inf[
        DType.float32
    ]()
    best = warp.lane_group_max[num_lanes=block_size](best)
    var blk = key // Int32(block_size)
    store_global_pred(
        block_max.unsafe_offset(Int(row) * Int(num_blocks) + Int(blk)),
        best,
        Int32(row_live and key % Int32(block_size) == 0 and blk < num_blocks),
    )


@inline(.always)
def _block_kept_score[
    dtype: DType, //, block_size: Int, lane_keys: Int
](
    keep: ImmPointer[UInt8, ImmUntrackedOrigin],
    num_blocks: Int32,
    value: Scalar[dtype],
    row: Int32,
    key: Int32,
    row_live: Bool,
) -> Scalar[dtype]:
    """Returns `value` if its row keeps its block of keys, else `-inf`.

    Parameters:
        dtype: Score element type.
        block_size: Keys per block.
        lane_keys: Consecutive keys a warp holds.

    Args:
        keep: Nonzero for each kept block, `[rows, num_blocks]`.
        num_blocks: Blocks per row; keys past the last block are dropped.
        value: The score.
        row: Row of `keep` the score belongs to.
        key: Key column.
        row_live: Whether the row is in the window.

    Returns:
        The score to store.
    """
    comptime assert (
        lane_keys % block_size == 0
    ), "a block must lie within one warp's span of keys"
    var blk = key // Int32(block_size)
    var in_range = row_live and blk < num_blocks
    # A dead row or block reads entry 0 and is dropped regardless. The load is
    # invariant and unbranched so it does not stall the epilogue.
    var kept = (
        keep.unsafe_offset(
            Int(row) * Int(num_blocks) + Int(blk) if in_range else 0
        )
        .unsafe_address_space_cast[.GLOBAL]()
        .unsafe_load[invariant=True]()
    )
    return value if in_range and kept != 0 else neg_inf[dtype]()


# Threads per CTA of the kept-block selection's pre- and post-pass.
comptime _KEEP_PASS_THREADS = 256


def _pin_kept_block_kernel[
    block_size: Int
](
    block_max: MutPointer[Float32, MutAnyOrigin],
    positions: ImmPointer[Int32, ImmutAnyOrigin],
    live: MutPointer[Int32, MutAnyOrigin],
    num_blocks: Int32,
    rows: Int32,
    ratio: Int32,
):
    """Writes each row's live block count and pins its newest visible block.

    A row's live blocks are those up to the block of its newest visible key,
    `((position + 1) // ratio - 1) // block_size`; that block's maximum
    becomes `+inf` so the selection always keeps it. One thread per row.
    """
    var row = Int(block_idx.x) * _KEEP_PASS_THREADS + Int(thread_idx.x)
    if row >= Int(rows):
        return
    var n = Int(num_blocks)
    # Floor division: a token before the first key pins nothing.
    var pinned = ((Int(positions[row]) + 1) // Int(ratio) - 1) // block_size
    live[row] = Int32(clamp(pinned + 1, 0, n))
    if pinned >= 0 and pinned < n:
        block_max[row * n + pinned] = inf[DType.float32]()


def _drop_dead_kept_blocks_kernel(
    block_max: ImmPointer[Float32, ImmutAnyOrigin],
    ids: MutPointer[Int32, MutAnyOrigin],
    num_blocks: Int32,
    top_k: Int32,
    slots: Int32,
):
    """Clears each selected id whose block has no visible key to `-1`.

    One thread per id slot.
    """
    var i = Int(block_idx.x) * _KEEP_PASS_THREADS + Int(thread_idx.x)
    if i >= Int(slots):
        return
    var b = Int(ids[i])
    if b < 0:
        return
    var row = i // Int(top_k)
    if not (block_max[row * Int(num_blocks) + b] > neg_inf[DType.float32]()):
        ids[i] = -1


# Threads per row expanding kept block ids into flags.
comptime _EXPAND_THREADS = 256


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_EXPAND_THREADS))
)
def _expand_kept_blocks_kernel(
    ids: ImmPointer[Int32, ImmutAnyOrigin],
    keep: MutPointer[UInt8, MutAnyOrigin],
    num_blocks: Int32,
    top_k: Int32,
):
    """Sets the flag of each kept block id, one row per CTA."""
    var row = Int(block_idx.x)
    var k = Int(top_k)
    var n = Int(num_blocks)
    for j in range(Int(thread_idx.x), k, _EXPAND_THREADS):
        var b = Int(ids[row * k + j])
        if b >= 0 and b < n:
            keep[row * n + b] = 1


struct BlockTopKScoreHook[block_size: Int](IndexerScoreHook):
    """Writes the ids of each row's `top_k` best blocks of `block_size` keys.

    A block's value is the maximum score of its keys below the scorer's bound,
    `-inf` if none is. Each row keeps the block of its newest visible key,
    `((position + 1) // ratio - 1) // block_size`, then the best of the
    blocks before it; `-inf` blocks are never kept. Which of the blocks tied
    with the k-th best are kept is unspecified. A row's ids come in no
    particular order, with `-1` in the slots it leaves unused.

    The maxima of one score row-window at a time live in a scratch buffer of
    `max_rows` rows, and the selection runs after each window is scored, so
    no `[tokens, num_blocks]` buffer exists.

    Parameters:
        block_size: Keys per block, a power of two.
    """

    comptime device_type: AnyType = Self
    """Device-side type; the struct is trivially device-passable."""

    comptime enabled = True
    """Always runs."""

    var block_max: MutPointer[Float32, MutUntrackedOrigin]
    """Window scratch, `[max_rows, num_blocks]` float32."""
    var num_blocks: Int32
    """Blocks per row."""
    var max_rows: Int32
    """Rows the scratch holds; caps the score window."""
    var live: MutPointer[Int32, MutUntrackedOrigin]
    """Window scratch, `[max_rows]` int32 live block counts."""
    var ids: MutPointer[Int32, MutUntrackedOrigin]
    """Kept block ids output, `[tokens, top_k]`."""
    var top_k: Int32
    """Blocks kept per row."""
    var positions: ImmPointer[Int32, ImmUntrackedOrigin]
    """Position of each query token, `[tokens]`."""
    var ratio: Int32
    """Tokens per key."""

    def __init__(
        out self,
        block_max: MutPointer[Float32, MutAnyOrigin],
        num_blocks: Int,
        max_rows: Int,
        live: MutPointer[Int32, MutAnyOrigin],
        ids: MutPointer[Int32, MutAnyOrigin],
        top_k: Int,
        positions: ImmPointer[Int32, ImmutAnyOrigin],
        ratio: Int,
    ):
        """Creates the hook.

        Args:
            block_max: Window scratch, `[max_rows, num_blocks]` float32, 16 B
                aligned.
            num_blocks: Blocks per row.
            max_rows: Rows the scratch holds, at least 1.
            live: Window scratch, `[max_rows]` int32.
            ids: Kept block ids output, `[tokens, top_k]`.
            top_k: Blocks kept per row, at most `PERSISTENT_TOPK_MAX_N`.
            positions: Position of each query token, `[tokens]`.
            ratio: Tokens per key.
        """
        self.block_max = block_max.unsafe_origin_cast[MutUntrackedOrigin]()
        self.num_blocks = Int32(num_blocks)
        self.max_rows = Int32(max_rows)
        self.live = live.unsafe_origin_cast[MutUntrackedOrigin]()
        self.ids = ids.unsafe_origin_cast[MutUntrackedOrigin]()
        self.top_k = Int32(top_k)
        self.positions = positions.unsafe_origin_cast[ImmUntrackedOrigin]()
        self.ratio = Int32(ratio)

    def window_rows(self) -> Int:
        """Returns the rows the scratch holds.

        Returns:
            `max_rows`.
        """
        return Int(self.max_rows)

    def window(
        self, ctx: DeviceContext, first_row: Int, rows: Int
    ) raises -> Self:
        """Sets the window's maxima to `-inf` and returns its hook.

        Args:
            ctx: Device context.
            first_row: Query token of the window's row 0.
            rows: Rows in the window, at most `max_rows`.

        Returns:
            The hook; its scratch row 0 is query token `first_row`.
        """
        ctx.enqueue_memset(
            DeviceBuffer(
                ctx,
                self.block_max.as_unsafe_any_origin(),
                rows * Int(self.num_blocks),
                owning=False,
            ),
            neg_inf[DType.float32](),
        )
        return self

    def after_window(
        self, ctx: DeviceContext, first_row: Int, rows: Int
    ) raises:
        """Selects the window's kept blocks into rows of `ids`.

        Args:
            ctx: Device context.
            first_row: Query token of the window's row 0.
            rows: Rows in the window.
        """
        var k = Int(self.top_k)
        if k == 0 or rows == 0:
            return
        if k > PERSISTENT_TOPK_MAX_N:
            raise Error(
                "BlockTopKScoreHook keeps at most ",
                PERSISTENT_TOPK_MAX_N,
                " blocks per row, got ",
                k,
            )
        var ids = self.ids.unsafe_offset(first_row * k)
        ctx.enqueue_function[_pin_kept_block_kernel[Self.block_size]](
            self.block_max.as_unsafe_any_origin(),
            self.positions.unsafe_offset(first_row).as_unsafe_any_origin(),
            self.live.as_unsafe_any_origin(),
            self.num_blocks,
            Int32(rows),
            self.ratio,
            grid_dim=ceildiv(rows, _KEEP_PASS_THREADS),
            block_dim=_KEEP_PASS_THREADS,
        )
        persistent_topk_block_split(
            ctx,
            self.block_max.as_imm().as_unsafe_any_origin(),
            ids.as_unsafe_any_origin(),
            Int(self.num_blocks),
            k,
            rows,
            Optional(RowBoundsPtr(self.live.as_imm().as_unsafe_any_origin())),
        )
        ctx.enqueue_function[_drop_dead_kept_blocks_kernel](
            self.block_max.as_imm().as_unsafe_any_origin(),
            ids.as_unsafe_any_origin(),
            self.num_blocks,
            self.top_k,
            Int32(rows * k),
            grid_dim=ceildiv(rows * k, _KEEP_PASS_THREADS),
            block_dim=_KEEP_PASS_THREADS,
        )

    @inline(.always)
    def score[
        dtype: DType, lane_keys: Int
    ](
        self,
        value: Scalar[dtype],
        row: Int32,
        key: Int32,
        key_live: Bool,
        row_live: Bool,
    ) -> Scalar[dtype]:
        """Folds `value` into its block's maximum and returns it unchanged.

        Parameters:
            dtype: Score element type.
            lane_keys: Consecutive keys a warp holds.

        Args:
            value: The score.
            row: Window row of the query token.
            key: Key column.
            key_live: Whether the key is below the scorer's bound.
            row_live: Whether the row is in the window.

        Returns:
            `value`.
        """
        _store_block_max[Self.block_size, lane_keys](
            self.block_max, self.num_blocks, value, row, key, key_live, row_live
        )
        return value

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"BlockTopKScoreHook"`.
        """
        return "BlockTopKScoreHook"


struct BlockTopKMaskScoreHook[block_size: Int](IndexerScoreHook):
    """Drops every key whose block of `block_size` keys a row does not keep.

    Each row names its kept blocks as `top_k` ids, `-1` for none; a dropped
    key scores `-inf`. Before a score row-window is scored, its rows' ids
    expand into a flag scratch of `max_rows` rows. The top-k still takes every
    key below a row's bound when it has at most `top_k` of them, so each row
    must keep at least that many keys for no dropped key to be selected.

    Parameters:
        block_size: Keys per block.
    """

    comptime device_type: AnyType = Self
    """Device-side type; the struct is trivially device-passable."""

    comptime enabled = True
    """Always runs."""

    var ids: ImmPointer[Int32, ImmUntrackedOrigin]
    """Kept block ids, `[tokens, top_k]`."""
    var top_k: Int32
    """Ids per row."""
    var keep: MutPointer[UInt8, MutUntrackedOrigin]
    """Window flag scratch, `[max_rows, num_blocks]`."""
    var num_blocks: Int32
    """Blocks per row; keys past the last block are dropped."""
    var max_rows: Int32
    """Rows the scratch holds; caps the score window."""

    def __init__(
        out self,
        ids: ImmPointer[Int32, ImmutAnyOrigin],
        top_k: Int,
        keep: MutPointer[UInt8, MutAnyOrigin],
        num_blocks: Int,
        max_rows: Int,
    ):
        """Creates the hook.

        Args:
            ids: Kept block ids, `[tokens, top_k]`.
            top_k: Ids per row.
            keep: Window flag scratch, `[max_rows, num_blocks]`.
            num_blocks: Blocks per row.
            max_rows: Rows the scratch holds, at least 1.
        """
        self.ids = ids.unsafe_origin_cast[ImmUntrackedOrigin]()
        self.top_k = Int32(top_k)
        self.keep = keep.unsafe_origin_cast[MutUntrackedOrigin]()
        self.num_blocks = Int32(num_blocks)
        self.max_rows = Int32(max_rows)

    def window_rows(self) -> Int:
        """Returns the rows the scratch holds.

        Returns:
            `max_rows`.
        """
        return Int(self.max_rows)

    def window(
        self, ctx: DeviceContext, first_row: Int, rows: Int
    ) raises -> Self:
        """Expands the window's kept ids into flags and returns its hook.

        Args:
            ctx: Device context.
            first_row: Query token of the window's row 0.
            rows: Rows in the window, at most `max_rows`.

        Returns:
            The hook; its scratch row 0 is query token `first_row`.
        """
        ctx.enqueue_memset(
            DeviceBuffer(
                ctx,
                self.keep.as_unsafe_any_origin(),
                rows * Int(self.num_blocks),
                owning=False,
            ),
            UInt8(0),
        )
        var ids = self.ids.unsafe_offset(first_row * Int(self.top_k))
        ctx.enqueue_function[_expand_kept_blocks_kernel](
            ids.as_unsafe_any_origin(),
            self.keep.as_unsafe_any_origin(),
            self.num_blocks,
            self.top_k,
            grid_dim=rows,
            block_dim=_EXPAND_THREADS,
        )
        return self

    def after_window(
        self, ctx: DeviceContext, first_row: Int, rows: Int
    ) raises:
        """Does nothing; the flags are rebuilt before each window.

        Args:
            ctx: Device context.
            first_row: Query token of the window's row 0.
            rows: Rows in the window.
        """
        pass

    @inline(.always)
    def score[
        dtype: DType, lane_keys: Int
    ](
        self,
        value: Scalar[dtype],
        row: Int32,
        key: Int32,
        key_live: Bool,
        row_live: Bool,
    ) -> Scalar[dtype]:
        """Returns `value` if its block is kept, else `-inf`.

        Parameters:
            dtype: Score element type.
            lane_keys: Consecutive keys a warp holds.

        Args:
            value: The score.
            row: Window row of the query token.
            key: Key column.
            key_live: Whether the key is below the scorer's bound.
            row_live: Whether the row is in the window.

        Returns:
            The score to store.
        """
        return _block_kept_score[Self.block_size, lane_keys](
            self.keep.as_imm(), self.num_blocks, value, row, key, row_live
        )

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"BlockTopKMaskScoreHook"`.
        """
        return "BlockTopKMaskScoreHook"


struct _IndexerSink[R: SparseIndexRemap, write_logical: Bool](TopkSink):
    """Writes a top-k slot's logical index and, through `R`, its remap.

    Parameters:
        R: Type of the remap output.
        write_logical: Whether the logical index is written at all.
    """

    var logical: MutPointer[Int32, MutUntrackedOrigin]
    """Flat logical output, `[rows, top_k]`."""
    var remap: Self.R
    """Second output derived from each logical index."""
    var batch: Int
    """Batch entry of the row this sink serves."""

    @inline(.always)
    def __init__(
        out self,
        logical: MutPointer[Int32, MutAnyOrigin],
        remap: Self.R,
        batch: Int,
    ):
        """Creates the sink for one row.

        Args:
            logical: Flat logical output.
            remap: Second output derived from each logical index.
            batch: Batch entry of the row.
        """
        self.logical = logical.unsafe_origin_cast[MutUntrackedOrigin]()
        self.remap = remap
        self.batch = batch

    @inline(.always)
    def store(self, idx: Int, col: Int32):
        """Writes slot `idx` of both outputs.

        Args:
            idx: Flat output slot.
            col: Selected key position, or `-1`.
        """
        comptime if Self.write_logical:
            self.logical[idx] = col
        self.remap.store(idx, self.batch, col)


struct _IndexerRowBound[
    causal: Bool, kpool: Int, R: SparseIndexRemap, write_logical: Bool
](RowBound):
    """Each top-k row's live key count, computed in the top-k kernel itself.

    The same `indexer_key_bound` the SM100 scorers apply in their store
    guards, so the top-k reads exactly the slots the scorer wrote (it clamps
    the bound to the score stride itself). The row's batch entry, found on
    the way, also keys `R`'s remap of each selected position.
    """

    comptime device_type: AnyType = Self
    """Device-side type alias; the struct is trivially device-passable."""

    comptime Sink = _IndexerSink[Self.R, Self.write_logical]
    """Writes the logical indices and their remap."""

    var row_offsets: ImmPointer[UInt32, ImmUntrackedOrigin]
    """Ragged query row offsets, `batch_size + 1` entries."""
    var cache_lengths: ImmPointer[UInt32, ImmUntrackedOrigin]
    """Per-batch cached-prefix length."""
    var batch_size: Int32
    """Number of batch entries."""
    var first_row: Int32
    """Token index of the launch's row 0 (the score window's first row)."""
    var remap: Self.R
    """Remap output, already advanced to the launch's row 0."""

    @inline(.always)
    def __init__(
        out self,
        row_offsets: ImmPointer[UInt32, ImmutAnyOrigin],
        cache_lengths: ImmPointer[UInt32, ImmutAnyOrigin],
        batch_size: Int,
        first_row: Int,
        remap: Self.R,
    ):
        """Captures the ragged metadata the bound is derived from.

        Args:
            row_offsets: Ragged query row offsets, `batch_size + 1` entries.
            cache_lengths: Per-batch cached-prefix length.
            batch_size: Number of batch entries.
            first_row: Token index of the launch's row 0.
            remap: Remap output, already advanced to the launch's row 0.
        """
        self.row_offsets = row_offsets.unsafe_origin_cast[ImmUntrackedOrigin]()
        self.cache_lengths = cache_lengths.unsafe_origin_cast[
            ImmUntrackedOrigin
        ]()
        self.batch_size = Int32(batch_size)
        self.first_row = Int32(first_row)
        self.remap = remap

    @inline(.always)
    def row(
        self, row: Int, out_idxs: MutPointer[Int32, MutAnyOrigin]
    ) -> Tuple[Int, Self.Sink]:
        """Returns the live key count of launch row `row` and its sink.

        Args:
            row: Row index within the launch.
            out_idxs: The launch's flat logical output.

        Returns:
            `indexer_key_bound(...)` for the row's token, and a sink that
            also remaps into the row's batch entry.
        """
        var token = Int(self.first_row) + row
        var batch = Int(self.batch_size)
        var lane = Int(lane_id())
        # The owning entry is the number of entries that end at or before the
        # token. Offsets are monotone, so a warp counts `SPAN` of them per
        # round with every load in flight at once, and takes the owner's
        # offsets and cache length from the same loads by shuffles: one load
        # latency per round where a serial walk takes one per entry.
        comptime vote_t = DType.uint32 if WARP_SIZE == 32 else DType.uint64
        comptime CH = 4
        comptime SPAN = CH * WARP_SIZE
        var batch_idx = 0
        var q_start = Int(self.row_offsets[0])
        var q_end = 0
        var cache_len = 0
        for base in range(0, batch, SPAN):
            var ends = SIMD[DType.uint32, CH](0)
            var lens = SIMD[DType.uint32, CH](0)
            comptime for c in range(CH):
                var b = base + c * WARP_SIZE + lane
                if b < batch:
                    ends[c] = self.row_offsets[b + 1]
                    lens[c] = self.cache_lengths[b]
            var ended = 0
            comptime for c in range(CH):
                var b = base + c * WARP_SIZE + lane
                ended += Int(
                    pop_count(
                        warp.vote[vote_t](b < batch and Int(ends[c]) <= token)
                    )
                )
            batch_idx += ended
            # The owner is entry `ended` of this round: slot `ended / WS` of
            # lane `ended % WS`, and its start is the entry before's end.
            var own = UInt32(ended % WARP_SIZE)
            var prev = UInt32((ended + WARP_SIZE - 1) % WARP_SIZE)
            comptime for c in range(CH):
                var end_c = Int(warp.shuffle_idx(ends[c], own))
                var len_c = Int(warp.shuffle_idx(lens[c], own))
                var prev_c = Int(warp.shuffle_idx(ends[c], prev))
                if c == ended // WARP_SIZE:
                    q_end = end_c
                    cache_len = len_c
                if ended > 0 and c == (ended - 1) // WARP_SIZE:
                    q_start = prev_c
            if ended < SPAN:
                break
        var seq_len = q_end - q_start
        return (
            indexer_key_bound[Self.kpool](
                cache_len + seq_len, seq_len, token - q_start, Int(Self.causal)
            ),
            Self.Sink(out_idxs, self.remap, batch_idx),
        )

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode_fields[Self](self, target)

    @staticmethod
    def get_type_name() -> String:
        """Returns the type name for runtime diagnostics.

        Returns:
            Always `"_IndexerRowBound"`.
        """
        return "_IndexerRowBound"


# ===----------------------------------------------------------------------=== #
# Main function: mla_indexer_ragged_float8_paged
# ===----------------------------------------------------------------------=== #


@inline(.always)
def mla_indexer_ragged_float8_paged[
    oi_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    iro_layout: TensorLayout,
    //,
    dtype: DType,
    KCollectionT: KVCollectionT,
    num_heads: Int,
    depth: Int,
    top_k: Int,
    mask_str: StaticString,
    scores_dtype: DType = _SCORES_DTYPE,
    kpool: Int = 1,
    Topk: IndexerTopk = BlockSplitTopk,
](
    output_indices: MutTileTensor[.int32, oi_layout, _],
    q: ImmTileTensor[dtype, q_layout, _],
    q_s: ImmTileTensor[.float32, qs_layout, _],
    input_row_offsets: ImmTileTensor[.uint32, iro_layout, _],
    k_collection: KCollectionT,
    layer_idx: UInt32,
    ctx: DeviceContext,
    scores_budget_bytes: Int = _SCORES_BUDGET_BYTES,
) raises:
    """Computes the FP8 indexer's top-k key positions, without a remap.

    See the overload taking `remap`; this one writes the logical indices
    only.

    Parameters:
        oi_layout: Layout of the top-k index output.
        q_layout: Layout of the query tensor.
        qs_layout: Layout of the query scales.
        iro_layout: Layout of the ragged query row offsets.
        dtype: Element type of the `q` query tensor, an FP8 dtype.
        KCollectionT: Type of the KV collection holding cached K values and
            K scales.
        num_heads: Number of attention heads per token.
        depth: Per-head key dimension (head size) in elements.
        top_k: Requested number of top-scoring key indices per token.
        mask_str: Name of the mask, `MaskName.NULL` or `MaskName.CAUSAL`.
        scores_dtype: Element type of the transient score matrix.
        kpool: Tokens per pooled cache row.
        Topk: The top-k over the score rows.

    Args:
        output_indices: Dense `[total_seq_len, top_k]` top-k output.
        q: Query tensor `[total_seq_len, num_heads, head_dim]` in FP8.
        q_s: Query scales `[total_seq_len, num_heads]` in float32.
        input_row_offsets: Ragged row offsets for queries `[batch_size + 1]`.
        k_collection: KV collection containing cached K values and K scales.
        layer_idx: Layer index for retrieving cache.
        ctx: Device context.
        scores_budget_bytes: Peak bytes the transient score matrix may occupy.
    """
    mla_indexer_ragged_float8_paged[
        dtype,
        KCollectionT,
        num_heads,
        depth,
        top_k,
        mask_str,
        scores_dtype,
        kpool,
        write_logical=True,
        Topk=Topk,
    ](
        output_indices,
        q,
        q_s,
        input_row_offsets,
        k_collection,
        layer_idx,
        ctx,
        NoRemap(),
        scores_budget_bytes,
    )


@inline(.always)
def mla_indexer_ragged_float8_paged[
    oi_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    iro_layout: TensorLayout,
    R: SparseIndexRemap,
    //,
    dtype: DType,
    KCollectionT: KVCollectionT,
    num_heads: Int,
    depth: Int,
    top_k: Int,
    mask_str: StaticString,
    scores_dtype: DType = _SCORES_DTYPE,
    kpool: Int = 1,
    write_logical: Bool = False,
    Topk: IndexerTopk = BlockSplitTopk,
](
    output_indices: MutTileTensor[.int32, oi_layout, _],
    q: ImmTileTensor[dtype, q_layout, _],
    q_s: ImmTileTensor[.float32, qs_layout, _],
    input_row_offsets: ImmTileTensor[.uint32, iro_layout, _],
    k_collection: KCollectionT,
    layer_idx: UInt32,
    ctx: DeviceContext,
    remap: R,
    scores_budget_bytes: Int = _SCORES_BUDGET_BYTES,
) raises:
    """Compute FP8 indexed attention scores using paged KV cache and return top-k indices.

    This function:
    1. Computes FP8 matmul between q and cached k (with scales), aggregated across heads
    2. Bounds each token's candidates by the mask (causal, etc.)
    3. Computes top-k indices per token (scores are summed across all heads)

    Parameters:
        oi_layout: Layout of the top-k index output.
        q_layout: Layout of the query tensor.
        qs_layout: Layout of the query scales.
        iro_layout: Layout of the ragged query row offsets.
        R: Type of `remap`.
        dtype: Element type of the `q` query tensor, an FP8 dtype.
        KCollectionT: Type of the KV collection holding cached K values and
            K scales.
        num_heads: Number of attention heads per token.
        depth: Per-head key dimension (head size) in elements.
        top_k: Requested number of top-scoring key indices to select per
            token.
        mask_str: Name of the mask to apply, either `MaskName.NULL` or
            `MaskName.CAUSAL`.
        scores_dtype: Element type of the transient score matrix. `float32` or
            `bfloat16`; the latter halves the buffer, so a row window under a
            fixed byte budget holds twice the rows. Honoured only on the SM100
            scorers -- the scalar fallback is f32-only and resolves to it, which
            no caller can observe because the matrix is internal.
        kpool: Tokens per pooled cache row. `1` scores one row per token;
            `k > 1` scores one pooled key per `k` consecutive tokens, so
            every candidate count and the caller's `top_k` are
            pool-granular.
        write_logical: Whether to write `output_indices` (defaults to
            `False`). `False` leaves it untouched, so it may be empty, and
            requires a remap: the remap output is then the only one.
        Topk: The top-k over the score rows.

    Args:
        output_indices: Dense output tensor for top-k indices [total_seq_len, top_k].
            Invalid positions (where there are fewer than top_k valid keys due to
            causal masking or shorter sequences) are filled with -1.
        q: Query tensor [total_seq_len, num_heads, head_dim] in FP8.
        q_s: Query scales [total_seq_len, num_heads] in float32.
        input_row_offsets: Ragged row offsets for queries [batch_size + 1].
        k_collection: KV collection containing cached K values and K scales.
            K scales are accessed via k_cache.scales (quantization_granularity=head_size).
        layer_idx: Layer index for retrieving cache.
        ctx: Device context.
        remap: Second output the top-k writes beside each logical index, laid
            out like `output_indices` (for example `PagedKVRemap`, the
            physical paged-cache rows a sparse MLA gather reads). Requires
            `kpool == 1`: a pooled index is not a token position.
        scores_budget_bytes: Peak bytes the transient score matrix may occupy.
            Longer batches are scored a row-window at a time to stay under it
            (see the chunking below). Runtime rather than comptime because
            nothing below it is specialized on the value -- it reaches only the
            `rows_per_chunk` arithmetic -- so a sweep over budgets costs no
            recompiles. Exposed so tests can force a window small enough to
            exercise the multi-chunk path on toy shapes.
    """
    # Verify that k_collection has scales enabled (required for MLA k_s).
    # For MLA, scales should have head_dim_granularity == 1 (one scale per token
    # per head), which requires quantization_granularity >= depth (head_size).
    comptime CacheType = KCollectionT.CacheType
    comptime assert (
        CacheType.quantization_enabled
    ), "k_collection must have quantization/scales enabled for MLA k_s values"
    comptime assert CacheType.quantization_granularity >= depth, (
        "k_collection.quantization_granularity must be >= depth (head_dim) for"
        " MLA (requires one scale per token per head, i.e. head_dim_granularity"
        " == 1)"
    )

    _mla_indexer_paged_impl[
        dtype,
        num_heads,
        depth,
        top_k,
        mask_str,
        scores_dtype,
        kpool,
        write_logical,
        Topk=Topk,
    ](
        output_indices,
        q,
        q_s,
        input_row_offsets,
        k_collection.get_key_cache(Int(layer_idx)),
        ctx,
        remap,
        NoScoreHook(),
        scores_budget_bytes,
    )


# ===----------------------------------------------------------------------=== #
# MXFP4: mla_indexer_ragged_mxfp4_paged
# ===----------------------------------------------------------------------=== #

comptime _MXFP4_INDEX_KV_PARAMS = KVCacheStaticParams(
    num_heads=1, head_size=MXFP4_INDEX_VALUE_BYTES, is_mla=True
)

comptime MXFP4IndexKCache[page_size: Int] = PagedKVCache[
    DType.uint8,
    _MXFP4_INDEX_KV_PARAMS,
    page_size,
    MutUntrackedOrigin,
    DefaultEngine[element_width=1],
    ImmutAnyOrigin,
    DefaultEngine[element_width=1],
    ImmutAnyOrigin,
    DefaultEngine[element_width=1],
    MutUntrackedOrigin,
    DefaultEngine[element_width=1],
    scale_dtype_=DType.uint32,
    quantization_granularity_=MXFP4_INDEX_VALUE_BYTES,
]
"""A paged MXFP4 index-key cache in `indexer_k_store`'s page layout.

A page holds `page_size` keys: their E2M1 values, then their 32-bit scale
words. The values are the cache's blocks and the words its scales, both
strided by the page pitch, so the FP8 indexer's paged operands and TMA tiles
read it unchanged.
"""


def _mxfp4_index_k_cache[
    page_size: Int
](
    base: MutPointer[UInt8, MutUntrackedOrigin],
    num_pages: Int64,
    page_stride: Int64,
    lookup_table: TileTensor[.uint32, ...],
    cache_lengths: TileTensor[.uint32, ...],
    max_seq_length: Int,
    max_cache_length: Int,
) -> MXFP4IndexKCache[page_size]:
    """Views `num_pages` pages, `page_stride` bytes apart from `base`."""
    comptime assert lookup_table.rank == 2 and cache_lengths.rank == 1
    var blocks = TileTensor(
        base,
        InternalLayout(
            Coord(
                num_pages,
                Idx[page_size],
                Idx[1],
                Idx[MXFP4_INDEX_VALUE_BYTES],
            ),
            Coord(
                page_stride,
                Idx[MXFP4_INDEX_VALUE_BYTES],
                Idx[MXFP4_INDEX_VALUE_BYTES],
                Idx[1],
            ),
        ),
    )
    var scales = TileTensor(
        base.unsafe_offset(page_size * MXFP4_INDEX_VALUE_BYTES).unsafe_bitcast[
            UInt32
        ](),
        InternalLayout(
            Coord(num_pages, Idx[page_size], Idx[1], Idx[1]),
            Coord(page_stride // 4, Idx[1], Idx[1], Idx[1]),
        ),
    )
    # The cache stores its 1D and 2D tensors on the default engine.
    var lengths = TileTensor(
        cache_lengths.ptr.unsafe_address_space_cast[
            .GENERIC
        ]().as_unsafe_any_origin(),
        InternalLayout(Coord(Int64(cache_lengths.dim[0]())), Coord(Idx[1])),
    )
    var lut_cols = Int64(lookup_table.dim[1]())
    var lut = TileTensor(
        lookup_table.ptr.unsafe_address_space_cast[
            .GENERIC
        ]().as_unsafe_any_origin(),
        InternalLayout(
            Coord(Int64(lookup_table.dim[0]()), lut_cols),
            Coord(lut_cols, Idx[1]),
        ),
    )
    return MXFP4IndexKCache[page_size](
        blocks,
        lengths,
        lut,
        UInt32(max_seq_length),
        UInt32(max_cache_length),
        scales=rebind[MXFP4IndexKCache[page_size].scales_tt_type](scales),
    )


def mxfp4_index_k_cache[
    page_size: Int
](
    cache: ImmTileTensor[.uint8, ...],
    lookup_table: ImmTileTensor[.uint32, ...],
    cache_lengths: ImmTileTensor[.uint32, ...],
    max_seq_length: Int,
    max_cache_length: Int,
) -> MXFP4IndexKCache[page_size]:
    """Views packed MXFP4 index-key pages as a paged cache.

    Parameters:
        page_size: Keys per page.

    Args:
        cache: Pages written by `indexer_k_store` with `fp4=True`,
            `[pages, page_stride]` row-major; `page_stride` is a multiple of
            16 and holds `page_size` records of `MXFP4_INDEX_RECORD_BYTES`.
        lookup_table: Physical page of each request's logical page,
            `[batch, max_pages]`.
        cache_lengths: Tokens each request had cached before this call,
            `[batch]`. The keys this call scores are already stored.
        max_seq_length: Most new tokens any request brings.
        max_cache_length: Most cached tokens any request has.

    Returns:
        The cache view.
    """
    var page_stride = Int64(cache.dim[1]())
    debug_assert(
        page_stride % 16 == 0
        and page_stride >= Int64(page_size * MXFP4_INDEX_RECORD_BYTES),
        "mxfp4 index pages need a 16-byte pitch holding every row and word",
    )
    comptime assert (
        cache_lengths.is_row_major and lookup_table.is_row_major
    ), "cache lengths and the lookup table must be row-major"
    return _mxfp4_index_k_cache[page_size](
        cache.ptr.unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_address_space_cast[.GENERIC](),
        Int64(cache.dim[0]()),
        page_stride,
        lookup_table,
        cache_lengths,
        max_seq_length,
        max_cache_length,
    )


def mxfp4_index_k_cache[
    page_size: Int
](
    blocks: TileTensor[.uint8, ...],
    page_stride: Int,
    layer: Int,
    lookup_table: TileTensor[.uint32, ...],
    cache_lengths: TileTensor[.uint32, ...],
    max_seq_length: Int,
    max_cache_length: Int,
) raises -> MXFP4IndexKCache[page_size]:
    """Views one layer of a multi-layer MXFP4 index-key leaf as a paged cache.

    Parameters:
        page_size: Keys per page.

    Args:
        blocks: The leaf's blocks, `[blocks, 1, layers, page_size, 1, record]`
            with `MXFP4_INDEX_RECORD_BYTES` records, written by
            `indexer_k_norm_rope_store_layer` with `fp4=True`.
        page_stride: Bytes from one block to the next, or `-1` when packed.
        layer: The layer's index within the leaf.
        lookup_table: Physical block of each request's logical page,
            `[batch, max_pages]`.
        cache_lengths: Tokens each request had cached before this call,
            `[batch]`. The keys this call scores are already stored.
        max_seq_length: Most new tokens any request brings.
        max_cache_length: Most cached tokens any request has.

    Returns:
        The cache view.

    Raises:
        If the leaf does not hold `page_size` MXFP4 records a page, its pitch
        or the layer is invalid, or the cache lengths or the lookup table are
        not row-major.
    """
    if Int(blocks.dim[3]()) != page_size:
        raise Error("index-key blocks do not hold page_size keys a page")
    if (
        Int(cache_lengths.layout.stride[0]().value()) != 1
        or Int(lookup_table.layout.stride[1]().value()) != 1
        or Int(lookup_table.layout.stride[0]().value())
        != Int(lookup_table.dim[1]())
    ):
        raise Error("cache lengths and the lookup table must be row-major")
    var base, pitch = index_key_layer_pages[MXFP4_INDEX_RECORD_BYTES](
        blocks, page_stride, layer
    )
    return _mxfp4_index_k_cache[page_size](
        base,
        Int64(blocks.dim[0]()),
        Int64(pitch),
        lookup_table,
        cache_lengths,
        max_seq_length,
        max_cache_length,
    )


@inline(.always)
def mla_indexer_ragged_mxfp4_paged[
    oi_layout: TensorLayout,
    q_layout: TensorLayout,
    qsf_layout: TensorLayout,
    w_layout: TensorLayout,
    iro_layout: TensorLayout,
    //,
    num_heads: Int,
    top_k: Int,
    mask_str: StaticString,
    page_size: Int,
    scores_dtype: DType = _SCORES_DTYPE,
    kpool: Int = 1,
    Topk: IndexerTopk = BlockSplitTopk,
](
    output_indices: MutTileTensor[.int32, oi_layout, _],
    q: ImmTileTensor[.uint8, q_layout, _],
    q_sf: ImmTileTensor[.uint32, qsf_layout, _],
    weights: ImmTileTensor[.float32, w_layout, _],
    input_row_offsets: ImmTileTensor[.uint32, iro_layout, _],
    k_cache: MXFP4IndexKCache[page_size],
    ctx: DeviceContext,
    scores_budget_bytes: Int = _SCORES_BUDGET_BYTES,
) raises:
    """Computes the MXFP4 indexer's top-k key positions, without a remap.

    See the overload taking `remap`; this one writes the logical indices
    only.

    Parameters:
        oi_layout: Layout of the top-k index output.
        q_layout: Layout of the query tensor.
        qsf_layout: Layout of the query block scales.
        w_layout: Layout of the per-head weights.
        iro_layout: Layout of the ragged query row offsets.
        num_heads: Index heads per token.
        top_k: Key positions to select per token.
        mask_str: `MaskName.NULL` or `MaskName.CAUSAL`.
        page_size: Keys per cache page.
        scores_dtype: Element type of the transient score matrix.
        kpool: Tokens per cache row, the compression ratio.
        Topk: The top-k over the score rows.

    Args:
        output_indices: Dense `[total_seq_len, top_k]` top-k output.
        q: Packed E2M1 queries, `[total_seq_len, num_heads, 64]`.
        q_sf: Query block scales, `[total_seq_len, num_heads]`.
        weights: Per-head weights, `[total_seq_len, num_heads]`.
        input_row_offsets: Ragged row offsets for queries `[batch_size + 1]`.
        k_cache: The index keys, from `mxfp4_index_k_cache`.
        ctx: Device context.
        scores_budget_bytes: Peak bytes the transient score matrix may occupy.
    """
    mla_indexer_ragged_mxfp4_paged[
        num_heads,
        top_k,
        mask_str,
        page_size,
        scores_dtype,
        kpool,
        write_logical=True,
        Topk=Topk,
    ](
        output_indices,
        q,
        q_sf,
        weights,
        input_row_offsets,
        k_cache,
        ctx,
        NoRemap(),
        scores_budget_bytes,
    )


@inline(.always)
def mla_indexer_ragged_mxfp4_paged[
    oi_layout: TensorLayout,
    q_layout: TensorLayout,
    qsf_layout: TensorLayout,
    w_layout: TensorLayout,
    iro_layout: TensorLayout,
    R: SparseIndexRemap,
    //,
    num_heads: Int,
    top_k: Int,
    mask_str: StaticString,
    page_size: Int,
    scores_dtype: DType = _SCORES_DTYPE,
    kpool: Int = 1,
    write_logical: Bool = False,
    Topk: IndexerTopk = BlockSplitTopk,
](
    output_indices: MutTileTensor[.int32, oi_layout, _],
    q: ImmTileTensor[.uint8, q_layout, _],
    q_sf: ImmTileTensor[.uint32, qsf_layout, _],
    weights: ImmTileTensor[.float32, w_layout, _],
    input_row_offsets: ImmTileTensor[.uint32, iro_layout, _],
    k_cache: MXFP4IndexKCache[page_size],
    ctx: DeviceContext,
    remap: R,
    scores_budget_bytes: Int = _SCORES_BUDGET_BYTES,
) raises:
    """Computes the MXFP4 indexer's top-k key positions over packed pages.

    The lightning indexer with MXFP4 queries and keys:

        score[t, s] = sum_h relu(q[t, h] . k[s]) * weights[t, h]

    where `q[t, h]` and `k[s]` are 128 E2M1 values, each run of 32 scaled by
    its UE8M0 exponent, and `weights` already folds the softmax and head
    scales. Scores land in a transient matrix and the bounded top-k selects
    from it, as `mla_indexer_ragged_float8_paged` does.

    Parameters:
        oi_layout: Layout of the top-k index output.
        q_layout: Layout of the query tensor.
        qsf_layout: Layout of the query block scales.
        w_layout: Layout of the per-head weights.
        iro_layout: Layout of the ragged query row offsets.
        R: Type of `remap`.
        num_heads: Index heads per token; 32.
        top_k: Key positions to select per token.
        mask_str: `MaskName.NULL` or `MaskName.CAUSAL`.
        page_size: Keys per cache page; a multiple of 128 or a power-of-two
            fraction of it down to 8.
        scores_dtype: Element type of the transient score matrix.
        kpool: Tokens per cache row, the compression ratio. A key row covers
            `kpool` consecutive tokens, so indices count rows.
        write_logical: Whether to write `output_indices`.
        Topk: The top-k over the score rows.

    Args:
        output_indices: Dense `[total_seq_len, top_k]` top-k output; `-1`
            fills a row with fewer live keys than `top_k`.
        q: Packed E2M1 queries, `[total_seq_len, num_heads, 64]`, two
            channels a byte with the even one in the low nibble.
        q_sf: Query block scales, `[total_seq_len, num_heads]`: each word is
            four UE8M0 bytes, channels 32j..32j+31 in byte j.
        weights: Per-head weights, `[total_seq_len, num_heads]`.
        input_row_offsets: Ragged row offsets for queries `[batch_size + 1]`.
        k_cache: The index keys, from `mxfp4_index_k_cache`.
        ctx: Device context.
        remap: Second output beside each logical index; see
            `mla_indexer_ragged_float8_paged`.
        scores_budget_bytes: Peak bytes the transient score matrix may occupy.
    """
    mla_indexer_ragged_mxfp4_paged[
        num_heads,
        top_k,
        mask_str,
        page_size,
        scores_dtype,
        kpool,
        write_logical,
        Topk,
    ](
        output_indices,
        q,
        q_sf,
        weights,
        input_row_offsets,
        k_cache,
        ctx,
        remap,
        NoScoreHook(),
        scores_budget_bytes,
    )


@inline(.always)
def mla_indexer_ragged_mxfp4_paged[
    oi_layout: TensorLayout,
    q_layout: TensorLayout,
    qsf_layout: TensorLayout,
    w_layout: TensorLayout,
    iro_layout: TensorLayout,
    R: SparseIndexRemap,
    H: IndexerScoreHook,
    //,
    num_heads: Int,
    top_k: Int,
    mask_str: StaticString,
    page_size: Int,
    scores_dtype: DType = _SCORES_DTYPE,
    kpool: Int = 1,
    write_logical: Bool = False,
    Topk: IndexerTopk = BlockSplitTopk,
](
    output_indices: MutTileTensor[.int32, oi_layout, _],
    q: ImmTileTensor[.uint8, q_layout, _],
    q_sf: ImmTileTensor[.uint32, qsf_layout, _],
    weights: ImmTileTensor[.float32, w_layout, _],
    input_row_offsets: ImmTileTensor[.uint32, iro_layout, _],
    k_cache: MXFP4IndexKCache[page_size],
    ctx: DeviceContext,
    remap: R,
    score_hook: H,
    scores_budget_bytes: Int = _SCORES_BUDGET_BYTES,
) raises:
    """Computes the MXFP4 indexer's top-k key positions with a score hook.

    The hook sees each score in the scorer's epilogue as it is stored for the
    top-k, so it can read the scores or drop keys from the selection. The op
    scores at most `score_hook.window_rows()` rows per window and calls the
    hook's `after_window` once each window is scored.

    Parameters:
        oi_layout: Layout of the top-k index output.
        q_layout: Layout of the query tensor.
        qsf_layout: Layout of the query block scales.
        w_layout: Layout of the per-head weights.
        iro_layout: Layout of the ragged query row offsets.
        R: Type of `remap`.
        H: Type of `score_hook`.
        num_heads: Index heads per token; 32.
        top_k: Key positions to select per token.
        mask_str: `MaskName.NULL` or `MaskName.CAUSAL`.
        page_size: Keys per cache page.
        scores_dtype: Element type of the transient score matrix.
        kpool: Tokens per cache row, the compression ratio.
        write_logical: Whether to write `output_indices`.
        Topk: The top-k over the score rows.

    Args:
        output_indices: Dense `[total_seq_len, top_k]` top-k output.
        q: Packed E2M1 queries, `[total_seq_len, num_heads, 64]`.
        q_sf: Query block scales, `[total_seq_len, num_heads]`.
        weights: Per-head weights, `[total_seq_len, num_heads]`.
        input_row_offsets: Ragged row offsets for queries `[batch_size + 1]`.
        k_cache: The index keys, from `mxfp4_index_k_cache`.
        ctx: Device context.
        remap: Second output beside each logical index.
        score_hook: Runs on every score before the top-k.
        scores_budget_bytes: Peak bytes the transient score matrix may occupy.
    """
    _mla_indexer_paged_impl[
        DType.uint8,
        num_heads,
        MXFP4_INDEX_VALUE_BYTES,
        top_k,
        mask_str,
        scores_dtype,
        kpool,
        write_logical,
        Topk,
    ](
        output_indices,
        q,
        weights,
        input_row_offsets,
        k_cache,
        ctx,
        remap,
        score_hook,
        scores_budget_bytes,
        q_sf.unsafe_ptr().as_unsafe_any_origin(),
    )


@inline(.always)
def _mla_indexer_paged_impl[
    oi_layout: TensorLayout,
    q_layout: TensorLayout,
    qs_layout: TensorLayout,
    iro_layout: TensorLayout,
    CacheT: KVCacheT,
    R: SparseIndexRemap,
    H: IndexerScoreHook,
    //,
    dtype: DType,
    num_heads: Int,
    depth: Int,
    top_k: Int,
    mask_str: StaticString,
    scores_dtype: DType,
    kpool: Int,
    write_logical: Bool,
    Topk: IndexerTopk = BlockSplitTopk,
](
    output_indices: MutTileTensor[.int32, oi_layout, _],
    q: ImmTileTensor[dtype, q_layout, _],
    q_s: ImmTileTensor[.float32, qs_layout, _],
    input_row_offsets: ImmTileTensor[.uint32, iro_layout, _],
    k_cache: CacheT,
    ctx: DeviceContext,
    remap: R,
    score_hook: H,
    scores_budget_bytes: Int,
    q_sf: OptionalPointer[UInt32, ImmutAnyOrigin] = None,
) raises:
    """Scores, bounds and selects the indexer's top-k over one paged K cache.

    The body of both public entries: `mla_indexer_ragged_float8_paged` over a
    KV collection's layer and `mla_indexer_ragged_mxfp4_paged` over packed
    MXFP4 pages. See the former for the parameters; `q_sf` carries MXFP4's
    query block scales and `q_s` its per-head weights. `score_hook` runs on
    every score in the MXFP4 scorer's epilogue.
    """
    comptime CacheType = CacheT
    comptime assert (
        top_k <= PERSISTENT_TOPK_MAX_N
    ), "the bounded top-k selects at most PERSISTENT_TOPK_MAX_N per row"

    # Only NULL (no mask) and CAUSAL masks are supported
    comptime assert (
        mask_str == MaskName.NULL.name or mask_str == MaskName.CAUSAL.name
    ), "mask_str must be either MaskName.NULL or MaskName.CAUSAL"

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    var total_seq_len = Int(q.dim[0]())
    if total_seq_len == 0:
        return

    # max_new_tokens is used for grid dimensions (maximum possible new tokens)
    var max_new_tokens = Int(k_cache.max_prompt_length())

    var k_operand = KVCacheMHAOperand(k_cache)
    var ks_operand = KVCacheScalesMHAOperand(k_cache)

    # MXFP4 has no other scorer: its tile is 32 heads at 64 bytes a row, and
    # it splits a page smaller than a key tile (ratio-2 pages of 64 rows).
    comptime FP4 = _is_mxfp4[dtype]()
    comptime use_sm100_scorer = ctx.target._has_blackwell_tcgen05() and (
        (
            num_heads == 32
            and depth == 64
            and _k_tile_fits_pages[
                _BM_KEY, type_of(k_operand).page_size, 1, split_pages=True
            ]()
        ) if FP4 else (
            num_heads in (64, 32, 8, 4)
            and depth == 128
            and _k_tile_fits_pages[
                _BM_KEY,
                type_of(k_operand).page_size,
                # `pool_row_tokens`, derived below.
                1 if CacheType.kv_params.head_size == depth else kpool,
            ]()
        )
    )
    comptime assert (
        not H.enabled or FP4
    ), "score hooks run in the MXFP4 scorer's epilogue"
    comptime assert not FP4 or use_sm100_scorer, (
        "the MXFP4 indexer runs only on the SM100 scorer: 32 heads of 128"
        " values, on pages of a whole number of 128-key tiles or a power-of-two"
        " fraction of one"
    )

    # gfx950 MFMA scorer: same score-buffer contract as SM100 (fused causal
    # guard, all slots written). Per-key-row (any page size); num_heads >= 32.
    comptime use_amd_scorer = (
        _has_gfx950_accelerator()
        and num_heads % 32 == 0
        and depth % 64 == 0
        and _AMD_DMA_PASS_BYTES % depth == 0
        and dtype == .float8_e4m3fn
        and type_of(k_operand).dtype == .float8_e4m3fn
    )

    comptime assert (
        kpool == 1 or not R.enabled
    ), "a remap needs token positions; pooled indices are pool ids"
    comptime assert (
        write_logical or R.enabled
    ), "skipping the logical indices leaves only the remap to write"
    comptime assert kpool == 1 or use_sm100_scorer or use_amd_scorer, (
        "pooled indexing (kpool > 1) is implemented only on the SM100 and"
        " gfx950 tensor-core scorers; the scalar fallback walks token rows"
    )

    # A pooled cache comes in two shapes: one `depth`-wide row per pool, or
    # (GLM-5.3-Flash) one `depth / kpool`-wide row per token with each pool's
    # key spread over its tokens' rows. The SM100 and AMD scorers read either.
    comptime cache_row_width = CacheType.kv_params.head_size
    comptime assert (
        cache_row_width == depth or cache_row_width * kpool == depth
    ), "the indexer cache row must hold a whole key or one token's share of it"
    comptime pool_row_tokens = 1 if cache_row_width == depth else kpool
    # The gfx950 scorer DMAs a wave's 32 keys from one page at a fixed
    # `head_size` row stride, which holds only for a single-head cache.
    comptime assert not use_amd_scorer or CacheType.kv_params.num_heads == 1, (
        "the gfx950 scorer reads a page's key rows at a head_size stride, so"
        " the indexer cache must hold one head"
    )

    # The scalar fallback is f32-only, so `scores_dtype` is
    # a preference, not a contract: no caller can observe which arm ran, and an
    # assert would reject every non-SM100 instantiation of a bf16 default. Must
    # resolve BEFORE `scores_align` below, which keys the row stride -- and
    # through it the scorer's dispatch -- on it.
    comptime scores_dt = (scores_dtype if use_sm100_scorer else DType.float32)

    # An upper bound on the candidate rows per token. Under graph-capture
    # replay this holds the capture-time bound, far above the batch's real
    # lengths, so use it only to size allocations and grid dims, never to size
    # per-step work. With `kpool > 1` a cache row is a pooled key covering
    # `kpool` tokens, so counts from here down are pool-granular, `top_k`
    # included.
    #
    # Rounded UP to whole 16-byte groups of the score dtype: this number is the
    # buffer's ROW STRIDE and the top-k reads each row as 16-byte vectors based
    # at `row * stride`, so a partial group under-aligns every odd row and drops
    # it to the scalar path. The round-up adds at most `scores_align - 1`
    # columns and stays a valid upper bound -- the top-k's row bound clamps
    # each row to its real key count and pads past it.
    #
    # It can move the scorer's dispatch: `_is_keysplit_shape` compares
    # `ceildiv(max_num_keys, 128)`, so a depth landing in a 128-key tile's last
    # `scores_align - 1` columns reads as one tile deeper.
    comptime scores_align = _elems_per_16b[scores_dt]()
    var max_num_keys = align_up(
        (Int(k_cache.max_context_length()) + max_new_tokens) // kpool,
        scores_align,
    )

    # Per-batch KV cache lengths (cached-prefix length), for the top-k's row
    # bounds.
    var cache_lengths = k_cache.cache_lengths_nd()

    comptime use_causal_mask = mask_str != MaskName.NULL.name

    # The score matrix is `total_seq_len x max_num_keys` f32 -- at long context
    # the largest allocation this op makes by an order of magnitude (~1.2 GB at
    # 4096 tokens over 76K keys) and what caps `--max-batch-input-tokens`. Score
    # it one ROW WINDOW at a time into one budget-sized buffer instead: a
    # chunk's rows are consumed by the top-k before the next overwrites them, so
    # peak scratch is `rows_per_chunk x max_num_keys` however large the batch
    # grows.
    #
    # Looping on the host is safe because prefill is never under device graph
    # capture (built at a fixed decode query width; replay rejects any other).
    # The window is over ROWS, not requests: `input_row_offsets` is device data
    # the host cannot read without a sync, while row boundaries are host-known
    # and every stage after the scorer is row-parallel over a base pointer and a
    # count, so they slice by pointer arithmetic alone.
    #
    # The SM100 and gfx950 scorers take a window; the scalar fallback stays one
    # whole-batch chunk.
    var rows_per_chunk = total_seq_len
    comptime if use_sm100_scorer or use_amd_scorer:
        var row_bytes = max_num_keys * size_of[scores_dt]()
        # Largest window the budget allows, remainder to a short final chunk.
        # Spreading the rows evenly instead -- same chunk count, no runt --
        # reads better and measured 8% SLOWER at 2048 and 4096 tokens: a chunk's
        # grid is `batch x ceildiv(min(max_seq_len, chunk_rows), N_TOKENS)`, so
        # a short final chunk launches a proportionally short grid and costs
        # almost nothing, while an even split turns that nearly-free tail into a
        # full-price one.
        rows_per_chunk = clamp(
            min(scores_budget_bytes // row_bytes, score_hook.window_rows()),
            1,
            total_seq_len,
        )

    var scores_buf = ctx.enqueue_create_buffer[scores_dt](
        rows_per_chunk * max_num_keys
    )

    for chunk_begin in range(0, total_seq_len, rows_per_chunk):
        var chunk_rows = min(rows_per_chunk, total_seq_len - chunk_begin)
        var scores_tile = TileTensor(
            scores_buf,
            row_major(chunk_rows, max_num_keys),
        )

        comptime if use_amd_scorer:
            comptime assert (
                scores_dt == .float32
            ), "the AMD scorer writes float32 scores"
            var scores_tile_f32 = TileTensor(
                scores_buf.unsafe_ptr().unsafe_bitcast[Float32](),
                row_major(chunk_rows, max_num_keys),
            )
            fp8_index_score_amd[
                dtype,
                type_of(k_operand),
                type_of(ks_operand),
                num_heads,
                depth,
                _is_cache_length_accurate=False,
                causal=mask_str == MaskName.CAUSAL.name,
                kpool=kpool,
                pool_row_tokens=pool_row_tokens,
            ](
                scores_tile_f32,
                q,
                q_s.as_imm(),
                k_operand,
                ks_operand,
                input_row_offsets,
                batch_size,
                max_new_tokens,
                max_num_keys,
                ctx,
                chunk_begin,
            )
        elif use_sm100_scorer:
            # The split `(block, row_in_block)` descriptor: folding the block
            # into the row spans the whole shared slab and leaves signed 32
            # bits on a large pool. See `PagedKVCache.create_paged_tma_tile`.
            # `rebind` because the collection's `CacheType.dtype` is not
            # syntactically `dtype`, though the route already requires them
            # equal -- the same rebind the scorer does on its own tiles.
            comptime k_box_rows = _k_box_rows[
                _BM_KEY, type_of(k_operand).page_size, pool_row_tokens
            ]()
            var k_tma_tile = rebind[KTMATileT[dtype, k_box_rows, depth]](
                k_cache.create_paged_tma_tile[
                    _index_swizzle[dtype, depth](),
                    BN=k_box_rows,
                    BK=depth,
                    pool_row_tokens=pool_row_tokens,
                ](ctx)
            )
            comptime if _is_mxfp4[dtype]():
                mxfp4_index_score_sm100[
                    type_of(k_operand),
                    type_of(ks_operand),
                    num_heads,
                    _is_cache_length_accurate=False,
                    kpool=kpool,
                    pool_row_tokens=pool_row_tokens,
                    k_box_rows=k_box_rows,
                ](
                    scores_tile,
                    q,
                    q_sf.value(),
                    q_s,
                    k_operand,
                    ks_operand,
                    rebind[KTMATileT[dtype, k_box_rows, 64]](k_tma_tile),
                    input_row_offsets,
                    batch_size,
                    max_new_tokens,
                    max_num_keys,
                    mask_str == MaskName.CAUSAL.name,
                    ctx,
                    chunk_begin,
                    score_hook.window(ctx, chunk_begin, chunk_rows),
                )
                score_hook.after_window(ctx, chunk_begin, chunk_rows)
            else:
                fp8_index_score_sm100[
                    dtype,
                    type_of(k_operand),
                    type_of(ks_operand),
                    num_heads,
                    depth,
                    _is_cache_length_accurate=False,
                    # Speculative-decode tile: 3 divides a 6-token MTP step, which
                    # the default 4-token tile at nh=32 covers only by spending 256
                    # MMA columns on 192 live ones. Inert at every other head count.
                    #
                    # The `max_seq_len` this entry passes is `max_prompt_length()`,
                    # the batch maximum of NEW tokens -- not a context length -- so
                    # the reachability bound is "no request in the batch brings more
                    # than 9 new tokens", not "not a prefill". A short prompt, a
                    # chunked-prefill final chunk, or a prefix-cache-hit tail of 3,
                    # 6 or 9 tokens does reach this tile when some entry's cache
                    # makes `max_num_keys` deep enough to open the key-split arm.
                    # That is intended: at <= 9 query tokens against >= 8065 keys
                    # the launch is decode-shaped by every measure the route uses,
                    # and it is already on the key-split arm without this hint --
                    # the tile only makes its MMA columns exact. What cannot happen
                    # is a many-token prefill landing here.
                    N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
                    kpool=kpool,
                    pool_row_tokens=pool_row_tokens,
                    k_box_rows=k_box_rows,
                ](
                    scores_tile,
                    q,
                    q_s,
                    k_operand,
                    ks_operand,
                    k_tma_tile,
                    input_row_offsets,
                    batch_size,
                    max_new_tokens,
                    max_num_keys,
                    mask_str == MaskName.CAUSAL.name,
                    ctx,
                    chunk_begin,
                )
        else:
            comptime assert num_heads % 16 == 0, (
                "the scalar fp8_index_kernel tiles heads by thread_dim_y == 8"
                " and is unvalidated below 16 heads; num_heads in {4, 8}"
                " requires the SM100 tensor-core path"
            )
            comptime block_tile_shape: Array[Int, 2] = [512, 128]
            comptime BM = block_tile_shape[0]
            comptime BN = block_tile_shape[1]
            comptime smem_use = size_of[
                IndexSmemStorage[dtype, num_heads, depth, BN]
            ]()
            comptime smem_available = ctx.default_device_info.shared_memory_per_multiprocessor - 1024

            comptime kernel = fp8_index_kernel[
                dtype,
                type_of(scores_tile).LayoutType,
                q_layout,
                qs_layout,
                type_of(k_operand),
                type_of(ks_operand),
                block_tile_shape,
                iro_layout,
                num_heads,
                depth,
            ]

            ctx.enqueue_function[kernel](
                scores_tile,
                q,
                q_s,
                k_operand,
                ks_operand,
                input_row_offsets,
                grid_dim=(
                    batch_size,
                    max_new_tokens,
                    ceildiv(max_num_keys, BM),
                ),
                block_dim=(16, 8, 1),
                shared_mem_bytes=smem_use,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem_available)
                ),
            )

        # The bounded top-k writes `output_indices` directly: a row's bound is
        # the `indexer_key_bound` the scorer wrote up to, columns past it are
        # never selected, and a row with fewer live keys than `top_k` gets a
        # `-1` tail -- including every row when `max_num_keys < top_k`, which
        # the single-block path pads however short `N` is.
        var row_bound = _IndexerRowBound[
            use_causal_mask, kpool, R, write_logical
        ](
            input_row_offsets.unsafe_ptr().as_unsafe_any_origin(),
            cache_lengths.unsafe_ptr().as_unsafe_any_origin(),
            batch_size,
            chunk_begin,
            remap.offset(chunk_begin * top_k),
        )
        Topk.select(
            ctx,
            scores_buf.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            output_indices.unsafe_ptr()
            .unsafe_offset(chunk_begin * top_k)
            .as_unsafe_any_origin(),
            max_num_keys,
            top_k,
            chunk_rows,
            row_bound,
        )

    _ = scores_buf
