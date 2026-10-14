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

from std.bit import pop_count
from std.builtin.device_passable import DeviceTypeEncoder
from std.sys import get_defined_int, size_of
from std.sys.info import _has_blackwell_tcgen05
from std.math import align_up, ceildiv, clamp

from layout import (
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.tile_tensor import ImmTileTensor, MutTileTensor

from max.gpu import WARP_SIZE, lane_id
import max.gpu.primitives.warp as warp
from max.gpu.host import DeviceContext, FuncAttribute

from kv_cache.paged_sparse_kv_index_remap import NoRemap, SparseIndexRemap
from kv_cache.types import KVCollectionT

from nn.index_fp8 import (
    IndexSmemStorage,
    _has_gfx950_accelerator,
    fp8_index_kernel,
    fp8_index_score_amd,
)
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    _INDEX_SWIZZLE,
    KTMATileT,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from nn.attention.gpu.sparse_index_fp8_sm100_prefill import (
    _k_box_rows,
    _k_tile_fits_pages,
)
from nn.attention.mha_mask import MaskName
from nn.attention.mha_operand import KVCacheMHAOperand, KVCacheScalesMHAOperand
from nn.attention.mha_utils import indexer_key_bound
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    RowBound,
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


struct _IndexerSink[R: SparseIndexRemap, write_logical: Bool](TopkSink):
    """Writes a top-k slot's logical index and, through `R`, its remap.

    Parameters:
        R: Type of the remap output.
        write_logical: Whether the logical index is written at all.
    """

    var logical: UnsafePointer[Int32, MutUntrackedOrigin]
    """Flat logical output, `[rows, top_k]`."""
    var remap: Self.R
    """Second output derived from each logical index."""
    var batch: Int
    """Batch entry of the row this sink serves."""

    @inline(.always)
    def __init__(
        out self,
        logical: UnsafePointer[Int32, MutAnyOrigin],
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

    var row_offsets: UnsafePointer[UInt32, ImmUntrackedOrigin]
    """Ragged query row offsets, `batch_size + 1` entries."""
    var cache_lengths: UnsafePointer[UInt32, ImmUntrackedOrigin]
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
        row_offsets: UnsafePointer[UInt32, ImmutAnyOrigin],
        cache_lengths: UnsafePointer[UInt32, ImmutAnyOrigin],
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
        self, row: Int, out_idxs: UnsafePointer[Int32, MutAnyOrigin]
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
        # The owning entry is the number of entries that end at or before the
        # token. Offsets are monotone, so a warp counts a whole warp-width of
        # them with one vote: a single load latency where a serial walk takes
        # one dependent load per entry.
        comptime vote_t = DType.uint32 if WARP_SIZE == 32 else DType.uint64
        var batch_idx = 0
        for base in range(0, batch, WARP_SIZE):
            var b = base + Int(lane_id())
            var ended = b < batch and Int(self.row_offsets[b + 1]) <= token
            batch_idx += Int(pop_count(warp.vote[vote_t](ended)))
        var q_start = Int(self.row_offsets[batch_idx])
        var seq_len = Int(self.row_offsets[batch_idx + 1]) - q_start
        var cache_len = Int(self.cache_lengths[batch_idx])
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

    var k_cache = k_collection.get_key_cache(Int(layer_idx))

    # max_new_tokens is used for grid dimensions (maximum possible new tokens)
    var max_new_tokens = Int(k_cache.max_prompt_length())

    var k_operand = KVCacheMHAOperand(k_cache)
    var ks_operand = KVCacheScalesMHAOperand(k_cache)

    comptime use_sm100_scorer = (
        _has_blackwell_tcgen05()
        and num_heads in (64, 32, 8, 4)
        and depth == 128
        and _k_tile_fits_pages[
            _BM_KEY,
            type_of(k_operand).page_size,
            # `pool_row_tokens`, derived below.
            1 if CacheType.kv_params.head_size == depth else kpool,
        ]()
    )

    # gfx950 MFMA scorer: same score-buffer contract as SM100 (fused causal
    # guard, all slots written). Per-key-row (any page size); num_heads >= 32.
    comptime use_amd_scorer = (
        _has_gfx950_accelerator()
        and num_heads % 32 == 0
        and depth % 64 == 0
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
    # Only the SM100 scorers take a window, so the scalar fallback stays one
    # whole-batch chunk.
    var rows_per_chunk = total_seq_len
    comptime if use_sm100_scorer:
        var row_bytes = max_num_keys * size_of[scores_dt]()
        # Largest window the budget allows, remainder to a short final chunk.
        # Spreading the rows evenly instead -- same chunk count, no runt --
        # reads better and measured 8% SLOWER at 2048 and 4096 tokens: a chunk's
        # grid is `batch x ceildiv(min(max_seq_len, chunk_rows), N_TOKENS)`, so
        # a short final chunk launches a proportionally short grid and costs
        # almost nothing, while an even split turns that nearly-free tail into a
        # full-price one.
        rows_per_chunk = clamp(
            scores_budget_bytes // row_bytes, 1, total_seq_len
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
                scores_buf.unsafe_ptr().bitcast[Float32](),
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
                    _INDEX_SWIZZLE,
                    BN=k_box_rows,
                    BK=depth,
                    pool_row_tokens=pool_row_tokens,
                ](ctx)
            )
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
        persistent_topk_block_split[RB=type_of(row_bound)](
            ctx,
            scores_buf.unsafe_ptr().as_imm().as_unsafe_any_origin(),
            output_indices.unsafe_ptr()
            .unsafe_offset(chunk_begin * top_k)
            .as_unsafe_any_origin(),
            max_num_keys,
            top_k,
            chunk_rows,
            Optional(row_bound),
        )

    _ = scores_buf
