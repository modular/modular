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
"""Gathers rows of a paged KV cache by an explicit per-row slot list.

The read side of ``kv_cache_store_ragged``: row ``r`` of ``slots`` belongs to
the request ``b`` with ``row_offsets[b] <= r < row_offsets[b + 1]``, and
``output[r, j, :]`` is the cache row at slot ``slots[r, j]`` of that request,
resolved through the request's lookup table exactly as ``load`` resolves a
token index. A leaf whose buffer holds ``slots_per_page`` rows per page is
paged by that count, because ``load`` reads the page size off the block shape,
so its slot is an entry index rather than a token index.

This is a pure copy in the cache's own dtype, so the result is bit-identical
to indexing the block buffer by hand, on every target.
"""

from std.sys import size_of
from std.sys.info import simd_width_of

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu

from kv_cache.context_parallel import ContextParallelLayout
from kv_cache.types import KVCacheT
from layout import Coord, TileTensor

from nn._ragged_utils import get_batch_from_row_offsets


def kv_cache_gather_rows_ragged[
    cache_t: KVCacheT,
    dtype: DType,
    //,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    slots: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    row_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    cache: cache_t,
    ctx: DeviceContext,
) raises:
    """Copies ``cache`` rows at ``slots`` into ``output``.

    Parameters:
        cache_t: The key or value cache type (inferred); one head per slot.
        dtype: The output element type (inferred); must be the cache's.

    Args:
        output: ``[num_rows, num_slots, head_dim]``, the last axis contiguous.
        slots: ``[num_rows, num_slots]`` slot indices. Every slot must be
            non-negative and inside its request's allocated pages.
        row_offsets: ``[batch + 1]`` ragged offsets mapping each row of
            ``slots`` to a request.
        cache: The key or value cache of one layer.
        ctx: Device context used to enqueue the GPU kernel.
    """

    comptime head_dim = cache_t.kv_params.head_size
    comptime assert (
        dtype == cache_t.dtype
    ), "gather_rows copies rows in the cache's own dtype"
    comptime assert (
        cache_t.kv_params.num_heads == 1
    ), "gather_rows reads caches that hold one head per slot"
    comptime assert output.flat_rank == 3 and slots.flat_rank == 2
    comptime assert row_offsets.flat_rank == 1

    var num_rows = Int(slots.dim[0]())
    var num_slots = Int(slots.dim[1]())
    debug_assert(
        Int(output.dynamic_stride(2)) == 1,
        "output must be contiguous along head_dim",
    )
    if num_rows == 0 or num_slots == 0:
        return

    # Same closure form as `kv_cache_store_ragged`: a KV cache view cannot be
    # captured by a unified closure yet.
    def copy_row[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var r = Int(idx[0])
        var j = Int(idx[1])
        var d = Int(idx[2])
        var b = get_batch_from_row_offsets(row_offsets, r)
        var slot = Int(slots[r, j])
        # `dtype == cache_t.dtype` is asserted above; the two are still
        # distinct symbols to the type checker.
        output.store[alignment=alignment](
            (r, j, d),
            rebind[SIMD[dtype, width]](cache.load[width=width](b, 0, slot, d)),
        )

    comptime simd_width = simd_width_of[cache_t.dtype, target=ctx.target]()

    elementwise[
        simd_width,
        _trace_description="kv_cache_gather_rows_ragged",
    ](copy_row, Coord(num_rows, num_slots, head_dim), ctx)


# ===-----------------------------------------------------------------------===#
# Prefill context parallelism: moving a context-parallel prefix between ranks
# ===-----------------------------------------------------------------------===#


@always_inline
def _pcp_row_words[
    cache_t: KVCacheT, scale_row_elems: Int
]() -> Tuple[Int, Int]:
    """Returns a cache row's value and scale widths in 32-bit words."""
    # The pack and unpack read a row's scales at head 0 only.
    comptime assert (
        cache_t.kv_params.num_heads == 1
    ), "PCP row packing reads caches that hold one head per slot"
    comptime value_bytes = cache_t.kv_params.head_size * size_of[
        cache_t.dtype
    ]()
    comptime assert (
        value_bytes % 4 == 0
    ), "a cache row must be whole 32-bit words"
    comptime if scale_row_elems == 0:
        return (value_bytes // 4, 0)
    else:
        comptime scale_bytes = scale_row_elems * size_of[cache_t.scale_dtype]()
        comptime assert (
            scale_bytes % 4 == 0
        ), "a cache row's scales must be whole 32-bit words"
        return (value_bytes // 4, scale_bytes // 4)


def kv_cache_pack_owned_rows[
    cache_t: KVCacheT,
    //,
    *,
    scale_row_elems: Int,
    col_offset: Int,
    dcp_size: Int,
    dcp_interleave: Int,
](
    output: TileTensor[mut=True, DType.uint8, address_space=.GENERIC, ...],
    row_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    cache: cache_t,
    dcp_rank: Int,
    ctx: DeviceContext,
) raises:
    """Packs the cached rows this context-parallel rank owns into `output`.

    Request `b` owns output rows `[row_offsets[b], row_offsets[b + 1])`, at
    least `max_local_len(cache_length(b))` of them: output row
    `row_offsets[b] + j` takes the request's local slot `j`, the global
    position `global_pos(j)` this rank stores (`ContextParallelLayout`). A
    slot past the rank's `local_len` is left as it is. Each row is the
    cache's raw bytes, the value row then its `scale_row_elems` scales, at
    byte column `col_offset` of `output`; rows past `row_offsets[B]` are
    left as they are.

    Every rank of a group packs the same row layout, so after an all-gather
    the row of position `p` of request `b` is
    `owner(p) * N + row_offsets[b] + local_slot(p)`, with `N` the rows each
    rank packed (`kv_cache_unpack_prefix_rows`).

    Parameters:
        cache_t: The paged cache type (inferred), one head per row.
        scale_row_elems: Scales per row; 0 for an unscaled cache.
        col_offset: First byte column of `output` the rows land at.
        dcp_size: Ranks of the context-parallel group.
        dcp_interleave: Consecutive positions a rank owns in turn.

    Args:
        output: `[N, width]` bytes, contiguous rows.
        row_offsets: `[B + 1]` output row offsets per request.
        cache: One layer of this rank's context-parallel cache, whose
            `cache_length` is each request's global prefix length.
        dcp_rank: This rank in `[0, dcp_size)`.
        ctx: Device context used to enqueue the GPU kernel.
    """
    comptime words = _pcp_row_words[cache_t, scale_row_elems]()
    comptime value_words = words[0]
    comptime row_words = words[0] + words[1]
    comptime assert col_offset % 4 == 0, "col_offset must be word aligned"
    comptime assert row_offsets.flat_rank == 1

    var num_rows = Int(output.dim[0]())
    var stride = Int(output.dynamic_stride(0))
    if num_rows == 0:
        return
    if stride % 4 != 0 or col_offset + 4 * row_words > Int(output.dim[1]()):
        raise Error("kv_cache_pack_owned_rows: output rows are too narrow")
    var out_ptr = output.unsafe_ptr()
    var cp = ContextParallelLayout[dcp_size, dcp_interleave](dcp_rank)

    def copy_word[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var r = Int(idx[0])
        var w = Int(idx[1])
        if r >= Int(row_offsets[row_offsets.num_elements() - 1]):
            return
        var b = get_batch_from_row_offsets(row_offsets, r)
        var slot = r - Int(row_offsets[b])
        if slot >= cp.local_len(Int(cache.cache_length(b))):
            return
        var dst = out_ptr.unsafe_offset(r * stride + col_offset).unsafe_bitcast[
            UInt32
        ]()
        if w < value_words:
            var src = cache.block_paged_ptr[1](b, slot, 0).unsafe_bitcast[
                UInt32
            ]()
            dst[w] = src[w]
        else:
            comptime if row_words > value_words:
                var src = cache.scales_block_paged_ptr(
                    b, slot, 0, 0
                ).unsafe_bitcast[UInt32]()
                dst[w] = src[w - value_words]

    elementwise[1, _trace_description="kv_cache_pack_owned_rows"](
        copy_word, Coord(num_rows, row_words), ctx
    )


def kv_cache_unpack_prefix_rows[
    cache_t: KVCacheT,
    //,
    *,
    scale_row_elems: Int,
    col_offset: Int,
    dcp_size: Int,
    dcp_interleave: Int,
](
    gathered: TileTensor[mut=False, DType.uint8, address_space=.GENERIC, ...],
    row_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    cache: cache_t,
    ctx: DeviceContext,
) raises:
    """Writes every rank's packed prefix rows into a whole-context cache.

    `gathered` is the group's all-gather of `kv_cache_pack_owned_rows`,
    `[dcp_size * N, width]`, rank-major. Its row `s * N + row_offsets[b] +
    j` holds rank `s`'s local slot `j` of request `b`, global position
    `p = global_pos_s(j)`; it lands at position `p` of request `b` in
    `cache`, a cache outside any context-parallel group, if `p` is below
    `cache_length(b)`.

    Parameters:
        cache_t: The paged cache type (inferred), one head per row.
        scale_row_elems: Scales per row; 0 for an unscaled cache.
        col_offset: First byte column of `gathered` the rows sit at.
        dcp_size: Ranks of the context-parallel group.
        dcp_interleave: Consecutive positions a rank owns in turn.

    Args:
        gathered: `[dcp_size * N, width]` bytes, contiguous rows.
        row_offsets: `[B + 1]` row offsets of each rank's packed rows.
        cache: One layer of the whole-context cache, whose `cache_length`
            is each request's prefix length.
        ctx: Device context used to enqueue the GPU kernel.
    """
    comptime words = _pcp_row_words[cache_t, scale_row_elems]()
    comptime value_words = words[0]
    comptime row_words = words[0] + words[1]
    comptime assert col_offset % 4 == 0, "col_offset must be word aligned"
    comptime assert row_offsets.flat_rank == 1

    var total = Int(gathered.dim[0]())
    var stride = Int(gathered.dynamic_stride(0))
    if total == 0:
        return
    if total % dcp_size != 0:
        raise Error(
            "kv_cache_unpack_prefix_rows: the gathered rows do not split"
            " evenly over the group"
        )
    if stride % 4 != 0 or col_offset + 4 * row_words > Int(gathered.dim[1]()):
        raise Error("kv_cache_unpack_prefix_rows: rows are too narrow")
    var per_rank = total // dcp_size
    var in_ptr = gathered.unsafe_ptr()

    def copy_word[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var g = Int(idx[0])
        var w = Int(idx[1])
        var owner, k = divmod(g, per_rank)
        if k >= Int(row_offsets[row_offsets.num_elements() - 1]):
            return
        var b = get_batch_from_row_offsets(row_offsets, k)
        var slot = k - Int(row_offsets[b])
        var pos = ContextParallelLayout[dcp_size, dcp_interleave](
            owner
        ).global_pos(slot)
        if pos >= Int(cache.cache_length(b)):
            return
        var src = in_ptr.unsafe_offset(g * stride + col_offset).unsafe_bitcast[
            UInt32
        ]()
        if w < value_words:
            var dst = cache.block_paged_ptr[1](b, pos, 0).unsafe_bitcast[
                UInt32
            ]()
            dst[w] = src[w]
        else:
            comptime if row_words > value_words:
                var dst = cache.scales_block_paged_ptr(
                    b, pos, 0, 0
                ).unsafe_bitcast[UInt32]()
                dst[w - value_words] = src[w]

    elementwise[1, _trace_description="kv_cache_unpack_prefix_rows"](
        copy_word, Coord(total, row_words), ctx
    )
