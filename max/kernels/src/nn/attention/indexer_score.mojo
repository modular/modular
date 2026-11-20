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
"""Lightning-indexer scores read straight out of a paged compressed leaf.

DeepSeek-V4's indexer scores every query row ``t`` against the compressed
entries it may see:

    score[t, c] = sum_h relu(q[t, h, :] . k[e(c), :]) * weights[t, h]

The candidate axis has ``2 * cap`` columns in the model's order: columns
``0 .. cap - 1`` are the request's entries closed before this chunk (entry
``c``, live while ``c < base[b]``), columns ``cap .. 2 * cap - 1`` its fresh
windows (entry ``base[b] + c - cap``, live while that entry is below the
query's ``cutoff[t]``). Every live entry is already stored in the leaf -- the
fresh ones by the store op that runs before this one in the same graph -- so
the per-token candidate table is never built. Dead columns are written as 0;
callers mask them before ranking.

The sum runs over the heads this op is given. Under tensor parallelism each
device passes its share of the heads and the partial scores are all-reduced
before the top-k.

GPU: one block per (query row, ``block_size // lanes`` columns), ``lanes``
threads per column, each holding the accumulators of ``num_heads // lanes``
heads; the column's first thread adds the heads up in order. The row's query
and weights sit in shared memory; each thread streams its entry once. Blocks
with no live column only write zeros. Column tiles past grid y's limit fold
onto the same blocks, which loop over them. Up to 4 rows, when the heads
split into eighths, the few live columns are spread over more blocks and
threads (64 threads, 8 per column); otherwise one thread per column. The
CPU path is the same arithmetic in the same order.
"""

from std.math import ceildiv
from std.memory import unsafe_stack_allocation

import max.gpu.primitives.warp as warp
from max.gpu import WARP_SIZE, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.sync import barrier

from kv_cache.types import KVCacheT
from layout import TileTensor

comptime MAX_GRID_Y = 65535
"""The largest grid y a CUDA or HIP launch takes."""


@inline(.always)
def _batch_of_row(
    row: Int,
    row_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    num_batches: Int,
) -> Int:
    var r = UInt32(row)
    for b in range(num_batches):
        var lo = rebind[UInt32](row_offsets.load((b,)))
        var hi = rebind[UInt32](row_offsets.load((b + 1,)))
        if r >= lo and r < hi:
            return b
    return 0


def _indexer_score_cpu[
    cache_t: KVCacheT,
    q_type: DType,
    //,
    head_dim: Int,
    chunk: Int,
](
    output: TileTensor[mut=True, .float32, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, q_type, address_space=.GENERIC, ...],
    weights: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    row_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    base_t: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    cutoff_t: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    cache: cache_t,
    num_rows: Int,
    num_heads: Int,
    num_batches: Int,
    num_cand: Int,
):
    var cap = num_cand // 2
    for t in range(num_rows):
        var b = _batch_of_row(t, row_offsets, num_batches)
        var base = Int(rebind[Int32](base_t.load((b,))))
        var cutoff = Int(rebind[Int32](cutoff_t.load((t,))))
        for c in range(num_cand):
            var e: Int
            var live: Bool
            if c < cap:
                e = c
                live = c < base
            else:
                e = base + c - cap
                live = e < cutoff
            var score = Float32(0)
            if live:
                for h in range(num_heads):
                    var acc = Float32(0)
                    for d0 in range(0, head_dim, chunk):
                        var k = cache.load[width=chunk, output_dtype=.float32](
                            b, 0, e, d0
                        )
                        var qv = SIMD[.float32, chunk](0)
                        comptime for i in range(chunk):
                            qv[i] = rebind[Float32](
                                q.load((t, h, d0 + i)).cast[.float32]()
                            )
                        acc += (qv * k).reduce_add()
                    score += max(acc, Float32(0)) * rebind[Float32](
                        weights.load((t, h))
                    )
            output.store((t, c), score)


def indexer_score_ragged_paged[
    cache_t: KVCacheT,
    q_type: DType,
    //,
    num_heads: Int,
](
    output: TileTensor[mut=True, .float32, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, q_type, address_space=.GENERIC, ...],
    weights: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    base: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    cutoff: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    cache: cache_t,
    ctx: DeviceContext,
) raises:
    """Scores every query row against its live compressed entries.

    Parameters:
        cache_t: The compressed leaf's key cache type (inferred); one head,
            paged by entry through ``slots_per_page``.
        q_type: Query element type (inferred).
        num_heads: Query heads summed into each score.

    Args:
        output: ``[num_rows, num_cand]`` scores, ``num_cand`` even; columns
            ``num_cand // 2`` on are the fresh windows.
        q: ``[num_rows, num_heads, head_dim]``, the last axis contiguous.
        weights: ``[num_rows, num_heads]`` per-head weights, contiguous.
        input_row_offsets: ``[batch + 1]`` ragged row offsets.
        base: ``[batch]`` entries each request had closed before this chunk.
        cutoff: ``[num_rows]`` entries each query may see, counting from 0.
        cache: This layer's compressed leaf.
        ctx: Device context used to enqueue the GPU kernel.
    """
    comptime head_dim = cache_t.kv_params.head_size
    comptime chunk = 8
    comptime assert (
        cache_t.kv_params.num_heads == 1
    ), "the compressed leaf holds a single head"
    comptime assert head_dim % chunk == 0
    comptime assert output.flat_rank == 2 and q.flat_rank == 3
    comptime assert weights.flat_rank == 2
    comptime assert input_row_offsets.flat_rank == 1
    comptime assert base.flat_rank == 1 and cutoff.flat_rank == 1

    var num_rows = Int(q.dim[0]())
    var num_batches = Int(input_row_offsets.dim[0]()) - 1
    var num_cand = Int(output.dim[1]())
    debug_assert(
        Int(q.dim[1]()) == num_heads and Int(q.dim[2]()) == head_dim,
        "q must be [num_rows, num_heads, head_dim]",
    )
    debug_assert(num_cand % 2 == 0, "num_cand must be even")
    debug_assert(
        Int(q.dynamic_stride(2)) == 1
        and Int(weights.dynamic_stride(1)) == 1
        and Int(output.dynamic_stride(1)) == 1,
        "q, weights and output must be contiguous along the last axis",
    )
    if num_rows == 0 or num_cand == 0:
        return

    comptime if ctx.target.is_cpu():
        _indexer_score_cpu[head_dim=head_dim, chunk=chunk](
            output,
            q,
            weights,
            input_row_offsets,
            base,
            cutoff,
            cache,
            num_rows,
            num_heads,
            num_batches,
            num_cand,
        )
    else:
        comptime if num_heads % 8 == 0:
            if num_rows <= 4:
                _indexer_score_gpu[
                    num_heads=num_heads,
                    head_dim=head_dim,
                    chunk=chunk,
                    block_size=64,
                    lanes=8,
                ](
                    output,
                    q,
                    weights,
                    input_row_offsets,
                    base,
                    cutoff,
                    cache,
                    num_rows,
                    num_batches,
                    num_cand,
                    ctx,
                )
                return
        _indexer_score_gpu[
            num_heads=num_heads,
            head_dim=head_dim,
            chunk=chunk,
            block_size=128,
            lanes=1,
        ](
            output,
            q,
            weights,
            input_row_offsets,
            base,
            cutoff,
            cache,
            num_rows,
            num_batches,
            num_cand,
            ctx,
        )


def indexer_score_grid(
    num_rows: Int, num_cand: Int, cols: Int, max_col_blocks: Int = MAX_GRID_Y
) -> Tuple[Int, Int]:
    """The ``(rows, column blocks)`` grid of a launch of ``cols``-wide tiles.

    Args:
        num_rows: Query rows, one per grid x.
        num_cand: Candidate columns.
        cols: Columns per tile.
        max_col_blocks: The cap on grid y; the tiles past it fold.

    Returns:
        The grid's x and y.
    """
    return (num_rows, min(ceildiv(num_cand, cols), max_col_blocks))


def _indexer_score_gpu[
    cache_t: KVCacheT,
    q_type: DType,
    //,
    num_heads: Int,
    head_dim: Int,
    chunk: Int,
    block_size: Int,
    lanes: Int,
    max_col_blocks: Int = MAX_GRID_Y,
](
    output: TileTensor[mut=True, .float32, address_space=.GENERIC, ...],
    q: TileTensor[mut=False, q_type, address_space=.GENERIC, ...],
    weights: TileTensor[mut=False, .float32, address_space=.GENERIC, ...],
    input_row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    base: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    cutoff: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    cache: cache_t,
    num_rows: Int,
    num_batches: Int,
    num_cand: Int,
    ctx: DeviceContext,
) raises:
    comptime cols = block_size // lanes
    comptime heads_per_lane = num_heads // lanes
    comptime staged = num_heads * head_dim // (block_size * chunk)
    comptime assert heads_per_lane * lanes == num_heads
    comptime assert WARP_SIZE % lanes == 0
    comptime assert output.flat_rank == 2 and q.flat_rank == 3
    var num_batches_i32 = Int32(num_batches)
    var num_cand_i32 = Int32(num_cand)
    var grid = indexer_score_grid(num_rows, num_cand, cols, max_col_blocks)

    def kernel() {
        var output,
        var q,
        var weights,
        var input_row_offsets,
        var base,
        var cutoff,
        var cache,
        var num_batches_i32,
        var num_cand_i32,
    }:
        with PDL():
            var t = block_idx.x
            var tid = thread_idx.x
            var part = tid % lanes
            var n = Int(num_cand_i32)
            var cap = n // 2

            var b = _batch_of_row(t, input_row_offsets, Int(num_batches_i32))
            var base_b = Int(rebind[Int32](base.load((b,))))
            var cutoff_t = Int(rebind[Int32](cutoff.load((t,))))

            var sq = unsafe_stack_allocation[
                num_heads * head_dim, Float32, address_space=.SHARED
            ]()
            var sw = unsafe_stack_allocation[
                num_heads, Float32, address_space=.SHARED
            ]()
            var query_staged = False
            for tile in range(block_idx.y, ceildiv(n, cols), grid_dim.y):
                var c0 = tile * cols
                var c = c0 + tid // lanes

                # Live columns are [0, base) and [cap, cap + cutoff - base); a
                # tile that misses both skips the query load. The branches are
                # block-uniform, so no barrier is stranded.
                var c1 = c0 + cols
                if not (
                    c0 < base_b or (c1 > cap and c0 < cap + cutoff_t - base_b)
                ):
                    if part == 0 and c < n:
                        output.store((t, c), Float32(0))
                    continue

                if not query_staged:
                    comptime if staged * block_size * chunk == (
                        num_heads * head_dim
                    ):
                        # Every query load is issued before the first store.
                        var qs = Array[SIMD[q_type, chunk], staged](fill=0)
                        comptime for j in range(staged):
                            var i = (j * block_size + tid) * chunk
                            var h = i // head_dim
                            qs[j] = rebind[SIMD[q_type, chunk]](
                                q.load[width=chunk]((t, h, i - h * head_dim))
                            )
                        comptime for j in range(staged):
                            sq.unsafe_store(
                                (j * block_size + tid) * chunk,
                                qs[j].cast[.float32](),
                            )
                    else:
                        for i in range(tid, num_heads * head_dim, block_size):
                            var h = i // head_dim
                            sq.unsafe_store(
                                i,
                                rebind[Float32](
                                    q.load((t, h, i - h * head_dim)).cast[
                                        .float32
                                    ]()
                                ),
                            )
                    for h in range(tid, num_heads, block_size):
                        sw.unsafe_store(
                            h, rebind[Float32](weights.load((t, h)))
                        )
                    barrier()
                    query_staged = True

                var e = 0
                var live = False
                if c < cap:
                    e = c
                    live = c < base_b
                elif c < n:
                    e = base_b + c - cap
                    live = e < cutoff_t

                var acc = Array[Float32, heads_per_lane](fill=Float32(0))
                if live:
                    comptime for d0 in range(0, head_dim, chunk):
                        var k = cache.load[width=chunk, output_dtype=.float32](
                            b, 0, e, d0
                        )
                        comptime for hh in range(heads_per_lane):
                            var qv = sq.unsafe_load[width=chunk](
                                (part * heads_per_lane + hh) * head_dim + d0
                            )
                            acc[hh] += (qv * k).reduce_add()

                # The column's first lane adds every head's term in head order.
                var score = Float32(0)
                comptime for p in range(lanes):
                    comptime for hh in range(heads_per_lane):
                        var a: Float32
                        comptime if lanes == 1:
                            a = acc[hh]
                        else:
                            a = warp.shuffle_down(acc[hh], UInt32(p))
                        if live:
                            score += max(a, Float32(0)) * sw.unsafe_load(
                                p * heads_per_lane + hh
                            )
                if part == 0 and c < n:
                    output.store((t, c), score)

    ctx.enqueue_function(
        kernel,
        grid_dim=(grid[0], grid[1]),
        block_dim=block_size,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
