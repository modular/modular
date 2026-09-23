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

"""A row with fewer live candidates than `max_k` must come back padded.

`_topk_stage2` stops reducing as soon as the best remaining candidate is the
dead sentinel. A caller reads all `max_k` slots and tells the padding apart by
that sentinel, so every slot past the last real pick has to be written --
otherwise the tail is whatever the output buffer held before the launch, which
reads back as real picks with arbitrary indices. DeepSeek-V4's lightning
indexer turned such a tail into out-of-range compressed-cache entry ids and
faulted with an illegal address.

Only the multi-block path is at risk: with `num_blocks_per_input == 1` stage 2
copies a stage-1 row that is already fully written. `N == BLOCK_SIZE *
NUM_BLOCKS` puts `num_blocks_per_input` at 4 under the `ceildiv(N, block_size)`
default as well as under the explicit argument the checks pass, so stage 2
takes the reduce path either way.

Both checks poison the output buffers before the launch, so an unwritten tail
fails deterministically instead of depending on what the allocator recycled.
"""

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.topk import topk_gpu
from std.testing import assert_equal
from std.utils.numerics import min_or_neg_inf

comptime DTYPE = DType.float32
comptime IDX = DType.int64
comptime BLOCK_SIZE = 256
comptime NUM_BLOCKS = 4
comptime N = BLOCK_SIZE * NUM_BLOCKS
comptime MAX_K = 8
comptime POISON_VAL = 1234.5
comptime POISON_IDX = 4242


def check_short_row_tail_is_padded(ctx: DeviceContext, batch_size: Int) raises:
    """Three live candidates against `max_k == 8`, so slots 3.. are padding."""
    comptime NUM_LIVE = 3
    var dead = min_or_neg_inf[DTYPE]()

    # Spread so the live values cannot all come from one stage-1 block.
    var live_idxs: List[Int] = [900, 5, 300]
    var live_vals: List[Float32] = [3.0, 2.0, 1.0]

    var in_buf = ctx.enqueue_create_buffer[DTYPE](batch_size * N)
    var out_vals = ctx.enqueue_create_buffer[DTYPE](batch_size * MAX_K)
    var out_idxs = ctx.enqueue_create_buffer[IDX](batch_size * MAX_K)

    var in_t = TileTensor(in_buf, row_major(Coord(batch_size, N)))
    with in_buf.map_to_host() as h:
        var t = TileTensor(h, row_major(Coord(batch_size, N)))
        for b in range(batch_size):
            for i in range(N):
                t[b, i] = dead
            for j in range(NUM_LIVE):
                t[b, live_idxs[j]] = live_vals[j]
    with out_vals.map_to_host() as h:
        for i in range(batch_size * MAX_K):
            h[i] = Scalar[DTYPE](POISON_VAL)
    with out_idxs.map_to_host() as h:
        for i in range(batch_size * MAX_K):
            h[i] = Scalar[IDX](POISON_IDX)

    topk_gpu[sampling=False, largest=True](
        ctx,
        MAX_K,
        in_t.as_unsafe_any_origin().as_imm(),
        TileTensor(out_vals, row_major(Coord(batch_size, MAX_K))),
        TileTensor(out_idxs, row_major(Coord(batch_size, MAX_K))),
        block_size=BLOCK_SIZE,
        num_blocks_per_input=NUM_BLOCKS,
    )
    ctx.synchronize()

    with out_vals.map_to_host() as h:
        for b in range(batch_size):
            for j in range(NUM_LIVE):
                assert_equal(h[b * MAX_K + j], live_vals[j])
            for j in range(NUM_LIVE, MAX_K):
                assert_equal(h[b * MAX_K + j], dead)

    with out_idxs.map_to_host() as h:
        for b in range(batch_size):
            for j in range(NUM_LIVE):
                assert_equal(Int(h[b * MAX_K + j]), live_idxs[j])
            # Slot NUM_LIVE is the iteration that found the row exhausted. It
            # carries the sentinel value, but its index is a stage-1 leftover
            # rather than -1, so only its range is guaranteed. Everything
            # after it is pure padding.
            for j in range(NUM_LIVE + 1, MAX_K):
                assert_equal(Int(h[b * MAX_K + j]), -1)
            for j in range(MAX_K):
                var idx = Int(h[b * MAX_K + j])
                if idx < -1 or idx >= N:
                    raise Error(
                        "top-k slot "
                        + String(j)
                        + " has out-of-range index "
                        + String(idx)
                    )

    _ = in_buf^
    _ = out_vals^
    _ = out_idxs^


def check_full_row_keeps_every_slot(ctx: DeviceContext) raises:
    """Exactly `max_k` live candidates, so the padding must never run."""
    var dead = min_or_neg_inf[DTYPE]()

    # Ranked out of index order so the merge, not the layout, sets the output
    # order.
    var live_idxs: List[Int] = [10, 20, 260, 300, 520, 600, 800, 1000]
    var live_vals: List[Float32] = [5.0, 8.0, 1.0, 6.0, 3.0, 7.0, 2.0, 4.0]
    var want_idxs: List[Int] = [20, 600, 300, 10, 1000, 520, 800, 260]
    var want_vals: List[Float32] = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    var in_buf = ctx.enqueue_create_buffer[DTYPE](N)
    var out_vals = ctx.enqueue_create_buffer[DTYPE](MAX_K)
    var out_idxs = ctx.enqueue_create_buffer[IDX](MAX_K)

    var in_t = TileTensor(in_buf, row_major(Coord(1, N)))
    with in_buf.map_to_host() as h:
        var t = TileTensor(h, row_major(Coord(1, N)))
        for i in range(N):
            t[0, i] = dead
        for j in range(MAX_K):
            t[0, live_idxs[j]] = live_vals[j]
    with out_vals.map_to_host() as h:
        for i in range(MAX_K):
            h[i] = Scalar[DTYPE](POISON_VAL)
    with out_idxs.map_to_host() as h:
        for i in range(MAX_K):
            h[i] = Scalar[IDX](POISON_IDX)

    topk_gpu[sampling=False, largest=True](
        ctx,
        MAX_K,
        in_t.as_unsafe_any_origin().as_imm(),
        TileTensor(out_vals, row_major(Coord(1, MAX_K))),
        TileTensor(out_idxs, row_major(Coord(1, MAX_K))),
        block_size=BLOCK_SIZE,
        num_blocks_per_input=NUM_BLOCKS,
    )
    ctx.synchronize()

    with out_vals.map_to_host() as h:
        for j in range(MAX_K):
            assert_equal(h[j], want_vals[j])

    with out_idxs.map_to_host() as h:
        for j in range(MAX_K):
            assert_equal(Int(h[j]), want_idxs[j])

    _ = in_buf^
    _ = out_vals^
    _ = out_idxs^


def main() raises:
    with DeviceContext() as ctx:
        check_short_row_tail_is_padded(ctx, 1)
        check_short_row_tail_is_padded(ctx, 4)
        check_full_row_keeps_every_slot(ctx)
        print("test_topk_stage2_tail_fill: OK")
