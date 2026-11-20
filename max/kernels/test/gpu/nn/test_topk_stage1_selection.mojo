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

"""The two-stage top-k selects the sorted top-k of every row, ties included.

Stage 1 retires a selected element instead of re-reading it. It used to do that
by writing the dead value over the element in a private copy of the row, which
the caller paid `2 * batch * N` bytes to allocate and copy; it now retires with
per-thread bookkeeping -- the value of the thread's last emission and a count
of how many of its elements at that value it has emitted -- and reads the
caller's row in place.

That bookkeeping replaces a rule the tie order depends on, so the whole
`[batch, k]` selection is compared against `_top_k_cpu`'s sorted reference
-- descending value, lowest index first -- rather than against a value
tolerance: a different element on a tie is a different token. There is one
sorted top-k per row, so a passing comparison pins the values, the indices, the
tie order and the order of the k outputs at once.

Row lengths stay at or below 2048 wherever the row is dense with duplicates,
because only there is the two-stage block partition contiguous and the tie
order the same as a sort's. Above that the partition interleaves blocks, ties
come out block-major, and only a row of distinct values (which fixes the
selection regardless of partition) is comparable -- `distinct` rows cover the
production geometry at 128k and row lengths that leave each thread a short
final pass.

The element-to-thread mapping is part of that tie order, not an implementation
detail: `check_tie_after_eviction` pins the one case where a thread holds both
a tied value and a better one and the heap has to evict its lowest tied index
before the block emits it.

Every case also checks the row the kernel read is unchanged, which is what
makes the copy unnecessary rather than merely redundant.
"""

from std.math import ceildiv

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.topk import _top_k_cpu, topk_gpu
from std.testing import assert_equal

comptime IDX = DType.int64


def _shuffled(n: Int, seed: UInt64) -> List[Int]:
    """A permutation of `0..n`, from a fixed linear congruential generator."""
    var ids = List[Int](capacity=n)
    for i in range(n):
        ids.append(i)
    var state = seed
    for i in range(n - 1, 0, -1):
        state = state * 6364136223846793005 + 1442695040888963407
        var j = Int((state >> 33) % UInt64(i + 1))
        var swap = ids[i]
        ids[i] = ids[j]
        ids[j] = swap
    return ids^


def fill_row[
    dtype: DType
](
    ptr: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    batch: Int,
    n: Int,
    fill: String,
):
    """Writes the row pattern named by `fill` into a `[batch, n]` buffer.

    Every value written is an integer, so it is exact in both dtypes. `flat`
    and `runs` put far more duplicates in one thread's slice than the register
    heap holds, which forces the re-scan path; `tied` scatters ties across
    threads and blocks; `distinct` permutes the row so the selection is unique
    but nothing about its index pattern is regular.
    """
    for b in range(batch):
        if fill == "distinct":
            var ids = _shuffled(n, UInt64(b) * 0x9E3779B97F4A7C15 + 1)
            for i in range(n):
                ptr[b * n + i] = Scalar[dtype](Float64(ids[i]))
            _ = ids^
            continue
        for i in range(n):
            var v: Int
            if fill == "flat":
                v = 0
            elif fill == "runs":
                v = (i // 37 + b) % 4
            else:
                v = (i * 7 + b * 3) % 6
            ptr[b * n + i] = Scalar[dtype](Float64(v))


def check_selection[
    dtype: DType, largest: Bool = True
](
    ctx: DeviceContext,
    batch: Int,
    n: Int,
    ks: Span[Int, ImmutAnyOrigin],
    fill: String,
    num_blocks_per_input: Optional[Int] = None,
    block_size: Optional[Int] = None,
) raises:
    """Runs every k in `ks` on one `[batch, n]` row set and checks the lot.

    One sorted reference per row serves every k: the top-k of a row is the
    first k of its sorted order, so comparing each run against that prefix also
    checks that a larger k does not move the smaller ones.

    Args:
        ctx: The device to run on.
        batch: Number of rows.
        n: Row length.
        ks: The k values to run, ascending.
        fill: The row pattern, see `fill_row`.
        num_blocks_per_input: Stage-1 partitions per row. Defaults to what
            `topk_gpu` picks for a row of this length, and is pinned to 1 (one
            contiguous block) to give each thread more elements than its
            register heap holds.
        block_size: Threads per block. Defaults to what `topk_gpu` picks for
            this input size; halving it doubles the row elements per thread.
    """
    var max_k = ks[len(ks) - 1]
    var label = String(
        "batch=",
        batch,
        " N=",
        n,
        " fill=",
        fill,
        " partitions=",
        num_blocks_per_input,
        " block=",
        block_size,
        " largest=",
        largest,
    )

    var in_host = ctx.enqueue_create_host_buffer[dtype](batch * n)
    fill_row[dtype](in_host.unsafe_ptr(), batch, n, fill)

    var in_dev = ctx.enqueue_create_buffer[dtype](batch * n)
    ctx.enqueue_copy(in_dev, in_host)

    var ref_vals = ctx.enqueue_create_host_buffer[dtype](batch * max_k)
    var ref_idxs = ctx.enqueue_create_host_buffer[IDX](batch * max_k)
    _top_k_cpu[dtype=dtype, out_idx_type=IDX, largest=largest](
        TileTensor(in_host, row_major(batch, n)),
        max_k,
        1,
        TileTensor(ref_vals, row_major(batch, max_k)),
        TileTensor(ref_idxs, row_major(batch, max_k)),
        1,
        True,
    )

    var partitions = num_blocks_per_input.or_else(min(ceildiv(n, 256), 8))
    var out_vals = ctx.enqueue_create_buffer[dtype](batch * max_k)
    var out_idxs = ctx.enqueue_create_buffer[IDX](batch * max_k)
    var in_t = (
        TileTensor(in_dev, row_major(batch, n)).as_unsafe_any_origin().as_imm()
    )

    for k in ks:
        topk_gpu[sampling=False, largest=largest](
            ctx,
            k,
            in_t,
            TileTensor(out_vals, row_major(batch, k)),
            TileTensor(out_idxs, row_major(batch, k)),
            block_size=block_size,
            num_blocks_per_input=partitions,
        )
        ctx.synchronize()

        with out_vals.map_to_host() as got_vals:
            with out_idxs.map_to_host() as got_idxs:
                for b in range(batch):
                    for j in range(k):
                        var at = b * k + j
                        var want = b * max_k + j
                        var msg = String(
                            " k=", k, " row=", b, " slot=", j, " ", label
                        )
                        assert_equal(got_vals[at], ref_vals[want], msg)
                        assert_equal(
                            Int(got_idxs[at]),
                            Int(ref_idxs[want]),
                            String("index", msg),
                        )

    var in_after = ctx.enqueue_create_host_buffer[dtype](batch * n)
    ctx.enqueue_copy(in_after, in_dev)
    ctx.synchronize()
    for i in range(batch * n):
        assert_equal(
            in_after[i], in_host[i], String("row modified at ", i, " ", label)
        )

    _ = in_host^
    _ = in_dev^
    _ = ref_vals^
    _ = ref_idxs^
    _ = out_vals^
    _ = out_idxs^
    _ = in_after^


def check_tie_after_eviction(ctx: DeviceContext) raises:
    """A tie index the heap evicts must still come out in index order.

    Thread 0 owns indices 0, 1024 and 2048 of this row and holds a 1.0 at the
    first two and the only 2.0 at the third; the 1s at 0-3 and 1024-1027 give
    the row eight tied entries in total. Filling the heap takes eight of them,
    and retiring the 2.0 hands its slot to whatever the threshold eviction picks
    -- index 0 here -- so the second element out is index 0, not index 1.

    Any change to which thread owns which index moves that choice: grouping a
    thread's slice into vectors puts all eight 1s in thread 0, the 2.0 evicts
    index 0, and a re-scan only finds it after index 1, so the emitted token
    changes. One partition means stage 2 forwards stage 1's order unchanged.
    """
    comptime dtype = DType.float32
    comptime n = 3072
    comptime k = 2
    comptime block_size = 256

    var in_host = ctx.enqueue_create_host_buffer[dtype](n)
    for i in range(n):
        in_host[i] = Scalar[dtype](0.0)
    for i in range(4):
        in_host[i] = Scalar[dtype](1.0)
        in_host[1024 + i] = Scalar[dtype](1.0)
    in_host[2048] = Scalar[dtype](2.0)

    var in_dev = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(in_dev, in_host)
    var out_vals = ctx.enqueue_create_buffer[dtype](k)
    var out_idxs = ctx.enqueue_create_buffer[IDX](k)

    topk_gpu[sampling=False, largest=True](
        ctx,
        k,
        TileTensor(in_dev, row_major(1, n)).as_unsafe_any_origin().as_imm(),
        TileTensor(out_vals, row_major(1, k)),
        TileTensor(out_idxs, row_major(1, k)),
        block_size=block_size,
        num_blocks_per_input=1,
    )
    ctx.synchronize()

    with out_vals.map_to_host() as vals:
        with out_idxs.map_to_host() as idxs:
            assert_equal(vals[0], Scalar[dtype](2.0))
            assert_equal(vals[1], Scalar[dtype](1.0))
            assert_equal(Int(idxs[0]), 2048)
            assert_equal(Int(idxs[1]), 0)

    var in_after = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.enqueue_copy(in_after, in_dev)
    ctx.synchronize()
    for i in range(n):
        assert_equal(in_after[i], in_host[i], String("row modified at ", i))

    _ = in_host^
    _ = in_dev^
    _ = out_vals^
    _ = out_idxs^
    _ = in_after^


def check_spike_and_ties[
    dtype: DType
](
    ctx: DeviceContext,
    n: Int,
    block_size: Int,
    partitions: Int,
    spike: Int,
    expect_second: Int,
) raises:
    """One dominant token on an otherwise all-tied row.

    Every other element is 0.0, so the second element out is decided by ties
    alone: the paying zero is whichever index the block partition puts first.
    The spike is what makes stage 1 evict while the row is still all ties --
    the periodic fills never produce that, and it is exactly the state where a
    heap that has dropped a low tied index emits a higher one first.
    """
    comptime k = 2
    var in_host = ctx.enqueue_create_host_buffer[dtype](n)
    for i in range(n):
        in_host[i] = Scalar[dtype](0.0)
    in_host[spike] = Scalar[dtype](1.0)

    var in_dev = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(in_dev, in_host)
    var out_vals = ctx.enqueue_create_buffer[dtype](k)
    var out_idxs = ctx.enqueue_create_buffer[IDX](k)

    topk_gpu[sampling=False, largest=True](
        ctx,
        k,
        TileTensor(in_dev, row_major(1, n)).as_unsafe_any_origin().as_imm(),
        TileTensor(out_vals, row_major(1, k)),
        TileTensor(out_idxs, row_major(1, k)),
        block_size=block_size,
        num_blocks_per_input=partitions,
    )
    ctx.synchronize()

    var label = String(" N=", n, " block=", block_size, " parts=", partitions)
    with out_vals.map_to_host() as vals:
        with out_idxs.map_to_host() as idxs:
            assert_equal(vals[0], Scalar[dtype](1.0), label)
            assert_equal(Int(idxs[0]), spike, label)
            assert_equal(vals[1], Scalar[dtype](0.0), label)
            assert_equal(Int(idxs[1]), expect_second, String("second", label))

    var in_after = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.enqueue_copy(in_after, in_dev)
    ctx.synchronize()
    for i in range(n):
        assert_equal(in_after[i], in_host[i], String("row modified at ", i))

    _ = in_host^
    _ = in_dev^
    _ = out_vals^
    _ = out_idxs^
    _ = in_after^


def check_spike_shapes(ctx: DeviceContext) raises:
    """`check_spike_and_ties` at a contiguous and at the production partition.
    """
    # One block over the row: the partition is contiguous, so the paying zero
    # is the row's lowest index, whichever thread holds it.
    check_spike_and_ties[.float32](ctx, 2048, 128, 1, 1026, 0)
    check_spike_and_ties[.bfloat16](ctx, 2048, 128, 1, 1026, 0)
    # Production shape: interleaved blocks, tie order block-major.
    check_spike_and_ties[.float32](ctx, 128256, 256, 8, 20480, 1)
    check_spike_and_ties[.bfloat16](ctx, 128256, 256, 8, 20480, 1)


def check_dense_ties(ctx: DeviceContext) raises:
    """Rows that are mostly duplicates, where the tie order is the whole test.
    """
    var ks = [1, 5, 8, 9, 50, 75, 255]
    # One element per thread, so every pass is a block reduction over ties.
    check_selection[.float32](ctx, 1, 1024, ks, "flat")
    check_selection[.float32](ctx, 3, 2048, ks, "tied")
    check_selection[.float32](ctx, 3, 1024, ks, "runs")
    # Row lengths the stride does not divide, so each thread's final pass is
    # short -- a length one past a stride multiple, and an odd one.
    check_selection[.float32](ctx, 2, 2000, ks, "runs")
    check_selection[.bfloat16](ctx, 3, 2048, ks, "tied")
    check_selection[.bfloat16](ctx, 2, 1024, ks, "flat")
    check_selection[.bfloat16](ctx, 2, 2000, ks, "runs")


def check_one_block_per_row(ctx: DeviceContext) raises:
    """One contiguous block per row: more elements per thread than the heap.

    A thread holds eight elements of the row here, so its register heap fills
    exactly and everything past k=8 comes from a re-scan that has to retire the
    elements it already emitted without touching the row.
    """
    var ks = [1, 5, 8, 9, 50, 75, 255]
    check_selection[.float32](ctx, 2, 2048, ks, "flat", num_blocks_per_input=1)
    check_selection[.float32](ctx, 2, 2048, ks, "tied", num_blocks_per_input=1)
    check_selection[.bfloat16](ctx, 2, 2048, ks, "runs", num_blocks_per_input=1)
    # Half the threads, so each thread holds twice the heap and has to evict:
    # the ties it evicts are the ones a re-scan must not hand out twice.
    check_selection[.float32](
        ctx, 2, 2048, ks, "tied", num_blocks_per_input=1, block_size=128
    )
    check_selection[.bfloat16](
        ctx, 2, 2048, ks, "runs", num_blocks_per_input=1, block_size=128
    )


def check_distinct_rows(ctx: DeviceContext) raises:
    """Rows with no ties, at the production partition and past 2048 elements.

    With distinct values the selection is unique, so these are comparable at
    any row length: they cover the interleaved block partition, the 128k row
    the serving path uses, and a row length with a partial last window.
    """
    var ks = [1, 5, 50, 75, 255]
    check_selection[.float32](ctx, 8, 2048, ks, "distinct")
    check_selection[.float32](ctx, 2, 128256, ks, "distinct")
    check_selection[.float32](ctx, 1, 128255, ks, "distinct")


def check_bottom_k(ctx: DeviceContext) raises:
    """`largest=False` flips every comparison in the retirement bookkeeping."""
    var ks = [1, 5, 8, 9, 50, 255]
    check_selection[.float32, False](ctx, 3, 2048, ks, "tied")
    check_selection[.float32, False](ctx, 2, 1024, ks, "runs")
    check_selection[.float32, False](
        ctx, 2, 2048, ks, "flat", num_blocks_per_input=1
    )


def main() raises:
    with DeviceContext() as ctx:
        check_tie_after_eviction(ctx)
        check_spike_shapes(ctx)
        check_dense_ties(ctx)
        check_one_block_per_row(ctx)
        check_distinct_rows(ctx)
        check_bottom_k(ctx)
        print("PASS")
