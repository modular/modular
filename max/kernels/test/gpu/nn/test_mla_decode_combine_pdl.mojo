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

"""Checks the SM100 MLA decode combine kernels under programmatic dependent launch.

The combine kernels release their dependents on entry, before their own
`griddepcontrol.wait`, long before they have written their output. That is
only correct because a dependent's `griddepcontrol.wait` blocks until the
whole prerequisite grid has completed and its memory is visible, not merely
until it has triggered. This test builds the chain that would expose a
violation, with no host synchronization between the launches:

  late-publish producer -> combine kernel -> combine kernel -> PDL consumer

- The producer releases its dependents on entry (as the MLA decode kernel
  does), sleeps, and only then publishes the split-K partials, so the combine
  kernel is resident and waiting long before its input exists.
- The combine kernel is the real one, launched through
  `mla_decode_combine_partial_outputs`.
- The consumer is launched with the PDL attribute, waits, and copies the
  combine output. The output buffer is poisoned with NaNs before every
  iteration, so a consumer that ran ahead of the combine kernel copies them.

Each iteration's consumer copy must equal, bit for bit, the combine output of
the same inputs run serialized with host synchronization between launches. The
consumer also timestamps its start; it must precede the end of the producer's
sleep, which proves the combine kernels released it early (removing the release
makes the test fail).

A second chain feeds the combine output into the batched matmul entry point
that `mla_decode_graph` uses for `W_UV`, launched under the PDL attribute:

- With 16 heads the matmul is a real batched one, laid out like the graph's
  (batch = heads, strided views over the `[rows, heads, dim]` buffers), so it
  reaches `dispatch_sm100_batched_matmul`, the production path.
- With one head the entry reroutes to the SM100 matmul dispatch with a
  runtime-extent weight view and more than one row, so the dispatch picks the
  structured matmul kernel (not the `gemv_split_k` path, which is for one row).

The matmul must wait on its predecessor too. Removing every
`wait_on_dependent_grids()` in `sm100_structured/default/matmul_kernels.mojo`
fails the 16-head chain; the one-head chain still passes then, so it checks the
reroute's results but is not evidence about that kernel's wait. The small-M
GEMV kernels are not covered.
"""

from std.math import ceildiv, nan
from std.memory import bitcast
from std.time import perf_counter_ns, sleep

from layout import Idx, TileTensor, row_major
from layout.tile_layout import Layout as TileLayout
from linalg.bmm import _batched_matmul_gpu
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_combine import (
    mla_decode_combine_partial_outputs,
)
from nn.attention.mha_utils import NullPointer

comptime dtype = DType.bfloat16
comptime accum_dtype = DType.float32
comptime HEAD_DIM = 512
comptime NUM_HEADS = 16
comptime ITERATIONS = 8
comptime PRODUCER_DELAY_NS = 500_000
# The consumer must have started at least this long before the producer's
# sleep ends, or the combine kernels did not release it early.
comptime EARLY_MARGIN_NS = 100_000
# Iterations 1 to ITERATIONS - 1 are checked; most of them must start early.
comptime MIN_EARLY_ITERATIONS = (ITERATIONS - 1) // 2 + 1


def _hash01(i: Int, seed: Int) -> Float32:
    var x = UInt32((i * 2654435761 + seed * 40503 + 12345) & 0x7FFFFFFF)
    x ^= x >> 13
    x *= 1274126177
    x ^= x >> 16
    return Float32(x & 0xFFFF) / 65536.0


def late_publish_producer(
    o_accum: MutPointer[Scalar[dtype], MutAnyOrigin],
    lse_accum: MutPointer[Scalar[accum_dtype], MutAnyOrigin],
    o_elems: Int32,
    lse_elems: Int32,
    seed: Int32,
    delay_ns: Int32,
    stamps: MutPointer[Scalar[DType.uint64], MutAnyOrigin],
    iteration: Int32,
):
    """Releases dependents immediately, then publishes its output late."""
    if block_idx.x == 0 and thread_idx.x == 0:
        stamps[2 * Int(iteration)] = UInt64(perf_counter_ns())
    launch_dependent_grids()
    sleep(Float64(delay_ns) * 1e-9)

    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    for i in range(tid, Int(o_elems), stride):
        o_accum[i] = Scalar[dtype](_hash01(i, Int(seed)) * 2.0 - 1.0)
    for i in range(tid, Int(lse_elems), stride):
        lse_accum[i] = _hash01(i, Int(seed) + 7) * 8.0 - 4.0


def pdl_consumer(
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    n: Int32,
    stamps: MutPointer[Scalar[DType.uint64], MutAnyOrigin],
    iteration: Int32,
):
    """Copies the combine output once the combine grid has completed."""
    if block_idx.x == 0 and thread_idx.x == 0:
        stamps[2 * Int(iteration) + 1] = UInt64(perf_counter_ns())
    wait_on_dependent_grids()

    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    for i in range(tid, Int(n), stride):
        dst[i] = src[i]


def _launch_producer(
    ctx: DeviceContext,
    o_accum_buf: DeviceBuffer[dtype],
    lse_buf: DeviceBuffer[accum_dtype],
    o_elems: Int,
    lse_elems: Int,
    seed: Int,
    delay_ns: Int,
    stamps: DeviceBuffer[DType.uint64],
) raises:
    ctx.enqueue_function[late_publish_producer](
        o_accum_buf.unsafe_ptr().as_unsafe_any_origin(),
        lse_buf.unsafe_ptr().as_unsafe_any_origin(),
        Int32(o_elems),
        Int32(lse_elems),
        Int32(seed),
        Int32(delay_ns),
        stamps.unsafe_ptr().as_unsafe_any_origin(),
        Int32(seed),
        # Few CTAs, so the combine grid can be resident alongside them: a
        # dependent grid is only released once all of its predecessor's CTAs
        # have started.
        grid_dim=16,
        block_dim=256,
    )


def _combine[
    num_splits: Int, ragged: Bool, warps_per_head: Int, split_parallel: Bool
](
    o_accum: TileTensor[dtype, address_space=.GENERIC, ...],
    lse_accum: TileTensor[accum_dtype, address_space=.GENERIC, ...],
    output: TileTensor[dtype, address_space=.GENERIC, ...],
    row_offsets: DeviceBuffer[DType.uint32],
    batch: Int,
    seq: Int,
    ctx: DeviceContext,
) raises:
    mla_decode_combine_partial_outputs[
        output_type=dtype,
        accum_type=accum_dtype,
        head_dim=HEAD_DIM,
        num_splits=num_splits,
        ragged=ragged,
        warps_per_head=warps_per_head,
        split_parallel=split_parallel,
    ](
        o_accum,
        lse_accum,
        output,
        row_offsets.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        NullPointer[DType.float32](),
        batch,
        seq,
        NUM_HEADS,
        ctx,
    )


def _check_consumer_started_early(
    stamps: DeviceBuffer[DType.uint64], ctx: DeviceContext
) raises:
    """The point of the release: the consumer must have started while the
    producer was still asleep, before the combine kernels could have finished.

    PDL only permits the early start, so a single late iteration (host launch
    delay, a busy GPU) is not a failure. Without the release the consumer starts
    only after the chain has drained, in none of the iterations, so requiring
    most of them to be early still catches its removal.
    """
    var stamps_host = List[UInt64](length=2 * ITERATIONS, fill=0)
    ctx.enqueue_copy(stamps_host, stamps)
    ctx.synchronize()
    # Iteration 0 is skipped: the first launch of each kernel pays a module
    # load on the host, which can outlast the producer's sleep.
    var early = 0
    for it in range(1, ITERATIONS):
        var producer_start = stamps_host[2 * it]
        var consumer_start = stamps_host[2 * it + 1]
        if consumer_start != 0 and consumer_start + UInt64(
            EARLY_MARGIN_NS
        ) <= producer_start + UInt64(PRODUCER_DELAY_NS):
            early += 1
    if early < MIN_EARLY_ITERATIONS:
        raise Error(
            "consumer started early in only ",
            early,
            " of ",
            ITERATIONS - 1,
            " iterations, need ",
            MIN_EARLY_ITERATIONS,
        )


def test_combine_pdl_chain[
    num_splits: Int, ragged: Bool, warps_per_head: Int, split_parallel: Bool
](batch: Int, seq: Int, ctx: DeviceContext) raises:
    print(
        "test_combine_pdl_chain splits=",
        num_splits,
        " ragged=",
        ragged,
        " wph=",
        warps_per_head,
        " split_parallel=",
        split_parallel,
        " batch=",
        batch,
        " seq=",
        seq,
    )

    # Ragged batches carry `seq - b % 2` tokens each, so the grid is padded.
    var offsets_host = List[UInt32](length=batch + 1, fill=0)
    for b in range(batch):
        var tokens = seq - (b % 2) if ragged else seq
        offsets_host[b + 1] = offsets_host[b] + UInt32(tokens)
    var total_rows = Int(offsets_host[batch]) if ragged else batch * seq

    var o_elems = num_splits * batch * seq * NUM_HEADS * HEAD_DIM
    var lse_elems = num_splits * batch * seq * NUM_HEADS
    var out_elems = total_rows * NUM_HEADS * HEAD_DIM

    var o_accum_buf = ctx.enqueue_create_buffer[dtype](o_elems)
    var lse_buf = ctx.enqueue_create_buffer[accum_dtype](lse_elems)
    var out_buf = ctx.enqueue_create_buffer[dtype](out_elems)
    var offsets_buf = ctx.enqueue_create_buffer[DType.uint32](batch + 1)
    var chain_dst = ctx.enqueue_create_buffer[dtype](ITERATIONS * out_elems)
    var stamps = ctx.enqueue_create_buffer[DType.uint64](2 * ITERATIONS)
    stamps.enqueue_fill(UInt64(0))
    ctx.enqueue_copy(offsets_buf, offsets_host)
    ctx.synchronize()

    var o_tt = TileTensor(
        o_accum_buf,
        row_major((num_splits, batch, seq, Idx[NUM_HEADS], Idx[HEAD_DIM])),
    )
    var lse_tt = TileTensor(
        lse_buf, row_major((num_splits, batch, seq, Idx[NUM_HEADS]))
    )
    var out_tt = TileTensor(
        out_buf, row_major((total_rows, Idx[NUM_HEADS], Idx[HEAD_DIM]))
    )

    var consumer_blocks = ceildiv(out_elems, 256)
    var poison = nan[dtype]()

    var ref_host = List[Scalar[dtype]](
        length=ITERATIONS * out_elems, fill=Scalar[dtype](0)
    )
    var ref_one = List[Scalar[dtype]](length=out_elems, fill=Scalar[dtype](0))

    # Serialized reference: host synchronization between every launch, so no
    # grid overlap is possible whatever the kernels do.
    for it in range(ITERATIONS):
        _launch_producer(
            ctx, o_accum_buf, lse_buf, o_elems, lse_elems, it, 0, stamps
        )
        ctx.synchronize()
        _combine[num_splits, ragged, warps_per_head, split_parallel](
            o_tt, lse_tt, out_tt, offsets_buf, batch, seq, ctx
        )
        ctx.synchronize()
        ctx.enqueue_copy(ref_one, out_buf)
        ctx.synchronize()
        for i in range(out_elems):
            ref_host[it * out_elems + i] = ref_one[i]

    # PDL chain: producer -> combine -> consumer back to back, one host sync at
    # the end. Poison first so a stale read is visible.
    for it in range(ITERATIONS):
        out_buf.enqueue_fill(poison)
        _launch_producer(
            ctx,
            o_accum_buf,
            lse_buf,
            o_elems,
            lse_elems,
            it,
            PRODUCER_DELAY_NS,
            stamps,
        )
        # Two combines back to back also cover the kernel chained with itself.
        for _ in range(2):
            _combine[num_splits, ragged, warps_per_head, split_parallel](
                o_tt, lse_tt, out_tt, offsets_buf, batch, seq, ctx
            )
        ctx.enqueue_function[pdl_consumer](
            out_buf.unsafe_ptr().as_unsafe_any_origin(),
            chain_dst.unsafe_ptr()
            .unsafe_offset(it * out_elems)
            .as_unsafe_any_origin(),
            Int32(out_elems),
            stamps.unsafe_ptr().as_unsafe_any_origin(),
            Int32(it),
            grid_dim=consumer_blocks,
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
    ctx.synchronize()

    var chain_host = List[Scalar[dtype]](
        length=ITERATIONS * out_elems, fill=Scalar[dtype](0)
    )
    ctx.enqueue_copy(chain_host, chain_dst)
    ctx.synchronize()

    var mismatches = 0
    var first_bad = -1
    for i in range(ITERATIONS * out_elems):
        var r = ref_host[i]
        if r != r:
            raise Error("serialized reference holds a NaN at ", i)
        if bitcast[DType.uint16, 1](chain_host[i]) != bitcast[DType.uint16, 1](
            r
        ):
            mismatches += 1
            if first_bad < 0:
                first_bad = i
    if mismatches != 0:
        raise Error(
            "PDL chain differs from the serialized run in ",
            mismatches,
            " of ",
            ITERATIONS * out_elems,
            " elements; first at ",
            first_bad,
            " (iteration ",
            first_bad // out_elems,
            ")",
        )

    _check_consumer_started_early(stamps, ctx)

    _ = o_accum_buf^
    _ = lse_buf^
    _ = out_buf^
    _ = offsets_buf^
    _ = chain_dst^
    _ = stamps^


comptime BMM_N = 128


def _combine_heads[
    num_splits: Int, heads: Int
](
    o_tt: TileTensor[dtype, address_space=.GENERIC, ...],
    lse_tt: TileTensor[accum_dtype, address_space=.GENERIC, ...],
    a_tt: TileTensor[dtype, address_space=.GENERIC, ...],
    offsets_buf: DeviceBuffer[DType.uint32],
    num_rows: Int,
    ctx: DeviceContext,
) raises:
    mla_decode_combine_partial_outputs[
        output_type=dtype,
        accum_type=accum_dtype,
        head_dim=HEAD_DIM,
        num_splits=num_splits,
        ragged=False,
        warps_per_head=2,
    ](
        o_tt,
        lse_tt,
        a_tt,
        offsets_buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        NullPointer[DType.float32](),
        num_rows,
        1,
        heads,
        ctx,
    )


def test_combine_to_bmm_pdl[
    num_splits: Int, heads: Int
](num_rows: Int, ctx: DeviceContext) raises:
    """Combine kernel followed by the `W_UV` batched matmul of the MLA graph.

    The batched matmul is launched with PDL, so the combine's early release
    lets it start before the combine output exists. It must wait.
    """
    print(
        "test_combine_to_bmm_pdl rows=",
        num_rows,
        " splits=",
        num_splits,
        " heads=",
        heads,
    )

    var o_elems = num_splits * num_rows * heads * HEAD_DIM
    var lse_elems = num_splits * num_rows * heads
    var a_elems = num_rows * heads * HEAD_DIM
    var c_elems = num_rows * heads * BMM_N

    var o_accum_buf = ctx.enqueue_create_buffer[dtype](o_elems)
    var lse_buf = ctx.enqueue_create_buffer[accum_dtype](lse_elems)
    var a_buf = ctx.enqueue_create_buffer[dtype](a_elems)
    var w_buf = ctx.enqueue_create_buffer[dtype](heads * BMM_N * HEAD_DIM)
    var c_buf = ctx.enqueue_create_buffer[dtype](c_elems)
    var offsets_buf = ctx.enqueue_create_buffer[DType.uint32](num_rows + 1)
    var chain_dst = ctx.enqueue_create_buffer[dtype](ITERATIONS * c_elems)
    # Only the kernels' signatures need these. The early-start assertion is on
    # the combine kernels' own consumer above; the matmul in the middle here
    # was not observed to release its dependents early, so a timestamp at its
    # consumer would say nothing about the combine kernels.
    var stamps = ctx.enqueue_create_buffer[DType.uint64](2 * ITERATIONS)
    var w_host = List[Scalar[dtype]](
        length=heads * BMM_N * HEAD_DIM, fill=Scalar[dtype](0)
    )
    for i in range(heads * BMM_N * HEAD_DIM):
        w_host[i] = Scalar[dtype](_hash01(i, 99) - 0.5)
    ctx.enqueue_copy(w_buf, w_host)
    ctx.synchronize()

    var o_tt = TileTensor(
        o_accum_buf,
        row_major((num_splits, num_rows, 1, Idx[heads], Idx[HEAD_DIM])),
    )
    var lse_tt = TileTensor(
        lse_buf, row_major((num_splits, num_rows, 1, Idx[heads]))
    )
    var a_tt = TileTensor(
        a_buf, row_major((num_rows, Idx[heads], Idx[HEAD_DIM]))
    )
    # The graph's `W_UV` views: batch = heads, strided over the buffers.
    var a_3d = a_tt.reshape(
        TileLayout(
            (Idx[heads], num_rows, Idx[HEAD_DIM]),
            (Idx[HEAD_DIM], Idx[heads * HEAD_DIM], Idx[1]),
        )
    )
    var w_3d = TileTensor(
        w_buf, row_major((Idx[heads], Idx[BMM_N], Idx[HEAD_DIM]))
    )
    var c_3d = TileTensor(
        c_buf, row_major((num_rows, Idx[heads], Idx[BMM_N]))
    ).reshape(
        TileLayout(
            (Idx[heads], num_rows, Idx[BMM_N]),
            (Idx[BMM_N], Idx[heads * BMM_N], Idx[1]),
        )
    )

    var ref_host = List[Scalar[dtype]](
        length=ITERATIONS * c_elems, fill=Scalar[dtype](0)
    )
    var ref_one = List[Scalar[dtype]](length=c_elems, fill=Scalar[dtype](0))
    for it in range(ITERATIONS):
        _launch_producer(
            ctx, o_accum_buf, lse_buf, o_elems, lse_elems, it, 0, stamps
        )
        ctx.synchronize()
        _combine_heads[num_splits, heads](
            o_tt, lse_tt, a_tt, offsets_buf, num_rows, ctx
        )
        ctx.synchronize()
        _batched_matmul_gpu[
            transpose_b=True, pdl_level=PDLLevel.OVERLAP_AT_BEGINNING
        ](c_3d, a_3d, w_3d, ctx)
        ctx.synchronize()
        ctx.enqueue_copy(ref_one, c_buf)
        ctx.synchronize()
        for i in range(c_elems):
            ref_host[it * c_elems + i] = ref_one[i]

    for it in range(ITERATIONS):
        a_buf.enqueue_fill(nan[dtype]())
        _launch_producer(
            ctx,
            o_accum_buf,
            lse_buf,
            o_elems,
            lse_elems,
            it,
            PRODUCER_DELAY_NS,
            stamps,
        )
        _combine_heads[num_splits, heads](
            o_tt, lse_tt, a_tt, offsets_buf, num_rows, ctx
        )
        _batched_matmul_gpu[
            transpose_b=True, pdl_level=PDLLevel.OVERLAP_AT_BEGINNING
        ](c_3d, a_3d, w_3d, ctx)
        ctx.enqueue_function[pdl_consumer](
            c_buf.unsafe_ptr().as_unsafe_any_origin(),
            chain_dst.unsafe_ptr()
            .unsafe_offset(it * c_elems)
            .as_unsafe_any_origin(),
            Int32(c_elems),
            stamps.unsafe_ptr().as_unsafe_any_origin(),
            Int32(it),
            grid_dim=ceildiv(c_elems, 256),
            block_dim=256,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
    ctx.synchronize()

    var chain_host = List[Scalar[dtype]](
        length=ITERATIONS * c_elems, fill=Scalar[dtype](0)
    )
    ctx.enqueue_copy(chain_host, chain_dst)
    ctx.synchronize()

    var mismatches = 0
    var first_bad = -1
    for i in range(ITERATIONS * c_elems):
        var r = ref_host[i]
        if r != r:
            raise Error("serialized reference holds a NaN at ", i)
        if bitcast[DType.uint16, 1](chain_host[i]) != bitcast[DType.uint16, 1](
            r
        ):
            mismatches += 1
            if first_bad < 0:
                first_bad = i
    if mismatches != 0:
        raise Error(
            (
                "combine -> batched matmul PDL chain differs from the"
                " serialized run in "
            ),
            mismatches,
            " of ",
            ITERATIONS * c_elems,
            " elements; first at ",
            first_bad,
            " (iteration ",
            first_bad // c_elems,
            ")",
        )

    _ = o_accum_buf^
    _ = lse_buf^
    _ = a_buf^
    _ = w_buf^
    _ = c_buf^
    _ = offsets_buf^
    _ = chain_dst^
    _ = stamps^


def main() raises:
    with DeviceContext() as ctx:
        # Main combine kernel: dense and ragged, both warp splits.
        test_combine_pdl_chain[4, False, 2, False](3, 2, ctx)
        test_combine_pdl_chain[7, True, 2, False](5, 3, ctx)
        test_combine_pdl_chain[16, False, 1, False](2, 1, ctx)
        # Split-parallel combine kernel.
        test_combine_pdl_chain[8, False, 1, True](3, 2, ctx)
        test_combine_pdl_chain[16, True, 1, True](5, 3, ctx)
        # Combine output consumed by a batch-1 W_UV matmul (structured kernel).
        test_combine_to_bmm_pdl[4, 1](3, ctx)
        test_combine_to_bmm_pdl[8, 1](9, ctx)
        # ... and by the production batched `W_UV` matmul (batch = heads).
        test_combine_to_bmm_pdl[4, NUM_HEADS](3, ctx)
        test_combine_to_bmm_pdl[8, NUM_HEADS](9, ctx)
