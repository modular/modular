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

"""Checks the SM100 FA4 workspace split-K combine kernel under programmatic
dependent launch.

The combine kernel releases its dependents right after launch, long before it
has written its output. That is only correct because a dependent's
`griddepcontrol.wait` blocks until the whole prerequisite grid has completed
and its memory is visible, not merely until it has triggered. The same
property is what would make a prologue release by the attention kernel safe.
This test builds the chain that would expose a violation, with no host
synchronization between the launches:

  late-publish producer -> `fa4_splitk_combine` -> `fa4_splitk_combine`
  -> PDL consumer

- The producer stands in for an attention kernel that releases its dependents
  on entry: it sleeps, and only then publishes `o_partial` / `lse_partial`, so
  the combine kernel is resident and waiting long before its input exists.
- The combine kernel is the real one, run twice back to back so it is also
  chained with itself.
- The consumer is launched with the PDL attribute, waits, and copies the
  combine output. The output buffer is poisoned with NaNs before every
  iteration, so a consumer that ran ahead of the combine kernel copies them.

Each iteration's consumer copy must equal, bit for bit, the combine output of
the same inputs run serialized with host synchronization between launches. The
consumer also timestamps its start; it must precede the end of the producer's
sleep, which proves the combine kernel released it early (removing the release
makes the test fail).

A second chain feeds the combine output into the real SM100 matmul dispatch at
a shape its tuned small-M table routes to `gemv_split_k`. That matmul is
launched with the PDL attribute, so it must wait on its own predecessor.
"""

from std.math import ceildiv, nan
from std.memory import bitcast
from std.time import perf_counter_ns, sleep

from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from nn.attention.gpu.nvidia.sm100.attention import MHA_PDL_LEVEL
from nn.attention.gpu.nvidia.sm100.fa4_splitk_combine import fa4_splitk_combine

comptime dtype = DType.bfloat16
comptime ITERATIONS = 8
comptime PRODUCER_DELAY_NS = 500_000
# The consumer must have started at least this long before the producer's
# sleep ends, or the combine kernel did not release it early.
comptime EARLY_MARGIN_NS = 100_000
# Iterations 1 to ITERATIONS - 1 are checked; most of them must start early.
comptime MIN_EARLY_ITERATIONS = (ITERATIONS - 1) // 2 + 1


def _hash01(i: Int, seed: Int) -> Float32:
    var x = UInt32((i * 2654435761 + seed * 40503 + 12345) & 0x7FFFFFFF)
    x ^= x >> 13
    x *= 1274126177
    x ^= x >> 16
    return Float32(x & 0xFFFF) / 65536.0


def late_publish_producer[
    intermediate_type: DType
](
    o_partial: MutPointer[Scalar[intermediate_type], MutAnyOrigin],
    lse_partial: MutPointer[Scalar[DType.float32], MutAnyOrigin],
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
        o_partial[i] = Scalar[intermediate_type](
            _hash01(i, Int(seed)) * 2.0 - 1.0
        )
    for i in range(tid, Int(lse_elems), stride):
        lse_partial[i] = _hash01(i, Int(seed) + 7) * 8.0 - 4.0


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


def _launch_producer[
    intermediate_type: DType
](
    ctx: DeviceContext,
    o_partial_buf: DeviceBuffer[intermediate_type],
    lse_buf: DeviceBuffer[DType.float32],
    o_elems: Int,
    lse_elems: Int,
    seed: Int,
    iteration: Int,
    delay_ns: Int,
    stamps: DeviceBuffer[DType.uint64],
) raises:
    ctx.enqueue_function[late_publish_producer[intermediate_type]](
        o_partial_buf.unsafe_ptr().as_unsafe_any_origin(),
        lse_buf.unsafe_ptr().as_unsafe_any_origin(),
        Int32(o_elems),
        Int32(lse_elems),
        Int32(seed),
        Int32(delay_ns),
        stamps.unsafe_ptr().as_unsafe_any_origin(),
        Int32(iteration),
        # Few CTAs, so the combine grid can be resident alongside them: a
        # dependent grid is only released once all of its predecessor's CTAs
        # have started.
        grid_dim=16,
        block_dim=256,
    )


def _combine[
    ov_depth: Int, intermediate_type: DType
](
    ctx: DeviceContext,
    o_partial_buf: DeviceBuffer[intermediate_type],
    lse_buf: DeviceBuffer[DType.float32],
    out_buf: DeviceBuffer[dtype],
    num_partitions: Int,
    num_rows_q: Int,
    num_q_heads: Int,
) raises:
    fa4_splitk_combine[dtype, ov_depth, intermediate_type](
        ctx,
        out_buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        o_partial_buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .as_unsafe_any_origin(),
        lse_buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin(),
        UInt32(num_partitions),
        UInt32(num_rows_q),
        UInt32(num_q_heads),
    )


def test_combine_pdl_chain[
    ov_depth: Int, intermediate_type: DType
](
    num_partitions: Int, num_rows_q: Int, num_q_heads: Int, ctx: DeviceContext
) raises:
    print(
        "test_combine_pdl_chain ov_depth=",
        ov_depth,
        " intermediate=",
        intermediate_type,
        " P=",
        num_partitions,
        " rows=",
        num_rows_q,
        " heads=",
        num_q_heads,
    )

    var rows_heads = num_rows_q * num_q_heads
    var o_elems = num_partitions * rows_heads * ov_depth
    var lse_elems = num_partitions * rows_heads
    var out_elems = rows_heads * ov_depth

    var o_partial_buf = ctx.enqueue_create_buffer[intermediate_type](o_elems)
    var lse_buf = ctx.enqueue_create_buffer[DType.float32](lse_elems)
    var out_buf = ctx.enqueue_create_buffer[dtype](out_elems)
    var chain_dst = ctx.enqueue_create_buffer[dtype](ITERATIONS * out_elems)
    var stamps = ctx.enqueue_create_buffer[DType.uint64](2 * ITERATIONS)
    stamps.enqueue_fill(UInt64(0))
    ctx.synchronize()

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
            ctx, o_partial_buf, lse_buf, o_elems, lse_elems, it, it, 0, stamps
        )
        ctx.synchronize()
        _combine[ov_depth, intermediate_type](
            ctx,
            o_partial_buf,
            lse_buf,
            out_buf,
            num_partitions,
            num_rows_q,
            num_q_heads,
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
            o_partial_buf,
            lse_buf,
            o_elems,
            lse_elems,
            it,
            it,
            PRODUCER_DELAY_NS,
            stamps,
        )
        for _ in range(2):
            _combine[ov_depth, intermediate_type](
                ctx,
                o_partial_buf,
                lse_buf,
                out_buf,
                num_partitions,
                num_rows_q,
                num_q_heads,
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

    # The point of the release: the consumer must have started while the
    # producer was still asleep, before the combine kernel could have finished.
    # PDL only permits the early start, so a single late iteration (host launch
    # delay, a busy GPU) is not a failure; without the release none is early.
    var stamps_host = List[UInt64](length=2 * ITERATIONS, fill=0)
    ctx.enqueue_copy(stamps_host, stamps)
    ctx.synchronize()
    # Iteration 0 is skipped: the first launch of each kernel pays a module
    # load on the host, which can outlast the producer's sleep.
    # With `-D MHA_PDL=false` the combine kernel compiles its release out, so
    # there is nothing to observe.
    comptime if MHA_PDL_LEVEL > PDLLevel.OFF:
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

    _ = o_partial_buf^
    _ = lse_buf^
    _ = out_buf^
    _ = chain_dst^
    _ = stamps^


# A tuned small-M, small-N, K = 56 heads * 128 = 7168 shape of the SM100 dense
# matmul dispatch, which runs `gemv_split_k` for 2 <= M < 9.
comptime MM_HEADS = 56
comptime MM_DEPTH = 128
comptime MM_K = MM_HEADS * MM_DEPTH
comptime MM_N = 384


def _launch_matmul[
    variant: Int
](
    c_buf: DeviceBuffer[dtype],
    a_buf: DeviceBuffer[dtype],
    w_buf: DeviceBuffer[dtype],
    num_rows: Int,
    ctx: DeviceContext,
) raises:
    """Launches the SM100 matmul dispatch with PDL on, with no epilogue
    (variant 0), a compute epilogue (1) or a store epilogue (2). The caller's
    epilogue selects which `gemv_split_k` instantiation runs."""
    var a = TileTensor(a_buf, row_major((num_rows, Idx[MM_K]))).as_imm()
    var w = TileTensor(w_buf, row_major(Idx[MM_N], Idx[MM_K])).as_imm()
    var c_ptr = (
        c_buf.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    )
    var c = TileTensor(c_ptr, row_major((num_rows, Idx[MM_N])))

    comptime if variant == 0:
        _matmul_gpu[transpose_b=True, pdl_level=PDLLevel.OVERLAP_AT_BEGINNING](
            c, a, w, ctx
        )
    elif variant == 1:

        def compute_fn[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
            return val * SIMD[_dtype, width](2)

        _matmul_gpu[transpose_b=True, pdl_level=PDLLevel.OVERLAP_AT_BEGINNING](
            c, a, w, compute_fn, ctx
        )
    else:
        var d = TileTensor(c_ptr, row_major((num_rows, Idx[MM_N])))
        var null_c_ptr: Optional[
            MutPointer[Scalar[dtype], MutUntrackedOrigin]
        ] = None
        var c_shape_only = TileTensor(
            null_c_ptr._unsafe_nullable(), row_major((num_rows, Idx[MM_N]))
        )

        @inline(.always)
        def epilogue_fn[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) {var d}:
            d.store(idx, val.cast[dtype]())

        _matmul_gpu[transpose_b=True, pdl_level=PDLLevel.OVERLAP_AT_BEGINNING](
            c_shape_only, a, w, epilogue_fn, ctx
        )


def test_combine_to_matmul_pdl[
    variant: Int
](num_rows: Int, num_partitions: Int, ctx: DeviceContext) raises:
    """Combine kernel followed by the real o-proj style matmul dispatch.

    The matmul is launched with the PDL attribute, so the combine's early
    release lets it start before the combine output exists. It must wait.
    """
    print(
        "test_combine_to_matmul_pdl variant=",
        variant,
        " rows=",
        num_rows,
        " P=",
        num_partitions,
    )

    var rows_heads = num_rows * MM_HEADS
    var o_elems = num_partitions * rows_heads * MM_DEPTH
    var lse_elems = num_partitions * rows_heads
    var a_elems = num_rows * MM_K
    var c_elems = num_rows * MM_N

    var o_partial_buf = ctx.enqueue_create_buffer[dtype](o_elems)
    var lse_buf = ctx.enqueue_create_buffer[DType.float32](lse_elems)
    var a_buf = ctx.enqueue_create_buffer[dtype](a_elems)
    var w_buf = ctx.enqueue_create_buffer[dtype](MM_N * MM_K)
    var c_buf = ctx.enqueue_create_buffer[dtype](c_elems)
    var chain_dst = ctx.enqueue_create_buffer[dtype](ITERATIONS * c_elems)
    var stamps = ctx.enqueue_create_buffer[DType.uint64](2 * ITERATIONS)
    var w_host = List[Scalar[dtype]](length=MM_N * MM_K, fill=Scalar[dtype](0))
    for i in range(MM_N * MM_K):
        w_host[i] = Scalar[dtype](_hash01(i, 99) - 0.5)
    ctx.enqueue_copy(w_buf, w_host)
    ctx.synchronize()

    var ref_host = List[Scalar[dtype]](
        length=ITERATIONS * c_elems, fill=Scalar[dtype](0)
    )
    var ref_one = List[Scalar[dtype]](length=c_elems, fill=Scalar[dtype](0))
    for it in range(ITERATIONS):
        _launch_producer(
            ctx, o_partial_buf, lse_buf, o_elems, lse_elems, it, it, 0, stamps
        )
        ctx.synchronize()
        _combine[MM_DEPTH, dtype](
            ctx,
            o_partial_buf,
            lse_buf,
            a_buf,
            num_partitions,
            num_rows,
            MM_HEADS,
        )
        ctx.synchronize()
        _launch_matmul[variant](c_buf, a_buf, w_buf, num_rows, ctx)
        ctx.synchronize()
        ctx.enqueue_copy(ref_one, c_buf)
        ctx.synchronize()
        for i in range(c_elems):
            ref_host[it * c_elems + i] = ref_one[i]

    # Producer -> combine -> matmul -> consumer, no host sync. The combine's
    # input is poisoned through the late producer, and `a` is poisoned so a
    # matmul that reads it early produces NaNs.
    for it in range(ITERATIONS):
        a_buf.enqueue_fill(nan[dtype]())
        _launch_producer(
            ctx,
            o_partial_buf,
            lse_buf,
            o_elems,
            lse_elems,
            it,
            it,
            PRODUCER_DELAY_NS,
            stamps,
        )
        _combine[MM_DEPTH, dtype](
            ctx,
            o_partial_buf,
            lse_buf,
            a_buf,
            num_partitions,
            num_rows,
            MM_HEADS,
        )
        _launch_matmul[variant](c_buf, a_buf, w_buf, num_rows, ctx)
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
            "combine -> matmul PDL chain differs from the serialized run in ",
            mismatches,
            " of ",
            ITERATIONS * c_elems,
            " elements; first at ",
            first_bad,
            " (iteration ",
            first_bad // c_elems,
            ")",
        )

    _ = o_partial_buf^
    _ = lse_buf^
    _ = a_buf^
    _ = w_buf^
    _ = c_buf^
    _ = chain_dst^
    _ = stamps^


def main() raises:
    with DeviceContext() as ctx:
        # On-ladder partition counts (comptime-unrolled combine) at the
        # production depths, bf16 and f32 partials.
        test_combine_pdl_chain[128, DType.bfloat16](4, 3, 8, ctx)
        test_combine_pdl_chain[128, DType.float32](16, 5, 8, ctx)
        test_combine_pdl_chain[64, DType.bfloat16](148, 2, 4, ctx)
        # Off-ladder count: the runtime-P fallback kernel.
        test_combine_pdl_chain[128, DType.bfloat16](13, 3, 8, ctx)
        # Combine output consumed by the real matmul dispatch (gemv_split_k).
        test_combine_to_matmul_pdl[0](3, 8, ctx)
        test_combine_to_matmul_pdl[0](6, 8, ctx)
        # The caller's epilogue picks the gemv_split_k instantiation.
        test_combine_to_matmul_pdl[1](3, 8, ctx)
        test_combine_to_matmul_pdl[2](3, 8, ctx)
