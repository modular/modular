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
"""PDL ordering tests for the SM100 persistent grouped GEMM.

With programmatic dependent launch the grouped GEMM becomes resident while the
kernel before it is still running, and its trigger releases the kernel after it
just as early. Every input a MoE layer's predecessors write (the activations,
the CSR offsets, the expert ids and the usage stats) must therefore be read
only after the kernel's own `wait_on_dependent_grids()`.

* `test_late_operands`: a producer releases its dependents on entry, sleeps,
  and only then publishes the real operands over zeroed ones. A GEMM that read
  any of them before its wait sees zero experts, or zero activations, and
  leaves its output at the sentinel or wrong. The result must match a PDL-off run on operands that were already
  in place.
* `test_gemm_chain`: two grouped GEMMs back to back, the second reading the
  first's output as its activations, both PDL-launched.
* `test_early_release`: a one-thread kernel launched behind the GEMM stamps
  the device clock before and after its own wait. With the release on entry it
  starts while the GEMM is still running, so the two stamps are far apart; if
  the release were removed it would start only as the GEMM's CTAs exit and the
  stamps would nearly coincide. The first two tests check outputs only, and
  stream order alone would also give the right outputs.
* `test_after_multiwave_dense`: a dense SM100 matmul that spans more than one
  wave of CTAs, and holds TMEM while it runs, precedes the grouped GEMM. The
  grouped GEMM's early CTAs must neither deadlock against it nor read its
  output early.

* `test_offsets_loads_not_invariant`: the scheduler reads the CSR offsets with
  plain loads when the kernel runs under PDL. An invariant load compiles to
  `ld.global.nc`, which ptxas may hoist above the wait, and no output check
  can see that hoist.

The output cases are bit-exact against the same call with PDL off.
"""

from std.time import global_perf_counter_ns, sleep

from layout import Idx, TileTensor, row_major
from layout._fillers import random
from linalg.grouped_matmul import grouped_matmul
from linalg.grouped_matmul_tile_scheduler import TileScheduler
from linalg.matmul.gpu import _matmul_gpu
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.testing import assert_equal, assert_true

comptime dtype = DType.bfloat16
comptime SENTINEL = Scalar[dtype](7777.0)


def publish_late(
    a_dst: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    a_src: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    a_len_i: Int32,
    offsets_dst: MutPointer[UInt32, MutUntrackedOrigin],
    offsets_src: MutPointer[UInt32, MutUntrackedOrigin],
    ids_dst: MutPointer[Int32, MutUntrackedOrigin],
    ids_src: MutPointer[Int32, MutUntrackedOrigin],
    stats_dst: MutPointer[UInt32, MutUntrackedOrigin],
    stats_src: MutPointer[UInt32, MutUntrackedOrigin],
    num_experts_i: Int32,
):
    """Releases dependents at once, then publishes the operands late."""
    var a_len = Int(a_len_i)
    var num_experts = Int(num_experts_i)
    launch_dependent_grids()
    sleep(0.05)
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, a_len, stride):
        a_dst[i] = a_src[i]
    for i in range(tid, num_experts + 1, stride):
        offsets_dst[i] = offsets_src[i]
    for i in range(tid, num_experts, stride):
        ids_dst[i] = ids_src[i]
    for i in range(tid, 2, stride):
        stats_dst[i] = stats_src[i]


def stamp_around_wait(times: MutPointer[UInt64, MutUntrackedOrigin]):
    """Records the device clock before and after this grid's own wait."""
    var before = global_perf_counter_ns()
    wait_on_dependent_grids()
    var after = global_perf_counter_ns()
    times[0] = before
    times[1] = after


def _fill_sentinel(ctx: DeviceContext, buf: DeviceBuffer[dtype], n: Int) raises:
    var host = ctx.enqueue_create_host_buffer[dtype](n)
    for i in range(n):
        host[i] = SENTINEL
    ctx.enqueue_copy(buf, host)


def _assert_same(
    ctx: DeviceContext,
    expect: DeviceBuffer[dtype],
    actual: DeviceBuffer[dtype],
    n: Int,
    what: String,
) raises:
    var e = ctx.enqueue_create_host_buffer[dtype](n)
    var a = ctx.enqueue_create_host_buffer[dtype](n)
    ctx.enqueue_copy(e, expect)
    ctx.enqueue_copy(a, actual)
    ctx.synchronize()
    var bad = 0
    var first = -1
    for i in range(n):
        if e[i] != a[i]:
            if first < 0:
                first = i
            bad += 1
    assert_equal(bad, 0, String(what, ": mismatches, first at ", first))


struct _Routing(Movable):
    var offsets: DeviceBuffer[.uint32]
    var ids: DeviceBuffer[.int32]
    var stats: DeviceBuffer[.uint32]
    var total: Int
    var max_tokens: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        counts: List[Int],
        ids: List[Int],
        num_experts: Int,
    ) raises:
        self.offsets = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
        self.ids = ctx.enqueue_create_buffer[.int32](num_experts)
        self.stats = ctx.enqueue_create_buffer[.uint32](2)
        var off_h = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
        var ids_h = ctx.enqueue_create_host_buffer[.int32](num_experts)
        var stats_h = ctx.enqueue_create_host_buffer[.uint32](2)
        off_h[0] = 0
        self.total = 0
        self.max_tokens = 0
        for e in range(num_experts):
            off_h[e + 1] = off_h[e] + UInt32(counts[e])
            ids_h[e] = Int32(ids[e])
            self.total += counts[e]
            self.max_tokens = max(self.max_tokens, counts[e])
        stats_h[0] = UInt32(self.max_tokens)
        stats_h[1] = UInt32(num_experts)
        ctx.enqueue_copy(self.offsets, off_h)
        ctx.enqueue_copy(self.ids, ids_h)
        ctx.enqueue_copy(self.stats, stats_h)
        ctx.synchronize()


def _random_buffer[
    n_experts: Int, rows: Int, cols: Int
](ctx: DeviceContext, scale: Float64) raises -> DeviceBuffer[dtype]:
    var buf = ctx.enqueue_create_buffer[dtype](n_experts * rows * cols)
    var host = ctx.enqueue_create_host_buffer[dtype](n_experts * rows * cols)
    random(
        TileTensor(host, row_major[n_experts, rows, cols]()),
        min=Scalar[dtype](-scale),
        max=Scalar[dtype](scale),
    )
    ctx.enqueue_copy(buf, host)
    return buf


def test_late_operands[
    level: PDLLevel, N: Int, K: Int, num_experts: Int
](counts: List[Int], ids: List[Int], ctx: DeviceContext) raises:
    print("late operands: level", level._level, "N", N, "K", K)
    var routing = _Routing(ctx, counts, ids, num_experts)
    var total = routing.total

    var a = ctx.enqueue_create_buffer[dtype](total * K)
    var a_host = ctx.enqueue_create_host_buffer[dtype](total * K)
    random(TileTensor(a_host, row_major(total, Idx[K])), min=-1.0, max=1.0)
    ctx.enqueue_copy(a, a_host)
    var b = _random_buffer[num_experts, N, K](ctx, 0.2)
    var c_ref = ctx.enqueue_create_buffer[dtype](total * N)
    var c = ctx.enqueue_create_buffer[dtype](total * N)
    _fill_sentinel(ctx, c_ref, total * N)
    _fill_sentinel(ctx, c, total * N)
    ctx.synchronize()

    var b_t = TileTensor(b, row_major[num_experts, N, K]())
    var offsets_t = TileTensor(routing.offsets, row_major(num_experts + 1))
    var ids_t = TileTensor(routing.ids, row_major(num_experts))
    var stats_t = TileTensor(routing.stats, row_major(Idx[2]))

    grouped_matmul(
        TileTensor(c_ref, row_major(total, Idx[N])),
        TileTensor(a, row_major(total, Idx[K])),
        b_t,
        offsets_t,
        ids_t,
        stats_t,
        ctx,
    )
    ctx.synchronize()

    # Run the PDL instantiation once on operands that are already in place:
    # loading its module on first launch can block the host until the producer
    # below has finished, which would hide the early launch this test needs.
    var c_warm = ctx.enqueue_create_buffer[dtype](total * N)
    grouped_matmul[pdl_level=level](
        TileTensor(c_warm, row_major(total, Idx[N])),
        TileTensor(a, row_major(total, Idx[K])),
        b_t,
        offsets_t,
        ids_t,
        stats_t,
        ctx,
    )
    ctx.synchronize()
    _assert_same(ctx, c_ref, c_warm, total * N, "operands in place")

    # The GEMM reads these; the producer overwrites them late.
    var a_in = ctx.enqueue_create_buffer[dtype](total * K)
    var offsets_in = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var ids_in = ctx.enqueue_create_buffer[.int32](num_experts)
    var stats_in = ctx.enqueue_create_buffer[.uint32](2)
    a_in.enqueue_fill(0)
    offsets_in.enqueue_fill(0)
    ids_in.enqueue_fill(0)
    stats_in.enqueue_fill(0)
    ctx.synchronize()

    ctx.enqueue_function[publish_late](
        a_in,
        a,
        Int32(total * K),
        offsets_in,
        routing.offsets,
        ids_in,
        routing.ids,
        stats_in,
        routing.stats,
        Int32(num_experts),
        grid_dim=8,
        block_dim=256,
    )
    grouped_matmul[pdl_level=level](
        TileTensor(c, row_major(total, Idx[N])),
        TileTensor(a_in, row_major(total, Idx[K])),
        b_t,
        TileTensor(offsets_in, row_major(num_experts + 1)),
        TileTensor(ids_in, row_major(num_experts)),
        TileTensor(stats_in, row_major(Idx[2])),
        ctx,
    )
    ctx.synchronize()
    _assert_same(ctx, c_ref, c, total * N, "late operands")


def _gemm_pair[
    level: PDLLevel, N1: Int, K1: Int, N2: Int, num_experts: Int
](
    ctx: DeviceContext,
    a: DeviceBuffer[dtype],
    b1: DeviceBuffer[dtype],
    b2: DeviceBuffer[dtype],
    mut mid: DeviceBuffer[dtype],
    mut dst: DeviceBuffer[dtype],
    offsets_buf: DeviceBuffer[.uint32],
    ids_buf: DeviceBuffer[.int32],
    stats_buf: DeviceBuffer[.uint32],
    total: Int,
) raises:
    grouped_matmul[pdl_level=level](
        TileTensor(mid, row_major(total, Idx[N1])),
        TileTensor(a, row_major(total, Idx[K1])),
        TileTensor(b1, row_major[num_experts, N1, K1]()),
        TileTensor(offsets_buf, row_major(num_experts + 1)),
        TileTensor(ids_buf, row_major(num_experts)),
        TileTensor(stats_buf, row_major(Idx[2])),
        ctx,
    )
    grouped_matmul[pdl_level=level](
        TileTensor(dst, row_major(total, Idx[N2])),
        TileTensor(mid, row_major(total, Idx[N1])),
        TileTensor(b2, row_major[num_experts, N2, N1]()),
        TileTensor(offsets_buf, row_major(num_experts + 1)),
        TileTensor(ids_buf, row_major(num_experts)),
        TileTensor(stats_buf, row_major(Idx[2])),
        ctx,
    )


def _dense_then_grouped[
    level: PDLLevel, N: Int, K: Int, dense_k: Int, num_experts: Int
](
    ctx: DeviceContext,
    x: DeviceBuffer[dtype],
    wd: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
    mut mid: DeviceBuffer[dtype],
    mut dst: DeviceBuffer[dtype],
    offsets_buf: DeviceBuffer[.uint32],
    ids_buf: DeviceBuffer[.int32],
    stats_buf: DeviceBuffer[.uint32],
    total: Int,
) raises:
    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=True,
        pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
    ](
        TileTensor(mid, row_major(total, Idx[K])),
        TileTensor(x, row_major(total, Idx[dense_k])).as_imm(),
        TileTensor(wd, row_major(Idx[K], Idx[dense_k])).as_imm(),
        ctx,
    )
    grouped_matmul[pdl_level=level](
        TileTensor(dst, row_major(total, Idx[N])),
        TileTensor(mid, row_major(total, Idx[K])),
        TileTensor(b, row_major[num_experts, N, K]()),
        TileTensor(offsets_buf, row_major(num_experts + 1)),
        TileTensor(ids_buf, row_major(num_experts)),
        TileTensor(stats_buf, row_major(Idx[2])),
        ctx,
    )


def test_gemm_chain[
    level: PDLLevel, N1: Int, K1: Int, N2: Int, num_experts: Int
](counts: List[Int], ids: List[Int], ctx: DeviceContext) raises:
    print("gemm chain: level", level._level, "N1", N1, "K1", K1, "N2", N2)
    var routing = _Routing(ctx, counts, ids, num_experts)
    var total = routing.total

    var a = ctx.enqueue_create_buffer[dtype](total * K1)
    var a_host = ctx.enqueue_create_host_buffer[dtype](total * K1)
    random(TileTensor(a_host, row_major(total, Idx[K1])), min=-1.0, max=1.0)
    ctx.enqueue_copy(a, a_host)
    var b1 = _random_buffer[num_experts, N1, K1](ctx, 0.2)
    var b2 = _random_buffer[num_experts, N2, N1](ctx, 0.2)
    var mid_ref = ctx.enqueue_create_buffer[dtype](total * N1)
    var out_ref = ctx.enqueue_create_buffer[dtype](total * N2)
    var mid = ctx.enqueue_create_buffer[dtype](total * N1)
    var out = ctx.enqueue_create_buffer[dtype](total * N2)
    _fill_sentinel(ctx, out_ref, total * N2)
    _fill_sentinel(ctx, out, total * N2)
    ctx.synchronize()

    _gemm_pair[PDLLevel.OFF, N1, K1, N2, num_experts](
        ctx,
        a,
        b1,
        b2,
        mid_ref,
        out_ref,
        routing.offsets,
        routing.ids,
        routing.stats,
        total,
    )
    ctx.synchronize()
    _gemm_pair[level, N1, K1, N2, num_experts](
        ctx,
        a,
        b1,
        b2,
        mid,
        out,
        routing.offsets,
        routing.ids,
        routing.stats,
        total,
    )
    ctx.synchronize()
    _assert_same(ctx, mid_ref, mid, total * N1, "chain first GEMM")
    _assert_same(ctx, out_ref, out, total * N2, "chain second GEMM")


def test_early_release[
    level: PDLLevel, N: Int, K: Int, num_experts: Int
](counts: List[Int], ids: List[Int], ctx: DeviceContext) raises:
    print("early release: level", level._level, "N", N, "K", K)
    var routing = _Routing(ctx, counts, ids, num_experts)
    var total = routing.total

    var a = ctx.enqueue_create_buffer[dtype](total * K)
    a.enqueue_fill(Scalar[dtype](0.5))
    var b = _random_buffer[num_experts, N, K](ctx, 0.2)
    var c = ctx.enqueue_create_buffer[dtype](total * N)
    var times = ctx.enqueue_create_buffer[DType.uint64](2)
    var times_host = ctx.enqueue_create_host_buffer[DType.uint64](2)
    ctx.synchronize()

    # The first iteration also warms up the kernel instantiation, so a
    # first-launch module load cannot serialize the launches that follow.
    var gaps = List[Int]()
    for i in range(6):
        grouped_matmul[pdl_level=level](
            TileTensor(c, row_major(total, Idx[N])),
            TileTensor(a, row_major(total, Idx[K])),
            TileTensor(b, row_major[num_experts, N, K]()),
            TileTensor(routing.offsets, row_major(num_experts + 1)),
            TileTensor(routing.ids, row_major(num_experts)),
            TileTensor(routing.stats, row_major(Idx[2])),
            ctx,
        )
        ctx.enqueue_function[stamp_around_wait](
            times,
            grid_dim=1,
            block_dim=1,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        ctx.enqueue_copy(times_host, times)
        ctx.synchronize()
        if i > 0:
            gaps.append(Int(times_host[1] - times_host[0]))
    sort(gaps)
    var gap_ns = gaps[len(gaps) // 2]
    print("  median start-to-release gap (ns):", gap_ns)
    # The GEMM runs for ~100 us here; a launch that follows its completion
    # leaves a gap of a few us at most.
    assert_true(
        gap_ns > 20_000,
        String("consumer did not start early: gap ", gap_ns, " ns"),
    )


def test_after_multiwave_dense[
    level: PDLLevel, N: Int, K: Int, num_experts: Int
](counts: List[Int], ids: List[Int], ctx: DeviceContext) raises:
    comptime dense_k = 256
    print("after dense: level", level._level, "N", N, "K", K)
    var routing = _Routing(ctx, counts, ids, num_experts)
    var total = routing.total

    var x = ctx.enqueue_create_buffer[dtype](total * dense_k)
    var x_host = ctx.enqueue_create_host_buffer[dtype](total * dense_k)
    random(
        TileTensor(x_host, row_major(total, Idx[dense_k])), min=-1.0, max=1.0
    )
    ctx.enqueue_copy(x, x_host)
    var wd = _random_buffer[1, K, dense_k](ctx, 0.2)
    var b = _random_buffer[num_experts, N, K](ctx, 0.2)
    var mid_ref = ctx.enqueue_create_buffer[dtype](total * K)
    var out_ref = ctx.enqueue_create_buffer[dtype](total * N)
    var mid = ctx.enqueue_create_buffer[dtype](total * K)
    var out = ctx.enqueue_create_buffer[dtype](total * N)
    _fill_sentinel(ctx, out_ref, total * N)
    _fill_sentinel(ctx, out, total * N)
    ctx.synchronize()

    _dense_then_grouped[PDLLevel.OFF, N, K, dense_k, num_experts](
        ctx,
        x,
        wd,
        b,
        mid_ref,
        out_ref,
        routing.offsets,
        routing.ids,
        routing.stats,
        total,
    )
    ctx.synchronize()
    _dense_then_grouped[level, N, K, dense_k, num_experts](
        ctx,
        x,
        wd,
        b,
        mid,
        out,
        routing.offsets,
        routing.ids,
        routing.stats,
        total,
    )
    ctx.synchronize()
    _assert_same(ctx, out_ref, out, total * N, "after dense")


def _offsets_after_wait[
    invariant_offsets: Bool
](
    offsets: ImmPointer[UInt32, ImmutAnyOrigin],
    num_active_experts: Int,
    first_m: MutPointer[UInt32, MutAnyOrigin],
):
    """Runs the grouped GEMM's scheduler on predecessor-written offsets."""
    wait_on_dependent_grids()
    var scheduler = TileScheduler[
        static_MN=256,
        tile_shape=(128, 128, 64),
        invariant_offsets=invariant_offsets,
    ](
        num_active_experts,
        TileTensor(offsets, row_major(num_active_experts + 1)),
    )
    first_m[0] = scheduler.fetch_next_work().m


def _offsets_ptx[invariant_offsets: Bool]() raises -> String:
    return _compile_code[
        _offsets_after_wait[invariant_offsets],
        emission_kind="asm",
        target=get_gpu_target["sm_100a"](),
    ]().asm


def test_offsets_loads_not_invariant() raises:
    print("offsets loads are not invariant")
    # Control: the default scheduler does use `ld.global.nc` for the offsets,
    # so the check below can fail.
    assert_true("ld.global.nc" in _offsets_ptx[True]())
    var ptx = _offsets_ptx[False]()
    assert_true("griddepcontrol.wait" in ptx)
    assert_true("ld.global.nc" not in ptx)


def run_all[level: PDLLevel](ctx: DeviceContext) raises:
    # Ragged, with an empty expert and out-of-order ids.
    var counts: List[Int] = [27, 0, 150, 1, 300, 3]
    var ids: List[Int] = [4, 0, 2, 5, 1, 3]
    # 2-CTA (N a multiple of 256) and 1-CTA tiles.
    test_late_operands[level, 1280, 1024, 6](counts, ids, ctx)
    test_late_operands[level, 2880, 512, 6](counts, ids, ctx)
    test_gemm_chain[level, 512, 256, 256, 6](counts, ids, ctx)
    test_gemm_chain[level, 640, 256, 384, 6](counts, ids, ctx)

    # More than a wave of dense-matmul CTAs ahead of the grouped GEMM.
    var big: List[Int] = [1500, 0, 2047, 1, 3000, 3]
    test_after_multiwave_dense[level, 1280, 512, 6](big, ids, ctx)
    test_after_multiwave_dense[level, 384, 512, 6](big, ids, ctx)


def main() raises:
    test_offsets_loads_not_invariant()
    with DeviceContext() as ctx:
        run_all[PDLLevel.ON](ctx)
        run_all[PDLLevel.OVERLAP_AT_BEGINNING](ctx)
        var big: List[Int] = [1500, 0, 2047, 1, 3000, 3]
        var ids: List[Int] = [4, 0, 2, 5, 1, 3]
        test_early_release[PDLLevel.OVERLAP_AT_BEGINNING, 2880, 4096, 6](
            big, ids, ctx
        )
