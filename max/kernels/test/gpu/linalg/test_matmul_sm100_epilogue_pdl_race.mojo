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
"""PDL correctness for the epilogue closures of the SM100 matmul (B200).

A closure can read a tensor that the predecessor grid writes, as the graph
compiler's `matmul + add` fusion reads its residual. The closure runs in the
epilogue warps, and the matmul releases its dependents before its own wait, so
it can be resident while its predecessor still runs.

`late_producer` releases its dependents at once, then sleeps before it writes
the residual that the closure reads. The residual buffer starts as NaN, so a
closure that reads it before the producer has finished stores NaN. Each output
must match the same matmul and closure with PDL off, bit for bit.

The shapes take the swapped-AB tile (M = 304) and the plain 2-SM tile
(M = 1024). Each runs a compute closure, which returns the sum, and a store
closure, which writes it.
"""

from std.math import nan
from std.random import rand
from std.sys.intrinsics import llvm_intrinsic

from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, launch_dependent_grids
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu.sm100_structured.default.dispatch import (
    matmul_dispatch_sm100,
)

comptime bf16 = DType.bfloat16


def late_producer(
    src: MutPointer[Scalar[bf16], MutAnyOrigin],
    dst: MutPointer[Scalar[bf16], MutAnyOrigin],
    length: Int32,
    delay_us: Int32,
):
    """Releases the dependent grid at entry, then writes `dst` late.

    The dependent can only launch once every block of this grid has released
    it, so the grid is small enough to be resident at once and strides over
    `dst`.
    """
    launch_dependent_grids()
    # nanosleep takes nanoseconds and caps at about 1 ms per call.
    var remaining = Int(delay_us)
    while remaining > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        remaining -= 1000
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(global_idx.x, Int(length), stride):
        dst[i] = src[i]


def run_matmul[
    N: Int, K: Int, store_closure: Bool, level: PDLLevel
](
    ctx: DeviceContext,
    m: Int,
    mut out_dev: DeviceBuffer[bf16],
    mut shape_dev: DeviceBuffer[bf16],
    a_dev: DeviceBuffer[bf16],
    b_dev: DeviceBuffer[bf16],
    resid_dev: DeviceBuffer[bf16],
) raises:
    """Runs `out = a @ b^T + resid`, the residual read in an epilogue closure.

    A store closure writes `out` itself; the dispatch then takes the output
    shape from `shape_dev`, which it does not write.
    """
    var out = TileTensor(out_dev, row_major(Int64(m), Idx[N]))
    var a = TileTensor(a_dev, row_major(Int64(m), Idx[K])).as_imm()
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var resid = TileTensor(resid_dev, row_major(Int64(m), Idx[N])).as_imm()

    comptime if store_closure:

        def store_sum[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var out, var resid}:
            comptime assert idx.rank == 2, "the matmul epilogue is rank 2"
            var sum = val + resid.load[width](idx).cast[dtype]()
            out.store(idx, sum.cast[bf16]())

        var shape = TileTensor(shape_dev, row_major(Int64(m), Idx[N]))
        matmul_dispatch_sm100[transpose_b=True, pdl_level=level](
            shape, a, b, store_sum, ctx
        )
    else:

        def add_resid[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var resid} -> SIMD[
            dtype, width
        ]:
            comptime assert idx.rank == 2, "the matmul epilogue is rank 2"
            return val + resid.load[width](idx).cast[dtype]()

        matmul_dispatch_sm100[transpose_b=True, pdl_level=level](
            out, a, b, add_resid, ctx
        )


def run_case[
    N: Int, K: Int, store_closure: Bool
](ctx: DeviceContext, m: Int) raises:
    """Checks one shape and closure kind behind a late producer, PDL on.

    The launches of all iterations go out with no host synchronization, so
    the matmul's own waits are the only ordering. The first launch of a kernel
    loads its module and does not start early, so it takes several iterations.
    """
    comptime iters = 4
    comptime delay_us = 1000
    print(t"epilogue closure PDL: M={m} N={N} K={K} store={store_closure}")

    var a_host = ctx.enqueue_create_host_buffer[bf16](m * K)
    var b_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var r_host = ctx.enqueue_create_host_buffer[bf16](m * N)
    ctx.synchronize()
    rand(a_host.unsafe_ptr(), m * K, min=-1.0, max=1.0)
    rand(b_host.unsafe_ptr(), N * K, min=-1.0, max=1.0)
    rand(r_host.unsafe_ptr(), m * N, min=-1.0, max=1.0)

    var a_dev = ctx.enqueue_create_buffer[bf16](m * K)
    var b_dev = ctx.enqueue_create_buffer[bf16](N * K)
    var r_src = ctx.enqueue_create_buffer[bf16](m * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(r_src, r_host)

    var ref_dev = ctx.enqueue_create_buffer[bf16](m * N)
    ref_dev.enqueue_fill(nan[bf16]())
    var shape_dev = ctx.enqueue_create_buffer[bf16](m * N)
    var resid_devs = List[DeviceBuffer[bf16]]()
    var out_devs = List[DeviceBuffer[bf16]]()
    for _ in range(iters):
        resid_devs.append(ctx.enqueue_create_buffer[bf16](m * N))
        out_devs.append(ctx.enqueue_create_buffer[bf16](m * N))
        resid_devs[len(resid_devs) - 1].enqueue_fill(nan[bf16]())
        out_devs[len(out_devs) - 1].enqueue_fill(nan[bf16]())
    ctx.synchronize()

    for i in range(iters):
        ctx.enqueue_function[late_producer](
            r_src,
            resid_devs[i],
            Int32(m * N),
            Int32(delay_us),
            grid_dim=8,
            block_dim=256,
        )
        run_matmul[N, K, store_closure, PDLLevel.OVERLAP_AT_BEGINNING](
            ctx, m, out_devs[i], shape_dev, a_dev, b_dev, resid_devs[i]
        )
    ctx.synchronize()

    run_matmul[N, K, store_closure, PDLLevel.OFF](
        ctx, m, ref_dev, shape_dev, a_dev, b_dev, r_src
    )
    ctx.synchronize()

    var want = ctx.enqueue_create_host_buffer[bf16](m * N)
    var got = ctx.enqueue_create_host_buffer[bf16](m * N)
    ctx.enqueue_copy(want, ref_dev)
    ctx.synchronize()
    for i in range(iters):
        ctx.enqueue_copy(got, out_devs[i])
        ctx.synchronize()
        var bad = 0
        for j in range(m * N):
            # Bit comparison: NaN never equals itself, so a stale read fails.
            if got[j].to_bits() != want[j].to_bits():
                bad += 1
        if bad != 0:
            raise Error(
                "PDL race: iteration ",
                i,
                " has ",
                bad,
                " of ",
                m * N,
                " outputs differing from the PDL-off matmul",
            )
    print("    PASS")


def main() raises:
    with DeviceContext() as ctx:
        # Swapped-AB tile, as in a transformer's text stream.
        run_case[3072, 9216, False](ctx, 304)
        run_case[3072, 9216, True](ctx, 304)
        # Plain 2-SM tile.
        run_case[3072, 3072, False](ctx, 1024)
        run_case[3072, 3072, True](ctx, 1024)
