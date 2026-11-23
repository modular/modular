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
"""PDL correctness for the epilogue closures of `gemm_mma_cpasync` (B200).

A closure can read tensors that the predecessor grid writes, such as a
residual and a gate. The closure runs in the compute warps, which do not take
the loader warps' wait, and the kernel can be resident while its predecessor
still runs.

`late_producer` releases its dependents at once, then sleeps before it writes
the residual and the one-element gate that the closure reads. Both buffers
start as NaN, so a closure that reads one before the producer has finished
stores NaN. Each output must match the same GEMM and closure with PDL off, bit
for bit.

This test passes on the current `ptxas` without the epilogue wait: `ptxas`
keeps the closure loads below the barrier chain that the load warps' wait
starts. It guards the `wait_on_dependent_grids` contract, not a failure seen
today, and it fails when the load warps' waits are removed.
"""

from std.math import nan
from std.random import rand
from std.sys.intrinsics import llvm_intrinsic

from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, launch_dependent_grids
from layout import Coord, Idx, TileTensor, row_major
from linalg.gemv import gemm_mma_cpasync

comptime bf16 = DType.bfloat16


def late_producer(
    src: MutPointer[Scalar[bf16], MutUntrackedOrigin],
    dst: MutPointer[Scalar[bf16], MutUntrackedOrigin],
    length: Int32,
    gate_src: MutPointer[Scalar[bf16], MutUntrackedOrigin],
    gate_dst: MutPointer[Scalar[bf16], MutUntrackedOrigin],
    delay_us: Int32,
):
    """Releases the dependent grid at entry, then writes `dst` and the gate late.

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
    if global_idx.x == 0:
        gate_dst[0] = gate_src[0]


def run_gemm[
    N: Int, K: Int, swap_ab: Bool, store_closure: Bool, level: PDLLevel
](
    ctx: DeviceContext,
    m: Int,
    mut out_dev: DeviceBuffer[bf16],
    mut shape_dev: DeviceBuffer[bf16],
    act_dev: DeviceBuffer[bf16],
    weight_dev: DeviceBuffer[bf16],
    resid_dev: DeviceBuffer[bf16],
    gate_dev: DeviceBuffer[bf16],
) raises:
    """Runs `out = gate * (act @ weight^T) + resid`, both read in the closure.

    A store closure writes `out` itself; the launch then takes the output
    shape from `shape_dev`, which it does not write.
    """
    var out = TileTensor(out_dev, row_major(Int64(m), Idx[N]))
    var act = TileTensor(act_dev, row_major(Int64(m), Idx[K])).as_imm()
    var weight = TileTensor(weight_dev, row_major(Idx[N], Idx[K])).as_imm()
    var resid = TileTensor(resid_dev, row_major(Int64(m), Idx[N])).as_imm()
    var gate = TileTensor(gate_dev, row_major(Idx[1])).as_imm()

    comptime if store_closure:

        def store_sum[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var out, var resid, var gate}:
            comptime assert idx.rank == 2, "the matmul epilogue is rank 2"
            var g = gate.load[1](Coord(Idx[0])).cast[dtype]()
            var sum = val * g + resid.load[width](idx).cast[dtype]()
            out.store(idx, sum.cast[bf16]())

        var shape = TileTensor(shape_dev, row_major(Int64(m), Idx[N]))
        gemm_mma_cpasync[pdl_level=level, swapAB=swap_ab](
            shape, act, weight, m, K, N, 1, store_sum, ctx
        )
    else:

        def add_resid[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var resid, var gate} -> SIMD[
            dtype, width
        ]:
            comptime assert idx.rank == 2, "the matmul epilogue is rank 2"
            var g = gate.load[1](Coord(Idx[0])).cast[dtype]()
            return val * g + resid.load[width](idx).cast[dtype]()

        gemm_mma_cpasync[pdl_level=level, swapAB=swap_ab](
            out, act, weight, m, K, N, 1, add_resid, ctx
        )


def run_case[
    N: Int, K: Int, swap_ab: Bool, store_closure: Bool
](ctx: DeviceContext, m: Int) raises:
    """Checks one config and closure kind behind a late producer, PDL on.

    The launches of all iterations go out with no host synchronization, so
    the GEMM's own waits are the only ordering. The first launch of a kernel
    loads its module and does not start early, so it takes several iterations.
    """
    comptime iters = 4
    comptime delay_us = 1000
    print(
        t"gemm_mma_cpasync epilogue PDL: M={m} N={N} K={K} swapAB={swap_ab}"
        t" store={store_closure}"
    )

    var act_host = ctx.enqueue_create_host_buffer[bf16](m * K)
    var weight_host = ctx.enqueue_create_host_buffer[bf16](N * K)
    var r_host = ctx.enqueue_create_host_buffer[bf16](m * N)
    ctx.synchronize()
    rand(act_host.unsafe_ptr(), m * K, min=-0.5, max=0.5)
    rand(weight_host.unsafe_ptr(), N * K, min=-0.5, max=0.5)
    rand(r_host.unsafe_ptr(), m * N, min=-1.0, max=1.0)

    var act_dev = ctx.enqueue_create_buffer[bf16](m * K)
    var weight_dev = ctx.enqueue_create_buffer[bf16](N * K)
    var r_src = ctx.enqueue_create_buffer[bf16](m * N)
    var gate_src = ctx.enqueue_create_buffer[bf16](1)
    gate_src.enqueue_fill(Scalar[bf16](1.0))
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(weight_dev, weight_host)
    ctx.enqueue_copy(r_src, r_host)

    var ref_dev = ctx.enqueue_create_buffer[bf16](m * N)
    ref_dev.enqueue_fill(nan[bf16]())
    var shape_dev = ctx.enqueue_create_buffer[bf16](m * N)
    var resid_devs = List[DeviceBuffer[bf16]]()
    var gate_devs = List[DeviceBuffer[bf16]]()
    var out_devs = List[DeviceBuffer[bf16]]()
    for _ in range(iters):
        resid_devs.append(ctx.enqueue_create_buffer[bf16](m * N))
        gate_devs.append(ctx.enqueue_create_buffer[bf16](1))
        out_devs.append(ctx.enqueue_create_buffer[bf16](m * N))
        resid_devs[len(resid_devs) - 1].enqueue_fill(nan[bf16]())
        gate_devs[len(gate_devs) - 1].enqueue_fill(nan[bf16]())
        out_devs[len(out_devs) - 1].enqueue_fill(nan[bf16]())
    ctx.synchronize()

    for i in range(iters):
        ctx.enqueue_function[late_producer](
            r_src,
            resid_devs[i],
            Int32(m * N),
            gate_src,
            gate_devs[i],
            Int32(delay_us),
            grid_dim=8,
            block_dim=256,
        )
        run_gemm[N, K, swap_ab, store_closure, PDLLevel.OVERLAP_AT_BEGINNING](
            ctx,
            m,
            out_devs[i],
            shape_dev,
            act_dev,
            weight_dev,
            resid_devs[i],
            gate_devs[i],
        )
    ctx.synchronize()

    run_gemm[N, K, swap_ab, store_closure, PDLLevel.OFF](
        ctx,
        m,
        ref_dev,
        shape_dev,
        act_dev,
        weight_dev,
        r_src,
        gate_src,
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
                " outputs differing from the PDL-off GEMM",
            )
    print("    PASS")


def main() raises:
    with DeviceContext() as ctx:
        run_case[4096, 2048, False, False](ctx, 8)
        run_case[4096, 2048, False, True](ctx, 8)
        run_case[4096, 2048, True, False](ctx, 8)
        run_case[4096, 2048, True, True](ctx, 8)
