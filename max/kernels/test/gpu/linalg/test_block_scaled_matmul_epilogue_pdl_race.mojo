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
"""PDL correctness for the epilogue closures of the SM100 block-scaled matmul.

A closure can read tensors that the predecessor grid writes, such as a
residual and a gate. The closure runs in the epilogue warps, and the matmul
releases its dependents before its own wait, so it can be resident while its
predecessor still runs.

`late_producer` releases its dependents at once, then sleeps before it writes
the residual and the one-element gate that the closure reads. Both buffers
start as NaN, so a closure that reads one before the producer has finished
stores NaN. Each output must match the same matmul and closure with PDL off,
bit for bit.

An M of 1 takes the small-BN kernel; a larger M takes the standard one.

This test passes on the current `ptxas` without the epilogue wait: `ptxas`
keeps the closure loads below the barrier chain that the load warps' wait
starts. It guards the `wait_on_dependent_grids` contract, not a failure seen
today, and it fails when the load warps' waits are removed.
"""

from std.math import ceildiv, nan
from std.random import rand
from std.sys.intrinsics import llvm_intrinsic
from std.testing import assert_equal

from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel, launch_dependent_grids
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    set_scale_factor,
)
from linalg.matmul.gpu.sm100.block_scaled_dispatch import (
    DISPATCH_HIT,
    heuristic_and_outliers_dispatch,
)
from linalg.utils import _as_compute_fn

comptime bf16 = DType.bfloat16
comptime fp8 = DType.float8_e4m3fn
comptime SF = MXFP8_SF_VECTOR_SIZE
comptime sf_type = MXFP8_SF_DTYPE


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


def run_matmul[
    N: Int, K: Int, store_closure: Bool, level: PDLLevel
](
    ctx: DeviceContext,
    m: Int,
    mut out_dev: DeviceBuffer[bf16],
    mut shape_dev: DeviceBuffer[bf16],
    a_dev: DeviceBuffer[fp8],
    b_dev: DeviceBuffer[fp8],
    a_sf_dev: DeviceBuffer[sf_type],
    b_sf_dev: DeviceBuffer[sf_type],
    resid_dev: DeviceBuffer[bf16],
    gate_dev: DeviceBuffer[bf16],
) raises:
    """Runs `out = gate * (a @ b^T) + resid`, both read in an epilogue closure.

    A store closure writes `out` itself; the dispatch then takes the output
    shape from `shape_dev`, which it does not write.
    """
    var out = TileTensor(out_dev, row_major(Int64(m), Idx[N]))
    var a = TileTensor(a_dev, row_major(Int64(m), Idx[K])).as_imm()
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K])).as_imm()
    var a_sf = TileTensor(
        a_sf_dev,
        row_major(
            ceildiv(m, SF_MN_GROUP_SIZE),
            Idx[ceildiv(K, SF * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    ).as_imm()
    var b_sf = TileTensor(
        b_sf_dev,
        row_major(
            Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
            Idx[ceildiv(K, SF * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        ),
    ).as_imm()
    var resid = TileTensor(resid_dev, row_major(Int64(m), Idx[N])).as_imm()
    var gate = TileTensor(gate_dev, row_major(Idx[1])).as_imm()

    var status: Int
    comptime if store_closure:

        def store_sum[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var out, var resid, var gate}:
            comptime assert idx.rank == 2, "the matmul epilogue is rank 2"
            var g = gate.load[1](Coord(Idx[0])).cast[dtype]()
            var sum = val * g + resid.load[width](idx).cast[dtype]()
            out.store(idx, sum.cast[bf16]())

        var shape = TileTensor(shape_dev, row_major(Int64(m), Idx[N]))
        status = heuristic_and_outliers_dispatch[
            SF_VECTOR_SIZE=SF, pdl_level=level
        ](shape, a, b, a_sf, b_sf, Float32(1.0), store_sum, ctx)
    else:

        def add_resid[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[dtype, width]) {var resid, var gate} -> SIMD[
            dtype, width
        ]:
            comptime assert idx.rank == 2, "the matmul epilogue is rank 2"
            var g = gate.load[1](Coord(Idx[0])).cast[dtype]()
            return val * g + resid.load[width](idx).cast[dtype]()

        status = heuristic_and_outliers_dispatch[
            SF_VECTOR_SIZE=SF, pdl_level=level
        ](out, a, b, a_sf, b_sf, Float32(1.0), _as_compute_fn(add_resid), ctx)
    assert_equal(status, DISPATCH_HIT)


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
    print(t"block-scaled epilogue PDL: M={m} N={N} K={K} store={store_closure}")

    var a_host = ctx.enqueue_create_host_buffer[fp8](m * K)
    var b_host = ctx.enqueue_create_host_buffer[fp8](N * K)
    var r_host = ctx.enqueue_create_host_buffer[bf16](m * N)
    ctx.synchronize()
    rand(a_host.unsafe_ptr(), m * K)
    rand(b_host.unsafe_ptr(), N * K)
    rand(r_host.unsafe_ptr(), m * N, min=-1.0, max=1.0)

    var a_sf_shape = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        Idx[ceildiv(K, SF * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_sf_shape = row_major(
        Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
        Idx[ceildiv(K, SF * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var a_sf_host = ctx.enqueue_create_host_buffer[sf_type](
        a_sf_shape.product()
    )
    var b_sf_host = ctx.enqueue_create_host_buffer[sf_type](
        b_sf_shape.product()
    )
    ctx.synchronize()
    # Scales are 1 inside the problem and 0 in the padding.
    for i in range(a_sf_shape.product()):
        a_sf_host[i] = 0
    for i in range(b_sf_shape.product()):
        b_sf_host[i] = 0
    var a_sf_t = TileTensor(a_sf_host, a_sf_shape)
    var b_sf_t = TileTensor(b_sf_host, b_sf_shape)
    for i in range(m):
        for k in range(0, K, SF):
            set_scale_factor[SF_VECTOR_SIZE=SF](
                a_sf_t, i, k, Scalar[sf_type](1)
            )
    for i in range(N):
        for k in range(0, K, SF):
            set_scale_factor[SF_VECTOR_SIZE=SF](
                b_sf_t, i, k, Scalar[sf_type](1)
            )

    var a_dev = ctx.enqueue_create_buffer[fp8](m * K)
    var b_dev = ctx.enqueue_create_buffer[fp8](N * K)
    var a_sf_dev = ctx.enqueue_create_buffer[sf_type](a_sf_shape.product())
    var b_sf_dev = ctx.enqueue_create_buffer[sf_type](b_sf_shape.product())
    var r_src = ctx.enqueue_create_buffer[bf16](m * N)
    var gate_src = ctx.enqueue_create_buffer[bf16](1)
    gate_src.enqueue_fill(Scalar[bf16](1.0))
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_sf_dev, a_sf_host)
    ctx.enqueue_copy(b_sf_dev, b_sf_host)
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
        run_matmul[N, K, store_closure, PDLLevel.OVERLAP_AT_BEGINNING](
            ctx,
            m,
            out_devs[i],
            shape_dev,
            a_dev,
            b_dev,
            a_sf_dev,
            b_sf_dev,
            resid_devs[i],
            gate_devs[i],
        )
    ctx.synchronize()

    run_matmul[N, K, store_closure, PDLLevel.OFF](
        ctx,
        m,
        ref_dev,
        shape_dev,
        a_dev,
        b_dev,
        a_sf_dev,
        b_sf_dev,
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
                " outputs differing from the PDL-off matmul",
            )
    print("    PASS")


def main() raises:
    with DeviceContext() as ctx:
        # M = 1 takes the small-BN kernel, M = 1024 the standard one.
        run_case[2048, 1024, False](ctx, 1)
        run_case[2048, 1024, True](ctx, 1)
        run_case[2048, 1024, False](ctx, 1024)
        run_case[2048, 1024, True](ctx, 1024)
