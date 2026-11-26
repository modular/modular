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
"""PDL correctness for the epilogue closures of the SM100 conv3d kernels (B200).

A closure can read tensors that the predecessor grid writes, such as a bias
and a residual. The closure runs in the epilogue warps, which do not take the
load warps' wait, and the conv releases its dependents before its own wait, so
it can be resident while its predecessor still runs.

`late_producer` releases its dependents at once, then sleeps before it writes
the bias and the residual that the closure reads. Both buffers start as NaN, so
a closure that reads one before the producer has finished stores NaN. Each
output must match the same conv and closure run after the producer has
finished, bit for bit.

The kernels are launched on a packed filter, so that the producer is the grid
right before each launch: the column, window and depth kernels.

This test passes on the current `ptxas` without the epilogue wait: `ptxas`
keeps the closure loads below the barrier chain that the load warps' wait
starts. It guards the `wait_on_dependent_grids` contract, not a failure seen
today, and it fails when the load warps' waits are removed.
"""

from std.math import nan
from std.random import rand
from std.sys.intrinsics import llvm_intrinsic

from layout import Coord, Idx, TileTensor, row_major
from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import launch_dependent_grids
from nn.conv.conv_utils import no_conv_simd_epilogue_fn
from nn.conv.gpu.nvidia.sm100.conv3d_col_umma import (
    launch_conv3d_col_umma_packed,
)
from nn.conv.gpu.nvidia.sm100.conv3d_tap_umma import (
    launch_conv3d_depth_umma_packed,
    launch_conv3d_window_umma_packed,
)

comptime bf16 = DType.bfloat16
comptime COLUMN = 0
comptime WINDOW = 1
comptime DEPTH = 2


def late_producer(
    resid_src: MutPointer[Scalar[bf16], MutAnyOrigin],
    resid_dst: MutPointer[Scalar[bf16], MutAnyOrigin],
    resid_len: Int32,
    bias_src: MutPointer[Scalar[bf16], MutAnyOrigin],
    bias_dst: MutPointer[Scalar[bf16], MutAnyOrigin],
    bias_len: Int32,
    delay_us: Int32,
):
    """Releases the dependent grid at entry, then writes the tensors late.

    The dependent can only launch once every block of this grid has released
    it, so the grid is small enough to be resident at once and strides over
    the tensors.
    """
    launch_dependent_grids()
    # nanosleep takes nanoseconds and caps at about 1 ms per call.
    var remaining = Int(delay_us)
    while remaining > 0:
        llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
        remaining -= 1000
    var stride = Int(grid_dim.x * block_dim.x)
    for i in range(global_idx.x, Int(resid_len), stride):
        resid_dst[i] = resid_src[i]
    for i in range(global_idx.x, Int(bias_len), stride):
        bias_dst[i] = bias_src[i]


def out_depth[kernel: Int](depth: Int) -> Int:
    """The output depth: the window kernel pads the depth, the others do not."""
    return depth if kernel == WINDOW else depth - 2


def run_conv[
    kernel: Int, CIN: Int, COUT: Int, H: Int, W: Int
](
    ctx: DeviceContext,
    depth: Int,
    mut act_dev: DeviceBuffer[bf16],
    packed_dev: DeviceBuffer[bf16],
    mut out_dev: DeviceBuffer[bf16],
    resid_dev: DeviceBuffer[bf16],
    bias_dev: DeviceBuffer[bf16],
) raises:
    """Runs `out = conv(act, filter) + bias[c] + resid` on one kernel."""
    var od = out_depth[kernel](depth)
    var act = TileTensor(act_dev, row_major(1, depth, H, W, Idx[CIN]))
    var out = TileTensor(out_dev, row_major(1, od, H, W, Idx[COUT]))
    var resid = TileTensor(
        resid_dev, row_major(1, od, H, W, Idx[COUT])
    ).as_imm()
    var bias = TileTensor(bias_dev, row_major(Idx[COUT])).as_imm()

    def add_bias_resid[
        dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[dtype, width]) {var resid, var bias} -> SIMD[
        dtype, width
    ]:
        var b = bias.load[width=width]((idx[4],)).cast[dtype]()
        return val + b + resid.load[width=width](idx).cast[dtype]()

    comptime if kernel == COLUMN:
        launch_conv3d_col_umma_packed[bf16, CIN, COUT, H, W, False, True](
            act,
            packed_dev,
            out,
            ctx,
            no_conv_simd_epilogue_fn,
            add_bias_resid,
        )
    elif kernel == WINDOW:
        launch_conv3d_window_umma_packed[bf16, COUT, H, W, False, True](
            act,
            packed_dev,
            out,
            ctx,
            no_conv_simd_epilogue_fn,
            add_bias_resid,
        )
    else:
        launch_conv3d_depth_umma_packed[bf16, CIN, COUT, H, W, 3, False, True](
            act,
            packed_dev,
            out,
            ctx,
            no_conv_simd_epilogue_fn,
            add_bias_resid,
        )


def run_case[
    kernel: Int, CIN: Int, COUT: Int, H: Int, W: Int
](ctx: DeviceContext, depth: Int, name: StringLiteral) raises:
    """Checks one kernel behind a late producer.

    The launches of all iterations go out with no host synchronization, so
    the conv's own waits are the only ordering. The first launch of a kernel
    loads its module and does not start early, so it takes several iterations.
    """
    comptime iters = 4
    comptime delay_us = 1000
    print(t"conv3d epilogue PDL: {name} {depth}x{H}x{W}x{CIN} -> {COUT}")

    var n_act = depth * H * W * CIN
    var n_filt = 27 * CIN * COUT
    var n_out = out_depth[kernel](depth) * H * W * COUT

    var act_host = ctx.enqueue_create_host_buffer[bf16](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[bf16](n_filt)
    var resid_host = ctx.enqueue_create_host_buffer[bf16](n_out)
    var bias_host = ctx.enqueue_create_host_buffer[bf16](COUT)
    ctx.synchronize()
    rand(act_host.unsafe_ptr(), n_act, min=-1.0, max=1.0)
    rand(filt_host.unsafe_ptr(), n_filt, min=-0.1, max=0.1)
    rand(resid_host.unsafe_ptr(), n_out, min=-1.0, max=1.0)
    rand(bias_host.unsafe_ptr(), COUT, min=-1.0, max=1.0)

    var act_dev = ctx.enqueue_create_buffer[bf16](n_act)
    # An already packed filter: any values do for a race check.
    var filt_dev = ctx.enqueue_create_buffer[bf16](n_filt)
    var resid_src = ctx.enqueue_create_buffer[bf16](n_out)
    var bias_src = ctx.enqueue_create_buffer[bf16](COUT)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(resid_src, resid_host)
    ctx.enqueue_copy(bias_src, bias_host)

    var ref_dev = ctx.enqueue_create_buffer[bf16](n_out)
    ref_dev.enqueue_fill(nan[bf16]())
    var resid_devs = List[DeviceBuffer[bf16]]()
    var bias_devs = List[DeviceBuffer[bf16]]()
    var out_devs = List[DeviceBuffer[bf16]]()
    for _ in range(iters):
        resid_devs.append(ctx.enqueue_create_buffer[bf16](n_out))
        bias_devs.append(ctx.enqueue_create_buffer[bf16](COUT))
        out_devs.append(ctx.enqueue_create_buffer[bf16](n_out))
        resid_devs[len(resid_devs) - 1].enqueue_fill(nan[bf16]())
        bias_devs[len(bias_devs) - 1].enqueue_fill(nan[bf16]())
        out_devs[len(out_devs) - 1].enqueue_fill(nan[bf16]())
    ctx.synchronize()

    for i in range(iters):
        ctx.enqueue_function[late_producer](
            resid_src,
            resid_devs[i],
            Int32(n_out),
            bias_src,
            bias_devs[i],
            Int32(COUT),
            Int32(delay_us),
            grid_dim=8,
            block_dim=256,
        )
        run_conv[kernel, CIN, COUT, H, W](
            ctx,
            depth,
            act_dev,
            filt_dev,
            out_devs[i],
            resid_devs[i],
            bias_devs[i],
        )
    ctx.synchronize()

    # The reference runs once the sources are in place, so nothing overlaps.
    run_conv[kernel, CIN, COUT, H, W](
        ctx, depth, act_dev, filt_dev, ref_dev, resid_src, bias_src
    )
    ctx.synchronize()

    var want = ctx.enqueue_create_host_buffer[bf16](n_out)
    var got = ctx.enqueue_create_host_buffer[bf16](n_out)
    ctx.enqueue_copy(want, ref_dev)
    ctx.synchronize()
    for i in range(iters):
        ctx.enqueue_copy(got, out_devs[i])
        ctx.synchronize()
        var bad = 0
        for j in range(n_out):
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
                n_out,
                " outputs differing from the unoverlapped conv",
            )
    print("    PASS")


def main() raises:
    with DeviceContext() as ctx:
        run_case[COLUMN, 384, 384, 30, 52](ctx, 4, "column")
        run_case[WINDOW, 16, 384, 30, 52](ctx, 3, "window")
        run_case[DEPTH, 192, 384, 60, 104](ctx, 3, "depth")
