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
"""PDL correctness for the epilogue closures of the SM100 conv2d kernels (B200).

A closure can read tensors that the predecessor grid writes, such as a bias
and a residual. The closure runs in the epilogue warps, which do not take the
load warps' wait, and the conv releases its dependents before its own wait, so
it can be resident while its predecessor still runs.

`late_producer` releases its dependents at once, then sleeps before it writes
the bias and the residual that the closure reads. Both buffers start as NaN, so
a closure that reads one before the producer has finished stores NaN. Each
output must match the same conv and closure run after the producer has
finished, bit for bit.

The kernels are launched directly, with a filter they take as it is, so that
the producer is the grid right before each launch: the column, 96-channel, 1x5
and patch-embed kernels.

This test passes on the current `ptxas` without the epilogue wait: `ptxas`
keeps the closure loads below the barrier chain that the load warps' wait
starts. It guards the `wait_on_dependent_grids` contract, not a failure seen
today, and it fails when the load warps' waits are removed.
"""

from std.math import nan
from std.random import rand
from std.sys.intrinsics import llvm_intrinsic

from layout import Coord, Idx, TileTensor, row_major
from linalg.utils import no_epilogue_fn
from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.grid_controls import launch_dependent_grids
from nn.conv.gpu.nvidia.sm100.conv2d_column_fprop import conv2d_column_fprop
from nn.conv.gpu.nvidia.sm100.conv2d_f96_fprop import conv2d_f96_fprop
from nn.conv.gpu.nvidia.sm100.conv2d_k1x5_fprop import conv2d_k1x5_fprop
from nn.conv.gpu.nvidia.sm100.conv2d_patch_fprop import (
    Conv2dPatchFprop,
    conv2d_patch_fprop,
)
from nn.conv.gpu.nvidia.sm100.conv_config import Conv2dProblemShape

comptime COLUMN = 0
comptime F96 = 1
comptime K1X5 = 2
comptime PATCH = 3


def late_producer[
    dt: DType
](
    resid_src: MutPointer[Scalar[dt], MutAnyOrigin],
    resid_dst: MutPointer[Scalar[dt], MutAnyOrigin],
    resid_len: Int32,
    bias_src: MutPointer[Scalar[dt], MutAnyOrigin],
    bias_dst: MutPointer[Scalar[dt], MutAnyOrigin],
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


def run_conv[
    dt: DType, kernel: Int, CIN: Int, COUT: Int, R: Int, S: Int, PAD: Int
](
    ctx: DeviceContext,
    h: Int,
    w: Int,
    mut act_dev: DeviceBuffer[dt],
    filter_dev: DeviceBuffer[dt],
    mut out_dev: DeviceBuffer[dt],
    resid_dev: DeviceBuffer[dt],
    bias_dev: DeviceBuffer[dt],
) raises:
    """Runs `out = conv(act, filter) + bias[c] + resid` on one kernel."""
    comptime stride = 14 if kernel == PATCH else 1
    var hout = (h + 2 * PAD - R) // stride + 1
    var wout = (w + 2 * PAD - S) // stride + 1
    var act = TileTensor(act_dev, row_major(1, h, w, Idx[CIN]))
    var out = TileTensor(out_dev, row_major(1, hout, wout, Idx[COUT]))
    var resid = TileTensor(
        resid_dev, row_major(1, hout, wout, Idx[COUT])
    ).as_imm()
    var bias = TileTensor(bias_dev, row_major(Idx[COUT])).as_imm()

    def add_bias_resid[
        dtype: DType, width: SIMDLength, alignment: Int = 1
    ](idx: Coord, val: SIMD[dtype, width]) {var resid, var bias} -> SIMD[
        dtype, width
    ]:
        var b = bias.load[width=width]((idx[3],)).cast[dtype]()
        return val + b + resid.load[width=width](idx).cast[dtype]()

    var problem = Conv2dProblemShape(
        batch=1,
        in_height=h,
        in_width=w,
        in_channels=CIN,
        out_channels=COUT,
        filter_h=R,
        filter_w=S,
        pad_h=PAD,
        pad_w=PAD,
    )
    comptime if kernel == PATCH:
        comptime patch = Conv2dPatchFprop[64, CIN, R, S]
        var packed = TileTensor(
            filter_dev, row_major(Idx[2 * COUT], Idx[patch.K])
        )
        conv2d_patch_fprop[64, CIN, R, S, False, True](
            out, act, packed, ctx, no_epilogue_fn, add_bias_resid
        )
    else:
        var filt = TileTensor(
            filter_dev, row_major(Idx[COUT], Idx[R], Idx[S], Idx[CIN])
        )
        comptime if kernel == COLUMN:
            conv2d_column_fprop[dt, has_compute_fn=True](
                out, act, filt, problem, ctx, compute_fn=add_bias_resid
            )
        elif kernel == F96:
            conv2d_f96_fprop[dt, has_compute_fn=True, nhwc_coords=True](
                out, act, filt, problem, ctx, compute_fn=add_bias_resid
            )
        else:
            conv2d_k1x5_fprop[dt, has_compute_fn=True, nhwc_coords=True](
                out, act, filt, problem, ctx, compute_fn=add_bias_resid
            )


def run_case[
    dt: DType, kernel: Int, CIN: Int, COUT: Int, R: Int, S: Int, PAD: Int
](ctx: DeviceContext, h: Int, w: Int, n_filt: Int, name: StringLiteral) raises:
    """Checks one kernel behind a late producer.

    The launches of all iterations go out with no host synchronization, so
    the conv's own waits are the only ordering. The first launch of a kernel
    loads its module and does not start early, so it takes several iterations.
    """
    comptime iters = 4
    comptime delay_us = 1000
    comptime stride = 14 if kernel == PATCH else 1
    var hout = (h + 2 * PAD - R) // stride + 1
    var wout = (w + 2 * PAD - S) // stride + 1
    print(t"conv epilogue PDL: {name} {h}x{w}x{CIN} -> {COUT}")

    var n_act = h * w * CIN
    var n_out = hout * wout * COUT

    var act_host = ctx.enqueue_create_host_buffer[dt](n_act)
    var filt_host = ctx.enqueue_create_host_buffer[dt](n_filt)
    var resid_host = ctx.enqueue_create_host_buffer[dt](n_out)
    var bias_host = ctx.enqueue_create_host_buffer[dt](COUT)
    ctx.synchronize()
    rand(act_host.unsafe_ptr(), n_act, min=-1.0, max=1.0)
    rand(filt_host.unsafe_ptr(), n_filt, min=-0.1, max=0.1)
    rand(resid_host.unsafe_ptr(), n_out, min=-1.0, max=1.0)
    rand(bias_host.unsafe_ptr(), COUT, min=-1.0, max=1.0)

    var act_dev = ctx.enqueue_create_buffer[dt](n_act)
    var filt_dev = ctx.enqueue_create_buffer[dt](n_filt)
    var resid_src = ctx.enqueue_create_buffer[dt](n_out)
    var bias_src = ctx.enqueue_create_buffer[dt](COUT)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(filt_dev, filt_host)
    ctx.enqueue_copy(resid_src, resid_host)
    ctx.enqueue_copy(bias_src, bias_host)

    var ref_dev = ctx.enqueue_create_buffer[dt](n_out)
    ref_dev.enqueue_fill(nan[dt]())
    var resid_devs = List[DeviceBuffer[dt]]()
    var bias_devs = List[DeviceBuffer[dt]]()
    var out_devs = List[DeviceBuffer[dt]]()
    for _ in range(iters):
        resid_devs.append(ctx.enqueue_create_buffer[dt](n_out))
        bias_devs.append(ctx.enqueue_create_buffer[dt](COUT))
        out_devs.append(ctx.enqueue_create_buffer[dt](n_out))
        resid_devs[len(resid_devs) - 1].enqueue_fill(nan[dt]())
        bias_devs[len(bias_devs) - 1].enqueue_fill(nan[dt]())
        out_devs[len(out_devs) - 1].enqueue_fill(nan[dt]())
    ctx.synchronize()

    for i in range(iters):
        ctx.enqueue_function[late_producer[dt]](
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
        run_conv[dt, kernel, CIN, COUT, R, S, PAD](
            ctx,
            h,
            w,
            act_dev,
            filt_dev,
            out_devs[i],
            resid_devs[i],
            bias_devs[i],
        )
    ctx.synchronize()

    # The reference runs once the sources are in place, so nothing overlaps.
    run_conv[dt, kernel, CIN, COUT, R, S, PAD](
        ctx, h, w, act_dev, filt_dev, ref_dev, resid_src, bias_src
    )
    ctx.synchronize()

    var want = ctx.enqueue_create_host_buffer[dt](n_out)
    var got = ctx.enqueue_create_host_buffer[dt](n_out)
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
    comptime bf16 = DType.bfloat16
    with DeviceContext() as ctx:
        # The 240x416 image of the 96 -> 96 channel conv.
        run_case[bf16, COLUMN, 96, 96, 3, 3, 1](
            ctx, 240, 416, 96 * 3 * 3 * 96, "column"
        )
        run_case[bf16, F96, 192, 96, 3, 3, 1](
            ctx, 120, 208, 96 * 3 * 3 * 192, "f96"
        )
        run_case[bf16, K1X5, 512, 1024, 1, 5, 0](
            ctx, 1, 550, 1024 * 5 * 512, "k1x5"
        )
        # Packed filter: the even and odd matrices, `[2 * F, K]`.
        comptime patch = Conv2dPatchFprop[64, 3, 14, 14]
        run_case[DType.float32, PATCH, 3, 1152, 14, 14, 0](
            ctx, 896, 896, 2 * 1152 * patch.K, "patch"
        )
