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
"""Benchmark the native Mojo 2D conv paths against cuDNN.

Compares `conv2d_gpu_naive_nhwc_rscf`, `dispatch_im2col_matmul_conv2d`,
and cuDNN on WAN VAE 2-D conv shapes where the SM100 fast path declines
on non-128-aligned channels.

Usage:
    bazel test --config=remote-b200 \
        //max/kernels/benchmarks:gpu/nn/bench_conv2d.mojo.run
"""

from std.math import ceildiv
from std.random import rand

from std.gpu.host import DeviceContext
from layout import Coord, Idx, Layout, LayoutTensor, TileTensor, row_major
from nn.conv.conv import conv2d_gpu_naive_nhwc_rscf, conv_cudnn
from nn.conv.gpu.im2col_matmul_2d import dispatch_im2col_matmul_conv2d

from std.utils.index import IndexList


def compute_conv2d_flops(
    batch: Int,
    h_out: Int,
    w_out: Int,
    c_out: Int,
    c_in: Int,
    r: Int,
    s: Int,
) -> Int:
    return 2 * batch * h_out * w_out * c_out * c_in * r * s


def bench_conv2d[
    dtype: DType,
    batch: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    out_channels: Int,
    filter_r: Int,
    filter_s: Int,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int = 1,
    stride_w: Int = 1,
](
    ctx: DeviceContext,
    label: StringLiteral,
    num_iters: Int = 20,
    warmup_iters: Int = 3,
) raises:
    comptime h_out = (in_height + 2 * pad_h - filter_r) // stride_h + 1
    comptime w_out = (in_width + 2 * pad_w - filter_s) // stride_w + 1

    comptime input_layout = Layout.row_major(
        batch, in_height, in_width, in_channels
    )
    comptime filter_rscf_layout = Layout.row_major(
        filter_r, filter_s, in_channels, out_channels
    )
    comptime filter_fcrs_layout = Layout.row_major(
        out_channels, in_channels, filter_r, filter_s
    )
    comptime output_layout = Layout.row_major(batch, h_out, w_out, out_channels)

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_rscf_layout.size())
    var output_size = comptime (output_layout.size())

    var flops = compute_conv2d_flops(
        batch,
        h_out,
        w_out,
        out_channels,
        in_channels,
        filter_r,
        filter_s,
    )

    print(
        label,
        ": NHWC=(",
        batch,
        ",",
        in_height,
        ",",
        in_width,
        ",",
        in_channels,
        ") filter=(",
        filter_r,
        "x",
        filter_s,
        ") C_out=",
        out_channels,
        " pad=(",
        pad_h,
        ",",
        pad_w,
        ") stride=(",
        stride_h,
        ",",
        stride_w,
        ") GFLOPS=",
        Float64(flops) / 1e9,
        sep="",
    )

    var input_host = alloc[Scalar[dtype]](input_size)
    var filter_rscf_host = alloc[Scalar[dtype]](filter_size)
    var filter_fcrs_host = alloc[Scalar[dtype]](filter_size)

    rand[dtype](input_host, input_size)
    rand[dtype](filter_rscf_host, filter_size)

    # RSCF [R,S,C,F] -> FCRS [F,C,R,S] for cuDNN.
    for f in range(out_channels):
        for c in range(in_channels):
            for r in range(filter_r):
                for s in range(filter_s):
                    var rscf_idx = (
                        r * filter_s * in_channels * out_channels
                        + s * in_channels * out_channels
                        + c * out_channels
                        + f
                    )
                    var fcrs_idx = (
                        f * in_channels * filter_r * filter_s
                        + c * filter_r * filter_s
                        + r * filter_s
                        + s
                    )
                    filter_fcrs_host[fcrs_idx] = filter_rscf_host[rscf_idx]

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_rscf_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_fcrs_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_naive_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_im2col_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_cudnn_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_rscf_dev, filter_rscf_host)
    ctx.enqueue_copy(filter_fcrs_dev, filter_fcrs_host)
    ctx.synchronize()

    var input_buf = LayoutTensor[dtype, input_layout](input_dev.unsafe_ptr())
    var filter_rscf_buf = LayoutTensor[dtype, filter_rscf_layout](
        filter_rscf_dev.unsafe_ptr()
    )
    var output_naive_buf = LayoutTensor[dtype, output_layout](
        output_naive_dev.unsafe_ptr()
    )

    var input_tt = TileTensor(
        input_dev.unsafe_ptr(),
        row_major(Coord(IndexList[4](batch, in_height, in_width, in_channels))),
    )
    var filter_rscf_tt = TileTensor(
        filter_rscf_dev.unsafe_ptr(),
        row_major(
            Coord(IndexList[4](filter_r, filter_s, in_channels, out_channels))
        ),
    )
    var filter_fcrs_tt = TileTensor(
        filter_fcrs_dev.unsafe_ptr(),
        row_major(
            Coord(IndexList[4](out_channels, in_channels, filter_r, filter_s))
        ),
    )
    var output_im2col_tt = TileTensor(
        output_im2col_dev.unsafe_ptr(),
        row_major(Coord(IndexList[4](batch, h_out, w_out, out_channels))),
    )
    var output_cudnn_tt = TileTensor(
        output_cudnn_dev.unsafe_ptr(),
        row_major(Coord(IndexList[4](batch, h_out, w_out, out_channels))),
    )

    var stride_idx = IndexList[2](stride_h, stride_w)
    var dilation_idx = IndexList[2](1, 1)
    var pad_idx = IndexList[2](pad_h, pad_w)

    comptime block_size = 16

    # Warmup all paths.
    for _ in range(warmup_iters):
        var grid_dim_x = ceildiv(w_out * h_out, block_size)
        var grid_dim_y = batch
        ctx.enqueue_function[
            conv2d_gpu_naive_nhwc_rscf[
                input_layout,
                filter_rscf_layout,
                output_layout,
                dtype,
                dtype,
                dtype,
                block_size,
                None,
            ],
            conv2d_gpu_naive_nhwc_rscf[
                input_layout,
                filter_rscf_layout,
                output_layout,
                dtype,
                dtype,
                dtype,
                block_size,
                None,
            ],
        ](
            input_buf,
            filter_rscf_buf,
            output_naive_buf,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            grid_dim=(grid_dim_x, grid_dim_y, 1),
            block_dim=(block_size, block_size, 1),
        )
        _ = dispatch_im2col_matmul_conv2d[
            dtype, dtype, dtype, filter_is_fcrs=False
        ](
            input_tt,
            filter_rscf_tt,
            output_im2col_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
        conv_cudnn[dtype, dtype, dtype](
            input_tt,
            filter_fcrs_tt,
            output_cudnn_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
    ctx.synchronize()

    # Timed: naive.
    @parameter
    @__copy_capture(input_buf, filter_rscf_buf, output_naive_buf)
    def naive_bench() raises:
        var grid_dim_x = ceildiv(w_out * h_out, block_size)
        var grid_dim_y = batch
        ctx.enqueue_function[
            conv2d_gpu_naive_nhwc_rscf[
                input_layout,
                filter_rscf_layout,
                output_layout,
                dtype,
                dtype,
                dtype,
                block_size,
                None,
            ],
            conv2d_gpu_naive_nhwc_rscf[
                input_layout,
                filter_rscf_layout,
                output_layout,
                dtype,
                dtype,
                dtype,
                block_size,
                None,
            ],
        ](
            input_buf,
            filter_rscf_buf,
            output_naive_buf,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            grid_dim=(grid_dim_x, grid_dim_y, 1),
            block_dim=(block_size, block_size, 1),
        )

    var naive_ns = ctx.execution_time[naive_bench](num_iters)
    var naive_ms = Float64(naive_ns) / 1e6 / Float64(num_iters)
    var naive_tflops = Float64(flops) / (naive_ms / 1000) / 1e12

    # Probe once to see if the im2col dispatcher accepts this shape;
    # a `False` return means we should skip the timed loop below
    # (otherwise we'd be measuring a no-op return).
    var im2col_handled = dispatch_im2col_matmul_conv2d[
        dtype, dtype, dtype, filter_is_fcrs=False
    ](
        input_tt,
        filter_rscf_tt,
        output_im2col_tt,
        stride_idx,
        dilation_idx,
        pad_idx,
        1,
        ctx,
    )
    ctx.synchronize()

    # Timed: im2col+matmul (skipped if dispatcher declined).
    var im2col_ms: Float64 = 0.0
    var im2col_tflops: Float64 = 0.0
    if im2col_handled:

        @parameter
        @__copy_capture(input_tt, filter_rscf_tt, output_im2col_tt)
        def im2col_bench() raises:
            _ = dispatch_im2col_matmul_conv2d[
                dtype, dtype, dtype, filter_is_fcrs=False
            ](
                input_tt,
                filter_rscf_tt,
                output_im2col_tt,
                stride_idx,
                dilation_idx,
                pad_idx,
                1,
                ctx,
            )

        var im2col_ns = ctx.execution_time[im2col_bench](num_iters)
        im2col_ms = Float64(im2col_ns) / 1e6 / Float64(num_iters)
        im2col_tflops = Float64(flops) / (im2col_ms / 1000) / 1e12

    # Timed: cuDNN.
    @parameter
    @__copy_capture(input_tt, filter_fcrs_tt, output_cudnn_tt)
    def cudnn_bench() raises:
        conv_cudnn[dtype, dtype, dtype](
            input_tt,
            filter_fcrs_tt,
            output_cudnn_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )

    var cudnn_ns = ctx.execution_time[cudnn_bench](num_iters)
    var cudnn_ms = Float64(cudnn_ns) / 1e6 / Float64(num_iters)
    var cudnn_tflops = Float64(flops) / (cudnn_ms / 1000) / 1e12

    # Cross-validate im2col vs cuDNN, but only when im2col actually ran.
    var max_diff: Float32 = 0.0
    if im2col_handled:
        var output_im2col_host = alloc[Scalar[dtype]](output_size)
        var output_cudnn_host = alloc[Scalar[dtype]](output_size)
        ctx.enqueue_copy(output_im2col_host, output_im2col_dev)
        ctx.enqueue_copy(output_cudnn_host, output_cudnn_dev)
        ctx.synchronize()
        for i in range(output_size):
            var a = output_im2col_host[i].cast[DType.float32]()
            var b = output_cudnn_host[i].cast[DType.float32]()
            var d = abs(a - b)
            if d > max_diff:
                max_diff = d
        output_im2col_host.free()
        output_cudnn_host.free()

    print(
        "  Naive Mojo : ", naive_ms, " ms  (", naive_tflops, " TFLOPS)", sep=""
    )
    if im2col_handled:
        print(
            "  im2col+gemm: ",
            im2col_ms,
            " ms  (",
            im2col_tflops,
            " TFLOPS)",
            sep="",
        )
    else:
        print(
            "  im2col+gemm: (dispatcher declined; R=S=1 / K<16 / N<16 /"
            " non-bf16)"
        )
    print(
        "  cuDNN      : ", cudnn_ms, " ms  (", cudnn_tflops, " TFLOPS)", sep=""
    )
    print(
        "  naive / cuDNN : ",
        naive_ms / cudnn_ms,
        "x  (lower is better)",
        sep="",
    )
    if im2col_handled:
        print(
            "  im2col / cuDNN: ",
            im2col_ms / cudnn_ms,
            "x  (lower is better)",
            sep="",
        )
        print("  max |im2col - cuDNN|: ", max_diff, sep="")
    print()

    input_host.free()
    filter_rscf_host.free()
    filter_fcrs_host.free()
    _ = input_dev^
    _ = filter_rscf_dev^
    _ = filter_fcrs_dev^
    _ = output_naive_dev^
    _ = output_im2col_dev^
    _ = output_cudnn_dev^


def main() raises:
    with DeviceContext() as ctx:
        print("=" * 70)
        print("WAN VAE 2D CONV BENCHMARK: naive vs im2col+matmul vs cuDNN")
        print("=" * 70)
        print()

        # WAN level-3 residual: the hottest Mojo VAE 2-D shape per nsys
        # (3x3, 96->96 at 240x416 full output spatial). Dominant
        # contributor to the ~3.2s Mojo VAE decode before Phase F.
        bench_conv2d[
            DType.bfloat16,
            batch=1,
            in_height=240,
            in_width=416,
            in_channels=96,
            out_channels=96,
            filter_r=3,
            filter_s=3,
            pad_h=1,
            pad_w=1,
        ](ctx, label="WAN_level3_res_96to96")

        # WAN level-2 -> level-3 transition (3x3, 192->96 at 240x416).
        # Non-aligned C_out -> declines SM100, hits Phase F.
        bench_conv2d[
            DType.bfloat16,
            batch=1,
            in_height=240,
            in_width=416,
            in_channels=192,
            out_channels=96,
            filter_r=3,
            filter_s=3,
            pad_h=1,
            pad_w=1,
        ](ctx, label="WAN_level3_transition_192to96")

        # WAN level-2 residual upsample 2D (3x3, 192->192 at 120x208).
        bench_conv2d[
            DType.bfloat16,
            batch=1,
            in_height=120,
            in_width=208,
            in_channels=192,
            out_channels=192,
            filter_r=3,
            filter_s=3,
            pad_h=1,
            pad_w=1,
        ](ctx, label="WAN_level2_res_192to192")

        # conv_out-ish shape (3x3, 96->3 at 240x416). C_out=3 is the
        # final RGB reduction; K>=16 and R*S > 1 so im2col qualifies.
        bench_conv2d[
            DType.bfloat16,
            batch=1,
            in_height=240,
            in_width=416,
            in_channels=96,
            out_channels=3,
            filter_r=3,
            filter_s=3,
            pad_h=1,
            pad_w=1,
        ](ctx, label="WAN_conv_out_96to3")

        print("=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
