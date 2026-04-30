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

Driven by `kbench` via a YAML shape sweep; one CSV row per `(impl, shape)`.
`impl` selects among `naive`, `im2col`, and `cudnn`. If the im2col
dispatcher declines a shape (R=S=1, K<16, N<16, non-bf16, grouped, etc.),
the bench raises an `Error` so kbench records the (impl, shape) as
failed rather than timing a no-op that misleadingly looks fastest.

Usage (standalone):
    bazel test --config=remote-b200 \\
        //max/kernels/benchmarks:gpu/nn/bench_conv2d.mojo.run

Usage (kbench):
    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- \\
        max/kernels/benchmarks/gpu/nn/bench_conv2d.yaml \\
        --target-accelerator cuda:b200 -c
"""

from std.math import ceildiv
from std.random import rand
from std.sys import get_defined_dtype, get_defined_int, get_defined_string

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import (
    UNKNOWN_VALUE,
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
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
    impl: StaticString,
](
    ctx: DeviceContext,
    mut b: Bench,
    label: String,
    verify: Bool,
    pad_h: Int,
    pad_w: Int,
    stride_h: Int,
    stride_w: Int,
) raises:
    var h_out = (in_height + 2 * pad_h - filter_r) // stride_h + 1
    var w_out = (in_width + 2 * pad_w - filter_s) // stride_w + 1

    comptime input_layout = Layout.row_major(
        batch, in_height, in_width, in_channels
    )
    comptime filter_rscf_layout = Layout.row_major(
        filter_r, filter_s, in_channels, out_channels
    )
    # Output spatial dims depend on runtime pad / stride, so leave them as
    # UNKNOWN_VALUE in the static layout and supply concrete sizes via a
    # RuntimeLayout below.
    comptime output_layout = Layout.row_major(
        batch, UNKNOWN_VALUE, UNKNOWN_VALUE, out_channels
    )

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_rscf_layout.size())
    var output_size = batch * h_out * w_out * out_channels

    var flops = compute_conv2d_flops(
        batch,
        h_out,
        w_out,
        out_channels,
        in_channels,
        filter_r,
        filter_s,
    )

    var bench_input_id = String(
        label,
        "/impl=",
        impl,
        "/dtype=",
        dtype,
        "/NHWC=(",
        batch,
        "x",
        in_height,
        "x",
        in_width,
        "x",
        in_channels,
        ")/filter=(",
        filter_r,
        "x",
        filter_s,
        ")/Cout=",
        out_channels,
        "/pad=(",
        pad_h,
        "x",
        pad_w,
        ")/stride=(",
        stride_h,
        "x",
        stride_w,
        ")",
    )

    var input_host = List(length=input_size, fill=Scalar[dtype](0))
    var filter_rscf_host = List(length=filter_size, fill=Scalar[dtype](0))
    var filter_fcrs_host = List(length=filter_size, fill=Scalar[dtype](0))
    rand[dtype](input_host)
    rand[dtype](filter_rscf_host)

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
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_ref_dev = ctx.enqueue_create_buffer[dtype](
        output_size if verify else 1
    )

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_rscf_dev, filter_rscf_host)
    ctx.enqueue_copy(filter_fcrs_dev, filter_fcrs_host)
    ctx.synchronize()

    var input_buf = LayoutTensor[dtype, input_layout](input_dev.unsafe_ptr())
    var filter_rscf_buf = LayoutTensor[dtype, filter_rscf_layout](
        filter_rscf_dev.unsafe_ptr()
    )
    var output_runtime_layout = RuntimeLayout[output_layout].row_major(
        IndexList[4](batch, h_out, w_out, out_channels)
    )
    var output_buf = LayoutTensor[dtype, output_layout](
        output_dev.unsafe_ptr(), output_runtime_layout
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
    var output_tt = TileTensor(
        output_dev.unsafe_ptr(),
        row_major(Coord(IndexList[4](batch, h_out, w_out, out_channels))),
    )

    var stride_idx = IndexList[2](stride_h, stride_w)
    var dilation_idx = IndexList[2](1, 1)
    var pad_idx = IndexList[2](pad_h, pad_w)

    comptime block_size = 16

    # Probe the im2col dispatcher once. On decline, raise so kbench logs
    # this (impl, shape) as failed instead of timing a no-op (which would
    # otherwise look fastest in the CSV).
    comptime if impl == "im2col":
        var accepted = dispatch_im2col_matmul_conv2d[
            dtype, dtype, dtype, filter_is_fcrs=False
        ](
            input_tt,
            filter_rscf_tt,
            output_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
        ctx.synchronize()
        if not accepted:
            raise Error(
                "dispatch_im2col_matmul_conv2d declined: " + bench_input_id
            )

    comptime if impl == "im2col":

        @parameter
        @always_inline
        @__copy_capture(input_tt, filter_rscf_tt, output_tt)
        def im2col_bench(mut bencher: Bencher) raises:
            @parameter
            @always_inline
            def kernel(ctx: DeviceContext) raises:
                _ = dispatch_im2col_matmul_conv2d[
                    dtype, dtype, dtype, filter_is_fcrs=False
                ](
                    input_tt,
                    filter_rscf_tt,
                    output_tt,
                    stride_idx,
                    dilation_idx,
                    pad_idx,
                    1,
                    ctx,
                )

            bencher.iter_custom[kernel](ctx)

        b.bench_function[im2col_bench](
            BenchId("conv2d_im2col", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )
    elif impl == "cudnn":

        @parameter
        @always_inline
        @__copy_capture(input_tt, filter_fcrs_tt, output_tt)
        def cudnn_bench(mut bencher: Bencher) raises:
            @parameter
            @always_inline
            def kernel(ctx: DeviceContext) raises:
                conv_cudnn[dtype, dtype, dtype](
                    input_tt,
                    filter_fcrs_tt,
                    output_tt,
                    stride_idx,
                    dilation_idx,
                    pad_idx,
                    1,
                    ctx,
                )

            bencher.iter_custom[kernel](ctx)

        b.bench_function[cudnn_bench](
            BenchId("conv2d_cudnn", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )
    else:
        # Naive Mojo NHWC-RSCF kernel.
        comptime naive_kernel = conv2d_gpu_naive_nhwc_rscf[
            input_layout,
            filter_rscf_layout,
            output_layout,
            dtype,
            dtype,
            dtype,
            block_size,
            None,
        ]
        var grid_dim_x = ceildiv(w_out * h_out, block_size)
        var grid_dim_y = batch

        @parameter
        @always_inline
        @__copy_capture(input_buf, filter_rscf_buf, output_buf)
        def naive_bench(mut bencher: Bencher) raises:
            @parameter
            @always_inline
            def kernel(ctx: DeviceContext) raises:
                ctx.enqueue_function[naive_kernel, naive_kernel](
                    input_buf,
                    filter_rscf_buf,
                    output_buf,
                    stride_idx,
                    dilation_idx,
                    pad_idx,
                    1,
                    grid_dim=(grid_dim_x, grid_dim_y, 1),
                    block_dim=(block_size, block_size, 1),
                )

            bencher.iter_custom[kernel](ctx)

        b.bench_function[naive_bench](
            BenchId("conv2d_naive", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )

    # Optional correctness cross-check against cuDNN.
    if verify:
        var output_ref_tt = TileTensor(
            output_ref_dev.unsafe_ptr(),
            row_major(Coord(IndexList[4](batch, h_out, w_out, out_channels))),
        )
        conv_cudnn[dtype, dtype, dtype](
            input_tt,
            filter_fcrs_tt,
            output_ref_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
        ctx.synchronize()
        var output_host = List(length=output_size, fill=Scalar[dtype](0))
        var output_ref_host = List(length=output_size, fill=Scalar[dtype](0))
        ctx.enqueue_copy(output_host, output_dev)
        ctx.enqueue_copy(output_ref_host, output_ref_dev)
        ctx.synchronize()
        var max_diff: Float32 = 0.0
        for i in range(output_size):
            var a = output_host[i].cast[DType.float32]()
            var c = output_ref_host[i].cast[DType.float32]()
            var d = abs(a - c)
            if d > max_diff:
                max_diff = d
        print("verify max |", impl, " - cuDNN| = ", max_diff, sep="")
        _ = output_host^
        _ = output_ref_host^

    _ = input_dev^
    _ = filter_rscf_dev^
    _ = filter_fcrs_dev^
    _ = output_dev^
    _ = output_ref_dev^
    _ = filter_fcrs_host^
    _ = filter_rscf_host^
    _ = input_host^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime N = get_defined_int["N", 1]()
    comptime H = get_defined_int["H", 240]()
    comptime W = get_defined_int["W", 416]()
    comptime C_in = get_defined_int["C_in", 96]()
    comptime C_out = get_defined_int["C_out", 96]()
    comptime R = get_defined_int["R", 3]()
    comptime S = get_defined_int["S", 3]()
    comptime impl = get_defined_string["impl", "naive"]()

    var label = arg_parse("label", String("conv2d"))
    var verify = arg_parse("verify", False)
    # pad / stride flow into the kernels as runtime IndexLists, so reading
    # them via arg_parse instead of get_defined_int avoids spinning up a
    # fresh compiled binary per (pad, stride) combination.
    var pad_h = arg_parse("pad_h", 1)
    var pad_w = arg_parse("pad_w", 1)
    var stride_h = arg_parse("stride_h", 1)
    var stride_w = arg_parse("stride_w", 1)
    var warmup_iters = arg_parse("warmup_iters", 2)
    var max_iters = arg_parse("max_iters", 20)
    var max_runtime_secs = arg_parse("max_runtime_secs", 3.0)

    var m = Bench(
        BenchConfig(
            num_repetitions=1,
            num_warmup_iters=warmup_iters,
            max_iters=max_iters,
            max_runtime_secs=max_runtime_secs,
        )
    )
    with DeviceContext() as ctx:
        bench_conv2d[
            dtype,
            N,
            H,
            W,
            C_in,
            C_out,
            R,
            S,
            impl,
        ](ctx, m, label, verify, pad_h, pad_w, stride_h, stride_w)
    m.dump_report()
