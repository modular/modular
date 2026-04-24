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
"""Benchmark the naive Mojo 3D conv kernel against cuDNN.

Targets the WAN VAE shape profile (bf16, NDHWC input, QRSCF filter) so we
can track progression of the native conv3d path toward cuDNN parity.

Usage:
    bazel run --config=remote-b200 \
        //max/kernels/benchmarks/gpu/nn:bench_conv3d
"""

from std.math import ceildiv
from std.random import rand

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from nn.conv.conv import conv3d_gpu_naive_ndhwc_qrscf, conv3d_cudnn

from std.utils.index import IndexList


def compute_conv3d_flops(
    batch: Int,
    d_out: Int,
    h_out: Int,
    w_out: Int,
    c_out: Int,
    c_in: Int,
    q: Int,
    r: Int,
    s: Int,
) -> Int:
    return 2 * batch * d_out * h_out * w_out * c_out * c_in * q * r * s


def bench_conv3d[
    dtype: DType,
    batch: Int,
    in_depth: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    out_channels: Int,
    filter_q: Int,
    filter_r: Int,
    filter_s: Int,
    pad_d: Int,
    pad_h: Int,
    pad_w: Int,
    stride_d: Int = 1,
    stride_h: Int = 1,
    stride_w: Int = 1,
](
    ctx: DeviceContext,
    label: StringLiteral,
    num_iters: Int = 50,
    warmup_iters: Int = 5,
) raises:
    comptime d_out = (in_depth + 2 * pad_d - filter_q) // stride_d + 1
    comptime h_out = (in_height + 2 * pad_h - filter_r) // stride_h + 1
    comptime w_out = (in_width + 2 * pad_w - filter_s) // stride_w + 1

    comptime input_layout = Layout.row_major(
        batch, in_depth, in_height, in_width, in_channels
    )
    comptime filter_qrscf_layout = Layout.row_major(
        filter_q, filter_r, filter_s, in_channels, out_channels
    )
    comptime filter_fcqrs_layout = Layout.row_major(
        out_channels, in_channels, filter_q, filter_r, filter_s
    )
    comptime output_layout = Layout.row_major(
        batch, d_out, h_out, w_out, out_channels
    )

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_qrscf_layout.size())
    var output_size = comptime (output_layout.size())

    var flops = compute_conv3d_flops(
        batch,
        d_out,
        h_out,
        w_out,
        out_channels,
        in_channels,
        filter_q,
        filter_r,
        filter_s,
    )

    print(
        label,
        ": NDHWC=(",
        batch,
        ",",
        in_depth,
        ",",
        in_height,
        ",",
        in_width,
        ",",
        in_channels,
        ") filter=(",
        filter_q,
        "x",
        filter_r,
        "x",
        filter_s,
        ") C_out=",
        out_channels,
        " pad=(",
        pad_d,
        ",",
        pad_h,
        ",",
        pad_w,
        ") stride=(",
        stride_d,
        ",",
        stride_h,
        ",",
        stride_w,
        ") GFLOPS=",
        Float64(flops) / 1e9,
        sep="",
    )

    # Host buffers.
    var input_host = alloc[Scalar[dtype]](input_size)
    var filter_qrscf_host = alloc[Scalar[dtype]](filter_size)
    var filter_fcqrs_host = alloc[Scalar[dtype]](filter_size)
    rand[dtype](input_host, input_size)
    rand[dtype](filter_qrscf_host, filter_size)

    # Convert QRSCF [Q,R,S,C,F] -> FCQRS [F,C,Q,R,S] for cuDNN.
    for f in range(out_channels):
        for c in range(in_channels):
            for q in range(filter_q):
                for r in range(filter_r):
                    for s in range(filter_s):
                        var qrscf_idx = (
                            q * filter_r * filter_s * in_channels * out_channels
                            + r * filter_s * in_channels * out_channels
                            + s * in_channels * out_channels
                            + c * out_channels
                            + f
                        )
                        var fcqrs_idx = (
                            f * in_channels * filter_q * filter_r * filter_s
                            + c * filter_q * filter_r * filter_s
                            + q * filter_r * filter_s
                            + r * filter_s
                            + s
                        )
                        filter_fcqrs_host[fcqrs_idx] = filter_qrscf_host[
                            qrscf_idx
                        ]

    # Device buffers.
    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_qrscf_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_fcqrs_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_naive_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_cudnn_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_qrscf_dev, filter_qrscf_host)
    ctx.enqueue_copy(filter_fcqrs_dev, filter_fcqrs_host)
    ctx.synchronize()

    # LayoutTensor views.
    var input_buf = LayoutTensor[dtype, input_layout](input_dev.unsafe_ptr())
    var filter_qrscf_buf = LayoutTensor[dtype, filter_qrscf_layout](
        filter_qrscf_dev.unsafe_ptr()
    )
    var filter_fcqrs_buf = LayoutTensor[dtype, filter_fcqrs_layout](
        filter_fcqrs_dev.unsafe_ptr()
    )
    var output_naive_buf = LayoutTensor[dtype, output_layout](
        output_naive_dev.unsafe_ptr()
    )
    var output_cudnn_buf = LayoutTensor[dtype, output_layout](
        output_cudnn_dev.unsafe_ptr()
    )

    # Naive kernel launch config.
    comptime block_size = 16
    var grid_dim_x = ceildiv(w_out * h_out, block_size)
    var grid_dim_y = ceildiv(d_out, block_size)
    var grid_dim_z = batch

    comptime naive_kernel = conv3d_gpu_naive_ndhwc_qrscf[
        input_layout,
        filter_qrscf_layout,
        output_layout,
        dtype,
        dtype,
        dtype,
        block_size,
        None,
    ]

    var stride_idx = IndexList[3](stride_d, stride_h, stride_w)
    var dilation_idx = IndexList[3](1, 1, 1)
    var pad_idx = IndexList[3](pad_d, pad_h, pad_w)

    # Warmup.
    for _ in range(warmup_iters):
        ctx.enqueue_function_experimental[naive_kernel](
            input_buf,
            filter_qrscf_buf,
            output_naive_buf,
            stride_idx,
            dilation_idx,
            pad_idx,
            Int(1),
            grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
            block_dim=(block_size, block_size, 1),
        )
        conv3d_cudnn[dtype, dtype, dtype](
            input_buf,
            filter_fcqrs_buf,
            output_cudnn_buf,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
    ctx.synchronize()

    @parameter
    @__copy_capture(input_buf, filter_qrscf_buf, output_naive_buf)
    def naive_bench() raises:
        ctx.enqueue_function_experimental[naive_kernel](
            input_buf,
            filter_qrscf_buf,
            output_naive_buf,
            stride_idx,
            dilation_idx,
            pad_idx,
            Int(1),
            grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
            block_dim=(block_size, block_size, 1),
        )

    var naive_ns = ctx.execution_time[naive_bench](num_iters)
    var naive_ms = Float64(naive_ns) / 1e6 / Float64(num_iters)
    var naive_tflops = Float64(flops) / (naive_ms / 1000) / 1e12

    @parameter
    @__copy_capture(input_buf, filter_fcqrs_buf, output_cudnn_buf)
    def cudnn_bench() raises:
        conv3d_cudnn[dtype, dtype, dtype](
            input_buf,
            filter_fcqrs_buf,
            output_cudnn_buf,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )

    var cudnn_ns = ctx.execution_time[cudnn_bench](num_iters)
    var cudnn_ms = Float64(cudnn_ns) / 1e6 / Float64(num_iters)
    var cudnn_tflops = Float64(flops) / (cudnn_ms / 1000) / 1e12

    var ratio = naive_ms / cudnn_ms
    print(
        "  Naive Mojo: ",
        naive_ms,
        " ms  (",
        naive_tflops,
        " TFLOPS)",
        sep="",
    )
    print(
        "  cuDNN:      ", cudnn_ms, " ms  (", cudnn_tflops, " TFLOPS)", sep=""
    )
    print("  Naive / cuDNN: ", ratio, "x slower  (lower is better)", sep="")
    print()

    input_host.free()
    filter_qrscf_host.free()
    filter_fcqrs_host.free()
    _ = input_dev^
    _ = filter_qrscf_dev^
    _ = filter_fcqrs_dev^
    _ = output_naive_dev^
    _ = output_cudnn_dev^


def main() raises:
    print("=" * 70)
    print("WAN VAE 3D CONV BENCHMARK: Naive Mojo (QRSCF) vs cuDNN")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # Tiny sanity shape, stays in L2.
        bench_conv3d[
            DType.bfloat16,
            batch=1,
            in_depth=8,
            in_height=16,
            in_width=16,
            in_channels=16,
            out_channels=16,
            filter_q=1,
            filter_r=1,
            filter_s=1,
            pad_d=0,
            pad_h=0,
            pad_w=0,
        ](ctx, label="sanity_1x1x1_16->16", num_iters=200, warmup_iters=10)

        # WAN post_quant_conv @ latent resolution (1x1x1, 16->16).
        bench_conv3d[
            DType.bfloat16,
            batch=1,
            in_depth=21,
            in_height=30,
            in_width=52,
            in_channels=16,
            out_channels=16,
            filter_q=1,
            filter_r=1,
            filter_s=1,
            pad_d=0,
            pad_h=0,
            pad_w=0,
        ](ctx, label="WAN_post_quant_conv", num_iters=100, warmup_iters=5)

        # WAN conv_in: 3x3x3, 16->384 at latent resolution.
        bench_conv3d[
            DType.bfloat16,
            batch=1,
            in_depth=21,
            in_height=30,
            in_width=52,
            in_channels=16,
            out_channels=384,
            filter_q=3,
            filter_r=3,
            filter_s=3,
            pad_d=1,
            pad_h=1,
            pad_w=1,
        ](ctx, label="WAN_conv_in_16to384", num_iters=50, warmup_iters=5)

        # WAN mid ResidualBlock: 3x3x3, 384->384.
        bench_conv3d[
            DType.bfloat16,
            batch=1,
            in_depth=21,
            in_height=30,
            in_width=52,
            in_channels=384,
            out_channels=384,
            filter_q=3,
            filter_r=3,
            filter_s=3,
            pad_d=1,
            pad_h=1,
            pad_w=1,
        ](ctx, label="WAN_mid_res_384to384", num_iters=20, warmup_iters=3)

        # WAN time_conv upsample: 3x1x1, 192->384 at 60x104 spatial.
        bench_conv3d[
            DType.bfloat16,
            batch=1,
            in_depth=21,
            in_height=60,
            in_width=104,
            in_channels=192,
            out_channels=384,
            filter_q=3,
            filter_r=1,
            filter_s=1,
            pad_d=1,
            pad_h=0,
            pad_w=0,
        ](ctx, label="WAN_time_conv_192to384", num_iters=20, warmup_iters=3)

        # WAN upsampled residual: 3x3x3, 192->192 at 42x120x208.
        bench_conv3d[
            DType.bfloat16,
            batch=1,
            in_depth=42,
            in_height=120,
            in_width=208,
            in_channels=192,
            out_channels=192,
            filter_q=3,
            filter_r=3,
            filter_s=3,
            pad_d=1,
            pad_h=1,
            pad_w=1,
        ](ctx, label="WAN_upsampled_res_192to192", num_iters=10, warmup_iters=2)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
