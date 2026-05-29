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

from std.math import ceildiv
from std.random import rand, seed
from std.sys import get_defined_dtype, get_defined_int, size_of

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import TileTensor, row_major
from state_space.causal_conv1d import causal_conv1d_channel_first_fwd_gpu


def _get_run_name[
    dtype: DType, width: Int
](batch: Int, dim: Int, seqlen: Int, silu: Bool) -> String:
    return String(
        "causal_conv1d_channel_first_fwd(",
        dtype,
        ") : width=",
        width,
        " : batch=",
        batch,
        ", dim=",
        dim,
        ", seqlen=",
        seqlen,
        ", silu=",
        silu,
    )


def execute_causal_conv1d[
    dtype: DType, width: Int
](
    ctx: DeviceContext,
    mut m: Bench,
    batch: Int,
    dim: Int,
    seqlen: Int,
    silu: Bool,
) raises:
    # Matches the validated test config (kNThreads=128, kNElts=4).
    comptime kNThreads = 128
    comptime kNElts = 4
    comptime kWidth = width

    # Host buffers, filled with random data.
    var input_host = ctx.enqueue_create_host_buffer[dtype](batch * dim * seqlen)
    var weight_host = ctx.enqueue_create_host_buffer[dtype](dim * width)
    var bias_host = ctx.enqueue_create_host_buffer[dtype](dim)
    rand[dtype](input_host.unsafe_ptr(), batch * dim * seqlen)
    rand[dtype](weight_host.unsafe_ptr(), dim * width)
    rand[dtype](bias_host.unsafe_ptr(), dim)

    # Device buffers.
    var input_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)
    var weight_device = ctx.enqueue_create_buffer[dtype](dim * width)
    var bias_device = ctx.enqueue_create_buffer[dtype](dim)
    var output_device = ctx.enqueue_create_buffer[dtype](batch * dim * seqlen)

    with ctx.push_context():
        ctx.enqueue_copy(input_device, input_host.unsafe_ptr())
        ctx.enqueue_copy(weight_device, weight_host.unsafe_ptr())
        ctx.enqueue_copy(bias_device, bias_host.unsafe_ptr())

    # Channel-first (B, C, L) TileTensors over the device buffers.
    var input_tt = TileTensor(input_device, row_major(batch, dim, seqlen))
    var weight_tt = TileTensor(weight_device, row_major(dim, width))
    var bias_tt = TileTensor(bias_device, row_major(dim))
    var output_tt = TileTensor(output_device, row_major(batch, dim, seqlen))

    # Channel-first strides.
    var x_batch_stride: UInt32 = UInt32(dim * seqlen)
    var x_c_stride: UInt32 = UInt32(seqlen)
    var x_l_stride: UInt32 = 1
    var weight_c_stride: UInt32 = UInt32(width)
    var weight_width_stride: UInt32 = 1
    var out_batch_stride: UInt32 = UInt32(dim * seqlen)
    var out_c_stride: UInt32 = UInt32(seqlen)
    var out_l_stride: UInt32 = 1
    var bias_stride: UInt32 = 1
    var silu_activation = Int8(silu)

    var compiled_func = ctx.compile_function[
        causal_conv1d_channel_first_fwd_gpu[
            dtype,
            dtype,
            dtype,
            kNThreads,
            kWidth,
            kNElts,
            dtype,
            input_tt.LayoutType,
            weight_tt.LayoutType,
            output_tt.LayoutType,
            bias_tt.LayoutType,
        ]
    ]()

    @parameter
    @always_inline
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            with ctx.push_context():
                ctx.enqueue_function(
                    compiled_func,
                    batch,
                    dim,
                    seqlen,
                    width,
                    input_tt,
                    weight_tt,
                    output_tt,
                    bias_tt,
                    x_batch_stride,
                    x_c_stride,
                    x_l_stride,
                    weight_c_stride,
                    weight_width_stride,
                    out_batch_stride,
                    out_c_stride,
                    out_l_stride,
                    bias_stride,
                    silu_activation,
                    grid_dim=(
                        ceildiv(seqlen, kNThreads * kNElts),
                        dim,
                        batch,
                    ),
                    block_dim=(kNThreads),
                )

        b.iter_custom[kernel_launch](ctx)

    # The kernel reads x + writes y (the dominant memory traffic), so report
    # bytes moved across both: 2 * batch * dim * seqlen elements.
    m.bench_function[bench_func](
        BenchId(_get_run_name[dtype, width](batch, dim, seqlen, silu)),
        [
            ThroughputMeasure(
                BenchMetric.bytes,
                2 * batch * dim * seqlen * size_of[dtype](),
            )
        ],
    )

    _ = input_host
    _ = weight_host
    _ = bias_host
    _ = input_device
    _ = weight_device
    _ = bias_device
    _ = output_device


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime width = get_defined_int["width", 4]()

    var batch = arg_parse("batch", 1)
    var dim = arg_parse("dim", 2048)
    var seqlen = arg_parse("seqlen", 4096)
    var silu = arg_parse("silu", True)

    seed(0)

    var m = Bench()
    with DeviceContext() as ctx:
        execute_causal_conv1d[dtype, width](ctx, m, batch, dim, seqlen, silu)

    m.dump_report()
