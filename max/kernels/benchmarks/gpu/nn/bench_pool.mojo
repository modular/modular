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

"""Benchmarks the GPU 2D pool kernels on NHWC tensors.

Run through kbench with bench_pool.yaml, or directly, for example:

    mojo -D op=avgpool -D dtype=bfloat16 -D in_shape=1x128x128x512 \
        -D kernel=2x2 -D stride=2x2 -D pad=0 bench_pool.mojo
"""

from std.sys import (
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
    size_of,
)

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, TileTensor, row_major
from nn.pool import avg_pool_gpu, max_pool_gpu
from nn.shapes import get_sliding_window_out_dim

from std.utils.index import IndexList


def bench_pool_gpu[
    in_rank: Int,
    kernel_rank: Int,
    stride_rank: Int,
    //,
    dtype: DType,
    op: StaticString,
    in_shape: IndexList[in_rank],
    kernel: IndexList[kernel_rank],
    stride: IndexList[stride_rank],
    pad: Int,
](ctx: DeviceContext, mut b: Bench) raises:
    """Benchmarks one pool shape.

    Parameters:
        in_rank: Rank of the input shape.
        kernel_rank: Rank of the window shape.
        stride_rank: Rank of the stride shape.
        dtype: Data type of the input and output tensors.
        op: Pooling method, either `"maxpool"` or `"avgpool"`.
        in_shape: NHWC input shape.
        kernel: Pool window size, `(window_h, window_w)`.
        stride: Pool stride, `(stride_h, stride_w)`.
        pad: Symmetric padding per side, on both spatial dimensions.
    """
    var out_shape = IndexList[in_rank](
        in_shape[0],
        get_sliding_window_out_dim(
            in_shape[1], kernel[0], 1, stride[0], 2 * pad
        ),
        get_sliding_window_out_dim(
            in_shape[2], kernel[1], 1, stride[1], 2 * pad
        ),
        in_shape[3],
    )
    var input_size = in_shape.flattened_length()
    var output_size = out_shape.flattened_length()

    var filter_stack: Array[Int32, _] = [
        Int32(kernel[0]),
        Int32(kernel[1]),
    ]
    var filter = TileTensor(filter_stack, row_major[2]())

    var stride_stack: Array[Int32, _] = [
        Int32(stride[0]),
        Int32(stride[1]),
    ]
    var strides = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilations = TileTensor(dilation_stack, row_major[2]())

    var padding_stack = Array[Int32, 4](fill=Int32(pad))
    var paddings = TileTensor(padding_stack, row_major[4]())

    var in_device = ctx.enqueue_create_buffer[dtype](input_size)
    var out_device = ctx.enqueue_create_buffer[dtype](output_size)
    ctx.enqueue_memset(in_device, 1)
    ctx.synchronize()

    var input = TileTensor(in_device, row_major(Coord(in_shape)))
    var output = TileTensor(out_device, row_major(Coord(out_shape)))

    @inline(.always)
    def bench_fn(
        mut b: Bencher,
    ) raises {
        mut output,
        imm input,
        imm ctx,
        imm filter,
        imm strides,
        imm dilations,
        imm paddings,
    }:
        @inline(.always)
        def kernel_launch(
            ctx: DeviceContext,
        ) raises {
            mut output,
            imm input,
            imm filter,
            imm strides,
            imm dilations,
            imm paddings,
        }:
            comptime if op == "maxpool":
                max_pool_gpu[int_type=DType.int32](
                    ctx,
                    input,
                    filter,
                    strides,
                    dilations,
                    paddings,
                    output,
                )
            else:
                avg_pool_gpu[int_type=DType.int32](
                    ctx,
                    input,
                    filter,
                    strides,
                    dilations,
                    paddings,
                    output,
                )

        bencher_iter_custom(b, kernel_launch, ctx)

    # Lower bound on memory traffic: read every input element once and write
    # every output element once.
    var total_bytes = (input_size + output_size) * size_of[dtype]()

    b.bench_function(
        bench_fn,
        BenchId(
            "pool_" + String(op),
            input_id=String(
                dtype,
                "/",
                in_shape,
                "/k",
                kernel,
                "s",
                stride,
                "p",
                pad,
            ),
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    ctx.synchronize()
    _ = in_device
    _ = out_device


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime op = get_defined_string["op", "maxpool"]()
    comptime in_shape = int_list_to_tuple[
        get_defined_shape["in_shape", "1x128x128x512"]()
    ]()
    comptime kernel = int_list_to_tuple[get_defined_shape["kernel", "2x2"]()]()
    comptime stride = int_list_to_tuple[get_defined_shape["stride", "2x2"]()]()
    comptime pad = get_defined_int["pad", 0]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_pool_gpu[dtype, op, in_shape, kernel, stride, pad](ctx, m)

    m.dump_report()
