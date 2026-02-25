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

from math import ceildiv
from sys import env_get_int, env_get_string

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import arg_parse
from internal_utils._utils import ValOrDim, dynamic, static
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu import _matmul_gpu, matmul_kernel_naive

from utils import IndexList


fn _get_run_name[
    transpose: Bool,
    in_dtype: DType,
    out_dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    name: String,
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
) -> String:
    return String(
        name,
        "(",
        in_dtype,
        "->",
        out_dtype,
        ") : ",
        shape_c_dim[0],
        ",",
        shape_c_dim[1],
        ",",
        shape_a_dim[1],
    )


fn bench_matmul[
    in_dtype: DType,
    out_dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    ctx: DeviceContext,
    mut h: Bench,
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
) raises:
    var mat_c_buf = ctx.enqueue_create_buffer[out_dtype](
        shape_c_dim[0] * shape_c_dim[1]
    )
    var mat_a_buf = ctx.enqueue_create_buffer[in_dtype](
        shape_a_dim[0] * shape_a_dim[1]
    )
    var mat_b_buf = ctx.enqueue_create_buffer[in_dtype](
        shape_b_dim[0] * shape_b_dim[1]
    )

    var mat_c = NDBuffer[out_dtype, 2, _, shape_c](
        mat_c_buf.unsafe_ptr(), shape_c_dim
    )
    var mat_a = NDBuffer[in_dtype, 2, _, shape_a](
        mat_a_buf.unsafe_ptr(), shape_a_dim
    )
    var mat_b = NDBuffer[in_dtype, 2, _, shape_b](
        mat_b_buf.unsafe_ptr(), shape_b_dim
    )

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[transpose_b=False, use_tensor_core=True](
                mat_c, mat_a, mat_b, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[
                False, in_dtype, out_dtype, shape_c, shape_a, shape_b
            ]("gemv_gevm", shape_c_dim, shape_a_dim, shape_b_dim)
        ),
        [
            ThroughputMeasure(
                BenchMetric.flops,
                2 * shape_c_dim[0] * shape_c_dim[1] * shape_b_dim[0],
            )
        ],
    )

    # Retain our buffers till the end.
    _ = mat_c_buf^
    _ = mat_a_buf^
    _ = mat_b_buf^


fn bench_matmul_transpose[
    in_dtype: DType,
    out_dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    ctx: DeviceContext,
    mut h: Bench,
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
) raises:
    var mat_c_buf = ctx.enqueue_create_buffer[out_dtype](
        shape_c_dim[0] * shape_c_dim[1]
    )
    var mat_a_buf = ctx.enqueue_create_buffer[in_dtype](
        shape_a_dim[0] * shape_a_dim[1]
    )
    var mat_b_buf = ctx.enqueue_create_buffer[in_dtype](
        shape_b_dim[0] * shape_b_dim[1]
    )

    var mat_c = NDBuffer[out_dtype, 2, _, shape_c](
        mat_c_buf.unsafe_ptr(), shape_c_dim
    )
    var mat_a = NDBuffer[in_dtype, 2, _, shape_a](
        mat_a_buf.unsafe_ptr(), shape_a_dim
    )
    var mat_b = NDBuffer[in_dtype, 2, _, shape_b](
        mat_b_buf.unsafe_ptr(), shape_b_dim
    )

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[transpose_b=True, use_tensor_core=True](
                mat_c, mat_a, mat_b, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[True, in_dtype, out_dtype, shape_c, shape_a, shape_b](
                "gemv_transpose", shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        [
            ThroughputMeasure(
                BenchMetric.flops,
                2 * shape_c_dim[0] * shape_c_dim[1] * shape_b_dim[1],
            )
        ],
    )

    # Retain our buffers till the end.
    _ = mat_c_buf^
    _ = mat_a_buf^
    _ = mat_b_buf^


fn bench_matmul_naive[
    in_type: DType,
    out_type: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    ctx: DeviceContext,
    mut h: Bench,
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
) raises:
    var mat_c_buf = ctx.enqueue_create_buffer[out_type](
        shape_c_dim[0] * shape_c_dim[1]
    )
    var mat_a_buf = ctx.enqueue_create_buffer[in_type](
        shape_a_dim[0] * shape_a_dim[1]
    )
    var mat_b_buf = ctx.enqueue_create_buffer[in_type](
        shape_b_dim[0] * shape_b_dim[1]
    )

    var mat_c = NDBuffer[out_type, 2, _, shape_c](
        mat_c_buf.unsafe_ptr(), shape_c_dim
    )
    var mat_a = NDBuffer[in_type, 2, _, shape_a](
        mat_a_buf.unsafe_ptr(), shape_a_dim
    )
    var mat_b = NDBuffer[in_type, 2, _, shape_b](
        mat_b_buf.unsafe_ptr(), shape_b_dim
    )

    var c_tensor = from_ndbuffer_row_major(mat_c)
    var a_tensor = from_ndbuffer_row_major(mat_a)
    var b_tensor = from_ndbuffer_row_major(mat_b)

    var M = shape_c_dim[0]
    var N = shape_c_dim[1]
    var K = shape_a_dim[1]

    comptime BLOCK_DIM = 16
    comptime WARPS_PER_BLOCK = 32

    @always_inline
    @__copy_capture(M, N, K)
    @parameter
    fn bench_func(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            comptime kernel = matmul_kernel_naive[
                out_type,
                in_type,
                in_type,
                c_tensor.layout,
                a_tensor.layout,
                b_tensor.layout,
                BLOCK_DIM,
            ]
            ctx.enqueue_function[kernel, kernel](
                c_tensor,
                a_tensor,
                b_tensor,
                M,
                N,
                K,
                grid_dim=(ceildiv(M, WARPS_PER_BLOCK), ceildiv(N, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[True, in_type, out_type, shape_c, shape_a, shape_b](
                "gemv_naive", shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        [
            ThroughputMeasure(
                BenchMetric.flops,
                2 * shape_c_dim[0] * shape_c_dim[1] * shape_b_dim[1],
            )
        ],
    )

    ctx.synchronize()

    # Retain our buffers till the end.
    _ = mat_c_buf^
    _ = mat_a_buf^
    _ = mat_b_buf^


fn create_matmul_bench[
    in_dtype: DType,
    out_dtype: DType,
](
    ctx: DeviceContext,
    mut h: Bench,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    mode: String = "default",
) raises:
    if mode == "default":
        bench_matmul[
            in_dtype,
            out_dtype,
            DimList(m.dim, n.dim),
            DimList(m.dim, k.dim),
            DimList(k.dim, n.dim),
        ](ctx, h, (m.value, n.value), (m.value, k.value), (k.value, n.value))

    elif mode == "transpose":
        bench_matmul_transpose[
            in_dtype,
            out_dtype,
            DimList(m.dim, n.dim),
            DimList(m.dim, k.dim),
            DimList(n.dim, k.dim),
        ](ctx, h, (m.value, n.value), (m.value, k.value), (n.value, k.value))
    elif mode == "naive":
        bench_matmul_naive[
            in_dtype,
            out_dtype,
            DimList(m.dim, n.dim),
            DimList(m.dim, k.dim),
            DimList(k.dim, n.dim),
        ](ctx, h, (m.value, n.value), (m.value, k.value), (k.value, n.value))


@parameter
fn get_dtype[output_type: String]() -> DType:
    if output_type == "float32":
        return DType.float32
    elif output_type == "float16":
        return DType.float16

    return DType.bfloat16


def main():
    var h = Bench()

    comptime input_type = DType.bfloat16

    var M = Int(arg_parse("M", 1))
    comptime N = env_get_int["N", 1]()
    comptime K = env_get_int["K", 1]()

    comptime output_type = get_dtype[
        env_get_string["output_type", "bfloat16"]()
    ]()

    var mode = arg_parse("mode", "default")  # [default, naive, transpose]
    var shape = IndexList[3](M, N, K)

    with DeviceContext() as ctx:
        create_matmul_bench[input_type, output_type](
            ctx,
            h,
            dynamic(shape[0]),
            static[N](),
            static[K](),
            mode,
        )

    h.dump_report()
