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
from std.sys import get_defined_int, get_defined_string

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu, matmul_kernel_naive

from std.utils import IndexList


def _get_run_name[
    transpose: Bool,
    in_dtype: DType,
    out_dtype: DType,
](name: String, shape_c: Coord, shape_a: Coord, shape_b: Coord,) -> String:
    return String(
        name,
        "(",
        in_dtype,
        "->",
        out_dtype,
        ") : ",
        shape_c[0],
        ",",
        shape_c[1],
        ",",
        shape_a[1],
    )


def bench_matmul[
    in_dtype: DType,
    out_dtype: DType,
](
    ctx: DeviceContext,
    mut h: Bench,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
) raises:
    var mat_c_buf = ctx.enqueue_create_buffer[out_dtype](
        shape_c[0].value() * shape_c[1].value()
    )
    var mat_a_buf = ctx.enqueue_create_buffer[in_dtype](
        shape_a[0].value() * shape_a[1].value()
    )
    var mat_b_buf = ctx.enqueue_create_buffer[in_dtype](
        shape_b[0].value() * shape_b[1].value()
    )

    var mat_c = TileTensor(mat_c_buf, row_major(shape_c))
    var mat_a = TileTensor(mat_a_buf, row_major(shape_a))
    var mat_b = TileTensor(mat_b_buf, row_major(shape_b))

    @parameter
    @always_inline
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[transpose_b=False, use_tensor_core=True](
                mat_c, mat_a, mat_b, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[False, in_dtype, out_dtype](
                "gemv_gevm", shape_c, shape_a, shape_b
            )
        ),
        [
            ThroughputMeasure(
                BenchMetric.flops,
                2
                * shape_c[0].value()
                * shape_c[1].value()
                * shape_b[0].value(),
            )
        ],
    )


def bench_matmul_transpose[
    in_dtype: DType,
    out_dtype: DType,
](
    ctx: DeviceContext,
    mut h: Bench,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
) raises:
    var mat_c_buf = ctx.enqueue_create_buffer[out_dtype](
        shape_c[0].value() * shape_c[1].value()
    )
    var mat_a_buf = ctx.enqueue_create_buffer[in_dtype](
        shape_a[0].value() * shape_a[1].value()
    )
    var mat_b_buf = ctx.enqueue_create_buffer[in_dtype](
        shape_b[0].value() * shape_b[1].value()
    )

    var mat_c = TileTensor(mat_c_buf, row_major(shape_c))
    var mat_a = TileTensor(mat_a_buf, row_major(shape_a))
    var mat_b = TileTensor(mat_b_buf, row_major(shape_b))

    @parameter
    @always_inline
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[transpose_b=True, use_tensor_core=True](
                mat_c, mat_a, mat_b, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[True, in_dtype, out_dtype](
                "gemv_transpose", shape_c, shape_a, shape_b
            )
        ),
        [
            ThroughputMeasure(
                BenchMetric.flops,
                2
                * shape_c[0].value()
                * shape_c[1].value()
                * shape_b[1].value(),
            )
        ],
    )


def bench_matmul_naive[
    in_type: DType,
    out_type: DType,
](
    ctx: DeviceContext,
    mut h: Bench,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
) raises:
    var mat_c_buf = ctx.enqueue_create_buffer[out_type](
        shape_c[0].value() * shape_c[1].value()
    )
    var mat_a_buf = ctx.enqueue_create_buffer[in_type](
        shape_a[0].value() * shape_a[1].value()
    )
    var mat_b_buf = ctx.enqueue_create_buffer[in_type](
        shape_b[0].value() * shape_b[1].value()
    )

    var c_tt = TileTensor(mat_c_buf.unsafe_ptr(), row_major(shape_c))
    var a_tt = TileTensor(mat_a_buf, row_major(shape_a))
    var b_tt = TileTensor(mat_b_buf, row_major(shape_b))

    var M = shape_c[0].value()
    var N = shape_c[1].value()
    var K = shape_a[1].value()

    comptime BLOCK_DIM = 16
    comptime WARPS_PER_BLOCK = 32

    @always_inline
    @__copy_capture(M, N, K)
    @parameter
    def bench_func(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            comptime kernel = matmul_kernel_naive[
                out_type,
                in_type,
                in_type,
                type_of(c_tt).LayoutType,
                type_of(a_tt).LayoutType,
                type_of(b_tt).LayoutType,
                BLOCK_DIM,
            ]
            ctx.enqueue_function[kernel, kernel](
                c_tt,
                a_tt.as_immut(),
                b_tt.as_immut(),
                M,
                N,
                K,
                grid_dim=(ceildiv(M, WARPS_PER_BLOCK), ceildiv(N, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[True, in_type, out_type](
                "gemv_naive", shape_c, shape_a, shape_b
            )
        ),
        [
            ThroughputMeasure(
                BenchMetric.flops,
                2
                * shape_c[0].value()
                * shape_c[1].value()
                * shape_b[1].value(),
            )
        ],
    )

    ctx.synchronize()

    # Retain our buffers till the end.
    _ = mat_c_buf^
    _ = mat_a_buf^
    _ = mat_b_buf^


def create_matmul_bench[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    in_dtype: DType,
    out_dtype: DType,
](
    ctx: DeviceContext,
    mut h: Bench,
    m: MType,
    n: NType,
    k: KType,
    mode: String = "default",
) raises:
    if mode == "default":
        bench_matmul[in_dtype, out_dtype](
            ctx, h, Coord(m, n), Coord(m, k), Coord(k, n)
        )

    elif mode == "transpose":
        bench_matmul_transpose[in_dtype, out_dtype](
            ctx, h, Coord(m, n), Coord(m, k), Coord(n, k)
        )
    elif mode == "naive":
        bench_matmul_naive[in_dtype, out_dtype](
            ctx, h, Coord(m, n), Coord(m, k), Coord(k, n)
        )


@parameter
def get_dtype[output_type: String]() -> DType:
    if output_type == "float32":
        return DType.float32
    elif output_type == "float16":
        return DType.float16

    return DType.bfloat16


def main() raises:
    var h = Bench()

    comptime input_type = DType.bfloat16

    var M = Int(arg_parse("M", 1))
    comptime N = get_defined_int["N", 1]()
    comptime K = get_defined_int["K", 1]()

    comptime output_type = get_dtype[
        get_defined_string["output_type", "bfloat16"]()
    ]()

    var mode = arg_parse("mode", "default")  # [default, naive, transpose]
    var shape = IndexList[3](M, N, K)

    with DeviceContext() as ctx:
        create_matmul_bench[input_type, output_type](
            ctx,
            h,
            Idx(shape[0]),
            Idx[N](),
            Idx[K](),
            mode,
        )

    h.dump_report()
