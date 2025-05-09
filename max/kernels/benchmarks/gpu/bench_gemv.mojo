# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceBuffer, DeviceContext
from internal_utils import DeviceNDBuffer, arg_parse
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive

from utils import IndexList


fn _get_run_name[
    transpose: Bool,
    type: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    name: String,
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
) -> String:
    var str = name
    str += "("
    str += type.__str__()
    str += ") : "
    str += shape_c_dim[0].__str__()
    str += ","
    str += shape_c_dim[1].__str__()
    str += ","
    str += shape_a_dim[1].__str__()
    return str


fn bench_matmul[
    dtype: DType,
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
    var mat_c = DeviceNDBuffer[dtype, 2, shape_c](shape_c_dim, ctx=ctx)
    var mat_a = DeviceNDBuffer[dtype, 2, shape_a](shape_a_dim, ctx=ctx)
    var mat_b = DeviceNDBuffer[dtype, 2, shape_b](shape_b_dim, ctx=ctx)

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[transpose_b=False, use_tensor_core=True](
                mat_c.tensor, mat_a.tensor, mat_b.tensor, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[False, dtype, shape_c, shape_a, shape_b](
                "gemv_gevm", shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        ThroughputMeasure(
            BenchMetric.flops,
            2 * shape_c_dim[0] * shape_c_dim[1] * shape_b_dim[0],
        ),
    )

    # Retain our buffers till the end.
    _ = mat_c^
    _ = mat_a^
    _ = mat_b^


fn bench_matmul_transpose[
    dtype: DType,
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
    var mat_c = DeviceNDBuffer[dtype, 2, shape_c](shape_c_dim, ctx=ctx)
    var mat_a = DeviceNDBuffer[dtype, 2, shape_a](shape_a_dim, ctx=ctx)
    var mat_b = DeviceNDBuffer[dtype, 2, shape_b](shape_b_dim, ctx=ctx)

    @parameter
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[transpose_b=True, use_tensor_core=True](
                mat_c.tensor, mat_a.tensor, mat_b.tensor, ctx
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[True, dtype, shape_c, shape_a, shape_b](
                "gemv_transpose", shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        ThroughputMeasure(
            BenchMetric.flops,
            2 * shape_c_dim[0] * shape_c_dim[1] * shape_b_dim[1],
        ),
    )

    # Retain our buffers till the end.
    _ = mat_c^
    _ = mat_a^
    _ = mat_b^


fn bench_matmul_naive[
    dtype: DType,
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
    var mat_c = DeviceNDBuffer[dtype, 2, shape_c](shape_c_dim, ctx=ctx)
    var mat_a = DeviceNDBuffer[dtype, 2, shape_a](shape_a_dim, ctx=ctx)
    var mat_b = DeviceNDBuffer[dtype, 2, shape_b](shape_b_dim, ctx=ctx)

    var M = shape_c_dim[0]
    var N = shape_c_dim[1]
    var K = shape_a_dim[1]

    alias BLOCK_DIM = 16
    alias WARPS_PER_BLOCK = 32

    @always_inline
    @__copy_capture(M, N, K)
    @parameter
    fn bench_func(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            ctx.enqueue_function[
                matmul_kernel_naive[dtype, dtype, dtype, BLOCK_DIM]
            ](
                mat_c.tensor.data,
                mat_a.tensor.data,
                mat_b.tensor.data,
                UInt(M),
                UInt(N),
                UInt(K),
                grid_dim=(ceildiv(M, WARPS_PER_BLOCK), ceildiv(N, BLOCK_DIM)),
                block_dim=(BLOCK_DIM, BLOCK_DIM),
            )

        b.iter_custom[kernel_launch](ctx)

    h.bench_function[bench_func](
        BenchId(
            _get_run_name[True, dtype, shape_c, shape_a, shape_b](
                "gemv_naive", shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        ThroughputMeasure(
            BenchMetric.flops,
            2 * shape_c_dim[0] * shape_c_dim[1] * shape_b_dim[1],
        ),
    )

    ctx.synchronize()

    # Retain our buffers till the end.
    _ = mat_c^
    _ = mat_a^
    _ = mat_b^


struct ValOrDim[dim: Dim = Dim()]:
    var value: Int

    fn __init__(out self):
        constrained[
            not dim.is_dynamic(),
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim.get()

    @implicit
    fn __init__(out self, v: Int):
        self.value = v


fn dynamic(d: Int) -> ValOrDim:
    return ValOrDim(d)


fn create_matmul_bench[
    dtype: DType
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
            dtype,
            DimList(m.dim, n.dim),
            DimList(m.dim, k.dim),
            DimList(k.dim, n.dim),
        ](ctx, h, (m.value, n.value), (m.value, k.value), (k.value, n.value))

    elif mode == "transpose":
        bench_matmul_transpose[
            dtype,
            DimList(m.dim, n.dim),
            DimList(m.dim, k.dim),
            DimList(n.dim, k.dim),
        ](ctx, h, (m.value, n.value), (m.value, k.value), (n.value, k.value))
    elif mode == "naive":
        bench_matmul_naive[
            dtype,
            DimList(m.dim, n.dim),
            DimList(m.dim, k.dim),
            DimList(k.dim, n.dim),
        ](ctx, h, (m.value, n.value), (m.value, k.value), (k.value, n.value))


fn main() raises:
    var h = Bench()

    alias dtype = DType.bfloat16
    alias M = 1

    var N = arg_parse("N", 1)
    var K = arg_parse("K", 1)
    var mode = arg_parse("mode", "default")  # [default, naive, transpose]
    var shape = IndexList[3](M, N, K)

    with DeviceContext() as ctx:
        create_matmul_bench[dtype](
            ctx,
            h,
            dynamic(shape[0]),
            dynamic(shape[1]),
            dynamic(shape[2]),
            mode,
        )

    h.dump_report()
