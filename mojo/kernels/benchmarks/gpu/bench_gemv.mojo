# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-build %s

from math import ceildiv
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host.device_context import DeviceBuffer, DeviceContext
from linalg.matmul_gpu import _matmul_gpu, matmul_kernel_naive
from internal_utils import DeviceNDBuffer, bench_compile_time
from utils import StaticIntTuple


fn _get_run_name[
    transpose: Bool,
    type: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    name: String,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
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
    inout h: Bench,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) raises:
    var mat_c = DeviceNDBuffer[dtype, 2, shape_c](shape_c_dim, ctx=ctx)
    var mat_a = DeviceNDBuffer[dtype, 2, shape_a](shape_a_dim, ctx=ctx)
    var mat_b = DeviceNDBuffer[dtype, 2, shape_b](shape_b_dim, ctx=ctx)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
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
    inout h: Bench,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) raises:
    var mat_c = DeviceNDBuffer[dtype, 2, shape_c](shape_c_dim, ctx=ctx)
    var mat_a = DeviceNDBuffer[dtype, 2, shape_a](shape_a_dim, ctx=ctx)
    var mat_b = DeviceNDBuffer[dtype, 2, shape_b](shape_b_dim, ctx=ctx)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
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
    inout h: Bench,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) raises:
    var mat_c = DeviceNDBuffer[dtype, 2, shape_c](shape_c_dim, ctx=ctx)
    var mat_a = DeviceNDBuffer[dtype, 2, shape_a](shape_a_dim, ctx=ctx)
    var mat_b = DeviceNDBuffer[dtype, 2, shape_b](shape_b_dim, ctx=ctx)

    var M = shape_c_dim[0]
    var N = shape_c_dim[1]
    var K = shape_a_dim[1]

    alias BLOCK_DIM = 16
    alias WARPS_PER_BLOCK = 32
    var func_gemv = ctx.compile_function[
        matmul_kernel_naive[
            dtype,
            dtype,
            dtype,
            BLOCK_DIM,
        ]
    ]()

    @always_inline
    @__copy_capture(M, N, K)
    @parameter
    fn bench_func(inout b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            ctx.enqueue_function(
                func_gemv,
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
    _ = func_gemv^


struct ValOrDim[dim: Dim = Dim()]:
    var value: Int

    fn __init__(inout self):
        constrained[
            not dim.is_dynamic(),
            "Can't construct a dynamic dim with no runtime value",
        ]()
        self.value = dim.get()

    fn __init__(inout self, v: Int):
        self.value = v


fn static[d: Int]() -> ValOrDim[d]:
    return ValOrDim[d]()


fn dynamic(d: Int) -> ValOrDim:
    return ValOrDim(d)


fn create_matmul_bench[
    dtype: DType
](
    ctx: DeviceContext, inout h: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    bench_matmul[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        DimList(k.dim, n.dim),
    ](ctx, h, (m.value, n.value), (m.value, k.value), (k.value, n.value))


fn create_matmul_bench_t[
    dtype: DType
](
    ctx: DeviceContext, inout h: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    bench_matmul_transpose[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        DimList(n.dim, k.dim),
    ](ctx, h, (m.value, n.value), (m.value, k.value), (n.value, k.value))


fn create_matmul_bench_n[
    dtype: DType
](
    ctx: DeviceContext, inout h: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    bench_matmul_naive[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        DimList(k.dim, n.dim),
    ](ctx, h, (m.value, n.value), (m.value, k.value), (k.value, n.value))


fn main() raises:
    var h = Bench()
    with DeviceContext() as ctx:
        var shape_list = List[StaticIntTuple[3]](
            StaticIntTuple[3](1, 5120, 3072),
            StaticIntTuple[3](1, 3072, 5120),
            StaticIntTuple[3](1, 3072, 12288),
            StaticIntTuple[3](1, 12288, 3072),
            StaticIntTuple[3](1, 3072, 3072),
            StaticIntTuple[3](1, 3072, 32768),
            StaticIntTuple[3](1, 32768, 3072),
            StaticIntTuple[3](1, 5120, 5120),
            StaticIntTuple[3](1, 32000, 4096),
            StaticIntTuple[3](1, 4096, 32000),
        )

        for s in range(len(shape_list)):
            create_matmul_bench[DType.bfloat16](
                ctx,
                h,
                dynamic(shape_list[s][0]),
                dynamic(shape_list[s][1]),
                dynamic(shape_list[s][2]),
            )
            create_matmul_bench_t[DType.bfloat16](
                ctx,
                h,
                dynamic(shape_list[s][0]),
                dynamic(shape_list[s][1]),
                dynamic(shape_list[s][2]),
            )
            create_matmul_bench_n[DType.bfloat16](
                ctx,
                h,
                dynamic(shape_list[s][0]),
                dynamic(shape_list[s][1]),
                dynamic(shape_list[s][2]),
            )

    h.dump_report()
