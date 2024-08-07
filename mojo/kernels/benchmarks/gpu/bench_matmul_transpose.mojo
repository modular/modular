# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %bare-mojo build %s

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu.host.device_context import DeviceBuffer, DeviceContext
from linalg.matmul_gpu import _matmul_gpu


fn _size[rank: Int](dims: StaticIntTuple[rank]) -> Int:
    var size = 1

    @parameter
    for i in range(rank):
        size *= dims[i]
    return size


fn _create_device_buffer[
    dtype: DType, rank: Int, shape: DimList
](ctx: DeviceContext, dynamic_shape: StaticIntTuple[rank]) raises -> Tuple[
    DeviceBuffer[dtype], NDBuffer[dtype, rank, shape]
]:
    var storage = ctx.create_buffer[dtype](_size(dynamic_shape))
    return (
        storage,
        NDBuffer[dtype, rank, shape](storage.ptr, dynamic_shape=dynamic_shape),
    )


fn _get_run_name[
    transpose: Bool,
    type: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
](
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) -> String:
    var str = String("matmul(")
    if transpose:
        str = String("matmul transpose(")
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
    inout b: Bench,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) raises:
    var mat_c = _create_device_buffer[dtype, 2, shape_c](ctx, shape_c_dim)
    var mat_a = _create_device_buffer[dtype, 2, shape_a](ctx, shape_a_dim)
    var mat_b = _create_device_buffer[dtype, 2, shape_b](ctx, shape_b_dim)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[transpose_b=False, use_tensor_core=True](
                mat_c[1], mat_a[1], mat_b[1], ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_func](
        BenchId(
            _get_run_name[False, dtype, shape_c, shape_a, shape_b](
                shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(
            BenchMetric.elements,
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
    inout b: Bench,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) raises:
    var mat_c = _create_device_buffer[dtype, 2, shape_c](ctx, shape_c_dim)
    var mat_a = _create_device_buffer[dtype, 2, shape_a](ctx, shape_a_dim)
    var mat_b = _create_device_buffer[dtype, 2, shape_b](ctx, shape_b_dim)

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            _matmul_gpu[transpose_b=True, use_tensor_core=True](
                mat_c[1], mat_a[1], mat_b[1], ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_func](
        BenchId(
            _get_run_name[True, dtype, shape_c, shape_a, shape_b](
                shape_c_dim, shape_a_dim, shape_b_dim
            )
        ),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(
            BenchMetric.elements,
            2 * shape_c_dim[0] * shape_c_dim[1] * shape_b_dim[1],
        ),
    )

    # Retain our buffers till the end.
    _ = mat_c^
    _ = mat_a^
    _ = mat_b^


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
    ctx: DeviceContext, inout b: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    bench_matmul[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        DimList(k.dim, n.dim),
    ](ctx, b, (m.value, n.value), (m.value, k.value), (k.value, n.value))


fn create_matmul_bench_t[
    dtype: DType
](
    ctx: DeviceContext, inout b: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    bench_matmul_transpose[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        DimList(n.dim, k.dim),
    ](ctx, b, (m.value, n.value), (m.value, k.value), (n.value, k.value))


fn main() raises:
    with DeviceContext() as ctx:
        var b = Bench()

        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(5120), dynamic(3072)
        )
        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(3072), dynamic(5120)
        )
        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(3072), dynamic(12288)
        )
        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(12288), dynamic(3072)
        )
        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(3072), dynamic(3072)
        )
        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(3072), dynamic(32768)
        )
        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(32768), dynamic(3072)
        )
        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(32000), dynamic(4096)
        )
        create_matmul_bench[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(4096), dynamic(32000)
        )

        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(5120), dynamic(3072)
        )
        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(3072), dynamic(5120)
        )
        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(3072), dynamic(12288)
        )
        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(12288), dynamic(3072)
        )
        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(3072), dynamic(3072)
        )
        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(3072), dynamic(32768)
        )
        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(32768), dynamic(3072)
        )
        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(32000), dynamic(4096)
        )
        create_matmul_bench_t[DType.bfloat16](
            ctx, b, dynamic(1), dynamic(4096), dynamic(32000)
        )

        b.dump_report()
