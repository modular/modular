# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug %s


from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from buffer.dimlist import _make_tuple
from gpu.host.device_context import DeviceBuffer, DeviceContext
from linalg.matmul_gpu import _matmul_gpu
from internal_utils import DeviceNDBuffer, bench_compile_time, env_get_dtype
from utils import StaticIntTuple
from sys import env_get_int, sizeof
from math import align_up
from memory import UnsafePointer
from gpu.cublas.cublas import (
    check_cublas_error,
    cublasContext,
    cublasCreate,
    cublasDestroy,
)
from linalg.cublas import cublas_matmul


fn _get_run_name[
    type: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_cublas: Bool,
](
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) -> String:
    var name = String("cublas_matmul" if use_cublas else "matmul") + "("
    name += str(type)
    name += ") : "
    # M
    name += str(shape_c_dim[0])
    # N
    name += (
        "_dynamic"
        + " x "
        + str(shape_c_dim[1]) if shape_c.at[0]().is_dynamic() else " x "
        + str(shape_c_dim[1])
    )
    # K
    name += (
        "_dynamic"
        + " x "
        + str(shape_a_dim[1]) if shape_c.at[1]().is_dynamic() else " x "
        + str(shape_a_dim[1])
    )
    name += "_dynamic" if shape_a.at[1]().is_dynamic() else ""
    name += " transpose_b" if transpose_b else ""
    name += " cache_busting" if cache_busting else ""
    return name


fn bench_matmul[
    dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    *,
    cache_busting: Bool,
    use_cublas: Bool,
    transpose_b: Bool = False,
](
    ctx: DeviceContext,
    inout b: Bench,
    shape_c_dim: StaticIntTuple[2],
    shape_a_dim: StaticIntTuple[2],
    shape_b_dim: StaticIntTuple[2],
) raises:
    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    @always_inline
    fn get_size(shape: StaticIntTuple[2]) -> Int:
        return shape[0] * shape[1]

    alias simd_size = 4
    var stride_a = align_up(get_size(shape_a_dim), simd_size)
    var stride_b = align_up(get_size(shape_b_dim), simd_size)
    var stride_c = align_up(get_size(shape_c_dim), simd_size)

    alias k128 = 128 * 1024 * 1024
    var cache_a = align_up(k128, stride_a * sizeof[dtype]()) // sizeof[dtype]()
    var cache_b = align_up(k128, stride_b * sizeof[dtype]()) // sizeof[dtype]()
    var cache_c = align_up(k128, stride_c * sizeof[dtype]()) // sizeof[dtype]()

    var buffer_a = ctx.create_buffer[dtype](cache_a)
    var buffer_b = ctx.create_buffer[dtype](cache_b)
    var buffer_c = ctx.create_buffer[dtype](cache_c)

    var cublas_handle = UnsafePointer[cublasContext]()
    check_cublas_error(cublasCreate(UnsafePointer.address_of(cublas_handle)))

    @parameter
    @__copy_capture(
        cache_a, cache_b, cache_c, stride_a, stride_b, stride_c, cublas_handle
    )
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var offset_a = 0
            var offset_b = 0
            var offset_c = 0

            @parameter
            if cache_busting:
                offset_a = (iteration * stride_a) % cache_a
                offset_b = (iteration * stride_b) % cache_b
                offset_c = (iteration * stride_c) % cache_c
            var tensor_a = NDBuffer[dtype, 2, shape_a](
                buffer_a.ptr + offset_a, shape_a_dim
            )
            var tensor_b = NDBuffer[dtype, 2, shape_b](
                buffer_b.ptr + offset_b, shape_b_dim
            )
            var tensor_c = NDBuffer[dtype, 2, shape_c](
                buffer_c.ptr + offset_c, shape_c_dim
            )

            @parameter
            if use_cublas:
                _ = cublas_matmul[use_tf32=True](
                    cublas_handle,
                    tensor_c,
                    tensor_a,
                    tensor_b,
                    c_row_major=True,
                    transpose_b=transpose_b,
                )

            else:
                _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
                    tensor_c, tensor_a, tensor_b, ctx
                )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_func](
        BenchId(
            _get_run_name[
                dtype,
                shape_c,
                shape_a,
                shape_b,
                transpose_b=transpose_b,
                cache_busting=cache_busting,
                use_cublas=use_cublas,
            ](shape_c_dim, shape_a_dim, shape_b_dim)
        ),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(
            BenchMetric.elements,
            # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
            2 * shape_c_dim[0] * shape_c_dim[1] * shape_a_dim[1],
        ),
    )

    check_cublas_error(cublasDestroy(cublas_handle))

    # Retain our buffers till the end.
    _ = buffer_a^
    _ = buffer_b^
    _ = buffer_c^


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
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_cublas: Bool,
](
    ctx: DeviceContext, inout b: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    var dynamic_b_shape = (n.value, k.value) if transpose_b else (
        k.value,
        n.value,
    )

    bench_matmul[
        dtype,
        DimList(m.dim, n.dim),
        DimList(m.dim, k.dim),
        static_b_shape,
        transpose_b=transpose_b,
        cache_busting=cache_busting,
        use_cublas=use_cublas,
    ](ctx, b, (m.value, n.value), (m.value, k.value), dynamic_b_shape)


fn compile_matmul_bench[
    dtype: DType, *, cache_busting: Bool = False
](
    ctx: DeviceContext, inout b: Bench, m: ValOrDim, n: ValOrDim, k: ValOrDim
) raises:
    var s: String = "type=" + str(dtype) + "/m=" + str(m.value) + ", n=" + str(
        n.value
    ) + ", k=" + str(k.value)
    # Note: important to pass list of BenchMetric's used by the computational benchmark (in this case, BenchMetric.elements)
    bench_compile_time[
        bench_matmul[
            dtype,
            DimList(m.dim, n.dim),
            DimList(m.dim, k.dim),
            DimList(k.dim, n.dim),
            cache_busting=cache_busting,
            use_cublas=False,
        ]
    ](b, "matmul/" + s)


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    alias M = env_get_int["M", 1]()
    alias N = env_get_int["N", 1]()
    alias K = env_get_int["K", 1]()

    alias cache_busting = env_get_int["cache_busting", False]()
    alias transpose_b = env_get_int["transpose_b", True]()

    var m = Bench()
    try:
        with DeviceContext() as ctx:
            # benchmarking matmul
            create_matmul_bench[
                dtype,
                transpose_b=transpose_b,
                cache_busting=cache_busting,
                use_cublas=False,
            ](
                ctx,
                m,
                dynamic(M),
                static[N](),
                static[K](),
            )

            create_matmul_bench[
                dtype,
                transpose_b=transpose_b,
                cache_busting=cache_busting,
                use_cublas=True,
            ](
                ctx,
                m,
                dynamic(M),
                static[N](),
                static[K](),
            )

            # benchmarking compilation time of matmul
            compile_matmul_bench[dtype, cache_busting=cache_busting](
                ctx,
                m,
                dynamic(M),
                static[N](),
                static[K](),
            )

    except e:
        print("CUDA_ERROR:", e)

    m.dump_report()
