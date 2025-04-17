# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-build-no-debug-no-assert %s


from math import align_up
from sys import env_get_bool, env_get_dtype, env_get_int, sizeof

import linalg.vendor_blas
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from buffer.dimlist import _make_tuple
from gpu._cublas.cublas import cublasContext, cublasCreate, cublasDestroy
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host.info import DEFAULT_GPU_ARCH
from internal_utils import DeviceNDBuffer, HostNDBuffer, arg_parse
from internal_utils._utils import (
    InitializationType,
    ValOrDim,
    dynamic,
    initialize,
    static,
    random_float8,
)
from linalg.matmul_gpu import _matmul_gpu
from memory import UnsafePointer

from utils import IndexList


fn _get_run_name[
    type: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
](
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
) -> String:
    var vendor_str = "vendor_matmul" if use_vendor_blas else String("matmul")
    var type_str = String("(", type, ") : ")
    # M
    var m_str = String(shape_c_dim[0], "_dynamic")
    # N
    var n_str = String(
        shape_c_dim[1],
        "_dynamic" if shape_c.at[1]().is_dynamic() else StaticString(""),
    )
    # K
    var k_str = String(
        shape_a_dim[1],
        "_dynamic" if shape_a.at[1]().is_dynamic() else StaticString(""),
    )

    var transpose_b_str = String(
        "/transpose_b=", "True" if transpose_b else StaticString("False")
    )
    var cache_busting_str = String(
        "/cache_busting=", "True" if cache_busting else StaticString("False")
    )
    return String(
        vendor_str,
        type_str,
        m_str,
        " x ",
        n_str,
        " x ",
        k_str,
        transpose_b_str,
        cache_busting_str,
    )


fn bench_matmul[
    dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    *,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    transpose_b: Bool = False,
](
    ctx: DeviceContext,
    handle: vendor_blas.Handle,
    mut b: Bench,
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
    init_type: InitializationType,
) raises:
    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    # update: using 512 to be 2x the infinity cache on MI300x
    @always_inline
    fn get_size(shape: IndexList[2]) -> Int:
        return shape[0] * shape[1]

    alias simd_size = 4
    var stride_a = align_up(get_size(shape_a_dim), simd_size)
    var stride_b = align_up(get_size(shape_b_dim), simd_size)
    var stride_c = align_up(get_size(shape_c_dim), simd_size)

    alias k128 = 512 * 1024 * 1024
    var cache_a = align_up(k128, stride_a * sizeof[dtype]()) // sizeof[dtype]()
    var cache_b = align_up(k128, stride_b * sizeof[dtype]()) // sizeof[dtype]()
    var cache_c = align_up(k128, stride_c * sizeof[DType.bfloat16]()) // sizeof[
        DType.bfloat16
    ]()

    var buffer_a = ctx.enqueue_create_buffer[dtype](cache_a)
    var buffer_b = ctx.enqueue_create_buffer[dtype](cache_b)
    var buffer_c = ctx.enqueue_create_buffer[DType.bfloat16](cache_c)

    var a_host = HostNDBuffer[dtype, 1](DimList(cache_a))
    var b_host = HostNDBuffer[dtype, 1](DimList(cache_b))

    @parameter
    if dtype == DType.float8_e4m3fn:
        random_float8(a_host.tensor)
        random_float8(b_host.tensor)
    else:
        initialize(a_host.tensor, init_type)
        initialize(b_host.tensor, init_type)

    ctx.enqueue_copy(buffer_a, a_host.tensor.data)
    ctx.enqueue_copy(buffer_b, b_host.tensor.data)
    ctx.synchronize()

    @parameter
    @__copy_capture(
        cache_a, cache_b, cache_c, stride_a, stride_b, stride_c, handle
    )
    @always_inline
    fn bench_func(mut b: Bencher):
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
            var tensor_a = NDBuffer[dtype, 2, MutableAnyOrigin, shape_a](
                buffer_a.unsafe_ptr() + offset_a, shape_a_dim
            )
            var tensor_b = NDBuffer[dtype, 2, MutableAnyOrigin, shape_b](
                buffer_b.unsafe_ptr() + offset_b, shape_b_dim
            )
            var tensor_c = NDBuffer[
                DType.bfloat16, 2, MutableAnyOrigin, shape_c
            ](buffer_c.unsafe_ptr() + offset_c, shape_c_dim)

            @parameter
            if use_vendor_blas:
                vendor_blas.matmul[use_tf32=True](
                    ctx,
                    handle,
                    tensor_c,
                    tensor_a,
                    tensor_b,
                    c_row_major=True,
                    transpose_b=transpose_b,
                )

            else:
                _matmul_gpu[
                    use_tensor_core=True,
                    transpose_b=transpose_b,
                ](tensor_c, tensor_a, tensor_b, ctx)

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
                use_vendor_blas=use_vendor_blas,
            ](shape_c_dim, shape_a_dim, shape_b_dim)
        ),
        # TODO: Pick relevant benchmetric
        ThroughputMeasure(
            BenchMetric.flops,
            # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
            2 * shape_c_dim[0] * shape_c_dim[1] * shape_a_dim[1],
        ),
    )

    # Retain our buffers till the end.
    _ = buffer_a^
    _ = buffer_b^
    _ = buffer_c^
    _ = a_host^
    _ = b_host^


fn create_matmul_bench[
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
](
    ctx: DeviceContext,
    handle: vendor_blas.Handle,
    mut b: Bench,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    init_type: InitializationType,
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
        use_vendor_blas=use_vendor_blas,
    ](
        ctx,
        handle,
        b,
        (m.value, n.value),
        (m.value, k.value),
        dynamic_b_shape,
        init_type,
    )


fn main() raises:
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()

    var M = Int(arg_parse("M", 1))
    alias N = env_get_int["N", 1]()
    alias K = env_get_int["K", 1]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    alias cache_busting = True
    alias transpose_b = True
    alias use_vendor_blas = env_get_bool["use_vendor_blas", False]()

    var m = Bench()
    with DeviceContext() as ctx:

        @parameter
        if dtype is DType.float8_e4m3fn:
            with vendor_blas.Handle[vendor_blas.Backend.CUBLASLT]() as handle:
                create_matmul_bench[
                    dtype,
                    transpose_b=transpose_b,
                    cache_busting=cache_busting,
                    use_vendor_blas=use_vendor_blas,
                ](
                    ctx,
                    handle,
                    m,
                    dynamic(M),
                    static[N](),
                    static[K](),
                    init_type,
                )

        else:
            with vendor_blas.Handle() as handle:
                create_matmul_bench[
                    dtype,
                    transpose_b=transpose_b,
                    cache_busting=cache_busting,
                    use_vendor_blas=use_vendor_blas,
                ](
                    ctx,
                    handle,
                    m,
                    dynamic(M),
                    static[N](),
                    static[K](),
                    init_type,
                )

    m.dump_report()
