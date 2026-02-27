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

from math import align_up, ceildiv
from sys import (
    env_get_bool,
    env_get_dtype,
    env_get_int,
    env_get_string,
    has_nvidia_gpu_accelerator,
    size_of,
    align_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from buffer import Dim, DimList, NDBuffer
from gpu import global_idx, grid_dim, block_dim
from gpu.host import DeviceBuffer, DeviceContext
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    assert_almost_equal,
    assert_with_measure,
    pytorch_like_tolerances_for,
)
from internal_utils._measure import relative_difference
from memory import LegacyUnsafePointer, bitcast

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from random import rand, Random
from internal_utils._utils import (
    InitializationType,
    init_vector_launch,
    ValOrDim,
    dynamic,
    static,
)
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.matmul.gpu import _matmul_gpu
from linalg.utils import elementwise_compute_lambda_type
from utils import IndexList


fn verify_matmul[
    dtype: DType,
    static_c_shape: DimList,
    static_a_shape: DimList,
    static_b_shape: DimList,
    *,
    transpose_b: Bool = False,
    init_on_gpu: Bool = True,
](
    ctx: DeviceContext,
    dynamic_c_shape: IndexList[2],
    dynamic_a_shape: IndexList[2],
    dynamic_b_shape: IndexList[2],
    init_type: InitializationType,
) raises:
    comptime c_type = DType.bfloat16

    var c_size = dynamic_c_shape[0] * dynamic_c_shape[1]
    var a_size = dynamic_a_shape[0] * dynamic_a_shape[1]
    var b_size = dynamic_b_shape[0] * dynamic_b_shape[1]

    var c_host_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ptr, dynamic_c_shape
    )
    var c_host_ref_ptr = UnsafePointer[Scalar[c_type]].alloc(c_size)
    var c_host_ref = NDBuffer[c_type, 2, _, static_c_shape](
        c_host_ref_ptr, dynamic_c_shape
    )

    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(a_size)
    var a_host = NDBuffer[dtype, 2, _, static_a_shape](
        a_host_ptr, dynamic_a_shape
    )
    var b_host_ptr = UnsafePointer[Scalar[dtype]].alloc(b_size)
    var b_host = NDBuffer[dtype, 2, _, static_b_shape](
        b_host_ptr, dynamic_b_shape
    )

    var a_device = ctx.enqueue_create_buffer[dtype](a_size)
    var a_device_nd = NDBuffer[dtype, 2, _, static_a_shape](
        a_device.unsafe_ptr(), dynamic_a_shape
    )
    var b_device = ctx.enqueue_create_buffer[dtype](b_size)
    var b_device_nd = NDBuffer[dtype, 2, _, static_b_shape](
        b_device.unsafe_ptr(), dynamic_b_shape
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device.unsafe_ptr(), dynamic_c_shape
    )
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = NDBuffer[c_type, 2, _, static_c_shape](
        c_device_ref.unsafe_ptr(), dynamic_c_shape
    )

    # Initialize matmul operands
    comptime if not init_on_gpu:
        comptime if dtype.is_float8():
            rand(a_host.data, a_host.num_elements())
            rand(b_host.data, b_host.num_elements())
        else:
            if init_type == InitializationType.zero:
                a_host.zero()
                b_host.zero()
            elif init_type == InitializationType.one:
                a_host.fill(1)
                b_host.fill(1)
            elif init_type == InitializationType.uniform_distribution:
                rand(a_host.data, a_host.num_elements())
                rand(b_host.data, b_host.num_elements())
            elif init_type == InitializationType.arange:
                for i in range(a_host.num_elements()):
                    a_host.data[i] = Scalar[dtype](i)
                for i in range(b_host.num_elements()):
                    b_host.data[i] = Scalar[dtype](i)
        # Move operands to the Device
        ctx.enqueue_copy(a_device, a_host_ptr)
        ctx.enqueue_copy(b_device, b_host_ptr)
    else:
        init_vector_launch[dtype](a_device, a_size, init_type, ctx)
        init_vector_launch[dtype](b_device, b_size, init_type, ctx)

    vendor_blas.matmul[use_tf32=True](
        ctx,
        c_device_ref_nd,
        a_device_nd,
        b_device_nd,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=transpose_b,
    ](c_device_nd, a_device_nd, b_device_nd, ctx)

    ctx.synchronize()

    ctx.enqueue_copy(c_host.data, c_device)
    ctx.enqueue_copy(c_host_ref.data, c_device_ref)
    ctx.synchronize()

    # Sanity check: verify outputs match expected zero/non-zero state
    fn is_all_zeros(
        ptr: UnsafePointer[Scalar[DType.bfloat16]], size: Int
    ) -> Bool:
        for i in range(size):
            if ptr[i] != 0:
                return False
        return True

    var c_is_zeros = is_all_zeros(c_host.data, c_size)
    var c_ref_is_zeros = is_all_zeros(c_host_ref.data, c_size)

    if init_type == InitializationType.zero:
        if not c_is_zeros:
            raise "matmul verification failed: kernel output should be all zeros for zero input"
        if not c_ref_is_zeros:
            raise "matmul verification failed: vendor BLAS output should be all zeros for zero input"
    else:
        if c_is_zeros:
            raise "matmul verification failed: kernel output is all zeros"
        if c_ref_is_zeros:
            raise "matmul verification failed: vendor BLAS output is all zeros"

    # Verify using relative difference measure
    assert_with_measure[relative_difference](
        c_host.data,
        c_host_ref.data,
        c_size,
        msg="matmul verification failed (relative_difference)",
        threshold=0.001,
    )

    # Verify element-wise with dtype-appropriate tolerances
    # float8 needs looser tolerances due to reduced precision
    var rtol: Float64
    var atol: Float64

    comptime if dtype.is_float8():
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol, atol = pytorch_like_tolerances_for[DType.bfloat16]()

    assert_almost_equal(
        c_host.data,
        c_host_ref.data,
        c_host.num_elements(),
        atol=atol,
        rtol=rtol,
    )
    print("\n=== TEST PASSED ===\n")


fn _get_run_name[
    dtype: DType,
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
    var vendor_str = "vendor_matmul" if use_vendor_blas else "matmul"
    var type_str = String("(", String(dtype), ") : ")
    # M
    var m_str = String(shape_c_dim[0], "_dynamic")
    # N
    var n_str = String(
        shape_c_dim[1],
        "_dynamic" if shape_c.at[1]().is_dynamic() else "",
    )
    # K
    var k_str = String(
        shape_a_dim[1],
        "_dynamic" if shape_a.at[1]().is_dynamic() else "",
    )

    var transpose_b_str = String(
        "/transpose_b=", "True" if transpose_b else "False"
    )
    var cache_busting_str = String(
        "/cache_busting=", "True" if cache_busting else "False"
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
    epilogue: Bool = False,
    register_based_epilogue: Bool = False,
](
    ctx: DeviceContext,
    mut b: Bench,
    shape_c_dim: IndexList[2],
    shape_a_dim: IndexList[2],
    shape_b_dim: IndexList[2],
    init_type: InitializationType,
    verify: Bool,
) raises:
    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    # update: using 512 to be 2x the infinity cache on MI300x
    @always_inline
    fn get_size(shape: IndexList[2]) -> Int:
        return shape[0] * shape[1]

    comptime simd_size = 4
    var cb_a = CacheBustingBuffer[dtype](get_size(shape_a_dim), simd_size, ctx)
    var cb_b = CacheBustingBuffer[dtype](get_size(shape_b_dim), simd_size, ctx)
    var cb_c = CacheBustingBuffer[DType.bfloat16](
        get_size(shape_c_dim), simd_size, ctx
    )
    var buffer_c_ref = ctx.enqueue_create_buffer[DType.bfloat16](cb_c.stride)

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(cb_a.alloc_size())
    var b_host_ptr = UnsafePointer[Scalar[dtype]].alloc(cb_b.alloc_size())

    # TODO: remove init_on_gpu flag and the loading on CPU
    comptime init_on_gpu = True

    comptime if not init_on_gpu:
        var a_host = NDBuffer[dtype, 1](a_host_ptr, cb_a.alloc_size())
        var b_host = NDBuffer[dtype, 1](b_host_ptr, cb_b.alloc_size())

        comptime if dtype.is_float8():
            rand(a_host.data, a_host.num_elements())
            rand(b_host.data, b_host.num_elements())
        else:
            if init_type == InitializationType.zero:
                a_host.zero()
                b_host.zero()
            elif init_type == InitializationType.one:
                a_host.fill(1)
                b_host.fill(1)
            elif init_type == InitializationType.uniform_distribution:
                rand(a_host.data, a_host.num_elements())
                rand(b_host.data, b_host.num_elements())
            elif init_type == InitializationType.arange:
                for i in range(a_host.num_elements()):
                    a_host.data[i] = Scalar[dtype](i)
                for i in range(b_host.num_elements()):
                    b_host.data[i] = Scalar[dtype](i)

        ctx.enqueue_copy(cb_a.device_buffer(), a_host_ptr)
        ctx.enqueue_copy(cb_b.device_buffer(), b_host_ptr)
        ctx.synchronize()
    else:
        cb_a.init_on_device(init_type, ctx)
        cb_b.init_on_device(init_type, ctx)

    # Helper to run vendor BLAS matmul - used by both benchmark and verification
    fn run_vendor_blas(
        ctx: DeviceContext,
        tensor_a: NDBuffer[dtype, 2, MutAnyOrigin, shape_a],
        tensor_b: NDBuffer[dtype, 2, MutAnyOrigin, shape_b],
        tensor_c: NDBuffer[DType.bfloat16, 2, MutAnyOrigin, shape_c],
    ) raises:
        vendor_blas.matmul[use_tf32=True](
            ctx,
            tensor_c,
            tensor_a,
            tensor_b,
            c_row_major=True,
            transpose_b=transpose_b,
        )

    @parameter
    @__copy_capture(
        cb_a,
        cb_b,
        cb_c,
    )
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var tensor_a = NDBuffer[dtype, 2, MutAnyOrigin, shape_a](
                cb_a.offset_ptr(iteration), shape_a_dim
            )
            var tensor_b = NDBuffer[dtype, 2, MutAnyOrigin, shape_b](
                cb_b.offset_ptr(iteration), shape_b_dim
            )
            var tensor_c = NDBuffer[DType.bfloat16, 2, MutAnyOrigin, shape_c](
                cb_c.offset_ptr(iteration), shape_c_dim
            )

            @parameter
            @always_inline
            @__copy_capture(tensor_c)
            fn test_lambda_add_coords_prod[
                _dtype: DType,
                width: Int,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                var x = tensor_c.load[width=width](idx).cast[_dtype]()
                var y = val * x
                return y

            comptime optional_lambda_fn = Optional[
                elementwise_compute_lambda_type
            ](test_lambda_add_coords_prod) if epilogue else None

            comptime if use_vendor_blas:
                run_vendor_blas(ctx, tensor_a, tensor_b, tensor_c)
            else:
                _matmul_gpu[
                    use_tensor_core=True,
                    transpose_b=transpose_b,
                    elementwise_compute_lambda_fn=optional_lambda_fn,
                    register_based_epilogue=register_based_epilogue,
                ](tensor_c, tensor_a, tensor_b, ctx)

        b.iter_custom[kernel_launch](ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
        2 * shape_c_dim[0] * shape_c_dim[1] * shape_a_dim[1],
    )
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
        [flops],
    )

    # Verification: compare our kernel output against vendor BLAS as reference.
    # The benchmark already wrote our kernel's output to buffer_c at offset 0
    # (iteration 0 uses offset 0), so we just need to run vendor BLAS once.
    comptime if not use_vendor_blas and not epilogue:
        if verify:
            verify_matmul[
                dtype=dtype,
                static_c_shape=shape_c,
                static_a_shape=shape_a,
                static_b_shape=shape_b,
                transpose_b=transpose_b,
                init_on_gpu=init_on_gpu,
            ](ctx, shape_c_dim, shape_a_dim, shape_b_dim, init_type)

    # Cleanup host pointers
    a_host_ptr.free()
    b_host_ptr.free()

    # Consume device buffers
    _ = cb_a^
    _ = cb_b^
    _ = cb_c^
    _ = buffer_c_ref^


fn create_matmul_bench[
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    epilogue: Bool,
    register_based_epilogue: Bool,
](
    ctx: DeviceContext,
    mut b: Bench,
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    init_type: InitializationType,
    verify: Bool,
) raises:
    comptime static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
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
        epilogue=epilogue,
        register_based_epilogue=register_based_epilogue,
    ](
        ctx,
        b,
        (m.value, n.value),
        (m.value, k.value),
        dynamic_b_shape,
        init_type,
        verify,
    )


def main() raises:
    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()

    var M = Int(arg_parse("M", 128))
    comptime N = env_get_int["N", 128]()
    comptime K = env_get_int["K", 128]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    var verify = arg_parse("verify", True)
    comptime cache_busting = True
    comptime transpose_b = True
    comptime use_vendor_blas = env_get_bool["use_vendor_blas", False]()
    comptime epilogue = env_get_bool["epilogue", False]()
    comptime register_based_epilogue = env_get_bool[
        "register_based_epilogue", True
    ]()

    var m = Bench()
    with DeviceContext() as ctx:
        create_matmul_bench[
            dtype,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            use_vendor_blas=use_vendor_blas,
            epilogue=epilogue,
            register_based_epilogue=register_based_epilogue,
        ](
            ctx,
            m,
            dynamic(M),
            static[N](),
            static[K](),
            init_type,
            verify,
        )

    m.dump_report()
