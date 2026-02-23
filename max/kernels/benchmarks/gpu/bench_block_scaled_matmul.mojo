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
    arg_parse,
    assert_almost_equal,
    assert_with_measure,
    pytorch_like_tolerances_for,
)
from internal_utils._measure import relative_difference
from memory import LegacyUnsafePointer, bitcast
from linalg.fp4_quantization import block_scaled_matmul

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from random import rand, Random
from internal_utils._utils import (
    InitializationType,
    ValOrDim,
    dynamic,
    init_vector_launch,
    static,
)
from layout import Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
)
from linalg.matmul.gpu import _matmul_gpu
from linalg.utils import elementwise_compute_lambda_type
from utils import IndexList
from gpu.host.info import B200


# GPU kernel to initialize MXFP8 scale buffers with random exponents.
# float8_e8m0fnu: exponent-only format, value = 2^(stored_value - 127).
# Random exponents 127 + (0,1,2,3) -> scale values of 1, 2, 4, 8.
# Each thread processes 4 elements for better memory throughput.
fn _init_block_scaled_scales_gpu[
    dtype: DType
](x: UnsafePointer[Scalar[dtype]], len: Int):
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x

    @parameter
    fn apply(values: SIMD[dtype, 4]):
        comptime for i in range(4):
            comptime if i == 3:
                if tid >= UInt(len):
                    return
            x[tid] = Scalar[dtype](values[i])
            tid += stride

    # Generate 4 random exponents per thread for better throughput.
    # step_uniform returns SIMD[float32, 4] with values in [0, 1).
    # Multiply by 4 and cast to get values 0, 1, 2, or 3.
    # Then add 127 to get exponents -> scale values of 1, 2, 4, 8.
    var rng = Random(offset=UInt64(tid))

    comptime if dtype == DType.float8_e8m0fnu:
        var rand_floats = rng.step_uniform() * 4
        var rand_u8 = rand_floats.cast[DType.uint8]() & 3
        var values = bitcast[dtype, 4](rand_u8 + 127)
        apply(values)
    else:
        var values = SIMD[dtype, 4](rng.step_uniform())
        apply(values)


fn _init_block_scaled_scales_launch[
    dtype: DType, block_dim: Int = 256
](out_device: DeviceBuffer[dtype], length: Int, context: DeviceContext,) raises:
    var num_blocks = ceildiv(ceildiv(length, 4), block_dim)
    # using num-threads = 1/4th of length to initialize the array

    comptime kernel = _init_block_scaled_scales_gpu[dtype]
    context.enqueue_function_experimental[kernel](
        out_device,
        length,
        grid_dim=(num_blocks),
        block_dim=(block_dim),
    )


fn _get_run_name[
    dtype: DType,
    shape_c: DimList,
    shape_a: DimList,
    shape_b: DimList,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    micro_scaling_mode: StaticString = "",
](M: Int, N: Int, K: Int,) -> String:
    var vendor_str = "vendor_matmul" if use_vendor_blas else "matmul"
    var type_str = String("(", micro_scaling_mode, "_", String(dtype), ") : ")
    # M
    var m_str = String(M, "_dynamic")
    # N
    var n_str = String(
        N,
        "_dynamic" if shape_c.at[1]().is_dynamic() else "",
    )
    # K
    var k_str = String(
        K,
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
    micro_scaling_mode: StaticString = "",
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
    var stride_a = align_up(get_size(shape_a_dim), simd_size)
    var stride_b = align_up(get_size(shape_b_dim), simd_size)
    var stride_c = align_up(get_size(shape_c_dim), simd_size)

    comptime k128 = 512 * 1024 * 1024
    var cache_a = (
        align_up(k128, stride_a * size_of[dtype]()) // size_of[dtype]()
    )
    var cache_b = (
        align_up(k128, stride_b * size_of[dtype]()) // size_of[dtype]()
    )
    var cache_c = (
        align_up(k128, stride_c * size_of[DType.bfloat16]())
        // size_of[DType.bfloat16]()
    )

    var buffer_a = ctx.enqueue_create_buffer[dtype](cache_a)
    var buffer_b = ctx.enqueue_create_buffer[dtype](cache_b)
    var buffer_c = ctx.enqueue_create_buffer[DType.bfloat16](cache_c)
    var buffer_c_ref = ctx.enqueue_create_buffer[DType.bfloat16](stride_c)

    # MXFP8 scale buffer allocation
    comptime scales_type = MXFP8_SF_DTYPE if micro_scaling_mode == "mxfp8" else NVFP4_SF_DTYPE
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE if micro_scaling_mode == "mxfp8" else NVFP4_SF_VECTOR_SIZE

    # M, N, K dimensions for scales calculation
    var M = shape_c_dim[0]
    var N = shape_c_dim[1]
    comptime K = shape_a.at[
        1
    ]() * 2 if micro_scaling_mode == "nvfp4" else shape_a.at[1]()

    # Calculate scale buffer shapes - 5D tensors for MXFP8 format
    comptime static_a_scales_shape = DimList(
        ceildiv(shape_a.at[0](), SF_MN_GROUP_SIZE),
        ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    comptime static_b_scales_shape = DimList(
        ceildiv(shape_b.at[0](), SF_MN_GROUP_SIZE),
        ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var dynamic_a_scales_shape = DimList(
        ceildiv(M, SF_MN_GROUP_SIZE),
        ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )
    var dynamic_b_scales_shape = DimList(
        ceildiv(N, SF_MN_GROUP_SIZE),
        ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K),
        Dim(SF_ATOM_M[0]),
        Dim(SF_ATOM_M[1]),
        Dim(SF_ATOM_K),
    )

    var a_scales_size = (
        ceildiv(M, SF_MN_GROUP_SIZE)
        * ceildiv(Int(K), SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_size = (
        ceildiv(N, SF_MN_GROUP_SIZE)
        * ceildiv(Int(K), SF_VECTOR_SIZE * SF_ATOM_K)
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var buffer_a_scales = ctx.enqueue_create_buffer[scales_type](a_scales_size)
    var buffer_b_scales = ctx.enqueue_create_buffer[scales_type](b_scales_size)

    _init_block_scaled_scales_launch[scales_type](
        buffer_a_scales, a_scales_size, ctx
    )
    _init_block_scaled_scales_launch[scales_type](
        buffer_b_scales, b_scales_size, ctx
    )

    # Host allocations
    var a_host_ptr = UnsafePointer[Scalar[dtype]].alloc(cache_a)
    var b_host_ptr = UnsafePointer[Scalar[dtype]].alloc(cache_b)

    # TODO: remove init_on_gpu flag and the loading on CPU
    comptime init_on_gpu = True

    comptime if not init_on_gpu:
        var a_host = NDBuffer[dtype, 1](a_host_ptr, cache_a)
        var b_host = NDBuffer[dtype, 1](b_host_ptr, cache_b)

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

        ctx.enqueue_copy(buffer_a, a_host_ptr)
        ctx.enqueue_copy(buffer_b, b_host_ptr)
        ctx.synchronize()
    else:
        init_vector_launch[dtype](buffer_a, cache_a, init_type, ctx)
        init_vector_launch[dtype](buffer_b, cache_b, init_type, ctx)

    # Helper to run vendor BLAS matmul - used by both benchmark and verification
    @parameter
    @__copy_capture(dynamic_a_scales_shape, dynamic_b_scales_shape)
    fn run_vendor_blas(
        ctx: DeviceContext,
        tensor_a: NDBuffer[dtype, 2, MutAnyOrigin, shape_a],
        tensor_b: NDBuffer[dtype, 2, MutAnyOrigin, shape_b],
        tensor_c: NDBuffer[DType.bfloat16, 2, MutAnyOrigin, shape_c],
    ) raises:
        var a_scales_nd = NDBuffer[
            scales_type, 5, MutAnyOrigin, static_a_scales_shape
        ](buffer_a_scales.unsafe_ptr(), dynamic_a_scales_shape)
        var b_scales_nd = NDBuffer[
            scales_type, 5, MutAnyOrigin, static_b_scales_shape
        ](buffer_b_scales.unsafe_ptr(), dynamic_b_scales_shape)

        var a_tensor = from_ndbuffer_row_major(tensor_a)
        var b_tensor = from_ndbuffer_row_major(tensor_b)
        var c_tensor = from_ndbuffer_row_major(tensor_c)
        var a_scales = from_ndbuffer_row_major(a_scales_nd)
        var b_scales = from_ndbuffer_row_major(b_scales_nd)

        vendor_blas.matmul[scales_type=scales_type](
            ctx,
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales=a_scales.get_immutable(),
            b_scales=b_scales.get_immutable(),
            transpose_b=True,
            c_row_major=True,
        )

    @parameter
    @__copy_capture(
        cache_a,
        cache_b,
        cache_c,
        stride_a,
        stride_b,
        stride_c,
        a_scales_size,
        b_scales_size,
        dynamic_a_scales_shape,
        dynamic_b_scales_shape,
    )
    @always_inline
    fn bench_func(mut b: Bencher):
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var offset_a = 0
            var offset_b = 0
            var offset_c = 0

            comptime if cache_busting:
                offset_a = (iteration * stride_a) % cache_a
                offset_b = (iteration * stride_b) % cache_b
                offset_c = (iteration * stride_c) % cache_c
            var tensor_a = NDBuffer[dtype, 2, MutAnyOrigin, shape_a](
                buffer_a.unsafe_ptr() + offset_a, shape_a_dim
            )
            var tensor_b = NDBuffer[dtype, 2, MutAnyOrigin, shape_b](
                buffer_b.unsafe_ptr() + offset_b, shape_b_dim
            )
            var tensor_c = NDBuffer[DType.bfloat16, 2, MutAnyOrigin, shape_c](
                buffer_c.unsafe_ptr() + offset_c, shape_c_dim
            )

            var a_scales_nd = NDBuffer[
                scales_type, 5, MutAnyOrigin, static_a_scales_shape
            ](buffer_a_scales.unsafe_ptr(), dynamic_a_scales_shape)
            var b_scales_nd = NDBuffer[
                scales_type, 5, MutAnyOrigin, static_b_scales_shape
            ](buffer_b_scales.unsafe_ptr(), dynamic_b_scales_shape)

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
                block_scaled_matmul[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    transpose_b=transpose_b,
                ](
                    tensor_c,
                    tensor_a,
                    tensor_b,
                    a_scales_nd,
                    b_scales_nd,
                    1.0,
                    ctx,
                )

        b.iter_custom[kernel_launch](ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
        2 * Int(M) * Int(N) * Int(K),
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
                micro_scaling_mode=micro_scaling_mode,
            ](Int(M), Int(N), Int(K))
        ),
        # TODO: Pick relevant benchmetric
        [flops],
    )

    # Verification: compare our kernel output against vendor BLAS as reference.
    # The benchmark already wrote our kernel's output to buffer_c at offset 0
    # (iteration 0 uses offset 0), so we just need to run vendor BLAS once.
    comptime if not use_vendor_blas and not epilogue:
        if verify:
            # Create tensors at offset 0 for verification
            var tensor_a = NDBuffer[dtype, 2, MutAnyOrigin, shape_a](
                buffer_a.unsafe_ptr(), shape_a_dim
            )
            var tensor_b = NDBuffer[dtype, 2, MutAnyOrigin, shape_b](
                buffer_b.unsafe_ptr(), shape_b_dim
            )
            var tensor_c_ref = NDBuffer[
                DType.bfloat16, 2, MutAnyOrigin, shape_c
            ](buffer_c_ref.unsafe_ptr(), shape_c_dim)

            # Run vendor BLAS to get reference output
            run_vendor_blas(ctx, tensor_a, tensor_b, tensor_c_ref)
            ctx.synchronize()

            # Copy results to host for comparison
            # Create non-owning DeviceBuffers with exact size for the copy
            var c_size = shape_c_dim[0] * shape_c_dim[1]
            var c_host = UnsafePointer[Scalar[DType.bfloat16]].alloc(c_size)
            var c_ref_host = UnsafePointer[Scalar[DType.bfloat16]].alloc(c_size)
            var c_view = DeviceBuffer[DType.bfloat16](
                ctx, buffer_c.unsafe_ptr(), c_size, owning=False
            )
            var c_ref_view = DeviceBuffer[DType.bfloat16](
                ctx, buffer_c_ref.unsafe_ptr(), c_size, owning=False
            )
            ctx.enqueue_copy(c_host, c_view)
            ctx.enqueue_copy(c_ref_host, c_ref_view)
            ctx.synchronize()

            # Sanity check: verify outputs match expected zero/non-zero state
            fn is_all_zeros(
                ptr: UnsafePointer[Scalar[DType.bfloat16]], size: Int
            ) -> Bool:
                for i in range(size):
                    if ptr[i] != 0:
                        return False
                return True

            var c_is_zeros = is_all_zeros(c_host, c_size)
            var c_ref_is_zeros = is_all_zeros(c_ref_host, c_size)

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
                c_host,
                c_ref_host,
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
                c_host,
                c_ref_host,
                c_size,
                msg="matmul verification failed",
                rtol=rtol,
                atol=atol,
            )

            c_host.free()
            c_ref_host.free()

    # Cleanup host pointers
    a_host_ptr.free()
    b_host_ptr.free()

    # Consume device buffers
    _ = buffer_a^
    _ = buffer_b^
    _ = buffer_c^
    _ = buffer_c_ref^
    _ = buffer_a_scales^
    _ = buffer_b_scales^


fn create_matmul_bench[
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    epilogue: Bool,
    register_based_epilogue: Bool,
    micro_scaling_mode: StaticString = "",
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
        micro_scaling_mode=micro_scaling_mode,
    ](
        ctx,
        b,
        (m.value, n.value),
        (m.value, k.value),
        dynamic_b_shape,
        init_type,
        verify,
    )


fn get_dtype[micro_scaling_mode: StaticString]() -> DType:
    comptime assert (
        micro_scaling_mode == "mxfp8" or micro_scaling_mode == "nvfp4"
    ), "invalid micro_scaling_mode"

    comptime if micro_scaling_mode == "mxfp8":  # micro_scaling_mode == "mxfp8"
        return DType.float8_e4m3fn
    else:  # micro_scaling_mode == "nvfp4"
        return DType.uint8


def main():
    comptime micro_scaling_mode = env_get_string["scaling_mode", "nvfp4"]()
    comptime dtype = get_dtype[micro_scaling_mode]()

    var M = Int(arg_parse("M", 1))
    comptime N = env_get_int["N", 1]()
    comptime K = env_get_int[
        "K", 2
    ]() // 2 if micro_scaling_mode == "nvfp4" else env_get_int["K", 1]()

    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    var verify = arg_parse("verify", False)
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
            micro_scaling_mode=micro_scaling_mode,
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
