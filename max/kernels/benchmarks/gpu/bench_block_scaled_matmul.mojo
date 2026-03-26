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

from std.math import align_up, ceildiv
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
    has_nvidia_gpu_accelerator,
    size_of,
    align_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu import global_idx, grid_dim, block_dim
from std.gpu.host import DeviceBuffer, DeviceContext
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    assert_almost_equal,
    assert_with_measure,
    pytorch_like_tolerances_for,
)
from internal_utils._measure import relative_difference
from std.memory import bitcast
from linalg.fp4_quantization import block_scaled_matmul

from std.random import rand, Random
from internal_utils._utils import InitializationType
from layout import (
    CoordLike,
    Coord,
    Layout,
    LayoutTensor,
    TileTensor,
    Idx,
    row_major,
)
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
from std.utils import IndexList
from std.gpu.host.info import B200


# GPU kernel to initialize MXFP8 scale buffers with random exponents.
# float8_e8m0fnu: exponent-only format, value = 2^(stored_value - 127).
# Random exponents 127 + (0,1,2,3) -> scale values of 1, 2, 4, 8.
# Each thread processes 4 elements for better memory throughput.
def _init_block_scaled_scales_gpu[
    dtype: DType
](x: UnsafePointer[Scalar[dtype], MutAnyOrigin], len: Int):
    var tid = global_idx.x
    var stride = grid_dim.x * block_dim.x

    @parameter
    def apply(values: SIMD[dtype, 4]):
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


def _init_block_scaled_scales_launch[
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


def _get_run_name[
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    micro_scaling_mode: StaticString = "",
](
    M: Int,
    N: Int,
    K: Int,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
) -> String:
    var vendor_str = "vendor_matmul" if use_vendor_blas else "matmul"
    var type_str = String("(", micro_scaling_mode, "_", String(dtype), ") : ")
    # M
    var m_str = String(M, "_dynamic")
    # N
    var n_str = String(
        N,
        "_dynamic" if not shape_c.element_types[1].is_static_value else "",
    )
    # K
    var k_str = String(
        K,
        "_dynamic" if not shape_a.element_types[1].is_static_value else "",
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


def bench_matmul[
    dtype: DType,
    *,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    transpose_b: Bool = False,
    epilogue: Bool = False,
    micro_scaling_mode: StaticString = "",
](
    ctx: DeviceContext,
    mut b: Bench,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
    init_type: InitializationType,
    verify: Bool,
) raises:
    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    # update: using 512 to be 2x the infinity cache on MI300x
    @always_inline
    def get_size(shape: Coord) -> Int:
        return shape[0].value() * shape[1].value()

    comptime simd_size = 4
    var cb_a = CacheBustingBuffer[dtype](get_size(shape_a), simd_size, ctx)
    var cb_b = CacheBustingBuffer[dtype](get_size(shape_b), simd_size, ctx)
    var cb_c = CacheBustingBuffer[DType.bfloat16](
        get_size(shape_c), simd_size, ctx
    )
    var buffer_c_ref = ctx.enqueue_create_buffer[DType.bfloat16](cb_c.stride)

    # MXFP8 scale buffer allocation
    comptime scales_type = MXFP8_SF_DTYPE if micro_scaling_mode == "mxfp8" else NVFP4_SF_DTYPE
    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE if micro_scaling_mode == "mxfp8" else NVFP4_SF_VECTOR_SIZE

    # M, N, K dimensions for scales calculation
    var M = shape_c[0].value()
    var N = shape_c[1].value()
    comptime K = shape_a.element_types[
        1
    ].static_value * 2 if micro_scaling_mode == "nvfp4" else shape_a.element_types[
        1
    ].static_value

    # Calculate scale buffer shapes - 5D tensors for MXFP8 format
    var a_scales_shape = Coord(
        Idx(ceildiv(shape_a[0].value(), SF_MN_GROUP_SIZE)),
        Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)](),
        Idx[SF_ATOM_M[0]](),
        Idx[SF_ATOM_M[1]](),
        Idx[SF_ATOM_K](),
    )
    var b_scales_shape = Coord(
        Idx[ceildiv(shape_b.element_types[0].static_value, SF_MN_GROUP_SIZE)](),
        Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)](),
        Idx[SF_ATOM_M[0]](),
        Idx[SF_ATOM_M[1]](),
        Idx[SF_ATOM_K](),
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
    var a_host_ptr = alloc[Scalar[dtype]](cb_a.alloc_size())
    var b_host_ptr = alloc[Scalar[dtype]](cb_b.alloc_size())

    # TODO: remove init_on_gpu flag and the loading on CPU
    comptime init_on_gpu = True

    comptime if not init_on_gpu:
        var a_host = TileTensor(a_host_ptr, row_major(Idx(cb_a.alloc_size())))
        var b_host = TileTensor(b_host_ptr, row_major(Idx(cb_b.alloc_size())))

        comptime if dtype.is_float8():
            rand(a_host.ptr, a_host.num_elements())
            rand(b_host.ptr, b_host.num_elements())
        else:
            if init_type == InitializationType.zero:
                _ = a_host.fill(0)
                _ = b_host.fill(0)
            elif init_type == InitializationType.one:
                _ = a_host.fill(1)
                _ = b_host.fill(1)
            elif init_type == InitializationType.uniform_distribution:
                rand(a_host.ptr, a_host.num_elements())
                rand(b_host.ptr, b_host.num_elements())
            elif init_type == InitializationType.arange:
                for i in range(a_host.num_elements()):
                    a_host.ptr[i] = Scalar[dtype](i)
                for i in range(b_host.num_elements()):
                    b_host.ptr[i] = Scalar[dtype](i)

        ctx.enqueue_copy(cb_a.device_buffer(), a_host_ptr)
        ctx.enqueue_copy(cb_b.device_buffer(), b_host_ptr)
        ctx.synchronize()
    else:
        cb_a.init_on_device(init_type, ctx)
        cb_b.init_on_device(init_type, ctx)

    # Helper to run vendor BLAS matmul - used by both benchmark and verification
    @parameter
    @__copy_capture(a_scales_shape, b_scales_shape)
    def run_vendor_blas(
        ctx: DeviceContext,
        a: TileTensor[dtype, ...],
        b: TileTensor[dtype, ...],
        c: TileTensor[mut=True, DType.bfloat16, ...],
    ) raises:
        var a_scales = TileTensor(buffer_a_scales, row_major(a_scales_shape))
        var b_scales = TileTensor(buffer_b_scales, row_major(b_scales_shape))

        vendor_blas.matmul[scales_type=scales_type](
            ctx,
            c,
            a,
            b,
            a_scales=a_scales.as_immut(),
            b_scales=b_scales.as_immut(),
            transpose_b=True,
            c_row_major=True,
        )

    @parameter
    @__copy_capture(
        cb_a,
        cb_b,
        cb_c,
        a_scales_size,
        b_scales_size,
        a_scales_shape,
        b_scales_shape,
    )
    @always_inline
    def bench_func(mut b: Bencher):
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
            var tensor_a = TileTensor(
                cb_a.offset_ptr(iteration), row_major(shape_a)
            )
            var tensor_b = TileTensor(
                cb_b.offset_ptr(iteration), row_major(shape_b)
            )
            var tensor_c = TileTensor(
                cb_c.offset_ptr(iteration), row_major(shape_c)
            )

            var a_scales_nd = TileTensor(
                buffer_a_scales.unsafe_ptr(), row_major(a_scales_shape)
            )
            var b_scales_nd = TileTensor(
                buffer_b_scales.unsafe_ptr(), row_major(b_scales_shape)
            )

            @parameter
            @always_inline
            @__copy_capture(tensor_c)
            def test_lambda_add_coords_prod[
                _dtype: DType,
                width: Int,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                comptime assert tensor_c.flat_rank >= 2
                var x = tensor_c.load[width=width](Coord(idx)).cast[_dtype]()
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
                transpose_b=transpose_b,
                cache_busting=cache_busting,
                use_vendor_blas=use_vendor_blas,
                micro_scaling_mode=micro_scaling_mode,
            ](
                Int(M),
                Int(N),
                Int(K),
                shape_c,
                shape_a,
                shape_b,
            )
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
            var tensor_a = TileTensor(cb_a.unsafe_ptr(), row_major(shape_a))
            var tensor_b = TileTensor(cb_b.unsafe_ptr(), row_major(shape_b))
            var tensor_c_ref = TileTensor(
                buffer_c_ref.unsafe_ptr(), row_major(shape_c)
            )

            # Run vendor BLAS to get reference output
            run_vendor_blas(ctx, tensor_a, tensor_b, tensor_c_ref)
            ctx.synchronize()

            # Copy results to host for comparison
            # Create non-owning DeviceBuffers with exact size for the copy
            var c_size = shape_c[0].value() * shape_c[1].value()
            var c_host = alloc[Scalar[DType.bfloat16]](c_size)
            var c_ref_host = alloc[Scalar[DType.bfloat16]](c_size)
            var c_view = DeviceBuffer[DType.bfloat16](
                ctx, cb_c.unsafe_ptr(), c_size, owning=False
            )
            var c_ref_view = DeviceBuffer[DType.bfloat16](
                ctx, buffer_c_ref.unsafe_ptr(), c_size, owning=False
            )
            ctx.enqueue_copy(c_host, c_view)
            ctx.enqueue_copy(c_ref_host, c_ref_view)
            ctx.synchronize()

            # Sanity check: verify outputs match expected zero/non-zero state
            def is_all_zeros(
                ptr: UnsafePointer[Scalar[DType.bfloat16], _], size: Int
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


def create_matmul_bench[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    epilogue: Bool,
    micro_scaling_mode: StaticString = "",
](
    ctx: DeviceContext,
    mut b: Bench,
    m: MType,
    n: NType,
    k: KType,
    init_type: InitializationType,
    verify: Bool,
) raises:
    var b_shape = Coord(
        Idx[NType.static_value if transpose_b else KType.static_value](),
        Idx[KType.static_value if transpose_b else NType.static_value](),
    )

    bench_matmul[
        dtype,
        transpose_b=transpose_b,
        cache_busting=cache_busting,
        use_vendor_blas=use_vendor_blas,
        epilogue=epilogue,
        micro_scaling_mode=micro_scaling_mode,
    ](
        ctx,
        b,
        Coord(m, n),
        Coord(m, k),
        b_shape,
        init_type,
        verify,
    )


def get_dtype[micro_scaling_mode: StaticString]() -> DType:
    comptime assert (
        micro_scaling_mode == "mxfp8" or micro_scaling_mode == "nvfp4"
    ), "invalid micro_scaling_mode"

    comptime if micro_scaling_mode == "mxfp8":  # micro_scaling_mode == "mxfp8"
        return DType.float8_e4m3fn
    else:  # micro_scaling_mode == "nvfp4"
        return DType.uint8


def main() raises:
    comptime micro_scaling_mode = get_defined_string["scaling_mode", "nvfp4"]()
    comptime dtype = get_dtype[micro_scaling_mode]()

    var M = Int(arg_parse("M", 1))
    comptime N = get_defined_int["N", 1]()
    comptime K = get_defined_int[
        "K", 2
    ]() // 2 if micro_scaling_mode == "nvfp4" else get_defined_int["K", 1]()

    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    var verify = arg_parse("verify", False)
    comptime cache_busting = True
    comptime transpose_b = True
    comptime use_vendor_blas = get_defined_bool["use_vendor_blas", False]()
    comptime epilogue = get_defined_bool["epilogue", False]()

    var m = Bench()
    with DeviceContext() as ctx:
        create_matmul_bench[
            dtype,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            use_vendor_blas=use_vendor_blas,
            epilogue=epilogue,
            micro_scaling_mode=micro_scaling_mode,
        ](
            ctx,
            m,
            Idx(M),
            Idx[N](),
            Idx[K](),
            init_type,
            verify,
        )

    m.dump_report()
