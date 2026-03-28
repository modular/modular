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

from std.math import ceildiv
from std.sys import (
    get_defined_bool,
    get_defined_int,
    get_defined_string,
    align_of,
)
from std.gpu import (
    global_idx_uint as global_idx,
    grid_dim_uint as grid_dim,
    block_dim_uint as block_dim,
    thread_idx_uint as thread_idx,
    block_idx_uint as block_idx,
)
import linalg.matmul.vendor.blas as vendor_blas
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.gpu.host import DeviceContext
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    pytorch_like_tolerances_for,
)
from internal_utils._measure import relative_difference
from linalg.fp4_quantization import block_scaled_matmul

from layout import (
    CoordLike,
    Coord,
    TileTensor,
    Idx,
    row_major,
)
from internal_utils._utils import (
    InitializationType,
    init_vector_launch,
    _init_block_scaled_scales_launch,
)
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
)
from linalg.utils import elementwise_compute_lambda_type
from std.utils import IndexList
from std.gpu.primitives import block


def _verify_buffers_gpu[
    c_type: DType, BLOCK_SIZE: Int
](
    output: UnsafePointer[Scalar[c_type], ImmutAnyOrigin],
    reference: UnsafePointer[Scalar[c_type], ImmutAnyOrigin],
    length: Int,
    atol: Float32,
    rtol: Float32,
    result: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """GPU kernel that computes verification metrics in one pass.

    Each block computes partial reductions and writes 5 Float32 values:
      [0] abs_diff_sum — for relative difference metric
      [1] abs_ref_sum  — for relative difference metric
      [2] max_violation — max(|x-y| - (atol + rtol*|y|)), <=0 means pass
      [3] out_nz — 1.0 if any output element is nonzero
      [4] ref_nz — 1.0 if any reference element is nonzero
    """
    # Per-thread accumulators
    var abs_diff_sum: Float32 = 0
    var abs_ref_sum: Float32 = 0
    var max_violation = Float32.MIN_FINITE
    var out_nz: Float32 = 0
    var ref_nz: Float32 = 0

    # Grid-stride loop
    var i = UInt(global_idx.x)
    var stride = UInt(grid_dim.x * block_dim.x)
    while i < UInt(length):
        var x = output[i].cast[DType.float32]()
        var y = reference[i].cast[DType.float32]()
        abs_diff_sum += abs(x - y)
        abs_ref_sum += abs(y)
        max_violation = max(max_violation, abs(x - y) - (atol + rtol * abs(y)))
        if x != 0:
            out_nz = 1.0
        if y != 0:
            ref_nz = 1.0
        i += stride

    # Block-wide reductions
    abs_diff_sum = block.sum[block_size=BLOCK_SIZE](abs_diff_sum)
    abs_ref_sum = block.sum[block_size=BLOCK_SIZE](abs_ref_sum)
    max_violation = block.max[block_size=BLOCK_SIZE](max_violation)
    out_nz = block.max[block_size=BLOCK_SIZE](out_nz)
    ref_nz = block.max[block_size=BLOCK_SIZE](ref_nz)

    # Each block writes its partial results
    if thread_idx.x == 0:
        var base = Int(block_idx.x) * 5
        result[base + 0] = abs_diff_sum
        result[base + 1] = abs_ref_sum
        result[base + 2] = max_violation
        result[base + 3] = out_nz
        result[base + 4] = ref_nz


def verify_matmul[
    c_type: DType,
    a_type: DType,
    scales_type: DType,
    micro_scaling_mode: StaticString,
    SF_VECTOR_SIZE: Int,
    *,
    transpose_b: Bool = False,
](
    ctx: DeviceContext,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
    init_type: InitializationType,
) raises:
    var c_size = shape_c[0].value() * shape_c[1].value()
    var a_size = shape_a[0].value() * shape_a[1].value()
    var b_size = shape_b[0].value() * shape_b[1].value()

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = TileTensor(a_device.unsafe_ptr(), row_major(shape_a))
    var b_device = ctx.enqueue_create_buffer[a_type](b_size)
    var b_device_nd = TileTensor(b_device.unsafe_ptr(), row_major(shape_b))
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = TileTensor(c_device.unsafe_ptr(), row_major(shape_c))
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = TileTensor(
        c_device_ref.unsafe_ptr(), row_major(shape_c)
    )

    init_vector_launch[a_type](a_device, a_size, init_type, ctx)
    init_vector_launch[a_type](b_device, b_size, init_type, ctx)

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

    var a_scales = TileTensor(buffer_a_scales, row_major(a_scales_shape))
    var b_scales = TileTensor(buffer_b_scales, row_major(b_scales_shape))

    vendor_blas.matmul[scales_type=scales_type](
        ctx,
        c_device_ref_nd,
        a_device_nd,
        b_device_nd,
        a_scales=a_scales.as_immut(),
        b_scales=b_scales.as_immut(),
        transpose_b=True,
        c_row_major=True,
    )

    block_scaled_matmul[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=transpose_b,
    ](
        c_device_nd,
        a_device_nd,
        b_device_nd,
        a_scales,
        b_scales,
        1.0,
        ctx,
    )

    ctx.synchronize()

    # Launch GPU verification kernel
    comptime NUM_BLOCKS = 32
    comptime BLOCK_SIZE = 256

    var rtol: Float32
    var atol: Float32
    comptime if a_type.is_float8():
        rtol = 1e-2
        atol = 1e-2
    else:
        var rtol64: Float64
        var atol64: Float64
        rtol64, atol64 = pytorch_like_tolerances_for[DType.bfloat16]()
        rtol = Float32(rtol64)
        atol = Float32(atol64)

    var result_device = ctx.enqueue_create_buffer[DType.float32](NUM_BLOCKS * 5)

    comptime kernel = _verify_buffers_gpu[c_type, BLOCK_SIZE]
    ctx.enqueue_function_experimental[kernel](
        c_device.unsafe_ptr(),
        c_device_ref.unsafe_ptr(),
        c_size,
        atol,
        rtol,
        result_device.unsafe_ptr(),
        grid_dim=NUM_BLOCKS,
        block_dim=BLOCK_SIZE,
    )

    # Copy back only NUM_BLOCKS * 5 Float32 values
    var result_host = alloc[Scalar[DType.float32]](NUM_BLOCKS * 5)
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    # Reduce partial results from all blocks
    var total_abs_diff: Float32 = 0
    var total_abs_ref: Float32 = 0
    var worst_violation = Float32.MIN_FINITE
    var any_out_nz: Float32 = 0
    var any_ref_nz: Float32 = 0

    for b_idx in range(NUM_BLOCKS):
        var base = b_idx * 5
        total_abs_diff += result_host[base + 0]
        total_abs_ref += result_host[base + 1]
        worst_violation = max(worst_violation, result_host[base + 2])
        any_out_nz = max(any_out_nz, result_host[base + 3])
        any_ref_nz = max(any_ref_nz, result_host[base + 4])

    result_host.free()

    # Check zero/nonzero expectations
    var c_is_zeros = any_out_nz == 0
    var c_ref_is_zeros = any_ref_nz == 0

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

    # Check relative difference: sum(|x-y|) / sum(|y|) <= 0.001
    if total_abs_ref > 0:
        var rel_diff = total_abs_diff / total_abs_ref
        if rel_diff > 0.001:
            raise String(
                "matmul verification failed (relative_difference): ",
                rel_diff,
                " > 0.001",
            )

    # Check element-wise tolerance: max(|x-y| - (atol + rtol*|y|)) <= 0
    if worst_violation > 0:
        raise String(
            (
                "matmul verification failed (element-wise tolerance): worst"
                " violation = "
            ),
            worst_violation,
        )

    print("\n=== TEST PASSED ===\n")


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
    run_benchmark: Bool,
) raises:
    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    # update: using 512 to be 2x the infinity cache on MI300x
    @always_inline
    def get_size(shape: Coord) -> Int:
        return shape[0].value() * shape[1].value()

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

    comptime simd_size = 4
    var cb_a = CacheBustingBuffer[dtype](get_size(shape_a), simd_size, ctx)
    var cb_b = CacheBustingBuffer[dtype](get_size(shape_b), simd_size, ctx)
    var cb_c = CacheBustingBuffer[DType.bfloat16](
        get_size(shape_c), simd_size, ctx
    )
    var cb_a_scales = CacheBustingBuffer[scales_type](a_scales_size, 16, ctx)
    var cb_b_scales = CacheBustingBuffer[scales_type](b_scales_size, 16, ctx)

    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)
    cb_a_scales.init_scales_on_device(init_type, ctx)
    cb_b_scales.init_scales_on_device(init_type, ctx)

    # Helper to run vendor BLAS matmul - used by both benchmark and verification
    @parameter
    @__copy_capture(a_scales_shape, b_scales_shape)
    def run_vendor_blas(
        ctx: DeviceContext,
        c: TileTensor[mut=True, DType.bfloat16, ...],
        a: TileTensor[dtype, ...],
        b: TileTensor[dtype, ...],
        a_scales: TileTensor[scales_type, ...],
        b_scales: TileTensor[scales_type, ...],
    ) raises:
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
    @always_inline
    def kernel_launch(ctx: DeviceContext, iteration: Int) raises:
        var a = TileTensor(cb_a.offset_ptr(iteration), row_major(shape_a))
        var b = TileTensor(cb_b.offset_ptr(iteration), row_major(shape_b))
        var c = TileTensor(cb_c.offset_ptr(iteration), row_major(shape_c))

        var a_scales = TileTensor(
            cb_a_scales.offset_ptr(iteration), row_major(a_scales_shape)
        )
        var b_scales = TileTensor(
            cb_b_scales.offset_ptr(iteration), row_major(b_scales_shape)
        )

        @parameter
        @always_inline
        @__copy_capture(c)
        def test_lambda_add_coords_prod[
            _dtype: DType,
            width: Int,
            *,
            alignment: Int = align_of[SIMD[_dtype, width]](),
        ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
            _dtype, width
        ]:
            comptime assert c.flat_rank >= 2
            var x = c.load[width=width](Coord(idx)).cast[_dtype]()
            var y = val * x
            return y

        comptime optional_lambda_fn = Optional[elementwise_compute_lambda_type](
            test_lambda_add_coords_prod
        ) if epilogue else None

        comptime if use_vendor_blas:
            run_vendor_blas(ctx, c, a, b, a_scales, b_scales)

        else:
            block_scaled_matmul[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                1.0,
                ctx,
            )

    @parameter
    @always_inline
    def bench_func(mut b: Bencher) raises:
        b.iter_custom[kernel_launch](ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
        2 * Int(M) * Int(N) * Int(K),
    )
    if run_benchmark:
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
    else:
        kernel_launch(ctx, 0)

    # Verification: compare our kernel output against vendor BLAS as reference.
    # The benchmark already wrote our kernel's output to buffer_c at offset 0
    # (iteration 0 uses offset 0), so we just need to run vendor BLAS once.
    comptime if not use_vendor_blas and not epilogue:
        if verify:
            verify_matmul[
                DType.bfloat16,
                dtype,
                scales_type,
                micro_scaling_mode,
                SF_VECTOR_SIZE,
                transpose_b=transpose_b,
            ](ctx, shape_c, shape_a, shape_b, init_type)


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
    run_benchmark: Bool,
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
        run_benchmark,
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
    comptime N = get_defined_int["N", 7168]()
    comptime K = get_defined_int[
        "K", 2048
    ]() // 2 if micro_scaling_mode == "nvfp4" else get_defined_int["K", 2048]()

    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    var verify = arg_parse("verify", True)
    comptime cache_busting = True
    comptime transpose_b = True
    comptime use_vendor_blas = get_defined_bool["use_vendor_blas", False]()
    comptime epilogue = get_defined_bool["epilogue", False]()

    var run_benchmark = arg_parse("run_benchmark", True)

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
            run_benchmark,
        )

    if run_benchmark:
        m.dump_report()
