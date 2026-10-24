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
from std.sys import (
    get_defined_bool,
    get_defined_int,
    get_defined_string,
)
from max.gpu import global_idx, grid_dim, block_dim, thread_idx, block_idx
import linalg.matmul.vendor.blas as vendor_blas
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    pytorch_like_tolerances_for,
)
from internal_utils._measure import relative_difference
from linalg.block_scaled_quantization import block_scaled_matmul
from linalg.matmul.gpu.amd import block_scaled_matmul_amd

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
    E2M1_TO_FLOAT32,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
    NVFP4_SF_DTYPE,
    W4A8_A_DTYPE,
    W4A8_B_DTYPE,
    block_scaled_umma_kind,
)
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.sm100.tile_scheduler import RasterOrder
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.utils.numerics import isfinite
from max.gpu.primitives import block

comptime AUTOTUNING_MODE = get_defined_bool["AUTOTUNING_MODE", False]()


def _scales_dtype[micro_scaling_mode: StaticString]() -> DType:
    """Returns E4M3 scales for NVFP4 and E8M0 for MXFP8, MXFP4 and W4A8."""
    return NVFP4_SF_DTYPE if micro_scaling_mode == "nvfp4" else MXFP8_SF_DTYPE


def _is_packed_fp4_mode[micro_scaling_mode: StaticString]() -> Bool:
    """Reports whether A is nibble-packed E2M1, stored as `K // 2` bytes."""
    return micro_scaling_mode == "nvfp4" or micro_scaling_mode == "mxfp4"


def _sf_vector_size[micro_scaling_mode: StaticString]() -> Int:
    """Returns 16 elements per scale for NVFP4 and 32 for every E8M0 format."""
    return (
        NVFP4_SF_VECTOR_SIZE if micro_scaling_mode
        == "nvfp4" else MXFP8_SF_VECTOR_SIZE
    )


def _tune_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_type: DType,
    scaling_kind: UMMAKind,
]() -> BlockScaledMatmulConfig[
    a_type, b_type, c_type, scales_type, scales_type, True
]:
    """Builds the config `block_scaled_matmul` launches under `AUTOTUNING_MODE`.

    Reads the same `TUNE_*` defines with the same defaults as that branch, so
    one kbench grid drives all four formats, including W4A8, which
    `block_scaled_matmul` does not accept. The two must stay in sync.

    Parameters:
        a_type: A operand element type.
        b_type: B operand element type.
        c_type: Output element type.
        scales_type: Scale-factor element type for both operands.
        scaling_kind: UMMA kind the config runs.

    Returns:
        The config named by the `TUNE_*` defines.
    """
    comptime cta_group = get_defined_int["TUNE_CTA_GROUP", 2]()
    return BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_type, scales_type, True
    ](
        scaling_kind=scaling_kind,
        mma_shape=(
            get_defined_int["TUNE_BM", 128]() * cta_group,
            get_defined_int["TUNE_BN", 128]() * cta_group,
            32,
        ),
        cluster_shape=(
            get_defined_int["TUNE_CLUSTER_DIM_X", 2](),
            get_defined_int["TUNE_CLUSTER_DIM_Y", 1](),
            get_defined_int["TUNE_CLUSTER_DIM_Z", 1](),
        ),
        block_swizzle_size=get_defined_int["TUNE_BLOCK_SWIZZLE_SIZE", 0](),
        raster_order=RasterOrder(
            Int32(get_defined_int["TUNE_RASTER_ORDER", 1]())
        ),
        cta_group=cta_group,
        AB_swapped=get_defined_bool["TUNE_AB_SWAPPED", False](),
        k_group_size=get_defined_int["TUNE_K_GROUP_SIZE", 1](),
        num_accum_pipeline_stages=get_defined_int["TUNE_NUM_ACCUM_STAGES", 2](),
    )


def _widen_e2m1_to_e4m3_gpu(
    packed: ImmPointer[Scalar[.uint8], ImmutAnyOrigin],
    widened: MutPointer[Scalar[.float8_e4m3fn], MutAnyOrigin],
    num_bytes: Int32,
):
    """Unpacks nibble-packed E2M1 (element 2i in the low nibble) to E4M3.

    Every E2M1 value is exact in E4M3, so an MXFP8 matmul on the widened
    operand with the same E8M0 scales computes the same products.
    """
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(num_bytes):
        var byte = packed[unsafe_offset=i]
        widened[unsafe_offset=2 * i] = E2M1_TO_FLOAT32[Int(byte & 0xF)].cast[
            .float8_e4m3fn
        ]()
        widened[unsafe_offset=2 * i + 1] = E2M1_TO_FLOAT32[Int(byte >> 4)].cast[
            .float8_e4m3fn
        ]()
        i += stride


def _widen_e2m1_to_e4m3_into(
    ctx: DeviceContext,
    packed: DeviceBuffer[.uint8],
    widened: DeviceBuffer[.float8_e4m3fn],
    num_bytes: Int,
) raises:
    """Writes the `2 * num_bytes` E2M1 values in `packed` to `widened`."""
    comptime kernel = _widen_e2m1_to_e4m3_gpu
    ctx.enqueue_function[kernel](
        packed.unsafe_ptr().as_imm(),
        widened,
        Int32(num_bytes),
        grid_dim=ceildiv(num_bytes, 256),
        block_dim=256,
    )


def _widen_e2m1_to_e4m3(
    ctx: DeviceContext, packed: DeviceBuffer[.uint8], num_bytes: Int
) raises -> DeviceBuffer[.float8_e4m3fn]:
    """Returns a new E4M3 buffer holding `2 * num_bytes` widened E2M1 values."""
    var widened = ctx.enqueue_create_buffer[.float8_e4m3fn](2 * num_bytes)
    _widen_e2m1_to_e4m3_into(ctx, packed, widened, num_bytes)
    return widened


def _pow2_e4m3_scales_gpu(
    random_bytes: ImmPointer[Scalar[.uint8], ImmutAnyOrigin],
    scales: MutPointer[Scalar[.float8_e4m3fn], MutAnyOrigin],
    length: Int32,
):
    """Maps random bytes to E4M3 scales in {1, 2, 4, 8}, the E8M0 init's set."""
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(length):
        var exponent = Int(random_bytes[unsafe_offset=i] & 3)
        scales[unsafe_offset=i] = Float32(1 << exponent).cast[.float8_e4m3fn]()
        i += stride


def _init_matched_operand[
    dtype: DType
](ctx: DeviceContext, buf: DeviceBuffer[dtype], length: Int) raises:
    """Fills an operand with uniformly random E2M1 values, in any format.

    Packed operands get random bytes, two E2M1 codes each; E4M3 operands get
    the same kind of codes widened. Every format then multiplies the same
    value distribution, which keeps power draw, and so the clock, comparable.

    Parameters:
        dtype: `uint8` (packed E2M1) or `float8_e4m3fn`.

    Args:
        ctx: Device context.
        buf: Operand buffer.
        length: Elements of `dtype` to fill.
    """
    comptime if dtype == .uint8:
        init_vector_launch[.uint8](
            rebind[DeviceBuffer[.uint8]](buf),
            length,
            InitializationType.uniform_distribution,
            ctx,
        )
    else:
        comptime assert dtype == .float8_e4m3fn, "matched data needs E4M3"
        var codes = ctx.enqueue_create_buffer[.uint8](length // 2)
        init_vector_launch[.uint8](
            codes, length // 2, InitializationType.uniform_distribution, ctx
        )
        _widen_e2m1_to_e4m3_into(
            ctx, codes, rebind[DeviceBuffer[.float8_e4m3fn]](buf), length // 2
        )
        ctx.synchronize()


def _init_matched_scales[
    dtype: DType
](ctx: DeviceContext, buf: DeviceBuffer[dtype], length: Int) raises:
    """Fills scales with powers of two drawn uniformly from {1, 2, 4, 8}.

    E8M0 scales already init to that set; E4M3 (NVFP4) scales are mapped to it
    instead of the default uniform [0, 1) draw.

    Parameters:
        dtype: `float8_e8m0fnu` or `float8_e4m3fn`.

    Args:
        ctx: Device context.
        buf: Scale buffer.
        length: Scales to fill.
    """
    comptime if dtype == MXFP8_SF_DTYPE:
        _init_block_scaled_scales_launch[dtype](buf, length, ctx)
    else:
        comptime assert dtype == NVFP4_SF_DTYPE, "unsupported scale dtype"
        var random_bytes = ctx.enqueue_create_buffer[.uint8](length)
        init_vector_launch[.uint8](
            random_bytes, length, InitializationType.uniform_distribution, ctx
        )
        comptime kernel = _pow2_e4m3_scales_gpu
        ctx.enqueue_function[kernel](
            random_bytes.unsafe_ptr().as_imm(),
            rebind[DeviceBuffer[.float8_e4m3fn]](buf),
            Int32(length),
            grid_dim=ceildiv(length, 256),
            block_dim=256,
        )
        ctx.synchronize()


def _verify_buffers_gpu[
    c_type: DType, BLOCK_SIZE: Int
](
    output: ImmPointer[Scalar[c_type], ImmutAnyOrigin],
    reference: ImmPointer[Scalar[c_type], ImmutAnyOrigin],
    length: Int32,
    atol: Float32,
    rtol: Float32,
    result: MutPointer[Float32, MutAnyOrigin],
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
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(length):
        var x = output[i].cast[.float32]()
        var y = reference[i].cast[.float32]()
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
        var base = block_idx.x * 5
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
    comptime assert shape_a.element_types[0].DTYPE.is_integral()
    comptime assert shape_a.element_types[1].DTYPE.is_integral()
    comptime assert shape_b.element_types[0].DTYPE.is_integral()
    comptime assert shape_b.element_types[1].DTYPE.is_integral()
    comptime assert shape_c.element_types[0].DTYPE.is_integral()
    comptime assert shape_c.element_types[1].DTYPE.is_integral()
    var c_size = Int(shape_c[0]) * Int(shape_c[1])
    var a_size = Int(shape_a[0]) * Int(shape_a[1])
    var b_size = Int(shape_b[0]) * Int(shape_b[1])

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = TileTensor(a_device, row_major(shape_a))
    var b_device = ctx.enqueue_create_buffer[a_type](b_size)
    var b_device_nd = TileTensor(b_device, row_major(shape_b))
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = TileTensor(c_device, row_major(shape_c))
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = TileTensor(c_device_ref, row_major(shape_c))

    init_vector_launch[a_type](a_device, a_size, init_type, ctx)
    init_vector_launch[a_type](b_device, b_size, init_type, ctx)

    # M, N, K dimensions for scales calculation
    var M = Int(shape_c[0])
    var N = Int(shape_c[1])
    comptime K = shape_a.element_types[1].static_value * (
        2 if _is_packed_fp4_mode[micro_scaling_mode]() else 1
    )

    # Calculate scale buffer shapes - 5D tensors for MXFP8 format
    var a_scales_shape = Coord(
        ceildiv(Int(shape_a[0]), SF_MN_GROUP_SIZE),
        Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_scales_shape = Coord(
        Idx[ceildiv(shape_b.element_types[0].static_value, SF_MN_GROUP_SIZE)],
        Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
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

    comptime if micro_scaling_mode == "mxfp4":
        # vendor_blas rejects E2M1 with E8M0 scales, so the reference is
        # cuBLASLt MXFP8 on both operands widened, with the same scales.
        var a_wide = _widen_e2m1_to_e4m3(
            ctx, rebind[DeviceBuffer[.uint8]](a_device), a_size
        )
        var b_wide = _widen_e2m1_to_e4m3(
            ctx, rebind[DeviceBuffer[.uint8]](b_device), b_size
        )
        vendor_blas.matmul[scales_type=scales_type](
            ctx,
            c_device_ref_nd,
            TileTensor(a_wide, row_major(Coord(M, Idx[K]))),
            TileTensor(
                b_wide,
                row_major(
                    Coord(Idx[shape_b.element_types[0].static_value], Idx[K])
                ),
            ),
            a_scales=a_scales.as_imm(),
            b_scales=b_scales.as_imm(),
            transpose_b=True,
            c_row_major=True,
        )
        ctx.synchronize()
    else:
        vendor_blas.matmul[scales_type=scales_type](
            ctx,
            c_device_ref_nd,
            a_device_nd,
            b_device_nd,
            a_scales=a_scales.as_imm(),
            b_scales=b_scales.as_imm(),
            transpose_b=True,
            c_row_major=True,
        )

    comptime if AUTOTUNING_MODE:
        _launch_tuned(
            c_device_nd, a_device_nd, b_device_nd, a_scales, b_scales, ctx
        )
    else:
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

    var rtol: Float32
    var atol: Float32
    comptime if a_type.is_float8():
        rtol = 1e-2
        atol = 1e-2
    else:
        var rtol64: Float64
        var atol64: Float64
        rtol64, atol64 = pytorch_like_tolerances_for[.bfloat16]()
        rtol = Float32(rtol64)
        atol = Float32(atol64)

    _check_against_reference(
        ctx, c_device, c_device_ref, init_type, atol, rtol, "vendor BLAS"
    )
    print("\n=== TEST PASSED ===\n")


def _check_against_reference[
    c_type: DType
](
    ctx: DeviceContext,
    output: DeviceBuffer[c_type],
    reference: DeviceBuffer[c_type],
    init_type: InitializationType,
    atol: Float32,
    rtol: Float32,
    reference_name: StaticString,
) raises:
    """Raises unless `output` matches `reference` element-wise.

    Passes when the summed relative difference is at most 0.001 and every
    element satisfies `|x - y| <= atol + rtol * |y|`; `atol = rtol = 0`
    demands exact equality.

    Args:
        ctx: Device context.
        output: Kernel output.
        reference: Reference output, the same length as `output`.
        init_type: Input initialization; zero inputs must give zero outputs.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        reference_name: Names the reference in error messages.
    """
    comptime NUM_BLOCKS = 32
    comptime BLOCK_SIZE = 256

    var result_device = ctx.enqueue_create_buffer[.float32](NUM_BLOCKS * 5)

    comptime kernel = _verify_buffers_gpu[c_type, BLOCK_SIZE]
    ctx.enqueue_function[kernel](
        output.unsafe_ptr().as_imm(),
        reference.unsafe_ptr().as_imm(),
        Int32(len(output)),
        atol,
        rtol,
        result_device,
        grid_dim=NUM_BLOCKS,
        block_dim=BLOCK_SIZE,
    )

    # Copy back only NUM_BLOCKS * 5 Float32 values
    var result_host = List(length=NUM_BLOCKS * 5, fill=Float32(0))
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

    # Check zero/nonzero expectations
    var c_is_zeros = any_out_nz == 0
    var c_ref_is_zeros = any_ref_nz == 0

    if init_type == InitializationType.zero:
        if not c_is_zeros:
            raise "matmul verification failed: kernel output should be all zeros for zero input"
        if not c_ref_is_zeros:
            raise String(
                "matmul verification failed: ",
                reference_name,
                " output should be all zeros for zero input",
            )
    else:
        if c_is_zeros:
            raise "matmul verification failed: kernel output is all zeros"
        if c_ref_is_zeros:
            raise String(
                "matmul verification failed: ",
                reference_name,
                " output is all zeros",
            )

    # A NaN poisons the sums, which makes the tolerance comparisons below
    # false, and the max reduction can drop it, so it would otherwise pass.
    if not isfinite(total_abs_diff) or not isfinite(total_abs_ref):
        raise String(
            "matmul verification failed: non-finite values against ",
            reference_name,
            " (sum |x-y| = ",
            total_abs_diff,
            ", sum |y| = ",
            total_abs_ref,
            ")",
        )

    # Check relative difference: sum(|x-y|) / sum(|y|) <= 0.001
    if total_abs_ref > 0:
        var rel_diff = total_abs_diff / total_abs_ref
        if rel_diff > 0.001:
            raise String(
                "matmul verification failed against ",
                reference_name,
                " (relative_difference): ",
                rel_diff,
                " > 0.001",
            )

    # Check element-wise tolerance: max(|x-y| - (atol + rtol*|y|)) <= 0
    if worst_violation > 0:
        raise String(
            "matmul verification failed against ",
            reference_name,
            " (element-wise tolerance): worst violation = ",
            worst_violation,
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
    config_str: String = "",
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
        String("/", config_str) if config_str else "",
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
    matched_data: Bool = False,
) raises:
    comptime assert shape_a.element_types[0].DTYPE.is_integral()
    comptime assert shape_a.element_types[1].DTYPE.is_integral()
    comptime assert shape_b.element_types[0].DTYPE.is_integral()
    comptime assert shape_b.element_types[1].DTYPE.is_integral()
    comptime assert shape_c.element_types[0].DTYPE.is_integral()
    comptime assert shape_c.element_types[1].DTYPE.is_integral()

    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    # update: using 512 to be 2x the infinity cache on MI300x
    @inline(.always)
    def get_size(shape: Coord) -> Int:
        return Int(shape[0]) * Int(shape[1])

    comptime scales_type = _scales_dtype[micro_scaling_mode]()
    comptime SF_VECTOR_SIZE = _sf_vector_size[micro_scaling_mode]()

    # M, N, K dimensions for scales calculation
    var M = Int(shape_c[0])
    var N = Int(shape_c[1])
    comptime K = shape_a.element_types[1].static_value * (
        2 if _is_packed_fp4_mode[micro_scaling_mode]() else 1
    )

    # Calculate scale buffer shapes - 5D tensors for MXFP8 format
    var a_scales_shape = Coord(
        ceildiv(Int(shape_a[0]), SF_MN_GROUP_SIZE),
        Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_scales_shape = Coord(
        Idx[ceildiv(shape_b.element_types[0].static_value, SF_MN_GROUP_SIZE)],
        Idx[ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
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
    var cb_c = CacheBustingBuffer[.bfloat16](get_size(shape_c), simd_size, ctx)
    var cb_a_scales = CacheBustingBuffer[scales_type](a_scales_size, 16, ctx)
    var cb_b_scales = CacheBustingBuffer[scales_type](b_scales_size, 16, ctx)

    if matched_data:
        _init_matched_operand(ctx, cb_a.device_buffer(), cb_a.alloc_size())
        _init_matched_operand(ctx, cb_b.device_buffer(), cb_b.alloc_size())
        _init_matched_scales(
            ctx, cb_a_scales.device_buffer(), cb_a_scales.alloc_size()
        )
        _init_matched_scales(
            ctx, cb_b_scales.device_buffer(), cb_b_scales.alloc_size()
        )
    else:
        cb_a.init_on_device(init_type, ctx)
        cb_b.init_on_device(init_type, ctx)
        cb_a_scales.init_scales_on_device(init_type, ctx)
        cb_b_scales.init_scales_on_device(init_type, ctx)

    # Helper to run vendor BLAS matmul - used by both benchmark and verification
    def run_vendor_blas(
        ctx: DeviceContext,
        c: TileTensor[mut=True, .bfloat16, ...],
        a: TileTensor[dtype, ...],
        b: TileTensor[dtype, ...],
        a_scales: TileTensor[scales_type, ...],
        b_scales: TileTensor[scales_type, ...],
    ) raises {var a_scales_shape, var b_scales_shape, imm}:
        vendor_blas.matmul[scales_type=scales_type](
            ctx,
            c,
            a,
            b,
            a_scales=a_scales.as_imm(),
            b_scales=b_scales.as_imm(),
            transpose_b=True,
            c_row_major=True,
        )

    @inline(.always)
    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {
        mut cb_a, mut cb_b, mut cb_c, mut cb_a_scales, mut cb_b_scales, imm
    }:
        var a = TileTensor(cb_a.offset_ptr(iteration), row_major(shape_a))
        var b = TileTensor(cb_b.offset_ptr(iteration), row_major(shape_b))
        var c = TileTensor(cb_c.offset_ptr(iteration), row_major(shape_c))

        var a_scales = TileTensor(
            cb_a_scales.offset_ptr(iteration), row_major(a_scales_shape)
        )
        var b_scales = TileTensor(
            cb_b_scales.offset_ptr(iteration), row_major(b_scales_shape)
        )

        comptime if use_vendor_blas:
            run_vendor_blas(ctx, c, a, b, a_scales, b_scales)
        elif AUTOTUNING_MODE:
            _launch_tuned(c, a, b, a_scales, b_scales, ctx)
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

    @inline(.always)
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
        2 * Int(M) * Int(N) * Int(K),
    )
    var config_str = String()
    comptime if AUTOTUNING_MODE and not use_vendor_blas:
        config_str = String(
            _tune_config[
                dtype,
                dtype,
                .bfloat16,
                scales_type,
                block_scaled_umma_kind[dtype, dtype, scales_type](),
            ]()
        )

    if run_benchmark:
        b.bench_function(
            bench_func,
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
                    config_str,
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
    matched_data: Bool = False,
) raises:
    var b_shape = Coord(
        Idx[NType.static_value if transpose_b else KType.static_value],
        Idx[KType.static_value if transpose_b else NType.static_value],
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
        matched_data,
    )


def get_dtype[micro_scaling_mode: StaticString]() -> DType:
    comptime assert (
        micro_scaling_mode == "mxfp8"
        or _is_packed_fp4_mode[micro_scaling_mode]()
    ), "invalid micro_scaling_mode"

    comptime if micro_scaling_mode == "mxfp8":
        return DType.float8_e4m3fn
    else:  # micro_scaling_mode in ("nvfp4", "mxfp4")
        return DType.uint8


def _launch_tuned[
    c_type: DType, a_type: DType, b_type: DType, scales_type: DType, //
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_type, ...],
    b_scales: TileTensor[scales_type, ...],
    ctx: DeviceContext,
) raises:
    """Launches the SM100 warp-specialized kernel at the `TUNE_*` config.

    Launches it the way `block_scaled_matmul` does under `AUTOTUNING_MODE`,
    for every format: that entry rejects W4A8, and for MXFP4 it also
    instantiates the small-BN dispatch path, which has no `kind::mxf4`.

    Args:
        c: Output, `[M, N]`.
        a: A operand, `[M, K]` E4M3 or `[M, K // 2]` packed E2M1.
        b: B operand, `[N, K]` E4M3 or `[N, K // 2]` packed E2M1.
        a_scales: Scales for `a` in the 5D TCGEN layout.
        b_scales: Scales for `b` in the 5D TCGEN layout.
        ctx: Device context.
    """
    comptime config = _tune_config[
        a_type,
        b_type,
        c_type,
        scales_type,
        block_scaled_umma_kind[a_type, b_type, scales_type](),
    ]()
    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=True, K=a.static_shape[1], config=config
    ](c, a, b, a_scales, b_scales, ctx)


def verify_w4a8[
    N: Int, K: Int
](ctx: DeviceContext, M: Int, init_type: InitializationType) raises:
    """Checks the W4A8 kernel at the `TUNE_*` config on fresh buffers.

    Widening the weights to E4M3 is exact, so the W4A8 output is checked
    against MXFP8 on the widened weights with the same scales, twice: our
    kernel at the same config, which issues the same MMA kind in the same K
    order and so must match exactly, and cuBLASLt within the MXFP8
    tolerance. This reruns the benchmarked binary; it does not inspect the
    timed launches.

    Parameters:
        N: Output columns (weight rows).
        K: Reduction extent in elements.

    Args:
        ctx: Device context.
        M: Output rows.
        init_type: Operand initialization.
    """
    comptime a_type = W4A8_A_DTYPE
    comptime b_type = W4A8_B_DTYPE
    comptime scales_type = MXFP8_SF_DTYPE
    comptime K_PACKED = K // 2
    comptime k_groups = ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    var sfa_size = ceildiv(M, SF_MN_GROUP_SIZE) * k_groups * sf_atom
    comptime sfb_size = ceildiv(N, SF_MN_GROUP_SIZE) * k_groups * sf_atom

    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[b_type](N * K_PACKED)
    var sfa_dev = ctx.enqueue_create_buffer[scales_type](sfa_size)
    var sfb_dev = ctx.enqueue_create_buffer[scales_type](sfb_size)
    var c_dev = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_mxfp8_dev = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var c_ref_dev = ctx.enqueue_create_buffer[.bfloat16](M * N)

    init_vector_launch[a_type](a_dev, M * K, init_type, ctx)
    init_vector_launch[b_type](b_dev, N * K_PACKED, init_type, ctx)
    _init_block_scaled_scales_launch[scales_type](sfa_dev, sfa_size, ctx)
    _init_block_scaled_scales_launch[scales_type](sfb_dev, sfb_size, ctx)

    var c_shape = row_major(Coord(M, Idx[N]))
    var a = TileTensor(a_dev, row_major(Coord(M, Idx[K])))
    var sfa = TileTensor(
        sfa_dev,
        row_major(
            Coord(
                ceildiv(M, SF_MN_GROUP_SIZE),
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    )
    var sfb = TileTensor(
        sfb_dev,
        row_major(
            Coord(
                Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    )

    _launch_tuned(
        TileTensor(c_dev, c_shape),
        a,
        TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K_PACKED]))),
        sfa,
        sfb,
        ctx,
    )

    var b_wide_dev = _widen_e2m1_to_e4m3(ctx, b_dev, N * K_PACKED)
    var b_wide = TileTensor(b_wide_dev, row_major(Coord(Idx[N], Idx[K])))
    _launch_tuned(TileTensor(c_mxfp8_dev, c_shape), a, b_wide, sfa, sfb, ctx)
    vendor_blas.matmul[scales_type=scales_type](
        ctx,
        TileTensor(c_ref_dev, c_shape),
        a,
        b_wide,
        a_scales=sfa.as_imm(),
        b_scales=sfb.as_imm(),
        transpose_b=True,
        c_row_major=True,
    )

    _check_against_reference(
        ctx,
        c_dev,
        c_mxfp8_dev,
        init_type,
        0,
        0,
        "the MXFP8 kernel at the same config on widened weights",
    )
    _check_against_reference(
        ctx,
        c_dev,
        c_ref_dev,
        init_type,
        1e-2,
        1e-2,
        "vendor BLAS MXFP8 on widened weights",
    )
    print("\n=== TEST PASSED ===\n")
    _ = b_wide_dev^


def bench_w4a8[
    N: Int, K: Int
](
    ctx: DeviceContext,
    mut b: Bench,
    M: Int,
    init_type: InitializationType,
    verify: Bool,
    run_benchmark: Bool,
    matched_data: Bool,
) raises:
    """Benchmarks W4A8: E4M3 activations against nibble-packed E2M1 weights.

    `block_scaled_matmul` rejects the mixed pair, so this launches the SM100
    warp-specialized kernel directly, at the config `_tune_config` builds from
    the same `TUNE_*` defines the other formats read.

    Parameters:
        N: Output columns (weight rows).
        K: Reduction extent in elements; weight rows are `K // 2` bytes.

    Args:
        ctx: Device context.
        b: Benchmark collection the run is added to.
        M: Output rows.
        init_type: Operand initialization.
        verify: Runs `verify_w4a8` after the benchmark.
        run_benchmark: Times the kernel; otherwise launches it once.
        matched_data: Times on `_init_matched_operand` data instead of
            `init_type`.
    """
    comptime a_type = W4A8_A_DTYPE
    comptime b_type = W4A8_B_DTYPE
    comptime scales_type = MXFP8_SF_DTYPE
    comptime K_PACKED = K // 2
    comptime k_groups = ceildiv(K, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
    comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K

    var shape_c = Coord(M, Idx[N])
    var shape_a = Coord(M, Idx[K])
    comptime shape_b = Coord(Idx[N], Idx[K_PACKED])
    var sfa_shape = row_major(
        Coord(
            ceildiv(M, SF_MN_GROUP_SIZE),
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    comptime sfb_shape = row_major(
        Coord(
            Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var cb_a = CacheBustingBuffer[a_type](M * K, ctx)
    var cb_b = CacheBustingBuffer[b_type](N * K_PACKED, ctx)
    var cb_c = CacheBustingBuffer[.bfloat16](M * N, ctx)
    var cb_sfa = CacheBustingBuffer[scales_type](
        ceildiv(M, SF_MN_GROUP_SIZE) * k_groups * sf_atom, ctx
    )
    var cb_sfb = CacheBustingBuffer[scales_type](
        ceildiv(N, SF_MN_GROUP_SIZE) * k_groups * sf_atom, ctx
    )

    if matched_data:
        _init_matched_operand(ctx, cb_a.device_buffer(), cb_a.alloc_size())
        _init_matched_operand(ctx, cb_b.device_buffer(), cb_b.alloc_size())
        _init_matched_scales(ctx, cb_sfa.device_buffer(), cb_sfa.alloc_size())
        _init_matched_scales(ctx, cb_sfb.device_buffer(), cb_sfb.alloc_size())
    else:
        cb_a.init_on_device(init_type, ctx)
        cb_b.init_on_device(init_type, ctx)
        cb_sfa.init_scales_on_device(init_type, ctx)
        cb_sfb.init_scales_on_device(init_type, ctx)

    @inline(.always)
    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_a, mut cb_b, mut cb_c, mut cb_sfa, mut cb_sfb, imm}:
        _launch_tuned(
            TileTensor(cb_c.offset_ptr(iteration), row_major(shape_c)),
            TileTensor(cb_a.offset_ptr(iteration), row_major(shape_a)),
            TileTensor(cb_b.offset_ptr(iteration), row_major(shape_b)),
            TileTensor(cb_sfa.offset_ptr(iteration), sfa_shape),
            TileTensor(cb_sfb.offset_ptr(iteration), sfb_shape),
            ctx,
        )

    @inline(.always)
    def bench_func(mut bencher: Bencher) raises {imm}:
        bencher_iter_custom(bencher, kernel_launch, ctx)

    if run_benchmark:
        var config_str = String(
            _tune_config[
                a_type,
                b_type,
                .bfloat16,
                scales_type,
                UMMAKind.KIND_MXF8F6F4,
            ]()
        )
        b.bench_function(
            bench_func,
            BenchId(
                _get_run_name[
                    a_type,
                    transpose_b=True,
                    cache_busting=True,
                    use_vendor_blas=False,
                    micro_scaling_mode="w4a8",
                ](M, N, K, shape_c, shape_a, shape_b, config_str)
            ),
            [ThroughputMeasure(BenchMetric.flops, 2 * M * N * K)],
        )
    else:
        kernel_launch(ctx, 0)

    if verify:
        verify_w4a8[N, K](ctx, M, init_type)


# ===----------------------------------------------------------------------=== #
# AMD MXFP4 native benchmark (CDNA4)
# ===----------------------------------------------------------------------=== #


def bench_mxfp4_amd[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    *,
    cache_busting: Bool,
    transpose_b: Bool = True,
    use_vendor_blas: Bool = False,
    epilogue: Bool = False,
    micro_scaling_mode: StaticString = "mxfp4_amd",
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
    """Benchmark native MXFP4 block-scaled matmul on AMD CDNA4.

    Uses block_scaled_matmul_amd with simple 2D scale tensors
    [rows, K//32] in float8_e8m0fnu. Output is float32.
    """
    comptime K_ELEMS = KType.static_value
    comptime K_PACKED = K_ELEMS // 2
    comptime N_VAL = NType.static_value
    comptime K_SCALES = K_ELEMS // 32

    var M = Int(m)
    var a_size = M * K_PACKED
    var b_size = N_VAL * K_PACKED
    var c_size = M * N_VAL
    var a_scales_size = M * K_SCALES
    var b_scales_size = N_VAL * K_SCALES

    var a_shape = row_major(M, Idx[K_PACKED])
    comptime b_shape = row_major(Idx[N_VAL], Idx[K_PACKED])
    var c_shape = row_major(M, Idx[N_VAL])
    var sfa_shape = row_major(M, Idx[K_SCALES])
    comptime sfb_shape = row_major(Idx[N_VAL], Idx[K_SCALES])

    comptime simd_size = 4
    var cb_a = CacheBustingBuffer[.uint8](a_size, simd_size, ctx)
    var cb_b = CacheBustingBuffer[.uint8](b_size, simd_size, ctx)
    var cb_c = CacheBustingBuffer[.float32](c_size, simd_size, ctx)
    var cb_sfa = CacheBustingBuffer[.float8_e8m0fnu](
        a_scales_size, simd_size, ctx
    )
    var cb_sfb = CacheBustingBuffer[.float8_e8m0fnu](
        b_scales_size, simd_size, ctx
    )

    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)
    cb_sfa.init_scales_on_device(init_type, ctx)
    cb_sfb.init_scales_on_device(init_type, ctx)

    # Run hipBLASLt on the given tensors via the handle-taking vendor_blas
    # entry. The 2D scale tiles are passed through as-is.
    @inline(.always)
    def run_vendor_blas(
        ctx: DeviceContext,
        c: TileTensor[mut=True, .float32, ...],
        a: TileTensor[.uint8, ...],
        b: TileTensor[.uint8, ...],
        sfa: TileTensor[.float8_e8m0fnu, ...],
        sfb: TileTensor[.float8_e8m0fnu, ...],
    ) raises {imm}:
        with ctx.push_context() as cur_ctx:
            vendor_blas.matmul[
                scales_type=DType.float8_e8m0fnu, has_scales=True
            ](
                cur_ctx,
                vendor_blas._get_global_handle[.uint8](ctx),
                c,
                a,
                b,
                a_scales=sfa.as_imm(),
                b_scales=sfb.as_imm(),
                transpose_b=True,
                c_row_major=True,
            )

    @inline(.always)
    def kernel_launch(
        ctx: DeviceContext, iteration: Int
    ) raises {mut cb_a, mut cb_b, mut cb_c, mut cb_sfa, mut cb_sfb, imm}:
        var a_tt = TileTensor[mut=False](cb_a.offset_ptr(iteration), a_shape)
        var b_tt = TileTensor[mut=False](cb_b.offset_ptr(iteration), b_shape)
        var c_tt = TileTensor[mut=True](cb_c.offset_ptr(iteration), c_shape)
        var sfa_tt = TileTensor[mut=False](
            cb_sfa.offset_ptr(iteration), sfa_shape
        )
        var sfb_tt = TileTensor[mut=False](
            cb_sfb.offset_ptr(iteration), sfb_shape
        )
        comptime if use_vendor_blas:
            run_vendor_blas(ctx, c_tt, a_tt, b_tt, sfa_tt, sfb_tt)
        else:
            block_scaled_matmul_amd(c_tt, a_tt, b_tt, sfa_tt, sfb_tt, ctx)

    @inline(.always)
    def bench_func(mut bencher: Bencher) raises {imm}:
        bencher_iter_custom(bencher, kernel_launch, ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        2 * M * N_VAL * K_ELEMS,
    )
    var run_name = _get_run_name[
        DType.uint8,
        transpose_b=transpose_b,
        cache_busting=cache_busting,
        use_vendor_blas=use_vendor_blas,
        micro_scaling_mode=micro_scaling_mode,
    ](
        M,
        N_VAL,
        K_ELEMS,
        Coord(M, Idx[N_VAL]),
        Coord(M, Idx[K_PACKED]),
        Coord(Idx[N_VAL], Idx[K_PACKED]),
    )

    if run_benchmark:
        b.bench_function(bench_func, BenchId(run_name), [flops])
    else:
        kernel_launch(ctx, 0)

    # Verify: run the FP4 kernel and hipBLASLt on the iter-0 buffers and
    # compare via _verify_buffers_gpu. Skipped when the benchmark itself
    # is hipBLASLt (nothing to verify against).
    comptime if not use_vendor_blas:
        if verify:
            var a_tt0 = TileTensor[mut=False](cb_a.offset_ptr(0), a_shape)
            var b_tt0 = TileTensor[mut=False](cb_b.offset_ptr(0), b_shape)
            var sfa_tt0 = TileTensor[mut=False](cb_sfa.offset_ptr(0), sfa_shape)
            var sfb_tt0 = TileTensor[mut=False](cb_sfb.offset_ptr(0), sfb_shape)
            var c_tt0 = TileTensor[mut=True](cb_c.offset_ptr(0), c_shape)

            # Fresh FP4 kernel run into iter-0 output slot.
            block_scaled_matmul_amd(c_tt0, a_tt0, b_tt0, sfa_tt0, sfb_tt0, ctx)

            # hipBLASLt reference into a separate buffer.
            var c_ref_buf = ctx.enqueue_create_buffer[.float32](c_size)
            var c_ref_tt = TileTensor[mut=True](c_ref_buf, c_shape)
            run_vendor_blas(ctx, c_ref_tt, a_tt0, b_tt0, sfa_tt0, sfb_tt0)

            # GPU-side element-wise comparison: per-block partial reduction.
            comptime NUM_BLOCKS = 32
            comptime BLOCK_SIZE = 256

            var rtol = Float32(1.3e-6)
            var atol = Float32(1e-5)
            var result_device = ctx.enqueue_create_buffer[.float32](
                NUM_BLOCKS * 5
            )

            comptime verify_kernel = _verify_buffers_gpu[
                DType.float32, BLOCK_SIZE
            ]
            ctx.enqueue_function[verify_kernel](
                c_tt0.ptr,
                c_ref_tt.ptr,
                Int32(c_size),
                atol,
                rtol,
                result_device,
                grid_dim=NUM_BLOCKS,
                block_dim=BLOCK_SIZE,
            )

            var result_host = List(length=NUM_BLOCKS * 5, fill=Float32(0))
            ctx.enqueue_copy(result_host, result_device)
            ctx.synchronize()

            var total_abs_diff: Float32 = 0
            var total_abs_ref: Float32 = 0
            var worst_violation = Float32.MIN_FINITE
            var out_nz: Float32 = 0
            var ref_nz: Float32 = 0
            for bi in range(NUM_BLOCKS):
                var base = bi * 5
                total_abs_diff += result_host[base + 0]
                total_abs_ref += result_host[base + 1]
                worst_violation = max(worst_violation, result_host[base + 2])
                out_nz = max(out_nz, result_host[base + 3])
                ref_nz = max(ref_nz, result_host[base + 4])

            if init_type != InitializationType.zero:
                if out_nz == 0:
                    raise "MXFP4 verify: kernel output is all zeros"
                if ref_nz == 0:
                    raise "MXFP4 verify: hipBLASLt reference is all zeros"

            if total_abs_ref > 0:
                var rel_diff = total_abs_diff / total_abs_ref
                if rel_diff > 0.001:
                    raise String(
                        "MXFP4 verify failed (relative_difference): ",
                        rel_diff,
                        " > 0.001",
                    )

            if worst_violation > 0:
                raise String(
                    (
                        "MXFP4 verify failed (element-wise tolerance):"
                        " worst violation = "
                    ),
                    worst_violation,
                )

            print("\n=== MXFP4 verify PASSED ===\n")
            _ = result_host^


def main() raises:
    comptime micro_scaling_mode = get_defined_string["scaling_mode", "nvfp4"]()

    var M = Int(arg_parse("M", 1))
    comptime N = get_defined_int["N", 7168]()
    var run_benchmark = arg_parse("run_benchmark", True)

    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    var verify = arg_parse("verify", True)
    # Times every format on the same E2M1 value distribution and power-of-two
    # scales, so cross-format clock and power comparisons see the same data.
    var matched_data = arg_parse("matched_data", False)

    var m = Bench()
    with DeviceContext() as ctx:
        comptime cache_busting = True
        comptime transpose_b = True
        comptime use_vendor_blas = get_defined_bool["use_vendor_blas", False]()
        comptime epilogue = get_defined_bool["epilogue", False]()

        comptime if micro_scaling_mode == "mxfp4_amd":
            bench_mxfp4_amd[
                cache_busting=cache_busting,
                transpose_b=transpose_b,
                use_vendor_blas=use_vendor_blas,
                epilogue=epilogue,
                micro_scaling_mode=micro_scaling_mode,
            ](
                ctx,
                m,
                M,
                Idx[N],
                Idx[get_defined_int["K", 2048]()],
                init_type,
                verify,
                run_benchmark,
            )
        elif micro_scaling_mode == "w4a8":
            comptime assert AUTOTUNING_MODE, (
                "w4a8 runs only at an explicit config: build with"
                " -D AUTOTUNING_MODE=True and the TUNE_* defines"
            )
            comptime assert not use_vendor_blas and not epilogue, (
                "no vendor BLAS takes the W4A8 pair, and the w4a8 mode has no"
                " epilogue variant"
            )
            bench_w4a8[N, get_defined_int["K", 2048]()](
                ctx, m, M, init_type, verify, run_benchmark, matched_data
            )
        else:
            comptime assert micro_scaling_mode != "mxfp4" or (
                AUTOTUNING_MODE and not use_vendor_blas
            ), (
                "mxfp4 runs only at an explicit config (-D AUTOTUNING_MODE=True"
                " and the TUNE_* defines); neither the dispatcher's vendor"
                " fallback nor vendor BLAS takes E2M1 with E8M0 scales"
            )
            comptime dtype = get_dtype[micro_scaling_mode]()
            comptime K = get_defined_int["K", 2048]() // (
                2 if _is_packed_fp4_mode[micro_scaling_mode]() else 1
            )

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
                M,
                Idx[N],
                Idx[K],
                init_type,
                verify,
                run_benchmark,
                matched_data,
            )

    if run_benchmark:
        m.dump_report()
