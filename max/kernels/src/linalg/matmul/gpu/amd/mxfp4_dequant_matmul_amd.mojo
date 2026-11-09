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
"""MXFP4 matmul on AMD CDNA GPUs with BF16 activations.

Two routes, split on M:

  * Decode and low batch (`M <= DECODE_MAX_M`): cast the BF16 activations to
    FP8 and run the native block-scaled MMA against the still-packed MXFP4
    weights. The CDNA4 `mfma.scale.f8f6f4` instruction consumes E4M3 A and
    E2M1 B with their E8M0 block scales in hardware, so the weights are never
    widened to one byte per element and no temporary weight buffer is
    allocated. This is the memory-bound regime, where dropping the weight
    materialization pass and reading B at half width is worth 2.5-4.8x.
  * Mid batch and up, and every shape on a part without the instruction: the
    same dequantize-to-FP8-then-tuned-FP8-GEMM route this file has always
    had. It is compute-bound enough there that the extra DRAM pass is hidden
    while the GEMM sustains ~2.3 PF/s. The mixed E4M3 x E2M1 tile is not tuned
    yet (see the W4A8 TODO on `BlockScaledMmaOp.use_smem_swizzle`), so it
    loses above the decode range.

The first route needs CDNA4 (`mfma.scale.f8f6f4`, gfx950); on any other part
only the second route exists, so nothing here instantiates the block-scaled
kernel for them.

Numerics match the dequant route exactly while every scaled weight stays
inside E4M3's representable range, which is what MX quantization produces: an
E2M1 value times a power of two is exact in E4M3 there, and the block scale is
derived from the block's largest magnitude. Outside that range the routes
differ, and the dequant route is the one that rounds first: for a block scale
of 2^-10, 0.5 * 2^-10 = 2^-11 is below E4M3's smallest subnormal 2^-9, so
`dequant_mxfp4` rounds it to zero before the GEMM while the MMA forms the
product in its exponent domain.

That bound is enforced, not documented: the caller passes the weight's
block-scale exponent range — one scalar pair on the packed-weight metadata,
computed where the scales are produced, not scanned per call — and the native
route only runs when that range proves representability. Callers that do not
pass it, or whose weights fall outside it, take the dequant route.

The weight and scale views are the other half of the contract, and it is a
type-level one: the native branch bakes both views' static shapes into the
kernel and reads the scales as one dense row-major run, so a view with a
runtime dimension or a strided scale view cannot instantiate it. Such callers
take the dequant route, which indexes both views through their own strides.
"""

from std.sys.info import _accelerator_arch
from max.gpu.host import DeviceContext
from layout import Idx, TensorLayout, TileTensor, row_major

from linalg.mxfp4_dequant import (
    _cast_bf16_to_fp8,
    _cast_bf16_to_fp8_unit_scales,
    dequant_mxfp4,
)
from linalg.matmul.gpu import _matmul_gpu

# Largest M routed to the native block-scaled MMA. Measured on MI355X at
# (N=2304, K=16384), (N=16384, K=2048) and (N=16384, K=6656): every shape wins
# up to M=256 (1.26-1.94x, and 2.7-4.9x at M=1) and no shape wins from M=512
# up. Prefill M keeps the tuned FP8 GEMM, which reaches ~2.3 PF/s where the
# mixed-format tile is untuned. Retune this with that tile, not in isolation.
comptime DECODE_MAX_M = 256

# MXFP4 block scales cover 32 elements; the block-scaled MMA takes one E8M0
# scale per block for each operand.
comptime MX_BLOCK_SIZE = 32

# `mfma.scale.f8f6f4` is a CDNA4 (gfx950) instruction. Every other AMD part
# only has the dequant route: `block_scaled_matmul_amd` would not compile at
# all for them, so the choice has to be comptime, not a runtime check.
comptime _HAS_BLOCK_SCALED_MMA = "gfx950" in _accelerator_arch()

# E8M0 exponents of the weight's block scales that keep every `weight * scale`
# representable in E4M3, which is what makes the two routes agree. MXFP4 values
# are E2M1, so the extremes are 0.5 and 6.0:
#   smallest 0.5 * 2^(min_exp - 127) >= 2^-9 (E4M3's smallest subnormal)
#     -> min_exp >= 119
#   largest  6.0 * 2^(max_exp - 127) <= 448 (E4M3's largest finite value)
#     -> max_exp <= 133
# A block scale of 2^-10, for instance, puts 0.5 at 2^-11: the dequant route
# rounds it to zero and the MMA forms it, so the routes disagree there.
comptime _MIN_SCALE_EXP_FOR_E4M3 = 119
comptime _MAX_SCALE_EXP_FOR_E4M3 = 133

# "Unknown scale range" for the two arguments below: nobody vouched for the
# bound, so the native route stays off.
comptime _SCALE_EXP_UNKNOWN = -1


@inline(.always)
def mxfp4_native_route_enabled(
    min_block_scale_exp: Int,
    max_block_scale_exp: Int,
    M: Int,
    static_N: Int,
    static_K: Int,
) -> Bool:
    """Whether `mxfp4_dequant_matmul_amd` takes the native block-scaled route.

    Answers the routing question for one call, so a test can assert it instead
    of inferring it from values that both routes produce identically. It is
    False unless every conjunct holds:

      * the part has `mfma.scale.f8f6f4` (gfx950), and the tiles cover K in
        whole 128-element blocks in K and whole 128-column blocks in N;
      * `M` is in the measured decode range;
      * the caller passed the weights' block-scale exponent range (an unknown
        range is `_SCALE_EXP_UNKNOWN`) and it keeps every `weight * scale`
        inside E4M3, so the two routes give the same numbers.

    Args:
        min_block_scale_exp: Smallest E8M0 byte among the weight's block
            scales, or `_SCALE_EXP_UNKNOWN`.
        max_block_scale_exp: Largest E8M0 byte among the weight's block
            scales, or `_SCALE_EXP_UNKNOWN`.
        M: Output rows of this call.
        static_N: Output columns.
        static_K: Reduction extent in logical elements.

    Returns:
        True if this call runs the native route.
    """
    return (
        _HAS_BLOCK_SCALED_MMA
        and static_K % 128 == 0
        # The dense block-scaled kernel's non-fused store clamps only on the
        # buffer extent, so a last column block that overruns N writes into the
        # next row. Require N to be a whole number of its 128-wide blocks.
        and static_N % 128 == 0
        and M > 0
        and M <= DECODE_MAX_M
        and min_block_scale_exp >= _MIN_SCALE_EXP_FOR_E4M3
        and max_block_scale_exp >= min_block_scale_exp
        and max_block_scale_exp <= _MAX_SCALE_EXP_FOR_E4M3
    )


def mxfp4_native_views_supported[
    weight_layout: TensorLayout,
    scales_layout: TensorLayout,
    static_scale_K: Int,
]() -> Bool:
    """Whether these weight and scale views can instantiate the native route.

    The type-level half of the routing question, which
    `mxfp4_native_route_enabled` cannot see: it is answered before any value
    exists. The native branch reads `static_shape` off both views and the
    block-scaled kernel states them as `comptime` asserts, so a view whose
    dimension is only known at runtime does not compile there — the fallback
    has to be a type-level gate, not a runtime condition. It is False unless:

      * both views are rank 2;
      * both views' shapes are known at compile time, and the scale view's K
        extent is `static_scale_K`, the one-entry-per-MX-block width this call
        needs;
      * the scale view is dense row-major: stride `(static_scale_K, 1)`. The
        native kernel reshapes `b_scales` to a flat row-major buffer and reads
        a run of `N * K_SCALES` contiguous bytes, so it cannot honour a
        strided view; the dequant route indexes the view through its own
        strides and can.

    Parameters:
        weight_layout: Layout of `b_packed`.
        scales_layout: Layout of `b_scales`.
        static_scale_K: Block scales per weight row, `static_K // 32`.

    Returns:
        True if `mxfp4_dequant_matmul_amd` may instantiate the native branch
        for views with these layouts.
    """
    comptime if weight_layout.flat_rank != 2 or scales_layout.flat_rank != 2:
        return False
    else:
        comptime views_shaped = (
            weight_layout.shape_known
            and scales_layout.shape_known
            and scales_layout.stride_known
        )
        comptime if not views_shaped:
            return False
        else:
            return (
                scales_layout.static_shape[1] == static_scale_K
                and scales_layout.static_stride[0] == static_scale_K
                and scales_layout.static_stride[1] == 1
            )


def mxfp4_dequant_matmul_amd(
    c: TileTensor[mut=True, .bfloat16, ...],
    a: TileTensor[mut=False, .bfloat16, ...],
    b_packed: TileTensor[mut=False, .uint8, ...],
    b_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    ctx: DeviceContext,
    min_block_scale_exp: Int = _SCALE_EXP_UNKNOWN,
    max_block_scale_exp: Int = _SCALE_EXP_UNKNOWN,
) raises:
    """MXFP4 matmul: native block-scaled MMA on packed B, or dequant + FP8 GEMM.

    The native route only runs when the caller passes the weight's block-scale
    exponent range and it proves every `weight * scale` is E4M3-representable
    (see `mxfp4_native_route_enabled`), and when both weight-side views are
    statically shaped and the scales dense row-major (see
    `mxfp4_native_views_supported`); otherwise — including when the range is
    unknown, the default — the dequant route runs, which is bit-exact whatever
    the scales are and honours either view's strides.

    Args:
        c: Output [M, N] in bfloat16.
        a: Activations [M, K] in bfloat16.
        b_packed: Weights [N, K//2] in uint8 (packed MXFP4). A runtime row or
            column extent, or non-dense scale strides, keep the call on the
            dequant route.
        b_scales: Weight scales [N, K//32] in float8_e8m0fnu. The native
            route needs this view dense row-major; any other strides are read
            correctly by the dequant route only.
        ctx: Device context.
        min_block_scale_exp: Smallest E8M0 byte among `b_scales`, from the
            weight's repack. Defaults to unknown, which stays on the dequant
            route.
        max_block_scale_exp: Largest E8M0 byte among `b_scales`. Same default
            and consequence.
    """
    var M = Int(c.dim[0]())
    comptime static_N = type_of(c).static_shape[1]
    comptime static_K = type_of(a).static_shape[1]
    comptime scale_K = static_K // MX_BLOCK_SIZE
    comptime fp8_type = DType.float8_e4m3fn

    # Instantiation gate: the native branch below cannot even be compiled
    # without the instruction and tile coverage.
    # The store side of the dense block-scaled kernel tiles N in 128-wide
    # blocks and clamps writes only at the buffer end, so a partial last block
    # would spill into the following row. N must cover whole blocks.
    # The weight and scale views have to be statically shaped and the scales
    # dense row-major on top of that, or the native branch does not elaborate
    # at all (see `mxfp4_native_views_supported`). Both are type-level facts,
    # so they belong here rather than in the runtime route predicate: a call
    # whose output N is static while a weight or scale view carries a runtime
    # dimension takes the dequant route, which reads those views at runtime.
    comptime _native_instantiable = (
        _HAS_BLOCK_SCALED_MMA
        and static_K % 128 == 0
        and static_N % 128 == 0
        and mxfp4_native_views_supported[
            type_of(b_packed).LayoutType,
            type_of(b_scales).LayoutType,
            scale_K,
        ]()
    )
    comptime if _native_instantiable:
        # Imported here, not at module scope: the block-scaled kernel is a
        # CDNA4 instruction wrapper, and this is the only place in the file
        # that may name it. `_native_instantiable` folds `_accelerator_arch()`
        # into a comptime branch, so a non-gfx950 compile never elaborates
        # this body and never instantiates that kernel.
        from linalg.matmul.gpu.amd.block_scaled_matmul_amd import (
            block_scaled_matmul_amd,
        )

        if mxfp4_native_route_enabled(
            min_block_scale_exp,
            max_block_scale_exp,
            M,
            static_N,
            static_K,
        ):
            # FP8 activations plus the unit E8M0 scale of every MX block: the
            # cast is unscaled, so the MMA applies exactly 2^0 and reproduces
            # the cast values.
            var a_fp8_buf = ctx.enqueue_create_buffer[fp8_type](M * static_K)
            var a_fp8_tt = TileTensor(a_fp8_buf, row_major((M, Idx[static_K])))
            var a_scales_buf = ctx.enqueue_create_buffer[.float8_e8m0fnu](
                M * scale_K
            )
            var a_scales_tt = TileTensor(
                a_scales_buf, row_major((M, Idx[scale_K]))
            )

            _cast_bf16_to_fp8_unit_scales(
                ctx, a_fp8_tt, a_scales_tt, a, M, static_K
            )

            # E4M3 A against E2M1 B: B stays packed, so the weights are read
            # once at half the width a dequantized copy would need.
            block_scaled_matmul_amd[lane_bytes=32, b_lane_bytes=16](
                c,
                a_fp8_tt.bitcast[.uint8]().as_imm(),
                b_packed,
                a_scales_tt.as_imm(),
                b_scales,
                ctx,
            )

            # Keep the activation buffers alive through the async launches.
            _ = a_fp8_buf^
            _ = a_scales_buf^
            return

    # Dequantize MXFP4 weights to FP8, then the tuned FP8 GEMM.
    var b_fp8_buf = ctx.enqueue_create_buffer[fp8_type](static_N * static_K)
    var b_fp8_tt = TileTensor(
        b_fp8_buf, row_major((Idx[static_N], Idx[static_K]))
    )

    dequant_mxfp4(
        ctx,
        b_fp8_tt,
        b_packed,
        b_scales,
        num_rows=static_N,
        num_cols=static_K,
    )

    var a_fp8_buf = ctx.enqueue_create_buffer[fp8_type](M * static_K)
    var a_fp8_tt = TileTensor(a_fp8_buf, row_major((M, Idx[static_K])))

    _cast_bf16_to_fp8(ctx, a_fp8_tt, a, M, static_K)

    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c, a_fp8_tt, b_fp8_tt, ctx
    )

    # Keep temp buffers alive through async GEMM enqueue.
    _ = b_fp8_buf^
    _ = a_fp8_buf^
