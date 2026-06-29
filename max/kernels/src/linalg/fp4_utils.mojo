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
from std.sys._assembly import inlined_assembly
from std.sys import is_nvidia_gpu, bit_width_of, llvm_intrinsic
from std.sys.info import _is_sm_100x_or_newer, _cdna_4_or_newer, align_of
from std.utils.numerics import FPUtils
from std.utils.index import IndexList
from std.memory import bitcast
from layout import Coord, CoordLike, Idx, Layout, LayoutTensor, TileTensor
from std.builtin.simd import _convert_f32_to_float8_ue8m0
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind

comptime SF_ATOM_M = (32, 4)
comptime SF_ATOM_K = 4
comptime SF_MN_GROUP_SIZE: Int = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
comptime SF_K_GROUP_SIZE[SF_VECTOR_SIZE: Int]: Int = SF_ATOM_K * SF_VECTOR_SIZE

comptime NVFP4_SF_VECTOR_SIZE = 16
comptime MXFP4_SF_VECTOR_SIZE = 32
comptime MXFP8_SF_VECTOR_SIZE = 32

comptime NVFP4_SF_DTYPE = DType.float8_e4m3fn
comptime MXFP4_SF_DTYPE = DType.float8_e8m0fnu
comptime MXFP8_SF_DTYPE = DType.float8_e8m0fnu

comptime FP4_E2M1_MANTISSA_WIDTH = 1
comptime FP4_E2M1_MAX_EXPONENT = 2

comptime E2M1_TO_FLOAT32 = SIMD[DType.float32, 16](
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@always_inline
def compute_mxfp4_even_scale(max_val: Float32) -> Scalar[DType.float8_e8m0fnu]:
    """Computes the OCP MXFP4 E8M0 scale using even-mode rounding.

    Even-mode rounding rounds the block maximum before deriving the scale
    exponent. This differs from ceil(max / 6) and preserves more precision for
    smaller values in the same 32-element block.
    """
    comptime FP32_MANTISSA_WIDTH = FPUtils[DType.float32].mantissa_width()
    # MXFP4 stores only a power-of-two scale. Pick the scale so the largest
    # value in the block still fits in FP4 E2M1 after rounding, where the
    # largest finite FP4 E2M1 value is 6.0 = 1.5 * 2^2.
    #
    # The add below rounds max_val at the FP4 mantissa boundary. If that
    # rounded value crosses into the next power-of-two bucket, its Float32
    # exponent increases. Subtracting 2 (the exponent of FP4's max value)
    # turns that rounded-max exponent into the E8M0 scale exponent.
    #
    # Conceptually, for a block like [1.6, 0.4, ...], even-mode chooses
    # scale 0.25: [1.6, 0.4] / 0.25 = [6.4, 1.6], which rounds to FP4
    # [6.0, 1.5] and dequantizes to [1.5, 0.375]. Ceil(max / 6) would choose
    # scale 0.5: [1.6, 0.4] / 0.5 = [3.2, 0.8], which rounds to FP4
    # [3.0, 1.0] and dequantizes to [1.5, 0.5].
    comptime ROUND_TO_FP4_E2M1_MANTISSA = 1 << (
        FP32_MANTISSA_WIDTH - FP4_E2M1_MANTISSA_WIDTH - 1
    )
    var max_bits = FPUtils[DType.float32].bitcast_to_uint(max_val)
    var rounded_max_bits = max_bits + type_of(max_bits)(
        ROUND_TO_FP4_E2M1_MANTISSA
    )
    var rounded_max = bitcast[DType.float32](rounded_max_bits)
    var scale_exp = (
        FPUtils[DType.float32].get_exponent_biased(rounded_max)
        - FP4_E2M1_MAX_EXPONENT
    )
    scale_exp = max(0, min(scale_exp, 254))
    return bitcast[DType.float8_e8m0fnu](UInt8(scale_exp))


def cast_uint_to_fp4e2m1[
    in_dtype: DType,
    in_width: SIMDSize,
    //,
    *,
    out_dtype: DType,
    out_width: Int,
](x: SIMD[in_dtype, in_width]) -> SIMD[out_dtype, out_width]:
    comptime assert in_dtype in (
        DType.uint32,
        DType.uint16,
        DType.uint8,
    ), "input_dtype must be uint32, uint16 or uint8"

    comptime FP4_E2M1_WIDTH = 4
    comptime FP4_E2M1_MASK = pow(2, FP4_E2M1_WIDTH) - 1
    comptime num_fp4_values = bit_width_of[in_dtype]() // FP4_E2M1_WIDTH

    comptime assert in_width * num_fp4_values == out_width, (
        "size mismatch: input_width * num_fp4_values must be equal to"
        " output_width"
    )

    # Unpack nibbles with comptime indices only. Dynamic indexing into a
    # SIMD value (the old `E2M1_TO_FLOAT32[Int(x)]` lookup) forces the
    # vector to spill to GPU local memory on every access, which makes the
    # dequant kernels local-memory bound (~35x slower than memory-bound).
    var nibbles = SIMD[DType.uint8, out_width]()
    comptime for i in range(in_width):
        comptime for shift in range(0, num_fp4_values):
            comptime BitsType = type_of(x[i].to_bits())
            var nib = (
                x[i].to_bits() >> BitsType(shift * FP4_E2M1_WIDTH)
            ) & BitsType(FP4_E2M1_MASK)
            nibbles[i * num_fp4_values + shift] = nib.cast[DType.uint8]()

    # Branchless E2M1 decode (1 sign, 2 exponent, 1 mantissa) by direct f32 bit
    # construction. Cheaper than the `1 << (exp - 1)` + int->float pow2 + dual
    # select form: all-integer until one bitcast, with a single select for the
    # e == 0 subnormal {0.0, 0.5}.
    #   normal    (e >= 1): 2^(e - 1) * (1 + 0.5 * man)
    #       -> sign | (e + 126) << 23 | man << 22  (bias 127, mantissa MSB = man)
    #   subnormal (e == 0): 0.5 * man  ->  sign | (man * 0x3F000000)
    var bits = nibbles.cast[DType.uint32]()
    var sign = (bits & 0x8) << 28
    var exp = (bits >> 1) & 0x3
    var man = bits & 0x1
    var normal = bitcast[DType.float32, out_width](
        sign | ((exp + 126) << 23) | (man << 22)
    )
    var subnormal = bitcast[DType.float32, out_width](sign | (man * 0x3F000000))
    return (
        exp.eq(SIMD[DType.uint32, out_width](0))
        .select(subnormal, normal)
        .cast[out_dtype]()
    )


comptime FP4E2M1_MARLIN_BIAS = Float32(1 << 14)
"""Exponent-bias factor (2^14) that ``decode_fp4e2m1_marlin`` defers to the
caller: the decode returns values scaled by 2^-14, so multiply the (per-block)
dequant scale by this constant to recover the true magnitude."""


def decode_fp4e2m1_marlin[
    width: SIMDSize, //
](packed: SIMD[DType.uint8, width]) -> SIMD[DType.float32, width * 2]:
    """Decodes packed E2M1 (two nibbles per byte) to f32 via Marlin bit
    positioning (vLLM ``csrc/.../quantization/marlin/dequant.h``).

    Each nibble is placed at bits[15:12] of an fp16 lane (two values per uint32);
    a single mask+shift+or maps the sign and the three magnitude bits into the
    fp16 field with NO branch -- the ``e == 0`` subnormal ({0.0, 0.5}) falls out
    as an fp16 denormal. The 2^14 exponent-bias factor is NOT applied here: the
    returned values are 2^-14 of the true magnitude and the caller folds
    ``FP4E2M1_MARLIN_BIAS`` into the dequant scale (free, since that multiply
    happens anyway). Much cheaper than the arithmetic decode in
    ``cast_uint_to_fp4e2m1`` (the fp16 values pack two per 32-bit op and the
    bias/subnormal handling costs nothing).
    """
    var x = packed.cast[DType.uint32]()
    # lo nibble -> bits[15:12] (fp16 lane 0), hi nibble -> bits[31:28] (lane 1)
    var q = ((x & 0x0F) << 12) | ((x & 0xF0) << 24)
    var out = (q & 0x80008000) | ((q & 0x70007000) >> 3)
    return bitcast[DType.float16, width * 2](out).cast[DType.float32]()


def cast_fp_to_fp4e2m1[
    dtype: DType,
    width: SIMDSize,
    //,
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime assert dtype in (
        DType.float32,
        DType.bfloat16,
        DType.float16,
    ), "dtype must be float32, bfloat16 or float16"
    # for float4_e2m1fn has only 16 values
    # (x >= 0.0) & (x <= 0.25)] => 0.0
    # (x > 0.25) & (x < 0.75)] => 0.5
    # (x >= 0.75) & (x <= 1.25)] => 1.0
    # (x > 1.25) & (x < 1.75)] => 1.5
    # (x >= 1.75) & (x <= 2.5)] => 2.0
    # (x > 2.5) & (x < 3.5)] => 3.0
    # (x >= 3.5) & (x <= 5.0)] => 4.0
    # (x > 5.0) => 6.0

    var sign = x.lt(0).select(-1.0, 1.0).cast[dtype]()
    var abs_x = abs(x)
    var result = SIMD[dtype, width]()

    comptime for i in range(width):
        if abs_x[i] <= 0.25:
            result[i] = 0.0
        elif abs_x[i] < 0.75:
            result[i] = 0.5
        elif abs_x[i] <= 1.25:
            result[i] = 1.0
        elif abs_x[i] < 1.75:
            result[i] = 1.5
        elif abs_x[i] <= 2.5:
            result[i] = 2.0
        elif abs_x[i] < 3.5:
            result[i] = 3.0
        elif abs_x[i] <= 5.0:
            result[i] = 4.0
        else:
            result[i] = 6.0
    return result * sign


def cast_fp32_to_fp4e2m1[
    width: SIMDSize,
    //,
](x: SIMD[DType.float32, width]) -> UInt32:
    comptime assert (
        is_nvidia_gpu() and _is_sm_100x_or_newer()
    ), "only supported on NVIDIA GPUs with SM 100 or newer"
    comptime assert width == 8, "width must be 8"

    comptime asm_code = """{
.reg .b8 byte0;
.reg .b8 byte1;
.reg .b8 byte2;
.reg .b8 byte3;
cvt.rn.satfinite.e2m1x2.f32   byte0, $2, $1;
cvt.rn.satfinite.e2m1x2.f32   byte1, $4, $3;
cvt.rn.satfinite.e2m1x2.f32   byte2, $6, $5;
cvt.rn.satfinite.e2m1x2.f32   byte3, $8, $7;
mov.b32 $0, {byte0, byte1, byte2, byte3};
}
"""
    return inlined_assembly[
        asm_code, UInt32, constraints="=r,f,f,f,f,f,f,f,f", has_side_effect=True
    ](x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])


def cast_f4e2m1x2_to_fp16x2(x: Scalar[DType.uint8]) -> SIMD[DType.float16, 2]:
    comptime assert (
        is_nvidia_gpu() and _is_sm_100x_or_newer()
    ), "only supported on NVIDIA GPUs with SM 100 or newer"

    comptime asm_code = """{
.reg .b8 byte0;
.reg .b8 byte1;
mov.b16 {byte0, byte1}, $1;
cvt.rn.f16x2.e2m1x2 $0, byte0;
}
"""
    var result = inlined_assembly[
        asm_code, UInt32, constraints="=r,h", has_side_effect=True
    ](UInt16(x))

    return bitcast[DType.float16, 2](result)


@always_inline
def cast_float_to_fp4e2m1_amd[
    dtype: DType, width: SIMDSize, //
](input: SIMD[dtype, width], scale: Float32) -> UInt32:
    comptime assert (
        _cdna_4_or_newer()
    ), "only supported on AMD CDNA4 or newer (MI355X)"
    comptime assert (
        width % 2 == 0 and width <= 8
    ), "width must be even and at most 8"

    var packed = UInt32(0)

    comptime for i in range(width // 2):
        comptime if dtype == DType.bfloat16:
            packed = llvm_intrinsic[
                "llvm.amdgcn.cvt.scalef32.pk.fp4.bf16",
                UInt32,
            ](packed, input.slice[2, offset=i * 2](), scale, Int32(i))
        elif dtype == DType.float32:
            packed = llvm_intrinsic[
                "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                UInt32,
            ](packed, input[i * 2], input[i * 2 + 1], scale, Int32(i))
        else:
            comptime assert False, "Unsupported dtype"

    return packed


def set_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
    width: SIMDSize,
](
    scales_tensor: LayoutTensor[mut=True, scales_dtype, scales_layout, ...],
    row_idx: Int,
    col_idx: Int,
    scale_value: SIMD[scales_dtype, width],
):
    comptime assert (
        scales_tensor.rank == 5
    ), "scales_tensor must be 5D for non-batched scales tensor"
    comptime assert (
        width <= SF_ATOM_K
    ), "width must be less than or equal to SF_ATOM_K"

    comptime align = align_of[SIMD[scales_dtype, width]]()
    scales_tensor.store[store_alignment=align](
        IndexList[5](
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
            (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
        ),
        scale_value,
    )


def set_scale_factor[
    scales_dtype: DType,
    width: SIMDSize,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: TileTensor[mut=True, scales_dtype, ...],
    row_idx: Int,
    col_idx: Int,
    scale_value: SIMD[scales_dtype, width],
):
    comptime assert (
        width <= SF_ATOM_K
    ), "width must be less than or equal to SF_ATOM_K"
    comptime assert scales_tensor.flat_rank >= 5, "scales_tensor must be 5D"

    scales_tensor.store[width=width](
        (
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE // SF_ATOM_M[0]),
            (col_idx // SF_VECTOR_SIZE % SF_ATOM_K),
        ),
        scale_value,
    )


def get_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    row_idx: Int,
    col_idx: Int,
) -> Scalar[scales_dtype]:
    comptime assert (
        scales_tensor.rank == 5
    ), "scales_tensor must be 5D for non-batched scales tensor"

    return rebind[Scalar[scales_dtype]](
        scales_tensor[
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
            (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
        ]
    )


def get_scale_factor[
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: TileTensor[mut=True, scales_dtype, ...],
    row_idx: Int,
    col_idx: Int,
) -> Scalar[scales_dtype]:
    comptime assert (
        scales_tensor.flat_rank >= 5
    ), "scales_tensor must be 5D for non-batched scales tensor"

    return rebind[Scalar[scales_dtype]](
        scales_tensor[
            Coord(
                row_idx // SF_MN_GROUP_SIZE,
                col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
                row_idx % SF_ATOM_M[0],
                (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
            )
        ]
    )


def set_batched_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    batch_idx: Int,
    row_idx: Int,
    col_idx: Int,
    scale_value: Scalar[scales_dtype],
):
    comptime assert (
        scales_tensor.rank == 6
    ), "scales_tensor must be 6D for batched scales tensor"

    scales_tensor[
        batch_idx,
        row_idx // SF_MN_GROUP_SIZE,
        col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
        row_idx % SF_ATOM_M[0],
        (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
        (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
    ] = scale_value


def set_batched_scale_factor[
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: TileTensor[mut=True, scales_dtype, ...],
    batch_idx: Int,
    row_idx: Int,
    col_idx: Int,
    scale_value: Scalar[scales_dtype],
):
    comptime assert (
        scales_tensor.flat_rank == 6
    ), "scales_tensor must be 6D for batched scales tensor"

    scales_tensor.store(
        (
            batch_idx,
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE // SF_ATOM_M[0]),
            (col_idx // SF_VECTOR_SIZE % SF_ATOM_K),
        ),
        scale_value,
    )


def get_batched_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    batch_idx: Int,
    row_idx: Int,
    col_idx: Int,
) -> Scalar[scales_dtype]:
    comptime assert (
        scales_tensor.rank == 6
    ), "scales_tensor must be 6D for batched scales tensor"

    return rebind[Scalar[scales_dtype]](
        scales_tensor[
            batch_idx,
            row_idx // SF_MN_GROUP_SIZE,
            col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
            row_idx % SF_ATOM_M[0],
            (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
            (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
        ]
    )


def get_batched_scale_factor[
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
](
    scales_tensor: TileTensor[mut=True, scales_dtype, ...],
    batch_idx: Int,
    row_idx: Int,
    col_idx: Int,
) -> Scalar[scales_dtype]:
    comptime assert (
        scales_tensor.flat_rank == 6
    ), "scales_tensor must be 6D for batched scales tensor"

    return rebind[Scalar[scales_dtype]](
        scales_tensor[
            Coord(
                batch_idx,
                row_idx // SF_MN_GROUP_SIZE,
                col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
                row_idx % SF_ATOM_M[0],
                (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
            )
        ]
    )


def convert_ref_scales_to_mxfp8_format[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    ref_scales_type: DType,
    scales_type: DType,
    ref_a_scales_layout: Layout,
    ref_b_scales_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    a_scales_origin: MutOrigin,
    b_scales_origin: MutOrigin,
    *,
    REF_BLOCK_SIZE: Int,
    SF_VECTOR_SIZE: Int,
](
    m: MType,
    n: NType,
    k: KType,
    ref_a_scales: LayoutTensor[ref_scales_type, ref_a_scales_layout, _],
    ref_b_scales: LayoutTensor[ref_scales_type, ref_b_scales_layout, _],
    a_scales: LayoutTensor[scales_type, a_scales_layout, a_scales_origin],
    b_scales: LayoutTensor[scales_type, b_scales_layout, b_scales_origin],
):
    comptime assert (
        ref_scales_type == DType.float32
    ), "Only support float32 reference scales"
    comptime assert (
        scales_type == DType.float8_e8m0fnu
    ), "Only support float8_e8m0fnu scales"
    comptime assert ref_a_scales_layout.rank() == 2, "ref_a_scales must be 2D"
    comptime assert ref_b_scales_layout.rank() == 2, "ref_b_scales must be 2D"
    comptime assert a_scales_layout.rank() == 5, "a_scales must be 5D"
    comptime assert b_scales_layout.rank() == 5, "b_scales must be 5D"

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    # initialize a_scales_tensor and b_scales_tensor based on reference scales
    for m in range(M):
        for k in range(K):
            a_scales[
                m // SF_MN_GROUP_SIZE,
                k // (SF_VECTOR_SIZE * SF_ATOM_K),
                m % SF_ATOM_M[0],
                (m % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                k % SF_ATOM_K,
            ] = rebind[Scalar[scales_type]](
                _convert_f32_to_float8_ue8m0[scales_type](
                    ref_a_scales[k // REF_BLOCK_SIZE, m]
                )
            )

    for n in range(N):
        for k in range(K):
            b_scales[
                n // SF_MN_GROUP_SIZE,
                k // (SF_VECTOR_SIZE * SF_ATOM_K),
                n % SF_ATOM_M[0],
                (n % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                k % SF_ATOM_K,
            ] = rebind[Scalar[scales_type]](
                _convert_f32_to_float8_ue8m0[scales_type](
                    ref_b_scales[n // REF_BLOCK_SIZE, k // REF_BLOCK_SIZE]
                )
            )


def get_scaling_kind[
    a_type: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
]() -> UMMAKind:
    comptime if a_type == DType.uint8 and scales_dtype == NVFP4_SF_DTYPE and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE:
        return UMMAKind.KIND_MXF4NVF4
    elif a_type == DType.uint8 and scales_dtype == MXFP4_SF_DTYPE and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE:
        return UMMAKind.KIND_MXF4
    else:
        comptime assert (
            a_type == DType.float8_e4m3fn
            and scales_dtype == MXFP8_SF_DTYPE
            and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        ), "unsupported a_type/scales_dtype for block-scaled matmul"
        return UMMAKind.KIND_MXF8F6F4
