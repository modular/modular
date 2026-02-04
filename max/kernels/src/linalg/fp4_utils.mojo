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
from sys._assembly import inlined_assembly
from sys import is_nvidia_gpu, bit_width_of
from sys.info import _is_sm_100x_or_newer, align_of
from utils.index import IndexList
from utils.numerics import FPUtils
from memory import bitcast
from layout import Layout, LayoutTensor
from internal_utils._utils import ValOrDim, dynamic, static
from builtin.simd import _convert_f32_to_float8_ue8m0
from utils import IndexList
from sys import _RegisterPackType


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


fn cast_uint_to_fp4e2m1[
    in_dtype: DType,
    in_width: Int,
    //,
    *,
    out_dtype: DType,
    out_width: Int,
](x: SIMD[in_dtype, in_width]) -> SIMD[out_dtype, out_width]:
    __comptime_assert in_dtype in (
        DType.uint32,
        DType.uint16,
        DType.uint8,
    ), "input_dtype must be uint32, uint16 or uint8"

    comptime FP4_E2M1_WIDTH = 4
    comptime FP4_E2M1_MASK = pow(2, FP4_E2M1_WIDTH) - 1
    comptime num_fp4_values = bit_width_of[in_dtype]() // FP4_E2M1_WIDTH

    __comptime_assert in_width * num_fp4_values == out_width, (
        "size mismatch: input_width * num_fp4_values must be equal to"
        " output_width"
    )

    var result = SIMD[out_dtype, out_width]()

    @parameter
    for i in range(in_width):

        @parameter
        for shift in range(0, num_fp4_values):
            comptime BitsType = type_of(x[i].to_bits())
            var x = (
                x[i].to_bits() >> BitsType(shift * FP4_E2M1_WIDTH)
            ) & BitsType(FP4_E2M1_MASK)
            result[i * num_fp4_values + shift] = E2M1_TO_FLOAT32[Int(x)].cast[
                out_dtype
            ]()
    return result


fn cast_fp_to_fp4e2m1[
    dtype: DType,
    width: Int,
    //,
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    __comptime_assert dtype in (
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

    @parameter
    for i in range(width):
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


fn cast_fp32_to_fp4e2m1[
    width: Int,
    //,
](x: SIMD[DType.float32, width]) -> UInt32:
    __comptime_assert (
        is_nvidia_gpu() and _is_sm_100x_or_newer()
    ), "only supported on NVIDIA GPUs with SM 100 or newer"
    __comptime_assert width == 8, "width must be 8"

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


fn cast_f4e2m1x2_to_fp16x2(x: Scalar[DType.uint8]) -> SIMD[DType.float16, 2]:
    __comptime_assert (
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


fn cast_f4e2m1x8_to_fp16x8(x: Scalar[DType.uint32]) -> SIMD[DType.float16, 8]:
    __comptime_assert (
        is_nvidia_gpu() and _is_sm_100x_or_newer()
    ), "only supported on NVIDIA GPUs with SM 100 or newer"

    comptime asm_code = """{
.reg .b8 byte0, byte1, byte2, byte3;
mov.b32 {byte3, byte2, byte1, byte0}, $4;
cvt.rn.f16x2.e2m1x2 $0, byte0;
cvt.rn.f16x2.e2m1x2 $1, byte1;
cvt.rn.f16x2.e2m1x2 $2, byte2;
cvt.rn.f16x2.e2m1x2 $3, byte3;
}
"""
    var casted_output = inlined_assembly[
        asm_code,
        _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
        constraints="=r,=r,=r,=r,r",
        has_side_effect=True,
    ](UInt32(x))

    var result = SIMD[DType.float16, 8]()

    @parameter
    for offset in range(0, 8, 2):
        result = result.insert[offset=offset](
            bitcast[DType.float16, 2](casted_output[offset // 2])
        )

    return result


fn cast_f4e2m1x16_to_fp16x16(
    x: Scalar[DType.uint64],
) -> SIMD[DType.float16, 16]:
    __comptime_assert (
        is_nvidia_gpu() and _is_sm_100x_or_newer()
    ), "only supported on NVIDIA GPUs with SM 100 or newer"

    var x_casted = bitcast[DType.uint32, 2](x)

    comptime asm_code = """{
.reg .b8 byte0, byte1, byte2, byte3;
.reg .b8 byte4, byte5, byte6, byte7;
mov.b32 {byte3, byte2, byte1, byte0}, $8;
mov.b32 {byte7, byte6, byte5, byte4}, $9;
cvt.rn.f16x2.e2m1x2 $0, byte0;
cvt.rn.f16x2.e2m1x2 $1, byte1;
cvt.rn.f16x2.e2m1x2 $2, byte2;
cvt.rn.f16x2.e2m1x2 $3, byte3;
cvt.rn.f16x2.e2m1x2 $4, byte4;
cvt.rn.f16x2.e2m1x2 $5, byte5;
cvt.rn.f16x2.e2m1x2 $6, byte6;
cvt.rn.f16x2.e2m1x2 $7, byte7;
}
"""
    var casted_output = inlined_assembly[
        asm_code,
        _RegisterPackType[
            UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32
        ],
        constraints="=r,=r,=r,=r,=r,=r,=r,=r,r,r",
        has_side_effect=True,
    ](UInt32(x_casted[1]), UInt32(x_casted[0]))

    var result = SIMD[DType.float16, 16]()

    @parameter
    for offset in range(0, 16, 2):
        result = result.insert[offset=offset](
            bitcast[DType.float16, 2](casted_output[offset // 2])
        )

    return result


@always_inline
fn nvfp4_scaled_tile_multiply_accumulate(
    a_tile: SIMD[DType.uint32, 8],
    b_tile: SIMD[DType.uint32, 8],
    sfa_packed: Scalar[DType.uint32],
    sfb_packed: Scalar[DType.uint32],
    accum_val: Float32,
) -> Float32:
    """Aggressively fused PTX code for NVFP4 scaling and multiplication.

    Processes tiles of FP4-E2M1 values, multiplies them together, scales by
    the combined scale factors, and accumulates the result.

    Args:
        a_tile: 8 uint32 values containing FP4-E2M1 packed A values (A0_0 through A1_3).
        b_tile: 8 uint32 values containing FP4-E2M1 packed B values (B0_0 through B1_3).
        sfa_packed: Packed scale factors for A (e4m3x2 format).
        sfb_packed: Packed scale factors for B (e4m3x2 format).
        accum_val: Initial accumulator value.

    Returns:
        Accumulated float32 result after scaling and multiplication.
    """
    __comptime_assert (
        is_nvidia_gpu() and _is_sm_100x_or_newer()
    ), "only supported on NVIDIA GPUs with SM 100 or newer"

    comptime asm_code = """{
.reg .b16 sfalo, sfahi, sfblo, sfbhi;
.reg .b32 sa01, sa23, sb01, sb23;
.reg .b32 scale01, scale23;
.reg .f32 s0, s1, s2, s3;
.reg .b8 a<4>, b<4>;
.reg .b32 fa<4>, fb<4>;
.reg .b32 p0, p1, p2, p3;
.reg .f16 h0, h1;
.reg .f32 f0, f1, acc0, acc1, acc2, acc3, tile_result, one;

mov.f32 one, 0f3f800000;
mov.b32 {sfalo, sfahi}, $17;
mov.b32 {sfblo, sfbhi}, $18;
cvt.rn.f16x2.e4m3x2 sa01, sfalo;
cvt.rn.f16x2.e4m3x2 sa23, sfahi;
cvt.rn.f16x2.e4m3x2 sb01, sfblo;
cvt.rn.f16x2.e4m3x2 sb23, sfbhi;
mul.rn.f16x2 scale01, sa01, sb01;
mul.rn.f16x2 scale23, sa23, sb23;
mov.b32 {h0, h1}, scale01;
cvt.f32.f16 s0, h0;
cvt.f32.f16 s1, h1;
mov.b32 {h0, h1}, scale23;
cvt.f32.f16 s2, h0;
cvt.f32.f16 s3, h1;

mov.b32 {a0, a1, a2, a3}, $1;
mov.b32 {b0, b1, b2, b3}, $9;
cvt.rn.f16x2.e2m1x2 fa0, a0;
cvt.rn.f16x2.e2m1x2 fa1, a1;
cvt.rn.f16x2.e2m1x2 fa2, a2;
cvt.rn.f16x2.e2m1x2 fa3, a3;
cvt.rn.f16x2.e2m1x2 fb0, b0;
cvt.rn.f16x2.e2m1x2 fb1, b1;
cvt.rn.f16x2.e2m1x2 fb2, b2;
cvt.rn.f16x2.e2m1x2 fb3, b3;
mul.rn.f16x2 p0, fa0, fb0;
fma.rn.f16x2 p0, fa1, fb1, p0;
fma.rn.f16x2 p0, fa2, fb2, p0;
fma.rn.f16x2 p0, fa3, fb3, p0;
mov.b32 {a0, a1, a2, a3}, $2;
mov.b32 {b0, b1, b2, b3}, $10;
cvt.rn.f16x2.e2m1x2 fa0, a0;
cvt.rn.f16x2.e2m1x2 fa1, a1;
cvt.rn.f16x2.e2m1x2 fa2, a2;
cvt.rn.f16x2.e2m1x2 fa3, a3;
cvt.rn.f16x2.e2m1x2 fb0, b0;
cvt.rn.f16x2.e2m1x2 fb1, b1;
cvt.rn.f16x2.e2m1x2 fb2, b2;
cvt.rn.f16x2.e2m1x2 fb3, b3;
fma.rn.f16x2 p0, fa0, fb0, p0;
fma.rn.f16x2 p0, fa1, fb1, p0;
fma.rn.f16x2 p0, fa2, fb2, p0;
fma.rn.f16x2 p0, fa3, fb3, p0;
mov.b32 {h0, h1}, p0;
cvt.f32.f16 f0, h0;
cvt.f32.f16 f1, h1;
add.f32 acc0, f0, f1;
mul.f32 acc0, acc0, s0;

mov.b32 {a0, a1, a2, a3}, $3;
mov.b32 {b0, b1, b2, b3}, $11;
cvt.rn.f16x2.e2m1x2 fa0, a0;
cvt.rn.f16x2.e2m1x2 fa1, a1;
cvt.rn.f16x2.e2m1x2 fa2, a2;
cvt.rn.f16x2.e2m1x2 fa3, a3;
cvt.rn.f16x2.e2m1x2 fb0, b0;
cvt.rn.f16x2.e2m1x2 fb1, b1;
cvt.rn.f16x2.e2m1x2 fb2, b2;
cvt.rn.f16x2.e2m1x2 fb3, b3;
mul.rn.f16x2 p1, fa0, fb0;
fma.rn.f16x2 p1, fa1, fb1, p1;
fma.rn.f16x2 p1, fa2, fb2, p1;
fma.rn.f16x2 p1, fa3, fb3, p1;
mov.b32 {a0, a1, a2, a3}, $4;
mov.b32 {b0, b1, b2, b3}, $12;
cvt.rn.f16x2.e2m1x2 fa0, a0;
cvt.rn.f16x2.e2m1x2 fa1, a1;
cvt.rn.f16x2.e2m1x2 fa2, a2;
cvt.rn.f16x2.e2m1x2 fa3, a3;
cvt.rn.f16x2.e2m1x2 fb0, b0;
cvt.rn.f16x2.e2m1x2 fb1, b1;
cvt.rn.f16x2.e2m1x2 fb2, b2;
cvt.rn.f16x2.e2m1x2 fb3, b3;
fma.rn.f16x2 p1, fa0, fb0, p1;
fma.rn.f16x2 p1, fa1, fb1, p1;
fma.rn.f16x2 p1, fa2, fb2, p1;
fma.rn.f16x2 p1, fa3, fb3, p1;
mov.b32 {h0, h1}, p1;
cvt.f32.f16 f0, h0;
cvt.f32.f16 f1, h1;
add.f32 acc1, f0, f1;
fma.rn.f32 acc0, acc1, s1, acc0;

mov.b32 {a0, a1, a2, a3}, $5;
mov.b32 {b0, b1, b2, b3}, $13;
cvt.rn.f16x2.e2m1x2 fa0, a0;
cvt.rn.f16x2.e2m1x2 fa1, a1;
cvt.rn.f16x2.e2m1x2 fa2, a2;
cvt.rn.f16x2.e2m1x2 fa3, a3;
cvt.rn.f16x2.e2m1x2 fb0, b0;
cvt.rn.f16x2.e2m1x2 fb1, b1;
cvt.rn.f16x2.e2m1x2 fb2, b2;
cvt.rn.f16x2.e2m1x2 fb3, b3;
mul.rn.f16x2 p2, fa0, fb0;
fma.rn.f16x2 p2, fa1, fb1, p2;
fma.rn.f16x2 p2, fa2, fb2, p2;
fma.rn.f16x2 p2, fa3, fb3, p2;
mov.b32 {a0, a1, a2, a3}, $6;
mov.b32 {b0, b1, b2, b3}, $14;
cvt.rn.f16x2.e2m1x2 fa0, a0;
cvt.rn.f16x2.e2m1x2 fa1, a1;
cvt.rn.f16x2.e2m1x2 fa2, a2;
cvt.rn.f16x2.e2m1x2 fa3, a3;
cvt.rn.f16x2.e2m1x2 fb0, b0;
cvt.rn.f16x2.e2m1x2 fb1, b1;
cvt.rn.f16x2.e2m1x2 fb2, b2;
cvt.rn.f16x2.e2m1x2 fb3, b3;
fma.rn.f16x2 p2, fa0, fb0, p2;
fma.rn.f16x2 p2, fa1, fb1, p2;
fma.rn.f16x2 p2, fa2, fb2, p2;
fma.rn.f16x2 p2, fa3, fb3, p2;
mov.b32 {h0, h1}, p2;
cvt.f32.f16 f0, h0;
cvt.f32.f16 f1, h1;
add.f32 acc2, f0, f1;
fma.rn.f32 acc0, acc2, s2, acc0;

mov.b32 {a0, a1, a2, a3}, $7;
mov.b32 {b0, b1, b2, b3}, $15;
cvt.rn.f16x2.e2m1x2 fa0, a0;
cvt.rn.f16x2.e2m1x2 fa1, a1;
cvt.rn.f16x2.e2m1x2 fa2, a2;
cvt.rn.f16x2.e2m1x2 fa3, a3;
cvt.rn.f16x2.e2m1x2 fb0, b0;
cvt.rn.f16x2.e2m1x2 fb1, b1;
cvt.rn.f16x2.e2m1x2 fb2, b2;
cvt.rn.f16x2.e2m1x2 fb3, b3;
mul.rn.f16x2 p3, fa0, fb0;
fma.rn.f16x2 p3, fa1, fb1, p3;
fma.rn.f16x2 p3, fa2, fb2, p3;
fma.rn.f16x2 p3, fa3, fb3, p3;
mov.b32 {a0, a1, a2, a3}, $8;
mov.b32 {b0, b1, b2, b3}, $16;
cvt.rn.f16x2.e2m1x2 fa0, a0;
cvt.rn.f16x2.e2m1x2 fa1, a1;
cvt.rn.f16x2.e2m1x2 fa2, a2;
cvt.rn.f16x2.e2m1x2 fa3, a3;
cvt.rn.f16x2.e2m1x2 fb0, b0;
cvt.rn.f16x2.e2m1x2 fb1, b1;
cvt.rn.f16x2.e2m1x2 fb2, b2;
cvt.rn.f16x2.e2m1x2 fb3, b3;
fma.rn.f16x2 p3, fa0, fb0, p3;
fma.rn.f16x2 p3, fa1, fb1, p3;
fma.rn.f16x2 p3, fa2, fb2, p3;
fma.rn.f16x2 p3, fa3, fb3, p3;
mov.b32 {h0, h1}, p3;
cvt.f32.f16 f0, h0;
cvt.f32.f16 f1, h1;
add.f32 acc3, f0, f1;
fma.rn.f32 tile_result, acc3, s3, acc0;
fma.rn.f32 $0, tile_result, one, $19;
}
"""
    # Inputs: 8 uint32 for A, 8 uint32 for B, 2 uint32 for scales
    # Total: 1 output (Float32) + 18 inputs = 19 constraints
    var result = inlined_assembly[
        asm_code,
        Float32,
        constraints="=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,f",
        has_side_effect=True,
    ](
        UInt32(a_tile[0]),
        UInt32(a_tile[1]),
        UInt32(a_tile[2]),
        UInt32(a_tile[3]),  # $1-$4
        UInt32(a_tile[4]),
        UInt32(a_tile[5]),
        UInt32(a_tile[6]),
        UInt32(a_tile[7]),  # $5-$8
        UInt32(b_tile[0]),
        UInt32(b_tile[1]),
        UInt32(b_tile[2]),
        UInt32(b_tile[3]),  # $9-$12
        UInt32(b_tile[4]),
        UInt32(b_tile[5]),
        UInt32(b_tile[6]),
        UInt32(b_tile[7]),  # $13-$16
        UInt32(sfa_packed),  # $17
        UInt32(sfb_packed),  # $18
        accum_val,  # $19
    )

    return result


fn set_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
    width: Int,
](
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    row_idx: Int,
    col_idx: Int,
    scale_value: SIMD[scales_dtype, width],
):
    constrained[
        scales_tensor.rank == 5,
        "scales_tensor must be 5D for non-batched scales tensor",
    ]()
    __comptime_assert (
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


fn get_scale_factor[
    scales_dtype: DType,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
    width: Int = 1,
](
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    row_idx: Int,
    col_idx: Int,
) -> SIMD[scales_dtype, width]:
    constrained[
        scales_tensor.rank == 5,
        "scales_tensor must be 5D for non-batched scales tensor",
    ]()
    __comptime_assert (
        width <= SF_ATOM_K
    ), "width must be less than or equal to SF_ATOM_K"

    return rebind[SIMD[scales_dtype, width]](
        scales_tensor.aligned_load[width=width](
            IndexList[5](
                row_idx // SF_MN_GROUP_SIZE,
                col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
                row_idx % SF_ATOM_M[0],
                (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
                (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
            )
        )
    )


fn set_batched_scale_factor[
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
    constrained[
        scales_tensor.rank == 6,
        "scales_tensor must be 6D for batched scales tensor",
    ]()

    scales_tensor[
        batch_idx,
        row_idx // SF_MN_GROUP_SIZE,
        col_idx // (SF_VECTOR_SIZE * SF_ATOM_K),
        row_idx % SF_ATOM_M[0],
        (row_idx % SF_MN_GROUP_SIZE) // SF_ATOM_M[0],
        (col_idx // SF_VECTOR_SIZE) % SF_ATOM_K,
    ] = rebind[Scalar[scales_dtype]](scale_value)


fn get_batched_scale_factor[
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
    constrained[
        scales_tensor.rank == 6,
        "scales_tensor must be 6D for batched scales tensor",
    ]()

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


fn convert_ref_scales_to_mxfp8_format[
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
    m: ValOrDim,
    n: ValOrDim,
    k: ValOrDim,
    ref_a_scales: LayoutTensor[ref_scales_type, ref_a_scales_layout, _],
    ref_b_scales: LayoutTensor[ref_scales_type, ref_b_scales_layout, _],
    a_scales: LayoutTensor[scales_type, a_scales_layout, a_scales_origin],
    b_scales: LayoutTensor[scales_type, b_scales_layout, b_scales_origin],
):
    __comptime_assert (
        ref_scales_type == DType.float32
    ), "Only support float32 reference scales"
    __comptime_assert (
        scales_type == DType.float8_e8m0fnu
    ), "Only support float8_e8m0fnu scales"
    __comptime_assert ref_a_scales_layout.rank() == 2, "ref_a_scales must be 2D"
    __comptime_assert ref_b_scales_layout.rank() == 2, "ref_b_scales must be 2D"
    __comptime_assert a_scales_layout.rank() == 5, "a_scales must be 5D"
    __comptime_assert b_scales_layout.rank() == 5, "b_scales must be 5D"

    var M = m.value
    var N = n.value
    var K = k.value

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
