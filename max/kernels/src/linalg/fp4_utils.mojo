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

    var result = SIMD[out_dtype, out_width]()

    comptime for i in range(in_width):
        comptime for shift in range(0, num_fp4_values):
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


fn cast_fp32_to_fp4e2m1[
    width: Int,
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


fn cast_f4e2m1x2_to_fp16x2(x: Scalar[DType.uint8]) -> SIMD[DType.float16, 2]:
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


fn get_scale_factor[
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
