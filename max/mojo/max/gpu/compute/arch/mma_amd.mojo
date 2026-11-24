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
"""AMD CDNA Matrix Cores implementation for matrix multiply-accumulate operations.

This module provides MMA implementations for AMD CDNA2, CDNA3 and CDNA4 data
center GPUs using the MFMA (Matrix Fused Multiply-Add) instructions, and for
CDNA5 (gfx1250), whose matrix cores take the `v_wmma` instructions instead.

Reference: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
"""

from std.sys import bit_width_of, llvm_intrinsic
from std.sys.info import _cdna_4_or_newer, _cdna_5_or_newer, _is_amd_rdna
from std.memory import bitcast

# Import helper functions from parent module
from ..mma import (
    _has_type,
    _has_shape,
    _unsupported_mma_op,
    get_amd_fp8_dtype,
    get_amd_bf8_dtype,
)

# Import RDNA implementation for consumer GPUs
from .mma_amd_rdna import _mma_wmma_rdna


@inline(.always)
def _mma_wmma_cdna[
    *,
    a_reuse: Bool = False,
    b_reuse: Bool = False,
](mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    comptime fp8_dtypes = (DType.float8_e4m3fn, DType.float8_e5m2)

    def _wmma_type_name[dtype: DType]() -> StaticString:
        comptime match dtype:
            case .float32:
                return ".f32"
            case .float16:
                return ".f16"
            case .bfloat16:
                return ".bf16"
            case .float8_e4m3fn:
                return ".fp8"
            case .float8_e5m2:
                return ".bf8"
            case _:
                comptime assert False, "unsupported dtype"

    def _wmma_intrinsic_base[
        intrinsic_name: StaticString
    ](a: SIMD, b: SIMD) {imm} -> type_of(d):
        # The i16 operand selects which halves of C the result negates: bit 0
        # the low half, bit 1 the high half, so 0 leaves C unmodified.
        return llvm_intrinsic[intrinsic_name, type_of(d)](
            a,
            b,
            Int16(0),
            c,
            a_reuse,
            b_reuse,
        )

    def _wmma_intrinsic[
        shape_name: StaticString,
        *,
        d_type: StaticString = _wmma_type_name[d.dtype](),
    ]() {imm} -> type_of(d):
        comptime a_type = _wmma_type_name[a.dtype]()
        comptime intrinsic_name = "llvm.amdgcn.wmma" + d_type + shape_name + a_type

        return _wmma_intrinsic_base[intrinsic_name](a, b)

    def _wmma_intrinsic_float8[shape_name: StaticString]() {imm} -> type_of(d):
        comptime a_type = _wmma_type_name[a.dtype]()
        comptime b_type = _wmma_type_name[b.dtype]()
        comptime d_type = _wmma_type_name[d.dtype]()
        comptime intrinsic_name = "llvm.amdgcn.wmma" + d_type + shape_name + a_type + b_type

        return _wmma_intrinsic_base[intrinsic_name](
            bitcast[.int32, a.length // 4](a), bitcast[.int32, b.length // 4](b)
        )

    # ===------------------------------------------------------------------===#
    # F16 = [F8 or BF8] * [F8 or BF8] + F16
    # F32 = [F8 or BF8] * [F8 or BF8] + F32
    # ===------------------------------------------------------------------===#
    comptime if (
        a.dtype in fp8_dtypes
        and b.dtype in fp8_dtypes
        and c.dtype == d.dtype
        and d.dtype
        in (
            DType.float16,
            DType.float32,
        )
    ):
        comptime if _has_shape[(32, 32, 8, 8)](
            a.length, b.length, c.length, d.length
        ):
            d = _wmma_intrinsic_float8[".16x16x64"]()
        elif _has_shape[(64, 64, 8, 8)](a.length, b.length, c.length, d.length):
            d = _wmma_intrinsic_float8[".16x16x128"]()
        else:
            _unsupported_mma_op[
                d.dtype,
                d.length,
                a.dtype,
                a.length,
                b.dtype,
                b.length,
                c.dtype,
                c.length,
            ]()

    # ===------------------------------------------------------------------===#
    # F16 = F16 * F16 + F16
    # F32 = F16 * F16 + F32
    # BF16 = BF16 * BF16 + BF16
    # F32 = BF16 * BF16 + F32
    # ===------------------------------------------------------------------===#
    elif (
        a.dtype == b.dtype
        and a.dtype.is_half_float()
        and c.dtype == d.dtype
        and (d.dtype in (a.dtype, DType.float32))
        and _has_shape[(16, 16, 8, 8)](a.length, b.length, c.length, d.length)
    ):
        d = _wmma_intrinsic[".16x16x32"]()

    # ===------------------------------------------------------------------===#
    # BF16 = BF16 * BF16 + F32
    # ===------------------------------------------------------------------===#
    elif (
        _has_type[
            (DType.bfloat16, DType.bfloat16, DType.float32, DType.bfloat16)
        ](a.dtype, b.dtype, c.dtype, d.dtype)
        and _has_shape[(16, 16, 8, 8)](a.length, b.length, c.length, d.length)
    ):
        d = _wmma_intrinsic[".16x16x32", d_type=".bf16f32"]()

    # ===------------------------------------------------------------------===#
    # F32 = F32 * F32 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[DType.float32](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(2, 2, 8, 8)](a.length, b.length, c.length, d.length):
        d = _wmma_intrinsic[".16x16x4"]()

    else:
        _unsupported_mma_op[
            d.dtype,
            d.length,
            a.dtype,
            a.length,
            b.dtype,
            b.length,
            c.dtype,
            c.length,
        ]()


@inline(.always)
def wmma_cdna[
    *,
    a_reuse: Bool = False,
    b_reuse: Bool = False,
](mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    """Computes one CDNA5 WMMA, `d = a * b + c`, with the given reuse hints.

    The instruction is picked from the dtypes and lane widths of the operands,
    the same way the generic `mma` picks it. Every combination below is one
    `v_wmma` instruction whose A and B operands live in registers, so the
    fragments are the per-lane slices of the 16x16 output tile.

    Supported combinations, as (A, B, C, D dtypes, lane widths of A/B/C/D):

    - (float8_e4m3fn or float8_e5m2 for A and B, float16 or float32 for C and
      D, 32/32/8/8), `16x16x64`. A and B pair freely, giving the `fp8.fp8`,
      `fp8.bf8`, `bf8.fp8` and `bf8.bf8` instruction forms.
    - The same four pairings and accumulators at 64/64/8/8, `16x16x128`.
    - (float16, float16, float16 or float32, 16/16/8/8) and (bfloat16,
      bfloat16, bfloat16 or float32, 16/16/8/8), `16x16x32`.
    - (bfloat16, bfloat16, float32, bfloat16, 16/16/8/8), `16x16x32`: the mixed
      form whose f32 accumulator becomes a bf16 result.
    - (float32, float32, float32, 2/2/8/8), `16x16x4`.

    Parameters:
        a_reuse: Hint that this WMMA reads the same A registers as the WMMA
            immediately before it in the final instruction order, which lets
            the matrix core skip reloading them.
        b_reuse: The same hint for the B registers.

    Args:
        d: The destination fragment, overwritten with the result.
        a: The A fragment.
        b: The B fragment.
        c: The accumulator fragment.

    Constraints:
        - Requires CDNA5 (gfx1250) or newer; the call fails to compile
          elsewhere.
        - The lane widths of A, B, C and D must match the instruction shape
          for their dtypes, as listed above.
        - A set reuse hint asserts that this WMMA reads the same A (or B)
          registers as the WMMA immediately before it in the final instruction
          order. The compiler may reorder WMMAs, so a caller must only set a
          hint where that order is guaranteed.
    """
    comptime assert _cdna_5_or_newer(), "wmma_cdna requires CDNA5 (gfx1250)"

    _mma_wmma_cdna[a_reuse=a_reuse, b_reuse=b_reuse](d, a, b, c)


comptime _WMMA_FORMAT_DTYPES = (
    DType.float8_e4m3fn,
    DType.float8_e5m2,
    DType.float6_e2m3fn,
    DType.float6_e3m2fn,
    DType.float4_e2m1fn,
)
"""The element dtypes a block-scaled WMMA decodes, in ISA format order.

A fragment held in one of these is the format's own data, not raw bits, so
its element type and the dtype it is passed with have to agree.
"""


def _wmma_matrix_fmt[dtype: DType]() -> Int32:
    """Returns the `matrix_a_fmt`/`matrix_b_fmt` code that decodes `dtype`."""
    comptime match dtype:
        case .float8_e4m3fn:
            return 0
        case .float8_e5m2:
            return 1
        case .float6_e2m3fn:
            return 2
        case .float6_e3m2fn:
            return 3
        case .float4_e2m1fn:
            return 4
        case _:
            comptime assert False, (
                "a block-scaled WMMA operand is float8_e4m3fn, float8_e5m2,"
                " float6_e2m3fn, float6_e3m2fn or float4_e2m1fn"
            )


def _wmma_scale_fmt[dtype: DType]() -> Int32:
    """Returns the `matrix_a_scale_fmt`/`matrix_b_scale_fmt` code of `dtype`."""
    comptime match dtype:
        case .float8_e8m0fnu:
            return 0
        case .float8_e4m3fn:
            return 2
        case _:
            comptime assert (
                False
            ), "a block-scaled WMMA scale is float8_e8m0fnu or float8_e4m3fn"


def _wmma_f8f6f4_operand_regs[dtype: DType]() -> Int:
    """Returns the 32-bit registers one 16x16x128 operand needs per lane.

    The instruction takes the same 64 elements per lane whatever the type, so
    the register tuple shrinks as the elements do: 16 registers of an 8-bit
    type, 12 of a 6-bit one and 8 of fp4.
    """
    comptime match dtype:
        case .float4_e2m1fn:
            return 8
        case .float6_e2m3fn | .float6_e3m2fn:
            # 12 registers is what the ISA asks for, but a SIMD length has to
            # be a power of two, so `wmma_scale_cdna` has to reject both types.
            return 12
        case _:
            return 16


@inline(.always)
def wmma_scale_cdna[
    *,
    a_type: DType,
    b_type: DType,
    scale_type: DType = DType.float8_e8m0fnu,
    scale16: Bool = False,
    a_scale_sel: Int = 0,
    b_scale_sel: Int = 0,
    a_reuse: Bool = False,
    b_reuse: Bool = False,
](
    mut d: SIMD[DType.float32, _],
    a: SIMD,
    b: SIMD,
    c: SIMD[DType.float32, _],
    a_scale: Scalar[DType.int64 if scale16 else DType.int32],
    b_scale: Scalar[DType.int64 if scale16 else DType.int32],
):
    """Computes one block-scaled CDNA5 WMMA, `d = a * b * scale + c`.

    The instruction is `v_wmma_scale`, or `v_wmma_scale16` with `scale16`,
    which multiplies every product of a block of the K dimension by that
    block's scale on top of the format's own per-element decode. A and B are
    expected to be the per-lane packed registers of the fragments, not one
    element per lane. The shape follows the accumulator width:

    - 8 f32 lanes: `16x16x128` over `f8f6f4`, so `a_type` and `b_type` pick
      each operand's format independently and 64 elements travel per lane.
    - 16 f32 lanes: `32x16x128` over `f4`, which is fp4 on both sides and 128
      elements of A against 64 of B per lane.

    The common block-scaling schemes, as the parameters they need:

    | Scheme | `a_type`, `b_type` | `scale_type` | `scale16` |
    |---|---|---|---|
    | MXFP8 | `float8_e4m3fn`, `float8_e5m2` | `float8_e8m0fnu` | False |
    | MXFP6 | `float6_e2m3fn`, `float6_e3m2fn` | `float8_e8m0fnu` | False |
    | MXFP4 | `float4_e2m1fn` | `float8_e8m0fnu` | False |
    | NVFP4-style | `float4_e2m1fn` | `float8_e4m3fn` | True |

    `scale16` is the ISA's `v_wmma_scale16` form, which takes two 32-bit scale
    registers per lane rather than one. Its block size is believed to be 16
    elements, against 32 for `scale`, but that is not confirmed here. The
    per-tensor fp32 scale of NVFP4 is not part of either instruction: apply it
    outside this call.

    MXFP6 is listed for completeness but cannot be called yet: its 12-register
    tuple is a 12-lane vector, and a SIMD length has to be a power of two, so
    no A or B value can hold one. The float6 types are rejected rather than
    silently widened.

    The ISA's third scale format, E5M3, has no `DType` and is not reachable.

    Parameters:
        a_type: The element type A's registers hold.
        b_type: The element type B's registers hold.
        scale_type: The type both operands' block scales are encoded in.
        scale16: Whether to issue the two-register `v_wmma_scale16` form.
        a_scale_sel: Which of A's scale rows to use, 0 or 1.
        b_scale_sel: Which of B's scale rows to use, 0 or 1.
        a_reuse: Hint that this WMMA reads the same A registers as the WMMA
            immediately before it in the final instruction order, which lets
            the matrix core skip reloading them.
        b_reuse: The same hint for the B registers.

    Args:
        d: The destination fragment, overwritten with the result.
        a: The A fragment: the format's packed registers in any container that
            fills the 32-bit register tuple it asks for, such as 16 `int32`
            lanes of fp8 or 64 `_uint4` lanes of fp4, or `a_type` elements.
        b: The B fragment's packed registers, the same way.
        c: The accumulator fragment.
        a_scale: A's block scales, 32-bit per lane, or 64-bit with `scale16`.
        b_scale: B's block scales, the same way.

    Constraints:
        - Requires CDNA5 (gfx1250) or newer; the call fails to compile
          elsewhere.
        - `a_type` and `b_type` are each `float8_e4m3fn`, `float8_e5m2` or
          `float4_e2m1fn`; `scale_type` is `float8_e8m0fnu` or
          `float8_e4m3fn`.
        - Only a `float4_e2m1fn` A and B pair takes `float8_e4m3fn` scales;
          every other pair takes `float8_e8m0fnu`.
        - The 16-lane shape is `float4_e2m1fn` only, because its instruction
          carries no format.
        - The float6 types are rejected: the 12-register tuple they ask for
          is a 12-lane vector, which no SIMD length can hold.
        - The widths of A and B must fill the register tuple their type asks
          for: 16 32-bit registers for the float8 types and 8 for
          `float4_e2m1fn` in the 8-lane shape, and 16 for A against 8 for B
          in the 32x16x128 shape.
        - A or B held in one of the types above must be `a_type` or `b_type`
          respectively; an integer container is the raw packed bits, sized by
          its width alone.
        - A set reuse hint asserts that this WMMA reads the same A (or B)
          registers as the WMMA immediately before it in the final instruction
          order. The compiler may reorder WMMAs, so a caller must only set a
          hint where that order is guaranteed.
    """
    comptime assert (
        _cdna_5_or_newer()
    ), "wmma_scale_cdna requires CDNA5 (gfx1250)"
    comptime assert (
        d.length == 8 or d.length == 16
    ), t"wmma_scale_cdna has no shape for a {Int(d.length)}-lane accumulator"
    comptime assert (
        c.length == d.length
    ), "wmma_scale_cdna needs C to have the accumulator's lane count"
    comptime assert (
        a_type != DType.float6_e2m3fn
        and a_type != DType.float6_e3m2fn
        and b_type != DType.float6_e2m3fn
        and b_type != DType.float6_e3m2fn
    ), (
        "the float6 types take 12 packed registers, a 12-lane vector, and a"
        " SIMD length has to be a power of two, so neither can be passed here"
    )
    comptime fp4_pair = a_type == DType.float4_e2m1fn and (
        b_type == DType.float4_e2m1fn
    )
    comptime assert scale_type == DType.float8_e8m0fnu or fp4_pair, (
        "only a float4_e2m1fn A and B pair takes float8_e4m3fn scales; every"
        " other pair takes float8_e8m0fnu"
    )
    comptime assert a_scale_sel in (
        0,
        1,
    ), t"a_scale_sel is a scale row, 0 or 1, not {a_scale_sel}"
    comptime assert b_scale_sel in (
        0,
        1,
    ), t"b_scale_sel is a scale row, 0 or 1, not {b_scale_sel}"

    # The 32x16x128 f4 form has a fixed width per operand, since both are fp4;
    # the 16x16x128 f8f6f4 form sizes both from the format.
    comptime a_regs = (
        16 if d.length == 16 else _wmma_f8f6f4_operand_regs[a_type]()
    )
    comptime b_regs = (
        8 if d.length == 16 else _wmma_f8f6f4_operand_regs[b_type]()
    )
    comptime assert a.dtype == a_type or a.dtype not in _WMMA_FORMAT_DTYPES, (
        t"A holds {a.dtype} elements but a_type is {a_type}: a typed fragment"
        t" must be the type it decodes as, and raw packed bits go in an"
        t" integer dtype"
    )
    comptime assert b.dtype == b_type or b.dtype not in _WMMA_FORMAT_DTYPES, (
        t"B holds {b.dtype} elements but b_type is {b_type}: a typed fragment"
        t" must be the type it decodes as, and raw packed bits go in an"
        t" integer dtype"
    )
    comptime a_fmt = _wmma_matrix_fmt[a_type]()
    comptime b_fmt = _wmma_matrix_fmt[b_type]()
    comptime scale_fmt = _wmma_scale_fmt[scale_type]()
    # `bit_width_of` measures the whole packed vector, the same width the
    # `bitcast` below checks, so a sub-byte container such as `_uint4` counts
    # its real width rather than a byte per element.
    comptime a_bits = bit_width_of[type_of(a)]()
    comptime b_bits = bit_width_of[type_of(b)]()
    comptime assert a_bits == a_regs * 32, (
        t"A must pack {a_regs * 32} bits, the register tuple its format asks"
        t" for, but has {a_bits}"
    )
    comptime assert b_bits == b_regs * 32, (
        t"B must pack {b_regs * 32} bits, the register tuple its format asks"
        t" for, but has {b_bits}"
    )

    comptime form = "scale16" if scale16 else "scale"

    comptime if d.length == 8:
        d = llvm_intrinsic[
            "llvm.amdgcn.wmma." + form + ".f32.16x16x128.f8f6f4",
            type_of(d),
        ](
            a_fmt,
            bitcast[.int32, a_regs](a),
            b_fmt,
            bitcast[.int32, b_regs](b),
            Int16(0),
            c,
            Int32(a_scale_sel),
            scale_fmt,
            a_scale,
            Int32(b_scale_sel),
            scale_fmt,
            b_scale,
            a_reuse,
            b_reuse,
        )
    else:
        comptime assert (
            fp4_pair
        ), "the 32x16x128 shape is float4_e2m1fn on both sides"
        d = llvm_intrinsic[
            "llvm.amdgcn.wmma." + form + ".f32.32x16x128.f4",
            type_of(d),
        ](
            bitcast[.int32, a_regs](a),
            bitcast[.int32, b_regs](b),
            Int16(0),
            c,
            Int32(a_scale_sel),
            scale_fmt,
            a_scale,
            Int32(b_scale_sel),
            scale_fmt,
            b_scale,
            a_reuse,
            b_reuse,
        )


@fieldwise_init
struct _AMD_F8F6F4_MATRIX_FORMAT(TrivialRegisterPassable):
    """Represents the matrix format value to control the type and shape for the inputs
    of the llvm.amdgcn.mfma.scale.f8f6f4 intrinsics.
    """

    var _value: Int32
    comptime float8_e4m3 = Self(0)
    comptime float8_e5m2 = Self(1)
    comptime float6_e2m3 = Self(2)
    comptime float6_e3m2 = Self(3)
    comptime float4_e2m1 = Self(4)

    def __init__(out self, value: Int):
        self._value = Int32(value)


@inline(.always)
def _mma_mfma_cdna(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    comptime zero: UInt32 = 0

    # CDNA3 supports the FNUZ float8 dtypes, and CDNA4 supports the Open
    # Compute Project (OCP) float8 dtypes.
    comptime fp8_dtype = get_amd_fp8_dtype().value()
    comptime bf8_dtype = get_amd_bf8_dtype().value()

    def _f8f6f4_intrinsic() {imm} -> SIMD[d.dtype, d.length]:
        comptime assert _cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"

        comptime intrinsic_name = "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4" if _has_shape[
            (32, 32, 4, 4)
        ](
            a.length, b.length, c.length, d.length
        ) else "llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4"

        def _matrix_format[dtype: DType]() -> _AMD_F8F6F4_MATRIX_FORMAT:
            return (
                _AMD_F8F6F4_MATRIX_FORMAT.float8_e4m3 if dtype
                == fp8_dtype else _AMD_F8F6F4_MATRIX_FORMAT.float8_e5m2
            )

        return llvm_intrinsic[intrinsic_name, SIMD[d.dtype, d.length]](
            bitcast[.int32, 8](a),
            bitcast[.int32, 8](b),
            c,
            _matrix_format[a.dtype](),
            _matrix_format[b.dtype](),
            zero,
            zero,
            zero,
            zero,
        )

    # ===------------------------------------------------------------------===#
    # F16 = F16 * F16 + F16
    # ===------------------------------------------------------------------===#
    comptime assert not _has_type[DType.float16](
        a.dtype, b.dtype, c.dtype, d.dtype
    ), "Function mma F16 * F16 + F16 is unsupported by AMD GPUs."

    # ===------------------------------------------------------------------===#
    # F32 = F16 * F16 + F32
    # ===------------------------------------------------------------------===#
    comptime if _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[4](
        a.length, b.length, c.length, d.length
    ):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x16f16", SIMD[d.dtype, d.length]
        ](a, b, c, zero, zero, zero)
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(4, 4, 16, 16)](
        a.length, b.length, c.length, d.length
    ):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.32x32x8f16", SIMD[d.dtype, d.length]
        ](a, b, c, zero, zero, zero)
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 8, 4, 4)](
        a.length, b.length, c.length, d.length
    ):
        comptime assert _cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x32.f16", SIMD[d.dtype, d.length]
        ](a, b, c, zero, zero, zero)
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 8, 16, 16)](
        a.length, b.length, c.length, d.length
    ):
        comptime assert _cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.32x32x16.f16", SIMD[d.dtype, d.length]
        ](a, b, c, zero, zero, zero)

    # ===------------------------------------------------------------------===#
    # F32 = BF16 * BF16 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[4](
        a.length, b.length, c.length, d.length
    ):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x16bf16.1k", SIMD[d.dtype, d.length]
        ](
            bitcast[.int16, 4](a),
            bitcast[.int16, 4](b),
            c,
            zero,
            zero,
            zero,
        )
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(4, 4, 16, 16)](
        a.length, b.length, c.length, d.length
    ):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.32x32x8bf16.1k", SIMD[d.dtype, d.length]
        ](
            bitcast[.int16, 4](a),
            bitcast[.int16, 4](b),
            c,
            zero,
            zero,
            zero,
        )
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 8, 4, 4)](
        a.length, b.length, c.length, d.length
    ):
        comptime assert _cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x32.bf16", SIMD[d.dtype, d.length]
        ](a, b, c, zero, zero, zero)
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 8, 16, 16)](
        a.length, b.length, c.length, d.length
    ):
        comptime assert _cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.32x32x16.bf16", SIMD[d.dtype, d.length]
        ](a, b, c, zero, zero, zero)

    # ===------------------------------------------------------------------===#
    # F32 = F32 * F32 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[DType.float32](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(1, 1, 4, 4)](a.length, b.length, c.length, d.length):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x4f32", SIMD[d.dtype, d.length]
        ](a, b, c, zero, zero, zero)

    # ===------------------------------------------------------------------===#
    # F32 = F8 * F8 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[(fp8_dtype, fp8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(8, 8, 4, 4)](a.length, b.length, c.length, d.length):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8", SIMD[d.dtype, d.length]
        ](
            bitcast[.int64, 1](a),
            bitcast[.int64, 1](b),
            c,
            zero,
            zero,
            zero,
        )
    elif _has_type[(fp8_dtype, fp8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(32, 32, 4, 4)](a.length, b.length, c.length, d.length):
        d = _f8f6f4_intrinsic()
    elif _has_type[(fp8_dtype, fp8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(32, 32, 16, 16)](a.length, b.length, c.length, d.length):
        d = _f8f6f4_intrinsic()

    # ===------------------------------------------------------------------===#
    # F32 = BF8 * BF8 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[(bf8_dtype, bf8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(8, 8, 4, 4)](a.length, b.length, c.length, d.length):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8", SIMD[d.dtype, d.length]
        ](
            bitcast[.int64, 1](a),
            bitcast[.int64, 1](b),
            c,
            zero,
            zero,
            zero,
        )
    elif _has_type[(bf8_dtype, bf8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(32, 32, 4, 4)](a.length, b.length, c.length, d.length):
        d = _f8f6f4_intrinsic()
    elif _has_type[(bf8_dtype, bf8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(32, 32, 16, 16)](a.length, b.length, c.length, d.length):
        d = _f8f6f4_intrinsic()

    else:
        _unsupported_mma_op[
            d.dtype,
            d.length,
            a.dtype,
            a.length,
            b.dtype,
            b.length,
            c.dtype,
            c.length,
        ]()


@inline(.always)
def _mma_amd(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    comptime if _is_amd_rdna():
        _mma_wmma_rdna(d, a, b, c)
    elif _cdna_5_or_newer():
        _mma_wmma_cdna(d, a, b, c)
    else:
        _mma_mfma_cdna(d, a, b, c)
