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

This module provides MMA implementations for AMD CDNA2, CDNA3, and CDNA4 data
center GPUs using the MFMA (Matrix Fused Multiply-Add) instructions.

Reference: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
"""

from std.sys import llvm_intrinsic
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
def _mma_wmma_cdna(mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    comptime fp8_dtypes = (DType.float8_e4m3fn, DType.float8_e5m2)

    def _wmma_type_name[dtype: DType]() -> StaticString:
        comptime if dtype == .float32:
            return ".f32"
        elif dtype == .float16:
            return ".f16"
        elif dtype == .bfloat16:
            return ".bf16"
        elif dtype == .float8_e4m3fn:
            return ".fp8"
        elif dtype == .float8_e5m2:
            return ".bf8"
        else:
            comptime assert False, "unsupported dtype"

    def _wmma_intrinsic_base[
        intrinsic_name: StaticString
    ](a: SIMD, b: SIMD) {imm} -> SIMD[d.dtype, d.length]:
        return llvm_intrinsic[intrinsic_name, SIMD[d.dtype, d.length]](
            a,
            b,
            UInt16(0),
            c,
            False,
            False,
        )

    def _wmma_intrinsic[
        shape_name: StaticString,
        *,
        d_type: StaticString = _wmma_type_name[d.dtype](),
    ]() {imm} -> SIMD[d.dtype, d.length]:
        comptime a_type = _wmma_type_name[a.dtype]()
        comptime intrinsic_name = "llvm.amdgcn.wmma" + d_type + shape_name + a_type

        return _wmma_intrinsic_base[intrinsic_name](a, b)

    def _wmma_intrinsic_float8[
        shape_name: StaticString
    ]() {imm} -> SIMD[d.dtype, d.length]:
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
