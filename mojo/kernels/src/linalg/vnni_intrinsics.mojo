# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file contains wrappers around Intel VNNI intrinsics. See
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&avx512techs=AVX512_VNNI
#
# ===----------------------------------------------------------------------===#

from DType import DType
from SIMD import SIMD
from Intrinsics import llvm_intrinsic
from TypeUtilities import rebind
from Assert import assert_param

# ===----------------------------------------------------------------------===#
# vpdpwssd
# ===----------------------------------------------------------------------===#


fn vpdpwssd(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int32, 16],
    b: SIMD[DType.int32, 16],
) -> SIMD[DType.int32, 16]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssd.512", __mlir_type.`!pop.simd<16, si32>`
    ](src.value, a.value, b.value)


fn vpdpwssd(
    src: SIMD[DType.int32, 8],
    a: SIMD[DType.int32, 8],
    b: SIMD[DType.int32, 8],
) -> SIMD[DType.int32, 8]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssd.256", __mlir_type.`!pop.simd<8, si32>`
    ](src.value, a.value, b.value)


fn vpdpwssd(
    src: SIMD[DType.int32, 4],
    a: SIMD[DType.int32, 4],
    b: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssd.128", __mlir_type.`!pop.simd<4, si32>`
    ](src.value, a.value, b.value)


# ===----------------------------------------------------------------------===#
# vpdpwssds
# ===----------------------------------------------------------------------===#


fn vpdpwssds(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int32, 16],
    b: SIMD[DType.int32, 16],
) -> SIMD[DType.int32, 16]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.512", __mlir_type.`!pop.simd<16, si32>`
    ](src.value, a.value, b.value)


fn vpdpwssds(
    src: SIMD[DType.int32, 8],
    a: SIMD[DType.int32, 8],
    b: SIMD[DType.int32, 8],
) -> SIMD[DType.int32, 8]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.256", __mlir_type.`!pop.simd<8, si32>`
    ](src.value, a.value, b.value)


fn vpdpwssds(
    src: SIMD[DType.int32, 4],
    a: SIMD[DType.int32, 4],
    b: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.128", __mlir_type.`!pop.simd<4, si32>`
    ](src.value, a.value, b.value)


# ===----------------------------------------------------------------------===#
# vpdpbusd
# ===----------------------------------------------------------------------===#


fn vpdpbusd_16(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int32, 16],
    b: SIMD[DType.int32, 16],
) -> SIMD[DType.int32, 16]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.512", __mlir_type.`!pop.simd<16, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusd_8(
    src: SIMD[DType.int32, 8],
    a: SIMD[DType.int32, 8],
    b: SIMD[DType.int32, 8],
) -> SIMD[DType.int32, 8]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.256", __mlir_type.`!pop.simd<8, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusd_4(
    src: SIMD[DType.int32, 4],
    a: SIMD[DType.int32, 4],
    b: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.128", __mlir_type.`!pop.simd<4, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusd[
    width: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    @parameter
    if width == 16:
        return rebind[SIMD[c_type, width]](
            vpdpbusd_16(
                rebind[SIMD[DType.int32, 16]](src),
                rebind[SIMD[DType.int32, 16]](a),
                rebind[SIMD[DType.int32, 16]](b),
            )
        )
    elif width == 8:
        return rebind[SIMD[c_type, width]](
            vpdpbusd_8(
                rebind[SIMD[DType.int32, 8]](src),
                rebind[SIMD[DType.int32, 8]](a),
                rebind[SIMD[DType.int32, 8]](b),
            )
        )
    else:
        assert_param[width == 4]()
        return rebind[SIMD[c_type, width]](
            vpdpbusd_4(
                rebind[SIMD[DType.int32, 4]](src),
                rebind[SIMD[DType.int32, 4]](a),
                rebind[SIMD[DType.int32, 4]](b),
            )
        )


# ===----------------------------------------------------------------------===#
# vpdpbusds
# ===----------------------------------------------------------------------===#


fn vpdpbusds(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int32, 16],
    b: SIMD[DType.int32, 16],
) -> SIMD[DType.int32, 16]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.512", __mlir_type.`!pop.simd<16, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusds(
    src: SIMD[DType.int32, 8],
    a: SIMD[DType.int32, 8],
    b: SIMD[DType.int32, 8],
) -> SIMD[DType.int32, 8]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.256", __mlir_type.`!pop.simd<8, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusds(
    src: SIMD[DType.int32, 4],
    a: SIMD[DType.int32, 4],
    b: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.128", __mlir_type.`!pop.simd<4, si32>`
    ](src.value, a.value, b.value)
