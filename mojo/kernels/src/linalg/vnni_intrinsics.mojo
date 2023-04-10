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

# ===----------------------------------------------------------------------===#
# vpdpwssd
# ===----------------------------------------------------------------------===#


fn vpdpwssd(
    src: SIMD[DType.si32, 16],
    a: SIMD[DType.si32, 16],
    b: SIMD[DType.si32, 16],
) -> SIMD[DType.si32, 16]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssd.512", __mlir_type.`!pop.simd<16, si32>`
    ](src.value, a.value, b.value)


fn vpdpwssd(
    src: SIMD[DType.si32, 8],
    a: SIMD[DType.si32, 8],
    b: SIMD[DType.si32, 8],
) -> SIMD[DType.si32, 8]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssd.256", __mlir_type.`!pop.simd<8, si32>`
    ](src.value, a.value, b.value)


fn vpdpwssd(
    src: SIMD[DType.si32, 4],
    a: SIMD[DType.si32, 4],
    b: SIMD[DType.si32, 4],
) -> SIMD[DType.si32, 4]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssd.128", __mlir_type.`!pop.simd<4, si32>`
    ](src.value, a.value, b.value)


# ===----------------------------------------------------------------------===#
# vpdpwssds
# ===----------------------------------------------------------------------===#


fn vpdpwssds(
    src: SIMD[DType.si32, 16],
    a: SIMD[DType.si32, 16],
    b: SIMD[DType.si32, 16],
) -> SIMD[DType.si32, 16]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.512", __mlir_type.`!pop.simd<16, si32>`
    ](src.value, a.value, b.value)


fn vpdpwssds(
    src: SIMD[DType.si32, 8],
    a: SIMD[DType.si32, 8],
    b: SIMD[DType.si32, 8],
) -> SIMD[DType.si32, 8]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.256", __mlir_type.`!pop.simd<8, si32>`
    ](src.value, a.value, b.value)


fn vpdpwssds(
    src: SIMD[DType.si32, 4],
    a: SIMD[DType.si32, 4],
    b: SIMD[DType.si32, 4],
) -> SIMD[DType.si32, 4]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.128", __mlir_type.`!pop.simd<4, si32>`
    ](src.value, a.value, b.value)


# ===----------------------------------------------------------------------===#
# vpdpbusd
# ===----------------------------------------------------------------------===#


fn vpdpbusd(
    src: SIMD[DType.si32, 16],
    a: SIMD[DType.si32, 16],
    b: SIMD[DType.si32, 16],
) -> SIMD[DType.si32, 16]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.512", __mlir_type.`!pop.simd<16, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusd(
    src: SIMD[DType.si32, 8],
    a: SIMD[DType.si32, 8],
    b: SIMD[DType.si32, 8],
) -> SIMD[DType.si32, 8]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.256", __mlir_type.`!pop.simd<8, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusd(
    src: SIMD[DType.si32, 4],
    a: SIMD[DType.si32, 4],
    b: SIMD[DType.si32, 4],
) -> SIMD[DType.si32, 4]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusd.128", __mlir_type.`!pop.simd<4, si32>`
    ](src.value, a.value, b.value)


# ===----------------------------------------------------------------------===#
# vpdpbusds
# ===----------------------------------------------------------------------===#


fn vpdpbusds(
    src: SIMD[DType.si32, 16],
    a: SIMD[DType.si32, 16],
    b: SIMD[DType.si32, 16],
) -> SIMD[DType.si32, 16]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.512", __mlir_type.`!pop.simd<16, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusds(
    src: SIMD[DType.si32, 8],
    a: SIMD[DType.si32, 8],
    b: SIMD[DType.si32, 8],
) -> SIMD[DType.si32, 8]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.256", __mlir_type.`!pop.simd<8, si32>`
    ](src.value, a.value, b.value)


fn vpdpbusds(
    src: SIMD[DType.si32, 4],
    a: SIMD[DType.si32, 4],
    b: SIMD[DType.si32, 4],
) -> SIMD[DType.si32, 4]:
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.128", __mlir_type.`!pop.simd<4, si32>`
    ](src.value, a.value, b.value)
