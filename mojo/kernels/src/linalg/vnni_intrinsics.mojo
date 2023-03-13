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

# ===----------------------------------------------------------------------===#
# vpdpwssd
# ===----------------------------------------------------------------------===#


fn vpdpwssd(
    src: SIMD[16, DType.si32],
    a: SIMD[16, DType.si32],
    b: SIMD[16, DType.si32],
) -> SIMD[16, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpwssd.512",
        _type : __mlir_type.`!pop.simd<16, si32>`,
    ](src.value, a.value, b.value)


fn vpdpwssd(
    src: SIMD[8, DType.si32],
    a: SIMD[8, DType.si32],
    b: SIMD[8, DType.si32],
) -> SIMD[8, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpwssd.256",
        _type : __mlir_type.`!pop.simd<8, si32>`,
    ](src.value, a.value, b.value)


fn vpdpwssd(
    src: SIMD[4, DType.si32],
    a: SIMD[4, DType.si32],
    b: SIMD[4, DType.si32],
) -> SIMD[4, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpwssd.128",
        _type : __mlir_type.`!pop.simd<4, si32>`,
    ](src.value, a.value, b.value)


# ===----------------------------------------------------------------------===#
# vpdpwssds
# ===----------------------------------------------------------------------===#


fn vpdpwssds(
    src: SIMD[16, DType.si32],
    a: SIMD[16, DType.si32],
    b: SIMD[16, DType.si32],
) -> SIMD[16, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpwssds.512",
        _type : __mlir_type.`!pop.simd<16, si32>`,
    ](src.value, a.value, b.value)


fn vpdpwssds(
    src: SIMD[8, DType.si32],
    a: SIMD[8, DType.si32],
    b: SIMD[8, DType.si32],
) -> SIMD[8, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpwssds.256",
        _type : __mlir_type.`!pop.simd<8, si32>`,
    ](src.value, a.value, b.value)


fn vpdpwssds(
    src: SIMD[4, DType.si32],
    a: SIMD[4, DType.si32],
    b: SIMD[4, DType.si32],
) -> SIMD[4, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpwssds.128",
        _type : __mlir_type.`!pop.simd<4, si32>`,
    ](src.value, a.value, b.value)


# ===----------------------------------------------------------------------===#
# vpdpbusd
# ===----------------------------------------------------------------------===#


fn vpdpbusd(
    src: SIMD[16, DType.si32],
    a: SIMD[16, DType.si32],
    b: SIMD[16, DType.si32],
) -> SIMD[16, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpbusd.512",
        _type : __mlir_type.`!pop.simd<16, si32>`,
    ](src.value, a.value, b.value)


fn vpdpbusd(
    src: SIMD[8, DType.si32],
    a: SIMD[8, DType.si32],
    b: SIMD[8, DType.si32],
) -> SIMD[8, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpbusd.256",
        _type : __mlir_type.`!pop.simd<8, si32>`,
    ](src.value, a.value, b.value)


fn vpdpbusd(
    src: SIMD[4, DType.si32],
    a: SIMD[4, DType.si32],
    b: SIMD[4, DType.si32],
) -> SIMD[4, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpbusd.128",
        _type : __mlir_type.`!pop.simd<4, si32>`,
    ](src.value, a.value, b.value)


# ===----------------------------------------------------------------------===#
# vpdpbusds
# ===----------------------------------------------------------------------===#


fn vpdpbusds(
    src: SIMD[16, DType.si32],
    a: SIMD[16, DType.si32],
    b: SIMD[16, DType.si32],
) -> SIMD[16, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpbusds.512",
        _type : __mlir_type.`!pop.simd<16, si32>`,
    ](src.value, a.value, b.value)


fn vpdpbusds(
    src: SIMD[8, DType.si32],
    a: SIMD[8, DType.si32],
    b: SIMD[8, DType.si32],
) -> SIMD[8, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpbusds.256",
        _type : __mlir_type.`!pop.simd<8, si32>`,
    ](src.value, a.value, b.value)


fn vpdpbusds(
    src: SIMD[4, DType.si32],
    a: SIMD[4, DType.si32],
    b: SIMD[4, DType.si32],
) -> SIMD[4, DType.si32]:
    return __mlir_op.`pop.call_llvm_intrinsic`[
        intrin:"llvm.x86.avx512.vpdpbusds.128",
        _type : __mlir_type.`!pop.simd<4, si32>`,
    ](src.value, a.value, b.value)
