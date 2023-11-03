# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import llvm_intrinsic
from memory.unsafe import bitcast


# ===----------------------------------------------------------------------===#
# dot product
# ===----------------------------------------------------------------------===#


fn _neon_dotprod(
    r: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.sdot.v4i32.v16i8", SIMD[DType.int32, 4]
    ](r, a, b)


fn _neon_dotprod(
    r: SIMD[DType.int32, 4], a: SIMD[DType.uint8, 16], b: SIMD[DType.uint8, 16]
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.udot.v4i32.v16i8", SIMD[DType.int32, 4]
    ](r, a, b)


fn _neon_dotprod_lane[
    lane: Int
](
    r: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]
) -> SIMD[DType.int32, 4]:
    # Helper to generate `sdot v0, v1, v2[lane]` instruction form.
    let tuple = bitcast[DType.int32, 4](b)[lane]
    let splat = bitcast[DType.int8, 16](SIMD[DType.int32, 4](tuple))
    return _neon_dotprod(r, a, splat)


fn _neon_dotprod_lane[
    lane: Int
](
    r: SIMD[DType.int32, 4], a: SIMD[DType.uint8, 16], b: SIMD[DType.uint8, 16]
) -> SIMD[DType.int32, 4]:
    # Helper to generate `udot v0, v1, v2[lane]` instruction form.
    let tuple = bitcast[DType.int32, 4](b)[lane]
    let splat = bitcast[DType.uint8, 16](SIMD[DType.int32, 4](tuple))
    return _neon_dotprod(r, a, splat)


# ===----------------------------------------------------------------------===#
# matrix multiply-accumulate
# ===----------------------------------------------------------------------===#


fn _neon_matmul(
    r: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.smmla.v4i32.v16i8", SIMD[DType.int32, 4]
    ](r, a, b)


fn _neon_matmul(
    r: SIMD[DType.int32, 4], a: SIMD[DType.uint8, 16], b: SIMD[DType.uint8, 16]
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.ummla.v4i32.v16i8", SIMD[DType.int32, 4]
    ](r, a, b)


fn _neon_matmul(
    r: SIMD[DType.int32, 4], a: SIMD[DType.uint8, 16], b: SIMD[DType.int8, 16]
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.usmmla.v4i32.v16i8", SIMD[DType.int32, 4]
    ](r, a, b)
