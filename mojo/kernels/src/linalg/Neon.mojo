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


fn _neon_dotprod[
    a_type: DType, b_type: DType, c_type: DType, simd_size: Int
](r: SIMD[c_type, simd_size], a: SIMD[a_type, 16], b: SIMD[b_type, 16]) -> SIMD[
    c_type, simd_size
]:
    @parameter
    if a_type == DType.uint8 and b_type == DType.uint8:
        return rebind[SIMD[c_type, simd_size]](
            _neon_dotprod(
                rebind[SIMD[DType.int32, 4]](r),
                rebind[SIMD[DType.uint8, 16]](a),
                rebind[SIMD[DType.uint8, 16]](b),
            )
        )
    else:
        return rebind[SIMD[c_type, simd_size]](
            _neon_dotprod(
                rebind[SIMD[DType.int32, 4]](r),
                rebind[SIMD[DType.int8, 16]](a),
                rebind[SIMD[DType.int8, 16]](b),
            )
        )


fn _neon_dotprod_lane[
    lane: Int
](
    r: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]
) -> SIMD[DType.int32, 4]:
    # Helper to generate `sdot v0, v1, v2[lane]` instruction form.
    var tuple = bitcast[DType.int32, 4](b)[lane]
    var splat = bitcast[DType.int8, 16](SIMD[DType.int32, 4](tuple))
    return _neon_dotprod(r, a, splat)


fn _neon_dotprod_lane[
    lane: Int
](
    r: SIMD[DType.int32, 4], a: SIMD[DType.uint8, 16], b: SIMD[DType.uint8, 16]
) -> SIMD[DType.int32, 4]:
    # Helper to generate `udot v0, v1, v2[lane]` instruction form.
    var tuple = bitcast[DType.int32, 4](b)[lane]
    var splat = bitcast[DType.uint8, 16](SIMD[DType.int32, 4](tuple))
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


fn _neon_matmul[
    a_type: DType, b_type: DType, c_type: DType, simd_size: Int
](r: SIMD[c_type, simd_size], a: SIMD[a_type, 16], b: SIMD[b_type, 16]) -> SIMD[
    c_type, simd_size
]:
    @parameter
    if a_type == DType.uint8 and b_type == DType.uint8:
        return rebind[SIMD[c_type, simd_size]](
            _neon_matmul(
                rebind[SIMD[DType.int32, 4]](r),
                rebind[SIMD[DType.uint8, 16]](a),
                rebind[SIMD[DType.uint8, 16]](b),
            )
        )
    elif a_type == DType.uint8 and b_type == DType.int8:
        return rebind[SIMD[c_type, simd_size]](
            _neon_matmul(
                rebind[SIMD[DType.int32, 4]](r),
                rebind[SIMD[DType.uint8, 16]](a),
                rebind[SIMD[DType.int8, 16]](b),
            )
        )
    else:
        return rebind[SIMD[c_type, simd_size]](
            _neon_matmul(
                rebind[SIMD[DType.int32, 4]](r),
                rebind[SIMD[DType.int8, 16]](a),
                rebind[SIMD[DType.int8, 16]](b),
            )
        )
    # FIXME should throw an error if s8u8
