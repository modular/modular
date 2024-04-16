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


fn _neon_dotprod[
    a_type: DType, b_type: DType, c_type: DType, width: Int
](
    r: SIMD[c_type, width],
    a: SIMD[a_type, width * 4],
    b: SIMD[b_type, width * 4],
) -> SIMD[c_type, width]:
    constrained[c_type.is_int32(), "the type of C must be int32"]()
    constrained[width == 4]()

    @parameter
    @always_inline
    fn dotprod_intrinsic[intrin: StringLiteral]() -> SIMD[c_type, width]:
        return rebind[SIMD[c_type, width]](
            llvm_intrinsic[intrin, SIMD[DType.int32, 4]](
                rebind[SIMD[c_type, 4]](r),
                rebind[SIMD[a_type, 16]](a),
                rebind[SIMD[b_type, 16]](b),
            )
        )

    @parameter
    if a_type == DType.uint8 and b_type == DType.uint8:
        return dotprod_intrinsic["llvm.aarch64.neon.udot.v4i32.v16i8"]()
    elif a_type == DType.int8 and b_type == DType.int8:
        return dotprod_intrinsic["llvm.aarch64.neon.sdot.v4i32.v16i8"]()
    else:
        constrained[False, "unsupported A and B types"]()
        return SIMD[c_type, width]()


fn _neon_dotprod_lane[
    lane: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    width: Int,
    b_width: Int,
](
    r: SIMD[c_type, width],
    a: SIMD[a_type, width * 4],
    b: SIMD[b_type, b_width],
) -> SIMD[c_type, width]:
    constrained[b_type.is_int8() or b_type.is_uint8(), "unsupported B type"]()
    constrained[4 <= b_width <= 16, "unsupported B width"]()
    constrained[0 <= lane < (b_width // 4), "invalid lane index"]()

    # Helper to generate `sdot r, a, b[lane]` instruction form.
    var tuple = bitcast[DType.int32, b_width // 4](b)[lane]
    var splat = bitcast[b_type, width * 4](SIMD[DType.int32, width](tuple))
    return _neon_dotprod(r, a, splat)


# ===----------------------------------------------------------------------===#
# matrix multiply-accumulate
# ===----------------------------------------------------------------------===#


fn _neon_matmul[
    a_type: DType, b_type: DType, c_type: DType, width: Int
](
    r: SIMD[c_type, width],
    a: SIMD[a_type, width * 4],
    b: SIMD[b_type, width * 4],
) -> SIMD[c_type, width]:
    constrained[c_type.is_int32(), "the type of C must be int32"]()
    constrained[width == 4]()

    @parameter
    @always_inline
    fn matmul_intrinsic[intrin: StringLiteral]() -> SIMD[c_type, width]:
        return rebind[SIMD[c_type, width]](
            llvm_intrinsic[intrin, SIMD[DType.int32, 4]](
                rebind[SIMD[c_type, 4]](r),
                rebind[SIMD[a_type, 16]](a),
                rebind[SIMD[b_type, 16]](b),
            )
        )

    @parameter
    if a_type == DType.uint8 and b_type == DType.uint8:
        return matmul_intrinsic["llvm.aarch64.neon.ummla.v4i32.v16i8"]()
    elif a_type == DType.uint8 and b_type == DType.int8:
        return matmul_intrinsic["llvm.aarch64.neon.usmmla.v4i32.v16i8"]()
    elif a_type == DType.int8 and b_type == DType.int8:
        return matmul_intrinsic["llvm.aarch64.neon.smmla.v4i32.v16i8"]()
    else:
        constrained[False, "unsupported A and B types"]()
        return SIMD[c_type, width]()
