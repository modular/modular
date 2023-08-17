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

from Intrinsics import llvm_intrinsic
from TypeUtilities import rebind
from Assert import assert_param
from sys.info import has_avx512_vnni

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


fn _dot_i8_to_i32_16(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int8, 64],
    b: SIMD[DType.int8, 64],
) -> SIMD[DType.int32, 16]:
    let t1 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", __mlir_type.`!pop.simd<32, si16>`
    ](a.value, b.value)
    let one16: SIMD[DType.int16, 32] = 1
    let t2 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddw.d.512", __mlir_type.`!pop.simd<16, si32>`
    ](t1, one16)
    return t2 + src


fn _dot_i8_to_i32_8(
    src: SIMD[DType.int32, 8],
    a: SIMD[DType.int8, 32],
    b: SIMD[DType.int8, 32],
) -> SIMD[DType.int32, 8]:
    let t1 = llvm_intrinsic[
        "llvm.x86.avx2.pmadd.ub.sw", __mlir_type.`!pop.simd<16, si16>`
    ](a.value, b.value)
    let one16: SIMD[DType.int16, 16] = 1
    let t2 = llvm_intrinsic[
        "llvm.x86.avx2.pmadd.wd", __mlir_type.`!pop.simd<8, si32>`
    ](t1, one16)
    return t2 + src


fn _dot_i8_to_i32_4(
    src: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    let t1 = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", __mlir_type.`!pop.simd<8, si16>`
    ](a.value, b.value)
    let one16: SIMD[DType.int16, 8] = 1
    let t2 = llvm_intrinsic[
        "llvm.x86.sse2.pmadd.wd", __mlir_type.`!pop.simd<4, si32>`
    ](t1, one16)
    return t2 + src


fn bitcast[
    dest_size: Int, dest_type: DType, src_size: Int, src_type: DType
](v: SIMD[src_type, src_size]) -> SIMD[dest_type, dest_size]:
    return __mlir_op.`pop.bitcast`[
        _type : __mlir_type[
            `!pop.simd<`, dest_size.value, `, `, dest_type.value, `>`
        ]
    ](v.value)


fn dot_i8_to_i32_AVX2[
    width: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """The dot product of the four bytes in each int32 element of a and b plus a int32 from src.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A uint8 SIMD vector.
        b: A int8 SIMD vector.

    Constraints:
        Requires AVX2.
        The size of the output vector must be 4, 8 or 16.
        The a argument has range [0,127] not [0, 255].
        The b argument has range [-128,127].

    Returns:
        A SIMD vector of width elements.
    """

    @parameter
    if width == 16:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_16(
                rebind[SIMD[DType.int32, 16]](src),
                bitcast[64, DType.int8](rebind[SIMD[DType.int32, 16]](a)),
                bitcast[64, DType.int8](rebind[SIMD[DType.int32, 16]](b)),
            )
        )
    elif width == 8:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_8(
                rebind[SIMD[DType.int32, 8]](src),
                bitcast[32, DType.int8](rebind[SIMD[DType.int32, 8]](a)),
                bitcast[32, DType.int8](rebind[SIMD[DType.int32, 8]](b)),
            )
        )
    else:
        assert_param[width == 4]()
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_4(
                rebind[SIMD[DType.int32, 4]](src),
                bitcast[16, DType.int8](rebind[SIMD[DType.int32, 4]](a)),
                bitcast[16, DType.int8](rebind[SIMD[DType.int32, 4]](a)),
            )
        )


fn dot_i8_to_i32_x86[
    width: Int,
    a_type: DType,
    b_type: DType,
    c_type: DType,
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """The dot product of the four bytes in each int32 element of a and b plus a int32 from src using VNNI or AVX2.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A uint8 SIMD vector.
        b: A int8 SIMD vector.

    Constraints:
        Requires AVX512_VNNI or AVX2.
        The size of the output vector must be 4, 8 or 16.
        The a argument has range [0,127] not [0, 255].
        The b argument has range [-128,127].

    Returns:
      A SIMD vector of width elements.
    """

    @parameter
    if has_avx512_vnni():
        return vpdpbusd[width](src, a, b)
    else:
        return dot_i8_to_i32_AVX2[width](src, a, b)
