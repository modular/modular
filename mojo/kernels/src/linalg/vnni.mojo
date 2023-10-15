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

from debug import trap
from sys import llvm_intrinsic
from sys.info import has_avx512_vnni, has_avx2, has_avx512f, has_sse4, is_x86

from memory.unsafe import bitcast

# ===----------------------------------------------------------------------===#
# vpdpwssd
# ===----------------------------------------------------------------------===#


fn vpdpwssd(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int32, 16],
    b: SIMD[DType.int32, 16],
) -> SIMD[DType.int32, 16]:
    @parameter
    if not has_avx512f():
        trap()
        return 0
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssd.512", SIMD[DType.int32, 16]
    ](src, a, b)


fn vpdpwssd(
    src: SIMD[DType.int32, 8], a: SIMD[DType.int32, 8], b: SIMD[DType.int32, 8]
) -> SIMD[DType.int32, 8]:
    @parameter
    if not has_avx512f():
        trap()
        return 0
    return llvm_intrinsic["llvm.x86.avx512.vpdpwssd.256", SIMD[DType.int32, 8]](
        src, a, b
    )


fn vpdpwssd(
    src: SIMD[DType.int32, 4], a: SIMD[DType.int32, 4], b: SIMD[DType.int32, 4]
) -> SIMD[DType.int32, 4]:
    @parameter
    if not has_avx512f():
        return 0
    return llvm_intrinsic["llvm.x86.avx512.vpdpwssd.128", SIMD[DType.int32, 4]](
        src, a, b
    )


# ===----------------------------------------------------------------------===#
# vpdpwssds
# ===----------------------------------------------------------------------===#


fn vpdpwssds(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int32, 16],
    b: SIMD[DType.int32, 16],
) -> SIMD[DType.int32, 16]:
    @parameter
    if not has_avx512f():
        trap()  # Should never be called
        return 0
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.512", SIMD[DType.int32, 16]
    ](src, a, b)


fn vpdpwssds(
    src: SIMD[DType.int32, 8], a: SIMD[DType.int32, 8], b: SIMD[DType.int32, 8]
) -> SIMD[DType.int32, 8]:
    @parameter
    if not has_avx512f():
        trap()  # Should never be called
        return 0
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.256", SIMD[DType.int32, 8]
    ](src, a, b)


fn vpdpwssds(
    src: SIMD[DType.int32, 4], a: SIMD[DType.int32, 4], b: SIMD[DType.int32, 4]
) -> SIMD[DType.int32, 4]:
    @parameter
    if not has_avx512f():
        trap()  # Should never be called
        return 0
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpwssds.128", SIMD[DType.int32, 4]
    ](src, a, b)


# ===----------------------------------------------------------------------===#
# vpdpbusd
# ===----------------------------------------------------------------------===#


fn vpdpbusd[
    width: Int, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    @parameter
    if not has_avx512f():
        trap()  # Should never be called
        return 0
    constrained[c_type == DType.int32, "the type of C must be int32"]()

    @parameter
    if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusd.512", SIMD[c_type, width]
        ](src, a, b)
    elif width == 8:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusd.256", SIMD[c_type, width]
        ](src, a, b)
    else:
        constrained[width == 4]()
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusd.128", SIMD[c_type, width]
        ](src, a, b)


# ===----------------------------------------------------------------------===#
# vpdpbusds
# ===----------------------------------------------------------------------===#


fn vpdpbusds(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int32, 16],
    b: SIMD[DType.int32, 16],
) -> SIMD[DType.int32, 16]:
    @parameter
    if not has_avx512f():
        trap()  # Should never be called
        return 0
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.512", SIMD[DType.int32, 16]
    ](src, a, b)


fn vpdpbusds(
    src: SIMD[DType.int32, 8], a: SIMD[DType.int32, 8], b: SIMD[DType.int32, 8]
) -> SIMD[DType.int32, 8]:
    @parameter
    if not has_avx512f():
        trap()  # Should never be called
        return 0
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.256", SIMD[DType.int32, 8]
    ](src, a, b)


fn vpdpbusds(
    src: SIMD[DType.int32, 4], a: SIMD[DType.int32, 4], b: SIMD[DType.int32, 4]
) -> SIMD[DType.int32, 4]:
    @parameter
    if not has_avx512f():
        trap()  # Should never be called
        return 0
    return llvm_intrinsic[
        "llvm.x86.avx512.vpdpbusds.128", SIMD[DType.int32, 4]
    ](src, a, b)


fn _dot_i8_to_i32_16(
    src: SIMD[DType.int32, 16], a: SIMD[DType.int8, 64], b: SIMD[DType.int8, 64]
) -> SIMD[DType.int32, 16]:
    @parameter
    if not has_avx512f():
        trap()  # Should never be called
        return 0
    let mask_hi = bitcast[DType.int8, 64](SIMD[DType.int16, 32](0x0100))
    let mask_lo = bitcast[DType.int8, 64](SIMD[DType.int16, 32](0x0001))
    let ah = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](a, mask_hi)
    let bh = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](mask_hi, b)
    let al = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](a, mask_lo)
    let bl = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](mask_lo, b)
    let t1 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddw.d.512", SIMD[DType.int32, 16]
    ](al, bl)
    let t2 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddw.d.512", SIMD[DType.int32, 16]
    ](ah, bh)
    return src + t1 + t2


fn _dot_i8_to_i32_8(
    src: SIMD[DType.int32, 8], a: SIMD[DType.int8, 32], b: SIMD[DType.int8, 32]
) -> SIMD[DType.int32, 8]:
    @parameter
    if not has_avx2():
        trap()  # Should never be called
        return 0
    let mask_hi = bitcast[DType.int8, 32](SIMD[DType.int16, 16](0x0100))
    let mask_lo = bitcast[DType.int8, 32](SIMD[DType.int16, 16](0x0001))

    let ah = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        a, mask_hi
    )
    let bh = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        mask_hi, b
    )
    let al = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        a, mask_lo
    )
    let bl = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        mask_lo, b
    )
    let t1 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](
        al, bl
    )
    let t2 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](
        ah, bh
    )
    return src + t1 + t2


fn _dot_i8_to_i32_4(
    src: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]
) -> SIMD[DType.int32, 4]:
    @parameter
    if not has_sse4():
        trap()  # Should never be called
        return 0

    let mask_hi = bitcast[DType.int8, 16](SIMD[DType.int16, 8](0x0100))
    let mask_lo = bitcast[DType.int8, 16](SIMD[DType.int16, 8](0x0001))

    let ah = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](a, mask_hi)
    let bh = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](mask_hi, b)
    let al = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](a, mask_lo)
    let bl = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](mask_lo, b)
    let t1 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, 4]](
        al, bl
    )
    let t2 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, 4]](
        ah, bh
    )
    return src + t1 + t2


fn _dot_i8_to_i32_saturated_16(
    src: SIMD[DType.int32, 16], a: SIMD[DType.int8, 64], b: SIMD[DType.int8, 64]
) -> SIMD[DType.int32, 16]:
    @parameter
    if not has_avx512f():
        return 0

    let t1 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](a, b)
    let t2 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddw.d.512", SIMD[DType.int32, 16]
    ](t1, SIMD[DType.int16, 32](1))
    return t2 + src


fn _dot_i8_to_i32_saturated_8(
    src: SIMD[DType.int32, 8], a: SIMD[DType.int8, 32], b: SIMD[DType.int8, 32]
) -> SIMD[DType.int32, 8]:
    @parameter
    if not has_avx2():
        trap()  # Should never be called
        return 0
    let t1 = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        a, b
    )
    let t2 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](
        t1, SIMD[DType.int16, 16](1)
    )
    return t2 + src


fn _dot_i8_to_i32_saturated_4(
    src: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    @parameter
    if not has_sse4():
        trap()  # Should never be called
        return 0
    let t1 = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](a, b)
    let t2 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, 4]](
        t1, SIMD[DType.int16, 8](1)
    )
    return t2 + src


fn dot_i8_to_i32_AVX2[
    width: Int, a_type: DType, b_type: DType, c_type: DType
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
        The a argument has range [0,255].
        The b argument has range [-128,127].

    Returns:
        A SIMD vector of width elements.
    """

    @parameter
    if not is_x86():
        trap()  # Should never be called
        return 0

    @parameter
    if width == 16:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_16(
                rebind[SIMD[DType.int32, 16]](src),
                bitcast[DType.int8, 64](rebind[SIMD[DType.int32, 16]](a)),
                bitcast[DType.int8, 64](rebind[SIMD[DType.int32, 16]](b)),
            )
        )
    elif width == 8:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_8(
                rebind[SIMD[DType.int32, 8]](src),
                bitcast[DType.int8, 32](rebind[SIMD[DType.int32, 8]](a)),
                bitcast[DType.int8, 32](rebind[SIMD[DType.int32, 8]](b)),
            )
        )
    else:
        constrained[width == 4]()
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_4(
                rebind[SIMD[DType.int32, 4]](src),
                bitcast[DType.int8, 16](rebind[SIMD[DType.int32, 4]](a)),
                bitcast[DType.int8, 16](rebind[SIMD[DType.int32, 4]](b)),
            )
        )


fn dot_i8_to_i32_saturated_AVX2[
    width: Int, a_type: DType, b_type: DType, c_type: DType
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
    if not is_x86():
        trap()  # Should never be called
        return 0

    @parameter
    if width == 16:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_saturated_16(
                rebind[SIMD[DType.int32, 16]](src),
                bitcast[DType.int8, 64](rebind[SIMD[DType.int32, 16]](a)),
                bitcast[DType.int8, 64](rebind[SIMD[DType.int32, 16]](b)),
            )
        )
    elif width == 8:
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_saturated_8(
                rebind[SIMD[DType.int32, 8]](src),
                bitcast[DType.int8, 32](rebind[SIMD[DType.int32, 8]](a)),
                bitcast[DType.int8, 32](rebind[SIMD[DType.int32, 8]](b)),
            )
        )
    else:
        constrained[width == 4]()
        return rebind[SIMD[c_type, width]](
            _dot_i8_to_i32_saturated_4(
                rebind[SIMD[DType.int32, 4]](src),
                bitcast[DType.int8, 16](rebind[SIMD[DType.int32, 4]](a)),
                bitcast[DType.int8, 16](rebind[SIMD[DType.int32, 4]](b)),
            )
        )


fn dot_i8_to_i32_x86[
    width: Int, a_type: DType, b_type: DType, c_type: DType
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
        The a argument has range [0,255].
        The b argument has range [-128,127].

    Returns:
      A SIMD vector of width elements.
    """
    constrained[is_x86()]()

    @parameter
    if has_avx512_vnni():
        return vpdpbusd[width](src, a, b)
    else:
        return dot_i8_to_i32_AVX2[width](src, a, b)


# Saturation is much faster but limits input a to range [0, 127] instead of [0, 255]
fn dot_i8_to_i32_saturated_x86[
    width: Int, a_type: DType, b_type: DType, c_type: DType
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
    if not is_x86():
        return 0

    @parameter
    if has_avx512_vnni():
        return vpdpbusd[width](src, a, b)
    else:
        return dot_i8_to_i32_saturated_AVX2[width](src, a, b)
