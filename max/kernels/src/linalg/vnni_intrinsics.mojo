# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
#
# This file contains wrappers around Intel VNNI intrinsics. See
# https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#expand=2206&avx512techs=AVX512_VNNI
#
# ===-----------------------------------------------------------------------===#

from sys import llvm_intrinsic
from sys.info import has_vnni

from memory.unsafe import bitcast

# ===-----------------------------------------------------------------------===#
# vpdpwssd
# ===-----------------------------------------------------------------------===#


fn vpdpwssd[
    width: Int, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    constrained[c_type is DType.int32, "the type of C must be int32"]()

    @parameter
    if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssd.512", SIMD[c_type, width]
        ](src, a, b)
    elif width == 8:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssd.256", SIMD[c_type, width]
        ](src, a, b)
    else:
        constrained[width == 4]()
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssd.128", SIMD[c_type, width]
        ](src, a, b)


# ===-----------------------------------------------------------------------===#
# vpdpwssds
# ===-----------------------------------------------------------------------===#


fn vpdpwssds[
    width: Int, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    constrained[c_type is DType.int32, "the type of C must be int32"]()

    @parameter
    if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssds.512", SIMD[c_type, width]
        ](src, a, b)
    elif width == 8:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssds.256", SIMD[c_type, width]
        ](src, a, b)
    else:
        constrained[width == 4]()
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssds.128", SIMD[c_type, width]
        ](src, a, b)


# ===-----------------------------------------------------------------------===#
# vpdpbusd
# ===-----------------------------------------------------------------------===#


fn vpdpbusd[
    width: Int, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    constrained[c_type is DType.int32, "the type of C must be int32"]()

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


# ===-----------------------------------------------------------------------===#
# vpdpbusds
# ===-----------------------------------------------------------------------===#


fn vpdpbusds[
    width: Int, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    constrained[c_type is DType.int32, "the type of C must be int32"]()

    @parameter
    if width == 16:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusds.512", SIMD[c_type, width]
        ](src, a, b)
    elif width == 8:
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusds.256", SIMD[c_type, width]
        ](src, a, b)
    else:
        constrained[width == 4]()
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusds.128", SIMD[c_type, width]
        ](src, a, b)


fn _dot_i8_to_i32_16(
    src: SIMD[DType.int32, 16], a: SIMD[DType.int8, 64], b: SIMD[DType.int8, 64]
) -> SIMD[DType.int32, 16]:
    var mask_hi = bitcast[DType.int8, 64](SIMD[DType.int16, 32](0x0100))
    var mask_lo = bitcast[DType.int8, 64](SIMD[DType.int16, 32](0x0001))
    var ah = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](a, mask_hi)
    var bh = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](mask_hi, b)
    var al = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](a, mask_lo)
    var bl = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](mask_lo, b)
    var t1 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddw.d.512", SIMD[DType.int32, 16]
    ](al, bl)
    var t2 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddw.d.512", SIMD[DType.int32, 16]
    ](ah, bh)
    return src + t1 + t2


fn _dot_i8_to_i32_8(
    src: SIMD[DType.int32, 8], a: SIMD[DType.int8, 32], b: SIMD[DType.int8, 32]
) -> SIMD[DType.int32, 8]:
    var mask_hi = bitcast[DType.int8, 32](SIMD[DType.int16, 16](0x0100))
    var mask_lo = bitcast[DType.int8, 32](SIMD[DType.int16, 16](0x0001))

    var ah = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        a, mask_hi
    )
    var bh = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        mask_hi, b
    )
    var al = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        a, mask_lo
    )
    var bl = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        mask_lo, b
    )
    var t1 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](
        al, bl
    )
    var t2 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](
        ah, bh
    )
    return src + t1 + t2


fn _dot_i8_to_i32_4(
    src: SIMD[DType.int32, 4], a: SIMD[DType.int8, 16], b: SIMD[DType.int8, 16]
) -> SIMD[DType.int32, 4]:
    var mask_hi = bitcast[DType.int8, 16](SIMD[DType.int16, 8](0x0100))
    var mask_lo = bitcast[DType.int8, 16](SIMD[DType.int16, 8](0x0001))

    var ah = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](a, mask_hi)
    var bh = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](mask_hi, b)
    var al = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](a, mask_lo)
    var bl = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](mask_lo, b)
    var t1 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, 4]](
        al, bl
    )
    var t2 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, 4]](
        ah, bh
    )
    return src + t1 + t2


fn pmaddubs[
    width: Int
](a: SIMD[DType.int32, width], b: SIMD[DType.int32, width]) -> SIMD[
    DType.int32, width
]:
    @parameter
    if width == 16:
        return rebind[SIMD[DType.int32, width]](
            bitcast[DType.int32, 16](
                llvm_intrinsic[
                    "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
                ](
                    bitcast[DType.int8, 64](a),
                    bitcast[DType.int8, 64](b),
                )
            )
        )
    elif width == 8:
        return rebind[SIMD[DType.int32, width]](
            bitcast[DType.int32, 8](
                llvm_intrinsic[
                    "llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]
                ](
                    bitcast[DType.int8, 32](a),
                    bitcast[DType.int8, 32](b),
                )
            )
        )
    else:
        constrained[width == 4]()
        return rebind[SIMD[DType.int32, width]](
            bitcast[DType.int32, 4](
                llvm_intrinsic[
                    "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
                ](
                    bitcast[DType.int8, 16](a),
                    bitcast[DType.int8, 16](b),
                )
            )
        )


fn pmaddw[
    width: Int
](a: SIMD[DType.int32, width], b: SIMD[DType.int32, width]) -> SIMD[
    DType.int32, width
]:
    @parameter
    if width == 16:
        return rebind[SIMD[DType.int32, width]](
            bitcast[DType.int32, 16](
                llvm_intrinsic[
                    "llvm.x86.avx512.pmaddw.d.512", SIMD[DType.int32, width]
                ](
                    bitcast[DType.int16, 32](a),
                    bitcast[DType.int16, 32](b),
                )
            )
        )
    elif width == 8:
        return rebind[SIMD[DType.int32, width]](
            bitcast[DType.int32, 8](
                llvm_intrinsic[
                    "llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, width]
                ](
                    bitcast[DType.int16, 16](a),
                    bitcast[DType.int16, 16](b),
                )
            )
        )
    else:
        constrained[width == 4]()
        return rebind[SIMD[DType.int32, width]](
            bitcast[DType.int32, 16](
                llvm_intrinsic[
                    "llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, width]
                ](
                    bitcast[DType.int16, 8](a),
                    bitcast[DType.int16, 8](b),
                )
            )
        )


fn _dot_i8_to_i32_saturated_16(
    src: SIMD[DType.int32, 16], a: SIMD[DType.int8, 64], b: SIMD[DType.int8, 64]
) -> SIMD[DType.int32, 16]:
    var t1 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddubs.w.512", SIMD[DType.int16, 32]
    ](a, b)
    var t2 = llvm_intrinsic[
        "llvm.x86.avx512.pmaddw.d.512", SIMD[DType.int32, 16]
    ](t1, SIMD[DType.int16, 32](1))
    return t2 + src


fn _dot_i8_to_i32_saturated_8(
    src: SIMD[DType.int32, 8], a: SIMD[DType.int8, 32], b: SIMD[DType.int8, 32]
) -> SIMD[DType.int32, 8]:
    var t1 = llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        a, b
    )
    var t2 = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]](
        t1, SIMD[DType.int16, 16](1)
    )
    return t2 + src


fn _dot_i8_to_i32_saturated_4(
    src: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    var t1 = llvm_intrinsic[
        "llvm.x86.ssse3.pmadd.ub.sw.128", SIMD[DType.int16, 8]
    ](a, b)
    var t2 = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[DType.int32, 4]](
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

    @parameter
    if has_vnni():
        return vpdpbusd(src, a, b)
    else:
        return dot_i8_to_i32_AVX2(src, a, b)


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
    if has_vnni():
        return vpdpbusd(src, a, b)
    else:
        return dot_i8_to_i32_saturated_AVX2(src, a, b)


fn dot_i16_to_i32_AVX2[
    width: Int, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """The dot product of the two words in each int32 element of a and b plus a int32 from src.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A int16 SIMD vector.
        b: A int16 SIMD vector.

    Constraints:
        Requires AVX2.
        The size of the output vector must be 4, 8 or 16.

    Returns:
        A SIMD vector of width elements.
    """

    var t: SIMD[c_type, width]

    @parameter
    if width == 16:
        t = llvm_intrinsic["llvm.x86.avx512.pmaddw.d.512", SIMD[c_type, width]](
            bitcast[DType.int16, 32](rebind[SIMD[DType.int32, 16]](a)),
            bitcast[DType.int16, 32](rebind[SIMD[DType.int32, 16]](b)),
        )
    elif width == 8:
        t = llvm_intrinsic["llvm.x86.avx2.pmadd.wd", SIMD[c_type, width]](
            bitcast[DType.int16, 16](rebind[SIMD[DType.int32, 8]](a)),
            bitcast[DType.int16, 16](rebind[SIMD[DType.int32, 8]](b)),
        )
    else:
        constrained[width == 4]()
        t = llvm_intrinsic["llvm.x86.sse2.pmadd.wd", SIMD[c_type, width]](
            bitcast[DType.int16, 8](rebind[SIMD[DType.int32, 4]](a)),
            bitcast[DType.int16, 8](rebind[SIMD[DType.int32, 4]](b)),
        )

    return src + t


fn dot_i16_to_i32_x86[
    width: Int, a_type: DType, b_type: DType, c_type: DType
](
    src: SIMD[c_type, width], a: SIMD[a_type, width], b: SIMD[b_type, width]
) -> SIMD[c_type, width]:
    """The dot product of the two words in each int32 element of a and b plus a int32 from src using VNNI or AVX2.

    Parameters:
        width: Size of the output SIMD vector.
        a_type: The DType for a.
        b_type: The DType for b.
        c_type: The DType for c.

    Args:
        src: A int32 SIMD vector.
        a: A int16 SIMD vector.
        b: A int16 SIMD vector.

    Constraints:
        Requires AVX512_VNNI or AVX2.
        The size of the output vector must be 4, 8 or 16.

    Returns:
      A SIMD vector of width elements.
    """

    @parameter
    if has_vnni():
        return vpdpwssd(src, a, b)
    else:
        return dot_i16_to_i32_AVX2(src, a, b)
