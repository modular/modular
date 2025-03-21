# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file tests the vnni intrinsics
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: avx2
# RUN: %mojo-no-debug %s

from sys.info import has_avx512f

from buffer import NDBuffer
from linalg.vnni_intrinsics import (
    dot_i8_to_i32_AVX2,
    dot_i8_to_i32_saturated_AVX2,
    dot_i8_to_i32_saturated_x86,
    dot_i8_to_i32_x86,
    dot_i16_to_i32_AVX2,
    dot_i16_to_i32_x86,
)
from memory import bitcast
from testing import assert_equal


def test_i8_to_i32():
    var a = NDBuffer[
        DType.uint8, 1, MutableAnyOrigin, 16 * 64
    ].stack_allocation[alignment=64]()
    var asat = NDBuffer[
        DType.uint8, 1, MutableAnyOrigin, 16 * 64
    ].stack_allocation[alignment=64]()
    var b = NDBuffer[DType.int8, 1, MutableAnyOrigin, 64 * 16].stack_allocation[
        alignment=64
    ]()

    var c = NDBuffer[
        DType.int32, 1, MutableAnyOrigin, 16 * 16
    ].stack_allocation[alignment=64]()
    var csat = NDBuffer[
        DType.int32, 1, MutableAnyOrigin, 16 * 16
    ].stack_allocation[alignment=64]()

    for i in range(16 * 64):
        a[i] = i & 255
        asat[i] = i & 127
        b[i] = (i & 255) - 128

    for i in range(16 * 16):
        c[i] = i
        csat[i] = c[i]

    var av16u = a.data.offset(128 + 64).bitcast[Int32]().load[width=16]()
    var av16s = asat.data.offset(128 + 64).bitcast[Int32]().load[width=16]()
    var bv16 = b.data.offset(0).bitcast[Int32]().load[width=16]()
    var cv16u: SIMD[DType.int32, 16] = 0
    var cv16s: SIMD[DType.int32, 16] = 0
    if has_avx512f():
        cv16u = dot_i8_to_i32_AVX2[16](c.data.load[width=16](), av16u, bv16)
        cv16s = dot_i8_to_i32_saturated_AVX2[16](
            c.data.load[width=16](), av16s, bv16
        )
    else:
        # split the vectors into high and low
        var cv8ul = dot_i8_to_i32_AVX2[8](
            c.data.load[width=8](), av16u.slice[8](), bv16.slice[8]()
        )
        var cv8sl = dot_i8_to_i32_saturated_AVX2[8](
            c.data.load[width=8](), av16s.slice[8](), bv16.slice[8]()
        )
        var cv8uh = dot_i8_to_i32_AVX2[8](
            c.data.offset(8).load[width=8](),
            av16u.slice[8, offset=8](),
            bv16.slice[8, offset=8](),
        )
        var cv8sh = dot_i8_to_i32_saturated_AVX2[8](
            c.data.offset(8).load[width=8](),
            av16s.slice[8, offset=8](),
            bv16.slice[8, offset=8](),
        )
        cv16u = cv8ul.join(cv8uh)
        cv16s = cv8sl.join(cv8sh)

    assert_equal(
        cv16u,
        SIMD[DType.int32, 16](
            -97906,
            -96769,
            -95504,
            -94111,
            -92590,
            -90941,
            -89164,
            -87259,
            -85226,
            -83065,
            -80776,
            -78359,
            -75814,
            -73141,
            -70340,
            -67411,
        ),
    )
    assert_equal(
        cv16s,
        SIMD[DType.int32, 16](
            -33138,
            -34049,
            -34832,
            -35487,
            -36014,
            -36413,
            -36684,
            -36827,
            -36842,
            -36729,
            -36488,
            -36119,
            -35622,
            -34997,
            -34244,
            -33363,
        ),
    )

    var av8u = a.data.offset(128 + 64).bitcast[Int32]().load[width=8]()
    var av8s = asat.data.offset(128 + 64).bitcast[Int32]().load[width=8]()
    var bv8 = b.data.offset(0).bitcast[Int32]().load[width=8]()
    var cv8u = dot_i8_to_i32_AVX2[8](c.data.load[width=8](), av8u, bv8)
    var cv8s = dot_i8_to_i32_saturated_AVX2[8](
        c.data.load[width=8](), av8s, bv8
    )

    assert_equal(
        cv8u,
        SIMD[DType.int32, 8](
            -97906, -96769, -95504, -94111, -92590, -90941, -89164, -87259
        ),
    )
    assert_equal(
        cv8s,
        SIMD[DType.int32, 8](
            -33138, -34049, -34832, -35487, -36014, -36413, -36684, -36827
        ),
    )

    var av4u = a.data.offset(128 + 64).bitcast[Int32]().load[width=4]()
    var av4s = asat.data.offset(128 + 64).bitcast[Int32]().load[width=4]()
    var bv4 = b.data.offset(0).bitcast[Int32]().load[width=4]()
    var cv4u = dot_i8_to_i32_AVX2[4](c.data.load[width=4](), av4u, bv4)
    var cv4s = dot_i8_to_i32_saturated_AVX2[4](
        c.data.load[width=4](), av4s, bv4
    )

    assert_equal(cv4u, SIMD[DType.int32, 4](-97906, -96769, -95504, -94111))
    assert_equal(cv4s, SIMD[DType.int32, 4](-33138, -34049, -34832, -35487))


def test_i16_to_i32():
    def test_simd_width[width: Int]():
        var a = SIMD[DType.int16, width * 2]()
        var b = SIMD[DType.int16, width * 2]()
        var c_start = SIMD[DType.int32, width]()
        var c_golden = SIMD[DType.int32, width]()

        @parameter
        for i in range(width * 2):
            a[i] = i * 17 - 191
            b[i] = i * 19 + 155

        @parameter
        for i in range(width):
            c_start[i] = i * 233 - 322

        @parameter
        for i in range(width):
            c_golden[i] = c_start[i]

            @parameter
            for j in range(2):
                var a_val = a[i * 2 + j].cast[DType.int32]()
                var b_val = b[i * 2 + j].cast[DType.int32]()
                c_golden[i] += a_val * b_val

        var c_avx2 = dot_i16_to_i32_AVX2(
            c_start,
            bitcast[DType.int32, width](a),
            bitcast[DType.int32, width](b),
        )
        assert_equal(c_golden, c_avx2)

        var c_x86 = dot_i16_to_i32_x86(
            c_start,
            bitcast[DType.int32, width](a),
            bitcast[DType.int32, width](b),
        )
        assert_equal(c_golden, c_x86)

    @parameter
    if has_avx512f():
        test_simd_width[16]()

    test_simd_width[8]()
    test_simd_width[4]()


def main():
    test_i8_to_i32()
    test_i16_to_i32()
