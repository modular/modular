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
# RUN: %mojo %s

from sys.info import has_avx512f

from buffer import Buffer
from LinAlg.vnni_intrinsics import (
    dot_i8_to_i32_AVX2,
    dot_i8_to_i32_saturated_AVX2,
    dot_i8_to_i32_saturated_x86,
    dot_i8_to_i32_x86,
    dot_i16_to_i32_AVX2,
    dot_i16_to_i32_x86,
)
from memory import memcmp
from testing import assert_equal


def test_i8_to_i32():
    var a = Buffer[DType.uint8, 16 * 64].aligned_stack_allocation[64]()
    var asat = Buffer[DType.uint8, 16 * 64].aligned_stack_allocation[64]()
    var b = Buffer[DType.int8, 64 * 16].aligned_stack_allocation[64]()

    var c = Buffer[DType.int32, 16 * 16].aligned_stack_allocation[64]()
    var csat = Buffer[DType.int32, 16 * 16].aligned_stack_allocation[64]()

    for i in range(16 * 64):
        a[i] = i & 255
        asat[i] = i & 127
        b[i] = (i & 255) - 128

    for i in range(16 * 16):
        c[i] = i
        csat[i] = c[i]

    var av16u = SIMD[size=16].load(
        a.data.offset(128 + 64).bitcast[DType.int32]()
    )
    var av16s = SIMD[size=16].load(
        asat.data.offset(128 + 64).bitcast[DType.int32]()
    )
    var bv16 = SIMD[size=16].load(b.data.offset(0).bitcast[DType.int32]())
    var cv16u: SIMD[DType.int32, 16] = 0
    var cv16s: SIMD[DType.int32, 16] = 0
    if has_avx512f():
        cv16u = dot_i8_to_i32_AVX2[16](SIMD[size=16].load(c.data), av16u, bv16)
        cv16s = dot_i8_to_i32_saturated_AVX2[16](
            SIMD[size=16].load(c.data), av16s, bv16
        )
    else:
        # split the vectors into high and low
        var cv8ul = dot_i8_to_i32_AVX2[8](
            SIMD[size=8].load(c.data), av16u.slice[8](), bv16.slice[8]()
        )
        var cv8sl = dot_i8_to_i32_saturated_AVX2[8](
            SIMD[size=8].load(c.data), av16s.slice[8](), bv16.slice[8]()
        )
        var cv8uh = dot_i8_to_i32_AVX2[8](
            SIMD[size=8].load(c.data.offset(8)),
            av16u.slice[8, offset=8](),
            bv16.slice[8, offset=8](),
        )
        var cv8sh = dot_i8_to_i32_saturated_AVX2[8](
            SIMD[size=8].load(c.data.offset(8)),
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

    var av8u = SIMD[size=8].load(a.data.offset(128 + 64).bitcast[DType.int32]())
    var av8s = SIMD[size=8].load(
        asat.data.offset(128 + 64).bitcast[DType.int32]()
    )
    var bv8 = SIMD[size=8].load(b.data.offset(0).bitcast[DType.int32]())
    var cv8u = dot_i8_to_i32_AVX2[8](SIMD[size=8].load(c.data), av8u, bv8)
    var cv8s = dot_i8_to_i32_saturated_AVX2[8](
        SIMD[size=8].load(c.data), av8s, bv8
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

    var av4u = SIMD[size=4].load(a.data.offset(128 + 64).bitcast[DType.int32]())
    var av4s = SIMD[size=4].load(
        asat.data.offset(128 + 64).bitcast[DType.int32]()
    )
    var bv4 = SIMD[size=4].load(b.data.offset(0).bitcast[DType.int32]())
    var cv4u = dot_i8_to_i32_AVX2[4](SIMD[size=4].load(c.data), av4u, bv4)
    var cv4s = dot_i8_to_i32_saturated_AVX2[4](
        SIMD[size=4].load(c.data), av4s, bv4
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
