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
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import has_avx512f

from buffer import Buffer
from memory import memcmp
from LinAlg.VNNI import (
    dot_i8_to_i32_AVX2,
    dot_i8_to_i32_saturated_AVX2,
    dot_i8_to_i32_saturated_x86,
    dot_i8_to_i32_x86,
)


fn main():
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

    var av16u = a.data.offset(128 + 64).bitcast[DType.int32]().load[width=16]()
    var av16s = asat.data.offset(128 + 64).bitcast[DType.int32]().load[
        width=16
    ]()
    var bv16 = b.data.offset(0).bitcast[DType.int32]().load[width=16]()
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

    # CHECK: [-97906, -96769, -95504, -94111, -92590, -90941, -89164, -87259, -85226, -83065, -80776, -78359, -75814, -73141, -70340, -67411]
    print(cv16u)
    # CHECK: [-33138, -34049, -34832, -35487, -36014, -36413, -36684, -36827, -36842, -36729, -36488, -36119, -35622, -34997, -34244, -33363]
    print(cv16s)

    var av8u = a.data.offset(128 + 64).bitcast[DType.int32]().load[width=8]()
    var av8s = asat.data.offset(128 + 64).bitcast[DType.int32]().load[width=8]()
    var bv8 = b.data.offset(0).bitcast[DType.int32]().load[width=8]()
    var cv8u = dot_i8_to_i32_AVX2[8](c.data.load[width=8](), av8u, bv8)
    var cv8s = dot_i8_to_i32_saturated_AVX2[8](
        c.data.load[width=8](), av8s, bv8
    )

    # CHECK: [-97906, -96769, -95504, -94111, -92590, -90941, -89164, -87259]
    print(cv8u)
    # CHECK: [-33138, -34049, -34832, -35487, -36014, -36413, -36684, -36827]
    print(cv8s)

    var av4u = a.data.offset(128 + 64).bitcast[DType.int32]().load[width=4]()
    var av4s = asat.data.offset(128 + 64).bitcast[DType.int32]().load[width=4]()
    var bv4 = b.data.offset(0).bitcast[DType.int32]().load[width=4]()
    var cv4u = dot_i8_to_i32_AVX2[4](c.data.load[width=4](), av4u, bv4)
    var cv4s = dot_i8_to_i32_saturated_AVX2[4](
        c.data.load[width=4](), av4s, bv4
    )

    # CHECK: [-97906, -96769, -95504, -94111]
    print(cv4u)
    # CHECK: [-33138, -34049, -34832, -35487]
    print(cv4s)
