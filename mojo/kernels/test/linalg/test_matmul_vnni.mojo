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

from memory import memcmp
from memory.buffer import Buffer
from VNNI import (
    dot_i8_to_i32_AVX2,
    dot_i8_to_i32_saturated_AVX2,
    dot_i8_to_i32_saturated_x86,
    dot_i8_to_i32_x86,
)


fn main():
    let a = Buffer[16 * 64, DType.uint8].aligned_stack_allocation[64]()
    let asat = Buffer[16 * 64, DType.uint8].aligned_stack_allocation[64]()
    let b = Buffer[64 * 16, DType.int8].aligned_stack_allocation[64]()

    let c = Buffer[16 * 16, DType.int32].aligned_stack_allocation[64]()
    let csat = Buffer[16 * 16, DType.int32].aligned_stack_allocation[64]()

    for i in range(16 * 64):
        a[i] = i & 255
        asat[i] = i & 127
        b[i] = (i & 255) - 128

    for i in range(16 * 16):
        c[i] = i
        csat[i] = c[i]

    let av16u = a.data.offset(128 + 64).bitcast[DType.int32]().simd_load[16]()
    let av16s = asat.data.offset(128 + 64).bitcast[DType.int32]().simd_load[
        16
    ]()
    let bv16 = b.data.offset(0).bitcast[DType.int32]().simd_load[16]()
    var cv16u: SIMD[DType.int32, 16] = 0
    var cv16s: SIMD[DType.int32, 16] = 0
    if has_avx512f():
        cv16u = dot_i8_to_i32_AVX2[16](c.data.simd_load[16](), av16u, bv16)
        cv16s = dot_i8_to_i32_saturated_AVX2[16](
            c.data.simd_load[16](), av16s, bv16
        )
    else:
        # split the vectors into high and low
        let cv8ul = dot_i8_to_i32_AVX2[8](
            c.data.simd_load[8](), av16u.slice[8](0), bv16.slice[8](0)
        )
        let cv8sl = dot_i8_to_i32_saturated_AVX2[8](
            c.data.simd_load[8](), av16s.slice[8](0), bv16.slice[8](0)
        )
        let cv8uh = dot_i8_to_i32_AVX2[8](
            c.data.offset(8).simd_load[8](), av16u.slice[8](8), bv16.slice[8](8)
        )
        let cv8sh = dot_i8_to_i32_saturated_AVX2[8](
            c.data.offset(8).simd_load[8](), av16s.slice[8](8), bv16.slice[8](8)
        )
        let cbufs = Buffer[16, DType.int32].aligned_stack_allocation[64]()
        let cbufu = Buffer[16, DType.int32].aligned_stack_allocation[64]()
        cbufs.simd_store[8](0, cv8sl)
        cbufs.simd_store[8](8, cv8sh)
        cbufu.simd_store[8](0, cv8ul)
        cbufu.simd_store[8](8, cv8uh)
        cv16u = cbufu.simd_load[16](0)
        cv16s = cbufs.simd_load[16](0)

    # CHECK: [-97906, -96769, -95504, -94111, -92590, -90941, -89164, -87259, -85226, -83065, -80776, -78359, -75814, -73141, -70340, -67411]
    print(cv16u)
    # CHECK: [-33138, -34049, -34832, -35487, -36014, -36413, -36684, -36827, -36842, -36729, -36488, -36119, -35622, -34997, -34244, -33363]
    print(cv16s)

    let av8u = a.data.offset(128 + 64).bitcast[DType.int32]().simd_load[8]()
    let av8s = asat.data.offset(128 + 64).bitcast[DType.int32]().simd_load[8]()
    let bv8 = b.data.offset(0).bitcast[DType.int32]().simd_load[8]()
    let cv8u = dot_i8_to_i32_AVX2[8](c.data.simd_load[8](), av8u, bv8)
    let cv8s = dot_i8_to_i32_saturated_AVX2[8](c.data.simd_load[8](), av8s, bv8)

    # CHECK: [-97906, -96769, -95504, -94111, -92590, -90941, -89164, -87259]
    print(cv8u)
    # CHECK: [-33138, -34049, -34832, -35487, -36014, -36413, -36684, -36827]
    print(cv8s)

    let av4u = a.data.offset(128 + 64).bitcast[DType.int32]().simd_load[4]()
    let av4s = asat.data.offset(128 + 64).bitcast[DType.int32]().simd_load[4]()
    let bv4 = b.data.offset(0).bitcast[DType.int32]().simd_load[4]()
    let cv4u = dot_i8_to_i32_AVX2[4](c.data.simd_load[4](), av4u, bv4)
    let cv4s = dot_i8_to_i32_saturated_AVX2[4](c.data.simd_load[4](), av4s, bv4)

    # CHECK: [-97906, -96769, -95504, -94111]
    print(cv4u)
    # CHECK: [-33138, -34049, -34832, -35487]
    print(cv4s)
