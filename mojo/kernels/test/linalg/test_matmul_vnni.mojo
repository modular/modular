# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file multiples a 16x64 matrix int8 matrix by a 64x16 int 8 matrix
# and outputs a 16x16 int32 matrix
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: avx512vnni
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory import memcmp
from memory.buffer import Buffer
from VNNI import (
    dot_i8_to_i32_AVX2,
    dot_i8_to_i32_saturated_AVX2,
    dot_i8_to_i32_x86,
    dot_i8_to_i32_saturated_x86,
)


fn gemm(
    a: Buffer[16 * 64, DType.uint8],
    b: Buffer[64 * 16, DType.int8],
    c: Buffer[16 * 16, DType.int32],
):
    for i in range(16):
        for k in range(64):
            for j in range(16):
                c[16 * i + j] += (
                    a[64 * i + k].cast[DType.int32]()
                    * b[16 * k + j].cast[DType.int32]()
                )


fn pack_vnni(b: Buffer[16 * 64, DType.int8], b2: Buffer[64 * 16, DType.int8]):
    for l in range(16):
        for j in range(16):
            for p in range(4):
                b2[64 * l + 4 * j + p] = b[64 * l + 16 * p + j]


fn gemm_8_to_32(
    a: Buffer[16 * 64, DType.uint8],
    b: Buffer[64 * 16, DType.int8],
    c: Buffer[16 * 16, DType.int32],
    t: Int,
):
    let bp = Buffer[64 * 16, DType.int8].stack_allocation()
    pack_vnni(b, bp)
    for i in range(16):
        var cv = c.data.offset(16 * i).simd_load[16]()
        for l in range(16):
            let av = a.data.offset(64 * i + 4 * l).bitcast[
                DType.int32
            ]().simd_load[1]()
            let bv = bp.data.offset(64 * l).bitcast[DType.int32]().simd_load[
                16
            ]()
            if t == 0:
                cv = dot_i8_to_i32_saturated_x86[16](cv, av, bv)
            elif t == 1:
                cv = dot_i8_to_i32_saturated_AVX2[16](cv, av, bv)
            elif t == 2:
                cv = dot_i8_to_i32_x86[16](cv, av, bv)
            else:
                cv = dot_i8_to_i32_AVX2[16](cv, av, bv)

        c.data.offset(16 * i).simd_store[16](cv)


fn main():
    let a = Buffer[16 * 64, DType.uint8].aligned_stack_allocation[64]()
    let asat = Buffer[16 * 64, DType.uint8].aligned_stack_allocation[64]()
    let b = Buffer[64 * 16, DType.int8].aligned_stack_allocation[64]()

    let c = Buffer[16 * 16, DType.int32].aligned_stack_allocation[64]()
    let csat = Buffer[16 * 16, DType.int32].aligned_stack_allocation[64]()
    let c0 = Buffer[16 * 16, DType.int32].aligned_stack_allocation[64]()
    let c1 = Buffer[16 * 16, DType.int32].aligned_stack_allocation[64]()
    let c2 = Buffer[16 * 16, DType.int32].aligned_stack_allocation[64]()
    let c3 = Buffer[16 * 16, DType.int32].aligned_stack_allocation[64]()

    for i in range(16 * 64):
        a[i] = i & 255
        asat[i] = i & 127
        b[i] = (i & 255) - 128

    # for i in range(64):
    #    print(b[i])

    for i in range(16 * 16):
        c[i] = i
        csat[i] = c[i]
        c0[i] = c[i]
        c1[i] = c[i]
        c2[i] = c[i]
        c3[i] = c[i]

    let av16u = a.data.offset(128 + 64).bitcast[DType.int32]().simd_load[16]()
    let av16s = asat.data.offset(128 + 64).bitcast[DType.int32]().simd_load[
        16
    ]()
    let bv16 = b.data.offset(0).bitcast[DType.int32]().simd_load[16]()
    let cv16u = dot_i8_to_i32_AVX2[16](c.data.simd_load[16](), av16u, bv16)
    let cv16s = dot_i8_to_i32_saturated_AVX2[16](
        c.data.simd_load[16](), av16s, bv16
    )

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

    gemm(a, b, c)
    gemm(asat, b, csat)
    gemm_8_to_32(asat, b, c0, 0)
    gemm_8_to_32(asat, b, c1, 1)
    gemm_8_to_32(a, b, c2, 2)
    gemm_8_to_32(a, b, c3, 3)

    var errors: Int = 0
    errors += memcmp(csat.data, c0.data, 16 * 16)
    errors += memcmp(csat.data, c1.data, 16 * 16)
    errors += memcmp(c.data, c2.data, 16 * 16)
    errors += memcmp(c.data, c3.data, 16 * 16)

    # CHECK: 0
    print(errors)
    if errors != 0:
        print("\nMatrices don't agree!")
