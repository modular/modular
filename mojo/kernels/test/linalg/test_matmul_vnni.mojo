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
# REQUIRES: avx2
# RUN: %mojo -debug-level full %s | FileCheck %s

from Range import range

from Buffer import Buffer
from VNNI import dot_i8_to_i32_AVX2, dot_i8_to_i32_x86
from Memory import memcmp


fn gemm(
    a: Buffer[16 * 64, DType.int8],
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


fn dot_i8_to_i32_emulate(
    src: SIMD[DType.int32, 16],
    a: SIMD[DType.int32, 16],
    b: SIMD[DType.int32, 16],
) -> SIMD[DType.int32, 16]:
    var c: SIMD[DType.int32, 16] = 0
    for i in range(16):
        var ai = a[i]
        var bi = b[i]
        var sum: Int32 = 0
        for j in range(4):
            let ab = ai & 0xFF
            ai >>= 8
            let bb = bi & 0xFF
            bi >>= 8
            sum += ab * bb
        sum += src[i]
        c[i] = sum
    return c


fn pack_vnni(b: Buffer[16 * 64, DType.int8], b2: Buffer[64 * 16, DType.int8]):
    for l in range(16):
        for j in range(16):
            for p in range(4):
                b2[64 * l + 4 * j + p] = b[64 * l + 16 * p + j]


fn gemm_8_to_32(
    a: Buffer[16 * 64, DType.int8],
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
                cv = dot_i8_to_i32_x86[16](cv, av, bv)
            elif t == 1:
                cv = dot_i8_to_i32_AVX2[16](cv, av, bv)
            else:
                cv = dot_i8_to_i32_emulate(cv, av, bv)

        c.data.offset(16 * i).simd_store[16](cv)


fn main():
    let a = Buffer[16 * 64, DType.int8].stack_allocation()
    let b = Buffer[64 * 16, DType.int8].stack_allocation()
    let c = Buffer[16 * 16, DType.int32].stack_allocation()
    let c0 = Buffer[16 * 16, DType.int32].stack_allocation()
    let c1 = Buffer[16 * 16, DType.int32].stack_allocation()
    let c2 = Buffer[16 * 16, DType.int32].stack_allocation()

    for i in range(16 * 64):
        a[i] = i & 127
        b[i] = (16 * 64 - i - 1) & 127

    for i in range(16 * 16):
        c[i] = i
        c0[i] = c[i]
        c1[i] = c[i]
        c2[i] = c[i]

    gemm(a, b, c)
    gemm_8_to_32(a, b, c0, 0)
    gemm_8_to_32(a, b, c1, 1)
    gemm_8_to_32(a, b, c2, 2)

    var errors: Int = 0
    errors += memcmp(c.data, c0.data, 16 * 16)
    errors += memcmp(c.data, c1.data, 16 * 16)
    errors += memcmp(c.data, c2.data, 16 * 16)
    # CHECK: 0
    print(errors)
    if errors != 0:
        print("\nMatrices don't agree!")
