# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Meant to be run on an AVX512 system

from memory.unsafe import DTypePointer
from List import Dim, DimList
from memory.buffer import Buffer, NDBuffer
from sys.intrinsics import PrefetchOptions

alias mr = 6
alias nr = 64

alias simd_size = 16


fn kernel6x4(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    n: Int,
    k: Int,
    kc: Int,
):
    let a = Buffer[Dim(), DType.float32](a_ptr, mr * k)
    let b = Buffer[Dim(), DType.float32](b_ptr, kc * nr)
    let c = Buffer[Dim(), DType.float32](c_ptr, mr * n)

    var cv0 = c.simd_load[simd_size](n * 0 + simd_size * 0)
    var cv1 = c.simd_load[simd_size](n * 0 + simd_size * 1)
    var cv2 = c.simd_load[simd_size](n * 0 + simd_size * 2)
    var cv3 = c.simd_load[simd_size](n * 0 + simd_size * 3)
    var cv4 = c.simd_load[simd_size](n * 1 + simd_size * 0)
    var cv5 = c.simd_load[simd_size](n * 1 + simd_size * 1)
    var cv6 = c.simd_load[simd_size](n * 1 + simd_size * 2)
    var cv7 = c.simd_load[simd_size](n * 1 + simd_size * 3)
    var cv8 = c.simd_load[simd_size](n * 2 + simd_size * 0)
    var cv9 = c.simd_load[simd_size](n * 2 + simd_size * 1)
    var cv10 = c.simd_load[simd_size](n * 2 + simd_size * 2)
    var cv11 = c.simd_load[simd_size](n * 2 + simd_size * 3)
    var cv12 = c.simd_load[simd_size](n * 3 + simd_size * 0)
    var cv13 = c.simd_load[simd_size](n * 3 + simd_size * 1)
    var cv14 = c.simd_load[simd_size](n * 3 + simd_size * 2)
    var cv15 = c.simd_load[simd_size](n * 3 + simd_size * 3)
    var cv16 = c.simd_load[simd_size](n * 4 + simd_size * 0)
    var cv17 = c.simd_load[simd_size](n * 4 + simd_size * 1)
    var cv18 = c.simd_load[simd_size](n * 4 + simd_size * 2)
    var cv19 = c.simd_load[simd_size](n * 4 + simd_size * 3)
    var cv20 = c.simd_load[simd_size](n * 5 + simd_size * 0)
    var cv21 = c.simd_load[simd_size](n * 5 + simd_size * 1)
    var cv22 = c.simd_load[simd_size](n * 5 + simd_size * 2)
    var cv23 = c.simd_load[simd_size](n * 5 + simd_size * 3)

    for pr in range(kc):
        let bv0 = b.simd_load[simd_size](4 * simd_size * pr + simd_size * 0)
        let bv1 = b.simd_load[simd_size](4 * simd_size * pr + simd_size * 1)
        let bv2 = b.simd_load[simd_size](4 * simd_size * pr + simd_size * 2)
        let bv3 = b.simd_load[simd_size](4 * simd_size * pr + simd_size * 3)
        b_ptr.offset(4 * simd_size * pr + simd_size * 16).prefetch[
            PrefetchOptions().for_read().high_locality().to_data_cache()
        ]()
        b_ptr.offset(4 * simd_size * pr + simd_size * 17).prefetch[
            PrefetchOptions().for_read().high_locality().to_data_cache()
        ]()
        b_ptr.offset(4 * simd_size * pr + simd_size * 18).prefetch[
            PrefetchOptions().for_read().high_locality().to_data_cache()
        ]()
        b_ptr.offset(4 * simd_size * pr + simd_size * 19).prefetch[
            PrefetchOptions().for_read().high_locality().to_data_cache()
        ]()

        var av = a_ptr.offset(0 * k + pr).simd_load[1]().cast[DType.float32]()
        cv0 += av * bv0
        cv1 += av * bv1
        cv2 += av * bv2
        cv3 += av * bv3

        av = a_ptr.offset(1 * k + pr).simd_load[1]().cast[DType.float32]()
        cv4 += av * bv0
        cv5 += av * bv1
        cv6 += av * bv2
        cv7 += av * bv3

        av = a_ptr.offset(2 * k + pr).simd_load[1]().cast[DType.float32]()
        cv8 += av * bv0
        cv9 += av * bv1
        cv10 += av * bv2
        cv11 += av * bv3

        av = a_ptr.offset(3 * k + pr).simd_load[1]().cast[DType.float32]()
        cv12 += av * bv0
        cv13 += av * bv1
        cv14 += av * bv2
        cv15 += av * bv3

        av = a_ptr.offset(4 * k + pr).simd_load[1]().cast[DType.float32]()
        cv16 += av * bv0
        cv17 += av * bv1
        cv18 += av * bv2
        cv19 += av * bv3

        av = a_ptr.offset(5 * k + pr).simd_load[1]().cast[DType.float32]()
        cv20 += av * bv0
        cv21 += av * bv1
        cv22 += av * bv2
        cv23 += av * bv3

    c.simd_store[simd_size](n * 0 + simd_size * 0, cv0)
    c.simd_store[simd_size](n * 0 + simd_size * 1, cv1)
    c.simd_store[simd_size](n * 0 + simd_size * 2, cv2)
    c.simd_store[simd_size](n * 0 + simd_size * 3, cv3)
    c.simd_store[simd_size](n * 1 + simd_size * 0, cv4)
    c.simd_store[simd_size](n * 1 + simd_size * 1, cv5)
    c.simd_store[simd_size](n * 1 + simd_size * 2, cv6)
    c.simd_store[simd_size](n * 1 + simd_size * 3, cv7)
    c.simd_store[simd_size](n * 2 + simd_size * 0, cv8)
    c.simd_store[simd_size](n * 2 + simd_size * 1, cv9)
    c.simd_store[simd_size](n * 2 + simd_size * 2, cv10)
    c.simd_store[simd_size](n * 2 + simd_size * 3, cv11)
    c.simd_store[simd_size](n * 3 + simd_size * 0, cv12)
    c.simd_store[simd_size](n * 3 + simd_size * 1, cv13)
    c.simd_store[simd_size](n * 3 + simd_size * 2, cv14)
    c.simd_store[simd_size](n * 3 + simd_size * 3, cv15)
    c.simd_store[simd_size](n * 4 + simd_size * 0, cv16)
    c.simd_store[simd_size](n * 4 + simd_size * 1, cv17)
    c.simd_store[simd_size](n * 4 + simd_size * 2, cv18)
    c.simd_store[simd_size](n * 4 + simd_size * 3, cv19)
    c.simd_store[simd_size](n * 5 + simd_size * 0, cv20)
    c.simd_store[simd_size](n * 5 + simd_size * 1, cv21)
    c.simd_store[simd_size](n * 5 + simd_size * 2, cv22)
    c.simd_store[simd_size](n * 5 + simd_size * 3, cv23)


fn kernel6x4_naive(
    a_ptr: DTypePointer[DType.float32],
    b_ptr: DTypePointer[DType.float32],
    c_ptr: DTypePointer[DType.float32],
    n: Int,
    k: Int,
    kc: Int,
):

    let a = Buffer[Dim(), DType.float32](a_ptr, mr * k)
    let b = Buffer[Dim(), DType.float32](b_ptr, kc * nr)
    let c = Buffer[Dim(), DType.float32](c_ptr, mr * n)

    for ir in range(mr):
        for jr in range(nr):
            for p in range(kc):
                c[ir * n + jr] += a[ir * k + p] * b[p * nr + jr]
