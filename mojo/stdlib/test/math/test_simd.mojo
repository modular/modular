# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota, pow
from sys.info import has_neon, simdwidthof

from testing import *


# CHECK-LABEL: test_simd
def test_simd():
    print("== test_simd")

    # CHECK: True
    print(SIMD[DType.index]().__len__() == simdwidthof[DType.index]())

    # CHECK: 4
    print(SIMD[DType.index, 4]().__len__())

    # CHECK: [0, 0, 0, 0]
    print(SIMD[DType.index, 4]())

    # CHECK: [1, 1, 1, 1]
    print(SIMD[DType.index, 4](1))

    # CHECK: [1.{{0+}}, 1.{{0+}}]
    print(SIMD[DType.float16, 2](True))

    var simd_val = iota[DType.index, 4]()

    # CHECK: [0, 1, 2, 3]
    print(simd_val)

    # CHECK: [1, 2, 3, 4]
    print(simd_val + 1)

    # CHECK: [0, 2, 4, 6]
    print(simd_val * 2)

    # CHECK: [1, 2, 3, 4]
    print(1 + simd_val)

    # CHECK: [0, 2, 4, 6]
    print(2 * simd_val)

    # CHECK: [0, 4, 8, 12]
    print(simd_val << 2)

    # CHECK: [0, 1, 4, 9]
    print(simd_val**2)

    # CHECK: [0, 1, 8, 27]
    print(simd_val**3)

    # CHECK: 3
    print(simd_val.reduce_max())

    # CHECK: 6
    print(simd_val.reduce_add())

    # Check: True
    print((simd_val > 2).reduce_or())

    # Check: False
    print((simd_val > 3).reduce_or())

    # Check: 3
    print(int(Float32(3.0)))

    # Check: -4
    print(int(Float32(-3.5)))

    # CHECK: [16, 20]
    print(iota[DType.index, 8](1).reduce_add[2]())

    # CHECK: [105, 384]
    print(iota[DType.index, 8](1).reduce_mul[2]())

    # CHECK: [1, 2]
    print(iota[DType.index, 8](1).reduce_min[2]())

    # CHECK: [7, 8]
    print(iota[DType.index, 8](1).reduce_max[2]())

    # CHECK: [1, 2, 3, 4]
    print(iota[DType.index, 4](1).reduce_max[4]())

    assert_equal(
        SIMD[DType.bool, 4](False, True, False, True)
        * SIMD[DType.bool, 4](False, True, True, False),
        SIMD[DType.bool, 4](False, True, False, True)
        & SIMD[DType.bool, 4](False, True, True, False),
    )

    assert_equal(int(Float64(0.25)), 0)
    assert_equal(int(Float64(-0.25)), 0)
    assert_equal(int(Float64(1.25)), 1)
    assert_equal(int(Float64(-1.25)), -1)
    assert_equal(int(Float64(-390.8)), -390)


# CHECK-LABEL: test_iota
fn test_iota():
    print("== test_iota")

    # CHECK: [0, 1, 2, 3]
    print(iota[DType.index, 4]())

    # CHECK: 0
    print(iota[DType.index, 1]())


# CHECK-LABEL: test_slice
fn test_slice():
    print("== test_slice")

    var val = iota[DType.index, 4]()

    # CHECK: [0, 1]
    print(val.slice[2]())

    # CHECK: [2, 3]
    print(val.slice[2, offset=2]())

    var s2 = iota[DType.int32, 2](0)

    # CHECK: 0
    print(s2.slice[1]())


# CHECK-LABEL: test_pow
fn test_pow():
    print("== test_pow")

    alias simd_width = 4

    var simd_val = iota[DType.float32, simd_width]()

    # CHECK: [0.0, 1.0, 4.0, 9.0]
    print(pow[DType.float32, DType.float32, simd_width](simd_val, 2.0))

    # CHECK: [inf, 1.0, 0.5, 0.3333333432674408]
    print(pow(simd_val, -1))

    # CHECK: [0.0, 1.0, 1.41421{{[0-9]+}}, 1.73205{{[0-9]+}}]
    print(pow[DType.float32, DType.float32, simd_width](simd_val, 0.5))

    # CHECK: [0.70710{{[0-9]+}}, 0.57735{{[0-9]+}}, 0.5, 0.44721{{[0-9]+}}]
    print(pow[DType.float32, DType.float32, simd_width](simd_val + 2, -0.5))

    # CHECK: [0.0, 1.0, 4.0, 9.0]
    print(pow(simd_val, SIMD[DType.int32, simd_width](2)))

    # CHECK: [0.0, 1.0, 8.0, 27.0]
    print(pow(simd_val, SIMD[DType.int32, simd_width](3)))

    var simd_val_int = iota[DType.int32, simd_width]()

    # CHECK: [0, 1, 4, 9]
    print(pow(simd_val_int, 2))


# CHECK-LABEL: test_simd_bool
fn test_simd_bool():
    print("== test_simd_bool")

    var v0 = iota[DType.index, 4]()

    # CHECK: [False, True, False, False]
    print((v0 > 0) & (v0 < 2))

    # CHECK: [True, False, False, True]
    print((v0 > 2) | (v0 < 1))


# CHECK-LABEL: test_join
fn test_join():
    print("== test_join")

    # CHECK: [3, 4]
    print(Int32(3).join(Int32(4)))

    var s0 = SIMD[DType.index, 4](0, 1, 2, 3)
    var s1 = SIMD[DType.index, 4](5, 6, 7, 8)

    # CHECK: [0, 1, 2, 3, 5, 6, 7, 8]
    print(s0.join(s1))

    var s2 = SIMD[DType.index, 2](5, 6)
    var s3 = SIMD[DType.index, 2](9, 10)

    # CHECK: [5, 6, 9, 10]
    print(s2.join(s3))

    var s4 = iota[DType.index, 32](1)
    var s5 = iota[DType.index, 32](33)
    # CHECK: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    # CHECK: 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    # CHECK: 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    # CHECK: 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    # CHECK: 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    # CHECK: 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    print(s4.join(s5))


def issue_1625():
    print("== issue_1625")
    var size = 16
    alias simd_width = 8
    var ptr = DTypePointer[DType.int64].alloc(size)
    for i in range(size):
        ptr[i] = i

    var x = ptr.load[width = 2 * simd_width](0)
    var evens_and_odds = x.deinterleave()

    assert_equal(
        evens_and_odds[0], SIMD[DType.int64, 8](0, 2, 4, 6, 8, 10, 12, 14)
    )
    assert_equal(
        evens_and_odds[1], SIMD[DType.int64, 8](1, 3, 5, 7, 9, 11, 13, 15)
    )
    ptr.free()


def main():
    test_simd()
    test_iota()
    test_slice()
    test_pow()
    test_simd_bool()
    test_join()
    issue_1625()
