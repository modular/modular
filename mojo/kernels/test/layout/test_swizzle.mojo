# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.layout import Layout
from layout.swizzle import Swizzle


# CHECK-LABEL: test_swizzle_basic
fn test_swizzle_basic():
    print("== test_swizzle_basic")

    alias thread_layout = Layout.row_major(8, 8)

    # swizzle every 16 threads by the least significant bit.
    var swizzle_bits1_per16 = Swizzle(1, 0, 4)

    # CHECK: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    # CHECK: 17 16 19 18 21 20 23 22 25 24 27 26 29 28 31 30
    # CHECK: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    # CHECK: 49 48 51 50 53 52 55 54 57 56 59 58 61 60 63 62
    for tid in range(thread_layout.size()):
        print(swizzle_bits1_per16(tid), end=" ")
        if (tid + 1) % 16 == 0:
            print()

    # swizzle every 8 threads by 2 least significant bits.
    var swizzle_bits2_per8 = Swizzle(2, 0, 3)

    # CHECK: 0 1 2 3 4 5 6 7
    # CHECK: 9 8 11 10 13 12 15 14
    # CHECK: 18 19 16 17 22 23 20 21
    # CHECK: 27 26 25 24 31 30 29 28
    # CHECK: 32 33 34 35 36 37 38 39
    # CHECK: 41 40 43 42 45 44 47 46
    # CHECK: 50 51 48 49 54 55 52 53
    # CHECK: 59 58 57 56 63 62 61 60
    for tid in range(thread_layout.size()):
        print(swizzle_bits2_per8(tid), end=" ")
        if (tid + 1) % 8 == 0:
            print()

    # swizzle every 16 threads the 2nd and 3rd least significant bits.
    var swizzle_bits2_base1_per8 = Swizzle(2, 1, 3)

    # CHECK: 0 1 2 3 4 5 6 7
    # CHECK: 8 9 10 11 12 13 14 15
    # CHECK: 18 19 16 17 22 23 20 21
    # CHECK: 26 27 24 25 30 31 28 29
    # CHECK: 36 37 38 39 32 33 34 35
    # CHECK: 44 45 46 47 40 41 42 43
    # CHECK: 54 55 52 53 50 51 48 49
    # CHECK: 62 63 60 61 58 59 56 57
    for tid in range(thread_layout.size()):
        # Verify the operator overloaded for different index types.
        var tid_u32 = UInt32(tid)
        print(swizzle_bits2_base1_per8(tid_u32), end=" ")
        if (tid + 1) % 8 == 0:
            print()


fn main():
    test_swizzle_basic()
