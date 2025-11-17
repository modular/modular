# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


fn main():
    comptime a, (b, c) = 1, (2, 3)
    # CHECK:      1
    # CHECK-NEXT: 2
    # CHECK-NEXT: 3
    print(a)
    print(b)
    print(c)
