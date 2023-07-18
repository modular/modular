# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: kgen %s %mojo_cpu_build_arch -emit -debug-level=full --O0 -o /dev/null

from Range import range
from IO import print


fn main():
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    for x in range(0, 3, 1):
        print(x)

    # CHECK: 9
    # CHECK: 6
    # CHECK: 3
    for y in range(9, 0, -3):
        print(y)

    # CHECK: 42
    for z in range(0, 0, -3):
        print(z)
    else:
        print(42)
