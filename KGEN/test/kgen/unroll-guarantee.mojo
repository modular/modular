# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate %s -S -o - -O0 | FileCheck %s

# COM: Ensure loops can be unrolled at -O0.

from Range import range


fn use(i: Int):
    pass


fn main():
    @unroll
    # CHECK-COUNT-3: kgen.call @"$unroll-guarantee::use
    for i in range(3):
        use(i)
