# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate %s -debug-level=full -S -o - -O0 | FileCheck %s

# COM: Ensure loops can be unrolled at -O0.
# COM: https://github.com/modularml/modular/issues/19008


@no_inline
fn use(i: Int):
    pass


fn main():
    @unroll
    # CHECK-COUNT-3: kgen.call @"$unroll-guarantee::use
    for i in range(3):
        use(i)
