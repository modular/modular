# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate %s -debug-level=full -debug-info-language=Mojo -S -o - -O0 | FileCheck %s

# COM: Ensure loops can be unrolled at -O0.
# COM: https://github.com/modularml/modular/issues/19008


@no_inline
fn three_times(i: Int):
    pass


fn main():
    @unroll
    # CHECK-COUNT-3: kgen.call @{{.*}}three_times
    for i in range(3):
        three_times(i)
