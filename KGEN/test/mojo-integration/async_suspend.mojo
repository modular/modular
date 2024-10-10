# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s
# RUN: %mojo --no-optimization %s | FileCheck %s

from runtime.asyncrt import run


@no_inline
async fn print_it(x: Int) -> Int:
    print(x)
    return 1


@no_inline
async fn does_multiple_things(init: Int):
    var a = init

    a += await print_it(a)
    a += await print_it(a)
    a += await print_it(a)
    print(a)


fn main():
    # CHECK: 1
    # CHECK-NEXT: 2
    # CHECK-NEXT: 3
    # CHECK-NEXT: 4
    run(does_multiple_things(1))
