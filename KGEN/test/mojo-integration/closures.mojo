# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s
from sys import argv


@no_inline
fn takeClosure(formatter: Coroutine[Int]) -> Int:
    return formatter()


@no_inline
fn makeClosure(x: Int) -> Coroutine[Int]:
    var z = x * x

    @__copy_capture(z)
    @parameter
    async fn formatter() -> Int:
        return z

    return formatter()


fn main():
    try:
        let x = atol(String(argv()[1]))
        let y = atol(String(argv()[2]))

        let formatter = makeClosure(x)
        let w = takeClosure(formatter)
        # CHECK: 4
        print(w)
    except e:
        print(e._message())
