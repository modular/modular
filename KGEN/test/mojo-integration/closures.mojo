# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s
from sys import argv


from runtime.llcl import run


@no_inline
fn takeClosure(owned formatter: Coroutine[Int]) -> Int:
    return run(formatter^)


@no_inline
fn makeClosure(x: Int) -> Coroutine[Int, __lifetime_of()]:
    var z = x * x

    @__copy_capture(z)
    @parameter
    async fn formatter() -> Int:
        return z

    return formatter()


fn main():
    try:
        var x = atol(String(argv()[1]))
        var y = atol(String(argv()[2]))

        var formatter = makeClosure(x)
        var w = takeClosure(formatter^)
        # CHECK: 4
        print(w)
    except e:
        print(e._message())
